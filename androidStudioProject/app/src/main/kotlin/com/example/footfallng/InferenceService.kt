package com.example.footfallng

import android.content.Context
import android.media.AudioAttributes
import android.media.AudioManager
import android.media.SoundPool
import android.media.ToneGenerator
import android.os.Handler
import android.os.HandlerThread
import android.os.Process
import android.util.Log
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONObject
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.Tensor
import java.io.File
import java.util.concurrent.ConcurrentLinkedDeque
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import kotlin.math.exp
import kotlin.math.sqrt

class InferenceService(private val context: Context) {
    companion object {
        private const val TAG = "InferenceService"
        private const val DEFAULT_WINDOW_SIZE = 100  // fallback if metadata missing
        private const val DEFAULT_THRESHOLD = 0.7f
    }

    // Data window for inference
    var onFootfallDetected: ((Long) -> Unit)? = null

    private data class Sample(val x: Float, val y: Float, val z: Float, val dt: Float)
    private val dataWindow = ConcurrentLinkedDeque<Sample>()
    private var lastTimestamp = 0L

    // PyTorch model & metadata
    private lateinit var module: Module
    private lateinit var scalerMean: FloatArray
    private lateinit var scalerScale: FloatArray
    private var windowSize: Int = DEFAULT_WINDOW_SIZE
    private var threshold: Float = DEFAULT_THRESHOLD
    private val modelInitialized = AtomicBoolean(false)
    var sampleThreshold: Float = 0.85f     // per-sample probability threshold

    /**
     * Global‐window gate (probability threshold)
     */
    var windowThreshold: Float = 0.45f

    /**
     * Minimum interval between consecutive detections (refractory period)
     */
    var minIntervalMs: Long = 500L

    /**
     * Number of samples to skip between consecutive inferences
     */
    var inferenceStride: Int = 10

    /**
     * Sliding window length (M) used for hit smoothing
     */
    var hitWindow: Int = 5

    /**
     * Required number of hits (N) inside the sliding window before confirming a footfall
     */
    var requiredHits: Int = 3
    private val recentHits: java.util.ArrayDeque<Boolean> = java.util.ArrayDeque()
    private var lastDetectionTime = 0L

    // Sound & state
    private val toneGenerator = ToneGenerator(AudioManager.STREAM_MUSIC, 100)
    private lateinit var soundPool: SoundPool
    private var beepId: Int = 0
    private val isInferenceActive = AtomicBoolean(false)
    private val detectionCount = AtomicInteger(0)
    private val _detectionCountFlow = kotlinx.coroutines.flow.MutableStateFlow(0)
    val detectionCountFlow: kotlinx.coroutines.flow.StateFlow<Int> = _detectionCountFlow
    private var samplesSinceLast = 0

    // Dedicated inference thread
    private val inferenceThread = HandlerThread("InferenceThread").also { it.start() }
    private val inferenceHandler = Handler(inferenceThread.looper)
    private val inferenceQueued = AtomicBoolean(false)

    // Pre-allocated buffers
    private lateinit var floatBuffer: FloatArray
    private lateinit var tensorShape: LongArray
    private var lastScannedIndex = 0

    /**
     * Copy asset to internal storage and return its absolute path.
     */
    private fun assetFilePath(ctx: Context, name: String): String {
        val file = File(ctx.filesDir, name)
        if (!file.exists()) {
            ctx.assets.open(name).use { input ->
                file.outputStream().use { output ->
                    input.copyTo(output)
                }
            }
        }
        return file.absolutePath
    }

    /**
     * Initialize PyTorch module and metadata.
     */
    fun initialize(): Boolean {
        return try {
            // 1) Load TorchScript model
            val modelPath = assetFilePath(context, "refined_v2_ts.pt")
            module = Module.load(modelPath)

            // 2) Load metadata JSON
            val metaJson = context.assets.open("metadata_refined_v2.json")
                .bufferedReader().use { it.readText() }
            val meta = JSONObject(metaJson)

            // dynamic window size & threshold
            windowSize = meta.optInt("window", DEFAULT_WINDOW_SIZE)
            threshold  = meta.optDouble("threshold", DEFAULT_THRESHOLD.toDouble()).toFloat()

            // Override with smaller window for lower latency
            windowSize = 100  // only half a second @200Hz
            threshold = 0.85f   // Higher threshold to prevent false detections

            // scaler params
            val scaler = meta.getJSONObject("scaler")
            val meanArr = scaler.getJSONArray("mean")
            val scaleArr = scaler.getJSONArray("scale")
            scalerMean = FloatArray(meanArr.length()) { i ->
                meanArr.getDouble(i).toFloat()
            }
            scalerScale = FloatArray(scaleArr.length()) { i ->
                scaleArr.getDouble(i).toFloat()
            }

            // Pre-allocate model input buffer
            floatBuffer = FloatArray(windowSize * 5)
            tensorShape = longArrayOf(1, windowSize.toLong(), 5)

            // Pre-load beep into SoundPool for minimal latency
            val attrs = AudioAttributes.Builder()
                .setUsage(AudioAttributes.USAGE_ASSISTANCE_SONIFICATION)
                .setContentType(AudioAttributes.CONTENT_TYPE_SONIFICATION)
                .build()
            soundPool = SoundPool.Builder()
                .setMaxStreams(1)
                .setAudioAttributes(attrs)
                .build()

            // Load "ding" from raw resources if present; else keep beepId = 0
            val resId = context.resources.getIdentifier("ding", "raw", context.packageName)
            if (resId != 0) {
                beepId = soundPool.load(context, resId, 1)
            }

            // Set high priority for inference thread
            Process.setThreadPriority(
                inferenceThread.threadId,
                Process.THREAD_PRIORITY_URGENT_AUDIO
            )

            modelInitialized.set(true)
            Log.d(TAG, "Model and metadata loaded (window=$windowSize, thr=$threshold)")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Initialization failed", e)
            false
        }
    }

    /** Start inference. */
    fun startInference(): Boolean {
        if (!modelInitialized.get()) {
            Log.e(TAG, "Cannot start inference: model not initialized")
            return false
        }
        isInferenceActive.set(true)
        detectionCount.set(0)
        _detectionCountFlow.value = 0
        dataWindow.clear()
        lastTimestamp = 0L
        lastDetectionTime = 0L
        Log.d(TAG, "Inference service started")
        return true
    }

    /** Stop inference. */
    fun stopInference() {
        isInferenceActive.set(false)
        Log.d(TAG, "Inference service stopped")
    }

    /**
     * Feed new accelerometer sample (x,y,z,timestamp in ms).
     */
    fun processAccelerometerData(
        x: Float, y: Float, z: Float,
        timestamp: Long = System.currentTimeMillis()
    ): Boolean {
        if (!isInferenceActive.get()) return false

        // compute dt
        val dt = if (lastTimestamp == 0L) 0f
        else (timestamp - lastTimestamp) / 1000f
        lastTimestamp = timestamp

        // add sample
        dataWindow.addLast(Sample(x, y, z, dt))
        if (dataWindow.size > windowSize) dataWindow.removeFirst()

        // Count samples since last inference
        samplesSinceLast++

        // run inference on dedicated thread when buffer full
        if (dataWindow.size >= windowSize &&
            samplesSinceLast >= inferenceStride &&  // Run every ~50ms (at 200Hz) 
            isInferenceActive.get() &&
            inferenceQueued.compareAndSet(false, true)
        ) {
            samplesSinceLast = 0
            inferenceHandler.post {
                try {
                    runModelInference()
                } finally {
                    inferenceQueued.set(false)
                }
            }
        }
        return true
    }

    /**
     * Run the PyTorch model on the current window.
     */
    private fun runModelInference() {
        if (!isInferenceActive.get()) return

        try {
            // build flat feature buffer: [1, windowSize, 5]
            var idx = 0
            for (s in dataWindow) {
                floatBuffer[idx++] = s.x
                floatBuffer[idx++] = s.y
                floatBuffer[idx++] = s.z
                floatBuffer[idx++] = sqrt(s.x*s.x + s.y*s.y + s.z*s.z)
                floatBuffer[idx++] = s.dt
            }
            // apply scaler
            for (i in floatBuffer.indices) {
                val fi = i % 5
                floatBuffer[i] = (floatBuffer[i] - scalerMean[fi]) / scalerScale[fi]
            }

            // run model
            val inputTensor = Tensor.fromBlob(floatBuffer, tensorShape)
            val outputs = module.forward(IValue.from(inputTensor)).toTuple()
            val seqLogits = outputs[0].toTensor().dataAsFloatArray
            val globalLogit = outputs[1].toTensor().dataAsFloatArray[0]  // Global prediction

            if (seqLogits.isNotEmpty()) {
                val lastIndex = seqLogits.size - 1
                val lastLogit = seqLogits[lastIndex]
                val p = 1f / (1f + exp(-lastLogit))
                val globalP = 1f / (1f + exp(-globalLogit))

                Log.d(
                    TAG,
                    "Probabilities: last=$p (sampleThr=$sampleThreshold), global=$globalP (winThr=$windowThreshold)"
                )

                // Only detect if both the last timestamp AND the model are confident
                // This prevents false positives from noise
                val isHit = (p > sampleThreshold && globalP > windowThreshold)

                // sliding window smoothing
                recentHits.addLast(isHit)
                if (recentHits.size > hitWindow) recentHits.removeFirst()

                if (recentHits.count { it } >= requiredHits) {
                    recentHits.clear()
                    handleFootfallDetection()
                }
            } else {
                Log.w(TAG, "Empty seqLogits array returned from model")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error during model inference", e)
        }
    }

    /** Handle a detected footfall event. */
    private fun handleFootfallDetection() {
        val now = System.currentTimeMillis()
        if (now - lastDetectionTime < minIntervalMs) return
        lastDetectionTime = now

        val count = detectionCount.incrementAndGet()
        _detectionCountFlow.value = count
        // Low-latency beep
        if (::soundPool.isInitialized && beepId != 0) {
            soundPool.play(beepId, 1f, 1f, 0, 0, 1f)
        } else {
            toneGenerator.startTone(ToneGenerator.TONE_PROP_BEEP, 50)
        }
        onFootfallDetected?.invoke(now)
        Log.d(TAG, "Footfall detected (#$count)")
    }

    /** Release resources. */
    fun cleanup() {
        toneGenerator.release()
        if (::soundPool.isInitialized) soundPool.release()
        inferenceThread.quitSafely()
        isInferenceActive.set(false)
    }
}