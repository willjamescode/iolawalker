package com.example.footfallng

import android.bluetooth.BluetoothDevice
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.os.Bundle
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.SeekBar
import android.widget.TextView
import androidx.fragment.app.Fragment
import com.google.android.material.switchmaterial.SwitchMaterial
import bolts.Task
import com.mbientlab.metawear.MetaWearBoard
import com.mbientlab.metawear.android.BtleService
import com.mbientlab.metawear.module.Accelerometer
import com.example.footfallng.data.SensorDataManager
import com.mbientlab.metawear.Route
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch
import kotlinx.coroutines.delay
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeout
import kotlinx.coroutines.withTimeoutOrNull
import kotlinx.coroutines.cancelChildren
import java.text.SimpleDateFormat
import java.util.concurrent.atomic.AtomicInteger
import java.util.ArrayDeque
import java.util.Date
import java.util.Locale

class DeviceSetupActivityFragment :
    Fragment(),
    ServiceConnection {

    interface FragmentSettings {
        val btDevice: BluetoothDevice
    }

    private lateinit var settings: FragmentSettings
    private var mwBoard: MetaWearBoard? = null

    // UI elements
    private lateinit var deviceAddressText: TextView
    private lateinit var connectionStatusText: TextView
    private lateinit var dataContentText: TextView
    private lateinit var startStreamButton: Button
    private lateinit var stopStreamButton: Button
    private lateinit var viewStoredDataButton: Button
    private lateinit var stepDetectorSwitch: SwitchMaterial
    private lateinit var stepCountText: TextView
    private lateinit var inferenceSwitch: SwitchMaterial
    private lateinit var inferenceStatusText: TextView

    // Tuning knob UI elements
    private lateinit var seekSampleThresh: SeekBar
    private lateinit var tvSampleThresh: TextView
    private lateinit var seekWindowThresh: SeekBar
    private lateinit var tvWindowThresh: TextView
    private lateinit var seekMinInterval: SeekBar
    private lateinit var tvMinInterval: TextView
    private lateinit var seekStride: SeekBar
    private lateinit var tvStride: TextView
    private lateinit var seekHitWindow: SeekBar
    private lateinit var tvHitWindow: TextView
    private lateinit var seekRequiredHits: SeekBar
    private lateinit var tvRequiredHits: TextView

    // Data storage manager
    private lateinit var sensorDataManager: SensorDataManager
    private var currentSessionId: String? = null

    // Streaming state
    private var isStreaming = false
    private var accelerometer: Accelerometer? = null

    // Footfall tracking
    private var footfallCount = 0

    // Coroutine scope for background operations
    private val ioScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private val mainScope = CoroutineScope(SupervisorJob() + Dispatchers.Main)
    private var isStoppingInProgress = false
    private var isStartingInProgress = false
    private var currentStopJob: Job? = null

    // Add safety timeout handler
    private val timeoutHandler = Handler(Looper.getMainLooper())
    private var emergencyStopRunnable: Runnable? = null

    // Track resources that need to be released
    private var routeSubscription: Task<Route>? = null

    // Track streaming metrics
    private val dataPointCounter = AtomicInteger(0)
    private var streamStartTime = 0L

    // Log buffering to prevent memory issues
    private val logBuffer = ArrayDeque<String>(50) // Keep only last 50 entries
    private var uiUpdateScheduled = false

    // Step detection service
    private var stepDetectorService: StepDetectorService? = null

    // Neural network inference service
    private lateinit var inferenceService: InferenceService

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Initialize data manager
        sensorDataManager = SensorDataManager(requireContext())

        // Initialize inference service
        inferenceService = InferenceService(requireContext())

        // Load model & metadata once; disable inference UI on failure
        val inferenceReady = inferenceService.initialize()
        if (!inferenceReady) {
            Log.e(
                "DeviceSetupFragment",
                "InferenceService initialization failed; disabling inference features"
            )
        }

        // Try to get settings from activity
        activity?.let { activity ->
            if (activity is FragmentSettings) {
                settings = activity
            }
        }

        requireContext().applicationContext.bindService(
            Intent(requireContext(), BtleService::class.java),
            this,
            Context.BIND_AUTO_CREATE
        )
    }

    override fun onDestroy() {
        super.onDestroy()

        // Clean up step detector
        stepDetectorService?.cleanup()
        stepDetectorService = null

        // Clean up inference service
        inferenceService.cleanup()

        // Cancel any ongoing operations
        ioScope.coroutineContext.cancelChildren()
        mainScope.coroutineContext.cancelChildren()

        // Remove any pending timeout callbacks
        emergencyStopRunnable?.let { timeoutHandler.removeCallbacks(it) }

        // Make sure we close any open data sessions
        forceDisconnectAndReconnect()
        sensorDataManager.stopSession()

        // Release any route subscriptions
        try {
            routeSubscription?.let { task ->
                if (task.isCompleted && !task.isFaulted && !task.isCancelled) {
                    task.result?.remove()
                }
            }
            routeSubscription = null
        } catch (e: Exception) {
            Log.e("DeviceSetupFragment", "Error cleaning up routes", e)
        }

        requireContext().applicationContext.unbindService(this)
    }

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        saved: Bundle?
    ): View {
        val view = inflater.inflate(R.layout.fragment_device_setup, container, false)

        deviceAddressText = view.findViewById(R.id.device_address)
        connectionStatusText = view.findViewById(R.id.connection_status)
        dataContentText = view.findViewById(R.id.data_content)
        startStreamButton = view.findViewById(R.id.btn_start_stream)
        stopStreamButton = view.findViewById(R.id.btn_stop_stream)
        viewStoredDataButton = view.findViewById(R.id.btn_view_stored_data)
        stepDetectorSwitch = view.findViewById(R.id.step_detector_switch)
        stepCountText = view.findViewById(R.id.step_count_text)
        inferenceSwitch = view.findViewById(R.id.inference_switch)
        inferenceStatusText = view.findViewById(R.id.inference_status_text)

        // Tuning knob bindings
        seekSampleThresh = view.findViewById(R.id.seek_sample_thresh)
        tvSampleThresh = view.findViewById(R.id.tv_sample_thresh)
        seekWindowThresh = view.findViewById(R.id.seek_window_thresh)
        tvWindowThresh = view.findViewById(R.id.tv_window_thresh)
        seekMinInterval = view.findViewById(R.id.seek_min_interval)
        tvMinInterval = view.findViewById(R.id.tv_min_interval)
        seekStride = view.findViewById(R.id.seek_stride)
        tvStride = view.findViewById(R.id.tv_stride)
        seekHitWindow = view.findViewById(R.id.seek_hit_window)
        tvHitWindow = view.findViewById(R.id.tv_hit_window)
        seekRequiredHits = view.findViewById(R.id.seek_required_hits)
        tvRequiredHits = view.findViewById(R.id.tv_required_hits)

        // Helper to bind seekbars
        fun bindKnob(
            seek: SeekBar,
            tv: TextView,
            max: Int,
            initialProgress: Int,
            onProgress: (Int) -> String
        ) {
            seek.max = max
            seek.progress = initialProgress.coerceIn(0, max)
            tv.text = onProgress(seek.progress)
            seek.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
                override fun onProgressChanged(s: SeekBar?, p: Int, fromUser: Boolean) {
                    tv.text = onProgress(p)
                }

                override fun onStartTrackingTouch(s: SeekBar?) {}
                override fun onStopTrackingTouch(s: SeekBar?) {}
            })
        }

        // 1) Per-sample threshold (0.30–0.95)
        bindKnob(
            seekSampleThresh,
            tvSampleThresh,
            95,
            ((inferenceService.sampleThreshold - 0.30f) / 0.65f * 100).toInt(),
        ) { prog ->
            val v = 0.30f + prog / 100f * 0.65f
            inferenceService.sampleThreshold = v
            getString(R.string.format_sample_threshold, v)
        }

        // 2) Global-window threshold (0.10–1.0)
        bindKnob(
            seekWindowThresh,
            tvWindowThresh,
            90,
            ((inferenceService.windowThreshold - 0.10f) / 0.90f * 100).toInt(),
        ) { prog ->
            val v = 0.10f + prog / 100f * 0.90f
            inferenceService.windowThreshold = v
            getString(R.string.format_window_threshold, v)
        }

        // 3) Minimum interval (0‒1000 ms)
        bindKnob(
            seekMinInterval,
            tvMinInterval,
            1000,
            inferenceService.minIntervalMs.toInt().coerceIn(0, 1000),
        ) { prog ->
            val v = prog.toLong()
            inferenceService.minIntervalMs = v
            getString(R.string.format_min_interval, v)
        }

        // 4) Inference stride (1–20 samples)
        bindKnob(
            seekStride,
            tvStride,
            20,
            inferenceService.inferenceStride.coerceIn(1, 20),
        ) { prog ->
            val v = (prog).coerceAtLeast(1)
            inferenceService.inferenceStride = v
            getString(R.string.format_stride, v)
        }

        // 5) Hit window M (1–20)
        bindKnob(
            seekHitWindow,
            tvHitWindow,
            20,
            inferenceService.hitWindow.coerceIn(1, 20),
        ) { prog ->
            val v = (prog).coerceAtLeast(1)
            inferenceService.hitWindow = v
            // ensure requiredHits not > hitWindow
            if (inferenceService.requiredHits > v) {
                inferenceService.requiredHits = v
                seekRequiredHits.progress = v
                tvRequiredHits.text = getString(R.string.format_required_hits, v)
            }
            getString(R.string.format_hit_window, v)
        }

        // 6) Required hits N (1–20)
        bindKnob(
            seekRequiredHits,
            tvRequiredHits,
            20,
            inferenceService.requiredHits.coerceIn(1, 20),
        ) { prog ->
            val v = (prog).coerceAtLeast(1)
            inferenceService.requiredHits = v
            getString(R.string.format_required_hits, v)
        }

        // Set up button click listeners
        startStreamButton.setOnClickListener {
            startDataStreaming()
        }

        stopStreamButton.setOnClickListener {
            Log.d("DeviceSetupFragment", "Stop button clicked")

            // Immediately update UI for user feedback - this must happen instantly
            handleStopButtonPressed()
        }

        viewStoredDataButton.setOnClickListener {
            navigateToSensorDataViewActivity()
        }

        // Initialize step detector service
        stepDetectorService = StepDetectorService(requireContext())

        // Set up step detector switch
        stepDetectorSwitch.setOnCheckedChangeListener { buttonView, isChecked ->
            if (isChecked) {
                startStepDetection()
            } else {
                stopStepDetection()
            }
        }

        // Set up inference switch
        inferenceSwitch.setOnCheckedChangeListener { buttonView, isChecked ->
            if (isChecked) {
                startInference()
            } else {
                stopInference()
            }
        }

        return view
    }

    override fun onServiceConnected(name: ComponentName?, binder: IBinder) {
        if (::settings.isInitialized) {
            mwBoard = (binder as BtleService.LocalBinder).getMetaWearBoard(settings.btDevice)
        }
    }

    override fun onServiceDisconnected(name: ComponentName?) = Unit

    fun reconnected() {
        // Update UI after reconnection
        updateConnectionStatus("Connected")

        // Log successful reconnection
        if (::settings.isInitialized) {
            Log.d("DeviceSetupFragment", "Reconnected to ${settings.btDevice.address}")
            addDataToLog("Reconnected to device")
        }
    }

    private fun addDataToLog(text: String) {
        if (::dataContentText.isInitialized) {
            val timestamp = SimpleDateFormat("HH:mm:ss", Locale.getDefault())
                .format(Date())

            val entry = "[$timestamp] $text"

            // Add to buffer, removing oldest entries if needed
            synchronized(logBuffer) {
                if (logBuffer.size >= 50) {
                    logBuffer.removeLast()
                }
                logBuffer.addFirst(entry)
            }

            // Throttle UI updates to reduce main thread load
            if (!uiUpdateScheduled) {
                uiUpdateScheduled = true
                mainScope.launch {
                    try {
                        delay(200) // Update UI at most 5x per second

                        // Create text from current buffer
                        val displayText = synchronized(logBuffer) {
                            logBuffer.joinToString("\n")
                        }

                        dataContentText.text = displayText
                    } finally {
                        uiUpdateScheduled = false
                    }
                }
            }
        }
    }

    fun setSettings(fragmentSettings: FragmentSettings) {
        settings = fragmentSettings

        // Update UI with device information if views are initialized
        if (::deviceAddressText.isInitialized) {
            deviceAddressText.text = settings.btDevice.address
            updateConnectionStatus("Connected")
        }

        // If mwBoard is already connected, use the new settings
        mwBoard?.let {
            // Disconnect the old board
            it.disconnectAsync()

            // Get the service binder and connect to the new device
            val service = activity?.applicationContext?.getSystemService(Context.BLUETOOTH_SERVICE)
            if (service is BtleService.LocalBinder) {
                mwBoard = service.getMetaWearBoard(settings.btDevice)
            }
        }
    }

    /**
     * Update the connection status display
     */
    private fun updateConnectionStatus(status: String) {
        if (::connectionStatusText.isInitialized) {
            connectionStatusText.text = status
            connectionStatusText.setTextColor(
                if (status == "Connected")
                    resources.getColor(android.R.color.holo_green_dark, null)
                else
                    resources.getColor(android.R.color.holo_red_dark, null)
            )
        }
    }

    /**
     * Start streaming data from the device
     */
    private fun startDataStreaming() {
        if (!::settings.isInitialized || mwBoard == null) {
            addDataToLog("Error: Device not connected")
            return
        }

        // Don't start if already streaming
        if (isStreaming || isStartingInProgress) {
            Log.d(
                "DeviceSetupFragment",
                "Already streaming or starting in progress, ignoring start request"
            )
            return
        }

        try {
            // Set flag to indicate start in progress
            isStartingInProgress = true

            // Reset counters
            dataPointCounter.set(0)
            streamStartTime = System.currentTimeMillis()

            // Immediately update UI for feedback
            startStreamButton.isEnabled = false
            stopStreamButton.isEnabled = false
            addDataToLog("Starting data stream...")

            // Start in background thread
            ioScope.launch {
                try {
                    // Create a new session for data recording
                    currentSessionId = sensorDataManager.startNewSession()

                    withContext(Dispatchers.Main) {
                        addDataToLog("Starting data stream from device ${settings.btDevice.address}")
                        addDataToLog("Recording data to file with session ID: $currentSessionId")
                    }

                    // Clean up any previous accelerometer references
                    accelerometer = null

                    // Get a fresh accelerometer instance
                    accelerometer = mwBoard!!.getModule(Accelerometer::class.java)

                    // Configure the accelerometer - simplify parameters for better performance
                    accelerometer?.configure()
                        ?.odr(200f)      // Increased sample rate to 200Hz
                        ?.range(4f)
                        ?.commit()

                    // Clear any existing route subscription
                    routeSubscription = null

                    // Start streaming accelerometer data - keep reference to subscription
                    routeSubscription = accelerometer?.acceleration()?.addRouteAsync { route ->
                        route.stream { data, env ->
                            val acceleration =
                                data.value(com.mbientlab.metawear.data.Acceleration::class.java)

                            // Save to data manager
                            sensorDataManager.saveSensorReading(
                                x = acceleration.x(),
                                y = acceleration.y(),
                                z = acceleration.z()
                            )

                            // Pass data to inference service if active
                            if (inferenceSwitch.isChecked) {
                                inferenceService.processAccelerometerData(
                                    acceleration.x(),
                                    acceleration.y(),
                                    acceleration.z()
                                )
                            }

                            // Increment data point counter
                            dataPointCounter.incrementAndGet()

                            // Data is being saved to file but not displayed in UI
                            // to improve performance and reduce memory usage
                        }
                    }

                    // Handle the route subscription result
                    routeSubscription?.continueWith { task ->
                        if (task.isFaulted) {
                            mainScope.launch(Dispatchers.Main) {
                                addDataToLog("Error: ${task.error?.message}")
                                isStartingInProgress = false
                                startStreamButton.isEnabled = true
                                stopStreamButton.isEnabled = false
                            }
                            return@continueWith null
                        }

                        // Start the accelerometer
                        accelerometer?.acceleration()?.start()
                        accelerometer?.start()

                        // Update UI state
                        mainScope.launch(Dispatchers.Main) {
                            isStreaming = true
                            isStartingInProgress = false
                            addDataToLog("Accelerometer streaming started and being saved to CSV file")

                            startStreamButton.isEnabled = false
                            stopStreamButton.isEnabled = true
                            Log.d(
                                "DeviceSetupFragment",
                                "Streaming started, buttons updated: Start=false, Stop=true"
                            )

                            // Notify activity that tracking is active
                            (activity as? DeviceSetupActivity)?.setFootfallTrackingState(true)

                            // Reset footfall counter
                            footfallCount = 0
                        }
                        null
                    }
                } catch (e: Exception) {
                    withContext(Dispatchers.Main) {
                        addDataToLog("Error: ${e.message}")
                        isStartingInProgress = false
                        startStreamButton.isEnabled = true
                        stopStreamButton.isEnabled = false
                    }
                }
            }
        } catch (e: Exception) {
            isStartingInProgress = false
            addDataToLog("Error: ${e.message}")
            startStreamButton.isEnabled = true
            stopStreamButton.isEnabled = false
        }
    }

    /**
     * Stop the sensor data streaming
     */
    private fun stopDataStreaming() {
        Log.d("DeviceSetupFragment", "stopDataStreaming called, isStreaming=$isStreaming")

        if (isStoppingInProgress) {
            return
        }

        // Set stopping flag
        isStoppingInProgress = true

        try {
            // Cancel any previous stop operations
            currentStopJob?.cancel()
            emergencyStopRunnable?.let { timeoutHandler.removeCallbacks(it) }

            // Create emergency stop timer - will force-kill operations if they take too long
            emergencyStopRunnable = Runnable {
                Log.e("DeviceSetupFragment", "EMERGENCY STOP triggered - operations took too long")
                forceDisconnectAndReconnect()
            }

            // Schedule emergency stop after 1.5 seconds if normal shutdown doesn't complete
            timeoutHandler.postDelayed(emergencyStopRunnable!!, 1500)

            // Perform stopping operations in background thread
            currentStopJob = ioScope.launch {
                try {
                    Log.d("DeviceSetupFragment", "Background stop operation started")

                    // Stop accelerometer first - this prevents new data from being collected
                    try {
                        accelerometer?.let { accel ->
                            withTimeoutOrNull(300L) { // Faster timeout for accelerometer
                                try {
                                    accel.acceleration()?.stop()
                                    accel.stop()
                                } catch (e: Exception) {
                                    Log.e("DeviceSetupFragment", "Error stopping accelerometer", e)
                                }
                            }
                        }
                    } catch (e: Exception) {
                        Log.e("DeviceSetupFragment", "Failed to stop accelerometer", e)
                    }

                    // Release route subscription after stopping accelerometer
                    try {
                        routeSubscription?.let { task ->
                            if (task.isCompleted && !task.isFaulted && !task.isCancelled) {
                                task.result?.remove()
                            }
                        }
                    } catch (e: Exception) {
                        Log.e("DeviceSetupFragment", "Error unsubscribing from route", e)
                    } finally {
                        routeSubscription = null
                    }

                    // Close data session
                    try {
                        sensorDataManager.stopSession()
                    } catch (e: Exception) {
                        Log.e("DeviceSetupFragment", "Failed to stop data session", e)
                    }

                    // Set accelerometer to null to avoid keeping references
                    accelerometer = null

                    // Cancel emergency timeout as we completed normally
                    mainScope.launch {
                        emergencyStopRunnable?.let { timeoutHandler.removeCallbacks(it) }
                        emergencyStopRunnable = null

                        // Reset stopping flag
                        isStoppingInProgress = false

                        Log.d("DeviceSetupFragment", "Normal stop completed successfully")
                    }

                } catch (e: Exception) {
                    Log.e("DeviceSetupFragment", "Error in background stop operation", e)
                    mainScope.launch {
                        isStoppingInProgress = false
                    }
                }
            }
        } catch (e: Exception) {
            Log.e("DeviceSetupFragment", "Error initiating stop", e)
            isStoppingInProgress = false
        }
    }

    /**
     * Immediately handle stop button press by updating UI state,
     * then trigger background stop operations
     */
    private fun handleStopButtonPressed() {
        // Immediately update UI state whether or not we're currently streaming
        startStreamButton.isEnabled = false
        stopStreamButton.isEnabled = false
        addDataToLog("Stopping data stream...")

        // Force UI to update immediately
        view?.post {
            stopStreamButton.invalidate()
            startStreamButton.invalidate()
        }

        // Update activity state
        (activity as? DeviceSetupActivity)?.setFootfallTrackingState(false)

        // Update streaming state flags
        isStreaming = false

        // Start background stop operations
        mainScope.launch {
            try {
                // Give UI time to update
                delay(50)

                // Calculate streaming duration
                val streamDuration = if (streamStartTime > 0) {
                    (System.currentTimeMillis() - streamStartTime) / 1000
                } else {
                    0
                }

                // For all streams, try normal stopping first
                stopDataStreaming()

                // Set a short timeout for completing the stop process
                withTimeoutOrNull(1000) {
                    while (isStoppingInProgress) {
                        delay(50)
                    }
                }

                // If we're still stopping after the timeout, use force disconnect
                if (isStoppingInProgress) {
                    addDataToLog("Stop operation taking too long, forcing disconnect")
                    forceDisconnectAndReconnect()
                }

                // Update UI state again
                delay(300)
                startStreamButton.isEnabled = true
                stopStreamButton.isEnabled = false
                addDataToLog("Accelerometer streaming stopped")

            } catch (e: Exception) {
                Log.e("DeviceSetupFragment", "Error in stop button handler", e)
                startStreamButton.isEnabled = true
                stopStreamButton.isEnabled = false
                forceDisconnectAndReconnect()
            }
        }
    }

    /**
     * Force disconnect from the device and then reconnect
     * This is a last-resort measure that always works even with long-running streams
     */
    private fun forceDisconnectAndReconnect() {
        try {
            Log.w("DeviceSetupFragment", "Force disconnecting from device")

            // Clear any ongoing operations
            isStoppingInProgress = false
            isStreaming = false
            isStartingInProgress = false

            // Reset counters
            dataPointCounter.set(0)
            streamStartTime = 0

            // Force-close data session
            sensorDataManager.stopSession()

            // Disconnect completely from the board
            mwBoard?.let { board ->
                try {
                    // Cancel any pending operations
                    accelerometer = null
                    routeSubscription = null

                    // Disconnect from the board
                    ioScope.launch {
                        try {
                            board.disconnectAsync()

                            // Give time for disconnect to complete
                            delay(500)

                            // Update UI on success
                            mainScope.launch {
                                addDataToLog("Device disconnected successfully")
                                startStreamButton.isEnabled = true
                                stopStreamButton.isEnabled = false
                            }

                            // Try to reconnect after a short delay
                            delay(1000)

                            try {
                                board.connectAsync()
                                mainScope.launch {
                                    addDataToLog("Device reconnected")
                                }
                            } catch (e: Exception) {
                                Log.e("DeviceSetupFragment", "Error reconnecting", e)
                            }

                        } catch (e: Exception) {
                            Log.e("DeviceSetupFragment", "Error during forced disconnect", e)
                        }
                    }
                } catch (e: Exception) {
                    Log.e("DeviceSetupFragment", "Exception in disconnect", e)
                }
            }

        } catch (e: Exception) {
            Log.e("DeviceSetupFragment", "Error during force disconnect", e)
        }
    }

    /**
     * Log a footfall event (volume button press)
     * Called from the activity when volume up button is pressed
     */
    fun logFootfall() {
        if (!isStreaming) {
            return
        }

        // Save the footfall event to the data file
        sensorDataManager.logFootfallEvent()

        // Update UI
        footfallCount++
        val message = "Footfall #$footfallCount logged"
        addDataToLog(message)
    }

    /**
     * Navigate to the activity that shows stored sensor data
     */
    private fun navigateToSensorDataViewActivity() {
        val intent = Intent(requireContext(), SensorDataViewActivity::class.java)
        startActivity(intent)
    }

    /**
     * Start the step detection with sound
     */
    private fun startStepDetection() {
        stepDetectorService?.let { service ->
            // Set callback to update UI
            service.onStepDetected = { count ->
                activity?.runOnUiThread {
                    stepCountText.text = getString(R.string.step_count_format, count)
                    stepCountText.visibility = View.VISIBLE
                }
            }

            // Start detection
            if (service.startDetecting()) {
                addDataToLog("Step detection with sound activated")
                service.resetStepCount()
                stepCountText.visibility = View.VISIBLE
            } else {
                addDataToLog("Error: Step detector not available on this device")
                stepDetectorSwitch.isChecked = false
                stepCountText.visibility = View.GONE
            }
        }
    }

    /**
     * Stop the step detection
     */
    private fun stopStepDetection() {
        stepDetectorService?.stopDetecting()
        addDataToLog("Step detection deactivated")
        stepCountText.visibility = View.GONE
    }

    /**
     * Start the inference service
     */
    private fun startInference() {
        inferenceService.startInference()
        inferenceStatusText.visibility = View.VISIBLE
        inferenceStatusText.text = getString(R.string.inference_count_format, 0)

        // Collect detection counts using Flow
        mainScope.launch {
            try {
                inferenceService.detectionCountFlow.collect { count ->
                    inferenceStatusText.text = getString(R.string.inference_count_format, count)
                }
            } catch (e: Exception) {
                Log.e("DeviceSetupFragment", "Error collecting inference counts", e)
            }
        }

        addDataToLog("Neural network inference started")
    }

    /**
     * Stop the inference service
     */
    private fun stopInference() {
        inferenceService.stopInference()
        inferenceStatusText.visibility = View.GONE
        addDataToLog("Neural network inference stopped")
    }
}