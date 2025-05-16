package com.example.footfallng

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.media.AudioManager
import android.media.ToneGenerator
import android.util.Log
import android.widget.Toast

/**
 * Service that uses Android's built-in step detector sensor to detect footfalls
 * and plays a sound when steps are detected.
 */
class StepDetectorService(private val context: Context) : SensorEventListener {

    private val TAG = "StepDetectorService"

    // Sensor manager and step detector sensor
    private val sensorManager: SensorManager =
        context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    private val stepDetectorSensor: Sensor? =
        sensorManager.getDefaultSensor(Sensor.TYPE_STEP_DETECTOR)

    // Sound generator for beep tones
    private var toneGenerator: ToneGenerator? = null

    // State
    private var isDetecting = false
    private var stepCount = 0

    // Callback for step detection
    var onStepDetected: ((Int) -> Unit)? = null

    init {
        try {
            // Initialize tone generator for beeps
            toneGenerator = ToneGenerator(AudioManager.STREAM_NOTIFICATION, 100)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize tone generator", e)
        }

        // Check if step detector is available
        if (stepDetectorSensor == null) {
            Log.e(TAG, "Step detector sensor not available on this device")
            Toast.makeText(
                context,
                "Step detector not available on this device.",
                Toast.LENGTH_LONG
            ).show()
        }
    }

    /**
     * Start detecting steps
     */
    fun startDetecting(): Boolean {
        if (stepDetectorSensor == null) {
            return false
        }

        if (!isDetecting) {
            // Register the step detector sensor
            sensorManager.registerListener(
                this,
                stepDetectorSensor,
                SensorManager.SENSOR_DELAY_NORMAL
            )

            isDetecting = true
            Log.d(TAG, "Step detection started")
            return true
        }
        return false
    }

    /**
     * Stop detecting steps
     */
    fun stopDetecting() {
        if (isDetecting) {
            sensorManager.unregisterListener(this)
            isDetecting = false
            Log.d(TAG, "Step detection stopped")
        }
    }

    /**
     * Get current detection state
     */
    fun isDetecting(): Boolean {
        return isDetecting
    }

    /**
     * Reset step counter
     */
    fun resetStepCount() {
        stepCount = 0
    }

    /**
     * Get current step count
     */
    fun getStepCount(): Int {
        return stepCount
    }

    /**
     * Clean up resources
     */
    fun cleanup() {
        stopDetecting()
        toneGenerator?.release()
        toneGenerator = null
    }

    // SensorEventListener implementation
    override fun onSensorChanged(event: SensorEvent) {
        if (event.sensor.type == Sensor.TYPE_STEP_DETECTOR) {
            // Step detector returns 1.0 when a step is detected
            if (event.values[0] == 1.0f) {
                stepCount++

                // Play a beep tone
                toneGenerator?.startTone(ToneGenerator.TONE_PROP_BEEP, 150)

                Log.d(TAG, "Step detected! Count: $stepCount")

                // Notify callback
                onStepDetected?.invoke(stepCount)
            }
        }
    }

    override fun onAccuracyChanged(sensor: Sensor, accuracy: Int) {
        // Not used but required by the interface
    }
}