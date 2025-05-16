package com.example.footfallng.data

import android.content.Context
import android.content.Intent
import android.util.Log
import androidx.annotation.WorkerThread
import androidx.core.content.ContextCompat
import java.io.BufferedWriter
import java.io.File
import java.io.FileWriter
import java.io.IOException
import java.text.SimpleDateFormat
import java.util.ArrayDeque
import java.util.Date
import java.util.Locale
import java.util.UUID
import java.util.concurrent.ArrayBlockingQueue
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicLong
import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.withLock
import kotlin.math.min

/**
 * Robust and efficient manager for sensor data optimized for 200Hz sample rates
 * 
 * Features:
 * - Buffered writes with background flushing
 * - Non-blocking producer/consumer pattern
 * - Minimal allocations on hot path
 * - Robust error handling and recovery
 * - Strict bounds on memory and file size
 * - Session management for continuous operation
 */
class SensorDataManager(private val context: Context) {

    companion object {
        private const val TAG = "SensorDataManager"
        
        // Sampling parameters
        private const val SAMPLE_RATE_HZ = 200               // 200Hz accelerometer sampling
        private const val MAX_FOOTFALLS_PER_SEC = 10         // Maximum expected step freq
        private const val EXPECTED_EVENTS_PER_SEC = SAMPLE_RATE_HZ + MAX_FOOTFALLS_PER_SEC
        
        // Buffer sizing
        private const val BUFFER_CAPACITY_SECONDS = 5        // Buffer capacity in seconds
        private const val BUFFER_SIZE = EXPECTED_EVENTS_PER_SEC * BUFFER_CAPACITY_SECONDS
        
        // Background writer settings
        private const val FLUSH_INTERVAL_MS = 1000L          // Flush interval in ms
        private const val FLUSH_THRESHOLD = EXPECTED_EVENTS_PER_SEC // Flush after 1s worth of data
        private const val FILE_WRITER_BUFFER_SIZE = 65536    // 64KB buffer for file writer
        private const val STRING_BUILDER_CAPACITY = 8192     // Preallocated StringBuilder size
        
        // Constants for file naming and headers
        private const val FOOTFALL_MARKER = "FOOTFALL"
        private const val DATA_LOSS_MARKER = "DATA_LOSS"
    }
    
    /**
     * Data class representing a sensor sample or event
     * Immutable fields to avoid unexpected modification
     */
    private data class Sample(
        val timestamp: Long,
        val x: Float,
        val y: Float,
        val z: Float,
        val isFootfall: Boolean,
        val sequenceNumber: Long
    )
    
    // Global sequence counter to maintain strict ordering
    private val sequenceCounter = AtomicLong(0)
    
    // Current session state
    private var currentSessionId: String? = null
    private var sessionStartTime = 0L 
    private val isRecording = AtomicBoolean(false)
    
    // Data collection & statistics
    private var sampleCount = AtomicInteger(0) 
    private var footfallCount = AtomicInteger(0)
    private var droppedSampleCount = AtomicInteger(0)
    private var lastReportTime = AtomicLong(0)
    
    // Logging & rate limiting
    private val logThrottle = ThrottledLogger(TAG)
    
    // Background processor state
    @Volatile private var backgroundThread: Thread? = null
    private var shouldStop = AtomicBoolean(false)
    
    // Bounded buffer for sample data
    private val sampleQueue = ArrayBlockingQueue<Sample>(BUFFER_SIZE)
    
    // File handling
    private val fileLock = ReentrantLock()
    private var writer: BufferedWriter? = null
    private val stringBuilder = StringBuilder(STRING_BUILDER_CAPACITY)  // Reuse StringBuilder
    
    /**
     * Start a new recording session
     * 
     * @return Session ID or null if start failed
     */
    fun startNewSession(): String {
        // Don't allow starting multiple sessions
        if (isRecording.get()) {
            stopSession()
        }
        
        try {
            // Generate new session ID and metadata
            val sessionId = generateSessionId()
            currentSessionId = sessionId
            sessionStartTime = System.currentTimeMillis()
            
            // Setup counters
            sampleCount.set(0)
            footfallCount.set(0)
            droppedSampleCount.set(0)
            sequenceCounter.set(0)
            
            // Setup recording file
            if (!setupRecordingFile(sessionId)) {
                Log.e(TAG, "Failed to create recording file")
                return "error"
            }
            
            // Start background processor
            shouldStop.set(false)
            startBackgroundProcessor()
            
            // Set recording active
            isRecording.set(true)
            
            Log.d(TAG, "Started new recording session: $sessionId")
            return sessionId
        } catch (e: Exception) {
            Log.e(TAG, "Error starting session", e)
            cleanupResources()
            return "error"
        }
    }
    
    /**
     * Stop the current recording session
     */
    fun stopSession() {
        if (!isRecording.get()) {
            return
        }
        
        isRecording.set(false)
        
        try {
            // Signal background thread to exit
            shouldStop.set(true)
            
            // Wait for background thread to complete (with timeout)
            backgroundThread?.let { thread ->
                try {
                    thread.join(2000)  // Wait up to 2 seconds
                } catch (e: InterruptedException) {
                    Log.w(TAG, "Interrupted while waiting for background thread")
                }
                
                // Force interrupt if still alive
                if (thread.isAlive) {
                    thread.interrupt()
                }
            }
            
            // Ensure final flush and cleanup
            flushQueueSynchronously()
            cleanupResources()
            
            val sessionDuration = (System.currentTimeMillis() - sessionStartTime) / 1000
            Log.d(TAG, "Session stopped. Duration: ${sessionDuration}s, " +
                "Samples: ${sampleCount.get()}, Footfalls: ${footfallCount.get()}")
            
        } catch (e: Exception) {
            Log.e(TAG, "Error stopping session", e)
        } finally {
            // Ensure resources are always cleaned up
            cleanupResources()
            currentSessionId = null
            backgroundThread = null
        }
    }
    
    /**
     * Save a sensor reading (accelerometer data)
     * Thread-safe, non-blocking implementation
     * 
     * @param x X-axis acceleration in G
     * @param y Y-axis acceleration in G
     * @param z Z-axis acceleration in G
     * @return true if sample was successfully queued
     */
    fun saveSensorReading(x: Float, y: Float, z: Float): Boolean {
        if (!isRecording.get()) {
            return false
        }
        
        // Validate data to avoid saving corrupted values
        if (x.isNaN() || y.isNaN() || z.isNaN() || 
            x.isInfinite() || y.isInfinite() || z.isInfinite() ||
            x < -20f || x > 20f || y < -20f || y > 20f || z < -20f || z > 20f) {
            logThrottle.log("Invalid sensor data rejected: $x, $y, $z", LogType.WARNING)
            return false
        }
        
        val sample = Sample(
            timestamp = System.currentTimeMillis(),
            x = x,
            y = y,
            z = z,
            isFootfall = false,
            sequenceNumber = sequenceCounter.getAndIncrement()
        )
        
        // Try to add to queue - non-blocking for responsiveness
        val success = sampleQueue.offer(sample)
        
        if (success) {
            sampleCount.incrementAndGet()
        } else {
            droppedSampleCount.incrementAndGet()
            
            // Log dropped samples (rate limited)
            val now = System.currentTimeMillis()
            if (now - lastReportTime.get() > 5000) {
                lastReportTime.set(now)
                Log.w(TAG, "Buffer full, dropped ${droppedSampleCount.get()} samples")
            }
        }
        
        return success
    }
    
    /**
     * Log a footfall event (high priority event)
     * 
     * @return true if event was successfully logged
     */
    fun logFootfallEvent(): Boolean {
        if (!isRecording.get()) {
            return false
        }
        
        val sample = Sample(
            timestamp = System.currentTimeMillis(),
            x = 0f,
            y = 0f,
            z = 0f,
            isFootfall = true,
            sequenceNumber = sequenceCounter.getAndIncrement()
        )
        
        // For critical events, we're willing to wait briefly
        val success = try {
            // Wait up to 100ms to record footfall
            sampleQueue.offer(sample, 100, TimeUnit.MILLISECONDS)
        } catch (e: InterruptedException) {
            false
        }
        
        if (success) {
            footfallCount.incrementAndGet()
        }
        
        return success
    }
    
    /**
     * Get list of all recorded session files
     */
    fun getSessionFiles(): List<File> {
        val dataDir = getDataDirectory()
        return dataDir.listFiles { file ->
            file.isFile && file.name.endsWith(".csv")
        }?.sortedByDescending { it.lastModified() } ?: emptyList()
    }
    
    /**
     * Delete all recorded data files
     */
    fun deleteAllData(): Boolean {
        // Stop any active recording first
        if (isRecording.get()) {
            stopSession()
        }
        
        var success = true
        val dataDir = getDataDirectory()
        
        try {
            dataDir.listFiles()?.forEach { file ->
                if (!file.delete()) {
                    Log.w(TAG, "Could not delete file: ${file.name}")
                    success = false
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error deleting data files", e)
            success = false
        }
        
        return success
    }
    
    /**
     * Get path to data directory
     */
    fun getDataDirectoryPath(): String {
        return getDataDirectory().absolutePath
    }
    
    /**
     * Get statistics about current recording session
     */
    fun getSessionStats(): Map<String, Any> {
        val duration = if (sessionStartTime > 0) {
            (System.currentTimeMillis() - sessionStartTime) / 1000
        } else 0
        
        return mapOf(
            "active" to isRecording.get(),
            "sessionId" to (currentSessionId ?: "none"),
            "duration" to duration,
            "samples" to sampleCount.get(),
            "footfalls" to footfallCount.get(),
            "dropped" to droppedSampleCount.get(),
            "buffered" to sampleQueue.size
        )
    }
    
    /**
     * Generate a unique session ID
     */
    private fun generateSessionId(): String {
        return UUID.randomUUID().toString().substring(0, 8)
    }
    
    /**
     * Setup file for recording
     */
    private fun setupRecordingFile(sessionId: String): Boolean {
        fileLock.withLock {
            try {
                // Create data directory if it doesn't exist
                val dataDir = getDataDirectory()
                if (!dataDir.exists() && !dataDir.mkdirs()) {
                    Log.e(TAG, "Failed to create data directory")
                    return false
                }
                
                // Create timestamped filename
                val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
                val fileName = "sensor_data_${timestamp}_$sessionId.csv"
                val file = File(dataDir, fileName)
                
                // Create buffered writer with 64KB buffer
                writer = BufferedWriter(FileWriter(file), FILE_WRITER_BUFFER_SIZE)
                
                // Write CSV header
                writer?.write("timestamp,accelerationX,accelerationY,accelerationZ,sequenceNumber,eventType\n")
                writer?.flush()
                
                return true
            } catch (e: IOException) {
                Log.e(TAG, "Error creating recording file", e)
                return false
            }
        }
    }
    
    /**
     * Start background thread to process queued samples
     */
    private fun startBackgroundProcessor() {
        backgroundThread = Thread({
            backgroundProcessingLoop()
        }, "SensorDataProcessor").apply {
            priority = Thread.NORM_PRIORITY - 1  // Slightly lower than normal priority
            isDaemon = true
            start()
        }
    }
    
    /**
     * Main processing loop for background thread
     */
    private fun backgroundProcessingLoop() {
        Log.d(TAG, "Background processor started")
        
        val tempBuffer = ArrayDeque<Sample>(FLUSH_THRESHOLD)
        var lastFlushTime = System.currentTimeMillis()
        
        try {
            while (!shouldStop.get()) {
                val now = System.currentTimeMillis()
                val timeSinceLastFlush = now - lastFlushTime
                val shouldFlushOnTime = timeSinceLastFlush >= FLUSH_INTERVAL_MS
                val shouldFlushOnSize = tempBuffer.size >= FLUSH_THRESHOLD
                
                // Attempt to pull more samples before flushing
                if (!shouldFlushOnTime && !shouldFlushOnSize) {
                    // Calculate time to wait (with safety)
                    val timeToWait = min(
                        FLUSH_INTERVAL_MS - timeSinceLastFlush + 10, 
                        FLUSH_INTERVAL_MS
                    )
                    
                    try {
                        // Wait up to one full interval for a sample (shorter if flush time approaching)
                        val sample = sampleQueue.poll(timeToWait, TimeUnit.MILLISECONDS)
                        if (sample != null) {
                            tempBuffer.add(sample)
                        }
                    } catch (e: InterruptedException) {
                        // Thread was interrupted, check if we should exit
                        if (shouldStop.get()) break
                    }
                    
                    // Re-check flush conditions
                    if (!shouldStop.get() && 
                        tempBuffer.isNotEmpty() && 
                        (System.currentTimeMillis() - lastFlushTime >= FLUSH_INTERVAL_MS || 
                         tempBuffer.size >= FLUSH_THRESHOLD)) {
                        flushBuffer(tempBuffer)
                        lastFlushTime = System.currentTimeMillis()
                    }
                    
                    continue
                }
                
                // Time to flush - gather any waiting samples first
                drainQueueToBuffer(tempBuffer, 0)
                
                if (tempBuffer.isNotEmpty()) {
                    flushBuffer(tempBuffer)
                    lastFlushTime = System.currentTimeMillis()
                }
            }
            
            // Final flush
            drainQueueToBuffer(tempBuffer, 100)  // Wait for stragglers
            if (tempBuffer.isNotEmpty()) {
                flushBuffer(tempBuffer)
            }
            
        } catch (e: Exception) {
            Log.e(TAG, "Error in background processor", e)
        } finally {
            Log.d(TAG, "Background processor stopped")
        }
    }
    
    /**
     * Drain samples from queue to temporary buffer
     */
    private fun drainQueueToBuffer(buffer: ArrayDeque<Sample>, waitTimeMs: Long) {
        // Fast path - drain immediately available samples
        val sample = sampleQueue.poll()
        if (sample != null) {
            buffer.add(sample)
        }
        
        // Drain all immediately available samples
        if (waitTimeMs <= 0) {
            sampleQueue.drainTo(buffer)
            return
        }
        
        // Wait a bit for final samples
        val deadline = System.currentTimeMillis() + waitTimeMs
        while (System.currentTimeMillis() < deadline) {
            val s = sampleQueue.poll(5, TimeUnit.MILLISECONDS) ?: break
            buffer.add(s)
        }
        
        // Final drain of any samples that arrived during the wait
        sampleQueue.drainTo(buffer)
    }
    
    /**
     * Synchronously flush all queued samples
     * Used during shutdown to ensure all data is written
     */
    private fun flushQueueSynchronously() {
        val buffer = ArrayDeque<Sample>(sampleQueue.size)
        sampleQueue.drainTo(buffer)
        
        if (buffer.isNotEmpty()) {
            flushBuffer(buffer)
        }
    }
    
    /**
     * Flush buffer to disk
     */
    @WorkerThread
    private fun flushBuffer(buffer: ArrayDeque<Sample>) {
        if (buffer.isEmpty()) return
        
        fileLock.withLock {
            try {
                val currentWriter = writer ?: return
                
                // Reuse StringBuilder
                stringBuilder.setLength(0)
                
                // Format all samples for writing
                for (sample in buffer) {
                    stringBuilder.append(sample.timestamp).append(',')
                    
                    if (sample.isFootfall) {
                        // Format footfall event
                        stringBuilder.append(FOOTFALL_MARKER).append(',')
                            .append(FOOTFALL_MARKER).append(',')
                            .append(FOOTFALL_MARKER)
                    } else {
                        // Format normal sample
                        stringBuilder.append(sample.x).append(',')
                            .append(sample.y).append(',')
                            .append(sample.z)
                    }
                    
                    // Add sequence number and event type
                    stringBuilder.append(',').append(sample.sequenceNumber).append(',')
                        .append(if (sample.isFootfall) "footfall" else "sample")
                        .append('\n')
                }
                
                // Batch write to file
                currentWriter.write(stringBuilder.toString())
                
                // Only flush if buffer is getting full or session is ending
                if (sampleQueue.size < BUFFER_SIZE * 0.2) {
                    currentWriter.flush()
                }
                
                buffer.clear()
                
            } catch (e: IOException) {
                Log.e(TAG, "Error writing to file", e)
            }
        }
    }
    
    /**
     * Clean up all resources
     */
    private fun cleanupResources() {
        fileLock.withLock {
            try {
                writer?.flush()
                writer?.close()
                writer = null
            } catch (e: IOException) {
                Log.e(TAG, "Error closing file", e)
            }
        }
    }
    
    /**
     * Get data directory, creating if needed
     */
    private fun getDataDirectory(): File {
        val dir = File(context.getExternalFilesDir(null), "sensor_data")
        if (!dir.exists()) {
            dir.mkdirs()
        }
        return dir
    }
    
    /**
     * Helper class for rate-limited logging
     */
    private inner class ThrottledLogger(private val tag: String) {
        private var lastLogTime = 0L
        private var suppressedLogs = 0
        
        fun log(message: String, type: LogType) {
            val now = System.currentTimeMillis()
            
            // Allow important logs to bypass throttling
            if (type == LogType.ERROR) {
                if (suppressedLogs > 0) {
                    Log.e(tag, "($suppressedLogs logs suppressed)")
                    suppressedLogs = 0
                }
                Log.e(tag, message)
                return
            }
            
            // Rate limit everything else to once per second
            if (now - lastLogTime > 1000) {
                if (suppressedLogs > 0) {
                    Log.d(tag, "($suppressedLogs logs suppressed)")
                    suppressedLogs = 0
                }
                
                when(type) {
                    LogType.DEBUG -> Log.d(tag, message)
                    LogType.INFO -> Log.i(tag, message)
                    LogType.WARNING -> Log.w(tag, message)
                    LogType.ERROR -> Log.e(tag, message) // Should never reach here
                }
                
                lastLogTime = now
            } else {
                suppressedLogs++
            }
        }
    }
    
    /**
     * Log types for throttled logger
     */
    private enum class LogType {
        DEBUG, INFO, WARNING, ERROR
    }
}