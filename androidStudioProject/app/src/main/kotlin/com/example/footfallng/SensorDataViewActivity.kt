package com.example.footfallng

import android.os.Bundle
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.ListView
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import com.example.footfallng.data.SensorDataManager
import java.io.BufferedReader
import java.io.File
import java.io.FileReader
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class SensorDataViewActivity : AppCompatActivity() {

    private lateinit var filesList: ListView
    private lateinit var dataInfo: TextView
    private lateinit var deleteButton: Button
    private lateinit var sensorDataManager: SensorDataManager

    private val files = mutableListOf<File>()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_sensor_data_view)

        // Initialize views
        filesList = findViewById(R.id.files_list)
        dataInfo = findViewById(R.id.data_info)
        deleteButton = findViewById(R.id.btn_delete_data)

        // Initialize data manager
        sensorDataManager = SensorDataManager(applicationContext)

        // Set up the files list
        loadSensorFiles()

        // Set up list item click listener
        filesList.setOnItemClickListener { _, _, position, _ ->
            val selectedFile = files[position]
            showFilePreview(selectedFile)
        }

        // Set up delete button
        deleteButton.setOnClickListener {
            confirmDeleteData()
        }

        // Show data directory path
        dataInfo.text = "Data stored in: ${sensorDataManager.getDataDirectoryPath()}"
    }

    override fun onResume() {
        super.onResume()
        // Refresh file list
        loadSensorFiles()
    }

    private fun loadSensorFiles() {
        // Get all sensor data files
        files.clear()
        files.addAll(sensorDataManager.getSessionFiles())

        // Create an adapter to show file names
        val fileNames = files.map { file ->
            // Extract timestamp from filename
            val name = file.name
            val size = file.length() / 1024 // Size in KB
            val date = SimpleDateFormat("yyyy-MM-dd HH:mm", Locale.getDefault())
                .format(Date(file.lastModified()))

            "$name ($size KB) - $date"
        }

        val adapter = ArrayAdapter(
            this,
            android.R.layout.simple_list_item_1,
            fileNames
        )

        filesList.adapter = adapter

        if (files.isEmpty()) {
            dataInfo.text = "No sensor data files found"
        } else {
            dataInfo.text = "${files.size} data files found"
        }
    }

    private fun showFilePreview(file: File) {
        try {
            val reader = BufferedReader(FileReader(file))
            val lines = mutableListOf<String>()

            // Read header
            val header = reader.readLine()
            lines.add(header ?: "No header")

            // Read first 10 data lines
            for (i in 0 until 10) {
                val line = reader.readLine() ?: break
                lines.add(line)
            }

            reader.close()

            // Show preview dialog
            AlertDialog.Builder(this)
                .setTitle(file.name)
                .setMessage(lines.joinToString("\n"))
                .setPositiveButton("OK") { dialog, _ -> dialog.dismiss() }
                .show()

        } catch (e: Exception) {
            Toast.makeText(this, "Error reading file: ${e.message}", Toast.LENGTH_SHORT).show()
        }
    }

    private fun confirmDeleteData() {
        AlertDialog.Builder(this)
            .setTitle("Delete Data")
            .setMessage("Are you sure you want to delete all sensor data files?")
            .setPositiveButton("Delete") { _, _ ->
                if (sensorDataManager.deleteAllData()) {
                    Toast.makeText(this, "All data files deleted", Toast.LENGTH_SHORT).show()
                    loadSensorFiles()
                } else {
                    Toast.makeText(this, "Failed to delete some files", Toast.LENGTH_SHORT).show()
                }
            }
            .setNegativeButton("Cancel") { dialog, _ -> dialog.dismiss() }
            .show()
    }
}