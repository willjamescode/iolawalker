package com.example.footfallng

import android.bluetooth.*
import android.content.*
import android.os.Bundle
import android.os.IBinder
import android.util.Log
import android.app.ProgressDialog
import android.widget.Button
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.fragment.app.commit
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.ContextCompat
import bolts.Task
import com.mbientlab.metawear.MetaWearBoard
import com.mbientlab.metawear.android.BtleService
import java.util.*
import android.Manifest
import android.content.pm.PackageManager

/**
 * Extension function to safely handle getParcelableExtra across Android versions
 */
@Suppress("DEPRECATION")
inline fun <reified T : android.os.Parcelable> Intent.getParcelableExtraCompat(key: String): T? {
    return if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.TIRAMISU) {
        getParcelableExtra(key, T::class.java)
    } else {
        getParcelableExtra(key)
    }
}

class MainActivity :
    AppCompatActivity(),
    BleScanFragment.ScanCallbacks,
    ServiceConnection {

    companion object {
        private const val REQUEST_START_APP = 1
        private const val TAG = "MainActivity"
        private const val ALL_PERMISSIONS_REQUEST_CODE = 123
        private const val TARGET_MAC_ADDRESS = "F3:CD:37:B7:4E:0B"
    }

    private var serviceBinder: BtleService.LocalBinder? = null
    private var mwBoard: MetaWearBoard? = null

    // Permissions needed based on Android API level
    private val permissionsNeeded: Array<String>
        get() = if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.S) {
            arrayOf(
                Manifest.permission.BLUETOOTH_SCAN,
                Manifest.permission.BLUETOOTH_CONNECT,
                Manifest.permission.ACCESS_FINE_LOCATION
            )
        } else {
            arrayOf(
                Manifest.permission.ACCESS_FINE_LOCATION
            )
        }

    // Permission launcher for Android API 30+
    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { permissions ->
        val allGranted = permissions.all { it.value }
        if (allGranted) {
            Log.d(TAG, "All permissions granted, binding to service")
            // Proceed with Bluetooth operations
            bindBleService()
        } else {
            Log.d(TAG, "Not all permissions granted")
            Toast.makeText(
                this,
                "Bluetooth permissions are required for this app to function",
                Toast.LENGTH_LONG
            ).show()
        }
    }

    /* ---------- Android life-cycle ------------------------------------ */

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        Log.d(TAG, "MainActivity onCreate called")

        try {
            // attach the scanning fragment
            if (savedInstanceState == null) {
                supportFragmentManager.commit {
                    replace(
                        R.id.scanner_fragment,                 // same container ID
                        BleScanFragment()                      // our replacement
                    )
                }
                Log.d(TAG, "BleScanFragment attached to MainActivity")
            }

            // Set up connect button for the known device
            findViewById<Button>(R.id.btn_connect_device).setOnClickListener {
                Log.d(TAG, "Connect button clicked")
                if (hasRequiredPermissions()) {
                    connectToKnownDevice()
                } else {
                    requestBluetoothPermissions()
                }
            }

            // Check permissions first, then bind to service if granted
            Log.d(TAG, "Checking permissions in onCreate")
            if (hasRequiredPermissions()) {
                Log.d(TAG, "Permissions granted, binding to service")
                bindBleService()
            } else {
                Log.d(TAG, "Permissions not granted, requesting permissions")
                requestBluetoothPermissions()
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error during MainActivity initialization", e)
            Toast.makeText(this, "Error initializing app: ${e.message}", Toast.LENGTH_LONG).show()
        }
    }

    private fun bindBleService() {
        // bind to the MetaWear service
        applicationContext.bindService(
            Intent(this, BtleService::class.java),
            this,
            BIND_AUTO_CREATE
        )
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)

        if (requestCode == ALL_PERMISSIONS_REQUEST_CODE) {
            if (grantResults.isNotEmpty() && grantResults.all { it == PackageManager.PERMISSION_GRANTED }) {
                // All permissions granted, proceed with BLE operations
                bindBleService()
            } else {
                Toast.makeText(
                    this,
                    "Bluetooth permissions are required for this app to function",
                    Toast.LENGTH_LONG
                ).show()
            }
        }
    }

    @Deprecated("Deprecated in Java")
    override fun onActivityResult(req: Int, res: Int, data: Intent?) {
        super.onActivityResult(req, res, data)

        if (req == REQUEST_START_APP) {
            (supportFragmentManager.findFragmentById(R.id.scanner_fragment)
                    as? BleScanFragment)?.startBleScan()
        }
    }

    override fun onDestroy() {
        super.onDestroy()

        // Ensure we properly unbind when the activity is destroyed
        serviceBinder?.let {
            applicationContext.unbindService(this)
        }
    }

    /* ---------- Direct connection to known device -------------------- */

    /**
     * Connect directly to our known device by MAC address
     * Call this after service binding is complete
     */
    private fun connectToKnownDevice() {
        if (serviceBinder == null) {
            Log.e(TAG, "Service not bound yet, can't connect to known device")
            return
        }

        // Check for Bluetooth permissions
        if (!hasRequiredPermissions()) {
            requestBluetoothPermissions()
            return
        }

        try {
            // Get the adapter and connect to the known device
            val bluetoothAdapter =
                (getSystemService(Context.BLUETOOTH_SERVICE) as BluetoothManager).adapter
            val device = bluetoothAdapter.getRemoteDevice(TARGET_MAC_ADDRESS)

            Log.d(TAG, "Found target device: ${device.name ?: "Unknown"} - ${device.address}")

            // Use the same connection logic as in onDeviceSelected
            connectToDevice(device)
        } catch (e: SecurityException) {
            Log.e(TAG, "Security exception accessing Bluetooth: ${e.message}")
            requestBluetoothPermissions()
        } catch (e: Exception) {
            Log.e(TAG, "Error connecting to known device: ${e.message}", e)
        }
    }

    /**
     * Connect to a specific device - extracted from onDeviceSelected
     * to be reusable for direct connection
     */
    private fun connectToDevice(device: BluetoothDevice) {
        mwBoard = serviceBinder!!.getMetaWearBoard(device)

        val dlg = ProgressDialog(this).apply {
            setTitle(getString(R.string.title_connecting))
            setMessage(getString(R.string.message_wait))
            setCancelable(false)
            setIndeterminate(true)
            setButton(
                DialogInterface.BUTTON_NEGATIVE,
                getString(android.R.string.cancel)
            ) { _, _ -> mwBoard?.disconnectAsync() }
            show()
        }

        mwBoard!!.connectAsync()
            .continueWithTask { t -> if (t.isCancelled || !t.isFaulted) t else reconnect(mwBoard!!) }
            .continueWith {
                if (!it.isCancelled) {
                    runOnUiThread { dlg.dismiss() }
                    startActivityForResult(
                        Intent(this, DeviceSetupActivity::class.java)
                            .putExtra(DeviceSetupActivity.EXTRA_BT_DEVICE, device),
                        REQUEST_START_APP
                    )
                }
                null
            }
    }

    /* ---------- ServiceConnection ------------------------------------- */

    override fun onServiceConnected(name: ComponentName?, service: IBinder) {
        Log.d(TAG, "Service connected to MainActivity")
        serviceBinder = service as BtleService.LocalBinder
        connectToKnownDevice()
    }

    override fun onServiceDisconnected(name: ComponentName?) = Unit

    /* ---------- BleScanFragment callbacks ----------------------------- */

    /** We only care about devices advertising the MetaWear service UUID. */
    override fun filterServiceUuids(): Array<UUID> =
        arrayOf(MetaWearBoard.METAWEAR_GATT_SERVICE)

    /** How long each scan session should last (ms). */
    override fun scanDuration(): Long = 10_000L

    override fun onDeviceSelected(device: BluetoothDevice) {
        // Check for Bluetooth permissions
        if (!hasRequiredPermissions()) {
            requestBluetoothPermissions()
            return
        }

        // Connect to the selected device
        connectToDevice(device)
    }

    /**
     * Check if we have all required permissions
     * Implementation of BleScanFragment.ScanCallbacks interface
     */
    override fun hasRequiredPermissions(): Boolean {
        val hasPermissions = permissionsNeeded.all { permission ->
            ContextCompat.checkSelfPermission(this, permission) == PackageManager.PERMISSION_GRANTED
        }
        Log.d(TAG, "hasRequiredPermissions check: $hasPermissions")
        return hasPermissions
    }

    /**
     * Request Bluetooth permissions
     * Implementation of BleScanFragment.ScanCallbacks interface
     */
    override fun requestBluetoothPermissions() {
        Log.d(TAG, "Requesting Bluetooth permissions")
        requestPermissionLauncher.launch(permissionsNeeded)
    }

    /* ---------- util --------------------------------------------------- */

    private fun reconnect(board: MetaWearBoard): Task<Void> =
        board.connectAsync()
            .continueWithTask { t -> if (t.isFaulted) reconnect(board) else t }
}