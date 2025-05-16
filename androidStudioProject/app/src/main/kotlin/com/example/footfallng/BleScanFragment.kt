package com.example.footfallng

import android.bluetooth.*
import android.bluetooth.le.ScanFilter
import android.bluetooth.le.ScanCallback
import android.bluetooth.le.ScanResult
import android.bluetooth.le.ScanSettings
import android.content.*
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.os.ParcelUuid
import android.util.Log
import androidx.fragment.app.Fragment
import java.util.*

/**
 * Fragment that handles BLE scanning for MetaWear devices
 * Will prioritize connecting to our target device with MAC F3:CD:37:B7:4E:0B
 */
class BleScanFragment : Fragment() {

    companion object {
        private const val TAG = "BleScanFragment"
        private const val TARGET_MAC_ADDRESS = "F3:CD:37:B7:4E:0B"
    }

    /* The parent Activity implements this interface. */
    interface ScanCallbacks {
        fun filterServiceUuids(): Array<UUID>
        fun scanDuration(): Long
        fun onDeviceSelected(device: BluetoothDevice)
        fun hasRequiredPermissions(): Boolean
        fun requestBluetoothPermissions()
    }

    private val parent: ScanCallbacks
        get() = requireActivity() as? ScanCallbacks
            ?: error("Parent activity must implement ScanCallbacks")

    private val btAdapter: BluetoothAdapter? by lazy {
        try {
            (requireContext().getSystemService(Context.BLUETOOTH_SERVICE) as? BluetoothManager)?.adapter
        } catch (e: SecurityException) {
            Log.e(TAG, "Security exception accessing Bluetooth adapter", e)
            parent.requestBluetoothPermissions()
            null
        } catch (e: Exception) {
            Log.e(TAG, "Error accessing Bluetooth adapter", e)
            null
        }
    }

    private val ui = Handler(Looper.getMainLooper())
    private val scanCallback = object : ScanCallback() {
        override fun onScanResult(type: Int, result: ScanResult?) {
            result?.device?.let { dev ->
                // Log discovery of device
                Log.d(TAG, "Found ${dev.address} (${dev.name ?: "Unknown"})")

                // Prioritize our target device if found
                if (dev.address == TARGET_MAC_ADDRESS) {
                    Log.d(TAG, "Found target device!")
                    stopBleScan()
                    parent.onDeviceSelected(dev)
                    return
                }

                // For other devices, check if they have the required service UUID
                if (result.scanRecord?.serviceUuids?.any {
                        parent.filterServiceUuids().contains(it.uuid)
                    } == true
                ) {
                    Log.d(TAG, "Found device with matching service UUID")
                    stopBleScan()
                    parent.onDeviceSelected(dev)
                }
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        retainInstance = true
        Log.d(TAG, "BleScanFragment onCreate")
    }

    override fun onResume() {
        super.onResume()
        Log.d(TAG, "BleScanFragment onResume")
        if (parent.hasRequiredPermissions()) {
            Log.d(TAG, "Permissions verified in fragment, starting scan")
            startBleScan()
        } else {
            Log.d(TAG, "Permissions needed, requesting from fragment")
            parent.requestBluetoothPermissions()
        }
    }

    override fun onPause() {
        super.onPause()
        Log.d(TAG, "BleScanFragment onPause")
        if (parent.hasRequiredPermissions()) {
            stopBleScan()
        }
    }

    /* ---------- public API -------------------------------------------- */

    fun startBleScan() {
        Log.d(TAG, "startBleScan called")
        // Verify we have permissions first
        if (!parent.hasRequiredPermissions()) {
            Log.d(TAG, "Permissions missing, requesting from startBleScan")
            parent.requestBluetoothPermissions()
            return
        }

        try {
            // Check if Bluetooth adapter exists
            val adapter = btAdapter
            if (adapter == null) {
                Log.e(TAG, "Could not get Bluetooth adapter")
                return
            }

            // Check if Bluetooth is enabled
            if (!adapter.isEnabled) {
                Log.d(TAG, "Bluetooth is disabled, requesting to enable")
                try {
                    startActivity(Intent(BluetoothAdapter.ACTION_REQUEST_ENABLE))
                } catch (e: SecurityException) {
                    Log.e(TAG, "Security exception when enabling Bluetooth", e)
                    parent.requestBluetoothPermissions()
                }
                return
            }

            Log.d(TAG, "Setting up scan filters")
            val filters = parent.filterServiceUuids().map {
                Log.d(TAG, "Adding filter for UUID: $it")
                ScanFilter.Builder().setServiceUuid(ParcelUuid(it)).build()
            }

            Log.d(TAG, "Starting BLE scan with ${filters.size} filters")
            adapter.bluetoothLeScanner?.let { scanner ->
                try {
                    scanner.startScan(
                        filters,
                        ScanSettings.Builder().build(),
                        scanCallback
                    )
                    Log.d(TAG, "BLE scan started successfully")
                    ui.postDelayed({
                        Log.d(TAG, "Scan timeout reached")
                        stopBleScan()
                    }, parent.scanDuration())
                } catch (e: SecurityException) {
                    Log.e(TAG, "Security exception when starting BLE scan", e)
                    parent.requestBluetoothPermissions()
                } catch (e: Exception) {
                    Log.e(TAG, "Error starting BLE scan: ${e.message}", e)
                }
            } ?: Log.e(TAG, "BluetoothLeScanner is null")
        } catch (e: SecurityException) {
            Log.e(TAG, "Security exception accessing Bluetooth", e)
            parent.requestBluetoothPermissions()
        } catch (e: Exception) {
            Log.e(TAG, "Error setting up BLE scan: ${e.message}", e)
        }
    }

    fun stopBleScan() {
        // Verify we have permissions first
        if (!parent.hasRequiredPermissions()) {
            return
        }

        try {
            val adapter = btAdapter
            if (adapter == null) {
                Log.d(TAG, "Could not get Bluetooth adapter to stop scan")
                return
            }

            adapter.bluetoothLeScanner?.let { scanner ->
                try {
                    scanner.stopScan(scanCallback)
                    Log.d(TAG, "BLE scan stopped")
                } catch (e: SecurityException) {
                    Log.e(TAG, "Security exception when stopping BLE scan", e)
                } catch (e: Exception) {
                    Log.e(TAG, "Error stopping BLE scan: ${e.message}", e)
                }
            } ?: Log.d(TAG, "BluetoothLeScanner is null when trying to stop scan")
        } catch (e: SecurityException) {
            Log.e(TAG, "Security exception accessing Bluetooth adapter", e)
        } catch (e: Exception) {
            Log.e(TAG, "Error accessing Bluetooth adapter: ${e.message}", e)
        }
    }
}