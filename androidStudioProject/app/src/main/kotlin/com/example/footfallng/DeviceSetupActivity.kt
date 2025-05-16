package com.example.footfallng

import android.bluetooth.BluetoothDevice
import android.content.ComponentName
import android.content.Intent
import android.content.ServiceConnection
import android.os.Bundle
import android.os.IBinder
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.widget.Toolbar
import bolts.Task
import com.mbientlab.metawear.MetaWearBoard
import com.mbientlab.metawear.android.BtleService
import android.view.KeyEvent
import android.widget.Toast

class DeviceSetupActivity : AppCompatActivity(), ServiceConnection {

    companion object {
        const val EXTRA_BT_DEVICE = "EXTRA_BT_DEVICE"
    }

    // Internal device reference
    private lateinit var bluetoothDevice: BluetoothDevice
    private lateinit var metawearBoard: MetaWearBoard

    // Create a specific implementation of FragmentSettings
    private inner class DeviceSettings : DeviceSetupActivityFragment.FragmentSettings {
        override val btDevice: BluetoothDevice
            get() = this@DeviceSetupActivity.bluetoothDevice
    }

    // Instance to share with fragments
    private val deviceSettings = DeviceSettings()

    // Flag to indicate if footfall tracking is active
    private var footfallTrackingActive = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_device_setup)
        findViewById<Toolbar>(R.id.toolbar).also(::setSupportActionBar)

        // Get the BluetoothDevice from the intent
        bluetoothDevice = intent.getParcelableExtraCompat<BluetoothDevice>(EXTRA_BT_DEVICE)
            ?: throw IllegalStateException("No BluetoothDevice provided")

        // Set up the fragment
        val fragment =
            supportFragmentManager.findFragmentById(R.id.device_setup_fragment) as? DeviceSetupActivityFragment
        fragment?.setSettings(deviceSettings)

        // Bind to the BtleService
        applicationContext.bindService(
            Intent(this, BtleService::class.java),
            this,
            BIND_AUTO_CREATE
        )
    }

    override fun onServiceConnected(name: ComponentName?, binder: IBinder) {
        val localBinder = binder as BtleService.LocalBinder
        metawearBoard = localBinder.getMetaWearBoard(bluetoothDevice)

        metawearBoard.onUnexpectedDisconnect {
            reconnect(metawearBoard).continueWith { task ->
                runOnUiThread {
                    val fragment =
                        supportFragmentManager.findFragmentById(R.id.device_setup_fragment) as? DeviceSetupActivityFragment
                    fragment?.reconnected()
                }
                null
            }
        }
    }

    override fun onServiceDisconnected(name: ComponentName?) {
        // Nothing to do
    }

    /**
     * Handle key events to capture volume button presses for footfall logging
     */
    override fun onKeyDown(keyCode: Int, event: KeyEvent?): Boolean {
        // Only handle volume up button as footfall marker
        if (keyCode == KeyEvent.KEYCODE_VOLUME_UP) {
            // If a session is active, log footfall
            if (footfallTrackingActive) {
                // Get fragment
                val fragment =
                    supportFragmentManager.findFragmentById(R.id.device_setup_fragment) as? DeviceSetupActivityFragment
                fragment?.logFootfall()

                // Provide haptic/visual feedback
                Toast.makeText(this, "Footfall logged", Toast.LENGTH_SHORT).show()

                // Consume the event
                return true
            }
        }

        // For all other keys, let the system handle it
        return super.onKeyDown(keyCode, event)
    }

    /**
     * Set the footfall tracking state
     * Called from the fragment when streaming starts/stops
     */
    fun setFootfallTrackingState(active: Boolean) {
        this.footfallTrackingActive = active

        if (active) {
            Toast.makeText(
                this,
                "Footfall tracking active - Press VOLUME UP to mark steps",
                Toast.LENGTH_LONG
            ).show()
        }
    }

    private fun reconnect(board: MetaWearBoard): Task<Void> {
        return board.connectAsync()
            .continueWithTask { task ->
                if (task.isFaulted) {
                    reconnect(board)
                } else {
                    task
                }
            }
    }
}