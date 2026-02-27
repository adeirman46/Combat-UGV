#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wideband RF Anti-Jam Encrypted Frequency Hopping Subscriber
- Mapped 0 to 6 GHz Spectrum
- Perfect Jammer sync using 10Hz ROS2 state updates from publisher
- Decrypts AES-256 hopping messages
"""
import sys
import signal
import base64
import threading

from PyQt5 import Qt, QtCore, QtWidgets
import sip

from gnuradio import qtgui, analog, blocks, filter, gr
from gnuradio.filter import firdes
from gnuradio.fft import window

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.backends import default_backend

SHARED_AES_KEY = b'this_is_a_32_byte_aes_256_key_!!'

# ============================================================
# Wideband Mapping (0 - 6 GHz)
# ============================================================
DISP_CENTER = 3.0e9
DISP_BW = 6.0e9
SAMP_RATE = 32000
INTERP = 10
DSP_FS = SAMP_RATE * INTERP

BANDS = [
    {"label": "433 MHz (UHF/Long Range)",   "freq": 433e6},
    {"label": "915 MHz (UHF/US ISM)",       "freq": 915e6},
    {"label": "1.3 GHz (L-Band/Radar)",     "freq": 1.3e9},
    {"label": "2.4 GHz (S-Band/Milrem)",    "freq": 2.4e9},
    {"label": "3.4 GHz (C-Band/5G)",        "freq": 3.4e9},
    {"label": "4.9 GHz (Public Safety)",    "freq": 4.9e9},
    {"label": "5.2 GHz (C-Band/Drone)",     "freq": 5.2e9},
    {"label": "5.8 GHz (C-Band/Video)",     "freq": 5.8e9},
]

def get_baseband_freq(real_freq):
    return (real_freq - DISP_CENTER) * (DSP_FS / DISP_BW)


class RFAntiJamSubscriberGUI(gr.top_block, Qt.QWidget):
    # PyQt Signals for thread-safe cross-thread GUI updates
    sync_received = QtCore.pyqtSignal(int, int, float)
    msg_received = QtCore.pyqtSignal(int, str, str)

    def __init__(self):
        gr.top_block.__init__(self, "rf_antijam_sub_wide", catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle("RF Anti-Jam Subscriber  |  0 - 6 GHz Spectrum")
        qtgui.util.check_set_qss()

        self.top_scroll_layout = Qt.QVBoxLayout()
        self.setLayout(self.top_scroll_layout)
        self.top_scroll = Qt.QScrollArea()
        self.top_scroll.setFrameStyle(Qt.QFrame.NoFrame)
        self.top_scroll_layout.addWidget(self.top_scroll)
        self.top_scroll.setWidgetResizable(True)
        self.top_widget = Qt.QWidget()
        self.top_scroll.setWidget(self.top_widget)
        self.top_layout = Qt.QVBoxLayout(self.top_widget)

        # Variables
        self.sig_idx = 3
        self.jam_idx = 3
        self.j_power = 0.0

        # Narrower filters
        self.tx_taps = firdes.low_pass(1.0, DSP_FS, 4000, 2000, window.WIN_BLACKMAN, 6.76)
        self.j_taps = firdes.low_pass(1.0, DSP_FS, 6000, 2000, window.WIN_BLACKMAN, 6.76)

        # GUI: Jammer power (synced visually)
        self._j_power_range = qtgui.Range(0, 10, 100e-3, 0, 200)
        self._j_power_win = qtgui.RangeWidget(self._j_power_range, self.set_j_power,
            "Jammer Power (Synced from Publisher)", "counter_slider", float, QtCore.Qt.Horizontal)
        self._j_power_win.setEnabled(False) # Read-only
        self.top_layout.addWidget(self._j_power_win)

        # GUI: Waterfall sink
        self.qtgui_waterfall_sink_x_0 = qtgui.waterfall_sink_c(
            1024, window.WIN_BLACKMAN_hARRIS, DISP_CENTER, DISP_BW, "", 1, None)
        self.qtgui_waterfall_sink_x_0.set_update_time(0.10)
        self.qtgui_waterfall_sink_x_0.enable_grid(True)
        self.qtgui_waterfall_sink_x_0.set_intensity_range(-140, 10)
        self.qtgui_waterfall_sink_x_0.set_line_label(0, "Wideband RF Signal")
        self._qtgui_waterfall_sink_x_0_win = sip.wrapinstance(self.qtgui_waterfall_sink_x_0.qwidget(), Qt.QWidget)
        self.top_layout.addWidget(self._qtgui_waterfall_sink_x_0_win)

        # GUI: Frequency sink
        self.qtgui_freq_sink_x_0 = qtgui.freq_sink_c(
            1024, window.WIN_BLACKMAN_hARRIS, DISP_CENTER, DISP_BW, "", 1, None)
        self.qtgui_freq_sink_x_0.set_update_time(0.10)
        self.qtgui_freq_sink_x_0.set_y_axis(-140, 10)
        self.qtgui_freq_sink_x_0.enable_grid(True)
        self.qtgui_freq_sink_x_0.set_line_color(0, "blue")
        self._qtgui_freq_sink_x_0_win = sip.wrapinstance(self.qtgui_freq_sink_x_0.qwidget(), Qt.QWidget)
        self.top_layout.addWidget(self._qtgui_freq_sink_x_0_win)

        # Labels
        info_bar = Qt.QHBoxLayout()
        
        sig_group = Qt.QGroupBox("Current Signal Band")
        sig_layout = Qt.QHBoxLayout()
        self._sig_combo = Qt.QComboBox()
        for b in BANDS: self._sig_combo.addItem(b["label"])
        self._sig_combo.setCurrentIndex(self.sig_idx)
        self._sig_combo.setEnabled(False)
        sig_layout.addWidget(self._sig_combo)
        sig_group.setLayout(sig_layout)
        info_bar.addWidget(sig_group)

        jam_group = Qt.QGroupBox("Jammer Status (Synced)")
        jam_layout = Qt.QHBoxLayout()
        self._jammer_label = Qt.QLabel("Inactive")
        self._jammer_label.setAlignment(QtCore.Qt.AlignCenter)
        self._jammer_label.setStyleSheet("color: green; font-weight: bold;")
        jam_layout.addWidget(self._jammer_label)
        jam_group.setLayout(jam_layout)
        info_bar.addWidget(jam_group)
        
        self.top_layout.addLayout(info_bar)

        # Received Messages Log
        recv_group = Qt.QGroupBox("ROS2 Decrypted Messages (Received)")
        recv_layout = Qt.QVBoxLayout()
        self.recv_log = Qt.QTextEdit()
        self.recv_log.setReadOnly(True)
        self.recv_log.setMaximumHeight(150)
        recv_layout.addWidget(self.recv_log)
        recv_group.setLayout(recv_layout)
        self.top_layout.addWidget(recv_group)

        # Connect signals
        self.sync_received.connect(self._on_sync_received)
        self.msg_received.connect(self._on_msg_received)

        ##################################################
        # DSP Blocks
        ##################################################
        self.fft_filter_xxx_0_0 = filter.fft_filter_ccc(1, self.j_taps, 1)
        self.fft_filter_xxx_0 = filter.fft_filter_ccc(1, self.tx_taps, 1)

        self.blocks_throttle2_0 = blocks.throttle(gr.sizeof_float*1, SAMP_RATE, True, int(0.1 * SAMP_RATE))
        self.blocks_multiply_xx_0_0 = blocks.multiply_vcc(1)
        self.blocks_multiply_xx_0 = blocks.multiply_vcc(1)
        self.blocks_add_xx_0 = blocks.add_vcc(1)

        j_bb = get_baseband_freq(BANDS[self.jam_idx]["freq"])
        s_bb = get_baseband_freq(BANDS[self.sig_idx]["freq"])

        self.analog_sig_source_x_0_0_0 = analog.sig_source_c(DSP_FS, analog.GR_COS_WAVE, j_bb, 1, 0, 0)
        self.analog_sig_source_x_0_0 = analog.sig_source_c(DSP_FS, analog.GR_COS_WAVE, s_bb, 1, 0, 0)
        self.analog_sig_source_x_0 = analog.sig_source_f(SAMP_RATE, analog.GR_COS_WAVE, 150, 1, 0, 0)
        self.analog_noise_source_x_0_0 = analog.noise_source_c(analog.GR_GAUSSIAN, self.j_power, 0)
        self.analog_noise_source_x_0 = analog.noise_source_c(analog.GR_GAUSSIAN, 0.01, 0)
        self.analog_nbfm_tx_0 = analog.nbfm_tx(
            audio_rate=SAMP_RATE, quad_rate=DSP_FS, tau=(75e-6), max_dev=2e3, fh=(-1.0))

        # Connections
        self.connect((self.analog_nbfm_tx_0, 0), (self.fft_filter_xxx_0, 0))
        self.connect((self.analog_noise_source_x_0, 0), (self.blocks_add_xx_0, 1))
        self.connect((self.analog_noise_source_x_0_0, 0), (self.fft_filter_xxx_0_0, 0))
        self.connect((self.analog_sig_source_x_0, 0), (self.blocks_throttle2_0, 0))
        self.connect((self.analog_sig_source_x_0_0, 0), (self.blocks_multiply_xx_0, 0))
        self.connect((self.analog_sig_source_x_0_0_0, 0), (self.blocks_multiply_xx_0_0, 1))
        self.connect((self.blocks_add_xx_0, 0), (self.qtgui_freq_sink_x_0, 0))
        self.connect((self.blocks_add_xx_0, 0), (self.qtgui_waterfall_sink_x_0, 0))
        self.connect((self.blocks_multiply_xx_0, 0), (self.blocks_add_xx_0, 0))
        self.connect((self.blocks_multiply_xx_0_0, 0), (self.blocks_add_xx_0, 2))
        self.connect((self.blocks_throttle2_0, 0), (self.analog_nbfm_tx_0, 0))
        self.connect((self.fft_filter_xxx_0, 0), (self.blocks_multiply_xx_0, 1))
        self.connect((self.fft_filter_xxx_0_0, 0), (self.blocks_multiply_xx_0_0, 0))

    def _on_sync_received(self, sig_idx, jam_idx, jam_pwr):
        """Thread-safe slot called automatically by Qt via pub's 10Hz sync topic"""
        self.set_signal_band(sig_idx)
        self.set_jammer_band(jam_idx)
        
        # Don't call set_j_power directly to avoid Qt looping the slider value change
        self.j_power = jam_pwr
        self.analog_noise_source_x_0_0.set_amplitude(jam_pwr)

        self._update_jammer_label()

    def _on_msg_received(self, sig_idx, plaintext, encrypted_preview):
        b = BANDS[sig_idx]
        self.recv_log.append(f"[{b['label']}] RX: \"{plaintext}\" (enc: {encrypted_preview}...)")

    def set_signal_band(self, idx):
        self.sig_idx = idx
        self._sig_combo.setCurrentIndex(idx)
        bb = get_baseband_freq(BANDS[idx]["freq"])
        self.analog_sig_source_x_0_0.set_frequency(bb)

    def set_jammer_band(self, idx):
        self.jam_idx = idx
        bb = get_baseband_freq(BANDS[idx]["freq"])
        self.analog_sig_source_x_0_0_0.set_frequency(bb)
        self._update_jammer_label()

    def set_j_power(self, power):
        # Unused directly as we overrode the slider
        pass
        
    def _update_jammer_label(self):
        if self.j_power > 0:
            if self.jam_idx == self.sig_idx:
                b = BANDS[self.jam_idx]
                self._jammer_label.setText(f"JAMMED @ {b['label']}")
                self._jammer_label.setStyleSheet("color: red; font-size: 14px; font-weight: bold;")
            else:
                self._jammer_label.setText("NOT JAMMED (Evading)")
                self._jammer_label.setStyleSheet("color: green; font-size: 14px; font-weight: bold;")
        else:
            self._jammer_label.setText("Inactive")
            self._jammer_label.setStyleSheet("color: green; font-size: 14px; font-weight: bold;")

    def closeEvent(self, event):
        self.stop()
        self.wait()
        event.accept()


class EncryptedSubscriberNode(Node):
    def __init__(self, top_block):
        super().__init__('rf_antijam_sub')
        self.tb = top_block
        self.sync_sub = self.create_subscription(String, 'rf_sync_topic', self.sync_callback, 10)
        self.enc_sub = self.create_subscription(String, 'encrypted_rf_topic', self.enc_callback, 10)

    def sync_callback(self, msg):
        # Format: SYNC|S:xx|J:xx|P:xx
        try:
            parts = msg.data.split('|')
            sig_idx = int(parts[1].split(':')[1])
            jam_idx = int(parts[2].split(':')[1])
            jam_pwr = float(parts[3].split(':')[1])
            self.tb.sync_received.emit(sig_idx, jam_idx, jam_pwr)
        except Exception as e:
            self.get_logger().error(f"Sync error: {e}")

    def decrypt_message(self, encoded_message):
        try:
            combined_data = base64.b64decode(encoded_message.encode('utf-8'))
            iv = combined_data[:16]
            ciphertext = combined_data[16:]
            cipher = Cipher(algorithms.AES(SHARED_AES_KEY), modes.CBC(iv), backend=default_backend())
            decryptor = cipher.decryptor()
            padded_data = decryptor.update(ciphertext) + decryptor.finalize()
            unpadder = padding.PKCS7(algorithms.AES.block_size).unpadder()
            plaintext_bytes = unpadder.update(padded_data) + unpadder.finalize()
            return plaintext_bytes.decode('utf-8')
        except Exception as e:
            return None

    def enc_callback(self, msg):
        # Format: ENC|S:xx|encrypted_base64
        try:
            parts = msg.data.split('|')
            sig_idx = int(parts[1].split(':')[1])
            enc = parts[2]
            plaintext = self.decrypt_message(enc)
            if plaintext:
                self.tb.msg_received.emit(sig_idx, plaintext, enc[:20])
        except Exception as e:
            self.get_logger().error(f"Decode error: {e}")


def main():
    qapp = Qt.QApplication(sys.argv)
    rclpy.init(args=sys.argv)

    tb = RFAntiJamSubscriberGUI()
    ros_node = EncryptedSubscriberNode(tb)

    tb.start()
    tb.show()

    spin_thread = threading.Thread(target=rclpy.spin, args=(ros_node,), daemon=True)
    spin_thread.start()

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()
        ros_node.destroy_node()
        rclpy.shutdown()
        Qt.QApplication.quit()

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    timer = Qt.QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None)

    qapp.exec_()
    tb.stop()
    tb.wait()


if __name__ == '__main__':
    main()
