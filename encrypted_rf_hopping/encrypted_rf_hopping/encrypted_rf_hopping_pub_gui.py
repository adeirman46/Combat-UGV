#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wideband RF Anti-Jam Encrypted Frequency Hopping Publisher
- Mapped 0 to 6 GHz Spectrum (visually displays 433 MHz, 2.4 GHz, 5.2 GHz, etc.)
- Auto-tracking jammer with 2.5 second delay (realistic hostile tracking)
- Instant continuous ROS2 state sync (jammer power and position match on Sub)
"""
import sys
import signal
import random
import os
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
    # Maps real-world GHz frequency to DSP baseband limits (+/- 160 kHz)
    return (real_freq - DISP_CENTER) * (DSP_FS / DISP_BW)


class RFAntiJamPublisherGUI(gr.top_block, Qt.QWidget):
    def __init__(self):
        gr.top_block.__init__(self, "rf_antijam_pub_wide", catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle("RF Anti-Jam Publisher  |  0 - 6 GHz Spectrum")
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
        self.sig_idx = 3     # Start at 2.4 GHz
        self.jam_idx = 3
        self.j_power = 0.0

        # Narrower filters so a single band looks distinct on a 6GHz wide screen
        self.tx_taps = firdes.low_pass(1.0, DSP_FS, 4000, 2000, window.WIN_BLACKMAN, 6.76)
        self.j_taps = firdes.low_pass(1.0, DSP_FS, 6000, 2000, window.WIN_BLACKMAN, 6.76)

        # GUI: Jammer power
        self._j_power_range = qtgui.Range(0, 10, 100e-3, 0, 200)
        self._j_power_win = qtgui.RangeWidget(self._j_power_range, self.set_j_power,
            "Jammer Power (Hostile Interdiction)", "counter_slider", float, QtCore.Qt.Horizontal)
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

        jam_group = Qt.QGroupBox("Hostile Jammer Tracking Status")
        jam_layout = Qt.QHBoxLayout()
        self._jammer_label = Qt.QLabel("Inactive")
        self._jammer_label.setAlignment(QtCore.Qt.AlignCenter)
        self._jammer_label.setStyleSheet("color: green; font-weight: bold;")
        jam_layout.addWidget(self._jammer_label)
        jam_group.setLayout(jam_layout)
        info_bar.addWidget(jam_group)
        
        self.top_layout.addLayout(info_bar)

        # Message Input
        msg_group = Qt.QGroupBox("ROS2 Encrypted Message (Anti-Jam Hopping)")
        msg_layout = Qt.QHBoxLayout()
        self.msg_input = Qt.QLineEdit()
        self.msg_input.setPlaceholderText("Type command, e.g. move=2,3,1  or  see_env()")
        self.msg_input.returnPressed.connect(self._on_send_clicked)
        self.send_btn = Qt.QPushButton("Transmit RF")
        self.send_btn.clicked.connect(self._on_send_clicked)
        msg_layout.addWidget(self.msg_input)
        msg_layout.addWidget(self.send_btn)
        msg_group.setLayout(msg_layout)
        self.top_layout.addWidget(msg_group)

        self.sent_log = Qt.QTextEdit()
        self.sent_log.setReadOnly(True)
        self.sent_log.setMaximumHeight(100)
        self.top_layout.addWidget(self.sent_log)

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

        self.ros_node = None

        # Jammer Tracking Timer (2-3 seconds as requested)
        self._jammer_track_timer = Qt.QTimer(self)
        self._jammer_track_timer.setSingleShot(True)
        self._jammer_track_timer.timeout.connect(self._execute_jammer_track)

        # 10 Hz continuous ROS sync timer (so Sub always matches Jammer slider)
        self._sync_timer = Qt.QTimer(self)
        self._sync_timer.timeout.connect(self._publish_sync)
        self._sync_timer.start(100)  # 100ms = 10Hz

    def _publish_sync(self):
        if self.ros_node:
            self.ros_node.publish_sync(self.sig_idx, self.jam_idx, self.j_power)

    def _execute_jammer_track(self):
        """Jammer locks onto the current signal band after delay"""
        if self.j_power > 0:
            self.set_jammer_band(self.sig_idx)

    def pick_antijam_band(self):
        options = [i for i in range(len(BANDS)) if i != self.jam_idx]
        return random.choice(options) if options else 0

    def _on_send_clicked(self):
        text = self.msg_input.text().strip()
        if text and self.ros_node:
            # 1. Pick new safe band
            new_idx = self.pick_antijam_band()
            self.set_signal_band(new_idx)

            # 2. Publish encrypted msg
            enc = self.ros_node.publish_msg(text, new_idx)
            
            # 3. Log it visually
            b = BANDS[new_idx]
            self.sent_log.append(f"[{b['label']}] TX: \"{text}\" -> {enc[:20]}...")
            self.msg_input.clear()

            # 4. Start jammer tracking delay (5000 ms = 5 seconds)
            self._jammer_track_timer.start(5000)

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
        self.j_power = power
        self.analog_noise_source_x_0_0.set_amplitude(power)
        self._update_jammer_label()
        
    def _update_jammer_label(self):
        if self.j_power > 0:
            if self.jam_idx == self.sig_idx:
                b = BANDS[self.jam_idx]
                self._jammer_label.setText(f"JAMMING @ {b['label']}")
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


class EncryptedPublisherNode(Node):
    def __init__(self):
        super().__init__('rf_antijam_pub')
        self.sync_pub = self.create_publisher(String, 'rf_sync_topic', 10)
        self.enc_pub = self.create_publisher(String, 'encrypted_rf_topic', 10)

    def publish_sync(self, sig_idx, jam_idx, jam_pwr):
        msg = String()
        msg.data = f"SYNC|S:{sig_idx}|J:{jam_idx}|P:{jam_pwr:.3f}"
        self.sync_pub.publish(msg)

    def encrypt_message(self, plaintext):
        iv = os.urandom(16)
        padder = padding.PKCS7(algorithms.AES.block_size).padder()
        padded_data = padder.update(plaintext.encode('utf-8')) + padder.finalize()
        cipher = Cipher(algorithms.AES(SHARED_AES_KEY), modes.CBC(iv), backend=default_backend())
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(padded_data) + encryptor.finalize()
        return base64.b64encode(iv + ciphertext).decode('utf-8')

    def publish_msg(self, plaintext, sig_idx):
        enc = self.encrypt_message(plaintext)
        msg = String()
        msg.data = f"ENC|S:{sig_idx}|{enc}"
        self.enc_pub.publish(msg)
        return enc


def main():
    qapp = Qt.QApplication(sys.argv)
    rclpy.init(args=sys.argv)

    tb = RFAntiJamPublisherGUI()
    ros_node = EncryptedPublisherNode()
    tb.ros_node = ros_node

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
