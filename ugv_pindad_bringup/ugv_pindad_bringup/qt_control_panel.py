#!/usr/bin/env python3
# ============================================================================
# UGV Pindad — PyQt5 Control Panel
# ============================================================================
# Graphical control panel for the UGV Pindad skid-steer vehicle.
#
# Features:
#   - Directional control buttons (Forward, Backward, Left, Right, Stop)
#   - Linear and angular speed sliders
#   - Real-time velocity display
#   - Keyboard capture (WASD keys work when window has focus)
#   - Emergency stop button (red, prominent)
#   - Dark military-themed UI
#
# Publishing:
#   Topic: /cmd_vel
#   Type:  geometry_msgs/msg/Twist
#
# Usage:
#   ros2 run ugv_pindad_bringup qt_control_panel
# ============================================================================

import sys
import threading

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QPushButton,
    QSlider,
    QLabel,
    QGroupBox,
    QFrame,
    QSizePolicy,
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QKeyEvent, QIcon


# ============================================================================
# Stylesheet — Dark military-themed UI
# ============================================================================
STYLESHEET = """
    /* ---- Main Window ---- */
    QMainWindow {
        background-color: #1a1a2e;
    }

    QWidget {
        background-color: #1a1a2e;
        color: #e0e0e0;
        font-family: 'Segoe UI', 'Roboto', sans-serif;
    }

    /* ---- Group Boxes ---- */
    QGroupBox {
        border: 2px solid #3a3a5e;
        border-radius: 8px;
        margin-top: 12px;
        padding-top: 16px;
        font-weight: bold;
        font-size: 13px;
        color: #8ecae6;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        left: 12px;
        padding: 0 6px;
    }

    /* ---- Direction Buttons ---- */
    QPushButton#btn_forward,
    QPushButton#btn_backward,
    QPushButton#btn_left,
    QPushButton#btn_right,
    QPushButton#btn_fwd_left,
    QPushButton#btn_fwd_right,
    QPushButton#btn_bwd_left,
    QPushButton#btn_bwd_right {
        background-color: #2d6a4f;
        color: white;
        border: 2px solid #40916c;
        border-radius: 8px;
        font-size: 18px;
        font-weight: bold;
        min-width: 70px;
        min-height: 50px;
        padding: 8px;
    }
    QPushButton#btn_forward:hover,
    QPushButton#btn_backward:hover,
    QPushButton#btn_left:hover,
    QPushButton#btn_right:hover,
    QPushButton#btn_fwd_left:hover,
    QPushButton#btn_fwd_right:hover,
    QPushButton#btn_bwd_left:hover,
    QPushButton#btn_bwd_right:hover {
        background-color: #40916c;
    }
    QPushButton#btn_forward:pressed,
    QPushButton#btn_backward:pressed,
    QPushButton#btn_left:pressed,
    QPushButton#btn_right:pressed,
    QPushButton#btn_fwd_left:pressed,
    QPushButton#btn_fwd_right:pressed,
    QPushButton#btn_bwd_left:pressed,
    QPushButton#btn_bwd_right:pressed {
        background-color: #52b788;
    }

    /* ---- Emergency Stop Button ---- */
    QPushButton#btn_stop {
        background-color: #c1121f;
        color: white;
        border: 3px solid #e5383b;
        border-radius: 10px;
        font-size: 20px;
        font-weight: bold;
        min-width: 70px;
        min-height: 50px;
        padding: 8px;
    }
    QPushButton#btn_stop:hover {
        background-color: #e5383b;
    }
    QPushButton#btn_stop:pressed {
        background-color: #ff6b6b;
    }

    /* ---- Sliders ---- */
    QSlider::groove:horizontal {
        height: 8px;
        background: #3a3a5e;
        border-radius: 4px;
    }
    QSlider::handle:horizontal {
        background: #8ecae6;
        width: 20px;
        margin: -6px 0;
        border-radius: 10px;
    }
    QSlider::sub-page:horizontal {
        background: #2d6a4f;
        border-radius: 4px;
    }

    /* ---- Labels ---- */
    QLabel {
        font-size: 12px;
    }
    QLabel#lbl_title {
        font-size: 20px;
        font-weight: bold;
        color: #8ecae6;
        padding: 8px;
    }
    QLabel#lbl_velocity {
        font-size: 14px;
        font-family: 'Courier New', monospace;
        color: #52b788;
        padding: 4px;
    }
    QLabel#lbl_status {
        font-size: 11px;
        color: #adb5bd;
        padding: 2px;
    }
"""


class QtControlPanel(QMainWindow):
    """
    PyQt5 graphical control panel for the UGV Pindad.

    Provides button-based and keyboard-based control with speed sliders.
    Publishes Twist messages to the diff_drive_controller.
    """

    def __init__(self, ros_node: Node):
        super().__init__()

        # Store reference to the ROS2 node for publishing
        self.ros_node = ros_node

        # ------------------------------------------------------------------
        # Current velocity state
        # ------------------------------------------------------------------
        self.linear_speed  = 1.0   # current linear speed setting (m/s)
        self.angular_speed = 1.0   # current angular speed setting (rad/s)
        self.current_linear  = 0.0  # active linear velocity
        self.current_angular = 0.0  # active angular velocity

        # ------------------------------------------------------------------
        # Window configuration
        # ------------------------------------------------------------------
        self.setWindowTitle("UGV Pindad — Control Panel")
        self.setMinimumSize(480, 600)
        self.setStyleSheet(STYLESHEET)

        # ------------------------------------------------------------------
        # Build the UI
        # ------------------------------------------------------------------
        self._build_ui()

        # ------------------------------------------------------------------
        # Continuous publishing timer (20 Hz)
        # ------------------------------------------------------------------
        # The diff_drive_controller has a cmd_vel_timeout, so we must
        # publish continuously to keep the robot moving.
        # ------------------------------------------------------------------
        self.publish_timer = QTimer()
        self.publish_timer.timeout.connect(self._publish_twist)
        self.publish_timer.start(50)  # 50ms = 20Hz

    # ======================================================================
    # UI CONSTRUCTION
    # ======================================================================

    def _build_ui(self):
        """Build the complete user interface."""

        # Central widget and main layout
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setSpacing(12)
        main_layout.setContentsMargins(16, 16, 16, 16)

        # ---- Title ----
        title = QLabel("⬡ UGV PINDAD CONTROL")
        title.setObjectName("lbl_title")
        title.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title)

        # ---- Direction Pad ----
        main_layout.addWidget(self._build_direction_pad())

        # ---- Speed Controls ----
        main_layout.addWidget(self._build_speed_controls())

        # ---- Velocity Display ----
        main_layout.addWidget(self._build_velocity_display())

        # ---- Status Bar ----
        status = QLabel("Keyboard: WASD/QE/ZC to move  |  SPACE = Stop  |  +/- = Speed")
        status.setObjectName("lbl_status")
        status.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(status)

    def _build_direction_pad(self) -> QGroupBox:
        """
        Build the directional control pad (3x3 grid of buttons).

        Layout:
            Q(↖)  W(↑)  E(↗)
            A(←)  STOP  D(→)
            Z(↙)  S(↓)  C(↘)
        """
        group = QGroupBox("DIRECTIONAL CONTROL")
        grid = QGridLayout()
        grid.setSpacing(6)

        # Button definitions: (row, col, text, objectName, linear, angular)
        buttons = [
            (0, 0, "↖ Q",  "btn_fwd_left",   1.0,  0.5),   # Forward-left arc
            (0, 1, "▲ W",  "btn_forward",     1.0,  0.0),   # Forward
            (0, 2, "↗ E",  "btn_fwd_right",   1.0, -0.5),   # Forward-right arc
            (1, 0, "◄ A",  "btn_left",        0.0,  1.0),   # Pivot left
            (1, 1, "■ STOP","btn_stop",        0.0,  0.0),   # Emergency stop
            (1, 2, "► D",  "btn_right",        0.0, -1.0),   # Pivot right
            (2, 0, "↙ Z",  "btn_bwd_left",   -1.0,  0.5),   # Backward-left arc
            (2, 1, "▼ S",  "btn_backward",    -1.0,  0.0),   # Backward
            (2, 2, "↘ C",  "btn_bwd_right",  -1.0, -0.5),   # Backward-right arc
        ]

        for row, col, text, name, linear, angular in buttons:
            btn = QPushButton(text)
            btn.setObjectName(name)
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

            # Connect button press/release for hold-to-drive behavior
            btn.pressed.connect(
                lambda lin=linear, ang=angular: self._on_direction(lin, ang)
            )
            btn.released.connect(self._on_stop)

            grid.addWidget(btn, row, col)

        group.setLayout(grid)
        return group

    def _build_speed_controls(self) -> QGroupBox:
        """
        Build linear and angular speed sliders.

        Each slider has a label showing the current value and
        allows the user to adjust the maximum speed for commands.
        """
        group = QGroupBox("SPEED SETTINGS")
        layout = QVBoxLayout()

        # ---- Linear Speed Slider ----
        lin_layout = QHBoxLayout()
        lin_label = QLabel("Linear (m/s):")
        lin_label.setFixedWidth(120)
        self.linear_slider = QSlider(Qt.Horizontal)
        self.linear_slider.setRange(1, 60)      # 0.1 to 6.0 m/s (x10) - Allows ~21.6 km/h
        self.linear_slider.setValue(20)           # default 2.0 m/s
        self.linear_slider.setTickInterval(5)
        self.linear_slider.setTickPosition(QSlider.TicksBelow)
        self.linear_value_label = QLabel("2.0")
        self.linear_value_label.setFixedWidth(40)
        self.linear_slider.valueChanged.connect(self._on_linear_slider)
        lin_layout.addWidget(lin_label)
        lin_layout.addWidget(self.linear_slider)
        lin_layout.addWidget(self.linear_value_label)
        layout.addLayout(lin_layout)

        # ---- Angular Speed Slider ----
        ang_layout = QHBoxLayout()
        ang_label = QLabel("Angular (rad/s):")
        ang_label.setFixedWidth(120)
        self.angular_slider = QSlider(Qt.Horizontal)
        self.angular_slider.setRange(1, 40)      # 0.1 to 4.0 rad/s (x10)
        self.angular_slider.setValue(15)           # default 1.5 rad/s
        self.angular_slider.setTickInterval(5)
        self.angular_slider.setTickPosition(QSlider.TicksBelow)
        self.angular_value_label = QLabel("1.5")
        self.angular_value_label.setFixedWidth(40)
        self.angular_slider.valueChanged.connect(self._on_angular_slider)
        ang_layout.addWidget(ang_label)
        ang_layout.addWidget(self.angular_slider)
        ang_layout.addWidget(self.angular_value_label)
        layout.addLayout(ang_layout)

        group.setLayout(layout)
        return group

    def _build_velocity_display(self) -> QGroupBox:
        """
        Build the real-time velocity readout display.

        Shows current linear.x and angular.z being published.
        """
        group = QGroupBox("LIVE VELOCITY")
        layout = QHBoxLayout()

        self.velocity_label = QLabel("Linear: +0.00 m/s  │  Angular: +0.00 rad/s")
        self.velocity_label.setObjectName("lbl_velocity")
        self.velocity_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.velocity_label)

        group.setLayout(layout)
        return group

    # ======================================================================
    # EVENT HANDLERS — Buttons
    # ======================================================================

    def _on_direction(self, linear_mult: float, angular_mult: float):
        """
        Handle directional button press.

        Applies the speed multipliers to get actual velocity values.

        Args:
            linear_mult:  -1.0 to 1.0 multiplier for linear speed.
            angular_mult: -1.0 to 1.0 multiplier for angular speed.
        """
        self.current_linear  = linear_mult  * self.linear_speed
        self.current_angular = angular_mult * self.angular_speed
        self._update_velocity_display()

    def _on_stop(self):
        """Handle button release or emergency stop — zero all velocities."""
        self.current_linear  = 0.0
        self.current_angular = 0.0
        self._update_velocity_display()

    # ======================================================================
    # EVENT HANDLERS — Sliders
    # ======================================================================

    def _on_linear_slider(self, value: int):
        """
        Update linear speed setting from slider.

        Slider range is 1–30, representing 0.1–3.0 m/s.
        """
        self.linear_speed = value / 10.0
        self.linear_value_label.setText(f"{self.linear_speed:.1f}")

    def _on_angular_slider(self, value: int):
        """
        Update angular speed setting from slider.

        Slider range is 1–20, representing 0.1–2.0 rad/s.
        """
        self.angular_speed = value / 10.0
        self.angular_value_label.setText(f"{self.angular_speed:.1f}")

    # ======================================================================
    # EVENT HANDLERS — Keyboard
    # ======================================================================

    def keyPressEvent(self, event: QKeyEvent):
        """
        Capture keyboard input when the Qt window has focus.

        Maps WASD/QEZC keys to directional commands, matching the
        terminal WASD teleop node bindings.
        """
        # Map Qt key codes to (linear_mult, angular_mult)
        key_map = {
            Qt.Key_W: ( 1.0,  0.0),   # Forward
            Qt.Key_S: (-1.0,  0.0),   # Backward
            Qt.Key_A: ( 0.0,  1.0),   # Pivot left
            Qt.Key_D: ( 0.0, -1.0),   # Pivot right
            Qt.Key_Q: ( 1.0,  0.5),   # Forward-left
            Qt.Key_E: ( 1.0, -0.5),   # Forward-right
            Qt.Key_Z: (-1.0,  0.5),   # Backward-left
            Qt.Key_C: (-1.0, -0.5),   # Backward-right
            Qt.Key_Space: (0.0, 0.0), # Emergency stop
        }

        if event.key() in key_map:
            lin, ang = key_map[event.key()]
            self.current_linear  = lin * self.linear_speed
            self.current_angular = ang * self.angular_speed
            self._update_velocity_display()

        elif event.key() in (Qt.Key_Plus, Qt.Key_Equal):
            # Increase speed via slider
            self.linear_slider.setValue(
                min(self.linear_slider.value() + 1, self.linear_slider.maximum())
            )
            self.angular_slider.setValue(
                min(self.angular_slider.value() + 1, self.angular_slider.maximum())
            )

        elif event.key() in (Qt.Key_Minus, Qt.Key_Underscore):
            # Decrease speed via slider
            self.linear_slider.setValue(
                max(self.linear_slider.value() - 1, self.linear_slider.minimum())
            )
            self.angular_slider.setValue(
                max(self.angular_slider.value() - 1, self.angular_slider.minimum())
            )

        else:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent):
        """
        Stop movement when a directional key is released.

        Implements hold-to-drive: robot moves only while key is held.
        Auto-repeat events are ignored to prevent stop-start stuttering.
        """
        # Ignore auto-repeat (held key generates repeated events)
        if event.isAutoRepeat():
            return

        stop_keys = {
            Qt.Key_W, Qt.Key_S, Qt.Key_A, Qt.Key_D,
            Qt.Key_Q, Qt.Key_E, Qt.Key_Z, Qt.Key_C,
        }

        if event.key() in stop_keys:
            self.current_linear  = 0.0
            self.current_angular = 0.0
            self._update_velocity_display()
        else:
            super().keyReleaseEvent(event)

    # ======================================================================
    # PUBLISHING
    # ======================================================================

    def _publish_twist(self):
        """
        Publish current velocity as a Twist message (called by QTimer at 20Hz).

        Continuous publishing is essential because the diff_drive_controller
        has a cmd_vel_timeout — if it doesn't receive commands, it stops.
        """
        twist = Twist()
        twist.linear.x  = self.current_linear
        twist.angular.z = self.current_angular
        self.ros_node.publisher.publish(twist)

    def _update_velocity_display(self):
        """Update the velocity readout label with current values."""
        self.velocity_label.setText(
            f"Linear: {self.current_linear:+.2f} m/s  │  "
            f"Angular: {self.current_angular:+.2f} rad/s"
        )

    # ======================================================================
    # CLEANUP
    # ======================================================================

    def closeEvent(self, event):
        """
        Handle window close — send stop command before exiting.

        Ensures the robot doesn't continue moving after the GUI is closed.
        """
        self.current_linear  = 0.0
        self.current_angular = 0.0
        self._publish_twist()
        self.ros_node.get_logger().info("Qt Control Panel closed. Robot stopped.")
        event.accept()


class QtControlNode(Node):
    """
    ROS2 node wrapper for the Qt Control Panel.

    Creates the Twist publisher used by the Qt GUI.
    """

    def __init__(self):
        super().__init__("qt_control_panel")

        # Create publisher for velocity commands
        self.publisher = self.create_publisher(
            Twist,
            "/cmd_vel",
            10,
        )

        self.get_logger().info("Qt Control Panel node initialized.")


def main(args=None):
    """
    Entry point for the Qt Control Panel.

    Initializes ROS2 in a background thread and runs the PyQt5
    application in the main thread (required by Qt).
    """
    # ------------------------------------------------------------------
    # Initialize ROS2
    # ------------------------------------------------------------------
    rclpy.init(args=args)
    node = QtControlNode()

    # ------------------------------------------------------------------
    # Run ROS2 spin in a background thread
    # ------------------------------------------------------------------
    # Qt requires the main thread for its event loop, so ROS2
    # spinning must happen in a separate daemon thread.
    # ------------------------------------------------------------------
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    # ------------------------------------------------------------------
    # Create and run the PyQt5 application
    # ------------------------------------------------------------------
    app = QApplication(sys.argv)
    app.setApplicationName("UGV Pindad Control Panel")

    window = QtControlPanel(node)
    window.show()

    # ------------------------------------------------------------------
    # Execute the Qt event loop (blocks until window is closed)
    # ------------------------------------------------------------------
    exit_code = app.exec_()

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    node.destroy_node()
    rclpy.shutdown()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
