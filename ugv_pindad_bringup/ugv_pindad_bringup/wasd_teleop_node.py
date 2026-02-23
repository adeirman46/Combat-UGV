#!/usr/bin/env python3
# ============================================================================
# UGV Pindad — WASD Keyboard Teleop Node
# ============================================================================
# Terminal-based keyboard teleoperation for the UGV Pindad.
# Uses raw terminal input (no external dependencies like pynput).
#
# Controls:
#   W / ↑    : Drive forward
#   S / ↓    : Drive backward
#   A / ←    : Pivot turn left  (right wheels fwd, left wheels back)
#   D / →    : Pivot turn right (left wheels fwd, right wheels back)
#   Q        : Arc turn forward-left
#   E        : Arc turn forward-right
#   Z        : Arc turn backward-left
#   C        : Arc turn backward-right
#   SPACE    : Emergency stop
#   + / =    : Increase speed
#   - / _    : Decrease speed
#   Ctrl+C   : Quit
#
# Publishing:
#   Topic: /cmd_vel
#   Type:  geometry_msgs/msg/Twist
# ============================================================================

import sys
import tty
import termios
import select
import signal

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


# --------------------------------------------------------------------------
# Key bindings configuration
# --------------------------------------------------------------------------
# Maps keyboard keys to (linear_x, angular_z) velocity multipliers.
# These multipliers are scaled by the current speed setting.
# --------------------------------------------------------------------------
KEY_BINDINGS = {
    # Basic directional controls
    "w": ( 1.0,  0.0),   # Forward
    "s": (-1.0,  0.0),   # Backward
    "a": ( 0.0,  1.0),   # Pivot left  (tank turn in place)
    "d": ( 0.0, -1.0),   # Pivot right (tank turn in place)

    # Diagonal / arc controls
    "q": ( 1.0,  0.5),   # Forward-left arc
    "e": ( 1.0, -0.5),   # Forward-right arc
    "z": (-1.0,  0.5),   # Backward-left arc
    "c": (-1.0, -0.5),   # Backward-right arc

    # Emergency stop
    " ": ( 0.0,  0.0),   # Space = full stop
}

# --------------------------------------------------------------------------
# Speed increment settings
# --------------------------------------------------------------------------
LINEAR_SPEED_STEP  = 0.1   # m/s per key press
ANGULAR_SPEED_STEP = 0.1   # rad/s per key press
MAX_LINEAR_SPEED   = 6.0   # maximum linear velocity (m/s) (~21.6 km/h)
MAX_ANGULAR_SPEED  = 4.0   # maximum angular velocity (rad/s)
MIN_SPEED          = 0.1   # minimum speed (m/s or rad/s)


# --------------------------------------------------------------------------
# Help message displayed on startup
# --------------------------------------------------------------------------
HELP_MESSAGE = """
╔══════════════════════════════════════════════════════════════╗
║             UGV PINDAD — WASD KEYBOARD TELEOP               ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║   Controls:        Q   W   E                                 ║
║                      \\ | /                                   ║
║                   A ←─ ■ ─→ D                                ║
║                      / | \\                                   ║
║                    Z   S   C                                 ║
║                                                              ║
║   W/S : Forward / Backward                                   ║
║   A/D : Pivot Left / Pivot Right (tank turn)                 ║
║   Q/E : Arc forward-left / forward-right                     ║
║   Z/C : Arc backward-left / backward-right                   ║
║                                                              ║
║   SPACE : Emergency Stop                                     ║
║   +/-   : Increase / Decrease speed                          ║
║   Ctrl+C: Quit                                               ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""


class WasdTeleopNode(Node):
    """
    ROS2 node for WASD keyboard teleoperation.

    Reads raw keyboard input from the terminal and publishes
    Twist messages to the diff_drive_controller.
    """

    def __init__(self):
        super().__init__("wasd_teleop_node")

        # ------------------------------------------------------------------
        # ROS2 publisher for velocity commands
        # ------------------------------------------------------------------
        self.publisher = self.create_publisher(
            Twist,
            "/cmd_vel",
            10,
        )

        # ------------------------------------------------------------------
        # Speed state — user can adjust with +/- keys
        # ------------------------------------------------------------------
        self.linear_speed  = 1.0   # current linear speed (m/s)
        self.angular_speed = 1.0   # current angular speed (rad/s)

        # ------------------------------------------------------------------
        # Terminal settings backup (restored on exit)
        # ------------------------------------------------------------------
        self.old_settings = termios.tcgetattr(sys.stdin)

        # ------------------------------------------------------------------
        # Timer for continuous command publishing (20 Hz)
        # ------------------------------------------------------------------
        self.current_twist = Twist()
        self.timer = self.create_timer(0.05, self._publish_callback)

        # ------------------------------------------------------------------
        # Timer for keyboard polling (50 Hz — responsive input)
        # ------------------------------------------------------------------
        self.key_timer = self.create_timer(0.02, self._key_poll_callback)

        self.get_logger().info("WASD Teleop Node started. Press keys to control.")

    def _publish_callback(self):
        """
        Periodically publish the current twist command.

        Publishing continuously (not just on key press) ensures the
        diff_drive_controller doesn't time out and stop the robot.
        """
        self.publisher.publish(self.current_twist)

    def _key_poll_callback(self):
        """
        Poll the terminal for keystrokes and update velocity commands.

        Uses select() for non-blocking reads so the ROS2 spin loop
        continues processing callbacks between key presses.
        """
        # Check if a key is available (non-blocking with 0.01s timeout)
        if select.select([sys.stdin], [], [], 0.01)[0]:
            key = sys.stdin.read(1).lower()
            self._process_key(key)

    def _process_key(self, key: str):
        """
        Process a single keystroke and update the velocity command.

        Args:
            key: Single character from keyboard input.
        """
        if key in KEY_BINDINGS:
            # ------------------------------------------------------------------
            # Directional key pressed — update twist with speed scaling
            # ------------------------------------------------------------------
            linear_mult, angular_mult = KEY_BINDINGS[key]
            self.current_twist.linear.x  = linear_mult  * self.linear_speed
            self.current_twist.angular.z = angular_mult * self.angular_speed

            # Log the current command for terminal feedback
            self._print_status()

        elif key in ("+", "="):
            # ------------------------------------------------------------------
            # Increase speed
            # ------------------------------------------------------------------
            self.linear_speed  = min(self.linear_speed  + LINEAR_SPEED_STEP,  MAX_LINEAR_SPEED)
            self.angular_speed = min(self.angular_speed + ANGULAR_SPEED_STEP, MAX_ANGULAR_SPEED)
            self._print_status()

        elif key in ("-", "_"):
            # ------------------------------------------------------------------
            # Decrease speed
            # ------------------------------------------------------------------
            self.linear_speed  = max(self.linear_speed  - LINEAR_SPEED_STEP,  MIN_SPEED)
            self.angular_speed = max(self.angular_speed - ANGULAR_SPEED_STEP, MIN_SPEED)
            self._print_status()

    def _print_status(self):
        """Print current velocity and speed settings to terminal."""
        print(
            f"\r  Linear: {self.current_twist.linear.x:+.2f} m/s  |  "
            f"Angular: {self.current_twist.angular.z:+.2f} rad/s  |  "
            f"Speed: {self.linear_speed:.1f}/{self.angular_speed:.1f}    ",
            end="",
            flush=True,
        )

    def start_raw_mode(self):
        """
        Switch terminal to raw mode for single-character input.

        In raw mode, keypresses are available immediately without
        waiting for Enter. Terminal settings are saved for restoration.
        """
        tty.setraw(sys.stdin.fileno())

    def restore_terminal(self):
        """
        Restore terminal to original settings.

        Called on shutdown to prevent leaving the terminal in raw mode,
        which would make it unusable.
        """
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)

    def stop(self):
        """Send a zero-velocity stop command before shutting down."""
        stop_twist = Twist()
        self.publisher.publish(stop_twist)
        self.get_logger().info("Robot stopped. Teleop node shutting down.")


def main(args=None):
    """
    Entry point for the WASD teleop node.

    Sets up raw terminal mode, spins the ROS2 node, and ensures
    clean terminal restoration on exit.
    """
    # Print help message before entering raw mode
    print(HELP_MESSAGE)

    # Initialize ROS2
    rclpy.init(args=args)
    node = WasdTeleopNode()

    try:
        # Switch to raw mode for immediate keypress detection
        node.start_raw_mode()

        # Spin the node (processes timers and callbacks)
        rclpy.spin(node)

    except KeyboardInterrupt:
        # Ctrl+C: graceful shutdown
        pass
    finally:
        # Always restore terminal and stop the robot
        node.restore_terminal()
        node.stop()
        node.destroy_node()
        rclpy.shutdown()
        print("\nTeleop node terminated cleanly.")


if __name__ == "__main__":
    main()
