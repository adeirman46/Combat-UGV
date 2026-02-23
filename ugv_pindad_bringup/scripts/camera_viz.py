#!/usr/bin/env python3

import sys
import math
import struct
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QGridLayout, QFrame, QComboBox)
from PyQt5.QtCore import QTimer, Qt, pyqtSignal, QThread
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QFont, QPen

# --------------------------------------------------------------------------------
# DARK THEME STYLESHEET
# --------------------------------------------------------------------------------
DARK_STYLESHEET = """
QMainWindow {
    background-color: #1e1e1e;
}
QWidget {
    background-color: #1e1e1e;
    color: #ffffff;
    font-family: 'Segoe UI', sans-serif;
}
QLabel {
    color: #e0e0e0;
    font-size: 14px;
    font-weight: bold;
}
QFrame {
    border: 1px solid #333333;
    border-radius: 5px;
    background-color: #252526;
}
QComboBox {
    background-color: #333333;
    color: #ffffff;
    border: 1px solid #555555;
    padding: 5px;
}
"""

class ImageLabel(QLabel):
    def __init__(self, title):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setText(f"Waiting for {title}...")
        self.setMinimumSize(320, 240)
        self.setStyleSheet("background-color: #000000; border: 1px solid #444;")
        self.scaled_pixmap = None

    def update_image(self, qimage):
        if qimage is None: 
            return
        pixmap = QPixmap.fromImage(qimage)
        self.scaled_pixmap = pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(self.scaled_pixmap)

    def resizeEvent(self, event):
        if self.scaled_pixmap:
            self.setPixmap(self.scaled_pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        super().resizeEvent(event)

class LidarWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(400, 400)
        self.points = []
        self.scale = 15.0 # pixels per meter
        self.range_max = 20.0 # meters radius to draw

    def update_points(self, points):
        self.points = points
        self.update() # Trigger paintEvent

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Background
        painter.fillRect(self.rect(), QColor(0, 0, 0))
        
        w = self.width()
        h = self.height()
        cx, cy = w / 2, h / 2
        
        # Draw Grid (Circles)
        painter.setPen(QPen(QColor(0, 50, 0), 1))
        for r in range(5, int(self.range_max)+1, 5):
            radius = r * self.scale
            painter.drawEllipse(int(cx - radius), int(cy - radius), int(radius*2), int(radius*2))
            
        # Draw Crosshair
        painter.drawLine(int(cx-10), int(cy), int(cx+10), int(cy))
        painter.drawLine(int(cx), int(cy-10), int(cx), int(cy+10))
        
        # Draw Points
        painter.setPen(QPen(QColor(0, 255, 0), 2))
        for x, y in self.points:
            # ROS X+ -> Screen Up (cy - x*scale)
            # ROS Y+ -> Screen Left (cx - y*scale)
            px = cx - (y * self.scale)
            py = cy - (x * self.scale)
            
            if 0 <= px < w and 0 <= py < h:
                painter.drawPoint(int(px), int(py))
                
        # Draw Robot Marker
        painter.setBrush(QColor(255, 0, 0))
        painter.drawRect(int(cx-5), int(cy-8), 10, 16) # Roughly vehicle shape

class RosThread(QThread):
    # Signals to update GUI (Sending dictionary with 'rgb' and 'depth' QImages)
    update_front = pyqtSignal(dict)
    update_rear = pyqtSignal(dict)
    update_left = pyqtSignal(dict)
    update_right = pyqtSignal(dict)
    update_lidar = pyqtSignal(list)

    def __init__(self):
        super().__init__()
        rclpy.init(args=None)
        self.node = Node('camera_viz_node')
        
        # Use Best Effort QoS for sensor data
        from rclpy.qos import qos_profile_sensor_data

        # Store latest frames
        self.frames = {
            "Front": {"rgb": None, "depth": None},
            "Rear":  {"rgb": None, "depth": None},
            "Left":  {"rgb": None, "depth": None},
            "Right": {"rgb": None, "depth": None},
        }

        # Subscriptions for RGB (image_raw) and Depth (depth_image)
        # Using separate callbacks or single parameterized one?
        # We need to know if it's RGB or Depth.
        
        # Front
        self.node.create_subscription(Image, '/front_camera/rgb/image_raw', lambda m: self.image_callback(m, "Front", "rgb"), qos_profile_sensor_data)
        self.node.create_subscription(Image, '/front_camera/depth/image_raw', lambda m: self.image_callback(m, "Front", "depth"), qos_profile_sensor_data)

        # Rear
        self.node.create_subscription(Image, '/rear_camera/rgb/image_raw', lambda m: self.image_callback(m, "Rear", "rgb"), qos_profile_sensor_data)
        self.node.create_subscription(Image, '/rear_camera/depth/image_raw', lambda m: self.image_callback(m, "Rear", "depth"), qos_profile_sensor_data)

        # Left
        self.node.create_subscription(Image, '/left_camera/rgb/image_raw', lambda m: self.image_callback(m, "Left", "rgb"), qos_profile_sensor_data)
        self.node.create_subscription(Image, '/left_camera/depth/image_raw', lambda m: self.image_callback(m, "Left", "depth"), qos_profile_sensor_data)

        # Right
        self.node.create_subscription(Image, '/right_camera/rgb/image_raw', lambda m: self.image_callback(m, "Right", "rgb"), qos_profile_sensor_data)
        self.node.create_subscription(Image, '/right_camera/depth/image_raw', lambda m: self.image_callback(m, "Right", "depth"), qos_profile_sensor_data)

        self.node.create_subscription(PointCloud2, '/lidar/points', self.lidar_callback, qos_profile_sensor_data)

    def image_callback(self, msg, name, type_):
        try:
            qimg = self.decode_image(msg)
            if qimg:
                self.frames[name][type_] = qimg
                
                # Emit signal with BOTH current frames for this camera
                # (The GUI decides which to show)
                data = self.frames[name]
                
                if name == "Front": self.update_front.emit(data)
                elif name == "Rear": self.update_rear.emit(data)
                elif name == "Left": self.update_left.emit(data)
                elif name == "Right": self.update_right.emit(data)
                
                if not hasattr(self, f"{name}_{type_}_received"):
                    print(f"[DEBUG] Received first {type_} from {name}!", flush=True)
                    setattr(self, f"{name}_{type_}_received", True)
        except Exception as e:
            print(f"[ERROR] Decode error {name} {type_}: {e}", flush=True)

    def decode_image(self, msg):
        h, w = msg.height, msg.width
        
        if msg.encoding == 'rgb8' or msg.encoding == 'r8g8b8':
            data = np.frombuffer(msg.data, dtype=np.uint8).reshape((h, w, 3))
            bytes_per_line = 3 * w
            return QImage(data.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
            
        elif msg.encoding == 'bgr8':
            data = np.frombuffer(msg.data, dtype=np.uint8).reshape((h, w, 3))
            data = data[..., ::-1].copy() 
            bytes_per_line = 3 * w
            return QImage(data.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()

        elif msg.encoding == '32FC1':
            data = np.frombuffer(msg.data, dtype=np.float32).reshape((h, w))
            data = np.nan_to_num(data, copy=False, nan=0.0, posinf=10.0, neginf=0.0)
            max_val = 10.0
            data = np.clip(data / max_val, 0, 1) * 255
            data = data.astype(np.uint8)
            data = np.stack((data,)*3, axis=-1)
            bytes_per_line = 3 * w
            return QImage(data.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
            
        return None

    def lidar_callback(self, msg):
        try:
            data = np.frombuffer(msg.data, dtype=np.float32)
            floats_per_point = msg.point_step // 4
            num_points = msg.width * msg.height
            if len(data) >= num_points * floats_per_point:
                reshaped = data.reshape(-1, floats_per_point)
                xy = reshaped[::5, :2] 
                points = xy.tolist()
                self.update_lidar.emit(points)
        except:
            pass

    def run(self):
        try:
            rclpy.spin(self.node)
        except:
            pass

    def stop(self):
        self.node.destroy_node()
        rclpy.shutdown()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("UGV Pindad - ZED 360 Sensing & LiDAR Inference")
        self.setGeometry(100, 100, 1280, 800)
        self.setStyleSheet(DARK_STYLESHEET)

        # Main Widget
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # Left Column: Cameras
        left_col = QWidget()
        left_layout = QVBoxLayout(left_col)
        
        # Controls Row
        ctrl_layout = QHBoxLayout()
        ctrl_label = QLabel("Camera View Mode:")
        self.view_selector = QComboBox()
        self.view_selector.addItems(["Original Frame (RGB)", "Depth Frame"])
        self.view_selector.currentIndexChanged.connect(self.refresh_displays)
        ctrl_layout.addWidget(ctrl_label)
        ctrl_layout.addWidget(self.view_selector)
        ctrl_layout.addStretch()
        left_layout.addLayout(ctrl_layout)

        # Camera Grid
        cam_grid = QGridLayout()
        left_layout.addLayout(cam_grid)

        self.cam_labels = {}
        self.latest_data = {} # Store latest dicts here

        self.cam_labels["Front"] = self.create_cam_frame("Front ZED", cam_grid, 0, 0)
        self.cam_labels["Rear"] = self.create_cam_frame("Rear ZED", cam_grid, 0, 1)
        self.cam_labels["Left"] = self.create_cam_frame("Left ZED", cam_grid, 1, 0)
        self.cam_labels["Right"] = self.create_cam_frame("Right ZED", cam_grid, 1, 1)

        main_layout.addWidget(left_col, stretch=2)

        # Right Column: LiDAR
        lidar_frame = QFrame()
        lidar_layout = QVBoxLayout(lidar_frame)
        label_lidar = QLabel("Velodyne VLP-16 (Top-Down)")
        label_lidar.setAlignment(Qt.AlignCenter)
        lidar_layout.addWidget(label_lidar)
        
        self.lidar_widget = LidarWidget()
        lidar_layout.addWidget(self.lidar_widget)
        
        main_layout.addWidget(lidar_frame, stretch=1)

        # ROS Thread
        self.ros_thread = RosThread()
        self.ros_thread.update_front.connect(lambda d: self.update_cam("Front", d))
        self.ros_thread.update_rear.connect(lambda d: self.update_cam("Rear", d))
        self.ros_thread.update_left.connect(lambda d: self.update_cam("Left", d))
        self.ros_thread.update_right.connect(lambda d: self.update_cam("Right", d))
        self.ros_thread.update_lidar.connect(self.lidar_widget.update_points)
        self.ros_thread.start()

    def create_cam_frame(self, title, layout, r, c):
        frame = QFrame()
        l = QVBoxLayout(frame)
        lbl = QLabel(title)
        lbl.setAlignment(Qt.AlignCenter)
        l.addWidget(lbl)
        img_lbl = ImageLabel(title)
        l.addWidget(img_lbl)
        layout.addWidget(frame, r, c)
        return img_lbl

    def update_cam(self, name, data):
        self.latest_data[name] = data
        self.refresh_single(name)

    def refresh_single(self, name):
        if name not in self.latest_data: return
        
        mode = self.view_selector.currentText()
        key = "rgb" if "Original" in mode else "depth"
        
        qimg = self.latest_data[name].get(key)
        if qimg:
            self.cam_labels[name].update_image(qimg)

    def refresh_displays(self):
        for name in self.cam_labels:
            self.refresh_single(name)

    def closeEvent(self, event):
        self.ros_thread.stop()
        self.ros_thread.wait(500)
        event.accept()

if __name__ == '__main__':
    print("STARTING CAMERA VIZ NODE...", flush=True)
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
