# ============================================================================
# UGV Pindad Bringup — Python Setup Configuration
# ============================================================================
# ament_python build configuration for the ugv_pindad_bringup package.
# Installs Python nodes, launch files, URDF, config, and mesh assets.
# ============================================================================

import os
from glob import glob
from setuptools import setup, find_packages

# --------------------------------------------------------------------------
# Package metadata
# --------------------------------------------------------------------------
PACKAGE_NAME = "ugv_pindad_bringup"

setup(
    name=PACKAGE_NAME,
    version="1.0.0",
    packages=find_packages(exclude=["test"]),

    # ------------------------------------------------------------------
    # Data files installed into the share directory
    # ------------------------------------------------------------------
    data_files=[
        # ament resource index (required for ROS2 package discovery)
        ("share/ament_index/resource_index/packages", ["resource/" + PACKAGE_NAME]),

        # Package manifest
        ("share/" + PACKAGE_NAME, ["package.xml"]),

        # Launch files
        ("share/" + PACKAGE_NAME + "/launch", glob("launch/*.launch.py")),

        # URDF / Xacro files
        ("share/" + PACKAGE_NAME + "/urdf", glob("urdf/*")),

        # Controller configuration
        ("share/" + PACKAGE_NAME + "/config", glob("config/*")),

        # Mesh files (STL)
        ("share/" + PACKAGE_NAME + "/meshes", glob("meshes/*")),

        # World files
        ("share/" + PACKAGE_NAME + "/worlds", glob("worlds/*")),
    ],

    # ------------------------------------------------------------------
    # Package metadata
    # ------------------------------------------------------------------
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Irman",
    maintainer_email="irman@ugv.dev",
    description="UGV Pindad skid-steer tank simulation with WASD + Qt GUI control",
    license="MIT",

    # ------------------------------------------------------------------
    # Console script entry points (ros2 run <pkg> <node>)
    # ------------------------------------------------------------------
    entry_points={
        "console_scripts": [
            # Terminal-based WASD keyboard teleop node
            "wasd_teleop_node = ugv_pindad_bringup.wasd_teleop_node:main",

            # PyQt5 graphical control panel
            "qt_control_panel = ugv_pindad_bringup.qt_control_panel:main",
        ],
    },
)
