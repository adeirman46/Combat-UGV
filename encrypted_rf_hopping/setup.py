import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'encrypted_rf_hopping'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*.launch.py'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='irman',
    maintainer_email='irman@local.host',
    description='AES-256 Encrypted ROS2 Pub/Sub with GNU Radio Anti-Jam Frequency Hopping',
    license='GPL-3.0',
    entry_points={
        'console_scripts': [
            'pub_gui = encrypted_rf_hopping.encrypted_rf_hopping_pub_gui:main',
            'sub_gui = encrypted_rf_hopping.encrypted_rf_hopping_sub_gui:main',
        ],
    },
)
