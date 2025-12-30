from setuptools import setup
import os
from glob import glob

package_name = 'tvc_controller'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include launch files
        (os.path.join('share', package_name, 'launch'),
            glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
        # Include config files - ADD THIS LINE
        (os.path.join('share', package_name, 'config'), 
            glob(os.path.join('config', '*.yaml'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your.email@example.com',
    description='LQR Controller for PX4',
    license='MIT',
    entry_points={
        'console_scripts': [
            'lqr_px4_controller = tvc_controller.lqr_controller_node:main',
            'auto_takeoff_land = tvc_controller.px4_simple_trajectory:main',
        ],
    },
)