import os
from setuptools import find_packages, setup
from glob import glob

package_name = 'gesture_hri_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',['resource/' + package_name]),
        ('share/' + package_name,['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*')),
        ('share/' + package_name + '/config', glob('config/*')),
        ('share/' + package_name + '/models', glob('models/*'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mgil',
    maintainer_email='mgil@iri.upc.edu',
    description='Hand gesture HRI ROS2 package',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'hand_gesture_recognition = gesture_hri_pkg.hand_gesture_recognition:main',
            'gesture_decision = gesture_hri_pkg.gesture_decision:main',
            'robot_controller = gesture_hri_pkg.robot_controller:main',
            'llm_client = gesture_hri_pkg.llm_client:main',
        ],
    },
)

