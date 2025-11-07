from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'auspex_perception'

setup(
    name=package_name,
    version='1.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Kai Sommer',
    maintainer_email='kai.sommer@unibw.de',
    description='AUSPEX Perception Package.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'sens_node = auspex_perception.object_detector:main',
        ],
    },
)
