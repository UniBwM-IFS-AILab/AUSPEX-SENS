from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'auspex_perception'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Bjoern Doeschl',
    maintainer_email='bjoern.doeschl@unibw.de',
    description='A summarized perception package.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'image_processing_main_node = auspex_perception.image_processing_main:main',
        ],
    },
)
