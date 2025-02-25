#!/usr/bin/env python3
from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch_ros.actions import Node

def launch_setup(context, *args, **kwargs):
    pkg_name = "auspex_perception"

    launch_array =  []

    #for i in range(0,drone_count):
    executors = Node(
        package=pkg_name,
        executable="image_processing_main_node",
        namespace="",
        emulate_tty=True,
        output='screen',)
    launch_array.append(executors)
        
    return launch_array
    
def generate_launch_description():
    return LaunchDescription(
        [OpaqueFunction(function=launch_setup)]
    )
