#!/usr/bin/python
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command
from launch_ros.actions import Node
from launch_ros.descriptions import ParameterValue


def generate_launch_description():

    realsense_launch_path = os.path.join(get_package_share_directory("realsense2_camera"), "launch", "rs_launch.py")

    return LaunchDescription(
        [
            Node(
                package="retargeting_benchmark",
                executable="main_franka_reduce_state_freq.py",
                name="franka_joint_states_freq_reduce",
                output="screen",
            ),
            Node(
                package="retargeting_benchmark",
                executable="main_robot_real_high_freq.py",
                name="robot_command_high_freq_interpolater",
                output="screen",
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(realsense_launch_path),
                launch_arguments={
                    "rgb_camera.color_profile": "1280x720x30",
                    "camera_name": "camera1",
                    "camera_namespace": "camera1",
                    "serial_no": "'238222076818'",
                    "enable_infra1": "false",
                    "enable_infra2": "false",
                    "enable_emitter": "false",
                    "depth_module.emitter_enabled": "false",
                    "enable_depth": "false",
                    # "rgb_camera.enable_auto_exposure": "false",
                    # "rgb_camera.exposure": "20000",
                }.items(),
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(realsense_launch_path),
                launch_arguments={
                    "rgb_camera.color_profile": "1280x720x30",
                    "camera_name": "camera2",
                    "camera_namespace": "camera2",
                    "serial_no": "'238222076841'",
                    "enable_infra1": "false",
                    "enable_infra2": "false",
                    "enable_emitter": "false",
                    "depth_module.emitter_enabled": "false",
                    "enable_depth": "false",
                    # "rgb_camera.enable_auto_exposure": "false",
                    # "rgb_camera.exposure": "20000",
                }.items(),
            ),
            Node(
                package="retargeting_benchmark",
                executable="record_video.py",
                name="record_video",
                output="screen",
            ),
        ]
    )
