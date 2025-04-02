import os

import launch
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():

    realsense_launch_path = os.path.join(get_package_share_directory("realsense2_camera"), "launch", "rs_launch.py")

    return LaunchDescription(
        [
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(realsense_launch_path),
                launch_arguments={
                    "rgb_camera.color_profile": "1280x720x30",
                    # "rgb_camera.enable_auto_exposure": "false",
                    # "rgb_camera.exposure": "600",
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
