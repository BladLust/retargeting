#!/usr/bin/env python3
import numpy as np
import time
import cv2

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

from robot_teleoperation import RobotTeleoperation
from rviz_visualize import RvizVisualizer


class RobotTeleoperationRos(Node, RobotTeleoperation):
    """
    RobotTeleoperation with ROS interface.
    """

    def __init__(self, node_name: str):
        super().__init__(node_name)
        RobotTeleoperation.__init__(self, mujoco_vis=True)

        self.cv_bridge = CvBridge()
        self.camera_K = None
        self.rviz_visualizer = RvizVisualizer(node=self)

        self.get_logger().info("Waiting for image stream ...")
        self.image_sub = self.create_subscription(
            Image,
            "/camera/camera/color/image_raw",
            self.image_callback,
            1,
        )
        self.cam_info_sub = self.create_subscription(
            CameraInfo,
            "/camera/camera/color/camera_info",
            self.cam_info_callback,
            1,
        )

    def cam_info_callback(self, msg):
        if self.camera_K is None:
            self.camera_K = np.array(msg.k).reshape(3, 3)

    def image_callback(self, msg):
        # Convert ROS Image message to OpenCV image
        try:
            color_img = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"Could not convert image: {e}")
            return

        self.process(color_img)

    def process(self, color_img):
        hand_kps_in_wrist, wrist_pose, qpos = self.rgb_retarget(
            color_img, self.camera_K
        )
        if hand_kps_in_wrist is not None:
            # visualize the human hand in rviz
            self.rviz_visualizer.publish_hand_detection_results(
                hand_kps_in_wrist, wrist_pose, frame_id="world"
            )

            # visualize the robot hand in rviz
            joints_name = self.robot_model.joint_names
            qpos_dof = self.robot_adaptor.forward_qpos(qpos)
            self.rviz_visualizer.publish_robot_joint_states(
                joints_name=joints_name, joints_pos=qpos_dof
            )

        print(qpos)


def main():
    node_name = "robot_teleoperation"

    rclpy.init(args=None)
    teleoperation = RobotTeleoperationRos(node_name)
    rclpy.spin(teleoperation)

    # Cleanup on shutdown
    teleoperation.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
