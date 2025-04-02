#!/usr/bin/env python3
import numpy as np
import rclpy
from franka_msgs.action import Grasp, Homing, Move
from rclpy.action import ActionServer
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray


class VirtualRobot(Node):
    def __init__(self):
        super().__init__("virtual_robot")
        # -------- hyper-parameters begin --------
        self.n_arm_joints = 7
        self.arm_joint_names = [f"panda_joint{i+1}" for i in range(self.n_arm_joints)]
        # -------- hyper-parameters end --------

        self.arm_joint_states_pub = self.create_publisher(JointState, "/franka/joint_states", 10)
        self.leap_joint_states_pub = self.create_publisher(JointState, "/leap_hand/joint_states", 10)

        self.arm_joint_pos_command_sub = self.create_subscription(
            Float64MultiArray,
            "franka/joint_impedance_command",
            self._arm_joint_pos_command_callback,
            10,
            callback_group=ReentrantCallbackGroup(),
        )
        self.leap_joint_pos_command_sub = self.create_subscription(
            JointState,
            "/cmd_leap",
            self._leap_joint_pos_command_callback,
            10,
            callback_group=ReentrantCallbackGroup(),
        )

        self.get_logger().info("Virtual robot node ready.")

    def _arm_joint_pos_command_callback(self, msg):
        joint_pos = msg.data
        joint_states = JointState()
        joint_states.name = self.arm_joint_names
        joint_states.position = joint_pos
        self.arm_joint_states_pub.publish(joint_states)

    def _leap_joint_pos_command_callback(self, msg):
        msg.name = [f"joint_{i}" for i in range(16)]
        self.leap_joint_states_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)

    robot = VirtualRobot()

    rclpy.spin(robot)


if __name__ == "__main__":
    main()
