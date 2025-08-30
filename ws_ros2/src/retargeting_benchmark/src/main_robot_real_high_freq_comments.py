#!/usr/bin/env python3
import threading
import time
from typing import List, Optional, Union

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration, Time
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.node import Node
from scipy.interpolate import CubicSpline
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, String

# Assuming these are custom utility functions for time conversion
from utils.utils_ros import stamp_to_seconds, time_to_seconds


class RobotRealHighFreq(Node):
    """
    This node acts as a high-frequency interpolator for robot joint commands.
    It subscribes to low-frequency target joint states and generates a smooth,
    high-frequency stream of commands using cubic spline interpolation. This is
    essential for commanding real robot hardware to move smoothly between keyframes.
    """

    def __init__(self):
        """Initializes the node, parameters, publishers, subscribers, and timer."""
        super().__init__("robot_real_high_freq")
        # -------- hyper-parameters begin --------
        # Number of joints for the robot arm.
        self.n_arm_joints = 7
        # Number of joints for the robot hand.
        self.n_hand_joints = 16
        # The number of past commands to use for spline interpolation is not used.
        self.window_size = 4
        # The number of recently *sent* high-frequency commands to store for smooth trajectory generation.
        self.hf_commands_window_size = 5
        # The period for the high-frequency timer callback (0.01s = 100Hz).
        self.timer_period = 0.01  # seconds
        # Flag to enable/disable visualization messages for RViz.
        self.rviz_viz = False
        # -------- hyper-parameters end --------

        # If visualization is enabled, create a publisher for the combined robot joint state.
        if self.rviz_viz:
            self.joint_names = [f"panda_joint{i + 1}" for i in range(7)] + [f"joint_{i}" for i in range(16)]
            self.robot_joint_state_vis_pub = self.create_publisher(JointState, "visualize/robot_joint_states", 10)

        # A sliding window to store the last few high-frequency commands that were sent.
        self.sent_hf_commands_window: List[np.ndarray] = []
        # A corresponding window to store the timestamps of the sent commands.
        self.sent_hf_commands_stamp_window: List[float] = []
        # This will hold the entire interpolated high-frequency trajectory to be executed.
        self.joint_pos_command_high_freq: Optional[np.ndarray] = None
        # A counter to track which command in the trajectory is being sent next.
        self.sent_num_command_high_freq: int = 0

        # A threading lock to prevent race conditions between the subscriber and timer callbacks
        # when accessing shared data like the high-frequency command trajectory.
        self.lock = threading.Lock()

        # Publisher for the robot arm's high-frequency joint commands.
        self.arm_joint_pos_command_pub = self.create_publisher(Float64MultiArray, "franka/joint_impedance_command", 10)
        # Publisher for the robot hand's high-frequency joint commands.
        self.hand_joint_pos_command_pub = self.create_publisher(JointState, "/cmd_leap", 10)

        # Subscriber to the low-frequency target joint positions.
        # A ReentrantCallbackGroup allows this callback and the timer callback to run concurrently.
        self.joint_pos_command_sub = self.create_subscription(
            JointState,
            "robot/joint_pos_command_low_freq",
            self._robot_command_callback,
            10,
            callback_group=ReentrantCallbackGroup(),
        )

        # The high-frequency timer that will publish interpolated commands.
        self.timer = self.create_timer(self.timer_period, self._timer_callback, callback_group=ReentrantCallbackGroup())
        self.get_logger().info("Ready ...")

    def _publish_arm_joint_pos_command(self, joint_pos: np.ndarray):
        """Helper function to publish a command to the robot arm."""
        assert len(joint_pos) == self.n_arm_joints
        msg = Float64MultiArray()
        msg.data = [float(i) for i in joint_pos]
        self.arm_joint_pos_command_pub.publish(msg)

    def _publish_hand_joint_pos_command(self, joint_pos: np.ndarray):
        """Helper function to publish a command to the robot hand."""
        assert len(joint_pos) == self.n_hand_joints
        msg = JointState()
        msg.position = joint_pos.tolist()
        self.hand_joint_pos_command_pub.publish(msg)

    def _robot_command_callback(self, msg: JointState):
        """
        Callback for low-frequency commands. Generates a smooth high-frequency
        trajectory to the new target using cubic spline interpolation.
        """
        self.get_logger().info(f"Received low-freqency joint command: {list(msg.position)}")

        current_time = time_to_seconds(self.get_clock().now())
        target_command = np.asarray(msg.position)
        # The timestamp in the message header indicates when the robot should *reach* the target.
        target_reach_time = stamp_to_seconds(msg.header.stamp)

        # If we have a history of recently sent commands, use them to create a smooth spline.
        if len(self.sent_hf_commands_window) >= 2:
            with self.lock:  # Use 'with' for safer lock handling
                # The points for interpolation are the past commands and the new target command.
                xs = self.sent_hf_commands_stamp_window + [target_reach_time]
                ys = self.sent_hf_commands_window + [target_command]

            # Create the spline function. 'natural' bc_type avoids wild oscillations at the ends.
            spline_func = CubicSpline(xs, ys, bc_type="natural")
            # Generate the time points for the new high-frequency trajectory.
            x_new = np.arange(current_time, target_reach_time, self.timer_period)
            # Calculate the interpolated joint positions for the entire trajectory.
            y_new = spline_func(x_new)
        else:
            # If there's not enough history, just jump to the target (happens at the start).
            y_new = np.asarray([target_command])

        # Safely update the shared trajectory data.
        with self.lock:
            self.joint_pos_command_high_freq = y_new
            # Reset the counter to start sending from the beginning of the new trajectory.
            self.sent_num_command_high_freq = 0

    def _timer_callback(self):
        """
        High-frequency callback (100Hz). Sends one command from the
        pre-computed trajectory at each tick.
        """
        with self.lock:
            # Check if there is a trajectory to execute and if we haven't sent all of its points yet.
            if (self.joint_pos_command_high_freq is not None) and self.sent_num_command_high_freq < len(
                self.joint_pos_command_high_freq
            ):
                # Get the next command from the trajectory.
                command = self.joint_pos_command_high_freq[self.sent_num_command_high_freq]

                # Split the command into arm and hand parts and publish them.
                self._publish_arm_joint_pos_command(command[: self.n_arm_joints])
                self._publish_hand_joint_pos_command(command[self.n_arm_joints :])

                # Increment the counter for the next tick.
                self.sent_num_command_high_freq += 1

                # --- Maintain the sliding window of sent commands ---
                # This window is used by the *next* low-frequency callback to ensure smooth transitions.
                self.sent_hf_commands_window.append(command)
                self.sent_hf_commands_stamp_window.append(time_to_seconds(self.get_clock().now()))
                if len(self.sent_hf_commands_window) > self.hf_commands_window_size:
                    self.sent_hf_commands_window.pop(0)
                    self.sent_hf_commands_stamp_window.pop(0)

                # If visualization is enabled, publish the current command to RViz.
                if self.rviz_viz:
                    msg = JointState()
                    msg.header.stamp = self.get_clock().now().to_msg()
                    msg.name = self.joint_names
                    msg.position = list(command)
                    self.robot_joint_state_vis_pub.publish(msg)


def main(args=None):
    """Standard ROS 2 entry point."""
    rclpy.init(args=args)

    robot = RobotRealHighFreq()

    # Spin the node to process callbacks.
    rclpy.spin(robot)

    # Cleanly destroy the node and shut down rclpy.
    robot.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
