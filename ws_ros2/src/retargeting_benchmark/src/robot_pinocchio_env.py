import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


class RobotPinocchioEnv:
    """
    An environment class based on RobotPinocchio.
    This class does not perform physical simulation but directly updates the state
    based on the set joint angles and optionally publishes joint_states via ROS for visualization in RViz.
    """

    def __init__(self, robot_model, node: Node = None):
        """
        Args:
            robot_model: A RobotPinocchio instance used for kinematic calculations.
            node: A ROS node (optional) used to publish joint_states messages.
        """
        self.robot_model = robot_model  # Depends on RobotPinocchio for kinematic calculations
        # Assumes robot_model provides a joint_names attribute; otherwise, it needs to be defined manually
        self.n_joints = len(robot_model.joint_names) if hasattr(robot_model, "joint_names") else 7
        self.timestep = 0.05  # Simulation timestep

        # Current joint state and target joint state
        self.current_joint_pos = np.zeros((self.n_joints))
        self.target_joint_pos = np.zeros((self.n_joints))
        self.one_step_time_record = time.time()

        # If a ROS node is provided, create a joint_states publisher
        self.node = node
        if self.node is not None:
            self.joint_state_pub = self.node.create_publisher(JointState, "joint_states", 10)
        else:
            self.joint_state_pub = None

        self.initial_robot_config()

    def initial_robot_config(self):
        """
        Set the initial state of the robot.
        """
        # Example initial state: assign specific angles to the first 7 joints, keep others at 0
        default_pos = np.zeros((self.n_joints))
        if self.n_joints >= 7:
            default_pos[:7] = np.array([0, -np.pi / 4, 0.0, -3.0 / 4.0 * np.pi, 0, np.pi / 2.0, 1.0 / 4.0 * np.pi])
        self.set_joint_pos(default_pos)

    def step(self):
        """
        Simulate one step.
        Simply updates the current state to the target state, publishes joint_states,
        and waits for one timestep.
        """
        self.current_joint_pos = self.target_joint_pos.copy()
        self.publish_joint_state()

        while (time.time() - self.one_step_time_record) < self.timestep:
            time.sleep(0.001)
        self.one_step_time_record = time.time()

    def set_joint_pos(self, qpos):
        """
        Forcefully set joint angles (ignoring physical dynamics) and update the target state.
        """
        if len(qpos) != self.n_joints:
            raise ValueError("The length of qpos must match n_joints")
        self.target_joint_pos = np.array(qpos).copy()
        self.current_joint_pos = np.array(qpos).copy()
        self.publish_joint_state()

    def get_joint_pos(self, update=True):
        """
        Return the current joint angles.
        """
        return self.current_joint_pos.copy()

    def get_target_joint_pos(self):
        """
        Return the target joint angles.
        """
        return self.target_joint_pos.copy()

    def ctrl_joint_pos(self, target_joint_pos):
        """
        Control the robot to reach the target joint angles (directly sets them here).
        """
        if len(target_joint_pos) != self.n_joints:
            raise ValueError("The length of target_joint_pos must match n_joints")
        self.set_joint_pos(target_joint_pos)

    def publish_joint_state(self):
        """
        If a ROS node is available, publish the current joint state to the 'joint_states' topic
        for visualizing the robot state in RViz.
        """
        if self.joint_state_pub is None:
            return
        msg = JointState()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        # Use joint_names provided by robot_model; if not available, generate default names
        if hasattr(self.robot_model, "joint_names"):
            msg.name = self.robot_model.joint_names
        else:
            msg.name = [f"joint_{i}" for i in range(self.n_joints)]
        msg.position = self.current_joint_pos.tolist()
        self.joint_state_pub.publish(msg)
