#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String


class TopicRateController(Node):
    def __init__(self):
        super().__init__("franka_joint_state_freq_reduce")

        # Create a subscription to the incoming topic (1000 Hz)
        self.subscription = self.create_subscription(
            JointState, "/franka/joint_states", self.listener_callback, 10  # The topic to receive messages from
        )

        # Create a publisher for the same message (30 Hz)
        self.publisher = self.create_publisher(JointState, "/franka/low_freq_joint_states", 10)

        # Create a timer to publish the message at 20 Hz
        self.timer = self.create_timer(1.0 / 20.0, self.publish_message)

        # Store the last received message
        self.last_received_message = None

    def listener_callback(self, msg):
        # Store the most recent message received from the 1000 Hz topic
        self.last_received_message = msg
        # self.get_logger().info(f"Received message: {msg.data}")

    def publish_message(self):
        if self.last_received_message is not None:
            self.publisher.publish(self.last_received_message)
            # self.get_logger().debug("Published low-frequency franka arm joint states in 20 Hz")


def main(args=None):
    rclpy.init(args=args)

    # Create the node
    node = TopicRateController()

    # Spin the node
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
