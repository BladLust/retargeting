#!/usr/bin/env python3
import os

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String


class RecordVideo(Node):
    def __init__(self):
        super().__init__("record_video")

        self.cv_bridge = CvBridge()
        self.img_1 = None
        self.img_2 = None
        self.b_recording = False
        self.b_start = False
        self.b_stop = False
        self.video_save_dir = ""
        self.video_output_1 = None
        self.video_output_2 = None

        self.fps = 10
        self.frame_size_1 = (1280, 720)  # must match the image size
        self.frame_size_2 = (1280, 720)
        self.codec = cv2.VideoWriter_fourcc(*"mp4v")

        # Subscribers
        self.create_subscription(Image, "/camera1/camera1/color/image_raw", self.image1Cb, 10)
        self.create_subscription(Image, "/camera2/camera2/color/image_raw", self.image2Cb, 10)
        self.create_subscription(String, "/record_video_command", self.videoRecordMsgCb, 10)

        self.timer = self.create_timer(0.1, self.timer_callback)

        self.get_logger().info("Record video node ready.")

    # --------------------------------------
    def image1Cb(self, msg):
        img = cv2.cvtColor(self.cv_bridge.imgmsg_to_cv2(msg), cv2.COLOR_RGB2BGR)
        self.img_1 = img.copy()

    # --------------------------------------
    def image2Cb(self, msg):
        img = cv2.cvtColor(self.cv_bridge.imgmsg_to_cv2(msg), cv2.COLOR_RGB2BGR)
        self.img_2 = img.copy()

    # --------------------------------------
    def videoRecordMsgCb(self, msg):
        if msg.data == "stop":
            if self.b_recording:
                self.b_stop = True
                self.b_recording = False

        elif msg.data != "":
            # can overwrite the previous non-stop stream
            self.b_start = True
            self.b_recording = True
            self.video_save_dir = msg.data

    def timer_callback(self):
        if self.b_start:
            try:
                self.video_output_1.release()
                self.video_output_2.release()
            except:
                pass

            self.get_logger().info("Start recording.")
            os.makedirs(self.video_save_dir, exist_ok=True)
            file_name_1 = os.path.join(self.video_save_dir, "camera1.mp4")
            file_name_2 = os.path.join(self.video_save_dir, "camera2.mp4")
            self.video_output_1 = cv2.VideoWriter(file_name_1, self.codec, self.fps, self.frame_size_1)
            self.video_output_2 = cv2.VideoWriter(file_name_2, self.codec, self.fps, self.frame_size_2)
            self.b_start = False

        if self.b_recording:
            # Check the image size
            if self.img_1 is not None and (
                self.img_1.shape[0] != self.frame_size_1[1] or self.img_1.shape[1] != self.frame_size_1[0]
            ):
                self.get_logger().error("Image 1 size does not match the frame size.")
            if self.img_2 is not None and (
                self.img_2.shape[0] != self.frame_size_2[1] or self.img_2.shape[1] != self.frame_size_2[0]
            ):
                self.get_logger().error("Image 2 size does not match the frame size.")

            if self.img_1 is not None:
                self.video_output_1.write(self.img_1)
            if self.img_2 is not None:
                self.video_output_2.write(self.img_2)

        if self.b_stop:
            self.video_output_1.release()
            self.video_output_2.release()
            self.b_stop = False
            self.get_logger().info("Finish recording.")


def main(args=None):
    rclpy.init(args=args)
    record_video = RecordVideo()
    rclpy.spin(record_video)
    record_video.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
