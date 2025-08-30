#!/usr/bin/env python3
import copy
import json
import os
import sys
import time
from datetime import datetime
from threading import Thread
from typing import List, Union

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from robot_adaptor import RobotAdaptor
from robot_benchmark import RobotBenchmark
from robot_control import RobotControl
from robot_pinocchio import RobotPinocchio
from robot_real_rviz import RobotReal
from robot_teleoperation import RobotTeleoperation
from rviz_visualize import RvizVisualizer
from utils.utils_keyboard import KeyboardListener
from vision_pro_detector import VisionProDetector


def flatten_stream_data(data_dict):
    """
    Extract the elements in 'stream' dict to the overall dictp
    """
    new_data_dict = {}
    for step, stream in enumerate(data_dict["stream"]):
        for key, value in stream.items():
            new_key = f"stream_{key}"
            if new_key not in new_data_dict:
                new_data_dict[new_key] = []
            new_data_dict[new_key].append(value)

    for key, value in data_dict.items():
        if key != "stream":
            new_data_dict[key] = value

    return new_data_dict


def rebuild_stream_data(data_dict):
    """
    Rebuld the stream data as a dict named 'stream'
    """
    prefix = "stream_"
    n_steps = data_dict["stream_left_wrist"].shape[0]
    new_data_dict = {"stream": []}
    for i in range(n_steps):
        new_data_dict["stream"].append({})

    for key, value in data_dict.items():
        if key.startswith(prefix):
            remaining_key = key[len(prefix) :]
            for step, array in enumerate(data_dict[key]):
                new_data_dict["stream"][step][remaining_key] = copy.deepcopy(array)
        else:
            new_data_dict[key] = value

    return new_data_dict


class RobotTeleoperationMain:
    def __init__(self):
        # --------- hyper-parameters ---------
        self.hand_type = "leap"  # "leap" or "shadow"
        if self.hand_type == "leap":
            urdf_file_name = os.readlink("assets/panda_leap_paxini.urdf")  # no touch bodies/joints/sensors
            actuated_joints_name = [f"panda_joint{i + 1}" for i in range(7)] + [f"joint_{i}" for i in range(16)]
            self.max_joint_speed = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.5] + [1.0] * 16
        elif self.hand_type == "shadow":
            urdf_file_name = os.readlink("assets/panda_shadow.urdf")  # no touch bodies/joints/sensors
            actuated_joints_name = [f"panda_joint{i + 1}" for i in range(7)] + [
                "WRJ2",
                "WRJ1",
                "FFJ4",
                "FFJ3",
                "FFJ2",
                "FFJ1",
                "LFJ5",
                "LFJ4",
                "LFJ3",
                "LFJ2",
                "LFJ1",
                "MFJ4",
                "MFJ3",
                "MFJ2",
                "MFJ1",
                "RFJ4",
                "RFJ3",
                "RFJ2",
                "RFJ1",
                "THJ5",
                "THJ4",
                "THJ3",
                "THJ2",
                "THJ1",
            ]
            self.max_joint_speed = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.5] + [1.0] * 24
        touch_joints_name: List[str] = []

        self.vision_pro_ip = "192.168.52.6"
        # self.vision_pro_ip = "192.168.60.250"
        self.input_device = "vision_pro"
        # self.input_device = "rgb"
        self.load_offline_data = False
        self.use_hardware = False
        self.use_virtual_hardware = False
        self.use_high_freq_interp = False
        self.use_ros = True

        if self.use_ros:
            rclpy.init(args=None)
            self.node = Node("main_node")
            self.executor = MultiThreadedExecutor()
            self.executor.add_node(self.node)
            self.spin_thread = Thread(target=self.executor.spin, daemon=True)
            self.spin_thread.start()
            self.rviz_visualizer = RvizVisualizer(node=self.node)
        else:
            self.node = None

        self.robot_model = RobotPinocchio(
            robot_file_path=urdf_file_name,
            robot_file_type="urdf",
        )
        self.robot_adaptor = RobotAdaptor(
            robot_model=self.robot_model,
            actuated_joints_name=actuated_joints_name,
            touch_joints_name=touch_joints_name,
        )
        self.robot_control = RobotControl(
            self.robot_model,
            self.robot_adaptor,
            hand_type=self.hand_type,
            use_hardware=self.use_hardware,
            use_virtual_hardware=self.use_virtual_hardware,
            use_high_freq_interp=self.use_high_freq_interp,
            node=self.node,
        )
        self.robot_teleop = RobotTeleoperation(
            hand_type=self.hand_type,
            robot_adaptor=self.robot_adaptor,
            robot_control=self.robot_control,
            qpos_init=self.robot_control.init_joint_pos,
            input_device=self.input_device,
            mujoco_vis=False,
            use_real_hardware=self.use_hardware,
        )
        # self.robot_benchmark = RobotBenchmark(robot_adaptor=self.robot_adaptor)
        if self.input_device == "vision_pro":
            if not self.load_offline_data:
                self.robot_teleop.detector.connect(avp_ip=self.vision_pro_ip)
        else:
            raise NotImplementedError()

        # check the retargeting type
        print(self.robot_teleop.retarget_type)

        # for keyboard control
        self.keyboard_listener = KeyboardListener()
        self.keyboard_listener.start_keyboard_listening_thread()
        # for recording data
        self.data = {}
        self.data["stream"] = []
        self.data["retarget_qpos"] = []

    def save_data(self, save_dir):
        file = os.path.join(save_dir, "data.npz")
        data = flatten_stream_data(self.data)
        np.savez(file, **data)
        print(f"Save stream data to {file}.")

    def load_data(self, file_name):
        loaded_data = np.load(file_name)
        data_dict = {key: loaded_data[key] for key in loaded_data.files}  # Convert npz data back to a dictionary
        data = rebuild_stream_data(data_dict)
        return data

    def main(self):
        # ----------- hyper-parameters -----------
        project_dir = "/home/timothy/retarget_benchmark-main"
        save_dir = os.path.join(project_dir, f"data/teleop_process/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
        os.makedirs(save_dir, exist_ok=True)
        if self.load_offline_data:
            file_name = os.path.join(project_dir, "data/test_teleop/vision_pro/data_2025-03-20_21-35-42.npz")
            data_dict = self.load_data(file_name)
            stream_data = data_dict["stream"]

        i = 0
        total_position_err = 0
        total_orientation_err = 0
        total_relative_position_err = 0
        total_relative_position_to_wrist_err = 0
        total_time_cost = 0

        position_err_list = []
        orientation_err_list = []
        relative_position_err_list = []
        relative_position_to_wrist_err_list = []
        time_cost_list = []

        # move to initial configuration
        self.robot_control.move_to_joint_pos(self.robot_control.init_joint_pos, max_joint_speed=self.max_joint_speed)

        if self.use_hardware and not self.use_virtual_hardware:
            self.robot_control.env.start_record_video(data_dir=save_dir)

        # i_start = 217
        # i_end = 250
        i_start = 0
        if self.load_offline_data:
            i_end = len(stream_data) - 1
        while True:
            t_frame_start = time.time()
            print(f"Frame {i}:")

            if self.load_offline_data:
                if i > i_end:
                    break
                if i < i_start:
                    i += 1
                    continue

            # -------- get human motion --------
            if self.load_offline_data:
                r = stream_data[i]
                # print("right_wrist: ", r["right_wrist"])
            # elif self.input_device == "rgb":
            #     r = self.robot_teleop.detector.get_rgb_stream()
            else:
                r = self.robot_teleop.detector.get_raw_stream()

            # print(f"Frame time cost 1: {(time.time() - t_frame_start):.3f}")

            # -------- retargeting --------
            if i == i_start:  # set initial poses
                init_joint_pos = self.robot_control.get_joint_pos(update=True)
                init_tcp_pose = self.robot_model.get_frame_pose(
                    "wrist", qpos=self.robot_adaptor.forward_qpos(init_joint_pos)
                )
                self.robot_teleop.set_robot_init_wrist_pose(init_tcp_pose)
                _, _, _, wrist_pose = self.robot_teleop.detector.detect(r)
                wrist_pose = self.robot_teleop.pose_from_avp_world_to_robot_world(wrist_pose)
                self.robot_teleop.set_avp_init_wrist_pose(wrist_pose)

            hand_kps_in_wrist, wrist_pose, qpos, err = self.robot_teleop.vision_pro_retarget(stream=r)
            # hand_kps_in_wrist, wrist_pose, qpos, err = self.robot_teleop.rgb_retarget()

            # print(f"Frame time cost 2: {(time.time() - t_frame_start):.3f}")

            # -------- control robot --------
            if self.use_hardware and i < i_start + 10:
                self.robot_control.move_to_joint_pos(qpos, max_joint_speed=self.max_joint_speed)
                print("Slowly move to the first 10 retargeted configuration.")
            else:
                self.robot_control.ctrl_joint_pos(qpos)

            self.robot_control.step()

            position_err_list.append(err["position_err"])
            orientation_err_list.append(err["orientation_err"])
            relative_position_err_list.append(err["relative_position_err"])
            relative_position_to_wrist_err_list.append(err["relative_position_to_wrist_err"])
            time_cost_list.append(err["optimizaion_time"])

            total_position_err += err["position_err"]
            total_orientation_err += err["orientation_err"]
            total_relative_position_err += err["relative_position_err"]
            total_relative_position_to_wrist_err += err["relative_position_to_wrist_err"]
            total_time_cost += err["optimizaion_time"]

            # print(f"Frame time cost 3: {(time.time() - t_frame_start):.3f}")

            # -------- visualization --------
            if hand_kps_in_wrist is not None:
                # visualize the human hand in rviz
                self.rviz_visualizer.publish_hand_detection_results(
                    hand_kps_in_wrist, wrist_pose, frame_id="visualize/world"
                )
                # visualize the robot hand in rviz
                joints_name = self.robot_model.joint_names
                qpos_dof = self.robot_adaptor.forward_qpos(qpos)
                self.rviz_visualizer.publish_robot_joint_states(joints_name=joints_name, joints_pos=qpos_dof)

            # print(f"Frame time cost 4: {(time.time() - t_frame_start):.3f}")

            # -------- record data --------
            self.data["stream"].append(r)
            self.data["retarget_qpos"].append(qpos)

            t_frame_end = time.time()
            print(f"Frame total time cost: {(t_frame_end - t_frame_start):.3f}")
            i += 1

            # quit loop criterian
            if "p" in self.keyboard_listener.pressed_keys:
                self.save_data(save_dir)
                if self.use_hardware and not self.use_virtual_hardware:
                    self.robot_control.env.stop_record_video()
                break

            # return to the initial configuration and re-start
            if self.load_offline_data:
                if i >= len(stream_data):
                    # i = 0
                    # self.robot_control.move_to_joint_pos(
                    #     self.robot_control.init_joint_pos, max_joint_speed=self.max_joint_speed
                    # )
                    self.save_data(save_dir)
                    if self.use_hardware and not self.use_virtual_hardware:
                        self.robot_control.env.stop_record_video()
                    break
        # --------------------------------- end loop ---------------------------------

        # save quantitative results
        if self.load_offline_data:
            print("---------------------------------------")
            print("average_position_err: ", total_position_err / len(stream_data))
            print("average_orientation_err: ", total_orientation_err / len(stream_data))
            print("average_relative_position_err: ", total_relative_position_err / len(stream_data))
            print("average_relative_position_to_wrist_err: ", total_relative_position_to_wrist_err / len(stream_data))
            print("average_time_cost: ", total_time_cost / len(stream_data))
            output_file = "data/simulation/shadow/complex_8.npz"
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            np.savez(
                output_file,
                position_err=np.array(position_err_list),
                orientation_err=np.array(orientation_err_list),
                relative_position_err=np.array(relative_position_err_list),
                relative_position_to_wrist_err=np.array(relative_position_to_wrist_err_list),
                time_cost=np.array(time_cost_list),
            )
            print(f"Saved quantitative results to {output_file}")


def main():
    teleoperation = RobotTeleoperationMain()
    teleoperation.main()


if __name__ == "__main__":
    main()
