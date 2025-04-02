import numpy as np

from robot_adaptor import RobotAdaptor


class RobotBenchmark:
    def __init__(self, hand_type: str, robot_adaptor: RobotAdaptor):
        self.robot_adaptor = robot_adaptor
        self.robot_model = robot_adaptor.robot_model
        self.hand_type = hand_type
        if self.hand_type == "leap":
            self.fingertip_links_names = [
                "thumb_tip_center",
                "finger1_tip_center",
                "finger2_tip_center",
                "finger3_tip_center",
            ]
        else:
            self.fingertip_links_names = [
                "thtip",
                "fftip",
                "mftip",
                "rftip",
                "lftip",
            ]

    def position_error(self, retarget_qpos, target_pos, type_id):
        """
        Calculate the position error between the retargeted qpos and the target qpos.
        """
        # self.robot_model.compute_forward_kinematics(retarget_qpos)
        if type_id == 1:
            retarget_links_pose_list = [self.robot_model.get_frame_pose(name) for name in self.fingertip_links_names]
            retarget_links_pose = np.stack(retarget_links_pose_list, axis=0)  # shape (n, 4, 4)
            retarget_links_pos = retarget_links_pose[:, 0:3, 3]  # shape (n, 3)
            if self.hand_type == "leap":
                idx = [4, 8, 12, 16]
            else:
                idx = [4, 8, 12, 16, 20]
            human_links_pos = np.array([target_pos[i] for i in idx])  # shape (n, 3)
            # print("retarget_links_pos:", retarget_links_pos)
            # print("human_links_pos:", human_links_pos)
            err = np.linalg.norm(retarget_links_pos - human_links_pos, axis=1)
        elif type_id == 2:
            retarget_links_pose_list = [self.robot_model.get_frame_pose("wrist")]
            retarget_links_pose = np.stack(retarget_links_pose_list, axis=0)
            retarget_links_pos = retarget_links_pose[0, 0:3, 3]
            human_links_pos = target_pos[0]
            err = np.linalg.norm(retarget_links_pos - human_links_pos)
        return err

    def orientation_error(self, retarget_qpos, target_qpos, type_id):
        """
        Calculate the orientation error between the retargeted qpos and the target qpos.
        """
        if type_id == 1:
            # Get each fingertip's 4x4 pose and extract the x-axis direction (global)
            retarget_links_pose_list = [self.robot_model.get_frame_pose(name) for name in self.fingertip_links_names]
            retarget_links_pose = np.stack(retarget_links_pose_list, axis=0)  # shape: (n, 4, 4)
            robot_x_dirs = retarget_links_pose[:, :3, 0]  # shape: (n, 3)
            robot_x_dirs = np.array([vec / np.linalg.norm(vec) for vec in robot_x_dirs])
            robot_z_dirs = retarget_links_pose[:, :3, 2]  # shape: (n, 3)
            robot_z_dirs = np.array([vec / np.linalg.norm(vec) for vec in robot_z_dirs])

            # Compute fingertip direction vectors from hand keypoints (assumed shape: (21, 3))
            hand_kps = np.array(target_qpos)
            if self.hand_type == "leap":
                fingertip_indices = [4, 8, 12, 16]
                next_indices = [3, 7, 11, 15]
            else:
                fingertip_indices = [4, 8, 12, 16, 20]
                next_indices = [3, 7, 11, 15, 19]
            human_dirs = []
            for tip_idx, next_idx in zip(fingertip_indices, next_indices):
                direction = hand_kps[tip_idx] - hand_kps[next_idx]
                norm_dir = np.linalg.norm(direction)
                direction = direction / norm_dir if norm_dir >= 1e-6 else np.array([1, 0, 0])
                human_dirs.append(direction)
            human_dirs = np.array(human_dirs)  # shape: (4, 3)

            # Compute angle error (in radians) between corresponding vectors
            if self.hand_type == "leap":
                dots = np.sum(robot_x_dirs * human_dirs, axis=1)
            else:
                dots = np.sum(robot_z_dirs * human_dirs, axis=1)
            dots = np.clip(dots, -1.0, 1.0)
            err = np.arccos(dots)
        return err

    def relative_position_error(self, retarget_qpos, target_pos, type_id):
        """
        Calculate the relative position error from primary fingertips to thumb fingertip
        between the retargeted qpos and the target qpos.
        """
        if type_id == 1:
            # Get retargeted primary fingertip positions (assumed to be 3 fingertips)
            if self.hand_type == "leap":
                primary_fingertip_names = [name for name in self.fingertip_links_names if name != "thumb_tip_center"]
            else:
                primary_fingertip_names = [name for name in self.fingertip_links_names if name != "thtip"]
            retarget_links_pose_list = [self.robot_model.get_frame_pose(name) for name in primary_fingertip_names]
            retarget_links_pose = np.stack(retarget_links_pose_list, axis=0)  # shape: (n, 4, 4)
            retarget_links_pos = retarget_links_pose[:, :3, 3]  # shape: (n, 3)

            # Get retargeted thumb tip position
            if self.hand_type == "leap":
                thumb_link_pose = self.robot_model.get_frame_pose("thumb_tip_center")
            else:
                thumb_link_pose = self.robot_model.get_frame_pose("thtip")
            thumb_link_pos = thumb_link_pose[:3, 3]

            # Compute relative vectors for the retargeted hand (from thumb to each primary fingertip)
            retarget_relative_vectors = retarget_links_pos - thumb_link_pos

            # For the target hand, assume target_pos is (21, 3) with:
            # thumb tip at index 4 and primary fingertips at indices [8, 12, 16]
            target_pos = np.array(target_pos)
            human_thumb_pos = target_pos[4]
            if self.hand_type == "leap":
                primary_indices = [8, 12, 16]
            else:
                primary_indices = [8, 12, 16, 20]
            human_fingertip_pos = target_pos[primary_indices]
            human_relative_vectors = human_fingertip_pos - human_thumb_pos

            # Calculate the Euclidean distance error for each primary finger
            error = np.linalg.norm(retarget_relative_vectors - human_relative_vectors, axis=1)
        return error

    def relative_position_to_wrist_error(self, retarget_qpos, target_pos, type_id):
        """
        Calculate the relative position error from all fingertips to wrist
        between the retargeted qpos and the target qpos.
        """
        if type_id == 1:
            # Get retargeted fingertip positions
            retarget_links_pose_list = [self.robot_model.get_frame_pose(name) for name in self.fingertip_links_names]
            retarget_links_pose = np.stack(retarget_links_pose_list, axis=0)
            retarget_links_pos = retarget_links_pose[:, :3, 3]
            if self.hand_type == "leap":
                wrist_link_pose = self.robot_model.get_frame_pose("wrist")
            else:
                wrist_link_pose = self.robot_model.get_frame_pose("ee_link")
            wrist_link_pos = wrist_link_pose[:3, 3]
            retarget_vector = retarget_links_pos - wrist_link_pos
            if self.hand_type == "leap":
                idx = [4, 8, 12, 16]
            else:
                idx = [4, 8, 12, 16, 20]
            human_links_pos = np.array([target_pos[i] for i in idx])  # shape (n, 3)
            human_wrist_pos = target_pos[0]
            human_vector = human_links_pos - human_wrist_pos
            error = np.linalg.norm(retarget_vector - human_vector, axis=1)
        return error
