import os
from abc import abstractmethod
from typing import List

import numpy as np


class RobotAdaptor:
    def __init__(
        self,
        robot_model,
        actuated_joints_name: List[str],
        touch_joints_name: List[str],
    ):
        self.robot_model = robot_model
        self.actuated_joints_name = actuated_joints_name
        self.touch_joints_name = touch_joints_name

        if len(self.actuated_joints_name) + len(self.touch_joints_name) != self.robot_model.dof:
            raise NotImplementedError("Currently, no support for coupled joints.")

        # 'model_idx' refers to the index in the model class 'self.robot_model'
        self.actuated_joints_model_idx = [self.robot_model.get_joint_index(name) for name in self.actuated_joints_name]
        self.touch_joints_model_idx = [self.robot_model.get_joint_index(name) for name in self.touch_joints_name]

    @property
    def doa(self) -> int:
        return len(self.actuated_joints_name)

    def check_doa(self, q):
        assert len(q) == self.doa

    def forward_qpos(self, qpos: np.ndarray) -> np.ndarray:
        """
        Args:
            qpos: position of the actuated joints
        Return:
            qpos_f: position of all dof joints
        """
        self.check_doa(qpos)
        qpos_dof = np.zeros((self.robot_model.dof))
        qpos_dof[self.actuated_joints_model_idx] = qpos.copy()
        qpos_dof[self.touch_joints_model_idx] = 0.0  # set the touch joints to be zero
        return qpos_dof

    def backward_qpos(self, qpos: np.ndarray) -> np.ndarray:
        """
        qpos_doa to qpos_dof.
        """
        self.robot_model.check_joint_dim(qpos)
        return qpos[self.actuated_joints_model_idx].copy()

    def backward_jacobian(self, jacobian: np.ndarray) -> np.ndarray:
        """
        Args:
            jacobian: shape (n_batch, 6, n_dof) computed by self.robot_model
        Return:
            jacobian: shape (n_batch, 6, n_doa)
        """
        jacobian_doa = jacobian[..., self.actuated_joints_model_idx]
        return jacobian_doa


if __name__ == "__main__":
    from robot_pinocchio import RobotPinocchio
    from utils.utils_mjcf import find_actuated_joints_name, find_touch_joints_name

    urdf_file_name = os.readlink("assets/panda_leap_tac3d.urdf")
    actuated_joints_name = [f"panda_joint{i+1}" for i in range(7)] + [f"joint_{i}" for i in range(16)]
    touch_joints_name: List[str] = []

    robot_model = RobotPinocchio(
        robot_file_path=urdf_file_name,
        robot_file_type="urdf",
    )

    robot_adaptor = RobotAdaptor(
        robot_model=robot_model,
        actuated_joints_name=actuated_joints_name,
        touch_joints_name=touch_joints_name,
    )

    doa = robot_adaptor.doa
    dof = robot_model.dof

    qpos_dof = robot_adaptor.forward_qpos(np.zeros((doa)))
    print(qpos_dof.shape)

    jaco_doa = robot_adaptor.backward_jacobian(np.zeros((6, dof)))
    print(jaco_doa.shape)
