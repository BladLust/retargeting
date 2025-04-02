""" """

import numpy as np
import os
import sys
from matplotlib import pyplot as plt

parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parentdir)
from robot_pinocchio import RobotPinocchio
from robot_adaptor import RobotAdaptor
from utils.utils_calc import diagRotMat

# ---------------- robot model -----------------

urdf_file_name = os.readlink(
    "assets/panda_leap_tac3d.urdf"
)  # no touch bodies/joints/sensors
mjcf_file_name = "assets/panda_leap_tac3d_touch_asset.xml"
actuated_joints_name = [f"panda_joint{i+1}" for i in range(7)] + [
    f"joint_{i}" for i in range(16)
]
touch_joints_name = []
robot_model = RobotPinocchio(
    robot_file_path=urdf_file_name,
    robot_file_type="urdf",
)
robot_adaptor = RobotAdaptor(
    robot_model=robot_model,
    actuated_joints_name=actuated_joints_name,
    touch_joints_name=touch_joints_name,
)

# ---------------- data -----------------

save_dir = "data/test_touch"
file = os.path.join(save_dir, "data.npy")
data = np.load(file, allow_pickle=True).item()

timestamp = np.array(data["timestamp"])
contact_forces = np.array(data["contact_forces"])
contact_moments = np.array(data["contact_moments"])
contact_locations = np.array(data["contact_locations"])
joint_pos = np.array(data["joint_pos"])
target_joint_pos = np.array(data["target_joint_pos"])
joint_torques = np.array(data["joint_torques"])

finger_idx = 1
select_joint_indices = [8, 9, 10]
joint_torques_ext = np.zeros_like(joint_pos)

for i in range(joint_pos.shape[0]):
    q = joint_pos[i]
    F_in_tip = np.concatenate(
        [
            contact_forces[i][finger_idx].reshape(-1, 1),
            contact_moments[i][finger_idx].reshape(-1, 1),
        ],
        axis=0,
    )  # shape (6, 1)
    fingertip_pose = robot_model.get_frame_pose("finger1_tip_center", qpos=q)
    F_in_world = diagRotMat(fingertip_pose[:3, :3]) @ F_in_tip
    jaco = robot_model.get_frame_space_jacobian("finger1_tip_center", qpos=q)
    jaco = robot_adaptor.backward_jacobian(jaco)
    tau_ext = jaco.T @ F_in_world
    joint_torques_ext[i] = tau_ext.reshape(-1)

# ---------------- figure -----------------


plt.figure(1)
plt.plot(timestamp, contact_forces[:, finger_idx, :], label="contact_force")
plt.plot(timestamp, contact_moments[:, finger_idx, :], label="contact_moment")
plt.legend()
plt.title("Contact Force & Moment")

plt.figure(2)
for i, joint_id in enumerate(select_joint_indices):
    plt.subplot(1, len(select_joint_indices), i + 1)
    plt.plot(
        timestamp,
        joint_torques_ext[:, joint_id],
        label=f"torque_ext_{joint_id}",
    )
    plt.legend()
plt.suptitle("External Joint Torques")

plt.figure(3)

for i, joint_id in enumerate(select_joint_indices):
    plt.subplot(1, len(select_joint_indices), i + 1)
    plt.plot(
        timestamp,
        joint_torques[:, joint_id],
        label=f"torque_{joint_id}",
    )
    plt.legend()
plt.suptitle("Joint Torques")

plt.figure(4)
delta_joint_pos = target_joint_pos - joint_pos
for i, joint_id in enumerate(select_joint_indices):
    plt.subplot(1, len(select_joint_indices), i + 1)
    plt.plot(
        timestamp,
        target_joint_pos[:, joint_id],
        label=f"target_{joint_id}",
    )
    plt.plot(
        timestamp,
        joint_pos[:, joint_id],
        label=f"obs_{joint_id}",
    )
    plt.plot(
        timestamp,
        delta_joint_pos[:, joint_id],
        label=f"delta_{joint_id}",
    )
    plt.legend()
    plt.ylim([-1.57, 1.57])
plt.suptitle("Joint Positions")

plt.show()
