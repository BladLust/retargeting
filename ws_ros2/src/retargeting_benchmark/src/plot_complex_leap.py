import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from utils.utils_plot import plotHistogram
from matplotlib.cm import get_cmap

params = {
    "font.family": "Times New Roman",
    "mathtext.fontset": "stix",
    "font.size": 16,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}
rcParams.update(params)

base_dir = "data/simulation/leap"

mean_pos_err = []
mean_ori_err = []
mean_rel_pos_err = []
mean_rel_pos_to_wrist_err = []

for i in range(9):
    pos_err_all = []
    ori_err_all = []
    rel_pos_err_all = []
    rel_pos_to_wrist_err_all = []

    file_name = f"complex_{i}.npz"
    file_path = os.path.join(base_dir, file_name)
    data = np.load(file_path)
    for idx in range(4):
        pos_err_all.extend(data["position_err"][:, idx])
        ori_err_all.extend(data["orientation_err"][:, idx])
        rel_pos_to_wrist_err_all.extend(data["relative_position_to_wrist_err"][:, idx])
        if idx < 3:
            rel_pos_err_all.extend(data["relative_position_err"][:, idx])

    print(f"pos_err_all: {len(pos_err_all)}")
    print(f"ori_err_all: {len(ori_err_all)}")
    print(f"rel_pos_err_all: {len(rel_pos_err_all)}")
    print(f"rel_pos_to_wrist_err_all: {len(rel_pos_to_wrist_err_all)}")

    mean_pos_err.append(np.mean(pos_err_all))
    mean_ori_err.append(np.mean(ori_err_all))
    mean_rel_pos_err.append(np.mean(rel_pos_err_all))
    mean_rel_pos_to_wrist_err.append(np.mean(rel_pos_to_wrist_err_all))

mean_pos_err = np.array(mean_pos_err) * 100
mean_ori_err = np.array(mean_ori_err)
mean_rel_pos_err = np.array(mean_rel_pos_err) * 100
mean_rel_pos_to_wrist_err = np.array(mean_rel_pos_to_wrist_err) * 100

# reshape for histogram input
mean_pos_err = np.array(mean_pos_err).reshape(-1, 1)
mean_ori_err = np.array(mean_ori_err).reshape(-1, 1)
mean_rel_pos_err = np.array(mean_rel_pos_err).reshape(-1, 1)
mean_rel_pos_to_wrist_err = np.array(mean_rel_pos_to_wrist_err).reshape(-1, 1)

if __name__ == "__main__":
    save_dir = os.path.join(base_dir, "plot")
    os.makedirs(save_dir, exist_ok=True)

    fig, axs = plt.subplots(1, 4, figsize=(21, 4))

    labels = ["Full", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8"]
    bar_labels = ["Error"]

    group_colors = get_cmap("Pastel1")
    bar_colors = [
        group_colors(0),  # Full
        group_colors(1), group_colors(1),  # A1, A2
        group_colors(2), group_colors(2),  # A3, A4
        group_colors(3), group_colors(3),  # A5, A6
        group_colors(4), group_colors(4),  # A7, A8
    ]

    # 子图1：Position Error
    plt.sca(axs[0])
    plotHistogram(
        data=mean_pos_err,
        x_labels=labels,
        bar_labels=bar_labels,
        bar_colors=bar_colors,
        x_width=0.7,
        border_width=0.2,
    )
    plt.grid(axis="y")

    # 子图2：Relative Position to Wrist Error
    plt.sca(axs[1])
    plotHistogram(
        data=mean_rel_pos_to_wrist_err,
        x_labels=labels,
        bar_labels=bar_labels,
        bar_colors=bar_colors,
        x_width=0.7,
        border_width=0.2,
    )
    plt.grid(axis="y")

    # 子图3：Relative Position Error
    plt.sca(axs[2])
    plotHistogram(
        data=mean_rel_pos_err,
        x_labels=labels,
        bar_labels=bar_labels,
        bar_colors=bar_colors,
        x_width=0.7,
        border_width=0.2,
    )
    plt.grid(axis="y")

    # 子图4：Orientation Error
    plt.sca(axs[3])
    plotHistogram(
        data=mean_ori_err,
        x_labels=labels,
        bar_labels=bar_labels,
        bar_colors=bar_colors,
        x_width=0.7,
        border_width=0.2,
    )
    plt.grid(axis="y")

    title_fontsize = 24

    axs[0].set_title("Fingertip Global Position", fontsize=title_fontsize)
    axs[0].set_ylabel("Error (cm)", fontsize=title_fontsize)
    axs[0].set_xlabel("Ablations", fontsize=title_fontsize)

    axs[1].set_title("Fingertip Relative Position to Wrist", fontsize=title_fontsize)
    axs[1].set_ylabel("Error (cm)", fontsize=title_fontsize)
    axs[1].set_xlabel("Ablations", fontsize=title_fontsize)

    axs[2].set_title("Fingertip Relative Position to Thumb", fontsize=title_fontsize)
    axs[2].set_ylabel("Error (cm)", fontsize=title_fontsize)
    axs[2].set_xlabel("Ablations", fontsize=title_fontsize)

    axs[3].set_title("Fingertip Orientation", fontsize=title_fontsize)
    axs[3].set_ylabel("Error (rad)", fontsize=title_fontsize)
    axs[3].set_xlabel("Ablations", fontsize=title_fontsize)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "fingertip_all_errors_complex_leap.pdf"), dpi=600)
    plt.show()
