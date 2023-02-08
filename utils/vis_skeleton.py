import os
import pickle
from argparse import ArgumentParser
import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
sns.set_style("whitegrid")
sns.set_palette("muted")
import moviepy.editor as mpe
from scipy.io.wavfile import write as write_wav
import pyquaternion as pyq


def create_hierarchy_nodes_from_txt(hierarchy_filepath):
    """
    Create hierarchy nodes: an array of markers used in the motion capture
    Args:
        hierarchy: bvh file header path for joint hierarchy

    Returns:
        nodes: array of markers to be used in motion processing

    """
    with open(hierarchy_filepath, "r") as f:
        hierarchy = f.readlines()

    joint_offsets = []
    joint_names = []

    for idx, _ in enumerate(hierarchy):
        hierarchy[idx] = hierarchy[idx].split()
        if not len(hierarchy[idx]) == 0:
            line_type = hierarchy[idx][0]
            if line_type == "OFFSET":
                offset = np.array([float(hierarchy[idx][1]),
                float(hierarchy[idx][2]), float(hierarchy[idx][3])])
                joint_offsets.append(offset)
            elif line_type == "ROOT" or line_type == "JOINT":
                joint_names.append(hierarchy[idx][1])
            elif line_type == "End":
                joint_names.append("End Site")

    nodes = []      
    if 'upper' in hierarchy_filepath:
        for idx, name in enumerate(joint_names):
            # assign parents
            if idx == 0:
                parent = None
            elif idx in [5, 8, 35]: #spine3->neck, shoulders
                parent = 4
            elif idx in [12, 16, 26]: #righthand->middle1, ring, index
                parent = 11
            elif idx in [17, 21]: # righthandring->righthandring1, righthandpinky
                parent = 16
            elif idx in [27, 31]: # righthandindex->righthandindex1, righthandthumb1
                parent = 26
            elif idx in [39, 43, 53]: #lefthand->middle1, ring, index
                parent = 38
            elif idx in [44, 48]: # lefthandring->lefthandring1, lefthandpinky
                parent = 43
            elif idx in [54, 58]: # lefthandindex->lefthandindex1, lefthandthumb1
                parent = 53
            else:
                parent = idx - 1

            # assign children
            if name == "End Site":
                children = None
            elif idx == 4:
                children = [5, 8, 35]
            elif idx == 11:
                children = [12, 16, 26]
            elif idx == 16:
                children = [17, 21]
            elif idx == 26:
                children = [27, 31]
            elif idx == 38:
                children = [39, 43, 53]
            elif idx == 43:
                children = [44, 48]
            elif idx == 53:
                children = [54, 58]
            else:
                children = [idx + 1]

            node = dict([("name", name), ("parent", parent), ("children", children), ("offset", joint_offsets[idx]), ("rel_degs", None), ("abs_qt", None), ("rel_pos", None), ("abs_pos", None)])
            if idx == 0:
                node["rel_pos"] = node["abs_pos"] = [float(0), float(0), float(0)]
                node["abs_qt"] = pyq.Quaternion()
            nodes.append(node)
        
    else:
        for idx, name in enumerate(joint_names):
            # assign parents
            if idx == 0:
                parent = None
            elif idx in [5, 10, 42]: #spine3->neck, shoulders
                parent = 4
            elif idx in [14, 19, 31]: #righthand->middle1, ring, index
                parent = 13
            elif idx in [20, 25]: # righthandring->righthandring1, righthandpinky
                parent = 19
            elif idx in [32, 37]: # righthandindex->righthandindex1, righthandthumb1
                parent = 31
            elif idx in [46, 51, 63]: #lefthand->middle1, ring, index
                parent = 45
            elif idx in [52, 57]: # righthandring->righthandring1, righthandpinky
                parent = 51
            elif idx in [64, 69]: # righthandindex->righthandindex1, righthandthumb1
                parent = 63
            elif idx in [1, 74, 81]: #hip->spine, legs
                parent = 0
            else:
                parent = idx - 1

            # assign children
            if name == "End Site":
                children = None
            elif idx == 4:
                children = [5, 10, 42]
            elif idx == 13:
                children = [14, 19, 31]
            elif idx == 19:
                children = [20, 25]
            elif idx == 31:
                children = [32, 37]
            elif idx == 45:
                children = [46, 51, 63]
            elif idx == 51:
                children = [52, 57]
            elif idx == 63:
                children = [64, 69]
            elif idx == 0:
                children = [1, 74, 81]
            else:
                children = [idx + 1]

            node = dict([("name", name), ("parent", parent), ("children", children), ("offset", joint_offsets[idx]), ("rel_degs", None), ("abs_qt", None), ("rel_pos", None), ("abs_pos", None)])
            if idx == 0:
                node["rel_pos"] = node["abs_pos"] = [float(0), float(0), float(0)]
                node["abs_qt"] = pyq.Quaternion()
            nodes.append(node)

    return nodes


def rot_vec_to_abs_pos_vec(frames, nodes, default_offset=[0, 0, 0]):
    """
    Transform vectors of the human motion from the joint angles to the
    absolute positions
    Args:
        frames: human motion in the join angles space,
        shape [T, num_joint, 3]
        nodes:  set of markers used in motion caption

    Returns:
        output_vectors : 3d coordinates of this human motion,
        shape [T, num_joint (end site included), 3]
    """
    frames = frames.reshape(len(frames), -1, 3)
    
    if frames.shape[1] == 45: # for hierarchy_upper
        zeros = np.zeros((len(frames), 1, 3))
        frames = np.concatenate([
            frames[:, :14],
            zeros,
            frames[:, 14:17],
            zeros,
            frames[:, 17:20],
            zeros,
            frames[:, 20:33],
            zeros,
            frames[:, 33:36],
            zeros,
            frames[:, 36:39],
            zeros,
            frames[:, 39:]
        ], axis=1)
        
    elif frames.shape[1] == 41:
        zeros = np.zeros((len(frames), 1, 3))
        frames = np.concatenate([
            zeros, # hips
            zeros, # Spine
            frames[:, :3],
            zeros,
            zeros,
            frames[:, 3:10],
            zeros,
            frames[:, 10:13],
            zeros,
            frames[:, 13:16],
            zeros,
            frames[:, 16:29],
            zeros,
            frames[:, 29:32],
            zeros,
            frames[:, 32:35],
            zeros,
            frames[:, 35:]
        ], axis=1)
        
    frames = frames.reshape(frames.shape[0], -1)
    positions = []
    num_joint = len(frames[0]) // 3 # 3 for (x,y,z) tuple
    for frame in frames:
        node_idx = 0
        for i in range(num_joint):
            stepi = i*3
            x_deg = float(frame[stepi])
            y_deg = float(frame[stepi+1])
            z_deg = float(frame[stepi+2])
            if nodes[node_idx]["name"] == "End Site":
                node_idx = node_idx + 1
            nodes[node_idx]["rel_degs"] = [x_deg, y_deg, z_deg]
            current_node = nodes[node_idx]
            node_idx = node_idx + 1
        for start_node in nodes:
            abs_pos = np.array(default_offset)
            current_node = start_node
            if start_node["children"] is not None:
                #= if not start_node["name"] = "end site"
                for child_idx in start_node["children"]:
                    child_node = nodes[child_idx]
                    child_offset = np.array(child_node["offset"])
                    qx = pyq.Quaternion(
                        axis=[1, 0, 0], degrees=start_node["rel_degs"][0]
                    )
                    qy = pyq.Quaternion(
                        axis=[0, 1, 0], degrees=start_node["rel_degs"][1]
                    )
                    qz = pyq.Quaternion(
                        axis=[0, 0, 1], degrees=start_node["rel_degs"][2]
                    )
                    qrot = qx * qy * qz
                    offset_rotated = qrot.rotate(child_offset)
                    child_node["rel_pos"] = start_node["abs_qt"].rotate(
                        offset_rotated
                    )
                    child_node["abs_qt"] = start_node["abs_qt"] * qrot
            while current_node["parent"] is not None:
                abs_pos = abs_pos + current_node["rel_pos"]
                current_node = nodes[current_node["parent"]]
            start_node["abs_pos"] = abs_pos
        line = []
        for node in nodes:
            line.append(node["abs_pos"])
        positions.append(line)
    return np.array(positions)


def plot_skeleton(skeleton, nodes, output_path="skeleton.jpg"):
    """
    :param skeleton: numpy array, shape [num_joint, 3]
    :param nodes: set of markers used in motion caption by "create_hierarchy_nodes_from_txt"
    """
    fig = plt.figure(dpi=200)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(skeleton[:, 0], skeleton[:, 2], zs=skeleton[:, 1], s=2, alpha=1)
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_zlabel("y")
    ax.set_zlim(-100, 100)
    ax.set_ylim(100, -100)
    ax.set_xlim(-100, 100)
    for i, node in enumerate(nodes):
        children = node["children"]
        if children != None:
            for child in children:
                ax.plot(
                    [skeleton[i, 0], skeleton[child, 0]],
                    [skeleton[i, 2], skeleton[child, 2]],
                    [skeleton[i, 1], skeleton[child, 1]],
                    c="blue",
                    linewidth=0.5
                )
    plt.savefig(output_path)
    plt.close()


def make_skeleton_video(skeleton_seq, interval, hierarchy_path, output_path="skeleton_video.mp4"):
    """
    :param skeleton_seq: numpy array, shape [T, num_joint, 3]
    :param nodes: set of markers used in motion caption by "create_hierarchy_nodes_from_txt"
    :interval: interval between consecutive frames, in millisec
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nodes = create_hierarchy_nodes_from_txt(hierarchy_path)

    # lines between nodes
    lines = []
    for i, node in enumerate(nodes):
        children = node["children"]
        if children != None:
            for child in children:
                lines.append([i, child])

    def update_graph(num):
        print(f"Rendering {num}/{skeleton_seq.shape[0]}")
        x, y, z = skeleton_seq[num][:, 0], skeleton_seq[num][:, 2], skeleton_seq[num][:, 1]
        graph._offsets3d = (x, y, z)
        title.set_text("3D Plot, step={}".format(num))
        for g_l, l in zip(graph_lines, lines):
            g_l[0].set_data(x[l], y[l])
            g_l[0].set_3d_properties(z[l])

    fig = plt.figure(dpi=200)
    ax = fig.add_subplot(111, projection="3d")
    title = ax.set_title("3D Plot")
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_zlabel("y")

    graph = ax.scatter([], [], [], color=sns.color_palette()[0], s=0.5, alpha=1)
    graph_lines = [ax.plot([], [], [], color=sns.color_palette()[0], linewidth=0.5) for i, _ in enumerate(lines)]

    ax.set_zlim(-100, 100)
    ax.set_ylim(100, -100)
    ax.set_xlim(-100, 100)

    anim = FuncAnimation(fig, update_graph, range(skeleton_seq.shape[0]), interval=interval)
    anim.save(output_path)

    plt.close()


def pose2skeleton(poses, hierarchy_path):
    poses.shape[1] == 225, "Full joint required, excluding translation"
    poses = poses.reshape(-1, 75, 3)
    nodes = create_hierarchy_nodes_from_txt(hierarchy_path)
    return rot_vec_to_abs_pos_vec(poses, nodes)


def visualize_sample_skeleton(
    out_poses,
    poses,
    pose_fps,
    wav,
    hierarchy_path,
    hand=False,
    output_path="sample_video.mp4"
):
    # assert poses.shape[1] == 225, "Full joint required, excluding translation"
    # assert out_poses.shape[1] == 225, "Full joint required, excluding translation"

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # convert pose to skeleton
    print('Converting pose to skeleton...')
    out_poses = out_poses.reshape(len(out_poses), -1, 3)
    poses = poses.reshape(len(poses), -1, 3)
    nodes = create_hierarchy_nodes_from_txt(hierarchy_path)
    skels = rot_vec_to_abs_pos_vec(poses, nodes)
    out_skels = rot_vec_to_abs_pos_vec(out_poses, nodes)

    # make comparing video
    # lines between nodes
    lines = []
    for i, node in enumerate(nodes):
        children = node["children"]
        if children != None:
            for child in children:
                lines.append([i, child])
                
    if hand:
        # lines of left hands, starts from 13 to 41
        rhand_lines = []
        for i, node in enumerate(nodes[13:42]):
            children = node["children"]
            if children != None:
                for child in children:
                    rhand_lines.append([i, child - 13])

        # lines of right hands, starts from 45 to 73
        lhand_lines = []
        for i, node in enumerate(nodes[45:74]):
            children = node["children"]
            if children != None:
                for child in children:
                    lhand_lines.append([i, child - 45])

    # plot left and right side by side
    # left is output, right is GT
    fig = plt.figure(dpi=400)
    # plot output
    ax = fig.add_subplot(2,4,(1,2), projection="3d")
    out_title = ax.set_title("")
    # ax.set_xlabel("x")
    # ax.set_ylabel("z")
    # ax.set_zlabel("y")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    out_graph = ax.scatter([], [], [], color=sns.color_palette()[0], s=0.05, alpha=1)
    out_graph_lines = [ax.plot([], [], [], color=sns.color_palette()[0], linewidth=0.5) for _ in lines]
    ax.set_zlim(-80, 80)
    ax.set_ylim(80, -80)
    ax.set_xlim(-80, 80)

    if hand:
        # plot output left hand
        ax = fig.add_subplot(2,4,5, projection="3d")
        out_lhand_title = ax.set_title("")
        # ax.set_xlabel("x")
        # ax.set_ylabel("z")
        # ax.set_zlabel("y")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        out_lhand_graph = ax.scatter([], [], [], color=sns.color_palette()[0], s=0.1, alpha=1)
        out_lhand_graph_lines = [ax.plot([], [], [], color=sns.color_palette()[0], linewidth=0.5) for _ in lhand_lines]
        ax.set_zlim(-15, 15)
        ax.set_ylim(15, -15)
        ax.set_xlim(-15, 15)

        # plot output right hand
        ax = fig.add_subplot(2,4,6, projection="3d")
        out_rhand_title = ax.set_title("")
        # ax.set_xlabel("x")
        # ax.set_ylabel("z")
        # ax.set_zlabel("y")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        out_rhand_graph = ax.scatter([], [], [], color=sns.color_palette()[0], s=0.1, alpha=1)
        out_rhand_graph_lines = [ax.plot([], [], [], color=sns.color_palette()[0], linewidth=0.5) for _ in rhand_lines]
        ax.set_zlim(-15, 15)
        ax.set_ylim(15, -15)
        ax.set_xlim(-15, 15)

    # plot GT
    ax = fig.add_subplot(2,4,(3,4), projection="3d")
    gt_title = ax.set_title("")
    # ax.set_xlabel("x")
    # ax.set_ylabel("z")
    # ax.set_zlabel("y")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    gt_graph = ax.scatter([], [], [], color=sns.color_palette()[0], s=0.1, alpha=1)
    gt_graph_lines = [ax.plot([], [], [], color=sns.color_palette()[0], linewidth=0.5) for _ in lines]
    ax.set_zlim(-80, 80)
    ax.set_ylim(80, -80)
    ax.set_xlim(-80, 80)

    if hand:
        # plot GT left hand
        ax = fig.add_subplot(2,4,7, projection="3d")
        gt_lhand_title = ax.set_title("")
        # ax.set_xlabel("x")
        # ax.set_ylabel("z")
        # ax.set_zlabel("y")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        gt_lhand_graph = ax.scatter([], [], [], color=sns.color_palette()[0], s=0.1, alpha=1)
        gt_lhand_graph_lines = [ax.plot([], [], [], color=sns.color_palette()[0], linewidth=0.5) for _ in lhand_lines]
        ax.set_zlim(-15, 15)
        ax.set_ylim(15, -15)
        ax.set_xlim(-15, 15)

        # plot GT right hand
        ax = fig.add_subplot(2,4,8, projection="3d")
        gt_rhand_title = ax.set_title("")
        # ax.set_xlabel("x")
        # ax.set_ylabel("z")
        # ax.set_zlabel("y")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        gt_rhand_graph = ax.scatter([], [], [], color=sns.color_palette()[0], s=0.1, alpha=1)
        gt_rhand_graph_lines = [
            ax.plot([], [], [], color=sns.color_palette()[0], linewidth=0.5)
            for _ in rhand_lines
        ]
        ax.set_zlim(-15, 15)
        ax.set_ylim(15, -15)
        ax.set_xlim(-15, 15)

    # function for updating animation
    def update_graph(num):
        if num % 100 == 0:
            print(f"\r[Info] Rendering {num}/{skels.shape[0]}", end="")

        # update output
        x, y, z = out_skels[num][:, 0], out_skels[num][:, 2], out_skels[num][:, 1]
        out_graph._offsets3d = (x, y, z)
        out_title.set_text("output, step={}".format(num))
        for g_l, l in zip(out_graph_lines, lines):
            g_l[0].set_data(x[l], y[l])
            g_l[0].set_3d_properties(z[l])
            
        if hand:
            # update output left hand
            lhand_skles = out_skels[:, 45:74].copy()
            lhand_skles -= lhand_skles[:, 0:1] # centerize
            x, y, z = lhand_skles[num][:, 0], lhand_skles[num][:, 2], lhand_skles[num][:, 1]
            out_lhand_graph._offsets3d = (x, y, z)
            out_lhand_title.set_text("left hand")
            for g_l, l in zip(out_lhand_graph_lines, lhand_lines):
                g_l[0].set_data(x[l], y[l])
                g_l[0].set_3d_properties(z[l])

            # update output right hand
            rhand_skles = out_skels[:, 13:42].copy()
            rhand_skles -= rhand_skles[:, 0:1] # centerize
            x, y, z = rhand_skles[num][:, 0], rhand_skles[num][:, 2], rhand_skles[num][:, 1]
            out_rhand_graph._offsets3d = (x, y, z)
            out_rhand_title.set_text("right hand")
            for g_l, l in zip(out_rhand_graph_lines, rhand_lines):
                g_l[0].set_data(x[l], y[l])
                g_l[0].set_3d_properties(z[l])

        # update gt
        x, y, z = skels[num][:, 0], skels[num][:, 2], skels[num][:, 1]
        gt_graph._offsets3d = (x, y, z)
        gt_title.set_text("GT, step={}".format(num))
        for g_l, l in zip(gt_graph_lines, lines):
            g_l[0].set_data(x[l], y[l])
            g_l[0].set_3d_properties(z[l])

        if hand:
            # update gt left hand
            lhand_skles = skels[:, 45:74].copy()
            lhand_skles -= lhand_skles[:, 0:1] # centerize
            x, y, z = lhand_skles[num][:, 0], lhand_skles[num][:, 2], lhand_skles[num][:, 1]
            gt_lhand_graph._offsets3d = (x, y, z)
            gt_lhand_title.set_text("left hand")
            for g_l, l in zip(gt_lhand_graph_lines, lhand_lines):
                g_l[0].set_data(x[l], y[l])
                g_l[0].set_3d_properties(z[l])

            # update gt right hand
            rhand_skles = skels[:, 13:42].copy()
            rhand_skles -= rhand_skles[:, 0:1] # centerize
            x, y, z = rhand_skles[num][:, 0], rhand_skles[num][:, 2], rhand_skles[num][:, 1]
            gt_rhand_graph._offsets3d = (x, y, z)
            gt_rhand_title.set_text("right hand")
            for g_l, l in zip(gt_rhand_graph_lines, rhand_lines):
                g_l[0].set_data(x[l], y[l])
                g_l[0].set_3d_properties(z[l])

    anim = FuncAnimation(fig, update_graph, range(skels.shape[0]), interval=1000/pose_fps)
    anim.save("tmp.mp4")
    plt.close()
    print()

    # concat wav to created video
    # write wav to file
    write_wav("tmp.wav", 16000, wav)
    video = mpe.VideoFileClip("tmp.mp4")
    # load wav and merge to video
    video = video.set_audio(mpe.AudioFileClip("tmp.wav"))
    video.write_videofile(output_path)
    # remove tmp wav file
    os.remove("tmp.wav")
    os.remove("tmp.mp4")


def main():
    p = ArgumentParser()
    p.add_argument('--pkl-path', metavar='PATH', help='Path for pickle file.')
    p.add_argument('--out-path', metavar='PATH', default='output.mp4')
    p.add_argument('--hierarchy-path', metavar='PATH', default='datasets/hierarchy.txt')
    p.add_argument('--pose-fps', default=20, help='FPS of pose.')
    args = p.parse_args()
    
    with open(args.pkl_path, "rb") as f:
        sample = pickle.load(f)
        out_pose = sample["out"]
        pose = sample["pose"]
        wav = sample["wav"]

    visualize_sample_skeleton(
        out_pose.reshape(len(out_pose), -1),
        pose.reshape(len(pose), -1),
        args.pose_fps,
        wav,
        args.hierarchy_path,
        output_path=args.out_path
    )


if __name__ == "__main__":
    main()