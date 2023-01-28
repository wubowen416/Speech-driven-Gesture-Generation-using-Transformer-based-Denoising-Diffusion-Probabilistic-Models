from typing import List
import numpy as np
import librosa
import math
from sklearn.preprocessing import normalize


def compute_angle_change_rate(
    dir_vec_seq_batch: np.ndarray, # (N,T,N_joint,3)
    angle_pairs: List[List[int]],
    joint_groups: List[List[int]] = None,
    group_weights: List[int] = None,
):
    assert len(dir_vec_seq_batch.shape) == 4
    batch_size, timesteps, num_joint, joint_dim = dir_vec_seq_batch.shape

    if joint_groups is None:
        joint_groups = [np.arange(len(angle_pairs))]
        group_weights = [1]

    all_vec = dir_vec_seq_batch.reshape(-1, num_joint, 3)

    # calculate MAAC: mean absolute angle change
    vec1_idxs, vec2_idxs = zip(*angle_pairs)
    vec1 = all_vec[:, vec1_idxs]
    vec2 = all_vec[:, vec2_idxs]
    vec1 = normalize(vec1.reshape(-1, 3), axis=1).reshape(
        -1, len(angle_pairs), joint_dim
    )
    vec2 = normalize(vec2.reshape(-1, 3), axis=1).reshape(
        -1, len(angle_pairs), joint_dim
    )
    dot_prod = np.sum((vec1 * vec2), axis=-1)
    dot_prod = np.clip(dot_prod, -1, 1)
    angle = np.arccos(dot_prod) / math.pi
    angle = angle.reshape(batch_size, timesteps, -1)
    angle_diff = np.abs(np.diff(angle, axis=1))
    maacs = np.mean(angle_diff, axis=(0, 1), keepdims=True)

    # calculate angle change rate
    angle_change_rate = np.divide(
        angle_diff, maacs, np.zeros_like(angle_diff), where=(maacs!=0)
    ) # (N,T,N_joint)
    # apply groups weight
    weights = np.zeros_like(angle_change_rate)
    for group, weight in zip(joint_groups, group_weights):
        weights[:, :, group] = weight
    angle_change_rate = np.mean(weights * angle_change_rate, axis=-1)
    angle_change_rate = np.concatenate(
        [np.zeros((batch_size, 1)), angle_change_rate], axis=1
    )

    return angle_change_rate


def extract_motion_beat_times(
    angle_change_rate: np.ndarray,
    motion_fps: int,
    thres: float
):
    times = []
    for t in range(2, angle_change_rate.shape[0]-1):
        if (
            angle_change_rate[t] < angle_change_rate[t - 1] and \
                angle_change_rate[t] < angle_change_rate[t + 1]
        ):
            if (
                angle_change_rate[t - 1] - angle_change_rate[t] >= thres or \
                    angle_change_rate[t + 1] - angle_change_rate[t] >= thres
            ):
                times.append(float(t) / motion_fps)
    return np.array(times)


def beat_consistency_score(
    dir_vec_seq_batch: np.ndarray, # (N,T,N_joint,3)
    motion_fps: int,
    angle_pairs: List[List[int]],
    wav_seq_batch: np.ndarray, # (N,T)
    wav_sr: int,
    joint_groups: List[List[int]] = None,
    group_weights: List[int] = None,
    motion_beat_threshold: float = 0.03,
    sigma: float = 0.1
):
    angle_change_rate = compute_angle_change_rate(
        dir_vec_seq_batch,
        angle_pairs,
        joint_groups,
        group_weights
    )

    scores = []
    for b in range(len(dir_vec_seq_batch)):
        motion_beat_time = extract_motion_beat_times(
            angle_change_rate[b], motion_fps, motion_beat_threshold
        )
        if (len(motion_beat_time) == 0):
            continue

        audio_beat_time = librosa.onset.onset_detect(
            y=wav_seq_batch[b], sr=wav_sr, units='time'
        )

        total = 0
        for audio in audio_beat_time:
            total += np.power(
                math.e, -np.min(np.power((audio - \
                    motion_beat_time), 2)) / (2 * sigma ** 2)
            )
        scores.append(total / len(audio_beat_time))

    return np.mean(scores)


def beat_recall_score(
    pred_dir_vec_seq_batch: np.ndarray,
    target_dir_vec_seq_batch: np.ndarray,
    motion_fps: int,
    angle_pairs: List[List[int]],
    joint_groups: List[List[int]] = None,
    groups_weight: List[int] = None,
    motion_beat_threshold: float = 0.03,
    sigma: float = 0.1
):
    pred_angle_change_rate = compute_angle_change_rate(
        pred_dir_vec_seq_batch,
        angle_pairs,
        joint_groups,
        groups_weight
    )
    target_angle_change_rate = compute_angle_change_rate(
        target_dir_vec_seq_batch,
        angle_pairs,
        joint_groups,
        groups_weight
    )

    scores = []
    for pred_acr, tgt_acr in zip(
        pred_angle_change_rate, target_angle_change_rate
    ):
        pred_motion_beat_time = extract_motion_beat_times(
            pred_acr, motion_fps, motion_beat_threshold
        )
        target_motion_beat_time = extract_motion_beat_times(
            tgt_acr, motion_fps, motion_beat_threshold
        )

        if len(target_motion_beat_time) == 0:
            continue

        total = 0
        for target_beat_time in target_motion_beat_time:
            total += np.power(
                math.e, -np.min(np.power((target_beat_time - \
                    pred_motion_beat_time), 2)) / (2 * sigma ** 2)
            )
        scores.append(total / len(target_motion_beat_time))

    return np.mean(scores)