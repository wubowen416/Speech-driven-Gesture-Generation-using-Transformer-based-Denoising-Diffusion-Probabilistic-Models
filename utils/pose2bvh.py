import pickle
import os
import glob
import scipy
from scipy import io
from scipy.io import wavfile
import numpy as np
from argparse import ArgumentParser
from scipy.signal import butter,filtfilt

import sys
sys.path.append('.')
from datasets.data_utils import euler2log_rot, log_rot2euler, unroll_log_rot


def butter_lowpass_filter(data):
    cutoff = 2
    fs = 18
    order = 2
    normal_cutoff = cutoff / 0.5 / fs
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y


def pose2bvh(
    bvh_filepath: str,
    pose: np.ndarray,
    hierarchy: str,
    fps: int = 20,
    root_translation: list = [0, 0, 0],
    filter=False
):
    num_frames = pose.shape[0]
    frame_time = 1 / fps
    
    if filter:
        log_rot = euler2log_rot(pose.reshape(-1, 3)).reshape(len(pose), -1, 3)
        log_rot = np.concatenate([unroll_log_rot(log_rot[:, i]) for i in range(log_rot.shape[1])], axis=1)
        filtered = np.array([butter_lowpass_filter(x) for x in log_rot.T]).T
        pose = log_rot2euler(filtered.reshape(-1, 3)).reshape(len(filtered), -1)

    translation_array = np.array(root_translation)[np.newaxis, :]
    translation_array = np.repeat(translation_array, num_frames, axis=0)
    motion = np.concatenate([translation_array, pose], axis=1)

    headers = hierarchy + [
        'MOTION\n', f'Frames: {num_frames}\n', f'Frame Time: {frame_time}'
    ]

    print(f'Write to {bvh_filepath}')
    np.savetxt(bvh_filepath, motion, header=''.join(headers), comments='')


def sample2bvh_batch(
    sample_dir_path: str, bvh_dir_path: str, hierarchy_path: str, filter: bool
):
    with open(hierarchy_path, 'r') as f:
        hierarchy = f.readlines()
    sample_filepaths = glob.glob(os.path.join(sample_dir_path, '*.pkl'))
    os.makedirs(bvh_dir_path, exist_ok=True)
    for sample_filepath in sample_filepaths:
        bvh_gt_filepath = os.path.join(
            bvh_dir_path,
            os.path.basename(sample_filepath.replace('.pkl', '-gt.bvh'))
        )
        bvh_out_filepath = os.path.join(
            bvh_dir_path,
            os.path.basename(sample_filepath.replace('.pkl', '-out.bvh'))
        )
        with open(sample_filepath, 'rb') as f:
            sample = pickle.load(f)
        pose_gt = sample['pose']
        pose_out = sample['out']
        pose2bvh(bvh_gt_filepath, pose_gt, hierarchy)
        pose2bvh(bvh_out_filepath, pose_out, hierarchy, filter=filter)

        # save wav file also
        wav_filepath = os.path.join(
            bvh_dir_path, 
            os.path.basename(sample_filepath.replace('.pkl', '.wav'))
        )
        scipy.io.wavfile.write(wav_filepath, 16000, sample['wav'])


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("--sample-dir", type=str, metavar="PATH")
    p.add_argument("--bvh-dir", type=str, metavar="PATH")
    p.add_argument("--hierarchy", type=str, metavar="PATH")
    p.add_argument("--filter", action="store_true", default=False)
    args = p.parse_args()

    sample2bvh_batch(args.sample_dir, args.bvh_dir, args.hierarchy, args.filter)