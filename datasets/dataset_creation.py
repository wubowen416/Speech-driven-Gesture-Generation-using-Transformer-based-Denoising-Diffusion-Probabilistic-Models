import os
from typing import List
from .data_utils import split_dataset
from .dataset import TrainDataset, TestDataset


def preprocess_data(
    src_dir_path: str,
    human_ids: List[int],
    pose_fps: int,
    wav_sr: int,
    sample_duration: float,
    spt_dir_path: str,
    joints: List[str] = None
):
    assert os.path.exists(src_dir_path), "Source data not found at {}".format(src_dir_path)
    print("[Info] Create processed samples at", spt_dir_path)
    if os.path.exists(spt_dir_path):
        raise FileExistsError(f'Data alreday exists at {spt_dir_path}. Mannulay remove before recreating.')
    os.makedirs(spt_dir_path)
    split_dataset(
        src_dir_path=src_dir_path,
        human_ids=human_ids,
        pose_fps=pose_fps,
        wav_sr=wav_sr,
        sample_duration=sample_duration,
        spt_dir_path=spt_dir_path,
        joints=joints
    )


def load_processed_datasets(
    pose_fps: int,
    wav_sr: int,
    spt_dir_path: str,
    dst_dir_path: str,
    pose_window_len: int,
    pose_stride_len: int,
    pose_representation: str
):
    assert os.path.exists(spt_dir_path), "Splited data not found at {}".format(spt_dir_path)
    train_dataset = TrainDataset(
        samples_path=os.path.join(spt_dir_path, "train_samples.pkl"),
        dst_dir_path=dst_dir_path,
        pose_window_len=pose_window_len,
        pose_stride_len=pose_stride_len,
        pose_fps=pose_fps,
        wav_sr=wav_sr,
        pose_representation=pose_representation
    )
    val_dataset = TrainDataset(
        samples_path=os.path.join(spt_dir_path, "val_samples.pkl"),
        dst_dir_path=dst_dir_path,
        pose_window_len=pose_window_len,
        pose_stride_len=pose_window_len,
        pose_fps=pose_fps,
        wav_sr=wav_sr,
        pose_representation=pose_representation
    )
    test_dataset = TestDataset(
        samples_path=os.path.join(spt_dir_path, "test_samples.pkl"),
        dst_dir_path=dst_dir_path,
        pose_window_len=pose_window_len,
        pose_stride_len=pose_window_len,
        pose_fps=pose_fps,
        wav_sr=wav_sr,
        pose_representation=pose_representation
    )
    return train_dataset, val_dataset, test_dataset
