import os
import pickle
import numpy as np
import torch as th
import joblib as jl
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from .data_utils import euler2ortho6d, euler2log_rot, unroll_log_rot, resample_pose_seq


class TrainDataset(Dataset):
    def __init__(
        self,
        samples_path: str,
        dst_dir_path: str,
        pose_window_len: int,
        pose_stride_len: int,
        pose_fps: int,
        wav_sr: int,
        pose_representation: str
    ):
        data_path = os.path.join(
            dst_dir_path,
            os.path.basename(samples_path).replace(
                '_samples.pkl', '_data.pkl'
            )
        )
        if os.path.exists(data_path):
            print("[Info] Load data from", data_path)
            with open(data_path, "rb") as f:
                data = pickle.load(f)
            wavs = data["wav"]
            poses = data["pose"]
        else:
            print("[Info] Create data at", data_path)
            os.makedirs(os.path.dirname(data_path), exist_ok=True)
            with open(samples_path, "rb") as f:
                samples = pickle.load(f)
            hids = samples["hid"] # (N,)
            poses = samples["pose"] # (N,T,C)
            wavs = samples["wav"] # (N,T)
            
            # downsample poses
            duration_in_sec = wavs.shape[1] / wav_sr
            poses = np.array([
                resample_pose_seq(x, duration_in_sec, pose_fps) for x in poses
            ])

            # convert poses to specific representation
            N, T, _ = poses.shape
            if pose_representation == "6d":
                poses = np.concatenate(
                    [euler2ortho6d(x.reshape(-1, 3)).reshape(T, -1)
                     for x in poses], axis=0)
            elif pose_representation == "euler":
                pass
            elif pose_representation == "log_rot":
                poses = np.array([
                    euler2log_rot(x.reshape(-1, 3)).reshape(T, -1, 3)
                    for x in poses
                ]) # (N,T,-1,3)
                poses = np.array([
                    np.concatenate([
                        unroll_log_rot(x[:, i, :])
                        for i in range(x.shape[1])
                    ], axis=1) for x in poses
                ]).reshape(N, T, -1) # (N,T,C)
            else:
                raise ValueError(
                    f"Unsupported pose_representation {pose_representation}")

            # get scaler and scale pose
            scaler_path = os.path.join(dst_dir_path, 'scaler.jl')
            if 'train' in data_path:
                scaler = StandardScaler().fit(poses.reshape(N*T, -1))
                jl.dump(scaler, scaler_path)
            else:
                scaler = jl.load(scaler_path)
            poses = scaler.transform(poses.reshape(N*T, -1)).reshape(N, T, -1)

            # number of chunks for each sample
            num_chunks = int(np.ceil(poses.shape[1] / pose_stride_len))

            # pad sequence
            poses = np.concatenate([
                poses, np.zeros((poses.shape[0], pose_window_len, poses.shape[2]))
            ], axis=1)
            wav_window_len = int(pose_window_len / pose_fps * wav_sr)
            wavs = np.concatenate([wavs, np.zeros((wavs.shape[0], wav_window_len))], axis=1)

            # slice sequence
            # batch indices for all samples
            batch_idxs = np.concatenate([
                np.full((num_chunks, 1), i, dtype=int)
                for i in range(poses.shape[0])
            ], axis=0)

            # pose
            # sequence indices for all samples
            pose_seq_frames = np.concatenate([[
                    np.arange(
                        i * pose_stride_len,
                        i * pose_stride_len + pose_window_len
                    )
                    for i in range(num_chunks)
                ]
                for _ in range(poses.shape[0])
            ], axis=0)
            poses = poses[batch_idxs, pose_seq_frames]

            # wav
            # convert pose frame number to wav frame number
            # to ensure they are synced.
            wav_seq_frames = np.array([
                np.arange(
                    int(pose_frames[0] / pose_fps * wav_sr),
                    int(pose_frames[0] / pose_fps * wav_sr) + wav_window_len
                )
                for pose_frames in pose_seq_frames
            ])
            wavs = wavs[batch_idxs, wav_seq_frames]

            # save sliced samples
            obj = {"wav": wavs, "pose": poses}
            with open(data_path, "wb") as f:
                pickle.dump(obj, f)

        # self.hids = hids
        self.wavs = wavs
        self.poses = poses
        print("[Info] pose shape:", poses.shape, "| wav shape:", wavs.shape)

    def __len__(self):
        return self.wavs.shape[0]

    def __getitem__(self, index):
        # hid = th.from_numpy(self.hids[index]).long()
        return {
            # "hid": hid,
            "wav": th.from_numpy(self.wavs[index]).float(),
            "pose": th.from_numpy(self.poses[index]).float()
        }

    def get_dims(self):
        dims = {"d_pose": self.poses.shape[2]}
        print("[Info] dims:", dims)
        return dims

    def get_samples(self):
        return {
            # "hid": th.from_numpy(self.hids).long(),
            "pose": th.from_numpy(self.poses).float(),
            "wav": th.from_numpy(self.wavs).float()
        }


class TestDataset(TrainDataset):
    def __init__(
        self,
        samples_path: str,
        dst_dir_path: str,
        pose_window_len: int,
        pose_stride_len: int,
        pose_fps: int,
        wav_sr: int,
        pose_representation: str
    ):
        super().__init__(
            samples_path,
            dst_dir_path,
            pose_window_len,
            pose_stride_len,
            pose_fps,
            wav_sr,
            pose_representation
        )
        seq_data_path = os.path.join(
            dst_dir_path,
            os.path.basename(samples_path).replace(
                '_samples.pkl', '_seqs.pkl'
            ))
        if os.path.exists(seq_data_path):
            print("[Info] Load data from", seq_data_path)
            with open(seq_data_path, "rb") as f:
                data = pickle.load(f)
            hid_seqs = data["hid"]
            wav_seqs = data["wav"]
            pose_seqs = data["pose"]
        else:
            print("[Info] Create data at", seq_data_path)
            os.makedirs(os.path.dirname(seq_data_path), exist_ok=True)
            with open(samples_path, "rb") as f:
                samples = pickle.load(f)
            hid_seqs = samples["hid"] # (N,)
            wav_seqs = samples["wav"] # (N,T)
            poses_seqs = samples["pose"] # (N,T,C)
            
            # downsample poses
            duration_in_sec = wav_seqs.shape[1] / wav_sr
            poses_seqs = np.array([
                resample_pose_seq(x, duration_in_sec, pose_fps) for x in poses_seqs
            ])
            
            # convert poses to specific representation
            N, T, _ = poses_seqs.shape
            if pose_representation == "6d":
                poses_seqs = np.concatenate(
                    [euler2ortho6d(x.reshape(-1, 3)).reshape(T, -1)
                        for x in poses_seqs], axis=0)
            elif pose_representation == "euler":
                pass
            elif pose_representation == "log_rot":
                poses_seqs = np.array([
                    euler2log_rot(x.reshape(-1, 3)).reshape(T, -1, 3)
                    for x in poses_seqs
                ]) # (N,T,-1,3)
                poses_seqs = np.array([
                    np.concatenate([
                        unroll_log_rot(x[:, i, :])
                        for i in range(x.shape[1])
                    ], axis=1) for x in poses_seqs
                ]).reshape(N, T, -1) # (N,T,C)
            else:
                raise ValueError(
                    f"Unsupported pose_representation {pose_representation}")
            scaler = jl.load(os.path.join(dst_dir_path, 'scaler.jl'))
            pose_seqs = scaler.transform(poses_seqs.reshape(N*T, -1)).reshape(N, T, -1)
            # save sliced samples
            obj = {"hid": hid_seqs, "wav": wav_seqs, "pose": pose_seqs}
            with open(seq_data_path, "wb") as f:
                pickle.dump(obj, f)

        self.hid_seqs = hid_seqs
        self.wav_seqs = wav_seqs
        self.pose_seqs = pose_seqs
        print("[Info] pose seqs shape:", pose_seqs.shape, "| wav seqs shape:", wav_seqs.shape)

    def get_seqs(self):
        return {
            # "hid": th.from_numpy(self.samples_hid).long(),
            "pose": th.from_numpy(self.pose_seqs).float(),
            "wav": th.from_numpy(self.wav_seqs).float()
        }
