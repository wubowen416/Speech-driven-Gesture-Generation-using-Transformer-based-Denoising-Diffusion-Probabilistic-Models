import os
import json
import pickle
import librosa
import textgrid
import numpy as np
from glob import glob
from sklearn.model_selection import train_test_split
from typing import List, Tuple
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R
from models.modules.ha2g.model.vocab import Vocab as HA2GVocab
from .pymo.parsers import BVHParser
from .pymo.preprocessing import Numpyfier, JointSelector, DownSampler
from .pymo.rotation_tools import euler2expmap, expmap2euler
from sklearn.pipeline import Pipeline


def euler2rot_mat(euler, seq='XYZ'):
    """
    Convert from euler to SO3.
    :param euler: euler angle around axis. Numpy array with shape [N,3]
    :param seq: Order for rotation. Refer to https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.from_euler.html#scipy.spatial.transform.Rotation.from_euler
    :return: Rotation matrix. Numpy array with shape [N,9]
    """
    r = R.from_euler(seq, euler, degrees=True)
    return r.as_matrix().reshape(-1, 9)


def rot_mat2ortho6d(rot_mat):
    """ Convert from SO3 to 6D representation using (14) 
    :param rot_mat: Rotation matrix. Numpy array with shape [N, 9]
    :return: 6d representation. Numpy array with shape [N, 6]
    """
    return rot_mat.reshape(-1, 3, 3)[:, :, [0, 1]].reshape(-1, 6)


def euler2ortho6d(euler, seq='XYZ'):
    """
    :param euler: euler angle around axis. Numpy array with shape [N,3]
    :param seq: Order for rotation.
    :return: 6d representation. Numpy array with shape [N, 6]
    """
    return rot_mat2ortho6d(euler2rot_mat(euler, seq=seq))


def normalize_vector(v):
    """
    :param v: numpy array with shape [N,D]
    :return: normalized vector with shape [N,D]
    """
    v_mag = np.sqrt((v**2).sum(1))# batch
    v_mag = np.maximum(v_mag, 1e-8)
    v = v/v_mag[:, np.newaxis]
    return v
    

def cross_product(u, v):
    """
    :param u: numpy array with shape [N,3]
    :param v: numpy array with shape [N,3]
    :return: cross product of u, v. numpy array with shape [N,3]
    """
    batch = u.shape[0]
    i = u[:,1]*v[:,2] - u[:,2]*v[:,1]
    j = u[:,2]*v[:,0] - u[:,0]*v[:,2]
    k = u[:,0]*v[:,1] - u[:,1]*v[:,0]
    return np.concatenate((i.reshape(batch,1), j.reshape(batch,1), k.reshape(batch,1)),1)
        
    
def ortho6d2rot_mat(ortho6d):
    """
    :param ortho6d: 6D representation of SO3. numpy array of shape [N,6]
    :return: SO3 rotation matrix. numpy arrray of shape [N,9] 
    """
    ortho6d = ortho6d.reshape(-1, 3, 2)
    x_raw = ortho6d[:,:,0]#batch*3
    y_raw = ortho6d[:,:,1]#batch*3
        
    x = normalize_vector(x_raw) #batch*3
    z = cross_product(x,y_raw) #batch*3
    z = normalize_vector(z)#batch*3
    y = cross_product(z,x)#batch*3
        
    x = x.reshape(-1,3,1)
    y = y.reshape(-1,3,1)
    z = z.reshape(-1,3,1)
    return np.concatenate((x,y,z), 2) #batch*3*3


def ortho6d2euler(ortho6d, seq='XYZ'):
    """
    :param ortho6d: 6d representation. Numpy array with shape [N, 6]
    :return: euler angle around axis. Numpy array with shape [N, 3]
    """
    rot_mat = ortho6d2rot_mat(ortho6d)
    r = R.from_matrix(rot_mat)
    return r.as_euler(seq=seq, degrees=True)


def euler2log_rot(euler, seq='XYZ', use_deg=True) -> np.ndarray:
    """
    :param euler: euler angle around axis. Numpy array with shape [N,3]
    :param seq: Order for rotation.
    :return: log mapped rotation. Numpy array with shape [N,3]
    """
    return np.array([euler2expmap(f, seq, use_deg) for f in euler])


def log_rot2euler(log_rot, seq='XYZ', use_deg=True) -> np.ndarray:
    """
    :param log_rot: log mapped rotation. Numpy array with shape [N,3]
    :return: euler angle rotation. Numpy array with shape [N,3]
    """
    return np.array([expmap2euler(f, seq, use_deg) for f in log_rot])


def bvh_line2euler_line(line):
    def euler_str2float(angle: str) -> float:
        """ constrain an angle to (-180, 180] """
        def func(angle):
            while angle > 180 or angle < -180:
                if angle > 180:
                    angle -= 360
                elif angle <= -180:
                    angle += 360
            return angle
        angle = float(angle)
        return func(angle)
    return [euler_str2float(x) for x in line.split()]


def load_from_bvh_old(bvh_path: str) -> Tuple[np.ndarray, float]:
    """ Correspoding to BEAT dataset. """
    with open(bvh_path, "r") as f:
        lines = f.readlines()
    num_frames = int(lines[429].split(":")[1])
    duration_in_sec = float(lines[430].split(":")[1]) * num_frames
    del lines[:431]
    lines = list(map(bvh_line2euler_line, lines))
    # convert to numpy
    eulers = np.array(lines).reshape(-1, 76, 3)
    eulers = eulers[:, 1:] # exclude traslation
    # set joint"s value to 0 to avoid oscillation
    # refer to analysis/data_inspect.ipynb for details
    eulers[:, 8] = 0
    eulers[:, 9, 0] = 0
    eulers[:, 14, [0, 1]] = 0
    eulers[:, 15, [0, 1]] = 0
    eulers[:, 16] = 0
    eulers[:, 21] = 0
    eulers[:, 26] = 0
    eulers[:, 31] = 0
    eulers[:, 35] = 0
    eulers[:, 36, 0] = 0
    eulers[:, 41, [0, 1]] = 0
    eulers[:, 42, [0, 1]] = 0
    eulers[:, 43] = 0
    eulers[:, 48] = 0
    eulers[:, 53] = 0
    eulers[:, 58] = 0
    eulers[:, 62] = 0
    eulers[:, 66] = 0
    eulers[:, 67, [0, 1]] = 0
    eulers[:, 68, [0, 1]] = 0
    eulers[:, 70, [1, 2]] = 0
    eulers[:, 72] = 0
    eulers[:, 73, [1, 2]] = 0
    eulers[:, 74] = 0
    # flatten
    eulers = eulers.reshape(eulers.shape[0], -1) # -> (T, d_euler)   
    return eulers, duration_in_sec


def load_from_bvh(
    bvh_path: str, joints: List[str] = None, tgt_fps=20
) -> Tuple[np.ndarray, float]:
    parser = BVHParser()
    parsed_data = parser.parse(bvh_path)
    
    if parser.framerate != 0.008333:
        raise ValueError(f"Framerate exception: {parser.framerate}")
    
    if joints is None: # full body
        data_pipe = Pipeline([
            ('dwnspl', DownSampler(tgt_fps)),
            ('npf', Numpyfier())
        ])
    else:
        data_pipe = Pipeline([
            ('dwnspl', DownSampler(tgt_fps)),
            ('jntsl', JointSelector(joints, include_root=False)),
            ('npf', Numpyfier())
        ])

    piped_data = data_pipe.fit_transform([parsed_data])[0] # (T, C)
    
    if 'hips' in joints:
        piped_data = piped_data[:, 3:]  # exclude root translation
        
    print(f'Motion data shape: {piped_data.shape}')
    duration_in_sec = piped_data.shape[0] / tgt_fps
    return piped_data, duration_in_sec


def load_from_face(facial_path: str, facial_norm: bool = False, src_fps: int = 60, tgt_fps: int = 15):
    reduce_factor = int(src_fps / tgt_fps)
    facial_each_file = []
    with open(facial_path, 'r') as facial_data_file:
        facial_data = json.load(facial_data_file)
        print(len(facial_data['frames']))
        for i, frame_data in enumerate(facial_data['frames']):
            if i % reduce_factor == 0:
                if facial_norm:
                    # facial_each_file.append((frame_data['weights']-self.mean_facial) / self.std_facial)
                    raise NotImplementedError
                else:
                    facial_each_file.append(frame_data['weights'])
    facial_each_file = np.array(facial_each_file)
    print(f'Face data shape: {facial_each_file.shape}')
    return facial_each_file, len(facial_each_file) / tgt_fps


def split_dataset(
    src_dir_path: str, # /PATH/TO/BEAT
    word_vec_path: str,
    human_ids: List[int],
    wav_sr: int,
    sample_duration: int,
    spt_dir_path: str,
    joints: List[str] = None
):
    os.makedirs("./log", exist_ok=True)
    log = open("./log/split_dataset.txt", "w")

    # make vocab
    # get all words
    print("[Info] Building vocab...")
    all_words = []
    for hid_idx, hid in enumerate(human_ids):
        humand_dir_path = os.path.join(src_dir_path, str(hid))
        text_grid_paths = glob(os.path.join(humand_dir_path, "*.TextGrid"))
        for file_idx, text_grid_path in enumerate(text_grid_paths):
            print(f'[Info] hid: {hid_idx}/{len(human_ids)} | file: {file_idx}/{len(text_grid_paths)}')
            # load words from textgrid file
            tg = textgrid.TextGrid.fromFile(text_grid_path)
            for words_info in tg[0]:
                words = words_info.mark
                assert len(words.split(" ")) == 1
                if words == "":
                    continue
                all_words.append(words)
    all_words = set(all_words)

    # create vocab
    vocab = HA2GVocab("word")
    vocab.load_word_vectors(word_vec_path)
    for word in all_words:
        vocab.index_word(word)
    with open(os.path.join(spt_dir_path, "vocab.pkl"), 'wb') as f:
        pickle.dump(vocab, f)
    
    print("[Info] Chunking samples...")
    hid_list = []
    pose_sample_list = []
    face_sample_list = []
    wav_sample_list = []
    word_ids_sample_list = []
    for hid_idx, hid in enumerate(human_ids):
        humand_dir_path = os.path.join(src_dir_path, str(hid))
        bvh_paths = glob(os.path.join(humand_dir_path, "*.bvh"))

        # chunk recordings to shorter samples
        for file_idx, bvh_path in enumerate(bvh_paths):
            print(f'[Info] hid: {hid_idx}/{len(human_ids)} | file: {file_idx}/{len(bvh_paths)}')
            
            if '1_wayne_1_1_2' in bvh_path:
                continue # skip since cannot sync
            
            try:
                # get file paths and check validity
                wav_path = bvh_path.replace(".bvh", ".wav")
                if not os.path.exists(wav_path):
                    # continue
                    raise FileNotFoundError("Wav file not found.")

                text_grid_path = bvh_path.replace(".bvh", ".TextGrid")
                if not os.path.exists(text_grid_path):
                    # continue
                    raise FileNotFoundError("TextGrid file not found.")
                
                face_path = bvh_path.replace(".bvh", ".json")
                if not os.path.exists(text_grid_path):
                    # continue
                    raise FileNotFoundError("Json file not found.")

                # 1. load files
                # load bvh file
                pose_fps = 20
                poses, pose_duration_in_sec = load_from_bvh(bvh_path, joints, pose_fps)

                # load wav file
                wav, _ = librosa.load(wav_path, sr=wav_sr)
                wav_duration_in_sec = len(wav) / wav_sr
                print('Wav shape:', wav.shape)
                
                # load facial file
                face_fps = 15
                faces, face_duration_in_sec = load_from_face(face_path, tgt_fps=face_fps)
                
                # begin time correction
                base_time = 5
                pose_begin_time = wav_begin_time = face_begin_time = 5
                if '1_wayne_0_1_8' in bvh_path:
                    wav_begin_time += 0.3
                elif '1_wayne_0_9_16' in bvh_path:
                    wav_begin_time += 1.0
                elif '1_wayne_0_17_24' in bvh_path:
                    wav_begin_time += 0.5
                elif '1_wayne_0_25_32' in bvh_path:
                    wav_begin_time += 0.8
                elif '1_wayne_0_33_40' in bvh_path:
                    wav_begin_time += 0.5
                elif '1_wayne_0_41_48' in bvh_path:
                    wav_begin_time += 0.5
                elif '1_wayne_0_49_56' in bvh_path:
                    wav_begin_time += 1
                elif '1_wayne_0_57_64' in bvh_path:
                    wav_begin_time += 0.5
                elif '1_wayne_0_65_72' in bvh_path:
                    wav_begin_time += 0.3
                elif '1_wayne_0_73_80' in bvh_path:
                    wav_begin_time += 0.2
                elif '1_wayne_0_81_86' in bvh_path:
                    wav_begin_time += 0.5
                elif '1_wayne_0_87_94' in bvh_path:
                    wav_begin_time += 1
                elif '1_wayne_0_95_102' in bvh_path:
                    wav_begin_time += 0.5
                elif '1_wayne_0_103_110' in bvh_path:
                    wav_begin_time += 1.0
                elif '1_wayne_0_111_118' in bvh_path:
                    wav_begin_time += 0.7
                # elif '1_wayne_1_1_2' in bvh_path: # cannot sync
                #     wav_begin_time += 1.0
                elif '1_wayne_1_3_4' in bvh_path:
                    wav_begin_time += 1
                # elif '1_wayne_1_5_6' in bvh_path: # no textGrid file, no double check
                #     wav_begin_time += 0.666667
                elif '1_wayne_1_7_8' in bvh_path:
                    wav_begin_time += 0
                # elif '1_wayne_1_9_10' in bvh_path: # no textGrid file, no double check
                #     wav_begin_time += 0.5
                elif '1_wayne_1_11_12' in bvh_path:
                    wav_begin_time += 0.8
                
                poses = poses[int(pose_begin_time * pose_fps):]
                pose_duration_in_sec -= pose_begin_time
                wav = wav[int(wav_begin_time * wav_sr):]
                wav_duration_in_sec -= wav_begin_time
                faces = faces[int(face_begin_time * face_fps):]

                # load textgrid to word_ids
                # same fps with pose
                tg = textgrid.TextGrid.fromFile(text_grid_path)
                word_ids = np.zeros((len(poses),))
                for word_info in tg[0]:
                    word = word_info.mark
                    if word == "":
                        continue
                    sf = int((word_info.minTime - base_time) * pose_fps)
                    if sf < 0 :
                        continue
                    ef = int((word_info.maxTime - base_time) * pose_fps)
                    word_ids[sf:ef] = vocab.get_word_index(word)

                # compute duration to same length
                shorter_duration = min([pose_duration_in_sec, wav_duration_in_sec, face_duration_in_sec])
                poses = poses[:int(shorter_duration * pose_fps)]
                faces = faces[:int(shorter_duration) * face_fps]
                word_ids = word_ids[:int(shorter_duration * pose_fps)]
                wav = wav[:int(shorter_duration * wav_sr)]

                # 2. cut to seq
                pose_window_len = int(sample_duration * pose_fps)
                face_window_len = int(sample_duration * face_fps)
                wav_window_len = int(sample_duration * wav_sr)
                
                start_time = np.arange(0, shorter_duration, sample_duration)[:-1]
                end_time = [int(st + sample_duration) for st in start_time]
                
                pose_sample_idxs = [
                    np.arange(int(st * pose_fps), int(et * pose_fps))[:pose_window_len]
                    for st, et in zip(start_time, end_time)
                ]
                
                face_sample_idxs = [
                    np.arange(int(st * face_fps), int(et * face_fps))[:face_window_len]
                    for st, et in zip(start_time, end_time)
                ]
                
                word_sample_idxs = pose_sample_idxs
                
                wav_sample_idxs = [
                    np.arange(int(st * wav_sr), int(et * wav_sr))[:wav_window_len]
                    for st, et in zip(start_time, end_time)
                ]

                pose_samples = poses[pose_sample_idxs, ...]
                face_samples = faces[face_sample_idxs, ...]
                word_ids_samples = word_ids[word_sample_idxs, ...]
                wav_samples = wav[wav_sample_idxs, ...]

                # append samples
                hid_list.append(np.array([hid] * pose_samples.shape[0]))
                pose_sample_list.append(pose_samples)
                face_sample_list.append(face_samples)
                wav_sample_list.append(wav_samples)
                word_ids_sample_list.append(word_ids_samples)

                print("[Info] Processed:", bvh_path, file=log)

            except Exception as msg:
                print("[Error]", msg, bvh_path)
                print("[Error]", msg, bvh_path, file=log)

    log.close()

    # concat
    hids = np.concatenate(hid_list, axis=0) # (N,)
    poses = np.concatenate(pose_sample_list, axis=0) # (N,T,d)
    faces = np.concatenate(face_sample_list, axis=0) # (N,T,d)
    wavs = np.concatenate(wav_sample_list, axis=0) # (N,T)
    word_ids = np.concatenate(word_ids_sample_list, axis=0) # (N,T)

    # split sampls to train, val, test to 8:1:1, stratified by hid
    train_hids, test_hids, \
    train_poses, test_poses, \
    train_faces, test_faces, \
    train_wavs, test_wavs, \
    train_word_ids, test_word_ids = train_test_split(
        hids, poses, faces, wavs, word_ids, test_size=0.2, shuffle=True, stratify=hids, random_state=0
    )
    test_hids, val_hids, \
    test_poses, val_poses, \
    test_faces, val_faces, \
    test_wavs, val_wavs, \
    test_word_ids, val_word_ids = train_test_split(
        test_hids, test_poses, test_faces, test_wavs, test_word_ids, test_size=0.5, shuffle=True, stratify=test_hids, random_state=0
    )

    # save
    obj = {"hid": train_hids, "pose": train_poses, "face": train_faces, "wav": train_wavs, "word_id": train_word_ids}
    with open(os.path.join(spt_dir_path, "train_seqs.pkl"), "wb") as f:
        pickle.dump(obj, f)
    obj = {"hid": val_hids, "pose": val_poses, "face": val_faces, "wav": val_wavs, "word_id": val_word_ids}
    with open(os.path.join(spt_dir_path, "val_seqs.pkl"), "wb") as f:
        pickle.dump(obj, f)
    obj = {"hid": test_hids, "pose":test_poses, "face": test_faces, "wav": test_wavs, "word_id": test_word_ids}
    with open(os.path.join(spt_dir_path, "test_seqs.pkl"), "wb") as f:
        pickle.dump(obj, f)
        

def split_dataset_zip(
    src_dir_path: str, # /PATH/TO/BEAT
    word_vec_path: str,
    human_ids: List[int],
    wav_sr: int,
    sample_duration: int,
    spt_dir_path: str,
    joints: List[str] = None) -> None:
    """Split dataset according to https://github.com/PantoMatrix/BEAT/issues/6,
    and save them into pickle files.
    """
    
    os.makedirs("./log", exist_ok=True)
    log = open("./log/split_dataset.txt", "w")

    # make vocab
    # get all words
    all_words = []
    for hid_idx, hid in enumerate(human_ids):
        humand_dir_path = os.path.join(src_dir_path, str(hid))
        text_grid_paths = glob(os.path.join(humand_dir_path, "*.TextGrid"))
        for file_idx, text_grid_path in enumerate(text_grid_paths):
            print(f'[Info] Building vocab | hid: {hid_idx}/{len(human_ids)} | file: {file_idx}/{len(text_grid_paths)}')
            # load words from textgrid file
            tg = textgrid.TextGrid.fromFile(text_grid_path)
            for words_info in tg[0]:
                words = words_info.mark
                assert len(words.split(" ")) == 1
                if words == "":
                    continue
                all_words.append(words)

    # create vocab
    vocab = Vocab("word")
    for word in all_words:
        vocab.index_word(word)
    vocab.load_word_vectors(word_vec_path)
    with open(os.path.join(spt_dir_path, "vocab.pkl"), 'wb') as f:
        pickle.dump(vocab, f)
        
    # split data
    train_hid_list = []
    train_pose_sample_list = []
    train_wav_sample_list = []
    train_word_ids_sample_list = []
    val_hid_list = []
    val_pose_sample_list = []
    val_wav_sample_list = []
    val_word_ids_sample_list = []
    test_hid_list = []
    test_pose_sample_list = []
    test_wav_sample_list = []
    test_word_ids_sample_list = []
    for hid_idx, hid in enumerate(human_ids):
        humand_dir_path = os.path.join(src_dir_path, str(hid))
        bvh_paths = glob(os.path.join(humand_dir_path, "*.bvh"))
        
        # Get val and test indices
        if hid in [1, 2, 3, 4, 6, 7, 8, 9, 11, 21]:
            # 4 hours
            test_seq_idxs = [1, 2, 3, 4, 5, 6, 7, 8, 65, 73, 81, 87, 95, 103, 111]
            test_conv_idxs = [1]
            val_seq_idxs = [56, 57, 58, 59, 60, 61, 62, 63, 64, 72, 80, 86, 94, 102, 110, 118]
            val_conv_idxs = [12]
        else:
            # 1 hours data file indices
            raise NotImplementedError

        # process one sample, chunk to shorter samples
        for file_idx, bvh_path in enumerate(bvh_paths):
            print(f'[Info] Spliting data | hid: {hid_idx}/{len(human_ids)} | file: {file_idx}/{len(bvh_paths)}')
            try:
                # check availability
                wav_path = bvh_path.replace(".bvh", ".wav")
                if not os.path.exists(wav_path):
                    # continue
                    raise FileNotFoundError("Wav file not found.")

                text_grid_path = bvh_path.replace(".bvh", ".TextGrid")
                if not os.path.exists(text_grid_path):
                    # continue
                    raise FileNotFoundError("TextGrid file not found.")

                # 1. load files
                # load bvh file
                pose_fps = 20
                poses, pose_duration_in_sec = load_from_bvh(bvh_path, joints, pose_fps)

                # load wav file
                wav, _ = librosa.load(wav_path, sr=wav_sr)
                wav_duration_in_sec = len(wav) / wav_sr
                
                # load textgrid to word_ids
                # same fps with pose
                tg = textgrid.TextGrid.fromFile(text_grid_path)
                word_ids = np.zeros((len(poses),))
                for word_info in tg[0]:
                    word = word_info.mark
                    if words == "":
                        continue
                    sf = int(word_info.minTime * pose_fps)
                    # ef = int(np.round(word_info.maxTime * wav_sr))
                    word_ids[sf] = vocab.get_word_index(word)

                # compute duration to sync
                shorter_duration = min([pose_duration_in_sec, wav_duration_in_sec])
                poses = poses[:int(shorter_duration * pose_fps)]
                word_ids = word_ids[:int(shorter_duration * pose_fps)]
                wav = wav[:int(shorter_duration * wav_sr)]

                # 2. slice
                pose_window_len = int(sample_duration * pose_fps)
                wav_window_len = int(sample_duration * wav_sr)
                
                start_time = np.arange(0, shorter_duration, sample_duration)[:-1]
                end_time = [int(st + sample_duration) for st in start_time]
                
                pose_sample_idxs = [
                    np.arange(int(st * pose_fps), int(et * pose_fps))[:pose_window_len]
                    for st, et in zip(start_time, end_time)
                ]
                
                wav_sample_idxs = [
                    np.arange(int(st * wav_sr), int(et * wav_sr))[:wav_window_len]
                    for st, et in zip(start_time, end_time)
                ]

                pose_samples = poses[pose_sample_idxs, ...]
                word_ids_samples = word_ids[pose_sample_idxs, ...]
                wav_samples = wav[wav_sample_idxs, ...]
                hid_samples = np.array([hid] * pose_samples.shape[0])
                
                # split according to filename
                _, _, ty, idx1, idx2 = os.path.splitext(os.path.basename(bvh_path))[0].split('_')
                assert idx1 == idx2, 'last two numbers of file name must agree.'
                idx1 = int(idx1)
                if ty == '0':
                    if idx1 in test_seq_idxs:
                        test_hid_list.append(hid_samples)
                        test_pose_sample_list.append(pose_samples)
                        test_wav_sample_list.append(wav_samples)
                        test_word_ids_sample_list.append(word_ids_samples)
                    elif idx1 in val_seq_idxs:
                        val_hid_list.append(hid_samples)
                        val_pose_sample_list.append(pose_samples)
                        val_wav_sample_list.append(wav_samples)
                        val_word_ids_sample_list.append(word_ids_samples)
                    else:
                        train_hid_list.append(hid_samples)
                        train_pose_sample_list.append(pose_samples)
                        train_wav_sample_list.append(wav_samples)
                        train_word_ids_sample_list.append(word_ids_samples)
                elif ty == '1':
                    if idx1 in test_conv_idxs:
                        test_hid_list.append(hid_samples)
                        test_pose_sample_list.append(pose_samples)
                        test_wav_sample_list.append(wav_samples)
                        test_word_ids_sample_list.append(word_ids_samples)
                    elif idx1 in val_conv_idxs:
                        val_hid_list.append(hid_samples)
                        val_pose_sample_list.append(pose_samples)
                        val_wav_sample_list.append(wav_samples)
                        val_word_ids_sample_list.append(word_ids_samples)
                    else:
                        train_hid_list.append(hid_samples)
                        train_pose_sample_list.append(pose_samples)
                        train_wav_sample_list.append(wav_samples)
                        train_word_ids_sample_list.append(word_ids_samples)
                else:
                    raise ValueError(f'Unsupported recording type -> {ty}')
                
            except Exception as msg:
                print(f"[Error] msg: {msg} | file: {bvh_path}")
                print(f"[Error] msg: {msg} | file: {bvh_path}", file=log)
    
    log.close()

    # concat
    train_hids = np.concatenate(train_hid_list, axis=0) # (N,)
    train_poses = np.concatenate(train_pose_sample_list, axis=0) # (N,T,d)
    train_wavs = np.concatenate(train_wav_sample_list, axis=0) # (N,T)
    train_word_ids = np.concatenate(train_word_ids_sample_list, axis=0) # (N,T)
    val_hids = np.concatenate(val_hid_list, axis=0) # (N,)
    val_poses = np.concatenate(val_pose_sample_list, axis=0) # (N,T,d)
    val_wavs = np.concatenate(val_wav_sample_list, axis=0) # (N,T)
    val_word_ids = np.concatenate(val_word_ids_sample_list, axis=0) # (N,T)
    test_hids = np.concatenate(test_hid_list, axis=0) # (N,)
    test_poses = np.concatenate(test_pose_sample_list, axis=0) # (N,T,d)
    test_wavs = np.concatenate(test_wav_sample_list, axis=0) # (N,T)
    test_word_ids = np.concatenate(test_word_ids_sample_list, axis=0) # (N,T)

    # save
    obj = {"hid": train_hids, "pose": train_poses, "wav": train_wavs, "word_id": train_word_ids}
    with open(os.path.join(spt_dir_path, "train_seqs.pkl"), "wb") as f:
        pickle.dump(obj, f)
    obj = {"hid": val_hids, "pose": val_poses, "wav": val_wavs, "word_id": val_word_ids}
    with open(os.path.join(spt_dir_path, "val_seqs.pkl"), "wb") as f:
        pickle.dump(obj, f)
    obj = {"hid": test_hids, "pose":test_poses, "wav": test_wavs, "word_id": test_word_ids}
    with open(os.path.join(spt_dir_path, "test_seqs.pkl"), "wb") as f:
        pickle.dump(obj, f)


def resample_pose_seq(poses, duration_in_sec, tgt_fps):
    n = len(poses)
    x = np.arange(0, n)
    y = poses
    f = interp1d(x, y, axis=0, kind="linear", fill_value="extrapolate")
    expected_n = duration_in_sec * tgt_fps
    x_new = np.arange(0, n, n / expected_n)
    interpolated_y = f(x_new)
    if hasattr(poses, "dtype"):
        interpolated_y = interpolated_y.astype(poses.dtype)
    return interpolated_y
