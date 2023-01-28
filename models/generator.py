from typing import Tuple
import numpy as np
from .model import Speech2GestureModelBase, Speech2GestureModelInpaint
from .modules.respace import GaussianSpacedDiffusion
import torch as th


class Generator:
    def __init__(
        self,
        model: Speech2GestureModelBase,
        diffusion: GaussianSpacedDiffusion,
    ) -> None:
        self.model = model
        self.diffusion = diffusion
        
    @th.no_grad()
    def gpu_warm_up_ddim(
        self,
        shape,
        model_kwargs,
        device,
        num_iteration: int = 10
    ):
        for i in range(num_iteration):
            print(f'[Info] Warm up step: {i}/{num_iteration}')
            self.diffusion.ddim_sample_loop(
                self.model,
                shape,
                model_kwargs=model_kwargs,
                device=device
            )
            
    def _choose_sample_func(
        self,
        sample_alg: str
    ):
        # choose sample algorithm
        if sample_alg == 'ddim':
            sample_func = self.diffusion.ddim_sample_loop
        elif sample_alg == 'ddpm':
            sample_func = self.diffusion.p_sample_loop
        else:
            raise ValueError(f'Unsupported sample algorithm: {sample_alg}')
        return sample_func
        
    @th.no_grad()
    def eval_infer_time_ddim(
        self, 
        shape, 
        model_kwargs,
        sample_alg: str = 'ddim',
        repetitions: int = 10,
        device='cpu'
    ):
        sample_func = self._choose_sample_func(sample_alg)
        starter = th.cuda.Event(enable_timing=True)
        ender = th.cuda.Event(enable_timing=True)
        timings = np.zeros((repetitions, 1))
        self.gpu_warm_up_ddim(shape, model_kwargs, device)
        with th.no_grad():
            for rep in range(repetitions):
                print(f'[Info] Current rep: {rep}/{repetitions}')
                starter.record()
                sample_func(
                    self.model,
                    shape,
                    model_kwargs=model_kwargs,
                    device=device,
                    progress=False
                )
                ender.record()
                th.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)
                timings[rep] = curr_time
        mean_syn = np.sum(timings) / repetitions
        std_syn = np.std(timings)
        return mean_syn, std_syn

    @th.no_grad()
    def generate_sequence(
        self,
        wav_seqs: th.Tensor,
        wav_sr: int,
        pose_dim: int,
        pose_fps: int,
        pose_window_len: int,
        pose_seed_len: int,
        return_dtype: str = 'tensor',
        smooth_trans: bool = True,
        trans_factor: float = None,
        init_poses: th.Tensor = None,
        sample_alg: str = 'ddim',
        batch_size: int = 64,
        device: str = 'cpu',
        progress: bool = True
    ):  
        assert len(wav_seqs.shape) == 2, 'Provide batch dimension'
        if init_poses is not None:
            assert len(init_poses.shape) == 3, 'Provide batch dimension'
            assert len(init_poses) == len(wav_seqs), 'Init pose batch size does not meet wav_seqs.'
            init_poses = init_poses.to(device)

        wav_seqs = wav_seqs.to(device)
        init_poses = init_poses.to(device)

        num_seq = len(wav_seqs)
        num_batches = int(np.ceil(num_seq / batch_size))
        seq_len = wav_seqs.shape[1] // wav_sr * pose_fps
        wav_seq_len = wav_seqs.shape[1]
        pose_stride_len = pose_window_len - pose_seed_len
        num_division = int(np.ceil(seq_len / pose_stride_len))
        if (seq_len - pose_seed_len) % pose_stride_len == 0:
            num_division -= 1
        wav_window_len = int(wav_sr * pose_window_len / pose_fps)
        
        outs = []
        for idx_batch in range(num_batches):
            print(f"[Info] Processing batch {idx_batch+1}/{num_batches}")

            # prepare batch data
            wav_seq = wav_seqs[
                idx_batch * batch_size:(idx_batch + 1) * batch_size
            ].to(device) # (N, T_wav)

            # generate each chunk for whole sequence iteratively
            # start and end frame for 1st chunk
            wav_start_frame = 0
            wav_end_frame = wav_start_frame + wav_window_len
            pose_start_frame = 0

            # start generation chunk by chunk
            samples = []
            for idx_div in range(num_division):
                print(f"[Info] Processing division {idx_div+1}/{num_division}")

                # prepare inputs
                wavs = wav_seq[:, wav_start_frame:wav_end_frame]
                inpaint_masks = th.ones((len(wavs), pose_window_len, 1), device=device)
                inpaint_masks[:, pose_seed_len:] = 0
                if idx_div == 0:
                    if init_poses is None:
                        inpaint_poses = inpaint_masks = None
                    else:
                        inpaint_poses = th.zeros((len(wavs), pose_window_len, pose_dim), device=device)
                        inpaint_poses[:, :pose_seed_len] = init_poses
                else:
                    inpaint_poses[:, :pose_seed_len] = sample[:, -pose_seed_len:] # (N,T,C)

                # padding for last division
                if wav_end_frame > wav_seq_len:
                    wavs = th.cat([
                        wavs,
                        th.zeros((len(wavs), wav_end_frame - wav_seq_len), device=device)], dim=1)

                sample = self.generate_sample(
                    (len(wavs), pose_dim, pose_window_len),
                    wavs,
                    inpaint_poses=inpaint_poses,
                    inpaint_masks=inpaint_masks,
                    sample_alg=sample_alg,
                    trans_factor=trans_factor,
                    pose_seed_len=pose_seed_len,
                    device=device,
                    progress=progress
                )

                # append generated sample
                samples.append(sample)

                # update start and end frame
                wav_start_frame = int(pose_start_frame / pose_fps * wav_sr)
                wav_end_frame = wav_start_frame + wav_window_len
                pose_start_frame += pose_stride_len

            # combine chunks
            combined = []
            for i, x in enumerate(samples):
                if smooth_trans:
                    if i > 0:
                        ratio = th.arange(
                            0, 1, 1/pose_seed_len, device=device
                        )[:pose_seed_len].view(1, -1, 1)
                        trans_x = x[:, :pose_seed_len] * ratio + \
                            samples[i-1][:, -pose_seed_len:] * (1 - ratio)
                        x = th.cat([trans_x, x[:, pose_seed_len:]], dim=1)
                if i < len(samples) - 1:
                    combined.append(x[:, :-pose_seed_len])
                else:
                    combined.append(x)
            combined = th.cat(combined, dim=1)[:, :seq_len]
            outs.append(combined)
        outs = th.concat(outs, dim=0)  # (N,T,C)
        outs = self.tensor2dtype(outs, return_dtype)
        return outs

    @th.no_grad()
    def eval_bpd(
        self,
        poses: th.Tensor,
        wavs: th.Tensor,
        pose_seed_len: int = None,
    ):
        model_kwargs = {"wav": wavs} # (N,T_wav)
        if isinstance(self.model, Speech2GestureModelInpaint):
            assert pose_seed_len is not None, 'Provide pose_seed_len for inpaint model.'
            inpaint_poses = poses.clone()
            inpaint_masks = th.ones_like(inpaint_poses)[:, :, :1] # (N,T,1)
            inpaint_masks[:, pose_seed_len:] = 0
            model_kwargs['inpaint_pose'] = inpaint_poses.transpose(0, 1) # (T,N,C)
            model_kwargs['inpaint_mask'] = inpaint_masks.transpose(0, 1) # (T,N,1)
        return self.diffusion.calc_bpd_loop(
            self.model,
            x_start=poses.transpose(1, 2), # -> (N,C,T)
            model_kwargs=model_kwargs
        )

    @th.no_grad()
    def generate_sample(
        self,
        shape: Tuple[int],  # (N,C,T)
        wavs: th.Tensor,  # (N,T)
        noise: th.Tensor = None,
        inpaint_poses: th.Tensor = None,  # (N,T,C)
        inpaint_masks: th.Tensor = None,  # (N,T,1)
        sample_alg: str = 'ddim',  # ['ddim', 'ddpm']
        trans_factor: float = None,
        pose_seed_len: int = None,
        return_dtype: str = 'tensor',  # ['tensor', 'cpu_tensor', 'array']
        device: str = 'cpu',
        progress: bool = True
    ):
        # move to device
        wavs = wavs.to(device)
        if inpaint_poses is not None:
            assert inpaint_masks is not None, 'Provide inpaint_masks.'
            inpaint_poses = inpaint_poses.to(device)
            inpaint_masks = inpaint_masks.to(device)

        assert len(wavs.shape) == 2, f'Wav dim should be (N,T). Got: {wavs.shape}'
        assert len(shape) == 3, f'Shape should be (N,C,T). Got: {shape}'
        
        # model additional input
        model_kwargs = {'wav': wavs}
        if isinstance(self.model, Speech2GestureModelInpaint):
            assert len(inpaint_poses.shape) == 3
            assert len(inpaint_masks.shape) == 3
            assert inpaint_masks.size()[:2] == inpaint_poses.size()[:2]
            model_kwargs['inpaint_pose'] = inpaint_poses.transpose(0, 1) # -> (T,N,C)
            model_kwargs['inpaint_mask'] = inpaint_masks.transpose(0, 1) # -> (T,N,1)

        sample_func = self._choose_sample_func(sample_alg)

        # denoise function
        denoise_fn = None
        if inpaint_poses is not None:
            assert inpaint_masks is not None, 'Provide inpaint_masks for inpainting.'
            if trans_factor is not None:
                assert trans_factor >= 0 and trans_factor <= 1
                assert pose_seed_len is not None, \
                    'Provide pose_seed_len when using trans_factor.'
                trans_factor = th.arange(
                    trans_factor, 1, (1 - trans_factor)/pose_seed_len, device=device
                )[None, :, None]
                trans_factor = th.cat([
                    trans_factor, 
                    th.ones((1, shape[2] - trans_factor.size(1), 1), device=device)
                ], dim=1)
            else:
                trans_factor = 0
                
            def denoise_fn(pred_x_start):
                """
                :param pred_x_start: tensor, shape [N,C,T]
                :return: tensor, shape [N,C,T]
                """
                pred_x_start = pred_x_start.transpose(1, 2) # -> (N,T,C)
                pred_x_start = (1 - trans_factor) * inpaint_masks * inpaint_poses + \
                    trans_factor * inpaint_masks * pred_x_start + \
                        (1 - inpaint_masks) * pred_x_start
                return pred_x_start.transpose(1, 2) # -> (N,C,T)

        # generate
        if noise is None:
            noise = th.randn(shape, device=device)
        sample = sample_func(
            self.model,
            shape,
            noise=noise,
            denoise_fn=denoise_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress
        )['sample'].transpose(1, 2) # -> (N,T,C)
        sample = self.tensor2dtype(sample, return_dtype)
        return sample

    @staticmethod
    def tensor2dtype(x: th.Tensor, dtype: str):
        # convert type
        if dtype == 'tensor':
            pass
        elif dtype == 'cpu_tensor':
            x = x.cpu()
        elif dtype == 'array':
            x = x.cpu().numpy()
        else:
            raise ValueError(f'Unsupported dtype: {dtype}')
        return x
