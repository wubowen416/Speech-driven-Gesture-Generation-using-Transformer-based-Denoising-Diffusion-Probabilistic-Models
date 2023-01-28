import os
import time
import json
import pickle
import numpy as np
import torch as th
import torch.multiprocessing as mp
from torch.distributed import destroy_process_group
from argparse import ArgumentParser
from datasets.data_utils import PoseTypeConverter
from datasets.dataset_creation import preprocess_data, load_processed_datasets
from models.model_creation import create_model
from models.model import Speech2GestureModelInpaint
from models.trainer import Trainer, create_training_data
from utils.json_config import JsonConfig
from utils.pytorch_ddp import ddp_setup
from utils.seed import fix_seed
from utils.string_parser import parse_steps
from models.eval_utils import beat_consistency_score, beat_recall_score
from models.generator import Generator


def preprocess(config):
    # cut to 1min samples
    preprocess_data(
        src_dir_path=config.Data.src_dir_path,
        human_ids=config.Data.human_ids,
        pose_fps=config.Data.pose_fps,
        wav_sr=config.Data.wav_sr,
        sample_duration=config.Data.sample_duration,
        spt_dir_path=config.Data.spt_dir_path,
        joints=config.Data.get('joints'))


def load_torch_datasets(config):
    # cut to shorter data for training
    return load_processed_datasets(
        pose_fps=config.Data.pose_fps,
        wav_sr=config.Data.wav_sr,
        spt_dir_path=config.Data.spt_dir_path,
        dst_dir_path=config.Data.dst_dir_path,
        pose_window_len=config.Data.pose_window_len,
        pose_stride_len=config.Data.pose_stride_len,
        pose_representation=config.Data.pose_representation
    )


def train_model(rank, world_size, config):
    ddp_setup(rank, world_size)

    # load dataset
    train_dataset, val_dataset, _ = load_torch_datasets(config)

    # model
    model, diffusion, optimizer, schedule_sampler, lr_scheduler = create_model(
        d_pose=train_dataset.get_dims()["d_pose"],
        model_params=config.Model,
        lr=config.Train.get("lr"),
        weight_decay=config.Train.get("weight_decay"),
        scheduler_params=config.Train.get("Scheduler"),
        is_training=True
    )
    # training objs
    train_data, val_data = create_training_data(
        train_dataset,
        val_dataset,
        config.Train.batch_size
    )
    trainer = Trainer(
        gpu_id=rank,
        model=model,
        train_data=train_data,
        val_data=val_data,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        diffusion=diffusion,
        schedule_sampler=schedule_sampler,
        metric=config.Train.metric,
        goal=config.Train.goal,
        project=config.Meta.project,
        name=config.Meta.name,
        log_dir=config.Meta.log_dir,
        loss_params=config.Train.get("Loss"),
        seed=config.Meta.seed,
        config=config,
    )
    max_epochs = round(
        parse_steps(config.Train.max_training_steps
    ) / len(train_data))
    early_stop_threshold_in_epoch = round(
        parse_steps(config.Train.early_stop_threshold_in_step
    ) / len(train_data))
    print(f"[Info] Max epochs: {max_epochs} " + \
        f"| Ealry stop threshold (epoch): {early_stop_threshold_in_epoch}")
    trainer.train(max_epochs, early_stop_threshold_in_epoch)
    destroy_process_group()


def load_eval_objs(config, device="cpu"):
    # Load data and model
    # load dataset
    _, _, test_dataset = load_torch_datasets(config)

    # model
    model, diffusion, _, _, _ = create_model(
        d_pose=test_dataset.get_dims()["d_pose"],
        model_params=config.Model,
        is_training=False
    )
    model = model.to(device)
    model.eval()
    # load checkpoint, load one of chkpts as they are same
    print("[Info] Load chkpt from {}".format(config.Meta.chkpt_path))
    chkpt = th.load(config.Meta.chkpt_path, map_location=th.device(device))
    model.load_state_dict(chkpt["model_state_dict"])
    
    # compile model for torch2.0+
    # not compatible with cuda11.8. (2023.1)
    # if int(th.__version__[0]) >= 2:
    #     print(f"[Info] PyTorch version greater than 2. Use torch.compile.")
    #     model = th.compile(model)

    generator = Generator(model, diffusion)

    return chkpt, test_dataset, generator


def get_device():
    return "cuda" if th.cuda.is_available() else "cpu"


@th.no_grad()
def eval_infer_time(config):
    device = get_device()
    _, dataset, generator = load_eval_objs(config, device)
    samples = dataset.get_samples()
    poses = samples['pose'][:1].to(device)
    wavs = samples['wav'][:1].to(device)
    t_mean, t_std = generator.eval_infer_time_ddim(
        poses.transpose(1, 2).size(),
        {'wav': wavs},
        device=device
    )
    print(t_mean, t_std)
    

@th.no_grad()
def evaluate(config):
    # evaluate on a single GPU
    device = get_device()
    chkpt, dataset, generator = load_eval_objs(config, device)
    ptc = PoseTypeConverter(
        os.path.join(config.Data.dst_dir_path, 'scaler.jl'), 
        config.Data.hierarchy_path
    )

    # load samples
    samples = dataset.get_samples()
    # poses = samples["pose"] # (N,T_pose,d_pose)
    # wavs = samples["wav"] # (N,T_wav)

    # mini-batch evaluation: CUDA out of memeory so use mini-batch
    batch_size = 64
    num_batches = int(np.ceil(len(samples["pose"]) / batch_size))
    metrics = {}
    output_all = []
    for i in range(num_batches):
        st = time.perf_counter()
        # prepare batch data
        poses = samples["pose"][
            i * batch_size:(i + 1) * batch_size].to(device) # (N,T_pose,C)
        wavs = samples["wav"][
            i * batch_size:(i + 1) * batch_size].to(device) # (N,T_wav)
    
        diffusion_terms = generator.eval_bpd(
            poses, wavs, config.Model.get('pose_seed_len')
        )
        # accumulate diffusion values
        for name, value in diffusion_terms.items():
            value = value.mean().item() / num_batches
            if name not in metrics:
                metrics[name] = value
            else:
                metrics[name] += value

        # generate results
        inpaint_poses = inpaint_masks = None
        if isinstance(generator.model, Speech2GestureModelInpaint):
            inpaint_poses = poses.clone() # (N,T,C)
            inpaint_masks = th.ones_like(inpaint_poses)[:, :, :1]
            inpaint_masks[:, config.Model.Generate.pose_seed_len:] = 0
            
        out = generator.generate_sample(
            poses.transpose(1, 2).shape,
            wavs,
            inpaint_poses=inpaint_poses, 
            inpaint_masks=inpaint_masks,
            sample_alg='ddim',
            trans_factor=config.Model.Generate.get('trans_factor'),
            pose_seed_len=config.Model.Generate.pose_seed_len,
            return_dtype='array',
            device=device
        )

        # convert to dir_vec for evaluation
        if config.Data.pose_representation == '6d':
            out_dir_vec = ptc.scaled_ortho6d_to_dir_vec(out)
            dir_vec = ptc.scaled_ortho6d_to_dir_vec(poses.cpu().numpy())
        elif config.Data.pose_representation == 'log_rot':
            out_dir_vec = ptc.scaled_log_rot_to_dir_vec(out)
            dir_vec = ptc.scaled_log_rot_to_dir_vec(poses.cpu().numpy())
        elif config.Data.pose_representation == 'euler':
            out_dir_vec = ptc.scaled_euler_to_dir_vec(out)
            dir_vec = ptc.scaled_euler_to_dir_vec(poses.cpu().numpy())
        else:
            raise ValueError(f'Unsupported pose repr: {config.Data.pose_representation}')

        beat_consistency = beat_consistency_score(
            out_dir_vec.reshape(*out_dir_vec.shape[:2], -1, 3),
            config.Data.pose_fps,
            ptc.angle_pairs,
            wavs.cpu().numpy(),
            config.Data.wav_sr
        ) / num_batches
        beat_recall = beat_recall_score(
            out_dir_vec.reshape(*out_dir_vec.shape[:2], -1, 3),
            dir_vec,
            config.Data.pose_fps,
            ptc.angle_pairs
        ) / num_batches

        if "beat_consistency" not in metrics:
            metrics["beat_consistency"] = beat_consistency
        else:
            metrics["beat_consistency"] += beat_consistency
        if "beat_recall" not in metrics:
            metrics["beat_recall"] = beat_recall
        else:
            metrics["beat_recall"] += beat_recall

        output_all.append(out)

        print(f"[Info] Processing batch {i+1}/{num_batches} | " + \
                f"Elapsed time: {time.perf_counter()-st:.2f}")

    # prepend 'test/'
    test_log_dict = {}
    for name, value in metrics.items():
        test_log_dict[f"test/{name}"] = value

    # write results to file
    result_dir_path = os.path.join(
        config.Meta.log_dir, config.Meta.name, "results"
    )
    os.makedirs(result_dir_path, exist_ok=True)
    with open(os.path.join(result_dir_path, "eval_results.json"), "w") as f:
        json.dump(test_log_dict, f, indent=2)

    # save pose results
    output_all = np.concatenate(output_all, axis=0)
    generated = {
        "out": output_all,
        "pose": samples["pose"].numpy(),
        "wav": samples["wav"].numpy()
    }
    with open(os.path.join(result_dir_path, "generated.pkl"), "wb") as f:
        pickle.dump(generated, f)

    # log to wandb
    if True:
        import wandb
        wandb.init(
            project=config.Meta.project,
            id=chkpt["wandb_id"],
            resume="must",
            dir=os.path.join(config.Meta.log_dir, config.Meta.name))
        wandb.log(test_log_dict)
        wandb.finish()


@th.no_grad()
def generate(config):
    # evaluate on a single GPU
    device = get_device()
    _, dataset, generator = load_eval_objs(config, device)
    ptc = PoseTypeConverter(
        os.path.join(config.Data.dst_dir_path, 'scaler.jl'), 
        config.Data.hierarchy_path
    )

    # get seq
    seqs = dataset.get_seqs()
    pose_seqs = seqs["pose"] # (N,T_pose,C) (23, 1200, 225)
    wav_seqs = seqs["wav"] # (N,T_wav) (23, 960000)

    out_seqs = generator.generate_sequence(
        wav_seqs,
        config.Data.wav_sr,
        dataset.get_dims()['d_pose'],
        config.Data.pose_fps,
        config.Data.pose_window_len,
        config.Model.Generate.pose_seed_len,
        return_dtype='array',
        smooth_trans=config.Model.Generate.get('smooth_transition'),
        trans_factor=config.Model.Generate.get('trans_factor'),
        init_poses=pose_seqs[:, :config.Model.Generate.pose_seed_len],
        device=device
    )

    # save generated seqs
    generated_dir_path = os.path.join(
        config.Meta.log_dir, config.Meta.name, "results/samples"
    )
    os.makedirs(generated_dir_path, exist_ok=True)
    for i, out_seq in enumerate(out_seqs):
        pose_seq = pose_seqs[i].numpy()
        if config.Data.pose_representation == "6d":
            out_seq = ptc.scaled_ortho6d_to_euler(out_seq)
            pose_seq = ptc.scaled_ortho6d_to_euler(pose_seq)
        elif config.Data.pose_representation == "log_rot":
            out_seq = ptc.scaled_log_rot_to_euler(out_seq)
            pose_seq = ptc.scaled_log_rot_to_euler(pose_seq)
        elif config.Data.pose_representation == "euler":
            pass
        else:
            raise ValueError(f"Unsupported pose_representation {config.Data.pose_representation}")
        obj = {
            "pose": pose_seq,
            "wav": wav_seqs[i].numpy(),
            "out": out_seq
        }
        print(f"[Info] Saved to {os.path.join(generated_dir_path, f'sample_{i}.pkl')}")
        with open(os.path.join(generated_dir_path, f"sample_{i}.pkl"), "wb") as f:
            pickle.dump(obj, f)


def main():
    parser = ArgumentParser()
    parser.add_argument("--phase", type=str, required=True, help="Select from [prep, data, train, eval, gen].")
    parser.add_argument("--config", type=str, metavar="PATH", required=True, help="Path to config file.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    args = parser.parse_args()
    config = JsonConfig(args.config)
    config["Meta"].update({
        "phase": args.phase,
        "config_path": args.config,
        "seed": args.seed
    })
    fix_seed(args.seed)

    # execute
    if args.phase == "prep":
        preprocess(config)
    elif args.phase == "data":
        load_torch_datasets(config)
    elif args.phase == "train":
        if config.Train.world_size == "auto":
            world_size = th.cuda.device_count()
        else:
            world_size = config.Train.world_size
        assert world_size > 0
        mp.spawn(train_model, args=(world_size, config), nprocs=world_size)
    elif args.phase == "eval":
        evaluate(config)
    elif args.phase == "eval-time":
        eval_infer_time(config)
    elif args.phase == "gen":
        generate(config)
    else:
        raise ValueError(f"phase {args.phase} not supported.")


if __name__ == "__main__":
    main()