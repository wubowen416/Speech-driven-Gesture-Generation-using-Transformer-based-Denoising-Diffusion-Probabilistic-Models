import os
import copy
import time
import wandb
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from utils.json_config import JsonConfig
from argparse import Namespace
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from .model import Speech2GestureModel, Speech2GestureModelInpaint
from .modules.gaussian_diffusion import GaussianDiffusion
from .modules.resample import ScheduleSampler


def create_training_data(
    train_dataset: Dataset,
    val_dataset: Dataset,
    batch_size: int
):
    train_data = DataLoader(
        train_dataset,
        batch_size=batch_size,
        pin_memory=True,
        sampler=DistributedSampler(train_dataset, shuffle=True, drop_last=True)
    )
    val_data = DataLoader(
        val_dataset,
        batch_size=batch_size,
        pin_memory=True,
        sampler=DistributedSampler(val_dataset, shuffle=False, drop_last=True)
    )

    return train_data, val_data


class Trainer:
    def __init__(
        self,
        gpu_id: int,
        model: Speech2GestureModel,
        train_data: DataLoader,
        val_data: DataLoader,
        optimizer: th.optim.Optimizer,
        lr_scheduler: th.optim.lr_scheduler._LRScheduler,
        diffusion: GaussianDiffusion,
        schedule_sampler: ScheduleSampler,
        metric: str,
        goal: str,
        project: str,
        name: str,
        log_dir: str,
        loss_params: JsonConfig,
        seed: int,
        config: JsonConfig,
        grad_norm_clip_value: float = None,
        grad_clip_value: float = None,
        log_step_gap: int = 100
    ):
        self.gpu_id = gpu_id
        self.model = model.to(self.gpu_id)
        self.train_data = train_data
        self.val_data = val_data
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.diffusion = diffusion
        self.schedule_sampler = schedule_sampler
        self.project = project
        self.log_dir = os.path.join(log_dir, name)
        self.loss_params = {} if loss_params is None else loss_params
        self.seed = seed
        self.name = name
        self.metric = metric
        self.goal = goal
        self.grad_norm_clip_value = grad_norm_clip_value
        self.grad_clip_value = grad_clip_value
        self.log_step_gap = log_step_gap

        self.model = DDP(model, device_ids=[self.gpu_id])
        self.best_state_dict = copy.deepcopy(self.model.module.state_dict())
        self.chkpt_path = os.path.join(
            self.log_dir, f"chkpts/chkpt_gpu{self.gpu_id}_seed{seed}.pt"
        )

        config["Meta"]["num_params"] = self.model.module.count_learnable_parameters()
        config["Meta"]["chkpt_path"] = self.chkpt_path
        if os.path.exists(self.chkpt_path):
            # resume from chkpt
            self._load_chkpt()
            wandb.init(project=project,
                       id=self.wandb_id,
                       resume="must",
                       config=config,
                       dir=self.log_dir)
        else:
            # initialize from chkpt or from scrach
            self.train_step = 0
            self.epochs_run = 0
            if self.goal == "minimize":
                self.best_metric_value = np.inf
            elif self.goal == "maximize":
                self.best_metric_value = -np.inf
            else:
                raise NotImplementedError(f"Unsupported goal: {self.goal}")
            os.makedirs(os.path.dirname(self.chkpt_path), exist_ok=True)
            # handle when more gpu involved when resuming
            if os.path.exists(os.path.join(self.log_dir, f"chkpts/chkpt_gpu0_seed{seed}.pt")):
                # resume from gpu_0's chkpt
                self._load_chkpt()
            # wandb log
            self.wandb_id = wandb.util.generate_id()
            wandb.init(project=project,
                name=f"gpu{self.gpu_id}_seed{seed}",
                id=self.wandb_id,
                config=config,
                dir=self.log_dir,
                group=name)
            self._save_chkpt()
        # dump args to local
        if self.gpu_id == 0:
            config.dump(os.path.join(self.log_dir, "config.json"))
        wandb.define_metric("train/step")
        wandb.define_metric("train/*", step_metric="train/step")
        wandb.define_metric("val/epochs_run")
        wandb.define_metric("val/*", step_metric="val/epochs_run")

    def _compute_loss(self, batch):
        # process batch
        poses = batch["pose"].to(self.gpu_id) # (N,T,C)
        wavs = batch["wav"].to(self.gpu_id) # (N,T_wav)

        # make model input
        model_kwargs = {"wav": wavs}

        # inpaint input
        if isinstance(self.model.module, Speech2GestureModelInpaint):
            pose_seed_len = self.model.module.pose_seed_len
            inpaint_poses = poses.clone()
            inpaint_masks = th.ones_like(inpaint_poses)[:, :, 0:1] # (N,T,1)
            inpaint_masks[:, pose_seed_len:] = 0
            model_kwargs["inpaint_pose"] = inpaint_poses.transpose(0, 1) # (T,N,C)
            model_kwargs["inpaint_mask"] = inpaint_masks.transpose(0, 1) # (T,N,1)

        # terms for diffusion
        # declare here for continuity loss
        x_start = poses.transpose(1, 2) # (N,T,C) -> (N,C,T)
        noise = th.randn_like(x_start) # (N,C,T)
        diffusion_steps, _ = self.schedule_sampler.sample(
            poses.size(0), self.gpu_id
        )

        # losses
        losses = {"loss": 0}

        # diffusion loss
        diffusion_returns = self.diffusion.training_losses(
            model=self.model,
            x_start=x_start,
            t=diffusion_steps,
            model_kwargs=model_kwargs,
            noise=noise
        ) # (N,1), (N,C,T)
        denoise_loss = diffusion_returns["mse"].mean()
        losses["loss"] += denoise_loss
        losses["denoise"] = denoise_loss

        # compute extra losses
        for loss_name, weight in self.loss_params.items():
            if loss_name == "speed_loss":
                pred_x_start = diffusion_returns["pred_x_start"] # (N,C,T)
                speed = th.abs(th.diff(x_start, dim=2)).mean([0, 1]) # (T-1,)
                speed_pred = th.abs(th.diff(pred_x_start, dim=2)).mean([0, 1]) # (T-1,)
                loss = wasserstein_distance_1d(speed, speed_pred)
                losses["speed"] = loss
                losses["loss"] += weight * loss
                
            elif loss_name == "speed_l1_loss":
                pred_x_start = diffusion_returns["pred_x_start"] # (N,C,T)
                speed = th.abs(th.diff(x_start, dim=2)).mean([0, 1]) # (T-1,)
                speed_pred = th.abs(th.diff(pred_x_start, dim=2)).mean([0, 1]) # (T-1,)
                loss = F.smooth_l1_loss(speed_pred, speed)
                losses["speed_l1"] = loss
                losses["loss"] += weight * loss

            elif loss_name == "speed_constraint_loss": 
                pred_x_start = diffusion_returns["pred_x_start"] # (N,C,T)
                loss = th.abs(th.diff(pred_x_start, dim=2)).mean()
                losses["speed_constraint"] = loss
                losses["loss"] += weight * loss

            else:
                raise ValueError(f"Unsupported loss: {loss_name}")

        return losses

    def _save_chkpt(self):
        chkpt = {
            "model_state_dict": self.model.module.state_dict(),
            "best_state_dict": self.best_state_dict,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
            "train_step": self.train_step,
            "epochs_run": self.epochs_run,
            "wandb_id": self.wandb_id,
            "best_metric_value": self.best_metric_value
        }
        th.save(chkpt, self.chkpt_path)
        # print(f"Epoch {epoch} | Training chkpt saved at {}")

    def _load_chkpt(self):
        chkpt = th.load(os.path.join(self.log_dir, f"chkpts/chkpt_gpu0_seed{self.seed}.pt")) # only load gpu_0's chkpt for all nodes
        self.model.module.load_state_dict(chkpt["model_state_dict"])
        self.best_state_dict = chkpt["best_state_dict"]
        self.optimizer.load_state_dict(chkpt["optimizer_state_dict"])
        self.lr_scheduler.load_state_dict(chkpt["lr_scheduler_state_dict"])
        self.train_step = chkpt["train_step"]
        self.epochs_run = chkpt["epochs_run"]
        self.wandb_id = chkpt["wandb_id"]
        self.best_metric_value = chkpt["best_metric_value"]
        print(f"[Info] [GPU{self.gpu_id}] Resuming training from chkpt at Epoch {self.epochs_run}")

    def _train_step(self):
        self.model.module.train()
        for batch in self.train_data:
            self.optimizer.zero_grad()
            loss_terms = self._compute_loss(batch)
            loss_terms["loss"].backward()
            grad_norm = compute_grad_norm(self.model.module)
            if self.grad_norm_clip_value is not None:
                clip_grad_norm_(self.model.module.parameters(), self.grad_norm_clip_value)
            if self.grad_clip_value is not None:
                clip_grad_value_(self.model.module.parameters(), self.grad_clip_value)
            self.optimizer.step()
            self.lr_scheduler.step()
            if self.train_step % self.log_step_gap == 0: 
                log_dict = {
                    "train/step": self.train_step,
                    "train/lr": self.lr_scheduler.get_last_lr()[0],
                    "train/grad_norm": grad_norm
                }
                for loss_name, value in loss_terms.items():
                    log_dict[f"train/{loss_name}"] = value.item()
                wandb.log(log_dict, step=self.train_step)
            self.train_step += 1

    @th.no_grad()
    def _val_step(self):
        self.model.module.eval()
        log_dict = {"epoch": self.epochs_run}
        for batch in self.val_data:
            loss_terms = self._compute_loss(batch)
            # accumulate loss terms
            for loss_name, value in loss_terms.items():
                value = value.item() / len(self.val_data)
                if not loss_name in log_dict.keys():
                    log_dict[loss_name] = value
                else:
                    log_dict[loss_name] += value
        # prepend "val" to log_dict
        val_log_dict = {}
        for name, value in log_dict.items():
            val_log_dict[f"val/{name}"] = value
        metric_value = val_log_dict[self.metric.replace("_", "/", 1)] # e.g., val_loss_value -> val/loss_value
        val_log_dict[self.metric] = metric_value
        wandb.log(val_log_dict, step=self.train_step)
        # update state dict if becomes better
        if is_improved(metric_value, self.best_metric_value, self.goal):
            self.best_state_dict = copy.deepcopy(self.model.module.state_dict())
            self.best_metric_value = metric_value
            self.early_stop_counter = 0
        else:
            self.early_stop_counter += 1
            if self.early_stop_counter == self.early_stop_threshold:
                self.early_stop = True
                print("[Info] Early stop threshold reached. Stop training.")

    def _run_epoch(self):
        st = time.time()
        batch_size = len(next(iter(self.train_data))["pose"])
        self._train_step()
        self._val_step()
        self.epochs_run += 1
        # print epoch info
        info = f"[Info] GPU: {self.gpu_id}"
        info += f" | Epoch: {self.epochs_run}/{self.max_epochs}"
        info += f" | Batchsize: {batch_size}"
        info += f" | Time: {(time.time() - st):.2f}"
        info += f" | Best metric: {self.best_metric_value:.6f}"
        info += f" | Early stop: {self.early_stop_counter}/{self.early_stop_threshold}"
        print(info)

    def train(self, max_epochs: int, early_stop_threshold: int):
        self.max_epochs = max_epochs
        # early stop
        self.early_stop_threshold = early_stop_threshold
        self.early_stop_counter = 0
        self.early_stop = False
        for _ in range(self.epochs_run, max_epochs):
            self._run_epoch()
            self._save_chkpt()
            if self.early_stop:
                break
        wandb.finish()


def wasserstein_distance_1d(xs, ys, eps=1e-12):
    assert len(xs.shape) == 1, "must be 1-dimensional"
    assert len(ys.shape) == 1, "must be 1-dimensional"
    # wassertein distance between gaussians
    mu1 = xs.mean()
    var1 = xs.var()
    mu2 = ys.mean()
    var2 = ys.var()
    dist_quad = (mu1 - mu2) ** 2 + (var1 + var2 - 2 * th.sqrt(var1.sqrt() * var2 * var1.sqrt()))
    if th.any(th.isnan(dist_quad)):
        print(mu1, mu2, var1, var2, dist_quad)
        raise ValueError("[Error] Nan value in loss")
    return th.maximum(dist_quad, th.zeros_like(dist_quad).fill_(eps)).sqrt()


def is_improved(metric_value: float, best_metric_value: float, goal: str) -> bool:
    """ Return True if the value becomes better. """
    if goal == "minimize":
        if metric_value < best_metric_value:
            return True
        else:
            return False
    elif goal == "maximize":
        if metric_value > best_metric_value:
            return True
        else:
            return False
    else:
        raise ValueError(f"metric_goal {goal} not supported.")


def compute_grad_norm(net: th.nn.Module) -> float:
    total_norm = 0
    for name, p in net.named_parameters():
        if p.requires_grad:
            if p.grad == None:
                print(name)
            param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    return total_norm ** (1. / 2)
