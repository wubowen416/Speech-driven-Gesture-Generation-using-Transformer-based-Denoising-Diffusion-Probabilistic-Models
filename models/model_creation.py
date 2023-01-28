import torch as th
from .lr_scheduler import NoamDecayLR, ConstantLR, NoamLR
from .model import Speech2GestureModel, Speech2GestureModelInpaint, Speech2GestureModelV2
from .nn import (
    CrossAttention,
    CrossAttentionGCN,
    DiffusionStepEncoder,
    UNetAttn,
    OnewayCrossAttention
)
from .modules.ha2g.speech_encoder import HA2GSpeechEncoder
from .modules.gaussian_diffusion import get_named_beta_schedule
from .modules.resample import create_named_schedule_sampler
from .modules.respace import GaussianSpacedDiffusion, space_timesteps
from utils.json_config import JsonConfig
from utils.string_parser import parse_steps


def create_lr_scheduler(params, optimzier):
    if params.type == "const":
        return ConstantLR(optimzier)
    elif params.type == "noam":
        return NoamDecayLR(optimzier, parse_steps(params.warmup_steps))
    elif params.type == "noamxf":
        return NoamLR(optimzier, params.d_model, parse_steps(params.warmup_steps))
    else:
        raise ValueError("Unsupport lr_scheduler type.")


def create_diffusion(diffusion_params, is_training):
    if diffusion_params.type == "gaussian":
        betas = get_named_beta_schedule(
            diffusion_params.noise_schedule,
            diffusion_params.diffusion_steps
        )
        if not diffusion_params.timestep_respacing or is_training:
            timestep_respacing = [diffusion_params.diffusion_steps]
        else:
            timestep_respacing = diffusion_params.timestep_respacing
        diffusion = GaussianSpacedDiffusion(
            use_timesteps=space_timesteps(
                diffusion_params.diffusion_steps, timestep_respacing),
            betas=betas,
            model_var_type=diffusion_params.model_var_type
        )
    else:
        raise ValueError
    return diffusion


def create_model(
    d_pose: int,
    model_params: JsonConfig,
    lr: float = 1e-2,
    weight_decay: float = None,
    scheduler_params: JsonConfig = None,
    is_training: bool = False
):
    if weight_decay is None:
        weight_decay = 0.0

    # speech encoder
    encoder_params = model_params.get("Encoder")
    if encoder_params.type == "ha2g":
        speech_encoder = HA2GSpeechEncoder(
            d_model=model_params.d_model,
            dropout_prob=model_params.dropout_prob
        )
    else:
        raise ValueError

    # pose decoder
    decoder_params = model_params.get("Decoder")
    if decoder_params.type == "cross_attention":
        decoder = CrossAttention(
            d_x=d_pose,
            d_memory=model_params.d_model,
            d_model=model_params.d_model,
            dropout_prob=model_params.dropout_prob,
            d_out=d_pose,
            heads=decoder_params.heads,
            n_layers=decoder_params.n_layers
        )
    elif decoder_params.type == "oneway_cross_attention":
        decoder = OnewayCrossAttention(
            d_x=d_pose,
            d_memory=model_params.d_model,
            d_model=model_params.d_model,
            dropout_prob=model_params.dropout_prob,
            d_out=d_pose,
            heads=decoder_params.heads,
            n_layers=decoder_params.n_layers,
        )
    elif decoder_params.type == "cross_attention_gcn":
        decoder = CrossAttentionGCN(
            d_x=d_pose,
            d_memory=model_params.d_model,
            d_model=model_params.d_model,
            dropout_prob=model_params.dropout_prob,
            d_out=d_pose,
            heads=decoder_params.heads,
            n_layers=decoder_params.n_layers,
            graph_layout=decoder_params.graph_layout,
            graph_strategy=decoder_params.graph_strategy
        )
    elif decoder_params.type == "unet_attention":
        decoder = UNetAttn(
            in_channels=d_pose,
            model_channels=model_params.d_model,
            out_channels=d_pose,
            num_res_blocks=decoder_params.num_res_blocks,
            attention_resolutions=decoder_params.attention_resolutions,
            window_len=decoder_params.window_len,
            pad_for_updown=True,
            dropout=model_params.dropout_prob,
            channel_mult=decoder_params.channel_mult,
            num_heads=decoder_params.num_heads,
            use_scale_shift_norm=True,
            encoder_channels=model_params.d_model
        )
    else:
        raise ValueError(f"Unsupported decoder type {decoder_params.type}.")

    # create diffusion step encoder
    diffusion_step_encoder = DiffusionStepEncoder(
        model_params.d_model, model_params.dropout_prob
    )

    # create diffusion
    diffusion_params = model_params.get("Diffusion")
    diffusion = create_diffusion(diffusion_params, is_training)

    # create model
    if model_params.type == "inpaint":
        model = Speech2GestureModelInpaint(
            d_pose,
            model_params.d_model,
            speech_encoder,
            decoder,
            diffusion_step_encoder,
            model_params.dropout_prob,
            pose_seed_len=model_params.Generate.pose_seed_len
        )
    elif model_params.type == "default":
        model = Speech2GestureModel(
            d_pose,
            model_params.d_model,
            speech_encoder,
            decoder,
            diffusion_step_encoder
        )
    elif model_params.type == "s2g_v2":
        model = Speech2GestureModelV2(
            d_pose,
            model_params.d_model,
            speech_encoder,
            decoder,
            diffusion_step_encoder
        )
    else:
        raise ValueError(f"Unsupported model_type {model_params.type}")

    # for fine-tune
    if is_training and model_params.get('start_chkpt'):
        print(f"[Info] Load chkpt as start from: {model_params.start_chkpt}")
        chkpt = th.load(model_params.start_chkpt)
        missing_keys, _ = model.load_state_dict(chkpt["best_state_dict"], strict=False)
        new_params, old_params = [], []
        for name, p in model.named_parameters():
            if name in missing_keys:
                print(f'[Warning] New param added: {name}.')
                new_params.append(p)
            else:
                old_params.append(p)
        optimizer = th.optim.AdamW([{'params': new_params, 'lr': lr * 10}, 
                                    {'params': old_params}], 
                                   lr=lr, weight_decay=weight_decay)

    optimizer = th.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    # diffusion schedule
    schedule_sampler = create_named_schedule_sampler("uniform", diffusion)

    # lr scheduler
    if scheduler_params is None:
        scheduler_params = JsonConfig({'type': 'const'})
    lr_scheduler = create_lr_scheduler(scheduler_params, optimizer)

    return model, diffusion, optimizer, schedule_sampler, lr_scheduler
