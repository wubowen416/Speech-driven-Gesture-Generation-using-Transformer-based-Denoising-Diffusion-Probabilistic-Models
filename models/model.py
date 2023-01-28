import torch as th
import torch.nn as nn
from .modules.ha2g.speech_encoder import HA2GSpeechEncoder
from abc import abstractclassmethod

class Speech2GestureModelBase(nn.Module):
    @abstractclassmethod
    def myforward():
        # Shape should be (T,N,*)
        raise NotImplementedError('[Error] Implement forward path.')

    def forward(self, x_t, t, **model_kwargs):
        x_t = x_t.permute(2, 0, 1)  # (N,C,T) -> (T,N,C)
        return self.myforward(
            x=x_t, diffusion_step=t, **model_kwargs).permute(1, 2, 0)  # (T,N,C) -> (N,C,T)

    def count_learnable_parameters(self):
        num_params = sum([p.numel() for p in self.parameters() if p.requires_grad])
        print("[Info] Number of parameters: {:,}".format(num_params))
        return num_params


class Speech2GestureModel(Speech2GestureModelBase):
    """ Base model. """
    def __init__(
        self,
        d_pose: int,
        d_model: int,
        speech_encoder: nn.Module,
        pose_decoder: nn.Module,
        diffusion_step_encoder: nn.Module
    ):
        super().__init__()
        self.diffusion_step_encoder = diffusion_step_encoder
        self.speech_encoder = speech_encoder
        self.pose_decoder = pose_decoder

        self.d_pose = d_pose
        self.d_model = d_model

    def myforward(
        self, *,
        x: th.Tensor,  # (T,N,d_pose)
        wav: th.Tensor,  # (N,T)
        diffusion_step: th.LongTensor  # (N,)
    ):
        # 1. encoding
        memory = []
        # 1.1 diffusion step encoding
        z_diffusion_step = self.diffusion_step_encoder(
            diffusion_step).unsqueeze(0)  # (1,N,d_model)
        memory.append(z_diffusion_step)
        # 1.2 speech features encoding
        if isinstance(self.speech_encoder, HA2GSpeechEncoder):
            z_low, z_mid, z_high = self.speech_encoder(
                wavform=wav)  # (N, T, d_model)
            z_low = z_low.transpose(0, 1)
            z_mid = z_mid.transpose(0, 1)
            z_high = z_high.transpose(0, 1)  # (N,T,d_model) -> (T,N,d_model)
            memory.append(z_low)
            memory.append(z_mid)
            memory.append(z_high)
        else:
            raise ValueError(
                f"Unsupported speech_encoder {type(self.speech_encoder)}"
            )

        memory = th.cat(memory, dim=0)  # (T_all, N, d_model)

        # 2. decoding
        x = self.pose_decoder(x=x, memory=memory)

        return x
    
    
class Speech2GestureModelV2(Speech2GestureModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.blend_layer = nn.Linear(3*args[1], args[1])
        
    def myforward(
        self, *,
        x: th.Tensor,  # (T,N,d_pose)
        wav: th.Tensor,  # (N,T)
        diffusion_step: th.LongTensor  # (N,)
    ):
        # 1. encoding
        memory = []
        # 1.1 diffusion step encoding
        z_diffusion_step = self.diffusion_step_encoder(
            diffusion_step).unsqueeze(0)  # (1,N,d_model)
        memory.append(z_diffusion_step)
        # 1.2 speech features encoding
        if isinstance(self.speech_encoder, HA2GSpeechEncoder):
            z_low, z_mid, z_high = self.speech_encoder(
                wavform=wav)  # (N, T, d_model)
            longgest = max([x.size(1) for x in [z_low, z_mid, z_high]])
            if z_low.size(1) < longgest:
                z_low = th.cat([th.zeros((z_low.size(0), longgest-z_low.size(1), z_low.size(2)), device=z_low.device), z_low], dim=1)
            if z_mid.size(1) < longgest:
                z_mid = th.cat([th.zeros((z_mid.size(0), longgest-z_mid.size(1), z_mid.size(2)), device=z_mid.device), z_mid], dim=1)
            if z_high.size(1) < longgest:
                z_high = th.cat([th.zeros((z_high.size(0), longgest-z_high.size(1), z_high.size(2)), device=z_high.device), z_high], dim=1)
            z = th.cat([z_low, z_mid, z_high], dim=-1) # -> (*, 3*d_model)
            z = self.blend_layer(z) # -> (*, d_model)
            memory.append(z.transpose(0, 1)) # -> (T,N,*)
        else:
            raise ValueError(
                f"Unsupported speech_encoder {type(self.speech_encoder)}"
            )

        memory = th.cat(memory, dim=0)  # (T_all, N, d_model)

        # 2. decoding
        x = self.pose_decoder(x=x, memory=memory)

        return x


class Speech2GestureModelInpaint(Speech2GestureModel):
    """ add x_cond (seed pose) and mask as input  """
    def __init__(
        self,
        d_pose: int,
        d_model: int,
        speech_encoder: nn.Module,
        pose_decoder: nn.Module,
        diffusion_step_encoder: nn.Module,
        dropout_prob: float,
        pose_seed_len: int
    ):
        super().__init__(
            d_model, d_model, speech_encoder, pose_decoder,
            diffusion_step_encoder
        )
        self.pose_seed_len = pose_seed_len
        self.proj = nn.Sequential(
            nn.Linear(d_pose+1, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_pose),
            nn.Dropout(dropout_prob)
        )

        # zero init weight as GLiDE
        self.proj.apply(self.zero_linear_layer)

    @staticmethod
    def zero_linear_layer(module):
        if isinstance(module, nn.Linear):
            module.weight.data.zero_()
            module.bias.data.zero_()

    def myforward(
        self, *,
        x: th.Tensor,  # (T,N,C)
        wav: th.Tensor,  # (N,T)
        diffusion_step: th.LongTensor,  # (N,)
        inpaint_pose: th.Tensor,  # (T,N,C)
        inpaint_mask: th.Tensor  # (T,N,1)
    ):
        x_inp = inpaint_pose * inpaint_mask
        x_inp = th.cat([x_inp, inpaint_mask], dim=-1)  # -> (T,N,d_pose+1)
        x = x + self.proj(x_inp) # (T,N,d_pose)
        return super().myforward(x=x, wav=wav, diffusion_step=diffusion_step)
