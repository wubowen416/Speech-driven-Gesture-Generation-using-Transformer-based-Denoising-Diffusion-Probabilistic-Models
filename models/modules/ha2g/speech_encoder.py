import numpy as np
import torch as th
import torch.nn as nn
from argparse import Namespace
from torchaudio.transforms import MelSpectrogram
from .model.hierarchy_net import Hierarchical_WavEncoder, PreEmphasis


class HA2GSpeechEncoder(nn.Module):
    ''' wrapper for HA2G encoder modules. Project output to d_model. '''
    def __init__(
        self,
        d_model: int,
        dropout_prob: float
    ):
        super().__init__()
        # wav encoder
        self.wav2spec = nn.Sequential(
            PreEmphasis(),
            MelSpectrogram(
                sample_rate=16000, 
                n_fft=1024, 
                hop_length=512,
                n_mels=128
            )
        )
        self.wav2spec.requires_grad_(False)
        self.mel_spec_norm = nn.InstanceNorm1d(128)
        self.wav_encoder = Hierarchical_WavEncoder(
            args=None, z_obj=False, pose_level=1, nOut=32
        ) # returns: weight, feat_low, feat_mid, feat_high, linear_blend_feat, (N,T,32)

        # added by Wu
        self.wav_proj_layer = nn.Linear(32, d_model)
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(
        self, *,
        wavform: th.Tensor,
    ):
        '''
        :param wavform: shape [N, T]
        :returns: tensors of shape [N, T, d_model]
        '''

        # encode wav    
        # with th.no_grad():
        # with th.cuda.amp.autocast(enabled=False):
        # Using requires_grad_(False) to prevent updating.
        mel_spec = self.wav2spec(wavform) + 1e-6
        mel_spec = self.mel_spec_norm(mel_spec)

        _, feat_low, feat_mid, feat_high, _ = self.wav_encoder(mel_spec, None)
        feat_low = self.wav_proj_layer(self.dropout(feat_low))
        # feat_low = None
        feat_mid = self.wav_proj_layer(self.dropout(feat_mid))
        # feat_mid = None
        feat_high = self.wav_proj_layer(self.dropout(feat_high))
        # feat_high = None

        return feat_low, feat_mid, feat_high
