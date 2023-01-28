import math
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from .modules.gcn.tgcn import ConvTemporalGraphical
from .modules.gcn.graph import Graph
from .modules.transformer import (
    MultiHeadAttention,
    MultiDConvHeadAttention,
    PositionalEncoding,
    FeedForward,
    SquaredReLU
)
from .modules.glide.unet import UNetModel


def diffusion_step_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = th.exp(
        -math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
    if dim % 2:
        embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class DiffusionStepEncoder(nn.Module):
    def __init__(self, d_model, dropout_rate):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
            nn.Dropout(p=dropout_rate)
        )
        self.d_model = d_model

    def forward(self, timesteps: th.Tensor):
        assert len(timesteps.shape) == 1
        x = diffusion_step_embedding(timesteps, dim=self.d_model)
        return self.proj(x)


class CrossAttentionLayer(nn.Module):
    def __init__(
        self,
        d_model,
        dropout_prob,
        self_attn: MultiHeadAttention,
        self_attn_mem: MultiHeadAttention,
        cross_attn: MultiHeadAttention,
        feed_forward: FeedForward,
        feed_forward_mem: FeedForward = None
    ) -> None:
        super().__init__()
        self.size = d_model
        self.dropout = nn.Dropout(dropout_prob)

        self.norm_self_attn = nn.LayerNorm([d_model])
        self.self_attn = self_attn

        self.norm_self_attn_mem = nn.LayerNorm([d_model])
        self.self_attn_mem = self_attn_mem

        self.norm_cross_attn = nn.LayerNorm([d_model])
        self.cross_attn = cross_attn

        self.norm_ff = nn.LayerNorm([d_model])
        self.feed_forward = feed_forward

        self.feed_forward_mem = feed_forward_mem
        if feed_forward_mem != None:
            self.norm_ff_mem = nn.LayerNorm([d_model])

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self, *,
        x: th.Tensor, # (T, N, d_model)
        memory: th.Tensor, # (T_mem, N, d_model)
    ):
        # self-attn
        # x
        z = self.norm_self_attn(x)
        z = self.self_attn(query=z, key=z, value=z, mask=None)
        x = x + self.dropout(z)
        # memory
        z = self.norm_self_attn_mem(memory)
        z = self.self_attn_mem(query=z, key=z, value=z, mask=None)
        memory = memory + self.dropout(z)

        # concat and cross-attn
        length_x = x.size(0)
        h = th.cat([x, memory], dim=0)
        z = self.norm_cross_attn(h)
        z = self.cross_attn(query=z, key=z, value=z, mask=None)
        h = h + self.dropout(z)
        # split
        x = h[:length_x]
        memory = h[length_x:]

        # ff
        z = self.norm_ff(x)
        z = self.feed_forward(z)
        x = x + self.dropout(z)

        if self.feed_forward_mem != None:
            z = self.norm_ff_mem(memory)
            z = self.feed_forward_mem(z)
            memory = memory + self.dropout(z)
        
        return x, memory


class OnewayCrossAttentionLayer(nn.Module):
    def __init__(
        self,
        d_model,
        dropout_prob,
        self_attn: MultiHeadAttention,
        cross_attn: MultiHeadAttention,
        feed_forward: FeedForward,
    ) -> None:
        super().__init__()
        self.size = d_model
        self.dropout = nn.Dropout(dropout_prob)

        self.norm_self_attn = nn.LayerNorm([d_model])
        self.self_attn = self_attn

        self.norm_cross_attn = nn.LayerNorm([d_model])
        self.cross_attn = cross_attn

        self.norm_ff = nn.LayerNorm([d_model])
        self.feed_forward = feed_forward

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self, *,
        x: th.Tensor, # (T, N, d_model)
        memory: th.Tensor, # (T_mem, N, d_model)
    ):
        # self-attn
        z = self.norm_self_attn(x)
        z = self.self_attn(query=z, key=z, value=z, mask=None)
        x = x + self.dropout(z)

        # cross-attn
        z = self.norm_cross_attn(x)
        z = self.cross_attn(query=z, key=memory, value=memory, mask=None)
        x = x + self.dropout(z)

        # ff
        z = self.norm_ff(x)
        z = self.feed_forward(z)
        x = x + self.dropout(z)
        
        return x


class OnewayCrossAttention(nn.Module):
    def __init__(
        self,
        d_x: int,
        d_memory: int,
        d_model: int,
        heads: int,
        dropout_prob: float,
        n_layers: int,
        d_out: int = None
    ) -> None:
        super().__init__()
        self.emb_x = nn.Linear(d_x, d_model)
        self.emb_mem = nn.Linear(d_memory, d_model)
        self.pe = PositionalEncoding(d_model, dropout_prob)

        # decoder layers
        self.layers = nn.ModuleList([
            OnewayCrossAttentionLayer(
                d_model=d_model,
                dropout_prob=dropout_prob,
                self_attn=MultiDConvHeadAttention(
                    heads, d_model, dropout_prob),
                cross_attn=MultiDConvHeadAttention(
                    heads, d_model, dropout_prob),
                feed_forward=FeedForward(
                    d_model, dropout=dropout_prob, activation=SquaredReLU())
            )
            for _ in range(n_layers)
        ])

        if d_out is None:
            d_out = d_model

        self.out_layers = nn.Sequential(
            nn.LayerNorm([self.layers[-1].size]),
            nn.Linear(d_model, d_out)
        )

    def forward(
        self, *,
        x: th.Tensor, # (T, N, d_x)
        memory: th.Tensor, # (T_mem, N, d_memory)
    ):
        # project to d_model
        x = self.pe(self.emb_x(x))
        memory = self.pe(self.emb_mem(memory))

        # forward
        for layer in self.layers:
            x = layer(x=x, memory=memory)
        return self.out_layers(x)


class CrossAttentionGCNLayer(CrossAttentionLayer):
    def __init__(
        self,
        d_model: int,
        dropout_prob: float,
        gcn: ConvTemporalGraphical,
        n_vertices: int,
        self_attn: MultiHeadAttention,
        self_attn_mem: MultiHeadAttention,
        cross_attn: MultiHeadAttention,
        feed_forward: FeedForward,
        feed_forward_mem: FeedForward = None
    ) -> None:
        super().__init__(
            d_model,
            dropout_prob,
            self_attn,
            self_attn_mem,
            cross_attn,
            feed_forward,
            feed_forward_mem
        )
        self.n_vertices = n_vertices
        self.norm_gcn = nn.LayerNorm([d_model//n_vertices])
        self.gcn = gcn

    def forward(
        self, *,
        x: th.Tensor, # (T, N, V, d_model//V)
        A: th.Tensor, # (K, V, V), adjacency matrix
        memory: th.Tensor, # (T_mem, N, d_model)
    ):
        # graph conv
        z = self.norm_gcn(x) # (T,N,V,d_model//V)
        z = z.permute(1, 3, 0, 2) # -> (N,C,T,V)
        z, _ = self.gcn(z, A) # -> (N,C,T,V)
        z = z.permute(2, 0, 3, 1) # -> (T,N,V,d_model//V)
        x = x + self.dropout(z)
        x = x.view(*x.size()[:2], -1) # -> (T,N,d_model)

        x, memory = super().forward(x=x, memory=memory)
        x = x.view(*x.size()[:2], self.n_vertices, -1) # -> (T,N,V,d_model//V)

        return x, memory


class CrossAttentionGCN(nn.Module):
    def __init__(
        self,
        d_x: int,
        d_memory: int,
        d_model: int,
        heads: int,
        dropout_prob: float,
        n_layers: int,
        d_out: int = None,
        graph_layout: str = "beat",
        graph_strategy: str = "spatial"
    ) -> None:
        super().__init__()

        self.pe = PositionalEncoding(d_model, dropout_prob)
        self.emb_mem = nn.Linear(d_memory, d_model)

        self.graph = Graph(graph_layout, graph_strategy)
        A = th.tensor(self.graph.A, dtype=th.float32, requires_grad=False)
        self.register_buffer('A', A)
        n_graphs = A.size(0)
        self.n_vertices = A.size(1)
        assert d_model % self.n_vertices == 0, \
            f"d_model {d_model} must be divisible by n_vertices {self.n_vertices}"
        
        self.emb_x = nn.Linear(d_x//self.n_vertices, d_model//self.n_vertices)

        # decoder layers
        self.layers = nn.ModuleList([
            CrossAttentionGCNLayer(
                d_model=d_model,
                dropout_prob=dropout_prob,
                gcn=ConvTemporalGraphical(
                    d_x//self.n_vertices, d_x//self.n_vertices, n_graphs),
                n_vertices=self.n_vertices,
                self_attn=MultiDConvHeadAttention(
                    heads, d_model, dropout_prob),
                self_attn_mem=MultiDConvHeadAttention(
                    heads, d_model, dropout_prob),
                cross_attn=MultiDConvHeadAttention(
                    heads, d_model, dropout_prob),
                feed_forward=FeedForward(
                    d_model, dropout=dropout_prob, activation=SquaredReLU()),
                feed_forward_mem=FeedForward(
                    d_model, dropout=dropout_prob, activation=SquaredReLU())
            )
            for _ in range(n_layers-1)
        ])
        # last layer does not feed-forward memory
        self.layers.append(CrossAttentionGCNLayer(
            d_model=d_model,
            dropout_prob=dropout_prob,
            gcn=ConvTemporalGraphical(
                d_x//self.n_vertices, d_x//self.n_vertices, n_graphs),
            n_vertices=self.n_vertices,
            self_attn=MultiDConvHeadAttention(
                heads, d_model, dropout_prob),
            self_attn_mem=MultiDConvHeadAttention(
                heads, d_model, dropout_prob),
            cross_attn=MultiDConvHeadAttention(
                heads, d_model, dropout_prob),
            feed_forward=FeedForward(
                d_model, dropout=dropout_prob, activation=SquaredReLU())
        ))

        if d_out is None:
            d_out = d_model

        assert d_out % self.n_vertices == 0, \
            f"d_out {d_out} must be divisible by n_vertices {self.n_vertices}"
        self.out_layers = nn.Linear(
            d_model//self.n_vertices, d_out//self.n_vertices)

    def forward(
        self, *,
        x: th.Tensor, # (T, N, d_pose)
        memory: th.Tensor, # (T_mem, N, d_memory)
    ):
        # project to d_model
        x = x.view(*x.size()[:2], self.n_vertices, -1) # -> (T,N,V,d_pose//V)
        x = self.emb_x(x) # -> (T,N,V,d_model//V)
        x = x.view(*x.size()[:2], -1) # -> (T,N,V,d_model)
        memory = self.emb_mem(memory) # -> (T_mem,N,d_model)

        # concat and positional encoding
        length_x = x.size(0)
        h = th.cat([x, memory], dim=0)
        h = self.pe(h)
        x = h[:length_x]
        memory = h[length_x:]

        # reshape
        x = x.view(*x.size()[:2], self.n_vertices, -1) # -> (T,N,V,d_model//V)

        # forward
        for layer in self.layers:
            x, memory = layer(x=x, A=self.A, memory=memory)
        x = self.out_layers(x) # -> (T,N,V,d_out//V)
        x = x.view(*x.size()[:2], -1) # -> (T,N,d_out)

        return x


class CrossAttention(nn.Module):
    def __init__(
        self,
        d_x: int,
        d_memory: int,
        d_model: int,
        heads: int,
        dropout_prob: float,
        n_layers: int,
        d_out: int = None
    ) -> None:
        super().__init__()
        self.emb_x = nn.Linear(d_x, d_model)
        self.emb_mem = nn.Linear(d_memory, d_model)
        self.pe = PositionalEncoding(d_model, dropout_prob)

        # decoder layers
        self.layers = nn.ModuleList([
            CrossAttentionLayer(
                d_model=d_model,
                dropout_prob=dropout_prob,
                self_attn=MultiDConvHeadAttention(heads, d_model, dropout_prob),
                self_attn_mem=MultiDConvHeadAttention(heads, d_model, dropout_prob),
                cross_attn=MultiDConvHeadAttention(heads, d_model, dropout_prob),
                feed_forward=FeedForward(d_model, dropout=dropout_prob, activation=SquaredReLU()),
                feed_forward_mem=FeedForward(d_model, dropout=dropout_prob, activation=SquaredReLU())
            )
            for _ in range(n_layers-1)
        ])
        # last layer does not feed-forward memory
        self.layers.append(CrossAttentionLayer(
            d_model=d_model,
            dropout_prob=dropout_prob,
            self_attn=MultiDConvHeadAttention(heads, d_model, dropout_prob),
            self_attn_mem=MultiDConvHeadAttention(heads, d_model, dropout_prob),
            cross_attn=MultiDConvHeadAttention(heads, d_model, dropout_prob),
            feed_forward=FeedForward(d_model, dropout=dropout_prob, activation=SquaredReLU())
        ))

        if d_out is None:
            d_out = d_model

        self.out_layers = nn.Sequential(
            nn.LayerNorm([self.layers[-1].size]),
            nn.Linear(d_model, d_out)
        )

    def forward(
        self, *,
        x: th.Tensor, # (T, N, d_x)
        memory: th.Tensor, # (T_mem, N, d_memory)
    ):
        # project to d_model
        x = self.emb_x(x)
        memory = self.emb_mem(memory)

        # concat and positional encoding
        length_x = x.size(0)
        h = th.cat([x, memory], dim=0)
        h = self.pe(h)
        x = h[:length_x]
        memory = h[length_x:]

        # forward
        for layer in self.layers:
            x, memory = layer(x=x, memory=memory)
        return self.out_layers(x)
    

class UNetAttn(UNetModel):
    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        window_len,
        pad_for_updown,
        dropout=0,
        channel_mult=...,
        conv_resample=True,
        dims=1,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        encoder_channels=None
    ):
        super().__init__(
            in_channels,
            model_channels,
            out_channels,
            num_res_blocks,
            attention_resolutions,
            dropout,
            channel_mult,
            conv_resample,
            dims,
            num_classes,
            use_checkpoint,
            use_fp16,
            num_heads,
            num_head_channels,
            num_heads_upsample,
            use_scale_shift_norm,
            resblock_updown,
            encoder_channels
        )
        
        if pad_for_updown:
            T = window_len
            while not is_divisible_by_2_n_times(T, len(channel_mult) - 1):
                T += 1
            if window_len % 2 == 0: # even
                self.pad_lens = ((T - window_len) // 2, (T - window_len) // 2)
            else:
                raise NotImplementedError(f"uneven length not supported.")
        else:
            assert is_divisible_by_2_n_times(window_len, len(channel_mult) - 1), \
                "T must be divisible by 2 after n-1 times downsample."
        
        self.pad_for_updown = pad_for_updown
    
    
    def forward(
        self, *,
        x: th.Tensor, # (T, N, d_x)
        memory: th.Tensor, # (T_mem, N, d_memory)
    ):
        # split memory to timestep embedding and audio embedding
        emb_time, emb_audio = memory[0], memory[1:] # (N,C), (T,N,C)
        
        emb_time = self.time_embed(emb_time)
        
        # adjust dimension
        x = x.permute(1, 2, 0) # -> (N,C,T)
        emb_audio = emb_audio.permute(1, 2, 0) # -> (N,C,T)
        
        if self.pad_for_updown:
            x = F.pad(x, self.pad_lens) # pad last dim
        
        # Forward: h - NCT, emb - NC, enc_out - NCT
        hs = []
        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb_time, emb_audio)
            hs.append(h)
        h = self.middle_block(h, emb_time, emb_audio)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb_time, emb_audio)
        h = h.type(x.dtype)
        x = self.out(h)
        
        if self.pad_for_updown:
            # cut to origin length
            x = x[:, :, self.pad_lens[0]:-self.pad_lens[1]]
            
        return x.permute(2, 0, 1) # -> (T,N,C)
    

def is_divisible_by_2_n_times(length, n):
    for _ in range(n):
        length /= 2
    return length % 2 == 0
