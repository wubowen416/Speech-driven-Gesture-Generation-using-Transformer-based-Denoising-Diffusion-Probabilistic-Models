import math
import torch as th
import torch.nn as nn

from typing import Optional, List


class SquaredReLU(nn.Module):
    ''' https://nn.labml.ai/transformers/primer_ez/index.html '''
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()

    def forward(self, x: th.Tensor):
        x = self.relu(x)
        return x * x


class SpatialDepthWiseConv(nn.Module):
    ''' https://nn.labml.ai/transformers/primer_ez/index.html '''
    def __init__(self, d_k: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=d_k, out_channels=d_k,
                              kernel_size=(kernel_size,), padding=(kernel_size - 1,), groups=d_k)
        assert kernel_size % 2 == 1, 'kernel_size must be odd.'
        self.crop_length = (kernel_size - 1) // 2

    def forward(self, x: th.Tensor):
        seq_len, batch_size, heads, d_k = x.shape
        x = x.permute(1, 2, 3, 0)
        x = x.view(batch_size * heads, d_k, seq_len)
        x = self.conv(x)
        # Original implementation: crop right most
        # x = x[:, :, :-(self.kernel_size - 1)]
        # My implementation:
        # instead of crop the right most output,
        # crop both side equally for balancing 
        # the begining and the end of a sequence.
        # Only applicable for odd kernel size.
        x = x[:, :, self.crop_length:-self.crop_length]

        x = x.view(batch_size, heads, d_k, seq_len)
        x = x.permute(3, 0, 1, 2)
        return x


class PrepareForMultiHeadAttention(nn.Module):
    ''' https://nn.labml.ai/transformers/mha.html#MHA '''
    def __init__(self, d_model: int, heads: int, d_k: int, bias: bool):
        super().__init__()
        self.linear = nn.Linear(d_model, heads * d_k, bias=bias)
        self.heads = heads
        self.d_k = d_k

    def forward(self, x: th.Tensor):
        head_shape = x.shape[:-1]
        x = self.linear(x)
        x = x.view(*head_shape, self.heads, self.d_k)
        return x


class MultiHeadAttention(nn.Module):
    ''' https://nn.labml.ai/transformers/mha.html#MHA '''
    def __init__(self, heads: int, d_model: int, dropout_prob: float = 0.1, bias: bool = True):
        super().__init__()
        assert d_model % heads == 0, 'd_model msut be divisible by heads.'
        self.d_k = d_model // heads
        self.heads = heads
        self.query = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=bias)
        self.key = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=bias)
        self.value = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=True)
        self.softmax = nn.Softmax(dim=1)
        self.output = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout_prob)
        self.scale = 1 / math.sqrt(self.d_k)
        self.attn = None

    def get_scores(self, query: th.Tensor, key: th.Tensor):
        return th.einsum('ibhd,jbhd->ijbh', query, key)

    def prepare_mask(self, mask: th.Tensor, query_shape: List[int], key_shape: List[int]):
        assert mask.shape[0] == 1 or mask.shape[0] == query_shape[0]
        assert mask.shape[1] == key_shape[0]
        assert mask.shape[2] == 1 or mask.shape[2] == query_shape[1]
        mask = mask.unsqueeze(-1)
        return mask

    def forward(
        self, 
        *,
        query: th.Tensor,
        key: th.Tensor,
        value: th.Tensor,
        mask: Optional[th.Tensor] = None
    ):
        '''
        Attend to all positions if mask is None.
        '''
        seq_len, batch_size, _ = query.shape

        if mask is not None:
            mask = self.prepare_mask(mask, query.shape, key.shape)

        query = self.query(query)
        key = self.key(key)
        value = self.value(value)
        scores = self.get_scores(query, key)
        scores *= self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = self.softmax(scores)
        attn = self.dropout(attn)
        x = th.einsum("ijbh,jbhd->ibhd", attn, value)
        self.attn = attn.detach()
        x = x.reshape(seq_len, batch_size, -1)
        return self.output(x)


class MultiDConvHeadAttention(MultiHeadAttention):
    def __init__(self, heads: int, d_model: int, dropout_prob: float = 0.1):
        super().__init__(heads, d_model, dropout_prob)
        self.query = nn.Sequential(self.query, SpatialDepthWiseConv(self.d_k))
        self.key = nn.Sequential(self.key, SpatialDepthWiseConv(self.d_k))
        self.value = nn.Sequential(self.value, SpatialDepthWiseConv(self.d_k))


class FeedForward(nn.Module):
    ''' https://nn.labml.ai/transformers/feed_forward.html '''

    _expansion = 4

    def __init__(
        self, 
        d_model: int, 
        d_ff: int = None,
        dropout: float = 0.1,
        activation=nn.ReLU(),
        bias1: bool = True,
        bias2: bool = True
    ):
        super().__init__()
        if d_ff is None:
            d_ff = self._expansion * d_model
        self.layer1 = nn.Linear(d_model, d_ff, bias=bias1)
        self.layer2 = nn.Linear(d_ff, d_model, bias=bias2)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        
    def forward(self, x: th.Tensor):
        x = self.activation(self.layer1(x))
        x = self.dropout(x)
        return self.layer2(x)


def get_positional_encoding(d_model: int, max_len: int = 5000):
    ''' https://nn.labml.ai/transformers/models.html '''
    encodings = th.zeros(max_len, d_model)
    position = th.arange(0, max_len, dtype=th.float32).unsqueeze(1)
    two_i = th.arange(0, d_model, 2, dtype=th.float32)
    div_term = th.exp(two_i * -(math.log(10000.0) / d_model))
    encodings[:, 0::2] = th.sin(position * div_term)
    encodings[:, 1::2] = th.cos(position * div_term)
    encodings = encodings.unsqueeze(1).requires_grad_(False)
    return encodings


class PositionalEncoding(nn.Module):
    ''' https://nn.labml.ai/transformers/models.html '''
    def __init__(self, d_model: int, dropout_prob: float, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout_prob)
        self.register_buffer('positional_encodings', get_positional_encoding(d_model, max_len), False)

    def forward(self, x: th.Tensor):
        pe = self.positional_encodings[:x.shape[0]].detach().requires_grad_(False)
        x = x + pe
        x = self.dropout(x)
        return x


class EmbeddingsWithPositionalEncoding(nn.Module):
    ''' https://nn.labml.ai/transformers/models.html '''
    def __init__(self, d_in: int, d_model: int, max_len: int = 5000):
        super().__init__()
        self.linear = nn.Linear(d_in, d_model)
        self.d_model = d_model
        self.register_buffer('positional_encodings', get_positional_encoding(d_model, max_len))

    def forward(self, x: th.Tensor):
        pe = self.positional_encodings[:x.shape[0]].requires_grad_(False)
        return self.linear(x) * math.sqrt(self.d_model) + pe


class TransformerLayer(nn.Module):
    ''' https://nn.labml.ai/transformers/models.html '''
    def __init__(
        self, 
        *,
        d_model: int,
        self_attn: MultiHeadAttention,
        feed_forward: FeedForward,
        dropout_prob: float,
        src_attn: MultiHeadAttention = None
    ):
        super().__init__()
        self.size = d_model
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.dropout = nn.Dropout(dropout_prob)
        self.norm_self_attn = nn.LayerNorm([d_model])
        if self.src_attn is not None:
            self.norm_src_attn = nn.LayerNorm([d_model])
        self.norm_ff = nn.LayerNorm([d_model])

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self, 
        *,
        x: th.Tensor,
        mask: th.Tensor,
        src: th.Tensor = None,
        src_mask: th.Tensor = None
    ):
        # self-attn
        z = self.norm_self_attn(x)
        self_attn = self.self_attn(query=z, key=z, value=z, mask=mask)
        x = x + self.dropout(self_attn)

        # src-attn
        if src is not None:
            z = self.norm_src_attn(x)
            attn_src = self.src_attn(query=z, key=src, value=src, mask=src_mask)
            x = x + self.dropout(attn_src)

        # feed-forward
        z = self.norm_ff(x)
        ff = self.feed_forward(z)
        x = x + self.dropout(ff)
        return x


class PrimerEZEncoder(nn.Module):
    ''' https://arxiv.org/abs/2109.08668 '''
    def __init__(
        self,
        d_x: int,
        d_model: int,
        heads: int,
        dropout_prob: float,
        n_layers: int,
        d_out: int = None
    ):
        super().__init__()
        self.pe = EmbeddingsWithPositionalEncoding(d_x, d_model)

        # decoder layers
        self.layers = nn.ModuleList([
            TransformerLayer(
                d_model=d_model,
                self_attn=MultiDConvHeadAttention(heads, d_model, dropout_prob),
                src_attn=None,
                feed_forward=FeedForward(d_model, dropout=dropout_prob, activation=SquaredReLU()),
                dropout_prob=dropout_prob
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
        self,
        x: th.Tensor,
        mask: th.Tensor = None
    ):
        '''
        :param x: shape [T_x, N, d_x]
        :param mask: shape [T_x, T_x, N]
        '''
        x = self.pe(x)
        for layer in self.layers:
            x = layer(x=x, mask=mask)
        return self.out_layers(x)


class PrimerEZDecoder(nn.Module):
    ''' https://arxiv.org/abs/2109.08668 '''
    def __init__(
        self,
        d_x: int,
        d_model: int,
        heads: int,
        dropout_prob: float,
        n_layers: int,
        d_out: int = None
    ):
        super().__init__()
        self.pe = EmbeddingsWithPositionalEncoding(d_x, d_model)

        # decoder layers
        self.layers = nn.ModuleList([
            TransformerLayer(
                d_model=d_model,
                self_attn=MultiDConvHeadAttention(heads, d_model, dropout_prob),
                src_attn=MultiDConvHeadAttention(heads, d_model, dropout_prob),
                feed_forward=FeedForward(d_model, dropout=dropout_prob, activation=SquaredReLU()),
                dropout_prob=dropout_prob
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
        self,
        x: th.Tensor,
        memory: th.Tensor,
        mask: th.Tensor = None,
        src_mask: th.Tensor = None
    ):
        '''
        :param x: shape [T_x, N, d_x]
        :param memory: shape [T_mem, N, C]
        :param mask: shape [T_x, T_mem, N]
        :param src_mask: shape [T_x, T_mem, N]
        :return: tensor of shape [T_x, N, d_x]
        '''
        x = self.pe(x)
        for layer in self.layers:
            x = layer(x=x, mask=mask, src=memory, src_mask=src_mask)
        return self.out_layers(x)