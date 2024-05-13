"""
The code is mainly reproduced from https://github.com/tnq177/transformers_without_tears/tree/master.
"""

import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F


class MultiheadAttention(nn.Module):
    """
    MultiheadAttention module
    I learned a lot from https://github.com/pytorch/fairseq/blob/master/fairseq/modules/multihead_attention.py
    """
    def __init__(self, args):
        super(MultiheadAttention, self).__init__()
        self.d_model = args.d_model
        self.n_heads = args.n_heads
        self.dropout = args.att_dropout
        self.use_transformer_bias = not args.no_transformer_bias
        self.attn_mech = args.attn_mech
        self.d_graphormer_params = args.d_graphormer_params

        if self.d_model % self.n_heads != 0:
            raise ValueError('Required: d_model % n_heads == 0.')

        self.head_dim = self.d_model // self.n_heads
        self.scale = self.head_dim ** -0.5

        # Parameters for linear projections of queries, keys, values and output
        self.weights = Parameter(torch.Tensor(4 * self.d_model, self.d_model))
        if self.use_transformer_bias:
            self.biases = Parameter(torch.Tensor(4 * self.d_model))

        if self.d_graphormer_params:
            self.lin_graphormer = nn.Linear(self.d_graphormer_params, self.n_heads, bias=False)

        # initializing
        # If we do Xavier normal initialization, std = sqrt(2/(2D))
        # but it's too big and causes unstability in PostNorm
        # so we use the smaller std of feedforward module, i.e. sqrt(2/(5D))
        mean = 0
        std = (2 / (5 * self.d_model)) ** 0.5
        nn.init.normal_(self.weights, mean=mean, std=std)
        if self.use_transformer_bias:
            nn.init.constant_(self.biases, 0.)

        if self.attn_mech == 'additive_v':
            self.additive_v = Parameter(torch.Tensor(self.n_heads, self.head_dim))
            nn.init.normal_(self.additive_v, mean=mean, std=std)

    def forward(self, q, k, v, mask, graphormer_dists=None, do_proj_qkv=True):
        """
        TODO: document shapes.

        :param q:
        :param k:
        :param v:
        :param mask:
        :param do_proj_qkv:
        :return:
        """
        bs, n, d = q.size()

        def _split_heads(tensor):
            bsz, length, d_model = tensor.size()
            return tensor.reshape(bsz, length, self.n_heads, self.head_dim).transpose(1, 2)

        if do_proj_qkv:
            q, k, v = self.proj_qkv(q, k, v)

        q = _split_heads(q)
        k = _split_heads(k)
        v = _split_heads(v)

        if self.attn_mech == 'dot-product':
            att_weights = torch.einsum('bhnd,bhdm->bhnm', q, k.transpose(-2, -1))
        elif self.attn_mech == 'additive_v' or self.attn_mech == 'additive_fixed':
            att_weights = (q.unsqueeze(-2) - k.unsqueeze(-3))
            if self.attn_mech == 'additive_v':
                att_weights = (att_weights * self.additive_v.unsqueeze(0).unsqueeze(2).unsqueeze(3))
            att_weights = att_weights.sum(dim=-1)
        else:
            raise NotImplementedError(f'"{self.attn_mech}" attention mechanism is not implemented.')
        att_weights = att_weights * self.scale

        if graphormer_dists is not None:
            if self.d_graphormer_params:
                # (bs, n, n, dists_d) -> (bs, n, n, n_heads) -> (bs, n_heads, n, n) = A.shape
                att_bias = self.lin_graphormer(graphormer_dists).permute(0, 3, 1, 2)
            else:
                # (bs, n, n, dists_d) -> (bs, 1, n, n) broadcastable with A
                att_bias = graphormer_dists.sum(dim=-1).unsqueeze(1)
            att_weights = att_weights + att_bias

        if mask is not None:
            att_weights.masked_fill_(mask.unsqueeze(1).unsqueeze(-1), -1e9)

        att_weights = F.softmax(att_weights, dim=-1)
        att_weights = F.dropout(att_weights, p=self.dropout, training=self.training)
        _att_weights = att_weights.reshape(-1, n, n)
        output = torch.bmm(_att_weights, v.reshape(bs * self.n_heads, -1, self.head_dim))
        output = output.reshape(bs, self.n_heads, n, self.head_dim).transpose(1, 2).reshape(bs, n, -1)
        output = self.proj_o(output)

        return output, att_weights

    def proj_qkv(self, q, k, v):
        qkv_same = q.data_ptr() == k.data_ptr() == v.data_ptr()
        kv_same = k.data_ptr() == v.data_ptr()

        if qkv_same:
            q, k, v = self._proj(q, end=3 * self.d_model).chunk(3, dim=-1)
        elif kv_same:
            q = self._proj(q, end=self.d_model)
            k, v = self._proj(k, start=self.d_model, end=3 * self.d_model).chunk(2, dim=-1)
        else:
            q = self.proj_q(q)
            k = self.proj_k(k)
            v = self.proj_v(v)

        return q, k, v

    def _proj(self, x, start=0, end=None):
        weight = self.weights[start:end, :]
        bias = None if not self.use_transformer_bias else self.biases[start:end]
        return F.linear(x, weight=weight, bias=bias)

    def proj_q(self, q):
        return self._proj(q, end=self.d_model)

    def proj_k(self, k):
        return self._proj(k, start=self.d_model, end=2 * self.d_model)

    def proj_v(self, v):
        return self._proj(v, start=2 * self.d_model, end=3 * self.d_model)

    def proj_o(self, x):
        return self._proj(x, start=3 * self.d_model)


class FeedForward(nn.Module):
    """FeedForward"""
    def __init__(self, args):
        super(FeedForward, self).__init__()
        self.dropout = args.ff_dropout
        self.d_model = args.d_model
        self.ff_dim = 4 * args.d_model
        self.use_transformer_bias = not args.no_transformer_bias

        self.in_proj = nn.Linear(self.d_model, self.ff_dim, bias=self.use_transformer_bias)
        self.out_proj = nn.Linear(self.ff_dim, self.d_model, bias=self.use_transformer_bias)

        # initializing
        mean = 0
        std = (2 / (self.ff_dim + self.d_model)) ** 0.5
        nn.init.normal_(self.in_proj.weight, mean=mean, std=std)
        nn.init.normal_(self.out_proj.weight, mean=mean, std=std)
        if self.use_transformer_bias:
            nn.init.constant_(self.in_proj.bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self, x):
        # my preliminary experiments show all RELU-variants
        # work the same and slower, RELU FTW!!!
        y = F.relu(self.in_proj(x))
        y = F.dropout(y, p=self.dropout, training=self.training)
        return self.out_proj(y)


class ScaleNorm(nn.Module):
    """ScaleNorm"""
    def __init__(self, scale, eps=1e-5):
        super(ScaleNorm, self).__init__()
        self.scale = Parameter(torch.tensor(scale))
        self.eps = eps

    def forward(self, x):
        norm = self.scale / torch.norm(x, dim=-1, keepdim=True).clamp(min=self.eps)
        return x * norm


class TransformerEncoder(nn.Module):
    """Self-attention Transformer Encoder"""
    def __init__(self, args):
        super(TransformerEncoder, self).__init__()
        self.residual_dropout = args.residual_dropout
        self.n_layers = args.n_layers
        self.pre_norm = args.pre_norm

        self.atts = nn.ModuleList([MultiheadAttention(args) for _ in range(self.n_layers)])
        self.ffs = nn.ModuleList([FeedForward(args) for _ in range(self.n_layers)])

        num_scales = self.n_layers * 2 + 1 if self.pre_norm else self.n_layers * 2
        if args.scnorm:
            self.scales = nn.ModuleList([ScaleNorm(args.d_model ** 0.5) for _ in range(num_scales)])
        else:
            self.scales = nn.ModuleList([nn.LayerNorm(args.d_model) for _ in range(num_scales)])

    def forward(self, src_inputs, src_mask, graphormer_dists=None):
        pre_norm = self.pre_norm
        post_norm = not pre_norm

        x = F.dropout(src_inputs, p=self.residual_dropout, training=self.training)
        for i in range(self.n_layers):
            att = self.atts[i]
            ff = self.ffs[i]
            att_scale = self.scales[2 * i]
            ff_scale = self.scales[2 * i + 1]

            residual = x
            x = att_scale(x) if pre_norm else x
            x, _ = att(q=x, k=x, v=x, mask=src_mask, graphormer_dists=graphormer_dists)
            x = residual + F.dropout(x, p=self.residual_dropout, training=self.training)
            x = att_scale(x) if post_norm else x

            residual = x
            x = ff_scale(x) if pre_norm else x
            x = ff(x)
            x = residual + F.dropout(x, p=self.residual_dropout, training=self.training)
            x = ff_scale(x) if post_norm else x

        x = self.scales[-1](x) if pre_norm else x
        return x
