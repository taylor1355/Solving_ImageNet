
from asyncio.log import logger
from audioop import bias
from grpc import xds_server_credentials
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

class BidirectionalAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        qkv_bias=True,
        attn_drop=0.,
        proj_drop=0.,
        args=None
    ):
        super().__init__()

        self.qkv_bias = qkv_bias
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.embed_dim = head_dim
        self.scale = head_dim ** -0.5
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop

        self.args = args

        self.activation = nn.GELU()

        self.reverse_parameters = []
        self.forward_parameters = []
        self.attention_parameters = []
        self.output_parameters = []

        self.instantiate_scoring_weights()
        self.instantiate_generator_weights()
        self.instantiate_output_weights()

        self.softmax = nn.Softmax(dim=-1)

    def instantiate_scoring_weights(self):
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=self.qkv_bias)
        self.attention_parameters.append(self.qkv)

    def instantiate_generator_weights(self):
        if self.args.is_bidirectional:
            self.G = nn.Linear(self.embed_dim, self.embed_dim * self.embed_dim, bias=False)
            self.bias_generator = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
            self.local_proj = nn.Linear(self.dim, self.dim, bias=self.qkv_bias)
            self.global_proj = nn.Linear(self.dim, self.dim, bias=self.qkv_bias)

            self.selection_lambda = nn.Parameter(torch.tensor(self.args.initial_lambda))
            self.selection_lambda.requires_grad = not self.args.freeze_lambda

            self.reverse_parameters += [
                self.selection_lambda,
                self.G,
                self.bias_generator,
                self.local_proj,
                self.global_proj
            ]

            if self.args.layer_norm:
                self.sa_norm = nn.LayerNorm(normalized_shape=self.dim, elementwise_affine=self.args.ln_affine_transform)
                self.isa_norm = nn.LayerNorm(normalized_shape=self.dim, elementwise_affine=self.args.ln_affine_transform)

                self.reverse_parameters += [
                    self.sa_norm,
                    self.isa_norm
                ]
            elif self.args.ln_affine_transform:
                raise ValueError("affine_transform=True requires layer_norm=True")


    def instantiate_output_weights(self):
        self.attn_drop = nn.Dropout(self.attn_drop)
        self.proj = nn.Linear(self.dim, self.dim)
        self.proj_drop = nn.Dropout(self.proj_drop)
        self.output_parameters += [self.attn_drop, self.proj, self.proj_drop]

    def apply_forward_attention(self, x, attn):
        B, N, C = x.shape
        v = self.forward_v(x).reshape(B, N, self.num_heads, C // self.num_heads).transpose(2, 1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return x

    def apply_inverse_attention(self, x, attn):
        B, N, C = x.shape
        global_inputs = self.activation(self.global_proj(x).reshape(B, N, self.num_heads, self.embed_dim).transpose(-3, -2))
        local_inputs = self.activation(self.local_proj(x).reshape(B, N, self.num_heads, self.embed_dim).transpose(-3, -2))

        global_summaries = (attn @ global_inputs)

        global_weights = self.G(global_inputs).reshape(B, self.num_heads, N, self.embed_dim, self.embed_dim)
        Wx = (local_inputs.unsqueeze(-2) @ global_weights).squeeze(-2).transpose(-3, -2).flatten(2)
        bias = self.bias_generator(global_summaries).transpose(-3, -2).flatten(2)
        output = Wx + bias
        return output

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        reverse_attn = F.softmax(attn, dim=-2)
        attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        reverse_attn = self.attn_drop(reverse_attn)
        sa_outputs = (attn @ v).transpose(1, 2).reshape(B, N, C)

        if self.args.is_bidirectional:
            isa_outputs = self.apply_inverse_attention(x, reverse_attn)

            if self.args.layer_norm:
                tensor_shape = sa_outputs.shape
                if not self.args.no_ln_flatten:
                    sa_outputs = sa_outputs.flatten(0, 1)
                    isa_outputs = isa_outputs.flatten(0, 1)

                if self.args.ln_all_dim:
                    sa_outputs = F.layer_norm(sa_outputs, (B, N, C))
                    isa_outputs = F.layer_norm(isa_outputs, (B, N, C))
                elif self.args.group_norm:
                    sa_outputs = sa_outputs.reshape(B, N, self.num_heads, C // self.num_heads)
                    sa_outputs = F.layer_norm(sa_outputs, [C // self.num_heads]).reshape(*tensor_shape)

                    isa_outputs = isa_outputs.reshape(B, N, self.num_heads, C // self.num_heads)
                    isa_outputs = F.layer_norm(isa_outputs, [C // self.num_heads]).reshape(*tensor_shape)
                else:
                    sa_outputs = self.sa_norm(sa_outputs).reshape(*tensor_shape)
                    isa_outputs = self.isa_norm(isa_outputs).reshape(*tensor_shape)

            forget_weight = torch.sigmoid(self.selection_lambda)
            output = sa_outputs * forget_weight + isa_outputs * (1. - forget_weight)
        else:
            output = sa_outputs

        x = self.proj(output)
        x = self.proj_drop(x)
        return x
