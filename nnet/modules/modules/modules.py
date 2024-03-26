# Copyright 2021, Maxime Burchi.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F

# NeuralNets
from nnet import modules

###############################################################################
# Transformer Modules
###############################################################################

class SwitchFFN(modules.Module):

    """ Mixture of Experts (MoE) Switch FFN Module, Single Routing Layer
    
    Reference: "SWITCH TRANSFORMERS: SCALING TO TRILLION PARAMETER MODELS WITH SIMPLE AND EFFICIENT SPARSITY" by Fedus et al.
    https://arxiv.org/abs/2101.03961

    """

    def __init__(self, num_experts, dim_model, dim_ffn, drop_rate, act_fun, inner_dropout, noise_eps=0.1):
        super(SwitchFFN, self).__init__()

        # Params
        self.num_experts = num_experts
        self.noise_eps = noise_eps        

        # Router
        self.router = modules.Linear(dim_model, self.num_experts)

        # Experts
        #self.experts = nn.ModuleList([module(**module_params) for _ in range(self.num_experts)])
        #self.experts = nn.ModuleList([Linear(in_features=module_params["dim_model"], out_features=module_params["dim_model"]) for _ in range(self.num_experts)])

        # Modules
        self.layernorm = modules.LayerNorm(dim_model)
        self.linear1 = modules.SwitchLinear(num_experts=num_experts, in_features=dim_model, out_features=dim_ffn)
        self.act_fun = modules.act_dict[act_fun]()
        self.dropout1 = modules.Dropout(p=drop_rate) if inner_dropout else nn.Identity()
        self.linear2 = modules.SwitchLinear(num_experts=num_experts, in_features=dim_ffn, out_features=dim_model)
        self.dropout2 = modules.Dropout(drop_rate)

    def compute_loss(self, router_probs, indices):

        # One Hot Indices (N, 1) -> (N, K)
        indices_one_hot = F.one_hot(indices.squeeze(dim=-1), self.num_experts).type(router_probs.dtype)

        # Router Count (K,)
        density = indices_one_hot.mean(axis=0)

        # Router Mean Prob (K,)
        density_proxy = router_probs.mean(axis=0)

        # Compute loss
        loss = (density_proxy * density).mean() * (self.num_experts ** 2)

        return loss

    def forward(self, x):

        # Shape
        tokens_shape = x.shape[:-1]

        # Flatten Tokens (..., Din) -> (N, Din)
        x = x.flatten(start_dim=0, end_dim=-2)

        # (N, Din) -> (N, K)
        router_logits = self.router(x)

        # Add noise for exploration across experts.
        if self.training:
            router_logits += torch.empty_like(router_logits).uniform_(1-self.noise_eps, 1+self.noise_eps)

        # Probabilities for each token of what expert it should be sent to.
        router_probs = router_logits.softmax(dim=-1)

        # Get Gate / Index (N, 1)
        gates, indices = router_probs.topk(k=1, dim=-1)

        # Compute load balancing loss.
        loss_switch = self.compute_loss(router_probs, indices)

        # Add Loss
        self.add_loss("switch", loss_switch)

        # Forward Experts
        #x = [self.experts[indices[token_id]](x[token_id:token_id+1]) for token_id in range(x.size(0))]

        # Stack Outputs (N, Dout)
        #x = torch.concat(x, dim=0)

        # Forward Modules
        x = self.layernorm(x)
        x = self.linear1(x, indices)
        x = self.act_fun(x)
        x = self.dropout1(x)
        x = self.linear2(x, indices)
        x = self.dropout2(x)

        # Gate Outputs
        x = x * gates

        # Reshape (N, Dout) -> (..., Dout)
        x = x.reshape(tokens_shape + x.shape[-1:])

        return x

class MultiHeadCrossAttentionModule(nn.Module):

    """Multi-Head Cross-Attention Module

    Args:
        dim_model: model feature dimension
        num_heads: number of attention heads
        Pdrop: residual dropout probability

    """

    def __init__(self, dim_model, num_heads, Pdrop):
        super(MultiHeadCrossAttentionModule, self).__init__()

        # Pre Norm
        self.norm = nn.LayerNorm(dim_model)

        # Multi-Head Cros-Attention
        self.mhca = modules.MultiHeadAttention(dim_model, num_heads)
            
        # Dropout
        self.dropout = nn.Dropout(Pdrop)

    def forward(self, x, x_enc, mask_enc=None):

        # Pre Norm
        x = self.norm(x)

        # Multi-Head Cross-Attention
        x, attention = self.mhca(Q=x, K=x_enc, V=x_enc, mask=mask_enc)

        # Dropout
        x = self.dropout(x)

        return x, attention



