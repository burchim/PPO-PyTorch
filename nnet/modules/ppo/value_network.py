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

# NeuralNets
from nnet import modules

class ValueNetwork(nn.Module):

    def __init__(
            self,  
            dim_input,
            hidden_size=64, 
            act_fun=nn.Tanh, 
            num_mlp_layers=2, 
            weight_init={"class": "orthogonal", "params": {"gain": 2.0**0.5}}, 
            bias_init="zeros", 
            dist_weight_init={"class": "orthogonal", "params": {"gain": 1.0}}, 
            dist_bias_init="zeros"
    ):
        super(ValueNetwork, self).__init__()

        if num_mlp_layers > 0:
            
            self.mlp = modules.MultiLayerPerceptron(
                dim_input=dim_input,
                dim_layers=[hidden_size for _ in range(num_mlp_layers)],
                act_fun=act_fun,
                norm=None,
                weight_init=weight_init,
                bias_init=bias_init
            )

            self.linear_proj = modules.Linear(
                in_features=hidden_size,
                out_features=1,
                weight_init=dist_weight_init,
                bias_init=dist_bias_init
            )

        else:

            self.mlp = nn.Identity()

            self.linear_proj = modules.Linear(
                in_features=dim_input,
                out_features=1,
                weight_init=dist_weight_init,
                bias_init=dist_bias_init
            )

    def forward(self, x):

        # MLP Layers
        x = self.mlp(x)

        # Output Proj
        value = self.linear_proj(x)

        return value
    