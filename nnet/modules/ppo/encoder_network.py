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

class EncoderNetwork(nn.Module):

    def __init__(
            self,  
            dim_input_cnn=4,
            dim_cnn=32,
            hidden_size=512,
            act_fun_cnn=nn.ReLU,
            dim_input_mlp=None,
            act_fun_mlp=nn.Tanh, 
            num_mlp_layers=2, 
            weight_init={"class": "orthogonal", "params": {"gain": 2.0**0.5}}, 
            bias_init="zeros"
    ):
        super(EncoderNetwork, self).__init__()

        assert dim_input_cnn!=None or dim_input_mlp!=None

        if dim_input_cnn is not None:

            # 84 -> 20 -> 9 -> 7
            self.cnn = nn.Sequential(
                modules.ConvNeuralNetwork(  
                    dim_input=dim_input_cnn,
                    dim_layers=[dim_cnn, 2*dim_cnn, 2*dim_cnn],
                    kernel_size=[8, 4, 3],
                    strides=[4, 2, 1],
                    act_fun=act_fun_cnn,
                    padding="valid",
                    weight_init=weight_init, 
                    bias_init=bias_init,
                    norm=None,
                    channels_last=False
                ),
                nn.Flatten(),
                modules.MultiLayerPerceptron(
                    dim_input=2*dim_cnn * 7 * 7,
                    dim_layers=hidden_size,
                    act_fun=act_fun_cnn,
                    weight_init=weight_init, 
                    bias_init=bias_init,
                    norm=None,
                )
            )

        else:
            self.cnn = None

        if dim_input_mlp is not None:
            self.mlp = modules.MultiLayerPerceptron(
                dim_input=dim_input_mlp,
                dim_layers=[hidden_size for _ in range(num_mlp_layers)],
                act_fun=act_fun_mlp,
                norm=None,
                weight_init=weight_init,
                bias_init=bias_init
            )
        else:
            self.mlp = None

    def forward_mlp(self, x):

        # MLP Layers
        x = self.mlp(x)

        return x
    
    def forward_cnn(self, x):

        # MLP Layers
        x = self.cnn(x)

        return x

    def forward(self, inputs):

        # To list
        if isinstance(inputs, torch.Tensor):
            inputs = [inputs]

        # Parse inputs
        inputs_cnn = []
        inputs_mlp = []
        for input in inputs:
            # (B, C, H, W)
            if input.dim() == 4:
                inputs_cnn.append(input)
            # (B, D)
            elif input.dim() == 2:
                inputs_mlp.append(input)
            else:
                raise Exception("input has dim shape: {}".format(input.shape))

        # Forward
        outputs = []
        if len(inputs_cnn) > 0:
            outputs.append(self.forward_cnn(torch.cat(inputs_cnn, dim=-1)))
        if len(inputs_mlp) > 0:
            outputs.append(self.forward_mlp(torch.cat(inputs_mlp, dim=-1)))
        
        # Concat outputs
        outputs = torch.cat(outputs, dim=-1)

        return outputs
    