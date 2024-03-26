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
from torch.nn import functional as F
from torch.distributions.kl import register_kl

class OneHotDist(torch.distributions.one_hot_categorical.OneHotCategoricalStraightThrough):

  def __init__(self, probs=None, logits=None, validate_args=None, uniform_mix=0.0):

    # Uniform Mix
    if uniform_mix > 0 and logits is not None:
      probs = F.softmax(logits, dim=-1)
      probs = (1 - uniform_mix) * probs + uniform_mix / probs.shape[-1]
      logits = torch.log(probs)
      super(OneHotDist, self).__init__(logits=logits, probs=None, validate_args=validate_args)
    else:
      super(OneHotDist, self).__init__(logits=logits, probs=probs, validate_args=validate_args)

  def mode(self):
    mode = super(OneHotDist, self).mode
    return mode.detach() + (self.logits - self.logits.detach())
  
"""
tf.reduce_sum(
        (tf.math.softmax(a_logits) *
         (tf.math.log_softmax(a_logits) - tf.math.log_softmax(b_logits))),
        axis=-1)

  
@register_kl(OneHotDist, OneHotDist)
def _kl_categorical_one_hot_categorical(p, q):
  t = p.probs * (p.logits - q.logits)
  #t[(q.probs == 0).expand_as(t)] = inf
  #t[(p.probs == 0).expand_as(t)] = 0
  return t.sum(-1)
"""