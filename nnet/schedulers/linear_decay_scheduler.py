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

# Neural Nets
from nnet.schedulers import Scheduler

class LinearDecayScheduler(Scheduler):

    def __init__(self, value_start, value_end, decay_steps):
        super(LinearDecayScheduler, self).__init__()

        # Scheduler Params
        self.value_start = value_start
        self.value_end = value_end
        self.decay_steps = decay_steps

    def get_val_step(self, step):

        # Compute Value
        if step >= self.decay_steps:
            val = self.value_end
        else:
            val = self.value_start - step * (self.value_start - self.value_end) / self.decay_steps

        return val