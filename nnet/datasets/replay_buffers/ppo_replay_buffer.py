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

# NeuralNets
from nnet import datasets
from nnet import utils

# Other
import os

class PPOReplayBuffer(datasets.Dataset):

    """ PPO Replay Buffer """

    def __init__(
            self, 
            num_workers, 
            batch_size, 
            epoch_length,
            root,
            shuffle=True,
            collate_fn=utils.CollateFn(inputs_params=[{"axis": 0}, {"axis": 1}, {"axis": 2}, {"axis": 3}, {"axis": 4}, {"axis": 5}], targets_params=[]), 
            buffer_name="PPOReplayBuffer"
        ):
        super(PPOReplayBuffer, self).__init__(num_workers=num_workers, batch_size=batch_size, collate_fn=collate_fn, shuffle=shuffle, root=root)

        # Buffer Params
        self.epoch_length = epoch_length
        self.buffer_name = buffer_name # name of buffer directory
        self.buffer_dir = os.path.join(root, self.buffer_name) # Buffer Dir

    def update(self, samples):

        # Update Samples
        self.samples = samples

        # Assert lengths
        for elements in self.samples:
            assert len(elements) == self.epoch_length * self.batch_size

    def __len__(self):

        return self.epoch_length * self.batch_size

    # Run in fork process, require share mem
    def __getitem__(self, n):

        # Sample
        sample = [elements[n] for elements in self.samples]

        return sample