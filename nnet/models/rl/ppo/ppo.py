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

# NeuralNets
from nnet import models
from nnet import optimizers
from nnet.structs import AttrDict
from nnet import envs
from nnet import utils
from nnet.modules import ppo as ppo_networks
from nnet import schedulers

# Other
import itertools

class PPO(models.Model):

    """ Proximal Policy Optimization (PPO)

    Proximal Policy Optimization Algorithms
    https://arxiv.org/abs/1707.06347
    
    """

    def __init__(self, env_name, override_config={}, name="PPO"):
        super(PPO, self).__init__(name=name)

        # Config
        self.config = AttrDict()
        env_name_split = env_name.split("-")
        self.config.env_name = env_name_split
        self.config.env_type = env_name_split[0]
        assert self.config.env_type in ["atari"]
        # Env
        if self.config.env_type == "atari":
            self.config.env_class = envs.atari.AtariEnv
            self.config.env_params = {"game": env_name_split[1], "history_frames": 4, "img_size": (84, 84), "action_repeat": 4, "grayscale_obs": True, "noop_max": 30, "repeat_action_probability": 0.0, "full_action_space": False, "terminal_on_life_loss": True}
            self.config.time_limit = float("inf")
            self.config.time_limit_eval = 108000
        # Training
        self.config.epochs = {"atari": 40000}[self.config.env_type]
        self.config.epoch_length = {"atari": 4}[self.config.env_type] # number of grad steps per epoch, will auto ajust mini batch size
        self.config.epochs_per_exploration = {"atari": 4}[self.config.env_type] # number of epochs between each exploration
        self.config.policy_steps_per_env = {"atari": 128}[self.config.env_type] # number of policy steps per env for exploration
        self.config.num_envs = {"atari": 8}[self.config.env_type]
        self.config.batch_size = (self.config.policy_steps_per_env * self.config.num_envs) // self.config.epoch_length
        self.config.precision = torch.float32
        self.config.parallel_envs = False
        self.config.grad_init_scale = 2048.0
        self.config.collate_fn = {
            "atari": utils.CollateFn(inputs_params=[{"axis": 0}, {"axis": 1}, {"axis": 2}, {"axis": 3}, {"axis": 4}, {"axis": 5}], targets_params=[]), 
        }[self.config.env_type]
        self.config.sign_rewards = {"atari": True}[self.config.env_type]
        # Eval
        self.config.eval_policy_mode = "sample"
        self.config.eval_episodes = 10
        # Optimizer
        self.config.adam_lr = {"atari": 2.5e-4}[self.config.env_type]
        self.config.adam_eps = 1e-5
        self.config.grad_max_norm = 0.5
        self.config.anneal_lr = True
        # PPO
        self.config.clip_eps = {"atari": 0.1}[self.config.env_type]
        self.config.gamma = 0.99
        self.config.gae_lambda = 0.95
        self.config.norm_adv = True
        self.config.clip_value_loss = True
        self.config.gae = True
        # Loss Scales
        self.config.policy_loss_scale = 1.0
        self.config.value_loss_scale = 0.5
        self.config.entropy_loss_scale = {"atari": 0.01}[self.config.env_type]
        # Architecture
        self.config.shared_network = {"atari": True}[self.config.env_type]
        self.config.image_channels = {"atari": 4}[self.config.env_type]
        self.config.dim_cnn = 32
        self.config.hidden_size = {"atari": 512}[self.config.env_type]
        self.config.num_mlp_layers = 2
        self.config.policy_discrete = {"atari": True}[self.config.env_type]
        # Env Params
        self.config.eval_env_params = {}
        self.config.train_env_params = {}
        # Log
        self.config.running_rewards_momentum = 0.05

        # Override Config
        for key, value in override_config.items():
            assert key in self.config
            self.config[key] = value

        # Create Train Envs
        self.env = envs.wrappers.BatchEnv([
            envs.wrappers.ResetOnException(
                envs.wrappers.TimeLimit(
                    self.config.env_class(**dict(self.config.env_params, **self.config.train_env_params)), 
                    time_limit=self.config.time_limit[env_i] if isinstance(self.config.time_limit, list) else self.config.time_limit
                )
            )
        for env_i in range(self.config.num_envs)], parallel=self.config.parallel_envs)

        # Create Eval Env
        if self.config.eval_episodes > 0:
            self.env_eval = envs.wrappers.ResetOnException(
                envs.wrappers.TimeLimit(
                    self.config.env_class(**dict(self.config.env_params, **self.config.eval_env_params)),
                    time_limit=self.config.time_limit_eval
                )
            )
        else:
            self.env_eval = None

        # Networks
        if self.config.shared_network:
            self.encoder_net = ppo_networks.EncoderNetwork(
                dim_input_cnn=self.config.image_channels,
                dim_cnn=self.config.dim_cnn,
                dim_input_mlp=self.env.lowd_state_size if hasattr(self.env, "lowd_state_size") else None,
                num_mlp_layers=self.config.num_mlp_layers,
                hidden_size=self.config.hidden_size
            )
            self.policy_net = ppo_networks.PolicyNetwork(
                dim_input=self.config.hidden_size,
                num_actions=self.env.num_actions,
                num_mlp_layers=0,
                hidden_size=self.config.hidden_size,
                discrete=self.config.policy_discrete
            )
            self.value_net = ppo_networks.ValueNetwork(
                dim_input=self.config.hidden_size,
                num_mlp_layers=0,
                hidden_size=self.config.hidden_size
            )
        else:
            self.encoder_net = None
            self.policy_net = ppo_networks.PolicyNetwork(
                dim_input=self.env.lowd_state_size,
                num_actions=self.env.num_actions,
                num_mlp_layers=self.config.num_mlp_layers,
                hidden_size=self.config.hidden_size,
                discrete=self.config.policy_discrete
            )
            self.value_net = ppo_networks.ValueNetwork(
                dim_input=self.env.lowd_state_size,
                num_mlp_layers=self.config.num_mlp_layers,
                hidden_size=self.config.hidden_size
            )

        # Training Infos
        self.register_buffer("episodes", torch.tensor(0))
        self.register_buffer("running_rewards", torch.tensor(0.0))
        self.register_buffer("ep_rewards", torch.zeros(self.config.num_envs), persistent=False)
        self.register_buffer("ep_steps", torch.zeros(self.config.num_envs), persistent=False)
        self.register_buffer("action_step", torch.tensor(0))

    def compile(self):

        if self.config.shared_network:
            model_params = itertools.chain(
                self.encoder_net.parameters(), 
                self.policy_net.parameters(), 
                self.value_net.parameters()
            )
        else:
            model_params = itertools.chain(
                self.policy_net.parameters(), 
                self.value_net.parameters()
            )
        
        # Compile Model
        super(PPO, self).compile(
            optimizer=optimizers.Adam(params=[{
                "params": model_params, 
                "lr": schedulers.LinearDecayScheduler(value_start=self.config.adam_lr, value_end=0.0, decay_steps=self.config.epoch_length * self.config.epochs) if self.config.anneal_lr else self.config.adam_lr, 
                "grad_max_norm": self.config.grad_max_norm, 
                "eps": self.config.adam_eps
            }]), 
            losses={},
            loss_weights={},
            metrics=None,
            decoders=None
        )   

    def set_replay_buffer(self, replay_buffer):

        # Replay Buffer
        self.replay_buffer = replay_buffer

    def on_train_begin(self):

        # Reset Envs
        self.history_obs = self.env.reset()

        # Exploration in case of reload checkpoint
        self.exploration_phase()

    def on_epoch_begin(self, epoch):
        super(PPO, self).on_epoch_begin(epoch=epoch)

        # Explore
        if (epoch-1) % self.config.epochs_per_exploration == 0:
            self.exploration_phase()

    def preprocess_inputs(self, state):

        def norm_image(image):

            assert image.dtype == torch.uint8

            return image.type(torch.float32) / 255
        
        # Tuple state
        if isinstance(state, tuple):
            state = tuple(norm_image(s) if s.dim()==4 else s for s in state)
        # List of Inputs
        elif isinstance(state, list):
            state = [norm_image(s) if s.dim()==4 else s for s in state]
        # State (could be image of lowd)
        else:
            state = norm_image(state) if state.dim()==4 else state

        return state

    def exploration_phase(self):

        # Eval Mode
        training = self.training
        encoder_net_env = self.encoder_net
        policy_net_env = self.policy_net
        value_net_env = self.value_net
        if self.config.shared_network:
            encoder_net_env.eval()
        policy_net_env.eval()
        value_net_env.eval()

        # Buffers
        states = []
        values = []
        actions = []
        policy_log_probs = []
        dones = []
        rewards = []

        # Recover from history
        obs = self.history_obs

        # Policy loop
        for step in range(self.config.policy_steps_per_env):

            # Add to buffer: time t
            states.append(obs.state)
            dones.append(obs.done)

            # Forward Network
            with torch.no_grad():

                # Transfer to device
                state = self.transfer_to_device(obs.state)

                # Scale Input
                state = self.preprocess_inputs(state)

                # Encoder
                if self.config.shared_network:
                    state_enc = encoder_net_env(state)
                else:
                    state_enc = state

                # Policy
                policy_dist = policy_net_env(state_enc)

                # Value (N, 1)
                value = value_net_env(state_enc)

                # Action (N, A)
                action = policy_dist.sample()

                # Policy Log prob (N,)
                policy_log_prob = policy_dist.log_prob(action)

            # Add to buffer: time t
            values.append(value.cpu())
            actions.append(action.cpu())
            policy_log_probs.append(policy_log_prob.cpu())

            # Env Step
            if self.model_step == 0:
                obs = self.env.step(self.env.sample())
            else:
                obs = self.env.step(action.argmax(dim=-1) if self.config.policy_discrete else action)

            # Add to buffer: time t
            if self.config.sign_rewards:
                rewards.append(torch.sign(obs.reward))
            else:
                rewards.append(obs.reward)

            ###############################################################################
            # Update Infos / Buffer
            ###############################################################################

            # Update training_infos
            self.action_step += self.env.action_repeat * self.config.num_envs
            self.ep_rewards += obs.reward.to(self.ep_rewards.device)
            self.ep_steps += self.env.action_repeat

            ###############################################################################
            # Reset Env
            ###############################################################################

            # Is_last = Game Over or Time Limit
            for env_i in range(self.config.num_envs):
                if obs.is_last[env_i]:

                    # Reset Env
                    obs_reset = self.env.envs[env_i].reset()

                    # Reset State only
                    obs.state[env_i] = obs_reset.state

                    # Add Infos
                    self.add_info("episode_steps", self.ep_steps[env_i].item())
                    self.add_info("episode_reward_total", self.ep_rewards[env_i].item())

                    # Update training_infos
                    self.episodes += 1
                    self.running_rewards.fill_(self.config.running_rewards_momentum * self.ep_rewards[env_i].item() + (1 - self.config.running_rewards_momentum) * self.running_rewards)
                    self.ep_rewards[env_i] = 0.0
                    self.ep_steps[env_i] = 0

        # Save history
        self.history_obs = obs

        # Stack Buffers
        states = torch.stack(states, dim=0) # (T, N, C, H, W)
        values = torch.stack(values, dim=0).squeeze(dim=-1) # (T, N)
        actions = torch.stack(actions, dim=0) # (T, N, A)
        policy_log_probs = torch.stack(policy_log_probs, dim=0) # (T, N)
        dones = torch.stack(dones, dim=0) # (T, N)
        rewards = torch.stack(rewards, dim=0) # (T, N)

        # bootstrap value if not done
        with torch.no_grad():

            # Transfer to device
            state = self.transfer_to_device(obs.state)

            # Scale Input
            state = self.preprocess_inputs(state)

            # Encoder
            if self.config.shared_network:
                state_enc = encoder_net_env(state)
            else:
                state_enc = state

            # Last Value (N, 1)
            next_value = self.value_net(state_enc).cpu()

            # Generalized Advantage Estimation
            if self.config.gae:

                advantages = torch.zeros_like(rewards)
                lastgaelam = 0
                for t in reversed(range(self.config.policy_steps_per_env)):

                    if t == self.config.policy_steps_per_env - 1:
                        nextnonterminal = 1.0 - obs.done
                        nextvalues = next_value.squeeze(dim=-1)
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]

                    # Return - Value (Nenv,): rt+1 + not_done * gamma * Vt+1 - Vt
                    delta = rewards[t] + self.config.gamma * nextvalues * nextnonterminal - values[t]

                    advantages[t] = lastgaelam = delta + self.config.gamma * self.config.gae_lambda * nextnonterminal * lastgaelam

                # Recover Returns
                returns = advantages + values

            # Default Advantage
            else:
                returns = torch.zeros_like(rewards)
                for t in reversed(range(self.config.policy_steps_per_env)):

                    if t == self.config.policy_steps_per_env - 1:
                        nextnonterminal = 1.0 - obs.done
                        next_return = next_value.squeeze(dim=-1)
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        next_return = returns[t + 1]

                    returns[t] = rewards[t] + self.config.gamma * nextnonterminal * next_return

                advantages = returns - values

        # Update Buffer
        self.replay_buffer.update((
            states.flatten(start_dim=0, end_dim=1), # (B*N, C, H, W)
            policy_log_probs.flatten(start_dim=0, end_dim=1), # (B*N,)
            actions.flatten(start_dim=0, end_dim=1), # (B*N, A)
            advantages.flatten(start_dim=0, end_dim=1), # (B*N,)
            returns.flatten(start_dim=0, end_dim=1), # (B*N,)
            values.flatten(start_dim=0, end_dim=1), # (B*N,)
        ))

        # Default Mode
        if self.config.shared_network:
            encoder_net_env.train(mode=training)
        policy_net_env.train(mode=training)
        value_net_env.train(mode=training)

    def train_step(self, inputs, targets, precision, grad_scaler, accumulated_steps, acc_step, eval_training):

        outputs = super(PPO, self).train_step(inputs, targets, precision, grad_scaler, accumulated_steps, acc_step, eval_training)

        # Update Infos
        self.infos["episodes"] = self.episodes.item()
        self.infos["running_rewards"] = round(self.running_rewards.item(), 2)
        self.infos["action_step"] = self.action_step.item()

        return outputs
    
    def forward(self, inputs):

        # Preprocess state (uint8 to float32)
        inputs = self.preprocess_inputs(inputs)

        # Unpack Inputs 
        states, policy_logits, actions, advantages, returns, values = inputs

        ###############################################################################
        # Forward
        ###############################################################################

        # Shared Network
        if self.config.shared_network:
            states_enc = self.encoder_net(states)
        else:
            states_enc = states

        # Forward Policy / Value Networks
        policy_dist = self.policy_net(states_enc)
        new_values = self.value_net(states_enc)

        ###############################################################################
        # Losses
        ###############################################################################

        # Compute Prob Ratio
        new_policy_logits = policy_dist.log_prob(actions)
        log_prob_ratio = new_policy_logits - policy_logits
        prob_ratio = log_prob_ratio.exp()

        # Norm Advantages
        if self.config.norm_adv:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Policy loss
        policy_loss_1 = - advantages * prob_ratio
        policy_loss_2 = - advantages * torch.clamp(prob_ratio, 1 - self.config.clip_eps, 1 + self.config.clip_eps)
        policy_loss = torch.max(policy_loss_1, policy_loss_2).mean()
        self.add_loss("policy", policy_loss, weight=self.config.policy_loss_scale)

        # Value loss
        new_values = new_values.squeeze(dim=-1)
        if self.config.clip_value_loss:
            value_loss_unclipped = (new_values - returns) ** 2
            value_clipped = values + torch.clamp(new_values - values, - self.config.clip_eps, self.config.clip_eps)
            value_loss_clipped = (value_clipped - returns) ** 2
            value_loss_max = torch.max(value_loss_unclipped, value_loss_clipped)
            value_loss = 0.5 * value_loss_max.mean()
        else:
            value_loss = 0.5 * ((new_values - returns) ** 2).mean()
        self.add_loss("value", value_loss, weight=self.config.value_loss_scale)

        # Entropy Loss
        policy_ent = policy_dist.entropy()
        self.add_info("policy_ent", policy_ent.mean().item())
        self.add_loss("entropy", - policy_ent.mean(), weight=self.config.entropy_loss_scale)

    def play(self, verbose=False, policy_mode="sample"):

        assert policy_mode in ["sample", "mode", "random"], policy_mode

        # Reset
        obs = self.env_eval.reset()
        state = self.transfer_to_device(obs.state)

        total_rewards = 0
        step = 0

        # Episode loop
        while 1:

            # Representation Network
            with torch.no_grad():

                state = state.unsqueeze(dim=0)
                state = self.preprocess_inputs(state)

                if self.config.shared_network:
                    state_enc = self.encoder_net(state)
                else:
                    state_enc = state

            # Policy
            if policy_mode == "sample":
                with torch.no_grad():
                    action = self.policy_net(state_enc).sample()
            elif policy_mode == "mode":
                with torch.no_grad():
                    action = self.policy_net(state_enc).mode()
            elif policy_mode == "random":
                action = self.transfer_to_device(self.env.sample()[:1])

            # Forward Env
            obs = self.env_eval.step(action.argmax(dim=-1).squeeze(dim=0) if self.config.policy_discrete else action.squeeze(dim=0))
            state = self.transfer_to_device(obs.state)
            step += self.env_eval.action_repeat
            total_rewards += obs.reward

            # Is_last = Game Over or Time Limit
            if obs.is_last:
                break

        return total_rewards, step
    
    def eval_step(self, inputs, targets, verbose=False):

        if self.device != "cuda:0":
            self.to("cuda:0")

        # play
        if self.config.eval_policy_mode == "both":
            outputs = {}
            score, steps = self.play(verbose=verbose, policy_mode="sample")
            outputs["score_sample"] = torch.tensor(score)
            outputs["steps_sample"] = torch.tensor(steps)
            score, steps = self.play(verbose=verbose, policy_mode="mode")
            outputs["score_mode"] = torch.tensor(score)
            outputs["steps_mode"] = torch.tensor(steps)
        else:
            score, steps = self.play(verbose=verbose, policy_mode=self.config.eval_policy_mode)
            outputs = {"score": torch.tensor(score), "steps": torch.tensor(steps)}

        # Update Infos
        for key, value in outputs.items():
            self.infos["ep_{}".format(key)] = value.item()

        return {}, outputs, {}, {}