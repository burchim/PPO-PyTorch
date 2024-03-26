import nnet
import os
import json

# Extract params from filename
env_name = os.environ["env_name"]
print("PPO selected env_name: {}".format(env_name))

# Override Config
override_config = os.environ.get("override_config", {})
print("override_config:", override_config)
if isinstance(override_config, str):
    override_config = json.loads(override_config)

# Model
model = nnet.models.rl.ppo.PPO(env_name=env_name, override_config=override_config)
model.compile()

# Training
precision = model.config.precision
grad_init_scale = model.config.grad_init_scale
epochs = model.config.epochs
epoch_length = model.config.epoch_length
eval_period_epoch = 100
saving_period_epoch = 100
num_workers = 0

# Callback Path
if os.environ.get("run_name", False):
    callback_path = "callbacks/PPO/{}/{}".format(os.environ["run_name"], env_name)
else:
    callback_path = "callbacks/PPO/{}".format(env_name)

# Replay Buffer
training_dataset = nnet.datasets.replay_buffers.PPOReplayBuffer(
    num_workers=num_workers,
    batch_size=model.config.batch_size,
    root=callback_path,
    epoch_length=epoch_length,
    collate_fn=model.config.collate_fn,
)
model.set_replay_buffer(training_dataset)

# Evaluation Dataset
evaluation_dataset = nnet.datasets.VoidDataset(num_steps=model.config.eval_episodes)