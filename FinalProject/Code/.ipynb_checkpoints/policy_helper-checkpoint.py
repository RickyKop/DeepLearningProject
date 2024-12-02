# imports for helper functions
import gymnasium as gym
import os
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList


# for video
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv

# import Policy Optimization Models
from stable_baselines3 import PPO
from stable_baselines3 import A2C

# records a video
def gen_video(vec_env, model, video_name):
  env_id = "ALE-Boxing-v5"
  video_folder = "logs/videos/"
  video_length = 500

  # obs is our initial observation
  obs = vec_env.reset()

  # Record the video starting at the first step
  vec_env = VecVideoRecorder(vec_env, video_folder,
                        record_video_trigger=lambda x: x == 0, video_length=video_length,
                        name_prefix=video_name)

  vec_env.reset()
  for _ in range(video_length + 1):
    # Predict action using the observations
    action, _ = model.predict(obs, deterministic=True)
    # environment processes action, returns next observations
    obs, _, _, _ = vec_env.step(action)
  # Save the video
  vec_env.close()

# returns an agent according to specifications
def create_agent(vec_env, model_type = "PPO", policy = "CnnPolicy", tensorboard = None):
  model_type = "PPO"
  model = None

  # recurrent variant of PPO utilizes temporal dependencies when 
  # sequential decision making is useful - check it out in Contrib Repo

  # PPO may run better on CPU than GPU when using MLP, but may run better on GPU
  # when running CNN...

  if model_type == "PPO":
    model = PPO("CnnPolicy", vec_env, verbose = 1, tensorboard_log = tensorboard, device="cuda")

  # A2C performs better than A3C with GPU's
  if model_type == "A2C":
    model = A2C("CnnPolicy", vec_env, verbose = 1, device="cuda")
  
  return model

#__________________________________________________________________________

# custom environments for custom reward functions
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.wrappers import TransformReward

# returns an environment whose reward encourages only aggresive RL Agent
def get_hard_hitter_boxing_env():
  def reward_hard_hit(reward):
    # if reward is negative, then we got hit, but we don't care about that
    # thus taking the max will ignore all negative rewards
    new_reward = max(0, reward)
    return new_reward
  base_env = gym.make("ALE/Boxing-v5")
  hard_hitter_boxing = TransformReward(base_env, reward_hard_hit)
  return hard_hitter_boxing

# returns an environment whose reward encourages only defensive RL Agent
def get_doge_all_boxing_env():
  def reward_dodge_all(reward):
    # if reward is positive, then we did a hit, but we don't care about that
    # thus taking the min will ignore all positive rewards
    new_reward = min(0, reward)
    return new_reward
  base_env = gym.make("ALE/Boxing-v5")
  dodge_all_boxing = TransformReward(base_env, reward_dodge_all)
  return dodge_all_boxing

# returns an environment whose reward encourages a soft defensive RL Agent
def get_dodge_soft_boxing_env(soft = .5):
  def reward_dodge_soft(reward):
    # if reward is positive, then we did a hit, but we scale it down
    # since we prioritize dodging, and are soft on landing hits
    if (reward > 0):
      return reward * soft
    else:
      return reward
  base_env = gym.make("ALE/Boxing-v5")
  dodge_soft_boxing = TransformReward(base_env, reward_dodge_soft)
  return dodge_soft_boxing

# returns an environment whose reward encourages a soft aggressive RL Agent
def get_hit_soft_boxing_env(soft = .5):
  def reward_hit_soft(reward):
    # if reward is negative, then we got hit, but we scale it down
    # since we prioritize hitting, and are soft on getting hit
    if (reward < 0):
      return reward * soft
    else:
      return reward
  base_env = gym.make("ALE/Boxing-v5")
  hit_soft_boxing = TransformReward(base_env, reward_hit_soft)
  return hit_soft_boxing
#__________________________________________________________________________
class variant_checkpointCallback(CheckpointCallback):
  def __init__(
        self,
        save_freq: int,
        save_path: str,
        name_prefix: str = "rl_model",
        save_replay_buffer: bool = False,
        save_vecnormalize: bool = False,
        verbose: int = 0,
        save_name: str = "Default Save"
    ):
    super().__init__(save_freq, save_path, name_prefix, save_replay_buffer, save_vecnormalize, verbose)
    self.save_name = save_name
  def _checkpoint_path(self, checkpoint_type: str = "", extension: str = "") -> str:
    return os.path.join(self.save_path,  self.save_name + ".zip")














