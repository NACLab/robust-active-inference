"""
File: train_prior.py
Author: Viet Nguyen
Date: 2024-04-04

Description: This contain the function API handling logic for training of
  prior preference data
"""

import os
import numpy as np
import gymnasium as gym
import stable_baselines3
import sb3_contrib
from stable_baselines3 import HerReplayBuffer
import embodied

from stable_baselines3.common.callbacks import BaseCallback
class CustomCallback(BaseCallback):
  """
  A custom callback that derives from ``BaseCallback``.

  :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
  """
  def __init__(self, verbose: int = 0):
      super().__init__(verbose)
      # Those variables will be accessible in the callback
      # (they are defined in the base class)
      # The RL model
      # self.model = None  # type: BaseAlgorithm
      # An alias for self.model.get_env(), the environment used for training
      # self.training_env # type: VecEnv
      # Number of time the callback was called
      # self.n_calls = 0  # type: int
      # num_timesteps = n_envs * n times env.step() was called
      # self.num_timesteps = 0  # type: int
      # local and global variables
      # self.locals = {}  # type: Dict[str, Any]
      # self.globals = {}  # type: Dict[str, Any]
      # The logger object, used to report things in the terminal
      # self.logger # type: stable_baselines3.common.logger.Logger
      # Sometimes, for event callback, it is useful
      # to have access to the parent object
      # self.parent = None  # type: Optional[BaseCallback]

      self._eps_data = []

  def _per_episode(self, ep, prefix="episode"):
    length = len(ep['reward']) - 1
    score = float(ep['reward'].astype(np.float64).sum())
    self.logger.record(f"{prefix}/length", length)
    self.logger.record(f"{prefix}/score", score)
    if "relative_stability" in ep:
      relative_stability = ep["relative_stability"][-1]
      self.logger.record(f"{prefix}/relative_stability", relative_stability)
    if "success_rate" in ep:
      success_rate = ep["success_rate"][-1]
      self.logger.record(f"{prefix}/success_rate", success_rate)
    if "coverage_mean" in ep:
      coverage_mean = ep["coverage_mean"][-1]
      self.logger.record(f"{prefix}/coverage_mean", coverage_mean)
    # record the number of steps that the agent is in the sucessful states in the episode.
    successfulness = np.sum(ep["is_success"].astype(np.float64)) / length
    self.logger.record(f"{prefix}/successfulness", successfulness)

  def _on_step(self) -> bool:
    """
    This method will be called by the model after each call to `env.step()`.

    For child callback (of an `EventCallback`), this will be called
    when the event is triggered.

    :return: If the callback returns False, training is aborted early.
    """
    obs = self.locals.get("infos")[0]["other_obs"] # return a list of info
    self._eps_data.append(obs.copy())

    if obs["is_last"] or obs["is_terminal"]:
      ep = {k: embodied.convert([self._eps_data[i][k] for i in range(len(self._eps_data))]) for k in self._eps_data[0].keys()}
      self._per_episode(ep)

      # reset
      self._eps_data = []

    return True


def train_prior(gym_env: gym.Env, config: embodied.Config):
  suite, env_name = config.task.split("_", 1)

  # Model class for expert
  try:
    model_cls = getattr(sb3_contrib, config.experts[suite].model)
  except:
    model_cls = getattr(stable_baselines3, config.experts[suite].model)

  # now train the expert model or load the model in
  if not os.path.exists(f"./{config.expert_models_dir}/{config.experts[suite].model_name}_{env_name}.zip"):
    kw = {"policy_kwargs": dict(config.experts[suite].policy_kwargs)} if "policy_kwargs" in config.experts[suite] else {}
    # NOTE: since embodied.config force to be the same type, we have to work around the shared layers
    # more information about the shared layer structure: https://stable-baselines3.readthedocs.io/en/sde/guide/custom_policy.html
    if "policy_kwargs" in config.experts[suite]:
      if "net_arch" in config.experts[suite].policy_kwargs:
        if "shared" in config.experts[suite].policy_kwargs.net_arch:
          policy_kw = dict(config.experts[suite].policy_kwargs)
          shared = policy_kw['net_arch'].pop('shared')
          policy_kw['net_arch'] = [*shared, dict(policy_kw['net_arch'])]
          kw = {"policy_kwargs": policy_kw}
    print(f"[SB3] Policy kwargs: {kw}")

    # TODO: Take into account net arch? `net_arch=dict(pi=[32, 32], vf=[64, 64])`
    # https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html
    try:
      # Initialize the model
      replay_buffer_cls = None
      replay_buffer_kwargs = None
      if config.experts[suite].use_her:
        replay_buffer_cls = HerReplayBuffer
        replay_buffer_kwargs = dict(
          n_sampled_goal=4,
          goal_selection_strategy=config.experts[suite].goal_selection_strategy,
        )
      model: sb3_contrib.TQC = model_cls(
        config.experts[suite].input_policy,
        gym_env,
        replay_buffer_class=replay_buffer_cls,
        replay_buffer_kwargs=replay_buffer_kwargs,
        verbose=1,
        **kw
      )
    except:
      print("Class did not use specified replay buffer, initializing agent with default option")
      model: sb3_contrib.TQC = model_cls(
        config.experts[suite].input_policy,
        gym_env,
        verbose=1,
        **kw
      )

    # Train model
    callback = CustomCallback(verbose=0)
    model.learn(config.experts[suite].timesteps, callback=callback)

    # Save model
    os.makedirs(config.expert_models_dir, exist_ok=True)
    model.save(f"./{config.expert_models_dir}/{config.experts[suite].model_name}_{env_name}")
    del model # remove to demonstrate saving and loading
  else:
    model: sb3_contrib.TQC = model_cls.load(f"./{config.expert_models_dir}/{config.experts[suite].model_name}_{env_name}", env=gym_env)
    # Train model
    callback = CustomCallback(verbose=0)
    model.learn(config.experts[suite].timesteps, callback=callback)
    # Save model
    os.makedirs(config.expert_models_dir, exist_ok=True)
    model.save(f"./{config.expert_models_dir}/{config.experts[suite].model_name}_{env_name}")
    del model # remove to demonstrate saving and loading


