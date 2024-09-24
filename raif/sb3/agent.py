"""
File: agent.py
Author: Viet Nguyen
Date: 2024-04-04

Description: This file define the agent that use stable-baseline3 loaded
  model as its policy
"""

import os
import embodied
import numpy as np
from ruamel import yaml

import stable_baselines3
import sb3_contrib

class ExpertAgent(embodied.Agent):

  configs = yaml.YAML(typ='safe').load(
      (embodied.Path(__file__).parent / 'configs_expert.yaml').read())

  def __init__(self, obs_space, act_space, step, config):
    self.config = config

  def setup(self, pseudo_env):
    ########## Model setup #########
    suite, env_name = self.config.task.split("_", 1)
    self.obs_key = self.config.experts[suite].obs_key
    gym_wrapped_env = embodied.wrappers.GymWrapperFinalLayer(pseudo_env, obs_key=self.obs_key)
    try:
      model_cls = getattr(sb3_contrib, self.config.experts[suite].model)
    except:
      model_cls = getattr(stable_baselines3, self.config.experts[suite].model)
    self.actor: sb3_contrib.TQC = model_cls.load(f"{self.config.expert_models_dir}/{self.config.experts[suite].model_name}_{env_name}", env=gym_wrapped_env)
    ########## End expert setup ##########

  def policy(self, obs, state=None, mode='train'):
    # "policy(obs, state=None, mode='train') -> act, state"
    if isinstance(self.obs_key, str):
      expert_obs = obs[self.obs_key]
    elif isinstance(self.obs_key, list | tuple):
      expert_obs = {k: obs[k] for k in self.obs_key}
    action, _ = self.actor.predict(expert_obs, deterministic=True)
    return {"action": action}, None


class POMDPAgent(embodied.Agent):

  configs = yaml.YAML(typ='safe').load(
      (embodied.Path(__file__).parent / 'configs_pomdp.yaml').read())

  def __init__(self, obs_space, act_space, step, config):
    pass



class MDPAgent(embodied.Agent):
  configs = yaml.YAML(typ='safe').load(
      (embodied.Path(__file__).parent / 'configs_mdp.yaml').read())

  def __init__(self, obs_space, act_space, step, config):
    pass

