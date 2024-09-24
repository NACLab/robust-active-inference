"""
File: base.py
Author: Viet Nguyen

Description: This contain the top-most level API such as creating environments,
  logger, wrappers. These API will be used in (almost) every experiment.
  Again, adapted from Danijar Hafner's dreamerv3 code.
"""

import importlib
import pathlib
import sys
import warnings
from functools import partial as bind
import os

import embodied
from embodied import wrappers


def make_logger(parsed, logdir, step, config):
  multiplier = config.env.get(config.task.split('_')[0], {}).get('repeat', 1)
  logger = embodied.Logger(step, [
      embodied.logger.TerminalOutput(config.filter),
      embodied.logger.JSONLOutput(logdir, 'metrics.jsonl'),
      embodied.logger.JSONLOutput(logdir, 'scores.jsonl', 'episode/score'),
      embodied.logger.TensorBoardOutput(logdir),
  ], multiplier)
  return logger


def make_replay(
    config, directory=None, is_eval=False, rate_limit=False, **kwargs):
  assert config.replay == 'uniform' or not rate_limit
  length = config.batch_length
  size = config.replay_size // 10 if is_eval else config.replay_size
  if config.replay == 'uniform' or is_eval:
    kw = {'online': config.replay_online}
    if rate_limit and config.run.train_ratio > 0:
      kw['samples_per_insert'] = config.run.train_ratio / config.batch_length
      kw['tolerance'] = 10 * config.batch_size
      kw['min_size'] = config.batch_size
    replay = embodied.replay.Uniform(length, size, directory, **kw)
  elif config.replay == 'reverb':
    replay = embodied.replay.Reverb(length, size, directory)
  elif config.replay == 'chunks':
    replay = embodied.replay.NaiveChunks(length, size, directory)
  else:
    raise NotImplementedError(config.replay)
  return replay


def make_envs(config, **overrides):
  suite, task = config.task.split('_', 1)
  ctors = []
  for index in range(config.envs.amount):
    ctor = lambda: make_env(config, **overrides)
    if config.envs.parallel != 'none':
      ctor = bind(embodied.Parallel, ctor, config.envs.parallel)
    if config.envs.restart:
      ctor = bind(wrappers.RestartOnException, ctor)
    ctors.append(ctor)
  envs = [ctor() for ctor in ctors]
  return embodied.BatchEnv(envs, parallel=(config.envs.parallel != 'none'))


def make_env(config, **overrides):
  suite, task = config.task.split('_', 1)
  if config.rc and (suite == "gymrobotics" or suite == "metaworld"):
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    os.environ["MUJOCO_GL"] = "egl"
  ctor = {
    'dummy': 'embodied.envs.dummy:Dummy',
    'gym': 'embodied.envs.gym:GymEnv',
    'openaigym': 'embodied.envs.openaigym:GymEnv',
    'metaworld': 'embodied.envs.metaworld:MetaWorldEnv',
    'robosuite': 'embodied.envs.robosuite:RobosuiteEnv',
  }[suite]
  if isinstance(ctor, str):
    module, cls = ctor.split(':')
    module = importlib.import_module(module)
    ctor = getattr(module, cls)
  kwargs = config.env.get(suite, {})
  kwargs.update(overrides)
  env = ctor(task, **kwargs)
  return wrap_env(env, config)


def wrap_env(env, config):
  args = config.wrapper
  for name, space in env.act_space.items():
    if name == 'reset':
      continue
    elif space.discrete:
      env = wrappers.OneHotAction(env, name)
    elif args.discretize:
      env = wrappers.DiscretizeAction(env, name, args.discretize)
    else:
      env = wrappers.NormalizeAction(env, name)
  if args.repeat > 1:
    env = wrappers.ActionRepeat(env, args.repeat)
  env = wrappers.ExpandScalars(env)
  if args.stateconcat:
    env = wrappers.StateConcatenation(env, args.stateconcat, args.stateconcat_target)
  if args.delta:
    env = wrappers.TransitionDelta(env, args.delta)
  if "additional_action_spaces" in args and args.additional_action_spaces:
    for key in args.additional_action_spaces:
      env = wrappers.ActionSpaceField(env, key=key)
  if args.length:
    env = wrappers.TimeLimit(env, args.length, args.reset)
  if args.checks:
    env = wrappers.CheckSpaces(env)
  for name, space in env.act_space.items():
    if not space.discrete:
      env = wrappers.ClipAction(env, name)
  # Environment measurement
  env = wrappers.EnvStepLambdaWrapper(env, wrappers.MeasureSuccessRate(config.task))
  env = wrappers.EnvStepLambdaWrapper(env, wrappers.MeasureRStability(100, 100, reward_offset=args.r_reward_offset))
  env = wrappers.EnvStepLambdaWrapper(env, wrappers.MeasureCoverage(config.task, env.obs_space))
  if "gym_env_final" in args and args.gym_env_final: # the observation key that we are having
    env = wrappers.GymWrapperFinalLayer(env, args.gym_env_final)
  return env
