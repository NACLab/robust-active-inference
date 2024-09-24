"""
File: eval_only.py
Author: Viet Nguyen, Danijar Hafner

Description: This contain the function API handling logic for evaluating
  environment. Adapted from Dreamerv3 codebase
"""

import re

import embodied
import numpy as np
from . import callbacks

def eval_only(agent, env, logger, args):

  logdir = embodied.Path(args.logdir)
  logdir.mkdirs()
  print('Logdir', logdir)
  should_log = embodied.when.Clock(args.log_every)
  step = logger.step
  metrics = embodied.Metrics()
  print('Observation space:', env.obs_space)
  print('Action space:', env.act_space)

  timer = embodied.Timer()
  timer.wrap('agent', agent, ['policy'])
  timer.wrap('env', env, ['step'])
  timer.wrap('logger', logger, ['write'])

  def print_episode(ep):
    length = len(ep['reward']) - 1
    score = float(ep['reward'].astype(np.float64).sum())
    print(f'Episode has {length} steps and return {score:.2f}.')
    return ep

  #### Record each key value pair of the episode after every episode
  nonzeros = set()
  def per_episode(ep, mode):
    length = len(ep['reward']) - 1
    score = float(ep['reward'].astype(np.float64).sum())
    ep_stats = {
      'length': length,
      'score': score
    }
    if "relative_stability" in ep:
      ep_stats["relative_stability"] = ep["relative_stability"][-1]
    if "success_rate" in ep:
      ep_stats["success_rate"] = ep["success_rate"][-1]
    if "coverage_mean" in ep:
      ep_stats["coverage_mean"] = ep["coverage_mean"][-1]
    # record the number of steps that the agent is in the sucessful states in the episode.
    ep_stats["successfulness"] = np.sum(ep["is_success"].astype(np.float64)) / length
    logger.add(ep_stats, prefix=('episode' if mode == 'train' else f'{mode}_episode'))
    # print(f'Episode has {length} steps and return {score:.2f}.')
    stats = {}
    for key in args.log_keys_video:
      if key in ep:
        stats[f'policy_{key}'] = ep[key]
    for key, value in ep.items():
      if not args.log_zeros and key not in nonzeros and (value == 0).all():
        continue
      nonzeros.add(key)
      if re.match(args.log_keys_sum, key):
        stats[f'sum_{key}'] = ep[key].sum()
      if re.match(args.log_keys_mean, key):
        stats[f'mean_{key}'] = ep[key].mean()
      if re.match(args.log_keys_max, key):
        stats[f'max_{key}'] = ep[key].max(0).mean()
    metrics.add(stats, prefix=f'{mode}_stats')
    return ep

  driver = embodied.Driver(env)
  driver.on_episode(lambda ep, worker: print_episode(ep))
  driver.on_episode(lambda ep, worker: per_episode(ep, mode="eval"))
  driver.on_step(lambda tran, _: step.increment())

  checkpoint = embodied.Checkpoint()
  checkpoint.agent = agent
  checkpoint.load(args.from_checkpoint, keys=['agent'])

  print('Start evaluation loop.')
  policy = lambda *args: agent.policy(*args, mode='eval')
  while step < args.steps:
    driver(policy, steps=100)
    if should_log(step):
      logger.add(metrics.result())
      logger.add(timer.stats(), prefix='timer')
      logger.write(fps=True)
  logger.write()


def eval_only_augmented(agent, env, logger, args):

  logdir = embodied.Path(args.logdir)
  logdir.mkdirs()
  print('Logdir', logdir)
  should_log = embodied.when.Clock(args.log_every)
  step = logger.step
  metrics = embodied.Metrics()
  print('Observation space:', env.obs_space)
  print('Action space:', env.act_space)

  timer = embodied.Timer()
  timer.wrap('agent', agent, ['policy'])
  timer.wrap('env', env, ['step'])
  timer.wrap('logger', logger, ['write'])


  driver = embodied.Driver(env)
  driver.on_episode(lambda ep, worker: callbacks.print_episode(ep))
  nonzeros = set()
  driver.on_episode(lambda ep, worker: callbacks.per_episode(ep, "eval", nonzeros, args, logger, metrics))
  driver.on_episode(lambda ep, worker: callbacks.lookback_on_episode(ep, args))
  driver.on_step(lambda tran, _: step.increment())

  checkpoint = embodied.Checkpoint()
  checkpoint.agent = agent
  checkpoint.load(args.from_checkpoint, keys=['agent'])

  print('Start evaluation loop.')
  def policy(*args):
    action, state = agent.policy(*args, mode='eval')
    # this is not an expert action, so you cannot self imitate, self_imitate variables
    # is only reserved for expert action in the prior preference data
    can_imitate = [False for i in range(len(env))] # NOTE: make sure this has the same shape as action
    return {**action, "can_self_imitate": can_imitate}, state
  while step < args.steps:
    driver(policy, steps=100)
    if should_log(step):
      logger.add(metrics.result())
      logger.add(timer.stats(), prefix='timer')
      logger.write(fps=True)
  logger.write()


