"""
File: collect_prior.py
Author: Viet Nguyen
Date: 2024-04-04

Description: This contain the function API handling logic for collection of
  prior preference data
  NOTE: Fully tested
"""

import re

import embodied
import numpy as np

from . import callbacks

# this will collect necessary prior preference data
def collect_prior(agent, env, replays, logger, args):

  logdir = embodied.Path(args.logdir)
  logdir.mkdirs()
  print('Logdir:', logdir)
  should_log = embodied.when.Clock(args.log_every)
  replay, positive_replay = replays
  step = logger.step
  metrics = embodied.Metrics()
  print('Observation space:', embodied.format(env.obs_space), sep='\n')
  print('Action space:', embodied.format(env.act_space), sep='\n')

  timer = embodied.Timer()
  timer.wrap('agent', agent, ['policy', 'train', 'report', 'save'])
  timer.wrap('env', env, ['step'])
  timer.wrap('positive_replay', positive_replay, ['save'])
  timer.wrap('replay', replay, ['save'])
  timer.wrap('logger', logger, ['write'])


  # Driver setup
  driver = embodied.Driver(env)

  # Print every episode
  driver.on_episode(lambda ep, worker: callbacks.print_episode(ep))

  # Process every episode
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
    metrics.add(stats, prefix=('stats' if mode == 'train' else f'{mode}_stats'))
    # Since we don't log in train step, we log in here
    if should_log(step):
      agg = metrics.result()
      logger.add(agg)
      logger.write(fps=True)
    return ep
  driver.on_episode(lambda ep, worker: per_episode(ep, "prior"))

  # lookback every episode
  driver.on_episode(lambda ep, worker: callbacks.lookback_on_episode(ep, args))

  # add to trajectory every episode
  driver.on_episode(lambda ep, worker: callbacks.add_trajectory(ep, replay, positive_replay, worker)) # add trajectory is last

  # Save every episode
  # epsdir = embodied.Path(args.logdir) / 'saved_episodes'
  # epsdir.mkdirs()
  # print('Saving episodes:', epsdir)
  # saver = embodied.Worker(callbacks.save, 'thread')
  # driver.on_episode(lambda ep, worker: saver(ep, epsdir))

  # increment
  driver.on_step(lambda tran, _: step.increment())


  print('Start collecting prior data.')
  def policy(*args):
    action, state = agent.policy(*args)
    # if this is the expert action, it is expected to be a successful action that you can imitate
    can_imitate = [True for i in range(len(env))] # NOTE: make sure this has the same shape as action
    return {**action, "can_self_imitate": can_imitate}, state
  while step < args.steps:
    driver(policy, steps=100)
  logger.write()

  # Finally, save the replay
  replay.save(wait=True)
  positive_replay.save(wait=True)

