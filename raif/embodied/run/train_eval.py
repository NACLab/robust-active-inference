"""
File: train_eval.py
Author: Viet Nguyen, Danijar Hafner

Description: This contain the function API handling logic for training and
  evaluation. Adapted from Dreamerv3 codebase
"""

import re

import embodied
import numpy as np

from . import callbacks

# This method will be used by dreamerv3
def train_eval(
    agent, train_env, eval_env, train_replay, eval_replay, logger, args):

  logdir = embodied.Path(args.logdir)
  logdir.mkdirs()
  print('Logdir', logdir)
  should_expl = embodied.when.Until(args.expl_until)
  should_train = embodied.when.Ratio(args.train_ratio / args.batch_steps)
  if "train_every" in args and args.train_every > 0:
    should_train = embodied.when.Every(args.train_every)
  should_log = embodied.when.Clock(args.log_every)
  should_save = embodied.when.Clock(args.save_every)
  should_eval = embodied.when.Every(args.eval_every, args.eval_initial)
  should_sync = embodied.when.Every(args.sync_every)
  step = logger.step
  updates = embodied.Counter()
  metrics = embodied.Metrics()
  print('Observation space:', embodied.format(train_env.obs_space), sep='\n')
  print('Action space:', embodied.format(train_env.act_space), sep='\n')

  timer = embodied.Timer()
  timer.wrap('agent', agent, ['policy', 'train', 'report', 'save'])
  timer.wrap('env', train_env, ['step'])
  if hasattr(train_replay, '_sample'):
    timer.wrap('replay', train_replay, ['_sample'])

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
    metrics.add(stats, prefix=('stats' if mode == 'train' else f'{mode}_stats'))
    return ep

  driver_train = embodied.Driver(train_env)
  driver_train.on_episode(lambda ep, worker: print_episode(ep))
  driver_train.on_episode(lambda ep, worker: per_episode(ep, mode='train'))
  driver_train.on_step(lambda tran, _: step.increment())
  driver_train.on_step(train_replay.add)
  driver_eval = embodied.Driver(eval_env)
  driver_eval.on_step(eval_replay.add)
  driver_eval.on_episode(lambda ep, worker: print_episode(ep))
  driver_eval.on_episode(lambda ep, worker: per_episode(ep, mode='eval'))

  random_agent = embodied.RandomAgent(train_env.act_space)
  print('Prefill train dataset.')
  while len(train_replay) < max(args.batch_steps, args.train_fill):
    driver_train(random_agent.policy, steps=100)
  print('Prefill eval dataset.')
  while len(eval_replay) < max(args.batch_steps, args.eval_fill):
    driver_eval(random_agent.policy, steps=100)
  logger.add(metrics.result())
  logger.write()

  dataset_train = agent.dataset(train_replay.dataset)
  dataset_eval = agent.dataset(eval_replay.dataset)
  state = [None]  # To be writable from train step function below.
  batch = [None]
  def train_step(tran, worker):
    for _ in range(should_train(step)):
      with timer.scope('dataset_train'):
        batch[0] = next(dataset_train)
      outs, state[0], mets = agent.train(batch[0], state[0])
      metrics.add(mets, prefix='train')
      if 'priority' in outs:
        train_replay.prioritize(outs['key'], outs['priority'])
      updates.increment()
    if should_sync(updates):
      agent.sync()
    if should_log(step):
      logger.add(metrics.result())
      logger.add(agent.report(batch[0]), prefix='report')
      with timer.scope('dataset_eval'):
        eval_batch = next(dataset_eval)
      logger.add(agent.report(eval_batch), prefix='eval')
      logger.add(train_replay.stats, prefix='replay')
      logger.add(eval_replay.stats, prefix='eval_replay')
      logger.add(timer.stats(), prefix='timer')
      logger.write(fps=True)
  driver_train.on_step(train_step)

  checkpoint = embodied.Checkpoint(logdir / 'checkpoint.ckpt')
  checkpoint.step = step
  checkpoint.agent = agent
  checkpoint.train_replay = train_replay
  checkpoint.eval_replay = eval_replay
  if args.from_checkpoint:
    checkpoint.load(args.from_checkpoint)
  checkpoint.load_or_save()
  should_save(step)  # Register that we jused saved.

  print('Start training loop.')
  policy_train = lambda *args: agent.policy(
      *args, mode='explore' if should_expl(step) else 'train')
  policy_eval = lambda *args: agent.policy(*args, mode='eval')
  while step < args.steps:
    if should_eval(step):
      print('Starting evaluation at step', int(step))
      driver_eval.reset()
      driver_eval(policy_eval, episodes=max(len(eval_env), args.eval_eps))
    driver_train(policy_train, steps=100)
    if should_save(step):
      checkpoint.save()
  logger.write()
  logger.write()
  checkpoint.save()


# this method will be used by active inference agent
def train_eval_augmented(
    agent, train_env, eval_env, train_replay, positive_replay, eval_replay, logger, args):

  logdir = embodied.Path(args.logdir)
  logdir.mkdirs()
  print('Logdir', logdir)
  should_train = embodied.when.Ratio(args.train_ratio / args.batch_steps)
  if "train_every" in args and args.train_every > 0:
    should_train = embodied.when.Every(args.train_every)
  should_log = embodied.when.Clock(args.log_every)
  should_save = embodied.when.Clock(args.save_every)
  should_eval = embodied.when.Every(args.eval_every, args.eval_initial)
  should_sync = embodied.when.Every(args.sync_every)
  step = logger.step
  updates = embodied.Counter()
  metrics = embodied.Metrics()
  print('Observation space:', embodied.format(train_env.obs_space), sep='\n')
  print('Action space:', embodied.format(train_env.act_space), sep='\n')

  timer = embodied.Timer()
  timer.wrap('agent', agent, ['policy', 'train', 'report', 'save'])
  timer.wrap('env', train_env, ['step'])
  if hasattr(train_replay, '_sample'):
    timer.wrap('replay', train_replay, ['_sample'])


  driver_train = embodied.Driver(train_env)
  driver_train.on_episode(lambda ep, worker: callbacks.print_episode(ep))
  nonzeros = set()
  driver_train.on_episode(lambda ep, worker: callbacks.per_episode(ep, "train", nonzeros, args, logger, metrics))
  driver_train.on_episode(lambda ep, worker: callbacks.lookback_on_episode(ep, args))
  driver_train.on_episode(lambda ep, worker: callbacks.add_trajectory(ep, train_replay, positive_replay, worker)) # add trajectory is last
  driver_train.on_step(lambda tran, _: step.increment())
  driver_eval = embodied.Driver(eval_env)
  driver_eval.on_step(eval_replay.add)
  driver_eval.on_episode(lambda ep, worker: callbacks.print_episode(ep))
  driver_eval.on_episode(lambda ep, worker: callbacks.per_episode(ep, "eval", nonzeros, args, logger, metrics))

  random_agent = embodied.RandomAgent(train_env.act_space)
  # Prefilling
  if len(train_replay) < args.batch_steps:
    print('Do not have enough data. Prefilling train dataset with random actions.')
    def random_policy(*args):
      action, state = random_agent.policy(*args)
      can_imitate = [False for i in range(len(train_replay))] # NOTE: make sure this has the same shape as action
      return {**action, "can_self_imitate": can_imitate}, state
    while len(train_replay) < args.batch_steps:
      driver_train(random_policy, steps=100)
    train_replay.save(wait=True)
    logger.add(metrics.result())
    logger.write()

  print('Prefill eval dataset.')
  while len(eval_replay) < max(args.batch_steps, args.eval_fill):
    driver_eval(random_agent.policy, steps=100)
  logger.add(metrics.result())
  logger.write()

  dataset_train = agent.dataset(train_replay.dataset)
  dataset_eval = agent.dataset(eval_replay.dataset)
  positive_dataset = agent.dataset(positive_replay.dataset)
  state = [None, None]  # To be writable from train step function below.
  batch = [None, None]  # To be writable from train step function below.
  def train_step(tran, worker):
    for i in range(should_train(step)):
      with timer.scope('dataset_train'):
        if i % 2 == 0 or len(positive_replay) < args.batch_steps:
          # if there is not enough positive data, also fall back to using normal replay
          batch[0] = next(dataset_train) # Half train with normal dataset
          outs, state[0], mets = agent.train(batch[0], state[0])
        else:
          # NOTE: Because we are doing the prior data sampling we are guaranteed to have
          batch[1] = next(positive_dataset) # Half train with positive dataset
          outs, state[1], mets = agent.train(batch[1], state[1])
      metrics.add(mets, prefix='train')
      if 'priority' in outs:
        train_replay.prioritize(outs['key'], outs['priority'])
      updates.increment()
    if should_sync(updates):
      agent.sync()
    if should_log(step):
      logger.add(metrics.result())
      tricky_number = np.random.randint(0, 2)
      tricky_number = (tricky_number + 1) % 2 if batch[tricky_number] is None else tricky_number
      report = agent.report(batch[tricky_number])
      logger.add(report, prefix='report')
      with timer.scope('dataset_eval'):
        eval_batch = next(dataset_eval)
      logger.add(agent.report(eval_batch), prefix='eval')
      logger.add(train_replay.stats, prefix='replay')
      logger.add(eval_replay.stats, prefix='eval_replay')
      logger.add(timer.stats(), prefix='timer')
      logger.write(fps=True)
  driver_train.on_step(train_step)

  checkpoint = embodied.Checkpoint(logdir / 'checkpoint.ckpt')
  checkpoint.step = step
  checkpoint.agent = agent
  checkpoint.train_replay = train_replay
  checkpoint.eval_replay = eval_replay
  if args.from_checkpoint:
    checkpoint.load(args.from_checkpoint)
  checkpoint.load_or_save()
  should_save(step)  # Register that we jused saved.

  print('Start training loop.')
  def policy_train(*args):
    action, state = agent.policy(*args, mode='train')
    # this is not an expert action, so you cannot self imitate, self_imitate variables
    # is only reserved for expert action in the prior preference data
    can_imitate = [False for i in range(len(train_env))] # NOTE: make sure this has the same shape as action
    return {**action, "can_self_imitate": can_imitate}, state
  def policy_eval(*args):
    action, state = agent.policy(*args, mode='eval')
    # this is not an expert action, so you cannot self imitate, self_imitate variables
    # is only reserved for expert action in the prior preference data
    can_imitate = [False for i in range(len(eval_env))] # NOTE: make sure this has the same shape as action
    return {**action, "can_self_imitate": can_imitate}, state
  while step < args.steps:
    if should_eval(step):
      print('Starting evaluation at step', int(step))
      driver_eval.reset()
      driver_eval(policy_eval, episodes=max(len(eval_env), args.eval_eps))
    driver_train(policy_train, steps=100)
    if should_save(step):
      checkpoint.save()
  logger.write()
  logger.write()

  checkpoint.save()