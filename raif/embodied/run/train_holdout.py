"""
File: train_holdout.py
Author: Viet Nguyen, Danijar Hafner

Description: This contain the function API handling logic for training and
  saving episode data. Adapted from Dreamerv3 codebase
"""

import re

import embodied
import numpy as np
from . import callbacks

def train_holdout(agent, env, train_replay, eval_replay, logger, args):

  logdir = embodied.Path(args.logdir)
  logdir.mkdirs()
  print('Logdir', logdir)
  should_expl = embodied.when.Until(args.expl_until)
  should_train = embodied.when.Ratio(args.train_ratio / args.batch_steps)
  if "train_every" in args and args.train_every > 0:
    should_train = embodied.when.Every(args.train_every)
  should_log = embodied.when.Clock(args.log_every)
  should_save = embodied.when.Clock(args.save_every)
  should_sync = embodied.when.Every(args.sync_every)
  step = logger.step
  updates = embodied.Counter()
  metrics = embodied.Metrics()
  print('Observation space:', embodied.format(env.obs_space), sep='\n')
  print('Action space:', embodied.format(env.act_space), sep='\n')

  timer = embodied.Timer()
  timer.wrap('agent', agent, ['policy', 'train', 'report', 'save'])
  timer.wrap('env', env, ['step'])
  if hasattr(train_replay, '_sample'):
    timer.wrap('replay', train_replay, ['_sample'])


  driver = embodied.Driver(env)
  driver.on_episode(lambda ep, worker: callbacks.print_episode(ep))
  nonzeros = set()
  driver.on_episode(lambda ep, worker: callbacks.per_episode(ep, "train", nonzeros, args, logger, metrics))
  driver.on_step(lambda tran, _: step.increment())
  driver.on_step(train_replay.add)

  print('Fill eval dataset.')
  driver_eval = embodied.Driver(env)
  driver_eval.on_step(eval_replay.add)
  random_agent = embodied.RandomAgent(env.act_space)
  while len(eval_replay) < max(args.batch_steps, args.eval_fill):
    print(len(eval_replay), max(args.batch_steps, args.eval_fill))
    driver_eval(random_agent.policy, steps=100)
  del driver_eval
  print('Prefill train dataset.')
  while len(train_replay) < max(args.batch_steps, args.train_fill):
    print(len(train_replay), max(args.batch_steps, args.train_fill))
    driver(random_agent.policy, steps=100)
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
  driver.on_step(train_step)

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
  policy = lambda *args: agent.policy(
      *args, mode='explore' if should_expl(step) else 'train')
  while step < args.steps:
    # scalars = collections.defaultdict(list)
    # for _ in range(args.eval_samples):
    #   for key, value in agent.report(next(dataset_eval)).items():
    #     if value.shape == ():
    #       scalars[key].append(value)
    # for name, values in scalars.items():
    #   logger.scalar(f'eval/{name}', np.array(values, np.float64).mean())
    # logger.write()
    driver(policy, steps=100)
    if should_save(step):
      checkpoint.save()
  logger.write()
  logger.write()
  checkpoint.save()

def train_holdout_augmented(agent, env, train_replay, positive_replay, eval_replay, logger, args):

  logdir = embodied.Path(args.logdir)
  logdir.mkdirs()
  print('Logdir', logdir)
  should_train = embodied.when.Ratio(args.train_ratio / args.batch_steps)
  if "train_every" in args and args.train_every > 0:
    should_train = embodied.when.Every(args.train_every)
  should_log = embodied.when.Clock(args.log_every)
  should_save = embodied.when.Clock(args.save_every)
  should_sync = embodied.when.Every(args.sync_every)
  step = logger.step
  updates = embodied.Counter()
  metrics = embodied.Metrics()
  print('Observation space:', embodied.format(env.obs_space), sep='\n')
  print('Action space:', embodied.format(env.act_space), sep='\n')

  timer = embodied.Timer()
  timer.wrap('agent', agent, ['policy', 'train', 'report', 'save'])
  timer.wrap('env', env, ['step'])
  if hasattr(train_replay, '_sample'):
    timer.wrap('replay', train_replay, ['_sample'])


  driver = embodied.Driver(env)
  driver.on_episode(lambda ep, worker: callbacks.print_episode(ep))
  nonzeros = set()
  driver.on_episode(lambda ep, worker: callbacks.per_episode(ep, "train", nonzeros, args, logger, metrics))
  driver.on_episode(lambda ep, worker: callbacks.lookback_on_episode(ep, args))
  driver.on_episode(lambda ep, worker: callbacks.add_trajectory(ep, train_replay, positive_replay, worker)) # add trajectory is last
  driver.on_step(lambda tran, _: step.increment())

  # Prefilling
  random_agent = embodied.RandomAgent(env.act_space)
  if len(train_replay) < args.batch_steps:
    print('Do not have enough data. Prefilling train dataset with random actions.')
    def random_policy(*args):
      action, state = random_agent.policy(*args)
      can_imitate = [False for i in range(len(env))] # NOTE: make sure this has the same shape as action
      return {**action, "can_self_imitate": can_imitate}, state
    while len(train_replay) < args.batch_steps:
      driver(random_policy, steps=100)
    train_replay.save(wait=True)
    logger.add(metrics.result())
    logger.write()

  print('Fill eval dataset.')
  driver_eval = embodied.Driver(env)
  driver_eval.on_step(eval_replay.add)
  while len(eval_replay) < max(args.batch_steps, args.eval_fill):
    print(len(eval_replay), max(args.batch_steps, args.eval_fill))
    driver_eval(random_agent.policy, steps=100)
  del driver_eval

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
  driver.on_step(train_step)

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
  def policy(*args):
    action, state = agent.policy(*args, mode='train')
    # this is not an expert action, so you cannot self imitate, self_imitate variables
    # is only reserved for expert action in the prior preference data
    can_imitate = [False for i in range(len(env))] # NOTE: make sure this has the same shape as action
    return {**action, "can_self_imitate": can_imitate}, state
  while step < args.steps:
    # scalars = collections.defaultdict(list)
    # for _ in range(args.eval_samples):
    #   for key, value in agent.report(next(dataset_eval)).items():
    #     if value.shape == ():
    #       scalars[key].append(value)
    # for name, values in scalars.items():
    #   logger.scalar(f'eval/{name}', np.array(values, np.float64).mean())
    # logger.write()
    driver(policy, steps=100)
    if should_save(step):
      checkpoint.save()
  logger.write()
  logger.write()
  checkpoint.save()