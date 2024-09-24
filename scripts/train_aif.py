"""
File: train_aif.py
Author: Viet Nguyen
Date: 2024-04-04

Description: This is the top-most file to conduct the training logic for Active Inference Agent
  with Contrastive Recurrent State Prior Preference model
"""

# %%

import importlib
import pathlib
import sys
import warnings
from functools import partial as bind
import os

warnings.filterwarnings('ignore', '.*box bound precision lowered.*')
warnings.filterwarnings('ignore', '.*using stateful random seeds*')
warnings.filterwarnings('ignore', '.*is a deprecated alias for.*')
warnings.filterwarnings('ignore', '.*truncated to dtype int32.*')
warnings.filterwarnings('ignore', '.*RGB-array rendering should return a numpy array.*')
warnings.filterwarnings('ignore', '.*Conversion of an array with ndim > 0 to a scalar is deprecated*')


directory = pathlib.Path(__file__).resolve()
directory = directory.parent
sys.path.append(str(directory.parent))

from raif import embodied
from raif.embodied import wrappers
from raif.embodied.run import make_env, make_envs, make_logger, make_replay

from raif.aif import agent as agt

def main(argv=None):

  parsed, other = embodied.Flags(configs=['defaults']).parse_known(argv)
  # Preping and parsing all configs and overrides
  config = embodied.Config(agt.Agent.configs['defaults'])
  for name in parsed.configs:
    config = config.update(agt.Agent.configs[name])
  config = embodied.Flags(config).parse(other)
  # Create path and necessary folders, Setup more path for the config
  # logdir initialization
  logdir = embodied.Path(config.logroot) / config.expname
  logdir.mkdirs()
  config = config.update({"logdir": str(logdir)})
  # DONE preparing config. Save config
  config.save(logdir / 'config.yaml')
  print(config, '\n')
  print('Logdir', logdir)
  args = embodied.Config(
    **config.run, logdir=config.logdir,
    batch_steps=config.batch_size * config.batch_length,
    self_imitation=config.self_imitation)

  logdir = embodied.Path(args.logdir)
  logdir.mkdirs()
  config.save(logdir / 'config.yaml')
  step = embodied.Counter()
  logger = make_logger(parsed, logdir, step, config)

  cleanup = []
  try:

    if args.script == 'train':
      replay = make_replay(config, logdir / 'replay')
      positive_replay = make_replay(config, logdir / 'positive_replay')
      env = make_envs(config)
      cleanup.append(env)
      agent = agt.Agent(env.obs_space, env.act_space, step, config)
      embodied.run.train_augmented(agent, env, (replay, positive_replay), logger, args)

    elif args.script == 'train_save':
      # TODO: WARNING: HAve not tested augmented training scripts
      replay = make_replay(config, logdir / 'replay')
      positive_replay = make_replay(config, logdir / 'positive_replay')
      env = make_envs(config)
      cleanup.append(env)
      agent = agt.Agent(env.obs_space, env.act_space, step, config)
      embodied.run.train_save_augmented(agent, env, (replay, positive_replay), logger, args)

    elif args.script == 'train_eval':
      # TODO: WARNING: HAve not tested augmented training scripts
      replay = make_replay(config, logdir / 'replay')
      eval_replay = make_replay(config, logdir / 'eval_replay', is_eval=True)
      positive_replay = make_replay(config, logdir / 'positive_replay')
      env = make_envs(config)
      eval_env = make_envs(config)  # mode='eval'
      cleanup += [env, eval_env]
      agent = agt.Agent(env.obs_space, env.act_space, step, config)
      embodied.run.train_eval_augmented(
          agent, env, eval_env, replay, positive_replay, eval_replay, logger, args)

    elif args.script == 'train_holdout':
      # TODO: WARNING: HAve not tested augmented training scripts
      replay = make_replay(config, logdir / 'replay')
      positive_replay = make_replay(config, logdir / 'positive_replay')
      if config.eval_dir:
        assert not config.train.eval_fill
        eval_replay = make_replay(config, config.eval_dir, is_eval=True)
      else:
        assert 0 < args.eval_fill <= config.replay_size // 10, args.eval_fill
        eval_replay = make_replay(config, logdir / 'eval_replay', is_eval=True)
      env = make_envs(config)
      cleanup.append(env)
      agent = agt.Agent(env.obs_space, env.act_space, step, config)
      embodied.run.train_holdout_augmented(
          agent, env, replay, positive_replay, eval_replay, logger, args)

    elif args.script == 'eval_only':
      # TODO: WARNING: HAve not tested augmented training scripts
      env = make_envs(config)  # mode='eval'
      cleanup.append(env)
      agent = agt.Agent(env.obs_space, env.act_space, step, config)
      embodied.run.eval_only_augmented(agent, env, logger, args)

    elif args.script == 'parallel':
      raise NotImplementedError("<Never Implemented Error>")
      assert config.run.actor_batch <= config.envs.amount, (
          config.run.actor_batch, config.envs.amount)
      step = embodied.Counter()
      env = make_env(config)
      agent = agt.Agent(env.obs_space, env.act_space, step, config)
      env.close()
      replay = make_replay(config, logdir / 'replay', rate_limit=True)
      embodied.run.parallel(
          agent, replay, logger, bind(make_env, config),
          num_envs=config.envs.amount, args=args)

    else:
      raise NotImplementedError(args.script)
  finally:
    for obj in cleanup:
      obj.close()


if __name__ == '__main__':
  if embodied.check_vscode_interactive():
    _args = [
      "--expname=test5",
      "--configs=gym_mtc,tiny,debug",
      "--run.steps=2000"
    ]
    main(_args)
  else:
    main(sys.argv[1:])
