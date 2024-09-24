import argparse

import numpy as np
import re
import importlib
import pathlib
import sys
sys.path.append(str(pathlib.Path(__file__).parent.parent))
import warnings
from functools import partial as bind
import os
from ruamel import yaml



import robosuite as suite
from robosuite import load_controller_config
from robosuite.utils.input_utils import input2action
from robosuite.wrappers import VisualizationWrapper
import time

from raif.embodied.run import callbacks
from raif import embodied
from raif import embodied
from raif.embodied import wrappers
from raif.embodied.run import make_env, make_envs, make_logger, make_replay

warnings.filterwarnings('ignore', '.*box bound precision lowered.*')
warnings.filterwarnings('ignore', '.*using stateful random seeds*')
warnings.filterwarnings('ignore', '.*is a deprecated alias for.*')
warnings.filterwarnings('ignore', '.*truncated to dtype int32.*')
warnings.filterwarnings('ignore', '.*RGB-array rendering should return a numpy array.*')
warnings.filterwarnings('ignore', '.*Conversion of an array with ndim > 0 to a scalar is deprecated*')


if __name__ == "__main__":

  # parser = argparse.ArgumentParser()
  # parser.add_argument("--task", type=str, default="robosuite_Lift")
  # parser.add_argument("--arm", type=str, default="right", help="Which arm to control (eg bimanual) 'right' or 'left'")
  # parser.add_argument("--switch-on-grasp", action="store_true", help="Switch gripper control on gripper action")
  # parser.add_argument("--toggle-camera-on-grasp", action="store_true", help="Switch camera angle on gripper action")
  # parser.add_argument(
  #     "--envconfig", type=str, default="single-arm-opposed", help="Specified environment configuration if necessary"
  # )
  # parser.add_argument("--device", type=str, default="keyboard")
  # parser.add_argument("--pos-sensitivity", type=float, default=1.0, help="How much to scale position user inputs")
  # parser.add_argument("--rot-sensitivity", type=float, default=1.0, help="How much to scale rotation user inputs")
  # args = parser.parse_args()

  ### Setup config
  configs = yaml.YAML(typ='safe').load(
    (embodied.Path(__file__).parent / 'robosuite_keyboard_configs.yaml').read())
  parsed, other = embodied.Flags(configs=['defaults']).parse_known(sys.argv[1:])
  # Preping and parsing all configs and overrides
  config = embodied.Config(configs['defaults'])
  for name in parsed.configs:
    config = config.update(configs[name])
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

  args = config.robosuite_config


  logdir = embodied.Path(config.logdir)
  logdir.mkdirs()
  config.save(logdir / 'config.yaml')
  step = embodied.Counter()
  logger = make_logger(parsed, logdir, step, config)


  ### Setup environment
  # Create environment
  ourenv = make_envs(config)
  env = ourenv.unroll[0]
  # Wrap this environment in a visualization wrapper
  env = VisualizationWrapper(env, indicator_configs=None)
  # Setup printing options for numbers
  np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})
  # initialize device
  if args.device == "keyboard":
    from robosuite.devices import Keyboard
    device = Keyboard(pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity)
    env.viewer.add_keypress_callback(device.on_press)
  elif args.device == "spacemouse":
    from robosuite.devices import SpaceMouse
    device = SpaceMouse(pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity)
  else:
    raise Exception("Invalid device choice: choose either 'keyboard' or 'spacemouse'.")

  # Setup env-related things
  replay = make_replay(config, logdir / 'replay')
  positive_replay = make_replay(config, logdir / 'positive_replay')
  should_log = embodied.when.Clock(config.run.log_every)
  step = logger.step
  metrics = embodied.Metrics()
  print('Observation space:', embodied.format(ourenv.obs_space), sep='\n')
  print('Action space:', embodied.format(ourenv.act_space), sep='\n')


  ######## Function setup ###################### Driver setup
  # Driver setup
  driver = embodied.Driver(ourenv)


  last_grasp = [0]
  cam_id = [0]
  num_cam = len(env.sim.model.camera_names)
  def setup_new_episode(ep, worker):
    env.reset()
    # Setup rendering
    cam_id[0] = 0
    env.render()
    # Initialize device control
    device.start_control()
    # Initialize variables that should the maintained between resets
    last_grasp[0] = 0
    return ep
  setup_new_episode(None, None)
  driver.on_episode(setup_new_episode)

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
    for key in config.run.log_keys_video:
      if key in ep:
        stats[f'policy_{key}'] = ep[key]
    for key, value in ep.items():
      if not config.run.log_zeros and key not in nonzeros and (value == 0).all():
        continue
      nonzeros.add(key)
      if re.match(config.run.log_keys_sum, key):
        stats[f'sum_{key}'] = ep[key].sum()
      if re.match(config.run.log_keys_mean, key):
        stats[f'mean_{key}'] = ep[key].mean()
      if re.match(config.run.log_keys_max, key):
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
  driver.on_episode(lambda ep, worker: callbacks.lookback_on_episode(ep, config))

  # add to trajectory every episode
  driver.on_episode(lambda ep, worker: callbacks.add_trajectory(ep, replay, positive_replay, worker)) # add trajectory is last

  # increment
  driver.on_step(lambda tran, _: step.increment())

  # because action listener is on a different thread, we can do a time sleep in the main thread to
  # force the environment to be slower
  driver.on_step(lambda tran, _: time.sleep(0.2))

  # policy
  def policy(obs, state):
    # Set active robot
    active_robot = env.robots[0] if args.envconfig == "bimanual" else env.robots[args.arm == "left"]

    # Get the newest action
    action, grasp = input2action(
      device=device, robot=active_robot, active_arm=args.arm, env_configuration=args.envconfig
    )

    ####################
    # If action is none, then this a reset so we should break, episode end here
    if action is None:
      # TODO: To be implemented
      return {"action": ourenv.act_space["action"].sample()[None], "reset": np.asarray([True])}, None

    ######################
    # If the current grasp is active (1) and last grasp is not (-1) (i.e.: grasping input just pressed),
    # toggle arm control and / or camera viewing angle if requested
    if last_grasp[0] < 0 < grasp:
      if args.switch_on_grasp:
        args.arm = "left" if args.arm == "right" else "right"
      if args.toggle_camera_on_grasp:
        cam_id[0] = (cam_id[0] + 1) % num_cam
        env.viewer.set_camera(camera_id=cam_id[0])
    # Update last grasp
    last_grasp[0] = grasp

    # Fill out the rest of the action space if necessary
    rem_action_dim = env.action_dim - action.size
    if rem_action_dim > 0:
      # Initialize remaining action space
      rem_action = np.zeros(rem_action_dim)
      # This is a multi-arm setting, choose which arm to control and fill the rest with zeros
      if args.arm == "right":
        action = np.concatenate([action, rem_action])
      elif args.arm == "left":
        action = np.concatenate([rem_action, action])
      else:
        # Only right and left arms supported
        print(
          "Error: Unsupported arm specified -- "
          "must be either 'right' or 'left'! Got: {}".format(args.arm)
        )
    elif rem_action_dim < 0:
      # We're in an environment with no gripper action space, so trim the action space to be the action dim
      action = action[: env.action_dim]
    env.render()
    if step.value % 100 == 0:
      print(f"[Step {step.value}]")
    return {"action": action[None], "reset": np.asarray([False]), "can_self_imitate": np.asarray([True])}, None


  ################## Simulation
  print('Start collecting prior data.')
  while step < config.run.steps:
    driver(policy, steps=100)
  logger.write()

  # Finally, save the replay
  replay.save(wait=True)
  positive_replay.save(wait=True)

