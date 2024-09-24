"""
File: robosuite.py
Author: Viet Nguyen
Date: 2024-07-02

Description: This file contains pointer to the main robosuite environment
"""

import numpy as np
from raif import embodied
import functools
from typing import List, Tuple, Dict
import platform
import os
import warnings
import sys
import io
import re

class RobosuiteEnv(embodied.Env):
  def __init__(self, task: str,
      robots: List[str] = ["Panda"],
      gripper_types: str = "default",
      controller: str = "OSC_POSE",
      env_config: str = "default",
      image_size: Tuple[int, int] = (128, 128), # width, height
      camera_names: List[str] = ["agentview", "sideview", "frontview", "birdview"],
      reward_shaping: bool = False,
      generate_images=True,
      has_renderer=False,
      has_offscreen_renderer=True,
      control_freq=20,
      **kwargs):
    """

    Args:
        task (str): Can be one in these string:
          - Single Arm Env: Door, Lift, NutAssembly, PickPlace, Stack, ToolHang
          - Two Arm Env: TwoArmHandover, TwoArmLift, TwoArmPegInHole, TwoArmTransport
        robots (List[str], optional): _description_. Defaults to ["Panda"]. Robots:
          SingleArm: IIWA, Jaco, Kinova3, Panda, Sawyer, UR5e
          Bimanual: Baxter
        gripper_types (str, optional): _description_. Defaults to "default". Gripper types:
          defualt, PandaGripper, RethinkGripper, ...?
        controller (str, optional): _description_. Defaults to "OSC_POSE". controllers:
          Position-based control: input actions are, by default, interpreted as delta values from the current state.
            `OSC_POSITION`: desired position
            `JOINT_POSITION`: desired joint configuration
            End-effector pose controller: delta rotations from the current end-effector orientation in the form of axis-angle coordinates (ax, ay, az)
              `OSC_POSE`: the rotation axes are taken relative to the global world coordinate frame
                the desired value is the 6D pose (position and orientation) of a controlled frame. We follow the formalism from [Khatib87].
              `IK_POSE`: the rotation axes are taken relative to the end-effector origin, NOT the global world coordinate frame!
          `JOINT_VELOCITY`: action dimension: number of joints
          `JOINT_TORQUE`: action dimension: number of joints
          More information: https://robosuite.ai/docs/modules/controllers.html
        env_config (str, optional): _description_. Defaults to "default". Setup of the two arms, possible values:
          default, single-arm-opposed, single-arm-parallel
        img_size (Tuple[int, int], optional): _description_. Defaults to (128, 128).
        camera_names (List[str], optional): _description_. Defaults to ["agentview", "sideview", "frontview", "birdview"].
          Available cameras: 'frontview', 'birdview', 'agentview', 'sideview', 'robot0_robotview', 'robot0_eye_in_hand', 'robot1_robotview', 'robot1_eye_in_hand', ...
        reward_shaping (bool, optional): _description_. Defaults to False.
    """
    import robosuite
    import robosuite.macros as macros
    macros.IMAGE_CONVENTION = "opencv"
    from robosuite.controllers import load_controller_config
    controller_config = load_controller_config(default_controller=controller)
    # For window machines
    if platform.system() == "Windows":
      os.environ['MUJOCO_GL'] = "osmesa"
      os.environ["PYOPENGL_PLATFORM"] = "osmesa"
      os.environ["MUJOCO_EGL_DEVICE_ID"] = "1" # https://github.com/google-deepmind/dm_control/issues/415#issue-1812075254

    gripper_types = 'WipingGripper' if task == 'Wipe' else gripper_types

    # https://robosuite.ai/docs/modules/environments.html
    # Api docs: https://robosuite.ai/docs/simulation/environment.html?highlight=camera_name#robot-environment
    # create an environment for policy learning from pixels
    self._env = robosuite.make(
      task,
      robots=robots, # load a robot
      gripper_types=gripper_types, # use default grippers per robot arm
      controller_configs=controller_config,# each arm is controlled using OSC
      env_configuration=env_config, # (two-arm envs only) arms face each other:
      has_renderer=has_renderer, # no on-screen rendering
      has_offscreen_renderer=has_offscreen_renderer, # off-screen rendering needed for image obs
      control_freq=control_freq, # 20 hz control for applied actions
      horizon=2000, # each episode have maximum of 2000 steps
      use_object_obs=True, # don't provide object observations to agent
      use_camera_obs=generate_images, # provide image observations to agent
      camera_names=camera_names, # use camera for observations
      camera_heights=image_size[1], # image height
      camera_widths=image_size[0], # image width
      reward_shaping=reward_shaping, # use a dense reward signal for learning
      **kwargs
    )
    self._task = task
    self._done = True
    self._reward_shaping = reward_shaping
    raw_spaces = {}
    should_expand_keys = set()
    for k, v in self._env.observation_spec().items():
      if not hasattr(v, 'shape') or v.shape == () or v.shape == (0,):
        raw_spaces[k] = embodied.Space(np.float64, (1,))
        should_expand_keys.add(k)
      elif len(v.shape) == (3,):
        raw_spaces[k] = embodied.Space(np.uint8, v.shape)
      else:
        raw_spaces[k] = embodied.Space(np.float64, v.shape)
    if task == "Wipe":
      raw_spaces.pop('robot0_gripper_qpos')
      raw_spaces.pop('robot0_gripper_qvel')
    self.raw_spaces = raw_spaces
    self.should_expand_keys = should_expand_keys


  @functools.cached_property
  def obs_space(self):
    return {
      **self.raw_spaces,
      'reward': embodied.Space(np.float32),
      'is_first': embodied.Space(bool),
      'is_last': embodied.Space(bool),
      'is_terminal': embodied.Space(bool),
      'success': embodied.Space(bool)
    }

  @functools.cached_property
  def act_space(self):
    low, high = self._env.action_spec
    action = embodied.Space(low.dtype, low.shape, low, high)
    return {"reset": embodied.Space(bool), "action": action}

  @functools.cached_property
  def unroll(self):
    return self._env

  def step(self, action):
    action = action.copy()
    reset = action.pop('reset')
    is_first = False
    is_last = False
    if reset or self._done:
      is_first = True
      raw_obs = self._env.reset()
      self._done = False
      reward = 0
    else:
      action = action["action"]
      raw_obs, reward, is_last, _ = self._env.step(action)
      self._done = is_last
    success = self._env._check_success()
    return self.__obs(raw_obs, reward, is_first, is_last, success)

  def __obs(self, raw_obs, reward, is_first, is_last, success):
    _raw_obs = dict(raw_obs)
    for k in self.should_expand_keys:
      _raw_obs[k] = np.asarray([_raw_obs[k]])
    if self._task == "Wipe":
      _raw_obs.pop("robot0_gripper_qpos")
      _raw_obs.pop("robot0_gripper_qvel")
    return dict(
      reward=reward,
      is_first=is_first,
      is_last=is_last,
      is_terminal=is_last,
      success=success,
      **_raw_obs,
    )

