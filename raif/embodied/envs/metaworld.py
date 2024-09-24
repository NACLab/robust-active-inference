"""
File: robosuite.py
Author: Viet Nguyen
Date: 2024-03-15

Description: This file contains pointer to the metaworld environment
"""

import functools

import embodied
import gymnasium as gym
import numpy as np
from PIL import Image

from .customs.third_parties.metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE

DEFAULT_CAMERA_CONFIGS = [
  {
    "distance": 1.2,
    "azimuth": 180,
    "elevation": 0.0,
    "lookat": np.array([0.0, 0.6, 0.35]), # (x, y, z)
  },
  {
    "distance": 1.2,
    "azimuth": 270.0,
    "elevation": -85.0,
    "lookat": np.array([0.0, 0.4, 0.35]),
  },
  {
    "distance": 0.8,
    "azimuth": 35.0,
    "elevation": -25.0,
    "lookat": np.array([-0.20, 0.6, 0.25]),
  },
]


class MetaWorldEnv(embodied.Env):
  def __init__(self, env, obs_key='state', act_key='action', reward_shaping=False,
      camera_config=DEFAULT_CAMERA_CONFIGS, image_size=(64, 64), generate_images=True,
      fixed_goal=False, seed=None, **kwargs):

    if seed is not None:
      before = np.random.get_state()
      np.random.seed(seed)

    if isinstance(env, str):
      gstr = "goal-observable"
      self._env = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[f"{env}-{gstr}"](render_mode="rgb_array", width=image_size[0], height=image_size[1], default_camera_configs=DEFAULT_CAMERA_CONFIGS, **kwargs)  # Create an environment with task `env`
    else:
      raise ValueError(env, "must be a string")
    self._image_obs_length = len(camera_config)
    self._generate_images = generate_images
    self._image_size = image_size # in (width, height)
    self._obs_dict = hasattr(self._env.observation_space, 'spaces')
    self._act_dict = hasattr(self._env.action_space, 'spaces')
    self._done = True
    self._info = None
    self._obs_key = obs_key
    self._act_key = act_key
    self._reward_shaping = reward_shaping

    if not fixed_goal:
      self._env._freeze_rand_vec = False

    if seed is not None:
      np.random.set_state(before)

  @property
  def info(self):
    return self._info

  def __getattr__(self, name):
    if name.startswith('__'):
      raise AttributeError(name)
    try:
      return getattr(self._env, name)
    except AttributeError:
      raise ValueError(name)

  @functools.cached_property
  def obs_space(self):
    """According to the paper, the original observation space (state) is 39-D
    example of the observation space
    Box(
      np.hstack(
        (
          _HAND_SPACE, (3D-tuple representing 3D position of the end-effector: x, y, z (in order - I guess so))
          gripper, (1-D scalar measure how open or close the gripper is)
          obj, (14D-tuple or 7D-tuple: two or one object, each object 7D)
            each object: (3D position x, y, z, and 4D quarternion representation)
            if there are no second object => the quantities are zero-ed out.
          _HAND_SPACE, (the same thing repeat as above)
          gripper, (the same thing repeat as above)
          obj, (the same thing repeat as above)
          goal, (the 3D position of the goal in x, y, z)
        )
      )
    )
    """
    if self._obs_dict:
      spaces = self._flatten(self._env.observation_space.spaces)
    else:
      spaces = {self._obs_key: self._env.observation_space}
    spaces = {k: self._convert(v) for k, v in spaces.items()}
    if self._generate_images:
      image_spaces = {f"image_{k}": embodied.Space(np.uint8, (*self._image_size[::-1], 3), 0, 255) for k in range(self._image_obs_length)}
    else:
      image_spaces = {}
    return {
        **spaces,
        **image_spaces,
        'reward': embodied.Space(np.float32),
        'is_first': embodied.Space(bool),
        'is_last': embodied.Space(bool),
        'is_terminal': embodied.Space(bool),
        # info dict as observation
        'success': embodied.Space(np.float32),
        'near_object': embodied.Space(np.float32),
        'grasp_success': embodied.Space(np.float32),
        'grasp_reward': embodied.Space(np.float32),
        'in_place_reward': embodied.Space(np.float32),
        'obj_to_target': embodied.Space(np.float32),
        'unscaled_reward': embodied.Space(np.float32),
    }

  @functools.cached_property
  def act_space(self):
    if self._act_dict:
      spaces = self._flatten(self._env.action_space.spaces)
    else:
      spaces = {self._act_key: self._env.action_space}
    spaces = {k: self._convert(v) for k, v in spaces.items()}
    spaces['reset'] = embodied.Space(bool)
    return spaces

  def step(self, action):
    if action['reset'] or self._done:
      self._done = False
      obs, _ = self._env.reset()
      obs, reward, done, truncated, self._info = self._env.step(np.zeros(self._env.action_space.shape))
      if not self._reward_shaping:
        reward = self._info["success"] - 1.0 # (when success, reward = 0, else reward = -1)
      self._done = done or truncated
      return self._obs(
        obs,
        reward,
        is_first=True,
        is_last=bool(self._done),
        is_terminal=bool(self._info.get('is_terminal', self._done)),
        info=self._info
      )
    if self._act_dict:
      action = self._unflatten(action)
    else:
      action = action[self._act_key]
    obs, reward, done, truncated, self._info = self._env.step(action)
    if not self._reward_shaping:
      # by default, the metaworld reward is dense, so we manually disable that
      reward = self._info["success"] - 1.0 # (when success, reward = 0, else reward = -1)
    self._done = done or truncated
    return self._obs(
      obs, reward,
      is_first=False,
      is_last=bool(self._done),
      is_terminal=bool(self._info.get('is_terminal', self._done)),
      info=self._info
    )

  def _obs(
      self, obs, reward, is_first=False, is_last=False, is_terminal=False, info={}):
    if not self._obs_dict:
      obs = {self._obs_key: obs}
    obs = self._flatten(obs)
    obs = {k: np.asarray(v) for k, v in obs.items()}
    if self._generate_images:
      images = self._env.render()
      # Resize the image
      images_dict = {f"image_{i}": images[i] for i in range(self._image_obs_length)}
    else:
      images_dict = {}
    obs.update(
        **images_dict,
        reward=np.float32(reward),
        is_first=is_first,
        is_last=is_last,
        is_terminal=is_terminal,
        **info)
    return obs

  def render(self):
    images = self._env.render()
    assert images is not None
    return images[self._image_obs_length - 1]

  def close(self):
    try:
      self._env.close()
    except Exception:
      pass

  def _flatten(self, nest, prefix=None):
    result = {}
    for key, value in nest.items():
      key = prefix + '/' + key if prefix else key
      if isinstance(value, gym.spaces.Dict):
        value = value.spaces
      if isinstance(value, dict):
        result.update(self._flatten(value, key))
      else:
        result[key] = value
    return result

  def _unflatten(self, flat):
    result = {}
    for key, value in flat.items():
      parts = key.split('/')
      node = result
      for part in parts[:-1]:
        if part not in node:
          node[part] = {}
        node = node[part]
      node[parts[-1]] = value
    return result

  def _convert(self, space):
    if hasattr(space, 'n'):
      return embodied.Space(np.int32, (), 0, space.n)
    return embodied.Space(space.dtype, space.shape, space.low, space.high)

