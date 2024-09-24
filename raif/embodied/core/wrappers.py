"""
File: wrapper.py
Author: Viet Nguyen, Danijar Hafner
Date: 2023-01-02

Description: This file contain environment wrappers
"""

import functools
import time
import numpy as np
import re
import gymnasium as gym
from typing import Dict, List, Tuple, Callable, Any
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections import deque

from . import base
from . import space as spacelib
from . import convert
from ..utils import tensorstats

ValueType = float | int | bool | np.ndarray
ObservationAndPreviousActionType = Dict[str, ValueType]

class TimeLimit(base.Wrapper):
  def __init__(self, env, duration, reset=True):
    super().__init__(env)
    self._duration = duration
    self._reset = reset
    self._step = 0
    self._done = False

  def step(self, action):
    if action['reset'] or self._done:
      self._step = 0
      self._done = False
      if self._reset:
        action.update(reset=True)
        return self.env.step(action)
      else:
        action.update(reset=False)
        obs = self.env.step(action)
        obs['is_first'] = True
        return obs
    self._step += 1
    obs = self.env.step(action)
    if self._duration and self._step >= self._duration:
      obs['is_last'] = True
    self._done = obs['is_last']
    return obs

class ActionRepeat(base.Wrapper):
  def __init__(self, env, repeat):
    super().__init__(env)
    self._repeat = repeat

  def step(self, action):
    if action['reset']:
      return self.env.step(action)
    reward = 0.0
    for _ in range(self._repeat):
      obs = self.env.step(action)
      reward += obs['reward']
      if obs['is_last'] or obs['is_terminal']:
        break
    obs['reward'] = np.float32(reward)
    return obs

class ClipAction(base.Wrapper):
  def __init__(self, env, key='action', low=-1, high=1):
    super().__init__(env)
    self._key = key
    self._low = low
    self._high = high

  def step(self, action):
    clipped = np.clip(action[self._key], self._low, self._high)
    return self.env.step({**action, self._key: clipped})

class NormalizeAction(base.Wrapper):
  def __init__(self, env, key='action'):
    super().__init__(env)
    self._key = key
    self._space = env.act_space[key]
    self._mask = np.isfinite(self._space.low) & np.isfinite(self._space.high)
    self._low = np.where(self._mask, self._space.low, -1)
    self._high = np.where(self._mask, self._space.high, 1)

  @functools.cached_property
  def act_space(self):
    low = np.where(self._mask, -np.ones_like(self._low), self._low)
    high = np.where(self._mask, np.ones_like(self._low), self._high)
    space = spacelib.Space(np.float32, self._space.shape, low, high)
    return {**self.env.act_space, self._key: space}

  def step(self, action):
    orig = (action[self._key] + 1) / 2 * (self._high - self._low) + self._low
    orig = np.where(self._mask, orig, action[self._key])
    return self.env.step({**action, self._key: orig})

class OneHotAction(base.Wrapper):
  def __init__(self, env, key='action'):
    super().__init__(env)
    self._count = int(env.act_space[key].high)
    self._key = key

  @functools.cached_property
  def act_space(self):
    shape = (self._count,)
    space = spacelib.Space(np.float32, shape, 0, 1)
    space.sample = functools.partial(self._sample_action, self._count)
    space._discrete = True
    return {**self.env.act_space, self._key: space}

  def step(self, action):
    if not action['reset']:
      assert action[self._key].min() == 0.0, action
      assert action[self._key].max() == 1.0, action
      assert action[self._key].sum() == 1.0, action
    index = np.argmax(action[self._key])
    return self.env.step({**action, self._key: index})

  @staticmethod
  def _sample_action(count):
    index = np.random.randint(0, count)
    action = np.zeros(count, dtype=np.float32)
    action[index] = 1.0
    return action

class ExpandScalars(base.Wrapper):
  """Every field in the obs space which is not a reward but a scalar, make it 1D array
    This is very convenient in concatenation of state and related housekeeping
  """
  def __init__(self, env):
    super().__init__(env)
    self._obs_expanded = []
    self._obs_space = {}
    for key, space in self.env.obs_space.items():
      if space.shape == () and key != 'reward' and not space.discrete:
        space = spacelib.Space(space.dtype, (1,), space.low, space.high)
        self._obs_expanded.append(key)
      self._obs_space[key] = space
    self._act_expanded = []
    self._act_space = {}
    for key, space in self.env.act_space.items():
      if space.shape == () and not space.discrete:
        space = spacelib.Space(space.dtype, (1,), space.low, space.high)
        self._act_expanded.append(key)
      self._act_space[key] = space

  @functools.cached_property
  def obs_space(self):
    return self._obs_space

  @functools.cached_property
  def act_space(self):
    return self._act_space

  def step(self, action):
    action = {
        key: np.squeeze(value, 0) if key in self._act_expanded else value
        for key, value in action.items()}
    obs = self.env.step(action)
    obs = {
        key: np.expand_dims(value, 0) if key in self._obs_expanded else value
        for key, value in obs.items()}
    return obs

class ExpandBatchDimSoft(base.Wrapper):
  def __init__(self, env):
    super().__init__(env)
    self._env = env

  @functools.cached_property
  def obs_space(self):
    return self._obs_space

  @functools.cached_property
  def act_space(self):
    return self._act_space

  def step(self, action):
    action = {key: np.squeeze(value, 0) for key, value in action.items()}
    obs = self.env.step(action)
    obs = {key: np.expand_dims(value, 0) for key, value in obs.items()}
    return obs

class FlattenTwoDimObs(base.Wrapper):
  def __init__(self, env):
    super().__init__(env)
    self._keys = []
    self._obs_space = {}
    for key, space in self.env.obs_space.items():
      if len(space.shape) == 2:
        space = spacelib.Space(
            space.dtype,
            (int(np.prod(space.shape)),),
            space.low.flatten(),
            space.high.flatten())
        self._keys.append(key)
      self._obs_space[key] = space

  @functools.cached_property
  def obs_space(self):
    return self._obs_space

  def step(self, action):
    obs = self.env.step(action).copy()
    for key in self._keys:
      obs[key] = obs[key].flatten()
    return obs

class FlattenTwoDimActions(base.Wrapper):
  def __init__(self, env):
    super().__init__(env)
    self._origs = {}
    self._act_space = {}
    for key, space in self.env.act_space.items():
      if len(space.shape) == 2:
        space = spacelib.Space(
            space.dtype,
            (int(np.prod(space.shape)),),
            space.low.flatten(),
            space.high.flatten())
        self._origs[key] = space.shape
      self._act_space[key] = space

  @functools.cached_property
  def act_space(self):
    return self._act_space

  def step(self, action):
    action = action.copy()
    for key, shape in self._origs.items():
      action[key] = action[key].reshape(shape)
    return self.env.step(action)

class TransitionDelta(base.Wrapper):
  def __init__(self, env, keys=r'.*'):
    super().__init__(env)
    excluded = ('is_first', 'is_last')
    extended_keys = {k for k in self.env.obs_space.keys() if
      (k not in excluded and not k.startswith('log_') and re.match(keys, k))}
    print(f"[TransitionDeltaWrapper] extended_keys: {extended_keys}")
    self.last_recorded_obs = {k: None for k in extended_keys}

  @functools.cached_property
  def obs_space(self):
    spaces = {}
    for k in self.last_recorded_obs.keys():
      delta_key = f"{k}_delta"
      assert delta_key not in self.env.obs_space, delta_key
      space = self.env.obs_space[k]
      delta = np.absolute(space.high-space.low)
      delta_space = spacelib.Space(space.dtype, space.shape, space.low-delta, space.high+delta)
      spaces.update({delta_key: delta_space})
    return {**self.env.obs_space, **spaces}

  def step(self, action):
    obs = self.env.step(action)
    if obs["is_first"]:
      # NOTE: TODO: Is it reasonable here?
      obs.update({f"{k}_delta": np.zeros_like(obs[k]) for k, v in self.last_recorded_obs.items()})
      self.last_recorded_obs = {k: np.zeros_like(obs[k]) for k, v in self.last_recorded_obs.items()}
    else:
      obs.update({f"{k}_delta": obs[k] - v for k, v in self.last_recorded_obs.items()})
      self.last_recorded_obs = {k: obs[k] for k in self.last_recorded_obs.keys()}
    return obs

class ActionSpaceField(base.Wrapper):
  def __init__(self, env, key="expert_action"):
    super().__init__(env)
    self.key = key

  @functools.cached_property
  def act_space(self):
    addition_action_space = self.env.act_space["action"].copy()
    return {**self.env.act_space, self.key: addition_action_space}

  def step(self, action):
    return self.env.step(action)

class ObservationSpaceField(base.Wrapper):
  def __init__(self, env, field_setter: Callable, key="in_expert_traj"):
    super().__init__(env)
    self.key = key
    self.field_setter = field_setter # (obs) -> np.ndarray # obs: Dict[str, np.ndarray]

  @functools.cached_property
  def obs_space(self):
    spaces = self.env.obs_space
    if self.key not in spaces:
      sample_obs = {k: v.sample() for k, v in self.env.obs_space.items()}
      sample_value = self.field_setter(sample_obs)
      sample_space = spacelib.Space(sample_value.dtype, sample_value.shape)
      spaces[self.key] = sample_space
      return spaces
    else:
      raise KeyError(self.key)

  def step(self, action):
    obs = self.env.step(action)
    return {**obs, self.key: self.field_setter(obs)}


class CheckSpaces(base.Wrapper):
  def __init__(self, env):
    super().__init__(env)

  def step(self, action):
    for key, value in action.items():
      self._check(value, self.env.act_space[key], key)
    obs = self.env.step(action)
    for key, value in obs.items():
      self._check(value, self.env.obs_space[key], key)
    return obs

  def _check(self, value, space, key):
    if not isinstance(value, (
        np.ndarray, np.generic, list, tuple, int, float, bool)):
      raise TypeError(f'Invalid type {type(value)} for key {key}.')
    if value in space:
      return
    dtype = np.array(value).dtype
    shape = np.array(value).shape
    lowest, highest = np.min(value), np.max(value)
    raise ValueError(
        f"Value for '{key}' with dtype {dtype}, shape {shape}, "
        f"lowest {lowest}, highest {highest} is not in {space}.")

class StateConcatenation(base.Wrapper):
  def __init__(self, env, keys=r"^$", target_key="state"):
    """Concatenate every state specified in a variable called state
    all the state specified has to have the same dimension `len(s.shape)`
    For example,
      shape1: (n0, n1, n2)
      shape2: (n3, n4, n5)
    Requirement:
      - n0 == n3
      - n1 == n4
      - len(shape1) == len(shape2)
    Note: The only thing allowed to be different is the last dimension

    Args:
      keys (str): regular expression of which key to be concatenated
      target_key (str): the target key to be concatenated to
    """
    super().__init__(env)
    self._target_key = target_key
    self.concat_keys = [k for k in self.env.obs_space.keys() if
      (not k.startswith('log_') and re.match(keys, k))]
    self.concat_keys.sort() # sort with order so the order is fixed. NOTE: Do this to avoid observation space concated togetehr mixedly (without order) in different runs.
    self.check_concat_keys()
    if self.state_concat_needed:
      print(f"[StateConcatenationWrapper] concat keys list: {self.concat_keys}")

  @functools.cached_property
  def state_concat_needed(self):
    return len(self.concat_keys) > 0 # and "state" not in self.env.obs_space # NOTE: we can allow the state in here

  def check_concat_keys(self):
    if self.state_concat_needed:
      assert self._target_key not in self.env.obs_space.keys(), f"Target key {self._target_key} exists in your observation space, please choose another target key"
      shapes = [self.env.obs_space[k].shape for k in self.concat_keys]
      dtypes = [self.env.obs_space[k].dtype for k in self.concat_keys]
      lenshape = len(shapes[0])
      for shapeid in range(lenshape - 1): # For shape (n0, n1, ..., n_n-1). Loop from 0 to n-2
        for i in range(len(shapes) - 1): # n0 == n1, n1 == n2, n2 == n3, ...
          assert shapes[i][shapeid] == shapes[i+1][shapeid], f"Wrong shape: {shapes[i][shapeid]} != {shapes[i+1][shapeid]}"
        assert dtypes[i] == dtypes[i+1], f"Wrong dtype: {dtypes[i]} != {dtypes[i+1]}"

  @functools.cached_property
  def obs_space(self):
    spaces = self.env.obs_space
    if self.state_concat_needed:
      shapes = [self.env.obs_space[k].shape for k in self.concat_keys]
      lastdim = sum(shape[-1] for shape in shapes)
      lows = np.concatenate([self.env.obs_space[k].low for k in self.concat_keys], axis=-1)
      highs = np.concatenate([self.env.obs_space[k].high for k in self.concat_keys], axis=-1)
      state_space = spacelib.Space(
        self.env.obs_space[list(self.concat_keys)[0]].dtype,
        (*shapes[0][:-1], lastdim),
        lows,
        highs
      )
      spaces[self._target_key] = state_space
    return spaces

  def step(self, action):
    obs = self.env.step(action)
    if self.state_concat_needed:
      obs[self._target_key] = np.concatenate([obs[k] for k in self.concat_keys], axis=-1)
    return obs

class DiscretizeAction(base.Wrapper):
  def __init__(self, env, key='action', bins=5):
    super().__init__(env)
    self._dims = np.squeeze(env.act_space[key].shape, 0).item()
    self._values = np.linspace(-1, 1, bins)
    self._key = key

  @functools.cached_property
  def act_space(self):
    shape = (self._dims, len(self._values))
    space = spacelib.Space(np.float32, shape, 0, 1)
    space.sample = functools.partial(
        self._sample_action, self._dims, self._values)
    space._discrete = True
    return {**self.env.act_space, self._key: space}

  def step(self, action):
    if not action['reset']:
      assert (action[self._key].min(-1) == 0.0).all(), action
      assert (action[self._key].max(-1) == 1.0).all(), action
      assert (action[self._key].sum(-1) == 1.0).all(), action
    indices = np.argmax(action[self._key], axis=-1)
    continuous = np.take(self._values, indices)
    return self.env.step({**action, self._key: continuous})

  @staticmethod
  def _sample_action(dims, values):
    indices = np.random.randint(0, len(values), dims)
    action = np.zeros((dims, len(values)), dtype=np.float32)
    action[np.arange(dims), indices] = 1.0
    return action

class ResizeImage(base.Wrapper):
  def __init__(self, env, size=(64, 64)):
    super().__init__(env)
    self._size = size
    self._keys = [
        k for k, v in env.obs_space.items()
        if len(v.shape) > 1 and v.shape[:2] != size]
    print(f'Resizing keys {",".join(self._keys)} to {self._size}.')
    if self._keys:
      from PIL import Image
      self._Image = Image

  @functools.cached_property
  def obs_space(self):
    spaces = self.env.obs_space
    for key in self._keys:
      shape = self._size + spaces[key].shape[2:]
      spaces[key] = spacelib.Space(np.uint8, shape)
    return spaces

  def step(self, action):
    obs = self.env.step(action)
    for key in self._keys:
      obs[key] = self._resize(obs[key])
    return obs

  def _resize(self, image):
    image = self._Image.fromarray(image)
    image = image.resize(self._size, self._Image.NEAREST)
    image = np.array(image)
    return image

class RenderImage(base.Wrapper):
  def __init__(self, env, key='image'):
    super().__init__(env)
    self._key = key
    try:
      self._shape = self.env.render().shape
    except:
      """ OpenAIGym: ResetNeeded: Cannot call `env.render()` before calling `env.reset()`, if this is a intended action, set 
        `disable_render_order_enforcing=True` on the OrderEnforcer wrapper.
      """
      self.env.step({"action": self.env.act_space["action"].sample(), "reset": True})
      self._shape = self.env.render().shape
    assert len(self._shape) >= 2, self._shape

  @functools.cached_property
  def obs_space(self):
    spaces = self.env.obs_space
    spaces[self._key] = spacelib.Space(np.uint8, self._shape)
    return spaces

  def step(self, action):
    obs = self.env.step(action)
    obs[self._key] = self.env.render()
    return obs

class RestartOnException(base.Wrapper):

  def __init__(
      self, ctor, exceptions=(Exception,), window=300, maxfails=2, wait=20):
    if not isinstance(exceptions, (tuple, list)):
        exceptions = [exceptions]
    self._ctor = ctor
    self._exceptions = tuple(exceptions)
    self._window = window
    self._maxfails = maxfails
    self._wait = wait
    self._last = time.time()
    self._fails = 0
    super().__init__(self._ctor())

  def step(self, action):
    try:
      return self.env.step(action)
    except self._exceptions as e:
      if time.time() > self._last + self._window:
        self._last = time.time()
        self._fails = 1
      else:
        self._fails += 1
      if self._fails > self._maxfails:
        raise RuntimeError('The env crashed too many times.')
      message = f'Restarting env after crash with {type(e).__name__}: {e}'
      print(message, flush=True)
      time.sleep(self._wait)
      self.env = self._ctor()
      action['reset'] = np.ones_like(action['reset'])
      return self.env.step(action)

class GymWrapperFinalLayer(gym.Env):
  def __init__(self, env: base.Env, obs_key: str | List[str]="state"):
    """This class gives the custom env a final layer of wrapping that makes it function like a gym environment
    this only supports single space box observation and single space box action
    and dict space box observation

    Args:
        env (base.Env): _description_
        obs_key (str, optional): _description_. Defaults to "state".
    """
    # super().__init__(env)
    self.env = env
    self._obs_dict = False
    Sequence = list | tuple
    if isinstance(obs_key, Sequence):
      self._obs_dict = True
      for k in obs_key:
        assert k in env.obs_space, k
    else:
      # print(f"[GYMWRAP] obs_key: {obs_key}, type: {type(obs_key)}")
      # print(f"[GYMWRAP] env.obs_space: {env.obs_space}")
      assert obs_key in env.obs_space, obs_key
    self.obs_key = obs_key

  def __getattr__(self, name):
    if name.startswith('__'):
      raise AttributeError(name)
    try:
      return getattr(self.env, name)
    except AttributeError:
      raise ValueError(name)

  @functools.cached_property
  def observation_space(self):
    if self._obs_dict:
      return gym.spaces.Dict({
        k: gym.spaces.Box(
          self.env.obs_space[k].low,
          self.env.obs_space[k].high,
          self.env.obs_space[k].shape,
          self.env.obs_space[k].dtype
        ) for k in self.obs_key
      })
    else:
      return gym.spaces.Box(
        self.env.obs_space[self.obs_key].low,
        self.env.obs_space[self.obs_key].high,
        self.env.obs_space[self.obs_key].shape,
        self.env.obs_space[self.obs_key].dtype
      )

  @functools.cached_property
  def action_space(self):
    return gym.spaces.Box(self.env.act_space["action"].low, self.env.act_space["action"].high)

  def reset(self, seed=0):
    obs = self.env.step({"action": self.env.act_space["action"].sample(), "reset": True})
    if self._obs_dict:
      info = {"other_obs": {k: v for k, v in obs.items() if k not in self.obs_key}}
      _obs = OrderedDict([(k, obs[k]) for k in self.obs_key])
    else:
      info = {"other_obs": {k: v for k, v in obs.items() if k not in [self.obs_key]}}
      _obs = obs[self.obs_key]
    return _obs, info

  def step(self, action):
    act_dict = {"action": action, "reset": False}
    obs = self.env.step(act_dict)
    if self._obs_dict:
      info = {"other_obs": {k: v for k, v in obs.items() if k not in self.obs_key}}
      _obs = OrderedDict([(k, obs[k]) for k in self.obs_key])
    else:
      info = {"other_obs": {k: v for k, v in obs.items() if k not in [self.obs_key]}}
      _obs = obs[self.obs_key]
    rew = obs["reward"]
    terminal = obs["is_last"]
    truncated = obs["is_last"]
    return _obs, rew, terminal, truncated, info

  def render(self):
    return self.env.render()


##################### Step Lambda and Env measurement #################
class EnvStepLambda(ABC):
  @functools.cached_property
  def space(self) -> Dict[str, spacelib.Space]:
    raise NotImplementedError

  @abstractmethod
  def __call__(self, transition: ObservationAndPreviousActionType) -> ValueType:
    raise NotImplementedError

class EnvStepLambdaWrapper(base.Wrapper):
  def __init__(self, env: base.Env, env_step_lambda: EnvStepLambda):
    super().__init__(env)
    assert isinstance(env_step_lambda, EnvStepLambda), env_step_lambda
    assert isinstance(env_step_lambda.space, dict), env_step_lambda
    for k, v in env_step_lambda.space.items():
      assert isinstance(v, spacelib.Space), env_step_lambda.space[k]
    self._env_step_lambda = env_step_lambda
    self._env = env

  @functools.cached_property
  def obs_space(self):
    spaces = self._env.obs_space
    additional_spaces = self._env_step_lambda.space
    for k, v in additional_spaces.items():
      assert k not in spaces, k
    return {**spaces, **additional_spaces}

  @functools.cached_property
  def act_space(self):
    return {**self._env.act_space}

  def step(self, action):
    obs = self._env.step(action)
    additional_obs = self._env_step_lambda({**obs, **action})
    return {**obs, **additional_obs}

class MeasureSuccessRate(EnvStepLambda):
  def __init__(self, task: str, **kwargs) -> None:
    suite, envname = task.split("_", 1)
    self._setup(suite, envname, **kwargs)
    self.success_fn = self._get_success_fn(suite, envname)
    self.success_histories = [deque([], maxlen=kwargs.get("window", 100))] # storring a windowed amount of episode recording whether each episode has succeeded
    self.current_success_rate = [0.0]
    self.has_succeeded_in_this_episode = False
    self._kwargs = kwargs

  def _setup(self, suite, envname, **kwargs):
    # This method setup necessary variables for running success function
    if suite == "gymrobotics":
      self.continuous_goal = 0
      self.success_step_to_goal = kwargs.get("success_step_to_goal", 1) # default continuous 10 steps to be successful
      self.distance_threshold = kwargs.get("distance_threshold", 0.05) # default distance threshold of 0.05

  def _get_success_fn(self, suite, envname) -> Callable[[ObservationAndPreviousActionType], bool]:
    default_success_fn = lambda o: False
    if suite == "gym":
      if envname == "mtc":
        success_fn = lambda o: o["state"][0] >= 0.45
      else:
        success_fn = default_success_fn
    elif suite == "gymrobotics":
      def goal_distance(goal_a, goal_b):
        assert goal_a.shape == goal_b.shape
        return np.linalg.norm(goal_a - goal_b, axis=-1)
      def compute_sparse_reward(transition: ObservationAndPreviousActionType) -> float:
        d = goal_distance(transition["achieved_goal"], transition["desired_goal"])
        return -(d > self.distance_threshold).astype(np.float32)
      def success_fn(transition: ObservationAndPreviousActionType) -> bool:
        if compute_sparse_reward(transition) == 0:
          self.continuous_goal += 1
        else:
          self.continuous_goal = 0
        return self.continuous_goal >= self.success_step_to_goal
    elif suite == "metaworld":
      success_fn = lambda o: o["success"][0] != 0.0
    elif suite == "robosuite":
      success_fn = lambda o: o["success"]
    else:
      success_fn = default_success_fn
      print(f"[MeasureSuccessRate] You have not implemented the success function for this enivonment, falling back to default. Is this a mistake?")
    return success_fn

  @functools.cached_property
  def space(self):
    return {
      "success_rate": spacelib.Space(np.float32, (), 0.0, 1.0),
      "is_success": spacelib.Space(bool, ()),
      "has_succeeded": spacelib.Space(bool, ()),
    }

  def __call__(self, transition: ObservationAndPreviousActionType) -> Dict[str, ValueType]:
    if "new_lap" in transition and transition["new_lap"]: # addtional meta action
      self.success_histories.append(deque([], maxlen=self._kwargs.get("window", 100)))
      self.current_success_rate.append(0.0)

    # if this is the next episode, increase the total episode count
    if transition["is_first"]:
      self.has_succeeded_in_this_episode = False

    # compute if we have succeeded or not
    is_success = self.success_fn(transition)
    # print(f"[SuccessRateCheck] is_success: {is_success}")
    # if yes, then assign that we have ever succeeded and compute the success rate
    if is_success: # We only have to recompute the success rate whenever there is a change in the success state
      if not self.has_succeeded_in_this_episode:
        self.success_histories[-1].append(1.0) # If successful, append 1
      self.current_success_rate[-1] = np.mean(list(self.success_histories[-1])) # recompute the success
      self.has_succeeded_in_this_episode = True

    # Recompute the success rate when it is the end of the episode
    if transition["is_last"] or transition["is_terminal"]:
      if not self.has_succeeded_in_this_episode: # If not successful, append 0
        self.success_histories[-1].append(0.0) # Note that we don't consider the successful case here because we have added it the first time we encounter sucessful state
        self.current_success_rate[-1] = np.mean(list(self.success_histories[-1])) # recompute

    return {
      "success_rate": np.asarray(self.current_success_rate[-1]),
      "is_success": is_success,
      "has_succeeded": self.has_succeeded_in_this_episode
    }

class MeasureRStability(EnvStepLambda):
  def __init__(self, window_size=100, last_k_episode=100, reward_offset: float=50.0):
    """Measure R-stabiulity as in Alex Ororbia's robot paper

    Args:
        env (_type_): _description_
    """
    self.window_size = window_size
    self.last_k_episode = last_k_episode
    self.cumulative_rewards = []
    self.current_cumulative_reward = 0.0
    self.reward_offset = reward_offset

  @functools.cached_property
  def space(self):
    return {"relative_stability": spacelib.Space(np.float32, ())}

  @property
  def relative_stability(self):
    if not self.cumulative_rewards:
      # TODO: What should be the initial value?
      return 1.0
    rolling_average_cum_rew = np.convolve(self.cumulative_rewards, np.ones(self.window_size)/self.window_size, mode='valid')
    max_reward = np.max(self.cumulative_rewards)
    if max_reward == 0:
      max_reward = 1
    rolling_average_cum_rew = rolling_average_cum_rew[-np.minimum(self.last_k_episode, len(rolling_average_cum_rew)):]
    return np.mean(np.abs((rolling_average_cum_rew - max_reward)/max_reward))

  def __call__(self, transition: ObservationAndPreviousActionType) -> ValueType:
    if "new_lap" in transition and transition["new_lap"]: # addtional meta action
      self.cumulative_rewards = []
      self.current_cumulative_reward = 0.0

    if transition["is_last"] or transition["is_terminal"]:
      self.cumulative_rewards.append(self.current_cumulative_reward)
      self.current_cumulative_reward = 0.0
    else:
      self.current_cumulative_reward += (transition["reward"] + self.reward_offset)
    return {"relative_stability": self.relative_stability}

class MeasureCoverage(EnvStepLambda):
  def __init__(self, task: str, obs_space: Dict[str, spacelib.Space], **kwargs):
    """Measure state coverage
    only support vector state space
    """
    suite, envname = task.split("_", 1)
    self.obs_space = obs_space
    self.suite = suite
    self.envname = envname
    self.kwargs = kwargs
    self._supported = True

    # Main setup
    self._setup(suite, envname, **kwargs)

  def _setup(self, suite, envname, **kwargs):
    # Have to setup these values bases on the environment
    # Denote the value in each state vector from the observation dictionary
    state_factor_idx: Dict[str, List[int]] = {}

    ############# WRITEME: This method setup necessary variables for running success function #####
    if suite == "gymrobotics":
      state_factor_idx: Dict[str, List[int]] = {"observation": [0, 1, 2]}
      lows = {k: np.asarray([1.1, 0.6, 0.4]) for k in state_factor_idx.keys()}
      highs = {k: np.asarray([1.5, 1.9, 0.8]) for k in state_factor_idx.keys()}
      bin_size = kwargs.get("bin_size", 5)
    elif suite == "gym":
      if envname == "mtc":
        state_factor_idx: Dict[str, List[int]] = {"state": [0, 1]}
      else:
        state_factor_idx: Dict[str, List[int]] = {"state": [0]}
      lows = {k: self.obs_space[k].low[v] for k, v in state_factor_idx.items()}
      highs = {k: self.obs_space[k].high[v] for k, v in state_factor_idx.items()}
      bin_size = kwargs.get("bin_size", 10)
    elif suite == "metaworld":
      state_factor_idx: Dict[str, List[int]] = {"state": [0, 1, 2]}
      """
      _HAND_SPACE = Box(
          np.array([-0.525, 0.348, -0.0525]),
          np.array([+0.525, 1.025, 0.7]),
          dtype=np.float64,
      )
      """
      lows = {k: np.asarray([-0.525, 0.348, -0.0525]) for k in state_factor_idx.keys()}
      highs = {k: np.asarray([+0.525, 1.025, 0.7]) for k in state_factor_idx.keys()}
      bin_size = kwargs.get("bin_size", 10)
    else:
      self._supported = False
      print(f"[Coverage] Mesurement not supported in this environment/suite: {suite}")
      # raise NotImplementedError("Try implement covergae method for your suite")
    ##############################################################

    ########################## SETUP HERE ########################
    if self._supported:
      # Checking discrete space
      for k, v in self.obs_space.items():
        if k in state_factor_idx:
          assert not v.discrete, f"Does not support discrete observation space {k}"
      # Number of factor/value for each space
      n_state_factors: Dict[str, int] = {k: len(v) for k, v in state_factor_idx.items()}
      # The actual bin flags array: {k: (bin_size, n_state_factors[k])}
      binss: Dict[str, np.ndarray] = {
        k: np.linspace(
          lows[k],
          highs[k],
          bin_size + 1
        )[1:-1, ...] for k, v in state_factor_idx.items()
      }
      # print(f"binss: {binss}")
      # the number of bins for each state factor
      state_sizes = {k: np.asarray([bin_size for i in range(v)]) for k, v in n_state_factors.items()}
      # Number of unique states permutation
      total_uniques = {k: np.prod(v) for k, v in state_sizes.items()}
      # Checkingh number overflow
      for k, v in total_uniques.items():
        assert v > 0, f"Too many state permutation caused the total number of permutations overflow at space {k}"
      # This is used in unique id computation
      bases: Dict[str, np.ndarray] = {
        k: np.concatenate(
          [v[::-1].cumprod()[::-1][1:], np.asarray([1,])],
          dtype=np.float32
        ) for k, v in state_sizes.items()
      }
      # This is the actual counter
      counter: Dict[str, set] = {k: set() for k in state_factor_idx.keys()}

      ### Setup to self
      self.state_factor_idx = state_factor_idx
      self.counter = counter
      self.bases = bases
      self.binss = binss
      self.n_state_factors = n_state_factors
      self.total_uniques = total_uniques

  @functools.cached_property
  def space(self):
    if self._supported:
      # Have to do this in order to not have the mean key as the coverage factor
      # We reserves the mean key for computing the mean across all coverages
      assert "mean" not in self.obs_space, "Can you consider something else other than the `mean` key for observation?"
      spaces = {f"coverage_{k}": spacelib.Space(np.float32, (), 0.0, 1.0) for k in self.state_factor_idx.keys()}
      spaces["coverage_mean"] = spacelib.Space(np.float32, (), 0.0, 1.0)
      return spaces
    else:
      return {}

  def __call__(self, transition: ObservationAndPreviousActionType) -> ValueType:
    """Count a particular continuous state or a discrete state

    Args:
      state (np.ndarray): _description_

    Returns:
      _type_: _description_
    """
    if self._supported:
      if "new_lap" in transition and transition["new_lap"]: # addtional meta action
        self.counter = {k: set() for k in self.state_factor_idx.keys()}

      for key in self.state_factor_idx.keys():
        state_factor_idx = self.state_factor_idx[key]
        state = transition[key]
        binss = self.binss[key]
        n_state_factors = self.n_state_factors[key]
        bases = self.bases[key]

        assert len(state.shape) == 1, "The state should be a 1D vector"
        state_bin_idx = []
        for i in range(n_state_factors):
          # print(f"state[state_factor_idx[i]]: {state[state_factor_idx[i]]}")
          id = np.digitize(state[state_factor_idx[i]], binss[:, i])
          state_bin_idx.append(id)

        # actual counting
        id = np.dot(state_bin_idx, bases).astype(int)
        self.counter[key].add(id)

      state_coverages = {f"coverage_{k}": float(len(self.counter[k])) / float(self.total_uniques[k]) for k in self.state_factor_idx.keys()}
      state_coverages["coverage_mean"] = np.mean(list(state_coverages.values()))
      return state_coverages
    else:
      return {}
