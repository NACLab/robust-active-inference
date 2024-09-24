# %%

import math
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
import numpy as np
from typing import Optional, Any, SupportsFloat, Tuple, Union, Dict
# from collections import OrderedDict
import os
# os.environ["PYOPENGL_PLATFORM"] = "osmesa"
# os.environ["MUJOCO_GL"] = "osmesa"
# https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/classic_control/continuous_mountain_car.py
def verify_number_and_cast(x: SupportsFloat) -> float:
    """Verify parameter is a single number and cast to a float."""
    try:
        x = float(x)
    except (ValueError, TypeError) as e:
        raise ValueError(f"An option ({x}) could not be converted to a float.") from e
    return x

def maybe_parse_reset_bounds(
    options: Optional[dict], default_low: float, default_high: float
) -> Tuple[float, float]:
    """
    This function can be called during a reset() to customize the sampling
    ranges for setting the initial state distributions.
    Args:
      options: Options passed in to reset().
      default_low: Default lower limit to use, if none specified in options.
      default_high: Default upper limit to use, if none specified in options.
    Returns:
      Tuple of the lower and upper limits.
    """
    if options is None:
        return default_low, default_high

    low = options.get("low") if "low" in options else default_low
    high = options.get("high") if "high" in options else default_high

    # We expect only numerical inputs.
    low = verify_number_and_cast(low)
    high = verify_number_and_cast(high)
    if low > high:
        raise ValueError(
            f"Lower bound ({low}) must be lower than higher bound ({high})."
        )

    return low, high

# Adapted from https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/classic_control/continuous_mountain_car.py
# Changing the reward to sparse reward. -1 for every steps, 0 for end
class MountainCarContinuous(gym.Env):
  """
  ## Description
  The Mountain Car MDP is a deterministic MDP that consists of a car placed stochastically
  at the bottom of a sinusoidal valley, with the only possible actions being the accelerations
  that can be applied to the car in either direction. The goal of the MDP is to strategically
  accelerate the car to reach the goal state on top of the right hill. There are two versions
  of the mountain car domain in gymnasium: one with discrete actions and one with continuous.
  This version is the one with continuous actions.
  This MDP first appeared in [Andrew Moore's PhD Thesis (1990)](https://www.cl.cam.ac.uk/techreports/UCAM-CL-TR-209.pdf)
  ```
  @TECHREPORT{Moore90efficientmemory-based,
      author = {Andrew William Moore},
      title = {Efficient Memory-based Learning for Robot Control},
      institution = {University of Cambridge},
      year = {1990}
  }
  ```
  ## Observation Space
  The observation is a `ndarray` with shape `(2,)` where the elements correspond to the following:
  | Num | Observation                          | Min  | Max | Unit         |
  |-----|--------------------------------------|------|-----|--------------|
  | 0   | position of the car along the x-axis | -Inf | Inf | position (m) |
  | 1   | velocity of the car                  | -Inf | Inf | position (m) |
  ## Action Space
  The action is a `ndarray` with shape `(1,)`, representing the directional force applied on the car.
  The action is clipped in the range `[-1,1]` and multiplied by a power of 0.0015.
  ## Transition Dynamics:
  Given an action, the mountain car follows the following transition dynamics:
  *velocity<sub>t+1</sub> = velocity<sub>t+1</sub> + force * self.power - 0.0025 * cos(3 * position<sub>t</sub>)*
  *position<sub>t+1</sub> = position<sub>t</sub> + velocity<sub>t+1</sub>*
  where force is the action clipped to the range `[-1,1]` and power is a constant 0.0015.
  The collisions at either end are inelastic with the velocity set to 0 upon collision with the wall.
  The position is clipped to the range [-1.2, 0.6] and velocity is clipped to the range [-0.07, 0.07].
  ## Reward
  A negative reward of *-0.1 * action<sup>2</sup>* is received at each timestep to penalise for
  taking actions of large magnitude. If the mountain car reaches the goal then a positive reward of +100
  is added to the negative reward for that timestep.
  ## Starting State
  The position of the car is assigned a uniform random value in `[-0.6 , -0.4]`.
  The starting velocity of the car is always assigned to 0.
  ## Episode End
  The episode ends if either of the following happens:
  1. Termination: The position of the car is greater than or equal to 0.45 (the goal position on top of the right hill)
  2. Truncation: The length of the episode is 999.
  ## Arguments
  ```python
  import gymnasium as gym
  gym.make('MountainCarContinuous-v0')
  ```
  On reset, the `options` parameter allows the user to change the bounds used to determine
  the new random state.
  ## Version History
  * v0: Initial versions release (1.0.0)
  """

  metadata = {
      "render_modes": ["human", "rgb_array"],
      "render_fps": 30,
  }

  def __init__(self, render_mode: Optional[str] = None, goal_velocity=0, easy_mode=True, reward_shaping=False, freeze_when_done=False):
      self.easy_mode = easy_mode
      self.reward_shaping = reward_shaping
      self.min_action = -1.0
      self.max_action = 1.0
      self.min_position = -1.2
      self.max_position = 0.6
      self.max_speed = 0.07
      self.goal_position = (
          0.45  # was 0.5 in gymnasium, 0.45 in Arnaud de Broissia's version
      )
      self.goal_velocity = goal_velocity
      self.power = 0.0015

      self.low_state = np.array(
          [self.min_position, -self.max_speed], dtype=np.float32
      )
      self.high_state = np.array(
          [self.max_position, self.max_speed], dtype=np.float32
      )

      self.render_mode = render_mode

      self.screen_width = 600
      self.screen_height = 400
      self.screen = None
      self.clock = None
      self.isopen = True

      self.action_space = spaces.Box(
          low=self.min_action, high=self.max_action, shape=(1,), dtype=np.float32
      )
      self.observation_space = spaces.Box(
          low=self.low_state, high=self.high_state, dtype=np.float32
      )

      # chores for keeping track of terminated
      self.terminated = False

      self.freeze_when_done = freeze_when_done

  def step(self, action: np.ndarray):
      position = self.state[0]
      velocity = self.state[1]

      if not self.terminated:
        force = min(max(action[0], self.min_action), self.max_action)
        velocity += force * self.power - 0.0025 * math.cos(3 * position)
        if velocity > self.max_speed:
            velocity = self.max_speed
        if velocity < -self.max_speed:
            velocity = -self.max_speed
        position += velocity
        if position > self.max_position:
            position = self.max_position
        if position < self.min_position:
            position = self.min_position
        if position == self.min_position and velocity < 0:
            velocity = 0

      # Convert a possible numpy bool to a Python bool.
      if self.easy_mode:
        terminated = bool(position >= self.goal_position)
      else:
        terminated = bool(position >= self.goal_position and velocity >= self.goal_velocity)

      self.terminated = terminated
      next_state = np.array([position, velocity], dtype=np.float32)

      if self.reward_shaping:
        # reward = 0
        # if terminated:
        #   reward = 100.0
        # reward -= math.pow(action[0], 2) * 0.1
        # reward based on dynamics of the environment
        # https://towardsdatascience.com/open-ai-gym-classic-control-problems-rl-dqn-reward-functions-16a1bc2b007
        reward = 500*((math.sin(3*next_state[0]) * 0.0025 + 0.5 * next_state[1] * next_state[1]) - (math.sin(3*self.state[0]) * 0.0025 + 0.5 * self.state[1] * self.state[1])) 
      else:
        reward = -1.0
        if terminated:
          reward = 0.0
          # reward = 100.0

      self.state = next_state

      if self.render_mode == "human":
          self.render()
    #   print(self.terminated)
      return self.state, reward, False if self.freeze_when_done else self.terminated, False, {}

  def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
      super().reset(seed=seed)
      # Note that if you use custom reset bounds, it may lead to out-of-bound
      # state/observations.
      low, high = maybe_parse_reset_bounds(options, -0.6, -0.4)
      self.state = np.array([self.np_random.uniform(low=low, high=high), 0])

      # reset the terminated state
      self.terminated = False

      if self.render_mode == "human":
        self.render()
      return np.array(self.state, dtype=np.float32), {}

  def _height(self, xs):
      return np.sin(3 * xs) * 0.45 + 0.55

  def render(self):
      if self.render_mode is None:
          assert self.spec is not None
          gym.logger.warn(
              "You are calling render method without specifying any render mode. "
              "You can specify the render_mode at initialization, "
              f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
          )
          return

      try:
          import pygame
          from pygame import gfxdraw
      except ImportError as e:
          raise DependencyNotInstalled(
              "pygame is not installed, run `pip install gymnasium[classic-control]`"
          ) from e

      if self.screen is None:
          pygame.init()
          if self.render_mode == "human":
              pygame.display.init()
              self.screen = pygame.display.set_mode(
                  (self.screen_width, self.screen_height)
              )
          else:  # mode == "rgb_array":
              self.screen = pygame.Surface((self.screen_width, self.screen_height))
      if self.clock is None:
          self.clock = pygame.time.Clock()

      world_width = self.max_position - self.min_position
      scale = self.screen_width / world_width
      carwidth = 40
      carheight = 20

      self.surf = pygame.Surface((self.screen_width, self.screen_height))
      self.surf.fill((255, 255, 255))

      pos = self.state[0]

      xs = np.linspace(self.min_position, self.max_position, 100)
      ys = self._height(xs)
      xys = list(zip((xs - self.min_position) * scale, ys * scale))

      pygame.draw.aalines(self.surf, points=xys, closed=False, color=(0, 0, 0))

      clearance = 10

      l, r, t, b = -carwidth / 2, carwidth / 2, carheight, 0
      coords = []
      for c in [(l, b), (l, t), (r, t), (r, b)]:
          c = pygame.math.Vector2(c).rotate_rad(math.cos(3 * pos))
          coords.append(
              (
                  c[0] + (pos - self.min_position) * scale,
                  c[1] + clearance + self._height(pos) * scale,
              )
          )

      gfxdraw.aapolygon(self.surf, coords, (0, 0, 0))
      gfxdraw.filled_polygon(self.surf, coords, (0, 0, 0))

      for c in [(carwidth / 4, 0), (-carwidth / 4, 0)]:
          c = pygame.math.Vector2(c).rotate_rad(math.cos(3 * pos))
          wheel = (
              int(c[0] + (pos - self.min_position) * scale),
              int(c[1] + clearance + self._height(pos) * scale),
          )

          gfxdraw.aacircle(
              self.surf, wheel[0], wheel[1], int(carheight / 2.5), (128, 128, 128)
          )
          gfxdraw.filled_circle(
              self.surf, wheel[0], wheel[1], int(carheight / 2.5), (128, 128, 128)
          )

      flagx = int((self.goal_position - self.min_position) * scale)
      flagy1 = int(self._height(self.goal_position) * scale)
      flagy2 = flagy1 + 50
      gfxdraw.vline(self.surf, flagx, flagy1, flagy2, (0, 0, 0))

      gfxdraw.aapolygon(
          self.surf,
          [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)],
          (204, 204, 0),
      )
      gfxdraw.filled_polygon(
          self.surf,
          [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)],
          (204, 204, 0),
      )

      self.surf = pygame.transform.flip(self.surf, False, True)
      self.screen.blit(self.surf, (0, 0))
      if self.render_mode == "human":
          pygame.event.pump()
          self.clock.tick(self.metadata["render_fps"])
          pygame.display.flip()

      elif self.render_mode == "rgb_array":
          return np.transpose(
              np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
          )

  def close(self):
      if self.screen is not None:
          import pygame

          pygame.display.quit()
          pygame.quit()
          self.isopen = False

class MountainCarContinuousImageObservation(MountainCarContinuous):

  def __init__(self, render_mode=None, goal_velocity=0, easy_mode=True, reward_shaping=False, freeze_when_done=False):
      super().__init__(render_mode, goal_velocity, easy_mode, reward_shaping, freeze_when_done)
      
      self.goal_state = [self.goal_position, self.goal_velocity]
      # TODO: Might try higher value here
      #   self.obs_height = 64
      #   self.obs_width = 96
    #   self.obs_height = 256
    #   self.obs_width = 384
      # self.obs_height = 128
      # self.obs_width = 192
      self.obs_height = 64
      self.obs_width = 64
      self.goal_image = self._observation(self.goal_state)
      # self.observation_space = spaces.Box(
      #   np.zeros((self.obs_height, self.obs_width, 3), dtype=np.uint8),
      #   np.ones((self.obs_height, self.obs_width, 3), dtype=np.uint8) * 255,
      # )
      # self.state_space = spaces.Box(
      #     low=self.low_state, high=self.high_state, dtype=np.float32
      # ) # the state space is now the same as MDP observation space
      self.observation_space = gym.spaces.Dict({
        "image": gym.spaces.Box(
          0, 255, (64, 64, 3), 
          dtype=np.uint8
        ),
        "state": self.observation_space
      })

  def step(self, action: np.ndarray):
    state, reward, terminated, truncated, info = super().step(action)
    observation = {}
    observation["image"] = self._observation(self.state)
    observation["state"] = self.state.copy()
    info["goal"] = self.goal_image
    info["success"] = self.terminated
    return observation, reward, terminated, truncated, info

  def _observation(self, state):
      try:
          import pygame
          from pygame import gfxdraw
      except ImportError as e:
          raise DependencyNotInstalled(
              "pygame is not installed, run `pip install gymnasium[classic-control]`"
          ) from e
      self.obs_screen = pygame.Surface((self.obs_width, self.obs_height))
      if self.clock is None:
          self.clock = pygame.time.Clock()
      world_width = self.max_position - self.min_position
      scale = self.obs_width / world_width
      # TODO: Change size of car to manipulate
      #   carwidth = 40 # this one is too large for 64 x 96 frame
      #   carheight = 20
      # carwidth = 12
      # carheight = 6
      carwidth = 10
      carheight = 5

      self.obs_surf = pygame.Surface((self.obs_width, self.obs_height))
      self.obs_surf.fill((20, 20, 20))

      pos = state[0]

      xs = np.linspace(self.min_position, self.max_position, 500) # used to be 100 here, 200 for more clearer line (i think :v)
      ys = self._height(xs)
      xys = list(zip((xs - self.min_position) * scale, ys * scale))

      pygame.draw.aalines(self.obs_surf, points=xys, closed=False, color=(200, 200, 200))

      # TODO: Change this to manipulate the height of the car in the frame
      clearance = 3.3 # 4 # 10 

      l, r, t, b = -carwidth / 2, carwidth / 2, carheight, 0
      coords = []
      for c in [(l, b), (l, t), (r, t), (r, b)]:
          c = pygame.math.Vector2(c).rotate_rad(math.cos(3 * pos))
          coords.append(
              (
                  c[0] + (pos - self.min_position) * scale,
                  c[1] + clearance + self._height(pos) * scale,
              )
          )

      gfxdraw.aapolygon(self.obs_surf, coords, (255, 255, 255))
      gfxdraw.filled_polygon(self.obs_surf, coords, (255, 255, 255))

      for c in [(carwidth / 4, 0), (-carwidth / 4, 0)]:
          c = pygame.math.Vector2(c).rotate_rad(math.cos(3 * pos))
          wheel = (
              int(c[0] + (pos - self.min_position) * scale),
              int(c[1] + clearance + self._height(pos) * scale),
          )

          gfxdraw.aacircle(
              self.obs_surf, wheel[0], wheel[1], int(carheight / 2.8), (100, 100, 100)
          )
          gfxdraw.filled_circle(
              self.obs_surf, wheel[0], wheel[1], int(carheight / 2.8), (100, 100, 100)
          )

      flagx = int((self.goal_position - self.min_position) * scale)
      flagy1 = int(self._height(self.goal_position) * scale)
      # change this to manipulate the height of the flag
      flag_height = 12 # 9 # originally: 50
      flagy2 = flagy1 + flag_height
      gfxdraw.vline(self.obs_surf, flagx, flagy1, flagy2, (255, 255, 255))

      # Also need to manipulate this to change how to draw the flags
      _a = -4 # -4 # originally: -10
      _b = 5 # 6 # originally: 25
      _c = -2 # -2 # originally: -5
      gfxdraw.aapolygon(
          self.obs_surf,
          [(flagx, flagy2), (flagx, flagy2 + _a), (flagx + _b, flagy2 + _c)],
          (204, 204, 0),
      )
      gfxdraw.filled_polygon(
          self.obs_surf,
          [(flagx, flagy2), (flagx, flagy2 + _a), (flagx + _b, flagy2 + _c)],
          (204, 204, 0),
      )

      self.obs_surf = pygame.transform.flip(self.obs_surf, False, True)
      self.obs_screen.blit(self.obs_surf, (0, 0))

      return np.transpose(
        np.array(pygame.surfarray.pixels3d(self.obs_screen)), axes=(1, 0, 2)
    )

  def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
      state, info = super().reset()
      observation = {}
      observation["image"] = self._observation(self.state)
      observation["state"] = self.state.copy()
      info["goal"] = self.goal_image
      # info["state"] = self.state
      info["success"] = self.terminated
      return observation, info

  def _height(self, xs):
      return np.sin(3 * xs) * 0.45 + 0.55

  def render(self):
      if self.render_mode is None:
          assert self.spec is not None
          gym.logger.warn(
              "You are calling render method without specifying any render mode. "
              "You can specify the render_mode at initialization, "
              f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
          )
          return

      try:
          import pygame
          from pygame import gfxdraw
      except ImportError as e:
          raise DependencyNotInstalled(
              "pygame is not installed, run `pip install gymnasium[classic-control]`"
          ) from e

      if self.screen is None:
          pygame.init()
          if self.render_mode == "human":
              pygame.display.init()
              self.screen = pygame.display.set_mode(
                  (self.screen_width, self.screen_height)
              )
          else:  # mode == "rgb_array":
              self.screen = pygame.Surface((self.screen_width, self.screen_height))
      if self.clock is None:
          self.clock = pygame.time.Clock()

      world_width = self.max_position - self.min_position
      scale = self.screen_width / world_width
      carwidth = 40
      carheight = 20

      self.surf = pygame.Surface((self.screen_width, self.screen_height))
      self.surf.fill((255, 255, 255))

      pos = self.state[0]

      xs = np.linspace(self.min_position, self.max_position, 100)
      ys = self._height(xs)
      xys = list(zip((xs - self.min_position) * scale, ys * scale))

      pygame.draw.aalines(self.surf, points=xys, closed=False, color=(0, 0, 0))

      clearance = 10

      l, r, t, b = -carwidth / 2, carwidth / 2, carheight, 0
      coords = []
      for c in [(l, b), (l, t), (r, t), (r, b)]:
          c = pygame.math.Vector2(c).rotate_rad(math.cos(3 * pos))
          coords.append(
              (
                  c[0] + (pos - self.min_position) * scale,
                  c[1] + clearance + self._height(pos) * scale,
              )
          )

      gfxdraw.aapolygon(self.surf, coords, (0, 0, 0))
      gfxdraw.filled_polygon(self.surf, coords, (0, 0, 0))

      for c in [(carwidth / 4, 0), (-carwidth / 4, 0)]:
          c = pygame.math.Vector2(c).rotate_rad(math.cos(3 * pos))
          wheel = (
              int(c[0] + (pos - self.min_position) * scale),
              int(c[1] + clearance + self._height(pos) * scale),
          )

          gfxdraw.aacircle(
              self.surf, wheel[0], wheel[1], int(carheight / 2.5), (128, 128, 128)
          )
          gfxdraw.filled_circle(
              self.surf, wheel[0], wheel[1], int(carheight / 2.5), (128, 128, 128)
          )

      flagx = int((self.goal_position - self.min_position) * scale)
      flagy1 = int(self._height(self.goal_position) * scale)
      flagy2 = flagy1 + 50
      gfxdraw.vline(self.surf, flagx, flagy1, flagy2, (0, 0, 0))

      gfxdraw.aapolygon(
          self.surf,
          [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)],
          (204, 204, 0),
      )
      gfxdraw.filled_polygon(
          self.surf,
          [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)],
          (204, 204, 0),
      )

      self.surf = pygame.transform.flip(self.surf, False, True)
      self.screen.blit(self.surf, (0, 0))
      if self.render_mode == "human":
          pygame.event.pump()
          self.clock.tick(self.metadata["render_fps"])
          pygame.display.flip()

      elif self.render_mode == "rgb_array":
          return np.transpose(
              np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
          )

  def close(self):
      if self.screen is not None:
          import pygame

          pygame.display.quit()
          pygame.quit()
          self.isopen = False

# import matplotlib.pyplot as plt
# from IPython.display import clear_output

# env = MountainCarContinuousImageObservation(render_mode="rgb_array")

# o, i = env.reset()
# plt.imshow(o["image"])
# plt.show()

# for i in range(50):
# #   action = env.action_space.sample()
#   action = np.ones((1,))
#   o, _, _, _, _ = env.step(action)
#   clear_output(wait=True)
#   plt.imshow(o["image"])
#   plt.show()
