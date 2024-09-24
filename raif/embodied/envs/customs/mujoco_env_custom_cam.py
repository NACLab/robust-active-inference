from os import path
from typing import Any, Dict, Optional, Tuple, Union, List

import numpy as np
from numpy.typing import NDArray

import gymnasium as gym
from gymnasium import error, logger, spaces
from gymnasium.spaces import Space

try:
    import mujoco
except ImportError as e:
    MUJOCO_IMPORT_ERROR = e
else:
    MUJOCO_IMPORT_ERROR = None


DEFAULT_SIZE = 480


class BaseMujocoEnv(gym.Env[np.float64, np.float32]):
    """Superclass for all MuJoCo environments."""

    def __init__(
        self,
        model_path,
        frame_skip,
        observation_space: Space,
        render_mode: Optional[str] = None,
        width: int = DEFAULT_SIZE,
        height: int = DEFAULT_SIZE,
        camera_id: Optional[int] = None,
        camera_name: Optional[str] = None,
    ):
        """Base abstract class for mujoco based environments.

        Args:
            model_path: Path to the MuJoCo Model.
            frame_skip: Number of MuJoCo simulation steps per gym `step()`.
            observation_space: The observation space of the environment.
            render_mode: The `render_mode` used.
            width: The width of the render window.
            height: The height of the render window.
            camera_id: The camera ID used.
            camera_name: The name of the camera used (can not be used in conjunction with `camera_id`).

        Raises:
            OSError: when the `model_path` does not exist.
            error.DependencyNotInstalled: When `mujoco` is not installed.
        """
        if model_path.startswith(".") or model_path.startswith("/"):
            self.fullpath = model_path
        elif model_path.startswith("~"):
            self.fullpath = path.expanduser(model_path)
        else:
            self.fullpath = path.join(path.dirname(__file__), "assets", model_path)
        if not path.exists(self.fullpath):
            raise OSError(f"File {self.fullpath} does not exist")

        self.width = width
        self.height = height
        # may use width and height
        self.model, self.data = self._initialize_simulation()

        self.init_qpos = self.data.qpos.ravel().copy()
        self.init_qvel = self.data.qvel.ravel().copy()

        self.frame_skip = frame_skip

        assert self.metadata["render_modes"] == [
            "human",
            "rgb_array",
            "depth_array",
        ], self.metadata["render_modes"]
        if "render_fps" in self.metadata:
            assert (
                int(np.round(1.0 / self.dt)) == self.metadata["render_fps"]
            ), f'Expected value: {int(np.round(1.0 / self.dt))}, Actual value: {self.metadata["render_fps"]}'

        self.observation_space = observation_space
        self._set_action_space()

        self.render_mode = render_mode
        self.camera_name = camera_name
        self.camera_id = camera_id

    def _set_action_space(self):
        bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds.T
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        return self.action_space

    # methods to override:
    # ----------------------------

    def reset_model(self) -> NDArray[np.float64]:
        """
        Reset the robot degrees of freedom (qpos and qvel).
        Implement this in each subclass.
        """
        raise NotImplementedError

    def _initialize_simulation(self) -> Tuple[Any, Any]:
        """
        Initialize MuJoCo simulation data structures mjModel and mjData.
        """
        raise NotImplementedError

    def _reset_simulation(self) -> None:
        """
        Reset MuJoCo simulation data structures, mjModel and mjData.
        """
        raise NotImplementedError

    def _step_mujoco_simulation(self, ctrl, n_frames) -> None:
        """
        Step over the MuJoCo simulation.
        """
        raise NotImplementedError

    def render(self) -> Union[NDArray[np.float64], None]:
        """
        Render a frame from the MuJoCo simulation as specified by the render_mode.
        """
        raise NotImplementedError

    # -----------------------------
    def _get_reset_info(self) -> Dict[str, float]:
        """Function that generates the `info` that is returned during a `reset()`."""
        return {}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)

        self._reset_simulation()

        ob = self.reset_model()
        info = self._get_reset_info()

        if self.render_mode == "human":
            self.render()
        return ob, info

    def set_state(self, qpos, qvel) -> None:
        """
        Set the joints position qpos and velocity qvel of the model. Override this method depending on the MuJoCo bindings used.
        """
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)

    @property
    def dt(self) -> float:
        return self.model.opt.timestep * self.frame_skip

    def do_simulation(self, ctrl, n_frames) -> None:
        """
        Step the simulation n number of frames and applying a control action.
        """
        # Check control input is contained in the action space
        if np.array(ctrl).shape != (self.model.nu,):
            raise ValueError(
                f"Action dimension mismatch. Expected {(self.model.nu,)}, found {np.array(ctrl).shape}"
            )
        self._step_mujoco_simulation(ctrl, n_frames)

    def close(self):
        """Close all processes like rendering contexts"""
        raise NotImplementedError

    def get_body_com(self, body_name) -> NDArray[np.float64]:
        """Return the cartesian position of a body frame"""
        raise NotImplementedError

    def state_vector(self) -> NDArray[np.float64]:
        """Return the position and velocity joint states of the model"""
        return np.concatenate([self.data.qpos.flat, self.data.qvel.flat])


class MujocoEnv(BaseMujocoEnv):
    """Superclass for MuJoCo environments."""

    def __init__(
        self,
        model_path,
        frame_skip,
        observation_space: Space,
        render_mode: Optional[str] = None,
        width: int = DEFAULT_SIZE,
        height: int = DEFAULT_SIZE,
        camera_id: Optional[int] = None,
        camera_name: Optional[str] = None,
        default_camera_config: Optional[dict] = None,
    ):
        if MUJOCO_IMPORT_ERROR is not None:
            raise error.DependencyNotInstalled(
                f"{MUJOCO_IMPORT_ERROR}. "
                "(HINT: you need to install mujoco, run `pip install gymnasium[mujoco]`.)"
            )

        super().__init__(
            model_path,
            frame_skip,
            observation_space,
            render_mode,
            width,
            height,
            camera_id,
            camera_name,
        )

        from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer

        # NOTE: modified
        # By default, if we are using only one camera, the list of cameras is
        # instantiate to a list of single camera
        # If we are using multiple camera, the main renderer will be the first one on the list
        self.multiple_camera = False
        if isinstance(default_camera_config, List):
            self.multiple_camera = True
            self.mujoco_renderers = [
                MujocoRenderer(
                    self.model, self.data, camera_config
                ) for camera_config in default_camera_config
            ]
            self.mujoco_renderer = self.mujoco_renderers[0]
        elif isinstance(default_camera_config, dict) or default_camera_config is None:
            self.mujoco_renderer = MujocoRenderer(
                self.model, self.data, default_camera_config
            )
            self.mujoco_renderers = [self.mujoco_renderer]
        else:
            raise ValueError(default_camera_config)

    def _initialize_simulation(
        self,
    ):
        model = mujoco.MjModel.from_xml_path(self.fullpath)
        # MjrContext will copy model.vis.global_.off* to con.off*
        model.vis.global_.offwidth = self.width
        model.vis.global_.offheight = self.height
        data = mujoco.MjData(model)
        return model, data

    def _reset_simulation(self):
        mujoco.mj_resetData(self.model, self.data)

    def set_state(self, qpos, qvel):
        super().set_state(qpos, qvel)
        self.data.qpos[:] = np.copy(qpos)
        self.data.qvel[:] = np.copy(qvel)
        if self.model.na == 0:
            self.data.act[:] = None
        mujoco.mj_forward(self.model, self.data)

    def _step_mujoco_simulation(self, ctrl, n_frames):
        self.data.ctrl[:] = ctrl

        mujoco.mj_step(self.model, self.data, nstep=n_frames)

        # As of MuJoCo 2.0, force-related quantities like cacc are not computed
        # unless there's a force sensor in the model.
        # See https://github.com/openai/gym/issues/1541
        mujoco.mj_rnePostConstraint(self.model, self.data)

    def render(self):
        return [renderer.render(
            self.render_mode, self.camera_id, self.camera_name
        ) for renderer in self.mujoco_renderers]

    def close(self):
        if self.mujoco_renderer is not None:
            self.mujoco_renderer.close()

    def get_body_com(self, body_name):
        return self.data.body(body_name).xpos
