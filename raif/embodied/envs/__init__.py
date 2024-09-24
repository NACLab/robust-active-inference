import gymnasium as gym
# from .customs.mtc import MountainCarContinuousImageObservation
gym.register("mtc", "embodied.envs.customs.mtc:MountainCarContinuousImageObservation")
gym.register("ant", "embodied.envs.customs.ant:SparseAntEnv")
