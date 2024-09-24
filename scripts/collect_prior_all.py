"""
File: collect_prior_all.py
Author: Viet Nguyen
Date: 2024-05-24

Description: The top most file which collect the episode from all the possible environments
  To run: python scripts/collect_prior_all.py
"""

# %%

import subprocess

# python scripts/collect_prior.py --configs collect metaworld --task metaworld_button-press-v2 --run.steps 3000 --logroot logs_expert --expname metaworld_button-press-v2
env_list = [
    "gym_mtc",
    "metaworld_button-press-v2",
    "metaworld_drawer-close-v2",
    "metaworld_window-open-v2",
    "metaworld_handle-pull-v2",
    "metaworld_door-close-v2",
    "metaworld_door-open-v2",
    "metaworld_reach-v2",
    "metaworld_pick-place-v2", # Experts failed on this task
    "metaworld_push-v2", # Experts do not succeed much on this task
    "metaworld_soccer-v2", # Experts failed on this task
    "metaworld_plate-slide-v2",
    "metaworld_disassemble-v2", # Experts failed on this task
    "metaworld_lever-pull-v2", # Experts do not succeed much on this task
    "robosuite_Door",
    "robosuite_Lift",
]


for env in env_list:
  suite, task = env.split("_", 1)

  if suite == "gym":
    suite = "gym_mtc"

  subprocess.call("python scripts/collect_prior.py " \
                f"--configs collect {suite} " \
                f"--task {env} " \
                f"--run.steps 10000 " \
                "--logroot logs_expert " \
                f"--expname {env} "
                , shell=True)




