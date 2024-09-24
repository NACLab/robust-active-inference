"""
File: rc_train_aif.py
Author: Viet Nguyen
Date: 2024-03-29

Description: This is the script that train the agent on multiple environment in parallel on RIT RC
"""

# The main file that run true aif agent

import subprocess
import sys
import os
import pathlib

log_root_folder = sys.argv[1]

configs = {
  "gym_mtc tiny": dict(
    tasks=["gym_mtc"]
  ),
  "metaworld medium": dict(
    tasks=[
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
      "metaworld_lever-pull-v2" # Experts do not succeed much on this task
    ]
  ),
  "robosuite medium": dict(
    tasks=[
      "robosuite_Door",
      "robosuite_Lift",
    ]
  )
}

chosen_params = []
chosen_collect_params = []

for trial in range(3, 4):
  for config, kwargs in configs.items():
    suite, size = config.split(" ")
    tasks = kwargs["tasks"]
    kwargs_str = " ".join([f"--{k} {v}" for k, v in kwargs.items() if k in chosen_params])
    collect_kwargs_str = " ".join([f"--{k} {v}" for k, v in kwargs.items() if k in chosen_collect_params])

    for task in tasks:
      expname = f"aif_{task}/trial_{trial}"

      bashstr = "#!/bin/bash\n\n"

      # If there is no prior preference data, then collect
      if not os.path.exists(pathlib.Path(log_root_folder) / expname / "positive_replay"):
        bashstr += f"python scripts/collect_prior.py " \
                  f"--logroot {log_root_folder} " \
                  f"--expname {expname} " \
                  f"--configs collect {suite} " \
                  f"--task {task} {collect_kwargs_str} " \
                  f"--rc True \n\n"

      # main training script
      # Actally, we need to keep agent seed different across trial but same across each agent and environment
      bashstr += f"python scripts/train_aif.py " \
                f"--configs {config} {kwargs_str} " \
                f"--logroot {log_root_folder} " \
                f"--expname {expname} " \
                f"--task {task} " \
                f"--rc True " \
                f"--seed {trial} "

      print(bashstr)

      with open("command.lock", "w") as f:
        f.write(bashstr)

      # NOTE: REMEMBER TO SWITCH BACK TO GOOD AMOUNT OF TIME LIMIT ON RUNNING
      subprocess.call(f"sbatch -J {expname} "\
                      f"-o '{log_root_folder}/rc/%x_%j.out' "\
                      f"-e '{log_root_folder}/rc/%x_%j.err' "\
                      f"--ntasks=1 "\
                      f"--mem-per-cpu=64g "\
                      f"-p tier3 "\
                      f"--account=none "\
                      f"--gres=gpu:a100:1 "\
                      f"--time=5-00:00:00 "\
                      f"command.lock", shell=True)
      if os.path.exists("command.lock"):
        os.remove("command.lock")


