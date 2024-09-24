# R-AIF: Solving Sparse-Reward Robotic Tasks from Pixels with Active Inference and World Models

<p align="center">
  <img src="docs/raif_experiment_results.gif" alt="Experiment Results of R-AIF Experiment"
    width="70%"/>
</p>

There is relatively less work that builds AIF models in the context of partially observable Markov decision processes (POMDPs) environments. In POMDP scenarios, the agent must understand the hidden state of the world from raw sensory observations, e.g., pixels. Additionally, less work exists in examining the most difficult form of POMDP-centered control: continuous action space POMDPs under **sparse reward signals**. This work addresses these issues by introducing novel **prior preference learning techniques** and **self-revision** schedules to help the agent excel in sparse-reward, continuous action, goal-based robotic control POMDP environments. This repository contains detailed documentation needed to implement our proposed agent.

The documentation can be found in [`docs/ImplementationDetailsDocumentation.pdf`](docs/ImplementationDetailsDocumentation.pdf)


## Requirements

* Operating System: Linux.
* The system is tested on 1 RTX 3060 and 1 A100.
* [Conda/Miniconda](https://docs.anaconda.com/free/miniconda/) -- python virtual environment/package manager:
```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash
~/miniconda3/bin/conda init zsh
```
* Create conda environment:
```bash
conda create -n jax pip python=3.10 -y
conda activate jax
```

* Install required libraries: `bash install.sh`
* Fully tested on Linux 22.04 LTS with 1 Nvidia RTX 3060 GPU

## Experiment

* In order to run R-AIF agent for gymnasium mountain car, we need to first collect expert data, we have already trained the expert in `MDP` environment using [`PPO algorithm`](https://arxiv.org/abs/1707.06347). We will collect a small number of seed episodes. In this case, let's collect 2000 steps.
```bash
python scripts/collect_prior.py --configs collect gym_mtc --run.steps 2000 --expname my_mountain_car_experiment
```

* After that, we can train R-AIF agent. Note that the experiment name `--expname` flag should be the same for two command in order for the agent to utilize the seeded data:
```bash
python scripts/train_aif.py --configs gym_mtc tiny --run.steps 2000000 --run.script train --run.log_every 60 --expname my_mountain_car_experiment
```

* You can then see the change logged in the tensorboard files in the `/logs` folder. To run tensorboard, execute: `tensorboard --logdir logs`. Then you will be able to see the results in the browser.

* To collect robosuite data from keyboard:
```bash
cd experimental
python robosuite_keyboard.py --configs robosuite --task robosuite_Door --expname robosuite_Door
```

* To collect the performance of the expert given the model, run:
```bash
python script/collect_prior_all.py
```

Note that not all of the expert agent is successful in completing the tasks (since some tasks are very hard). R-AIF learns from these data and tries to even score better than the experts.

# Miscellaneous

* NOTE: `gcc` is required for building `robosuite`. On `slurm`, we might want to enable gcc by `spack load gcc@9.3.0/hufzekv` before doing `bash install.sh`

