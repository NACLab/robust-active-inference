# R-AIF: Solving Sparse-Reward Robotic Tasks from Pixels with Active Inference and World Models

<p align="center">
  <img src="docs/raif_experiment_results.gif" alt="Experiment Results of R-AIF Experiment"
    width="70%"/>
</p>

There is relatively little work that builds general active inference (AIF) models in the context of partially observable Markov decision processes (POMDPs), particularly those that characterize visual, pixel-level environments. Notably, in these POMDP scenarios, the agent must work to understand (infer) the hidden state of the world from raw sensory observations, e.g., pixel intensities. Additionally, even less work exists in examining the most difficult form of POMDP-centered control: continuous action space POMDPs under **sparse reward signals**. This work addresses these issues by introducing a novel AIF framework (which we call robust AIF; R-AIF), incorporating new **prior preference learning techniques** and **self-revision** schedules to help the agent excel in sparse-reward, continuous action, goal-based robotic control environments. This repository contains detailed documentation needed to run our proposed R-AIF agent(s).

The implementation documentation can be found in [`docs/ImplementationDetailsDocumentation.pdf`](docs/ImplementationDetailsDocumentation.pdf), which is very helpful for understanding the logic of (and mathematical framework behind) this research effort and its corresponding code base.

<!-- The preprint of our work can be found here: [ArXiv version](https://arxiv.org/abs/2409.14216) -->

<!-- [**ArXiv Preprint**](https://arxiv.org/abs/2409.14216)
| [**Video Description**](https://youtu.be/4dH1D17ry4s?si=bvUhJ9IIgi3J1zeV)
| [**Blog**](https://vietdung.me/publications/raif) -->

### ArXiv Preprint, Project Video Description, and Research Blog Can be Found Below
[![ArXiv Preprint](https://img.shields.io/badge/arXiv-2409.14216-B31B1B.svg?style=for-the-badge)](https://arxiv.org/abs/2409.14216)
[![Video Description](https://img.shields.io/badge/youtube-project_video-FF0000?style=for-the-badge&logo=youtube&logoColor=white)](https://youtu.be/4dH1D17ry4s?si=bvUhJ9IIgi3J1zeV)
[![Blog](https://img.shields.io/badge/Website-Research_Blog-961212?style=for-the-badge)](https://vietdung.me/publications/raif)

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

## Experiment(s)

* In order to run the R-AIF agent for the Open-AI gymnasium mountain car problem, we need to first collect expert data. To side-step collecting expensive human-expert data, we have already trained the "expert" in a `MDP` environment using [`PPO algorithm`](https://arxiv.org/abs/1707.06347). Here, we collect a small number of seed episodes. In this repo's case, let's collect `2000` steps as follows:
```bash
python scripts/collect_prior.py --configs collect gym_mtc --run.steps 2000 --expname my_mountain_car_experiment
```

* After collecting seed data, we can then train an R-AIF agent. Note that the experiment name `--expname` flag should be the same for the two commands in order for the agent to correctly utilize the seeded data:
```bash
python scripts/train_aif.py --configs gym_mtc tiny --run.steps 2000000 --run.script train --run.log_every 60 --expname my_mountain_car_experiment
```

* After executing the above, you can then see the change logged in the tensorboard files in the `/logs` folder. To run tensorboard, execute: `tensorboard --logdir logs`. Then, you will be able to observe the results in your browser.

* To collect robosuite data from the keyboard, run the following:
```bash
cd experimental
python robosuite_keyboard.py --configs robosuite --task robosuite_Door --expname robosuite_Door
```

* To collect the performance measurements of the expert given the (R-)AIF model, next run:
```bash
python script/collect_prior_all.py
```

Note that not all of the expert agents are successful in completing the tasks (since some control tasks are very hard). R-AIF learns from and goes beyond this data, trying to score even better than the experts themselves.

# Miscellaneous

* NOTE: `gcc` is required for building `robosuite`. On `slurm`, one might want to enable gcc by `spack load gcc@9.3.0/hufzekv` before doing `bash install.sh`

