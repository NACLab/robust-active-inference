#!/bin/bash -l

# Before using this file, install python 3.10, and then activate it

# pytorch CUDA 11.8. install pytorch with conda make jax not realizing gpu
# Torch will use cuda 11.8 while jax will use cuda 12
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Jax install CUDA 11 installation and related libraries
pip install optax==0.2.2 flax==0.8.2 tensorflow-cpu==2.16.1 tf-keras==2.16.0 tensorflow-probability==0.24.0
pip install -U "jax[cuda12]"

# Install other related libraies (gymnasium_robotics requires pettingzoo and imageio)
pip install seaborn rich ruamel.yaml==0.17.32 opencv-python opencv-python-headless pettingzoo==1.24.3 imageio==2.33.1 tensorflow-datasets

# installing rl env libs
pip install gymnasium==0.29.1 gym==0.24.1 dm-control

# Robosuite. NOTE: On this require gcc to build. On RC, we might want to enable
#   gcc by `spack load gcc@9.3.0/hufzekv` before doing `bash install.sh`
pip install git+ssh://git@github.com/rxng8/robosuite.git@master

# Install stable baseline3
pip install 'stable-baselines3[extra]==2.3.0' 'sb3-contrib==2.3.0'

# Install ffmpeg for rendering gifs
conda install -c conda-forge ffmpeg=6.1.1 -y # version 7 does not work

# Robosuite requires setting up macro
# This requires python 3.10, otherwise replace the python version in the line
python $CONDA_PREFIX/lib/python3.10/site-packages/robosuite/scripts/setup_macros.py

