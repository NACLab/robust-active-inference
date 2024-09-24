#!/bin/bash -l

source /home/vn1747/.conda/etc/profile.d/conda.sh
conda activate jax

# cancel all job from me
scancel -u vn1747

# Define to call the submission logic in the python file
log_root_folder="./logs"

# NOTE: ALWAYS CHECK IF YOU REMOVE THE LOG DIR
# rm -rf ${log_root_folder}
mkdir -p "${log_root_folder}/rc"

# Train the main aif agent
python scripts/rc/rc_aif.py ${log_root_folder}
