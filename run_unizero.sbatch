#!/bin/bash
#SBATCH --account=education-eemcs-courses-cse3000
#SBATCH --partition=gpu-a100,gpu-v100

# The default run4
#SBATCH --time=4:00:00

# The default number of parallel tasks per job is 1
#SBATCH --ntasks=1

# Request 1 CPU per active thread of your program (assume 1 unless you specifically set this)
# The default-per-cpu=1024
#SBATCH --cpus-per-task=4

#SBATCH --gpus-per-task=1

# The default memory per node is 1024 megabytes (1GB) (for multiple tasks, specify --mem-per-cpu instead)
#SBATCH --mem-per-cpu=3072


module load apptainer

# Path to your project and container
export PROJECT_DIR="/home/${USER}/projects/UniZero"
export SIF_FILE="${PROJECT_DIR}/container.sif"

# Run inside the container, binding your code into /workspace
srun apptainer exec --nv \
    --bind "${PROJECT_DIR}:/workspace" \
    "${SIF_FILE}" \
    python -m zoo.atari.config.atari_unizero_config \
      --env BoxingNoFrameskip-v4 \
      --seed 0