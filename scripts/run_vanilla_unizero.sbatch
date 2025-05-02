#!/bin/bash
#SBATCH --account=education-eemcs-courses-cse3000
#SBATCH --partition=gpu-a100,gpu-v100

# The default run4
#SBATCH --time=4:00:00

# The default number of parallel tasks per job is 1
#SBATCH --ntasks=1

# Request 1 CPU per active thread of your program (assume 1 unless you specifically set this)
# The default-per-cpu=1024
#SBATCH --cpus-per-task=6

#SBATCH --gpus-per-task=1

# The default memory per node is 1024 megabytes (1GB) (for multiple tasks, specify --mem-per-cpu instead)
#SBATCH --mem-per-cpu=4096

#SBATCH --mail-type=BEGIN,END,FAIL

#SBATCH -o outputs/slurm-%j.out
#SBATCH -e outputs/slurm-%j.err

module load apptainer

# Path to your project and container
export PROJECT_DIR="/home/${USER}/projects/UniZero"
export SIF_FILE="${PROJECT_DIR}/container.sif"
export WANDB_API_KEY=e8eb27778c68fe84c0d4fd15a0240624895bab3f
# Run inside the container, binding your code into /workspace
srun apptainer exec --nv \
    --bind "${PROJECT_DIR}:/workspace" \
    "${SIF_FILE}" \
    python -m zoo.atari.config.atari_unizero_vanilla_config\
      --env BoxingNoFrameskip-v4 \
      --seed 0