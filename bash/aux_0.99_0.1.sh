#!/usr/bin/env bash

#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --exclude=compute011

source $BASH_CFG
activate_venv

# run from bash directory
time python ../aux_0.99_0.1.py \
--game BreakoutNoFrameskip-v4 --run $SLURM_ARRAY_TASK_ID