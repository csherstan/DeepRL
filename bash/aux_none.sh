#!/usr/bin/env bash

#SBATCH --cpus-per-task=2
#SBATCH --mem=18G
#SBATCH --gres=gpu:1
#SBATCH --time=0-50:00


source $BASH_CFG
activate_venv

echo $GAME

# run from bash directory
time python ../aux_none.py \
--game $GAME --run $SLURM_ARRAY_TASK_ID --data_dir $DATA_DIR/$GAME/aux_none