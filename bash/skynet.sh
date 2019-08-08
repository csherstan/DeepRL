#!/usr/bin/env bash
export PROJECT_PATH="/home/sherstan/workspace/return_ae"
export BASH_CFG=$PROJECT_PATH/return_ae/bash/skynet.sh
export DATA=/home/sherstan/data
export WORKSPACE=/home/sherstan/workspace
export THREE_D_CTRL=${WORKSPACE}/3d_control_deep_rl

activate_venv () {
    source activate return_ae
}