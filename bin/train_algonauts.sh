#!/bin/bash

#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --job-name=micro
#SBATCH --partition=agpu72
#SBATCH --nodelist=c[2005-2008]

set -x
set -e

RND_MASTER_PORT=$(( ( RANDOM % 10000 )  + 1000 ))

# Slurm
OFFSET=${OFFSET:-0}
WORLD_SIZE=${WORLD_SIZE:-1}
MASTER_ADDR=${MASTER_ADDR:-c1905}
MASTER_PORT=${MASTER_PORT:-$RND_MASTER_PORT}
NGPUS=${NGPUS:-1}
PARTITION=${PARTITION:-'agpu72'}
JOB_NAME=${JOB_NAME:-'cviu'}
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
CPUS_PER_TASK=${CPUS_PER_TASK:-4}
NODE_LIST=${NODE_LIST:-'c1905'}

RUN_MODE=${RUN_MODE:-"dist"}

export WORLD_SIZE=$WORLD_SIZE
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT
export OFFSET=${OFFSET}

model_name='vit_small_patch16_224'

# Data
batch_size=32
lr=2.5e-4
distributed=True
epochs=100
img_size=224
saveckp_freq=5
fold=0
output_dir=logs/baseline/${model_name}/${fold}/
data_dir=/scratch/1576189/data

# export CUDA_VISIBLE_DEVICES=0

if [ "$RUN_MODE" = "dist" ]; then
        command="python -u -m torch.distributed.launch --nproc_per_node=${NGPUS} --master_port ${MASTER_PORT}"
elif [ "$RUN_MODE" = "slurm" ]; then
        command="srun -p ${PARTITION} \
                --job-name ${JOB_NAME} \
                --gres=gpu:${GPUS_PER_NODE} \
                --ntasks-per-node ${GPUS_PER_NODE} \
                --ntasks=${NGPUS} \
                --nodelist=${NODE_LIST} \
                --time=3-00:00:00 \
                --cpus-per-task=${CPUS_PER_TASK} python"
elif [ "$RUN_MODE" = "slurm_sbatch" ]; then
        command="srun python"
else
        command="python"
fi

echo "Run command ", $command

PYTHONPATH=. $command \
        scripts/train.py \
        --model_name ${model_name} \
        --output_dir ${output_dir} \
        --data_dir ${data_dir} \
        --batch_size_per_gpu ${batch_size} \
        --lr ${lr} \
        --img_size ${img_size} \
        --fold ${fold} \
        --epochs ${epochs} \
        --distributed ${distributed} \
        --saveckp_freq ${saveckp_freq} \
        --num_workers 16 \
        --use_fp16 False
