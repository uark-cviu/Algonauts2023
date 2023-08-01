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
SUBJECT=${SUBJECT:-"subj01"}
subject=${SUBJECT}

export WORLD_SIZE=$WORLD_SIZE
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT
export OFFSET=${OFFSET}

# model_name='vit_small_patch16_224'
# model_name='convnext_base_in22ft1k'
# model_name='seresnext101d_32x8d'
# model_name='coat_lite_medium_384.in1k'
model_name='convnext_xlarge.fb_in22k_ft_in1k_384'
# model_name='resnetrs420'
# model_name='maxvit_large_tf_384.in21k_ft_in1k'
# model_name='tf_efficientnet_b7.ns_jft_in1k'
# model_name='resnetrs420'
# model_name='maxvit_base_tf_384.in21k_ft_in1k'
# model_name='eva_giant_patch14_224.clip_ft_in1k'
# model_name='ssl_resnext50_32x4d'
# model_name='xception71'

# Data
batch_size=8
lr=2.5e-4
distributed=True
epochs=12
img_size=384
saveckp_freq=5
scheduler='onecycle'
num_folds=5

JOB_NAME=${model_name}-${subject}


# export CUDA_VISIBLE_DEVICES=0

if [ "$RUN_MODE" = "dist" ]; then
        command="python -u -m torch.distributed.launch --nproc_per_node=${NGPUS} --master_port ${MASTER_PORT}"
elif [ "$RUN_MODE" = "dist_new" ]; then
        command="torchrun --nproc_per_node=${NGPUS} --master_port ${MASTER_PORT}"
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


output_dir=logs_new_loss_nc/finetune_0/${subject}/${model_name}/
# output_dir=/scr1/1594489/logs/roi_pcc_l1_384_ema_ft_backbone/${subject}/${model_name}/
# data_dir=/scratch/1576189/data
data_dir=data/${subject}
csv_file=${data_dir}/kfold.csv

pretrained=logs/leaveone/${subject}/${model_name}/
# pseudo_dir=predictions_vision/ensemble_convnext_xlarge_seresnext101d_32x8d/
pseudo_dir='none'

PYTHONPATH=. $command \
        scripts/train.py \
        --model_name ${model_name} \
        --output_dir ${output_dir} \
        --data_dir ${data_dir} \
        --pseudo_dir ${pseudo_dir} \
        --subject ${SUBJECT} \
        --csv_file ${csv_file} \
        --pretrained ${pretrained} \
        --batch_size_per_gpu ${batch_size} \
        --lr ${lr} \
        --img_size ${img_size} \
        --epochs ${epochs} \
        --distributed ${distributed} \
        --saveckp_freq ${saveckp_freq} \
        --scheduler ${scheduler} \
        --num_workers 4 \
        --num_folds ${num_folds} \
        --use_fp16 False \
        --side ${side} \
        --use_fp16 True \
