#!/bin/bash

# GPU config
USE_TPU=False

# Experiment (data/checkpoint/directory) config
DATA_PATH=datasets/AES_HD/
DATASET=AES_HD
CKP_DIR=checkpoints/aes_hd
WARM_START=False
RESULT_PATH=results

# Optimization config
LEARNING_RATE=2.5e-4
CLIP=0.25
MIN_LR_RATIO=0.004
WARMUP_STEPS=0

# Training config
TRAIN_BSZ=256
EVAL_BSZ=16
TRAIN_STEPS=30000
ITERATIONS=10000
SAVE_STEPS=10000

# Model config
N_LAYER=2
D_MODEL=128
N_HEAD=2
D_HEAD=64
D_INNER=256
DROPOUT=0.05
DROPATT=0.05
CONV_KERNEL_SIZE=11
POOL_SIZE=2
CLAMP_LEN=700
UNTIE_R=True
SMOOTH_POS_EMB=False
UNTIE_POS_EMB=True

# Parameter initialization
INIT=normal
INIT_STD=0.02
INIT_RANGE=0.1

# Evaluation config
MAX_EVAL_BATCH=10


if [[ $1 == 'train' ]]; then
    python train_trans.py \
        --use_tpu=${USE_TPU} \
        --data_path=${DATA_PATH} \
	--dataset=${DATASET} \
        --checkpoint_dir=${CKP_DIR} \
        --warm_start=${WARM_START} \
        --result_path=${RESULT_PATH} \
        --learning_rate=${LEARNING_RATE} \
        --clip=${CLIP} \
        --min_lr_ratio=${MIN_LR_RATIO} \
        --warmup_steps=${WARMUP_STEPS} \
        --train_batch_size=${TRAIN_BSZ} \
        --eval_batch_size=${EVAL_BSZ} \
        --train_steps=${TRAIN_STEPS} \
        --iterations=${ITERATIONS} \
        --save_steps=${SAVE_STEPS} \
        --n_layer=${N_LAYER} \
        --d_model=${D_MODEL} \
        --n_head=${N_HEAD} \
        --d_head=${D_HEAD} \
        --d_inner=${D_INNER} \
        --dropout=${DROPOUT} \
        --dropatt=${DROPATT} \
        --conv_kernel_size=${CONV_KERNEL_SIZE} \
        --pool_size=${POOL_SIZE} \
        --clamp_len=${CLAMP_LEN} \
        --untie_r=${UNTIE_R} \
	--smooth_pos_emb=${SMOOTH_POS_EMB} \
	--untie_pos_emb=${UNTIE_POS_EMB} \
        --init=${INIT} \
        --init_std=${INIT_STD} \
        --init_range=${INIT_RANGE} \
        --max_eval_batch=${MAX_EVAL_BATCH} \
	--do_train=True
elif [[ $1 == 'test' ]]; then
    python train_trans.py \
        --use_tpu=${USE_TPU} \
        --data_path=${DATA_PATH} \
	--dataset=${DATASET} \
        --checkpoint_dir=${CKP_DIR} \
        --warm_start=${WARM_START} \
        --result_path=${RESULT_PATH} \
        --learning_rate=${LEARNING_RATE} \
        --clip=${CLIP} \
        --min_lr_ratio=${MIN_LR_RATIO} \
        --warmup_steps=${WARMUP_STEPS} \
        --train_batch_size=${TRAIN_BSZ} \
        --eval_batch_size=${EVAL_BSZ} \
        --train_steps=${TRAIN_STEPS} \
        --iterations=${ITERATIONS} \
        --save_steps=${SAVE_STEPS} \
        --n_layer=${N_LAYER} \
        --d_model=${D_MODEL} \
        --n_head=${N_HEAD} \
        --d_head=${D_HEAD} \
        --d_inner=${D_INNER} \
        --dropout=${DROPOUT} \
        --dropatt=${DROPATT} \
        --conv_kernel_size=${CONV_KERNEL_SIZE} \
        --pool_size=${POOL_SIZE} \
        --clamp_len=${CLAMP_LEN} \
        --untie_r=${UNTIE_R} \
	--smooth_pos_emb=${SMOOTH_POS_EMB} \
	--untie_pos_emb=${UNTIE_POS_EMB} \
        --init=${INIT} \
        --init_std=${INIT_STD} \
        --init_range=${INIT_RANGE} \
        --max_eval_batch=${MAX_EVAL_BATCH} \
	--do_train=False
else
    echo "unknown argument 1"
fi
