#!/usr/bin/env bash
#SBATCH -n 1
#SBATCH -c 32
#SBATCH -t 01:00:00
#SBATCH -p gpu
#SBATCH --gpus=1          
#SBATCH -o generate_%j.out


python prior/generate.py \
    --save_dir /Users/code/mt_tree/dataset --dataset_type mothernet\
    --padding False \
    --device cpu --n_jobs 1 \
    --batch_size 256 \
    --max_seq_len 1024 \
    --max_features 100 \
    --prior_type prior_bag \
    --num_batches 1 \
