#!/usr/bin/env bash
#SBATCH -A
#SBATCH -n 1
#SBATCH -c 32
#SBATCH -t 00:40:00
#SBATCH -p gpu
#SBATCH --gpus=1          
#SBATCH --array=1-30
#SBATCH -o gen_data_%A_%a.out

cd mt_tree

python prior/generate.py \
    --dataset_type tabiclv2 \
    --save_dir /Users/code/mt_tree/dataset \
    --dataset_config_path tabiclv2/graph_lib/config/stage1.json \
    --num_batches 10 \
    --n_jobs 2 \
    # --resume_from 20000 \
    # --num_batches 20000 \
