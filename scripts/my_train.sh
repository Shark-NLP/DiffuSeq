#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:rtx8000:4
#SBATCH --requeue
#SBATCH --time=1-23:59:59
#SBATCH --mem=360GB
#SBATCH --output=logs/res_%j.out
#SBATCH --error=logs/err_%j.err
#SBATCH --job-name='diffuseq-replica'
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ad6489@nyu.edu

export WANDB_API_KEY=$(cat wandb_login.txt)
overlay=/scratch/ad6489/pytorch-example/overlay_img
img=/scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif

singularity exec --nv \
	--overlay $overlay:ro \
	$img \
       	/bin/bash -c \
	"source /ext3/env.sh; torchrun --nproc_per_node=4 --master_port=12233 /scratch/ad6489/thesis/DiffuSeq/scripts/run_train.py \
	--diff_steps 2000 \
	--lr 0.001 \
	--learning_steps 50000 \
	--save_interval 10000 \
	--seed 102 \
	--noise_schedule sqrt \
	--hidden_dim 128 \
	--bsz 2048 \
	--dataset qqp \
	--data_dir /scratch/ad6489/thesis/DiffuSeq/datasets/QQP \
	--vocab bert \
	--seq_len 128 \
	--schedule_sampler lossaware \
	--notes test-qqp" \
	> log.out 2> log.err
