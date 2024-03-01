#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --requeue
#SBATCH --time=1:59:59
#SBATCH --mem=48GB
#SBATCH --output=logs/res_%j.out
#SBATCH --error=logs/err_%j.err
#SBATCH --job-name='diffuseq-replica'
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ad6489@nyu.edu

export WANDB_API_KEY=$(cat wandb_login.txt)
python --version
# overlay=/scratch/ad6489/pytorch-example/overlay_img
# img=/scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif
overlay=/scratch/ad6489/pytorch-example/overlay_img2
img=/scratch/work/public/singularity/cuda11.1-cudnn8-devel-ubuntu18.04.sif

singularity exec --nv \
	--overlay $overlay:ro \
	$img \
       	/bin/bash -c \
	"source /ext3/env.sh; torchrun --nproc_per_node=1 --master_port=12233 /scratch/ad6489/thesis/DiffuSeq/scripts/run_train.py \
	--diff_steps 2000 \
	--lr 0.0001 \
	--learning_steps 50000 \
	--save_interval 5000 \
	--seed 102 \
	--noise_schedule sqrt \
	--hidden_dim 256 \
	--bsz 2048 \
	--dataset iwslt14 \
	--data_dir /scratch/ad6489/thesis/DiffuSeq/datasets/iwslt14 \
	--vocab bert \
	--seq_len 256 \
	--schedule_sampler lossaware \
	--app '--use_fp16 True' \
	--notes test-iwslt14" \
	> log.out 2> log.err
	# --resume_checkpoint /scratch/ad6489/thesis/DiffuSeq/diffusion_models/diffuseq_qqp_h128_lr0.0001_t2000_sqrt_lossaware_seed102_test-qqp20231109-01:58:16/ema_0.9999_020000.pt \  

