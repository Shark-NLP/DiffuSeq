#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4
#SBATCH --requeue
#SBATCH --time=5:00:00
#SBATCH --mem=360GB
#SBATCH --output=logs/res_%j.out
#SBATCH --error=logs/err_%j.err
#SBATCH --job-name='diffuseq-replica-decode'
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ad6489@nyu.edu

export WANDB_API_KEY=$(cat wandb_login.txt)
overlay=/scratch/ad6489/pytorch-example/overlay
img=/scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif

singularity exec --nv \
	--overlay $overlay:ro \
	$img \
       	/bin/bash -c \
	"source /ext3/env.sh; CUDA_VISIBLE_DEVICES=4 python -u /scratch/ad6489/thesis/DiffuSeq/scripts/run_decode_solver.py \
	--model_dir diffusion_models/diffuseq_qqp_h128_lr0.0001_t2000_sqrt_lossaware_seed102_test-qqp20231015-19:22:30 \
	--seed 123 \
	--bsz 100 \
	--step 10 \
	--split test"
	> log_dec.out 2> log_dec.err
