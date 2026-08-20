#!/bin/bash
#
# Pretrain Quetzal on GEOM-Drugs across 4 GPUs on one node. Upstream utility:
# this project uses the released GEOM checkpoint and freezes it, so run this
# only to substitute a different prior.
#
# Submit from the repository root:  sbatch scripts/prior/train_geom_slurm.sh
#
# To continue past the 24h limit, resubmit with the same --name, or pass
# --resume_path=<ckpt>.
#
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --partition=compute_full_node
#SBATCH --time=24:00:00
#SBATCH --job-name=train
#SBATCH --output=slurm/%j.out

source ~/.bashrc
conda activate quetzal

wandb offline

srun --nodes=1 --tasks-per-node=4 python train.py --devices=4 --name=geom_run --dataset=geom --vis_every_n_epochs=1 --lr=2e-4 --sigma_data=2.5 --bsz=40 --packlen=512 --packdepth=10
