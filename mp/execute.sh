#!/bin/bash -l
#SBATCH --job-name="vgg19_micro"
#SBATCH -D .
#SBATCH --gres=gpu:2
#SBATCH --output=./logs/logs_%j.out
#SBATCH --error=./logs/logs_%j.err
#SBATCH --nodelist=dolphin,shark
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --time=120:30:00
#SBATCH --partition=part0
#SBATCH --wait-all-nodes=1

srun pwd 
srun hostname
srun whoami

srun bash -c 'date +"%Y-%m-%d %H:%M:%S.%N"'

source /home/ohs/anaconda3/etc/profile.d/conda.sh
conda activate torch110

srun bash -c 'echo SLURM_NTASKS=$SLURM_NTASKS SLURM_NODEID=$SLURM_NODEID, SLURM_LOCALID=$SLURM_LOCALID, SLURM_PROCID=$SLURM_PROCID'

srun python -V

srun python parallel.py
