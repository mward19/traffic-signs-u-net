#!/bin/bash

#SBATCH --time=3:00:00   # walltime
#SBATCH --ntasks=8   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=16384M   # memory per CPU core
#SBATCH -J "Train on traffic signs"   # job name
#SBATCH --mail-user=matthew.merrill.ward@gmail.com   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
python train.py