#!/usr/bin/bash

#SBATCH --job-name=cobaya
#SBATCH --output=%x.%j.out
#SBATCH --partition=ccm
#SBATCH --constraint=genoa
#SBATCH --ntasks=24
#SBATCH --cpus-per-task=4
#SBATCH --mem=0
#SBATCH --mem-bind=none
#SBATCH --time=7-00:00:00


export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

source $HOME/ceph/universe/gomp/activate.sh


hostname; pwd; date

srun --mem=0 cobaya run --resume --allow-changes cobaya.yaml

date
