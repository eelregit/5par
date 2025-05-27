#!/usr/bin/bash

#SBATCH --job-name=cobaya
#SBATCH --output=%x.%j.out
#SBATCH --partition=ccm
#SBATCH --constraint=genoa
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=6
#SBATCH --mem=0
#SBATCH --time=7-00:00:00


# NOTE unlike serial_job.sh, this is symlinked into each job directory
# and should be submitted there!


export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

source $HOME/ceph/universe/gomp/activate.sh


hostname; pwd; date

srun --mem=0 cobaya run --resume cobaya.yaml

date
