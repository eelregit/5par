#!/usr/bin/bash

#SBATCH --job-name=cobaya
#SBATCH --output=%x.%j.out
#SBATCH --partition=ccm
#SBATCH --constraint=genoa
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=6
#SBATCH --mem=0
#SBATCH --mem-bind=none
#SBATCH --time=7-00:00:00


export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

source $HOME/ceph/universe/gomp/activate.sh


hostname; pwd; date

while true
do
  srun --mem=0 --time=00:15:00 cobaya run --resume --allow-changes cobaya.yaml
done

date
