#!/usr/bin/bash

#SBATCH --job-name=gomp0
#SBATCH --output=%x-%j.out
#SBATCH --partition=ccm
#SBATCH --constraint=icelake
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=15G
#SBATCH --time=7-00:00:00


export OMP_NUM_THREADS=4

source $HOME/ceph/cobaya/activate.sh


hostname; pwd; date

time srun cobaya run gomp_0.input.yaml

date
