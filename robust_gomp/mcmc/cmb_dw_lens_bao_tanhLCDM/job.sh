#!/usr/bin/bash

#SBATCH --job-name=cobaya
#SBATCH --output=%x-%j.out
#SBATCH --partition=deimos
#SBATCH --mem-per-cpu=7200M
#SBATCH --time=7-00:00:00


export OMP_NUM_THREADS=4

source $HOME/Projects/gomp/activate.sh


hostname; pwd; date

time mpirun -n 16 -ppn 4 cobaya run --resume cobaya.yaml

date
