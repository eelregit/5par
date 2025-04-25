#!/usr/bin/bash

#SBATCH --job-name=cobaya
#SBATCH --output=%x-%j.out
#SBATCH --partition=deimos


export OMP_NUM_THREADS=4

source $HOME/Projects/gomp/activate.sh


hostname; pwd; date

time mpirun -n 16 -ppn 4 cobaya run cobaya.yaml

date
