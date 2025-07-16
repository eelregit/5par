#!/usr/bin/bash

#SBATCH --job-name=cobaya
#SBATCH --output=%x-%j.out
#SBATCH --partition=deimos
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=6
#SBATCH --mem=0
#SBATCH --time=7-00:00:00



# NOTE symlinked into each job directory and should be submitted there


export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

source $HOME/Projects/gomp/activate.sh


hostname; pwd; date

[[ -f cobaya.covmat ]] && SCRIPT=cobaya_covmat.yaml || SCRIPT=cobaya.yaml

time mpirun -n $SLURM_NTASKS -ppn $SLURM_CPUS_PER_TASK cobaya run --resume $SCRIPT

date
