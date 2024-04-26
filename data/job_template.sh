#!/usr/bin/bash

#SBATCH --job-name=21cm-{i_start}-{i_stop}
#SBATCH --output=%x-%j.out
#SBATCH --partition=genx
#SBATCH --constraint=icelake
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=7G
#SBATCH --time=7-00:00:00


module --force purge
source $HOME/mamba/bin/activate 21cm

hostname; pwd; date

time -p python run.py {i_start} {i_stop}

date
