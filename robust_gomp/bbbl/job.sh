#!/usr/bin/bash

#SBATCH --job-name=bbbl
#SBATCH --output=%x-%j.out
#SBATCH --partition=gpu
#SBATCH --constraint=a100
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-cpu=15G
#SBATCH --time=7-00:00:00


source $HOME/miniforge3/bin/activate

hostname; pwd; date

for kind in MC LSS Grid QMC
do
  for dens in 1e-{3..5}
  do
    for b in 0.5 1 2
    do
      python bubbles.py 512 $kind $dens $b
    done
  done
done

date
