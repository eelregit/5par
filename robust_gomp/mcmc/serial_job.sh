#!/usr/bin/bash

#SBATCH --job-name=serial
#SBATCH --output=%x.%j.out
#SBATCH --partition=ccm
#SBATCH --constraint=genoa
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=0
#SBATCH --mem-bind=none
#SBATCH --time=7-00:00:00


export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

source $HOME/ceph/universe/gomp/activate.sh


hostname; pwd; date

#srun --exact -n 1 ...

srun -n 1 --mem=0 cobaya run --resume --output cmb_dw_bao_w0waMnu/serial cmb_dw_bao_w0waMnu/serial_covmat.yaml > cmb_dw_bao_w0waMnu/serial.$SLURM_JOB_ID.out 2>&1 &
srun -n 1 --mem=0 cobaya run --resume --output cmb_dw_lens_bao_w0waMnu/serial cmb_dw_lens_bao_w0waMnu/serial_covmat.yaml > cmb_dw_lens_bao_w0waMnu/serial.$SLURM_JOB_ID.out 2>&1 &
srun -n 1 --mem=0 cobaya run --resume --output cmb_dw_lens_w0waMnu/serial cmb_dw_lens_w0waMnu/serial_covmat.yaml > cmb_dw_lens_w0waMnu/serial.$SLURM_JOB_ID.out 2>&1 &
srun -n 1 --mem=0 cobaya run --resume --output cmb_dw_w0waMnu/serial cmb_dw_w0waMnu/serial_covmat.yaml > cmb_dw_w0waMnu/serial.$SLURM_JOB_ID.out 2>&1 &
wait

date
