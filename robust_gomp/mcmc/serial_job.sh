#!/usr/bin/bash

#SBATCH --job-name=serial
#SBATCH --output=%x.%j.out
#SBATCH --partition=ccm
#SBATCH --constraint=genoa
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=0
#SBATCH --mem-bind=none
#SBATCH --time=7-00:00:00


export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

source $HOME/ceph/universe/gomp/activate.sh


hostname; pwd; date

#srun --exact -n 1 ...

while true
do
  srun -n 1 --mem=0 --time=08:00:00 cobaya run --resume --allow-changes --output cmb_dw_bao_LCDM/serial cmb_dw_bao_LCDM/serial_covmat.yaml
done > cmb_dw_bao_LCDM/serial.$SLURM_JOB_ID.out 2>&1 &
while true
do
  srun -n 1 --mem=0 --time=08:00:00 cobaya run --resume --allow-changes --output cmb_dw_bao_Mnu/serial cmb_dw_bao_Mnu/serial_covmat.yaml
done > cmb_dw_bao_Mnu/serial.$SLURM_JOB_ID.out 2>&1 &
while true
do
  srun -n 1 --mem=0 --time=08:00:00 cobaya run --resume --allow-changes --output cmb_dw_bao_w0wa/serial cmb_dw_bao_w0wa/serial_covmat.yaml
done > cmb_dw_bao_w0wa/serial.$SLURM_JOB_ID.out 2>&1 &
while true
do
  srun -n 1 --mem=0 --time=08:00:00 cobaya run --resume --allow-changes --output cmb_dw_bao_w0waMnu/serial cmb_dw_bao_w0waMnu/serial_covmat.yaml
done > cmb_dw_bao_w0waMnu/serial.$SLURM_JOB_ID.out 2>&1 &
while true
do
  srun -n 1 --mem=0 --time=08:00:00 cobaya run --resume --allow-changes --output cmb_dw_LCDM/serial cmb_dw_LCDM/serial_covmat.yaml
done > cmb_dw_LCDM/serial.$SLURM_JOB_ID.out 2>&1 &
while true
do
  srun -n 1 --mem=0 --time=08:00:00 cobaya run --resume --allow-changes --output cmb_dw_lens_bao_LCDM/serial cmb_dw_lens_bao_LCDM/serial_covmat.yaml
done > cmb_dw_lens_bao_LCDM/serial.$SLURM_JOB_ID.out 2>&1 &
while true
do
  srun -n 1 --mem=0 --time=08:00:00 cobaya run --resume --allow-changes --output cmb_dw_lens_bao_Mnu/serial cmb_dw_lens_bao_Mnu/serial_covmat.yaml
done > cmb_dw_lens_bao_Mnu/serial.$SLURM_JOB_ID.out 2>&1 &
while true
do
  srun -n 1 --mem=0 --time=08:00:00 cobaya run --resume --allow-changes --output cmb_dw_lens_bao_w0wa/serial cmb_dw_lens_bao_w0wa/serial_covmat.yaml
done > cmb_dw_lens_bao_w0wa/serial.$SLURM_JOB_ID.out 2>&1 &
while true
do
  srun -n 1 --mem=0 --time=08:00:00 cobaya run --resume --allow-changes --output cmb_dw_lens_bao_w0waMnu/serial cmb_dw_lens_bao_w0waMnu/serial_covmat.yaml
done > cmb_dw_lens_bao_w0waMnu/serial.$SLURM_JOB_ID.out 2>&1 &
while true
do
  srun -n 1 --mem=0 --time=08:00:00 cobaya run --resume --allow-changes --output cmb_dw_lens_LCDM/serial cmb_dw_lens_LCDM/serial_covmat.yaml
done > cmb_dw_lens_LCDM/serial.$SLURM_JOB_ID.out 2>&1 &
while true
do
  srun -n 1 --mem=0 --time=08:00:00 cobaya run --resume --allow-changes --output cmb_dw_lens_Mnu/serial cmb_dw_lens_Mnu/serial_covmat.yaml
done > cmb_dw_lens_Mnu/serial.$SLURM_JOB_ID.out 2>&1 &
while true
do
  srun -n 1 --mem=0 --time=08:00:00 cobaya run --resume --allow-changes --output cmb_dw_lens_w0wa/serial cmb_dw_lens_w0wa/serial_covmat.yaml
done > cmb_dw_lens_w0wa/serial.$SLURM_JOB_ID.out 2>&1 &
while true
do
  srun -n 1 --mem=0 --time=08:00:00 cobaya run --resume --allow-changes --output cmb_dw_lens_w0waMnu/serial cmb_dw_lens_w0waMnu/serial_covmat.yaml
done > cmb_dw_lens_w0waMnu/serial.$SLURM_JOB_ID.out 2>&1 &
while true
do
  srun -n 1 --mem=0 --time=08:00:00 cobaya run --resume --allow-changes --output cmb_dw_Mnu/serial cmb_dw_Mnu/serial_covmat.yaml
done > cmb_dw_Mnu/serial.$SLURM_JOB_ID.out 2>&1 &
while true
do
  srun -n 1 --mem=0 --time=08:00:00 cobaya run --resume --allow-changes --output cmb_dw_w0wa/serial cmb_dw_w0wa/serial_covmat.yaml
done > cmb_dw_w0wa/serial.$SLURM_JOB_ID.out 2>&1 &
while true
do
  srun -n 1 --mem=0 --time=08:00:00 cobaya run --resume --allow-changes --output cmb_dw_w0waMnu/serial cmb_dw_w0waMnu/serial_covmat.yaml
done > cmb_dw_w0waMnu/serial.$SLURM_JOB_ID.out 2>&1 &

wait

date
