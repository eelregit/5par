#!/usr/bin/env python

#SBATCH --job-name=symb
#SBATCH --output=%x-%j.out
#SBATCH --partition=ccm
#SBATCH --constraint=genoa
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=7-00:00:00


import os
import sys

import numpy as np
import sympy
import pysr


num_nodes = int(os.environ['SLURM_NNODES'])
num_cores = int(os.environ['SLURM_CPUS_ON_NODE'])
print(f'{num_cores} cores on each of {num_nodes} nodes', flush=True)
num_cores *= num_nodes


var_names = ['s8', 'ns', 'h', 'Ob', 'Om', 'zt', 'Tv', 'LX']
var_cols = [2, 3, 4, 5, 6, 7, 8, 9]
var_edge = np.loadtxt('../xHI.txt', dtype='f4', usecols=var_cols)
var_core = np.loadtxt('../xHI_core.txt', dtype='f4', usecols=var_cols)
var = np.concatenate((var_edge, var_core), axis=0)
num_sim_half = 512
num_sim = 2 * num_sim_half
num_sim_qtr = num_sim_half // 2
num_a = 127
var = var.reshape(num_sim, num_a, len(var_cols))
var = var[:, 0, :]  # ignore the time dimension

tilt = np.loadtxt('../pivottilt_6.txt', dtype='f4', usecols=1)

#var_train = var.reshape(2, num_sim_half, len(var_cols))[:, :num_sim_qtr].reshape(2 * num_sim_qtr, -1)
#tilt_train = tilt.reshape(2, num_sim_half)[:, :num_sim_qtr].ravel()
var_train = var
tilt_train = tilt


kwargs = dict(
    # Search Space, Complexity, & Objective
    binary_operators=['+', '-', '*', '/', '^'],
    unary_operators=['exp', 'log',],
    #complexity_of_constants=0.2,
    #complexity_of_variables=0.2,
    #constraints={'^':(-1, 4), 'exp':4, 'log':4},
    #nested_constraints={
    #    'exp':{'exp':0, 'log':1},
    #    'log':{'exp':1, 'log':0},
    #},
    maxsize=54,
    #maxdepth=8,
    #warmup_maxsize_by=1e-2,
    parsimony=1e-3,
    adaptive_parsimony_scaling=1000,

    # Search Size
    niterations=40000,
    early_stop_condition=('stop_if(loss, complexity) = loss < 1e-5 && complexity < 21'),
    populations=num_cores*4,
    population_size=49,
    ncyclesperiteration=10000,

    # Mutations
    weight_simplify=0.01,
    weight_optimize=0.001,

    # Performance & Parallelization
    procs=num_cores,
    cluster_manager='slurm',
    #batching=True,
    #batch_size=100,
    turbo=True,

    # Monitoring
    verbosity=1,
    print_precision=2,
    progress=False,
)

try:
    model = pysr.PySRRegressor.from_file(sys.argv[1], warm_start=True, **kwargs)
    model.refresh()
except IndexError:
    model = pysr.PySRRegressor(**kwargs)

model.fit(var_train, tilt_train, variable_names=var_names)
#pysr.sr.Main.eval('flush(stdout)')
