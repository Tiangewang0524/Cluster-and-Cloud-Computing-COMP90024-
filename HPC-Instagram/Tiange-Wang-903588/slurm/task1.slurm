#!/bin/bash
#SBATCH -p physical
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:10:00

echo 'Run the task with 1 node and 1 core'

module load Python/3.5.2-goolf-2015a
mpiexec python Instagram_rank.py