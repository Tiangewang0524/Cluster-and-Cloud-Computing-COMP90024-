#!/bin/bash
#SBATCH -p physical
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=00:10:00

echo 'Run the task with 1 node and 8 cores'

module load Python/3.5.2-goolf-2015a
mpiexec python Instagram_rank.py