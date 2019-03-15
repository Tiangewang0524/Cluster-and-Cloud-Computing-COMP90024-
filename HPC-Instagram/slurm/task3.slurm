#!/bin/bash
#SBATCH -p physical
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --time=00:10:00

echo 'Run the task with 2 node and 8 core'

module load Python/3.5.2-goolf-2015a
mpiexec python Instagram_rank.py