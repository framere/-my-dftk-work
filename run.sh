#!/bin/sh
#SBATCH -J run_test
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p andoria
#SBATCH --time=1-00:00:00

JULIA=/home/fmereto/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia

$JULIA He.jl
