#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --time=72:00:00
#SBATCH --output=asymG.out
#SBATCH --error=asymG.err
#SBATCH --mem=10GB

# export OMP_NUM_THREADS=1

module load mpich-x86_64


#Afr=1.1, default AA S/N ratio and Vsm = 10
mpiexec -np 24 python model_asymmetry_parameterspace.py generate -rms -R 200 0.164 0.002 230 0.16 0.003 -N 10000 -SN 4 104 1
mpiexec -np 24 python model_asymmetry_parameterspace.py measure doublehornAA_Afr1.10_rms
python model_asymmetry_parameterspace.py statistics doublehornAA_Afr1.10_rms

#Afr=1.25, default AA S/N ratio and Vsm = 10
mpiexec -np 24 python model_asymmetry_parameterspace.py generate -rms -R 200 0.164 0.002 270 0.15 0.006 -N 10000 -SN 4 104 1
mpiexec -np 24 python model_asymmetry_parameterspace.py measure doublehornAA_Afr1.25_rms
python model_asymmetry_parameterspace.py statistics doublehornAA_Afr1.25_rms

#Afr=1.5, default AA S/N ratio and Vsm = 10
mpiexec -np 24 python model_asymmetry_parameterspace.py generate -rms -R 200 0.164 0.002 350 0.136 0.01 -N 10000 -SN 4 104 1
mpiexec -np 24 python model_asymmetry_parameterspace.py measure doublehornAA_Afr1.50_rms
python model_asymmetry_parameterspace.py statistics doublehornAA_Afr1.50_rms




