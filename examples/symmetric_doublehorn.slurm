#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --time=72:00:00
#SBATCH --output=symDH.out
#SBATCH --error=symDH.err
#SBATCH --mem=10GB

# export OMP_NUM_THREADS=1

module load mpich-x86_64


#ALFALFA S/N definition
mpiexec -np 24 python model_asymmetry_parameterspace.py generate -rms -N 10000 -SN 4 104 1 -AA -opt Vsm10
mpiexec -np 24 python model_asymmetry_parameterspace.py measure doublehornAAVsm10_rms
python model_asymmetry_parameterspace.py statistics doublehornAAVsm10_rms

#higher smoothing
mpiexec -np 24 python model_asymmetry_parameterspace.py generate -rms -N 10000 -SN 4 104 1 -AA -Vs 50 -opt Vsm50
mpiexec -np 24 python model_asymmetry_parameterspace.py measure doublehornAAVsm50_rms
python model_asymmetry_parameterspace.py statistics doublehornAAVsm50_rms


#Peak/rms S/N definition
mpiexec -np 24 python model_asymmetry_parameterspace.py generate -rms -N 10000 -SN 4 104 1 -PN -opt Vsm10
mpiexec -np 24 python model_asymmetry_parameterspace.py measure doublehornPNVsm10_rms
python model_asymmetry_parameterspace.py statistics doublehornPNVsm10_rms

#higher smoothing
mpiexec -np 24 python model_asymmetry_parameterspace.py generate -rms -N 10000 -SN 4 104 1 -PN -Vs 50 -opt Vsm50
mpiexec -np 24 python model_asymmetry_parameterspace.py measure doublehornPNVsm50_rms
python model_asymmetry_parameterspace.py statistics doublehornPNVsm50_rms


