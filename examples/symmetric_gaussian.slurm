#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --time=72:00:00
#SBATCH --output=symG.out
#SBATCH --error=symG.err
#SBATCH --mem=10GB

# export OMP_NUM_THREADS=1

module load mpich-x86_64


#ALFALFA S/N definition
mpiexec -np 24 python model_asymmetry_parameterspace.py generate -rms -G 90 -N 10000 -SN 4 104 1 -AA -opt Vsm10
mpiexec -np 24 python model_asymmetry_parameterspace.py measure gaussianAAVsm10_rms
python model_asymmetry_parameterspace.py statistics gaussianAAVsm10_rms

#higher smoothing
mpiexec -np 24 python model_asymmetry_parameterspace.py generate -rms -G 90 -N 10000 -SN 4 104 1 -AA -Vs 50 -opt Vsm10
mpiexec -np 24 python model_asymmetry_parameterspace.py measure gaussianAAVsm50_rms
python model_asymmetry_parameterspace.py statistics gaussianAAVsm50_rms


#Peak/rms S/N definition
mpiexec -np 24 python model_asymmetry_parameterspace.py generate -rms -G 90 -N 10000 -SN 4 104 1 -PN -opt Vsm10
mpiexec -np 24 python model_asymmetry_parameterspace.py measure gaussianPNVsm10_rms
python model_asymmetry_parameterspace.py statistics gaussianPNVsm10_rms

#higher smoothing
mpiexec -np 24 python model_asymmetry_parameterspace.py generate -rms -G 90 -N 10000 -SN 4 104 1 -PN -Vs 50 -opt Vsm10
mpiexec -np 24 python model_asymmetry_parameterspace.py measure gaussianPNVsm50_rms
python model_asymmetry_parameterspace.py statistics gaussianPNVsm50_rms
