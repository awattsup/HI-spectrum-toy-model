import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import glob 
from matplotlib.lines import Line2D

from mpi4py import MPI
from astropy.table import Table
from astropy.convolution import convolve, Box1DKernel
from scipy.stats import norm, mode
from scipy.special import erf

	

basedir = '/media/data/models/HI_spectrum_toy_model/models/doublehornAACsm10.0_Afr1.00_rms/measurements/'

SN50_measurements = '{basedir}SN50_SN_Afr_measured.dat'
SN20_measurements = '{basedir}SN20_SN_Afr_measured.dat'
SN10_measurements = '{basedir}SN10_SN_Afr_measured.dat'

file_list = [SN50_smeasurements,SN20_smeasurements,SN10_smeasurements]

A_measurements = np.zeros([10000,3,2])

for ff in range(len(file_list)):
	

	measurements = np.genfromtxt(file)
	Afr_w20 = measurements[:,8]
	A_measurements[:,ff,0] = Afr_w20
	Afr_w20[0:-1:2] = 1./Afr_w20[0:-1:2]
	A_measurements[:,ff,1] = Afr_w20




	vel_bins = spectra[:,0]
	spectrum = spectra[:,1]
	Vres = np.abs(vel_bins[1] - vel_bins[0])

	Peaklocs = locate_peaks(spectrum)
	Peaks = spectrum[Peaklocs]

	width_20 = locate_width(spectrum, Peaks, 0.2e0)

	w20 = (width_20[1] - width_20[0]) * Vres

	Sint_noiseless_w20, Afr_noiseless_w20 = areal_asymmetry(spectrum, width_20, Vres)

	for run in range(len(spectra[0]) - 2):
		obs_spectrum = spectra[:, run + 2]

		Sint_w20, Afr_w20 = areal_asymmetry(obs_spectrum, width_20, Vres)
		
		A_w20 = Sint_w20[0] / Sint_w20[1]

		A_measurements[run,ff,0] = A_w20
		A_measurements[run,ff,1] = Afr_w20
		
	print(np.mean(A_measurements[:,ff,1]),mode(A_measurements[:,ff,1]))


fig = plt.figure(figsize = (15,9))
gs  = gridspec.GridSpec(1, 2, left = 0.1, right = 0.99, top=0.97, bottom = 0.11)
Aax = fig.add_subplot(gs[0,0])
Afrax = fig.add_subplot(gs[0,1])

Aax.set_xlim([-0.25,0.25])
Afrax.set_xlim([0.99,1.85])

Aax.tick_params( direction = 'in', labelsize = 25,length=8,width=1.5)
Aax.tick_params( direction = 'in', labelsize = 25,length=8,width=1.5)

Afrax.tick_params( direction = 'in', labelsize = 25,length=8,width=1.5)
Afrax.tick_params( direction = 'in', labelsize = 25,length=8,width=1.5)


Aax.hist(np.log10(A_measurements[:,0,0]),bins=41,histtype= 'step',fill=False,density=True
	, lw = 3, color='Black', ls = ':')
Aax.hist(np.log10(A_measurements[:,1,0]),bins=41,histtype= 'step',fill=False,density=True
	, lw = 3, color='Blue', ls = '--')
Aax.hist(np.log10(A_measurements[:,2,0]),bins=41,histtype= 'step',fill=False,density=True
	, lw = 3, color='Red', ls = '-')

Afrax.hist(A_measurements[:,0,1],bins=np.arange(1,1.9,0.01),histtype= 'step',cumulative=True,fill=False,density=True
	, lw = 3, color='Black', ls = ':')
Afrax.hist(A_measurements[:,1,1],bins=np.arange(1,1.9,0.01),histtype= 'step',cumulative=True,fill=False,density=True
	, lw = 3, color='Blue', ls = '--')
Afrax.hist(A_measurements[:,2,1],bins=np.arange(1,1.9,0.01),histtype= 'step',cumulative=True,fill=False,density=True
	, lw = 3, color='Red', ls = '-')



Afrax.set_xlabel('Asymmetry measure A$_{fr}$',fontsize=27)
Aax.set_xlabel('log$_{10}(A)$',fontsize=27)

Afrax.set_ylabel('Cumulative Hisogram',fontsize=27)
Aax.set_ylabel('Histogram Density',fontsize=27)



legend = [Line2D([0], [0], color = 'Black',ls = ':', linewidth = 3),
			Line2D([0], [0], color = 'Blue',ls = '--', linewidth = 3),
			Line2D([0], [0], color = 'Red',ls = '-', linewidth = 3)]

Aax.legend(legend,['S/N = 50','S/N = 20','S/N = 10'],fontsize=24)

fig.savefig('./figures/logA_Afr_dist.png')
plt.show()
