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





# def main1():

# 	dim = 2000
# 	incl = 50.
# 	model= 'FE'
# 	MHI = 1.e10
# 	Vdisp = 0
# 	dist = 150.e0
# 	rms_temp = -1
# 	Vres = 2.e0
# 	Vsm = 10
# 	if Vsm == 0:
# 		Vsm = Vres

# 	mjy_conv = 1.e3 / (2.356e5  * (dist ** 2.e0))
# 	Sint = mjy_conv * MHI 
		
# 	Vmin = -800.e0
# 	Vmax = 800.e0

# 	input_params = [incl, model,
# 					0, MHI, 1, 1.65, -1, -1 , 1, 1.65, -1, -1,
# 					0, 200, 0.164, 0.002, 200, 0.164, 0.002,
# 					Vdisp, dist, rms_temp, Vmin, Vmax, Vres, Vsm]		
# 	radius, costheta, R_opt = create_arrays(dim, input_params)
# 	obs_mom0, rad1d, input_HI = create_mom0(radius, costheta, input_params, R_opt)
# 	obs_mom1, input_RC  = create_mom1(radius, costheta, rad1d, input_params, R_opt)
# 	vel_bins, base_spectrum, Sint = hi_spectra(obs_mom0, obs_mom1, input_params)

# 	spectrum = smooth_spectra(vel_bins, base_spectrum, input_params)

# 	Peaklocs = locate_peaks(spectrum)
# 	Peaks = spectrum[Peaklocs]
# 	width_full = locate_width(spectrum, Peaks, 0.2e0)
# 	width = (width_full[1] - width_full[0]) * Vres
# 	Sint, Afr = areal_asymmetry(spectrum, width_full, Vres)

# 	fig = plt.figure(figsize = (12,10))
# 	gs  = gridspec.GridSpec(1, 3, hspace=0, wspace = 0,left = 0.09, right = 0.99, top=0.99, bottom = 0.09)
# 	SN50_ax = fig.add_subplot(gs[0,0])
# 	SN20_ax = fig.add_subplot(gs[0,1],sharex = SN50_ax,sharey = SN50_ax)
# 	SN10_ax = fig.add_subplot(gs[0,2],sharex = SN50_ax,sharey = SN50_ax)

# 	# SN50dist_ax = fig.add_subplot(gs[0,1::])
# 	# SN20dist_ax = fig.add_subplot(gs[1,1::])
# 	# SN10dist_ax = fig.add_subplot(gs[2,1::])


# 	axes = [SN50_ax,SN20_ax,SN10_ax]

# 	SN_input = [50,20,10]
# 	Afr_thresh = [1.05,1.12,1.3]

# 	for ssnn in range(len(SN_input)):
# 		SN = SN_input[ssnn]
# 		RMS_sm = rms_from_StN(SN, Sint[2], width, Vsm)
# 		RMS_input = RMS_sm * np.sqrt(int(Vsm / Vres))

# 		input_params[21] = RMS_input
# 		ax = axes[ssnn]
# 		obs_Afr = -1
# 		while(obs_Afr < Afr_thresh[ssnn]):
# 			# print('........')
# 			obs_spectrum, noise_array = add_noise(base_spectrum, input_params)
# 			# obs_Sint, obs_Afr = areal_asymmetry(obs_spectrum,width_full,Vres)
# 			# print(obs_Afr)
# 			obs_spectrum = smooth_spectra(vel_bins, obs_spectrum, input_params)
# 			obs_Sint, obs_Afr = areal_asymmetry(obs_spectrum,width_full,Vres)
# 			# print(obs_Afr)
# 			# print('........')

		

# 		minvel = vel_bins[int(np.floor(width_full[0]))] + Vres * (width_full[0]- int(np.floor(width_full[0])))
# 		maxvel = vel_bins[int(np.floor(width_full[1]))] + Vres * (width_full[1] - int(np.floor(width_full[1])))
# 		midvel = 0.5e0*(minvel + maxvel)

# 		ax.plot(vel_bins,obs_spectrum, color = 'Black',ls = '--')
# 		# ax.plot(vel_bins,spectrum, color = 'Red')
# 		ax.plot([minvel,minvel],[0,np.nanmax(spectrum)], color = 'Blue',ls = '-')
# 		ax.plot([maxvel,maxvel],[0,np.nanmax(spectrum)], color = 'Blue',ls = '-')
# 		ax.plot([midvel,midvel],[0,np.nanmax(spectrum)], color = 'Blue',ls = '-')
# 		ax.text(0.1,0.8,'S/N = {SN}'.format(SN=SN),transform=ax.transAxes)
# 		ax.text(0.1,0.75,'S$_{{int,L}}$ = {SL:.2f}'.format(SL=obs_Sint[0]),transform=ax.transAxes)
# 		ax.text(0.1,0.7,'S$_{{int,R}}$ = {SR:.2f}'.format(SR=obs_Sint[1]),transform=ax.transAxes)
# 		ax.text(0.1,0.65,'A$_{{fr}}$ = {Afr:.2f}'.format(Afr = obs_Afr),transform=ax.transAxes)


# 		# SN50_dist = np.genfromtxt('/media/data/model_parameterspace/doublehornAAVsm10_Afr1.00_rms/SN50_measured.dat')[1::,5]
# 		# SN20_dist = np.genfromtxt('/media/data/model_parameterspace/doublehornAAVsm10_Afr1.00_rms/SN20_measured.dat')[1::,5]
# 		# SN10_dist = np.genfromtxt('/media/data/model_parameterspace/doublehornAAVsm10_Afr1.00_rms/SN10_measured.dat')[1::,5]

# 		# SN50dist_ax.hist(SN50_dist, bins = np.arange(1,1.5,0.1), histtype='step',color='Black',fill=False)
# 		# SN20dist_ax.hist(SN20_dist, bins = np.arange(1,1.5,0.1), histtype='step',color='Black',fill=False)
# 		# SN10dist_ax.hist(SN10_dist, bins = np.arange(1,1.5,0.1), histtype='step',color='Black',fill=False)

# 	plt.show()


def main2():
	
	
	SN50_spec = '/media/data/model_parameterspace/doublehornAAVsm10_Afr1.00_rms/SN50_spectra.dat'
	SN20_spec = '/media/data/model_parameterspace/doublehornAAVsm10_Afr1.00_rms/SN20_spectra.dat'
	SN10_spec = '/media/data/model_parameterspace/doublehornAAVsm10_Afr1.00_rms/SN10_spectra.dat'

	file_list = [SN50_spec,SN20_spec,SN10_spec]

	A_measurements = np.zeros([10000,3,2])

	for ff in range(len(file_list)):
		file = file_list[ff]
		f = open(file)
		for line in f:
			split = line.split(' ')
			if split[1] == 'rms':
				rms = float(split[3])
			if split[1] == 'Vsm':
				Vsm = float(split[3])
				break
		f.close()

		spectra = np.loadtxt(file,ndmin=2)

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

	fig.savefig('./logA_Afr_dist.png')
	plt.show()



if __name__ == '__main__':
	# main1()
	main2()