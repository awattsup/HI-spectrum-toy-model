"""
HI spectra model generation, measurement, and visualisation.

...

Self-contained suite of functions for generating, measuring,
calculating statistics and plotting model HI spectra. This python 
file can be run directly from the command line and uses argparse
to specify which function to run (e.g. model generation, plotting)
and if desired, include user defined inputs.  


...

Author
Adam B. Watts; November 2018
International Centre for Radio Astronomy Research
The University of Western Australia
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import glob 

from mpi4py import MPI
from astropy.table import Table
from astropy.convolution import convolve, Box1DKernel
from scipy.stats import norm, mode
from scipy.special import erf

def generate(args):	
	"""
	Generates model HI spectrum and adds noise realisations

    Parameters
    ----------
    args : list
        List of input arguments and options
        	Full width for top-hat inputs 	[kms/s]
        	Std.dev for Gaussian inputs 	[km/s]
        	Inclincation, dispersion, HI and RC parameters for toy model generation inputs
        	S/N min, max and step

    Returns
    -------
    Files : named by directory and S/N
    	"_spectra.dat"
        header with spectrum shape inputs, RMS and smoothed resolution
        velocity channels, perfect spectrum and N model spectra
    """

	comm = MPI.COMM_WORLD
	rank = comm.Get_rank()
	nproc = comm.Get_size()

	print('Generating specta with N = ', nproc, ' processors, this is proc ', rank)

	dim = 2000
	incl = args.incl
	model= 'FE'
	MHI = 1.e10
	Vdisp = args.Vdisp
	dist = 150.e0
	rms_temp = -1
	Vres = 2.e0
	Vsm = args.Vsm
	if Vsm == 0:
		Vsm = Vres
	if len(args.HI) == 2:
		HI_asym = 0
		args.HI.extend(args.HI)
	elif len(args.HI) == 4:
		HI_asym = 1
	else:
		print('Incorrect number of HI distribution arguments specified.')
		print('There should be 2 (symmetric) or 4 (asymmetric)')
		exit()
	if len(args.RC) == 3:
		RC_asym = 0
		args.RC.extend(args.RC)
	elif len(args.RC) == 6:
		RC_asym = 1
	else:
		print('Incorrect number of rotation curve arguments specified.')
		print('There should be 3 (symmetric) or 6 (asymmetric)')
		exit()
	Vsmflag = 0

	mjy_conv = 1.e3 / (2.356e5  * (dist ** 2.e0))
	Sint = mjy_conv * MHI 
	if rank == 0:
		if args.PN:
			SN_type = 'PN'
		elif args.AA:
			SN_type = 'AA'

		if args.GS[0] != 0:
			base = 'gaussian'
		elif args.TH[0] != 0:
			base = 'tophat'
		else:
			base = 'doublehorn'

		if args.width:
			model_type = 'width'
			Vmin = -1500.e0
			Vmax = 1500.e0

		elif args.Nch:
			model_type = 'Nch'
			Vmin = -500.e0
			Vmax = 500.e0

		elif args.rms:
			model_type = 'rms'
			Vmin = -800.e0
			Vmax = 800.e0

	if rank != 0:
		base = None
		Vmin = None
		Vmax = None

	base = comm.bcast(base, root = 0)
	Vmin = comm.bcast(Vmin, root = 0)
	Vmax = comm.bcast(Vmax, root = 0)


	if args.Nch:
		model_type = 'Nch'
		Vres = 0.1 						#set to high resolution for initial spectrum calculation
		Vsm = Vres 						#set to be the same to allow spectrum to have ~300km/s width
		if rank == 0:
			if args.TH[0] != 0:
				input_params = [incl, model,
							HI_asym, MHI, 0, 0, -1, -1, 0, 0, -1, -1,
							RC_asym, 0, 0, 0, 0, 0, 0,
							Vdisp, dist, rms_temp, Vmin, Vmax, Vres, Vsm]	

				vel_bins = np.arange(Vmin, Vmax , Vres)
				base_spectrum = Tophat(vel_bins, args.TH[0])
				base_spectrum = base_spectrum * (Sint / (np.nansum(base_spectrum) * Vres))
				
			elif args.GS[0] != 0:
				input_params = [incl, model,
					HI_asym, MHI, 0, 0, -1, -1, 0, 0, -1, -1,
					RC_asym, 0, 0, 0, 0, 0, 0,
					Vdisp, dist, rms_temp, Vmin, Vmax, Vres, Vsm]	

				vel_bins = np.arange(Vmin, Vmax, Vres)
				base_spectrum = Gaussian_PDF(vel_bins, 0, args.GS)
				
				base_spectrum = base_spectrum * (Sint / (np.nansum(base_spectrum) * Vres))

			else:
				print('Double-horn not currently supported for this option')
				exit()
				# input_params = [incl, model,
				# 				HI_asym, MHI, args.HI[0], args.HI[1], -1, -1, 0, 0, -1, -1,
				# 				RC_asym, args.RC[0], args.RC[1], args.RC[2], 0, 0, 0,
				# 				Vdisp, dist, rms_temp, Vmin, Vmax, Vres, Vsm]		
				# radius, costheta, R_opt = create_arrays(dim, input_params)
				# obs_mom0, rad1d, input_HI = create_mom0(radius, costheta, input_params, R_opt)
				# obs_mom1, input_RC  = create_mom1(radius, costheta, rad1d, input_params, R_opt)
				# vel_bins, base_spectrum, Sint = hi_spectra(obs_mom0, obs_mom1, input_params)
			spectrum = base_spectrum
			if len(np.where(spectrum ==  np.nanmax(spectrum))[0]) > 3:
				Peaks = [np.nanmax(spectrum), np.nanmax(spectrum)]
			else:
				Peaklocs = locate_peaks(spectrum)
				Peaks = spectrum[Peaklocs]
			width_full = locate_width(spectrum, Peaks, 0.2e0)
			width = (width_full[1] - width_full[0]) * Vres
			
			model_SN = args.SN_range[0]
			Nch_range = np.concatenate((np.arange(0.5,1,0.1),
						np.arange(1,10,0.2) ,
						np.arange(10,100,2),
						np.arange(100,600,100)))

			Vres_range = width / Nch_range
			
			if args.PN:
				RMS_sm = np.nanmax(spectrum) / SN_range
			elif args.AA:
				RMS_sm = rms_from_StN(model_SN, Sint, width, Vres_range)

			RMS_input = RMS_sm * np.sqrt(int(Vsm / Vres))

		if rank != 0:
			Nch_range = None
			RMS_input = None
			base_spectrum = None
			spectrum = None
			input_params = None
			RMS_sm = None
			Vres_range = None

		Nch_range = comm.bcast(Nch_range, root = 0)
		Vres_range = comm.bcast(Vres_range, root = 0)
		RMS_input = comm.bcast(RMS_input, root = 0)
		base_spectrum = comm.bcast(base_spectrum, root = 0)
		spectrum = comm.bcast(spectrum, root = 0)
		input_params = comm.bcast(input_params, root = 0)
		RMS_sm = comm.bcast(RMS_sm, root = 0)

		N_Nchvals = len(RMS_input)
		for ii in range(rank, N_Nchvals ,nproc):
			print('proc', rank, 'Generating realisations for Nch = ', Nch_range[ii])
			input_params[21] = RMS_input[ii]
			input_params[24] = Vres_range[ii]
			Vres = Vres_range[ii]

			if args.TH[0] != 0:
				vel_bins = np.arange(Vmin, Vmax , Vres)
				base_spectrum = Tophat(vel_bins, args.TH[0])
				base_spectrum = base_spectrum * (Sint / (np.nansum(base_spectrum) * Vres))
				
			elif args.GS[0] != 0:
				vel_bins = np.arange(Vmin, Vmax, Vres)
				base_spectrum = Gaussian_PDF(vel_bins, 0, args.GS)
				base_spectrum = base_spectrum * (Sint / (np.nansum(base_spectrum) * Vres))
			
			spectra = np.zeros([len(base_spectrum), args.Nmodels[0] + 2])
			spectra[:,0] = vel_bins
			spectra[:,1] = base_spectrum
			for n in range(args.Nmodels[0]):
				obs_spectrum = add_noise(base_spectrum, input_params)
				# obs_spectrum = smooth_spectra(vel_bins, obs_spectrum, input_params)
				spectra[:, n + 2] = obs_spectrum

			filename = '{md}Nch{Nch:.2f}_spectra.dat'.format( 
				md = model_directory, Nch = Nch_range[ii])
			
			if args.TH[0] != 0:
				header = 'Tophat = {width}\nrms = {rms:0.5f}\nVsm = {Vsm}\n'.format(
						width = args.TH[0], rms = RMS_sm[ii], Vsm = Vres_range[ii])
			elif args.GS[0] != 0:
				header = 'Gaussian = {sigma}\nrms = {rms:0.5f}\nVsm = {Vsm}\n'.format(
						sigma = args.GS, rms = RMS_sm[ii], Vsm = Vres_range[ii])
			else:
				header = 'HI = {model}, {H1}, {H2}\nRC = {R1}, {R2}, {R3}\nrms = {rms:0.5f}\nVsm = {Vsm}\n'.format(
					model = model, H1=args.HI[0], H2 = args.HI[1], 
					R1 = args.RC[0], R2 = args.RC[1], R3 = args.RC[2],
					rms = RMS_sm[ii], Vsm = Vsm)
			np.savetxt(filename, spectra, header = header,fmt = "%.6e")

	elif args.width:	
		model_type = 'width'
		SN_range = np.arange(args.SN_range[0], args.SN_range[1] + args.SN_range[2], args.SN_range[2])
		RMS_sm = 2.0e0
		if Vsm == 0:
			Vsm = Vres
			Vsmflag = 1
		RMS_input = RMS_sm * np.sqrt(int(Vsm / Vres))
		if Vsmflag == 1:
			Vsm = 0

		if args.PN:
			if args.TH[0] != 0:
				profile_widths = THwidth_from_StNPN(SN_range, Sint, RMS_sm, Vres)
			elif args.GS[0] != 0:
				profile_widths = GSwidth_from_StNPN(SN_range, Sint, RMS_sm)
			else:
				print('Double-horn not currently supported for this option')
				exit()
			# profile_widths -= 0.5 * Vsm 									#smoothing de-correction

		elif args.AA:
			if Vsm == 0:
				Vsm = Vres
				Vsmflag = 1
			profile_widths = width_from_StNAA(SN_range, Sint, RMS_sm, Vsm) 
			if Vsmflag == 1:
				Vsm = 0
			if args.TH[0] != 0:
				profile_widths = profile_widths
				# print(profile_widths)
				# profile_widths -=  0.5 * Vsm 									#smoothing de-correction
			elif args.GS[0] != 0:
				# profile_widths -= 0.5 * Vsm 									#smoothing de-correction
				profile_widths =  profile_widths / (2.e0 * np.sqrt(2.e0 * np.log(5.e0)))

			else:
				print('Double-horn not currently supported for this option')
				exit()

		if rank != 0:
			profile_widths = None
		profile_widths = comm.bcast(profile_widths, root=0)

		N_SNvals = len(profile_widths)


		for ii in range(rank * int(N_SNvals / nproc),(rank + 1) * int(N_SNvals / nproc)):
			print('proc', rank, 'Generating realisations for SN = ', int(SN_range[ii]))

			if args.TH[0] != 0:
				input_params = [incl, model,
							HI_asym, MHI, 0, 0, -1, -1, 0, 0, -1, -1,
							RC_asym, 0, 0, 0, 0, 0, 0,
							Vdisp, dist, rms_temp, Vmin, Vmax, Vres, Vsm]	

				vel_bins = np.arange(Vmin, Vmax , Vres)
				base_spectrum = Tophat(vel_bins, profile_widths[ii])
				base_spectrum = base_spectrum * (Sint / (np.nansum(base_spectrum) * Vres))
				
			elif args.GS[0] != 0:
				input_params = [incl, model,
					HI_asym, MHI, 0, 0, -1, -1, 0, 0, -1, -1,
					RC_asym, 0, 0, 0, 0, 0, 0,
					Vdisp, dist, rms_temp, Vmin, Vmax, Vres, Vsm]	

				vel_bins = np.arange(Vmin, Vmax, Vres)
				base_spectrum = Gaussian_PDF(vel_bins, 0, profile_widths[ii])
				base_spectrum = base_spectrum * (Sint / (np.nansum(base_spectrum) * Vres))
			else:
				print('Double-horn not currently supported for this option')
				exit()

			# spectrum = smooth_spectra(vel_bins, base_spectrum, input_params)
			spectrum = base_spectrum
			if base != 'doublehorn':
				Peaks = [np.nanmax(spectrum), np.nanmax(spectrum)]
			else:
				Peaklocs = locate_peaks(spectrum)
				Peaks = spectrum[Peaklocs]
			width_full = locate_width(spectrum, Peaks, 0.2e0)
			width = (width_full[1] - width_full[0]) * Vres

			spectra = np.zeros([len(spectrum), args.Nmodels[0] + 2])
			spectra[:,0] = vel_bins
			spectra[:,1] = spectrum

			input_params[21] = RMS_input
			for n in range(args.Nmodels[0]):
				obs_spectrum = add_noise(base_spectrum, input_params)
				# obs_spectrum = smooth_spectra(vel_bins, obs_spectrum, input_params)
				spectra[:, n + 2] = obs_spectrum

			filename = '{md}SN{SN}_spectra.dat'.format( 
				md = model_directory, SN = int(SN_range[ii]))
			
			if args.TH[0] != 0:
				header = 'Tophat = {width}\nrms = {rms:0.5f}\nVsm = {Vsm}\n'.format(
						width = profile_widths[ii], rms = RMS_sm, Vsm = Vsm)
			elif args.GS[0] != 0:
				header = 'Gaussian = {sigma}\nrms = {rms:0.5f}\nVsm = {Vsm}\n'.format(
						sigma = profile_widths[ii], rms = RMS_sm, Vsm = Vsm)
			else:
				header = 'HI = {model}, {H1}, {H2}\nRC = {R1}, {R2}, {R3}\nrms = {rms:0.5f}\nVsm = {Vsm}\n'.format(
					model = model, H1=args.HI[0], H2 = args.HI[1], 
					R1 = args.RC[0], R2 = args.RC[1], R3 = args.RC[2],
					rms = RMS_sm[ii], Vsm = Vsm)
			np.savetxt(filename, spectra, header = header,fmt = "%.4e")

	elif args.rms:
		if rank == 0:
			if args.TH[0] != 0:
				input_params = [incl, model,
							HI_asym, MHI, 0, 0, -1, -1, 0, 0, -1, -1,
							RC_asym, 0, 0, 0, 0, 0, 0,
							Vdisp, dist, rms_temp, Vmin, Vmax, Vres, Vsm]	

				vel_bins = np.arange(Vmin,Vmax , Vres)
				base_spectrum = Tophat(vel_bins, args.TH[0])
				base_spectrum = base_spectrum * (Sint / (np.nansum(base_spectrum) * Vres))
				
			elif args.GS[0] != 0:
				input_params = [incl, model,
					HI_asym, MHI, 0, 0, -1, -1, 0, 0, -1, -1,
					RC_asym, 0, 0, 0, 0, 0, 0,
					Vdisp, dist, rms_temp, Vmin, Vmax, Vres, Vsm]	

				vel_bins = np.arange(Vmin, Vmax, Vres)
				base_spectrum = Gaussian_PDF(vel_bins, 0, args.GS[0], args.GS[1])
				
				base_spectrum = base_spectrum * (Sint / (np.nansum(base_spectrum) * Vres))

			else:
				input_params = [incl, model,
								HI_asym, MHI, args.HI[0], args.HI[1], -1, -1 , args.HI[2], args.HI[3], -1, -1,
								RC_asym, args.RC[0], args.RC[1], args.RC[2], args.RC[3], args.RC[4], args.RC[5],
								Vdisp, dist, rms_temp, Vmin, Vmax, Vres, Vsm]		
				radius, costheta, R_opt = create_arrays(dim, input_params)
				obs_mom0, rad1d, input_HI = create_mom0(radius, costheta, input_params, R_opt)
				obs_mom1, input_RC  = create_mom1(radius, costheta, rad1d, input_params, R_opt)
				vel_bins, base_spectrum, Sint = hi_spectra(obs_mom0, obs_mom1, input_params)

			spectrum = smooth_spectra(vel_bins, base_spectrum, input_params)
			if len(np.where(spectrum ==  np.nanmax(spectrum))[0]) > 3:
				Peaks = [np.nanmax(spectrum), np.nanmax(spectrum)]
			else:
				Peaklocs = locate_peaks(spectrum)
				Peaks = spectrum[Peaklocs]
			width_full = locate_width(spectrum, Peaks, 0.2e0)
			width = (width_full[1] - width_full[0]) * Vres
			Sint, Afr = areal_asymmetry(spectrum, width_full, Vres)

			SN_range = np.arange(args.SN_range[0], args.SN_range[1] + args.SN_range[2], args.SN_range[2])
			if args.PN:
				RMS_sm = np.nanmax(spectrum) / SN_range
			elif args.AA:
				RMS_sm = rms_from_StN(SN_range, Sint, width, Vsm)
			RMS_input = RMS_sm * np.sqrt(int(Vsm / Vres))

			model_directory = './{base}{St}{option}_Afr{Afr:.2f}_{mt}/'.format(
				base = base, Afr = Afr, St = SN_type, mt = model_type, option = args.option)
			if len(glob.glob(model_directory)) == 0:
				os.mkdir(model_directory)

		if rank != 0:
			model_directory = None
			RMS_input = None
			base_spectrum = None
			spectrum = None
			vel_bins = None
			input_params = None
			SN_range = None
			RMS_sm = None
		model_directory = comm.bcast(model_directory, root = 0)
		SN_range = comm.bcast(SN_range, root = 0)
		RMS_input = comm.bcast(RMS_input, root = 0)
		base_spectrum = comm.bcast(base_spectrum, root = 0)
		spectrum = comm.bcast(spectrum, root = 0)
		vel_bins = comm.bcast(vel_bins, root = 0)
		input_params = comm.bcast(input_params, root = 0)
		RMS_sm = comm.bcast(RMS_sm, root = 0)

		N_SNvals = len(RMS_input)
		spectra = np.zeros([len(spectrum), args.Nmodels[0] + 2])
		spectra[:,0] = vel_bins
		spectra[:,1] = spectrum
		for ii in range(rank * int(N_SNvals / nproc),(rank + 1) * int(N_SNvals / nproc)):
			print('proc', rank, 'Generating realisations for SN = ', int(SN_range[ii]))
			input_params[21] = RMS_input[ii]
			for n in range(args.Nmodels[0]):
				obs_spectrum = add_noise(base_spectrum, input_params)
				obs_spectrum = smooth_spectra(vel_bins, obs_spectrum, input_params)
				spectra[:, n + 2] = obs_spectrum

			filename = '{md}SN{SN}_spectra.dat'.format( 
				md = model_directory, SN = int(SN_range[ii]))
			
			if args.TH[0] != 0:
				header = 'Tophat = {width}\nrms = {rms:0.5f}\nVsm = {Vsm}\n'.format(
						width = args.TH[0], rms = RMS_sm[ii], Vsm = Vsm)
			elif args.GS[0] != 0:
				header = 'Gaussian = {sigma} {alpha}\nrms = {rms:0.5f}\nVsm = {Vsm}\n'.format(
						sigma = args.GS[0], alpha = args.GS[1] , rms = RMS_sm[ii], Vsm = Vsm)
			else:
				header = 'HI = {model}, {H1}, {H2}, {H3}, {H4} \nRC = {R1}, {R2}, {R3}, {R4}, {R5}, {R6} \n' \
						'rms = {rms:0.5f}\nVsm = {Vsm}\n'.format(
					model = model, H1=args.HI[0], H2 = args.HI[1], H3=args.HI[2], H4 = args.HI[3], 
					R1 = args.RC[0], R2 = args.RC[1], R3 = args.RC[2], R4 = args.RC[3], 
					R5 = args.RC[4], R6 = args.RC[5], rms = RMS_sm[ii], Vsm = Vsm)
			np.savetxt(filename, spectra, header = header,fmt = "%.4e")

def measure(args):
	"""
	Measure integrated flux, S/N, and asymemtry of mock spectra

    Parameters
    ----------
    args : list
        List of input arguments and options
        	Model directory
        	
    Returns
    -------
    Files : named by directory and S/N
    	"_measured.dat"
        header with perfect spectrum's asymmetry, input RMS, and smoothed resolution
        Integrated flux, S/N, and asymmetry of mock spectra
    """
	comm = MPI.COMM_WORLD
	rank = comm.Get_rank()
	nproc = comm.Get_size()

	if rank == 0:
		if args.dir[-1]=='/':
			model_code = args.dir.split('/')[-2]
		else:
			model_code = args.dir.split('/')[-1]
			args.dir = '{dir}/'.format(dir=args.dir)

		if 'AA' in args.dir:
			SN_type = 'AA'
		elif 'PN' in args.dir:
			SN_type = 'PN'
		base = model_code.split(SN_type)[0]

		if 'Nch' in args.dir:
			model_type = 'Nch'
		elif 'width' in args.dir:
			model_type = 'width'
		elif 'rms' in args.dir:
			model_type = 'rms'

		file_list = glob.glob('{dir}*_spectra.dat'.format(dir = args.dir))

		if model_type == 'Nch':

			Nch_range = np.array([])
			Vres_range = np.array([])

			for file in file_list:
				f = open(file)
				for line in f:
					split = line.split(' ')
					if split[1] == 'rms':
						rms = float(split[3])
					if split[1] == 'Vsm':
						Vsm = float(split[3])
						break
				f.close()
				Vres_range = np.append(Vres_range, Vsm)
				Nch = file.split('Nch')[-1].split('_')[0]
				Nch_range = np.append(Nch_range,np.float(Nch))
			Nch_sort = np.argsort(Nch_range)
			Nch_range = Nch_range[Nch_sort]
			Vres_range = Vres_range[Nch_sort]
			file_list = np.array(file_list)[Nch_sort]
			# print(file_list)


		Nfiles = len(file_list)

	else:
		Nfiles = None
		base = None
		SN_type = None
		file_list = None
		model_type = None
	Nfiles = comm.bcast(Nfiles, root = 0)
	base  = comm.bcast(base, root = 0)
	SN_type = comm.bcast(SN_type, root = 0)
	file_list = comm.bcast(file_list, root = 0)
	model_type = comm.bcast(model_type, root = 0)

	if model_type == 'Nch':
		loop_range = range(rank, Nfiles ,nproc)
	else: 
		loop_range = range(rank * int(Nfiles / nproc), (rank + 1) * int(Nfiles / nproc))

	for ff in loop_range:
		file = file_list[ff]
		print('proc', rank, 'measuring ', file)
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
		print('file loaded')
		vel_bins = spectra[:,0]
		spectrum = spectra[:,1]
		Vres = np.abs(vel_bins[1] - vel_bins[0])
		if Vsm == 0:
			Vsm = Vres
		if base != 'doublehorn':
			Peaks = [np.nanmax(spectrum), np.nanmax(spectrum)]
		else:
			Peaklocs = locate_peaks(spectrum)
			Peaks = spectrum[Peaklocs]

		width_20 = locate_width(spectrum, Peaks, 0.2e0)
		width_50 = locate_width(spectrum, Peaks, 0.5e0)

		w20 = (width_20[1] - width_20[0]) * Vres
		w50 = (width_50[1] - width_50[0]) * Vres

		Sint_noiseless_w20, Afr_noiseless_w20 = areal_asymmetry(spectrum, width_20, Vres)
		Sint_noiseless_w50, Afr_noiseless_w50 = areal_asymmetry(spectrum, width_50, Vres)
		measurements = np.zeros([len(spectra[0]) - 1, 9])
		measurements[0,:] = [Sint_noiseless_w20, -1, -1, -1, -1, Afr_noiseless_w20,
			Afr_noiseless_w50, -1, -1]
		for run in range(len(spectra[0]) - 2):
			obs_spectrum = spectra[:, run + 2]

			Sint_w20, Afr_w20 = areal_asymmetry(obs_spectrum, width_20, Vres)
			Sint_w50, Afr_w50 = areal_asymmetry(obs_spectrum, width_50, Vres)
			
			if SN_type == 'PN':
				SN_w20 = np.nanmax(obs_spectrum) / rms
				SN_w50 = np.nanmax(obs_spectrum) / rms
			elif SN_type == 'AA':
				SN_w20 = StN(Sint_w20, w20, rms, Vsm)
				SN_w50 = StN(Sint_w50, w50, rms, Vsm)

			spec_max = np.nanmax(spectrum)
			if base != 'doublehorn':
				Peaks_obs = [np.nanmax(obs_spectrum) - rms,np.nanmax(obs_spectrum) - rms]
			else:
				tol = int(30./Vres)						#select the maximum channel around the known peaks
				Peaks_obs = [np.nanmax(obs_spectrum[Peaklocs[0] - tol:Peaklocs[0] + tol]) - rms,
						np.nanmax(obs_spectrum[Peaklocs[1] - tol:Peaklocs[1] + tol]) - rms]

			width_20_obs = locate_width(spectrum, Peaks_obs, 0.2e0)
			width_50_obs = locate_width(spectrum, Peaks_obs, 0.5e0)
			if all(w > 0 for w in width_20_obs) and all(i == False for i in np.isinf(width_20_obs)):
				Sint_w20_obs, Afr_w20_obs = areal_asymmetry(obs_spectrum, width_20_obs, Vres)
				w20_obs = (width_20_obs[1] - width_20_obs[0]) * Vres
				if SN_type == 'AA':
					SN_w20_obs = StN(Sint_w20_obs, w20_obs, rms, Vsm)
				elif SN_type == 'PN':
					SN_w20_obs = np.nanmax(obs_spectrum) / rms
			else:
				Afr_w20_obs = -1
				SN_w20_obs = -1
			if all(w > 0 for w in width_50_obs) and all(i == False for i in np.isinf(width_50_obs)):
				Sint_w50_obs, Afr_w50_obs = areal_asymmetry(obs_spectrum, width_50_obs, Vres)
				w50_obs = (width_50_obs[1] - width_50_obs[0]) * Vres
				if SN_type == 'AA':
					SN_w50_obs = StN(Sint_w50_obs, w50_obs, rms, Vsm)
				elif SN_type == 'PN':
					SN_w50_obs = np.nanmax(spectrum) / rms
			else:
				Afr_w50_obs = -1
				SN_w50_obs = -1

			measurements[run + 1,:] = [Sint_w20, SN_w20, SN_w50, SN_w20_obs,
				SN_w50_obs, Afr_w20, Afr_w50, Afr_w20_obs, Afr_w50_obs]
		base = file.split('_spectra.dat')[0]
		filename = '{base}_measured.dat'.format(base=base)
		header = 'rms = {rms:0.5f}\nVsm = {Vsm}\n'.format(rms = rms, Vsm = Vsm)
		np.savetxt(filename, measurements, header=header, fmt="%.4e")

def statistics(args):
	"""
	Calculates statistics and density of model spectra in bins of S/N and asymmetry

    Parameters
    ----------
    args : list
        List of input arguments and options
        	Model directory
        	
    Returns
    -------
    Files : named by directory and S/N
        "_Afrstatistics.dat"
        Astropy table
        	1st row: all -1 except asymmetry of perfect profile in 2nd column
        	mean S/N in bin, mode of asymmetry, 50th, 75th, 90th percentiles
        "_Afrdensityplot.dat"
        2D array of number of models in a 2D bin
        	Rows: bins of asymmetry
        	Columns: bins of S/N
    """
	if args.dir[-1]=='/':
		model_code = args.dir.split('/')[-2]
	else:
		model_code = args.dir.split('/')[-1]
		args.dir = '{dir}/'.format(dir = args.dir)

	if 'AA' in args.dir:
		SN_type = 'AA'
	elif 'PN' in args.dir:
		SN_type = 'PN'

	if 'Nch' in args.dir:
		model_type = 'Nch'
	elif 'width' in args.dir:
		model_type = 'width'
	elif 'rms' in args.dir:
		model_type = 'rms'
	file_list = glob.glob('{dir}*_measured.dat'.format(dir = args.dir))

	if model_type == 'Nch':

		Nch_range = np.array([])
		Vres_range = np.array([])

		for file in file_list:
			f = open(file)
			for line in f:
				split = line.split(' ')
				if split[1] == 'rms':
					rms = float(split[3])
				if split[1] == 'Vsm':
					Vsm = float(split[3])
					break
			f.close()
			Vres_range = np.append(Vres_range, Vsm)
			Nch = file.split('Nch')[-1].split('_')[0]
			Nch_range = np.append(Nch_range,np.float(Nch))
		Nch_sort = np.argsort(Nch_range)
		Nch_range = Nch_range[Nch_sort]
		Vres_range = Vres_range[Nch_sort]
		file_list = np.array(file_list)[Nch_sort]

		Sint = np.array([])
		SN_w20 = np.array([])
		SN_w50 = np.array([])
		SN_w20_obs = np.array([])
		SN_w50_obs = np.array([])
		Afr_w20 = np.array([])
		Afr_w50 = np.array([])
		Afr_w20_obs = np.array([])
		Afr_w50_obs = np.array([])

		names = ['w20', 'w50', 'w20_obs', 'w50_obs']

		column_names = ['Nch', 'Vres']
		formats = {'Nch':'4.2f', 'Vres': '3.4f'}

		dP = 5
		percentile_vals = np.arange(5, 95 + dP, dP)
		percentiles = np.zeros(len(percentile_vals) + 1)

		names = ['w20', 'w50', 'w20_obs', 'w50_obs']
		for name in names:
			column_names.extend(['avg_SN_{n}'.format(n = name)])						
			formats['avg_SN_{n}'.format(n = name)] = '4.2f'

			for pp in range(len(percentile_vals)):
				column_names.extend(['P{X}_{n}'.format(X = int(percentile_vals[pp]), n = name)])
				formats['P{X}_{n}'.format(X = int(percentile_vals[pp]), n = name)] = '.5f'
		
		Afr_statistics = np.zeros([len(file_list), 2 + len(percentiles) * len(names)])
		Afr_statistics[:,0] = Nch_range
		Afr_statistics[:,1] = Vres_range

		for ff in range(len(file_list)):
			measurements = np.genfromtxt(file_list[ff])
			measurements = measurements[1::,:]
			Sint = measurements[:, 0]
			SN_w20 = measurements[:, 1]
			SN_w50 = measurements[:, 2]
			SN_w20_obs = measurements[:, 3]
			SN_w50_obs = measurements[:, 4]
			Afr_w20 = measurements[:, 5]
			Afr_w50 = measurements[:, 6]
			Afr_w20_obs = measurements[:, 7]
			Afr_w50_obs = measurements[:, 8]

			SN = [SN_w20, SN_w50, SN_w20_obs, SN_w50_obs]
			Afr = [Afr_w20, Afr_w50, Afr_w20_obs, Afr_w50_obs]

			for ss in range(len(SN)):
				percentiles[0] = np.mean(SN[ss])		
		
				for pp in range(len(percentile_vals)):
					percentiles[pp + 1] = np.percentile(Afr[ss], percentile_vals[pp])
		
				Afr_statistics[ff, (ss * len(percentiles) + 2):
								((ss + 1) * len(percentiles) + 2)] = percentiles

		table = Table(rows = Afr_statistics, names = column_names)
		stats_filename = '{dir}statistics.dat'.format(dir = args.dir) 
		table.write(stats_filename, formats = formats, format = 'ascii', overwrite = True)

	else:	
		Sint = np.array([])
		SN_w20 = np.array([])
		SN_w50 = np.array([])
		SN_w20_obs = np.array([])
		SN_w50_obs = np.array([])
		Afr_w20 = np.array([])
		Afr_w50 = np.array([])
		Afr_w20_obs = np.array([])
		Afr_w50_obs = np.array([])

		for file in file_list:
			measurements = np.genfromtxt(file)
			measurements = measurements[1::,:]
			Sint = np.append(Sint, measurements[:, 0])
			SN_w20 = np.append(SN_w20, measurements[:, 1])
			SN_w50 = np.append(SN_w50, measurements[:, 2])
			SN_w20_obs = np.append(SN_w20_obs, measurements[:, 3])
			SN_w50_obs = np.append(SN_w50_obs, measurements[:, 4])
			Afr_w20 = np.append(Afr_w20, measurements[:, 5])
			Afr_w50 = np.append(Afr_w50, measurements[:, 6])
			Afr_w20_obs = np.append(Afr_w20_obs, measurements[:, 7])
			Afr_w50_obs = np.append(Afr_w50_obs, measurements[:, 8])

		SN_bins = np.arange(int(np.nanmax([5, np.nanmin(SN_w20)])), 
				int(np.nanmax(SN_w20) * 0.1) * 10, 4)
		Afr_densitybins = np.arange(1., 3, 0.05)
		Afr_densityplot = np.zeros([len(Afr_densitybins) - 1, len(SN_bins) - 1])


		dP = 5
		percentile_vals = np.arange(5, 95 + dP, dP)
		percentiles = np.zeros(len(percentile_vals) + 2)

		names = ['w20', 'w50', 'w20_obs', 'w50_obs']
		SN = [SN_w20, SN_w50, SN_w20_obs, SN_w50_obs]
		Afr = [Afr_w20, Afr_w50, Afr_w20_obs, Afr_w50_obs]

		Afr_statistics = np.zeros([len(SN_bins) - 1, 1 + len(percentiles) * len(SN)])
		Afr_statistics[:,0] = SN_bins[0:-1]

		column_names = ['SN_bin']
		formats = {'SN_bin':'4.2f'}

		for mm in range(len(SN)):
			SN_vals = SN[mm]
			Afr_vals = Afr[mm]

			column_names.extend(['avg_SN_{n}'.format(n = names[mm]),
								'mode_Afr_{n}'.format(n = names[mm])])
			formats['avg_SN_{n}'.format(n = names[mm])] = '4.2f'
			formats['mode_Afr_{n}'.format(n = names[mm])] = '4.2f'

			for pp in range(len(percentile_vals)):
				column_names.extend(['P{X}_{n}'.format(X = int(percentile_vals[pp]), n = names[mm])])
				formats['P{X}_{n}'.format(X = int(percentile_vals[pp]), n = names[mm])] = '.5f'

			for ii in range(len(SN_bins) - 1):
				SN_low = SN_bins[ii]
				SN_high = SN_bins[ii + 1]

				inbin_SN = SN_vals[(SN_vals >= SN_low) & (SN_vals < SN_high)]
				inbin_Afr = Afr_vals[(SN_vals >= SN_low) & (SN_vals < SN_high)]

				if len(inbin_SN) != 0:
					if mm == 1:
						for jj in range(len(Afr_densitybins) - 1):
							Afr_low = Afr_densitybins[jj]
							Afr_high = Afr_densitybins[jj + 1]
							incell = Afr_vals[(Afr_vals >= Afr_low) & (Afr_vals < Afr_high)]
							Afr_densityplot[jj,ii] = float(len(incell)) / float(len(inbin_SN))
				
					percentiles[0] = np.mean(inbin_SN)		
					percentiles[1] = mode(inbin_Afr)[0][0]
				
					for pp in range(len(percentile_vals)):
						percentiles[pp + 2] = np.percentile(inbin_Afr,percentile_vals[pp])
				
					Afr_statistics[ii, (mm * len(percentiles) + 1):
										((mm + 1) * len(percentiles) + 1)] = percentiles

		table = Table(rows = Afr_statistics, names = column_names)
		stats_filename = '{dir}statistics.dat'.format(dir = args.dir) 
		table.write(stats_filename, formats = formats, format = 'ascii', overwrite = True)
		
		density_filename = '{dir}densityplot.dat'.format(dir = args.dir)
		np.savetxt(density_filename, Afr_densityplot)

def plot_spectrum(args):
	"""
	Plots a model spectrum and residual

    Parameters
    ----------
    args : list
        List of input arguments and options
        	Model directory
        	S/N value of spectrum
        	Model number of spectrum 
        	Flag to save plot

    Returns
    -------
    Plot: Velocity vs Spectral flux [mJy]
        Observed spectrum (black, spectrum axis)
        Input spectrum(red, spectrum axis)
        Measurement limits (green, spectrum axis
        Observed - input (black, residual axis)
        Input RMS (red, residual axis)
        Measurement limits (green, spectrum axis)
    """

	if args.dir[-1]=='/':
		model_code = args.dir.split('/')[-2]
	else:
		model_code = args.dir.split('/')[-1]
		args.dir = '{dir}/'.format(dir=args.dir)
	if 'AA' in args.dir:
		SN_type = 'AA'
	elif 'PN' in args.dir:
		SN_type = 'PN'
	base = model_code.split(SN_type)[0]

	spectra_file = '{dir}SN{SN}_spectra.dat'.format(dir=args.dir,mc=model_code,SN=args.SN)
	f = open(spectra_file)
	for line in f:
		split = line.split(' ')
		if split[1] == 'rms':
			rms = float(split[3])
		if split[1] == 'Vsm':
			Vsm = float(split[3])
			break
	f.close()

	spectra = np.loadtxt(spectra_file, ndmin=2)
	vel_bins = spectra[:,0]
	spectrum = spectra[:,1]
	obs_spectrum = spectra[:,args.num + 2 - 1]
	noise = obs_spectrum - spectrum

	Vres = np.abs(vel_bins[1] - vel_bins[0])
	spec_max = np.nanmax(spectrum)
	if base != 'doublehorn':
		Peaks = [spec_max, spec_max]
		Peaklocs_obs = [np.where(obs_spectrum == np.nanmax(obs_spectrum))[0],
				np.where(obs_spectrum == np.nanmax(obs_spectrum))[0]]
	else:
		Peaklocs = locate_peaks(spectrum)
		Peaks = spectrum[Peaklocs]
		tol = int(30. / Vres)						#select the maximum channel around the known peaks
		Peaks_obs = [np.nanmax(obs_spectrum[Peaklocs[0] - tol:Peaklocs[0] + tol]),
				np.nanmax(obs_spectrum[Peaklocs[1] - tol:Peaklocs[1] + tol])]
		Peaklocs_obs = [Peaklocs[0] - tol + np.where(obs_spectrum[Peaklocs[0] - tol:Peaklocs[0] + tol] == 
				Peaks_obs[0])[0],
				Peaklocs[1] - tol + np.where(obs_spectrum[Peaklocs[1] - tol:Peaklocs[1] + tol] == 
				Peaks_obs[1])[0]]	
	width = locate_width(spectrum, Peaks, 0.20e0)
	Sint, Afr = areal_asymmetry(obs_spectrum, width, Vres)

	minvel = vel_bins[int(np.floor(width[0]))] + Vres * (width[0]- int(np.floor(width[0])))
	maxvel = vel_bins[int(np.floor(width[1]))] + Vres * (width[1] - int(np.floor(width[1])))
	midvel = 0.5e0*(minvel + maxvel)
	PeakL_vel = vel_bins[Peaklocs_obs[0]]
	PeakR_vel = vel_bins[Peaklocs_obs[1]]

	fig = plt.figure(figsize = (10,10))
	gs = gridspec.GridSpec(3, 1, hspace = 0)
	spec_ax = fig.add_subplot(gs[0:2,0])
	noise_ax = fig.add_subplot(gs[2,0], sharex = spec_ax)

	spec_ax.tick_params(axis = 'y', which = 'both', direction = 'in', labelsize = 9)
	spec_ax.tick_params(axis = 'x', which = 'both', direction = 'in', labelsize = 0.)
	noise_ax.tick_params(axis = 'both',which = 'both',direction = 'in', labelsize = 9)
	noise_ax.set_xlabel('Velocity [km/s]', fontsize = 10)
	noise_ax.set_ylabel('Residual  [mJy]', fontsize = 10)
	noise_ax.set_ylim([-4 * rms,4 * rms])
	spec_ax.set_ylabel('Spectral Flux [mJy]', fontsize = 10)
	spec_ax.set_title('{mc}  A = {A:.3f}  SN = {SN}  N = {N}'.format(
			mc = model_code, A = Afr, SN = args.SN, N = args.num))

	spec_ax.plot(vel_bins,spectrum, color = 'Red', zorder = 0, ls = '-', linewidth = 2)
	spec_ax.plot(vel_bins,obs_spectrum, color = 'Black', zorder = -1, ls = '--')
	noise_ax.plot(vel_bins,noise, color = 'Black', zorder = 1, ls = '--')
	noise_ax.plot([vel_bins[0], vel_bins[-1]], [-1.e0 * rms, -1.e0 * rms],
			color = 'Red', ls = '-', zorder = 0, linewidth = 2)
	noise_ax.plot([vel_bins[0],vel_bins[-1]],[rms,rms], 
			color = 'Red', ls = '-', zorder = 0, linewidth = 2)
	spec_ax.plot([minvel,minvel], [np.min(obs_spectrum), np.max(obs_spectrum)],
			ls = ':', linewidth = 3, color = 'Green', zorder = 0)
	spec_ax.plot([maxvel,maxvel], [np.min(obs_spectrum), np.max(obs_spectrum)],
			ls = ':', linewidth = 3, color = 'Green', zorder = 0)
	spec_ax.plot([midvel,midvel], [np.min(obs_spectrum), np.max(obs_spectrum)],
			ls = '--', linewidth = 3, color = 'Green', zorder = 0)

	spec_ax.plot([PeakL_vel,PeakL_vel], [np.max(obs_spectrum) - rms, np.max(obs_spectrum)],
			ls = '--', linewidth = 2, color = 'Blue', zorder = 0)
	spec_ax.plot([PeakR_vel,PeakR_vel], [np.max(obs_spectrum) - rms, np.max(obs_spectrum)],
			ls = '--', linewidth = 2, color = 'Blue', zorder = 0)

	if args.save == True:
		plotdir = '{dir}figures/{mc}_SN{SN}_N{N}_spectrum.png'.format(
			dir = args.dir, mc = model_code, SN = args.SN, N = args.num)
		fig.savefig(plotdir, dpi = 150)
	else:	
		plt.show()

def density_plot(args):
	"""
	Plots the density-plot of the S/N - asymmetry parameter space with statistics overlaid

    Parameters
    ----------
    args : list
        List of input arguments and options
        	Model directory

    Returns
    -------
    Plot: S/N vs Asymmetry
    	Colour coded densities by fraction of total points in a S/N bin in bins of Asymmetry
    	Mode in each S/N bin (black)
    	50th, 75th and 90th Percentiles
    """

	if args.dir[-1] == '/':
		model_code = args.dir.split('/')[-2]
	else:
		model_code = args.dir.split('/')[-1]
		args.dir = '{dir}/'.format(dir = args.dir)

	AA_density_filename = glob.glob('{dir}{mc}_AA_Afrdensityplot.dat'.format(
		dir = args.dir, mc = model_code))[0]
	AA_stats_filename = glob.glob('{dir}{mc}_AA_Afrstatistics.dat'.format(
		dir = args.dir, mc = model_code))[0]

	PN_density_filename = glob.glob('{dir}{mc}_PN_Afrdensityplot.dat'.format(
		dir = args.dir, mc = model_code))[0]
	PN_stats_filename = glob.glob('{dir}{mc}_PN_Afrstatistics.dat'.format(
		dir = args.dir, mc = model_code))[0]

	Afr_AA_densityplot = np.loadtxt(AA_density_filename, ndmin = 2)
	Afr_AA_statistics = np.loadtxt(AA_stats_filename, ndmin = 2, skiprows = 1 )
	Afr_noiseless = Afr_AA_statistics[0,2]
	Afr_AA_statistics = Afr_AA_statistics[1::,:]
	SN_AA_bins = Afr_AA_statistics[:,0]
	mu_SN = Afr_AA_statistics[:,1]
	mode_Afr = Afr_AA_statistics[:,2]
	P50_Afr = Afr_AA_statistics[:,3]
	P75_Afr = Afr_AA_statistics[:,4]
	P90_Afr = Afr_AA_statistics[:,5]
	SN_AA_bins = np.append(SN_AA_bins,(SN_AA_bins[-1] + np.abs(np.diff(SN_AA_bins)[0])))
	Afr_densitybins = np.arange(1.,3,0.05)
	Afr_AA_densityplot[(Afr_AA_densityplot == 0)] = np.nan

	fig = plt.figure(figsize = (10,10))
	plt.pcolormesh(SN_AA_bins, Afr_densitybins, Afr_AA_densityplot, cmap = plt.cm.Blues)
	plt.xlabel('S/N (AA)',fontsize = 12)
	plt.ylabel('$A_{fr}$',fontsize = 12)
	plt.plot(mu_SN,mode_Afr, color = 'Black', label = 'Mode')
	plt.plot(mu_SN,P50_Afr, color = 'Purple', label = '50th %tile')
	plt.plot(mu_SN,P75_Afr, color = 'Orange', label = '75th %tile')
	plt.plot(mu_SN,P90_Afr, color = 'Green', label = '90th %tile')
	plt.ylim(0.99,1.5)
	plt.xlim(np.min(SN_AA_bins),np.max(SN_AA_bins))
	plt.legend()
	plt.tight_layout()
	if args.save == True:
		plotdir = '{dir}figures/{mc}_AA_densityplot.png'.format(
			dir = args.dir, mc = model_code)
		fig.savefig(plotdir, dpi = 150)
	else:	
		plt.show()

	Afr_PN_densityplot = np.loadtxt(PN_density_filename, ndmin = 2)
	Afr_PN_statistics = np.loadtxt(PN_stats_filename, ndmin = 2, skiprows = 1 )
	Afr_noiseless = Afr_PN_statistics[0,2]
	Afr_PN_statistics = Afr_PN_statistics[1::,:]
	SN_PN_bins = Afr_PN_statistics[:,0]
	mu_SN = Afr_PN_statistics[:,1]
	mode_Afr = Afr_PN_statistics[:,2]
	P50_Afr = Afr_PN_statistics[:,3]
	P75_Afr = Afr_PN_statistics[:,4]
	P90_Afr = Afr_PN_statistics[:,5]
	SN_PN_bins = np.append(SN_PN_bins,(SN_PN_bins[-1] + np.abs(np.diff(SN_PN_bins)[0])))
	Afr_densitybins = np.arange(1.,3,0.05)
	Afr_PN_densityplot[(Afr_PN_densityplot == 0)] = np.nan

	fig = plt.figure(figsize = (10,10))
	plt.pcolormesh(SN_PN_bins, Afr_densitybins, Afr_PN_densityplot, cmap = plt.cm.Blues)
	plt.xlabel('S/N (P/$\sigma$)',fontsize = 12)
	plt.ylabel('$A_{fr}$',fontsize = 12)
	plt.plot(mu_SN,mode_Afr, color = 'Black', label = 'Mode')
	plt.plot(mu_SN,P50_Afr, color = 'Purple', label = '50th %tile')
	plt.plot(mu_SN,P75_Afr, color = 'Orange', label = '75th %tile')
	plt.plot(mu_SN,P90_Afr, color = 'Green', label = '90th %tile')
	plt.ylim(0.99,1.5)
	plt.xlim(np.min(SN_PN_bins),np.max(SN_PN_bins))
	plt.legend()
	plt.tight_layout()
	if args.save == True:
		plotdir = '{dir}figures/{mc}_PN_densityplot.png'.format(
			dir = args.dir, mc = model_code)
		fig.savefig(plotdir, dpi = 150)
	else:	
		plt.show()
##########################################################################################
#	Model generation functions
##########################################################################################

def create_arrays(dim, params):
	"""
	Creates 2D arrays of radius and angle for the HI toy model

    Parameters
    ----------
    dim : int 	[pixels]
        Dimension N 
	params : list
		List of input parameters
			params[0] = Galaxy inclination 	[deg]
        	
    Returns
    -------
 	radius : N x N array 	[pixels]
 		2D array of galactocentric radii
 	costheta: N x N array
 		2D array of cos(theta) = [-pi, pi] values where theta is the angle counter clockwise from
 		the receding major axis (defined as the positive x-axis)
 	R_opt : float 	[pixels]
 		Value of the optical radius in pixels defined as N/4, making Rmax = 2 R_opt
    """

	radius = np.zeros([dim, dim])
	costheta = np.zeros([dim, dim])
	incl = 1.e0 / np.cos(params[0] * np.pi / 180.e0)								#inclination correction goes as 1/cos
	for yy in range(dim):
		for xx in range(dim):
			xcoord = (xx + 1.e0) - 0.5e0 * (dim + 1)
			ycoord = (yy + 1.e0) - 0.5e0 * (dim + 1)
			rad = np.sqrt( xcoord * xcoord + (ycoord * ycoord * incl * incl) )	#y coordinate is projected by inclination
			if rad <= 0.5e0 * (dim + 1.e0):
				radius[yy, xx] = rad
				if xcoord != 0:
					costheta[yy, xx] = (np.sign(xcoord) *
						np.cos(np.arctan((ycoord * incl) / xcoord)) )
				else:
					costheta[yy, xx] = (np.sign(xcoord) *
						np.cos(np.sign(ycoord) * np.pi * 0.5e0) )
			else:
				radius[yy, xx] = float('nan')							#no data outside galaxy radius
				costheta[yy, xx] = float('nan')							#further routines will conserve NaN
	R_opt = dim / 4.e0													#define image to cover 2 optical radii						
	return radius, costheta, R_opt

def flat2exp(radius, R_0, R_e, R_opt, MT):
	"""
	Creates a 2D HI mass map with a raidal distribution which 
	transitions from constant surface density to an exponential decline

    Parameters
    ----------
    radius : N x N array 	[pixels]
        2D array of galactocentric radii
    R_0 : float 	[1 / R_opt]
    	Radius where the radial distribution transitions from flat to exponential
    R_e : float 	[1 / R_opt]
    	Scale length of exponential decline
    R_opt : float 	[pixels]
    	Optical radius
    MT : float 	[Msun]
    	Total HI mass

    Returns
    -------
	mom0 : N x N array 	[Msun / pixel]
		2D array of HI mass in each pixel
    """

	R_0 = R_0 * R_opt
	R_e = 1.e0 / (R_e * R_opt)
	mom0  = np.exp(-1.e0 * (radius - R_0) * R_e)
	if np.isscalar(R_0) == False:
		mom0[np.where((radius < R_0) == True)] = 1.e0
	else:
		mom0[np.where(radius < R_0)] = 1.e0
	mom0 = mom0 * (MT / np.nansum(mom0))
	return mom0

def polyex_RC(radius, costheta, V0, scalePE, R_opt, aa, incl):
	"""
	Creates a 2D projected velocity map using the Polyex rotation curve (RC) defined 
	by Giovanelli & Haynes 2002, and used by Catinella, Giovanelli & Haynes 2006

    Parameters
    ----------
    radius : N x N array 	[pixels]
        2D array of galactocentric radii
    costheta : N x N array
        2D array of cos(theta) values from the receding major axis
    V_0 : float 	[km/s]
    	Amplitude of RC
    scalePE : float 	[1 / R_opt]
    	Scale length of exponential inner RC
    R_opt : float 	[pixels]
    	Optical radius
    aa : float
    	Slope of outer, linear part of RC
    incl : float 	[deg]
    	Galaxy inclination

    Returns
    -------
	mom1 : N x N array 	[km/s]
		2D array of inclination corrected rotational velocity of each pixel
	"""

	incl = np.sin(incl * (np.pi / 180.e0))
	R_PE = 1.e0 / (scalePE * R_opt)											#rotation curve scale length Catinella+06
	mom1 = ( (V0 * (1.e0 - np.exp((-1.e0 * radius) * R_PE)) * 
		(1.e0 + aa * radius * R_PE)) * costheta * incl )
	return mom1

def size_mass_relation(MT):
	"""
	Calculates the size of the HI disk using the Wang+16 HI size-mass relation

    Parameters
    ----------
    MT : float 	[Msun]
        Total HI mass

    Returns
    -------
	DHI : float [pc]
		Diameter where the HI surface density equals 1 Msun/pc 
	"""

	DHI = 10.e0 ** (0.506 * np.log10(MT) - 3.293e0)
	DHI = DHI * 1.e3  						#convert to pc
	return DHI

def create_mom0(radius, costheta, params, R_opt):
	"""
	Generates a 2D HI mass map for symmetric or asymmetric distribution inputs

    Parameters
    ----------
    radius : N x N array 	[pixels]
        2D array of galactocentric radii
    costheta : N x N array
        2D array of cos(theta) values from the receding major axis
    params : list
    	List of model input parameters
    		params[0] = Galaxy inclination 	[deg]
	    	params[1] = Model type
	    	params[2] = Asymmetry flag
	    	params[3] = Total HI mass 		[Msun]
	    	params[4:7] = Receding side / symmetric input parameters
	    	params[8:11] = Approaching side parameters
	R_opt : float 	[pixels]
    	Optical radius 

    Returns
    -------
	mom0_map : N x N array 	[Msun/pc^2]
		2D array of projected HI surface densities
	rad1d : array 	[1 / R_opt]
		Radii bins for measured radial HI surface densities
	hi_profile : 2 element list of arrays 	[Msun/pc^2]
		Radial projected HI surface density profiles of 
		receding and approaching side respectively
	"""

	dim  = len(radius)
	mom0_map = np.zeros([dim,dim])
	hi_model = params[1]
	flag_asymhi = params[2]						
	MT = params[3]								
	if flag_asymhi ==0:							#no asymmetries
		p1 = params[4]
		p2 = params[5]
		p3 = params[6]
		p4 = params[7]
	elif flag_asymhi == 1:						#asymmetric distribution
		p1_rec = params[4]
		p2_rec = params[5]
		p3_rec = params[6]
		p4_rec = params[7]
		p1_app = params[8]
		p2_app = params[9]
		p3_app = params[10]
		p4_app = params[11]						#linearly interpolate over azimuth for smooth
												#variation of approaching/receeding parameters
		p1 = p1_app * (1.e0 + (((p1_rec - p1_app)/p1_app) * 0.5e0* (costheta + 1.e0)))
		p2 = p2_app * (1.e0 + (((p2_rec - p2_app)/p2_app) * 0.5e0* (costheta + 1.e0)))
		p3 = p3_app * (1.e0 + (((p3_rec - p3_app)/p3_app) * 0.5e0* (costheta + 1.e0)))
		p4 = p4_app * (1.e0 + (((p4_rec - p4_app)/p4_app) * 0.5e0* (costheta + 1.e0)))
	else:
		print('Invalid HI asymmetry flag')
		exit()
	if hi_model == 'DG':
		mom0_map = double_gauss(radius, p1, p2, p3, p4, R_opt, MT)
	if hi_model == 'SG':
		mom0_map = single_gauss(radius, p1, p2, p3, R_opt, MT)
	if hi_model == 'P':
		mom0_map = polynomial(radius, p1, p2, p3, p4, R_opt, MT)
	if hi_model == 'BB':
		mom0_map = BB_universal(radius, p1, R_opt, MT)
	if hi_model == 'FE':
		mom0_map = flat2exp(radius, p1, p2, R_opt, MT)	

	incl  = np.cos(params[0] * np.pi / 180.e0)							#measure radial HI distribution
	Rstep = (0.5e0 * dim) / 50.
	rad1d = np.arange(0, int((dim) / 2) + 2.e0 * Rstep, Rstep)
	hi_receding = np.arange(len(rad1d) - 1)
	hi_approaching = np.arange(len(rad1d) - 1)

	radius_temp = np.zeros([dim,dim])									#make approaching side have negative 
	radius_temp[:, 0:int(dim / 2)] = -1.e0 * radius[:, 0:int(dim / 2)]	#radius to make summing easier
	radius_temp[:, int(dim / 2)::] = radius[:, int(dim / 2)::]
	for bb in range(len(rad1d) - 1):
		bin_low = rad1d[bb]
		bin_high  = rad1d[bb + 1]
		inbin_app = mom0_map[(radius_temp <= -1.e0 * bin_low) & (radius_temp >
					 -1.e0 * bin_high)]
		inbin_rec = mom0_map[(radius_temp >= bin_low) & (radius_temp < bin_high)]

		hi_approaching[bb] = np.nansum(inbin_app) * incl / len(inbin_app)
		hi_receding[bb] = np.nansum(inbin_rec) * incl / len(inbin_rec)		#inclination corrected 
	rad1d = rad1d[0:-1] / R_opt

	DHI = size_mass_relation(MT)				# disk size from size-mass relation (Wang+16)
	pc_pix = DHI / len(radius)					# assumes 1 Msun/pc^2 occurs at 2 R_opt
	pixarea_pc = pc_pix * pc_pix
	hi_profile = [hi_receding / pixarea_pc, hi_approaching / pixarea_pc]
	mom0_map = mom0_map / pixarea_pc

	return mom0_map, rad1d , hi_profile

def create_mom1(radius, costheta, rad1d, params, R_opt):
	"""
	Generates a 2D gas velocity map for symmetric or asymmetric distribution inputs

    Parameters
    ----------
    radius : N x N array 	[pixels]
        2D array of galactocentric radii
    costheta : N x N array
        2D array of cos(theta) values from the receding major axis
    rad1d : array [1 / R_opt]
    	Radii bins for measuring input rotation curve
    params : list
    	List of model input parameters
    		params[0] = Galaxy inclination 	[deg]
	    	params[12] = Asymmetry flag
	    	params[13:15] = Receding side / symmetric input parameters
	    	params[16:18] = Approaching side parameters
	    	params[19] = Velocity dispersion 	[km/s]
	R_opt : float 	[pixels]
    	Optical radius 

    Returns
    -------
	mom1_map : N x N array 	[km/s]
		2D array of projected gas rotational velcoities
	input_RC : array 	[km/s]
		Projected input rotation curve
	"""
	
	dim = len(radius)
	mom1_map = np.zeros([dim,dim])
	flag_asymrc = params[12]						#flag for asymmetic RC
	incl = params[0]
	Vdisp = params[19]							#velocity dispersion

	if flag_asymrc == 0:						#RC is symmetric
		V0 = params[13]
		scalePE = params[14]
		alpha = params[15]

		RC = polyex_RC(rad1d * R_opt, 1.e0, V0, scalePE, R_opt, alpha, incl)
		input_RC = [RC, RC]
	elif flag_asymrc == 1:						#RC is asymmetric
		V0_rec = params[13]
		scalePE_rec = params[14]
		alpha_rec = params[15]
		V0_app = params[16]
		scalePE_app = params[17]
		alpha_app = params[18]

		V0 = V0_app * (1.e0 + (((V0_rec - V0_app) / V0_app) 
				* 0.5e0 * (costheta + 1.e0)))
		scalePE = scalePE_app * (1.e0 + (((scalePE_rec - scalePE_app) / scalePE_app)
				* 0.5e0 * (costheta + 1.e0)))
		alpha = alpha_app * (1.e0 + (((alpha_rec - alpha_app) / alpha_app) 
				* 0.5e0 * (costheta + 1.e0)))

		RC_rec = polyex_RC(rad1d * R_opt, 1.e0, V0_rec, scalePE_rec, R_opt, alpha_rec, incl)
		RC_app = polyex_RC(rad1d * R_opt, 1.e0, V0_app, scalePE_app, R_opt, alpha_app, incl)
		input_RC = [RC_rec, RC_app]
	else: 
		print('Invalic RC asymmetry flag')
		exit()
	mom1_map = polyex_RC(radius, costheta, V0, scalePE, R_opt, alpha, incl)

	if Vdisp >= 0:								#add velocity dispersion
		mom1_map = np.random.normal(mom1_map, Vdisp)

	return mom1_map, input_RC

def hi_spectra(mom0, mom1, params):
	"""
	Generate obseved HI spectrum from mom0 and mom1 maps

    Parameters
    ----------
    mom0 : N x N array 	[Msun/pc^2]
        2D array of projected gas surface densities
    mom1 : N x N array 	[km/s]
        2D array of projected gas rotational velocities
    params : list
    	List of model input parameters
    		params[3] = Total HI mass 	[Msun]
	    	params[20] = Distance to galaxy 	[Mpc]
	    	params[22] = Minimum measurement velocity 	[km/s]
	    	params[23] = Maximum measurement velocity 	[km/s]
	    	params[24] = Measurement velocity resolution [km/s]

    Returns
    -------
	vel_bins : array 	[km/s]
		Observed velocity axis
	spectrum : array 	[mJy]
		Noiseless HI spectrum 
	Sint : float 	[mJy km/s]
		Integrated flux of spectrum 
	"""

	MT = params[3]
	dist = params[20]							
	Vmin  = params[22]
	Vmax  = params[23]
	Vstep  = params[24]							
	DHI = size_mass_relation(MT)				#disk size from size-mass relation (Wang+16)
	pc_pix = DHI / len(mom0)
	pixarea_pc = pc_pix * pc_pix	

	Vcent = 0.5e0 * (Vmax + Vmin)
	mom1 = mom1 + Vcent
	vel_bins = np.arange(Vmin, Vmax , Vstep)
	spectrum = np.zeros(len(vel_bins))
	mjy_conv = 1.e3 / (2.356e5 * Vstep * (dist ** 2.e0))			#convert to mJy 

	for vel in range(len(vel_bins) - 1):
		inbin = mom0[(mom1 >= vel_bins[vel]) & (mom1 < vel_bins[vel + 1] )]
		spectrum[vel] = np.nansum(inbin) * pixarea_pc * mjy_conv  
	Sint = mjy_conv * MT * Vstep
	return vel_bins, spectrum, Sint

def add_noise(spectrum, params):
	"""
	Add noise to noiseless HI spectrum

    Parameters
    ----------
    spectrum : array 	[mJy]
		Noiseless HI spectrum 
	params : list
		List of model input parameters
			params[21] = input RMS measurement noise [mJy]

    Returns
    -------
	obs_spectrum : array 	[mJy]
		Observed spectrum with measurement noise 
	"""

	RMS = params[21]
	noise_arr = np.random.normal(np.zeros(len(spectrum)), RMS)
	obs_spectrum = spectrum + noise_arr
	return obs_spectrum

def smooth_spectra(vel_bins, spectrum, params):
	"""
	Boxcar smooth observed spectrum 

    Parameters
    ----------
    vel_bins : array 	[km/s]
		Observed velocity axis 
	spectrum : array 	[mJy]
		Observed spectrum with RMS noise added 
	params : list
		List of model input parameters
			params[24] = Measurement velocity resolution	[km/s]
			params[25] = Smoothed velocity resolution		[km/s]

    Returns
    -------
	smoothed_spectrum : array
		Smoothed observed spectrum 	[mJy]
	"""

	Vres = params[24]
	Vsm = params[25]
	if Vsm > 0:
		box_channels = int(Vsm / Vres)
		smoothed_spectrum = convolve(spectrum, Box1DKernel(box_channels)) 
	else:
		smoothed_spectrum = spectrum
	return smoothed_spectrum

def StN(Sint, width, rms, Vsm):
	"""
	Signal to noise as defined by ALFALFA

    Parameters
    ----------
    Sint : float 	[mJy km/s]
		Integrated flux 	
    width : float 	[km/s]
		Width of spectrum with same limits as Sint 	
    rms : float 	[mJy]
		RMS measurement noise 	
    Vsm : float 	[km/s]
		Smoothed velocity resolution	

    Returns
    -------
	SN : float
		Signal to noise of spectrum
	"""

	SN = ((Sint / width) / rms) * np.sqrt(0.5e0 * width / Vsm)
	return SN

def rms_from_StN(SN, Sint, width, Vsm):
	"""
	Convert a signal to noise to an RMS value

    Parameters
    ----------
    SN : float 
    	Signal to noise of spectrum
    Sint : float 	[mJy km/s]
		Integrated flux of spectrum
    width : float 	[km/s]
		Width of spectrum with same limits as Sint
    Vsm : float 	[km/s]
		Smoothed velocity resolution

    Returns
    -------
    rms : float 	[mJy]
		RMS measurement noise
	"""

	rms = ((Sint / width) / SN) * np.sqrt(0.5e0 * width / Vsm)
	return rms

def width_from_StNAA(SN, Sint, rms, Vsm):
	"""
	Convert an ALFALFA signal to noise to a velocity width

    Parameters
    ----------
    SN : float 
    	Signal to noise of spectrum
    Sint : float 	[mJy km/s]
		Integrated flux of spectrum
    rms : float 	[mjy]
		RMS measurement noise
    Vsm : float 	[km/s]
		Smoothed velocity resolution

    Returns
    -------
    width : float 	[km/s]
		width of profile
	"""

	width = ((Sint / rms) * (1. / np.sqrt(2. * Vsm)) * (1. / SN))**2.e0 
	return width

def THwidth_from_StNPN(SN, Sint, rms, Vres):
	"""
	Convert peak signal to noise to a velocity width for a top-hat profile
	Parameters
	----------
	SN : float
		Signal to noise of spectrum
	Sint : float 	[mJy km/s]
		Integrated flux of spectrum
	rms : float 	[mjy]
		RMS measurement noise

	Returns
	width : float [km/s]
		Width of corresponding top-hat spectrum
	"""
	peak = SN * rms - rms
	width = (Sint / peak)
	return width

def GSwidth_from_StNPN(SN, Sint, rms):
	"""
	Convert peak signal to noise to standard deviation for a Gaussian profile
	Parameters
	----------
	SN : float
		Signal to noise of spectrum
	Sint : float 	[mJy km/s]
		Integrated flux of spectrum
	rms : float 	[mjy]
		RMS measurement noise

	Returns
	sigma : float [km/s]
		Standard deviation of corresponding Gaussian spectrum
	"""
	peak = SN * rms - rms
	sigma = Sint / (peak * np.sqrt(2 * np.pi))
	return sigma

def Gaussian_CDF(x, mu, sigma):
	"""
	Return the cumpative probability of a Gaussian distribution 

    Parameters
    ----------
    x: array
    	Value to be evaluated at
    mu : float 	
		Mean
    sigma : float 	
		Standard deviation

    Returns
    -------
    C : float / array 
		Probability of < x
	"""
	C = 0.5 * (1 + erf( (x - mu) / (np.sqrt(2.e0) * sigma) ))
	return C

def Gaussian_PDF(vel, mu, sigma, alpha):
	"""
	Return the probability density of a Gaussian distribution 

    Parameters
    ----------
    vel: array
    	Observed velocities
    mu : float 	
		Mean
    sigma : float 	
		Standard deviation
	alpha : float
		Skewness

    Returns
    -------
    G : float / array 
		Probability density of variable(s)
	"""
	G = 2.e0 * np.exp(-1.e0 * ((vel - mu) * (vel - mu))/(2.e0 * sigma * sigma)) * Gaussian_CDF(alpha * vel , mu, sigma)
	return G

def Tophat(vel, width):
	"""
	Return a Tophat

    Parameters
    ----------
    vel: array
    	Observed velocities 
 	width : int 	[km/s]
 		Width of Tophat window

    Returns
    -------
    T : array 
		Array with 1's in region defined by width and 0s elsewhere
	"""

	width = width / np.abs(np.diff(vel)[0])
	T = np.zeros(len(vel))
	if int((0.5 * (len(vel) - width))) == int((0.5 * (len(vel) + width))):
		T[int((0.5 * (len(vel) - width))):int((0.5 * (len(vel) + width))) + 1] = 1.e0
	else:
		T[int((0.5 * (len(vel) - width))):int((0.5 * (len(vel) + width)))] = 1.e0
	return T

def locate_peaks(spectrum):
	"""
	Locate peaks in a nosieless spectrum by iterating from the outsides of the profile inward

	Parameters
	----------
	spectrum : array
		Input spectrum

	Returns
	-------
	peaks : list
		List of outermost peak locations in channels
	"""

	chan = 0
	while(chan < len(spectrum)-1 and (spectrum[chan] - spectrum[chan - 1]) * 
			(spectrum[chan + 1] - spectrum[chan]) > -1.e-10):
		chan += 1
		PeaklocL = chan
	if chan == len(spectrum):
		PeaklocL = -1
	
	chan = len(spectrum) - 2
	while(chan > 0 and (spectrum[chan] - spectrum[chan - 1]) * 
			(spectrum[chan + 1] - spectrum[chan]) > -1.e-10):
		chan -= 1
		PeaklocR = chan
	if chan == 0:
		PeaklocR = -1
	
	peaks = [PeaklocL,PeaklocR]
	return peaks

def locate_width(spectrum, peaks, level):
	"""
	Locate the N% level of the peak on the left and right side of a spectrum

	Parameters
	----------
	spectrum : array
		Input spectrum
	peaks : list
		Value of each peak
	level : float
		N% of the peaks to measure

	Returns
	-------
	Wloc : list
		Location of N% of each peak in channels
	"""

	# channels = range(len(spectrum))	
	SpeakL = peaks[0]
	SpeakR = peaks[1]
	wL = -1
	wR = -1
	chan = 0
	while(chan < len(spectrum)-1 and spectrum[chan] < level * SpeakL):
		chan += 1
		wL = chan - 1 + ((level * SpeakL - spectrum[chan - 1]) / (
			spectrum[chan] - spectrum[chan - 1])) 

	chan = len(spectrum) - 2
	while(chan > 0 and spectrum[chan] < level * SpeakR):
		chan -= 1
		wR = chan + 1 + -1.e0 * ((level * SpeakR - spectrum[chan + 1]) / (
			spectrum[chan] - spectrum[chan + 1])) 

	Wloc = [wL,wR]
	return Wloc

def areal_asymmetry(spectrum, limits, Vres):
	"""
	Measure the asymmetry parameter and integrated flux of a spectrum between limits

	Parameters
	----------
	spectrum : array
		Input spectrum to measure
	limits : list
		Lower and upper channels to measure between
	Vres : float
		Velocity resolution

	Returns
	-------
	Sint : float
		Integrated flux in units of input spectrum * Vres
	Afr : float
		Asymmetry parameter
	"""

	minchan = limits[0]
	maxchan = limits[1]
	midchan = 0.5e0 * (minchan + maxchan)
	min_val = np.interp(minchan,[np.floor(minchan),np.ceil(minchan)],
			[spectrum[int(np.floor(minchan))],spectrum[int(np.ceil(minchan))]])
	max_val = np.interp(maxchan,[np.floor(maxchan),np.ceil(maxchan)],
			[spectrum[int(np.floor(maxchan))],spectrum[int(np.ceil(maxchan))]])
	mid_val = np.interp(midchan,[np.floor(midchan),np.ceil(midchan)],
			[spectrum[int(np.floor(midchan))],spectrum[int(np.ceil(midchan))]])

	Sedge_L = min_val * (np.ceil(minchan) - minchan)
	Sedge_R = max_val * (maxchan - np.floor(maxchan))
	Smid_L = mid_val * (midchan - np.floor(midchan))
	Smid_R = mid_val * (np.ceil(midchan) - midchan )

	S_L = spectrum[int(np.ceil(minchan)):int(np.floor(midchan) + 1)]
	# S_L = S_L[S_L > 0]
	S_L = np.nansum(S_L)
	S_R = spectrum[int(np.ceil(midchan)):int(np.floor(maxchan) + 1)]
	# S_R = S_R[S_R > 0]
	S_R = np.nansum(S_R)

	Sint_L = (Sedge_L + S_L + Smid_L) * Vres
	Sint_R = (Sedge_R + S_R + Smid_R) * Vres
	Sint = Sint_L + Sint_R

	Afr =  Sint_L / Sint_R 
	if Afr < 1.e0:
		Afr = 1.e0 / Afr

	return Sint, Afr

##########################################################################################
#	Argument parsing for using functions
##########################################################################################

global_parser = argparse.ArgumentParser(description = __doc__,
					formatter_class = argparse.ArgumentDefaultsHelpFormatter)
subparsers = global_parser.add_subparsers()

generation_parser = subparsers.add_parser('generate', help = 'Generate model spectra',
					formatter_class = argparse.ArgumentDefaultsHelpFormatter)
generation_parser.add_argument('-rms','--rms', action = 'store_true',
					help = 'Flag for changing RMS models', default = True)
generation_parser.add_argument('-width','--width', action = 'store_true',
					help = 'Flag for changing width models', default = False)
generation_parser.add_argument('-Nch','--Nch', action = 'store_true',
					help = 'Flag for changing channel width models', default = False)
generation_parser.add_argument('-N', '--Nmodels', type = int, nargs = 1,
					help = 'Number of spectra in each model', default = [100])
generation_parser.add_argument('-T', '--TH', type = float, nargs = 1,
					help = 'Width for an input top-hat spectrum', default = [0])
generation_parser.add_argument('-G', '--GS', type = float, nargs = 2,
					help = 'Sigma for an input Gaussian', default = [0,0])
generation_parser.add_argument('-i', '--incl', type = float,
					help = 'Galaxy inclination', default = 50.e0 )
generation_parser.add_argument('-Vs', '--Vsm', type = float,
					help = 'Velocity smoothing', default = 10.e0 )
generation_parser.add_argument('-Vd', '--Vdisp', type = float,
					help = 'Velocity dispersion', default = 0.e0 )
generation_parser.add_argument('-H', '--HI', nargs = '+', type = float,
					help = 'HI distribution parameters', default = [1.e0, 1.65e0])
generation_parser.add_argument('-R', '--RC', nargs = '+', type = float, 
					help = 'Rotation curve parameters', default = [200.e0, 0.164e0, 0.002e0])
generation_parser.add_argument('-SN', '--SN_range', nargs = 3, type = float,
					help = 'S/N max, min & step', default = [10, 50, 5])
generation_parser.add_argument('-PN','--PN', action='store_true',
					help = 'Flag to indicate Peak/rms S/N definion', default = False)
generation_parser.add_argument('-AA','--AA', action='store_true',
					help = 'Flag to indicate ALFALFA S/N definion', default = True)
generation_parser.add_argument('-opt', '--option', type = str,
					help = 'Optional directory name', default = '')
generation_parser.set_defaults(func=generate)


measurement_parser = subparsers.add_parser('measure', help = 'Measure asymmetries of models',
					formatter_class = argparse.ArgumentDefaultsHelpFormatter)
measurement_parser.add_argument('dir', metavar = 'directory',
					help = 'Model directory of measurements')
measurement_parser.set_defaults(func = measure)


statistics_parser = subparsers.add_parser('statistics',
					help = 'Calculate asymmetry distribution statistics',
					formatter_class = argparse.ArgumentDefaultsHelpFormatter)
statistics_parser.add_argument('dir', metavar = 'directory',
					help = 'Model directory of measurements')
statistics_parser.set_defaults(func = statistics)


plot_spectrum_parser = subparsers.add_parser('plot', help='Plot model spectrum',
					formatter_class = argparse.ArgumentDefaultsHelpFormatter)
plot_spectrum_parser.add_argument('dir', metavar = 'directory',
					help = 'Model directory of measurements')
plot_spectrum_parser.add_argument('-SN', '--SN', type = int,
					help = 'S/N of model',default = 20)
plot_spectrum_parser.add_argument('-N', '--num', type = int,
					help = 'Model number', default = 1)
plot_spectrum_parser.add_argument('-S', '--save', action = 'store_true',
					help = 'Flag to save plot', default = False)
plot_spectrum_parser.set_defaults(func = plot_spectrum)


density_plot_parser = subparsers.add_parser('density', help='Plot model suite density plot',
					formatter_class = argparse.ArgumentDefaultsHelpFormatter)
density_plot_parser.add_argument('dir', metavar = 'directory',
					help = 'Model directory of measurements')
density_plot_parser.add_argument('-S', '--save', action = 'store_true',
					help = 'Flag to save plot', default = False)
density_plot_parser.set_defaults(func = density_plot)


global_parser.add_argument('-v', '--version', action = 'version', version = '1.1.0')

def main():
	args = global_parser.parse_args()
	args.func(args)

if __name__ == '__main__':
	main()
