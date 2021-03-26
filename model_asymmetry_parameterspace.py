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
# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
import os
import glob 

from mpi4py import MPI
from astropy.table import Table
from scipy.stats import norm, mode
from functions import *

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

		file_list = glob.glob(f'{args.dir}*_spectra.dat')

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

		width_rms = locate_width(spectrum,[1,1],rms)
		width_2rms = locate_width(spectrum,[1,1],2.e0 * rms)
		width_20 = locate_width(spectrum, Peaks, 0.2e0)
		width_50 = locate_width(spectrum, Peaks, 0.5e0)

		wrms = (width_rms[1] - width_rms[0]) * Vres
		w2rms = (width_2rms[1] - width_2rms[0]) * Vres
		w20 = (width_20[1] - width_20[0]) * Vres
		w50 = (width_50[1] - width_50[0]) * Vres


		Sint_noiseless_rms, Afr_noiseless_rms = areal_asymmetry(spectrum, width_rms, Vres)
		Sint_noiseless_2rms, Afr_noiseless_2rms = areal_asymmetry(spectrum, width_2rms, Vres)
		Sint_noiseless_w20, Afr_noiseless_w20 = areal_asymmetry(spectrum, width_20, Vres)
		Sint_noiseless_w50, Afr_noiseless_w50 = areal_asymmetry(spectrum, width_50, Vres)
	
		measurements = []
		measurements_noiseless = [Sint_noiseless_rms, -1, Afr_noiseless_rms,
								Sint_noiseless_2rms, -1, Afr_noiseless_2rms,
								Sint_noiseless_w20, -1, Afr_noiseless_w20,
								-1, -1, -1,
								Sint_noiseless_w50, -1, Afr_noiseless_w50,
								-1, -1, -1
								]
		measurements.append(measurements_noiseless)
		for run in range(len(spectra[0]) - 2):
			obs_spectrum = spectra[:, run + 2]

			Sint_rms, Afr_rms = areal_asymmetry(obs_spectrum, width_rms, Vres)
			Sint_2rms, Afr_2rms = areal_asymmetry(obs_spectrum, width_2rms, Vres)
			Sint_w20, Afr_w20 = areal_asymmetry(obs_spectrum, width_20, Vres)
			Sint_w50, Afr_w50 = areal_asymmetry(obs_spectrum, width_50, Vres)
			
			if SN_type == 'PN':
				SN_rms = np.nanmax(obs_spectrum) / rms
				SN_2rms = np.nanmax(obs_spectrum) / rms
				SN_w20 = np.nanmax(obs_spectrum) / rms
				SN_w50 = np.nanmax(obs_spectrum) / rms
			elif SN_type == 'AA':
				SN_rms = StN(Sint_rms, wrms, rms, Vsm)
				SN_2rms = StN(Sint_2rms, w2rms, rms, Vsm)
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

			spectrum_measurements = [
				Sint_rms, SN_rms, Afr_rms,
				Sint_2rms, SN_2rms, Afr_2rms,
				Sint_w20, SN_w20, Afr_w20,
				Sint_w20_obs, SN_w20_obs, Afr_w20_obs,
				Sint_w50, SN_w50, Afr_w50,
				Sint_w50_obs, SN_w50_obs, Afr_w50_obs
				]

			measurements.append(spectrum_measurements)
			# measurements[run + 1,:] = [Sint_w20, SN_w20, SN_w50, SN_w20_obs,
			# 	SN_w50_obs, Afr_w20, Afr_w50, Afr_w20_obs, Afr_w50_obs]
		measurements = np.array(measurements)
		base = file.split('_spectra.dat')[0]
		filename = f'{base}_measured.dat'
		header = f'rms = {rms:0.5f}\nVsm = {Vsm}\n'
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

		# Sint_rms = np.array([])
		# SN_rms = np.array([])
		# Afr_rms = np.array([])
		# Sint_2rms = np.array([])
		# SN_2rms = np.array([])
		# Afr_2rms = np.array([])
		# Sint_w20 = np.array([])
		# SN_w20 = np.array([])
		# Afr_w20 = np.array([])
		# Sint_w20_obs = np.array([])
		# SN_w20_obs = np.array([])
		# Afr_w20_obs = np.array([])
		# Sint_w50 = np.array([])
		# SN_w50 = np.array([])
		# Afr_w50 = np.array([])
		# Sint_w50_obs = np.array([])
		# SN_w50_obs = np.array([])
		# Afr_w50_obs = np.array([])

		names = ['rms', '2rms', 'w20', 'w20_obs', 'w50', 'w50_obs']

		column_names = ['Nch', 'Vres']
		formats = {'Nch':'4.2f', 'Vres': '3.4f'}

		dP = 5
		percentile_vals = np.arange(5, 95 + dP, dP)
		percentiles = np.zeros(len(percentile_vals) + 1)

		# names = ['w20', 'w50', 'w20_obs', 'w50_obs']
		for name in names:
			column_names.extend([f'avg_SN_{name}'])						
			formats[f'avg_SN_{name}'] = '4.2f'

			for pp in range(len(percentile_vals)):
				value = int(percentile_vals[pp])
				column_names.extend([f'P{value}_{name}'])
				formats[f'P{value}_{name}'] = '.5f'
		
		Afr_statistics = np.zeros([len(file_list), 2 + len(percentiles) * len(names)])
		Afr_statistics[:,0] = Nch_range
		Afr_statistics[:,1] = Vres_range

		for ff in range(len(file_list)):
			measurements = np.genfromtxt(file_list[ff])
			measurements = measurements[1::,:]
			Sint_rms = measurements[:, 0]
			SN_rms = measurements[:, 1]
			Afr_rms = measurements[:, 2]

			Sint_2rms = measurements[:, 3]
			SN_2rms = measurements[:, 4]
			Afr_2rms = measurements[:, 5]
			
			Sint_w20 = measurements[:, 6]
			SN_w20 = measurements[:, 7]
			Afr_w20 = measurements[:, 8]


			Sint_w20_obs = measurements[:, 9]
			SN_w20_obs = measurements[:, 10]
			Afr_w20_obs = measurements[:, 11]

			Sint_w50 = measurements[:, 12]
			SN_w50 = measurements[:, 13]
			Afr_w50 = measurements[:, 14]
			
			Sint_w50_obs = measurements[:, 15]
			SN_w50_obs = measurements[:, 16]
			Afr_w50_obs = measurements[:, 17]

			SN = [SN_rms, SN_2rms, SN_w20, SN_w20_obs, SN_w50, SN_w50_obs]
			Afr = [Afr_rms, Afr_2rms, Afr_w20, Afr_w20_obs, Afr_w50, Afr_w50_obs]

			for ss in range(len(SN)):
				percentiles[0] = np.mean(SN[ss])		
		
				for pp in range(len(percentile_vals)):
					percentiles[pp + 1] = np.percentile(Afr[ss], percentile_vals[pp])
		
				Afr_statistics[ff, (ss * len(percentiles) + 2):
								((ss + 1) * len(percentiles) + 2)] = percentiles

		table = Table(rows = Afr_statistics, names = column_names)
		stats_filename = f'{args.dir}statistics.dat'
		table.write(stats_filename, formats = formats, format = 'ascii', overwrite = True)

	else:	
		Sint_rms = np.array([])
		SN_rms = np.array([])
		Afr_rms = np.array([])
		Sint_2rms = np.array([])
		SN_2rms = np.array([])
		Afr_2rms = np.array([])
		Sint_w20 = np.array([])
		SN_w20 = np.array([])
		Afr_w20 = np.array([])
		Sint_w20_obs = np.array([])
		SN_w20_obs = np.array([])
		Afr_w20_obs = np.array([])
		Sint_w50 = np.array([])
		SN_w50 = np.array([])
		Afr_w50 = np.array([])
		Sint_w50_obs = np.array([])
		SN_w50_obs = np.array([])
		Afr_w50_obs = np.array([])

		for file in file_list:
			measurements = np.genfromtxt(file)
			measurements = measurements[1::,:]

			Sint_rms = np.append(Sint_rms,measurements[:, 0])
			SN_rms = np.append(SN_rms,measurements[:, 1])
			Afr_rms = np.append(Afr_rms,measurements[:, 2])

			Sint_2rms = np.append(Sint_2rms,measurements[:, 3])
			SN_2rms = np.append(SN_2rms,measurements[:, 4])
			Afr_2rms = np.append(Afr_2rms,measurements[:, 5])
			
			Sint_w20 = np.append(Sint_w20,measurements[:, 6])
			SN_w20 = np.append(SN_w20,measurements[:, 7])
			Afr_w20 = np.append(Afr_w20,measurements[:, 8])

			Sint_w20_obs = np.append(Sint_w20_obs,measurements[:, 9])
			SN_w20_obs = np.append(SN_w20_obs,measurements[:, 10])
			Afr_w20_obs = np.append(Afr_w20_obs,measurements[:, 11])

			Sint_w50 = np.append(Sint_w50,measurements[:, 12])
			SN_w50 = np.append(SN_w50,measurements[:, 13])
			Afr_w50 = np.append(Afr_w50,measurements[:, 14])
			
			Sint_w50_obs = np.append(Sint_w50_obs,measurements[:, 15])
			SN_w50_obs = np.append(SN_w50_obs,measurements[:, 16])
			Afr_w50_obs = np.append(Afr_w50_obs,measurements[:, 17])

		SN_bins = np.arange(int(np.nanmax([5, np.nanmin(SN_w20)])), 
				int(np.nanmax(SN_w20) * 0.1) * 10, 4)
		Afr_densitybins = np.arange(1., 3, 0.05)
		Afr_densityplot = np.zeros([len(Afr_densitybins) - 1, len(SN_bins) - 1])


		dP = 5
		percentile_vals = np.arange(5, 95 + dP, dP)
		percentiles = np.zeros(len(percentile_vals) + 2)

		names = ['rms', '2rms', 'w20', 'w20_obs', 'w50', 'w50_obs']
		SN = [SN_rms, SN_2rms, SN_w20, SN_w20_obs, SN_w50, SN_w50_obs]
		Afr = [Afr_rms, Afr_2rms, Afr_w20, Afr_w20_obs, Afr_w50, Afr_w50_obs]

		Afr_statistics = np.zeros([len(SN_bins) - 1, 1 + len(percentiles) * len(SN)])
		Afr_statistics[:,0] = SN_bins[0:-1]

		column_names = ['SN_bin']
		formats = {'SN_bin':'4.2f'}

		for mm in range(len(SN)):
			SN_vals = SN[mm]
			Afr_vals = Afr[mm]

			column_names.extend([f'avg_SN_{names[mm]}',
								f'mode_Afr_{names[mm]}'])
			formats[f'avg_SN_{names[mm]}'] = '4.2f'
			formats[f'mode_Afr_{names[mm]}'] = '4.2f'

			for pp in range(len(percentile_vals)):
				value = int(percentile_vals[pp])
				column_names.extend([f'P{value}_{names[mm]}'])
				formats[f'P{value}_{names[mm]}'] = '.5f'

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
		stats_filename = f'{args.dir}statistics.dat'
		table.write(stats_filename, formats = formats, format = 'ascii', overwrite = True)
		
		# density_filename = f'{args.dir}densityplot.dat'
		# np.savetxt(density_filename, Afr_densityplot)

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

	if False:
		fig = plt.figure(figsize = (10,10))
		gs = gridspec.GridSpec(1, 1)
		spec_ax = fig.add_subplot(gs[0,0])

		spec_ax.tick_params(axis = 'both', which = 'both', direction = 'in', labelsize = 18)
		spec_ax.set_ylabel('Spectral Flux [mJy]', fontsize = 20)
		spec_ax.set_xlabel('Velocity [km/s]', fontsize = 20)
		spec_ax.plot(vel_bins,obs_spectrum, color = 'Black', zorder = -1, ls = '-')
		spec_ax.text(0.05,0.9,f'S/N = {args.SN}',fontsize=20,transform=spec_ax.transAxes)
		spec_ax.text(0.05,0.85,f'A$_{{fr}}$ = {Afr:.2f}',fontsize=20,transform=spec_ax.transAxes)
		plt.tight_layout()
		plt.show()

	if True:
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
#	Now in functions.py

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
