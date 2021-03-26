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



def create_arrays(dim, params)
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

	DHI = 10.e0 ** (0.506 * np.log10(0.83*MT) - 3.293e0)
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
	pc_pix = DHI / (len(radius))					# assumes 1 Msun/pc^2 occurs at 1 R_opt = 1 R_25 (Wang+14)
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
	pc_pix = DHI / (0.5*len(mom0))
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
