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

#### global variables
badflag = -123456						#flag for pixels not covered by model

default_params = {'dim':1000, 'incl':60, 'MHI':1.e9, 'dist':50,
				'HImod':'FE', 'HIparams':[1,2.35],
				'RCparams':[200,0.164,0.002], 'Vdisp':10,
				'Vlim':400, 'Vres':2, 'Vsm':10, 'RMS':0}

# default_params = {'dim':1000, 'incl':60, 'MHI':1.e9, 'dist':50,
# 				'HImod':'FE', 'HIparams':[[1,1],[2.35,1.35]],
# 				'RCparams':[200,0.164,0.002], 'Vdisp':7,
# 				'Vlim':400, 'Vres':5, 'Vsm':10, 'RMS':0}

# default_params = {'dim':1000, 'incl':60, 'MHI':1.e9, 'dist':50,
# 				'HImod':'FE', 'HIparams':[1,2.35],
# 				'RCparams':[[200,150],[0.164,0.164],[0.002,0.002]], 'Vdisp':7,
# 				'Vlim':400, 'Vres':5, 'Vsm':10, 'RMS':0}


def mock_global_HI_spectrum(params=default_params):

	if params['HImod'] == 'gaussian':
		vel_bins, spectrum = create_Gaussian_spectrum(params)
		mom0 = mom1 = mom2 = -1

	elif params['HImod'] == 'tophat':
		vel_bins, spectrum = create_Tophat_spectrum(params)
		mom0 = mom1 = mom2 = -1

	else:
		radius, costheta, Ropt = create_arrays(params)
		mom0 = create_mom0(radius, costheta, params)
		mom1 = create_mom1(radius, costheta, params)
		mom2 = create_mom2(radius, params)

		vel_bins, spectrum = create_HI_spectrum(mom0, mom1, mom2, params)

		mom0[mom0==badflag]=np.nan
		mom1[mom1==badflag]=np.nan
		mom2[mom2==badflag]=np.nan


	mJy_conv = flux_to_mJy(params)		
	spectrum *= mJy_conv
	# Sint = np.sum(spectrum) * params['Vres']

	if params['RMS'] > 0:
		spectrum = add_noise(spectrum, params)

	if params['Vsm'] != 0:
		spectrum = smooth_spectrum(spectrum, params)

	return [mom0, mom1, mom2], np.array([vel_bins,spectrum]).T

#### creation functions

def create_arrays(params):
	"""
	Creates 2D arrays of radius and angle for the HI toy model

    Parameters
    ----------
	params : list
		List of input parameters
			dim = dimension N
			incl = Galaxy inclination 	[deg]
        	
    Returns
    -------
 	radius : N x N array 	[pixels]
 		2D array of galactocentric radii
 	costheta: N x N array
 		2D array of cos(theta) = [-pi, pi] values where theta is the angle counter clockwise from
 		the receding major axis (defined as the positive x-axis)
 	Ropt : float 	[pixels]
 		Value of the optical radius in pixels defined as N/4, making Rmax = 2 Ropt
    """
	dim = params['dim']
	radius = np.zeros([dim, dim])
	costheta = np.zeros([dim, dim])
	incl = 1.e0 / np.cos(params['incl'] * np.pi / 180.e0)						#inclination correction goes as 1/cos
	Ropt = 4.e0 / dim														#define image to cover 2 optical radii						
	for yy in range(dim):
		for xx in range(dim):
			xcoord = (xx + 1.e0) - 0.5e0 * (dim + 1)
			ycoord = (yy + 1.e0) - 0.5e0 * (dim + 1)
			rad = np.sqrt( xcoord * xcoord + (ycoord * ycoord * incl * incl) )	#y coordinate is projected by inclination
			if rad <= 0.5e0 * (dim + 1.e0):
				radius[yy, xx] = rad * Ropt
				if xcoord != 0:
					costheta[yy, xx] = (np.sign(xcoord) *
						np.cos(np.arctan((ycoord * incl) / xcoord)) )
				else:
					costheta[yy, xx] = (np.sign(xcoord) *
						np.cos(np.sign(ycoord) * np.pi * 0.5e0) )
			else:
				radius[yy, xx] = badflag							#no data outside galaxy radius
				costheta[yy, xx] = badflag							#removed NaNs to speed up
	
	Ropt = dim / 4.e0 												#return Ropt in Npixels
	return radius, costheta, Ropt

def create_mom0(radius, costheta, params):
	"""
	Generates a 2D HI mass map for symmetric or asymmetric distribution inputs

    Parameters
    ----------
    radius : N x N array 	[pixels]
        2D array of galactocentric radii
    costheta : N x N array
        2D array of cos(theta) values from the receding major axis
    params : dict
    	List of model input parameters
    		HIparams = HI distribution parameters
	    	HImod = HI distribution model type

    Returns
    -------
	mom0_map : N x N array 	[Msun/pix]
		2D array of projected HI surface densities
	"""								

	if isinstance(params['HIparams'][0],list):			#check for lists of parameters - asym cases
		HIparam_arrays = []									
		for pp in range(len(params['HIparams'])):		#interpolate azimuthally
			p_rec = params['HIparams'][pp][0]
			p_app = params['HIparams'][pp][1]
			print(p_rec,p_app)
			p = p_app * (1.e0 + (((p_rec - p_app)/p_app) * 0.5e0* (costheta + 1.e0)))
			HIparam_arrays.append(p)
	else:												#symmetric case
		HIparam_arrays = params['HIparams']
	params['HIparam_arrays'] = HIparam_arrays

	if params['HImod'] == 'FE':
		mom0_map = flat2exp(radius, params)

	dist_norm = np.sum(mom0_map[mom0_map != badflag])
	mom0_map[mom0_map != badflag] *= params['MHI']/dist_norm			#normalise to total HI mass
	return mom0_map

def create_mom1(radius, costheta, params):
	"""
	Generates a 2D gas velocity map for symmetric or asymmetric distribution inputs

    Parameters
    ----------
    radius : N x N array 	[pixels]
        2D array of galactocentric radii
    costheta : N x N array
        2D array of cos(theta) values from the receding major axis
    params : dict
    	List of model input parameters
    		incl = Galaxy inclination 	[deg]
	    	RCparams = Rotation curve parameters

    Returns
    -------
	mom1_map : N x N array 	[km/s]
		2D array of projected gas rotational velcoities
	"""
	
	if isinstance(params['RCparams'][0],list):			#check for lists of parameters - asym cases
		RCparam_arrays = []									
		for pp in range(len(params['RCparams'])):		#interpolate azimuthally
			p_rec = params['RCparams'][pp][0]
			p_app = params['RCparams'][pp][1]
			p = p_app * (1.e0 + (((p_rec - p_app)/p_app) * 0.5e0* (costheta + 1.e0)))
			RCparam_arrays.append(p)
	else:												#symmetric case
		RCparam_arrays = params['RCparams']
	params['RCparam_arrays'] = RCparam_arrays
	
	mom1_map = polyex_RC(radius, costheta, params)
	return mom1_map

def create_mom2(radius, params):
	"""
	Generates a 2D velocity dispersion map for constant or radial dispersion profile

    Parameters
    ----------
    radius : N x N array 	[pixels]
        2D array of galactocentric radii
    params : dict
    	List of model input parameters
    		Vdisp = Velocity dispersion	[km/s]

    Returns
    -------
	mom2_map : N x N array 	[km/s]
		2D array of velocity dispersion
	"""

	dim  = len(radius)
	mom2_map = np.zeros([dim,dim])
	mom2_map += params['Vdisp']

	mom2_map[radius == badflag] = badflag

	return mom2_map

def create_HI_spectrum(mom0, mom1, mom2, params):
	"""
	Generate obseved HI spectrum from mom0, mom1, and mom2 maps

    Parameters
    ----------
    mom0 : N x N array 	[Msun/pix]
        2D array of projected gas surface densities
    mom1 : N x N array 	[km/s]
        2D array of projected gas rotational velocities
    mom2 : N x N array  [km/s]
    	2D array of gas velocity dispersion
    params : dict
    	List of model input parameters
    		Vres = Velocity resolution 	[km/s]
	    	Vlim = Velocity limits 		[km/s]
	    	dist = Distance to galaxy 	[Mpc]

    Returns
    -------
	vel_bins : array 	[km/s]
		Observed velocity axis
	spectrum : array 	[mJy]
		Noiseless HI spectrum 
	Sint : float 	[mJy km/s]
		Integrated flux of spectrum 
	"""

	vel_bins = np.arange(-params['Vlim'], params['Vlim'], params['Vres'])
	spectrum = np.zeros(len(vel_bins))

	mom2 = mom2[mom2 != badflag].flatten()
	mom1 = mom1[mom1 != badflag].flatten()
	mom0 = mom0[mom0 != badflag].flatten()

	for pp in range(len(mom0)):
		spectrum += mom0[pp] * Gaussian_PDF(vel = vel_bins, 		#MHI flux density in Msun s /km
											mu = mom1[pp], 
											sigma = mom2[pp])

	return vel_bins, spectrum


def create_Gaussian_spectrum(params):				
	
	vel_bins = np.arange(-params['Vlim'], params['Vlim'], params['Vres'])
	if len(params['HIparams']) == 3:
		spectrum =Gaussian_PDF(vel_bins, params['HIparams'][0],params['HIparams'][1],params['HIparams'][2])
	else:
		spectrum =Gaussian_PDF(vel_bins, params['HIparams'][0],params['HIparams'][1])
	spectrum *= params['MHI']

	return vel_bins, spectrum

def create_Tophat_spectrum(params):

	vel_bins = np.arange(-params['Vlim'], params['Vlim'], params['Vres'])
	spectrum = Tophat(vel_bins, params['HIparams'])
	spectrum *= params['MHI'] / (np.sum(spectrum)*params['Vres'])

	return vel_bins, spectrum


#### distributions and RCs

def flat2exp(radius, params):
	"""
	Creates a 2D HI mass map with a raidal distribution which 
	transitions from constant surface density to an exponential decline

    Parameters
    ----------
    radius : N x N array 	[pixels]
        2D array of galactocentric radii
   
	params : dict
		Contains HI model parameters
		MHI : float 	[Msun]
    		Total HI mass'
   		Rt : float / N x N array 	[1 / Ropt]
    		Radius where the radial distribution transitions from flat to exponential
   		Re : float / N x N array 	[1 / Ropt]
    		Scale length of exponential decline

    Returns
    -------
	mom0 : N x N array 	[Msun / pixel]
		2D array of HI mass in each pixel
    """

	Rt = params['HIparam_arrays'][0]
	Re = params['HIparam_arrays'][1]

	Re = 1.e0 / Re
	mom0  = np.exp(-1.e0 * (radius - Rt) * Re)		#distribute exponentionally (e^0 at r=Rt)
	if np.isscalar(Rt) == False:						#saturate interior to Rt	
		mom0[np.where((radius < Rt) == True)] = 1.e0	#Rt varies between sides
	else:												
		mom0[np.where(radius < Rt)] = 1.e0				#symmetric case
	mom0[radius == badflag] = badflag				#flag pixels not in model
	return mom0

def polyex_RC(radius, costheta, params,obs=True):
	"""
	Creates a 2D projected velocity map using the Polyex rotation curve (RC) defined 
	by Giovanelli & Haynes 2002, and used by Catinella, Giovanelli & Haynes 2006

    Parameters
    ----------
    radius : N x N array 	[pixels]
        2D array of galactocentric radii
    costheta : N x N array
        2D array of cos(theta) values from the receding major axis
    params : dict
    	Contains RC model parameters
	    V_0 : float / N x N array 	[km/s]
	    	Amplitude of RC
	    scalePE : float / N x N array 	[1 / Ropt]
	    	Scale length of exponential inner RC
	    aa : float / N x N array
	    	Slope of outer, linear part of RC
	    incl : float 	[deg]
	    	Galaxy inclination

    Returns
    -------
	mom1 : N x N array 	[km/s]
		2D array of inclination corrected rotational velocity of each pixel
	"""
	if obs:
		incl = np.sin(params['incl'] * (np.pi / 180.e0))
	else:
		incl = 1.e0
	V0 = params['RCparam_arrays'][0]
	R_PE = 1.e0 / params['RCparam_arrays'][1]									#rotation curve scale length Catinella+06
	aa = params['RCparam_arrays'][2]

	mom1 = ( (V0 * (1.e0 - np.exp((-1.e0 * radius) * R_PE)) * 
		(1.e0 + aa * radius * R_PE)) * costheta * incl )
	mom1[radius == badflag] = badflag									#flag pixels outside model
	return mom1



def input_HIdist(params):
	"""
	The radial profile for a given HI distribution

    Parameters
    ----------
	params : dict
		Contains HI model parameters
		MHI : float 	[Msun]
			Total HI mass'
		Rt : float		[1 / Ropt]
			Radius where the radial distribution transitions from flat to exponential
   		Re : float 		[1 / Ropt]
			Scale length of exponential decline

    Returns
    -------
	input_HI : list
		List of radii and radial HI surface density(ies)
    """

	radius = np.arange(0,2,0.005)

	if isinstance(params['HIparams'][0],list):			#check for lists of parameters - asym cases
		
		params_temp = params
		params_temp['HIparam_arrays'] = [p[0] for p in params['HIparams']]
		HIdist_rec =  flat2exp(radius,params_temp)
		params_temp['HIparam_arrays'] = [p[1] for p in params['HIparams']]
		HIdist_app =  flat2exp(radius,params_temp)

		pix_scale = ((4.e0/params['dim'])*(4.e0/params['dim'])/np.sin(params['incl']*np.pi/180.))

		Anorm_rec = np.sum(HIdist_rec*2.e0*np.pi*(radius)*np.abs(np.diff(radius)[0])) / pix_scale	#integrate 2piR(dR)(Sigma)
		Anorm_rec = params['MHI']/Anorm_rec

		Anorm_app = np.sum(HIdist_app*2.e0*np.pi*(radius)*np.abs(np.diff(radius)[0])) / pix_scale	#integrate 2piR(dR)(Sigma)
		Anorm_app = params['MHI']/Anorm_app

		input_HI = [radius,Anorm_rec*HIdist_rec,Anorm_app*HIdist_app] 	

	else:												#symmetric case
		params_temp = params
		params['HIparam_arrays'] = params['HIparams']
		HIdist = flat2exp(radius,params)

		# logDHI = Wang16_HIsizemass(np.log10(params['MHI']))
		# RHI = 0.5*(10.**logDHI) * 1.e3 					#in pc
		# Ropt = 0.66*RHI

		pix_scale = ((4.e0/params['dim'])*(4.e0/params['dim'])/np.sin(params['incl']*np.pi/180.))

		Anorm = np.sum(HIdist*2.e0*np.pi*(radius)*np.abs(np.diff(radius)[0])) / pix_scale	#integrate 2piR(dR)(Sigma)
		Anorm = params['MHI']/Anorm

		input_HI = [radius,Anorm*HIdist]


	return input_HI

	# Rt = params['HIparams'][0]
	# Re = params['HIparams'][1]

	# Re = 1.e0 / Re
	# mom0  = np.exp(-1.e0 * (radius - Rt) * Re)		#distribute exponentionally (e^0 at r=Rt)
	# if np.isscalar(Rt) == False:						#saturate interior to Rt	
	# 	mom0[np.where((radius < Rt) == True)] = 1.e0	#Rt varies between sides
	# else:												
	# 	mom0[np.where(radius < Rt)] = 1.e0				#symmetric case



	# mom0 = mom0 * (params['MHI'] / np.sum(mom0[mom0 != badflag]))	#normalise to total HI mass
	# mom0[radius == badflag] = badflag								#assign bad flags
	# return mom0

def input_RC(params):
	"""
	The rotation curve for the input model

    Parameters
    ----------
	params : dict
    	Contains RC model parameters
		V_0 : float / N x N array 	[km/s]
	    	Amplitude of RC
		scalePE : float / N x N array 	[1 / Ropt]
	    	Scale length of exponential inner RC
		aa : float / N x N array
	    	Slope of outer, linear part of RC

    Returns
    -------
	RC : list 	
		List of radii, and input RC(s)
    """

	radius = np.arange(0,2,0.01)

	if isinstance(params['RCparams'][0],list):			#check for lists of parameters - asym cases
		
		params_temp = params
		params_temp['RCparam_arrays'] = [p[0] for p in params['RCparams']]
		RC_rec =  polyex_RC(radius,1,params_temp,obs=False)
		params_temp['RCparam_arrays'] = [p[1] for p in params['RCparams']]
		RC_app =  polyex_RC(radius,1,params_temp,obs=False)

		RC = [radius,RC_rec,RC_app] 	

	else:												#symmetric case
		params_temp = params
		params['RCparam_arrays'] = params['RCparams']
		RC = polyex_RC(radius,1,params,obs=False)

		RC = [radius,RC]

	return RC

#### observational effects & measurements

def flux_to_mJy(params):
	
	if params['dist'] != 0:
		mJy_conv = 1.e3 / (2.356e5 * (params['dist'] * params['dist']))
	else:
		mJy_conv = 1.	
	return mJy_conv

def add_noise(spectrum, params):
	"""
	Add noise to noiseless HI spectrum

    Parameters
    ----------
    spectrum : array 	[mJy]
		Noiseless HI spectrum 
	params : list
		List of model input parameters
			RMS = input RMS measurement noise [mJy]

    Returns
    -------
	obs_spectrum : array 	[mJy]
		Observed spectrum with measurement noise 
	"""

	RMS = params['RMS']
	noise_arr = np.random.normal(np.zeros(len(spectrum)), RMS)
	obs_spectrum = spectrum + noise_arr
	return obs_spectrum

def smooth_spectrum(vel_bins, spectrum, params):
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
			Vres = Measurement velocity resolution	[km/s]
			Vsm = Smoothed velocity resolution		[km/s]

    Returns
    -------
	smoothed_spectrum : array
		Smoothed observed spectrum 	[mJy]
	"""

	Vres = params['Vres']
	Vsm = params['Vsm']
	box_channels = int(Vsm / Vres)
	smoothed_spectrum = convolve(spectrum, Box1DKernel(box_channels)) 
	return smoothed_spectrum


def Wang16_HIsizemass(logMHI = None, logDHI = None):
	"""
	Calculates the size or diameter of the HI disk using the Wang+16 HI size-mass relation

	Parameters
	----------
	MHI : float 	[Msun]
	    Total HI mass

	Returns
	-------
	DHI : float [kpc]
		Diameter where the HI surface density equals 1 Msun/pc
	"""
	if np.ndim(logMHI)>0 or np.isscalar(logMHI!=None):
		logDHI = 0.506 * logMHI - 3.293
		return logDHI

	elif np.ndim(logDHI)>0 or np.isscalar(logDHI != None):
		logMHI = (logDHI + 3.293) / 0.506
		return logMHI
	else:
		print('Incorrect input')



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

def integrated_SN(Sint, width, rms, Vsm):
	"""
	Integrated signal to noise ratio as defined by ALFALFA

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

def peak_SN(Peak, rms):
	"""
	Peak / rms signal to noise ratio  (gross)

    Parameters
    ----------
    Prak : float 	[mJy]
		Peak spectral flux	
    rms : float 	[mJy]
		RMS measurement noise 	

    Returns
    -------
	SN : float
		Signal to noise of spectrum
	"""

	SN = Peak / rms
	return SN


def measure_radial_profile():
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
	rad1d = rad1d[0:-1] / Ropt

	DHI = size_mass_relation(MHI)				# disk size from size-mass relation (Wang+16)
	pc_pix = DHI / (len(radius))					# assumes 1 Msun/pc^2 occurs at 1 Ropt = 1 R_25 (Wang+14)
	pixarea_pc = pc_pix * pc_pix
	hi_profile = [hi_receding / pixarea_pc, hi_approaching / pixarea_pc]
	mom0_map = mom0_map / pixarea_pc



def rms_from_integrated_SN(SN, Sint, width, Vsm):
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

def rms_from_peak_SN(SN, Peak):
	"""
	Convert a signal to noise to an RMS value

    Parameters
    ----------
    SN : float 
    	Signal to noise of spectrum
    Peak : float 	[mJy]
		Peak flux of spectrum

    Returns
    -------
    rms : float 	[mJy]
		RMS measurement noise
	"""

	rms = Peak / SN
	return rms


def width_from_SN(SN, Sint, rms, Vsm):
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

def Tophat_width_from_SNpeak(SN, Sint, rms, Vres):
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

def Gaussian_width_from_SNpeak(SN, Sint, rms):
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


def Gaussian_PDF(vel, mu, sigma, alpha=0):
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

	if alpha != 0:							#include skewness 
		G = 2.e0 * np.exp(-0.5e0 * ((vel - mu) * (vel - mu))/(sigma * sigma)) * Gaussian_CDF(alpha * vel , mu, sigma)
	else:
		G = 1.e0 / (sigma * np.sqrt(2.e0*np.pi) ) * np.exp(-0.5e0 * ((vel - mu) * (vel - mu))/(sigma * sigma)) 
	return G

def Gaussian_CDF(x, mu, sigma):
	"""
	Return the cumulative probability of a Gaussian distribution 

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


if __name__ == '__main__':
	mock_global_HI_spectrum()