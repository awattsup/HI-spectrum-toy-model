import numpy as np
import model_asymmetry_parameterspace as masym
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D





def main():


	dim = 2000
	incl = 60
	model= 'FE'
	MHI = 1.e10
	Vdisp = 10
	dist = 150.e0
	rms_temp = -1
	Vres = 2.e0
	Vsm = 10
	
	mjy_conv = 1.e3 / (2.356e5  * (dist ** 2.e0))
	Sint = mjy_conv * MHI 
	
	Vmin = -500.e0
	Vmax = 500.e0



	HI_params_list = [[0.5, 1.65e0, 0.5, 1.65e0],
				[0.5, 1.65e0, 0.3, 1.23e0],
				[0.5, 1.65e0, 0.5, 1.65e0]]

	RC_params_list = [[200.e0, 0.164e0, 0.002e0, 200e0, 0.164e0, 0.002e0],
				[200.e0, 0.164e0, 0.002e0, 200e0, 0.164e0, 0.002e0],
				[200.e0, 0.164e0, 0.002e0, 266e0, 0.13e0, 0.002e0]]

	#   ,[200.e0, 0.164e0, 0.002e0, 266e0, 0.164e0, 0.002e0] # Afr=1.2


	fig = plt.figure(figsize = (20,16))

	gs = gridspec.GridSpec(3, 3, top = 0.99, right = 0.99, bottom  = 0.06, left = 0.07,hspace=0,wspace=0.25)

	ax1 = fig.add_subplot(gs[0,0])
	ax2 = fig.add_subplot(gs[0,1],sharex=ax1)
	ax3 = fig.add_subplot(gs[0,2])

	ax4 = fig.add_subplot(gs[1,0],sharex=ax1,sharey=ax1)
	ax5 = fig.add_subplot(gs[1,1],sharex=ax1,sharey=ax2)
	ax6 = fig.add_subplot(gs[1,2],sharex=ax3,sharey=ax3)

	ax7 = fig.add_subplot(gs[2,0],sharex=ax1,sharey=ax1)
	ax8 = fig.add_subplot(gs[2,1],sharex=ax1,sharey=ax2)
	ax9 = fig.add_subplot(gs[2,2],sharex=ax3,sharey=ax3)

	ax1.set_ylim([0.5,15])
	ax2.set_ylim([0,260])
	ax3.set_ylim([-0.1,45])


	axlist = [[ax1,ax2,ax3],[ax4,ax5,ax6],[ax7,ax8,ax9]]

	for ii in range(3):
		axes1 = axlist[ii][0]
		axes2 = axlist[ii][1]
		axes3 = axlist[ii][2]

		HI_params = HI_params_list[ii]
		RC_params = RC_params_list[ii]

		
		input_params = [incl, model,
						1, MHI, HI_params[0], HI_params[1], -1, -1 , HI_params[2], HI_params[3], -1, -1,
						1, RC_params[0], RC_params[1], RC_params[2], RC_params[3], RC_params[4], RC_params[5],
						Vdisp, dist, rms_temp, Vmin, Vmax, Vres, Vsm]		
		radius, costheta, R_opt = masym.create_arrays(dim, input_params)
		obs_mom0, rad1d, input_HI = masym.create_mom0(radius, costheta, input_params, R_opt)
		obs_mom1, input_RC  = masym.create_mom1(radius, costheta, rad1d, input_params, R_opt)
		vel_bins, base_spectrum, Sint = masym.hi_spectra(obs_mom0, obs_mom1, input_params)
		spectrum = masym.smooth_spectra(vel_bins, base_spectrum, input_params)


		Peaklocs = masym.locate_peaks(spectrum)
		Peaks = spectrum[Peaklocs]
		width_full = masym.locate_width(spectrum, Peaks, 0.2e0)
		width = (width_full[1] - width_full[0]) * Vres
		Sint, Afr = masym.areal_asymmetry(spectrum, width_full, Vres)

		width_vel = [vel_bins[0] + width_full[0]*Vres, vel_bins[0] + width_full[1]*Vres]
		width_mid = (width_vel[1]+ width_vel[0])*0.5


		print(Afr)
		# print(np.log10(np.nansum(spectrum))

		axes1.plot(rad1d,input_HI[0],color='Red',lw = 8)
		axes1.plot(rad1d,input_HI[1],ls=':',color='Blue',lw = 8)
		axes1.set_yscale('log')


		axes2.plot(rad1d,input_RC[0],color='Red',lw = 8)
		axes2.plot(rad1d,input_RC[1],ls=':',color='Blue',lw = 8)


		axes3.plot(vel_bins,spectrum,color='Black',lw = 5)
		axes3.plot([width_vel[0],width_vel[0]],[0,40],lw=3,ls='-',color='Grey')
		axes3.plot([width_vel[1],width_vel[1]],[0,40],lw=3,ls='-',color='Grey')


		fillrange1 = np.arange(width_vel[0],width_mid,0.5)
		yvals1 = np.interp(fillrange1,vel_bins,spectrum)

		fillrange2 = np.arange(width_mid,width_vel[1],0.5)
		yvals2 = np.interp(fillrange2,vel_bins,spectrum)

		axes3.fill_between(fillrange1,0,yvals1,color='Blue',alpha=0.6)
		axes3.fill_between(fillrange2,0,yvals2,color='Red',alpha=0.6)

		axes3.text(0.03,0.9,'A$_{{fr}}$ = {Afr:.2f}'.format(Afr = Afr), 
				transform=axes3.transAxes,fontsize = 23)


		axes1.tick_params(axis='both',which='both',direction='in',labelsize=25,length=8,width=0.75 )
		axes1.tick_params(axis='both',which='minor',length=4,width=0.75 )
		axes2.tick_params(axis='both',which='both',direction='in',labelsize=25,length=8,width=0.75 )
		axes2.tick_params(axis='both',which='minor',length=4,width=0.75 )
		axes3.tick_params(axis='both',which='both',direction='in',labelsize=25,length=8,width=0.75 )
		axes3.tick_params(axis='both',which='minor',length=4,width=0.75 )


		if ii < 2:
			axes1.tick_params(axis='x',which='both',labelsize=0)
			axes2.tick_params(axis='x',which='both',labelsize=0)
			axes3.tick_params(axis='x',which='both',labelsize=0)

		axes1.set_ylabel('$\Sigma_{HI}$ [M$_{\odot}$ pc$^{-2}$]',fontsize=27)
		axes2.set_ylabel('Circular Velocity [km s$^{-1}$]',fontsize=27)
		axes3.set_ylabel('Flux Density [Jy]',fontsize=27)

		axes1.set_yticks([0.5,1,2,5,10])
		axes1.set_yticklabels([0.5,1,2,5,10])
		axes2.set_yticks([0,50,100,150,200])
		axes2.set_yticklabels([0,50,100,150,200])

		axes1.set_xticks([0,0.5,1,1.5,2])
		axes1.set_xticklabels([0,0.5,1,1.5,2])
		axes2.set_xticks([0,0.5,1,1.5,2])
		axes2.set_xticklabels([0,0.5,1,1.5,2])

	ax7.set_xlabel('R / R$_{opt}$',fontsize=27)
	ax8.set_xlabel('R / R$_{opt}$',fontsize=27)
	ax9.set_xlabel('Line of sight velocity [km s$^{-1}$]',fontsize=27)


	leg = [Line2D([0], [0], color = 'Red', ls = '-', linewidth = 3),
			Line2D([0], [0], color = 'Blue', ls = ':', linewidth = 3)]

	ax1.legend(leg,labels=('Receding','Approaching'),fontsize=25,loc=4)		

	fig.savefig('./figures/Afr_demo.png')

	# plt.show()
	# exit()










if __name__ == '__main__':
	main()
















