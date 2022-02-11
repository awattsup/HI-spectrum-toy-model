import numpy as np
# import model_asymmetry_parameterspace as masym
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec
from functions import *
from matplotlib.lines import Line2D



def model_construction():


	params = {'dim':1000, 'incl':45, 'MHI':1.e9, 'dist':50,
				'HImod':'FE', 'HIparams':[1,1.65],
				'RCparams':[200,0.164,0.002], 'Vdisp':10,
				'Vlim':400, 'Vres':2, 'Vsm':10, 'RMS':0}

	default_model = mock_global_HI_spectrum(params)

	mom_maps = default_model[0]
	spec = default_model[1]

	fig = plt.figure(figsize = (15,5.5))
	gs  = gridspec.GridSpec(1,3, left = 0.06, right = 0.995, top=0.99, bottom = 0.08,wspace=0.15)

	mom0 = fig.add_subplot(gs[0,0])
	sideview = fig.add_subplot(gs[0,1])
	spec_ax = fig.add_subplot(gs[0,2])

	cb_ax1 = mom0.inset_axes([0.025,0.95,0.95,0.05])


	x0 = 500
	y0 = 500

	phi = np.linspace(0,2*np.pi,1000)
	R = 500
	incl = 45
	xx = R * np.cos(phi) + x0
	yy = R * np.sin(phi) 
	zz = np.zeros(len(phi))

	xx_proj = xx
	yy_proj = yy*np.cos(incl * np.pi/180.) + y0

	zz_proj = yy*np.sin(incl*np.pi/180)  + y0


	R_obs = 300
	phi_obs = 45

	xx_obs = R_obs * np.cos(phi_obs*np.pi/180) + x0
	yy_obs = R_obs * np.sin(phi_obs*np.pi/180) 
	zz_obs = 0

	xx_obs_proj = xx_obs
	yy_obs_proj = yy_obs*np.cos(incl * np.pi/180.) + y0

	zz_obs_proj = yy_obs*np.sin(incl*np.pi/180)  + y0

	img = mom0.pcolormesh(np.linspace(0,1000,1000),np.linspace(0,1000,1000),np.log10(mom_maps[0]),cmap='Blues',alpha=0.5,zorder=-10)


	mom0.plot([x0,xx_obs_proj],[y0,yy_obs_proj],color='Orange',zorder=-1)
	mom0.text(x0-60,y0-50,'($x_0$,$y_0$)',color='Orange',fontsize=16)
	mom0.text(x0+100,y0+20,'$\phi$',color='Orange',fontsize=16)
	mom0.text(x0+55,y0+70,'R',color='Orange',fontsize=16,rotation=40)

	mom0.plot(x0+180*np.cos(np.arange(0,34,0.01)*np.pi/180),
				y0+180*np.sin(np.arange(0,34,0.01)*np.pi/180),
					color='Orange',lw=1)


	mom0.text(xx_obs_proj-180,yy_obs_proj-15,"($x'$,$y'$)",color='Orange',fontsize=16)
	mom0.text(xx_obs_proj+40,yy_obs_proj-15,"M$_{HI}$",color='Red',fontsize=16)
	# mom0.fill_between([xx_obs_proj-25,xx_obs_proj+25],[yy_obs_proj-25],[yy_obs_proj+25],color='Red',alpha=0.5)
	mom0.fill([xx_obs_proj-35,xx_obs_proj-28,xx_obs_proj+35,xx_obs_proj+28,xx_obs_proj-35], 
			[yy_obs_proj-35,yy_obs_proj+28,yy_obs_proj+35,yy_obs_proj-28,yy_obs_proj-35],
						color='Red',alpha=0.7)

	mom0.arrow(xx_obs_proj,yy_obs_proj,-120,120,color='Black',width=3,head_width=20)
	mom0.text(xx_obs_proj-130,yy_obs_proj+160,"V$_c$",color='Black',fontsize=16)

	mom0.arrow(xx_obs_proj,yy_obs_proj,0,120,color='Black',width=3,head_width=20)
	mom0.text(xx_obs_proj-10,yy_obs_proj+170,"V$_c$cos($\phi$)",color='Black',fontsize=16)


	ch_1 = fig.colorbar(img,cax=cb_ax1,orientation='horizontal')
	cb_ax1.tick_params(direction='in',labelsize=14)
	cb_ax1.set_title('log M$_{\odot}$ pix$^{-1}$',pad=-8,fontsize=18)


	sideview.plot(zz_proj,yy_proj,color='Blue',zorder=-5,lw=2)
	sideview.text(x0+20,y0+50,'$i$',color='Black',fontsize=18)
	
	sideview.plot(x0+180*np.cos(np.arange(45,90,0.01)*np.pi/180),
					y0+180*np.sin(np.arange(45,90,0.01)*np.pi/180),
					color='Black',lw=1)


	sideview.plot([x0,zz_obs_proj],[y0,yy_obs_proj],color='Orange',lw=2)
	sideview.arrow(zz_obs_proj,yy_obs_proj,120,120,color='Black',width=3,head_width=20)
	sideview.text(zz_obs_proj+120,yy_obs_proj+70,"V$_c$cos($\phi$)",color='Black',fontsize=16)

	sideview.arrow(zz_obs_proj,yy_obs_proj,120,0,color='Black',width=3,head_width=20)
	sideview.text(zz_obs_proj,yy_obs_proj-50,"V$_{los}$=",
						color='Black',fontsize=16)
	sideview.text(zz_obs_proj,yy_obs_proj-100,"V$_c$cos($\phi$)sin($i$)",
						color='Black',fontsize=16)

	

	pix_spec = 100 * Gaussian_PDF(vel = spec[:,0], mu = 100, 
											sigma = 10)

	spec_ax.plot(spec[:,0],pix_spec,color='Red')
	spec_ax.fill_between(spec[:,0],pix_spec,color='Red',alpha=0.5)
	spec_ax.text(30,0.6*np.max(pix_spec),'M$_{HI}$',color='Red',fontsize=16)


	spec_ax.plot(spec[:,0],spec[:,1],color='Black',lw=2)
	
	spec_ax.plot([100,100],[spec_ax.get_ylim()[0],1.2*np.max(pix_spec)],color='Black',lw=1,ls='--')
	spec_ax.text(90,1.02*np.max(pix_spec),'V$_{los}$',color='Black',fontsize=16)

	spec_ax.plot([100,100],[spec_ax.get_ylim()[0],1.2*np.max(pix_spec)],color='Black',lw=1,ls='--')

	spec_ax.errorbar([100], [0.4*np.max(pix_spec)], xerr=[15],
				capsize=4,ecolor='Black',capthick=2,lw=2)
	spec_ax.text(40,0.4*np.max(pix_spec),'$\sigma_{D}$',color='Black',fontsize=16)

	mom0.set_xlim([0,1000])
	mom0.set_ylim([0,1000])

	sideview.set_xlim([0,1000])
	sideview.set_ylim([0,1000])

	spec_ax.set_xlim([-300,300])

	mom0.set_xticks([0,500,1000])
	mom0.set_xticklabels(['0','0.5','1'])
	mom0.set_yticks([0,500,1000])
	mom0.set_yticklabels(['0','0.5','1'])

	sideview.set_xticks([0,500,1000])
	sideview.set_xticklabels(['0','0.5','1'])
	sideview.set_yticks([0,500,1000])
	sideview.set_yticklabels(['0','0.5','1'])



	mom0.set_xlabel('x [N$_{pix}$]',fontsize=20)
	mom0.set_ylabel('y [N$_{pix}$]',fontsize=20)
	sideview.set_xlabel('side-on projection',fontsize=20)
	# sideview.set_ylabel('y [N$_{pix}$]',fontsize=20)


	spec_ax.set_xlabel('Line-of-sight velocity [km s$^{-1}$]',fontsize=20)
	spec_ax.set_ylabel('Flux density [mJy]',fontsize=20,labelpad=-5)




	mom0.tick_params(which='both',axis='both',direction='in',labelsize=18)
	sideview.tick_params(which='both',axis='both',direction='in',labelsize=18)
	spec_ax.tick_params(which='both',axis='both',direction='in',labelsize=18)
	sideview.tick_params(which='both',axis='x',direction='in',labelsize=0)

	mom0.set_aspect('equal')
	sideview.set_aspect('equal')
	spec_ax.set_aspect(np.diff(spec_ax.get_xlim()) / np.diff(spec_ax.get_ylim()))

	mom0.plot(mom0.get_xlim(),[y0,y0],color='Grey',ls='--',lw=1,zorder=-5)
	mom0.plot([x0,x0],mom0.get_ylim(),color='Grey',ls='--',lw=1,zorder=-5)
	sideview.plot(sideview.get_xlim(),[y0,y0],color='Grey',ls='--',lw=1,zorder=-5)
	sideview.plot([x0,x0],sideview.get_ylim(),color='Grey',ls='--',lw=1,zorder=-5)



	plt.show()
	fig.savefig('/home/awatts/programs/HI-spectrum-toy-model/figure_codes/figures/model_schematic.png')








def vary_model_parameters_plot():
	params = {'dim':500, 'incl':45, 'MHI':1.e9, 'dist':150,
				'HImod':'FE', 'Vdisp':10,
				'Vlim':400, 'Vres':2, 'Vsm':0, 'RMS':0}
	
	HImodel1 = [0.5, 1.65]
	HImodel2 = [1,2.5]
	HImodel3 = [0, 0.8]

	RCmodel1 = [200,0.164,0.002]
	RCmodel2 = [225,0.149,0.003]
	RCmodel3 = [170,0.178,0.011]

	default_params = params
	default_params['HIparams'] = HImodel1
	default_params['RCparams'] = RCmodel1

	default_model = mock_global_HI_spectrum(default_params)
	default_HIdist = input_HIdist(default_params)
	default_RC = input_RC(default_params)

	mommap_dx = np.arange(-2,2+4./params['dim'],4./params['dim'])
	mommap_dx_cent = np.arange(-2,2,4./params['dim']) + 0.5*4./params['dim']

	fig = plt.figure(figsize = (8.25,np.sqrt(2)*8.25))

	gs = gridspec.GridSpec(34,20, left = 0.08,bottom=0.025,top=0.99,right=0.99, hspace=0, wspace=0)

	sigma_ax = fig.add_subplot(gs[0:7,0:5])
	RC_ax = fig.add_subplot(gs[0:7,7:12],sharex = sigma_ax)

	spec_row0 = fig.add_subplot(gs[0:7,14::])

	mom0_ax1 = fig.add_subplot(gs[8:13,0:6])
	mom0_ax2 = fig.add_subplot(gs[13:18,0:6])#,sharex=mom0_ax1,sharey=mom0_ax1)
	mom0_ax3 = fig.add_subplot(gs[18:23,0:6])#,sharex=mom0_ax1,sharey=mom0_ax1)
	mom0_ax4 = fig.add_subplot(gs[23:28,0:6])#,sharex=mom0_ax1,sharey=mom0_ax1)
	mom0_ax5 = fig.add_subplot(gs[28:33,0:6])

	mom0_cbax = mom0_ax5.inset_axes([0.04,0.13,0.92,0.08],transform=mom0_ax5.transAxes)

	mom1_ax1 = fig.add_subplot(gs[8:13,6:12])#,sharex=mom0_ax1,sharey=mom0_ax1)
	mom1_ax2 = fig.add_subplot(gs[13:18,6:12])#,sharex=mom0_ax1,sharey=mom0_ax1)
	mom1_ax3 = fig.add_subplot(gs[18:23,6:12])#,sharex=mom0_ax1,sharey=mom0_ax1)
	mom1_ax4 = fig.add_subplot(gs[23:28,6:12])#,sharex=mom0_ax1,sharey=mom0_ax1)
	mom1_ax5 = fig.add_subplot(gs[28:33,6:12])


	mom1_cbax = mom1_ax5.inset_axes([0.04,0.13,0.92,0.08],transform=mom1_ax5.transAxes)

	spec_row1 = fig.add_subplot(gs[8:15,14::],sharex = spec_row0)
	spec_row2 = fig.add_subplot(gs[16:24,14::],sharex = spec_row0)
	spec_row3 = fig.add_subplot(gs[25:33,14::],sharex = spec_row0)


	mom_axes = [mom0_ax1,mom0_ax2,mom0_ax3,mom0_ax4,mom0_ax5,
				mom1_ax1,mom1_ax2,mom1_ax3,mom1_ax4,mom1_ax5]

	all_axes = [sigma_ax,RC_ax,
				mom0_ax1,mom0_ax2,mom0_ax3,mom0_ax4,mom0_ax5,
				mom1_ax1,mom1_ax2,mom1_ax3,mom1_ax4,mom1_ax5,
				spec_row0,spec_row1,spec_row2,spec_row3]

	for mm in all_axes:
		mm.tick_params(which='both',axis='both',direction='in',labelsize=16)

	sigma_ax.set_xlabel('R/R$_{opt}$',fontsize=16,labelpad=-15)
	RC_ax.set_xlabel('R/R$_{opt}$',fontsize=16,labelpad=-15)
	sigma_ax.set_ylabel('log$\Sigma_{HI}$ [M$_{\odot}$ pix$^{-1}$]',fontsize=16,labelpad=-3)
	RC_ax.set_ylabel('V$_{C}$ [km s$^{-1}$]',fontsize=16,labelpad=-5)

	# sigma_ax.set_yscale('log')
	sigma_ax.set_yticks([3.5,4,4.5])


	for mm in [spec_row0,spec_row1,spec_row2,spec_row3]:
		mm.set_ylabel('Flux density [mJy]',fontsize=16,labelpad=0)
	spec_row3.set_xlabel('Velocity [km s$^{-1}$]',fontsize=16,labelpad=-1)



	for ii in range(5):
		mom_axes[ii].set_ylabel('R/R$_{opt}$',fontsize=16)
		if ii<4:
			mom_axes[ii].tick_params(axis='x',labelsize=0)
			mom_axes[5 + ii].tick_params(axis='x',labelsize=0)

		mom_axes[ii].set_xticks([-1,0,1])
		mom_axes[ii].set_yticks([-1,0,1,2])
		mom_axes[5+ii].set_xticks([-1,0,1])
		mom_axes[5+ii].set_yticks([-1,0,1,2])

	mom0_ax5.set_yticks([-2,-1,0,1,2])
	mom1_ax5.set_yticks([-2,-1,0,1,2])
	mom0_ax5.set_xticks([-2,-1,0,1,2])
	mom1_ax5.set_xticks([-2,-1,0,1,2])

	# mom0_ax1.set_yticks([-1,0,1,2])
	# mom1_ax1.set_yticks([-1,0,1,2])

	mom0_ax5.set_xlabel('R/R$_{opt}$',fontsize=16)
	mom1_ax5.set_xlabel('R/R$_{opt}$',fontsize=16)

	default_model = mock_global_HI_spectrum(default_params)
	default_mommaps = default_model[0]
	default_spec = default_model[1]
	# default_spec[:,1] = default_spec[:,1] / (0.5*np.max(default_spec[:,1]))
	default_HIdist = input_HIdist(default_params)
	default_RC = input_RC(default_params)

	sigma_ax.plot(default_HIdist[0],np.log10(default_HIdist[1]),color='Black')
	RC_ax.plot(default_RC[0],default_RC[1],color='Black')

	spec_row0.plot(default_spec[:,0],default_spec[:,1],color='Black')
	spec_row1.plot(default_spec[:,0],default_spec[:,1],color='Black')
	spec_row2.plot(default_spec[:,0],default_spec[:,1],color='Black')
	spec_row3.plot(default_spec[:,0],default_spec[:,1],color='Black')

	mom0_def = mom0_ax1.pcolormesh(mommap_dx,mommap_dx,np.log10(default_mommaps[0]))
	mom1_def = mom1_ax1.pcolormesh(mommap_dx,mommap_dx,default_mommaps[1],cmap='RdBu_r')
	
	mom0_ax1.contour(mommap_dx_cent,mommap_dx_cent,np.log10(default_mommaps[0]),levels=[3.8,4],
						colors='White',linestyles = ['--','-'])
	mom1_ax1.contour(mommap_dx_cent,mommap_dx_cent,default_mommaps[1],levels=[-120,-60,60,120],
						colors='Black',linestyles = ['--','-','-','--'])

	HImodels = [HImodel2,HImodel3]
	RCmodels = [RCmodel2,RCmodel3]
	incl = [25,65]
	Vdisp = [5,25]

	# mom0_vmin = np.nanmin(np.log10(default_mommaps[0]))
	# mom0_vmax = np.nanmax(np.log10(default_mommaps[0]))
	
	# mom1_vmin = np.nanmin(default_mommaps[1])
	# mom1_vmax = np.nanmax(default_mommaps[1])

	for ii in range(8):
		if ii >1:
			params['HIparams'] = HImodel1
		else:
			params['HIparams'] = HImodels[ii]

		if ii <= 1 or ii >= 4:
			params['RCparams'] = RCmodel1
		else:
			params['RCparams'] = RCmodels[ii%2]

		if ii>3 and ii<=5:
			params['incl'] = incl[ii%4]
			params['HIparams'] = HImodel1
			params['RCparams'] = RCmodel1
		if ii>5:
			params['incl'] = 45
			params['HIparams'] = HImodel1
			params['RCparams'] = RCmodel1
			params['Vdisp'] = Vdisp[ii%6]
		
		model = mock_global_HI_spectrum(params)
		print(params['HIparams'])
		print(params['RCparams'])

		mommaps = model[0]
		spec = model[1]
		# spec[:,1] = spec[:,1] / (0.5*np.max(spec[:,1]))
		HIdist = input_HIdist(params)
		RC = input_RC(params)

		# print(np.log10(np.nansum(mommaps[0])))

		# sigma_ax.plot(HIdist[0],HIdist[1])
		# RC_ax.plot(RC[0],RC[1])

		if ii==0:
			spec_row1.plot(spec[:,0],spec[:,1],color='Red')
			mom0_2 = mom0_ax2.pcolormesh(mommap_dx,mommap_dx,np.log10(mommaps[0]))
			mom0_ax2.contour(mommap_dx_cent,mommap_dx_cent,np.log10(mommaps[0]),levels=[3.8,4],
						colors='White',linestyles = ['--','-'])
			sigma_ax.plot(HIdist[0],np.log10(HIdist[1]),color='Red')

		elif ii==1:
			spec_row1.plot(spec[:,0],spec[:,1],color='Blue')
			mom0_3 = mom0_ax3.pcolormesh(mommap_dx,mommap_dx,np.log10(mommaps[0]))
			mom0_ax3.contour(mommap_dx_cent,mommap_dx_cent,np.log10(mommaps[0]),levels=[3.8,4],
						colors='White',linestyles = ['--','-'])
			sigma_ax.plot(HIdist[0],np.log10(HIdist[1]),color='Blue')
			mom0_vmin = np.nanmin(np.log10(mommaps[0]))
			mom0_vmax = np.nanmax(np.log10(mommaps[0]))

		elif ii==2:
			spec_row2.plot(spec[:,0],spec[:,1],color='Orange')
			mom1_2 = mom1_ax2.pcolormesh(mommap_dx,mommap_dx,mommaps[1],cmap='RdBu_r')
			mom1_ax2.contour(mommap_dx_cent,mommap_dx_cent,mommaps[1],levels=[-120,-60,60,120],
						colors='Black',linestyles = ['--','-','-','--'])
			RC_ax.plot(RC[0],RC[1],color='Orange')
			mom1_vmin = np.nanmin(mommaps[1])
			mom1_vmax = np.nanmax(mommaps[1])


		elif ii==3:
			spec_row2.plot(spec[:,0],spec[:,1],color='Green')
			mom1_3 = mom1_ax3.pcolormesh(mommap_dx,mommap_dx,mommaps[1],cmap='RdBu_r')
			mom1_3 = mom1_ax3.contour(mommap_dx_cent,mommap_dx_cent,mommaps[1],levels=[-120,-60,60,120],
						colors='Black',linestyles = ['--','-','-','--'])
			RC_ax.plot(RC[0],RC[1],color='Green')


		elif ii==4: 
			spec_row3.plot(spec[:,0],spec[:,1],color='Magenta')
			mom0_4 = mom0_ax4.pcolormesh(mommap_dx,mommap_dx,np.log10(mommaps[0]))
			mom0_ax4.contour(mommap_dx_cent,mommap_dx_cent,np.log10(mommaps[0]),levels=[3.8,4],
						colors='White',linestyles = ['--','-'])
			mom1_4 = mom1_ax4.pcolormesh(mommap_dx,mommap_dx,mommaps[1],cmap='RdBu_r')
			mom1_ax4.contour(mommap_dx_cent,mommap_dx_cent,mommaps[1],levels=[-120,-60,60,120],
						colors='Black',linestyles = ['--','-','-','--'])

		elif ii==5:
			spec_row3.plot(spec[:,0],spec[:,1],color='Cyan')
			mom0_5 = mom0_ax5.pcolormesh(mommap_dx,mommap_dx,np.log10(mommaps[0]))
			mom0_ax5.contour(mommap_dx_cent,mommap_dx_cent,np.log10(mommaps[0]),levels=[3.8,4],
						colors='White',linestyles = ['--','-'])
			mom1_5 = mom1_ax5.pcolormesh(mommap_dx,mommap_dx,mommaps[1],cmap='RdBu_r')
			mom1_5 = mom1_ax5.contour(mommap_dx_cent,mommap_dx_cent,mommaps[1],levels=[-120,-60,60,120],
						colors='Black',linestyles = ['--','-','-','--'])

		elif ii==6:
			spec_row0.plot(spec[:,0],spec[:,1],color='Magenta')
		elif ii==7:
			spec_row0.plot(spec[:,0],spec[:,1],color='Cyan')


	mom0_def.set_clim(vmin=mom0_vmin,vmax=mom0_vmax)
	mom0_2.set_clim(vmin=mom0_vmin,vmax=mom0_vmax)
	mom0_3.set_clim(vmin=mom0_vmin,vmax=mom0_vmax)
	mom0_4.set_clim(vmin=mom0_vmin,vmax=mom0_vmax)
	mom0_5.set_clim(vmin=mom0_vmin,vmax=mom0_vmax)

	mom1_def.set_clim(vmin=mom1_vmin,vmax=mom1_vmax)
	mom1_2.set_clim(vmin=mom1_vmin,vmax=mom1_vmax)
	mom1_3.set_clim(vmin=mom1_vmin,vmax=mom1_vmax)
	mom1_4.set_clim(vmin=mom1_vmin,vmax=mom1_vmax)
	mom1_5.set_clim(vmin=mom1_vmin,vmax=mom1_vmax)

	mom0_cb = fig.colorbar(mom0_def,cax=mom0_cbax,orientation='horizontal')
	mom1_cb = fig.colorbar(mom1_def,cax=mom1_cbax,orientation='horizontal')

	mom1_cb.ax.plot([-120,-120],mom1_cb.ax.get_ylim(),color='Black',ls='--')
	mom1_cb.ax.plot([-60,-60],mom1_cb.ax.get_ylim(),color='Black',ls='-')
	mom1_cb.ax.plot([60,60],mom1_cb.ax.get_ylim(),color='Black',ls='-')
	mom1_cb.ax.plot([120,120],mom1_cb.ax.get_ylim(),color='Black',ls='--')

	mom0_cb.ax.plot([3.8,3.8],mom0_cb.ax.get_ylim(),color='White',ls='--')
	mom0_cb.ax.plot([4,4],mom0_cb.ax.get_ylim(),color='White',ls='-')

	mom0_cbax.tick_params(direction='in')
	mom1_cbax.tick_params(direction='in')
	mom0_cbax.set_title('log M$_{\odot}$ pix$^{-1}$',pad=-8)
	mom1_cbax.set_title('km s$^{-1}$',pad=-8)

	spec_row0.set_xlim([-250,250])
	spec_row0.set_ylim([spec_row0.get_ylim()[0],1.5*spec_row0.get_ylim()[1]])
	spec_row1.set_ylim([spec_row1.get_ylim()[0],1.5*spec_row1.get_ylim()[1]])
	spec_row2.set_ylim([spec_row2.get_ylim()[0],1.5*spec_row2.get_ylim()[1]])
	spec_row3.set_ylim([spec_row3.get_ylim()[0],1.5*spec_row3.get_ylim()[1]])

	leg = [Line2D([0], [0], color = 'Black', ls = '-', linewidth = 2),
			Line2D([0], [0], color = 'Red', ls = '-', linewidth = 2),
			Line2D([0], [0], color = 'Blue', ls = '-', linewidth = 2)
			]
	sigma_ax.legend(leg,labels=('D1','D2','D3'),fontsize=12,loc=0,frameon=False,
										handletextpad=0.5,labelspacing=0.1)		

	leg = [Line2D([0], [0], color = 'Black', ls = '-', linewidth = 2),
			Line2D([0], [0], color = 'Orange', ls = '-', linewidth = 2),
			Line2D([0], [0], color = 'Green', ls = '-', linewidth = 2)
			]
	RC_ax.legend(handles=leg,labels=['R1','R2','R3'],fontsize=12,loc=0,frameon=False,
										handletextpad=0.5,labelspacing=0.1)	

	leg = [Line2D([0], [0], color = 'Black', ls = '-', linewidth = 2),
			Line2D([0], [0], color = 'Red', ls = '-', linewidth = 2),
			Line2D([0], [0], color = 'Blue', ls = '-', linewidth = 2)
			]
	spec_row1.legend(handles=leg,labels=['D1,R1','D2,R1','D3,R1'],fontsize=12,loc=0,frameon=False,
										handletextpad=0.5,labelspacing=0.1)	

	leg = [Line2D([0], [0], color = 'Black', ls = '-', linewidth = 2),
			Line2D([0], [0], color = 'Orange', ls = '-', linewidth = 2),
			Line2D([0], [0], color = 'Green', ls = '-', linewidth = 2)
			]
	spec_row2.legend(handles=leg,labels=['D1,R1','D1,R2','D1,R3'],fontsize=12,loc=0,frameon=False,
										handletextpad=0.5,labelspacing=0.1)	

	leg = [Line2D([0], [0], color = 'Black', ls = '-', linewidth = 2),
			Line2D([0], [0], color = 'Magenta', ls = '-', linewidth = 2),
			Line2D([0], [0], color = 'Cyan', ls = '-', linewidth = 2)
			]
	spec_row3.legend(handles=leg,labels=['$i=45^{\circ}$',f'$i={incl[0]}^{{\circ}}$',f'$i={incl[1]}^{{\circ}}$'],fontsize=12,loc=0,frameon=False,
										handletextpad=0.5,labelspacing=0.1)	

	leg = [Line2D([0], [0], color = 'Black', ls = '-', linewidth = 2),
			Line2D([0], [0], color = 'Magenta', ls = '-', linewidth = 2),
			Line2D([0], [0], color = 'Cyan', ls = '-', linewidth = 2)
			]
	spec_row0.legend(handles=leg,labels=['$\sigma_{{D}}=10$kms$^{{-1}}$',
											f'$\sigma_{{D}}={Vdisp[0]}$kms$^{{-1}}$',
											f'$\sigma_{{D}}={Vdisp[1]}$kms$^{{-1}}$'],
											fontsize=12,loc=0,frameon=False,
											handletextpad=0.5,labelspacing=0.1)


	mom0_ax1.text(0.8,0.85,'D1',color='Black',fontsize=15,transform=mom0_ax1.transAxes)
	mom1_ax1.text(0.8,0.85,'R1',color='Black',fontsize=15,transform=mom1_ax1.transAxes)
	
	mom0_ax2.text(0.8,0.85,'D2',color='Red',fontsize=15,transform=mom0_ax2.transAxes)
	mom1_ax2.text(0.8,0.85,'R2',color='Orange',fontsize=15,transform=mom1_ax2.transAxes)
	
	mom0_ax3.text(0.8,0.85,'D3',color='Blue',fontsize=15,transform=mom0_ax3.transAxes)
	mom1_ax3.text(0.8,0.85,'R3',color='Green',fontsize=15,transform=mom1_ax3.transAxes)
	
	mom1_ax4.text(0.62,0.85,f'$i$={incl[0]}$^{{\circ}}$',color='Black',fontsize=15,transform=mom1_ax4.transAxes)
	mom1_ax5.text(0.62,0.85,f'$i$={incl[1]}$^{{\circ}}$',color='Black',fontsize=15,transform=mom1_ax5.transAxes)
	

	for mm in mom_axes:
		mm.set_aspect('equal')

	# sigma_ax.set_aspect(np.abs(np.diff(sigma_ax.get_xlim()))/np.abs(np.diff(sigma_ax.get_ylim())))
	# RC_ax.set_aspect(np.abs(np.diff(RC_ax.get_xlim()))/np.abs(np.diff(RC_ax.get_ylim())))
	# for mm in [spec_row0,spec_row1,spec_row2,spec_row3]:
	# 	mm.set_aspect(np.abs(np.diff(mm.get_xlim()))/np.abs(np.diff(mm.get_ylim())))
	# plt.show()
	fig.savefig('./figure_codes/figures/vary_model_parameters.png')

def asym_model_parameters_plot():
	params = {'dim':500, 'incl':45, 'MHI':1.e9, 'dist':150,
				'HImod':'FE', 'Vdisp':10,
				'Vlim':400, 'Vres':2, 'Vsm':0, 'RMS':0}
	
	# HImodel1 = [0.5, 1.65]
	# HImodel2 = [0.1,2.5]
	# HImodel3 = [0, 0.8]

	# RCmodel1 = [200,0.164,0.002]
	# RCmodel2 = [225,0.149,0.003]
	# RCmodel3 = [170,0.178,0.011]

	HImodel1 = [0.5, 1.65]
	HImodel2 = [[0.5,1], [1.65,2.5]]
	HImodel3 = [[0.5,0.0001],[1.65, 0.8]]

	RCmodel1 = [200,0.164,0.002]
	RCmodel2 = [[200,225],[0.164,0.149],[0.002,0.003]]
	RCmodel3 = [[200,170],[0.164,0.178],[0.002,0.011]]

	default_params = params
	default_params['HIparams'] = HImodel1
	default_params['RCparams'] = RCmodel1

	default_model = mock_global_HI_spectrum(default_params)
	default_HIdist = input_HIdist(default_params)
	default_RC = input_RC(default_params)

	mommap_dx = np.arange(-2,2+4./params['dim'],4./params['dim'])
	mommap_dx_cent = np.arange(-2,2,4./params['dim']) + 0.5*4./params['dim']


	fig = plt.figure(figsize = (8.25,1.1*8.5))

	gs = gridspec.GridSpec(23,20, left = 0.08,bottom=0.08,top=0.99,right=0.99, hspace=0, wspace=0)

	sigma_ax = fig.add_subplot(gs[0:7,0:5])
	RC_ax = fig.add_subplot(gs[0:7,7:12],sharex = sigma_ax)

	# spec_row0 = fig.add_subplot(gs[0:8,14::])

	mom0_ax1 = fig.add_subplot(gs[8:13,0:6])
	mom0_ax2 = fig.add_subplot(gs[13:18,0:6])#,sharex=mom0_ax1,sharey=mom0_ax1)
	mom0_ax3 = fig.add_subplot(gs[18:23,0:6])#,sharex=mom0_ax1,sharey=mom0_ax1)


	mom0_cbax = mom0_ax3.inset_axes([0.04,0.1,0.92,0.05],transform=mom0_ax3.transAxes)

	mom1_ax1 = fig.add_subplot(gs[8:13,6:12])#,sharex=mom0_ax1,sharey=mom0_ax1)
	mom1_ax2 = fig.add_subplot(gs[13:18,6:12])#,sharex=mom0_ax1,sharey=mom0_ax1)
	mom1_ax3 = fig.add_subplot(gs[18:23,6:12])#,sharex=mom0_ax1,sharey=mom0_ax1)


	mom1_cbax = mom1_ax3.inset_axes([0.04,0.1,0.92,0.05],transform=mom1_ax3.transAxes)

	spec_row1 = fig.add_subplot(gs[6:14,14::])
	spec_row2 = fig.add_subplot(gs[15:23,14::],sharex = spec_row1)



	mom_axes = [mom0_ax1,mom0_ax2,mom0_ax3,
				mom1_ax1,mom1_ax2,mom1_ax3]

	all_axes = [sigma_ax,RC_ax,
				mom0_ax1,mom0_ax2,mom0_ax3,
				mom1_ax1,mom1_ax2,mom1_ax3,
				spec_row1,spec_row2]

	for mm in all_axes:
		mm.tick_params(which='both',axis='both',direction='in',labelsize=16)

	sigma_ax.set_xlabel('R/R$_{opt}$',fontsize=16,labelpad=-15)
	RC_ax.set_xlabel('R/R$_{opt}$',fontsize=16,labelpad=-15)
	sigma_ax.set_ylabel('log$\Sigma_{HI}$ [M$_{\odot}$ pix$^{-1}$]',fontsize=16,labelpad=-3)
	RC_ax.set_ylabel('V$_{C}$ [km s$^{-1}$]',fontsize=16,labelpad=-5)

	# sigma_ax.set_yscale('log')
	sigma_ax.set_yticks([3.5,4,4.5])

	for mm in [spec_row1,spec_row2]:
		mm.set_ylabel('Flux density [mJy]',fontsize=16,labelpad=0)
	spec_row2.set_xlabel('Velocity [km s$^{-1}$]',fontsize=16,labelpad=-4)



	for ii in range(3):
		mom_axes[ii].set_ylabel('R/R$_{opt}$',fontsize=16)
		if ii<2:
			mom_axes[ii].tick_params(axis='x',labelsize=0)
			mom_axes[3 + ii].tick_params(axis='x',labelsize=0)

		mom_axes[ii].set_xticks([-1,0,1])
		mom_axes[ii].set_yticks([-1,0,1,2])
		mom_axes[3+ii].set_xticks([-1,0,1])
		mom_axes[3+ii].set_yticks([-1,0,1,2])

	mom0_ax3.set_yticks([-2,-1,0,1,2])
	mom1_ax3.set_yticks([-2,-1,0,1,2])
	mom0_ax3.set_xticks([-2,-1,0,1,2])
	mom1_ax3.set_xticks([-2,-1,0,1,2])

	# mom0_ax1.set_yticks([-1,0,1,2])
	# mom1_ax1.set_yticks([-1,0,1,2])

	mom0_ax3.set_xlabel('R/R$_{opt}$',fontsize=16)
	mom1_ax3.set_xlabel('R/R$_{opt}$',fontsize=16)

	default_model = mock_global_HI_spectrum(default_params)
	default_mommaps = default_model[0]
	default_spec = default_model[1]
	# default_spec[:,1] = default_spec[:,1] / (0.5*np.max(default_spec[:,1]))
	default_HIdist = input_HIdist(default_params)
	default_RC = input_RC(default_params)

	sigma_ax.plot(default_HIdist[0],np.log10(default_HIdist[1]),color='Black')
	RC_ax.plot(default_RC[0],default_RC[1],color='Black')

	spec_row1.plot(default_spec[:,0],default_spec[:,1],color='Black')
	spec_row2.plot(default_spec[:,0],default_spec[:,1],color='Black')

	mom0_def = mom0_ax1.pcolormesh(mommap_dx,mommap_dx,np.log10(default_mommaps[0]))
	mom1_def = mom1_ax1.pcolormesh(mommap_dx,mommap_dx,default_mommaps[1],cmap='RdBu_r')
	
	mom0_ax1.contour(mommap_dx_cent,mommap_dx_cent,np.log10(default_mommaps[0]),levels=[3.8,4],
						colors='White',linestyles = ['--','-'])
	mom1_ax1.contour(mommap_dx_cent,mommap_dx_cent,default_mommaps[1],levels=[-120,-60,60,120],
						colors='Black',linestyles = ['--','-','-','--'])

	HImodels = [HImodel2,HImodel3]
	RCmodels = [RCmodel2,RCmodel3]

	# mom0_vmin = np.nanmin(np.log10(default_mommaps[0]))
	# mom0_vmax = np.nanmax(np.log10(default_mommaps[0]))
	
	# mom1_vmin = np.nanmin(default_mommaps[1])
	# mom1_vmax = np.nanmax(default_mommaps[1])

	for ii in range(4):
		if ii >1:
			params['HIparams'] = HImodel1
		else:
			params['HIparams'] = HImodels[ii]

		if ii <= 1 or ii >= 4:
			params['RCparams'] = RCmodel1
		else:
			params['RCparams'] = RCmodels[ii%2]
		
		model = mock_global_HI_spectrum(params)


		mommaps = model[0]
		spec = model[1]
		# spec[:,1] = spec[:,1] / (0.5*np.max(spec[:,1]))
		HIdist = input_HIdist(params)
		RC = input_RC(params)


		# sigma_ax.plot(HIdist[0],HIdist[1])
		# RC_ax.plot(RC[0],RC[1])

		if ii==0:
			spec_row1.plot(spec[:,0],spec[:,1],color='Red')
			mom0_2 = mom0_ax2.pcolormesh(mommap_dx,mommap_dx,np.log10(mommaps[0]))
			mom0_ax2.contour(mommap_dx_cent,mommap_dx_cent,np.log10(mommaps[0]),levels=[3.8,4],
						colors='White',linestyles = ['--','-'])
			sigma_ax.plot(HIdist[0],np.log10(HIdist[2]),color='Red')

		elif ii==1:
			spec_row1.plot(spec[:,0],spec[:,1],color='Blue')
			mom0_3 = mom0_ax3.pcolormesh(mommap_dx,mommap_dx,np.log10(mommaps[0]))
			mom0_ax3.contour(mommap_dx_cent,mommap_dx_cent,np.log10(mommaps[0]),levels=[3.8,4],
						colors='White',linestyles = ['--','-'])
			sigma_ax.plot(HIdist[0],np.log10(HIdist[2]),color='Blue')
			mom0_vmin = np.nanmin(np.log10(mommaps[0]))
			mom0_vmax = np.nanmax(np.log10(mommaps[0]))

		elif ii==2:
			spec_row2.plot(spec[:,0],spec[:,1],color='Orange')
			mom1_2 = mom1_ax2.pcolormesh(mommap_dx,mommap_dx,mommaps[1],cmap='RdBu_r')
			mom1_ax2.contour(mommap_dx_cent,mommap_dx_cent,mommaps[1],levels=[-120,-60,60,120],
						colors='Black',linestyles = ['--','-','-','--'])
			RC_ax.plot(RC[0],RC[2],color='Orange')
			mom1_vmin = np.nanmin(mommaps[1])
			mom1_vmax = np.nanmax(mommaps[1])


		elif ii==3:
			spec_row2.plot(spec[:,0],spec[:,1],color='Green')
			mom1_3 = mom1_ax3.pcolormesh(mommap_dx,mommap_dx,mommaps[1],cmap='RdBu_r')
			mom1_3 = mom1_ax3.contour(mommap_dx_cent,mommap_dx_cent,mommaps[1],levels=[-120,-60,60,120],
						colors='Black',linestyles = ['--','-','-','--'])
			RC_ax.plot(RC[0],RC[2],color='Green')


	mom0_def.set_clim(vmin=mom0_vmin,vmax=mom0_vmax)
	mom0_2.set_clim(vmin=mom0_vmin,vmax=mom0_vmax)
	mom0_3.set_clim(vmin=mom0_vmin,vmax=mom0_vmax)

	mom1_def.set_clim(vmin=mom1_vmin,vmax=mom1_vmax)
	mom1_2.set_clim(vmin=mom1_vmin,vmax=mom1_vmax)
	mom1_3.set_clim(vmin=mom1_vmin,vmax=mom1_vmax)

	mom0_cb = fig.colorbar(mom0_def,cax=mom0_cbax,orientation='horizontal')
	mom1_cb = fig.colorbar(mom1_def,cax=mom1_cbax,orientation='horizontal')

	mom1_cb.ax.plot([-120,-120],mom1_cb.ax.get_ylim(),color='Black',ls='--')
	mom1_cb.ax.plot([-60,-60],mom1_cb.ax.get_ylim(),color='Black',ls='-')
	mom1_cb.ax.plot([60,60],mom1_cb.ax.get_ylim(),color='Black',ls='-')
	mom1_cb.ax.plot([120,120],mom1_cb.ax.get_ylim(),color='Black',ls='--')

	mom0_cb.ax.plot([3.8,3.8],mom0_cb.ax.get_ylim(),color='White',ls='--')
	mom0_cb.ax.plot([4,4],mom0_cb.ax.get_ylim(),color='White',ls='-')

	mom0_cbax.tick_params(direction='in')
	mom1_cbax.tick_params(direction='in')
	mom0_cbax.set_title('logM$_{\odot}$pix$^{-1}$',pad=-8,color='White')
	mom1_cbax.set_title('kms$^{-1}$',pad=-8)


	spec_row1.set_xlim([-250,250])
	# spec_row0.set_ylim([spec_row0.get_ylim()[0],1.5*spec_row0.get_ylim()[1]])
	spec_row1.set_ylim([spec_row1.get_ylim()[0],1.5*spec_row1.get_ylim()[1]])
	spec_row2.set_ylim([spec_row2.get_ylim()[0],1.5*spec_row2.get_ylim()[1]])


	leg = [Line2D([0], [0], color = 'Black', ls = '-', linewidth = 2),
			Line2D([0], [0], color = 'Red', ls = '-', linewidth = 2),
			Line2D([0], [0], color = 'Blue', ls = '-', linewidth = 2)
			]
	sigma_ax.legend(leg,labels=('D1','D2','D3'),fontsize=12,loc=0,
										handletextpad=0.5,labelspacing=0.1)		
	leg = [Line2D([0], [0], color = 'Black', ls = '-', linewidth = 2),
			Line2D([0], [0], color = 'Orange', ls = '-', linewidth = 2),
			Line2D([0], [0], color = 'Green', ls = '-', linewidth = 2)
			]
	RC_ax.legend(handles=leg,labels=['R1','R2','R3'],fontsize=12,loc=0,
										handletextpad=0.5,labelspacing=0.1)	
	leg = [Line2D([0], [0], color = 'Black', ls = '-', linewidth = 2),
			Line2D([0], [0], color = 'Red', ls = '-', linewidth = 2),
			Line2D([0], [0], color = 'Blue', ls = '-', linewidth = 2)
			]
	spec_row1.legend(handles=leg,labels=['D1:D1,R1','D2:D1,R1','D3:D1,R1'],fontsize=12,loc=0,
										handletextpad=0.5,labelspacing=0.1)	
	leg = [Line2D([0], [0], color = 'Black', ls = '-', linewidth = 2),
			Line2D([0], [0], color = 'Orange', ls = '-', linewidth = 2),
			Line2D([0], [0], color = 'Green', ls = '-', linewidth = 2)
			]
	spec_row2.legend(handles=leg,labels=['D1,R1:R1','D1,R2:R1','D1,R3:R1'],fontsize=12,loc=0,
										handletextpad=0.5,labelspacing=0.1)	

	mom0_ax1.text(0.6,0.87,'D1:D1',color='Black',fontsize=15,transform=mom0_ax1.transAxes)
	mom1_ax1.text(0.6,0.87,'R1:R1',color='Black',fontsize=15,transform=mom1_ax1.transAxes)
	
	mom0_ax2.text(0.6,0.87,'D2:D1',color='Red',fontsize=15,transform=mom0_ax2.transAxes)
	mom1_ax2.text(0.6,0.87,'R2:R1',color='Orange',fontsize=15,transform=mom1_ax2.transAxes)
	
	mom0_ax3.text(0.6,0.87,'D3:D1',color='Blue',fontsize=15,transform=mom0_ax3.transAxes)
	mom1_ax3.text(0.6,0.87,'R3:R1',color='Green',fontsize=15,transform=mom1_ax3.transAxes)


	for mm in mom_axes:
		mm.set_aspect('equal')

	# sigma_ax.set_aspect(np.abs(np.diff(sigma_ax.get_xlim()))/np.abs(np.diff(sigma_ax.get_ylim())))
	# RC_ax.set_aspect(np.abs(np.diff(RC_ax.get_xlim()))/np.abs(np.diff(RC_ax.get_ylim())))
	# for mm in [spec_row0,spec_row1,spec_row2,spec_row3]:
	# 	mm.set_aspect(np.abs(np.diff(mm.get_xlim()))/np.abs(np.diff(mm.get_ylim())))
	# plt.show()
	fig.savefig('./figure_codes/figures/asym_model_parameters.png')



def TNG100_paper_plot():

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

	# ax1.set_ylim([0.5,15])
	ax2.set_ylim([0,260])
	# ax3.set_ylim([-0.1,45])


	axlist = [[ax1,ax2,ax3],[ax4,ax5,ax6],[ax7,ax8,ax9]]

	params = {'dim':500, 'incl':60, 'MHI':1.e10, 'dist':150.e0,
				'HImod':'FE','Vdisp':10,
				'Vlim':500, 'Vres':2.e0, 'Vsm':10, 'RMS':0}

	
	# mjy_conv = 1.e3 / (2.356e5  * (dist ** 2.e0))
	# Sint = mjy_conv * MHI 
	

	HI_params_list = [[0.5, 1.65e0],
				[[0.5,0.3], [1.65e0,1.23e0]],
				[0.5, 1.65e0]]

	RC_params_list = [[200.e0, 0.164e0, 0.002e0],
				[200e0, 0.164e0, 0.002e0],
				[[200.e0, 266e0], [0.164e0, 0.13e0], [0.002e0,0.002e0]]]


	for ii in range(0,3):
		axes1 = axlist[ii][0]
		axes2 = axlist[ii][1]
		axes3 = axlist[ii][2]
		print(HI_params_list[ii])
		print(RC_params_list[ii])

		params['HIparams'] = HI_params_list[ii]
		params['RCparams'] = RC_params_list[ii]

		model = mock_global_HI_spectrum(params=params)
		vel_bins = (model[1])[:,0]
		spectrum = (model[1])[:,1]

		Peaklocs = locate_peaks(spectrum)
		Peaks = spectrum[Peaklocs]
		width_full = locate_width(spectrum, Peaks, 0.2e0)
		width = (width_full[1] - width_full[0]) * params['Vres']
		Sint, Afr = areal_asymmetry(spectrum, width_full, params['Vres'])

		width_vel = [vel_bins[0] + width_full[0]*params['Vres'], vel_bins[0] + width_full[1]*params['Vres']]
		width_mid = (width_vel[1]+ width_vel[0])*0.5


		# print(Afr)
		# print(np.log10(np.nansum(spectrum))

		
		input_HI=  input_HIdist(params)
		RC = input_RC(params)	
		if ii !=1:
			axes1.plot(input_HI[0],input_HI[1],color='Red',lw = 8)
			axes1.plot(input_HI[0],input_HI[1],ls=':',color='Blue',lw = 8)
		else:
			axes1.plot(input_HI[0],input_HI[1],color='Red',lw = 8)
			axes1.plot(input_HI[0],input_HI[2],ls=':',color='Blue',lw = 8)
		axes1.set_yscale('log')

		if ii !=2:
			axes2.plot(RC[0],RC[1],color='Red',lw = 8)
			axes2.plot(RC[0],RC[1],ls=':',color='Blue',lw = 8)
		else:
			axes2.plot(RC[0],RC[1],color='Red',lw = 8)
			axes2.plot(RC[0],RC[2],ls=':',color='Blue',lw = 8)


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

		axes1.set_ylabel('$\Sigma_{HI}$ [M$_{\odot}$ pix$^{-1}$]',fontsize=27)
		axes2.set_ylabel('Circular Velocity [km s$^{-1}$]',fontsize=27)
		axes3.set_ylabel('Flux Density [mJy]',fontsize=27)

		# axes1.set_yticks([0.5,1,2,5,10])
		# axes1.set_yticklabels([0.5,1,2,5,10])
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

	fig.savefig('./figure_codes/figures/Afr_demo_v2.png')

	# plt.show()
	# exit()










if __name__ == '__main__':
	# TNG100_paper_plot()
	model_construction()
	# vary_model_parameters_plot()
	# asym_model_parameters_plot()















