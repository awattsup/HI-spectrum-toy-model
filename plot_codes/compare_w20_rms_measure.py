import numpy as np 
from astropy.table import Table 
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt 
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
import sys

basedir = '/media/data/models/SN_asym_parameterspace/models/'
bases = [f'{basedir}doublehorn',f'{basedir}gaussian',f'{basedir}tophat']
lines = ['-','--',':']
Vsms = [10,20,50]

fig = plt.figure(figsize = (12,12))
gs  = gridspec.GridSpec(3, 3, hspace = 0.02,wspace=0.02,left = 0.08, right = 0.98, top=0.99, bottom = 0.08)





types = ['w20','rms','2rms']	


for kk in range(len(Vsms)):
	Vsm = Vsms[kk] 

	DH_ax = fig.add_subplot(gs[0,kk])
	G_ax = fig.add_subplot(gs[1,kk],sharey = DH_ax,sharex = DH_ax)
	TH_ax = fig.add_subplot(gs[2,kk],sharey = DH_ax,sharex = DH_ax)

	axes = [DH_ax,G_ax,TH_ax]

	DH_ax.set_ylim(0.99,1.42)
	DH_ax.set_yticks([1.0,1.1,1.2,1.3,1.4])
	DH_ax.set_xlim(5,102)
	DH_ax.set_xscale('log')

	DH_ax.set_xticks([7,10,25,50,75,100])
	DH_ax.xaxis.set_major_formatter(ticker.ScalarFormatter())

	for ii in range(len(bases)):
		base = bases[ii]
		axis = axes[ii]
		filename = f'{base}AAVsm{Vsm}_Afr1.00_rms/statistics.dat'
		
		stats = Table.read(filename, format='ascii')

		for jj in range(len(types)):

			axis.plot(stats[f'avg_SN_{types[jj]}'],stats[f'P50_{types[jj]}'], color = 'Green', ls = lines[jj],zorder = -1,lw=2)
			axis.plot(stats[f'avg_SN_{types[jj]}'],stats[f'P90_{types[jj]}'], color = 'Orange', ls = lines[jj],zorder = 0,lw=2)

		
		axis.tick_params(axis = 'both',which = 'both',direction = 'in', labelsize = 16,length = 8, width = 1.25)
		axis.tick_params(axis = 'both',which = 'minor',direction = 'in', labelsize = 16,length = 4, width = 1.25)


		if kk == 0:
			axis.set_ylabel('Asymmetry measure $A_{fr}$',fontsize = 16)
		else:
			axis.tick_params(axis = 'y',which = 'both', direction = 'in', labelsize = 0)

		if ii == 2:
			axis.set_xlabel('$S/N$',fontsize = 16)
		else:
			axis.tick_params(axis = 'x',which = 'both', direction = 'in', labelsize = 0)

	
plt.show()

