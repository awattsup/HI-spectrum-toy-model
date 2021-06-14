import numpy as np 
from astropy.table import Table 
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt 
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
import sys


bases1 = ['doublehorn','gaussian','tophat','narrowgaussian']
bases2 = ['doublehorn','gaussian','tophat','narrow']


basedir1 = '/media/data/models/HI_spectrum_toy_model/models/'
basedir2 = '/media/data/models/SN_asym_parameterspace_old/models/'



fig = plt.figure(figsize = (6,12))
gs  = gridspec.GridSpec(4, 1, wspace = 0,left = 0.14, right = 0.95, top=0.99, bottom = 0.06)


for ii in range(len(bases1)):
	base1 = bases1[ii]
	base2 = bases2[ii]

	ax = fig.add_subplot(gs[ii,0])
	if ii == 3:
		ax.set_xlabel('$S/N$',fontsize = 16)
	ax.set_ylabel('Asymmetry measure $A_{fr}$',fontsize = 16)

	ax.set_ylim(0.99,1.42)
	ax.set_yticks([1.0,1.1,1.2,1.3,1.4])
	ax.set_xlim(5,102)
	ax.set_xscale('log')

	ax.set_xticks([7,10,25,50,75,100])
	ax.xaxis.set_major_formatter(ticker.ScalarFormatter())


	ax.tick_params(axis = 'both',which = 'both',direction = 'in', labelsize = 16,length = 8, width = 1.25)
	ax.tick_params(axis = 'both',which = 'minor',direction = 'in', labelsize = 16,length = 4, width = 1.25)
	ax.tick_params(axis = 'y',which = 'both', direction = 'in', labelsize = 16)

	new_filename = f'{basedir1}{base1}AAVsm10.0_Afr1.00_rms/statistics/SN_Afr_statistics.fits'
	newstats = Table.read(new_filename)

	ax.plot(newstats['avg_SN_w20'],newstats['P50_w20'], color = 'Blue', ls = '-',lw=2)
	ax.plot(newstats['avg_SN_w20'],newstats['P90_w20'], color = 'Blue', ls = '--',lw=2)


	old_filename = f'{basedir2}{base2}AAVsm10_Afr1.00_rms/statistics.dat'
	oldstats = Table.read(old_filename,format='ascii')

	ax.plot(oldstats['avg_SN_w20'],oldstats['P50_w20'], color = 'Red', ls = '-')
	ax.plot(oldstats['avg_SN_w20'],oldstats['P90_w20'], color = 'Red', ls = '--')


	legend = [Line2D([0], [0], color= 'White', ls='-'),
		Line2D([0], [0], color = 'Black', ls='-'),
		Line2D([0], [0], color = 'Black', ls='--'),
		Line2D([0], [0], color = 'Red', ls='-'),
		Line2D([0], [0], color= 'Blue', ls='-')]


	ax.legend(legend,[base2,'50$^{th}$ percentile','90$^{th}$ percentile', 'Old model','New model'],
			fontsize=12)


plotdir = './figures/compare_old_new_model_percentiles.png'
fig.savefig(plotdir)

plt.close()

