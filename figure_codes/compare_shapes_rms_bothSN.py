import numpy as np 
from astropy.table import Table 
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt 
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
import sys


bases = ['doublehorn','gaussian','tophat','narrowgaussian']
lines = ['-','--',':','-.']

basedir = '/media/data/models/HI_spectrum_toy_model/models/'

fig = plt.figure(figsize = (12,4))
gs  = gridspec.GridSpec(1, 2, wspace = 0,left = 0.08, right = 0.98, top=0.99, bottom = 0.14)
AA_ax = fig.add_subplot(gs[0,0])
PN_ax = fig.add_subplot(gs[0,1],sharey = AA_ax,sharex = AA_ax)

AA_ax.set_xlabel('$S/N$',fontsize = 16)
AA_ax.set_ylabel('Asymmetry measure $A_{fr}$',fontsize = 16)
PN_ax.set_xlabel('$S/N_{Peak}$',fontsize = 16)

AA_ax.set_ylim(0.99,1.42)
AA_ax.set_yticks([1.0,1.1,1.2,1.3,1.4])
AA_ax.set_xlim(5,102)
AA_ax.set_xscale('log')

AA_ax.set_xticks([7,10,25,50,75,100])
AA_ax.xaxis.set_major_formatter(ticker.ScalarFormatter())

PN_ax.set_xticks([7,10,25,50,75,100])
PN_ax.xaxis.set_major_formatter(ticker.ScalarFormatter())

AA_ax.tick_params(axis = 'both',which = 'both',direction = 'in', labelsize = 16,length = 8, width = 1.25)
AA_ax.tick_params(axis = 'both',which = 'minor',direction = 'in', labelsize = 16,length = 4, width = 1.25)
AA_ax.tick_params(axis = 'y',which = 'both', direction = 'in', labelsize = 16)

PN_ax.tick_params(axis = 'both',which = 'both',direction = 'in', labelsize = 16,length = 8, width = 1.25)
PN_ax.tick_params(axis = 'both',which = 'minor',direction = 'in', labelsize = 16,length = 4, width = 1.25)
PN_ax.tick_params(axis = 'y',which = 'both',direction = 'in', labelsize = 0)

legend = [Line2D([0], [0], color = 'Green', ls='-'),
		Line2D([0], [0], color = 'Orange', ls='-'),
		Line2D([0], [0], color = 'Black', ls='-'),
		Line2D([0], [0], color= 'Black', ls='--'),
		Line2D([0], [0], color= 'Black', ls=':'),
		Line2D([0], [0], color= 'Black', ls='-.')]

for ii in range(len(bases)):
	base = bases[ii]

	AA_filename = f'{basedir}{base}AAVsm10_Afr1.00_rms/statistics/SN_Afr_statistics.fits'
	AAstats = Table.read(AA_filename)

	AA_ax.plot(AAstats['avg_SN_w20'],AAstats['P50_w20'], color = 'Green', ls = lines[ii],zorder = -1,lw=2)
	AA_ax.plot(AAstats['avg_SN_w20'],AAstats['P90_w20'], color = 'Orange', ls = lines[ii],zorder = 0,lw=2)

	# PN_filename = f'{basedir}{base}PNVsm10_Afr1.00_rms/statistics/SN_Afr_statistics.fits'
	# PNstats = Table.read(PN_filename)

	# PN_ax.plot(PNstats['avg_SN_w20'],PNstats['P50_w20'], color = 'Green', ls = lines[ii],zorder = -1,lw=2)
	# PN_ax.plot(PNstats['avg_SN_w20'],PNstats['P90_w20'], color = 'Orange', ls = lines[ii],zorder = 0,lw=2)

	if base == 'doublehorn':
		DHP50 = np.array(AAstats['P50_w20'])
		DHP90 = np.array(AAstats['P90_w20'])
	if base == 'narrowgaussian':
		NP50 = np.array(AAstats['P50_w20'])
		NP90 = np.array(AAstats['P90_w20'])

AA_ax.legend(legend,['50$^{th}$ percentile','90$^{th}$ percentile', 'Double-horn','Gaussian','Top-hat','Narrow'],
		fontsize=12)
plt.show()

plotdir = 'figures/shapecompare_rms.png'.format(base = base)
fig.savefig(plotdir, dpi = 150)

plt.close()

print(DHP50)
print(DHP50 - NP50)
print((DHP50 - NP50) / (1-NP50))
print(DHP90 - NP90)
print((DHP90 - NP90) / (1-NP90))