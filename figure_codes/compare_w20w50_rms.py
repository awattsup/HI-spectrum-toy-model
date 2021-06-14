import numpy as np 
from astropy.table import Table 
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec
import sys


bases = ['doublehorn','gaussian','tophat']

bigfig = plt.figure(figsize = (10,15))
gs  = gridspec.GridSpec(3, 1, hspace = 0, wspace = 0,left = 0.08, right = 0.99, top=0.99, bottom = 0.05)

DHAA = bigfig.add_subplot(gs[0,0])
GAA = bigfig.add_subplot(gs[1,0],sharex = DHAA, sharey = DHAA)
THAA = bigfig.add_subplot(gs[2,0],sharex = DHAA, sharey = DHAA)

THAA.set_xlabel('$S/N_{AA}$',fontsize = 14)
DHAA.set_ylabel('$A_{fr}$',fontsize = 14)
GAA.set_ylabel('$A_{fr}$',fontsize = 14)
THAA.set_ylabel('$A_{fr}$',fontsize = 14)

DHAA.set_ylim(0.99,1.57)
DHAA.set_xlim(5,102)
DHAA.set_xscale('log')

DHAA.tick_params(axis = 'both',which = 'both',direction = 'in', labelsize = 12)
GAA.tick_params(axis = 'both',which = 'both',direction = 'in', labelsize = 12)
THAA.tick_params(axis = 'both',which = 'both',direction = 'in', labelsize = 12)


DHAA.tick_params(axis = 'x',which = 'both',direction = 'in', labelsize = 0)
GAA.tick_params(axis = 'x',which = 'both', direction = 'in', labelsize = 0)


legend_all = [Line2D([0],[0], color='White', ls='-'),
			Line2D([0], [0], color = 'Green', ls='-'),
			Line2D([0], [0], color = 'Orange', ls='-'),
			Line2D([0], [0], color= 'Black', ls='--'),
			Line2D([0], [0], color= 'Black', ls=':')]

for ii in range(len(bases)):
	base = bases[ii]
	AA_filename = '{base}AAVsm10_rms/statistics.dat'.format(base=base)
	# PN_filename = '{base}PNVsm10_rms/statistics.dat'.format(base=base)

	AAstats = Table.read(AA_filename, format='ascii')
	# PNstats = Table.read(PN_filename, format='ascii')

	if base == 'doublehorn':
		DHAA.plot(AAstats['avg_SN_w20'],AAstats['P50_w20'], color = 'Green', ls = '--')
		DHAA.plot(AAstats['avg_SN_w50'],AAstats['P50_w50'], color = 'Green', ls = ':')

		DHAA.plot(AAstats['avg_SN_w20'],AAstats['P90_w20'], color = 'Orange', ls = '--')
		DHAA.plot(AAstats['avg_SN_w50'],AAstats['P90_w50'], color = 'Orange', ls = ':')

		DHAA.legend(legend_all,['Double horn','50$^{th}$ %ile','90$^{th}$ %ile', '$W_{20}$', '$W_{50}$'])
	
	if base == 'gaussian':
		GAA.plot(AAstats['avg_SN_w20'],AAstats['P50_w20'], color = 'Green', ls = '--')
		GAA.plot(AAstats['avg_SN_w50'],AAstats['P50_w20'], color = 'Green', ls = ':')

		GAA.plot(AAstats['avg_SN_w20'],AAstats['P90_w20'], color = 'Orange', ls = '--')
		GAA.plot(AAstats['avg_SN_w50'],AAstats['P90_w50'], color = 'Orange', ls = ':')


		GAA.legend(legend_all,['Gaussian','50$^{th}$ %ile','90$^{th}$ %ile', '$W_{20}$', '$W_{50}$'])

	if base == 'tophat':
		THAA.plot(AAstats['avg_SN_w20'],AAstats['P50_w20'], color = 'Green', ls = '--')
		THAA.plot(AAstats['avg_SN_w50'],AAstats['P50_w50'], color = 'Green', ls = ':')

		THAA.plot(AAstats['avg_SN_w20'],AAstats['P90_w20'], color = 'Orange', ls = '--')
		THAA.plot(AAstats['avg_SN_w50'],AAstats['P90_w50'], color = 'Orange', ls = ':')

		THAA.legend(legend_all,['Top-hat','50$^{th}$ %ile','90$^{th}$ %ile', '$W_{20}$', '$W_{50}$'])

# plt.show()
plotdir = 'figures/w20w50compare_rms.png'
bigfig.savefig(plotdir, dpi = 150)

	