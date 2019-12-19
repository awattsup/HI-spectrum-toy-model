import numpy as np 
from astropy.table import Table 
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec
import sys


bases = ['doublehorn','gaussian','tophat']

bigfig = plt.figure(figsize = (10,15))
gs  = gridspec.GridSpec(3, 1, hspace = 0, wspace = 0,left = 0.1, right = 0.97, top=0.98, bottom = 0.08)

DHAA = bigfig.add_subplot(gs[0,0])
GAA = bigfig.add_subplot(gs[1,0],sharex = DHAA, sharey = DHAA)
THAA = bigfig.add_subplot(gs[2,0],sharex = DHAA, sharey = DHAA)

THAA.set_xlabel('$S/N_{AA}$',fontsize = 24)
DHAA.set_ylabel('$A_{fr}$',fontsize = 24)
GAA.set_ylabel('$A_{fr}$',fontsize = 24)
THAA.set_ylabel('$A_{fr}$',fontsize = 24)


DHAA.set_ylim(0.99,1.57)
DHAA.set_xlim(5,102)
DHAA.set_xscale('log')

DHAA.tick_params(axis = 'both',which = 'both',direction = 'in', labelsize = 24)
GAA.tick_params(axis = 'both',which = 'both',direction = 'in', labelsize = 24)
THAA.tick_params(axis = 'both',which = 'both',direction = 'in', labelsize = 24)

DHAA.tick_params(axis = 'x',which = 'both',direction = 'in', labelsize = 0)
GAA.tick_params(axis = 'x',which = 'both', direction = 'in', labelsize = 0)

legend = [Line2D([0], [0], color = 'Green', ls='-'),
			Line2D([0], [0], color = 'Orange', ls='-'),
			Line2D([0], [0], color = 'Black', ls='-'),
			Line2D([0], [0], color= 'Black', ls='--')]

legend_all = [Line2D([0],[0], color='White', ls='-'),
			Line2D([0], [0], color = 'Green', ls='-'),
			Line2D([0], [0], color = 'Orange', ls='-'),
			Line2D([0], [0], color = 'Black', ls='-'),
			Line2D([0], [0], color= 'Black', ls='--')]

for ii in range(len(bases)):
	base = bases[ii]
	AA_filename = '{base}AAVsm10_rms/statistics.dat'.format(base=base)


	AAstats = Table.read(AA_filename, format='ascii')

	if base == 'doublehorn':
		DHAA.plot(AAstats['avg_SN_w20'],AAstats['P50_w20'], color = 'Green', ls = '-')
		DHAA.plot(AAstats['avg_SN_w20_obs'],AAstats['P50_w20_obs'], color = 'Green', ls = '--')

		DHAA.plot(AAstats['avg_SN_w20'],AAstats['P90_w20'], color = 'Orange', ls = '-')
		DHAA.plot(AAstats['avg_SN_w20_obs'],AAstats['P90_w20_obs'], color = 'Orange', ls = '--')

		DHAA.legend(legend_all,['Double horn','50$^{th}$ %ile','90$^{th}$ %ile','$W_{20}$', '$W_{20,obs}$'],fontsize=20)
	
	if base == 'gaussian':
		GAA.plot(AAstats['avg_SN_w20'],AAstats['P50_w20'], color = 'Green', ls = '-')
		GAA.plot(AAstats['avg_SN_w20_obs'],AAstats['P50_w20_obs'], color = 'Green', ls = '--')

		GAA.plot(AAstats['avg_SN_w20'],AAstats['P90_w20'], color = 'Orange', ls = '-')
		GAA.plot(AAstats['avg_SN_w20_obs'],AAstats['P90_w20_obs'], color = 'Orange', ls = '--')

		GAA.legend(legend_all,['Gaussian','50$^{th}$ %ile','90$^{th}$ %ile','$W_{20}$', '$W_{20,obs}$'],fontsize=20)

	if base == 'tophat':
		THAA.plot(AAstats['avg_SN_w20'],AAstats['P50_w20'], color = 'Green', ls = '-')
		THAA.plot(AAstats['avg_SN_w20_obs'],AAstats['P50_w20_obs'], color = 'Green', ls = '--')

		THAA.plot(AAstats['avg_SN_w20'],AAstats['P90_w20'], color = 'Orange', ls = '-')
		THAA.plot(AAstats['avg_SN_w20_obs'],AAstats['P90_w20_obs'], color = 'Orange', ls = '--')

		THAA.legend(legend_all,['Top-hat','50$^{th}$ %ile','90$^{th}$ %ile','$W_{20}$', '$W_{20,obs}$'],fontsize=20)

plt.show()
plotdir = 'figures/obsw20compare_rms.png'
bigfig.savefig(plotdir, dpi = 150)

	