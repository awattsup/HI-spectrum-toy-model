import numpy as np 
from astropy.table import Table 
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec
import sys


bases = ['doublehorn','gaussian']
lines = ['-','--',':','-.']

fig = plt.figure(figsize = (12,6))
gs  = gridspec.GridSpec(1, 2, wspace = 0,left = 0.08, right = 0.98, top=0.95, bottom = 0.11)
DH_ax = fig.add_subplot(gs[0,0])
GA_ax = fig.add_subplot(gs[0,1],sharey = DH_ax,sharex = DH_ax)

DH_ax.set_xlabel('$S/N_{AA}$',fontsize = 20)
DH_ax.set_ylabel('Asymmetry measure $A_{fr}$',fontsize = 20)
GA_ax.set_xlabel('$S/N_{AA}$',fontsize = 20)
DH_ax.set_title('Double-horn $A_{fr}$ = 1.25', fontsize=18)
GA_ax.set_title('Gaussian $A_{fr}$ = 1.25', fontsize=18)

DH_ax.set_ylim(0.99,1.8)
# DH_ax.set_yticks([1.0,1.1,1.2,1.3,1.4])
DH_ax.set_xlim(5,102)
DH_ax.set_xscale('log')

DH_ax.tick_params(axis = 'both',which = 'both',direction = 'in', labelsize = 20,length = 8, width = 1.25)
DH_ax.tick_params(axis = 'both',which = 'minor',direction = 'in', labelsize = 20,length = 4, width = 1.25)
DH_ax.tick_params(axis = 'y',which = 'both', direction = 'in', labelsize = 20)

GA_ax.tick_params(axis = 'both',which = 'both',direction = 'in', labelsize = 20,length = 8, width = 1.25)
GA_ax.tick_params(axis = 'both',which = 'minor',direction = 'in', labelsize = 20,length = 4, width = 1.25)
GA_ax.tick_params(axis = 'y',which = 'both',direction = 'in', labelsize = 0)

legend = [Line2D([0], [0], color = 'Green', ls='-'),
		Line2D([0], [0], color = 'Orange', ls='-'),
		Line2D([0], [0], color = 'Black', ls='-'),
		Line2D([0], [0], color= 'Black', ls='--'),
		Line2D([0], [0], color= 'Black', ls=':'),
		Line2D([0], [0], color= 'Black', ls='-.')]

axes = [DH_ax,GA_ax]

for ii in range(len(bases)):
	base = bases[ii]
	ax = axes[ii]

	sym_filename = '{base}AAVsm10_Afr1.00_rms/statistics.dat'.format(base=base)
	sym_stats = Table.read(sym_filename, format='ascii')

	ax.plot(sym_stats['avg_SN_w20'],sym_stats['P50_w20'], color = 'Green', ls = '-',zorder = -1,lw=2,label='50$^{th}$ symmetric percentile')
	ax.plot(sym_stats['avg_SN_w20'],sym_stats['P90_w20'], color = 'Orange', ls = '-',zorder = 0,lw=2,label='90$^{th}$ symmetric percentile')
	
	ax.plot(sym_stats['avg_SN_w20'],0.25 + sym_stats['P50_w20'], color = 'Blue', ls = '-',zorder = -1,lw=2,label='0.25 + 50$^{th}$ symmetric percentile')
	ax.plot(sym_stats['avg_SN_w20'],0.25 + sym_stats['P90_w20'], color = 'Red', ls = '-',zorder = 0,lw=2,label='0.25 + 90$^{th}$ symmetric percentile')

	asym_filename = '{base}AAVsm10_Afr1.25_rms/statistics.dat'.format(base=base)
	asym_stats = Table.read(asym_filename, format='ascii')

	ax.plot(asym_stats['avg_SN_w20'],asym_stats['P25_w20'], color = 'Green', ls = '--',zorder = -1,lw=2,label='25$^{th}$ asymmetric percentile')
	ax.plot(asym_stats['avg_SN_w20'],asym_stats['P75_w20'], color = 'Green', ls = '--',zorder = 0,lw=2,label='75$^{th}$ asymmetric percentile')
	ax.plot(asym_stats['avg_SN_w20'],asym_stats['P5_w20'], color = 'Orange', ls = '--',zorder = 0,lw=2,label='5$^{th}$ asymmetric percentile')
	ax.plot(asym_stats['avg_SN_w20'],asym_stats['P95_w20'], color = 'Orange', ls = '--',zorder = 0,lw=2,label='95$^{th}$ asymmetric percentile')
	# ax.plot(asym_stats['avg_SN_w20'],asym_stats['mode_Afr_w20'], color = 'Black', ls = '--',zorder = 0,lw=2,label='Mode asymmetry measurement')
	
	ax.plot([5,102],[1.25,1.25],color='Grey')
	# if base == 'doublehorn':
	# 	DHP50 = np.array(AAstats['P50_w20'])
	# 	DHP90 = np.array(AAstats['P90_w20'])
	# if base == 'narrow':
	# 	NP50 = np.array(AAstats['P50_w20'])
	# 	NP90 = np.array(AAstats['P90_w20'])

# DH_ax.legend(legend,[,'90$^{th}$ percentile', 'Double-horn','Gaussian'],
# 		fontsize=15)
plt.legend()
plt.show()

plotdir = 'figures/asymcompare_Afr1.25_rms.png'.format(base = base)
fig.savefig(plotdir, dpi = 150)

plt.close()

# print(DHP50 - NP50)
# print((DHP50 - NP50) / NP50)
# print(DHP90 - NP90)
# print((DHP90 - NP90) / NP90)