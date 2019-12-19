import numpy as np 
from astropy.table import Table 
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import sys


bases = ['doublehorn','gaussian','tophat']

bigfig = plt.figure(figsize = (12,18))
gs  = gridspec.GridSpec(3, 2, hspace = 0, wspace = 0,left = 0.08, right = 0.98, top=0.98, bottom = 0.05)

DHAA = bigfig.add_subplot(gs[0,0])
DHPN = bigfig.add_subplot(gs[0,1],sharex = DHAA, sharey = DHAA)

GAA = bigfig.add_subplot(gs[1,0],sharex = DHAA, sharey = DHAA)
GPN = bigfig.add_subplot(gs[1,1],sharex = DHAA, sharey = DHAA)

THAA = bigfig.add_subplot(gs[2,0],sharex = DHAA, sharey = DHAA)
THPN = bigfig.add_subplot(gs[2,1],sharex = DHAA, sharey = DHAA)

THAA.set_xlabel('$S/N$',fontsize = 20)
THPN.set_xlabel('$S/N_{Peak}$',fontsize = 20)
DHAA.set_ylabel('$A_{fr}$',fontsize = 20)
GAA.set_ylabel('$A_{fr}$',fontsize = 20)
THAA.set_ylabel('$A_{fr}$',fontsize = 20)

DHAA.set_ylim(0.99,1.42)
DHAA.set_xlim(5,102)
DHAA.set_xscale('log')

DHAA.tick_params(axis = 'both',which = 'both',direction = 'in', labelsize = 20)
GAA.tick_params(axis = 'both',which = 'both',direction = 'in', labelsize = 20)
THAA.tick_params(axis = 'both',which = 'both',direction = 'in', labelsize = 20)
THPN.tick_params(axis = 'both',which = 'both',direction = 'in', labelsize = 20)

DHAA.tick_params(axis = 'x',which = 'both',direction = 'in', labelsize = 0)
GAA.tick_params(axis = 'x',which = 'both', direction = 'in', labelsize = 0)
DHPN.tick_params(axis = 'both',which = 'both',direction = 'in', labelsize = 0)
GPN.tick_params(axis = 'both',which = 'both',direction = 'in', labelsize = 0)
THPN.tick_params(axis = 'y',which = 'both',direction = 'in', labelsize = 0)

smallfig = plt.figure(figsize = (12,4))
gs1  = gridspec.GridSpec(1, 2, hspace = 0, wspace = 0,left = 0.08, right = 0.98, top=0.98, bottom = 0.14)

GAA1 = smallfig.add_subplot(gs1[0,0])
GPN1 = smallfig.add_subplot(gs1[0,1],sharex = GAA, sharey = GAA)

GAA1.set_xlabel('$S/N$',fontsize =16)
GPN1.set_xlabel('$S/N_{Peak}$',fontsize =16)
GAA1.set_ylabel('Asymmetry measure $A_{fr}$',fontsize =16)
GAA1.set_ylim(0.99,1.42)
GAA1.set_xlim(5,102)
GAA1.set_xscale('log')
GAA1.set_yticks([1.0,1.1,1.2,1.3,1.4])
GAA1.set_xticks([7,10,25,50,75,100])
GAA1.xaxis.set_major_formatter(ticker.ScalarFormatter())
GPN1.set_xticks([7,10,25,50,75,100])
GPN1.xaxis.set_major_formatter(ticker.ScalarFormatter())
GAA1.tick_params(axis = 'both',which = 'both',direction = 'in', labelsize =16,length = 8, width = 1.25)
GAA1.tick_params(axis = 'both',which = 'minor',direction = 'in', labelsize =16,length = 4, width = 1.25)
GPN1.tick_params(axis = 'both',which = 'both', direction = 'in', labelsize = 16,length = 8, width = 1.25)
GPN1.tick_params(axis = 'both',which = 'minor', direction = 'in', labelsize = 16,length = 4, width = 1.25)
GPN1.tick_params(axis = 'y',which = 'both',direction = 'in', labelsize = 0)



legend = [Line2D([0], [0], color = 'Green', ls='-'),
			Line2D([0], [0], color = 'Orange', ls='-'),
			Line2D([0], [0], color = 'Black', ls='-'),
			Line2D([0], [0], color= 'Black', ls='--'),
			Line2D([0], [0], color= 'Black', ls=':')]

legend_all = [Line2D([0],[0], color='White', ls='-'),
			Line2D([0], [0], color = 'Green', ls='-'),
			Line2D([0], [0], color = 'Orange', ls='-'),
			Line2D([0], [0], color = 'Black', ls='-'),
			Line2D([0], [0], color= 'Black', ls='--'),
			Line2D([0], [0], color= 'Black', ls=':')]

for ii in range(len(bases)):
	base = bases[ii]
	AA_V10_filename = '{base}AAVsm10_Afr1.00_rms/statistics.dat'.format(base=base)
	PN_V10_filename = '{base}PNVsm10_Afr1.00_rms/statistics.dat'.format(base=base)

	AA_V20_filename = '{base}AAVsm20_Afr1.00_rms/statistics.dat'.format(base=base)
	PN_V20_filename = '{base}PNVsm20_Afr1.00_rms/statistics.dat'.format(base=base)

	AA_V50_filename = '{base}AAVsm50_Afr1.00_rms/statistics.dat'.format(base=base)
	PN_V50_filename = '{base}PNVsm50_Afr1.00_rms/statistics.dat'.format(base=base)

	AAV10stats = Table.read(AA_V10_filename, format='ascii')
	AAV20stats = Table.read(AA_V20_filename, format='ascii')
	AAV50stats = Table.read(AA_V50_filename, format='ascii')

	PNV10stats = Table.read(PN_V10_filename, format='ascii')
	PNV20stats = Table.read(PN_V20_filename, format='ascii')
	PNV50stats = Table.read(PN_V50_filename, format='ascii')


	if base == 'doublehorn':
		DHAA.plot(AAV10stats['avg_SN_w20'],AAV10stats['P50_w20'], color = 'Green', ls = '-')
		DHAA.plot(AAV20stats['avg_SN_w20'],AAV20stats['P50_w20'], color = 'Green', ls = '--')
		DHAA.plot(AAV50stats['avg_SN_w20'],AAV50stats['P50_w20'], color = 'Green', ls = ':')

		DHAA.plot(AAV10stats['avg_SN_w20'],AAV10stats['P90_w20'], color = 'Orange', ls = '-')
		DHAA.plot(AAV20stats['avg_SN_w20'],AAV20stats['P90_w20'], color = 'Orange', ls = '--')
		DHAA.plot(AAV50stats['avg_SN_w20'],AAV50stats['P90_w20'], color = 'Orange', ls = ':')

		DHPN.plot(PNV10stats['avg_SN_w20'],PNV10stats['P50_w20'], color = 'Green', ls = '-')
		DHPN.plot(PNV20stats['avg_SN_w20'],PNV20stats['P50_w20'], color = 'Green', ls = '--')
		DHPN.plot(PNV50stats['avg_SN_w20'],PNV50stats['P50_w20'], color = 'Green', ls = ':')

		DHPN.plot(PNV10stats['avg_SN_w20'],PNV10stats['P90_w20'], color = 'Orange', ls = '-')
		DHPN.plot(PNV20stats['avg_SN_w20'],PNV20stats['P90_w20'], color = 'Orange', ls = '--')
		DHPN.plot(PNV50stats['avg_SN_w20'],PNV50stats['P90_w20'], color = 'Orange', ls = ':')

		DHAA.legend(legend_all,['Double horn','50$^{th}$ percentile','90$^{th}$ percentile',
		 'Vsm = 10 km/s', 'Vsm = 20 km/s', 'Vsm = 50 km/s'],fontsize=15)
	
	if base == 'gaussian':
		GAA.plot(AAV10stats['avg_SN_w20'],AAV10stats['P50_w20'], color = 'Green', ls = '-')
		GAA.plot(AAV20stats['avg_SN_w20'],AAV20stats['P50_w20'], color = 'Green', ls = '--')
		GAA.plot(AAV50stats['avg_SN_w20'],AAV50stats['P50_w20'], color = 'Green', ls = ':')

		GAA.plot(AAV10stats['avg_SN_w20'],AAV10stats['P90_w20'], color = 'Orange', ls = '-')
		GAA.plot(AAV20stats['avg_SN_w20'],AAV20stats['P90_w20'], color = 'Orange', ls = '--')
		GAA.plot(AAV50stats['avg_SN_w20'],AAV50stats['P90_w20'], color = 'Orange', ls = ':')

		GPN.plot(PNV10stats['avg_SN_w20'],PNV10stats['P50_w20'], color = 'Green', ls = '-')
		GPN.plot(PNV20stats['avg_SN_w20'],PNV20stats['P50_w20'], color = 'Green', ls = '--')
		GPN.plot(PNV50stats['avg_SN_w20'],PNV50stats['P50_w20'], color = 'Green', ls = ':')

		GPN.plot(PNV10stats['avg_SN_w20'],PNV10stats['P90_w20'], color = 'Orange', ls = '-')
		GPN.plot(PNV20stats['avg_SN_w20'],PNV20stats['P90_w20'], color = 'Orange', ls = '--')
		GPN.plot(PNV50stats['avg_SN_w20'],PNV50stats['P90_w20'], color = 'Orange', ls = ':')

		GAA.legend(legend_all,['Gaussian','50$^{th}$ percentile','90$^{th}$ percentile', 
			'Vsm = 10 km/s', 'Vsm = 20 km/s', 'Vsm = 50 km/s'],fontsize=15)

		GAA1.plot(AAV10stats['avg_SN_w20'],AAV10stats['P50_w20'],lw = 2, color = 'Green', ls = '-')
		GAA1.plot(AAV20stats['avg_SN_w20'],AAV20stats['P50_w20'],lw = 2, color = 'Green', ls = '--')
		GAA1.plot(AAV50stats['avg_SN_w20'],AAV50stats['P50_w20'],lw = 2, color = 'Green', ls = ':')

		GAA1.plot(AAV10stats['avg_SN_w20'],AAV10stats['P90_w20'],lw = 2, color = 'Orange', ls = '-')
		GAA1.plot(AAV20stats['avg_SN_w20'],AAV20stats['P90_w20'],lw = 2, color = 'Orange', ls = '--')
		GAA1.plot(AAV50stats['avg_SN_w20'],AAV50stats['P90_w20'],lw = 2, color = 'Orange', ls = ':')

		GPN1.plot(PNV10stats['avg_SN_w20'],PNV10stats['P50_w20'],lw = 2, color = 'Green', ls = '-')
		GPN1.plot(PNV20stats['avg_SN_w20'],PNV20stats['P50_w20'],lw = 2, color = 'Green', ls = '--')
		GPN1.plot(PNV50stats['avg_SN_w20'],PNV50stats['P50_w20'],lw = 2, color = 'Green', ls = ':')

		GPN1.plot(PNV10stats['avg_SN_w20'],PNV10stats['P90_w20'],lw = 2, color = 'Orange', ls = '-')
		GPN1.plot(PNV20stats['avg_SN_w20'],PNV20stats['P90_w20'],lw = 2, color = 'Orange', ls = '--')
		GPN1.plot(PNV50stats['avg_SN_w20'],PNV50stats['P90_w20'],lw = 2, color = 'Orange', ls = ':')
		GAA1.legend(legend_all,['Gaussian','50$^{th}$ percentile','90$^{th}$ percentile', 
			'Vsm = 10 km/s', 'Vsm = 20 km/s', 'Vsm = 50 km/s'],fontsize=12)

	if base == 'tophat':
		THAA.plot(AAV10stats['avg_SN_w20'],AAV10stats['P50_w20'], color = 'Green', ls = '-')
		THAA.plot(AAV20stats['avg_SN_w20'],AAV20stats['P50_w20'], color = 'Green', ls = '--')
		THAA.plot(AAV50stats['avg_SN_w20'],AAV50stats['P50_w20'], color = 'Green', ls = ':')

		THAA.plot(AAV10stats['avg_SN_w20'],AAV10stats['P90_w20'], color = 'Orange', ls = '-')
		THAA.plot(AAV20stats['avg_SN_w20'],AAV20stats['P90_w20'], color = 'Orange', ls = '--')
		THAA.plot(AAV50stats['avg_SN_w20'],AAV50stats['P90_w20'], color = 'Orange', ls = ':')

		THPN.plot(PNV10stats['avg_SN_w20'],PNV10stats['P50_w20'], color = 'Green', ls = '-')
		THPN.plot(PNV20stats['avg_SN_w20'],PNV20stats['P50_w20'], color = 'Green', ls = '--')
		THPN.plot(PNV50stats['avg_SN_w20'],PNV50stats['P50_w20'], color = 'Green', ls = ':')

		THPN.plot(PNV10stats['avg_SN_w20'],PNV10stats['P90_w20'], color = 'Orange', ls = '-')
		THPN.plot(PNV20stats['avg_SN_w20'],PNV20stats['P90_w20'], color = 'Orange', ls = '--')
		THPN.plot(PNV50stats['avg_SN_w20'],PNV50stats['P90_w20'], color = 'Orange', ls = ':')

		THAA.legend(legend_all,['Top-hat','50$^{th}$ percentile','90$^{th}$ percentile', 
			'Vsm = 10 km/s', 'Vsm = 20 km/s', 'Vsm = 50 km/s'],fontsize=12)

plt.show()
plotdir = 'figures/smoothcompare_rms.png'
bigfig.savefig(plotdir, dpi = 150)
plotdir1 = 'figures/smoothcompare_small_rms.png'
smallfig.savefig(plotdir1, dpi = 150)


