import numpy as np 
from astropy.table import Table 
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec
import sys


def main():
	file = './doublehornAAVsm10_rms/statistics.dat'


	data = Table.read(file,format='ascii')
	SNvals = np.array(data['avg_SN_w20'])
	P40 = np.array(data['P40_w20'])
	P45 = np.array(data['P45_w20'])
	P50 = np.array(data['P50_w20'])
	P55 = np.array(data['P55_w20'])
	P60 = np.array(data['P60_w20'])
	P65 = np.array(data['P65_w20'])
	P70 = np.array(data['P70_w20'])
	P75 = np.array(data['P75_w20'])
	P80 = np.array(data['P80_w20'])
	P85 = np.array(data['P85_w20'])
	P90 = np.array(data['P90_w20'])
	P95 = np.array(data['P95_w20'])

	Pvals = np.array([40,45,50,55,60,65,70,75,80,85,90,95])

	
	P40_fit, P40_fit_covar = curve_fit(P_SN,SNvals,P40)
	P45_fit, P45_fit_covar = curve_fit(P_SN,SNvals,P45)
	P50_fit, P50_fit_covar = curve_fit(P_SN,SNvals,P50)
	P55_fit, P55_fit_covar = curve_fit(P_SN,SNvals,P55)
	P60_fit, P60_fit_covar = curve_fit(P_SN,SNvals,P60)
	P65_fit, P65_fit_covar = curve_fit(P_SN,SNvals,P65)
	P70_fit, P70_fit_covar = curve_fit(P_SN,SNvals,P70)
	P75_fit, P75_fit_covar = curve_fit(P_SN,SNvals,P75)
	P80_fit, P80_fit_covar = curve_fit(P_SN,SNvals,P80)	
	P85_fit, P85_fit_covar = curve_fit(P_SN,SNvals,P85)	
	P90_fit, P90_fit_covar = curve_fit(P_SN,SNvals,P90)
	P95_fit, P95_fit_covar = curve_fit(P_SN,SNvals,P95)
	

	print(P40_fit)
	print(P45_fit)
	print(P50_fit)
	print(P55_fit)
	print(P60_fit)
	print(P65_fit)
	print(P70_fit)
	print(P75_fit)
	print(P80_fit)	
	print(P85_fit)	
	print(P90_fit)
	print(P95_fit)
	# exit()

	Ps = [P40,P45,P50,P55,P60,P65,P70,P75,P80,P85,P90,P95]
	fits = [P40_fit,P45_fit,P50_fit,P55_fit,P60_fit,P65_fit,P70_fit,P75_fit,P80_fit,P85_fit,P90_fit,P95_fit]

	for ii in range(len(Ps)):
		plt.plot(SNvals,Ps[ii],color='Red')
		plt.plot(np.arange(6,100),P_SN(np.arange(6,100),int(1000*fits[ii][0])/1000.,int(1000*fits[ii][1])/1000.),color='Black',ls='-')
		plt.xscale('log')
	plt.show()

def P_SN(SN,a,b):
	P = 1/(a*(SN-b))+1
	return P


def a_P(PX,p1,p2,p3):
	A = 1./(p1*(PX-p2))**p3 
	return A


def b_P(PX,p4,p5,p6,p7):
	B = (p4*(PX))**p6 + p7
	return B


def P_SN_all(SN,PX):
	a = a_P(PX,4.06487186e-03,-1.94511875e+02,5.76070982e+00)
	b = b_P(PX,1.1924e-02,3.99999979e+01,1.68957780e+00,-1.94081909e-01)
	print('a',a)
	print('b',b)
	P = 1/(a*(SN-b))+1
	return P


if __name__ == '__main__':
	main()



















