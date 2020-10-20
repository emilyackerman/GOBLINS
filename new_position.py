import numpy as np

def new_position(par, eps, upper, lower, ind):
	npar = len(par)
	new_par = par
	for i in range(0,npar):
		if i in ind:
			scale = ((np.log(upper[i]) - np.log(lower[i])))/2
			new_par[i] = par[i]*np.exp(eps*np.random.randn(1)*scale)
			if (new_par[i] > upper[i]): 
				new_par[i] = (lower[i]/upper[i])*new_par[i]
			if new_par[i] < lower[i]: 
				new_par[i] = (upper[i]/lower[i])*new_par[i]
	return new_par