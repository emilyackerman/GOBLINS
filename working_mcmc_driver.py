import numpy as np
import matplotlib.pyplot as plt
import copy

from scipy.integrate import odeint

from two_state_model import f2
from error import errorfxn
from new_position import new_position 

## autotuning parameter for input (jump size), noise on initial parameters for goblins to explore space based on chain temp 

# Import data
tvec = np.genfromtxt('data_h1n1.csv', usecols = 0, delimiter = ',',
                     skip_header = 1) # timepoints
data1 = np.genfromtxt('data_h1n1.csv', usecols = [1,2], delimiter = ',',
                     skip_header = 1) # data
std1 =  np.genfromtxt('std_h1n1.csv', usecols = [1,2], delimiter = ',',
                     skip_header = 1)#std dev of data
data5 = 1.2*np.genfromtxt('data_h1n1.csv', usecols = [1,2], delimiter = ',',
                     skip_header = 1) # data
std5 =  1.2*np.genfromtxt('std_h1n1.csv', usecols = [1,2], delimiter = ',',
                     skip_header = 1)#std dev of data
                     # **usecols must change when changing model states
#data_in1 = (tvec, data1, std1)

# set initial conditions
u0 = data1[0,] #DO I NEED TO HAVE TWO ICS? 
par1 = np.array([78.6, 50.6, 11.6, 20.3, 29.8, 68.8])
# par1 = np.array([61.95286361, 14.03003867, 4.65720391, 10.1239236, 8.16896111, 11.18812161])
par5 = np.array([71.8, 52.6, 14.5, 21.1, 26.6, 67.9])
# par5 = np.array([7.22900443e+05, 5.01994804e+05, 1.44979755e+05, 2.13131990e-03, 5.27653885e-03, 2.26577285e-01])
upper = np.maximum(par1, par5)*1000
lower = np.minimum(par1, par5)/1000
print(upper, lower, 'bounds')


def g(model, t, u0, p):
    """Return integration of model"""
    sol = odeint(model, u0, t, args=(p,))
    return sol


# initialize parameters
beta = np.array([1.0]) 
eps = 0.005/np.sqrt(beta) 
nruns = 400
nchains = len(beta) 
npar = len(par1) 
output = 10

# initialize matricies 
yh1n1 = np.zeros([nchains, npar]) #current par
ychainh1n1 = np.zeros([nchains, nruns, npar]) #all saved pars 
pyh1n1 = np.zeros([nchains, npar]) #proposed pars

yh5n1 = np.zeros([nchains, npar]) #current par
ychainh5n1 = np.zeros([nchains, nruns, npar]) #all saved pars 
pyh5n1 = np.zeros([nchains, npar]) #proposed pars


Energy = np.zeros([nchains, nruns]) #Energy
energyh1n1 = np.zeros([nchains, nruns]) #EnergyH1N1
energyh5n1 = np.zeros([nchains, nruns])
e = np.zeros([nchains, 1]) #E
eh1 = np.zeros([nchains, 1]) #EH1
eh5 = np.zeros([nchains, 1])
accept = np.zeros([nchains, 1]) 
reject = np.zeros([nchains, 1])
accept_swap = np.zeros([nchains-1, 1])
reject_swap = np.zeros([nchains-1, 1])

pars2change = np.array([1,2,3,4])    
parsnotchanged = np.array([0,5])

for c in np.arange(0, nchains): 
    print(c)
    tmax = 7
    tspan = np.arange(0,tmax+.05,0.05)
    ychainh1n1[c, 0, :] = copy.deepcopy(par1[:])
    yh1n1[c, :] = copy.deepcopy(par1[:])
    ychainh5n1[c, 0, :] = copy.deepcopy(par5[:])
    yh5n1[c, :] = copy.deepcopy(par5[:])
    sol = g(f2, tspan, u0, par1)
    energyh1n1[c, 0] = errorfxn(tvec, tspan, sol, data1, std1)
    sol = g(f2, tspan, u0, par5)
    energyh5n1[c, 0] = errorfxn(tvec, tspan, sol, data5, std5)
    Energy[c, 0] = energyh1n1[c, 0] + energyh5n1[c, 0] 

for run in np.arange(0, nruns-1): 
    for c in np.arange(0, nchains): 
        Energy[c, run+1]  = Energy[c, run]
        energyh1n1[c, run+1] = energyh1n1[c, run]
        ychainh1n1[c, run+1,:] = ychainh1n1[c, run,:]
        energyh5n1[c, run+1] = energyh5n1[c, run]
        ychainh5n1[c, run+1,:] = ychainh5n1[c, run,:]

    for c in np.arange(0, nchains): 
        # print('top', yh1n1[c,:])
        print("\n top", yh1n1)
        pyh1n1[c,:] = new_position(yh1n1[c,:], eps[c], upper, lower, np.arange(0,len(par1))) 
        print("after new position", yh1n1) 
        sol = g(f2, tspan, u0, pyh1n1[c,:])
        eh1[c] = errorfxn(tvec, tspan, sol, data1, std1)

        pyh5n1[c,:] = copy.deepcopy(yh5n1[c,:])
        # print('pyh1n1', pyh1n1[c,:])
        # print('ehh1', eh1[c])
        # print('yh5n1', yh5n1[c,:])
        pyh5n1[c,parsnotchanged] = pyh1n1[c, parsnotchanged]
        # print('pyh5n1', pyh5n1[c,:])
        pyh5n1[c,:] = new_position(pyh5n1[c,:], eps[c], upper, lower, pars2change) 
        # print('pyh5n1 post new position', pyh5n1[c,:])
        sol = g(f2, tspan, u0, pyh5n1[c,:])
        eh5[c] = errorfxn(tvec, tspan, sol, data5, std5)
        # print('eh5', eh5[c]) 

        e[c] = eh1[c] + eh5[c]
        #delta = (Energy[c, run+1] - e[c])
        delta = (e[c] - Energy[c, run+1])
        prob = np.min((1, np.exp(-delta)))
        rand_prob = np.random.uniform(0,1)
        print("above if yh1n1", yh1n1, "yh5n1", yh5n1)
        # print(rand_prob, prob, -delta, np.exp(-delta))
        if rand_prob < prob:
            yh1n1[c,:] = copy.deepcopy(pyh1n1[c,:])
            yh5n1[c,:] = copy.deepcopy(pyh5n1[c,:])
            print("POST", c, run, yh1n1[c,:], yh5n1[c,:])
            accept[c] += 1
            Energy[c, run+1] = e[c]
            energyh1n1[c, run+1] = eh1[c]
            energyh5n1[c, run+1] = eh5[c]
        else:
            print("REJECT", c, run, yh1n1[c,:], yh5n1[c,:])
            reject[c] += 1

        #print(Energy)
#         #for c_index in np.arange(): 
    ## JORDAN WANTS TO SEE BOTH energyh1n1 AND energyh5n1 (IN GRAPHIC?), make an input option when making mcmc into function

# 10/28 IS THIS IN THE RIGHT SPOT? MISSING AN INDENT? WHY IS H5N1 ALWAYS THE SAME PAR SET? 
    for c in np.arange(0,nchains): 
        print('yh1n1', yh1n1, '\n yh5n1', yh5n1)
        ychainh1n1[c, run+1, :] = yh1n1[c,:]
        ychainh5n1[c, run+1, :] = yh5n1[c,:]
        # print('bottom ychainh5n1', ychainh5n1[c,run+1, :])

        if (run)%output == 0: 
            print("At iteration", run, "for chain temperature", beta[c])
            acceptance_rate = accept[c]/(accept[c]+reject[c])
            print("Acceptance rate = ", acceptance_rate)
            print("accept: ", accept[c], " reject: ", reject[c])
            #swapping_rate =
            print("Min energy = ", np.min(Energy[c,0:run+1]))
            print("Max energy = ", np.max(Energy[c,0:run+1]))
            print("H1N1", ychainh1n1[c, run+1, :])
            print("H5N1", ychainh5n1[c, run+1, :], "\n")
