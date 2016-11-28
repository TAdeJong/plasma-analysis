import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize
import sys
from termcolor import colored

def line(x,a,x0) :
    return a*x+x0

def texsci(number):
    return "\\num{{{0:.2e}}}".format(number)

if __name__ == "__main__":
    if(len(sys.argv) < 2) :
        binfile = './data.csv'
    else :
        binfile = sys.argv[1]
    try :
        data = np.loadtxt(binfile, delimiter=',')
    except IOError:
        print colored('Warning:', 'red', attrs=['bold']), "Data file'", binfile, "' not found. Using data.csv instead."
        data = np.loadtxt('data.csv', delimiter=',')
    X = data[:,0]
    T = data[:,1]
    pfit,pconv = optimize.curve_fit(line,X,T,(1,0))
    Xcont = np.linspace(np.max(X),np.min(X),100)
    fig, (ax1, ax3) = plt.subplots(2,
            sharex=True,
            gridspec_kw={'height_ratios' : [3,1]}
            )
    ax1.plot(X*1e-6,T, 'bx', label='Measured calculation time')
    ax1.plot(Xcont*1e-6,
            line(Xcont,*pfit),
            '-', 
            color='black', 
            label=r'$t ='+texsci(pfit[0]*1e6)+r'\cdot s / \text{Mpxl} +'+texsci(pfit[1])+'s$')
    ax1.set_ylabel(r'$t$ (s)')
    ax1.legend(loc=2)
    ax3.set_xlabel(r'\#MPixels ')
    ax3.set_ylabel(r'$\Delta t $ (s)')
    ax3.plot(X*1e-6,(T-line(X,*pfit)),'x', color='black', label='residues')
    plt.tight_layout()
    plt.show()
