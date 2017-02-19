import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import sys

import numpy.ma as ma

matplotlib.rcParams['axes.labelsize'] = '40'
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['savefig.dpi'] = '300'
matplotlib.rcParams['xtick.labelsize'] = 30 
matplotlib.rcParams['ytick.labelsize'] = 30 

if (len(sys.argv) < 2) :
	print("Usage: \n dataplot.py path_to_binfile [clamp value]")
	sys.exit()

binfile = sys.argv[1]
linedata = []
minLength = 150.0
tend = 290
tbegin = 0
tscale = 1e-3
for i in np.r_[[0,2,5,20],np.linspace(10,tend,15)] :
    lengthdata=np.fromfile(binfile+str(int(i))+'_lengths.bin', dtype="float32")
    winddata=np.fromfile(binfile+str(int(i))+'_windings.bin', dtype="float32")
    datasize = int(np.sqrt(lengthdata.shape[0]))
    lengthdata=lengthdata.reshape(datasize, datasize)
    winddata=winddata.reshape(datasize, datasize)
    masked= ma.masked_where(lengthdata<minLength,winddata)
#    masked = masked.filled(0)
    linedata.append((masked[datasize/2,:],str(float(i)*tscale)))
fig = plt.figure()
sizes = fig.get_size_inches()
fig.set_size_inches(sizes[0]*2,sizes[1]*2)
ax = fig.add_subplot(111)

matplotlib.rcParams['legend.numpoints'] = 1
norm = matplotlib.colors.Normalize(vmin=tbegin*tscale,vmax=tend*tscale)
cmap = matplotlib.cm.get_cmap('viridis_r')
s_m = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
s_m.set_array([])

#Wist je dat dit ook zonder forloop kan? Al kan ik niet vinden hoe.
for dataset,t in linedata :
    x = np.linspace(0,2.5,datasize)
#    ax.plot(x,-1.*dataset,'.',label='t='+i, color=cmap((int(i)-tbegin)/float(tend-tbegin)))
    ax.plot(x,-1.*dataset,'.',label='t='+t, color=s_m.to_rgba(float(t)))
ax.set_ylim([0,4.5])
ax.set_ylabel(r'$\imath$')
ax.set_xlabel(r'$x$')
tlabels = np.array([0,100,200,tend])*tscale
cbar = plt.colorbar(s_m, ticks = tlabels)
cbar.ax.set_yticklabels([r'$t_\eta = 0$',r'$t_\eta = 0.1$',r'$t_\eta = 0.2$',r'$t_\eta = 0.29$'],
        fontsize='30'
        )
cbar.ax.invert_yaxis()
plt.tight_layout()
plt.savefig('tobiastestdingen.png')
