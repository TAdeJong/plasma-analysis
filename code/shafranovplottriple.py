import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import sys

import numpy.ma as ma

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['axes.labelsize'] = 'xx-large'
matplotlib.rcParams['savefig.dpi'] = '300'

if (len(sys.argv) < 2) :
	print("Usage: \n dataplot.py path_to_binfile [clamp value]")
	sys.exit()
binfile = sys.argv[1]
minLength = 150.0
times = [0,100,200]
datasize = 1024*2 
masked = {}
lims = {}
for t in times :
    print(t)
    lengthdata=np.fromfile(binfile+str(t)+'_lengths.bin', dtype="float32")
    winddata=np.fromfile(binfile+str(t)+'_windings.bin', dtype="float32")
    assert int(np.sqrt(lengthdata.shape[0])) == datasize
    lengthdata=lengthdata.reshape(datasize, datasize)
    winddata=winddata.reshape(datasize, datasize)
    masked[t]= ma.masked_where(lengthdata<minLength,-1*winddata)
    lims[t] = [np.min(masked[t]),np.max(masked[t])]

cmap = plt.cm.get_cmap('plasma',8192)
lims[0] = [2.9,5]
#lims[100] = [0.95,1.175]
#lims[200] = [0.720,0.85]
lims[200] = lims[100] = [0.72,1.175]
extends = {100: [0,2.5,-2.5,2.5], 200: [0,2.5,-2.5,2.5], 0: [0,2,-1,1]}

for t in times :
    fig = plt.figure(t)
    fig_size = fig.get_size_inches()
    if t != 0:
        fig.set_size_inches(fig_size[0]*(3/5),fig_size[1]*(2/2),forward=True)
    ax = fig.add_subplot(111)
    img = ax.imshow(masked[t], 
        clim=lims[t],
        cmap=cmap,
        extent = extends[t],
        )
    ax.set_ylabel('$z$')
    ax.set_xlabel('$x$')
    cbar = plt.colorbar(img)
    cbar.ax.set_title(r'$\imath$')
    plt.title(r'$t_\eta = '+"{0:.2f}".format(t*1e-3)+'$')
    plt.tight_layout()
#cbar.ax.set_yticklabels([ '$2/3$', r'$5/8$',r'$3/5$',r'$5/8$','$0.62$'])
    plt.savefig('./animation'+str(t)+'_fig.png')
