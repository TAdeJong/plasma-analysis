import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys

import numpy.ma as ma

fuz = 1/4.;
if (len(sys.argv) < 2) :
	print("Usage: \n dataplot.py path_to_binfile [clamp value]")
	sys.exit()
elif (len(sys.argv) > 2) :
	fuz = float(sys.argv[2])
binfile = sys.argv[1]
lengthdata=np.fromfile(binfile+'_lengths.bin', dtype="float32")
minLength = 300.0
winddata=np.fromfile(binfile+'_windings.bin', dtype="float32")
datasize = int(np.sqrt(lengthdata.shape[0]))
lengthdata=lengthdata.reshape(datasize, datasize)
winddata=winddata.reshape(datasize, datasize)
masked= ma.masked_where(lengthdata<minLength,winddata)
clampVal = np.mean(masked)
dev = ma.std(masked)
fractions = sorted([-1./3,-2./3,-1./5,-2./5,-3./5,-4./5,-1./4,-1./2,-3./4,-2./7,-3./7,-4./7,-5./7,-6./7,-3./8,-5./8,-7./8])
print(fractions)
cmap = plt.cm.get_cmap('nipy_spectral',8192)
fig = plt.figure()
img = plt.imshow(masked.filled(0), clim=[-0.667,-0.595], cmap=cmap)
#img.set_cmap('nipy_spectral')
cbar = plt.colorbar(ticks = fractions)
cbar.ax.set_yticklabels([ '$2/3$', r'$5/8$',r'$3/5$',r'$5/8$','$0.62$'])
fig_size = fig.get_size_inches()
fig.set_size_inches(fig_size[0],fig_size[0]*0.8,forward=True)
plt.show()
# plt.savefig(sys.argv[1].rsplit(".",1)[0]+'_fig.png')
