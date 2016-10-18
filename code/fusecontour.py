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
img = plt.contourf(masked.filled(0), levels=fractions, clim=[-1,0])
img.set_cmap('nipy_spectral')
plt.colorbar(img, ticks = fractions)
plt.show()
# plt.savefig(sys.argv[1].rsplit(".",1)[0]+'_fig.png')
