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
print(dev)
img = plt.imshow(masked.filled(0), clim=((1-dev*fuz)*clampVal,(1+dev*fuz)*clampVal))
img.set_cmap('hot')
plt.colorbar()
plt.show()
# plt.savefig(sys.argv[1].rsplit(".",1)[0]+'_fig.png')
