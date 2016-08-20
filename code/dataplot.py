import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys

def loadCudaStream(name):
    """
    reads the file specified by name into a numpy array (and removes
    the superfluous fourth bit from cuda's float4)

    np.shape(data)=(N,3) where N is the length of a streamline
    """
    data=np.fromfile(name, dtype="float32")
    data=data.reshape(int(len(data)/4), 4)
    data=np.delete(data,3,1)
    return data
clampVal = 1;
if (len(sys.argv) < 2) :
	print("Usage: \n dataplot.py path_to_binfile [clamp value]")
	sys.exit()
elif (len(sys.argv) > 2) :
	clampVal = float(sys.argv[2])
binfile = sys.argv[1]
data=np.fromfile(binfile, dtype="float32")
datasize = int(np.sqrt(data.shape[0]))
data=data.reshape(datasize, datasize)
data = np.minimum(data,clampVal*np.ones(data.shape))
data = np.maximum(data,-1*clampVal*np.ones(data.shape))

img = plt.imshow(data)
#img.set_cmap('hot')
plt.colorbar()
# plt.show()
plt.savefig(sys.argv[1].rsplit(".",1)[0]+'_fig.png')

