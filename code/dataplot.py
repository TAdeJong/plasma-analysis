import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
#
data=np.fromfile("../datadir/windings.bin", dtype="float32")
data=data.reshape(64, 64)
data = np.minimum(data,10*np.ones(data.shape))
data = np.maximum(data,-10*np.ones(data.shape))

img = plt.imshow(data)
#img.set_cmap('hot')
plt.colorbar()
plt.show()

