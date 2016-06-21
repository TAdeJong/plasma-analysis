import numpy as np
from matplotlib import pyplot as plt

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
data=np.fromfile("../datadir/test.bin", dtype="float32")
data=data.reshape(int(len(data)/4), 4)

plt.plot(data[:,0], data[:,1])
plt.show()

