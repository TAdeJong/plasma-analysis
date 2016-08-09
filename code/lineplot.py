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
data=np.fromfile("../datadir/data.bin", dtype="float32")
data=data.reshape(64, int(len(data)/(4*64)) , 4)
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
for i in range(0,8,1) :
    ax.plot(data[i,:,0], data[i,:,1], data[i,:,2])
plt.show()

