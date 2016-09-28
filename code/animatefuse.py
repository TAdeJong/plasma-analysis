import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from mpl_toolkits.mplot3d import Axes3D
import sys

import numpy.ma as ma


i=0
fuz = 1;
minLength = 300.0
if (len(sys.argv) < 2) :
	print("Usage: \n dataplot.py path_to_binfile [clamp value]")
	sys.exit()
elif (len(sys.argv) > 2) :
	fuz = float(sys.argv[2])
binfile = sys.argv[1]
lengthdata=np.fromfile(binfile+str(i)+'_lengths.bin', dtype="float32")
winddata=np.fromfile(binfile+str(i)+'_windings.bin', dtype="float32")
datasize = int(np.sqrt(lengthdata.shape[0]))
fig = plt.figure()
ims = []
for i in np.arange(149) :
    lengthdata=np.fromfile(binfile+str(i)+'_lengths.bin', dtype="float32")
    winddata=np.fromfile(binfile+str(i)+'_windings.bin', dtype="float32")
    lengthdata=lengthdata.reshape(datasize, datasize)
    winddata=winddata.reshape(datasize, datasize)
    masked= ma.masked_where(lengthdata<minLength,winddata)
    clampVal = np.mean(masked)
    dev = ma.std(masked)
    img = plt.imshow(masked.filled(0), 
#            clim=[(1-np.sign(clampVal)*dev*fuz)*clampVal,(1+np.sign(clampVal)*dev*fuz)*clampVal],
#            clim=[(1-np.sign(clampVal)*fuz)*clampVal,(1+np.sign(clampVal)*fuz)*clampVal],
            clim=[-1.2,-0.4],
            animated=True, 
            cmap='nipy_spectral')
#    img = plt.imshow(masked.filled(0), clim=(-1,0), animated=True, cmap='hot')
    ims.append([img])

ani = anim.ArtistAnimation(fig, ims, interval=300, blit=True, repeat_delay=1000)
#ani.save('animation.mp4')
plt.show()
# plt.savefig(sys.argv[1].rsplit(".",1)[0]+'_fig.png')
