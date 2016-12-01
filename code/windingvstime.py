import numpy as np
import matplotlib.pyplot as plt
import matplotlib
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
linedata = []
for i in np.linspace(140,33,15) :
    lengthdata=np.fromfile(binfile+str(int(i))+'_lengths.bin', dtype="float32")
    minLength = 500.0
    winddata=np.fromfile(binfile+str(int(i))+'_windings.bin', dtype="float32")
    datasize = int(np.sqrt(lengthdata.shape[0]))
    lengthdata=lengthdata.reshape(datasize, datasize)
    winddata=winddata.reshape(datasize, datasize)
    masked= ma.masked_where(lengthdata<minLength,winddata)
    masked = masked.filled(0)
    linedata.append((masked[datasize/2,:],str(int(i))))
fig = plt.figure()
ax = fig.add_subplot(111)
matplotlib.rcParams['legend.numpoints'] = 1
cmap = matplotlib.cm.get_cmap('jet')
#Wist je dat dit ook zonder forloop kan?
for dataset,i in linedata :
    x = np.linspace(0,np.pi,datasize)
    ax.plot(x,dataset,'.',label='t='+i, color=cmap(int(i)/140.))
ax.set_ylim([-1,-0.5])
ax.set_xlim([x[100],x[800]])
ax.set_ylabel('Winding number')
ax.set_xlabel(r'$\sim r$')
plt.legend(loc='right')
plt.show()
