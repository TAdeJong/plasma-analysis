import functions as f
import numpy as np
from multiprocessing import Pool


def runprocess(varnumber):
    try:
        simdir = '/media/NAS/smiet/pencil_sims/Rings/Paper_twists/n3_256_iso_VF_twist_2_5/'
        varnumber, time, radius, twist = f.analyzeFieldMaster(simdir, seedtype='grid', varnumber = varnumber, startseed = [0,.8,-.25], lMax = 4000, verbose=1)
        return np.array([varnumber, time, radius, twist])
    except:
        return np.zeros(4)*np.nan

nproc=1
simrange = range(130,131)

timeresults = Pool(nproc).map(runprocess,simrange)
timeresults = np.array(timeresults)
print timeresults
np.save('timeresults' +str(simrange[0])+'_'+str(simrange[-1])+'.npy', timeresults)
