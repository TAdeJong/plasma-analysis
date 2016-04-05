#
#functions.py
#functions that fit the degenerate torus, find the radius, that kind of thing
#

#Comment blaat for git

import streamData as sr
import numpy as np
import linefn as lf
from matplotlib import pyplot as plt
from matplotlib import gridspec
from matplotlib import rc

#KNOWN ISSUES:
#plot: y-axis, but positive or negative? if you change that, it doesn't change in the graph
#poincare intersections with line: faulty at islands. 
#

def analyzeField(simdir, varnumber=30, startseed = [0,1,0], lMax = 1500, nPoints=250, verbose = 0):
    """
    Take the field from the simulation directory, and create a poincare plot, 
    make and find the degenerate torus and the twist of the field. 
    """
    if verbose: print 'starting on VAR'+str(varnumber) +'.dat'
    bb, p, time = sr.readField(simdir +'data/', 'VAR'+str(varnumber) )
    center, radius, normal, POT = fitTorus(bb, p, start=startseed, lMax = 500, verbose=verbose)
    if verbose: print 'pot is'; print POT
    poincareSeeds = poincareSeedsLine(POT, nPoints = nPoints)
    ss = streamSeeds(bb, p, poincareSeeds, lMax=lMax)
    poincarePoints = poincareYZ(ss)
    if verbose: print 'got the poincare!'
    twists, distances = streamTwistDis(ss, lMin=lMax)
    if verbose: print twists
    plotPretty(poincarePoints, twists, distances, savename = 'poincare'+str(varnumber))
    #saveData()
    return filename, POT, ss

def analyzeField2(simdir, varnumber=30, startseed = [0,.6,.4], lMax = 1500, nPoints=250, verbose = 0):
    """
    Take tie field from the simulation directory, create a poincare plot, 
    and find the twist and the radius. This time in a memory-efficient way.
    """
    if verbose: print 'starting on VAR'+str(varnumber) +'.dat'
    bb, p, time = sr.readField(simdir +'data/', 'VAR'+str(varnumber) )
    center, radius, normal, POT = fitTorus(bb, p, start=startseed, lMax = 500, verbose=verbose)
    #if verbose: print 'pot is'; print POT
    poincareSeeds = randomSeeds(-1.5, 1.5, 0, 3, nPoints = nPoints)
    poincarePoints, twists, distances  = PoincareFromSeeds(bb, p, poincareSeeds, point = [0,0,0], normal = [1,0,0], globalcenter =  center, lMax=lMax, verbose=verbose)
    #if verbose: print twists
    twiststream = sr.streamSingle(bb, p, lMax=1000, xx=1.15*(POT-center)+center)
    twist = lf.getTwist2(twiststream)
    plotPretty(poincarePoints, twists, distances, 'savedir/poincare'+str(varnumber), hline = center[2])
    saveData(poincarePoints, twists, distances, varnumber)
    return varnumber, time, radius, twist

def analyzeFieldMaster(simdir, seedtype = 'grid', varnumber=30, startseed = [0,.6,.4], lMax = 1500, nPoints=250, verbose = 0):
    """
    Take tie field from the simulation directory, create a poincare plot, 
    and find the twist and the radius. This time in a memory-efficient way.
    """
    if verbose: print 'starting on VAR'+str(varnumber) +'.dat'
    bb, p, time = sr.readField(simdir +'data/', 'VAR'+str(varnumber) )
    center, radius, normal, POT = fitTorus(bb, p, start=startseed, lMax = 500, verbose=verbose)
    #if verbose: print 'pot is'; print POT
    if seedtype == 'grid':
        poincareSeeds = gridSeeds(ymin = 0, ymax = 3, zmin = -1.5, zmax=1.5, ysteps = 12, zsteps=12)
    elif seedtype == 'line':
        poincareSeeds = poincareSeedsLine(POT, nPoints = nPoints)
    else:
        poincareSeeds = randomSeeds(-1.5, 1.5, 0, 3, nPoints = nPoints)
    poincarePoints, twists, distances  = PoincareFromSeeds(bb, p, poincareSeeds, point = [0,0,0], normal = [1,0,0], globalcenter =  center, lMax=lMax, verbose=verbose)
    #if verbose: print twists
    twiststream = sr.streamSingle(bb, p, lMax=1000, xx=1.15*(POT-center)+center)
    twist = lf.getTwist2(twiststream)
    plotPretty(poincarePoints, twists, distances, 'savedir/poincare'+str(varnumber), hline = center[2])
    saveData(poincarePoints, twists, distances, varnumber)
    return varnumber, time, radius, twist






def saveData(poincarePoints, twists, distances, varnumber):
    np.save('savedir/poincarePoints'+str(varnumber)+'.npy', poincarePoints)
    np.save('savedir/twists'+str(varnumber)+'.npy', twists)
    np.save('savedir/distances'+str(varnumber)+'.npy', distances)
    


def PoincareFromSeeds(bb, p, poincareSeeds, point, normal, globalcenter, twistSeed = [1,0,0],   lMax = 3000, verbose = 0):
    """
    streams all the seeds in the array, returns the poincareSet, consisting
    of the points where the stream lines cross the plane defined by point and normal, 
    and the twist of the stream lines together with their distances. 
    """
    if verbose: print 'starting the tracers'
    poincareSet=[]
    twists=[]
    distances=[]
    for i in range(poincareSeeds.shape[0]):
        if verbose: print 'streamline number %d' %i
        s=sr.streamSingle(bb, p, lMax=lMax, xx=poincareSeeds[i,:])
        slPoincare= streamCrossings(s, point, normal)
        poincareSet.append(slPoincare)
        if s.l >500:
            #if verbose: print 'streamline long enough!'
            slcenter, slradius, slnormal = lf.getCRN(s)
            sltwist = lf.getTwist(s, slcenter, slnormal)
            #if verbose: print 'going to get the distances'
            distancepoints =  getDistancesForStreamline(slPoincare, globalcenter, slradius)
            if verbose: print distancepoints
            for dis in distancepoints:
                twists.append(sltwist) 
                distances.append(dis)
        #if verbose: print 'finished streamline number %d' %i
    return poincareSet, twists, distances






def getDistancesForStreamline(slPoincare, globalcenter, slradius):
    """
    calculates at which y positions along the line crossing through the center
    of the torus the streamed surface lies
    """
    mins = []
    maxs = []
    for i in range(slPoincare.shape[0]): #go over all sets of positions
        if slPoincare[i, 0]>0:# get only the right half
            if slPoincare[i,0]<slradius:#get the points that coincide with the left part of the circular cross-section
                mins.append(slPoincare[i,:])
            else:
                maxs.append(slPoincare[i,:])
    mins =  np.array(mins); maxs = np.array(maxs)
    distances = []
    if mins.size: #catch the emtpy case
        mindex = np.argmin(np.absolute(mins[:,1]-globalcenter[2]))#find the point that is closest to the horizontal line through the center.
        if np.absolute(mins[mindex, 1]-globalcenter[2])<.1: distances.append(mins[mindex,0]) #check if it is close enough and append. catches islands.
    if maxs.size:
        maxdex = np.argmin(np.absolute(maxs[:,1]-globalcenter[2]))
        if np.absolute(maxs[maxdex, 1] - globalcenter[2]) <.1: distances.append(maxs[maxdex,0] )
    return distances


def streamCrossings(s, point, normal):
    """
    returns the positions where the streamline croses the plane defined by 
    point and normal. Positions are in the (y, z) coordinates of the simulation
    """
    crossingIndices=lf.getCrossings(s, point, normal) #index points in the line
    crossingsYZ = []
    for j in crossingIndices:
        crossCoord = lf.pointOnPlane(s.tracers[j], s.tracers[j+1], point, normal) #interpolates between the two indices around the plane crossing
        crossingsYZ.append([np.dot([0,1,0],crossCoord), np.dot([0,0,1], crossCoord)])
    return np.array(crossingsYZ)



def plotPoincare(poincarePoints, savename, xmin=0, xmax= 3, ymin=-1.5, ymax = 1.5):
    plt.ioff()
    width = 4
    height = 6 
    plt.rc("figure.subplot", left = 0.1)
    plt.rc("figure.subplot", right = 0.9)
    plt.rc("figure.subplot", bottom = 0.15)
    plt.rc("figure.subplot", top = 0.95)
    fig = plt.figure(figsize = (width, height))
    ax = fig.add_subplot(111)
    colors=['r','b','g']
    if len(poincarePoints.shape)==3:
        for i in range(len(poincarePoints)):
            plt.scatter(poincarePoints[i][:,0], poincarePoints[i][:,1], marker='.', s=0.5, color=colors[np.mod(i,3)])


    plt.xticks(fontsize=20, family = 'serif')
    plt.yticks(fontsize=20, family = 'serif')
    # make plot pretty
    ax.dist = 5
    ax.tick_params(axis = 'both', which = 'major', length = 8, labelsize = 20)
    ax.tick_params(axis = 'both', which = 'minor', length = 4, labelsize = 20)

    plt.xlabel(r'$x_1$', fontsize = 25)
    plt.ylabel(r'$x_2$', fontsize = 25)
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)


    plt.xticks(fontsize=20, family = 'serif')
    plt.yticks(fontsize=20, family = 'serif')
    plt.savefig(savename[-14:]+'.png', bbox_inches='tight', dpi = 300)

def plotPretty(poincarePoints, twists, distances, savename, hline = 0, ymin=0, ymax=3, zmin=-1.5, zmax = 1.5, verbose=0):
    plt.ioff()
    #set standard stuff, as serif tick labelsize
    rc('font',**{'family':'serif',})
    ls=20 #labelsize


    # prepare the plot
    width = 6
    height = 8

    fig = plt.figure(figsize=(width, height))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1], sharex = ax0)

#subfigure1: the poincare plot

    if verbose:print 'plotting one'
    colors=['r','b','g']
    for i in range(len(poincarePoints)):
        if poincarePoints[i].size:
            ax0.scatter(poincarePoints[i][:,0], poincarePoints[i][:,1], marker=',',s=0.18, color=colors[np.mod(i,3)])

    ax0.set_ylim(zmin,zmax)
    ax0.set_xlim(ymin, ymax)
    #ax0.text(.1,.8,r'(a)', size = 18)



    ax0.set_ylabel(r'$z$', fontsize = ls)
    ax0.set_xlabel(r'$y$', fontsize = ls)
    ax0.axhline(y=hline, linewidth = 1, color = 'k')
    #for v in vlines:
    #    ax0.axvline(x=v, linewidth = 1, color = 'k')

    #plotting the twists
    ax1.scatter(distances, twists, marker = '.', s=1, color = 'b')
    twistav = np.average(twists)
    ax1.set_ylim(twistav*.75, twistav*1.25)
    plt.savefig(savename, bbox_inches='tight', dpi=300)

def fitTorus(bb, p, bar=.01, inttype='RK6', lMax=500, start=[1,0,0], verbose=0):
    """
    tries to find the smallest of a set of nested toroidal field structures, if
    the field consists of nested toroidal structures. 
    """
    streamline = sr.streamSingle(bb, p, lMax=500, xx=start)
    center, radius, normal = lf.getCRN(streamline)
    pos = start
    pos2 =  center + lf.perp(normal)*radius
    i=0
    while (lf.l(pos-pos2)>bar):
        pos=pos2
        streamline = sr.streamSingle(bb, p, lMax = lMax, xx=pos)
        center, radius, normal = lf.getCRN(streamline)
        pos2 =  center + lf.perp(normal)*radius
        if verbose: print 'fitdistance: %f' %(lf.l(pos-pos2))
        i +=1
        if i>20:
            if verbose: print 'did not find torus, using bad data'
            return center, radius, normal, pos2
    if verbose: print 'torus found!'
    return center, radius, normal, pos2

def poincareSeedsLine(POT,  nPoints = 500, start=-1, stop=1, center = [0,0,0]):
    """
    returns an array of the positions to seed the stream tracing
    the positions lie on a line from going from center through the degenerate
    torus. start should be negative for the points to start before the degenerate 
    torus, and stop should be positive to continue after, in units of torus radius.
    """
    return (POT)+np.outer(np.linspace(start,stop,nPoints), POT-center)


def gridSeeds(ymin=0, ymax=3, zmin=-.2, zmax=.2, ysteps=3, zsteps=3):
    print 'calculating grid'
    if ysteps*zsteps>250:
        print 'this is more than 250 seedpoints, are you sure?!!! meh, I will do as i am told'
    return np.array(cartesian(([0.], np.linspace(ymin, ymax, ysteps), np.linspace(zmin, zmax, zsteps))))

def cartesian(arrays, out=None):
    """
    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n / arrays[0].size
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in xrange(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out


def randomSeeds(ymin, ymax, zmin, zmax, nPoints = 200):
    np.random.seed()
    return np.array([np.zeros(nPoints), np.random.random(nPoints)*(ymax-ymin)+ymin, np.random.random(nPoints)*(zmax-zmin)+zmin]).T

def streamSeeds(bb, p, xx, lMax = 3000, verbose = 0):
    """
    streams all the seeds in the array, returns all the streamlines. [Needs lots of
    memory!]
    """
    if verbose: print 'starting the tracers'
    ss=[]
    for i in range(xx.shape[0]):
        s=sr.streamSingle(bb, p, lMax=lMax, xx=xx[i,:])
        ss.append(s)
        if verbose & i%10==0: print 'streamline number %d' %i

    return ss



def poincareYZ(ss, point = [0,0,0], normal = [1,0,0], perp = [0,1,0], verbose = 0):
    """
    returns the points where the streamlines in the array ss cross the plane 
    determined by point and normal.
    """
    poincareSet = []
    for i in range(len(ss)):
        crossingIndices=lf.getCrossings(ss[i], point, normal) #index points in the line
        crossingsYZ = []
        for j in crossingIndices:
            crossCoord = lf.pointOnPlane(ss[i].tracers[j], ss[i].tracers[j+1], point, normal)
            crossingsYZ.append([np.dot([0,1,0],crossCoord), np.dot([0,0,1], crossCoord)])
        poincareSet.append(np.array(crossingsYZ))
    return poincareSet


def streamTwists(ss, lMin =1000):
    """
    returns the twists 
    """
    twists = []
    for s in ss:
        if s.l > lMin:
            twist=lf.getTwist2(s)
            twists.append(twist)
    return twists

def distances(poincareSeeds, center = [0,0,0]):
    distances=[]
    for i in range(poincareSeeds.shape[1]):
        distances.append(lf.l(poincareSeeds[i]-center))
    return distances


def streamTwistDis(ss, lMin =1000):
    """
    returns the twists and the distances from the origin. 
    """
    twists = []
    distances = []
    for s in ss:
        if s.l > lMin:
            twist=lf.getTwist2(s)
            twists.append(twist)
            distances.append(lf.l(s.tracers[0]))
    return twists, distances


