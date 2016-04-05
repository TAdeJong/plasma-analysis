#
# linefn.py linefunctions
#
# This file contains a set of functions that operate on streamlines from the
#PENCIL-CODE in order to topologically analyze the field configurations
#
# functions range from finding the geometrical center of all points on the line
# to finding the winding number of the configuration
#
# assumes objects of the class stream as written by candelaresi in
# streamlines.py
#
# coded by Chris Smiet (csmiet) on sep 10 2014



import numpy as np
import glob
import re

def getCenter(streamline):
    """
    this function returns the geometrical center of all points in the stream
    line
    """
    center =  np.array([.0,.0,.0])
    center[0]=np.sum(streamline.tracers[:,0])/np.size(streamline.tracers[:,0])
    center[1]=np.sum(streamline.tracers[:,1])/np.size(streamline.tracers[:,1])
    center[2]=np.sum(streamline.tracers[:,2])/np.size(streamline.tracers[:,2])
    return center

def getRadius(streamline,center):
    """
    this function returns the average distance of the points to the
    geometrical center
    """
    radius=.0
    radius = np.sum(np.sqrt(np.sum((streamline.tracers - center)**2 ,axis=1)))/np.size(streamline.tracers[:,0])
    return radius


def getRadius2(streamline):
    """
    this function returns the average distance of the points to the
    geometrical center, without the center being previously calculated
    """
    center = getCenter(streamline)
    return (np.sum(np.sqrt(np.sum((streamline.tracers - center)**2 ,axis=1)))/np.size(streamline.tracers[:,0]))

def getCRN(streamline):
    center = getCenter(streamline)
    radius = getRadius(streamline, center)
    normal = getNormal(streamline, center)
    return center, radius, normal

def getNormal(streamline, center):
    """
    this function calculates the normal vector, the orientation of the torus
    it does this by calculating the cross product between the difference vector and the 
    vector to the point, and averaging these vectors weighedly
    """
    differenceVectors = streamline.tracers[1:] - streamline.tracers[:-1]  #calculate the vectors from each linepoint to the next
    vectors = np.cross((streamline.tracers - center)[:-1], differenceVectors) #calculate the cross product between the vector fom the center to the point, and the point to the next
    vectors = vectors/np.sqrt(np.sum((vectors)**2, axis=1))[:,np.newaxis]     #average them to get the normal vector (incorrectly done because all have to be counted equally :/)
    normalsum = np.sum(vectors, axis =0)
    normal = normalsum/np.sqrt(np.sum(normalsum**2)) 
    return normal 


def getCrossingNr(streamline, point, normal):
    """
    returns the number of times the streamline crosses the plane defined by the point and the normal vector.  
    """
    sides = np.dot( (streamline.tracers-point), normal)>0 #calculate the side of the plane each point falls on by projecting on the normal vector
    return np.sum((sides[:-1]^sides[1:]))                 #calculate the xor of the shifted sidetables (which is only true if one is shifted) and return the sum 


def getCrossings(streamline, point = [0., 0., 0.], normal=[1.,0.,0.]):
    """
    returns the positions in the streamline array where the crossings occur [I believe the array index before the crossings is returned, check this]
    """
    sides = np.dot( (streamline.tracers-point), normal)>0 #calculate the side of the plane each point falls on by projecting on the normal vector
    return (sides[:-1]^sides[1:]).nonzero()[0]                 #calculate the xor of the shifted sidetables (which is only true if one is shifted) and return the sum

def dToPlane(dispoint, point, normal):
    return  np.dot( (dispoint-point), normal)



def pointOnPlane(p1, p2, point, normal): 
    """
    calculate the point on the plane that is between the two points.
    """
    if np.sign(dToPlane(p1, point, normal))== np.sign(dToPlane(p2, point, normal)):
        print 'WARNING: POINTS NOT ON DIFFERENT SIDE OF PLANE'
    linevec = p1-p2 #vector along the line
    distance =(np.dot( (point - p1),normal))/(np.dot(linevec, normal)) #see wikipedia, Line-plane_intersection
    return distance*linevec + p1


#def perp(vector):
#    """
#    returns a vector perpendicular to the input. Not general, will give a vector of the type (a,b,0)  
#    """    
#    perpvector = np.array([1,1,-(vector[0] + vector[1])/vector[2]])
#    return  perpvector/np.sqrt(np.sum(perpvector**2))

def perp(vector):
    """
    returns a vector perpendicular to the input. Not general, will give a vector of the type (a,b,0)  
    """    
    perpvector = np.cross(vector, [1,0,0])
    return  perpvector/np.sqrt(np.sum(perpvector**2))

def perpZ(vector):
    """
    returns a vector perpendicular to the input vector, more readily in the z-direction
    """
    perpvector = np.array([-vector[2]/(vector[0]+vector[1]),-vector[2]/(vector[0]+vector[1]),1])
    return perpvector/np.sqrt(np.sum(perpvector**2))

def getTwist(streamline, center, normal):                 #returns the twist of the donut, defined as poloidal winding divided by poloidal winding
    poloidal = getCrossingNr(streamline, center, normal)  #calculate poloidal winding (times two, as it counts every crossing)
    #print poloidal
    toroidal = getCrossingNr(streamline, center, perp(normal)) # calculate toroidal winding (crossings of a plan perpendicular to the normal vector)
    #print toroidal
    twist = np.nan if toroidal == 0 else float(poloidal)/toroidal
    return twist 



def planeCrossingPoints(streamline, point, normal, verbose = 0):
    """
    returns the points on the plane (defined by point and normal)
    where the streamline crosses
    """
    #calculate two orthonormal vectors in the plane where we are looking for
    #the intersection    
    x1=perpZ(normal) #calculate a vector perpendicular to the normal
    x2=np.cross(normal, x1) # and another vector perpendicular to both
    x2/=l(x2)
    if verbose: print x1, x2
    crossingsX1X2 = []
    crossingIndices=getCrossings(streamline, point, normal)
    for i in crossingIndices:
        crossCoord= pointOnPlane(streamline.tracers[i], streamline.tracers[i+1], point, normal)
        crossingsX1X2.append([np.dot(x1,crossCoord), np.dot(x2,crossCoord)])


    return np.array(crossingsX1X2)

def getTwist2(streamline, verbose = 1):                                #returns the twist if center and normal have not been calculated yet)
    center = getCenter(streamline)
    normal = getNormal(streamline, center)
    poloidal = getCrossingNr(streamline, center, normal)  #calculate poloidal winding (times two, as it counts every crossing)
    if verbose: print 'poloidal winding is %d' %poloidal
    toroidal = getCrossingNr(streamline, center, perp(normal)) # calculate toroidal winding (crossings of a plan perpendicular to the normal vector)
    if verbose: print 'toroidal winding is %d' %toroidal
    twist = np.nan if toroidal == 0 else float(poloidal)/toroidal
    return twist



def l(vec):
    return (np.sqrt(np.sum(vec**2)))

def unitl(vec):
    return vec/l(vec)

def twistAlongLine(streamline, n, streamlength, linelength):
    return 0 
    
    
def twistInTime(animations, seed):
    return 0

def radiusInTime(animations, seed):
    radii = []
    pos=seed
    for fn in animations:
        print fn
        pos = fitPOT(fn, 0.01, 'simple', pos)
        radii.append(l(pos))
    return radii

def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

def animationsmaker(path):  
    return sorted(glob.glob(path + '*.vtk'), key = natural_key)

