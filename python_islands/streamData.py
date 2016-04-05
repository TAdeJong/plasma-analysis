#
# stream.py
#
# Streamline routines.

import numpy as np
import vtk as vtk
import pencil as pc
from vtk.util import numpy_support as VN
from os import listdir

class poincareResults:
    """
    poincareResults -- class that holds all information on a given animation
    """
    def __init__(self, streamlines, poincarePoints, radius, time, center, twists, twistpositions, distances):
        self.streamlines = streamlines; self.poincarePoints = poincarePoints
        self.radius = radius ; self.time = time; self.center = center
        self.twists = twists
        self.twistpositions = twistpositions; self.distances = distances


class streamSingle:
    """
    streamSingle -- Holds the traced streamline.
    """
    
    def __init__(self, vv, p, interpolation = 'weighted', integration = 'RK6', hMin = 2e-3, hMax = 2e3, lMax = 500, tol = 1e-5, iterMax = 1e8, xx = np.array([0,0,0])):
        """
        Creates, and returns the traced streamline.
        
        call signature:
        
          streamInit(vv, p, interpolation = 'weighted', integration = 'simple', hMin = 2e-3, hMax = 2e4, lMax = 500, tol = 1e-2, iterMax = 1e3, xx = np.array([0,0,0])):
        
        Trace streamlines.
        
        Keyword arguments:
        
         *vv*:
            Vector field to be traced.
         
         *p*:
            Struct containing simulation parameters.
            
         *interpolation*:
            Interpolation of the vector field.
            'mean': takes the mean of the adjacent grid point.
            'weighted': weights the adjacent grid points according to their distance.
       
         *integration*:
            Integration method.
            'simple': low order method.
            'RK6': Runge-Kutta 6th order.
       
         *hMin*:
            Minimum step length for and underflow to occur.
        
         *hMax*:
            Parameter for the initial step length.
        
         *lMax*:
            Maximum length of the streamline. Integration will stop if l >= lMax.
        
         *tol*:
            Tolerance for each integration step. Reduces the step length if error >= tol.
         
         *iterMax*:
            Maximum number of iterations.     
         
         *xx*:
            Initial seed.
        """
        
        self.tracers = np.zeros([iterMax, 3], dtype = 'float32')  # tentative streamline length
        
        tol2 = tol**2
        dh   = np.sqrt(hMax*hMin) # initial step size
        
        # declare vectors
        xMid    = np.zeros(3)
        xSingle = np.zeros(3)
        xHalf   = np.zeros(3)
        xDouble = np.zeros(3)
        
        # initialize the coefficient for the 6th order adaptive time step RK
        a = np.zeros(6); b = np.zeros((6,5)); c = np.zeros(6); cs = np.zeros(6)
        k = np.zeros((6,3))
        a[1] = 0.2; a[2] = 0.3; a[3] = 0.6; a[4] = 1; a[5] = 0.875
        b[1,0] = 0.2;
        b[2,0] = 3/40.; b[2,1] = 9/40.
        b[3,0] = 0.3; b[3,1] = -0.9; b[3,2] = 1.2
        b[4,0] = -11/54.; b[4,1] = 2.5; b[4,2] = -70/27.; b[4,3] = 35/27.
        b[5,0] = 1631/55296.; b[5,1] = 175/512.; b[5,2] = 575/13824.
        b[5,3] = 44275/110592.; b[5,4] = 253/4096.
        c[0] = 37/378.; c[2] = 250/621.; c[3] = 125/594.; c[5] = 512/1771.
        cs[0] = 2825/27648.; cs[2] = 18575/48384.; cs[3] = 13525/55296.
        cs[4] = 277/14336.; cs[5] = 0.25
    
        # do the streamline tracing
        self.tracers[0,:] = xx
        outside = False
        sl = 0
        l = 0
                
        if (integration == 'simple'):
            while ((l < lMax) and (sl < iterMax-1) and (not(np.isnan(xx[0]))) and (outside == False)):
                # (a) single step (midpoint method)                    
                xMid = xx + 0.5*dh*vecInt(xx, vv, p, interpolation)
                xSingle = xx + dh*vecInt(xMid, vv, p, interpolation)
            
                # (b) two steps with half stepsize
                xMid = xx + 0.25*dh*vecInt(xx, vv, p, interpolation)
                xHalf = xx + 0.5*dh*vecInt(xMid, vv, p, interpolation)
                xMid = xHalf + 0.25*dh*vecInt(xHalf, vv, p, interpolation)
                xDouble = xHalf + 0.5*dh*vecInt(xMid, vv, p, interpolation)
            
                # (c) check error (difference between methods)
                dist2 = np.sum((xSingle-xDouble)**2)
                if (dist2 > tol2):
                    dh = 0.5*dh
                    if (abs(dh) < hMin):
                        print "Error: stepsize underflow"
                        break
                else:
                    l += np.sqrt(np.sum((xx-xDouble)**2))
                    xx = xDouble.copy()
                    if (abs(dh) < hMin):
                        dh = 2*dh
                    sl += 1
                    self.tracers[sl,:] = xx.copy()
                    if ((dh > hMax) or (np.isnan(dh))):
                        dh = hMax
                    # check if this point lies outside the domain
                    if ((xx[0] < p.Ox) or (xx[0] > p.Ox+p.Lx) or (xx[1] < p.Oy) or (xx[1] > p.Oy+p.Ly) or (xx[2] < p.Oz) or (xx[2] > p.Oz+p.Lz)):
                        outside = True
                        
        if (integration == 'RK6'):
            while ((l < lMax) and (sl < iterMax-1) and (not(np.isnan(xx[0]))) and (outside == False)):
                k[0,:] = dh*vecInt(xx, vv, p, interpolation)                            
                k[1,:] = dh*vecInt(xx + b[1,0]*k[0,:], vv, p, interpolation)
                k[2,:] = dh*vecInt(xx + b[2,0]*k[0,:] + b[2,1]*k[1,:], vv, p, interpolation)
                k[3,:] = dh*vecInt(xx + b[3,0]*k[0,:] + b[3,1]*k[1,:] + b[3,2]*k[2,:], vv, p, interpolation)
                k[4,:] = dh*vecInt(xx + b[4,0]*k[0,:] + b[4,1]*k[1,:] + b[4,2]*k[2,:] + b[4,3]*k[3,:], vv, p, interpolation)
                k[5,:] = dh*vecInt(xx + b[5,0]*k[0,:] + b[5,1]*k[1,:] + b[5,2]*k[2,:] + b[5,3]*k[3,:] + b[5,4]*k[4,:], vv, p, interpolation)

                xNew  = xx + c[0]*k[0,:]  + c[1]*k[1,:]  + c[2]*k[2,:]  + c[3]*k[3,:]  + c[4]*k[4,:]  + c[5]*k[5,:]
                xNewS = xx + cs[0]*k[0,:] + cs[1]*k[1,:] + cs[2]*k[2,:] + cs[3]*k[3,:] + cs[4]*k[4,:] + cs[5]*k[5,:]

                delta2 = np.dot((xNew-xNewS), (xNew-xNewS))
                delta = np.sqrt(delta2)

                if (delta2 > tol2):
                    dh = dh*(0.9*abs(tol/delta))**0.2
                    if (abs(dh) < hMin):
                        print "Error: step size underflow"
                        break
                else:
                    l += np.sqrt(np.sum((xx-xNew)**2))
                    xx = xNew                        
                    if (abs(dh) < hMin):
                        dh = 2*dh
                    sl += 1
                    self.tracers[sl,:] = xx
                    if ((dh > hMax) or (np.isnan(dh))):
                        dh = hMax
                    # check if this point lies outside the domain
                    if ((xx[0] < p.Ox) or (xx[0] > p.Ox+p.Lx) or (xx[1] < p.Oy) or (xx[1] > p.Oy+p.Ly) or (xx[2] < p.Oz) or (xx[2] > p.Oz+p.Lz)):
                        outside = True
                if ((dh > hMax) or (delta == 0) or (np.isnan(dh))):
                    dh = hMax
        
        self.tracers = np.resize(self.tracers, (sl, 3))
        self.l = l
        self.sl = np.int(sl)
        self.p = p
      
      
class stream:
    """
    stream -- Holds the array of streamlines.
    """
    
    def __init__(self, dataDir = 'data/', fileName = 'var.dat', streamFile = 'stream.vtk', interpolation = 'weighted', integration = 'RK6', hMin = 2e-3, hMax = 2e4, lMax = 500, tol = 1e-2, iterMax = 1e3, xx = np.array([0,0,0])):
        """
        Creates, and returns the traced streamline.
        
        call signature:
        
          streamInit(datadir = 'data/', fileName = 'save.dat, interpolation = 'weighted', integration = 'simple', hMin = 2e-3, hMax = 2e4, lMax = 500, tol = 1e-2, iterMax = 1e3, xx = np.array([0,0,0]))
        
        Trace magnetic streamlines.
        
        Keyword arguments:
        
         *dataDir*:
            Data directory.
            
         *fileName*:
            Name of the file with the field information.
            
         *interpolation*:
            Interpolation of the vector field.
            'mean': takes the mean of the adjacent grid point.
            'weighted': weights the adjacent grid points according to their distance.
       
         *integration*:
            Integration method.
            'simple': low order method.
            'RK6': Runge-Kutta 6th order.
       
         *hMin*:
            Minimum step length for and underflow to occur.
        
         *hMax*:
            Parameter for the initial step length.
        
         *lMax*:
            Maximum length of the streamline. Integration will stop if l >= lMax.
        
         *tol*:
            Tolerance for each integration step. Reduces the step length if error >= tol.
         
         *iterMax*:
            Maximum number of iterations.     
         
         *xx*:
            Initial seeds.
        """
        
        # read the data
        var = pc.read_var(datadir = dataDir, varfile = fileName, magic = 'bb', quiet = True, trimall = True)
        grid = pc.read_grid(datadir = dataDir, quiet = True)
        
        vv = var.bb
        
        p = pClass()
        p.dx = var.dx; p.dy = var.dy; p.dz = var.dz        
        p.Ox = var.x[0]; p.Oy = var.y[0]; p.Oz = var.z[0]
        p.Lx = grid.Lx; p.Ly = grid.Ly; p.Lz = grid.Lz
        p.nx = var.bb.shape[1]; p.ny = var.bb.shape[2]; p.nz = var.bb.shape[3]

        ss = []
        for i in range(xx.shape[1]):
            s = streamSingle(vv, p, interpolation = 'weighted', integration = 'simple', hMin = hMin, hMax = hMax, lMax = lMax, tol = tol, iterMax = iterMax, xx = xx[:,i])
            ss.append(s)
        slMax = 0
        for i in range(xx.shape[1]):
            if (slMax < ss[i].sl):
                slMax = ss[i].sl
        self.tracers = np.zeros((xx.shape[1], slMax, 3)) + np.nan
        self.sl = np.zeros(xx.shape[1], dtype = 'int32')
        self.l = np.zeros(xx.shape[1])
        for i in range(xx.shape[1]):
            self.tracers[i,:ss[i].sl,:] = ss[i].tracers
            self.sl[i] = ss[i].sl
            self.l[i] = ss[i].l
        self.p = s.p
        self.nt = xx.shape[1]
        
        # save into vtk file
        if (streamFile != []):
            writer = vtk.vtkPolyDataWriter()
            writer.SetFileName(dataDir + '/' + streamFile)
            polyData = vtk.vtkPolyData()
            fieldData = vtk.vtkFieldData()
            # field containing length of stream lines for later decomposition
            field = VN.numpy_to_vtk(self.l)
            field.SetName('l')
            fieldData.AddArray(field)
            field = VN.numpy_to_vtk(self.sl.astype(np.int32))
            field.SetName('sl')
            fieldData.AddArray(field)
            # streamline parameters
            tmp = range(10)            
            tmp[0] = np.array([hMin], dtype = 'float32'); field = VN.numpy_to_vtk(tmp[0]); field.SetName('hMin'); fieldData.AddArray(field)
            tmp[1] = np.array([hMax], dtype = 'float32'); field = VN.numpy_to_vtk(tmp[1]); field.SetName('hMax'); fieldData.AddArray(field)
            tmp[2] = np.array([lMax], dtype = 'float32'); field = VN.numpy_to_vtk(tmp[2]); field.SetName('lMax'); fieldData.AddArray(field)
            tmp[3] = np.array([tol], dtype = 'float32'); field = VN.numpy_to_vtk(tmp[3]); field.SetName('tol'); fieldData.AddArray(field)
            tmp[4] = np.array([iterMax], dtype = 'int32'); field = VN.numpy_to_vtk(tmp[4]); field.SetName('iterMax'); fieldData.AddArray(field)
            tmp[5] = np.array([self.nt], dtype = 'int32'); field = VN.numpy_to_vtk(tmp[5]); field.SetName('nt'); fieldData.AddArray(field)
            # fields containing simulation parameters stored in paramFile
            dic = dir(p)
            params = range(len(dic))
            i = 0
            for attr in dic:
                if( attr[0] != '_'):
                    params[i] = getattr(p, attr)
                    params[i] = np.array([params[i]], dtype = type(params[i]))
                    field = VN.numpy_to_vtk(params[i])
                    field.SetName(attr)
                    fieldData.AddArray(field)
                    i += 1
            # all streamlines as continuous array of points
            points = vtk.vtkPoints()
            for i in range(xx.shape[1]):
                for sl in range(self.sl[i]):
                    points.InsertNextPoint(self.tracers[i,sl,:])
            polyData.SetPoints(points)
            polyData.SetFieldData(fieldData)
            writer.SetInput(polyData)
            writer.SetFileTypeToBinary()
            writer.Write()


class readStream:
    """
    readStream -- Holds the streamlines.
    """

    def __init__(self, dataDir = 'data', streamFile = 'stream.vtk'):
        """
        Read the initial streamlines.
        
        call signature:
        
          readStream(dataDir = 'data', streamFile = 'stream.vtk')
          
        Keyword arguments:
         *dataDir*:
            Data directory.
            
         *streamFile*:
            Read the initial streamline from this file.
        """
    
        # load the data
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(dataDir + '/' + streamFile)
        reader.Update()
        output = reader.GetOutput()
        
        # get the fields
        field = output.GetFieldData()
        nArrays = field.GetNumberOfArrays()
        class params: pass
        p = params()
        for i in range(nArrays):            
            arrayName = field.GetArrayName(i)
            if any(arrayName == np.array(['l', 'sl'])):
                setattr(self, arrayName, VN.vtk_to_numpy(field.GetArray(arrayName)))
            elif any(arrayName == np.array(['hMin', 'hMax', 'lMax', 'tol', 'iterMax', 'nt'])):
                setattr(self, arrayName, VN.vtk_to_numpy(field.GetArray(arrayName))[0])
            else:
                # change this if parameters can have more than one entry
                setattr(p, arrayName, VN.vtk_to_numpy(field.GetArray(arrayName))[0])
        setattr(self, 'p', p)
        
        # get the points
        points = output.GetPoints()
        pointsData = points.GetData()
        data = VN.vtk_to_numpy(pointsData)
        #data = np.swapaxes(data, 0, 1)
        print self.nt
        print self.sl
        print data.shape
        tracers = np.zeros([self.nt, np.max(self.sl), 3], dtype = data.dtype)
        sl = 0
        for i in range(self.nt):
            #if (i > 0):
                #sl = self.sl[i-1]
            #else:
                #sl = 0
            print sl, self.sl[i]
            tracers[i,:self.sl[i],:] = data[sl:sl+self.sl[i],:]
            sl += self.sl[i]
        setattr(self, 'tracers', tracers)
def vecInt(xx, vv, p, interpolation = 'weighted'):
    """
    Interpolates the field around this position.
    
    call signature:
    
        vecInt(xx, vv, p, interpolation = 'weighted')
    
    Keyword arguments:
    
    *xx*:
      Position vector around which will be interpolated.
    
    *vv*:
      Vector field to be interpolated.
    
    *p*:
      Parameter struct.
    
    *interpolation*:
      Interpolation of the vector field.
      'mean': takes the mean of the adjacent grid point.
      'weighted': weights the adjacent grid points according to their distance.
    """
    
    # find the adjacent indices
    i  = (xx[0]-p.Ox)/p.dx
    if (i < 0):
        i = 0
    if (i > p.nx-1):
        i = p.nx-1
    ii = np.array([int(np.floor(i)), \
                    int(np.ceil(i))])
    
    j  = (xx[1]-p.Oy)/p.dy    
    if (j < 0):
        j = 0
    if (j > p.ny-1):
        j = p.ny-1
    jj = np.array([int(np.floor(j)), \
                    int(np.ceil(j))])
    
    k  = (xx[2]-p.Oz)/p.dz
    if (k < 0):
        k = 0
    if (k > p.nz-1):
        k = p.nz-1
    kk = np.array([int(np.floor(k)), \
                    int(np.ceil(k))])
    
    vv = np.swapaxes(vv, 1, 3)
    # interpolate the field
    if (interpolation == 'mean'):
        return np.mean(vv[:,ii[0]:ii[1]+1,jj[0]:jj[1]+1,kk[0]:kk[1]+1], axis = (1,2,3))
    if(interpolation == 'weighted'):
        if (ii[0] == ii[1]): w1 = np.array([1,1])
        else: w1 = (i-ii[::-1])
        if (jj[0] == jj[1]): w2 = np.array([1,1])
        else: w2 = (j-jj[::-1])
        if (kk[0] == kk[1]): w3 = np.array([1,1])
        else: w3 = (k-kk[::-1])
        weight = abs(w1.reshape((2,1,1))*w2.reshape((1,2,1))*w3.reshape((1,1,2)))
        return np.sum(vv[:,ii[0]:ii[1]+1,jj[0]:jj[1]+1,kk[0]:kk[1]+1]*weight, axis = (1,2,3))/np.sum(weight)


def readField(simdir, varfile):
        var = pc.read_var(datadir = simdir,  varfile = varfile, magic = 'bb', quiet = True, trimall = True)
        grid = pc.read_grid(datadir = simdir, quiet = True)
        
        bb = var.bb
        
        p = pClass()
        p.dx = var.dx; p.dy = var.dy; p.dz = var.dz        
        p.Ox = var.x[0]; p.Oy = var.y[0]; p.Oz = var.z[0]
        p.Lx = grid.Lx; p.Ly = grid.Ly; p.Lz = grid.Lz
        p.nx = var.bb.shape[1]; p.ny = var.bb.shape[2]; p.nz = var.bb.shape[3]
        return bb, p, var.t
    

# class containing simulation parameters
class pClass:
    def __init__(self):
        self.dx = 0; self.dy = 0; self.dz = 0
        self.Ox = 0; self.Oy = 0; self.Oz = 0
        self.Lx = 0; self.Ly = 0; self.Lz = 0
        self.nx = 0; self.ny = 0; self.nz = 0
        
