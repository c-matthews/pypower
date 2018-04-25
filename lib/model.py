import numpy as np
import cmath
from numpy import random
import time
from scipy.sparse import csr_matrix, csc_matrix
from sys import exit

class Model:
    
    def __init__( self,ini, comm ):
        
        self.comm = comm
          
        self.mydir = ini.get("model","dir")
        self.pq_scale = ini.getfloat("model","pq_scale",0.1)
        self.loadscale = ini.getfloat("model","load_scale",1.0)
        self.thresh = ini.getfloat("model","thresh",0.3)
        self.eps = ini.getfloat("model","epsilon",0.05)
        self.kt = ini.getfloat("model","kt",1.0)
        self.seed = ini.getint("model","seed",0)
        self.use_ic = ini.getboolean("model","ic",False)
        self.takedown = ini.getint("model","remline",0) -1
        self.nodedown = ini.getint("model","remnode",0) -1
        self.burntime = ini.getfloat("model","burntime",0) 
        
    def config(self): 
         
        self.from_to = np.loadtxt( self.mydir + "/from_to.txt" , dtype="int")
        self.line_susceptances = np.loadtxt( self.mydir + "/line_susc.txt" )
        self.pq_data = np.loadtxt( self.mydir + "/pq_data.txt" )
        self.genlist = np.loadtxt( self.mydir + "/genlist.txt" , dtype="int")
        self.loadlist = np.loadtxt( self.mydir + "/loadlist.txt" , dtype="int")
        self.slacklist = np.loadtxt( self.mydir + "/slacklist.txt" , dtype="int")
        
        if self.seed==0:
            self.seed = int( time.time() ) % 100000
            self.seed  = self.comm.bcast( self.seed  , root=0 )
            
        self.seed = self.seed + self.comm.Get_rank()
        self.Random = random.RandomState( self.seed )
        
        self.nbus = int(np.max( self.from_to ))
        self.nline = np.shape( self.from_to )[0]
        
        if self.use_ic: 
            self.ic = np.loadtxt( self.mydir + "/ic.txt" )
            self.ic_f = self.ic[0:self.nbus]
            self.ic_a = self.ic[self.nbus:2*self.nbus]
            self.ic_m = self.ic[2*self.nbus:3*self.nbus]
        else:
            self.ic_f = np.zeros( self.nbus )
            self.ic_a = np.zeros( self.nbus )
            self.ic_m = np.ones( self.nbus )
        
        
        self.AA = np.zeros( (self.nbus, self.nline) )
        
        for jj in np.arange(self.nline):
            i1,i2 = self.from_to[jj,:]-1
            self.AA[ i1 , jj ] = 1.
            self.AA[ i2 , jj ] = -1.

        self.sAAt = csr_matrix( self.AA.T )
            
        self.pq_data[ self.slacklist-1 , : ] = [0,0,0]
        self.pq_data[ self.slacklist-1 , : ] = -np.sum( self.pq_data, axis=0 )
        self.pq_data *= self.pq_scale 
        self.pq_data[ self.loadlist -1 ] *= self.loadscale
        self.snet = self.pq_data[:,0] + self.pq_data[:,1] - 1j * self.pq_data[:,2]
        self.Pload = self.pq_data[:,1] 
        self.Pload[ self.slacklist-1 ] = 0
        self.P = self.snet.real
        self.Q = self.snet.imag
        self.Q[ self.genlist-1 ] = 0
        
        #  self.sigmaP = self.noisemag*np.sqrt( np.abs( self.P ) ) * np.sqrt(self.eps)
        #  self.sigmaQ = self.noisemag*np.sqrt( np.abs( self.Q ) ) * np.sqrt(self.eps)
        self.sigmaP =    np.sqrt(self.kt*2*self.eps)
        self.sigmaQ =    np.sqrt(self.kt*2*self.eps)
        
        self.bb = self.line_susceptances 
        self.oobb = 1.0 / self.bb
        self.oobbt = self.thresh / self.oobb

        #self.BB =    np.dot( self.AA , np.diag( self.bb ) ) 
        #self.BB = np.dot (self.BB , self.AA.T )

        # keep track of the time (in seconds)
        self.time = 0
    
    def assemble_ybus(self, g):
         
        
        BB =    np.dot( self.AA , np.diag( self.bb * g ) ) 
        BB = np.dot (BB , self.AA.T ) 
        
        sBB = csr_matrix( BB )

        return sBB
    
        
    def dH_dfreq( self, f ):
    
        return f
    
    def dH_dangle( self, a , m , ybus ): 
        
        v = m * np.exp( 1j * a )
    
        #factor = v * ( np.dot(ybus , v ) ).conjugate()
        factor = v * ( ybus.dot( v ) ).conjugate()

        ff = self.P - factor.imag

        #P=self.P
        #print P[0] -
        #zz1 = P[0] + 0
        #zz2 = P[0] + 0
        #zz3 = P[0] + 0 

        #print ybus[0,:]
        
        #for jj in np.arange( self.nbus) :
        #    zz1 += m[0] * m[jj] * ybus[0,jj] * (np.cos( a[0] - a[jj] ) - np.sin( a[0] - a[jj]) )
        #    zz2 += m[0] * m[jj] * ybus[0,jj] * (np.cos( a[0] - a[jj] ) + 0*np.sin( a[0] - a[jj]) )
        #    zz3 += m[0] * m[jj] * ybus[0,jj] * (0*np.cos( a[0] - a[jj] ) - np.sin( a[0] - a[jj]) )
            
        #print ""
        #print [ff[0],zz1,zz2,zz3]
        #print ""
        
        #exit()
        
        
        
        ff[ self.slacklist -1 ] = 0
        
        return ff
    
    def dH_dmag(self,  a , m , ybus ):
    
        v = m * np.exp( 1j * a )
        #factor = (v) * ( np.dot(ybus , v ) ).conjugate()
        factor = (v) * ( ybus.dot(v) ).conjugate()
        
        ff = (self.Q + factor.real)/m
        ff[ self.slacklist -1 ] = 0
        ff[ self.genlist -1 ] = 0
        
        return ff
    
    def dH_danglemag( self, a , m , ybus ): 
        
        v = m * np.exp( 1j * a )
    
        #factor = v * ( np.dot(ybus , v ) ).conjugate()
        factor = v * ( ybus.dot(v) ).conjugate()
        
        fangle = self.P - factor.imag  
        fangle[ self.slacklist -1 ] = 0
        
        fmag = (self.Q + factor.real)/m
        fmag[ self.slacklist -1 ] = 0
        fmag[ self.genlist -1 ] = 0

        return fangle, fmag 
    
    def rand_angle(self  ):
    
        ff = self.sigmaP * self.Random.randn( self.nbus )
        ff[ self.slacklist -1 ] = 0 
        
        return ff
    
    def rand_mag(self  ):
    
        ff = self.sigmaQ * self.Random.randn( self.nbus )
        ff[ self.slacklist -1 ] = 0
        ff[ self.genlist -1 ] = 0
        
        return ff
    
    def energy(self, freq, angle, mag, ybus, gamma ):
        

        # Update load with time dependence
        # we start with a very simple model:
        #
        # P(t) = P_m * (1.0 + alpha*sin(\omega t))
        # where
        omega = np.pi/(1.5*3600)
        alpha = 0.5

        P = np.copy(self.P)
        Q = np.copy(self.Q)

        if True:
            P[self.loadlist - 1] = P[self.loadlist - 1]*(alpha*np.sin(omega*self.time) + 1)
 
        # compute energy served
        Pload = np.copy(P)
        Pload[self.genlist - 1] = 0
        Pload[self.slacklist - 1] = 0
        energy_served = -np.sum(Pload[self.nactive])

        v = mag * np.exp( 1j * angle )
        
        Yv = ybus.dot(v) 
        #Yv = np.dot( ybus , v )
        kk1 = (v.conjugate() * Yv ) 
        
        k1 = np.sum(kk1)*0.5
        
        k2 = np.dot(freq , freq)*0.5
        
        k3 = np.dot(P ,angle)

        k4 = np.dot(Q, np.log(mag))
            
        Energy = k1 + k2 + k3 + k4
        
        #Av = np.dot( self.AA.T , v )
        Av = self.sAAt.dot(v)
        
        LE = (Av.conjugate() * Av) * (self.bb * gamma) 
        
        df = freq
        da = P + kk1.imag
        dm = (Q + kk1.real)/mag
        da[ self.slacklist -1 ] = 0
        
        dm[ self.slacklist -1 ] = 0
        dm[ self.genlist -1 ] = 0


        #exit()

        return Energy.real , LE.real, df, da, dm, energy_served
    
    
    
    def getic_gamma(self):
        
        gg = np.ones( self.nline )
        
        if (self.takedown>=0):
            gg[ self.takedown ] = 0
            
        if (self.nodedown>=0):
            
            for jj in np.arange(self.nline):
                i1,i2 = self.from_to[jj,:]-1
                if (i1==self.nodedown) or (i2==self.nodedown):
                    gg[ jj ] = 0
        
        return gg
        
    
    
    def getic_freq(self):
        
        return np.copy( self.ic_f )
        
    
    def getic_angle(self):
        
        return np.copy( self.ic_a )
        
    
    def getic_mag(self):
        
        return np.copy( self.ic_m )
    
    def checklines(self, le , gamma ):
        
        #nle = (gamma * le) * self.oobb
        
        #nle = (gamma * le)
        # le already has gamma in it
         
        overmask = np.ones( np.shape(le) ) 

        overlim = (overmask*le) > self.oobbt
        
        isover = np.array(overlim).any()  
        
        if isover:
            for i in range(len(le)):
                if overlim[i]:
                    print "Line %d is over limit. le: %g, lim: %g" % (i, le[i], self.oobbt[i])
        if False:
            print str(160), self.time, le[160], self.oobbt[160]


        return isover,overlim
    
    def removeline( self, g, ol ):
        """
        ol: over limit vector
        g: 
        """
        
        gg = g - ol
        gg = [ int(x) for x in gg ]
        
        inum = np.arange(self.nbus)
        
        for id, ig in enumerate(gg):
            
            if ig < 1: 
                continue
            
            ifrom = self.from_to[ id, 0 ] -1
            ito =   self.from_to[ id ,1 ] -1
            
            nfrom = inum[ ifrom ]
            nto   = inum[ ito   ]
            
            nmin = np.min( [nfrom, nto] )
             
            
            inum[ inum==nfrom ] = nmin
            inum[ inum==nto   ] = nmin
               
        islack =  inum[ self.slacklist-1 ]
        
        nactive =   inum==islack 
        
          
        for id, ig in enumerate(gg):
            
            if ig<1: 
                continue
            
            ifrom = self.from_to[ id, 0 ] -1
            ito =   self.from_to[ id ,1 ] -1
            
            if (inum[ ifrom ] != islack ):
                gg[id] = 0
                
            if (inum[ ito ] != islack ):
                gg[id] = 0
             
        #print nactive
        #print np.sum(self.Pload[nactive])


        new_ls =  np.sum( self.Pload[ nactive ] ) / np.sum( self.Pload ) 

        self.nactive = nactive

        return gg, new_ls, nactive
            
    
