import numpy as np
import cmath
from numpy import random


class Model:
    
    def __init__( self,ini, comm ):
        
        self.comm = comm
          
        self.mydir = ini.get("model","dir")
        self.pq_scale = ini.getfloat("model","pq_scale",0.1)
        self.loadscale = ini.getfloat("model","load_scale",1.0)
        self.thresh = ini.getfloat("model","thresh",0.3)
        self.eps = ini.getfloat("model","epsilon",0.05)
        self.seed = ini.getint("model","seed",0)
        self.use_ic = ini.getboolean("model","ic",False)
        
    def config(self): 
         
        self.from_to = np.loadtxt( self.mydir + "/from_to.txt" , dtype="int")
        self.line_susceptances = np.loadtxt( self.mydir + "/line_susc.txt" )
        self.pq_data = np.loadtxt( self.mydir + "/pq_data.txt" )
        self.genlist = np.loadtxt( self.mydir + "/genlist.txt" , dtype="int")
        self.loadlist = np.loadtxt( self.mydir + "/loadlist.txt" , dtype="int")
        self.slacklist = np.loadtxt( self.mydir + "/slacklist.txt" , dtype="int")
        
        
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
        
        self.sigmaP = np.sqrt( np.abs( self.P ) ) * np.sqrt(self.eps)
        self.sigmaQ = np.sqrt( np.abs( self.Q ) ) * np.sqrt(self.eps)
        
        self.bb = self.line_susceptances 
        
        #self.BB =    np.dot( self.AA , np.diag( self.bb ) ) 
        #self.BB = np.dot (self.BB , self.AA.T ) 
    
    def assemble_ybus(self, g):
         
        
        BB =    np.dot( self.AA , np.diag( self.bb * g ) ) 
        BB = np.dot (BB , self.AA.T ) 
        
        return BB
    
        
    def dH_dfreq( self, f ):
    
        return f
    
    def dH_dangle( self, a , m , ybus ): 
        
        v = m * np.exp( 1j * a )
    
        factor = v * ( np.dot(ybus , v ) ).conjugate()
        
        ff = self.P - factor.imag
        ff[ self.slacklist -1 ] = 0
        
        return ff
    
    def dH_dmag(self,  a , m , ybus ):
    
        v = m * np.exp( 1j * a )
        factor = (v) * ( np.dot(ybus , v ) ).conjugate()
        
        ff = (self.Q + factor.real)/m
        ff[ self.slacklist -1 ] = 0
        ff[ self.genlist -1 ] = 0
        
        return ff
    
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
        
        v = mag * np.exp( 1j * angle )
        
        Yv = np.dot( ybus , v )
        k1 = (v.conjugate() * Yv )*0.5
        k1 = np.sum(k1)
        
        k2 = np.dot( freq , freq ) * 0.5
        
        k3 = np.dot( self.P , angle )
        
        k4 = np.dot( self.Q , np.log(mag) )
        
        Energy = k1+k2+k3+k4
        
        Av = np.dot( self.AA.T , v )
        
        LE = (Av.conjugate() * Av) * self.bb * gamma
        
        
        return Energy.real , LE.real
    
    
    
    def getic_gamma(self):
        
        return np.ones( self.nline )
        
    
    
    def getic_freq(self):
        
        return np.copy( self.ic_f )
        
    
    def getic_angle(self):
        
        return np.copy( self.ic_a )
        
    
    def getic_mag(self):
        
        return np.copy( self.ic_m )
    
    def checklines(self, le , gamma ):
        
        nle = gamma * le / self.bb
        
        overlim = nle > self.thresh
        
        isover = np.sum(overlim)>0
        
        return isover,overlim
    
    def removeline( self, g, ol ):
        
        gg = g - ol
        gg = [ int(x) for x in gg ]
        
        inum = np.arange(self.nbus)
        
        for id,ig in enumerate(gg):
            
            if ig<1: 
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
        
          
        for id,ig in enumerate(gg):
            
            if ig<1: 
                continue
            
            ifrom = self.from_to[ id, 0 ] -1
            ito =   self.from_to[ id ,1 ] -1
            
            if (inum[ ifrom ] != islack ):
                gg[id] = 0
                
            if (inum[ ito ] != islack ):
                gg[id] = 0
             
        #print nactive
        #print self.Pload
        
        new_ls =  np.sum( self.Pload[ nactive ] ) / np.sum( self.Pload ) 
        
        return gg, new_ls
            
    