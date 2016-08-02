import numpy as np
import cmath   

class TrajSampler:
    
    def __init__( self,ini, comm, model, integrator, output ):
        
        self.ini = ini
        self.comm = comm
        self.model = model
        self.ig = integrator
        self.output = output
          
        self.steps = ini.getint("trajsampler","steps")  
        self.walkers = ini.getint("trajsampler","walkers",1) 
        
        
    def config(self):
        
        pass
            
    def run(self):
        
        MyTasks = np.arange( self.comm.Get_rank(), self.walkers, self.comm.Get_size() )
        
        for task in MyTasks:
        
            ii = 0
        
            gamma = self.model.getic_gamma()
            
            f = self.model.getic_freq()
            a = self.model.getic_angle()
            m = self.model.getic_mag()
            
            FF = np.zeros( ( len(f), self.steps ) )
            AA = np.zeros( ( len(f), self.steps ) )
            MM = np.zeros( ( len(f), self.steps ) )
            EN = np.zeros( (  self.steps ) )
            LE = np.zeros( ( self.model.nline, self.steps ) )
            G = np.zeros( ( self.model.nline, self.steps ) )
            
            
            while ii < self.steps :
                 
                
                f,a,m,F,A,M,E,L,jj,g = self.ig.adv( f, a , m , self.steps - ii , gamma )
                
                FF[:,ii:] = np.copy(F)
                AA[:,ii:] = np.copy(A)
                MM[:,ii:] = np.copy(M)
                EN[ii:] = np.copy(E)
                LE[:,ii:] = np.copy(L)
                G[:,ii:] = np.tile(gamma, (self.steps-ii,1) ).T
                
                
                ii += jj
                
                gamma = g
                
            T = np.arange(1, self.steps+1) * self.ig.dt
            
            self.output.AddOutput( task, FF, AA, MM,EN,LE, G, T )
            