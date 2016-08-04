import numpy as np
import cmath   

class TrajSampler:
    
    def __init__( self,ini, comm, model, integrator, output ):
        
        self.ini = ini
        self.comm = comm
        self.model = model
        self.ig = integrator
        self.output = output
          
        self.stoptime = self.ig.stoptime
        self.stoplines = self.ig.stoplines
        self.stopload = self.ig.stopload
        self.walkers = ini.getint("trajsampler","walkers",1) 
        
        
    def config(self):
        
        self.steps = 1 + int( self.stoptime / self.ig.dt )
            
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
            G = np.tile(gamma, (self.steps,1) ).T
            LS = np.ones( (  self.steps ) )
            
            anodes = np.ones( self.model.nbus )>0
            
            time = 0 
            ls = 1.0
            
            
            while self.ig.keepgoing( time , gamma, ls  ) :
                 
                
                f,a,m,F,A,M,E,L,jj,g,nls, anodes = self.ig.adv( f, a , m , self.steps - ii , gamma, anodes )
                
                FF[:,ii:] = np.copy(F)
                AA[:,ii:] = np.copy(A)
                MM[:,ii:] = np.copy(M)
                EN[ii:] = np.copy(E)
                LE[:,ii:] = np.copy(L)
                ii += jj
                time += jj * self.ig.dt
                
                if nls>=0:
                    LS[ii:] = nls 
                    ls = nls
                
                gamma = g
                G[:,ii:] = np.tile(gamma, (self.steps-ii,1) ).T
                
            T = np.arange(1, ii+1) * self.ig.dt
            FF = FF[:,:ii]
            AA = AA[:,:ii]
            MM = MM[:,:ii]
            EN = EN[:ii]
            LE = LE[:,:ii]
            G = G[:,:ii]
            
            self.output.AddOutput( task, FF, AA, MM,EN,LE, G, T, LS )
            