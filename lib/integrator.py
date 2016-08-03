import numpy as np
import cmath  

class Integrator:
    
    def __init__( self,ini, comm, model ):
        
        self.comm = comm
        self.model = model
          
        self.dt = ini.getfloat("integrator","timestep") 
        
    def config(self): 
          
        self.dt2 = self.dt * 0.5
        self.sqdt = np.sqrt(self.dt)
        
    def adv(self, freq, angle, mag, N , gamma ):
        
        ybus = self.model.assemble_ybus(gamma) 
        f = freq
        a = angle
        m = mag
        g = gamma
        
        F = np.zeros( (len(f), N) )
        A = np.zeros( (len(a), N) )
        M = np.zeros( (len(m), N) )
        
        EN = np.zeros( N )
        LE = np.zeros( (self.model.nline, N) )
        
        ii = 0
        nls = -1
        
        while ii<N :
            
            f,a,m = self.step(f,a,m, ybus)
            
            en,le = self.model.energy(f,a,m,ybus, g) 
            
            io,ol = self.model.checklines( le , g )
            
                
            
            F[:,ii] = f
            A[:,ii] = a
            M[:,ii] = m
            EN[ii] = en
            LE[:,ii] = le
            
            ii+=1
            
            if io: 
                print "fail!"
                print 1+np.arange(self.model.nline)[ol] 
                g, nls = self.model.removeline(g , ol )
                break
            
        return f,a,m,F,A,M,EN,LE,ii,g, nls
            
            
    def step(self, f, a, m, yb):
        
        f = f - self.dt2 * self.model.dH_dangle(a,m,yb )
        
        a = a + self.dt * self.model.dH_dfreq( f )
        
        f = f - self.dt2 * self.model.dH_dangle(a,m,yb )
        
        m = m - self.dt * self.model.eps * self.model.dH_dmag(a,m,yb) + self.sqdt * self.model.rand_mag()
        
        a = a - self.dt * self.model.eps * self.model.dH_dangle(a,m,yb) + self.sqdt * self.model.rand_angle()
        
        return f,a,m
        
        