import numpy as np
import cmath  

class Integrator:
    
    def __init__( self,ini, comm, model ):
        
        self.comm = comm
        self.model = model
          
        self.dt = ini.getfloat("integrator","timestep") 
          
        self.stoptime = ini.getfloat("integrator","stop_time", 10000.0)  
        self.stoplines = ini.getint("integrator","stop_lines", self.model.nline)  
        self.stopload = ini.getfloat("integrator","stop_load", 0.0)  
        
    def config(self): 
          
        self.dt2 = self.dt * 0.5
        self.sqdt = np.sqrt(self.dt)
        
    def adv(self, freq, angle, mag, N , gamma, anodes ):
        
        ybus = self.model.assemble_ybus(gamma) 
        f = freq
        a = angle
        m = mag
        g = gamma
        ann = anodes
        
        F = np.zeros( (len(f), N) )
        A = np.zeros( (len(a), N) )
        M = np.zeros( (len(m), N) )
        
        EN = np.zeros( N )
        LE = np.zeros( (self.model.nline, N) )
        
        ii = 0
        nls = -1 
        
        while ii<N :
            
            f,a,m = self.step(f,a,m, ybus, ann)
            
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
                g, nls, ann = self.model.removeline(g , ol )
                break
            
        return f,a,m,F,A,M,EN,LE,ii,g, nls, ann
            
            
    def step(self, f, a, m, yb, anodes):
        
        f[anodes] = f[anodes] - self.dt2 * self.model.dH_dangle(a,m,yb )[anodes]
        
        a[anodes] = a[anodes] + self.dt * self.model.dH_dfreq( f )[anodes]
        
        f[anodes] = f[anodes] - self.dt2 * self.model.dH_dangle(a,m,yb )[anodes]
        
        m[anodes] = m[anodes] - self.dt * self.model.eps * self.model.dH_dmag(a,m,yb)[anodes] + self.sqdt * self.model.rand_mag()[anodes]
        
        a[anodes] = a[anodes] - self.dt * self.model.eps * self.model.dH_dangle(a,m,yb)[anodes] + self.sqdt * self.model.rand_angle()[anodes]
        
        return f,a,m
        
    def keepgoing(self, time, g , ls ):
        
        if ( time >= self.stoptime ):
            return False
        
        if ( self.model.nline - np.sum( np.array(g) ) >= self.stoplines ):
            return False
        
        if (ls <= self.stopload ):
            return False
        
        return True
        
        