import numpy as np
import cmath  

class Integrator:
    
    def __init__( self,ini, comm, model, output ):
        
        self.comm = comm
        self.model = model
        self.output = output
          
        self.dt = ini.getfloat("integrator","timestep") 
        self.stepstyle = ini.get("integrator","style","default") 
          
        self.stoptime = ini.getfloat("integrator","stop_time", 10000.0)  
        self.stoplines = ini.getint("integrator","stop_lines", self.model.nline)  
        self.stopload = ini.getfloat("integrator","stop_load", 0.0)  
        
    def config(self): 
          
        self.dt2 = self.dt * 0.5
        self.dt4 = self.dt2 * 0.5
        self.sqdt = np.sqrt(self.dt)
        self.sqdt2 = np.sqrt(self.dt2)
        self.sqdt4 = np.sqrt(self.dt4) 

        self.step = self.step_lm
        if (self.stepstyle=="euler"):
            self.step = self.step_euler

        if (self.stepstyle=="heun"):
            self.step = self.step_heun

        if (self.stepstyle=="lm"):
            self.step = self.step_lm 

        if (self.stepstyle=="baoab"):
            self.step = self.step_baoab 
        
    def adv(self, freq, angle, mag, N , gamma, anodes ):
        """ Advance system

            Args:
            + freq: system frequency
            + angle: voltage angle
            + mag: voltage magnitude
            + N: what is this?
            + gamma: 

            Return

        """
        ybus = self.model.assemble_ybus(gamma)  

        f = freq
        a = angle
        m = mag
        g = gamma
        ann = anodes

        lineout = 0
            
        F = np.array([])
        A = np.array([]) 
        M = np.array([]) 
        EN = np.array([]) 
        LE = np.array([]) 
            
        if (self.output.SaveTraj):    F = np.zeros( (len(f), N) )
        if (self.output.SaveTraj):    A = np.zeros( (len(a), N) )
        if (self.output.SaveTraj):    M = np.zeros( (len(m), N) )

        # always save served energy
        en_serv_hist = np.zeros(N)
        
        if (self.output.SaveEnergy):    EN = np.zeros( N )
        if (self.output.SaveLineEnergy):    LE = np.zeros( (self.model.nline, N) )
        
        ii = 0
        io = 0
        nls = -1 
        xtra = None
        en, le, df, da, dm, energy_served = self.model.energy(f, a, m, ybus, g) 

        while ii < N :
            
            f, a, m, xtra = self.step(f, a, m, df, da, dm, xtra, ybus, ann)
            en, le, df, da, dm, energy_served = self.model.energy(f, a, m, ybus, g)
            io, ol = self.model.checklines(le , g)
            
            if (self.output.SaveTraj):    F[:,ii] = f
            if (self.output.SaveTraj):    A[:,ii] = a
            if (self.output.SaveTraj):    M[:,ii] = m
            if (self.output.SaveEnergy):    EN[ii] = en
            if (self.output.SaveLineEnergy):    LE[:,ii] = le# * self.model.oobb
            en_serv_hist[ii] = energy_served

            ii += 1
            self.model.time += self.dt
            
            if io: 
                lineout = 1 + np.arange(self.model.nline)[ol]
                
                if len(lineout) > 1:
                    lineout = lineout[0]
                    
                g, nls, ann = self.model.removeline(g , ol)
                break
            
        return f,a,m,F,A,M,EN,LE,ii,g, nls, ann, lineout, en_serv_hist
 

    def step_heun(self, f, a, m, df,da,dm,xtra, yb, anodes):

        ff=f
        aa=a
        mm=m

        #dhda,dhdm = self.model.dH_danglemag(aa,mm,yb )
        #dhdf = self.model.dH_dfreq( ff ) 

        ka = np.copy(aa)
        km = np.copy(mm)
        kf = np.copy(ff)
        ra = self.model.rand_angle()  
        rm = self.model.rand_mag() 

        kf[anodes] = ff[anodes] - self.dt * da[anodes]
        ka[anodes] = aa[anodes] - (self.dt * self.model.eps) * da[anodes] + self.sqdt * ra[anodes]
        ka[anodes] = ka[anodes] + self.dt * df[anodes]
        km[anodes] = mm[anodes] - (self.dt * self.model.eps) * dm[anodes] + self.sqdt * rm[anodes]
        


        dhda2,dhdm2 = self.model.dH_danglemag(ka,km,yb )
        dhdf2 = self.model.dH_dfreq( kf ) 

        dhda = (da + dhda2)*0.5
        dhdm = (dm + dhdm2)*0.5
        dhdf = (df + dhdf2)*0.5

        ff[anodes] = ff[anodes] - self.dt * dhda[anodes]
        aa[anodes] = aa[anodes] - (self.dt * self.model.eps) * dhda[anodes] + self.sqdt * ra[anodes]
        aa[anodes] = aa[anodes] + self.dt * dhdf[anodes]
        mm[anodes] = mm[anodes] - (self.dt * self.model.eps) * dhdm[anodes] + self.sqdt * rm[anodes]
        
        return ff,aa,mm,None

    def step_lm(self, f, a, m, df,da,dm,xtra, yb, anodes):

        ff=f
        aa=a
        mm=m

        if (xtra==None):
            ra = self.model.rand_angle()
            rm = self.model.rand_mag()
        else:
            ra = xtra[0]
            rm = xtra[1]


        #dhda,dhdm = self.model.dH_danglemag(aa,mm,yb )
        #dhdf = self.model.dH_dfreq( ff )  

        Ra = self.model.rand_angle()
        Rm = self.model.rand_mag()

        ff[anodes] = ff[anodes] - self.dt * da[anodes]
        aa[anodes] = aa[anodes]  - (self.dt * self.model.eps) * da[anodes] + (0.5*self.sqdt) * ((Ra[anodes]+ ra[anodes]))
        aa[anodes] = aa[anodes] + self.dt * df[anodes]
        mm[anodes] = mm[anodes] - (self.dt * self.model.eps) * dm[anodes] + (0.5*self.sqdt) * ((Rm[anodes]+rm[anodes]) )
               

        return ff,aa,mm,[Ra,Rm]

    def step_euler(self, f, a, m, df,da,dm, xtra, yb, anodes):

        ff=f
        aa=a
        mm=m

        #dhda,dhdm = self.model.dH_danglemag(aa,mm,yb )
        #dhdf = self.model.dH_dfreq( ff ) 

        #sqdt = square root time step

        ff[anodes] = ff[anodes] - self.dt * da[anodes]
        aa[anodes] = aa[anodes] - (self.dt * self.model.eps) * da[anodes] + self.sqdt * self.model.rand_angle()[anodes]
        aa[anodes] = aa[anodes] + self.dt * df[anodes]
        mm[anodes] = mm[anodes] - (self.dt * self.model.eps) * dm[anodes] + self.sqdt * self.model.rand_mag()[anodes]
         

        return ff,aa,mm,None

    def step_baoab(self, f, a, m, df,da,dm, xtra, yb, anodes):

        ff=f
        aa=a
        mm=m
 

        ff[anodes] = ff[anodes] - self.dt * da[anodes]
        aa[anodes] = aa[anodes] + self.dt2 * self.model.dH_dfreq( ff )[anodes]

        dhda,dhdm = self.model.dH_danglemag(aa,mm,yb )
        ra  = self.model.rand_angle()
        rm  = self.model.rand_mag()
        ka = np.copy( aa )
        km = np.copy( mm )

        ka[anodes] = aa[anodes] - (self.dt * self.model.eps) * dhda[anodes] + self.sqdt * ra[anodes]
        km[anodes] = mm[anodes] - (self.dt * self.model.eps) * dhdm[anodes] + self.sqdt * rm[anodes]
         
        dhda2,dhdm2 = self.model.dH_danglemag(ka,km,yb )

        aa[anodes] = aa[anodes] - (self.dt2 * self.model.eps) * (dhda2[anodes] + dhda[anodes]) + self.sqdt * ra[anodes]
        mm[anodes] = mm[anodes] - (self.dt2 * self.model.eps) * (dhdm2[anodes] + dhdm[anodes]) + self.sqdt * rm[anodes]
         
        aa[anodes] = aa[anodes] + self.dt2 * self.model.dH_dfreq( ff )[anodes]






        return ff,aa,mm,None
        
    def keepgoing(self, time, g , ls ):

        """ 
            (Q, Adrian) This function seems to halt the integration if:

            a) run out of time
            b) To few lines.
            c) Flow though line surpasses limit?

            Confirm
        """
        
        if ( time >= self.stoptime ):
            return False
        
        if ( self.model.nline - np.sum( np.array(g) ) >= self.stoplines ):
            return False
        
        if (ls <= self.stopload ):
            return False
        
        return True
        
        
