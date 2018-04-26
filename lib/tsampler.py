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
        
    def addevent(self, EV_, enum_,t,ls,g, lo):
        
        EV = EV_
        if (self.output.SaveEvents):    
            EV[0,enum_] = t
            EV[1,enum_] = ls
            EV[2,enum_] = lo
            EV[3:,enum_] = g
        
        return EV, enum_ + 1
    
            
    def run(self):
        
        MyTasks = np.arange( self.comm.Get_rank(), self.walkers, self.comm.Get_size() )
        
        for task in MyTasks:
        
            ii = 0
        
            gamma = self.model.getic_gamma()
            
            f = self.model.getic_freq()
            a = self.model.getic_angle()
            m = self.model.getic_mag()
            
            FF = np.array([])
            AA = np.array([]) 
            MM = np.array([]) 
            EV = np.array([]) 
            EN = np.array([]) 
            LE = np.array([]) 
            G = np.array([]) 
            LS = np.array([]) 
            T = np.array([])
            ENERGY_SERVED = np.array([])
            
            if (self.output.SaveTraj):    FF = np.zeros( ( len(f), self.steps ) )
            if (self.output.SaveTraj):    AA = np.zeros( ( len(f), self.steps ) )
            if (self.output.SaveTraj):    MM = np.zeros( ( len(f), self.steps ) )
            if (self.output.SaveEvents):    EV = np.zeros( (  len(gamma)+3 , self.model.nline ) )
            if (self.output.SaveEnergy):    EN = np.zeros( (  self.steps ) )
            if (self.output.SaveLineEnergy):    LE = np.zeros( ( self.model.nline, self.steps ) )
            if (self.output.SaveGamma):    G = np.tile(gamma, (self.steps,1) ).T
            if (self.output.SaveLoad):    LS = np.ones( (  self.steps ) )
            ENERGY_SERVED = np.zeros(self.steps)
                        
            # Do burn in
            # (Q, Adrian): I don't understand the burn-in.
            # Confirm

            nburn = int(self.model.burntime / self.ig.dt)
            _,_,anodes = self.model.removeline( np.ones( self.model.nline ) , 0 )


            # flag so i Know we're in burn-time

            if nburn > 0:
                self.ig.burn = True
                f,a,m,_,_,_,_,_,_,_,_,_,_,_  = self.ig.adv(f, a, m, nburn, np.ones(self.model.nline), anodes)
            self.ig.burn = False

            time = 0 
            _,ls,anodes = self.model.removeline( gamma , 0 )
            enum = 0
            EV,enum = self.addevent(EV,enum,time,ls,gamma,0)
            
            
            
            while self.ig.keepgoing( time , gamma, ls  ) :
                 
                # step whole system
                f,a,m,F,A,M,E,L,jj,g,nls, anodes,lo, en_serv_hist  = self.ig.adv(f, a, m, self.steps - ii, gamma, anodes)
                
                if (self.output.SaveTraj):    FF[:,ii:] = np.copy(F)
                if (self.output.SaveTraj):    AA[:,ii:] = np.copy(A)
                if (self.output.SaveTraj):    MM[:,ii:] = np.copy(M)
                if (self.output.SaveEnergy):    EN[ii:] = np.copy(E)
                if (self.output.SaveLineEnergy):    LE[:,ii:] = np.copy(L)
                ENERGY_SERVED[ii:] = np.copy(en_serv_hist)

                ii += jj
                time += jj * self.ig.dt
                
                if nls>=0:
                    if (self.output.SaveLoad):    LS[ii:] = nls 
                    ls = nls
                    EV,enum = self.addevent(EV,enum,time,ls,g,lo)
                
                gamma = g
                if (self.output.SaveGamma):    G[:,ii:] = np.tile(gamma, (self.steps-ii,1) ).T

            if (self.output.SaveTime):    T = np.arange(1, ii+1) * self.ig.dt
            if (self.output.SaveTraj):    FF = FF[:,:ii]
            if (self.output.SaveTraj):    AA = AA[:,:ii]
            if (self.output.SaveTraj):    MM = MM[:,:ii]
            if (self.output.SaveEnergy):    EN = EN[:ii]
            if (self.output.SaveLineEnergy):    LE = LE[:,:ii]
            if (self.output.SaveGamma):    G = G[:,:ii]
            if (self.output.SaveEvents):    EV=EV[:,:enum]
            ENERGY_SERVED = ENERGY_SERVED[:ii]
            
            self.output.AddOutput( task, FF, AA, MM,EN,LE, G, T, LS, EV, ENERGY_SERVED)
            

