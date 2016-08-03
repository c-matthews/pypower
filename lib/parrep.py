import numpy as np
import cmath   
from mpi4py import MPI

class ParRep:
    
    def __init__( self,ini, comm, model, integrator, output ):
        
        self.ini = ini
        self.comm = comm
        self.model = model
        self.ig = integrator
        self.output = output
          
        self.maxtime = ini.getfloat("parrep","maxtime")  
        self.walkers = ini.getint("parrep","walkers",1) 
        self.t_decor = ini.getfloat("parrep","t_decor")  
        self.t_dephase = ini.getfloat("parrep","t_dephase")  
        self.pstep = ini.getint("parrep","parasteps",1)  
        self.reps = ini.getint("parrep","repeats",1)  
        
        self.maxevents = model.nline
        
    def config(self):
        
        self.s_decor = 1 + int(self.t_decor / self.ig.dt)
        self.s_dephase = 1 + int(self.t_dephase / self.ig.dt)
        
    def add_event(self,G,LS,T,enum,g,nls, time):
        
        GG = G
        LSLS = LS
        TT = T
        
        enum = enum + 1
        
        TT[enum] = time
        GG[:,enum] = g
        LSLS[enum] = nls
        
        return GG,LSLS,TT,enum 
    
    def sync_state(self, f,a,m,gamma):
        
        
        f = self.comm.bcast( f , root=0 )
        a = self.comm.bcast( a , root=0 )
        m = self.comm.bcast( m , root=0 )
        gamma = self.comm.bcast( gamma , root=0 )
        
        return f,a,m,gamma
        
        
        
    def run(self):
        
        myid = self.comm.Get_rank()  
        
        for repnum in np.arange(self.reps):
        
            gamma = self.model.getic_gamma()
            
            f = self.model.getic_freq()
            a = self.model.getic_angle()
            m = self.model.getic_mag()
            
            G = np.ones( (self.model.nline , self.maxevents) )
            LS = np.ones( self.maxevents )
            T = np.zeros( self.maxevents )
            enum = 0
            
            time = 0.0
            
            while (time < self.maxtime):
                
                ####
                # DECOR STAGE
                ####
                
                if (myid==0):
                    jj = 0
                    
                    while (jj < self.s_decor):
                        #f,a,m,F,A,M,E,L,jj,g,nls = self.ig.adv( f, a , m , self.s_decor , gamma )
                        #f,a,m,jj,gamma,nls = self.ig.adv( f, a , m , self.s_decor , gamma )[0,1,2,8,9,10]
                        f,a,m,_,_,_,_,_,jj,g,nls = self.ig.adv( f, a , m , self.s_decor , gamma ) 
                    
                        time += jj * self.ig.dt
                        gamma = g
                        
                        if (jj<self.s_decor):
                            G,LS,T,enum = self.add_event(G,LS,T,enum, gamma, nls, time )
                            
                        if (time>= self.maxtime):
                            break
                
                ####
                # DEPHASE STAGE
                ####
                
                
                if (myid==0) and time<self.maxtime:
                    jj = 0  
                    F = np.tile(f, (self.walkers,1) ).T
                    A = np.tile(a, (self.walkers,1) ).T
                    M = np.tile(m, (self.walkers,1) ).T
                    
                    for ii in np.arange( self.walkers ):
                        f = F[:,ii]
                        a = A[:,ii]
                        m = M[:,ii]
                        
                        jj = 0
                        
                        while (jj < self.s_dephase): 
                            f,a,m,_,_,_,_,_,jj,_,_ = self.ig.adv( f, a , m , self.s_dephase , gamma ) 
                        
                            if (jj<self.s_dephase): 
                                ri =  self.model.Random.randint( ii+1 ) 
                                f = F[:,ri]
                                a = A[:,ri]
                                m = M[:,ri]
                                
                        F[:,ii] = f
                        A[:,ii] = a
                        M[:,ii] = m
                        
                
                ####
                # PARALLEL STAGE
                ####
                
                if (myid==0):
                    FF = [F[:, np.arange(jj, self.walkers, self.comm.Get_size())]  for jj in np.arange(self.comm.Get_size() )  ] 
                    AA = [A[:, np.arange(jj, self.walkers, self.comm.Get_size())]  for jj in np.arange(self.comm.Get_size() )  ] 
                    MM = [M[:, np.arange(jj, self.walkers, self.comm.Get_size())]  for jj in np.arange(self.comm.Get_size() )  ] 
                else:
                    FF= None
                    AA = None
                    MM = None
                    
                F = self.comm.scatter( FF  , root = 0 )
                A = self.comm.scatter( AA  , root = 0 )
                M = self.comm.scatter( MM  , root = 0 )
                time = self.comm.bcast( time , root=0 )
                gamma = self.comm.bcast( gamma , root=0 )

                print [myid , np.shape(F) ]

                ntasks = np.shape(F)[1]
                    
                event = -1
                gchk  = -1

                while (time < self.maxtime ):
                    
                    for ii in np.arange( ntasks ):
                        
                        F[:,ii], A[:,ii] , M[:,ii],_,_,_,_,_,jj,g,nls = self.ig.adv( F[:,ii], A[:,ii] , M[:,ii] , self.pstep , gamma ) 
                    
                        if (jj<self.pstep):
                            event = myid
                            break
                    
                    gchk = self.comm.allreduce( event, op=MPI.MAX )
                    
                    if (gchk<0 ):
                        time += self.ig.dt * self.pstep * self.walkers
                    else:
                        jj = self.comm.bcast( jj , root= gchk )
                        gamma = self.comm.bcast( g , root= gchk )
                        nls = self.comm.bcast( nls , root= gchk )
                        f = self.comm.bcast( F[:,ii] , root= gchk )
                        a = self.comm.bcast( A[:,ii] , root= gchk )
                        m = self.comm.bcast( M[:,ii] , root= gchk )
                        time += self.ig.dt * self.pstep * ( gchk  / self.comm.Get_size() ) + jj * self.ig.dt
                        break
                    
                
                        
                if (gchk>=0) and (myid==0):
                    G,LS,T,enum = self.add_event(G,LS,T,enum, gamma, nls, time )
            
            
            if (myid==0):    
                self.output.AddOutput( repnum , [], [], [],[],[], G, T, LS )
            
            