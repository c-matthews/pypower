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
          
        self.stoptime = self.ig.stoptime
        self.stoplines = self.ig.stoplines
        self.stopload = self.ig.stopload
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
    
    def sync_state(self, f,a,m,gamma, time, ls, anodes):
        
        
        f = self.comm.bcast( f , root=0 )
        a = self.comm.bcast( a , root=0 )
        m = self.comm.bcast( m , root=0 )
        gamma = self.comm.bcast( gamma , root=0 )
        time = self.comm.bcast( time , root=0 )
        ls = self.comm.bcast( ls , root=0 )
        anodes = self.comm.bcast( anodes , root=0 )
        
        return f,a,m,gamma,time,ls,anodes
        
        
        
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
            ls = 1.0
            
            anodes = np.ones( self.model.nbus )>0
            
            if (myid==0):
                print ""
                print " -- Beginning run %d." % (repnum+1)
                print ""
            
            while (  self.ig.keepgoing( time , gamma, ls  )  ):
                
                ####
                # DECOR STAGE
                ####
                
                if (myid==0):
                    jj = 0
                    print "DECOR step, time: %f  ls:%f    events: %d." % (time, ls, enum)
                    
                    while (jj < self.s_decor) and ( self.ig.keepgoing( time , gamma, ls  ) ):
                        #f,a,m,F,A,M,E,L,jj,g,nls = self.ig.adv( f, a , m , self.s_decor , gamma )
                        #f,a,m,jj,gamma,nls = self.ig.adv( f, a , m , self.s_decor , gamma )[0,1,2,8,9,10]
                        f,a,m,_,_,_,_,_,jj,g,nls,anodes = self.ig.adv( f, a , m , self.s_decor , gamma,anodes ) 
                    
                        time += jj * self.ig.dt
                        gg = gamma
                        gamma = g 
                        
                        
                        if (self.diff_g(g,gg) ):
                            G,LS,T,enum = self.add_event(G,LS,T,enum, gamma, nls, time )
                            jj=0
                            ls = nls
                            print " - event, time: %f   ls: %f     events: %d." % (time, ls , enum)
                            
                
                f,a,m,gamma,time,ls,anodes = self.sync_state( f,a,m,gamma,time, ls,anodes )
                
                if (self.ig.keepgoing( time , gamma, ls  ) == False ):
                    break
                
                ####
                # DEPHASE STAGE
                #### 
                
                MyTasks = np.arange(myid, self.walkers, self.comm.Get_size())
                ntasks = len(MyTasks)
                F = np.tile(f, (ntasks,1) ).T
                A = np.tile(a, (ntasks,1) ).T
                M = np.tile(m, (ntasks,1) ).T
                if (myid==0):
                    print "DEPHASE step, time: %f  ls:%f    events: %d." % (time, ls, enum)
                    
                for ii in np.arange( ntasks ) :
                    f = F[:,ii]
                    a = A[:,ii]
                    m = M[:,ii] 
                    tt = 0
                    
                    while (tt==0):
                        f,a,m,_,_,_,_,_,tt,g,_,_ = self.ig.adv( f, a , m , self.s_dephase , gamma,anodes ) 
                                
                        if (self.diff_g(g,gamma)): 
                            ri =  self.model.Random.randint( ii+1 ) 
                            f = F[:,ri]
                            a = A[:,ri]
                            m = M[:,ri]
                            tt = 0
                                    
                    F[:,ii] = f
                    A[:,ii] = a
                    M[:,ii] = m
                     
                
                ####
                # PARALLEL STAGE
                ####
                if (myid==0): 
                    print "PARALLEL step, time: %f  ls:%f    events: %d." % (time, ls, enum)
                    
                event = -1
                gchk  = -1

                while (self.ig.keepgoing( time , gamma, ls  ) ):
                    
                    for ii in np.arange( ntasks ):
                        
                        F[:,ii], A[:,ii] , M[:,ii],_,_,_,_,_,jj,g,nls, ann = self.ig.adv( F[:,ii], A[:,ii] , M[:,ii] , self.pstep , gamma, anodes ) 
                    
                        if (self.diff_g(g,gamma)):
                            event = myid
                            break
                    
                    gchk = self.comm.allreduce( event, op=MPI.MAX )
                    
                    if (gchk<0 ):
                        time += self.ig.dt * self.pstep * self.walkers
                    else:
                        jj = self.comm.bcast( jj , root= gchk )
                        gamma = self.comm.bcast( g , root= gchk )
                        ls = self.comm.bcast( nls , root= gchk )
                        f = self.comm.bcast( F[:,ii] , root= gchk )
                        a = self.comm.bcast( A[:,ii] , root= gchk )
                        m = self.comm.bcast( M[:,ii] , root= gchk )
                        anodes = self.comm.bcast( ann , root= gchk )
                        time += self.ig.dt * self.pstep * ( float(gchk)  / self.comm.Get_size() ) + jj * self.ig.dt
                        break
                    
                
                        
                if (gchk>=0) and (myid==0):
                    G,LS,T,enum = self.add_event(G,LS,T,enum, gamma, ls, time )
                    print " - event, time: %f   ls: %f     events: %d." % (time, ls , enum)
            
            
            if (myid==0):    
                T=T[:enum]
                LS=LS[:enum]
                G=G[:,:enum]
                self.output.AddOutput( repnum , [], [], [],[],[], G, T, LS )
            
    def diff_g( self, x , y ):
        
        k = np.array(x)-np.array(y)
        k = np.sum( k*k )
        
        if (k>0):
            return True
        else:
            return False
    
        