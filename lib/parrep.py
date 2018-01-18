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
        
    def add_event(self,G,LS,T,enum,g,nls, time,EV_, lo):
        
        GG = G
        LSLS = LS
        TT = T
        
        EV = EV_
        if (self.output.SaveEvents):    
            EV[0,enum] = time
            EV[1,enum] = nls
            EV[2,enum] = lo
            EV[3:,enum] = g
        
        
        if (self.output.SaveTime):    TT[enum] = time
        if (self.output.SaveGamma):    GG[:,enum] = g
        if (self.output.SaveLoad):    LSLS[enum] = nls
        
        enum = enum + 1
         
        
        return GG,LSLS,TT,enum,EV 
    
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
            
            
            FF = np.array([])
            AA = np.array([]) 
            MM = np.array([]) 
            EV = np.array([]) 
            EN = np.array([]) 
            LE = np.array([]) 
            G = np.array([]) 
            LS = np.array([]) 
            T = np.array([])
            
            if (self.output.SaveGamma):    G = np.ones( (self.model.nline , self.maxevents) )
            if (self.output.SaveLoad):    LS = np.ones( self.maxevents )
            if (self.output.SaveTime):    T = np.zeros( self.maxevents )
            if (self.output.SaveEvents):    EV = np.zeros( (  len(gamma)+3 , self.model.nline ) )
            
            
            # Do burn in

            nburn = int(self.model.burntime / self.ig.dt)
            _,_,anodes = self.model.removeline( np.ones( self.model.nline ) , 0 )
            if nburn>0:
                f,a,m,_,_,_,_,_,_,_,_,_,_  = self.ig.adv( f, a , m , nburn , np.ones( self.model.nline ) , anodes )



            enum = 0
            
            time = 0.0
            _,ls,anodes = self.model.removeline( gamma , 0 )
            
            #anodes = np.ones( self.model.nbus )>0
            
            if (myid==0):
                print ""
                print " -- Beginning run %d." % (repnum+1)
                print ""
                G,LS,T,enum,EV = self.add_event(G,LS,T,enum, gamma, ls, time,EV,0 )
                
            
            while (  self.ig.keepgoing( time , gamma, ls  )  ):
                
                ####
                # DECOR STAGE
                ####
                
                if (myid==0):
                    jj = 0
                    print "DECOR step, time: %f  ls:%f    events: %d." % (time, ls, enum)
                    
                    while (jj < self.s_decor) and ( self.ig.keepgoing( time , gamma, ls  ) ):
                        #f,a,m,F,A,M,E,L,jj,g,nls,lineout = self.ig.adv( f, a , m , self.s_decor , gamma )
                        #f,a,m,jj,gamma,nls,lineout = self.ig.adv( f, a , m , self.s_decor , gamma )[0,1,2,8,9,10]
                        f,a,m,_,_,_,_,_,jj,g,nls,anodes,lineout = self.ig.adv( f, a , m , self.s_decor , gamma,anodes ) 
                        
                        time += jj * self.ig.dt
                        gg = gamma
                        gamma = g 
                        
                        
                        if (self.diff_g(g,gg) ):
                            G,LS,T,enum,EV = self.add_event(G,LS,T,enum, gamma, nls, time,EV,lineout )
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
                     

                for ii in range(ntasks):
                    FF = F[:,ii]
                    AA = A[:,ii]
                    MM = M[:,ii] 

                    while (True):

                        f = np.copy(FF)
                        a = np.copy(AA)
                        m = np.copy(MM)

                        f,a,m,_,_,_,_,_,_,g,_,_,lineout = self.ig.adv( f, a , m , self.s_dephase , gamma,anodes )

                        if (self.diff_g(g,gamma)):
                            FF = np.random.normal(scale=np.std(FF.flatten()), size=np.shape(FF) )
                            continue

                        break
                    
                    F[:,ii] = f
                    A[:,ii] = a
                    M[:,ii] = m

                    
                #for tt in np.arange( int( self.s_dephase*0.1) ):
                    
                #    for ii in np.arange( ntasks ) :
                #        f = F[:,ii]
                #        a = A[:,ii]
                #        m = M[:,ii]  
                #        
                #        f,a,m,_,_,_,_,_,_,g,_,_,lineout = self.ig.adv( f, a , m , 10 , gamma,anodes ) 
                #                    
                #        if (self.diff_g(g,gamma)): 
                #            ri =  self.model.Random.randint( ii+1 ) 
                #            f = F[:,ri]
                #            a = A[:,ri]
                #            m = M[:,ri] 
                #                        
                #        F[:,ii] = f
                #        A[:,ii] = a
                #        M[:,ii] = m
                     
                
                ####
                # PARALLEL STAGE
                ####
                
                self.comm.Barrier()
                if (myid==0): 
                    print "PARALLEL step, time: %f  ls:%f    events: %d." % (time, ls, enum)
                    
                event = -1
                etime = self.pstep +1
                gchk  = -1
                jj = 0

                while (self.ig.keepgoing( time , gamma, ls  ) ):
                    
                    for ii in np.arange( ntasks ):
                        
                        F[:,ii], A[:,ii] , M[:,ii],_,_,_,_,_,jj,g,nls, ann,lineout = self.ig.adv( F[:,ii], A[:,ii] , M[:,ii] , self.pstep , gamma, anodes ) 
                    
                        if (self.diff_g(g,gamma)):
                            event = myid
                            etime = jj
                            break
                    
                    gchk = self.comm.allreduce( event, op=MPI.MAX )
                    
                    if (gchk<0 ):
                        time += self.ig.dt * self.pstep * self.walkers
                    else:
                        gtime = self.comm.allreduce( etime, op=MPI.MINLOC )
                        #print [jj, MPI.COMM_WORLD.Get_rank(), ntasks]
                        etime = gtime[0]
                        gchk = gtime[1]
                        jj = etime  #self.comm.bcast( jj , root= gchk )
                        gamma = self.comm.bcast( g , root= gchk )
                        ls = self.comm.bcast( nls , root= gchk )
                        f = self.comm.bcast( F[:,ii] , root= gchk )
                        a = self.comm.bcast( A[:,ii] , root= gchk )
                        m = self.comm.bcast( M[:,ii] , root= gchk )
                        anodes = self.comm.bcast( ann , root= gchk )
                        lineout = self.comm.bcast( lineout , root= gchk )
                        time +=  etime * self.ig.dt * self.walkers + gchk * self.ig.dt
                        #time += self.ig.dt * self.pstep * ( float(gchk)  / self.walkers ) + jj * self.ig.dt
                        break
                    
                
                        
                if (gchk>=0) and (myid==0):
                    G,LS,T,enum,EV = self.add_event(G,LS,T,enum, gamma, ls, time,EV ,lineout )
                    print " - event, time: %f   ls: %f     events: %d." % (time, ls , enum)
            
            
            if (myid==0):    
                if (self.output.SaveTime):    T=T[:enum]
                if (self.output.SaveLoad):    LS=LS[:enum]
                if (self.output.SaveGamma):    G=G[:,:enum]
                if (self.output.SaveEvents):    EV=EV[:,:enum]
                self.output.AddOutput( repnum, FF, AA, MM,EN,LE, G, T, LS,EV )
            
    def diff_g( self, x , y ):
        
        k = np.array(x)-np.array(y)
        k = np.sum( k*k )
        
        if (k>0):
            return True
        else:
            return False
    
         
        