 
import numpy as np
 
import time
import datetime
import os

class Output:

    def __init__(self, ini, comm ):

        self.OutputPath = ini.get("output","dir")

        self.AppendTime = ini.getboolean("output","appendtime",False)

        self.SaveTraj = ini.getboolean("output","savetraj",True) 
        self.SaveEnergy = ini.getboolean("output","saveenergy",True) 
        self.SaveLineEnergy = ini.getboolean("output","savelineenergy",True) 
        self.SaveGamma = ini.getboolean("output","savegamma",True) 
        self.SaveTime = ini.getboolean("output","savetime",True) 

        self.PrintFreq = ini.getint("output","printfreq",0)
        self.PrintTime = ini.getint("output","printtime",0)
        
        self.ostep = ini.getint("output","savefreq",1)
        
        self.comm = comm


        
    
    def config(self):
 
        if self.AppendTime:
            self.OutputPath += time.strftime("_%y%m%d_%H%M%S")
        
        
        self.OutputPath = self.comm.bcast( self.OutputPath , root=0 )
        
        rank = self.comm.Get_rank()
        if (rank==0):
            print("## Output to %s"%self.OutputPath)
            try:
                os.makedirs( self.OutputPath )
            except OSError:
                if not os.path.isdir( self.OutputPath):
                    raise

        self.IDList = list()

        self.FList = list()
        self.AList = list()
        self.MList = list()
        
        self.ENList = list()
        self.LEList = list()
        self.GList = list()
        self.TList = list()
        
        self.lastprinttime = 0
        self.lastprintsteps = 0
                    
        
        
    
    def save(self):

        #tt = np.array([ time.time() - self.starttime])

        for ii, id in enumerate( self.IDList):

            if (self.SaveTraj):
                path = self.OutputPath + "/f." + str(id)
                np.savetxt(path, self.FList[ii] , fmt="%.4e") 
                path = self.OutputPath + "/a." + str(id)
                np.savetxt(path, self.AList[ii], fmt="%.4e" ) 
                path = self.OutputPath + "/m." + str(id)
                np.savetxt(path, self.MList[ii] , fmt="%.4e") 

            if (self.SaveEnergy):
                path = self.OutputPath + "/e." + str(id)
                np.savetxt(path, self.ENList[ii] , fmt="%.5e") 

            if (self.SaveLineEnergy):
                path = self.OutputPath + "/le." + str(id)
                np.savetxt(path, self.LEList[ii] , fmt="%.4e") 

            if (self.SaveGamma):
                path = self.OutputPath + "/g." + str(id)
                np.savetxt(path, self.GList[ii] , fmt="%d") 

            if (self.SaveTime):
                path = self.OutputPath + "/t." + str(id)
                np.savetxt(path, self.TList[ii] , fmt="%.6e") 
                
                
            
        

    def AddOutput(self, id , F, A , M, EN, LE, G, T ):

        self.IDList.append( id )

        if (self.SaveTraj):
            self.FList.append( np.copy( F[:,::self.ostep] ) )
            self.AList.append( np.copy( A[:,::self.ostep] ) )
            self.MList.append( np.copy( M[:,::self.ostep] ) )

        if (self.SaveEnergy):
            self.ENList.append( np.copy( EN[::self.ostep] ) ) 

        if (self.SaveLineEnergy):
            self.LEList.append( np.copy( LE[:,::self.ostep] ) ) 

        if (self.SaveGamma):
            self.GList.append( np.copy( G[:,::self.ostep] ) ) 

        if (self.SaveTime):
            self.TList.append( np.copy( T[::self.ostep] ) ) 
            
             
        
        
    
