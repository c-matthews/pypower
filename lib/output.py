import numpy as np
 
import time
import datetime
import os

class Output:

    def __init__(self, ini, comm, model ):

        self.OutputPath = ini.get("output","dir")

        self.AppendTime = ini.getboolean("output","appendtime",False)

        self.SaveTraj = ini.getboolean("output","savetraj",True) 
        self.SaveEnergy = ini.getboolean("output","saveenergy",True) 
        self.SaveLineEnergy = ini.getboolean("output","savelineenergy",True) 
        self.SaveGamma = ini.getboolean("output","savegamma",True) 
        self.SaveTime = ini.getboolean("output","savetime",True) 
        self.SaveLoad = ini.getboolean("output","saveload",True) 
        self.SaveEvents = ini.getboolean("output","saveevents",True) 
        self.SaveGraph = ini.getboolean("output","savegraph",False) 


        self.PrintFreq = ini.getint("output","printfreq",0)
        self.PrintTime = ini.getint("output","printtime",0)
        
        self.ostep = ini.getint("output","savefreq",1)
        
        self.comm = comm
        self.model = model


        
    
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
        self.LSList = list()
        self.EVList = list()
        self.GraphList = list()
        
        self.lastprinttime = 0
        self.lastprintsteps = 0
         
                    
        
    def savetime(self, t ):
        
        path = self.OutputPath + "/time.txt"
        np.savetxt(path, np.array([t]) )
    
    def save(self):

        #tt = np.array([ time.time() - self.starttime])

        for ii, id in enumerate( self.IDList):

            if (self.SaveTraj) and (len(self.FList)>0):
                path = self.OutputPath + "/f." +  str(id)
                np.savetxt(path, self.FList[ii] , fmt="%.4e") 
                path = self.OutputPath + "/a." +  str(id)
                np.savetxt(path, self.AList[ii], fmt="%.4e" ) 
                path = self.OutputPath + "/m." +  str(id)
                np.savetxt(path, self.MList[ii] , fmt="%.4e") 

            if (self.SaveEnergy) and (len(self.ENList)>0):
                path = self.OutputPath + "/e." +  str(id)
                np.savetxt(path, self.ENList[ii] , fmt="%.5e") 

            if (self.SaveLineEnergy) and (len(self.LEList)>0):
                path = self.OutputPath + "/le." +  str(id)
                xx=self.LEList[ii]
                np.savetxt(path, xx , fmt="%.4e") 

            if (self.SaveGamma) and (len(self.GList)>0):
                path = self.OutputPath + "/g." +  str(id)
                np.savetxt(path, self.GList[ii] , fmt="%d") 

            if (self.SaveTime) and (len(self.TList)>0):
                path = self.OutputPath + "/t." +  str(id)
                np.savetxt(path, self.TList[ii] , fmt="%.6e") 

            if (self.SaveLoad) and (len(self.LSList)>0):
                path = self.OutputPath + "/ls." +  str(id)
                np.savetxt(path, self.LSList[ii] , fmt="%.4e") 

            if (self.SaveEvents) and (len(self.EVList)>0):
                path = self.OutputPath + "/ev." +  str(id)
                np.savetxt(path, self.EVList[ii] , fmt="%.5e") 
                

            if (self.SaveGraph) and (len(self.GList)>0):
                le = self.LEList[ii]
                g = self.GList[ii]
                for jj in range( len( g[0,:] ) ):
                    path = self.OutputPath + "/graph." +  str(id) + "." + str(jj)
                    self.WriteGraph( path , le[:,jj] , g[:,jj]  )
                
            
        

    def AddOutput(self, id , F, A , M, EN, LE, G, T, LS,EV ):

        self.IDList.append( id )

        if (self.SaveTraj) and (np.size(F)>0):
            self.FList.append( np.copy( F[:,::self.ostep] ) )
            self.AList.append( np.copy( A[:,::self.ostep] ) )
            self.MList.append( np.copy( M[:,::self.ostep] ) )

        if (self.SaveEnergy) and (np.size(EN)>0):
            self.ENList.append( np.copy( EN[::self.ostep] ) ) 

        if (self.SaveLineEnergy or self.SaveGraph) and (np.size(LE)>0):
            self.LEList.append( np.copy( LE[:,::self.ostep] ) ) 

        if (self.SaveGamma or self.SaveGraph) and (np.size(G)>0):
            self.GList.append( np.copy( G[:,::self.ostep] ) ) 

        if (self.SaveTime) and (np.size(T)>0):
            self.TList.append( np.copy( T[::self.ostep] ) ) 

        if (self.SaveLoad) and (np.size(LS)>0):
            self.LSList.append( np.copy( LS[::self.ostep] ) ) 

        if (self.SaveEvents) and (np.size(EV)>0):
            self.EVList.append( np.copy( EV[::self.ostep] ) )  
            
              
        

    def WriteGraph(self, filename, LE, gamma):
          
        tfile = open(filename, "w")
        
        tfile.write("graph G {\n")

        tfile.write('node [label="",shape="circle",width=.25,height=.25]\n')
        tfile.write('edge [style=bold]\n')
        tfile.write('overlap=scale \n')
        tfile.write('orientation=landscape\n')
 
        for ii in range(len( self.model.from_to[:,0]) ):

            ft = self.model.from_to[ii,:] 

            zz = (LE[ii] / self.model.thresh) * gamma[ii]
            cflt = 0.65*np.sqrt(1-zz)
            lflt = 0.4 + 0.6*(zz>0.5)
            if (cflt>0.65):
                cflt = 0.65
            if (cflt<0):
                cflt = 0
            if (lflt>1):
                lflt = 1
            if (lflt<0):
                lflt=0
 
            sstr = ""

            if (gamma[ii]==0):
                sstr = ",style=dotted"
            else:
                if (zz>0.5):
                    sstr = ",style=bold"
            
            s = str(int( ft[0] ) ) + " -- " + str(int(ft[1])) + ' [color="' + str(cflt) + ',' + str(lflt) +',1.0]"' + sstr + "];\n"

            tfile.write(s)
        
        tfile.write("}\n")
        
        tfile.close()

        
    



