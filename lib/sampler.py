import numpy as np
import cmath  
from tsampler import TrajSampler

class Sampler:
    
    def __init__( self,ini, comm, model, integrator, output ):
        
        self.ini = ini
        self.comm = comm
        self.model = model
        self.ig = integrator
        self.output = output
          
        self.style = ini.get("sampler","style")  
        self.Rs = ini.getint("sampler","repeats",1) 
        
        
    def config(self):
        
        if self.style=="parrep":
            pass
        else:
            self.sampler = TrajSampler( self.ini, self.comm, self.model, self.ig, self.output )
            
        self.sampler.config()
        
        
    def run(self):
        
        for ii in np.arange(self.Rs):
             
            self.sampler.run()