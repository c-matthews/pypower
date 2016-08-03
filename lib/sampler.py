import numpy as np
import cmath  
from tsampler import TrajSampler
from parrep import ParRep

class Sampler:
    
    def __init__( self,ini, comm, model, integrator, output ):
        
        self.ini = ini
        self.comm = comm
        self.model = model
        self.ig = integrator
        self.output = output
          
        self.style = ini.get("sampler","style")   
        
        
    def config(self):
        
        if self.style=="parrep":
            self.sampler = ParRep( self.ini, self.comm, self.model, self.ig, self.output )
        else:
            self.sampler = TrajSampler( self.ini, self.comm, self.model, self.ig, self.output )
            
        self.sampler.config()
        
        
    def run(self): 
            
            self.sampler.run()