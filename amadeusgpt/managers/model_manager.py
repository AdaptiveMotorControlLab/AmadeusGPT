from typing import List
from .base import Manager

class ModelManager(Manager):
    def __init__(self, config):
        self.config = config
        self.sam_config = config['sam_info']
        self.sa_config = config['dlc_info']
        self.init_SA_model()
        self.init_SAM_model()
    
    def init_SA_model(self):
        pass
    def init_SAM_model(self):
        pass
    

                
    


