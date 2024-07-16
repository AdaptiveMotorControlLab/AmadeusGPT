from .base import Manager

class ModelManager(Manager):
    """
    Not implemented yet. Perhaps we don't need a model manager.
    """
    def __init__(self, config):
        self.config = config     

    def init_SA_model(self):
        pass

    def init_SAM_model(self):
        pass
