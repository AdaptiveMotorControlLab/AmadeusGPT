from amadeusgpt.managers import (Manager, AnimalManager,
ObjectManager,RelationshipManager, 
EventManager, VisualManager, ModelManager, GUIManager)
import random
random.seed(78)
from pydantic import BaseModel
from typing import Dict
import ast
from .programs.api_registry import DEFAULT_REGISTRY, CORE_API_REGISTRY
"""
write a class called FuncObj that inhertis from pydantic BaseModel that takes a function string,
use AST to parse the function string to input and output types, function name, args, kwargs, and function body
"""
class FuncObj(BaseModel):
    function_string: str
    input_type: Dict[str, str]
    output_type: Dict[str, str]
    function_name: str
    args: list
    kwargs: dict
    function_body: str


class AnimalBehaviorAnalysis:
    """
    This class holds methods and objects that are useful for analyzing animal behavior.
    It owns multiple manager classes that are responsible for different aspects of the analysis.   
    """
    
    def __init__(self, config, use_cache = False):
      
        self.model_manager = ModelManager(config)
        # animal manager needs keypoint_file_path and model_manager for pose
        self.animal_manager = AnimalManager(config, self.model_manager)
               
        # object manager needs sam_info, seriralized pickle objects
        self.object_manager =  ObjectManager(config, 
                                             self.model_manager,
                                             self.animal_manager)

                                             
        # relationship manager needs animal_manager and object_manager
        self.relationship_manager = RelationshipManager(config, 
                                                        self.animal_manager,
                                                        self.object_manager,
                                                        use_cache = use_cache)       
        # event manager needs refernce to object_manager, animal_manager, and relationship_manager
        self.event_manager = EventManager(config, 
                                          self.object_manager,
                                          self.animal_manager,
                                          self.relationship_manager,
                                          use_cache=use_cache)
        
        # some managers need references to others to do their job
        self.visual_manager = VisualManager(config, 
                                            self.animal_manager,
                                            self.object_manager)
        
        self.gui_manager = GUIManager(config, self.object_manager)


                                                       

        # check all attributes that are inheritance of manager classes
        # and attach them as methods to the main class               
        self._attach_manager_methods()
        
    def _attach_manager_methods(self):
        method_names = set()
        for attr in dir(self):
            if isinstance(getattr(self,attr), Manager):
                manager = getattr(self, attr)
                cls_name = manager.__class__.__name__
                for method_name in DEFAULT_REGISTRY[cls_name]:
                    if hasattr(manager, method_name):

                        method = getattr(manager, method_name)
                        setattr(self, method_name, method.__get__(manager, manager.__class__))
                        if method_name not in method_names:                            
                            method_names.add(method_name)
                        else:
                            raise ValueError(f"Method {method_name} already exists in the class")
            

    def summary(self, manager_name = None):
        for manager in [self.animal_manager, self.object_manager, self.relationship_manager, self.event_manager, self.visual_manager]:
            if manager_name is not None:
                if manager.__class__.__name__ != manager_name:
                    continue
            print(manager.__class__.__name__)
            manager.summary()


if __name__ == "__main__":
    

    pass
  