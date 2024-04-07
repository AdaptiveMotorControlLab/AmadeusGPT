from .base import Manager
from .model_manager import ModelManager
from typing import List, Dict,Any
from amadeusgpt.analysis_objects.object import Object, ROIObject 
from amadeusgpt.programs.api_registry import register_class_methods, register_core_api

@register_class_methods
class ObjectManager(Manager):
    def __init__(self, 
                 config: Dict[str, Any], 
                 model_manager: ModelManager,
                 ):
        self.config = config
        self.model_manager = model_manager
        self.roi_objects = []
        self.seg_objects = []
        self.load_from_disk = config["object_info"]["load_objects_from_disk"]
        if self.load_from_disk:
            self.load_objects_from_disk()
        else:
            self.init()

    def summary(self):
        print("roi_objects: ", self.get_roi_object_names())
        print("seg_objects: ", self.get_seg_object_names())

    def get_roi_object_names(self)-> List[str]:
        return [obj.name for obj in self.roi_objects]
    
    def get_seg_object_names(self)-> List[str]:
        return [obj.name for obj in self.seg_objects]
       
    def load_objects_from_disk(self):
        pass
    def get_roi_objects(self)-> List[Object]:
        return self.roi_objects
    
    def get_seg_objects(self)-> list[Object]:
        return self.seg_objects
        
    def get_objects(self)-> List[Object]:
        return self.roi_objects + self.seg_objects
    def get_object_names(self)-> List[str]:
        return self.get_roi_object_names() + self.get_seg_object_names()

    def init(self):
        # run sam inference
        pass
        
    def add_roi_object(self, data: Any)-> None:
        # the user can add an object to the roi_objects
        if not isinstance(data, ROIObject):
            if isinstance(data, list):
                for e in data:
                    self.add_roi_object(e)
                return 
            else:
                roi_name = f'ROI_{len(self.get_object_names())}'
                object = ROIObject(roi_name, data)

        else:
            object = data
        self.roi_objects.append(object)

    def get_serializeable_list_names(self) -> List[str]:
        return ['roi_objects', 'seg_objects']