from amadeusgpt.analysis_objects.relationship import Relationship
from amadeusgpt.analysis_objects.event import EventGraph
from amadeusgpt.utils import timer_decorator
from .base import Manager
from .object_manager import ObjectManager
from .animal_manager import AnimalManager
from .relationship_manager import RelationshipManager
import numpy as np
from typing import List, Dict, Any, Union, Set, Optional
from amadeusgpt.analysis_objects.event import BaseEvent, Event
from amadeusgpt.api_registry import register_class_methods, register_core_api
from .base import Manager, cache_decorator
from amadeusgpt.analysis_objects.relationship import Orientation

def find_complement_number(string):
    digits = ""
    for char in string:
        if char.isdigit() or char == ".":
            digits += char
    return str(360 - float(digits))

def find_complement_operator(string):
    operator = ""
    for char in string:
        if not char.isdigit() and not char == ".":
            operator += char
    operator = operator.strip()
    if operator == "<":
        return ">="
    elif operator == "<=":
        return ">"
    elif operator == ">":
        return "<="
    elif operator == ">=":
        return "<"

def process_animal_object_relation(relation_query: str, comparison: str, animal_object_relation: np.ndarray):
    if relation_query in ["relatve_head_angle"]:
        complement_relation = find_complement_operator(comparison)
        digits = find_complement_number(comparison)
        complement_comparison = complement_relation + digits
        relation_string = "numeric_quantity" + complement_comparison
        complement_animal_object_relation = eval(relation_string)
        animal_object_relation |= complement_animal_object_relation
    return animal_object_relation

@register_class_methods
class EventManager(Manager):
    def __init__(self, 
                 config: Dict[str, any],
                 object_manager: ObjectManager,
                 animal_manager: AnimalManager,
                 relationship_manager: RelationshipManager,
                 use_cache:bool = False
                 ):
        super().__init__(config, use_cache = use_cache)
        self.config = config
        self.object_manager = object_manager
        self.animal_manager = animal_manager
        self.relationship_manager = relationship_manager
        self.video_file_path = self.config["video_info"]["video_file_path"]
        self.pixels_per_cm = self.config["video_info"]["pixels_per_cm"]
        self.animals_object_events = []
        self.animals_animals_events = []
        self.animals_state_events = []
    @register_core_api
    @cache_decorator
    def get_animals_object_events(
        self,
        object_name: str,
        relation_query: str,
        comparison: Optional[str] = None,
        negate=False,
        bodypart_names: Optional[Union[List[str], None]] = None,
        min_window: int = 0,
        max_window: int = 1000000,
        smooth_window_size = 3,
    ) -> List[BaseEvent]:
        """
        This function is only used when there is object with name involved in the queries.               
        object_name : str
        This parameter represents the name of the object of interest. It is expected to be a string.
        The accepted naming conventions include numeric strings (e.g., '0', '1', '2', ...), or
        the prefix 'ROI' followed by a number (e.g., 'ROI0', 'ROI1', ...). 
        relation_query: str. Must be one of ['to_left', 'to_right', 'to_below', 'to_above', 'overlap', 'distance', 'angle', 'orientation']
        comparison : str, Must be a comparison operator followed by a number like <50, optional
        bodypart_names: List[str], optional
           bodyparts of the animal
        min_window: min length of the event to include
        max_window: max length of the event to include
        negate: bool, default false
           whether to negate the spatial events. For example, if negate is set True, inside roi would be outside roi       
        Examples
        --------
         # find events where the animal is to the left of object 6  
         def task_program():
             behavior_analysis = AnimalBehaviorAnalysis()
             left_to_object6_events = behavior_analysis.animals_object_events('6', 'to_left',  bodypart_names = ['all'])
             return left_to_object6_events  
        """        
        # ugly fix for ROI
        if (
            object_name not in self.object_manager.get_object_names()
        ):
            raise ValueError(f"{object_name} is not defined. You need to define or draw the ROI first. Please do not write code before the user provides the ROI")
        # we only use numbers to represent objects for now
        object_name = object_name.replace("object", "")
        if bodypart_names is not None:
            self.animal_manager.update_roi_keypoint_by_names(bodypart_names)

        # TODO add angles between two animals' axis
        is_numeric_query = False
        if relation_query in [
            "distance",
            "relative_angle",
            "orientation",
            "closest_distance",
            "relative_head_angle",
            "relative_speed",
        ]:
            is_numeric_query = True            
                    
        animals_objects_relations = self.relationship_manager.get_animals_objects_relationships(animal_bodyparts_names=bodypart_names)
       
        ret_events = []
        for animal_objects_relationship in animals_objects_relations:
            if animal_objects_relationship.object_name != object_name:
                continue
            # to construct the events
            mask:List[bool] = None
            sender_animal_name = animal_objects_relationship.sender_animal_name

            if is_numeric_query:
                numeric_quantity  = animal_objects_relationship.query_relationship(relation_query)
                # e.g. "250 >= 10"
                relation_string = "numeric_quantity" + comparison
                mask = eval(relation_string)            
                mask = process_animal_object_relation(relation_query, comparison, mask)
            else:
                object_names = self.object_manager.get_object_names()
                assert (
                    object_name in object_names
                ), f"{object_name} not in available list of objects. Available objects are {object_names}"
                mask = animal_objects_relationship.query_relationship(relation_query)


            events: List[BaseEvent] = Event.mask2events(mask, 
                                                        self.video_file_path, 
                                                        sender_animal_name,
                                                        set(),
                                                        {object_name},
                                                        smooth_window_size = smooth_window_size)
                                                                                                                  
            if negate:
                events = Event.event_negate(events)
           
            events = Event.filter_events_by_duration(events, min_window, max_window)
           
            ret_events.extend(events)
        
        if bodypart_names is not None:
            self.animal_manager.restore_roi_keypoint()

        return ret_events
    @cache_decorator
    @register_core_api
    def get_animals_state_events(
        self,
        query_name : str,
        comparison : str,
        bodypart_names: Optional[List[str]] = None,
        min_window:Optional[int]=0,
        max_window:Optional[int]=1000000,
        smooth_window_size:Optional[int]=3
    ) -> List[BaseEvent]:
        """
        Parameters
        ----------
        query_name: str
            Must be one of 'speed', 'acceleration', 'bodypart_pairwise_distance'
        comparison: str
            Must be a comparison operator followed by a number like <50,
        Returns
        -------
        List[BaseEvent]
        Examples that create task programs using this API. 
        --------
        # A task program that captures events where animal moving faster than 3 pixels across frames.       
        def get_faster_than_3():
            behavior_analysis = AnimalBehaviorAnalysis()
            animal_faster_than_3_events =  behavior_analysis.get_animals_state_events('speed', '>3')
            return animal_faster_than_3_events
        # A task program that captures events where animal's nose to its own tail_base distance larger than 10
        def example_task_program():
            behavior_analysis = AnimalBehaviorAnalysis()
            nose_tail_base_larger_than_10_events =  behavior_analysis.get_animals_state_events(['bodypart_pairwise_distance'], '>10', bodypart_names = ['nose', 'tail_base'])
            return nose_tail_base_larger_than_10_events 
        """
        if bodypart_names is not None:
            self.animal_manager.update_roi_keypoint_by_names(bodypart_names)
        ret_events = []

        for sender_animal_name in self.animal_manager.get_animal_names():
            # to construct the events
            state = self.animal_manager.query_animal_states(sender_animal_name, query_name)
            if len(state.shape) == 3:
                # the mask must be 2D. For example, if the user asks the speed of the animal. We make it the average of the speed of all the bodyparts
                state = np.nanmedian(state, axis=1)

            relation_string = "state" + comparison

            mask = eval(relation_string)
         
            events = Event.mask2events(mask,
                                        self.video_file_path,
                                        sender_animal_name,
                                        set(),
                                        set(),
                                        smooth_window_size = smooth_window_size)
        
             
            events = Event.filter_events_by_duration(events, min_window, max_window)
            ret_events.extend(events)

        ret_events = sorted(ret_events, key=lambda x: x.start)
        if bodypart_names is not None:
            self.animal_manager.restore_roi_keypoint()

        return ret_events
    @cache_decorator
    def get_events_from_relationship(self,
                                    relationship: Relationship,
                                    relation_query: str,
                                    comparison: str,                                 
                                    smooth_window_size: int) -> List[BaseEvent]:
 
        mask = relationship.query_relationship(relation_query)
     
        # determine whether the mask is a numpy of float or numpy of boolean
        if mask.dtype == float:
            relation_string = 'mask' + comparison
            mask =  eval(relation_string)  

         
        sender_animal_name = relationship.sender_animal_name
        receiver_animal_names = set([relationship.receiver_animal_name])
        if relationship.object_name is not None:
            object_names = set([relationship.object_name])
        else:
            object_names = set()
       
        events = Event.mask2events(mask,
                                    self.video_file_path,
                                    sender_animal_name,
                                    receiver_animal_names,
                                    object_names,
                                    smooth_window_size = smooth_window_size)


        return events
                                     
    @cache_decorator
    def get_animals_animals_events(
        self,
        cross_animal_query_list:List[str]=[],
        cross_animal_comparison_list:List[str]=[],
        bodypart_names: Optional[Union[List[str], None]]=None,
        otheranimal_bodypart_names: Optional[Union[List[str], None]] = None,
        min_window:int = 11,
        max_window:int = 1000000,
        smooth_window_size:int =3,
    )-> List[BaseEvent]:
        """
        The function is specifically for capturing multiple animals social events        
        Parameters
        ----------        
        cross_animal_query_list: 
        list of queries describing relative states among animals. The valid relative states can be and only can be any subset of the following ['to_left', 'to_right', 'to_below', 'to_above', 'overlap', 'distance', 'relative_speed', 'orientation', 'closest_distance', 'relative_angle', 'relative_head_angle']. Note distance and closest_distance are distinct queries. Also relative_angle and relative_head_angle are distinct queries.        
        cross_animal_comparison_list:
	    This list consists of comparison operators such as '==', '<', '>', '<=', '>='. Every comparison operator uniquely corresponds to an item in relation_query_list.
        IMPORTANT: the `inter_individual_animal_state_comparison_list[i]` and `inter_individual_animal_state_query_list[i]` should correspond to each other. Also the length of two lists should be the same
        Note the same relative state can appear twice to define a range.  For example, we can have ['distance', 'distance'] correspond to comparison list ['<100', '>20']       
        bodypart_names:
        list of bodyparts for the this animal. By default, it is None, which means all bodyparts are included
        otheranimal_bodypart_names: list[str], optional
        list of bodyparts for the other animals. By default, it is None, which means all bodyparts are included
        min_window: int, optional
        Only include events that are longer than min_window
        pixels_per_cm: int, optional
        how many pixels for 1 centimer
        smooth_window_size: int, optional
        smooth window size for smoothing the events.
        Returns
        -------        
        List[BaseEvent]
        Examples that create task programs using this API. 
        --------
        # A customized task program. Define <|run_after|> as a behavior that when animal A and other animals are within 100 pixels and other animals are in front of animal A.
        def run_after():
            behavior_analysis = AnimalBehaviorAnalysis()
            run_after_social_events = behavior_analysis.get_animals_animals_events(cross_animal_query_list = ['distance', 'orientation'], 
                cross_animal_comparison_list = [f'< 100', f'=={Orientation.FRONT}'])
            return run_after_social_events 
        # Define <|far_away|> as a social behavior where distance between animals is larger than 20 pixels and smaller than 100 pixels and head angle less than 15
        def far_away():
            behavior_analysis = AnimalBehaviorAnalysis()
            far_away_social_events = behavior_analysis.get_animals_animals_events(cross_animal_query_list = ['distance', 'distance', 'relative_head_angle'],
                cross_animal_comparison_list = ['> 20', '< 100', '<15'])                          
            return far_away_social_events 
        """
       
        animals_animals_relationships = self.relationship_manager.get_animals_animals_relationships(
            sender_animal_bodyparts_names=bodypart_names,
            receiver_animal_bodyparts_names=otheranimal_bodypart_names
        )
        all_events = []
        for relationship in animals_animals_relationships:            
            for query, comparison in zip(cross_animal_query_list, cross_animal_comparison_list):

                events = self.get_events_from_relationship( 
                                                                relationship,
                                                                query, 
                                                                comparison,                                                              
                                                                smooth_window_size)
                all_events.extend(events)
        graph = EventGraph.init_from_list(all_events)
        graphs = []
        for animal_name in self.animal_manager.get_animal_names():
            for receiver_animal_name in self.animal_manager.get_animal_names():
                if animal_name != receiver_animal_name:
                    subgraph = EventGraph.fuse_subgraph_by_kvs(
                        graph,
                        {'sender_animal_name': animal_name,
                        'receiver_animal_names': set([receiver_animal_name])},
                        number_of_overlap_for_fusion=0 if len(cross_animal_query_list) == 1 else len(cross_animal_query_list)
                    )
                    graphs.append(subgraph)
              
        graph = EventGraph.merge_subgraphs(graphs)
        
        ret_events = graph.to_list()

        ret_events = Event.filter_events_by_duration(ret_events, min_window, max_window)
        

        return ret_events


    def get_composite_events(self,
                            *events_list:List[List[BaseEvent]],
                            composition_type : str = "sequential", 
                            merge_keys: Optional[List[str]] = None,                            
                             ) -> List[BaseEvent]:
        """
        Parameters
        ----------
        events_list: List[dict] 
        Returns
        -------
        List[BaseEvent]
        Examples
        --------
         # An example task program that get events for the animal's nose overlaps the roi0, left eye overlaps the roi0 and tail base not overlap the roi0     
         def NoseEyeNoTailOverlapRoi0():
             behavior_analysis = AnimalBehaviorAnalysis()
             nose_left_eye_in_roi0_events =  behavior_analysis.animals_object_events('ROI0', 'overlap', bodypart_names = ['nose', 'left_eye'], negate = False)        
             tail_base_not_in_roi0_events = behavior_analysis.animals_object_events('ROI0', 'overlap', bodypart_names = ['tail base'], negate = True)
             return Event.add_simultaneous_events(nose_left_eye_in_roi0_events, tail_base_not_in_roi0_events)             
        """
        assert len(events_list) > 1, "You need to provide at least two events to merge"
        assert composition_type in ["sequential", "logical_and", "logical_or"], "composition_type must be either 'sequential' or 'logical_or', or 'logical_and'"

        if composition_type == "sequential":
            graph_list = [EventGraph.init_from_list(events) for events in events_list]
            graphs = []
            for animal_name in self.animal_manager.get_animal_names():
                for i in range(1, len(graph_list)):            
                    sub_graph = EventGraph.concat_graphs(graph_list[i-1], 
                                                         graph_list[i],
                                                         {'sender_animal_name': animal_name}
                    )
                    graphs.append(sub_graph)
            graph = EventGraph.merge_subgraphs(graphs)
            return graph.to_list()
        
        elif composition_type == "logical_or":
            all_events = []
            for events in events_list:
                all_events.extend(events)

            return all_events                

        elif composition_type == "logical_and":
            all_events = []
            for idx, events in enumerate(events_list):               
                all_events.extend(events)
            graph = EventGraph.init_from_list(all_events)
            graphs = []
            if merge_keys is None:                
                for animal_name in self.animal_manager.get_animal_names():
                    subgraph = EventGraph.fuse_subgraph_by_kvs( 
                                        graph,
                                        {'sender_animal_name': animal_name},
                                        number_of_overlap_for_fusion=len(events_list)
                    )
                    graphs.append(subgraph)
                
            else:
                
                for animal_name in self.animal_manager.get_animal_names():
                    if "receiver_animal_names" in merge_keys:
                        receiver_animal_names = [animal_name for animal_name in self.animal_manager.get_animal_names()]                        
                        for receiver_animal_name in receiver_animal_names:
                            if animal_name != receiver_animal_name:
                                receiver_animal_name = set([receiver_animal_name])
                                subgraph = EventGraph.fuse_subgraph_by_kvs( 
                                                        graph,
                                                        {'sender_animal_name': animal_name, 
                                                        'receiver_animal_names': receiver_animal_name},
                                                        number_of_overlap_for_fusion=len(events_list)
                                    )
                               
                               
                                graphs.append(subgraph)
                    if "object_names" in merge_keys:
                        for object_name in self.object_manager.get_object_names():
                            subgraph = EventGraph.fuse_subgraph_by_kvs(
                                                graph,   
                                                {'sender_animal_name': animal_name, 
                                                'object_names': object_name},
                                                number_of_overlap_for_fusion=len(events_list)
                            )
                            graphs.append(subgraph)
            graph = EventGraph.merge_subgraphs(graphs)
            return graph.to_list()  
    
    def get_serializeable_list_names(self) -> List[str]:
        return ["animals_object_events", "animals_animals_events", "animals_state_events"]
