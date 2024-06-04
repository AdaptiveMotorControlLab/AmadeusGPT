from asyncio import events
from cv2 import merge
import numpy as np
from typing import List, Tuple, Union, Optional, Dict, Set, Any
from numpy import block, ndarray, object_, rec
from amadeusgpt.analysis_objects.base import AnalysisObject
from amadeusgpt.utils import get_fps, smooth_boolean_mask, group_consecutive, get_video_length
import copy
import os
from collections import defaultdict
from functools import reduce



class BaseEvent(AnalysisObject):
    def __init__(self, 
                 start: int,
                 end: int,
                 video_file_path: str,
                 data_length: int,
                 ):
        # sometimes the frame length is not equal to the data length due to incomplete reading of the video file from pose
        self.video_file_path = video_file_path
        self.data_length = data_length
        self.start = start
        self.end = end
        assert os.path.exists(self.video_file_path), f"the video file {video_file_path} does not exist"
        self.frame_length = get_video_length(self.video_file_path)     

        duration_in_frames = self.end - self.start
        self.duration_in_frames = duration_in_frames
        duration_in_seconds  = round( duration_in_frames / get_fps(self.video_file_path), 2)
        if duration_in_seconds < 0:
            self.duration_in_seconds = 0
        else:
            self.duration_in_seconds = round( duration_in_frames / get_fps(self.video_file_path), 2)

    def __len__(self)-> int:
        return self.end - self.start + 1
   
    def generate_mask(self) -> ndarray:
        temp = np.zeros(self.data_length, dtype=bool)
        temp[self.start : self.end + 1] = True
        return temp 
    
    @classmethod
    def get_start_end_from_mask(cls, mask: ndarray) -> Tuple[int, int]:
        """
        Get the indices of first true and last true in the binary mask
        """
        start = np.argmax(mask)
        end = len(mask) - np.argmax(mask[::-1]) - 1
        return start, end

    def __lt__(self, other):
        return self.duration_in_frames < other.duration_in_frames
           


class Event(BaseEvent):
    """
    Methods   
    """
    def __init__(self,
                start: int,
                end: int,
                video_file_path: str,
                data_length: int,
                sender_animal_name: str,
                receiver_animal_names: Optional[Set[str]] = set(),
                object_names: Optional[Set[str]] = set()
                ):
        # we allow only single sender animal for the event

        super().__init__(start, end, video_file_path, data_length)
        assert isinstance(sender_animal_name, str), "sender_animal_name must be a string"
        self.sender_animal_name = sender_animal_name
        if receiver_animal_names is not None:
            assert isinstance(receiver_animal_names, set), "receiver_animal_names must be a set"        
        self.receiver_animal_names = receiver_animal_names
        if object_names is not None:
            assert isinstance(object_names, set), "object_names must be a set"
        self.object_names = object_names

    def __str__(self):
        return f'''
sender: {self.sender_animal_name}
receiver: {self.receiver_animal_names}
object: {self.object_names}
duration_in_seconds: {self.duration_in_seconds}
duration_in_frames: {self.duration_in_frames}
start: {self.start}
end: {self.end}
'''
    

    def __hash__(self):
        # Generate a hash based on a tuple of sorted (attribute, value) pairs
        # This accounts for all attributes dynamically
        items = tuple(sorted(self.__dict__.items()))
        return hash(items)            

    def summary(self):
        ignore = ['mask', 'video_file_path']
        for attr_name in self.__dict__:
            if attr_name not in ignore:
                print (f'{attr_name}: {self.__dict__[attr_name]}')
    @classmethod
    def blockfy(cls, masks):
        blocks = []
        start = None
        for i, val in enumerate(masks):
            if val is not None and start is None:
                # Start of a new block
                start = i
            elif not val and start is not None:
                # End of a block
                if start != i - 1:
                    blocks.append((start, i - 1))
                start = None
        # If there is an open block at the end of the array, add it to the list
        if start is not None:
            blocks.append((start, len(masks) - 1))
        return blocks        
    
    @classmethod
    # UNDER CONSTRUCTION
    def remove_overlapping_events(cls, events:List[BaseEvent]) -> List[BaseEvent]:
        # Assuming events are already sorted by start time
        filtered_events = []
        prev_event = None
        for event in events:           
            if prev_event and event.end == prev_event.end:
                # Replace the previous event with the current one if they have the same end time
                filtered_events[-1] = event
            elif not prev_event or event.start >= prev_event.end:
                # Add the event if there is no previous event or it doesn't overlap
                filtered_events.append(event)

            prev_event = event
        return filtered_events
   
    @classmethod
    # UNDER CONSTRUCTION
    def event_negate(cls, events: List[BaseEvent]) -> List[BaseEvent]:
        """
        Get list of events that are negates of those events
        The core attributes must stay the same
        """        
        #mask = np.zeros_like(events[0].mask, dtype=bool)
        mask = np.zeros(events[0].data_length, dtype=bool)
        sender_animal_name = events[0].sender_animal_name
        for event in events:
            assert sender_animal_name == event.sender_animal_name
        video_file_path = events[0].video_file_path
        for event in events:
            mask |= event.mask
        negate_mask = ~mask

        start,end = cls.get_start_end_from_mask(negate_mask)

        negate_events = Event.mask2events(start, end, video_file_path, sender_animal_name)

        return negate_events

    @classmethod
    def check_max_in_sum(cls, events: List[BaseEvent]):
        """
        Check the maximum value in the sum of the masks of the events
        """
        masks = [event.generate_mask() for event in events]
        _sum = np.sum(masks, axis = 0)
        return np.max(_sum)


    @classmethod
    def events2onemask(cls, events: List[BaseEvent])-> ndarray:
        """
        return events to one mask that is a union of all the events
        """
        invovled_animals = set()
        for event in events:
            invovled_animals.add(event.sender_animal_name)
        assert len(invovled_animals) <= 1, f"This function must be called for events that have the same sender animal, you had {invovled_animals}"
        assert len(events) > 0, "events list is empty"
        
        mask = np.zeros(events[0].data_length, dtype=bool)
        for event in events:
            start, end = event.start, event.end            
            mask[start : end + 1] = True       
        return mask

    @classmethod
    def mask2events(cls, 
                     mask:ndarray, 
                     video_file_path: str,
                     sender_animal_name: str,
                     receiver_animal_names: Set[str],
                     object_names: Set[str],
                     smooth_window_size: int =5
                     ) -> List[BaseEvent]:
        """
        Turn a binary mask to a list of Events
        The resulted event has the same sender animal name
        Each segment of the mask might have multiple receiver animals and object names
        Returns
        """
        if smooth_window_size is not None:
            mask = smooth_boolean_mask(mask, smooth_window_size)
        blocks = cls.blockfy(mask)
        events = []
        
        for block in blocks:
            start, end = block          
            if end is not None:
                events.append(Event(start,
                                    end,
                                    video_file_path, 
                                    len(mask),
                                    sender_animal_name,
                                    receiver_animal_names = receiver_animal_names,
                                    object_names = object_names, 
                                    ))
        
        return events
  
   
    @classmethod
    # UNDER CONSTRUCTION
    def filter_events_by_duration(cls, events: List[BaseEvent], 
                                  min_duration: Union[int, float], 
                                  max_duration: Union[int, float],
                                  unit: str = "frames"
                                  ) -> List[BaseEvent]:
        """
        Filter events by duration in frames
        """
        assert unit in ["frames", "seconds"], "must be either frames and seconds"

        if unit == "seconds":
            min_duration = int(min_duration * get_fps(events[0].video_file_path))
            max_duration = int(max_duration * get_fps(events[0].video_file_path))
            return [event for event in events if event.duration_in_seconds >= min_duration and event.duration_in_seconds <= max_duration]
        else:
            min_duration = int(min_duration)
            max_duration = int(max_duration)

            return [event for event in events if event.duration_in_frames >= min_duration and event.duration_in_frames <= max_duration]
        
    @classmethod
    # UNDER CONSTRUCTION
    def concat_two_events(cls,
                           early_event: BaseEvent,
                           late_event: BaseEvent,
                           ) -> Union[None, BaseEvent]:
        """
        Concatenate two events into one, fill the gap between two events if there are
        """
        assert early_event.sender_animal_name == late_event.sender_animal_name

        object_names = early_event.object_names
        receiver_animal_names = early_event.receiver_animal_names

        new_event = Event(early_event.start,
                          late_event.end,
                          early_event.video_file_path,
                          early_event.data_length,
                          early_event.sender_animal_name,
                          receiver_animal_names = receiver_animal_names,
                          object_names = object_names)
        return new_event


# Trying to implement temporal graph
class Node:
    def __init__(self, start: int,
                  children: List[BaseEvent]):
        self.children = children
        self.start: int = start
        self.next: Node = None
        self.prev: Node = None

    def get_children_by_key(self, key: str):
        """        
        """
        assert key in ['sender_animal_name', 'receiver_animal_name', 'object_name']
        return [child for child in self.children if child.key == key]

    @classmethod
    def copy(cls, node):
        ret = cls(node.start, node.children)
        ret.prev = None
        ret.next = None
        return ret
      
class EventGraph:
    """
    Implementing a temporal graph
    Every node keeps a collection of events that have the same start time

    """
    def __init__(self):
        self.head: Node = None
        self.n_nodes = 0    

    def to_list(self)-> List[BaseEvent]:
        ret = []
        cur_node = self.head
        while cur_node is not None:
            ret.extend(cur_node.children)
            cur_node = cur_node.next
        ret = sorted(ret, key=lambda x: x.start)
        ret = [e for e in ret if e.duration_in_frames > 1]
                   
        return ret
    @property
    def animal_names(self):
        return list(set([event.sender_animal_name for event in self.to_list()]))

    @classmethod
    def check_list_sorted(cls, events: List[BaseEvent])-> bool:
        """
        Check if the list of events is sorted
        """
        temp = [event.start for event in events]
        return temp == sorted(temp)

    @classmethod
    def init_from_list(cls, events: List[BaseEvent])-> "EventGraph":
        graph = cls()
        start_time_dict = {}

        for event in events:
            start = event.start
            if start in start_time_dict:
                start_time_dict[start].append(event)
            else:
                start_time_dict[start] = [event]
        
        for start_time in start_time_dict:

            node = Node(start_time, 
                        start_time_dict[start_time])
            graph.insert_node(node)

        return graph

    @classmethod
    def init_from_mask(cls,
                    mask: ndarray, 
                    video_file_path: str,
                    sender_animal_name: str, 
                    receiver_animal_names: Set[str],
                    object_names: Set[str]) -> "EventGraph":

        events = Event.mask2events(mask, 
                                   video_file_path, 
                                   sender_animal_name, 
                                   receiver_animal_names=receiver_animal_names, 
                                   object_names=object_names)        



        graph = cls.init_from_list(events)        
       
        return graph
    
    def insert_node(self, new_node: Node):
        """
        Insert a new node into the graph
        We greedily insert the node after the first node that has a start time less than the new node
        We leave the follow-up operations in other functions
        """
        # we might reuse those nodes so we need to clean them        

        new_node = Node.copy(new_node)
        # find the right place to insert the node and update the linked list                
        # empty graph             

        if self.head is None:
            self.head = new_node
            self.n_nodes+=1
            return     
 
        # only one node in the graph   

        if new_node.start < self.head.start:
            new_node.next = self.head
            self.head.prev = new_node
            self.head = new_node
            return 
        if self.head.next is None:          
            if new_node.start == self.head.start:
                self.head.children.extend(new_node.children)
            else:
                self.head.next = new_node
                new_node.prev = self.head
            self.n_nodes+=1
            return

        cur_node = self.head


        while cur_node.next is not None:
            if new_node.start > cur_node.start and new_node.start < cur_node.next.start:                        
                new_node.next = cur_node.next
                new_node.next.prev = new_node
                cur_node.next = new_node
                new_node.prev = cur_node                                
                self.n_nodes+=1
                return 
            elif cur_node.start == new_node.start: 
                # let's check not to insert a basically equivalent node
                cur_node.children.extend(new_node.children)
                self.n_nodes+=1
                return
            else:
                cur_node = cur_node.next 
        # the new node is the last node
        
        cur_node.next = new_node
        new_node.prev = cur_node

        self.n_nodes+=1
        

    @classmethod
    def animal_group_subgraph(cls,
                                graph: "EventGraph",
                                n_individuals: int):
        """
        WIP
        Retrieve the subgraph where multiple animals satisfy the same condition at the same time.        
        """
        events = graph.to_list()
        animals = set([event.sender_animal_name for event in events])
        masks = []
        for idx, animal_name in enumerate(animals):
            events_animal = [event for event in events if event.sender_animal_name == animal_name]
            # find the overlapping between events_animal_a and events_animal_b
            mask = Event.events2onemask(events_animal)
            masks.append(mask)
        masks = np.concatenate(masks, axis = 0)
        _sum = np.sum(masks, axis = 0)
        ret = np.zeros_like(mask, dtype = bool)
        ret[_sum == n_individuals] = True
        # get where _sum 
        return cls.init_from_mask(ret, events[0].video_file_path, events[0].sender_animal_name, set(), set())



    @classmethod
    def fuse_subgraph_by_kvs(cls,
                        graph: "EventGraph",
                        merge_kvs: Dict[str, Any],
                        number_of_overlap_for_fusion: int = 0,
                        allow_more_than_2_overlap:bool = False
                    ) -> "EventGraph":
        """
        number_of_overlap_for_fusion is a parameter for logical and.
        For example, if there are two conditions to be met in the masks we look for locations that have overlap as 2
        """       
        # retrieve all events that satisfy the conditions (k=v) 
        events = graph.traverse_by_kvs(merge_kvs)

        if not allow_more_than_2_overlap:
            assert Event.check_max_in_sum(events) <= number_of_overlap_for_fusion, f"Detected overlap {Event.check_max_in_sum(events)}. But we only allow {number_of_overlap_for_fusion} overlap for fusion."
       
        new_graph = cls() 
        if len(events) == 0:
            return new_graph
        
        
        masks = [event.generate_mask() for event in events]         
        _sum = np.sum(masks, axis = 0)          
        mask = np.zeros_like(_sum, dtype=bool)
        # in case there are many events overlap, 
        # if the events come from one single condition, there is no overlap
        # so the overlap must come from different conditions
        mask[(_sum >= number_of_overlap_for_fusion)] = True
        
        # the logic of merging receiver names across all events from kv traverse is wrong
        # especially when we have animal state event to be fused
        events = Event.mask2events(mask,
                                        events[0].video_file_path,
                                        events[0].sender_animal_name,
                                        receiver_animal_names = set.union(*[event.receiver_animal_names for event in events]),
                                        object_names = set.union(*[event.object_names for event in events])
                                        )     
        for event in events:
            new_graph.insert_node(Node(event.start, [event]))       
      

        return new_graph

    
    @classmethod
    def merge_subgraphs(cls, graph_list: List["EventGraph"])-> "EventGraph":
        """
        Merge graphs into one graph. There is no fusion. Nodes are just added based on the start time
        """
        graph = cls()
        for g in graph_list:
            head = g.head
            while head is not None:
                graph.insert_node(head)
                head = head.next
        return graph

    def traverse_by_kvs(self, kvs: Dict[str, Any])-> List[BaseEvent]:
        """
        Traverse the graph and get a list of events that have the same key
        """  
        ret = []
        cur_node = self.head
        while cur_node is not None:
            for event in cur_node.children:
                conditions = []
                for k,v in kvs.items():
                    # empty receiver name set() should match any node
                    if isinstance(v, set):                       
                        conditions.append(len(getattr(event,k)) == 0 or v.issubset(getattr(event, k)))
                    else:   
                        conditions.append(getattr(event, k) == v)                        
                if all(conditions):
                    ret.append(event)

            cur_node = cur_node.next
        return ret
       
    @classmethod
    def concat_graphs(cls,
                        graph1: "EventGraph",
                        graph2: "EventGraph",
                        merge_kvs: Dict[str, Any],
                        max_interval_between_sequential_events
                        ) -> "EventGraph":
        """
        Concatenate two graphs in a sequential manner
        For every node in the concated graph, it must be a node from grpah1 followed by a node from graph2
        """
        graph = cls()      

        events1 = graph1.traverse_by_kvs(merge_kvs)
        events2 = graph2.traverse_by_kvs(merge_kvs)

        secure_events1 = []
        secure_events2 = []


        for i, event2 in enumerate(events2):
            for j, event1 in enumerate(events1):
                
                # skip if event1 is not the closest event to event2
                if j < len(events1) - 1 and events1[j+1].end <= event2.start:
                    continue
                else:
                    # this is strict continuous
                    #if event1.end >= event2.start and event1.end <= event2.end:
                    # this is not strict continuous
                    if  event1.duration_in_frames > 0 and \
                        event1.end <= event2.start and \
                        event2.start - event1.end <= max_interval_between_sequential_events:
                        secure_events1.append(event1)
                        secure_events2.append(event2)
                        new_event = Event.concat_two_events(event1, event2)
                        graph.insert_node(Node(new_event.start, [new_event]))
                        break                
        # print ('check secure events1')
        # for event in secure_events1:
        #     print (event)
        # print ('check secure events2')
        # for event in secure_events2:
        #     print (event)

        return graph


    def display_graph(self):
        #print ('asked to display')
        cur_node = self.head
        while cur_node is not None:
            for event in cur_node.children:
                print (event.start, event.end)

            cur_node = cur_node.next

