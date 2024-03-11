from amadeusgpt.analysis_objects.visualization import BaseVisualization, GraphVisualization, KeypointVisualization, SceneVisualization, EventVisualization, GraphVisualization
from .base import Manager
from .animal_manager import AnimalManager
from .object_manager import ObjectManager
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Union, Optional
from collections import defaultdict
from amadeusgpt.api_registry import register_class_methods, register_core_api
from amadeusgpt.analysis_objects.event import BaseEvent


def mask2distance(locations):
    assert len(locations.shape) == 2
    assert locations.shape[1] == 2
    diff = np.abs(np.diff(locations, axis=0))
    distances = np.linalg.norm(diff, axis=1)
    overall_distance = np.sum(distances)
    return overall_distance

@register_class_methods
class VisualManager(Manager):
    def __init__(self, 
                 config: Dict[str, Any],
                 animal_manager: AnimalManager,
                 object_manager: ObjectManager,
                 ):
        super().__init__(config)    
        self.config = config
        self.video_file_path = config['video_info']['video_file_path']
        self.animal_manager = animal_manager
        self.object_manager = object_manager

  
    def get_scene_visualization(self,                                 
                                 scene_frame_number: int,
                                 axs: Optional[plt.Axes] = None,
                                 ):
        # returns a vis id
        if axs is None:
            fig, axs = plt.subplots(1)
        objects = self.object_manager.get_objects()
        
        return SceneVisualization(
            axs,
            objects,
            self.video_file_path,
            scene_frame_number)
    @register_core_api
    def get_keypoint_visualization(self, 
                                    render:bool = False,
                                    bodypart_names: Optional[List[str]] = None,
                                    axs: Optional[plt.Axes] = None,
                                    average_keypoints: bool = True,
                                    frames:Optional[range] =  None,
                                    events: Optional[List[BaseEvent]] = None)-> None:
                
        """
        A function that visualizes the keypoints of the animals in the video. The 
        plot can optionally use events to sample the time points to visualize the corresponding trajectory.
        """

        if frames is None:
            frames = range(self.animal_manager.get_data_length())

        if axs is None:
            if events is not None and len(events) > 0 and \
                len(events[0].receiver_animal_names)>0:
           
                fig, axs = plt.subplots(self.animal_manager.get_n_individuals(), 
                                        self.animal_manager.get_n_individuals()-1)
                for sender_idx, sender_animal in enumerate(self.animal_manager.get_animals()):
                    other_animals = [other_animal for other_animal in self.animal_manager.get_animals() if other_animal.name != sender_animal.name]
                    for receiver_idx, receiver_animal in enumerate(other_animals):            
                        scene_vis = self.get_scene_visualization(self.config['video_info']['scene_frame_number'],
                                                                axs = axs[sender_idx][receiver_idx])
                                                
                        # always feed the full keypoints to visualization
                        full_keypoints = sender_animal.get_keypoints(average_keypoints=False)[frames, :]
                        full_bodypart_names = sender_animal.get_keypoint_names()
                        keypoint_vis = KeypointVisualization(axs[sender_idx][receiver_idx],
                                                            fig,                                                                                                  
                                                            full_keypoints,
                                                            sender_animal.get_name(),
                                                            receiver_animal.get_name(),
                                                            full_bodypart_names,
                                                            bodypart_names,                                                 
                                                            self.animal_manager.get_n_individuals(),
                                                            average_keypoints=average_keypoints,
                                                            events = events)
                        scene_vis.draw()
                        keypoint_vis.draw()

            else:
                if self.animal_manager.get_n_individuals() == 1:
                    fig, axs = plt.subplots(1)
                    axs = [axs]
                else:
                    fig, axs = plt.subplots(self.animal_manager.get_n_individuals())
                
                for idx, sender_animal in enumerate(self.animal_manager.get_animals()):
                    scene_vis = self.get_scene_visualization(self.config['video_info']['scene_frame_number'],
                                                                axs = axs[idx])
                    axs[idx].set_ylabel(sender_animal.get_name())     
                    # always feed the full keypoints to visualization
                    full_keypoints = sender_animal.get_keypoints(average_keypoints=False)[frames, :]
                    full_bodypart_names = sender_animal.get_keypoint_names()
                    keypoint_vis = KeypointVisualization(axs[idx],
                                                        fig,                                                                                                  
                                                        full_keypoints,
                                                        sender_animal.get_name(),
                                                        set(),
                                                        full_bodypart_names,
                                                        bodypart_names,                                                 
                                                        self.animal_manager.get_n_individuals(),
                                                        average_keypoints=average_keypoints,
                                                        events = events)
                    scene_vis.draw()
                    keypoint_vis.draw()
        if render:
            plt.show()

    def get_ethogram_visualization(self,
                                events: List[BaseEvent],
                                verbose: bool = False,
                                render: bool = False,
                                axs: Optional[plt.Axes] = None,
                                        ):
        """
        By default, it show just diplays the ethogram according
        to the output of the task program.
        However, since every event contains the information about the object, receiver animal and sender animal
        We can still make it more informative if we want

        """
        pass
        

    @register_core_api
    def get_animal_animal_visualization(self,
                                   events: List[BaseEvent],
                                   render:bool = False,
                                   axs: Optional[plt.Axes] = None)-> None:
        """
        A function that visualizes the ethogram of the animals in the video. It takes
        a list of events and visualizes the events in the video. 
        """

        involved_animals = set()
        for event in events:
            involved_animals = involved_animals.union(event.receiver_animal_names)     
        if axs is None:
            if len(involved_animals) > 0:
                fig, axs = plt.subplots(self.animal_manager.get_n_individuals(), 
                                        self.animal_manager.get_n_individuals()-1,
                                        constrained_layout=True
                                        )

                for sender_idx, sender_animal in enumerate(self.animal_manager.get_animals()):
                    other_animals = [other_animal for other_animal in self.animal_manager.get_animals() if other_animal.name != sender_animal.name]
                    for receiver_idx, receiver_animal in enumerate(other_animals):
                        if receiver_idx == 0:
                            axs[sender_idx][receiver_idx].set_ylabel(sender_animal.get_name())
                        event_vis = EventVisualization(axs[sender_idx][receiver_idx],
                                                    events,
                                                    sender_animal.get_name(),
                                                    set([receiver_animal.get_name()]),
                                                    self.config['video_info']['video_file_path'])
                        event_vis.draw()
            else:
                if self.animal_manager.get_n_individuals() == 1:
                    fig, axs = plt.subplots(1)
                    axs = [axs]
                else:
                    fig, axs = plt.subplots(self.animal_manager.get_n_individuals(),
                                        constrained_layout=True)

                for idx, animal in enumerate(self.animal_manager.get_animals()):
                    if idx == 0:
                            axs[idx].set_ylabel(animal.get_name())
                    event_vis = EventVisualization(axs[idx],
                                                events,
                                                animal.get_name(),
                                                set(),
                                                self.config['video_info']['video_file_path'])
                    event_vis.draw()
                                       

                
        if render:
            plt.subplots_adjust()
            plt.show() 
                                              
    
    def get_graph_visualization(self,
                                graph,
                                render:bool = False,
                                axs: Optional[plt.Axes] = None,
                                ):
        fig, axs = plt.subplots(1)
        graph_visualization = GraphVisualization(fig, axs, graph)
        ani = graph_visualization.draw()
        if render:
            plt.show()
    
    def get_serializeable_list_names(self) -> List[str]:
        return super().get_serializeable_list_names()