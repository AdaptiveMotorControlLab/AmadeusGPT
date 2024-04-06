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
from matplotlib.patches import Wedge
import cv2


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
    
    def get_head_orientation_visualization(self,
                                           events,
                                           render:bool = False):
        ARROW_LENGTH = 20
        CONE_WIDTH = 30
        CONE_LENGTH = 150
        animals = self.animal_manager.get_animals()
        head_cs = [animal.calc_head_cs() for animal in animals]
        fig, axs = plt.subplots(self.animal_manager.get_n_individuals(), 
                                        self.animal_manager.get_n_individuals()-1, squeeze = False)
        for sender_idx, sender_animal in enumerate(self.animal_manager.get_animals()):
            other_animals = [other_animal for other_animal in self.animal_manager.get_animals() if other_animal.name != sender_animal.name]
            axs[sender_idx][0].set_ylabel(sender_animal.get_name())
            for receiver_idx, receiver_animal in enumerate(other_animals):
                
                for event in events:
                    if receiver_animal.get_name() in event.receiver_animal_names:
                        start_frame = event.start
                        break

                scene_vis = self.get_scene_visualization(start_frame,
                                        axs = axs[sender_idx][receiver_idx])
                cs = [cs_[start_frame] for cs_ in head_cs]
                sexy_colors = ['fcf6bd', 'ffafcc', 'a2d2ff']
                sexy_colors = [f'#{c}' for c in sexy_colors]
                poses = self.animal_manager.get_keypoints()[start_frame]

                receiver_animal_kpts = receiver_animal.get_keypoints()[start_frame]
                receiver_animal_kpts = np.nanmean(receiver_animal_kpts, axis=0)
                # add text on that keypoint position
                axs[sender_idx][receiver_idx].text(receiver_animal_kpts[0],
                                                    receiver_animal_kpts[1], 
                                                    receiver_animal.get_name(),
                                                    color='white',
                                                    fontsize=12)

                scene_vis.draw()
                for pose, cs_, color in zip(poses, cs, sexy_colors):

                    axs[sender_idx][receiver_idx].scatter(*pose.T, s=1, c=color)
                    theta = np.rad2deg(np.arctan2(cs_[1, 0], cs_[0, 0]))
                    origin = tuple(cs_[:2, 2].astype(int))

                    xhat = cs_[:2, 0] * ARROW_LENGTH
                    yhat = -cs_[:2, 1] * ARROW_LENGTH
                    w = Wedge(origin, CONE_LENGTH, theta - CONE_WIDTH // 2, theta + CONE_WIDTH // 2, alpha=0.6, ec='none',\
                fc=color)
                    axs[sender_idx][receiver_idx].add_artist(w)
                    if True:
                        axs[sender_idx][receiver_idx].arrow(*origin, *xhat, head_width=10, color='r')
                        axs[sender_idx][receiver_idx].arrow(*origin, *yhat, head_width=10, color='g')
        if render:
            plt.show()
                

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
                                        self.animal_manager.get_n_individuals()-1, squeeze = False)

                for sender_idx, sender_animal in enumerate(self.animal_manager.get_animals()):
                    other_animals = [other_animal for other_animal in self.animal_manager.get_animals() if other_animal.name != sender_animal.name]
                    for receiver_idx, receiver_animal in enumerate(other_animals):
                        # change this to the start time of an event
                        start_frame = events[0].start
                        scene_vis = self.get_scene_visualization(start_frame,
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
                                        constrained_layout=True,
                                        squeeze = False
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
    

    def write_video(self, video_file_path, out_name, events):
        cap = cv2.VideoCapture(video_file_path)
            
        data = []
        for event in events:
            # mark who is the initiator
            # get the keypoints of sender and receiver(s)
            # get the x,y of the keypoint
            # get the time slice, sorted by time            
            sender_animal_name = event.sender_animal_name
            receiver_animal_names = event.receiver_animal_names
            time_slices = (event.start, event.end)
            sender_keypoints = self.animal_manager.get_animal_by_name(sender_animal_name).get_keypoints()
            receiver_keypoints = [self.animal_manager.get_animal_by_name(receiver_animal_name).get_keypoints() for receiver_animal_name in receiver_animal_names]
            sender_keypoints = np.nanmean(sender_keypoints, axis=1)
            sender_speeds = self.animal_manager.get_animal_by_name(sender_animal_name).get_speed()

            receiver_keypoints = [np.nanmean(receiver_keypoints, axis=1) for receiver_keypoints in receiver_keypoints]

            data.append({'time_slice': time_slices,
                        'sender_keypoints': sender_keypoints,
                        'receiver_keypoints': receiver_keypoints})                         
        
        total_duration = sum([event.duration_in_seconds for event in events])
        
        if total_duration < 0.5:
            return 
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Adjust the codec as needed
        out = cv2.VideoWriter(f'{out_name}', fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

        for triple in data:
            time_slice = triple['time_slice']
            sender_keypoints = triple['sender_keypoints']
            receiver_keypoints = triple['receiver_keypoints']
            cap.set(cv2.CAP_PROP_POS_FRAMES, time_slice[0])
            offset = 0
            
            while cap.isOpened():
                current_frame = time_slice[0] + offset
                ret, frame = cap.read()
                if not ret:
                    break
                if time_slice[0] <= current_frame < time_slice[1]:
                    # select the keypoint based on the frame number
                    sender_location = sender_keypoints[current_frame]
                    # put the text "sender" and "receiver" on corresponding location
                    sender_speed = np.nanmean(sender_speeds[current_frame])
                    sender_location = sender_location.astype(int)
                    cv2.putText(frame, 'sender', (sender_location[0], sender_location[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
                    speed_text = f"Sender Speed {sender_speed:.2f} units/frame"
                    speed_text_location = (frame.shape[1] - 200, frame.shape[0] - 50)  # 200 pixels wide space, 30 pixels from the bottom
                    cv2.putText(frame, speed_text, speed_text_location, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                    for receiver_id, receiver_location in enumerate(receiver_keypoints):
                        receiver_location = receiver_location[current_frame].astype(int)
            
                        # Calculate the distance between sender and receiver
                        distance = np.linalg.norm(sender_location - receiver_location)
                        # Format the distance to display only 2 decimal places
                        distance_text = f"Dist(receiver{receiver_id}) {distance:.2f} units"                        
                        
                        distance_text_location = (frame.shape[1] - 200, frame.shape[0] - 15 * (receiver_id + 1))  # 200 pixels wide space, 10 pixels from the bottom
                                                
                        cv2.putText(frame, distance_text, distance_text_location, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                        cv2.putText(frame, 'receiver', (receiver_location[0], receiver_location[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

                    out.write(frame)
                offset += 1
                if current_frame == time_slice[1]:
                    break

        # Release everything when job is finished
        cap.release()
        out.release()
        cv2.destroyAllWindows()        

    def generate_video_clips_from_events(self, video_file, 
                                         events: List[BaseEvent],
                                           behavior_name):
        """
        This function takes a list of events and generates video clips from the events
        1) For the same events, we first group events based on the video
        2) For the same event on the same video, we plot the animal name and the "sender" of the event
        3) Then we write those videos to the disk
        """
        
        videoname = video_file.split('/')[-1].replace('.mp4', '').replace('.avi', '')            
        video_name = f'{videoname}_{behavior_name}_video.mp4' 
        
        self.write_video(video_file,
                        video_name,
                        events)

    