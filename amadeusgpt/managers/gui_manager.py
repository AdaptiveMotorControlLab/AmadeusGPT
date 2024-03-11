from .base import Manager
from .object_manager import ObjectManager
from typing import List, Dict, Any
import cv2
import matplotlib.lines as mlines
import matplotlib.pyplot as plt

from matplotlib.path import Path
from matplotlib.widgets import  PolygonSelector
import itertools
from amadeusgpt.api_registry import register_class_methods, register_core_api

class ROISelector:
    # Use itertools.cycle to cycle through colors indefinitely
    cmap = itertools.cycle(["red", "blue", "yellow", "green"])

    def __init__(self, axs):
        self.axs = axs
        self.selector = PolygonSelector(self.axs, self.onselect)
        self.paths = []

    def roi_select_event(self, vertices):
        # Create a new list for vertices to close the path without modifying the original list
        closed_vertices = vertices + [vertices[0]]
        path = Path(closed_vertices)
        self.paths.append(path)
        
        # Use the next color from the cycle for each new ROI
        edgecolor = next(type(self).cmap)
        
        # Add the new ROI patch
        self.axs.add_patch(
            plt.Polygon(path.vertices, fill=None, edgecolor=edgecolor)
        )
        
        # Update the legend for each ROI
        handles = [
            mlines.Line2D([], [], color=next(type(self).cmap), label=f"ROI {i}")
            for i in range(len(self.paths))
        ]
        self.axs.legend(handles=handles, loc="upper right")
        
        # Explicitly update the figure
        plt.draw()

    def onselect(self, vertices):
        self.roi_select_event(vertices)

def select_roi_from_video(video_filename):
    cap = cv2.VideoCapture(video_filename)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    middle_frame_index = total_frames // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_index)
    ret, frame = cap.read()
    fig, axs = plt.subplots(1)
    axs.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB
    selector = ROISelector(axs)
    plt.show()
    return selector.paths

def select_roi_from_plot(fig, ax):
    selector = ROISelector(ax)
    fig.show()
    return selector.paths

@register_class_methods
class GUIManager(Manager):
    def __init__(self, config: Dict[str, Any],
                 object_manager: ObjectManager
                 ):
        self.config = config
        self.object_manager = object_manager
        self.video_file_path = config["video_info"]["video_file_path"]
        self.videos = {}
    
    def add_roi_from_video_selection(self)-> None:
        ret = select_roi_from_video(self.video_file_path)
        print ("object to be added", ret)
        self.object_manager.add_roi_object(ret)

    
    def get_serializeable_list_names(self) -> List[str]:
        return []
    
    