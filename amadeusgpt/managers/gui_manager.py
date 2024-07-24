from typing import Any, Dict, List

import cv2
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.widgets import PolygonSelector

from amadeusgpt.analysis_objects.object import ROIObject
from amadeusgpt.behavior_analysis.identifier import Identifier
from amadeusgpt.programs.api_registry import register_class_methods

from .base import Manager
from .object_manager import ObjectManager


class ROISelector:
    roi_count = 0
    cmap = ["red", "blue", "yellow", "green"]

    def __init__(self, axs, object_manager):
        self.axs = axs
        self.object_manager = object_manager
        self.selector = PolygonSelector(self.axs, self.onselect)
        self.paths = []

    def roi_select_event(self, vertices):
        # once the bounding box is done drawing, run the following command
        first_point = vertices[0]
        vertices.append(first_point)
        path = Path(vertices)
        self.paths.append(path)
        # self.axs.clear()
        for i, path in enumerate(self.paths):
            self.axs.add_patch(
                plt.Polygon(path.vertices, fill=None, edgecolor=type(self).cmap[i])
            )
        handles = [
            mlines.Line2D([], [], color=self.cmap[i], label=f"ROI{i}")
            for i in range(len(self.paths))
        ]
        self.axs.legend(handles=handles, loc="upper right")

        # saving roi figure

    def onselect(self, vertices):
        self.roi_select_event(vertices)
        figure_output = "roi_figure.png"
        plt.savefig(figure_output, dpi=800)

        # Here you can add any further processing of the polygons
        self.object_manager.roi_objects = []
        for idx, path in enumerate(self.paths):
            self.object_manager.add_roi_object(ROIObject(f"ROI{idx}", path))

        # Assuming the object_manager's add_roi_object is meant to handle the completed polygons


@register_class_methods
class GUIManager(Manager):
    def __init__(self, identifier: Identifier, object_manager: ObjectManager):
        self.config = identifier.config
        self.video_file_path = identifier.video_file_path
        self.object_manager = object_manager
        if self.video_file_path is None:
            return
        self.videos = {}

    def add_roi_from_video_selection(self):
        cap = cv2.VideoCapture(self.video_file_path)
        if not cap.isOpened():
            print("Error opening video file.")
            return []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        middle_frame_index = total_frames // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_index)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            print("Failed to retrieve frame.")
            return []
        fig, ax = plt.subplots(1)
        ax.imshow(frame)
        self.selector = ROISelector(ax, self.object_manager)

    def get_serializeable_list_names(self) -> List[str]:
        return []
