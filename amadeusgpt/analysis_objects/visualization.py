from abc import abstractmethod
from typing import Any, Dict, List, Optional, Set

import cv2
import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
from scipy.signal import medfilt

from amadeusgpt.analysis_objects.event import get_fps, get_video_length
from amadeusgpt.analysis_objects.object import Object, ROIObject, SegObject
from amadeusgpt.utils import filter_kwargs_for_function

from .base import AnalysisObject
from .event import BaseEvent, Event, EventGraph


class BaseVisualization(AnalysisObject):
    @abstractmethod
    def display(self):
        # display the plot, either in matplotlib or some other way
        pass


class MatplotlibVisualization(BaseVisualization):
    # annotate ax and fig with the matplotlib types
    def __init__(self, axs: Axes):
        self.axs = axs

    # plot method for MatPlotLibVisualization must return matplotlib fig, ax
    @abstractmethod
    def draw(self, **kwargs) -> None:
        """
        Abstract method that must be implemented by subclasses to
        define how the data is plotted on the matplotlib figure.
        """
        pass

    def display(self):
        """
        Display the matplotlib figure.
        """
        plt.show()

    def save(self, *args, **kwargs):
        """
        Save the matplotlib figure to a file.
        """
        self.figure.savefig(*args, **kwargs)

    def set_title(self, title):
        """
        Set the title of the matplotlib plot.
        """
        self.ax.set_title(title)


class SceneVisualization(MatplotlibVisualization):
    # the scene always includes the objects and the scene frame
    def __init__(
        self,
        axs: Axes,
        objects: List[Object],
        video_file_path: str,
        scene_frame_number: int,
    ):

        super().__init__(axs)
        self.objects = objects
        self.scene_frame = self._get_scene_frame(video_file_path, scene_frame_number)

    def _get_scene_frame(self, video_file_path: str, scene_frame_number: int):
        cap = cv2.VideoCapture(video_file_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, scene_frame_number)
        ret, frame = cap.read()
        cap.release()
        cv2.destroyAllWindows()
        return frame

    def _draw_seg_objects(self):
        scene_frame = self.scene_frame
        seg_objects = [obj for obj in self.objects if isinstance(obj, SegObject)]
        if len(seg_objects) == 0:
            return
        mask_img = np.ones(
            (
                seg_objects[0].segmentation.shape[0],
                seg_objects[0].segmentation.shape[1],
                4,
            )
        )
        mask_img[:, :, 3] = 0
        alpha = 0.2
        for idx, seg_object in enumerate(seg_objects):
            m = seg_object.segmentation
            color_mask = cmap(idx)[:3]
            color_mask = np.concatenate([color_mask, [alpha]])
            mask_img[m] = color_mask

        scene_frame = Image.fromarray(scene_frame, "RGB")
        mask_frame = mask_img.convert("RGBA")
        img = Image.blend(scene_frame, mask_frame, alpha=0.1)
        # imshow is not blocking
        plt.imshow(img)

    def _draw_roi_objects(self):
        roi_objects = [obj for obj in self.objects if isinstance(obj, ROIObject)]
        if len(roi_objects) == 0:
            return
        for roi_object in roi_objects:
            path = roi_object.get_path()
            patch = patches.PathPatch(path, facecolor="none", edgecolor="red", lw=1)
            self.axs.add_patch(patch)

    def draw(self, **kwargs) -> None:
        # for scene visualization, it is just overlapping objects with the scene frame

        self._draw_seg_objects()
        self._draw_roi_objects()
        if self.scene_frame is not None:
            self.axs.imshow(self.scene_frame)


class KeypointVisualization(MatplotlibVisualization):
    def __init__(
        self,
        axs: Axes,
        figure: Figure,
        keypoints: np.ndarray,
        sender_animal_name: str,
        receiver_animal_name: str,
        full_bodypart_names: List[str],
        bodypart_names_to_draw: List[str],
        n_individuals: int,
        average_keypoints: Optional[bool] = True,
        events: Optional[List[BaseEvent]] = None,
    ):
        assert len(keypoints.shape) == 3
        super().__init__(axs)
        self.figure = figure

        if average_keypoints is True:
            if bodypart_names_to_draw is not None:
                self.average_keypoints = False
            else:
                self.average_keypoints = True
        else:
            self.average_keypoints = False

        if self.average_keypoints:
            self.keypoints = np.nanmedian(keypoints, axis=1)
        else:
            self.keypoints = keypoints
            if bodypart_names_to_draw is not None:
                self.bodypart_names_to_draw = bodypart_names_to_draw
            else:
                self.bodypart_names_to_draw = full_bodypart_names

        self.full_bodypart_names = full_bodypart_names

        self.n_individuals = n_individuals
        self.events = events
        self.sender_animal_name = sender_animal_name
        if isinstance(receiver_animal_name, str):
            self.receiver_animal_name = {receiver_animal_name}
        else:
            self.receiver_animal_name = receiver_animal_name

    def draw(self, **kwargs):
        # using matplotlib
        if self.events is not None and len(self.events) == 0:
            return
        if self.events:
            self._event_plot_trajectory(**kwargs)
        else:
            self._plot_trajectory(**kwargs)

    def _plot_trajectory(
        self,
        **kwargs,
    ):
        data = self.keypoints

        self.axs.invert_yaxis()
        self.axs.set_xticklabels([])
        self.axs.set_yticklabels([])
        #### specify default kwargs for plotting

        if "cmap" in kwargs:
            cmap = kwargs["cmap"]
        elif "colormap" in kwargs:
            cmap = kwargs["colormap"]
        else:
            cmap = "rainbow"

        time_colors = plt.get_cmap(cmap)(np.linspace(0, 1, data.shape[0]))

        filtered_kwargs = filter_kwargs_for_function(plt.scatter, kwargs)

        if self.average_keypoints:
            scatter = self.axs.scatter(
                data[:, 0],
                data[:, 1],
                c=time_colors,
                label=self.sender_animal_name,
                **filtered_kwargs,
                s=5,
            )
        else:
            bodypart_indices = [
                self.full_bodypart_names.index(bodypart)
                for bodypart in self.bodypart_names_to_draw
            ]
            for idx in range(len(bodypart_indices)):
                scatter = self.axs.scatter(
                    data[:, bodypart_indices[idx], 0],
                    data[:, bodypart_indices[idx], 1],
                    c=time_colors,
                    label=self.full_bodypart_names[idx],
                    **filtered_kwargs,
                    s=5,
                )
        divider = make_axes_locatable(self.axs)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        cbar = self.figure.colorbar(
            matplotlib.cm.ScalarMappable(cmap=kwargs.get("cmap", "rainbow")),
            cax=cax,
        )
        cbar.set_label("Time")

    def display(self):
        super().display()

    def save(self, *args, **kwargs):
        super().save(*args, **kwargs)

    def _event_plot_trajectory(self, **kwargs):

        data = self.keypoints
        events = self.events
        events = [
            event
            for event in events
            if event.sender_animal_name == self.sender_animal_name
            and self.receiver_animal_name.issubset(event.receiver_animal_names)
        ]

        self.axs.set_title(
            ""
            if len(self.receiver_animal_name) == 0
            else list(self.receiver_animal_name)[0]
        )

        for event_id, event in enumerate(events):

            line_colors = plt.get_cmap(kwargs.get("cmap", "rainbow"))(
                np.linspace(0, 1, len(events))
            )
            mask = event.generate_mask()
            # averaging across bodyparts
            if self.average_keypoints:
                masked_data = data[mask]
            else:
                if self.bodypart_names_to_draw is not None:
                    bodypart_indices = [
                        self.full_bodypart_names.index(bodypart)
                        for bodypart in self.bodypart_names_to_draw
                    ]
                    masked_data = data[mask][:, bodypart_indices]
                    masked_data = np.nanmedian(masked_data, axis=1)

            # add median filter after animal is represented by center
            k = 5
            if masked_data.shape[0] < k:
                k = 1
            masked_data = medfilt(masked_data, kernel_size=(k, 1))
            if masked_data.shape[0] == 0:
                continue

            if not kwargs.get("use_3d", False):
                x, y = masked_data[:, 0], masked_data[:, 1]
                _mask = (x != 0) & (y != 0)

                x = x[_mask]
                y = y[_mask]
                if len(x) < 1:
                    continue

                scatter = self.axs.plot(
                    x,
                    y,
                    label=f"event{event_id}",
                    color=line_colors[event_id],
                    alpha=0.5,
                )
                scatter = self.axs.scatter(
                    x[0],
                    y[0],
                    marker="*",
                    s=100,
                    color=line_colors[event_id],
                    alpha=0.5,
                    **kwargs,
                )
                self.axs.scatter(
                    x[-1],
                    y[-1],
                    marker="x",
                    s=100,
                    color=line_colors[event_id],
                    alpha=0.5,
                    **kwargs,
                )
            else:
                # TODO
                # implement 3d event plot
                pass

        return self.axs

    def display(self):
        super().display()

    def save(self, *args, **kwargs):
        super().save(*args, **kwargs)


def _plot_ethogram(etho_obj: Dict[str, Any], ax, video_path, cmap="rainbow"):
    # etho_obj -> {'animal_name': mask}

    fps = get_fps(video_path)
    video_length = get_video_length(video_path)
    n_rois = len(etho_obj)

    if n_rois == 0:
        return

    cmap = plt.cm.get_cmap(cmap, n_rois)

    colors = cmap(np.linspace(0, 1, n_rois))
    pos = []

    for mask in etho_obj.values():
        pos.append(np.flatnonzero(mask) / fps)

    def format_func(value, tick_number):
        # format the value to minute:second format
        minutes = int(value // 60)
        seconds = int(value % 60)
        ret = f"{minutes:02d}:{seconds:02d}"
        return ret

    ax.set_xlim([0, video_length / fps])
    ax.eventplot(pos, colors=colors)
    ax.xaxis.set_major_formatter(FuncFormatter(format_func))
    ax.set_xlabel("Time (mm:ss)")


class EventVisualization(MatplotlibVisualization):
    def __init__(
        self,
        axs: Axes,
        events: List[BaseEvent],
        sender_animal_name: str,
        receiver_animal_names: Set[str],
        video_file_path: str,
    ):

        super().__init__(axs)
        self.events = events
        self.sender_animal_name = sender_animal_name
        self.receiver_animal_name = receiver_animal_names
        self.video_file_path = video_file_path

    def draw(self):
        events = self.events
        events = [
            event
            for event in events
            if event.sender_animal_name == self.sender_animal_name
            and self.receiver_animal_name.issubset(event.receiver_animal_names)
        ]

        self.axs.spines["right"].set_visible(False)
        self.axs.spines["top"].set_visible(False)
        self.axs.set_yticklabels([])

        title = (
            self.sender_animal_name
            if len(self.receiver_animal_name) == 0
            else list(self.receiver_animal_name)[0]
        )
        self.axs.set_title(title)
        if len(events) == 0:
            video_length = get_video_length(self.video_file_path)
            etho_obj = {self.sender_animal_name: np.zeros(video_length)}
        else:
            etho_obj = {self.sender_animal_name: Event.events2onemask(events)}
        _plot_ethogram(etho_obj, self.axs, self.video_file_path)


class GraphVisualization(MatplotlibVisualization):
    def __init__(self, figure, axs: Axes, graph: EventGraph):
        super().__init__(axs)
        self.figure = figure
        self.axs = axs
        self.graph = graph
        self.nx_graph = nx.DiGraph()
        self.init_nx_graph()

    def init_nx_graph(self):
        # convert my EventGraph to a networkx graph
        cur_node = self.graph.head

        node_count = 0
        prev_name = None
        while cur_node is not None:
            node_count += 1
            parent_name = f"Node_{node_count}"
            self.nx_graph.add_node(parent_name, start=cur_node.start)
            if prev_name is not None:
                self.nx_graph.add_edge(prev_name, parent_name)
            for event_id, event in enumerate(cur_node.children):
                child_name = f"{node_count+1}_event_{event_id}"
                self.nx_graph.add_node(child_name, start=event.start)
                self.nx_graph.add_edge(parent_name, child_name)
            prev_name = parent_name
            cur_node = cur_node.next

    def update(self, time):
        # Filter nodes and edges to display based on the provided time
        self.axs.clear()  # Clear the axes for the new frame
        nodes_to_display = [
            n for n, d in self.nx_graph.nodes(data=True) if d["start"] <= time
        ]
        subG = self.nx_graph.subgraph(nodes_to_display)
        from collections import defaultdict

        # Group and sort nodes by their start time within subG
        nodes_by_time = defaultdict(list)
        for node, data in subG.nodes(data=True):
            nodes_by_time[data["start"]].append(node)
        sorted_times = sorted(nodes_by_time.keys())

        # Calculate positions: Different x for each start time, same y for same start time
        pos = {}
        x_index = 0  # Initialize x position
        current_x_position = 0  # Track x position for nodes with the same start time

        # Assuming sorted_times is a list of unique start times sorted in ascending order
        last_time = None  # Track the last time to adjust x position for nodes with different start times
        x_spacing = 2  # Horizontal spacing factor
        y_spacing = 0.5
        for time in sorted_times:
            nodes_at_time = nodes_by_time[time]
            if time != last_time:
                x_index += 1  # Move to the next x position for a new time
                last_time = time
                current_x_position = (
                    x_index * x_spacing
                )  # Calculate new x position based on time

            # Calculate y positions for nodes with the same start time
            for y_index, node in enumerate(nodes_at_time):
                # Nodes with the same start time have the same x but different y
                # Adjust (-y_index * y_spacing) if you prefer nodes to spread upwards
                pos[node] = (current_x_position, y_index * y_spacing)

        nx.draw(
            subG,
            pos,
            ax=self.axs,
            with_labels=True,
            node_color="lightblue",
            edge_color="gray",
            font_size=10,
        )
        self.axs.set_title(f"Graph at time {time}")

    def draw(self):
        ani = FuncAnimation(self.figure, self.update, frames=range(100), repeat=False)
        return ani

    def display(self):
        plt.show()
