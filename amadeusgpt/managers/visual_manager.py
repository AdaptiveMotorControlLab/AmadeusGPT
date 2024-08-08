import os
from typing import Any, Dict, List, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Wedge

from amadeusgpt.analysis_objects.animal import AnimalSeq
from amadeusgpt.analysis_objects.event import Event
from amadeusgpt.analysis_objects.visualization import (EventVisualization,
                                                       GraphVisualization,
                                                       KeypointVisualization,
                                                       SceneVisualization)
from amadeusgpt.behavior_analysis.identifier import Identifier
from amadeusgpt.programs.api_registry import (register_class_methods,
                                              register_core_api)

from .animal_manager import AnimalManager
from .base import Manager
from .object_manager import ObjectManager


def mask2distance(locations):
    assert len(locations.shape) == 2
    assert locations.shape[1] == 2
    diff = np.abs(np.diff(locations, axis=0))
    distances = np.linalg.norm(diff, axis=1)
    overall_distance = np.sum(distances)
    return overall_distance


@register_class_methods
class VisualManager(Manager):
    def __init__(
        self,
        identifier: Identifier,
        animal_manager: AnimalManager,
        object_manager: ObjectManager,
    ):
        super().__init__(identifier.config)
        self.config = identifier.config
        self.video_file_path = identifier.video_file_path
        self.keypoint_file_path = identifier.keypoint_file_path

        self.animal_manager = animal_manager
        self.object_manager = object_manager

    def get_scene_image(self):
        scene_frame_index = self.config["video_info"].get("scene_frame_number", 1)
        if os.path.exists(self.video_file_path):
            cap = cv2.VideoCapture(self.video_file_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, scene_frame_index)
            ret, frame = cap.read()
            return frame
        else:
            return None

    def get_scene_visualization(
        self,
        scene_frame_number: int,
        axs: Optional[plt.Axes] = None,
    ):
        # returns a vis id
        if axs is None:
            fig, axs = plt.subplots(1)

        objects = self.object_manager.get_objects()

        return SceneVisualization(
            axs, objects, self.video_file_path, scene_frame_number
        )

    def get_head_orientation_visualization(self, events=None, render: bool = False):
        ARROW_LENGTH = 20
        CONE_WIDTH = 30
        CONE_LENGTH = 150
        animals = self.animal_manager.get_animals()
        head_cs = [animal.calc_head_cs() for animal in animals]
        fig, axs = plt.subplots(
            self.animal_manager.get_n_individuals(),
            self.animal_manager.get_n_individuals() - 1,
            squeeze=False,
        )
        for sender_idx, sender_animal in enumerate(self.animal_manager.get_animals()):
            other_animals = [
                other_animal
                for other_animal in self.animal_manager.get_animals()
                if other_animal.name != sender_animal.name
            ]
            axs[sender_idx][0].set_ylabel(sender_animal.get_name())
            for receiver_idx, receiver_animal in enumerate(other_animals):

                for event in events:
                    start_frame = event.start
                    break

                scene_vis = self.get_scene_visualization(
                    start_frame, axs=axs[sender_idx][receiver_idx]
                )
                cs = [cs_[start_frame] for cs_ in head_cs]
                sexy_colors = ["fcf6bd", "ffafcc", "a2d2ff"]
                sexy_colors = [f"#{c}" for c in sexy_colors]
                poses = self.animal_manager.get_keypoints()[start_frame]

                receiver_animal_kpts = receiver_animal.get_keypoints()[start_frame]
                receiver_animal_kpts = np.nanmean(receiver_animal_kpts, axis=0)
                # add text on that keypoint position
                axs[sender_idx][receiver_idx].text(
                    receiver_animal_kpts[0],
                    receiver_animal_kpts[1],
                    receiver_animal.get_name(),
                    color="white",
                    fontsize=12,
                )

                scene_vis.draw()
                for pose, cs_, color in zip(poses, cs, sexy_colors):

                    axs[sender_idx][receiver_idx].scatter(*pose.T, s=1, c=color)
                    theta = np.rad2deg(np.arctan2(cs_[1, 0], cs_[0, 0]))
                    origin = tuple(cs_[:2, 2].astype(int))

                    xhat = cs_[:2, 0] * ARROW_LENGTH
                    yhat = -cs_[:2, 1] * ARROW_LENGTH
                    w = Wedge(
                        origin,
                        CONE_LENGTH,
                        theta - CONE_WIDTH // 2,
                        theta + CONE_WIDTH // 2,
                        alpha=0.6,
                        ec="none",
                        fc=color,
                    )
                    axs[sender_idx][receiver_idx].add_artist(w)
                    if True:
                        axs[sender_idx][receiver_idx].arrow(
                            *origin, *xhat, head_width=10, color="r"
                        )
                        axs[sender_idx][receiver_idx].arrow(
                            *origin, *yhat, head_width=10, color="g"
                        )
        if render:
            plt.show()
        return fig, axs

    @register_core_api
    def get_frame_rate(self):
        """
        A function that returns the frame rate of the video.
        """
        cap = cv2.VideoCapture(self.video_file_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        return fps

    # @register_core_api
    def get_keypoint_visualization(
        self,
        render: bool = False,
        bodypart_names: Optional[List[str]] = None,
        fig: Optional[Figure] = None,
        axs: Optional[plt.Axes] = None,
        average_keypoints: bool = True,
        frames: Optional[range] = None,
        events: Optional[List[Event]] = [],
    ) -> None:
        """
        A function that visualizes the keypoints of the animals in the video. The
        plot can optionally use events to sample the time points to visualize the corresponding trajectory.
        """

        if frames is None:
            frames = range(self.animal_manager.get_data_length())

        if axs is None:
            if (
                events is not None
                and len(events) > 0
                and len(events[0].receiver_animal_names) > 0
            ):

                fig, axs = plt.subplots(
                    nrows=self.animal_manager.get_n_individuals(),
                    ncols=self.animal_manager.get_n_individuals() - 1,
                    squeeze=False,
                )
                axs = np.atleast_1d(axs)

                for sender_idx, sender_animal in enumerate(
                    self.animal_manager.get_animals()
                ):
                    other_animals = [
                        other_animal
                        for other_animal in self.animal_manager.get_animals()
                        if other_animal.name != sender_animal.name
                    ]
                    for receiver_idx, receiver_animal in enumerate(other_animals):
                        # change this to the start time of an event
                        start_frame = events[0].start
                        scene_vis = self.get_scene_visualization(
                            start_frame, axs=axs[sender_idx][receiver_idx]
                        )

                        # always feed the full keypoints to visualization
                        full_keypoints = sender_animal.get_keypoints(
                            average_keypoints=False
                        )[frames, :]
                        full_bodypart_names = sender_animal.get_keypoint_names()
                        keypoint_vis = KeypointVisualization(
                            axs[sender_idx][receiver_idx],
                            fig,
                            full_keypoints,
                            sender_animal.get_name(),
                            receiver_animal.get_name(),
                            full_bodypart_names,
                            bodypart_names,
                            self.animal_manager.get_n_individuals(),
                            average_keypoints=average_keypoints,
                            events=events,
                            use_3d=self.config["keypoint_info"].get("use_3d", False),
                        )
                        scene_vis.draw()
                        keypoint_vis.draw()

            else:
                if self.animal_manager.get_n_individuals() == 1:
                    fig, axs = plt.subplots(1)
                    axs = np.atleast_1d(axs)
                else:
                    fig, axs = plt.subplots(self.animal_manager.get_n_individuals())
                    axs = np.atleast_1d(axs)

                for idx, sender_animal in enumerate(self.animal_manager.get_animals()):
                    if not self.config["keypoint_info"].get("use_3d", False):

                        scene_vis = self.get_scene_visualization(
                            self.config["video_info"]["scene_frame_number"],
                            axs=axs[idx],
                        )
                        scene_vis.draw()

                    axs[idx].set_ylabel(sender_animal.get_name())
                    # always feed the full keypoints to visualization
                    full_keypoints = sender_animal.get_keypoints(
                        average_keypoints=False
                    )[frames, :]
                    full_bodypart_names = sender_animal.get_keypoint_names()
                    keypoint_vis = KeypointVisualization(
                        axs[idx],
                        fig,
                        full_keypoints,
                        sender_animal.get_name(),
                        set(),
                        full_bodypart_names,
                        bodypart_names,
                        self.animal_manager.get_n_individuals(),
                        average_keypoints=average_keypoints,
                        events=events,
                    )

                    keypoint_vis.draw()

        if render:
            plt.show()
        return fig, axs

    def get_ethogram_visualization(
        self,
        events: List[Event],
        verbose: bool = False,
        render: bool = False,
        axs: Optional[plt.Axes] = None,
    ):
        """
        By default, it show just displays the ethogram according
        to the output of the task program.
        However, since every event contains the information about the object, receiver animal and sender animal
        We can still make it more informative if we want

        """
        fig, axs = plt.subplots(len(self.animal_manager.get_animals()), 1)

        axs = np.atleast_1d(axs)
        for idx, animal in enumerate(self.animal_manager.get_animals()):
            event_vis = EventVisualization(
                axs[idx], events, animal.get_name(), set(), self.video_file_path
            )
            axs[idx].set_ylabel(animal.get_name())
            event_vis.draw()
        if render:
            plt.show()
        return fig, axs

    # @register_core_api
    def get_animal_animal_visualization(
        self,
        events: List[Event],
        render: bool = False,
        axs: Optional[plt.Axes] = None,
    ) -> None:
        """
        A function that visualizes the ethogram of the animals in the video. It takes
        a list of events and visualizes the events in the video.
        """

        involved_animals = set()
        for event in events:
            involved_animals = involved_animals.union(event.receiver_animal_names)
        if axs is None:
            if len(involved_animals) > 0:
                fig, axs = plt.subplots(
                    self.animal_manager.get_n_individuals(),
                    self.animal_manager.get_n_individuals() - 1,
                    constrained_layout=True,
                    squeeze=False,
                )

                for sender_idx, sender_animal in enumerate(
                    self.animal_manager.get_animals()
                ):
                    other_animals = [
                        other_animal
                        for other_animal in self.animal_manager.get_animals()
                        if other_animal.name != sender_animal.name
                    ]
                    for receiver_idx, receiver_animal in enumerate(other_animals):
                        if receiver_idx == 0:
                            axs[sender_idx][receiver_idx].set_ylabel(
                                sender_animal.get_name()
                            )
                        event_vis = EventVisualization(
                            axs[sender_idx][receiver_idx],
                            events,
                            sender_animal.get_name(),
                            set([receiver_animal.get_name()]),
                            self.video_file_path,
                        )
                        event_vis.draw()
            else:
                if self.animal_manager.get_n_individuals() == 1:
                    fig, axs = plt.subplots(1)
                    axs = np.atleast_1d(axs)
                else:
                    fig, axs = plt.subplots(
                        self.animal_manager.get_n_individuals(), constrained_layout=True
                    )

                for idx, animal in enumerate(self.animal_manager.get_animals()):
                    if idx == 0:
                        axs[idx].set_ylabel(animal.get_name())
                    event_vis = EventVisualization(
                        axs[idx], events, animal.get_name(), set(), self.video_file_path
                    )
                    event_vis.draw()

        if render:
            plt.subplots_adjust()
            plt.show()

    def get_graph_visualization(
        self,
        graph,
        render: bool = False,
        axs: Optional[plt.Axes] = None,
    ):
        fig, axs = plt.subplots(1)
        graph_visualization = GraphVisualization(fig, axs, graph)
        ani = graph_visualization.draw()
        if render:
            plt.show()

    def get_serializeable_list_names(self) -> List[str]:
        return super().get_serializeable_list_names()

    def plot_chessboard_regions(self, frame: np.ndarray):
        objects = self.object_manager.get_grid_objects()

        for obj in objects:
            x_min, y_min, x_max, y_max = obj.x_min, obj.y_min, obj.x_max, obj.y_max
            center = (int((x_min + x_max) / 2), int((y_min + y_max) / 2))
            region_name = obj.get_name()
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(
                frame,
                region_name,
                center,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        return frame

    def sender_visual_cone_on_frame(
        self, sender_animal: AnimalSeq, frame: np.ndarray, current_frame_id: int
    ):
        """
        This function will draw a cone on the frame to show the orientation of the sender animal
        """
        ARROW_LENGTH = 20
        CONE_WIDTH = 30
        CONE_LENGTH = 70

        color = "#fcf6bd"
        cs_ = sender_animal.calc_head_cs()[current_frame_id]

        origin = tuple(cs_[:2, 2].astype(int))

        xhat = cs_[:2, 0] * ARROW_LENGTH
        yhat = -cs_[:2, 1] * ARROW_LENGTH
        arrow_tip = (origin[0] + int(xhat[0]), origin[1] + int(yhat[0]))
        cv2.arrowedLine(
            frame,
            origin,
            arrow_tip,
            cv2.cvtColor(
                np.uint8(
                    [[[int(color.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4)]]]
                ),
                cv2.COLOR_RGB2BGR,
            )[0][0].tolist(),
            2,
            tipLength=0.5,
        )

        # Calculate cone vertices
        cone_direction = np.array([xhat[0], yhat[0]])
        cone_direction_norm = cone_direction / np.linalg.norm(cone_direction)
        left_vertex = origin + np.dot(
            cv2.getRotationMatrix2D((0, 0), CONE_WIDTH / 2, CONE_LENGTH)[:2, :2],
            cone_direction_norm,
        ).astype(int)
        right_vertex = origin + np.dot(
            cv2.getRotationMatrix2D((0, 0), -CONE_WIDTH / 2, CONE_LENGTH)[:2, :2],
            cone_direction_norm,
        ).astype(int)

        # Draw cone
        pts = np.array([origin, tuple(left_vertex), tuple(right_vertex)], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(
            frame,
            [pts],
            isClosed=True,
            color=cv2.cvtColor(
                np.uint8(
                    [[[int(color.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4)]]]
                ),
                cv2.COLOR_RGB2BGR,
            )[0][0].tolist(),
            thickness=2,
        )

        return frame

    def write_video(self, out_folder, video_file_path, out_name, events):
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
            sender_keypoints = self.animal_manager.get_animal_by_name(
                sender_animal_name
            ).get_keypoints()

            sender_keypoints = np.nanmean(sender_keypoints, axis=1)
            sender_speeds = self.animal_manager.get_animal_by_name(
                sender_animal_name
            ).get_speed()

            receiver_keypoints = [
                self.animal_manager.get_animal_by_name(
                    receiver_animal_name
                ).get_keypoints()
                for receiver_animal_name in receiver_animal_names
            ]
            receiver_keypoints = [
                np.nanmean(receiver_keypoints, axis=1)
                for receiver_keypoints in receiver_keypoints
            ]

            data.append(
                {
                    "start_time": time_slices[0],
                    "time_slice": time_slices,
                    "receiver_animal_names": list(receiver_animal_names),
                    "sender_keypoints": sender_keypoints,
                    "sender_animal_name": sender_animal_name,
                    "receiver_keypoints": receiver_keypoints,
                }
            )
            # sort the data by start_time
        data = sorted(data, key=lambda x: x["start_time"])
        total_duration = sum([event.duration_in_seconds for event in events])
        if total_duration < 0.0:
            return

        fourcc = cv2.VideoWriter_fourcc(*"avc1")  # Adjust the codec as needed

        out_videos = []

        for idx, triple in enumerate(data):

            out_video_path = os.path.join(
                out_folder, out_name.replace(".mp4", f"_{idx}.mp4")
            )
            out = cv2.VideoWriter(
                out_video_path,
                fourcc,
                30.0,
                (int(cap.get(3)), int(cap.get(4))),
            )
            out_videos.append(out_video_path)

            time_slice = triple["time_slice"]
            sender_animal_name = triple["sender_animal_name"]
            sender_keypoints = triple["sender_keypoints"]
            receiver_keypoints = triple["receiver_keypoints"]
            cap.set(cv2.CAP_PROP_POS_FRAMES, time_slice[0])
            offset = 0

            while cap.isOpened():
                current_frame = time_slice[0] + offset
                ret, frame = cap.read()
                # frame = self.plot_chessboard_regions(frame)
                if not ret:
                    break

                if time_slice[0] <= current_frame < time_slice[1]:
                    # select the keypoint based on the frame number

                    if self.config["keypoint_info"].get(
                        "head_orientation_keypoints", False
                    ):
                        frame = self.sender_visual_cone_on_frame(
                            self.animal_manager.get_animal_by_name(sender_animal_name),
                            frame,
                            current_frame,
                        )

                    sender_location = sender_keypoints[current_frame]

                    # put the text "sender" and "receiver" on corresponding location
                    sender_speed = np.nanmean(sender_speeds[current_frame])
                    sender_location = sender_location.astype(int)
                    cv2.putText(
                        frame,
                        "sender",
                        (sender_location[0], sender_location[1]),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 0, 0),
                        2,
                        cv2.LINE_AA,
                    )
                    speed_text = f"Sender Speed {sender_speed:.2f} units/frame"
                    speed_text_location = (
                        frame.shape[1] - 400,
                        frame.shape[0] - 50,
                    )  # 200 pixels wide space, 30 pixels from the bottom

                    # put the frame index in the upper right
                    cv2.putText(
                        frame,
                        f"Frame {current_frame}",
                        (frame.shape[1] - 200, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                    cv2.putText(
                        frame,
                        speed_text,
                        speed_text_location,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    for receiver_id, receiver_location in enumerate(receiver_keypoints):
                        receiver_location = receiver_location[current_frame].astype(int)

                        # Calculate the distance between sender and receiver
                        distance = np.linalg.norm(sender_location - receiver_location)
                        # Format the distance to display only 2 decimal places
                        distance_text = (
                            f"Dist(receiver{receiver_id}) {distance:.2f} units"
                        )

                        distance_text_location = (
                            frame.shape[1] - 400,
                            frame.shape[0] - 15 * (receiver_id + 1),
                        )  # 200 pixels wide space, 10 pixels from the bottom

                        cv2.putText(
                            frame,
                            distance_text,
                            distance_text_location,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )
                        cv2.putText(
                            frame,
                            "receiver",
                            (receiver_location[0], receiver_location[1]),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            2,
                            cv2.LINE_AA,
                        )
                    out.write(frame)
                offset += 1
                if current_frame == time_slice[1]:
                    out.release()
                    break

        # Release everything when job is finished
        cap.release()
        cv2.destroyAllWindows()
        return out_videos

    def generate_video_clips_from_events(
        self, out_folder, events: List[Event], behavior_name
    ):
        """
        This function takes a list of events and generates video clips from the events
        1) For the same events, we first group events based on the video
        2) For the same event on the same video, we plot the animal name and the "sender" of the event
        3) Then we write those videos to the disk
        """
        video_file = self.video_file_path

        videoname = video_file.split("/")[-1].replace(".mp4", "").replace(".avi", "")
        video_name = f"{videoname}_{behavior_name}_video.mp4"

        out_videos = self.write_video(out_folder, video_file, video_name, events)
        return out_videos
