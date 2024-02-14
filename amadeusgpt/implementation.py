from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import inspect
from amadeusgpt import utils
from amadeusgpt.gui import select_roi_from_plot, select_roi_from_video
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.style.context("dark_background")
import os
import pickle
import platform
import time
from collections import defaultdict, deque
from enum import IntEnum
from functools import lru_cache
from pathlib import Path

import cv2
import matplotlib.path as mpath
import msgpack
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
from moviepy.video.io.VideoFileClip import VideoFileClip
from PIL import Image
from pycocotools import mask as mask_decoder
from scipy.signal import savgol_filter
from scipy.signal import medfilt
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
from tqdm import tqdm
import streamlit as st
from amadeusgpt.utils import (_plot_ethogram,
    filter_kwargs_for_function, 
    timer_decorator,
    frame_number_to_minute_seconds, 
    get_fps
)
from amadeusgpt.logger import AmadeusLogger

# set file level matplotlib parameters
# I do not know what happens when the main.py also renders plots. Will I need to configure it there?
params = {
    "axes.labelsize": 10,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "text.usetex": False,
    "figure.figsize": [8, 8],
    "font.size": 10,
}
plt.rcParams.update(params)

import random

random.seed(78)

scene_frame_number = 0



class Database:
    """
    A singleton that stores all data. Should be easy to integrate with a Nonsql database
    """

    local_database = defaultdict(dict)
    local_dataset_name = ""

    @classmethod
    def add(cls, class_name, name, val):
        if os.environ.get("streamlit_app", False):
            class_name = class_name + st.session_state.get("example", "")
            if f"database" in st.session_state:
                st.session_state[f"database"][class_name][name] = val
        else:
            class_name = class_name + cls.local_dataset_name
            cls.local_database[class_name][name] = val

    @classmethod
    def get(cls, class_name, name):
        if os.environ.get("streamlit_app", False):
            class_name = class_name + st.session_state.get("example", "")
            if name in st.session_state[f"database"][class_name]:
                return st.session_state[f"database"][class_name][name]
            else:
                return None
        else:
            class_name = class_name + cls.local_dataset_name
            return cls.local_database[class_name][name]

    @classmethod
    def exist(cls, class_name, name):
        if os.environ.get("streamlit_app", False):
            class_name = class_name + st.session_state.get("example", "")
            return (
                class_name in st.session_state[f"database"]
                and name in st.session_state[f"database"][class_name]
            )
        else:
            class_name = class_name + cls.local_dataset_name
            return (
                class_name in cls.local_database
                and name in cls.local_database[class_name]
            )

    @classmethod
    def delete(cls, class_name, name):
        if os.environ.get("streamlit_app", False):
            class_name = class_name + st.session_state.get("example", "")
            del st.session_state[f"database"][class_name][name]
        else:
            class_name = class_name + cls.local_dataset_name
            del cls.local_database[class_name][name]


class Scene:
    @classmethod
    def get_scene_frame(cls, events=None):
        video_file_path = AnimalBehaviorAnalysis.get_video_file_path()
        cap = cv2.VideoCapture(video_file_path)
        if events is None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, scene_frame_number)
            ret, frame = cap.read()
        else:
            # TODO I can write better code than this
            if len(events) <= 1:
                events = Event.flatten_events(events)
                mask = Event.events2onemask(events)
                if np.sum(mask) == 0:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                else:
                    start = np.where(mask)[0][0]
                    end = np.where(mask)[0][-1]
                    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
                    ret, frame = cap.read()

            else:
                frame = []
                for animal_id in events:
                    _events = events[animal_id]
                    mask = Event.events2onemask(_events)
                    if np.sum(mask) == 0:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, _frame = cap.read()
                        frame.append(_frame)
                    else:
                        start = np.where(mask)[0][0]
                        end = np.where(mask)[0][-1]
                        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
                        ret, _frame = cap.read()
                        frame.append(_frame)

        cap.release()
        # cv2.destroyAllWindows()
        return frame


def blockfy(masks):
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


def event_attach(early_event, late_event, continuous=True):
    """
    Attaches a event that temporally follows this event
    The resulted event should be continuous in time
    Examples
    --------
    """
    # two events can overlap but this event must happen first
    assert early_event.start <= late_event.start
    # all the events have the same length that span the whole video
    temp = np.zeros_like(early_event.mask, dtype=bool)
    if continuous:
        # events such as leave or enter requires back to back
        # events that are non exclusive only require early.end > late.start
        if (
            early_event.end + 1 == late_event.start
            or early_event.end > late_event.start
        ):
            # should we allow a small temporal gap for jittering or something?

            temp[early_event.start : late_event.end + 1] = True
    else:
        temp[early_event.start : late_event.end + 1] = True

    return Event(temp)


class AnimalEvent(dict):
    """
    A data structure that is really designed for multiple animals
    Instead of having List[Event] as in single animal, AnimalEvent is
    basically Dict[str, List[Event]] that makes List[Event] under each animal
    Though a a future extension if we do want to include animal_name -> animal_name -> List[Event] is a to-do.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        for animal_name, event_list in self.items():
            event_list.remove_overlapping_events()

    def filter_tensor_by_events(self, data, frame_range=None):
        ret = []
        temp_data = np.ones_like(data) * np.nan
        for animal_name, events in self.items():
            mask = Event.events2onemask(events)
            if frame_range is not None:
                temp_mask = np.zeros_like(mask, dtype=bool)
                temp_mask[frame_range] = True
                mask &= temp_mask
            temp_data[mask] = data[mask]
            ret.append(temp_data)
        ret = np.concatenate(ret, axis=1)
        assert len(data.shape) == len(ret.shape)
        return ret

    def __or__(self, other):
        if isinstance(other, AnimalEvent):
            ret = {}
            for animal_id, events in self.items():
                this_mask = Event.events2onemask(events)
                other_mask = Event.events2onemask(other[animal_id])
                and_mask = this_mask | other_mask
                ret[animal_id] = Event.masks2events(and_mask)
            return AnimalEvent(ret)

    def __and__(self, other):
        if isinstance(other, AnimalEvent):
            ret = {}
            for animal_id, events in self.items():
                this_mask = Event.events2onemask(events)
                other_mask = Event.events2onemask(other[animal_id])
                and_mask = this_mask & other_mask
                ret[animal_id] = Event.masks2events(and_mask)
            return AnimalEvent(ret)

    def __getitem__(self, key):        
        if isinstance(key, int):
            ret = {}
            for animal_name, event_list in self.items():            
                ret[animal_name] = EventList([event_list[key]])
            return AnimalEvent(ret)
        elif isinstance(key, slice):
            ret = {}
            for animal_name, event_list in self.items():            
                ret[animal_name] = EventList(event_list[key.start:key.stop:key.step])
            return AnimalEvent(ret)            

        elif isinstance(key, str):
            return  super().__getitem__(key)

                   
    @property
    def duration(self):
        total_duration = 0
        for animal_name, event_list in self.items():
            for event in event_list:
                total_duration += event.duration
        return total_duration


class Event:
    """
    Methods   
    """

    def __init__(self, mask, object=None):
        self.mask = np.squeeze(mask)
        self.object = object
        if np.sum(mask) == 0:
            # invalid start is very late
            self.start = len(self.mask)
            # invalid end is very early
            self.end = -1
        else:
            self.start = np.where(self.mask)[0][0]
            self.end = np.where(self.mask)[0][-1]
        self._duration = self.end - self.start

    def __len__(self):
        return np.sum(self.mask)

    def __getitem__(self, key):
        assert key in ["start", "end"], f"{key} not supported"
        return getattr(self, key)

    @property
    def duration(self):
        video_file_path = AnimalBehaviorAnalysis.get_video_file_path()
        return round(np.sum(self.mask) / get_fps(video_file_path), 2)

    def __lt__(self, other):
        return self._duration < other._duration

    @classmethod
    def flatten_events(cls, events):
        """
        events can be
        1) animal_name -> list[events]
        2) list[events]
        3) animal_name -> animal_name -> list[events]
        Returns
        -------
        List[events]
        """
        if isinstance(events, list):
            raise ValueError(f"{type(events)} should not call flatten_events")

        elif isinstance(events, AnimalEvent):
            ret = EventList()
            for animal_name, _events in events.items():
                ret.extend(_events)
            return ret
        elif isinstance(events, AnimalAnimalEvent):
            # case for dictionary of AnimalEvent
            ret = EventList()
            for animal_name, _dict in events.items():
                for other_name, _events in _dict.items():
                    ret.extend(_events)
            return ret
        else:
            raise NotImplementedError("Something Wrong")

    @classmethod
    def events2onemask(cls, events):
        """
        Parameters:
        events: List[Event]
        """
        if isinstance(events, dict):
            events = Event.flatten_events(events)
        mask = np.zeros(len(AnimalBehaviorAnalysis.get_keypoints()), dtype=bool)
        for event in events:
            mask |= event.mask
        return mask

    @classmethod
    def summary(cls, events):
        """
        Parameters:
        events: List[Event]
        """
        print("-----")
        for event in events:
            print(f"start: {event.start}, end: {event.end}")
        print("-----")

    @classmethod
    def length(cls, events):
        """
        Examples
        --------
        >>> # number of frame the animal stays in the ROI for first time
        >>> def task_program():
        >>>     behavior_analysis = AnimalBehaviorAnalysis()
        >>>     first_time_in_roi_event = behavior_analysis.animals_object_events('ROI', 'overlap', bodyparts = ['all'])[0]
        >>>     length_in_roi = Event.event_length(first_time_in_roi_event)
        >>>     return length_in_roi
        """
        if isinstance(events, AnimalEvent):
            return [Event.length(event) for event in events.values()]
        if isinstance(events, list):
            _sum = 0
            for event in events:
                _sum += np.sum(event.mask)
            return _sum
        elif isinstance(events, Event):
            return np.sum(events.mask)
        elif isinstance(events, dict):
            value_type = next(iter(events.values()))
            if isinstance(value_type, dict):
                return cls.length(cls.flatten_events(events))
        else:
            raise ValueError(f"{type(events)} not supported")

    @classmethod
    def count_bouts(cls, events, smooth_window=1):
        """
        Examples
        --------
        >>> # Define "running" as a behavior where the animal moves faster than 0.5, count number of bouts running happens with smoothing window 5
        >>> def task_program():
        >>>     behavior_analysis = AnimalBehaviorAnalysis()
        >>>     behavior_analysis.add_task_program("running")
        >>>     animal_speed = behavior_analysis.get_kinematics(['all'], 'speed')
        >>>     average_animal_speed = np.nanmedian(animal_speed, axis = 1)
        >>>     running_events = Event.masks2events(average_animal_speed > 0.5)
        >>>     count_running = Event.count_bouts(running_events, smooth_window = 5)
        >>>     return count_running
        """
        if isinstance(events, AnimalEvent):
            events = Event.flatten_events(events)
        elif isinstance(events, AnimalAnimalEvent):        
            events = Event.flatten_events(events)        
        df = Database.get("AnimalBehaviorAnalysis", "df")
        mask = np.zeros(len(df), dtype=bool)
        for event in events:
            mask |= event.mask
        mask = utils.smooth_boolean_mask(mask, smooth_window)
        res = sum(1 for _ in utils.group_consecutive(np.flatnonzero(mask)))
        return f"Number of bouts is {res}"

    @classmethod
    def event_negate(cls, events, object=None):
        """
        Get events that are negates of those events
        """
        df = Database.get("AnimalBehaviorAnalysis", "df")
        mask = np.zeros(len(df), dtype=bool)
        for event in events:
            mask |= event.mask
        negate_mask = ~mask
        negate_events = Event.masks2events(negate_mask, object=object)

        return negate_events

    @classmethod
    def masks2events(cls, masks, object=None):
        """
        Turn a binary mask to a list of Events
        Returns
        -------
        List(Event)
        """
        blocks = blockfy(masks)
        events = EventList()
        for block in blocks:
            start, end = block
            # containing the tail too
            if end is not None:
                temp = np.zeros_like(masks, dtype=bool)
                temp[start : end + 1] = True
                events.append(Event(masks & temp, object=object))
        # if there is no event. Just return a event with all false
        # I feel this is a ugly fix
        if len(events) == 0:
            events.append(Event(np.zeros_like(masks, dtype=bool), object=object))
        return events

    @classmethod
    def add_simultaneous_events(cls, *events_list):
        """
        Examples
        --------
        >>> get events for animal's nose in the roi and animal's tail_base not in the roi
        >>> def task_program():
        >>>     behavior_analysis = AnimalBehaviorAnalysis()
        >>>     nose_in_roi_events =  behavior_analysis.animals_object_events('ROI', 'overlap', bodyparts = ['nose'])
        >>>     tail_base_in_roi_events = behavior_analysis.animals_object_events('ROI', 'overlap', bodyparts = ['tail_base'])
        >>>     tail_base_not_in_roi_events = behavior_analysis.animals_object_events('ROI', 'overlap', bodyparts = ['tail_base'], negate = True)
        >>>     return Event.add_simultaneous_events(nose_in_roi_events, tail_base_not_in_roi_events)
        """
        if len(events_list) == 1:
            return AnimalEvent(events_list[0])

        animal_names = AnimalBehaviorAnalysis.get_animal_object_names()
        for events in events_list:
            missings = list(set(animal_names) - set(events.keys()))
            for missing in missings:
                animal_names.remove(missing)

        ret = defaultdict(list)
        for animal_name in animal_names:
            ret[animal_name] = events_list[0][animal_name]
            cur = 1
            while cur < len(events_list):
                if animal_name in events_list[cur]:
                    merged_events = cls.join_events(
                        ret[animal_name], events_list[cur][animal_name]
                    )
                    ret[animal_name] = merged_events
                else:
                    ret[animal_name] = EventList()
                cur += 1
        return AnimalEvent(ret)

    @classmethod
    def filter_event_list(cls, events: List[any], smooth_window=5):
        # smoothing events is particular important for meaningful event merging
        mask = Event.events2onemask(events).astype(int)
        from amadeusgpt.utils import smooth_boolean_mask

        mask = smooth_boolean_mask(mask, smooth_window)
        return Event.masks2events(mask)

    @classmethod
    def add_sequential_events(
        cls,
        *events_list,
        min_event_length=0,
        max_event_length=1000000,
        continuous=False,
    ):       
        animal_names = AnimalBehaviorAnalysis.get_animal_object_names()
        for events in events_list:
            missings = list(set(animal_names) - set(events.keys()))
            for missing in missings:
                animal_names.remove(missing)
        ret = defaultdict(EventList)
        # need to flatten this if its a return from animals_social_events
        if isinstance(events_list[0], AnimalAnimalEvent):        
            # making them list so it can be modified
            events_list = list(events_list)
            for i, events in enumerate(events_list):
                for animal_name in events:
                    lst = []
                    for object_name in events[animal_name]:
                        lst.extend(events[animal_name][object_name])
                    events_list[i][animal_name] = lst

        for animal_name in animal_names:     
            ret[animal_name] = events_list[0][animal_name]

            cur = 1
            while cur < len(events_list):
                if animal_name in events_list[cur]:                   
                    ret[animal_name] = cls.attach_events(
                        ret[animal_name], events_list[cur][animal_name], continuous=continuous
                    )
                else:
                    ret[animal_name] = EventList()
                cur += 1
            filtered_events = EventList()
            for event in ret[animal_name]:
                if (
                    Event.length(event) > min_event_length
                    and Event.length(event) < max_event_length
                ):
                    filtered_events.append(event)
            ret[animal_name] = filtered_events

        return AnimalEvent(ret)

    @classmethod
    def join_events(cls, events_A, events_B):
        events_A_mask = cls.events2onemask(events_A)
        events_B_mask = cls.events2onemask(events_B)

        mask = events_A_mask & events_B_mask
        events = cls.masks2events(mask)

        return events

    @classmethod
    def attach_events(cls, events_early, events_late, continuous=True):
        new_events = EventList()     
        for i, event_early in enumerate(events_early):
            for j, event_late in enumerate(events_late):
                # found a match
                if event_late.start > event_early.start:
                    merged_event = event_attach(
                        event_early, event_late, continuous=continuous
                    )
                    if np.sum(merged_event.mask) > 0:
                        new_events.append(merged_event)
                    break
        if len(new_events) == 0:
            df = Database.get("AnimalBehaviorAnalysis", "df")
            return [Event(np.zeros(len(df), dtype=bool))]

        return new_events

    def __str__(self):
        return f"This bout starts at frame {self.start}, ends at frame {self.end}"


class EventList(list):
    def __init__(self, data = None):
        if data is None:
            data = []
        # sort the events based on starting time
        sorted_data = sorted(data, key=lambda x: x.start)
        super().__init__(sorted_data)
        self.remove_overlapping_events()

    def remove_overlapping_events(self):
        # Assuming events are already sorted by start time
        filtered_events = []
        prev_event = None
        for event in self:           
            if prev_event and event.end == prev_event.end:
                # Replace the previous event with the current one if they have the same end time
                filtered_events[-1] = event
            elif not prev_event or event.start >= prev_event.end:
                # Add the event if there is no previous event or it doesn't overlap
                filtered_events.append(event)

            prev_event = event
        
        self.clear()
        self.extend(filtered_events)
       
class AnimalAnimalEvent(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # prune empty list
        for animal_name, animal_events in list(self.items()):
            for other_animal_name, event_list in list(animal_events.items()):
                if len(event_list) == 0:
                    animal_events.pop(other_animal_name)

        for animal_name, animal_events in list(self.items()):
            if len(animal_events) == 0:
                self.pop(animal_name)

    @property
    def duration(self):
        ret = {}
        for animal_name, animal_events in self.items():
            if animal_name not in ret:
                ret[animal_name] = 0
            ret[animal_name] += animal_events.duration
        return ret


class Object:
    def __init__(self, object_name, masks=None, object_path=None, canvas_path=None):
        """
        TODO: instead of using a point, use the true segmentation as reference point
        object_name: str for referencing the object
        segmentation : the mask
        area : the area of the mask in pixels
        bbox : the boundary box of the mask in XYWH format
        predicted_iou : the model's own prediction for the quality of the mask
        point_coords : the sampled input point that generated this mask
        stability_score : an additional measure of mask quality
        crop_box : the crop of the image used to generate this mask in XYWH format

        Attributes
        ----------
        center: x,y the center of the object
        """
        self.object_name = object_name

        # building a polygon path for contain operation
        if AnimalBehaviorAnalysis.get_video_file_path():
            video_frame = Scene.get_scene_frame()
        if masks:
            self.bbox = masks.get("bbox")
            self.area = masks["area"]
            # _seg could be either binary mask or rle string
            _seg = masks.get("segmentation")

            # this is rle format
            if "counts" in _seg:
                self.segmentation = mask_decoder.decode(_seg)
            else:
                self.segmentation = masks.get("segmentation")
            point_coords = np.where(self.segmentation)
            point_coords = zip(point_coords[0], point_coords[1])
            # need to revert x and y for matplotlib plotting
            points = [[p[1], p[0]] for p in point_coords]
            points = np.array(points)
            self.points = points
            self.Path = self.points2Path(points)
            x, y, w, h = self.bbox
            self.x_min, self.y_min, self.x_max, self.y_max = x, y, x + w, y + h
            self.center = np.array([x + w / 2, y + h / 2])

        elif object_path:
            self.Path = object_path
            vertices = self.Path.vertices
            self.x_min = np.nanmin(vertices[:, 0])
            self.x_max = np.nanmax(vertices[:, 0])
            self.y_min = np.nanmin(vertices[:, 1])
            self.y_max = np.nanmax(vertices[:, 1])
            # Calculate the area of the convex hull using the Shoelace formula
            x = vertices[:, 0]
            y = vertices[:, 1]
            self.points = vertices
            self.area = 0.5 * np.abs(
                np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))
            )
            self.center = np.array([np.mean(vertices[:, 0]), np.mean(vertices[:, 1])])
        elif canvas_path is not None:
            points = canvas_path
            self.Path = self.points2Path(points)
            vertices = self.Path.vertices
            self.x_min = np.nanmin(vertices[:, 0])
            self.x_max = np.nanmax(vertices[:, 0])
            self.y_min = np.nanmin(vertices[:, 1])
            self.y_max = np.nanmax(vertices[:, 1])
            # Calculate the area of the convex hull using the Shoelace formula
            x = vertices[:, 0]
            y = vertices[:, 1]
            self.points = vertices
            self.area = 0.5 * np.abs(
                np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))
            )
            self.center = np.array([np.mean(vertices[:, 0]), np.mean(vertices[:, 1])])
        else:
            return ValueError("must give valid initialization to object class")

    def points2Path(self, points):
        path = None
        if len(points) > 2:
            # hull = ConvexHull(points)
            # points are in counter-clockwise order
            # vertices = hull.vertices.astype(np.int64)
            # points are mesh points. We need to convert them to convex hull
            path_data = []
            path_data.append((mpath.Path.MOVETO, points[0]))
            for point in points[1:]:
                path_data.append((mpath.Path.LINETO, point))
            path_data.append((mpath.Path.CLOSEPOLY, points[0]))
            codes, verts = zip(*path_data)
            path = mpath.Path(verts, codes)
        return path

    def get_center(self):
        return self.center

    def get_xmin(self):
        return self.x_min

    def get_xmax(self):
        return self.x_max

    def get_ymin(self):
        return self.y_min

    def get_ymax(self):
        return self.y_max

    def is_valid(self):
        # subclass can overwrite it
        return True

    @classmethod
    def get_animal_object_relationship(cls, animalseq, obj):
        """
        Returns
        """
        # animal_name -> {object_name -> spatial relations}
        ret = {}
        relationship = animalseq.relationships_with_static_object(obj)
        ret[animalseq.object_name] = {obj.object_name: relationship}

        return ret

    @classmethod
    def get_animals_animals_relationship(
        cls, bodyparts_indices, other_animal_bodyparts_indices
    ):
        ret = defaultdict(dict)
        animals = AnimalBehaviorAnalysis.get_animal_objects_across_frames(
            bodyparts_indices
        )

        if set(other_animal_bodyparts_indices) == set(bodyparts_indices):
            # if bodypart indices are same, then we can reuse the animal class
            other_animals = animals
        else:
            other_animals = AnimalBehaviorAnalysis.get_animal_objects_across_frames(
                other_animal_bodyparts_indices
            )

        get_other_animal_names = lambda name: [
            _name
            for _name in AnimalBehaviorAnalysis.get_animal_object_names()
            if _name != name
        ]
        for animal_name, animalseq in animals.items():
            other_animal_names = get_other_animal_names(animal_name)
            for other_animal_name in other_animal_names:
                other_animalseq = other_animals[other_animal_name]
                relationship = animalseq.relationships_with_moving_object(
                    other_animalseq
                )
                ret[animalseq.object_name].update(
                    {other_animalseq.object_name: relationship}
                )
        return ret

    def __getitem__(self, key):
        return getattr(self, key)

    def distance(self, other_object):
        # we use the center of two objects for calculating distance
        return np.linalg.norm(self.center - other_object.center)

    def overlap(self, other_object):
        # if there is a polygon path corresponding this object, use the contain_point
        # otherwise, use the bounding box representation
        if self.Path is not None:
            return self.Path.contains_point(other_object.center)
        else:
            other_object_center = other_object.center
            return (
                other_object_center <= self.x_max
                and other_object_center <= self.y_max
                and other_object_center >= self.x_min
                and other_object_center >= self.y_min
            )

    def to_left(self, other_object):
        # whether the other object is to the left of this object
        return other_object.center[0] <= self.x_min

    def to_right(self, other_object):
        # whether the other object is to the right of this object
        return other_object.center[0] >= self.x_max

    def to_above(self, other_object):
        # whether the other object is to the above of this object
        return other_object.center[1] <= self.y_min

    def to_below(self, other_object):
        # whether the other object is to the below of this object
        return other_object.center[1] >= self.y_max

    def plot(self, ax):
        x, y = zip(*self.points)
        # Plot the polygon
        ax.plot(x, y, "b-")  # 'b-' means blue line


class AnimalSeq:
    """
    Because we support passing bodyparts indices for initializing an AnimalSeq object,
    body center, left, right, above, top are relative to the subset of keypoints.
    Attributes
    ----------
    self._coords: arr potentially subset of keypoints
    self.wholebody: full set of keypoints. This is important for overlap relationship
    """

    def __init__(self, animal_name, coords, wholebody):
        self.object_name = animal_name
        self.wholebody = wholebody
        self._coords = coords
        self._center = None
        self._x_min = None
        self._y_min = None
        self._x_max = None
        self._y_max = None
        self._paths = []

    @property
    def paths(self):
        if not self._paths:
            for ind in range(self._coords.shape[0]):
                self._paths.append(self.get_path(ind))
        return self._paths

    @lru_cache
    def get_path(self, ind):
        xy = self.wholebody[ind]
        xy = np.nan_to_num(xy)
        if np.all(xy == 0):
            return None

        hull = ConvexHull(xy)
        vertices = hull.vertices
        path_data = []
        path_data.append((mpath.Path.MOVETO, xy[vertices[0]]))
        for point in xy[vertices[1:]]:
            path_data.append((mpath.Path.LINETO, point))
        path_data.append((mpath.Path.CLOSEPOLY, xy[vertices[0]]))
        codes, verts = zip(*path_data)
        return mpath.Path(verts, codes)

    @property
    def center(self):
        if self._center is None:
            self._center = np.nanmedian(self._coords, axis=1).squeeze()

        return self._center

    @property
    def x_min(self):
        if self._x_min is None:
            self._x_min = np.nanmin(self._coords[..., 0], axis=1)
        return self._x_min

    @property
    def x_max(self):
        if self._x_max is None:
            self._x_max = np.nanmax(self._coords[..., 0], axis=1)
        return self._x_max

    @property
    def y_min(self):
        if self._y_min is None:
            self._y_min = np.nanmin(self._coords[..., 1], axis=1)
        return self._y_min

    @property
    def y_max(self):
        if self._y_max is None:
            keypoints = AnimalBehaviorAnalysis.get_keypoints()
            self._y_max = np.nanmax(self._coords[..., 1], axis=1)
        return self._y_max

    def relationships_with_static_object(self, other_obj):
        c = other_obj.center

        to_left = self.center[..., 0] <= other_obj.get_xmin()
        to_right = self.center[..., 0] >= other_obj.get_xmax()
        to_below = self.center[..., 1] >= other_obj.get_ymax()
        to_above = self.center[..., 1] <= other_obj.get_ymin()

        distance = np.linalg.norm(self.center - c, axis=1)
        overlap = other_obj.Path.contains_points(self.center)
        orientation = calc_orientation_in_egocentric_animal(
            self, other_obj.get_center()
        )
        ret = {
            "to_left": to_left,
            "to_right": to_right,
            "to_below": to_below,
            "to_above": to_above,
            "distance": distance,
            "overlap": overlap,
            "orientation": orientation,
        }
        return ret

    def get_pairwise_distance(self, arr1, arr2):
        # (n_frame,  n_kpts, 2)
        assert len(arr1.shape) == 3 and len(arr2.shape) == 3
        # pariwise distance (n_frames, n_kpts, n_kpts)
        pairwise_distances = (
            np.ones((arr1.shape[0], arr1.shape[1], arr2.shape[1])) * 100000
        )
        for frame_id in range(arr1.shape[0]):
            pairwise_distances[frame_id] = cdist(arr1[frame_id], arr2[frame_id])

        return pairwise_distances

    def relationships_with_moving_object(self, other_obj):
        """
        Right now other_obj is assumed to be AnimalSeq. We might want to support ObjectSeq at some point
        """
        c = other_obj.center
        to_left = self.x_max <= other_obj.x_min
        to_right = self.x_min >= other_obj.x_max
        to_below = self.y_min >= other_obj.y_max
        to_above = self.y_max <= other_obj.y_min
        distance = np.linalg.norm(self.center - c, axis=1)
        # I have _coords for both this and other because people could want a subset of animal keypoints
        overlap = []
        # we only do nan to num here because doing it in other places give bad looking trajectory
        robust_center = np.nan_to_num(self.center)
        for path_id, other_path in enumerate(other_obj.paths):
            if other_path is None:
                overlap.append(False)
                continue
            overlap.append(other_path.contains_point(np.array(robust_center[path_id])))
        overlap = np.array(overlap)
        angles = calc_angle_between_2d_coordinate_systems(
            MABE.get_cs(self), MABE.get_cs(other_obj)
        )
        mouse_cs = calc_head_cs(self)

        head_cs_inv = []
        mouse_cs_inv = np.full_like(mouse_cs, np.nan)
        valid = np.isclose(np.linalg.det(mouse_cs[:, :2, :2]), 1)
        mouse_cs_inv[valid] = np.linalg.inv(mouse_cs[valid])
        head_cs_inv.append(mouse_cs_inv)
        relative_speed = np.abs(np.diff(np.linalg.norm(self.center - c, axis=-1)))
        relative_speed = np.pad(relative_speed, (0, 1), mode="constant")
        head_angles = calc_angle_in_egocentric_animal(head_cs_inv, c)
        orientation = calc_orientation_in_egocentric_animal(self, other_obj.center)
        closest_distance = np.nanmin(
            self.get_pairwise_distance(self._coords, other_obj._coords), axis=(1, 2)
        )
        return {
            "to_left": to_left,
            "to_right": to_right,
            "to_below": to_below,
            "to_above": to_above,
            "distance": distance,
            "overlap": overlap,
            "relative_angle": angles,
            "relative_head_angle": head_angles,
            "closest_distance": closest_distance,
            "relative_speed": relative_speed,
            "orientation": orientation,
        }

    def form_animal_coordinate_system(self):
        pass


class MABE:
    @classmethod
    def get_cs(cls, animal_seq):
        neck = animal_seq.wholebody[:, 3]
        tailbase = animal_seq.wholebody[:, 9]
        body_axis = neck - tailbase
        body_axis_norm = body_axis / np.linalg.norm(body_axis, axis=1, keepdims=True)
        # Get a normal vector pointing left
        mediolat_axis_norm = body_axis_norm[:, [1, 0]].copy()
        mediolat_axis_norm[:, 0] *= -1
        nrows = len(body_axis_norm)
        mouse_cs = np.zeros((nrows, 3, 3))
        rot = np.stack((body_axis_norm, mediolat_axis_norm), axis=2)
        mouse_cs[:, :2, :2] = rot
        mouse_cs[:, :, 2] = np.c_[
            animal_seq.wholebody[:, 6], np.ones(nrows)
        ]  # center back

        return mouse_cs


class Orientation(IntEnum):
    FRONT = 1
    BACK = 2
    LEFT = 3
    RIGHT = 4


def calc_orientation_in_egocentric_animal(animal_seq, p):
    "Express the 2D points p into the mouse-centric coordinate system."
    mouse_cs = MABE.get_cs(animal_seq)
    mouse_cs_inv = np.full_like(mouse_cs, np.nan)
    valid = np.isclose(np.linalg.det(mouse_cs[:, :2, :2]), 1)
    mouse_cs_inv[valid] = np.linalg.inv(mouse_cs[valid])
    if p.ndim == 2:
        p = np.pad(p, pad_width=((0, 0), (0, 1)), mode="constant", constant_values=1)
        p_in_mouse = np.squeeze(mouse_cs_inv @ p[:, :, None])
    else:
        p_in_mouse = mouse_cs_inv @ [*p, 1]  # object center in mouse coordinate system
    p_in_mouse = p_in_mouse[:, :2]
    theta = np.arctan2(
        p_in_mouse[:, 1], p_in_mouse[:, 0]
    )  # relative angle between the object and the mouse body axis
    theta = np.rad2deg(theta % (2 * np.pi))
    orientation = np.zeros(theta.shape[0])
    np.place(orientation, np.logical_or(theta >= 330, theta <= 30), Orientation.FRONT)
    np.place(orientation, np.logical_and(theta >= 150, theta <= 210), Orientation.BACK)
    np.place(orientation, np.logical_and(theta > 30, theta < 150), Orientation.LEFT)
    np.place(orientation, np.logical_and(theta > 210, theta < 330), Orientation.RIGHT)
    return orientation


def calc_head_cs(animal_seq):
    nose = animal_seq.wholebody[:, 0]
    neck = animal_seq.wholebody[:, 3]
    head_axis = nose - neck
    head_axis_norm = head_axis / np.linalg.norm(head_axis, axis=1, keepdims=True)
    # Get a normal vector pointing left
    mediolat_axis_norm = head_axis_norm[:, [1, 0]].copy()
    mediolat_axis_norm[:, 0] *= -1
    nrows = len(head_axis_norm)
    mouse_cs = np.zeros((nrows, 3, 3))
    rot = np.stack((head_axis_norm, mediolat_axis_norm), axis=2)
    mouse_cs[:, :2, :2] = rot
    mouse_cs[:, :, 2] = np.c_[neck, np.ones(nrows)]
    return mouse_cs


def calc_angle_between_2d_coordinate_systems(cs1, cs2):
    R1 = cs1[:, :2, :2]
    R2 = cs2[:, :2, :2]
    dot = np.einsum("ij, ij -> i", R1[:, 0], R2[:, 0])
    return np.rad2deg(np.arccos(dot))


def calc_angle_in_egocentric_animal(mouse_cs_inv, p):
    "Express the 2D points p into the mouse-centric coordinate system."
    if p.ndim == 2:
        p = np.pad(p, pad_width=((0, 0), (0, 1)), mode="constant", constant_values=1)
        p_in_mouse = np.squeeze(mouse_cs_inv @ p[:, :, None])
    else:
        p_in_mouse = mouse_cs_inv @ [*p, 1]  # object center in mouse coordinate system
    p_in_mouse = p_in_mouse[:, :2]
    theta = np.arctan2(
        p_in_mouse[:, 1], p_in_mouse[:, 0]
    )  # relative angle between the object and the mouse body axis
    theta = np.rad2deg(theta % (2 * np.pi))
    return theta


class Segmentation:
    """
    Base class for segmentation.
    Should support saving the mask to disk and loading it automatically
    This is because model like SAM can take a long time
    """

    def __init__(self, filename=None):
        """
        filename specifies the path to the potential serialized segmentation file
        We make sure that the segmentation files have same formats
        """
        self.filename = filename
        self.pickledata = None
        self.load()

    def load_msgpack(self):
        object_list = {
            0: "barrel",
            1: "cotton",
            2: "food",
            3: "igloo",
            4: "tower",
            5: "tread",
            6: "tunnel",
            7: "water",
        }

        with open(self.filename, "rb") as f:
            unpacker = msgpack.Unpacker(f, raw=False)
            print("loading seg from maushaus file")
            for frame_id, data_at_frame in enumerate(unpacker):
                mask_dict = {}
                for object in data_at_frame:
                    assert frame_id == object["frame_id"]
                    object_name = object_list[object["category_id"]]
                    bbox = object["bbox"]
                    # because maushaus does not have area, I calculate it from bbox
                    x, y, w, h = bbox
                    image_size = object["segmentation"]["size"]
                    # try not to evaluate the string
                    mask_dict[object_name] = {
                        "segmentation": object["segmentation"],
                        "area": w * h,
                        "bbox": bbox,
                    }
                break
            # now let's just use the first frame
            self.pickledata = mask_dict

    def load_pickle(self):
        if os.path.exists(self.filename):
            with open(self.filename, "rb") as f:
                self.pickledata = pickle.load(f)

    def load(self):
        if self.filename is not None:
            if self.filename.endswith("msgpack"):
                self.load_msgpack()
            elif self.filename.endswith("pickle"):
                self.load_pickle()
            else:
                raise ValueError(f"{self.filename} not supported")

    def save_to_pickle(self, data, filename):
        with open(filename, "wb") as f:
            pickle.dump(data, f)


class MausHausSeg(Segmentation):
    def __init__(self, filename=None):
        super().__init__(filename=filename)

    def get_objects(self):
        ret = {}
        if self.pickledata is not None:
            print("building maushaus objects from rle string")
            for object_name, masks in self.pickledata.items():
                ret[object_name] = Object(object_name, masks=masks)
            return ret
        else:
            raise ValueError("We only support loading from MausHaus for now")


class SAM(Segmentation):
    """
    Class that captures the state of objects, supported by Seg everything
    """

    def __init__(self, ckpt_path, model_type, filename=None):
        super().__init__(filename=filename)
        from segment_anything import (
            SamAutomaticMaskGenerator,
            SamPredictor,
            sam_model_registry,
        )

        sam = sam_model_registry[model_type](checkpoint=ckpt_path)
        device = "cpu" if platform.system() == "Darwin" else "cuda"
        sam.to(device)
        self.mask_generator = SamAutomaticMaskGenerator(sam)

    def generate_mask(self, image):
        masks = self.mask_generator.generate(image)
        return masks

    def generate_mask_at_video(self, video_file_path, frame_id):
        cap = cv2.VideoCapture(video_file_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, scene_frame_number)
        ret, frame = cap.read()
        masks = self.generate_mask(frame)
        cap.release()
        # cv2.destroyAllWindows()
        return masks

    def get_objects(self, video_file_path):
        # assuming objects are still
        if self.pickledata is None:
            masks = self.generate_mask_at_video(video_file_path, 0)
            objects = {}
            for object_name, mask in enumerate(masks):
                obj = Object(str(object_name), masks=mask)
                objects[str(object_name)] = obj
            return objects
        else:
            return self.pickledata

 


class AnimalBehaviorAnalysis:
    """
    This class holds methods and objects that are useful for analyzing animal behavior.
    It no longer holds the states of objects directly. Instead, it references to the Database
    singleton object. This is to make the class more stateless and easier to use in a web app.    
    """
    
    # to be deprecated
    task_programs = {}
    # to be deprecated
    task_program_results = {}
    # if a function has a parameter, it assumes the result_buffer has it
    # special dataset flags set to be False
    maushaus_seg_info = False

    @classmethod
    def get_bodypart_index(cls, bodypart):
        return Database.get(cls.__name__, "bodyparts").index(bodypart)

    @classmethod
    @property
    def cache_objects(cls):
        if not Database.exist("AnimalBehaviorAnalysis", "cache_objects"):
            Database.add("AnimalBehaviorAnalysis", "cache_objects", True)
        return Database.get("AnimalBehaviorAnalysis", "cache_objects")

    @classmethod
    def set_cache_objects(cls, val):
        Database.add("AnimalBehaviorAnalysis", "cache_objects", val)

    @classmethod
    @property
    def result_buffer(cls):
        if not Database.exist("AnimalBehaviorAnalysis", "result_buffer"):
            Database.add("AnimalBehaviorAnalysis", "result_buffer", None)
        return Database.get("AnimalBehaviorAnalysis", "result_buffer")

    @classmethod
    def release_cache_objects(cls):
        """
        For web app, switching from one example to the another requires a release of cached objects
        """
        if Database.exist(cls.__name__, "animal_objects"):
            Database.delete(cls.__name__, "animal_objects")
        if Database.exist(cls.__name__, "animals_objects_relations"):
            Database.delete(cls.__name__, "animals_objects_relations")
        if Database.exist(cls.__name__, "roi_objects"):
            Database.delete(cls.__name__, "roi_objects")


    @classmethod
    @property
    def n_individuals(cls):
        return Database.get(cls.__name__, "n_individuals")

    @classmethod
    @property
    def n_kpts(cls):
        return Database.get(cls.__name__, "n_kpts")

    @classmethod
    def get_scene_frame(cls):
        return Scene.get_scene_frame()

    @classmethod
    def set_dataset(cls, dataset):
        Database.add(cls.__name__, "dataset", dataset)

    @classmethod
    def get_dataset(cls):
        return Database.get(cls.__name__, "dataset")

    def _superanimal_inference(
        self, video_file_path, superanimal_name, scale_list, video_adapt
    ):
        import deeplabcut

        progress_obj = st.progress(0)
        deeplabcut.video_inference_superanimal(
            [video_file_path],
            superanimal_name,
            scale_list=scale_list,
            progress_obj=progress_obj,
            video_adapt=True,
            pseudo_threshold=0.5,
        )

    def superanimal_video_inference(
        self,
        superanimal_name="superanimal_topviewmouse",
        scale_list=[],
        video_adapt=False,
    ):
        """
        Examples
        --------
        >>> # extract pose from the video file with superanimal name superanimal_topviewmouse
        >>> def task_program():
        >>>     superanimal_name = "superanimal_topviewmouse"
        >>>     keypoint_file_path = AnimalBehaviorAnalysis.superanimal_video_inference(superanimal_name)
        >>>     return keypoint_file_path
        """

        import glob

        if "streamlit_cloud" in os.environ:
            raise NotImplementedError(
                "Due to resource limitation, we do not support superanimal inference in the app"
            )

        video_file_path = type(self).get_video_file_path()

        self._superanimal_inference(
            video_file_path, superanimal_name, scale_list, video_adapt
        )

        vname = Path(video_file_path).stem
        resultfolder = Path(video_file_path).parent
        # resultfile should be a h5
        # right now let's consider there is only one file
        # in the future we need to consider multiple files
        print("resultfolder", resultfolder)
        resultfile = glob.glob(os.path.join(resultfolder, vname + "DLC*.h5"))[0]
        print("resultfile", resultfile)
        if os.path.exists(resultfile):
            Database.add(type(self).__name__, "keypoint_file_path", resultfile)

        else:
            raise ValueError(f"{resultfile} not exists")

        pose_video_file = resultfile.replace(".h5", "_labeled.mp4")
        Database.add("AnimalBehaviorAnalysis", "pose_video_file", pose_video_file)

        return pose_video_file

    def get_kinematics(self, bodyparts: List[str], kin_type: str):
        """
        Calculate the kinematics of bodyparts names. The type of kinematics is specified
        by type
        Parameters
        ----------
        bodyparts: List[str]
        kin_type: str. Must be one of 'location', 'speed',  'acceleration',
        Returns
        ------
        np.ndarray(shape = (None, None, None), dtype float
        Examples
        --------
        >>> # get the position of nose
        >>> def task_program():
        >>>     behavior_analysis = AnimalBehaviorAnalysis()
        >>>     location = behavior_analysis.get_kinematics(['nose'], 'location')
        >>>     return location
        >>> # add "nose_speed" as a task program that calculates the speed of nose
        >>> def task_program():
        >>>     behavior_analysis = AnimalBehaviorAnalysis()
        >>>     speed = behavior_analysis.get_kinematics(['nose'], 'speed')
        >>>     behavior_analysis.add_task_program("nose_speed")
        >>>     return speed
        >>> # get acceleration of all keypoints
        >>> def task_program():
        >>>     behavior_analysis = AnimalBehaviorAnalysis()
        >>>     acceleration = behavior_analysis.get_kinematics(['all'], 'acceleration')
        >>>     return acceleration
        >>> # get velocity of head_center and body_center
        >>> def task_program():
        >>>     behavior_analysis = AnimalBehaviorAnalysis()
        >>>     selected_bodyparts = ['head_center', 'body_center']
        >>>     acceleration = beahvior_analysis.get_kinematics(selected_bodyparts, 'acceleration')
        >>>     return acceleration
        >>> # number of frames the animal appears in the roi.
        >>> def task_program():
        >>>     behavior_analysis = AnimalBehaviorAnalysis()
        >>>     events = behavior_analysis.animals_object_events('ROI', 'overlap', bodypart = 'all')
        >>>     return behavior_analysis.count_bouts(events)
        >>> # draw the occurrence plot in the roi
        >>> def task_program():
        >>>     behavior_analysis = AnimalBehaviorAnalysis()
        >>>     behavior_analysis.plot_occurence_in_roi()
        """
        assert kin_type in ["location", "velocity", "acceleration", "speed"]
        ret = None
        import dlc2kinematics

        df = Database.get(type(self).__name__, "df")
        n_kpts = Database.get(type(self).__name__, "n_kpts")
        n_individuals = Database.get(type(self).__name__, "n_individuals")

        if kin_type == "velocity":
            ret = dlc2kinematics.compute_velocity(df, bodyparts=bodyparts)

        elif kin_type == "acceleration":
            ret = dlc2kinematics.compute_acceleration(df, bodyparts=bodyparts)

        elif kin_type == "speed":
            ret = dlc2kinematics.compute_speed(df, bodyparts=bodyparts)
        elif kin_type == "location":
            if bodyparts[0] == "all":
                mask = np.ones(df.shape[1], dtype=bool)
            else:
                mask = df.columns.get_level_values("bodyparts").isin(bodyparts)
            ret = df.loc[:, mask]
        else:
            raise ValueError(f"{kin_type} is not supported")
        n_kpts = len(bodyparts) if bodyparts != ["all"] else n_kpts
        ret = ret.to_numpy().reshape(ret.shape[0], n_individuals, n_kpts, -1)[..., :2]
        return ret

    @classmethod
    def create_labeled_video(cls, videoname):
        from moviepy.video.io.bindings import mplfig_to_npimage
        from moviepy.video.io.VideoFileClip import VideoFileClip

        global frame_index
        frame_index = 0

        def draw_keypoints(frame, keypoints):
            # Convert the frame to a numpy array
            # frame = np.array(frame)
            # Loop over the keypoints and draw them on the frame

            global frame_index
            if frame_index == len(keypoints):
                return frame
            keypoints = keypoints[frame_index]
            for animal_id, animal_keypoints in enumerate(keypoints):
                x = int(np.nanmedian(animal_keypoints, axis=0)[0])
                y = int(np.nanmedian(animal_keypoints, axis=0)[1])

                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
                cv2.putText(
                    frame,
                    f"animal{animal_id}",
                    (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )
            # Convert the numpy array back to an image
            frame_index += 1
            return frame

        video = VideoFileClip(Database.get(cls.__name__, "video_file_path"))
        n_individuals = Database.get(cls.__name__, "n_individuals")
        n_kpts = Database.get(cls.__name__, "n_kpts")
        keypoints = cls.get_keypoints()
        keypoints = keypoints.reshape(keypoints.shape[0], n_individuals, n_kpts, -1)[
            ..., :2
        ]

        keypoints_clip = video.fl_image(lambda frame: draw_keypoints(frame, keypoints))
        keypoints_clip.write_videofile(f"{videoname}")

    def event_plot_trajectory(self, data, events, fig, ax, **kwargs):
        if not isinstance(ax, (list, np.ndarray)):
            ax = [ax]
        # if type(self).get_video_file_path() is not None:
        #     frame = Scene.get_scene_frame(events = events)
        #     for i in range(len(ax)):
        #         ax[i].imshow(frame)
        objects = {}
        objects.update(type(self).get_seg_objects())
        objects.update(type(self).get_roi_objects())
        n_individuals = Database.get(type(self).__name__, "n_individuals")
        # animal_names = [f"animal{idx}" for idx in range(n_individuals)]
        animal_names = events.keys()
        # if 'streamlit' not in os.environ:
        # for object_name, object in objects.items():
        #     for animal_id, animal_name in enumerate(animal_names):
        #         obj = objects[object_name]
        #         x, y = obj.center
        #         ax[animal_id].text(x, y, object_name, ha="center", va="center")

        # setting ax[0] here because we only want first animal's view for objects
        type(self).show_seg(type(self).get_seg_objects(), ax=ax[0])

        for animal_id, animal_name in enumerate(animal_names):
            ax[animal_id].set_title(f"{animal_name}")
        scatter = None
        if len(data.shape) == 3:
            data = data.reshape(data.shape[0], 1, data.shape[1], data.shape[2])
        for animal_id, (animal_name, object_events) in enumerate(events.items()):
            for event_id, event in enumerate(object_events):
                line_colors = plt.get_cmap(kwargs.get("cmap", "rainbow"))(
                    np.linspace(0, 1, len(object_events))
                )
                mask = event.mask
                # averaging across bodyparts
                masked_data = np.nanmedian(data, axis=2)[mask]
                # add median filter after animal is represented by center
                k = 5
                if masked_data.shape[0] < k:
                    k = 1
                masked_data = medfilt(masked_data, kernel_size=(k, 1, 1))
                if masked_data.shape[0] == 0:
                    continue
                x, y = masked_data[:, animal_id, 0], masked_data[:, animal_id, 1]
                x = x[x.nonzero()]
                y = y[y.nonzero()]
                if len(x) < 1:
                    continue

                scatter = ax[animal_id].plot(
                    x,
                    y,
                    label=f"event{event_id}",
                    color=line_colors[event_id],
                    alpha=0.5,
                )
                scatter = ax[animal_id].scatter(
                    x[0],
                    y[0],
                    marker="*",
                    s=100,
                    color=line_colors[event_id],
                    alpha=0.5,
                    **kwargs,
                )
                ax[animal_id].scatter(
                    x[-1],
                    y[-1],
                    marker="x",
                    s=100,
                    color=line_colors[event_id],
                    alpha=0.5,
                    **kwargs,
                )
                # not every useful to show legends
                # ax[animal_id].legend(loc='center left', bbox_to_anchor=(1, 0.5))
        # if scatter:
        #     divider = make_axes_locatable(ax[-1])
        #     cax = divider.append_axes("right", size="5%", pad=0.05)
        #     cbar = fig.colorbar(matplotlib.cm.ScalarMappable(cmap=kwargs.get("cmap", "rainbow")), cax = cax)
        #     cbar.set_label("Time")
        return ax

    @timer_decorator
    def plot_trajectory(
        self,
        bodyparts: List[str],
        events=None,
        axes=None,
        fig=None,
        frame_range=None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        bodyparts : List[str]
        events: dict
        Returns
        -------
        None
        Examples
        --------
        >>> # plot trajectory of the animal.
        >>> def task_program():
        >>>     behavior_analysis = AnimalBehaviorAnalysis()
        >>>     behavior_analysis.plot_trajectory(["all"])
        >>>     return
        >>> # plot trajectory of the animal with the events that animal overlaps with object 6
        >>> def task_program():
        >>>     behavior_analysis = AnimalBehaviorAnalysis()
        >>>     overlap_events = behavior_analysis.animals_object_events('6', ['overlap'], bodypart = 'all')
        >>>     behavior_analysis.plot_trajectory(["nose"], events = overlap_events)
        >>>     return
        """
        data = type(self).get_keypoints()

        if frame_range and events:
            raise ValueError(
                "We currently do not support frame selection and behavior capturing at the same time"
            )

        if frame_range is not None:
            data = data[frame_range]
        all_bodyparts = Database.get(type(self).__name__, "bodyparts")
        n_individuals = Database.get(type(self).__name__, "n_individuals")
        if bodyparts == ["all"]:
            indices = np.arange(len(all_bodyparts))
        else:
            for bodypart in bodyparts:
                if bodypart not in all_bodyparts:
                    raise ValueError(
                        f"{bodypart} not defined in the data. Supported bodyparts are {all_bodyparts}"
                    )

            indices = [all_bodyparts.index(b) for b in bodyparts]

        if axes is None:
            n_animals = n_individuals
            if events:
                fig, axes = plt.subplots(ncols=len(events))
            else:
                fig, axes = plt.subplots(ncols=n_animals)

            # plt.tight_layout()
            frame = None
            if Database.exist(type(self).__name__, "video_file_path"):
                frame = Scene.get_scene_frame(events=events)
                if "run_sam" in os.environ:
                    if isinstance(frame, list):
                        seg_objects = AnimalBehaviorAnalysis.get_seg_objects()
                        mask_frame = AnimalBehaviorAnalysis.show_seg(seg_objects)
                        mask_frame = (mask_frame * 255).astype(np.uint8)
                        for i in range(len(frame)):
                            frame[i] = (frame[i]).astype(np.uint8)
                            image1 = Image.fromarray(frame[i], "RGB")
                            image1 = image1.convert("RGBA")
                            image2 = Image.fromarray(mask_frame, mode="RGBA")
                            frame[i] = Image.blend(image1, image2, alpha=0.1)

                    else:
                        seg_objects = AnimalBehaviorAnalysis.get_seg_objects()
                        mask_frame = AnimalBehaviorAnalysis.show_seg(seg_objects)
                        mask_frame = (mask_frame * 255).astype(np.uint8)
                        frame = (frame).astype(np.uint8)
                        image1 = Image.fromarray(frame, "RGB")
                        image1 = image1.convert("RGBA")
                        image2 = Image.fromarray(mask_frame, mode="RGBA")
                        frame = Image.blend(image1, image2, alpha=0.1)
                if isinstance(frame, list):
                    frame = np.stack(frame)
                else:
                    frame = np.array(frame)
                # if 'streamlit_app' not in os.environ and 'run_sam' in os.environ:

                #     for obj_name, obj in seg_objects.items():
                #         x, y = obj.center
                #         cv2.putText(
                #             frame,
                #             obj_name,
                #             (int(x), int(y)),
                #             cv2.FONT_HERSHEY_SIMPLEX,
                #             1,
                #             (0, 0, 255),
                #             2,
                #         )

                # such that it aligns with imshow
                if events:
                    if len(events) > 1:
                        for i in range(len(events)):
                            # Do not need to care about xlim and ylim if images are shown
                            axes[i].invert_yaxis()
                            if frame is not None:
                                axes[i].imshow(frame[i])
                                axes[i].set_xticklabels([])
                                axes[i].set_yticklabels([])
                    else:
                        axes.invert_yaxis()
                        if frame is not None:
                            axes.imshow(frame)
                            axes.set_xticklabels([])
                            axes.set_yticklabels([])

                else:
                    if n_animals > 1:
                        for i in range(n_animals):
                            # Do not need to care about xlim and ylim if images are shown
                            axes[i].invert_yaxis()
                            if frame is not None:
                                axes[i].imshow(frame[i])
                                axes[i].set_xticklabels([])
                                axes[i].set_yticklabels([])
                    else:
                        axes.invert_yaxis()
                        if frame is not None:
                            axes.imshow(frame)
                            axes.set_xticklabels([])
                            axes.set_yticklabels([])
        #### specify default kwargs for plotting

        if "cmap" in kwargs:
            cmap = kwargs["cmap"]
        elif "colormap" in kwargs:
            cmap = kwargs["colormap"]
        else:
            cmap = "rainbow"

        time_colors = plt.get_cmap(cmap)(np.linspace(0, 1, data.shape[0]))

        filtered_kwargs = filter_kwargs_for_function(plt.scatter, kwargs)

        if events is None:
            # if all, lets only plot the center of the mouse
            if not isinstance(axes, (list, np.ndarray)):
                axes = [axes]
            if bodyparts == ["all"]:
                data = np.nanmedian(data, axis=2)
                for i in range(n_individuals):
                    scatter = axes[i].scatter(
                        data[:, i, 0],
                        data[:, i, 1],
                        c=time_colors,
                        label=f"animal{i}",
                        **filtered_kwargs,
                        s=5,
                    )
                divider = make_axes_locatable(axes[-1])
                cax = divider.append_axes("right", size="5%", pad=0.05)
                cbar = fig.colorbar(
                    matplotlib.cm.ScalarMappable(cmap=kwargs.get("cmap", "rainbow")),
                    cax=cax,
                )
                cbar.set_label("Time")
            else:
                for b_id in indices:
                    if len(data.shape) > 3:
                        for i in range(n_individuals):
                            axes[i].set_title(f"animal{i}")
                            scatter = axes[i].scatter(
                                data[:, i, b_id, 0],
                                data[:, i, b_id, 1],
                                c=time_colors,
                                label=all_bodyparts[b_id],
                                **filtered_kwargs,
                            )
                    else:
                        scatter = axes.scatter(
                            data[:, b_id, 0],
                            data[:, b_id, 1],
                            c=time_colors,
                            label=all_bodyparts[b_id],
                            **filtered_kwargs,
                        )
                divider = make_axes_locatable(axes[-1])
                cax = divider.append_axes("right", size="5%", pad=0.05)
                cbar = fig.colorbar(
                    matplotlib.cm.ScalarMappable(cmap=kwargs.get("cmap", "rainbow")),
                    cax=cax,
                )
                cbar.set_label("Time")
        else:
            self.event_plot_trajectory(data, events, fig, axes, **kwargs)

        plt.savefig("trajectory.png", transparent=True, dpi=300)
        text_message = "Here is a trajectory plot using keypoints extracted from DeepLabCut. The keypoints are colorized across time."

        return fig, axes, text_message

    @classmethod
    def mask2distance(cls, locations):
        assert len(locations.shape) == 2
        assert locations.shape[1] == 2
        diff = np.abs(np.diff(locations, axis=0))
        distances = np.linalg.norm(diff, axis=1)
        overall_distance = np.sum(distances)
        return overall_distance

    @classmethod
    @timer_decorator
    def plot_ethogram(cls, events: AnimalEvent):
        n_individuals = Database.get("AnimalBehaviorAnalysis", "n_individuals")
        fig, axes = plt.subplots(
            nrows=len(events),
            figsize=(10, 2),
            squeeze=False,
        )
        plt.subplots_adjust(hspace=0.9)
        axes = axes.reshape(len(events))
        # remove top and right axis for better visualization

        for i, animal_name in enumerate(events):
            axes[i].spines["right"].set_visible(False)
            axes[i].spines["top"].set_visible(False)
            axes[i].set_yticklabels([])
            axes[i].set_ylabel(f"{animal_name}", rotation="horizontal", labelpad=20)

        if isinstance(events, AnimalEvent):
            for animal_id, animal_name in enumerate(events):
                ret = {}
                event_list = events[animal_name]
                mask = np.zeros(len(AnimalBehaviorAnalysis.get_keypoints()), dtype=bool)               
                for event in event_list:
                    mask |= event.mask
                if np.sum(mask) != 0:
                    ret[f"{animal_name}"] = mask
                if len(ret) > 0:
                    _plot_ethogram(ret, axes[animal_id], cls.get_video_file_path())

        plt.savefig("ethogram.png", transparent=True, dpi=300)

        info = "Ethogram plot gives intuitive explanation of when events happen "
        return fig, axes, info

    @classmethod
    def plot_distance_travelled(cls, event_list, object_list):
        """
        Parameters
        ----------
        event_list: List[AnimalEvent]
        object_list: List[str]
        Examples
        --------
        >>> # plot distance travelled in roi0 and roi1
        >>> def task_program():
        >>>     behavior_analysis = AnimalBehaviorAnalysis()
        >>>     in_roi0_events = behavior_analysis.animals_object_events('ROI0', 'overlap', bodyparts = ['all'])
        >>>     in_roi1_events = behavior_analysis.animals_object_events('ROI1', 'overlap', bodyparts = ['all'])
        >>>     distance_travel_plot_info = plot_distance_travelled([in_roi0_events, in_roi1_events], ['ROI0', 'ROI1'])
        >>>     return distance_travel_plot_info
        """

        locations = cls.get_keypoints()
        robust_centers = np.nanmedian(locations, axis=2)
        # return AnimalEvent to List of events
        assert len(event_list) == len(
            object_list
        ), "length of events and list of objects must match"

        distance_travel_per_animal = defaultdict(list)
        animal_names = cls.get_animal_object_names()
        for animal_name in animal_names:
            distance_travel_per_animal[animal_name] = [0] * len(object_list)

        for object_idx, roi_events in enumerate(event_list):
            for animal_idx, animal_name in enumerate(roi_events):
                _events = roi_events[animal_name]
                for event in _events:
                    distance_travel_per_animal[animal_name][
                        object_idx
                    ] += cls.mask2distance(robust_centers[event.mask, animal_idx])

        n_animals = Database.get(cls.__name__, "n_individuals")

        fig, axes = plt.subplots(ncols=n_animals)
        if not isinstance(axes, (list, np.ndarray)):
            axes = [axes]

        colors = ["gray"] * len(object_list)

        for animal_idx, animal_name in enumerate(distance_travel_per_animal):
            axes[animal_idx].bar(
                object_list, distance_travel_per_animal[animal_name], color=colors
            )
            axes[animal_idx].set_xlabel("ROIs")
            axes[animal_idx].set_ylabel("Distance travelled in ROIs (Pixels)")

        plt.savefig("distance_travelled.png", transparent=True, dpi=300)

        return (
            fig,
            axes,
            "distance travelled plot for combined rois or rois respectively",
        )

    def plot_rois(cls):
        roi_objects = cls.get_roi_objects()
        fig, axes = plt.subplots(1, 1, figsize=(12, 8))
        for roi_name, roi_object in roi_objects.items():
            roi_object.plot(axes)
        return fig, axes

    @classmethod
    def set_image_file_path(cls, value):
        # cls.image_file_path = value
        Database.add(cls._name__, "image_file_path", value)

    @classmethod
    def get_image_file_path(cls):
        # return cls.image_file_path
        return Database.get(cls.__name__, "image_file_path")

    @classmethod
    def set_action_label(cls, value):
        # cls.action_label = value
        Database.add(cls.__name__, "action_label", value)

    @classmethod
    def get_pixel_per_1cm(cls, nose_name, tailbase_name):
        # this should be equivalent to 10 cm in physical world
        length_in_pixel = cls.calculate_distance_between_bodyparts(
            nose_name, tailbase_name
        )
        cls.pixel_per_1cm = length_in_pixel / 10
        return cls.pixel_per_1cm

    @classmethod
    def reject_outlier_keypoints(cls, keypoints, threshold_in_stds=2):
        temp = np.ones_like(keypoints) * np.nan
        for i in range(keypoints.shape[0]):
            for j in range(keypoints.shape[1]):
                # Calculate the center of keypoints
                center = np.nanmean(keypoints[i, j], axis=0)

                # Calculate the standard deviation of keypoints
                std_dev = np.nanstd(keypoints[i, j], axis=0)

                # Create a mask of the keypoints that are within the threshold
                mask = np.all(
                    (keypoints[i, j] > (center - threshold_in_stds * std_dev))
                    & (keypoints[i, j] < (center + threshold_in_stds * std_dev)),
                    axis=1,
                )

                # Apply the mask to the keypoints and store them in the filtered_keypoints array
                temp[i, j][mask] = keypoints[i, j][mask]
        return temp

    @classmethod
    def ast_fillna_2d(cls, arr: np.ndarray) -> np.ndarray:
        """
        Fills NaN values in a 4D keypoints array using linear interpolation.

        Parameters:
        arr (np.ndarray): A 4D numpy array of shape (n_frames, n_individuals, n_kpts, n_dims).

        Returns:
        np.ndarray: The 4D array with NaN values filled.
        """
        n_frames, n_individuals, n_kpts, n_dims = arr.shape
        arr_reshaped = arr.reshape(n_frames, -1)
        x = np.arange(n_frames)
        for i in range(arr_reshaped.shape[1]):
            valid_mask = ~np.isnan(arr_reshaped[:, i])
            if np.all(valid_mask):
                continue
            elif np.any(valid_mask):
                # Perform interpolation when there are some valid points
                arr_reshaped[:, i] = np.interp(x, x[valid_mask], arr_reshaped[valid_mask, i])
            else:
                # Handle the case where all values are NaN
                # Replace with a default value or another suitable handling
                arr_reshaped[:, i].fill(0)  # Example: filling with 0

        return arr_reshaped.reshape(n_frames, n_individuals, n_kpts, n_dims)

    @classmethod
    @timer_decorator
    def set_keypoint_file_path(cls, value, overwrite=False):
        # cls.keypoint_file_path = value

        if Database.exist(cls.__name__, "df") and not overwrite:
            # meaning it is cached
            return
        Database.add(cls.__name__, "keypoint_file_path", value)

        df = pd.read_hdf(value)
        bodyparts = list(df.columns.get_level_values("bodyparts").unique())
        scorer = df.columns.get_level_values("scorer")[0]
        Database.add(cls.__name__, "df", df)
        Database.add(cls.__name__, "bodyparts", bodyparts)
        Database.add(cls.__name__, "scorer", scorer)
        # to save time for debug

        if len(df.columns.levels) > 3:
            n_individuals = len(df.columns.levels[1])
        else:
            n_individuals = 1

        Database.add(cls.__name__, "n_individuals", n_individuals)
        n_frames = df.shape[0]
        n_kpts = len(bodyparts)
        if hasattr(cls, "maushaus_seg_info") and cls.maushaus_seg_info:
            # remove environmental keypoints in maushaus
            _df_array = df.to_numpy().reshape((n_frames, n_individuals, n_kpts, -1))[
                ..., :2
            ]
            _df_array = _df_array[:, :, : n_kpts - 4]
            _df_array = savgol_filter(_df_array, 11, 1, axis=0)
            bodyparts = bodyparts[:-4]
            n_kpts = len(bodyparts)
        else:
            _df_array = df.to_numpy().reshape((n_frames, n_individuals, n_kpts, -1))[
                ..., :2
            ]

            _df_array = cls.reject_outlier_keypoints(_df_array)
            # rejecting causes a lot of nans and we need to fill them
            _df_array = cls.ast_fillna_2d(_df_array)
            # not all datasets have this confidence field. BE CAREFUL!
            if (
                df.to_numpy().reshape((n_frames, n_individuals, n_kpts, -1)).shape[-1]
                > 2
            ):
                confidence = df.to_numpy().reshape(
                    (n_frames, n_individuals, n_kpts, -1)
                )[..., -1]
                Database.add(cls.__name__, "confidence", confidence)
        Database.add(cls.__name__, "_df_array", _df_array)
        Database.add(cls.__name__, "n_kpts", n_kpts)

    @classmethod
    def get_acceleration(cls):
        # Calculate differences in keypoints between frames (velocity)
        velocities = np.diff(cls.get_keypoints(), axis=0) / 30
        # Calculate differences in velocities between frames (acceleration)
        accelerations = (
            np.diff(velocities, axis=0) / 30
        )  # divided by frame rate to get acceleration in pixels/second^2
        # Pad accelerations to match the original shape
        accelerations = np.concatenate(
            [np.zeros((2,) + accelerations.shape[1:]), accelerations]
        )
        # Compute the magnitude of the acceleration from the acceleration vectors
        magnitudes = np.sqrt(np.sum(np.square(accelerations), axis=-1, keepdims=True))
        return magnitudes

    @classmethod
    def get_speed(cls):
        # Calculate differences in keypoints between frames (velocity)
        keypoints = cls.get_keypoints()
        velocities = (
            np.diff(keypoints, axis=0) / 30
        )  # divided by frame rate to get speed in pixels/second
        # Pad velocities to match the original shape
        velocities = np.concatenate([np.zeros((1,) + velocities.shape[1:]), velocities])
        # Compute the speed from the velocity
        speeds = np.sqrt(np.sum(np.square(velocities), axis=-1, keepdims=True))
        return speeds

    @classmethod
    def get_dataframe(cls):
        return Database.get(cls.__name__, "df")

    @classmethod
    def get_velocity(cls):
        # Pad velocities to match the original shape of keypoints
        # Adding a zero row at the beginning
        velocities = (
            np.diff(cls.get_keypoints(), axis=0) / 30
        )  # divided by frame rate to get speed in pixels/second
        velocities = np.concatenate(
            [np.zeros((1,) + velocities.shape[1:]), velocities], axis=0
        )

        return velocities

    @classmethod
    def get_keypoints(cls):
        # return cls._df_array
        return Database.get(cls.__name__, "_df_array")

    @classmethod
    def get_bodypart_indices(cls, names):
        ret = []
        bodypart_names = cls.get_bodypart_names()
        for name in names:
            if name not in cls.get_bodypart_names():
                raise ValueError(
                    f"bodypart {name} is not valid. The valid bodyparts are {cls.get_bodypart_names()}"
                )
            ret.append(bodypart_names.index(name))
        return ret

    @classmethod
    def get_animal_centers(cls):
        keypoints = cls.get_keypoints()
        return np.nanmedian(keypoints, axis=2)

    @classmethod
    def get_object_center(cls, object_name):
        objects_dict = cls.get_seg_objects()
        object_center = objects_dict[object_name].get_center()
        return object_center

    @classmethod
    def get_orientation_vector(cls, b1_name, b2_name):
        b1 = cls.get_keypoints()[:, :, cls.get_bodypart_index(b1_name), :]
        b2 = cls.get_keypoints()[:, :, cls.get_bodypart_index(b2_name), :]
        return b1 - b2

    @classmethod
    def get_keypoint_file_path(cls):
        return Database.get(cls.__name__, "keypoint_file_path")

    @classmethod
    def set_video_file_path(cls, value):
        Database.add(cls.__name__, "video_file_path", value)

    @classmethod
    def get_video_file_path(cls):
        if Database.exist(cls.__name__, "video_file_path"):
            return Database.get(cls.__name__, "video_file_path")
        else:
            return None

    @classmethod
    def get_rois_from_video_select(cls):
        if Database.exist(cls.__name__, "roi_objects"):
            return Database.get(cls.__name__, "roi_objects")
        else:
            video_file_path = Database.get(cls.__name__, "video_file_path")
            rois = select_roi_from_video(video_file_path)
            Database.add(cls.__name__, "roi_objects", rois)
            return rois

    @classmethod
    def roi_objects_exist(cls):
        return Database.exist(cls.__name__, "roi_objects")

    @classmethod
    def set_roi_objects(cls, value):
        Database.add(cls.__name__, "roi_objects", value)

    @classmethod
    def get_roi_objects(cls):
        if Database.exist(cls.__name__, "roi_objects"):
            return Database.get(cls.__name__, "roi_objects")
        else:
            if Database.exist(cls.__name__, "roi_pickle_path") and os.path.exists(
                cls.roi_pickle_path
            ):
                with open(cls.roi_pickle_path, "rb") as f:
                    # cls.roi_objects = pickle.load(f)
                    Database.add(cls.__name__, "roi_objects", pickle.load(f))
            else:
                if Database.exist(cls.__name__, "video_file_path"):
                    rois = cls.get_rois_from_video_select()
                elif cls.result_buffer:
                    rois = cls.get_rois_from_plot()
                else:
                    rois = []
                Database.add(
                    cls.__name__,
                    "roi_objects",
                    {
                        f"ROI{i}": Object(f"ROI{i}", object_path=roi)
                        for i, roi in enumerate(rois)
                    },
                )
                if Database.exist(cls.__name__, "roi_pickle_path"):
                    with open(cls.roi_pickle_path, "wb") as f:
                        pickle.dump(Database.get(cls.__name__, "roi_objects"), f)
                        print("roi objects saved to disk")

            return Database.get(cls.__name__, "roi_objects")

    @classmethod
    def get_roi_object_names(cls):
        if not Database.exist(cls.__name__, "roi_objects"):
            return []
        return list(Database.get(cls.__name__, "roi_objects").keys())

    def get_confidence_of_kinematics(self, bodyparts: List[str]):
        """
        Examples
        --------
        >>> get confidence of the nose, same for nose location, nose speed or nose acceleration
        >>> def task_program():
        >>>     behavior_analysis = AnimalBehaviorAnalysis()
        >>>     selected_kpts = ['nose']
        >>>     confidence_nose_location = behavior_analysis.get_confidence_of_kinematics(selected_kpts)
        >>>     return confidence_nose_location
        """
        bodyparts = Database.get(type(self).__name__, "bodyparts")
        indices = [bodyparts.index(b) for b in bodyparts]
        confidence = Database.get(type(self).__name__, "confidence")
        return confidence[:, :, indices]

    @classmethod
    def show_seg(cls, anns, ax=None):
        if len(anns) == 0:
            return
        # anns is a dictionary of objects with {'object_name': Object}
        ### this is only for maushaus
        # maushaus_names = ['water', 'igloo', 'tower', 'cotton', 'tunnel', 'barrel', 'food', 'tread']
        # sorted_anns = [anns[key] for key in maushaus_names]
        ###
        sorted_anns = sorted(anns.values(), key=(lambda x: x["area"]), reverse=True)

        cmap = plt.cm.get_cmap("rainbow", len(sorted_anns))
        if ax:
            ax.set_autoscale_on(False)

        img = np.ones(
            (
                sorted_anns[0]["segmentation"].shape[0],
                sorted_anns[0]["segmentation"].shape[1],
                4,
            )
        )
        img[:, :, 3] = 0

        alpha = 0.2
        for idx, ann in enumerate(sorted_anns):
            m = ann["segmentation"]
            # color_mask = np.random.random((1, 3)).tolist()[0]
            color_mask = cmap(idx)[:3]
            color_mask = np.concatenate([color_mask, [alpha]])

            img[m] = color_mask

        ret = img
        plt.imshow(ret)
        plt.savefig("segmentation.png", dpi=300, transparent=True)
        return ret

    @classmethod
    def get_animal_objects_across_frames(cls, bodyparts_indices):
        # Trying to use AnimalSeq here
        # make sure not to cache animal_objects here because bodyparts_indices can be different
        if (
            Database.exist(cls.__name__, "cache_objects")
            and Database.get(cls.__name__, "cache_objects") is True
            and Database.exist(cls.__name__, "animal_objects")
        ):
            return Database.get(cls.__name__, "animal_objects")
        else:
            animal_names = cls.get_animal_object_names()
            res = {}
            for animal_id, animal_name in enumerate(animal_names):
                data = cls.get_keypoints()[:, animal_id, bodyparts_indices]
                res[animal_name] = AnimalSeq(
                    animal_name, data, cls.get_keypoints()[:, animal_id]
                )
            Database.add(cls.__name__, "animal_objects", res)
            return res

    @classmethod
    def get_animal_object_names(cls):
        n_individuals = Database.get(cls.__name__, "n_individuals")
        return [f"animal{idx}" for idx in range(n_individuals)]

    @classmethod
    def get_animals_objects_relations(
        cls, bodyparts_indices, otheranimal_bodyparts_indices
    ):
        # Make sure not to cache this because bodyparts can be different in each call
        if (
            Database.exist(cls.__name__, "cache_objects")
            and Database.get(cls.__name__, "cache_objects") is True
            and Database.exist(cls.__name__, "animals_objects_relations")
        ):
            return Database.get(cls.__name__, "animals_objects_relations")
        # roi, sam, animals are all objects
        roi_objs = cls.get_roi_objects()
        seg_objs = cls.get_seg_objects()

        start = time.time()
        animal_objs = cls.get_animal_objects_across_frames(bodyparts_indices)
        end = time.time()
        objs = {}
        objs.update(roi_objs)
        objs.update(seg_objs)

        # the key optimization opportunity here is to make following block faster
        # I don't know if we can vectorize the operations below. Maybe not.
        # therefore, it might be wise if we can vectorize the code
        start = time.time()
        # print ('building animal object relation')
        animals_objects_relations = defaultdict(dict)

        for animal_name, animal_object in animal_objs.items():
            # IMPORTANT! NEED TO INITIALIZE FIRST
            animals_objects_relations[animal_name]
            for object_name, object in objs.items():
                animal_object_relations = Object.get_animal_object_relationship(
                    animal_object, object
                )
                animals_objects_relations[animal_name].update(
                    animal_object_relations[animal_name]
                )

        animals_animals_relations = Object.get_animals_animals_relationship(
            bodyparts_indices, otheranimal_bodyparts_indices
        )

        for animal_name in animals_objects_relations:
            animals_objects_relations[animal_name].update(
                animals_animals_relations[animal_name]
            )
        end = time.time()

        Database.add(
            cls.__name__, "animals_objects_relations", animals_objects_relations
        )
        return animals_objects_relations

    @classmethod
    def set_sam_info(cls, ckpt_path=None, model_type=None, pickle_path=None):
        Database.add(
            cls.__name__,
            "sam_info",
            {
                "ckpt_path": ckpt_path,
                "model_type": model_type,
                "pickle_path": pickle_path,
            },
        )

    @classmethod
    def get_sam_info(cls):
        return Database.get(cls.__name__, "sam_info")

    @classmethod
    def get_sam_objects(cls):
        sam_info = cls.get_sam_info()
        ckpt_path = sam_info["ckpt_path"]
        model_type = sam_info["model_type"]
        pickle_path = sam_info["pickle_path"]
        # do not initialize sam to sam time
        video_file_path = Database.get(cls.__name__, "video_file_path")        
        if not os.path.exists(pickle_path) and st.session_state['enable_SAM'] == 'Yes':
            sam = SAM(ckpt_path, model_type, filename=pickle_path)
            Database.add(cls.__name__, "sam", sam)
            Database.add(
                cls.__name__, "sam_objects", sam.get_objects(video_file_path)
            )
            sam.save_to_pickle(sam.get_objects(video_file_path), pickle_path)
            return sam.get_objects(video_file_path)

        else:
            if os.path.exists(pickle_path):                
                with open(pickle_path, "rb") as f:
                    ret = pickle.load(f)
                    return ret
            else:
                return {}

    @classmethod
    def get_maushaus_seg_objects(cls, filename=None):
        if not hasattr(cls, "maushaus_seg_objects"):
            cls.maushaus_seg = MausHausSeg(filename=filename)
            cls.maushaus_seg_objects = cls.maushaus_seg.get_objects()
            frame = None
            if cls.get_video_file_path():
                frame = Scene.get_scene_frame()
            if frame is not None:
                fig, ax = plt.subplots(1, figsize=(8, 4))
                ax.imshow(frame)
                # for here let's just use the first item
                # for obj_name, obj in cls.maushaus_seg_objects.items():
                #     x,y = obj.center
                #     ax.text(x, y, obj_name, ha ='center', va = 'center')
                cls.show_seg(cls.maushaus_seg_objects, ax=ax)

        return cls.maushaus_seg_objects

    @classmethod
    def set_seg_filename(cls, value):
        Database.add(cls.__name__, "seg_filename", value)

    @classmethod
    def set_maushaus_seg_info(cls, value):
        Database.add(cls.__name__, "maushaus_seg_info", value)

    @classmethod
    def get_seg_objects(cls):
        if Database.exist(cls.__name__, "seg_objects"):
            return Database.get(cls.__name__, "seg_objects")
        if Database.exist(cls.__name__, "sam_info"):
            Database.add(cls.__name__, "seg_objects", cls.get_sam_objects())
        elif Database.exist(cls.__name__, "maushaus_seg_info") and Database.get(
            cls.__name__, "maushaus_seg_info"
        ):
            seg_filename = Database.get(cls.__name__, "seg_filename")
            Database.add(
                cls.__name__,
                "seg_objects",
                cls.get_maushaus_seg_objects(filename=seg_filename),
            )
        else:
            Database.add(cls.__name__, "seg_objects", [])
        return Database.get(cls.__name__, "seg_objects")

    @classmethod
    def get_full_object_names(cls):
        objects = cls.get_object_names()
        return [f"object{idx}" for idx in objects]

    @classmethod
    def get_object_names(cls):
        names = cls.get_seg_object_names() + cls.get_roi_object_names()
        return names

    @classmethod
    def get_animal_names(cls):
        return cls.get_animal_object_names()

    @classmethod
    def get_bodypart_names(cls):
        return Database.get(cls.__name__, "bodyparts")

    @classmethod
    def get_seg_object_names(cls):
        return [e for e in cls.get_seg_objects()]

    def enter_object(self, object_name, bodyparts: List[str] = ["all"]) -> List[Event]:
        overlap_object_events = self.animals_object_events(
            object_name, "overlap", bodyparts=bodyparts
        )
        outside_object_events = self.animals_object_events(
            object_name, "overlap", bodyparts=bodyparts, negate=True
        )
        enter_events = Event.add_sequential_events(
            outside_object_events, overlap_object_events, continuous=True
        )
        # I think enter is an instant action
        for animal_name, event_list in enter_events.items():
            for event in event_list:
                start = event.start
                end = event.end
                event.mask[start:end] = False
                event.start = end - 1
                event.end = end

        return enter_events

    def leave_object(self, object_name, bodyparts: List[str] = ["all"]) -> List[Event]:
        overlap_object_events = self.animals_object_events(
            object_name, "overlap", bodyparts=bodyparts
        )
        outside_object_events = self.animals_object_events(
            object_name, "overlap", bodyparts=bodyparts, negate=True
        )
        leave_events = Event.add_sequential_events(
            overlap_object_events, outside_object_events, continuous=True
        )
        return leave_events

    @classmethod
    @property
    @timer_decorator
    def bodypart_pariwise_relation(cls):
        if not Database.exist("AnimalBehaviorAnalysis", "bodypart_pariwise_relation"):
            keypoints = cls.get_keypoints()
            # Calculate pairwise differences along the bodyparts dimension
            diff = keypoints[..., np.newaxis, :, :] - keypoints[..., :, np.newaxis, :]
            # Calculate squared distances
            sq_dist = np.sum(diff**2, axis=-1)
            # Finally, calculate Euclidean distances, shape would be (n_frames, n_animals, n_bodyparts, n_bodyparts)
            distances = np.sqrt(sq_dist)
            ret = {"bodypart_pairwise_distance": distances}
            Database.add("AnimalBehaviorAnalysis", "bodypart_pariwise_relation", ret)

        return Database.get("AnimalBehaviorAnalysis", "bodypart_pariwise_relation")

    @timer_decorator
    def animals_state_events(
        self,
        state_type,
        comparison,
        bodyparts=["all"],
        min_window=0,
        pixels_per_cm=8,
        smooth_window_size=5,
    ):
        animal_names = type(self).get_animal_names()
        animal_kinematics = ["speed", "acceleration"]
        animal_bodypart_relation = ["bodypart_pairwise_distance"]
        assert (
            state_type in animal_kinematics + animal_bodypart_relation
        ), f"{state_type} not in supported {animal_kinematics + animal_bodypart_relation}"

        if state_type in animal_kinematics:
            state_data = self.get_kinematics(bodyparts, state_type)
            if state_data.shape[-1] == 2:
                # calculate the norm
                state_data = np.linalg.norm(state_data, axis=1)
            # always reduce the bodypart dimension. Need to hanld multiple bodypart situation
            state_data = np.nanmedian(state_data, axis=2)

        elif state_type in animal_bodypart_relation:
            state_data = type(self).bodypart_pariwise_relation
            state_data = state_data[state_type]
            bodypart_indices = []
            for bodypart in bodyparts:
                bodypart_indices.append(type(self).get_bodypart_index(bodypart))
            assert (
                len(bodypart_indices) == 2
            ), "Only support pairwise bodypart relation query"
            state_data = state_data[:, :, bodypart_indices[0], bodypart_indices[1]]

        ret = {}

        for animal_id, animal_name in enumerate(animal_names):
            state_data_animal = state_data[:, animal_id].squeeze()
            eval_string = f"state_data_animal{comparison}"
            mask = eval(f"state_data_animal{comparison}")
            ret[animal_name] = Event.masks2events(mask)
        return AnimalEvent(ret)

    @timer_decorator
    def animals_object_events(
        self,
        object_name: str,
        relation_query,
        comparison=None,
        negate=False,
        bodyparts: List[str] = ["all"],
        otheranimal_bodyparts=["all"],
        min_window=0,
        max_window=1000000,
        pixels_per_cm=8,
        smooth_window_size=5,
    ) -> Dict[str, List[Event]]:
        """
        Parameters
        ----------
        object_name : str. Name of the object
        relation_query: str. Must be one of 'to_left', 'to_right', 'to_below', 'to_above', 'overlap', 'distance', 'angle', 'orientation'
        comparison : str, Must be a comparison operator followed by a number like <50, optional
        bodyparts: List[str], optional
        min_window: min length of the event to include
        max_window: max length of the event to include
        negate: bool
           whether to negate the spatial events
        Returns:
        -------
        dict[str, List[Event]]
        Examples
        --------
        >>> # find where the animal is to the left of object 6
        >>> def task_program():
        >>>     behavior_analysis = AnimalBehaviorAnalysis()
        >>>     events = behavior_analysis.animals_object_events('6', 'to_left', bodyparts = ['all'])
        >>>     return events
        >>> # get events where animals are closer than 30 to animal0
        >>> def task_program():
        >>>     behavior_analysis = AnimalBehaviorAnalysis()
        >>>     events = behavior_analysis.animals_object_events('animal0', 'disntace', comparison = '<30',  bodyparts = ['nose'])
        >>>     return events
        """

        # ugly fix for ROI
        if (
            object_name not in AnimalBehaviorAnalysis.get_object_names()  + AnimalBehaviorAnalysis.get_animal_object_names()
        ):
            raise ValueError(f"{object_name} is not defined. You need to define or draw the ROI first. Please do not write code before the user provides the ROI")
        # we only use numbers to represent objects for now
        object_name = object_name.replace("object", "")

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
        else:
            if len(bodyparts) > 1:
                # if it is binary spatial events for multiple bodyparts, decompose the API call to two and do simultaneous event merge.
                events_list = EventList()
                for bodypart in bodyparts:
                    _event_dict = self.animals_object_events(
                        object_name,
                        relation_query,
                        comparison=comparison,
                        negate=negate,
                        bodyparts=[bodypart],
                        otheranimal_bodyparts=otheranimal_bodyparts,
                        min_window=min_window,
                        max_window=max_window,
                        pixels_per_cm=pixels_per_cm,
                        smooth_window_size=smooth_window_size,
                    )
                    events_list.append(_event_dict)
                return Event.add_simultaneous_events(*events_list)

        #  Note I carefully name the variables in this function
        #  animals_objects_relations_list {animal_name -> obj_name -> relation_name -> relation_dict}
        n_kpts = Database.get(type(self).__name__, "n_kpts")
        all_bodyparts = Database.get(type(self).__name__, "bodyparts")
        bodyparts_indices = np.arange(n_kpts)

        if bodyparts != ["all"]:
            bodyparts_indices = []
            for bodypart in bodyparts:
                if bodypart not in all_bodyparts:
                    raise ValueError(f"{bodypart} not defined in {all_bodyparts}")
                bodyparts_indices.append(all_bodyparts.index(bodypart))

        if otheranimal_bodyparts != ["all"]:
            otheranimal_bodyparts_indices = [
                all_bodyparts.index(bodypart) for bodypart in otheranimal_bodyparts
            ]
        else:
            otheranimal_bodyparts_indices = bodyparts_indices

        animals_objects_relations = type(self).get_animals_objects_relations(
            bodyparts_indices, otheranimal_bodyparts_indices
        )

        condition_mask = []
        from collections import defaultdict

        # for downstream tasks to use, I still need to support animals_objects_relation
        #  animals_object_relation_list dict(animal_name ->  [bool])
        # by default we consider all animals and will try to return all animals
        animals_object_relation_masks = {}
        for animal_name, animal_objects_relations in animals_objects_relations.items():
            if object_name == animal_name:
                continue
            if is_numeric_query:
                numeric_quantity = animal_objects_relations[object_name][relation_query]
                # e.g. "250 >= 10"
                relation_string = "numeric_quantity" + comparison
                animal_object_relation = eval(relation_string)

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

                if relation_query in ["relatve_head_angle"]:
                    complement_relation = find_complement_operator(comparison)
                    digits = find_complement_number(comparison)
                    complement_comparison = complement_relation + digits
                    relation_string = "numeric_quantity" + complement_comparison
                    complement_animal_object_relation = eval(relation_string)
                    animal_object_relation |= complement_animal_object_relation
                animals_object_relation_masks[animal_name] = animal_object_relation
            else:
                object_names = AnimalBehaviorAnalysis.get_object_names()
                assert (
                    object_name in animal_objects_relations
                ), f"{object_name} not in available list of objects. Available objects are {object_names}"
                animal_object_relation = animal_objects_relations[object_name][
                    relation_query
                ]
                animals_object_relation_masks[animal_name] = animal_object_relation
        animals_object_events = defaultdict(list)
        # animals_object_events dict(animal_name ->  List[events])
        for animal_key in animals_object_relation_masks:
            condition_mask = animals_object_relation_masks[animal_key]
            condition_mask = np.array(condition_mask)
            events = Event.masks2events(condition_mask, object=object_name)
            if negate:
                events = Event.event_negate(events, object=object_name)
            filtered_events = EventList()
            for event in events:
                if (
                    Event.length(event) > min_window
                    and Event.length(event) < max_window
                ):
                    filtered_events.append(event)
            # smooth the filtered events
            filtered_events = Event.filter_event_list(
                filtered_events, smooth_window=smooth_window_size
            )
            animals_object_events[animal_key] = filtered_events
        ret_event_dict = AnimalEvent(animals_object_events)
        return ret_event_dict

    @timer_decorator
    def animals_social_events(
        self,
        inter_individual_animal_state_query_list=[],
        inter_individual_animal_state_comparison_list=[],
        individual_animal_state_query_list=[],
        individual_animal_state_comparison_list=[],
        bodyparts=["all"],
        otheranimal_bodyparts=["all"],
        min_window=11,
        pixels_per_cm=8,
        smooth_window_size=5,
    ):
        # assert len(inter_individual_animal_state_query_list) == len(inter_individual_animal_state_comparison_list), "length of relation_query_list must be same as comparison_list and they should have one to one mapping"
        # assert len(individual_animal_state_comparison_list) == len(
        #     individual_animal_state_query_list
        # ), "length of individual_animal_state_comparison_list must be same as individual_animal_state_query_list and they should have one to one mapping"

        valid_inter_individual_animal_state_relation_query_list = [
            "to_left",
            "to_right",
            "to_below",
            "to_above",
            "overlap",
            "distance",
            "orientation",
            "relative_speed",
            "closest_distance",
            "relative_angle",
            "relative_head_angle",
        ]
        valid_individual_animal_state_query_list = ["speed", "acceleration"]

        assert set(inter_individual_animal_state_query_list).issubset(
            valid_inter_individual_animal_state_relation_query_list
        ), f"inter_individual_animal_state_query_list must be subset of {valid_inter_individual_animal_state_relation_query_list}. However, you gave {inter_individual_animal_state_query_list}"
        # assert set(individual_animal_state_query_list).issubset(valid_individual_animal_state_query_list), f"individual_animal_state_query_list must be subset of {valid_individual_animal_state_query_list}. However, you gave {individual_animal_state_query_list}"
        # assert len(bodyparts) == 1, "bodyparts must be length of 1"
        # assert len(otheranimal_bodyparts) == 1, "otheranimal_bodyparts must be length of 1"

        animal_names = AnimalBehaviorAnalysis.get_animal_object_names()
        behavior_analysis = AnimalBehaviorAnalysis()
        ret_events = AnimalAnimalEvent()
        ### getting events for animal-state constraints
        animals_state_relations = defaultdict(list)
        for animal_name in animal_names:
            # initialization
            animals_state_relations[animal_name]
            ret_events[animal_name] = AnimalEvent()
        for relation_query, comparison in zip(
            individual_animal_state_query_list, individual_animal_state_comparison_list
        ):
            animals_state_relation = behavior_analysis.animals_state_events(
                relation_query,
                comparison,
                bodyparts=bodyparts,
                pixels_per_cm=pixels_per_cm,
            )

            for animal_id, animal_name in enumerate(animal_names):
                animals_state_relations[animal_name].extend(
                    animals_state_relation[animal_name]
                )

        ### getting events for animal-animal interactions
        for animal_id, animal_name in enumerate(animal_names):
            other_animals = [name for name in animal_names if name != animal_name]
            for other_animal in other_animals:
                animal_animal_relations_events_list = EventList()
                for relation_query, comparison in zip(
                    inter_individual_animal_state_query_list,
                    inter_individual_animal_state_comparison_list,
                ):
                    animals_animal_events = behavior_analysis.animals_object_events(
                        other_animal,
                        relation_query,
                        bodyparts=bodyparts,
                        otheranimal_bodyparts=otheranimal_bodyparts,
                        pixels_per_cm=pixels_per_cm,
                        comparison=comparison,
                    )

                    animal_animal_events = {
                        animal_name: animals_animal_events[animal_name]
                    }
                    animal_animal_relations_events_list.append(animal_animal_events)
                animal_self_other_relations = []
                # this checking is important. By definition, empty list in merging will make everything empty
                if len(animals_state_relations[animal_name]) > 0:
                    animal_self_other_relations.append(
                        {animal_name: animals_state_relations[animal_name]}
                    )
                animal_self_other_relations.extend(animal_animal_relations_events_list)
                merged_events = Event.add_simultaneous_events(
                    *animal_self_other_relations
                )[animal_name]
                ret_events[animal_name][other_animal] = merged_events

        ### does smoothing and min window filter
        for animal_name in ret_events:
            for other_animal in ret_events[animal_name]:
                temp_events = EventList()
                events = ret_events[animal_name][other_animal]
                events = Event.filter_event_list(
                    events, smooth_window=smooth_window_size
                )
                for event in events:
                    if Event.length(event) > min_window:
                        temp_events.append(event)
                ret_events[animal_name][other_animal] = temp_events

        return AnimalAnimalEvent(ret_events)

    def generate_videos_by_events(self, events: List[Event]):
        """
        Examples
        --------
        >>> # find where the animal is to the left of object 6 and create videos for those events
        >>> def task_program():
        >>>     behavior_analysis = AnimalBehaviorAnalysis()
        >>>     left_to_object_events = behavior_analysis.animals_object_events('6', ['to_left'])
        >>>     behavior_analysis.generate_videos_by_events(left_to_object_events)
        >>>     return
        """
        video_file_path = AnimalBehaviorAnalysis.get_video_file_path()
        clip = VideoFileClip(video_file_path)
        fps = clip.fps
        for event_id, event in enumerate(events):
            start_frame = np.where(event.mask)[0][0]
            end_frame = np.where(event.mask)[0][-1]
            start_in_seconds = start_frame / fps
            end_in_seconds = end_frame / fps
            output_file = f"event{event_id}.mp4"
            trimmed_clip = clip.subclip(start_in_seconds, end_in_seconds)
            writer = FFMPEG_VideoWriter(
                output_file, trimmed_clip.size, fps=trimmed_clip.fps
            )
            trimmed_clip.write_videofile(output_file)
            print(f"generated video at {output_file}")
        return None
