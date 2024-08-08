import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from numpy import ndarray

from amadeusgpt.analysis_objects.animal import AnimalSeq
from amadeusgpt.analysis_objects.event import Event
from amadeusgpt.behavior_analysis.identifier import Identifier
from amadeusgpt.programs.api_registry import (register_class_methods,
                                              register_core_api)

from .base import Manager
from scipy.interpolate import interp1d

def get_orientation_vector(cls, b1_name, b2_name):
    b1 = cls.get_keypoints()[:, :, cls.get_bodypart_index(b1_name), :]
    b2 = cls.get_keypoints()[:, :, cls.get_bodypart_index(b2_name), :]
    return b1 - b2


def interpolate_keypoints(keypoints):
    """
    Interpolate missing (NaN or 0) keypoints in neighboring frames.
    
    Parameters:
    keypoints (numpy array): Array of shape (n_frames, n_individuals, n_keypoints, n_dim).
    
    Returns:
    numpy array: Interpolated keypoints array.
    """
    n_frames, n_individuals, n_keypoints, n_dim = keypoints.shape
    
    # Replace zeros with NaNs for interpolation purposes
    keypoints[keypoints == 0] = np.nan
    
    # Function to interpolate along the frames axis
    def interpolate_along_frames(data):
        for individual in range(n_individuals):
            for keypoint in range(n_keypoints):
                for dim in range(n_dim):
                    # Extract the data for the current dimension
                    values = data[:, individual, keypoint, dim]
                    valid_mask = ~np.isnan(values)
                    
                    if valid_mask.sum() > 1:
                        # Interpolate only if we have more than one valid value
                        interp_fn = interp1d(np.flatnonzero(valid_mask), values[valid_mask], bounds_error=False, fill_value="extrapolate")
                        values[~valid_mask] = interp_fn(np.flatnonzero(~valid_mask))
                        data[:, individual, keypoint, dim] = values
                        
        return data
    
    # Interpolate missing values
    keypoints = interpolate_along_frames(keypoints)
    
    # Replace NaNs back with zeros if needed
    keypoints[np.isnan(keypoints)] = 0
    
    return keypoints


def reject_outlier_keypoints(keypoints: ndarray, threshold_in_stds: int = 2):
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


@register_class_methods
class AnimalManager(Manager):
    def __init__(self, identifier: Identifier):
        """ """
        self.config = identifier.config
        self.video_file_path = identifier.video_file_path
        self.keypoint_file_path = identifier.keypoint_file_path
        self.animals: List[AnimalSeq] = []
        self.full_keypoint_names = []
        self.superanimal_predicted_video = None
        self.superanimal_name = None
        self.init_pose()

    def configure_animal_from_meta(self, meta_info):
        """
        Set the max individuals here
        Set the superanimal model here
        """
        self.max_individuals = int(meta_info["individuals"])
        species = meta_info["species"]
        if species == "topview_mouse":
            self.superanimal_name = "superanimal_topviewmouse_hrnetw32"
        elif species == "sideview_quadruped":
            self.superanimal_name = "superanimal_quadruped_hrnetw32"
        else:
            self.superanimal_name = None

    def init_pose(self):

        if not os.path.exists(self.keypoint_file_path):
            # no need to initialize here
            return

        if self.keypoint_file_path.endswith(".h5"):
            all_keypoints = self._process_keypoint_file_from_h5()
        elif self.keypoint_file_path.endswith(".json"):
            # could be coco format
            all_keypoints = self._process_keypoint_file_from_json()
        for individual_id in range(self.n_individuals):
            animal_name = f"animal_{individual_id}"
            # by default, we initialize all animals with the same keypoints and all the keypoint names

            animalseq = AnimalSeq(
                animal_name, all_keypoints[:, individual_id], self.keypoint_names
            )
            if (
                self.config["keypoint_info"]
                and "body_orientation_keypoints" in self.config["keypoint_info"]
            ):
                animalseq.set_body_orientation_keypoints(
                    self.config["keypoint_info"]["body_orientation_keypoints"]
                )

            if (
                self.config["keypoint_info"]
                and "head_orientation_keypoints" in self.config["keypoint_info"]
            ):
                animalseq.set_head_orientation_keypoints(
                    self.config["keypoint_info"]["head_orientation_keypoints"]
                )

            self.animals.append(animalseq)

    def _process_keypoint_file_from_h5(self) -> ndarray:
        df = pd.read_hdf(self.keypoint_file_path)
        self.full_keypoint_names = list(
            df.columns.get_level_values("bodyparts").unique()
        )
        self.keypoint_names = [k for k in self.full_keypoint_names]
        if len(df.columns.levels) > 3:
            self.n_individuals = len(df.columns.levels[1])
        else:
            self.n_individuals = 1
        self.n_frames = df.shape[0]
        self.n_kpts = len(self.keypoint_names)

        # whether to keep the 3rd dimension in the last axis
        if (
            self.config["keypoint_info"].get("use_3d", False) == True
            or self.config["keypoint_info"].get("include_confidence", False) == True
        ):
            df_array = df.to_numpy().reshape(
                (self.n_frames, self.n_individuals, self.n_kpts, -1)
            )
        else:
            df_array = df.to_numpy().reshape(
                (self.n_frames, self.n_individuals, self.n_kpts, -1)
            )[..., :2]

        df_array = reject_outlier_keypoints(df_array)
        df_array = interpolate_keypoints(df_array)
        return df_array



    def _process_keypoint_file_from_json(self) -> ndarray:
        # default as the mabe predicted keypoints from mmpose-superanimal-topviewmouse
        # {'0': ['bbox':[], 'keypoints':[]}
        with open(self.keypoint_file_path, "r") as f:
            data = json.load(f)

        self.n_individuals = 3
        self.n_frames = len(data)
        self.n_kpts = 27
        self.keypoint_names = [
            "nose",
            "left_ear",
            "right_ear",
            "left_ear_tip",
            "right_ear_tip",
            "left_eye",
            "right_eye",
            "neck",
            "mid_back",
            "mouse_center",
            "mid_backend",
            "mid_backend2",
            "mid_backend3",
            "tail_base",
            "tail1",
            "tail2",
            "tail3",
            "tail4",
            "tail5",
            "left_shoulder",
            "left_midside",
            "left_hip",
            "right_shoulder",
            "right_midside",
            "right_hip",
            "tail_end",
            "head_midpoint",
        ]

        keypoints = (
            np.ones((self.n_frames, self.n_individuals, self.n_kpts, 2)) * np.nan
        )
        for frame_id, frame_data in data.items():
            frame_id = int(frame_id)
            for individual_id, individual_data in enumerate(frame_data):
                if individual_id > self.n_individuals - 1:
                    continue
                keypoints[frame_id, individual_id] = np.array(
                    individual_data["keypoints"]
                )[..., :2]
        return keypoints

    @register_core_api
    def get_data_length(self) -> int:
        """
        Get the number of frames in the data.
        """
        return self.n_frames

    @register_core_api
    def get_animals(self) -> List[AnimalSeq]:
        """
        Get the animals.
        """
        return self.animals

    @register_core_api
    def filter_array_by_events(
        self, array: np.ndarray, animal_name: str, events: List[Event]
    ) -> np.ndarray:
        """
        Filter the array based on the events.
        The array is describing the animal with animal_name. The expected shape (n_frames, n_kpts, n_dims)
        It then returns the array filerted by the masks corresponding to the events.
        If the events is empty, it will return a nan.
        """

        if len(events) == 0:
            return np.ones_like(array) * np.nan

        mask = np.zeros(events[0].data_length, dtype=bool)

        for event in events:
            if event.sender_animal_name != animal_name:
                continue
            mask[event.start : event.end + 1] = 1

        return array[mask]

    @register_core_api
    def get_animal_names(self) -> List[str]:
        """
        Get the names of all the animals.
        """
        return [animal.get_name() for animal in self.animals]

    def get_animal_by_name(self, name: str) -> AnimalSeq:
        animal_names = self.get_animal_names()
        index = animal_names.index(name)
        return self.animals[index]

    @register_core_api
    def get_keypoints(self) -> ndarray:
        """
        Get the keypoints of animals. The keypoints are of shape  (n_frames, n_individuals, n_kpts, n_dims)
        n_dims is 2 (x,y) for 2D keypoints and 3 (x,y,z) for 3D keypoints.
        Do not forget the n_individuals dimension. If there is only one animal, the n_individuals dimension is 1.
        Optionally, you can pass a list of events to filter the keypoints based on the events.
        """

        if os.path.exists(self.video_file_path) and not os.path.exists(
            self.keypoint_file_path
        ):

            if self.superanimal_name is None:
                raise ValueError(
                    "Couldn't determine the species of the animal from the image. Change the scene index"
                )

            # only import here because people who choose the minimal installation might not have deeplabcut
            import deeplabcut
            from deeplabcut.modelzoo.video_inference import \
                video_inference_superanimal

            video_suffix = Path(self.video_file_path).suffix

            self.keypoint_file_path = self.video_file_path.replace(
                video_suffix, "_" + self.superanimal_name + ".h5"
            )
            self.superanimal_predicted_video = self.keypoint_file_path.replace(
                ".h5", "_labeled.mp4"
            )

            if not os.path.exists(self.keypoint_file_path):
                print(f"going to inference video with {self.superanimal_name}")
                video_inference_superanimal(
                    videos=[self.video_file_path],
                    superanimal_name=self.superanimal_name,
                    max_individuals=self.max_individuals,
                    video_adapt=False,
                )

            if os.path.exists(self.keypoint_file_path):
                self.init_pose()

        ret = np.stack([animal.get_keypoints() for animal in self.animals], axis=1)
        return ret

    @register_core_api
    def get_speed(
        self,
    ) -> ndarray:
        """
        Get the speed of all animals. The shape is  (n_frames, n_individuals, n_kpts, 1) # 1 is the scalar speed
        The speed is an unsigned scalar value. speed larger than 0 means moving.
        """
        return np.stack([animal.get_speed() for animal in self.animals], axis=1)

    @register_core_api
    def get_velocity(self) -> ndarray:
        """
        Get the velocity. The shape is (n_frames, n_individuals, n_kpts, n_dim) n_dim is 2 or 3
        The velocity is a vector.
        """
        return np.stack([animal.get_velocity() for animal in self.animals], axis=1)

    @register_core_api
    def get_acceleration_mag(self) -> ndarray:
        """
        Get the magnitude of acceleration. The shape is of shape  (n_frames, n_individuals, 1)
        The acceleration is a vector.
        """
        return np.stack(
            [animal.get_acceleration_mag() for animal in self.animals], axis=1
        )

    @register_core_api
    def get_n_individuals(self) -> int:
        """
        Get the number of animals in the data.
        """
        return self.n_individuals

    @register_core_api
    def get_n_kpts(self) -> int:
        """
        Get the number of keypoints in the data.
        """
        return self.n_kpts

    @register_core_api
    def get_keypoint_names(self) -> List[str]:
        """
        Get the names of the bodyparts. This is used to index the keypoints for a specific bodypart.
        """
        # this is to initialize
        self.get_keypoints()

        return self.full_keypoint_names

    def query_animal_states(self, animal_name: str, query: str) -> np.ndarray | None:
        for animal in self.animals:
            if animal.get_name() == animal_name:
                return animal.query_states(query)

    def update_roi_keypoint_by_names(self, roi_keypoint_names: List[str]) -> None:
        for animal in self.animals:
            animal.update_roi_keypoint_by_names(roi_keypoint_names)

    def restore_roi_keypoint(self) -> None:
        for animal in self.animals:
            animal.keypoints = animal.whole_body

    def get_serializeable_list_names(self) -> List[str]:
        return ["animals"]
