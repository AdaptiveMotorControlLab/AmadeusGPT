import os
import pickle
from typing import Any, Dict, List

import cv2
import numpy as np

from amadeusgpt.analysis_objects.object import GridObject, Object, ROIObject
from amadeusgpt.behavior_analysis.identifier import Identifier
from amadeusgpt.config import Config
from amadeusgpt.managers.animal_manager import AnimalManager
from amadeusgpt.managers.base import Manager
from amadeusgpt.programs.api_registry import (register_class_methods,
                                              register_core_api)

np.set_printoptions(suppress=True)


@register_class_methods
class ObjectManager(Manager):
    def __init__(
        self,
        identifier: Identifier,
        animal_manager: AnimalManager,
    ):
        self.config = identifier.config
        self.video_file_path = identifier.video_file_path
        self.animal_manager = animal_manager
        self.roi_objects = []
        self.seg_objects = []

        self.load_from_disk = self.config["object_info"].get(
            "load_objects_from_disk", False
        )

        if self.load_from_disk:
            self.load_objects_from_disk()
        else:
            self.init()

        ##### Fields used for grids
        # grid_name -> {x,y,w,h}
        self.grids = {}
        # grid_name -> GridObject
        self.grid_objects = []
        # e.g., [['A1,A2], ['B1,'B2]]
        self.grid_labels = []
        self.animal2labelarray = {}
        self.occupation_heatmap = {}
        #####
        # let's not use grid objects for now
        if self.config["object_info"].get("use_grid_objects", False):
            if os.path.exists(self.video_file_path):
                self.create_grids()
                self.create_grid_objects()
        #     self.create_grid_labels()

    def summary(self):
        print("roi_objects: ", self.get_roi_object_names())
        print("seg_objects: ", self.get_seg_object_names())

    def get_seg_object_names(self) -> List[str]:
        return [obj.name for obj in self.seg_objects]

    def load_objects_from_disk(self):
        pass

    def get_roi_object_names(self) -> List[str]:

        return [obj.name for obj in self.roi_objects]

    def get_roi_objects(self) -> List[Object]:
        return self.roi_objects

    def filter_duplicates(self, roi_objects):
        """
        Certain GUI like streamlit canvas keeps adding the same object multiple times
        """
        seen = set()
        unique_set = []
        for roi_object in roi_objects:
            if roi_object.name not in seen:
                seen.add(roi_object.name)
                unique_set.append(roi_object)
        return unique_set

    def add_roi_object(self, object) -> None:
        # the user can add an object to the roi_objects
        self.roi_objects.append(object)
        self.roi_objects = self.filter_duplicates(self.roi_objects)

    def save_roi_objects(self, path: str) -> None:
        roi_obects = self.get_roi_objects()
        data = {}
        for obj in roi_obects:
            data[obj.name] = {"Path": obj.Path}
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load_roi_objects(self, path: str) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        for name, path in data.items():
            self.add_roi_object(ROIObject(name, path["Path"]))

    def get_seg_objects(self) -> list[Object]:
        return self.seg_objects

    def add_sam_object(self, object) -> None:
        sam_info = self.config["sam_info"]
        ckpt_path = sam_info["ckpt_path"]
        model_type = sam_info["model_type"]
        pickle_path = sam_info["pickle_path"]
        sam = SAM(ckpt_path, model_type, filename=pickle_path)
        sam.save_to_pickle(sam.get_objects(self.video_file_path), pickle_path)

    def get_objects(self) -> List[Object]:
        return self.roi_objects + self.seg_objects + self.grid_objects

    @register_core_api
    def get_object_names(self) -> List[str]:
        """
        Returns the names of all objects in the scene.
        """
        return (
            self.get_roi_object_names()
            + self.get_seg_object_names()
            + self.get_grid_object_names()
        )

    def create_grids(self):
        """
        Creates a grids-like structure based on the image dimensions.

        Parameters:
        - image_width: The width of the image in pixels.
        - image_height: The height of the image in pixels.
        - square_size: The size of each square in the grids. Default is 50 pixels.

        Returns:
        - A dictionary representing the grids with unique names for each square.
        """

        cap = cv2.VideoCapture(self.video_file_path)
        ret, frame = cap.read()
        self.image_height, self.image_width = frame.shape[:2]
        cap.release()
        self.square_size = self.image_height // 4
        self.squares_on_height = self.image_height // self.square_size
        self.squares_on_width = self.image_width // self.square_size
        self.grid_labels = [
            [f"{chr(65+i)}{j+1}" for j in range(self.squares_on_width)]
            for i in range(self.squares_on_height)
        ]

        for i in range(self.squares_on_height):
            for j in range(self.squares_on_width):
                square_name = f"{chr(65+i)}{j+1}"  # Naming squares as A1, A2, ..., etc.
                self.grids[square_name] = {
                    "x": j * self.square_size,
                    "y": i * self.square_size,
                    "w": self.square_size,
                    "h": self.square_size,
                }

    def create_grid_objects(self):
        for grid_name in self.grids:
            grid = self.grids[grid_name]
            object = GridObject(grid_name, grid)
            self.add_grid_object(object)

    def add_grid_object(self, obj: GridObject) -> None:
        self.grid_objects.append(obj)

    def get_grid_objects(self) -> List[GridObject]:
        return self.grid_objects

    def get_grid_object_names(self) -> List[str]:
        return [obj.name for obj in self.grid_objects]

    def create_grid_labels(self):
        # there is a more efficient way to get animal-chessboard region relationships

        img_height = self.image_height
        img_width = self.image_width
        # [[A1 A1],
        #  [B1,B2]]
        grid_labels = self.grid_labels

        def map_to_grid(center):
            # Normalize keypoint coordinates to [0, 1] range based on image dimensions
            x, y = center[0], center[1]
            norm_x = x / img_width
            norm_y = y / img_height
            # Scale normalized coordinates to grid dimensions
            col = int(norm_x * self.squares_on_width)
            row = int(norm_y * self.squares_on_height)
            # Ensure the indices are within bounds
            col = min(col, self.squares_on_width - 1)
            row = min(row, self.squares_on_height - 1)
            # Calculate the grid label
            return grid_labels[row][col]

        for animal in self.animal_manager.get_animals():
            # n_frames, 2
            animal_center = animal.center
            # apply map_to_grid to every frame on animal_center using map
            array = list(map(map_to_grid, animal_center))
            self.animal2labelarray[animal.name] = np.array(array)

    def init(self):
        # run sam inference
        pass

    def get_occupation_heatmap(self):
        """

        # can be either heatmap per animal or heatmap as the mean of all animals
        # the heatmap indicates how many frames the animal fall in that region
        Occupation heatmap
           A1 A2 A3 A4
        B1 10 0  0  0
        B2 0  0  0  0
        B3 0  0  0  0
        B4 0  0  0  0
        Temporal heatmap
        """
        if len(self.occupation_heatmap) > 0:
            return self.occupation_heatmap
        self.occupation_heatmap = {}

        for animal in self.animal_manager.get_animals():
            # grid_labels is a [n_frames, 1] array where each element is the grid name of the region the animal is in

            labelarray = self.animal2labelarray[animal.name]
            # create a heatmap
            self.occupation_heatmap[animal.name] = np.zeros(
                (self.squares_on_height, self.squares_on_width), dtype=np.float32
            )
            for grid_name in labelarray:
                row = ord(grid_name[0]) - 65
                col = int(grid_name[1:]) - 1
                self.occupation_heatmap[animal.name][row, col] += 1
        for animal_name in self.occupation_heatmap:
            # normalize the heatmap to sum to 1
            # turn it to percentage
            self.occupation_heatmap[animal_name] = np.round(
                (
                    self.occupation_heatmap[animal_name]
                    / self.occupation_heatmap[animal_name].sum()
                ),
                2,
            )

        return self.occupation_heatmap

    def get_serializeable_list_names(self) -> List[str]:
        return ["roi_objects", "seg_objects"]


if __name__ == "__main__":
    config = Config("../../tests/nishant.yaml")
    model_manager = ModelManager(config)
    object_manager = ObjectManager(
        config, model_manager, AnimalManager(config, model_manager)
    )

    # print(object_manager.animal2labelarray)
    heatmap = object_manager.get_occupation_heatmap()
    print(heatmap)
    print(object_manager.grid_labels)
