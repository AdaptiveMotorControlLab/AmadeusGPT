from .model_manager import ModelManager
from .base import Manager
from amadeusgpt.analysis_objects.object import AnimalSeq
import numpy as np
from numpy import ndarray
import pandas as pd
from typing import List, Dict
from amadeusgpt.api_registry import register_class_methods, register_core_api



def get_orientation_vector(cls, b1_name, b2_name):
    b1 = cls.get_keypoints()[:, :, cls.get_bodypart_index(b1_name), :]
    b2 = cls.get_keypoints()[:, :, cls.get_bodypart_index(b2_name), :]
    return b1 - b2

def ast_fillna_2d(arr: ndarray) -> ndarray:
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

def reject_outlier_keypoints(keypoints:ndarray,
                              threshold_in_stds:int=2):
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
    def __init__(self, config: Dict[str, str], model_manager: ModelManager):
        """    
        """
        self.config = config
        self.model_manager = model_manager
        self.animals: List[AnimalSeq] = []
        self.keypoint_file_path = config['dlc_info']['keypoint_file_path']
        if self.keypoint_file_path.endswith('.h5'):
            all_keypoints = self._process_keypoint_file_from_h5()
        else:
            # could be coco format
            pass
        for individual_id in range(self.n_individuals):
            animal_name = f'animal_{individual_id}'
            # by default, we initialize all animals with the same keypoints and all the keypoint names
            self.animals.append(AnimalSeq(animal_name, 
                                                  all_keypoints[:, individual_id],
                                                  self.keypoint_names))


    def _process_keypoint_file_from_h5(self) -> ndarray:
        df = pd.read_hdf(self.keypoint_file_path)
        self.full_keypoint_names = list(df.columns.get_level_values("bodyparts").unique())
        self.keypoint_names = [k for k in self.full_keypoint_names]
        if len(df.columns.levels) > 3:
            self.n_individuals = len(df.columns.levels[1])
        else:
            self.n_individuals = 1       
       
        self.n_frames = df.shape[0]
        self.n_kpts = len(self.keypoint_names)
              
        df_array = df.to_numpy().reshape((self.n_frames, self.n_individuals, self.n_kpts, -1))[
            ..., :2
        ]

        df_array = reject_outlier_keypoints(df_array)
        df_array = ast_fillna_2d(df_array)
        return df_array        

       

    
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
    def get_animal_names(self)-> List[str]:
        """
        Get the names of all the animals.
        """
        return [animal.get_name() for animal in self.animals]

    def get_animals_by_names(self, names:List[str])-> List[AnimalSeq]:
        return [self.animals[name] for name in names]
    

    @register_core_api
    def get_keypoints(self) -> ndarray:
        """
        Get the keypoints.
        """
        return np.array([animal.get_keypoints() for animal in self.animals])
    @register_core_api
    def get_speed(self) -> ndarray:
        """
        Get the speed.
        """
        return np.array([animal.get_speed() for animal in self.animals])
    
    @register_core_api
    def get_acceleration(self) -> ndarray:
        """
        Get the acceleration.
        """
        return np.array([animal.get_acceleration() for animal in self.animals])


    @register_core_api
    def get_n_individuals(self) -> int:
        """
        Get the number of individuals in the data.
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
        Get the names of the bodyparts.
        """
        return self.full_keypoint_names

    def query_animal_states(self, animal_name :str, query: str) -> ndarray:
                
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
        return ['animals']