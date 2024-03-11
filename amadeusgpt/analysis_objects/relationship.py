from .base import AnalysisObject
from .object import Object, AnimalSeq
from scipy.spatial.distance import cdist
from enum import IntEnum
import numpy as np
from typing import Dict, Any, List, Union
from numpy import ndarray

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
    nose = animal_seq.whole_body[:, 0]
    neck = animal_seq.whole_body[:, 3]
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

def get_pairwise_distance(arr1, arr2):
    # we want to make sure this uses a fast implementation
    # (n_frame,  n_kpts, 2)
    assert len(arr1.shape) == 3 and len(arr2.shape) == 3
    # pariwise distance (n_frames, n_kpts, n_kpts)
    pairwise_distances = (
        np.ones((arr1.shape[0], arr1.shape[1], arr2.shape[1])) * 100000
    )
    for frame_id in range(arr1.shape[0]):
        pairwise_distances[frame_id] = cdist(arr1[frame_id], arr2[frame_id])

    return pairwise_distances

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

class MABE:
    @classmethod
    def get_cs(cls, animal_seq):
        neck = animal_seq.whole_body[:, 3]
        tailbase = animal_seq.whole_body[:, 9]
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
            animal_seq.whole_body[:, 6], np.ones(nrows)
        ]  # center back

        return mouse_cs

class Relationship(AnalysisObject):
    """
    We never want to cache this.
    data: This attribute is a dictionary that contains the relationship between two objects
    each value of self.data is a numpy array that is either boolean or a float
    """
    def get_name(self):
        return self.__name__
    def query_relationship(self, relation_query:str) -> ndarray:
        ret =  self.data[relation_query]   
        return ret

    def summary(self):
        print (self.__class__.__name__)
        for attr_name in self.__dict__:
            print(f'{attr_name} has {self.__dict__[attr_name]}')
  
class AnimalObjectRelationship(Relationship):
    """
    To be referenced in the animal class
    """    
    def __init__(self, 
                 animal: AnimalSeq, 
                 other_obj: Object,
                 sender_animal_bodyparts_names: Union[List[str], None] = None,
                 ):
        self.sender_animal_name = animal.get_name()
        self.object_name = other_obj.get_name()
        self.sender_animal_bodyparts_names = sender_animal_bodyparts_names

        self.data: Dict[str, Any] = self._relationships_with_static_object(animal, other_obj)
 

    def __eq__(self, other: Relationship) -> bool:
        return self.sender_animal_name == other.sender_animal_name and \
        self.object_name == other.object_name and \
        self.sender_animal_bodyparts_names == other.sender_animal_bodyparts_names

    def _relationships_with_static_object(self, animal: AnimalSeq, other_obj: Object) -> Dict[str, Any]:
        if self.sender_animal_bodyparts_names is not None:
            # the keypoints of animal get updated when we update the roi bodypart names
            animal.update_roi_keypoint_names(self.sender_animal_bodyparts_names)

        c = other_obj.get_center()
        to_left = animal.get_center()[..., 0] <= other_obj.get_xmin()
        to_right = animal.get_center()[..., 0] >= other_obj.get_xmax()
        to_below = animal.get_center()[..., 1] >= other_obj.get_ymax()
        to_above = animal.get_center()[..., 1] <= other_obj.get_ymin()

        distance = np.linalg.norm(animal.get_center() - c, axis=1)
        overlap = other_obj.Path.contains_points(animal.get_center())
        # orientation = calc_orientation_in_egocentric_animal(
        #     animal, other_obj.get_center()
        # )
        return  {
            "to_left": to_left,
            "to_right": to_right,
            "to_below": to_below,
            "to_above": to_above,
            "distance": distance,
            "overlap": overlap,
            #"orientation": orientation,
        }
class AnimalAnimalRelationship(Relationship):
    """
    To be referenced in the animal class
    """
    def __init__(self, animal: AnimalSeq, 
                 other_animal: AnimalSeq,
                 sender_animal_bodyparts_names: Union[List[str], None] = None,
                 receiver_animal_bodyparts_names: Union[List[str], None] = None
                 ):
        
        self.sender_animal_name = animal.get_name()
        self.receiver_animal_name = other_animal.get_name()
        self.sender_animal_bodyparts_names = sender_animal_bodyparts_names
        self.receiver_animal_bodyparts_names = receiver_animal_bodyparts_names
        self.object_name = None
        self.data:Dict[str,Any] = self._animal_animal_relationship(animal, other_animal)

    def __eq__(self, other: Relationship) -> bool:
        return self.sender_animal_name == other.sender_animal_name and \
        self.receiver_animal_name == other.receiver_animal_name and \
        self.sender_animal_bodyparts_names == other.sender_animal_bodyparts_names and \
        self.receiver_animal_bodyparts_names == other.receiver_animal_bodyparts_names
    

    @classmethod
    def get_valid_inter_individual_relationship_queries(cls):
        return [
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

    def _animal_animal_relationship(self, 
                                    sender_animal: AnimalSeq,
                                    receiver_animal: AnimalSeq,                                   
                                    ) -> Dict[str, Any]:
        """
        This is the relationship between two animals
        """
        if self.sender_animal_bodyparts_names is not None:
            sender_animal.update_roi_keypoint_names(self.sender_animal_bodyparts_names)
        if self.receiver_animal_bodyparts_names is not None:
            receiver_animal.update_roi_keypoint_names(self.receiver_animal_bodyparts_names)

        c = sender_animal.get_center()

        to_left = sender_animal.get_xmax() <= receiver_animal.get_xmin()
        to_right = sender_animal.get_xmin() >= receiver_animal.get_xmax()
        to_below = sender_animal.get_ymin() >= receiver_animal.get_ymax()
        to_above = sender_animal.get_ymax() <= receiver_animal.get_ymin()
        distance = np.linalg.norm(sender_animal.get_center() - receiver_animal.get_center(), axis=1)
        # I have _coords for both this and other because people could want a subset of animal keypoints
        overlap = []
        # we only do nan to num here because doing it in other places give bad looking trajectory
        robust_center = np.nan_to_num(sender_animal.get_center())
        for path_id, other_path in enumerate(receiver_animal.get_paths()):
            if other_path is None:
                overlap.append(False)
                continue
            overlap.append(other_path.contains_point(np.array(robust_center[path_id])))
        overlap = np.array(overlap)

        # this angle is calculated using hardecoded MABE bodyparts
        # need to change in the future
        # angles = calc_angle_between_2d_coordinate_systems(
        #     MABE.get_cs(sender_animal), MABE.get_cs(receiver_animal)
        # )
        mouse_cs = calc_head_cs(sender_animal)

        head_cs_inv = []
        mouse_cs_inv = np.full_like(mouse_cs, np.nan)
        valid = np.isclose(np.linalg.det(mouse_cs[:, :2, :2]), 1)
        mouse_cs_inv[valid] = np.linalg.inv(mouse_cs[valid])
        head_cs_inv.append(mouse_cs_inv)
        relative_speed = np.abs(np.diff(np.linalg.norm(sender_animal.get_center() - c, axis=-1)))
        relative_speed = np.pad(relative_speed, (0, 1), mode="constant")
        head_angles = calc_angle_in_egocentric_animal(head_cs_inv, c)
        orientation = calc_orientation_in_egocentric_animal(sender_animal, receiver_animal.get_center())
        closest_distance = np.nanmin(
            get_pairwise_distance(sender_animal.keypoints, receiver_animal.keypoints), axis=(1, 2)
        )
        return {
            "to_left": to_left,
            "to_right": to_right,
            "to_below": to_below,
            "to_above": to_above,
            "distance": distance,
            "overlap": overlap,
            #"relative_angle": angles,
            "relative_head_angle": head_angles,
            "closest_distance": closest_distance,
            "relative_speed": relative_speed,
            "orientation": orientation,
        }
