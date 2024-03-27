from ast import Dict
from .base import AnalysisObject
import numpy as np
import matplotlib.path as mpath
from typing import List, Dict, Any
from pycocotools import mask as mask_decoder
from scipy.spatial import ConvexHull
from numpy import ndarray
from functools import lru_cache

class Object(AnalysisObject):    
    def __init__(self,                  
                 name: str):
        """
        TODO: instead of using a point, use the true segmentation as reference point
        name: str for referencing the object
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
        self.name = name
        self.center = None
        self.x_min = None
        self.x_max = None
        self.y_min = None
        self.y_max = None
        self.Path = None
        self.points = None
    def get_name(self):
        return self.name
    def get_data(self):
        """
        The class multiple fields
        """
        return self


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

    def __getitem__(self, key):
        return getattr(self, key)

    def summary(self):
        for attr_name in self.__dict__:
            print(f'{attr_name} has {self.__dict__[attr_name]}')

    def distance(self, other_object):
        # we use the center of two objects for calculating distance
        return np.linalg.norm(self.center - other_object.center)
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
    def get_path(self):
        """
        The representation of points in the object
        """
        return self.Path

    def overlap(self, other_object):
        # if there is a polygon path corresponding this object, use the contain_point
        # otherwise, use the bounding box representation
        if self.get_path() is not None:
            return self.path.contains_point(other_object.center)
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
   


class SegObject(Object):
    """
        segmentation : the mask
        area : the area of the mask in pixels
        bbox : the boundary box of the mask in XYWH format
        predicted_iou : the model's own prediction for the quality of the mask
        point_coords : the sampled input point that generated this mask
        stability_score : an additional measure of mask quality
        crop_box : the crop of the image used to generate this mask in XYWH format
    """
    def __init__(self, name: str, masks:dict):        
        super().__init__(name)
        self.masks = masks
        self.bbox = self.masks.get("bbox")
        self.area = self.masks["area"]
        # _seg could be either binary mask or rle string
        _seg:dict = self.masks.get("segmentation")
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
    def plot(self, ax):
        x, y = zip(*self.points)
        # Plot the polygon
        ax.plot(x, y, "b-")  # 'b-' means blue line

class MatplotlibObject(Object):
    def __init__(self, name: str, object_path):        
        super().__init__(name)
        self.object_path = object_path
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
class ROIObject(Object):
    def __init__(self, name: str, canvas_path):
        super().__init__(name)
        self.canvas_path = canvas_path
        points = canvas_path
        if isinstance(canvas_path, mpath.Path):
            self.Path = canvas_path
        else:
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
        

class Animal(Object):        
    def get_keypoint_names(self):
        """
        keypoint names should be the basic attributes
        """
        pass
    def summary(self):
        print (self.__class__.__name__)
        for attr_name in self.__dict__:
            print(f'{attr_name} has {self.__dict__[attr_name]}')

class AnimalSeq(Animal):
    """
    Because we support passing bodyparts indices for initializing an AnimalSeq object,
    body center, left, right, above, top are relative to the subset of keypoints.
    Attributes
    ----------
    self._coords: arr potentially subset of keypoints
    self.wholebody: full set of keypoints. This is important for overlap relationship
    """
    def __init__(self, animal_name: str,
                 keypoints: ndarray, 
                 keypoint_names: List[str]):        
        self.name = animal_name
        self.whole_body: ndarray = keypoints
        self.keypoint_names = keypoint_names      
        self._paths = []
        self.state = {}
        self.kinematics_types = ["speed", "acceleration"]
        self.bodypart_relation = ["bodypart_pairwise_distance"]
        self.support_body_orientation = False
        self.support_head_orientation = False
        # self.keypoints are updated by indices of keypoint names given
        keypoint_indices = [keypoint_names.index(name) for name in keypoint_names]
        self.keypoints = self.whole_body[:, keypoint_indices]


    def update_roi_keypoint_by_names(self, keypoint_names: List[str]):
        # update self.keypoints based on keypoint names given
        if keypoint_names is None:
            return
        keypoint_indices = [self.keypoint_names.index(name) for name in keypoint_names]
        self.keypoints = self.whole_body[:, keypoint_indices]

    def restore_roi_keypoint(self):
        self.keypoints = self.whole_body

    def set_body_orientation_keypoints(self, body_orientation_keypoints: Dict[str, Any]):
        self.neck_name = body_orientation_keypoints["neck"]
        self.tail_base_name = body_orientation_keypoints["tail_base"]
        self.animal_center_name = body_orientation_keypoints["animal_center"]
        self.support_body_orientation = True
    def set_head_orientation_keypoints(self, head_orientation_keypoints: Dict[str, Any]):
        self.nose_name = head_orientation_keypoints["nose"]
        self.neck_name = head_orientation_keypoints["neck"]
        self.support_head_orientation = True

    # all the properties cannot be cached because update could happen
    def get_paths(self):   
        paths = []
        for ind in range(self.whole_body.shape[0]):
            paths.append(self.get_path(ind))
        return paths

    def get_path(self, ind):
        xy = self.whole_body[ind]
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

    def get_keypoints(self, average_keypoints = False)-> ndarray:
        if average_keypoints:
            return np.nanmedian(self.keypoints, axis=1)
        return self.keypoints

    def get_center(self):
        return np.nanmedian(self.keypoints, axis=1).squeeze()

    def get_xmin(self):       
        return np.nanmin(self.keypoints[..., 0], axis=1)

    def get_xmax(self):        
        return np.nanmax(self.keypoints[..., 0], axis=1)
        
    def get_ymin(self):       
        return np.nanmin(self.keypoints[..., 1], axis=1)

    def get_ymax(self):       
        return np.nanmax(self.keypoints[..., 1], axis=1)

    def get_keypoint_names(self):
        return self.keypoint_names

    def query_states(self, query: str) -> ndarray:
        assert query in ["speed",
                         "acceleration",
                         "bodypart_pairwise_distance"], f"{query} is not supported"

        if query == "speed":
            self.state[query] = self.get_speed()
        elif query == "acceleration":
            self.state[query] = self.get_acceleration()
        elif query == "bodypart_pairwise_distance":
            self.state[query] = self.get_bodypart_wise_relation()

        return self.state[query]
   
    def get_speed(self)-> ndarray:
        keypoints = self.get_keypoints()
        speed = (
            np.diff(keypoints, axis=0) / 30
        )  # divided by frame rate to get speed in pixels/second
        # Pad velocities to match the original shape
        speed = np.concatenate([np.zeros((1,) + speed.shape[1:]), speed])
        # Compute the speed from the velocity
        speed = np.sqrt(np.sum(np.square(speed), axis=-1, keepdims=True))
        return speed

    def get_acceleration(self)-> ndarray:
        keypoints = self.get_keypoints()
        # Calculate differences in keypoints between frames (velocity)
        velocities = np.diff(keypoints, axis=0) / 30
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
    def get_bodypart_wise_relation(self):
        keypoints = self.get_keypoints()
        diff = keypoints[..., np.newaxis, :, :] - keypoints[..., :, np.newaxis, :]
        sq_dist = np.sum(diff**2, axis=-1)
        distances = np.sqrt(sq_dist)
        return distances
    

    def get_body_cs(self,
               ):
        # this only works for topview
        neck_index = self.keypoint_names.index(self.neck_name)
        tailbase_index = self.keypoint_names.index(self.tail_base_name)
        neck = self.whole_body[:, neck_index]
        tailbase = self.whole_body[:, tailbase_index]
        body_axis = neck - tailbase
        body_axis_norm = body_axis / np.linalg.norm(body_axis, axis=1, keepdims=True)
        # Get a normal vector pointing left
        mediolat_axis_norm = body_axis_norm[:, [1, 0]].copy()
        mediolat_axis_norm[:, 0] *= -1
        nrows = len(body_axis_norm)
        animal_cs = np.zeros((nrows, 3, 3))
        rot = np.stack((body_axis_norm, mediolat_axis_norm), axis=2)
        animal_cs[:, :2, :2] = rot
        animal_center_index = self.keypoint_names.index(self.animal_center_name)
        animal_cs[:, :, 2] = np.c_[
            self.whole_body[:, animal_center_index], np.ones(nrows)
        ]  # center back

        return animal_cs
    def calc_head_cs(self):
        nose_index = self.keypoint_names.index(self.nose_name)
        nose = self.whole_body[:, nose_index]
        neck_index = self.keypoint_names.index(self.neck_name)
        neck = self.whole_body[:, neck_index]
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