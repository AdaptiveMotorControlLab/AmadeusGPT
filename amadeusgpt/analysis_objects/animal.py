from typing import Any, Dict, List

import matplotlib.path as mpath
import numpy as np
from numpy import ndarray
from scipy.spatial import ConvexHull
from amadeusgpt.analysis_objects.object import Object


class Animal(Object):
    def get_keypoint_names(self):
        """
        keypoint names should be the basic attributes
        """
        pass

    def summary(self):
        print(self.__class__.__name__)
        for attr_name in self.__dict__:
            print(f"{attr_name} has {self.__dict__[attr_name]}")


class AnimalSeq(Animal):
    """
    Because we support passing bodyparts indices for initializing an AnimalSeq object,
    body center, left, right, above, top are relative to the subset of keypoints.
    Attributes
    ----------
    self.wholebody: np.ndarray of keypoints of all bodyparts
    self.keypoint
    """

    def __init__(self, animal_name: str, keypoints: ndarray, keypoint_names: List[str]):
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
        self.center = np.nanmedian(self.whole_body, axis=1)

    def update_roi_keypoint_by_names(self, keypoint_names: List[str]):
        # update self.keypoints based on keypoint names given
        if keypoint_names is None:
            return
        keypoint_indices = [self.keypoint_names.index(name) for name in keypoint_names]
        self.keypoints = self.whole_body[:, keypoint_indices]

    def restore_roi_keypoint(self):
        self.keypoints = self.whole_body

    def set_body_orientation_keypoints(
        self, body_orientation_keypoints: Dict[str, Any]
    ):
        self.neck_name = body_orientation_keypoints["neck"]
        self.tail_base_name = body_orientation_keypoints["tail_base"]
        self.animal_center_name = body_orientation_keypoints["animal_center"]
        self.support_body_orientation = True

    def set_head_orientation_keypoints(
        self, head_orientation_keypoints: Dict[str, Any]
    ):
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

    def get_keypoints(self, average_keypoints=False) -> ndarray:
        assert (
            len(self.keypoints.shape) == 3
        ), f"keypoints shape is {self.keypoints.shape}"
        if not average_keypoints:
            return self.keypoints
        else:
            return np.nanmean(self.keypoints, axis=1)

    def get_center(self):
        """
        median is more robust than mean
        """
        return np.nanmedian(self.keypoints, axis=1).squeeze()

    def get_xmin(self):
        return np.nanmin(self.keypoints[..., 0], axis=1)

    def get_xmax(self):
        return np.nanmax(self.keypoints[..., 0], axis=1)

    def get_ymin(self):
        return np.nanmin(self.keypoints[..., 1], axis=1)

    def get_ymax(self):
        return np.nanmax(self.keypoints[..., 1], axis=1)

    def get_zmin(self):
        return np.nanmin(self.keypoints[..., 2], axis=1)

    def get_zmax(self):
        return np.nanmax(self.keypoints[..., 2], axis=1)

    def get_keypoint_names(self):
        return self.keypoint_names
    

    def query_states(self, query: str) -> ndarray:
        assert query in [
            "speed",
            "acceleration_mag",
            "bodypart_pairwise_distance",
        ], f"{query} is not supported"

        if query == "speed":
            self.state[query] = self.get_speed()
        elif query == "acceleration_mag":
            self.state[query] = self.get_acceleration_mag()
        elif query == "bodypart_pairwise_distance":
            self.state[query] = self.get_bodypart_wise_relation()

        return self.state[query]

    def get_velocity(self) -> ndarray:
        keypoints = self.get_keypoints()
        velocity = np.diff(keypoints, axis=0) / 30
        velocity = np.concatenate([np.zeros((1,) + velocity.shape[1:]), velocity])
        assert len(velocity.shape) == 3
        return velocity

    def get_speed(self) -> ndarray:
        keypoints = self.get_keypoints()
        velocity = (
            np.diff(keypoints, axis=0) / 30
        )  # divided by frame rate to get speed in pixels/second
        # Pad velocities to match the original shape
        velocity = np.concatenate([np.zeros((1,) + velocity.shape[1:]), velocity])
        # Compute the speed from the velocity
        speed = np.linalg.norm(velocity, axis=-1)
        speed = np.expand_dims(speed, axis=-1)
        assert len(speed.shape) == 3
        return speed

    def get_acceleration(self) -> ndarray:
        velocities = self.get_velocity()
        accelerations = (
            np.diff(velocities, axis=0) / 30
        )  # divided by frame rate to get acceleration in pixels/second^2
        # Pad accelerations to match the original shape
        accelerations = np.concatenate(
            [np.zeros((1,) + accelerations.shape[1:]), accelerations], axis=0
        )
        assert len(accelerations.shape) == 3
        return accelerations

    def get_acceleration_mag(self) -> ndarray:
        """
        Returns the magnitude of the acceleration vector
        """
        accelerations = self.get_acceleration()
        acceleration_mag = np.linalg.norm(accelerations, axis=-1)
        acceleration_mag = np.expand_dims(acceleration_mag, axis=-1)
        assert len(acceleration_mag.shape) == 3
        return acceleration_mag

    def get_bodypart_wise_relation(self):
        keypoints = self.get_keypoints()
        diff = keypoints[..., np.newaxis, :, :] - keypoints[..., :, np.newaxis, :]
        sq_dist = np.sum(diff**2, axis=-1)
        distances = np.sqrt(sq_dist)
        return distances

    def get_body_cs(
        self,
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


if __name__ == "__main__":
    # unit testing the shape of kinematics data
    # acceleration, acceleration_mag, velocity, speed, and keypoints

    from amadeusgpt.config import Config
    from amadeusgpt.main import AMADEUS

    config = Config(
        "/Users/shaokaiye/AmadeusGPT-dev/amadeusgpt/configs/MausHaus_template.yaml"
    )
    amadeus = AMADEUS(config)
    analysis = amadeus.get_analysis()
    # get an instance of animal
    animal = analysis.animal_manager.get_animals()[0]

    print("velocity shape", animal.get_velocity().shape)
    print("speed shape", animal.get_speed().shape)
    print("acceleration shape", animal.get_acceleration().shape)
    print("acceleration_mag shape", animal.get_acceleration_mag().shape)

    print(animal.query_states("acceleration_mag").shape)
