import matplotlib.path as mpath
import numpy as np

from .base import AnalysisObject


class Object(AnalysisObject):
    def __init__(self, name: str):
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
            print(f"{attr_name} has {self.__dict__[attr_name]}")

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

    def __init__(self, name: str, masks: dict):
        super().__init__(name)
        self.masks = masks
        self.bbox = self.masks.get("bbox")
        self.area = self.masks["area"]
        # _seg could be either binary mask or rle string
        _seg: dict = self.masks.get("segmentation")
        # this is rle format
        from pycocotools import mask as mask_decoder

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
        self.area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
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
        self.area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        self.center = np.array([np.mean(vertices[:, 0]), np.mean(vertices[:, 1])])


class GridObject(Object):
    def __init__(self, name: str, region):
        super().__init__(name)
        self.region = region
        self.x_min = region["x"]
        self.y_min = region["y"]
        self.x_max = region["x"] + region["w"]
        self.y_max = region["y"] + region["h"]
        self.center = np.array(
            [self.x_min + region["w"] / 2, self.y_min + region["h"] / 2]
        )

        self.Path = self.points2Path(
            [
                [self.x_min, self.y_min],
                [self.x_max, self.y_min],
                [self.x_max, self.y_max],
                [self.x_min, self.y_max],
            ]
        )
        self.points = np.array(
            [
                [self.x_min, self.y_min],
                [self.x_max, self.y_min],
                [self.x_max, self.y_max],
                [self.x_min, self.y_max],
            ]
        )
        self.area = region["w"] * region["h"]
