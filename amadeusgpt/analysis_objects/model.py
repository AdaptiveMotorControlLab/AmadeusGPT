"""
This class is for deep learning models
"""

from .base import AnalysisObject
from .object import Object
from typing import List, Dict, Any, Union
import platform
import cv2
import numpy as np
import msgpack
import pickle
import os



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
     # extract pose from the video file with superanimal name superanimal_topviewmouse
     def task_program():
         superanimal_name = "superanimal_topviewmouse"
         keypoint_file_path = AnimalBehaviorAnalysis.superanimal_video_inference(superanimal_name)
         return keypoint_file_path
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

from segment_anything import (
    SamAutomaticMaskGenerator,
    SamPredictor,
    sam_model_registry,
)

class Model(AnalysisObject):
    def __init__(self, config):
        self.config = config  



class DeepLabCut(Model):
    pass


class Segmentation(Model):
    """
    Base class for segmentation.
    Should support saving the mask to disk and loading it automatically
    This is because model like SAM can take a long time
    """

    def __init__(self, config: Dict[str, any]):
        """
        filename specifies the path to the potential serialized segmentation file
        We make sure that the segmentation files have same formats
        """
        self.filename = config['seg_filename']
        self.pickledata = None
        self.load()
    def get_name(self)->str:
        return self.filename
    
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

class SAM(Segmentation):
    """
    Class that captures the state of objects, supported by Seg everything
    """
    
    #def __init__(self, ckpt_path, model_type, filename=None):
    def __init__(self, sam_info: Dict[str, Any]):
        super().__init__(sam_info)
        self.sam_info = sam_info
        self.ckpt_path: Union[str, None] = self.sam_info.get("ckpt_path")
        self.model_type: Union[str,None] = self.sam_info.get("model_type")
        self.scene_frame_number = self.sam_info.get("scene_frame_number")

        sam = sam_model_registry[self.model_type](checkpoint=self.ckpt_path)
        device = "cpu" if platform.system() == "Darwin" else "cuda"
        sam.to(device)
        self.mask_generator = SamAutomaticMaskGenerator(sam)

    def generate_mask(self, image):
        masks = self.mask_generator.generate(image)
        return masks

    def generate_mask_at_frame(self, video_file_path, frame_id):
        cap = cv2.VideoCapture(video_file_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        masks = self.generate_mask(frame)
        cap.release()
        # cv2.destroyAllWindows()
        return masks
    

    def get_objects(self, video_file_path: str, sam_info: Dict[str, Any], frame_id = 0):         
        # assuming objects are still
        if self.pickledata is None:
            masks = self.generate_mask_at_frame(video_file_path, frame_id)
            objects = {}
            for object_name, mask in enumerate(masks):
                obj = Object(str(object_name), masks=mask)
                objects[str(object_name)] = obj
            return objects
        else:
            return self.pickledata
    

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
    