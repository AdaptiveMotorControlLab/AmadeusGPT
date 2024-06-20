import os
import time

import pandas as pd

from amadeusgpt import AMADEUS, AnimalBehaviorAnalysis, utils

root_dir = "examples/MausHaus"


MausHaus_queries = [
    "When is the animal on the treadmill, which is object 5?",  # Fig 5 (b) Q 1
    "Plot the trajectory of the animal",  # Fig 5 (b) Q2,
    "When is the animal close to the object 35, if I define close as less than 50 pixels?",
]


def test_MausHaus():
    keypoint_file = os.path.join(root_dir, "maushaus_trimmedDLC_snapshot-1000.h5")
    video_file = os.path.join(
        root_dir, "maushaus_trimmedDLC_snapshot-1000_labeled_x264.mp4"
    )
    sam_pickle = os.path.join(root_dir, "sam_object.pickle")

    pd.reset_option("display.max_colwidth")

    AnimalBehaviorAnalysis.set_cache_objects(True)
    AnimalBehaviorAnalysis.set_video_file_path(video_file)
    AnimalBehaviorAnalysis.set_keypoint_file_path(keypoint_file)
    AnimalBehaviorAnalysis.set_sam_info(
        ckpt_path="", model_type="", pickle_path=sam_pickle
    )
    for query in MausHaus_queries:
        # fake roi to avoid the matplotlib thing
        AnimalBehaviorAnalysis.set_roi_objects({})
        AMADEUS.chat_iteration(query)


test_MausHaus()
