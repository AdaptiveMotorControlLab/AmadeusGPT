import os
import pickle
import time

import pandas as pd

import amadeusgpt
from amadeusgpt import AMADEUS, AnimalBehaviorAnalysis, utils

print(amadeusgpt.__file__)

root_dir = "examples/EPM"

# this should include all paper examples using the saved roi pickle

EPM_queries = [
    "How much time does the mouse spend in the closed arm, which is ROI0?",  # Fig 5, Query 1
    "How much time does the mouse spend outside the closed arm?",  # Fig 5, Query 2
    "define head_dips as a behavior where the mouse's mouse_center and neck are in ROI0 which is open arm while head_midpoint is outside ROI1 which is the cross-shape area. When does head_dips happen and what is the number of bouts for head_dips?",  # head dip example
]


def test_EPM():
    keypoint_file = os.path.join(root_dir, "EPM_11DLC_snapshot-1000.h5")
    video_file = os.path.join(root_dir, "EPM_11DLC_snapshot-1000_labeled_x264.mp4")

    AnimalBehaviorAnalysis.set_video_file_path(video_file)
    AnimalBehaviorAnalysis.set_keypoint_file_path(keypoint_file)

    for query in EPM_queries:
        with open("examples/EPM/roi_objects.pickle", "rb") as f:
            roi_objects = pickle.load(f)
            AnimalBehaviorAnalysis.set_roi_objects(roi_objects)
        AMADEUS.chat_iteration(query)
        AMADEUS.release_cache_objects()


test_EPM()
