from amadeusgpt import AMADEUS, AnimalBehaviorAnalysis, utils
import pandas as pd
import os
import time

root_dir = "examples/MABe"

# Fig 5 (c)
MABE_queries = [
    "Define <|chases|> as a social behavior where closest distance between this animal and other animals is less than 40 pixels and the angle between this and other animals have to be less than 30 and this animal has to travel faster than 2.",
    "Define watch as a social behavior where distance between animals is less than 260 and larger than 50 and head angle between animals is less than 15. The smooth_window_size is 15.",
]


def test_MABE():
    keypoint_file = os.path.join(root_dir, "mabe_EGS8X2MN4SSUGFWAV976.h5")
    video_file = os.path.join(root_dir, "EGS8X2MN4SSUGFWAV976.mp4")

    AnimalBehaviorAnalysis.set_cache_objects(True)
    AnimalBehaviorAnalysis.set_video_file_path(video_file)
    AnimalBehaviorAnalysis.set_keypoint_file_path(keypoint_file)

    for query in MABE_queries:
        AnimalBehaviorAnalysis.set_roi_objects({})
        AMADEUS.chat_iteration(query)


test_MABE()
