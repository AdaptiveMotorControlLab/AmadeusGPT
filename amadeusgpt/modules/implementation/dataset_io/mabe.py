import numpy as np
import pandas as pd

from amadeusgpt.implementation import AnimalBehaviorAnalysis


def load_mabe_dataset(self):
    """
    Examples
    --------
    >>> # load mabe 2022 dataset
    >>> def task_program():
    >>>     behavior_analysis = AnimalBehaviorAnalysis()
    >>>     return behavior_analysis.load_monkey_dataset()
    """

    a = np.load("user_train.npy", allow_pickle=True).item()
    sequences = a["sequences"]
    chase_videos = []
    ret = {}
    for video_id in sequences:
        annotations = sequences[video_id]["annotations"]
        keypoints = sequences[video_id]["keypoints"]
        if np.sum(annotations[0]) > 0:
            chase_videos.append(video_id)
    for chase_video in chase_videos:
        annotations = sequences[chase_video]["annotations"]
        keypoints = sequences[chase_video]["keypoints"]
        ret[chase_video] = {
            "annotations": annotations,
            "keypoints": keypoints[..., ::-1],
        }
    # let's use one video for now
    video_name = chase_videos[0]
    keypoints = ret[video_name]["keypoints"].reshape(keypoints.shape[0], -1)
    print("keypoint length", len(keypoints))
    action_labels = ret[video_name]["annotations"][0]
    print("action label length", len(action_labels))
    bodyparts = [
        "nose",
        "left ear",
        "right ear",
        "neck",
        "left forepaw",
        "right forepaw",
        "center back",
        "left hindpaw",
        "right hindpaw",
        "tail base",
        "tail middle",
        "tail tip",
    ]
    individuals = ["mouse1", "mouse2", "mouse3"]
    columnindex = pd.MultiIndex.from_product(
        [["dlcscorer"], individuals, bodyparts, ["x", "y"]],
        names=["scorer", "individuals", "bodyparts", "coords"],
    )
    df = pd.DataFrame(
        keypoints, columns=columnindex, index=np.arange(keypoints.shape[0])
    )
    output_filename = f"mabe_{video_name}.h5"
    df.to_hdf(output_filename, key="df_with_missing", mode="w")
    AnimalBehaviorAnalysis.set_keypoint_file_path(output_filename)
    AnimalBehaviorAnalysis.set_action_label(action_labels)
    dataset = {"x": keypoints, "y": action_labels}

    AnimalBehaviorAnalysis.set_dataset(dataset)
    return dataset


AnimalBehaviorAnalysis.load_mabe_dataset = load_mabe_dataset
