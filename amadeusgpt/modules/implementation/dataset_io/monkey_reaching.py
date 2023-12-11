import numpy as np
import pandas as pd

from amadeusgpt.implementation import AnimalBehaviorAnalysis


def load_monkey_dataset(self):
    import cebra

    monkey_pos = cebra.datasets.init("area2-bump-pos-active")
    monkey_target = cebra.datasets.init(
        "area2-bump-target-active"
    ).discrete_index.numpy()

    monkey_keypoints = monkey_pos.continuous_index.numpy()
    monkey_keypoints = np.concatenate(
        (monkey_keypoints, np.ones((monkey_keypoints.shape[0], 1))), axis=1
    )
    bodyparts = ["hand"]
    columnindex = pd.MultiIndex.from_product(
        [["dlcscorer"], bodyparts, ["x", "y", "confidence"]],
        names=["scorer", "bodyparts", "coords"],
    )
    df = pd.DataFrame(
        monkey_keypoints,
        columns=columnindex,
        index=np.arange(monkey_keypoints.shape[0]),
    )
    output_filename = "monkey_reaching.h5"
    df.to_hdf(output_filename, key="df_with_missing", mode="w")
    AnimalBehaviorAnalysis.set_keypoint_file_path(output_filename)
    AnimalBehaviorAnalysis.set_neural_data(monkey_pos.neural)
    # just a pointer to dataset and we do not assume anything yet
    dataset = {"x": monkey_keypoints, "y": monkey_target}
    print(monkey_keypoints.shape)
    print(monkey_target.shape)
    AnimalBehaviorAnalysis.set_dataset(dataset)
    return dataset


AnimalBehaviorAnalysis.load_monkey_dataset = load_monkey_dataset
