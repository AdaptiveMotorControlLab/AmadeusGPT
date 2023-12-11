from collections import defaultdict
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from amadeusgpt.implementation import AnimalBehaviorAnalysis, Database
from amadeusgpt.utils import _plot_ethogram


def plot_object_ethogram(self, object_names):
    # better to have empty events handling
    n_animals = Database.get("AnimalBehaviorAnalysis", "n_individuals")
    # this step is needed to initialize object segmentation
    AnimalBehaviorAnalysis.get_seg_objects()
    n_objects = len(object_names)

    # axes = axes.reshape(n_objects, n_animals)
    animals_objects_events = defaultdict(dict)
    behavior_analysis = AnimalBehaviorAnalysis()
    for object_name in object_names:
        animals_object_event = behavior_analysis.animals_object_events(
            object_name, "overlap", bodyparts=["all"]
        )
        for animal_name in animals_object_event:
            animals_objects_events[animal_name][object_name] = animals_object_event[
                animal_name
            ]
    plt.close("all")
    valid_objects = []
    for animal_id, (animal_name, animal_objects_events) in enumerate(
        animals_objects_events.items()
    ):
        for object_id, (object_name, animal_object_events) in enumerate(
            animal_objects_events.items()
        ):
            ret = {}
            mask = np.zeros(len(AnimalBehaviorAnalysis.get_keypoints()), dtype=bool)
            for event in animal_object_events:
                mask |= event.mask
            if np.sum(mask) != 0:
                valid_objects.append(object_id)
    fig, axes = plt.subplots(
        nrows=len(valid_objects),
        ncols=n_animals,
        sharex=True,
        # sharey=True,
        tight_layout=True,
        squeeze=False,
    )
    for animal_id, (animal_name, animal_objects_events) in enumerate(
        animals_objects_events.items()
    ):
        for object_id, (object_name, animal_object_events) in enumerate(
            animal_objects_events.items()
        ):
            if object_id not in valid_objects:
                continue
            offset_id = valid_objects.index(object_id)
            ret = {}
            axes[0, animal_id].set_title(animal_name)
            mask = np.zeros(len(AnimalBehaviorAnalysis.get_keypoints()), dtype=bool)
            for event in animal_object_events:
                mask |= event.mask
            if np.sum(mask) != 0:
                ret[f"object{object_name}"] = mask
            if len(ret) > 0:
                _plot_ethogram(ret, axes[offset_id, animal_id])
            axes[offset_id, animal_id].spines["top"].set_visible(False)
            axes[offset_id, animal_id].spines["right"].set_visible(False)
            axes[offset_id, 0].set_yticks([0])
            axes[offset_id, 0].set_yticklabels([object_name])
    axes[-1, 0].set_xlabel("Frame #")

    return fig, axes


AnimalBehaviorAnalysis.plot_object_ethogram = plot_object_ethogram
