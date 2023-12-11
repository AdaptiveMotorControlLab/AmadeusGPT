import cebra
import numpy as np
from cebra import CEBRA
from amadeusgpt.utils import filter_kwargs_for_function
from amadeusgpt.implementation import AnimalBehaviorAnalysis
from .transform import align_poses
import matplotlib.pyplot as plt
import os


def compute_embedding_with_cebra_and_plot_embedding(
    self, inputs, n_dimension=3, color_by_time=False, color_by_objet=False, **kwargs
):
    features = inputs.reshape(inputs.shape[0], -1)
    features = np.nan_to_num(features)

    cebra_params = dict(
        model_architecture="offset10-model",
        batch_size=512,
        learning_rate=3e-4,
        temperature=1.12,
        output_dimension=n_dimension,
        max_iterations=1 if "DEBUG" in os.environ else 500,
        distance="cosine",
        conditional="time_delta",
        device="cuda_if_available",
        verbose=False,
        time_offsets=10,
    )
    plot_kwargs = {k: v for k, v in kwargs.items() if k not in cebra_params}

    model = CEBRA(**cebra_params)

    model.fit(features)
    embeddings = model.transform(features)

    behavior_analysis = AnimalBehaviorAnalysis()
    print("plot_kwargs")
    print(plot_kwargs)
    embed_plot_info = behavior_analysis.plot_neural_behavior_embedding(
        embeddings,
        color_by_time=color_by_time,
        color_by_objet=color_by_objet,
        **plot_kwargs,
    )

    return embeddings, embed_plot_info


AnimalBehaviorAnalysis.compute_embedding_with_cebra_and_plot_embedding = (
    compute_embedding_with_cebra_and_plot_embedding
)
