import numpy as np
import umap
from umap.umap_ import UMAP
from amadeusgpt.implementation import AnimalBehaviorAnalysis
from .transform import align_poses


def compute_embedding_with_umap_and_plot_embedding(
    self, inputs, n_dimension=2, color_by_time=False, color_by_object=False, **kwargs
):
    features = inputs.reshape(inputs.shape[0], -1)
    features = np.nan_to_num(features)
    reducer = UMAP(n_components=n_dimension, min_dist=0.5)
    embedding = reducer.fit_transform(features)

    behavior_analysis = AnimalBehaviorAnalysis()
    embed_plot_info = behavior_analysis.plot_neural_behavior_embedding(
        embedding,
        color_by_time=color_by_time,
        color_by_object=color_by_object,
        **kwargs,
    )

    return embedding, embed_plot_info


AnimalBehaviorAnalysis.compute_embedding_with_umap_and_plot_embedding = (
    compute_embedding_with_umap_and_plot_embedding
)
