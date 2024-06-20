import numpy as np
from umap.umap_ import UMAP

from amadeusgpt.programs.api_registry import register_integration_api


@register_integration_api
def get_umap_embedding(self, inputs, n_dimension=2):
    """
    This function takes non-centered keyoints and calculate the embeddings using the UMAP algorithm.
    Parameters
    ----------
    inputs: np.ndarray 4d tensor of shape (n_frames, n_individuals, n_kpts, n_features)
    n_dimensions: int, optional, default to be 2
    """
    features = inputs.reshape(inputs.shape[0], -1)
    features = np.nan_to_num(features)
    reducer = UMAP(n_components=n_dimension, min_dist=0.5)
    embedding = reducer.fit_transform(features)

    return embedding
