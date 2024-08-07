import numpy as np
from cebra import CEBRA

from amadeusgpt.programs.api_registry import register_integration_api


# those functions need to have a self keyword to access attributes in AnimalBehaviorAnalysis
@register_integration_api
def get_cebra_embedding(self, inputs: np.ndarray, n_dimension=3) -> np.ndarray:
    """
    This function takes non-centered keypoints and calculate the embeddings using the CEBRA algorithm.
    Parameters
    ----------
    inputs: np.ndarray 4d tensor of shape (n_frames, n_individuals, n_kpts, n_features)
    n_dimensions: int, optional, default to be 3
    Returns
    -------
    ndarray of shape (n_frames,  n_dimension)
    """
    features = inputs.reshape(inputs.shape[0], -1)
    features = np.nan_to_num(features)

    print("features shape", features.shape)
    cebra_params = dict(
        model_architecture="offset10-model",
        batch_size=512,
        learning_rate=3e-4,
        temperature=1.12,
        output_dimension=n_dimension,
        max_iterations=10,
        distance="cosine",
        conditional="time_delta",
        device="cpu",
        verbose=True,
        time_offsets=10,
    )
    print("got here1")
    model = CEBRA(**cebra_params)
    print("got here2")
    model.fit(features)
    print("got here3")
    embeddings = model.transform(features)
    print("got here4")
    return embeddings
