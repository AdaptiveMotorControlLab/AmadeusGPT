def compute_embedding_with_cebra_and_plot_embedding(
    self,
    inputs,
    n_dimension=3,
    max_iterations=100,
    color_by_time=False,
    color_by_objet=False,
    **kwargs,
):
    """
    The color of embedding encodes segmentation objects in the scene
    Parameters
    ----------
    inputs: np.ndarray 4d tensor of shape (n_frames, n_individuals, n_kpts, n_features)
    kwargs: same optional parameters of plt.scatter
    Examples
    --------
    >>> # create a 3 dimensional embedding of speed using cebra with colormap rainbow
    >>> def task_program():
    >>>     behavior_analysis = AnimalBehaviorAnalysis()
    >>>     speed = behavior_analysis.get_speed()
    >>>     embedding, embed_plot_info = behavior_analysis.compute_embedding_with_cebra_and_plot_embedding(speed, n_dimension = 3, colormap = 'rainbow')
    >>>     return embedding, embed_plot_info
    >>> # create 3 d cebra embedding with limb keypoints Elbow and Shoulder
    >>> def task_program():
    >>>     behavior_analysis = AnimalBehaviorAnalysis()
    >>>     keypoints = behavior_analysis.get_keypoints()
    >>>     limb_indices = behavior_analysis.get_bodypart_indices(['Elbow', 'Shoulder'])
    >>>     embedding, embed_plot_info = behavior_analysis.compute_embedding_with_cebra_and_plot_embedding(keypoints[:,:,limb_indices], n_dimension = 3)
    >>>     return embedding, embed_plot_info

    """
    return compute_embedding_with_cebra_and_plot_embedding(
        inputs, n_dimension=n_dimension, max_iterations=max_iterations, kwargs=kwargs
    )
