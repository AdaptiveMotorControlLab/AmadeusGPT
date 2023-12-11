def compute_embedding_with_umap_and_plot_embedding(
    inputs, n_dimension=3, color_by_object=False, color_by_time=False, **kwargs
):
    """
    Parameters
    ----------
    inputs: np.ndarray 4d tensor of shape (n_frames, n_individuals, n_kpts, n_features)
    kwargs: same optional parameters of plt.scatter
    Examples
    --------
    >>> # use the speed to create a 2d embedding with umap
    >>> def task_program():
    >>>     behavior_analysis = AnimalBehaviorAnalysis()
    >>>     speed = behavior_analysis.get_speed()
    >>>     embedding, embed_plot_info = behavior_analysis.compute_embedding_with_umap_and_plot_embedding(speed, n_dimension = 2)
    >>>     return embedding, embed_plot_info
    """
    return compute_embedding_with_umap_and_plot_embedding(
        inputs, n_dimension=n_dimension
    )
