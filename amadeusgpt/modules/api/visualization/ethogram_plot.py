def plot_object_ethogram(self, object_names):
    """
    Object ethogram that shows the relationship between animals and objects.
    Parameters
    ----------
    object_names: List[str]
    list of object names
    Return:
    -------
    Tuple[plt.Figure, plt.Axes]
        tuple of figure and axes
    Examples
    --------
    >>> # plot object ethogram that shows overlapping relationship between animals and objects
    >>> def task_program():
    >>>     behavior_analysis = AnimalBehaviorAnalysis()
    >>>     object_names = behavior_analysis.get_object_names()
    >>>     etho_plot_info = behavior_analysis.plot_object_ethogram(object_names)
    >>>     return etho_plot_info
    """
    return plot_object_ethogram(object_names)
