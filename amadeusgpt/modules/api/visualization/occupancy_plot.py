def plot_occupancy_heatmap(self, events=None, **kwargs):
    """
    Parameters
    ----------
    events: Union[EventDict, Dict[str, EventDict]], optional
    Examples
    --------
     # plot occupancy heatmap colored by velocity
     def task_program():
         behavior_analysis = AnimalBehaviorAnalysis()
         fig, ax = behavior_analysis.plot_occupancy_heatmap()
         return fig, ax
    """
    return plot_occupancy_heatmap(events)
