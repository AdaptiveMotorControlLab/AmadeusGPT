def plot_gait_analysis_results(
    self, gait_analysis_results, limb_keypoints: List[str], color_stance="plum"
):
    """    
    Visualize the results of gait analysis.
    Parameters
    ----------
    analysis: output from run_gait_analysis
    limb_keypoints: list of keypoint names on limb
    Returns
    -------
    Tuple[plt.Figure, List[plt.Axes], plot_message]
    Always return a tuple of figure and axes
    Examples
    --------
     # After running a gait analysis on the keypoint 'Offfrontfoot', show the results with stance periods colored in blue. Plot the limb using the points 'Offfrontfoot', 'Offfrontfetlock', and 'Offknee'.
     def task_program():
         behavior_analysis = AnimalBehaviorAnalysis()
         analysis = behavior_analysis.run_gait_analysis(limb_keypoint_names=['front_right_paw'])
         limb_keypoints = ['Offfrontfoot', 'Offfrontfetlock', 'Offknee']
         plot_info = behavior_analysis.plot_gait_analysis_results(analysis, limb_keypoints, color_stance='blue')
         return plot_info
    """
    return plot_gait_analysis_results(
        gait_analysis_results, limb_keypoints, color_stance=color_stance
    )
