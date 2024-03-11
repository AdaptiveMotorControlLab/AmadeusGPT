def run_gait_analysis(self,limb_keypoint_names):
    """
    Compute an animal's gait parameters given a list of distal keypoints.
    Examples
    --------
     # Analyze the animal's gait based on the keypoints 'front_right_paw' and 'back_right_paw'.
     def task_program():
         behavior_analysis = AnimalBehaviorAnalysis()
         analysis = behavior_analysis.run_gait_analysis(limb_keypoint_names=['front_right_paw', 'back_right_paw'])
         return analysis
    """
    return run_gait_analysis(limb_keypoint_names)
