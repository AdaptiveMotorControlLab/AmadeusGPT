from amadeusgpt.behavior_analysis import AnimalBehaviorAnalysis

analysis_fac = {}

def create_analysis(config, video_file_path, keypoint_file_path):

    identifier = video_file_path
    if str(identifier) not in analysis_fac:
        analysis_fac[identifier] = AnimalBehaviorAnalysis(config, video_file_path, keypoint_file_path)

    return analysis_fac[identifier]
