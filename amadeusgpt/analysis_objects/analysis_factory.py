from amadeusgpt.implementation import AnimalBehaviorAnalysis

analysis_fac = {}

def create_analysis(config):
    if str(config) not in analysis_fac:
        analysis_fac[str(config)] = AnimalBehaviorAnalysis(config)

    return analysis_fac[str(config)]
