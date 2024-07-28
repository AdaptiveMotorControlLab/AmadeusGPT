from amadeusgpt.behavior_analysis.animal_behavior_analysis import \
    AnimalBehaviorAnalysis
from amadeusgpt.behavior_analysis.identifier import Identifier

analysis_fac = {}


def create_analysis(identifier: Identifier):

    if str(identifier) not in analysis_fac:
        analysis_fac[str(identifier)] = AnimalBehaviorAnalysis(identifier)
    return analysis_fac[str(identifier)]
