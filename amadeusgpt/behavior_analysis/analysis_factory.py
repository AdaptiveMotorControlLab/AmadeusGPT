from amadeusgpt.behavior_analysis.animal_behavior_analysis import \
    AnimalBehaviorAnalysis
from amadeusgpt.behavior_analysis.identifier import Identifier

analysis_fac = {}


def create_analysis(identifier: Identifier):

    if identifier not in analysis_fac:
        analysis_fac[identifier] = AnimalBehaviorAnalysis(identifier)
    return analysis_fac[identifier]
