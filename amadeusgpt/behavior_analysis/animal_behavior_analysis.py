import types

from amadeusgpt.behavior_analysis.identifier import Identifier
from amadeusgpt.managers import (AnimalManager, EventManager, GUIManager,
                                 Manager, ObjectManager, RelationshipManager,
                                 VisualManager)
from amadeusgpt.programs.api_registry import (DEFAULT_REGISTRY,
                                              INTEGRATION_API_REGISTRY)


class AnimalBehaviorAnalysis:
    """
    This class holds methods and objects that are useful for analyzing animal behavior.
    It owns multiple manager classes that are responsible for different aspects of the analysis.
    """

    def __init__(self, identifier: Identifier, **kwargs):

        # animal manager needs keypoint_file_path and model_manager for pose
        self.animal_manager = AnimalManager(identifier)

        # object manager needs sam_info, seriralized pickle objects
        self.object_manager = ObjectManager(identifier, self.animal_manager)

        # relationship manager needs animal_manager and object_manager
        self.relationship_manager = RelationshipManager(
            identifier, self.animal_manager, self.object_manager
        )

        # event manager needs reference to object_manager, animal_manager, and relationship_manager
        self.event_manager = EventManager(
            identifier,
            self.object_manager,
            self.animal_manager,
            self.relationship_manager,
        )

        # some managers need references to others to do their job
        self.visual_manager = VisualManager(
            identifier, self.animal_manager, self.object_manager
        )

        self.gui_manager = GUIManager(identifier, self.object_manager)

        # check all attributes that are inheritance of manager classes
        # and attach them as methods to the main class
        self._attach_manager_methods()

        # attach the integration methods to the main class
        self._attach_integration_methods()

    # Put the methods of the managers in the main behavior analysis class
    # So we can save tokens and make it easier for LLM to learn to use the methods
    def _attach_manager_methods(self):
        method_names = set()
        for attr in dir(self):
            if isinstance(getattr(self, attr), Manager):
                manager = getattr(self, attr)
                cls_name = manager.__class__.__name__
                for method_name in DEFAULT_REGISTRY[cls_name]:
                    if hasattr(manager, method_name):

                        method = getattr(manager, method_name)
                        setattr(
                            self,
                            method_name,
                            method.__get__(manager, manager.__class__),
                        )
                        if method_name not in method_names:
                            method_names.add(method_name)
                        else:
                            raise ValueError(
                                f"Method {method_name} already exists in the class"
                            )

    def _attach_integration_methods(self):
        ### adding the methods from the dictionary to the class
        for method_name, func_info in INTEGRATION_API_REGISTRY.items():
            # this bound still needs to happen before we can register the function into a instance method at this class
            bound_method = types.MethodType(func_info["func"], self)
            setattr(self, method_name, bound_method)

    def summary(self, manager_name=None):
        for manager in [
            self.animal_manager,
            self.object_manager,
            self.relationship_manager,
            self.event_manager,
            self.visual_manager,
        ]:
            if manager_name is not None:
                if manager.__class__.__name__ != manager_name:
                    continue
            print(manager.__class__.__name__)
            manager.summary()


if __name__ == "__main__":

    pass
