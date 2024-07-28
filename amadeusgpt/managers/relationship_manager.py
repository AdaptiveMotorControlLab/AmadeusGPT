from typing import Any, Dict, List, Union

from amadeusgpt.analysis_objects.relationship import (AnimalAnimalRelationship,
                                                      AnimalObjectRelationship,
                                                      Relationship)
from amadeusgpt.behavior_analysis.identifier import Identifier
from amadeusgpt.programs.api_registry import register_class_methods

from .animal_manager import AnimalManager
from .base import Manager, cache_decorator
from .object_manager import ObjectManager


@register_class_methods
class RelationshipManager(Manager):
    def __init__(
        self,
        identifier: Identifier,
        animal_manager: AnimalManager,
        object_manager: ObjectManager,
        use_cache: bool = False,
    ):
        super().__init__(identifier.config, use_cache=use_cache)
        self.config = identifier.config
        self.animal_manager = animal_manager
        self.object_manager = object_manager
        self.animals_objects_relationships = {}
        self.animals_animals_relationships = {}
        self._cache = {}
        self.use_cache = use_cache

    @cache_decorator
    def get_animals_objects_relationships(
        self, animal_bodyparts_names: Union[List[str], None] = None
    ) -> List[Relationship]:

        # roi, sam, animals are all objects
        roi_objs = self.object_manager.get_roi_objects()
        seg_objs = self.object_manager.get_seg_objects()
        grid_objs = self.object_manager.get_grid_objects()
        animals = self.animal_manager.get_animals()

        # there might be other objs
        objs = roi_objs + seg_objs + grid_objs

        # the key optimization opportunity here is to make following block faster
        # I don't know if we can vectorize the operations below. Maybe not.
        # therefore, it might be wise if we can vectorize the code

        animals_objects_relations = []

        for animal in animals:
            if animal_bodyparts_names is not None:
                # the keypoints of animal get updated when we update the roi bodypart names
                animal.update_roi_keypoint_by_names(animal_bodyparts_names)
            for object in objs:
                animal_object_relations = AnimalObjectRelationship(
                    animal, object, animal_bodyparts_names
                )
                animals_objects_relations.append(animal_object_relations)

        return animals_objects_relations

    @cache_decorator
    def get_animals_animals_relationships(
        self,
        sender_animal_bodyparts_names: Union[List[str], None] = None,
        receiver_animal_bodyparts_names: Union[List[str], None] = None,
    ) -> List[Relationship]:
        """
        This function basically returns the pairwise relationships between animals
        """

        animals = self.animal_manager.get_animals()
        animals_animals_relationships = []

        for sender_animal_idx, sender_animal in enumerate(animals):
            for receiver_animal_idx, receiver_animal in enumerate(animals):
                if sender_animal.get_name() != receiver_animal.get_name():
                    if sender_animal_bodyparts_names is not None:
                        # the keypoints of animal get updated when we update the roi bodypart names
                        sender_animal.update_roi_keypoint_by_names(
                            sender_animal_bodyparts_names
                        )
                    if receiver_animal_bodyparts_names is not None:
                        # the keypoints of animal get updated when we update the roi bodypart names
                        receiver_animal.update_roi_keypoint_by_names(
                            receiver_animal_bodyparts_names
                        )

                    animal_animal_relationship = AnimalAnimalRelationship(
                        sender_animal,
                        receiver_animal,
                        sender_animal_bodyparts_names,
                        receiver_animal_bodyparts_names,
                    )
                    animals_animals_relationships.append(animal_animal_relationship)
                    sender_animal.restore_roi_keypoint()
                    receiver_animal.restore_roi_keypoint()

        return animals_animals_relationships

    def get_serializeable_list_names(self) -> List[str]:
        return ["animals_objects_relationships", "animals_animals_relationships"]
