import os
import re
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np

from amadeusgpt.analysis_objects.event import Event, EventGraph
from amadeusgpt.analysis_objects.relationship import Relationship
from amadeusgpt.behavior_analysis.identifier import Identifier
from amadeusgpt.programs.api_registry import (register_class_methods,
                                              register_core_api)

from .animal_manager import AnimalManager
from .base import Manager, cache_decorator
from .object_manager import ObjectManager
from .relationship_manager import RelationshipManager


def find_complement_number(string):
    digits = ""
    for char in string:
        if char.isdigit() or char == ".":
            digits += char
    return str(360 - float(digits))


def find_complement_operator(string):
    operator = ""
    for char in string:
        if not char.isdigit() and not char == ".":
            operator += char
    operator = operator.strip()
    if operator == "<":
        return ">="
    elif operator == "<=":
        return ">"
    elif operator == ">":
        return "<="
    elif operator == ">=":
        return "<"


def process_animal_object_relation(
    relation_query: str, comparison: str, animal_object_relation: np.ndarray
):
    if relation_query in ["relatve_head_angle"]:
        complement_relation = find_complement_operator(comparison)
        digits = find_complement_number(comparison)
        complement_comparison = complement_relation + digits
        relation_string = "numeric_quantity" + complement_comparison
        complement_animal_object_relation = eval(relation_string)
        animal_object_relation |= complement_animal_object_relation
    return animal_object_relation


@register_class_methods
class EventManager(Manager):
    def __init__(
        self,
        identifier: Identifier,
        object_manager: ObjectManager,
        animal_manager: AnimalManager,
        relationship_manager: RelationshipManager,
        use_cache: bool = False,
    ):
        self.config = identifier.config
        super().__init__(self.config, use_cache=use_cache)

        self.object_manager = object_manager
        self.animal_manager = animal_manager
        self.relationship_manager = relationship_manager
        self.video_file_path = identifier.video_file_path
        if not os.path.exists(self.video_file_path):
            # no need to initialize
            return

        self.animals_object_events = []
        self.animals_animals_events = []
        self.animals_state_events = []

    @register_core_api
    def get_animals_object_events(
        self,
        object_name: Optional[str] = "",
        query: Optional[str] = "",
        negate: Optional[bool] = False,
        bodypart_names: Optional[Union[List[str], None]] = None,
        min_window: int = 0,
        max_window: int = 1000000,
        smooth_window_size: int = 3,
    ) -> List[Event]:
        """
        This function is used to retrieve events that involve the interactions between animals and objects.
        object_name : str
        This parameter represents the name of the object of interest. It is expected to be a string. Must be one of the object names returned from function get_object_names.
        query: str. Examples are 'overlap == True', 'distance==30', 'angle<20',
        bodypart_names: List[str], optional
           bodyparts of the animal. Default to be None which means all keypoints of the animal are considered.
        min_window: min length of the event to include
        max_window: max length of the event to include
        negate: bool, default false
           whether to negate the spatial events. For example, if negate is set True, inside roi would be outside roi
        """
        pattern = r"(==|<=|>=|<|>)"
        if bodypart_names is not None:
            self.animal_manager.update_roi_keypoint_by_names(bodypart_names)

        animals_objects_relations = (
            self.relationship_manager.get_animals_objects_relationships(
                animal_bodyparts_names=bodypart_names
            )
        )

        ret_events = []
        for animal_objects_relationship in animals_objects_relations:
            if animal_objects_relationship.object_name != object_name:
                continue
            # to construct the events
            mask: List[bool] = None
            sender_animal_name = animal_objects_relationship.sender_animal_name
            comparison_operator = re.findall(pattern, query)[0].strip()
            _query = query.split(comparison_operator)[0].strip()
            _comparison = comparison_operator + "".join(
                query.split(comparison_operator)[1:]
            )
            numeric_quantity = animal_objects_relationship.query_relationship(_query)
            relation_string = "numeric_quantity" + _comparison
            mask = eval(relation_string)
            mask = process_animal_object_relation(_query, _comparison, mask)

            events: List[Event] = Event.mask2events(
                mask,
                self.video_file_path,
                sender_animal_name,
                set(),
                {object_name},
                smooth_window_size=smooth_window_size,
            )

            if negate:
                events = Event.event_negate(events)

            events = Event.filter_events_by_duration(events, min_window, max_window)

            ret_events.extend(events)

        if bodypart_names is not None:
            self.animal_manager.restore_roi_keypoint()

        return ret_events

    # @cache_decorator
    @register_core_api
    def get_animals_state_events(
        self,
        mask: np.ndarray,
        min_window: int = 10,
        max_window: int = 1000000,
    ) -> List[Event]:
        """
        Parameters
        ----------
        mask: np.ndarray, optional.
            The mask must be of shape (n_frames, n_individuals). It is a boolean mask that describes the condition for the behavior.
            If n_individuals is 1, the shape should be (n_frames, 1)
        min_window : int, optional, default 10
            Only include events that are longer than min_window
        max_window : int, optional, default 1000000
            Only include events that are shorter than max_window
        Returns
        -------
        List[Event]
        --------
        """

        if len(mask.shape) == 1:
            mask = mask.reshape(-1, 1)

        ret_events = []       
        for animal_idx, sender_animal_name in enumerate(
            self.animal_manager.get_animal_names()
        ):
            # to construct the events

            events = Event.mask2events(
                mask[:, animal_idx],
                self.video_file_path,
                sender_animal_name,
                set(),
                set(),
            )       
            events = Event.filter_events_by_duration(events, min_window, max_window)        
            ret_events.extend(events)

        ret_events = sorted(ret_events, key=lambda x: x.start)

        return ret_events

    @cache_decorator
    def get_events_from_relationship(
        self,
        relationship: Relationship,
        relation_query: str,
        comparison: str,
        smooth_window_size: int,
    ) -> List[Event]:

        mask = relationship.query_relationship(relation_query)

        # determine whether the mask is a numpy of float or numpy of boolean

        if mask.dtype != bool:
            relation_string = "mask" + comparison
            mask = eval(relation_string)

        sender_animal_name = relationship.sender_animal_name
        receiver_animal_names = set([relationship.receiver_animal_name])
        if relationship.object_name is not None:
            object_names = set([relationship.object_name])
        else:
            object_names = set()

        events = Event.mask2events(
            mask,
            self.video_file_path,
            sender_animal_name,
            receiver_animal_names,
            object_names,
            smooth_window_size=smooth_window_size,
        )

        return events

    # @cache_decorator
    @register_core_api
    def get_animals_animals_events(
        self,
        cross_animal_query_list: List = [],
        bodypart_names: Optional[Union[List[str], None]] = None,
        otheranimal_bodypart_names: Optional[Union[List[str], None]] = None,
        min_window: int = 11,
        max_window: int = 1000000,
        smooth_window_size: int = 3,
    ) -> List[Event]:
        """
        The function is for capturing behaviors that involve multiple animals. Don't fill the bodypart_names and otheranimal_bodypart_names unless you know the names of the bodyparts.
        When multiple queries are passed, they are combined as logical_and, not logical_or or sequential.

        Parameters
        ----------
        cross_animal_query_list:
        list of queries describing conditions among animals. It takes form of {type_of_query}{operator}{value}
        Examples are:
        'orientation==Orientation.FRONT', 'orientation==Orientation.BACK', 'distance<10', 'relative_angle>30',
        The allowed type of query is chosen from ["overlap", "distance", "relative_speed", "orientation", "closest_distance", "relative_angle", "relative_head_angle"]
        bodypart_names:
        list of bodyparts for the this animal. By default, it is None, which means all bodyparts are included.
        If set, the given bodyparts will be the "ROI" bodyparts that are used for the queries.
        otheranimal_bodypart_names: list[str], optional
        list of bodyparts for the other animals. By default, it is None, which means all bodyparts are included.
        If set, the given bodyparts will be the "ROI" bodyparts that are used for the queries.
        min_window: int, optional, default 11
        Only include events that are longer than min_window
        max_window: int, optional, default 100000
        smooth_window_size: int, optional
        smooth window size for smoothing the events.
        Returns
        -------
        List[Event]
        Note
        ----
        To capture a range for a numerical query  (e.g., relative_speed) between 3 and 10, one can do:
        get_animals_animals_events(cross_animal_query_list = ['relative_speed<10', 'relative_speed>3'],

        """

        if min_window is None:
            min_window = 3
        if max_window is None:
            max_window = 1000000

        animals_animals_relationships = (
            self.relationship_manager.get_animals_animals_relationships(
                sender_animal_bodyparts_names=bodypart_names,
                receiver_animal_bodyparts_names=otheranimal_bodypart_names,
            )
        )
        pattern = r"(==|<=|>=|<|>)"
        all_events = []
        for relationship in animals_animals_relationships:
            for query in cross_animal_query_list:
                # assert that query must contain one of the operators
                # find the operator
                # note we need to strip the spaces
                comparison_operator = re.findall(pattern, query)[0].strip()
                _query = query.split(comparison_operator)[0].strip()

                _comparison = comparison_operator + "".join(
                    query.split(comparison_operator)[1:]
                )

                events = self.get_events_from_relationship(
                    relationship, _query, _comparison, smooth_window_size
                )
                all_events.extend(events)
        graph = EventGraph.init_from_list(all_events)

        ###

        if len(cross_animal_query_list) > 1:
            graphs = []
            for animal_name in self.animal_manager.get_animal_names():
                for receiver_animal_name in self.animal_manager.get_animal_names():
                    if animal_name != receiver_animal_name:
                        subgraph = EventGraph.fuse_subgraph_by_kvs(
                            graph,
                            {
                                "sender_animal_name": animal_name,
                                "receiver_animal_names": set([receiver_animal_name]),
                            },
                            number_of_overlap_for_fusion=len(cross_animal_query_list),
                        )
                        graphs.append(subgraph)

            graph = EventGraph.merge_subgraphs(graphs)

        ret_events = graph.to_list()
        ret_events = Event.filter_events_by_duration(ret_events, min_window, max_window)

        return ret_events

    @register_core_api
    def get_duration(self, events: List[Event]) -> int:
        """
        This function is for calculating the total duration of a list events.
        The return value is in seconds.
        """
        return sum([event.duration_in_seconds for event in events])

    @register_core_api
    def get_composite_events(
        self,
        events_A: List[Event],
        events_B: List[Event],
        composition_type: Literal[
            "sequential", "logical_and", "logical_or"
        ] = "logical_and",
        max_interval_between_sequential_events: int = 15,
        min_window: int = 15,
        max_window: int = 100000,
    ) -> List[Event]:
        """
        This function is for combining two sets of events.
        Parameters
        ----------
        events_A: The first events to combine. When composition type is 'sequential', this event happens first.
        events_B: The second events to combine. When composition type is 'sequential', this event happens second.
        max_interval_between_sequential_events: int, optional, default 15. Only used for 'sequential' composition type. The interval should be adjusted based on the nature of the two events.
        comopsition_type:
        'logical_and': two behaviors happen at the same time.
        'logical_or': one of the behaviors happen
        'sequential': the second behavior happens after the first behavior.

        Returns
        -------
        List[Event]
        """
        assert composition_type in [
            "sequential",
            "logical_and",
            "logical_or",
        ], "composition_type must be either 'sequential' or 'logical_or', or 'logical_and'"
        events_list = [events_A, events_B]
        if composition_type == "sequential":
            graph_list = [EventGraph.init_from_list(events) for events in events_list]
            graphs = []
            count = 0
            for animal_name in self.animal_manager.get_animal_names():
                for i in range(1, len(graph_list)):
                    count += 1
                    sub_graph = EventGraph.concat_graphs(
                        graph_list[i - 1],
                        graph_list[i],
                        {"sender_animal_name": animal_name},
                        max_interval_between_sequential_events,
                    )
                    graphs.append(sub_graph)

            graph = EventGraph.merge_subgraphs(graphs)

        elif composition_type == "logical_or":
            all_events = []
            for events in events_list:
                all_events.extend(events)
            ret = Event.filter_events_by_duration(all_events, min_window, max_window)
            return ret

        elif composition_type == "logical_and":
            all_events = []
            for idx, events in enumerate(events_list):
                all_events.extend(events)
            graph = EventGraph.init_from_list(all_events)
            graphs = []

            # we first fuse events from different task programs that involve animal-animal interactions
            for animal_name in self.animal_manager.get_animal_names():
                receiver_animal_names = [
                    animal_name
                    for animal_name in self.animal_manager.get_animal_names()
                ]
                for receiver_animal_name in receiver_animal_names:
                    if animal_name != receiver_animal_name:
                        receiver_animal_name = set([receiver_animal_name])
                        animal_animal_subgraph = EventGraph.fuse_subgraph_by_kvs(
                            graph,
                            {
                                "sender_animal_name": animal_name,
                                "receiver_animal_names": receiver_animal_name,
                            },
                            number_of_overlap_for_fusion=2,
                        )
                        graphs.append(animal_animal_subgraph)
            # we then fuse events from different task programs that involve animal-object interactions
            for object_name in self.object_manager.get_object_names():
                # if we strictly require the object needs to match, then EPM head dipping example won't work.
                # so we cannot require objects to match. This causes some ambiguity.
                animal_object_subgraph = EventGraph.fuse_subgraph_by_kvs(
                    graph,
                    {
                        "sender_animal_name": animal_name,
                    },
                    # "object_names": object_name},
                    number_of_overlap_for_fusion=2,
                )
                graphs.append(animal_object_subgraph)

            # fuse events from different task programs that involve animal states
            for animal_name in self.animal_manager.get_animal_names():
                animal_state_subgraph = EventGraph.handle_animal_state_fusion(
                    graph,
                    {
                        "sender_animal_name": animal_name,
                        "receiver_animal_names": set([]),
                    },
                )
                graphs.append(animal_state_subgraph)

            graph = EventGraph.merge_subgraphs(graphs)

        ret = graph.to_list()
        ret = Event.filter_events_by_duration(ret, min_window, max_window)
        return ret

    # @register_core_api
    def from_mask(
        self,
        mask_tensor: np.ndarray,
    ) -> List[Event]:
        """

        This function expects to take a binary mask to describe a condition for the behavior and returns a list of events.
        For animal-animal interaction, it expects shape of  (n_frames, n_individuals, n_individuals)
        For condition about animals' own states, it expects shape of (n_frames, n_individuals)
        Returns
        -------
        List[Event]
        """
        assert (
            len(mask_tensor.shape) == 2 or len(mask_tensor.shape) == 3
        ), "mask_tensor must be of shape (n_frames, n_individuals) or (n_frames, n_individuals, n_individuals)"
        # mask is of shape (n_frames, n_individuals)
        ret = []
        if len(mask_tensor.shape) == 2:
            for animal_id, animal_name in enumerate(
                self.animal_manager.get_animal_names()
            ):
                mask = mask_tensor[:, animal_id]
                events = Event.mask2events(
                    mask,
                    self.video_file_path,
                    animal_name,
                    set([]),
                    set([]),
                    smooth_window_size=3,
                )
                ret.extend(events)

        elif len(mask_tensor.shape) == 3:
            for animal_id, animal_name in enumerate(
                self.animal_manager.get_animal_names()
            ):
                for other_animal_id, other_animal_name in enumerate(
                    self.animal_manager.get_animal_names()
                ):
                    if animal_id == other_animal_id:
                        continue
                    mask = mask_tensor[:, animal_id, other_animal_id]

                    events = Event.mask2events(
                        mask,
                        self.video_file_path,
                        animal_name,
                        set([other_animal_name]),
                        set([]),
                        smooth_window_size=3,
                    )
                    ret.extend(events)
        return ret

    def get_serializeable_list_names(self) -> List[str]:
        return [
            "animals_object_events",
            "animals_animals_events",
            "animals_state_events",
        ]
