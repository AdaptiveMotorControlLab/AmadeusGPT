import dataclasses
from dataclasses import dataclass, field, fields
import traceback
from typing import List, Dict
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from amadeusgpt.implementation import (
    AnimalBehaviorAnalysis,
    AnimalAnimalEvent,
    AnimalEvent,
)
import sys
from amadeusgpt.utils import frame_number_to_minute_seconds, get_fps, parse_error_message_from_python


@dataclass
class Figure:
    plot_type: str    
    axes: list
    figure: plt.Figure = field(default=None)
    plot_caption: str = ""


@dataclass
class AmadeusAnswer:
    # needs to further decompose to chain of thoughts etc.
    chain_of_thoughts: str = None
    str_answer: str = ''
    error_function_code: str = None
    function_code: str = None
    plots: List[Dict[str, any]] = field(default_factory=list) 
    # this is different from chain of thought, as it needs to combine function outputs
    error_message: str = None
    has_error: bool = False
    summary: str = None
    ndarray: List[any] = field(default_factory=list)
    role = "ai"

    def asdict(self):
        return dataclasses.asdict(self)

    @classmethod
    def fromdict(cls, _dict):
        """
        We only parse those fields that are defined here
        """
        instance = cls()
        field_list = fields(cls)
        for _field in field_list:
            if _field.name in _dict:
                setattr(instance, _field.name, _dict[_field.name])

        return instance

    def parse_plot_tuple(self, tu, plot_type="general_plot"):
        data = {}
        data["plot_type"] = plot_type
        for e in tu:
            if isinstance(e, matplotlib.figure.Figure):
                data["figure"] = e
            elif "Axes" in str(e):
                data["axes"] = e        
            elif isinstance(e, str):
                data["plot_caption"] = ""  

        if "figure" in data and "axes" in data:
            ret = Figure(**data)  
        else:
            ret = None

        return ret

    def get_plots_for_animal_animal_events(
        self, animal_animal_events: AnimalAnimalEvent
    ):
        behavior_analysis = AnimalBehaviorAnalysis()
        video_file_path = AnimalBehaviorAnalysis.get_video_file_path()
        for animal_name, animal_events in animal_animal_events.items():
            etho_plot_info = behavior_analysis.plot_ethogram(animal_events)
            traj_plot_info = behavior_analysis.plot_trajectory(
                ["all"], events=animal_events
            )
            etho_figure_obj = self.parse_plot_tuple(
                etho_plot_info, plot_type="ethogram"
            )
            etho_caption = ""
            for other_animal_name in animal_animal_events[animal_name]:
                event_list = animal_animal_events[animal_name][other_animal_name]
                for event in event_list:
                    etho_caption += f"The interaction between {animal_name} and {other_animal_name} happens {frame_number_to_minute_seconds(event.start,video_file_path), frame_number_to_minute_seconds(event.end,video_file_path)}, and it lasts {event.duration:.2f} seconds\n"
            etho_figure_obj.plot_caption = etho_caption

            traj_figure_obj = self.parse_plot_tuple(
                traj_plot_info, plot_type="trajectory"
            )

            self.plots.append(traj_figure_obj)
            self.plots.append(etho_figure_obj)

    def get_plots_for_animal_events(self, animal_events: AnimalEvent):
        """
        We always plot ethogram and trajectories for events
        """
        behavior_analysis = AnimalBehaviorAnalysis()
        video_file_path = AnimalBehaviorAnalysis.get_video_file_path()
        etho_plot_info = behavior_analysis.plot_ethogram(animal_events)
        traj_plot_info = behavior_analysis.plot_trajectory(
            ["all"], events=animal_events
        )
        etho_figure_obj = self.parse_plot_tuple(etho_plot_info, plot_type="ethogram")
        etho_caption = ""
        for animal_name in animal_events:
            event_list = animal_events[animal_name]
            for event in event_list:
                etho_caption += f"For {animal_name}, the behavior happens {frame_number_to_minute_seconds(event.start, video_file_path), frame_number_to_minute_seconds(event.end, video_file_path)}, and it lasts {event.duration:.2f} seconds\n"

        etho_figure_obj.plot_caption = etho_caption

        traj_figure_obj = self.parse_plot_tuple(traj_plot_info, plot_type="trajectory")

        self.plots.append(traj_figure_obj)
        self.plots.append(etho_figure_obj)


    @classmethod
    def from_error_message(cls):
        instance = AmadeusAnswer()                 
        instance.has_error = True
        instance.error_message = parse_error_message_from_python()
        return instance

    @classmethod
    def from_text_answer(cls, text):
        instance = AmadeusAnswer()                 
        instance.chain_of_thoughts = text
        return instance    

    @classmethod
    def from_function_returns(cls, function_returns, function_code, thought_process):
        """
        function_returns: Tuple(ret1, ret2, ret3 ... )
        populate the data fields from function returns.
        """
        instance = AmadeusAnswer()      
        instance.function_code = function_code
        instance.chain_of_thoughts = thought_process
        # If the function returns are tuple, try to parse the tuple and generate plots 
        if isinstance(function_returns, tuple):
            # if the returns contain plots, the return must be tuple (fig, axes)
            _plots = instance.parse_plot_tuple(function_returns)
            if _plots:
                instance.plots.append(_plots)       
        # without wrapping it in a list, the following for loop can cause problems     
        if isinstance(function_returns, tuple):
            function_returns = list(function_returns)
        else:
            function_returns = [function_returns]
       
        for function_return in function_returns:                    
            if isinstance(function_return, (pd.Series, pd.DataFrame, np.ndarray)):
                if isinstance(function_return, (pd.Series,pd.DataFrame)):
                    function_return = function_return.to_numpy()            
            elif isinstance(function_return, AnimalEvent):
                instance.get_plots_for_animal_events(function_return)
            elif isinstance(function_return, AnimalAnimalEvent):
                instance.get_plots_for_animal_animal_events(function_return)
            else:
                if not isinstance(
                    function_return,
                    (matplotlib.figure.Figure, matplotlib.axes._axes.Axes),
                ):
                    instance.str_answer = str(function_return)
        return instance
