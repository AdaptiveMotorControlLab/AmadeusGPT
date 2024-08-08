from amadeusgpt import AMADEUS
from amadeusgpt.behavior_analysis.identifier import Identifier
import pytest
from amadeusgpt import create_project
from amadeusgpt.programs.task_program_registry import TaskProgramLibrary


@TaskProgramLibrary.register_task_program(creator="human")
def plot_trajectory(identifier : Identifier):
    """
    This task program describes the approach events between any pair of two animals.
    """
    # behavior_analysis was defined in the namespace. Just take this as syntax
    analysis = create_analysis(identifier)    
    fig, axs = analysis.visual_manager.get_keypoint_visualization()
                                                                          
    return fig, axs 

@pytest.mark.parametrize("example_name", ["MausHaus", "MausHaus3D"])
def test_task_program(example_name):
        data_folder = f"examples/{example_name}"
        result_folder = "temp_result_folder"

        kwargs = {
            "llm_info.max_tokens": 2000,
            "llm_info.temperature": 0.0,
            "llm_info.keep_last_n_messages": 2,
            "object_info.load_objects_from_disk": False,
            "object_info.use_grid_objects": False,
            "keypoint_info.use_3d": False,
            "keypoint_info.include_confidence": False
        }

        config = create_project(data_folder, result_folder, **kwargs)

        # Create an AMADEUS instance
        amadeus = AMADEUS(config, use_vlm=False)
        amadeus.run_task_program("plot_trajectory")