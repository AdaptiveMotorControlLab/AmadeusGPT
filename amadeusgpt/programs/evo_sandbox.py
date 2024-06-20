import numpy as np

from amadeusgpt.programs.sandbox import Sandbox
from amadeusgpt.programs.task_program_registry import TaskProgramLibrary


class EvoSandbox(Sandbox):
    def __init__(self, config):
        super().__init__(config)
        self.task_program_library = TaskProgramLibrary().get_task_programs()
        self.breed_info = None
        # string of detailed scores
        # - total duration in seconds
        # - number of videos occured
        self.detailed_scores = {}
        # just a scalar value
        self.scores = {}

    def update_breed_info(
        self, task_program1_docs, task_program2_docs, composition_type
    ):
        self.breed_info = (task_program1_docs, task_program2_docs, composition_type)

    def get_breed_info(self):
        return self.breed_info

    def get_grid_info(self):
        analysis = create_analysis(self.config)
        # [[A1 A1],
        #  [B1,B2]]
        grid_labels = analysis.object_manager.grid_labels
        grid_label_text = []
        for i in range(len(grid_labels)):
            grid_label_text.append(",".join(grid_labels[i]))
        grid_label_text = "\n".join(grid_label_text)
        # [[0, 20],
        #  [60, 20]]

        occupation_heatmap = analysis.object_manager.get_occupation_heatmap()

        occupation_heatmap = [
            np.expand_dims(e, axis=0) for e in occupation_heatmap.values()
        ]
        occupation_heatmap = np.concatenate(occupation_heatmap, axis=0)
        occupation_heatmap = np.round(np.mean(occupation_heatmap, axis=0), 2)

        return grid_label_text, occupation_heatmap

    def get_useful_info(self):
        # useful information includes
        # 1) keypoint names of the animals
        # 2) median speed of the animals
        analysis = create_analysis(self.config)
        keypoint_names = analysis.get_keypoint_names()
        # reducing the n_frames and n_keypoints dimensions
        # speed n_frames, n_individuals, n_kpts, 1
        speed = analysis.get_speed()
        median_speeds = np.nanmedian(np.abs(speed), axis=(1, 2))
        percentile_90 = round(np.percentile(median_speeds, 90), 2)
        percentile_30 = round(np.percentile(median_speeds, 30), 2)

        # text = ""
        # for animal_id, (median_speed, max_speed) in enumerate(zip(median_speeds, max_speeds)):
        #     median_speed = np.round(median_speed.item(),3)
        #     max_speed = np.round(max_speed.item(), 3)
        #     text += f"animal{animal_id}'s speed: median:{median_speed}, max:{max_speed} \n"
        text = f"""
            - A speed above {percentile_90} pixels/frame is considered "fast".
            - A speed below {percentile_30} pixels/frame is considered "slow".
            - A distance greater than 15 pixels is considered "far".
            - A distance less than 3 pixels is considered "close".
        """
        ret = f"```useful_info\nkeypoint_names: {keypoint_names}\speed stats:\n{text}\n```"
        # print (ret)
        return ret

    def get_task_program_docs(self):
        ret = "```taskprograms\n"
        keys, scores = [], []
        for key, value in self.scores.items():
            keys.append(key)
            scores.append(value)
        keys = np.array(keys)
        scores = np.array(scores)

        scores += 10
        total_score = sum(scores)
        probabilities = [score / total_score for score in scores]
        prob_dict = dict(zip(keys, probabilities))

        for name, prob in prob_dict.items():
            task_program = self.task_program_library[name]
            description = task_program.json_obj["docstring"]
            by_llm = False
            if task_program["creator"] != "human":
                by_llm = True
            if by_llm:
                ret += f"{name}(config):\n- description:{description}\n- (this program is created by you) {self.detailed_scores[name]}\n"
            else:
                ret += f"{name}(config):\n- description:{description}\n- (this program is a primitive task program) {self.detailed_scores[name]}\n"

        ret += "\n```"
        # print ('task program block')
        # print (ret)
        return ret
