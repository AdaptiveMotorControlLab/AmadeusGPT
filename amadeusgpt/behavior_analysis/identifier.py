import os

from amadeusgpt.config import Config


class Identifier:
    """
    config contains project level meta
    other parameters help identify the unique params needed for animal behavior analysis
    Currently video_file_path and keypoint_file_path.
    Can be more in the future
    """

    def __init__(
        self, config: Config | dict, video_file_path: str, keypoint_file_path: str
    ):

        self.config = config
        self.video_file_path = video_file_path
        self.keypoint_file_path = keypoint_file_path

    def __str__(self):
        return f"""------
video_file_path: {self.video_file_path} 
keypoint_file_path: {self.keypoint_file_path}
config: {self.config}
------
"""

    def __eq__(self, other):
        if os.path.exists(self.video_file_path):
            return os.path.abspath(self.video_file_path) == os.path.abspath(
                other.video_file_path
            )
        else:
            return os.path.abspath(self.keypoint_file_path) == os.path.abspath(
                other.keypoint_file_path
            )

    def __hash__(self):
        if os.path.exists(self.video_file_path):
            return hash(os.path.abspath(self.video_file_path))
        else:
            return hash(os.path.abspath(self.keypoint_file_path))
