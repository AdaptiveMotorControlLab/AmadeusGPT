from amadeusgpt.analysis_objects.llm import LLM
import cv2
import numpy as np 
import time 
from amadeusgpt.config import Config
import base64
import io 

class VideoSampler:
    """
    Sample a video into segments and then sample the segments into multiple frames using OpenCV.
    
    Attributes:
        video_path (str): Path to the video file.
        segment_duration (float): Duration of each segment in seconds.
        frames_per_segment (int): Number of frames to sample from each segment.
    """
    def __init__(self, video_path, segment_duration, frames_per_segment):
        """
        Initializes the VideoSampler with the specified video path, segment duration, and frames per segment.
        
        Args:
            video_path (str): Path to the video file.
            segment_duration (float): Duration of each video segment in seconds.
            frames_per_segment (int): Number of frames to extract per segment.
        """
        self.video_path = video_path
        self.segment_duration = segment_duration
        self.frames_per_segment = frames_per_segment

    def process_video(self):
        """
        Processes the video by segmenting it and then extracting frames from each segment.
        
        Returns:
            dict: A dictionary containing segment indices and their corresponding frames.
        """
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        segment_frame_count = int(self.segment_duration * fps)
        frames_to_capture = np.linspace(0, segment_frame_count - 1, self.frames_per_segment, dtype=int)

        video_data = {}
        segment_index = 0
        start = time.time()
        while cap.isOpened():
            segment_frames = []
            base_frame_index = segment_index * segment_frame_count
            for i in range(segment_frame_count):
                ret, frame = cap.read()
                if not ret:
                    break
                if i in frames_to_capture:
                    segment_frames.append(frame)

            if segment_frames:
                video_data[segment_index] = segment_frames
                segment_index += 1
            else:
                break

            # Skip to the next segment
            cap.set(cv2.CAP_PROP_POS_FRAMES, base_frame_index + segment_frame_count)
        end = time.time()
        print(f'Time taken for {total_frames} frames:', end - start)
        cap.release()
        return video_data


class VideoLLM(LLM):
    """
    Run GPT-4o on concatenated frames of a video
    """
    def __init__(self, config):
        super().__init__(config)
    def encode_image(self, image):
        result, buffer = cv2.imencode(".jpeg", image)
        image_bytes = io.BytesIO(buffer)
        base64_image = base64.b64encode(image_bytes.getvalue()).decode("utf-8")
        return base64_image

    def images_to_video(self, image_list, output_path, fps, size=None):
        """
        Convert a list of images (numpy arrays) to a video.

        Args:
        image_list (list of numpy.ndarray): List containing image frames.
        output_path (str): Path where the output video will be saved.
        fps (int): Frames per second in the output video.
        size (tuple): Optional. Size of the video as (width, height).

        Raises:
        ValueError: If image_list is empty or size mismatch occurs.
        """
        if not image_list:
            raise ValueError("The image list is empty")

        # Determine the size of the video
        if size is None:
            # Take size from the first image
            height, width = image_list[0].shape[:2]
            size = (width, height)
        else:
            # Check if all images are of the same size
            height, width = size
            if any((img.shape[1] != width or img.shape[0] != height) for img in image_list):
                raise ValueError("All images must be of the same size as 'size' parameter")

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
        out = cv2.VideoWriter(output_path, fourcc, fps, size)

        # Write frames to the video
        for frame in image_list:
            # Convert to BGR format for saving
            if len(frame.shape) == 2 or frame.shape[2] == 1:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            out.write(frame)

        # Release everything when job is finished
        out.release()
        print(f"Video saved as {output_path}")

    def speak(self, video_data):
        from amadeusgpt.agentic.video_llm_prompt import _get_system_prompt
        self.system_prompt = _get_system_prompt()    
        temp = [e for e in video_data.values()]
        images = []
        for segment in temp:
            images.extend(segment)
        print ('total images GPT-4 sees', len(images))
        self.images_to_video(images, 'gpt4_sees_this.mp4', 30)
     

        multi_image_content = self.prepare_multi_image_content(images)       
        self.update_history("system", self.system_prompt)
        self.update_history(
            "user", "This video is about a mouse in its home cage. The circular object in the center is a treadmill. The colorful dots are keypoints from DeepLabCut. You can ignore those dots.", multi_image_content=multi_image_content, in_place = True)

        response = self.connect_gpt(self.context_window)
        text = response.choices[0].message.content.strip()
        print(text)
       


if __name__  ==  '__main__':
    video_path = '/Users/shaokaiye/AmadeusGPT-dev/examples/MausHaus/maushaus_trimmedDLC_snapshot-1000_labeled_x264.mp4'
    config = Config('/Users/shaokaiye/AmadeusGPT-dev/amadeusgpt/configs/template.yaml')
    segment_duration = 10
    frames_per_segment = 1

    video_sampler = VideoSampler(video_path, segment_duration, frames_per_segment)
    video_data = video_sampler.process_video()

    video_data = {k:v for k,v in video_data.items() if k < 30}
    video_llm = VideoLLM(config=config)
    video_llm.speak(video_data)
