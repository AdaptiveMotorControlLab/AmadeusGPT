"""
Define a series of transform that embedding algorithms might find usesful
Not sure yet how much this can be done by GPT-4 or done by us
"""

import numpy as np


def align_poses(self, keypoints: np.ndarray) -> np.ndarray:
    """
    Aligns 2D keypoints by translating them to the mean position and scaling them based on the bounding box size.

    Args:
        keypoints (numpy.ndarray): 2D array of keypoints with shape (n_frames, n_kpts, 2).

    Returns:
        numpy.ndarray: Aligned keypoints with the same shape as the input.
    """
    n_frames, n_individuals, n_kpts, _ = keypoints.shape
    aligned_keypoints = np.copy(keypoints)
    # Compute mean position across all frames
    mean_position = np.mean(aligned_keypoints, axis=(0, 2))
    # Translate keypoints to mean position
    aligned_keypoints -= mean_position
    # Compute maximum absolute coordinate value across all frames
    max_abs_coord = np.max(np.abs(aligned_keypoints))
    # Scale keypoints based on bounding box size
    aligned_keypoints /= max_abs_coord

    return aligned_keypoints
