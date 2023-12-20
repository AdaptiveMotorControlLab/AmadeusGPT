import cv2
import matplotlib
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.patches import PathPatch, Polygon
from matplotlib.path import Path
from matplotlib.widgets import Button, PolygonSelector


class ROISelector:
    roi_count = 0
    cmap = ["red", "blue", "yellow", "green"]

    def __init__(self, axs):
        self.axs = axs
        self.selector = PolygonSelector(self.axs, self.onselect)
        self.paths = []

    def roi_select_event(self, vertices):
        # once the bounding box is done drawing, run the following command
        first_point = vertices[0]
        vertices.append(first_point)
        path = Path(vertices)
        self.paths.append(path)
        # self.axs.clear()
        for i, path in enumerate(self.paths):
            self.axs.add_patch(
                plt.Polygon(path.vertices, fill=None, edgecolor=type(self).cmap[i])
            )
        handles = [
            mlines.Line2D([], [], color=self.cmap[i], label=f"ROI{i}")
            for i in range(len(self.paths))
        ]
        self.axs.legend(handles=handles, loc="upper right")

        # saving roi figure

    def onselect(self, vertices):
        self.roi_select_event(vertices)
        figure_output = "roi_figure.png"
        print(f"saving ROI to {figure_output}")
        plt.savefig(figure_output, dpi=800)


def select_roi_from_video(video_filename):
    cap = cv2.VideoCapture(video_filename)
    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Get the middle frame index
    middle_frame_index = int(total_frames / 2)
    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_index)
    ret, frame = cap.read()
    fig, axs = plt.subplots(1)
    axs.imshow(frame)
    selector = ROISelector(axs)
    #plt.show()
    return selector.paths


def select_roi_from_plot(fig, ax):
    selector = ROISelector(ax)
    fig.show()
    return selector.paths


if __name__ == "__main__":
    roi = select_roi_from_video("OFT_5.mp4")
    data = pd.read_hdf("OFT_5DLC_snapshot-5000.h5")
    data = data.dropna(how="all").to_numpy().reshape(-1, 27, 3)[..., :2]
    data = np.nanmean(data, axis=1)
    print(data.shape)
    mask = roi.contains_points(data)

    frame_ids = np.arange(1, len(mask) + 1)  # Frame IDs

    # Compute the counts of True values in the mask for each bin
    bin_counts = []
    bin_edges = np.arange(0, len(mask) + 1, 10)
    for i in range(len(bin_edges) - 1):
        start = bin_edges[i]
        end = bin_edges[i + 1]
        bin_mask = mask[start:end]
        bin_count = np.sum(bin_mask)
        bin_counts.append(bin_count)

    # Create a bar graph of the bin counts
    plt.bar(bin_edges[:-1], bin_counts, width=10)

    # Add labels and a title
    plt.xlabel("Frame ID")
    plt.ylabel("Count of True values in mask")
    plt.title("Animal occurrence in ROI")
    # Show the plot
    #plt.show()
