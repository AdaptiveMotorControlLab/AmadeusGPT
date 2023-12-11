import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from amadeusgpt.implementation import AnimalBehaviorAnalysis
from matplotlib.collections import LineCollection


def _make_line_collection(
    coords, links, start=0, end=-1, inds=None, color_stance="plum", alpha=0.5):
    color = mcolors.to_rgb("gray")
    colors = np.array([color] * len(coords))
    if inds is not None:
        mask = np.zeros(coords.shape[0], dtype=bool)
        for ind1, ind2 in inds:
            mask[ind1 : ind2 + 1] = True
        colors[mask] = mcolors.to_rgb(color_stance)
    sl = slice(start, end)
    colors = colors[sl]
    segs = coords[sl, links].reshape((-1, 2, 2))
    colors = np.repeat(colors, len(links), axis=0)
    coll = LineCollection(segs, colors=colors, alpha=alpha)
    return coll, segs


def plot_gait_analysis_results(
    self, gait_analysis_results, limb_keypoints, color_stance="plum"
):
    coords = AnimalBehaviorAnalysis.get_keypoints()[:, 0]
    stance_inds = gait_analysis_results["stances"][0]

    fig, ax = plt.subplots(sharex=True, sharey=True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")

    skeleton = []
    for kpt1, kpt2 in zip(limb_keypoints, limb_keypoints[1:]):
        skeleton.append(
            (
                AnimalBehaviorAnalysis.get_bodypart_index(kpt1),
                AnimalBehaviorAnalysis.get_bodypart_index(kpt2),
            ),
        )
    coll, segs = _make_line_collection(
        coords, skeleton, inds=stance_inds, color_stance=color_stance
    )
    xmin, ymin = np.nanmin(segs, axis=(0, 1))
    xmax, ymax = np.nanmax(segs, axis=(0, 1))
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.invert_yaxis()
    ax.add_collection(coll)
    ax.set_ylabel("Limb")
    return fig, ax, "Visualization of the gait"


AnimalBehaviorAnalysis.plot_gait_analysis_results = plot_gait_analysis_results

