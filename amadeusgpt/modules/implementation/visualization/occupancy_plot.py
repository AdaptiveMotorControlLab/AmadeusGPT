import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PatchCollection
from scipy.stats.kde import gaussian_kde

from amadeusgpt.implementation import AnimalBehaviorAnalysis, Event


def plot_occupancy_heatmap_per_animal(
    self,
    events=None,
    bodyparts=["all"],
    kin_type="velocity",
    bins=50,
    cmap="PuRd",
    cmap_path="cividis",
    color_contours="k",
    ax=None,
    xy=None,
    padding=50,
):
    xy = np.squeeze(xy)
    if events:
        mask = Event.events2onemask(events)
        if np.sum(mask) == 0:
            return ax
        xy = xy[mask]

    xmin, ymin = xy.min(axis=0) - padding
    xmin = max(xmin, 0)
    ymin = max(ymin, 0)

    xmax, ymax = xy.max(axis=0) + padding

    print("xmin,ymin, xmax,ymax")
    print(xmin, ymin, xmax, ymax)

    # Perform the kernel density estimate
    bins = complex(bins)
    xx, yy = np.mgrid[xmin:xmax:bins, ymin:ymax:bins]
    pos = np.vstack([xx.ravel(), yy.ravel()])
    kernel = gaussian_kde(xy.T)
    f = np.reshape(kernel(pos).T, xx.shape)

    if ax is None:
        _, ax = plt.subplots()
    ax.set_aspect("equal")
    ax.contourf(xx, yy, f, cmap=cmap, zorder=-1)
    if cmap_path:
        if kin_type == "velocity":
            # the user actually means the diff of speed

            speed = self.get_kinematics(bodyparts=["all"], kin_type="speed")
            speed = np.nanmean(speed, axis=1)
            diff_speed = np.diff(speed)
            kin = np.squeeze(diff_speed)

        points = xy.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        norm = plt.Normalize(*np.percentile(kin, [5, 95]))
        lc = LineCollection(segments, cmap=cmap_path, norm=norm)
        lc.set_array(kin)
        lc.set_linewidth(0.5)
        line = ax.add_collection(lc)
        cbar = ax.get_figure().colorbar(line, ax=ax)
        cbar.set_label(f"{kin_type}", rotation=270, labelpad=10)
    if color_contours:
        ax.contour(xx, yy, f, colors=color_contours, linewidths=0.5, zorder=-1)
    ax.grid("off")
    ax.axis("off")
    return ax


def plot_occupancy_heatmap(
    self,
    events=None,
    bodyparts=["all"],
    kin_type="velocity",
    bins=50,
    cmap_path="cividis",
    color_contours="k",
    padding=50,
    **kwargs,
):
    """
    Occupancy heatmap should be colored by the chosen kin_type.
    By default, the kin_type is location. In that case, the plot trajectory function is called
    """
    if "cmap" not in kwargs:
        cmap = "viridis"
    else:
        cmap = kwargs["cmap"]
    data = type(self).get_animal_centers()
    data = np.nan_to_num(data)
    n_individuals = type(self).n_individuals
    n_kpts = 1
    fig, ax = plt.subplots(nrows=1, ncols=1)

    if type(self).n_individuals > 1:
        xy = data.reshape(data.shape[0], n_individuals, n_kpts)[..., :2]
    else:
        xy = data.reshape(data.shape[0], n_kpts, -1)[..., :2]
    if events:
        for animal_id, (animal_name, object_dict) in enumerate(events.items()):
            if len(xy.shape) == 4:
                xy = xy[:, animal_id]
            for object_name, object_events in object_dict.items():
                self.plot_occupancy_heatmap_per_animal(
                    events=object_events,
                    xy=xy,
                    bodyparts=bodyparts,
                    kin_type=kin_type,
                    bins=bins,
                    cmap=cmap,
                    cmap_path=cmap_path,
                    color_contours=color_contours,
                    ax=ax,
                    padding=padding,
                )
    else:
        self.plot_occupancy_heatmap_per_animal(
            xy=xy,
            bodyparts=bodyparts,
            kin_type=kin_type,
            bins=bins,
            cmap=cmap,
            cmap_path=cmap_path,
            color_contours=color_contours,
            ax=ax,
            padding=padding,
        )

    return fig, ax


AnimalBehaviorAnalysis.plot_occupancy_heatmap_per_animal = (
    plot_occupancy_heatmap_per_animal
)
AnimalBehaviorAnalysis.plot_occupancy_heatmap = plot_occupancy_heatmap
