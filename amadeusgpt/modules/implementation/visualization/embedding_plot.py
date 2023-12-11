import matplotlib.pyplot as plt
import numpy as np
from amadeusgpt.utils import filter_kwargs_for_function

params = {
    "axes.labelsize": 10,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "text.usetex": False,
    "figure.figsize": [4.5, 4.5],
    "font.size": 10,
}

from amadeusgpt.implementation import AnimalBehaviorAnalysis, Event
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.rcParams.update(params)
plt.tight_layout()


def plot_neural_behavior_embedding(
    self, embedding, color_by_object=False, color_by_time=False, **kwargs
):
    behavior_analysis = AnimalBehaviorAnalysis()
    # it is likely that the user is expecting the embedding colored by objects
    c = None
    object_names = []

    if color_by_object and color_by_time:
        color_by_object = False
        color_by_time = True

    if not color_by_object and not color_by_time:
        color_by_time = True

    if color_by_object:
        c = np.zeros(len(AnimalBehaviorAnalysis.get_keypoints()), dtype=int)
        # object_names = [
        #     "water",
        #     "igloo",
        #     "tower",
        #     "cotton",
        #     "tunnel",
        #     "barrel",
        #     "food",
        #     "tread",
        # ]
        object_names = AnimalBehaviorAnalysis.get_seg_object_names()
        if len(object_names) > 1:
            print("not supposed to be here")
            for object_id, object_name in enumerate(object_names):
                temp = np.zeros(len(AnimalBehaviorAnalysis.get_keypoints()), dtype=bool)
                # let's only care about overlap event
                object_events = behavior_analysis.animals_object_events(
                    object_name, "overlap"
                )
                temp |= Event.events2onemask(object_events)
                c[temp] = object_id

    elif color_by_time:
        c = np.arange(len(AnimalBehaviorAnalysis.get_keypoints()), dtype=int)

    fig = plt.figure(figsize=(5, 5))
    # c[:8] = np.arange(8)
    print("np unique", np.unique(c))

    # name_dict = {
    #     "water": "water basin",
    #     "igloo": "mouse hut",
    #     "tower": "elevated platform",
    #     "cotton": "cotton roll",
    #     "tunnel": "half tube",
    #     "barrel": "platform barrel",
    #     "food": "food pellet",
    #     "tread": "treadmill",
    # }
    # new_names = [name_dict[name] for name in object_names]
    # label = [new_names[idx] for idx in c]

    if "cmap" in kwargs:
        cmap = kwargs["cmap"]
    elif "colormap" in kwargs:
        cmap = kwargs["colormap"]
    else:
        cmap = "rainbow"

    if embedding.shape[1] == 2:
        ax = plt.subplot(111)
        scatter = ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=c,
            cmap=cmap,
            s=0.05,
        )

    elif embedding.shape[1] == 3:
        ax = plt.subplot(111, projection="3d")
        scatter = ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            embedding[:, 2],
            c=c,
            cmap=cmap,
            s=kwargs.pop("s", 0.05),
        )
    else:
        raise ValueError(f"Dimension {embedding.shape[1]} not supported.")
    ax.axis("off")
    # ax.legend(markerscale=15, loc="lower right", bbox_to_anchor=(1.4, 0.5))
    # ax.legend()
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    ax_pos = ax.get_position()
    cbar_ax = fig.add_axes([ax_pos.x1 + 0.02, ax_pos.y0, 0.02, ax_pos.height])
    cbar = plt.colorbar(scatter, orientation="vertical", cax=cbar_ax)
    if len(object_names) > 1:
        cbar.set_label("Objects")
    else:
        cbar.set_label("Time")
    # if embedding.shape[1] == 3:
    #     for i in range(20):
    #         ax.view_init(elev=i * 10, azim=90)
    #         plt.savefig(
    #             f"embedding_plot{i}.png", transparent=True, dpi=300, bbox_inches="tight"
    #         )
    # else:
    #     plt.savefig(
    #         "embedding_plot.png", transparent=True, dpi=300, bbox_inches="tight"
    #     )
    plot_info = "Embedding plot is colored by the object the animal interacts with"
    return fig, ax, plot_info


AnimalBehaviorAnalysis.plot_neural_behavior_embedding = plot_neural_behavior_embedding
