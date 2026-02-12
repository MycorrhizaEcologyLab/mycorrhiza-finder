"""Training history plotting functionality."""

import io

import matplotlib
import numpy as np
from numpy.typing import NDArray

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def initialize() -> None:
    """Define graph style."""
    plt.style.use("classic")


def draw(
    history: dict[str, list[float]],
    epochs: int,
    title: str,
    x_range: NDArray[np.int_],
    t_name: str,
    v_name: str,
) -> io.BytesIO:
    """Plot a particular quantity for the training and validation sets.

    Args:
        history: Dictionary of training history containing entries under `t_name` for
            training and `v_name` for validation.
        title: Plot title.
        epochs: Number of epochs, used only for setting x-axis limits.
        x_range: Range of x values (epochs) that history covers.
        t_name: Name of the training set entry in `history`.
        v_name: Name of the validation set entry in `history`.

    Returns: In-memory PNG image of the plot.
    """
    plt.clf()

    plt.grid(True)

    t_values = history[t_name]
    v_values = history[v_name]
    plt.plot(x_range, t_values, "b-o", label="Training set")
    plt.plot(x_range, v_values, "g-s", label="Validation set")

    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title(title)

    padding = 0.1
    legend_pos = "upper right"

    if title[0:4] == "Loss":
        plt.xlim(-padding, epochs + padding)

    else:
        legend_pos = "lower right"
        plt.axis([-padding, epochs + padding, 0, 1])

    axes = plt.gca()
    axes.autoscale(enable=True, axis="x", tight=False)
    axes.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.legend(loc=legend_pos)
    plt.draw()

    plot_data = io.BytesIO()
    plt.savefig(plot_data, format="png")

    return plot_data
