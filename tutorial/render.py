"""Render the Schelling trajectory saved by inference.py.

Run from the project directory (defaults match inference.py):
    python render.py

Or specify the inputs and output explicitly:
    python render.py --colors render_data/color_array.npy \
        --proportions render_data/neighborhood_proportions.npy \
        --output render_data/schelling.mp4 --fps 30

Dependencies: numpy, matplotlib, and FFmpeg for MP4 output.
For GIF output, use --output render_data/schelling.gif (requires Pillow).
JAX, ABMAX, and model_final.py do not need to be imported to render these files.

color_array has shape (time_steps, NUM_AGENTS), in cell-ID order:
    +1 = blue agent, -1 = green agent, 0 = empty cell.
neighborhood_proportions has shape (time_steps, 4), ordered:
    [blue area, green area, mixed area, none area].
These are neighborhood classifications, not counts of agent colors. "None"
includes both empty cells and boundary cells removed by the model's metric.

inference.py saves after each update, so the first frame is simulation step 1.
The footer reports fitness for this entire saved trajectory using the final
FITNESS_WINDOW steps. model_final.py averages this same quantity over scenarios
when evaluating a GP candidate; these two files contain one scenario.
"""

from __future__ import annotations

import argparse
from math import isqrt
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Video generation also works without a display.

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter, writers
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
import numpy as np


FITNESS_WINDOW = 50  # Same default as model_final.py and inference.py.
BLUE = "#2563eb"
GREEN = "#16a34a"
EMPTY = "#ffffff"


def _validate_proportions(neighborhood_proportions):
    proportions = np.asarray(neighborhood_proportions, dtype=np.float64)
    if proportions.ndim != 2 or proportions.shape[1] != 4 or not len(proportions):
        raise ValueError("neighborhood_proportions must have shape (time_steps, 4).")
    if not np.isfinite(proportions).all():
        raise ValueError("neighborhood_proportions contains non-finite values.")
    if np.any((proportions < 0) | (proportions > 1)):
        raise ValueError("Neighborhood proportions must be between 0 and 1.")
    if not np.allclose(proportions.sum(axis=1), 1.0, rtol=0, atol=1e-5):
        raise ValueError("The four neighborhood proportions must sum to 1 at every step.")
    return proportions


def _validate_render_data(color_array, neighborhood_proportions):
    colors = np.asarray(color_array)
    proportions = _validate_proportions(neighborhood_proportions)
    if colors.ndim != 2 or not all(colors.shape):
        raise ValueError("color_array must have shape (time_steps, NUM_AGENTS).")
    if colors.shape[0] != proportions.shape[0]:
        raise ValueError("The two arrays must contain the same number of time steps.")
    if not np.isin(colors, [-1.0, 0.0, 1.0]).all():
        raise ValueError("color_array must contain only -1 (green), 0 (empty), or 1 (blue).")
    return colors, proportions


def load_render_data(color_path, proportions_path):
    """Load numeric .npy files, including files produced by jnp.save."""
    colors = np.load(color_path, allow_pickle=False)
    proportions = np.load(proportions_path, allow_pickle=False)
    return _validate_render_data(colors, proportions)


def cell_ids_to_xy(num_agents, grid_size=None):
    """Match ABMAX's cell_id_to_XY, used in SchellingAgent.set_neighbor_ids."""
    if grid_size is None:
        grid_size = isqrt(num_agents)
    if grid_size <= 0 or grid_size * grid_size != num_agents:
        raise ValueError("NUM_AGENTS must equal GRID_SIZE squared.")
    ids = np.arange(num_agents)
    return ids % grid_size, ids // grid_size, grid_size


def compute_trajectory_fitness(neighborhood_proportions, fitness_window=FITNESS_WINDOW):
    """Compute model_final.py's negative mean C-index for one trajectory.

    For each retained time step, C = min(blue, green, mixed) / their sum.
    When their sum is zero, C is zero. The fourth (none) column is excluded.
    If fewer than fitness_window steps were saved, all available steps are used,
    matching the model's [-FITNESS_WINDOW:] slice.
    """
    if fitness_window <= 0:
        raise ValueError("fitness_window must be positive.")
    proportions = _validate_proportions(neighborhood_proportions)
    final_window = proportions[-fitness_window:, :3]
    retained_area = final_window.sum(axis=-1)
    c_index = np.divide(
        final_window.min(axis=-1),
        retained_area,
        out=np.zeros_like(retained_area),
        where=retained_area > 0.0,
    )
    return -float(c_index.mean())


def create_animation(
    color_array,
    neighborhood_proportions,
    *,
    grid_size=None,
    fitness_window=FITNESS_WINDOW,
    fps=30,
):
    """Create the grid animation and return (figure, animation, fitness)."""
    colors, proportions = _validate_render_data(color_array, neighborhood_proportions)
    if fps <= 0:
        raise ValueError("fps must be positive.")
    num_steps, num_agents = colors.shape
    xs, ys, grid_size = cell_ids_to_xy(num_agents, grid_size)
    fitness = compute_trajectory_fitness(proportions, fitness_window)
    window_length = min(fitness_window, num_steps)

    # Grid arrays are indexed [Y, X]; plotting uses X horizontally, Y vertically.
    # Only colors change. Cell coordinates remain fixed throughout the episode.
    grid = np.empty((grid_size, grid_size), dtype=colors.dtype)
    grid[ys, xs] = colors[0]
    cmap = ListedColormap([GREEN, EMPTY, BLUE])
    norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)

    fig, ax = plt.subplots(figsize=(8.4, 9.4))
    fig.subplots_adjust(left=0.09, right=0.95, bottom=0.23, top=0.83)
    image = ax.imshow(grid, cmap=cmap, norm=norm, origin="lower", interpolation="nearest")
    ax.set_aspect("equal")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    tick_step = max(1, grid_size // 10)
    ax.set_xticks(np.arange(0, grid_size, tick_step))
    ax.set_yticks(np.arange(0, grid_size, tick_step))
    boundaries = np.arange(-0.5, grid_size, 1.0)
    ax.set_xticks(boundaries, minor=True)
    ax.set_yticks(boundaries, minor=True)
    ax.grid(which="minor", color="#cbd5e1", linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    title = ax.set_title("", fontsize=12, pad=12, linespacing=1.5)

    fig.legend(
        handles=[
            Patch(facecolor=BLUE, label="Blue agent (+1)"),
            Patch(facecolor=GREEN, label="Green agent (-1)"),
            Patch(facecolor=EMPTY, edgecolor="#94a3b8", label="Empty cell (0)"),
        ],
        loc="lower center", bbox_to_anchor=(0.5, 0.155), ncol=3, frameon=False,
    )
    fig.text(
        0.5, 0.11,
        f"Trajectory fitness: {fitness:.6f}  (final {window_length} steps)",
        ha="center", fontsize=12,
    )
    fig.text(
        0.5, 0.075,
        "Fitness = -mean(C); C = min(blue, green, mixed) / (blue + green + mixed)",
        ha="center", fontsize=9,
    )
    fig.text(
        0.5, 0.04, 'Area "none" includes empty cells and removed boundary cells.',
        ha="center", fontsize=9, color="#475569",
    )

    def update(frame):
        grid[ys, xs] = colors[frame]
        image.set_data(grid)
        blue, green, mixed, none = proportions[frame]
        title.set_text(
            f"Schelling model | step {frame + 1}/{num_steps}\n"
            f"Blue area: {blue:.4f}    Green area: {green:.4f}\n"
            f"Mixed area: {mixed:.4f}    None area: {none:.4f}"
        )
        return image, title

    update(0)
    animation = FuncAnimation(
        fig, update, frames=num_steps, interval=1000.0 / fps,
        blit=False, repeat=False, cache_frame_data=False,
    )
    return fig, animation, fitness


def render_trajectory(
    color_array,
    neighborhood_proportions,
    filename="render_data/schelling.mp4",
    *,
    grid_size=None,
    fitness_window=FITNESS_WINDOW,
    fps=30,
    dpi=120,
):
    """Save an MP4 or GIF; return the trajectory fitness shown in the video."""
    output_path = Path(filename)
    if not output_path.suffix:
        output_path = output_path.with_suffix(".mp4")
    if dpi <= 0:
        raise ValueError("dpi must be positive.")
    if output_path.suffix.lower() == ".mp4":
        if not writers.is_available("ffmpeg"):
            raise RuntimeError(
                "FFmpeg is needed for MP4 output. Install ffmpeg or choose a .gif output."
            )
        writer = FFMpegWriter(
            fps=fps, codec="libx264",
            extra_args=["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-pix_fmt", "yuv420p"],
            metadata={"title": "Schelling model trajectory"},
        )
    elif output_path.suffix.lower() == ".gif":
        writer = PillowWriter(fps=fps)
    else:
        raise ValueError("The output filename must end in .mp4 or .gif.")

    fig, animation, fitness = create_animation(
        color_array, neighborhood_proportions, grid_size=grid_size,
        fitness_window=fitness_window, fps=fps,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        animation.save(str(output_path), writer=writer, dpi=dpi)
    finally:
        plt.close(fig)
    return fitness


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--colors", type=Path, default=Path("render_data/color_array.npy"))
    parser.add_argument("--proportions", type=Path, default=Path("render_data/neighborhood_proportions.npy"))
    parser.add_argument("-o", "--output", type=Path, default=Path("render_data/schelling.mp4"))
    parser.add_argument("--grid-size", type=int, default=None, help="Infer from NUM_AGENTS by default (400 -> 20).")
    parser.add_argument("--fitness-window", type=int, default=FITNESS_WINDOW, help="Final steps used for fitness (default: 50).")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--dpi", type=int, default=120)
    args = parser.parse_args()

    try:
        colors, proportions = load_render_data(args.colors, args.proportions)
        fitness = render_trajectory(
            colors, proportions, args.output, grid_size=args.grid_size,
            fitness_window=args.fitness_window, fps=args.fps, dpi=args.dpi,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"Rendering failed: {exc}\n")

    output_path = args.output if args.output.suffix else args.output.with_suffix(".mp4")
    print(f"Saved {output_path} ({colors.shape[0]} frames); trajectory fitness = {fitness:.6f}")


if __name__ == "__main__":
    main()
