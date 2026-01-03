"""Visualization helpers for quantum walks."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple, Union

import numpy as np

_STYLE_APPLIED = False


def _apply_style(plt) -> None:
    global _STYLE_APPLIED
    if _STYLE_APPLIED:
        return
    plt.style.use("dark_background")
    plt.rcParams.update(
        {
            "figure.facecolor": "#0B0E14",
            "axes.facecolor": "#0B0E14",
            "axes.edgecolor": "#2E3440",
            "axes.labelcolor": "#D8DEE9",
            "xtick.color": "#C0C7D1",
            "ytick.color": "#C0C7D1",
            "text.color": "#E5E9F0",
            "grid.color": "#2E3440",
        }
    )
    _STYLE_APPLIED = True


def _require_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for visualization") from exc
    _apply_style(plt)
    return plt


def plot_grid_heatmap(
    probabilities_t: np.ndarray,
    width: int,
    height: int,
    title: str,
    output_path: Union[str, Path],
) -> Path:
    plt = _require_matplotlib()
    data = probabilities_t.reshape(height, width)

    fig, ax = plt.subplots()
    fig.patch.set_facecolor("#0B0E14")
    ax.set_facecolor("#0B0E14")
    image = ax.imshow(data, cmap="magma", origin="lower")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.set_aspect("equal")
    if title:
        ax.set_title(title)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_maze_heatmap(
    probabilities_t: np.ndarray,
    width: int,
    height: int,
    positions: Iterable[Iterable[int]],
    title: str,
    output_path: Union[str, Path],
    start: Optional[Tuple[int, int]] = None,
    goal: Optional[Tuple[int, int]] = None,
    path: Optional[List[Tuple[int, int]]] = None,
) -> Path:
    plt = _require_matplotlib()
    import matplotlib.patheffects as path_effects
    grid = np.full((height, width), np.nan, dtype=float)
    for idx, coord in enumerate(positions):
        x, y = coord
        grid[y, x] = probabilities_t[idx]

    masked = np.ma.masked_invalid(grid)
    cmap = plt.cm.magma.copy()
    cmap.set_bad(color="#0B0E14")

    fig, ax = plt.subplots()
    fig.patch.set_facecolor("#0B0E14")
    ax.set_facecolor("#0B0E14")
    image = ax.imshow(masked, cmap=cmap, origin="lower")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if title:
        ax.set_title(title)

    if path:
        xs = [pos[0] for pos in path]
        ys = [pos[1] for pos in path]
        ax.plot(
            xs,
            ys,
            color="#4CC9F0",
            linewidth=2.4,
            alpha=0.9,
            path_effects=[
                path_effects.Stroke(linewidth=5.0, foreground="#1F6FEB", alpha=0.4),
                path_effects.Normal(),
            ],
        )

    if start is not None:
        ax.scatter(
            [start[0]],
            [start[1]],
            c="#00C853",
            s=70,
            label="start",
            edgecolors="#E5E9F0",
            linewidths=1.2,
        )
    if goal is not None:
        ax.scatter(
            [goal[0]],
            [goal[1]],
            c="#FF5252",
            s=70,
            label="goal",
            edgecolors="#E5E9F0",
            linewidths=1.2,
        )
    if start is not None or goal is not None:
        ax.legend(loc="upper right", framealpha=0.85)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_maze_3d(
    probabilities_t: np.ndarray,
    width: int,
    height: int,
    depth: int,
    positions: Iterable[Iterable[int]],
    edges: Iterable[Tuple[int, int]],
    title: str,
    output_path: Union[str, Path],
    start_index: Optional[int] = None,
    goal_index: Optional[int] = None,
    path_indices: Optional[List[int]] = None,
    elev: float = 28.0,
    azim: float = 30.0,
) -> Path:
    plt = _require_matplotlib()
    fig = plt.figure(figsize=(7.5, 7.5))
    fig.patch.set_facecolor("#0B0E14")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#0B0E14")

    coords = np.array(list(positions), dtype=float)
    if coords.size == 0:
        raise ValueError("maze3d positions are empty")

    try:
        from mpl_toolkits.mplot3d.art3d import Line3DCollection

        segments = []
        for a, b in edges:
            segments.append([coords[a], coords[b]])
        edge_collection = Line3DCollection(segments, colors="#1F6FEB", linewidths=0.6, alpha=0.18)
        ax.add_collection3d(edge_collection)
    except Exception:
        for a, b in edges:
            ax.plot(
                [coords[a, 0], coords[b, 0]],
                [coords[a, 1], coords[b, 1]],
                [coords[a, 2], coords[b, 2]],
                color="#1F6FEB",
                linewidth=0.6,
                alpha=0.18,
            )

    probs = np.array(probabilities_t, dtype=float)
    max_prob = float(np.max(probs)) if probs.size else 0.0
    if max_prob <= 0:
        sizes = np.full(len(coords), 28.0)
    else:
        sizes = 28.0 + (probs / max_prob) * 200.0

    scatter = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        coords[:, 2],
        c=probs,
        s=sizes,
        cmap="viridis",
        marker="s",
        alpha=0.9,
        edgecolors="#0B0E14",
        linewidths=0.2,
        depthshade=True,
    )

    if path_indices:
        xs = [coords[idx, 0] for idx in path_indices]
        ys = [coords[idx, 1] for idx in path_indices]
        zs = [coords[idx, 2] for idx in path_indices]
        ax.plot(xs, ys, zs, color="#4CC9F0", linewidth=2.2, alpha=0.9)

    if start_index is not None:
        ax.scatter(
            [coords[start_index, 0]],
            [coords[start_index, 1]],
            [coords[start_index, 2]],
            c="#00C853",
            s=160,
            edgecolors="#E5E9F0",
            linewidths=1.0,
            label="start",
        )
    if goal_index is not None:
        ax.scatter(
            [coords[goal_index, 0]],
            [coords[goal_index, 1]],
            [coords[goal_index, 2]],
            c="#FF5252",
            s=160,
            edgecolors="#E5E9F0",
            linewidths=1.0,
            label="goal",
        )
    if start_index is not None or goal_index is not None:
        ax.legend(loc="upper right", framealpha=0.85)

    ax.view_init(elev=elev, azim=azim)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(-0.5, height - 0.5)
    ax.set_zlim(-0.5, depth - 0.5)
    try:
        ax.set_box_aspect((width, height, depth))
    except Exception:
        pass
    if title:
        ax.set_title(title)
    fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_maze3d_heatmap(
    probabilities_t: np.ndarray,
    width: int,
    height: int,
    depth: int,
    positions: Iterable[Iterable[int]],
    title: str,
    output_path: Union[str, Path],
    start_index: Optional[int] = None,
    goal_index: Optional[int] = None,
    elev: float = 28.0,
    azim: float = 30.0,
) -> Path:
    plt = _require_matplotlib()
    fig = plt.figure(figsize=(7.5, 7.5))
    fig.patch.set_facecolor("#0B0E14")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#0B0E14")

    grid = np.zeros((depth, height, width), dtype=float)
    coords = np.array(list(positions), dtype=int)
    if coords.size == 0:
        raise ValueError("maze3d positions are empty")
    for idx, (x, y, z) in enumerate(coords):
        grid[z, y, x] = probabilities_t[idx]

    max_prob = float(np.max(grid)) if grid.size else 0.0
    if max_prob <= 0:
        normalized = np.zeros_like(grid)
    else:
        normalized = grid / max_prob

    mask = normalized > 0
    if not np.any(mask):
        mask = np.ones_like(normalized, dtype=bool)

    cmap = plt.cm.magma
    colors = cmap(normalized)
    colors[..., 3] = 0.15 + 0.85 * normalized

    ax.voxels(mask, facecolors=colors, edgecolor="#0B0E14", linewidth=0.2)

    def center_of(index: int) -> Tuple[float, float, float]:
        x, y, z = coords[index]
        return x + 0.5, y + 0.5, z + 0.5

    if start_index is not None:
        sx, sy, sz = center_of(start_index)
        ax.scatter([sx], [sy], [sz], c="#00C853", s=140, edgecolors="#E5E9F0", linewidths=1.0, label="start")
    if goal_index is not None:
        gx, gy, gz = center_of(goal_index)
        ax.scatter([gx], [gy], [gz], c="#FF5252", s=140, edgecolors="#E5E9F0", linewidths=1.0, label="goal")
    if start_index is not None or goal_index is not None:
        ax.legend(loc="upper right", framealpha=0.85)

    ax.view_init(elev=elev, azim=azim)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_zlim(0, depth)
    try:
        ax.set_box_aspect((width, height, depth))
    except Exception:
        pass
    if title:
        ax.set_title(title)

    mappable = plt.cm.ScalarMappable(cmap=cmap)
    mappable.set_array(grid)
    fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_polar_heatmap(
    probabilities_t: np.ndarray,
    rings: int,
    sectors: int,
    title: str,
    output_path: Union[str, Path],
    edges: Optional[Iterable[Iterable[int]]] = None,
    start: Optional[Tuple[int, int]] = None,
    goal: Optional[Tuple[int, int]] = None,
    path: Optional[List[Tuple[int, int]]] = None,
) -> Path:
    plt = _require_matplotlib()
    from matplotlib.patches import Wedge
    import matplotlib.patheffects as path_effects
    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("#0B0E14")
    ax.set_facecolor("#0B0E14")

    max_prob = float(np.max(probabilities_t)) if probabilities_t.size else 1.0
    if max_prob <= 0:
        max_prob = 1.0
    norm = plt.Normalize(vmin=0.0, vmax=max_prob)
    cmap = plt.cm.magma

    edge_set = None
    if edges is not None:
        edge_set = set()
        for edge in edges:
            if len(edge) != 2:
                continue
            a, b = int(edge[0]), int(edge[1])
            edge_set.add((min(a, b), max(a, b)))

    def index_of(ring: int, sector: int) -> int:
        return ring * sectors + sector

    def has_edge(a: int, b: int) -> bool:
        if edge_set is None:
            return True
        return (min(a, b), max(a, b)) in edge_set

    for ring in range(rings):
        for sector in range(sectors):
            idx = ring * sectors + sector
            theta1 = (sector / sectors) * 360.0
            theta2 = ((sector + 1) / sectors) * 360.0
            color = cmap(norm(float(probabilities_t[idx])))
            wedge = Wedge(
                (0.0, 0.0),
                ring + 1.0,
                theta1,
                theta2,
                width=1.0,
                facecolor=color,
                edgecolor="none",
                linewidth=0.0,
            )
            ax.add_patch(wedge)

    if edge_set is not None:
        wall_color = "#7D828A"
        wall_width = 1.2
        for ring in range(rings):
            for sector in range(sectors):
                idx = index_of(ring, sector)
                theta1 = (sector / sectors) * 2 * np.pi
                theta2 = ((sector + 1) / sectors) * 2 * np.pi
                next_sector = (sector + 1) % sectors

                if ring == rings - 1 or not has_edge(idx, index_of(ring + 1, sector)):
                    radius = ring + 1.0
                    angles = np.linspace(theta1, theta2, 24)
                    xs = radius * np.cos(angles)
                    ys = radius * np.sin(angles)
                    ax.plot(xs, ys, color=wall_color, linewidth=wall_width, alpha=0.9)

                if not has_edge(idx, index_of(ring, next_sector)):
                    r1 = ring
                    r2 = ring + 1.0
                    x1, y1 = r1 * np.cos(theta2), r1 * np.sin(theta2)
                    x2, y2 = r2 * np.cos(theta2), r2 * np.sin(theta2)
                    ax.plot([x1, x2], [y1, y2], color=wall_color, linewidth=wall_width, alpha=0.9)
    else:
        for ring in range(1, rings + 1):
            circle = plt.Circle((0.0, 0.0), ring, fill=False, color="#2E3440", linewidth=0.6, alpha=0.6)
            ax.add_patch(circle)

    if path:
        xs = []
        ys = []
        for ring, sector in path:
            angle = (sector + 0.5) / sectors * 2 * np.pi
            radius = ring + 0.5
            xs.append(radius * np.cos(angle))
            ys.append(radius * np.sin(angle))
        ax.plot(
            xs,
            ys,
            color="#4CC9F0",
            linewidth=2.4,
            alpha=0.9,
            path_effects=[
                path_effects.Stroke(linewidth=5.0, foreground="#1F6FEB", alpha=0.4),
                path_effects.Normal(),
            ],
        )

    def plot_marker(coord: Tuple[int, int], color: str, label: str) -> None:
        ring, sector = coord
        angle = (sector + 0.5) / sectors * 2 * np.pi
        radius = ring + 0.5
        ax.scatter(
            [radius * np.cos(angle)],
            [radius * np.sin(angle)],
            c=color,
            s=70,
            label=label,
            edgecolors="#E5E9F0",
            linewidths=1.2,
        )

    if start is not None:
        plot_marker(start, "#00C853", "start")
    if goal is not None:
        plot_marker(goal, "#FF5252", "goal")
    if start is not None or goal is not None:
        ax.legend(loc="upper right", framealpha=0.85)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    if title:
        ax.set_title(title)

    limit = rings + 0.5
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_line(
    probabilities_t: np.ndarray,
    title: str,
    output_path: Union[str, Path],
) -> Path:
    plt = _require_matplotlib()

    fig, ax = plt.subplots()
    fig.patch.set_facecolor("#0B0E14")
    ax.set_facecolor("#0B0E14")
    ax.bar(range(len(probabilities_t)), probabilities_t, color="#4CC9F0", alpha=0.85)
    ax.set_xlabel("node")
    ax.set_ylabel("probability")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    if title:
        ax.set_title(title)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_hypercube_projection(
    probabilities_t: np.ndarray,
    dimensions: int,
    title: str,
    output_path: Union[str, Path],
    start_index: Optional[int] = None,
    goal_index: Optional[int] = None,
    path_indices: Optional[List[int]] = None,
) -> Path:
    if dimensions < 1:
        raise ValueError("dimensions must be positive")
    bits_x = (dimensions + 1) // 2
    bits_y = dimensions // 2
    width = 1 << bits_x
    height = 1 << bits_y

    grid = np.zeros((height, width), dtype=float)
    for idx, prob in enumerate(probabilities_t):
        x_bits = idx & ((1 << bits_x) - 1)
        y_bits = idx >> bits_x
        grid[y_bits, x_bits] = prob

    plt = _require_matplotlib()
    fig, ax = plt.subplots()
    fig.patch.set_facecolor("#0B0E14")
    ax.set_facecolor("#0B0E14")
    image = ax.imshow(grid, cmap="magma", origin="lower")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xlabel("x bits")
    ax.set_ylabel("y bits")
    ax.set_aspect("equal")
    if title:
        ax.set_title(title)

    def coord_for(index: int) -> Tuple[int, int]:
        x_bits = index & ((1 << bits_x) - 1)
        y_bits = index >> bits_x
        return x_bits, y_bits

    if path_indices:
        xs = []
        ys = []
        for index in path_indices:
            x, y = coord_for(index)
            xs.append(x)
            ys.append(y)
        ax.plot(xs, ys, color="#4CC9F0", linewidth=2.0, alpha=0.8)

    if start_index is not None:
        x, y = coord_for(start_index)
        ax.scatter([x], [y], c="#00C853", s=60, edgecolors="#E5E9F0", linewidths=1.0, label="start")
    if goal_index is not None:
        x, y = coord_for(goal_index)
        ax.scatter([x], [y], c="#FF5252", s=60, edgecolors="#E5E9F0", linewidths=1.0, label="goal")
    if start_index is not None or goal_index is not None:
        ax.legend(loc="upper right", framealpha=0.85)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_hypercube_graph(
    probabilities_t: np.ndarray,
    dimensions: int,
    title: str,
    output_path: Union[str, Path],
    start_index: Optional[int] = None,
    goal_index: Optional[int] = None,
    path_indices: Optional[List[int]] = None,
) -> Path:
    if dimensions < 1:
        raise ValueError("dimensions must be positive")

    nodes = 1 << dimensions
    indices = np.arange(nodes, dtype=int)
    bit_ids = np.arange(dimensions, dtype=int)
    bits = ((indices[:, None] >> bit_ids) & 1).astype(float)
    signs = bits * 2.0 - 1.0
    angles = np.linspace(0, 2 * math.pi, dimensions, endpoint=False)
    axes = np.stack((np.cos(angles), np.sin(angles)), axis=1)
    coords = signs @ axes
    coords /= max(1.0, math.sqrt(dimensions))

    plt = _require_matplotlib()
    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    fig.patch.set_facecolor("#0B0E14")
    ax.set_facecolor("#0B0E14")

    try:
        from matplotlib.collections import LineCollection

        segments = []
        for idx in range(nodes):
            for bit in range(dimensions):
                neighbor = idx ^ (1 << bit)
                if idx < neighbor:
                    segments.append([coords[idx], coords[neighbor]])
        edge_collection = LineCollection(segments, colors="#1F6FEB", linewidths=0.6, alpha=0.18)
        ax.add_collection(edge_collection)
    except Exception:
        for idx in range(nodes):
            for bit in range(dimensions):
                neighbor = idx ^ (1 << bit)
                if idx < neighbor:
                    ax.plot(
                        [coords[idx, 0], coords[neighbor, 0]],
                        [coords[idx, 1], coords[neighbor, 1]],
                        color="#1F6FEB",
                        linewidth=0.6,
                        alpha=0.18,
                    )

    probs = np.array(probabilities_t, dtype=float)
    max_prob = float(np.max(probs)) if probs.size else 0.0
    if max_prob <= 0:
        sizes = np.full(nodes, 40.0)
    else:
        sizes = 40.0 + (probs / max_prob) * 240.0
    scatter = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=probs,
        s=sizes,
        cmap="viridis",
        alpha=0.9,
        edgecolors="none",
    )

    if path_indices:
        xs = [coords[idx, 0] for idx in path_indices]
        ys = [coords[idx, 1] for idx in path_indices]
        ax.plot(xs, ys, color="#4CC9F0", linewidth=2.2, alpha=0.85)

    if start_index is not None:
        ax.scatter(
            [coords[start_index, 0]],
            [coords[start_index, 1]],
            c="#00C853",
            s=130,
            edgecolors="#E5E9F0",
            linewidths=1.0,
            label="start",
        )
    if goal_index is not None:
        ax.scatter(
            [coords[goal_index, 0]],
            [coords[goal_index, 1]],
            c="#FF5252",
            s=130,
            edgecolors="#E5E9F0",
            linewidths=1.0,
            label="goal",
        )
    if start_index is not None or goal_index is not None:
        ax.legend(loc="upper right", framealpha=0.85)

    fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    if title:
        ax.set_title(title)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_hypercube_3d(
    probabilities_t: np.ndarray,
    dimensions: int,
    title: str,
    output_path: Union[str, Path],
    start_index: Optional[int] = None,
    goal_index: Optional[int] = None,
    path_indices: Optional[List[int]] = None,
    elev: float = 28.0,
    azim: float = 30.0,
) -> Path:
    if dimensions < 1:
        raise ValueError("dimensions must be positive")

    nodes = 1 << dimensions
    indices = np.arange(nodes, dtype=int)
    bit_ids = np.arange(dimensions, dtype=int)
    bits = ((indices[:, None] >> bit_ids) & 1).astype(float)
    signs = bits * 2.0 - 1.0

    phi = (1 + 5**0.5) / 2
    angles = 2 * math.pi * bit_ids / phi
    z = 1 - 2 * (bit_ids + 0.5) / dimensions
    r = np.sqrt(np.clip(1 - z * z, 0, 1))
    basis = np.stack((r * np.cos(angles), r * np.sin(angles), z), axis=1)

    coords = signs @ basis
    coords /= max(1.0, math.sqrt(dimensions))

    plt = _require_matplotlib()
    fig = plt.figure(figsize=(7.5, 7.5))
    fig.patch.set_facecolor("#0B0E14")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#0B0E14")

    try:
        from mpl_toolkits.mplot3d.art3d import Line3DCollection

        segments = []
        for idx in range(nodes):
            for bit in range(dimensions):
                neighbor = idx ^ (1 << bit)
                if idx < neighbor:
                    segments.append([coords[idx], coords[neighbor]])
        edge_collection = Line3DCollection(segments, colors="#1F6FEB", linewidths=0.6, alpha=0.15)
        ax.add_collection3d(edge_collection)
    except Exception:
        for idx in range(nodes):
            for bit in range(dimensions):
                neighbor = idx ^ (1 << bit)
                if idx < neighbor:
                    ax.plot(
                        [coords[idx, 0], coords[neighbor, 0]],
                        [coords[idx, 1], coords[neighbor, 1]],
                        [coords[idx, 2], coords[neighbor, 2]],
                        color="#1F6FEB",
                        linewidth=0.6,
                        alpha=0.15,
                    )

    probs = np.array(probabilities_t, dtype=float)
    max_prob = float(np.max(probs)) if probs.size else 0.0
    if max_prob <= 0:
        sizes = np.full(nodes, 40.0)
    else:
        sizes = 40.0 + (probs / max_prob) * 220.0
    scatter = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        coords[:, 2],
        c=probs,
        s=sizes,
        cmap="viridis",
        alpha=0.9,
        edgecolors="none",
        depthshade=True,
    )

    if path_indices:
        xs = [coords[idx, 0] for idx in path_indices]
        ys = [coords[idx, 1] for idx in path_indices]
        zs = [coords[idx, 2] for idx in path_indices]
        ax.plot(xs, ys, zs, color="#4CC9F0", linewidth=2.0, alpha=0.85)

    if start_index is not None:
        ax.scatter(
            [coords[start_index, 0]],
            [coords[start_index, 1]],
            [coords[start_index, 2]],
            c="#00C853",
            s=150,
            edgecolors="#E5E9F0",
            linewidths=1.0,
            label="start",
        )
    if goal_index is not None:
        ax.scatter(
            [coords[goal_index, 0]],
            [coords[goal_index, 1]],
            [coords[goal_index, 2]],
            c="#FF5252",
            s=150,
            edgecolors="#E5E9F0",
            linewidths=1.0,
            label="goal",
        )
    if start_index is not None or goal_index is not None:
        ax.legend(loc="upper right", framealpha=0.85)

    ax.view_init(elev=elev, azim=azim)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    limit = float(np.max(np.abs(coords))) if coords.size else 1.0
    limit = max(limit, 1.0) * 1.05
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-limit, limit)
    if title:
        ax.set_title(title)
    fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_cube_projection(
    probabilities_t: np.ndarray,
    size: int,
    title: str,
    output_path: Union[str, Path],
) -> Path:
    if size < 2:
        raise ValueError("size must be >= 2")
    width = size * size
    height = size
    grid = np.zeros((height, width), dtype=float)

    for idx, prob in enumerate(probabilities_t):
        z, rem = divmod(idx, size * size)
        y, x = divmod(rem, size)
        gx = x + size * y
        grid[z, gx] = prob

    plt = _require_matplotlib()
    fig, ax = plt.subplots()
    fig.patch.set_facecolor("#0B0E14")
    ax.set_facecolor("#0B0E14")
    image = ax.imshow(grid, cmap="magma", origin="lower")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xlabel("x + size*y")
    ax.set_ylabel("z")
    ax.set_aspect("equal")
    if title:
        ax.set_title(title)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def render_grid_frames(
    probabilities: np.ndarray,
    width: int,
    height: int,
    output_dir: Union[str, Path],
    prefix: str = "frame",
    progress: Optional[Callable[[int, int], None]] = None,
) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    total = len(probabilities)
    for idx, probs in enumerate(probabilities):
        path = output_dir / f"{prefix}_{idx:03d}.png"
        plot_grid_heatmap(probs, width, height, f"step {idx}", path)
        frames.append(path)
        if progress:
            progress(idx + 1, total)
    return frames


def render_maze_frames(
    probabilities: np.ndarray,
    width: int,
    height: int,
    positions: Iterable[Iterable[int]],
    output_dir: Union[str, Path],
    prefix: str = "frame",
    progress: Optional[Callable[[int, int], None]] = None,
) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    total = len(probabilities)
    for idx, probs in enumerate(probabilities):
        path = output_dir / f"{prefix}_{idx:03d}.png"
        plot_maze_heatmap(probs, width, height, positions, f"step {idx}", path)
        frames.append(path)
        if progress:
            progress(idx + 1, total)
    return frames


def render_maze3d_frames(
    probabilities: np.ndarray,
    width: int,
    height: int,
    depth: int,
    positions: Iterable[Iterable[int]],
    edges: Iterable[Tuple[int, int]],
    output_dir: Union[str, Path],
    prefix: str = "frame",
    progress: Optional[Callable[[int, int], None]] = None,
    elev: float = 28.0,
    azim_start: float = 30.0,
    azim_end: float = 390.0,
) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    total = len(probabilities)
    positions_list = list(positions)
    edges_list = list(edges)
    for idx, probs in enumerate(probabilities):
        path = output_dir / f"{prefix}_{idx:03d}.png"
        if total > 1:
            azim = azim_start + (azim_end - azim_start) * (idx / (total - 1))
        else:
            azim = azim_start
        plot_maze_3d(
            probs,
            width,
            height,
            depth,
            positions_list,
            edges_list,
            f"step {idx}",
            path,
            elev=elev,
            azim=azim,
        )
        frames.append(path)
        if progress:
            progress(idx + 1, total)
    return frames


def render_polar_frames(
    probabilities: np.ndarray,
    rings: int,
    sectors: int,
    output_dir: Union[str, Path],
    edges: Optional[Iterable[Iterable[int]]] = None,
    prefix: str = "frame",
    progress: Optional[Callable[[int, int], None]] = None,
) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    total = len(probabilities)
    for idx, probs in enumerate(probabilities):
        path = output_dir / f"{prefix}_{idx:03d}.png"
        plot_polar_heatmap(probs, rings, sectors, f"step {idx}", path, edges=edges)
        frames.append(path)
        if progress:
            progress(idx + 1, total)
    return frames


def render_line_frames(
    probabilities: np.ndarray,
    output_dir: Union[str, Path],
    prefix: str = "frame",
    progress: Optional[Callable[[int, int], None]] = None,
) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    total = len(probabilities)
    for idx, probs in enumerate(probabilities):
        path = output_dir / f"{prefix}_{idx:03d}.png"
        plot_line(probs, f"step {idx}", path)
        frames.append(path)
        if progress:
            progress(idx + 1, total)
    return frames


def render_hypercube_frames(
    probabilities: np.ndarray,
    dimensions: int,
    output_dir: Union[str, Path],
    prefix: str = "frame",
    progress: Optional[Callable[[int, int], None]] = None,
) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    total = len(probabilities)
    for idx, probs in enumerate(probabilities):
        path = output_dir / f"{prefix}_{idx:03d}.png"
        plot_hypercube_projection(probs, dimensions, f"step {idx}", path)
        frames.append(path)
        if progress:
            progress(idx + 1, total)
    return frames


def render_hypercube_frames_3d(
    probabilities: np.ndarray,
    dimensions: int,
    output_dir: Union[str, Path],
    prefix: str = "frame",
    progress: Optional[Callable[[int, int], None]] = None,
    elev: float = 28.0,
    azim_start: float = 30.0,
    azim_end: float = 390.0,
) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    total = len(probabilities)
    for idx, probs in enumerate(probabilities):
        path = output_dir / f"{prefix}_{idx:03d}.png"
        if total > 1:
            azim = azim_start + (azim_end - azim_start) * (idx / (total - 1))
        else:
            azim = azim_start
        plot_hypercube_3d(probs, dimensions, f"step {idx}", path, elev=elev, azim=azim)
        frames.append(path)
        if progress:
            progress(idx + 1, total)
    return frames


def render_cube_frames(
    probabilities: np.ndarray,
    size: int,
    output_dir: Union[str, Path],
    prefix: str = "frame",
    progress: Optional[Callable[[int, int], None]] = None,
) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    total = len(probabilities)
    for idx, probs in enumerate(probabilities):
        path = output_dir / f"{prefix}_{idx:03d}.png"
        plot_cube_projection(probs, size, f"step {idx}", path)
        frames.append(path)
        if progress:
            progress(idx + 1, total)
    return frames


def assemble_gif(frames: list[Path], output_path: Union[str, Path], fps: int = 6) -> Path:
    try:
        import imageio.v3 as imageio
    except ImportError as exc:
        raise RuntimeError("imageio is required for GIF export") from exc

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    images = [imageio.imread(frame) for frame in frames]
    imageio.imwrite(output_path, images, duration=1 / fps, loop=0)
    return output_path
