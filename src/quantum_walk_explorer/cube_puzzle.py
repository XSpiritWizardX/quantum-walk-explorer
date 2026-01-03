"""Dynamic cube puzzle (3x3x3) utilities."""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

import numpy as np

from .maze import MazeError


def cube_index(x: int, y: int, z: int, size: int = 3) -> int:
    return z * (size * size) + y * size + x


def cube_coords(index: int, size: int = 3) -> Tuple[int, int, int]:
    z, rem = divmod(index, size * size)
    y, x = divmod(rem, size)
    return x, y, z


def cube_adjacency(size: int = 3) -> np.ndarray:
    if size < 2:
        raise MazeError("cube size must be >= 2")
    nodes = size ** 3
    adjacency = np.zeros((nodes, nodes), dtype=float)
    for z in range(size):
        for y in range(size):
            for x in range(size):
                idx = cube_index(x, y, z, size)
                for dx, dy, dz in [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]:
                    nx, ny, nz = x + dx, y + dy, z + dz
                    if 0 <= nx < size and 0 <= ny < size and 0 <= nz < size:
                        neighbor = cube_index(nx, ny, nz, size)
                        adjacency[idx, neighbor] = 1.0
    return adjacency


def rotation_permutation(
    size: int,
    axis: str,
    layer: int,
    direction: int,
) -> List[int]:
    if axis not in ("x", "y", "z"):
        raise MazeError("axis must be x, y, or z")
    if not (0 <= layer < size):
        raise MazeError("layer out of bounds")
    if direction not in (-1, 1):
        raise MazeError("direction must be -1 or 1")

    n = size - 1
    perm = list(range(size ** 3))

    for idx in range(size ** 3):
        x, y, z = cube_coords(idx, size)
        if axis == "x" and x == layer:
            if direction == 1:
                ny, nz = n - z, y
            else:
                ny, nz = z, n - y
            new_idx = cube_index(x, ny, nz, size)
            perm[new_idx] = idx
        elif axis == "y" and y == layer:
            if direction == 1:
                nx, nz = z, n - x
            else:
                nx, nz = n - z, x
            new_idx = cube_index(nx, y, nz, size)
            perm[new_idx] = idx
        elif axis == "z" and z == layer:
            if direction == 1:
                nx, ny = n - y, x
            else:
                nx, ny = y, n - x
            new_idx = cube_index(nx, ny, z, size)
            perm[new_idx] = idx

    return perm


def dynamic_cube_adjacencies(
    size: int,
    steps: int,
    seed: Optional[int] = None,
    shift_rate: float = 0.3,
) -> Tuple[List[np.ndarray], List[Dict[str, int]]]:
    if steps <= 0:
        raise MazeError("steps must be positive")
    if not (0.0 <= shift_rate <= 1.0):
        raise MazeError("shift_rate must be between 0 and 1")

    rng = random.Random(seed)
    base = cube_adjacency(size)
    nodes = base.shape[0]
    order = list(range(nodes))
    adjacencies: List[np.ndarray] = []
    rotations: List[Dict[str, int]] = []

    for _ in range(steps):
        if rng.random() < shift_rate:
            axis = rng.choice(["x", "y", "z"])
            layer = rng.randint(0, size - 1)
            direction = rng.choice([-1, 1])
            perm = rotation_permutation(size, axis, layer, direction)
            order = [order[i] for i in perm]
            rotations.append({"axis": axis, "layer": layer, "direction": direction})
        else:
            rotations.append({"axis": -1, "layer": -1, "direction": 0})
        adjacencies.append(base[np.ix_(order, order)])

    return adjacencies, rotations
