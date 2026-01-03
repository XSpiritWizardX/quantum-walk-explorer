"""Graph builders for quantum walk experiments."""

from __future__ import annotations

import numpy as np


def grid_graph(width: int, height: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")

    n = width * height
    adj = np.zeros((n, n), dtype=float)
    positions: list[tuple[int, int]] = []

    def idx(x: int, y: int) -> int:
        return y * width + x

    for y in range(height):
        for x in range(width):
            i = idx(x, y)
            positions.append((x, y))
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height:
                    j = idx(nx, ny)
                    adj[i, j] = 1.0

    return adj, positions


def line_graph(nodes: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
    if nodes <= 1:
        raise ValueError("nodes must be >= 2")

    adj = np.zeros((nodes, nodes), dtype=float)
    for i in range(nodes - 1):
        adj[i, i + 1] = 1.0
        adj[i + 1, i] = 1.0

    positions = [(i, 0) for i in range(nodes)]
    return adj, positions


def from_edge_list(nodes: int, edges: list[tuple[int, int]]) -> np.ndarray:
    if nodes <= 0:
        raise ValueError("nodes must be positive")

    adj = np.zeros((nodes, nodes), dtype=float)
    for a, b in edges:
        if not (0 <= a < nodes and 0 <= b < nodes):
            raise ValueError("edge index out of bounds")
        if a == b:
            continue
        adj[a, b] = 1.0
        adj[b, a] = 1.0

    return adj
