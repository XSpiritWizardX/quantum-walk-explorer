"""Hypercube maze utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from .maze import MazeError


def hypercube_adjacency(dimensions: int) -> np.ndarray:
    nodes = 1 << dimensions
    adjacency = np.zeros((nodes, nodes), dtype=float)
    for node in range(nodes):
        for bit in range(dimensions):
            neighbor = node ^ (1 << bit)
            adjacency[node, neighbor] = 1.0
    return adjacency


def permute_adjacency(adjacency: np.ndarray, permutation: List[int]) -> np.ndarray:
    perm = np.array(permutation, dtype=int)
    return adjacency[np.ix_(perm, perm)]


def dynamic_hypercube_adjacencies(
    dimensions: int,
    steps: int,
    seed: Optional[int] = None,
    shift_rate: float = 0.2,
) -> List[np.ndarray]:
    if steps <= 0:
        raise MazeError("steps must be positive")
    if not (0.0 <= shift_rate <= 1.0):
        raise MazeError("shift_rate must be between 0 and 1")

    rng = np.random.default_rng(seed)
    base = hypercube_adjacency(dimensions)
    nodes = base.shape[0]
    permutation = list(range(nodes))
    adjacencies = []

    for _ in range(steps):
        if rng.random() < shift_rate:
            a, b = rng.choice(nodes, size=2, replace=False)
            permutation[a], permutation[b] = permutation[b], permutation[a]
        adjacencies.append(permute_adjacency(base, permutation))

    return adjacencies


def generate_hypercube(dimensions: int) -> dict:
    if dimensions < 1:
        raise MazeError("dimensions must be >= 1")
    if dimensions > 12:
        raise MazeError("dimensions too large for interactive use")

    nodes = 1 << dimensions
    edges: List[Tuple[int, int]] = []
    for node in range(nodes):
        for bit in range(dimensions):
            neighbor = node ^ (1 << bit)
            if neighbor > node:
                edges.append((node, neighbor))

    labels = [format(i, f"0{dimensions}b") for i in range(nodes)]

    return {
        "type": "hypercube",
        "dimensions": dimensions,
        "nodes": nodes,
        "edges": [[a, b] for a, b in edges],
        "labels": labels,
        "start_index": 0,
        "goal_index": nodes - 1,
    }


def write_hypercube(path: Union[str, Path], data: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
    return path


def load_hypercube(path: Union[str, Path]) -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if data.get("type") != "hypercube":
        raise MazeError("not a hypercube file")

    dimensions = data.get("dimensions")
    edges = data.get("edges")
    labels = data.get("labels")
    start_index = data.get("start_index", 0)
    goal_index = data.get("goal_index")

    if not isinstance(dimensions, int) or dimensions < 1:
        raise MazeError("invalid dimensions")
    nodes = 1 << dimensions

    if not isinstance(edges, list):
        raise MazeError("edges must be a list")
    if not isinstance(labels, list) or len(labels) != nodes:
        raise MazeError("labels must be a list matching node count")

    if goal_index is None:
        goal_index = nodes - 1

    if not (0 <= start_index < nodes and 0 <= goal_index < nodes):
        raise MazeError("start/goal index out of bounds")

    for edge in edges:
        if not (isinstance(edge, list) and len(edge) == 2):
            raise MazeError("edges must be pairs")
        a, b = edge
        if not (isinstance(a, int) and isinstance(b, int)):
            raise MazeError("edge indices must be integers")
        if not (0 <= a < nodes and 0 <= b < nodes):
            raise MazeError("edge index out of bounds")

    return {
        "type": "hypercube",
        "dimensions": dimensions,
        "nodes": nodes,
        "edges": edges,
        "labels": labels,
        "start_index": start_index,
        "goal_index": goal_index,
    }


def load_hypercube_graph(
    path: Union[str, Path],
) -> Tuple[np.ndarray, dict, Dict[int, int]]:
    data = load_hypercube(path)
    nodes = data["nodes"]

    adjacency = np.zeros((nodes, nodes), dtype=float)
    for a, b in data["edges"]:
        adjacency[a, b] = 1.0
        adjacency[b, a] = 1.0

    positions = list(data["labels"])
    index_map = {idx: idx for idx in range(nodes)}

    graph_info = {
        "type": "hypercube",
        "dimensions": data["dimensions"],
        "nodes": nodes,
        "positions": positions,
        "start_index": data["start_index"],
        "goal_index": data["goal_index"],
    }

    return adjacency, graph_info, index_map
