"""Maze parsing and solving helpers."""

from __future__ import annotations

from collections import deque
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

OPEN_CHARS = {".", "S", "G", " "}
WALL_CHAR = "#"


class MazeError(ValueError):
    pass


def _otsu_threshold(values: np.ndarray) -> int:
    histogram, _ = np.histogram(values.ravel(), bins=256, range=(0, 256))
    total = values.size
    sum_total = np.dot(np.arange(256), histogram)

    sum_b = 0.0
    weight_b = 0.0
    max_var = 0.0
    threshold = 128

    for i in range(256):
        weight_b += histogram[i]
        if weight_b == 0:
            continue
        weight_f = total - weight_b
        if weight_f == 0:
            break
        sum_b += i * histogram[i]
        mean_b = sum_b / weight_b
        mean_f = (sum_total - sum_b) / weight_f
        var_between = weight_b * weight_f * (mean_b - mean_f) ** 2
        if var_between > max_var:
            max_var = var_between
            threshold = i

    return int(threshold)


def _cleanup_mask(open_mask: np.ndarray, min_component: int) -> np.ndarray:
    if min_component <= 1:
        return open_mask

    height, width = open_mask.shape
    visited = np.zeros_like(open_mask, dtype=bool)
    cleaned = open_mask.copy()

    for y in range(height):
        for x in range(width):
            if not open_mask[y, x] or visited[y, x]:
                continue
            stack = [(x, y)]
            component = []
            visited[y, x] = True
            while stack:
                cx, cy = stack.pop()
                component.append((cx, cy))
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < width and 0 <= ny < height:
                        if open_mask[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((nx, ny))
            if len(component) < min_component:
                for cx, cy in component:
                    cleaned[cy, cx] = False

    return cleaned


def read_maze(path: Union[str, Path]) -> List[str]:
    text = Path(path).read_text(encoding="utf-8")
    lines = [line.rstrip("\n") for line in text.splitlines() if line.strip()]
    if not lines:
        raise MazeError("maze file is empty")
    width = len(lines[0])
    if width == 0:
        raise MazeError("maze width is zero")
    for line in lines:
        if len(line) != width:
            raise MazeError("maze lines must be the same width")
    return lines


def build_maze_graph(
    rows: List[str],
) -> Tuple[np.ndarray, dict, Dict[Tuple[int, int], int]]:
    if not rows:
        raise MazeError("maze file is empty")
    width = len(rows[0])
    if width == 0:
        raise MazeError("maze width is zero")
    for line in rows:
        if len(line) != width:
            raise MazeError("maze lines must be the same width")

    height = len(rows)

    open_cells: List[Tuple[int, int]] = []
    start: Optional[Tuple[int, int]] = None
    goal: Optional[Tuple[int, int]] = None

    for y, row in enumerate(rows):
        for x, char in enumerate(row):
            if char in OPEN_CHARS:
                open_cells.append((x, y))
                if char == "S":
                    if start is not None:
                        raise MazeError("maze has multiple start cells")
                    start = (x, y)
                elif char == "G":
                    if goal is not None:
                        raise MazeError("maze has multiple goal cells")
                    goal = (x, y)
            elif char != WALL_CHAR:
                raise MazeError(f"invalid maze character: {char}")

    if start is None or goal is None:
        raise MazeError("maze requires S (start) and G (goal)")

    coord_to_index = {coord: idx for idx, coord in enumerate(open_cells)}
    adjacency = np.zeros((len(open_cells), len(open_cells)), dtype=float)

    for idx, (x, y) in enumerate(open_cells):
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            neighbor = (nx, ny)
            if neighbor in coord_to_index:
                adjacency[idx, coord_to_index[neighbor]] = 1.0

    graph_info = {
        "type": "maze",
        "width": width,
        "height": height,
        "positions": open_cells,
        "maze": rows,
        "start_xy": list(start),
        "goal_xy": list(goal),
    }

    return adjacency, graph_info, coord_to_index


def build_maze3d_graph(
    width: int,
    height: int,
    depth: int,
    edges: Iterable[Tuple[int, int]],
    start_xyz: Tuple[int, int, int],
    goal_xyz: Tuple[int, int, int],
) -> Tuple[np.ndarray, dict, Dict[Tuple[int, int, int], int]]:
    if width < 1 or height < 1 or depth < 1:
        raise MazeError("maze3d dimensions must be >= 1")

    total_nodes = width * height * depth
    positions: List[Tuple[int, int, int]] = []
    coord_to_index: Dict[Tuple[int, int, int], int] = {}
    index = 0
    for z in range(depth):
        for y in range(height):
            for x in range(width):
                positions.append((x, y, z))
                coord_to_index[(x, y, z)] = index
                index += 1

    adjacency = np.zeros((total_nodes, total_nodes), dtype=float)
    normalized_edges: List[Tuple[int, int]] = []
    for edge in edges:
        a, b = edge
        if not (0 <= a < total_nodes and 0 <= b < total_nodes):
            raise MazeError("maze3d edge index out of bounds")
        adjacency[a, b] = 1.0
        adjacency[b, a] = 1.0
        normalized_edges.append((a, b))

    if start_xyz not in coord_to_index:
        raise MazeError("maze3d start position out of bounds")
    if goal_xyz not in coord_to_index:
        raise MazeError("maze3d goal position out of bounds")
    if start_xyz == goal_xyz:
        raise MazeError("maze3d start and goal must differ")

    graph_info = {
        "type": "maze3d",
        "width": width,
        "height": height,
        "depth": depth,
        "positions": positions,
        "edges": normalized_edges,
        "start_xyz": list(start_xyz),
        "goal_xyz": list(goal_xyz),
    }

    return adjacency, graph_info, coord_to_index


def load_maze_graph(
    path: Union[str, Path],
) -> Tuple[np.ndarray, dict, Dict[Tuple[int, int], int]]:
    rows = read_maze(path)
    return build_maze_graph(rows)


def generate_maze(
    cell_width: int,
    cell_height: int,
    seed: Optional[int] = None,
) -> List[str]:
    if cell_width < 2 or cell_height < 2:
        raise MazeError("maze width/height must be >= 2 to place start and goal")

    rnd = random.Random(seed)
    grid_width = cell_width * 2 + 1
    grid_height = cell_height * 2 + 1
    grid = [[WALL_CHAR for _ in range(grid_width)] for _ in range(grid_height)]

    def carve_cell(cx: int, cy: int) -> None:
        gx, gy = 2 * cx + 1, 2 * cy + 1
        grid[gy][gx] = "."

    visited = {(0, 0)}
    stack = [(0, 0)]
    carve_cell(0, 0)

    while stack:
        cx, cy = stack[-1]
        neighbors = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < cell_width and 0 <= ny < cell_height and (nx, ny) not in visited:
                neighbors.append((nx, ny))

        if not neighbors:
            stack.pop()
            continue

        nx, ny = rnd.choice(neighbors)
        x1, y1 = 2 * cx + 1, 2 * cy + 1
        x2, y2 = 2 * nx + 1, 2 * ny + 1
        grid[(y1 + y2) // 2][(x1 + x2) // 2] = "."
        grid[y2][x2] = "."
        visited.add((nx, ny))
        stack.append((nx, ny))

    start = (1, 1)
    goal = (grid_width - 2, grid_height - 2)
    grid[start[1]][start[0]] = "S"
    grid[goal[1]][goal[0]] = "G"

    return ["".join(row) for row in grid]


def generate_maze_3d(
    width: int,
    height: int,
    depth: int,
    seed: Optional[int] = None,
) -> dict:
    if width < 2 or height < 2 or depth < 2:
        raise MazeError("maze3d dimensions must be >= 2")

    rnd = random.Random(seed)

    def index_of(x: int, y: int, z: int) -> int:
        return x + y * width + z * width * height

    visited = {(0, 0, 0)}
    stack = [(0, 0, 0)]
    edges: set[Tuple[int, int]] = set()

    while stack:
        x, y, z = stack[-1]
        neighbors = []
        for dx, dy, dz in [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]:
            nx, ny, nz = x + dx, y + dy, z + dz
            if 0 <= nx < width and 0 <= ny < height and 0 <= nz < depth and (nx, ny, nz) not in visited:
                neighbors.append((nx, ny, nz))

        if not neighbors:
            stack.pop()
            continue

        nx, ny, nz = rnd.choice(neighbors)
        a = index_of(x, y, z)
        b = index_of(nx, ny, nz)
        edges.add((min(a, b), max(a, b)))
        visited.add((nx, ny, nz))
        stack.append((nx, ny, nz))

    start_xyz = (0, 0, 0)
    goal_xyz = (width - 1, height - 1, depth - 1)

    return {
        "type": "maze3d",
        "width": width,
        "height": height,
        "depth": depth,
        "edges": [list(edge) for edge in sorted(edges)],
        "start_xyz": list(start_xyz),
        "goal_xyz": list(goal_xyz),
    }


def write_maze(path: Union[str, Path], rows: Iterable[str]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(rows) + "\n"
    path.write_text(text, encoding="utf-8")
    return path


def generate_polar_maze(
    rings: int,
    sectors: int,
    seed: Optional[int] = None,
) -> dict:
    if rings < 2:
        raise MazeError("rings must be >= 2")
    if sectors < 4:
        raise MazeError("sectors must be >= 4")

    rnd = random.Random(seed)
    total_cells = rings * sectors

    def index_of(ring: int, sector: int) -> int:
        return ring * sectors + sector

    visited = {(0, 0)}
    stack = [(0, 0)]
    edges = set()

    while stack:
        ring, sector = stack[-1]
        neighbors = []
        for dr, ds in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr = ring + dr
            ns = (sector + ds) % sectors
            if 0 <= nr < rings and (nr, ns) not in visited:
                neighbors.append((nr, ns))

        if not neighbors:
            stack.pop()
            continue

        nr, ns = rnd.choice(neighbors)
        a = index_of(ring, sector)
        b = index_of(nr, ns)
        edges.add((min(a, b), max(a, b)))
        visited.add((nr, ns))
        stack.append((nr, ns))

    start = (0, 0)
    goal = (rings - 1, sectors // 2)

    return {
        "type": "polar",
        "rings": rings,
        "sectors": sectors,
        "edges": [[a, b] for a, b in sorted(edges)],
        "start_rs": list(start),
        "goal_rs": list(goal),
        "total_cells": total_cells,
    }


def write_polar_maze(path: Union[str, Path], data: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
    return path


def write_maze3d(path: Union[str, Path], data: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
    return path


def load_maze3d(path: Union[str, Path]) -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if data.get("type") != "maze3d":
        raise MazeError("not a maze3d file")

    width = data.get("width")
    height = data.get("height")
    depth = data.get("depth")
    edges = data.get("edges")
    start = data.get("start_xyz")
    goal = data.get("goal_xyz")

    if not isinstance(width, int) or not isinstance(height, int) or not isinstance(depth, int):
        raise MazeError("invalid maze3d dimensions")
    if width < 1 or height < 1 or depth < 1:
        raise MazeError("maze3d dimensions must be >= 1")
    if not isinstance(edges, list):
        raise MazeError("maze3d edges must be a list")
    if not (isinstance(start, list) and isinstance(goal, list)):
        raise MazeError("maze3d start/goal must be lists")
    if len(start) != 3 or len(goal) != 3:
        raise MazeError("maze3d start/goal must have three values")

    total_nodes = width * height * depth
    for edge in edges:
        if not (isinstance(edge, list) and len(edge) == 2):
            raise MazeError("maze3d edges must be pairs")
        a, b = edge
        if not (isinstance(a, int) and isinstance(b, int)):
            raise MazeError("maze3d edge indices must be integers")
        if not (0 <= a < total_nodes and 0 <= b < total_nodes):
            raise MazeError("maze3d edge index out of bounds")

    return {
        "type": "maze3d",
        "width": width,
        "height": height,
        "depth": depth,
        "edges": edges,
        "start_xyz": start,
        "goal_xyz": goal,
    }


def load_maze3d_graph(
    path: Union[str, Path],
) -> Tuple[np.ndarray, dict, Dict[Tuple[int, int, int], int]]:
    data = load_maze3d(path)
    edges = [(edge[0], edge[1]) for edge in data["edges"]]
    start = tuple(data["start_xyz"])
    goal = tuple(data["goal_xyz"])
    return build_maze3d_graph(
        data["width"],
        data["height"],
        data["depth"],
        edges,
        start,
        goal,
    )


def load_polar_maze(path: Union[str, Path]) -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if data.get("type") != "polar":
        raise MazeError("not a polar maze file")

    rings = data.get("rings")
    sectors = data.get("sectors")
    edges = data.get("edges")
    start = data.get("start_rs")
    goal = data.get("goal_rs")

    if not isinstance(rings, int) or not isinstance(sectors, int):
        raise MazeError("invalid rings/sectors")
    if rings < 2 or sectors < 4:
        raise MazeError("rings must be >= 2 and sectors >= 4")
    if not isinstance(edges, list):
        raise MazeError("edges must be a list")
    if not (isinstance(start, list) and isinstance(goal, list)):
        raise MazeError("start_rs/goal_rs must be lists")

    nodes = rings * sectors
    for edge in edges:
        if not (isinstance(edge, list) and len(edge) == 2):
            raise MazeError("edges must be pairs")
        a, b = edge
        if not (isinstance(a, int) and isinstance(b, int)):
            raise MazeError("edge indices must be integers")
        if not (0 <= a < nodes and 0 <= b < nodes):
            raise MazeError("edge index out of bounds")

    return {
        "type": "polar",
        "rings": rings,
        "sectors": sectors,
        "edges": edges,
        "start_rs": start,
        "goal_rs": goal,
    }


def load_polar_graph(
    path: Union[str, Path],
) -> Tuple[np.ndarray, dict, Dict[Tuple[int, int], int]]:
    data = load_polar_maze(path)
    return build_polar_graph(data)


def build_polar_graph(
    data: dict,
) -> Tuple[np.ndarray, dict, Dict[Tuple[int, int], int]]:
    rings = data["rings"]
    sectors = data["sectors"]
    edges = data["edges"]

    nodes = rings * sectors
    adjacency = np.zeros((nodes, nodes), dtype=float)
    for a, b in edges:
        adjacency[a, b] = 1.0
        adjacency[b, a] = 1.0

    positions = [(r, s) for r in range(rings) for s in range(sectors)]
    coord_to_index = {coord: idx for idx, coord in enumerate(positions)}

    graph_info = {
        "type": "polar",
        "rings": rings,
        "sectors": sectors,
        "positions": positions,
        "edges": edges,
        "start_rs": data["start_rs"],
        "goal_rs": data["goal_rs"],
    }

    return adjacency, graph_info, coord_to_index


def shortest_path_adjacency(
    adjacency: np.ndarray,
    start: int,
    goal: int,
) -> Optional[List[int]]:
    if start == goal:
        return [start]

    queue = deque([start])
    came_from: Dict[int, Optional[int]] = {start: None}

    while queue:
        current = queue.popleft()
        if current == goal:
            break
        neighbors = np.flatnonzero(adjacency[current])
        for neighbor in neighbors:
            idx = int(neighbor)
            if idx not in came_from:
                came_from[idx] = current
                queue.append(idx)

    if goal not in came_from:
        return None

    path = []
    current: Optional[int] = goal
    while current is not None:
        path.append(current)
        current = came_from[current]

    path.reverse()
    return path


def load_maze_image(
    path: Union[str, Path],
    threshold: int = 128,
    invert: bool = False,
    max_size: Optional[int] = 128,
    start_xy: Optional[Tuple[int, int]] = None,
    goal_xy: Optional[Tuple[int, int]] = None,
    detect_markers: bool = True,
    auto_threshold: bool = False,
    cleanup: bool = False,
    min_component: int = 20,
) -> Tuple[np.ndarray, dict, Dict[Tuple[int, int], int]]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise MazeError("Pillow is required for image mazes") from exc

    if threshold < 0 or threshold > 255:
        raise MazeError("threshold must be between 0 and 255")

    image = Image.open(path)
    orig_width, orig_height = image.size

    if max_size is not None and max_size > 0:
        scale = max(orig_width, orig_height) / max_size
        if scale > 1:
            new_width = max(1, int(round(orig_width / scale)))
            new_height = max(1, int(round(orig_height / scale)))
            image = image.resize((new_width, new_height), resample=Image.NEAREST)

    rgb = image.convert("RGB")
    gray = image.convert("L")
    if cleanup:
        try:
            from PIL import ImageFilter
        except ImportError as exc:
            raise MazeError("Pillow is required for image cleanup") from exc
        gray = gray.filter(ImageFilter.MedianFilter(size=3))
    rgb_array = np.array(rgb)
    array = np.array(gray)
    if array.ndim != 2:
        raise MazeError("maze image must be grayscale or convertible to grayscale")

    detected_start = False
    detected_goal = False
    endpoint_mode = "explicit" if (start_xy or goal_xy) else "auto"
    if detect_markers and (start_xy is None or goal_xy is None):
        red = rgb_array[:, :, 0]
        green = rgb_array[:, :, 1]
        blue = rgb_array[:, :, 2]
        red_mask = (red > 200) & (green < 80) & (blue < 80)
        green_mask = (green > 200) & (red < 80) & (blue < 80)

        if start_xy is None and red_mask.any():
            ys, xs = np.where(red_mask)
            start_xy = (int(np.mean(xs)), int(np.mean(ys)))
            detected_start = True
            endpoint_mode = "marker"

        if goal_xy is None and green_mask.any():
            ys, xs = np.where(green_mask)
            goal_xy = (int(np.mean(xs)), int(np.mean(ys)))
            detected_goal = True
            endpoint_mode = "marker"

    if auto_threshold:
        threshold = _otsu_threshold(array)

    if invert:
        open_mask = array < threshold
    else:
        open_mask = array >= threshold

    if cleanup:
        open_mask = _cleanup_mask(open_mask, min_component)

    height, width = open_mask.shape
    if open_mask.sum() < 2:
        raise MazeError("maze image must have at least two open cells")

    def find_first_open(reverse: bool = False) -> Tuple[int, int]:
        xs = range(width - 1, -1, -1) if reverse else range(width)
        ys = range(height - 1, -1, -1) if reverse else range(height)
        for y in ys:
            for x in xs:
                if open_mask[y, x]:
                    return (x, y)
        raise MazeError("maze image has no open cells")

    def farthest_from(seed: Tuple[int, int]) -> Tuple[int, int]:
        queue = deque([seed])
        distances = -np.ones((height, width), dtype=int)
        distances[seed[1], seed[0]] = 0
        farthest = seed
        max_dist = 0

        while queue:
            x, y = queue.popleft()
            dist = distances[y, x]
            if dist > max_dist:
                max_dist = dist
                farthest = (x, y)
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height and open_mask[ny, nx]:
                    if distances[ny, nx] == -1:
                        distances[ny, nx] = dist + 1
                        queue.append((nx, ny))

        return farthest

    def validate_point(point: Tuple[int, int], label: str) -> None:
        if not (0 <= point[0] < width and 0 <= point[1] < height):
            raise MazeError(f"{label} position out of bounds")

    if start_xy is not None:
        validate_point(start_xy, "start")
        open_mask[start_xy[1], start_xy[0]] = True
    if goal_xy is not None:
        validate_point(goal_xy, "goal")
        open_mask[goal_xy[1], goal_xy[0]] = True

    if start_xy is None and goal_xy is None:
        seed = find_first_open(reverse=False)
        start_xy = farthest_from(seed)
        goal_xy = farthest_from(start_xy)
        endpoint_mode = "farthest"
    elif start_xy is None:
        goal_xy = goal_xy or find_first_open(reverse=True)
        start_xy = farthest_from(goal_xy)
        endpoint_mode = "farthest"
    elif goal_xy is None:
        start_xy = start_xy or find_first_open(reverse=False)
        goal_xy = farthest_from(start_xy)
        endpoint_mode = "farthest"

    if start_xy == goal_xy:
        raise MazeError("start and goal must be different")

    if not open_mask[start_xy[1], start_xy[0]]:
        raise MazeError("start position is a wall")
    if not open_mask[goal_xy[1], goal_xy[0]]:
        raise MazeError("goal position is a wall")

    grid = [[WALL_CHAR for _ in range(width)] for _ in range(height)]
    open_cells: List[Tuple[int, int]] = []
    for y in range(height):
        for x in range(width):
            if open_mask[y, x]:
                grid[y][x] = "."
                open_cells.append((x, y))

    grid[start_xy[1]][start_xy[0]] = "S"
    grid[goal_xy[1]][goal_xy[0]] = "G"
    rows = ["".join(row) for row in grid]

    coord_to_index = {coord: idx for idx, coord in enumerate(open_cells)}
    adjacency = np.zeros((len(open_cells), len(open_cells)), dtype=float)

    for idx, (x, y) in enumerate(open_cells):
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            neighbor = (nx, ny)
            if neighbor in coord_to_index:
                adjacency[idx, coord_to_index[neighbor]] = 1.0

    graph_info = {
        "type": "maze",
        "width": width,
        "height": height,
        "positions": open_cells,
        "maze": rows,
        "start_xy": list(start_xy),
        "goal_xy": list(goal_xy),
        "source": "image",
        "image_path": str(path),
        "threshold": threshold,
        "invert": invert,
        "max_size": max_size,
        "original_size": [orig_width, orig_height],
        "auto_threshold": auto_threshold,
        "cleanup": cleanup,
        "min_component": min_component,
        "detect_markers": detect_markers,
        "detected_start": detected_start,
        "detected_goal": detected_goal,
        "endpoint_mode": endpoint_mode,
    }

    return adjacency, graph_info, coord_to_index


def shortest_path(
    rows: Iterable[str],
    start: Tuple[int, int],
    goal: Tuple[int, int],
) -> Optional[List[Tuple[int, int]]]:
    rows = list(rows)
    height = len(rows)
    width = len(rows[0]) if height else 0

    def is_open(x: int, y: int) -> bool:
        return 0 <= x < width and 0 <= y < height and rows[y][x] in OPEN_CHARS

    if not (is_open(*start) and is_open(*goal)):
        return None

    queue = deque([start])
    came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}

    while queue:
        current = queue.popleft()
        if current == goal:
            break
        x, y = current
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            neighbor = (nx, ny)
            if neighbor not in came_from and is_open(nx, ny):
                came_from[neighbor] = current
                queue.append(neighbor)

    if goal not in came_from:
        return None

    path = []
    current: Optional[Tuple[int, int]] = goal
    while current is not None:
        path.append(current)
        current = came_from[current]

    path.reverse()
    return path
