"""Flask web UI for Quantum Walk Explorer."""

from __future__ import annotations

from datetime import datetime
import base64
import io
import json
import logging
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from threading import Lock, Thread
from uuid import uuid4

import numpy as np
from flask import Flask, abort, jsonify, redirect, render_template, request, send_from_directory, url_for
from werkzeug.utils import secure_filename

from quantum_walk_explorer.cube_puzzle import dynamic_cube_adjacencies
from quantum_walk_explorer.hypercube import dynamic_hypercube_adjacencies, hypercube_adjacency
from quantum_walk_explorer.maze import (
    MazeError,
    build_maze_graph,
    build_maze3d_graph,
    build_polar_graph,
    generate_maze,
    generate_maze_3d,
    generate_polar_maze,
    load_maze_image,
    shortest_path,
    shortest_path_adjacency,
)
from quantum_walk_explorer.runlog import create_payload, load_run, save_run
from quantum_walk_explorer.qiskit_backend import (
    build_qiskit_circuit,
    continuous_time_walk_qiskit,
    qiskit_available,
    save_qiskit_artifacts,
)
from quantum_walk_explorer.visualize import (
    assemble_gif,
    plot_cube_projection,
    plot_hypercube_graph,
    plot_hypercube_projection,
    plot_maze_3d,
    plot_maze3d_heatmap,
    plot_maze_heatmap,
    plot_polar_heatmap,
    render_cube_frames,
    render_hypercube_frames_3d,
    render_hypercube_frames,
    render_maze3d_frames,
    render_maze_frames,
    render_polar_frames,
)
from quantum_walk_explorer.walk import basis_state, continuous_time_walk, probabilities, time_dependent_walk


BASE_DIR = Path(__file__).resolve().parent
RUNS_DIR = BASE_DIR / "web_runs"
UPLOADS_DIR = BASE_DIR / "uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp"}
JOB_TIMEOUT_SECONDS = 300
PREVIEW_SCALE = 4

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("quantum-walk-web")


RUNS_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

_JOBS: Dict[str, Dict[str, Any]] = {}
_JOBS_LOCK = Lock()


@app.context_processor
def inject_runtime_flags() -> Dict[str, bool]:
    return {"qiskit_available": qiskit_available()}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def create_run_id() -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_{uuid4().hex[:6]}"


def parse_int(value: Optional[str], default: int, min_value: Optional[int] = None) -> int:
    if value is None or value == "":
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError("invalid integer value") from exc
    if min_value is not None and parsed < min_value:
        raise ValueError("value below minimum")
    return parsed


def parse_float(value: Optional[str], default: float, min_value: Optional[float] = None) -> float:
    if value is None or value == "":
        return default
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError("invalid float value") from exc
    if min_value is not None and parsed < min_value:
        raise ValueError("value below minimum")
    return parsed


def parse_point(value: Optional[str]) -> Optional[Tuple[int, int]]:
    if not value:
        return None
    parts = value.split(",")
    if len(parts) != 2:
        raise ValueError("point must be in x,y format")
    return int(parts[0].strip()), int(parts[1].strip())


def build_preview_png_bytes(maze_rows: List[str], start: Tuple[int, int], goal: Tuple[int, int]) -> bytes:
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise MazeError("Pillow is required for image previews") from exc

    height = len(maze_rows)
    width = len(maze_rows[0]) if height else 0
    img = Image.new("RGB", (width, height), color=(10, 10, 10))
    pixels = img.load()
    for y, row in enumerate(maze_rows):
        for x, ch in enumerate(row):
            if ch != "#":
                pixels[x, y] = (230, 230, 230)

    draw = ImageDraw.Draw(img)
    draw.ellipse((start[0] - 1, start[1] - 1, start[0] + 1, start[1] + 1), fill=(255, 0, 0))
    draw.ellipse((goal[0] - 1, goal[1] - 1, goal[0] + 1, goal[1] + 1), fill=(0, 255, 0))

    scale = PREVIEW_SCALE
    if scale > 1:
        img = img.resize((width * scale, height * scale), resample=Image.NEAREST)

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


def build_preview_image(maze_rows: List[str], start: Tuple[int, int], goal: Tuple[int, int]) -> str:
    data = build_preview_png_bytes(maze_rows, start, goal)
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def write_one_pager_pdf(run_dir: Path, title: str, stats: Dict[str, Any]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    image_paths = [
        run_dir / "maze_input.png",
        run_dir / "heatmap.png",
        run_dir / "solution.png",
        run_dir / "walk.gif",
    ]

    images = []
    for path in image_paths:
        if path.exists():
            try:
                from PIL import Image

                images.append(Image.open(path))
            except Exception:
                images.append(None)
        else:
            images.append(None)

    fig = plt.figure(figsize=(8.5, 11))
    fig.patch.set_facecolor("white")
    fig.text(0.08, 0.95, title, fontsize=18, fontweight="bold")
    fig.text(0.08, 0.915, "Quantum Walk Explorer", fontsize=10, color="#4a5568")

    stats_text = "\n".join([f"{key}: {value}" for key, value in stats.items()])
    fig.text(0.08, 0.86, stats_text, fontsize=10, color="#1a202c")

    positions = [(0.08, 0.52), (0.52, 0.52), (0.08, 0.12), (0.52, 0.12)]
    for (left, bottom), image in zip(positions, images):
        ax = fig.add_axes([left, bottom, 0.4, 0.32])
        ax.axis("off")
        if image is not None:
            ax.imshow(image)
        else:
            ax.text(0.5, 0.5, "No image", ha="center", va="center", fontsize=9, color="#718096")

    pdf_path = run_dir / "onepager.pdf"
    fig.savefig(pdf_path, format="pdf")
    plt.close(fig)


def write_frames_pdf(frames: List[Path], output_path: Path) -> None:
    if not frames:
        return
    try:
        from PIL import Image
    except ImportError:
        return

    images = []
    for frame in frames:
        try:
            img = Image.open(frame).convert("RGB")
            images.append(img)
        except Exception:
            continue

    if not images:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(output_path, save_all=True, append_images=images[1:])


def compute_sim_steps(
    steps: int,
    gif_steps: Optional[int],
    gif: bool,
    solve_path: Optional[List[Tuple[int, int]]],
    gif_path_multiplier: float,
) -> int:
    sim_steps = steps
    if gif_steps is not None:
        sim_steps = max(sim_steps, gif_steps)
    elif gif and solve_path is not None:
        sim_steps = max(sim_steps, int(math.ceil(len(solve_path) * gif_path_multiplier)))
    return sim_steps


def compute_states(
    adjacency: np.ndarray,
    gamma: float,
    times: np.ndarray,
    start_state: np.ndarray,
    backend: str,
    compare: bool,
) -> Tuple[np.ndarray, Optional[float]]:
    if backend == "qiskit":
        if not qiskit_available():
            raise RuntimeError("qiskit backend requested but qiskit is not installed")
        states = continuous_time_walk_qiskit(adjacency, gamma, times, start_state)
        compare_error = None
        if compare:
            states_numpy = continuous_time_walk(adjacency, gamma, times, start_state)
            probs_qiskit = probabilities(states)
            probs_numpy = probabilities(states_numpy)
            compare_error = float(np.max(np.abs(probs_numpy - probs_qiskit)))
        return states, compare_error

    states = continuous_time_walk(adjacency, gamma, times, start_state)
    return states, None

def generate_run(
    image_path: Path,
    run_id: str,
    threshold: int,
    invert: bool,
    max_size: int,
    start_xy: Optional[Tuple[int, int]],
    goal_xy: Optional[Tuple[int, int]],
    detect_markers: bool,
    auto_threshold: bool,
    cleanup: bool,
    min_component: int,
    steps: int,
    dt: float,
    gamma: float,
    solve: bool,
    gif: bool,
    gif_steps: Optional[int],
    gif_path_multiplier: float,
    backend: str,
    qiskit_compare: bool,
    progress: Optional[callable] = None,
) -> Path:
    if progress:
        progress("loading image")
    adjacency, graph_info, coord_to_index = load_maze_image(
        image_path,
        threshold=threshold,
        invert=invert,
        max_size=max_size,
        start_xy=start_xy,
        goal_xy=goal_xy,
        detect_markers=detect_markers,
        auto_threshold=auto_threshold,
        cleanup=cleanup,
        min_component=min_component,
    )

    start = tuple(graph_info["start_xy"])
    start_index = coord_to_index[start]

    solve_path = None
    if solve:
        if progress:
            progress("solving maze")
        solve_path = shortest_path(graph_info["maze"], start, tuple(graph_info["goal_xy"]))

    sim_steps = compute_sim_steps(steps, gif_steps, gif, solve_path, gif_path_multiplier)
    times = (list(range(sim_steps)))
    start_state = basis_state(adjacency.shape[0], start_index)

    if progress:
        progress("simulating quantum walk")
    times_array = np.array(times) * dt
    states, compare_error = compute_states(adjacency, gamma, times_array, start_state, backend, qiskit_compare)
    probs = probabilities(states)

    params = {
        "gamma": gamma,
        "dt": dt,
        "steps": sim_steps,
        "start_index": start_index,
        "steps_requested": steps,
        "gif_steps": gif_steps,
        "gif_path_multiplier": gif_path_multiplier,
        "backend": backend,
    }
    if compare_error is not None:
        params["qiskit_compare_max_error"] = compare_error

    payload = create_payload(
        graph=graph_info,
        params=params,
        times=times_array.tolist(),
        probabilities=probs.tolist(),
    )

    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    save_run(run_dir / "run.json", payload)

    if backend == "qiskit":
        try:
            circuit = build_qiskit_circuit(adjacency, gamma, times_array[-1], start_state)
            save_qiskit_artifacts(circuit, run_dir)
        except Exception:
            logger.warning("Run %s: failed to save Qiskit artifacts", run_id)
        if compare_error is not None:
            compare_path = run_dir / "qiskit_compare.json"
            compare_path.write_text(
                json.dumps({"max_error": compare_error}, indent=2),
                encoding="utf-8",
            )

    if progress:
        progress("rendering heatmap")
    plot_maze_heatmap(
        probs[-1],
        graph_info["width"],
        graph_info["height"],
        graph_info["positions"],
        "Final probability",
        run_dir / "heatmap.png",
        start=tuple(graph_info["start_xy"]),
        goal=tuple(graph_info["goal_xy"]),
    )

    if solve and solve_path:
        solution = {
            "path": [list(step) for step in solve_path],
            "length": max(len(solve_path) - 1, 0),
        }
        (run_dir / "solution.json").write_text(json.dumps(solution, indent=2), encoding="utf-8")
        if progress:
            progress("rendering solution overlay")
        plot_maze_heatmap(
            probs[-1],
            graph_info["width"],
            graph_info["height"],
            graph_info["positions"],
            "Maze solution",
            run_dir / "solution.png",
            start=tuple(graph_info["start_xy"]),
            goal=tuple(graph_info["goal_xy"]),
            path=solve_path,
        )

    if gif:
        def frame_progress(current: int, total: int) -> None:
            if progress:
                progress(f"rendering frames {current}/{total}")

        if progress:
            progress("rendering GIF")
        frames = render_maze_frames(
            probs,
            graph_info["width"],
            graph_info["height"],
            graph_info["positions"],
            run_dir / "frames",
            progress=frame_progress,
        )
        assemble_gif(frames, run_dir / "walk.gif", fps=6)
        write_frames_pdf(frames, run_dir / "frames.pdf")

    stats = {
        "steps": sim_steps,
        "dt": dt,
        "gamma": gamma,
        "goal_prob": float(probs[-1][coord_to_index[tuple(graph_info["goal_xy"])]]) if graph_info.get("goal_xy") else None,
        "endpoint_mode": graph_info.get("endpoint_mode"),
    }
    write_one_pager_pdf(run_dir, "Maze Run Summary", stats)

    if progress:
        progress("done")
    return run_dir


def generate_ascii_run(
    run_id: str,
    rows: List[str],
    steps: int,
    dt: float,
    gamma: float,
    solve: bool,
    gif: bool,
    gif_steps: Optional[int],
    gif_path_multiplier: float,
    backend: str,
    qiskit_compare: bool,
    progress: Optional[callable] = None,
) -> Path:
    if progress:
        progress("building maze")

    adjacency, graph_info, coord_to_index = build_maze_graph(rows)
    graph_info["source"] = "generated"

    start = tuple(graph_info["start_xy"])
    start_index = coord_to_index[start]

    solve_path = None
    if solve:
        if progress:
            progress("solving maze")
        solve_path = shortest_path(graph_info["maze"], start, tuple(graph_info["goal_xy"]))

    sim_steps = compute_sim_steps(steps, gif_steps, gif, solve_path, gif_path_multiplier)
    times = np.arange(sim_steps) * dt
    start_state = basis_state(adjacency.shape[0], start_index)

    if progress:
        progress("simulating quantum walk")
    states, compare_error = compute_states(adjacency, gamma, times, start_state, backend, qiskit_compare)
    probs = probabilities(states)

    params = {
        "gamma": gamma,
        "dt": dt,
        "steps": sim_steps,
        "start_index": start_index,
        "steps_requested": steps,
        "gif_steps": gif_steps,
        "gif_path_multiplier": gif_path_multiplier,
        "backend": backend,
    }
    if compare_error is not None:
        params["qiskit_compare_max_error"] = compare_error

    payload = create_payload(
        graph=graph_info,
        params=params,
        times=times.tolist(),
        probabilities=probs.tolist(),
    )

    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "maze.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
    save_run(run_dir / "run.json", payload)

    if backend == "qiskit":
        try:
            circuit = build_qiskit_circuit(adjacency, gamma, times[-1], start_state)
            save_qiskit_artifacts(circuit, run_dir)
        except Exception:
            logger.warning("Run %s: failed to save Qiskit artifacts", run_id)
        if compare_error is not None:
            compare_path = run_dir / "qiskit_compare.json"
            compare_path.write_text(
                json.dumps({"max_error": compare_error}, indent=2),
                encoding="utf-8",
            )

    if progress:
        progress("rendering heatmap")
    plot_maze_heatmap(
        probs[-1],
        graph_info["width"],
        graph_info["height"],
        graph_info["positions"],
        "Final probability",
        run_dir / "heatmap.png",
        start=tuple(graph_info["start_xy"]),
        goal=tuple(graph_info["goal_xy"]),
    )

    preview_bytes = build_preview_png_bytes(
        graph_info["maze"],
        tuple(graph_info["start_xy"]),
        tuple(graph_info["goal_xy"]),
    )
    (run_dir / "maze_input.png").write_bytes(preview_bytes)

    if solve and solve_path:
        solution = {
            "path": [list(step) for step in solve_path],
            "length": max(len(solve_path) - 1, 0),
        }
        (run_dir / "solution.json").write_text(json.dumps(solution, indent=2), encoding="utf-8")
        if progress:
            progress("rendering solution overlay")
        plot_maze_heatmap(
            probs[-1],
            graph_info["width"],
            graph_info["height"],
            graph_info["positions"],
            "Maze solution",
            run_dir / "solution.png",
            start=tuple(graph_info["start_xy"]),
            goal=tuple(graph_info["goal_xy"]),
            path=solve_path,
        )

    if gif:
        def frame_progress(current: int, total: int) -> None:
            if progress:
                progress(f"rendering frames {current}/{total}")

        if progress:
            progress("rendering GIF")
        frames = render_maze_frames(
            probs,
            graph_info["width"],
            graph_info["height"],
            graph_info["positions"],
            run_dir / "frames",
            progress=frame_progress,
        )
        assemble_gif(frames, run_dir / "walk.gif", fps=6)
        write_frames_pdf(frames, run_dir / "frames.pdf")

    stats = {
        "steps": sim_steps,
        "dt": dt,
        "gamma": gamma,
        "goal_prob": float(probs[-1][coord_to_index[tuple(graph_info["goal_xy"])]]) if graph_info.get("goal_xy") else None,
    }
    write_one_pager_pdf(run_dir, "Generated Maze Summary", stats)

    if progress:
        progress("done")

    return run_dir


def generate_hypercube_run(
    run_id: str,
    dimensions: int,
    steps: int,
    dt: float,
    gamma: float,
    dynamic: bool,
    shift_rate: float,
    seed: Optional[int],
    gif: bool,
    gif_3d: bool,
    backend: str,
    qiskit_compare: bool,
    progress: Optional[callable] = None,
) -> Path:
    if progress:
        progress("building hypercube")
    adjacency = hypercube_adjacency(dimensions)
    nodes = adjacency.shape[0]
    start_index = 0
    goal_index = nodes - 1
    labels = [format(i, f"0{dimensions}b") for i in range(nodes)]

    solve_path = None
    if dynamic:
        if backend == "qiskit":
            raise RuntimeError("qiskit backend is not supported for dynamic hypercubes")
        adjacencies = dynamic_hypercube_adjacencies(dimensions, steps, seed=seed, shift_rate=shift_rate)
    else:
        adjacencies = None
        solve_path = shortest_path_adjacency(adjacency, start_index, goal_index)

    start_state = basis_state(nodes, start_index)
    if progress:
        progress("simulating quantum walk")
    compare_error = None
    if adjacencies is None:
        times = np.arange(steps) * dt
        states, compare_error = compute_states(adjacency, gamma, times, start_state, backend, qiskit_compare)
    else:
        states = time_dependent_walk(adjacencies, gamma, dt, start_state)
        times = np.arange(steps) * dt

    probs = probabilities(states)

    goal_probs = probs[:, goal_index]
    best_step = int(np.argmax(goal_probs))
    best_prob = float(goal_probs[best_step])

    params = {
        "gamma": gamma,
        "dt": dt,
        "steps": steps,
        "start_index": start_index,
        "goal_index": goal_index,
        "dynamic": dynamic,
        "shift_rate": shift_rate,
        "dynamic_seed": seed,
        "backend": backend,
    }
    if compare_error is not None:
        params["qiskit_compare_max_error"] = compare_error

    graph_info = {
        "type": "hypercube",
        "dimensions": dimensions,
        "nodes": nodes,
        "positions": labels,
        "start_index": start_index,
        "goal_index": goal_index,
    }

    payload = create_payload(
        graph=graph_info,
        params=params,
        times=(np.arange(steps) * dt).tolist(),
        probabilities=probs.tolist(),
    )

    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    save_run(run_dir / "run.json", payload)

    if backend == "qiskit":
        try:
            circuit = build_qiskit_circuit(adjacency, gamma, times[-1], start_state)
            save_qiskit_artifacts(circuit, run_dir)
        except Exception:
            logger.warning("Run %s: failed to save Qiskit artifacts", run_id)
        if compare_error is not None:
            compare_path = run_dir / "qiskit_compare.json"
            compare_path.write_text(
                json.dumps({"max_error": compare_error}, indent=2),
                encoding="utf-8",
            )

    if progress:
        progress("rendering heatmap")
    plot_hypercube_projection(
        probs[-1],
        dimensions,
        "Final probability",
        run_dir / "heatmap.png",
        start_index=start_index,
        goal_index=goal_index,
    )

    if progress:
        progress("rendering hypercube map")
    plot_hypercube_graph(
        probs[-1],
        dimensions,
        "Hypercube connectivity map",
        run_dir / "graph.png",
        start_index=start_index,
        goal_index=goal_index,
    )

    solution = {
        "goal_index": goal_index,
        "best_step": best_step,
        "best_probability": best_prob,
    }
    if not dynamic and solve_path is not None:
        solution["path_indices"] = solve_path
        solution["path_labels"] = [labels[idx] for idx in solve_path]
        plot_hypercube_graph(
            probs[-1],
            dimensions,
            "Hypercube solution path",
            run_dir / "solution.png",
            start_index=start_index,
            goal_index=goal_index,
            path_indices=solve_path,
        )
    (run_dir / "solution.json").write_text(json.dumps(solution, indent=2), encoding="utf-8")

    if gif:
        def frame_progress(current: int, total: int) -> None:
            if progress:
                progress(f"rendering frames {current}/{total}")

        if progress:
            progress("rendering GIF")
        if gif_3d:
            frames = render_hypercube_frames_3d(
                probs,
                dimensions,
                run_dir / "frames",
                progress=frame_progress,
            )
        else:
            frames = render_hypercube_frames(probs, dimensions, run_dir / "frames", progress=frame_progress)
        assemble_gif(frames, run_dir / "walk.gif", fps=6)
        write_frames_pdf(frames, run_dir / "frames.pdf")

    stats = {
        "dimensions": dimensions,
        "steps": steps,
        "dt": dt,
        "gamma": gamma,
        "best_prob": best_prob,
        "dynamic": dynamic,
        "shift_rate": shift_rate,
    }
    write_one_pager_pdf(run_dir, "Hypercube Run Summary", stats)

    if progress:
        progress("done")

    return run_dir


def generate_maze3d_run(
    run_id: str,
    width: int,
    height: int,
    depth: int,
    seed: Optional[int],
    steps: int,
    dt: float,
    gamma: float,
    solve: bool,
    gif: bool,
    gif_steps: Optional[int],
    gif_path_multiplier: float,
    backend: str,
    qiskit_compare: bool,
    progress: Optional[callable] = None,
) -> Path:
    if progress:
        progress("building 3d maze")

    data = generate_maze_3d(width, height, depth, seed=seed)
    edges = [(edge[0], edge[1]) for edge in data["edges"]]
    start_xyz = tuple(data["start_xyz"])
    goal_xyz = tuple(data["goal_xyz"])
    adjacency, graph_info, coord_to_index = build_maze3d_graph(
        width,
        height,
        depth,
        edges,
        start_xyz,
        goal_xyz,
    )
    graph_info["source"] = "generated"

    start_index = coord_to_index[start_xyz]
    goal_index = coord_to_index[goal_xyz]

    solve_indices = None
    solve_coords = None
    if solve:
        if progress:
            progress("solving maze")
        solve_indices = shortest_path_adjacency(adjacency, start_index, goal_index)
        if solve_indices is not None:
            solve_coords = [graph_info["positions"][idx] for idx in solve_indices]

    sim_steps = compute_sim_steps(steps, gif_steps, gif, solve_coords, gif_path_multiplier)
    times = np.arange(sim_steps) * dt
    start_state = basis_state(adjacency.shape[0], start_index)

    if progress:
        progress("simulating quantum walk")
    states, compare_error = compute_states(adjacency, gamma, times, start_state, backend, qiskit_compare)
    probs = probabilities(states)

    params = {
        "gamma": gamma,
        "dt": dt,
        "steps": sim_steps,
        "start_index": start_index,
        "goal_index": goal_index,
        "steps_requested": steps,
        "gif_steps": gif_steps,
        "gif_path_multiplier": gif_path_multiplier,
        "backend": backend,
    }
    if compare_error is not None:
        params["qiskit_compare_max_error"] = compare_error

    payload = create_payload(
        graph=graph_info,
        params=params,
        times=times.tolist(),
        probabilities=probs.tolist(),
    )

    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    save_run(run_dir / "run.json", payload)
    (run_dir / "maze3d.json").write_text(json.dumps(data, indent=2), encoding="utf-8")

    if backend == "qiskit":
        try:
            circuit = build_qiskit_circuit(adjacency, gamma, times[-1], start_state)
            save_qiskit_artifacts(circuit, run_dir)
        except Exception:
            logger.warning("Run %s: failed to save Qiskit artifacts", run_id)
        if compare_error is not None:
            compare_path = run_dir / "qiskit_compare.json"
            compare_path.write_text(
                json.dumps({"max_error": compare_error}, indent=2),
                encoding="utf-8",
            )

    if progress:
        progress("rendering 3d heatmap")
    plot_maze3d_heatmap(
        probs[-1],
        width,
        height,
        depth,
        graph_info["positions"],
        "Final probability",
        run_dir / "heatmap.png",
        start_index=start_index,
        goal_index=goal_index,
    )

    if solve and solve_indices:
        solution = {
            "path_indices": solve_indices,
            "path_xyz": [list(coord) for coord in solve_coords] if solve_coords else None,
            "length": max(len(solve_indices) - 1, 0),
        }
        (run_dir / "solution.json").write_text(json.dumps(solution, indent=2), encoding="utf-8")
        if progress:
            progress("rendering solution overlay")
        plot_maze_3d(
            probs[-1],
            width,
            height,
            depth,
            graph_info["positions"],
            graph_info["edges"],
            "Maze3D solution",
            run_dir / "solution.png",
            start_index=start_index,
            goal_index=goal_index,
            path_indices=solve_indices,
        )

    if gif:
        def frame_progress(current: int, total: int) -> None:
            if progress:
                progress(f"rendering frames {current}/{total}")

        if progress:
            progress("rendering GIF")
        frames = render_maze3d_frames(
            probs,
            width,
            height,
            depth,
            graph_info["positions"],
            graph_info["edges"],
            run_dir / "frames",
            progress=frame_progress,
        )
        assemble_gif(frames, run_dir / "walk.gif", fps=6)
        write_frames_pdf(frames, run_dir / "frames.pdf")

    stats = {
        "width": width,
        "height": height,
        "depth": depth,
        "steps": sim_steps,
        "dt": dt,
        "gamma": gamma,
        "goal_prob": float(probs[-1][goal_index]),
    }
    write_one_pager_pdf(run_dir, "Maze3D Run Summary", stats)

    if progress:
        progress("done")

    return run_dir


def generate_polar_run(
    run_id: str,
    rings: int,
    sectors: int,
    seed: Optional[int],
    steps: int,
    dt: float,
    gamma: float,
    solve: bool,
    gif: bool,
    gif_steps: Optional[int],
    gif_path_multiplier: float,
    backend: str,
    qiskit_compare: bool,
    progress: Optional[callable] = None,
) -> Path:
    if progress:
        progress("building polar maze")

    data = generate_polar_maze(rings, sectors, seed=seed)
    adjacency, graph_info, coord_to_index = build_polar_graph(data)
    graph_info["source"] = "generated"

    start_rs = tuple(graph_info["start_rs"])
    goal_rs = tuple(graph_info["goal_rs"])
    start_index = coord_to_index[start_rs]
    goal_index = coord_to_index[goal_rs]

    solve_path = None
    if solve:
        if progress:
            progress("solving maze")
        path_indices = shortest_path_adjacency(adjacency, start_index, goal_index)
        if path_indices is not None:
            solve_path = [graph_info["positions"][idx] for idx in path_indices]

    sim_steps = compute_sim_steps(steps, gif_steps, gif, solve_path, gif_path_multiplier)
    times = np.arange(sim_steps) * dt
    start_state = basis_state(adjacency.shape[0], start_index)

    if progress:
        progress("simulating quantum walk")
    states, compare_error = compute_states(adjacency, gamma, times, start_state, backend, qiskit_compare)
    probs = probabilities(states)

    params = {
        "gamma": gamma,
        "dt": dt,
        "steps": sim_steps,
        "start_index": start_index,
        "goal_index": goal_index,
        "steps_requested": steps,
        "gif_steps": gif_steps,
        "gif_path_multiplier": gif_path_multiplier,
        "backend": backend,
    }
    if compare_error is not None:
        params["qiskit_compare_max_error"] = compare_error

    payload = create_payload(
        graph=graph_info,
        params=params,
        times=times.tolist(),
        probabilities=probs.tolist(),
    )

    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    save_run(run_dir / "run.json", payload)
    (run_dir / "polar.json").write_text(json.dumps(data, indent=2), encoding="utf-8")

    if backend == "qiskit":
        try:
            circuit = build_qiskit_circuit(adjacency, gamma, times[-1], start_state)
            save_qiskit_artifacts(circuit, run_dir)
        except Exception:
            logger.warning("Run %s: failed to save Qiskit artifacts", run_id)
        if compare_error is not None:
            compare_path = run_dir / "qiskit_compare.json"
            compare_path.write_text(
                json.dumps({"max_error": compare_error}, indent=2),
                encoding="utf-8",
            )

    if progress:
        progress("rendering heatmap")
    plot_polar_heatmap(
        probs[-1],
        rings,
        sectors,
        "Final probability",
        run_dir / "heatmap.png",
        edges=data["edges"],
        start=start_rs,
        goal=goal_rs,
    )

    if solve and solve_path:
        solution = {
            "path": [list(step) for step in solve_path],
            "length": max(len(solve_path) - 1, 0),
        }
        (run_dir / "solution.json").write_text(json.dumps(solution, indent=2), encoding="utf-8")
        if progress:
            progress("rendering solution overlay")
        plot_polar_heatmap(
            probs[-1],
            rings,
            sectors,
            "Polar solution",
            run_dir / "solution.png",
            edges=data["edges"],
            start=start_rs,
            goal=goal_rs,
            path=solve_path,
        )

    if gif:
        def frame_progress(current: int, total: int) -> None:
            if progress:
                progress(f"rendering frames {current}/{total}")

        if progress:
            progress("rendering GIF")
        frames = render_polar_frames(
            probs,
            rings,
            sectors,
            run_dir / "frames",
            edges=data["edges"],
            progress=frame_progress,
        )
        assemble_gif(frames, run_dir / "walk.gif", fps=6)
        write_frames_pdf(frames, run_dir / "frames.pdf")

    stats = {
        "rings": rings,
        "sectors": sectors,
        "steps": sim_steps,
        "dt": dt,
        "gamma": gamma,
        "goal_prob": float(probs[-1][goal_index]),
    }
    write_one_pager_pdf(run_dir, "Polar Maze Summary", stats)

    if progress:
        progress("done")

    return run_dir


def generate_cube_run(
    run_id: str,
    size: int,
    steps: int,
    dt: float,
    gamma: float,
    shift_rate: float,
    seed: Optional[int],
    gif: bool,
    progress: Optional[callable] = None,
) -> Path:
    if progress:
        progress("building cube puzzle")

    adjacencies, rotations = dynamic_cube_adjacencies(
        size,
        steps,
        seed=seed,
        shift_rate=shift_rate,
    )
    nodes = size ** 3
    start_index = 0
    goal_index = nodes - 1
    start_state = basis_state(nodes, start_index)

    if progress:
        progress("simulating quantum walk")
    states = time_dependent_walk(adjacencies, gamma, dt, start_state)
    probs = probabilities(states)

    goal_probs = probs[:, goal_index]
    best_step = int(np.argmax(goal_probs))
    best_prob = float(goal_probs[best_step])

    params = {
        "gamma": gamma,
        "dt": dt,
        "steps": steps,
        "shift_rate": shift_rate,
        "seed": seed,
    }

    graph_info = {
        "type": "cube",
        "size": size,
        "nodes": nodes,
        "start_index": start_index,
        "goal_index": goal_index,
        "rotations": rotations,
    }

    payload = create_payload(
        graph=graph_info,
        params=params,
        times=(np.arange(steps) * dt).tolist(),
        probabilities=probs.tolist(),
    )

    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    save_run(run_dir / "run.json", payload)

    if progress:
        progress("rendering heatmap")
    plot_cube_projection(
        probs[-1],
        size,
        "Final probability",
        run_dir / "heatmap.png",
    )

    solution = {
        "goal_index": goal_index,
        "best_step": best_step,
        "best_probability": best_prob,
    }
    (run_dir / "solution.json").write_text(json.dumps(solution, indent=2), encoding="utf-8")

    if gif:
        def frame_progress(current: int, total: int) -> None:
            if progress:
                progress(f"rendering frames {current}/{total}")

        if progress:
            progress("rendering GIF")
        frames = render_cube_frames(probs, size, run_dir / "frames", progress=frame_progress)
        assemble_gif(frames, run_dir / "walk.gif", fps=6)
        write_frames_pdf(frames, run_dir / "frames.pdf")

    stats = {
        "size": size,
        "steps": steps,
        "dt": dt,
        "gamma": gamma,
        "best_prob": best_prob,
        "shift_rate": shift_rate,
    }
    write_one_pager_pdf(run_dir, "Cube Puzzle Run Summary", stats)

    if progress:
        progress("done")

    return run_dir


def run_summary(run_dir: Path) -> Optional[Dict[str, Any]]:
    run_path = run_dir / "run.json"
    if not run_path.exists():
        return None
    data = load_run(run_path)
    graph = data.get("graph", {})
    params = data.get("params", {})
    times = data.get("times", [])
    probabilities_list = data.get("probabilities", [])

    last_probs = probabilities_list[-1] if probabilities_list else []
    goal_prob = None
    goal_coord = graph.get("goal_xy") or graph.get("goal_xyz") or graph.get("goal_rs")
    positions = graph.get("positions", [])
    if goal_coord and last_probs:
        goal_tuple = tuple(goal_coord)
        for idx, pos in enumerate(positions):
            if tuple(pos) == goal_tuple:
                goal_prob = float(last_probs[idx])
                break

    solution_path = run_dir / "solution.json"
    solution_length = None
    if solution_path.exists():
        try:
            solution = json.loads(solution_path.read_text(encoding="utf-8"))
            solution_length = solution.get("length")
        except json.JSONDecodeError:
            solution_length = None

    return {
        "id": run_dir.name,
        "created_at": data.get("created_at"),
        "params": params,
        "times": times,
        "steps": params.get("steps"),
        "goal_prob": goal_prob,
        "solution_length": solution_length,
        "assets": {
            "heatmap": (run_dir / "heatmap.png").exists(),
            "graph": (run_dir / "graph.png").exists(),
            "solution": (run_dir / "solution.png").exists(),
            "gif": (run_dir / "walk.gif").exists(),
            "frames_pdf": (run_dir / "frames.pdf").exists(),
            "input": (run_dir / "maze_input.png").exists(),
            "maze3d": (run_dir / "maze3d.json").exists(),
            "polar": (run_dir / "polar.json").exists(),
            "qiskit_png": (run_dir / "qiskit_circuit.png").exists(),
            "qiskit_qasm": (run_dir / "qiskit_circuit.qasm").exists(),
            "qiskit_compare": (run_dir / "qiskit_compare.json").exists(),
            "onepager": (run_dir / "onepager.pdf").exists(),
        },
    }


def list_runs() -> List[Dict[str, Any]]:
    runs: List[Dict[str, Any]] = []
    if not RUNS_DIR.exists():
        return runs
    for run_dir in sorted(RUNS_DIR.iterdir(), reverse=True):
        if not run_dir.is_dir():
            continue
        summary = run_summary(run_dir)
        if summary:
            runs.append(summary)
    return runs


def _set_job(run_id: str, status: str, detail: Optional[str] = None, error: Optional[str] = None) -> None:
    def progress_value(status_value: str, detail_value: Optional[str]) -> int:
        if status_value == "complete":
            return 100
        if status_value == "error":
            return 100
        if status_value == "queued":
            return 5
        if not detail_value:
            return 10
        if detail_value.startswith("rendering frames"):
            match = re.search(r"(\\d+)\\s*/\\s*(\\d+)", detail_value)
            if match:
                current = int(match.group(1))
                total = int(match.group(2))
                if total > 0:
                    return min(99, 60 + int((current / total) * 35))
        detail_map = {
            "loading image": 15,
            "building 3d maze": 20,
            "building polar maze": 22,
            "solving maze": 25,
            "simulating quantum walk": 55,
            "rendering heatmap": 75,
            "rendering 3d heatmap": 78,
            "rendering hypercube map": 80,
            "rendering solution overlay": 85,
            "rendering GIF": 92,
            "done": 100,
        }
        return detail_map.get(detail_value, 20)

    with _JOBS_LOCK:
        job = _JOBS.setdefault(run_id, {"status": "queued"})
        previous_status = job.get("status")
        previous_detail = job.get("detail")
        job["status"] = status
        if detail is not None:
            job["detail"] = detail
        if error is not None:
            job["error"] = error
        job["progress"] = progress_value(status, detail)
        if status == "running" and "started_at" not in job:
            job["started_at"] = datetime.now().isoformat()
        if status == "complete":
            job["finished_at"] = datetime.now().isoformat()
        if "logs" not in job:
            job["logs"] = []
        if status != previous_status or (detail and detail != previous_detail):
            job["logs"].append(
                {
                    "ts": datetime.now().isoformat(timespec="seconds"),
                    "message": detail or status,
                }
            )
        if error:
            job["logs"].append(
                {
                    "ts": datetime.now().isoformat(timespec="seconds"),
                    "message": f"error: {error}",
                }
            )
        job["last_update"] = datetime.now().isoformat()
    if error:
        logger.error("Run %s failed: %s", run_id, error)
    else:
        logger.info("Run %s status: %s", run_id, detail or status)

    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        log_path = run_dir / "job.log"
        if detail or error or status != previous_status:
            entry = {
                "ts": datetime.now().isoformat(timespec="seconds"),
                "status": status,
                "detail": detail or status,
                "error": error,
            }
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry) + "\n")
    except Exception:
        logger.warning("Run %s: failed to persist job log", run_id)


def _get_job(run_id: str) -> Optional[Dict[str, Any]]:
    with _JOBS_LOCK:
        job = _JOBS.get(run_id)
        if job:
            return dict(job)
    return None


def _get_job_logs(run_id: str, since: int) -> Optional[Dict[str, Any]]:
    with _JOBS_LOCK:
        job = _JOBS.get(run_id)
        if not job:
            return None
        logs = job.get("logs", [])
        total = len(logs)
        since = max(0, since)
        if since >= total:
            return {"logs": [], "next_index": total}
        return {"logs": logs[since:], "next_index": total}


def _run_job(
    run_id: str,
    upload_path: Path,
    threshold: int,
    invert: bool,
    max_size: int,
    start_xy: Optional[Tuple[int, int]],
    goal_xy: Optional[Tuple[int, int]],
    detect_markers: bool,
    auto_threshold: bool,
    cleanup: bool,
    min_component: int,
    steps: int,
    dt: float,
    gamma: float,
    solve: bool,
    gif: bool,
    gif_steps: Optional[int],
    gif_multiplier: float,
    backend: str,
    qiskit_compare: bool,
) -> None:
    _set_job(run_id, "running", "queued")
    try:
        run_dir = generate_run(
            upload_path,
            run_id,
            threshold,
            invert,
            max_size,
            start_xy,
            goal_xy,
            detect_markers,
            auto_threshold,
            cleanup,
            min_component,
            steps,
            dt,
            gamma,
            solve,
            gif,
            gif_steps,
            gif_multiplier,
            backend,
            qiskit_compare,
            progress=lambda step: _set_job(run_id, "running", step),
        )
        try:
            from PIL import Image

            with Image.open(upload_path) as image:
                image.convert("L").save(run_dir / "maze_input.png")
        except Exception:
            logger.warning("Run %s: failed to save maze input preview", run_id)

        _set_job(run_id, "complete", "complete")
    except Exception as exc:
        _set_job(run_id, "error", "error", str(exc))


def _run_job_generated(
    run_id: str,
    rows: List[str],
    steps: int,
    dt: float,
    gamma: float,
    solve: bool,
    gif: bool,
    gif_steps: Optional[int],
    gif_multiplier: float,
    backend: str,
    qiskit_compare: bool,
) -> None:
    _set_job(run_id, "running", "queued")
    try:
        generate_ascii_run(
            run_id,
            rows,
            steps,
            dt,
            gamma,
            solve,
            gif,
            gif_steps,
            gif_multiplier,
            backend,
            qiskit_compare,
            progress=lambda step: _set_job(run_id, "running", step),
        )
        _set_job(run_id, "complete", "complete")
    except Exception as exc:
        _set_job(run_id, "error", "error", str(exc))


def _run_job_maze3d(
    run_id: str,
    width: int,
    height: int,
    depth: int,
    seed: Optional[int],
    steps: int,
    dt: float,
    gamma: float,
    solve: bool,
    gif: bool,
    gif_steps: Optional[int],
    gif_multiplier: float,
    backend: str,
    qiskit_compare: bool,
) -> None:
    _set_job(run_id, "running", "queued")
    try:
        generate_maze3d_run(
            run_id,
            width,
            height,
            depth,
            seed,
            steps,
            dt,
            gamma,
            solve,
            gif,
            gif_steps,
            gif_multiplier,
            backend,
            qiskit_compare,
            progress=lambda step: _set_job(run_id, "running", step),
        )
        _set_job(run_id, "complete", "complete")
    except Exception as exc:
        _set_job(run_id, "error", "error", str(exc))


def _run_job_polar(
    run_id: str,
    rings: int,
    sectors: int,
    seed: Optional[int],
    steps: int,
    dt: float,
    gamma: float,
    solve: bool,
    gif: bool,
    gif_steps: Optional[int],
    gif_multiplier: float,
    backend: str,
    qiskit_compare: bool,
) -> None:
    _set_job(run_id, "running", "queued")
    try:
        generate_polar_run(
            run_id,
            rings,
            sectors,
            seed,
            steps,
            dt,
            gamma,
            solve,
            gif,
            gif_steps,
            gif_multiplier,
            backend,
            qiskit_compare,
            progress=lambda step: _set_job(run_id, "running", step),
        )
        _set_job(run_id, "complete", "complete")
    except Exception as exc:
        _set_job(run_id, "error", "error", str(exc))


def _run_job_hypercube(
    run_id: str,
    dimensions: int,
    steps: int,
    dt: float,
    gamma: float,
    dynamic: bool,
    shift_rate: float,
    seed: Optional[int],
    gif: bool,
    gif_3d: bool,
    backend: str,
    qiskit_compare: bool,
) -> None:
    _set_job(run_id, "running", "queued")
    try:
        generate_hypercube_run(
            run_id,
            dimensions,
            steps,
            dt,
            gamma,
            dynamic,
            shift_rate,
            seed,
            gif,
            gif_3d,
            backend,
            qiskit_compare,
            progress=lambda step: _set_job(run_id, "running", step),
        )
        _set_job(run_id, "complete", "complete")
    except Exception as exc:
        _set_job(run_id, "error", "error", str(exc))


def _run_job_cube(
    run_id: str,
    size: int,
    steps: int,
    dt: float,
    gamma: float,
    shift_rate: float,
    seed: Optional[int],
    gif: bool,
) -> None:
    _set_job(run_id, "running", "queued")
    try:
        generate_cube_run(
            run_id,
            size,
            steps,
            dt,
            gamma,
            shift_rate,
            seed,
            gif,
            progress=lambda step: _set_job(run_id, "running", step),
        )
        _set_job(run_id, "complete", "complete")
    except Exception as exc:
        _set_job(run_id, "error", "error", str(exc))


@app.route("/")
def landing():
    return render_template("index.html")


@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "GET":
        return render_template("upload.html")

    file = request.files.get("maze_image")
    if file is None or file.filename == "":
        return render_template("upload.html", error="Please choose a maze image to upload."), 400

    if not allowed_file(file.filename):
        return render_template("upload.html", error="Unsupported file type."), 400

    filename = secure_filename(file.filename)
    run_id = create_run_id()
    upload_path = UPLOADS_DIR / f"{run_id}_{filename}"
    file.save(upload_path)

    try:
        threshold = parse_int(request.form.get("threshold"), 128, 0)
        invert = request.form.get("invert") == "on"
        max_size = parse_int(request.form.get("max_size"), 128, 16)
        detect_markers = request.form.get("detect_markers") == "on"
        auto_threshold = request.form.get("auto_threshold") == "on"
        cleanup = request.form.get("cleanup") == "on"
        min_component = parse_int(request.form.get("min_component"), 20, 1)
        steps = parse_int(request.form.get("steps"), 60, 2)
        dt = parse_float(request.form.get("dt"), 0.35, 0.01)
        gamma = parse_float(request.form.get("gamma"), 1.0, 0.01)
        solve = request.form.get("solve") == "on"
        gif = request.form.get("gif") == "on"
        backend = "qiskit" if request.form.get("qiskit") == "on" else "numpy"
        qiskit_compare = request.form.get("qiskit_compare") == "on"
        gif_steps_raw = request.form.get("gif_steps")
        gif_steps = None
        if gif_steps_raw:
            gif_steps = parse_int(gif_steps_raw, 0, 0)
            if gif_steps == 0:
                gif_steps = None
        gif_multiplier = parse_float(request.form.get("gif_multiplier"), 2.0, 0.1)
        start_xy = parse_point(request.form.get("start"))
        goal_xy = parse_point(request.form.get("goal"))
    except ValueError as exc:
        return render_template("upload.html", error=str(exc)), 400

    if backend == "qiskit" and not qiskit_available():
        return render_template("upload.html", error="Qiskit backend requested but qiskit is not installed."), 400
    if qiskit_compare and backend != "qiskit":
        return render_template("upload.html", error="Qiskit compare requires the Qiskit backend."), 400

    _set_job(run_id, "queued", "queued")
    thread = Thread(
        target=_run_job,
        kwargs={
            "run_id": run_id,
            "upload_path": upload_path,
            "threshold": threshold,
            "invert": invert,
            "max_size": max_size,
            "start_xy": start_xy,
            "goal_xy": goal_xy,
            "detect_markers": detect_markers,
            "auto_threshold": auto_threshold,
            "cleanup": cleanup,
            "min_component": min_component,
            "steps": steps,
            "dt": dt,
            "gamma": gamma,
            "solve": solve,
            "gif": gif,
            "gif_steps": gif_steps,
            "gif_multiplier": gif_multiplier,
            "backend": backend,
            "qiskit_compare": qiskit_compare,
        },
        daemon=True,
    )
    thread.start()

    return redirect(url_for("status", run_id=run_id))


@app.route("/generate", methods=["POST"])
def generate():
    run_id = create_run_id()
    try:
        width = parse_int(request.form.get("width"), 12, 2)
        height = parse_int(request.form.get("height"), 12, 2)
        seed_raw = request.form.get("seed")
        seed = int(seed_raw) if seed_raw else None
        steps = parse_int(request.form.get("steps"), 60, 2)
        dt = parse_float(request.form.get("dt"), 0.35, 0.01)
        gamma = parse_float(request.form.get("gamma"), 1.0, 0.01)
        solve = request.form.get("solve") == "on"
        gif = request.form.get("gif") == "on"
        backend = "qiskit" if request.form.get("qiskit") == "on" else "numpy"
        qiskit_compare = request.form.get("qiskit_compare") == "on"
        gif_steps_raw = request.form.get("gif_steps")
        gif_steps = None
        if gif_steps_raw:
            gif_steps = parse_int(gif_steps_raw, 0, 0)
            if gif_steps == 0:
                gif_steps = None
        gif_multiplier = parse_float(request.form.get("gif_multiplier"), 2.0, 0.1)
    except ValueError as exc:
        return render_template("upload.html", error=str(exc)), 400

    if backend == "qiskit" and not qiskit_available():
        return render_template("upload.html", error="Qiskit backend requested but qiskit is not installed."), 400
    if qiskit_compare and backend != "qiskit":
        return render_template("upload.html", error="Qiskit compare requires the Qiskit backend."), 400

    rows = generate_maze(width, height, seed=seed)

    _set_job(run_id, "queued", "queued")
    thread = Thread(
        target=_run_job_generated,
        kwargs={
            "run_id": run_id,
            "rows": rows,
            "steps": steps,
            "dt": dt,
            "gamma": gamma,
            "solve": solve,
            "gif": gif,
            "gif_steps": gif_steps,
            "gif_multiplier": gif_multiplier,
            "backend": backend,
            "qiskit_compare": qiskit_compare,
        },
        daemon=True,
    )
    thread.start()

    return redirect(url_for("status", run_id=run_id))


@app.route("/maze3d", methods=["POST"])
def maze3d_run():
    run_id = create_run_id()
    try:
        width = parse_int(request.form.get("width"), 6, 2)
        height = parse_int(request.form.get("height"), 6, 2)
        depth = parse_int(request.form.get("depth"), 6, 2)
        seed_raw = request.form.get("seed")
        seed = int(seed_raw) if seed_raw else None
        steps = parse_int(request.form.get("steps"), 60, 2)
        dt = parse_float(request.form.get("dt"), 0.35, 0.01)
        gamma = parse_float(request.form.get("gamma"), 1.0, 0.01)
        solve = request.form.get("solve") == "on"
        gif = request.form.get("gif") == "on"
        backend = "qiskit" if request.form.get("qiskit") == "on" else "numpy"
        qiskit_compare = request.form.get("qiskit_compare") == "on"
        gif_steps_raw = request.form.get("gif_steps")
        gif_steps = None
        if gif_steps_raw:
            gif_steps = parse_int(gif_steps_raw, 0, 0)
            if gif_steps == 0:
                gif_steps = None
        gif_multiplier = parse_float(request.form.get("gif_multiplier"), 2.0, 0.1)
    except ValueError as exc:
        return render_template("upload.html", error=str(exc)), 400

    if backend == "qiskit" and not qiskit_available():
        return render_template("upload.html", error="Qiskit backend requested but qiskit is not installed."), 400
    if qiskit_compare and backend != "qiskit":
        return render_template("upload.html", error="Qiskit compare requires the Qiskit backend."), 400

    _set_job(run_id, "queued", "queued")
    thread = Thread(
        target=_run_job_maze3d,
        kwargs={
            "run_id": run_id,
            "width": width,
            "height": height,
            "depth": depth,
            "seed": seed,
            "steps": steps,
            "dt": dt,
            "gamma": gamma,
            "solve": solve,
            "gif": gif,
            "gif_steps": gif_steps,
            "gif_multiplier": gif_multiplier,
            "backend": backend,
            "qiskit_compare": qiskit_compare,
        },
        daemon=True,
    )
    thread.start()

    return redirect(url_for("status", run_id=run_id))


@app.route("/polar", methods=["POST"])
def polar_run():
    run_id = create_run_id()
    try:
        rings = parse_int(request.form.get("rings"), 6, 2)
        sectors = parse_int(request.form.get("sectors"), 16, 4)
        seed_raw = request.form.get("seed")
        seed = int(seed_raw) if seed_raw else None
        steps = parse_int(request.form.get("steps"), 60, 2)
        dt = parse_float(request.form.get("dt"), 0.35, 0.01)
        gamma = parse_float(request.form.get("gamma"), 1.0, 0.01)
        solve = request.form.get("solve") == "on"
        gif = request.form.get("gif") == "on"
        backend = "qiskit" if request.form.get("qiskit") == "on" else "numpy"
        qiskit_compare = request.form.get("qiskit_compare") == "on"
        gif_steps_raw = request.form.get("gif_steps")
        gif_steps = None
        if gif_steps_raw:
            gif_steps = parse_int(gif_steps_raw, 0, 0)
            if gif_steps == 0:
                gif_steps = None
        gif_multiplier = parse_float(request.form.get("gif_multiplier"), 2.0, 0.1)
    except ValueError as exc:
        return render_template("upload.html", error=str(exc)), 400

    if backend == "qiskit" and not qiskit_available():
        return render_template("upload.html", error="Qiskit backend requested but qiskit is not installed."), 400
    if qiskit_compare and backend != "qiskit":
        return render_template("upload.html", error="Qiskit compare requires the Qiskit backend."), 400

    _set_job(run_id, "queued", "queued")
    thread = Thread(
        target=_run_job_polar,
        kwargs={
            "run_id": run_id,
            "rings": rings,
            "sectors": sectors,
            "seed": seed,
            "steps": steps,
            "dt": dt,
            "gamma": gamma,
            "solve": solve,
            "gif": gif,
            "gif_steps": gif_steps,
            "gif_multiplier": gif_multiplier,
            "backend": backend,
            "qiskit_compare": qiskit_compare,
        },
        daemon=True,
    )
    thread.start()

    return redirect(url_for("status", run_id=run_id))


@app.route("/preview", methods=["POST"])
def preview():
    file = request.files.get("maze_image")
    if file is None or file.filename == "":
        return jsonify({"error": "missing file"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "unsupported file type"}), 400

    filename = secure_filename(file.filename)
    temp_path = UPLOADS_DIR / f"preview_{uuid4().hex}_{filename}"
    file.save(temp_path)

    try:
        threshold = parse_int(request.form.get("threshold"), 128, 0)
        invert = request.form.get("invert") == "on"
        max_size = parse_int(request.form.get("max_size"), 128, 16)
        detect_markers = request.form.get("detect_markers") == "on"
        auto_threshold = request.form.get("auto_threshold") == "on"
        cleanup = request.form.get("cleanup") == "on"
        min_component = parse_int(request.form.get("min_component"), 20, 1)
        start_xy = parse_point(request.form.get("start"))
        goal_xy = parse_point(request.form.get("goal"))
    except ValueError as exc:
        temp_path.unlink(missing_ok=True)
        return jsonify({"error": str(exc)}), 400

    try:
        _, graph_info, _ = load_maze_image(
            temp_path,
            threshold=threshold,
            invert=invert,
            max_size=max_size,
            start_xy=start_xy,
            goal_xy=goal_xy,
            detect_markers=detect_markers,
            auto_threshold=auto_threshold,
            cleanup=cleanup,
            min_component=min_component,
        )
        preview_data = build_preview_image(
            graph_info["maze"],
            tuple(graph_info["start_xy"]),
            tuple(graph_info["goal_xy"]),
        )
    except (MazeError, ValueError) as exc:
        temp_path.unlink(missing_ok=True)
        return jsonify({"error": f"Maze error: {exc}"}), 400

    temp_path.unlink(missing_ok=True)
    return jsonify(
        {
            "preview": preview_data,
            "start": graph_info["start_xy"],
            "goal": graph_info["goal_xy"],
            "endpoint_mode": graph_info.get("endpoint_mode"),
        }
    )


@app.route("/hypercube", methods=["POST"])
def hypercube_run():
    run_id = create_run_id()
    try:
        dimensions = parse_int(request.form.get("dimensions"), 6, 1)
        steps = parse_int(request.form.get("steps"), 60, 2)
        dt = parse_float(request.form.get("dt"), 0.35, 0.01)
        gamma = parse_float(request.form.get("gamma"), 1.0, 0.01)
        dynamic = request.form.get("dynamic") == "on"
        shift_rate = parse_float(request.form.get("shift_rate"), 0.2, 0.0)
        seed_raw = request.form.get("seed")
        seed = int(seed_raw) if seed_raw else None
        gif = request.form.get("gif") == "on"
        gif_3d = request.form.get("gif_3d") == "on"
        backend = "qiskit" if request.form.get("qiskit") == "on" else "numpy"
        qiskit_compare = request.form.get("qiskit_compare") == "on"
    except ValueError as exc:
        return render_template("upload.html", error=str(exc)), 400

    if backend == "qiskit" and not qiskit_available():
        return render_template("upload.html", error="Qiskit backend requested but qiskit is not installed."), 400
    if backend == "qiskit" and dynamic:
        return render_template("upload.html", error="Qiskit backend is not supported for dynamic hypercubes."), 400
    if qiskit_compare and backend != "qiskit":
        return render_template("upload.html", error="Qiskit compare requires the Qiskit backend."), 400

    _set_job(run_id, "queued", "queued")
    thread = Thread(
        target=_run_job_hypercube,
        kwargs={
            "run_id": run_id,
            "dimensions": dimensions,
            "steps": steps,
            "dt": dt,
            "gamma": gamma,
            "dynamic": dynamic,
            "shift_rate": shift_rate,
            "seed": seed,
            "gif": gif,
            "gif_3d": gif_3d,
            "backend": backend,
            "qiskit_compare": qiskit_compare,
        },
        daemon=True,
    )
    thread.start()

    return redirect(url_for("status", run_id=run_id))


@app.route("/cube", methods=["POST"])
def cube_run():
    run_id = create_run_id()
    try:
        size = parse_int(request.form.get("size"), 3, 2)
        steps = parse_int(request.form.get("steps"), 60, 2)
        dt = parse_float(request.form.get("dt"), 0.35, 0.01)
        gamma = parse_float(request.form.get("gamma"), 1.0, 0.01)
        shift_rate = parse_float(request.form.get("shift_rate"), 0.3, 0.0)
        seed_raw = request.form.get("seed")
        seed = int(seed_raw) if seed_raw else None
        gif = request.form.get("gif") == "on"
        backend = "qiskit" if request.form.get("qiskit") == "on" else "numpy"
    except ValueError as exc:
        return render_template("upload.html", error=str(exc)), 400

    if backend == "qiskit":
        return render_template("upload.html", error="Qiskit backend is not supported for cube puzzles yet."), 400

    _set_job(run_id, "queued", "queued")
    thread = Thread(
        target=_run_job_cube,
        kwargs={
            "run_id": run_id,
            "size": size,
            "steps": steps,
            "dt": dt,
            "gamma": gamma,
            "shift_rate": shift_rate,
            "seed": seed,
            "gif": gif,
        },
        daemon=True,
    )
    thread.start()

    return redirect(url_for("status", run_id=run_id))


@app.route("/solutions")
def solutions():
    runs = list_runs()
    focus = request.args.get("focus")
    return render_template("solutions.html", runs=runs, focus=focus)


@app.route("/status/<run_id>")
def status(run_id: str):
    job = _get_job(run_id)
    if job is None:
        abort(404)
    return render_template("status.html", run_id=run_id, job=job)


@app.route("/api/runs/<run_id>")
def run_status(run_id: str):
    job = _get_job(run_id)
    if job is None:
        return jsonify({"error": "not found"}), 404
    if job.get("status") == "running":
        started_at = job.get("started_at")
        if started_at:
            try:
                started = datetime.fromisoformat(started_at)
                if (datetime.now() - started).total_seconds() > JOB_TIMEOUT_SECONDS:
                    _set_job(run_id, "error", "timeout", "job exceeded time limit")
                    job = _get_job(run_id) or job
            except ValueError:
                pass
    return jsonify(job)


@app.route("/api/runs/<run_id>/logs")
def run_logs(run_id: str):
    since_raw = request.args.get("since", "0")
    try:
        since = int(since_raw)
    except ValueError:
        since = 0
    result = _get_job_logs(run_id, since)
    if result is None:
        return jsonify({"error": "not found"}), 404
    return jsonify(result)

@app.route("/runs/<path:filename>")
def run_file(filename: str):
    full_path = (RUNS_DIR / filename).resolve()
    if RUNS_DIR not in full_path.parents and full_path != RUNS_DIR:
        abort(404)
    if not full_path.exists():
        abort(404)
    return send_from_directory(RUNS_DIR, filename)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
