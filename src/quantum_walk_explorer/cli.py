"""CLI entry points for Quantum Walk Explorer."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .graph import grid_graph, line_graph
from .maze import (
    MazeError,
    generate_maze,
    generate_maze_3d,
    generate_polar_maze,
    load_maze_image,
    load_maze_graph,
    load_maze3d_graph,
    load_polar_graph,
    shortest_path,
    shortest_path_adjacency,
    write_maze,
    write_maze3d,
    write_polar_maze,
)
from .hypercube import (
    dynamic_hypercube_adjacencies,
    generate_hypercube,
    load_hypercube_graph,
    write_hypercube,
)
from .cube_puzzle import dynamic_cube_adjacencies
from .qiskit_backend import (
    build_qiskit_circuit,
    continuous_time_walk_qiskit,
    qiskit_available,
    save_qiskit_artifacts,
)
from .runlog import create_payload, load_run, save_run
from .visualize import (
    assemble_gif,
    plot_grid_heatmap,
    plot_hypercube_projection,
    plot_cube_projection,
    plot_line,
    plot_maze_3d,
    plot_maze3d_heatmap,
    plot_maze_heatmap,
    plot_polar_heatmap,
    render_grid_frames,
    render_cube_frames,
    render_hypercube_frames_3d,
    render_hypercube_frames,
    render_line_frames,
    render_maze3d_frames,
    render_maze_frames,
    render_polar_frames,
)
from .walk import basis_state, continuous_time_walk, probabilities, time_dependent_walk, top_k


def _build_graph(args):
    if args.graph == "grid":
        adjacency, positions = grid_graph(args.width, args.height)
        graph_info = {
            "type": "grid",
            "width": args.width,
            "height": args.height,
            "positions": positions,
        }
        return adjacency, graph_info, None

    if args.graph == "line":
        adjacency, positions = line_graph(args.nodes)
        graph_info = {
            "type": "line",
            "nodes": args.nodes,
            "positions": positions,
        }
        return adjacency, graph_info, None

    if args.graph == "maze":
        if args.maze and args.maze_image:
            raise ValueError("use only one of --maze or --maze-image")
        if args.maze_image:
            adjacency, graph_info, coord_to_index = load_maze_image(
                args.maze_image,
                threshold=args.image_threshold,
                invert=args.image_invert,
                max_size=args.image_max_size,
                start_xy=tuple(args.start) if args.start else None,
                goal_xy=tuple(args.goal) if args.goal else None,
                detect_markers=args.image_detect_markers,
                auto_threshold=args.image_auto_threshold,
                cleanup=args.image_cleanup,
                min_component=args.image_min_component,
            )
        else:
            if not args.maze:
                raise ValueError("--maze is required for maze graphs")
            adjacency, graph_info, coord_to_index = load_maze_graph(args.maze)
            if args.goal is not None:
                gx, gy = args.goal
                if (gx, gy) not in coord_to_index:
                    raise ValueError("goal position is a wall or out of bounds")
                graph_info["goal_xy"] = [gx, gy]
        return adjacency, graph_info, coord_to_index

    if args.graph == "polar":
        if not args.maze:
            raise ValueError("--maze is required for polar graphs")
        adjacency, graph_info, coord_to_index = load_polar_graph(args.maze)
        return adjacency, graph_info, coord_to_index

    if args.graph == "maze3d":
        if not args.maze:
            raise ValueError("--maze is required for maze3d graphs")
        adjacency, graph_info, coord_to_index = load_maze3d_graph(args.maze)
        return adjacency, graph_info, coord_to_index

    if args.graph == "hypercube":
        if not args.maze:
            raise ValueError("--maze is required for hypercube graphs")
        adjacency, graph_info, coord_to_index = load_hypercube_graph(args.maze)
        return adjacency, graph_info, coord_to_index

    raise ValueError(f"Unsupported graph type: {args.graph}")


def _default_start_index(graph_info: Dict[str, Any]) -> int:
    if graph_info["type"] == "grid":
        width = graph_info["width"]
        height = graph_info["height"]
        x = width // 2
        y = height // 2
        return y * width + x
    if graph_info["type"] == "line":
        return graph_info["nodes"] // 2
    raise ValueError("unknown graph type")


def _resolve_start_index(
    args,
    graph_info: Dict[str, Any],
    coord_to_index: Optional[Dict[Tuple[int, ...], int]] = None,
) -> Tuple[int, Optional[Tuple[int, int]]]:
    if graph_info["type"] == "grid":
        if args.start is not None:
            x, y = args.start
            if not (0 <= x < graph_info["width"] and 0 <= y < graph_info["height"]):
                raise ValueError("start position out of bounds")
            return y * graph_info["width"] + x, (x, y)
        index = _default_start_index(graph_info)
        x = index % graph_info["width"]
        y = index // graph_info["width"]
        return index, (x, y)

    if graph_info["type"] == "line":
        if args.start_index is not None:
            if not (0 <= args.start_index < graph_info["nodes"]):
                raise ValueError("start index out of bounds")
            return args.start_index, None
        return _default_start_index(graph_info), None

    if graph_info["type"] == "maze":
        if coord_to_index is None:
            raise ValueError("maze coordinate map is missing")
        if args.start is not None:
            x, y = args.start
            if (x, y) not in coord_to_index:
                raise ValueError("start position is a wall or out of bounds")
            return coord_to_index[(x, y)], (x, y)
        start_xy = tuple(graph_info["start_xy"])
        return coord_to_index[start_xy], start_xy

    if graph_info["type"] == "polar":
        if coord_to_index is None:
            raise ValueError("polar coordinate map is missing")
        if args.start is not None:
            ring, sector = args.start
            if not (0 <= ring < graph_info["rings"] and 0 <= sector < graph_info["sectors"]):
                raise ValueError("start ring/sector out of bounds")
            return coord_to_index[(ring, sector)], (ring, sector)
        start_rs = tuple(graph_info["start_rs"])
        return coord_to_index[start_rs], start_rs

    if graph_info["type"] == "hypercube":
        if args.start_index is not None:
            if not (0 <= args.start_index < graph_info["nodes"]):
                raise ValueError("start index out of bounds")
            return args.start_index, None
        return graph_info["start_index"], None

    if graph_info["type"] == "maze3d":
        if coord_to_index is None:
            raise ValueError("maze3d coordinate map is missing")
        start_xyz = tuple(graph_info["start_xyz"])
        return coord_to_index[start_xyz], None

    raise ValueError("unknown graph type")


def _timestamp_label(label: Optional[str]) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if label:
        safe = "".join(ch for ch in label if ch.isalnum() or ch in ("-", "_"))
        if safe:
            return f"{stamp}_{safe}"
    return stamp


def run_command(args) -> int:
    try:
        adjacency, graph_info, coord_to_index = _build_graph(args)
    except MazeError as exc:
        raise ValueError(f"maze error: {exc}") from exc
    start_index, start_xy = _resolve_start_index(args, graph_info, coord_to_index)

    solve_path = None
    solve_path_kind = None
    if args.solve:
        if graph_info["type"] == "maze":
            start = start_xy if start_xy is not None else tuple(graph_info["start_xy"])
            goal = tuple(graph_info["goal_xy"])
            solve_path = shortest_path(graph_info["maze"], start, goal)
            solve_path_kind = "coord"
        elif graph_info["type"] == "polar":
            if coord_to_index is None:
                raise ValueError("polar coordinate map is missing")
            goal_rs = tuple(graph_info["goal_rs"])
            goal_index = coord_to_index[goal_rs]
            path_indices = shortest_path_adjacency(adjacency, start_index, goal_index)
            if path_indices is not None:
                solve_path = [graph_info["positions"][idx] for idx in path_indices]
                solve_path_kind = "coord"
        elif graph_info["type"] == "hypercube":
            goal_index = graph_info["goal_index"]
            path_indices = shortest_path_adjacency(adjacency, start_index, goal_index)
            if path_indices is not None:
                solve_path = path_indices
                solve_path_kind = "index"
        elif graph_info["type"] == "maze3d":
            if coord_to_index is None:
                raise ValueError("maze3d coordinate map is missing")
            goal_xyz = tuple(graph_info["goal_xyz"])
            goal_index = coord_to_index[goal_xyz]
            path_indices = shortest_path_adjacency(adjacency, start_index, goal_index)
            if path_indices is not None:
                solve_path = path_indices
                solve_path_kind = "index"

    sim_steps = args.steps
    if args.gif_steps is not None:
        if args.gif_steps < 2:
            raise ValueError("--gif-steps must be >= 2")
        sim_steps = max(sim_steps, args.gif_steps)
    elif args.solve and args.gif and solve_path is not None:
        if args.gif_path_multiplier <= 0:
            raise ValueError("--gif-path-multiplier must be > 0")
        target = int(math.ceil(len(solve_path) * args.gif_path_multiplier))
        sim_steps = max(sim_steps, target)

    times = np.arange(sim_steps) * args.dt
    start_state = basis_state(adjacency.shape[0], start_index)

    dynamic_adjacencies = None
    if graph_info["type"] == "hypercube" and args.hypercube_dynamic:
        dynamic_adjacencies = dynamic_hypercube_adjacencies(
            graph_info["dimensions"],
            sim_steps,
            seed=args.hypercube_seed,
            shift_rate=args.hypercube_shift_rate,
        )

    if args.backend == "qiskit":
        if not qiskit_available():
            raise ValueError("qiskit backend requested but qiskit is not installed")
        states = continuous_time_walk_qiskit(adjacency, args.gamma, times, start_state)
    else:
        if dynamic_adjacencies is not None:
            states = time_dependent_walk(dynamic_adjacencies, args.gamma, args.dt, start_state)
        else:
            states = continuous_time_walk(adjacency, args.gamma, times, start_state)
    probs = probabilities(states)

    qiskit_compare = None
    if args.qiskit_compare:
        if dynamic_adjacencies is not None:
            print("Qiskit compare skipped: dynamic graphs are not supported")
        else:
            if not qiskit_available():
                raise ValueError("qiskit compare requested but qiskit is not installed")
            if args.backend == "qiskit":
                states_qiskit = states
                states_numpy = continuous_time_walk(adjacency, args.gamma, times, start_state)
            else:
                states_numpy = states
                states_qiskit = continuous_time_walk_qiskit(adjacency, args.gamma, times, start_state)
            probs_numpy = probabilities(states_numpy)
            probs_qiskit = probabilities(states_qiskit)
            qiskit_compare = float(np.max(np.abs(probs_numpy - probs_qiskit)))
            print(f"Qiskit compare max error: {qiskit_compare:.6e}")

    params = {
        "gamma": args.gamma,
        "dt": args.dt,
        "steps": sim_steps,
        "start_index": start_index,
        "backend": args.backend,
    }
    if qiskit_compare is not None:
        params["qiskit_compare_max_error"] = qiskit_compare
    if dynamic_adjacencies is not None:
        params["dynamic"] = True
        params["shift_rate"] = args.hypercube_shift_rate
        params["dynamic_seed"] = args.hypercube_seed
    if sim_steps != args.steps:
        params["steps_requested"] = args.steps
    if args.gif_steps is not None:
        params["gif_steps"] = args.gif_steps
    if args.solve and args.gif and solve_path is not None:
        params["gif_path_multiplier"] = args.gif_path_multiplier
    if start_xy is not None:
        if graph_info["type"] == "polar":
            params["start_rs"] = list(start_xy)
            graph_info["start_rs"] = list(start_xy)
        else:
            params["start_xy"] = list(start_xy)
            if graph_info["type"] == "maze":
                graph_info["start_xy"] = list(start_xy)

    payload = create_payload(
        graph=graph_info,
        params=params,
        times=times.tolist(),
        probabilities=probs.tolist(),
    )

    run_dir = Path(args.outdir) / _timestamp_label(args.label)
    run_path = save_run(run_dir / "run.json", payload)

    print(f"Saved run to {run_path}")

    if args.qiskit_artifacts:
        if dynamic_adjacencies is not None:
            print("Qiskit artifacts skipped: dynamic graphs are not supported")
        else:
            if not qiskit_available():
                raise ValueError("qiskit artifacts requested but qiskit is not installed")
            circuit = build_qiskit_circuit(adjacency, args.gamma, times[-1], start_state)
            artifacts = save_qiskit_artifacts(circuit, run_dir)
            if artifacts:
                print("Saved Qiskit artifacts:")
                for key, path in artifacts.items():
                    print(f"  {key}: {path}")
            else:
                print("No Qiskit artifacts were generated")

    if qiskit_compare is not None:
        compare_path = run_dir / "qiskit_compare.json"
        compare_payload = {
            "max_error": qiskit_compare,
            "backend": args.backend,
            "steps": sim_steps,
        }
        compare_path.write_text(json.dumps(compare_payload, indent=2), encoding="utf-8")
        print(f"Saved {compare_path}")

    last_probs = probs[-1]
    best = top_k(last_probs, k=args.top)
    print("Top nodes:")
    for idx, prob in best:
        if graph_info["type"] == "grid":
            x = idx % graph_info["width"]
            y = idx // graph_info["width"]
            print(f"  node {idx} (x={x}, y={y}) -> {prob:.4f}")
        elif graph_info["type"] == "maze":
            x, y = graph_info["positions"][idx]
            print(f"  node {idx} (x={x}, y={y}) -> {prob:.4f}")
        elif graph_info["type"] == "polar":
            ring, sector = graph_info["positions"][idx]
            print(f"  node {idx} (ring={ring}, sector={sector}) -> {prob:.4f}")
        else:
            print(f"  node {idx} -> {prob:.4f}")

    if args.heatmap:
        if graph_info["type"] == "grid":
            plot_grid_heatmap(
                last_probs,
                graph_info["width"],
                graph_info["height"],
                "Final probability",
                run_dir / "heatmap.png",
            )
            print(f"Wrote {run_dir / 'heatmap.png'}")
        elif graph_info["type"] == "maze":
            plot_maze_heatmap(
                last_probs,
                graph_info["width"],
                graph_info["height"],
                graph_info["positions"],
                "Final probability",
                run_dir / "heatmap.png",
                start=start_xy,
                goal=tuple(graph_info["goal_xy"]),
            )
            print(f"Wrote {run_dir / 'heatmap.png'}")
        elif graph_info["type"] == "polar":
            plot_polar_heatmap(
                last_probs,
                graph_info["rings"],
                graph_info["sectors"],
                "Final probability",
                run_dir / "heatmap.png",
                edges=graph_info.get("edges"),
                start=start_xy,
                goal=tuple(graph_info["goal_rs"]),
            )
            print(f"Wrote {run_dir / 'heatmap.png'}")
        elif graph_info["type"] == "maze3d":
            goal_index = None
            if coord_to_index is not None:
                goal_xyz = tuple(graph_info["goal_xyz"])
                goal_index = coord_to_index.get(goal_xyz)
            plot_maze3d_heatmap(
                last_probs,
                graph_info["width"],
                graph_info["height"],
                graph_info["depth"],
                graph_info["positions"],
                "Final probability",
                run_dir / "heatmap.png",
                start_index=start_index,
                goal_index=goal_index,
            )
            print(f"Wrote {run_dir / 'heatmap.png'}")
        elif graph_info["type"] == "hypercube":
            plot_hypercube_projection(
                last_probs,
                graph_info["dimensions"],
                "Final probability",
                run_dir / "heatmap.png",
                start_index=graph_info.get("start_index"),
                goal_index=graph_info.get("goal_index"),
            )
            print(f"Wrote {run_dir / 'heatmap.png'}")
        else:
            plot_line(last_probs, "Final probability", run_dir / "bar.png")
            print(f"Wrote {run_dir / 'bar.png'}")

    if args.gif:
        if graph_info["type"] == "grid":
            frames = render_grid_frames(probs, graph_info["width"], graph_info["height"], run_dir / "frames")
            try:
                assemble_gif(frames, run_dir / "walk.gif", fps=args.fps)
                print(f"Wrote {run_dir / 'walk.gif'}")
            except RuntimeError as exc:
                print(f"GIF export skipped: {exc}")
        elif graph_info["type"] == "maze":
            frames = render_maze_frames(
                probs,
                graph_info["width"],
                graph_info["height"],
                graph_info["positions"],
                run_dir / "frames",
            )
            try:
                assemble_gif(frames, run_dir / "walk.gif", fps=args.fps)
                print(f"Wrote {run_dir / 'walk.gif'}")
            except RuntimeError as exc:
                print(f"GIF export skipped: {exc}")
        elif graph_info["type"] == "polar":
            frames = render_polar_frames(
                probs,
                graph_info["rings"],
                graph_info["sectors"],
                run_dir / "frames",
                edges=graph_info.get("edges"),
            )
            try:
                assemble_gif(frames, run_dir / "walk.gif", fps=args.fps)
                print(f"Wrote {run_dir / 'walk.gif'}")
            except RuntimeError as exc:
                print(f"GIF export skipped: {exc}")
        elif graph_info["type"] == "hypercube":
            if args.hypercube_gif_3d:
                frames = render_hypercube_frames_3d(probs, graph_info["dimensions"], run_dir / "frames")
            else:
                frames = render_hypercube_frames(probs, graph_info["dimensions"], run_dir / "frames")
            try:
                assemble_gif(frames, run_dir / "walk.gif", fps=args.fps)
                print(f"Wrote {run_dir / 'walk.gif'}")
            except RuntimeError as exc:
                print(f"GIF export skipped: {exc}")
        elif graph_info["type"] == "maze3d":
            frames = render_maze3d_frames(
                probs,
                graph_info["width"],
                graph_info["height"],
                graph_info["depth"],
                graph_info["positions"],
                graph_info["edges"],
                run_dir / "frames",
            )
            try:
                assemble_gif(frames, run_dir / "walk.gif", fps=args.fps)
                print(f"Wrote {run_dir / 'walk.gif'}")
            except RuntimeError as exc:
                print(f"GIF export skipped: {exc}")
        elif graph_info["type"] == "line":
            frames = render_line_frames(probs, run_dir / "frames")
            try:
                assemble_gif(frames, run_dir / "walk.gif", fps=args.fps)
                print(f"Wrote {run_dir / 'walk.gif'}")
            except RuntimeError as exc:
                print(f"GIF export skipped: {exc}")
        else:
            print("GIF export only supports grid, maze, maze3d, polar, line, or hypercube graphs")

    if args.solve:
        if graph_info["type"] not in ("maze", "polar", "hypercube", "maze3d"):
            print("--solve only applies to maze, polar, hypercube, or maze3d graphs")
        else:
            path = solve_path
            if path is None:
                print("No path found from start to goal")
            else:
                if graph_info["type"] == "hypercube":
                    labels = [graph_info["positions"][idx] for idx in path] if solve_path_kind == "index" else []
                    solution = {
                        "path_indices": path,
                        "path_labels": labels,
                        "length": max(len(path) - 1, 0),
                    }
                elif graph_info["type"] == "maze3d":
                    path_xyz = [graph_info["positions"][idx] for idx in path] if solve_path_kind == "index" else []
                    solution = {
                        "path_indices": path,
                        "path_xyz": path_xyz,
                        "length": max(len(path) - 1, 0),
                    }
                else:
                    solution = {
                        "path": [list(step) for step in path],
                        "length": max(len(path) - 1, 0),
                    }
                solution_path = run_dir / "solution.json"
                with solution_path.open("w", encoding="utf-8") as handle:
                    json.dump(solution, handle, indent=2)
                print(f"Wrote {solution_path}")
                if graph_info["type"] == "maze":
                    plot_maze_heatmap(
                        last_probs,
                        graph_info["width"],
                        graph_info["height"],
                        graph_info["positions"],
                        "Maze solution",
                        run_dir / "solution.png",
                        start=start_xy if start_xy is not None else tuple(graph_info["start_xy"]),
                        goal=tuple(graph_info["goal_xy"]),
                        path=path,
                    )
                    print(f"Wrote {run_dir / 'solution.png'}")
                elif graph_info["type"] == "polar":
                    plot_polar_heatmap(
                        last_probs,
                        graph_info["rings"],
                        graph_info["sectors"],
                        "Polar maze solution",
                        run_dir / "solution.png",
                        edges=graph_info.get("edges"),
                        start=start_xy if start_xy is not None else tuple(graph_info["start_rs"]),
                        goal=tuple(graph_info["goal_rs"]),
                        path=path,
                    )
                    print(f"Wrote {run_dir / 'solution.png'}")
                elif graph_info["type"] == "maze3d":
                    goal_index = None
                    if coord_to_index is not None:
                        goal_xyz = tuple(graph_info["goal_xyz"])
                        goal_index = coord_to_index.get(goal_xyz)
                    plot_maze_3d(
                        last_probs,
                        graph_info["width"],
                        graph_info["height"],
                        graph_info["depth"],
                        graph_info["positions"],
                        graph_info["edges"],
                        "Maze3D solution",
                        run_dir / "solution.png",
                        start_index=start_index,
                        goal_index=goal_index,
                        path_indices=path if solve_path_kind == "index" else None,
                    )
                    print(f"Wrote {run_dir / 'solution.png'}")
                elif graph_info["type"] == "hypercube":
                    plot_hypercube_projection(
                        last_probs,
                        graph_info["dimensions"],
                        "Hypercube solution",
                        run_dir / "solution.png",
                        start_index=graph_info.get("start_index"),
                        goal_index=graph_info.get("goal_index"),
                        path_indices=path if solve_path_kind == "index" else None,
                    )
                    print(f"Wrote {run_dir / 'solution.png'}")

    return 0


def replay_command(args) -> int:
    payload = load_run(args.run)
    graph = payload.get("graph", {})
    graph_type = graph.get("type")

    probs = np.array(payload.get("probabilities", []), dtype=float)
    if probs.size == 0:
        raise ValueError("run file has no probabilities")

    run_dir = Path(args.output_dir) if args.output_dir else Path(args.run).parent
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.heatmap:
        last_probs = probs[-1]
        if graph_type == "grid":
            width = graph.get("width")
            height = graph.get("height")
            if width is None or height is None:
                raise ValueError("grid run missing width/height")
            plot_grid_heatmap(last_probs, width, height, "Final probability", run_dir / "heatmap_replay.png")
            print(f"Wrote {run_dir / 'heatmap_replay.png'}")
        elif graph_type == "maze":
            width = graph.get("width")
            height = graph.get("height")
            positions = graph.get("positions")
            start_xy = graph.get("start_xy")
            goal_xy = graph.get("goal_xy")
            if width is None or height is None or positions is None:
                raise ValueError("maze run missing width/height/positions")
            plot_maze_heatmap(
                last_probs,
                width,
                height,
                positions,
                "Final probability",
                run_dir / "heatmap_replay.png",
                start=tuple(start_xy) if start_xy else None,
                goal=tuple(goal_xy) if goal_xy else None,
            )
            print(f"Wrote {run_dir / 'heatmap_replay.png'}")
        elif graph_type == "polar":
            rings = graph.get("rings")
            sectors = graph.get("sectors")
            start_rs = graph.get("start_rs")
            goal_rs = graph.get("goal_rs")
            if rings is None or sectors is None:
                raise ValueError("polar run missing rings/sectors")
            plot_polar_heatmap(
                last_probs,
                rings,
                sectors,
                "Final probability",
                run_dir / "heatmap_replay.png",
                edges=graph.get("edges"),
                start=tuple(start_rs) if start_rs else None,
                goal=tuple(goal_rs) if goal_rs else None,
            )
            print(f"Wrote {run_dir / 'heatmap_replay.png'}")
        elif graph_type == "hypercube":
            dimensions = graph.get("dimensions")
            if dimensions is None:
                raise ValueError("hypercube run missing dimensions")
            plot_hypercube_projection(
                last_probs,
                dimensions,
                "Final probability",
                run_dir / "heatmap_replay.png",
                start_index=graph.get("start_index"),
                goal_index=graph.get("goal_index"),
            )
            print(f"Wrote {run_dir / 'heatmap_replay.png'}")
        elif graph_type == "maze3d":
            width = graph.get("width")
            height = graph.get("height")
            depth = graph.get("depth")
            positions = graph.get("positions")
            edges = graph.get("edges")
            start_xyz = graph.get("start_xyz")
            goal_xyz = graph.get("goal_xyz")
            if width is None or height is None or depth is None or positions is None or edges is None:
                raise ValueError("maze3d run missing width/height/depth/positions/edges")
            start_index = None
            goal_index = None
            if start_xyz and positions:
                try:
                    start_index = positions.index(start_xyz)
                except ValueError:
                    start_index = None
            if goal_xyz and positions:
                try:
                    goal_index = positions.index(goal_xyz)
                except ValueError:
                    goal_index = None
            plot_maze3d_heatmap(
                last_probs,
                width,
                height,
                depth,
                positions,
                "Final probability",
                run_dir / "heatmap_replay.png",
                start_index=start_index,
                goal_index=goal_index,
            )
            print(f"Wrote {run_dir / 'heatmap_replay.png'}")
        elif graph_type == "line":
            plot_line(last_probs, "Final probability", run_dir / "bar_replay.png")
            print(f"Wrote {run_dir / 'bar_replay.png'}")
        else:
            print("Replay heatmap skipped: unknown graph type")

    if args.gif:
        if graph_type == "grid":
            width = graph.get("width")
            height = graph.get("height")
            if width is None or height is None:
                raise ValueError("grid run missing width/height")
            frames = render_grid_frames(probs, width, height, run_dir / "frames_replay")
            try:
                assemble_gif(frames, run_dir / "walk_replay.gif", fps=args.fps)
                print(f"Wrote {run_dir / 'walk_replay.gif'}")
            except RuntimeError as exc:
                print(f"GIF export skipped: {exc}")
        elif graph_type == "maze":
            width = graph.get("width")
            height = graph.get("height")
            positions = graph.get("positions")
            if width is None or height is None or positions is None:
                raise ValueError("maze run missing width/height/positions")
            frames = render_maze_frames(probs, width, height, positions, run_dir / "frames_replay")
            try:
                assemble_gif(frames, run_dir / "walk_replay.gif", fps=args.fps)
                print(f"Wrote {run_dir / 'walk_replay.gif'}")
            except RuntimeError as exc:
                print(f"GIF export skipped: {exc}")
        elif graph_type == "polar":
            rings = graph.get("rings")
            sectors = graph.get("sectors")
            if rings is None or sectors is None:
                raise ValueError("polar run missing rings/sectors")
            frames = render_polar_frames(probs, rings, sectors, run_dir / "frames_replay", edges=graph.get("edges"))
            try:
                assemble_gif(frames, run_dir / "walk_replay.gif", fps=args.fps)
                print(f"Wrote {run_dir / 'walk_replay.gif'}")
            except RuntimeError as exc:
                print(f"GIF export skipped: {exc}")
        elif graph_type == "maze3d":
            width = graph.get("width")
            height = graph.get("height")
            depth = graph.get("depth")
            positions = graph.get("positions")
            edges = graph.get("edges")
            if width is None or height is None or depth is None or positions is None or edges is None:
                raise ValueError("maze3d run missing width/height/depth/positions/edges")
            frames = render_maze3d_frames(
                probs,
                width,
                height,
                depth,
                positions,
                edges,
                run_dir / "frames_replay",
            )
            try:
                assemble_gif(frames, run_dir / "walk_replay.gif", fps=args.fps)
                print(f"Wrote {run_dir / 'walk_replay.gif'}")
            except RuntimeError as exc:
                print(f"GIF export skipped: {exc}")
        elif graph_type in ("line", "hypercube"):
            if graph_type == "hypercube":
                dimensions = graph.get("dimensions")
                if dimensions is None:
                    raise ValueError("hypercube run missing dimensions")
                if args.hypercube_gif_3d:
                    frames = render_hypercube_frames_3d(probs, dimensions, run_dir / "frames_replay")
                else:
                    frames = render_hypercube_frames(probs, dimensions, run_dir / "frames_replay")
            else:
                frames = render_line_frames(probs, run_dir / "frames_replay")
            try:
                assemble_gif(frames, run_dir / "walk_replay.gif", fps=args.fps)
                print(f"Wrote {run_dir / 'walk_replay.gif'}")
            except RuntimeError as exc:
                print(f"GIF export skipped: {exc}")
        else:
            print("GIF export only supports grid, maze, maze3d, polar, line, or hypercube graphs")

    return 0


def maze_generate_command(args) -> int:
    try:
        if args.style == "polar":
            data = generate_polar_maze(args.rings, args.sectors, seed=args.seed)
            if args.output:
                output_path = Path(args.output)
            else:
                filename = f"polar_{_timestamp_label(args.label)}.json"
                output_path = Path(args.outdir) / filename
            write_polar_maze(output_path, data)
            print(f"Wrote {output_path}")
            if args.show:
                print(json.dumps(data, indent=2))
        elif args.style == "hypercube":
            data = generate_hypercube(args.dimensions)
            if args.output:
                output_path = Path(args.output)
            else:
                filename = f"hypercube_{_timestamp_label(args.label)}.json"
                output_path = Path(args.outdir) / filename
            write_hypercube(output_path, data)
            print(f"Wrote {output_path}")
            if args.show:
                print(json.dumps(data, indent=2))
        elif args.style == "maze3d":
            data = generate_maze_3d(args.width, args.height, args.depth, seed=args.seed)
            if args.output:
                output_path = Path(args.output)
            else:
                filename = f"maze3d_{_timestamp_label(args.label)}.json"
                output_path = Path(args.outdir) / filename
            write_maze3d(output_path, data)
            print(f"Wrote {output_path}")
            if args.show:
                print(json.dumps(data, indent=2))
        else:
            rows = generate_maze(args.width, args.height, seed=args.seed)
            if args.output:
                output_path = Path(args.output)
            else:
                filename = f"maze_{_timestamp_label(args.label)}.txt"
                output_path = Path(args.outdir) / filename
            write_maze(output_path, rows)
            print(f"Wrote {output_path}")
            if args.show:
                print("\n".join(rows))
    except MazeError as exc:
        raise ValueError(f"maze error: {exc}") from exc

    return 0


def cube_command(args) -> int:
    adjacencies, rotations = dynamic_cube_adjacencies(
        args.size,
        args.steps,
        seed=args.seed,
        shift_rate=args.shift_rate,
    )
    nodes = args.size ** 3
    start_index = 0
    goal_index = nodes - 1
    start_state = basis_state(nodes, start_index)

    states = time_dependent_walk(adjacencies, args.gamma, args.dt, start_state)
    probs = probabilities(states)

    goal_probs = probs[:, goal_index]
    best_step = int(np.argmax(goal_probs))
    best_prob = float(goal_probs[best_step])

    graph_info = {
        "type": "cube",
        "size": args.size,
        "nodes": nodes,
        "start_index": start_index,
        "goal_index": goal_index,
        "rotations": rotations,
    }

    params = {
        "gamma": args.gamma,
        "dt": args.dt,
        "steps": args.steps,
        "shift_rate": args.shift_rate,
        "seed": args.seed,
    }

    payload = create_payload(
        graph=graph_info,
        params=params,
        times=(np.arange(args.steps) * args.dt).tolist(),
        probabilities=probs.tolist(),
    )

    run_dir = Path(args.outdir) / _timestamp_label(args.label)
    run_path = save_run(run_dir / "run.json", payload)
    print(f"Saved run to {run_path}")

    solution = {
        "goal_index": goal_index,
        "best_step": best_step,
        "best_probability": best_prob,
    }
    solution_path = run_dir / "solution.json"
    with solution_path.open("w", encoding="utf-8") as handle:
        json.dump(solution, handle, indent=2)
    print(f"Wrote {solution_path}")

    if args.heatmap:
        plot_cube_projection(probs[-1], args.size, "Final probability", run_dir / "heatmap.png")
        print(f"Wrote {run_dir / 'heatmap.png'}")

    if args.gif:
        frames = render_cube_frames(probs, args.size, run_dir / "frames")
        try:
            assemble_gif(frames, run_dir / "walk.gif", fps=args.fps)
            print(f"Wrote {run_dir / 'walk.gif'}")
        except RuntimeError as exc:
            print(f"GIF export skipped: {exc}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Quantum Walk Explorer")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="run a new quantum walk")
    run.add_argument("--graph", choices=["grid", "line", "maze", "polar", "maze3d", "hypercube"], default="grid")
    run.add_argument("--width", type=int, default=8, help="grid width")
    run.add_argument("--height", type=int, default=8, help="grid height")
    run.add_argument("--nodes", type=int, default=16, help="line nodes")
    run.add_argument("--maze", help="path to ASCII maze or polar/hypercube JSON file")
    run.add_argument("--maze-image", help="path to maze image file")
    run.add_argument("--steps", type=int, default=24)
    run.add_argument("--dt", type=float, default=0.4)
    run.add_argument("--gamma", type=float, default=1.0)
    run.add_argument("--backend", choices=["numpy", "qiskit"], default="numpy")
    run.add_argument(
        "--start",
        type=int,
        nargs=2,
        metavar=("A", "B"),
        help="grid/maze start x y, or polar start ring sector",
    )
    run.add_argument("--goal", type=int, nargs=2, metavar=("X", "Y"), help="maze goal x y")
    run.add_argument("--image-threshold", type=int, default=128, help="maze image threshold 0-255")
    run.add_argument("--image-invert", action="store_true", help="invert maze image colors")
    run.add_argument("--image-max-size", type=int, default=128, help="max maze image size")
    run.add_argument("--image-detect-markers", dest="image_detect_markers", action="store_true", default=True)
    run.add_argument("--no-image-detect-markers", dest="image_detect_markers", action="store_false")
    run.add_argument("--image-auto-threshold", action="store_true", help="auto threshold (Otsu)")
    run.add_argument("--image-cleanup", action="store_true", help="remove tiny components in image mask")
    run.add_argument("--image-min-component", type=int, default=20, help="min size for open components")
    run.add_argument("--start-index", type=int, help="line start index")
    run.add_argument("--top", type=int, default=5, help="top nodes to report")
    run.add_argument("--outdir", default="runs")
    run.add_argument("--label", default=None)
    run.add_argument("--heatmap", dest="heatmap", action="store_true", default=True)
    run.add_argument("--no-heatmap", dest="heatmap", action="store_false")
    run.add_argument("--gif", action="store_true", help="export animated gif")
    run.add_argument("--fps", type=int, default=6)
    run.add_argument("--gif-steps", type=int, default=None, help="override GIF frame count")
    run.add_argument(
        "--gif-path-multiplier",
        type=float,
        default=2.0,
        help="maze solve frames = path length * multiplier (default: 2.0)",
    )
    run.add_argument("--solve", action="store_true", help="solve maze with BFS path")
    run.add_argument("--hypercube-dynamic", action="store_true", help="enable dynamic hypercube shifts")
    run.add_argument("--hypercube-shift-rate", type=float, default=0.2, help="hypercube shift probability per step")
    run.add_argument("--hypercube-seed", type=int, default=None, help="hypercube shift seed")
    run.add_argument("--hypercube-gif-3d", action="store_true", help="render hypercube GIF as 3D animation")
    run.add_argument("--qiskit-artifacts", action="store_true", help="export Qiskit circuit diagram + QASM")
    run.add_argument("--qiskit-compare", action="store_true", help="compare Qiskit vs NumPy probabilities")
    run.set_defaults(func=run_command)

    replay = subparsers.add_parser("replay", help="replay a logged run")
    replay.add_argument("--run", required=True, help="path to run.json")
    replay.add_argument("--output-dir", default=None)
    replay.add_argument("--heatmap", dest="heatmap", action="store_true", default=True)
    replay.add_argument("--no-heatmap", dest="heatmap", action="store_false")
    replay.add_argument("--gif", action="store_true")
    replay.add_argument("--fps", type=int, default=6)
    replay.add_argument("--hypercube-gif-3d", action="store_true", help="render hypercube GIF as 3D animation")
    replay.set_defaults(func=replay_command)

    maze = subparsers.add_parser("maze", help="generate a random maze")
    maze.add_argument("--style", choices=["grid", "polar", "maze3d", "hypercube"], default="grid")
    maze.add_argument("--width", type=int, default=10, help="grid maze cell width")
    maze.add_argument("--height", type=int, default=10, help="grid maze cell height")
    maze.add_argument("--depth", type=int, default=6, help="maze3d depth")
    maze.add_argument("--rings", type=int, default=6, help="polar maze ring count")
    maze.add_argument("--sectors", type=int, default=16, help="polar maze sector count")
    maze.add_argument("--seed", type=int, default=None, help="random seed")
    maze.add_argument("--dimensions", type=int, default=6, help="hypercube dimensions")
    maze.add_argument("--output", default=None, help="output maze path")
    maze.add_argument("--outdir", default="mazes", help="output directory")
    maze.add_argument("--label", default=None, help="label for the maze filename")
    maze.add_argument("--show", action="store_true", help="print maze to stdout")
    maze.set_defaults(func=maze_generate_command)

    cube = subparsers.add_parser("cube", help="run a dynamic cube puzzle")
    cube.add_argument("--size", type=int, default=3, help="cube size (default: 3)")
    cube.add_argument("--steps", type=int, default=60)
    cube.add_argument("--dt", type=float, default=0.35)
    cube.add_argument("--gamma", type=float, default=1.0)
    cube.add_argument("--shift-rate", type=float, default=0.3, help="rotation probability per step")
    cube.add_argument("--seed", type=int, default=None)
    cube.add_argument("--outdir", default="runs")
    cube.add_argument("--label", default=None)
    cube.add_argument("--heatmap", dest="heatmap", action="store_true", default=True)
    cube.add_argument("--no-heatmap", dest="heatmap", action="store_false")
    cube.add_argument("--gif", action="store_true")
    cube.add_argument("--fps", type=int, default=6)
    cube.set_defaults(func=cube_command)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)
