#!/usr/bin/env python3
"""Generate a small Qiskit-backed demo run with circuit artifacts."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np

from quantum_walk_explorer.graph import grid_graph
from quantum_walk_explorer.qiskit_backend import (
    build_qiskit_circuit,
    continuous_time_walk_qiskit,
    qiskit_available,
    save_qiskit_artifacts,
)
from quantum_walk_explorer.runlog import create_payload, save_run
from quantum_walk_explorer.visualize import plot_grid_heatmap
from quantum_walk_explorer.walk import basis_state, continuous_time_walk, probabilities


def main() -> int:
    if not qiskit_available():
        print("Qiskit is not installed. Run: pip install qiskit")
        return 1

    width = 4
    height = 4
    steps = 20
    dt = 0.35
    gamma = 1.0

    adjacency, positions = grid_graph(width, height)
    start_index = (height // 2) * width + (width // 2)
    start_state = basis_state(adjacency.shape[0], start_index)
    times = np.arange(steps) * dt

    states_qiskit = continuous_time_walk_qiskit(adjacency, gamma, times, start_state)
    probs_qiskit = probabilities(states_qiskit)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("runs") / f"qiskit_demo_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    payload = create_payload(
        graph={
            "type": "grid",
            "width": width,
            "height": height,
            "positions": positions,
        },
        params={
            "gamma": gamma,
            "dt": dt,
            "steps": steps,
            "start_index": start_index,
            "backend": "qiskit",
        },
        times=times.tolist(),
        probabilities=probs_qiskit.tolist(),
    )
    save_run(run_dir / "run.json", payload)

    plot_grid_heatmap(
        probs_qiskit[-1],
        width,
        height,
        "Qiskit demo heatmap",
        run_dir / "heatmap.png",
    )

    circuit = build_qiskit_circuit(adjacency, gamma, times[-1], start_state)
    artifacts = save_qiskit_artifacts(circuit, run_dir)

    states_numpy = continuous_time_walk(adjacency, gamma, times, start_state)
    probs_numpy = probabilities(states_numpy)
    max_error = float(np.max(np.abs(probs_numpy - probs_qiskit)))
    compare_path = run_dir / "qiskit_compare.json"
    compare_path.write_text(json.dumps({"max_error": max_error}, indent=2), encoding="utf-8")

    print(f"Saved demo run to {run_dir}")
    if artifacts:
        for key, path in artifacts.items():
            print(f"  {key}: {path}")
    print(f"  compare: {compare_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
