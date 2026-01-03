"""Qiskit-backed simulation utilities."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Optional

import numpy as np


def qiskit_available() -> bool:
    try:
        import qiskit  # noqa: F401
    except ImportError:
        return False
    return True


def _normalize(state: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(state)
    if norm == 0:
        raise ValueError("state norm is zero")
    return state / norm


def _pad_to_power_of_two(
    adjacency: np.ndarray,
    start_state: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    n = adjacency.shape[0]
    dim = 1 << (n - 1).bit_length()
    if dim == n:
        return adjacency, start_state, n

    adj_pad = np.zeros((dim, dim), dtype=float)
    adj_pad[:n, :n] = adjacency

    state_pad = np.zeros(dim, dtype=complex)
    state_pad[:n] = start_state

    return adj_pad, state_pad, n


def continuous_time_walk_qiskit(
    adjacency: np.ndarray,
    gamma: float,
    times: Iterable[float],
    start_state: np.ndarray,
) -> np.ndarray:
    from qiskit.quantum_info import Operator, Statevector

    adjacency = np.array(adjacency, dtype=float)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("adjacency must be a square matrix")
    if not np.allclose(adjacency, adjacency.T):
        raise ValueError("adjacency must be symmetric")

    start_state = np.array(start_state, dtype=complex)
    start_state = _normalize(start_state)

    adj_pad, state_pad, trim = _pad_to_power_of_two(adjacency, start_state)

    evals, evecs = np.linalg.eigh(adj_pad)
    evecs_dag = evecs.conj().T

    start_sv = Statevector(state_pad)
    times = list(times)
    states = np.empty((len(times), trim), dtype=complex)

    for idx, t in enumerate(times):
        phases = np.exp(-1j * gamma * evals * t)
        unitary = (evecs * phases) @ evecs_dag
        evolved = start_sv.evolve(Operator(unitary))
        states[idx] = np.array(evolved.data[:trim])

    return states


def build_qiskit_circuit(
    adjacency: np.ndarray,
    gamma: float,
    time: float,
    start_state: np.ndarray,
):
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import UnitaryGate
    from qiskit.quantum_info import Operator

    adjacency = np.array(adjacency, dtype=float)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("adjacency must be a square matrix")
    if not np.allclose(adjacency, adjacency.T):
        raise ValueError("adjacency must be symmetric")

    start_state = np.array(start_state, dtype=complex)
    start_state = _normalize(start_state)

    adj_pad, state_pad, _ = _pad_to_power_of_two(adjacency, start_state)
    num_qubits = int(math.log2(adj_pad.shape[0]))

    evals, evecs = np.linalg.eigh(adj_pad)
    evecs_dag = evecs.conj().T
    phases = np.exp(-1j * gamma * evals * time)
    unitary = (evecs * phases) @ evecs_dag

    qc = QuantumCircuit(num_qubits)
    qc.initialize(state_pad, range(num_qubits))
    qc.append(UnitaryGate(Operator(unitary)), range(num_qubits))
    return qc


def _dump_qasm(circuit) -> Optional[str]:
    try:
        return circuit.qasm()
    except Exception:
        pass
    try:
        from qiskit.qasm3 import dumps

        return dumps(circuit)
    except Exception:
        return None


def save_qiskit_artifacts(circuit, output_dir: Path) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {}

    qasm = _dump_qasm(circuit)
    if qasm:
        qasm_path = output_dir / "qiskit_circuit.qasm"
        qasm_path.write_text(qasm, encoding="utf-8")
        artifacts["qasm"] = str(qasm_path)

    try:
        fig = circuit.draw(output="mpl")
        png_path = output_dir / "qiskit_circuit.png"
        fig.savefig(png_path, dpi=160)
        artifacts["png"] = str(png_path)
    except Exception:
        pass

    return artifacts
