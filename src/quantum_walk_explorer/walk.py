"""Continuous-time quantum walk simulation utilities."""

from __future__ import annotations

import numpy as np


def basis_state(size: int, index: int) -> np.ndarray:
    if size <= 0:
        raise ValueError("size must be positive")
    if not (0 <= index < size):
        raise ValueError("index out of bounds")

    state = np.zeros(size, dtype=complex)
    state[index] = 1.0 + 0.0j
    return state


def _normalize(state: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(state)
    if norm == 0:
        raise ValueError("state norm is zero")
    return state / norm


def continuous_time_walk(
    adjacency: np.ndarray,
    gamma: float,
    times: np.ndarray,
    start_state: np.ndarray,
) -> np.ndarray:
    adjacency = np.array(adjacency, dtype=float)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("adjacency must be a square matrix")
    if not np.allclose(adjacency, adjacency.T):
        raise ValueError("adjacency must be symmetric")

    n = adjacency.shape[0]
    start_state = np.array(start_state, dtype=complex).reshape(n)
    start_state = _normalize(start_state)

    evals, evecs = np.linalg.eigh(adjacency)
    coeffs = evecs.conj().T @ start_state

    states = np.empty((len(times), n), dtype=complex)
    for idx, t in enumerate(times):
        phases = np.exp(-1j * gamma * evals * t)
        states[idx] = evecs @ (phases * coeffs)

    return states


def time_dependent_walk(
    adjacencies: list[np.ndarray],
    gamma: float,
    dt: float,
    start_state: np.ndarray,
) -> np.ndarray:
    if not adjacencies:
        raise ValueError("adjacencies must be non-empty")
    size = adjacencies[0].shape[0]
    for adj in adjacencies:
        if adj.shape != (size, size):
            raise ValueError("all adjacency matrices must share the same shape")
        if not np.allclose(adj, adj.T):
            raise ValueError("adjacency must be symmetric")

    state = _normalize(np.array(start_state, dtype=complex).reshape(size))
    states = np.empty((len(adjacencies), size), dtype=complex)

    for idx, adj in enumerate(adjacencies):
        evals, evecs = np.linalg.eigh(adj)
        coeffs = evecs.conj().T @ state
        phases = np.exp(-1j * gamma * evals * dt)
        state = evecs @ (phases * coeffs)
        states[idx] = state

    return states


def probabilities(states: np.ndarray) -> np.ndarray:
    return np.abs(states) ** 2


def top_k(probabilities_t: np.ndarray, k: int = 5) -> list[tuple[int, float]]:
    if k <= 0:
        raise ValueError("k must be positive")

    flat = probabilities_t.ravel()
    k = min(k, flat.size)
    indices = np.argpartition(-flat, k - 1)[:k]
    ordered = indices[np.argsort(-flat[indices])]
    return [(int(i), float(flat[i])) for i in ordered]
