# Quantum Walk Explorer

Quantum Walk Explorer simulates continuous-time quantum walks on small graphs and logs the probability landscape so you can visualize interference patterns.

## Features

- Continuous-time quantum walk simulator (NumPy, optional Qiskit backend)
- Graphs: grid, line, ASCII maze, polar maze, 3D maze, hypercube, cube puzzle
- Maze solvers (BFS) with path overlays
- Image-to-maze import
- Visuals: heatmaps, 3D views, GIFs
- Run logging + replay
- Web UI for uploads, generators, and solution archive

## Quickstart

Requires Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .

# Run a grid walk and save a heatmap
python -m quantum_walk_explorer run --graph grid --width 10 --height 10 --steps 30 --dt 0.35

# Run a line walk
python -m quantum_walk_explorer run --graph line --nodes 20 --steps 40 --dt 0.25
```

Runs are saved under `runs/` with a `run.json` plus any generated images.

You can also use the `qwe` console script if you prefer:

```bash
qwe run --graph grid --gif
```

## Maze inputs

### ASCII maze

Use `#` = wall, `.` or space = open, `S` = start, `G` = goal.

Example maze:

```
##########
#S..#....#
#.#.#.##.#
#.#...#..#
#..##...G#
##########
```

Run the walk and solve it (shortest path via BFS):

```bash
qwe run --graph maze --maze mazes/simple.txt --solve --gif
```

This writes `solution.json` and `solution.png` with the BFS path overlaid.

### Maze from image

You can import a black-and-white maze image (dark = wall, light = open by default):

```bash
qwe run --graph maze --maze-image path/to/maze.png --solve --gif
```

Tips:
- `--image-invert` if walls are light.
- `--image-threshold 0-255` or `--image-auto-threshold`.
- `--image-cleanup` + `--image-min-component` to remove noise.
- `--image-max-size 128` to downscale large images.
- `--start x y` / `--goal x y` to override endpoints.
- Optional: red start marker + green goal marker.

### Generated mazes

Random grid maze:

```bash
qwe maze --width 12 --height 8 --seed 7 --output mazes/random.txt
qwe run --graph maze --maze mazes/random.txt --solve --gif
```

Polar maze (ring/sector coordinates):

```bash
qwe maze --style polar --rings 8 --sectors 24 --seed 7 --output mazes/polar.json
qwe run --graph polar --maze mazes/polar.json --solve --gif
```

3D maze (rotating 3D view):

```bash
qwe maze --style maze3d --width 6 --height 6 --depth 6 --seed 12 --output mazes/maze3d.json
qwe run --graph maze3d --maze mazes/maze3d.json --solve --gif
```

Hypercube maze:

```bash
qwe maze --style hypercube --dimensions 6 --output mazes/hypercube.json
qwe run --graph hypercube --maze mazes/hypercube.json --solve --gif
```

Dynamic hypercube (rooms shift over time):

```bash
qwe run --graph hypercube --maze mazes/hypercube.json --hypercube-dynamic --hypercube-shift-rate 0.3 --gif
```

Cube puzzle (3x3x3) with rotating layers:

```bash
qwe cube --size 3 --steps 80 --shift-rate 0.35 --gif
```

### GIF controls

If you want more frames than simulation steps, use `--gif-steps`:

```bash
qwe run --graph maze --maze mazes/random.txt --solve --gif --gif-steps 120
```

To scale GIF frames based on solution length, use `--gif-path-multiplier` (default 2.0):

```bash
qwe run --graph maze --maze mazes/random.txt --solve --gif --gif-path-multiplier 3
```

## Replay a run

```bash
python -m quantum_walk_explorer replay --run runs/20240101_120000/run.json
```

## Web UI (landing, upload, solutions)

Launch the web app locally:

```bash
pip install -e .
export REDIS_URL=redis://localhost:6379/0
celery -A webapp.celery_app worker --loglevel=info
python webapp.py
```

Run the Celery worker in a separate terminal so the web server stays responsive while rendering.

Visit `http://localhost:5000` for:

- Landing page with employer-focused story
- Maze generators and image upload
- Solutions archive (heatmaps, GIFs, solution overlays, run logs)
- Hypercube run form and one-page PDF exports
- 3D maze generator with rotating cell-by-cell visualization
- Polar maze generator with ring/sector controls

Run a quick smoke check before launching:

```bash
python scripts/smoke_check.py
```

If you do not want Redis locally, you can run in eager mode:

```bash
CELERY_EAGER=1 python webapp.py
```

## Deploy on Render

This repo includes `render.yaml`. Create a new Render Web Service from the repo and deploy.
The service starts with:

```bash
gunicorn webapp:app
```

By default, the web service also starts a Celery worker in the same container so you only need one paid service.

### Free Redis option (Upstash)

If you do not want to pay for Render Redis, use a free Redis provider like Upstash:

1. Create a free Redis database on Upstash.
2. Copy the Redis URL (starts with `rediss://`).
3. In Render, add an environment variable `REDIS_URL` with that URL.
4. Redeploy; the web service will start the worker and connect to Redis.

## Redis + Celery architecture

```text
Browser
  |
  v
Web (Flask + Gunicorn)
  |  enqueue job
  v
Redis (broker + result backend)
  |  dequeue job
  v
Worker (Celery)
  |
  v
Artifacts in runs/ + status polling
```

In short: the web service stays responsive, Redis queues the work, and the worker does the heavy rendering.

## Qiskit backend (optional)

If you want to show the walk evolving via Qiskit, install it separately:

```bash
pip install qiskit
```

Then run with:

```bash
qwe run --graph maze --maze mazes/random.txt --backend qiskit
```

This backend pads the state to the next power-of-two dimension, so it is best for small graphs.

### Qiskit showcase artifacts

Generate circuit diagrams + QASM alongside a run:

```bash
qwe run --graph grid --width 6 --height 6 --steps 20 --backend qiskit --qiskit-artifacts --qiskit-compare
```

Or run the dedicated demo script (writes to `runs/`):

```bash
python scripts/qiskit_demo.py
```

Web UI: enable “Use Qiskit backend” on maze generators or uploads to capture circuit diagrams and QASM in the Solutions archive.

## How it works

For a graph with adjacency matrix `A`, the walk evolves under the unitary:

```
|psi(t)> = exp(-i * gamma * A * t) |psi(0)>
```

The probability at each node is `|psi|^2`, which is logged and visualized.
