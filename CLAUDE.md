# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Rocket League bot training project using reinforcement learning. It trains agents in a headless physics simulation (RocketSim) via the RLGym API, with PPO as the RL algorithm (rlgym-ppo).

## Environment Setup

Dependencies are managed with `uv`. Two of the key packages are installed from git:
- `rlgym-sim` — from `https://github.com/AechPro/rocket-league-gym-sim` (main branch)
- `rlgym-ppo` — from `https://github.com/AechPro/rlgym-ppo`

```bash
uv sync          # install all dependencies
source .venv/bin/activate
```

The project also includes a `flake.nix` for a Nix dev shell (primarily provides Rust toolchain, which RocketSim needs to build).

## Running Training

```bash
python src/example.py   # start training (spawns 32 parallel sim processes)
```

Training logs to Weights & Biases (`log_to_wandb=True`). Checkpoints are saved every 100k timesteps. The timestep limit is set to 1 billion.

## Running on SLURM (HPC cluster)

```bash
sbatch batch.sh   # submit training job to SLURM scheduler
```

`batch.sh` activates the venv and runs `src/example.py` on the `dev` partition with 64 cores and 375 GB RAM.

## Architecture

### Key files

- **`src/example.py`** — main training entry point. Defines `build_rocketsim_env()` which constructs the rlgym-sim environment (reward function, obs builder, action parser, terminal conditions), and `ExampleLogger` which collects and reports metrics to W&B. The `Learner` from rlgym-ppo orchestrates multi-process rollout collection and PPO updates.

- **`src/main.py`** — minimal RocketSim scratch/test script. Creates a bare `rs.Arena` and queries boost pad state; not part of the training pipeline.

- **`src/collision_meshes/soccar/`** — `.cmf` mesh files required by RocketSim to simulate the Soccar arena geometry. These must be present for `rs.Arena` to initialize correctly.

### Data flow

1. `Learner` spawns `n_proc` (32) worker processes, each running an independent copy of the rlgym-sim environment built by `build_rocketsim_env()`.
2. Workers collect rollouts (observations, actions, rewards) using the shared policy.
3. Rollouts are buffered (`exp_buffer_size=150000`) and used to update the policy with PPO every `ts_per_iteration` (50k) timesteps.
4. `ExampleLogger._collect_metrics` is called each step to capture game state; `_report_metrics` aggregates and pushes averages to W&B.

### Environment configuration (in `build_rocketsim_env`)

| Parameter | Value |
|---|---|
| Team size | 1v1 |
| Tick skip | 8 (at 120 Hz → 15 Hz agent decisions) |
| Episode timeout | 10 seconds of no touch |
| Reward | `CombinedReward`: velocity-to-ball (0.01), velocity-ball-to-goal (0.1), event-based goals/demos (10.0) |
| Observation | `DefaultObs` with normalized position, velocity, angular velocity |
| Action space | `ContinuousAction` |
