# Rumi mjlab environment

![Rumi locomotion](teaser.gif)

Reinforcement learning tasks for the Rumi quadruped robot using the mjlab framework.

## Overview

This repository contains two RL tasks for training the Rumi quadruped:
- **Velocity tracking** - Train Rumi to walk and track commanded body velocities (flat and rough terrain)
- **Get-up** - Train Rumi to recover from a fall and get back on its feet

## Repository Structure

```
rumi_mjlab/
├── src/
│   ├── rumi_velocity/              # Velocity tracking task
│   │   ├── __init__.py             # Task registration (2 variants: flat/rough)
│   │   ├── env_cfgs.py             # Environment configs (sensors, rewards, terminations)
│   │   ├── rl_cfg.py               # PPO hyperparameters
│   │   └── rumi/
│   │       ├── rumi_constants.py   # Robot definition (actuators, collision, init state)
│   │       └── xmls/
│   │           ├── rumi.xml        # MuJoCo MJCF model
│   │           └── assets/         # Mesh files (.obj, .stl)
│   └── rumi_getup/                 # Get-up task
│       ├── __init__.py             # Task registration
│       ├── env_cfgs.py             # Environment configs
│       └── rl_cfg.py               # PPO hyperparameters
├── pyproject.toml                  # Project dependencies and configuration
└── README.md                       # This file
```

## Registered Tasks

- `Mjlab-Velocity-Flat-Rumi` - Velocity tracking on flat terrain
- `Mjlab-Velocity-Rough-Rumi` - Velocity tracking on rough terrain
- `Mjlab-Getup-Rumi` - Get-up and recovery task

## Usage

### Velocity Tracking

```sh
# Sanity check: watch Rumi stand and fall under zero actions
uv run play Mjlab-Velocity-Flat-Rumi --agent zero

# Train on flat terrain
CUDA_VISIBLE_DEVICES=0 uv run train Mjlab-Velocity-Flat-Rumi \
  --env.scene.num-envs 4096 \
  --agent.max-iterations 3_000

# Train on rough terrain
CUDA_VISIBLE_DEVICES=0 uv run train Mjlab-Velocity-Rough-Rumi \
  --env.scene.num-envs 4096 \
  --agent.max-iterations 3_000

# Play the trained checkpoint
uv run play Mjlab-Velocity-Flat-Rumi --wandb-run-path <wandb-run-path>
```

### Get-up Task

```sh
# Train Rumi to get up from falls
CUDA_VISIBLE_DEVICES=0 uv run train Mjlab-Getup-Rumi \
  --env.scene.num-envs 4096 \
  --agent.max-iterations 3_000

# Play the trained checkpoint
uv run play Mjlab-Getup-Rumi --wandb-run-path <wandb-run-path>
```

## Dependencies

- Python >=3.10, <3.14
- mjlab (local editable install from ../mjlab)
- mujoco-warp (Google DeepMind's MuJoCo-Warp integration)
