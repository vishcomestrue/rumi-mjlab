# Rumi Velocity Task

A reinforcement learning task for training a quadruped robot (Rumi) to track velocity commands on flat and rough terrain using MuJoCo simulation with PPO.

---

## Table of Contents

1. [Task Overview](#task-overview)
2. [File Structure](#file-structure)
3. [How to Run](#how-to-run)
4. [Actor Network](#actor-network)
5. [Critic Network](#critic-network)
6. [Observations](#observations)
7. [Actions](#actions)
8. [Rewards](#rewards)
9. [Curriculum](#curriculum)
10. [Domain Randomization](#domain-randomization)
11. [Termination](#termination)
12. [Velocity Commands](#velocity-commands)
13. [Sim2Real Considerations](#sim2real-considerations)
14. [PPO Hyperparameters](#ppo-hyperparameters)
15. [Robot Physical Parameters](#robot-physical-parameters)

---

## Task Overview

The robot starts at a standing height (body z ≈ 0.2 m) and must learn to track twist commands (linear x/y velocity, yaw rate) on flat or rough terrain. The task runs for 20 seconds per episode at 50 Hz control.

**Task IDs (registry):**
- `Mjlab-Velocity-Rough-Rumi` — procedurally generated rough terrain with terrain curriculum
- `Mjlab-Velocity-Flat-Rumi` — flat plane, no terrain curriculum

| Parameter | Value |
|-----------|-------|
| Actor observation size | 48 |
| Critic observation size | 75 |
| Action size | 12 |
| Control frequency | 50 Hz |
| Simulation timestep | 5 ms (200 Hz) |
| Decimation | 4 (200 Hz sim / 50 Hz control) |
| Episode length | 20 seconds (~1000 steps) |
| Initial body height | 0.2 m |

---

## File Structure

```
rumi_velocity/
├── __init__.py            # Task registration (Mjlab-Velocity-Rough-Rumi, Mjlab-Velocity-Flat-Rumi)
├── env_cfgs.py            # Full environment config (observations, rewards, curriculum, DR)
├── rl_cfg.py              # PPO hyperparameter configuration
└── rumi/
    ├── rumi_constants.py  # Actuator params, collision config, initial state, action scale
    └── xmls/
        ├── rumi.xml       # MuJoCo model (body, 4 legs, sensors, collisions)
        ├── scene.xml      # Scene with rumi.xml + terrain
        └── assets/        # Mesh files (.obj, .stl) for body, legs, couplers
```

---

## How to Run

**Training (rough terrain):**
```bash
mjlab train "Mjlab-Velocity-Rough-Rumi"
```

**Training (flat terrain):**
```bash
mjlab train "Mjlab-Velocity-Flat-Rumi"
```

**Playback (deterministic, push disabled, randomized terrain):**
```bash
mjlab play "Mjlab-Velocity-Rough-Rumi"
mjlab play "Mjlab-Velocity-Flat-Rumi"
```

In play mode for the flat variant, linear x velocity commands span `(-1.0, 1.0)` m/s and angular z spans `(-1.0, 1.0)` rad/s.

---

## Actor Network

| Parameter | Value |
|-----------|-------|
| Hidden layers | (512, 256, 128) |
| Activation | ELU |
| Observation normalization | Disabled |
| Stochastic policy | Yes |
| Initial log std | 1.0 |
| Noise std type | Log |

---

## Critic Network

| Parameter | Value |
|-----------|-------|
| Hidden layers | (512, 256, 128) |
| Activation | ELU |
| Observation normalization | Disabled |
| Stochastic | No (deterministic value function) |

---

## Observations

The actor and critic receive different observation vectors. The **actor** uses an asymmetric, hardware-compatible set of observations (no ground-truth velocities), while the **critic** gets additional privileged information including true base velocity and foot state.

### Actor Observations (48 dims, noisy during training)

| # | Term | Dims | Description | Noise (train only) |
|---|------|------|-------------|--------------------|
| 1 | `imu_lin_acc` | 3 | Linear acceleration from IMU (replaces base linear velocity) | Uniform [-0.5, 0.5] |
| 2 | `base_ang_vel` | 3 | Angular velocity in base frame (IMU gyroscope) | Uniform [-0.2, 0.2] |
| 3 | `projected_gravity` | 3 | Gravity vector in base frame from IMU quaternion | Uniform [-0.05, 0.05] |
| 4 | `joint_pos` | 12 | Joint positions relative to default pose (4 legs × 3 DOF) | Uniform [-0.01, 0.01] rad |
| 5 | `joint_vel` | 12 | Joint velocities | Uniform [-1.5, 1.5] rad/s |
| 6 | `actions` | 12 | Previous action (temporal coherence) | None |
| 7 | `command` | 3 | Current velocity command (lin_x, lin_y, ang_z) | None |

**Notes:**
- `base_lin_vel` is excluded from the actor (not available on hardware without external tracking).
- `height_scan` (terrain raycast) is excluded (Rumi has no onboard depth sensor).
- `imu_lin_acc` provides an indirect velocity signal via integration, which is sim2real compatible.
- During play/deployment, corruption is disabled (`enable_corruption=False`).

### Critic Observations (75 dims, always clean)

The critic receives all actor terms (without noise) plus privileged information unavailable on hardware:

| # | Term | Dims | Description | Source |
|---|------|------|-------------|--------|
| 1 | `base_lin_vel` | 3 | Ground-truth linear velocity in base frame | Simulator (privileged) |
| 2 | `imu_lin_acc` | 3 | Linear acceleration (same as actor) | IMU sensor |
| 3 | `base_ang_vel` | 3 | Angular velocity | IMU sensor |
| 4 | `projected_gravity` | 3 | Gravity vector in base frame | IMU sensor |
| 5 | `joint_pos` | 12 | Joint positions relative to default | Encoders |
| 6 | `joint_vel` | 12 | Joint velocities | Simulator |
| 7 | `actions` | 12 | Previous action | — |
| 8 | `command` | 3 | Velocity command | Command manager |
| 9 | `foot_height` | 4 | Height of each foot site (FL, FR, BL, BR) | Foot site positions |
| 10 | `foot_air_time` | 4 | Time each foot has been in the air | Contact sensor |
| 11 | `foot_contact` | 4 | Binary ground contact state per foot | Contact sensor |
| 12 | `foot_contact_forces` | 12 | 3D contact force per foot (log-scaled) | Contact sensor |

---

## Actions

**Type:** Joint position control (`PositionAction`) with `use_default_offset=True`

Actions are interpreted as **delta offsets from the default joint pose**. The network outputs 12 values scaled per-joint before being added to the default pose.

| Parameter | Value |
|-----------|-------|
| Action dims | 12 (one per joint) |
| Action type | Target joint position |
| Action scale (all joints) | 0.075 rad |
| Scale formula | `0.25 × effort_limit / stiffness = 0.25 × 6.0 / 20.0` |

**Joint order:**
```
FL_hip, FL_thigh, FL_calf
FR_hip, FR_thigh, FR_calf
BL_hip, BL_thigh, BL_calf
BR_hip, BR_thigh, BR_calf
```

**Default / Standing Joint Positions:**

All joints default to 0.0 rad. This is the reference for action offsets.

---

## Rewards

All reward terms are summed with their respective weights each step.

### Velocity Tracking Rewards

| Term | Weight | Formula | Notes |
|------|--------|---------|-------|
| `track_linear_velocity` | 2.0 | `exp(-xy_error / 0.25)` | Tracks commanded lin_x and lin_y velocities; std = sqrt(0.25) |
| `track_angular_velocity` | 2.0 | `exp(-z_error / 0.5)` | Tracks commanded yaw rate; std = sqrt(0.5) |

### Posture Rewards

| Term | Weight | Formula | Notes |
|------|--------|---------|-------|
| `upright` | 1.0 | `exp(-xy_gravity² / 0.2)` | Encourages level base orientation |
| `pose` | 1.0 | `exp(-mean(error² / std²))` | Speed-dependent joint posture tracking |

**`pose` per-joint standard deviations (Rumi-specific):**

| Speed regime | Hip std (rad) | Thigh std (rad) | Calf std (rad) |
|-------------|---------------|-----------------|----------------|
| Standing (< 0.05 m/s) | 0.05 | 0.05 | 0.1 |
| Walking (0.05–1.5 m/s) | 0.3 | 0.3 | 0.6 |
| Running (≥ 1.5 m/s) | 0.3 | 0.3 | 0.6 |

### Penalty Terms

| Term | Weight | Formula | Notes |
|------|--------|---------|-------|
| `dof_pos_limits` | -1.0 | Soft joint limit violation penalty | Activates at 90% of hard limits |
| `action_rate_l2` | -0.1 | `-sum((a_t - a_{t-1})²)` | Penalizes rapid action changes |
| `foot_slip` | -0.1 | `-sum(vel_xy_norm²)` for feet in contact | Penalizes foot sliding during ground contact |
| `soft_landing` | -1e-5 | Penalizes high impact forces at touchdown | Encourages gentle foot placement |

### Disabled Rewards (available in framework)

| Term | Default weight | Notes |
|------|---------------|-------|
| `foot_clearance` | 0.0 | Target 0.07 m configured but disabled |
| `foot_swing_height` | 0.0 | Target 0.07 m configured but disabled |
| `body_ang_vel` | 0.0 | Base angular velocity penalty |
| `angular_momentum` | 0.0 | Angular momentum penalty |
| `air_time` | 0.0 | Foot air-time reward |

---

## Curriculum

### Terrain Curriculum (`terrain_levels`) — Rough terrain only

Tracks how far the robot walks each episode relative to the terrain tile width:
- If distance walked > `terrain_width / 2`: progress to harder terrain level
- If distance walked < `required_distance × 0.5`: regress to easier terrain level

Disabled in flat terrain variant and in play mode.

### Velocity Command Curriculum (`command_vel`) — Rough terrain only

Progressively widens the commanded velocity range over training:

| Stage | Training Step Threshold | Lin vel x (m/s) | Ang vel z (rad/s) |
|-------|------------------------|-----------------|-------------------|
| 0 | 0 (start) | (-1.0, 1.0) | (-0.5, 0.5) |
| 1 | 120,000 steps (5000 × 24) | (-1.5, 2.0) | (-0.7, 0.7) |
| 2 | 240,000 steps (10000 × 24) | (-2.0, 3.0) | (-0.7, 0.7) |

Disabled in flat terrain variant and in play mode.

---

## Domain Randomization

### Startup (applied once per robot instantiation)

| Event | Parameter | Range | Notes |
|-------|-----------|-------|-------|
| `foot_friction` | Foot geom friction | [0.3, 1.2] | Absolute value; all 4 feet share same value |
| `encoder_bias` | Joint position bias | [-0.015, 0.015] rad | Constant offset per joint per episode |
| `base_com` | Body COM offset | x,y: ±0.025 m; z: ±0.03 m | Additive shift to torso COM |

### Interval (periodic perturbations during episode)

| Event | Interval | Parameter | Range |
|-------|----------|-----------|-------|
| `push_robot` | 1.0–3.0 s | Linear velocity | x,y: ±0.5 m/s; z: ±0.4 m/s |
| `push_robot` | 1.0–3.0 s | Angular velocity | roll,pitch: ±0.52 rad/s; yaw: ±0.78 rad/s |

Push robot is **disabled in play mode**.

### Reset (applied each episode reset)

| Event | Parameter | Range | Notes |
|-------|-----------|-------|-------|
| `reset_base` | Base position | x,y: ±0.5 m; z: +0.01–0.05 m; yaw: ±π | Random starting pose |
| `reset_robot_joints` | Joint pos / vel | 0.0 / 0.0 | No perturbation (starts at default) |

### Currently Disabled (Available in Framework)

| Parameter | Description |
|-----------|-------------|
| `randomize_pd_gains` | Randomize stiffness (kp) and damping (kd) per episode |
| `randomize_effort_limits` | Randomize actuator torque limits |
| Ground terrain flatness | Rough terrain uses procedural generator; no real-world irregular shapes |
| Payload mass | No body mass perturbation |

---

## Termination

| Condition | Value | Notes |
|-----------|-------|-------|
| Timeout | 20 seconds (~1000 steps) | Normal episode end |
| `fell_over` (bad orientation) | Roll/pitch > 70° from upright | Terminates when gravity z-projection is too large |
| `illegal_contact` | Non-foot body touches terrain | Body, thigh, or calf collision with ground |

---

## Velocity Commands

Commands are sampled from a `UniformVelocityCommandCfg` and resampled periodically during each episode.

| Parameter | Value |
|-----------|-------|
| Resampling interval | 3.0 – 8.0 seconds (uniform random) |
| Standing environments | 10% of envs receive zero velocity command |
| Heading control | Enabled (30% of envs get heading targets) |
| Heading stiffness | 0.5 |
| Lin vel x range (stage 0) | (-1.0, 1.0) m/s |
| Lin vel y range | (-1.0, 1.0) m/s |
| Ang vel z range (stage 0) | (-0.5, 0.5) rad/s |
| Heading range | (-π, π) rad |
| Visualization z-offset | 0.5 m |

---

## Sim2Real Considerations

### Actuator Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Stiffness (kp) | 20.0 N·m/rad | All joints uniform |
| Damping (kd) | 0.0 N·m·s/rad | PD controller (set in software) |
| Joint damping (XML) | 0.59436 N·m·s/rad | Passive joint damping from model |
| Friction loss (XML) | 0.001 N·m | Joint static friction |
| Effort limit | ±6.0 N·m | All joints |
| Armature (joint inertia) | 0.01623 kg·m² | All joints |

### Control Loop

| Parameter | Value |
|-----------|-------|
| Policy inference frequency | 50 Hz |
| Simulator integration frequency | 200 Hz |
| Decimation factor | 4 |
| Action applied duration | 20 ms |

### Observation Sim2Real Notes

- `imu_lin_acc` replaces `base_lin_vel` (which requires external tracking). The IMU accelerometer is available on hardware.
- `base_ang_vel` is read from the IMU gyroscope — available on hardware.
- `projected_gravity` is derived from the IMU quaternion — available on hardware.
- `joint_vel` is exact in sim (`data.qvel`) but must be finite-differenced from encoders on hardware, introducing quantization noise and one-step lag.
- Critic's `base_lin_vel` is privileged (ground-truth) — not used on hardware.

### Gaps in Domain Randomization (Sim2Real Risk Areas)

1. **Joint velocity estimation** — Sim uses exact `qvel`; hardware uses finite-differenced encoder readings.
2. **Actuator dynamics** — No actuator delay, backlash, or velocity-dependent torque saturation modeled.
3. **Mass/inertia** — No body mass perturbation beyond COM shift.
4. **PD gain mismatch** — Hardware PD gains must match exactly (`kp=20, kd=0`).
5. **Contact model** — Soft contact model (solimp) in sim may differ from real contact dynamics.
6. **Terrain flatness** — Flat variant trained only on a plane; rough variant uses procedural tiles that may not match real surfaces.

---

## PPO Hyperparameters

| Parameter | Value |
|-----------|-------|
| Learning rate | 1.0e-3 |
| Discount factor (γ) | 0.99 |
| GAE lambda (λ) | 0.95 |
| Clip parameter (ε) | 0.2 |
| Value loss coefficient | 1.0 |
| Entropy coefficient | 0.01 |
| Clipped value loss | Yes |
| Max gradient norm | 1.0 |
| Learning epochs per rollout | 5 |
| Mini-batches per epoch | 4 |
| Steps per environment per rollout | 24 |
| Desired KL divergence | 0.01 |
| LR schedule | Adaptive |
| Max training iterations | 10,000 |
| Checkpoint save interval | Every 50 iterations |

---

## Robot Physical Parameters

### Actuator Config

| Parameter | Value |
|-----------|-------|
| Stiffness (kp) | 20.0 N·m/rad |
| Damping (kd) | 0.0 N·m·s/rad |
| Effort limit | ±6.0 N·m |
| Armature | 0.01623 kg·m² |
| Joint damping (passive) | 0.59436 N·m·s/rad |
| Friction loss | 0.001 N·m |

### Contact / Collision Config

| Parameter | Value |
|-----------|-------|
| Foot shape | Sphere (geom: `*_foot_collision`) |
| Foot friction coefficient | Randomized [0.3, 1.2] (startup DR) |
| Contact softness (solimp, feet) | [0.9, 0.95, 0.023] |
| Contact dimensionality (foot) | 3D (condim = 3) |
| Contact dimensionality (body) | 1D (condim = 1) |
| Soft joint limit factor | 0.9 |

### Initial State

| Parameter | Value |
|-----------|-------|
| Body position | (0, 0, 0.2) m |
| Body orientation | Random yaw ∈ (-π, π), upright |
| Joint positions | 0.0 rad (all joints) |
| Joint velocities | 0.0 rad/s |
