# Rumi Getup Task

A reinforcement learning task for training a quadruped robot (Rumi) to stand up from a lying position using MuJoCo simulation with PPO.

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
12. [Sim2Real Considerations](#sim2real-considerations)
13. [PPO Hyperparameters](#ppo-hyperparameters)
14. [Robot Physical Parameters](#robot-physical-parameters)

---

## Task Overview

The robot starts nearly flat on the ground (body z ≈ 0.1 m, all joints at 0 rad) and must learn to stand up to a target height defined by a curriculum. The task runs for 20 seconds per episode at 50 Hz control.

**Task ID (registry):** `Mjlab-Getup-Rumi`

| Parameter | Value |
|-----------|-------|
| Observation size | 41 |
| Action size | 12 |
| Control frequency | 50 Hz |
| Simulation timestep | 5 ms (200 Hz) |
| Decimation | 4 (200 Hz sim / 50 Hz control) |
| Episode length | 20 seconds (~1000 steps) |
| Target body height | 0.18 m (curriculum target) |
| Min body height | 0.16 m |

---

## File Structure

```
rumi_getup/
├── __init__.py            # Task registration (Mjlab-Getup-Rumi)
├── env_cfgs.py            # Full environment config (observations, rewards, curriculum, DR)
├── rl_cfg.py              # PPO hyperparameter configuration
├── runner.py              # Custom runner that logs Rumi params to W&B
├── mdp/
│   ├── __init__.py        # MDP module exports
│   ├── observations.py    # FK-based body height observation (sim2real compatible)
│   └── rewards.py         # foot_contact_penalty reward
└── rumi/
    ├── rumi_constants.py  # Actuator params, collision config, initial state, action scale
    ├── kinematics.py      # Forward kinematics for height estimation (13-DOF chain per leg)
    └── xmls/
        ├── rumi.xml       # MuJoCo model (body, 4 legs, sensors, collisions)
        ├── scene.xml      # Scene with rumi.xml + floor plane
        └── assets/        # Mesh files (.obj, .stl) for body, legs, couplers
```

---

## How to Run

**Training:**
```bash
mjlab train "Mjlab-Getup-Rumi"
```

**Playback (deterministic, fixed target height 0.21–0.28 m):**
```bash
mjlab play "Mjlab-Getup-Rumi"
```

The runner automatically:
- Initialises PPO and parallel environments
- Logs training metrics and Rumi physical constants to Weights & Biases (W&B)
- Saves checkpoints every 100 iterations
- Applies curriculum learning updates based on training step count

---

## Actor Network

| Parameter | Value |
|-----------|-------|
| Hidden layers | (512, 256, 128) |
| Activation | ELU |
| Observation normalization | Enabled (online EMA) |
| Stochastic policy | Yes |
| Initial log std | 1.0 |
| Noise std type | Log |

---

## Critic Network

| Parameter | Value |
|-----------|-------|
| Hidden layers | (512, 256, 128) |
| Activation | ELU |
| Observation normalization | Enabled |
| Stochastic | No (deterministic value function) |

---

## Observations

Both actor and critic receive the same 41-dimensional observation vector with identical terms. The only difference is that **the actor's observations are corrupted with noise during training** (`enable_corruption=True`), while the **critic always sees clean observations** (`enable_corruption=False`). This is a standard asymmetric actor-critic setup for robustness.

### Actor Observations (41 dims, noisy during training)

| # | Term | Dims | Description | Noise (train only) | Normalization range |
|---|------|------|-------------|---------------------|---------------------|
| 1 | `body_height` | 1 | FK + IMU-based height estimate. Height = `FOOT_RADIUS - min_foot_z` after rotating foot positions to world frame via IMU quaternion. | None | [0.10, 0.25] m, clipped to [-0.5, 1.5] |
| 2 | `target_height` | 1 | Curriculum target height (normalized) | None | [0.10, 0.25] m |
| 3 | `projected_gravity` | 3 | Gravity vector expressed in body frame via IMU quaternion | Uniform [-0.05, 0.05] | [-1, 1] per component |
| 4 | `joint_pos` | 12 | Joint positions relative to default pose (4 legs × 3 DOF) | Uniform [-0.01, 0.01] rad | Joint limits |
| 5 | `joint_vel` | 12 | Joint velocities relative to default | Uniform [-1.5, 1.5] rad/s | [-2π, 2π] rad/s |
| 6 | `actions` | 12 | Previous action (for temporal coherence) | None | - |

### Critic Observations (41 dims, always clean)

The critic uses the exact same 41 terms in the same order, but with `enable_corruption=False` — noise is **never** applied, not even during training. This gives the critic a cleaner value estimate to guide policy updates.

| # | Term | Dims | Description | Noise | Normalization range |
|---|------|------|-------------|-------|---------------------|
| 1 | `body_height` | 1 | Same FK + IMU height (identical function, no noise) | None | [0.10, 0.25] m, clipped to [-0.5, 1.5] |
| 2 | `target_height` | 1 | Curriculum target height (normalized) | None | [0.10, 0.25] m |
| 3 | `projected_gravity` | 3 | Gravity vector in body frame via IMU | None | [-1, 1] per component |
| 4 | `joint_pos` | 12 | Joint positions relative to default | None | Joint limits |
| 5 | `joint_vel` | 12 | Joint velocities relative to default | None | [-2π, 2π] rad/s |
| 6 | `actions` | 12 | Previous action | None | - |

**Notes:**
- `base_lin_vel` and `base_ang_vel` are excluded from both actor and critic (not available reliably on hardware).
- `body_height` uses forward kinematics from encoder readings + IMU orientation — fully sim2real compatible, no ground-truth z used.
- During play/deployment, actor noise is also disabled (`enable_corruption=False`).

### FK Height Estimation — Leg Geometry

```
Hip to thigh offset:  (0.0, +0.0595, 0.0) m   [same fixed vector for all legs]
Thigh to calf offset: (-0.20214, 0.0, -0.038) m
Calf to foot offset:  (0.20216, 0.0, -0.04446) m
Foot radius:          0.02 m

Hip attachment in root body frame:
  FL: ( 0.0000,  0.0000, 0.0)
  FR: ( 0.0000, -0.1096, 0.0)
  BL: (-0.2806,  0.0000, 0.0)
  BR: (-0.2806, -0.1096, 0.0)
```

Right legs (FR, BR) apply an additional Rx(π) body rotation, which mirrors the y/z axes relative to left legs.

---

## Actions

**Type:** Joint position control (`PositionAction`) with `use_default_offset=True`

Actions are interpreted as **delta offsets from the default joint pose**. The network outputs 12 values in `[-1, 1]` which are scaled per-joint before being added to the default pose to form target positions.

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

**Joint limits (hard limits):**

| Joint | Min (rad) | Max (rad) |
|-------|-----------|-----------|
| Hip | -0.5 | 0.5 |
| Thigh | -0.4 | 1.5 |
| Calf (left legs: FL, BL) | -1.51 | 0.1 |
| Calf (right legs: FR, BR) | -0.1 | 1.51 |

Soft joint limit factor: **0.9** (penalty activates at 90% of hard limits).

### Default / Standing Joint Positions

The default joint pose (used as the zero-offset reference for actions and target pose for symmetry rewards):

| Joint | Default pos (rad) |
|-------|-------------------|
| FL_hip | 0.0 |
| FL_thigh | 0.0 |
| FL_calf | -0.410 |
| FR_hip | 0.0 |
| FR_thigh | 0.0 |
| FR_calf | +0.410 |
| BL_hip | 0.0 |
| BL_thigh | 0.0 |
| BL_calf | -0.491 |
| BR_hip | 0.0 |
| BR_thigh | 0.0 |
| BR_calf | +0.491 |

---

## Rewards

All reward terms are summed with their respective weights each step.

### Primary Rewards

| Term | Weight | Formula | Notes |
|------|--------|---------|-------|
| `height` | 1.0 | `exp(-\|height - target\| / 0.1)` | Gaussian reward for reaching curriculum target height |
| `upright` | 3.0 | `exp(-2.0 × \|\|up_vec - gravity_b\|\|²)` | Exponential reward for maintaining upright orientation |

### Regularization Rewards

| Term | Weight | Formula | Notes |
|------|--------|---------|-------|
| `joint_symmetry` | 0.5 | `exp(-mean(symmetric_pair_errors))` | Encourages left-right leg symmetry |
| `hip_stability` | 0.3 | `exp(-sum(hip_pos²))` | Discourages hip abduction/adduction |

### Penalty Terms

| Term | Weight | Formula | Notes |
|------|--------|---------|-------|
| `foot_contact` | 0.5 | `-count(feet_not_in_contact)` | Penalizes feet lifted off the ground; 0 when all 4 feet grounded, -4 when all lifted |
| `action_rate` | 0.01 | `-sum((a_t - a_{t-1})²)` | Penalizes rapid action changes |
| `dof_vel` | 0.01 | `-sum(joint_vel²)` | Penalizes large joint velocities (10× base default of 0.001) |
| `torques` | 0.001 | `-(L2 + L1 norm of torques)` | Energy efficiency penalty (10× base default of 0.0001) |
| `joint_limits` | 1.0 | Negative penalty beyond soft limits | Penalizes joint limit violations |
| `illegal_contact` | 0.5 | `-count(non-foot ground contacts)` | Penalizes body/leg collisions with ground |

**`joint_symmetry` symmetric pairs:**
- Hip joints: FL ≈ FR (same sign)
- Thigh joints: FL ≈ -FR (mirrored mounting)
- Calf joints: FL ≈ -FR (mirrored mounting)

**Note on `foot_contact` vs `illegal_contact`:** The robot starts lying on the ground, so body contacts are penalized (not terminated). Once standing, all 4 feet should remain in ground contact.

---

## Curriculum

The curriculum progressively increases difficulty by widening and raising the target standing height range.

### Height Curriculum Stages

| Stage | Training Step Threshold | Min Height | Max Height |
|-------|------------------------|-----------|-----------|
| 0 | 0 (start) | 0.24 m | 0.25 m |
| 1 | 48,000 steps (2000 × 24) | 0.21 m | 0.28 m |
| 2 | 72,000 steps (3000 × 24) | 0.18 m | 0.31 m |

- Step count = training iterations × `num_steps_per_env` (24).
- Stage 0 is a narrow band just above the resting height — easy to achieve early in training.
- Stage 2 requires a full standing posture.
- Target height is sampled uniformly within `[min, max]` each episode reset.
- The sampled target height feeds both the `height` reward and the `target_height` observation.
- In **play mode**, target height is fixed to [0.21, 0.28] m for deterministic evaluation.

---

## Domain Randomization

### Currently Enabled

| Type | Parameter | Range | When Applied |
|------|-----------|-------|-------------|
| Joint perturbation | Joint position offset | [-0.1, 0.1] rad | On episode reset |
| Joint perturbation | Joint velocity offset | [-0.05, 0.05] rad/s | On episode reset |
| Observation noise | `projected_gravity` | Uniform [-0.05, 0.05] | Every step (train only) |
| Observation noise | `joint_pos` | Uniform [-0.01, 0.01] rad | Every step (train only) |
| Observation noise | `joint_vel` | Uniform [-1.5, 1.5] rad/s | Every step (train only) |

### Currently Disabled (Available in Framework)

The following randomizations exist in the base framework but are not configured for Rumi getup. These should be considered for improving sim2real transfer:

| Parameter | Description |
|-----------|-------------|
| `randomize_pd_gains` | Randomize stiffness (kp) and damping (kd) per episode |
| `randomize_effort_limits` | Randomize actuator torque limits |
| `randomize_encoder_bias` | Add constant joint encoder calibration offset |
| Ground friction | No friction randomization (fixed flat plane, μ = 0.6) |
| Payload mass | No mass/inertia perturbation |
| Gravity direction | No gravity direction perturbation |

---

## Termination

| Condition | Value | Notes |
|-----------|-------|-------|
| Timeout | 20 seconds (~1000 steps) | Normal episode end |
| Max tilt angle | 30° from vertical | Early termination if robot falls too far |

Body contacts with the ground are penalized (not terminated) because the robot starts lying flat and needs time to stand.

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

**Important:** The PD gains in simulation (`kp=20, kd=0`) must match the hardware controller PD gains exactly. The XML also includes passive joint damping of 0.59436 N·m·s/rad — this is a model parameter that may need system identification from hardware.

### Control Loop

| Parameter | Value |
|-----------|-------|
| Policy inference frequency | 50 Hz |
| Simulator integration frequency | 200 Hz |
| Decimation factor | 4 |
| Action applied duration | 20 ms |

On hardware, ensure the motor controller runs at or above 50 Hz to match policy assumptions.

### Contact Parameters

| Parameter | Value |
|-----------|-------|
| Foot shape | Sphere, radius = 0.02 m |
| Ground friction coefficient | 0.6 |
| Contact softness (solimp) | [0.9, 0.95, 0.023] |
| Contact dimensionality (foot) | 3D (condim = 3) |
| Contact dimensionality (body) | 1D (condim = 1) |

The foot sphere radius (0.02 m) and friction (0.6) should be validated against hardware measurements on the real terrain surface.

### Height Estimation (Sim2Real Compatible)

The policy uses FK + IMU height estimation, **not** ground-truth simulator z-position. This makes the height observation directly deployable on hardware:

```
height = FOOT_RADIUS - min(z_feet_in_world)
z_feet_in_world = R_imu @ (forward_kinematics(joint_pos) + p_body)
```

**Hardware requirements:**
- 12 joint encoders (all joints)
- IMU providing quaternion orientation (w, x, y, z)

### Initial State / Reset Conditions

| Parameter | Value |
|-----------|-------|
| Body position | (0, 0, 0.1) m |
| Body orientation | Identity quaternion (upright) |
| Joint positions | 0.0 rad (all joints, neutral/flat) |
| Joint velocities | 0.0 rad/s |
| Joint perturbation on reset | ±0.1 rad position, ±0.05 rad/s velocity |

On hardware, the robot should be placed roughly flat before running the policy. The perturbation during training adds robustness to initial configuration variation.

### Joint Velocity: Sim vs Hardware

In simulation, `joint_vel` reads directly from MuJoCo's `data.qvel` — the ground-truth analytical velocity produced by the physics integrator (exact, noiseless, zero-lag). On real hardware, joint velocity is not directly sensed; it is **finite-differenced from encoder positions**:

```
joint_vel_hardware ≈ (q_t - q_{t-1}) / dt
```

This introduces:
- **Quantization noise** — limited by encoder resolution
- **One timestep delay** — velocity reflects positions from the previous control step
- **Jitter at low speeds** — especially near zero velocity

The current mitigation is the `±1.5 rad/s` uniform noise on `joint_vel` in the **actor** during training. This is a wide-band additive noise approximation — it does not model the correlated, lag-shifted character of finite-difference velocity. The **critic** receives `qvel` with zero noise always.

For improved sim2real transfer, consider:
- Explicitly computing finite-differenced velocity in the observation function and using that in both sim and hardware
- Or adding randomized lag (1–2 step delay) to `joint_vel` during training

### Gaps in Domain Randomization (Sim2Real Risk Areas)

The following are **not randomized** in the current training and may cause sim2real gap:

1. **Joint velocity estimation** — Sim uses exact `qvel`; hardware uses finite-differenced encoder readings (see above).
2. **Ground friction** — Policy trained only at friction = 0.6. Slippery or high-grip surfaces may degrade performance.
3. **Actuator dynamics** — No actuator delay, no backlash, no velocity-dependent torque limits modeled.
4. **Mass/inertia** — No body mass perturbation. Hardware mass deviations from model will affect dynamics.
5. **PD gain mismatch** — Hardware PD gains must match exactly (`kp=20, kd=0`). Small mismatches cause instability.
6. **Joint encoder offset** — No calibration bias randomization. Zero offsets assumed.
7. **Terrain flatness** — Trained only on flat ground. Uneven surfaces not covered.
8. **Contact model** — Soft contact model (solimp) in sim may differ from real contact dynamics.

---

## PPO Hyperparameters

| Parameter | Value |
|-----------|-------|
| Learning rate | 3.0e-4 |
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
| Checkpoint save interval | Every 100 iterations |

---

## Robot Physical Parameters

### Body Masses

| Segment | Mass (kg) |
|---------|-----------|
| Torso | 0.800 |
| Hip (×4) | 0.010 each |
| Thigh (×4) | 0.340 each |
| Calf (×4) | 0.058 each |
| **Total** | **~2.82 kg** |

### Torso Inertia

| Parameter | Value |
|-----------|-------|
| Inertia diagonal | (0.0001, 0.002, 0.002) kg·m² |
| COM offset | (-0.140, -0.054, 0.0) m from body origin |

### Leg Geometry (FK offsets from kinematics.py)

Hip attachment positions in root body frame:

| Leg | x (m) | y (m) | z (m) |
|-----|--------|--------|--------|
| FL | 0.0000 | 0.0000 | 0.0 |
| FR | 0.0000 | -0.1096 | 0.0 |
| BL | -0.2806 | 0.0000 | 0.0 |
| BR | -0.2806 | -0.1096 | 0.0 |

Fixed link offset vectors (same for all legs; right legs apply Rx(π) mirroring):

| Segment | Offset (m) |
|---------|-----------|
| Hip → Thigh | (0.0, +0.0595, 0.0) |
| Thigh → Calf | (-0.20214, 0.0, -0.038) |
| Calf → Foot | (0.20216, 0.0, -0.04446) |
| Foot radius | 0.02 m (sphere) |
