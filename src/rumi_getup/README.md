# Rumi Getup Task

A reinforcement learning task for training a quadruped robot (Rumi) to stand up from a lying position using MuJoCo simulation with PPO.

---

## Table of Contents

1. [Task Overview](#task-overview)
2. [Actor Network](#actor-network)
3. [Critic Network](#critic-network)
4. [Observations](#observations)
5. [Actions](#actions)
6. [Rewards](#rewards)
7. [Curriculum](#curriculum)
8. [Domain Randomization](#domain-randomization)
9. [Sim2Real Considerations](#sim2real-considerations)
10. [PPO Hyperparameters](#ppo-hyperparameters)
11. [Robot Physical Parameters](#robot-physical-parameters)

---

## Task Overview

The robot starts lying flat (pose near ground, height ~0.1 m) and must learn to stand up to a target height defined by a curriculum. The task runs for 20 seconds per episode at 50 Hz control.

| Parameter | Value |
|-----------|-------|
| Observation size | 41 |
| Action size | 12 |
| Control frequency | 50 Hz |
| Simulation timestep | 5 ms (200 Hz) |
| Decimation | 4 (200 Hz sim / 50 Hz control) |
| Episode length | 20 seconds (~1000 steps) |

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

The critic uses the same observation space as the actor (41 dims).

---

## Observations

**Total size: 41 dimensions**

Observations are concatenated in the following order:

| # | Term | Dims | Description | Noise (train) | Normalization range |
|---|------|------|-------------|---------------|---------------------|
| 1 | `body_height` | 1 | FK + IMU-based height estimate (not ground-truth z). Height = `FOOT_RADIUS - min_foot_z` after rotating foot positions to world frame. | None | [0.10, 0.25] m, clipped to [-0.5, 1.5] |
| 2 | `target_height` | 1 | Curriculum target height (normalized) | None | [0.10, 0.25] m |
| 3 | `projected_gravity` | 3 | Gravity vector expressed in body frame via IMU quaternion | Uniform [-0.05, 0.05] | [-1, 1] per component |
| 4 | `joint_pos` | 12 | Joint positions relative to default pose (12 joints: 4 legs × 3 DOF) | Uniform [-0.01, 0.01] rad | Joint limits |
| 5 | `joint_vel` | 12 | Joint velocities relative to default | Uniform [-1.5, 1.5] rad/s | [-2π, 2π] rad/s |
| 6 | `actions` | 12 | Previous action (for temporal coherence) | None | - |

**Notes:**
- `base_lin_vel` and `base_ang_vel` are excluded (not available reliably on hardware).
- `body_height` uses forward kinematics from encoder readings + IMU orientation — fully sim2real compatible.
- Observation noise is only applied during training, not during play/deployment.

### FK Height Estimation Leg Geometry

```
Hip to thigh offset:  (0.0, ±0.0595, 0.0) m
Thigh to calf offset: (-0.20214, 0.0, -0.038) m
Calf to foot offset:  (0.20216, 0.0, -0.04446) m
Foot radius:          0.02 m
```

---

## Actions

**Type:** Joint position control (`PositionAction`) with `use_default_offset=True`

Actions are interpreted as **delta offsets from the default joint pose**. The network outputs 12 values in `[-1, 1]` which are scaled per-joint before being added to the default pose to form target positions.

| Parameter | Value |
|-----------|-------|
| Action dims | 12 (one per joint) |
| Action type | Target joint position |
| Action scale (all joints) | 0.075 rad |
| Scale formula | `0.25 * effort_limit / stiffness = 0.25 * 6.0 / 20.0` |

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
| Calf (left legs) | -1.51 | 0.1 |
| Calf (right legs) | -0.1 | 1.51 |

Soft joint limit factor: **0.9** (penalty activates at 90% of hard limits).

---

## Rewards

All reward terms are summed with their respective weights each step.

### Primary Rewards

| Term | Weight | Formula | Notes |
|------|--------|---------|-------|
| `height` | 1.0 | `exp(-|height - target| / 0.1)` | Gaussian reward for reaching curriculum target height |
| `upright` | 3.0 | `exp(-2.0 * ||up_vec - gravity_b||²)` | Exponential reward for maintaining upright orientation |

### Regularization Rewards

| Term | Weight | Formula | Notes |
|------|--------|---------|-------|
| `joint_symmetry` | 0.5 | `exp(-mean(symmetric_pair_errors))` | Encourages left-right leg symmetry |
| `hip_stability` | 0.3 | `exp(-sum(hip_pos²))` | Discourages hip abduction/adduction |

### Penalty Terms

| Term | Weight | Formula | Notes |
|------|--------|---------|-------|
| `action_rate` | 0.01 | `-sum((a_t - a_{t-1})²)` | Penalizes rapid action changes |
| `dof_vel` | 0.01 | `-sum(joint_vel²)` | Penalizes large joint velocities (Rumi override: 10× default) |
| `torques` | 0.001 | `-(L2 + L1 norm of torques)` | Energy efficiency penalty (Rumi override: 10× default) |
| `joint_limits` | 1.0 | Negative penalty beyond soft limits | Penalizes joint limit violations |
| `illegal_contact` | 0.5 | `-count(non-foot ground contacts)` | Penalizes body/leg collisions with ground |

**`joint_symmetry` symmetric pairs:**
- Hip joints: FL ≈ FR (same sign)
- Thigh joints: FL ≈ -FR (mirrored mounting)
- Calf joints: FL ≈ -FR (mirrored mounting)

---

## Curriculum

The curriculum progressively increases the difficulty by widening and raising the target standing height range.

### Height Curriculum Stages

| Stage | Training Step Threshold | Min Height | Max Height |
|-------|------------------------|-----------|-----------|
| 0 | 0 (start) | 0.24 m | 0.25 m |
| 1 | 36,000 steps | 0.21 m | 0.28 m |
| 2 | 72,000 steps | 0.18 m | 0.31 m |

- Steps are environment steps (each step = 0.02 s of sim time).
- Stage 0 is a narrow band just above the resting height — easy to achieve accidentally.
- Stage 2 requires a full standing posture.
- Target height is sampled uniformly within the current stage's `[min, max]` range each episode reset.
- The sampled target height is used in both the `height` reward and the `target_height` observation.

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
| Ground friction | No friction randomization (fixed flat plane) |
| Payload mass | No mass/inertia perturbation |
| Gravity direction | No gravity direction perturbation |

---

## Sim2Real Considerations

### Actuator Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Stiffness (kp) | 20.0 N·m/rad | All joints uniform |
| Damping (kd) | 0.0 N·m·s/rad | PD controller (set in software) |
| Joint damping (XML) | 0.45 N·m·s/rad | Passive joint damping from model |
| Friction loss (XML) | 0.01 N·m | Joint static friction |
| Effort limit | ±6.0 N·m | All joints |
| Armature (joint inertia) | 0.02 kg·m² | All joints |

**Important:** The PD gains in simulation (`kp=20, kd=0`) must match the hardware controller PD gains exactly. The XML also includes passive joint damping of 0.45 N·m·s/rad — this is a model parameter and may need identification from hardware.

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
| Contact softness (solimp) | [0.9, 0.95, 0.022] |
| Contact dimensionality | 3D (condim = 3) |

The foot sphere radius (0.02 m) and friction (0.6) should be validated against hardware measurements on the real terrain surface.

### Height Estimation (Sim2Real Compatible)

The policy uses FK + IMU height estimation, **not** ground-truth simulator z-position. This makes the height observation directly deployable on hardware:

```
height = FOOT_RADIUS - min(z_feet_in_world)
z_feet_in_world = R_imu @ (forward_kinematics(joint_pos) + p_body)
```

**Hardware requirements:**
- 12 joint encoders (all joints)
- IMU providing quaternion orientation

### Initial State / Reset Conditions

| Parameter | Value |
|-----------|-------|
| Body position | (0, 0, 0.1) m |
| Body orientation | Identity quaternion (upright) |
| Joint positions | 0.0 rad (neutral) |
| Joint velocities | 0.0 rad/s |
| Joint perturbation on reset | ±0.1 rad position, ±0.05 rad/s velocity |

On hardware, the robot should be placed roughly flat before running the policy. The perturbation during training adds robustness to initial configuration variation.

### Gaps in Domain Randomization (Sim2Real Risk Areas)

The following are **not randomized** in the current training and may cause sim2real gap:

1. **Ground friction** — Policy trained only at friction = 0.6. Slippery or high-grip surfaces may degrade performance.
2. **Actuator dynamics** — No actuator delay, no backlash, no velocity-dependent torque limits modeled.
3. **Mass/inertia** — No body mass perturbation. Hardware mass deviations from model will affect dynamics.
4. **PD gain mismatch** — Hardware PD gains must match exactly (`kp=20, kd=0`). Small mismatches cause instability.
5. **Joint encoder offset** — No calibration bias randomization. Zero offsets assumed.
6. **Terrain flatness** — Trained only on flat ground. Uneven surfaces not covered.
7. **Contact model** — Soft contact model (solimp) in sim may differ from real contact dynamics.

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

### Leg Geometry (FK offsets)

| Segment | Offset (m) |
|---------|-----------|
| Body → Hip | (±0.11, ±0.0595, -0.007) |
| Hip → Thigh | (0.0, ±0.0595, 0.0) |
| Thigh → Calf | (-0.20214, 0.0, -0.038) |
| Calf → Foot | (0.20216, 0.0, -0.04446) |
