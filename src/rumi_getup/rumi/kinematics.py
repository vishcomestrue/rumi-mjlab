"""Forward kinematics for Rumi quadruped robot.

Geometry constants are extracted directly from rumi.xml.
All computations are batched over num_envs (N).
"""

from __future__ import annotations

import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# Robot geometry constants (metres, from rumi.xml)
# ---------------------------------------------------------------------------

# Hip body attachment positions in root body frame
_FL_HIP_POS: tuple[float, float, float] = (0.0, 0.0000, 0.0)
_FR_HIP_POS: tuple[float, float, float] = (0.0, -0.1096, 0.0)
_BL_HIP_POS: tuple[float, float, float] = (-0.2806, 0.0000, 0.0)
_BR_HIP_POS: tuple[float, float, float] = (-0.2806, -0.1096, 0.0)

# Fixed link offset vectors in each joint's local frame (same for all legs)
_HIP_TO_THIGH: tuple[float, float, float] = (0.0, 0.0595, 0.0)
_THIGH_TO_CALF: tuple[float, float, float] = (-0.20214, 0.0, -0.038)
_CALF_TO_FOOT: tuple[float, float, float] = (0.20216, 0.0, -0.04446)

FOOT_RADIUS: float = 0.02  # foot sphere radius (metres)

# ---------------------------------------------------------------------------
# Rotation helpers (batched over N environments)
# ---------------------------------------------------------------------------


def _rx(q: Tensor) -> Tensor:
  """Rotation matrices around the X axis.

  Args:
    q: Angles in radians, shape (N,).

  Returns:
    Rotation matrices, shape (N, 3, 3).
  """
  c, s = torch.cos(q), torch.sin(q)
  R = q.new_zeros(q.shape[0], 3, 3)
  R[:, 0, 0] = 1.0
  R[:, 1, 1] = c
  R[:, 1, 2] = -s
  R[:, 2, 1] = s
  R[:, 2, 2] = c
  return R


def _ry(q: Tensor) -> Tensor:
  """Rotation matrices around the Y axis.

  Args:
    q: Angles in radians, shape (N,).

  Returns:
    Rotation matrices, shape (N, 3, 3).
  """
  c, s = torch.cos(q), torch.sin(q)
  R = q.new_zeros(q.shape[0], 3, 3)
  R[:, 0, 0] = c
  R[:, 0, 2] = s
  R[:, 1, 1] = 1.0
  R[:, 2, 0] = -s
  R[:, 2, 2] = c
  return R


def _rx_pi(device: torch.device, dtype: torch.dtype) -> Tensor:
  """Rx(π) = diag(1, -1, -1), shape (1, 3, 3) for batch broadcasting."""
  return torch.tensor(
    [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
    device=device,
    dtype=dtype,
  ).unsqueeze(0)


def _quat_to_rot(q: Tensor) -> Tensor:
  """Quaternion (w, x, y, z) to rotation matrix.

  Args:
    q: Unit quaternions, shape (N, 4), convention (w, x, y, z).

  Returns:
    Rotation matrices, shape (N, 3, 3).
  """
  w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
  R = q.new_zeros(q.shape[0], 3, 3)
  R[:, 0, 0] = 1 - 2 * (y * y + z * z)
  R[:, 0, 1] = 2 * (x * y - w * z)
  R[:, 0, 2] = 2 * (x * z + w * y)
  R[:, 1, 0] = 2 * (x * y + w * z)
  R[:, 1, 1] = 1 - 2 * (x * x + z * z)
  R[:, 1, 2] = 2 * (y * z - w * x)
  R[:, 2, 0] = 2 * (x * z - w * y)
  R[:, 2, 1] = 2 * (y * z + w * x)
  R[:, 2, 2] = 1 - 2 * (x * x + y * y)
  return R


# ---------------------------------------------------------------------------
# Per-leg FK helper
# ---------------------------------------------------------------------------


def _fk_leg(
  q_hip: Tensor,
  q_thigh: Tensor,
  q_calf: Tensor,
  hip_pos: tuple[float, float, float],
  is_right: bool,
) -> Tensor:
  """Foot position in root body frame for one leg.

  Traces the full FK chain:
    body → (hip body quat) → hip joint → thigh offset
         → (thigh body quat) → thigh joint → calf offset
         → calf joint → foot site

  Left legs (FL, BL): no fixed body quats; hip axis +X, thigh +Y, calf -Y.
  Right legs (FR, BR): Rx(π) on hip and thigh bodies (XML quat="0 1 0 0");
    hip axis -X, thigh axis -Y, calf axis +Y.

  Args:
    q_hip: Hip joint angles, shape (N,).
    q_thigh: Thigh joint angles, shape (N,).
    q_calf: Calf joint angles, shape (N,).
    hip_pos: Hip attachment point in root body frame (x, y, z).
    is_right: True for FR / BR legs.

  Returns:
    Foot positions in root body frame, shape (N, 3).
  """
  dev, dtype = q_hip.device, q_hip.dtype

  v_h2t = q_hip.new_tensor(_HIP_TO_THIGH)   # (3,)
  v_t2c = q_hip.new_tensor(_THIGH_TO_CALF)  # (3,)
  v_c2f = q_hip.new_tensor(_CALF_TO_FOOT)   # (3,)

  # Translation: start at hip attachment, broadcast to (N, 3)
  t = q_hip.new_tensor(hip_pos).unsqueeze(0).expand(q_hip.shape[0], 3)

  # --- Hip body quat + hip joint ---
  # Right legs: Rx(π) @ Rx(-q_h). Left legs: Rx(+q_h).
  if is_right:
    R = _rx_pi(dev, dtype) @ _rx(-q_hip)  # (N, 3, 3)
  else:
    R = _rx(q_hip)                         # (N, 3, 3)

  # Translate to thigh joint origin
  t = t + (R @ v_h2t.unsqueeze(-1)).squeeze(-1)  # (N, 3)

  # --- Thigh body quat + thigh joint ---
  # Right legs: apply another Rx(π), then Ry(-q_t). Left legs: Ry(+q_t).
  if is_right:
    R = R @ _rx_pi(dev, dtype) @ _ry(-q_thigh)
  else:
    R = R @ _ry(q_thigh)

  # Translate to calf joint origin
  t = t + (R @ v_t2c.unsqueeze(-1)).squeeze(-1)

  # --- Calf joint (no body quat for any leg) ---
  # Right legs: axis +Y → Ry(+q_c). Left legs: axis -Y → Ry(-q_c).
  R = R @ (_ry(q_calf) if is_right else _ry(-q_calf))

  # Foot site position
  return t + (R @ v_c2f.unsqueeze(-1)).squeeze(-1)  # (N, 3)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def foot_positions_in_body_frame(joint_pos: Tensor) -> Tensor:
  """All four foot positions in the root body frame via FK.

  Uses only joint angles and hard-coded Rumi geometry — no ground-truth
  simulation state (e.g. qpos[2]) is accessed, so this runs identically
  on real hardware.

  Joint order follows the MuJoCo XML definition order:
    indices 0-2:   FL hip / thigh / calf
    indices 3-5:   FR hip / thigh / calf
    indices 6-8:   BL hip / thigh / calf
    indices 9-11:  BR hip / thigh / calf

  Args:
    joint_pos: Joint angles in radians, shape (N, 12).

  Returns:
    Foot positions in root body frame, shape (N, 4, 3).
    Leg order: [FL, FR, BL, BR].
  """
  fl = _fk_leg(joint_pos[:, 0], joint_pos[:, 1], joint_pos[:, 2], _FL_HIP_POS, False)
  fr = _fk_leg(joint_pos[:, 3], joint_pos[:, 4], joint_pos[:, 5], _FR_HIP_POS, True)
  bl = _fk_leg(joint_pos[:, 6], joint_pos[:, 7], joint_pos[:, 8], _BL_HIP_POS, False)
  br = _fk_leg(joint_pos[:, 9], joint_pos[:, 10], joint_pos[:, 11], _BR_HIP_POS, True)
  return torch.stack([fl, fr, bl, br], dim=1)  # (N, 4, 3)


def estimate_body_height(joint_pos: Tensor, body_quat: Tensor) -> Tensor:
  """Estimate body height above ground via FK + IMU orientation.

  Assumes the lowest foot is at ground level (z_world = foot_radius).
  This mirrors the exact calculation used on real hardware: motor encoders
  supply joint_pos, the IMU supplies body_quat.

  Args:
    joint_pos: Joint angles in radians, shape (N, 12).
    body_quat: Body orientation quaternion (w, x, y, z) from IMU,
      shape (N, 4).

  Returns:
    Estimated root body height above ground in metres, shape (N,).
  """
  feet_body = foot_positions_in_body_frame(joint_pos)  # (N, 4, 3)

  # Rotate foot positions from body frame → world frame
  R_body = _quat_to_rot(body_quat)                      # (N, 3, 3)
  feet_world = (
    R_body.unsqueeze(1) @ feet_body.unsqueeze(-1)       # (N, 4, 3, 1)
  ).squeeze(-1)                                          # (N, 4, 3)

  # Height = foot_radius offset minus the lowest foot's world-z
  min_foot_z = feet_world[:, :, 2].min(dim=1).values    # (N,)
  return FOOT_RADIUS - min_foot_z
