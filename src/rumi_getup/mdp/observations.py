"""Rumi-specific observation terms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import BuiltinSensor

from rumi_getup.rumi.kinematics import estimate_body_height

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def fk_body_height(
  env: ManagerBasedRlEnv,
  imu_sensor_name: str = "robot/orientation",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  min_height: float = 0.10,
  max_height: float = 0.25,
) -> torch.Tensor:
  """Normalized body height estimated from FK + IMU orientation.

  Uses joint positions (motor encoders) and body orientation (IMU framequat
  sensor). No ground-truth height (qpos[2]) is used, so this observation is
  identical in simulation and on real hardware.

  Normalization and clipping match the base body_height observation for
  drop-in compatibility.

  Args:
    env: The environment.
    imu_sensor_name: Scene key for the IMU framequat sensor. The XML sensor
      named "orientation" is scoped to entity "robot", so the default
      "robot/orientation" matches rumi.xml directly.
    asset_cfg: Asset configuration for joint position access.
    min_height: Minimum height for normalization (metres).
    max_height: Maximum height for normalization (metres).

  Returns:
    Normalized height, shape (N, 1), clipped to [-0.5, 1.5].
  """
  asset = env.scene[asset_cfg.name]
  imu: BuiltinSensor = env.scene[imu_sensor_name]

  # (N, 12) absolute joint angles in XML definition order
  joint_pos = asset.data.joint_pos
  # (N, 4) quaternion (w, x, y, z) from IMU framequat sensor
  body_quat = imu.data

  raw_height = estimate_body_height(joint_pos, body_quat).unsqueeze(-1)  # (N, 1)
  normalized = (raw_height - min_height) / (max_height - min_height)
  return torch.clamp(normalized, -0.5, 1.5)
