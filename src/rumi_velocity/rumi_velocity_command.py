"""Rumi-specific velocity command with discretized sampling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.tasks.velocity.mdp.velocity_command import (
  UniformVelocityCommand,
  UniformVelocityCommandCfg,
)

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


class RumiVelocityCommand(UniformVelocityCommand):
  """Velocity command that rounds sampled values to 1 decimal place (0.1 m/s steps)."""

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    super()._resample_command(env_ids)
    self.vel_command_b[env_ids] = (
      torch.round(self.vel_command_b[env_ids] * 10) / 10
    )


@dataclass(kw_only=True)
class RumiVelocityCommandCfg(UniformVelocityCommandCfg):
  def build(self, env: ManagerBasedRlEnv) -> RumiVelocityCommand:
    return RumiVelocityCommand(self, env)
