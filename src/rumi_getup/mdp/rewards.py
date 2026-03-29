"""Rumi getup task rewards."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def foot_contact_penalty(
    env: ManagerBasedRlEnv,
    sensor_name: str,
) -> torch.Tensor:
    """Penalize feet not in contact with the ground.

    Counts the number of feet off the ground and returns a negative reward.
    With 4 feet and all off the ground the penalty is -4; all grounded is 0.

    Args:
        env: The environment.
        sensor_name: Name of the foot contact sensor (one slot per foot).

    Returns:
        Penalty tensor of shape (num_envs,).
    """
    from mjlab.sensor import ContactSensor

    sensor: ContactSensor = env.scene[sensor_name]
    assert sensor.data.found is not None
    # found shape: [B, N] where N = num_feet * num_slots (num_slots=1)
    # found > 0 means contact exists
    feet_in_contact = (sensor.data.found > 0).float()
    return -torch.sum(1.0 - feet_in_contact, dim=-1)
