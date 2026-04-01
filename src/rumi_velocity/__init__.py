from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import (
  rumi_flat_env_cfg,
  rumi_rough_env_cfg,
)
from .rl_cfg import rumi_ppo_runner_cfg
from .runner import RumiVelocityRunner

register_mjlab_task(
  task_id="Mjlab-Velocity-Rough-Rumi",
  env_cfg=rumi_rough_env_cfg(),
  play_env_cfg=rumi_rough_env_cfg(play=True),
  rl_cfg=rumi_ppo_runner_cfg(),
  runner_cls=RumiVelocityRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Rumi",
  env_cfg=rumi_flat_env_cfg(),
  play_env_cfg=rumi_flat_env_cfg(play=True),
  rl_cfg=rumi_ppo_runner_cfg(),
  runner_cls=RumiVelocityRunner,
)
