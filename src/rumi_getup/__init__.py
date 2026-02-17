"""Rumi getup task registration."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.rl.runner import MjlabOnPolicyRunner

from .env_cfgs import rumi_getup_env_cfg
from .rl_cfg import rumi_getup_ppo_runner_cfg

register_mjlab_task(
    task_id="Mjlab-Getup-Rumi",
    env_cfg=rumi_getup_env_cfg(),
    play_env_cfg=rumi_getup_env_cfg(play=True),
    rl_cfg=rumi_getup_ppo_runner_cfg(),
    runner_cls=MjlabOnPolicyRunner,
)
