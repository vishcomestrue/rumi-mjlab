"""Custom runner for Rumi velocity task."""

from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner


class RumiVelocityRunner(VelocityOnPolicyRunner):
    """Runner that logs Rumi physical parameters to W&B at run start.

    Extends VelocityOnPolicyRunner (which handles ONNX export on save) and
    patches ``logger.init_logging_writer`` so that after the base class
    initialises wandb (and calls ``store_config``), the Rumi-specific
    actuator / collision / action-scale constants are appended to the
    wandb run config as ``rumi_params``.
    """

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        original_init = self.logger.init_logging_writer

        def _patched_init() -> None:
            original_init()         # wandb.init() + store_config happen here
            self._log_rumi_params()

        self.logger.init_logging_writer = _patched_init
        try:
            super().learn(num_learning_iterations, init_at_random_ep_len)
        finally:
            self.logger.init_logging_writer = original_init

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _log_rumi_params(self) -> None:
        """Push Rumi physical constants to the active wandb run config."""
        try:
            import wandb

            if wandb.run is None:
                return
        except ImportError:
            return

        from rumi_velocity.rumi.rumi_constants import (
            ARMATURE,
            DAMPING,
            DAMPING_RATIO,
            EFFORT_LIMIT,
            FRICTIONLOSS,
            FULL_COLLISION,
            INIT_STATE,
            JOINT_DAMPING,
            NATURAL_FREQ,
            RUMI_ACTION_SCALE,
            STIFFNESS,
        )

        action_scale_factor = 0.25
        action_scale_value = action_scale_factor * EFFORT_LIMIT / STIFFNESS

        # FULL_COLLISION stores per-regex dicts; extract the single foot entry.
        foot_friction = list(next(iter(FULL_COLLISION.friction.values())))
        foot_solimp = list(next(iter(FULL_COLLISION.solimp.values())))

        wandb.config.update(
            {
                "rumi_params": {
                    "actuator": {
                        "effort_limit": EFFORT_LIMIT,
                        "armature": ARMATURE,
                        "frictionloss": FRICTIONLOSS,
                        "joint_damping": JOINT_DAMPING,
                        "natural_freq_rad_s": NATURAL_FREQ,
                        "damping_ratio": DAMPING_RATIO,
                        "stiffness_kp": STIFFNESS,
                        "damping_kd": DAMPING,
                    },
                    "init_state": {
                        "pos": list(INIT_STATE.pos),
                        "joint_pos": dict(INIT_STATE.joint_pos),
                        "joint_vel": dict(INIT_STATE.joint_vel),
                    },
                    "collision": {
                        "foot_friction": foot_friction,
                        "foot_solimp": foot_solimp,
                        "foot_condim": 3,
                        "body_condim": 1,
                    },
                    "action_scale": {
                        "formula_factor": action_scale_factor,
                        "effort_limit": EFFORT_LIMIT,
                        "stiffness": STIFFNESS,
                        "computed_scale": action_scale_value,
                        "per_joint_pattern": dict(RUMI_ACTION_SCALE),
                    },
                }
            },
            allow_val_change=True,
        )
