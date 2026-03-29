"""Rumi getup environment configuration."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.getup import mdp as getup_mdp
from mjlab.tasks.getup.getup_env_cfg import GetupTaskCfg, make_getup_env_cfg

import rumi_getup.mdp as rumi_mdp
from rumi_getup.rumi.rumi_constants import (
    RUMI_ACTION_SCALE,
    get_rumi_robot_cfg,
)


def rumi_getup_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create Rumi getup environment configuration.

    Args:
        play: If True, use play mode settings.

    Returns:
        Complete environment configuration for Rumi getup task.
    """
    # Get Rumi robot configuration
    robot_cfg = get_rumi_robot_cfg()

    # Define task-specific parameters for Rumi
    task_cfg = GetupTaskCfg(
        # Target standing pose
        target_joint_pos={
            # Front Left leg
            "FL_hip_joint": 0.0,
            "FL_thigh_joint": 0.0,
            "FL_calf_joint": -0.41,
            # Front Right leg
            "FR_hip_joint": 0.0,
            "FR_thigh_joint": 0.0,
            "FR_calf_joint": 0.41,
            # Back Left leg
            "BL_hip_joint": 0.0,
            "BL_thigh_joint": 0.0,
            "BL_calf_joint": -0.491,
            # Back Right leg
            "BR_hip_joint": 0.0,
            "BR_thigh_joint": 0.0,
            "BR_calf_joint": 0.491,
        },
        # Initial seated pose (crouched position)
        # Rumi joint names: FL/FR/BL/BR + _hip_joint/_thigh_joint/_calf_joint
        seated_joint_pos={
            # Front Left leg
            "FL_hip_joint": 0.0,
            "FL_thigh_joint": 0.0,
            "FL_calf_joint": 0.0,
            # Front Right leg
            "FR_hip_joint": 0.0,
            "FR_thigh_joint": 0.0,
            "FR_calf_joint": 0.0,
            # Back Left leg
            "BL_hip_joint": 0.0,
            "BL_thigh_joint": 0.0,
            "BL_calf_joint": 0.0,
            # Back Right leg
            "BR_hip_joint": 0.0,
            "BR_thigh_joint": 0.0,
            "BR_calf_joint": 0.0,
        },
        # Rumi body height when standing (from INIT_STATE pos z=0.2)
        target_body_height=0.18,
        min_body_height=0.16,
        # Maximum tilt angle before termination
        max_tilt_angle=30.0,
        # Episode timeout
        episode_timeout=20.0,
        # Action scale
        action_scale=0.1,
    )

    # Rumi specific settings
    body_name = "body"
    foot_site_names = ["FL", "FR", "BL", "BR"]
    geom_names = tuple(f"{name}_foot_collision" for name in foot_site_names)

    # Create base getup config
    cfg = make_getup_env_cfg(
        robot_cfg=robot_cfg,
        robot_name="rumi",
        body_name=body_name,
        foot_site_names=foot_site_names,
        task_cfg=task_cfg,
        play=play,
    )

    # Customize simulation settings for Rumi
    cfg.sim.mujoco.ccd_iterations = 500
    cfg.sim.contact_sensor_maxmatch = 500

    # Contact sensor: detect foot-ground contact per foot (FL, FR, BL, BR)
    foot_contact_cfg = ContactSensorCfg(
        name="foot_ground_touch",
        primary=ContactMatch(
            mode="geom",
            entity="robot",
            pattern=r"(FL|FR|BL|BR)_foot_collision",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )
    cfg.scene.sensors = (cfg.scene.sensors or ()) + (foot_contact_cfg,)

    # Penalize any foot not in contact with the ground.
    # With all 4 feet grounded: 0. Each lifted foot adds -1 penalty.
    cfg.rewards["foot_contact"] = RewardTermCfg(
        func=rumi_mdp.foot_contact_penalty,
        weight=0.5,
        params={"sensor_name": foot_contact_cfg.name},
    )

    # Contact sensor: detect non-foot geoms touching the terrain
    nonfoot_ground_cfg = ContactSensorCfg(
        name="nonfoot_ground_touch",
        primary=ContactMatch(
            mode="geom",
            entity="robot",
            pattern=r".*_collision\d*$",
            exclude=tuple(geom_names),
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )
    cfg.scene.sensors = (cfg.scene.sensors or ()) + (nonfoot_ground_cfg,)

    # Penalize non-foot contacts rather than terminating, since the robot
    # starts on the ground and needs time to stand up.
    cfg.rewards["illegal_contact"] = RewardTermCfg(
        func=getup_mdp.illegal_contact_penalty,
        weight=0.5,
        params={"sensor_name": nonfoot_ground_cfg.name},
    )

    # Set Rumi-specific action scale
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = RUMI_ACTION_SCALE

    # Customize viewer for Rumi
    cfg.viewer.body_name = body_name
    cfg.viewer.distance = 2.0
    cfg.viewer.elevation = -10.0

    # ==================================================================
    # Modifications in observations
    # ==================================================================
    del cfg.observations["actor"].terms["base_lin_vel"]
    del cfg.observations["critic"].terms["base_lin_vel"]
    del cfg.observations["actor"].terms["base_ang_vel"]
    del cfg.observations["critic"].terms["base_ang_vel"]

    # Replace ground-truth height with FK estimate (encoder + IMU only).
    # Uses rumi.xml "orientation" framequat sensor scoped to "robot" entity.
    fk_height_term = ObservationTermCfg(func=rumi_mdp.fk_body_height)
    cfg.observations["actor"].terms["body_height"] = fk_height_term
    cfg.observations["critic"].terms["body_height"] = fk_height_term
    
    # ==================================================================
    # Reward Weight Tuning (Set to 0 to disable, adjust to test)
    # ==================================================================
    # Uncomment to override default weights for new rewards:

    # Joint symmetry (left-right leg matching)
    # cfg.rewards["joint_symmetry"].weight = 1.0  # Default: 0.5

    # Hip stability (keep hips near neutral)
    # cfg.rewards["hip_stability"].weight = 0.5  # Default: 0.3

    # Joint velocity smoothness
    cfg.rewards["dof_vel"].weight = 0.01  # Default: 0.001

    # Torque minimization
    cfg.rewards["torques"].weight = 0.001  # Default: 0.0001

    # To test incrementally, enable ONE at a time:
    # Step 1: Enable joint_symmetry (set to 0.5)
    # Step 2: Enable hip_stability (set to 0.3)
    # Step 3: Enable dof_vel (set to 0.001)
    # Step 4: Enable torques (set to 0.0001)

    # ==================================================================
    # Curriculum Learning - Target Height
    # ==================================================================
    # Override default curriculum with Rumi-specific stages
    # Curriculum will be logged to wandb as: Curriculum/target_height/min_height
    #                                    and: Curriculum/target_height/max_height

    # In play mode, fix the target height so behaviour is deterministic.
    if play:
        cfg.events["randomize_target_height"].params["min_height"] = 0.20
        cfg.events["randomize_target_height"].params["max_height"] = 0.30

    cfg.curriculum["target_height"] = CurriculumTermCfg(
        func=getup_mdp.target_height_curriculum,
        params={
            "height_stages": [
                {"step": 0, "min_height": 0.24, "max_height": 0.25},
                {"step": 1500 * 24, "min_height": 0.21, "max_height": 0.28},
                {"step": 3000 * 24, "min_height": 0.18, "max_height": 0.31},
                # {"step": 3000 * 24, "min_height": 0.14, "max_height": 0.30},
            ],
        },
    )

    return cfg
