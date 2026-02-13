"""Rumi velocity environment configurations."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from rumi_velocity.rumi.rumi_constants import (
  RUMI_ACTION_SCALE,
  get_rumi_robot_cfg,
)


def rumi_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Rumi rough terrain velocity configuration."""
  cfg = make_velocity_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500

  cfg.scene.entities = {"robot": get_rumi_robot_cfg()}

  foot_names = ("FL", "FR", "BL", "BR")
  site_names = ("FL", "FR", "BL", "BR")
  geom_names = tuple(f"{name}_foot_collision" for name in foot_names)

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(mode="geom", pattern=geom_names, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  nonfoot_ground_cfg = ContactSensorCfg(
    name="nonfoot_ground_touch",
    primary=ContactMatch(
      mode="geom",
      entity="robot",
      # Grab all collision geoms...
      pattern=r".*_collision\d*$",
      # Except for the foot geoms.
      exclude=tuple(geom_names),
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found",),
    reduce="none",
    num_slots=1,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    nonfoot_ground_cfg,
  )

  # Remove terrain scan sensor (not used for Rumi)
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = RUMI_ACTION_SCALE

  cfg.viewer.body_name = "body"
  cfg.viewer.distance = 2.0
  cfg.viewer.elevation = -10.0

  cfg.observations["critic"].terms["foot_height"].params[
    "asset_cfg"
  ].site_names = site_names

  # Remove height scan observations (not used for Rumi)
  cfg.observations["actor"].terms.pop("height_scan", None)
  cfg.observations["critic"].terms.pop("height_scan", None)

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("body",)

  cfg.rewards["pose"].params["std_standing"] = {
    ".*_hip_joint": 0.05,
    ".*_thigh_joint": 0.05,
    ".*_calf_joint": 0.1,
  }
  cfg.rewards["pose"].params["std_walking"] = {
    ".*_hip_joint": 0.3,
    ".*_thigh_joint": 0.3,
    ".*_calf_joint": 0.6,
  }
  cfg.rewards["pose"].params["std_running"] = {
    ".*_hip_joint": 0.3,
    ".*_thigh_joint": 0.3,
    ".*_calf_joint": 0.6,
  }

  cfg.rewards["upright"].params["asset_cfg"].body_names = ("body",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("body",)

  for reward_name in ["foot_clearance", "foot_swing_height", "foot_slip"]:
    cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

  # Set foot height target to 7cm instead of 10cm
  cfg.rewards["foot_clearance"].params["target_height"] = 0.07
  cfg.rewards["foot_swing_height"].params["target_height"] = 0.07

  # Enable all rewards matching Go1 configuration
  cfg.rewards["upright"].weight = 1.0
  cfg.rewards["pose"].weight = 1.0
  cfg.rewards["dof_pos_limits"].weight = -1.0
  cfg.rewards["action_rate_l2"].weight = -0.1
  cfg.rewards["foot_clearance"].weight = -2.0
  cfg.rewards["foot_swing_height"].weight = -0.25
  cfg.rewards["foot_slip"].weight = -0.1
  cfg.rewards["soft_landing"].weight = -1e-5
  # Keep disabled (same as Go1)
  cfg.rewards["body_ang_vel"].weight = 0.0
  cfg.rewards["angular_momentum"].weight = 0.0
  cfg.rewards["air_time"].weight = 0.0

  cfg.terminations["illegal_contact"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={"sensor_name": nonfoot_ground_cfg.name},
  )

  cmd = cfg.commands["twist"]
  assert isinstance(cmd, UniformVelocityCommandCfg)
  cmd.viz.z_offset = 0.5

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def rumi_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Rumi flat terrain velocity configuration."""
  cfg = rumi_rough_env_cfg(play=play)

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64

  # Switch to flat terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Disable terrain curriculum (not present in play mode since rough clears all).
  cfg.curriculum.pop("terrain_levels", None)

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-1.5, 2.0)
    twist_cmd.ranges.ang_vel_z = (-0.7, 0.7)

  return cfg
