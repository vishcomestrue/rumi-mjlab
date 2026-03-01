"""Rumi quadruped constants."""

from pathlib import Path

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

_HERE = Path(__file__).parent

RUMI_XML: Path = _HERE / "xmls" / "rumi.xml"
assert RUMI_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, RUMI_XML.parent / "assets", meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(RUMI_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec


##
# Actuator config.
##

EFFORT_LIMIT = 6.0
ARMATURE = 0.031
# FRICTIONLOSS = 0.008453

# PD gains derived from armature, targeting 6 Hz natural frequency.
NATURAL_FREQ = 1 * 2.0 * 3.1415926535  # 6 Hz
DAMPING_RATIO = 2.0

# Base stiffness for hip/thigh
# STIFFNESS = ARMATURE * NATURAL_FREQ**2
# DAMPING = 2 * DAMPING_RATIO * ARMATURE * NATURAL_FREQ
STIFFNESS = 10.0
DAMPING = 0.0

# Separate actuator configs for hip/thigh vs calf
RUMI_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_joint", ".*_thigh_joint", ".*_calf_joint",),
  stiffness=STIFFNESS,
  damping=DAMPING,
  effort_limit=EFFORT_LIMIT,
  armature=ARMATURE,
  # frictionloss=FRICTIONLOSS,
)

##
# Keyframes.
##

INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.1),
  joint_pos={
    ".*_hip_joint": 0.0,
    "FL_thigh_joint": 0.0,
    "FR_thigh_joint": 0.0,
    "BL_thigh_joint": 0.0,
    "BR_thigh_joint": 0.0,
    "FL_calf_joint": 0.0,
    "FR_calf_joint": 0.0,
    "BL_calf_joint": 0.0,
    "BR_calf_joint": 0.0,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

_foot_regex = r"^[FB][LR]_foot_collision$"

FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={_foot_regex: 3, ".*_collision": 1},
  priority={_foot_regex: 1},
  friction={_foot_regex: (0.6,)},
  solimp={_foot_regex: (0.9, 0.95, 0.023)},
  contype=1,
  conaffinity=0,
)

##
# Final config.
##

RUMI_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(RUMI_ACTUATOR,),
  soft_joint_pos_limit_factor=0.9,
)


def get_rumi_robot_cfg() -> EntityCfg:
  """Get a fresh Rumi robot configuration instance.

  Returns a new EntityCfg instance each time to avoid mutation issues when
  the config is shared across multiple places.
  """
  return EntityCfg(
    init_state=INIT_STATE,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=RUMI_ARTICULATION,
  )


RUMI_ACTION_SCALE: dict[str, float] = {}
for _a in RUMI_ARTICULATION.actuators:
  assert isinstance(_a, BuiltinPositionActuatorCfg)
  _e = _a.effort_limit
  _s = _a.stiffness
  _names = _a.target_names_expr
  assert _e is not None
  for _n in _names:
    RUMI_ACTION_SCALE[_n] = 0.25 * _e / _s


if __name__ == "__main__":
  import mujoco.viewer as viewer
  from mjlab.entity.entity import Entity

  robot = Entity(get_rumi_robot_cfg())

  viewer.launch(robot.spec.compile())
