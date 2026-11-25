"""L1-W + Airbot 2PWBC task on the shared manager-based runtime."""

from dataclasses import dataclass, field

import numpy as np
from unilab.base import registry
from unilab.base.scene import SceneCfg

from legged_manipulation_unilab.assets import ASSETS_ROOT_PATH
from legged_manipulation_unilab.tasks.go2_arm.base import Asset, Go2ArmSensor
from legged_manipulation_unilab.tasks.go2_arm.manip_loco import (
    CommandsConfig,
    Go2ArmManipLocoCfg,
    InitState,
    RewardParameters,
    make_go2_arm_manip_loco_env,
)
from legged_manipulation_unilab.tasks.l1_w_arm.rewards import L1WArmReward

_SCENE = str(ASSETS_ROOT_PATH / "robots/l1_w_arm/scene_2pwbc.xml")
_FEET = ("FBL", "FAR", "RBL", "RAR")
_WHEELS = ("FBL", "FAR", "RAR", "RBL")


@dataclass
class L1WAsset(Asset):
    base_name: str = "BASE_LINK"
    arm_mount_name: str = "arm_base_on_l1"
    wheel_joint_names: tuple[str, ...] = tuple(f"{name}_FOOT_JOINT" for name in _WHEELS)
    wheel_radius: float = 0.071
    feet_geom_names: tuple[str, ...] = tuple(f"{name}_FOOT" for name in _FEET)


@dataclass
class L1WSensor(Go2ArmSensor):
    feet_force: list[str] = field(default_factory=lambda: [f"{name}_tire_force" for name in _FEET])
    feet_pos: list[str] = field(default_factory=lambda: [f"{name}_pos" for name in _FEET])
    leg_collision_groups: tuple[tuple[str, ...], ...] = tuple(
        (f"{name}_thigh_contact", f"{name}_calf_contact") for name in _FEET
    )


@dataclass
class L1WCommandsConfig(CommandsConfig):
    enable_gait_phase: bool = False
    pure_x_command_prob: float = 0.35
    gait_frequency: float = 0.0


@dataclass
class L1WRewardParameters(RewardParameters):
    swing_reward_sigma: float = 0.01


@registry.envcfg("L1WArmManipLoco")
@dataclass
class L1WArmManipLocoCfg(Go2ArmManipLocoCfg):
    scene: SceneCfg = field(
        default_factory=lambda: SceneCfg(model_file=_SCENE, default_keyframe_name="home")
    )
    model_file: str = _SCENE
    init_state: InitState = field(default_factory=lambda: InitState(pos=[0.0, 0.0, 0.4]))
    asset: Asset = field(default_factory=L1WAsset)
    sensor: Go2ArmSensor = field(default_factory=L1WSensor)
    command_config: CommandsConfig = field(default_factory=L1WCommandsConfig)
    reward_parameters: RewardParameters = field(
        default_factory=lambda: L1WRewardParameters(base_height_target=0.30, target_foot_height=0.10)
    )
    reward_term_type = L1WArmReward
    imagined_reward_class: str = "legged_manipulation_unilab.tasks.l1_w_arm.rewards.L1WImaginedRewards"
    stage2_disabled_rewards: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.model_file != _SCENE and self.scene.model_file == _SCENE:
            self.scene = SceneCfg(model_file=self.model_file, default_keyframe_name="home")
        self._configure_managers()

    def robot_body_names(self) -> tuple[str, ...]:
        return (
            "BASE_LINK",
            *(f"{leg}_{link}_LINK" for leg in ("FL", "FR", "RL", "RR")
              for link in ("ABAD", "HIP", "KNEE", "FOOT")),
            self.asset.arm_mount_name,
            "arm_base",
            *(f"link{i}" for i in range(1, 7)),
        )

    def mass_randomization_bodies(self) -> tuple[str, ...]:
        return tuple(name for name in self.robot_body_names() if name != self.asset.arm_mount_name)

    def position_actuator_gains(self) -> dict[str, np.ndarray]:
        return {**super().position_actuator_gains(), "actuator_ids": np.r_[0:12, 16:22].astype(np.int32)}


def make_l1_w_arm_manip_loco_env(cfg: L1WArmManipLocoCfg, num_envs: int = 1, backend_type: str = "mujoco"):
    return make_go2_arm_manip_loco_env(cfg, num_envs, backend_type)


registry.register_env("L1WArmManipLoco", make_l1_w_arm_manip_loco_env, "mujoco")
