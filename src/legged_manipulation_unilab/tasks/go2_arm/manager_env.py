"""Manager-Based action, command, observation, reward, and termination terms."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from unilab.base.entity import Entity
from unilab.dtype_config import get_global_dtype
from unilab.envs.manager_based_rl_env import ManagerBasedRlEnv
from unilab.managers.action_manager import ActionTerm, ActionTermCfg
from unilab.managers.command_manager import CommandTerm, CommandTermCfg
from unilab.managers.manager_base import ManagerTermBase
from unilab.utils.rotation import (
    np_matrix_from_quat,
    np_quat_apply_batched,
    np_quat_conjugate_batched,
    np_quat_from_euler_xyz,
    np_quat_mul_batched,
    np_yaw_quat,
)

from legged_manipulation_unilab.tasks.geometry import (
    np_cartesian_to_spherical as _cart2sphere,
)
from legged_manipulation_unilab.tasks.geometry import np_quat_orientation_error_local
from legged_manipulation_unilab.tasks.geometry import (
    np_spherical_to_cartesian as _sphere2cart,
)


def clip_ee_goal_cart(cart: np.ndarray, goal_ee: Any) -> np.ndarray:
    """Keep an interactive EE goal inside the configured spherical workspace."""
    dtype = get_global_dtype()
    value = np.asarray(cart, dtype=dtype).reshape(3)
    sphere = _cart2sphere(value[None, :])[0]
    sphere[0] = np.clip(sphere[0], *goal_ee.sphere_l_range)
    sphere[1] = np.clip(sphere[1], *goal_ee.sphere_phi_range)
    sphere[2] = np.clip(sphere[2], *goal_ee.sphere_theta_range)
    return _sphere2cart(sphere[None, :])[0]


def _task_cfg(env: ManagerBasedRlEnv) -> Any:
    cfg = env._cfg
    if not hasattr(cfg, "goal_ee"):
        raise TypeError("Go2Arm manager terms require Go2ArmManipLocoCfg")
    return cfg


def _command(env: Any) -> "Go2ArmManipLocoCommand":
    term = env.command_manager.get_term("task")
    if not isinstance(term, Go2ArmManipLocoCommand):
        raise TypeError(f"Expected Go2ArmManipLocoCommand, got {type(term).__name__}")
    return term


_UPVECTOR_SENSOR = "upvector"


def _normal_cdf(x: np.ndarray) -> np.ndarray:
    """Vectorized normal CDF (maximum approximation error about 8e-8)."""
    z = np.asarray(x, dtype=np.float64)
    t = 1.0 / (1.0 + 0.2316419 * np.abs(z))
    poly = t * (
        0.319381530
        + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429)))
    )
    positive = 1.0 - np.exp(-0.5 * z * z) * 0.3989422804014327 * poly
    return np.where(z >= 0.0, positive, 1.0 - positive)


def _wbc2p_desired_contact(phase: np.ndarray, command: np.ndarray, kappa: float) -> np.ndarray:
    """Reference WTW's smoothed stance probability for each foot."""
    # The reference retains raw self.foot_indices for Raibert, but changes
    # the local foot_indices used by clock_inputs and desired_contact_states
    # to phase 0.25 when the velocity command is zero.
    foot_phase = np.where(
        np.linalg.norm(command[:, :3], axis=1, keepdims=True) < 0.1,
        0.25,
        phase,
    )
    p = np.mod(foot_phase, 1.0)

    def cdf(offset: float) -> np.ndarray:
        return _normal_cdf((p - offset) / kappa)

    stance = cdf(0.0) * (1.0 - cdf(0.5)) + cdf(1.0) * (1.0 - cdf(1.5))
    return np.clip(stance, 0.0, 1.0).astype(get_global_dtype())


_ARM_TOUCH_SENSORS = (
    "arm_touch_base",
    "arm_touch_link1",
    "arm_touch_link2",
    "arm_touch_link3",
    "arm_touch_link4",
    "arm_touch_link5",
    "arm_touch_link6",
    "arm_touch_eef",
    "arm_touch_g2base",
)

_LEG_COLLISION_SENSOR_GROUPS = tuple(
    (f"{leg}_thigh_contact", f"{leg}_calf_contact1", f"{leg}_calf_contact2")
    for leg in ("FL", "FR", "RL", "RR")
)


@dataclass(kw_only=True)
class Go2ArmIKActionCfg(ActionTermCfg):
    """Leg and arm position targets with optional wheel velocity targets."""

    def build(self, env: "ManagerBasedRlEnv") -> "Go2ArmIKAction":  # pyright: ignore[reportIncompatibleMethodOverride]
        return Go2ArmIKAction(self, env)


class Go2ArmIKAction(ActionTerm):
    cfg: Go2ArmIKActionCfg

    def __init__(self, cfg: Go2ArmIKActionCfg, env: "ManagerBasedRlEnv"):
        super().__init__(cfg, env)
        cfg_task = _task_cfg(env)
        self._task_cfg = cfg_task
        entity: Entity = env.scene[cfg.entity_name]
        joint_ids, joint_names = entity.find_joints_by_actuator_names((".*",))
        self._wheel_names = tuple(cfg_task.asset.wheel_joint_names)
        self._arm_start = 12 + len(self._wheel_names)
        self._action_dim = self._arm_start + 6
        if len(joint_ids) != self._action_dim or tuple(joint_names[12:self._arm_start]) != self._wheel_names:
            raise ValueError(f"Expected 12 legs, {self._wheel_names} wheels and 6 arm actuators; got {joint_names}")
        self._entity = entity
        self._joint_ids = np.asarray(joint_ids, dtype=np.intp)
        self._joint_ids.setflags(write=False)
        self._joint_names = tuple(joint_names)
        arm_names = set(cfg_task.asset.arm_joint_names)
        self._arm_rows = np.asarray(
            [index for index, name in enumerate(joint_names) if name in arm_names], dtype=np.intp
        )
        if self._arm_rows.size != 6:
            raise ValueError("Go2Arm manager action could not resolve all six arm actuators")
        self._site_id = int(env._backend.get_site_ids([cfg_task.asset.ee_site_name])[0])
        self._arm_dof_ids = np.asarray(
            env._backend.get_joint_dof_indices(cfg_task.asset.arm_joint_names), dtype=np.intp
        )
        self._ee_pos_view = env.scene.bind_sensor_data((cfg_task.sensor.ee_local_pos,))
        self._ee_quat_view = env.scene.bind_sensor_data((cfg_task.sensor.ee_local_quat,))
        self._arm_ref_quat_view = env.scene.bind_sensor_data((cfg_task.sensor.arm_ref_world_quat,))
        dtype = get_global_dtype()
        self._raw_actions = np.zeros((env.num_envs, self._action_dim), dtype=dtype)
        self._target = np.zeros_like(self._raw_actions)
        self._previous_target = np.zeros_like(self._raw_actions)
        self._last_last_target = np.zeros_like(self._raw_actions)
        self._unclipped_target = np.zeros_like(self._raw_actions)
        self._previous_unclipped_target = np.zeros_like(self._raw_actions)
        self._last_last_unclipped_target = np.zeros_like(self._raw_actions)
        self._leg_target_clip_fraction = np.zeros(env.num_envs, dtype=dtype)
        self._has_previous_leg_target = np.zeros(env.num_envs, dtype=bool)
        self._stand_action_smoothing = float(cfg_task.control_config.stand_action_smoothing)
        if not 0.0 <= self._stand_action_smoothing < 1.0:
            raise ValueError("control_config.stand_action_smoothing must be in [0, 1)")

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def raw_action(self) -> np.ndarray:
        return self._raw_actions

    def reset(self, env_ids: np.ndarray | slice | None = None) -> None:
        ids = env_ids if env_ids is not None else slice(None)
        self._raw_actions[ids] = 0.0
        self._target[ids] = 0.0
        self._previous_target[ids] = 0.0
        self._last_last_target[ids] = 0.0
        self._unclipped_target[ids] = 0.0
        self._previous_unclipped_target[ids] = 0.0
        self._last_last_unclipped_target[ids] = 0.0
        self._leg_target_clip_fraction[ids] = 0.0
        self._has_previous_leg_target[ids] = False

    def process_actions(self, actions: np.ndarray) -> None:
        if actions.shape != (self.num_envs, self._action_dim):
            raise ValueError(f"Go2Arm action shape must be {(self.num_envs, self._action_dim)}")
        if not np.isfinite(actions).all():
            raise ValueError("Go2Arm action contains NaN or Inf")
        self._last_last_target[:] = self._previous_target
        self._previous_target[:] = self._target
        self._last_last_unclipped_target[:] = self._previous_unclipped_target
        self._previous_unclipped_target[:] = self._unclipped_target
        self._raw_actions[:] = actions
        if self._task_cfg.arm_stage.freeze_arm_joints:
            self._raw_actions[:, self._arm_start:] = 0.0
        exec_actions = (
            self._env.action_manager.prev_action
            if self._task_cfg.control_config.simulate_action_latency
            else self._raw_actions
        )
        default = self._entity.data.default_joint_pos
        leg_action = exec_actions[:, :12].copy()
        hip_scale = float(self._task_cfg.control_config.hip_action_scale_reduction)
        leg_action[:, [0, 3, 6, 9]] *= hip_scale
        leg_target = leg_action * self._task_cfg.control_config.action_scale
        leg_target += default[:, :12]
        # Suppress target chatter at zero command without affecting locomotion.
        # Filtering is applied at the *actuator target* in both training and
        # playback, so there is no training/deployment control-path mismatch.
        # The first target after reset is never blended with a zero buffer.
        alpha = self._stand_action_smoothing
        if alpha > 0.0:
            vel_command = _command(self._env).vel_command_b
            stationary = ~_command(self._env)._command_is_moving(vel_command)
            blend = stationary & self._has_previous_leg_target
            if np.any(blend):
                leg_target[blend] = (
                    alpha * self._previous_target[blend, :12]
                    + (1.0 - alpha) * leg_target[blend]
                )
        if self._task_cfg.arm_stage.freeze_arm_joints:
            arm_target = np.broadcast_to(default[:, self._arm_start:], (self.num_envs, 6)).copy()
        else:
            mode = str(self._task_cfg.control_config.arm_action_mode).lower()
            if mode == "raw":
                # RoboDuet-compatible path: the six policy outputs directly
                # control six arm position residuals around the XML home pose.
                arm_target = (
                    default[:, self._arm_start:]
                    + exec_actions[:, self._arm_start:]
                    * self._task_cfg.control_config.arm_action_scale
                )
            elif mode == "ik":
                command = _command(self._env)
                ee_pos = np.asarray(self._ee_pos_view.read(), dtype=get_global_dtype())
                ee_quat = self._ee_quat_view.read()
                arm_pos = self._entity.data.joint_pos[:, self._joint_ids[self._arm_rows]]
                dq = self._compute_arm_ik_delta(
                    command.curr_ee_goal_cart,
                    ee_pos,
                    command.ee_goal_orn_quat,
                    ee_quat,
                )
                arm_target = (
                    arm_pos
                    + exec_actions[:, self._arm_start:] * self._task_cfg.control_config.arm_action_scale
                    + self._task_cfg.ik.gain * dq
                )
            else:
                raise ValueError(
                    "control_config.arm_action_mode must be 'raw' or 'ik', "
                    f"got {self._task_cfg.control_config.arm_action_mode!r}"
                )
        self._target[:, :12] = leg_target
        self._target[:, 12:self._arm_start] = (
            exec_actions[:, 12:self._arm_start] * self._task_cfg.control_config.wheel_action_scale
        )
        self._target[:, self._arm_start:] = arm_target
        self._unclipped_target[:] = self._target
        ranges = self._entity.data.actuator_ctrl_range
        self._leg_target_clip_fraction[:] = np.mean(
            (self._target[:, :12] < ranges[:12, 0])
            | (self._target[:, :12] > ranges[:12, 1]),
            axis=1,
        )
        np.clip(self._target, ranges[:, 0], ranges[:, 1], out=self._target)
        self._has_previous_leg_target[:] = True

    def apply_actions(self) -> None:
        if self._wheel_names:
            position_rows = np.r_[0:12, self._arm_start:self._action_dim]
            wheels = slice(12, self._arm_start)
            self._entity.set_joint_position_target(
                self._target[:, position_rows], joint_ids=self._joint_ids[position_rows]
            )
            self._entity.set_joint_velocity_target(
                self._target[:, wheels], joint_ids=self._joint_ids[wheels]
            )
        else:
            self._entity.set_joint_position_target(self._target, joint_ids=self._joint_ids)

    def _compute_arm_ik_delta(
        self,
        goal_pos: np.ndarray,
        curr_pos: np.ndarray,
        goal_quat: np.ndarray,
        curr_quat: np.ndarray,
    ) -> np.ndarray:
        cfg = self._task_cfg.ik
        pos_err = goal_pos - curr_pos
        backend = getattr(self._env, "_backend")
        jacp_w, jacr_w = backend.get_site_jacobian_w(
            self._site_id,
            self._arm_dof_ids,
        )
        ref_rot_w = np_matrix_from_quat(self._arm_ref_quat_view.read())
        rot_w_to_b = np.swapaxes(ref_rot_w, 1, 2)
        jacp_b = np.matmul(rot_w_to_b, jacp_w)
        if cfg.use_orientation:
            if cfg.orientation_mode == "target":
                orn_err = np_quat_orientation_error_local(goal_quat, curr_quat)
            elif cfg.orientation_mode == "zero_error":
                orn_err = np.zeros_like(pos_err)
            else:
                raise ValueError("ik.orientation_mode must be 'target' or 'zero_error'")
            jac = np.concatenate([jacp_b, np.matmul(rot_w_to_b, jacr_w)], axis=1)
            dpose = np.concatenate([pos_err, orn_err], axis=1)
        else:
            jac = jacp_b
            dpose = pos_err
        identity = np.eye(jac.shape[1], dtype=jac.dtype)[None, :, :]
        lhs = np.matmul(jac, np.swapaxes(jac, 1, 2)) + identity * (cfg.damping**2)
        solved = np.linalg.solve(lhs, dpose[:, :, None])
        dq = np.matmul(np.swapaxes(jac, 1, 2), solved)[:, :, 0]
        if cfg.dq_clip > 0.0:
            np.clip(dq, -cfg.dq_clip, cfg.dq_clip, out=dq)
        return dq


@dataclass(kw_only=True)
class Go2ArmManipLocoCommandCfg(CommandTermCfg):
    def build(self, env: "ManagerBasedRlEnv") -> "Go2ArmManipLocoCommand":  # pyright: ignore[reportIncompatibleMethodOverride]
        return Go2ArmManipLocoCommand(self, env)


class Go2ArmManipLocoCommand(CommandTerm):
    """Velocity command plus task-owned EE trajectory and gait phase state."""

    cfg: Go2ArmManipLocoCommandCfg

    def __init__(self, cfg: Go2ArmManipLocoCommandCfg, env: "ManagerBasedRlEnv"):
        super().__init__(cfg, env)
        cfg_task = _task_cfg(env)
        self._task_cfg = cfg_task
        n = env.num_envs
        dtype = get_global_dtype()
        self.vel_command_b = np.zeros((n, 3), dtype=dtype)
        has_gait_phase = cfg_task.command_config.enable_gait_phase
        self.phase = np.zeros(n, dtype=np.float32) if has_gait_phase else None
        self.pitch_roll = np.zeros((n, 2), dtype=dtype)
        self.plan_raw = np.zeros_like(self.pitch_roll)
        self.previous_plan_raw = np.zeros_like(self.pitch_roll)
        self.yaw_frame = cfg_task.wbc_switch.world_model_control
        if self.yaw_frame:
            import mujoco

            model = mujoco.MjModel.from_xml_path(str(cfg_task.scene.model_file))
            robot = env.scene["robot"]
            mount = model.body(cfg_task.asset.arm_mount_name)
            self.ground_height = float(model.geom(cfg_task.asset.ground).pos[2])
            self.reference_height = (
                self.ground_height + float(robot.data.default_root_state[0, 2]) + float(mount.pos[2])
            )
            self._robot_data = robot.data
            if cfg_task.asset.wheel_joint_names:
                site = model.site(cfg_task.asset.ee_site_name)
                body = model.body(int(model.site_bodyid[site.id]))
                self._tcp_body_ids = env._backend.get_body_ids((body.name,))
                self._tcp_rows = np.arange(n, dtype=np.intp)
                self._tcp_offset = np.broadcast_to(site.pos.astype(dtype), (n, 3))
                self._tcp_rotation = np.broadcast_to(site.quat.astype(dtype), (n, 4))
            else:
                self._tcp_pos_view = env.scene.bind_sensor_data((cfg_task.sensor.ee_local_pos,))
                self._tcp_quat_view = env.scene.bind_sensor_data((cfg_task.sensor.ee_local_quat,))

        self.feet_phase = np.zeros(
            (n, len(cfg_task.sensor.feet_force) if has_gait_phase else 0), dtype=np.float32
        )
        self.gait_frequency = float(cfg_task.command_config.gait_frequency)
        self.curr_ee_goal_cart = np.zeros((n, 3), dtype=dtype)
        self.curr_ee_goal_sphere = np.zeros((n, 3), dtype=dtype)
        self.curr_ee_goal_world = np.zeros((n, 3), dtype=dtype)
        self.ee_goal_quat_world = np.zeros((n, 4), dtype=dtype)
        self.ee_goal_orn_euler = np.zeros((n, 3), dtype=dtype)
        self.ee_goal_orn_quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=dtype), (n, 1))
        self._ee_start_sphere = np.zeros((n, 3), dtype=dtype)
        self._ee_goal_sphere = np.zeros((n, 3), dtype=dtype)
        self._arm_goal_timer = np.zeros(n, dtype=np.int32)
        self._traj_steps = np.ones(n, dtype=np.int32)
        self._traj_total_steps = np.ones(n, dtype=np.int32)
        self._episode_tracking_sum = np.zeros(n, dtype=np.float64)
        self._episode_steps = np.zeros(n, dtype=np.int32)
        self._expansion_count = 0
        self._ctrl_dt = float(env.step_dt)
        self._metrics = {
            "tracking_lin_vel": np.zeros(n, dtype=np.float32),
            "episode_steps": np.zeros(n, dtype=np.float32),
        }
        self._resetting_ids: np.ndarray | None = None
        self._pending_ee_reset = np.zeros(n, dtype=bool)
        self._hold_velocity_command = False
        self._hold_ee_command = False
        self._goal_view = env.scene.bind_sensor_data(("armbasepoint_world_pos",))
        self._goal_quat_view = env.scene.bind_sensor_data(("armbasepoint_world_quat",))

    @property
    def command(self) -> np.ndarray:
        return self.vel_command_b

    def enable_velocity_hold(self) -> None:
        self._hold_velocity_command = True

    def enable_ee_hold(self) -> None:
        self._hold_ee_command = True
        self._write_default_orientation(np.arange(self.num_envs, dtype=np.int32))

    def set_velocity_command(self, command: np.ndarray) -> None:
        value = np.asarray(command, dtype=get_global_dtype()).reshape(3)
        low = np.asarray(self._task_cfg.command_config.vel_limit[0], dtype=get_global_dtype())
        high = np.asarray(self._task_cfg.command_config.vel_limit[1], dtype=get_global_dtype())
        self.vel_command_b[:] = np.clip(value, low, high)

    def set_ee_goal_cart(self, cart: np.ndarray) -> None:
        self._write_ee_goal_cart(cart)

    def set_ee_goal_quat(self, quat: np.ndarray) -> None:
        """Set the held tool orientation in the task frame, using wxyz order."""
        value = np.asarray(quat, dtype=get_global_dtype()).reshape(4)
        value = value / np.linalg.norm(value)
        matrix = np_matrix_from_quat(value)
        self.ee_goal_orn_quat[:] = value
        self.ee_goal_orn_euler[:] = (
            np.arctan2(matrix[2, 1], matrix[2, 2]),
            np.arctan2(-matrix[2, 0], np.hypot(matrix[2, 1], matrix[2, 2])),
            np.arctan2(matrix[1, 0], matrix[0, 0]),
        )

    def reset(self, env_ids: np.ndarray | slice | None) -> dict[str, float]:
        ids = env_ids if isinstance(env_ids, np.ndarray) else np.arange(self.num_envs)
        if not isinstance(ids, np.ndarray):  # narrows for type checkers
            raise TypeError("Command reset IDs must be an ndarray")
        # The reference resets gait_indices per episode before sampling a new
        # command, including episodes that start with a standing command.
        if self.phase is not None:
            self.phase[ids] = 0.0
        self.plan_raw[ids] = 0.0
        self.previous_plan_raw[ids] = 0.0
        self.pitch_roll[ids] = 0.0
        self._resetting_ids = ids.copy()
        try:
            result = super().reset(ids)
        finally:
            self._resetting_ids = None
        # Update the curriculum from the finished episode after the reset
        # resample (legacy reset-plan ordering: the fresh commands still use
        # the pre-expansion range), then clear the episode accumulators.
        self._update_command_curriculum(ids)
        self._episode_tracking_sum[ids] = 0.0
        self._episode_steps[ids] = 0
        return result

    def record_tracking(self, value: np.ndarray) -> None:
        """Accumulate the unweighted tracking_lin_vel value for the curriculum."""
        self._episode_tracking_sum += np.asarray(value, dtype=np.float64)

    def _update_command_curriculum(self, env_ids: np.ndarray) -> None:
        """Expand vel_limit when the finished episodes tracked well (legacy rule).

        Mean per-step unweighted tracking_lin_vel above the threshold expands
        the sampling range by ``step_size``, clamped to ``max_vel_limit``.
        Without this the tiny initial range keeps standing near-optimal and
        velocity tracking is never trained.
        """
        curriculum = self._task_cfg.curriculum_config
        if not curriculum.enable:
            return
        ep_steps = np.maximum(self._episode_steps[env_ids], 1)
        mean_per_step = float(np.mean(self._episode_tracking_sum[env_ids] / ep_steps))
        if mean_per_step > float(curriculum.threshold):
            step = np.asarray(curriculum.step_size, dtype=np.float64)
            max_limit = np.asarray(curriculum.max_vel_limit, dtype=np.float64)
            low = np.asarray(self._task_cfg.command_config.vel_limit[0], dtype=np.float64)
            high = np.asarray(self._task_cfg.command_config.vel_limit[1], dtype=np.float64)
            self._task_cfg.command_config.vel_limit = [
                np.clip(low - step, -max_limit, 0.0).tolist(),
                np.clip(high + step, 0.0, max_limit).tolist(),
            ]
            self._expansion_count += 1
            print(
                f"[curriculum] step {self._env.common_step_counter}: vel_limit -> "
                f"low={self._task_cfg.command_config.vel_limit[0]} "
                f"high={self._task_cfg.command_config.vel_limit[1]} "
                f"(expansion #{self._expansion_count}, batch mean {mean_per_step:.3f})"
            )

    def _update_metrics(self, env_ids: np.ndarray | None = None) -> None:
        del env_ids

    def _sample_velocity(self, env_ids: np.ndarray) -> None:
        cfg = self._task_cfg.command_config
        count = len(env_ids)
        low = np.asarray(cfg.vel_limit[0], dtype=get_global_dtype())
        high = np.asarray(cfg.vel_limit[1], dtype=get_global_dtype())
        commands = self._env.rng.uniform(low, high, size=(count, 3)).astype(get_global_dtype())
        pure_forward = getattr(cfg, "pure_x_command_prob", 0.0)
        if pure_forward:
            commands[self._env.rng.random(count) < pure_forward, 1:] = 0.0
        commands[~self._command_is_moving(commands)] = 0.0
        prob = float(cfg.zero_command_prob)
        if prob > 0.0:
            zero = self._env.rng.random(count) < prob
            commands[zero] = 0.0
        self.vel_command_b[env_ids] = commands

    def _resample_command(self, env_ids: np.ndarray) -> None:
        if not self._hold_velocity_command:
            self._sample_velocity(env_ids)
            if self.yaw_frame and self._task_cfg.arm_stage.freeze_arm_joints:
                self.pitch_roll[env_ids] = self._env.rng.uniform(-0.4, 0.4, (len(env_ids), 2))
        resetting = self._resetting_ids
        if resetting is not None and np.array_equal(env_ids, resetting):
            self._reset_ee_goals(env_ids)
        self._write_feet_phase(env_ids)

    def _write_ee_goal_cart(self, cart: np.ndarray) -> bool:
        clipped = clip_ee_goal_cart(cart, self._task_cfg.goal_ee)
        point = np.asarray(clipped, dtype=get_global_dtype()).reshape(3)
        cfg = self._task_cfg.goal_ee
        upper = np.asarray(cfg.collision_upper_limits, dtype=get_global_dtype())
        lower = np.asarray(cfg.collision_lower_limits, dtype=get_global_dtype())
        if (np.all(point < upper) and np.all(point > lower)) or point[2] < self._ee_underground_limit():
            return False
        self.curr_ee_goal_cart[:] = point
        sphere = _cart2sphere(self.curr_ee_goal_cart)
        self.curr_ee_goal_sphere[:] = sphere
        self._ee_start_sphere[:] = sphere
        self._ee_goal_sphere[:] = sphere
        self._arm_goal_timer[:] = 0
        self._traj_steps[:] = 1
        self._traj_total_steps[:] = np.iinfo(np.int32).max // 4
        return True

    def _write_default_orientation(self, env_ids: np.ndarray) -> None:
        cfg = self._task_cfg.goal_ee
        self.ee_goal_orn_euler[env_ids] = [
            cfg.default_orn_roll,
            cfg.arm_induced_pitch,
            0.0,
        ]
        quat = np_quat_from_euler_xyz(
            self.ee_goal_orn_euler[env_ids, 0],
            self.ee_goal_orn_euler[env_ids, 1],
            self.ee_goal_orn_euler[env_ids, 2],
        )
        self.ee_goal_orn_quat[env_ids] = np.asarray(quat, dtype=get_global_dtype())

    def _update_command(self, env_ids: np.ndarray | None) -> None:
        # Base calls this after both episode and mid-episode resampling.  Mid-episode
        # calls are full-batch and therefore advance the phase/trajectory; reset
        # calls scope only reset rows and leave their newly sampled phase at zero.
        if env_ids is not None:
            return
        cfg = self._task_cfg
        # RoboDuet advances gait_indices even for zero velocity commands;
        # only clock_inputs shown to the policy are frozen at phase 0.25.
        if self.phase is not None:
            self.phase = np.fmod(self.phase + cfg.ctrl_dt * self.gait_frequency, 1.0).astype(
                np.float32
            )
            if cfg.asset.wheel_joint_names:
                self.phase[~self._gait_is_active(self.vel_command_b)] = 0.0
            self._write_feet_phase(slice(None))
        self._update_ee_trajectory()
        self._episode_steps += 1
        self._log_command_state()

    def _log_command_state(self) -> None:
        """Publish the live velocity range so curriculum progress is observable."""
        extras = getattr(self._env, "extras", None)
        log = extras.get("log") if isinstance(extras, dict) else None
        if not isinstance(log, dict):
            return
        low, high = self._task_cfg.command_config.vel_limit
        for axis, label in enumerate("xyz"):
            log[f"command/vel_low_{label}"] = float(low[axis])
            log[f"command/vel_high_{label}"] = float(high[axis])
        log["command/curriculum_expansions"] = self._expansion_count

    @property
    def body_command(self) -> np.ndarray:
        return np.column_stack((self.vel_command_b, self.pitch_roll))

    def policy_command(self) -> np.ndarray:
        return np.column_stack(
            (self.body_command, self.feet_phase, self.curr_ee_goal_cart, self.goal_tool_angles())
        ).astype(np.float32, copy=False)

    def goal_tool_angles(self) -> np.ndarray:
        """Original RoboDuet alpha/beta/gamma projection of the target tool axes."""
        r = np_matrix_from_quat(self.ee_goal_orn_quat)
        return np.column_stack((np.arctan2(r[:, 2, 1], r[:, 1, 1]),
                                np.arctan2(r[:, 0, 2], r[:, 2, 2]),
                                np.arctan2(r[:, 1, 0], r[:, 0, 0])))

    def set_body_plan(self, command: np.ndarray, raw: np.ndarray) -> None:
        self.previous_plan_raw[:] = self.plan_raw
        self.plan_raw[:] = raw
        self.pitch_roll[:] = command

    def ee_yaw_pose(self) -> tuple[np.ndarray, np.ndarray]:
        backend = self._env._backend
        if self._task_cfg.asset.wheel_joint_names:
            # UniSim refreshes tracked bodies to the integrated qpos. Authored
            # site sensors still represent the start of the final substep.
            positions, rotations = backend.get_body_pose_w_rows(self._tcp_rows, self._tcp_body_ids)
            body_quat = rotations[:, 0]
            offset = positions[:, 0] + np_quat_apply_batched(body_quat, self._tcp_offset)
            tcp_quat = np_quat_mul_batched(body_quat, self._tcp_rotation)
        else:
            mount_quat = self._goal_quat_view.read()
            local_quat = self._tcp_quat_view.read()
            offset = self._goal_view.read() + np_quat_apply_batched(mount_quat, self._tcp_pos_view.read())
            tcp_quat = np_quat_mul_batched(mount_quat, local_quat)
        offset[:, :2] -= self._robot_data.root_link_pos_w[:, :2]
        offset[:, 2] -= self.reference_height
        inv_heading = np_quat_conjugate_batched(np_yaw_quat(self._robot_data.root_link_quat_w))
        return (
            np_quat_apply_batched(inv_heading, offset),
            np_quat_mul_batched(inv_heading, tcp_quat),
        )

    def post_compute(self) -> None:
        if self.yaw_frame:
            # Reset events refresh the robot/sensors after CommandManager.reset.
            # Initialize against that refreshed pose before publishing observations.
            reset_ids = np.flatnonzero(self._pending_ee_reset).astype(np.int32)
            if reset_ids.size:
                self._reset_ee_goals(reset_ids)
                self._pending_ee_reset[reset_ids] = False
            stage = self._task_cfg.arm_stage
            if stage.freeze_arm_joints and stage.disable_ee_goal_trajectory:
                # Stage 1 has no EE tracking task. Show the fixed arm's actual TCP.
                position, _ = self.ee_yaw_pose()
                self.curr_ee_goal_cart[:] = position
                self.curr_ee_goal_sphere[:] = _cart2sphere(position)
            yaw = self._robot_data.heading_w
            x, y, z = self.curr_ee_goal_cart.T
            self.curr_ee_goal_world[:, 0] = self._robot_data.root_link_pos_w[:, 0] + np.cos(yaw) * x - np.sin(yaw) * y
            self.curr_ee_goal_world[:, 1] = self._robot_data.root_link_pos_w[:, 1] + np.sin(yaw) * x + np.cos(yaw) * y
            self.curr_ee_goal_world[:, 2] = self.reference_height + z
            self.ee_goal_quat_world[:] = np_quat_mul_batched(
                np_yaw_quat(self._robot_data.root_link_quat_w), self.ee_goal_orn_quat
            )
            return
        pos = np.asarray(self._goal_view.read(), dtype=get_global_dtype())
        quat = self._goal_quat_view.read()
        rotation = np_matrix_from_quat(quat)
        self.curr_ee_goal_world[:] = pos + np.einsum("nij,nj->ni", rotation, self.curr_ee_goal_cart)

    def _command_is_moving(self, commands: np.ndarray) -> np.ndarray:
        moving = np.any(np.abs(np.asarray(commands)[:, :3]) > 0.1, axis=1)
        if self._task_cfg.asset.wheel_joint_names and self.phase is not None:
            moving |= self._gait_is_active(commands)
        return moving

    def _gait_is_active(self, commands: np.ndarray) -> np.ndarray:
        cfg = self._task_cfg.command_config
        thresholds = [cfg.gait_x_threshold, cfg.gait_lateral_threshold, cfg.gait_yaw_threshold]
        return np.any(np.abs(commands[:, :3]) > thresholds, axis=1) & (self.gait_frequency > 0)

    def clock_inputs(self) -> np.ndarray:
        phase = np.where(
            np.linalg.norm(self.vel_command_b, axis=1, keepdims=True) < 0.1,
            0.25, self.feet_phase,
        )
        clock = np.sin(2.0 * np.pi * phase)
        if self._task_cfg.asset.wheel_joint_names:
            clock *= self._gait_is_active(self.vel_command_b)[:, None]
        return clock

    def desired_contact(self, kappa: float) -> np.ndarray:
        if self._task_cfg.wbc_switch.use_unified_observation:
            return _wbc2p_desired_contact(self.feet_phase, self.vel_command_b, kappa)
        return (self.feet_phase < 0.6).astype(get_global_dtype())

    def _write_feet_phase(self, env_ids: np.ndarray | slice) -> None:
        if self.phase is None:
            return
        phase = self.phase[env_ids]
        feet = self.feet_phase[env_ids].copy()
        feet[:, 0] = (phase + 0.5) % 1.0
        feet[:, 3] = (phase + 0.5) % 1.0
        feet[:, 1] = phase
        feet[:, 2] = phase
        self.feet_phase[env_ids] = feet

    def _sample_timing(self, env_ids: np.ndarray) -> None:
        cfg = self._task_cfg.goal_ee
        count = len(env_ids)
        traj = self._env.rng.uniform(*cfg.traj_time_range, size=count)
        hold = self._env.rng.uniform(*cfg.hold_time_range, size=count)
        self._traj_steps[env_ids] = np.maximum(1, np.round(traj / self._ctrl_dt).astype(np.int32))
        self._traj_total_steps[env_ids] = self._traj_steps[env_ids] + np.maximum(
            0, np.round(hold / self._ctrl_dt).astype(np.int32)
        )

    def _ee_underground_limit(self) -> float:
        limit = float(self._task_cfg.goal_ee.underground_limit)
        return max(limit, self.ground_height - self.reference_height) if self.yaw_frame else limit

    def _collision_unsafe(self, starts: np.ndarray, goals: np.ndarray) -> np.ndarray:
        cfg = self._task_cfg.goal_ee
        count = max(2, cfg.num_collision_check_samples)
        fraction = np.linspace(0.0, 1.0, count, dtype=get_global_dtype())
        path = starts[:, None, :] + (goals - starts)[:, None, :] * fraction[None, :, None]
        cartesian = _sphere2cart(path.reshape(-1, 3)).reshape(len(starts), count, 3)
        upper = np.asarray(cfg.collision_upper_limits, dtype=get_global_dtype())
        lower = np.asarray(cfg.collision_lower_limits, dtype=get_global_dtype())
        inside = np.all(cartesian < upper, axis=2) & np.all(cartesian > lower, axis=2)
        underground = np.any(cartesian[:, :, 2] < self._ee_underground_limit(), axis=1)
        return np.any(inside, axis=1) | underground

    def _sample_goal_spheres(self, env_ids: np.ndarray, starts: np.ndarray) -> None:
        cfg = self._task_cfg.goal_ee
        dtype = get_global_dtype()
        candidates = np.broadcast_to(starts, (len(env_ids), 3)).copy()
        remaining = np.arange(len(env_ids), dtype=np.int32)
        for _ in range(max(1, cfg.num_resample_attempts)):
            samples = np.column_stack(
                [
                    self._env.rng.uniform(*cfg.sphere_l_range, size=len(remaining)),
                    self._env.rng.uniform(*cfg.sphere_phi_range, size=len(remaining)),
                    self._env.rng.uniform(*cfg.sphere_theta_range, size=len(remaining)),
                ]
            ).astype(dtype)
            candidates[remaining] = samples
            unsafe = self._collision_unsafe(starts[remaining], samples)
            remaining = remaining[unsafe]
            if remaining.size == 0:
                break
        if self.yaw_frame:
            candidates[remaining] = starts[remaining]
        self._ee_goal_sphere[env_ids] = candidates

    def _sample_goal_orientation(self, env_ids: np.ndarray, *, is_init: bool) -> None:
        cfg = self._task_cfg.goal_ee
        count = len(env_ids)
        base_roll = np.full(count, cfg.default_orn_roll, dtype=get_global_dtype())
        pitch = np.full(count, cfg.arm_induced_pitch, dtype=get_global_dtype())
        yaw = np.zeros(count, dtype=get_global_dtype())
        if is_init:
            roll_delta = self._env.rng.uniform(*cfg.delta_orn_r, size=count)
            pitch_delta = self._env.rng.uniform(*cfg.delta_orn_p, size=count)
            yaw_delta = self._env.rng.uniform(*cfg.delta_orn_y, size=count)
        else:
            roll_delta = self._env.rng.uniform(*cfg.delta_orn_r, size=count)
            pitch_delta = self._env.rng.uniform(*cfg.delta_orn_p, size=count)
            yaw_delta = self._env.rng.uniform(*cfg.delta_orn_y, size=count)
        self.ee_goal_orn_euler[env_ids, 0] = base_roll + roll_delta
        self.ee_goal_orn_euler[env_ids, 1] = pitch + pitch_delta
        self.ee_goal_orn_euler[env_ids, 2] = yaw + yaw_delta
        quat = np_quat_from_euler_xyz(
            self.ee_goal_orn_euler[env_ids, 0],
            self.ee_goal_orn_euler[env_ids, 1],
            self.ee_goal_orn_euler[env_ids, 2],
        )
        self.ee_goal_orn_quat[env_ids] = np.asarray(quat, dtype=get_global_dtype())

    def _reset_ee_goals(self, env_ids: np.ndarray) -> None:
        cfg = self._task_cfg
        stage = cfg.arm_stage
        if self.yaw_frame:
            self._pending_ee_reset[env_ids] = True
            position, _ = self.ee_yaw_pose()
            initial = position[env_ids].copy()
        else:
            initial = np.broadcast_to(
                np.asarray(cfg.goal_ee.init_ee_cart, dtype=get_global_dtype()),
                (len(env_ids), 3),
            ).copy()
        if self._hold_ee_command:
            held = initial if self.yaw_frame else clip_ee_goal_cart(initial[0], cfg.goal_ee)
            sphere = _cart2sphere(np.broadcast_to(held, (len(env_ids), 3)))
            self.curr_ee_goal_cart[env_ids] = held
            self.curr_ee_goal_sphere[env_ids] = sphere
            self._ee_start_sphere[env_ids] = sphere
            self._ee_goal_sphere[env_ids] = sphere
            self._traj_steps[env_ids] = 1
            self._traj_total_steps[env_ids] = np.iinfo(np.int32).max // 4
            self._arm_goal_timer[env_ids] = 0
            self._write_default_orientation(env_ids)
            return
        if stage.disable_ee_goal_trajectory:
            fixed = np.asarray(stage.fixed_ee_goal_cart, dtype=get_global_dtype())[None, :]
            if self.yaw_frame:
                fixed = _cart2sphere(initial)
            self._ee_start_sphere[env_ids] = fixed
            self._ee_goal_sphere[env_ids] = fixed
            self._traj_steps[env_ids] = 1
            self._traj_total_steps[env_ids] = 1
            self._arm_goal_timer[env_ids] = 0
            self.curr_ee_goal_sphere[env_ids] = fixed
            self.curr_ee_goal_cart[env_ids] = _sphere2cart(fixed)
            self._sample_goal_orientation(env_ids, is_init=True)
            return
        init_sphere = _cart2sphere(initial)
        self._ee_start_sphere[env_ids] = init_sphere
        self._sample_goal_spheres(env_ids, np.broadcast_to(init_sphere, (len(env_ids), 3)).copy())
        self._sample_timing(env_ids)
        self._arm_goal_timer[env_ids] = 0
        self.curr_ee_goal_sphere[env_ids] = init_sphere
        self.curr_ee_goal_cart[env_ids] = _sphere2cart(
            np.broadcast_to(init_sphere, (len(env_ids), 3))
        )
        self._sample_goal_orientation(env_ids, is_init=True)

    def _update_ee_trajectory(self) -> None:
        cfg = self._task_cfg
        stage = cfg.arm_stage
        if self._hold_ee_command:
            return
        if stage.disable_ee_goal_trajectory:
            if self.yaw_frame:
                # Replay command schedules use the saved start, with no simulator reads.
                self.curr_ee_goal_sphere[:] = self._ee_start_sphere
                self.curr_ee_goal_cart[:] = _sphere2cart(self._ee_start_sphere)
                return
            fixed = np.asarray(stage.fixed_ee_goal_cart, dtype=get_global_dtype())[None, :]
            self.curr_ee_goal_cart[:] = fixed
            self.curr_ee_goal_sphere[:] = _cart2sphere(fixed)
        else:
            self._arm_goal_timer += 1
            expired = np.flatnonzero(self._arm_goal_timer >= self._traj_total_steps).astype(
                np.int32
            )
            if expired.size:
                self._ee_start_sphere[expired] = self._ee_goal_sphere[expired]
                self._sample_goal_spheres(expired, self._ee_start_sphere[expired])
                self._sample_goal_orientation(expired, is_init=False)
                self._sample_timing(expired)
                self._arm_goal_timer[expired] = 0
            fraction = np.clip(self._arm_goal_timer / self._traj_steps, 0.0, 1.0).astype(
                get_global_dtype()
            )[:, None]
            self.curr_ee_goal_sphere[:] = (
                self._ee_start_sphere + (self._ee_goal_sphere - self._ee_start_sphere) * fraction
            )
            self.curr_ee_goal_cart[:] = _sphere2cart(self.curr_ee_goal_sphere)


class Go2ArmObservation(ManagerTermBase):
    def __init__(self, cfg, env: "ManagerBasedRlEnv"):
        super().__init__(env)
        self._task_cfg = _task_cfg(env)
        names = (
            self._task_cfg.sensor.local_linvel,
            self._task_cfg.sensor.gyro,
            "upvector",
            self._task_cfg.sensor.ee_local_pos,
            self._task_cfg.sensor.ee_local_quat,
        )
        self._views = env.scene.bind_sensor_data(names)
        if self._views.dimensions != (3, 3, 3, 3, 4):
            raise ValueError(f"Unexpected Go2Arm sensor dimensions: {self._views.dimensions}")
        self._entity: Entity = env.scene["robot"]

    def __call__(self, env: "ManagerBasedRlEnv", actor: bool) -> np.ndarray:
        del env
        raw = self._raw_observation(actor)
        use_unified = bool(getattr(self._task_cfg.wbc_switch, "use_unified_observation", False))
        if actor and use_unified:
            return raw
        return raw[:, 3:] if actor else raw

    def _raw_observation(self, actor: bool) -> np.ndarray:
        cfg = self._task_cfg
        sensors = np.asarray(self._views.read(), dtype=get_global_dtype())
        linvel = sensors[:, 0:3]
        gyro = sensors[:, 3:6]
        gravity = sensors[:, 6:9]
        ee_pos = sensors[:, 9:12]
        ee_quat = sensors[:, 12:16]
        joint_pos = self._entity.data.joint_pos - self._entity.data.default_joint_pos
        joint_vel = self._entity.data.joint_vel
        command = _command(self._env)
        if actor:
            noise = cfg.noise_config
            level = noise.level
            if level > 0.0:

                def add(value: np.ndarray, scale: float) -> np.ndarray:
                    sample = self._env.rng.uniform(-1.0, 1.0, value.shape)
                    return value + sample.astype(get_global_dtype()) * level * scale

                linvel = add(linvel, noise.scale_linvel)
                gyro = add(gyro, noise.scale_gyro)
                gravity = add(gravity, noise.scale_gravity)
                joint_pos = add(joint_pos, noise.scale_joint_angle)
                joint_vel = add(joint_vel, noise.scale_joint_vel)
                ee_pos = add(ee_pos, noise.scale_ee_pos)
        use_unified = bool(getattr(self._task_cfg.wbc_switch, "use_unified_observation", False))
        if not use_unified:
            goal = command.curr_ee_goal_cart
            return np.concatenate(
                [
                    linvel,
                    gyro,
                    -gravity,
                    command.vel_command_b,
                    command.feet_phase,
                    joint_pos,
                    joint_vel,
                    ee_pos,
                    goal,
                    ee_pos - goal,
                    self._env.action_manager.action,
                ],
                axis=1,
                dtype=get_global_dtype(),
            )
        # Reference VelocityTrackingEasyEnv.compute_observations() builds the
        # unified 63-D frame in this order.  The clock is the four foot-wise
        # sine phases; a stationary command maps every foot to phase 0.25.
        arm_goal = np.concatenate(
            [
                _cart2sphere(command.curr_ee_goal_cart),
                command.goal_tool_angles() if command.yaw_frame else command.ee_goal_orn_euler,
            ],
            axis=1,
        )
        if self._task_cfg.arm_stage.freeze_arm_joints:
            arm_goal = np.zeros_like(arm_goal)
        roll = np.arctan2(gravity[:, 1:2], gravity[:, 2:3])
        pitch = np.arctan2(
            -gravity[:, 0:1], np.linalg.norm(gravity[:, 1:3], axis=1, keepdims=True)
        )
        clock = command.feet_phase
        if command.phase is not None:
            clock = command.clock_inputs()
        base = np.concatenate(
            [
                joint_pos,
                joint_vel[:, :12] * 0.05,
                self._env.action_manager.action,
                command.vel_command_b * np.array([2.0, 2.0, 0.25], dtype=get_global_dtype()),
                arm_goal,
                roll,
                pitch,
                clock,
            ],
            axis=1,
            dtype=get_global_dtype(),
        )
        if actor:
            return base
        # The reference privileged vector starts with normalized friction and
        # restitution, followed by EE l/p/y and orientation quaternion.
        # Material randomization scalars are not exposed by the current MJCF.
        if command.yaw_frame:
            ee_pos, ee_quat = command.ee_yaw_pose()
        privileged = np.concatenate(
            [
                np.zeros((ee_pos.shape[0], 2), dtype=get_global_dtype()),
                _cart2sphere(ee_pos),
                ee_quat,
            ],
            axis=1,
            dtype=get_global_dtype(),
        )
        return np.concatenate([base, privileged], axis=1, dtype=get_global_dtype())


class ManipLocoReward(ManagerTermBase):
    """Shared MuJoCo rewards for the manipulation and locomotion tasks."""

    def __init__(self, cfg, env: "ManagerBasedRlEnv"):
        super().__init__(env)
        self._name = str(cfg.params["name"])
        self._task_cfg = _task_cfg(env)
        self._arm = slice(-6, None)
        # Go2 actuator legs are FL, FR, RL, RR. L1-W uses FL, FR, RR, RL.
        self._mirror_rear_right, self._mirror_rear_left = (
            (slice(6, 9), slice(9, 12))
            if self._task_cfg.asset.wheel_joint_names
            else (slice(9, 12), slice(6, 9))
        )
        self._leg_collision_groups = (
            self._task_cfg.sensor.leg_collision_groups or _LEG_COLLISION_SENSOR_GROUPS
        )
        self._feet_force_names = tuple(self._task_cfg.sensor.feet_force)
        self._feet_pos_names = tuple(self._task_cfg.sensor.feet_pos)
        self._feet_vel_names = tuple(
            name.replace("_pos", "_global_linvel") for name in self._feet_pos_names
        )
        sensor_names = [
            self._task_cfg.sensor.local_linvel,
            self._task_cfg.sensor.gyro,
            _UPVECTOR_SENSOR,
            *self._feet_force_names,
            *self._feet_pos_names,
            *self._feet_vel_names,
            *_ARM_TOUCH_SENSORS,
            *(name for group in self._leg_collision_groups for name in group),
        ]
        self._views = env.scene.bind_sensor_data(tuple(sensor_names))
        # Slice by the bound view's per-sensor widths. Foot-contact and arm-touch
        # sensors are 1-wide in the MJCF; fixed 3-wide offsets would silently
        # repoint the gait/contact rewards at neighbouring channels.
        slices: dict[str, slice] = {}
        offset = 0
        for name, width in zip(self._views.names, self._views.dimensions, strict=True):
            slices[name] = slice(offset, offset + width)
            offset += width
        self._slices = slices
        for name in self._feet_pos_names:
            width = self._slices[name].stop - self._slices[name].start
            if width != 3:
                raise ValueError(
                    f"Go2Arm reward expects a 3-wide foot position sensor '{name}', "
                    f"got width {width}"
                )
        self._ee_view = env.scene.bind_sensor_data((self._task_cfg.sensor.ee_local_pos,))
        self._ee_quat_view = env.scene.bind_sensor_data((self._task_cfg.sensor.ee_local_quat,))
        self._ee_relative_view = env.scene.bind_sensor_data(
            (self._task_cfg.sensor.ee_relative_pos,)
        )
        self._entity: Entity = env.scene["robot"]
        self._torque_view = None
        # The reward manager skips zero-weight terms, including effort terms.
        if self._name in {"torques", "energy", "loco_energy", "arm_energy"} and float(cfg.weight) != 0.0:
            self._torque_view = env.scene.bind_sensor_data(
                tuple(f"{joint_name}_torque" for joint_name in self._entity.joint_names)
            )
            if self._torque_view.dimensions != (1,) * len(self._entity.joint_names):
                raise ValueError("Unexpected Go2Arm actuator-force sensor dimensions")
        self._previous_joint_vel = np.array(self._entity.data.joint_vel, copy=True)
        self._last_acceleration = np.zeros_like(self._previous_joint_vel)
        self._last_contacts = np.zeros((env.num_envs, len(self._feet_force_names)), dtype=bool)

    def reset(self, env_ids: np.ndarray | slice | None) -> None:
        """Restart finite-difference acceleration after the selected envs reset."""
        current = np.asarray(self._entity.data.joint_vel)
        if env_ids is None:
            env_ids = slice(None)
        self._previous_joint_vel[env_ids] = current[env_ids]
        self._last_contacts[env_ids] = False

    def _wm_reward(self, env, command, name, cfg):
        if name not in {
            "orientation_control", "orientation_heuristic", "arm_manip_commands_tracking_combine",
            "arm_control_limits", "arm_control_smoothness_1", "arm_dof_acc",
            "object_distance", "object_distance_l2",
        }:
            return None
        if name in {"orientation_control", "orientation_heuristic"}:
            # Row Z of the body-to-world rotation is world up in body axes.
            up = np_matrix_from_quat(self._entity.data.root_link_quat_w)[:, 2]
            if name == "orientation_control":
                pitch, roll = command.pitch_roll.T
                desired_xy = np.column_stack((np.sin(pitch) * np.cos(roll), -np.sin(roll)))
                return np.square(up[:, :2] - desired_xy).sum(-1)
            pitch = np.arctan2(-up[:, 0], np.hypot(up[:, 1], up[:, 2]))
            dz = command.curr_ee_goal_cart[:, 2] + command.reference_height - self._entity.data.root_link_pos_w[:, 2]
            return np.where(dz < -.1, (pitch - .4) ** 2, np.where(dz > .4, (pitch + .3) ** 2, 0.))
        if name == "arm_control_limits":
            return np.maximum(np.abs(command.plan_raw) - .4, 0.).sum(axis=-1)
        if name == "arm_control_smoothness_1":
            return (np.square(command.plan_raw - command.previous_plan_raw) *
                    (command.previous_plan_raw != 0)).sum(axis=-1)
        if name == "arm_dof_acc":
            vel = self._entity.data.joint_vel
            acc = (vel[:, self._arm] - self._previous_joint_vel[:, self._arm]) / env.step_dt
            self._previous_joint_vel[:] = vel
            return np.square(acc).sum(axis=-1)
        pos, quat = command.ee_yaw_pose()
        goal = command.curr_ee_goal_cart
        if name == "arm_manip_commands_tracking_combine":
            lpy = _cart2sphere(pos) - _cart2sphere(goal)
            lpy[:, 1:] = np.arctan2(np.sin(lpy[:, 1:]), np.cos(lpy[:, 1:]))
            r = np_matrix_from_quat(quat)
            angles = np.column_stack((np.arctan2(r[:, 2, 1], r[:, 1, 1]),
                                      np.arctan2(r[:, 0, 2], r[:, 2, 2]),
                                      np.arctan2(r[:, 1, 0], r[:, 0, 0])))
            angles -= command.goal_tool_angles()
            angles = np.arctan2(np.sin(angles), np.cos(angles))
            lpy_err = (np.abs(lpy) / cfg.manip_lpy_range).sum(-1)
            angle_err = (np.abs(angles) / cfg.manip_rpy_range).sum(-1)
            return np.exp(-cfg.manip_weight_lpy * lpy_err - cfg.manip_weight_rpy * angle_err)
        distance = np.square(pos - goal).sum(-1)
        return np.exp(-distance / cfg.object_sigma) if name == "object_distance" else distance

    def _read_joint_torques(self) -> np.ndarray:
        if self._torque_view is None:
            raise RuntimeError("Go2Arm torque sensors were not initialized")
        return np.asarray(self._torque_view.read())

    def __call__(self, env: "ManagerBasedRlEnv", name: str | None = None) -> np.ndarray:
        if name is not None and name != self._name:
            raise ValueError("Go2Arm reward term was bound to another reward name")
        cfg = self._task_cfg.reward_parameters
        command = _command(env)
        if command.yaw_frame:
            revised = self._wm_reward(env, command, name, cfg)
            if revised is not None:
                revised = np.asarray(revised, dtype=get_global_dtype())
                if not np.isfinite(revised).all():
                    raise FloatingPointError(f"2PWBC reward '{name}' produced NaN/Inf")
                return revised
        sensors = np.asarray(self._views.read(), dtype=get_global_dtype())
        linvel = sensors[:, self._slices[self._task_cfg.sensor.local_linvel]]
        gyro = sensors[:, self._slices[self._task_cfg.sensor.gyro]]
        gravity = sensors[:, self._slices[_UPVECTOR_SENSOR]]
        feet_force = np.stack(
            [sensors[:, self._slices[name]] for name in self._feet_force_names], axis=1
        )
        feet_pos = np.stack(
            [sensors[:, self._slices[name]] for name in self._feet_pos_names], axis=1
        )
        feet_vel = np.stack(
            [sensors[:, self._slices[name]] for name in self._feet_vel_names], axis=1
        )
        contacts = np.stack([sensors[:, self._slices[name]] for name in _ARM_TOUCH_SENSORS], axis=1)
        ee_pos = np.asarray(self._ee_view.read(), dtype=get_global_dtype())
        ee_relative_pos = np.asarray(self._ee_relative_view.read(), dtype=get_global_dtype())
        joint_pos = self._entity.data.joint_pos
        joint_vel = self._entity.data.joint_vel
        base_height = self._entity.data.root_link_pos_w[:, 2]
        cmd = command.vel_command_b
        defaults = self._entity.data.default_joint_pos
        if name == "tracking_lin_vel":
            value = np.exp(
                -np.sum(np.square(cmd[:, :2] - linvel[:, :2]), axis=1) / cfg.tracking_sigma
            )
            command.record_tracking(value)
        elif name == "tracking_ang_vel":
            value = np.exp(-np.square(cmd[:, 2] - gyro[:, 2]) / cfg.tracking_sigma)
        elif name == "lin_vel_z":
            value = np.square(linvel[:, 2])
        elif name == "ang_vel_xy":
            value = np.sum(np.square(gyro[:, :2]), axis=1)
        elif name == "roll":
            value = np.square(gravity[:, 0])
        elif name == "base_height":
            value = np.square(base_height - cfg.base_height_target)
        elif name == "similar_to_default":
            value = np.sum(np.abs(joint_pos - defaults), axis=1)
        elif name == "leg_pose":
            weights = np.array([1.0, 1.0, 0.1] * 4, dtype=get_global_dtype())
            value = np.sum(weights * np.square(joint_pos[:, :12] - defaults[:, :12]), axis=1)
        elif name == "mirrorup":
            error = 0.5 * (
                np.square(joint_pos[:, :3] - joint_pos[:, self._mirror_rear_right]).sum(axis=1)
                + np.square(joint_pos[:, 3:6] - joint_pos[:, self._mirror_rear_left]).sum(axis=1)
            )
            value = error * np.clip(gravity[:, 2], 0.0, 1.0)
        elif name == "dof_pos_limits":
            if self._task_cfg.wbc_switch.use_unified_observation:
                limits = np.asarray(self._entity.data.soft_joint_pos_limits)
                midpoint = np.mean(limits, axis=1)
                half_range = 0.5 * (limits[:, 1] - limits[:, 0]) * cfg.soft_dof_pos_limit
                lower = midpoint - half_range
                upper = midpoint + half_range
                # The Airbot home pose differs from RoboDuet's arm URDF: its
                # joint3=0 falls just outside a 90% soft range. A frozen arm
                # cannot remove that penalty, so train Stage-1 on leg limits
                # and retain the home pose inside Stage-2's soft interval.
                active_dofs = np.arange(12)
                if not self._task_cfg.arm_stage.freeze_arm_joints:
                    active_dofs = np.r_[active_dofs, joint_pos.shape[1]-6:joint_pos.shape[1]]
                lower = np.minimum(lower[active_dofs], defaults[:, active_dofs])
                upper = np.maximum(upper[active_dofs], defaults[:, active_dofs])
                active_pos = joint_pos[:, active_dofs]
                value = np.sum(
                    np.maximum(lower - active_pos, 0.0)
                    + np.maximum(active_pos - upper, 0.0),
                    axis=1,
                )
            elif not cfg.leg_dof_upper_limits or not cfg.leg_dof_lower_limits:
                value = np.zeros(env.num_envs, dtype=get_global_dtype())
            else:
                upper = np.asarray(cfg.leg_dof_upper_limits, dtype=get_global_dtype())
                lower = np.asarray(cfg.leg_dof_lower_limits, dtype=get_global_dtype())
                margin = cfg.dof_pos_limit_margin
                leg_pos = joint_pos[:, :12]
                over = np.square(np.maximum(leg_pos - upper + margin, 0.0))
                under = np.square(np.maximum(lower + margin - leg_pos, 0.0))
                value = np.sum(over + under, axis=1)
        elif name == "action_rate":
            joint_slice = slice(0, 12) if self._task_cfg.wbc_switch.use_unified_observation else slice(None)
            value = np.sum(
                np.square(
                    env.action_manager.action[:, joint_slice]
                    - env.action_manager.prev_action[:, joint_slice]
                ),
                axis=1,
            )
        elif name == "stand_action_rate":
            # Unlike position-to-default stand_still, directly discourages
            # high-frequency leg commands without locking the stance pose.
            stationary = ~command._command_is_moving(cmd)
            delta = env.action_manager.action[:, :12] - env.action_manager.prev_action[:, :12]
            value = stationary.astype(get_global_dtype()) * np.sum(np.square(delta), axis=1)
        elif name in {"action_smoothness_1", "action_smoothness_2"}:
            action_term = env.action_manager.get_term("joint_pos")
            current_target = action_term._unclipped_target[:, :12]
            previous_target = action_term._previous_unclipped_target[:, :12]
            previous_action = env.action_manager.prev_action[:, :12]
            if name == "action_smoothness_1":
                delta = current_target - previous_target
                mask = previous_action != 0.0
            else:
                delta = (
                    current_target
                    - 2.0 * previous_target
                    + action_term._last_last_unclipped_target[:, :12]
                )
                mask = (previous_action != 0.0) & (env.action_manager.prev_prev_action[:, :12] != 0.0)
            value = np.sum(np.square(delta) * mask, axis=1)
        elif name == "torques":
            value = np.sum(np.square(self._read_joint_torques()), axis=1)
        elif name in {"energy", "loco_energy"}:
            torques = self._read_joint_torques()
            sl = slice(None) if name == "energy" else slice(0, 12)
            value = np.sum(np.square(torques[:, sl] * joint_vel[:, sl]), axis=1)
        elif name == "dof_vel":
            value = np.sum(np.square(joint_vel[:, :12]), axis=1)
        elif name == "dof_acc":
            acceleration = (joint_vel - self._previous_joint_vel) / env.step_dt
            self._previous_joint_vel[:] = joint_vel
            self._last_acceleration[:] = acceleration
            value = np.sum(np.square(acceleration[:, :12]), axis=1)
        elif name == "feet_slip":
            feet_vel = self._foot_contact_velocities(feet_vel, joint_vel)
            contact = np.linalg.norm(feet_force, axis=2) > 1.0
            contact_filter = contact | self._last_contacts
            self._last_contacts[:] = contact
            value = np.sum(contact_filter * np.sum(np.square(feet_vel[:, :, :2]), axis=2), axis=1)
        elif name == "collision":
            # The reference penalizes contacts on thigh and calf bodies,
            # counting each body once even when both calf geoms touch.
            value = np.zeros(env.num_envs, dtype=get_global_dtype())
            for thigh, *calf in self._leg_collision_groups:
                thigh_contact = sensors[:, self._slices[thigh]][:, 0] > 0.0
                calf_contact = np.column_stack([sensors[:, self._slices[name]] for name in calf])
                value += thigh_contact.astype(get_global_dtype())
                value += (calf_contact > 0).any(axis=-1).astype(get_global_dtype())
        elif name == "hip_action_l2":
            action = np.asarray(env.action_manager.action)
            value = np.sum(np.square(action[:, [0, 3, 6, 9]]), axis=1)
        elif name == "arm_manip_commands_tracking_combine":
            # Match RoboDuet's l/p/y + r/p/y tracking reward rather than
            # replacing it with an unscaled Cartesian distance.  The relative
            # endpoint sensor is expressed from the arm base on the Go2,
            # equivalent to the legacy grasper position in base coordinates.
            actual_lpy = _cart2sphere(ee_relative_pos)
            target_lpy = _cart2sphere(command.curr_ee_goal_cart)
            lpy_range = np.asarray(cfg.manip_lpy_range, dtype=get_global_dtype())
            lpy_error = np.sum(np.abs(actual_lpy - target_lpy) / lpy_range, axis=1)
            actual_quat = self._ee_quat_view.read()
            orn_error = np_quat_orientation_error_local(command.ee_goal_orn_quat, actual_quat)
            rpy_range = np.asarray(cfg.manip_rpy_range, dtype=get_global_dtype())
            rpy_error = np.sum(np.abs(orn_error) / rpy_range, axis=1)
            value = np.exp(
                -(
                    float(cfg.manip_weight_lpy) * lpy_error
                    + float(cfg.manip_weight_rpy) * rpy_error
                )
            )
        elif name in {"orientation", "orientation_heuristic", "orientation_control"}:
            # The Go2Arm command interface has no body pitch/roll command
            # channel; its projected-gravity term is represented by roll.
            value = np.sum(np.square(gravity[:, :2]), axis=1)
        elif name == "arm_energy":
            torques = self._read_joint_torques()[:, self._arm]
            value = np.sum(np.square(torques * joint_vel[:, self._arm]), axis=1)
        elif name == "arm_dof_vel":
            value = np.sum(np.square(joint_vel[:, self._arm]), axis=1)
        elif name == "arm_dof_acc":
            value = np.sum(np.square(self._last_acceleration[:, self._arm]), axis=1)
        elif name == "arm_action_rate":
            action = np.asarray(env.action_manager.action)
            previous = np.asarray(env.action_manager.prev_action)
            value = np.sum(np.square(previous[:, self._arm] - action[:, self._arm]), axis=1)
        elif name == "arm_action_smoothness_1":
            action_term = env.action_manager.get_term("joint_pos")
            value = np.sum(
                np.square(
                    action_term._target[:, self._arm]
                    - action_term._previous_target[:, self._arm]
                ),
                axis=1,
            )
            value *= (np.any(np.abs(action_term._previous_target[:, self._arm]) > 0.0, axis=1)).astype(
                get_global_dtype()
            )
        elif name == "arm_control_smoothness_1":
            # The reference term penalizes the absent four-dimensional
            # ``plan_actions`` channel (body attitude/velocity planner). The
            # raw-joint UniLab profile has no such channel; evaluating this
            # term on arm actions would be a different reward.
            value = np.zeros(env.num_envs, dtype=get_global_dtype())
        elif name == "arm_control_limits":
            # The reference RoboDuet term constrains ``plan_actions`` (two
            # body pitch/roll planner outputs), not the six raw arm joint
            # actions.  Go2ArmManipLoco has no plan-action channel; applying
            # that penalty to joint actions is a semantic mismatch and makes
            # a Stage-2 arm head receive a spurious large reward gradient.
            value = np.zeros(env.num_envs, dtype=get_global_dtype())
        elif name == "stand_still":
            still = (~command._command_is_moving(cmd)).astype(get_global_dtype())
            value = still * np.sum(np.abs(joint_pos[:, :12] - defaults[:, :12]), axis=1)
        elif name == "foot_drag":
            contact = np.linalg.norm(feet_force, axis=2) < 0.5
            height_error = np.clip(cfg.target_foot_height / 2.0 - feet_pos[:, :, 2], 0.0, None)
            value = np.sum(np.square(height_error) * contact, axis=1)
        elif name == "object_distance":
            value = np.exp(
                -np.sum(np.square(ee_pos - command.curr_ee_goal_cart), axis=1) / cfg.object_sigma
            )
        elif name == "object_distance_l2":
            value = np.sum(np.square(ee_pos - command.curr_ee_goal_cart), axis=1)
        elif name == "arm_collision":
            value = np.sum(contacts[:, :, 0], axis=1)
        else:
            value = self._robot_reward(env, name, command, feet_force, feet_pos, feet_vel)
        value = np.asarray(value, dtype=get_global_dtype())
        if not np.isfinite(value).all():
            raise FloatingPointError(f"2PWBC reward '{name or self._name}' produced NaN/Inf")
        return value

    def _foot_contact_velocities(self, feet_vel, joint_vel):
        return feet_vel

    def _robot_reward(self, env, name, command, feet_force, feet_pos, feet_vel):
        raise KeyError(f"Unknown 2PWBC reward term '{name or self._name}'")


class Go2ArmReward(ManipLocoReward):
    """MuJoCo reward terms for Go2's gait and foot placement."""

    def _robot_reward(self, env, name, command, feet_force, feet_pos, feet_vel):
        cfg = self._task_cfg.reward_parameters
        cmd = command.vel_command_b
        if name == "feet_air_time":
            swing = command.feet_phase >= 0.6
            contact = np.linalg.norm(feet_force, axis=2) > 0.1
            value = np.sum(swing * (~contact), axis=1) / len(self._feet_force_names)
        elif name == "tracking_contacts_shaped_force":
            # RoboDuet/WTW contact shaping: penalize force while a foot is
            # expected to be in swing.  The MuJoCo sensor exposes a force
            # vector per foot, so use its norm as in the reference.
            force_norm = np.linalg.norm(feet_force, axis=2)
            desired = command.desired_contact(cfg.kappa_gait_probs)
            value = -np.sum(
                (1.0 - desired)
                * (1.0 - np.exp(-np.square(force_norm) / cfg.gait_force_sigma)),
                axis=1,
            ) / len(self._feet_force_names)
        elif name == "tracking_contacts_shaped_vel":
            speed = np.linalg.norm(feet_vel, axis=2)
            desired = command.desired_contact(cfg.kappa_gait_probs)
            value = -np.sum(
                desired * (1.0 - np.exp(-np.square(speed) / cfg.gait_vel_sigma)), axis=1
            ) / len(self._feet_pos_names)
        elif name == "feet_clearance_cmd_linear":
            phase = np.asarray(command.feet_phase, dtype=get_global_dtype())
            swing_phase = 1.0 - np.abs(1.0 - np.clip(phase * 2.0 - 1.0, 0.0, 1.0) * 2.0)
            target_height = 0.04 * swing_phase + 0.02 + self._task_cfg.asset.wheel_radius
            swing = 1.0 - command.desired_contact(cfg.kappa_gait_probs)
            value = np.sum(
                np.square(target_height - feet_pos[:, :, 2]) * swing, axis=1
            )
        elif name == "raibert_heuristic":
            root = np.asarray(self._entity.data.root_link_pos_w)
            yaw = np.asarray(self._entity.data.heading_w)
            translated = feet_pos - root[:, None, :]
            cosine = np.cos(yaw)[:, None]
            sine = np.sin(yaw)[:, None]
            foot_x = cosine * translated[:, :, 0] + sine * translated[:, :, 1]
            foot_y = -sine * translated[:, :, 0] + cosine * translated[:, :, 1]
            phase = np.abs(1.0 - 2.0 * command.feet_phase) - 0.5
            forward_offset = phase * cmd[:, 0:1] * (0.5 / 3.0)
            yaw_offset = phase * (cmd[:, 2:3] * 0.225) * (0.5 / 3.0)
            yaw_offset[:, 2:4] *= -1.0
            nominal = np.asarray(self._task_cfg.asset.raibert_centers)
            desired_x = nominal[:, 0] + forward_offset
            desired_y = nominal[:, 1] + yaw_offset
            value = np.sum(np.square(foot_x - desired_x) + np.square(foot_y - desired_y), axis=1)
        elif name == "swing_feet_z":
            swing = command.feet_phase >= 0.6
            error = np.square(feet_pos[:, :, 2] - cfg.target_foot_height)
            value = np.sum(np.exp(-error / 0.01) * swing, axis=1) / len(self._feet_pos_names)
        elif name == "contact":
            contact = np.linalg.norm(feet_force, axis=2) > 0.1
            stance = (command.feet_phase < 0.6) | (command.gait_frequency < 1.0e-8)
            value = np.sum(contact == stance, axis=1) / len(self._feet_force_names)
        else:
            return super()._robot_reward(env, name, command, feet_force, feet_pos, feet_vel)
        return value


class Go2ArmFallen(ManagerTermBase):
    def __init__(self, cfg, env: "ManagerBasedRlEnv"):
        super().__init__(env)
        self._view = env.scene.bind_sensor_data(("upvector",))
        self._entity: Entity = env.scene["robot"]
        self._height_limit = float(_task_cfg(env).reward_parameters.terminal_body_height)

    def __call__(self, env: "ManagerBasedRlEnv") -> np.ndarray:
        del env
        up = np.asarray(self._view.read()[:, 2], dtype=get_global_dtype())
        base_height = np.asarray(self._entity.data.root_link_pos_w[:, 2], dtype=get_global_dtype())
        return (up <= 0.5) | (base_height <= self._height_limit)


class Go2ArmManagerBasedRlEnv(ManagerBasedRlEnv):
    """MBA runtime with a visual playback oracle for MuJoCo.

    MuJoCo compiles the training model with visual meshes discarded.  UniSim's
    per-env playback path therefore receives that physics-only model unless a
    task supplies its own oracle.  When this task has no fixed model variants,
    the original scene MJCF has exactly the same dynamics topology and is the
    correct visual oracle.
    """

    def init_state(self):
        """Expose the live velocity command to UniLab's interactive viewer.

        The generic viewer predates Manager-Based command terms and reads the
        command from ``env.state.info['commands']``.  Keep that compatibility
        view pointed at the task-owned command array; no copy is made, so
        keyboard writes affect the observation used by the policy immediately.
        """
        state = super().init_state()
        state.info["commands"] = _command(self).vel_command_b
        return state

    def step(self, actions: np.ndarray):
        """Attach the leg and arm reward channels to the UniLab state info."""
        clip = float(self._cfg.control_config.clip_actions)
        if np.isfinite(clip) and clip > 0.0:
            # Matches the reference LeggedRobot.step() normalization.clip_actions
            # contract.  The clipped tensor is what ActionManager stores in
            # action history and what action-rate rewards observe.
            actions = np.clip(actions, -clip, clip)
        state = super().step(actions)
        manager = self.reward_manager
        # These are hybrid-only terms in the reference RoboDuet config.  They
        # are zero in Stage-1's pretrained reward scales and are introduced
        # during the Stage-2 switch.  Previously they were always added to the
        # locomotion channel, so Stage-1 learned a low crouched posture while
        # optimizing arm/manipulation penalties that should not exist yet.
        hybrid_terms = {
            "arm_manip_commands_tracking_combine",
            "hip_action_l2",
            "orientation_heuristic",
            "arm_collision",
            "arm_control_smoothness_1",
            "arm_control_limits",
            "arm_energy",
            "arm_dof_vel",
            "arm_dof_acc",
            "arm_action_rate",
            "arm_action_smoothness_1",
        }
        # RoboDuet's arm reward receives every term except the two base
        # velocity-tracking terms, matching the reference implementation.
        arm_excluded = {"tracking_lin_vel", "tracking_ang_vel", "wheel_roll"}
        switch = getattr(self._cfg, "wbc_switch", None)
        start = int(getattr(switch, "start_steps", 0))
        duration = max(1, int(getattr(switch, "duration_steps", 2400)))
        progress = (float(self.common_step_counter) - start) / float(duration)
        # Keep the reward-scale mix and cross-task coupling distinct.
        # Reference GlobalSwitch uses full hybrid reward scales but caps the
        # cross-head advantage coefficient at 0.5.
        frozen = bool(self._cfg.arm_stage.freeze_arm_joints)
        # A frozen-arm run is RoboDuet Stage-1 locomotion pretraining.  The
        # reference switches *both* the hybrid reward scales and the arm head
        # at the stage boundary; merely setting beta=0 still leaked all
        # manipulation penalties into the locomotion channel and produced the
        # crouched Stage-1 checkpoints seen previously.
        hybrid_mix = 0.0 if frozen else float(np.clip(progress, 0.0, 1.0))
        beta = 0.5 * hybrid_mix
        if frozen:
            beta = 0.0
        leg = np.zeros(self.num_envs, dtype=get_global_dtype())
        arm = np.zeros(self.num_envs, dtype=get_global_dtype())
        leg_positive = np.zeros(self.num_envs, dtype=get_global_dtype())
        leg_negative = np.zeros(self.num_envs, dtype=get_global_dtype())
        arm_positive = np.zeros(self.num_envs, dtype=get_global_dtype())
        arm_negative = np.zeros(self.num_envs, dtype=get_global_dtype())
        for idx, name in enumerate(manager.active_terms):
            weighted = manager._step_reward[:, idx] * self.step_dt
            if name == "tracking_lin_vel" and not frozen:
                ratio = float(getattr(switch, "hybrid_tracking_lin_vel_ratio", 0.7))
                weighted = weighted * (1.0 + hybrid_mix * (ratio - 1.0))
            elif name == "tracking_ang_vel" and not frozen:
                ratio = float(getattr(switch, "hybrid_tracking_ang_vel_ratio", 0.5))
                weighted = weighted * (1.0 + hybrid_mix * (ratio - 1.0))
            elif name in self._cfg.stage2_disabled_rewards and not frozen and switch.world_model_control:
                weighted = weighted * (1.0 - hybrid_mix)
            elif name == "orientation_control":
                # WTW pretraining uses -5; the hybrid profile uses -10.
                weighted = weighted * (0.5 + 0.5 * hybrid_mix)
            leg_term = weighted if name not in hybrid_terms else weighted * hybrid_mix
            leg += leg_term
            leg_positive += np.maximum(leg_term, 0.0)
            leg_negative += np.minimum(leg_term, 0.0)
            if name not in arm_excluded:
                arm_term = weighted if name not in hybrid_terms else weighted * hybrid_mix
                arm += arm_term
                arm_positive += np.maximum(arm_term, 0.0)
                arm_negative += np.minimum(arm_term, 0.0)
        reward_cfg = self._cfg.reward_parameters
        if bool(getattr(reward_cfg, "only_positive_rewards_ji22_style", False)):
            sigma = max(float(getattr(reward_cfg, "sigma_rew_neg", 0.02)), 1e-8)
            # Exact legged-gym Ji22 aggregation used by RoboDuet:
            # positive_mass * exp(negative_mass / sigma).  Applying it after
            # the stage mix keeps the two channels numerically consistent with
            # the reference while preserving per-term diagnostics.
            leg = leg_positive * np.exp(np.clip(leg_negative / sigma, -80.0, 0.0))
            arm = arm_positive * np.exp(np.clip(arm_negative / sigma, -80.0, 0.0))
            state.info["log"]["wbc2p/leg_penalty_factor"] = float(
                np.mean(np.exp(np.clip(leg_negative / sigma, -80.0, 0.0)))
            )
        if frozen and switch.world_model_control:
            arm.fill(0.0)
        state.info["log"]["wbc2p/leg_positive"] = float(np.mean(leg_positive))
        state.info["log"]["wbc2p/leg_negative"] = float(np.mean(leg_negative))
        state.info["log"]["wbc2p/reward_raw"] = float(np.mean(state.reward))
        state.info["log"]["wbc2p/reward_leg"] = float(np.mean(leg))
        state.info["log"]["wbc2p/reward_arm"] = float(np.mean(arm))
        state.info["wbc2p_reward"] = np.stack((leg, arm), axis=1)
        action_abs = np.max(np.abs(self.action_manager.action), axis=1)
        action_leg_abs = np.max(np.abs(self.action_manager.action[:, :12]), axis=1)
        action_arm_abs = np.max(np.abs(self.action_manager.action[:, -6:]), axis=1)
        state.info["log"]["wbc2p/action_abs_max"] = float(np.max(action_abs))
        state.info["log"]["wbc2p/action_leg_abs_max"] = float(np.max(action_leg_abs))
        state.info["log"]["wbc2p/action_arm_abs_max"] = float(np.max(action_arm_abs))
        # ``action_manager.action`` is the raw policy output by UniLab
        # contract.  The action term's target is logged separately so a
        # diagnostic run can distinguish policy/reward growth from actuator
        # target growth without changing either one.
        action_term = self.action_manager.get_term("joint_pos")
        state.info["log"]["wbc2p/leg_target_clip_fraction"] = float(
            np.mean(action_term._leg_target_clip_fraction)
        )
        target = getattr(action_term, "_target", None)
        if target is not None:
            state.info["log"]["wbc2p/target_abs_max"] = float(np.max(np.abs(target)))
        state.info["wbc2p_arm_enabled"] = np.asarray(not frozen, dtype=np.bool_)
        # The environment reports the stage coefficient for diagnostics.
        # The learner couples its imagined rewards with cooperation.beta.
        state.info["wbc2p_hybrid_beta"] = np.asarray(
            beta, dtype=get_global_dtype()
        )
        if self._cfg.wbc_switch.use_unified_observation:
            # Expose the transformed leg reward as the public scalar.
            state = state.replace(reward=leg)
        return state

    def export_world_model_state(self) -> np.ndarray:
        """Export root pose/world twist and joint state in actuator order."""
        data = self.scene["robot"].data
        with self.scene._scoped_state_reads():
            return np.concatenate(
                (data.root_link_pos_w, data.root_link_quat_w,
                 data.root_link_lin_vel_w, data.root_link_ang_vel_w,
                 data.joint_pos, data.joint_vel),
                axis=-1, dtype=np.float32,
            )

    def get_local_linvel(self) -> np.ndarray:
        """Return body-frame velocity for interactive target/current arrows."""
        return np.asarray(
            self._backend.get_sensor_data(self._cfg.sensor.local_linvel),
            dtype=get_global_dtype(),
        )

    def get_playback_model(self, env_index: int | None = None) -> Any:
        if getattr(self._cfg, "fixed_model_variants", None) is None:
            visual_model_file = self.get_scene_visual_model_file()
            if visual_model_file:
                return Path(visual_model_file)
        return super().get_playback_model(env_index)


class Go2ArmPush(ManagerTermBase):
    """Fixed world-frame force push dispatched through the Entity wrench contract."""

    def __init__(self, cfg, env: "ManagerBasedRlEnv"):
        super().__init__(env)
        cfg_task = _task_cfg(env)
        body_name = cfg_task.domain_rand.get("push_body_name", cfg_task.asset.base_name)
        self._entity: Entity = env.scene["robot"]
        body_ids, _ = self._entity.find_bodies((body_name,))
        _, self._body_ids = self._entity.bind_body_wrench(
            body_ids,
            torque=False,
            term_name="go2_arm_push",
        )
        self._force = np.asarray(cfg_task.domain_rand["max_force"], dtype=np.float64)[None, None, :]

    def __call__(
        self,
        env: "ManagerBasedRlEnv",
        env_ids: np.ndarray | None,
        force: np.ndarray | tuple[float, float, float],
    ) -> None:
        ids = (
            np.arange(env.num_envs, dtype=np.int32)
            if env_ids is None
            else np.asarray(env_ids, dtype=np.int32)
        )
        samples = np.asarray(force, dtype=np.float64)
        if samples.shape != (3,):
            raise ValueError(f"go2_arm_push force must have shape (3,), got {samples.shape}")
        forces = np.broadcast_to(
            samples[None, None, :],
            (len(ids), self._body_ids.size, 3),
        ).copy()
        self._entity.apply_body_wrench_to_sim(
            forces,
            None,
            self._body_ids,
            env_ids=ids,
            term_name="go2_arm_push",
        )
