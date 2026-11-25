"""Snapshot the existing Manager-Based task and replay its exogenous command clocks."""

from __future__ import annotations

from copy import copy
from types import SimpleNamespace

import numpy as np

from legged_manipulation_unilab.tasks.go2_arm.manager_env import (
    _ARM_TOUCH_SENSORS,
    _LEG_COLLISION_SENSOR_GROUPS,
    _command,
)

# Root pose and world twist have the same packing for both robots.
POS, QUAT, LIN, ANG = slice(0, 3), slice(3, 7), slice(7, 10), slice(10, 13)


class StateLayout:
    def __init__(self, joint_count):
        self.joint_count = joint_count
        self.wheels = slice(12, joint_count - 6)
        self.q = slice(13, 13 + joint_count)
        self.dq = slice(self.q.stop, self.q.stop + joint_count)
        self.force = slice(self.dq.stop, self.dq.stop + 12)
        self.foot_vel = slice(self.force.stop, self.force.stop + 12)
        self.torque = slice(self.foot_vel.stop, self.foot_vel.stop + joint_count)
        self.collision = slice(self.torque.stop, self.torque.stop + 8)
        self.touch = slice(self.collision.stop, self.collision.stop + len(_ARM_TOUCH_SENSORS))
        self.state_dim = self.touch.stop
        self.control_widths = (joint_count,) * 5 + (2, 1)


# Keep the existing Go2 layout available to its callers.
_GO2 = StateLayout(18)
Q, DQ = _GO2.q, _GO2.dq
FORCE, FOOT_VEL, TORQUE, COLLISION, TOUCH = (
    _GO2.force, _GO2.foot_vel, _GO2.torque, _GO2.collision, _GO2.touch
)
STATE_DIM = _GO2.state_dim
CONTROL_WIDTHS = _GO2.control_widths

COMMAND_FIELDS = (
    "vel_command_b",
    "pitch_roll",
    "phase",
    "feet_phase",
    "curr_ee_goal_cart",
    "curr_ee_goal_sphere",
    "ee_goal_orn_euler",
    "ee_goal_orn_quat",
    "_ee_start_sphere",
    "_ee_goal_sphere",
    "_arm_goal_timer",
    "_traj_steps",
    "_traj_total_steps",
    "time_left",
)


class TaskBridge:
    def __init__(self, env):
        self.env = env
        self.robot = env.scene["robot"]
        self.command = _command(env)
        self.command_dim = 11 + self.command.feet_phase.shape[-1]
        self.command_fields = tuple(
            name for name in COMMAND_FIELDS if getattr(self.command, name) is not None
        )
        self.action = env.action_manager.get_term("joint_pos")
        cfg = env.cfg
        self.layout = StateLayout(len(self.robot.joint_names))
        self.force = env.scene.bind_sensor_data(tuple(cfg.sensor.feet_force))
        self.foot_vel = env.scene.bind_sensor_data(
            tuple(n.replace("_pos", "_global_linvel") for n in cfg.sensor.feet_pos)
        )
        self.torque = env.scene.bind_sensor_data(
            tuple(f"{n}_torque" for n in self.robot.joint_names)
        )
        self.touch = env.scene.bind_sensor_data(_ARM_TOUCH_SENSORS)
        collision_groups = cfg.sensor.leg_collision_groups or _LEG_COLLISION_SENSOR_GROUPS
        self.collision = env.scene.bind_sensor_data(tuple(n for group in collision_groups for n in group))
        offsets = np.cumsum([0, *(len(group) for group in collision_groups)])
        self.collision_slices = tuple(slice(start, end) for start, end in zip(offsets[:-1], offsets[1:], strict=True))
        backend = env._backend
        self.root_id = int(backend.get_body_ids((cfg.asset.base_name,))[0])
        self.body_ids = backend.get_body_subtree_ids(self.root_id)
        self.dof_ids = backend.get_joint_dof_indices(self.robot.joint_names)
        self.geom_ids = [
            list(env._backend.get_geom_names()).index(n)
            for n in (cfg.asset.ground, *cfg.asset.feet_geom_names)
        ]
        self.context_defaults = {
            "geom_friction": backend.get_geom_friction(),
            "body_mass": backend.get_body_mass(),
            "body_ipos": backend.get_body_ipos(),
            "dof_armature": backend.get_dof_armature(),
            "kp": backend.get_reset_term_default("kp"),
            "kd": backend.get_reset_term_default("kd"),
        }

    def state(self):
        collision = self.collision.read() > 0
        groups = [collision[:, columns] for columns in self.collision_slices]
        contacts = np.column_stack(
            [flag for group in groups for flag in (group[:, 0], group[:, 1:].any(-1))]
        )
        # A heading-invariant dynamics input cannot identify world-frame
        # tangential foot velocity. Keep its supervised target invariant too.
        yaw = self.robot.data.heading_w[:, None]
        cos, sin = np.cos(yaw), np.sin(yaw)
        velocity = self.foot_vel.read().reshape(self.env.num_envs, 4, 3)
        vx, vy, vz = velocity.transpose(2, 0, 1)
        foot_velocity = np.stack((cos * vx + sin * vy, -sin * vx + cos * vy, vz), axis=-1)
        return np.concatenate(
            (
                self.env.export_world_model_state(),
                self.force.read(),
                foot_velocity.reshape(self.env.num_envs, 12),
                self.torque.read(),
                contacts,
                self.touch.read(),
            ),
            axis=-1,
            dtype=np.float32,
        )

    def context(self):
        cached = self.robot._reset_state._committed_randomization
        fields = {
            name: cached.get(name, np.broadcast_to(default, (self.env.num_envs, *default.shape)))
            for name, default in self.context_defaults.items()
        }
        return np.column_stack(
            (
                fields["geom_friction"][:, self.geom_ids, 0],
                fields["body_mass"][:, self.body_ids],
                fields["dof_armature"][:, self.dof_ids],
                fields["body_ipos"][:, self.root_id],
                fields["kp"],
                fields["kd"],
            )
        ).astype(np.float32)

    def command_state(self):
        return np.column_stack([getattr(self.command, k) for k in self.command_fields]).astype(
            np.float32
        )

    def command_schedule(self, snapshots, steps, rng):
        """Fork the actual task command term, without a simulator or future states."""
        command = copy(self.command)
        command._env = SimpleNamespace(rng=rng, num_envs=len(snapshots))
        command._resetting_ids = None
        command._episode_steps = np.zeros(len(snapshots), dtype=np.int32)
        command.command_counter = np.zeros(len(snapshots), dtype=np.int64)
        command.metrics = {}
        command._metrics = {}
        cursor = 0
        for k in self.command_fields:
            src = getattr(self.command, k)
            width = src.shape[1] if src.ndim == 2 else 1
            setattr(
                command,
                k,
                snapshots[:, cursor : cursor + width]
                .reshape((len(snapshots), *src.shape[1:]))
                .astype(src.dtype, copy=True),
            )
            cursor += width
        result = []
        for _ in range(steps):
            result.append(command.policy_command())
            command.compute(self.env.step_dt)
        return np.stack(result)

    def current_command(self):
        return self.command.policy_command()

    def controller(self):
        a = self.action
        return np.concatenate(
            (
                self.env.action_manager.action,
                self.env.action_manager.prev_action,
                a._target,
                a._unclipped_target,
                a._previous_unclipped_target,
                self.command.plan_raw,
                a._has_previous_leg_target[:, None],
            ),
            axis=-1,
            dtype=np.float32,
        )
