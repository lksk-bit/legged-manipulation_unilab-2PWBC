"""Differentiable counterpart to the MuJoCo rewards and controls."""

from __future__ import annotations

import mujoco
import torch
from hydra.utils import get_class
from torch import nn

from .bridge import ANG, LIN, POS, QUAT
from .geometry import (
    Go2Kinematics,
    attitude,
    desired_up,
    ee_tracking,
    heading,
    inverse,
    rotate,
    unit,
    yaw_pose,
    yaw_quat,
)

HYBRID = frozenset(
    {
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
)
VELOCITY_TRACKING = frozenset({"tracking_lin_vel", "tracking_ang_vel"})


class Task(nn.Module):
    def __init__(self, bridge, stage2, termination_softness):
        super().__init__()
        env, robot = bridge.env, bridge.robot
        self.cfg = env.cfg
        self.layout = bridge.layout
        self.arm = slice(-6, None)
        self.wheels = self.layout.wheels
        self.mirror_rear_right, self.mirror_rear_left = (
            (slice(6, 9), slice(9, 12))
            if self.cfg.asset.wheel_joint_names
            else (slice(9, 12), slice(6, 9))
        )
        self.dt = float(env.step_dt)
        self.stage2 = stage2
        self.softness = termination_softness
        self.reference_height = bridge.command.reference_height
        model = mujoco.MjModel.from_xml_path(str(env.cfg.scene.model_file))
        feet = self.cfg.asset.feet_geom_names
        self.fk = Go2Kinematics(
            model,
            robot.joint_names,
            env.cfg.asset.base_name,
            (env.cfg.asset.ee_site_name, *feet, "imu"),
        )
        self.register_buffer(
            "defaults", torch.tensor(robot.data.default_joint_pos[0], dtype=torch.float32)
        )
        self.register_buffer(
            "control_bounds", torch.tensor(robot.data.actuator_ctrl_range, dtype=torch.float32)
        )
        joint_limits = torch.tensor(robot.data.soft_joint_pos_limits, dtype=torch.float32)
        joint_limits[self.wheels] = 0.0
        center = joint_limits.mean(-1)
        half = (joint_limits[:, 1] - center) * env.cfg.reward_parameters.soft_dof_pos_limit
        self.register_buffer("lower", torch.minimum(center - half, self.defaults))
        self.register_buffer("upper", torch.maximum(center + half, self.defaults))
        self.register_buffer("hip", torch.tensor([0, 3, 6, 9]))
        active_positions = list(range(12))
        if self.stage2:
            active_positions += list(range(self.wheels.stop, self.layout.joint_count))
        self.register_buffer("position_joints", torch.tensor(active_positions, dtype=torch.long))
        self.robot_rewards = get_class(self.cfg.imagined_reward_class)(self.cfg, self.layout)
        self.register_buffer(
            "leg_scale",
            torch.tensor(
                [
                    self.cfg.control_config.action_scale
                    * (self.cfg.control_config.hip_action_scale_reduction if i % 3 == 0 else 1.0)
                    for i in range(12)
                ]
            ),
        )
        manager = env.reward_manager
        self.scales = {}
        for name, term_config in zip(manager.active_terms, manager._term_cfgs, strict=True):
            weight = float(term_config.weight)
            if not stage2 and name in HYBRID or stage2 and name in self.cfg.stage2_disabled_rewards:
                continue
            if name == "orientation_control" and not stage2:
                weight *= 0.5
            if name in VELOCITY_TRACKING and stage2:
                switch = self.cfg.wbc_switch
                weight *= (
                    switch.hybrid_tracking_lin_vel_ratio
                    if name == "tracking_lin_vel"
                    else switch.hybrid_tracking_ang_vel_ratio
                )
            if weight:
                self.scales[name] = weight * self.dt

    def control(self, action, memory, command, raw_plan):
        """Same raw residual, latency, stance filter, and limit as Go2ArmIKAction."""
        control = self.cfg.control_config
        last_action, _, last_target, last_raw, _, _, valid = memory.split(self.layout.control_widths, -1)
        executed = last_action if control.simulate_action_latency else action
        leg = self.defaults[:12] + self.leg_scale * executed[..., :12]
        stationary = self.stationary(command).unsqueeze(-1)
        alpha = control.stand_action_smoothing
        leg = torch.where(
            stationary & (valid > 0),
            alpha * last_target[..., :12] + (1 - alpha) * leg,
            leg,
        )
        arm = self.defaults[self.arm] + executed[..., self.arm] * (
            control.arm_action_scale if self.stage2 else 0.0
        )
        wheels = executed[..., self.wheels] * control.wheel_action_scale
        raw = torch.cat((leg, wheels, arm), -1)
        target = torch.clamp(raw, self.control_bounds[:, 0], self.control_bounds[:, 1])
        history = torch.cat(
            (
                action,
                last_action,
                target,
                raw,
                last_raw,
                raw_plan,
                torch.ones_like(valid),
            ),
            -1,
        )
        return target, history

    def stationary(self, command):
        stationary = (command[..., :3].abs() <= 0.1).all(-1)
        cfg = self.cfg.command_config
        if self.cfg.asset.wheel_joint_names and cfg.enable_gait_phase and cfg.gait_frequency > 0:
            thresholds = command.new_tensor([
                cfg.gait_x_threshold, cfg.gait_lateral_threshold, cfg.gait_yaw_threshold
            ])
            stationary = stationary & ~(command[..., :3].abs() > thresholds).any(-1)
        return stationary

    def kinematics(self, state):
        return self.fk(state[..., POS], state[..., QUAT], state[..., self.layout.q])

    def ee(self, state, kinematics=None):
        p, q = self.kinematics(state) if kinematics is None else kinematics
        return yaw_pose(
            state[..., POS], state[..., QUAT], p[..., 0, :], q[..., 0, :], self.reference_height
        )

    def termination(self, state, steps):
        up = rotate(inverse(state[..., QUAT]), unit(state[..., QUAT], 2))
        # SuperMimic's sigmoid of normalized threshold violation. The two
        # thresholds and episode clock are those of Go2ArmFallen/TimeOut.
        height_limit = self.cfg.reward_parameters.terminal_body_height
        error = torch.maximum((1 - up[..., 2]) / 0.5, (height_limit - state[..., 2]) / 0.05 + 1.0)
        fallen = torch.sigmoid((error - 1) / self.softness)
        return torch.where(steps >= round(self.cfg.max_episode_seconds / self.dt), 1.0, fallen)

    def rewards(self, prev, state, command, plan, old, history, kinematics=None, ee=None):
        cfg = self.cfg.reward_parameters
        cmd = command[..., :3]
        goal, angles = command[..., -6:].split(3, -1)
        stationary = self.stationary(command)
        p, q = self.kinematics(state) if kinematics is None else kinematics
        yaw = yaw_quat(heading(state[..., QUAT]))
        ee_pos, ee_quat = self.ee(state, (p, q)) if ee is None else ee
        feet = p[..., 1:5, :]
        imu = p[..., -1, :] - state[..., POS]
        lin = rotate(
            inverse(state[..., QUAT]), state[..., LIN] + torch.linalg.cross(state[..., ANG], imu)
        )
        ang = rotate(inverse(state[..., QUAT]), state[..., ANG])
        up = rotate(inverse(state[..., QUAT]), unit(state[..., QUAT], 2))
        # The legacy 'roll' term reads the world IMU z axis in MuJoCo.
        up_world = rotate(state[..., QUAT], unit(state[..., QUAT], 2))
        pitch = attitude(state[..., QUAT])[..., 1]
        force = state[..., self.layout.force].unflatten(-1, (4, 3)).norm(dim=-1)
        previous_force = prev[..., self.layout.force].unflatten(-1, (4, 3)).norm(dim=-1)
        contact = (8 * (force - 1)).sigmoid()
        previous_contact = (8 * (previous_force - 1)).sigmoid() * old[..., -1:]
        contact_filter = 1 - (1 - contact) * (1 - previous_contact)
        qj, dq, torque = state[..., self.layout.q], state[..., self.layout.dq], state[..., self.layout.torque]
        acceleration = (dq - prev[..., self.layout.dq]) / self.dt
        action, previous, target, raw, previous_raw, raw_plan, _ = history.split(self.layout.control_widths, -1)
        _, last_last_action, last_target, _, last_last_raw, last_plan, _ = old.split(
            self.layout.control_widths, -1
        )
        q_active = qj[..., self.position_joints]

        def square_sum(x):
            return x.square().sum(-1)

        near = (10 * (force - 0.1)).sigmoid()
        dz = goal[..., 2] + self.reference_height - state[..., 2]
        # Same thresholds and target pitches as the reference. The branch
        # depends on goal/root height; pitch error inside it remains smooth.
        heuristic = torch.where(
            dz < -0.1, (pitch - 0.4).square(), torch.where(dz > 0.4, (pitch + 0.3).square(), 0.0)
        )
        delta_action = action - previous
        terms = {
            "tracking_lin_vel": (
                -square_sum(cmd[..., :2] - lin[..., :2]) / cfg.tracking_sigma
            ).exp(),
            "tracking_ang_vel": (-(cmd[..., 2] - ang[..., 2]).square() / cfg.tracking_sigma).exp(),
            "lin_vel_z": lin[..., 2].square(),
            "ang_vel_xy": square_sum(ang[..., :2]),
            "roll": up_world[..., 0].square(),
            "base_height": (state[..., 2] - cfg.base_height_target).square(),
            "leg_pose": (
                (qj[..., :12] - self.defaults[:12]).square() * qj.new_tensor([1.0, 1.0, 0.1] * 4)
            ).sum(-1),
            "mirrorup": 0.5 * (
                square_sum(qj[..., :3] - qj[..., self.mirror_rear_right])
                + square_sum(qj[..., 3:6] - qj[..., self.mirror_rear_left])
            ) * up_world[..., 2].clamp(0.0, 1.0),
            "action_rate": square_sum(delta_action[..., :12]),
            "joint_action_rate": square_sum(torch.cat(
                (delta_action[..., :12], delta_action[..., self.arm]), -1
            )),
            "joint_dof_acc": square_sum(torch.cat(
                (acceleration[..., :12], acceleration[..., self.arm]), -1
            )),
            "wheel_dof_acc": square_sum(acceleration[..., self.wheels]),
            "stand_action_rate": stationary * square_sum(delta_action[..., :12]),
            "action_smoothness_1": (
                (raw[..., :12] - previous_raw[..., :12]).square() * (previous[..., :12] != 0)
            ).sum(-1),
            "action_smoothness_2": (
                (raw[..., :12] - 2 * previous_raw[..., :12] + last_last_raw[..., :12]).square()
                * (previous[..., :12] != 0)
                * (last_last_action[..., :12] != 0)
            ).sum(-1),
            "dof_pos_limits": (
                (self.lower[self.position_joints] - q_active).relu() + (q_active - self.upper[self.position_joints]).relu()
            ).sum(-1),
            "stand_still": stationary * (qj[..., :12] - self.defaults[:12]).abs().sum(-1),
            "foot_drag": (0.5 * cfg.target_foot_height - feet[..., 2])
            .relu()
            .square()
            .mul(near)
            .sum(-1),
            "object_distance": (-square_sum(ee_pos - goal) / cfg.object_sigma).exp(),
            "object_distance_l2": square_sum(ee_pos - goal),
            "arm_collision": state[..., self.layout.touch].sum(-1),
            "orientation": square_sum(up_world[..., :2]),
            "torques": square_sum(torque),
            "dof_vel": square_sum(dq[..., :12]),
            "dof_acc": square_sum(acceleration[..., :12]),
            "collision": state[..., self.layout.collision].sum(-1),
            "hip_action_l2": square_sum(action[..., self.hip]),
            "arm_manip_commands_tracking_combine": ee_tracking(ee_pos, ee_quat, goal, angles, cfg),
            "orientation_heuristic": heuristic,
            "orientation_control": square_sum((up - desired_up(plan))[..., :2]),
            "arm_control_smoothness_1": ((raw_plan - last_plan).square() * (last_plan != 0)).sum(
                -1
            ),
            "arm_control_limits": (raw_plan.abs() - 0.4).relu().sum(-1),
            "arm_energy": square_sum((torque * dq)[..., self.arm]),
            "loco_energy": square_sum((torque * dq)[..., :12]),
            "arm_dof_vel": square_sum(dq[..., self.arm]),
            "arm_dof_acc": square_sum(acceleration[..., self.arm]),
            "arm_action_rate": square_sum(delta_action[..., self.arm]),
            "arm_action_smoothness_1": square_sum(target[..., self.arm] - last_target[..., self.arm])
            * (last_target[..., self.arm] != 0).any(-1),
        }
        terms.update(self.robot_rewards(
            state, command, feet, q[..., 1:5, :], yaw, force, contact_filter, delta_action
        ))

        def channel(names):
            vals = torch.stack([self.scales[k] * terms[k] for k in names], -1)
            if cfg.only_positive_rewards_ji22_style:
                return (
                    vals.clamp_min(0).sum(-1)
                    * (vals.clamp_max(0).sum(-1) / cfg.sigma_rew_neg).clamp_min(-80).exp()
                )
            return vals.sum(-1)

        leg = channel(self.scales)
        arm = (
            channel([k for k in self.scales if k not in VELOCITY_TRACKING and k != "wheel_roll"])
            if self.stage2
            else torch.zeros_like(leg)
        )
        return leg, arm
