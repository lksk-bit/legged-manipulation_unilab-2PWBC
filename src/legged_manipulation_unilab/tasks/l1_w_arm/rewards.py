"""L1-W wheel and ground contact rewards for MuJoCo and imagined rollouts."""

import numpy as np
import torch
from torch import nn

from legged_manipulation_unilab.algos.wbc2p.geometry import inverse, rotate, unit
from legged_manipulation_unilab.tasks.go2_arm.manager_env import ManipLocoReward

WHEEL_Y = (0.159412, -0.159399, -0.159350, 0.159350)
ACTUATOR_TO_SENSOR = (0, 1, 3, 2)


class L1WArmReward(ManipLocoReward):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        asset = self._task_cfg.asset
        self._wheels = slice(12, 12 + len(asset.wheel_joint_names))
        if self._name == "feet_slip":
            feet = asset.feet_geom_names
            self._wheel_axes = env.scene.bind_sensor_data(tuple(f"{name}_axis" for name in feet))
            self._wheel_foot_order = [12 + asset.wheel_joint_names.index(f"{name}_JOINT") for name in feet]

    def _foot_contact_velocities(self, feet_vel, joint_vel):
        axis = self._wheel_axes.read().reshape(feet_vel.shape)
        spin = axis * joint_vel[:, self._wheel_foot_order, None]
        feet_vel += np.cross(spin, [0.0, 0.0, -self._task_cfg.asset.wheel_radius])
        return feet_vel

    def _robot_reward(self, env, name, command, feet_force, feet_pos, feet_vel):
        if name in {"contact", "swing_feet_z", "wheel_roll"}:
            cfg = self._task_cfg.reward_parameters
            active = command._gait_is_active(command.vel_command_b)[:, None]
            swing = np.zeros(feet_force.shape[:2], dtype=feet_force.dtype)
            if command.phase is not None:
                swing = active / (1 + np.exp(-(command.feet_phase - 0.6) / cfg.phase_transition_width))
            if name == "swing_feet_z":
                error = np.square(feet_pos[:, :, 2] - cfg.target_foot_height)
                return (np.exp(-error / cfg.swing_reward_sigma) * swing).mean(axis=-1)
            contact = feet_force[..., 2] > 0.1
            if name == "wheel_roll":
                cmd = command.vel_command_b
                rolling = (np.abs(cmd[:, 0]) > 0.01) | (np.abs(cmd[:, 2]) > 0.01)
                target = (cmd[:, 0:1] - cmd[:, 2:3] * np.asarray(WHEEL_Y)) / self._task_cfg.asset.wheel_radius
                error = np.abs((self._entity.data.joint_vel[:, self._wheels] - target) / 10.0)
                huber = np.where(error < 1.0, 0.5 * error**2, error - 0.5)
                support = (contact * (1 - swing))[:, ACTUATOR_TO_SENSOR]
                reward = 1 - np.minimum(
                    (huber * support).sum(-1) / support.sum(-1).clip(min=1.0), 1.0
                )
                reward *= support.sum(-1).clip(max=1.0)
                return np.where(rolling & (support.sum(-1) > 0), reward, 0.0)
            return (contact * (1 - swing) + (~contact) * swing).mean(axis=-1)
        if name == "joint_action_rate":
            delta = env.action_manager.action - env.action_manager.prev_action
            return np.square(np.concatenate((delta[:, :12], delta[:, -6:]), axis=-1)).sum(-1)
        if name in {"joint_dof_acc", "wheel_dof_acc"}:
            velocity = self._entity.data.joint_vel
            acceleration = (velocity - self._previous_joint_vel) / env.step_dt
            self._previous_joint_vel[:] = velocity
            selected = acceleration[:, self._wheels] if name == "wheel_dof_acc" else np.concatenate(
                (acceleration[:, :12], acceleration[:, -6:]), axis=-1
            )
            return np.square(selected).sum(-1)
        if name == "wheel_action_rate":
            actions = env.action_manager
            delta = actions.action[:, self._wheels] - actions.prev_action[:, self._wheels]
            return np.square(delta).sum(axis=-1)
        if name == "wheel_stand_still":
            still = np.all(np.abs(command.vel_command_b) <= 0.01, axis=-1)
            return still * np.square(self._entity.data.joint_vel[:, self._wheels]).sum(axis=-1)
        return super()._robot_reward(env, name, command, feet_force, feet_pos, feet_vel)


class L1WImaginedRewards(nn.Module):
    def __init__(self, cfg, layout):
        super().__init__()
        self.layout = layout
        self.cfg = cfg.reward_parameters
        command = cfg.command_config
        self.thresholds = (command.gait_x_threshold, command.gait_lateral_threshold, command.gait_yaw_threshold)
        self.frequency = command.gait_frequency
        self.has_gait_phase = command.enable_gait_phase
        self.radius = cfg.asset.wheel_radius
        self.wheel_foot_order = tuple(
            12 + cfg.asset.wheel_joint_names.index(f"{foot}_JOINT")
            for foot in cfg.asset.feet_geom_names
        )

    def forward(self, state, command, feet, feet_quat, yaw, force, contact_filter, delta_action):
        feet_vel = state[..., self.layout.foot_vel].unflatten(-1, (4, 3))
        dq = state[..., self.layout.dq]
        axle = rotate(feet_quat, unit(feet_quat, 1))
        spin = axle * dq[..., self.wheel_foot_order].unsqueeze(-1)
        offset = spin.new_tensor([0.0, 0.0, -self.radius]).expand_as(spin)
        feet_vel = feet_vel + rotate(
            inverse(yaw).unsqueeze(-2).expand_as(feet_quat), torch.linalg.cross(spin, offset)
        )
        stationary = (command[..., :3].abs() <= 0.01).all(-1)
        terms = {
            "feet_slip": (contact_filter * feet_vel[..., :2].square().sum(-1)).sum(-1),
            "wheel_action_rate": delta_action[..., self.layout.wheels].square().sum(-1),
            "wheel_stand_still": stationary * dq[..., self.layout.wheels].square().sum(-1),
        }
        swing = torch.zeros_like(force)
        if self.has_gait_phase:
            active = (command[..., :3].abs() > command.new_tensor(self.thresholds)).any(-1)
            active = active & (self.frequency > 0)
            swing = ((command[..., 5:9] - 0.6) / self.cfg.phase_transition_width).sigmoid()
            swing = swing * active.unsqueeze(-1)
        # The reference's FK contact estimate supplies state gradients even
        # before learned contact-force outputs are accurate.
        contact = ((self.radius - feet[..., 2]) / 0.012).sigmoid()
        rolling = (command[..., 0].abs() > 0.01) | (command[..., 2].abs() > 0.01)
        target = (
            command[..., 0:1] - command[..., 2:3] * command.new_tensor(WHEEL_Y)
        ) / self.radius
        error = ((dq[..., self.layout.wheels] - target) / 10.0).abs()
        huber = torch.where(error < 1.0, 0.5 * error.square(), error - 0.5)
        support = (contact * (1 - swing))[..., ACTUATOR_TO_SENSOR]
        wheel_roll = 1 - (
            (huber * support).sum(-1) / support.sum(-1).clamp_min(1.0)
        ).clamp_max(1.0)
        # Smooth contact probabilities approach zero without becoming exactly
        # zero. Require support confidence to avoid rewarding unloaded wheels.
        wheel_roll = wheel_roll * support.sum(-1).clamp_max(1.0)
        terms.update({
            "contact": (contact * (1 - swing) + (1 - contact) * swing).mean(-1),
            "wheel_roll": torch.where(
                rolling & (support.sum(-1) > 0), wheel_roll, torch.zeros_like(wheel_roll)
            ),
            "swing_feet_z": (
                (-(feet[..., 2] - self.cfg.target_foot_height).square() / self.cfg.swing_reward_sigma).exp() * swing
            ).mean(-1),
        })
        return terms
