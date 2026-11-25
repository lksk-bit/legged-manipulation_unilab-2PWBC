"""Differentiable Go2 gait and foot contact rewards."""

import torch
from torch import nn

from legged_manipulation_unilab.algos.wbc2p.bridge import POS
from legged_manipulation_unilab.algos.wbc2p.geometry import inverse, rotate


class Go2ImaginedRewards(nn.Module):
    """Mirror Go2's gait and foot placement terms during model rollouts."""

    def __init__(self, cfg, layout):
        super().__init__()
        self.cfg = cfg.reward_parameters
        self.foot_velocity = layout.foot_vel
        self.register_buffer("raibert_centers", torch.tensor(cfg.asset.raibert_centers))

    def forward(self, state, command, feet, feet_quat, yaw, force, contact_filter, delta_action):
        cfg = self.cfg
        cmd = command[..., :3]
        feet_vel = state[..., self.foot_velocity].unflatten(-1, (4, 3))
        near = (10 * (force - 0.1)).sigmoid()

        def square_sum(x):
            return x.square().sum(-1)

        phase = command[..., 5:-6]
        clock = torch.where(cmd.norm(dim=-1, keepdim=True) < 0.1, 0.25, phase)

        def cdf(x):
            return torch.special.ndtr(x / cfg.kappa_gait_probs)

        desired = cdf(clock) * (1 - cdf(clock - 0.5)) + cdf(clock - 1) * (1 - cdf(clock - 1.5))
        swing = phase >= 0.6
        swing_height = 0.04 * (1 - (1 - 2 * (2 * phase - 1).clamp(0, 1)).abs()) + 0.02
        raibert_phase = (1 - 2 * phase).abs() - 0.5
        desired_x = self.raibert_centers[:, 0] + raibert_phase * cmd[..., 0:1] * (0.5 / 3.0)
        desired_y = self.raibert_centers[:, 1] + (
            raibert_phase * cmd[..., 2:3] * (0.225 * 0.5 / 3.0)
            * cmd.new_tensor([1.0, 1.0, -1.0, -1.0])
        )
        feet_xy = rotate(
            inverse(yaw).unsqueeze(-2).expand_as(feet_quat), feet - state[..., None, POS]
        )[..., :2]
        return {
            "feet_slip": (contact_filter * square_sum(feet_vel[..., :2])).sum(-1),
            "raibert_heuristic": square_sum(feet_xy[..., 0] - desired_x)
            + square_sum(feet_xy[..., 1] - desired_y),
            "contact": torch.where(~swing, near, 1 - near).mean(-1),
            "swing_feet_z": (
                (-(feet[..., 2] - cfg.target_foot_height).square() / 0.01).exp() * swing
            ).mean(-1),
            "feet_air_time": (swing * (1 - near)).mean(-1),
            "tracking_contacts_shaped_force": -(
                (1 - desired) * -torch.expm1(-force.square() / cfg.gait_force_sigma)
            ).mean(-1),
            "tracking_contacts_shaped_vel": -(
                desired * -torch.expm1(-square_sum(feet_vel) / cfg.gait_vel_sigma)
            ).mean(-1),
            "feet_clearance_cmd_linear": (
                (swing_height - feet[..., 2]).square() * (1 - desired)
            ).sum(-1),
        }
