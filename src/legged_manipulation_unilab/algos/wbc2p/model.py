"""Generalized-coordinate dynamics and independent deployable policies.

The integration follows SuperMimic's GenCoordRepresentation: predict changes
in root-frame and joint velocities, then integrate pose over the substeps.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.functional import normalize

from .bridge import ANG, LIN, POS, QUAT, StateLayout
from .geometry import attitude, cart_to_sphere, inverse, multiply, rotate, rotation_vector, unit


def mlp(inputs, hidden, outputs):
    widths = [inputs, *hidden]
    layers = []
    for a, b in zip(widths, widths[1:]):
        layers.extend((nn.Linear(a, b), nn.ELU()))
    return nn.Sequential(*layers, nn.Linear(widths[-1], outputs))


def run_mlp(net, x):
    """Use BF16 GEMMs on CUDA and return the result for FP32 state arithmetic."""
    if not x.is_cuda:
        return net(x)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return net(x).float()


def wheel_action_bounds(env):
    if not env.cfg.asset.wheel_joint_names:
        return None
    controls = env.scene["robot"].data.actuator_ctrl_range[12:-6]
    return torch.tensor(controls, dtype=torch.float32) / env.cfg.control_config.wheel_action_scale


class Normalizer(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.register_buffer("mean", torch.zeros(size))
        self.register_buffer("var", torch.ones(size))
        self.register_buffer("count", torch.tensor(0.0, dtype=torch.float64))

    @torch.no_grad()
    def update(self, x):
        x = x.reshape(-1, self.mean.numel())
        variance, mean = torch.var_mean(x, dim=0, unbiased=False)
        weight = (x.shape[0] / (self.count + x.shape[0])).to(x.dtype)
        self.var.lerp_(variance, weight)
        self.var.add_(weight * (1 - weight) * (mean - self.mean).square())
        self.mean.lerp_(mean, weight)
        self.count.add_(x.shape[0])

    def scale(self):
        return self.var.clamp_min(1e-4).sqrt()

    def forward(self, x):
        return ((x - self.mean) / self.scale()).clamp(-10, 10)


class Dynamics(nn.Module):
    def __init__(self, hidden, dt, substeps, context_dim, layout=None):
        super().__init__()
        self.layout = layout or StateLayout(18)
        self.rep_dim = 10 + 2 * self.layout.joint_count
        self.sensor_dim = self.layout.state_dim - self.layout.dq.stop
        self.velocity_dim = 6 + self.layout.joint_count
        self.dt = dt
        self.substep_coeff = (substeps + 1) / (2 * substeps)
        self.state_norm = Normalizer(self.rep_dim)
        self.action_norm = Normalizer(self.layout.joint_count)
        self.context_norm = Normalizer(context_dim)
        self.sensor_norm = Normalizer(self.sensor_dim)
        # Independent calibrated sensor targets share the dynamics trunk.
        self.net = mlp(self.rep_dim + self.layout.joint_count + context_dim, hidden, self.velocity_dim + self.sensor_dim)
        nn.init.uniform_(self.net[-1].weight, -0.001, 0.001)
        nn.init.zeros_(self.net[-1].bias)

    def mask_wheel_angles(self, joint_pos):
        wheels = self.layout.wheels
        if wheels.start == wheels.stop:
            return joint_pos
        return torch.cat(
            (joint_pos[..., :wheels.start], torch.zeros_like(joint_pos[..., wheels]), joint_pos[..., wheels.stop:]),
            dim=-1,
        )

    def represent(self, s):
        q = s[..., QUAT]
        v = rotate(inverse(q), s[..., LIN])
        w = rotate(inverse(q), s[..., ANG])
        g = -rotate(inverse(q), unit(q, 2))
        return torch.cat((s[..., 2:3], g, v, w, self.mask_wheel_angles(s[..., self.layout.q]), s[..., self.layout.dq]), -1)

    @torch.no_grad()
    def update_normalizers(self, s, target, context, next_s):
        self.state_norm.update(self.represent(s))
        self.action_norm.update(target)
        self.context_norm.update(context)
        self.sensor_norm.update(next_s[..., self.layout.dq.stop:])

    def forward(self, state, target, context):
        rep = self.represent(state)
        x = torch.cat(
            (self.state_norm(rep), self.action_norm(target), self.context_norm(context)), -1
        )
        pred = run_mlp(self.net, x).float()
        dv, dw, dj = pred[..., :3], pred[..., 3:6], pred[..., 6:self.velocity_dim]
        q = state[..., QUAT]
        c, dt = self.substep_coeff, self.dt
        root = state[..., POS] + rotate(q, rep[..., 4:7] + c * dv) * dt
        rot = normalize(multiply(q, rotation_vector((rep[..., 7:10] + c * dw) * dt)), dim=-1)
        v = rotate(q, rep[..., 4:7] + dv)
        w = rotate(q, rep[..., 7:10] + dw)
        joint_velocity = state[..., self.layout.dq]
        joint = state[..., self.layout.q] + self.mask_wheel_angles((joint_velocity + c * dj) * dt)
        sensor = pred[..., self.velocity_dim:] * self.sensor_norm.scale() + self.sensor_norm.mean
        start = self.layout.collision.start - self.layout.dq.stop
        end = self.layout.collision.stop - self.layout.dq.stop
        logits = pred[..., start + self.velocity_dim : end + self.velocity_dim] + torch.logit(
            self.sensor_norm.mean[start:end].clamp(0.005, 0.995)
        )
        sensor = torch.cat((sensor[..., :start], logits.sigmoid(), sensor[..., end:].relu()), -1)
        return torch.cat((root, rot, v, w, joint, joint_velocity + dj, sensor), -1)


class Actor(nn.Module):
    def __init__(self, width, history, hidden, outputs, *, bf16=False):
        super().__init__()
        self.bf16 = bf16
        self.normalizer = Normalizer(width)
        self.encoder = mlp(width * history, [hidden[0]], 32)
        self.head = mlp(width + 32, hidden, outputs)
        nn.init.uniform_(self.head[-1].weight, -0.01, 0.01)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, history):
        x = self.normalizer(history)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.bf16 and x.is_cuda):
            code = self.encoder(x.flatten(-2))
            action = self.head(torch.cat((x[..., -1, :], code), -1))
        return action.to(x.dtype)


def push(history, frame):
    return torch.cat((history[..., 1:, :], frame.unsqueeze(-2)), -2)


class DualPolicy(nn.Module):
    ARM_WIDTH = 20

    def __init__(self, defaults, history, hidden, stage2, planner_scale, clip, *, bf16=False, wheel_bounds=None, has_gait_phase=True, gait_thresholds=None):
        super().__init__()
        self.register_buffer("defaults", defaults)
        self.history = history
        self.clip = clip
        self.planner_scale = planner_scale
        self.arm_start = defaults.numel() - 6
        self.has_gait_phase = has_gait_phase
        self.gait_thresholds = gait_thresholds
        self.leg_width = 52 + 2 * (self.arm_start - 12) + (4 if self.has_gait_phase else 0)
        self.register_buffer("wheel_bounds", wheel_bounds, persistent=False)
        self.leg = Actor(self.leg_width, history, hidden, self.arm_start, bf16=bf16)
        self.arm = Actor(self.ARM_WIDTH, history, hidden, 8, bf16=bf16) if stage2 else None

    def zeros(self, n):
        return (
            self.defaults.new_zeros(n, self.history, self.leg_width),
            self.defaults.new_zeros(n, self.history, self.ARM_WIDTH),
        )

    def plan(self, raw):
        scaled = raw * self.planner_scale
        return torch.maximum(
            torch.minimum(scaled, raw.new_tensor([0.3, 0.4])), raw.new_tensor([-0.4, -0.4])
        ), scaled

    def forward(self, joints, joint_vel, quat, previous_action, command, histories, noise=None):
        """Consume only deployable proprioception/commands, arm first, same-step plan."""
        pos = joints - self.defaults
        rpy = attitude(quat)
        ee_goal = torch.cat((cart_to_sphere(command[..., -6:-3]), command[..., -3:]), -1)
        arm_frame = torch.cat((pos[..., self.arm_start:], previous_action[..., self.arm_start:], ee_goal, rpy), -1)
        arm_history = push(histories[1], arm_frame)
        if self.arm is None:
            arm_action = torch.zeros_like(pos[..., self.arm_start:])
            plan = command[..., 3:5]
            raw_plan = torch.zeros_like(plan)
        else:
            arm_action, raw = self.arm(arm_history).split((6, 2), -1)
            if noise is not None:
                arm_action = arm_action + noise[..., self.arm_start:self.arm_start + 6]
                raw = raw + noise[..., self.arm_start + 6:]
            plan, raw_plan = self.plan(raw)
        vel = command[..., :3]
        gravity = -rotate(inverse(quat), unit(quat, 2))
        goal = ee_goal if self.arm is not None else torch.zeros_like(ee_goal)
        leg_features = (
            gravity,
            pos[..., :12],
            0.05 * joint_vel[..., :self.arm_start],
            previous_action[..., :self.arm_start],
            vel * vel.new_tensor([2.0, 2.0, 0.25]),
            plan,
            goal,
            rpy,
        )
        if self.has_gait_phase:
            phase = torch.where(vel.norm(dim=-1, keepdim=True) < 0.1, 0.25, command[..., 5:-6])
            clock = (2 * torch.pi * phase).sin()
            if self.gait_thresholds is not None:
                active = (vel.abs() > vel.new_tensor(self.gait_thresholds)).any(-1, keepdim=True)
                clock = clock * active
            leg_features += (clock,)
        leg_frame = torch.cat(leg_features, -1)
        leg_history = push(histories[0], leg_frame)
        leg_action = self.leg(leg_history)
        if noise is not None:
            leg_action = leg_action + noise[..., :self.arm_start]
        if self.wheel_bounds is not None:
            wheels = leg_action[..., 12:].clamp(self.wheel_bounds[:, 0], self.wheel_bounds[:, 1])
            leg_action = torch.cat((leg_action[..., :12], wheels), dim=-1)
        return (
            torch.cat((leg_action, arm_action), -1).clamp(-self.clip, self.clip),
            plan,
            raw_plan,
            (leg_history, arm_history),
        )
