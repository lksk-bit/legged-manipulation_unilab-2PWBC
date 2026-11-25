"""Go2+Airbot kinematics in MuJoCo's wxyz convention.

The MJCF endpoint site already describes the Airbot TCP and tool axes. The
ARX5's additional grasper offset is specific to its different end body.
"""

from __future__ import annotations

import mujoco
import torch
from torch import Tensor, nn


def rotate(q: Tensor, v: Tensor) -> Tensor:
    xyz = q[..., 1:]
    cross = 2 * torch.linalg.cross(xyz, v)
    return v + q[..., :1] * cross + torch.linalg.cross(xyz, cross)


def inverse(q: Tensor) -> Tensor:
    return torch.cat((q[..., :1], -q[..., 1:]), -1)


def multiply(a: Tensor, b: Tensor) -> Tensor:
    aw, av = a[..., :1], a[..., 1:]
    bw, bv = b[..., :1], b[..., 1:]
    return torch.cat(
        (aw * bw - (av * bv).sum(-1, keepdim=True), aw * bv + bw * av + torch.linalg.cross(av, bv)),
        -1,
    )


def rotation_vector(v: Tensor) -> Tensor:
    angle = torch.linalg.vector_norm(v, dim=-1, keepdim=True)
    return torch.cat(((angle / 2).cos(), 0.5 * torch.sinc(angle / (2 * torch.pi)) * v), -1)


def unit(q: Tensor, axis: int) -> Tensor:
    return torch.eye(3, device=q.device, dtype=q.dtype)[axis].expand_as(q[..., 1:])


def heading(q: Tensor) -> Tensor:
    forward = rotate(q, unit(q, 0))
    return torch.atan2(forward[..., 1], forward[..., 0])


def yaw_quat(yaw: Tensor) -> Tensor:
    zeros = torch.zeros_like(yaw)
    return torch.stack(((yaw / 2).cos(), zeros, zeros, (yaw / 2).sin()), -1)


def tool_angles(q: Tensor) -> Tensor:
    x, y, z = (rotate(q, unit(q, i)) for i in range(3))
    return torch.stack(
        (
            torch.atan2(y[..., 2], y[..., 1]),
            torch.atan2(z[..., 0], z[..., 2]),
            torch.atan2(x[..., 1], x[..., 0]),
        ),
        -1,
    )


def attitude(q: Tensor) -> Tensor:
    up = rotate(inverse(q), unit(q, 2))
    return torch.stack(
        (torch.atan2(up[..., 1], up[..., 2]), torch.atan2(-up[..., 0], up[..., 1:].norm(dim=-1))),
        -1,
    )


def desired_up(plan: Tensor) -> Tensor:
    # Vendor orientation_control uses quat_roll(-roll) * quat_pitch(-pitch).
    pitch, roll = plan.unbind(-1)
    return torch.stack((pitch.sin() * roll.cos(), -roll.sin(), pitch.cos() * roll.cos()), -1)


def cart_to_sphere(p: Tensor) -> Tensor:
    x, y, z = p.unbind(-1)
    return torch.stack(
        (p.norm(dim=-1), torch.atan2(z, p[..., :2].norm(dim=-1)), torch.atan2(y, x)), -1
    )


def yaw_pose(root: Tensor, quat: Tensor, p: Tensor, q: Tensor, height: float):
    yaw = yaw_quat(heading(quat))
    origin = torch.cat((root[..., :2], torch.full_like(root[..., :1], height)), -1)
    return rotate(inverse(yaw), p - origin), multiply(inverse(yaw), q)


class Go2Kinematics(nn.Module):
    """Static MJCF tree with differentiable hinge and site transforms."""

    def __init__(self, model: mujoco.MjModel, joint_names, root_name, site_names):
        super().__init__()
        self.root = model.body(root_name).id
        joints = {model.joint(name).id: i for i, name in enumerate(joint_names)}
        sites = [model.site(name).id for name in site_names]
        self.site_bodies = [int(model.site_bodyid[i]) for i in sites]
        needed = set()
        for body in self.site_bodies:
            while body != self.root:
                if body == 0:
                    raise ValueError("Site is outside the robot kinematic tree")
                needed.add(body)
                body = int(model.body_parentid[body])
        self.links = []
        for b in sorted(needed):
            ids = [int(model.body_jntadr[b] + j) for j in range(model.body_jntnum[b])]
            if len(ids) > 1 or any(model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE for j in ids):
                raise ValueError("Go2 FK expects at most one hinge per non-root body")
            j = ids[0] if ids else None
            self.links.append(
                (
                    b,
                    int(model.body_parentid[b]),
                    j,
                    joints[j] if j is not None else None,
                    float(model.qpos0[model.jnt_qposadr[j]]) if j is not None else 0.0,
                )
            )
        for name, array in (
            ("body_pos", model.body_pos),
            ("body_quat", model.body_quat),
            ("axis", model.jnt_axis),
            ("anchor", model.jnt_pos),
            ("site_pos", model.site_pos[sites]),
            ("site_quat", model.site_quat[sites]),
        ):
            self.register_buffer(name, torch.tensor(array, dtype=torch.float32))

    def forward(self, root: Tensor, quat: Tensor, q: Tensor):
        poses = {self.root: (root, quat)}
        for body, parent, joint, index, ref in self.links:
            p, r = poses[parent]
            p = p + rotate(r, self.body_pos[body].expand_as(p))
            r = multiply(r, self.body_quat[body].expand_as(r))
            if joint is not None:
                delta = rotation_vector((q[..., index : index + 1] - ref) * self.axis[joint])
                anchor = self.anchor[joint].expand_as(p)
                p = p + rotate(r, anchor - rotate(delta, anchor))
                r = multiply(r, delta)
            poses[body] = (p, r)
        positions, rotations = [], []
        for i, body in enumerate(self.site_bodies):
            p, r = poses[body]
            positions.append(p + rotate(r, self.site_pos[i].expand_as(p)))
            rotations.append(multiply(r, self.site_quat[i].expand_as(r)))
        return torch.stack(positions, -2), torch.stack(rotations, -2)


def ee_tracking(pos, quat, goal, goal_angles, cfg):
    lpy = cart_to_sphere(pos) - cart_to_sphere(goal)
    angles = tool_angles(quat) - goal_angles
    # Angular coordinates live on the circle.
    lpy = torch.cat((lpy[..., :1], torch.atan2(lpy[..., 1:].sin(), lpy[..., 1:].cos())), -1)
    angles = torch.atan2(angles.sin(), angles.cos())
    return torch.exp(
        -cfg.manip_weight_lpy * (lpy.abs() / pos.new_tensor(cfg.manip_lpy_range)).sum(-1)
        - cfg.manip_weight_rpy * (angles.abs() / pos.new_tensor(cfg.manip_rpy_range)).sum(-1)
    )
