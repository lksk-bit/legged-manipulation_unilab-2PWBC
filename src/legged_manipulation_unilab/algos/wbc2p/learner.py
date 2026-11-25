"""Alternating supervised dynamics and actor-critic BPTT.

SuperMimic's multi-step Bryson-weighted state loss, survival-masked
differentiable return, critic bootstrap and soft-done GAE are adapted to
separate RoboDuet objectives. Real dynamics is never in an actor graph.
"""

from __future__ import annotations

from contextlib import nullcontext
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch._functorch import config as functorch_config
from torch.nn.functional import binary_cross_entropy

from .bridge import ANG, LIN, POS, QUAT
from .geometry import inverse, multiply
from .model import DualPolicy, Dynamics, mlp, run_mlp, wheel_action_bounds


def soft_gae(values, last_value, rewards, done, gamma, lam):
    advantage = 0.0
    returns = []
    for value, reward, termination in reversed(list(zip(values, rewards, done, strict=True))):
        discount = gamma * (1 - termination)
        advantage = reward + discount * (last_value + lam * advantage) - value
        returns.append(value + advantage)
        last_value = value
    return torch.stack(returns[::-1])


class Learner:
    def __init__(self, task, bridge, cfg, device):
        self.cfg, self.task, self.bridge = cfg, task, bridge
        self.device = torch.device(device)
        self.stage2 = task.stage2
        self.layout = bridge.layout
        self.rng = np.random.default_rng(int(cfg.algo.seed) + 1)
        context_dim = bridge.context().shape[-1]
        self.world = Dynamics(
            cfg.world_model.hidden_dims, task.dt, task.cfg.sim_substeps, context_dim, self.layout
        ).to(device)
        self.policy = DualPolicy(
            task.defaults,
            cfg.policy.history_length,
            cfg.policy.hidden_dims,
            self.stage2,
            cfg.policy.planner_scale,
            task.cfg.control_config.clip_actions,
            bf16=True,
            wheel_bounds=wheel_action_bounds(bridge.env),
            has_gait_phase=bridge.command.phase is not None,
            gait_thresholds=(
                (task.cfg.command_config.gait_x_threshold,
                 task.cfg.command_config.gait_lateral_threshold,
                 task.cfg.command_config.gait_yaw_threshold)
                if task.cfg.asset.wheel_joint_names else None
            ),
        ).to(device)
        self.names = ("leg", "arm") if self.stage2 else ("leg",)
        critic_width = self.world.rep_dim + self.layout.joint_count + bridge.command_dim + 4 + context_dim + 1 + 7
        self.critics = nn.ModuleDict(
            {name: mlp(critic_width, cfg.policy.hidden_dims, 1) for name in self.names}
        ).to(device)
        for critic in self.critics.values():
            nn.init.zeros_(critic[-1].weight)
            nn.init.zeros_(critic[-1].bias)
        self.modules = {"world_model": self.world, "leg_actor": self.policy.leg}
        rates = {"world_model": cfg.world_model.learning_rate, "leg_actor": cfg.optimizer.leg_lr}
        if self.stage2:
            self.modules["arm_actor"] = self.policy.arm
            rates["arm_actor"] = cfg.optimizer.arm_lr
        for name in self.names:
            self.modules[f"{name}_critic"] = self.critics[name]
            rates[f"{name}_critic"] = cfg.optimizer.critic_lr
        self.optimizers = {
            k: torch.optim.Adam(m.parameters(), lr=rates[k], betas=(0.9, 0.95))
            for k, m in self.modules.items()
        }

        # Compile only the tensor paths. Replay, optimizer updates, and the
        # two separate autograd.grad calls stay outside the compiled functions.
        self.wm_loss = self.dynamics_loss
        self.imag_step = self.imagined_step
        if cfg.training.compile:
            self.wm_loss = torch.compile(
                self.wm_loss, backend=cfg.training.compile_backend, fullgraph=True
            )
            self.imag_step = torch.compile(
                self.imag_step, backend=cfg.training.compile_backend, fullgraph=True
            )

    def dynamics_loss(self, start, action, context, targets):
        predictions = []
        pred = start
        for target, dynamics_context in zip(action.unbind(), context.unbind(), strict=True):
            pred = self.world(pred, target, dynamics_context)
            predictions.append(pred)
        pred = torch.stack(predictions)
        rotation_error = 2 * multiply(inverse(targets[..., QUAT]), pred[..., QUAT])[..., 1:]
        sensor_error = (
            pred[..., self.layout.dq.stop:] - targets[..., self.layout.dq.stop:]
        ) / self.world.sensor_norm.scale().clamp_min(0.1)
        joint_error = pred[..., self.layout.q] - targets[..., self.layout.q]
        wheels = self.layout.wheels
        if wheels.start != wheels.stop:
            joint_error = torch.cat((joint_error[..., :wheels.start], joint_error[..., wheels.stop:]), -1)
        groups = {
            "root_position": (pred[..., POS] - targets[..., POS]) / 0.01,
            "root_orientation": rotation_error / 0.02,
            "root_velocity": pred[..., LIN] - targets[..., LIN],
            "root_angular_velocity": (pred[..., ANG] - targets[..., ANG]) / 3.14,
            "joint_position": joint_error / 0.1,
            "joint_velocity": (pred[..., self.layout.dq] - targets[..., self.layout.dq]) / 3.14,
            "sensors": torch.cat(
                (
                    sensor_error[..., : self.layout.collision.start - self.layout.dq.stop],
                    sensor_error[..., self.layout.collision.stop - self.layout.dq.stop :],
                ),
                -1,
            ),
        }
        squared = {name: error.square().mean() for name, error in groups.items()}
        collision = binary_cross_entropy(pred[..., self.layout.collision], targets[..., self.layout.collision])
        channels = sum(error.shape[-1] for error in groups.values())
        loss = (
            sum(squared[name] * error.shape[-1] for name, error in groups.items()) / channels
            + collision
        )
        return loss, pred[-1].detach(), {**squared, "collision": collision}

    def train_dynamics(self, batch):
        self.world.requires_grad_(True)
        targets = batch["next_state"]
        loss, prediction, groups = self.wm_loss(
            batch["state"][0], batch["target"], batch["context"], targets
        )
        self.optimize("world_model", loss)
        with torch.no_grad():
            pred_ee, _ = self.task.ee(prediction)
            true_ee, _ = self.task.ee(targets[-1])
            ee_error = (pred_ee - true_ee).norm(dim=-1).mean()
        stats = {
            "WorldModel/final_ee_error_m": ee_error,
            "WorldModel/loss": loss.detach(),
            **{f"WorldModel/{k}": v.detach() for k, v in groups.items()},
        }
        wheels = self.layout.wheels
        if wheels.start != wheels.stop:
            velocities = slice(self.layout.dq.start + wheels.start, self.layout.dq.start + wheels.stop)
            wheel_error = prediction[..., velocities] - targets[-1, ..., velocities]
            stats["WorldModel/final_wheel_velocity_rmse_rad_s"] = wheel_error.square().mean().sqrt()
        return stats

    def optimize(self, key, loss):
        opt = self.optimizers[key]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.modules[key].parameters(), 1.0, error_if_nonfinite=True)
        opt.step()

    def critic_obs(self, state, controller, task_command, context, elapsed, ee):
        position, quat = ee
        return torch.cat(
            (
                self.world.state_norm(self.world.represent(state)),
                controller[..., :self.layout.joint_count],
                task_command,
                position,
                quat,
                state[..., self.layout.force].unflatten(-1, (4, 3)).norm(dim=-1) / 30,
                self.world.context_norm(context),
                (elapsed * self.task.dt / self.task.cfg.max_episode_seconds).unsqueeze(-1),
            ),
            -1,
        )

    def imagined_step(self, state, controller, context, command, history, elapsed):
        action, plan, raw_plan, history = self.policy(
            state[..., self.layout.q], state[..., self.layout.dq], state[..., QUAT], controller[..., :self.layout.joint_count], command, history
        )
        target, next_controller = self.task.control(action, controller, command, raw_plan)
        next_state = self.world(state, target, context)
        kinematics = self.task.kinematics(next_state)
        ee = self.task.ee(next_state, kinematics)
        leg, arm = self.task.rewards(
            state, next_state, command, plan, controller, next_controller, kinematics, ee
        )
        done = self.task.termination(next_state, elapsed + 1)
        critic_command = torch.cat((command[..., :3], plan, command[..., 5:]), -1)
        return next_state, next_controller, history, ee, critic_command, leg, arm, done

    def train_policies(self, batch):
        # Stage 2 backpropagates twice through the same compiled rollout.
        context = (
            functorch_config.patch(donated_buffer=False)
            if self.stage2 and self.cfg.training.compile
            else nullcontext()
        )
        with context:
            return self._train_policies(batch)

    def _train_policies(self, batch):
        schedule_start = perf_counter()
        horizon = int(self.cfg.bptt.horizon)
        gamma, lam = self.cfg.bptt.gamma, self.cfg.bptt.lam
        schedule = self.bridge.command_schedule(
            batch["command_state"][0].cpu().numpy(), horizon + 1, self.rng
        )
        schedule = torch.as_tensor(schedule, device=self.device)
        schedule_time = perf_counter() - schedule_start
        if self.device.type == "cuda":
            stream = torch.cuda.current_stream(self.device)

            def mark():
                event = torch.cuda.Event(enable_timing=True)
                event.record(stream)
                return event

            def elapsed_time(start, end):
                return start.elapsed_time(end) / 1000.0
        else:
            mark = perf_counter

            def elapsed_time(start, end):
                return end - start

        rollout_start = mark()
        self.world.requires_grad_(False)
        self.critics.requires_grad_(False)
        s, ctrl, context, elapsed = (
            batch[k][0] for k in ("state", "controller", "context", "elapsed")
        )
        history = batch["leg_frame"], batch["arm_frame"]
        survival, returns = 1.0, {name: 0.0 for name in self.names}
        bootstrap_inputs, rewards, dones = [], {name: [] for name in self.names}, []
        beta = self.cfg.cooperation.beta if self.stage2 else 0.0
        ee = self.task.ee(s)
        for t in range(horizon):
            next_s, next_ctrl, history, next_ee, critic_command, leg, arm, probability = (
                self.imag_step(s, ctrl, context, schedule[t], history, elapsed + t)
            )
            coupled = {"leg": leg + beta * arm, "arm": arm + beta * leg}
            with torch.no_grad():
                bootstrap_inputs.append(
                    self.critic_obs(s, ctrl, critic_command, context, elapsed + t, ee)
                )
            for name in self.names:
                returns[name] = returns[name] + gamma**t * survival * coupled[name]
                rewards[name].append(coupled[name].detach())
            dones.append(probability.detach())
            survival = survival * (1 - probability)
            s, ctrl, ee = next_s, next_ctrl, next_ee
        final_command = schedule[-1]
        _, final_plan, _, _ = self.policy(
            s[..., self.layout.q], s[..., self.layout.dq], s[..., QUAT], ctrl[..., :self.layout.joint_count], final_command, history
        )
        final_command = torch.cat((final_command[..., :3], final_plan, final_command[..., 5:]), -1)
        last = self.critic_obs(s, ctrl, final_command, context, elapsed + horizon, ee)
        logs = {
            "Imagined/survival": survival.detach().mean(),
            **{f"Imagined/reward_{name}": torch.stack(rewards[name]).mean() for name in self.names},
        }
        actor_start = mark()
        backward_marks = {}
        for index, name in enumerate(self.names):
            key = f"{name}_actor"
            value = run_mlp(self.critics[name], last).squeeze(-1)
            loss = -(returns[name] + gamma**horizon * survival * value).mean()
            # autograd traverses both policies, but only the requested policy
            # receives this objective's gradient. Do both before either step.
            params = tuple(self.modules[key].parameters())
            backward_start = mark()
            grad = torch.autograd.grad(loss, params, retain_graph=index < len(self.names) - 1)
            backward_marks[name] = backward_start, mark()
            opt = self.optimizers[key]
            opt.zero_grad(set_to_none=True)
            for param, derivative in zip(params, grad, strict=True):
                param.grad = derivative
            logs[f"Loss/{key}"] = loss.detach()
            logs[f"Policy/{name}_grad_norm"] = nn.utils.clip_grad_norm_(
                params, 1.0, error_if_nonfinite=True
            )
            nn.utils.clip_grad_value_(params, 0.1)
        for name in self.names:
            self.optimizers[f"{name}_actor"].step()

        critic_start = mark()
        self.critics.requires_grad_(True)
        obs = torch.stack(bootstrap_inputs)
        for name in self.names:
            critic = self.critics[name]
            with torch.no_grad():
                values = run_mlp(critic, obs).squeeze(-1)
                value_targets = soft_gae(
                    values,
                    run_mlp(critic, last.detach()).squeeze(-1),
                    torch.stack(rewards[name]),
                    torch.stack(dones),
                    gamma,
                    lam,
                )
            flat_obs, flat_target = obs.flatten(0, 1), value_targets.flatten()
            losses = []
            for _ in range(self.cfg.bptt.critic_epochs):
                for ix in torch.randperm(len(flat_obs), device=obs.device).chunk(4):
                    value = run_mlp(critic, flat_obs[ix]).squeeze(-1)
                    loss = (value - flat_target[ix]).square().mean()
                    self.optimize(f"{name}_critic", loss)
                    losses.append(loss.detach())
            logs[f"Loss/{name}_critic"] = torch.stack(losses).mean()
        finished = mark()
        if self.device.type == "cuda":
            # Read all timings at the existing end-of-update CUDA boundary.
            finished.synchronize()
        logs.update(
            {
                "Perf/bptt_schedule_time": schedule_time,
                "Perf/bptt_rollout_time": elapsed_time(rollout_start, actor_start),
                "Perf/bptt_actor_time": elapsed_time(actor_start, critic_start),
                "Perf/bptt_critic_time": elapsed_time(critic_start, finished),
                **{
                    f"Perf/bptt_{name}_backward_time": elapsed_time(*marks)
                    for name, marks in backward_marks.items()
                },
            }
        )
        return logs
