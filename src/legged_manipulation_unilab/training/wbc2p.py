"""Training and agent-only keyboard playback for 2PWBC."""

from __future__ import annotations

import json
import time
from collections import deque
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.tensorboard import SummaryWriter
from unilab.base.config_adapter import BackendAdapter, create_env

from legged_manipulation_unilab.algos.wbc2p.bridge import QUAT, TaskBridge
from legged_manipulation_unilab.algos.wbc2p.geometry import desired_up, inverse, rotate, unit
from legged_manipulation_unilab.algos.wbc2p.learner import Learner
from legged_manipulation_unilab.algos.wbc2p.model import DualPolicy, wheel_action_bounds
from legged_manipulation_unilab.algos.wbc2p.replay import Replay
from legged_manipulation_unilab.algos.wbc2p.task import Task
from legged_manipulation_unilab.tasks.go2_arm.manager_env import _command
from legged_manipulation_unilab.training.wm_panel import TrainingPanel

_ENV_STEP_TIMINGS = {
    "apply_action_ms": "Perf/env_apply_action_time",
    "step_core_ms": "Perf/env_step_core_time",
    "backend_physics_ms": "Perf/env_physics_time",
    "update_state_ms": "Perf/env_update_state_time",
}


def checkpoint_path(run, checkpoint=-1):
    from legged_manipulation_unilab.cli import _resolve_eval_run

    source = _resolve_eval_run(run)
    if source.is_file():
        return source
    if str(checkpoint) != "-1":
        name = f"model_{checkpoint}.pt" if str(checkpoint).isdigit() else str(checkpoint)
        return source / name
    models = [p for p in source.glob("model_*.pt") if p.stem[6:].isdigit()]
    return max(models, key=lambda p: int(p.stem[6:]))


def load_checkpoint(path, device):
    data = torch.load(path, map_location=device, weights_only=True)
    if data.get("format") == "roboduet_wm_v1":
        cfg = data["config"]
        env = cfg["env"]
        if "roboduet_switch" in env:
            env["wbc_switch"] = env.pop("roboduet_switch")
        scene = env.get("scene", {})
        if Path(scene.get("model_file") or "").name == "scene_roboduet.xml":
            from legged_manipulation_unilab.assets import ASSETS_ROOT_PATH

            scene["model_file"] = str(ASSETS_ROOT_PATH / "robots/l1_w_arm/scene_2pwbc.xml")
        data["format"] = "wbc2p_v1"
    if data.get("format") != "wbc2p_v1":
        raise ValueError(f"{path} is not a 2PWBC checkpoint")
    return data


def _save_atomic(data, path):
    tmp = path.with_suffix(".tmp")
    torch.save(data, tmp)
    tmp.replace(path)


def save_checkpoint(run, iteration, learner, cfg):
    config = OmegaConf.to_container(cfg, resolve=True)
    _save_atomic(
        {
            "format": "wbc2p_v1",
            "iteration": iteration,
            "config": config,
            "modules": {k: v.state_dict() for k, v in learner.modules.items()},
            "optimizers": {k: v.state_dict() for k, v in learner.optimizers.items()},
        },
        run / f"model_{iteration}.pt",
    )
    for side in learner.names:
        _save_atomic(
            {
                "format": "wbc2p_actor_v1",
                "side": side,
                "iteration": iteration,
                "config": config,
                "default_joint_pos": learner.policy.defaults,
                "state_dict": learner.modules[f"{side}_actor"].state_dict(),
            },
            run / f"{side}_actor.pt",
        )


def _check_wheeled_policy_layout(checkpoint):
    if (
        checkpoint["config"]["training"]["task_name"] == "L1WArmManipLoco"
        and checkpoint["modules"]["leg_actor"]["normalizer.mean"].numel() not in (60, 64)
    ):
        raise ValueError(
            "Unsupported L1-W policy observation width."
        )


def resume(learner, checkpoint, cfg):
    saved = checkpoint["config"]
    if saved["training"]["task_name"] != cfg.training.task_name:
        raise ValueError("2PWBC checkpoint belongs to a different robot task")
    _check_wheeled_policy_layout(checkpoint)
    if checkpoint["modules"]["leg_actor"]["normalizer.mean"].numel() != learner.policy.leg_width:
        raise ValueError(
            "Checkpoint gait-phase observation width differs from the current task. "
            "Start Stage 1 from scratch with the current gait configuration."
        )
    source = saved["stage"]
    if cfg.stage == "stage1" and source == "stage2":
        raise ValueError("Cannot resume Stage-1 from Stage-2")
    for name, module in learner.modules.items():
        if name in checkpoint["modules"]:
            module.load_state_dict(checkpoint["modules"][name])
            opt = learner.optimizers[name]
            rates = [group["lr"] for group in opt.param_groups]
            opt.load_state_dict(checkpoint["optimizers"][name])
            for group, rate in zip(opt.param_groups, rates, strict=True):
                group["lr"] = rate
        elif source == cfg.stage:
            raise ValueError(f"Checkpoint is missing {name}")
    return int(checkpoint["iteration"]) + 1


def _make_env(cfg, envs, *, collector_observations=False):
    from legged_manipulation_unilab.assets import ensure_assets

    ensure_assets()
    override = BackendAdapter(cfg, root_dir=Path.cwd()).build_task_env_cfg_override()
    if cfg.training.task_name == "L1WArmManipLoco" and "reward_implementation" not in cfg:
        # Old L1-W checkpoints stored the shared Go2 class for every term.
        from legged_manipulation_unilab.tasks.l1_w_arm.rewards import L1WArmReward

        for term in override["rewards"].values():
            if isinstance(term, dict) and term.get("func") == (
                "legged_manipulation_unilab.tasks.go2_arm.manager_env.Go2ArmReward"
            ):
                term["func"] = f"{L1WArmReward.__module__}.{L1WArmReward.__name__}"
    if collector_observations:
        override["wm_collector_observations"] = True
    return create_env(cfg, num_envs=envs, env_cfg_override=override)


class Collector:
    def __init__(self, learner, replay, env, bridge, cfg):
        self.learner, self.replay, self.cfg, self.env, self.bridge = (
            learner,
            replay,
            cfg,
            env,
            bridge,
        )
        self.device = learner.device
        self.layout = bridge.layout
        self.history = learner.policy.zeros(env.num_envs)
        self.episode_lengths = deque(maxlen=100)
        self.noise_scale = torch.tensor(
            [cfg.replay.exploration_std] * self.layout.joint_count + [cfg.replay.planner_exploration_std] * 2,
            device=self.device,
        )
        self.state = None
        self.context = None

    def snapshot(self):
        bridge, env = self.bridge, self.env
        return {
            "state": self.state,
            "context": self.context,
            "controller": bridge.controller(),
            "command_state": bridge.command_state(),
            "elapsed": env.episode_length_buf.astype(np.float32, copy=True),
        }

    def pack_transitions(self, records):
        """Upload the CPU fields together and keep policy-generated history on-device."""
        keys = tuple(records[0])
        tensors = []
        for key in keys:
            arrays = np.stack([row[key] for row in records])
            tensors.append(arrays.reshape(*arrays.shape[:2], -1))
        packed = torch.from_numpy(np.concatenate(tensors, axis=-1, dtype=np.float32))
        widths = [a.shape[-1] for a in tensors]

        def unpack(data):
            return {
                key: field.squeeze(-1) if records[0][key].ndim == 1 else field
                for key, field in zip(keys, data.split(widths, dim=-1), strict=True)
            }

        learner_fields = unpack(packed.to(self.device))
        replay_fields = (
            learner_fields
            if self.replay.storage_device == torch.device(self.device)
            else unpack(packed)
        )
        return learner_fields, replay_fields

    @torch.no_grad()
    def collect(self, steps):
        rewards, plans, records, dones = [], [], [], []
        leg_frames, arm_frames = [], []
        reward_terms: dict[str, list[float]] = {}
        terminations = np.zeros(2)
        step_time = reset_time = 0.0
        step_timings = {}
        bridge, learner, env = self.bridge, self.learner, self.env
        if self.state is None:
            self.state, self.context = bridge.state(), bridge.context()
        n = self.layout.joint_count
        for _ in range(steps):
            record = self.snapshot()
            # Only the deployable inputs cross to the GPU during the step.
            actor_input = np.concatenate(
                (
                    self.state[:, self.layout.q],
                    self.state[:, self.layout.dq],
                    self.state[:, QUAT],
                    record["controller"][:, :n],
                    bridge.current_command(),
                ),
                axis=-1,
            )
            q, dq, quat, previous, command = (
                torch.from_numpy(actor_input).to(self.device).split((n, n, 4, n, bridge.command_dim), dim=-1)
            )
            noise = torch.randn((env.num_envs, n + 2), device=self.device) * self.noise_scale
            actions, plan, raw_plan, self.history = learner.policy(
                q, dq, quat, previous, command, self.history, noise
            )
            action_np, plan_np, raw_plan_np = np.split(
                torch.cat((actions, plan, raw_plan), -1).cpu().numpy(), [n, n + 2], axis=-1
            )
            bridge.command.set_body_plan(plan_np, raw_plan_np)
            started = time.perf_counter()
            result = env.step(np.ascontiguousarray(action_np))
            step_time += time.perf_counter() - started
            timing = result.info.get("timing", {})
            for key, metric in _ENV_STEP_TIMINGS.items():
                if key in timing:
                    step_timings[metric] = step_timings.get(metric, 0.0) + timing[key] / 1000.0

            # The next state is reused as the next observation unless a
            # terminal reset changes the physics.
            self.state = bridge.state()
            record.update(
                next_state=self.state, target=bridge.action._target.copy(), action=action_np
            )
            records.append(record)
            plans.append(plan)
            leg_frames.append(self.history[0][:, -1])
            arm_frames.append(self.history[1][:, -1])
            done = result.terminated | result.truncated
            dones.append(done)
            rewards.append(result.info["wbc2p_reward"].mean(axis=0))
            for k, v in result.info["log"].items():
                if k.startswith("reward/") or k in (
                    "wbc2p/leg_penalty_factor",
                    "wbc2p/leg_target_clip_fraction",
                ):
                    reward_terms.setdefault(k, []).append(float(v))
            terminations += (result.terminated.sum(), result.truncated.sum())
            if done.any():
                self.episode_lengths.extend(env.episode_length_buf[done].tolist())
                started = time.perf_counter()
                env.reset(np.flatnonzero(done).astype(np.int32))
                reset_time += time.perf_counter() - started
                self.state, self.context = bridge.state(), bridge.context()
                keep = torch.from_numpy(~done).to(self.device)[:, None, None]
                self.history = tuple(h * keep for h in self.history)

        batch, replay_fields = self.pack_transitions(records)
        leg_frame, arm_frame = torch.stack(leg_frames), torch.stack(arm_frames)
        learner.world.update_normalizers(
            batch["state"], batch["target"], batch["context"], batch["next_state"]
        )
        learner.policy.leg.normalizer.update(leg_frame)
        if learner.policy.arm is not None:
            learner.policy.arm.normalizer.update(arm_frame)
        self.replay.append_many(
            torch.from_numpy(np.stack(dones)),
            **replay_fields,
            stage=torch.full(
                (steps, env.num_envs),
                2 if learner.stage2 else 1,
                dtype=torch.int8,
                device=self.replay.storage_device,
            ),
            leg_frame=leg_frame,
            arm_frame=arm_frame,
        )
        quat = batch["next_state"][..., QUAT]
        up = rotate(inverse(quat), unit(quat, 2))
        goal_up = desired_up(torch.stack(plans))

        def tilt(vector):
            x, y, z = vector.unbind(-1)
            return torch.stack((torch.atan2(-x, torch.hypot(y, z)), torch.atan2(y, z)), -1)

        pitch_mae, roll_mae = (tilt(up) - tilt(goal_up)).abs().mean((0, 1)).rad2deg().tolist()
        leg, arm = np.mean(rewards, axis=0)
        stats = {
            "Real/reward_leg": leg,
            "Real/reward_arm": arm,
            "Real/falls": terminations[0],
            "Real/timeouts": terminations[1],
            "Real/pitch_mae_deg": pitch_mae,
            "Real/roll_mae_deg": roll_mae,
            "Perf/mujoco_time": step_time + reset_time,
            "Perf/env_step_time": step_time,
            "Perf/env_reset_time": reset_time,
            **step_timings,
            **{k: float(np.mean(values)) for k, values in reward_terms.items()},
        }
        if self.episode_lengths:
            stats["Real/mean_episode_length"] = float(np.mean(self.episode_lengths))
        return stats


def train(cfg):
    if cfg.stage not in ("stage1", "stage2"):
        raise ValueError("stage must be stage1 or stage2")
    if cfg.world_model.representation != "gencoord" or cfg.policy.plan_vel:
        raise ValueError("2PWBC supports gencoord and a pitch/roll-only planner")
    if cfg.env.control_config.arm_action_mode != "raw":
        raise ValueError("2PWBC requires the raw arm actuator interface")
    stage1 = cfg.stage == "stage1"
    if (
        cfg.env.arm_stage.freeze_arm_joints != stage1
        or cfg.env.arm_stage.disable_ee_goal_trajectory != stage1
        or not cfg.env.wbc_switch.world_model_control
        or not cfg.env.wbc_switch.use_unified_observation
        or cfg.env.wbc_switch.start_steps != 0
        or cfg.env.wbc_switch.duration_steps != 0
    ):
        raise ValueError(
            "Use stage=stage1 or stage=stage2 to select the control and reward contract"
        )
    if cfg.replay.storage_device not in ("learner", "cpu"):
        raise ValueError("replay.storage_device must be learner or cpu")
    if cfg.replay.capacity < cfg.world_model.prediction_window:
        raise ValueError("replay.capacity must cover the dynamics prediction window")
    if (
        min(
            cfg.policy.history_length,
            cfg.world_model.prediction_window,
            cfg.world_model.updates,
            cfg.bptt.horizon,
            cfg.bptt.critic_epochs,
            cfg.algo.max_iterations,
            cfg.algo.save_interval,
            cfg.algo.num_envs,
            cfg.algo.num_steps_per_env,
            cfg.world_model.batch_size,
            cfg.bptt.batch_size,
            cfg.bptt.termination_softness,
        )
        <= 0
    ):
        raise ValueError("History, windows, updates and intervals must be positive")
    steps_per_update = Fraction(cfg.algo.num_steps_per_env)
    utd = cfg.algo.get("utd")
    if utd is not None:
        if not 0 < utd <= 1:
            raise ValueError("algo.utd must be in (0, 1] to collect at least one step per update")
        steps_per_update = 1 / Fraction(str(utd))
    devices = cfg.training.devices or []
    if len(devices) > 1 or (devices and cfg.training.device is not None):
        raise ValueError(
            "2PWBC uses one learner device. Choose training.device or one training.devices entry"
        )
    device = torch.device(
        f"cuda:{devices[0]}"
        if devices
        else cfg.training.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    torch.manual_seed(cfg.algo.seed)
    source = (
        None
        if str(cfg.algo.load_run) == "-1"
        else checkpoint_path(cfg.algo.load_run, cfg.algo.checkpoint)
    )
    if cfg.stage == "stage2" and source is None:
        raise ValueError("Stage-2 requires a 2PWBC Stage-1 or Stage-2 checkpoint")
    run = (
        Path(cfg.training.log_dir)
        if cfg.training.log_dir
        else Path(cfg.training.log_root)
        / str(cfg.training.task_name)
        / f"{time.strftime('%Y-%m-%d_%H-%M-%S')}_{time.time_ns() % 1000000000:09d}_{cfg.stage}"
    )
    run.mkdir(parents=True, exist_ok=True)
    if (run / "run_config.json").exists():
        raise ValueError(f"Run directory already contains training data: {run}")
    env = _make_env(cfg, cfg.algo.num_envs, collector_observations=True)
    try:
        env.set_autoreset(False)
        env.reset(seed=cfg.algo.seed)
        bridge = TaskBridge(env)
        stage2 = cfg.stage == "stage2"
        task = Task(bridge, stage2, cfg.bptt.termination_softness).to(device)
        learner = Learner(task, bridge, cfg, device)
        start = resume(learner, load_checkpoint(source, device), cfg) if source else 0
        storage = device if cfg.replay.storage_device == "learner" else "cpu"
        replay = Replay(cfg.replay.capacity, env.num_envs, device, cfg.algo.seed, storage)
        collector = Collector(learner, replay, env, bridge, cfg)
        (run / "run_config.json").write_text(
            json.dumps(
                {
                    "config": OmegaConf.to_container(cfg, resolve=True),
                },
                indent=2,
            )
        )
        precision = "BF16 MLP / FP32 state" if device.type == "cuda" else "FP32"
        print(f"2PWBC {cfg.stage} on {device} ({precision}): {run.resolve()}", flush=True)
        panel = TrainingPanel(
            cfg.stage,
            device,
            env.num_envs,
            cfg.algo.max_iterations,
            enabled=cfg.training.live_panel,
        )
        with SummaryWriter(str(run)) as writer, panel:
            remaining = max(cfg.replay.warmup_steps, cfg.world_model.prediction_window)
            while remaining:
                chunk = min(remaining, cfg.algo.num_steps_per_env)
                collector.collect(chunk)
                remaining -= chunk
            panel.start_updates(compiling=cfg.training.compile)
            stop = start + cfg.algo.max_iterations
            for iteration in range(start, stop):
                # Take the difference in cumulative collection budgets. Using
                # the saved iteration preserves the alternating phase on resume.
                collect_steps = int((iteration + 1) * steps_per_update) - int(
                    iteration * steps_per_update
                )
                begin = time.perf_counter()
                stats = collector.collect(collect_steps)
                collected = time.perf_counter()
                for _ in range(cfg.world_model.updates):
                    batch = replay.sample(
                        cfg.world_model.prediction_window, cfg.world_model.batch_size, policy=False
                    )
                    if batch is None:
                        raise RuntimeError(
                            "No contiguous episode segment in replay for the WM window"
                        )
                    stats.update(learner.train_dynamics(batch))
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                wm_finished = time.perf_counter()
                starts = replay.sample(1, cfg.bptt.batch_size, cfg.policy.history_length)
                stats.update(learner.train_policies(starts))
                stats = {k: float(v) for k, v in stats.items()}
                finished = time.perf_counter()
                stats.update(
                    {
                        "Perf/collection_steps_per_env": collect_steps,
                        "Perf/collection_time": collected - begin,
                        "Perf/learning_time": finished - collected,
                        "Perf/world_model_time": wm_finished - collected,
                        "Perf/bptt_time": finished - wm_finished,
                        "Perf/replay_and_collection_time": collected
                        - begin
                        - stats["Perf/mujoco_time"],
                        "Perf/total_fps": env.num_envs * collect_steps / (finished - begin),
                    }
                )
                for key, value in stats.items():
                    writer.add_scalar(key, value, iteration)
                checkpoint = None
                if (iteration + 1) % cfg.algo.save_interval == 0 or iteration + 1 == stop:
                    save_checkpoint(run, iteration, learner, cfg)
                    checkpoint = run / f"model_{iteration}.pt"
                panel.update(iteration, stats, checkpoint)
    finally:
        env.close()


class PolicySession:
    """The keyboard viewer protocol, driven only by the two deployable actors."""

    def __init__(self, env, policy):
        self.env, self.policy = env, policy
        self.data = env.scene["robot"].data
        self.command = _command(env)
        self.device = policy.defaults.device
        self.step_count = 0
        self.history = policy.zeros(env.num_envs)

    def reset(self):
        self.env.reset()
        self.history = self.policy.zeros(self.env.num_envs)
        self.step_count = 0

    def advance(self, controls):
        if not controls.consume_step_permission():
            return False
        c = self.command
        commands = c.policy_command()
        inputs = [
            torch.as_tensor(x, dtype=torch.float32, device=self.device)
            for x in (
                self.data.joint_pos,
                self.data.joint_vel,
                self.data.root_link_quat_w,
                self.env.action_manager.action,
                commands,
            )
        ]
        action, plan, raw_plan, self.history = self.policy(*inputs, self.history)
        c.set_body_plan(plan.cpu().numpy(), raw_plan.cpu().numpy())
        self.env.step(action.cpu().numpy())
        self.step_count += 1
        return True

    def physics_state(self):
        return self.env.get_physics_state_snapshot()


def play(path, device=None, speed=1.0, *, keyboard=True):
    from legged_manipulation_unilab.training.keyboard_play import run_keyboard_viewer

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = load_checkpoint(path, "cpu")
    _check_wheeled_policy_layout(ckpt)
    cfg = OmegaConf.create(ckpt["config"])
    if cfg.training.task_name == "L1WArmManipLoco":
        width = ckpt["modules"]["leg_actor"]["normalizer.mean"].numel()
        with open_dict(cfg):
            cfg.env.command_config.enable_gait_phase = width == 64
            if width == 64 and cfg.env.command_config.get("gait_frequency", 0) <= 0:
                cfg.env.command_config.gait_frequency = 1.5
    cfg.interactive.speed = speed
    cfg.training.play_only = True
    env = _make_env(cfg, 1)
    try:
        defaults = torch.tensor(
            env.scene["robot"].data.default_joint_pos[0], dtype=torch.float32, device=device
        )
        policy = DualPolicy(
            defaults,
            cfg.policy.history_length,
            cfg.policy.hidden_dims,
            cfg.stage == "stage2",
            cfg.policy.planner_scale,
            env.cfg.control_config.clip_actions,
            wheel_bounds=wheel_action_bounds(env),
            has_gait_phase=env.command_manager.get_term("task").phase is not None,
            gait_thresholds=(
                (env.cfg.command_config.gait_x_threshold,
                 env.cfg.command_config.gait_lateral_threshold,
                 env.cfg.command_config.gait_yaw_threshold)
                if env.cfg.asset.wheel_joint_names else None
            ),
        ).to(device)
        policy.leg.load_state_dict(ckpt["modules"]["leg_actor"])
        if policy.arm is not None:
            policy.arm.load_state_dict(ckpt["modules"]["arm_actor"])
        run_keyboard_viewer(PolicySession(env, policy.eval()), cfg, keyboard=keyboard)
    finally:
        env.close()
