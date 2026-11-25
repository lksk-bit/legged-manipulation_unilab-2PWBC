"""Focused regression tests for the new cross-policy control contract."""

from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch
from hydra.utils import get_class
from omegaconf import OmegaConf

from legged_manipulation_unilab.algos.wbc2p.bridge import (
    DQ,
    QUAT,
    STATE_DIM,
    Q,
    TaskBridge,
)
from legged_manipulation_unilab.algos.wbc2p.geometry import (
    ee_tracking,
    multiply,
    rotation_vector,
    yaw_pose,
    yaw_quat,
)
from legged_manipulation_unilab.algos.wbc2p.learner import Learner
from legged_manipulation_unilab.algos.wbc2p.replay import Replay
from legged_manipulation_unilab.algos.wbc2p.task import Task
from legged_manipulation_unilab.cli import CONF_ROOT, compose_config
from legged_manipulation_unilab.training.wbc2p import (
    Collector,
    PolicySession,
    _make_env,
    load_checkpoint,
    resume,
    save_checkpoint,
)


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    monkeypatch.setenv("LEGGED_MANIPULATION_ASSET_CACHE", str(tmp_path / "assets"))
    cfg = compose_config(
        "wbc2p",
        "mujoco",
        [
            "world_model.hidden_dims=[32,32]",
            "policy.hidden_dims=[24,24]",
            "policy.history_length=4",
            "bptt.horizon=4",
            "bptt.critic_epochs=1",
        ],
        task="go2",
    )
    env = _make_env(cfg, 4)
    try:
        env.set_autoreset(False)
        env.reset(seed=915)
        yield cfg, env
    finally:
        env.close()


def test_go2_stage1_uses_task_raibert_weight():
    profile = OmegaConf.load(CONF_ROOT / "task" / "go2" / "rewards.yaml")
    stage1 = compose_config("wbc2p", "mujoco", ["stage=stage1"], task="go2")
    stage2 = compose_config(
        "wbc2p",
        "mujoco",
        ["stage=stage2", "reward.raibert_heuristic.weight=-10.0"],
        task="go2",
    )
    assert stage1.reward.raibert_heuristic.weight == profile.reward.raibert_heuristic.weight
    assert stage2.reward.raibert_heuristic.weight == 0.0


def test_reward_profiles_are_independent():
    go2 = compose_config("wbc2p", "mujoco", ["reward.base_height.weight=-7"], task="go2")
    l1w = compose_config("wbc2p", "mujoco", [], task="l1w")
    assert go2.reward.base_height.weight == -7
    assert l1w.reward.base_height.weight == -10
    assert go2.reward.feet_slip.func != l1w.reward.feet_slip.func
    assert go2.env.imagined_reward_class != l1w.env.imagined_reward_class
    assert "wheel_action_rate" not in go2.reward
    assert "raibert_heuristic" not in l1w.reward


def test_go2_attitude_command_relaxes_penalties_in_both_rollouts(runtime):
    _, env = runtime
    bridge = TaskBridge(env)
    task = Task(bridge, False, 0.08)
    parameters = env.cfg.reward_parameters
    parameters.only_positive_rewards_ji22_style = False
    plan = torch.tensor([[0., 0.], [0.3, 0.], [0., -0.3], [0.3, -0.3]], requires_grad=True)
    plan_np = plan.detach().numpy()
    decay = torch.exp(-(plan.detach() / 0.25).square().sum(-1))
    env.step(np.full((4, 18), 0.25, dtype=np.float32))
    state = torch.from_numpy(bridge.state())
    command = torch.from_numpy(bridge.current_command())
    memory = torch.from_numpy(bridge.controller())
    zero_plan = torch.zeros_like(plan)

    for name, minimum in {
        "lin_vel_z": 0.5, "ang_vel_xy": 0.7, "base_height": 0.3, "leg_pose": 0.15,
    }.items():
        expected = minimum + (1 - minimum) * decay
        term = env.reward_manager.get_term_cfg(name)
        bridge.command.set_body_plan(zero_plan.numpy(), zero_plan.numpy())
        baseline = term.func(env, **term.params)
        assert np.all(baseline > 0), name
        bridge.command.set_body_plan(plan_np, zero_plan.numpy())
        attenuated = term.func(env, **term.params)
        np.testing.assert_allclose(attenuated, baseline * expected.numpy(), rtol=1e-6)

        task.scales = {name: -1.0}
        baseline, _ = task.rewards(state, state, command, zero_plan, memory, memory)
        attenuated, _ = task.rewards(state, state, command, plan, memory, memory)
        torch.testing.assert_close(attenuated, baseline * expected)
        gradient, = torch.autograd.grad(attenuated.sum(), plan)
        torch.testing.assert_close(gradient[0], torch.zeros(2))
        assert gradient[1, 0] > 0
        assert gradient[2, 1] < 0


@pytest.mark.parametrize("robot", ["go2", "l1w"])
@pytest.mark.parametrize("stage", ["stage1", "stage2"])
def test_initial_ee_goal_uses_refreshed_reset_pose(robot, stage, tmp_path, monkeypatch):
    monkeypatch.setenv("LEGGED_MANIPULATION_ASSET_CACHE", str(tmp_path / "assets"))
    cfg = compose_config("wbc2p", "mujoco", [f"stage={stage}"], task=robot)
    env = _make_env(cfg, 2, collector_observations=True)
    try:
        command = env.command_manager.get_term("task")
        for seed in (202, 203):
            env.reset(seed=seed)
            np.testing.assert_allclose(
                command.curr_ee_goal_cart, command.ee_yaw_pose()[0], atol=1e-5
            )
            assert not command._pending_ee_reset.any()
            for _ in range(3):
                env.step(np.zeros((2, len(env.scene["robot"].joint_names)), dtype=np.float32))
                if stage == "stage1":
                    np.testing.assert_allclose(
                        command.curr_ee_goal_cart, command.ee_yaw_pose()[0], atol=1e-5
                    )
        bridge = TaskBridge(env)
        schedule = bridge.command_schedule(
            bridge.command_state(), 3, np.random.default_rng(202)
        )
        assert np.isfinite(schedule).all()
    finally:
        env.close()


@pytest.mark.parametrize("robot", ["go2", "l1w"])
def test_mirrorup_matches_reference_and_preserves_gradients(robot, tmp_path, monkeypatch):
    monkeypatch.setenv("LEGGED_MANIPULATION_ASSET_CACHE", str(tmp_path / "assets"))
    cfg = compose_config("wbc2p", "mujoco", [], task=robot)
    env = _make_env(cfg, 4, collector_observations=True)
    try:
        env.reset(seed=915)
        bridge = TaskBridge(env)
        task = Task(bridge, False, 0.08)
        # Isolate this term from the positive-reward shaping and other terms.
        task.cfg.reward_parameters.only_positive_rewards_ji22_style = False
        task.scales = {"mirrorup": -0.05 * env.step_dt}
        state = torch.from_numpy(bridge.state())
        joint = np.zeros_like(bridge.robot.data.joint_pos)
        rear_right, rear_left = (slice(9, 12), slice(6, 9)) if robot == "go2" else (
            slice(6, 9), slice(9, 12)
        )
        joint[:, :3] = [0.1, 0.8, -1.5]
        joint[:, 3:6] = [-0.2, 1.0, -1.3]
        joint[:, rear_right] = joint[:, :3] + [0.2, 0.3, 0.4]
        joint[:, rear_left] = joint[:, 3:6] + [0.1, 0.2, 0.3]
        joint[:, 12:] = 3.0  # Wheels and arm must not enter the symmetry error.
        up_z = np.array([1.0, 0.5, -1.0, 0.0])
        expected = 0.5 * (0.2**2 + 0.3**2 + 0.4**2 + 0.1**2 + 0.2**2 + 0.3**2)
        expected = expected * np.clip(up_z, 0, 1)
        term = env.reward_manager.get_term_cfg("mirrorup").func
        sensors = term._views.read().copy()
        sensors[:, term._slices["upvector"]] = np.column_stack(
            (np.zeros(4), np.sqrt(1 - up_z**2), up_z)
        )
        monkeypatch.setattr(term, "_views", SimpleNamespace(read=lambda: sensors))
        monkeypatch.setattr(term, "_entity", SimpleNamespace(data=SimpleNamespace(
            joint_pos=joint,
            joint_vel=bridge.robot.data.joint_vel,
            default_joint_pos=bridge.robot.data.default_joint_pos,
            root_link_pos_w=bridge.robot.data.root_link_pos_w,
        )))
        np.testing.assert_allclose(term(env, name="mirrorup"), expected, atol=1e-6)
        state[:, bridge.layout.q] = torch.from_numpy(joint)
        tilt = torch.zeros(4, 3)
        tilt[:, 0] = torch.from_numpy(np.arccos(up_z)).float()
        state[:, QUAT] = rotation_vector(tilt)
        state.requires_grad_()
        memory = torch.from_numpy(bridge.controller())
        command = torch.from_numpy(bridge.current_command())
        reward, _ = task.rewards(
            state, state, command, torch.zeros(4, 2), memory, memory
        )
        torch.testing.assert_close(
            reward / task.scales["mirrorup"], torch.tensor(expected).float(), atol=1e-6, rtol=1e-5
        )
        gradient = torch.autograd.grad(reward.sum(), state)[0]
        assert torch.isfinite(gradient).all()
        assert gradient[0, bridge.layout.q][:12].abs().sum() > 0
        assert gradient[1, QUAT].abs().sum() > 0
        assert torch.count_nonzero(gradient[:, bridge.layout.q][:, 12:]) == 0
    finally:
        env.close()


def test_fk_yaw_frame_matches_mujoco_and_retains_tilt(runtime):
    cfg, env = runtime
    bridge = TaskBridge(env)
    task = Task(bridge, False, 0.08).double()
    assert type(task.robot_rewards) is get_class(cfg.reward_implementation.imagined)
    assert type(env.reward_manager.get_term_cfg("feet_slip").func) is get_class(
        cfg.reward_implementation.mujoco
    )
    model = mujoco.MjModel.from_xml_path(str(env.cfg.scene.model_file))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    joint = torch.tensor([data.joint(name).qpos.item() for name in bridge.robot.joint_names])
    root = torch.tensor(data.qpos[:3].copy())
    goal = root.new_tensor([0.4, 0.1, 0.14])
    angles = root.new_tensor([0.2, -0.1, 0.1])

    def measure(roll, pitch, yaw):
        tilt = torch.tensor([roll, pitch, 0.0], dtype=root.dtype, requires_grad=True)
        quat = multiply(yaw_quat(root.new_tensor(yaw)), rotation_vector(tilt))
        pos, orn = task.fk(root, quat, joint.to(root))
        data.qpos[3:7] = quat.detach().numpy()
        mujoco.mj_forward(model, data)
        for i, name in enumerate(("endpoint", "FL", "FR", "RL", "RR", "imu")):
            np.testing.assert_allclose(pos[i].detach(), data.site(name).xpos, atol=1e-7)
            oracle = np.zeros(4)
            mujoco.mju_mat2Quat(oracle, data.site(name).xmat)
            assert abs(float(orn[i].detach() @ torch.from_numpy(oracle))) > 0.999999
        p, q = yaw_pose(root, quat, pos[0], orn[0], task.reference_height)
        reward = ee_tracking(p, q, goal, angles, env.cfg.reward_parameters)
        grad = torch.autograd.grad(reward, tilt)[0]
        return p.detach(), q.detach(), reward.detach(), grad

    home = measure(0.0, 0.0, 0.0)
    for roll, pitch in ((0.15, 0.0), (0.0, -0.2)):
        tilted = measure(roll, pitch, 0.7)
        assert (tilted[0] - home[0]).norm() > 0.005
        assert (tilted[2] - home[2]).abs() > 0.0001
        assert tilted[3][0 if roll else 1].abs() > 0.0001
    torch.testing.assert_close(measure(0.15, -0.2, 0.8)[0], measure(0.15, -0.2, 0.0)[0])
    actual_pos, actual_quat = (
        torch.as_tensor(x, dtype=torch.float32) for x in bridge.command.ee_yaw_pose()
    )
    fk_pos, fk_quat = task.float().ee(torch.from_numpy(bridge.state()))
    torch.testing.assert_close(fk_pos, actual_pos, atol=3e-6, rtol=0)
    assert ((fk_quat * actual_quat).sum(-1).abs() > 0.99999).all()

    term = env.reward_manager.get_term_cfg("arm_manip_commands_tracking_combine")
    real = term.func(env, **term.params)
    imagined = ee_tracking(
        actual_pos,
        actual_quat,
        torch.from_numpy(bridge.command.curr_ee_goal_cart),
        torch.from_numpy(bridge.command.goal_tool_angles()),
        env.cfg.reward_parameters,
    )
    np.testing.assert_allclose(real, imagined.detach().numpy(), atol=2e-6)


def test_two_stage_transitions_and_planner_only_ee_gradient(runtime, tmp_path):
    cfg, env = runtime
    torch.manual_seed(13)
    bridge = TaskBridge(env)
    task = Task(bridge, False, 0.08)
    first = Learner(task, bridge, cfg, "cpu")
    command = torch.from_numpy(bridge.current_command())
    state = torch.from_numpy(bridge.state())
    controller = torch.from_numpy(bridge.controller())
    ctx = torch.from_numpy(bridge.context())
    assert command[:, 3:5].abs().sum() > 0
    with torch.no_grad():
        action, plan, raw, frames = first.policy(
            state[:, Q],
            state[:, DQ],
            state[:, QUAT],
            controller[:, :18],
            command,
            first.policy.zeros(4),
        )
        torch.testing.assert_close(frames[0][:, -1, 42:44], command[:, 3:5])
        assert not action[:, 12:].any()
        expected_target, _ = task.control(action, controller, command, raw)
        bridge.command.set_body_plan(plan.numpy(), raw.numpy())
        env.step(action.numpy())
        torch.testing.assert_close(expected_target, torch.from_numpy(bridge.action._target))

    replay = Replay(12, env.num_envs, "cpu", 42)
    collector = Collector(first, replay, env, bridge, cfg)
    collector.collect(7)
    dynamics_batch = replay.sample(3, 8)
    assert dynamics_batch["state"].shape == (3, 8, STATE_DIM)
    assert first.train_dynamics(dynamics_batch)["WorldModel/loss"].isfinite()
    assert first.train_policies(replay.sample(1, 8, 4))["Loss/leg_actor"].isfinite()
    save_checkpoint(tmp_path, 0, first, cfg)
    checkpoint = load_checkpoint(tmp_path / "model_0.pt", "cpu")

    cfg.stage = "stage2"
    second_env = _make_env(cfg, 4, collector_observations=True)
    try:
        second_env.set_autoreset(False)
        second_env.reset(seed=916)
        assert second_env.observation_manager.active_terms == {"actor": ["last_action"]}
        bridge = TaskBridge(second_env)
        task = Task(bridge, True, 0.08)
        second = Learner(task, bridge, cfg, "cpu")
        assert resume(second, checkpoint, cfg) == 1
        for name in ("world_model", "leg_actor", "leg_critic"):
            for p1, p2 in zip(
                first.modules[name].parameters(), second.modules[name].parameters(), strict=True
            ):
                torch.testing.assert_close(p1, p2)

        # Force the numerical check through the planner and leg action only.
        # The six arm actions are held constant in the transition.
        state = torch.from_numpy(bridge.state())
        command = torch.from_numpy(bridge.current_command())
        controller = torch.from_numpy(bridge.controller())
        ctx = torch.from_numpy(bridge.context())

        def policy():
            return second.policy(
                state[:, Q],
                state[:, DQ],
                state[:, QUAT],
                controller[:, :18],
                command,
                second.policy.zeros(4),
            )

        neutral_action = policy()[0]
        with torch.no_grad():
            second.policy.arm.head[-1].bias[6:] += 0.1
        action, plan, raw, history = policy()
        assert not torch.allclose(neutral_action[:, :12], action[:, :12])
        torch.testing.assert_close(history[0][:, -1, 42:44], plan)
        assert action.shape == (4, 18)
        test_action = torch.cat((action[:, :12], torch.zeros_like(action[:, 12:])), dim=-1)
        target, _ = task.control(test_action, controller, command, raw)
        second.world.requires_grad_(False)
        nxt = second.world(state, target, ctx)
        ee_pos, ee_q = task.ee(nxt)
        loss = -ee_tracking(
            ee_pos, ee_q, command[:, 9:12], command[:, 12:15], second_env.cfg.reward_parameters
        ).mean()
        leg_grad, plan_grad, actor_grad = torch.autograd.grad(
            loss,
            (action, plan, second.policy.arm.head[-1].bias),
        )
        assert leg_grad[:, :12].abs().sum() > 0
        assert plan_grad.abs().sum() > 0
        assert actor_grad[6:].abs().sum() > 0
        assert not actor_grad[:6].any()

        replay = Replay(12, second_env.num_envs, "cpu", 45)
        collector = Collector(second, replay, second_env, bridge, cfg)
        collect_stats = collector.collect(7)
        assert collect_stats["Perf/env_physics_time"] > 0
        assert collect_stats["Perf/env_update_state_time"] > 0
        second.train_dynamics(replay.sample(3, 8))
        before = {
            name: second.modules[f"{name}_actor"].head[-1].weight.detach().clone()
            for name in second.names
        }
        policy_stats = second.train_policies(replay.sample(1, 8, 4))
        for name in ("rollout", "leg_backward", "arm_backward", "critic"):
            assert policy_stats[f"Perf/bptt_{name}_time"] > 0
        for name, weight in before.items():
            assert not torch.equal(weight, second.modules[f"{name}_actor"].head[-1].weight)
        session = PolicySession(second_env, second.policy)
        session.reset()
        from unilab.visualization.interactive_playback import PlaybackControls

        with torch.inference_mode():
            assert session.advance(PlaybackControls())
        assert second_env.action_manager.action.shape == (4, 18)
    finally:
        second_env.close()


def test_l1w_wheel_targets_and_planner_gradient(tmp_path, monkeypatch):
    monkeypatch.setenv("LEGGED_MANIPULATION_ASSET_CACHE", str(tmp_path / "assets"))
    cfg = compose_config(
        "wbc2p",
        "mujoco",
        ["stage=stage2", "world_model.hidden_dims=[32,32]", "policy.hidden_dims=[24,24]"],
        task="l1_w_arm_manip_loco",
    )
    env = _make_env(cfg, 4, collector_observations=True)
    try:
        env.set_autoreset(False)
        env.reset(seed=916)
        bridge = TaskBridge(env)
        layout = bridge.layout
        task = Task(bridge, True, 0.08)
        learner = Learner(task, bridge, cfg, "cpu")
        state = torch.from_numpy(bridge.state())
        context = torch.from_numpy(bridge.context())
        memory = torch.from_numpy(bridge.controller())
        command = torch.from_numpy(bridge.current_command())
        assert state.shape == (4, 120)
        assert context.shape == (4, 103)
        assert bridge.command.phase.shape == (4,)
        assert bridge.command.feet_phase.shape == (4, 4)
        assert command.shape == (4, 15)
        bridge.command.enable_velocity_hold()
        bridge.command.vel_command_b[:] = [
            [0.5, 0, 0], [0, 0.2, 0], [0, 0, 0.4], [1.2, 0, 0],
        ]
        bridge.command.phase[:] = 0.1
        bridge.command._update_command(None)
        np.testing.assert_array_equal(
            bridge.command._gait_is_active(bridge.command.vel_command_b),
            [False, True, True, True],
        )
        assert bridge.command.phase[0] == 0
        np.testing.assert_allclose(bridge.command.phase[1:], 0.13, atol=1e-6)
        np.testing.assert_array_equal(bridge.command.clock_inputs()[0], np.zeros(4))
        command = torch.from_numpy(bridge.current_command())
        schedule = bridge.command_schedule(
            bridge.command_state()[:2], 3, np.random.default_rng(916)
        )
        assert schedule.shape == (3, 2, 15)
        np.testing.assert_allclose(schedule[0], command.numpy()[:2])
        np.testing.assert_allclose(schedule[1, 0, 5:9], [0.5, 0, 0, 0.5], atol=1e-6)
        np.testing.assert_allclose(
            schedule[1, 1, 5:9], (schedule[0, 1, 5:9] + 0.03) % 1.0, atol=1e-6
        )
        small_lateral = command.new_tensor([[0, 0.09, 0]])
        assert bridge.command._command_is_moving(small_lateral.numpy())[0]
        assert not task.stationary(small_lateral)[0]
        assert "raibert_heuristic" not in env.reward_manager.active_terms
        assert "tracking_contacts_shaped_force" not in env.reward_manager.active_terms
        assert "feet_clearance_cmd_linear" not in task.scales
        assert task.scales["base_height"] == pytest.approx(-10 * env.step_dt)
        assert type(task.robot_rewards) is get_class(cfg.reward_implementation.imagined)
        assert type(env.reward_manager.get_term_cfg("wheel_action_rate").func) is get_class(
            cfg.reward_implementation.mujoco
        )
        term = env.reward_manager.get_term_cfg("base_height")
        np.testing.assert_allclose(
            term.func(env, **term.params), (state[:, 2].numpy() - 0.30) ** 2
        )
        assert learner.world.net[0].in_features == 179
        assert learner.world.net[-1].out_features == 91

        action, plan, raw, history = learner.policy(
            state[..., layout.q], state[..., layout.dq], state[..., QUAT],
            memory[..., :22], command, learner.policy.zeros(4)
        )
        assert action.shape == (4, 22)
        assert history[0].shape == (4, 30, 64)
        torch.testing.assert_close(history[0][:, -1, 50:52], plan)
        torch.testing.assert_close(
            history[0][:, -1, -4:], torch.from_numpy(bridge.command.clock_inputs())
        )
        kin = task.kinematics(state)
        force = state[..., layout.force].unflatten(-1, (4, 3)).norm(dim=-1)
        terms = task.robot_rewards(
            state, command, kin[0][:, 1:5], kin[1][:, 1:5],
            yaw_quat(state.new_zeros(4)), force, torch.ones_like(force),
            torch.zeros(4, 22),
        )
        assert terms["swing_feet_z"][0] == 0
        assert torch.all(terms["swing_feet_z"][1:] > 0)

        # Restrict the EE loss to the wheel actions to isolate the cross-policy path.
        wheel_action = action[..., layout.wheels]
        wheel_only = torch.cat(
            (torch.zeros_like(action[..., :12]), wheel_action, torch.zeros_like(action[..., -6:])),
            dim=-1,
        )
        target, _ = task.control(wheel_only, memory, command, raw)
        next_state = learner.world(state, target, context)
        tcp, _ = task.ee(next_state)
        planner_bias = learner.policy.arm.head[-1].bias
        (gradient,) = torch.autograd.grad((tcp - command[..., -6:-3]).square().mean(), planner_bias)
        assert gradient[6:].abs().sum() > 0
        assert not gradient[:6].any()

        with torch.no_grad():
            shifted = state.clone()
            shifted[..., layout.q][..., layout.wheels] += 128.0
            shifted_next = learner.world(shifted, target, context)
            expected = next_state.clone()
            expected[..., layout.q][..., layout.wheels] += 128.0
            torch.testing.assert_close(shifted_next, expected, atol=0, rtol=0)

            chosen = torch.zeros(4, 22)
            chosen[:, layout.wheels] = torch.tensor([0.2, -0.3, 0.4, -0.5])
            expected_target, _ = task.control(chosen, memory, command, torch.zeros(4, 2))
            env.step(chosen.numpy())
            torch.testing.assert_close(torch.from_numpy(bridge.action._target), expected_target)
            torch.testing.assert_close(
                torch.from_numpy(env._backend.get_state("ctrl")["ctrl"]),
                expected_target,
            )
            pos, quat = bridge.command.ee_yaw_pose()
            predicted_pos, predicted_quat = task.ee(torch.from_numpy(bridge.state()))
            np.testing.assert_allclose(pos, predicted_pos, atol=3e-6)
            assert ((torch.from_numpy(quat) * predicted_quat).sum(-1).abs() > 0.99999).all()
    finally:
        env.close()


def test_l1w_rolling_reward_support_order_and_gradients(tmp_path, monkeypatch):
    from legged_manipulation_unilab.tasks.l1_w_arm.rewards import WHEEL_Y

    monkeypatch.setenv("LEGGED_MANIPULATION_ASSET_CACHE", str(tmp_path / "assets"))
    cfg = compose_config("wbc2p", "mujoco", [], task="l1w")
    env = _make_env(cfg, 4, collector_observations=True)
    try:
        env.reset(seed=202)
        bridge = TaskBridge(env)
        task = Task(bridge, False, 0.08)
        c = bridge.command
        c.vel_command_b[:] = [[0.5, 0, 0], [0, 0.25, 0], [0, 0, 0.5], [0, 0, 0]]
        c.phase[:] = 0.2
        c._write_feet_phase(slice(None))
        command = torch.from_numpy(c.policy_command())
        state = torch.from_numpy(bridge.state())
        targets = (
            c.vel_command_b[:, 0:1] - c.vel_command_b[:, 2:3] * np.asarray(WHEEL_Y)
        ) / env.cfg.asset.wheel_radius
        velocity = np.zeros((4, 22), dtype=np.float32)
        velocity[:, 12:16] = targets
        force = np.zeros((4, 4, 3), dtype=np.float32)
        force[..., 2] = 50
        term = env.reward_manager.get_term_cfg("wheel_roll").func
        monkeypatch.setattr(term, "_entity", SimpleNamespace(data=SimpleNamespace(
            joint_vel=velocity
        )))
        real = term._robot_reward(env, "wheel_roll", c, force, None, None)
        np.testing.assert_allclose(real, [1, 0, 1, 0], atol=1e-6)
        # Only rear-left support. Sensor RBL=2 must select actuator RBL=3.
        force[:] = 0
        force[:, 2, 2] = 50
        velocity[0, 15] += 10
        real = term._robot_reward(env, "wheel_roll", c, force, None, None)
        assert real[0] == pytest.approx(0.5, abs=1e-6)
        force[:] = 0
        np.testing.assert_array_equal(
            term._robot_reward(env, "wheel_roll", c, force, None, None), np.zeros(4)
        )
        state[:, bridge.layout.dq] = torch.from_numpy(velocity)
        state.requires_grad_()
        feet = state.new_zeros(4, 4, 3)
        feet[..., 2] = -0.02
        quat = state.new_zeros(4, 4, 4)
        quat[..., 0] = 1
        yaw = quat[:, 0]

        def reward(positions):
            return task.robot_rewards(
                state, command, positions, quat, yaw,
                state.new_full((4, 4), 50),
                state.new_ones(4, 4), state.new_zeros(4, 22),
            )["wheel_roll"]

        rolling = reward(feet)
        assert rolling[0] < rolling[2]
        assert rolling[1] == 0 and rolling[3] == 0
        gradient = torch.autograd.grad(rolling.sum(), state)[0]
        assert gradient[0, bridge.layout.dq][15].abs() > 0
        airborne = feet.clone()
        airborne[..., 2] = 0.5
        assert reward(airborne).abs().max() < 1e-10
        # Low-speed commanded rolling is not a wheel standstill condition.
        c.vel_command_b[:] = [0.05, 0, 0]
        still = env.reward_manager.get_term_cfg("wheel_stand_still").func
        monkeypatch.setattr(still, "_entity", term._entity)
        np.testing.assert_array_equal(
            still._robot_reward(env, "wheel_stand_still", c, force, None, None), np.zeros(4)
        )
    finally:
        env.close()


def test_replay_respects_reset_and_ring_wrap():
    replay = Replay(5, 1, "cpu", 0)
    for i in range(11):
        x = torch.tensor([[float(i)]])
        replay.append(
            torch.tensor([i == 8]),
            state=x,
            next_state=x + 1,
            target=x,
            context=x,
            leg_frame=x,
            arm_frame=x,
        )
    sample = replay.sample(3, 32)
    assert set(sample["state"][0, :, 0].tolist()) == {6.0}
    history = replay.sample(1, 32, 4)
    start_at_reset = history["state"][0, :, 0] == 9
    assert start_at_reset.any()
    assert not history["leg_frame"][start_at_reset].any()


def test_imagined_command_clocks_resample_without_touching_real_task(runtime):
    _, env = runtime
    bridge = TaskBridge(env)
    command = bridge.command
    command.time_left[:] = env.step_dt / 2
    snapshot = bridge.command_state()
    result = bridge.command_schedule(snapshot, 3, np.random.default_rng(7))
    np.testing.assert_allclose(result[0], bridge.current_command())
    assert not np.allclose(result[1, :, :5], result[0, :, :5])
    np.testing.assert_allclose(result[2, :, :5], result[1, :, :5])
    np.testing.assert_allclose(bridge.command_state(), snapshot)


def test_replay_chunk_matches_single_writes_after_ring_wrap():
    sequential, chunked = [Replay(5, 2, "cpu", 4) for _ in range(2)]
    state = torch.arange(16, dtype=torch.float32).reshape(8, 2, 1)
    done = torch.zeros(8, 2, dtype=torch.bool)
    done[2, 0] = done[5, 1] = True
    for x, terminal in zip(state, done, strict=True):
        sequential.append(terminal, state=x)
    # Writes longer than capacity must discard only the oldest transitions.
    chunked.append_many(done[:1], state=state[:1])
    chunked.append_many(done[1:], state=state[1:])
    assert sequential.cursor == chunked.cursor == 8
    for k in sequential.data:
        torch.testing.assert_close(sequential.data[k], chunked.data[k])
    torch.testing.assert_close(sequential.episode, chunked.episode)
