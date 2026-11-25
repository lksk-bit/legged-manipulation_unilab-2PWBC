"""MuJoCo keyboard controls and overlays for 2PWBC playback."""

from __future__ import annotations

import tempfile
import time
from collections import deque
from threading import Lock
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from unilab.visualization.debug_primitives import (
    append_debug_primitives_to_scene,
    quat_from_z_axis,
)
from unilab.visualization.interactive_playback import KeyboardCommander, PlaybackControls
from unisim.backend.base import DebugPrimitive
from unisim.backend.mujoco.playback import resolve_render_play_model_files

from legged_manipulation_unilab.tasks.go2_arm.manager_env import Go2ArmManipLocoCommand, _command

_KEY_ENTER, _KEY_KP_ENTER = 257, 335
_KEY_BACKSPACE = 259
_KEY_RIGHT, _KEY_LEFT, _KEY_DOWN, _KEY_UP = 262, 263, 264, 265
_VELOCITY_STAND_DEADZONE = 0.1
_DEFAULT_KEYBOARD_STEP_LIN = 0.2
_DEFAULT_KEYBOARD_STEP_ANG = 0.2
_DEFAULT_KEYBOARD_STEP_EE = 0.02


class PendingKeyQueue:
    """GLFW key callbacks may fire during ``viewer.sync()``; env work stays on the play loop."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._keys: deque[int] = deque()

    def push(self, keycode: int) -> None:
        with self._lock:
            self._keys.append(int(keycode))

    def drain(self) -> list[int]:
        with self._lock:
            keys = list(self._keys)
            self._keys.clear()
        return keys


class EndEffectorCommander:
    """Mutable EE Cartesian goal in the task frame, driven by keyboard nudges."""

    AXIS_X = 0
    AXIS_Y = 1
    AXIS_Z = 2

    def __init__(self, home: np.ndarray, *, step: float = _DEFAULT_KEYBOARD_STEP_EE) -> None:
        self.home = np.asarray(home, dtype=np.float64).reshape(3).copy()
        self.cart = self.home.copy()
        self.step = float(step)

    @classmethod
    def from_command(
        cls, command: Go2ArmManipLocoCommand, *, step: float = _DEFAULT_KEYBOARD_STEP_EE
    ) -> "EndEffectorCommander":
        home = np.asarray(command._task_cfg.goal_ee.init_ee_cart, dtype=np.float64)
        commander = cls(home, step=step)
        commander.cart[:] = np.asarray(command.curr_ee_goal_cart[0], dtype=np.float64)
        return commander

    def nudge(self, axis: int, sign: float) -> None:
        self.cart[int(axis)] += self.step * (1.0 if sign >= 0 else -1.0)

    def go_home(self) -> None:
        self.cart[:] = self.home

    def describe(self) -> str:
        return (
            f"ee x={self.cart[0]:+.3f} y={self.cart[1]:+.3f} z={self.cart[2]:+.3f} "
            f"(step={self.step:.3f} m)"
        )


def disable_play_terminations(env: Any) -> None:
    """Drop timeout/fallen terms so teleop play does not auto-end the episode.

    Manual reset (Backspace) still works. Autoreset stays off so a future
    termination, if re-enabled, would not silently bounce the robot.
    """
    manager = getattr(env, "termination_manager", None)
    if manager is not None:
        names = list(getattr(manager, "active_terms", []) or [])
        for attr in ("_term_names", "_term_cfgs", "_class_term_cfgs"):
            terms = getattr(manager, attr, None)
            if isinstance(terms, list):
                terms.clear()
        dones = getattr(manager, "_term_dones", None)
        if isinstance(dones, dict):
            dones.clear()
        for buf_name in ("_truncated_buf", "_terminated_buf"):
            buf = getattr(manager, buf_name, None)
            if isinstance(buf, np.ndarray):
                buf[:] = False
        if names:
            print(f"[play] disabled terminations: {', '.join(names)} (Backspace still resets)")
        else:
            print("[play] episode terminations already empty")
    pending = getattr(env, "_manual_reset_pending", None)
    if isinstance(pending, np.ndarray):
        pending[:] = False


def bind_velocity_teleop(env: Any) -> Go2ArmManipLocoCommand:
    """Make the keyboard the velocity source; leave EE resampling untouched."""
    command = _command(env)
    command.enable_velocity_hold()
    command.set_velocity_command(np.zeros(3, dtype=np.float64))
    env.set_autoreset(False)
    disable_play_terminations(env)
    return command


def apply_keyboard_velocity(command: Go2ArmManipLocoCommand, commander: KeyboardCommander) -> None:
    command.set_velocity_command(commander.command)


def apply_keyboard_ee(command: Go2ArmManipLocoCommand, ee_commander: EndEffectorCommander) -> None:
    command.set_ee_goal_cart(ee_commander.cart)
    ee_commander.cart[:] = np.asarray(command.curr_ee_goal_cart[0], dtype=np.float64)


def _home_ee_after_reset(
    command: Go2ArmManipLocoCommand, ee_commander: EndEffectorCommander
) -> None:
    if command.yaw_frame:
        # Read the TCP after MuJoCo restores the MJCF keyframe, before the policy acts.
        position, quat = command.ee_yaw_pose()
        command.set_ee_goal_cart(position[0])
        command.set_ee_goal_quat(quat[0])
        command.post_compute()
        ee_commander.home[:] = command.curr_ee_goal_cart[0]
    ee_commander.go_home()


def push_keyboard_command(
    command: Go2ArmManipLocoCommand,
    commander: KeyboardCommander,
    ee_commander: EndEffectorCommander | None = None,
) -> None:
    apply_keyboard_velocity(command, commander)
    if ee_commander is not None:
        apply_keyboard_ee(command, ee_commander)


def _commander_from_env(env: Any, cfg: DictConfig) -> KeyboardCommander:
    command = _command(env)
    return KeyboardCommander.from_vel_limit(
        command._task_cfg.command_config.vel_limit,
        step_lin=float(
            OmegaConf.select(
                cfg, "interactive.keyboard_step_lin", default=_DEFAULT_KEYBOARD_STEP_LIN
            )
        ),
        step_ang=float(
            OmegaConf.select(
                cfg, "interactive.keyboard_step_ang", default=_DEFAULT_KEYBOARD_STEP_ANG
            )
        ),
    )


def _ee_commander_from_env(
    command: Go2ArmManipLocoCommand, cfg: DictConfig
) -> EndEffectorCommander:
    return EndEffectorCommander.from_command(
        command,
        step=float(
            OmegaConf.select(cfg, "interactive.keyboard_step_ee", default=_DEFAULT_KEYBOARD_STEP_EE)
        ),
    )


def apply_viewer_key(
    keycode: int,
    *,
    controls: PlaybackControls,
    commander: KeyboardCommander,
    session: Any,
    command: Go2ArmManipLocoCommand,
    ee_commander: EndEffectorCommander | None = None,
    keyboard: bool = True,
) -> None:
    if keycode == ord(" "):
        paused = controls.toggle_pause()
        print(f"[keyboard-play] {'paused' if paused else 'resumed'} (space)")
        return
    if keycode in (ord("N"), ord("n")):
        controls.request_single_step()
        if not controls.paused:
            controls.pause()
            print("[keyboard-play] paused for single-step mode (n)")
        print("[keyboard-play] single step requested (n)")
        return
    if keycode in (ord("+"), ord("=")):
        controls.set_speed(controls.speed * 1.25)
        print(f"[keyboard-play] speed={controls.speed:.2f}x")
        return
    if keycode in (ord("-"), ord("_")):
        controls.set_speed(controls.speed / 1.25)
        print(f"[keyboard-play] speed={controls.speed:.2f}x")
        return
    if keycode == _KEY_BACKSPACE:
        commander.zero()
        session.reset()
        if keyboard:
            if ee_commander is not None:
                _home_ee_after_reset(command, ee_commander)
            push_keyboard_command(command, commander, ee_commander)
        print("[keyboard-play] reset (backspace)")
        return
    if not keyboard or _handle_command_key(commander, keycode):
        return
    if ee_commander is not None:
        _handle_ee_key(ee_commander, keycode)


def _handle_command_key(commander: KeyboardCommander, keycode: int) -> bool:
    if keycode == _KEY_UP:
        commander.nudge(commander.AXIS_VX, +1.0)
    elif keycode == _KEY_DOWN:
        commander.nudge(commander.AXIS_VX, -1.0)
    elif keycode == _KEY_LEFT:
        commander.nudge(commander.AXIS_VYAW, +1.0)
    elif keycode == _KEY_RIGHT:
        commander.nudge(commander.AXIS_VYAW, -1.0)
    elif keycode == ord("7"):
        commander.nudge(commander.AXIS_VY, +1.0)
    elif keycode == ord("8"):
        commander.nudge(commander.AXIS_VY, -1.0)
    elif keycode in (_KEY_ENTER, _KEY_KP_ENTER):
        commander.zero()
    else:
        return False
    print(f"[keyboard-play] {commander.describe()}")
    return True


def _handle_ee_key(ee_commander: EndEffectorCommander, keycode: int) -> bool:
    if keycode in (ord("u"), ord("U")):
        ee_commander.nudge(ee_commander.AXIS_X, +1.0)
    elif keycode in (ord("j"), ord("J")):
        ee_commander.nudge(ee_commander.AXIS_X, -1.0)
    elif keycode in (ord("h"), ord("H")):
        ee_commander.nudge(ee_commander.AXIS_Y, +1.0)
    elif keycode in (ord("k"), ord("K")):
        ee_commander.nudge(ee_commander.AXIS_Y, -1.0)
    elif keycode in (ord("y"), ord("Y")):
        ee_commander.nudge(ee_commander.AXIS_Z, +1.0)
    elif keycode in (ord("o"), ord("O")):
        ee_commander.nudge(ee_commander.AXIS_Z, -1.0)
    elif keycode in (ord("l"), ord("L")):
        ee_commander.go_home()
    else:
        return False
    print(f"[keyboard-play] {ee_commander.describe()}")
    return True


def _load_viewer_model(env: Any):
    import mujoco

    try:
        with tempfile.TemporaryDirectory(prefix="legged-keyboard-play-") as tmp_dir:
            model_files = resolve_render_play_model_files(env, num_envs=1, tmp_dir=tmp_dir)
            model_file = model_files[0] if isinstance(model_files, list) else model_files
            return mujoco.MjModel.from_xml_path(str(model_file))
    except Exception as exc:
        print(f"[keyboard-play] visual playback model unavailable ({exc}); using scene XML")
    scene = getattr(getattr(env, "cfg", None), "scene", None)
    model_file = None if scene is None else getattr(scene, "model_file", None)
    if model_file:
        return mujoco.MjModel.from_xml_path(str(model_file))
    playback_model = env.get_playback_model()
    if isinstance(playback_model, str):
        return mujoco.MjModel.from_xml_path(playback_model)
    return playback_model


def _viewer_overlays(command: Go2ArmManipLocoCommand, data: Any) -> list[DebugPrimitive]:
    target = tuple(float(v) for v in command.curr_ee_goal_world[0])
    overlays = [
        DebugPrimitive(kind="sphere", pos=target, size=(0.03,), rgba=(0.9, 0.2, 0.2, 0.8)),
        DebugPrimitive(
            kind="frame",
            pos=target,
            quat=tuple(float(v) for v in command.ee_goal_quat_world[0]),
            size=(0.12,),
        ),
    ]
    vx, vy = command.vel_command_b[0, :2]
    speed = float(np.hypot(vx, vy))
    if speed > _VELOCITY_STAND_DEADZONE:
        yaw = data.heading_w[0]
        cos, sin = np.cos(yaw), np.sin(yaw)
        overlays.append(
            DebugPrimitive(
                kind="arrow",
                pos=tuple(float(v) for v in data.root_link_pos_w[0]),
                quat=quat_from_z_axis((cos * vx - sin * vy, sin * vx + cos * vy, 0.0)),
                size=(0.3 + 0.4 * speed,),
                rgba=(0.2, 0.5, 0.9, 0.8),
            )
        )
    return overlays


def run_keyboard_viewer(session: Any, cfg: DictConfig, *, keyboard: bool = True) -> None:
    import mujoco
    import mujoco.viewer

    env = session.env
    if keyboard:
        command = bind_velocity_teleop(env)
    else:
        command = _command(env)
        env.set_autoreset(False)
        disable_play_terminations(env)
    commander = _commander_from_env(env, cfg)
    ee_commander = None
    if keyboard and not env.cfg.arm_stage.freeze_arm_joints:
        command.enable_ee_hold()
        ee_commander = _ee_commander_from_env(command, cfg)
    if keyboard:
        apply_keyboard_velocity(command, commander)
    if ee_commander is not None:
        apply_keyboard_ee(command, ee_commander)
    controls = PlaybackControls(
        paused=bool(OmegaConf.select(cfg, "interactive.start_paused", default=False)),
        speed=float(OmegaConf.select(cfg, "interactive.speed", default=1.0)),
    )
    mj_model = _load_viewer_model(env)
    viz_data = mujoco.MjData(mj_model)
    state_spec = mujoco.mjtState.mjSTATE_FULLPHYSICS
    ctrl_dt = float(getattr(env.cfg, "ctrl_dt", 0.02))

    session.reset()
    if keyboard:
        if ee_commander is not None:
            _home_ee_after_reset(command, ee_commander)
        push_keyboard_command(command, commander, ee_commander)
    pending_keys = PendingKeyQueue()

    def _on_key(keycode: int) -> None:
        pending_keys.push(keycode)

    if keyboard:
        print("[keyboard-play] Keyboard teleop ENABLED:")
        print("  Up / Down    : chassis forward / backward (vx)")
        print("  Left / Right : chassis turn left / right (vyaw)")
        print("  7 / 8        : chassis strafe left / right (vy; main keyboard)")
        print("  Enter        : chassis full stop")
        if ee_commander is not None:
            print("  U / J        : EE +x / -x (yaw-aligned frame)")
            print("  H / K        : EE +y / -y")
            print("  Y / O        : EE +z / -z")
            print("  L            : EE home (MJCF initial pose)")
        else:
            print("  Arm keys     : disabled for this checkpoint")
    print("  Backspace    : reset episode (zeros velocity, homes EE)")
    print("  Space        : pause / resume")
    print("  Timeout / fallen terminations are disabled; only Backspace resets.")
    print(
        f"  Linear step is {commander.step_lin:.2f} m/s "
        f"(gait/policy treat |cmd| <= {_VELOCITY_STAND_DEADZONE} as stand)."
    )
    if ee_commander is not None:
        print(f"  EE step is {ee_commander.step:.3f} m, clipped to the spherical workspace.")

    with mujoco.viewer.launch_passive(mj_model, viz_data, key_callback=_on_key) as viewer:
        with torch.inference_mode():
            while viewer.is_running():
                t0 = time.perf_counter()
                for keycode in pending_keys.drain():
                    apply_viewer_key(
                        keycode,
                        controls=controls,
                        commander=commander,
                        session=session,
                        command=command,
                        ee_commander=ee_commander,
                        keyboard=keyboard,
                    )
                if keyboard:
                    push_keyboard_command(command, commander, ee_commander)
                session.advance(controls)
                phys = session.physics_state()[0].astype(np.float64)
                mujoco.mj_setState(mj_model, viz_data, phys, state_spec)
                mujoco.mj_forward(mj_model, viz_data)
                viewer.user_scn.ngeom = 0
                append_debug_primitives_to_scene(
                    viewer.user_scn, _viewer_overlays(command, session.data)
                )
                viewer.sync()
                sleep_s = (ctrl_dt / max(float(controls.speed), 1e-6)) - (time.perf_counter() - t0)
                if sleep_s > 0.0:
                    time.sleep(sleep_s)
