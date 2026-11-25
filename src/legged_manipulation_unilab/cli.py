"""Compose 2PWBC training and MuJoCo evaluation configurations."""

from __future__ import annotations

import argparse
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

CONF_ROOT = Path(__file__).resolve().parent / "conf" / "wbc2p"
_TASK_ALIASES = {
    "go2": "go2",
    "l1w": "l1w",
    "l1_w_arm_manip_loco": "l1w",
}
_TASK_NAMES = {
    "go2": "Go2ArmManipLoco",
    "l1w": "L1WArmManipLoco",
}


def _resolve_eval_run(value: str | Path) -> Path:
    """Resolve a checkpoint, a training run, or the parent of several runs."""
    path = Path(value).expanduser().resolve()
    if path.is_file():
        return path
    if not path.is_dir():
        raise ValueError(f"Checkpoint source does not exist: {path}")
    if any(path.glob("model_*.pt")):
        return path
    for run in sorted(path.iterdir(), reverse=True):
        if run.is_dir() and any(run.glob("model_*.pt")):
            return run
    raise ValueError(f"No model_*.pt checkpoints found under {path}")


def _override_key(override: str) -> str:
    return override.lstrip("+~").split("=", 1)[0]


def compose_config(
    algo: str,
    sim: str,
    overrides: list[str],
    task: str = "go2",
) -> DictConfig:
    if algo != "wbc2p" or sim != "mujoco":
        raise ValueError("This project trains wbc2p on MuJoCo")
    if task not in _TASK_ALIASES:
        raise ValueError(f"Unknown task: {task}")
    owner = _TASK_ALIASES[task]
    reserved = {"task", "training.task_name", "training.sim_backend", "training.play_only"}
    if any(_override_key(item) in reserved for item in overrides):
        raise ValueError("Use CLI flags to select the task, simulator and train/eval mode")

    OmegaConf.register_new_resolver("wm_stage1", lambda stage: stage == "stage1", replace=True)
    with initialize_config_dir(config_dir=str(CONF_ROOT), version_base="1.3"):
        cfg = compose(config_name="config", overrides=[f"task={owner}/mujoco", *overrides])
    if cfg.training.task_name != _TASK_NAMES[owner] or cfg.training.sim_backend != sim:
        raise ValueError("Overrides must preserve task and simulator identity")
    if cfg.stage == "stage2":
        for name in cfg.env.stage2_disabled_rewards:
            term = cfg.reward.get(name)
            if term is not None:
                term.weight = 0.0
    return cfg


def _main(*, play: bool, argv: list[str] | None = None) -> None:
    description = "Play a 2PWBC checkpoint" if play else "Train Go2 or L1-W in MuJoCo"
    parser = argparse.ArgumentParser(description=description, allow_abbrev=False)
    parser.add_argument("--task", choices=_TASK_ALIASES, default="go2")
    parser.add_argument("--algo", choices=("wbc2p",), default="wbc2p")
    parser.add_argument("--sim", choices=("mujoco",), default="mujoco")
    parser.add_argument("--cfg", action="store_true", help="Print the resolved configuration")
    if play:
        parser.add_argument("--run", help="Checkpoint file, run directory, or parent log group")
        parser.add_argument("--checkpoint", help="Checkpoint iteration or filename")
    args, overrides = parser.parse_known_args(argv)
    for override in overrides:
        if override.startswith("--"):
            parser.error(f"unknown option: {override}")
    cfg = compose_config(args.algo, args.sim, overrides, task=args.task)
    if play:
        for option, key in (("run", "load_run"), ("checkpoint", "checkpoint")):
            value = getattr(args, option)
            if value is None:
                continue
            if any(_override_key(item) == f"algo.{key}" for item in overrides):
                parser.error(f"use --{option} or algo.{key}, not both")
            cfg.algo[key] = value
    cfg.training.play_only = play
    if args.cfg:
        print(OmegaConf.to_yaml(cfg, resolve=True))
        return

    from legged_manipulation_unilab.training import wbc2p

    if play:
        if str(cfg.algo.load_run) in {"", "-1"}:
            parser.error("provide --run or algo.load_run")
        wbc2p.play(
            wbc2p.checkpoint_path(cfg.algo.load_run, cfg.algo.checkpoint),
            cfg.training.device,
            cfg.interactive.speed,
            keyboard=cfg.interactive.keyboard,
        )
    else:
        wbc2p.train(cfg)


def train_main() -> None:
    _main(play=False)


def eval_main() -> None:
    _main(play=True)
