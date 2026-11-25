"""Command-line and Hydra contracts for the two 2PWBC tasks."""

import pytest
from omegaconf import OmegaConf

from legged_manipulation_unilab.cli import _main, compose_config
from legged_manipulation_unilab.interactive import main as interactive_main
from legged_manipulation_unilab.training import wbc2p


@pytest.mark.parametrize(
    ("task", "registered_name"),
    [("go2", "Go2ArmManipLoco"), ("l1w", "L1WArmManipLoco")],
)
def test_wm_task_entrypoint(task, registered_name, capsys, monkeypatch):
    monkeypatch.setattr(wbc2p, "train", lambda cfg: pytest.fail("Unexpected training"))
    _main(play=False, argv=["--task", task, "--cfg", "stage=stage2", "algo.utd=0.4"])
    cfg = OmegaConf.create(capsys.readouterr().out)
    assert cfg.training.task_name == registered_name
    assert cfg.training.sim_backend == "mujoco"
    assert cfg.algo.algo == "wbc2p"
    assert cfg.algo.utd == 0.4
    assert cfg.env.arm_stage.freeze_arm_joints is False
    assert set(cfg.algo) == {
        "algo",
        "algo_log_name",
        "seed",
        "num_envs",
        "num_steps_per_env",
        "utd",
        "max_iterations",
        "save_interval",
        "load_run",
        "checkpoint",
    }


def test_wm_training_rejects_evaluation_options():
    with pytest.raises(SystemExit) as result:
        _main(play=False, argv=["--run", "irrelevant"])
    assert result.value.code == 2


@pytest.mark.parametrize("interface", ["legged-eval", "2pwbc-play"])
def test_wm_playback_selects_checkpoint(interface, tmp_path, monkeypatch):
    run = tmp_path / "run with spaces"
    run.mkdir()
    for name in ("model_9.pt", "model_10.pt"):
        (run / name).touch()
    calls = []
    monkeypatch.setattr(wbc2p, "play", lambda *args, **kwargs: calls.append((args, kwargs)))
    if interface == "legged-eval":
        _main(
            play=True, argv=["--run", str(run), "training.device=cpu", "interactive.keyboard=true"]
        )
    else:
        interactive_main(["--run", str(run), "--device", "cpu"])
    assert calls == [((run / "model_10.pt", "cpu", 1.0), {"keyboard": True})]


def test_wm_identity_and_backend_are_fixed():
    with pytest.raises(ValueError, match="identity"):
        compose_config("wbc2p", "mujoco", ["training={task_name:Unknown}"])
    with pytest.raises(ValueError, match="MuJoCo"):
        compose_config("wbc2p", "motrix", [])
