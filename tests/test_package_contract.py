"""The installed WM package registers both robots and carries their meshes."""

import socket
import subprocess
import sys

import pytest

from legged_manipulation_unilab import assets


def test_installed_registration_in_spawn(tmp_path):
    script = tmp_path / "spawn_check.py"
    script.write_text("""import multiprocessing

TASKS = ("Go2ArmManipLoco", "L1WArmManipLoco")

def worker(queue):
    from unilab.base.registry import ensure_registries, list_registered_envs
    ensure_registries()
    registered = list_registered_envs()
    queue.put({task: set(registered[task]["available_backends"]) for task in TASKS})

if __name__ == "__main__":
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    child = context.Process(target=worker, args=(queue,))
    child.start()
    assert queue.get(timeout=30) == {task: {"mujoco"} for task in TASKS}
    child.join(timeout=30)
    assert child.exitcode == 0
""")
    subprocess.run([sys.executable, str(script)], check=True, cwd=tmp_path, timeout=60)


def test_bundled_assets_work_offline_and_repair_cache(monkeypatch, tmp_path):
    mujoco = pytest.importorskip("mujoco")
    monkeypatch.setenv("LEGGED_MANIPULATION_ASSET_CACHE", str(tmp_path / "assets"))
    monkeypatch.setattr(
        socket.socket, "connect", lambda *_: pytest.fail("Unexpected network access")
    )
    root = assets.ensure_assets()
    for scene, actions in (
        ("go2_arm/scene_flat.xml", 18),
        ("l1_w_arm/scene_2pwbc.xml", 22),
    ):
        model = mujoco.MjModel.from_xml_path(str(root / "robots" / scene))
        assert model.nu == actions
    mesh = next((root / "robots").rglob("*.obj"))
    original = mesh.read_bytes()
    mesh.write_bytes(b"x" * len(original))
    assert assets.ensure_assets() == root
    assert mesh.read_bytes() == original
