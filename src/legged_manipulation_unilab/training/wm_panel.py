"""Live 2PWBC metrics in the style of Uni-SuperMimic's terminal panel."""

from __future__ import annotations

from collections import deque

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeRemainingColumn
from rich.table import Table
from rich.text import Text


class TrainingPanel:
    def __init__(self, stage, device, num_envs, iterations, enabled=True):
        self.console = Console()
        self.stage = stage
        self.device = device
        self.num_envs = num_envs
        self.progress = Progress(
            TextColumn("{task.description} {task.completed}/{task.total}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=self.console,
            expand=True,
        )
        self.task = self.progress.add_task(stage, total=iterations, start=False)
        self.history = deque(maxlen=120)
        self.stats = {}
        self.status = "Filling replay"
        self.checkpoint = None
        self.live = (
            Live(self.render(), console=self.console, refresh_per_second=2)
            if enabled and self.console.is_terminal
            else None
        )

    def __enter__(self):
        if self.live is not None:
            self.live.start(refresh=True)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.live is not None:
            self.live.stop()

    def start_updates(self, *, compiling=False):
        self.status = "Compiling first update" if compiling else "World model + BPTT"
        self.progress.start_task(self.task)
        if self.live is not None:
            self.live.update(self.render(), refresh=True)
        elif compiling:
            print("Compiling first WM and BPTT update", flush=True)

    def update(self, iteration, stats, checkpoint=None):
        self.status = "World model + BPTT"
        self.stats = stats
        self.history.append(stats["Real/reward_leg"])
        self.progress.advance(self.task)
        if checkpoint is not None:
            self.checkpoint = checkpoint
        if self.live is not None:
            self.live.update(self.render(), refresh=True)
        else:
            arm = f" arm={stats['Loss/arm_actor']:.5f}" if "Loss/arm_actor" in stats else ""
            print(
                f"{self.stage} iteration {iteration}: WM={stats['WorldModel/loss']:.5f} "
                f"leg={stats['Loss/leg_actor']:.5f}{arm} "
                f"{stats['Perf/total_fps']:,.0f} steps/s",
                flush=True,
            )

    def sparkline(self):
        values = list(self.history)[-25:]
        if not values:
            return ""
        ticks = "▁▂▃▄▅▆▇█"
        try:
            ticks.encode(self.console.encoding or "utf-8")
        except UnicodeEncodeError:
            ticks = " .:-=+*#"
        lo, hi = min(values), max(values)
        return "".join(
            ticks[int((len(ticks) - 1) * (v - lo) / max(hi - lo, 1e-12))] for v in values
        )

    def render(self):
        groups = (
            (
                "World Model",
                (
                    ("State loss", "WorldModel/loss"),
                    ("Final EE error (m)", "WorldModel/final_ee_error_m"),
                ),
            ),
            (
                "Policy",
                (
                    ("Leg actor / critic", "Loss/leg_actor", "Loss/leg_critic"),
                    ("Arm actor / critic", "Loss/arm_actor", "Loss/arm_critic"),
                    ("Leg / arm grad norm", "Policy/leg_grad_norm", "Policy/arm_grad_norm"),
                    ("Imagined leg / arm reward", "Imagined/reward_leg", "Imagined/reward_arm"),
                    ("Final survival", "Imagined/survival"),
                ),
            ),
            (
                "MuJoCo",
                (
                    ("Leg / arm reward", "Real/reward_leg", "Real/reward_arm"),
                    ("Pitch / roll error (°)", "Real/pitch_mae_deg", "Real/roll_mae_deg"),
                    ("Mean episode length", "Real/mean_episode_length"),
                    ("Fall / timeout count", "Real/falls", "Real/timeouts"),
                ),
            ),
            (
                "Timing",
                (
                    ("Steps per second", "Perf/total_fps"),
                    ("Collected steps per env", "Perf/collection_steps_per_env"),
                    ("Collect / learn (s)", "Perf/collection_time", "Perf/learning_time"),
                    ("MuJoCo step/reset (s)", "Perf/mujoco_time"),
                    (
                        "Physics / state update (s)",
                        "Perf/env_physics_time",
                        "Perf/env_update_state_time",
                    ),
                    (
                        "Action / explicit reset (s)",
                        "Perf/env_apply_action_time",
                        "Perf/env_reset_time",
                    ),
                    ("WM / BPTT (s)", "Perf/world_model_time", "Perf/bptt_time"),
                    (
                        "Rollout / actor / critic (s)",
                        "Perf/bptt_rollout_time",
                        "Perf/bptt_actor_time",
                        "Perf/bptt_critic_time",
                    ),
                    (
                        "Leg / arm backward (s)",
                        "Perf/bptt_leg_backward_time",
                        "Perf/bptt_arm_backward_time",
                    ),
                ),
            ),
        )
        table = Table.grid(padding=(0, 1), expand=True)
        table.add_column(style="cyan")
        table.add_column()
        table.add_column(justify="right")
        for group, rows in groups:
            for i, (label, *keys) in enumerate(rows):
                values = " / ".join(
                    f"{self.stats[key]:.4g}" if key in self.stats else "—" for key in keys
                )
                table.add_row(group if i == 0 else "", label, values)
        trend = self.sparkline()
        footer = Text(f"Leg reward  {trend}") if trend else Text("")
        if self.checkpoint is not None:
            footer.append(f"    Checkpoint  {self.checkpoint.name}", style="dim")
        return Panel(
            Group(
                self.progress,
                Text(f"{self.device}  ·  {self.num_envs} envs  ·  {self.status}"),
                table,
                footer,
            ),
            title="2PWBC | MuJoCo",
            border_style="bright_blue",
        )
