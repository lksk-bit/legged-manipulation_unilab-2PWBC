# Project layout

| Path | Responsibility |
|---|---|
| `src/legged_manipulation_unilab/conf/wbc2p/` | Hydra defaults and the Go2 and L1-W MuJoCo task configs |
| `src/legged_manipulation_unilab/tasks/` | UniLab robot registration, controls, commands, observations and real rewards |
| `src/legged_manipulation_unilab/algos/wbc2p/` | Full-state dynamics, replay, dual policies, differentiable rewards and BPTT |
| `src/legged_manipulation_unilab/training/` | MuJoCo collection, alternating training, logging and keyboard playback |
| `src/legged_manipulation_unilab/assets/robots/` | Bundled Go2, Airbot and L1-W MuJoCo assets |
| `migration/` | Two-stage launchers and [training commands](../migration/README.md) |

Go2 uses 18 actuator targets and a 108-feature World Model state. L1-W uses 22 targets and a 120-feature state. Its four wheel angle inputs are masked in the learned dynamics, and the model predicts wheel velocities. Both tasks use heading-aligned end-effector targets and preserve arm-to-locomotion gradients through the differentiable World Model.

Run `uv run pytest` for the registry, MuJoCo, configuration, control, replay and cross-policy gradient tests.
