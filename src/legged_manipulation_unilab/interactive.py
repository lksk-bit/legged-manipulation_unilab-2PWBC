"""Open a 2PWBC checkpoint with the MuJoCo keyboard viewer."""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="2PWBC keyboard playback")
    parser.add_argument("--run", required=True, help="Checkpoint file, run directory, or log group")
    parser.add_argument("--checkpoint", default="-1", help="Checkpoint iteration or filename")
    parser.add_argument("--device", help="Policy device, for example cpu or cuda:0")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--no-keyboard", action="store_true", help="Use sampled task commands")
    args = parser.parse_args(argv)

    from legged_manipulation_unilab.training import wbc2p

    wbc2p.play(
        wbc2p.checkpoint_path(args.run, args.checkpoint),
        args.device,
        args.speed,
        keyboard=not args.no_keyboard,
    )


if __name__ == "__main__":
    main()
