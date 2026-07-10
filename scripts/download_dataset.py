"""Download the LeRobot dataset used for training/eval.

Pulls `lerobot/aloha_sim_transfer_cube_human` via LeRobotDataset, which caches it
under the HuggingFace hub cache. No physical robot required.

Run on the GPU box (or locally if you install the `[data]` extra):
    python scripts/download_dataset.py
"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="lerobot/aloha_sim_transfer_cube_human")
    args = parser.parse_args()

    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    except Exception as exc:  # pragma: no cover - import guard for local Macs
        raise SystemExit(
            "Could not import lerobot. Install the data extra:\n"
            '    pip install -e ".[data]"\n'
            f"(original error: {exc})"
        )

    print(f"Downloading {args.repo_id} ...")
    ds = LeRobotDataset(args.repo_id)
    print(f"Done. {ds.num_episodes} episodes, {ds.num_frames} frames.")
    print("Features:", list(ds.features))


if __name__ == "__main__":
    main()
