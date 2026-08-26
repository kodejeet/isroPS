"""CLI benchmark runner across dataset directories."""

import argparse
import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

from lunar_correspondence.config import load_config
from lunar_correspondence.pipeline import RegistrationPipeline


def main():
    parser = argparse.ArgumentParser(
        description="Run registration benchmark over dataset pairs."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/default.yaml",
        help="Path to config file",
    )
    args = parser.parse_args()

    print("[*] Benchmark runner initialized.")
    config = load_config(args.config)
    pipeline = RegistrationPipeline(config)
    print(f"[*] Pipeline ready: {config['pipeline']['name']}")


if __name__ == "__main__":
    main()
