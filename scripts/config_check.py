from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app_config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print the resolved application configuration.")
    parser.add_argument("--show-secrets", action="store_true")
    parser.add_argument("--env-file", default=".env")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env_file = None if args.env_file.lower() == "none" else args.env_file
    print(load_config(env_file=env_file).to_json(mask_secrets=not args.show_secrets))


if __name__ == "__main__":
    main()
