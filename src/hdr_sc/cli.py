from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Callable

from .config import HDRSCConfig
from .io import save_selection
from .selector import HDRSCSelector


def _load_factory(specification: str) -> Callable:
    if ":" not in specification:
        raise ValueError("adapter must use the form module:factory")
    module_name, object_name = specification.split(":", 1)
    module = importlib.import_module(module_name)
    factory = getattr(module, object_name)
    if not callable(factory):
        raise TypeError("adapter factory is not callable")
    return factory


def _read_manifest(path: str | Path) -> list[str]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    paths = [
        line.strip()
        for line in lines
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not paths:
        raise ValueError("pool manifest is empty")
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run HDR-SC support selection")
    parser.add_argument("--pool-manifest", required=True)
    parser.add_argument("--adapter", required=True, help="Python module:factory")
    parser.add_argument("--adapter-config", default="")
    parser.add_argument("--support-size", required=True, type=int)
    parser.add_argument("--max-candidates", type=int, default=500)
    parser.add_argument("--tail-fraction", type=float, default=0.10)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--epsilon-weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    pool_paths = _read_manifest(args.pool_manifest)
    adapter_config = {}
    if args.adapter_config:
        adapter_config = json.loads(
            Path(args.adapter_config).read_text(encoding="utf-8")
        )
    adapter = _load_factory(args.adapter)(adapter_config)
    selector = HDRSCSelector(
        HDRSCConfig(
            support_size=args.support_size,
            max_candidates=args.max_candidates,
            tail_fraction=args.tail_fraction,
            beta=args.beta,
            epsilon_weight=args.epsilon_weight,
            seed=args.seed,
        )
    )
    result = selector.select(pool_paths, adapter)
    save_selection(result, args.output)


if __name__ == "__main__":
    main()

