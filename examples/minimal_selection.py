from pathlib import Path

from examples.plugin_adapter import build_adapter
from hdr_sc import HDRSCConfig, HDRSCSelector
from hdr_sc.io import save_selection


def main() -> None:
    pool_paths = [f"normal_{index:02d}.png" for index in range(12)]
    selector = HDRSCSelector(
        HDRSCConfig(
            support_size=2,
            max_candidates=50,
            tail_fraction=0.10,
            beta=1.0,
            seed=0,
        )
    )
    result = selector.select(pool_paths, build_adapter({"dimension": 8}))
    save_selection(result, Path("output") / "minimal-example")
    print("Selected indices:", result.selected.support_indices)
    print(
        "Selected paths:",
        [result.pool_paths[index] for index in result.selected.support_indices],
    )


if __name__ == "__main__":
    main()

