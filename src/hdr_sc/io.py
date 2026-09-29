from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

from .selector import SelectionResult


def save_selection(result: SelectionResult, output_dir: str | Path) -> None:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    ranking_path = destination / "candidate_ranking.csv"
    with ranking_path.open("w", newline="", encoding="utf-8") as stream:
        fieldnames = [
            "rank",
            "candidate_id",
            "support_indices",
            "evaluation_indices",
            "mean_normal_risk",
            "tail_normal_risk",
            "total_normal_risk",
            "tail_count",
            "Q_img",
            "admissible",
            "R_spatial",
            "spatial_rank",
            "selected",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in result.ranking:
            writer.writerow(
                {
                    "rank": row.rank,
                    "candidate_id": row.candidate_id,
                    "support_indices": ";".join(map(str, row.support_indices)),
                    "evaluation_indices": ";".join(map(str, row.evaluation_indices)),
                    "mean_normal_risk": row.mean_risk,
                    "tail_normal_risk": row.tail_risk,
                    "total_normal_risk": row.total_risk,
                    "tail_count": row.tail_count,
                    "Q_img": row.total_risk,
                    "admissible": row.admissible,
                    "R_spatial": row.spatial_risk,
                    "spatial_rank": row.spatial_rank,
                    "selected": row.candidate_id == result.selected.candidate_id,
                }
            )

    selected = result.selected
    payload = {
        "method": "HDR-SC",
        "config": asdict(result.config),
        "pool_paths": list(result.pool_paths),
        "candidate_count": len(result.ranking),
        "Q_min": result.q_min,
        "median_Q": result.median_q,
        "MAD_Q": result.mad_q,
        "epsilon": result.epsilon,
        "gate_roundoff_slack": 1e-12,
        "omega_size": result.omega_size,
        "selected_Q_img_rank": selected.rank,
        "Q_img": selected.total_risk,
        "R_spatial": selected.spatial_risk,
        "spatial_status": "singleton_skip" if result.omega_size == 1 else "computed",
        "detector": result.detector,
        "selected_candidate_id": selected.candidate_id,
        "selected_support_indices": list(selected.support_indices),
        "selected_support_paths": [
            result.pool_paths[index] for index in selected.support_indices
        ],
        "mean_normal_risk": selected.mean_risk,
        "tail_normal_risk": selected.tail_risk,
        "total_normal_risk": selected.total_risk,
    }
    (destination / "selected_support.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )
