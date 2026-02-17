from __future__ import annotations

from pathlib import Path

import torch as th


def stacked_history(result: dict) -> dict:
    out = dict(result)
    if result.get("pos"):
        out["pos_tensor"] = th.stack(result["pos"], dim=0)
    if result.get("curr"):
        out["curr_tensor"] = th.stack(result["curr"], dim=0)
    if result.get("pot"):
        out["pot_tensor"] = th.stack(result["pot"], dim=0)
    return out


def save_results_pt(result: dict, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    th.save(stacked_history(result), p)


def _split_result_by_batch(result: dict) -> list[dict]:
    pos_t = th.stack(result["pos"], dim=0) if result.get("pos") else None
    if pos_t is None or pos_t.ndim < 3:
        return [result]
    batch_size = pos_t.shape[1]
    split = []
    for b in range(batch_size):
        item = dict(result)
        item["start_pos"] = result["start_pos"][b : b + 1] if isinstance(result.get("start_pos"), th.Tensor) else result.get("start_pos")
        item["target"] = result["target"][b : b + 1] if isinstance(result.get("target"), th.Tensor) else result.get("target")
        if result.get("pos"):
            item["pos"] = [frame[b : b + 1] for frame in result["pos"]]
        if result.get("curr"):
            item["curr"] = [frame[b : b + 1] for frame in result["curr"]]
        if result.get("pot"):
            item["pot"] = [frame[b : b + 1] for frame in result["pot"]]
        split.append(item)
    return split


def save_results_split_batches(result: dict, dir_path: str | Path, prefix: str) -> None:
    out_dir = Path(dir_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    split = _split_result_by_batch(result)
    for i, item in enumerate(split, start=1):
        path = out_dir / f"{prefix}_batch_{i:03d}.pt"
        save_results_pt(item, path)
