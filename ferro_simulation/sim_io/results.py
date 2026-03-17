from __future__ import annotations

from pathlib import Path

import torch as th


def _stack_time_series(value):
    if isinstance(value, th.Tensor):
        return value
    if isinstance(value, list) and value and isinstance(value[0], th.Tensor):
        return th.stack(value, dim=0)
    return None


def stacked_history(result: dict) -> dict:
    out = dict(result)
    pos_t = _stack_time_series(result.get("pos"))
    voltage_t = _stack_time_series(result.get("voltage"))
    curr_t = _stack_time_series(result.get("curr"))
    pot_t = _stack_time_series(result.get("pot"))
    if pos_t is not None:
        out["pos_tensor"] = pos_t
    if voltage_t is not None:
        out["voltage_tensor"] = voltage_t
    if curr_t is not None:
        out["curr_tensor"] = curr_t
    if pot_t is not None:
        out["pot_tensor"] = pot_t
    return out


def save_results_pt(result: dict, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    th.save(stacked_history(result), p)


def _split_result_by_batch(result: dict) -> list[dict]:
    pos_t = _stack_time_series(result.get("pos"))
    if pos_t is None or pos_t.ndim < 3:
        return [result]
    batch_size = pos_t.shape[1]
    voltage_t = _stack_time_series(result.get("voltage"))
    curr_t = _stack_time_series(result.get("curr"))
    pot_t = _stack_time_series(result.get("pot"))
    split = []
    for b in range(batch_size):
        item = dict(result)
        item["start_pos"] = result["start_pos"][b : b + 1] if isinstance(result.get("start_pos"), th.Tensor) else result.get("start_pos")
        item["target"] = result["target"][b : b + 1] if isinstance(result.get("target"), th.Tensor) else result.get("target")
        if pos_t is not None:
            item["pos"] = [frame[b : b + 1] for frame in pos_t]
        if voltage_t is not None:
            item["voltage"] = [frame[b : b + 1] for frame in voltage_t]
        if curr_t is not None:
            item["curr"] = [frame[b : b + 1] for frame in curr_t]
        if pot_t is not None:
            item["pot"] = [frame[b : b + 1] for frame in pot_t]
        split.append(item)
    return split


def save_results_split_batches(result: dict, dir_path: str | Path, prefix: str) -> None:
    out_dir = Path(dir_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    split = _split_result_by_batch(result)
    for i, item in enumerate(split, start=1):
        path = out_dir / f"{prefix}_batch_{i:03d}.pt"
        save_results_pt(item, path)
