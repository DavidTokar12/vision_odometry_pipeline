from __future__ import annotations

import pickle

from dataclasses import asdict
from dataclasses import is_dataclass
from pathlib import Path

import numpy as np


NEW_PATH = Path(
    "/workspaces/vision_odometry_pipeline/debug_output/main_parking/frame_0009/vo_state.pkl"
)
OLD_PATH = Path(
    "/workspaces/vision_odometry_pipeline/debug_output/main_parking_old/frame_0009/vo_state.pkl"
)


EXCLUDE_FIELDS = {"image_buffer"}


def _load(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def _np_summary(x: np.ndarray) -> str:
    finite = np.isfinite(x)
    n_finite = int(finite.sum())
    n_total = x.size
    if n_finite == 0:
        return f"shape={x.shape} dtype={x.dtype} finite=0/{n_total}"
    xf = x[finite]
    return (
        f"shape={x.shape} dtype={x.dtype} "
        f"finite={n_finite}/{n_total} "
        f"min={xf.min():.6g} max={xf.max():.6g} mean={xf.mean():.6g} std={xf.std():.6g}"
    )


def _field_items(obj) -> dict:
    if is_dataclass(obj):
        d = asdict(obj)
    else:
        d = dict(vars(obj))
    return {k: v for k, v in d.items() if k not in EXCLUDE_FIELDS}


def _compare_arrays(a: np.ndarray, b: np.ndarray, rtol=1e-5, atol=1e-6) -> str:
    if a.shape != b.shape:
        return f"DIFF shape {a.shape} vs {b.shape}"
    if a.dtype != b.dtype:
        return f"DIFF dtype {a.dtype} vs {b.dtype}"

    # Finite checks first (more informative than allclose)
    af = np.isfinite(a)
    bf = np.isfinite(b)
    if not np.array_equal(af, bf):
        return f"DIFF finite-mask: a_finite={af.sum()}/{a.size} b_finite={bf.sum()}/{b.size}"

    if not np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True):
        diff = a - b
        ad = np.abs(diff)
        max_abs = float(ad.max()) if ad.size else 0.0
        denom = np.maximum(np.abs(a), np.abs(b))
        rel = np.divide(
            ad, denom, out=np.zeros_like(ad, dtype=np.float64), where=denom > 0
        )
        max_rel = float(rel.max()) if rel.size else 0.0
        l2 = float(np.linalg.norm(diff.ravel()))
        return f"DIFF allclose failed: max_abs={max_abs:.6g} max_rel={max_rel:.6g} l2={l2:.6g}"

    return "OK"


def main():
    old_state = _load(OLD_PATH)
    new_state = _load(NEW_PATH)

    old = _field_items(old_state)
    new = _field_items(new_state)

    print("=== Loaded ===")
    print(f"OLD: {OLD_PATH}")
    print(f"NEW: {NEW_PATH}")
    print()

    for key in ["frame_id", "pipline_init_stage", "initial_avg_depth"]:
        if key in old or key in new:
            print(f"{key}: old={old.get(key)!r}  new={new.get(key)!r}")
    print()

    keys = sorted(set(old.keys()) | set(new.keys()))
    for k in keys:
        if k in EXCLUDE_FIELDS:
            continue

        ov = old.get(k, None)
        nv = new.get(k, None)

        print(f"--- {k} ---")
        if ov is None:
            print("old: <missing>")
        elif isinstance(ov, np.ndarray):
            print("old:", _np_summary(ov))
        else:
            print("old:", repr(ov))

        if nv is None:
            print("new: <missing>")
        elif isinstance(nv, np.ndarray):
            print("new:", _np_summary(nv))
        else:
            print("new:", repr(nv))

        if isinstance(ov, np.ndarray) and isinstance(nv, np.ndarray):
            print("cmp:", _compare_arrays(ov, nv))
        else:
            print("cmp:", "OK" if ov == nv else "DIFF")

        print()

    if isinstance(old.get("pose"), np.ndarray) and isinstance(
        new.get("pose"), np.ndarray
    ):
        T_old = old["pose"]
        T_new = new["pose"]
        dT = np.linalg.inv(T_old) @ T_new
        t = dT[:3, 3]
        t_norm = float(np.linalg.norm(t))
        print("=== Pose delta (inv(old) @ new) ===")
        print("translation:", t, " | norm:", f"{t_norm:.6g}")
        print("dT:\n", dT)


if __name__ == "__main__":
    main()
