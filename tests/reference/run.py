"""JSON dispatch shim for the Phase 11 P11-2 proptest harness.

Reads a JSON request on stdin, dispatches to the appropriate numpy
reference function, and writes a JSON response on stdout. The Rust
proptest harness invokes this script once per case via subprocess.

Why a shim and not direct PyO3 bindings? The harness ships with
``proptest``-only dev-dependencies (locked by SPEC.md / Cargo.toml).
Adding PyO3 would require a build-time Python toolchain on every
contributor's machine and a non-trivial cargo feature gate. Subprocess
JSON keeps the dependency surface flat: ``python3 + numpy`` only,
which the rest of ``tests/reference/`` already requires.

Request shape::

    {"op": "<op_name>", "args": {...}}

Response shape::

    {"ok": true, "result": {...}}                 # success
    {"ok": false, "error": "<message>"}           # failure (caller checks)

Supported ops:
  * ``qwen_layer`` — full Qwen layer forward (mirrors Rust qwen_forward).
  * ``mle`` — multilinear extension evaluation (mirrors mle_evaluate).

Field elements are passed as plain JSON numbers (``int`` in [0, M31)).
Arrays of field elements are JSON lists of ints. Numpy ``np.uint32``
serializes to plain ``int`` via ``int(x)``.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from typing import Any, Dict


def _ensure_path() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)


def _to_jsonable(value: Any) -> Any:
    """Recursively convert numpy arrays / scalars to JSON-friendly types."""
    import numpy as np

    if isinstance(value, np.ndarray):
        return [int(x) for x in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def op_qwen_layer(args: Dict[str, Any]) -> Dict[str, Any]:
    """Run the full Qwen layer forward.

    args:
      x:        list[int] (length d_model), field elements in [0, M31)
      weights:  dict of named weight arrays, all flat lists of int
      config:   dict with d_model, d_ff, num_q_heads, num_kv_heads, d_head,
                v_num_heads, v_d_head, silu_scale, sigmoid_scale.
    returns:
      Full trace dict (matches Python reference's qwen_layer_forward).
    """
    import numpy as np
    from reference import qwen_layer_forward

    config = args["config"]
    x = np.asarray(args["x"], dtype=np.uint32)
    weights = {k: np.asarray(v, dtype=np.uint32) for k, v in args["weights"].items()}

    trace = qwen_layer_forward(x, weights, config)
    return _to_jsonable(trace)


def op_gdn_recurrence(args: Dict[str, Any]) -> Dict[str, Any]:
    """Run the audit-mode GDN recurrence forward (P10-7).

    args:
      initial_state: list[int] of length H*d_k*d_v (uint32 / M31).
      steps:         list of dicts with keys q/k/v/gate/beta (uint32 lists).
      config:        dict with num_heads, d_k, d_v.

    returns:
      ``{"trajectory_digest": "<hex32>", "s_final": [...],
         "o_seq": [[...], ...]}`` — keys mirror the Rust trace.
    """
    import numpy as np
    from gdn_recurrence import gdn_recurrence_forward

    initial_state = np.asarray(args["initial_state"], dtype=np.uint32)
    steps_in = args["steps"]
    steps = [
        {
            "q": np.asarray(s["q"], dtype=np.uint32),
            "k": np.asarray(s["k"], dtype=np.uint32),
            "v": np.asarray(s["v"], dtype=np.uint32),
            "gate": np.asarray(s["gate"], dtype=np.uint32),
            "beta": np.asarray(s["beta"], dtype=np.uint32),
        }
        for s in steps_in
    ]
    out = gdn_recurrence_forward(initial_state, steps, args["config"])
    return {
        "trajectory_digest": out["trajectory_digest"],
        "s_final": _to_jsonable(out["s_final"]),
        "o_seq": [_to_jsonable(o) for o in out["o_seq"]],
    }


def op_mle(args: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate the multilinear extension of ``evals`` at ``point``.

    args:
      evals: list[int] of length 2^k (M31 field elements)
      point: list[int] of length k (M31 field elements)
    returns:
      {"value": int} — MLE evaluation in [0, M31).
    """
    import numpy as np
    from mle import mle_evaluate

    evals = np.asarray(args["evals"], dtype=np.uint32)
    point = np.asarray(args["point"], dtype=np.uint32)
    val = mle_evaluate(evals, point)
    return {"value": int(val)}


_OPS = {
    "qwen_layer": op_qwen_layer,
    "mle": op_mle,
    "gdn_recurrence": op_gdn_recurrence,
}


def main() -> int:
    _ensure_path()
    try:
        request = json.load(sys.stdin)
        op_name = request["op"]
        op_args = request.get("args", {})
        if op_name not in _OPS:
            raise KeyError(f"unknown op: {op_name!r}; known: {list(_OPS)}")
        result = _OPS[op_name](op_args)
        json.dump({"ok": True, "result": result}, sys.stdout)
    except Exception as exc:  # noqa: BLE001
        # Surface the failure to the Rust harness so proptest can shrink.
        json.dump(
            {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "trace": traceback.format_exc(),
            },
            sys.stdout,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
