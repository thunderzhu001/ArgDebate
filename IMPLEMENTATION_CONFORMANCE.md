# Implementation conformance boundary

`code_release/` is an inspectable reference implementation snapshot. The
historical evaluation artifacts do not retain a runtime Git commit or source
hash for each dataset--generator cell, so this repository cannot certify that
the snapshot generated every reported output. The frozen outcome data and
their numerical replay are not included in this code-only repository.

The following material differences between the recovered paper-facing
specification and the snapshot are retained as documented, without changing
the historical outputs or retroactively changing the implementation.

| Surface | Paper-facing specification | Reference snapshot | Interpretation |
|---|---|---|---|
| TruthfulQA route input | Item-local debate record | The TruthfulQA driver derives gate features and routed-skip answers from Majority, Weighted, and Vanilla predictions | The snapshot does not certify a self-contained historical selector. |
| Dataset drivers | Common route-aware policy | TruthfulQA uses the comparator-derived gate; StrategyQA calls the manager directly | Exact historical cell-to-driver binding is unavailable. |
| `no-conflict` | Normalized answer agreement | The conflict detector can report no detected conflict without exact normalized equality | Route labels require code-surface qualification. |
| Graph resolution | A usable graph and a separated proposal score | The manager can return resolved when refreshed conflicts disappear, without an exposed top-two separation predicate | The snapshot status does not establish the specified separation condition. |
| Convergence | Infinity-norm residual, explicit status, tolerance, and cap | QBAF evaluation uses Euclidean change and returns strengths without a convergence flag | A historical convergence status cannot be reconstructed from this snapshot. |
| Fallback | Post-graph scored and independent deadlock fallback are distinguished | Early-stop and round-limit paths use graph-weighted selection; deadlock fallback is separate | A flattened `fallback` label does not by itself identify its selection mechanism. |
| Relation estimation | Question-aware ordered candidates with the documented confidence and failure policy | The builder prompts on claim/evidence pairs without the question, skips same-agent pairs, and a failed judged call can return a heuristic typed relation | The snapshot does not recover unavailable historical edges or their failure semantics. |

Offline test success establishes the behavior covered by those tests. It does
not recover historical provider calls, unavailable debate traces, or
cell-specific source provenance, and it does not make the reported method
comparisons record-matched or compute-matched.
