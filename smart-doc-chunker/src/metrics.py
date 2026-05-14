import threading
import time
from collections import defaultdict


class Metrics:
    def __init__(self) -> None:
        self._lock          = threading.Lock()
        self._timers:  dict[str, float]         = {}
        self._samples: dict[str, list[float]]   = defaultdict(list)
        self._log:     list[dict]               = []

    # ── timing ────────────────────────────────────────────────────────────────

    def start_timer(self, name: str) -> None:
        with self._lock:
            self._timers[name] = time.perf_counter()

    def end_timer(self, name: str) -> float:
        """Stop a named timer, record the elapsed ms, and return it."""
        end = time.perf_counter()
        with self._lock:
            start = self._timers.pop(name, None)
            if start is None:
                raise KeyError(f"No active timer named '{name}'")
            elapsed_ms = (end - start) * 1_000
            self._samples[name].append(elapsed_ms)
        return elapsed_ms

    # ── event log ─────────────────────────────────────────────────────────────

    def log_request(self, data: dict) -> None:
        entry = {"timestamp_ms": time.time() * 1_000}
        entry.update(data)
        with self._lock:
            self._log.append(entry)

    # ── summary ───────────────────────────────────────────────────────────────

    def get_summary(self) -> dict:
        with self._lock:
            samples_snapshot = {k: list(v) for k, v in self._samples.items()}
            log_snapshot     = list(self._log)

        summary: dict = {"timers": {}, "log_count": len(log_snapshot)}

        for name, values in samples_snapshot.items():
            summary["timers"][name] = _describe(values)

        return summary

    # ── helpers ───────────────────────────────────────────────────────────────

    def reset(self) -> None:
        with self._lock:
            self._timers.clear()
            self._samples.clear()
            self._log.clear()

    @property
    def log(self) -> list[dict]:
        with self._lock:
            return list(self._log)


# ---------------------------------------------------------------------------
# Internal statistics
# ---------------------------------------------------------------------------

def _describe(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "mean_ms": 0.0, "min_ms": 0.0,
                "max_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0}
    n      = len(values)
    total  = sum(values)
    sorted_v = sorted(values)
    return {
        "count":   n,
        "mean_ms": round(total / n, 3),
        "min_ms":  round(sorted_v[0], 3),
        "max_ms":  round(sorted_v[-1], 3),
        "p50_ms":  round(_percentile(sorted_v, 50), 3),
        "p95_ms":  round(_percentile(sorted_v, 95), 3),
    }


def _percentile(sorted_values: list[float], p: int) -> float:
    if not sorted_values:
        return 0.0
    n   = len(sorted_values)
    idx = (p / 100) * (n - 1)
    lo  = int(idx)
    hi  = min(lo + 1, n - 1)
    return sorted_values[lo] + (idx - lo) * (sorted_values[hi] - sorted_values[lo])
