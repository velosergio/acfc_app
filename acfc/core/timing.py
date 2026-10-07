from __future__ import annotations


def fmt_hms(seconds: float | None) -> str:
    if seconds is None:
        return "--:--:--"
    s = int(seconds)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def eta_seconds(elapsed: float, done: int, total: int) -> float | None:
    """Tiempo restante estimado con el ritmo observado; None si todavia no hay ritmo medible."""
    if done <= 0:
        return None
    return max(0.0, elapsed / done * (total - done))
