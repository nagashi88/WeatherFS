from weatheredge.probability.gaussian import PROB_FLOOR, PROB_CEIL, _clamp_prob


def ensemble_probability(
    ensemble_temps: list[float],
    threshold: float,
    direction: str = "above",
    clamp: bool = True,
) -> float:
    # BugFix M1: Use >= for "above" so P(above) + P(below) = 1.0 at boundary
    # BugFix from Agent 2: Guard against empty list (ZeroDivisionError)
    # Clamped to [PROB_FLOOR, PROB_CEIL] when clamp=True (default).
    # Pass clamp=False for diagnostic/audit raw probability (used by raw_model_prob).
    if not ensemble_temps:
        return 0.5
    if direction == "above":
        count = sum(1 for t in ensemble_temps if t >= threshold)
    else:
        count = sum(1 for t in ensemble_temps if t < threshold)
    raw = count / len(ensemble_temps)
    return _clamp_prob(raw) if clamp else float(raw)
