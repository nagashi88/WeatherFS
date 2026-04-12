from weatheredge.probability.gaussian import PROB_FLOOR, PROB_CEIL, _clamp_prob


def ensemble_probability(ensemble_temps: list[float], threshold: float, direction: str = "above") -> float:
    # BugFix M1: Use >= for "above" so P(above) + P(below) = 1.0 at boundary
    # BugFix from Agent 2: Guard against empty list (ZeroDivisionError)
    # Clamped to [0.05, 0.95] to prevent infinite-edge traps
    if not ensemble_temps:
        return 0.5
    if direction == "above":
        count = sum(1 for t in ensemble_temps if t >= threshold)
    else:
        count = sum(1 for t in ensemble_temps if t < threshold)
    return _clamp_prob(count / len(ensemble_temps))
