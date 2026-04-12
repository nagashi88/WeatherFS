from scipy.stats import t as student_t

# Probability floor/ceiling to prevent "infinite edge" traps.
# P(YES) can never be below 0.10 or above 0.95 — weather is stochastic,
# no forecast is ever 100% certain.
# Floor raised from 0.05 → 0.10: 5% was overconfident (Brier 0.50 evidence);
# a 10% floor honestly reflects irreducible uncertainty in daily-max forecasting.
PROB_FLOOR = 0.10
PROB_CEIL = 0.95

# Student-T degrees of freedom. df=5 gives fatter tails than Gaussian,
# better reflecting real weather error distributions (heat spikes, cold snaps).
STUDENT_T_DF = 5


def _clamp_prob(p: float) -> float:
    """Clamp probability to [PROB_FLOOR, PROB_CEIL]."""
    return max(PROB_FLOOR, min(PROB_CEIL, p))


def gaussian_probability(forecast_temp: float, threshold: float, rmse: float, direction: str = "above") -> float:
    """Probability using Student-T distribution (replaces Gaussian for fatter tails).

    Student-T with df=5 assigns ~3x more probability to events >3 sigma from mean,
    preventing the 'near-zero probability' trap on bucket markets.
    Output clamped to [0.05, 0.95] to prevent infinite-edge bets.
    """
    # BugFix H6: Guard against RMSE=0 which produces NaN
    if rmse <= 0:
        rmse = 3.0
    if direction == "above":
        return _clamp_prob(1 - student_t.cdf(threshold, df=STUDENT_T_DF, loc=forecast_temp, scale=rmse))
    return _clamp_prob(student_t.cdf(threshold, df=STUDENT_T_DF, loc=forecast_temp, scale=rmse))
