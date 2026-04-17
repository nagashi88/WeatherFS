from scipy.stats import t as student_t

from weatheredge.probability.gaussian import PROB_FLOOR, PROB_CEIL, STUDENT_T_DF, _clamp_prob


def bucket_probability(
    forecast_temp: float,
    bucket_low: float,
    bucket_high: float,
    rmse: float,
    clamp: bool = True,
) -> float:
    """
    Probability that actual temp falls within [bucket_low, bucket_high].
    Per PRD Section 4.6.

    Uses Student-T (df=5) for fatter tails than Gaussian — prevents near-zero
    probabilities for bucket markets 2+ sigma from the forecast mean.
    Output clamped to [PROB_FLOOR, PROB_CEIL] when clamp=True (default).
    Pass clamp=False for diagnostic/audit raw probability (used by raw_model_prob
    and the phantom-edge guard: a bucket clamped to the floor has near-zero
    true probability, so the floor creates fake "buy_yes" edge against market
    prices a few cents above 0).
    """
    # BugFix H6: Guard against RMSE=0
    if rmse <= 0:
        rmse = 3.0
    p_below_high = student_t.cdf(bucket_high, df=STUDENT_T_DF, loc=forecast_temp, scale=rmse)
    p_below_low = student_t.cdf(bucket_low, df=STUDENT_T_DF, loc=forecast_temp, scale=rmse)
    raw = max(0.0, min(1.0, p_below_high - p_below_low))
    return _clamp_prob(raw) if clamp else float(raw)
