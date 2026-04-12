"""TimesFM Quantile Tail strategy — high-conviction trades on non-Gaussian tails.

This strategy exploits the key insight that the current Gaussian CDF approach
assumes symmetric, thin-tailed temperature distributions. In reality, temperature
distributions are often skewed (e.g., heat spikes from urban heat island effects,
cold outbreaks from arctic air intrusions).

TimesFM's quantile outputs capture these fat tails. When the TimesFM quantile
probability diverges significantly from the Gaussian CDF estimate, it signals
that the market is likely mispriced because most participants use Gaussian-like
models.

Alpha source: "Kurtosis Gap" — when TimesFM sees a fat tail that Gaussian misses,
we trade with higher conviction.
"""

from weatheredge.agents.base import Signal, MarketOpportunity
from weatheredge.config import Settings


def timesfm_quantile_strategy(opportunity: MarketOpportunity, settings: Settings) -> Signal:
    """Trade when TimesFM quantile probability diverges from Gaussian — fat tail alpha.

    Rules:
    - TimesFM probability must be available
    - Kurtosis gap: |timesfm_prob - gaussian_prob| >= 0.10 (TimesFM sees something Gaussian misses)
    - TimesFM and blended must agree on direction (both > or both < market price)
    - Min net edge: 8% (same as aggressive — the alpha is in precision, not threshold)
    - Confidence scales with the kurtosis gap magnitude
    """
    market_id = opportunity.market_id

    # Must have TimesFM probability
    if opportunity.timesfm_prob is None:
        return Signal(
            market_id=market_id, side="no_trade", confidence=0.0,
            agent_name="TimesFM_Quantile", reasoning="No TimesFM probability available",
        )

    # Must have Gaussian probability for comparison
    if opportunity.gaussian_prob is None:
        return Signal(
            market_id=market_id, side="no_trade", confidence=0.0,
            agent_name="TimesFM_Quantile", reasoning="No Gaussian probability for comparison",
        )

    tfm_prob = opportunity.timesfm_prob
    gauss_prob = opportunity.gaussian_prob
    market_price = opportunity.market_price

    # Kurtosis gap: how much does TimesFM diverge from Gaussian?
    # This measures where the market (likely Gaussian-priced) is wrong.
    kurtosis_gap = tfm_prob - gauss_prob

    # Need a meaningful divergence (at least 10 percentage points)
    if abs(kurtosis_gap) < 0.10:
        return Signal(
            market_id=market_id, side="no_trade", confidence=0.0,
            agent_name="TimesFM_Quantile",
            reasoning=f"Kurtosis gap {kurtosis_gap:.2f} < 0.10 — Gaussian and TimesFM agree",
        )

    # Determine trade direction from TimesFM's view
    # TimesFM says probability is higher than market → buy YES
    # TimesFM says probability is lower than market → buy NO
    tfm_edge_yes = tfm_prob - market_price
    tfm_edge_no = (1.0 - tfm_prob) - (1.0 - market_price)

    if tfm_edge_yes > tfm_edge_no and tfm_edge_yes > 0:
        side = "buy_yes"
        effective_edge = opportunity.net_edge
    elif tfm_edge_no > 0:
        side = "buy_no"
        effective_edge = opportunity.net_edge_no
    else:
        return Signal(
            market_id=market_id, side="no_trade", confidence=0.0,
            agent_name="TimesFM_Quantile", reasoning="No positive edge from TimesFM view",
        )

    # Check minimum edge (using blended probability which includes TimesFM)
    min_edge = settings.aggressive_min_edge
    if effective_edge < min_edge:
        return Signal(
            market_id=market_id, side="no_trade", confidence=0.0,
            agent_name="TimesFM_Quantile",
            reasoning=f"Net edge {effective_edge:.2f} < {min_edge} minimum",
        )

    # Ensure blended probability agrees with trade direction
    if side == "buy_yes" and opportunity.model_probability <= market_price:
        return Signal(
            market_id=market_id, side="no_trade", confidence=0.0,
            agent_name="TimesFM_Quantile",
            reasoning="TimesFM bullish but blended model disagrees",
        )
    if side == "buy_no" and opportunity.model_probability >= market_price:
        return Signal(
            market_id=market_id, side="no_trade", confidence=0.0,
            agent_name="TimesFM_Quantile",
            reasoning="TimesFM bearish but blended model disagrees",
        )

    # NWP error correction bonus: if TimesFM also predicts an NWP bias that
    # supports this trade direction, increase confidence
    error_bonus = 0.0
    if opportunity.timesfm_error_adj is not None and abs(opportunity.timesfm_error_adj) > 0.5:
        # Error adjustment supports the trade direction
        if side == "buy_yes" and opportunity.timesfm_error_adj > 0:
            error_bonus = 0.10  # NWP underestimating → supports YES
        elif side == "buy_no" and opportunity.timesfm_error_adj < 0:
            error_bonus = 0.10  # NWP overestimating → supports NO

    # Confidence: scales with kurtosis gap magnitude
    # 0.10 gap → 0.4 base, 0.20 gap → 0.7, 0.30+ gap → 1.0
    confidence = min((abs(kurtosis_gap) - 0.05) / 0.25, 1.0)
    confidence = max(0.3, confidence + error_bonus)
    confidence = min(1.0, confidence)

    reasoning = (
        f"TimesFM tail alpha: kurtosis_gap={kurtosis_gap:+.2f} "
        f"(tfm={tfm_prob:.2f} vs gauss={gauss_prob:.2f}), "
        f"edge={effective_edge:.2f}"
    )
    if error_bonus > 0:
        reasoning += f", NWP error correction supports ({opportunity.timesfm_error_adj:+.1f}F)"

    return Signal(
        market_id=market_id,
        side=side,
        confidence=confidence,
        agent_name="TimesFM_Quantile",
        reasoning=reasoning,
    )
