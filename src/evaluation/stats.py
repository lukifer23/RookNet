"""
Statistical utilities for chess model evaluation.
Includes:
- Simplified Glicko-2 calculation for head-to-head match result aggregation.
- Sequential Probability Ratio Test (SPRT) for deciding whether a new model is stronger.
The implementations are intentionally lightweight (pure-Python, no third-party deps)
so they can run anywhere, including minimal CI containers.
"""
from __future__ import annotations

import math
from typing import Tuple

__all__ = [
    "glicko2_update",
    "sprt_llr",
    "sprt_decide",
]

# ---------------------------------------------------------------------------
# GLICKO-2 (simplified, single update step)
# ---------------------------------------------------------------------------
# Constants from Glicko-2 paper
TAU = 0.5  # volatilty constraint (typical default)
PI = math.pi


def _g(phi: float) -> float:
    return 1 / math.sqrt(1 + 3 * phi ** 2 / PI ** 2)


def _e(mu: float, mu_j: float, phi_j: float) -> float:
    return 1 / (1 + math.exp(-_g(phi_j) * (mu - mu_j)))


def glicko2_update(
    rating: float,
    rd: float,
    vol: float,
    score: float,
    opponent_rating: float,
    opponent_rd: float,
) -> Tuple[float, float, float]:
    """Perform a single Glicko-2 rating period update.

    Args:
        rating:       player rating (Elo scale)
        rd:           rating deviation
        vol:          rating volatility
        score:        result of the game (1, 0.5, 0)
        opponent_*:   opponent rating and deviation

    Returns:
        new_rating, new_rd, new_vol
    """
    # Convert to Glicko-2 internal scale
    mu = (rating - 1500) / 173.7178
    phi = rd / 173.7178
    mu_j = (opponent_rating - 1500) / 173.7178
    phi_j = opponent_rd / 173.7178

    # Step 2: expected score
    E = _e(mu, mu_j, phi_j)
    g_ = _g(phi_j)

    # Step 3: variance
    v = 1 / (g_ ** 2 * E * (1 - E))

    # Step 4: delta
    delta = v * g_ * (score - E)

    # Step 5: volatility update (simplified with single iteration of Newton)
    a = math.log(vol ** 2)
    A = a
    B = None
    if delta ** 2 > phi ** 2 + v:
        B = math.log(delta ** 2 - phi ** 2 - v)
    else:
        k = 1
        while True:
            B = a - k * TAU
            if _f(B, delta, phi, v, a) < 0:
                break
            k += 1

    # Newton–Raphson iteration (one step is usually enough for our use-case)
    fA = _f(A, delta, phi, v, a)
    fB = _f(B, delta, phi, v, a)
    for _ in range(10):
        C = A + (A - B) * fA / (fB - fA)
        fC = _f(C, delta, phi, v, a)
        if fC * fB < 0:
            A = B
            fA = fB
        else:
            fA /= 2
        B = C
        fB = fC
        if abs(B - A) < 1e-6:
            break
    new_vol = math.exp(A / 2)

    # Step 6: new deviation
    phi_star = math.sqrt(phi ** 2 + new_vol ** 2)
    phi_prime = 1 / math.sqrt(1 / phi_star ** 2 + 1 / v)

    # Step 7: new rating
    mu_prime = mu + phi_prime ** 2 * g_ * (score - E)

    # Convert back to Elo scale
    new_rating = 173.7178 * mu_prime + 1500
    new_rd = 173.7178 * phi_prime

    return new_rating, new_rd, new_vol


def _f(x, delta, phi, v, a):
    ex = math.exp(x)
    num = ex * (delta ** 2 - phi ** 2 - v - ex)
    den = 2 * (phi ** 2 + v + ex) ** 2
    return (num / den) - ((x - a) / (TAU ** 2))

# ---------------------------------------------------------------------------
# SPRT implementation (LLR for binomial process)
# ---------------------------------------------------------------------------
# We assume H0: p = 0.5, H1: p = 0.5 + epsilon


def sprt_llr(wins: int, draws: int, losses: int, epsilon: float = 0.05) -> float:
    """Compute log-likelihood ratio for SPRT given results."""
    n = wins + draws + losses
    if n == 0:
        return 0.0
    # Score weights: win=1, draw=0.5, loss=0
    score = wins + 0.5 * draws
    p0 = 0.5
    p1 = 0.5 + epsilon
    # Clamp p1 <1
    p1 = min(p1, 0.999)
    llr = score * math.log(p1 / p0) + (n - score) * math.log((1 - p1) / (1 - p0))
    return llr


def sprt_decide(llr: float, alpha: float = 0.05, beta: float = 0.05) -> str:
    """Return 'H1', 'H0', or 'continue' based on LLR vs thresholds."""
    A = math.log((1 - beta) / alpha)  # upper threshold
    B = math.log(beta / (1 - alpha))  # lower threshold
    if llr > A:
        return "H1"
    if llr < B:
        return "H0"
    return "continue" 