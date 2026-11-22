"""
agent.cost_tracker, token counting and USD cost accounting.

Tracks cumulative token usage and estimated API cost across all OpenRouter
calls in a session.  Cost per token is configurable to match the model's
published pricing.
"""

from __future__ import annotations

# GLM-5.3-Flash pricing on OpenRouter (z.ai promo rates)
COST_PER_INPUT_TOKEN_USD: float = 0.075 / 1_000_000
COST_PER_OUTPUT_TOKEN_USD: float = 0.015 / 1_000_000


class CostTracker:
    """
    Thread-safe (via simple attribute updates, single-threaded async context)
    cumulative cost and token counter for one session.
    """

    def __init__(self) -> None:
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.cost_usd: float = 0.0

    def record(self, input_tokens: int, output_tokens: int) -> None:
        """
        Record token usage from one API call and update cumulative cost.

        Parameters
        ----------
        input_tokens:
            Prompt + image tokens consumed.
        output_tokens:
            Completion tokens generated.
        """
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.cost_usd = (
            self.total_input_tokens * COST_PER_INPUT_TOKEN_USD
            + self.total_output_tokens * COST_PER_OUTPUT_TOKEN_USD
        )

    @property
    def total_tokens(self) -> int:
        """Total tokens (input + output) consumed so far."""
        return self.total_input_tokens + self.total_output_tokens

    def summary(self) -> dict:
        """
        Return a summary dict compatible with the /stats endpoint.

        Returns
        -------
        {"total_tokens": int, "cost_usd": float}
        """
        return {"total_tokens": self.total_tokens, "cost_usd": self.cost_usd}
