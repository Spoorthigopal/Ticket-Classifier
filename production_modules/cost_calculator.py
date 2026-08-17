"""
Token cost calculation and session-level cost tracking.

# PRODUCTION NOTE: In a real system, persist cost data to a database per
# user/org/request, set up budget alerts, expose a cost dashboard, and
# integrate with billing systems. Use OpenAI's usage API for reconciliation.
"""
"""
Cost calculator for Gemini API calls.
"""
"""
Cost calculator for Gemini API calls.
"""

import os
from dataclasses import dataclass

GEMINI_PRICING = {
    "gemini-2.5-flash": {
        "input_cost_per_1m_tokens": 0.075,
        "output_cost_per_1m_tokens": 0.3,
    },
    "gemini-2.5-pro": {
        "input_cost_per_1m_tokens": 1.5,
        "output_cost_per_1m_tokens": 6.0,
    },
    "gemini-3.5-flash": {
        "input_cost_per_1m_tokens": 0.075,
        "output_cost_per_1m_tokens": 0.3,
    },
    "gemini-pro": {
        "input_cost_per_1m_tokens": 0.5,
        "output_cost_per_1m_tokens": 1.5,
    },
}


@dataclass
class CostInfo:
    model: str
    input_tokens: int
    output_tokens: int
    total_cost_usd: float


class SessionTracker:
    def __init__(self):
        self.calls = 0
        self.total_cost = 0.0

    def add(self, cost_info: CostInfo):
        self.calls += 1
        self.total_cost += cost_info.total_cost_usd

    @property
    def summary(self):
        return {"calls": self.calls, "total_cost_usd": self.total_cost}


session_tracker = SessionTracker()


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> CostInfo:
    if model not in GEMINI_PRICING:
        raise ValueError(f"Model {model} not in pricing table")

    pricing = GEMINI_PRICING[model]
    input_cost = (input_tokens / 1_000_000) * pricing["input_cost_per_1m_tokens"]
    output_cost = (output_tokens / 1_000_000) * pricing["output_cost_per_1m_tokens"]
    total_cost = input_cost + output_cost

    cost_info = CostInfo(
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_cost_usd=total_cost,
    )

    session_tracker.add(cost_info)
    return cost_info


def count_tokens(text: str, model: str = "gemini-2.5-flash") -> int:
    """
    Approximate token count for Gemini models.
    Gemini uses ~4 characters per token on average.
    """
    return max(1, len(text) // 4)


if __name__ == "__main__":
    cost = calculate_cost("gemini-2.5-flash", 100, 50)
    print(f"Cost: ${cost.total_cost_usd:.6f}")