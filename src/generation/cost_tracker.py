from __future__ import annotations

# Cost per 1M tokens: (input, output)
COST_TABLE: dict[str, tuple[float, float]] = {
    # OpenAI
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4-turbo": (10.00, 30.00),
    "gpt-4": (30.00, 60.00),
    "gpt-3.5-turbo": (0.50, 1.50),
    "o1": (15.00, 60.00),
    "o1-mini": (3.00, 12.00),
    "o3-mini": (1.10, 4.40),
    # Anthropic
    "claude-3-5-sonnet": (3.00, 15.00),
    "claude-3-5-haiku": (0.80, 4.00),
    "claude-3-opus": (15.00, 75.00),
    "claude-sonnet-4": (3.00, 15.00),
    "claude-haiku-4": (0.80, 4.00),
    "claude-opus-4": (15.00, 75.00),
}


def _match_model(model: str) -> tuple[float, float] | None:
    model_lower = model.lower()
    for pattern, costs in COST_TABLE.items():
        if pattern in model_lower:
            return costs
    return None


def calculate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    costs = _match_model(model)
    if costs is None:
        return 0.0  # Ollama / unknown = free

    input_cost_per_m, output_cost_per_m = costs
    return (input_tokens / 1_000_000 * input_cost_per_m) + (
        output_tokens / 1_000_000 * output_cost_per_m
    )
