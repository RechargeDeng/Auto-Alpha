# src/agent/state.py
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class State:
    trading_idea: str = ""
    # # Hypothesis fields (matching RD Agent structure)
    # hypothesis: str = ""
    # reason: str = ""
    # concise_reason: str = ""
    # concise_observation: str = ""
    # concise_justification: str = ""
    # concise_knowledge: str = ""

    hypothesis: Dict[str, Any] = field(default_factory=dict)

    seed_alphas: List[Dict[str, Any]] = field(default_factory=list)

    backtest_results: List[Dict[str, Any]] = field(default_factory=list)

    evaluation_feedback: Dict[str, Any] = field(default_factory=dict)

    sota_alphas: List[Dict[str, Any]] = field(default_factory=list)

    accepted_alphas: List[Dict[str, Any]] = field(default_factory=list)

    iteration: int = 0
    single_hp_iteration: int = 0
    max_iterations: int = 10
