from agent.state import State
EVALUATOR_SYSTEM_PROMPT = """You are a quantitative research assistant specializing in alpha factor evaluation.

Your task is to analyze backtest results of alpha expressions and extract
generalizable insights that can guide the next round of alpha generation.

Focus on:
- Structural patterns
- Operator effectiveness
- Field usage
- Parameter preferences

Do NOT propose new alphas.
Do NOT restate raw numbers.
Think in terms of research insights.
"""

def build_evaluator_prompt(state: State) -> str:
    lines = []

    num_good = 0
    for r in state.backtest_results:
        if (r["rank_ic"] >= 0.02) and (r["ic"] >= 0.01):
            num_good += 1

        lines.append(
            f"""
Expression: {r['expr']}
IC: {r['ic']}
Rank_IC: {r['rank_ic']}
ICIR: {r['ic_ir']}
Rank_ICIR: {r['rank_ic_ir']}
Long_Alpha_Sharpe: {r['long_alpha_sharpe']}
Long_TVR: {r['long_tvr']}
"""
        )

    results_text = "\n".join(lines)

    return f"""
You are a quantitative research evaluator.

Backtest results:
{results_text}

Evaluation Rule:
- An alpha is considered HIGH-PERFORMING if Rank_IC >= 0.02 and IC >= 0.01.
- Otherwise, it is LOW-PERFORMING.
  
Some things you need to pay attention to:
- Long_Alpha_Sharpe refers to the Sharpe ratio of alpha's excess return relative to the CSI1000 index.
- In cross-sectional equity data, returns are fat-tailed and noisy.IC (Pearson) measures linear magnitude alignment and is highly sensitive to extreme returns. It can be unstable and near zero even when a signal exists.Rank IC (Spearman) measures monotonic ordering and is much more robust to outliers.
- All factor values are residuals after regressing against the Barra market capitalization factor and its nonlinear term.

Computed statistics:
- Number of high-performing alphas: {num_good}

Analysis Instructions:

1. First determine whether any alpha satisfies Rank_IC >= 0.02 and IC >= 0.01.
2. Set "has_high_performers" to true if num_good > 0, otherwise false.
3. If has_high_performers is true:
   - Analyze structural success patterns.
   - Compare high vs low performers.
4. If has_high_performers is false:
   - You MUST set "high_performing_patterns" to an empty list [].
   - Do NOT invent hypothetical success patterns.
   - Focus analysis on failure patterns and improvement directions.

5. Base insights strictly on the actual backtest results shown above.
6. Do not speculate beyond observed evidence.

Return ONLY valid JSON with the following structure:

{{
  "has_high_performers": true/false,
  "high_performing_patterns": [
    {{
      "pattern": "...",
      "evidence": "..."
    }}
  ],
  "low_performing_patterns": [
    {{
      "pattern": "...",
      "failure_reason": "..."
    }}
  ],
  "parameter_insights": {{
    "preferred_windows": [...],
    "unstable_ranges": [...]
  }},
  "summary": "Concise research-level conclusion"
}}

Critical constraints:
- If has_high_performers is false, then high_performing_patterns MUST be [].
- No text outside JSON.
- No trailing commas.
- Double quotes only.
- Ensure valid JSON.

!!! before you respond, double-check your JSON is valid !!!
"""