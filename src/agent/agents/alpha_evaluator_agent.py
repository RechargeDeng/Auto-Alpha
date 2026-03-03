# src/agent/agents/alpha_evaluator_agent.py
from typing import Any, Dict, List, Optional
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from agent.state import State
from agent.prompts.alpha_evaluator_prompts import (
    EVALUATOR_SYSTEM_PROMPT,
    build_evaluator_prompt
)
import json
from agent.agents.agent_utils import extract_text

def update_sota_alphas(state, backtest_results):
    MAX_SOTA = 24
    # 1️⃣ 确保 state 上有这个属性
    if not hasattr(state, "sota_alphas") or state.sota_alphas is None:
        state.sota_alphas = []

    sota = state.sota_alphas

    for r in backtest_results:
        # 2️⃣ 先过你原有的硬条件
        if (r["rank_ic"] < 0.02) or (r["ic"] < 0.01):
            continue

        # 3️⃣ 还没满，直接加
        if len(sota) < MAX_SOTA:
            sota.append(r)
            sota.sort(key=lambda x: x["long_alpha_sharpe"], reverse=True)
            continue

        # 4️⃣ 已满：和当前最差的比较
        worst = sota[-1]

        if r["long_alpha_sharpe"] > worst["long_alpha_sharpe"]:
            # 替换最差的
            sota[-1] = r
            # 重新排序
            sota.sort(key=lambda x: x["long_alpha_sharpe"], reverse=True)

    # 5️⃣ 写回 state（原地修改其实已经够了，这里显式一点）
    state.sota_alphas = sota

    return state


async def alpha_evaluator_agent(
    state: State, config: RunnableConfig
) -> Dict[str, Any]:

    llm = ChatOpenAI(model="gpt-5.2", temperature=0.2)

    user_prompt = build_evaluator_prompt(state)

    response = await llm.ainvoke(
        [
            {"role": "system", "content": EVALUATOR_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
    )

    content = response.content
    content = extract_text(content)
    json_start = content.find("{")
    json_end = content.rfind("}") + 1
    feedback = json.loads(content[json_start:json_end])

    print("EVALUATION FEEDBACK:", feedback)
    # 顺便选出 SOTA（给 selector / generator 用）
    # sota = [
    #     r for r in state.backtest_results
    #     if r["rank_ic"] > 0.02 and r["long_alpha_sharpe"] > 2.1
    # ]

    update_sota_alphas(state, state.backtest_results)

    return {
        "evaluation_feedback": feedback,
        "sota_alphas": state.sota_alphas
    }