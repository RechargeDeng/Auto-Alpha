# src/agent/agents/alpha_generator_agent.py
from typing import Any, Dict, List
import json
from datetime import datetime
import os

from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI

from agent.state import State
from agent.prompts.alpha_prompts import (
    ALPHA_SYSTEM_PROMPT,
    ALPHA_INITIAL_PROMPT,
    ALPHA_ITERATION_PROMPT,
    ALPHA_OUTPUT_FORMAT,
)
from agent.database.faiss_utils import load_faiss_db
from agent.agents.agent_utils import extract_text


# ========= FAISS DB =========
FAISS_FIELDS_PATH = "/titan_gluster/bdeng/auto-alpha/Auto-Alpha/src/agent/database/faiss_layer2_fields"
FAISS_OPERATORS_PATH = "/titan_gluster/bdeng/auto-alpha/Auto-Alpha/src/agent/database/faiss_operators"

def save_seed_alphas_to_json(
    seed_alphas: List[Dict[str, Any]],
    iteration: int,
    path: str = "/titan_gluster/bdeng/auto-alpha/Auto-Alpha/rag_fields/seed_alphas.json",
):
    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "iteration": iteration,
        "seed_alphas": seed_alphas,
    }

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w") as f:
        json.dump(record, f, indent=2)

async def alpha_generator_agent(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """
    Generate alpha expressions using ONLY allowed fields and operators.
    """
    print(f"[alpha_generator] iteration={state.iteration}")
    llm = ChatOpenAI(model="gpt-5.2", temperature=0.3)

    # ---------- Load FAISS knowledge ----------
    fields_db = load_faiss_db(FAISS_FIELDS_PATH)
    ops_db = load_faiss_db(FAISS_OPERATORS_PATH)

    # ---------- Retrieve relevant knowledge ----------
    hypothesis_text = json.dumps(state.hypothesis, ensure_ascii=False)

    field_docs = fields_db.similarity_search(hypothesis_text, k=48)
    operator_docs = ops_db.similarity_search(hypothesis_text, k=16)

    field_knowledge = "\n".join(
        f"- {d.page_content}" for d in field_docs
    )

    operator_knowledge = "\n".join(
        f"- {d.page_content}" for d in operator_docs
    )

    # ---------- Iteration control ----------
    is_first_iteration = state.iteration == 0
    print(f"[alpha_generator] is_first_iteration={is_first_iteration}")
    num_factors = 32 if is_first_iteration else 16

    if is_first_iteration:
        user_prompt = ALPHA_INITIAL_PROMPT.format(
            hypothesis=state.hypothesis,
            num_factors=num_factors,
            field_knowledge=field_knowledge,
            operator_knowledge=operator_knowledge,
            output_format=ALPHA_OUTPUT_FORMAT,
        )
    else:
        factor_history = "\n".join(
            f"- {a['expr']} | Rank_IC={a.get('rank_ic')} Sharpe={a.get('long_alpha_sharpe')}"
            for a in state.sota_alphas
        ) ################################### maybe can add alphas description?

        has_sota = bool(state.sota_alphas)
        SOTA_STATUS = "HAS_SUCCESSFUL_ALPHAS" if has_sota else "NO_SUCCESSFUL_ALPHAS"

        user_prompt = ALPHA_ITERATION_PROMPT.format(
            hypothesis=state.hypothesis,
            sota_status=SOTA_STATUS,
            factor_history=factor_history,
            performance_summary=state.evaluation_feedback,
            num_factors=num_factors,
            field_knowledge=field_knowledge,
            operator_knowledge=operator_knowledge,
            output_format=ALPHA_OUTPUT_FORMAT,
        )

    # ---------- Call LLM ----------
    response = await llm.ainvoke(
        [
            {"role": "system", "content": ALPHA_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
    )

    # ---------- Parse JSON ----------
    content = response.content
    content = extract_text(content)
    json_start = content.find("{")
    json_end = content.rfind("}") + 1

    factors_dict = json.loads(content[json_start:json_end])

    seed_alphas = []
    for name, data in factors_dict.items():
        seed_alphas.append(
            {
                "alphaID": name,
                "expr": data["expr"],
                "desc": data["description"],
                "used_fields": data["used_fields"],
                "used_operators": data["used_operators"],
            }
        )
    save_seed_alphas_to_json(seed_alphas, state.iteration)
    return {"seed_alphas": seed_alphas}