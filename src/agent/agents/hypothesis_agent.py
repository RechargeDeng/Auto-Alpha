# src/agent/agents/hypothesis_agent.py
from typing import Dict, Any
import json
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableConfig

from agent.state import State
from agent.prompts.hypothesis_prompts import (
    HYPOTHESIS_SYSTEM_PROMPT,
    HYPOTHESIS_INITIAL_PROMPT,
    HYPOTHESIS_ITERATION_PROMPT,
    HYPOTHESIS_OUTPUT_FORMAT,
)
from agent.database.faiss_utils import load_faiss_db
from agent.agents.agent_utils import extract_text


# ---------- FAISS loaders ----------
faiss_modules = load_faiss_db("/titan_gluster/bdeng/auto-alpha/Auto-Alpha/src/agent/database/faiss_layer1_modules")
faiss_alpha101 = load_faiss_db("/titan_gluster/bdeng/auto-alpha/Auto-Alpha/src/agent/database/faiss_alpha101_paper_db")
faiss_papers = load_faiss_db("/titan_gluster/bdeng/auto-alpha/Auto-Alpha/src/agent/database/faiss_paper0")


def retrieve_knowledge(db, query: str, k: int = 4) -> str:
    docs = db.similarity_search(query, k=k)
    return "\n".join(d.page_content for d in docs)


async def hypothesis_agent(state: State, config: RunnableConfig) -> Dict[str, Any]:
    print(f"trading idea: {state.trading_idea}")
    llm = ChatOpenAI(model="gpt-5.2", temperature=0.3)

    # ---------- RAG ----------
    module_knowledge = retrieve_knowledge(
        faiss_modules,
        query=state.trading_idea
    )

    paper_knowledge = retrieve_knowledge(
        faiss_papers,
        query=state.trading_idea
    )

    alpha101_knowledge = retrieve_knowledge(
        faiss_alpha101,
        query=state.trading_idea
    )
    print(f"[hypothesis_agent] iteration: {state.iteration}")
    is_first_iteration = state.iteration == 0

    if is_first_iteration:
        user_prompt = HYPOTHESIS_INITIAL_PROMPT.format(
            trading_idea=state.trading_idea,
            module_knowledge=module_knowledge,
            paper_knowledge=paper_knowledge,
            alpha101_knowledge=alpha101_knowledge,
            output_format=HYPOTHESIS_OUTPUT_FORMAT,
        )
    else:
        user_prompt = HYPOTHESIS_ITERATION_PROMPT.format(
            previous_hypothesis=state.hypothesis,
            performance_summary=state.evaluation_feedback,
            module_knowledge=module_knowledge,
            paper_knowledge=paper_knowledge,
            alpha101_knowledge=alpha101_knowledge,
            output_format=HYPOTHESIS_OUTPUT_FORMAT,
        )

    response = await llm.ainvoke(
        [
            {"role": "system", "content": HYPOTHESIS_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
    )
    print(f"[hypothesis_agent] response: {response.content}")
    content = response.content
    content = extract_text(content)
    json_start = content.find("{")
    json_end = content.rfind("}") + 1
    hypothesis_data = content[json_start:json_end]

    hypothesis_dict = json.loads(hypothesis_data)

    return {
        "hypothesis": hypothesis_dict,
        "iteration": state.iteration
    }