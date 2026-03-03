# src/agent/graph.py
import os
from typing import List, Dict, Any
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver


from agent.state import State
from agent.agents.user_input_agent import user_input_agent
from agent.agents.hypothesis_agent import hypothesis_agent
from agent.agents.alpha_generator_agent import alpha_generator_agent
from agent.agents.alpha_evaluator_agent import alpha_evaluator_agent
import sys
sys.path.append('/titan_gluster/bdeng/auto-alpha/Auto-Alpha/src')
from run_backtest import run_backtest
from datetime import datetime
import json
import numpy as np
#from agent.agents.alpha_coder_agent import alpha_coder_agent
#from agent.database.checkpointer_api import get_checkpoint_manager


def backtest_bridge(state: State) -> Dict[str, Any]:
    print(f"[backtest] testing {len(state.seed_alphas)} alphas")
    exprs = [a["expr"] for a in state.seed_alphas]

    results = run_backtest(exprs)  # 你自己的系统
    df = results
    exprs = [a["expr"] for a in state.seed_alphas]
    expr_desc_map = {a["expr"]: a["desc"] for a in state.seed_alphas}


    records = []
    for _, row in df[df["fml"].isin(expr_desc_map)].iterrows():
        desc = expr_desc_map[row["fml"]]
        records.append(
            {
                "expr": row["fml"],
                "desc": desc,
                "ic": float(row["ic"]),
                "rank_ic": float(row["rank_ic"]),
                "ic_ir": float(row["ic_ir"]),
                "rank_ic_ir": float(row["rank_ic_ir"]),
                "long_alpha_sharpe": float(row["long_alpha_sharpe"]),
                "long_tvr": float(row["long_tvr"]),
                # 你需要的都可以加
            }
        )

    return {
        "backtest_results": records
    }

def save_accepted_alphas_to_json(
    accepted: List[Dict[str, Any]],
    iteration: int,
    path: str = "/titan_gluster/bdeng/auto-alpha/Auto-Alpha/rag_fields/accepted_alphas_ver0.jsonl",
):
    if not accepted:
        return

    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "iteration": iteration,
        "accepted_alphas": accepted,
    }

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")

def save_sota_alphas_to_json(
    accepted: List[Dict[str, Any]],
    iteration: int,
    path: str = "/titan_gluster/bdeng/auto-alpha/Auto-Alpha/rag_fields/sota_alphas_ver0.jsonl",
):
    if not accepted:
        return

    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "iteration": iteration,
        "accepted_alphas": accepted,
    }

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")

def alpha_selector(state: State) -> Dict[str, Any]:
    if not state.backtest_results:
        return {
            "iteration": state.iteration + 1
        }

    accepted = []
    for r in state.backtest_results:
        if r["ic"] >= 0.01 and r["rank_ic"] >= 0.02:
            accepted.append(r)

    save_accepted_alphas_to_json(accepted, state.iteration)

    return {
        "accepted_alphas": accepted,
        "iteration": state.iteration + 1
    }

def should_continue(state: State) -> str:
    results = state.backtest_results
    MAX_ITER = state.max_iterations if state.max_iterations else 10

    if not results:
        return "stop"

    avg_ic = np.mean([r["ic"] for r in results])
    best_ic = max(r["ic"] for r in results)

    avg_rank_ic = np.mean([r["rank_ic"] for r in results])
    best_rank_ic = max(r["rank_ic"] for r in results)

    avg_sharpe = np.mean([r["long_alpha_sharpe"] for r in results])
    best_sharpe = max(r["long_alpha_sharpe"] for r in results)

    if state.iteration >= MAX_ITER:
        save_sota_alphas_to_json(state.sota_alphas, state.iteration)
        return "stop"

    # 1️⃣ 假设完全失败
    if (best_ic < 0.01 or best_rank_ic < 0.015) and state.single_hp_iteration > 5:
        state.single_hp_iteration = 0
        return "revise_hypothesis"

    # 2️⃣ 还有潜力
    if (best_ic >= 0.01 and best_rank_ic >= 0.015) or state.single_hp_iteration <= 5:
        state.single_hp_iteration += 1
        return "refine_alpha"


    return "refine_alpha"

def create_graph():
    """Create and configure the LangGraph workflow."""

    # Define the graph workflow
    workflow = StateGraph(State)

    # Add agents to the graph
    workflow.add_node("user_input", user_input_agent)
    workflow.add_node("hypothesis_generator", hypothesis_agent)
    workflow.add_node("alpha_generator", alpha_generator_agent)
    # workflow.add_node("alpha_coder", alpha_coder_agent)
    workflow.add_node("backtest", backtest_bridge)
    workflow.add_node("alpha_evaluator", alpha_evaluator_agent)
    workflow.add_node("alpha_selector", alpha_selector)

    # Connect the agents
    workflow.add_edge("__start__", "user_input")
    workflow.add_edge("user_input", "hypothesis_generator")
    workflow.add_edge("hypothesis_generator", "alpha_generator")
    #workflow.add_edge("alpha_generator", "alpha_coder")
    workflow.add_edge("alpha_generator", "backtest")
    workflow.add_edge("backtest", "alpha_evaluator")
    workflow.add_edge("alpha_evaluator", "alpha_selector")

    workflow.add_conditional_edges(
        "alpha_selector",
        should_continue,
        {
            "refine_alpha": "alpha_generator",
            "revise_hypothesis": "hypothesis_generator",
            "stop": "__end__"
        }
    )

    # # Configure checkpointing
    # use_postgres = os.environ.get("USE_POSTGRES_CHECKPOINT", "true").lower() == "true"

    # if use_postgres:
    #     # Use PostgreSQL checkpointer
    #     checkpointer = get_checkpoint_manager()
    #     # Create the graph with checkpointing
    #     graph = workflow.compile(checkpointer=checkpointer)
    # else:
    #     # Fallback to memory checkpointer for development
    #     checkpointer = None
    #     graph = workflow.compile(checkpointer=MemorySaver())
    #graph = workflow.compile(checkpointer=MemorySaver())
    graph = workflow.compile()
    graph.name = "Alpha Generation and Coding Workflow"
    return graph


# Create the graph
graph = create_graph()
