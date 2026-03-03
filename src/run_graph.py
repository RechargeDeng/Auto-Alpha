import asyncio
from agent.graph import graph
from agent.state import State
from dotenv import load_dotenv
from typing import List, Dict, Any
load_dotenv()

#TRADING_IDEA = "Mid/Short-term order flow imbalance predicts next 5-minute returns,The idea embodied in tsrank{tsmean{SpreadOrder_vs_OrderLine.BuyValue_Skew_Ratio,24},240} might be a good starting point."
TRADING_IDEA = "bid/ask imbalance and short-term volume expansion predict returns, reflecting intraday liquidity fragmentation and retail-driven order splitting."


async def main():
    # 1️⃣ 初始化 State（这是整个系统的“世界初始状态”）
    init_state = State(
        trading_idea=TRADING_IDEA,
        iteration=0,
        max_iterations=48,
    )

    # 2️⃣ 启动 LangGraph
    final_state = await graph.ainvoke(init_state)

    # 3️⃣ 打印最终结果
    print("\n===== FINAL STATE =====")
    print("Iterations:", final_state["iteration"])
    print("Accepted alphas:", final_state["accepted_alphas"])

if __name__ == "__main__":
    asyncio.run(main())

# nohup python -u /titan_gluster/bdeng/auto-alpha/Auto-Alpha/src/run_graph.py > /titan_gluster/bdeng/auto-alpha/Auto-Alpha/logs/run_graph.out 2>&1 &

# ps -ef | grep run_graph.py
# tail -f run_graph.out
# pkill -f run_graph.py
# kill -9 PID