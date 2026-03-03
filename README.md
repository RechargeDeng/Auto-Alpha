# Auto-Alpha

Auto-Alpha is an LLM-driven automated alpha research framework built with LangGraph.  
It forms a closed research loop:

Idea → Hypothesis → Alpha Generation → Backtest → Evaluation → Iteration

The system autonomously proposes research hypotheses, generates executable alpha expressions, evaluates them via backtesting, and refines future generations based on structured feedback.

---

## 🚀 Features

- **Multi-Agent Workflow (LangGraph)**
  - Modular research pipeline
  - Iterative hypothesis refinement
  - Conditional branching based on evaluation results

- **RAG-Augmented Alpha Generation**
  - FAISS-based vector retrieval
  - Field/operator constraint enforcement
  - Knowledge-driven hypothesis support

- **Structured Alpha DSL Generation**
  - Grammar-constrained expressions
  - Window-bound validation
  - Field/operator consistency checks

- **Backtest Integration**
  - External backtest bridge
  - Automated IC / RankIC / Sharpe / TVR evaluation
  - Structured result parsing

- **Iterative Optimization**
  - Threshold-based alpha filtering
  - SOTA alpha tracking
  - Feedback-driven regeneration

---

## 🧠 Architecture Overview

```
User Idea
   ↓
Hypothesis Generator
   ↓
Alpha Generator (DSL)
   ↓
Backtest Engine
   ↓
Evaluator (Performance Attribution)
   ↓
Selector (Accept / Iterate / Refine)
```

Each component is implemented as a LangGraph node, forming a state-driven research loop.

---

## 📁 Repository Structure

```
src/
│
├── agent/
│   ├── graph.py               # LangGraph workflow definition
│   ├── state.py               # Workflow state structure
│   ├── agents/                # Individual agent implementations
│   └── prompts/               # LLM prompt templates
│
├── run_backtest.py            # Backtest bridge interface
│
langgraph.json                 # Graph configuration entry
.env.example                   # Environment variable template
pyproject.toml                 # Project configuration & dependencies
```

---

## ⚙️ Installation

### 1. Clone Repository

```bash
git clone https://github.com/RechargeDeng/Auto-Alpha.git
cd Auto-Alpha
```

### 2. Install Dependencies

```bash
pip install -U pip
pip install -e .
```

Dependencies include:

- langgraph
- langchain
- openai
- faiss
- sqlalchemy
- pandas
- numpy

---

## 🔐 Environment Setup

Create a `.env` file from the example:

```bash
cp .env.example .env
```

At minimum, configure:

```
OPENAI_API_KEY=your_api_key_here
```

Optional (for tracing):

```
LANGSMITH_PROJECT=AutoAlpha
```

⚠ Never commit `.env`.

---

## ▶ Running the Workflow

### Option A: Using LangGraph CLI

The entry graph is defined in:

```
langgraph.json
```

Graph reference:

```
src/agent/graph.py:graph
```

Run via LangGraph CLI (if installed):

```bash
langgraph dev
```

---

### Option B: Python Execution

You can also import and execute the graph manually:

```python
from src.agent.graph import graph

result = graph.invoke({
    "idea": "Short-term reversal in high turnover stocks"
})
```

---

## 🧩 Backtest Integration

`run_backtest.py` bridges alpha expressions to your internal backtest engine.

To adapt for your environment:

- Replace hardcoded paths with environment variables
- Abstract backtest interface as:

```python
run_backtest(expressions: List[str]) -> pd.DataFrame
```

---

## 📊 Output Artifacts

The system stores:

- Seed alphas
- Accepted alphas
- SOTA alpha pool
- Evaluation summaries

Useful for research reproducibility and future retrieval.

---

## 🛠 Development Notes

- Python ≥ 3.9
- Formatting via Ruff (configured in `pyproject.toml`)
- Modular node design for easy extension

---

## 🔮 Roadmap

- Replace hardcoded FAISS paths with config layer
- Pluggable backtest backends
- Multi-objective alpha optimization
- Portfolio-level evaluation
- Online learning integration

---

## 📄 License

MIT License

---

## 👤 Author

Boyu Deng

---

Auto-Alpha aims to explore the future of autonomous quantitative research by combining LLM reasoning, structured financial DSL generation, and systematic backtesting.
