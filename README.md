# Retail Analytics Copilot

A local AI agent for answering retail analytics questions using Northwind DB and local documentation. Built with LangGraph, DSPy, and SQLite.

## Graph Design

- **Hybrid routing**: Router classifies questions into `rag`, `sql`, or `hybrid` paths using DSPy
- **RAG pipeline**: BM25 retriever fetches relevant doc chunks; Planner extracts constraints (dates, KPIs, entities)
- **SQL pipeline**: Generator produces SQLite queries using live schema; Executor runs queries and captures results
- **Synthesis & repair**: Synthesizer combines SQL results and docs into typed answers with citations; repair loop retries up to 2x on errors

## DSPy Optimization

**Module optimized**: `GenerateSQL` (NL→SQL generation)  
**Optimizer**: `BootstrapFewShot`  
**Metric**: Valid SQL execution rate

| Metric | Before | After | Delta |
|--------|--------|-------|-------|
| Valid SQL Rate | 95.00% | 95.00% | 0% |

The base module already performed at 95% (19/20 queries valid). The optimizer maintained this performance while embedding learned patterns as few-shot demonstrations. Training set: 20 examples.

## Trade-offs & Assumptions

- **CostOfGoods**: Northwind lacks a `Cost` field. Implemented `CostOfGoods ≈ 0.7 * UnitPrice` as specified.
- **SQL post-processing**: Added regex fixes for table name normalization and common typos to handle Phi-3.5 3.8B limitations.
- **Error recovery**: Extensive fallback extraction ensures answers are returned even when JSON parsing fails, adding complexity but improving resilience.
- **Data mismatch**: Database contains 2012-2023 dates, but test questions reference 1997. SQL queries are correct but return empty results for 1997 dates.

## Usage

```bash
# Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# Ensure Ollama is running: ollama pull phi3.5:3.8b-mini-instruct-q4_K_M

# Run
python3 run_agent_hybrid.py --batch sample_questions_hybrid_eval.jsonl --out outputs_hybrid.jsonl
```
