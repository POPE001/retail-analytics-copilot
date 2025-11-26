import json
import click
from agent.graph_hybrid import get_graph
from typing import List, Dict
import sys
from datetime import datetime

@click.command()
@click.option('--batch', required=True, help='Path to input JSONL file')
@click.option('--out', required=True, help='Path to output JSONL file')
def main(batch, out):
    """
    Run the hybrid agent on a batch of questions.
    """
    print(f"Loading batch from {batch}...")
    
    try:
        with open(batch, 'r') as f:
            questions = [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Error: File {batch} not found.")
        sys.exit(1)

    # Initialize Graph
    app = get_graph()
    
    # Initialize trace log
    trace_log = []
    trace_log.append({
        "timestamp": datetime.now().isoformat(),
        "event": "batch_start",
        "total_questions": len(questions)
    })
    
    results = []
    
    for i, item in enumerate(questions):
        question_id = item['id']
        print(f"Processing {i+1}/{len(questions)}: {question_id}")
        
        trace_log.append({
            "timestamp": datetime.now().isoformat(),
            "event": "question_start",
            "question_id": question_id,
            "question": item["question"]
        })
        
        initial_state = {
            "question": item["question"],
            "format_hint": item["format_hint"],
            "classification": None,
            "retrieved_docs": [],
            "constraints": {},
            "sql_query": None,
            "sql_result": None,
            "final_answer": None,
            "citations": [],
            "explanation": "",
            "repair_count": 0,
            "error": None
        }
        
        try:
            # Run the graph
            final_state = app.invoke(initial_state)
            
            # Log completion
            trace_log.append({
                "timestamp": datetime.now().isoformat(),
                "event": "question_complete",
                "question_id": question_id,
                "classification": final_state.get("classification"),
                "repair_count": final_state.get("repair_count", 0),
                "has_sql": final_state.get("sql_query") is not None,
                "sql_error": final_state.get("error") is not None,
                "has_answer": final_state.get("final_answer") is not None
            })
            
            # Calculate confidence: combine retrieval score, SQL success, non-empty rows, repair count
            confidence = 1.0
            final_answer = final_state.get("final_answer")
            
            # Check if answer is None or invalid - significantly reduces confidence
            if final_answer is None or final_answer == "None" or (isinstance(final_answer, str) and "None" in str(final_answer)):
                confidence = 0.3  # No valid answer
            elif final_state.get("error"):
                confidence = 0.3  # SQL error significantly reduces confidence
            elif final_state.get("repair_count", 0) > 0:
                confidence = max(0.3, 0.7 - (final_state.get("repair_count", 0) * 0.2))  # Down-weight when repaired
            elif final_state.get("sql_result"):
                sql_result = final_state.get("sql_result")
                rows = sql_result.get("rows", []) if sql_result else []
                # Check if SQL returned empty results when it shouldn't
                if len(rows) == 0 and final_state.get("sql_query"):
                    confidence = 0.4  # Empty results reduce confidence
                elif len(rows) > 0:
                    # Check if any row has null values in key fields
                    if rows:
                        row = rows[0]
                        null_count = sum(1 for v in row.values() if v is None)
                        if null_count > 0:
                            confidence = 0.6  # Some null values reduce confidence
            elif final_state.get("classification") == "rag":
                # For RAG-only, check if we have retrieved docs
                if final_state.get("retrieved_docs"):
                    # Use average retrieval score if available
                    scores = [d.get("score", 0) for d in final_state.get("retrieved_docs", [])]
                    if scores:
                        avg_score = sum(scores) / len(scores)
                        confidence = min(0.9, max(0.6, avg_score))  # Normalize to 0.6-0.9 range
                else:
                    confidence = 0.4  # No docs retrieved
            
            # Construct output
            output = {
                "id": item["id"],
                "final_answer": final_state.get("final_answer"),
                "sql": final_state.get("sql_query") or "",
                "confidence": round(confidence, 2),
                "explanation": final_state.get("explanation") or "",
                "citations": final_state.get("citations") or []
            }
            results.append(output)
            
        except Exception as e:
            print(f"Error processing {item['id']}: {e}")
            trace_log.append({
                "timestamp": datetime.now().isoformat(),
                "event": "question_error",
                "question_id": question_id,
                "error": str(e)
            })
            # Fallback error output
            results.append({
                "id": item["id"],
                "final_answer": None,
                "sql": "",
                "confidence": 0.0,
                "explanation": f"Error: {str(e)}",
                "citations": []
            })

    # Write trace log to console (can be redirected to file)
    print("\n=== Execution Trace ===")
    for entry in trace_log:
        print(json.dumps(entry))
    print("=== End Trace ===\n")
    
    print(f"Writing results to {out}...")
    with open(out, 'w') as f:
        for res in results:
            f.write(json.dumps(res) + "\n")

if __name__ == '__main__':
    main()

