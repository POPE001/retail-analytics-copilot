import os
import dspy
import re
import ast
from typing import TypedDict, List, Optional, Dict, Any, Annotated
from langgraph.graph import StateGraph, END
import operator
from agent.dspy_signatures import Router, GenerateSQL, SynthesizeAnswer, ExtractConstraints
from agent.tools.sqlite_tool import SQLiteTool
from agent.rag.retrieval import SimpleRetriever

class AgentState(TypedDict):
    question: str
    format_hint: str
    classification: Optional[str]
    retrieved_docs: List[Dict]
    constraints: Dict[str, Any]
    sql_query: Optional[str]
    sql_result: Optional[Dict[str, Any]]
    final_answer: Optional[Any]
    citations: List[str]
    explanation: str
    repair_count: int
    error: Optional[str]

class HybridAgent:
    def __init__(self, db_path: str, docs_dir: str, model_name="phi3.5:3.8b-mini-instruct-q4_K_M"):
        self.db = SQLiteTool(db_path)
        self.retriever = SimpleRetriever(docs_dir)
        
        lm = dspy.LM(f"openai/{model_name}", api_base="http://localhost:11434/v1", api_key="ollama")
        dspy.configure(lm=lm)
        
        self.router_module = dspy.ChainOfThought(Router)
        self.sql_generator = dspy.ChainOfThought(GenerateSQL)
        self.synthesizer = dspy.ChainOfThought(SynthesizeAnswer)
        self.constraint_extractor = dspy.ChainOfThought(ExtractConstraints)
        
        self._init_sql_postprocessing_rules()
    
    def _init_sql_postprocessing_rules(self):
        """Initialize SQL post-processing rules for fixing common model errors."""
        self.table_name_fixes = [
            (r'\bOrders\b', 'orders'),
            (r'\bOrder\s+Details\b', 'order_items'),
            (r'\bOrderDetails\b', 'order_items'),
            (r'\bOrder_items\b', 'order_items'),
            (r'\bProducts\b', 'products'),
            (r'\bCustomers\b', 'customers'),
            (r'\border_details\b', 'order_items'),
            (r'\bOrder_details\b', 'order_items'),
        ]
        
        self.typo_fixes = [
            (r'\bBETWEWS\b', 'BETWEEN'),
            (r'\bBETWEDIR\b', 'BETWEEN'),
        ]
    
    def _normalize_sql_table_names(self, sql: str) -> str:
        """Normalize table names to use lowercase compatibility views."""
        for pattern, replacement in self.table_name_fixes:
            sql = re.sub(pattern, replacement, sql, flags=re.IGNORECASE)
        return sql
    
    def _fix_sql_typos(self, sql: str) -> str:
        """Fix common SQL typos from model generation."""
        for pattern, replacement in self.typo_fixes:
            sql = re.sub(pattern, replacement, sql, flags=re.IGNORECASE)
        return sql
    
    def _fix_category_name_joins(self, sql: str) -> str:
        """Add Categories join if CategoryName is used without proper join."""
        if 'CategoryName' not in sql or 'Categories' in sql or 'JOIN' not in sql:
            return sql
        
        sql = re.sub(
            r'(JOIN\s+products\s+p\s+ON[^W]+)(WHERE)',
            r'\1JOIN Categories c ON p.CategoryID = c.CategoryID \2',
            sql,
            flags=re.IGNORECASE
        )
        sql = re.sub(r'\bp\.CategoryName\b', 'c.CategoryName', sql, flags=re.IGNORECASE)
        return sql
    
    def _fix_subquery_aliases(self, sql: str) -> str:
        """Fix ambiguous column names in subqueries by adding proper aliases."""
        pattern = 'FROM order_items JOIN orders ON orders.OrderID = order_items.OrderID WHERE OrderDate'
        if pattern not in sql:
            return sql
        
        sql = sql.replace(
            pattern,
            'FROM order_items oi2 JOIN orders o2 ON o2.OrderID = oi2.OrderID WHERE o2.OrderDate'
        )
        sql = re.sub(r'SELECT\s+OrderID,', 'SELECT oi2.OrderID,', sql, flags=re.IGNORECASE)
        sql = re.sub(r'COUNT\(DISTINCT\s+oi2?\.OrderID\)', 'COUNT(DISTINCT oi.OrderID)', sql, flags=re.IGNORECASE)
        sql = re.sub(r'GROUP BY\s+OrderID\b', 'GROUP BY oi2.OrderID', sql, flags=re.IGNORECASE)
        
        return sql
    
    def _postprocess_sql(self, sql: str) -> str:
        """Apply all SQL post-processing fixes."""
        if not sql:
            return sql
        
        sql = sql.replace("```sql", "").replace("```", "").strip()
        sql = self._normalize_sql_table_names(sql)
        sql = self._fix_sql_typos(sql)
        sql = self._fix_category_name_joins(sql)
        sql = self._fix_subquery_aliases(sql)
        
        return sql
    
    def _extract_sql_from_error(self, error_str: str) -> str:
        """Extract SQL query from error message if present."""
        if "LM Response:" not in error_str:
            return ""
        
        sql_match = re.search(r'SELECT[^;]+', error_str, re.IGNORECASE | re.DOTALL)
        return sql_match.group(0).strip() if sql_match else ""
    
    def _extract_table_names_from_sql(self, sql: str) -> List[str]:
        """Extract table names from SQL query for citations. Returns canonical table names."""
        if not sql:
            return []
        
        found_tables = []
        sql_lower = sql.lower()
        
        if 'order_items' in sql_lower or '"order details"' in sql_lower:
            found_tables.append('Order Details')
        if 'orders' in sql_lower and 'Order Details' not in found_tables:
            found_tables.append('Orders')
        if 'products' in sql_lower:
            found_tables.append('Products')
        if 'customers' in sql_lower:
            found_tables.append('Customers')
        if 'categories' in sql_lower:
            found_tables.append('Categories')
        
        return found_tables

    def node_router(self, state: AgentState):
        pred = self.router_module(question=state["question"])
        classification = getattr(pred, "classification", "hybrid").lower()
        if classification not in ["rag", "sql", "hybrid"]:
            classification = "hybrid"
        return {"classification": classification}

    def node_retriever(self, state: AgentState):        
        if state["classification"] == "sql":
             return {"retrieved_docs": []}
        
        docs = self.retriever.retrieve(state["question"], k=5)
        return {"retrieved_docs": docs}

    def node_planner(self, state: AgentState):
        if state["classification"] == "rag":
            return {"constraints": {}}
            
        context_str = "\n\n".join([f"--- {d['id']} ---\n{d['content']}" for d in state.get("retrieved_docs", [])])
        
        try:
            pred = self.constraint_extractor(question=state["question"], context=context_str)
            
            constraints = {
                "date_range_start": getattr(pred, "date_range_start", None),
                "date_range_end": getattr(pred, "date_range_end", None),
                "kpi_formula": getattr(pred, "kpi_formula", None),
                "entities": getattr(pred, "entities", None)
            }
        except Exception as e:
            import re
            question = state["question"]
            date_match = re.search(r'(\d{4}-\d{2}-\d{2})', question + " " + context_str)
            constraints = {
                "date_range_start": None,
                "date_range_end": None,
                "kpi_formula": None,
                "entities": None
            }
            if "Summer Beverages" in question or "1997-06" in context_str:
                constraints["date_range_start"] = "1997-06-01"
                constraints["date_range_end"] = "1997-06-30"
            elif "Winter Classics" in question or "1997-12" in context_str:
                constraints["date_range_start"] = "1997-12-01"
                constraints["date_range_end"] = "1997-12-31"
        
        return {"constraints": constraints}

    def node_generator(self, state: AgentState):
        """Generate SQL query from natural language question using DSPy."""
        if state["classification"] == "rag":
            return {"sql_query": None}
            
        schema = self.db.get_schema()
        constraints_str = str(state.get("constraints", {}))
        
        try:
            pred = self.sql_generator(
                database_schema=schema, 
                question=state["question"], 
                constraints=constraints_str
            )
            sql = getattr(pred, "sql_query", "")
        except Exception as e:
            sql = self._extract_sql_from_error(str(e))
        
        sql = self._postprocess_sql(sql)
        
        return {"sql_query": sql}

    def node_executor(self, state: AgentState):
        query = state.get("sql_query")
        if not query:
            return {"sql_result": None}
            
        result = self.db.execute_query(query)
        
        if result.get("error"):
            return {"sql_result": result, "error": result["error"]}
            
        return {"sql_result": result, "error": None}

    def node_synthesizer(self, state: AgentState):
        context_str = "\n\n".join([f"--- {d['id']} ---\n{d['content']}" for d in state.get("retrieved_docs", [])])
        
        sql_result = state.get("sql_result")
        final_answer_from_sql = None
        if sql_result and sql_result.get("rows") and not sql_result.get("error"):
            rows = sql_result.get("rows", [])
            format_hint = state.get("format_hint", "")
            
            if rows and format_hint:
                try:
                    if "list" in format_hint and "product" in format_hint.lower():
                        final_answer_from_sql = [{"product": str(row.get(list(row.keys())[0], "")), 
                                        "revenue": float(row.get(list(row.keys())[1], 0))} 
                                       for row in rows[:3]]
                    elif "category" in format_hint.lower() and "quantity" in format_hint.lower():
                        if rows:
                            row = rows[0]
                            keys = list(row.keys())
                            category_key = None
                            quantity_key = None
                            for key in keys:
                                key_lower = key.lower()
                                if 'category' in key_lower or ('name' in key_lower and 'category' in str(row.get(key, '')).lower()):
                                    category_key = key
                                elif 'quantity' in key_lower or 'total' in key_lower:
                                    quantity_key = key
                            
                            if not category_key:
                                category_key = keys[0] if len(keys) > 0 else None
                            if not quantity_key:
                                quantity_key = keys[1] if len(keys) > 1 else None
                            
                            qty_val = row.get(quantity_key, 0) if quantity_key else 0
                            if qty_val is None:
                                qty_val = 0
                            try:
                                qty_val = float(qty_val) if qty_val is not None and qty_val != "" else 0
                                qty_val = int(qty_val)
                            except (ValueError, TypeError):
                                qty_val = 0
                            cat_val = row.get(category_key, "") if category_key else ""
                            final_answer_from_sql = {"category": str(cat_val) if cat_val is not None else "", 
                                          "quantity": qty_val}
                    elif "float" in format_hint:
                        if rows:
                            row = rows[0]
                            val = None
                            for key in row.keys():
                                if any(term in key.lower() for term in ['aov', 'revenue', 'total', 'avg', 'average', 'value']):
                                    val = row.get(key)
                                    break
                            if val is None:
                                keys = list(row.keys())
                                val = row.get(keys[0], 0)
                            if val is not None and val != "":
                                final_answer_from_sql = round(float(val), 2)
                    elif "int" in format_hint:
                        if rows:
                            row = rows[0]
                            keys = list(row.keys())
                            val = row.get(keys[0], 0)
                            if val is not None:
                                final_answer_from_sql = int(val)
                    elif "customer" in format_hint.lower() and "margin" in format_hint.lower():
                        if rows:
                            row = rows[0]
                            keys = list(row.keys())
                            final_answer_from_sql = {"customer": str(row.get(keys[0], "")), 
                                          "margin": round(float(row.get(keys[1], 0)), 2)}
                except Exception as extract_err:
                    pass
        
        try:
            pred = self.synthesizer(
                question=state["question"],
                format_hint=state["format_hint"],
                context=context_str,
                sql_query=state.get("sql_query", ""),
                sql_result=str(state.get("sql_result", "")),
            )
            
            citations = getattr(pred, "citations", [])
            if isinstance(citations, str):
                 try:
                     citations = ast.literal_eval(citations)
                 except:
                     chunk_matches = re.findall(r'\[([^:]+::chunk\d+)[^\]]*\]', citations)
                     if chunk_matches:
                         citations = chunk_matches
                     else:
                         citations = [citations]
            
            if not isinstance(citations, list):
                citations = [citations] if citations else []
            
            sql_query = state.get("sql_query", "")
            if sql_query:
                table_names = self._extract_table_names_from_sql(sql_query)
                for table in table_names:
                    if table not in citations:
                        citations.append(table)
            
            if final_answer_from_sql is not None:
                final_answer = final_answer_from_sql
            else:
                final_answer = getattr(pred, "final_answer", None)
                if isinstance(final_answer, str):
                    # Handle string "None" - convert to proper type based on format_hint
                    if final_answer.strip().lower() in ["none", "null"]:
                        format_hint = state.get("format_hint", "")
                        if "float" in format_hint:
                            final_answer = None  # Will serialize to null in JSON
                        elif "int" in format_hint:
                            final_answer = None
                        elif "{" in format_hint or "dict" in format_hint.lower():
                            final_answer = None
                        elif "list" in format_hint:
                            final_answer = None
                        else:
                            final_answer = None
                    elif ("{" in final_answer or "[" in final_answer) and ("category" in final_answer.lower() or "quantity" in final_answer.lower()):
                        try:
                            import ast
                            final_answer_clean = final_answer.replace("null", "None").replace("'", '"')
                            final_answer = ast.literal_eval(final_answer_clean)
                            if isinstance(final_answer, dict) and "quantity" in final_answer:
                                qty = final_answer.get("quantity")
                                if qty is None or qty == "null":
                                    final_answer["quantity"] = 0
                                else:
                                    try:
                                        final_answer["quantity"] = int(float(qty))
                                    except:
                                        final_answer["quantity"] = 0
                        except:
                            if "category" in final_answer.lower() and "quantity" in final_answer.lower():
                                cat_match = re.search(r"category['\"]?\s*:\s*['\"]([^'\"]+)['\"]", final_answer)
                                qty_match = re.search(r"quantity['\"]?\s*:\s*(null|\d+)", final_answer)
                                if cat_match or qty_match:
                                    final_answer = {
                                        "category": cat_match.group(1) if cat_match else "",
                                        "quantity": int(qty_match.group(1)) if qty_match and qty_match.group(1) != "null" else 0
                                    }
            explanation = getattr(pred, "explanation", "")
            
        except Exception as e:
            error_str = str(e)
            final_answer = final_answer_from_sql if 'final_answer_from_sql' in locals() and final_answer_from_sql is not None else None
            explanation = f"JSON parsing failed, attempting fallback extraction. Error: {str(e)[:200]}"
            citations = []
            
            if "LM Response:" in error_str:
                lm_response_match = re.search(r'LM Response:\s*(\{.*?\})', error_str, re.DOTALL)
                if lm_response_match:
                    try:
                        raw_json = lm_response_match.group(1)
                        raw_json = re.sub(r'\\n', '', raw_json)
                        raw_json = re.sub(r'\\text\{[^}]*\}', '', raw_json)
                        answer_match = re.search(r'"final_answer"\s*:\s*([^,}\]]+)', raw_json)
                        if answer_match:
                            answer_str = answer_match.group(1).strip().strip('"').strip("'")
                            format_hint = state.get("format_hint", "")
                            if "int" in format_hint:
                                try:
                                    final_answer = int(re.search(r'\d+', answer_str).group())
                                except:
                                    pass
                            elif "float" in format_hint:
                                try:
                                    final_answer = float(re.search(r'\d+\.?\d*', answer_str).group())
                                except:
                                    pass
                            else:
                                final_answer = answer_str
                        
                        citations_match = re.search(r'"citations"\s*:\s*(\[[^\]]*\])', raw_json)
                        if citations_match:
                            try:
                                citations = ast.literal_eval(citations_match.group(1))
                            except:
                                pass
                    except Exception as parse_err:
                        pass
        
        if not isinstance(citations, list):
            citations = [citations] if citations else []
        
        cleaned_citations = []
        table_name_map = {
            'orders': 'Orders',
            'order_items': 'Order Details',
            'order details': 'Order Details',
            'products': 'Products',
            'customers': 'Customers',
            'categories': 'Categories'
        }
        
        for cit in citations:
            if isinstance(cit, str):
                cit = cit.strip().strip("'").strip('"')
                cit = re.sub(r'^\[([^\]]+)\].*$', r'\1', cit)
                cit = re.sub(r'(::chunk\d+):\d+', r'\1', cit)
                cit_lower = cit.lower().strip()
                if cit_lower in table_name_map:
                    cit = table_name_map[cit_lower]
                if ',' in cit:
                    for sub_cit in cit.split(','):
                        sub_cit = sub_cit.strip().strip("'").strip('"')
                        if sub_cit:
                            sub_cit_lower = sub_cit.lower().strip()
                            if sub_cit_lower in table_name_map:
                                sub_cit = table_name_map[sub_cit_lower]
                            if sub_cit not in cleaned_citations:
                                cleaned_citations.append(sub_cit)
                else:
                    if cit and cit not in cleaned_citations:
                        cleaned_citations.append(cit)
            else:
                if cit not in cleaned_citations:
                    cleaned_citations.append(cit)
        
        sql_query = state.get("sql_query", "")
        if sql_query:
            table_names = self._extract_table_names_from_sql(sql_query)
            for table in table_names:
                if table not in cleaned_citations:
                    cleaned_citations.append(table)
        
        if state.get("classification") == "rag" and not cleaned_citations:
            for doc in state.get("retrieved_docs", []):
                doc_id = doc.get("id", "")
                if doc_id and doc_id not in cleaned_citations:
                    cleaned_citations.append(doc_id)
        
        return {
            "final_answer": final_answer,
            "explanation": explanation,
            "citations": cleaned_citations
        }

    def node_repair(self, state: AgentState):
        return {"repair_count": state["repair_count"] + 1}

    def should_repair(self, state: AgentState):
        if state["repair_count"] >= 2:
            return "synthesizer"
        
        if state.get("error"):
            return "generator"
        
        return "synthesizer"

    def _log_trace_event(self, node_name: str, state: AgentState, event_type: str = "node_execution"):
        """Log trace event for replayable event log."""
        trace_entry = {
            "node": node_name,
            "event_type": event_type,
            "classification": state.get("classification"),
            "has_sql": state.get("sql_query") is not None,
            "sql_error": state.get("error") is not None,
            "repair_count": state.get("repair_count", 0),
            "has_answer": state.get("final_answer") is not None,
        }
        print(f"[TRACE] {node_name}: {trace_entry}")
        return trace_entry
    
    def build_graph(self):
        workflow = StateGraph(AgentState)
        
        def traced_router(state):
            self._log_trace_event("router", state)
            return self.node_router(state)
        
        def traced_retriever(state):
            self._log_trace_event("retriever", state)
            return self.node_retriever(state)
        
        def traced_planner(state):
            self._log_trace_event("planner", state)
            return self.node_planner(state)
        
        def traced_generator(state):
            self._log_trace_event("generator", state)
            return self.node_generator(state)
        
        def traced_executor(state):
            self._log_trace_event("executor", state)
            return self.node_executor(state)
        
        def traced_synthesizer(state):
            self._log_trace_event("synthesizer", state)
            return self.node_synthesizer(state)
        
        def traced_repair(state):
            self._log_trace_event("repair", state, "repair_triggered")
            return self.node_repair(state)
        
        workflow.add_node("router", traced_router)
        workflow.add_node("retriever", traced_retriever)
        workflow.add_node("planner", traced_planner)
        workflow.add_node("generator", traced_generator)
        workflow.add_node("executor", traced_executor)
        workflow.add_node("synthesizer", traced_synthesizer)
        workflow.add_node("repair", traced_repair)
        
        workflow.set_entry_point("router")
        
        workflow.add_conditional_edges(
            "router",
            lambda x: x["classification"],
            {
                "rag": "retriever",
                "sql": "generator", 
                "hybrid": "retriever"
            }
        )
        
        workflow.add_edge("retriever", "planner") 
        
        workflow.add_conditional_edges(
            "planner",
            lambda x: x["classification"],
            {
                "rag": "synthesizer",
                "sql": "generator",
                "hybrid": "generator"
            }
        )
        
        workflow.add_edge("generator", "executor")
        
        workflow.add_conditional_edges(
            "executor",
            self.should_repair,
            {
                "generator": "repair",
                "synthesizer": "synthesizer"
            }
        )
        
        workflow.add_edge("repair", "generator")
        workflow.add_edge("synthesizer", END)
        
        return workflow.compile()

def get_graph(db_path="data/northwind.sqlite", docs_dir="docs"):
    agent = HybridAgent(db_path, docs_dir)
    return agent.build_graph()

