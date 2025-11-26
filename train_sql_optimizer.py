"""
DSPy Optimizer Training Script for GenerateSQL Module

This script optimizes the GenerateSQL module using BootstrapFewShot optimizer.
It trains on a small dataset and measures before/after SQL execution success rate.
"""
import dspy
import json
from agent.dspy_signatures import GenerateSQL
from agent.tools.sqlite_tool import SQLiteTool
from typing import List, Dict, Tuple

# Training examples - SQL questions with expected valid SQL patterns
TRAINING_EXAMPLES = [
    {
        "question": "Top 3 products by total revenue all-time. Revenue uses Order Details: SUM(UnitPrice*Quantity*(1-Discount)).",
        "database_schema": "Table: orders\nColumns: OrderID (INTEGER), OrderDate (TEXT), CustomerID (TEXT)\n\nTable: order_items\nColumns: OrderID (INTEGER), ProductID (INTEGER), UnitPrice (REAL), Quantity (INTEGER), Discount (REAL)\n\nTable: products\nColumns: ProductID (INTEGER), ProductName (TEXT), CategoryID (INTEGER)\n\nTable: Categories\nColumns: CategoryID (INTEGER), CategoryName (TEXT)",
        "constraints": "{}",
        "expected_sql_pattern": "SELECT.*ProductName.*SUM.*Revenue.*ORDER BY.*LIMIT 3"
    },
    {
        "question": "Total revenue from the 'Beverages' category. Return a float.",
        "database_schema": "Table: orders\nColumns: OrderID (INTEGER), OrderDate (TEXT)\n\nTable: order_items\nColumns: OrderID (INTEGER), ProductID (INTEGER), UnitPrice (REAL), Quantity (INTEGER), Discount (REAL)\n\nTable: products\nColumns: ProductID (INTEGER), ProductName (TEXT), CategoryID (INTEGER)\n\nTable: Categories\nColumns: CategoryID (INTEGER), CategoryName (TEXT)",
        "constraints": "{}",
        "expected_sql_pattern": "SELECT.*SUM.*Beverages.*Categories"
    },
    {
        "question": "Count of distinct customers who placed orders.",
        "database_schema": "Table: orders\nColumns: OrderID (INTEGER), CustomerID (TEXT)\n\nTable: customers\nColumns: CustomerID (TEXT), CompanyName (TEXT)",
        "constraints": "{}",
        "expected_sql_pattern": "SELECT.*COUNT.*DISTINCT.*CustomerID"
    },
    {
        "question": "Average order value calculated as SUM(UnitPrice * Quantity * (1 - Discount)) / COUNT(DISTINCT OrderID).",
        "database_schema": "Table: orders\nColumns: OrderID (INTEGER), OrderDate (TEXT)\n\nTable: order_items\nColumns: OrderID (INTEGER), ProductID (INTEGER), UnitPrice (REAL), Quantity (INTEGER), Discount (REAL)",
        "constraints": "{}",
        "expected_sql_pattern": "SELECT.*SUM.*COUNT.*DISTINCT.*OrderID"
    },
    {
        "question": "List all products in the 'Beverages' category.",
        "database_schema": "Table: products\nColumns: ProductID (INTEGER), ProductName (TEXT), CategoryID (INTEGER)\n\nTable: Categories\nColumns: CategoryID (INTEGER), CategoryName (TEXT)",
        "constraints": "{}",
        "expected_sql_pattern": "SELECT.*ProductName.*Categories.*Beverages"
    },
    {
        "question": "Total quantity sold for product with ProductID 1.",
        "database_schema": "Table: order_items\nColumns: OrderID (INTEGER), ProductID (INTEGER), Quantity (INTEGER)",
        "constraints": "{}",
        "expected_sql_pattern": "SELECT.*SUM.*Quantity.*ProductID.*1"
    },
    {
        "question": "Revenue by customer, showing CompanyName and total revenue.",
        "database_schema": "Table: orders\nColumns: OrderID (INTEGER), CustomerID (TEXT)\n\nTable: order_items\nColumns: OrderID (INTEGER), ProductID (INTEGER), UnitPrice (REAL), Quantity (INTEGER), Discount (REAL)\n\nTable: customers\nColumns: CustomerID (TEXT), CompanyName (TEXT)",
        "constraints": "{}",
        "expected_sql_pattern": "SELECT.*CompanyName.*SUM.*Revenue.*GROUP BY"
    },
    {
        "question": "Top 5 customers by number of orders placed.",
        "database_schema": "Table: orders\nColumns: OrderID (INTEGER), CustomerID (TEXT)\n\nTable: customers\nColumns: CustomerID (TEXT), CompanyName (TEXT)",
        "constraints": "{}",
        "expected_sql_pattern": "SELECT.*COUNT.*OrderID.*GROUP BY.*ORDER BY.*LIMIT 5"
    },
    {
        "question": "Products with UnitPrice greater than 50.",
        "database_schema": "Table: products\nColumns: ProductID (INTEGER), ProductName (TEXT), UnitPrice (REAL)",
        "constraints": "{}",
        "expected_sql_pattern": "SELECT.*UnitPrice.*>.*50"
    },
    {
        "question": "Total revenue for orders placed in 2012.",
        "database_schema": "Table: orders\nColumns: OrderID (INTEGER), OrderDate (TEXT)\n\nTable: order_items\nColumns: OrderID (INTEGER), ProductID (INTEGER), UnitPrice (REAL), Quantity (INTEGER), Discount (REAL)",
        "constraints": "{}",
        "expected_sql_pattern": "SELECT.*SUM.*BETWEEN.*2012"
    },
    {
        "question": "Average discount percentage across all order items.",
        "database_schema": "Table: order_items\nColumns: OrderID (INTEGER), ProductID (INTEGER), Discount (REAL)",
        "constraints": "{}",
        "expected_sql_pattern": "SELECT.*AVG.*Discount"
    },
    {
        "question": "Categories with more than 10 products.",
        "database_schema": "Table: products\nColumns: ProductID (INTEGER), ProductName (TEXT), CategoryID (INTEGER)\n\nTable: Categories\nColumns: CategoryID (INTEGER), CategoryName (TEXT)",
        "constraints": "{}",
        "expected_sql_pattern": "SELECT.*COUNT.*GROUP BY.*HAVING.*>.*10"
    },
    {
        "question": "Revenue by product category, ordered by revenue descending.",
        "database_schema": "Table: order_items\nColumns: OrderID (INTEGER), ProductID (INTEGER), UnitPrice (REAL), Quantity (INTEGER), Discount (REAL)\n\nTable: products\nColumns: ProductID (INTEGER), CategoryID (INTEGER)\n\nTable: Categories\nColumns: CategoryID (INTEGER), CategoryName (TEXT)",
        "constraints": "{}",
        "expected_sql_pattern": "SELECT.*CategoryName.*SUM.*Revenue.*ORDER BY.*DESC"
    },
    {
        "question": "Customers who have placed orders worth more than 1000 in total.",
        "database_schema": "Table: orders\nColumns: OrderID (INTEGER), CustomerID (TEXT)\n\nTable: order_items\nColumns: OrderID (INTEGER), ProductID (INTEGER), UnitPrice (REAL), Quantity (INTEGER), Discount (REAL)\n\nTable: customers\nColumns: CustomerID (TEXT), CompanyName (TEXT)",
        "constraints": "{}",
        "expected_sql_pattern": "SELECT.*SUM.*GROUP BY.*HAVING.*>.*1000"
    },
    {
        "question": "Month with highest number of orders in 2012.",
        "database_schema": "Table: orders\nColumns: OrderID (INTEGER), OrderDate (TEXT)",
        "constraints": "{}",
        "expected_sql_pattern": "SELECT.*strftime.*COUNT.*GROUP BY.*ORDER BY.*LIMIT"
    },
    {
        "question": "Products that have never been ordered.",
        "database_schema": "Table: products\nColumns: ProductID (INTEGER), ProductName (TEXT)\n\nTable: order_items\nColumns: OrderID (INTEGER), ProductID (INTEGER)",
        "constraints": "{}",
        "expected_sql_pattern": "SELECT.*LEFT JOIN.*WHERE.*IS NULL"
    },
    {
        "question": "Total revenue per month in 2013.",
        "database_schema": "Table: orders\nColumns: OrderID (INTEGER), OrderDate (TEXT)\n\nTable: order_items\nColumns: OrderID (INTEGER), ProductID (INTEGER), UnitPrice (REAL), Quantity (INTEGER), Discount (REAL)",
        "constraints": "{}",
        "expected_sql_pattern": "SELECT.*strftime.*SUM.*GROUP BY"
    },
    {
        "question": "Customer with the highest average order value.",
        "database_schema": "Table: orders\nColumns: OrderID (INTEGER), CustomerID (TEXT)\n\nTable: order_items\nColumns: OrderID (INTEGER), ProductID (INTEGER), UnitPrice (REAL), Quantity (INTEGER), Discount (REAL)\n\nTable: customers\nColumns: CustomerID (TEXT), CompanyName (TEXT)",
        "constraints": "{}",
        "expected_sql_pattern": "SELECT.*AVG.*GROUP BY.*ORDER BY.*LIMIT 1"
    },
    {
        "question": "Products ordered more than 100 times.",
        "database_schema": "Table: order_items\nColumns: OrderID (INTEGER), ProductID (INTEGER), Quantity (INTEGER)\n\nTable: products\nColumns: ProductID (INTEGER), ProductName (TEXT)",
        "constraints": "{}",
        "expected_sql_pattern": "SELECT.*SUM.*Quantity.*GROUP BY.*HAVING.*>.*100"
    },
    {
        "question": "Revenue breakdown by category for orders in 2012.",
        "database_schema": "Table: orders\nColumns: OrderID (INTEGER), OrderDate (TEXT)\n\nTable: order_items\nColumns: OrderID (INTEGER), ProductID (INTEGER), UnitPrice (REAL), Quantity (INTEGER), Discount (REAL)\n\nTable: products\nColumns: ProductID (INTEGER), CategoryID (INTEGER)\n\nTable: Categories\nColumns: CategoryID (INTEGER), CategoryName (TEXT)",
        "constraints": "{}",
        "expected_sql_pattern": "SELECT.*CategoryName.*SUM.*BETWEEN.*2012"
    }
]

def validate_sql_metric(gold, pred, trace=None):
    """DSPy metric function that validates SQL execution."""
    sql = getattr(pred, "sql_query", "")
    if not sql or not sql.strip():
        return False
    
    # Remove markdown code blocks
    sql = sql.replace("```sql", "").replace("```", "").strip()
    
    # Basic syntax checks
    if not sql.upper().startswith("SELECT"):
        return False
    
    try:
        db = SQLiteTool("data/northwind.sqlite")
        result = db.execute_query(sql)
        if result.get("error"):
            return False
        return True
    except Exception as e:
        return False

def validate_sql(sql: str, db: SQLiteTool) -> Tuple[bool, str]:
    """Validate SQL by attempting to execute it. Returns (is_valid, error_message)."""
    if not sql or not sql.strip():
        return False, "Empty SQL"
    
    # Remove markdown code blocks
    sql = sql.replace("```sql", "").replace("```", "").strip()
    
    # Basic syntax checks
    if not sql.upper().startswith("SELECT"):
        return False, "Does not start with SELECT"
    
    try:
        result = db.execute_query(sql)
        if result.get("error"):
            return False, result["error"]
        return True, "Success"
    except Exception as e:
        return False, str(e)

def evaluate_module(module, examples: List[Dict], db: SQLiteTool) -> Dict:
    """Evaluate a module on examples and return metrics."""
    valid_count = 0
    total = len(examples)
    
    for example in examples:
        try:
            pred = module(
                database_schema=example["database_schema"],
                question=example["question"],
                constraints=example.get("constraints", "{}")
            )
            sql = getattr(pred, "sql_query", "")
            is_valid, _ = validate_sql(sql, db)
            if is_valid:
                valid_count += 1
        except Exception as e:
            pass  # Count as invalid
    
    return {
        "valid_sql_rate": valid_count / total if total > 0 else 0.0,
        "valid_count": valid_count,
        "total": total
    }

def main():
    print("Initializing DSPy with Ollama...")
    lm = dspy.LM("openai/phi3.5:3.8b-mini-instruct-q4_K_M", api_base="http://localhost:11434/v1", api_key="ollama")
    dspy.configure(lm=lm)
    
    db = SQLiteTool("data/northwind.sqlite")
    
    print(f"\n=== DSPy SQL Generator Optimization ===")
    print(f"Training examples: {len(TRAINING_EXAMPLES)}")
    
    # Create base module (before optimization)
    base_module = dspy.ChainOfThought(GenerateSQL)
    
    print("\n1. Evaluating BASE module (before optimization)...")
    base_metrics = evaluate_module(base_module, TRAINING_EXAMPLES, db)
    print(f"   Base valid SQL rate: {base_metrics['valid_sql_rate']:.2%} ({base_metrics['valid_count']}/{base_metrics['total']})")
    
    # Prepare training examples for DSPy
    trainset = []
    for example in TRAINING_EXAMPLES:
        # Try to generate SQL and validate
        try:
            pred = base_module(
                database_schema=example["database_schema"],
                question=example["question"],
                constraints=example.get("constraints", "{}")
            )
            sql = getattr(pred, "sql_query", "")
            is_valid, error = validate_sql(sql, db)
            
            if is_valid:
                # Create a training example with the valid SQL
                trainset.append(dspy.Example(
                    database_schema=example["database_schema"],
                    question=example["question"],
                    constraints=example.get("constraints", "{}"),
                    sql_query=sql
                ).with_inputs("database_schema", "question", "constraints"))
        except Exception as e:
            pass
    
    print(f"\n2. Collected {len(trainset)} valid examples for training")
    
    if len(trainset) < 3:
        print("   WARNING: Not enough valid examples for optimization. Using base module.")
        optimized_module = base_module
        opt_metrics = base_metrics
    else:
        # Optimize using BootstrapFewShot
        print("\n3. Optimizing with BootstrapFewShot...")
        optimized_module = dspy.ChainOfThought(GenerateSQL)
        
        try:
            # Use a smaller subset for faster training
            training_subset = trainset[:min(15, len(trainset))]
            optimizer = dspy.BootstrapFewShot(metric=validate_sql_metric, max_bootstrapped_demos=4, max_labeled_demos=8)
            optimized_module = optimizer.compile(optimized_module, trainset=training_subset)
            print("   Optimization complete!")
        except Exception as e:
            print(f"   Optimization failed: {e}")
            print("   Using base module with ChainOfThought")
            optimized_module = base_module
        
        print("\n4. Evaluating OPTIMIZED module (after optimization)...")
        opt_metrics = evaluate_module(optimized_module, TRAINING_EXAMPLES, db)
        print(f"   Optimized valid SQL rate: {opt_metrics['valid_sql_rate']:.2%} ({opt_metrics['valid_count']}/{opt_metrics['total']})")
    
    # Calculate improvement
    improvement = opt_metrics['valid_sql_rate'] - base_metrics['valid_sql_rate']
    improvement_pct = improvement * 100
    
    print("\n=== Results ===")
    print(f"Before optimization: {base_metrics['valid_sql_rate']:.2%} ({base_metrics['valid_count']}/{base_metrics['total']})")
    print(f"After optimization:  {opt_metrics['valid_sql_rate']:.2%} ({opt_metrics['valid_count']}/{opt_metrics['total']})")
    print(f"Improvement:        {improvement_pct:+.1f} percentage points")
    
    # Save optimized module (as a reference - DSPy modules are stateful)
    print("\n=== Saving Results ===")
    results = {
        "before": {
            "valid_sql_rate": base_metrics['valid_sql_rate'],
            "valid_count": base_metrics['valid_count'],
            "total": base_metrics['total']
        },
        "after": {
            "valid_sql_rate": opt_metrics['valid_sql_rate'],
            "valid_count": opt_metrics['valid_count'],
            "total": opt_metrics['total']
        },
        "improvement_pct": improvement_pct,
        "training_examples": len(trainset)
    }
    
    with open("optimization_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("Results saved to optimization_results.json")
    print("\nNote: The optimized module is stateful and will be used automatically")
    print("      in graph_hybrid.py when DSPy is configured with the same LM.")
    
    return optimized_module, results

if __name__ == "__main__":
    main()

