import dspy
from typing import List, Literal

class Router(dspy.Signature):
    """Classify a user question into one of three categories:
    - rag: The question can be answered purely from documentation (policies, calendars, definitions). No database access needed.
    - sql: The question requires aggregation or data retrieval from the database (orders, products, customers). No specific policy/definition context needed.
    - hybrid: The question requires both documentation (e.g. for definitions, date ranges, special categories) AND database access.
    """
    question = dspy.InputField(desc="The user's question.")
    classification = dspy.OutputField(desc="One of: rag, sql, hybrid")

class GenerateSQL(dspy.Signature):
    """Generate a valid SQLite query based on the schema and user question.
    
    CRITICAL RULES:
    1. USE ONLY THESE LOWERCASE VIEWS:
       - 'orders' (OrderID, OrderDate, ShippedDate, CustomerID)
       - 'order_items' (OrderID, ProductID, UnitPrice, Quantity, Discount)
       - 'products' (ProductID, ProductName, CategoryID, UnitPrice) - NOTE: NO CategoryName, must join Categories
       - 'customers' (CustomerID, CompanyName)
       - 'Categories' table (CategoryID, CategoryName) - use this for category names
    
    2. DO NOT USE tables like "Order Details", "Products", "Orders" (Capitalized).
    3. SQLITE SYNTAX ONLY - NO PostgreSQL syntax like ::timestamp, ::date, etc. Use SQLite functions.
    4. JOIN PATTERNS:
       - orders.OrderID = order_items.OrderID
       - order_items.ProductID = products.ProductID
       - products.CategoryID = Categories.CategoryID (for CategoryName)
    5. REVENUE FORMULA: SUM(order_items.UnitPrice * order_items.Quantity * (1 - order_items.Discount))
    6. AOV FORMULA: SUM(order_items.UnitPrice * order_items.Quantity * (1 - order_items.Discount)) / COUNT(DISTINCT orders.OrderID)
    7. DATE FORMAT: Use orders.OrderDate. Filter with: orders.OrderDate BETWEEN 'YYYY-MM-DD' AND 'YYYY-MM-DD'
    8. For date filtering, ALWAYS JOIN orders: orders.OrderID = order_items.OrderID
    9. Output ONLY the SQL string, no markdown, no PostgreSQL syntax.
    """
    database_schema = dspy.InputField(desc="The database schema.")
    question = dspy.InputField(desc="The user's question.")
    constraints = dspy.InputField(desc="constraints", optional=True)
    sql_query = dspy.OutputField(desc="SQL query starting with SELECT...")

class SynthesizeAnswer(dspy.Signature):
    """Generate a final answer based on the retrieved context and/or SQL execution results.
    Ensure the answer adheres strictly to the requested format (e.g. integer, float, specific JSON structure).
    Include citations for all sources used (tables and doc chunks).
    """
    question = dspy.InputField()
    format_hint = dspy.InputField(desc="The required format of the output.")
    context = dspy.InputField(desc="Retrieved text chunks from documentation.", optional=True)
    sql_query = dspy.InputField(desc="The SQL query executed, if any.", optional=True)
    sql_result = dspy.InputField(desc="The result rows from the SQL query.", optional=True)
    
    final_answer = dspy.OutputField(desc="The precise answer matching the format hint.")
    explanation = dspy.OutputField(desc="A brief explanation (<= 2 sentences).")
    citations = dspy.OutputField(desc="List of strings: table names and doc chunk IDs used.")

class ExtractConstraints(dspy.Signature):
    """Extract specific constraints from the question and documentation to help with SQL generation.
    Look for date ranges, specific category names, or KPI definitions.
    """
    question = dspy.InputField()
    context = dspy.InputField(desc="Retrieved documentation chunks.")
    
    date_range_start = dspy.OutputField(desc="YYYY-MM-DD or None", optional=True)
    date_range_end = dspy.OutputField(desc="YYYY-MM-DD or None", optional=True)
    kpi_formula = dspy.OutputField(desc="Description of how to calculate the metric", optional=True)
    entities = dspy.OutputField(desc="List of specific entities (categories, products) mentioned", optional=True)

