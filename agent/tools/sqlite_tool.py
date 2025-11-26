import sqlite3
from typing import List, Dict, Any, Optional

class SQLiteTool:
    def __init__(self, db_path: str):
        self.db_path = db_path

    def get_schema(self) -> str:
        """Returns the schema of the database including table names and columns."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get list of tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' OR type='view';")
            tables = cursor.fetchall()
            
            schema_str = ""
            for table in tables:
                table_name = table[0]
                # Skip sqlite internal tables
                if table_name.startswith('sqlite_'):
                    continue
                    
                cursor.execute(f"PRAGMA table_info('{table_name}')")
                columns = cursor.fetchall()
                
                column_strs = []
                for col in columns:
                    column_strs.append(f"{col[1]} ({col[2]})")
                
                schema_str += f"Table: {table_name}\nColumns: {', '.join(column_strs)}\n\n"
            
            conn.close()
            return schema_str
        except Exception as e:
            return f"Error getting schema: {str(e)}"

    def execute_query(self, query: str) -> Dict[str, Any]:
        """
        Executes a SQL query and returns the results.
        Returns a dictionary with keys: 'columns', 'rows', 'error'.
        """
        try:
            if ';' in query.strip('; '):
                 pass
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(query)
            
            columns = [description[0] for description in cursor.description] if cursor.description else []
            rows = cursor.fetchall()
            
            conn.close()
            
            # Convert rows to list of dicts for easier consumption
            result_rows = []
            for row in rows:
                result_rows.append(dict(zip(columns, row)))
                
            return {
                "columns": columns,
                "rows": result_rows,
                "error": None
            }
        except Exception as e:
            return {
                "columns": [],
                "rows": [],
                "error": str(e)
            }

