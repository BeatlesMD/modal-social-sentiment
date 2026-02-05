"""
Modal functions for querying DuckDB data.

These functions allow running arbitrary SQL queries against the sentiment database
for exploration and analysis.
"""

import json
from pathlib import Path

import modal

from modal_app.common import base_image, data_volume
from src.config import DUCKDB_PATH

app = modal.App()
MAX_QUERY_LIMIT = 1000


@app.function(
    image=base_image,
    volumes={"/data": data_volume},
    timeout=300,
)
def query_duckdb(
    sql: str,
    limit: int = 1000,
    format: str = "json",
) -> dict:
    """
    Execute a SQL query against the DuckDB database.
    
    Args:
        sql: SQL query to execute (SELECT statements only for safety)
        limit: Maximum number of rows to return (default: 1000)
        format: Output format - "json" (list of dicts) or "table" (columnar)
    
    Returns:
        Dictionary with:
        - columns: List of column names
        - rows: Query results (format depends on format param)
        - row_count: Number of rows returned
        - query: The executed query (for reference)
    """
    import duckdb
    import structlog

    logger = structlog.get_logger()

    normalized_sql = sql.strip().rstrip(";").strip()
    if not normalized_sql:
        raise ValueError("SQL query cannot be empty")

    sql_upper = normalized_sql.upper()
    if not (sql_upper.startswith("SELECT") or sql_upper.startswith("WITH")):
        raise ValueError("Only SELECT/CTE queries are allowed for safety")

    # Block multiple statements; only a single read query is allowed.
    if ";" in normalized_sql:
        raise ValueError("Only a single SELECT statement is allowed")

    try:
        safe_limit = int(limit)
    except (TypeError, ValueError) as exc:
        raise ValueError("limit must be an integer") from exc

    safe_limit = max(1, min(safe_limit, MAX_QUERY_LIMIT))
    executed_sql = f"SELECT * FROM ({normalized_sql}) AS query_result LIMIT {safe_limit}"

    if format not in {"json", "table"}:
        raise ValueError("format must be 'json' or 'table'")

    db_path = DUCKDB_PATH
    if not Path(db_path).exists():
        raise FileNotFoundError(f"DuckDB file not found at {db_path}")

    try:
        conn = duckdb.connect(db_path, read_only=True)
        cursor = conn.execute(executed_sql)

        # Get column names
        columns = [desc[0] for desc in cursor.description] if cursor.description else []

        # Fetch results
        rows = cursor.fetchall()

        # Format results based on requested format
        if format == "json":
            # Convert to list of dictionaries
            result_rows = [dict(zip(columns, row)) for row in rows]
        else:
            # Return as list of tuples (table format)
            result_rows = rows

        logger.info("Query executed successfully", row_count=len(rows), columns=len(columns))

        return {
            "columns": columns,
            "rows": result_rows,
            "row_count": len(rows),
            "query": executed_sql,
            "requested_query": normalized_sql,
            "limit_applied": safe_limit,
            "format": format,
        }

    except Exception as e:
        logger.error("Query execution failed", error=str(e), query=normalized_sql)
        raise
    finally:
        if "conn" in locals():
            conn.close()


@app.function(
    image=base_image,
    volumes={"/data": data_volume},
    timeout=60,
)
def get_schema() -> dict:
    """
    Get the database schema (tables and columns).
    
    Returns:
        Dictionary with table schemas
    """
    import duckdb
    import structlog

    logger = structlog.get_logger()

    db_path = DUCKDB_PATH
    if not Path(db_path).exists():
        raise FileNotFoundError(f"DuckDB file not found at {db_path}")

    try:
        conn = duckdb.connect(db_path, read_only=True)

        # Get all tables
        tables_result = conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
        ).fetchall()

        tables = {}
        for (table_name,) in tables_result:
            # Get columns for each table
            columns_result = conn.execute(
                f"""
                SELECT 
                    column_name,
                    data_type,
                    is_nullable
                FROM information_schema.columns
                WHERE table_name = ?
                ORDER BY ordinal_position
                """,
                [table_name],
            ).fetchall()

            tables[table_name] = [
                {
                    "name": col_name,
                    "type": data_type,
                    "nullable": is_nullable == "YES",
                }
                for col_name, data_type, is_nullable in columns_result
            ]

        conn.close()

        logger.info("Schema retrieved", table_count=len(tables))

        return {"tables": tables}

    except Exception as e:
        logger.error("Schema retrieval failed", error=str(e))
        raise


@app.function(
    image=base_image,
    volumes={"/data": data_volume},
    timeout=60,
)
def get_sample_queries() -> dict:
    """
    Get example queries for exploring the data.
    
    Returns:
        Dictionary with categorized example queries
    """
    return {
        "basic": {
            "total_messages": "SELECT COUNT(*) as total FROM messages",
            "by_source": """
                SELECT source, COUNT(*) as count
                FROM messages
                GROUP BY source
                ORDER BY count DESC
            """,
            "by_sentiment": """
                SELECT sentiment_simple, COUNT(*) as count
                FROM messages
                WHERE sentiment_simple IS NOT NULL
                GROUP BY sentiment_simple
                ORDER BY count DESC
            """,
        },
        "recent": {
            "recent_negative": """
                SELECT title, source, sentiment_simple, created_at
                FROM messages
                WHERE sentiment_simple = 'negative'
                  AND created_at >= CURRENT_DATE - INTERVAL '7 days'
                ORDER BY created_at DESC
                LIMIT 20
            """,
            "recent_questions": """
                SELECT title, source, created_at, url
                FROM messages
                WHERE content_type = 'question'
                  AND created_at >= CURRENT_DATE - INTERVAL '30 days'
                ORDER BY created_at DESC
                LIMIT 20
            """,
        },
        "topics": {
            "topic_distribution": """
                SELECT topic, COUNT(*) as count
                FROM messages m,
                UNNEST(from_json(m.topics, '["VARCHAR"]')) AS t(topic)
                WHERE m.topics IS NOT NULL AND m.topics != '[]'
                GROUP BY topic
                ORDER BY count DESC
            """,
            "negative_by_topic": """
                SELECT topic, COUNT(*) as negative_count
                FROM messages m,
                UNNEST(from_json(m.topics, '["VARCHAR"]')) AS t(topic)
                WHERE m.sentiment_simple = 'negative'
                  AND m.topics IS NOT NULL AND m.topics != '[]'
                GROUP BY topic
                ORDER BY negative_count DESC
            """,
        },
        "analytics": {
            "sentiment_over_time": """
                SELECT 
                    DATE_TRUNC('day', created_at)::DATE as date,
                    COUNT(*) FILTER (WHERE sentiment_simple = 'positive') as positive,
                    COUNT(*) FILTER (WHERE sentiment_simple = 'negative') as negative,
                    COUNT(*) FILTER (WHERE sentiment_simple = 'neutral') as neutral
                FROM messages
                WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
                GROUP BY DATE_TRUNC('day', created_at)
                ORDER BY date
            """,
            "unanswered_questions": """
                SELECT m.id, m.title, m.source, m.created_at, m.url
                FROM messages m
                WHERE m.content_type = 'question'
                  AND m.created_at >= CURRENT_DATE - INTERVAL '30 days'
                  AND NOT EXISTS (
                      SELECT 1
                      FROM messages r
                      WHERE r.parent_id = m.id OR r.thread_id = m.id
                  )
                ORDER BY m.created_at DESC
                LIMIT 50
            """,
        },
    }
