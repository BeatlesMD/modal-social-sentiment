#!/usr/bin/env python3
"""
Helper script to query DuckDB via Modal.

Usage:
    python scripts/query_db.py "SELECT COUNT(*) FROM messages"
    python scripts/query_db.py --schema
    python scripts/query_db.py --examples
"""

import json
import sys

import modal

# Import the query functions from the deployed app
query_duckdb = modal.Function.from_name("modal-social-sentiment", "query_duckdb")
get_schema = modal.Function.from_name("modal-social-sentiment", "get_schema")
get_sample_queries = modal.Function.from_name("modal-social-sentiment", "get_sample_queries")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--schema":
        print("Fetching database schema...\n")
        schema = get_schema.remote()
        print(json.dumps(schema, indent=2))
        return

    if len(sys.argv) > 1 and sys.argv[1] == "--examples":
        print("Example queries:\n")
        examples = get_sample_queries.remote()
        print(json.dumps(examples, indent=2))
        return

    if len(sys.argv) < 2:
        print("Usage:")
        print('  python scripts/query_db.py "SELECT ..."')
        print("  python scripts/query_db.py --schema")
        print("  python scripts/query_db.py --examples")
        sys.exit(1)

    sql = sys.argv[1]
    print(f"Executing: {sql}\n")

    try:
        result = query_duckdb.remote(sql=sql, limit=1000, format="json")
        print(f"✓ Rows: {result['row_count']}")
        print(f"✓ Columns: {', '.join(result['columns'])}\n")
        print("Results:")
        print(json.dumps(result["rows"], indent=2, default=str))
    except Exception as e:
        print(f"✗ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
