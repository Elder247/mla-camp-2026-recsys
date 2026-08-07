#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path


TOKEN_PATTERN = re.compile(r"(?:y[01]_|t[01]_|AQAD-)[A-Za-z0-9_\-]+")


def mask(text: object) -> str:
    return TOKEN_PATTERN.sub("***", str(text))


def print_issues(issues: object, indent: int = 0) -> None:
    for issue in issues or []:
        message = getattr(issue, "message", None) or issue
        print(f"{' ' * indent}- {mask(message)}", file=sys.stderr)
        nested = getattr(issue, "issues", None)
        if nested:
            print_issues(nested, indent + 2)


def validate(query: str) -> bool:
    from yql.client.explain import YqlSqlValidateRequest

    operation = YqlSqlValidateRequest(query, sql=True, syntax_version=1)
    operation.run()
    operation.get_results()
    if operation.is_success:
        print("YQL validation: OK")
        return True
    print("YQL validation: FAILED", file=sys.stderr)
    print_issues(operation.errors)
    return False


def run(query: str, token_path: Path, output: Path) -> None:
    from yql.api.v1.client import YqlClient

    client = YqlClient(
        server="yql.yandex.net",
        port=443,
        token_path=str(token_path),
    )
    operation = client.query(query, title="YQL ML Camp history candidates")
    operation.run()
    results = operation.get_results()
    if not operation.is_success:
        print_issues(operation.errors)
        raise RuntimeError("YQL operation failed")

    frames = []
    for table in results:
        table.fetch_full_data()
        frames.append(table.full_dataframe)
    if len(frames) != 1:
        raise RuntimeError(f"Expected one result table, got {len(frames)}")

    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].to_parquet(output, index=False)
    print(f"YQL rows: {len(frames[0])}")
    print(f"YQL artifact: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and run a SELECT-only YQL file")
    parser.add_argument("query", type=Path)
    parser.add_argument("--token-path", type=Path, default=Path.home() / ".yql" / "token")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    os.environ["YQL_TOKEN_PATH"] = str(args.token_path)
    query = args.query.read_text(encoding="utf-8")
    if not validate(query):
        raise SystemExit(1)
    if args.validate_only:
        return
    if args.output is None:
        parser.error("--output is required unless --validate-only is set")
    run(query, args.token_path, args.output)


if __name__ == "__main__":
    main()
