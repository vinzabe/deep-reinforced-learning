"""CLI: `python -m hfsupply.cli scan <model-dir>`."""
from __future__ import annotations

import argparse
import json
import sys

from .analyst import LLMModelAnalyst
from .findings import Severity
from .pipeline import RepoScanner


def _cmd_scan(args: argparse.Namespace) -> int:
    report = RepoScanner().scan(args.path)
    data = report.to_dict()
    if args.analyse:
        verdict = LLMModelAnalyst().analyse(data)
        data["verdict"] = verdict.to_dict()
    txt = json.dumps(data, indent=2)
    if args.output == "-":
        print(txt)
    else:
        with open(args.output, "w") as fh:
            fh.write(txt)
    print(
        f"files={report.total_files} findings={report.total_findings} "
        f"max_severity={report.highest_severity().value}",
        file=sys.stderr,
    )
    if args.fail_on_high:
        return 1 if report.highest_severity().rank >= Severity.HIGH.rank else 0
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser("hfsupply")
    sub = parser.add_subparsers(dest="cmd", required=True)
    scan = sub.add_parser("scan")
    scan.add_argument("path")
    scan.add_argument("-o", "--output", default="-")
    scan.add_argument("--analyse", action="store_true")
    scan.add_argument("--fail-on-high", action="store_true")
    scan.set_defaults(fn=_cmd_scan)
    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
