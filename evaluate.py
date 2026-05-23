import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_ROME_PATH = ROOT / "results" / "rome_results.json"
DEFAULT_MEMIT_PATH = ROOT / "results" / "memit_results.json"
DEFAULT_JSON_OUTPUT = ROOT / "results" / "evaluation_summary.json"
DEFAULT_CSV_OUTPUT = ROOT / "results" / "evaluation_summary.csv"
DEFAULT_MARKDOWN_OUTPUT = ROOT / "results" / "evaluation_summary.md"


def parse_args():
    parser = argparse.ArgumentParser(description="Task 4: comprehensive evaluation.")
    parser.add_argument("--rome", type=Path, default=DEFAULT_ROME_PATH)
    parser.add_argument("--memit", type=Path, default=DEFAULT_MEMIT_PATH)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--csv-output", type=Path, default=DEFAULT_CSV_OUTPUT)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MARKDOWN_OUTPUT)
    return parser.parse_args()


def read_json(path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def pct(value):
    return None if value is None else round(value * 100, 2)


def mean(records, key):
    values = [float(record[key]) for record in records if key in record]
    return sum(values) / len(values) if values else None


def count_success(records, key):
    return sum(1 for record in records if float(record.get(key, 0.0)) >= 1.0)


def summarize_rome(result):
    records = result.get("records", [])
    es = result.get("ES", mean(records, "ES"))
    ps = result.get("PS", mean(records, "PS"))
    ns = result.get("NS", mean(records, "NS"))
    return {
        "method": "ROME",
        "num_edits": result.get("num_edits", len(records)),
        "num_evaluated": len(records),
        "ES": es,
        "PS": ps,
        "NS": ns,
        "ES_percent": pct(es),
        "PS_percent": pct(ps),
        "NS_percent": pct(ns),
        "success_count": count_success(records, "ES"),
        "generalization_success_count": count_success(records, "PS"),
        "locality_success_count": count_success(records, "NS"),
        "elapsed_seconds": result.get("elapsed_seconds"),
        "edit_seconds": sum(record.get("edit_seconds", 0.0) for record in records),
        "peak_cuda_memory_mb": result.get("peak_cuda_memory_mb"),
        "notes": "Single-fact editing; each fact was edited after model reload/reset.",
    }


def summarize_memit(result):
    records = result.get("records", [])
    es = result.get("ES", mean(records, "success"))
    ps = result.get("PS", mean(records, "PS"))
    ns = result.get("NS", mean(records, "NS"))
    has_ps_ns = ps is not None and ns is not None
    return {
        "method": "MEMIT",
        "num_edits": result.get("num_edits", len(records)),
        "num_evaluated": result.get("num_evaluated", len(records)),
        "ES": es,
        "PS": ps,
        "NS": ns,
        "ES_percent": pct(es),
        "PS_percent": pct(ps),
        "NS_percent": pct(ns),
        "success_count": count_success(records, "success"),
        "generalization_success_count": count_success(records, "PS")
        if has_ps_ns
        else None,
        "locality_success_count": count_success(records, "NS")
        if has_ps_ns
        else None,
        "elapsed_seconds": result.get("elapsed_seconds"),
        "edit_seconds": result.get("edit_seconds"),
        "peak_cuda_memory_mb": result.get("peak_cuda_memory_mb"),
        "notes": (
            "Batch editing. "
            + (
                "PS/NS were computed from rephrase/locality prompts. "
                if has_ps_ns
                else "PS/NS are unavailable because the MEMIT data lacks rephrase/locality prompts. "
            )
            + "Current run used identity covariance "
            f"={result.get('identity_covariance')} and low_memory_context="
            f"{result.get('low_memory_context')}."
        ),
    }


def failure_examples_rome(result, limit=5):
    examples = []
    for record in result.get("records", []):
        failures = []
        if float(record.get("ES", 0.0)) < 1:
            failures.append("ES")
        if float(record.get("PS", 0.0)) < 1:
            failures.append("PS")
        if float(record.get("NS", 0.0)) < 1:
            failures.append("NS")
        if failures:
            examples.append(
                {
                    "method": "ROME",
                    "failed_metrics": ",".join(failures),
                    "prompt": record.get("prompt"),
                    "target_new": record.get("target_new"),
                    "direct_output": record.get("direct_output"),
                    "rephrase_output": record.get("rephrase_output"),
                    "locality_output": record.get("locality_output"),
                }
            )
    return examples[:limit]


def failure_examples_memit(result, limit=5):
    examples = []
    for record in result.get("records", []):
        failures = []
        if float(record.get("success", 0.0)) < 1:
            failures.append("ES")
        if "PS" in record and float(record.get("PS", 0.0)) < 1:
            failures.append("PS")
        if "NS" in record and float(record.get("NS", 0.0)) < 1:
            failures.append("NS")
        if failures:
            examples.append(
                {
                    "method": "MEMIT",
                    "failed_metrics": ",".join(failures),
                    "prompt": record.get("prompt"),
                    "target_new": record.get("target_new"),
                    "output": record.get("output"),
                    "rephrase_output": record.get("rephrase_output"),
                    "locality_output": record.get("locality_output"),
                }
            )
    return examples[:limit]


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "num_edits",
        "num_evaluated",
        "ES_percent",
        "PS_percent",
        "NS_percent",
        "success_count",
        "generalization_success_count",
        "locality_success_count",
        "elapsed_seconds",
        "edit_seconds",
        "peak_cuda_memory_mb",
        "notes",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def md_value(value):
    return "N/A" if value is None else str(value)


def write_markdown(path, rows, failure_examples):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Task 4 Comprehensive Evaluation",
        "",
        "| Method | Edits | Evaluated | ES | PS | NS | ES Count | PS Count | NS Count | Time (s) | Peak CUDA MB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {method} | {num_edits} | {num_evaluated} | {es} | {ps} | {ns} | "
            "{es_count} | {ps_count} | {ns_count} | {time} | {cuda} |".format(
                method=row["method"],
                num_edits=row["num_edits"],
                num_evaluated=row["num_evaluated"],
                es=md_value(row["ES_percent"]),
                ps=md_value(row["PS_percent"]),
                ns=md_value(row["NS_percent"]),
                es_count=row["success_count"],
                ps_count=md_value(row["generalization_success_count"]),
                ns_count=md_value(row["locality_success_count"]),
                time=round(row["elapsed_seconds"], 2)
                if row.get("elapsed_seconds") is not None
                else "N/A",
                cuda=round(row["peak_cuda_memory_mb"], 2)
                if row.get("peak_cuda_memory_mb") is not None
                else "N/A",
            )
        )

    lines.extend(["", "## Notes", ""])
    for row in rows:
        lines.append(f"- {row['method']}: {row['notes']}")

    lines.extend(["", "## Failure Examples", ""])
    for example in failure_examples:
        lines.append(
            f"- {example['method']} [{example['failed_metrics']}]: "
            f"{example['prompt']} -> {example['target_new']}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    rows = []
    failures = []

    rome = read_json(args.rome)
    if rome is not None:
        rows.append(summarize_rome(rome))
        failures.extend(failure_examples_rome(rome))

    memit = read_json(args.memit)
    if memit is not None:
        rows.append(summarize_memit(memit))
        failures.extend(failure_examples_memit(memit))

    summary = {
        "rows": rows,
        "failure_examples": failures,
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    with args.json_output.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    write_csv(args.csv_output, rows)
    write_markdown(args.markdown_output, rows, failures)

    print("Task 4 Comprehensive Evaluation")
    for row in rows:
        print(
            f"{row['method']}: ES={md_value(row['ES_percent'])}%, "
            f"PS={md_value(row['PS_percent'])}%, NS={md_value(row['NS_percent'])}%, "
            f"counts={row['success_count']}/"
            f"{md_value(row['generalization_success_count'])}/"
            f"{md_value(row['locality_success_count'])}"
        )
    print(f"Saved JSON: {args.json_output}")
    print(f"Saved CSV: {args.csv_output}")
    print(f"Saved Markdown: {args.markdown_output}")


if __name__ == "__main__":
    main()
