import argparse
import json
import random
from pathlib import Path

import requests


DEFAULT_URL = "https://rome.baulab.info/data/dsets/counterfact.json"
DEFAULT_OUTPUT = Path("data/batch_data_eval.json")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare a CounterFact subset with rephrase/locality fields."
    )
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def format_prompt(template, subject):
    if "{}" in template:
        return template.format(subject)
    return template


def get_target_text(value):
    if isinstance(value, dict):
        return value.get("str") or value.get("text") or value.get("answer")
    return value


def get_rephrase_prompt(item, fallback_prompt):
    prompts = item.get("paraphrase_prompts") or item.get("rephrase_prompts") or []
    for prompt in prompts:
        if isinstance(prompt, str) and prompt.strip() and prompt != fallback_prompt:
            return prompt
    return fallback_prompt


def convert_item(item):
    rewrite = item["requested_rewrite"]
    subject = rewrite["subject"]
    prompt = format_prompt(rewrite["prompt"], subject)
    target_new = get_target_text(rewrite["target_new"])
    ground_truth = get_target_text(rewrite["target_true"])

    return {
        "prompt": prompt,
        "subject": subject,
        "target_new": target_new,
        "ground_truth": ground_truth,
        "rephrase_prompt": get_rephrase_prompt(item, prompt),
    }


def attach_rotated_locality(records):
    """Use another sampled fact as locality to test unrelated-fact preservation."""
    if len(records) < 2:
        raise ValueError("At least two records are required to attach locality prompts.")

    total = len(records)
    for idx, record in enumerate(records):
        offset = 1
        while offset < total:
            neighbor = records[(idx + offset) % total]
            if (
                neighbor["subject"] != record["subject"]
                and neighbor["ground_truth"] != record["target_new"]
            ):
                record["locality_prompt"] = neighbor["prompt"]
                record["locality_ground_truth"] = neighbor["ground_truth"]
                break
            offset += 1
        if "locality_prompt" not in record:
            neighbor = records[(idx + 1) % total]
            record["locality_prompt"] = neighbor["prompt"]
            record["locality_ground_truth"] = neighbor["ground_truth"]


def prepare_counterfact_subset(url, output, sample_size, seed):
    print(f"Downloading CounterFact from {url} ...")
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    dataset = response.json()
    print(f"Loaded {len(dataset)} raw records.")

    random.seed(seed)
    random.shuffle(dataset)

    formatted = []
    skipped = 0
    for item in dataset:
        record = convert_item(item)
        if not record["rephrase_prompt"] or record["rephrase_prompt"] == record["prompt"]:
            skipped += 1
            continue
        formatted.append(record)
        if len(formatted) >= sample_size:
            break

    if len(formatted) < sample_size:
        raise RuntimeError(
            f"Only collected {len(formatted)} usable records; skipped {skipped}."
        )

    attach_rotated_locality(formatted)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(formatted, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(formatted)} records to {output}")
    print(f"Skipped {skipped} records without usable rephrase/locality fields.")


def main():
    args = parse_args()
    prepare_counterfact_subset(args.url, args.output, args.sample_size, args.seed)


if __name__ == "__main__":
    main()
