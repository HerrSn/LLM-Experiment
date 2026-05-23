import argparse
import gc
import json
import sys
import time
import tracemalloc
from pathlib import Path

import edit_rome


ROOT = Path(__file__).resolve().parent
EASYEDIT_DIR = ROOT / "EasyEdit"
DEFAULT_DATA_PATH = ROOT / "data" / "batch_data_eval.json"
DEFAULT_HPARAMS_PATH = ROOT / "hparams" / "MEMIT" / "qwen2.5-0.5b.yaml"
DEFAULT_OUTPUT_PATH = ROOT / "results" / "memit_results.json"

BaseEditor = None
MEMITHyperParams = None
nethook = None
torch = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Task 3: batch knowledge editing with EasyEdit MEMIT."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--hparams", type=Path, default=DEFAULT_HPARAMS_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--max-new-tokens", type=int, default=10)
    parser.add_argument("--eval-limit", type=int, default=500)
    parser.add_argument(
        "--eval-progress-interval",
        type=int,
        default=20,
        help="Print evaluation progress every N records. Set 0 to disable.",
    )
    parser.add_argument(
        "--full-context",
        action="store_true",
        help="Use MEMIT generated context templates. By default, only '{}' is used to save VRAM.",
    )
    parser.add_argument(
        "--real-cov",
        action="store_true",
        help="Use EasyEdit covariance statistics. By default, identity covariance is used to avoid dataset downloads.",
    )
    parser.add_argument(
        "--fp32",
        action="store_true",
        help="Load the model in fp32. By default, fp16 is used to save VRAM.",
    )
    return parser.parse_args()


def load_easyedit_runtime():
    global BaseEditor, MEMITHyperParams, nethook, torch
    if str(EASYEDIT_DIR) not in sys.path:
        sys.path.insert(0, str(EASYEDIT_DIR))

    edit_rome.patch_transformers_compat()
    edit_rome.patch_transformers_generation_compat()
    edit_rome.patch_huggingface_hub_compat()

    import torch as torch_module
    from EasyEdit.easyeditor import BaseEditor as BaseEditorClass
    from EasyEdit.easyeditor import MEMITHyperParams as MEMITHyperParamsClass
    from EasyEdit.easyeditor.util import nethook as nethook_module

    torch = torch_module
    BaseEditor = BaseEditorClass
    MEMITHyperParams = MEMITHyperParamsClass
    nethook = nethook_module
    edit_rome.torch = torch_module
    edit_rome.nethook = nethook_module


def load_records(path, limit):
    with path.open("r", encoding="utf-8") as f:
        records = json.load(f)

    if limit is not None:
        records = records[:limit]

    required = {"prompt", "subject", "target_new", "ground_truth"}
    for idx, record in enumerate(records, start=1):
        missing = required - set(record)
        if missing:
            raise ValueError(f"Record {idx} is missing keys: {sorted(missing)}")
        if record["subject"] not in record["prompt"]:
            raise ValueError(
                f"Record {idx} subject {record['subject']!r} is not in prompt "
                f"{record['prompt']!r}. MEMIT needs the subject span."
            )
    return records


def build_requests(records):
    return [
        {
            "case_id": idx,
            "prompt": record["prompt"],
            "subject": record["subject"],
            "target_new": record["target_new"],
            "ground_truth": record["ground_truth"],
            "portability": {},
            "locality": {},
        }
        for idx, record in enumerate(records)
    ]


def patch_memit_low_memory(*, use_identity_cov, use_full_context):
    memit_main = None
    for module_name, module in sys.modules.items():
        if module_name.endswith(".models.memit.memit_main"):
            memit_main = module
            break
    if memit_main is None:
        raise RuntimeError("MEMIT module was not loaded by EasyEdit runtime.")

    if not use_full_context:
        memit_main.CONTEXT_TEMPLATES_CACHE = [["{}"]]

    if not use_identity_cov:
        return

    def identity_cov(model, tok, layer_name, *args, hparams=None, **kwargs):
        weight = nethook.get_parameter(model, f"{layer_name}.weight")
        cov_dim = weight.shape[1]
        device = f"cuda:{hparams.device}"
        print(f"Using identity covariance for {layer_name}: {cov_dim} x {cov_dim}")
        return torch.eye(cov_dim, dtype=torch.float32, device=device)

    memit_main.get_cov = identity_cov


def new_editor(hparams_path, *, fp16):
    hparams = MEMITHyperParams.from_hparams(str(hparams_path))
    hparams.fp16 = fp16
    hparams.batch_size = 1
    return BaseEditor.from_hparams(hparams)


def apply_memit(editor, requests):
    start = time.perf_counter()
    edited_model, weights_copy = editor.apply_algo(
        editor.model,
        editor.tok,
        requests,
        editor.hparams,
        copy=False,
        return_orig_weights=True,
        keep_original_weight=False,
    )
    return edited_model, weights_copy, time.perf_counter() - start


def evaluate_outputs(model, tokenizer, records, max_new_tokens, progress_interval=20):
    outputs = []
    total = len(records)
    for idx, record in enumerate(records):
        if progress_interval and (idx == 0 or idx % progress_interval == 0):
            print(f"Evaluating {idx + 1}/{total} ...")
        response = edit_rome.generate_text(
            model, tokenizer, record["prompt"], max_new_tokens
        )
        item = {
            "case_id": idx,
            "prompt": record["prompt"],
            "target_new": record["target_new"],
            "ground_truth": record["ground_truth"],
            "output": response,
            "success": float(edit_rome.contains_answer(response, record["target_new"])),
        }
        if record.get("rephrase_prompt"):
            rephrase_output = edit_rome.generate_text(
                model, tokenizer, record["rephrase_prompt"], max_new_tokens
            )
            item["rephrase_prompt"] = record["rephrase_prompt"]
            item["rephrase_output"] = rephrase_output
            item["PS"] = float(edit_rome.contains_answer(rephrase_output, record["target_new"]))
        if record.get("locality_prompt") and record.get("locality_ground_truth"):
            locality_output = edit_rome.generate_text(
                model, tokenizer, record["locality_prompt"], max_new_tokens
            )
            item["locality_prompt"] = record["locality_prompt"]
            item["locality_ground_truth"] = record["locality_ground_truth"]
            item["locality_output"] = locality_output
            item["NS"] = float(
                edit_rome.contains_answer(locality_output, record["locality_ground_truth"])
            )
        outputs.append(item)
    if total and progress_interval:
        print(f"Evaluating {total}/{total} done.")
    return outputs


def mean(items, key):
    return sum(item[key] for item in items) / len(items) if items else 0.0


def main():
    args = parse_args()
    load_easyedit_runtime()
    records = load_records(args.data, args.limit)
    requests = build_requests(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    patch_memit_low_memory(
        use_identity_cov=not args.real_cov,
        use_full_context=args.full_context,
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    gc.collect()
    tracemalloc.start()
    total_start = time.perf_counter()

    editor = new_editor(args.hparams, fp16=not args.fp32)
    edited_model, weights_copy, edit_seconds = apply_memit(editor, requests)

    eval_records = records[: args.eval_limit] if args.eval_limit else []
    eval_start = time.perf_counter()
    evaluations = evaluate_outputs(
        edited_model,
        editor.tok,
        eval_records,
        args.max_new_tokens,
        args.eval_progress_interval,
    )
    eval_seconds = time.perf_counter() - eval_start

    current_memory, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    cuda_peak_mb = (
        torch.cuda.max_memory_allocated() / (1024 * 1024)
        if torch.cuda.is_available()
        else None
    )

    summary = {
        "method": "MEMIT",
        "framework": "EasyEdit",
        "data_path": str(args.data),
        "hparams_path": str(args.hparams),
        "num_edits": len(records),
        "num_evaluated": len(evaluations),
        "identity_covariance": not args.real_cov,
        "low_memory_context": not args.full_context,
        "elapsed_seconds": time.perf_counter() - total_start,
        "edit_seconds": edit_seconds,
        "eval_seconds": eval_seconds,
        "peak_python_memory_mb": peak_memory / (1024 * 1024),
        "peak_cuda_memory_mb": cuda_peak_mb,
        "ES": mean(evaluations, "success"),
        "PS": mean(evaluations, "PS"),
        "NS": mean(evaluations, "NS"),
        "records": evaluations,
    }

    with args.output.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nTask 3 MEMIT summary")
    print(f"Edits: {summary['num_edits']}")
    print(f"Evaluated: {summary['num_evaluated']}")
    print(f"ES: {summary['ES']:.3f}")
    if summary["PS"] is not None:
        print(f"PS: {summary['PS']:.3f}")
    if summary["NS"] is not None:
        print(f"NS: {summary['NS']:.3f}")
    print(f"Edit seconds: {summary['edit_seconds']:.2f}")
    print(f"Elapsed seconds: {summary['elapsed_seconds']:.2f}")
    if cuda_peak_mb is not None:
        print(f"Peak CUDA memory MB: {cuda_peak_mb:.1f}")
    print(f"Peak Python memory MB: {summary['peak_python_memory_mb']:.1f}")
    print(f"Saved: {args.output}")

    # Keep the edited model alive until all generation/evaluation is done, then restore.
    if weights_copy:
        edit_rome.restore_weights(editor.model, weights_copy)


if __name__ == "__main__":
    main()
