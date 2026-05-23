import argparse
import gc
import json
import sys
import time
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EASYEDIT_DIR = ROOT / "EasyEdit"


DEFAULT_DATA_PATH = ROOT / "data" / "custom_data.json"
DEFAULT_HPARAMS_PATH = ROOT / "hparams" / "ROME" / "qwen2.5-0.5b.yaml"
DEFAULT_OUTPUT_PATH = ROOT / "results" / "rome_results.json"
BaseEditor = None
ROMEHyperParams = None
nethook = None
torch = None


def load_easyedit_runtime():
    global BaseEditor, ROMEHyperParams, nethook, torch
    if str(EASYEDIT_DIR) not in sys.path:
        sys.path.insert(0, str(EASYEDIT_DIR))

    try:
        import torch as torch_module
        patch_transformers_compat()
        patch_transformers_generation_compat()
        patch_huggingface_hub_compat()
        from EasyEdit.easyeditor import BaseEditor as BaseEditorClass
        from EasyEdit.easyeditor import ROMEHyperParams as ROMEHyperParamsClass
        from EasyEdit.easyeditor.util import nethook as nethook_module
    except ModuleNotFoundError as exc:
        missing = exc.name or "a required package"
        raise SystemExit(
            f"Missing dependency: {missing}. Install the EasyEdit runtime dependencies "
            f"first, for example: pip install torch transformers accelerate pyyaml"
        ) from exc

    torch = torch_module
    BaseEditor = BaseEditorClass
    ROMEHyperParams = ROMEHyperParamsClass
    nethook = nethook_module


def patch_huggingface_hub_compat():
    """Provide HfFolder for packages that still expect the old hub API."""
    try:
        import huggingface_hub
    except ModuleNotFoundError:
        raise

    if hasattr(huggingface_hub, "HfFolder"):
        return

    class HfFolder:
        @classmethod
        def get_token(cls):
            get_token = getattr(huggingface_hub, "get_token", None)
            return get_token() if get_token is not None else None

        @classmethod
        def save_token(cls, token):
            login = getattr(huggingface_hub, "login", None)
            if login is not None:
                login(token=token, add_to_git_credential=False)

        @classmethod
        def delete_token(cls):
            logout = getattr(huggingface_hub, "logout", None)
            if logout is not None:
                logout()

    huggingface_hub.HfFolder = HfFolder


def patch_transformers_generation_compat():
    """Provide generation output aliases expected by older EasyEdit modules."""
    try:
        import transformers.generation.utils as generation_utils
    except ModuleNotFoundError:
        raise

    fallback_pairs = {
        "GreedySearchOutput": "GenerateNonBeamOutput",
        "GreedySearchEncoderDecoderOutput": "GenerateEncoderDecoderOutput",
        "GreedySearchDecoderOnlyOutput": "GenerateDecoderOnlyOutput",
        "SampleDecoderOnlyOutput": "GenerateDecoderOnlyOutput",
        "SampleEncoderDecoderOutput": "GenerateEncoderDecoderOutput",
        "BeamSearchDecoderOnlyOutput": "GenerateBeamDecoderOnlyOutput",
        "BeamSearchEncoderDecoderOutput": "GenerateBeamEncoderDecoderOutput",
    }

    for old_name, new_name in fallback_pairs.items():
        if hasattr(generation_utils, old_name):
            continue
        if hasattr(generation_utils, new_name):
            setattr(generation_utils, old_name, getattr(generation_utils, new_name))
        elif hasattr(generation_utils, "GenerateOutput"):
            setattr(generation_utils, old_name, getattr(generation_utils, "GenerateOutput"))

    if "transformers.generation.beam_search" not in sys.modules:
        beam_search_module = types.ModuleType("transformers.generation.beam_search")

        class BeamScorer:
            pass

        class BeamSearchScorer(BeamScorer):
            pass

        for name in ("BeamScorer", "BeamSearchScorer"):
            if hasattr(generation_utils, name):
                setattr(beam_search_module, name, getattr(generation_utils, name))
            else:
                setattr(beam_search_module, name, locals()[name])
        sys.modules["transformers.generation.beam_search"] = beam_search_module


def patch_transformers_compat():
    """Patch symbols moved in newer transformers versions.

    EasyEdit imports its BLIP2/Qformer modules from the package root even when
    this script only uses text-only ROME. Older EasyEdit code expects these
    helpers in transformers.pytorch_utils; recent transformers versions removed
    or moved them, so provide compatible fallbacks before EasyEdit is imported.
    """
    try:
        import transformers.modeling_utils as modeling_utils
        import transformers.pytorch_utils as pytorch_utils
        import torch as torch_module
        from torch import nn
    except ModuleNotFoundError:
        raise

    def fallback_find_pruneable_heads_and_indices(
        heads, n_heads, head_size, already_pruned_heads
    ):
        mask = torch_module.ones(n_heads, head_size)
        heads = set(heads) - already_pruned_heads
        for head in heads:
            head = head - sum(1 if pruned_head < head else 0 for pruned_head in already_pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch_module.arange(len(mask))[mask].long()
        return heads, index

    def fallback_prune_linear_layer(layer, index, dim=0):
        index = index.to(layer.weight.device)
        weight = layer.weight.index_select(dim, index).clone().detach()

        if layer.bias is not None:
            if dim == 1:
                bias = layer.bias.clone().detach()
            else:
                bias = layer.bias[index].clone().detach()
        else:
            bias = None

        new_size = list(layer.weight.size())
        new_size[dim] = len(index)
        new_layer = nn.Linear(
            new_size[1],
            new_size[0],
            bias=layer.bias is not None,
        ).to(device=layer.weight.device, dtype=layer.weight.dtype)

        new_layer.weight.requires_grad = False
        new_layer.weight.copy_(weight.contiguous())
        new_layer.weight.requires_grad = True

        if bias is not None:
            new_layer.bias.requires_grad = False
            new_layer.bias.copy_(bias.contiguous())
            new_layer.bias.requires_grad = True

        return new_layer

    compat_fallbacks = {
        "find_pruneable_heads_and_indices": fallback_find_pruneable_heads_and_indices,
        "prune_linear_layer": fallback_prune_linear_layer,
    }
    for name, fallback in compat_fallbacks.items():
        if hasattr(pytorch_utils, name):
            continue
        replacement = getattr(modeling_utils, name, fallback)
        setattr(pytorch_utils, name, replacement)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Task 2: single fact editing with EasyEdit ROME."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--hparams", type=Path, default=DEFAULT_HPARAMS_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument(
        "--full-context",
        action="store_true",
        help="Use the context templates from the ROME hparams file. By default, only '{}' is used to save VRAM.",
    )
    parser.add_argument(
        "--fp32",
        action="store_true",
        help="Load the model in fp32. By default, fp16 is used to save VRAM.",
    )
    parser.add_argument(
        "--reload-each-edit",
        action="store_true",
        help="Reload the base model before every fact instead of restoring edited weights.",
    )
    return parser.parse_args()


def load_records(path):
    with path.open("r", encoding="utf-8") as f:
        records = json.load(f)

    required_keys = {
        "prompt",
        "subject",
        "target_new",
        "ground_truth",
        "rephrase_prompt",
        "locality_prompt",
        "locality_ground_truth",
    }
    for idx, record in enumerate(records, start=1):
        missing = required_keys - set(record)
        if missing:
            raise ValueError(f"Record {idx} is missing keys: {sorted(missing)}")
        if record["subject"] not in record["prompt"]:
            raise ValueError(
                f"Record {idx} subject {record['subject']!r} is not in prompt "
                f"{record['prompt']!r}. ROME needs the subject span."
            )
    return records


def build_request(record):
    return {
        "prompt": record["prompt"],
        "subject": record["subject"],
        "target_new": record["target_new"],
        "ground_truth": record["ground_truth"],
        "rephrase_prompt": record["rephrase_prompt"],
        "locality": {
            "neighborhood": {
                "prompt": record["locality_prompt"],
                "ground_truth": record["locality_ground_truth"],
            }
        },
        "portability": {},
    }


def generate_text(model, tokenizer, prompt, max_new_tokens):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if text.startswith(prompt):
        text = text[len(prompt) :]
    return text.strip()


def contains_answer(response, answer):
    return answer.casefold() in response.casefold()


def restore_weights(model, weights_copy):
    with torch.no_grad():
        for weight_name, old_weight in weights_copy.items():
            param = nethook.get_parameter(model, weight_name)
            param[...] = old_weight.to(param.device)


def new_editor(hparams_path, *, low_memory=True, fp16=True):
    hparams = ROMEHyperParams.from_hparams(str(hparams_path))
    hparams.fp16 = fp16
    if low_memory:
        hparams.context_template_length_params = []
    return BaseEditor.from_hparams(hparams)


def edit_one(editor, record, max_new_tokens):
    request = build_request(record)
    start = time.perf_counter()
    edited_model, weights_copy = editor.apply_algo(
        editor.model,
        editor.tok,
        [request],
        editor.hparams,
        copy=False,
        return_orig_weights=True,
        keep_original_weight=False,
    )
    edit_seconds = time.perf_counter() - start

    direct_output = generate_text(
        edited_model, editor.tok, record["prompt"], max_new_tokens
    )
    rephrase_output = generate_text(
        edited_model, editor.tok, record["rephrase_prompt"], max_new_tokens
    )
    locality_output = generate_text(
        edited_model, editor.tok, record["locality_prompt"], max_new_tokens
    )

    restore_weights(editor.model, weights_copy)

    return {
        "prompt": record["prompt"],
        "target_new": record["target_new"],
        "ground_truth": record["ground_truth"],
        "rephrase_prompt": record["rephrase_prompt"],
        "locality_prompt": record["locality_prompt"],
        "locality_ground_truth": record["locality_ground_truth"],
        "direct_output": direct_output,
        "rephrase_output": rephrase_output,
        "locality_output": locality_output,
        "ES": float(contains_answer(direct_output, record["target_new"])),
        "PS": float(contains_answer(rephrase_output, record["target_new"])),
        "NS": float(contains_answer(locality_output, record["locality_ground_truth"])),
        "edit_seconds": edit_seconds,
    }


def mean(items, key):
    return sum(item[key] for item in items) / len(items) if items else 0.0


def main():
    args = parse_args()
    load_easyedit_runtime()
    records = load_records(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    total_start = time.perf_counter()
    results = []
    editor = None

    if not args.reload_each_edit:
        editor = new_editor(
            args.hparams,
            low_memory=not args.full_context,
            fp16=not args.fp32,
        )

    for idx, record in enumerate(records, start=1):
        print(f"\n[{idx}/{len(records)}] Editing: {record['prompt']} -> {record['target_new']}")
        if args.reload_each_edit:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            editor = new_editor(
                args.hparams,
                low_memory=not args.full_context,
                fp16=not args.fp32,
            )

        result = edit_one(editor, record, args.max_new_tokens)
        results.append(result)

        print(f"Direct    : {result['direct_output']}")
        print(f"Rephrase  : {result['rephrase_output']}")
        print(f"Locality  : {result['locality_output']}")
        print(f"ES/PS/NS  : {result['ES']:.0f}/{result['PS']:.0f}/{result['NS']:.0f}")

        if args.reload_each_edit:
            del editor
            editor = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    summary = {
        "method": "ROME",
        "framework": "EasyEdit",
        "data_path": str(args.data),
        "hparams_path": str(args.hparams),
        "num_edits": len(results),
        "reset_strategy": "reload_model" if args.reload_each_edit else "restore_weights",
        "elapsed_seconds": time.perf_counter() - total_start,
        "ES": mean(results, "ES"),
        "PS": mean(results, "PS"),
        "NS": mean(results, "NS"),
        "records": results,
    }

    with args.output.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nTask 2 ROME summary")
    print(f"Edits: {summary['num_edits']}")
    print(f"ES: {summary['ES']:.3f}")
    print(f"PS: {summary['PS']:.3f}")
    print(f"NS: {summary['NS']:.3f}")
    print(f"Elapsed seconds: {summary['elapsed_seconds']:.2f}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
