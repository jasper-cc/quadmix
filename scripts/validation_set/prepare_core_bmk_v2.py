#!/usr/bin/env python3
"""
Prepare CORE-BMK v2 validation set for QuaDMix proxy model.

Based on analysis of the 21 CORE tasks, we found that:
1. The QuaDMix paper's BMK baseline uses only tasks with Answer% > 10%
2. Tasks with Answer% <= 10% have context that dominates the loss signal
3. A 1M proxy model cannot meaningfully learn from high-context tasks (ppl~4800)

This script creates a v2 validation set using only the 10 "BMK-like" tasks:
  - hellaswag_zeroshot (37.5%)
  - piqa (67.6%)
  - bigbench_repeat_copy_logic (44.5%)
  - copa (39.1%)
  - openbook_qa (25.0%)
  - winogrande (22.5%)
  - winograd (21.8%)
  - arc_challenge (17.4%)
  - arc_easy (15.4%)
  - bigbench_qa_wikidata (14.5%)

Key differences from v1 (core_22tasks):
  - Only 10 tasks (vs 21)
  - Full-sequence loss (vs continuation-only)
  - Average Answer% = 30.5% (vs 5.7%)

Usage:
  python scripts/validation_set/prepare_core_bmk_v2.py

Output: core_bmk_10tasks_v2_tokenized.pt
  - token_ids:    LongTensor [num_docs, block_size]   padded
  - loss_mask:    BoolTensor  [num_docs, block_size]   True for all non-padding tokens
  - task_labels:  list[str]   per-doc task label
  - metadata:     dict        source info
"""

import os
import re
import random
import argparse
import json
import yaml

import numpy as np
import torch
from transformers import AutoTokenizer


EVAL_BUNDLE_DIR = os.environ.get(
    "EVAL_BUNDLE_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'eval_bundle'),
)
OUTPUT_DIR = "data"
SEED = 42
BLOCK_SIZE = 2048
NUM_SAMPLES_PER_TASK = 20000

BMK_TASKS = [
    "hellaswag_zeroshot",
    "piqa",
    "bigbench_repeat_copy_logic",
    "copa",
    "openbook_qa",
    "winogrande",
    "winograd",
    "arc_challenge",
    "arc_easy",
    "bigbench_qa_wikidata",
]


def parse_choice_from_query(query, choice_label):
    """Parse the full choice text from query when choices field only contains labels."""
    pattern = rf'{re.escape(choice_label)}[.)]\s*(.+?)(?=\n[A-Z][.)]|\nAnswer:|$)'
    match = re.search(pattern, query, re.DOTALL)
    if match:
        return match.group(1).strip()
    return choice_label


def load_task_items(eval_bundle_dir, dataset_uri):
    """Load JSONL items for a given dataset_uri relative to eval_bundle/eval_data/."""
    filepath = os.path.join(eval_bundle_dir, "eval_data", dataset_uri)
    items = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def extract_context_continuation(task_type, item, delimiter=" "):
    """Extract (context, continuation) from a CORE item."""
    if task_type == "multiple_choice":
        query = item.get("query", "").rstrip()
        choices = item.get("choices", [])
        gold = item.get("gold", 0)
        if not query or not choices:
            return "", ""
        if 0 <= gold < len(choices):
            choice = choices[gold]
            if len(choice) <= 2 and choice.isalpha():
                choice = parse_choice_from_query(query, choice)
            return query + delimiter, choice
        return "", ""
    elif task_type == "language_modeling":
        context = item.get("context", "").strip()
        continuation = item.get("continuation", "")
        if not continuation.strip():
            return "", ""
        if context:
            return context + delimiter, continuation
        return "", continuation
    elif task_type == "schema":
        context_options = item.get("context_options", [])
        gold = item.get("gold", 0)
        continuation = item.get("continuation", "")
        if context_options and 0 <= gold < len(context_options):
            return context_options[gold] + delimiter, continuation
        return "", ""
    else:
        return "", ""


def main():
    parser = argparse.ArgumentParser(
        description="Create CORE-BMK v2 validation set for QuaDMix"
    )
    parser.add_argument("--block-size", type=int, default=BLOCK_SIZE,
                        help=f"Block size for tokenization (default: {BLOCK_SIZE})")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR,
                        help=f"Output directory (default: {OUTPUT_DIR})")
    parser.add_argument("--eval-bundle", type=str, default=EVAL_BUNDLE_DIR,
                        help=f"Path to CORE eval bundle directory (default: {EVAL_BUNDLE_DIR})")
    parser.add_argument("--num-samples-per-task", type=int, default=NUM_SAMPLES_PER_TASK,
                        help=f"Max samples per task, 0=all available (default: {NUM_SAMPLES_PER_TASK})")
    parser.add_argument("--seed", type=int, default=SEED,
                        help=f"Random seed (default: {SEED})")
    args = parser.parse_args()

    block_size = args.block_size
    output_dir = args.output_dir
    seed = args.seed
    os.makedirs(output_dir, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Tokenizer: GPT-NeoX-20B, vocab={tokenizer.vocab_size}")

    core_yaml_path = os.path.join(args.eval_bundle, "core.yaml")
    with open(core_yaml_path, "r") as f:
        core_config = yaml.safe_load(f)

    all_tasks = core_config["icl_tasks"]
    print(f"Loaded {len(all_tasks)} tasks from core.yaml")

    task_map = {}
    for task in all_tasks:
        label = task["label"]
        if label not in task_map:
            task_map[label] = task

    print(f"\nFiltering to {len(BMK_TASKS)} BMK-like tasks (Answer% > 10%)...")
    for t in BMK_TASKS:
        if t not in task_map:
            print(f"  WARNING: {t} not found in core.yaml")

    all_samples = []
    task_meta = []

    for label in BMK_TASKS:
        if label not in task_map:
            continue

        task = task_map[label]
        task_type = task["icl_task_type"]
        dataset_uri = task["dataset_uri"]
        delimiter = task.get("continuation_delimiter", " ")

        print(f"\nLoading task: {label} (type={task_type}, uri={dataset_uri})")
        items = load_task_items(args.eval_bundle, dataset_uri)
        print(f"  Total items: {len(items)}")

        pairs = []
        for item in items:
            ctx, cont = extract_context_continuation(task_type, item, delimiter)
            if cont.strip():
                full_text = ctx + cont
                pairs.append((ctx, cont, full_text))

        print(f"  Valid pairs: {len(pairs)}")

        if len(pairs) == 0:
            print(f"  SKIPPED: no valid pairs")
            continue

        if args.num_samples_per_task > 0:
            target = min(args.num_samples_per_task, len(pairs))
            sampled = random.sample(pairs, target)
            capped = len(pairs) > target
        else:
            sampled = pairs
            capped = False

        avg_ctx_chars = sum(len(ctx) for ctx, _, _ in sampled) / len(sampled)
        avg_ans_chars = sum(len(cont) for _, cont, _ in sampled) / len(sampled)
        avg_full_chars = sum(len(full) for _, _, full in sampled) / len(sampled)
        ans_ratio = avg_ans_chars / avg_full_chars * 100 if avg_full_chars > 0 else 0

        print(f"  Sampled: {len(sampled)}" + (f" (capped from {len(pairs)})" if capped else f" (all used)"))
        print(f"  Avg chars: ctx={avg_ctx_chars:.0f}, ans={avg_ans_chars:.0f}, full={avg_full_chars:.0f}, ans%={ans_ratio:.1f}%")

        all_samples.extend([(ctx, cont, label) for ctx, cont, _ in sampled])
        task_meta.append({
            "label": label,
            "task_type": task_type,
            "dataset_uri": dataset_uri,
            "continuation_delimiter": delimiter,
            "items_loaded": len(items),
            "valid_pairs": len(pairs),
            "sampled": len(sampled),
            "capped": capped,
            "avg_context_chars": round(avg_ctx_chars, 1),
            "avg_answer_chars": round(avg_ans_chars, 1),
            "avg_answer_ratio": round(ans_ratio, 1),
        })

    print(f"\n{'='*60}")
    print(f"Total samples across {len(task_meta)} BMK-like tasks: {len(all_samples)}")

    print(f"\nTokenizing (block_size={block_size}, full-sequence loss)...")
    all_ids = []
    all_masks = []
    task_labels = []

    for idx, (context, continuation, task_label) in enumerate(all_samples):
        full_text = context + continuation
        full_ids = tokenizer.encode(full_text, add_special_tokens=False)

        seq_len = min(len(full_ids), block_size)

        ids_tensor = torch.LongTensor(full_ids)
        if len(ids_tensor) > block_size:
            ids_tensor = ids_tensor[:block_size]
            pad_len = 0
        else:
            pad_len = block_size - len(ids_tensor)
            ids_tensor = torch.cat([ids_tensor, torch.zeros(pad_len, dtype=torch.long)])

        loss_mask = torch.zeros(block_size, dtype=torch.bool)
        loss_mask[:seq_len] = True

        all_ids.append(ids_tensor)
        all_masks.append(loss_mask)
        task_labels.append(task_label)

        if (idx + 1) % 5000 == 0:
            print(f"  {idx+1}/{len(all_samples)}")

    token_ids = torch.stack(all_ids)
    loss_mask = torch.stack(all_masks)

    total_tokens = token_ids.numel()
    non_padding = (token_ids != tokenizer.pad_token_id).sum().item()
    masked_tokens = loss_mask.sum().item()
    print(f"\n  Tokenized: {token_ids.shape}")
    print(f"  Non-padding tokens: {non_padding:,} / {total_tokens:,} ({non_padding/total_tokens*100:.1f}%)")
    print(f"  Loss tokens (loss_mask=True): {masked_tokens:,} ({masked_tokens/non_padding*100:.1f}% of non-padding)")

    output = {
        "token_ids": token_ids,
        "loss_mask": loss_mask,
        "task_labels": task_labels,
        "metadata": {
            "num_docs": len(token_ids),
            "block_size": block_size,
            "tokenizer": "gpt-neox-20b",
            "tokenizer_vocab": tokenizer.vocab_size,
            "model_vocab": 50432,
            "loss_strategy": "full_sequence",
            "source": f"CORE-BMK-{len(task_meta)}tasks-v2",
            "eval_bundle": args.eval_bundle,
            "num_samples_per_task": args.num_samples_per_task,
            "seed": seed,
            "description": (
                f"CORE-BMK v2: {len(task_meta)} BMK-like tasks (Answer% > 10%), "
                "full-sequence loss for better proxy model signal. "
                "Based on analysis showing QuaDMix paper BMK uses only high-answer-ratio tasks."
            ),
            "tasks": task_meta,
        }
    }

    save_path = os.path.join(output_dir, "core_bmk_10tasks_v2_tokenized.pt")
    torch.save(output, save_path)
    file_size = os.path.getsize(save_path) / 1024**2
    print(f"\n  Saved: {save_path} ({file_size:.0f} MB)")
    print("Done!")


if __name__ == "__main__":
    main()
