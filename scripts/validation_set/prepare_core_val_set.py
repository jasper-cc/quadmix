#!/usr/bin/env python3
"""
Prepare CORE benchmark-based validation set for QuaDMix proxy model.

Loads 22 tasks from the CORE eval bundle, extracts context + continuation
per task type, tokenizes with GPT-NeoX-20B. The loss_mask marks ONLY
continuation tokens (continuation-only loss), maximizing signal-to-noise
ratio for discriminating between data mixtures.

Text extraction per icl_task_type (context | continuation):
  - multiple_choice:    query + delimiter  |  choices[gold]
  - language_modeling:  context + delimiter |  continuation
  - schema:             context_options[gold] + delimiter |  continuation

Usage:
  python scripts/validation_set/prepare_core_val_set.py

Output: core_22tasks_tokenized.pt
  - token_ids:    LongTensor [num_docs, block_size]   padded
  - loss_mask:    BoolTensor  [num_docs, block_size]   True for continuation tokens only
  - task_labels:  list[str]   per-doc task label (e.g., "hellaswag_zeroshot")
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
NUM_SAMPLES_PER_TASK = 5000
MIN_CONTINUATION_TOKENS = 1


def parse_choice_from_query(query, choice_label):
    """Parse the full choice text from query when choices field only contains labels.
    
    Handles formats like:
        "Choices:\nA. option text\nB. another option\nAnswer:"
        "Choices:\nA) option text\nB) another option"
    """
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
        description="Create CORE benchmark validation set for QuaDMix"
    )
    parser.add_argument("--block-size", type=int, default=BLOCK_SIZE,
                        help=f"Block size for tokenization (default: {BLOCK_SIZE})")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR,
                        help=f"Output directory (default: {OUTPUT_DIR})")
    parser.add_argument("--eval-bundle", type=str, default=EVAL_BUNDLE_DIR,
                        help=f"Path to CORE eval bundle directory (default: {EVAL_BUNDLE_DIR})")
    parser.add_argument("--num-samples-per-task", type=int, default=NUM_SAMPLES_PER_TASK,
                        help=f"Max samples per task, 0=all available (default: {NUM_SAMPLES_PER_TASK})")
    parser.add_argument("--min-continuation-tokens", type=int, default=MIN_CONTINUATION_TOKENS,
                        help=f"Min continuation tokens to keep a sample (default: {MIN_CONTINUATION_TOKENS})")
    parser.add_argument("--seed", type=int, default=SEED,
                        help=f"Random seed (default: {SEED})")
    args = parser.parse_args()

    block_size = args.block_size
    output_dir = args.output_dir
    seed = args.seed
    os.makedirs(output_dir, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)

    # Tokenizer (same as openhermes ref script)
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Tokenizer: GPT-NeoX-20B, vocab={tokenizer.vocab_size}")

    # Load CORE task config
    core_yaml_path = os.path.join(args.eval_bundle, "core.yaml")
    with open(core_yaml_path, "r") as f:
        core_config = yaml.safe_load(f)

    tasks = core_config["icl_tasks"]
    print(f"Loaded {len(tasks)} tasks from core.yaml")

    min_cont_tok = args.min_continuation_tokens
    all_samples = []
    task_meta = []
    seen_uris = set()
    filtered_tasks = []

    for task in tasks:
        label = task["label"]
        task_type = task["icl_task_type"]
        dataset_uri = task["dataset_uri"]
        delimiter = task.get("continuation_delimiter", " ")

        if dataset_uri in seen_uris:
            print(f"\nSkipping task: {label} (duplicate of {dataset_uri})")
            continue
        seen_uris.add(dataset_uri)

        print(f"\nLoading task: {label} (type={task_type}, uri={dataset_uri})")
        items = load_task_items(args.eval_bundle, dataset_uri)
        print(f"  Total items: {len(items)}")

        pairs = []
        for item in items:
            ctx, cont = extract_context_continuation(task_type, item, delimiter)
            if cont.strip():
                cont_tok = tokenizer.encode(cont, add_special_tokens=False)
                if len(cont_tok) >= min_cont_tok:
                    pairs.append((ctx, cont, len(cont_tok)))

        print(f"  Valid pairs (continuation >= {min_cont_tok} tokens): {len(pairs)}")

        if len(pairs) == 0:
            print(f"  FILTERED OUT: no samples with >= {min_cont_tok} continuation tokens")
            filtered_tasks.append(label)
            continue

        if args.num_samples_per_task > 0:
            target = min(args.num_samples_per_task, len(pairs))
            sampled = random.sample(pairs, target)
            capped = len(pairs) > target
        else:
            sampled = pairs
            capped = False

        avg_cont_tok = sum(ct for _, _, ct in sampled) / len(sampled)
        print(f"  Sampled: {len(sampled)}" + (f" (capped from {len(pairs)})" if capped else f" (all {len(pairs)} used)") + f", avg cont tokens: {avg_cont_tok:.1f}")

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
            "avg_continuation_tokens": round(avg_cont_tok, 1),
        })

    if filtered_tasks:
        print(f"\n{'='*60}")
        print(f"Filtered out {len(filtered_tasks)} tasks (continuation < {min_cont_tok} tokens):")
        for t in filtered_tasks:
            print(f"  - {t}")

    print(f"\n{'='*60}")
    print(f"Total samples across all tasks: {len(all_samples)}")

    print(f"\nTokenizing (block_size={block_size}, continuation-only loss)...")
    all_ids = []
    all_masks = []
    task_labels = []

    for idx, (context, continuation, task_label) in enumerate(all_samples):
        ctx_ids = tokenizer.encode(context, add_special_tokens=False) if context else []
        cont_ids = tokenizer.encode(continuation, add_special_tokens=False)
        full_ids = ctx_ids + cont_ids

        ctx_len = len(ctx_ids)
        cont_len = len(cont_ids)
        seq_len = min(len(full_ids), block_size)

        ids_tensor = torch.LongTensor(full_ids)
        if len(ids_tensor) > block_size:
            ids_tensor = ids_tensor[:block_size]
            cont_end = block_size
        else:
            cont_end = seq_len
            pad_len = block_size - len(ids_tensor)
            ids_tensor = torch.cat([ids_tensor, torch.zeros(pad_len, dtype=torch.long)])

        cont_start = min(ctx_len, block_size)
        loss_mask = torch.zeros(block_size, dtype=torch.bool)
        if cont_start < cont_end:
            loss_mask[cont_start:cont_end] = True

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
    print(f"  Continuation tokens (loss_mask=True): {masked_tokens:,} ({masked_tokens/non_padding*100:.1f}% of non-padding)")
    print(f"  Context tokens (masked out): {non_padding - masked_tokens:,}")

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
            "loss_strategy": "continuation_only",
            "min_continuation_tokens": min_cont_tok,
            "source": f"CORE-benchmark-{len(task_meta)}tasks",
            "eval_bundle": args.eval_bundle,
            "num_samples_per_task": args.num_samples_per_task,
            "seed": seed,
            "description": (
                f"CORE benchmark {len(task_meta)}-task validation set (deduplicated); "
                "loss_mask=True on continuation tokens only (continuation-only loss), "
                "maximizing signal-to-noise for discriminating data mixtures"
            ),
            "tasks": task_meta,
        }
    }

    save_path = os.path.join(output_dir, "core_22tasks_tokenized.pt")
    torch.save(output, save_path)
    file_size = os.path.getsize(save_path) / 1024**2
    print(f"\n  Saved: {save_path} ({file_size:.0f} MB)")
    print("Done!")


if __name__ == "__main__":
    main()
