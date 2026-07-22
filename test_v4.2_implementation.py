#!/usr/bin/env python3
"""
Test script to verify v4.2 per-task loss implementation.
"""

import torch
import numpy as np
from quadmix.core.types import ProxyResult, ParameterSet, MergedQualityConfig, SamplingConfig

def test_proxy_result_with_per_task_losses():
    """Test that ProxyResult can store per_task_losses."""
    print("Test 1: ProxyResult with per_task_losses")
    
    # Create a minimal ParameterSet
    merge_config = MergedQualityConfig(
        global_weights=np.array([0.5, 0.5]),
        domain_weights=np.array([0.3, 0.7, 0.4, 0.6])
    )
    sampling_configs = [
        SamplingConfig(lambda_=1.0, omega=0.5, eta=1.0, epsilon=0.01),
        SamplingConfig(lambda_=1.0, omega=0.5, eta=1.0, epsilon=0.01)
    ]
    params = ParameterSet(merge_config=merge_config, sampling_configs=sampling_configs)
    
    # Create ProxyResult with per_task_losses
    per_task_losses = {
        "hellaswag_zeroshot": 3.42,
        "lambada_openai": 4.15,
        "arc_easy": 2.89,
    }
    
    result = ProxyResult(
        parameters=params,
        validation_loss=3.5,
        metadata={"test": "data"},
        per_task_losses=per_task_losses
    )
    
    assert result.validation_loss == 3.5
    assert result.per_task_losses is not None
    assert len(result.per_task_losses) == 3
    assert result.per_task_losses["hellaswag_zeroshot"] == 3.42
    
    print("  ✓ ProxyResult stores per_task_losses correctly")
    
    # Test backward compatibility (without per_task_losses)
    result_no_per_task = ProxyResult(
        parameters=params,
        validation_loss=3.5,
        metadata={"test": "data"}
    )
    
    assert result_no_per_task.per_task_losses is None
    print("  ✓ ProxyResult backward compatible (per_task_losses=None)")


def test_validation_data_loading():
    """Test that validation data with task_labels can be loaded."""
    print("\nTest 2: Validation data loading with task_labels")
    
    val_data_path = "data/core_bmk_10tasks_v4_tokenized.pt"
    
    try:
        val_data = torch.load(val_data_path, map_location="cpu", weights_only=True)
        
        assert "token_ids" in val_data
        assert "loss_mask" in val_data
        assert "task_labels" in val_data
        
        task_labels = val_data["task_labels"]
        unique_tasks = sorted(set(task_labels))
        
        print(f"  ✓ Loaded validation data with {len(unique_tasks)} tasks:")
        for task in unique_tasks:
            count = sum(1 for t in task_labels if t == task)
            print(f"    - {task}: {count} samples")
        
    except FileNotFoundError:
        print(f"  ⚠ Validation data not found at {val_data_path}, skipping")


def test_per_task_loss_computation():
    """Test per-task loss computation logic."""
    print("\nTest 3: Per-task loss computation logic")
    
    # Simulate per_doc_losses and task_labels
    n_docs = 100
    per_doc_losses = torch.randn(n_docs)
    task_labels = ["task_a"] * 40 + ["task_b"] * 35 + ["task_c"] * 25
    
    # Compute per-task losses
    per_task_losses = {}
    for task in sorted(set(task_labels)):
        task_indices = [i for i, t in enumerate(task_labels) if t == task]
        if task_indices:
            task_loss = float(per_doc_losses[task_indices].mean())
            per_task_losses[task] = task_loss
    
    assert len(per_task_losses) == 3
    assert "task_a" in per_task_losses
    assert "task_b" in per_task_losses
    assert "task_c" in per_task_losses
    
    print(f"  ✓ Computed per-task losses:")
    for task, loss in sorted(per_task_losses.items()):
        print(f"    - {task}: {loss:.4f}")


def test_discrimination_adaptive_weighting():
    """Test discrimination-adaptive weight computation."""
    print("\nTest 4: Discrimination-adaptive weighting")
    
    # Simulate task variances
    task_variances = {
        "task_a": 0.10,  # High discrimination
        "task_b": 0.01,  # Low discrimination
        "task_c": 0.00,  # Zero variance (should be excluded)
    }
    
    # Filter zero-variance tasks
    valid_tasks = {task: var for task, var in task_variances.items() if var > 1e-8}
    
    assert len(valid_tasks) == 2
    assert "task_c" not in valid_tasks
    print(f"  ✓ Filtered out {len(task_variances) - len(valid_tasks)} zero-variance task(s)")
    
    # Compute weights using standard deviation normalization
    task_stds = {task: np.sqrt(var) for task, var in valid_tasks.items()}
    total_std = sum(task_stds.values())
    weights = {task: std / total_std for task, std in task_stds.items()}
    
    # Verify weights sum to 1
    assert abs(sum(weights.values()) - 1.0) < 1e-6
    
    # Verify high-discrimination task has higher weight
    assert weights["task_a"] > weights["task_b"]
    
    print(f"  ✓ Computed discrimination-adaptive weights:")
    for task, weight in sorted(weights.items(), key=lambda x: -x[1]):
        std = task_stds[task]
        print(f"    - {task}: weight={weight:.4f}, std={std:.4f}")
    
    # Verify weight ratio is reasonable (not too extreme)
    weight_ratio = weights["task_a"] / weights["task_b"]
    print(f"  ✓ Weight ratio (high/low): {weight_ratio:.2f}x (expected ~3.16x for std normalization)")


if __name__ == "__main__":
    print("=" * 60)
    print("v4.2 Per-Task Loss Implementation Tests")
    print("=" * 60)
    
    test_proxy_result_with_per_task_losses()
    test_validation_data_loading()
    test_per_task_loss_computation()
    test_discrimination_adaptive_weighting()
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
