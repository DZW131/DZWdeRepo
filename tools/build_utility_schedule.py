#!/usr/bin/env python3
"""Build the single worker-independent schedule shared by all TCRD branches."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from tool.GenDataset import Stage1_TrainDataset
from tools.tcrd_common import dataset_fingerprint, sha256_file, write_json


def build_schedule(train_root: str, output: str, seed=42, epochs=5, batch_size=20):
    dataset = Stage1_TrainDataset(train_root, dataset="bcss", img_size=224)
    if len(dataset) != 23422:
        raise AssertionError(f"Expected 23,422 BCSS samples, found {len(dataset)}")
    steps = len(dataset) // batch_size
    generator = np.random.default_rng(seed)
    indices = np.empty((epochs, steps, batch_size), dtype=np.int32)
    augmentation_seeds = np.empty((epochs, steps, batch_size), dtype=np.int64)
    model_seeds = np.empty((epochs, steps), dtype=np.int64)
    for epoch in range(epochs):
        permutation = generator.permutation(len(dataset))[:steps * batch_size]
        indices[epoch] = permutation.reshape(steps, batch_size)
        augmentation_seeds[epoch] = generator.integers(
            0, np.iinfo(np.int64).max, size=(steps, batch_size), dtype=np.int64
        )
        model_seeds[epoch] = generator.integers(
            0, np.iinfo(np.int64).max, size=steps, dtype=np.int64
        )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    np.savez_compressed(
        output,
        indices=indices,
        augmentation_seeds=augmentation_seeds,
        model_seeds=model_seeds,
    )
    metadata = {
        "seed": seed, "epochs": epochs, "batch_size": batch_size,
        "dataset_samples": len(dataset), "steps_per_epoch": steps,
        "scheduled_samples_per_epoch": steps * batch_size,
        "dropped_samples_per_epoch": len(dataset) - steps * batch_size,
        "dataset_order_sha256": dataset_fingerprint(dataset, train_root),
        "schedule_sha256": sha256_file(output),
        "worker_independent_augmentation": True,
        "common_model_seed_per_step": True,
    }
    write_json(output.with_suffix(".json"), metadata)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(build_schedule(args.train_root, args.output))


if __name__ == "__main__":
    main()
