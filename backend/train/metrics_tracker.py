"""
Metrics tracking utilities for training runs.
Logs per-epoch metrics (accuracy, loss, f1, etc.) plus contextual factors that
influence accuracy: config, data distribution, hyperparameters, device info,
random seeds, and runtime metadata. Writes JSONL for structured logs and CSV for quick analysis.
"""

import os
import json
import csv
import time
from pathlib import Path
from dataclasses import asdict
from typing import Dict, Any, Optional, List

import torch


class MetricsTracker:
    def __init__(
        self,
        output_dir: str,
        run_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.start_time_s = time.time()
        self.run_id = run_id or str(int(self.start_time_s))
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.jsonl_path = self.output_dir / f"metrics_{self.run_id}.jsonl"
        self.csv_path = self.output_dir / f"metrics_{self.run_id}.csv"
        self.context_snapshot = context or {}

        # Prepare CSV header lazily on first write
        self._csv_initialized = False

    @staticmethod
    def _device_info() -> Dict[str, Any]:
        cuda_available = torch.cuda.is_available()
        return {
            "cuda_available": cuda_available,
            "cuda_device_count": torch.cuda.device_count() if cuda_available else 0,
            "device": "cuda" if cuda_available else "cpu",
            "cuda_name_0": torch.cuda.get_device_name(0) if cuda_available and torch.cuda.device_count() > 0 else None,
        }

    def update_context(self, extra: Dict[str, Any]) -> None:
        self.context_snapshot.update(extra or {})

    def log_epoch(
        self,
        epoch_index: int,
        split: str,
        metrics: Dict[str, Any],
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log metrics for an epoch.
        - epoch_index: 0-based epoch
        - split: "train" | "val" | "test"
        - metrics: e.g., {"accuracy": 0.93, "loss": 0.21, "precision": 0.9, ...}
        - extra: any additional info (e.g., class distribution, lr, batch_size)
        """
        record = {
            "ts": time.time(),
            "run_id": self.run_id,
            "epoch": epoch_index,
            "split": split,
            "metrics": metrics,
            "context": {
                **self.context_snapshot,
                **self._device_info(),
            },
        }
        if extra:
            record["extra"] = extra

        # Always append JSONL
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        # Also write/update CSV with a flattened view of core fields
        flat = {
            "ts": record["ts"],
            "run_id": record["run_id"],
            "epoch": record["epoch"],
            "split": record["split"],
        }
        # Flatten metrics
        for k, v in (metrics or {}).items():
            flat[f"metric.{k}"] = v
        # Flatten selected context keys (avoid exploding CSV width)
        for k in [
            "model.backbone", "model.dropout", "training.batch_size",
            "training.learning_rate", "training.weight_decay", "training.epochs",
            "data.train_ratio", "data.val_ratio", "data.num_workers", "data.pin_memory",
            "seed", "device", "cuda_available", "cuda_device_count", "cuda_name_0",
            "dataset.class_names", "dataset.class_counts",
        ]:
            val = None
            if k in self.context_snapshot:
                val = self.context_snapshot.get(k)
            elif k in self._device_info():
                val = self._device_info().get(k)
            flat[f"ctx.{k}"] = val

        self._write_csv_row(flat)

    def _write_csv_row(self, row: Dict[str, Any]) -> None:
        # Initialize header on first write
        if not self._csv_initialized:
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                writer.writeheader()
                writer.writerow(row)
            self._csv_initialized = True
        else:
            # If keys changed between writes, extend header
            if not self.csv_path.exists():
                self._csv_initialized = False
                self._write_csv_row(row)
                return
            # Read existing header
            with open(self.csv_path, "r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader)
            if set(header) != set(row.keys()):
                # Extend header with new keys (preserve order: existing + new)
                new_header = header + [k for k in row.keys() if k not in header]
                rows: List[List[str]] = []
                with open(self.csv_path, "r", newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for existing in reader:
                        rows.append([existing.get(h, "") for h in new_header])
                with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(new_header)
                    writer.writerows(rows)
                    writer.writerow([row.get(h, "") for h in new_header])
            else:
                with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                    writer.writerow(row)
