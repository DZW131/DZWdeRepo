"""HQRF Phase-0 provenance and artifact helpers."""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    rows = list(rows); path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8"); return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def check_train_path(path):
    value = Path(path).resolve().as_posix().lower()
    if not value.endswith("/bcss-wsss/training") or "/val" in value or "/test" in value or "luad" in value:
        raise ValueError("HQRF Phase-0 is restricted to BCSS training")


def install_train_access_guard():
    accesses = set()
    def guard(event, args):
        if event == "open" and args and isinstance(args[0], (str, bytes)):
            value = str(args[0]).replace("\\", "/").lower()
            if "luad" in value and not value.endswith(".py"):
                raise RuntimeError("LUAD access prohibited")
            if "bcss-wsss" in value:
                if "/training/" not in value:
                    raise RuntimeError("Non-training BCSS access prohibited: " + value)
                accesses.add(value)
    sys.addaudithook(guard)
    return accesses


def protected_sources(root):
    root = Path(root)
    files = ["train_sshr.py", "network/resnet38d.py", "network/resnet38_cls.py", "tool/GenDataset.py", "tool/torchutils.py"]
    result = {}
    for name in files:
        reference = subprocess.check_output(["git", "show", "4e9a2887b220d17e27649d72a3d13f32b7ebe8f9:" + name], cwd=root)
        actual = (root / name).read_bytes().replace(b"\r\n", b"\n")
        if reference.replace(b"\r\n", b"\n") != actual:
            raise RuntimeError("Protected official source changed: " + name)
        result[name] = hashlib.sha256(actual).hexdigest()
    return result
