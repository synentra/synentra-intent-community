#!/usr/bin/env python3
"""Generate labels.json from an ONNX export directory's config.json."""

import json
import os
import sys


def main():
    if len(sys.argv) != 2:
        print("Usage: generate_labels_json.py <onnx_dir>")
        sys.exit(1)

    onnx_dir = sys.argv[1]
    config_path = os.path.join(onnx_dir, "config.json")
    labels_path = os.path.join(onnx_dir, "labels.json")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    id2label = config.get("id2label")

    if not id2label:
        raise ValueError(f"id2label not found in {config_path}")

    labels = [
        id2label[str(i)]
        for i in range(len(id2label))
    ]

    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2)

    print(f"labels.json written to {labels_path} with {len(labels)} labels.")


if __name__ == "__main__":
    main()