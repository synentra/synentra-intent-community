#!/usr/bin/env python3
"""Generate labels.json from an ONNX export directory (config.json)."""

import json
import os
import sys

def main():
    if len(sys.argv) != 2:
        print("Usage: generate_labels_json.py <onnx_dir>")
        sys.exit(1)

    onnx_dir = sys.argv[1]
    config_path = os.path.join(onnx_dir, "config.json")

    with open(config_path, "r") as f:
        config = json.load(f)

    # Map id2label to an ordered list of labels
    labels = [config["id2label"][str(i)] for i in range(len(config["id2label"]))]

    # Output to release_temp/labels.json (working directory is repo root)
    os.makedirs("release_temp", exist_ok=True)
    with open("release_temp/labels.json", "w") as f:
        json.dump(labels, f)

    print(f"labels.json written with {len(labels)} labels.")

if __name__ == "__main__":
    main()