#!/usr/bin/env python3
"""
 SentiFlow Plugin Builder (sfqbuild.py)
 -------------------------------------
 Official utility for creating, packaging, extracting,
 and validating .sfq plugin modules.
"""

import os
import json
import shutil
import zipfile
import argparse
from datetime import datetime

PLUGIN_TEMPLATE = {
    "id": "sfq.example.plugin",
    "name": "ExamplePlugin",
    "version": "1.0.0",
    "author": "YourName",
    "description": "Describe your plugin.",
    "extends": [],
    "entrypoint": "main:init_plugin",
    "hooks": {},
    "provides": {
        "ops": [],
        "qpu_gates": [],
        "tensor_attributes": []
    },
    "permissions": {
        "allow_tensor_mutation": True,
        "allow_qpu_write": False,
        "allow_filesystem_access": False,
        "allow_network_access": False
    },
    "plugin_format": "SFQ/1.0",
    "senti_min_version": "0.5"
}


# -----------------------------------------------------------------------------
# UTILITY: VALIDATE MANIFEST
# -----------------------------------------------------------------------------
def validate_manifest(path):
    try:
        with open(path, "r") as f:
            manifest = json.load(f)
    except Exception as e:
        return False, f"Cannot read JSON: {e}"

    required = ["id", "name", "version", "author", "entrypoint", "plugin_format"]

    for key in required:
        if key not in manifest:
            return False, f"Missing required field '{key}'"

    if manifest["plugin_format"] != "SFQ/1.0":
        return False, "Unsupported plugin_format (expected SFQ/1.0)"

    return True, "Manifest is valid."


# -----------------------------------------------------------------------------
# PACKAGE DIRECTORY INTO .SFQ
# -----------------------------------------------------------------------------
def package_sfq(source_dir, output_file):
    if not os.path.exists(source_dir):
        raise FileNotFoundError(f"Directory not found: {source_dir}")

    manifest_path = os.path.join(source_dir, "plugin.json")
    valid, msg = validate_manifest(manifest_path)
    if not valid:
        raise Exception(f"Manifest validation failed: {msg}")

    with zipfile.ZipFile(output_file, "w", zipfile.ZIP_DEFLATED) as z:
        for root, dirs, files in os.walk(source_dir):
            for f in files:
                abs_path = os.path.join(root, f)
                rel_path = os.path.relpath(abs_path, source_dir)
                z.write(abs_path, rel_path)

    return output_file


# -----------------------------------------------------------------------------
# EXTRACT .SFQ INTO A DIRECTORY
# -----------------------------------------------------------------------------
def extract_sfq(sfq_file, output_dir):
    with zipfile.ZipFile(sfq_file, "r") as z:
        z.extractall(output_dir)


# -----------------------------------------------------------------------------
# CREATE NEW PLUGIN SKELETON
# -----------------------------------------------------------------------------
def create_plugin(name):
    safe = name.lower().replace(" ", "_")
    root = f"{safe}_plugin"

    os.makedirs(root, exist_ok=True)
    os.makedirs(root + "/resources", exist_ok=True)

    # plugin.json
    manifest = PLUGIN_TEMPLATE.copy()
    manifest["name"] = name
    manifest["id"] = f"sfq.{safe}.001"
    manifest["author"] = "SentiFlow Developer"
    manifest["version"] = "0.1.0"
    manifest["description"] = f"{name} plugin for SentiFlow."

    with open(root + "/plugin.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # main.py
    with open(root + "/main.py", "w") as f:
        f.write(
"""def init_plugin(sflow):
    print("[SFQ] Plugin initialized: TEMPLATE")

def on_tensor_create(t):
    pass
"""
        )

    return root


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SentiFlow .sfq Plugin Builder")

    parser.add_argument("--new", help="Create new plugin skeleton")
    parser.add_argument("--package", nargs=2, help="Package directory into .sfq")
    parser.add_argument("--extract", nargs=2, help="Extract .sfq file")
    parser.add_argument("--validate", help="Validate a plugin.json")
    
    args = parser.parse_args()

    if args.new:
        result = create_plugin(args.new)
        print(f"Created plugin skeleton: {result}")

    if args.package:
        src, dst = args.package
        output = package_sfq(src, dst)
        print(f"Packaged plugin: {output}")

    if args.extract:
        sfq_file, outdir = args.extract
        extract_sfq(sfq_file, outdir)
        print(f"Extracted to: {outdir}")

    if args.validate:
        valid, msg = validate_manifest(args.validate)
        print("VALID" if valid else "INVALID", "-", msg)
