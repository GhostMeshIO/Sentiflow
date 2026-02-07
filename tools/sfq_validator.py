import json

REQUIRED_FIELDS = ["id", "name", "version", "author", "entrypoint", "plugin_format"]

def validate_plugin_manifest(path):
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except Exception as e:
        return False, f"Invalid JSON: {e}"

    for field in REQUIRED_FIELDS:
        if field not in data:
            return False, f"Missing field: {field}"

    if data["plugin_format"] != "SFQ/1.0":
        return False, "plugin_format must be SFQ/1.0"

    return True, "Manifest valid"
