"""Resolve proof-template scope names against the current Lean export."""

import argparse
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
MAPPING = HERE / "scope_names.json"


def derive(previous):
    mappings = {}
    for current in sorted((HERE / "generated").glob("Stage_*.lean")):
        if current.stem.endswith("_Facts"):
            continue
        old = re.findall(r"^def (\S+) \(", (previous / current.name).read_text(), re.M)
        new = re.findall(r"^def (\S+) \(", current.read_text(), re.M)
        if len(old) != len(new):
            raise ValueError(f"Scope count changed in {current.name}")
        retained = set(old) & set(new)
        for before, after in zip(old, new):
            if before != after and (before in retained or after in retained):
                raise ValueError(f"Scope order changed in {current.name}: {before}, {after}")
        changes = {before: after for before, after in zip(old, new) if before != after}
        if changes:
            mappings[current.stem] = changes
    return mappings


def translate(text, module=None, reverse=False):
    mappings = json.loads(MAPPING.read_text())
    names = {}
    for stage, changes in mappings.items():
        for before, after in changes.items():
            if reverse:
                before, after = after, before
            names[f"{stage}.{before}"] = f"{stage}.{after}"
            if stage == module:
                names[before] = after
    pattern = r"(?<![\w])(?:" + "|".join(map(re.escape, sorted(names, key=len, reverse=True)))
    return re.sub(pattern + r")(?![\w])", lambda match: names[match[0]], text)


def write_lean(path, text):
    Path(path).write_text(translate(text))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", type=Path, required=True)
    args = parser.parse_args()
    mappings = derive(args.previous)
    MAPPING.write_text(json.dumps(mappings, indent=2, sort_keys=True) + "\n")
    print(f"Derived {sum(map(len, mappings.values()))} scope renamings")


if __name__ == "__main__":
    main()
