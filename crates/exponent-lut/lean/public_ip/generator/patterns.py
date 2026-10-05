"""Destructuring patterns of generated constraint chains.

A generated root is a chain `constraints_0 ∧ ... ∧ constraints_k` whose last conjunct calls the next
link and whose final link ends in `outputs = ...`. The pattern of a root names every runtime
relation after its output wire (`hX` for `w_X_0`, `hX_c` for `w_X_c`), drops trivial bound facts,
and names the output equation `hout`.
"""

import re

CALL = re.compile(r"^\s*(\S+)\.constraints_(\d+)\b")


def constraints(text: str, root: str) -> dict[int, list[str]]:
    """The conjunct lines of every `root.constraints_k`."""
    links: dict[int, list[str]] = {}
    pattern = re.compile(rf"^abbrev {re.escape(root)}\.constraints_(\d+) .*:= *$")
    lines = text.splitlines()
    for index, line in enumerate(lines):
        match = pattern.match(line)
        if not match:
            continue
        body = []
        for entry in lines[index + 1:]:
            if not entry.startswith("  "):
                break
            body.append(entry.strip().removesuffix(" ∧"))
        links[int(match.group(1))] = body
    return links


def name(conjunct: str) -> str:
    if conjunct.startswith("outputs ="):
        return "hout"
    if conjunct.startswith("(∀"):
        return "hloop"
    if not conjunct.startswith(("MxxRuntime.", "MxxIR.")) or conjunct.startswith("MxxIR.selectInt"):
        return "-"
    wire = re.findall(r"w_(\d+)_(\w+)", conjunct)[-1]
    return f"h{wire[0]}" if wire[1] == "0" else f"h{wire[0]}_{wire[1]}"


def pattern(text: str, root: str, start: int = 0) -> list[str]:
    links = constraints(text, root)
    names: list[str] = []
    link = start
    while True:
        body = links[link]
        for conjunct in body[:-1]:
            names.append(name(conjunct))
        last = body[-1]
        call = CALL.match(last)
        if call and call.group(1).endswith(root.split(".")[-1]) and "constraints_" in last:
            link = int(call.group(2))
            continue
        names.append(name(last))
        return names


def witnesses(text: str, root: str) -> list[str]:
    """The existential witnesses of a parallel root, in binder order."""
    body = text.split(f"def {root} ", 1)[1].split("\ndef ", 1)[0]
    return [f"w{a}" if b == "0" else f"w{a}_{b}" for a, b in re.findall(r"∃ \(w_(\d+)_(\w+) :", body)]


if __name__ == "__main__":
    import sys

    path, root = sys.argv[1], sys.argv[2]
    text = open(path, encoding="utf-8").read()
    names = (witnesses(text, root) if not root.endswith(".body") else []) + pattern(
        text, root.removesuffix(".body"))
    print("⟨" + ", ".join(names) + "⟩")
