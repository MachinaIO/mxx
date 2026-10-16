"""Read an exported module and its imported partitions in dependency order.

Partition files keep their parent module's namespace. Follow imports rather than
sorting part numbers: callers must only see declarations available to Lean there.
"""
from pathlib import Path
import re


def partition_base(module):
    return re.sub(r'_Part_\d+$', '', module)


def module_sources(gdir, module):
    """Return `(physical module, source)` pairs for one logical export module."""
    sources, visited = [], set()
    base = partition_base(module)

    def visit(name):
        if name in visited:
            return
        visited.add(name)
        text = (Path(gdir) / f'{name}.lean').read_text()
        for dependency in re.findall(r'^import (\w+)\s*$', text, re.M):
            if partition_base(dependency) == base:
                visit(dependency)
        sources.append((name, text))

    visit(module)
    return sources
