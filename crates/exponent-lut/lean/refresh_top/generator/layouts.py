"""The backend layout of every ring of a parameter directory, by ring name.

`layouts(d)` maps each ring abbreviation of `TopSetup.lean` (`Q1`, `Q0`, `M01`, …) that the backend
`generated/Backend.lean` lays out to the name of its layout (`layout<k>`).
"""
import re


def layouts(d):
    setup = open(f'{d}/TopSetup.lean').read()
    rings = dict(re.findall(r'^abbrev (\w+) : Nat := (\d+)$', setup, re.M))
    backend = open(f'{d}/generated/Backend.lean').read()
    by_modulus = {m: name for name, m in re.findall(r'^def (layout\d+) : MxxRuntime\.RegularLayout (\d+) :=',
                                                     backend, re.M)}
    return {ring: by_modulus[m] for ring, m in rings.items() if m in by_modulus}
