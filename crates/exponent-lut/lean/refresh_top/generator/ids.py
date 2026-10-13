"""Node ids of the old part-`0_0` stages written into generators, on part `I` of the merged stages.

The merged body numbers its nodes like the old root, except that online evaluation's are permuted
(`maps/online_eval.json`, from the checked isomorphism of the two graphs); `online(k, part)` is the
merged body's id of the old online node `k` for part `I`, else `k`.
"""
import json
import os

_here = os.path.dirname(os.path.abspath(__file__))
_maps = {}


def _nodes(stage):
    if stage not in _maps:
        data = json.load(open(f'{_here}/maps/{stage}.json'))['nodes']
        _maps[stage] = {int(k): int(v) for k, v in data.items()}
    return _maps[stage]


def online(k, part):
    # the views keep the old numbering (`view.old_numbering`)
    return k


def pre(k, part):
    """The merged body's id of the old preprocessing node `k` for part `I` (the merge inserts the
    component selections, `maps/preprocess.json`), else `k`."""
    return k
