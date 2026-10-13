"""Write `Part_<c_j>_Assembly.lean`: the theorem of part `(c, j)` over the preprocessing, encryption
and online roots as `Claim.Runs` applies them, destructuring all three (the preprocessing's names
prefixed `p`, the encryption's `q`, see `chain.prefixed`).

Usage: `python3 gen_assembly.py <parameter directory>`.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
from chain import prefixed
from gen_part import parts_of
from patterns import destruct_root

d = sys.argv[1]
claim = open(f'{d}/generated/Claim.lean').read()
for part in parts_of(d):
    lines = {}
    for stage in ('preprocess', 'encrypt', 'online_eval'):
        ns = f'Stage_{stage}_{part}'
        line = next(l.strip().rstrip('∧').strip() for l in claim.split('\n') if l.strip().startswith(f'{ns}.generatedRoot'))
        lines[stage] = re.sub(r'execution\.«stage_(\d+)»', r'out\1', line)
    outs = sorted({int(x) for l in lines.values() for x in re.findall(r'\bout(\d+)\b', l)})
    ds = {s: destruct_root(f'{d}/generated/Stage_{s}_{part}.lean', f'Stage_{s}_{part}', 'generatedRoot', h)
          for s, h in (('preprocess', 'hpre'), ('encrypt', 'henc'), ('online_eval', 'hon'))}
    body = f'''import Claim
import Part_{part}_KeyMask

/-! The assembly of part `{part}` (written by `generator/gen_assembly.py`). -/

namespace RefreshTopProof.Part_{part}

open Mxx.Primitives MxxRuntime MxxExponentLut RefreshTopProof GeneratedClaim

set_option maxHeartbeats 0 in
set_option maxRecDepth 65536 in
theorem partAssembly {{model : HashModel}} {{external : ExternalInputs}} {' '.join(f'{{out{k} : _}}' for k in outs)}
    (hpre : {lines['preprocess']})
    (henc : {lines['encrypt']})
    (hon : {lines['online_eval']}) : True := by
{prefixed(ds['preprocess'], 'p')}{prefixed(ds['encrypt'], 'q')}{ds['online_eval']}  trivial

end RefreshTopProof.Part_{part}
'''
    body = body.replace('generatedRoot Backend.backend hashModel', 'generatedRoot Backend.backend model')
    open(f'{d}/Part_{part}_Assembly.lean', 'w').write(body)
    break
