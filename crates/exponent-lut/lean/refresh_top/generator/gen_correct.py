"""Write `TopCorrect.lean`: every run of the worst-case relation decodes the ideal output.

The run's conjuncts are the external inputs' validity, the payload and the mask, the three stages
of every part, the decode stage, the requirement and the ideal. The part theorems
(`Part_<c_j>_Assembly.lean`) give the residues of every chain of every row; `decode_zero` turns
them into the decode stage's output `0`, which is the ideal output.

Usage: `python3 gen_correct.py <parameter directory>`.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
from chain import binder_groups, group_names
from gen_part import parts_of
from claim import decode_externals

d = sys.argv[1]
claim = open(f'{d}/generated/Claim.lean').read()
valid = claim[claim.index('def ValidExternals'):]
valid = valid[valid.index(':=') + 2:valid.index('\n\n')]
conj = [c.strip().rstrip('∧').strip() for c in valid.strip().split('\n') if c.strip()]
runs = claim[claim.index('def Runs'):claim.index('/-- Every execution')]
rconj = [l.strip().rstrip('∧').strip() for l in runs.split('\n')[2:] if l.strip()]

# The validity conjunct of every external input.
vname = {}
for k, c in enumerate(conj):
    for x in set(re.findall(r'external\.(input_\d+)', c)):
        vname.setdefault(x, f'v{k}')


def valid_fact(ty):
    """A proof of `ty`, a fact about one external input, from its validity conjunct."""
    x = re.search(r'external\.(input_\d+)', ty).group(1)
    v = vname[x]
    if ty.startswith('∀'):
        return f'(fun i ↦ by have := {v} i; omega)'
    return f'(by have := {v}; omega)'


# The requirement facts: the families in the order the requirement takes them.
req_line = next(l for l in rconj if l.startswith('Requirement_0.generatedRoot'))
req_inputs = re.findall(r'external\.(input_\d+)', req_line)
facts = open(f'{d}/RequirementRoot.lean').read()
fgroups, fconcl = binder_groups(facts, 'requirementFacts')
fwires = [group_names(g)[0] for g in fgroups if g.startswith('{w_')]
wire_input = dict(zip(fwires, req_inputs))
fparts = re.split(r' ∧ (?=\(∀|w_\d+_0 = 1 - )', ' '.join(fconcl.split()))
fnames = []
sums = {}
for k, f in enumerate(fparts):
    m = re.match(r'\(∀ o : Fin \d+, ∑ k : Fin 5, (w_\d+_0) \(cellIndex o k\) = 1\)', f)
    if m:
        sums[wire_input[m.group(1)]] = f'rq{k}'
    fnames.append(f'rq{k}')

parts = parts_of(d)
# The stage the correctness claim compares with the ideal output.
target = re.search(r'execution\.«(stage_\d+)» = execution\.«ideal»', claim).group(1)
X = decode_externals(d)
label = wire_input[fwires[0]]
# The mask selectors of `σ₀` and each row's fresh-noise selectors, from the part theorems.
concl = {p: open(f'{d}/Part_{p}_Assembly.lean').read() for p in parts}
mask_sel = re.search(r'sigmaOf (external\.input_\d+ external\.input_\d+)', concl[parts[0]]).group(1)
fresh_sel = [re.search(r'freshE (external\.input_\d+ external\.input_\d+)', concl[f'0_{j}']).group(1)
             for j in range(2)]
out = ['import RequirementRoot', 'import TopSecrets', 'import TopDecodeRun', 'import TopGadgets', 'import TopFinal']
out += [f'import Part_{p}_Assembly' for p in parts]
out.append('''
/-! Every run of the worst-case relation decodes the ideal output (written by
`generator/gen_correct.py`). -/

namespace RefreshTopProof

open Mxx.Primitives MxxRuntime MxxExponentLut

set_option maxHeartbeats 0 in
set_option maxRecDepth 65536 in
theorem correct {model : HashModel} {external : GeneratedClaim.ExternalInputs}
    {execution : GeneratedClaim.Execution} (h : GeneratedClaim.Runs model external execution) :
    execution.«TARGET» = execution.«ideal» := by'''.replace('TARGET', target))
names = ['hv'] + [f'r{k}' for k in range(1, len(rconj))]
out.append(f'  obtain ⟨{", ".join(names)}⟩ := h')
out.append(f'  obtain ⟨{", ".join(f"v{k}" for k in range(len(conj)))}⟩ := hv')
pay = next(k for k, l in enumerate(rconj) if l.startswith('Stage_payload'))
mask = next(k for k, l in enumerate(rconj) if l.startswith('Stage_mask'))
req = next(k for k, l in enumerate(rconj) if l.startswith('Requirement_0'))
ideal = next(k for k, l in enumerate(rconj) if l.startswith('Ideal'))
dec = next(k for k, l in enumerate(rconj) if l.startswith('Stage_decode'))
out.append(f'  obtain ⟨r, hrE, hr⟩ := payload_ternary r{pay}')
out.append(f'  obtain ⟨sm, hsmE, hsm⟩ := mask_ternary r{mask}')
out.append(f'  obtain ⟨{", ".join(fnames)}⟩ := requirementFacts r{req} r{len(rconj) - 1}')
out.append('  obtain ⟨G, hG⟩ := gadgetQ1')
for p in parts:
    text = open(f'{d}/Part_{p}_Assembly.lean').read()
    groups, _ = binder_groups(text, 'partAssembly')
    args = []
    for g in groups:
        if not g.startswith('('):
            continue
        flat = ' '.join(g.split())
        n = group_names(g)[0]
        ty = flat[flat.index(' : ') + 3:-1]
        if n in ('hr', 'hsm'):
            args.append(n)
        elif re.fullmatch(r'h\d+', n) and ty.startswith('out'):
            args.append('hrE' if 'reducePoly Q0 N r' in ty else 'hsmE')
        elif n.startswith('hsum'):
            args.append(sums[f'input_{n[4:]}'])
        elif n == 'hlc':
            args.append(f'(fun k ↦ ({fnames[0]} k).2.1)')
        elif n == 'hlr':
            args.append(f'(fun k ↦ by rw [← ({fnames[0]} k).1]; exact ({fnames[0]} k).2.2)')
        elif n == 'h' + X['ep'][6:]:
            args.append(fnames[-2])
        elif n in ('hpre', 'henc', 'hon'):
            stage = {'hpre': 'preprocess', 'henc': 'encrypt', 'hon': 'online_eval'}[n]
            k = next(k for k, l in enumerate(rconj) if l.startswith(f'Stage_{stage}_{p}.'))
            args.append(f'r{k}')
        elif n == 'hG':
            args.append('hG')
        else:
            args.append(valid_fact(ty))
    out.append(f'  have P{p} := Part_{p}.partAssembly ' + ' '.join(args))
out.append(f'''  have hσ : external.{X["sigma"]} = sigmaOf {mask_sel} := by
    have hl : ∀ k, external.{label} k = (labelExp k : Int) := fun k ↦ ({fnames[0]} k).1
    funext o
    rw [{fnames[-1]} o, sigmaOf, zOf]
    simp only [hl]
  rw [ideal_zero r{ideal}]
  refine decode_zero r{dec} hG (fun j ↦ ![freshE {fresh_sel[0]},
    freshE {fresh_sel[1]}] j) (fun j c i ↦ by fin_cases j <;> exact freshE_bound _ _ c i)
    {' '.join('?_' for _ in parts)}''')
for p in sorted(parts, key=lambda p: (p[2], p[0])):
    out.append(f'  · rw [hσ{", hrE" if p.endswith("_0") else ""}]\n    exact P{p}')
out.append('\nend RefreshTopProof\n')
open(f'{d}/TopCorrect.lean', 'w').write('\n'.join(out))
