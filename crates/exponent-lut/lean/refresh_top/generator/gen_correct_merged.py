"""Write `TopCorrect.lean` for the merged stages: every run of the worst-case relation decodes the
ideal output.

The run's conjuncts are the external inputs' validity, the payload and the mask, the three merged
stages, the decode stage, the requirement and the ideal. The instance theorem
(`Part_I_Assembly.lean`, proven once for a symbolic instance `i0`) gives the residue of every
instance from the stages' bodies at the instance (`StageWrappers.lean`); its hypotheses on packed
external inputs hold member by member (`get_all8`, `get_const8`). `decode_zero` turns the eight
residues into the decode stage's output `0`, which is the ideal output.

Usage: `python3 gen_correct_merged.py <parameter directory> [modular]`; with `modular` the instance
theorem is `Part_I.Modular.partAssembly` of `Modular_Part_I_Assembly.lean` (the same statement,
proven over the facts modules, `generator/gen_modular_assembly.py`).
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
from chain import binder_groups, group_names
from claim import decode_externals

d = sys.argv[1]
MODULAR = len(sys.argv) > 2 and sys.argv[2] == 'modular'
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

# The requirement facts: the families in the order the requirement takes them.
req_line = next(l for l in rconj if l.startswith('Requirement_0.generatedRoot'))
req_inputs = re.findall(r'external\.(input_\d+)', req_line)
facts = open(f'{d}/RequirementRoot.lean').read()
fgroups, fconcl = binder_groups(facts, 'requirementFacts')
fwires = [group_names(g)[0] for g in fgroups if g.startswith('{w_')]
wire_input = dict(zip(fwires, req_inputs))
fparts = re.split(r' ∧ (?=\(∀|w_\d+_0 = 1 - )', ' '.join(fconcl.split()))
fnames = [f'rq{k}' for k in range(len(fparts))]
sums = {}
for k, f in enumerate(fparts):
    m = re.match(r'\(∀ o : Fin \d+, ∑ k : Fin 5, (w_\d+_0) \(cellIndex o k\) = 1\)', f)
    if m:
        sums[wire_input[m.group(1)]] = f'rq{k}'
label = wire_input[fwires[0]]
X = decode_externals(d)
target = re.search(r'execution\.«(stage_\d+)» = execution\.«ideal»', claim).group(1)

# the types of the external inputs
_ext = claim[claim.index('structure ExternalInputs where'):claim.index('def ValidExternals')]
INPUT_TYPES = dict(re.findall(r'^  (input_\d+) : (.*)$', _ext, re.M))

PACK = re.compile(r'\(\[(external\.input_\d+(?:, external\.input_\d+)*)\]\.get ⟨i0, hi0⟩\)')


def member_fact(ty):
    """A proof of `ty`, a fact about external inputs (no packs)."""
    m = re.fullmatch(r'∀ o : Fin \d+, ∑ k : Fin 5, external\.(input_\d+) \(cellIndex o k\) = 1', ty)
    if m:
        return sums[m.group(1)]
    m = re.fullmatch(r'∀ k, external\.(input_\d+) k \* external\.\1 k = 1', ty)
    if m:
        return f'(fun k ↦ ({fnames[0]} k).2.1)'
    if re.fullmatch(r'∀ k, external\.input_\d+ k \* 2 \+ \(1 - external\.input_\d+ k\) \* 8 = \(labelExp k : Int\) \* 4', ty):
        return f'(fun k ↦ by rw [← ({fnames[0]} k).1]; exact ({fnames[0]} k).2.2)'
    if ty == f'external.{X["ep"]} = 1 - external.{X["e"]}':
        return fnames[-2]
    xs = sorted(set(re.findall(r'external\.(input_\d+)', ty)))
    have = ' '.join(f'have := {vname[x]}{" i" if ty.startswith("∀") else ""};' for x in xs)
    if ty.startswith('∀'):
        return f'(fun i ↦ by {have} omega)'
    return f'(by {have} omega)'


def hyp_proof(ty):
    """A proof of the instance theorem's hypothesis `ty` at instance `i0`."""
    packs = PACK.findall(ty)
    if not packs:
        return member_fact(ty)
    members = [p.split(', ') for p in packs]
    if all(len(set(m)) == 1 for m in members):
        plain = PACK.sub(lambda m: m.group(1).split(', ')[0], ty)
        return f'(by simp only [get_const8]; exact {member_fact(plain)})'
    assert len(set(packs)) == 1, ty
    body = ty.replace(f'([{packs[0]}].get ⟨i0, hi0⟩)', 'x')
    proofs = [member_fact(ty.replace(f'([{packs[0]}].get ⟨i0, hi0⟩)', m)) for m in members[0]]
    xty = INPUT_TYPES[members[0][0].split('.')[1]]
    return f'(get_all8 (P := fun (x : {xty}) ↦ {body}) {" ".join(proofs)} ⟨i0, hi0⟩)'


text = open(f'{d}/Part_I_Assembly.lean').read()
groups, concl = binder_groups(text, 'partAssembly')
nums = {}
for l in rconj:
    m = re.match(r'Stage_(\w+)\.generatedRoot Backend\.backend hashModel stage_(\d+)_params', l)
    if m:
        nums[m.group(1)] = int(m.group(2))
names = ['hv'] + [f'r{k}' for k in range(1, len(rconj))]
rk = lambda prefix: 'r' + str(next(k for k, l in enumerate(rconj) if l.startswith(prefix)))
args = []
for g in groups:
    if not g.startswith('('):
        continue
    flat = ' '.join(g.split())
    n = group_names(g)[0]
    ty = flat[flat.index(' : ') + 3:-1]
    if n in ('hr', 'hsm', 'hG'):
        args.append(n)
    elif re.fullmatch(r'h\d+', n) and ty.startswith('out'):
        args.append('hrE' if 'reducePoly Q0 N r' in ty else 'hsmE')
    elif n in ('hpre', 'henc', 'hon'):
        stage = {'hpre': 'preprocess', 'henc': 'encrypt', 'hon': 'online_eval'}[n]
        args.append(f'({stage}_at {rk(f"Stage_{stage}.")} ⟨i0, hi0⟩)')
    else:
        args.append(hyp_proof(ty))
mask_sel = re.search(r'sigmaOf \(\[(external\.input_\d+)[^\]]*\]\.get ⟨i0, hi0⟩\) \(\[(external\.input_\d+)', concl)
fresh = re.search(r'freshE \(\[(external\.input_\d+), (external\.input_\d+)[^\]]*\]\.get ⟨i0, hi0⟩\) '
                  r'\(\[(external\.input_\d+), (external\.input_\d+)', concl)
dec_line = next(l for l in rconj if l.startswith('Stage_decode'))
dec_args = re.findall(r'execution\.«stage_\d+»(?:\.[12])*|external\.input_\d+', dec_line)
on = nums['online_eval']
# the residues the decode stage reads, in its input order: unpacked member `k` of the online
# root's residue family (`online_eval_unpack`)
res_pos = [a.count('.2') for a in dec_args if a.startswith(f'execution.«stage_{on}»')]
inst = [p - 7 for p in res_pos]

out = ['import RequirementRoot', 'import TopSecrets', 'import TopGadgets', 'import TopFinal',
       'import TopDecodeCrt', 'import TopUnpack', 'import StageWrappers',
       'import Modular_Part_I_Assembly' if MODULAR else 'import Part_I_Assembly']
out.append('''
/-! Every run of the worst-case relation decodes the ideal output (written by
`generator/gen_correct_merged.py`). -/

namespace RefreshTopProof

open Mxx.Primitives MxxRuntime MxxExponentLut

/-- A fact about every member of a packed input holds at every instance. -/
theorem get_all8 {α : Type} {P : α → Prop} {a0 a1 a2 a3 a4 a5 a6 a7 : α} (h0 : P a0) (h1 : P a1)
    (h2 : P a2) (h3 : P a3) (h4 : P a4) (h5 : P a5) (h6 : P a6) (h7 : P a7) (i : Fin 8) :
    P ([a0, a1, a2, a3, a4, a5, a6, a7].get i) := by
  fin_cases i <;> assumption

@[simp] theorem get_const8 {α : Type} (a : α) (i : Fin 8) : [a, a, a, a, a, a, a, a].get i = a := by
  fin_cases i <;> rfl

set_option maxHeartbeats 0 in
set_option maxRecDepth 65536 in
theorem correct {model : HashModel} {external : GeneratedClaim.ExternalInputs}
    {execution : GeneratedClaim.Execution} (h : GeneratedClaim.Runs model external execution) :
    execution.«TARGET» = execution.«ideal» := by'''.replace('TARGET', target))
out.append(f'  obtain ⟨{", ".join(names)}⟩ := h')
out.append(f'  obtain ⟨{", ".join(f"v{k}" for k in range(len(conj)))}⟩ := hv')
out.append(f'  obtain ⟨r, hrE, hr⟩ := payload_ternary {rk("Stage_payload")}')
out.append(f'  obtain ⟨sm, hsmE, hsm⟩ := mask_ternary {rk("Stage_mask")}')
out.append(f'  obtain ⟨{", ".join(fnames)}⟩ := requirementFacts {rk("Requirement_0")} r{len(rconj) - 1}')
out.append('  obtain ⟨G, hG⟩ := gadgetQ1')
out.append('  -- the instance theorem at every instance')
out.append('  have P := fun (i0 : Nat) (hi : i0 < 8) ↦ haveI : Fact (i0 < 8) := ⟨hi⟩\n'
           '    ' + ('Part_I.Modular.partAssembly' if MODULAR else 'Part_I.partAssembly') + ' (i0 := i0) ' + ' '.join(args))
out.append('  have U := online_eval_unpack ' + rk('Stage_online_eval.'))
out.append(f'''  have hσ : external.{X["sigma"]} = sigmaOf {mask_sel.group(1)} {mask_sel.group(2)} := by
    have hl : ∀ k, external.{label} k = (labelExp k : Int) := fun k ↦ ({fnames[0]} k).1
    funext o
    rw [{fnames[-1]} o, sigmaOf, zOf]
    simp only [hl]
  rw [ideal_zero {rk("Ideal")}]
  refine decode_zero {rk("Stage_decode")} hG (fun j ↦ ![freshE {fresh.group(1)} {fresh.group(3)},
    freshE {fresh.group(2)} {fresh.group(4)}] j)
    (fun j c i ↦ by fin_cases j <;> exact lt_of_lt_of_le (freshE_bound _ _ c i) (by decide))
    {' '.join('?_' for _ in inst)}''')
proj = lambda k: f'U.{"2." * k}1' if k < 7 else 'U.2.2.2.2.2.2.2'
for i0 in inst:
    out.append(f'  · rw [hσ{", hrE" if i0 % 2 == 0 else ""}, {proj(i0)}]\n    exact P {i0} (by decide)')
out.append('\nend RefreshTopProof\n')
open(f'{d}/TopCorrect.lean', 'w').write('\n'.join(out))
