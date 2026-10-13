"""Write `Gen_Correct.lean`: every run of the generated claim decodes the ideal output.

From `GeneratedClaim.Runs` the theorem `correct` takes
* the ternary secrets (every stage whose root is one `uniformIntervalSample (-1) 1` sample),
* the requirement facts (`GenRequirement.requirement_facts`, `gen_requirement2.py`),
* a gadget matrix of the decode stage's gadget modulus (the backend's layout),
* the part theorem: every CRT term the decode stage reads is the term of its expected row plus a
  small error row (`∀ i : Fin S`, one application per instance index, nothing enumerated),
* the looped decode (`decode_zero_loops`, `gen_decode_loops.py`) and the ideal stage (`… * 0`),
and concludes `execution.«decode» = execution.«ideal»` (the claim's `CorrectnessClaim` sides).

Until the part theorem of the instance pipeline exists, `--part` is omitted and the file states
it as `part_terms` with the agreed shape, proved by `sorry` (the only `sorry`).

Usage: `python3 gen_correct2.py <parameter directory> [--out FILE] [--part MODULE:THEOREM]`;
`python3 gen_correct2.py <average directory> --certificate [--guard]` writes
`proof/RefreshTopCertificate.lean` of an average-case package (`--guard`: check the axioms).
"""
import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xmodel  # noqa: E402
from xmodel import strip_parens, tuple_items  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('d')
ap.add_argument('--out')
ap.add_argument('--part', nargs='?', const='Gen_Part')
ap.add_argument('--certificate', action='store_true')
ap.add_argument('--guard', action='store_true')
a = ap.parse_args()
d = a.d


def fail(msg):
    sys.exit(f'gen_correct2: {msg}')


def check(cond, msg):
    if not cond:
        fail(msg)


# ---------------------------------------------------------------------------------------------
# The average-case certificate

if a.certificate:
    claim = open(f'{d}/generated/Claim.lean').read()
    m = re.search(r'execution\.(«\w+») = execution\.(«\w+»)', claim.split('def CorrectnessClaim', 1)[1])
    check(m, 'CorrectnessClaim sides')
    lhs, rhs = m.groups()
    guard = '''/-- info: 'RefreshTopProof.certificate' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
''' if a.guard else ''
    open(f'{d}/proof/RefreshTopCertificate.lean', 'w').write(f'''import BridgeRuns
import Gen_Correct

/-!
# The average-case correctness claim of refresh_top

Every average-case run is a worst-case run (`GeneratedClaim.runs_bridge`), and every worst-case run
decodes the ideal output (`RefreshTopProof.GenCorrect.correct`): the set of tapes on which some run
fails is empty, so its measure is `0` (written by `generator/gen_correct2.py --certificate`).
-/

namespace RefreshTopProof

theorem certificate : GeneratedClaim.CorrectnessClaim := by
  intro model external
  have hempty : {{tape | ∃ execution, GeneratedClaim.Runs model external tape execution ∧
      ¬ execution.{lhs} = execution.{rhs}}} = ∅ := by
    ext tape
    simp only [Set.mem_setOf_eq, Set.mem_empty_iff_false, iff_false]
    rintro ⟨execution, hruns, hne⟩
    exact hne (GenCorrect.correct (GeneratedClaim.runs_bridge hruns))
  rw [hempty, MeasureTheory.measure_empty]
  exact zero_le _

{guard}#print axioms certificate

end RefreshTopProof
''')
    print(f'wrote {d}/proof/RefreshTopCertificate.lean', file=sys.stderr)
    sys.exit(0)

# ---------------------------------------------------------------------------------------------
# The claim

gdir = f'{d}/generated'
out_path = a.out or f'{d}/Gen_Correct.lean'
claim = open(f'{gdir}/Claim.lean').read()
m = re.search(r'execution\.(«\w+») = execution\.(«\w+»)', claim.split('def CorrectnessClaim', 1)[1])
check(m, 'CorrectnessClaim sides')
LHS, RHS = m.groups()
calls = xmodel.parse_claim(gdir)
names, secrets = [], []
dec = ideal = req = reqtrue = None
for k, c in enumerate(calls):
    if c['module'] is None:
        t = c['text']
        if t.startswith('ValidExternals'):
            names.append('hv')
        elif re.match(r'^execution\.«\w+» = true$', t):
            names.append('hreq')
            reqtrue = t
        else:
            fail(f'Runs conjunct {t[:60]}')
        continue
    mod = c['module']
    out = c['output'].strip()
    nm = f'r_{mod}'
    names.append(nm)
    c['name'] = nm
    params = c['params'][-1]
    c['param'] = params
    if out == f'execution.{LHS}':
        dec = c
    elif out == f'execution.{RHS}':
        ideal = c
    elif mod.startswith('Requirement_'):
        req = c
    elif c['args'] in ([], ['']) and len(c['params']) == 1:
        text = open(f'{gdir}/{mod}.lean').read()
        body = re.search(r'abbrev generatedRoot\.constraints_0 [^\n]*\n((?:  [^\n]*\n)*)', text)
        ot = re.search(r'def generatedRoot \(params : Params\) \(_ : Unit\) \(outputs : (.*?)\) : Prop', text)
        if body and ot and re.fullmatch(r'\s*MxxRuntime\.uniformIntervalSample \(-1\) \(1\) (w_\d+_\d+) ∧\n'
                                        r'\s*outputs = \1\n', body.group(1)):
            ty = re.sub(r'\b(modulus_\d+)\b', rf'{mod}.\1', ot.group(1))
            mm = re.match(r'^Mxx\.Primitives\.ExactMatrix (\S+) (\d+) 1 1$', ty)
            check(mm, f'{mod}: secret type {ty}')
            c['q'], c['n'] = mm.group(1), mm.group(2)
            secrets.append(c)
check(dec and ideal and req and reqtrue, 'decode / ideal / requirement stages')
check(len(names) == len(calls), 'Runs conjuncts')

# ---------------------------------------------------------------------------------------------
# The decode theorem

dl = open(f'{d}/TopDecodeLoops.lean').read()
NS = re.search(r'^namespace (\S+)', dl, re.M).group(1)
m = re.search(r'theorem decode_zero_loops \{.*?\n(.*?) :=\n', dl, re.S)
check(m, 'decode_zero_loops in TopDecodeLoops.lean (CRT tables not found?)')
stmt = m.group(0)
mg = re.search(r'\{G : ExactMatrix (\d+) (\d+) 1 (\d+)\} \(hG : gadgetMatrixRuns backend (\d+) (\d+) G\)', stmt)
check(mg, 'gadget hypothesis')
QG, N, MG, BASE, DIG = mg.groups()
mE = re.search(r'\(E : (Nat → ErrorMatrix \d+ 1 \d+)\)\s*\(hE : (.*?)\)\s*\(hterm : (.*?)\) :\n\s*out = 0', stmt, re.S)
check(mE, 'error rows / terms hypotheses')
ETY, HE, HTERM = mE.group(1), ' '.join(mE.group(2).split()), ' '.join(mE.group(3).split())
DIN = 'decodeInputs external execution'
HTERM = re.sub(r'\binputs\b', f'({DIN})', HTERM)
HTERM = re.sub(r'\b(decode_qN|decode_KN|decodeValue)\b', rf'{NS}.\1', HTERM)
backend = open(f'{gdir}/Backend.lean').read()
lay = [k for k, q in re.findall(r'^def (layout\d+) : MxxRuntime\.RegularLayout (\d+)', backend, re.M) if q == QG]
check(len(lay) == 1, f'backend layout of {QG}')

# the requirement facts' arguments
req_args = ' '.join(f'({x})' for x in req['args'])

# ---------------------------------------------------------------------------------------------
# Emit

pattern = ', '.join(names)
sec_haves = ''.join(f'\n  have hsec_{c["module"]} := secret_{c["module"]} {c["name"]}' for c in secrets)
sec_args = ' '.join(f'hsec_{c["module"]}' for c in secrets)
sec_hyps = ''.join(
    f'\n    (hsec_{c["module"]} : ∃ s : ErrorPoly {c["n"]}, execution.{c["output"].split(".", 1)[1]} = '
    f'(fun _ _ ↦ reducePoly {c["q"]} {c["n"]} s) ∧ polyNorm s ≤ 1)' for c in secrets)
part_sig = f'''{{model : HashModel}} {{external : GeneratedClaim.ExternalInputs}}
    {{execution : GeneratedClaim.Execution}} (h : GeneratedClaim.Runs model external execution){sec_hyps}
    (hreq : GenRequirement.Facts {req_args})
    {{G : ExactMatrix {QG} {N} 1 {MG}}} (hG : gadgetMatrixRuns Backend.backend {BASE} {DIG} G) :
    ∃ E : {ETY}, ({HE}) ∧
      {HTERM}'''
if a.part:
    # the part theorem of the instance pipeline (written by gen_part2.py):
    # `GenPart.part hR hreq i hG : TermOf (qOf i) (KOf i) (term execution i) (valueOf … G (i % R))`
    part_imports = f'import {a.part}\n'
    mS = re.search(r'hterm : ∀ i : Fin (\d+), TermOf \(decode_qN \(i / (\d+)\)\)', stmt)
    check(mS, 'instance count / components of decode_zero_loops')
    part_block = f'''
/-- The online CRT tables are the decode tables: instance `i` uses prime and basis element `i / R`
(one `decide` over the instance tables). -/
theorem tables_q : ∀ i : Fin GenPart.S, GenPart.qOf i = {NS}.decode_qN (i / GenPart.R) := by decide

theorem tables_K : ∀ i : Fin GenPart.S, GenPart.KOf i = {NS}.decode_KN (i / GenPart.R) := by decide

/-- The part theorem's requirement hypothesis `GenPart.Req` is exactly the requirement facts. -/
theorem req_of_facts {{external : GeneratedClaim.ExternalInputs}} (hreq : GenRequirement.Facts {req_args}) :
    GenPart.Req external := hreq
'''
    correct_tail = f'''  have hreq := req_of_facts F
  obtain ⟨G, hG⟩ := decode_gadget
  rw [ideal_zero {ideal['name']}]
  refine {NS}.decode_zero_loops {dec['name']} hG (GenPart.errE external) (GenPart.errE_bound hreq) (fun i ↦ ?_)
  have hT := GenPart.part hrun hreq i hG
  rw [tables_q i, tables_K i] at hT
  exact hT'''
else:
    part_imports = ''
    part_block = f'''
/-- PLACEHOLDER for the part theorem of the instance pipeline (package D): every CRT term the
decode stage reads is the term of its expected row plus a small error row, at every instance
index. Replaced by `--part` (module `Gen_Part`) once it exists. -/
theorem part_terms {part_sig} := by
  sorry
'''
    correct_tail = f'''  obtain ⟨G, hG⟩ := decode_gadget
  obtain ⟨E, hE, hT⟩ := part_terms hrun {sec_args} F hG
  rw [ideal_zero {ideal['name']}]
  exact {NS}.decode_zero_loops {dec['name']} hG E hE hT'''

sec_lemmas = ''.join(f'''
/-- The secret of `{c["module"]}` is one ternary polynomial. -/
theorem secret_{c["module"]} {{out : ExactMatrix {c["q"]} {c["n"]} 1 1}}
    (h : {c["module"]}.generatedRoot GeneratedClaim.{c["param"]} () out) :
    ∃ s : ErrorPoly {c["n"]}, out = (fun _ _ ↦ reducePoly {c["q"]} {c["n"]} s) ∧ polyNorm s ≤ 1 := by
  obtain ⟨w, hw, hout⟩ := h
  obtain ⟨s, hs, hn⟩ := ternary_poly hw
  exact ⟨s, hout.trans hs, hn⟩
''' for c in secrets)

dec_args = ', '.join(dec['args'] + ['()'])

text = f'''import Claim
import Gen_Requirement
import TopDecodeLoops
{part_imports}
/-! Every run of the generated claim decodes the ideal output (written by
`generator/gen_correct2.py`): the secrets are ternary, the requirement gives its facts
(`GenRequirement.requirement_facts`), the part theorem gives every CRT term the decode stage reads
(`∀ i`, one application per instance index), the looped decode outputs `0`
(`{NS}.decode_zero_loops`), and the ideal output is `0`. -/

set_option maxRecDepth 65536

namespace RefreshTopProof.GenCorrect

open Mxx.Primitives MxxRuntime MxxExponentLut RefreshTopProof GenDecode
{sec_lemmas}
/-- The ideal output is `0`. -/
theorem ideal_zero {{e out : Int}} (h : {ideal['module']}.generatedRoot GeneratedClaim.{ideal['param']} e out) :
    out = 0 := by
  obtain ⟨_, h⟩ := h
  simp only [{ideal['module']}.generatedRoot.body, {ideal['module']}.generatedRoot.constraints_0] at h
  rw [h, mul_zero]

/-- The decode stage's inputs in a run. -/
abbrev decodeInputs (external : GeneratedClaim.ExternalInputs) (execution : GeneratedClaim.Execution) :
    {NS}.Inputs :=
  ({dec_args})

/-- The backend's gadget matrix of the decode stage's gadget modulus. -/
theorem decode_gadget : ∃ G : ExactMatrix {QG} {N} 1 {MG}, gadgetMatrixRuns Backend.backend {BASE} {DIG} G :=
  ⟨_, Backend.{lay[0]}, by simp [Backend.backend], by decide, by decide, by decide, rfl⟩
{part_block}
set_option maxHeartbeats 0 in
/-- Every run decodes the ideal output. -/
theorem correct {{model : HashModel}} {{external : GeneratedClaim.ExternalInputs}}
    {{execution : GeneratedClaim.Execution}} (h : GeneratedClaim.Runs model external execution) :
    execution.{LHS} = execution.{RHS} := by
  have hrun := h
  obtain ⟨{pattern}⟩ := h{sec_haves}
  have F := GenRequirement.requirement_facts {req['name']} hreq
{correct_tail}

end RefreshTopProof.GenCorrect
'''
open(out_path, 'w').write(text)
print(f'wrote {out_path} ({"part of " + a.part if a.part else "placeholder part_terms"}; '
      f'{len(secrets)} secrets)', file=sys.stderr)
