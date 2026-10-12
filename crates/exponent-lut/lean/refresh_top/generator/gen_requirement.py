"""Write `RequirementFacts.lean`: the facts the requirement checks (written per check of
`generated/Requirement_0.lean`).

Usage: `python3 gen_requirement.py <parameter directory>`.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
from patterns import parse

d = sys.argv[1]
path = f'{d}/generated/Requirement_0.lean'
text = open(path).read()
_, names, atoms = parse(path, 'generatedRoot')
checks = []
for n, a in zip(names, atoms):
    m = re.search(r'sequential_generatedRoot_(\d+) params i \(current, \((w_\d+_0), \(\)\)\) next\) \(Int\.toNat \((\d+)\)\)', a)
    if m:
        checks.append((int(m.group(1)), m.group(2), int(m.group(3))))
out = ['''import TopRequirement

/-! The facts the requirement checks (written by `generator/gen_requirement.py`). -/

namespace RefreshTopProof

open Mxx.Primitives MxxRuntime MxxExponentLut
''']
for K, fam, M in checks:
    SZ = 5 * M
    out.append(f'''
theorem reqInner{K} {{i0 i : Nat}} {{cur next : Int}} {{F : Fin {SZ} → Int}}
    (h : Requirement_0.sequential_sequential_generatedRoot_{K}_5 GeneratedClaim.requirement_0_params i0 i
      (cur, F, Int.ofNat i0, ()) next) :
    ∃ hk : 5 * i0 + i < {SZ}, next = cur + F ⟨5 * i0 + i, hk⟩ := by
  unfold Requirement_0.sequential_sequential_generatedRoot_{K}_5 at h
  dsimp only at h
  obtain ⟨w7, h⟩ := h
  simp only [Requirement_0.sequential_sequential_generatedRoot_{K}_5.constraints_0] at h
  obtain ⟨h0, hlt, hg, hout⟩ := h
  obtain ⟨hk, rfl⟩ := familyGet_int hg h0 hlt
  simp only [Int.ofNat_eq_natCast] at hk ⊢
  refine ⟨by omega, ?_⟩
  rw [hout]
  congr 2
  apply Fin.ext
  show (Int.ofNat i0 * 5 + Int.ofNat i).toNat = 5 * i0 + i
  rw [Int.ofNat_eq_natCast, Int.ofNat_eq_natCast]
  omega

theorem reqCheck{K} {{i : Nat}} {{cur next : Bool}} {{F : Fin {SZ} → Int}}
    (h : Requirement_0.sequential_generatedRoot_{K} GeneratedClaim.requirement_0_params i (cur, F, ()) next)
    (hn : next = true) :
    cur = true ∧ ∀ hi : i < {M}, ∑ k : Fin 5, F (cellIndex (⟨i, hi⟩ : Fin {M}) k) = 1 := by
  obtain ⟨w, hb⟩ := h
  unfold Requirement_0.sequential_generatedRoot_{K}.body at hb
  dsimp only at hb
  simp only [Requirement_0.sequential_generatedRoot_{K}.constraints_0] at hb
  obtain ⟨-, hloop, hout⟩ := hb
  have hsum := IterRuns.sum_of_step (fun k ↦ if h : 5 * i + k < {SZ} then F ⟨5 * i + k, h⟩ else 0)
    (fun k c n _ hb ↦ by obtain ⟨hk, rfl⟩ := reqInner{K} hb; simp [hk]) hloop
  subst hn
  replace hout := hout.symm
  simp only [decide_eq_true_eq, ite_add_ite_eq_two] at hout
  refine ⟨by simpa using hout.1, fun hi ↦ ?_⟩
  rw [hout.2, show Int.toNat 5 = 5 from rfl, zero_add, Finset.sum_range] at hsum
  rw [hsum]
  refine Finset.sum_congr rfl fun k _ ↦ ?_
  rw [dif_pos (by omega)]
  rfl
''')
    pass
out.append('\nend RefreshTopProof\n')
open(f'{d}/RequirementFacts.lean', 'w').write('\n'.join(out))

# The root: every check holds.
from patterns import destruct_root
body = text[text.index('abbrev generatedRoot.body'):]
ins = re.findall(r'let (w_\d+_0) : ([^\n]*?) := inputs', body)
fam = {w: t for w, t in ins}
# The checks after the families are written for a requirement whose source exponent is `w_214_0`
# (`w_215_0 = 1 - w_214_0`); `shift` renumbers their wires and atoms for this requirement.
_e = int(re.search(r'let w_\d+_0 : Int := \(w_\d+_0 - w_(\d+)_0\)', body).group(1))
shift = lambda s: re.sub(r'\b(w_|h)(2[012]\d)(?=_0)', lambda m: m.group(1) + str(int(m.group(2)) + _e - 214), s)
binders = ' '.join(f'{{{w} : {t}}}' for w, t in ins)
tup = '(' + ', '.join(w for w, _ in ins) + ', ())'
facts = ['(∀ k : Fin 5, w_3_0 k = (labelExp k : Int) ∧ w_4_0 k * w_4_0 k = 1 ∧ w_5_0 k * 2 + (1 - w_4_0 k) * 8 = w_3_0 k * 4)']
for K, f, M in checks:
    facts.append(f'(∀ o : Fin {M}, ∑ k : Fin 5, {f} (cellIndex o k) = 1)')
facts.append(shift('w_212_0 = 1 - w_214_0'))
facts.append(shift('(∀ i : Fin 8, w_221_0 i = 1 - ((∑ k : Fin 5, w_15_0 (cellIndex i k) * w_3_0 k + ∑ k : Fin 5, w_17_0 (cellIndex i k) * w_3_0 k) % 8) / 4)'))
proof = [destruct_root(path, 'Requirement_0', 'generatedRoot', 'h')]
proof.append(shift('''  subst ht
  have a222 := IterRuns.all_of_step (fun i ↦ ∀ hi : i < 8, w_221_0 ⟨i, hi⟩ = 1 - ((∑ k : Fin 5, w_15_0 (cellIndex (⟨i, hi⟩ : Fin 8) k) * w_3_0 k + ∑ k : Fin 5, w_17_0 (cellIndex (⟨i, hi⟩ : Fin 8) k) * w_3_0 k) % 8) / 4)
    (fun i c n _ hb hn ↦ reqSigma hb hn) h222_0 hout.symm
  obtain ⟨h220, f222⟩ := a222
  simp only [w_220_0, w_218_0, w_217_0, w_216_0, w_215_0, w_213_0, w_211_0, w_219_0, decide_eq_true_eq,
    ite_add_ite_eq_two] at h220
  obtain ⟨h210, f212⟩ := h220'''))
prev = 'h210'
names_f = []
for K, f, M in reversed(checks):
    proof.append(f'''  obtain ⟨i{K}, f{K}⟩ := IterRuns.all_of_step (fun i ↦ ∀ hi : i < {M}, ∑ k : Fin 5, {f} (cellIndex (⟨i, hi⟩ : Fin {M}) k) = 1)
    (fun i c n _ hb hn ↦ reqCheck{K} hb hn) h{K}_0 (by simpa using {prev})''')
    prev = f'i{K}'
    names_f.append(K)
proof.append(f'''  obtain ⟨i6, f6⟩ := IterRuns.all_of_step (fun i ↦ ∀ hi : i < 5, w_3_0 ⟨i, hi⟩ = (labelExp ⟨i, hi⟩ : Int) ∧
      w_4_0 ⟨i, hi⟩ * w_4_0 ⟨i, hi⟩ = 1 ∧ w_5_0 ⟨i, hi⟩ * 2 + (1 - w_4_0 ⟨i, hi⟩) * 8 = w_3_0 ⟨i, hi⟩ * 4)
    (fun i c n _ hb hn ↦ reqLabel hb hn) h6_0 (by simpa using {prev})
  refine ⟨fun k ↦ f6 k.val (by simpa using k.isLt) k.isLt, ''' + ', '.join(
    f'fun o ↦ f{K} o.val (by simpa using o.isLt) o.isLt' for K, _, _ in checks) + ''', f212.symm ▸ rfl,
    fun i ↦ f222 i.val (by simpa using i.isLt) i.isLt⟩''')
root = f'''import ReqLabel
import ReqSigma

/-! The facts the requirement checks, from its run (written by `generator/gen_requirement.py`). -/

namespace RefreshTopProof

open Mxx.Primitives MxxRuntime MxxExponentLut

set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
theorem requirementFacts {binders} {{out : Bool}}
    (h : Requirement_0.generatedRoot GeneratedClaim.requirement_0_params {tup} out)
    (ht : out = true) :
    {' ∧ '.join(facts)} := by
''' + '\n'.join(proof) + '\n\nend RefreshTopProof\n'
open(f'{d}/RequirementRoot.lean', 'w').write(root)
