"""LwrInputs.lean: the requirement and the ideal output, computed in loops."""
from info import wrap
from lean_patterns import pattern


def inputs(I):
    B, C, M, N, W, s = I.B, I.C, I.M, I.n, I.W, I.stride
    o = [f'''import LwrSetup

/-!
# The requirement and the ideal output

The requirement checks every member of `a` in one loop and sums every block's selectors in a loop;
it accepts plaintext monomials `c X^e` of `a` with `c X^e = X^({s}a)` and one-hot selectors. The ideal
output sums `s_b(i) a(coords b i)` over the cells of every block in a loop, and is the parity of the
readout `τ(z mod {W})` of the total `z`.
-/

namespace LwrProof

open Mxx.Primitives MxxRuntime MxxExponentLut

set_option maxRecDepth 100000

variable {{tape : SampleTape}} {{path : List Nat}}

theorem ite_add_ite_eq_two (p q : Prop) [Decidable p] [Decidable q] :
    ((if p then (1 : Int) else 0) + (if q then 1 else 0) = 2) ↔ p ∧ q := by
  by_cases hp : p <;> by_cases hq : q <;> simp [hp, hq]

/-- One iteration of the member loop keeps a true state only for an accepted member. -/
theorem member_step {{ac ae a : Fin {M} → Int}} {{i : Nat}} {{current next : Bool}}
    (h : Requirement_0.sequential_generatedRoot_6 tape path GeneratedClaim.requirement_0_params i
      (current, ac, ae, a, ()) next) (hnext : next = true) :
    current = true ∧ ∀ hi : i < {M}, ac ⟨i, hi⟩ * ac ⟨i, hi⟩ = 1 ∧
      ae ⟨i, hi⟩ * 2 + (1 - ac ⟨i, hi⟩) * {N} = a ⟨i, hi⟩ * {2 * s} := by
  unfold Requirement_0.sequential_generatedRoot_6 at h
  dsimp only at h
  obtain ⟨w4, w14, w23, -, -, h4, -, -, h14, -, -, h23, hout⟩ := h
  obtain ⟨hi, rfl⟩ := familyGet_eq h4
  obtain ⟨hi14, rfl⟩ := familyGet_eq h14
  obtain ⟨hi23, rfl⟩ := familyGet_eq h23
  subst hnext
  replace hout := hout.symm
  simp only [decide_eq_true_eq, ite_add_ite_eq_two] at hout
  obtain ⟨⟨hc, hm⟩, hr⟩ := hout
  refine ⟨by simpa using hc, fun _ ↦ ⟨hm, ?_⟩⟩
  linarith
''']
    for b in range(B):
        d = 10 + 10 * b
        o.append(f'''
/-- One iteration of the block-{b} selector loop adds the selector of cell `i`. -/
theorem selector_step{b} {{s : Fin {C} → Int}} {{i : Nat}} {{current next : Int}}
    (h : Requirement_0.sequential_generatedRoot_{d} tape path GeneratedClaim.requirement_0_params i
      (current, s, ()) next) :
    ∃ hi : i < {C}, next = current + s ⟨i, hi⟩ := by
  unfold Requirement_0.sequential_generatedRoot_{d} at h
  dsimp only at h
  obtain ⟨w3, -, -, h3, hout⟩ := h
  obtain ⟨hi, rfl⟩ := familyGet_eq h3
  exact ⟨hi, hout⟩
''')
    for b in range(B):
        d = 4 + 4 * b
        o.append(f'''
/-- One iteration of the block-{b} sum of the ideal adds `s(i) a(coords {b} i)`. -/
theorem ideal_step{b} {{s : Fin {C} → Int}} {{a : Fin {M} → Int}} {{i : Nat}} {{current next : Int}}
    (h : Ideal.sequential_generatedRoot_{d} tape path GeneratedClaim.ideal_params i
      (current, s, a, ()) next) :
    ∃ hi : i < {C}, next = current + s ⟨i, hi⟩ * a ⟨coords {b} ⟨i, hi⟩, coords_lt {b} ⟨i, hi⟩⟩ := by
  unfold Ideal.sequential_generatedRoot_{d} at h
  dsimp only at h
  obtain ⟨w3, w{C * 3 + 5}, w{C * 3 + 6}, -, -, h3, -, -, -, hsel, -, -, h{C * 3 + 6}, hout⟩ := h
  obtain ⟨hi, rfl⟩ := familyGet_eq h3
  have hc : w{C * 3 + 5} = Int.ofNat (coords {b} ⟨i, hi⟩) :=
    (select_eq_selectInt hsel).trans (selectInt_map_ofNat (coordsTable {b}) i)
  subst hc
  obtain ⟨hi', rfl⟩ := familyGet_eq h{C * 3 + 6}
  exact ⟨hi, hout⟩
''')
    sels = wrap([f's{b}' for b in range(B)], ' ', 6, 30)
    sel_tuple = wrap([f's{b}' for b in range(B)], ', ', 6, 20)
    sums = wrap([f'∑ i, s{b} i = 1' for b in range(B)], ' ∧ ', 6, 6)
    o.append(f'''
/-- A selector loop sums its family. -/
theorem selector_sum {{s : Fin {C} → Int}} {{final : Int}} {{body : Nat → Int → Int → Prop}}
    (hstep : ∀ i current next, body i current next → ∃ hi : i < {C}, next = current + s ⟨i, hi⟩)
    (h : MxxIR.IterRuns body (Int.toNat {C}) 0 final) : final = ∑ i, s i := by
  rw [IterRuns.sum_of_step (fun i ↦ if h : i < {C} then s ⟨i, h⟩ else 0) (fun i current next _ hb ↦ by
    obtain ⟨hi, rfl⟩ := hstep i current next hb
    simp [hi]) h, zero_add]
  exact sum_range_fin s

/-- An ideal block loop sums `s(i) a(coords b i)`. -/
theorem block_sum {{s : Fin {C} → Int}} {{a : Fin {M} → Int}} (b : Fin {B}) {{final : Int}}
    {{body : Nat → Int → Int → Prop}}
    (hstep : ∀ i current next, body i current next →
      ∃ hi : i < {C}, next = current + s ⟨i, hi⟩ * a ⟨coords b ⟨i, hi⟩, coords_lt b ⟨i, hi⟩⟩)
    (h : MxxIR.IterRuns body (Int.toNat {C}) 0 final) :
    final = ∑ i, s i * a ⟨coords b i, coords_lt b i⟩ := by
  rw [IterRuns.sum_of_step (fun i ↦ if h : i < {C} then s ⟨i, h⟩ * a ⟨coords b ⟨i, h⟩,
      coords_lt b ⟨i, h⟩⟩ else 0) (fun i current next _ hb ↦ by
    obtain ⟨hi, rfl⟩ := hstep i current next hb
    simp [hi]) h, zero_add]
  exact sum_range_fin (fun i ↦ s i * a ⟨coords b i, coords_lt b i⟩)
''')
    # The requirement root.
    rp = pattern(f'{I.d}/generated/Requirement_0.lean', 'generatedRoot')[1:-1].split(', ')
    loops = [x for x in rp if x not in ('-', 'hout')]
    assert loops == ['h6_0'] + [f'h{10 + 10 * b}_0' for b in range(B)], loops
    pat = wrap(['hm' if x == 'h6_0' else x for x in rp], ', ', 4, 10)
    nest = 'hw6'
    for b in range(B):
        nest = f'⟨{nest}, e{10 + 10 * b}⟩'
    goals = '\n'.join(
        f'  · rw [← selector_sum (fun i c n hb ↦ selector_step{b} hb) h{10 + 10 * b}_0]; exact e{10 + 10 * b}'
        for b in range(B))
    o.append(f'''
set_option maxHeartbeats 16000000 in
set_option maxRecDepth 100000 in
/-- The requirement: `c_k² = 1` and `2 e_k + {N} (1 - c_k) = {2 * s} a_k` for every member, and every
block's selectors sum to one. -/
theorem requirement_holds {{ac ae a : Fin {M} → Int}} {{{sels} : Fin {C} → Int}} {{out : Bool}}
    (h : Requirement_0.generatedRoot tape path GeneratedClaim.requirement_0_params
      (ac, ae, a, {sel_tuple}, ()) out) (ht : out = true) :
    (∀ k : Fin {M}, ac k * ac k = 1 ∧ ae k * 2 + (1 - ac k) * {N} = a k * {2 * s}) ∧
      {sums} := by
  obtain ⟨w, hb⟩ := h
  unfold Requirement_0.generatedRoot.body at hb
  dsimp only at hb
  obtain ⟨{pat}⟩ := hb
  subst ht
  replace hout := hout.symm
  simp only [decide_eq_true_eq, ite_add_ite_eq_two] at hout
  obtain {nest} := hout
  have hmem := (IterRuns.all_of_step (fun i ↦ ∀ hi : i < {M}, ac ⟨i, hi⟩ * ac ⟨i, hi⟩ = 1 ∧
      ae ⟨i, hi⟩ * 2 + (1 - ac ⟨i, hi⟩) * {N} = a ⟨i, hi⟩ * {2 * s})
    (fun i current next _ hb hn ↦ member_step hb hn) hm (by simpa using hw6)).2
  refine ⟨fun k ↦ hmem k.val (lt_of_lt_of_eq k.isLt rfl) k.isLt,
    {wrap(['?_'] * B, ', ', 4, 4)}⟩
{goals}
''')
    # The ideal root.
    ip = pattern(f'{I.d}/generated/Ideal.lean', 'generatedRoot')[1:-1].split(', ')
    loops = [x for x in ip if x not in ('-', 'hout')]
    assert loops[:B] == [f'h{4 + 4 * b}_0' for b in range(B)] and len(loops) == B + 3, loops
    ideal_tuple = wrap(['s0', 'a'] + [f's{b}' for b in range(1, B)], ', ', 6, 70)
    total = wrap(['0'] + [f'∑ i, s{b} i * a ⟨coords {b} i, coords_lt {b} i⟩' for b in range(B)], ' + ', 4, 2)
    rws = wrap([f'block_sum {b} (fun i c n hb ↦ ideal_step{b} hb) h{4 + 4 * b}_0' for b in range(B)], ', ', 4, 13)
    o.append(f'''
/-- The total `z = Σ_b Σ_i s_b(i) a(coords b i)` of the selected coordinates. -/
def selectedSum (a : Fin {M} → Int) ({sels} : Fin {C} → Int) : Int :=
  {total}

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 100000 in
/-- The ideal output: the parity of `τ(z mod {W}) = ⌊{I.pprime} (z mod {W}) / {W}⌋ + 1`. -/
theorem ideal_value {{a : Fin {M} → Int}} {{{sels} : Fin {C} → Int}} {{out : Int}}
    (h : Ideal.generatedRoot tape path GeneratedClaim.ideal_params ({ideal_tuple}, ()) out) :
    out = (selectedSum a {sels} % {W} * {I.pprime} / {W} + 1) % 2 := by
  obtain ⟨w, hb⟩ := h
  unfold Ideal.generatedRoot.body at hb
  dsimp only at hb
  obtain ⟨{wrap(ip, ', ', 4, 10)}⟩ := hb
  rw [hout, {rws}]
  rfl

end LwrProof
''')
    return ''.join(o)
