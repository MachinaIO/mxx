import PeNumeric
import Requirement_0
import ExponentIterate

/-!
# The requirement

A true requirement means: every attribute is a multiple of `8`, the tables of the first step
are below `8` and the last table below `2`, and the chain `v_(i+1) = t[256 i + v_i + x_(pos i)]` from
`v_0 = 0` ends at `v_2 = 0`.
-/

namespace PeProof

open Mxx.Primitives MxxRuntime MxxExponentLut

theorem familyGet_val {α : Type} {count : Nat} {f : Fin count → α} {i : Int} {o : α}
    (h : familyGetDynamic f i o) (hi0 : 0 ≤ i) (hi : i < count) : o = f ⟨i.toNat, by omega⟩ := by
  obtain ⟨q, hq, ho⟩ := h
  rw [ho]
  congr 1
  ext
  simp only
  omega

theorem iter_two {σ : Type} {body : Nat → σ → σ → Prop} {init final : σ}
    (h : MxxIR.IterRuns body (Int.toNat 2) init final) :
    ∃ middle, body 0 init middle ∧ body 1 middle final := by
  obtain ⟨c1, h1, b1⟩ := MxxIR.IterRuns.step_of_succ (count := 1) h
  obtain ⟨c0, h0, b0⟩ := MxxIR.IterRuns.step_of_succ (count := 0) h1
  cases h0
  exact ⟨c1, b0, b1⟩

theorem and_step {cur c : Bool}
    (h : decide ((if cur = true then (1 : Int) else 0) + (if c = true then 1 else 0) = 2) = true) :
    cur = true ∧ c = true := by
  cases cur <;> cases c <;> simp_all

set_option maxHeartbeats 4000000 in
theorem requirement_facts {tape : SampleTape} {x : Fin 2 → Int} {t : Fin 512 → Int} {pos : Fin 2 → Int}
    {out : Bool}
    (h : Requirement_0.generatedRoot tape [10] GeneratedClaim.requirement_0_params (x, t, pos, ()) out)
    (ht : out = true) :
    (∀ i, x i % 8 = 0) ∧ (∀ j : Fin 512, (j : Nat) < 256 → t j < 8) ∧
      (∀ j : Fin 512, 256 ≤ (j : Nat) → t j < 2) ∧
      ∃ v : Fin 3 → Int, v 0 = 0 ∧ v 2 = 0 ∧ ∀ i : Fin 2, ∃ (p : Fin 2) (e : Fin 512),
        pos i = ((p : Nat) : Int) ∧ ((e : Nat) : Int) = 256 * (i : Nat) + v i.castSucc + x p ∧
        v i.succ = t e := by
  obtain ⟨w, -, h2, -, h6, -, h13, -, h21, hout⟩ := h
  have hw : w.w_2_0 = true ∧ w.w_6_0 = true ∧ w.w_13_0 = true ∧ w.w_21_0 = 0 := by
    subst ht
    revert hout
    cases w.w_2_0 <;> cases w.w_6_0 <;> cases w.w_13_0 <;> by_cases hz : w.w_21_0 = 0 <;> simp [hz]
  obtain ⟨hw2, hw6, hw13, hw21⟩ := hw
  refine ⟨fun i ↦ ?_, fun j hj ↦ ?_, fun j hj ↦ ?_, ?_⟩
  · exact (MxxExponentLut.IterRuns.all_of_step (P := fun i ↦ ∀ hi : i < 2, x ⟨i, hi⟩ % 8 = 0)
      (fun i cur nxt hi hb hn ↦ by
        obtain ⟨w4, h0, h1, hg, -, hnx⟩ := hb
        rw [hnx] at hn
        obtain ⟨hc, hm⟩ := and_step hn
        refine ⟨hc, fun hi' ↦ ?_⟩
        have := of_decide_eq_true hm
        rw [familyGet_val hg h0 h1] at this
        exact this) h2 hw2).2 i i.isLt i.isLt
  · exact (MxxExponentLut.IterRuns.all_of_step (P := fun i ↦ ∀ hi : i < 256, t ⟨i, by omega⟩ < 8)
      (fun i cur nxt hi hb hn ↦ by
        obtain ⟨w4, h0, h1, hg, hnx⟩ := hb
        rw [hnx] at hn
        obtain ⟨hc, hm⟩ := and_step hn
        refine ⟨hc, fun hi' ↦ ?_⟩
        have := of_decide_eq_true hm
        rw [familyGet_val hg h0 h1] at this
        exact this) h6 hw6).2 j hj hj
  · have := (MxxExponentLut.IterRuns.all_of_step (P := fun i ↦ ∀ hi : 256 + i < 512, t ⟨256 + i, hi⟩ < 2)
      (fun i cur nxt hi hb hn ↦ by
        obtain ⟨w6, h0, h1, hg, hnx⟩ := hb
        rw [hnx] at hn
        obtain ⟨hc, hm⟩ := and_step hn
        refine ⟨hc, fun hi' ↦ ?_⟩
        have := of_decide_eq_true hm
        rw [familyGet_val hg h0 h1] at this
        exact this) h13 hw13).2 (j - 256) (by omega) (by omega)
    have e : (⟨256 + ((j : Nat) - 256), by omega⟩ : Fin 512) = j := Fin.ext (by simp only; omega)
    rw [e] at this
    exact this
  · obtain ⟨s1, b0, b1⟩ := iter_two h21
    have step : ∀ (i : Nat) (hi : i < 2) (cur nxt : Int),
        Requirement_0.sequential_generatedRoot_21 tape ([10] ++ [21, i]) GeneratedClaim.requirement_0_params i
          (cur, (t, (x, (pos, ())))) nxt →
        ∃ (p : Fin 2) (e : Fin 512), pos ⟨i, hi⟩ = ((p : Nat) : Int) ∧
          ((e : Nat) : Int) = 256 * i + cur + x p ∧ nxt = t e := by
      intro i hi cur nxt hb
      obtain ⟨w7, w8, w11, h0, h1, hg7, h7a, h7b, hg8, h10a, h10b, hg11, hnx⟩ := hb
      have e7 := familyGet_val hg7 h0 h1
      have e8 := familyGet_val hg8 h7a h7b
      have e11 := familyGet_val hg11 h10a h10b
      simp only at e7 e8 e11 h10a h10b
      have hoi : Int.ofNat i = (i : Int) := rfl
      refine ⟨⟨w7.toNat, by omega⟩, ⟨(Int.ofNat i * 256 + (cur + w8)).toNat, by omega⟩, ?_, ?_, ?_⟩
      · rw [show (⟨i, hi⟩ : Fin 2) = ⟨(Int.ofNat i).toNat, by omega⟩ from rfl, ← e7]
        simp only
        omega
      · simp only
        rw [← e8]
        omega
      · rw [hnx, e11]
    refine ⟨![0, s1, w.w_21_0], rfl, hw21, fun i ↦ ?_⟩
    fin_cases i
    · exact step 0 (by norm_num) _ _ b0
    · exact step 1 (by norm_num) _ _ b1

end PeProof
