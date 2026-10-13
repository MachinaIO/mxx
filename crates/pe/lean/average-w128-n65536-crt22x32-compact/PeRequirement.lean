import PeNumeric
import Requirement_0
import ExponentIterate

/-!
# The requirement

A true requirement means: every attribute is a multiple of `8`, the tables of the first three steps
are below `8` and the last table below `2`, and the chain `v_(i+1) = t[128 i + v_i + x_(pos i)]` from
`v_0 = 0` ends at `v_4 = 0`.
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

theorem iter_four {σ : Type} {body : Nat → σ → σ → Prop} {init final : σ}
    (h : MxxIR.IterRuns body (Int.toNat 4) init final) :
    ∃ s1 s2 s3, body 0 init s1 ∧ body 1 s1 s2 ∧ body 2 s2 s3 ∧ body 3 s3 final := by
  obtain ⟨c3, h3, b3⟩ := MxxIR.IterRuns.step_of_succ (count := 3) h
  obtain ⟨c2, h2, b2⟩ := MxxIR.IterRuns.step_of_succ (count := 2) h3
  obtain ⟨c1, h1, b1⟩ := MxxIR.IterRuns.step_of_succ (count := 1) h2
  obtain ⟨c0, h0, b0⟩ := MxxIR.IterRuns.step_of_succ (count := 0) h1
  cases h0
  exact ⟨c1, c2, c3, b0, b1, b2, b3⟩

theorem and_step {cur c : Bool}
    (h : decide ((if cur = true then (1 : Int) else 0) + (if c = true then 1 else 0) = 2) = true) :
    cur = true ∧ c = true := by
  cases cur <;> cases c <;> simp_all

set_option maxHeartbeats 4000000 in
theorem requirement_facts {tape : SampleTape} {x : Fin 4 → Int} {t : Fin 512 → Int} {pos : Fin 4 → Int}
    {out : Bool}
    (h : Requirement_0.generatedRoot tape [9] GeneratedClaim.requirement_0_params (x, t, pos, ()) out)
    (ht : out = true) :
    (∀ i, x i % 8 = 0) ∧ (∀ j : Fin 512, (j : Nat) < 384 → t j < 8) ∧
      (∀ j : Fin 512, 384 ≤ (j : Nat) → t j < 2) ∧
      ∃ v : Fin 5 → Int, v 0 = 0 ∧ v 4 = 0 ∧ ∀ i : Fin 4, ∃ (p : Fin 4) (e : Fin 512),
        pos i = ((p : Nat) : Int) ∧ ((e : Nat) : Int) = 128 * (i : Nat) + v i.castSucc + x p ∧
        v i.succ = t e := by
  obtain ⟨w, -, h2, -, h6, -, h13, -, h21, hout⟩ := h
  have hw : w.w_2_0 = true ∧ w.w_6_0 = true ∧ w.w_13_0 = true ∧ w.w_21_0 = 0 := by
    subst ht
    revert hout
    cases w.w_2_0 <;> cases w.w_6_0 <;> cases w.w_13_0 <;> by_cases hz : w.w_21_0 = 0 <;> simp [hz]
  obtain ⟨hw2, hw6, hw13, hw21⟩ := hw
  refine ⟨fun i ↦ ?_, fun j hj ↦ ?_, fun j hj ↦ ?_, ?_⟩
  · exact (MxxExponentLut.IterRuns.all_of_step (P := fun i ↦ ∀ hi : i < 4, x ⟨i, hi⟩ % 8 = 0)
      (fun i cur nxt hi hb hn ↦ by
        obtain ⟨w4, h0, h1, hg, -, hnx⟩ := hb
        rw [hnx] at hn
        obtain ⟨hc, hm⟩ := and_step hn
        refine ⟨hc, fun hi' ↦ ?_⟩
        have := of_decide_eq_true hm
        rw [familyGet_val hg h0 h1] at this
        exact this) h2 hw2).2 i i.isLt i.isLt
  · exact (MxxExponentLut.IterRuns.all_of_step (P := fun i ↦ ∀ hi : i < 512, t ⟨i, hi⟩ < 8)
      (fun i cur nxt hi hb hn ↦ by
        obtain ⟨w4, h0, h1, hg, hnx⟩ := hb
        rw [hnx] at hn
        obtain ⟨hc, hm⟩ := and_step hn
        refine ⟨hc, fun hi' ↦ ?_⟩
        have := of_decide_eq_true hm
        rw [familyGet_val hg h0 h1] at this
        exact this) h6 hw6).2 j hj j.isLt
  · have := (MxxExponentLut.IterRuns.all_of_step (P := fun i ↦ ∀ hi : 384 + i < 512, t ⟨384 + i, hi⟩ < 2)
      (fun i cur nxt hi hb hn ↦ by
        obtain ⟨w6, h0, h1, hg, hnx⟩ := hb
        rw [hnx] at hn
        obtain ⟨hc, hm⟩ := and_step hn
        refine ⟨hc, fun hi' ↦ ?_⟩
        have := of_decide_eq_true hm
        rw [familyGet_val hg h0 h1] at this
        exact this) h13 hw13).2 (j - 384) (by omega) (by omega)
    have e : (⟨384 + ((j : Nat) - 384), by omega⟩ : Fin 512) = j := Fin.ext (by simp only; omega)
    rw [e] at this
    exact this
  · obtain ⟨s1, s2, s3, b0, b1, b2, b3⟩ := iter_four h21
    have step : ∀ (i : Nat) (hi : i < 4) (cur nxt : Int),
        Requirement_0.sequential_generatedRoot_21 tape ([9] ++ [21, i]) GeneratedClaim.requirement_0_params i
          (cur, (t, (x, (pos, ())))) nxt →
        ∃ (p : Fin 4) (e : Fin 512), pos ⟨i, hi⟩ = ((p : Nat) : Int) ∧
          ((e : Nat) : Int) = 128 * i + cur + x p ∧ nxt = t e := by
      intro i hi cur nxt hb
      obtain ⟨w7, w8, w11, h0, h1, hg7, h7a, h7b, hg8, h10a, h10b, hg11, hnx⟩ := hb
      have e7 := familyGet_val hg7 h0 h1
      have e8 := familyGet_val hg8 h7a h7b
      have e11 := familyGet_val hg11 h10a h10b
      simp only at e7 e8 e11 h10a h10b
      have hoi : Int.ofNat i = (i : Int) := rfl
      refine ⟨⟨w7.toNat, by omega⟩, ⟨(Int.ofNat i * 128 + (cur + w8)).toNat, by omega⟩, ?_, ?_, ?_⟩
      · rw [show (⟨i, hi⟩ : Fin 4) = ⟨(Int.ofNat i).toNat, by omega⟩ from rfl, ← e7]
        simp only
        omega
      · simp only
        rw [← e8]
        omega
      · rw [hnx, e11]
    refine ⟨![0, s1, s2, s3, w.w_21_0], rfl, hw21, fun i ↦ ?_⟩
    fin_cases i
    · exact step 0 (by norm_num) _ _ b0
    · exact step 1 (by norm_num) _ _ b1
    · exact step 2 (by norm_num) _ _ b2
    · exact step 3 (by norm_num) _ _ b3

end PeProof
