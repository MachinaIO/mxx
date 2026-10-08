import PeKeygen

/-!
# Decryption

The decode stage contracts the family row of the ciphertext with the key and rounds the phase:
`c_m - d₂ - (σ₁((b - B²) k̃₀) + σ_K((b - B²) k̃₁) + Σᵢ (σᵢ(h - H¹) + [0 | g]) vᵢ + (c_F - A_F¹) v_F)`.
-/

namespace PeProof

open Mxx.Primitives MxxRuntime MxxExponentLut

/-- The joined row `[0 | g]`. -/
noncomputable def zeroG (g : ExactMatrix Q N 1 10) : ExactMatrix Q N 1 20 := joinRow ![0, g]

abbrev HelperState := ExactMatrix Q N 1 1

set_option maxHeartbeats 64000000 in
/-- One helper block `i` of the decode loop adds `(σᵢ(h - H¹) + [0 | g]) vᵢ`. -/
theorem decode_helper_step {model : HashModel} {tape : SampleTape} {path : List Nat} {key : ByteArray}
    {hf0 hf1 : Fin 1 → ExactMatrix Q N 1 10} {gO : ExactMatrix Q N 1 20}
    {v : Fin 2 → ExactMatrix Q N 20 1} {i : Nat} (hi : i < 2) {s t : HelperState}
    (h : Stage_decode.sequential_generatedRoot_23 Backend.backend model tape path
      GeneratedClaim.stage_8_params i (s, hf0, hf1, key, gO, v, ()) t) :
    let D := joinRow ![hf0 0, hf1 0] - joinRow (helperKey model key 1)
    (i = 0 → t = s + (D + gO) * v 0) ∧ (i = 1 → t = s + (substituteMatrix K hK D + gO) * v 1) := by
  intro D
  obtain ⟨w3, w5, w6, w8, w9, w10, w12, w17, -, -, -, x3, -, -, x5, x6, -, -, -, x8, -, -, -, x9, x10,
    -, -, -, x12, -, -, x17, xout⟩ := h
  have hmod : Int.fmod (Int.ofNat i) 1 = Int.ofNat 0 := by
    rw [Int.fmod_eq_emod_of_nonneg _ (by omega)]
    simp
  have hdiv : Int.fdiv (Int.ofNat i) 1 = Int.ofNat i := by
    rw [Int.fdiv_eq_ediv_of_nonneg _ (by omega)]
    simp
  rw [hmod] at x3 x5 x8 x9
  rw [hdiv] at x12
  have hs : MxxIR.selectInt (Int.ofNat 0) [8193] = (K : Int) := by decide
  rw [hs] at x8 x9
  obtain ⟨_, hw3⟩ := familyGet_eq x3
  obtain ⟨_, hw5⟩ := familyGet_eq x5
  obtain ⟨_, hw17⟩ := familyGet_eq x17
  have hw6 : w6 = joinRow ![hf0 0, hf1 0] := by
    rw [hw3, hw5] at x6
    exact concat_joinRow (r := ![hf0 0, hf1 0]) x6
  have hw10 : w10 = joinRow (helperKey model key 1) := by
    have hkey : ∀ j, helperKey model key 1 j = ![w8, w9] j := by
      intro j
      fin_cases j
      · exact (hashSample_eq x8).symm
      · exact (hashSample_eq x9).symm
    rw [concat_joinRow (r := ![w8, w9]) x10]
    congr 1
    funext j
    exact (hkey j).symm
  subst xout
  refine ⟨fun h0 ↦ ?_, fun h1 ↦ ?_⟩
  · subst h0
    rw [show MxxIR.selectInt (Int.ofNat 0) [1, 8193] = 1 from by decide] at x12
    have hw12 := routing_one x12
    simp only [matrixAdd, matrixMul, matrixSub, hw12, hw6, hw10, hw17, D]
    rfl
  · subst h1
    rw [show MxxIR.selectInt (Int.ofNat 1) [1, 8193] = (K : Int) from by decide] at x12
    have hw12 := routing_K' x12
    simp only [matrixAdd, matrixMul, matrixSub, hw12, hw6, hw10, hw17, D]
    rfl

abbrev DecodeInputs := ExactMatrix Q N 1 1 × ByteArray × ExactMatrix Q N 1 12 × ExactMatrix Q N 12 1 ×
  ExactMatrix Q N 12 1 × (Fin 1 → ExactMatrix Q N 1 10) × (Fin 1 → ExactMatrix Q N 1 10) ×
  (Fin 2 → ExactMatrix Q N 20 1) × ExactMatrix Q N 1 10 × ExactMatrix Q N 1 10 × ExactMatrix Q N 1 10 ×
  ExactMatrix Q N 1 10 × ExactMatrix Q N 20 1 × Unit

set_option maxHeartbeats 64000000 in
/-- The decode stage rounds every coefficient of the phase. -/
theorem decode_explicit {model : HashModel} {tape : SampleTape} {path : List Nat}
    {cm : ExactMatrix Q N 1 1} {key : ByteArray} {b : ExactMatrix Q N 1 12} {k0 k1 : ExactMatrix Q N 12 1}
    {hf0 hf1 : Fin 1 → ExactMatrix Q N 1 10} {v : Fin 2 → ExactMatrix Q N 20 1}
    {c0 c1 a0 a1 : ExactMatrix Q N 1 10} {vF : ExactMatrix Q N 20 1} {out : Fin N → Int}
    (h : Stage_decode.generatedRoot Backend.backend model tape path GeneratedClaim.stage_8_params
      (cm, key, b, k0, k1, hf0, hf1, v, c0, c1, a0, a1, vF, ()) out)
    {g : ExactMatrix Q N 1 10} (hg : gadgetMatrixRuns Backend.backend 32768 10 g) :
    let W := b - anchorSecond model key
    let D := joinRow ![hf0 0, hf1 0] - joinRow (helperKey model key 1)
    let phase := cm - messageRow model key 1 -
      (W * k0 + substituteMatrix K hK (W * k1) +
        ((D + zeroG g) * v 0 + (substituteMatrix K hK D + zeroG g) * v 1) +
        (joinRow ![c0, c1] - joinRow ![a0, a1]) * vF)
    ∀ k : Fin N, thresholdDecode 2 8192 (k.val : Int) phase (out k) := by
  intro W D phase
  obtain ⟨w, h2, h6, h10, h14, h20, h21, -, h23, h27, h30, -, h36⟩ := h
  have hout := (h36 0).2
  subst hout
  have hw2 : w.w_2_0 = messageRow model key 1 := hashSample_eq h2
  have hw6 : w.w_6_0 = anchorSecond model key := hashSample_eq h6
  have hw10 := routing_one h10
  have hw14 := routing_K' h14
  have hg20 : w.w_20_0 = g := gadgetMatrixRuns_unique h20 hg
  have hw21 : w.w_21_0 = zeroG g := by
    rw [concat_joinRow (r := ![0, w.w_20_0]) h21, hg20]
    rfl
  have hw27 : w.w_27_0 = joinRow ![c0, c1] := concat_joinRow (r := ![c0, c1]) h27
  have hw30 : w.w_30_0 = joinRow ![a0, a1] := concat_joinRow (r := ![a0, a1]) h30
  -- The two helper blocks.
  rw [hw21] at h23
  have hcount : Int.toNat 2 = 1 + 1 := rfl
  rw [hcount] at h23
  obtain ⟨s1, hs1, hb1⟩ := MxxIR.IterRuns.step_of_succ h23
  obtain ⟨s0, hs0, hb0⟩ := MxxIR.IterRuns.step_of_succ hs1
  cases hs0
  have h0 := (decode_helper_step (by omega) hb0).1 rfl
  have h1 := (decode_helper_step (by omega) hb1).2 rfl
  have hw23 : w.w_23_0 = (D + zeroG g) * v 0 + (substituteMatrix K hK D + zeroG g) * v 1 := by
    rw [h1, h0]
    simp only [zero_add]
    rfl
  intro k
  have hk := (h36 k).1
  convert hk using 2
  simp only [phase, W, matrixSub, matrixAdd, matrixMul, hw2, hw6, hw10, hw14, hw23, hw27, hw30]
  abel

end PeProof
