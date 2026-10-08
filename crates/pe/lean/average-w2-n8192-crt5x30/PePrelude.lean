import PeEncrypt

/-!
# The encryption prelude

Under the secret `s`, the anchor encoding `b = s B + B² + e_b`, the attribute ciphertext
`C = P Z + X^x G` for `P = [B; B² - b]`, and the message component
`c_m = s d₁ + d₂ + e_m + ⌊Q/2⌋ m`. Every sample is named by its tape site.
-/

namespace PeProof

open Mxx.Primitives MxxRuntime MxxExponentLut

/-- The law of the prelude samples. -/
abbrev preludeLaw : SamplerLaw := .gaussian (319 / 100) 30

/-- The anchor error `e_b`. -/
noncomputable def ebAt (tape : SampleTape) (path : List Nat) : ErrorMatrix N 1 12 :=
  tapeWitness tape (path ++ [8]) preludeLaw

/-- The attribute matrix `Z`. -/
noncomputable def zAt (tape : SampleTape) (path : List Nat) : ErrorMatrix N 12 20 :=
  tapeWitness tape (path ++ [11, 0, 1]) preludeLaw

/-- The message error `e_m`. -/
noncomputable def emAt (tape : SampleTape) (path : List Nat) : ErrorMatrix N 1 1 :=
  tapeWitness tape (path ++ [16]) preludeLaw

/-- `⌊Q/2⌋`. -/
abbrev halfQ : Int := 712851031313528105616668575764770600691113984

abbrev PreludeOutputs := (Fin 1 → ExactMatrix Q N 1 10) × (Fin 1 → ExactMatrix Q N 1 10) ×
  (Fin 1 → ExactMatrix Q N 1 10) × (Fin 1 → ExactMatrix Q N 1 10) × ExactMatrix Q N 1 12 ×
  ExactMatrix Q N 1 1 × Unit

set_option maxHeartbeats 16000000 in
theorem prelude_explicit {model : HashModel} {tape : SampleTape} {path : List Nat}
    {B : ExactMatrix Q N 1 12} {x : Fin 1 → Int} {key : ByteArray} {secret : ExactMatrix Q N 1 1}
    {m : Fin N → Int} {outputs : PreludeOutputs}
    (h : Stage_prelude.generatedRoot Backend.backend model tape path GeneratedClaim.stage_3_params
      (B, x, key, secret, m, ()) outputs)
    (hx : x 0 = 0) {sm : ErrorPoly N} (hsecret : secret = fun _ _ ↦ reducePoly Q N sm)
    {g : ExactMatrix Q N 1 10} (hg : gadgetMatrixRuns Backend.backend 32768 10 g) :
    let Z := reduceMatrix Q N 12 20 (zAt tape path)
    let b := outputs.2.2.2.2.1
    b = reducePoly Q N sm • B + anchorSecond model key + reduceMatrix Q N 1 12 (ebAt tape path) ∧
    outputs.2.2.2.2.2.1 = reducePoly Q N sm • messageRow model key 0 + messageRow model key 1 +
      reduceMatrix Q N 1 1 (emAt tape path) +
      ((halfQ : ExactPoly Q N) • (fun _ _ ↦ polynomialOfCoefficients m) : ExactMatrix Q N 1 1) ∧
    outputs.1 0 = rowWindow 10 0 (by omega) (B * Z) + g ∧
    outputs.2.1 0 = rowWindow 10 10 (by omega) (B * Z) ∧
    outputs.2.2.1 0 = rowWindow 10 0 (by omega) ((anchorSecond model key - b) * Z) ∧
    outputs.2.2.2.1 0 = rowWindow 10 10 (by omega) ((anchorSecond model key - b) * Z) + g ∧
    CoeffBound (ebAt tape path) 30 ∧ CoeffBound (zAt tape path) 30 ∧ CoeffBound (emAt tape path) 30 := by
  intro Z b
  obtain ⟨w, h2, h4, h8, -, h11, h12, h14, h16, h20, hout⟩ := h
  have hg2 : w.w_2_0 = g := gadgetMatrixRuns_unique h2 hg
  obtain ⟨he8, hb8⟩ := gaussianSampleAt_explicit hN h8
  obtain ⟨he16, hb16⟩ := gaussianSampleAt_explicit hN h16
  have hw4 : w.w_4_0 = anchorSecond model key := hashSample_eq h4
  have hw12 : w.w_12_0 = messageRow model key 0 := hashSample_eq h12
  have hw14 : w.w_14_0 = messageRow model key 1 := hashSample_eq h14
  have hw20 : w.w_20_0 = fun _ _ ↦ polynomialOfCoefficients m := h20
  obtain ⟨w1, w3, w10, w13, w14, x1, -, -, -, -, -, -, x3, -, -, -, -, -, -, x10, -, -, -, -, -, -, x13,
    -, -, -, -, -, -, x14, xout⟩ := h11 0
  obtain ⟨hz, hbz⟩ := gaussianSampleAt_explicit hN x1
  subst hout
  have hb : b = reducePoly Q N sm • B + anchorSecond model key + reduceMatrix Q N 1 12 (ebAt tape path) := by
    simp only [b, matrixAdd, matrixMulScalarLeft_eq, hsecret, hw4, he8, ebAt]
  have hmono : (multiplyMonomial (1 : ExactMatrix Q N 1 1) (x 0) : ExactMatrix Q N 1 1) = 1 := by
    funext a c
    fin_cases a; fin_cases c
    simp [multiplyMonomial, hx]
  have hb8' : CoeffBound (ebAt tape path) 30 := by
    intro r c k
    have := hb8 r c k
    simpa [ebAt] using this
  have hbz' : CoeffBound (zAt tape path) 30 := by
    intro r c k
    have := hbz r c k
    simpa [zAt] using this
  have hb16' : CoeffBound (emAt tape path) 30 := by
    intro r c k
    have := hb16 r c k
    simpa [emAt] using this
  refine ⟨hb, ?_, ?_, ?_, ?_, ?_, hb8', hbz', hb16'⟩
  · simp only [matrixAdd, matrixMulScalarLeft_eq, hsecret, hw12, hw14, he16, emAt, hw20,
      matrixPolynomial_single]
  · simp only [Prod.mk.injEq] at xout
    obtain ⟨ho0, -⟩ := xout
    have hw3 := sliceMatrix_window 0 x3 (by simp) (by omega)
    show w.w_11_0 0 = _
    rw [ho0, hw3]
    funext r c
    simp [matrixAdd, matrixMul, hmono, hg2, Z, zAt, hz, rowWindow]
  · simp only [Prod.mk.injEq] at xout
    obtain ⟨-, ho1, -⟩ := xout
    have hw10 := sliceMatrix_window 10 x10 (by simp) (by omega)
    show w.w_11_1 0 = _
    rw [ho1, hw10]
    simp [matrixMul, Z, zAt, hz]
  · simp only [Prod.mk.injEq] at xout
    obtain ⟨-, -, ho2, -⟩ := xout
    have hw13 := sliceMatrix_window 0 x13 (by simp) (by omega)
    show w.w_11_2 0 = _
    rw [ho2, hw13]
    simp [matrixMul, matrixSub, Z, zAt, hz, hw4, b]
  · simp only [Prod.mk.injEq] at xout
    obtain ⟨-, -, -, ho3, -⟩ := xout
    have hw14' := sliceMatrix_window 10 x14 (by simp) (by omega)
    show w.w_11_3 0 = _
    rw [ho3, hw14']
    funext r c
    simp [matrixAdd, matrixMul, matrixSub, hmono, hg2, Z, zAt, hz, hw4, b, rowWindow]

/-- The error of the attribute ciphertext's opening: `-(e_b Z)`, by component. -/
noncomputable def rhsErr (tape : SampleTape) (path : List Nat) (j : Fin 2) : ErrorMatrix N 1 10 :=
  fun r c ↦ -((ebAt tape path * zAt tape path) r ⟨10 * (j : Nat) + c, by omega⟩)

theorem rhsErr_bound {tape : SampleTape} {path : List Nat} (heb : CoeffBound (ebAt tape path) 30)
    (hz : CoeffBound (zAt tape path) 30) (j : Fin 2) :
    CoeffBound (rhsErr tape path j) (12 * N * 30 * 30) := by
  have hb := coeffBound_mul hN heb hz
  intro r c k
  have := hb r ⟨10 * (j : Nat) + c, by omega⟩ k
  simpa [rhsErr, ebAt, zAt] using this

/-- The attribute ciphertext opens under `(s, 1)` to `(s, 1) G` up to `-(e_b Z)`, for `x₀ = 0`. -/
theorem prelude_rhs {model : HashModel} {key : ByteArray} {tape : SampleTape} {path : List Nat}
    {B : ExactMatrix Q N 1 12} {sm : ErrorPoly N} {g : ExactMatrix Q N 1 10}
    {b : ExactMatrix Q N 1 12} {C00 C01 C10 C11 : ExactMatrix Q N 1 10}
    (hb : b = reducePoly Q N sm • B + anchorSecond model key + reduceMatrix Q N 1 12 (ebAt tape path))
    (h00 : C00 = rowWindow 10 0 (by omega) (B * reduceMatrix Q N 12 20 (zAt tape path)) + g)
    (h01 : C01 = rowWindow 10 10 (by omega) (B * reduceMatrix Q N 12 20 (zAt tape path)))
    (h10 : C10 = rowWindow 10 0 (by omega)
      ((anchorSecond model key - b) * reduceMatrix Q N 12 20 (zAt tape path)))
    (h11 : C11 = rowWindow 10 10 (by omega)
      ((anchorSecond model key - b) * reduceMatrix Q N 12 20 (zAt tape path)) + g) :
    ∀ j, reducePoly Q N sm • ![C00, C01] j + ![C10, C11] j =
      reducePoly Q N (unitPair sm j) • g + reduceMatrix Q N 1 10 (rhsErr tape path j) := by
  have hP : (anchorSecond model key - b) * reduceMatrix Q N 12 20 (zAt tape path) =
      -(reducePoly Q N sm • (B * reduceMatrix Q N 12 20 (zAt tape path))) -
        reduceMatrix Q N 1 20 (ebAt tape path * zAt tape path) := by
    rw [hb, reduceMatrix_mul]
    simp only [Matrix.sub_mul, Matrix.add_mul, Matrix.smul_mul]
    abel
  intro j
  funext r c
  rw [Subsingleton.elim r 0]
  fin_cases j
  · simp [h00, h10, hP, rowWindow, unitPair, rhsErr, reduceMatrix]
    ring
  · simp [h01, h11, hP, rowWindow, unitPair, rhsErr, reduceMatrix]
    ring

end PeProof
