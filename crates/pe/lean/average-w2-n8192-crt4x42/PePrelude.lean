import PeEncrypt

/-!
# The encryption prelude

Under the secret `s`, the anchor encoding `b = s B + B² + e_b`, the attribute ciphertext
`C = P Z + X^x G` for `P = [B; B² - b]`, and the message component
`c_m = s d₁ + d₂ + e_m + ⌊Q/2⌋ m`. Every sample is named by its tape site. A loop samples `Z` one
column `z_j` at a time and adds `P z_j` at column `j` through a window of a constant unit row, so
`zLoop` reads the rows `P Z` back with `IterRuns.accumulate_entries`.
-/

namespace PeProof

open Mxx.Primitives MxxRuntime MxxExponentLut

/-- The law of the prelude samples. -/
abbrev preludeLaw : SamplerLaw := .gaussian (319 / 100) 20

/-- The anchor error `e_b`. -/
noncomputable def ebAt (tape : SampleTape) (path : List Nat) : ErrorMatrix N 1 10 :=
  tapeWitness tape (path ++ [6]) preludeLaw

/-- Column `j` of `Z`, sampled by iteration `j` of the column loop. -/
noncomputable def zColAt (tape : SampleTape) (path : List Nat) (j : Nat) : ErrorMatrix N 10 1 :=
  tapeWitness tape (path ++ [11, 0, 3, j, 2]) preludeLaw

/-- The attribute matrix `Z`, column by column. -/
noncomputable def zAt (tape : SampleTape) (path : List Nat) : ErrorMatrix N 10 16 :=
  fun r c ↦ zColAt tape path c r 0

/-- The message error `e_m`. -/
noncomputable def emAt (tape : SampleTape) (path : List Nat) : ErrorMatrix N 1 1 :=
  tapeWitness tape (path ++ [16]) preludeLaw

/-- `⌊Q/2⌋`. -/
abbrev halfQ : Int := 187072069502007624630008562053914270362791364190208

abbrev PreludeOutputs := (Fin 1 → ExactMatrix Q N 1 8) × (Fin 1 → ExactMatrix Q N 1 8) ×
  (Fin 1 → ExactMatrix Q N 1 8) × (Fin 1 → ExactMatrix Q N 1 8) × ExactMatrix Q N 1 10 ×
  ExactMatrix Q N 1 1 × Unit

/-- The rows `P₁ Z` and `P₂ Z` that the column loop accumulates. -/
abbrev ZRows := ExactMatrix Q N 1 16 × ExactMatrix Q N 1 16 × Unit

set_option maxHeartbeats 8000000 in
/-- Iteration `j` of the column loop samples `z_j` and adds `P₁ z_j` and `P₂ z_j` at column `j`. -/
theorem zLoop_step {model : HashModel} {tape : SampleTape} {path : List Nat}
    {P1 P2 : ExactMatrix Q N 1 10} {j : Nat} {s t : ZRows}
    (h : Stage_prelude.sequential_parallel_generatedRoot_11_3 Backend.backend model tape
      (path ++ [11, 0] ++ [3, j]) GeneratedClaim.stage_3_params 0 j (s.1, (s.2.1, (P1, (P2, ())))) t) :
    (∀ w, t.1 0 w = s.1 0 w + (P1 * reduceMatrix Q N 10 1 (zColAt tape path j)) 0 0 *
        (unitRow ((j : Nat) : Int) : ExactMatrix Q N 1 16) 0 w) ∧
      (∀ w, t.2.1 0 w = s.2.1 0 w + (P2 * reduceMatrix Q N 10 1 (zColAt tape path j)) 0 0 *
        (unitRow ((j : Nat) : Int) : ExactMatrix Q N 1 16) 0 w) ∧
      CoeffBound (zColAt tape path j) 20 := by
  obtain ⟨z, u, hz, -, -, -, -, -, -, -, -, hu, hout⟩ := h
  obtain ⟨hz, hzb⟩ := gaussianSampleAt_explicit hN hz
  have hsite : path ++ [11, 0] ++ [3, j] ++ [2] = path ++ [11, 0, 3, j, 2] := by simp
  rw [hsite] at hz hzb
  have hw := unitRow_window (c := ((j : Nat) : Int)) (by simp only [Int.ofNat_eq_natCast]; omega) hu
  subst hout hw hz
  refine ⟨fun w ↦ ?_, fun w ↦ ?_, hzb⟩ <;>
    simp only [matrixAdd, matrixMulScalarLeft_eq, matrixMul, zColAt, Matrix.add_apply, Matrix.smul_apply,
      smul_eq_mul]

set_option maxHeartbeats 8000000 in
/-- The column loop ends with `P₁ Z` and `P₂ Z`, whose columns are bounded samples. -/
theorem zLoop {model : HashModel} {tape : SampleTape} {path : List Nat}
    {P1 P2 : ExactMatrix Q N 1 10} {final : ZRows}
    (h : MxxIR.IterRuns (fun (i : Nat) (current next : ZRows) ↦
      Stage_prelude.sequential_parallel_generatedRoot_11_3 Backend.backend model tape (path ++ [11, 0] ++ [3, i])
        GeneratedClaim.stage_3_params 0 i (current.1, (current.2.1, (P1, (P2, ())))) next)
      (Int.toNat 16) (0, (0, ())) final) :
    final.1 = P1 * reduceMatrix Q N 10 16 (zAt tape path) ∧
      final.2.1 = P2 * reduceMatrix Q N 10 16 (zAt tape path) ∧
      CoeffBound (zAt tape path) 20 := by
  have hcount : Int.toNat 16 = 16 := rfl
  rw [hcount] at h
  have hcol : ∀ (P : ExactMatrix Q N 1 10) (w : Fin 16),
      (P * reduceMatrix Q N 10 1 (zColAt tape path w)) 0 0 =
        (P * reduceMatrix Q N 10 16 (zAt tape path)) 0 w := by
    intro P w
    simp only [Matrix.mul_apply, reduceMatrix, zAt]
  have hk1 := IterRuns.accumulate_entries (fun s : ZRows ↦ s.1)
    (fun w (j : Fin 16) ↦ (P1 * reduceMatrix Q N 10 1 (zColAt tape path j)) 0 0 *
      (unitRow ((j : Nat) : Int) : ExactMatrix Q N 1 16) 0 w)
    (fun _ _ ↦ True) trivial (fun _ _ _ _ _ hb ↦ ⟨(zLoop_step hb).1, trivial⟩) h
  have hk2 := IterRuns.accumulate_entries (fun s : ZRows ↦ s.2.1)
    (fun w (j : Fin 16) ↦ (P2 * reduceMatrix Q N 10 1 (zColAt tape path j)) 0 0 *
      (unitRow ((j : Nat) : Int) : ExactMatrix Q N 1 16) 0 w)
    (fun _ _ ↦ True) trivial (fun _ _ _ _ _ hb ↦ ⟨(zLoop_step hb).2.1, trivial⟩) h
  have hb := MxxIR.IterRuns.invariant
    (Invariant := fun i (_ : ZRows) ↦ ∀ j < i, CoeffBound (zColAt tape path j) 20)
    (fun _ h ↦ absurd h (Nat.not_lt_zero _))
    (fun _ _ _ hs hb j hj ↦ by
      rcases Nat.lt_succ_iff_lt_or_eq.mp hj with hj | rfl
      · exact hs j hj
      · exact (zLoop_step hb).2.2) h
  refine ⟨?_, ?_, fun r c k ↦ hb c c.isLt r 0 k⟩
  · funext r w
    have h1 := hk1 w
    simp only at h1
    rw [Subsingleton.elim r 0, h1, sum_unitRow_mul (fun j ↦ (P1 * reduceMatrix Q N 10 1
      (zColAt tape path j)) 0 0) w, hcol]
    simp
  · funext r w
    have h2 := hk2 w
    simp only at h2
    rw [Subsingleton.elim r 0, h2, sum_unitRow_mul (fun j ↦ (P2 * reduceMatrix Q N 10 1
      (zColAt tape path j)) 0 0) w, hcol]
    simp

set_option maxHeartbeats 16000000 in
theorem prelude_explicit {model : HashModel} {tape : SampleTape} {path : List Nat}
    {B : ExactMatrix Q N 1 10} {x : Fin 1 → Int} {key : ByteArray} {secret : ExactMatrix Q N 1 1}
    {m : Fin N → Int} {outputs : PreludeOutputs}
    (h : Stage_prelude.generatedRoot Backend.backend model tape path GeneratedClaim.stage_3_params
      (B, key, secret, x, m, ()) outputs)
    (hx : x 0 = 0) {sm : ErrorPoly N} (hsecret : secret = fun _ _ ↦ reducePoly Q N sm)
    {g : ExactMatrix Q N 1 8} (hg : gadgetMatrixRuns Backend.backend 2097152 8 g) :
    let Z := reduceMatrix Q N 10 16 (zAt tape path)
    let b := outputs.2.2.2.2.1
    b = reducePoly Q N sm • B + anchorSecond model key + reduceMatrix Q N 1 10 (ebAt tape path) ∧
    outputs.2.2.2.2.2.1 = reducePoly Q N sm • messageRow model key 0 + messageRow model key 1 +
      reduceMatrix Q N 1 1 (emAt tape path) +
      ((halfQ : ExactPoly Q N) • (fun _ _ ↦ polynomialOfCoefficients m) : ExactMatrix Q N 1 1) ∧
    outputs.1 0 = rowWindow 8 0 (by omega) (B * Z) + g ∧
    outputs.2.1 0 = rowWindow 8 8 (by omega) (B * Z) ∧
    outputs.2.2.1 0 = rowWindow 8 0 (by omega) ((anchorSecond model key - b) * Z) ∧
    outputs.2.2.2.1 0 = rowWindow 8 8 (by omega) ((anchorSecond model key - b) * Z) + g ∧
    CoeffBound (ebAt tape path) 20 ∧ CoeffBound (zAt tape path) 20 ∧ CoeffBound (emAt tape path) 20 := by
  intro Z b
  obtain ⟨w, h2, h6, h10, -, h11, h12, h14, h16, h20, hout⟩ := h
  have hg10 : w.w_10_0 = g := gadgetMatrixRuns_unique h10 hg
  obtain ⟨he6, hb6⟩ := gaussianSampleAt_explicit hN h6
  obtain ⟨he16, hb16⟩ := gaussianSampleAt_explicit hN h16
  have hw2 : w.w_2_0 = anchorSecond model key := hashSample_eq h2
  have hw12 : w.w_12_0 = messageRow model key 0 := hashSample_eq h12
  have hw14 : w.w_14_0 = messageRow model key 1 := hashSample_eq h14
  have hw20 : w.w_20_0 = fun _ _ ↦ polynomialOfCoefficients m := h20
  obtain ⟨v, -, hloop, -, -, -, -, -, -, x0, -, -, -, -, -, -, x1, -, -, -, -, -, -, x2, -, -, -, -, -, -,
    x3, vout⟩ := h11 0
  obtain ⟨htop, hbottom, hbz⟩ := zLoop hloop
  simp only at htop hbottom
  subst hout
  have hb : b = reducePoly Q N sm • B + anchorSecond model key + reduceMatrix Q N 1 10 (ebAt tape path) := by
    simp only [b, matrixAdd, matrixMulScalarLeft_eq, hsecret, hw2, he6, ebAt]
  have hmono : (multiplyMonomial (1 : ExactMatrix Q N 1 1) (x 0) : ExactMatrix Q N 1 1) = 1 := by
    funext a c
    fin_cases a; fin_cases c
    simp [multiplyMonomial, hx]
  have hb6' : CoeffBound (ebAt tape path) 20 := by
    intro r c k
    have := hb6 r c k
    simpa [ebAt] using this
  have hb16' : CoeffBound (emAt tape path) 20 := by
    intro r c k
    have := hb16 r c k
    simpa [emAt] using this
  have hlower : (matrixSub w.w_2_0 (matrixAdd (matrixAdd (matrixMulScalarLeft secret B) w.w_2_0) w.w_6_0) :
      ExactMatrix Q N 1 10) = anchorSecond model key - b := by
    simp only [b, hw2]
    rfl
  rw [hlower] at hbottom
  refine ⟨hb, ?_, ?_, ?_, ?_, ?_, hb6', hbz, hb16'⟩
  · simp only [matrixAdd, matrixMulScalarLeft_eq, hsecret, hw12, hw14, he16, emAt, hw20,
      matrixPolynomial_single]
  · simp only [Prod.mk.injEq] at vout
    obtain ⟨ho0, -⟩ := vout
    have hw0 := sliceMatrix_window 0 x0 (by simp) (by omega)
    show w.w_11_0 0 = _
    rw [ho0, hw0, htop]
    funext r c
    simp [matrixAdd, hmono, hg10, Z, rowWindow]
  · simp only [Prod.mk.injEq] at vout
    obtain ⟨-, ho1, -⟩ := vout
    have hw1 := sliceMatrix_window 8 x1 (by simp) (by omega)
    show w.w_11_1 0 = _
    rw [ho1, hw1, htop]
  · simp only [Prod.mk.injEq] at vout
    obtain ⟨-, -, ho2, -⟩ := vout
    have hw2' := sliceMatrix_window 0 x2 (by simp) (by omega)
    show w.w_11_2 0 = _
    rw [ho2, hw2', hbottom]
  · simp only [Prod.mk.injEq] at vout
    obtain ⟨-, -, -, ho3, -⟩ := vout
    have hw3 := sliceMatrix_window 8 x3 (by simp) (by omega)
    show w.w_11_3 0 = _
    rw [ho3, hw3, hbottom]
    funext r c
    simp [matrixAdd, hmono, hg10, Z, rowWindow]

/-- The error of the attribute ciphertext's opening: `-(e_b Z)`, by component. -/
noncomputable def rhsErr (tape : SampleTape) (path : List Nat) (j : Fin 2) : ErrorMatrix N 1 8 :=
  fun r c ↦ -((ebAt tape path * zAt tape path) r ⟨8 * (j : Nat) + c, by omega⟩)

theorem rhsErr_bound {tape : SampleTape} {path : List Nat} (heb : CoeffBound (ebAt tape path) 20)
    (hz : CoeffBound (zAt tape path) 20) (j : Fin 2) :
    CoeffBound (rhsErr tape path j) (10 * N * 20 * 20) := by
  have hb := coeffBound_mul hN heb hz
  intro r c k
  have := hb r ⟨8 * (j : Nat) + c, by omega⟩ k
  simpa [rhsErr] using this

/-- The attribute ciphertext opens under `(s, 1)` to `(s, 1) G` up to `-(e_b Z)`, for `x₀ = 0`. -/
theorem prelude_rhs {model : HashModel} {key : ByteArray} {tape : SampleTape} {path : List Nat}
    {B : ExactMatrix Q N 1 10} {sm : ErrorPoly N} {g : ExactMatrix Q N 1 8}
    {b : ExactMatrix Q N 1 10} {C00 C01 C10 C11 : ExactMatrix Q N 1 8}
    (hb : b = reducePoly Q N sm • B + anchorSecond model key + reduceMatrix Q N 1 10 (ebAt tape path))
    (h00 : C00 = rowWindow 8 0 (by omega) (B * reduceMatrix Q N 10 16 (zAt tape path)) + g)
    (h01 : C01 = rowWindow 8 8 (by omega) (B * reduceMatrix Q N 10 16 (zAt tape path)))
    (h10 : C10 = rowWindow 8 0 (by omega)
      ((anchorSecond model key - b) * reduceMatrix Q N 10 16 (zAt tape path)))
    (h11 : C11 = rowWindow 8 8 (by omega)
      ((anchorSecond model key - b) * reduceMatrix Q N 10 16 (zAt tape path)) + g) :
    ∀ j, reducePoly Q N sm • ![C00, C01] j + ![C10, C11] j =
      reducePoly Q N (unitPair sm j) • g + reduceMatrix Q N 1 8 (rhsErr tape path j) := by
  have hP : (anchorSecond model key - b) * reduceMatrix Q N 10 16 (zAt tape path) =
      -(reducePoly Q N sm • (B * reduceMatrix Q N 10 16 (zAt tape path))) -
        reduceMatrix Q N 1 16 (ebAt tape path * zAt tape path) := by
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
