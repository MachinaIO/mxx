import PeStepRel
import ExponentScale
import Stage_prelude
import Stage_prelude_Facts

/-!
# The encryption prelude

Under the secret `s`, the attribute ciphertext of position `i` has the public top row `R_i`
(hashed, `rhs-top`) and the bottom row `X^{x_i} (s, 1) g - s R_i + e`; the anchor is
`b_Q = s̄ B_Q + B_Q^{(2)} + e_b` over `R_{PQ}`; and the message component is
`c_m = s d₀ + d₁ + e_m + ⌊Q/2⌋ m`.
-/

set_option maxRecDepth 200000

namespace PeProof

open Mxx.Primitives MxxRuntime MxxExponentLut

/-- The law of the prelude samples. -/
abbrev preludeLaw : SamplerLaw := .gaussian (319 / 100) 31

/-- `mxx-pe` -/
def peTag : Blob := [109, 120, 120, 45, 112, 101]
/-- The hex digest of the PE schema. -/
def peSchema : Blob := [51, 52, 101, 99, 56, 49, 55, 48, 55, 98, 100, 100, 54, 99, 102, 98, 102, 99, 99, 52, 50, 51, 99, 55, 99, 98, 54, 97, 53, 51, 51, 100, 52, 57, 98, 102, 98, 101, 99, 56, 98, 57, 100, 97, 57, 101, 98, 98, 56, 48, 97, 52, 54, 57, 52, 51, 53, 53, 52, 48, 99, 101, 50, 48]
/-- `anchor-second-row` -/
def anchorBytes : Blob := [97, 110, 99, 104, 111, 114, 45, 115, 101, 99, 111, 110, 100, 45, 114, 111, 119]
/-- `message-row` -/
def messageBytes : Blob := [109, 101, 115, 115, 97, 103, 101, 45, 114, 111, 119]
/-- `rhs-top` -/
def rhsTopBytes : Blob := [114, 104, 115, 45, 116, 111, 112]
/-- `member` -/
def memberBytes : Blob := [109, 101, 109, 98, 101, 114]

/-- Component `j` of the public top row of the attribute ciphertext of position `i`. -/
noncomputable def rhsTop (model : HashModel) (key : ByteArray) (i j : Int) : ExactMatrix Q N 1 30 :=
  model.sample Q N 1 30 key (completeHashTag dynamicPrefix [.bytes schemaBytes, .bytes rhsTopBytes,
    .u64Le 0, .bytes memberBytes, .u64Le i, .bytes componentBytes, .u64Le j])

/-- The hashed second anchor row `B_Q^{(2)}`. -/
noncomputable def anchorSecond (model : HashModel) (key : ByteArray) : ExactMatrix PQ N 1 295 :=
  model.sample PQ N 1 295 key (completeHashTag peTag [.bytes peSchema, .bytes anchorBytes])

/-- The hashed message rows `d_j`. -/
noncomputable def messageRow (model : HashModel) (key : ByteArray) (j : Int) : ExactMatrix Q N 1 1 :=
  model.sample Q N 1 1 key (completeHashTag peTag [.bytes peSchema, .bytes messageBytes, .u64Le j])

/-- The error of component `j` of the attribute ciphertext of position `i`. -/
noncomputable def attrErr (tape : SampleTape) (path : List Nat) (i : Nat) (j : Fin 2) : ErrorMatrix N 1 30 :=
  tapeWitness tape (path ++ [6, i] ++ [![11, 19] j]) preludeLaw

/-- The anchor error `e_b`. -/
noncomputable def anchorErr (tape : SampleTape) (path : List Nat) : ErrorMatrix N 1 295 :=
  tapeWitness tape (path ++ [12]) preludeLaw

/-- The message error `e_m`. -/
noncomputable def messageErr (tape : SampleTape) (path : List Nat) : ErrorMatrix N 1 1 :=
  tapeWitness tape (path ++ [18]) preludeLaw

/-- `⌊Q/2⌋`. -/
abbrev halfQ : Int := 35846658492324948421952126721744601951110048738142724512388317637992329893241430421904827283995948231327744

/-- The centered rebase of a small integer polynomial is its reduction. -/
theorem rebase_small {q p : Nat} (hq : 2 < q) {s : ErrorPoly N} (hs : polyNorm s ≤ 1)
    {out : ExactMatrix p N 1 1} (h : centeredRebaseRuns (fun _ _ ↦ reducePoly q N s) out) :
    out = fun _ _ ↦ reducePoly p N s := by
  rw [centeredRebaseRuns_reduce h]
  funext r c
  rw [centeredIntLift_reducePoly (by omega) hN _ fun i ↦ by
    have := coeff_natAbs_le_polyNorm s i
    omega]

/-- `X^x` for `0 ≤ x < 2N`. -/
theorem monomial_Xe {x : Int} (h0 : 0 ≤ x) (hx : x < 2 * N) :
    (multiplyMonomial (1 : ExactMatrix Q N 1 1) x : ExactMatrix Q N 1 1) =
      fun _ _ ↦ reducePoly Q N (Xe x.toNat) := by
  funext a c
  rw [Subsingleton.elim a 0, Subsingleton.elim c 0]
  simp only [multiplyMonomial, Matrix.one_apply_eq, one_mul, reducePoly_Xe]
  rw [show (2 * ((N : Nat) : Int)) = 65536 from rfl, Int.emod_eq_of_lt h0 (by simpa using hx)]

set_option maxHeartbeats 4000000 in
/-- The attribute ciphertext of position `i` opens under `(s, 1)` to `X^{x_i} (s, 1) g` up to its
Gaussian error. -/
theorem prelude_attr {model : HashModel} {tape : SampleTape} {path : List Nat} {i : Nat}
    {key : ByteArray} {sm : ErrorPoly N} {x : Int} (hx0 : 0 ≤ x) (hx : x < 2 * N)
    {outputs : ExactMatrix Q N 1 30 × ExactMatrix Q N 1 30 × Unit}
    (h : Stage_prelude.parallel_generatedRoot_6 Backend.backend model tape (path ++ [6, i])
      GeneratedClaim.stage_3_params i ((fun _ _ ↦ reducePoly Q N sm), (x, (key, (1, ())))) outputs) :
    ∀ j : Fin 2, reducePoly Q N sm • rhsTop model key i j + ![outputs.1, outputs.2.1] j =
      reducePoly Q N (Xe x.toNat * unitPair sm j) • g0 + reduceMatrix Q N 1 30 (attrErr tape path i j) ∧
      CoeffBound (attrErr tape path i j) 31 := by
  obtain ⟨w1, w8, w11, w16, w19, hg, h8, h11, h16, h19, hout⟩ := h
  have hw1 : w1 = g0 := gadgetMatrixRuns_unique hg g0_runs
  obtain ⟨he11, hb11⟩ := gaussianSampleAt_explicit hN h11
  obtain ⟨he19, hb19⟩ := gaussianSampleAt_explicit hN h19
  have hw8 : w8 = rhsTop model key i 0 := hashSample_eq h8
  have hw16 : w16 = rhsTop model key i 1 := hashSample_eq h16
  have hmono := monomial_Xe hx0 hx
  subst hout hw1 hw8 hw16 he11 he19
  intro j
  fin_cases j
  · refine ⟨?_, hb11⟩
    funext r c
    rw [Subsingleton.elim r 0]
    simp only [matrixAdd, matrixSub, matrixMulScalarLeft_eq,
      matrixMulScalarRight_eq, hmono, unitPair, attrErr, Matrix.add_apply,
      Matrix.smul_apply, smul_eq_mul, map_mul]
    simp
    ring
  · refine ⟨?_, hb19⟩
    funext r c
    rw [Subsingleton.elim r 0]
    simp only [matrixAdd, matrixSub, matrixMulScalarLeft_eq,
      matrixMulScalarRight_eq, hmono, unitPair, attrErr, Matrix.add_apply,
      Matrix.smul_apply, smul_eq_mul, map_mul]
    simp
    ring

abbrev PreludeOutputs := (Fin 45 → ExactMatrix Q N 1 30) ×
  (Fin 45 → ExactMatrix Q N 1 30) × (Fin 2 → ExactMatrix Q N 1 30) × (Fin 2 → ExactMatrix Q N 1 30) ×
  ExactMatrix PQ N 1 295 × ExactMatrix Q N 1 1 × Unit

set_option maxHeartbeats 16000000 in
/-- The prelude: the four attribute ciphertexts, the anchor over `R_{PQ}`, and the message
component, each with its Gaussian error. -/
theorem prelude_explicit {model : HashModel} {tape : SampleTape} {path : List Nat}
    {secret : ExactMatrix Q N 1 1} {x : Fin 2 → Int} {key : ByteArray} {BQ : ExactMatrix PQ N 1 295}
    {m : Fin N → Int} {outputs : PreludeOutputs}
    (h : Stage_prelude.generatedRoot Backend.backend model tape path GeneratedClaim.stage_3_params
      (secret, key, x, BQ, m, ()) outputs)
    {sm : ErrorPoly N} (hsecret : secret = fun _ _ ↦ reducePoly Q N sm) (hs : polyNorm sm ≤ 1)
    (hx : ∀ i, 0 ≤ x i ∧ x i ≤ 248) :
    (∀ (i : Fin 2) (j : Fin 2), reducePoly Q N sm • rhsTop model key i j + ![outputs.2.2.1 i, outputs.2.2.2.1 i] j =
      reducePoly Q N (Xe (x i).toNat * unitPair sm j) • g0 + reduceMatrix Q N 1 30 (attrErr tape path i j) ∧
      CoeffBound (attrErr tape path i j) 31) ∧
    outputs.2.2.2.2.1 = reducePoly PQ N sm • BQ + anchorSecond model key +
      reduceMatrix PQ N 1 295 (anchorErr tape path) ∧
    CoeffBound (anchorErr tape path) 31 ∧
    outputs.2.2.2.2.2.1 = reducePoly Q N sm • messageRow model key 0 + messageRow model key 1 +
      reduceMatrix Q N 1 1 (messageErr tape path) +
      ((halfQ : ExactPoly Q N) • (fun _ _ ↦ polynomialOfCoefficients m) : ExactMatrix Q N 1 1) ∧
    CoeffBound (messageErr tape path) 31 := by
  obtain ⟨w, hbody⟩ := h
  let ctx : Stage_prelude.generatedRoot.Ctx Backend.backend model tape path GeneratedClaim.stage_3_params :=
    ⟨(secret, key, x, BQ, m, ()), w⟩
  have F := Stage_prelude.generatedRoot.facts ctx hbody
  have h5 := F.c0.h7_0
  change centeredRebaseRuns secret _ at h5
  rw [hsecret] at h5
  have hw7 := rebase_small (q := Q) (by norm_num) hs h5
  obtain ⟨he10, hb10⟩ := gaussianSampleAt_explicit hN F.c0.h12_0
  obtain ⟨he16, hb16⟩ := gaussianSampleAt_explicit hN F.c0.h18_0
  have hw10 : Stage_prelude.generatedRoot.w_10_0 ctx = anchorSecond model key := hashSample_eq F.c0.h10_0
  have hw14 : Stage_prelude.generatedRoot.w_14_0 ctx = messageRow model key 0 := hashSample_eq F.c0.h14_0
  have hw16 : Stage_prelude.generatedRoot.w_16_0 ctx = messageRow model key 1 := hashSample_eq F.c0.h16_0
  have hw22 : Stage_prelude.generatedRoot.w_22_0 ctx = fun _ _ ↦ polynomialOfCoefficients m := F.c0.h22_0
  rw [F.c0.hout]
  refine ⟨fun i j ↦ ?_, ?_, hb10, ?_, hb16⟩
  · have hi := F.c0.h6_1 i
    change Stage_prelude.parallel_generatedRoot_6 _ _ _ _ _ _ (secret, (x i, (key, (1, ())))) _ at hi
    rw [hsecret] at hi
    have := prelude_attr (hx i).1 (by have := (hx i).2; push_cast; omega) hi j
    exact this
  · show matrixAdd (matrixAdd (matrixMulScalarLeft (Stage_prelude.generatedRoot.w_7_0 ctx) BQ)
      (Stage_prelude.generatedRoot.w_10_0 ctx)) (Stage_prelude.generatedRoot.w_12_0 ctx) = _
    rw [hw7, hw10, he10]
    rfl
  · show matrixAdd (matrixAdd (matrixAdd (matrixMulScalarLeft secret (Stage_prelude.generatedRoot.w_14_0 ctx))
      (Stage_prelude.generatedRoot.w_16_0 ctx)) (Stage_prelude.generatedRoot.w_18_0 ctx))
      (matrixMulScalarLeft (matrixPolynomial [halfQ]) (Stage_prelude.generatedRoot.w_22_0 ctx)) = _
    rw [hw14, hw16, he16, hw22, hsecret]
    funext r c
    rw [Subsingleton.elim r 0, Subsingleton.elim c 0]
    simp only [matrixAdd, matrixMulScalarLeft_eq, Matrix.add_apply, Matrix.smul_apply, smul_eq_mul,
      messageErr]
    simp [matrixPolynomial]
    rfl

end PeProof
