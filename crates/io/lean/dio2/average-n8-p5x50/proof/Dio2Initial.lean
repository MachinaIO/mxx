import W_Stage_initial_Facts
import Dio2Inv
import ExponentRefresh

/-!
# The initial step input

Every root row the obfuscator encrypts (`Stage_initial`) is a root row of `S_0` (`RootRow`) under
the sampled mask `s̄_0` (the stage's first input).
-/

namespace Dio2Proof

open Mxx.Primitives MxxRuntime MxxExponentLut W_Stage_initial

/-- The tag prefix of `ONEPRIV` (`rns-bgg/root-input/dio2/one-private`). -/
def privatePrefix : Blob :=
  [114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100,
    105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]

/-- `ONEPRIV` of `S_0`: the encoding of the payload row `(t̄, 1)` (the stage's fifth input). -/
theorem initial_private {backend : BackendContext} {model : HashModel} {params : Params}
    {inputs : _} {outputs : _} (h : generatedRoot backend model params inputs outputs)
    {sm : ErrorPoly 8} (hmask : inputs.1 0 0 = reducePoly modulus_0 8 sm) :
    ∃ G, gadgetMatrixRuns backend 4096 25 G ∧
      RootRow model inputs.2.2.1 privatePrefix [] sm G
        (![matrixMulScalarLeft 1 inputs.2.2.2.2.1, matrixMulScalarLeft 1 1])
        (![(outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1),
          (outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1)]) := by
  obtain ⟨witness, hbody⟩ := h
  let ctx : generatedRoot.Ctx backend model params := ⟨inputs, witness⟩
  have F := generatedRoot.facts ctx hbody
  refine ⟨generatedRoot.w_425_0 ctx, F.c5.h425_0, ?_⟩
  -- Block `(k, j)` of the root key is tag component `2 k + j`.
  have e00 : rootKey model inputs.2.2.1 privatePrefix [] modulus_0 25 0 0 =
      generatedRoot.w_417_0 ctx := by rw [hashSample_eq F.c5.h417_0]; rfl
  have e01 : rootKey model inputs.2.2.1 privatePrefix [] modulus_0 25 0 1 =
      generatedRoot.w_432_0 ctx := by rw [hashSample_eq F.c5.h432_0]; rfl
  have e10 : rootKey model inputs.2.2.1 privatePrefix [] modulus_0 25 1 0 =
      generatedRoot.w_419_0 ctx := by rw [hashSample_eq F.c5.h419_0]; rfl
  have e11 : rootKey model inputs.2.2.1 privatePrefix [] modulus_0 25 1 1 =
      generatedRoot.w_434_0 ctx := by rw [hashSample_eq F.c5.h434_0]; rfl
  apply encode_rel hmask (u := ![generatedRoot.w_428_0 ctx, generatedRoot.w_441_0 ctx])
    (u' := ![generatedRoot.w_429_0 ctx, generatedRoot.w_442_0 ctx])
  · intro j; fin_cases j
    · exact F.c5.h428_0
    · exact F.c5.h441_0
  · intro j; fin_cases j
    · exact F.c5.h429_0
    · exact F.c5.h442_0
  · intro j
    rw [F.c5.hout]
    fin_cases j
    · simp only [Fin.zero_eta, Fin.isValue, Matrix.cons_val_zero, e00, e10]
      rfl
    · simp only [Fin.mk_one, Fin.isValue, Matrix.cons_val_one, e01, e11]
      rfl

end Dio2Proof


namespace Dio2Proof

open Mxx.Primitives MxxRuntime MxxExponentLut Mxx.ExponentLut.Refresh W_Stage_initial

/-- Row `k` of a two-row matrix. -/
def rowOf {q n c : Nat} (B : ExactMatrix q n 2 c) (k : Fin 2) : ExactMatrix q n 1 c :=
  fun _ column ↦ B k column

/-- The hint of `S_0`: `b_0 = s̄_0 B_0[0] + B_0[1] + e` with `e` of coefficients at most one, for
the anchor `B_0` (the stage's second input). -/
theorem initial_hint {backend : BackendContext} {model : HashModel} {params : Params}
    {inputs : _} {outputs : _} (h : generatedRoot backend model params inputs outputs)
    {sm : ErrorPoly 8} (hmask : inputs.1 0 0 = reducePoly modulus_0 8 sm) :
    ∃ e : ErrorMatrix 8 1 54, CoeffBound e 1 ∧
      outputs.1 = maskPart (reducePoly modulus_0 8 sm) (rowOf inputs.2.1) +
        reduceMatrix modulus_0 8 1 54 e := by
  obtain ⟨witness, hbody⟩ := h
  let ctx : generatedRoot.Ctx backend model params := ⟨inputs, witness⟩
  have F := generatedRoot.facts ctx hbody
  obtain ⟨e, he, hue⟩ := uniform_bits_sub F.c0.h5_0 F.c0.h6_0
  refine ⟨e, he, ?_⟩
  rw [F.c5.hout]
  have hcat := F.c0.h2_0
  show generatedRoot.w_2_0 ctx * inputs.2.1 +
    (generatedRoot.w_5_0 ctx - generatedRoot.w_6_0 ctx) = _
  rw [hue]
  congr 1
  funext row column
  rw [Matrix.mul_apply, Fin.sum_univ_two, hcat row 0, hcat row 1]
  have hrow : row = 0 := Subsingleton.elim _ _
  subst hrow
  simp only [Fin.isValue, Fin.coe_ofNat_eq_mod, Nat.zero_mod, zero_lt_one, ↓reduceDIte,
    Nat.one_mod, lt_self_iff_false, Nat.sub_self, maskPart, rowOf, Matrix.add_apply,
    Matrix.smul_apply, smul_eq_mul]
  show inputs.1 0 0 * _ + (1 : ExactMatrix modulus_0 8 1 1) 0 0 * _ = _
  rw [hmask]
  simp

end Dio2Proof
