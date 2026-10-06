import LwrLutE
import LwrPreprocessE
import LwrEncryptE

/-!
# The model of a run (prototype)

Every input of the error analysis as a function of the sample tape: the payload and mask secrets
and the Gaussian errors, each clamped to its sampler's range (which every run reads unchanged),
and the hash-derived keys.
-/

namespace LwrProof

open Mxx.Primitives MxxRuntime MxxExponentLut

variable {tape : SampleTape} {path : List Nat}

/-- `rns-bgg/lut-b/LutId(0)/5/6` -/
def lutBPrefix : Blob :=
  [114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 98, 47, 76, 117, 116, 73, 100, 40, 48,
    41, 47, 53, 47, 54]

/-- The hash-derived switching tops `B_{0j}` of branch `i` of the readout LUT. -/
noncomputable def lutB (model : HashModel) (key : ByteArray) (i : Fin 16) (j : Fin 2) :
    ExactMatrix Q1 N 1 2 :=
  model.sample Q1 N 1 2 key (completeHashTag lutBPrefix
    [.bytes [115, 105, 103, 109, 97], .u64Le (sigma i), .bytes [98, 114, 97, 110, 99, 104],
      .u64Le (i : Int), .bytes componentBytes, .u64Le (j : Int)])

set_option maxHeartbeats 8000000 in
/-- The switching tops of the readout LUT are hash-derived. -/
theorem preprocess_switchTop {model : HashModel} {key : ByteArray} {payload : ExactMatrix Q7 N 1 1}
    {b0 x0 b1 x1 b2 x2 b3 x3 b4 x4 : Fin 14 → Int} {outputs}
    (h : Stage_preprocess.generatedRoot Backend.backend model tape path GeneratedClaim.stage_2_params
      (key, payload, b0, x0, b1, x1, b2, x2, b3, x3, b4, x4, ()) outputs) (i : Fin 16) :
    ![outputs.1 i, outputs.2.1 i] = lutB model key i := by
  obtain ⟨w, hb⟩ := h
  unfold Stage_preprocess.generatedRoot.body at hb
  extract_lets at hb
  have h3_3 := hb.2.1
  have hout := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  subst hout
  obtain ⟨w_1_0, w_2_0, w_4_0, w_5_0, w_12_0, w_17_0, w_23_0, w_28_0, w_35_0, w_40_0, w_46_0, w_51_0, w_59_0, w_64_0, w_70_0, w_75_0, w_82_0, w_87_0, w_93_0, w_98_0, w_107_0, w_110_0, w_116_0, w_120_0, -, -, h1_0, -, -, h2_0, h4_0, -, -, h5_0, -, -, -, -, h12_0, -, -, -, -, h17_0, -, -, -, -, h23_0, -, -, -, -, h28_0, -, -, -, -, h35_0, -, -, -, -, h40_0, -, -, -, -, h46_0, -, -, -, -, h51_0, -, -, -, -, h59_0, -, -, -, -, h64_0, -, -, -, -, h70_0, -, -, -, -, h75_0, -, -, -, -, h82_0, -, -, -, -, h87_0, -, -, -, -, h93_0, -, -, -, -, h98_0, h107_0, h110_0, h116_0, h120_0, hout⟩ := h3_3 i
  simp only [Prod.mk.injEq] at hout
  obtain ⟨ho0, ho1, -⟩ := hout
  funext j
  fin_cases j
  · simp only []
    rw [ho0, hashSample_eq h1_0]
    rfl
  · simp only []
    rw [ho1, hashSample_eq h2_0]
    rfl

/-! ## Inputs -/

/-- The payload secret of the model: its samples clamped to `[-1, 1]`. -/
noncomputable def payloadM (tape : SampleTape) : ErrorPoly N :=
  clampedWitness (rows := 1) (columns := 1) tape [0, 0] (.interval (-1) 1) (-1) 1 0 0

/-- The mask secret of the model: its samples clamped to `[-1, 1]`. -/
noncomputable def maskM (tape : SampleTape) : ErrorPoly N :=
  clampedWitness (rows := 1) (columns := 1) tape [1, 0] (.interval (-1) 1) (-1) 1 0 0

theorem secret_model {site : List Nat} {out : ExactMatrix Q7 N 1 1}
    (h : uniformIntervalSampleAt tape site (-1) 1 out) :
    out = (fun _ _ ↦ reducePoly Q7 N
      (clampedWitness (rows := 1) (columns := 1) tape site (.interval (-1) 1) (-1) 1 0 0)) := by
  obtain ⟨hw, -⟩ := ternarySampleAt_explicit hN h
  rw [hw, clampedWitness_eq_of_interval hN (rows := 1) (columns := 1) h]

theorem payload_model {out : ExactMatrix Q7 N 1 1}
    (h : Stage_payload.generatedRoot tape [0] GeneratedClaim.stage_0_params () out) :
    out = fun _ _ ↦ reducePoly Q7 N (payloadM tape) := by
  obtain ⟨w, hw, hout⟩ := h
  rw [hout]
  exact secret_model hw

theorem mask_model {out : ExactMatrix Q7 N 1 1}
    (h : Stage_mask.generatedRoot tape [1] GeneratedClaim.stage_1_params () out) :
    out = fun _ _ ↦ reducePoly Q7 N (maskM tape) := by
  obtain ⟨w, hw, hout⟩ := h
  rw [hout]
  exact secret_model hw

/-- An identity `X - Y + s Z + U = p v` entrywise is `v = divInt p (X - Y + s • Z + U)`. -/
theorem div_of_identity {rows columns : Nat} {p : ℤ} (hp : p ≠ 0) {sm : ErrorPoly N}
    {X Y Z U v : ErrorMatrix N rows columns}
    (h : ∀ row column, X row column - Y row column + sm * Z row column + U row column =
      ((p : ℤ) : ErrorPoly N) * v row column) :
    v = divInt p (X - Y + sm • Z + U) :=
  (divInt_eq hN hp fun row column ↦ by
    simp only [Matrix.add_apply, Matrix.sub_apply, Matrix.smul_apply, smul_eq_mul]
    exact h row column).symm

section Model

variable (model : HashModel) (key : ByteArray) (ext : GeneratedClaim.ExternalInputs)

/-! ## The plaintexts and the constants -/

/-- The plaintext `X^(2 a_k)` of member `k` of `a`. -/
noncomputable def alphaOf (k : Fin 17) : ErrorPoly N := Xe (2 * (ext.input_3 k).toNat)

/-- The selector bits of every block. -/
def selOf : Fin 5 → Fin 14 → Int := ![ext.input_6, ext.input_8, ext.input_10, ext.input_12, ext.input_14]

/-- The selector plaintext of cell `i` of block `b`. -/
noncomputable def nuOf (b : Fin 5) (i : Fin 14) : ErrorPoly N := (selOf ext b i : ErrorPoly N)

/-- The sum of the coordinates selected by the blocks before `b`. -/
def zOf (b : Nat) : Nat :=
  ((Finset.univ.filter fun b' : Fin 5 ↦ b'.val < b).sum fun b' ↦
    ∑ i : Fin 14, selOf ext b' i * ext.input_3 ⟨coords b' i, coords_lt b' i⟩).toNat

/-- The state plaintext `X^(2 z_b)` entering block `b`. -/
noncomputable def muOf (b : Nat) : ErrorPoly N := Xe (2 * zOf ext b)

/-- The native gadget of `Q_q'` rebased to `Q_q`. -/
noncomputable abbrev gReb (q q' m : Nat) : ExactMatrix q N 1 m :=
  theOutput centeredRebaseRuns (gadgetOf q' m)

end Model

end LwrProof
