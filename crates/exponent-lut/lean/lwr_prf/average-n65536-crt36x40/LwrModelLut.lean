import LwrModel33
import LwrLutE

/-!
# The model of the readout LUT

The switching ciphertexts of the LUT (the hash-derived tops and the bottoms of their payload
relation), the summed key, error, target and row of the 128 branches, the numerator and the
error of the ModDown to `Q1`, and the finalized error.
-/

namespace LwrProof

open Mxx.Primitives MxxRuntime MxxExponentLut

section ModelLut

variable (model : HashModel) (key : ByteArray) (ext : GeneratedClaim.ExternalInputs)

/-- The bottom `B_{1j}` of the switching ciphertext of branch `i`. -/
noncomputable def lutSwitchBot (tape : SampleTape) (i : Fin 128) (j : Fin 2) : ExactMatrix Q1 N 1 4 :=
  (lutCoefficient Q1 inverse1 (sigma i).toNat * reducePoly Q1 N (unitPair (payloadM tape) j)) •
      gadgetOf Q1 4 + reduceMatrix Q1 N 1 4 (switchError tape [2] i j) -
    reducePoly Q1 N (substitution (sigma i).toNat (sigma_odd i) (payloadM tape)) • lutB model key i j

/-- The switching ciphertexts of the LUT, indexed by row, component and branch. -/
noncomputable def lutFam (tape : SampleTape) : Fin 2 → Fin 2 → Fin 128 → ExactMatrix Q1 N 1 4 :=
  fun k j i ↦ ![lutB model key i j, lutSwitchBot model key tape i j] k

/-- The summed helper keys of the 128 branches. -/
noncomputable def lutKsum (tape : SampleTape) : Fin 2 → Fin 2 → ExactMatrix Q2 N 1 4 :=
  ∑ i : Fin 128, lutKeyK model key (key34 model key ext tape) (lutFam model key tape) i

/-- The summed error of the 128 branches. -/
noncomputable def lutErr (tape : SampleTape) : Fin 2 → ErrorMatrix N 1 4 :=
  ∑ i : Fin 128, lutBranchErr (maskM tape) (err34 model key ext tape) (fun i ↦ helperError tape [3] i)
    (key34 model key ext tape) (lutFam model key tape) i

/-- The summed target of the 128 branches. -/
noncomputable def lutTsum (tape : SampleTape) : Fin 2 → ExactMatrix Q2 N 1 4 :=
  fun j ↦ ∑ i : Fin 128, scaleUpMatrix Q2 P2
    (reducePoly Q1 N (substitution (sigma i).toNat (sigma_odd i) (muOf ext 34)) •
      (reducePoly Q1 N (substitution (sigma i).toNat (sigma_odd i) (payloadM tape)) •
        lutFam model key tape 0 j i + lutFam model key tape 1 j i))

/-- The summed row of the 128 branches. -/
noncomputable def lutRow (tape : SampleTape) : Fin 2 → ExactMatrix Q2 N 1 4 :=
  encRow (maskM tape) (lutKsum model key ext tape) (lutTsum model key ext tape) (lutErr model key ext tape)

/-- The numerator of the LUT ModDown to `Q1`. -/
noncomputable def lutNumer (tape : SampleTape) (j : Fin 2) : ErrorMatrix N 1 4 :=
  lutErr model key ext tape j - downResidue P2 (lutRow model key ext tape j) +
    maskM tape • downResidue P2 (lutKsum model key ext tape 0 j) +
    downResidue P2 (lutKsum model key ext tape 1 j)

/-- The error of the LUT after its ModDown to `Q1`. -/
noncomputable def lutV (tape : SampleTape) (j : Fin 2) : ErrorMatrix N 1 4 :=
  divInt P2 (lutNumer model key ext tape j)

/-- The numerator of the LUT is divisible by the dropped factor. -/
def divLut (tape : SampleTape) : Prop :=
  ∀ j row column, lutNumer model key ext tape j row column =
    ((P2 : Int) : ErrorPoly N) * divInt P2 (lutNumer model key ext tape j) row column

/-- The finalized error of the readout. -/
noncomputable def finalErr (tape : SampleTape) (j : Fin 2) : ErrorMatrix N 1 1 :=
  (lutV model key ext tape j - ∑ i : Fin 128,
    substitution (sigma i).toNat (sigma_odd i) (muOf ext 34) • switchError tape [2] i j) * finalDigits

end ModelLut

end LwrProof
