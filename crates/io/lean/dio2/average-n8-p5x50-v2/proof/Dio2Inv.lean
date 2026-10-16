import ExponentEncoding

/-!
# Root rows

A step input of Diamond iO 2 is a set of root rows at `Q_0` (`RnsBggRoot::encode`):

`c_j = s̄ K_(0j) + K_(1j) - T_j G + e_j`,

under the mask `(s̄, 1)`, the root key `K` of the row's tag (four blocks hashed under the step's
root hash key) and the gadget `G` of `Q_0`, with the target `T_j` and an error of coefficients at
most one (the obfuscator's `u - u'` for uniform bits `u, u'`, or the CBD(1) noise of a refresh).
This is `Rel s̄ K (T • G) c 1`.
-/

namespace Dio2Proof

open Mxx.Primitives MxxRuntime MxxExponentLut

variable {n q m : Nat}

/-- The difference of two uniform bits has coefficients in `[-1, 1]`. -/
theorem uniform_bits_sub {u u' : ExactMatrix q n 1 m} (hu : uniformIntervalSample 0 1 u)
    (hu' : uniformIntervalSample 0 1 u') :
    ∃ e : ErrorMatrix n 1 m, CoeffBound e 1 ∧ u - u' = reduceMatrix q n 1 m e := by
  obtain ⟨-, w, rfl, hw⟩ := hu
  obtain ⟨-, w', rfl, hw'⟩ := hu'
  refine ⟨w - w', fun row column coefficient ↦ ?_, (reduceMatrix_sub w w').symm⟩
  have h1 := hw row column coefficient
  have h2 := hw' row column coefficient
  simp only [Matrix.sub_apply, Negacyclic.coeff_sub]
  omega

/-- A root row encoded by the obfuscator: `s K_(0j) + 1 K_(1j) - T_j G + (u_j - u'_j)` with the
mask `s = reduce s̄` and uniform bits `u, u'` is an encoding of `T • G` under `K`, error one. -/
theorem encode_rel {s : ExactMatrix q n 1 1} {sm : ErrorPoly n}
    (hs : s 0 0 = reducePoly q n sm) {K : Fin 2 → Fin 2 → ExactMatrix q n 1 m}
    {T : Fin 2 → ExactMatrix q n 1 1} {G : ExactMatrix q n 1 m}
    {u u' c : Fin 2 → ExactMatrix q n 1 m}
    (hu : ∀ j, uniformIntervalSample 0 1 (u j)) (hu' : ∀ j, uniformIntervalSample 0 1 (u' j))
    (hc : ∀ j, c j = matrixAdd (matrixSub (matrixAdd (matrixMulScalarLeft s (K 0 j))
      (matrixMulScalarLeft (1 : ExactMatrix q n 1 1) (K 1 j))) (matrixMulScalarLeft (T j) G))
      (matrixSub (u j) (u' j))) :
    Rel sm K (fun j ↦ matrixMulScalarLeft (T j) G) c 1 := by
  choose e he hue using fun j ↦ uniform_bits_sub (hu j) (hu' j)
  refine ⟨e, he, fun j ↦ ?_⟩
  rw [hc j, ← hue j]
  funext row column
  simp only [matrixAdd, matrixSub, matrixMulScalarLeft, Matrix.add_apply, Matrix.sub_apply,
    Matrix.smul_apply, smul_eq_mul, hs]
  simp

/-- The bytes of the tag component `"component"`. -/
def componentBytes : Blob := [99, 111, 109, 112, 111, 110, 101, 110, 116]

/-- Block `(k, j)` of the root key of the root object tagged `(pre, comps)` under the root hash
key `key` (`RnsBggRoot::key`: component `2 k + j` of the tag, a `1 × m` row at `q`). -/
noncomputable def rootKey (model : HashModel) (key : ByteArray) (pre : Blob)
    (comps : List HashTagComponent) (q m : Nat) (k j : Fin 2) : ExactMatrix q n 1 m :=
  model.sample q n 1 m key
    (completeHashTag pre (comps ++ [.bytes componentBytes, .u64Le (2 * k.val + j.val)]))

/-- A hash sample is the model's sample at its key and tag. -/
theorem hashSample_eq {model : HashModel} {pre : Blob} {comps : List HashTagComponent}
    {key : ByteArray} {out : ExactMatrix q n 1 m} (h : hashSample model pre comps key out) :
    out = model.sample q n 1 m key (completeHashTag pre comps) :=
  h.2.2

/-- A root row: the encoding of the targets `T_j G` under the root key of its tag and the mask
`(s̄, 1)`, with error at most one. -/
def RootRow (model : HashModel) (key : ByteArray) (pre : Blob) (comps : List HashTagComponent)
    (sm : ErrorPoly n) (G : ExactMatrix q n 1 m) (T : Fin 2 → ExactMatrix q n 1 1)
    (c : Fin 2 → ExactMatrix q n 1 m) : Prop :=
  Rel sm (rootKey model key pre comps q m) (fun j ↦ matrixMulScalarLeft (T j) G) c 1

end Dio2Proof
