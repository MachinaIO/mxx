import TopLemmas

/-!
# The decode stage

The refresh of the slot `q₀` outputs the residue `y_j mod q₀`. The decode stage subtracts the
expected row `σ A'_j₀ + A'_j₁ - Δ₀ r_j x G^st₀`, reduced from `Q₀` to `q₀`, and counts the
coefficients that threshold-decode to one modulo `q₀`. When the residue is `y_j + e'_j` modulo `q₀`
for a small integer row `e'_j`, the difference is `e'_j` and nothing decodes to one.
-/

namespace RefreshTopProof

open Mxx.Primitives MxxRuntime MxxExponentLut


/-! ## One coefficient -/

/-- A coefficient `e` with `4 |e| + 4 < q` threshold-decodes to zero. -/
theorem decode_small {q : Nat} (hq : 1 < q) (hodd : q % 2 = 1) {D : ExactMatrix q N 1 1}
    {e : ErrorPoly N} (hD : D 0 0 = reducePoly q N e) (he : ∀ i, 4 * |e.coeff i| + 4 < q)
    {position : Int} {out : Int} (h : thresholdDecode 2 8 position D out) : out = 0 := by
  obtain ⟨_, _, _, _, index, _, _, hout⟩ := h
  rw [hout, hD, reducePoly_coeff_eq hq hN]
  have := threshold_decode_value hq hodd (b := 0) (Or.inl rfl) (he index)
  simpa using this

/-! ## The residue -/

/-- `res` is the residue of `v` modulo its prime, coefficient by coefficient. -/
def Residue {q : Nat} (res : ExactMatrix q N 1 6) (v : ExactMatrix Q0 N 1 6) : Prop :=
  ∀ c i, (res 0 c).coeff i = ((((v 0 c).coeff i).val : Nat) : ZMod q)

/-- The difference of the residue of `Y + e'` and `Y` reduced to `q₀` is `e'` modulo `q₀`. -/
theorem residue_sub {Y : ExactMatrix Q0 N 1 6} {Yr res : ExactMatrix q0 N 1 6} (E : ErrorMatrix N 1 6)
    (h : Residue res (Y + reduceMatrix Q0 N 1 6 E)) (hY : modulusReduceRuns Y Yr) :
    matrixSub res Yr = reduceMatrix q0 N 1 6 E := by
  haveI : NeZero Q0 := ⟨by decide⟩
  haveI : Fact (1 < q0) := ⟨by decide⟩
  obtain ⟨-, hdvd, rfl⟩ := hY
  funext row c
  obtain rfl : row = 0 := Subsingleton.elim _ _
  apply negacyclic_ext hN
  intro i
  have hc : ∀ x : ZMod Q0, ((x.val : ℕ) : ZMod q0) = ZMod.cast x := fun x ↦ (ZMod.cast_eq_val x).symm
  simp only [matrixSub, Matrix.sub_apply, Negacyclic.coeff_sub, reduceMatrix, modulusReduce]
  rw [h c i, polynomialOfCoefficients_coeff (by decide) hN, reducePoly_coeff_eq (by decide) hN]
  simp only [Matrix.add_apply, Negacyclic.coeff_add, reduceMatrix, reducePoly_coeff_eq (by decide : 1 < Q0) hN]
  rw [hc, ZMod.cast_add hdvd, ZMod.cast_intCast hdvd]
  rw [Int.ofNat_eq_natCast, Int.cast_natCast, hc]
  ring

/-! ## Slices, the expected row, and the decode of a small difference -/

/-- A column of a small difference decodes to zero. -/
theorem column_zero (E : ErrorMatrix N 1 6) (hE : ∀ c i, 4 * |(E 0 c).coeff i| + 4 < (q0 : Int))
    {x : ExactMatrix q0 N 1 1} {a b : Int}
    (h : sliceMatrix (reduceMatrix q0 N 1 6 E) 0 1 a b x) (ha : a.toNat < 6) {position out : Int}
    (hd : thresholdDecode 2 8 position x out) : out = 0 :=
  decode_small (by decide) (by decide) (slice_entry h ha) (fun i ↦ hE _ i) hd

/-- σ₀ as the decode stage builds it from its coefficients. -/
noncomputable def sigmaPoly (s : Fin 8 → Int) : ExactMatrix Q0 N 1 1 :=
  matrixAdd (matrixAdd (matrixAdd (matrixAdd (matrixAdd (matrixAdd (matrixAdd
    (multiplyMonomial (liftInteger (s 0)) 0) (multiplyMonomial (liftInteger (s 1)) 1))
    (multiplyMonomial (liftInteger (s 2)) 2)) (multiplyMonomial (liftInteger (s 3)) 3))
    (multiplyMonomial (liftInteger (s 4)) 4)) (multiplyMonomial (liftInteger (s 5)) 5))
    (multiplyMonomial (liftInteger (s 6)) 6)) (multiplyMonomial (liftInteger (s 7)) 7)

/-- The expected row `σ A₀ + A₁ - Δ₀ r x G^st₀` of the decode stage. -/
noncomputable def expectedRow (A : ExactMatrix Q0 N 2 6) (σ r : ExactMatrix Q0 N 1 1) (c e : Int)
    (G : ExactMatrix Q1 N 1 6) : ExactMatrix Q0 N 1 6 :=
  matrixSub (matrixAdd (matrixMulScalarLeft σ (fun _ k ↦ A 0 k)) (fun _ k ↦ A 1 k))
    (matrixMulScalarLeft (matrixMulScalarLeft (matrixMulScalarLeft
      (matrixPolynomial [1073740529]) r) (multiplyMonomial (liftInteger c) e))
      (centeredRebase G))

end RefreshTopProof
