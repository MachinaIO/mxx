import ExponentScale
import ExponentAutomorphism
import ExponentSampling

/-!
# Runtime relations as equations

Generated graphs state their runtime operations as relations. For every program and parameter set,
a hash sample is the hash model's matrix, a family read returns the member at its index, a reduced
polynomial reduces to its image at a divisor modulus, a ternary sample is the reduction of an
integer polynomial of norm at most one, a Gaussian sample is the reduction of an integer matrix
within its cutoff, and the plaintext `c X^e` of the plaintext bindings is a monomial.
-/

namespace MxxExponentLut

open Mxx.Primitives MxxRuntime

variable {n : Nat}

theorem hashSample_eq {q rows columns : Nat} {model : HashModel} {tagPrefix : Blob}
    {components : List HashTagComponent} {key : ByteArray}
    {output : ExactMatrix q n rows columns} (h : hashSample model tagPrefix components key output) :
    output = model.sample q n rows columns key (completeHashTag tagPrefix components) := h.2.2

theorem matrixMulScalarLeft_eq {q rows columns : Nat} (a : ExactMatrix q n 1 1)
    (x : ExactMatrix q n rows columns) : matrixMulScalarLeft a x = a 0 0 • x := rfl

theorem matrixMulScalarRight_eq {q rows columns : Nat} (x : ExactMatrix q n rows columns)
    (a : ExactMatrix q n 1 1) : matrixMulScalarRight x a = a 0 0 • x := by
  funext i j
  simp [matrixMulScalarRight, mul_comm]

/-- Reduction of a reduced integer polynomial to a divisor modulus. -/
theorem modulusReduce_poly {q p : Nat} (hq : 1 < q) (hp : 1 < p) (hn : 0 < n) {s : ErrorPoly n}
    {out : ExactMatrix p n 1 1}
    (h : modulusReduceRuns (fun _ _ ↦ reducePoly q n s : ExactMatrix q n 1 1) out) :
    out = fun _ _ ↦ reducePoly p n s :=
  modulusReduceRuns_reduce hq hp hn h (fun _ _ ↦ s) rfl

theorem modulusReduce_one {q p : Nat} (hq : 1 < q) (hp : 1 < p) (hn : 0 < n)
    {out : ExactMatrix p n 1 1} (h : modulusReduceRuns (1 : ExactMatrix q n 1 1) out) : out = 1 := by
  have hone : (1 : ExactMatrix q n 1 1) = fun _ _ ↦ reducePoly q n 1 := by
    funext i j
    rw [Subsingleton.elim i 0, Subsingleton.elim j 0]
    simp
  rw [hone] at h
  rw [modulusReduce_poly hq hp hn h]
  funext i j
  rw [Subsingleton.elim i 0, Subsingleton.elim j 0]
  simp

/-- A family read at a natural-number index. -/
theorem familyGet_eq {α : Type} {count k : Nat} {f : Fin count → α} {out : α}
    (h : familyGetDynamic f (Int.ofNat k) out) : ∃ hk : k < count, out = f ⟨k, hk⟩ := by
  obtain ⟨position, hposition, hout⟩ := h
  have hk : position.val = k := Int.ofNat.inj hposition
  subst hk
  exact ⟨position.isLt, hout⟩

/-- A ternary sample as one integer polynomial. -/
theorem ternary_poly {q : Nat} {output : ExactMatrix q n 1 1}
    (h : uniformIntervalSample (-1) 1 output) :
    ∃ s : ErrorPoly n, output = (fun _ _ ↦ reducePoly q n s) ∧ polyNorm s ≤ 1 := by
  obtain ⟨w, hw, hb⟩ := uniformIntervalSample_ternary h
  refine ⟨w 0 0, ?_, hb 0 0⟩
  funext i j
  rw [hw, Subsingleton.elim i 0, Subsingleton.elim j 0]
  rfl

/-- A Gaussian sample as an integer matrix within its cutoff. -/
theorem gaussian_bound {q rows columns : Nat} {sigma : Rat} {cutoff : Int}
    {output : ExactMatrix q n rows columns} (h : gaussianSample sigma cutoff output) :
    ∃ e : ErrorMatrix n rows columns, output = reduceMatrix q n rows columns e ∧
      CoeffBound e cutoff.toNat := by
  obtain ⟨w, hw, hb⟩ := gaussianSample_bounded h
  exact ⟨w, hw, fun i j k ↦ (coeff_natAbs_le_polyNorm _ k).trans (hb i j)⟩

/-- A ternary tape sample as one integer polynomial. -/
theorem ternary_poly_at (hn : 0 < n) {q : Nat} {tape : SampleTape} {site : List Nat}
    {output : ExactMatrix q n 1 1} (h : uniformIntervalSampleAt tape site (-1) 1 output) :
    ∃ s : ErrorPoly n, output = (fun _ _ ↦ reducePoly q n s) ∧ polyNorm s ≤ 1 :=
  ternary_poly (uniformIntervalSampleAt_sample hn h)

/-- A Gaussian tape sample as an integer matrix within its cutoff. -/
theorem gaussian_bound_at (hn : 0 < n) {q rows columns : Nat} {tape : SampleTape} {site : List Nat}
    {sigma : Rat} {cutoff : Int} {output : ExactMatrix q n rows columns}
    (h : gaussianSampleAt tape site sigma cutoff output) :
    ∃ e : ErrorMatrix n rows columns, output = reduceMatrix q n rows columns e ∧
      CoeffBound e cutoff.toNat :=
  gaussian_bound (gaussianSampleAt_sample hn h)

/-- The plaintext `c X^e`. -/
noncomputable def message (coefficient exponent : Int) : ErrorPoly n :=
  (coefficient : ErrorPoly n) *
    AdjoinRoot.root (negacyclicModulus n Int) ^ (exponent % (2 * n : Nat)).toNat

theorem multiplyMonomial_liftInteger {q : Nat} (coefficient exponent : Int) :
    multiplyMonomial (liftInteger coefficient : ExactMatrix q n 1 1) exponent =
      fun _ _ ↦ reducePoly q n (message coefficient exponent) := by
  funext i j
  simp [multiplyMonomial, liftInteger, message, reducePoly]

/-- The negacyclic root `X` at modulus `q`. -/
noncomputable abbrev X (q : Nat) : ExactPoly q n := AdjoinRoot.root (negacyclicModulus n (ZMod q))

/-- The monomial `X^k` with integer coefficients. -/
noncomputable abbrev Xe (k : Nat) : ErrorPoly n := AdjoinRoot.root (negacyclicModulus n Int) ^ k

theorem polyNorm_Xe (hn : 1 < n) (k : Nat) : polyNorm (Xe (n := n) k) ≤ 1 := by
  have h := polyNorm_root_pow_mul_le (n := n) hn k 1
  rw [mul_one] at h
  refine h.trans (polyNorm_le_of_coeff fun i ↦ ?_)
  have h1 := Negacyclic.coeff_root_pow (R := Int) (n := n) (by omega) ⟨0, by omega⟩ i
  rw [show ((⟨0, by omega⟩ : Fin n) : Nat) = 0 from rfl, pow_zero] at h1
  rw [h1]
  split <;> simp

theorem reducePoly_Xe {q : Nat} (k : Nat) : reducePoly q n (Xe k) = X q ^ k := by
  rw [map_pow]
  congr 1
  simp [reducePoly]

/-- A plaintext `1 · X^e` with `0 ≤ e < 2n` is the monomial `X^e`. -/
theorem message_one {c e : Int} {k : Nat} (hc : c = 1) (he : e = k) (hk : k < 2 * n) :
    message (n := n) c e = Xe k := by
  subst hc he
  rw [message, Int.cast_one, one_mul, Int.emod_eq_of_lt (by omega) (by omega), Int.toNat_natCast]

theorem natCast_self_poly (q : Nat) : (q : ExactPoly q n) = 0 := by
  rw [← map_natCast (algebraMap (ZMod q) (ExactPoly q n)), ZMod.natCast_self, map_zero]

theorem inverse_mul {q inverse width k : Nat} (h : inverse * width = 1 + q * k) :
    (inverse : ExactPoly q n) * ((width : Nat) : ExactPoly q n) = 1 := by
  rw [← Nat.cast_mul, h, Nat.cast_add, Nat.cast_mul, natCast_self_poly, zero_mul, add_zero,
    Nat.cast_one]

/-- The gadget matrix of a regular layout. -/
theorem gadget_exists {q m : Nat} {backend : BackendContext} {base : Int}
    {layout : RegularLayout q} (hlayout : backend.regularLayout q n = some layout)
    (hbase : base = layout.base) (hdigits : (m : Int) = layout.digitCount)
    (hwidth : m = 1 * layout.digitCount) :
    ∃ g : ExactMatrix q n 1 m, gadgetMatrixRuns backend base m g :=
  ⟨_, layout, hlayout, hbase, hdigits, hwidth, rfl⟩

end MxxExponentLut
