import MxxIR
import MxxRuntime
import BggMultiplication
import ExponentEncoding
import ExponentDecode
import ExponentSampling
import ExponentPublic

/-!
# Encodings through one layer of the Boolean BGG+ evaluation

`Enc s sG c A x E` states that `c = s A - x (s G) + e` with every coefficient of the integer
error `e` at most `E`, where `sG = s G` and `x` is a constant plaintext. The lemmas follow the six
candidates that `evaluate_boolean_encoding_layers` computes for every slot: the encodings of zero,
one, the left operand, its negation, the product, and the XOR, each with the error bound of the
corresponding row of `mxx_khe::noise_simulate::boolean`.
-/

namespace RingGswBgg

open Mxx.Primitives MxxRuntime

variable {q n m : Nat}

/-- The plaintext matrix of a bit. -/
noncomputable def bitMatrix (b : Bool) : ExactMatrix q n 1 1 :=
  liftInteger (if b then 1 else 0)

/-- `c` encodes `x` under `A` with an error of coefficients at most `E`. -/
def Enc (s : ExactMatrix q n 1 2) (sG : ExactMatrix q n 1 m) (c : ExactMatrix q n 1 m)
    (A : ExactMatrix q n 2 m) (x : ExactMatrix q n 1 1) (E : Nat) : Prop :=
  ∃ e : ErrorMatrix n 1 m,
    c = s * A - matrixMulScalarLeft x sG + reduceMatrix q n 1 m e ∧ CoeffBound e E

theorem matrixMulScalarLeft_eq_smul {r c : Nat} (x : ExactMatrix q n 1 1)
    (M : ExactMatrix q n r c) : matrixMulScalarLeft x M = x 0 0 • M := by
  ext i j
  simp [matrixMulScalarLeft]

theorem matrixMulScalarRight_eq_smul {r c : Nat} (M : ExactMatrix q n r c)
    (x : ExactMatrix q n 1 1) : matrixMulScalarRight M x = x 0 0 • M := by
  ext i j
  simp [matrixMulScalarRight, mul_comm]

theorem Enc.mono {s : ExactMatrix q n 1 2} {sG : ExactMatrix q n 1 m}
    {c : ExactMatrix q n 1 m} {A : ExactMatrix q n 2 m} {x : ExactMatrix q n 1 1} {E E' : Nat}
    (h : Enc s sG c A x E) (hle : E ≤ E') : Enc s sG c A x E' := by
  obtain ⟨e, he, hb⟩ := h
  exact ⟨e, he, fun r c k ↦ (hb r c k).trans hle⟩

theorem Enc.add {s : ExactMatrix q n 1 2} {sG : ExactMatrix q n 1 m}
    {c₁ c₂ : ExactMatrix q n 1 m} {A₁ A₂ : ExactMatrix q n 2 m} {x₁ x₂ : ExactMatrix q n 1 1}
    {E₁ E₂ : Nat} (h₁ : Enc s sG c₁ A₁ x₁ E₁) (h₂ : Enc s sG c₂ A₂ x₂ E₂) :
    Enc s sG (matrixAdd c₁ c₂) (matrixAdd A₁ A₂) (matrixAdd x₁ x₂) (E₁ + E₂) := by
  obtain ⟨e₁, he₁, hb₁⟩ := h₁
  obtain ⟨e₂, he₂, hb₂⟩ := h₂
  refine ⟨e₁ + e₂, ?_, coeffBound_add hb₁ hb₂⟩
  simp only [matrixAdd, matrixMulScalarLeft_eq_smul, he₁, he₂, reduceMatrix_add, Matrix.mul_add,
    Matrix.add_apply, add_smul]
  abel

theorem coeffBound_neg {rows columns : Nat} {e : ErrorMatrix n rows columns} {E : Nat}
    (h : CoeffBound e E) : CoeffBound (-e) E := fun r c k ↦ by
  simpa only [Matrix.neg_apply, Negacyclic.coeff_neg, Int.natAbs_neg] using h r c k

theorem Enc.sub {s : ExactMatrix q n 1 2} {sG : ExactMatrix q n 1 m}
    {c₁ c₂ : ExactMatrix q n 1 m} {A₁ A₂ : ExactMatrix q n 2 m} {x₁ x₂ : ExactMatrix q n 1 1}
    {E₁ E₂ : Nat} (h₁ : Enc s sG c₁ A₁ x₁ E₁) (h₂ : Enc s sG c₂ A₂ x₂ E₂) :
    Enc s sG (matrixSub c₁ c₂) (matrixSub A₁ A₂) (matrixSub x₁ x₂) (E₁ + E₂) := by
  obtain ⟨e₁, he₁, hb₁⟩ := h₁
  obtain ⟨e₂, he₂, hb₂⟩ := h₂
  refine ⟨e₁ - e₂, ?_, ?_⟩
  · simp only [matrixSub, matrixMulScalarLeft_eq_smul, he₁, he₂, MxxExponentLut.reduceMatrix_sub,
      Matrix.mul_sub, Matrix.sub_apply, sub_smul]
    abel
  · rw [sub_eq_add_neg]
    exact coeffBound_add hb₁ (coeffBound_neg hb₂)

theorem bitMatrix_apply (b : Bool) :
    (bitMatrix b : ExactMatrix q n 1 1) 0 0 = if b then 1 else 0 := by
  cases b <;> simp [bitMatrix, liftInteger]

/-- The BGG+ product `c_L D + x_L c_R` of encodings, for an exact decomposition `G D = A_R` with
digits at most `β` and a bit plaintext `x_L`: the error is `e_L d + x_L e_R`. -/
theorem Enc.mul {s : ExactMatrix q n 1 2} {sG : ExactMatrix q n 1 m} {G : ExactMatrix q n 2 m}
    (hsG : sG = s * G) (hn : 0 < n)
    {cL cR : ExactMatrix q n 1 m} {AL AR : ExactMatrix q n 2 m} {xL xR : ExactMatrix q n 1 1}
    {EL ER β : Nat} {D : ExactMatrix q n m m} {d : ErrorMatrix n m m}
    (hL : Enc s sG cL AL xL EL) (hR : Enc s sG cR AR xR ER) (hGD : G * D = AR)
    (hD : D = reduceMatrix q n m m d) (hd : CoeffBound d β) {b : Bool} (hx : xL = bitMatrix b) :
    Enc s sG (matrixAdd (matrixMul cL D) (matrixMulScalarRight cR xL)) (matrixMul AL D)
      (matrixMulScalarLeft xL xR) (m * n * EL * β + ER) := by
  obtain ⟨eL, heL, hbL⟩ := hL
  obtain ⟨eR, heR, hbR⟩ := hR
  have hp : xL 0 0 = if b then 1 else 0 := by rw [hx, bitMatrix_apply]
  refine ⟨eL * d + (if b then eR else 0), ?_, ?_⟩
  · have hred : reduceMatrix q n 1 m (if b then eR else 0) = xL 0 0 • reduceMatrix q n 1 m eR := by
      rw [hp]
      cases b
      · funext r c
        simp [reduceMatrix_apply]
      · simp
    have hsGD : sG * D = s * AR := by rw [hsG, Matrix.mul_assoc, hGD]
    simp only [matrixAdd, matrixMul]
    rw [heL, heR, reduceMatrix_add, reduceMatrix_mul, ← hD, hred]
    simp only [matrixMulScalarRight_eq_smul, matrixMulScalarLeft_eq_smul, Matrix.add_mul,
      Matrix.sub_mul, Matrix.smul_mul, Matrix.mul_assoc, hsGD, smul_sub, smul_add, smul_smul,
      Matrix.smul_apply, smul_eq_mul]
    abel
  · refine coeffBound_add (coeffBound_mul hn hbL hd) ?_
    cases b
    · intro r c k; simp
    · exact hbR

/-- Doubling an encoding, as the XOR candidate does with the product: the error doubles. -/
theorem Enc.double {s : ExactMatrix q n 1 2} {sG : ExactMatrix q n 1 m}
    {c : ExactMatrix q n 1 m} {A : ExactMatrix q n 2 m} {x : ExactMatrix q n 1 1} {E : Nat}
    (h : Enc s sG c A x E) {two : ExactMatrix q n 1 1} (htwo : two 0 0 = 2) :
    Enc s sG (matrixMulScalarRight c two) (matrixMulScalarRight A two)
      (matrixMulScalarLeft x two) (E + E) := by
  obtain ⟨e, he, hb⟩ := h
  refine ⟨e + e, ?_, coeffBound_add hb hb⟩
  rw [he, reduceMatrix_add]
  simp only [matrixMulScalarRight_eq_smul, matrixMulScalarLeft_eq_smul, Matrix.smul_apply,
    smul_eq_mul, htwo, mul_two, add_smul, smul_sub, smul_add, two_smul, Matrix.mul_add]

/-! ## Bit plaintexts and the ideal gates -/

theorem one_eq_bitMatrix : (1 : ExactMatrix q n 1 1) = bitMatrix true := by
  ext i j
  fin_cases i; fin_cases j
  simp [bitMatrix, liftInteger]

theorem bitMatrix_sub_self (x : ExactMatrix q n 1 1) : matrixSub x x = bitMatrix false := by
  ext i j
  fin_cases i; fin_cases j
  simp [matrixSub, bitMatrix, liftInteger]

theorem bitMatrix_not (b : Bool) :
    matrixSub (1 : ExactMatrix q n 1 1) (bitMatrix b) = bitMatrix (!b) := by
  ext i j
  fin_cases i; fin_cases j
  cases b <;> simp [matrixSub, bitMatrix, liftInteger]

theorem bitMatrix_and (a b : Bool) :
    matrixMulScalarLeft (bitMatrix a : ExactMatrix q n 1 1) (bitMatrix b) = bitMatrix (a && b) := by
  ext i j
  fin_cases i; fin_cases j
  cases a <;> cases b <;> simp [matrixMulScalarLeft, bitMatrix, liftInteger]

theorem bitMatrix_xor (a b : Bool) {two : ExactMatrix q n 1 1} (htwo : two 0 0 = 2) :
    matrixSub (matrixAdd (bitMatrix a) (bitMatrix b))
        (matrixMulScalarLeft (matrixMulScalarLeft (bitMatrix a) (bitMatrix b)) two) =
      (bitMatrix (xor a b) : ExactMatrix q n 1 1) := by
  funext i j
  fin_cases i; fin_cases j
  simp only [matrixSub, matrixAdd, matrixMulScalarLeft, Matrix.sub_apply, Matrix.add_apply]
  cases a <;> cases b <;> (simp [bitMatrix, liftInteger, htwo]; try norm_num)

theorem liftInteger_bit {v : Int} (h : 0 ≤ v ∧ v ≤ 1) :
    (liftInteger v : ExactMatrix q n 1 1) = bitMatrix (decide (v = 1)) := by
  have : v = 0 ∨ v = 1 := by omega
  rcases this with rfl | rfl <;> simp [bitMatrix]

/-- The Boolean candidates of the ideal evaluation, as the generated graph computes them. -/
theorem ideal_candidates (a b : Bool) :
    decide ((if false then (1 : Int) else 0) = 1) = false ∧
      decide ((if true then (1 : Int) else 0) = 1) = true ∧
      decide ((if a then (1 : Int) else 0) = 0) = !a ∧
      decide ((if a then (1 : Int) else 0) + (if b then 1 else 0) = 2) = (a && b) ∧
      decide ((if a then (1 : Int) else 0) + (if b then 1 else 0) = 1) = xor a b := by
  cases a <;> cases b <;> decide

/-! ## Selection -/

theorem select_two {α : Type} {k : Int} {a b o : α} (h : select k [a, b] o) :
    (k = 0 ∧ o = a) ∨ (k = 1 ∧ o = b) := by
  obtain ⟨⟨i, hi⟩, hk, ho⟩ := h
  simp only [List.length_cons, List.length_nil] at hi
  interval_cases i
  · exact Or.inl ⟨hk.symm, ho⟩
  · exact Or.inr ⟨hk.symm, ho⟩

theorem select_six {α : Type} {k : Int} {a₀ a₁ a₂ a₃ a₄ a₅ o : α}
    (h : select k [a₀, a₁, a₂, a₃, a₄, a₅] o) :
    (k = 0 ∧ o = a₀) ∨ (k = 1 ∧ o = a₁) ∨ (k = 2 ∧ o = a₂) ∨ (k = 3 ∧ o = a₃) ∨
      (k = 4 ∧ o = a₄) ∨ (k = 5 ∧ o = a₅) := by
  obtain ⟨⟨i, hi⟩, hk, ho⟩ := h
  simp only [List.length_cons, List.length_nil] at hi
  interval_cases i
  · exact Or.inl ⟨hk.symm, ho⟩
  · exact Or.inr (Or.inl ⟨hk.symm, ho⟩)
  · exact Or.inr (Or.inr (Or.inl ⟨hk.symm, ho⟩))
  · exact Or.inr (Or.inr (Or.inr (Or.inl ⟨hk.symm, ho⟩)))
  · exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inl ⟨hk.symm, ho⟩))))
  · exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inr ⟨hk.symm, ho⟩))))

theorem familyGetDynamic_eq {α : Type} {count : Nat} {f : Fin count → α} {k : Int} {a b : α}
    (ha : familyGetDynamic f k a) (hb : familyGetDynamic f k b) : a = b := by
  obtain ⟨i, hi, rfl⟩ := ha
  obtain ⟨j, hj, rfl⟩ := hb
  rw [Fin.ext (by omega : i.val = j.val)]

/-! ## One slot of a layer -/

/-- The error bound of a layer whose operands' errors are at most `E`, for digits at most `β`:
the XOR candidate `2 E + 2 (m n β E + E)`. -/
def nextBound (m n β E : Nat) : Nat := (2 * (m * n * β) + 4) * E

theorem le_nextBound {m n β E X : Nat} (h : X ≤ 4 * E) : X ≤ nextBound m n β E := by
  unfold nextBound
  rw [Nat.add_mul]
  omega

theorem product_le_nextBound (m n β E : Nat) : m * n * E * β + E ≤ nextBound m n β E := by
  unfold nextBound
  have : m * n * E * β = m * n * β * E := by ring
  rw [this, Nat.add_mul, Nat.mul_assoc 2]
  omega

theorem xor_le_nextBound (m n β E : Nat) :
    E + E + (m * n * E * β + E + (m * n * E * β + E)) ≤ nextBound m n β E := by
  unfold nextBound
  have : m * n * E * β = m * n * β * E := by ring
  rw [this, Nat.add_mul, Nat.mul_assoc 2]
  omega

/-- One slot of a layer: whichever gate the slot selects, and whether it is active, its encoding
has the error bound `nextBound` and encodes the bit of the ideal gate. -/
theorem slot_layer {s : ExactMatrix q n 1 2} {sG : ExactMatrix q n 1 m} {G : ExactMatrix q n 2 m}
    (hsG : sG = s * G) (hn : 0 < n) {β E E₀ : Nat} (hE₀ : E₀ ≤ E)
    {c₁ : ExactMatrix q n 1 m} {A₁ : ExactMatrix q n 2 m} {one : ExactMatrix q n 1 1}
    (hone : Enc s sG c₁ A₁ one E₀) (hone1 : one = 1)
    {cL cR : ExactMatrix q n 1 m} {AL AR : ExactMatrix q n 2 m} {xL xR : ExactMatrix q n 1 1}
    {bl br : Bool} (hL : Enc s sG cL AL xL E) (hxL : xL = bitMatrix bl)
    (hR : Enc s sG cR AR xR E) (hxR : xR = bitMatrix br)
    {D : ExactMatrix q n m m} {d : ErrorMatrix n m m} (hGD : G * D = AR)
    (hD : D = reduceMatrix q n m m d) (hd : CoeffBound d β)
    {two : ExactMatrix q n 1 1} (htwo : two 0 0 = 2)
    {k act : Int} {v vo : ExactMatrix q n 1 m} {p po : ExactMatrix q n 2 m}
    {x xo : ExactMatrix q n 1 1} {b bo : Bool}
    (hv : select k [matrixSub c₁ c₁, c₁, cL, matrixSub c₁ cL,
      matrixAdd (matrixMul cL D) (matrixMulScalarRight cR xL),
      matrixSub (matrixAdd cL cR)
        (matrixMulScalarRight (matrixAdd (matrixMul cL D) (matrixMulScalarRight cR xL)) two)] v)
    (hp : select k [matrixSub A₁ A₁, A₁, AL, matrixSub A₁ AL, matrixMul AL D,
      matrixSub (matrixAdd AL AR) (matrixMulScalarRight (matrixMul AL D) two)] p)
    (hx : select k [matrixSub one one, one, xL, matrixSub one xL, matrixMulScalarLeft xL xR,
      matrixSub (matrixAdd xL xR) (matrixMulScalarLeft (matrixMulScalarLeft xL xR) two)] x)
    (hb : select k [false, true, bl, !bl, bl && br, xor bl br] b)
    (hva : select act [matrixSub c₁ c₁, v] vo) (hpa : select act [matrixSub A₁ A₁, p] po)
    (hxa : select act [matrixSub one one, x] xo) (hba : select act [false, b] bo) :
    Enc s sG vo po xo (nextBound m n β E) ∧ xo = bitMatrix bo := by
  have hα := product_le_nextBound m n β E
  have hzero : Enc s sG (matrixSub c₁ c₁) (matrixSub A₁ A₁) (matrixSub one one)
      (nextBound m n β E) :=
    (Enc.sub hone hone).mono (le_nextBound (by omega))
  have hzeroBit : matrixSub one one = bitMatrix false := bitMatrix_sub_self one
  have hand := Enc.mul hsG hn hL hR hGD hD hd hxL
  have hselected : Enc s sG v p x (nextBound m n β E) ∧ x = bitMatrix b := by
    rcases select_six hv with ⟨hk, rfl⟩ | ⟨hk, rfl⟩ | ⟨hk, rfl⟩ | ⟨hk, rfl⟩ | ⟨hk, rfl⟩ |
        ⟨hk, rfl⟩ <;>
      subst hk <;>
      rcases select_six hp with ⟨h1, rfl⟩ | ⟨h1, rfl⟩ | ⟨h1, rfl⟩ | ⟨h1, rfl⟩ | ⟨h1, rfl⟩ |
        ⟨h1, rfl⟩ <;> norm_num at h1 <;>
      rcases select_six hx with ⟨h2, rfl⟩ | ⟨h2, rfl⟩ | ⟨h2, rfl⟩ | ⟨h2, rfl⟩ | ⟨h2, rfl⟩ |
        ⟨h2, rfl⟩ <;> norm_num at h2 <;>
      rcases select_six hb with ⟨h3, rfl⟩ | ⟨h3, rfl⟩ | ⟨h3, rfl⟩ | ⟨h3, rfl⟩ | ⟨h3, rfl⟩ |
        ⟨h3, rfl⟩ <;> norm_num at h3
    · exact ⟨hzero, hzeroBit⟩
    · exact ⟨hone.mono (le_nextBound (by omega)), by rw [hone1, one_eq_bitMatrix]⟩
    · exact ⟨hL.mono (le_nextBound (by omega)), hxL⟩
    · refine ⟨(Enc.sub hone hL).mono (le_nextBound (by omega)), ?_⟩
      rw [hone1, hxL, bitMatrix_not]
    · refine ⟨hand.mono hα, ?_⟩
      rw [hxL, hxR, bitMatrix_and]
    · refine ⟨(Enc.sub (Enc.add hL hR) (Enc.double hand htwo)).mono
        (xor_le_nextBound m n β E), ?_⟩
      rw [hxL, hxR, bitMatrix_xor bl br htwo]
  rcases select_two hva with ⟨ha, rfl⟩ | ⟨ha, rfl⟩ <;> subst ha <;>
    rcases select_two hpa with ⟨h1, rfl⟩ | ⟨h1, rfl⟩ <;> norm_num at h1 <;>
    rcases select_two hxa with ⟨h2, rfl⟩ | ⟨h2, rfl⟩ <;> norm_num at h2 <;>
    rcases select_two hba with ⟨h3, rfl⟩ | ⟨h3, rfl⟩ <;> norm_num at h3
  · exact ⟨hzero, hzeroBit⟩
  · exact hselected

/-! ## Two loops in lock-step -/

theorem iterRuns_invariant₂ {S T : Type} {body₁ : Nat → S → S → Prop} {body₂ : Nat → T → T → Prop}
    {Inv : Nat → S → T → Prop} {count : Nat} {a₀ a : S} {b₀ b : T}
    (h₁ : MxxIR.IterRuns body₁ count a₀ a) (h₂ : MxxIR.IterRuns body₂ count b₀ b)
    (h₀ : Inv 0 a₀ b₀)
    (step : ∀ i x y x' y', Inv i x y → body₁ i x x' → body₂ i y y' → Inv (i + 1) x' y') :
    Inv count a b := by
  induction h₁ generalizing b with
  | zero => cases h₂; exact h₀
  | step previous stepRun ih =>
    obtain ⟨mid, hmid, hstep⟩ := MxxIR.IterRuns.step_of_succ h₂
    exact step _ _ _ _ _ (ih hmid h₀) stepRun hstep

/-! ## Decoding the output -/

/-- `s = (s_bar, 1)` maps the target `T = (0; h)` of the decoding action to `h`. -/
theorem secret_mul_target {sbar : ExactMatrix q n 1 1} {s : ExactMatrix q n 1 2}
    (hs : concatColumns sbar (1 : ExactMatrix q n 1 1) s) {h : ExactMatrix q n 1 1}
    {T : ExactMatrix q n 2 1} (hT : concatRows (0 : ExactMatrix q n 1 1) h T) :
    s * T = h := by
  ext i j
  fin_cases i; fin_cases j
  simp only [Matrix.mul_apply, Fin.sum_univ_two]
  rw [hs, hs, hT, hT]
  simp [Matrix.one_apply]

theorem intCast_coeff_zero (hn : 0 < n) (c : Int) : ((c : ErrorPoly n)).coeff ⟨0, hn⟩ = c := by
  have h := intCast_mul_coeff c (1 : ErrorPoly n) ⟨0, hn⟩
  rw [mul_one] at h
  have h1 := Negacyclic.coeff_root_pow (R := Int) (n := n) hn ⟨0, hn⟩ ⟨0, hn⟩
  rw [pow_zero, if_pos rfl] at h1
  rw [h, h1, mul_one]

/-- Decryption of the output: when `c` encodes the bit `b` under `A` with error at most `E`, the
decoding action is an exact decomposition `G D = T` whose digits are constant and at most `β`,
`s T = floor(q / 2)`, and `4 m E β + 4 < q`, threshold decoding of `c D - s (A D)` returns `b`. -/
theorem decode_output (hq : 1 < q) (hodd : q % 2 = 1) (hn : 0 < n)
    {s : ExactMatrix q n 1 2} {sG : ExactMatrix q n 1 m} {G : ExactMatrix q n 2 m}
    (hsG : sG = s * G) {c : ExactMatrix q n 1 m} {A : ExactMatrix q n 2 m}
    {x : ExactMatrix q n 1 1} {E : Nat} {b : Bool} (henc : Enc s sG c A x E)
    (hx : x = bitMatrix b) {T : ExactMatrix q n 2 1} {D : ExactMatrix q n m 1}
    {d : ErrorMatrix n m 1} {β : Nat} (hGD : G * D = T) (hD : D = reduceMatrix q n m 1 d)
    (hd : CoeffBound d β) (hconstant : IsConstantMatrix d)
    (hsT : s * T = liftInteger (((q / 2 : Nat) : Int))) (hB : 4 * (m * E * β) + 4 < q)
    {decoded : Int}
    (hdecode : thresholdDecode 2 1 0 (matrixSub (matrixMul c D) (matrixMul s (matrixMul A D)))
      decoded) :
    decide (decoded ≠ 0) = b := by
  obtain ⟨e, he, hb⟩ := henc
  obtain ⟨_, _, _, _, index, hindex, _, hout⟩ := hdecode
  have hk : index = ⟨0, hn⟩ := Fin.ext (by exact_mod_cast hindex)
  subst hk
  have hbit : (if b then (1 : Int) else 0) = 0 ∨ (if b then (1 : Int) else 0) = 1 := by
    cases b <;> simp
  have hphase : matrixSub (matrixMul c D) (matrixMul s (matrixMul A D)) =
      -((x 0 0) • liftInteger (((q / 2 : Nat) : Int))) + reduceMatrix q n 1 1 (e * d) := by
    have hsGD : sG * D = s * T := by rw [hsG, Matrix.mul_assoc, hGD]
    simp only [matrixSub, matrixMul]
    rw [he, reduceMatrix_mul, ← hD, ← hsT, ← hsGD]
    simp only [matrixMulScalarLeft_eq_smul, Matrix.add_mul, Matrix.sub_mul, Matrix.smul_mul,
      Matrix.mul_assoc]
    abel
  have hentry : (matrixSub (matrixMul c D) (matrixMul s (matrixMul A D)) 0 0).coeff ⟨0, hn⟩ =
      ((-(((q / 2 : Nat) : Int) * (if b then 1 else 0)) + ((e * d) 0 0).coeff ⟨0, hn⟩ : Int) :
        ZMod q) := by
    rw [hphase, hx]
    simp only [Matrix.add_apply, Matrix.neg_apply, Matrix.smul_apply, smul_eq_mul, liftInteger,
      reduceMatrix_apply, bitMatrix_apply]
    rw [show (if b then (1 : ExactPoly q n) else 0) = (((if b then 1 else 0 : Int)) : ExactPoly q n)
      from by cases b <;> simp]
    rw [← Int.cast_mul, ← reducePoly_intCast, ← map_neg, ← map_add, reducePoly_coeff_eq hq hn,
      Negacyclic.coeff_add, Negacyclic.coeff_neg, intCast_coeff_zero hn]
    push_cast
    ring_nf
  rw [hout, hentry]
  have hbound : 4 * |((e * d) 0 0).coeff ⟨0, hn⟩| + 4 < q := by
    have hnorm := coeffBound_iff_matrixNorm_le.mp hb
    have hdnorm := coeffBound_iff_matrixNorm_le.mp hd
    have hmul := (matrixNorm_mul_right_constant_le hn hconstant (a := e)).trans
      (Nat.mul_le_mul (Nat.mul_le_mul_left m hnorm) hdnorm)
    have hcoeff := (coeffBound_iff_matrixNorm_le.mpr hmul) 0 0 ⟨0, hn⟩
    rw [Int.abs_eq_natAbs]
    omega
  have hvalue := MxxExponentLut.threshold_decode_value hq hodd hbit hbound
  rw [hvalue]
  cases b <;> simp

/-! ## Deterministic relations -/

theorem hashSample_unique {rows columns : Nat} {model : HashModel} {tag : Blob}
    {components : List HashTagComponent} {key : ByteArray}
    {a b : ExactMatrix q n rows columns} (ha : hashSample model tag components key a)
    (hb : hashSample model tag components key b) : a = b :=
  ha.2.2.trans hb.2.2.symm

theorem sliceMatrix_unique {rows columns outRows outColumns : Nat}
    {input : ExactMatrix q n rows columns} {r₀ r₁ c₀ c₁ : Int}
    {a b : ExactMatrix q n outRows outColumns} (ha : sliceMatrix input r₀ r₁ c₀ c₁ a)
    (hb : sliceMatrix input r₀ r₁ c₀ c₁ b) : a = b := by
  obtain ⟨h₀, h₁, h₂, h₃, h₄, h₅, h₆, h₇, ha⟩ := ha
  obtain ⟨-, -, -, -, -, -, -, -, hb⟩ := hb
  simp only [Int.ofNat_eq_natCast] at h₀ h₁ h₂ h₃ h₄ h₅ h₆ h₇
  ext i j
  have hi : r₀.toNat + i.val < rows := by have := i.isLt; omega
  have hj : c₀.toNat + j.val < columns := by have := j.isLt; omega
  rw [ha i j hi hj, hb i j hi hj]

/-- The full slice of a matrix is the matrix. -/
theorem sliceMatrix_full {rows columns : Nat} {input output : ExactMatrix q n rows columns}
    (h : sliceMatrix input 0 rows 0 columns output) : output = input := by
  obtain ⟨-, -, -, -, -, -, -, -, h⟩ := h
  ext i j
  rw [h i j (by simp) (by simp)]
  simp

theorem concatColumns_unique {rows left right : Nat} {a : ExactMatrix q n rows left}
    {b : ExactMatrix q n rows right} {x y : ExactMatrix q n rows (left + right)}
    (hx : concatColumns a b x) (hy : concatColumns a b y) : x = y := by
  ext i j
  rw [hx, hy]

theorem gadgetMatrixRuns_unique {rows columns : Nat} {backend : BackendContext} {base digits : Int}
    {a b : ExactMatrix q n rows columns} (ha : gadgetMatrixRuns backend base digits a)
    (hb : gadgetMatrixRuns backend base digits b) : a = b := by
  obtain ⟨la, hla, -, -, hwa, ha⟩ := ha
  obtain ⟨lb, hlb, -, -, hwb, hb⟩ := hb
  have hsame : la = lb := Option.some.inj (hla.symm.trans hlb)
  subst hsame
  rw [ha, hb]

/-- An exact regular gadget: when the backend drops no tower, `G D = T` for the decomposition `D`
of `T`, whose digits are at most half the base. -/
theorem gadget_exact (hq : 1 < q) (hn : 0 < n) {rows columns inner : Nat}
    {backend : BackendContext} {base digits : Int} {G : ExactMatrix q n rows inner}
    {T : ExactMatrix q n rows columns} {D : ExactMatrix q n inner columns}
    (hg : gadgetMatrixRuns backend base digits G) (hd : gadgetDecomposeRuns backend base digits T D)
    (hexact : ∀ layout, backend.regularLayout q n = some layout → layout.droppedModuli = 0) :
    G * D = T ∧ PreimageWithin D (base.toNat / 2) := by
  obtain ⟨layout, hl, hbase, -, hw, hg⟩ := hg
  obtain ⟨other, ho, -, -, hw', hd, -⟩ := hd
  have hsame : other = layout := Option.some.inj (ho.symm.trans hl)
  subst hsame
  subst inner
  simp only [castMatrixColumns, castMatrixRows] at hg hd
  refine ⟨?_, ?_⟩
  · rw [hg, hd]
    exact regularGadgetMatrix_reconstruct other T hq hn (hexact other ho)
  · rw [hd, hbase, Int.toNat_natCast]
    exact regularDecomposeMatrix_bounded other T hq hn

/-- `1 ⊗ M = M` for a row `M`. -/
theorem tensorRuns_one {columns : Nat} {M : ExactMatrix q n 1 columns}
    {out : ExactMatrix q n (1 * 1) (1 * columns)} (h : tensorRuns (1 : ExactMatrix q n 1 1) M out) :
    ∀ j : Fin columns, out 0 ⟨j.val, by omega⟩ = M 0 j := by
  intro j
  have := h 0 0 0 j
  simp only [Fin.val_zero, zero_mul, zero_add, Matrix.one_apply_eq, one_mul] at this
  exact this

/-- A sampled Gaussian error is the reduction of an integer error within the cutoff. -/
theorem gaussian_error (hn : 0 < n) {rows columns : Nat} {tape : SampleTape} {site : List Nat}
    {sigma : Rat} {cutoff : Int} {e : ExactMatrix q n rows columns}
    (h : gaussianSampleAt tape site sigma cutoff e) :
    ∃ w : ErrorMatrix n rows columns, e = reduceMatrix q n rows columns w ∧
      CoeffBound w cutoff.toNat :=
  (MxxExponentLut.gaussianSampleAt_sample hn h).2.2.1

end RingGswBgg
