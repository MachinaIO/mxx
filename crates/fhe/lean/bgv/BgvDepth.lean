import BgvCorrectness
import BgvGate1

set_option backward.isDefEq.respectTransparency false

/-! Correctness of the generated BGV scheme for the circuit `x₀ · (x₁ · x₂)`
(`GeneratedScheme.CorrectnessClaim`).

A ciphertext of level `ℓ` satisfies `PhaseFacts`: its decryption phase is a small integer
polynomial `W` with `c_ℓ W ≡ M` modulo `t`, where `M` is a plaintext whose native evaluations are
the slots of the message and `c_ℓ` is the correction factor of the level (`1`, `q3`, and
`q3² q2`). Fresh encryptions establish it at level 0. The circuit's graph is the switch of `x₀`
down to level 1 followed by the graphs of the gate on level-0 arguments and of the gate on level-1
arguments (`eval_spec`, which reads it as instances of `Gate0` and `Gate1`); `convert_phase`,
`gate0_phase`, and `gate1_phase` carry the invariant to level 2, where decryption returns the
message of a ciphertext satisfying it, in every execution.

The gate on level-1 arguments relinearizes with the key of the top level: the graph reduces the key
modulo `q1 q2 P`, keeps the columns of the remaining primes, and scales the value by
`κ ≡ q3⁻¹ (mod q1 q2)` before ModUp, so the key's gadget factors `P q2 q3` and `P q1 q3`
recombine the digits (`gate1_spec`). -/

namespace BgvCorrect
open MxxSecurity MxxFhe GeneratedScheme Mxx.Primitives MxxRuntime BgvSecurity

abbrev Q1 : Nat := 18014398507892737
abbrev QLP : Nat := 23384026193386519304488586269947798086183317045249
abbrev kappa : Int := 149176486870899040271091973769539

/-! ## RNS digits over two primes -/

/-- With one prime per normalized digit over two primes, digit `j` is the centered CRT term of
prime `j`. -/
theorem rnsDigit_two_single {p1 p2 : Nat} (h1 : 1 < p1) (h2 : 1 < p2) (value : Int) :
    rnsDigit [p1, p2] 1 0 true value = rnsCentered p1 (value * rnsInverse p1 p2) ∧
    rnsDigit [p1, p2] 1 1 true value = rnsCentered p2 (value * rnsInverse p2 p1) := by
  have e1 : p1 * p2 / p1 = p2 := Nat.mul_div_cancel_left _ (by omega)
  have e2 : p1 * p2 / p2 = p1 := Nat.mul_div_cancel _ (by omega)
  refine ⟨?_, ?_⟩ <;>
  simp only [rnsDigit, List.drop, List.take, List.prod_cons,
    List.prod_nil, mul_one, List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, add_zero,
    ite_true, ← Int.natCast_div, e1, e2, Nat.div_self (by omega : 0 < p1), Nat.div_self (by omega : 0 < p2),
    Int.ofNat_eq_natCast, Nat.cast_one, one_mul, rnsInverse_one h1, rnsInverse_one h2]

/-- The centered CRT digits of two coprime moduli reconstruct `value` modulo their product. -/
theorem crt_two_centered {p1 p2 : Nat} (h1 : 1 < p1) (h2 : 1 < p2) (h12 : Nat.Coprime p1 p2)
    (value : Int) :
    (p2 : Int) * rnsCentered p1 (value * rnsInverse p1 p2) +
      (p1 : Int) * rnsCentered p2 (value * rnsInverse p2 p1) ≡ value [ZMOD (p1 * p2 : Nat)] := by
  rw [Int.modEq_iff_dvd]
  have h12' : IsCoprime (p1 : Int) p2 := Nat.isCoprime_iff_coprime.mpr h12
  have a := (crt_term_modEq h1 h12.symm value).add (dvd_modEq_zero (dvd_refl (p1 : Int))
    (rnsCentered p2 (value * rnsInverse p2 p1)))
  have b := (dvd_modEq_zero (dvd_refl (p2 : Int)) (rnsCentered p1 (value * rnsInverse p1 p2))).add
    (crt_term_modEq h2 h12 value)
  rw [add_zero] at a
  rw [zero_add] at b
  rw [Int.modEq_iff_dvd] at a b
  push_cast
  exact h12'.mul_dvd (by simpa [mul_comm] using a) (by simpa [mul_comm] using b)

/-! ## The gate from level 1 to level 2 -/

/-- The relinearization key of the top level as the gate from level 1 uses it: modulo
`q1 q2 P`, with the columns of `q1` and `q2`. -/
noncomputable def key1 (ek : ExactMatrix QP n 2 3) : ExactMatrix QLP n 2 2 :=
  fun r j ↦ modulusReduce (p := QLP) ek r (Fin.castSucc j)

/-- Relinearization at level 1 with the reduced key. -/
def RelinFacts1 (d0 d1 d2 : ExactPoly QL n) (rk : ExactMatrix QLP n 2 2)
    (out : ExactMatrix QL n 2 1) : Prop :=
  ∃ (digits : Fin 2 → ErrorPoly n) (correction quotient : Fin 2 → ErrorPoly n),
    polyNorm (digits 0) ≤ q1 / 2 ∧ polyNorm (digits 1) ≤ q2 / 2 ∧
    (∀ i, ((q2 * q3 : Nat) : Int) * (digits 0).coeff i +
      ((q1 * q3 : Nat) : Int) * (digits 1).coeff i ≡ (canonicalLift d0).coeff i [ZMOD QL]) ∧
    (∀ r, polyNorm (correction r) ≤ P / 2) ∧
    (∀ r : Fin 2, ((P : Int) : ErrorPoly n) * quotient r =
      canonicalLift (∑ j : Fin 2, rk r j * reducePoly QLP n (digits j)) +
        ((t : Int) : ErrorPoly n) * correction r) ∧
    out 0 0 = d1 + reducePoly QL n (quotient 0) ∧ out 1 0 = d2 + reducePoly QL n (quotient 1)

/-- Modulus switching from level 1: ModDown by `q2`. -/
def ModSwitchFacts1 (ct : ExactMatrix QL n 2 1) (out : ExactMatrix Q1 n 2 1) : Prop :=
  ∃ correction quotient : Fin 2 → ErrorPoly n, (∀ r, polyNorm (correction r) ≤ q2 / 2) ∧
    (∀ r : Fin 2, ((q2 : Int) : ErrorPoly n) * quotient r =
      canonicalLift (ct r 0) + ((t : Int) : ErrorPoly n) * correction r) ∧
    ∀ r : Fin 2, out r 0 = reducePoly Q1 n (quotient r)

/-- What the gate from level 1 establishes about its output. -/
def GateFacts1 (c0 c1 : ExactMatrix QL n 2 1) (ek : ExactMatrix QP n 2 3)
    (out : ExactMatrix Q1 n 2 1) : Prop :=
  ∃ relin : ExactMatrix QL n 2 1,
    RelinFacts1 (c0 0 0 * c1 0 0) (c0 0 0 * c1 1 0 + c0 1 0 * c1 0 0) (c0 1 0 * c1 1 0)
      (key1 ek) relin ∧
    ModSwitchFacts1 relin out

/-- The digits of `κ d0` recombine to `d0` under the key's gadget factors. -/
theorem kappa_crt (d0 : ExactPoly QL n) (i : Fin n) :
    ((q2 * q3 : Nat) : Int) * rnsCentered q1
        (Int.ofNat ((((kappa : Int) : ExactPoly QL n) * d0).coeff i).val * rnsInverse q1 q2) +
      ((q1 * q3 : Nat) : Int) * rnsCentered q2
        (Int.ofNat ((((kappa : Int) : ExactPoly QL n) * d0).coeff i).val * rnsInverse q2 q1) ≡
      (canonicalLift d0).coeff i [ZMOD QL] := by
  set v := Int.ofNat ((((kappa : Int) : ExactPoly QL n) * d0).coeff i).val
  have hcrt := crt_two_centered (p1 := q1) (p2 := q2) (by decide) (by decide) (by norm_num) v
  have hv : v ≡ kappa * (canonicalLift d0).coeff i [ZMOD QL] := by
    have : ((kappa : Int) : ExactPoly QL n) * d0 =
        reducePoly QL n (((kappa : Int) : ErrorPoly n) * canonicalLift d0) := by
      rw [map_mul, reducePoly_intCast, reducePoly_canonicalLift (by decide) hn]
    show Int.ofNat ((((kappa : Int) : ExactPoly QL n) * d0).coeff i).val ≡ _ [ZMOD QL]
    rw [this, reducePoly_coeff (q := QL) (by decide) hn, intCast_mul_coeff, Int.ofNat_eq_natCast,
      val_intCast_emod (q := QL) (by decide)]
    exact Int.mod_modEq _ _
  have hk : (q3 : Int) * kappa ≡ 1 [ZMOD QL] := by decide
  calc ((q2 * q3 : Nat) : Int) * rnsCentered q1 (v * rnsInverse q1 q2) +
        ((q1 * q3 : Nat) : Int) * rnsCentered q2 (v * rnsInverse q2 q1)
      = (q3 : Int) * ((q2 : Int) * rnsCentered q1 (v * rnsInverse q1 q2) +
          (q1 : Int) * rnsCentered q2 (v * rnsInverse q2 q1)) := by push_cast; ring
    _ ≡ (q3 : Int) * v [ZMOD QL] := hcrt.mul_left _
    _ ≡ (q3 : Int) * (kappa * (canonicalLift d0).coeff i) [ZMOD QL] := hv.mul_left _
    _ = ((q3 : Int) * kappa) * (canonicalLift d0).coeff i := by ring
    _ ≡ 1 * (canonicalLift d0).coeff i [ZMOD QL] := hk.mul_right _
    _ = (canonicalLift d0).coeff i := one_mul _

theorem gate1_spec {c0 c1 : ExactMatrix QL n 2 1} {ek : ExactMatrix QP n 2 3}
    {out : ExactMatrix Q1 n 2 1}
    (h : Gate1.generatedRoot Gate1.params (c0, c1, ek, ()) out) : GateFacts1 c0 c1 ek out := by
  obtain ⟨w, h⟩ := h
  dsimp only [Gate1.generatedRoot.body] at h
  obtain ⟨ht, -, -, -, -, -, -, s3, -, -, -, -, -, -, s4, -, -, -, -, -, -, s5, -, -, -, -, -, -,
    s7, cat8c, cat8, -, -, -, -, -, -, s9, -, -, -, -, -, -, s10, cat11, hred, -, -, -, -, -, -,
    s14, -, -, -, -, -, -, s15, hup, hdown, hms, rfl⟩ := h
  obtain ⟨t0, t1, t2, t3⟩ := tensorRuns_two ht
  have e3 := sliceMatrix_entry s3 (by decide) (by decide)
  have e4 := sliceMatrix_entry s4 (by decide) (by decide)
  have e5 := sliceMatrix_entry s5 (by decide) (by decide)
  have e7 := sliceMatrix_entry s7 (by decide) (by decide)
  have e9 := sliceMatrix_entry s9 (by decide) (by decide)
  have e10 := sliceMatrix_entry s10 (by decide) (by decide)
  have e15 := sliceMatrix_entry s15 (by decide) (by decide)
  obtain ⟨a0, a1⟩ := concatRows_one_one cat8c 0
  obtain ⟨b0, b1, b2⟩ := concatRows_two_one cat8 0
  obtain ⟨c0', c1'⟩ := concatRows_one_one cat11 0
  have hd0 : w.w_15_0 0 0 = c0 0 0 * c1 0 0 := by
    rw [e15]; exact (b0.trans (a0.trans e3)).trans t0
  have hd1 : w.w_11_0 0 0 = c0 0 0 * c1 1 0 + c0 1 0 * c1 0 0 := by
    rw [c0', e9]
    refine b1.trans (a1.trans ?_)
    simp only [matrixAdd, Matrix.add_apply, e4, e5]
    change w.w_2_0 1 0 + w.w_2_0 2 0 = _
    rw [t1, t2]
  have hd2 : w.w_11_0 1 0 = c0 1 0 * c1 1 0 := by
    rw [c1', e10]
    exact (b2.trans e7).trans t3
  -- The key of the top level, reduced to level 1.
  have hkey (r j : Fin 2) : w.w_14_0 r j = key1 ek r j := by
    obtain ⟨_, _, _, _, _, _, _, _, hout⟩ := s14
    rw [hout r j (by omega) (by omega), hred.2.2]
    show modulusReduce (p := QLP) ek _ _ = modulusReduce (p := QLP) ek r (Fin.castSucc j)
    congr 1 <;> exact Fin.ext (by simp)
  -- Relinearization.
  obtain ⟨_, _, _, _, _, _, _, _, hdigit⟩ := hup
  let v : Fin n → Int := fun i ↦
    Int.ofNat ((((kappa : Int) : ExactPoly QL n) * w.w_15_0 0 0).coeff i).val
  let digits : Fin 2 → ErrorPoly n := fun j ↦ intPoly fun i ↦
    rnsDigit [18014398507892737, 18014398508138497] 1 j.val true (v i)
  have hdigits (j : Fin 2) : w.w_17_0 j 0 = reducePoly QLP n (digits j) := by
    rw [hdigit ⟨j.val, by omega⟩ 0 j 0 (by simp), polynomialOfCoefficients_eq_reduce]
    rfl
  have hsingle (i : Fin n) := rnsDigit_two_single (p1 := q1) (p2 := q2) (by decide) (by decide) (v i)
  have hdrop : [18014398507892737, 18014398508138497, 72057594037616641].filter
      (fun prime ↦ QL % prime != 0) = [P] := by decide
  have hspec (r : Fin 2) :=
    rnsModDown_single_exact hdown hdrop (by decide) (by decide) (by decide) r 0
  choose correction quotient hbound hexact hquot using hspec
  have hproduct (r : Fin 2) : (matrixMul w.w_14_0 w.w_17_0) r 0 =
      ∑ j : Fin 2, key1 ek r j * reducePoly QLP n (digits j) := by
    simp only [matrixMul, Matrix.mul_apply, hdigits, hkey]
  -- Modulus switching.
  have hdropL : [18014398507892737, 18014398508138497].filter
      (fun prime ↦ Q1 % prime != 0) = [q2] := by decide
  have hspecL (r : Fin 2) :=
    rnsModDown_single_exact hms hdropL (by decide) (by decide) (by decide) r 0
  choose correctionL quotientL hboundL hexactL hquotL using hspecL
  refine ⟨matrixAdd w.w_11_0 w.w_19_0, ⟨digits, correction, quotient, ?_, ?_, ?_, hbound,
    fun r ↦ ?_, ?_, ?_⟩, ⟨correctionL, quotientL, hboundL, hexactL, hquotL⟩⟩
  · apply polyNorm_le_of_coeff; intro i
    simp only [digits, intPoly_coeff (by decide : 0 < n), Fin.val_zero, (hsingle i).1]
    exact rnsCentered_natAbs_le (by decide) _
  · apply polyNorm_le_of_coeff; intro i
    simp only [digits, intPoly_coeff (by decide : 0 < n), Fin.val_one, (hsingle i).2]
    exact rnsCentered_natAbs_le (by decide) _
  · intro i
    simp only [digits, intPoly_coeff (by decide : 0 < n), Fin.val_zero, Fin.val_one,
      (hsingle i).1, (hsingle i).2]
    rw [← hd0]
    exact kappa_crt (w.w_15_0 0 0) i
  · rw [hexact r, hproduct r]; rfl
  · simp only [matrixAdd, Matrix.add_apply, hd1, hquot]
  · simp only [matrixAdd, Matrix.add_apply, hd2, hquot]

theorem gate1_exists (c0 c1 : ExactMatrix QL n 2 1) (ek : ExactMatrix QP n 2 3) :
    ∃ out, Gate1.generatedRoot Gate1.params (c0, c1, ek, ()) out := by
  let w2 : ExactMatrix QL n 4 1 := fun r _ ↦
    ![c0 0 0 * c1 0 0, c0 0 0 * c1 1 0, c0 1 0 * c1 0 0, c0 1 0 * c1 1 0] r
  let w8 : ExactMatrix QL n 3 1 := fun r _ ↦ ![w2 0 0, w2 1 0 + w2 2 0, w2 3 0] r
  let w11 : ExactMatrix QL n 2 1 := fun r _ ↦ ![w8 1 0, w8 2 0] r
  let w15 : ExactMatrix QL n 1 1 := fun _ _ ↦ w8 0 0
  let w17 : ExactMatrix QLP n 2 1 := fun r _ ↦ polynomialOfCoefficients fun i ↦
    rnsDigit [18014398507892737, 18014398508138497] 1 r.val true
      (Int.ofNat (((matrixScale 149176486870899040271091973769539 w15) 0 0).coeff i).val)
  have hup : rnsModUpRuns [18014398507892737, 18014398508138497] 1 true
      (matrixScale 149176486870899040271091973769539 w15) w17 := by
    refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
      by norm_num, fun digit row outRow column h ↦ ?_⟩
    fin_cases digit <;> fin_cases row <;> fin_cases outRow <;> fin_cases column <;>
      first | (simp at h; done) | rfl
  obtain ⟨w19, h19⟩ := rnsModDownRuns_exists (p := QL)
    (basis := [18014398507892737, 18014398508138497, 72057594037616641])
    (plaintext := 1032193) (matrixMul (key1 ek) w17) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by decide) (by norm_num) (by decide)
  obtain ⟨w21, h21⟩ := rnsModDownRuns_exists (p := Q1)
    (basis := [18014398507892737, 18014398508138497])
    (plaintext := 1032193) (matrixAdd w11 w19) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by decide) (by norm_num) (by decide)
  refine ⟨w21, ⟨w2, fun _ _ ↦ w2 0 0, fun _ _ ↦ w2 1 0, fun _ _ ↦ w2 2 0, fun _ _ ↦ w2 3 0, w8,
    (fun r _ ↦ ![w2 0 0, w2 1 0 + w2 2 0] r : ExactMatrix QL n 2 1), fun _ _ ↦ w8 1 0,
    fun _ _ ↦ w8 2 0, w11, modulusReduce ek, key1 ek, w15, w17, w19, w21⟩, ?_⟩
  dsimp only [Gate1.generatedRoot.body]
  exact ⟨(fun i1 i2 j1 j2 ↦ by fin_cases i1 <;> fin_cases i2 <;> fin_cases j1 <;> fin_cases j2 <;> rfl),
      by decide,
      by decide,
      by decide,
      by decide,
      by decide,
      by decide,
      ⟨by decide, by decide, by decide, by decide, by decide, by decide, by decide, by decide, fun row column _ _ ↦ by fin_cases row; fin_cases column; rfl⟩,
      by decide,
      by decide,
      by decide,
      by decide,
      by decide,
      by decide,
      ⟨by decide, by decide, by decide, by decide, by decide, by decide, by decide, by decide, fun row column _ _ ↦ by fin_cases row; fin_cases column; rfl⟩,
      by decide,
      by decide,
      by decide,
      by decide,
      by decide,
      by decide,
      ⟨by decide, by decide, by decide, by decide, by decide, by decide, by decide, by decide, fun row column _ _ ↦ by fin_cases row; fin_cases column; rfl⟩,
      by decide,
      by decide,
      by decide,
      by decide,
      by decide,
      by decide,
      ⟨by decide, by decide, by decide, by decide, by decide, by decide, by decide, by decide, fun row column _ _ ↦ by fin_cases row; fin_cases column; rfl⟩,
      (fun row column ↦ by fin_cases row <;> fin_cases column <;> rfl),
      (fun row column ↦ by fin_cases row <;> fin_cases column <;> rfl),
      by decide,
      by decide,
      by decide,
      by decide,
      by decide,
      by decide,
      ⟨by decide, by decide, by decide, by decide, by decide, by decide, by decide, by decide, fun row column _ _ ↦ by fin_cases row; fin_cases column; rfl⟩,
      by decide,
      by decide,
      by decide,
      by decide,
      by decide,
      by decide,
      ⟨by decide, by decide, by decide, by decide, by decide, by decide, by decide, by decide, fun row column _ _ ↦ by fin_cases row; fin_cases column; rfl⟩,
      (fun row column ↦ by fin_cases row <;> fin_cases column <;> rfl),
      ⟨by norm_num, by norm_num, rfl⟩,
      by decide,
      by decide,
      by decide,
      by decide,
      by decide,
      by decide,
      ⟨by decide, by decide, by decide, by decide, by decide, by decide, by decide, by decide, fun row column _ _ ↦ by fin_cases row <;> fin_cases column <;> rfl⟩,
      by decide,
      by decide,
      by decide,
      by decide,
      by decide,
      by decide,
      ⟨by decide, by decide, by decide, by decide, by decide, by decide, by decide, by decide, fun row column _ _ ↦ by fin_cases row; fin_cases column; rfl⟩,
      hup,
      h19,
      h21,
      rfl⟩

/-! ## Phase arithmetic of the gate from level 1 -/

/-- The gadget factors of the reduced key: those of the top level's first two digits. -/
def factor1 : Fin 2 → Int := ![(P : Int) * ((q2 * q3 : Nat) : Int), (P : Int) * ((q1 * q3 : Nat) : Int)]

theorem factor_castSucc (j : Fin 2) : factor (Fin.castSucc j) = factor1 j := by
  obtain ⟨g0, g1, -⟩ := factor_eq
  fin_cases j
  · exact g0
  · exact g1

/-- The reduced key keeps the key relation modulo `q1 q2 P`. -/
theorem key1_rows {ek : ExactMatrix QP n 2 3} {S D : ErrorPoly n} {Erk : Fin 3 → ErrorPoly n}
    (hrk : ∀ j : Fin 3, ek 1 j = reducePoly QP n S * ek 0 j +
      ((t : Int) : ExactPoly QP n) * reducePoly QP n (Erk j) +
      reducePoly QP n D * ((factor j : Int) : ExactPoly QP n)) (j : Fin 2) :
    key1 ek 1 j = reducePoly QLP n S * key1 ek 0 j +
      ((t : Int) : ExactPoly QLP n) * reducePoly QLP n (Erk (Fin.castSucc j)) +
      reducePoly QLP n D * ((factor1 j : Int) : ExactPoly QLP n) := by
  have hm (X : ErrorPoly n) := modulusReduce_reducePoly (q := QP) (p := QLP) (by decide)
    (by decide) (by norm_num) hn X
  set A := canonicalLift (ek 0 (Fin.castSucc j)) with hAdef
  have hA : ek 0 (Fin.castSucc j) = reducePoly QP n A :=
    (reducePoly_canonicalLift (by decide) hn _).symm
  have h1 : ek 1 (Fin.castSucc j) = reducePoly QP n (S * A + ((t : Int) : ErrorPoly n) *
      Erk (Fin.castSucc j) + D * ((factor (Fin.castSucc j) : Int) : ErrorPoly n)) := by
    rw [hrk, hA]
    simp only [map_add, map_mul, reducePoly_intCast]
  have k0 : key1 ek 0 j = reducePoly QLP n A := by
    show polynomialOfCoefficients _ = _
    rw [hA]
    exact hm A
  have k1 : key1 ek 1 j = reducePoly QLP n (S * A + ((t : Int) : ErrorPoly n) *
      Erk (Fin.castSucc j) + D * ((factor (Fin.castSucc j) : Int) : ErrorPoly n)) := by
    show polynomialOfCoefficients _ = _
    rw [h1]
    exact hm _
  rw [k0, k1, factor_castSucc]
  simp only [map_add, map_mul, reducePoly_intCast]

/-- Relinearization at level 1 moves `s^2 d0` into the pair and adds `t X`. -/
theorem relin_phase1 {S D : ErrorPoly n} {Erk : Fin 2 → ErrorPoly n} {rk : ExactMatrix QLP n 2 2}
    {digits : Fin 2 → ErrorPoly n} {C K : Fin 2 → ErrorPoly n} {d0 : ExactPoly QL n}
    (hD : ∀ i, D.coeff i ≡ (S * S).coeff i [ZMOD Q])
    (hrk : ∀ j : Fin 2, rk 1 j = reducePoly QLP n S * rk 0 j +
      ((t : Int) : ExactPoly QLP n) * reducePoly QLP n (Erk j) +
      reducePoly QLP n D * ((factor1 j : Int) : ExactPoly QLP n))
    (hcrt : ∀ i, ((q2 * q3 : Nat) : Int) * (digits 0).coeff i +
      ((q1 * q3 : Nat) : Int) * (digits 1).coeff i ≡ (canonicalLift d0).coeff i [ZMOD QL])
    (hK : ∀ r : Fin 2, ((P : Int) : ErrorPoly n) * K r =
      canonicalLift (∑ j : Fin 2, rk r j * reducePoly QLP n (digits j)) +
        ((t : Int) : ErrorPoly n) * C r) :
    ∃ X : ErrorPoly n, ((P : Int) : ErrorPoly n) * X =
        Erk 0 * digits 0 + Erk 1 * digits 1 + C 1 - S * C 0 ∧
      reducePoly QL n (K 1) - reducePoly QL n S * reducePoly QL n (K 0) =
        reducePoly QL n (S * S) * d0 + ((t : Int) : ExactPoly QL n) * reducePoly QL n X := by
  have hA (r : Fin 2) : reducePoly QLP n (canonicalLift (∑ j : Fin 2, rk r j *
      reducePoly QLP n (digits j))) = ∑ j : Fin 2, rk r j * reducePoly QLP n (digits j) :=
    reducePoly_canonicalLift (by decide) hn _
  have hM : reducePoly QL n (((q2 * q3 : Nat) : Int) * digits 0 +
      ((q1 * q3 : Nat) : Int) * digits 1) = reducePoly QL n (canonicalLift d0) := by
    rw [reducePoly_eq_iff (by decide) hn]
    intro i
    simp only [Negacyclic.coeff_add, intCast_mul_coeff]
    exact hcrt i
  obtain ⟨Y, hY⟩ := exists_of_reducePoly_eq (q := QL) (by decide) hn hM
  have hdiff : reducePoly QLP n (canonicalLift (∑ j : Fin 2, rk 1 j * reducePoly QLP n (digits j)) -
      S * canonicalLift (∑ j : Fin 2, rk 0 j * reducePoly QLP n (digits j))) = reducePoly QLP n
      (((t : Int) : ErrorPoly n) * (Erk 0 * digits 0 + Erk 1 * digits 1) +
        ((P : Int) : ErrorPoly n) * (D * (((q2 * q3 : Nat) : Int) * digits 0 +
          ((q1 * q3 : Nat) : Int) * digits 1))) := by
    rw [map_sub, map_mul, hA, hA]
    simp only [Fin.sum_univ_two, hrk, map_add, map_mul, reducePoly_intCast, factor1,
      Matrix.cons_val_zero, Matrix.cons_val_one]
    push_cast
    ring
  obtain ⟨Z, hZ⟩ := exists_of_reducePoly_eq (q := QLP) (by decide) hn hdiff
  have hQLP : ((QLP : Int) : ErrorPoly n) =
      ((QL : Int) : ErrorPoly n) * ((P : Int) : ErrorPoly n) := by
    rw [← intCast_errorPoly_mul]; norm_num
  have key : ((t : Int) : ErrorPoly n) * (Erk 0 * digits 0 + Erk 1 * digits 1 + C 1 - S * C 0) =
      ((P : Int) : ErrorPoly n) * (K 1 - S * K 0 - D * canonicalLift d0 -
        ((QL : Int) : ErrorPoly n) * (D * Y) - ((QL : Int) : ErrorPoly n) * Z) := by
    rw [hQLP] at hZ
    linear_combination -(hK 1) + S * (hK 0) - hZ - ((P : Int) : ErrorPoly n) * D * hY
  obtain ⟨X, hX⟩ := exists_of_coprime_mul hn (by decide) key
  refine ⟨X, hX.symm, ?_⟩
  have hcancel := intCast_mul_cancel hn (by decide : ((P : Int)) ≠ 0)
    (show ((P : Int) : ErrorPoly n) * (((t : Int) : ErrorPoly n) * X) =
      ((P : Int) : ErrorPoly n) * (K 1 - S * K 0 - D * canonicalLift d0 -
        ((QL : Int) : ErrorPoly n) * (D * Y) - ((QL : Int) : ErrorPoly n) * Z) by
      rw [← key, hX]; ring)
  have hred := congrArg (reducePoly QL n) hcancel
  have hDS : reducePoly QL n D = reducePoly QL n (S * S) :=
    (reducePoly_eq_iff (by decide) hn _ _).mpr fun i ↦
      (hD i).of_dvd (by norm_num : ((QL : Nat) : Int) ∣ ((Q : Nat) : Int))
  have hQ0 : reducePoly QL n ((QL : Int) : ErrorPoly n) = 0 := by
    have := reducePoly_modulus_mul (n := n) QL 1
    rwa [mul_one] at this
  simp only [map_sub, map_mul] at hred
  rw [hQ0, hDS, reducePoly_canonicalLift (q := QL) (by decide) hn, reducePoly_intCast] at hred
  rw [map_mul] at hred ⊢
  linear_combination -hred

theorem relin_error_norm1 {S X : ErrorPoly n} {Erk digits : Fin 2 → ErrorPoly n}
    {C : Fin 2 → ErrorPoly n} (hS : polyNorm S ≤ 1) (hErk : ∀ j, polyNorm (Erk j) ≤ 20)
    (hδ0 : polyNorm (digits 0) ≤ q1 / 2) (hδ1 : polyNorm (digits 1) ≤ q2 / 2)
    (hC : ∀ r, polyNorm (C r) ≤ P / 2)
    (hX : ((P : Int) : ErrorPoly n) * X = Erk 0 * digits 0 + Erk 1 * digits 1 + C 1 - S * C 0) :
    polyNorm X ≤ (n * 20 * (q1 / 2) + n * 20 * (q2 / 2) + P / 2 + n * 1 * (P / 2)) / P := by
  apply polyNorm_le_div_of_mul (by decide) hX
  have m (a b : ErrorPoly n) {x y : Nat} (ha : polyNorm a ≤ x) (hb : polyNorm b ≤ y) :
      polyNorm (a * b) ≤ n * x * y :=
    (polyNorm_mul_le hn a b).trans (Nat.mul_le_mul (Nat.mul_le_mul_left _ ha) hb)
  exact (polyNorm_sub_le _ _).trans (Nat.add_le_add ((polyNorm_add_le _ _).trans
    (Nat.add_le_add ((polyNorm_add_le _ _).trans
      (Nat.add_le_add (m _ _ (hErk 0) hδ0) (m _ _ (hErk 1) hδ1))) (hC 1))) (m _ _ hS (hC 0)))

/-- Modulus switching from level 1 divides the phase by `q2` exactly after its `t`-correction. -/
theorem modswitch_phase1 {S W : ErrorPoly n} {c : ExactMatrix QL n 2 1}
    {C' K' : Fin 2 → ErrorPoly n}
    (hW : c 1 0 - reducePoly QL n S * c 0 0 = reducePoly QL n W)
    (hK' : ∀ r : Fin 2, ((q2 : Int) : ErrorPoly n) * K' r =
      canonicalLift (c r 0) + ((t : Int) : ErrorPoly n) * C' r) :
    ∃ W' : ErrorPoly n, ((q2 : Int) : ErrorPoly n) * W' =
        W + ((t : Int) : ErrorPoly n) * (C' 1 - S * C' 0) ∧
      reducePoly Q1 n (K' 1) - reducePoly Q1 n S * reducePoly Q1 n (K' 0) = reducePoly Q1 n W' := by
  have hlift : reducePoly QL n (canonicalLift (c 1 0) - S * canonicalLift (c 0 0)) =
      reducePoly QL n W := by
    rw [map_sub, map_mul, reducePoly_canonicalLift (by decide) hn,
      reducePoly_canonicalLift (by decide) hn, hW]
  obtain ⟨Z, hZ⟩ := exists_of_reducePoly_eq (q := QL) (by decide) hn hlift
  have hQ : ((QL : Int) : ErrorPoly n) = ((q2 : Int) : ErrorPoly n) * ((Q1 : Int) : ErrorPoly n) := by
    rw [← intCast_errorPoly_mul]; norm_num
  refine ⟨K' 1 - S * K' 0 - ((Q1 : Int) : ErrorPoly n) * Z, ?_, ?_⟩
  · rw [hQ] at hZ
    linear_combination (hK' 1) - S * (hK' 0) + hZ
  · rw [map_sub, map_mul (reducePoly Q1 n) ((Q1 : Int) : ErrorPoly n), reducePoly_intCast,
      show ((Q1 : Int) : ExactPoly Q1 n) = 0 by
        have := reducePoly_modulus_mul (n := n) Q1 1
        rwa [mul_one, reducePoly_intCast] at this, map_sub, map_mul]
    ring

/-! ## The phase invariant of a level -/

/-- A ciphertext modulo `q` whose decryption phase under the secret `S` is the integer
polynomial `W`, of norm at most `B`, with `c W ≡ M` modulo `t`. -/
def PhaseFacts {q : Nat} (S : ErrorPoly n) (B : Nat) (c : Int) (M : ExactPoly t n)
    (ct : ExactMatrix q n 2 1) : Prop :=
  ∃ W : ErrorPoly n, ct 1 0 - reducePoly q n S * ct 0 0 = reducePoly q n W ∧ polyNorm W ≤ B ∧
    (c : ExactPoly t n) * reducePoly t n W = M

/-- The phase bounds of levels 0, 1, and 2. -/
abbrev B0 : Nat := t * (n * 20 * 1 + 20 + n * 1 * 20) + t / 2
abbrev B1 : Nat := 56258131176
abbrev B2 : Nat := 5667643679

/-- A fresh encryption has a level-0 phase. -/
theorem fresh_phaseFacts {pk : ExactMatrix Q n 2 1} {S E : ErrorPoly n} (hS : polyNorm S ≤ 1)
    (hE : polyNorm E ≤ 20)
    (hpk : pk 1 0 = reducePoly Q n S * pk 0 0 + ((t : Int) : ExactPoly Q n) * reducePoly Q n E)
    {m : Fin n → Int} {ct : ExactMatrix Q n 2 1} (hc : EncFacts pk m ct) :
    PhaseFacts S B0 1 (plainOf m 0 0) ct := by
  obtain ⟨U, E1, E2, hU, hE1, hE2, hc0, hc1⟩ := hc
  refine ⟨_, fresh_phase (A := pk 0 0) (S := S) (E := E) (E2 := E2)
      (M := centeredIntLift (plainOf m 0 0)) (c1 := ct 1 0) hc0 (by rw [hc1, hpk]),
    fresh_phase_norm hS hE hU hE1 hE2 (polyNorm_centeredIntLift (by decide) (plainOf m 0 0)), ?_⟩
  rw [Int.cast_one, one_mul]
  exact reduce_t_lift _ (plainOf m) _ rfl

/-- The key facts the gates use, for the secret `S`. -/
structure GateKey (ek : ExactMatrix QP n 2 3) (S : ErrorPoly n) : Prop where
  hS : polyNorm S ≤ 1
  rows : ∃ (Erk : Fin 3 → ErrorPoly n) (D : ErrorPoly n), (∀ j, polyNorm (Erk j) ≤ 20) ∧
    (∀ i, D.coeff i ≡ (S * S).coeff i [ZMOD Q]) ∧
    ∀ j : Fin 3, ek 1 j = reducePoly QP n S * ek 0 j +
      ((t : Int) : ExactPoly QP n) * reducePoly QP n (Erk j) +
      reducePoly QP n D * ((factor j : Int) : ExactPoly QP n)

/-- Gate 0 carries a level-0 phase to a level-1 phase. -/
theorem gate0_phase {ek : ExactMatrix QP n 2 3} {S : ErrorPoly n} (hk : GateKey ek S)
    {Mx My : ExactPoly t n} {c0 c1 : ExactMatrix Q n 2 1}
    (hx : PhaseFacts S B0 1 Mx c0) (hy : PhaseFacts S B0 1 My c1) {out : ExactMatrix QL n 2 1}
    (hg : GateFacts c0 c1 ek out) : PhaseFacts S B1 q3 (Mx * My) out := by
  obtain ⟨hS, Erk, D, hErk, hD, hrk⟩ := hk
  obtain ⟨Vx, phx, hVx, hMx⟩ := hx
  obtain ⟨Vy, phy, hVy, hMy⟩ := hy
  obtain ⟨relin, ⟨digits, C, K, hδ0, hδ1, hδ2, hcrt, hC, hK, hr0, hr1⟩, ⟨C', K', hC', hK', hms⟩⟩ := hg
  obtain ⟨X, hX, hrel⟩ := relin_phase hD hrk hcrt hK
  set W := Vx * Vy + ((t : Int) : ErrorPoly n) * X
  have phrel : relin 1 0 - reducePoly Q n S * relin 0 0 = reducePoly Q n W := by
    rw [hr0, hr1]
    simp only [W, map_add, map_mul, reducePoly_intCast] at hrel ⊢
    linear_combination hrel + (c1 1 0 - reducePoly Q n S * c1 0 0) * phx + reducePoly Q n Vx * phy
  obtain ⟨W', hW', hphaseL⟩ := modswitch_phase phrel hK'
  refine ⟨W', ?_, ?_, ?_⟩
  · rw [hms 0, hms 1]; exact hphaseL
  · have hXn := relin_error_norm hS hErk hδ0 hδ1 hδ2 hC hX
    have hWn : polyNorm W ≤ n * B0 * B0 + t * ((n * 20 * (q1 / 2) + n * 20 * (q2 / 2) +
        n * 20 * (q3 / 2) + P / 2 + n * 1 * (P / 2)) / P) :=
      (polyNorm_add_le _ _).trans (Nat.add_le_add ((polyNorm_mul_le hn _ _).trans
        (Nat.mul_le_mul (Nat.mul_le_mul_left _ hVx) hVy))
        ((polyNorm_intCast_mul_le _ _).trans (Nat.mul_le_mul_left _ hXn)))
    have hΔn : polyNorm (C' 1 - S * C' 0) ≤ q3 / 2 + n * 1 * (q3 / 2) :=
      (polyNorm_sub_le _ _).trans (Nat.add_le_add (hC' 1)
        ((polyNorm_mul_le hn _ _).trans (Nat.mul_le_mul (Nat.mul_le_mul_left _ hS) (hC' 0))))
    have hW'n := polyNorm_le_div_of_mul (by decide) hW'
      ((polyNorm_add_le _ _).trans (Nat.add_le_add hWn
        ((polyNorm_intCast_mul_le _ _).trans (Nat.mul_le_mul_left _ hΔn))))
    exact hW'n.trans (by decide +kernel)
  · have hred_t := congrArg (reducePoly t n) hW'
    rw [map_mul, map_add, map_mul, reducePoly_intCast, reducePoly_intCast, ht0, zero_mul,
      add_zero] at hred_t
    rw [hred_t, ← hMx, ← hMy]
    simp only [W, map_add, map_mul, reducePoly_intCast, ht0, zero_mul, add_zero, Int.cast_one,
      one_mul]

/-- Gate 1 carries a level-1 phase to a level-2 phase. -/
theorem gate1_phase {ek : ExactMatrix QP n 2 3} {S : ErrorPoly n} (hk : GateKey ek S)
    {Mx My : ExactPoly t n} {c0 c1 : ExactMatrix QL n 2 1}
    (hx : PhaseFacts S B1 q3 Mx c0) (hy : PhaseFacts S B1 q3 My c1) {out : ExactMatrix Q1 n 2 1}
    (hg : GateFacts1 c0 c1 ek out) : PhaseFacts S B2 (q3 * q3 * q2) (Mx * My) out := by
  obtain ⟨hS, Erk, D, hErk, hD, hrk⟩ := hk
  obtain ⟨Vx, phx, hVx, hMx⟩ := hx
  obtain ⟨Vy, phy, hVy, hMy⟩ := hy
  obtain ⟨relin, ⟨digits, C, K, hδ0, hδ1, hcrt, hC, hK, hr0, hr1⟩, ⟨C', K', hC', hK', hms⟩⟩ := hg
  obtain ⟨X, hX, hrel⟩ := relin_phase1 (Erk := fun j ↦ Erk (Fin.castSucc j)) hD (key1_rows hrk)
    hcrt hK
  set W := Vx * Vy + ((t : Int) : ErrorPoly n) * X
  have phrel : relin 1 0 - reducePoly QL n S * relin 0 0 = reducePoly QL n W := by
    rw [hr0, hr1]
    simp only [W, map_add, map_mul, reducePoly_intCast] at hrel ⊢
    linear_combination hrel + (c1 1 0 - reducePoly QL n S * c1 0 0) * phx +
      reducePoly QL n Vx * phy
  obtain ⟨W', hW', hphaseL⟩ := modswitch_phase1 phrel hK'
  refine ⟨W', ?_, ?_, ?_⟩
  · rw [hms 0, hms 1]; exact hphaseL
  · have hXn := relin_error_norm1 hS (fun j ↦ hErk (Fin.castSucc j)) hδ0 hδ1 hC hX
    have hWn : polyNorm W ≤ n * B1 * B1 + t * ((n * 20 * (q1 / 2) + n * 20 * (q2 / 2) +
        P / 2 + n * 1 * (P / 2)) / P) :=
      (polyNorm_add_le _ _).trans (Nat.add_le_add ((polyNorm_mul_le hn _ _).trans
        (Nat.mul_le_mul (Nat.mul_le_mul_left _ hVx) hVy))
        ((polyNorm_intCast_mul_le _ _).trans (Nat.mul_le_mul_left _ hXn)))
    have hΔn : polyNorm (C' 1 - S * C' 0) ≤ q2 / 2 + n * 1 * (q2 / 2) :=
      (polyNorm_sub_le _ _).trans (Nat.add_le_add (hC' 1)
        ((polyNorm_mul_le hn _ _).trans (Nat.mul_le_mul (Nat.mul_le_mul_left _ hS) (hC' 0))))
    have hW'n := polyNorm_le_div_of_mul (by decide) hW'
      ((polyNorm_add_le _ _).trans (Nat.add_le_add hWn
        ((polyNorm_intCast_mul_le _ _).trans (Nat.mul_le_mul_left _ hΔn))))
    exact hW'n.trans (by decide +kernel)
  · have hred_t := congrArg (reducePoly t n) hW'
    rw [map_mul, map_add, map_mul, reducePoly_intCast, reducePoly_intCast, ht0, zero_mul,
      add_zero] at hred_t
    rw [← hMx, ← hMy]
    have hW : reducePoly t n W = reducePoly t n Vx * reducePoly t n Vy := by
      simp only [W, map_add, map_mul, reducePoly_intCast, ht0, zero_mul, add_zero]
    push_cast
    linear_combination ((q3 : ExactPoly t n) * (q3 : ExactPoly t n)) * hred_t +
      ((q3 : ExactPoly t n) * (q3 : ExactPoly t n)) * hW

/-! ## Decryption and plaintext values at every level -/

/-- A small phase decrypts to the plaintext it carries. -/
theorem phase_decode {q : Nat} (hq : 1 < q) {S : ErrorPoly n} {B : Nat} {c : Int}
    {M : ExactPoly t n} {ct : ExactMatrix q n 2 1} {sk : ExactMatrix Q n 1 1}
    (hsq : modulusReduce (p := q) sk 0 0 = reducePoly q n S) (hB : 2 * B < q)
    (h : PhaseFacts S B c M ct) :
    matrixScale c (centeredRebase (p := t) (phaseMat ct sk)) =
      (fun _ _ ↦ M : ExactMatrix t n 1 1) := by
  obtain ⟨W, hW, hWn, hM⟩ := h
  have hsmall (i : Fin n) : 2 * (W.coeff i).natAbs < q := by
    have := natAbs_coeff_le_of_polyNorm hWn i
    omega
  have hphase : phaseMat ct sk 0 0 = reducePoly q n W := by
    simp only [phaseMat, matrixAdd, Matrix.add_apply, matrixMulScalarLeft, matrixNeg,
      Matrix.neg_apply, hsq]
    rw [← hW]
    ring
  refine one_one_of_entry ?_
  show (c : ExactPoly t n) * centeredRebase (p := t) (phaseMat ct sk) 0 0 = M
  rw [centeredRebase_eq, hphase, centeredIntLift_reducePoly hq hn _ hsmall, hM]

theorem modQ1_sk {sk : ExactMatrix Q n 1 1} {S : ErrorPoly n} (hsk : sk 0 0 = reducePoly Q n S) :
    modulusReduce (p := Q1) sk 0 0 = reducePoly Q1 n S := by
  simp only [modulusReduce, hsk]
  exact modulusReduce_reducePoly (by decide) (by decide) (by norm_num) hn _

theorem decode_table_out : Decrypt_1.generatedRoot.table_11 = decodeTable := rfl
theorem decrypt_out_eq (ct : ExactMatrix Q1 n 2 1) (sk : ExactMatrix Q n 1 1) (native : Fin n → Int)
    (hv : polynomialValues true (matrixScale 129102 (centeredRebase (p := t) (phaseMat ct sk)))
      native) :
    Decrypt_1.generatedRoot.comp Decrypt_1_params (ct, sk, ()) =
      pure fun i ↦ native ⟨_, decodeEntry_lt i⟩ := by
  unfold Decrypt_1.generatedRoot.comp
  congr 1
  refine determined_eq ?_ ?_
  · refine ⟨⟨fun _ _ ↦ ct 0 0, modulusReduce sk, fun _ _ ↦ ct 1 0,
      centeredRebase (p := t) (phaseMat ct sk), native, decodeValues,
      fun i ↦ native ⟨_, decodeEntry_lt i⟩⟩, ?_⟩
    dsimp only [Decrypt_1.generatedRoot.body]
    rw [decode_table_out]
    refine ⟨by decide, by decide, by decide, by decide, by decide, by decide, slice_row0 ct,
      ⟨by decide, by norm_num, rfl⟩, by decide, by decide, by decide, by decide, by decide,
      by decide, slice_row1 ct, ⟨by decide, rfl⟩, hv, decodeValues_holds, rfl, fun i ↦ ?_, rfl⟩
    refine ⟨native ⟨_, decodeEntry_lt i⟩, ?_, ?_, ⟨⟨_, decodeEntry_lt i⟩, ?_, rfl⟩, rfl⟩
    · rw [decodeValues_eq]; exact Int.natCast_nonneg _
    · rw [decodeValues_eq]
      show ((packedEntry 13 decodeTable i.val : Nat) : Int) < 8192
      exact_mod_cast decodeEntry_lt i
    · rw [decodeValues_eq]
  · rintro out ⟨w, h⟩
    dsimp only [Decrypt_1.generatedRoot.body] at h
    rw [decode_table_out] at h
    obtain ⟨-, -, -, -, -, -, s0, hred, -, -, -, -, -, -, s1, hreb, hv', htab, -, hpar, rfl⟩ := h
    have e0 : w.w_1_0 = fun _ _ ↦ ct 0 0 :=
      one_one_of_entry (sliceMatrix_entry s0 (by decide) (by decide))
    have e1 : w.w_6_0 = fun _ _ ↦ ct 1 0 :=
      one_one_of_entry (sliceMatrix_entry s1 (by decide) (by decide))
    have h8 : w.w_8_0 = centeredRebase (p := t) (phaseMat ct sk) := by
      rw [hreb.2, e0, e1, hred.2.2]
      rfl
    rw [h8] at hv'
    have h9 := values_unique hv' hv
    funext i
    obtain ⟨w2, -, -, ⟨pos, hpos, hw2⟩, hout⟩ := hpar i
    rw [hout, hw2, h9]
    show native pos = _
    congr 1
    exact Fin.ext (by have := hpos.trans (decode_values _ htab i); exact_mod_cast this)

theorem eval_of_values {M : ExactMatrix t n 1 1} {v : Fin n → Int}
    (h : polynomialValues true M v) (index : Fin n) :
    (v ⟨nttBitReverse 13 index.val, bitReverse_lt _ index.isLt⟩ : ZMod t) =
      oddPowerSum (nativeNttRoot t n : ZMod t) index.val (M 0 0) := by
  simp only [polynomialValues, ite_true] at h
  obtain ⟨-, -, bits, hb, he⟩ := polynomialNttRuns_eval h
  rw [bits_eq hb] at he
  obtain ⟨slot, hs, hv⟩ := he index
  have : slot = ⟨nttBitReverse 13 index.val, bitReverse_lt _ index.isLt⟩ := Fin.ext hs
  subst this
  simp only [Bool.false_eq_true, ite_false] at hv
  rw [hv]
  unfold oddPowerSum
  refine Finset.sum_congr rfl fun c _ ↦ ?_
  rw [zmod_val_cast]

/-- The native slots of a product of plaintexts are the products of their slots. -/
theorem mul_values {Mx My : ExactPoly t n} {x y : Fin n → Int}
    (hx : polynomialValues true (fun _ _ ↦ Mx : ExactMatrix t n 1 1) (nativeOf x))
    (hy : polynomialValues true (fun _ _ ↦ My : ExactMatrix t n 1 1) (nativeOf y)) :
    polynomialValues true (fun _ _ ↦ Mx * My : ExactMatrix t n 1 1)
      (nativeOf fun i ↦ x i * y i % 1032193) := by
  have hroot := nttPrimitiveRoot_pow_n hn native_primitive
  show polynomialValues true _ (prodNative x y)
  refine values_of_eval (fun i ↦ ⟨Int.emod_nonneg _ (by decide), Int.emod_lt_of_pos _ (by decide)⟩)
    fun index ↦ ?_
  simp only [prodNative]
  rw [show ((1032193 : Int)) = ((t : Nat) : Int) from rfl, ZMod.intCast_mod, Int.cast_mul,
    eval_of_values hx, eval_of_values hy, oddPowerSum_mul hn hroot]

/-! ## Switching an argument from level 0 down to level 1 -/

/-- The switch from level 0 to level 1 inside a gate on arguments of both levels: ModDown by
`q3`. -/
abbrev SwitchRuns (c : ExactMatrix Q n 2 1) (out : ExactMatrix QL n 2 1) : Prop :=
  rnsModDownRuns [18014398507892737, 18014398508138497, 18014398508400641] (1032193) c out

theorem switch_spec {c : ExactMatrix Q n 2 1} {out : ExactMatrix QL n 2 1}
    (hms : SwitchRuns c out) : ModSwitchFacts c out := by
  have hdropL : [18014398507892737, 18014398508138497, 18014398508400641].filter
      (fun prime ↦ QL % prime != 0) = [q3] := by decide
  have hspec (r : Fin 2) :=
    rnsModDown_single_exact hms hdropL (by decide) (by decide) (by decide) r 0
  choose correction quotient hbound hexact hquot using hspec
  exact ⟨correction, quotient, hbound, hexact, hquot⟩

theorem switch_exists (c : ExactMatrix Q n 2 1) : ∃ out, SwitchRuns c out :=
  rnsModDownRuns_exists (p := QL)
    (basis := [18014398507892737, 18014398508138497, 18014398508400641])
    (plaintext := 1032193) c (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by decide) (by norm_num) (by decide)

/-- Switching down carries a level-0 phase to a level-1 phase. -/
theorem convert_phase {S : ErrorPoly n} (hS : polyNorm S ≤ 1) {M : ExactPoly t n}
    {c : ExactMatrix Q n 2 1} (hx : PhaseFacts S B0 1 M c) {out : ExactMatrix QL n 2 1}
    (hm : ModSwitchFacts c out) : PhaseFacts S B1 q3 M out := by
  obtain ⟨V, ph, hV, hM⟩ := hx
  obtain ⟨C', K', hC', hK', hms⟩ := hm
  obtain ⟨W', hW', hphaseL⟩ := modswitch_phase ph hK'
  refine ⟨W', ?_, ?_, ?_⟩
  · rw [hms 0, hms 1]; exact hphaseL
  · have hΔn : polyNorm (C' 1 - S * C' 0) ≤ q3 / 2 + n * 1 * (q3 / 2) :=
      (polyNorm_sub_le _ _).trans (Nat.add_le_add (hC' 1)
        ((polyNorm_mul_le hn _ _).trans (Nat.mul_le_mul (Nat.mul_le_mul_left _ hS) (hC' 0))))
    have hW'n := polyNorm_le_div_of_mul (by decide) hW'
      ((polyNorm_add_le _ _).trans (Nat.add_le_add hV
        ((polyNorm_intCast_mul_le _ _).trans (Nat.mul_le_mul_left _ hΔn))))
    exact hW'n.trans (by decide +kernel)
  · have hred_t := congrArg (reducePoly t n) hW'
    rw [map_mul, map_add, map_mul, reducePoly_intCast, reducePoly_intCast, ht0, zero_mul,
      add_zero] at hred_t
    rw [hred_t, ← hM, Int.cast_one, one_mul]

/-! ## The circuit `x₀ · (x₁ · x₂)` -/

/-- The evaluation of the circuit: `x₀` switched down to level 1, the gate on the level-0
arguments `x₁` and `x₂`, and the gate on the two level-1 results. -/
theorem eval_spec {c0 c1 c2 : ExactMatrix Q n 2 1} {ek : ExactMatrix QP n 2 3}
    {out : ExactMatrix Q1 n 2 1}
    (h : Eval.generatedRoot Eval_params (c0, c1, c2, ek, ()) out) :
    ∃ s r, ModSwitchFacts c0 s ∧ GateFacts c1 c2 ek r ∧ GateFacts1 s r ek out := by
  obtain ⟨w, h⟩ := h
  dsimp only [Eval.generatedRoot.body] at h
  obtain ⟨hs,
    g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, g11, g12, g13, g14, g15, g16, g17, g18, g19, g20,
    g21, g22, g23, g24, g25, g26, g27, g28, g29, g30, g31, g32, g33, g34, g35, g36, g37, g38,
    g39, g40, g41, g42, g43, g44, g45, g46, g47, g48, g49, g50, g51, g52, g53, g54, g55, g56, h1⟩ := h
  exact ⟨w.w_1_0, w.w_20_0, switch_spec hs,
    gate_spec ⟨⟨w.w_4_0, w.w_5_0, w.w_6_0, w.w_7_0, w.w_9_0, w.w_10_0, w.w_10_concat_1, w.w_11_0, w.w_12_0, w.w_13_0, w.w_15_0, w.w_16_0, w.w_18_0, w.w_20_0⟩,
      g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, g11, g12, g13, g14, g15, g16, g17, g18, g19, g20,
      g21, g22, g23, g24, g25, g26, g27, g28, g29, g30, g31, g32, g33, g34, g35, g36, g37, g38,
      g39, g40, g41, g42, g43, g44, g45, g46, g47, g48, g49, g50, g51, g52, g53, g54, g55, g56, rfl⟩,
    gate1_spec ⟨⟨w.w_21_0, w.w_22_0, w.w_23_0, w.w_24_0, w.w_26_0, w.w_27_0, w.w_27_concat_1, w.w_28_0, w.w_29_0, w.w_30_0, w.w_31_0, w.w_32_0, w.w_33_0, w.w_35_0, w.w_37_0, w.w_39_0⟩, h1⟩⟩

theorem eval_exists (c0 c1 c2 : ExactMatrix Q n 2 1) (ek : ExactMatrix QP n 2 3) :
    ∃ out, Eval.generatedRoot Eval_params (c0, c1, c2, ek, ()) out := by
  obtain ⟨r, w0, hw0⟩ := gate_exists c1 c2 ek
  dsimp only [Gate0.generatedRoot.body] at hw0
  obtain ⟨
    g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, g11, g12, g13, g14, g15, g16, g17, g18, g19, g20,
    g21, g22, g23, g24, g25, g26, g27, g28, g29, g30, g31, g32, g33, g34, g35, g36, g37, g38,
    g39, g40, g41, g42, g43, g44, g45, g46, g47, g48, g49, g50, g51, g52, g53, g54, g55, g56, rfl⟩ := hw0
  obtain ⟨s, hs⟩ := switch_exists c0
  obtain ⟨out, w1, hw1⟩ := gate1_exists s (w0.w_18_0) ek
  exact ⟨out, ⟨s, w0.w_2_0, w0.w_3_0, w0.w_4_0, w0.w_5_0, w0.w_7_0, w0.w_8_0, w0.w_8_concat_1, w0.w_9_0, w0.w_10_0, w0.w_11_0, w0.w_13_0, w0.w_14_0, w0.w_16_0, w0.w_18_0, w1.w_2_0, w1.w_3_0, w1.w_4_0, w1.w_5_0, w1.w_7_0, w1.w_8_0, w1.w_8_concat_1, w1.w_9_0, w1.w_10_0, w1.w_11_0, w1.w_13_0, w1.w_14_0, w1.w_15_0, w1.w_17_0, w1.w_19_0, w1.w_21_0⟩, hs,
    g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, g11, g12, g13, g14, g15, g16, g17, g18, g19, g20,
    g21, g22, g23, g24, g25, g26, g27, g28, g29, g30, g31, g32, g33, g34, g35, g36, g37, g38,
    g39, g40, g41, g42, g43, g44, g45, g46, g47, g48, g49, g50, g51, g52, g53, g54, g55, g56, hw1⟩

theorem matrixScale_congr {a b : Int} (h : (a : ExactPoly t n) = (b : ExactPoly t n))
    (X : ExactMatrix t n 1 1) : matrixScale a X = matrixScale b X := by
  funext r c
  simp only [matrixScale, h]

theorem scale2_eq :
    ((129102 : Int) : ExactPoly t n) = (((q3 * q3 * q2 : Int)) : ExactPoly t n) := by
  rw [intCast_eq_algebraMap, intCast_eq_algebraMap (value := (q3 * q3 * q2 : Int))]
  congr 1

/-! ## Correctness -/

theorem fresh_facts {pk : ExactMatrix Q n 2 1} {S E : ErrorPoly n} (hS : polyNorm S ≤ 1)
    (hE : polyNorm E ≤ 20)
    (hpk : pk 1 0 = reducePoly Q n S * pk 0 0 + ((t : Int) : ExactPoly Q n) * reducePoly Q n E)
    {m : Fin n → Int} (hm : scheme.ValidMsg m) {ct : ExactMatrix Q n 2 1} (hc : EncFacts pk m ct) :
    PhaseFacts S B0 1 (plainOf m 0 0) ct ∧
      polynomialValues true (fun _ _ ↦ plainOf m 0 0 : ExactMatrix t n 1 1) (nativeOf m) := by
  refine ⟨fresh_phaseFacts hS hE hpk hc, ?_⟩
  have : (fun _ _ ↦ plainOf m 0 0 : ExactMatrix t n 1 1) = plainOf m := one_one_of_entry rfl
  rw [this]
  exact fresh_values _ hm

theorem correctness : CorrectnessClaim := by
  intro messages hvalid
  refine probTrue_of_always ?_
  unfold Scheme.evaluationFails
  refine always_bind keyGen_facts fun keys hk ↦ ?_
  obtain ⟨pk, ek, sk⟩ := keys
  obtain ⟨S, E, Erk, D, hS, hE, hErk, hD, hsk, hpk, hrk⟩ := hk
  have hgk : GateKey ek S := ⟨hS, Erk, D, hErk, hD, hrk⟩
  refine always_bind (encAll_facts pk 3 messages) fun cts hcts ↦ ?_
  obtain ⟨c0, h0, f0⟩ := hcts ⟨0, by decide⟩
  obtain ⟨c1, h1, f1⟩ := hcts ⟨1, by decide⟩
  obtain ⟨c2, h2, f2⟩ := hcts ⟨2, by decide⟩
  obtain ⟨p0, v0⟩ := fresh_facts hS hE hpk (hvalid _) f0
  obtain ⟨p1, v1⟩ := fresh_facts hS hE hpk (hvalid _) f1
  obtain ⟨p2, v2⟩ := fresh_facts hS hE hpk (hvalid _) f2
  have hev : evaluation.eval ek cts =
      (Eval.generatedRoot.comp Eval_params (c0, c1, c2, ek, ()) >>= fun o ↦
        pure (Ciphertext.ct1 o)) := by
    simp only [evaluation]
    rw [h0, h1, h2]
  have hplain : evaluation.plain messages = fun i ↦
      messages ⟨0, by decide⟩ i * (messages ⟨1, by decide⟩ i * messages ⟨2, by decide⟩ i % 1032193) %
        1032193 :=
    plain_eq _ _ _
  show Always _ (evaluation.eval ek cts >>= _)
  rw [hev, bind_assoc]
  unfold Eval.generatedRoot.comp
  simp only [pure_bind]
  have hR := determined_holds (relation := fun outputs ↦
    ∃ witness, Eval.generatedRoot.body Eval_params (c0, c1, c2, ek, ()) outputs witness)
    (eval_exists c0 c1 c2 ek)
  obtain ⟨s, r, hms, hg0, hg1⟩ := eval_spec hR
  set ct := MxxSecurity.determined fun outputs ↦
    ∃ witness, Eval.generatedRoot.body Eval_params (c0, c1, c2, ek, ()) outputs witness
  have hph := gate1_phase hgk (convert_phase hS p0 hms) (gate0_phase hgk p1 p2 hg0) hg1
  have hval := mul_values v0 (mul_values v1 v2)
  have hdec := phase_decode (q := Q1) (by decide) (modQ1_sk hsk) (by decide) hph
  show Always _ (dec sk (Ciphertext.ct1 ct) >>= _)
  simp only [dec]
  rw [decrypt_out_eq ct sk _ (by rw [matrixScale_congr scale2_eq, hdec]; exact hval), pure_bind,
    hplain, decode_fresh]
  exact always_pure (by simp)

end BgvCorrect
