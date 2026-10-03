import Stage_keygen
import Stage_encrypt_x
import Stage_encrypt_y
import Stage_multiply
import Stage_relinearize
import Stage_modswitch
import Stage_decrypt
import Ideal
import RuntimeLemmas

/-!
Each lemma restates one generated BGV stage relation through integer polynomial witnesses:
secrets and errors are reductions of bounded integer polynomials, and every exported value is
an explicit ring expression in them.
-/

namespace MxxFheBgv

open Mxx.Primitives MxxRuntime

abbrev n : Nat := 8192
abbrev t : Nat := 1032193
abbrev q1 : Nat := 18014398507892737
abbrev q2 : Nat := 18014398508138497
abbrev q3 : Nat := 18014398508400641
abbrev P : Nat := 72057594037616641
abbrev Q : Nat := 5846006548020969210596774788483421161649837621249
abbrev QP : Nat := 421249166578543632464236954685046510773404157767471124412817604609
abbrev QL : Nat := 324518553605595287786984016396289

/-- The slot permutations, as the packed tables the generated stages embed: encoding reads logical
slot `encodeTable[j]` into native evaluation `j`, and decoding reads native evaluation
`decodeTable[i]` into logical slot `i`. -/
abbrev encodeTable : Nat := Stage_encrypt_x.generatedRoot.table_14
abbrev decodeTable : Nat := Stage_decrypt.generatedRoot.table_12

/-- The hybrid key-switching gadget `P * Q / q_j` of each digit. -/
def gadget : Fin 3 → Int :=
  ![23384026194045816390345886928489845842762721017857,
    23384026193726801671378034657273812936317868343297,
    23384026193386519304488586269947798086183317045249]

theorem hn : 0 < n := by decide

theorem keygen_spec {outputs}
    (h : Stage_keygen.generatedRoot { unit := () } () outputs) :
    ∃ (S E : ErrorPoly n) (Erk : Fin 3 → ErrorPoly n) (D : ErrorPoly n),
      polyNorm S ≤ 1 ∧ polyNorm E ≤ 20 ∧ (∀ j, polyNorm (Erk j) ≤ 20) ∧
      (∀ i, D.coeff i ≡ (S * S).coeff i [ZMOD Q]) ∧
      outputs.2.2.1 0 0 = reducePoly Q n S ∧
      outputs.1 1 0 = reducePoly Q n S * outputs.1 0 0 +
        ((t : Int) : ExactPoly Q n) * reducePoly Q n E ∧
      ∀ j : Fin 3, outputs.2.1 1 j = reducePoly QP n S * outputs.2.1 0 j +
        ((t : Int) : ExactPoly QP n) * reducePoly QP n (Erk j) +
        reducePoly QP n D * ((gadget j : Int) : ExactPoly QP n) := by
  obtain ⟨witness, _, hS, hE, hpk, _, hred1, hreb, hErk, hred2, hup, hcat1, hcat2, hrk, hout⟩ := h
  subst hout
  obtain ⟨Smat, hSmat, hSbound⟩ := uniformIntervalSample_ternary hS
  obtain ⟨Emat, hEmat, hEbound⟩ := gaussianSample_bounded hE
  obtain ⟨Ermat, hErmat, hErbound⟩ := gaussianSample_bounded hErk
  -- The ternary secret reduced to every ring of the chain.
  have hs : witness.w_1_0 0 0 = reducePoly Q n (Smat 0 0) := by rw [hSmat]; rfl
  have hsmall (i : Fin n) : 2 * ((Smat 0 0).coeff i).natAbs < q1 := by
    have := (coeff_natAbs_le_polyNorm (Smat 0 0) i).trans (hSbound 0 0)
    unfold q1
    omega
  have hs1 : witness.w_9_0 0 0 = reducePoly q1 n (Smat 0 0) := by
    obtain ⟨_, _, h9⟩ := hred1
    rw [h9]
    simp only [modulusReduce, hs]
    exact modulusReduce_reducePoly (by decide) (by decide) (by decide) hn _
  have hsP : witness.w_10_0 0 0 = reducePoly QP n (Smat 0 0) := by
    obtain ⟨_, h10⟩ := hreb
    rw [h10]
    simp only [centeredRebase, hs1]
    exact centeredRebase_reducePoly (by decide) hn _ hsmall
  have hsQ : witness.w_16_0 0 0 = reducePoly Q n (Smat 0 0) := by
    obtain ⟨_, _, h16⟩ := hred2
    rw [h16]
    simp only [modulusReduce, hs]
    exact modulusReduce_reducePoly (by decide) (by decide) (dvd_refl _) hn _
  -- The CRT lift of `s^2` that every relinearization-key target scales.
  obtain ⟨_, _, _, _, _, _, _, _, hT⟩ := hup
  let v : Fin n → Int := fun i ↦
    Int.ofNat ((matrixMulScalarLeft witness.w_16_0 witness.w_16_0 0 0).coeff i).val
  let D : ErrorPoly n := intPoly fun i ↦
    rnsDigit [18014398507892737, 18014398508138497, 18014398508400641] 3 0 false (v i)
  have hT0 : witness.w_18_0 0 0 = reducePoly QP n D := by
    rw [hT ⟨0, by decide⟩ 0 0 0 rfl, polynomialOfCoefficients_eq_reduce]
  have hD (i : Fin n) : D.coeff i ≡ (Smat 0 0 * Smat 0 0).coeff i [ZMOD Q] := by
    rw [intPoly_coeff hn, rnsDigit_three_whole (by decide) (by decide) (by decide)]
    refine (crt_three_centered (by decide) (by decide) (by decide) (by decide) (by decide)
      (by decide) (v i)).trans ?_
    simp only [v, matrixMulScalarLeft, hsQ, ← map_mul, Int.ofNat_eq_natCast]
    rw [reducePoly_coeff (by decide) hn, val_intCast_emod (by decide)]
    exact Int.mod_modEq _ _
  refine ⟨Smat 0 0, Emat 0 0, fun j ↦ Ermat 0 j, D, hSbound 0 0,
    by simpa using hEbound 0 0, fun j ↦ by simpa using hErbound 0 j, hD, hs, ?_, ?_⟩
  · obtain ⟨h0, h1⟩ := concatRows_one_one hpk 0
    simp only
    rw [h0, h1]
    simp only [matrixAdd, matrixMulScalarLeft, Matrix.add_apply, matrixPolynomial_single, hEmat,
      reduceMatrix_apply, hs]
    push_cast
    ring
  · obtain ⟨g0, g1⟩ := concatColumns_one_one hcat1
    obtain ⟨g0', g1', g2⟩ := concatColumns_two_one hcat2
    intro j
    obtain ⟨r0, r1⟩ := concatRows_one_one hrk j
    simp only
    rw [r0, r1]
    simp only [matrixAdd, matrixMulScalarLeft, Matrix.add_apply, matrixPolynomial_single, hErmat,
      reduceMatrix_apply, hsP]
    have hgadget : witness.w_25_0 0 j = witness.w_18_0 0 0 * ((gadget j : Int) : ExactPoly QP n) := by
      fin_cases j
      · change witness.w_25_0 0 0 = witness.w_18_0 0 0 * ((gadget 0 : Int) : ExactPoly QP n)
        rw [g0', g0]; simp [matrixMulScalarLeft, matrixPolynomial_single, gadget]
      · change witness.w_25_0 0 1 = witness.w_18_0 0 0 * ((gadget 1 : Int) : ExactPoly QP n)
        rw [g1', g1]; simp [matrixMulScalarLeft, matrixPolynomial_single, gadget]
      · change witness.w_25_0 0 2 = witness.w_18_0 0 0 * ((gadget 2 : Int) : ExactPoly QP n)
        rw [g2]; simp [matrixMulScalarLeft, matrixPolynomial_single, gadget]
    rw [hgadget, hT0]
    push_cast
    ring

/-- What an encryption stage establishes: `ct = pk * u + t * (e1, e2) + (0, m~)`, where `m~` is
the centered lift of the plaintext polynomial whose native evaluations are the permuted slots. -/
def EncryptionFacts (pk : ExactMatrix Q n 2 1) (x : Fin n → Int) (ct : ExactMatrix Q n 2 1) :
    Prop :=
  ∃ (U E1 E2 : ErrorPoly n) (m : ExactMatrix t n 1 1) (native : Fin n → Int),
    polyNorm U ≤ 1 ∧ polyNorm E1 ≤ 20 ∧ polyNorm E2 ≤ 20 ∧
    polynomialFromValues true native m ∧
    (∀ j : Fin n, ∃ k : Fin n, k.val = packedEntry 13 encodeTable j.val ∧ native j = x k) ∧
    ct 0 0 = pk 0 0 * reducePoly Q n U + ((t : Int) : ExactPoly Q n) * reducePoly Q n E1 ∧
    ct 1 0 = pk 1 0 * reducePoly Q n U + ((t : Int) : ExactPoly Q n) * reducePoly Q n E2 +
      reducePoly Q n (centeredIntLift (m 0 0))

theorem encrypt_x_spec {pk : ExactMatrix Q n 2 1} {x : Fin n → Int} {ct : ExactMatrix Q n 2 1}
    (h : Stage_encrypt_x.generatedRoot { unit := () } (pk, x, ()) ct) : EncryptionFacts pk x ct := by
  obtain ⟨witness, _, _, _, _, _, _, hs0, hU, hE1, _, _, _, _, _, _, hs1, hE2, hvals, _, hpar,
    hfrom, hreb, hcat, hout⟩ := h
  subst hout
  obtain ⟨Umat, hUmat, hUb⟩ := uniformIntervalSample_ternary hU
  obtain ⟨E1mat, hE1mat, hE1b⟩ := gaussianSample_bounded hE1
  obtain ⟨E2mat, hE2mat, hE2b⟩ := gaussianSample_bounded hE2
  have ha : witness.w_1_0 0 0 = pk 0 0 := sliceMatrix_entry hs0 (by decide) (by decide)
  have hb : witness.w_8_0 0 0 = pk 1 0 := sliceMatrix_entry hs1 (by decide) (by decide)
  have htable : polynomialValues false
      (packedPolynomial (q := t) (n := n) 13 n encodeTable) witness.w_15_0 := hvals
  have hentry (i : Fin n) : witness.w_15_0 i = packedEntry 13 encodeTable i.val :=
    polynomialValues_packed (by decide) hn htable
      (fun j ↦ (Nat.mod_lt _ (by decide)).trans_le (by decide)) i
  have hrebase : witness.w_18_0 0 0 = reducePoly Q n (centeredIntLift (witness.w_17_0 0 0)) := by
    rw [hreb.2]
    exact centeredRebase_eq _ 0 0
  obtain ⟨c0, c1⟩ := concatRows_one_one hcat 0
  refine ⟨Umat 0 0, E1mat 0 0, E2mat 0 0, witness.w_17_0, witness.w_16_0, hUb 0 0,
    by simpa using hE1b 0 0, by simpa using hE2b 0 0, hfrom, fun j ↦ ?_, ?_, ?_⟩
  · obtain ⟨value, _, _, ⟨position, hposition, hvalue⟩, hnative⟩ := hpar j
    refine ⟨position, ?_, hnative.trans hvalue⟩
    have := hposition.trans (hentry j)
    exact_mod_cast this
  · rw [c0]
    simp only [matrixAdd, matrixMulScalarLeft, Matrix.add_apply, matrixPolynomial_single, ha,
      hUmat, hE1mat, reduceMatrix_apply]
    push_cast
    ring
  · rw [c1]
    simp only [matrixAdd, matrixMulScalarLeft, Matrix.add_apply, matrixPolynomial_single, hb,
      hUmat, hE2mat, reduceMatrix_apply, hrebase]
    push_cast
    ring

theorem encrypt_y_spec {pk : ExactMatrix Q n 2 1} {x : Fin n → Int} {ct : ExactMatrix Q n 2 1}
    (h : Stage_encrypt_y.generatedRoot { unit := () } (pk, x, ()) ct) : EncryptionFacts pk x ct := by
  obtain ⟨witness, _, _, _, _, _, _, hs0, hU, hE1, _, _, _, _, _, _, hs1, hE2, hvals, _, hpar,
    hfrom, hreb, hcat, hout⟩ := h
  subst hout
  obtain ⟨Umat, hUmat, hUb⟩ := uniformIntervalSample_ternary hU
  obtain ⟨E1mat, hE1mat, hE1b⟩ := gaussianSample_bounded hE1
  obtain ⟨E2mat, hE2mat, hE2b⟩ := gaussianSample_bounded hE2
  have ha : witness.w_1_0 0 0 = pk 0 0 := sliceMatrix_entry hs0 (by decide) (by decide)
  have hb : witness.w_8_0 0 0 = pk 1 0 := sliceMatrix_entry hs1 (by decide) (by decide)
  have htable : polynomialValues false
      (packedPolynomial (q := t) (n := n) 13 n encodeTable) witness.w_15_0 := hvals
  have hentry (i : Fin n) : witness.w_15_0 i = packedEntry 13 encodeTable i.val :=
    polynomialValues_packed (by decide) hn htable
      (fun j ↦ (Nat.mod_lt _ (by decide)).trans_le (by decide)) i
  have hrebase : witness.w_18_0 0 0 = reducePoly Q n (centeredIntLift (witness.w_17_0 0 0)) := by
    rw [hreb.2]
    exact centeredRebase_eq _ 0 0
  obtain ⟨c0, c1⟩ := concatRows_one_one hcat 0
  refine ⟨Umat 0 0, E1mat 0 0, E2mat 0 0, witness.w_17_0, witness.w_16_0, hUb 0 0,
    by simpa using hE1b 0 0, by simpa using hE2b 0 0, hfrom, fun j ↦ ?_, ?_, ?_⟩
  · obtain ⟨value, _, _, ⟨position, hposition, hvalue⟩, hnative⟩ := hpar j
    refine ⟨position, ?_, hnative.trans hvalue⟩
    have := hposition.trans (hentry j)
    exact_mod_cast this
  · rw [c0]
    simp only [matrixAdd, matrixMulScalarLeft, Matrix.add_apply, matrixPolynomial_single, ha,
      hUmat, hE1mat, reduceMatrix_apply]
    push_cast
    ring
  · rw [c1]
    simp only [matrixAdd, matrixMulScalarLeft, Matrix.add_apply, matrixPolynomial_single, hb,
      hUmat, hE2mat, reduceMatrix_apply, hrebase]
    push_cast
    ring

theorem multiply_spec {lhs rhs : ExactMatrix Q n 2 1} {quad : ExactMatrix Q n 3 1}
    (h : Stage_multiply.generatedRoot { unit := () } (lhs, rhs, ()) quad) :
    quad 0 0 = lhs 0 0 * rhs 0 0 ∧ quad 1 0 = lhs 0 0 * rhs 1 0 + lhs 1 0 * rhs 0 0 ∧
      quad 2 0 = lhs 1 0 * rhs 1 0 := by
  obtain ⟨witness, htensor, _, _, _, _, _, _, s0, _, _, _, _, _, _, s1, _, _, _, _, _, _, s2,
    _, _, _, _, _, _, s3, cat1, cat2, hout⟩ := h
  subst hout
  obtain ⟨t0, t1, t2, t3⟩ := tensorRuns_two htensor
  have e0 := sliceMatrix_entry s0 (by decide) (by decide)
  have e1 := sliceMatrix_entry s1 (by decide) (by decide)
  have e2 := sliceMatrix_entry s2 (by decide) (by decide)
  have e3 := sliceMatrix_entry s3 (by decide) (by decide)
  obtain ⟨a0, a1⟩ := concatRows_one_one cat1 0
  obtain ⟨b0, b1, b2⟩ := concatRows_two_one cat2 0
  refine ⟨?_, ?_, ?_⟩
  · rw [b0, a0, e0]; exact t0
  · rw [b1, a1]
    simp only [matrixAdd, Matrix.add_apply, e1, e2]
    change witness.w_2_0 1 0 + witness.w_2_0 2 0 = _
    rw [t1, t2]
  · rw [b2, e3]; exact t3

/-- Relinearization: the leading component's normalized CRT digits, the key product, and
the exact division by `P` that ModDown performs on each output row. -/
def RelinearizationFacts (quad : ExactMatrix Q n 3 1) (rk : ExactMatrix QP n 2 3)
    (out : ExactMatrix Q n 2 1) : Prop :=
  ∃ (digits : Fin 3 → ErrorPoly n) (correction quotient : Fin 2 → ErrorPoly n),
    polyNorm (digits 0) ≤ q1 / 2 ∧ polyNorm (digits 1) ≤ q2 / 2 ∧ polyNorm (digits 2) ≤ q3 / 2 ∧
    (∀ i, ((q2 * q3 : Nat) : Int) * (digits 0).coeff i + ((q1 * q3 : Nat) : Int) * (digits 1).coeff i +
      ((q1 * q2 : Nat) : Int) * (digits 2).coeff i ≡ (canonicalLift (quad 0 0)).coeff i [ZMOD Q]) ∧
    (∀ r, polyNorm (correction r) ≤ P / 2) ∧
    (∀ r : Fin 2, ((P : Int) : ErrorPoly n) * quotient r =
      canonicalLift (∑ j : Fin 3, rk r j * reducePoly QP n (digits j)) +
        ((t : Int) : ErrorPoly n) * correction r) ∧
    out 0 0 = quad 1 0 + reducePoly Q n (quotient 0) ∧
    out 1 0 = quad 2 0 + reducePoly Q n (quotient 1)

theorem relinearize_spec {quad : ExactMatrix Q n 3 1} {rk : ExactMatrix QP n 2 3}
    {out : ExactMatrix Q n 2 1}
    (h : Stage_relinearize.generatedRoot { unit := () } (quad, rk, ()) out) :
    RelinearizationFacts quad rk out := by
  obtain ⟨witness, _, _, _, _, _, _, s1, _, _, _, _, _, _, s2, hcat, _, _, _, _, _, _, s0, hup,
    hdown, hout⟩ := h
  subst hout
  have e1 : witness.w_1_0 0 0 = quad 1 0 := sliceMatrix_entry s1 (by decide) (by decide)
  have e2 : witness.w_2_0 0 0 = quad 2 0 := sliceMatrix_entry s2 (by decide) (by decide)
  have e0 : witness.w_5_0 0 0 = quad 0 0 := sliceMatrix_entry s0 (by decide) (by decide)
  obtain ⟨c0, c1⟩ := concatRows_one_one hcat 0
  obtain ⟨_, _, _, _, _, _, _, _, hdigit⟩ := hup
  let v : Fin n → Int := fun i ↦ Int.ofNat ((witness.w_5_0 0 0).coeff i).val
  let digits : Fin 3 → ErrorPoly n := fun j ↦ intPoly fun i ↦
    rnsDigit [18014398507892737, 18014398508138497, 18014398508400641] 1 j.val true (v i)
  have hdigits (j : Fin 3) : witness.w_6_0 j 0 = reducePoly QP n (digits j) := by
    rw [hdigit ⟨j.val, by omega⟩ 0 j 0 (by simp), polynomialOfCoefficients_eq_reduce]
  have hsingle (i : Fin n) := rnsDigit_three_single (p1 := q1) (p2 := q2) (p3 := q3)
    (by decide) (by decide) (by decide) (v i)
  have hdrop : [18014398507892737, 18014398508138497, 18014398508400641, 72057594037616641].filter
      (fun prime ↦ Q % prime != 0) = [P] := by decide
  have hspec (r : Fin 2) := rnsModDown_single_exact hdown hdrop (by decide) (by decide) hn r 0
  choose correction quotient hbound hexact hquot using hspec
  have hproduct (r : Fin 2) : (matrixMul rk witness.w_6_0) r 0 =
      ∑ j : Fin 3, rk r j * reducePoly QP n (digits j) := by
    simp only [matrixMul, Matrix.mul_apply, hdigits]
  refine ⟨digits, correction, quotient, ?_, ?_, ?_, ?_, hbound, fun r ↦ ?_, ?_, ?_⟩
  · apply polyNorm_le_of_coeff; intro i
    simp only [digits, intPoly_coeff hn, Fin.val_zero, (hsingle i).1]
    exact rnsCentered_natAbs_le (by decide) _
  · apply polyNorm_le_of_coeff; intro i
    simp only [digits, intPoly_coeff hn, Fin.val_one, (hsingle i).2.1]
    exact rnsCentered_natAbs_le (by decide) _
  · apply polyNorm_le_of_coeff; intro i
    simp only [digits, intPoly_coeff hn, Fin.val_two, (hsingle i).2.2]
    exact rnsCentered_natAbs_le (by decide) _
  · intro i
    simp only [digits, intPoly_coeff hn, Fin.val_zero, Fin.val_one, Fin.val_two, (hsingle i).1,
      (hsingle i).2.1, (hsingle i).2.2]
    rw [canonicalLift_coeff hn, ← e0]
    exact crt_three_centered (by decide) (by decide) (by decide) (by decide) (by decide)
      (by decide) (v i)
  · rw [hexact r, hproduct r]; rfl
  · simp only [matrixAdd, Matrix.add_apply, c0, e1, hquot]
  · simp only [matrixAdd, Matrix.add_apply, c1, e2, hquot]

/-- Modulus switching: ModDown by the last prime of each component. -/
def ModSwitchFacts (ct : ExactMatrix Q n 2 1) (out : ExactMatrix QL n 2 1) : Prop :=
  ∃ correction quotient : Fin 2 → ErrorPoly n, (∀ r, polyNorm (correction r) ≤ q3 / 2) ∧
    (∀ r : Fin 2, ((q3 : Int) : ErrorPoly n) * quotient r =
      canonicalLift (ct r 0) + ((t : Int) : ErrorPoly n) * correction r) ∧
    ∀ r : Fin 2, out r 0 = reducePoly QL n (quotient r)

theorem modswitch_spec {ct : ExactMatrix Q n 2 1} {out : ExactMatrix QL n 2 1}
    (h : Stage_modswitch.generatedRoot { unit := () } ct out) : ModSwitchFacts ct out := by
  obtain ⟨witness, hdown, hout⟩ := h
  subst hout
  have hdrop : [18014398507892737, 18014398508138497, 18014398508400641].filter
      (fun prime ↦ QL % prime != 0) = [q3] := by decide
  have hspec (r : Fin 2) := rnsModDown_single_exact hdown hdrop (by decide) (by decide) hn r 0
  choose correction quotient hbound hexact hquot using hspec
  exact ⟨correction, quotient, hbound, hexact, hquot⟩

/-- Decryption: the phase `c1 - s c0`, its centered plaintext scaled by the inverse correction
factor, and the decoded slots read through the decoding table. -/
def DecryptionFacts (ct : ExactMatrix QL n 2 1) (S : ErrorPoly n) (slots : Fin n → Int) : Prop :=
  ∃ phase : ExactMatrix QL n 1 1, phase 0 0 = ct 1 0 - reducePoly QL n S * ct 0 0 ∧
  ∃ (plain : ExactMatrix t n 1 1) (native : Fin n → Int),
    plain 0 0 = reducePoly t n (centeredIntLift (phase 0 0)) * ((998911 : Int) : ExactPoly t n) ∧
    polynomialValues true plain native ∧
    ∀ i : Fin n, ∃ k : Fin n, k.val = packedEntry 13 decodeTable i.val ∧ slots i = native k

theorem decrypt_spec {ct : ExactMatrix QL n 2 1} {sk : ExactMatrix Q n 1 1} {S : ErrorPoly n}
    {outputs} (hsk : sk 0 0 = reducePoly Q n S)
    (h : Stage_decrypt.generatedRoot { unit := () } (ct, sk, ()) outputs) :
    DecryptionFacts ct S outputs := by
  obtain ⟨witness, _, _, _, _, _, _, s0, hred, _, _, _, _, _, _, s1, hreb, hvals, htable, _, hpar,
    hout⟩ := h
  subst hout
  have e0 : witness.w_1_0 0 0 = ct 0 0 := sliceMatrix_entry s0 (by decide) (by decide)
  have e1 : witness.w_6_0 0 0 = ct 1 0 := sliceMatrix_entry s1 (by decide) (by decide)
  have hs : witness.w_3_0 0 0 = reducePoly QL n S := by
    obtain ⟨_, _, h3⟩ := hred
    rw [h3]
    simp only [modulusReduce, hsk]
    exact modulusReduce_reducePoly (by decide) (by decide) (by decide) hn _
  have hdecode : polynomialValues false
      (packedPolynomial (q := t) (n := n) 13 n decodeTable) witness.w_13_0 := htable
  have hentry (i : Fin n) : witness.w_13_0 i = packedEntry 13 decodeTable i.val :=
    polynomialValues_packed (by decide) hn hdecode
      (fun j ↦ (Nat.mod_lt _ (by decide)).trans_le (by decide)) i
  refine ⟨matrixAdd (matrixMulScalarLeft witness.w_1_0 (matrixNeg witness.w_3_0)) witness.w_6_0,
    ?_, matrixMulScalarLeft witness.w_8_0 (matrixPolynomial [998911]), witness.w_11_0, ?_,
    hvals, fun i ↦ ?_⟩
  · simp only [matrixAdd, matrixMulScalarLeft, matrixNeg, Matrix.add_apply, Matrix.neg_apply,
      e0, e1, hs]
    ring
  · simp only [matrixMulScalarLeft, matrixPolynomial_single, hreb.2]
    rw [centeredRebase_eq]
  · obtain ⟨value, _, _, ⟨position, hposition, hvalue⟩, hnative⟩ := hpar i
    refine ⟨position, ?_, hnative.trans hvalue⟩
    have := hposition.trans (hentry i)
    exact_mod_cast this

theorem ideal_spec {x y out : Fin n → Int}
    (h : Ideal.generatedRoot { unit := () } (x, y, ()) out) : ∀ i, out i = x i * y i % t := by
  obtain ⟨witness, _, hpar, hout⟩ := h
  subst hout
  intro i
  obtain ⟨_, hi⟩ := hpar i
  exact hi

end MxxFheBgv
