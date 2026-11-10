import Scheme
import BgvFirstHop
import BgvGate0

set_option backward.isDefEq.respectTransparency false

/-! Correctness facts of the generated BGV scheme for fresh ciphertexts and the gate on two
level-0 arguments; `BgvDepth` extends them to the exported circuit
(`GeneratedScheme.CorrectnessClaim`).

`Always P mx` says that every execution of `mx` returns a value satisfying `P`. Because the
closed forms of key generation and encryption clamp every sample to its law's support, the
noise bounds hold for every execution, and the decryption failure event never occurs. The phase
arithmetic follows the legacy round-trip proof (`crates/fhe/lean/bgv/BgvProof.lean`). -/

namespace BgvCorrect
open MxxSecurity MxxFhe GeneratedScheme Mxx.Primitives MxxRuntime BgvSecurity
/-- Every execution of `mx` returns a value satisfying `P`. -/
def Always {α : Type} (P : α → Prop) (mx : Comp α) : Prop :=
  ∃ my : Comp {a // P a}, mx = Subtype.val <$> my

theorem always_pure {α : Type} {P : α → Prop} {a : α} (h : P a) : Always P (pure a) :=
  ⟨pure ⟨a, h⟩, by simp⟩

theorem always_bind {α β : Type} {P : α → Prop} {Q : β → Prop} {mx : Comp α} {f : α → Comp β}
    (hx : Always P mx) (hf : ∀ a, P a → Always Q (f a)) : Always Q (mx >>= f) := by
  classical
  obtain ⟨my, rfl⟩ := hx
  refine ⟨my >>= fun a ↦ Classical.choose (hf a.1 a.2), ?_⟩
  simp only [map_bind, bind_map_left]
  exact bind_congr fun a ↦ Classical.choose_spec (hf a.1 a.2)

theorem always_true {α : Type} (mx : Comp α) : Always (fun _ ↦ True) mx :=
  ⟨(fun a ↦ ⟨a, trivial⟩) <$> mx, by simp⟩

theorem always_mono {α : Type} {P Q : α → Prop} {mx : Comp α} (h : Always P mx)
    (hPQ : ∀ a, P a → Q a) : Always Q mx := by
  obtain ⟨my, rfl⟩ := h
  exact ⟨(fun a ↦ ⟨a.1, hPQ a.1 a.2⟩) <$> my, by simp [Functor.map_map]⟩

theorem probTrue_of_always {mx : Comp Bool} (h : Always (· = false) mx) : probTrue mx = 0 := by
  obtain ⟨my, rfl⟩ := h
  have : (Subtype.val <$> my : Comp Bool) = (fun _ ↦ false) <$> my := by
    congr 1
    funext a
    exact a.2
  simp only [probTrue, this, OracleComp.evalDist_map_const]
  simp

abbrev q1 : Nat := 18014398507892737
abbrev q2 : Nat := 18014398508138497
abbrev q3 : Nat := 18014398508400641
abbrev P : Nat := 72057594037616641
abbrev QL : Nat := 324518553605595287786984016396289

/-- What key generation establishes: a ternary secret `S`, bounded errors, and the key rows. -/
def KeyFacts (pk : ExactMatrix Q n 2 1) (ek : ExactMatrix QP n 2 3) (sk : ExactMatrix Q n 1 1) :
    Prop :=
  ∃ (S E : ErrorPoly n) (Erk : Fin 3 → ErrorPoly n) (D : ErrorPoly n),
    polyNorm S ≤ 1 ∧ polyNorm E ≤ 20 ∧ (∀ j, polyNorm (Erk j) ≤ 20) ∧
    (∀ i, D.coeff i ≡ (S * S).coeff i [ZMOD Q]) ∧
    sk 0 0 = reducePoly Q n S ∧
    pk 1 0 = reducePoly Q n S * pk 0 0 + ((t : Int) : ExactPoly Q n) * reducePoly Q n E ∧
    ∀ j : Fin 3, ek 1 j = reducePoly QP n S * ek 0 j +
      ((t : Int) : ExactPoly QP n) * reducePoly QP n (Erk j) +
      reducePoly QP n D * ((factor j : Int) : ExactPoly QP n)

theorem errNorm (v : Fin n → Int) : polyNorm (intPoly fun k ↦ errLaw.clamp (v k)) ≤ 20 := by
  apply polyNorm_le_of_coeff
  intro i
  rw [intPoly_coeff (by decide)]
  have := SamplerLaw.clamp_supports errLaw (v i)
  simp only [SamplerLaw.Supports, ne_eq] at this
  rw [if_pos (by norm_num)] at this
  rw [abs_le] at this
  omega

theorem secNorm (vS : Fin (1 * 1 * n) → Int) : polyNorm (intPoly (secOf vS)) ≤ 1 := by
  apply polyNorm_le_of_coeff
  intro i
  rw [intPoly_coeff (by decide)]
  have := secOf_bound vS i
  omega
theorem keyGen_facts : Always (fun x : ExactMatrix Q n 2 1 × ExactMatrix QP n 2 3 × ExactMatrix Q n 1 1 ↦
    KeyFacts x.1 x.2.1 x.2.2) keyGen := by
  unfold keyGen
  rw [keygen_eq]
  simp only [bind_assoc, pure_bind]
  refine always_bind (always_true _) fun ka _ ↦ always_bind (always_true _) fun vS _ ↦
    always_bind (always_true _) fun ve _ ↦ always_bind (always_true _) fun pa _ ↦
    always_bind (always_true _) fun vE _ ↦ always_pure ?_
  refine ⟨intPoly (secOf vS), intPoly fun k ↦ errLaw.clamp (vE k),
    fun j ↦ intPoly fun k ↦ errLaw.clamp (ve (finProdFinEquiv (finProdFinEquiv ((0 : Fin 1), j), k))),
    intPoly fun i ↦ rnsDigit basisQ 3 0 false (Int.ofNat ((matrixMulScalarLeft
      (modulusReduce (p := Q) (mat11 (q := Q) ternLaw vS))
      (modulusReduce (p := Q) (mat11 (q := Q) ternLaw vS)) 0 0).coeff i).val),
    ?_, ?_, fun j ↦ ?_, ?_, ?_, ?_, fun j ↦ ?_⟩
  · exact secNorm vS
  · exact errNorm (fun k ↦ vE k)
  · exact errNorm _
  · exact sqLift_modEq vS
  · dsimp only
    exact mat11_reduce vS
  · dsimp only
    rw [rowsOf_one, rowsOf_zero]
    simp only [pkRaw, mat11_coeffs, polyOfInts, polynomialOfCoefficients_eq_reduce, tPoly]
    rw [show (intPoly fun k : Fin n ↦ ternLaw.clamp (vS k)) = intPoly (secOf vS) from rfl]
    push_cast
    ring
  · dsimp only
    rw [rowsOf_one, rowsOf_zero]
    simp only [keyRaw, secQP_eq, sqLift, polynomialOfCoefficients_eq_reduce, tPoly, matrixOf]
    push_cast
    ring

/-- What a fresh encryption of `m` under `pk` establishes. -/
def EncFacts (pk : ExactMatrix Q n 2 1) (m : Fin n → Int) (ct : ExactMatrix Q n 2 1) : Prop :=
  ∃ U E1 E2 : ErrorPoly n, polyNorm U ≤ 1 ∧ polyNorm E1 ≤ 20 ∧ polyNorm E2 ≤ 20 ∧
    ct 0 0 = pk 0 0 * reducePoly Q n U + ((t : Int) : ExactPoly Q n) * reducePoly Q n E1 ∧
    ct 1 0 = pk 1 0 * reducePoly Q n U + ((t : Int) : ExactPoly Q n) * reducePoly Q n E2 +
      reducePoly Q n (centeredIntLift (plainOf m 0 0))

/-- A level-0 ciphertext that encrypts `m` under `pk`. -/
def Fresh (pk : ExactMatrix Q n 2 1) (m : Fin n → Int) (c : Ciphertext) : Prop :=
  ∃ ct, c = Ciphertext.ct0 ct ∧ EncFacts pk m ct
theorem enc_facts (pk : ExactMatrix Q n 2 1) (m : Fin n → Int) :
    Always (Fresh pk m) (scheme.enc pk m) := by
  show Always _ (enc pk m)
  unfold enc
  rw [encrypt_eq]
  simp only [bind_assoc, pure_bind]
  refine always_bind (always_true _) fun vU _ ↦ always_bind (always_true _) fun v1 _ ↦
    always_bind (always_true _) fun v2 _ ↦ always_pure ⟨_, rfl, intPoly (secOf vU),
      intPoly fun k ↦ errLaw.clamp (v1 k), intPoly fun k ↦ errLaw.clamp (v2 k), ?_, ?_, ?_, ?_, ?_⟩
  · exact secNorm vU
  · exact errNorm (fun k ↦ v1 k)
  · exact errNorm (fun k ↦ v2 k)
  · rw [encOf_zero]
    simp only [mat11_coeffs, polyOfInts, polynomialOfCoefficients_eq_reduce, tPoly]
    rw [show (intPoly fun k : Fin n ↦ ternLaw.clamp (vU k)) = intPoly (secOf vU) from rfl]
    push_cast
    ring
  · rw [encOf_one]
    simp only [mat11_coeffs, polyOfInts, polynomialOfCoefficients_eq_reduce, tPoly, encodeSlots,
      centeredRebase_eq]
    rw [show (intPoly fun k : Fin n ↦ ternLaw.clamp (vU k)) = intPoly (secOf vU) from rfl]
    push_cast
    ring

theorem encAll_facts (pk : ExactMatrix Q n 2 1) :
    ∀ (count : Nat) (msgs : Fin count → Msg),
      Always (fun cts ↦ ∀ i, Fresh pk (msgs i) (cts i)) (scheme.encAll pk count msgs)
  | 0, _ => always_pure fun i ↦ i.elim0
  | count + 1, msgs => by
    simp only [Scheme.encAll]
    refine always_bind (enc_facts pk (msgs 0)) fun head hhead ↦
      always_bind (encAll_facts pk count fun i ↦ msgs i.succ) fun tail htail ↦ always_pure ?_
    intro i
    refine Fin.cases ?_ (fun j ↦ ?_) i
    · simpa using hhead
    · simpa using htail j

/-! ## Decryption -/

/-- The slot-permutation table of the decoding. -/
abbrev decodeTable : Nat := Decrypt_0.generatedRoot.table_10

theorem decodeEntry_lt (i : Fin n) : packedEntry 13 decodeTable i.val < n :=
  (Nat.mod_lt _ (by decide)).trans_le (by decide)

theorem decode_values (w : Fin n → Int)
    (h : polynomialValues false (packedPolynomial (q := t) (n := n) 13 8192 decodeTable) w) :
    ∀ i, w i = packedEntry 13 decodeTable i.val :=
  polynomialValues_packed (by decide) (by decide) h
    (fun j ↦ (Nat.mod_lt _ (by decide)).trans_le (by decide))

/-- The coefficient values of the packed decoding table. -/
noncomputable def decodeValues : Fin n → Int := fun i ↦
  (((packedPolynomial (q := t) (n := n) 13 8192 decodeTable 0 0).coeff i).val : Int)

theorem decodeValues_holds :
    polynomialValues false (packedPolynomial (q := t) (n := n) 13 8192 decodeTable) decodeValues := by
  simp only [polynomialValues, Bool.false_eq_true, if_false]
  intro i
  rfl

theorem decodeValues_eq (i : Fin n) : decodeValues i = packedEntry 13 decodeTable i.val :=
  decode_values _ decodeValues_holds i

set_option maxRecDepth 100000 in
theorem bitReverse_involutive (k : Nat) (hk : k < n) :
    nttBitReverse 13 (nttBitReverse 13 k) = k ∧ nttBitReverse 13 k < n := by
  have := checkBelow_spec (test := fun k ↦
    bitReverseFast 13 (bitReverseFast 13 k 0) 0 == k && decide (bitReverseFast 13 k 0 < 8192))
    (bound := 8192) (by decide +kernel) k hk
  simp only [bitReverseFast_eq, zero_mul, zero_add, Bool.and_eq_true, beq_iff_eq,
    decide_eq_true_eq] at this
  exact this

theorem bits_eq {bits : Nat} (h : n = 2 ^ bits) : bits = 13 :=
  Nat.pow_right_injective (le_refl 2) (h.symm.trans (by decide))

/-- The native evaluations of a plaintext are unique. -/
theorem values_unique {plain : ExactMatrix t n 1 1} {v v' : Fin n → Int}
    (h : polynomialValues true plain v) (h' : polynomialValues true plain v') : v = v' := by
  simp only [polynomialValues, if_true] at h h'
  obtain ⟨-, hr, bits, hb, he⟩ := polynomialNttRuns_eval h
  obtain ⟨-, hr', bits', hb', he'⟩ := polynomialNttRuns_eval h'
  rw [bits_eq hb] at he
  rw [bits_eq hb'] at he'
  funext s
  obtain ⟨hinv, hlt⟩ := bitReverse_involutive s.val s.isLt
  obtain ⟨s0, hs0, hv0⟩ := he ⟨_, hlt⟩
  obtain ⟨s1, hs1, hv1⟩ := he' ⟨_, hlt⟩
  have e0 : s0 = s := Fin.ext (hs0.trans hinv)
  have e1 : s1 = s := Fin.ext (hs1.trans hinv)
  rw [e0] at hv0
  rw [e1] at hv1
  simp only [Bool.false_eq_true, if_false] at hv0 hv1
  have hmod := (ZMod.intCast_eq_intCast_iff' _ _ t).mp (hv0.trans hv1.symm)
  rw [Int.emod_eq_of_lt (hr s).1 (hr s).2, Int.emod_eq_of_lt (hr' s).1 (hr' s).2] at hmod
  exact hmod

/-- The decryption phase `c₁ - s c₀` as the graph computes it. -/
noncomputable def phaseMat {q : Nat} (ct : ExactMatrix q n 2 1) (sk : ExactMatrix Q n 1 1) :
    ExactMatrix q n 1 1 :=
  matrixAdd (matrixMulScalarLeft (fun _ _ ↦ ct 0 0) (matrixNeg (modulusReduce (p := q) sk)))
    (fun _ _ ↦ ct 1 0)

theorem one_one_of_entry {q : Nat} {a b : ExactMatrix q n 1 1} (h : a 0 0 = b 0 0) : a = b := by
  funext r c
  fin_cases r; fin_cases c
  exact h
theorem decrypt0_eq (ct : ExactMatrix Q n 2 1) (sk : ExactMatrix Q n 1 1) (native : Fin n → Int)
    (hv : polynomialValues true (centeredRebase (p := t) (phaseMat ct sk)) native) :
    Decrypt_0.generatedRoot.comp Decrypt_0_params (ct, sk, ()) =
      pure fun i ↦ native ⟨_, decodeEntry_lt i⟩ := by
  unfold Decrypt_0.generatedRoot.comp
  congr 1
  refine determined_eq ?_ ?_
  · refine ⟨⟨fun _ _ ↦ ct 0 0, modulusReduce sk, fun _ _ ↦ ct 1 0,
      centeredRebase (p := t) (phaseMat ct sk), native, decodeValues,
      fun i ↦ native ⟨_, decodeEntry_lt i⟩⟩, ?_⟩
    dsimp only [Decrypt_0.generatedRoot.body]
    refine ⟨by decide, by decide, by decide, by decide, by decide, by decide, slice_row0 ct,
      ⟨by decide, dvd_refl _, rfl⟩, by decide, by decide, by decide, by decide, by decide,
      by decide, slice_row1 ct, ⟨by decide, rfl⟩, hv, decodeValues_holds, rfl, fun i ↦ ?_, rfl⟩
    refine ⟨native ⟨_, decodeEntry_lt i⟩, ?_, ?_, ⟨⟨_, decodeEntry_lt i⟩, ?_, rfl⟩, rfl⟩
    · rw [decodeValues_eq]; exact Int.natCast_nonneg _
    · rw [decodeValues_eq]
      show ((packedEntry 13 decodeTable i.val : Nat) : Int) < 8192
      exact_mod_cast decodeEntry_lt i
    · rw [decodeValues_eq]
  · rintro out ⟨w, h⟩
    dsimp only [Decrypt_0.generatedRoot.body] at h
    obtain ⟨-, -, -, -, -, -, s0, hred, -, -, -, -, -, -, s1, hreb, hv', htab, -, hpar, rfl⟩ := h
    have e0 : w.w_1_0 = fun _ _ ↦ ct 0 0 := one_one_of_entry (sliceMatrix_entry s0 (by decide) (by decide))
    have e1 : w.w_6_0 = fun _ _ ↦ ct 1 0 := one_one_of_entry (sliceMatrix_entry s1 (by decide) (by decide))
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

/-! ## The gate: multiply, relinearize, and switch down -/

/-- Relinearization of `(d0, d1, d2)`: the normalized CRT digits of `d0`, the key product, and
the exact division by `P`. -/
def RelinFacts (d0 d1 d2 : ExactPoly Q n) (rk : ExactMatrix QP n 2 3) (out : ExactMatrix Q n 2 1) :
    Prop :=
  ∃ (digits : Fin 3 → ErrorPoly n) (correction quotient : Fin 2 → ErrorPoly n),
    polyNorm (digits 0) ≤ q1 / 2 ∧ polyNorm (digits 1) ≤ q2 / 2 ∧ polyNorm (digits 2) ≤ q3 / 2 ∧
    (∀ i, ((q2 * q3 : Nat) : Int) * (digits 0).coeff i + ((q1 * q3 : Nat) : Int) * (digits 1).coeff i +
      ((q1 * q2 : Nat) : Int) * (digits 2).coeff i ≡ (canonicalLift d0).coeff i [ZMOD Q]) ∧
    (∀ r, polyNorm (correction r) ≤ P / 2) ∧
    (∀ r : Fin 2, ((P : Int) : ErrorPoly n) * quotient r =
      canonicalLift (∑ j : Fin 3, rk r j * reducePoly QP n (digits j)) +
        ((t : Int) : ErrorPoly n) * correction r) ∧
    out 0 0 = d1 + reducePoly Q n (quotient 0) ∧ out 1 0 = d2 + reducePoly Q n (quotient 1)

/-- Modulus switching: ModDown by the last prime of each component. -/
def ModSwitchFacts (ct : ExactMatrix Q n 2 1) (out : ExactMatrix QL n 2 1) : Prop :=
  ∃ correction quotient : Fin 2 → ErrorPoly n, (∀ r, polyNorm (correction r) ≤ q3 / 2) ∧
    (∀ r : Fin 2, ((q3 : Int) : ErrorPoly n) * quotient r =
      canonicalLift (ct r 0) + ((t : Int) : ErrorPoly n) * correction r) ∧
    ∀ r : Fin 2, out r 0 = reducePoly QL n (quotient r)

/-- What the gate establishes about its output. -/
def GateFacts (c0 c1 : ExactMatrix Q n 2 1) (rk : ExactMatrix QP n 2 3) (out : ExactMatrix QL n 2 1) :
    Prop :=
  ∃ relin : ExactMatrix Q n 2 1,
    RelinFacts (c0 0 0 * c1 0 0) (c0 0 0 * c1 1 0 + c0 1 0 * c1 0 0) (c0 1 0 * c1 1 0) rk relin ∧
    ModSwitchFacts relin out
theorem gate_spec {c0 c1 : ExactMatrix Q n 2 1} {rk : ExactMatrix QP n 2 3} {out : ExactMatrix QL n 2 1}
    (h : Gate0.generatedRoot Gate0.params (c0, c1, rk, ()) out) : GateFacts c0 c1 rk out := by
  obtain ⟨w, h⟩ := h
  dsimp only [Gate0.generatedRoot.body] at h
  obtain ⟨ht, -, -, -, -, -, -, s3, -, -, -, -, -, -, s4, -, -, -, -, -, -, s5, -, -, -, -, -, -, s7, cat8c, cat8, -, -, -, -, -, -, s9, -, -, -, -, -, -, s10, cat11, -, -, -, -, -, -, s13, hup, hdown, hms, rfl⟩ := h
  obtain ⟨t0, t1, t2, t3⟩ := tensorRuns_two ht
  have e3 := sliceMatrix_entry s3 (by decide) (by decide)
  have e4 := sliceMatrix_entry s4 (by decide) (by decide)
  have e5 := sliceMatrix_entry s5 (by decide) (by decide)
  have e7 := sliceMatrix_entry s7 (by decide) (by decide)
  have e9 := sliceMatrix_entry s9 (by decide) (by decide)
  have e10 := sliceMatrix_entry s10 (by decide) (by decide)
  have e13 := sliceMatrix_entry s13 (by decide) (by decide)
  obtain ⟨a0, a1⟩ := concatRows_one_one cat8c 0
  obtain ⟨b0, b1, b2⟩ := concatRows_two_one cat8 0
  obtain ⟨c0', c1'⟩ := concatRows_one_one cat11 0
  have hd0 : w.w_13_0 0 0 = c0 0 0 * c1 0 0 := by rw [e13]; exact (b0.trans (a0.trans e3)).trans t0
  have hd1 : w.w_11_0 0 0 = c0 0 0 * c1 1 0 + c0 1 0 * c1 0 0 := by
    rw [c0', e9]
    refine b1.trans (a1.trans ?_)
    simp only [matrixAdd, Matrix.add_apply, e4, e5]
    change w.w_2_0 1 0 + w.w_2_0 2 0 = _
    rw [t1, t2]
  have hd2 : w.w_11_0 1 0 = c0 1 0 * c1 1 0 := by
    rw [c1', e10]
    exact (b2.trans e7).trans t3
  -- Relinearization.
  obtain ⟨_, _, _, _, _, _, _, _, hdigit⟩ := hup
  let v : Fin n → Int := fun i ↦ Int.ofNat ((w.w_13_0 0 0).coeff i).val
  let digits : Fin 3 → ErrorPoly n := fun j ↦ intPoly fun i ↦
    rnsDigit [18014398507892737, 18014398508138497, 18014398508400641] 1 j.val true (v i)
  have hdigits (j : Fin 3) : w.w_14_0 j 0 = reducePoly QP n (digits j) := by
    rw [hdigit ⟨j.val, by omega⟩ 0 j 0 (by simp), polynomialOfCoefficients_eq_reduce]
  have hsingle (i : Fin n) := rnsDigit_three_single (p1 := q1) (p2 := q2) (p3 := q3)
    (by decide) (by decide) (by decide) (v i)
  have hdrop : [18014398507892737, 18014398508138497, 18014398508400641, 72057594037616641].filter
      (fun prime ↦ Q % prime != 0) = [P] := by decide
  have hspec (r : Fin 2) := rnsModDown_single_exact hdown hdrop (by decide) (by decide) (by decide) r 0
  choose correction quotient hbound hexact hquot using hspec
  have hproduct (r : Fin 2) : (matrixMul rk w.w_14_0) r 0 =
      ∑ j : Fin 3, rk r j * reducePoly QP n (digits j) := by
    simp only [matrixMul, Matrix.mul_apply, hdigits]
  -- Modulus switching.
  have hdropL : [18014398507892737, 18014398508138497, 18014398508400641].filter
      (fun prime ↦ QL % prime != 0) = [q3] := by decide
  have hspecL (r : Fin 2) := rnsModDown_single_exact hms hdropL (by decide) (by decide) (by decide) r 0
  choose correctionL quotientL hboundL hexactL hquotL using hspecL
  refine ⟨matrixAdd w.w_11_0 w.w_16_0, ⟨digits, correction, quotient, ?_, ?_, ?_, ?_, hbound,
    fun r ↦ ?_, ?_, ?_⟩, ⟨correctionL, quotientL, hboundL, hexactL, hquotL⟩⟩
  · apply polyNorm_le_of_coeff; intro i
    simp only [digits, intPoly_coeff (by decide : 0 < n), Fin.val_zero, (hsingle i).1]
    exact rnsCentered_natAbs_le (by decide) _
  · apply polyNorm_le_of_coeff; intro i
    simp only [digits, intPoly_coeff (by decide : 0 < n), Fin.val_one, (hsingle i).2.1]
    exact rnsCentered_natAbs_le (by decide) _
  · apply polyNorm_le_of_coeff; intro i
    simp only [digits, intPoly_coeff (by decide : 0 < n), Fin.val_two, (hsingle i).2.2]
    exact rnsCentered_natAbs_le (by decide) _
  · intro i
    simp only [digits, intPoly_coeff (by decide : 0 < n), Fin.val_zero, Fin.val_one, Fin.val_two,
      (hsingle i).1, (hsingle i).2.1, (hsingle i).2.2]
    rw [canonicalLift_coeff (by decide), ← hd0]
    exact crt_three_centered (by decide) (by decide) (by decide) (by decide) (by decide)
      (by decide) (v i)
  · rw [hexact r, hproduct r]; rfl
  · simp only [matrixAdd, Matrix.add_apply, hd1, hquot]
  · simp only [matrixAdd, Matrix.add_apply, hd2, hquot]

theorem rnsModDownRuns_exists {q p rows columns : Nat} {basis : List Nat} {plaintext : Int}
    (input : ExactMatrix q n rows columns) (h1 : basis.prod = q) (h2 : basis ≠ [])
    (h3 : basis.Pairwise Nat.Coprime) (h4 : ∀ prime ∈ basis, 2 < prime ∧ (2 * n) ∣ (prime - 1))
    (h5 : 1 < p) (h6 : p < q) (h7 : p * (basis.filter (fun prime ↦ p % prime != 0)).prod = q)
    (h8 : 2 ≤ plaintext)
    (h9 : Int.gcd plaintext (Int.ofNat (basis.filter (fun prime ↦ p % prime != 0)).prod) = 1) :
    ∃ output : ExactMatrix p n rows columns, rnsModDownRuns basis plaintext input output :=
  ⟨_, h1, h2, h3, h4, h5, h6, h7, h8, h9, fun _ _ ↦ rfl⟩
theorem gate_exists (c0 c1 : ExactMatrix Q n 2 1) (rk : ExactMatrix QP n 2 3) :
    ∃ out, Gate0.generatedRoot Gate0.params (c0, c1, rk, ()) out := by
  let w2 : ExactMatrix Q n 4 1 := fun r _ ↦
    ![c0 0 0 * c1 0 0, c0 0 0 * c1 1 0, c0 1 0 * c1 0 0, c0 1 0 * c1 1 0] r
  let w8 : ExactMatrix Q n 3 1 := fun r _ ↦ ![w2 0 0, w2 1 0 + w2 2 0, w2 3 0] r
  let w11 : ExactMatrix Q n 2 1 := fun r _ ↦ ![w8 1 0, w8 2 0] r
  let w14 : ExactMatrix QP n 3 1 := fun r _ ↦ polynomialOfCoefficients fun i ↦
    rnsDigit [18014398507892737, 18014398508138497, 18014398508400641] 1 r.val true
      (Int.ofNat ((w8 0 0).coeff i).val)
  have hup : rnsModUpRuns [18014398507892737, 18014398508138497, 18014398508400641] 1 true
      (fun _ _ ↦ w8 0 0 : ExactMatrix Q n 1 1) w14 := by
    refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
      by norm_num, fun digit row outRow column h ↦ ?_⟩
    fin_cases digit <;> fin_cases row <;> fin_cases outRow <;> fin_cases column <;>
      first | (simp at h; done) | rfl
  obtain ⟨w16, h16⟩ := rnsModDownRuns_exists (p := Q)
    (basis := [18014398507892737, 18014398508138497, 18014398508400641, 72057594037616641])
    (plaintext := 1032193) (matrixMul rk w14) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by decide) (by norm_num) (by decide)
  obtain ⟨w18, h18⟩ := rnsModDownRuns_exists (p := QL)
    (basis := [18014398507892737, 18014398508138497, 18014398508400641])
    (plaintext := 1032193) (matrixAdd w11 w16) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by decide) (by norm_num) (by decide)
  refine ⟨w18, ⟨w2, fun _ _ ↦ w2 0 0, fun _ _ ↦ w2 1 0, fun _ _ ↦ w2 2 0, fun _ _ ↦ w2 3 0, w8,
    (fun r _ ↦ ![w2 0 0, w2 1 0 + w2 2 0] r : ExactMatrix Q n 2 1), fun _ _ ↦ w8 1 0,
    fun _ _ ↦ w8 2 0, w11, fun _ _ ↦ w8 0 0, w14, w16, w18⟩, ?_⟩
  dsimp only [Gate0.generatedRoot.body]
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
      by decide,
      by decide,
      by decide,
      by decide,
      by decide,
      by decide,
      ⟨by decide, by decide, by decide, by decide, by decide, by decide, by decide, by decide, fun row column _ _ ↦ by fin_cases row; fin_cases column; rfl⟩,
      hup,
      h16,
      h18,
      rfl⟩

/-- The plaintext value of the circuit `x₀ · (x₁ · x₂)`: slot-wise products modulo `t`. -/
theorem plain_eq (x y z : Fin n → Int) :
    (determined fun outputs : Fin 8192 → Int ↦ Plain.generatedRoot Plain_params (x, y, z, ()) outputs) =
      fun i ↦ x i * (y i * z i % 1032193) % 1032193 := by
  refine determined_eq ⟨⟨fun i ↦ y i * z i % 1032193, fun i ↦ x i * (y i * z i % 1032193) % 1032193⟩,
    rfl, fun i ↦ ⟨show (1032193 : Int) ≠ 0 by decide, rfl⟩, rfl,
    fun i ↦ ⟨show (1032193 : Int) ≠ 0 by decide, rfl⟩, rfl⟩ ?_
  rintro out ⟨w, -, h4, -, h5, rfl⟩
  funext i
  rw [(h5 i).2, (h4 i).2]

/-! ## Phase arithmetic (ported from the legacy round-trip proof) -/

theorem hn : 0 < n := by decide

theorem polyNorm_sub_le (x y : ErrorPoly n) : polyNorm (x - y) ≤ polyNorm x + polyNorm y := by
  rw [sub_eq_add_neg]
  exact (polyNorm_add_le _ _).trans (by rw [polyNorm_neg])

/-- The fresh-encryption phase `c1 - s c0 = t (e u + e2 - s e1) + m~` over the integers. -/
theorem fresh_phase {A : ExactPoly Q n} {S E U E1 E2 M : ErrorPoly n} {c0 c1 : ExactPoly Q n}
    (hc0 : c0 = A * reducePoly Q n U + ((t : Int) : ExactPoly Q n) * reducePoly Q n E1)
    (hc1 : c1 = (reducePoly Q n S * A + ((t : Int) : ExactPoly Q n) * reducePoly Q n E) *
      reducePoly Q n U + ((t : Int) : ExactPoly Q n) * reducePoly Q n E2 + reducePoly Q n M) :
    c1 - reducePoly Q n S * c0 =
      reducePoly Q n (((t : Int) : ErrorPoly n) * (E * U + E2 - S * E1) + M) := by
  subst hc0 hc1
  simp only [map_add, map_mul, map_sub, reducePoly_intCast]
  ring

/-- The static bound on a fresh phase. -/
theorem fresh_phase_norm {S E U E1 E2 M : ErrorPoly n} (hS : polyNorm S ≤ 1)
    (hE : polyNorm E ≤ 20) (hU : polyNorm U ≤ 1) (hE1 : polyNorm E1 ≤ 20)
    (hE2 : polyNorm E2 ≤ 20) (hM : polyNorm M ≤ t / 2) :
    polyNorm (((t : Int) : ErrorPoly n) * (E * U + E2 - S * E1) + M) ≤
      t * (n * 20 * 1 + 20 + n * 1 * 20) + t / 2 := by
  have hEU := (polyNorm_mul_le hn E U).trans (Nat.mul_le_mul (Nat.mul_le_mul_left _ hE) hU)
  have hSE := (polyNorm_mul_le hn S E1).trans (Nat.mul_le_mul (Nat.mul_le_mul_left _ hS) hE1)
  have hinner : polyNorm (E * U + E2 - S * E1) ≤ n * 20 * 1 + 20 + n * 1 * 20 :=
    (polyNorm_sub_le _ _).trans (Nat.add_le_add ((polyNorm_add_le _ _).trans
      (Nat.add_le_add hEU hE2)) hSE)
  refine (polyNorm_add_le _ _).trans (Nat.add_le_add ?_ hM)
  refine (polyNorm_intCast_mul_le _ _).trans ?_
  exact Nat.mul_le_mul_left _ hinner


theorem factor_eq : factor 0 = (P : Int) * ((q2 * q3 : Nat) : Int) ∧
    factor 1 = (P : Int) * ((q1 * q3 : Nat) : Int) ∧ factor 2 = (P : Int) * ((q1 * q2 : Nat) : Int) := by
  refine ⟨?_, ?_, ?_⟩ <;> simp [factor]

theorem intCast_errorPoly_mul (a b : Int) :
    (((a * b : Int)) : ErrorPoly n) = (a : ErrorPoly n) * (b : ErrorPoly n) := by push_cast; ring

/-- Relinearization moves `s^2 d0` into the pair and adds `t X` for a key-switching error `X`
with `P X = sum_j e_j d_j + c1 - s c0`. -/
theorem relin_phase {S D : ErrorPoly n} {Erk : Fin 3 → ErrorPoly n} {rk : ExactMatrix QP n 2 3}
    {digits : Fin 3 → ErrorPoly n} {C K : Fin 2 → ErrorPoly n} {d0 : ExactPoly Q n}
    (hD : ∀ i, D.coeff i ≡ (S * S).coeff i [ZMOD Q])
    (hrk : ∀ j : Fin 3, rk 1 j = reducePoly QP n S * rk 0 j +
      ((t : Int) : ExactPoly QP n) * reducePoly QP n (Erk j) +
      reducePoly QP n D * ((factor j : Int) : ExactPoly QP n))
    (hcrt : ∀ i, ((q2 * q3 : Nat) : Int) * (digits 0).coeff i +
      ((q1 * q3 : Nat) : Int) * (digits 1).coeff i + ((q1 * q2 : Nat) : Int) * (digits 2).coeff i ≡
        (canonicalLift d0).coeff i [ZMOD Q])
    (hK : ∀ r : Fin 2, ((P : Int) : ErrorPoly n) * K r =
      canonicalLift (∑ j : Fin 3, rk r j * reducePoly QP n (digits j)) +
        ((t : Int) : ErrorPoly n) * C r) :
    ∃ X : ErrorPoly n, ((P : Int) : ErrorPoly n) * X =
        Erk 0 * digits 0 + Erk 1 * digits 1 + Erk 2 * digits 2 + C 1 - S * C 0 ∧
      reducePoly Q n (K 1) - reducePoly Q n S * reducePoly Q n (K 0) =
        reducePoly Q n (S * S) * d0 + ((t : Int) : ExactPoly Q n) * reducePoly Q n X := by
  have hA (r : Fin 2) : reducePoly QP n (canonicalLift (∑ j : Fin 3, rk r j *
      reducePoly QP n (digits j))) = ∑ j : Fin 3, rk r j * reducePoly QP n (digits j) :=
    reducePoly_canonicalLift (by decide) hn _
  obtain ⟨g0, g1, g2⟩ := factor_eq
  -- The CRT-normalized digits recombine to the leading component modulo `Q`.
  have hM : reducePoly Q n (((q2 * q3 : Nat) : Int) * digits 0 + ((q1 * q3 : Nat) : Int) * digits 1 +
      ((q1 * q2 : Nat) : Int) * digits 2) = reducePoly Q n (canonicalLift d0) := by
    rw [reducePoly_eq_iff (by decide) hn]
    intro i
    simp only [Negacyclic.coeff_add, intCast_mul_coeff]
    exact hcrt i
  obtain ⟨Y, hY⟩ := exists_of_reducePoly_eq (q := Q) (by decide) hn hM
  -- Over `Q P`, the key rows leave only the encrypted error and the scaled target.
  have hdiff : reducePoly QP n (canonicalLift (∑ j : Fin 3, rk 1 j * reducePoly QP n (digits j)) -
      S * canonicalLift (∑ j : Fin 3, rk 0 j * reducePoly QP n (digits j))) = reducePoly QP n
      (((t : Int) : ErrorPoly n) * (Erk 0 * digits 0 + Erk 1 * digits 1 + Erk 2 * digits 2) +
        ((P : Int) : ErrorPoly n) * (D * (((q2 * q3 : Nat) : Int) * digits 0 +
          ((q1 * q3 : Nat) : Int) * digits 1 + ((q1 * q2 : Nat) : Int) * digits 2))) := by
    rw [map_sub, map_mul, hA, hA]
    simp only [Fin.sum_univ_three, hrk, map_add, map_mul, reducePoly_intCast, g0, g1, g2]
    push_cast
    ring
  obtain ⟨Z, hZ⟩ := exists_of_reducePoly_eq (q := QP) (by decide) hn hdiff
  have hQP : ((QP : Int) : ErrorPoly n) = ((Q : Int) : ErrorPoly n) * ((P : Int) : ErrorPoly n) := by
    rw [← intCast_errorPoly_mul]; norm_num
  have key : ((t : Int) : ErrorPoly n) *
      (Erk 0 * digits 0 + Erk 1 * digits 1 + Erk 2 * digits 2 + C 1 - S * C 0) =
      ((P : Int) : ErrorPoly n) * (K 1 - S * K 0 - D * canonicalLift d0 -
        ((Q : Int) : ErrorPoly n) * (D * Y) - ((Q : Int) : ErrorPoly n) * Z) := by
    rw [hQP] at hZ
    linear_combination -(hK 1) + S * (hK 0) - hZ - ((P : Int) : ErrorPoly n) * D * hY
  obtain ⟨X, hX⟩ := exists_of_coprime_mul hn (by decide) key
  refine ⟨X, hX.symm, ?_⟩
  have hcancel := intCast_mul_cancel hn (by decide : ((P : Int)) ≠ 0)
    (show ((P : Int) : ErrorPoly n) * (((t : Int) : ErrorPoly n) * X) =
      ((P : Int) : ErrorPoly n) * (K 1 - S * K 0 - D * canonicalLift d0 -
        ((Q : Int) : ErrorPoly n) * (D * Y) - ((Q : Int) : ErrorPoly n) * Z) by
      rw [← key, hX]; ring)
  have hred := congrArg (reducePoly Q n) hcancel
  have hDS : reducePoly Q n D = reducePoly Q n (S * S) :=
    (reducePoly_eq_iff (by decide) hn _ _).mpr hD
  have hQ0 : reducePoly Q n ((Q : Int) : ErrorPoly n) = 0 := by
    have := reducePoly_modulus_mul (n := n) Q 1
    rwa [mul_one] at this
  simp only [map_sub, map_mul] at hred
  rw [hQ0, hDS, reducePoly_canonicalLift (q := Q) (by decide) hn, reducePoly_intCast] at hred
  rw [map_mul] at hred ⊢
  linear_combination -hred


/-- Exact division by a positive integer divides the norm. -/
theorem polyNorm_le_div_of_mul {c : Nat} (hc : 0 < c) {X Y : ErrorPoly n} {bound : Nat}
    (h : ((c : Int) : ErrorPoly n) * X = Y) (hY : polyNorm Y ≤ bound) :
    polyNorm X ≤ bound / c := by
  apply polyNorm_le_of_coeff
  intro i
  rw [Nat.le_div_iff_mul_le hc]
  have hcoeff := congrArg (fun z : ErrorPoly n ↦ z.coeff i) h
  simp only [intCast_mul_coeff] at hcoeff
  have := natAbs_coeff_le_of_polyNorm hY i
  rw [← hcoeff, Int.natAbs_mul] at this
  simpa [mul_comm] using this

theorem relin_error_norm {S X : ErrorPoly n} {Erk digits : Fin 3 → ErrorPoly n}
    {C : Fin 2 → ErrorPoly n} (hS : polyNorm S ≤ 1) (hErk : ∀ j, polyNorm (Erk j) ≤ 20)
    (hδ0 : polyNorm (digits 0) ≤ q1 / 2) (hδ1 : polyNorm (digits 1) ≤ q2 / 2)
    (hδ2 : polyNorm (digits 2) ≤ q3 / 2) (hC : ∀ r, polyNorm (C r) ≤ P / 2)
    (hX : ((P : Int) : ErrorPoly n) * X =
      Erk 0 * digits 0 + Erk 1 * digits 1 + Erk 2 * digits 2 + C 1 - S * C 0) :
    polyNorm X ≤ (n * 20 * (q1 / 2) + n * 20 * (q2 / 2) + n * 20 * (q3 / 2) + P / 2 +
      n * 1 * (P / 2)) / P := by
  apply polyNorm_le_div_of_mul (by decide) hX
  have m (a b : ErrorPoly n) {x y : Nat} (ha : polyNorm a ≤ x) (hb : polyNorm b ≤ y) :
      polyNorm (a * b) ≤ n * x * y :=
    (polyNorm_mul_le hn a b).trans (Nat.mul_le_mul (Nat.mul_le_mul_left _ ha) hb)
  exact (polyNorm_sub_le _ _).trans (Nat.add_le_add ((polyNorm_add_le _ _).trans
    (Nat.add_le_add ((polyNorm_add_le _ _).trans (Nat.add_le_add ((polyNorm_add_le _ _).trans
      (Nat.add_le_add (m _ _ (hErk 0) hδ0) (m _ _ (hErk 1) hδ1))) (m _ _ (hErk 2) hδ2)))
      (hC 1))) (m _ _ hS (hC 0)))

/-- Modulus switching divides the phase by `q3` exactly after its `t`-correction. -/
theorem modswitch_phase {S W : ErrorPoly n} {c : ExactMatrix Q n 2 1}
    {C' K' : Fin 2 → ErrorPoly n}
    (hW : c 1 0 - reducePoly Q n S * c 0 0 = reducePoly Q n W)
    (hK' : ∀ r : Fin 2, ((q3 : Int) : ErrorPoly n) * K' r =
      canonicalLift (c r 0) + ((t : Int) : ErrorPoly n) * C' r) :
    ∃ W' : ErrorPoly n, ((q3 : Int) : ErrorPoly n) * W' =
        W + ((t : Int) : ErrorPoly n) * (C' 1 - S * C' 0) ∧
      reducePoly QL n (K' 1) - reducePoly QL n S * reducePoly QL n (K' 0) = reducePoly QL n W' := by
  have hlift : reducePoly Q n (canonicalLift (c 1 0) - S * canonicalLift (c 0 0)) =
      reducePoly Q n W := by
    rw [map_sub, map_mul, reducePoly_canonicalLift (by decide) hn,
      reducePoly_canonicalLift (by decide) hn, hW]
  obtain ⟨Z, hZ⟩ := exists_of_reducePoly_eq (q := Q) (by decide) hn hlift
  have hQ : ((Q : Int) : ErrorPoly n) = ((q3 : Int) : ErrorPoly n) * ((QL : Int) : ErrorPoly n) := by
    rw [← intCast_errorPoly_mul]; norm_num
  refine ⟨K' 1 - S * K' 0 - ((QL : Int) : ErrorPoly n) * Z, ?_, ?_⟩
  · rw [hQ] at hZ
    linear_combination (hK' 1) - S * (hK' 0) + hZ
  · rw [map_sub, map_mul (reducePoly QL n) ((QL : Int) : ErrorPoly n), reducePoly_intCast,
      show ((QL : Int) : ExactPoly QL n) = 0 by
        have := reducePoly_modulus_mul (n := n) QL 1
        rwa [mul_one, reducePoly_intCast] at this, map_sub, map_mul]
    ring



/-! ## Native evaluations of plaintexts -/

theorem eval_of_fromValues {native : Fin n → Int} {M : ExactMatrix t n 1 1}
    (h : polynomialFromValues true native M) (index : Fin n) :
    (native ⟨nttBitReverse 13 index.val, bitReverse_lt _ index.isLt⟩ : ZMod t) =
      oddPowerSum (nativeNttRoot t n : ZMod t) index.val (M 0 0) := by
  simp only [polynomialFromValues, if_true] at h
  obtain ⟨-, -, bits, hb, he⟩ := polynomialNttRuns_eval h
  rw [bits_eq hb] at he
  obtain ⟨slot, hs, hv⟩ := he index
  have : slot = ⟨nttBitReverse 13 index.val, bitReverse_lt _ index.isLt⟩ := Fin.ext hs
  subst this
  simp only [if_true] at hv
  rw [hv]
  unfold oddPowerSum
  refine Finset.sum_congr rfl fun c _ ↦ ?_
  rw [zmod_val_cast]

theorem values_of_eval {P : ExactMatrix t n 1 1} {v : Fin n → Int}
    (hrange : ∀ i, 0 ≤ v i ∧ v i < t)
    (heval : ∀ index : Fin n, (v ⟨nttBitReverse 13 index.val, bitReverse_lt _ index.isLt⟩ : ZMod t) =
      oddPowerSum (nativeNttRoot t n : ZMod t) index.val (P 0 0)) :
    polynomialValues true P v := by
  show polynomialNttRuns t false _ v
  refine ⟨by decide, fun prime hprime hdvd ↦ ?_, 13, nativeNttRoot t n, by norm_num, native_lt,
    fun prime hprime hdvd ↦ ?_, fun i ↦ ⟨(hrange i).1, (hrange i).2⟩, ?_⟩
  · have hp : prime ∣ t := dvd_trans (dvd_mul_right prime prime) hdvd
    have hpt := (Nat.prime_dvd_prime_iff_eq hprime (Fact.out : t.Prime)).mp hp
    subst hpt
    exact absurd (Nat.le_of_dvd (by decide) hdvd) (by decide)
  · have hpt := (Nat.prime_dvd_prime_iff_eq hprime (Fact.out : t.Prime)).mp hdvd
    subst hpt
    exact ⟨by decide, by rw [Nat.mod_eq_of_lt native_lt]; exact native_primitive,
      Nat.mod_eq_of_lt native_lt⟩
  · simp only [Bool.false_eq_true, if_false]
    intro index
    refine ⟨⟨nttBitReverse 13 index.val, bitReverse_lt _ index.isLt⟩, rfl, ?_⟩
    rw [← ZMod.intCast_eq_intCast_iff', heval index]
    push_cast
    unfold oddPowerSum
    refine Finset.sum_congr rfl fun k _ ↦ ?_
    simp only [ZMod.natCast_val, ZMod.cast_id', id]

/-- A valid message's plaintext evaluates to its native slots. -/
theorem fresh_values (m : Fin n → Int) (hm : ∀ i, 0 ≤ m i ∧ m i ≤ 1032192) :
    polynomialValues true (plainOf m) (nativeOf m) :=
  values_of_eval (fun i ↦ ⟨(hm _).1, lt_of_le_of_lt (hm _).2 (by decide : (1032192 : Int) < (t : Int))⟩)
    (eval_of_fromValues (plainOf_spec m))

/-- The native slots of a product of plaintexts. -/
def prodNative (x y : Fin n → Int) : Fin n → Int := fun j ↦ nativeOf x j * nativeOf y j % 1032193

theorem product_values (x y : Fin n → Int) :
    polynomialValues true (fun _ _ ↦ plainOf x 0 0 * plainOf y 0 0 : ExactMatrix t n 1 1)
      (prodNative x y) := by
  have hroot := nttPrimitiveRoot_pow_n hn native_primitive
  refine values_of_eval (fun i ↦ ⟨Int.emod_nonneg _ (by decide), Int.emod_lt_of_pos _ (by decide)⟩)
    fun index ↦ ?_
  simp only [prodNative]
  rw [show ((1032193 : Int)) = ((t : Nat) : Int) from rfl, ZMod.intCast_mod, Int.cast_mul,
    eval_of_fromValues (plainOf_spec x), eval_of_fromValues (plainOf_spec y),
    oddPowerSum_mul hn hroot]

set_option maxRecDepth 100000 in
theorem tables_consistent (k : Nat) (hk : k < n) :
    packedEntry 13 encodeTable (packedEntry 13 decodeTable k) = k := by
  have := checkBelow_spec (test := fun k ↦
    packedEntry 13 encodeTable (packedEntry 13 decodeTable k) == k)
    (bound := 8192) (by decide +kernel) k hk
  simpa using this

theorem decode_fresh (m : Fin n → Int) : (fun i ↦ nativeOf m ⟨_, decodeEntry_lt i⟩) = m := by
  funext i
  unfold nativeOf
  congr 1
  exact Fin.ext (tables_consistent i.val i.isLt)

theorem decode_product (x y : Fin n → Int) :
    (fun i ↦ prodNative x y ⟨_, decodeEntry_lt i⟩) = fun i ↦ x i * y i % 1032193 := by
  funext i
  simp only [prodNative, nativeOf]
  have h : (⟨packedEntry 13 encodeTable (packedEntry 13 decodeTable i.val),
      encodeEntry_lt ⟨_, decodeEntry_lt i⟩⟩ : Fin n) = i :=
    Fin.ext (tables_consistent i.val i.isLt)
  rw [h]

theorem ht0 : ((t : Int) : ExactPoly t n) = 0 := by
  have := reducePoly_modulus_mul (n := n) t 1
  rwa [mul_one, reducePoly_intCast] at this

theorem reduce_t_lift (V : ErrorPoly n) (m : ExactMatrix t n 1 1) (A : ErrorPoly n)
    (hV : V = ((t : Int) : ErrorPoly n) * A + centeredIntLift (m 0 0)) :
    reducePoly t n V = m 0 0 := by
  rw [hV, map_add, map_mul, reducePoly_intCast, ht0, zero_mul, zero_add,
    reducePoly_centeredIntLift (q := t) (by decide) hn]

theorem modQ_sk {sk : ExactMatrix Q n 1 1} {S : ErrorPoly n} (hsk : sk 0 0 = reducePoly Q n S) :
    modulusReduce (p := Q) sk 0 0 = reducePoly Q n S := by
  simp only [modulusReduce, hsk]
  exact modulusReduce_reducePoly (by decide) (by decide) (dvd_refl _) hn _

theorem modQL_sk {sk : ExactMatrix Q n 1 1} {S : ErrorPoly n} (hsk : sk 0 0 = reducePoly Q n S) :
    modulusReduce (p := QL) sk 0 0 = reducePoly QL n S := by
  simp only [modulusReduce, hsk]
  exact modulusReduce_reducePoly (by decide) (by decide) (by decide) hn _
theorem fresh_plain {pk : ExactMatrix Q n 2 1} {ek : ExactMatrix QP n 2 3} {sk : ExactMatrix Q n 1 1}
    (hk : KeyFacts pk ek sk) {m : Fin n → Int} {ct : ExactMatrix Q n 2 1} (hc : EncFacts pk m ct) :
    centeredRebase (p := t) (phaseMat ct sk) = plainOf m := by
  obtain ⟨S, E, Erk, D, hS, hE, -, -, hsk, hpk, -⟩ := hk
  obtain ⟨U, E1, E2, hU, hE1, hE2, hc0, hc1⟩ := hc
  have ph := fresh_phase (A := pk 0 0) (S := S) (E := E) (E2 := E2) (M := centeredIntLift (plainOf m 0 0)) (c1 := ct 1 0) hc0
    (by rw [hc1, hpk])
  have hphase : phaseMat ct sk 0 0 =
      reducePoly Q n (((t : Int) : ErrorPoly n) * (E * U + E2 - S * E1) + centeredIntLift (plainOf m 0 0)) := by
    simp only [phaseMat, matrixAdd, Matrix.add_apply, matrixMulScalarLeft, matrixNeg,
      Matrix.neg_apply, modQ_sk hsk]
    rw [← ph]
    ring
  have hV := fresh_phase_norm hS hE hU hE1 hE2 (polyNorm_centeredIntLift (by decide) (plainOf m 0 0))
  have hsmall (i : Fin n) : 2 * ((((t : Int) : ErrorPoly n) * (E * U + E2 - S * E1) + centeredIntLift (plainOf m 0 0)).coeff i).natAbs < Q := by
    have h1 := natAbs_coeff_le_of_polyNorm hV i
    have h2 : t * (n * 20 * 1 + 20 + n * 1 * 20) + t / 2 < Q / 2 := by decide
    have h3 := lt_of_le_of_lt h1 h2
    omega
  refine one_one_of_entry ?_
  rw [centeredRebase_eq, hphase, centeredIntLift_reducePoly (q := Q) (by decide) hn _ hsmall]
  exact reduce_t_lift _ _ _ rfl
theorem gate_plain {pk : ExactMatrix Q n 2 1} {ek : ExactMatrix QP n 2 3} {sk : ExactMatrix Q n 1 1}
    (hk : KeyFacts pk ek sk) {x y : Fin n → Int} {c0 c1 : ExactMatrix Q n 2 1}
    (hx : EncFacts pk x c0) (hy : EncFacts pk y c1) {out : ExactMatrix QL n 2 1}
    (hg : GateFacts c0 c1 ek out) :
    matrixScale 998911 (centeredRebase (p := t) (phaseMat out sk)) =
      (fun _ _ ↦ plainOf x 0 0 * plainOf y 0 0 : ExactMatrix t n 1 1) := by
  obtain ⟨S, E, Erk, D, hS, hE, hErk, hD, hsk, hpk, hrk⟩ := hk
  obtain ⟨Ux, E1x, E2x, hUx, hE1x, hE2x, hcx0, hcx1⟩ := hx
  obtain ⟨Uy, E1y, E2y, hUy, hE1y, hE2y, hcy0, hcy1⟩ := hy
  obtain ⟨relin, ⟨digits, C, K, hδ0, hδ1, hδ2, hcrt, hC, hK, hr0, hr1⟩, ⟨C', K', hC', hK', hms⟩⟩ := hg
  set Mx := centeredIntLift (plainOf x 0 0)
  set My := centeredIntLift (plainOf y 0 0)
  set Vx := ((t : Int) : ErrorPoly n) * (E * Ux + E2x - S * E1x) + Mx
  set Vy := ((t : Int) : ErrorPoly n) * (E * Uy + E2y - S * E1y) + My
  have phx := fresh_phase (S := S) (E := E) (E2 := E2x) (M := Mx) (c1 := c0 1 0) hcx0
    (by rw [hcx1, hpk])
  have phy := fresh_phase (S := S) (E := E) (E2 := E2y) (M := My) (c1 := c1 1 0) hcy0
    (by rw [hcy1, hpk])
  obtain ⟨X, hX, hrel⟩ := relin_phase hD hrk hcrt hK
  set W := Vx * Vy + ((t : Int) : ErrorPoly n) * X
  have phrel : relin 1 0 - reducePoly Q n S * relin 0 0 = reducePoly Q n W := by
    rw [hr0, hr1]
    simp only [W, map_add, map_mul, reducePoly_intCast] at hrel ⊢
    linear_combination hrel + (c1 1 0 - reducePoly Q n S * c1 0 0) * phx + reducePoly Q n Vx * phy
  obtain ⟨W', hW', hphaseL⟩ := modswitch_phase phrel hK'
  have hfinal : phaseMat out sk 0 0 = reducePoly QL n W' := by
    simp only [phaseMat, matrixAdd, Matrix.add_apply, matrixMulScalarLeft, matrixNeg,
      Matrix.neg_apply, modQL_sk hsk, hms 0, hms 1]
    rw [← hphaseL]
    ring
  have hVx := fresh_phase_norm hS hE hUx hE1x hE2x (polyNorm_centeredIntLift (by decide) (plainOf x 0 0))
  have hVy := fresh_phase_norm hS hE hUy hE1y hE2y (polyNorm_centeredIntLift (by decide) (plainOf y 0 0))
  have hXn := relin_error_norm hS hErk hδ0 hδ1 hδ2 hC hX
  have hWn : polyNorm W ≤ n * (t * (n * 20 * 1 + 20 + n * 1 * 20) + t / 2) *
      (t * (n * 20 * 1 + 20 + n * 1 * 20) + t / 2) + t * ((n * 20 * (q1 / 2) + n * 20 * (q2 / 2) +
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
  have hbound : polyNorm W' < 56258131176 := hW'n.trans_lt (by decide +kernel)
  have hsmall (i : Fin n) : 2 * (W'.coeff i).natAbs < QL := by
    have := natAbs_coeff_le_of_polyNorm hbound.le i
    unfold QL
    omega
  have hfactor : ((998911 : Int) : ExactPoly t n) = ((q3 : Int) : ExactPoly t n) := by
    rw [intCast_eq_algebraMap, intCast_eq_algebraMap (value := (q3 : Int))]
    congr 1
  have hred_t := congrArg (reducePoly t n) hW'
  rw [map_mul, map_add, map_mul, reducePoly_intCast, reducePoly_intCast, ht0, zero_mul,
    add_zero] at hred_t
  refine one_one_of_entry ?_
  show ((998911 : Int) : ExactPoly t n) * centeredRebase (p := t) (phaseMat out sk) 0 0 = _
  rw [centeredRebase_eq, hfinal, centeredIntLift_reducePoly (q := QL) (by decide) hn _ hsmall, hfactor,
    hred_t]
  change reducePoly t n (Vx * Vy + ((t : Int) : ErrorPoly n) * X) = _
  rw [map_add, map_mul, map_mul (reducePoly t n) ((t : Int) : ErrorPoly n), reducePoly_intCast,
    ht0, zero_mul, add_zero, reduce_t_lift Vx (plainOf x) _ rfl, reduce_t_lift Vy (plainOf y) _ rfl]

theorem determined_holds {α : Type} [Nonempty α] {relation : α → Prop} (h : ∃ a, relation a) :
    relation (determined relation) :=
  Classical.epsilon_spec h
end BgvCorrect
