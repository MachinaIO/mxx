import DiamondProofParameters
import DiamondEncryptedGateProof
import Stage_encrypt

open Mxx.Primitives MxxRuntime
open DiamondProofParameters

namespace DiamondGeneratedProof

/- These local theorems concern the generated structural example. Numeric acceptance and the
   invariant of the enclosing circuit loop remain separate obligations. -/

theorem generated_public_gate_selection (backend : BackendContext) (hashModel : HashModel)
    (params : Stage_encrypt.Params) (layer lane : Nat) (active : Int) (kind : Fin 6)
    (left right base output : ExactMatrix q n 1 ell)
    (hactive : (lane : Int) < active)
    (hrun : Stage_encrypt.parallel_sequential_generatedRoot_32_14 backend hashModel
      params layer lane ((kind.val : Int), left, right, active, base, ()) output) :
    ∃ digits : ExactMatrix q n ell ell,
      gadgetDecomposeRuns backend params.diamond_gadget_base params.diamond_digit_count
        right digits ∧
      output = [base - base, base, left, base - left, left * digits,
        left + right - matrixMulScalarRight (left * digits)
          (matrixPolynomial [2] : ExactMatrix q n 1 1)].get kind := by
  dsimp only [Stage_encrypt.parallel_sequential_generatedRoot_32_14] at hrun
  rcases hrun with ⟨digits, selected, masked, hdecomp, _, _, _, hselect,
    _, _, _, hmask, hout⟩
  have hflag : decide (Int.ofNat lane ≤ active - 1) = true := by
    apply decide_eq_true
    change (lane : Int) ≤ active - 1
    omega
  rw [hflag, if_pos rfl] at hmask
  rcases hselect with ⟨position, hposition, hselected⟩
  have hp : position = kind := by
    apply Fin.ext
    dsimp at hposition
    omega
  subst position
  rcases hmask with ⟨position, hposition, hmasked⟩
  have hp : position = (⟨1, by decide⟩ : Fin 2) := by
    apply Fin.ext
    dsimp at hposition ⊢
    omega
  subst position
  exact ⟨digits, hdecomp, hout.trans (hmasked.trans hselected)⟩

theorem generated_public_inactive (backend : BackendContext) (hashModel : HashModel)
    (params : Stage_encrypt.Params) (layer lane : Nat) (active kind : Int)
    (left right base output : ExactMatrix q n 1 ell)
    (hinactive : active ≤ (lane : Int))
    (hrun : Stage_encrypt.parallel_sequential_generatedRoot_32_14 backend hashModel
      params layer lane (kind, left, right, active, base, ()) output) : output = 0 := by
  dsimp only [Stage_encrypt.parallel_sequential_generatedRoot_32_14] at hrun
  rcases hrun with ⟨digits, selected, masked, _, _, _, _, _, _, _, _, hmask, hout⟩
  have hflag : decide (Int.ofNat lane ≤ active - 1) = false := by
    apply decide_eq_false
    change ¬ (lane : Int) ≤ active - 1
    omega
  rw [hflag] at hmask
  rcases hmask with ⟨position, hposition, hmasked⟩
  have hp : position = (⟨0, by decide⟩ : Fin 2) := by
    apply Fin.ext
    dsimp at hposition ⊢
    omega
  subst position
  simpa [List.get, matrixSub] using hout.trans hmasked

theorem generated_ciphertext_gate_selection (backend : BackendContext)
    (params : Stage_decrypt.Params) (layer lane : Nat) (kind : Fin 6)
    (zero one left notValue product xorValue output : ExactMatrix q n 1 ell)
    (hrun : Stage_decrypt.parallel_sequential_generatedRoot_67_30 backend params layer lane
      ((kind.val : Int), zero, one, left, notValue, product, xorValue, ()) output) :
    output = [zero, one, left, notValue, product, xorValue].get kind := by
  rcases hrun with ⟨selected, _, _, _, ⟨position, hposition, hselected⟩, hout⟩
  have hp : position = kind := by
    apply Fin.ext
    dsimp at hposition
    omega
  subst position
  exact hout.trans hselected

theorem generated_ciphertext_active_selection (backend : BackendContext)
    (params : Stage_decrypt.Params) (layer lane : Nat) (active flag : Int)
    (zero selected output : ExactMatrix q n 1 ell)
    (hflag : Stage_decrypt.parallel_sequential_generatedRoot_67_3 backend params
      layer lane active flag)
    (hrun : Stage_decrypt.parallel_sequential_generatedRoot_67_31 backend params layer lane
      (flag, zero, selected, ()) output) :
    output = if (lane : Int) < active then selected else zero := by
  rcases hrun with ⟨value, _, _, _, ⟨position, hposition, hselected⟩, hout⟩
  by_cases ha : (lane : Int) < active
  · have hf : flag = 1 := by
      have hb : decide (Int.ofNat lane ≤ active - 1) = true := by
        apply decide_eq_true
        change (lane : Int) ≤ active - 1
        omega
      dsimp only [Stage_decrypt.parallel_sequential_generatedRoot_67_3] at hflag
      rw [hb] at hflag
      exact hflag
    have hp : position = (⟨1, by decide⟩ : Fin 2) := by
      apply Fin.ext
      dsimp at hposition ⊢
      omega
    subst position
    simpa [ha, List.get] using hout.trans hselected
  · have hf : flag = 0 := by
      have hb : decide (Int.ofNat lane ≤ active - 1) = false := by
        apply decide_eq_false
        change ¬ (lane : Int) ≤ active - 1
        omega
      dsimp only [Stage_decrypt.parallel_sequential_generatedRoot_67_3] at hflag
      rw [hb] at hflag
      exact hflag
    have hp : position = (⟨0, by decide⟩ : Fin 2) := by
      apply Fin.ext
      dsimp at hposition ⊢
      omega
    subst position
    simpa [ha, List.get] using hout.trans hselected

theorem generated_encrypted_zero (backend : BackendContext)
    (params : Stage_decrypt.Params) (layer lane : Nat)
    (one output : ExactMatrix q n 1 ell)
    (hrun : Stage_decrypt.parallel_sequential_generatedRoot_67_6 backend params layer lane
      (one, one, ()) output) : output = 0 := by
  change output = one - one at hrun
  simpa using hrun

theorem generated_encrypted_not (backend : BackendContext)
    (params : Stage_decrypt.Params) (layer lane : Nat)
    (onePublic leftPublic oneCipher leftCipher oneError leftError output :
      ExactMatrix q n 1 ell)
    (secret payload : ExactMatrix q n 1 1) (message : ExactPoly q n)
    (hone : oneCipher = secret * onePublic - (payload * gadget) + oneError)
    (hleft : leftCipher = secret * leftPublic - message • (payload * gadget) + leftError)
    (hrun : Stage_decrypt.parallel_sequential_generatedRoot_67_14 backend params layer lane
      (oneCipher, leftCipher, ()) output) :
    output = secret * (onePublic - leftPublic) - (1 - message) • (payload * gadget) +
      (oneError - leftError) := by
  change output = oneCipher - leftCipher at hrun
  rw [hrun, hone, hleft]
  simp only [Matrix.mul_sub, sub_smul, one_smul]
  abel

theorem generated_encrypted_xor (params : Stage_decrypt.Params) (layer lane : Nat)
    (leftPublic rightPublic leftCipher rightCipher leftError rightError
      productTerm messageTerm product sum doubled output : ExactMatrix q n 1 ell)
    (secret messageMatrix : ExactMatrix q n 1 1)
    (rightMessage : ExactPoly q n) (digits : ExactMatrix q n ell ell)
    (hleft : leftCipher = secret * leftPublic -
      messageMatrix 0 0 • (secret * gadget) + leftError)
    (hright : rightCipher = secret * rightPublic -
      rightMessage • (secret * gadget) + rightError)
    (hdecompose : Stage_decrypt.parallel_sequential_generatedRoot_67_19
      DiamondBackend.backend params layer lane rightPublic digits)
    (hproduct : Stage_decrypt.parallel_sequential_generatedRoot_67_20
      DiamondBackend.backend params layer lane (leftCipher, digits, ()) productTerm)
    (hmessage : Stage_decrypt.parallel_sequential_generatedRoot_67_24
      DiamondBackend.backend params layer lane (rightCipher, messageMatrix, ()) messageTerm)
    (hproductSum : Stage_decrypt.parallel_sequential_generatedRoot_67_25
      DiamondBackend.backend params layer lane (productTerm, messageTerm, ()) product)
    (hsum : Stage_decrypt.parallel_sequential_generatedRoot_67_26
      DiamondBackend.backend params layer lane (leftCipher, rightCipher, ()) sum)
    (hdouble : Stage_decrypt.parallel_sequential_generatedRoot_67_28
      DiamondBackend.backend params layer lane
      (product, (matrixPolynomial [2] : ExactMatrix q n 1 1), ()) doubled)
    (hsub : Stage_decrypt.parallel_sequential_generatedRoot_67_29
      DiamondBackend.backend params layer lane (sum, doubled, ()) output) :
    output = secret * (leftPublic + rightPublic - (2 : ExactPoly q n) •
      (leftPublic * digits)) -
      (messageMatrix 0 0 + rightMessage - 2 * (messageMatrix 0 0 * rightMessage)) •
        (secret * gadget) +
      (leftError + rightError - (2 : ExactPoly q n) •
        (leftError * digits + messageMatrix 0 0 • rightError)) ∧ PreimageWithin digits D := by
  obtain ⟨hp, hd⟩ := generated_encrypted_product params layer lane leftPublic rightPublic
    leftCipher rightCipher leftError rightError productTerm messageTerm product
    secret secret secret messageMatrix rightMessage digits hleft hright
    hdecompose hproduct hmessage hproductSum
  have hs : sum = leftCipher + rightCipher := hsum
  have ht : doubled = (2 : ExactPoly q n) • product := by
    rw [hdouble]
    funext row column
    simp [matrixMulScalarRight, matrixPolynomial, Matrix.smul_apply, mul_comm]
  have ho : output = sum - doubled := hsub
  refine ⟨?_, hd⟩
  rw [ho, hs, ht, hp]
  rw [Mxx.Bgg.linear_add_core hleft hright]
  simp only [Matrix.mul_sub, Matrix.mul_add, Matrix.mul_smul, sub_smul, add_smul,
    smul_sub, smul_add, smul_smul]
  abel

theorem boolean_error_neg {n rows cols B : Nat} {e : ErrorMatrix n rows cols}
    (h : CoeffBound e B) : CoeffBound (-e) B := by
  intro row column coefficient
  simpa using h row column coefficient

theorem boolean_error_sub {n rows cols A B : Nat} {e f : ErrorMatrix n rows cols}
    (he : CoeffBound e A) (hf : CoeffBound f B) : CoeffBound (e - f) (A + B) := by
  simpa only [sub_eq_add_neg] using coeffBound_add he (boolean_error_neg hf)

/-- The Boolean multiplier retains the right input error whenever the left message is one. -/
theorem boolean_product_error_bound {n ell B D : Nat} (hn : 0 < n)
    (left right : ErrorMatrix n 1 ell) (digits : ErrorMatrix n ell ell) (bit : Bool)
    (hl : CoeffBound left B) (hr : CoeffBound right B) (hd : CoeffBound digits D) :
    CoeffBound (left * digits + if bit then right else 0) ((ell * n * D + 1) * B) := by
  have hp := coeffBound_mul hn hl hd
  have hb : CoeffBound (if bit then right else 0) B := by
    cases bit
    · intro row column coefficient
      simp
    · exact hr
  convert coeffBound_add hp hb using 1
  ring

/-- The actual XOR expression is sum minus twice product; both product error terms are kept. -/
theorem boolean_xor_error_bound {n ell B D : Nat} (hn : 0 < n)
    (left right : ErrorMatrix n 1 ell) (digits : ErrorMatrix n ell ell) (bit : Bool)
    (hl : CoeffBound left B) (hr : CoeffBound right B) (hd : CoeffBound digits D) :
    CoeffBound (left + right - ((left * digits + if bit then right else 0) +
      (left * digits + if bit then right else 0))) ((2 * (ell * n * D) + 4) * B) := by
  have hp := boolean_product_error_bound hn left right digits bit hl hr hd
  convert boolean_error_sub (coeffBound_add hl hr) (coeffBound_add hp hp) using 1
  ring

/-- Local Boolean BGG invariant. The error is an integer witness, before reduction modulo q. -/
def BooleanEncodingWithin (secret : ExactMatrix q n 1 1)
    (publicKey : ExactMatrix q n 1 ell) (message : ExactPoly q n)
    (ciphertext : ExactMatrix q n 1 ell) (bound : Nat) : Prop :=
  Approx ciphertext (secret * publicKey - message • (secret * gadget)) bound

theorem boolean_encoding_mono {secret publicKey message ciphertext A B}
    (h : BooleanEncodingWithin secret publicKey message ciphertext A) (hle : A ≤ B) :
    BooleanEncodingWithin secret publicKey message ciphertext B := by
  obtain ⟨error, equation, bound⟩ := h
  exact ⟨error, equation, fun row column coefficient ↦ (bound row column coefficient).trans hle⟩

theorem boolean_reduce_sub {q n rows cols : Nat} (left right : ErrorMatrix n rows cols) :
    reduceMatrix q n rows cols (left - right) =
      reduceMatrix q n rows cols left - reduceMatrix q n rows cols right := by
  funext row column
  exact (reducePoly q n).map_sub _ _

theorem boolean_product_error_reduce {q n ell : Nat}
    (left right : ErrorMatrix n 1 ell) (digits : ErrorMatrix n ell ell) (bit : Bool) :
    reduceMatrix q n 1 ell (left * digits + if bit then right else 0) =
      reduceMatrix q n 1 ell left * reduceMatrix q n ell ell digits +
        (if bit then (1 : ExactPoly q n) else 0) • reduceMatrix q n 1 ell right := by
  rw [reduceMatrix_add, reduceMatrix_mul]
  have hz : reduceMatrix q n 1 ell (0 : ErrorMatrix n 1 ell) = 0 := by
    funext row column
    exact (reducePoly q n).map_zero
  cases bit <;> simp [hz]

theorem boolean_xor_error_reduce {q n ell : Nat}
    (left right : ErrorMatrix n 1 ell) (digits : ErrorMatrix n ell ell) (bit : Bool) :
    reduceMatrix q n 1 ell (left + right -
      ((left * digits + if bit then right else 0) +
        (left * digits + if bit then right else 0))) =
      reduceMatrix q n 1 ell left + reduceMatrix q n 1 ell right -
        (2 : ExactPoly q n) • (reduceMatrix q n 1 ell left *
          reduceMatrix q n ell ell digits +
            (if bit then (1 : ExactPoly q n) else 0) • reduceMatrix q n 1 ell right) := by
  rw [boolean_reduce_sub, reduceMatrix_add, reduceMatrix_add,
    boolean_product_error_reduce]
  simp only [two_smul]

theorem generated_encrypted_not_within (backend : BackendContext)
    (params : Stage_decrypt.Params) (layer lane B : Nat)
    (onePublic leftPublic oneCipher leftCipher output : ExactMatrix q n 1 ell)
    (secret : ExactMatrix q n 1 1) (message : ExactPoly q n)
    (hone : BooleanEncodingWithin secret onePublic 1 oneCipher B)
    (hleft : BooleanEncodingWithin secret leftPublic message leftCipher B)
    (hrun : Stage_decrypt.parallel_sequential_generatedRoot_67_14 backend params layer lane
      (oneCipher, leftCipher, ()) output) :
    BooleanEncodingWithin secret (onePublic - leftPublic) (1 - message) output (2 * B) := by
  obtain ⟨oneError, hone, honeBound⟩ := hone
  obtain ⟨leftError, hleft, hleftBound⟩ := hleft
  have hone' : oneCipher = secret * onePublic - secret * gadget +
      reduceMatrix q n 1 ell oneError := by simpa only [one_smul] using hone
  refine ⟨oneError - leftError, ?_, ?_⟩
  · rw [boolean_reduce_sub]
    exact generated_encrypted_not backend params layer lane onePublic leftPublic
      oneCipher leftCipher _ _ output secret secret message hone' hleft hrun
  · simpa only [two_mul] using boolean_error_sub honeBound hleftBound

/-- Both actual nonlinear gate outputs have bounded integer witnesses. The digit lift comes
    from the same generated decomposition relation used by both outputs. -/
theorem generated_encrypted_product_xor_within
    (params : Stage_decrypt.Params) (layer lane B : Nat)
    (leftPublic rightPublic leftCipher rightCipher productTerm messageTerm product
      sum doubled output : ExactMatrix q n 1 ell)
    (secret messageMatrix : ExactMatrix q n 1 1)
    (rightMessage : ExactPoly q n) (bit : Bool) (digits : ExactMatrix q n ell ell)
    (hbit : messageMatrix 0 0 = if bit then 1 else 0)
    (hleft : BooleanEncodingWithin secret leftPublic (messageMatrix 0 0) leftCipher B)
    (hright : BooleanEncodingWithin secret rightPublic rightMessage rightCipher B)
    (hdecompose : Stage_decrypt.parallel_sequential_generatedRoot_67_19
      DiamondBackend.backend params layer lane rightPublic digits)
    (hproduct : Stage_decrypt.parallel_sequential_generatedRoot_67_20
      DiamondBackend.backend params layer lane (leftCipher, digits, ()) productTerm)
    (hmessage : Stage_decrypt.parallel_sequential_generatedRoot_67_24
      DiamondBackend.backend params layer lane (rightCipher, messageMatrix, ()) messageTerm)
    (hproductSum : Stage_decrypt.parallel_sequential_generatedRoot_67_25
      DiamondBackend.backend params layer lane (productTerm, messageTerm, ()) product)
    (hsum : Stage_decrypt.parallel_sequential_generatedRoot_67_26
      DiamondBackend.backend params layer lane (leftCipher, rightCipher, ()) sum)
    (hdouble : Stage_decrypt.parallel_sequential_generatedRoot_67_28
      DiamondBackend.backend params layer lane
      (product, (matrixPolynomial [2] : ExactMatrix q n 1 1), ()) doubled)
    (hsub : Stage_decrypt.parallel_sequential_generatedRoot_67_29
      DiamondBackend.backend params layer lane (sum, doubled, ()) output) :
    BooleanEncodingWithin secret (leftPublic * digits)
      (messageMatrix 0 0 * rightMessage) product ((a + 1) * B) ∧
    BooleanEncodingWithin secret
      (leftPublic + rightPublic - (2 : ExactPoly q n) • (leftPublic * digits))
      (messageMatrix 0 0 + rightMessage - 2 * (messageMatrix 0 0 * rightMessage))
      output (factor * B) := by
  obtain ⟨leftError, hl, hlBound⟩ := hleft
  obtain ⟨rightError, hr, hrBound⟩ := hright
  obtain ⟨hp, hd⟩ := generated_encrypted_product params layer lane leftPublic rightPublic
    leftCipher rightCipher _ _ productTerm messageTerm product secret secret secret
    messageMatrix rightMessage digits hl hr hdecompose hproduct hmessage hproductSum
  obtain ⟨hx, _⟩ := generated_encrypted_xor params layer lane leftPublic rightPublic
    leftCipher rightCipher _ _ productTerm messageTerm product sum doubled output
    secret messageMatrix rightMessage digits hl hr hdecompose hproduct hmessage hproductSum
    hsum hdouble hsub
  obtain ⟨digitLift, hdigits, hdBound⟩ := hd
  constructor
  · refine ⟨leftError * digitLift + if bit then rightError else 0, ?_,
      boolean_product_error_bound (by decide) leftError rightError digitLift bit
        hlBound hrBound hdBound⟩
    rw [boolean_product_error_reduce, ← hdigits, ← hbit]
    exact hp
  · refine ⟨leftError + rightError -
      ((leftError * digitLift + if bit then rightError else 0) +
        (leftError * digitLift + if bit then rightError else 0)), ?_,
      boolean_xor_error_bound (by decide) leftError rightError digitLift bit
        hlBound hrBound hdBound⟩
    rw [boolean_xor_error_reduce, ← hdigits, ← hbit]
    exact hx

theorem boolean_encoding_zero (secret : ExactMatrix q n 1 1) (B : Nat) :
    BooleanEncodingWithin secret 0 0 0 B := by
  refine ⟨0, ?_, ?_⟩
  · simp only [Matrix.mul_zero, zero_smul, sub_zero, zero_add]
    funext row column
    exact (reducePoly q n).map_zero.symm
  · intro row column coefficient
    simp

/-- One proof for each of the six gate constructors, universally quantified over runtime
    operands and the lane. The uniform bound is derived from the actual candidate scopes. -/
theorem generated_boolean_candidates_within
    (params : Stage_decrypt.Params) (layer lane B : Nat)
    (onePublic leftPublic rightPublic oneCipher leftCipher rightCipher zero notValue
      productTerm messageTerm product sum doubled xorValue : ExactMatrix q n 1 ell)
    (secret messageMatrix : ExactMatrix q n 1 1)
    (rightMessage : ExactPoly q n) (bit : Bool) (digits : ExactMatrix q n ell ell)
    (hbit : messageMatrix 0 0 = if bit then 1 else 0)
    (hone : BooleanEncodingWithin secret onePublic 1 oneCipher B)
    (hleft : BooleanEncodingWithin secret leftPublic (messageMatrix 0 0) leftCipher B)
    (hright : BooleanEncodingWithin secret rightPublic rightMessage rightCipher B)
    (hzero : Stage_decrypt.parallel_sequential_generatedRoot_67_6
      DiamondBackend.backend params layer lane (oneCipher, oneCipher, ()) zero)
    (hnot : Stage_decrypt.parallel_sequential_generatedRoot_67_14
      DiamondBackend.backend params layer lane (oneCipher, leftCipher, ()) notValue)
    (hdecompose : Stage_decrypt.parallel_sequential_generatedRoot_67_19
      DiamondBackend.backend params layer lane rightPublic digits)
    (hproduct : Stage_decrypt.parallel_sequential_generatedRoot_67_20
      DiamondBackend.backend params layer lane (leftCipher, digits, ()) productTerm)
    (hmessage : Stage_decrypt.parallel_sequential_generatedRoot_67_24
      DiamondBackend.backend params layer lane (rightCipher, messageMatrix, ()) messageTerm)
    (hproductSum : Stage_decrypt.parallel_sequential_generatedRoot_67_25
      DiamondBackend.backend params layer lane (productTerm, messageTerm, ()) product)
    (hsum : Stage_decrypt.parallel_sequential_generatedRoot_67_26
      DiamondBackend.backend params layer lane (leftCipher, rightCipher, ()) sum)
    (hdouble : Stage_decrypt.parallel_sequential_generatedRoot_67_28
      DiamondBackend.backend params layer lane
      (product, (matrixPolynomial [2] : ExactMatrix q n 1 1), ()) doubled)
    (hsub : Stage_decrypt.parallel_sequential_generatedRoot_67_29
      DiamondBackend.backend params layer lane (sum, doubled, ()) xorValue) :
    ∀ kind : Fin 6, BooleanEncodingWithin secret
      ([0, onePublic, leftPublic, onePublic - leftPublic, leftPublic * digits,
        leftPublic + rightPublic - (2 : ExactPoly q n) • (leftPublic * digits)].get kind)
      ([0, 1, messageMatrix 0 0, 1 - messageMatrix 0 0,
        messageMatrix 0 0 * rightMessage,
        messageMatrix 0 0 + rightMessage - 2 * (messageMatrix 0 0 * rightMessage)].get kind)
      ([zero, oneCipher, leftCipher, notValue, product, xorValue].get kind)
      (factor * B) := by
  have hz := generated_encrypted_zero DiamondBackend.backend params layer lane
    oneCipher zero hzero
  have hn := generated_encrypted_not_within DiamondBackend.backend params layer lane B
    onePublic leftPublic oneCipher leftCipher notValue secret (messageMatrix 0 0) hone hleft hnot
  obtain ⟨hp, hx⟩ := generated_encrypted_product_xor_within params layer lane B
    leftPublic rightPublic leftCipher rightCipher productTerm messageTerm product sum doubled
    xorValue secret messageMatrix rightMessage bit digits hbit hleft hright hdecompose
    hproduct hmessage hproductSum hsum hdouble hsub
  intro kind
  fin_cases kind
  · simpa only [List.get, hz] using boolean_encoding_zero secret (factor * B)
  · exact boolean_encoding_mono hone (by
      simpa only [one_mul] using Nat.mul_le_mul_right B
        (show 1 ≤ factor by unfold factor; omega))
  · exact boolean_encoding_mono hleft (by
      simpa only [one_mul] using Nat.mul_le_mul_right B
        (show 1 ≤ factor by unfold factor; omega))
  · exact boolean_encoding_mono hn (Nat.mul_le_mul_right B
      (show 2 ≤ factor by unfold factor; omega))
  · exact boolean_encoding_mono hp (Nat.mul_le_mul_right B
      (show a + 1 ≤ factor by unfold factor; omega))
  · exact hx

/-- Boolean messages are closed under the same six arithmetic gate expressions. -/
theorem boolean_gate_message_closed {R : Type*} [CommRing R] (left right : R)
    (hl : left = 0 ∨ left = 1) (hr : right = 0 ∨ right = 1) (kind : Fin 6) :
    ([0, 1, left, 1 - left, left * right, left + right - 2 * (left * right)].get kind) = 0 ∨
    ([0, 1, left, 1 - left, left * right, left + right - 2 * (left * right)].get kind) = 1 := by
  fin_cases kind <;> rcases hl with rfl | rfl <;> rcases hr with rfl | rfl <;> norm_num [List.get]

/-- The actual three selectors preserve the invariant at their shared gate kind. Candidate
    invariants are local results such as generated_boolean_candidates_within. -/
theorem generated_selected_encoding_within (backend : BackendContext)
    (params : Stage_decrypt.Params) (layer lane B : Nat) (kind : Fin 6)
    (secret : ExactMatrix q n 1 1)
    (ciphertexts publicKeys : Fin 6 → ExactMatrix q n 1 ell)
    (messages : Fin 6 → ExactMatrix q n 1 1)
    (output publicOutput : ExactMatrix q n 1 ell) (messageOutput : ExactMatrix q n 1 1)
    (hinvariant : ∀ k, BooleanEncodingWithin secret (publicKeys k) (messages k 0 0)
      (ciphertexts k) B)
    (hcipher : Stage_decrypt.parallel_sequential_generatedRoot_67_30 backend params layer lane
      ((kind.val : Int), ciphertexts 0, ciphertexts 1, ciphertexts 2, ciphertexts 3,
        ciphertexts 4, ciphertexts 5, ()) output)
    (hpublic : Stage_decrypt.parallel_sequential_generatedRoot_67_41 backend params layer lane
      ((kind.val : Int), publicKeys 0, publicKeys 1, publicKeys 2, publicKeys 3,
        publicKeys 4, publicKeys 5, ()) publicOutput)
    (hmessage : Stage_decrypt.parallel_sequential_generatedRoot_67_52 backend params layer lane
      ((kind.val : Int), messages 0, messages 1, messages 2, messages 3,
        messages 4, messages 5, ()) messageOutput) :
    BooleanEncodingWithin secret publicOutput (messageOutput 0 0) output B := by
  have hc := generated_ciphertext_gate_selection backend params layer lane kind
    (ciphertexts 0) (ciphertexts 1) (ciphertexts 2) (ciphertexts 3)
    (ciphertexts 4) (ciphertexts 5) output hcipher
  have hp := generated_ciphertext_gate_selection backend params layer lane kind
    (publicKeys 0) (publicKeys 1) (publicKeys 2) (publicKeys 3)
    (publicKeys 4) (publicKeys 5) publicOutput hpublic
  rcases hmessage with ⟨selected, _, _, _, ⟨position, hposition, hselected⟩, hm⟩
  have heq : position = kind := by
    apply Fin.ext
    dsimp at hposition
    omega
  subst position
  have hm' := hm.trans hselected
  have hi := hinvariant kind
  fin_cases kind <;> simpa only [List.get] using hc.symm ▸ hp.symm ▸ hm'.symm ▸ hi

/-- Masking all three actual encoding components preserves the local bound, including the
    inactive branch whose ciphertext, public key, and message are all zero. -/
theorem generated_masked_encoding_within (backend : BackendContext)
    (params : Stage_decrypt.Params) (layer lane B : Nat) (active flag : Int)
    (secret : ExactMatrix q n 1 1)
    (selected publicSelected output publicOutput : ExactMatrix q n 1 ell)
    (messageSelected messageOutput : ExactMatrix q n 1 1)
    (hinvariant : BooleanEncodingWithin secret publicSelected (messageSelected 0 0) selected B)
    (hflag : Stage_decrypt.parallel_sequential_generatedRoot_67_3 backend params
      layer lane active flag)
    (hcipher : Stage_decrypt.parallel_sequential_generatedRoot_67_31 backend params layer lane
      (flag, 0, selected, ()) output)
    (hpublic : Stage_decrypt.parallel_sequential_generatedRoot_67_42 backend params layer lane
      (flag, 0, publicSelected, ()) publicOutput)
    (hmessage : Stage_decrypt.parallel_sequential_generatedRoot_67_53 backend params layer lane
      (flag, 0, messageSelected, ()) messageOutput) :
    BooleanEncodingWithin secret publicOutput (messageOutput 0 0) output B := by
  have hc := generated_ciphertext_active_selection backend params layer lane active flag
    0 selected output hflag hcipher
  have hp := generated_ciphertext_active_selection backend params layer lane active flag
    0 publicSelected publicOutput hflag hpublic
  rcases hmessage with ⟨value, _, _, _, ⟨position, hposition, hselected⟩, hout⟩
  by_cases ha : (lane : Int) < active
  · have hb : decide (Int.ofNat lane ≤ active - 1) = true := by
      apply decide_eq_true
      change (lane : Int) ≤ active - 1
      omega
    dsimp only [Stage_decrypt.parallel_sequential_generatedRoot_67_3] at hflag
    rw [hb] at hflag
    have hf : flag = 1 := hflag
    have heq : position = (⟨1, by decide⟩ : Fin 2) := by
      apply Fin.ext
      dsimp at hposition ⊢
      omega
    subst position
    have hm : messageOutput = messageSelected := hout.trans hselected
    rw [hc, hp, hm]
    simpa only [ha, ↓reduceIte] using hinvariant
  · have hb : decide (Int.ofNat lane ≤ active - 1) = false := by
      apply decide_eq_false
      change ¬ (lane : Int) ≤ active - 1
      omega
    dsimp only [Stage_decrypt.parallel_sequential_generatedRoot_67_3] at hflag
    rw [hb] at hflag
    have hf : flag = 0 := hflag
    have heq : position = (⟨0, by decide⟩ : Fin 2) := by
      apply Fin.ext
      dsimp at hposition ⊢
      omega
    subst position
    have hm : messageOutput = 0 := hout.trans hselected
    have hz := boolean_encoding_zero secret B
    rw [hc, hp, hm]
    simpa only [ha, ↓reduceIte, Matrix.zero_apply] using hz

#print axioms generated_public_gate_selection
#print axioms generated_public_inactive
#print axioms generated_ciphertext_gate_selection
#print axioms generated_ciphertext_active_selection
#print axioms generated_encrypted_zero
#print axioms generated_encrypted_not
#print axioms generated_encrypted_xor
#print axioms boolean_xor_error_bound
#print axioms generated_encrypted_not_within
#print axioms generated_encrypted_product_xor_within
#print axioms generated_boolean_candidates_within
#print axioms boolean_gate_message_closed
#print axioms generated_selected_encoding_within
#print axioms generated_masked_encoding_within

end DiamondGeneratedProof
