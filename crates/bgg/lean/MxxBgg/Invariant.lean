import MxxBgg.Boolean

namespace Mxx.Bgg

open Mxx.Primitives

variable {q n secretColumns gadgetColumns preimageBound : Nat}

/- A Boolean BGG layer uses one shared mask and gadget.  Fixing the payload to
   that mask makes the family type dependent only on its public matrix and
   message, while retaining the exact `Encoding` equation. -/
structure BooleanEncodingValue
    (mask : ExactMatrix q n 1 secretColumns)
    (gadget : ExactMatrix q n secretColumns gadgetColumns) where
  ciphertext : ExactMatrix q n 1 gadgetColumns
  publicMatrix : ExactMatrix q n secretColumns gadgetColumns
  message : ExactPoly q n
  encoding : Encoding ciphertext mask mask publicMatrix gadget message

inductive BooleanGateKind
  | zero | one | copy | not | and | xor
  deriving DecidableEq

structure BooleanGateSpec (count : Nat) where
  kind : BooleanGateKind
  left : Fin count
  right : Fin count

/- The Boolean result selected by a gate record.  This is the small semantic
   counterpart of the runtime gate opcode; it intentionally contains no
   encoding or error state. -/
def BooleanGateSpec.outputBit {count : Nat}
    (spec : BooleanGateSpec count) (previous : Fin count → Bool) : Bool :=
  match spec.kind with
  | .zero => false
  | .one => true
  | .copy => previous spec.left
  | .not => !(previous spec.left)
  | .and => previous spec.left && previous spec.right
  | .xor => previous spec.left != previous spec.right

/- The generic circuit model owns signed-record validation and finite source
   decoding.  BGG only maps that decoded operation to its certificate-carrying
   gate representation; no second opcode parser is maintained here. -/
noncomputable def BooleanGateSpec.ofActive {count : Nat} (countPos : 0 < count) :
    Mxx.Gadgets.LayeredBoolCircuit.ActiveGateSpec count → BooleanGateSpec count
  | .zero => { kind := .zero, left := ⟨0, countPos⟩, right := ⟨0, countPos⟩ }
  | .one => { kind := .one, left := ⟨0, countPos⟩, right := ⟨0, countPos⟩ }
  | .copy left => { kind := .copy, left, right := ⟨0, countPos⟩ }
  | .not left => { kind := .not, left, right := ⟨0, countPos⟩ }
  | .and left right => { kind := .and, left, right }
  | .xor left right => { kind := .xor, left, right }

theorem BooleanGateSpec.ofActive_outputBit {count : Nat} (countPos : 0 < count)
    (spec : Mxx.Gadgets.LayeredBoolCircuit.ActiveGateSpec count)
    (previous : Fin count → Bool) :
    (BooleanGateSpec.ofActive countPos spec).outputBit previous = spec.outputBit previous := by
  cases spec <;> rfl

def finOfInt {count : Nat} (value : Int) (h : 0 ≤ value ∧ value < count) : Fin count :=
  ⟨value.toNat, by
    exact (Int.toNat_lt h.1).mpr h.2⟩

/- The typed interpretation of one live runtime slot.  The slot is indexed by the normalized
   active width, while the source references are indexed by the normalized previous width. -/
noncomputable def activeGateSpec {shape : Mxx.Gadgets.LayeredBoolCircuitShape}
    (circuit : Mxx.Gadgets.LayeredBoolCircuit shape) (valid : circuit.Valid)
    (layer : Fin shape.depth) (slot : Fin (circuit.activeWidth layer)) :
    BooleanGateSpec (circuit.previousNatWidth layer) :=
  BooleanGateSpec.ofActive (Mxx.Gadgets.LayeredBoolCircuit.previousNatWidth_pos valid layer)
    (Mxx.Gadgets.LayeredBoolCircuit.activeGateSpec valid layer slot)

theorem activeGateSpec_outputBit {shape : Mxx.Gadgets.LayeredBoolCircuitShape}
    {circuit : Mxx.Gadgets.LayeredBoolCircuit shape} (valid : circuit.Valid)
    (layer : Fin shape.depth) (slot : Fin (circuit.activeWidth layer))
    (previous : Fin (circuit.previousNatWidth layer) → Bool) :
    (activeGateSpec circuit valid layer slot).outputBit previous =
      (Mxx.Gadgets.LayeredBoolCircuit.activeGateSpec valid layer slot).outputBit previous :=
  BooleanGateSpec.ofActive_outputBit
    (Mxx.Gadgets.LayeredBoolCircuit.previousNatWidth_pos valid layer) _ _

structure BooleanPreimageCertificate
    (mask : ExactMatrix q n 1 secretColumns)
    (gadget : ExactMatrix q n secretColumns gadgetColumns)
    (left right : BooleanEncodingValue mask gadget)
    (preimageBound : Nat) where
  decomposition : ExactMatrix q n gadgetColumns gadgetColumns
  actualTarget : ExactMatrix q n secretColumns gadgetColumns
  relation : RightPreimage gadget decomposition actualTarget
  targetApprox : Approx actualTarget right.publicMatrix
  target_zero : targetApprox.error = 0
  maskMagnitude : MagnitudeFact mask
  preimageLift : BoundedLift decomposition preimageBound
  messageLift : ErrorPoly n
  message_reduce : reducePoly q n messageLift = left.message

structure BooleanProductCertificate
    (mask : ExactMatrix q n 1 secretColumns)
    (gadget : ExactMatrix q n secretColumns gadgetColumns)
    (left right : BooleanEncodingValue mask gadget) where
  product : BooleanEncodingValue mask gadget
  message_eq : product.message = left.message * right.message

noncomputable def xorStep
    {mask : ExactMatrix q n 1 secretColumns}
    {gadget : ExactMatrix q n secretColumns gadgetColumns}
    (left right product : BooleanEncodingValue mask gadget)
    (message_eq : product.message = left.message * right.message) :
    BooleanEncodingValue mask gadget := by
  rcases product with ⟨productCiphertext, productPublic, productMessage, productEncoding⟩
  dsimp at message_eq
  subst productMessage
  exact {
    ciphertext := (left.ciphertext + right.ciphertext) -
      (2 : ExactPoly q n) • productCiphertext
    publicMatrix := (left.publicMatrix + right.publicMatrix) -
      (2 : ExactPoly q n) • productPublic
    message := (left.message + right.message) -
      (2 : ExactPoly q n) * (left.message * right.message)
    encoding := xor_gate left.encoding right.encoding productEncoding }

/- Eliminating the product message equality before constructing the XOR value keeps the
   dependent encoding transport out of downstream carry proofs. -/
theorem xorStep_carries
    {mask : ExactMatrix q n 1 secretColumns}
    {gadget : ExactMatrix q n secretColumns gadgetColumns}
    (left right product : BooleanEncodingValue mask gadget)
    (message_eq : product.message = left.message * right.message)
    (oneMessage : ExactPoly q n) (leftBit rightBit : Bool)
    (leftCarries : EncodingCarriesBool left.encoding oneMessage leftBit)
    (rightCarries : EncodingCarriesBool right.encoding oneMessage rightBit)
    (oneMessageIdempotent : oneMessage * oneMessage = oneMessage) :
    EncodingCarriesBool (xorStep left right product message_eq).encoding oneMessage
      (leftBit != rightBit) := by
  rcases product with ⟨productCiphertext, productPublic, productMessage, productEncoding⟩
  dsimp at message_eq
  subst productMessage
  simpa [xorStep] using
    (xor_gate_carries left.encoding right.encoding productEncoding leftBit rightBit
      leftCarries rightCarries oneMessageIdempotent)

/- A missing certificate makes a gate unavailable rather than silently
   constructing an unrelated value.  XOR carries the actual AND product used
   by the Rust lowering, so its error is not guessed from source bits. -/
noncomputable def gateStep {count : Nat}
    {mask : ExactMatrix q n 1 secretColumns}
    {gadget : ExactMatrix q n secretColumns gadgetColumns}
    (previous : Fin count → BooleanEncodingValue mask gadget)
    (one : BooleanEncodingValue mask gadget)
    (spec : BooleanGateSpec count)
    (preimage : Option (BooleanPreimageCertificate mask gadget (previous spec.left)
      (previous spec.right) preimageBound))
    (product : Option (BooleanProductCertificate mask gadget (previous spec.left)
      (previous spec.right))) :
    Option (BooleanEncodingValue mask gadget) :=
  match spec.kind with
  | .zero => some {
      ciphertext := one.ciphertext - one.ciphertext
      publicMatrix := one.publicMatrix - one.publicMatrix
      message := 0
      encoding := zero_gate one.encoding }
  | .one => some one
  | .copy => some (previous spec.left)
  | .not => some {
      ciphertext := one.ciphertext - (previous spec.left).ciphertext
      publicMatrix := one.publicMatrix - (previous spec.left).publicMatrix
      message := one.message - (previous spec.left).message
      encoding := not_gate one.encoding (previous spec.left).encoding }
  | .and => match preimage with
    | none => none
    | some certificate => some {
        ciphertext := (previous spec.left).ciphertext * certificate.decomposition +
          (previous spec.left).message • (previous spec.right).ciphertext
        publicMatrix := (previous spec.left).publicMatrix * certificate.decomposition
        message := (previous spec.left).message * (previous spec.right).message
        encoding := and_gate (previous spec.left).encoding (previous spec.right).encoding
          certificate.relation certificate.targetApprox certificate.maskMagnitude
          certificate.preimageLift certificate.messageLift certificate.message_reduce }
  | .xor => match product with
    | none => none
    | some productCertificate =>
      some (xorStep (previous spec.left) (previous spec.right)
        productCertificate.product productCertificate.message_eq)

/- A family invariant records only the error bound.  It deliberately does not
   retain a symbolic expansion of a ciphertext or try to prove cancellation
   of its large terms: the `Encoding` equation already fixes that meaning. -/
def FamilyErrorWithin {count : Nat}
    (ciphertexts : Fin count → ExactMatrix q n 1 gadgetColumns)
    (masks payloads : Fin count → ExactMatrix q n 1 secretColumns)
    (publics : Fin count → ExactMatrix q n secretColumns gadgetColumns)
    (messages : Fin count → ExactPoly q n)
    (gadget : ExactMatrix q n secretColumns gadgetColumns)
    (family : ∀ index, Encoding (ciphertexts index) (masks index) (payloads index)
      (publics index) gadget (messages index))
    (bound : Nat) : Prop :=
  ∀ index, EncodingErrorWithin (family index) bound

theorem familyErrorWithin_reindex {count sourceCount : Nat}
    (ciphertexts : Fin sourceCount → ExactMatrix q n 1 gadgetColumns)
    (masks payloads : Fin sourceCount → ExactMatrix q n 1 secretColumns)
    (publics : Fin sourceCount → ExactMatrix q n secretColumns gadgetColumns)
    (messages : Fin sourceCount → ExactPoly q n)
    (gadget : ExactMatrix q n secretColumns gadgetColumns)
    (family : ∀ index, Encoding (ciphertexts index) (masks index) (payloads index)
      (publics index) gadget (messages index))
    (indices : Fin count → Fin sourceCount) (bound : Nat)
    (hfamily : FamilyErrorWithin ciphertexts masks payloads publics messages gadget family bound) :
    FamilyErrorWithin (fun index => ciphertexts (indices index))
      (fun index => masks (indices index)) (fun index => payloads (indices index))
      (fun index => publics (indices index)) (fun index => messages (indices index)) gadget
      (fun index => family (indices index)) bound := by
  intro index
  exact hfamily _

theorem familyErrorWithin_select {count branchCount : Nat}
    (ciphertexts : Fin branchCount → Fin count → ExactMatrix q n 1 gadgetColumns)
    (masks payloads : Fin branchCount → Fin count → ExactMatrix q n 1 secretColumns)
    (publics : Fin branchCount → Fin count → ExactMatrix q n secretColumns gadgetColumns)
    (messages : Fin branchCount → Fin count → ExactPoly q n)
    (gadget : ExactMatrix q n secretColumns gadgetColumns)
    (families : ∀ branch index, Encoding (ciphertexts branch index)
      (masks branch index) (payloads branch index) (publics branch index)
      gadget (messages branch index))
    (selector : Fin count → Fin branchCount) (bound : Nat)
    (hfamilies : ∀ branch index,
      EncodingErrorWithin (families branch index) bound) :
    FamilyErrorWithin (fun index => ciphertexts (selector index) index)
      (fun index => masks (selector index) index)
      (fun index => payloads (selector index) index)
      (fun index => publics (selector index) index)
      (fun index => messages (selector index) index) gadget
      (fun index => families (selector index) index) bound := by
  intro index
  exact hfamilies _ _

theorem zero_gate_error_within
    {oneCiphertext : ExactMatrix q n 1 gadgetColumns}
    {mask payload : ExactMatrix q n 1 secretColumns}
    {onePublic gadget : ExactMatrix q n secretColumns gadgetColumns}
    {oneMessage : ExactPoly q n}
    (one : Encoding oneCiphertext mask payload onePublic gadget oneMessage) :
    EncodingErrorWithin (zero_gate one) 0 := by
  exact zero_error_bound

theorem add_gate_error_within
    {leftCiphertext rightCiphertext : ExactMatrix q n 1 gadgetColumns}
    {mask payload : ExactMatrix q n 1 secretColumns}
    {leftPublic rightPublic gadget : ExactMatrix q n secretColumns gadgetColumns}
    {leftMessage rightMessage : ExactPoly q n}
    (left : Encoding leftCiphertext mask payload leftPublic gadget leftMessage)
    (right : Encoding rightCiphertext mask payload rightPublic gadget rightMessage)
    {leftBound rightBound : Nat}
    (leftWithin : EncodingErrorWithin left leftBound)
    (rightWithin : EncodingErrorWithin right rightBound) :
    EncodingErrorWithin (add left right) (leftBound + rightBound) := by
  rcases left with ⟨leftError, leftEquation⟩
  rcases right with ⟨rightError, rightEquation⟩
  exact add_error_bound leftWithin rightWithin

theorem sub_gate_error_within
    {leftCiphertext rightCiphertext : ExactMatrix q n 1 gadgetColumns}
    {mask payload : ExactMatrix q n 1 secretColumns}
    {leftPublic rightPublic gadget : ExactMatrix q n secretColumns gadgetColumns}
    {leftMessage rightMessage : ExactPoly q n}
    (left : Encoding leftCiphertext mask payload leftPublic gadget leftMessage)
    (right : Encoding rightCiphertext mask payload rightPublic gadget rightMessage)
    {leftBound rightBound : Nat}
    (leftWithin : EncodingErrorWithin left leftBound)
    (rightWithin : EncodingErrorWithin right rightBound) :
    EncodingErrorWithin (sub left right) (leftBound + rightBound) := by
  rcases left with ⟨leftError, leftEquation⟩
  rcases right with ⟨rightError, rightEquation⟩
  exact sub_error_bound leftWithin rightWithin

theorem not_gate_error_within
    {oneCiphertext inputCiphertext : ExactMatrix q n 1 gadgetColumns}
    {mask : ExactMatrix q n 1 secretColumns}
    {payload : ExactMatrix q n 1 secretColumns}
    {onePublic inputPublic gadget : ExactMatrix q n secretColumns gadgetColumns}
    {oneMessage inputMessage : ExactPoly q n}
    (one : Encoding oneCiphertext mask payload onePublic gadget oneMessage)
    (input : Encoding inputCiphertext mask payload inputPublic gadget inputMessage)
    {oneBound inputBound : Nat}
    (oneWithin : EncodingErrorWithin one oneBound)
    (inputWithin : EncodingErrorWithin input inputBound) :
    EncodingErrorWithin (not_gate one input) (oneBound + inputBound) := by
  exact sub_gate_error_within one input oneWithin inputWithin

theorem small_scalar_two_error_within
    {ciphertext : ExactMatrix q n 1 gadgetColumns}
    {mask payload : ExactMatrix q n 1 secretColumns}
    {pub gadget : ExactMatrix q n secretColumns gadgetColumns}
    {message : ExactPoly q n}
    (input : Encoding ciphertext mask payload pub gadget message)
    {bound : Nat} (inputWithin : EncodingErrorWithin input bound) :
    EncodingErrorWithin (small_scalar_two input) (2 * bound) := by
  rcases input with ⟨inputError, inputEquation⟩
  exact small_scalar_two_error_bound inputWithin

theorem xor_gate_error_within
    {leftCiphertext rightCiphertext productCiphertext : ExactMatrix q n 1 gadgetColumns}
    {mask payload : ExactMatrix q n 1 secretColumns}
    {leftPublic rightPublic productPublic gadget :
      ExactMatrix q n secretColumns gadgetColumns}
    {leftMessage rightMessage : ExactPoly q n}
    (left : Encoding leftCiphertext mask payload leftPublic gadget leftMessage)
    (right : Encoding rightCiphertext mask payload rightPublic gadget rightMessage)
    (product : Encoding productCiphertext mask payload productPublic gadget
      (leftMessage * rightMessage))
    {leftBound rightBound productBound : Nat}
    (leftWithin : EncodingErrorWithin left leftBound)
    (rightWithin : EncodingErrorWithin right rightBound)
    (productWithin : EncodingErrorWithin product productBound) :
    EncodingErrorWithin (xor_gate left right product)
      (leftBound + rightBound + 2 * productBound) := by
  rcases left with ⟨leftError, leftEquation⟩
  rcases right with ⟨rightError, rightEquation⟩
  rcases product with ⟨productError, productEquation⟩
  exact xor_error_bound leftWithin rightWithin productWithin

/- When the preimage target is exact, its target error is literally zero.  The
   multiplication bound therefore has only the preimage-consumed error and
   the right ciphertext error; no target-error or extra ring factor remains. -/
theorem multiplication_error_bound_zero_target
    {leftCiphertext rightCiphertext : ExactMatrix q n 1 gadgetColumns}
    {leftMask leftPayload rightMask rightPayload : ExactMatrix q n 1 secretColumns}
    {leftPublic rightPublic gadget : ExactMatrix q n secretColumns gadgetColumns}
    {leftMessage rightMessage : ExactPoly q n}
    (left : Encoding leftCiphertext leftMask leftPayload leftPublic gadget leftMessage)
    (right : Encoding rightCiphertext rightMask rightPayload rightPublic gadget rightMessage)
    {decomposition : ExactMatrix q n gadgetColumns gadgetColumns}
    {actualTarget : ExactMatrix q n secretColumns gadgetColumns}
    (targetApprox : Approx actualTarget rightPublic)
    (target_zero : targetApprox.error = 0)
    (leftMaskMagnitude : MagnitudeFact leftMask)
    (preimageLift : BoundedLift decomposition preimageBound)
    (messageLift : ErrorPoly n)
    (leftErrorBound rightErrorBound messageBound : Nat)
    (left_error_le : matrixNorm left.error ≤ leftErrorBound)
    (right_error_le : matrixNorm right.error ≤ rightErrorBound)
    (message_le : polyNorm messageLift ≤ messageBound)
    (ring_dimension_pos : 0 < n) :
    matrixNorm
        (left.error * preimageLift.witness + messageLift • right.error) ≤
      gadgetColumns * n * leftErrorBound * preimageBound +
        n * messageBound * rightErrorBound := by
  have h := multiplication_error_bound left right targetApprox leftMaskMagnitude
    preimageLift messageLift leftErrorBound rightErrorBound 0 messageBound
    left_error_le right_error_le (by simp [target_zero]) message_le ring_dimension_pos
  simpa [target_zero, add_zero, Nat.add_zero] using h

theorem multiplication_error_bound_constant_message
    {leftCiphertext rightCiphertext : ExactMatrix q n 1 gadgetColumns}
    {leftMask leftPayload rightMask rightPayload : ExactMatrix q n 1 secretColumns}
    {leftPublic rightPublic gadget : ExactMatrix q n secretColumns gadgetColumns}
    {leftMessage rightMessage : ExactPoly q n}
    (left : Encoding leftCiphertext leftMask leftPayload leftPublic gadget leftMessage)
    (right : Encoding rightCiphertext rightMask rightPayload rightPublic gadget rightMessage)
    {decomposition : ExactMatrix q n gadgetColumns gadgetColumns}
    (preimageLift : BoundedLift decomposition preimageBound)
    (messageLift : ErrorPoly n)
    (message_constant : IsConstantPoly messageLift)
    (leftErrorBound rightErrorBound messageBound : Nat)
    (left_error_le : matrixNorm left.error ≤ leftErrorBound)
    (right_error_le : matrixNorm right.error ≤ rightErrorBound)
    (message_le : polyNorm messageLift ≤ messageBound)
    (ring_dimension_pos : 0 < n) :
    matrixNorm
        (left.error * preimageLift.witness + messageLift • right.error) ≤
      gadgetColumns * n * leftErrorBound * preimageBound +
        messageBound * rightErrorBound := by
  have hleft :
      matrixNorm (left.error * preimageLift.witness) ≤
        gadgetColumns * n * leftErrorBound * preimageBound := by
    exact (matrixNorm_mul_le ring_dimension_pos).trans
      (by
        simpa [Nat.mul_assoc] using
          Nat.mul_le_mul_left (gadgetColumns * n)
            (Nat.mul_le_mul left_error_le preimageLift.norm_le))
  have hmessage :
      matrixNorm (messageLift • right.error) ≤ messageBound * rightErrorBound := by
    rw [← scalarLift_mul]
    have hconstant : IsConstantMatrix (scalarLift messageLift) := by
      intro row column
      exact message_constant
    apply (matrixNorm_mul_left_constant_le (n := n) ring_dimension_pos
      hconstant).trans
    calc
      1 * matrixNorm (scalarLift messageLift) * matrixNorm right.error ≤
          1 * messageBound * rightErrorBound := by
        exact Nat.mul_le_mul (Nat.mul_le_mul_left 1 (by simpa [scalarLift_norm] using message_le))
          right_error_le
      _ = messageBound * rightErrorBound := by simp
  exact (matrixNorm_add_le _ _).trans (Nat.add_le_add hleft hmessage)

/- Boolean BGG messages are represented by the two constant integer polynomials
   `0` and `1`.  Keeping this fact explicit lets the constant-side matrix
   lemma remove the ring-dimension factor from the message-error product. -/
noncomputable def booleanMessageLift (bit : Bool) : ErrorPoly n :=
  if bit then 1 else 0

theorem booleanMessageLift_constant (bit : Bool) :
    IsConstantPoly (booleanMessageLift (n := n) bit) := by
  cases bit
  · change IsConstantPoly (0 : ErrorPoly n)
    simp [IsConstantPoly]
  · intro coefficient hcoefficient
    change (1 : ErrorPoly n).coeff coefficient = 0
    unfold Negacyclic.coeff
    split
    · rename_i hnzero
      subst n
      exact Fin.elim0 coefficient
    · rename_i hnzero
      rw [show (1 : ErrorPoly n) =
          AdjoinRoot.mk (negacyclicModulus n Int) (1 : Polynomial Int) by simp]
      rw [AdjoinRoot.modByMonicHom_mk]
      have hmod : (1 : Polynomial Int) %ₘ negacyclicModulus n Int = 1 := by
        apply (Polynomial.modByMonic_eq_self_iff
          (Polynomial.monic_X_pow_add_C 1 hnzero)).mpr
        change (1 : Polynomial Int).degree <
          ((Polynomial.X : Polynomial Int) ^ n + Polynomial.C 1).degree
        have hnpos : 0 < n := Nat.pos_of_ne_zero hnzero
        rw [Polynomial.degree_X_pow_add_C hnpos]
        simpa using hnpos
      rw [hmod]
      rw [Polynomial.coeff_one]
      simp [hcoefficient]

theorem booleanMessageLift_norm_le_one (ring_dimension_pos : 0 < n) (bit : Bool) :
    polyNorm (booleanMessageLift (n := n) bit) ≤ 1 := by
  cases bit
  · simp [booleanMessageLift]
  · have hconstant : IsConstantPoly (1 : ErrorPoly n) := by
      simpa [booleanMessageLift] using booleanMessageLift_constant (n := n) true
    rw [show booleanMessageLift (n := n) true = (1 : ErrorPoly n) by
      simp [booleanMessageLift]]
    rw [polyNorm_constant_eq ring_dimension_pos _ hconstant]
    unfold Negacyclic.coeff
    split
    · rename_i hnzero
      exact False.elim ((Nat.ne_of_gt ring_dimension_pos) hnzero)
    · rename_i hnzero
      rw [show (1 : ErrorPoly n) =
          AdjoinRoot.mk (negacyclicModulus n Int) (1 : Polynomial Int) by simp]
      rw [AdjoinRoot.modByMonicHom_mk]
      have hmod : (1 : Polynomial Int) %ₘ negacyclicModulus n Int = 1 := by
        apply (Polynomial.modByMonic_eq_self_iff
          (Polynomial.monic_X_pow_add_C 1 hnzero)).mpr
        change (1 : Polynomial Int).degree <
          ((Polynomial.X : Polynomial Int) ^ n + Polynomial.C 1).degree
        have hnpos : 0 < n := Nat.pos_of_ne_zero hnzero
        rw [Polynomial.degree_X_pow_add_C hnpos]
        simpa using hnpos
      rw [hmod]
      rw [Polynomial.coeff_one_zero]
      norm_num

theorem booleanMessageLift_reduces (q : Nat) (bit : Bool) :
    reducePoly q n (booleanMessageLift (n := n) bit) =
      boolMessage (1 : ExactPoly q n) bit := by
  cases bit <;> simp [booleanMessageLift, boolMessage]

theorem booleanMessageLift_facts (ring_dimension_pos : 0 < n) (bit : Bool) :
    IsConstantPoly (booleanMessageLift (n := n) bit) ∧
      polyNorm (booleanMessageLift (n := n) bit) ≤ 1 :=
  ⟨booleanMessageLift_constant bit, booleanMessageLift_norm_le_one ring_dimension_pos bit⟩

/- The gate-level theorem keeps the relation and target approximation at the
   same boundary as `multiply`.  Its tighter message term is available only
   with the explicit constant-message invariant. -/
set_option maxHeartbeats 2000000 in
theorem multiply_error_within_zero_target
    {leftCiphertext rightCiphertext : ExactMatrix q n 1 gadgetColumns}
    {leftMask leftPayload rightMask rightPayload : ExactMatrix q n 1 secretColumns}
    {leftPublic rightPublic gadget : ExactMatrix q n secretColumns gadgetColumns}
    {leftMessage rightMessage : ExactPoly q n}
    (left : Encoding leftCiphertext leftMask leftPayload leftPublic gadget leftMessage)
    (right : Encoding rightCiphertext rightMask rightPayload rightPublic gadget rightMessage)
    {decomposition : ExactMatrix q n gadgetColumns gadgetColumns}
    {actualTarget : ExactMatrix q n secretColumns gadgetColumns}
    (relation : RightPreimage gadget decomposition actualTarget)
    (targetApprox : Approx actualTarget rightPublic)
    (target_zero : targetApprox.error = 0)
    (leftMaskMagnitude : MagnitudeFact leftMask)
    (preimageLift : BoundedLift decomposition preimageBound)
    (messageLift : ErrorPoly n)
    (message_reduce : reducePoly q n messageLift = leftMessage)
    (mask_eq : leftMask = rightMask)
    (leftPayload_eq : leftPayload = leftMask)
    (leftErrorBound rightErrorBound messageBound : Nat)
    (left_error_le : matrixNorm left.error ≤ leftErrorBound)
    (right_error_le : matrixNorm right.error ≤ rightErrorBound)
    (message_le : polyNorm messageLift ≤ messageBound)
    (ring_dimension_pos : 0 < n) :
    EncodingErrorWithin
      (multiply left right relation targetApprox leftMaskMagnitude preimageLift messageLift
        message_reduce mask_eq leftPayload_eq)
      (gadgetColumns * n * leftErrorBound * preimageBound +
        n * messageBound * rightErrorBound) := by
  rcases left with ⟨leftError, leftEquation⟩
  rcases right with ⟨rightError, rightEquation⟩
  change matrixNorm
      (leftError * preimageLift.witness + messageLift • rightError -
        messageLift • (leftMaskMagnitude.lift * targetApprox.error)) ≤ _
  rw [target_zero]
  simpa using multiplication_error_bound_zero_target
    ⟨leftError, leftEquation⟩ ⟨rightError, rightEquation⟩ targetApprox target_zero
    leftMaskMagnitude preimageLift messageLift leftErrorBound rightErrorBound messageBound
    left_error_le right_error_le message_le ring_dimension_pos

set_option maxHeartbeats 2000000 in
theorem multiply_error_within_zero_target_constant_message
    {leftCiphertext rightCiphertext : ExactMatrix q n 1 gadgetColumns}
    {leftMask leftPayload rightMask rightPayload : ExactMatrix q n 1 secretColumns}
    {leftPublic rightPublic gadget : ExactMatrix q n secretColumns gadgetColumns}
    {leftMessage rightMessage : ExactPoly q n}
    (left : Encoding leftCiphertext leftMask leftPayload leftPublic gadget leftMessage)
    (right : Encoding rightCiphertext rightMask rightPayload rightPublic gadget rightMessage)
    {decomposition : ExactMatrix q n gadgetColumns gadgetColumns}
    {actualTarget : ExactMatrix q n secretColumns gadgetColumns}
    (relation : RightPreimage gadget decomposition actualTarget)
    (targetApprox : Approx actualTarget rightPublic)
    (target_zero : targetApprox.error = 0)
    (leftMaskMagnitude : MagnitudeFact leftMask)
    (preimageLift : BoundedLift decomposition preimageBound)
    (messageLift : ErrorPoly n)
    (message_reduce : reducePoly q n messageLift = leftMessage)
    (mask_eq : leftMask = rightMask)
    (leftPayload_eq : leftPayload = leftMask)
    (message_constant : IsConstantPoly messageLift)
    (leftErrorBound rightErrorBound messageBound : Nat)
    (left_error_le : matrixNorm left.error ≤ leftErrorBound)
    (right_error_le : matrixNorm right.error ≤ rightErrorBound)
    (message_le : polyNorm messageLift ≤ messageBound)
    (ring_dimension_pos : 0 < n) :
    EncodingErrorWithin
      (multiply left right relation targetApprox leftMaskMagnitude preimageLift messageLift
        message_reduce mask_eq leftPayload_eq)
      (gadgetColumns * n * leftErrorBound * preimageBound +
        messageBound * rightErrorBound) := by
  rcases left with ⟨leftError, leftEquation⟩
  rcases right with ⟨rightError, rightEquation⟩
  change matrixNorm
      (leftError * preimageLift.witness + messageLift • rightError -
        messageLift • (leftMaskMagnitude.lift * targetApprox.error)) ≤ _
  rw [target_zero]
  simpa using multiplication_error_bound_constant_message
    ⟨leftError, leftEquation⟩ ⟨rightError, rightEquation⟩ preimageLift messageLift
    message_constant leftErrorBound rightErrorBound messageBound left_error_le
    right_error_le message_le ring_dimension_pos

set_option maxHeartbeats 2000000 in
theorem and_gate_error_within_zero_target
    {leftCiphertext rightCiphertext : ExactMatrix q n 1 gadgetColumns}
    {mask : ExactMatrix q n 1 secretColumns}
    {payload : ExactMatrix q n 1 secretColumns}
    {leftPublic rightPublic gadget : ExactMatrix q n secretColumns gadgetColumns}
    {leftMessage rightMessage : ExactPoly q n}
    (left : Encoding leftCiphertext mask mask leftPublic gadget leftMessage)
    (right : Encoding rightCiphertext mask payload rightPublic gadget rightMessage)
    {decomposition : ExactMatrix q n gadgetColumns gadgetColumns}
    {actualTarget : ExactMatrix q n secretColumns gadgetColumns}
    (relation : RightPreimage gadget decomposition actualTarget)
    (targetApprox : Approx actualTarget rightPublic)
    (target_zero : targetApprox.error = 0)
    (maskMagnitude : MagnitudeFact mask)
    (preimageLift : BoundedLift decomposition preimageBound)
    (messageLift : ErrorPoly n)
    (message_reduce : reducePoly q n messageLift = leftMessage)
    (message_constant : IsConstantPoly messageLift)
    (leftErrorBound rightErrorBound messageBound : Nat)
    (left_error_le : matrixNorm left.error ≤ leftErrorBound)
    (right_error_le : matrixNorm right.error ≤ rightErrorBound)
    (message_le : polyNorm messageLift ≤ messageBound)
    (ring_dimension_pos : 0 < n) :
    EncodingErrorWithin
      (and_gate left right relation targetApprox maskMagnitude preimageLift messageLift
        message_reduce)
      (gadgetColumns * n * leftErrorBound * preimageBound +
        messageBound * rightErrorBound) := by
  exact multiply_error_within_zero_target_constant_message left right relation targetApprox
    target_zero maskMagnitude preimageLift messageLift message_reduce rfl rfl message_constant
    leftErrorBound rightErrorBound messageBound left_error_le right_error_le message_le
    ring_dimension_pos

/- A one-layer selection and every reindex/gather operation preserve a common
   bound.  The selector is index-local, so this theorem does not inspect or
   normalize any carrier expression. -/
theorem one_layer_error_within
    {count branchCount : Nat}
    (ciphertexts : Fin branchCount → Fin count → ExactMatrix q n 1 gadgetColumns)
    (masks payloads : Fin branchCount → Fin count → ExactMatrix q n 1 secretColumns)
    (publics : Fin branchCount → Fin count → ExactMatrix q n secretColumns gadgetColumns)
    (messages : Fin branchCount → Fin count → ExactPoly q n)
    (gadget : ExactMatrix q n secretColumns gadgetColumns)
    (candidates : ∀ branch index, Encoding (ciphertexts branch index)
      (masks branch index) (payloads branch index) (publics branch index)
      gadget (messages branch index))
    (selector : Fin count → Fin branchCount) (bound : Nat)
    (candidateWithin : ∀ branch index,
      EncodingErrorWithin (candidates branch index) bound) :
    FamilyErrorWithin (fun index => ciphertexts (selector index) index)
      (fun index => masks (selector index) index)
      (fun index => payloads (selector index) index)
      (fun index => publics (selector index) index)
      (fun index => messages (selector index) index) gadget
      (fun index => candidates (selector index) index) bound :=
  familyErrorWithin_select ciphertexts masks payloads publics messages gadget candidates
    selector bound candidateWithin

def gateBound (kind : BooleanGateKind) (n gadgetColumns preimageBound oneBound leftBound
    rightBound productBound messageBound : Nat) : Nat :=
  match kind with
  | .zero => 0
  | .one => oneBound
  | .copy => leftBound
  | .not => oneBound + leftBound
  | .and => gadgetColumns * n * leftBound * preimageBound + messageBound * rightBound
  | .xor => leftBound + rightBound + 2 * productBound

/- If the message lift is one of the Boolean constant lifts, its norm is at most
   one.  For a common incoming bound B, write A = g * n * delta; the AND
   recurrence is then (A + 1) * B. -/
def booleanAndUniformBound (n gadgetColumns preimageBound baseBound : Nat) : Nat :=
  (gadgetColumns * n * preimageBound + 1) * baseBound

theorem and_bound_le_boolean_uniform
    (n gadgetColumns preimageBound baseBound leftBound rightBound : Nat)
    (left_le : leftBound ≤ baseBound) (right_le : rightBound ≤ baseBound) :
    gateBound .and n gadgetColumns preimageBound 0 leftBound rightBound 0 1 ≤
      booleanAndUniformBound n gadgetColumns preimageBound baseBound := by
  unfold gateBound booleanAndUniformBound
  calc
    gadgetColumns * n * leftBound * preimageBound + 1 * rightBound ≤
        gadgetColumns * n * baseBound * preimageBound + baseBound := by
      exact Nat.add_le_add
        (Nat.mul_le_mul_right preimageBound
          (Nat.mul_le_mul_left (gadgetColumns * n) left_le))
        (by simpa using right_le)
    _ = (gadgetColumns * n * preimageBound + 1) * baseBound := by ring

/- In the standard six-operation Boolean lowering, the XOR consumes the
   preceding AND result.  With both direct operands bounded by B, this gives
   (2 * A + 4) * B, where A = g * n * delta. -/
def booleanSixGateUniformBound (n gadgetColumns preimageBound baseBound : Nat) : Nat :=
  (2 * (gadgetColumns * n * preimageBound) + 4) * baseBound

theorem xor_bound_after_boolean_and_le
    (n gadgetColumns preimageBound baseBound leftBound rightBound productBound : Nat)
    (left_le : leftBound ≤ baseBound) (right_le : rightBound ≤ baseBound)
    (product_le :
      productBound ≤ booleanAndUniformBound n gadgetColumns preimageBound baseBound) :
    gateBound .xor n gadgetColumns preimageBound 0 leftBound rightBound productBound 0 ≤
      booleanSixGateUniformBound n gadgetColumns preimageBound baseBound := by
  unfold gateBound booleanSixGateUniformBound
  calc
    leftBound + rightBound + 2 * productBound ≤
        baseBound + baseBound +
          2 * booleanAndUniformBound n gadgetColumns preimageBound baseBound := by
      exact Nat.add_le_add (Nat.add_le_add left_le right_le)
        (Nat.mul_le_mul_left 2 product_le)
    _ = (2 * (gadgetColumns * n * preimageBound) + 4) * baseBound := by
      unfold booleanAndUniformBound
      ring

structure CertifiedGateOutput
    (mask : ExactMatrix q n 1 secretColumns)
    (gadget : ExactMatrix q n secretColumns gadgetColumns)
    (bound : Nat) where
  output : BooleanEncodingValue mask gadget
  within : EncodingErrorWithin output.encoding bound

noncomputable def certifiedZeroStep
    {mask : ExactMatrix q n 1 secretColumns}
    {gadget : ExactMatrix q n secretColumns gadgetColumns}
    (one : BooleanEncodingValue mask gadget) :
    CertifiedGateOutput mask gadget 0 :=
  { output := {
      ciphertext := one.ciphertext - one.ciphertext
      publicMatrix := one.publicMatrix - one.publicMatrix
      message := 0
      encoding := zero_gate one.encoding }
    within := zero_error_bound }

def certifiedOneStep
    {mask : ExactMatrix q n 1 secretColumns}
    {gadget : ExactMatrix q n secretColumns gadgetColumns}
    (one : BooleanEncodingValue mask gadget) (bound : Nat)
    (within : EncodingErrorWithin one.encoding bound) :
    CertifiedGateOutput mask gadget bound :=
  { output := one, within := within }

def certifiedCopyStep {count : Nat}
    {mask : ExactMatrix q n 1 secretColumns}
    {gadget : ExactMatrix q n secretColumns gadgetColumns}
    (previous : Fin count → BooleanEncodingValue mask gadget)
    (index : Fin count) (bound : Nat)
    (within : EncodingErrorWithin (previous index).encoding bound) :
    CertifiedGateOutput mask gadget bound :=
  { output := previous index, within := within }

noncomputable def certifiedNotStep
    {mask : ExactMatrix q n 1 secretColumns}
    {gadget : ExactMatrix q n secretColumns gadgetColumns}
    (one input : BooleanEncodingValue mask gadget)
    {oneBound inputBound : Nat}
    (oneWithin : EncodingErrorWithin one.encoding oneBound)
    (inputWithin : EncodingErrorWithin input.encoding inputBound) :
    CertifiedGateOutput mask gadget (oneBound + inputBound) :=
  { output := {
      ciphertext := one.ciphertext - input.ciphertext
      publicMatrix := one.publicMatrix - input.publicMatrix
      message := one.message - input.message
      encoding := not_gate one.encoding input.encoding }
    within := not_gate_error_within one.encoding input.encoding oneWithin inputWithin }

noncomputable def certifiedAndStep
    {mask : ExactMatrix q n 1 secretColumns}
    {gadget : ExactMatrix q n secretColumns gadgetColumns}
    (left right : BooleanEncodingValue mask gadget)
    (certificate : BooleanPreimageCertificate mask gadget left right preimageBound)
    {leftBound rightBound messageBound : Nat}
    (leftWithin : EncodingErrorWithin left.encoding leftBound)
    (rightWithin : EncodingErrorWithin right.encoding rightBound)
    (message_constant : IsConstantPoly certificate.messageLift)
    (message_le : polyNorm certificate.messageLift ≤ messageBound)
    (ring_dimension_pos : 0 < n) :
    CertifiedGateOutput mask gadget
      (gadgetColumns * n * leftBound * preimageBound + messageBound * rightBound) :=
  { output := {
      ciphertext := left.ciphertext * certificate.decomposition +
        left.message • right.ciphertext
      publicMatrix := left.publicMatrix * certificate.decomposition
      message := left.message * right.message
      encoding := and_gate left.encoding right.encoding certificate.relation
        certificate.targetApprox certificate.maskMagnitude certificate.preimageLift
        certificate.messageLift certificate.message_reduce }
    within := and_gate_error_within_zero_target left.encoding right.encoding
      certificate.relation certificate.targetApprox certificate.target_zero
      certificate.maskMagnitude certificate.preimageLift certificate.messageLift
      certificate.message_reduce message_constant leftBound rightBound messageBound leftWithin
      rightWithin message_le ring_dimension_pos }

set_option maxHeartbeats 1000000 in
noncomputable def certifiedXorStep
    {mask : ExactMatrix q n 1 secretColumns}
    {gadget : ExactMatrix q n secretColumns gadgetColumns}
    (left right : BooleanEncodingValue mask gadget)
    (certificate : BooleanProductCertificate mask gadget left right)
    {leftBound rightBound productBound : Nat}
    (leftWithin : EncodingErrorWithin left.encoding leftBound)
    (rightWithin : EncodingErrorWithin right.encoding rightBound)
    (productWithin : EncodingErrorWithin certificate.product.encoding productBound) :
    CertifiedGateOutput mask gadget (leftBound + rightBound + 2 * productBound) := by
  rcases certificate with ⟨product, message_eq⟩
  rcases product with ⟨productCiphertext, productPublic, productMessage, productEncoding⟩
  dsimp at message_eq
  subst productMessage
  let output : BooleanEncodingValue mask gadget := {
      ciphertext := (left.ciphertext + right.ciphertext) -
        (2 : ExactPoly q n) • productCiphertext
      publicMatrix := (left.publicMatrix + right.publicMatrix) -
        (2 : ExactPoly q n) • productPublic
      message := (left.message + right.message) -
        (2 : ExactPoly q n) * (left.message * right.message)
      encoding := xor_gate left.encoding right.encoding productEncoding }
  refine { output := output, within := ?_ }
  exact xor_gate_error_within left.encoding right.encoding
    productEncoding
    leftWithin rightWithin (by simpa using productWithin)

/- The dispatcher is intentionally certificate-carrying.  Its constructors
   are the only inputs accepted by the layer builder, so projection cannot
   discard a proof or replace a failed gate by a default value. -/
theorem encodingErrorWithin_mono
    {ciphertext : ExactMatrix q n 1 gadgetColumns}
    {mask payload : ExactMatrix q n 1 secretColumns}
    {publicMatrix gadget : ExactMatrix q n secretColumns gadgetColumns}
    {message : ExactPoly q n} {small large : Nat}
    (value : Encoding ciphertext mask payload publicMatrix gadget message)
    (within : EncodingErrorWithin value small) (bound_le : small ≤ large) :
    EncodingErrorWithin value large := within.trans bound_le

inductive CertifiedGateWitness {count : Nat}
    (mask : ExactMatrix q n 1 secretColumns)
    (gadget : ExactMatrix q n secretColumns gadgetColumns)
    (previous : Fin count → BooleanEncodingValue mask gadget)
    (one : BooleanEncodingValue mask gadget)
    (spec : BooleanGateSpec count)
    (preimageBound : Nat)
    (nextBound : Nat) where
  | zero (kind_eq : spec.kind = .zero)
      (within : EncodingErrorWithin (zero_gate one.encoding)
        (gateBound .zero n gadgetColumns preimageBound 0 0 0 0 0))
      (bound_le : gateBound .zero n gadgetColumns preimageBound 0 0 0 0 0 ≤ nextBound) :
      CertifiedGateWitness mask gadget previous one spec preimageBound nextBound
  | one (kind_eq : spec.kind = .one) (oneBound : Nat)
      (within : EncodingErrorWithin one.encoding
        (gateBound .one n gadgetColumns preimageBound oneBound 0 0 0 0))
      (bound_le : gateBound .one n gadgetColumns preimageBound oneBound 0 0 0 0 ≤ nextBound) :
      CertifiedGateWitness mask gadget previous one spec preimageBound nextBound
  | copy (kind_eq : spec.kind = .copy) (leftBound : Nat)
      (within : EncodingErrorWithin (previous spec.left).encoding
        (gateBound .copy n gadgetColumns preimageBound 0 leftBound 0 0 0))
      (bound_le : gateBound .copy n gadgetColumns preimageBound 0 leftBound 0 0 0 ≤ nextBound) :
      CertifiedGateWitness mask gadget previous one spec preimageBound nextBound
  | not (kind_eq : spec.kind = .not) (oneBound inputBound : Nat)
      (within : EncodingErrorWithin
      (not_gate one.encoding (previous spec.left).encoding)
        (gateBound .not n gadgetColumns preimageBound oneBound inputBound 0 0 0))
      (bound_le : gateBound .not n gadgetColumns preimageBound oneBound inputBound 0 0 0 ≤ nextBound) :
      CertifiedGateWitness mask gadget previous one spec preimageBound nextBound
  | and (kind_eq : spec.kind = .and) (leftBound rightBound messageBound : Nat)
      (certificate : BooleanPreimageCertificate mask gadget (previous spec.left)
      (previous spec.right) preimageBound)
      (message_constant : IsConstantPoly certificate.messageLift)
      (within : EncodingErrorWithin
        (and_gate (previous spec.left).encoding (previous spec.right).encoding
          certificate.relation certificate.targetApprox certificate.maskMagnitude
          certificate.preimageLift certificate.messageLift certificate.message_reduce)
        (gateBound .and n gadgetColumns preimageBound 0 leftBound rightBound 0 messageBound))
      (bound_le : gateBound .and n gadgetColumns preimageBound 0 leftBound rightBound 0 messageBound ≤ nextBound) :
      CertifiedGateWitness mask gadget previous one spec preimageBound nextBound
  | xor (kind_eq : spec.kind = .xor) (leftBound rightBound productBound : Nat)
      (certificate : BooleanProductCertificate mask gadget (previous spec.left)
      (previous spec.right))
      (within : EncodingErrorWithin
        (xorStep (previous spec.left) (previous spec.right)
          certificate.product certificate.message_eq).encoding
          (gateBound .xor n gadgetColumns preimageBound 0 leftBound rightBound productBound 0))
      (bound_le : gateBound .xor n gadgetColumns preimageBound 0 leftBound rightBound productBound 0 ≤ nextBound) :
      CertifiedGateWitness mask gadget previous one spec preimageBound nextBound

noncomputable def certifiedGateStep {count : Nat}
    {mask : ExactMatrix q n 1 secretColumns}
    {gadget : ExactMatrix q n secretColumns gadgetColumns}
    {previous : Fin count → BooleanEncodingValue mask gadget}
    {one : BooleanEncodingValue mask gadget}
    {spec : BooleanGateSpec count}
    {preimageBound : Nat}
    {nextBound : Nat} :
    CertifiedGateWitness mask gadget previous one spec preimageBound nextBound →
      CertifiedGateOutput mask gadget nextBound := by
  intro witness
  cases witness with
  | zero _ within bound_le =>
      let output : BooleanEncodingValue mask gadget := {
        ciphertext := one.ciphertext - one.ciphertext
        publicMatrix := one.publicMatrix - one.publicMatrix
        message := 0
        encoding := zero_gate one.encoding }
      exact { output := output, within := encodingErrorWithin_mono _ within bound_le }
  | one _ _ within bound_le =>
      exact { output := one, within := encodingErrorWithin_mono _ within bound_le }
  | copy _ _ within bound_le =>
      exact { output := previous spec.left, within := encodingErrorWithin_mono _ within bound_le }
  | not _ _ _ within bound_le =>
      let output : BooleanEncodingValue mask gadget := {
        ciphertext := one.ciphertext - (previous spec.left).ciphertext
        publicMatrix := one.publicMatrix - (previous spec.left).publicMatrix
        message := one.message - (previous spec.left).message
        encoding := not_gate one.encoding (previous spec.left).encoding }
      exact { output := output, within := encodingErrorWithin_mono _ within bound_le }
  | and _ _ _ _ certificate _message_constant within bound_le =>
      let output : BooleanEncodingValue mask gadget := {
        ciphertext := (previous spec.left).ciphertext * certificate.decomposition +
          (previous spec.left).message • (previous spec.right).ciphertext
        publicMatrix := (previous spec.left).publicMatrix * certificate.decomposition
        message := (previous spec.left).message * (previous spec.right).message
        encoding := and_gate (previous spec.left).encoding (previous spec.right).encoding
          certificate.relation certificate.targetApprox certificate.maskMagnitude
          certificate.preimageLift certificate.messageLift certificate.message_reduce }
      exact { output := output, within := encodingErrorWithin_mono _ within bound_le }
  | xor _ _ _ _ certificate within bound_le =>
      let output := xorStep (previous spec.left) (previous spec.right)
        certificate.product certificate.message_eq
      exact { output := output, within := encodingErrorWithin_mono _ within bound_le }

theorem certifiedGateStep_carries {count : Nat}
    {mask : ExactMatrix q n 1 secretColumns}
    {gadget : ExactMatrix q n secretColumns gadgetColumns}
    {previous : Fin count → BooleanEncodingValue mask gadget}
    {one : BooleanEncodingValue mask gadget}
    {spec : BooleanGateSpec count}
    {preimageBound nextBound : Nat}
    (witness : CertifiedGateWitness mask gadget previous one spec preimageBound nextBound)
    (oneMessage : ExactPoly q n) (previousBits : Fin count → Bool)
    (previousCarries : ∀ index,
      EncodingCarriesBool (previous index).encoding oneMessage (previousBits index))
    (oneCarries : EncodingCarriesBool one.encoding oneMessage true)
    (oneMessageIdempotent : oneMessage * oneMessage = oneMessage) :
    EncodingCarriesBool (certifiedGateStep witness).output.encoding oneMessage
      (spec.outputBit previousBits) := by
  cases witness with
  | zero kind_eq _ _ =>
      simpa [BooleanGateSpec.outputBit, kind_eq] using zero_gate_carries_false one.encoding
  | one kind_eq _ _ _ =>
      simpa [certifiedGateStep, BooleanGateSpec.outputBit, kind_eq] using oneCarries
  | copy kind_eq _ _ _ =>
      simpa [BooleanGateSpec.outputBit, kind_eq] using previousCarries spec.left
  | not kind_eq _ _ _ _ =>
      have hone : one.message = oneMessage := by
        simpa [EncodingCarriesBool, IsBooleanMessage, boolMessage] using oneCarries
      have hinput : IsBooleanMessage (previous spec.left).message oneMessage
          (previousBits spec.left) := by
        exact previousCarries spec.left
      simp [certifiedGateStep, xorStep, kind_eq, BooleanGateSpec.outputBit]
      unfold EncodingCarriesBool IsBooleanMessage at ⊢
      change one.message - (previous spec.left).message =
        boolMessage oneMessage (!previousBits spec.left)
      rw [hone, hinput]
      exact bool_not_message oneMessage (previousBits spec.left)
  | and kind_eq _ _ _ certificate _ _ _ =>
      simpa [BooleanGateSpec.outputBit, kind_eq] using
        (and_gate_carries (previous spec.left).encoding (previous spec.right).encoding
          certificate.relation certificate.targetApprox certificate.maskMagnitude
          certificate.preimageLift certificate.messageLift certificate.message_reduce
          (previousBits spec.left) (previousBits spec.right)
          (previousCarries spec.left) (previousCarries spec.right)
          oneMessageIdempotent)
  | xor kind_eq _ _ _ certificate _ _ =>
      simpa [certifiedGateStep, BooleanGateSpec.outputBit, kind_eq] using
        (xorStep_carries (previous spec.left) (previous spec.right) certificate.product
          certificate.message_eq oneMessage (previousBits spec.left) (previousBits spec.right)
          (previousCarries spec.left) (previousCarries spec.right) oneMessageIdempotent)

noncomputable def certifiedBuildLayer {width count : Nat}
    {mask : ExactMatrix q n 1 secretColumns}
    {gadget : ExactMatrix q n secretColumns gadgetColumns}
    {nextBound : Nat}
    {previous : Fin count → BooleanEncodingValue mask gadget}
    {one : BooleanEncodingValue mask gadget}
    {specs : Fin width → BooleanGateSpec count}
    {preimageBound : Nat}
    (witnesses : ∀ index, CertifiedGateWitness mask gadget previous one
      (specs index) preimageBound nextBound) :
    Fin width → BooleanEncodingValue mask gadget :=
  fun index => (certifiedGateStep (witnesses index)).output

theorem certifiedBuildLayer_error_within {width count : Nat}
    {mask : ExactMatrix q n 1 secretColumns}
    {gadget : ExactMatrix q n secretColumns gadgetColumns}
    {nextBound : Nat}
    {previous : Fin count → BooleanEncodingValue mask gadget}
    {one : BooleanEncodingValue mask gadget}
    {specs : Fin width → BooleanGateSpec count}
    {preimageBound : Nat}
    (witnesses : ∀ index, CertifiedGateWitness mask gadget previous one
      (specs index) preimageBound nextBound) :
    ∀ index, EncodingErrorWithin
      (certifiedBuildLayer witnesses index).encoding nextBound := by
  intro index
  exact (certifiedGateStep (witnesses index)).within

/- Every output in a certified layer carries the Boolean result of its gate
   specification.  The theorem is pointwise in the layer index, which lets a
   caller use it directly with the active slots returned by a layered circuit
   evaluator. -/
theorem certifiedBuildLayer_carries {width count : Nat}
    {mask : ExactMatrix q n 1 secretColumns}
    {gadget : ExactMatrix q n secretColumns gadgetColumns}
    {nextBound : Nat}
    {previous : Fin count → BooleanEncodingValue mask gadget}
    {one : BooleanEncodingValue mask gadget}
    {specs : Fin width → BooleanGateSpec count}
    {preimageBound : Nat}
    (witnesses : ∀ index, CertifiedGateWitness mask gadget previous one
      (specs index) preimageBound nextBound)
    (oneMessage : ExactPoly q n) (previousBits : Fin count → Bool)
    (previousCarries : ∀ index,
      EncodingCarriesBool (previous index).encoding oneMessage (previousBits index))
    (oneCarries : EncodingCarriesBool one.encoding oneMessage true)
    (oneMessageIdempotent : oneMessage * oneMessage = oneMessage) :
    ∀ index, EncodingCarriesBool
      (certifiedBuildLayer witnesses index).encoding oneMessage
      ((specs index).outputBit previousBits) := by
  intro index
  exact certifiedGateStep_carries (witnesses index) oneMessage previousBits
    previousCarries oneCarries oneMessageIdempotent

theorem certifiedBuildLayer_family_error_within {width count : Nat}
    {mask : ExactMatrix q n 1 secretColumns}
    {gadget : ExactMatrix q n secretColumns gadgetColumns}
    {nextBound : Nat}
    {previous : Fin count → BooleanEncodingValue mask gadget}
    {one : BooleanEncodingValue mask gadget}
    {specs : Fin width → BooleanGateSpec count}
    {preimageBound : Nat}
    (witnesses : ∀ index, CertifiedGateWitness mask gadget previous one
      (specs index) preimageBound nextBound) :
    FamilyErrorWithin
      (fun index => (certifiedBuildLayer witnesses index).ciphertext)
      (fun _ => mask) (fun _ => mask)
      (fun index => (certifiedBuildLayer witnesses index).publicMatrix)
      (fun index => (certifiedBuildLayer witnesses index).message) gadget
      (fun index => (certifiedBuildLayer witnesses index).encoding) nextBound := by
  exact certifiedBuildLayer_error_within witnesses

/- A dependent layer state keeps the runtime width in its type.  Thus a gate in layer `i` can
   only reference an element of the immediately preceding active layer; padded records never
   enter the state. -/
structure ExactLayerState
    {q n secretColumns gadgetColumns : Nat}
    (mask : ExactMatrix q n 1 secretColumns)
    (gadget : ExactMatrix q n secretColumns gadgetColumns)
    (oneMessage : ExactPoly q n) (width : Nat) where
  values : Fin width → BooleanEncodingValue mask gadget
  bits : Fin width → Bool
  carries : ∀ index, EncodingCarriesBool (values index).encoding oneMessage (bits index)
  noiseBound : Nat
  within : ∀ index, EncodingErrorWithin (values index).encoding noiseBound

noncomputable def exactAdvance
    {q n secretColumns gadgetColumns count width : Nat}
    {mask : ExactMatrix q n 1 secretColumns}
    {gadget : ExactMatrix q n secretColumns gadgetColumns}
    {oneMessage : ExactPoly q n}
    (previous : ExactLayerState mask gadget oneMessage count)
    (one : BooleanEncodingValue mask gadget)
    (specs : Fin width → BooleanGateSpec count)
    {preimageBound nextBound : Nat}
    (witnesses : ∀ index, CertifiedGateWitness mask gadget previous.values one
      (specs index) preimageBound nextBound)
    (oneCarries : EncodingCarriesBool one.encoding oneMessage true)
    (oneMessageIdempotent : oneMessage * oneMessage = oneMessage) :
    ExactLayerState mask gadget oneMessage width :=
  { values := certifiedBuildLayer witnesses
    bits := fun index => (specs index).outputBit previous.bits
    carries := certifiedBuildLayer_carries witnesses oneMessage previous.bits previous.carries
      oneCarries oneMessageIdempotent
    noiseBound := nextBound
    within := certifiedBuildLayer_error_within witnesses }

theorem exactAdvance_value {q n secretColumns gadgetColumns count width : Nat}
    {mask : ExactMatrix q n 1 secretColumns}
    {gadget : ExactMatrix q n secretColumns gadgetColumns}
    {oneMessage : ExactPoly q n}
    (previous : ExactLayerState mask gadget oneMessage count)
    (one : BooleanEncodingValue mask gadget)
    (specs : Fin width → BooleanGateSpec count)
    {preimageBound nextBound : Nat}
    (witnesses : ∀ index, CertifiedGateWitness mask gadget previous.values one
      (specs index) preimageBound nextBound)
    (oneCarries : EncodingCarriesBool one.encoding oneMessage true)
    (oneMessageIdempotent : oneMessage * oneMessage = oneMessage) (index : Fin width) :
    (exactAdvance previous one specs witnesses oneCarries oneMessageIdempotent).values index =
      (certifiedGateStep (witnesses index)).output := rfl

theorem exactAdvance_bit {q n secretColumns gadgetColumns count width : Nat}
    {mask : ExactMatrix q n 1 secretColumns}
    {gadget : ExactMatrix q n secretColumns gadgetColumns}
    {oneMessage : ExactPoly q n}
    (previous : ExactLayerState mask gadget oneMessage count)
    (one : BooleanEncodingValue mask gadget)
    (specs : Fin width → BooleanGateSpec count)
    {preimageBound nextBound : Nat}
    (witnesses : ∀ index, CertifiedGateWitness mask gadget previous.values one
      (specs index) preimageBound nextBound)
    (oneCarries : EncodingCarriesBool one.encoding oneMessage true)
    (oneMessageIdempotent : oneMessage * oneMessage = oneMessage) (index : Fin width) :
    (exactAdvance previous one specs witnesses oneCarries oneMessageIdempotent).bits index =
    (specs index).outputBit previous.bits := rfl

/- This is the operational connection: the BGG certificate layer and the
   runtime Boolean layer consume the same validated active records.  The gate
   witnesses provide only cryptographic certificates; the Boolean output is
   derived from `LayeredBoolCircuit.evaluateLayer?`. -/
theorem exactAdvance_matches_runtimeLayer
    {shape : Mxx.Gadgets.LayeredBoolCircuitShape}
    {circuit : Mxx.Gadgets.LayeredBoolCircuit shape}
    {mask : ExactMatrix q n 1 secretColumns}
    {gadget : ExactMatrix q n secretColumns gadgetColumns}
    {oneMessage : ExactPoly q n}
    (valid : circuit.Valid) (layer : Fin shape.depth)
    (previous : ExactLayerState mask gadget oneMessage (circuit.previousNatWidth layer))
    (one : BooleanEncodingValue mask gadget)
    {preimageBound nextBound : Nat}
    (witnesses : ∀ slot, CertifiedGateWitness mask gadget previous.values one
      (activeGateSpec circuit valid layer slot) preimageBound nextBound)
    (oneCarries : EncodingCarriesBool one.encoding oneMessage true)
    (oneMessageIdempotent : oneMessage * oneMessage = oneMessage) :
    circuit.evaluateLayer? layer.val (Array.ofFn previous.bits) =
      some (Array.ofFn
        (exactAdvance previous one (fun slot => activeGateSpec circuit valid layer slot)
          witnesses oneCarries oneMessageIdempotent).bits) := by
  rw [Mxx.Gadgets.LayeredBoolCircuit.evaluateLayer?_of_activeSpecs valid layer previous.bits]
  congr 2
  funext slot
  exact (activeGateSpec_outputBit valid layer slot previous.bits).symm

/- The existential width package is the finite dependent fold used by generated applications. -/
abbrev ExactLayerSigma
    {q n secretColumns gadgetColumns : Nat}
    (mask : ExactMatrix q n 1 secretColumns)
    (gadget : ExactMatrix q n secretColumns gadgetColumns)
    (oneMessage : ExactPoly q n) :=
  Sigma (ExactLayerState mask gadget oneMessage)

/- Transporting a layer state changes only its finite index presentation.  It
   never changes a ciphertext, Boolean bit, or established error bound. -/
def castExactLayerState
    {q n secretColumns gadgetColumns width width' : Nat}
    {mask : ExactMatrix q n 1 secretColumns}
    {gadget : ExactMatrix q n secretColumns gadgetColumns}
    {oneMessage : ExactPoly q n}
    (width_eq : width = width')
    (state : ExactLayerState mask gadget oneMessage width) :
    ExactLayerState mask gadget oneMessage width' := by
  subst width_eq
  exact state

/- A certified run is built one live circuit layer at a time.  A step gives
   gate certificates for the current typed state and constructs the next
   state with `exactAdvance`; it does not assume any result from the runtime
   evaluator.  The `position_eq` field pins each constructor to the next
   `List.range` position used by `evaluateUnchecked?`. -/
inductive CertifiedLayeredRun
    {shape : Mxx.Gadgets.LayeredBoolCircuitShape}
    {circuit : Mxx.Gadgets.LayeredBoolCircuit shape}
    {mask : ExactMatrix q n 1 secretColumns}
    {gadget : ExactMatrix q n secretColumns gadgetColumns}
    {oneMessage : ExactPoly q n}
    (valid : circuit.Valid)
    (one : BooleanEncodingValue mask gadget)
    (oneCarries : EncodingCarriesBool one.encoding oneMessage true)
    (oneMessageIdempotent : oneMessage * oneMessage = oneMessage)
    (initial : ExactLayerState mask gadget oneMessage shape.inputWidth) :
    (completed : Nat) → ExactLayerSigma mask gadget oneMessage → Prop
  | initial :
      CertifiedLayeredRun valid one oneCarries oneMessageIdempotent initial 0
        ⟨shape.inputWidth, initial⟩
  | step {completed width}
      {state : ExactLayerState mask gadget oneMessage width}
      (run : CertifiedLayeredRun valid one oneCarries oneMessageIdempotent initial
        completed ⟨width, state⟩)
      (position : Fin shape.depth) (position_eq : position.val = completed)
      (width_eq : width = circuit.previousNatWidth position)
      {preimageBound nextBound : Nat}
      (witnesses : ∀ slot, CertifiedGateWitness mask gadget
        (castExactLayerState width_eq state).values one
        (activeGateSpec circuit valid position slot) preimageBound nextBound) :
      CertifiedLayeredRun valid one oneCarries oneMessageIdempotent initial
        (completed + 1)
        ⟨circuit.activeWidth position,
          exactAdvance (castExactLayerState width_eq state) one
            (fun slot => activeGateSpec circuit valid position slot)
            witnesses oneCarries oneMessageIdempotent⟩

/- Replaying a certified run through the runtime evaluator produces exactly
   the Boolean bits stored in its final BGG state.  The proof is an induction
   over the certificate constructors, so each `evaluateLayer?` success is
   derived from live gate records and their witnesses rather than supplied by
   a caller. -/
theorem CertifiedLayeredRun.runtime
    {shape : Mxx.Gadgets.LayeredBoolCircuitShape}
    {circuit : Mxx.Gadgets.LayeredBoolCircuit shape}
    {mask : ExactMatrix q n 1 secretColumns}
    {gadget : ExactMatrix q n secretColumns gadgetColumns}
    {oneMessage : ExactPoly q n}
    {valid : circuit.Valid}
    {one : BooleanEncodingValue mask gadget}
    {oneCarries : EncodingCarriesBool one.encoding oneMessage true}
    {oneMessageIdempotent : oneMessage * oneMessage = oneMessage}
    {initial : ExactLayerState mask gadget oneMessage shape.inputWidth}
    {completed} {terminal : ExactLayerSigma mask gadget oneMessage}
    (run : CertifiedLayeredRun valid one oneCarries oneMessageIdempotent initial
      completed terminal) :
    match terminal with
    | ⟨_, state⟩ =>
        (List.range completed).foldlM
          (fun previous layer => circuit.evaluateLayer? layer previous)
          (Array.ofFn initial.bits) = some (Array.ofFn state.bits) := by
  induction run with
  | initial => simp
  | @step completed width state run position position_eq width_eq preimageBound nextBound witnesses ih =>
      rw [Mxx.Gadgets.LayeredBoolCircuit.foldlM_range_succ]
      rw [ih]
      change circuit.evaluateLayer? completed (Array.ofFn state.bits) = _
      rw [← position_eq]
      have bits_eq : Array.ofFn (castExactLayerState width_eq state).bits =
          Array.ofFn state.bits := by
        subst width_eq
        rfl
      rw [← bits_eq]
      exact exactAdvance_matches_runtimeLayer valid position
        (castExactLayerState width_eq state) one witnesses oneCarries oneMessageIdempotent

def finalLayer {shape : Mxx.Gadgets.LayeredBoolCircuitShape}
    {circuit : Mxx.Gadgets.LayeredBoolCircuit shape} (valid : circuit.Valid) : Fin shape.depth :=
  ⟨shape.depth - 1,
    Nat.sub_lt (Nat.pos_of_ne_zero (Nat.ne_of_gt (valid.1.2.2.1))) Nat.zero_lt_one⟩

def outputIndex {shape : Mxx.Gadgets.LayeredBoolCircuitShape}
    (circuit : Mxx.Gadgets.LayeredBoolCircuit shape) (valid : circuit.Valid) :
    Fin (circuit.activeWidth ⟨shape.depth - 1,
      Nat.sub_lt (Nat.pos_of_ne_zero (Nat.ne_of_gt (valid.1.2.2.1))) Nat.zero_lt_one⟩) := by
  have depthPos : 0 < shape.depth := valid.1.2.2.1
  have outputNonnegative : 0 ≤ circuit.outputSource := valid.2.2.2.1
  have outputBound : circuit.outputSource <
      circuit.activeGateCounts ⟨shape.depth - 1, by omega⟩ := by
    simpa [Mxx.Gadgets.LayeredBoolCircuit.finalActiveCount, Nat.ne_of_gt depthPos] using
      valid.2.2.2.2
  have lastNonnegative : 0 ≤ circuit.activeGateCounts ⟨shape.depth - 1, by
      exact Nat.sub_lt (Nat.pos_of_ne_zero (Nat.ne_of_gt depthPos)) Nat.zero_lt_one⟩ := by
    exact le_trans (by norm_num) (valid.2.1 _).1
  have lastCountEq : circuit.activeGateCounts ⟨shape.depth - 1, by
      exact Nat.sub_lt (Nat.pos_of_ne_zero (Nat.ne_of_gt depthPos)) Nat.zero_lt_one⟩ =
      (circuit.activeWidth ⟨shape.depth - 1, by
        exact Nat.sub_lt (Nat.pos_of_ne_zero (Nat.ne_of_gt depthPos)) Nat.zero_lt_one⟩ : Int) := by
    exact (Int.toNat_of_nonneg lastNonnegative).symm
  rw [lastCountEq] at outputBound
  exact finOfInt circuit.outputSource ⟨outputNonnegative, by
    exact outputBound⟩

/- Reading the selected output from a runtime array is the same finite slot
   used by the BGG final-state theorem. -/
theorem runtime_outputSource
    {shape : Mxx.Gadgets.LayeredBoolCircuitShape}
    {circuit : Mxx.Gadgets.LayeredBoolCircuit shape} (valid : circuit.Valid)
    (bits : Fin (circuit.activeWidth ⟨shape.depth - 1,
      Nat.sub_lt (Nat.pos_of_ne_zero (Nat.ne_of_gt (valid.1.2.2.1))) Nat.zero_lt_one⟩) → Bool) :
    Mxx.Gadgets.LayeredBoolCircuit.valueAt? (Array.ofFn bits) circuit.outputSource =
      some (bits (outputIndex circuit valid)) := by
  unfold Mxx.Gadgets.LayeredBoolCircuit.valueAt?
  rw [if_pos valid.2.2.2.1]
  have index_eq : (outputIndex circuit valid).val = circuit.outputSource.toNat := by
    simp [outputIndex, finOfInt]
  have index_lt : circuit.outputSource.toNat < (Array.ofFn bits).size := by
    simpa [index_eq] using (outputIndex circuit valid).isLt
  rw [Array.getElem?_eq_getElem index_lt]
  simp only [Option.some.injEq]
  rw [Array.getElem_ofFn index_lt]
  apply congrArg bits
  exact Fin.ext index_eq.symm

/- The complete certificate replay proves the runtime's selected Boolean
   result.  Its only input-side premise identifies the encrypted initial
   family with the instance/witness bits; the final evaluator result is
   reconstructed from the inductive gate certificates. -/
theorem CertifiedLayeredRun.evaluateUnchecked
    {shape : Mxx.Gadgets.LayeredBoolCircuitShape}
    {circuit : Mxx.Gadgets.LayeredBoolCircuit shape}
    {mask : ExactMatrix q n 1 secretColumns}
    {gadget : ExactMatrix q n secretColumns gadgetColumns}
    {oneMessage : ExactPoly q n}
    {valid : circuit.Valid}
    {one : BooleanEncodingValue mask gadget}
    {oneCarries : EncodingCarriesBool one.encoding oneMessage true}
    {oneMessageIdempotent : oneMessage * oneMessage = oneMessage}
    {initial : ExactLayerState mask gadget oneMessage shape.inputWidth}
    (instanceBits : Fin shape.instanceWidth → Bool)
    (witness : Fin shape.witnessWidth → Bool)
    (initial_runtime : Array.ofFn initial.bits =
      (Array.ofFn instanceBits).append (Array.ofFn witness))
    (final : ExactLayerState mask gadget oneMessage
      (circuit.activeWidth ⟨shape.depth - 1,
        Nat.sub_lt (Nat.pos_of_ne_zero (Nat.ne_of_gt (valid.1.2.2.1))) Nat.zero_lt_one⟩))
    (run : CertifiedLayeredRun valid one oneCarries oneMessageIdempotent initial
      shape.depth ⟨_, final⟩) :
    circuit.evaluateUnchecked? instanceBits witness =
      some (final.bits (outputIndex circuit valid)) := by
  rw [Mxx.Gadgets.LayeredBoolCircuit.evaluateUnchecked?_of_finalLayer circuit
    instanceBits witness (Array.ofFn final.bits)]
  · exact runtime_outputSource valid final.bits
  · rw [← initial_runtime]
    exact CertifiedLayeredRun.runtime run

def valueAt
    {q n secretColumns gadgetColumns width : Nat}
    {mask : ExactMatrix q n 1 secretColumns}
    {gadget : ExactMatrix q n secretColumns gadgetColumns}
    {oneMessage : ExactPoly q n}
    (state : ExactLayerState mask gadget oneMessage width) (source : Int)
    (sourceValid : 0 ≤ source ∧ source < width) :
    BooleanEncodingValue mask gadget :=
  state.values (finOfInt source ⟨sourceValid.1, sourceValid.2⟩)

theorem valueAt_bit
    {q n secretColumns gadgetColumns width : Nat}
    {mask : ExactMatrix q n 1 secretColumns}
    {gadget : ExactMatrix q n secretColumns gadgetColumns}
    {oneMessage : ExactPoly q n}
    (state : ExactLayerState mask gadget oneMessage width) (source : Int)
    (sourceValid : 0 ≤ source ∧ source < width) :
    EncodingCarriesBool (valueAt state source sourceValid).encoding oneMessage
      (state.bits (finOfInt source ⟨sourceValid.1, sourceValid.2⟩)) :=
  state.carries _

theorem selectedOutput_carries
    {shape : Mxx.Gadgets.LayeredBoolCircuitShape}
    {circuit : Mxx.Gadgets.LayeredBoolCircuit shape}
    {mask : ExactMatrix q n 1 secretColumns}
    {gadget : ExactMatrix q n secretColumns gadgetColumns}
    {oneMessage : ExactPoly q n}
    (valid : circuit.Valid)
    (state : ExactLayerState mask gadget oneMessage
      (circuit.activeWidth ⟨shape.depth - 1,
        Nat.sub_lt (Nat.pos_of_ne_zero (Nat.ne_of_gt (valid.1.2.2.1))) Nat.zero_lt_one⟩)) :
    EncodingCarriesBool (state.values (outputIndex circuit valid)).encoding oneMessage
      (state.bits (outputIndex circuit valid)) :=
  state.carries _

/- The selected ciphertext inherits the error bound established while its
   final layer was constructed.  No separate final-bound assumption is used. -/
theorem selectedOutput_within
    {shape : Mxx.Gadgets.LayeredBoolCircuitShape}
    {circuit : Mxx.Gadgets.LayeredBoolCircuit shape}
    {mask : ExactMatrix q n 1 secretColumns}
    {gadget : ExactMatrix q n secretColumns gadgetColumns}
    {oneMessage : ExactPoly q n}
    (valid : circuit.Valid)
    (state : ExactLayerState mask gadget oneMessage
      (circuit.activeWidth ⟨shape.depth - 1,
        Nat.sub_lt (Nat.pos_of_ne_zero (Nat.ne_of_gt (valid.1.2.2.1))) Nat.zero_lt_one⟩)) :
    EncodingErrorWithin (state.values (outputIndex circuit valid)).encoding state.noiseBound :=
  state.within _

example {count : Nat} {mask : ExactMatrix q n 1 secretColumns}
    {gadget : ExactMatrix q n secretColumns gadgetColumns}
    {previous : Fin count → BooleanEncodingValue mask gadget}
    {one : BooleanEncodingValue mask gadget}
    {spec : BooleanGateSpec count} {nextBound : Nat}
    {preimageBound : Nat}
    (witness : CertifiedGateWitness mask gadget previous one spec preimageBound nextBound) :
    EncodingErrorWithin (certifiedGateStep witness).output.encoding nextBound := by
  cases witness <;> exact (certifiedGateStep _).within

/- The pointwise proof is intentionally kept in the gate-specific lemmas above
   and the application layer; a single dependent `match gateStep ...` theorem
   causes Lean elaboration to unfold all matrix dimensions exponentially. -/
end Mxx.Bgg
