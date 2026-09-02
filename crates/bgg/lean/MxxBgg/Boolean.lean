import MxxBgg.Bounds
import MxxGadgets.BooleanCircuit

namespace Mxx.Bgg

open Mxx.Primitives

variable {q n secretColumns gadgetColumns preimageBound : Nat}

/- The Boolean message relation is independent of ciphertext metadata. False
   is the zero payload, while true is the public one-message payload. -/
noncomputable def boolMessage (oneMessage : ExactPoly q n) (bit : Bool) : ExactPoly q n :=
  if bit then oneMessage else 0

def IsBooleanMessage (message oneMessage : ExactPoly q n) (bit : Bool) : Prop :=
  message = boolMessage oneMessage bit

theorem boolMessage_false (oneMessage : ExactPoly q n) :
    boolMessage oneMessage false = 0 := by
  simp [boolMessage]

theorem boolMessage_true (oneMessage : ExactPoly q n) :
    boolMessage oneMessage true = oneMessage := by
  simp [boolMessage]

theorem oneMessage_represents_true
    (oneMessage : ExactPoly q n) :
    IsBooleanMessage oneMessage oneMessage true := by
  simp [IsBooleanMessage, boolMessage]

def EncodingCarriesBool
    {ciphertext : ExactMatrix q n 1 gadgetColumns}
    {mask payload : ExactMatrix q n 1 secretColumns}
    {pub gadget : ExactMatrix q n secretColumns gadgetColumns}
    {message : ExactPoly q n}
    (encoding : Encoding ciphertext mask payload pub gadget message)
    (oneMessage : ExactPoly q n) (bit : Bool) : Prop :=
  IsBooleanMessage message oneMessage bit

theorem one_encoding_carries_true
    {ciphertext : ExactMatrix q n 1 gadgetColumns}
    {mask payload : ExactMatrix q n 1 secretColumns}
    {pub gadget : ExactMatrix q n secretColumns gadgetColumns}
    (one : Encoding ciphertext mask payload pub gadget (1 : ExactPoly q n)) :
    EncodingCarriesBool one (1 : ExactPoly q n) true := by
  exact oneMessage_represents_true _

theorem bool_not_message
    (oneMessage : ExactPoly q n) (bit : Bool) :
    oneMessage - boolMessage oneMessage bit = boolMessage oneMessage (!bit) := by
  cases bit <;> simp [boolMessage]

theorem bool_and_message
    (oneMessage : ExactPoly q n) (hIdempotent : oneMessage * oneMessage = oneMessage)
    (left right : Bool) :
    boolMessage oneMessage left * boolMessage oneMessage right =
      boolMessage oneMessage (left && right) := by
  cases left <;> cases right <;> simp [boolMessage, hIdempotent]

theorem bool_xor_message
    (oneMessage : ExactPoly q n)
    (hIdempotent : oneMessage * oneMessage = oneMessage) (left right : Bool) :
    boolMessage oneMessage left + boolMessage oneMessage right -
        (2 : ExactPoly q n) *
          (boolMessage oneMessage left * boolMessage oneMessage right) =
      boolMessage oneMessage (left != right) := by
  cases left <;> cases right <;> simp [boolMessage, hIdempotent] <;> ring

theorem bool_zero_message (oneMessage : ExactPoly q n) :
    boolMessage oneMessage false = 0 := boolMessage_false _

theorem bool_xor_truth_table (oneMessage : ExactPoly q n) (hOne : oneMessage = 1) :
    ∀ left right : Bool,
      boolMessage oneMessage left + boolMessage oneMessage right -
          (2 : ExactPoly q n) *
            (boolMessage oneMessage left * boolMessage oneMessage right) =
        boolMessage oneMessage (left != right) := by
  intro left right
  exact bool_xor_message oneMessage (by simp [hOne]) left right

theorem bool_not_truth_table (oneMessage : ExactPoly q n) :
    ∀ bit : Bool, oneMessage - boolMessage oneMessage bit =
      boolMessage oneMessage (!bit) := by
  intro bit
  exact bool_not_message oneMessage bit

theorem bool_and_truth_table (oneMessage : ExactPoly q n)
    (hIdempotent : oneMessage * oneMessage = oneMessage) :
    ∀ left right : Bool,
      boolMessage oneMessage left * boolMessage oneMessage right =
        boolMessage oneMessage (left && right) := by
  intro left right
  exact bool_and_message oneMessage hIdempotent left right

noncomputable def zero_gate
    {oneCiphertext : ExactMatrix q n 1 gadgetColumns}
    {mask payload : ExactMatrix q n 1 secretColumns}
    {onePublic gadget : ExactMatrix q n secretColumns gadgetColumns}
    {oneMessage : ExactPoly q n}
    (one : Encoding oneCiphertext mask payload onePublic gadget oneMessage) :
    Encoding (oneCiphertext - oneCiphertext) mask payload (onePublic - onePublic) gadget 0 := by
  rcases one with ⟨oneError, oneEquation⟩
  refine ⟨oneError - oneError, ?_⟩
  have h := linear_sub_core oneEquation oneEquation
  rw [h, ← reduceMatrix_sub]
  simp

theorem zero_gate_message
    (oneMessage : ExactPoly q n) :
    IsBooleanMessage (0 : ExactPoly q n) oneMessage false := by
  simp [IsBooleanMessage, boolMessage]

theorem zero_gate_carries_false
    {oneCiphertext : ExactMatrix q n 1 gadgetColumns}
    {mask payload : ExactMatrix q n 1 secretColumns}
    {onePublic gadget : ExactMatrix q n secretColumns gadgetColumns}
    {oneMessage : ExactPoly q n}
    (one : Encoding oneCiphertext mask payload onePublic gadget oneMessage) :
    EncodingCarriesBool (zero_gate one) oneMessage false := by
  exact zero_gate_message oneMessage

/- Public, composable error interfaces for the linear gates. -/
def EncodingErrorWithin
    {ciphertext : ExactMatrix q n 1 gadgetColumns}
    {mask payload : ExactMatrix q n 1 secretColumns}
    {pub gadget : ExactMatrix q n secretColumns gadgetColumns}
    {message : ExactPoly q n}
    (encoding : Encoding ciphertext mask payload pub gadget message) (bound : Nat) : Prop :=
  matrixNorm encoding.error ≤ bound

theorem zero_error_bound
    {error : ErrorMatrix n 1 gadgetColumns} :
    matrixNorm (error - error) ≤ 0 := by simp

theorem add_error_bound
    {left right : ErrorMatrix n 1 gadgetColumns} {leftBound rightBound : Nat}
    (left_le : matrixNorm left ≤ leftBound) (right_le : matrixNorm right ≤ rightBound) :
    matrixNorm (left + right) ≤ leftBound + rightBound := by
  exact (matrixNorm_add_le _ _).trans (Nat.add_le_add left_le right_le)

theorem sub_error_bound
    {left right : ErrorMatrix n 1 gadgetColumns} {leftBound rightBound : Nat}
    (left_le : matrixNorm left ≤ leftBound) (right_le : matrixNorm right ≤ rightBound) :
    matrixNorm (left - right) ≤ leftBound + rightBound := by
  exact (matrixNorm_sub_le _ _).trans (Nat.add_le_add left_le right_le)

theorem not_error_bound
    {oneError inputError : ErrorMatrix n 1 gadgetColumns} {oneBound inputBound : Nat}
    (one_le : matrixNorm oneError ≤ oneBound) (input_le : matrixNorm inputError ≤ inputBound) :
    matrixNorm (oneError - inputError) ≤ oneBound + inputBound :=
  sub_error_bound one_le input_le

theorem small_scalar_two_error_bound
    {error : ErrorMatrix n 1 gadgetColumns} {errorBound : Nat}
    (error_le : matrixNorm error ≤ errorBound) :
    matrixNorm ((2 : ErrorPoly n) • error) ≤ 2 * errorBound := by
  exact (matrixNorm_two_smul_le error).trans (Nat.mul_le_mul_left 2 error_le)

theorem xor_error_bound
    {leftError rightError productError : ErrorMatrix n 1 gadgetColumns}
    {leftBound rightBound productBound : Nat}
    (left_le : matrixNorm leftError ≤ leftBound)
    (right_le : matrixNorm rightError ≤ rightBound)
    (product_le : matrixNorm productError ≤ productBound)
    :
    matrixNorm ((leftError + rightError) - (2 : ErrorPoly n) • productError) ≤
      leftBound + rightBound + 2 * productBound := by
  apply (matrixNorm_sub_le _ _).trans
  apply Nat.add_le_add (add_error_bound left_le right_le)
  exact small_scalar_two_error_bound product_le

/- Componentwise addition and subtraction are the linear BGG+ gates.  Their
   errors are the corresponding integer sums, so no exact-value lift is used. -/
noncomputable def add
    {leftCiphertext rightCiphertext : ExactMatrix q n 1 gadgetColumns}
    {mask payload : ExactMatrix q n 1 secretColumns}
    {leftPublic rightPublic gadget : ExactMatrix q n secretColumns gadgetColumns}
    {leftMessage rightMessage : ExactPoly q n}
    (left : Encoding leftCiphertext mask payload leftPublic gadget leftMessage)
    (right : Encoding rightCiphertext mask payload rightPublic gadget rightMessage) :
    Encoding (leftCiphertext + rightCiphertext) mask payload (leftPublic + rightPublic) gadget
      (leftMessage + rightMessage) := by
  rcases left with ⟨leftError, leftEquation⟩
  rcases right with ⟨rightError, rightEquation⟩
  refine ⟨leftError + rightError, ?_⟩
  have h := linear_add_core leftEquation rightEquation
  rw [h, ← reduceMatrix_add]

noncomputable def sub
    {leftCiphertext rightCiphertext : ExactMatrix q n 1 gadgetColumns}
    {mask payload : ExactMatrix q n 1 secretColumns}
    {leftPublic rightPublic gadget : ExactMatrix q n secretColumns gadgetColumns}
    {leftMessage rightMessage : ExactPoly q n}
    (left : Encoding leftCiphertext mask payload leftPublic gadget leftMessage)
    (right : Encoding rightCiphertext mask payload rightPublic gadget rightMessage) :
    Encoding (leftCiphertext - rightCiphertext) mask payload (leftPublic - rightPublic) gadget
      (leftMessage - rightMessage) := by
  rcases left with ⟨leftError, leftEquation⟩
  rcases right with ⟨rightError, rightEquation⟩
  refine ⟨leftError - rightError, ?_⟩
  have h := linear_sub_core leftEquation rightEquation
  rw [h, ← reduceMatrix_sub]

/- NOT uses the public encoding of one.  Its payload equation is exposed by
   `sub`, and the caller supplies the one-message fact when instantiating it. -/
noncomputable def not_gate
    {oneCiphertext inputCiphertext : ExactMatrix q n 1 gadgetColumns}
    {mask : ExactMatrix q n 1 secretColumns}
    {payload : ExactMatrix q n 1 secretColumns}
    {onePublic inputPublic gadget : ExactMatrix q n secretColumns gadgetColumns}
    {oneMessage inputMessage : ExactPoly q n}
    (one : Encoding oneCiphertext mask payload onePublic gadget oneMessage)
    (input : Encoding inputCiphertext mask payload inputPublic gadget inputMessage) :
    Encoding (oneCiphertext - inputCiphertext) mask payload (onePublic - inputPublic) gadget
      (oneMessage - inputMessage) :=
  sub one input

theorem not_gate_carries
    {oneCiphertext inputCiphertext : ExactMatrix q n 1 gadgetColumns}
    {mask payload : ExactMatrix q n 1 secretColumns}
    {onePublic inputPublic gadget : ExactMatrix q n secretColumns gadgetColumns}
    {oneMessage inputMessage : ExactPoly q n} (one : Encoding oneCiphertext mask payload onePublic gadget oneMessage)
    (input : Encoding inputCiphertext mask payload inputPublic gadget inputMessage)
    (inputBit : Bool)
    (hone : IsBooleanMessage oneMessage oneMessage true)
    (hinput : IsBooleanMessage inputMessage oneMessage inputBit) :
    EncodingCarriesBool (not_gate one input) oneMessage (!inputBit) := by
  unfold EncodingCarriesBool IsBooleanMessage at *
  rw [hone, hinput]
  exact bool_not_message oneMessage inputBit

/- This is the exact `small_scalar_mul` by the public scalar two used by the
   Rust lowering.  The integer witness is reduced through the ring homomorphism. -/
noncomputable def small_scalar_two
    {ciphertext : ExactMatrix q n 1 gadgetColumns}
    {mask payload : ExactMatrix q n 1 secretColumns}
    {pub gadget : ExactMatrix q n secretColumns gadgetColumns}
    {message : ExactPoly q n}
    (input : Encoding ciphertext mask payload pub gadget message) :
    Encoding ((2 : ExactPoly q n) • ciphertext) mask payload
      ((2 : ExactPoly q n) • pub) gadget ((2 : ExactPoly q n) * message) := by
  rcases input with ⟨inputError, inputEquation⟩
  refine ⟨(2 : ErrorPoly n) • inputError, ?_⟩
  have core := scalar_two_core inputEquation
  rw [core]
  rw [reduceMatrix_int_smul]
  have htwo : reducePoly q n (2 : ErrorPoly n) = (2 : ExactPoly q n) := by
    have h : (2 : ErrorPoly n) = (1 : ErrorPoly n) + 1 := by ring
    rw [h, (reducePoly q n).map_add]
    norm_num
  rw [htwo]

/- AND is BGG+ multiplication with the usual shared-secret invariant.  The
   right-public transition is explicit, so the theorem remains valid for an
   unrestricted ideal target and its integer error witness. -/
noncomputable def and_gate
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
    (maskMagnitude : MagnitudeFact mask)
    (preimageLift : BoundedLift decomposition preimageBound)
    (messageLift : ErrorPoly n)
    (message_reduce : reducePoly q n messageLift = leftMessage) :
    Encoding
      (leftCiphertext * decomposition + leftMessage • rightCiphertext)
      mask payload (leftPublic * decomposition) gadget (leftMessage * rightMessage) :=
  multiply left right relation targetApprox maskMagnitude preimageLift messageLift message_reduce
    rfl rfl

theorem and_gate_carries
    {leftCiphertext rightCiphertext : ExactMatrix q n 1 gadgetColumns}
    {mask payload : ExactMatrix q n 1 secretColumns}
    {leftPublic rightPublic gadget : ExactMatrix q n secretColumns gadgetColumns}
    {leftMessage rightMessage oneMessage : ExactPoly q n}
    (left : Encoding leftCiphertext mask mask leftPublic gadget leftMessage)
    (right : Encoding rightCiphertext mask payload rightPublic gadget rightMessage)
    {decomposition : ExactMatrix q n gadgetColumns gadgetColumns}
    {actualTarget : ExactMatrix q n secretColumns gadgetColumns}
    (relation : RightPreimage gadget decomposition actualTarget)
    (targetApprox : Approx actualTarget rightPublic)
    (maskMagnitude : MagnitudeFact mask)
    (preimageLift : BoundedLift decomposition preimageBound)
    (messageLift : ErrorPoly n)
    (message_reduce : reducePoly q n messageLift = leftMessage)
    (leftBit rightBit : Bool)
    (hleft : IsBooleanMessage leftMessage oneMessage leftBit)
    (hright : IsBooleanMessage rightMessage oneMessage rightBit)
    (hone : oneMessage * oneMessage = oneMessage) :
    EncodingCarriesBool
      (and_gate left right relation targetApprox maskMagnitude preimageLift messageLift message_reduce)
      oneMessage (leftBit && rightBit) := by
  unfold EncodingCarriesBool IsBooleanMessage at *
  rw [hleft, hright]
  exact bool_and_message oneMessage hone leftBit rightBit

/- XOR is the standard polynomial identity `x + y - 2xy`.  The two product
   inputs are passed explicitly to keep the exact payload/error equations
   visible at the operation boundary. -/
noncomputable def xor_gate
    {leftCiphertext rightCiphertext productCiphertext : ExactMatrix q n 1 gadgetColumns}
    {mask payload : ExactMatrix q n 1 secretColumns}
    {leftPublic rightPublic productPublic gadget :
      ExactMatrix q n secretColumns gadgetColumns}
    {leftMessage rightMessage : ExactPoly q n}
    (left : Encoding leftCiphertext mask payload leftPublic gadget leftMessage)
    (right : Encoding rightCiphertext mask payload rightPublic gadget rightMessage)
    (product : Encoding productCiphertext mask payload productPublic gadget
      (leftMessage * rightMessage)) :
    Encoding
      ((leftCiphertext + rightCiphertext) - (2 : ExactPoly q n) • productCiphertext)
      mask payload ((leftPublic + rightPublic) - (2 : ExactPoly q n) • productPublic) gadget
      ((leftMessage + rightMessage) -
        (2 : ExactPoly q n) * (leftMessage * rightMessage)) := by
  exact sub (add left right) (small_scalar_two product)

theorem xor_gate_carries
    {leftCiphertext rightCiphertext productCiphertext : ExactMatrix q n 1 gadgetColumns}
    {mask payload : ExactMatrix q n 1 secretColumns}
    {leftPublic rightPublic productPublic gadget : ExactMatrix q n secretColumns gadgetColumns}
    {leftMessage rightMessage oneMessage : ExactPoly q n}
    (left : Encoding leftCiphertext mask payload leftPublic gadget leftMessage)
    (right : Encoding rightCiphertext mask payload rightPublic gadget rightMessage)
    (product : Encoding productCiphertext mask payload productPublic gadget
      (leftMessage * rightMessage))
    (leftBit rightBit : Bool)
    (hleft : IsBooleanMessage leftMessage oneMessage leftBit)
    (hright : IsBooleanMessage rightMessage oneMessage rightBit)
    (hone : oneMessage * oneMessage = oneMessage) :
    EncodingCarriesBool (xor_gate left right product) oneMessage (leftBit != rightBit) := by
  unfold EncodingCarriesBool IsBooleanMessage at *
  rw [hleft, hright]
  exact bool_xor_message oneMessage hone leftBit rightBit

/- The recursive message interpretation mirrors `MxxGadgets.BoolExpr.eval`; it
   is the interface used when a Boolean circuit is evaluated compositionally. -/
end Mxx.Bgg

namespace Mxx.Gadgets

open Mxx.Primitives

noncomputable def BoolExpr.message {inputCount : Nat}
    (oneMessage : ExactPoly q n) (inputs : Fin inputCount → Bool) :
    BoolExpr inputCount → ExactPoly q n
  | .input index => Mxx.Bgg.boolMessage oneMessage (inputs index)
  | .constant value => Mxx.Bgg.boolMessage oneMessage value
  | .not argument => oneMessage - argument.message oneMessage inputs
  | .and left right => left.message oneMessage inputs * right.message oneMessage inputs
  | .xor left right =>
      left.message oneMessage inputs + right.message oneMessage inputs -
        (2 : ExactPoly q n) * (left.message oneMessage inputs * right.message oneMessage inputs)

theorem BoolExpr.message_eq_boolMessage {inputCount : Nat}
    (expression : BoolExpr inputCount) (oneMessage : ExactPoly q n)
    (hOne : oneMessage = 1) (inputs : Fin inputCount → Bool) :
    expression.message oneMessage inputs =
      Mxx.Bgg.boolMessage oneMessage (expression.eval inputs) := by
  induction expression with
  | input index => rfl
  | constant value => rfl
  | not argument ih =>
      simp only [BoolExpr.message, BoolExpr.eval_not]
      rw [ih]
      exact Mxx.Bgg.bool_not_message oneMessage _
  | and left right ihLeft ihRight =>
      simp only [BoolExpr.message, BoolExpr.eval_and]
      rw [ihLeft, ihRight]
      exact Mxx.Bgg.bool_and_message oneMessage (by simp [hOne]) _ _
  | xor left right ihLeft ihRight =>
      simp only [BoolExpr.message, BoolExpr.eval_xor]
      rw [ihLeft, ihRight]
      cases hleft : left.eval inputs <;> cases hright : right.eval inputs <;>
        simpa [hleft, hright] using
          (Mxx.Bgg.bool_xor_message oneMessage (by simp [hOne])
            (left.eval inputs) (right.eval inputs))

theorem BoolCircuit.message_eq_boolMessage {inputCount : Nat}
    (circuit : BoolCircuit inputCount) (oneMessage : ExactPoly q n)
    (hOne : oneMessage = 1) (inputs : Fin inputCount → Bool) :
    circuit.output.message oneMessage inputs =
      Mxx.Bgg.boolMessage oneMessage (circuit.eval inputs) := by
  exact circuit.output.message_eq_boolMessage oneMessage hOne inputs

end Mxx.Gadgets
