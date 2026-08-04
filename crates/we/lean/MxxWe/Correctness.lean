import Mxx.Ir
import Mxx.Toolkit.Negacyclic
import MxxWe.BoundRecurrence
import Mathlib.Tactic

namespace MxxWe

/-- The six public opcodes accepted by the shape-parameterized Boolean circuit format. -/
inductive BooleanGate where
  | constantFalse
  | constantTrue
  | copy
  | not
  | and
  | xor
  deriving DecidableEq, Repr

/-- The error transfer implemented by the BGG encoding handlers.  This definition is deliberately
worst-case: every polynomial product includes the full ring-dimension factor. -/
def gateNoise (gate : BooleanGate) (ringDimension publicColumns digitBound oneError left right : Nat) :
    Nat :=
  let product := productBound ringDimension publicColumns left digitBound +
    productBound ringDimension 1 right 1
  match gate with
  | .constantFalse => 2 * oneError
  | .constantTrue => oneError
  | .copy => left
  | .not => oneError + left
  | .and => product
  | .xor => left + right + 2 * product

theorem productBound_mono {ringDimension innerDimension left right leftBound rightBound : Nat}
    (left_le : left ≤ leftBound) (right_le : right ≤ rightBound) :
    productBound ringDimension innerDimension left right ≤
      productBound ringDimension innerDimension leftBound rightBound := by
  simp only [productBound]
  gcongr

/-- `gateStep` covers every opcode and every pair of dynamically selected predecessors whose
individual error is covered by `input`. -/
theorem gateNoise_le_gateStep (gate : BooleanGate)
    (ringDimension publicColumns digitBound oneError left right input : Nat)
    (left_le : left ≤ input) (right_le : right ≤ input) :
    gateNoise gate ringDimension publicColumns digitBound oneError left right ≤
      gateStep ringDimension publicColumns digitBound oneError input := by
  let product := productBound ringDimension publicColumns input digitBound +
    productBound ringDimension 1 input 1
  have leftProduct :
      productBound ringDimension publicColumns left digitBound ≤
        productBound ringDimension publicColumns input digitBound :=
    productBound_mono left_le (Nat.le_refl _)
  have rightProduct :
      productBound ringDimension 1 right 1 ≤ productBound ringDimension 1 input 1 :=
    productBound_mono right_le (Nat.le_refl _)
  have product_le :
      productBound ringDimension publicColumns left digitBound +
          productBound ringDimension 1 right 1 ≤ product :=
    Nat.add_le_add leftProduct rightProduct
  cases gate <;>
    simp only [gateNoise, gateStep] <;>
    dsimp only [product] at product_le ⊢
  · exact le_max_left _ _
  · exact le_trans (le_max_left _ _) (le_max_right _ _)
  · exact le_trans left_le <| le_trans (le_max_left _ _) <|
      le_trans (le_max_right _ _) (le_max_right _ _)
  · exact le_trans (Nat.add_le_add_left left_le _) <| le_trans (le_max_left _ _) <|
      le_trans (le_max_right _ _) <| le_trans (le_max_right _ _) (le_max_right _ _)
  · exact le_trans product_le <| le_trans (le_max_left _ _) <|
      le_trans (le_max_right _ _) <| le_trans (le_max_right _ _) <|
        le_trans (le_max_right _ _) (le_max_right _ _)
  · have xor_le : left + right +
        2 * (productBound ringDimension publicColumns left digitBound +
          productBound ringDimension 1 right 1) ≤ 2 * input + 2 * product := by
      omega
    exact le_trans xor_le <| le_trans (le_max_right _ _) <|
      le_trans (le_max_right _ _) <| le_trans (le_max_right _ _) <|
        le_trans (le_max_right _ _) (le_max_right _ _)

/-- Concrete error left after one input-injection transition.  This is the exact error term from
`(signal * base + initialError) * transition` after applying the preimage relation. -/
def propagatedStateNoise (signal transitionError initialError transition : Mxx.Matrix) :
    Mxx.Matrix :=
  Mxx.matrixAdd (Mxx.matrixMul signal transitionError)
    (Mxx.matrixMul initialError transition)

theorem propagatedStateNoise_norm_le (q ringDimension stateRows stateColumns signalBound
    transitionErrorBound initialErrorBound transitionBound : Nat) [NeZero q]
    (signal transitionError initialError transition : Mxx.Matrix)
    (signalShape : Mxx.Toolkit.MatrixShape signal q ringDimension 1 stateRows)
    (transitionErrorShape :
      Mxx.Toolkit.MatrixShape transitionError q ringDimension stateRows stateColumns)
    (initialErrorShape :
      Mxx.Toolkit.MatrixShape initialError q ringDimension 1 stateColumns)
    (transitionShape :
      Mxx.Toolkit.MatrixShape transition q ringDimension stateColumns stateColumns)
    (signalNorm : Mxx.maxCenteredCoefficientNorm signal ≤ signalBound)
    (transitionErrorNorm :
      Mxx.maxCenteredCoefficientNorm transitionError ≤ transitionErrorBound)
    (initialErrorNorm : Mxx.maxCenteredCoefficientNorm initialError ≤ initialErrorBound)
    (transitionNorm : Mxx.maxCenteredCoefficientNorm transition ≤ transitionBound) :
    Mxx.maxCenteredCoefficientNorm
        (propagatedStateNoise signal transitionError initialError transition) ≤
      productBound ringDimension stateRows signalBound transitionErrorBound +
        productBound ringDimension stateColumns initialErrorBound transitionBound := by
  have signalProductShape :=
    Mxx.Toolkit.matrixMul_shape signal transitionError signalShape transitionErrorShape
  have initialProductShape :=
    Mxx.Toolkit.matrixMul_shape initialError transition initialErrorShape transitionShape
  apply le_trans
    (Mxx.Toolkit.matrixAdd_norm_le q _ _ signalProductShape.modulus
      initialProductShape.modulus)
  apply Nat.add_le_add
  · simpa [productBound] using
      Mxx.Toolkit.matrixMul_norm_le q ringDimension stateRows signalBound
        transitionErrorBound signal transitionError signalShape.modulus
        transitionErrorShape.modulus signalShape.ringDimension
        transitionErrorShape.ringDimension signalShape.columns transitionErrorShape.rows
        signalNorm transitionErrorNorm
  · simpa [productBound] using
      Mxx.Toolkit.matrixMul_norm_le q ringDimension stateColumns initialErrorBound
        transitionBound initialError transition initialErrorShape.modulus
        transitionShape.modulus initialErrorShape.ringDimension transitionShape.ringDimension
        initialErrorShape.columns transitionShape.rows initialErrorNorm transitionNorm

/-- The error produced by projecting an injected state through a bounded preimage. -/
def projectedStateNoise (stateNoise preimage : Mxx.Matrix) : Mxx.Matrix :=
  Mxx.matrixMul stateNoise preimage

theorem projectedStateNoise_norm_le (q ringDimension stateColumns outputColumns stateBound
    preimageBound : Nat) [NeZero q] (stateNoise preimage : Mxx.Matrix)
    (stateShape : Mxx.Toolkit.MatrixShape stateNoise q ringDimension 1 stateColumns)
    (preimageShape :
      Mxx.Toolkit.MatrixShape preimage q ringDimension stateColumns outputColumns)
    (stateNorm : Mxx.maxCenteredCoefficientNorm stateNoise ≤ stateBound)
    (preimageNorm : Mxx.maxCenteredCoefficientNorm preimage ≤ preimageBound) :
    Mxx.maxCenteredCoefficientNorm (projectedStateNoise stateNoise preimage) ≤
      productBound ringDimension stateColumns stateBound preimageBound := by
  simpa [projectedStateNoise, productBound] using
    Mxx.Toolkit.matrixMul_norm_le q ringDimension stateColumns stateBound preimageBound
      stateNoise preimage stateShape.modulus preimageShape.modulus stateShape.ringDimension
      preimageShape.ringDimension stateShape.columns preimageShape.rows stateNorm preimageNorm

/-- The exact error recurrence for a Boolean multiplication gate. -/
def productGateNoise (leftError rightError leftPlaintext rightDecomposed : Mxx.Matrix) :
    Mxx.Matrix :=
  Mxx.matrixAdd (Mxx.matrixMul leftError rightDecomposed)
    (Mxx.matrixMul leftPlaintext rightError)

theorem productGateNoise_norm_le (q ringDimension publicColumns leftBound rightBound digitBound :
    Nat) [NeZero q]
    (leftError rightError leftPlaintext rightDecomposed : Mxx.Matrix)
    (leftShape : Mxx.Toolkit.MatrixShape leftError q ringDimension 1 publicColumns)
    (rightShape : Mxx.Toolkit.MatrixShape rightError q ringDimension 1 publicColumns)
    (plaintextShape : Mxx.Toolkit.MatrixShape leftPlaintext q ringDimension 1 1)
    (decomposedShape :
      Mxx.Toolkit.MatrixShape rightDecomposed q ringDimension publicColumns publicColumns)
    (leftNorm : Mxx.maxCenteredCoefficientNorm leftError ≤ leftBound)
    (rightNorm : Mxx.maxCenteredCoefficientNorm rightError ≤ rightBound)
    (plaintextNorm : Mxx.maxCenteredCoefficientNorm leftPlaintext ≤ 1)
    (decomposedNorm : Mxx.maxCenteredCoefficientNorm rightDecomposed ≤ digitBound) :
    Mxx.maxCenteredCoefficientNorm
        (productGateNoise leftError rightError leftPlaintext rightDecomposed) ≤
      productBound ringDimension publicColumns leftBound digitBound +
        productBound ringDimension 1 rightBound 1 := by
  have leftProductShape := Mxx.Toolkit.matrixMul_shape leftError rightDecomposed
    leftShape decomposedShape
  have rightProductShape := Mxx.Toolkit.matrixMul_shape leftPlaintext rightError
    plaintextShape rightShape
  apply le_trans
    (Mxx.Toolkit.matrixAdd_norm_le q _ _ leftProductShape.modulus rightProductShape.modulus)
  apply Nat.add_le_add
  · simpa [productBound] using
      Mxx.Toolkit.matrixMul_norm_le q ringDimension publicColumns leftBound digitBound
        leftError rightDecomposed leftShape.modulus decomposedShape.modulus
        leftShape.ringDimension decomposedShape.ringDimension leftShape.columns
        decomposedShape.rows leftNorm decomposedNorm
  · have bound :=
      Mxx.Toolkit.matrixMul_norm_le q ringDimension 1 1 rightBound leftPlaintext rightError
        plaintextShape.modulus rightShape.modulus plaintextShape.ringDimension
        rightShape.ringDimension plaintextShape.columns rightShape.rows plaintextNorm rightNorm
    simpa [productBound, Nat.mul_comm, Nat.mul_left_comm, Nat.mul_assoc] using bound

/-- Concrete gate error selected by the dynamic Boolean opcode. -/
def booleanGateNoiseMatrix (gate : BooleanGate) (oneError leftError rightError leftPlaintext
    rightDecomposed : Mxx.Matrix) : Mxx.Matrix :=
  let product := productGateNoise leftError rightError leftPlaintext rightDecomposed
  match gate with
  | .constantFalse => Mxx.matrixSubtract oneError oneError
  | .constantTrue => oneError
  | .copy => leftError
  | .not => Mxx.matrixSubtract oneError leftError
  | .and => product
  | .xor => Mxx.matrixSubtract (Mxx.matrixAdd leftError rightError)
      (Mxx.matrixScale 2 product)

theorem booleanGateNoiseMatrix_norm_le (q ringDimension publicColumns digitBound oneBound
    leftBound rightBound : Nat) [NeZero q]
    (gate : BooleanGate) (oneError leftError rightError leftPlaintext rightDecomposed : Mxx.Matrix)
    (oneShape : Mxx.Toolkit.MatrixShape oneError q ringDimension 1 publicColumns)
    (leftShape : Mxx.Toolkit.MatrixShape leftError q ringDimension 1 publicColumns)
    (rightShape : Mxx.Toolkit.MatrixShape rightError q ringDimension 1 publicColumns)
    (plaintextShape : Mxx.Toolkit.MatrixShape leftPlaintext q ringDimension 1 1)
    (decomposedShape :
      Mxx.Toolkit.MatrixShape rightDecomposed q ringDimension publicColumns publicColumns)
    (oneNorm : Mxx.maxCenteredCoefficientNorm oneError ≤ oneBound)
    (leftNorm : Mxx.maxCenteredCoefficientNorm leftError ≤ leftBound)
    (rightNorm : Mxx.maxCenteredCoefficientNorm rightError ≤ rightBound)
    (plaintextNorm : Mxx.maxCenteredCoefficientNorm leftPlaintext ≤ 1)
    (decomposedNorm : Mxx.maxCenteredCoefficientNorm rightDecomposed ≤ digitBound) :
    Mxx.maxCenteredCoefficientNorm
        (booleanGateNoiseMatrix gate oneError leftError rightError leftPlaintext
          rightDecomposed) ≤
      gateNoise gate ringDimension publicColumns digitBound oneBound leftBound rightBound := by
  have productNorm := productGateNoise_norm_le q ringDimension publicColumns leftBound
    rightBound digitBound leftError rightError leftPlaintext rightDecomposed leftShape rightShape
    plaintextShape decomposedShape leftNorm rightNorm plaintextNorm decomposedNorm
  have productShape : Mxx.Toolkit.MatrixShape
      (productGateNoise leftError rightError leftPlaintext rightDecomposed)
      q ringDimension 1 publicColumns := by
    apply Mxx.Toolkit.matrixAdd_shape
    · exact Mxx.Toolkit.matrixMul_shape leftError rightDecomposed leftShape decomposedShape
    · exact Mxx.Toolkit.matrixMul_shape leftPlaintext rightError plaintextShape rightShape
  cases gate
  · simpa [booleanGateNoiseMatrix, gateNoise, two_mul] using
      le_trans
        (Mxx.Toolkit.matrixSubtract_norm_le q oneError oneError oneShape.modulus
          oneShape.modulus)
        (Nat.add_le_add oneNorm oneNorm)
  · simpa [booleanGateNoiseMatrix, gateNoise] using oneNorm
  · simpa [booleanGateNoiseMatrix, gateNoise] using leftNorm
  · simpa [booleanGateNoiseMatrix, gateNoise] using
      le_trans
        (Mxx.Toolkit.matrixSubtract_norm_le q oneError leftError oneShape.modulus
          leftShape.modulus)
        (Nat.add_le_add oneNorm leftNorm)
  · simpa [booleanGateNoiseMatrix, gateNoise] using productNorm
  · have sumShape := Mxx.Toolkit.matrixAdd_shape leftError rightError leftShape rightShape
    have scaledShape := Mxx.Toolkit.matrixScale_shape 2
      (productGateNoise leftError rightError leftPlaintext rightDecomposed) productShape
    have sumNorm : Mxx.maxCenteredCoefficientNorm (Mxx.matrixAdd leftError rightError) ≤
        leftBound + rightBound :=
      le_trans
        (Mxx.Toolkit.matrixAdd_norm_le q leftError rightError leftShape.modulus
          rightShape.modulus)
        (Nat.add_le_add leftNorm rightNorm)
    have scaledNorm : Mxx.maxCenteredCoefficientNorm
        (Mxx.matrixScale 2
          (productGateNoise leftError rightError leftPlaintext rightDecomposed)) ≤
        2 * (productBound ringDimension publicColumns leftBound digitBound +
          productBound ringDimension 1 rightBound 1) :=
      le_trans
        (Mxx.Toolkit.matrixScale_norm_le q 2
          (productGateNoise leftError rightError leftPlaintext rightDecomposed)
          productShape.modulus)
        (Nat.mul_le_mul_left 2 productNorm)
    apply le_trans
      (Mxx.Toolkit.matrixSubtract_norm_le q _ _ sumShape.modulus scaledShape.modulus)
    exact Nat.add_le_add sumNorm scaledNorm

theorem booleanGateNoiseMatrix_shape (q ringDimension publicColumns : Nat)
    (gate : BooleanGate) (oneError leftError rightError leftPlaintext rightDecomposed : Mxx.Matrix)
    (oneShape : Mxx.Toolkit.MatrixShape oneError q ringDimension 1 publicColumns)
    (leftShape : Mxx.Toolkit.MatrixShape leftError q ringDimension 1 publicColumns)
    (rightShape : Mxx.Toolkit.MatrixShape rightError q ringDimension 1 publicColumns)
    (plaintextShape : Mxx.Toolkit.MatrixShape leftPlaintext q ringDimension 1 1)
    (decomposedShape :
      Mxx.Toolkit.MatrixShape rightDecomposed q ringDimension publicColumns publicColumns) :
    Mxx.Toolkit.MatrixShape
      (booleanGateNoiseMatrix gate oneError leftError rightError leftPlaintext rightDecomposed)
      q ringDimension 1 publicColumns := by
  have productShape : Mxx.Toolkit.MatrixShape
      (productGateNoise leftError rightError leftPlaintext rightDecomposed)
      q ringDimension 1 publicColumns := by
    apply Mxx.Toolkit.matrixAdd_shape
    · exact Mxx.Toolkit.matrixMul_shape leftError rightDecomposed leftShape decomposedShape
    · exact Mxx.Toolkit.matrixMul_shape leftPlaintext rightError plaintextShape rightShape
  cases gate
  · exact Mxx.Toolkit.matrixSubtract_shape oneError oneError oneShape oneShape
  · exact oneShape
  · exact leftShape
  · exact Mxx.Toolkit.matrixSubtract_shape oneError leftError oneShape leftShape
  · exact productShape
  · apply Mxx.Toolkit.matrixSubtract_shape
    · exact Mxx.Toolkit.matrixAdd_shape leftError rightError leftShape rightShape
    · exact Mxx.Toolkit.matrixScale_shape 2 _ productShape

/-- Exact concrete error term canceled by the final Diamond decoder. -/
def finalDecoderNoise (decoderError kError oneError circuitError rDecomposed : Mxx.Matrix) :
    Mxx.Matrix :=
  Mxx.matrixSubtract decoderError
    (Mxx.matrixAdd kError
      (Mxx.matrixMul (Mxx.matrixSubtract oneError circuitError) rDecomposed))

theorem finalDecoderNoise_norm_le (q ringDimension publicColumns oneBound circuitBound digitBound :
    Nat) [NeZero q]
    (decoderError kError oneError circuitError rDecomposed : Mxx.Matrix)
    (decoderShape : Mxx.Toolkit.MatrixShape decoderError q ringDimension 1 1)
    (kShape : Mxx.Toolkit.MatrixShape kError q ringDimension 1 1)
    (oneShape : Mxx.Toolkit.MatrixShape oneError q ringDimension 1 publicColumns)
    (circuitShape : Mxx.Toolkit.MatrixShape circuitError q ringDimension 1 publicColumns)
    (rShape : Mxx.Toolkit.MatrixShape rDecomposed q ringDimension publicColumns 1)
    (decoderNorm : Mxx.maxCenteredCoefficientNorm decoderError ≤ oneBound)
    (kNorm : Mxx.maxCenteredCoefficientNorm kError ≤ oneBound)
    (oneNorm : Mxx.maxCenteredCoefficientNorm oneError ≤ oneBound)
    (circuitNorm : Mxx.maxCenteredCoefficientNorm circuitError ≤ circuitBound)
    (rNorm : Mxx.maxCenteredCoefficientNorm rDecomposed ≤ digitBound) :
    Mxx.maxCenteredCoefficientNorm
        (finalDecoderNoise decoderError kError oneError circuitError rDecomposed) ≤
      oneBound + oneBound +
        productBound ringDimension publicColumns (oneBound + circuitBound) digitBound := by
  have differenceShape :=
    Mxx.Toolkit.matrixSubtract_shape oneError circuitError oneShape circuitShape
  have productShape :=
    Mxx.Toolkit.matrixMul_shape (Mxx.matrixSubtract oneError circuitError) rDecomposed
      differenceShape rShape
  have sumShape := Mxx.Toolkit.matrixAdd_shape kError
    (Mxx.matrixMul (Mxx.matrixSubtract oneError circuitError) rDecomposed) kShape productShape
  apply le_trans
    (Mxx.Toolkit.matrixSubtract_norm_le q _ _ decoderShape.modulus sumShape.modulus)
  rw [Nat.add_assoc]
  apply Nat.add_le_add decoderNorm
  apply le_trans
    (Mxx.Toolkit.matrixAdd_norm_le q _ _ kShape.modulus productShape.modulus)
  apply Nat.add_le_add kNorm
  apply le_trans
    (Mxx.Toolkit.matrixMul_norm_le q ringDimension publicColumns
      (oneBound + circuitBound) digitBound (Mxx.matrixSubtract oneError circuitError)
      rDecomposed differenceShape.modulus rShape.modulus differenceShape.ringDimension
      rShape.ringDimension differenceShape.columns rShape.rows
      (le_trans
        (Mxx.Toolkit.matrixSubtract_norm_le q oneError circuitError oneShape.modulus
          circuitShape.modulus)
        (Nat.add_le_add oneNorm circuitNorm)) rNorm)
  simp [productBound]

/-- The concrete Diamond error recurrence for one input-injection layer followed by one Boolean
circuit layer is covered exactly by `diamondFinalBound` instantiated with both counts equal to one.
-/
theorem diamondFinalBound_oneInput_oneLayer_norm_le
    (q ringDimension stateRows stateColumns publicColumns gadgetBase samplerError preimageBound :
      Nat)
    [NeZero q]
    (gate : BooleanGate)
    (signal transitionError initialError transition statePreimage leftError rightError
      leftPlaintext rightDecomposed decoderError kError rDecomposed : Mxx.Matrix)
    (signalShape : Mxx.Toolkit.MatrixShape signal q ringDimension 1 stateRows)
    (transitionErrorShape :
      Mxx.Toolkit.MatrixShape transitionError q ringDimension stateRows stateColumns)
    (initialErrorShape :
      Mxx.Toolkit.MatrixShape initialError q ringDimension 1 stateColumns)
    (transitionShape :
      Mxx.Toolkit.MatrixShape transition q ringDimension stateColumns stateColumns)
    (statePreimageShape :
      Mxx.Toolkit.MatrixShape statePreimage q ringDimension stateColumns publicColumns)
    (leftShape : Mxx.Toolkit.MatrixShape leftError q ringDimension 1 publicColumns)
    (rightShape : Mxx.Toolkit.MatrixShape rightError q ringDimension 1 publicColumns)
    (plaintextShape : Mxx.Toolkit.MatrixShape leftPlaintext q ringDimension 1 1)
    (rightDecomposedShape :
      Mxx.Toolkit.MatrixShape rightDecomposed q ringDimension publicColumns publicColumns)
    (decoderShape : Mxx.Toolkit.MatrixShape decoderError q ringDimension 1 1)
    (kShape : Mxx.Toolkit.MatrixShape kError q ringDimension 1 1)
    (rShape : Mxx.Toolkit.MatrixShape rDecomposed q ringDimension publicColumns 1)
    (signalNorm : Mxx.maxCenteredCoefficientNorm signal ≤ 1)
    (transitionErrorNorm :
      Mxx.maxCenteredCoefficientNorm transitionError ≤ samplerError)
    (initialErrorNorm : Mxx.maxCenteredCoefficientNorm initialError ≤ samplerError)
    (transitionNorm : Mxx.maxCenteredCoefficientNorm transition ≤ preimageBound)
    (statePreimageNorm : Mxx.maxCenteredCoefficientNorm statePreimage ≤ preimageBound)
    (leftNorm : Mxx.maxCenteredCoefficientNorm leftError ≤
      2 * productBound ringDimension stateColumns
        (productBound ringDimension stateRows 1 samplerError +
          productBound ringDimension stateColumns samplerError preimageBound)
        preimageBound)
    (rightNorm : Mxx.maxCenteredCoefficientNorm rightError ≤
      2 * productBound ringDimension stateColumns
        (productBound ringDimension stateRows 1 samplerError +
          productBound ringDimension stateColumns samplerError preimageBound)
        preimageBound)
    (plaintextNorm : Mxx.maxCenteredCoefficientNorm leftPlaintext ≤ 1)
    (rightDecomposedNorm : Mxx.maxCenteredCoefficientNorm rightDecomposed ≤
      max (gadgetBase / 2) 1)
    (decoderNorm : Mxx.maxCenteredCoefficientNorm decoderError ≤
      productBound ringDimension stateColumns
        (productBound ringDimension stateRows 1 samplerError +
          productBound ringDimension stateColumns samplerError preimageBound)
        preimageBound)
    (kNorm : Mxx.maxCenteredCoefficientNorm kError ≤
      productBound ringDimension stateColumns
        (productBound ringDimension stateRows 1 samplerError +
          productBound ringDimension stateColumns samplerError preimageBound)
        preimageBound)
    (rNorm : Mxx.maxCenteredCoefficientNorm rDecomposed ≤ max (gadgetBase / 2) 1) :
    let stateNoise := propagatedStateNoise signal transitionError initialError transition
    let oneError := projectedStateNoise stateNoise statePreimage
    let circuitError := booleanGateNoiseMatrix gate oneError leftError rightError leftPlaintext
      rightDecomposed
    Mxx.maxCenteredCoefficientNorm
        (finalDecoderNoise decoderError kError oneError circuitError rDecomposed) ≤
      diamondFinalBound ringDimension stateRows stateColumns publicColumns 1 1 gadgetBase
        samplerError preimageBound := by
  dsimp only
  let stateBound := productBound ringDimension stateRows 1 samplerError +
    productBound ringDimension stateColumns samplerError preimageBound
  let oneBound := productBound ringDimension stateColumns stateBound preimageBound
  let digitBound := max (gadgetBase / 2) 1
  let inputBound := 2 * oneBound
  let outputBound := gateStep ringDimension publicColumns digitBound oneBound inputBound
  have stateNoiseShape : Mxx.Toolkit.MatrixShape
      (propagatedStateNoise signal transitionError initialError transition)
      q ringDimension 1 stateColumns := by
    apply Mxx.Toolkit.matrixAdd_shape
    · exact Mxx.Toolkit.matrixMul_shape signal transitionError signalShape transitionErrorShape
    · exact Mxx.Toolkit.matrixMul_shape initialError transition initialErrorShape transitionShape
  have stateNoiseNorm : Mxx.maxCenteredCoefficientNorm
      (propagatedStateNoise signal transitionError initialError transition) ≤ stateBound := by
    exact propagatedStateNoise_norm_le q ringDimension stateRows stateColumns 1 samplerError
      samplerError preimageBound signal transitionError initialError transition signalShape
      transitionErrorShape initialErrorShape transitionShape signalNorm transitionErrorNorm
      initialErrorNorm transitionNorm
  have oneShape : Mxx.Toolkit.MatrixShape
      (projectedStateNoise
        (propagatedStateNoise signal transitionError initialError transition) statePreimage)
      q ringDimension 1 publicColumns :=
    Mxx.Toolkit.matrixMul_shape _ _ stateNoiseShape statePreimageShape
  have oneNorm : Mxx.maxCenteredCoefficientNorm
      (projectedStateNoise
        (propagatedStateNoise signal transitionError initialError transition) statePreimage) ≤
      oneBound := by
    exact projectedStateNoise_norm_le q ringDimension stateColumns publicColumns stateBound
      preimageBound _ statePreimage stateNoiseShape statePreimageShape stateNoiseNorm
      statePreimageNorm
  have circuitNorm : Mxx.maxCenteredCoefficientNorm
      (booleanGateNoiseMatrix gate
        (projectedStateNoise
          (propagatedStateNoise signal transitionError initialError transition) statePreimage)
        leftError rightError leftPlaintext rightDecomposed) ≤ outputBound := by
    apply le_trans
      (booleanGateNoiseMatrix_norm_le q ringDimension publicColumns digitBound oneBound
        inputBound inputBound gate _ leftError rightError leftPlaintext rightDecomposed oneShape
        leftShape rightShape plaintextShape rightDecomposedShape oneNorm leftNorm rightNorm
        plaintextNorm rightDecomposedNorm)
    exact gateNoise_le_gateStep gate ringDimension publicColumns digitBound oneBound inputBound
      inputBound inputBound (Nat.le_refl _) (Nat.le_refl _)
  have circuitShape := booleanGateNoiseMatrix_shape q ringDimension publicColumns gate
    (projectedStateNoise
      (propagatedStateNoise signal transitionError initialError transition) statePreimage)
    leftError rightError leftPlaintext rightDecomposed oneShape leftShape rightShape plaintextShape
    rightDecomposedShape
  apply le_trans
    (finalDecoderNoise_norm_le q ringDimension publicColumns oneBound outputBound digitBound
      decoderError kError
      (projectedStateNoise
        (propagatedStateNoise signal transitionError initialError transition) statePreimage)
      (booleanGateNoiseMatrix gate
        (projectedStateNoise
          (propagatedStateNoise signal transitionError initialError transition) statePreimage)
        leftError rightError leftPlaintext rightDecomposed)
      rDecomposed decoderShape kShape oneShape circuitShape rShape decoderNorm kNorm oneNorm
      circuitNorm rNorm)
  simp [diamondFinalBound, injectionBound, injectionStep, circuitBound, stateBound, oneBound,
    digitBound, inputBound, outputBound, Nat.add_comm]
/-- Exact interval decoder used by the Diamond graph after all ring operations have returned the
canonical nonnegative residue. -/
def decodeBooleanInterval (modulus value : Int) : Bool :=
  decide (modulus / 4 ≤ value) && decide (value ≤ 3 * (modulus / 4))

private theorem error_lt_quarter_bounds {modulus error : Int}
    (modulus_ge : 4 ≤ modulus) (errorBound : error.natAbs < (modulus / 4).toNat) :
    -(modulus / 4) < error ∧ error < modulus / 4 := by
  have quarter_nonnegative : 0 ≤ modulus / 4 := by omega
  have absolute : (error.natAbs : Int) < modulus / 4 := by
    rw [← Int.toNat_of_nonneg quarter_nonnegative]
    exact_mod_cast errorBound
  constructor
  · by_cases nonnegative : 0 ≤ error
    · omega
    · have negAbsolute : (-error).natAbs = error.natAbs := by simp
      have negNonnegative : 0 ≤ -error := by omega
      rw [← negAbsolute, Int.natAbs_of_nonneg negNonnegative] at absolute
      omega
  · by_cases nonnegative : 0 ≤ error
    · rw [Int.natAbs_of_nonneg nonnegative] at absolute
      exact absolute
    · omega

/-- A residue within the checker-certified strict quarter-modulus error interval decodes to the
original Boolean.  The proof covers all positive modulus congruence classes, not only moduli that
are multiples of four. -/
theorem decodeBooleanInterval_correct (message : Bool) (modulus error : Int)
    (modulus_ge : 4 ≤ modulus)
    (errorBound : error.natAbs < (modulus / 4).toNat) :
    decodeBooleanInterval modulus
      (((if message then modulus / 2 else 0) + error) % modulus) = message := by
  obtain ⟨lower, upper⟩ := error_lt_quarter_bounds modulus_ge errorBound
  have modulus_pos : 0 < modulus := by omega
  cases message
  · by_cases nonnegative : 0 ≤ error
    · have reduced : error % modulus = error := Int.emod_eq_of_lt nonnegative (by omega)
      simp [decodeBooleanInterval, reduced]
      omega
    · have shifted : (error + modulus) % modulus = error + modulus :=
        Int.emod_eq_of_lt (by omega) (by omega)
      have reduced : error % modulus = error + modulus := by
        simpa [Int.add_emod] using shifted
      simp [decodeBooleanInterval, reduced]
      omega
  · have signalLower : modulus / 4 ≤ modulus / 2 + error := by omega
    have signalUpper : modulus / 2 + error ≤ 3 * (modulus / 4) := by omega
    have signalNonnegative : 0 ≤ modulus / 2 + error := by omega
    have signalBelow : modulus / 2 + error < modulus := by omega
    have reduced : (modulus / 2 + error) % modulus = modulus / 2 + error :=
      Int.emod_eq_of_lt signalNonnegative signalBelow
    simp [decodeBooleanInterval, reduced, signalLower, signalUpper]

/-- Final bridge used by the protocol proof: the executable checker bounds the concrete centered
error, and the algebraic Diamond invariant identifies the output coefficient with message signal
plus that error modulo the ciphertext modulus. -/
theorem diamondDecoderCorrect (message : Bool)
    (ringDimension stateRows stateColumns publicColumns inputCount layerCount gadgetBase
      samplerError preimage modulus actualError : Nat)
    (modulus_ge : 4 ≤ modulus)
    (accepted : diamondChecker ringDimension stateRows stateColumns publicColumns inputCount
      layerCount gadgetBase samplerError preimage modulus = true)
    (actual_le : actualError ≤ diamondFinalBound ringDimension stateRows stateColumns
      publicColumns inputCount layerCount gadgetBase samplerError preimage)
    (error : Int) (error_le : error.natAbs ≤ actualError) :
    decodeBooleanInterval modulus
      (((if message then (modulus : Int) / 2 else 0) + error) % modulus) = message := by
  have actual_lt := diamondChecker_sound _ _ _ _ _ _ _ _ _ _ _ accepted actual_le
  have error_lt : error.natAbs < modulus / 4 := Nat.lt_of_le_of_lt error_le actual_lt
  exact decodeBooleanInterval_correct message modulus error (by exact_mod_cast modulus_ge)
    (by exact_mod_cast error_lt)

end MxxWe
