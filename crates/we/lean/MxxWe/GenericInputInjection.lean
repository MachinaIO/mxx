import MxxWe.DiamondGeneric

open Mxx.Toolkit

namespace MxxWe

noncomputable section

abbrev RingMatrix (q n rows columns : Nat) :=
  _root_.Matrix (Fin rows) (Fin columns) (Negacyclic q n)

def polynomialScale {q n rows columns : Nat} (scalar : RingMatrix q n 1 1)
    (matrix : RingMatrix q n rows columns) : RingMatrix q n rows columns :=
  fun row column ↦ scalar 0 0 * matrix row column

/-- Algebraic BGG+ invariant used by the generated Diamond execution proof. -/
def EncodingEquation {q n columns : Nat} (secret : RingMatrix q n 1 1)
    (vector publicKey : RingMatrix q n 1 columns) (plaintext : RingMatrix q n 1 1)
    (gadget error : RingMatrix q n 1 columns) : Prop :=
  vector = polynomialScale secret (publicKey - polynomialScale plaintext gadget) + error

theorem encoding_add {q n columns : Nat} (secret : RingMatrix q n 1 1)
    (leftVector leftKey leftError rightVector rightKey rightError gadget :
      RingMatrix q n 1 columns)
    (leftPlaintext rightPlaintext : RingMatrix q n 1 1)
    (left : EncodingEquation secret leftVector leftKey leftPlaintext gadget leftError)
    (right : EncodingEquation secret rightVector rightKey rightPlaintext gadget rightError) :
    EncodingEquation secret (leftVector + rightVector) (leftKey + rightKey)
      (leftPlaintext + rightPlaintext) gadget (leftError + rightError) := by
  rw [EncodingEquation, left, right]
  ext row column
  fin_cases row
  simp [polynomialScale]
  ring

theorem encoding_subtract {q n columns : Nat} (secret : RingMatrix q n 1 1)
    (leftVector leftKey leftError rightVector rightKey rightError gadget :
      RingMatrix q n 1 columns)
    (leftPlaintext rightPlaintext : RingMatrix q n 1 1)
    (left : EncodingEquation secret leftVector leftKey leftPlaintext gadget leftError)
    (right : EncodingEquation secret rightVector rightKey rightPlaintext gadget rightError) :
    EncodingEquation secret (leftVector - rightVector) (leftKey - rightKey)
      (leftPlaintext - rightPlaintext) gadget (leftError - rightError) := by
  rw [EncodingEquation, left, right]
  ext row column
  fin_cases row
  simp [polynomialScale]
  ring

theorem encoding_scale {q n columns : Nat} (factor secret plaintext : RingMatrix q n 1 1)
    (vector publicKey gadget error : RingMatrix q n 1 columns)
    (equation : EncodingEquation secret vector publicKey plaintext gadget error) :
    EncodingEquation secret (polynomialScale factor vector)
      (polynomialScale factor publicKey) (factor * plaintext) gadget
      (polynomialScale factor error) := by
  rw [EncodingEquation, equation]
  ext row column
  fin_cases row
  simp [polynomialScale, _root_.Matrix.mul_apply]
  ring

theorem injection_transition {q n stateColumns : Nat}
    (initial : RingMatrix q n 1 stateColumns) (initialSignal : RingMatrix q n 1 2)
    (initialBase : RingMatrix q n 2 stateColumns)
    (initialError : RingMatrix q n 1 stateColumns)
    (transition : RingMatrix q n stateColumns stateColumns)
    (selector : RingMatrix q n 2 2) (nextBase transitionError : RingMatrix q n 2 stateColumns)
    (initialEquation : initial = initialSignal * initialBase + initialError)
    (transitionEquation : initialBase * transition = selector * nextBase + transitionError) :
    initial * transition =
      (initialSignal * selector) * nextBase +
        (initialSignal * transitionError + initialError * transition) := by
  rw [initialEquation, _root_.Matrix.add_mul, _root_.Matrix.mul_assoc,
    transitionEquation, _root_.Matrix.mul_add]
  rw [← _root_.Matrix.mul_assoc]
  abel

theorem injection_projection {q n stateColumns outputColumns : Nat}
    (state : RingMatrix q n 1 stateColumns) (signal : RingMatrix q n 1 2)
    (base : RingMatrix q n 2 stateColumns) (stateError : RingMatrix q n 1 stateColumns)
    (preimage : RingMatrix q n stateColumns outputColumns)
    (target : RingMatrix q n 2 outputColumns)
    (stateEquation : state = signal * base + stateError)
    (preimageEquation : base * preimage = target) :
    state * preimage = signal * target + stateError * preimage := by
  rw [stateEquation, _root_.Matrix.add_mul, _root_.Matrix.mul_assoc, preimageEquation]

private theorem finCasesOne {α : Type} (zero : α) (succ : Fin 1 → α) :
    Fin.cases zero succ (1 : Fin 2) = succ 0 := by
  have oneEq : (1 : Fin 2) = Fin.succ 0 := by decide
  rw [oneEq]
  exact Fin.cases_succ 0

theorem pairColumns_mul_diagonalPair {q n : Nat}
    (secret message nextSecret : RingMatrix q n 1 1) :
    pairColumns secret message * diagonalPair nextSecret 1 =
      pairColumns (secret * nextSecret) message := by
  ext row column
  fin_cases row
  fin_cases column <;>
    norm_num [pairColumns, diagonalPair, _root_.Matrix.mul_apply, Fin.sum_univ_two,
      finCasesOne]

theorem pairColumns_mul_specialSelector {q n : Nat}
    (secret message nextSecret bit : RingMatrix q n 1 1) :
    pairColumns secret message * pairRows (pairColumns nextSecret (nextSecret * bit)) 0 =
      pairColumns (secret * nextSecret) ((secret * nextSecret) * bit) := by
  ext row column
  fin_cases row
  fin_cases column <;>
    norm_num [pairColumns, pairRows, _root_.Matrix.mul_apply, Fin.sum_univ_two,
      finCasesOne, mul_assoc]

theorem normal_injected_state_equation {q n stateColumns : Nat}
    (secret message nextSecret : RingMatrix q n 1 1)
    (initial : RingMatrix q n 1 stateColumns) (initialBase : RingMatrix q n 2 stateColumns)
    (initialError : RingMatrix q n 1 stateColumns)
    (transition : RingMatrix q n stateColumns stateColumns)
    (nextBase transitionError : RingMatrix q n 2 stateColumns)
    (initialEquation : initial = pairColumns secret message * initialBase + initialError)
    (transitionEquation :
      initialBase * transition = diagonalPair nextSecret 1 * nextBase + transitionError) :
    initial * transition =
      pairColumns (secret * nextSecret) message * nextBase +
        (pairColumns secret message * transitionError + initialError * transition) := by
  rw [injection_transition initial (pairColumns secret message) initialBase initialError
    transition (diagonalPair nextSecret 1) nextBase transitionError initialEquation
    transitionEquation, pairColumns_mul_diagonalPair]

theorem special_injected_state_equation {q n stateColumns : Nat}
    (secret message nextSecret bit : RingMatrix q n 1 1)
    (initial : RingMatrix q n 1 stateColumns) (initialBase : RingMatrix q n 2 stateColumns)
    (initialError : RingMatrix q n 1 stateColumns)
    (transition : RingMatrix q n stateColumns stateColumns)
    (nextBase transitionError : RingMatrix q n 2 stateColumns)
    (initialEquation : initial = pairColumns secret message * initialBase + initialError)
    (transitionEquation : initialBase * transition =
      pairRows (pairColumns nextSecret (nextSecret * bit)) 0 * nextBase + transitionError) :
    initial * transition =
      pairColumns (secret * nextSecret) ((secret * nextSecret) * bit) * nextBase +
        (pairColumns secret message * transitionError + initialError * transition) := by
  rw [injection_transition initial (pairColumns secret message) initialBase initialError
    transition (pairRows (pairColumns nextSecret (nextSecret * bit)) 0) nextBase
    transitionError initialEquation transitionEquation,
    pairColumns_mul_specialSelector secret message nextSecret bit]

theorem final_decoder_algebra {q n publicColumns : Nat}
    (messageHalf secret kKey : RingMatrix q n 1 1)
    (oneKey circuitKey gadget : RingMatrix q n 1 publicColumns)
    (rDecomposed : RingMatrix q n publicColumns 1)
    (oneVector circuitVector : RingMatrix q n 1 publicColumns)
    (kVector decoderVector : RingMatrix q n 1 1)
    (oneError circuitError : RingMatrix q n 1 publicColumns)
    (kError decoderError : RingMatrix q n 1 1)
    (oneEquation : EncodingEquation secret oneVector oneKey 1 gadget oneError)
    (circuitEquation : EncodingEquation secret circuitVector circuitKey 1 gadget circuitError)
    (kEquation : kVector = polynomialScale secret kKey + messageHalf + kError)
    (decoderEquation : decoderVector = polynomialScale secret
      (kKey + (oneKey - circuitKey) * rDecomposed) + decoderError) :
    decoderVector - (kVector + (oneVector - circuitVector) * rDecomposed) =
      -messageHalf +
        (decoderError - (kError + (oneError - circuitError) * rDecomposed)) := by
  rw [EncodingEquation] at oneEquation circuitEquation
  rw [oneEquation, circuitEquation, kEquation, decoderEquation]
  ext row column
  fin_cases row
  fin_cases column
  simp only [_root_.Matrix.add_apply, _root_.Matrix.sub_apply, _root_.Matrix.neg_apply,
    polynomialScale, _root_.Matrix.mul_apply]
  have finOne (index : Fin 1) : index = 0 := Subsingleton.elim _ _
  simp_rw [finOne]
  simp
  have signalSum :
      secret 0 0 *
          (∑ x, (oneKey 0 x * rDecomposed x 0 -
            circuitKey 0 x * rDecomposed x 0)) =
        ∑ x, (secret 0 0 * oneKey 0 x * rDecomposed x 0 -
          secret 0 0 * circuitKey 0 x * rDecomposed x 0) := by
    rw [Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro index _
    ring
  have splitSum :
      (∑ x,
        (secret 0 0 * oneKey 0 x * rDecomposed x 0 -
            secret 0 0 * circuitKey 0 x * rDecomposed x 0 +
          oneError 0 x * rDecomposed x 0 - circuitError 0 x * rDecomposed x 0)) =
        (∑ x, (secret 0 0 * oneKey 0 x * rDecomposed x 0 -
          secret 0 0 * circuitKey 0 x * rDecomposed x 0)) +
        ∑ x, (oneError 0 x * rDecomposed x 0 -
          circuitError 0 x * rDecomposed x 0) := by
    rw [← Finset.sum_add_distrib]
    apply Finset.sum_congr rfl
    intro index _
    ring
  have keyDifferenceSum :
      (∑ x, (oneKey 0 x - circuitKey 0 x) * rDecomposed x 0) =
        ∑ x, (oneKey 0 x * rDecomposed x 0 -
          circuitKey 0 x * rDecomposed x 0) := by
    apply Finset.sum_congr rfl
    intro index _
    ring
  have encodingDifferenceSum :
      (∑ x,
        (secret 0 0 * (oneKey 0 x - gadget 0 x) + oneError 0 x -
          (secret 0 0 * (circuitKey 0 x - gadget 0 x) + circuitError 0 x)) *
            rDecomposed x 0) =
        ∑ x,
          (secret 0 0 * oneKey 0 x * rDecomposed x 0 -
              secret 0 0 * circuitKey 0 x * rDecomposed x 0 +
            oneError 0 x * rDecomposed x 0 -
              circuitError 0 x * rDecomposed x 0) := by
    apply Finset.sum_congr rfl
    intro index _
    ring
  have errorDifferenceSum :
      (∑ x, (oneError 0 x - circuitError 0 x) * rDecomposed x 0) =
        ∑ x, (oneError 0 x * rDecomposed x 0 -
          circuitError 0 x * rDecomposed x 0) := by
    apply Finset.sum_congr rfl
    intro index _
    ring
  rw [keyDifferenceSum, encodingDifferenceSum, splitSum, errorDifferenceSum]
  rw [mul_add, signalSum]
  ring

theorem decodeFromCongruence (p : DiamondWeParameters) (message : Bool)
    (accepted : diamondWeChecker p = true) (actualBound : Nat)
    (actualBoundLe : actualBound ≤ p.finalBound) (residual noisy : Mxx.Matrix)
    (residualModulus : residual.modulus = p.modulus)
    (residualBound : Mxx.maxCenteredCoefficientNorm residual ≤ actualBound)
    (noisyCanonical : noisy.coefficients.headD 0 =
      Mxx.reduceCoefficient p.modulus (noisy.coefficients.headD 0))
    (congruent : (noisy.coefficients.headD 0 : ZMod p.modulus) =
      (((if message then (p.modulus : Int) / 2 else 0) +
        Mxx.centeredCoefficient p.modulus (residual.coefficients.headD 0) : Int) :
        ZMod p.modulus)) :
    decodeBooleanInterval p.modulus (noisy.coefficients.headD 0) = message := by
  have modulusPositive : 0 < p.modulus := lt_of_lt_of_le (by omega)
    (diamondWeChecker_modulus_ge p accepted)
  letI : NeZero p.modulus := ⟨modulusPositive.ne'⟩
  let error := Mxx.centeredCoefficient p.modulus (residual.coefficients.headD 0)
  have errorLeNorm : error.natAbs ≤ Mxx.maxCenteredCoefficientNorm residual := by
    simpa [error, residualModulus] using Mxx.headD_natAbs_le_norm residual
  have decoded := diamondWeGenericDecoderCorrect p message accepted actualBound actualBoundLe error
    (le_trans errorLeNorm residualBound)
  have actualEq : noisy.coefficients.headD 0 =
      ((if message then (p.modulus : Int) / 2 else 0) + error) % p.modulus :=
    canonical_eq_emod_of_zmod_eq p.modulus _ _ noisyCanonical (by simpa [error] using congruent)
  rw [actualEq]
  exact decoded

end

end MxxWe
