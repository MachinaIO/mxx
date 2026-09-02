import MxxBgg.Bounds
import MxxGadgets.Decomposition
import MxxGadgets.InputInjector
import MxxGadgets.InjectorInvariant
import MxxWe.DiamondWE.Model
import MxxWe.DiamondWE.Decoder

namespace Mxx.We.DiamondWE

open Mxx.Primitives

open Mxx.Gadgets

/- The generated evaluator proof constructs one dependent chain from the actual carried state and
   selected transition at every loop iteration.  The four public output fields form the attachment
   surface for the runtime loop output: each is definitionally tied to the final chain state, while
   a later operational theorem may identify `outputValue` with the traced family element. -/
structure InjectorFoldTrace
    {q n sourceRows columns resultRows count : Nat} where
  hn : 0 < n
  chain : InjectorStateInvariant.IndexedTransitionChain
    (q := q) (n := n) (sourceRows := sourceRows) (columns := columns)
    (resultRows := resultRows) hn count
  outputSource : ExactMatrix q n sourceRows columns
  outputLeft : ExactMatrix q n resultRows sourceRows
  outputValue : ExactMatrix q n resultRows columns
  outputNoiseBound : Nat
  outputSource_eq : outputSource = chain.final.source
  outputLeft_eq : outputLeft = chain.final.left
  outputValue_eq : outputValue = chain.final.value
  outputNoiseBound_eq : outputNoiseBound = chain.final.stateNoiseBound

/- One concrete preimage-consuming loop step.  This is the exact equation used by the generated
   transition proof: `value = left * source + e` and `source * preimage = target`, while the
   target approximation says `target = transitionLeft * nextSource + E`.  The returned equation
   keeps the target ideal and exposes precisely `leftMagnitude.lift * E + e * preimage` as its
   integer witness. -/
theorem injector_transition_equation
    {q n sourceRows columns resultRows : Nat}
    {source : ExactMatrix q n sourceRows columns}
    {left : ExactMatrix q n resultRows sourceRows}
    {value : ExactMatrix q n resultRows columns}
    {stateNoiseBound : Nat}
    (state : InjectorStateInvariant source left value stateNoiseBound)
    (transition : InjectorStateInvariant.Transition source)
    :
    value * transition.actualPreimage =
      (left * transition.transitionLeft) * transition.nextSource +
        reduceMatrix q n resultRows columns
          (state.leftMagnitude.lift * transition.targetApprox.error +
            state.approximation.error * transition.preimageLift.witness) := by
  let approximation := Mxx.Gadgets.input_injector_consumption source transition.actualPreimage
    transition.actualTarget left value
    (transition.transitionLeft * transition.nextSource) transition.relation state.leftMagnitude
    transition.preimageLift state.approximation transition.targetApprox
  have equation := approximation.equation
  simpa [Matrix.mul_assoc] using equation

/- The final invariant is already stored in the final dependent state.  Rewriting by the four
   output identities exposes it at the application-facing matrices and bound. -/
noncomputable def injector_fold_trace_invariant
    {q n sourceRows columns resultRows count : Nat}
    (trace : InjectorFoldTrace (q := q) (n := n) (sourceRows := sourceRows)
      (columns := columns) (resultRows := resultRows) (count := count)) :
    InjectorStateInvariant trace.outputSource trace.outputLeft trace.outputValue
      trace.outputNoiseBound := by
  have invariant := trace.chain.finalInvariant
  rw [trace.outputSource_eq, trace.outputLeft_eq, trace.outputValue_eq,
    trace.outputNoiseBound_eq]
  exact invariant

/- These aliases expose the algebraic steps that a generated Diamond view may use.  They are
   deliberately local equations, not a graph-wide normal form or a theorem about an unexamined
   candidate graph. -/
theorem input_injector_step
    {R : Type u} [Semiring R]
    {s k c r : Type v} [Fintype s] [Fintype k]
    {B : Matrix s k R} {K : Matrix k c R} {P E : Matrix s c R}
    {X eX : Matrix r k R} {L : Matrix r s R}
    (hvalue : X = L * B + eX)
    (htarget : B * K = P + E) :
    X * K = L * P + (L * E + eX * K) := by
  exact Mxx.Primitives.consume_rectangular_semiring B K (P + E) P L X eX E hvalue htarget rfl

/- The application layer obtains this approximation only from the primitive preimage relation,
   bounded lifts, and the two incoming approximations.  It accepts no caller-provided output
   equation or output-error bound. -/
noncomputable def input_injector_step_with_bound
    {q n sourceRows inner targetColumns resultRows : Nat}
    (hn : 0 < n)
    (source : ExactMatrix q n sourceRows inner)
    (actualPreimage : ExactMatrix q n inner targetColumns)
    (actualTarget : ExactMatrix q n sourceRows targetColumns)
    (left : ExactMatrix q n resultRows sourceRows)
    (value : ExactMatrix q n resultRows inner)
    (idealTarget : ExactMatrix q n sourceRows targetColumns)
    (relation : RightPreimage source actualPreimage actualTarget)
    (leftMagnitude : MagnitudeFact left)
    {preimageBound xNoiseBound targetNoiseBound : Nat}
    (preimageLift : BoundedLift actualPreimage preimageBound)
    (valueApprox : ApproxWithin value (left * source) xNoiseBound)
    (targetApprox : ApproxWithin actualTarget idealTarget targetNoiseBound) :
    ApproxWithin (value * actualPreimage) (left * idealTarget)
      (sourceRows * n * leftMagnitude.bound * targetNoiseBound +
        inner * n * xNoiseBound * preimageBound) :=
  Mxx.Gadgets.input_injector_within hn source actualPreimage actualTarget left value idealTarget
    relation leftMagnitude preimageLift valueApprox targetApprox

theorem bgg_layer_step
    {R : Type u} [CommRing R]
    {secret gadgetCols : Type v} [Fintype secret] [Fintype gadgetCols]
    {gadget leftPublic rightPublic targetError : Matrix secret gadgetCols R}
    {decomposition : Matrix gadgetCols gadgetCols R}
    {leftCiphertext rightCiphertext : Matrix (Fin 1) gadgetCols R}
    {mask leftPayload rightPayload : Matrix (Fin 1) secret R}
    {leftMessage rightMessage : R}
    {leftError rightError : Matrix (Fin 1) gadgetCols R}
    (leftEquation :
      leftCiphertext = mask * leftPublic - leftMessage • (leftPayload * gadget) + leftError)
    (rightEquation :
      rightCiphertext = mask * rightPublic - rightMessage • (rightPayload * gadget) + rightError)
    (leftPayload_eq : leftPayload = mask)
    (targetEquation : gadget * decomposition = rightPublic + targetError) :
    leftCiphertext * decomposition + leftMessage • rightCiphertext =
      mask * (leftPublic * decomposition) -
        (leftMessage * rightMessage) • (rightPayload * gadget) +
        (leftError * decomposition + leftMessage • rightError -
          leftMessage • (mask * targetError)) := by
  exact Mxx.Bgg.multiplication_core leftEquation rightEquation leftPayload_eq targetEquation

/- One BGG layer returns its exact encoding equation together with the tight coefficient-norm
   bound derived from the primitive witnesses.  Each matrix product contributes one factor `n`. -/
noncomputable def bgg_layer_with_bound
    {q n secretColumns gadgetColumns preimageBound : Nat}
    {leftCiphertext rightCiphertext : ExactMatrix q n 1 gadgetColumns}
    {leftMask leftPayload rightMask rightPayload : ExactMatrix q n 1 secretColumns}
    {leftPublic rightPublic gadget : ExactMatrix q n secretColumns gadgetColumns}
    {leftMessage rightMessage : ExactPoly q n}
    (left : Mxx.Bgg.Encoding leftCiphertext leftMask leftPayload leftPublic gadget leftMessage)
    (right : Mxx.Bgg.Encoding
      rightCiphertext rightMask rightPayload rightPublic gadget rightMessage)
    {decomposition : ExactMatrix q n gadgetColumns gadgetColumns}
    {actualTarget : ExactMatrix q n secretColumns gadgetColumns}
    (relation : RightPreimage gadget decomposition actualTarget)
    (targetApprox : Approx actualTarget rightPublic)
    (leftMaskMagnitude : MagnitudeFact leftMask)
    (preimageLift : BoundedLift decomposition preimageBound)
    (messageLift : ErrorPoly n)
    (message_reduce : reducePoly q n messageLift = leftMessage)
    (mask_eq : leftMask = rightMask)
    (leftPayload_eq : leftPayload = leftMask)
    (leftErrorBound rightErrorBound targetErrorBound messageBound : Nat)
    (left_error_le : matrixNorm left.error ≤ leftErrorBound)
    (right_error_le : matrixNorm right.error ≤ rightErrorBound)
    (target_error_le : matrixNorm targetApprox.error ≤ targetErrorBound)
    (message_le : polyNorm messageLift ≤ messageBound)
    (hn : 0 < n) :
    Subtype (fun output : Mxx.Bgg.Encoding
      (leftCiphertext * decomposition + leftMessage • rightCiphertext)
      leftMask rightPayload (leftPublic * decomposition) gadget (leftMessage * rightMessage) =>
      matrixNorm output.error ≤
        gadgetColumns * n * leftErrorBound * preimageBound +
          (n * messageBound * rightErrorBound +
            n * messageBound *
              (secretColumns * n * leftMaskMagnitude.bound * targetErrorBound))) := by
  let output := Mxx.Bgg.multiply left right relation targetApprox leftMaskMagnitude preimageLift
    messageLift message_reduce mask_eq leftPayload_eq
  refine ⟨output, ?_⟩
  change matrixNorm
      (left.error * preimageLift.witness + messageLift • right.error -
        messageLift • (leftMaskMagnitude.lift * targetApprox.error)) ≤ _
  exact Mxx.Bgg.multiplication_error_bound left right targetApprox leftMaskMagnitude
    preimageLift messageLift leftErrorBound rightErrorBound targetErrorBound messageBound
    left_error_le right_error_le target_error_le message_le hn

theorem radix_step
    {q n : Nat} (system : RadixSystem q n) (value : ExactPoly q n) :
    value = ∑ limb : system.Limb,
      reducePoly q n (system.weight limb * system.digit value limb) :=
  radix_reconstruct system value

theorem approximation_error_coefficient_le
    {q n bound : Nat} (hn : 0 < n)
    {actual ideal : ExactMatrix q n 1 1}
    (approximation : ApproxWithin actual ideal bound) :
    ((approximation.error 0 0).coeff ⟨0, hn⟩).natAbs ≤ bound := by
  calc
    ((approximation.error 0 0).coeff ⟨0, hn⟩).natAbs ≤
        polyNorm (approximation.error 0 0) :=
      coeff_natAbs_le_polyNorm _ _
    _ ≤ matrixNorm approximation.error := by
      exact (Finset.le_sup (s := Finset.univ)
        (f := fun column => polyNorm (approximation.error 0 column))
        (Finset.mem_univ 0)).trans
        (Finset.le_sup (s := Finset.univ)
          (f := fun row => Finset.univ.sup
            (fun column => polyNorm (approximation.error row column)))
          (Finset.mem_univ 0))
    _ ≤ bound := approximation.norm_le

/- The exact coefficient-zero bridge consumes the same integer witness carried by
   `ApproxWithin`; `coefficient` is the canonical runtime residue represented as an `Int`. -/
theorem decode_interval_of_approximation
    (parameters : Parameters) (message : Bool)
    {bound : Nat}
    {actual : ExactMatrix parameters.modulus parameters.ringDimension 1 1}
    (approximation : ApproxWithin actual (idealPlaintext parameters message) bound)
    (coefficient : Int)
    (coefficient_eq : coefficient =
      ((actual 0 0).coeff ⟨0, parameters.valid.2.1⟩).val) :
    bound < decoderNoiseThreshold parameters.modulus →
      decodeInterval parameters.modulus coefficient = message := by
  intro hnoise
  let q := parameters.modulus
  let n := parameters.ringDimension
  letI : Fact (1 < q) := ⟨parameters.valid.1⟩
  let i : Fin n := ⟨0, parameters.valid.2.1⟩
  let error : Int := (approximation.error 0 0).coeff i
  have hcoeff : error.natAbs ≤ bound := by
    dsimp [error]
    exact (coeff_natAbs_le_polyNorm _ _).trans
      ((Finset.le_sup (s := Finset.univ)
        (f := fun c => polyNorm (approximation.error 0 c))
        (Finset.mem_univ 0)).trans
        ((Finset.le_sup (s := Finset.univ)
          (f := fun r => Finset.sup Finset.univ
            (fun c => polyNorm (approximation.error r c)))
          (Finset.mem_univ 0)).trans approximation.norm_le))
  have error_lower : -(bound : Int) ≤ error := by
    have hcoeff' : (error.natAbs : Int) ≤ bound := by exact_mod_cast hcoeff
    by_cases hnonneg : 0 ≤ error
    · rw [Int.natAbs_of_nonneg hnonneg] at hcoeff'
      omega
    · have hnonpos : error ≤ 0 := le_of_not_ge hnonneg
      rw [Int.ofNat_natAbs_of_nonpos hnonpos] at hcoeff'
      omega
  have error_upper : error ≤ bound := by
    have hcoeff' : (error.natAbs : Int) ≤ bound := by exact_mod_cast hcoeff
    by_cases hnonneg : 0 ≤ error
    · rw [Int.natAbs_of_nonneg hnonneg] at hcoeff'
      omega
    · have hnonpos : error ≤ 0 := le_of_not_ge hnonneg
      rw [Int.ofNat_natAbs_of_nonpos hnonpos] at hcoeff'
      omega
  have heq := congrArg (fun value : ExactMatrix parameters.modulus
      parameters.ringDimension 1 1 =>
      (value 0 0).coeff i) approximation.equation
  have hactual :
      (actual 0 0).coeff i =
        (idealPlaintext parameters message 0 0).coeff i + (error : ZMod q) := by
    dsimp [error] at heq ⊢
    have heq' : (actual 0 0).coeff i =
        (idealPlaintext parameters message 0 0 +
          reduceMatrix parameters.modulus parameters.ringDimension 1 1
            approximation.toApprox.error 0 0).coeff i := heq
    rw [Negacyclic.coeff_add, reduceMatrix_apply,
      reducePoly_coeff parameters.valid.1 parameters.valid.2.1] at heq'
    exact heq'
  have hideal :
      (idealPlaintext parameters message 0 0).coeff i =
        ((parameters.modulus / 2 * messageNat message : Nat) : Int) := by
    change (algebraMap (ZMod parameters.modulus)
      (ExactPoly parameters.modulus parameters.ringDimension)
      ((parameters.modulus / 2 * messageNat message : Nat) : Int)).coeff i = _
    rw [← mul_one (algebraMap (ZMod parameters.modulus)
      (ExactPoly parameters.modulus parameters.ringDimension)
      ((parameters.modulus / 2 * messageNat message : Nat) : Int))]
    rw [Negacyclic.coeff_smul]
    have hcoeff_one :
        (1 : ExactPoly parameters.modulus parameters.ringDimension).coeff i = 1 := by
      have hroot := Negacyclic.coeff_root_pow (R := ZMod parameters.modulus)
        parameters.valid.2.1 (⟨0, parameters.valid.2.1⟩ : Fin parameters.ringDimension)
        (⟨0, parameters.valid.2.1⟩ : Fin parameters.ringDimension)
      simpa [i] using hroot
    rw [hcoeff_one]
    simp
  have geometry : DecoderGeometryValid q := ParametersData.geometryValid parameters.valid
  have safe := decoder_safe_of_lt_threshold geometry hnoise
  have hq : 1 < q := by simpa [q] using parameters.valid.1
  have hqpos : 0 < q := lt_trans Nat.zero_lt_one hq
  have hbound_lt_q : bound < q := by
    exact lt_of_lt_of_le hnoise
      ((decoder_threshold_le_half geometry).trans (Nat.div_le_self q 2))
  have hbound_lt_q' : (bound : Int) < q := by exact_mod_cast hbound_lt_q
  cases message with
  | false =>
      have hactual' : (actual 0 0).coeff i = (error : ZMod q) := by
        simpa [hideal, messageNat] using hactual
      by_cases hnonneg : 0 ≤ error
      · have herrq : error < (q : Int) := by
          have hhalf := decoder_threshold_le_half geometry
          have hnoise_nat : bound < q / 2 := lt_of_lt_of_le hnoise hhalf
          have hnoise' : (bound : Int) < ((q / 2 : Nat) : Int) := by exact_mod_cast hnoise_nat
          have hhalf_le : ((q / 2 : Nat) : Int) ≤ q := by
            exact_mod_cast (Nat.div_le_self q 2)
          exact lt_of_le_of_lt error_upper hbound_lt_q'
        have hval := val_of_intCast_of_nonneg_lt hqpos hnonneg herrq
        apply decode_interval_of_centered_error geometry hnoise false error coefficient
          error_lower error_upper
        left
        constructor
        · exact hnonneg
        · calc
            coefficient = ((actual 0 0).coeff i).val := coefficient_eq
            _ = error := by rw [hactual']; exact hval
      · have hneg : error < 0 := lt_of_not_ge hnonneg
        have herrq : -(q : Int) < error := by
          have hhalf := decoder_threshold_le_half geometry
          have hnoise_nat : bound < q / 2 := lt_of_lt_of_le hnoise hhalf
          have hnoise' : (bound : Int) < ((q / 2 : Nat) : Int) := by exact_mod_cast hnoise_nat
          have hhalf_le : ((q / 2 : Nat) : Int) ≤ q := by
            exact_mod_cast (Nat.div_le_self q 2)
          omega
        have hval := val_of_intCast_of_neg hqpos herrq hneg
        apply decode_interval_of_centered_error geometry hnoise false error coefficient
          error_lower error_upper
        right
        constructor
        · exact hneg
        · calc
            coefficient = ((actual 0 0).coeff i).val := coefficient_eq
            _ = q + error := by rw [hactual']; exact hval
  | true =>
      have hactual' : (actual 0 0).coeff i =
          ((parameters.modulus / 2 : Nat) : Int) + error := by
        rw [hactual, hideal]
        simp [messageNat, Nat.cast_mul, Int.cast_add]
      have half_lower : q / 2 ≥ decoderQuarter q + bound := safe.2.2.1
      have modulus_upper : q - (3 * decoderQuarter q + bound) > 0 := safe.2.1
      have half_le_three : q / 2 ≤ 3 * decoderQuarter q := geometry.half_le_three_quarters
      have q_upper : 3 * decoderQuarter q + bound < q := by omega
      have z_nonneg : 0 ≤ ((parameters.modulus / 2 : Nat) : Int) + error := by
        have hnat : decoderQuarter q + bound ≤ q / 2 := half_lower
        have hnat' : (decoderQuarter q + bound : Int) ≤ (q / 2 : Nat) := by exact_mod_cast hnat
        omega
      have z_lt_q : ((parameters.modulus / 2 : Nat) : Int) + error < q := by
        have hnat' : 3 * decoderQuarter q + bound < q := q_upper
        have hnat'' : (3 * decoderQuarter q + bound : Int) < q := by exact_mod_cast hnat'
        have hhalf' : ((q / 2 : Nat) : Int) ≤ 3 * decoderQuarter q := by
          exact_mod_cast half_le_three
        omega
      have hval := val_of_intCast_of_nonneg_lt hqpos z_nonneg z_lt_q
      apply decode_interval_of_centered_error geometry hnoise true error coefficient
        error_lower error_upper
      calc
        coefficient = ((actual 0 0).coeff i).val := coefficient_eq
        _ = ((parameters.modulus / 2 : Nat) : Int) + error := by
          rw [hactual']; simpa [Int.cast_add] using hval
        _ = q / 2 + error := by
          change ((q / 2 : Nat) : Int) + error = q / 2 + error
          simp [q, Int.natCast_div]

end Mxx.We.DiamondWE
