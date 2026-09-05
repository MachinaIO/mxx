import ExponentBounds
import MxxBgg.Multiplication

namespace MxxExponentLut

open Mxx.Primitives
open scoped BigOperators

/-! Integer error witnesses for the four operations interpreted in `src/noise.rs`.
The matrix actions are the actual integer lifts of the runtime operands. Their
column gains must be established from the concrete decomposition; no noise
invariant or simulator output is assumed as an action-gain premise. -/

theorem private_multiplication_bound {n rows inner columns inputBound helperBound gain : Nat}
    (hn : 0 < n) {input : ErrorMatrix n rows inner}
    {action : ErrorMatrix n inner columns} {helper : ErrorMatrix n rows columns}
    (inputBounded : CoeffBound input inputBound)
    (actionBounded : ActionGain action gain)
    (helperBounded : CoeffBound helper helperBound) :
    CoeffBound (input * action + helper) (gain * inputBound + helperBound) :=
  coeffBound_add (right_action_bound hn inputBounded actionBounded) helperBounded

theorem public_multiplication_bound {n rows inner columns privateBound publicBound gain : Nat}
    (hn : 0 < n) {privateError : ErrorMatrix n rows columns}
    {publicError : ErrorMatrix n rows inner} {action : ErrorMatrix n inner columns}
    (privateBounded : CoeffBound privateError privateBound)
    (publicBounded : CoeffBound publicError publicBound)
    (actionBounded : ActionGain action gain) :
    CoeffBound (privateError + publicError * action) (privateBound + gain * publicBound) :=
  coeffBound_add privateBounded (right_action_bound hn publicBounded actionBounded)

theorem addition_bound {n rows columns leftBound rightBound : Nat}
    {left right : ErrorMatrix n rows columns}
    (leftBounded : CoeffBound left leftBound) (rightBounded : CoeffBound right rightBound) :
    CoeffBound (left + right) (leftBound + rightBound) :=
  coeffBound_add leftBounded rightBounded

theorem lut_branch_bound
    {n rows inner columns inputBound helperBound directGain helperGain : Nat}
    (hn : 0 < n) {input helperInput : ErrorMatrix n rows inner}
    {directAction helperAction : ErrorMatrix n inner columns}
    {helper : ErrorMatrix n rows columns}
    (inputBounded : CoeffBound input inputBound)
    (helperInputBounded : CoeffBound helperInput helperBound)
    (helperBounded : CoeffBound helper helperBound)
    (directBounded : ActionGain directAction directGain)
    (helperActionBounded : ActionGain helperAction helperGain) :
    CoeffBound (input * directAction + (helper + helperInput * helperAction))
      (directGain * inputBound + helperBound * (1 + helperGain)) := by
  have h := coeffBound_add (right_action_bound hn inputBounded directBounded)
    (coeffBound_add helperBounded (right_action_bound hn helperInputBounded helperActionBounded))
  convert h using 1
  ring

/-- Branch aggregation and structural family reduction need no independence assumption. -/
theorem sum_bound {n rows columns : Nat} {α : Type} [DecidableEq α]
    (members : Finset α) (error : α → ErrorMatrix n rows columns) (bound : α → Nat)
    (bounded : ∀ member ∈ members, CoeffBound (error member) (bound member)) :
    CoeffBound (∑ member ∈ members, error member) (∑ member ∈ members, bound member) := by
  induction members using Finset.induction_on with
  | empty => simpa using (coeffBound_zero (n := n) (rows := rows) (columns := columns))
  | @insert member members hnot ih =>
    simp only [Finset.sum_insert hnot]
    exact coeffBound_add (bounded member (Finset.mem_insert_self _ _))
      (ih (fun i hi => bounded i (Finset.mem_insert_of_mem hi)))

/-- A bounded action preserves a modular approximation through its actual integer lift. -/
theorem approximate_right_action {q n rows inner columns bound gain : Nat} (hn : 0 < n)
    {actual ideal : ExactMatrix q n rows inner} {action : ErrorMatrix n inner columns}
    (close : Approx actual ideal bound) (bounded : ActionGain action gain) :
    Approx (actual * reduceMatrix q n inner columns action)
      (ideal * reduceMatrix q n inner columns action) (gain * bound) := by
  obtain ⟨error, equation, errorBound⟩ := close
  refine ⟨error * action, ?_, right_action_bound hn errorBound bounded⟩
  rw [equation, Matrix.add_mul, reduceMatrix_mul]

/-- Private RHS multiplication consumes both the cached decomposition and its
GSW error equation. The same integer action controls inherited error; the
monomial message acts on the fresh helper error with unit coefficient gain. -/
theorem private_encoding_transfer {q n secret columns inputBound helperBound gain : Nat}
    (hn : 0 < n)
    {ciphertext : ExactMatrix q n 1 columns}
    {mask : ExactMatrix q n 1 secret} {publicKey : ExactMatrix q n secret columns}
    {payloadGadget : ExactMatrix q n 1 columns}
    {message rhsMessage : ExactPoly q n} {messageInteger : ErrorPoly n}
    {inputError helperError : ErrorMatrix n 1 columns}
    {action : ErrorMatrix n columns columns}
    (encodingEquation : ciphertext = mask * publicKey - message • payloadGadget +
      reduceMatrix q n 1 columns inputError)
    (actionEquation : payloadGadget * reduceMatrix q n columns columns action =
      rhsMessage • payloadGadget - reduceMatrix q n 1 columns helperError)
    (messageLift : message = reducePoly q n messageInteger)
    (messageBound : polynomialL1 messageInteger ≤ 1)
    (inputBounded : CoeffBound inputError inputBound)
    (helperBounded : CoeffBound helperError helperBound)
    (actionBounded : ActionGain action gain) :
    Approx (ciphertext * reduceMatrix q n columns columns action)
      (mask * (publicKey * reduceMatrix q n columns columns action) -
        (message * rhsMessage) • payloadGadget) (gain * inputBound + helperBound) := by
  refine ⟨inputError * action + messageInteger • helperError, ?_, ?_⟩
  · rw [encodingEquation, Matrix.add_mul, Matrix.sub_mul, Matrix.smul_mul,
      Matrix.mul_assoc, actionEquation, smul_sub, smul_smul,
      reduceMatrix_add, reduceMatrix_mul, reduce_polynomial_action, ← messageLift]
    abel
  · have hm := left_polynomial_bound hn helperBounded messageBound
    simp only [one_mul] at hm
    exact coeffBound_add (right_action_bound hn inputBounded actionBounded) hm

/-- The fixed-RHS sampler uses a positive Gaussian error. This form consumes
its cache equation directly, including the sign of the propagated fresh error. -/
theorem private_encoding_transfer_sampled {q n secret columns inputBound helperBound gain : Nat}
    (hn : 0 < n)
    {ciphertext : ExactMatrix q n 1 columns}
    {mask : ExactMatrix q n 1 secret} {publicKey : ExactMatrix q n secret columns}
    {payloadGadget : ExactMatrix q n 1 columns}
    {message rhsMessage : ExactPoly q n} {messageInteger : ErrorPoly n}
    {inputError sampledError : ErrorMatrix n 1 columns}
    {action : ErrorMatrix n columns columns}
    (encodingEquation : ciphertext = mask * publicKey - message • payloadGadget +
      reduceMatrix q n 1 columns inputError)
    (actionEquation : payloadGadget * reduceMatrix q n columns columns action =
      rhsMessage • payloadGadget + reduceMatrix q n 1 columns sampledError)
    (messageLift : message = reducePoly q n messageInteger)
    (messageBound : polynomialL1 messageInteger ≤ 1)
    (inputBounded : CoeffBound inputError inputBound)
    (sampledBounded : CoeffBound sampledError helperBound)
    (actionBounded : ActionGain action gain) :
    Approx (ciphertext * reduceMatrix q n columns columns action)
      (mask * (publicKey * reduceMatrix q n columns columns action) -
        (message * rhsMessage) • payloadGadget) (gain * inputBound + helperBound) := by
  have negBound : CoeffBound (-sampledError) helperBound := by
    intro row column coefficient
    simpa using sampledBounded row column coefficient
  apply private_encoding_transfer hn encodingEquation (helperError := -sampledError)
    ?_ messageLift messageBound inputBounded negBound actionBounded
  rw [actionEquation]
  have hn : reduceMatrix q n 1 columns (-sampledError) =
      -reduceMatrix q n 1 columns sampledError := by
    ext row column
    exact map_neg (reducePoly q n) _
  rw [hn, sub_neg_eq_add]

/-- Public multiplication uses the public monomial on the private error and
the private decomposition on the independently sampled public-label error. -/
theorem public_encoding_transfer {q n secret columns privateBound publicBound gain : Nat}
    (hn : 0 < n)
    {publicCiphertext privateCiphertext : ExactMatrix q n 1 columns}
    {mask commonSecret payload : ExactMatrix q n 1 secret}
    {publicKey privateKey gadget : ExactMatrix q n secret columns}
    {publicMessage privateMessage : ExactPoly q n} {messageInteger : ErrorPoly n}
    {publicError privateError : ErrorMatrix n 1 columns}
    {action : ErrorMatrix n columns columns}
    (publicEquation : publicCiphertext = mask * publicKey -
      publicMessage • (commonSecret * gadget) + reduceMatrix q n 1 columns publicError)
    (privateEquation : privateCiphertext = commonSecret * privateKey -
      privateMessage • (payload * gadget) + reduceMatrix q n 1 columns privateError)
    (decomposition : gadget * reduceMatrix q n columns columns action = privateKey)
    (messageLift : publicMessage = reducePoly q n messageInteger)
    (messageBound : polynomialL1 messageInteger ≤ 1)
    (publicBounded : CoeffBound publicError publicBound)
    (privateBounded : CoeffBound privateError privateBound)
    (actionBounded : ActionGain action gain) :
    Approx (publicCiphertext * reduceMatrix q n columns columns action +
      publicMessage • privateCiphertext)
      (mask * (publicKey * reduceMatrix q n columns columns action) -
        (publicMessage * privateMessage) • (payload * gadget))
      (privateBound + gain * publicBound) := by
  refine ⟨publicError * action + messageInteger • privateError, ?_, ?_⟩
  · rw [Mxx.Bgg.multiplication_core publicEquation privateEquation decomposition,
      reduceMatrix_add, reduceMatrix_mul, reduce_polynomial_action, ← messageLift]
  · have hm := left_polynomial_bound hn privateBounded messageBound
    simp only [one_mul] at hm
    simpa only [Nat.add_comm] using
      coeffBound_add (right_action_bound hn publicBounded actionBounded) hm

/-- Structural aggregation retains one integer error witness per actual member.
No stochastic independence is needed, including for broadcast family inputs. -/
theorem family_approximate_sum {q n rows columns : Nat} {α : Type} [DecidableEq α]
    (members : Finset α) (actual ideal : α → ExactMatrix q n rows columns)
    (bound : α → Nat)
    (close : ∀ i ∈ members, Approx (actual i) (ideal i) (bound i)) :
    Approx (∑ i ∈ members, actual i) (∑ i ∈ members, ideal i)
      (∑ i ∈ members, bound i) := by
  induction members using Finset.induction_on with
  | empty =>
    simp only [Finset.sum_empty]
    exact ⟨0, by ext i j; simp [reduceMatrix, reducePoly], coeffBound_zero⟩
  | @insert i members hnot ih =>
    obtain ⟨e, he, hb⟩ := close i (Finset.mem_insert_self _ _)
    obtain ⟨es, hes, hbs⟩ := ih (fun j hj => close j (Finset.mem_insert_of_mem hj))
    simp only [Finset.sum_insert hnot]
    refine ⟨e + es, ?_, coeffBound_add hb hbs⟩
    rw [he, hes, reduceMatrix_add]
    abel

end MxxExponentLut
