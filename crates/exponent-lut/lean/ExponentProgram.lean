import ExponentTransfers
import ExponentReadout
import ExponentLut

namespace MxxExponentLut
open Mxx.Primitives
open scoped BigOperators

/-- Application proof state for one actual encoding wire. -/
structure ProgramState (q n secret columns : Nat) where
  ciphertext : ExactMatrix q n 1 columns
  publicKey : ExactMatrix q n secret columns
  message : ExactPoly q n
  mask : ExactMatrix q n 1 secret
  payload : ExactMatrix q n 1 secret

def ProgramState.Invariant {q n secret columns : Nat}
    (gadget : ExactMatrix q n secret columns) (state : ProgramState q n secret columns)
    (bound : Nat) : Prop :=
  Approx state.ciphertext (state.mask * state.publicKey -
    state.message • (state.payload * gadget)) bound

/-- The optimized BGG identity LUT with an encrypted RHS. This proof node
models the single branch action; it is not a separate Program opcode. -/
noncomputable def ProgramState.identityLutOutput {q n secret columns : Nat}
    (state : ProgramState q n secret columns) (action : ErrorMatrix n columns columns)
    (rhs : ExactPoly q n) : ProgramState q n secret columns where
  ciphertext := state.ciphertext * reduceMatrix q n columns columns action
  publicKey := state.publicKey * reduceMatrix q n columns columns action
  message := state.message * rhs
  mask := state.mask
  payload := state.payload

noncomputable def ProgramState.publicOutput {q n secret columns : Nat}
    (privateState publicState : ProgramState q n secret columns)
    (action : ErrorMatrix n columns columns) : ProgramState q n secret columns where
  ciphertext := publicState.ciphertext * reduceMatrix q n columns columns action +
    publicState.message • privateState.ciphertext
  publicKey := publicState.publicKey * reduceMatrix q n columns columns action
  message := publicState.message * privateState.message
  mask := publicState.mask
  payload := privateState.payload

noncomputable def ProgramState.addOutput {q n secret columns : Nat}
    (left right : ProgramState q n secret columns) : ProgramState q n secret columns where
  ciphertext := left.ciphertext + right.ciphertext
  publicKey := left.publicKey + right.publicKey
  message := left.message + right.message
  mask := left.mask
  payload := left.payload

noncomputable def ProgramState.familyOutput {q n secret columns count : Nat}
    (states : Fin count → ProgramState q n secret columns)
    (mask payload : ExactMatrix q n 1 secret) : ProgramState q n secret columns where
  ciphertext := ∑ i, (states i).ciphertext
  publicKey := ∑ i, (states i).publicKey
  message := ∑ i, (states i).message
  mask := mask
  payload := payload

noncomputable def ProgramState.lutOutput {q n secret columns count : Nat}
    (state : ProgramState q n secret columns)
    (index : Fin count → Nat) (hodd : ∀ i, Odd (index i))
    (raw maskCipher : Fin count → ExactMatrix q n 1 columns)
    (maskKey : Fin count → ExactMatrix q n secret columns)
    (action helperAction : Fin count → ErrorMatrix n columns columns)
    (coefficient : Fin count → ExactPoly q n) : ProgramState q n secret columns where
  ciphertext := ∑ i, (maskCipher i * reduceMatrix q n columns columns (helperAction i) +
    raw i * reduceMatrix q n columns columns (action i))
  publicKey := ∑ i, maskKey i * reduceMatrix q n columns columns (helperAction i)
  message := ∑ i, readoutAutomorphismHom (index i) (hodd i) state.message * coefficient i
  mask := state.mask
  payload := state.payload

/-- The existing LUT derivation accepts fused coefficients directly. Its
message is the table's linear map on the input/RHS product; no private-step
proof or intermediate ciphertext is required. -/
theorem ProgramState.fusedLut_message {q n secret columns count : Nat}
    (state : ProgramState q n secret columns)
    (index : Fin count → Nat) (hodd : ∀ i, Odd (index i))
    (raw maskCipher : Fin count → ExactMatrix q n 1 columns)
    (maskKey : Fin count → ExactMatrix q n secret columns)
    (action helperAction : Fin count → ErrorMatrix n columns columns)
    (coefficient : Fin count → ExactPoly q n) (rhs : ExactPoly q n) :
    (state.lutOutput index hodd raw maskCipher maskKey action helperAction
      (fun i => coefficient i * readoutAutomorphismHom (index i) (hodd i) rhs)).message =
    ∑ i, readoutAutomorphismHom (index i) (hodd i) (state.message * rhs) * coefficient i := by
  exact lut_fused_message index hodd state.message rhs coefficient

/-- Proof structure consumed by extracted graph facts, not another evaluator.
Each node contains primitive equations and recursive input proofs. The bound
indices are the exact worst-case recurrences in `src/noise.rs`; no constructor
assumes the output invariant. -/
inductive ProgramDerivation {q n secret columns : Nat}
    (gadget : ExactMatrix q n secret columns) : ProgramState q n secret columns → Nat → Prop
  | input (state : ProgramState q n secret columns) (error : ErrorMatrix n 1 columns) (bound : Nat)
      (equation : state.ciphertext = state.mask * state.publicKey -
        state.message • (state.payload * gadget) + reduceMatrix q n 1 columns error)
      (bounded : CoeffBound error bound) : ProgramDerivation gadget state bound
  | identityLutStep {state : ProgramState q n secret columns} {bound : Nat}
      (previous : ProgramDerivation gadget state bound)
      (action : ErrorMatrix n columns columns) (rhs : ExactPoly q n)
      (sampledError : ErrorMatrix n 1 columns) (messageLift : ErrorPoly n) (gain cutoff : Nat)
      (cache : (state.payload * gadget) * reduceMatrix q n columns columns action =
        rhs • (state.payload * gadget) + reduceMatrix q n 1 columns sampledError)
      (lift : state.message = reducePoly q n messageLift)
      (monomial : polynomialL1 messageLift ≤ 1)
      (sampled : CoeffBound sampledError cutoff) (actionBound : ActionGain action gain) :
      ProgramDerivation gadget (state.identityLutOutput action rhs) (gain * bound + cutoff)
  | publicStep {privateState publicState : ProgramState q n secret columns}
      {privateBound publicBound : Nat}
      (privatePrevious : ProgramDerivation gadget privateState privateBound)
      (publicPrevious : ProgramDerivation gadget publicState publicBound)
      (action : ErrorMatrix n columns columns) (messageLift : ErrorPoly n) (gain : Nat)
      (shared : publicState.payload = privateState.mask)
      (decomposition : gadget * reduceMatrix q n columns columns action = privateState.publicKey)
      (lift : publicState.message = reducePoly q n messageLift)
      (monomial : polynomialL1 messageLift ≤ 1) (actionBound : ActionGain action gain) :
      ProgramDerivation gadget (privateState.publicOutput publicState action)
        (privateBound + gain * publicBound)
  | addStep {left right : ProgramState q n secret columns} {leftBound rightBound : Nat}
      (leftPrevious : ProgramDerivation gadget left leftBound)
      (rightPrevious : ProgramDerivation gadget right rightBound)
      (sharedMask : right.mask = left.mask) (sharedPayload : right.payload = left.payload) :
      ProgramDerivation gadget (left.addOutput right) (leftBound + rightBound)
  | lutStep {state : ProgramState q n secret columns} {bound log count : Nat}
      (previous : ProgramDerivation gadget state bound)
      (hq : 1 < q) (hdimension : n = 2 ^ log)
      (index : Fin count → Nat) (hodd : ∀ i, Odd (index i))
      (raw maskCipher : Fin count → ExactMatrix q n 1 columns)
      (maskKey : Fin count → ExactMatrix q n secret columns)
      (action helperAction : Fin count → ErrorMatrix n columns columns)
      (switchError maskError : Fin count → ErrorMatrix n 1 columns)
      (coefficient : Fin count → ExactPoly q n)
      (cutoff : Nat) (actionGain helperGain : Fin count → Nat)
      (hrun : ∀ i, MxxRuntime.ringAutomorphismRuns (index i : Int) state.ciphertext (raw i))
      (hmessage : SignedMonomial state.message)
      (hgadget : ∀ i, MxxRuntime.ringAutomorphism (index i) gadget = gadget)
      (hswitch : ∀ i, MxxRuntime.ringAutomorphism (index i) state.payload * gadget *
          reduceMatrix q n columns columns (action i) =
        coefficient i • (state.payload * gadget) + reduceMatrix q n 1 columns (switchError i))
      (halign : ∀ i, gadget * reduceMatrix q n columns columns (helperAction i) =
        MxxRuntime.ringAutomorphism (index i) state.publicKey * reduceMatrix q n columns columns (action i))
      (hmask : ∀ i, maskCipher i = state.mask * maskKey i -
        MxxRuntime.ringAutomorphism (index i) state.mask * gadget +
          reduceMatrix q n 1 columns (maskError i))
      (hswitchBound : ∀ i, CoeffBound (switchError i) cutoff)
      (hmaskBound : ∀ i, CoeffBound (maskError i) cutoff)
      (haction : ∀ i, ActionGain (action i) (actionGain i))
      (hhelper : ∀ i, ActionGain (helperAction i) (helperGain i)) :
      ProgramDerivation gadget
        (state.lutOutput index hodd raw maskCipher maskKey action helperAction coefficient)
        (lutNoiseBound bound cutoff actionGain helperGain)
  | familyReduce {count : Nat} (states : Fin count → ProgramState q n secret columns)
      (bounds : Fin count → Nat) (mask payload : ExactMatrix q n 1 secret)
      (previous : ∀ i, ProgramDerivation gadget (states i) (bounds i))
      (sharedMask : ∀ i, (states i).mask = mask)
      (sharedPayload : ∀ i, (states i).payload = payload) :
      ProgramDerivation gadget (ProgramState.familyOutput states mask payload) (∑ i, bounds i)

/-- Compose an identity LUT with a public multiplication using the actual scalar
execution equation. Cached actions and both error contributions remain explicit. -/
theorem program_private_public_transfer {q n secret columns : Nat}
    {gadget : ExactMatrix q n secret columns}
    {state publicState : ProgramState q n secret columns}
    {bound publicBound privateGain publicGain cutoff : Nat}
    (previous : ProgramDerivation gadget state bound)
    (publicPrevious : ProgramDerivation gadget publicState publicBound)
    (privateAction publicAction : ErrorMatrix n columns columns)
    (rhs : ExactPoly q n) (sampledError : ErrorMatrix n 1 columns)
    (messageLift publicLift : ErrorPoly n)
    (cache : (state.payload * gadget) * reduceMatrix q n columns columns privateAction =
      rhs • (state.payload * gadget) + reduceMatrix q n 1 columns sampledError)
    (lift : state.message = reducePoly q n messageLift)
    (monomial : polynomialL1 messageLift ≤ 1)
    (sampleBound : CoeffBound sampledError cutoff)
    (privateActionBound : ActionGain privateAction privateGain)
    (shared : publicState.payload = state.mask)
    (decomposition : gadget * reduceMatrix q n columns columns publicAction =
      state.publicKey * reduceMatrix q n columns columns privateAction)
    (publicMessageLift : publicState.message = reducePoly q n publicLift)
    (publicMonomial : polynomialL1 publicLift ≤ 1)
    (publicActionBound : ActionGain publicAction publicGain)
    {actual current publicCipher : ExactMatrix q n 1 columns}
    {privateCache publicCache : ExactMatrix q n columns columns}
    {publicMessage : ExactPoly q n}
    (currentEquation : state.ciphertext = current)
    (publicCipherEquation : publicState.ciphertext = publicCipher)
    (publicMessageEquation : publicState.message = publicMessage)
    (privateCacheEquation : privateCache = reduceMatrix q n columns columns privateAction)
    (publicCacheEquation : publicCache = reduceMatrix q n columns columns publicAction)
    (actualEquation : actual = publicCipher * publicCache +
      publicMessage • (current * privateCache)) :
    ∃ result : ProgramState q n secret columns,
      ProgramDerivation gadget result (privateGain * bound + cutoff + publicGain * publicBound) ∧
      result.ciphertext = actual ∧
      result.publicKey = publicState.publicKey * reduceMatrix q n columns columns publicAction ∧
      result.message = publicState.message * (state.message * rhs) ∧
      result.mask = publicState.mask ∧ result.payload = state.payload := by
  have privateDerived := ProgramDerivation.identityLutStep previous privateAction rhs sampledError
    messageLift privateGain cutoff cache lift monomial sampleBound privateActionBound
  have derived := ProgramDerivation.publicStep privateDerived publicPrevious publicAction publicLift
    publicGain shared decomposition publicMessageLift publicMonomial publicActionBound
  refine ⟨_, derived, ?_, rfl, rfl, rfl, rfl⟩
  dsimp [ProgramState.identityLutOutput, ProgramState.publicOutput]
  rw [currentEquation, publicCipherEquation, publicMessageEquation,
    ← privateCacheEquation, ← publicCacheEquation]
  exact actualEquation.symm

/-- Structural induction over actual primitive equations propagates the
encoding invariant through scalar gates and every member of finite families. -/
theorem ProgramDerivation.sound {q n secret columns : Nat} (hn : 0 < n)
    {gadget : ExactMatrix q n secret columns} {state : ProgramState q n secret columns}
    {bound : Nat} (run : ProgramDerivation gadget state bound) :
    state.Invariant gadget bound := by
  induction run with
  | input state error bound equation bounded => exact ⟨error, equation, bounded⟩
  | identityLutStep previous action rhs sampledError messageLift gain cutoff cache lift monomial sampled actionBound ih =>
    obtain ⟨error, equation, bounded⟩ := ih
    exact private_encoding_transfer_sampled hn equation cache lift monomial bounded sampled actionBound
  | publicStep privatePrevious publicPrevious action messageLift gain shared decomposition lift monomial actionBound ihp ihu =>
    obtain ⟨privateError, privateEquation, privateBounded⟩ := ihp
    obtain ⟨publicError, publicEquation, publicBounded⟩ := ihu
    rw [shared] at publicEquation
    exact public_encoding_transfer hn publicEquation privateEquation decomposition lift monomial
      publicBounded privateBounded actionBound
  | addStep leftPrevious rightPrevious sharedMask sharedPayload ihl ihr =>
    obtain ⟨el, hel, hbl⟩ := ihl
    obtain ⟨er, her, hbr⟩ := ihr
    refine ⟨el + er, ?_, coeffBound_add hbl hbr⟩
    dsimp [ProgramState.addOutput]
    rw [hel, her, sharedMask, sharedPayload, reduceMatrix_add, Matrix.mul_add, add_smul]
    abel
  | lutStep previous hq hdimension index hodd raw maskCipher maskKey action helperAction
      switchError maskError coefficient cutoff actionGain helperGain hrun hmessage hgadget
      hswitch halign hmask hswitchBound hmaskBound haction hhelper ih =>
    exact lut_encoding_transfer hq hdimension index hodd raw maskCipher _ _ _ gadget
      maskKey action helperAction switchError maskError _ coefficient actionGain helperGain ih
      hrun hmessage hgadget hswitch halign hmask hswitchBound hmaskBound haction hhelper
  | familyReduce states bounds mask payload previous sharedMask sharedPayload ih =>
    have h := approximate_sum Finset.univ (fun i => (states i).ciphertext)
      (fun i => mask * (states i).publicKey - (states i).message • (payload * gadget))
      bounds (fun i _ => by simpa [ProgramState.Invariant, sharedMask i, sharedPayload i] using ih i)
    simpa only [ProgramState.Invariant, ProgramState.familyOutput, Finset.sum_sub_distrib,
      Matrix.mul_sum, Finset.sum_smul] using h

/-- Map and zip-map preserve the exact index set; each member is proved by the
same instruction relation before a subsequent family reduction. -/
theorem program_family_map_sound {q n secret columns count : Nat} (hn : 0 < n)
    {gadget : ExactMatrix q n secret columns} {states : Fin count → ProgramState q n secret columns}
    {bounds : Fin count → Nat} (runs : ∀ i, ProgramDerivation gadget (states i) (bounds i)) :
    ∀ i, (states i).Invariant gadget (bounds i) :=
  fun i => (runs i).sound hn

end MxxExponentLut
