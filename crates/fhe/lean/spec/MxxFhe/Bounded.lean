import MxxFhe.Defs

/-!
# IND-CPA security with reductions of bounded cost

`Secure` names the simulators of its reductions, and a reader checks that they are efficient.
`SecureWithin` instead asks only that, for each allowed assumption it uses, a simulator exists
that is a pair of IR programs (`MxxSecurity.IR.Program`) of total cost at most `bound`, run in an
evaluation context whose backend the statement fixes and whose hash function is arbitrary. A program
can compute only what the operations of the IR compute, at the cost `MxxSecurity.IR.Program.cost`
charges them, so the statement itself bounds the work of each reduction beyond running the
adversary once.

The programs exchange values with the scheme through a `Codec`, which converts by type and
computes nothing.
-/

namespace MxxFhe

open MxxSecurity MxxSecurity.IR

/-- The conversions between the values of a scheme and IR values. -/
structure Codec (S : Scheme) where
  msg : S.Msg → Value
  pk : Value → Option S.PK
  ek : Value → Option S.EK
  ct : Value → Option S.CT

/-- A simulator as two IR programs. `setup` reads the named challenge components of an
assumption and returns `public_key`, `evaluation_key`, and any state, by name; `encrypt` reads
`message` and any setup outputs by name and returns `ciphertext`. -/
structure ProgramSimulator where
  setup : Program
  encrypt : Program

/-- The total cost of a simulator's programs. -/
def ProgramSimulator.cost (sim : ProgramSimulator) : Option Nat := do
  pure ((← sim.setup.cost) + (← sim.encrypt.cost))

/-- The reduction around a program simulator, with its programs run in `ctx`: it runs the
adversary once on the simulated view, with the challenge encrypting its right (`right = true`)
or left message, and returns its guess, or `false` if a program fails. -/
noncomputable def reduceProgram {S : Scheme} {Challenge : Type} (ctx : Context) (codec : Codec S)
    (challengeValues : Challenge → List (String × Value)) (sim : ProgramSimulator)
    (A : Adversary S) (right : Bool) (challenge : Challenge) : Comp Bool := do
  let some setup ← sim.setup.runNamed ctx (challengeValues challenge) | pure false
  let some pk := (lookupNamed setup "public_key").bind codec.pk | pure false
  let some ek := (lookupNamed setup "evaluation_key").bind codec.ek | pure false
  let (left, rightMessage, state) ← A.choose pk ek
  let message := codec.msg (if right then rightMessage else left)
  let some outputs ← sim.encrypt.runNamed ctx (("message", message) :: setup) | pure false
  let some ct := (lookupNamed outputs "ciphertext").bind codec.ct | pure false
  A.guess state ct

/-- An allowed assumption a statement uses: its identifier and parameters. -/
structure AssumptionRef where
  id : AssumptionId
  params : id.Params

/-- IND-CPA security relative to the listed assumptions with reductions of bounded cost: there
are program simulators of cost at most `bound`, one per assumption, such that for every hash
function, every adversary, and all bounds `ε i` on the advantages of the reductions it induces
when the programs run with that hash function and `backend`, in both worlds, the
adversary's advantage is at most `2 Σ ε i + δ`. The simulators are chosen before the hash
function, so they cannot depend on it. -/
def SecureWithin (backend : MxxSecurity.IR.Backend) (S : Scheme) (codec : Codec S)
    (uses : List AssumptionRef) (bound : Nat) (δ : ENNReal) : Prop :=
  ∃ sims : Fin uses.length → ProgramSimulator,
    (∀ i, ∃ cost, (sims i).cost = some cost ∧ cost ≤ bound) ∧
    ∀ (hash : MxxRuntime.HashModel) (A : Adversary S) (ε : Fin uses.length → ENNReal),
      (∀ i right, (uses.get i).id.advantage (uses.get i).params
        (reduceProgram ⟨hash, backend⟩ codec
          ((uses.get i).id.challengeValues (uses.get i).params) (sims i) A right) ≤ ε i) →
      advantage S A ≤ 2 * ∑ i, ε i + δ

end MxxFhe
