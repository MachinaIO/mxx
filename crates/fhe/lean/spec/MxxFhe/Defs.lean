import MxxSecurity

/-!
# Public-key homomorphic encryption

The syntax, correctness, and security of a public-key homomorphic encryption scheme, at the level
of abstraction of the literature. A generated scheme instance fills `Scheme` with computations generated from its
DSL graphs; everything in this file is written by hand and is part of the trusted statement.

* **Syntax.** `KeyGen` returns a public key, an evaluation key, and a secret key; `Enc` encrypts
  a message under the public key; `Eval` evaluates a circuit on ciphertexts with the evaluation
  key; `Dec` decrypts with the secret key. An `Evaluation` is one circuit as the scheme evaluates
  it, with its plaintext value.
* **Correctness.** For a circuit and all valid messages, decrypting its evaluation on fresh
  encryptions gives its plaintext value: in every execution (`WorstCaseCorrect`), or except with
  probability `ε` over the randomness of the algorithms (`AverageCaseCorrect`).
* **Security.** IND-CPA, where the adversary sees the public key and the evaluation key.
* **Reductions.** A security statement bounds the advantage by the advantages of reductions
  against the allowed assumptions. A reduction is the fixed template `reduce` around a
  `Simulator`, whose computations are generated from DSL graphs: it runs the adversary once,
  so its cost is the adversary's plus that of the graphs, which are finite circuits.
-/

namespace MxxFhe

open MxxSecurity

/-- A public-key homomorphic encryption scheme. -/
structure Scheme where
  /-- Plaintexts. -/
  Msg : Type
  /-- The message space: the plaintexts correctness covers. -/
  ValidMsg : Msg → Prop
  /-- Public keys. -/
  PK : Type
  /-- Evaluation keys. -/
  EK : Type
  /-- Secret keys. -/
  SK : Type
  /-- Ciphertexts. -/
  CT : Type
  /-- `KeyGen`. -/
  keyGen : Comp (PK × EK × SK)
  /-- `Enc`. -/
  enc : PK → Msg → Comp CT
  /-- `Dec`. -/
  dec : SK → CT → Comp Msg

/-- A circuit on `inputs` inputs as `S` evaluates it: its plaintext value and `Eval` of it on
ciphertexts of its inputs. -/
structure Evaluation (S : Scheme) (inputs : Nat) where
  /-- The circuit's value on plaintexts. -/
  plain : (Fin inputs → S.Msg) → S.Msg
  /-- `Eval` of the circuit. -/
  eval : S.EK → (Fin inputs → S.CT) → Comp S.CT

/-- Encrypt every message, in index order. -/
def Scheme.encAll (S : Scheme) (pk : S.PK) : (count : Nat) → (Fin count → S.Msg) →
    Comp (Fin count → S.CT)
  | 0, _ => pure Fin.elim0
  | count + 1, messages => do
      let head ← S.enc pk (messages 0)
      let tail ← S.encAll pk count fun i ↦ messages i.succ
      pure (Fin.cons head tail)

/-- Whether `Dec(sk, Eval(ek, C, Enc(pk, m₁), …, Enc(pk, mₖ)))` differs from `C(m₁, …, mₖ)`
for fresh keys, where `C` is the circuit of `E`. -/
def Scheme.evaluationFails (S : Scheme) [DecidableEq S.Msg] {inputs : Nat}
    (E : Evaluation S inputs) (messages : Fin inputs → S.Msg) : Comp Bool := do
  let (pk, ek, sk) ← S.keyGen
  let cts ← S.encAll pk inputs messages
  let result ← E.eval ek cts
  let decrypted ← S.dec sk result
  pure (decide (decrypted ≠ E.plain messages))

/-- Average-case correctness of a circuit: on valid messages, decryption of its evaluation fails
with probability at most `ε`. -/
def AverageCaseCorrect (S : Scheme) [DecidableEq S.Msg] {inputs : Nat} (E : Evaluation S inputs)
    (ε : ENNReal) : Prop :=
  ∀ messages, (∀ i, S.ValidMsg (messages i)) → probTrue (S.evaluationFails E messages) ≤ ε

/-- Worst-case correctness of a circuit: on valid messages, every execution decrypts its
evaluation correctly. -/
def WorstCaseCorrect (S : Scheme) [DecidableEq S.Msg] {inputs : Nat} (E : Evaluation S inputs) :
    Prop :=
  ∀ messages, (∀ i, S.ValidMsg (messages i)) → probTrue (S.evaluationFails E messages) = 0

/-- An IND-CPA adversary: it chooses two messages from the public and evaluation keys, then
guesses from the challenge ciphertext. -/
structure Adversary (S : Scheme) where
  /-- What the adversary keeps between its two stages. -/
  State : Type
  /-- Choose the two messages. -/
  choose : S.PK → S.EK → Comp (S.Msg × S.Msg × State)
  /-- Guess from the challenge ciphertext. -/
  guess : State → S.CT → Comp Bool

/-- The world where the challenge encrypts the right message (`right = true`) or the left one. -/
def world (S : Scheme) (A : Adversary S) (right : Bool) : Comp Bool := do
  let (pk, ek, _) ← S.keyGen
  let (left, rightMessage, state) ← A.choose pk ek
  let ct ← S.enc pk (if right then rightMessage else left)
  A.guess state ct

/-- The IND-CPA advantage of `A`. -/
noncomputable def advantage (S : Scheme) (A : Adversary S) : ENNReal :=
  distance (world S A false) (world S A true)

/-- A simulator of the adversary's view from an assumption challenge. -/
structure Simulator (S : Scheme) (Challenge : Type) where
  /-- What the simulator keeps between its two stages. -/
  State : Type
  /-- The public and evaluation keys from the challenge. -/
  setup : Challenge → Comp (S.PK × S.EK × State)
  /-- The challenge ciphertext of a message. -/
  encrypt : State → S.Msg → Comp S.CT

/-- The reduction around a simulator: it runs the adversary once on the simulated view, with
the challenge encrypting its right (`right = true`) or left message, and returns its guess. -/
def reduce {S : Scheme} {Challenge : Type} (sim : Simulator S Challenge) (A : Adversary S)
    (right : Bool) (challenge : Challenge) : Comp Bool := do
  let (pk, ek, simState) ← sim.setup challenge
  let (left, rightMessage, state) ← A.choose pk ek
  let ct ← sim.encrypt simState (if right then rightMessage else left)
  A.guess state ct

/-- One use of an allowed assumption: its identifier, parameters, and the simulator of its
reduction. -/
structure AssumptionUse (S : Scheme) where
  id : AssumptionId
  params : id.Params
  simulator : Simulator S (id.Challenge params)

/-- IND-CPA security relative to the listed assumptions: for every adversary and all bounds
`ε i` on the advantages of the reductions it induces, in both worlds, the adversary's advantage
is at most `2 Σ ε i + δ`. -/
def Secure (S : Scheme) (uses : List (AssumptionUse S)) (δ : ENNReal) : Prop :=
  ∀ (A : Adversary S) (ε : Fin uses.length → ENNReal),
    (∀ i right, (uses.get i).id.advantage (uses.get i).params
      (reduce (uses.get i).simulator A right) ≤ ε i) →
    advantage S A ≤ 2 * ∑ i, ε i + δ

end MxxFhe
