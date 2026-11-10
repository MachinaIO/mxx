import MxxSecurity.Sampling

/-!
# Allowed hardness assumptions

A security statement may assume only the assumptions listed in `AssumptionId`, which mirrors
`mxx_ir_core::security::AssumptionId`. Each assumption is a distinguishing game written here,
by hand, between a real and a random challenge distribution. The games never refer to generated
graphs, so an assumption cannot restate the claim it is used for. A protocol declaration names
the identifiers it uses and their parameters; the exported statement then assumes, for each, a
bound on the advantage of the declared reduction.

Ring elements live in `Z_q[X]/(X^n + 1)` (`Mxx.Primitives.ExactPoly q n`). A secret or error
polynomial is drawn coefficient by coefficient from a `MxxRuntime.SamplerLaw`, as the runtime
samplers draw them.
-/

namespace MxxSecurity

open MxxRuntime Mxx.Primitives

/-- The assumptions a security statement may use. -/
inductive AssumptionId where
  /-- Decision Ring-LWE (`DecisionRlwe`). -/
  | decisionRlwe
  /-- Circular security of a key-switching key from `s²` to `s` (`RlweKeySwitchCircular`). -/
  | rlweKeySwitchCircular
  deriving DecidableEq, Repr

/-- The polynomial with the given integer coefficients, reduced modulo `q`. -/
noncomputable def polyOfInts {q n : Nat} (values : Fin n → Int) : ExactPoly q n :=
  polynomialOfCoefficients values

/-- A uniformly random `1 × k` row over `Z_q[X]/(X^n + 1)`. -/
noncomputable def uniformRow (q n k : Nat) : Comp (ExactMatrix q n 1 k) :=
  sampleMatrix (.residue q) q n 1 k

/-! ## Decision Ring-LWE -/

namespace DecisionRlwe

/-- Decision Ring-LWE with `samples` samples sharing one secret. -/
structure Params where
  /-- The modulus `q`. -/
  modulus : Nat
  /-- The ring dimension `n` of `Z_q[X]/(X^n + 1)`. -/
  ringDimension : Nat
  /-- The law of each secret coefficient. -/
  secretLaw : SamplerLaw
  /-- The law of each error coefficient. -/
  errorLaw : SamplerLaw
  /-- The number of samples. -/
  samples : Nat
  deriving DecidableEq

/-- A challenge: the public row `a` and the row `b`. -/
abbrev Challenge (p : Params) :=
  ExactMatrix p.modulus p.ringDimension 1 p.samples ×
    ExactMatrix p.modulus p.ringDimension 1 p.samples

/-- `(a, s a + e)` for uniform `a`, a secret `s`, and an error row `e`. -/
noncomputable def real (p : Params) : Comp (Challenge p) := do
  let a ← uniformRow p.modulus p.ringDimension p.samples
  let s ← sampleFamily p.secretLaw p.ringDimension
  let e ← sampleMatrix p.errorLaw p.modulus p.ringDimension 1 p.samples
  pure (a, (polyOfInts s : ExactPoly p.modulus p.ringDimension) • a + e)

/-- `(a, u)` for uniform `a` and `u`. -/
noncomputable def random (p : Params) : Comp (Challenge p) := do
  let a ← uniformRow p.modulus p.ringDimension p.samples
  let u ← uniformRow p.modulus p.ringDimension p.samples
  pure (a, u)

end DecisionRlwe

/-! ## Circular security of a key-switching key -/

namespace RlweKeySwitchCircular

/-- A key-switching key from `s²` to `s` over the extended modulus, published next to a
public key over the ciphertext modulus, both under one secret `s`. -/
structure Params where
  /-- The ciphertext modulus `Q`. -/
  modulus : Nat
  /-- The extended key-switching modulus `PQ`. -/
  extendedModulus : Nat
  /-- The ring dimension `n`. -/
  ringDimension : Nat
  /-- The law of each secret coefficient. -/
  secretLaw : SamplerLaw
  /-- The law of each error coefficient, in the public key and in the key. -/
  errorLaw : SamplerLaw
  /-- The plaintext modulus `t` scaling every error. -/
  plaintextModulus : Nat
  /-- The integer factor `c_j` of each key column: column `j` encrypts `c_j s²`. -/
  factors : List Nat
  deriving DecidableEq

/-- A challenge: the public key `(a, s a + t e)` over `Q` and the key rows `(a', b')` over
`PQ`. -/
abbrev Challenge (p : Params) :=
  (ExactMatrix p.modulus p.ringDimension 1 1 × ExactMatrix p.modulus p.ringDimension 1 1) ×
    (ExactMatrix p.extendedModulus p.ringDimension 1 p.factors.length ×
      ExactMatrix p.extendedModulus p.ringDimension 1 p.factors.length)

/-- The public key over `Q` under the secret coefficients `s`. -/
noncomputable def publicKey (p : Params) (s : Fin p.ringDimension → Int) :
    Comp (ExactMatrix p.modulus p.ringDimension 1 1 × ExactMatrix p.modulus p.ringDimension 1 1) :=
  do
  let a ← uniformRow p.modulus p.ringDimension 1
  let e ← sampleMatrix p.errorLaw p.modulus p.ringDimension 1 1
  pure (a, (polyOfInts s : ExactPoly p.modulus p.ringDimension) • a + (p.plaintextModulus : ExactPoly p.modulus p.ringDimension) • e)

/-- The real key: `b'_j = s a'_j + t e'_j + c_j s²` over `PQ`. -/
noncomputable def real (p : Params) : Comp (Challenge p) := do
  let s ← sampleFamily p.secretLaw p.ringDimension
  let pk ← publicKey p s
  let a ← uniformRow p.extendedModulus p.ringDimension p.factors.length
  let e ← sampleMatrix p.errorLaw p.extendedModulus p.ringDimension 1 p.factors.length
  let secret : ExactPoly p.extendedModulus p.ringDimension := polyOfInts s
  let target : ExactMatrix p.extendedModulus p.ringDimension 1 p.factors.length :=
    fun _ column ↦ ((p.factors.get column : Nat) : ExactPoly p.extendedModulus p.ringDimension) *
      (secret * secret)
  pure (pk, (a, secret • a +
    (p.plaintextModulus : ExactPoly p.extendedModulus p.ringDimension) • e + target))

/-- The same public key next to uniform key rows. -/
noncomputable def random (p : Params) : Comp (Challenge p) := do
  let s ← sampleFamily p.secretLaw p.ringDimension
  let pk ← publicKey p s
  let a ← uniformRow p.extendedModulus p.ringDimension p.factors.length
  let u ← uniformRow p.extendedModulus p.ringDimension p.factors.length
  pure (pk, (a, u))

end RlweKeySwitchCircular

/-! ## The list -/

/-- The parameters of an assumption. -/
def AssumptionId.Params : AssumptionId → Type
  | .decisionRlwe => DecisionRlwe.Params
  | .rlweKeySwitchCircular => RlweKeySwitchCircular.Params

/-- The challenge of an assumption at its parameters. -/
def AssumptionId.Challenge : (id : AssumptionId) → id.Params → Type
  | .decisionRlwe, p => DecisionRlwe.Challenge p
  | .rlweKeySwitchCircular, p => RlweKeySwitchCircular.Challenge p

/-- The real challenge distribution. -/
noncomputable def AssumptionId.real : (id : AssumptionId) → (p : id.Params) → Comp (id.Challenge p)
  | .decisionRlwe, p => DecisionRlwe.real p
  | .rlweKeySwitchCircular, p => RlweKeySwitchCircular.real p

/-- The random challenge distribution. -/
noncomputable def AssumptionId.random :
    (id : AssumptionId) → (p : id.Params) → Comp (id.Challenge p)
  | .decisionRlwe, p => DecisionRlwe.random p
  | .rlweKeySwitchCircular, p => RlweKeySwitchCircular.random p

/-- The advantage of a distinguisher `B` of the challenge distributions of `id` at `p`. -/
noncomputable def AssumptionId.advantage (id : AssumptionId) (p : id.Params)
    (B : id.Challenge p → Comp Bool) : ENNReal :=
  distance (id.real p >>= B) (id.random p >>= B)

end MxxSecurity
