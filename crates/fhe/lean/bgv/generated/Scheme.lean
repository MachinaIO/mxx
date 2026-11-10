import Backend
import Decrypt_0
import Decrypt_1
import Encrypt
import Eval
import Keygen
import Plain
import MxxFhe

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000

/-! The public-key homomorphic encryption scheme of the exported graphs and its
correctness and security statements. Generated deterministically; never edit by hand. -/

namespace GeneratedScheme

open MxxSecurity MxxFhe

def Decrypt_0_params : Decrypt_0.Params := { «unit» := () }
def Decrypt_1_params : Decrypt_1.Params := { «unit» := () }
def Encrypt_params : Encrypt.Params := { «unit» := () }
def Eval_params : Eval.Params := { «unit» := () }
def Keygen_params : Keygen.Params := { «unit» := () }
def Plain_params : Plain.Params := { «unit» := () }

abbrev Msg := Fin 8192 → Int
abbrev PK := Mxx.Primitives.ExactMatrix 5846006548020969210596774788483421161649837621249 8192 2 1
abbrev EK := Mxx.Primitives.ExactMatrix 421249166578543632464236954685046510773404157767471124412817604609 8192 2 3
abbrev SK := Mxx.Primitives.ExactMatrix 5846006548020969210596774788483421161649837621249 8192 1 1

/-- A ciphertext of each type. -/
inductive Ciphertext where
  | ct0 (value : Mxx.Primitives.ExactMatrix 5846006548020969210596774788483421161649837621249 8192 2 1)
  | ct1 (value : Mxx.Primitives.ExactMatrix 18014398507892737 8192 2 1)

instance : Nonempty Ciphertext := ⟨.ct0 (Classical.arbitrary _)⟩

noncomputable def keyGen : Comp (PK × EK × SK) := do
  let outputs ← Keygen.generatedRoot.comp Keygen_params ()
  pure (outputs.2.1, outputs.1, outputs.2.2.1)

noncomputable def enc (pk : PK) (m : Msg) : Comp Ciphertext := do
  let outputs ← Encrypt.generatedRoot.comp Encrypt_params ((pk), ((m), ()))
  pure (.ct0 outputs)

noncomputable def dec (sk : SK) : Ciphertext → Comp Msg
  | .ct0 value => do
    let outputs ← Decrypt_0.generatedRoot.comp Decrypt_0_params ((value), ((sk), ()))
    pure outputs
  | .ct1 value => do
    let outputs ← Decrypt_1.generatedRoot.comp Decrypt_1_params ((value), ((sk), ()))
    pure outputs

/-- The scheme. -/
noncomputable def scheme : MxxFhe.Scheme where
  Msg := Msg
  ValidMsg m := (∀ contract_i_0 : Fin 8192, ((0 : Int) ≤ (m contract_i_0) ∧ (m contract_i_0) ≤ (1032192 : Int)))
  PK := PK
  EK := EK
  SK := SK
  CT := Ciphertext
  keyGen := keyGen
  enc := enc
  dec := dec

instance : DecidableEq scheme.Msg := inferInstanceAs (DecidableEq Msg)

/-- The circuit as the scheme evaluates it. `Eval` on an argument that is not fresh returns
its first argument. -/
noncomputable def evaluation : MxxFhe.Evaluation scheme 3 where
  plain messages :=
    let outputs := MxxSecurity.determined fun outputs : Fin 8192 → Int ↦
      Plain.generatedRoot Plain_params ((messages ⟨0, by decide⟩), ((messages ⟨1, by decide⟩), ((messages ⟨2, by decide⟩), ()))) outputs
    outputs
  eval ek cts :=
    match cts ⟨0, by decide⟩, cts ⟨1, by decide⟩, cts ⟨2, by decide⟩ with
    | .ct0 c0, .ct0 c1, .ct0 c2 => do
      let outputs ← Eval.generatedRoot.comp Eval_params ((c0), ((c1), ((c2), ((ek), ()))))
      pure (.ct1 outputs)
    | _, _, _ => pure (cts ⟨0, by decide⟩)

def params_0 : MxxSecurity.AssumptionId.Params MxxSecurity.AssumptionId.rlweKeySwitchCircular := ({ modulus := 5846006548020969210596774788483421161649837621249, extendedModulus := 421249166578543632464236954685046510773404157767471124412817604609, ringDimension := 8192, secretLaw := (MxxRuntime.SamplerLaw.interval (-1) (1)), errorLaw := (MxxRuntime.SamplerLaw.gaussian (3602879701896397/1125899906842624) (20)), plaintextModulus := 1032193, factors := [23384026194045816390345886928489845842762721017857, 23384026193726801671378034657273812936317868343297, 23384026193386519304488586269947798086183317045249] } : MxxSecurity.RlweKeySwitchCircular.Params)

def params_1 : MxxSecurity.AssumptionId.Params MxxSecurity.AssumptionId.decisionRlwe := ({ modulus := 5846006548020969210596774788483421161649837621249, ringDimension := 8192, secretLaw := (MxxRuntime.SamplerLaw.interval (-1) (1)), errorLaw := (MxxRuntime.SamplerLaw.gaussian (3602879701896397/1125899906842624) (20)), samples := 1 } : MxxSecurity.DecisionRlwe.Params)

def params_2 : MxxSecurity.AssumptionId.Params MxxSecurity.AssumptionId.decisionRlwe := ({ modulus := 5846006548020969210596774788483421161649837621249, ringDimension := 8192, secretLaw := (MxxRuntime.SamplerLaw.interval (-1) (1)), errorLaw := (MxxRuntime.SamplerLaw.gaussian (3602879701896397/1125899906842624) (20)), samples := 2 } : MxxSecurity.DecisionRlwe.Params)

/-- Every execution decrypts the circuit's evaluation correctly. -/
def CorrectnessClaim : Prop :=
  MxxFhe.WorstCaseCorrect scheme evaluation

/-- The conversions between the scheme's values and IR values; a challenge ciphertext is
fresh. -/
noncomputable def codec : MxxFhe.Codec scheme where
  msg := MxxSecurity.IR.ToValue.encode (α := Msg)
  pk := MxxSecurity.IR.FromValue.decode (α := PK)
  ek := MxxSecurity.IR.FromValue.decode (α := EK)
  ct value := (MxxSecurity.IR.FromValue.decode value : Option (Mxx.Primitives.ExactMatrix 5846006548020969210596774788483421161649837621249 8192 2 1)).map Ciphertext.ct0

/-- The declared assumptions. -/
noncomputable def assumptions : List MxxFhe.AssumptionRef :=
  [⟨MxxSecurity.AssumptionId.rlweKeySwitchCircular, params_0⟩,
    ⟨MxxSecurity.AssumptionId.decisionRlwe, params_1⟩,
    ⟨MxxSecurity.AssumptionId.decisionRlwe, params_2⟩]

/-- IND-CPA security relative to the declared assumptions, without a statistical term, with
reductions whose simulators are IR programs of cost at most 1073741824, for every backend
and hash function the programs run with. -/
def BoundedSecurityClaim : Prop :=
  ∀ backend : MxxSecurity.IR.Backend,
    MxxFhe.SecureWithin backend scheme codec assumptions 1073741824 0

end GeneratedScheme
