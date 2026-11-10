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

abbrev Msg := Int
abbrev PK := (Mxx.Primitives.ExactMatrix 5234636801 1024 1 1) × (Mxx.Primitives.ExactMatrix 5234636801 1024 1 1) × (Fin 5898240 → Int) × (Fin 8192 → Int)
abbrev EK := (Fin 720 → Mxx.Primitives.ExactMatrix 5234636801 1024 1 12) × (Fin 720 → Mxx.Primitives.ExactMatrix 5234636801 1024 1 12) × (Fin 5898240 → Int) × (Fin 8192 → Int)
abbrev SK := Fin 720 → Int

/-- A ciphertext of each type. -/
inductive Ciphertext where
  | ct0 (value : (Fin 720 → Int) × (Int))
  | ct1 (value : (Fin 720 → Int) × (Int))

instance : Nonempty Ciphertext := ⟨.ct0 (Classical.arbitrary _)⟩

noncomputable def keyGen : Comp (PK × EK × SK) := do
  let outputs ← Keygen.generatedRoot.comp Backend.backend Keygen_params ()
  pure ((outputs.2.2.2.2.1, outputs.2.2.2.2.2.1, outputs.2.2.2.2.2.2.1, outputs.2.2.2.2.2.2.2.1), (outputs.1, outputs.2.1, outputs.2.2.2.2.2.2.1, outputs.2.2.2.2.2.2.2.1), outputs.2.2.2.2.2.2.2.2.1)

noncomputable def enc (pk : PK) (m : Msg) : Comp Ciphertext := do
  let outputs ← Encrypt.generatedRoot.comp Encrypt_params ((pk).2.2.1, ((pk).1, ((pk).2.1, ((m), ((pk).2.2.2, ())))))
  pure (.ct0 (outputs.1, outputs.2.1))

noncomputable def dec (sk : SK) : Ciphertext → Comp Msg
  | .ct0 value => do
    let outputs ← Decrypt_0.generatedRoot.comp Decrypt_0_params ((value).2, ((value).1, ((sk), ())))
    pure outputs
  | .ct1 value => do
    let outputs ← Decrypt_1.generatedRoot.comp Decrypt_1_params ((value).2, ((value).1, ((sk), ())))
    pure outputs

/-- The scheme. -/
noncomputable def scheme : MxxFhe.Scheme where
  Msg := Msg
  ValidMsg m := ((0 : Int) ≤ m ∧ m ≤ (1 : Int))
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
noncomputable def evaluation : MxxFhe.Evaluation scheme 2 where
  plain messages :=
    let outputs := MxxSecurity.determined fun outputs : Int ↦
      Plain.generatedRoot Plain_params ((messages ⟨0, by decide⟩), ((messages ⟨1, by decide⟩), ())) outputs
    outputs
  eval ek cts :=
    match cts ⟨0, by decide⟩, cts ⟨1, by decide⟩ with
    | .ct0 c0, .ct0 c1 => do
      let outputs ← Eval.generatedRoot.comp Backend.backend Eval_params ((ek).2.2.1, ((c0).2, ((c1).2, ((c0).1, ((c1).1, ((ek).1, ((ek).2.1, ((ek).2.2.2, ()))))))))
      pure (.ct1 (outputs.1, outputs.2.1))
    | _, _ => pure (cts ⟨0, by decide⟩)

/-- Decryption of the circuit's evaluation fails with probability at most `2^-100`. -/
def CorrectnessClaim : Prop :=
  MxxFhe.AverageCaseCorrect scheme evaluation ((2 : ENNReal)⁻¹ ^ 100)

end GeneratedScheme
