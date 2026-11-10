import TfheEval

/-!
# One run in closed form

`runOf messages ω` is the ciphertext of the NAND of the encryptions of `messages` on the tape `ω`,
and `failR_eq` says that the run fails exactly when its decryption differs from the NAND of the
messages.
-/

set_option maxRecDepth 16384

namespace MxxFheTfhe

open Mxx.Primitives MxxRuntime MxxSecurity GeneratedScheme

/-- The encryption inputs of message `m` under the keys of `ω`. -/
noncomputable def encInputsOf (ω : Key → ℤ) (m : Int) : EncInputs :=
  (kskAOf ω, smp (.residue Q) 13 ω, pkBOf ω, m, kskBOf ω, ())

/-- The fresh encryption of `m` at sites `base`, `base + 1`, `base + 2`. -/
noncomputable def freshOf (ω : Key → ℤ) (base : Nat) (m : Int) : (Fin 720 → Int) × Int × Unit :=
  encOf (encInputsOf ω m) (smp bitLaw base ω) (smp ringLaw (base + 1) ω) (smp ringLaw (base + 2) ω)

/-- The inputs of the NAND gate on the fresh encryptions of `messages`. -/
noncomputable def evalInputsOf (messages : Fin 2 → Int) (ω : Key → ℤ) : EvalInputs :=
  (kskAOf ω, (freshOf ω 15 (messages 0)).2.1, (freshOf ω 18 (messages 1)).2.1,
    (freshOf ω 15 (messages 0)).1, (freshOf ω 18 (messages 1)).1, fun i ↦ (bskOf ω i).1,
    fun i ↦ (bskOf ω i).2.1, kskBOf ω, ())

/-- The NAND of the fresh encryptions of `messages`. -/
noncomputable def runOf (messages : Fin 2 → Int) (ω : Key → ℤ) : (Fin 720 → Int) × Int × Unit :=
  evalOf (evalInputsOf messages ω)

theorem failR_eq (messages : Fin 2 → Int) (ω : Key → ℤ) :
    failR messages ω = decide (decOf (runOf messages ω).2.1 (runOf messages ω).1 (secretOf ω) ≠
      1 - messages 0 * messages 1) := by
  unfold failR
  simp only [keygenOut_eq, encOut_eq, evalOut_eq, decrypt_eq, plain_eq]
  rfl

end MxxFheTfhe
