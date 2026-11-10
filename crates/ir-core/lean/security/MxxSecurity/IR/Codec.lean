import MxxSecurity.IR.Cost
import MxxSecurity.Assumptions

/-!
# Values of statements as IR values

A statement that runs an IR program passes it values of its own types and reads values of its
own types back. These conversions are fixed here, by type, so a program receives and returns
exactly the values a statement names: a matrix as a matrix of the same ring and shape, and an
integer family as a family of integers. Neither direction computes anything.
-/

namespace MxxSecurity.IR

open Mxx.Primitives

/-- Types a statement passes to IR programs. -/
class ToValue (α : Type) where
  encode : α → Value

/-- Types a statement reads from IR programs; `none` for a value of another type or shape. -/
class FromValue (α : Type) where
  decode : Value → Option α

instance {q n r c : Nat} : ToValue (ExactMatrix q n r c) where
  encode m := .matrix q n r c m

instance {q n r c : Nat} : FromValue (ExactMatrix q n r c) where
  decode
    | .matrix q' n' r' c' m =>
      if h : q' = q ∧ n' = n ∧ r' = r ∧ c' = c then
        match q', n', r', c', h, m with
        | _, _, _, _, ⟨rfl, rfl, rfl, rfl⟩, m => some m
      else none
    | _ => none

instance {k : Nat} : ToValue (Fin k → Int) where
  encode f := .family (List.ofFn fun i ↦ .int (f i))

instance {k : Nat} : FromValue (Fin k → Int) where
  decode
    | .family members => do
      let values ← members.mapM Value.toInt?
      if h : values.length = k then some fun i ↦ values[i.val]'(h ▸ i.isLt) else none
    | _ => none

theorem decode_encode_matrix {q n r c : Nat} (m : ExactMatrix q n r c) :
    (FromValue.decode (ToValue.encode m) : Option (ExactMatrix q n r c)) = some m := by
  simp [FromValue.decode, ToValue.encode]

/-- The value named `name`. -/
def lookupNamed (env : List (String × Value)) (name : String) : Option Value :=
  (env.find? (·.1 = name)).map (·.2)

/-- Run `program` in `ctx` on the values its root inputs name in `env`, returning its root
outputs by name; `none` if an input is missing or the program fails. -/
noncomputable def Program.runNamed (program : Program) (ctx : Context)
    (env : List (String × Value)) : Comp (Option (List (String × Value))) :=
  match program.inputs.mapM (lookupNamed env) with
  | none => pure none
  | some inputs => do
    let outputs ← program.run ctx inputs
    pure (outputs.map program.outputs.zip)

end MxxSecurity.IR

namespace MxxSecurity

open IR

/-- The components of an assumption's challenge as named IR values, the inputs a reduction's
setup program reads; the names are those of `mxx_ir_core::security::AssumptionId`. -/
def AssumptionId.challengeValues : (id : AssumptionId) → (p : id.Params) → id.Challenge p →
    List (String × Value)
  | .decisionRlwe, _, challenge =>
      [("a", ToValue.encode challenge.1), ("b", ToValue.encode challenge.2)]
  | .rlweKeySwitchCircular, _, challenge =>
      [("pk_a", ToValue.encode challenge.1.1), ("pk_b", ToValue.encode challenge.1.2),
        ("key_a", ToValue.encode challenge.2.1), ("key_b", ToValue.encode challenge.2.2)]

end MxxSecurity
