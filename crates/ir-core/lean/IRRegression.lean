import IRRel
import IRIterRuns

namespace MxxIR

/- A small executable proof fixture for the generic family and loop rules.  It
   intentionally uses only ordinary integers; cryptographic relations live in
   the owning runtime package. -/
def relatedBody (i : Fin 4) : Rel Int (Int × Int) :=
  fun input output => output = (input + i.1, input - i.1)

def relatedFamily : Rel Int (Fin 4 → Int × Int) :=
  fun input output => pointwise (fun i => relatedBody i) (fun _ => input) output

theorem relatedFamily_apply (input : Int) (output : Fin 4 → Int × Int)
    (h : relatedFamily input output) (i : Fin 4) :
    output i = (input + i.1, input - i.1) := by
  exact (h i)

def twoStateStep : Nat → (Int × Int) → (Int × Int) → Prop :=
  fun _ current next => next = (current.1 + 1, current.2 - 1)

theorem twoState_zero (state : Int × Int) : IterRuns twoStateStep 0 state state :=
  .zero state

theorem twoState_one (state : Int × Int) :
    IterRuns twoStateStep 1 state (state.1 + 1, state.2 - 1) := by
  apply IterRuns.step (IterRuns.zero state)
  rfl

end MxxIR
