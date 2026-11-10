import MxxSecurity.Assumptions
import VCVio.OracleComp.EvalDist.Measure
import VCVio.EvalDist.EvalDistEq

/-!
# Proof tools for sampled computations

Lemmas for proving generated statements; nothing here is part of a statement.

* `determined_congr`: computations generated from equivalent relations return the same output.
* `swap`: independent computations commute in distribution (`EvalDistEq`).
* `probTrue_congr`, `distance_le_*`: advantages of games equal in distribution, and the
  triangle inequality that chains game hops.
-/

namespace MxxSecurity

open MeasureTheory OracleComp

theorem determined_congr {α : Type} [Nonempty α] {left right : α → Prop}
    (h : ∀ value, left value ↔ right value) : determined left = determined right := by
  have : left = right := funext fun value ↦ propext (h value)
  rw [this]

instance (query : SamplerQuery) : Countable (samplerSpec.Range query) := by
  cases query with
  | law _ => exact inferInstanceAs (Countable Int)
  | preimage request => exact inferInstanceAs (Countable request.Coefficients)

/-- Independent computations commute in distribution. -/
theorem swap {α β γ : Type} (mx : Comp α) (my : Comp β) (f : α → β → Comp γ) :
    (mx >>= fun a ↦ my >>= fun b ↦ f a b) =ᵈ (my >>= fun b ↦ mx >>= fun a ↦ f a b) :=
  let : MeasurableSpace γ := ⊤
  EvalDistEq.of_evalDist_eq (OracleComp.evalDist_bind_bind_swap mx my f)

/-- Games equal in distribution win with the same probability. -/
theorem probTrue_congr {left right : Comp Bool} (h : left =ᵈ right) :
    probTrue left = probTrue right := by
  unfold probTrue
  rw [EvalDistEq.evalDist_eq h]

theorem distance_comm (left right : Comp Bool) : distance left right = distance right left := by
  unfold distance
  rw [max_comm]

theorem distance_self (game : Comp Bool) : distance game game = 0 := by
  simp [distance]

theorem distance_congr {left left' right right' : Comp Bool} (hl : left =ᵈ left')
    (hr : right =ᵈ right') : distance left right = distance left' right' := by
  unfold distance
  rw [probTrue_congr hl, probTrue_congr hr]

/-- The triangle inequality of the distinguishing advantage. -/
theorem distance_triangle (a b c : Comp Bool) :
    distance a c ≤ distance a b + distance b c := by
  unfold distance
  set x := probTrue a
  set y := probTrue b
  set z := probTrue c
  refine max_le ?_ ?_
  · calc x - z ≤ (x - y) + (y - z) := tsub_le_tsub_add_tsub
      _ ≤ max (x - y) (y - x) + max (y - z) (z - y) := add_le_add (le_max_left _ _) (le_max_left _ _)
  · calc z - x ≤ (z - y) + (y - x) := tsub_le_tsub_add_tsub
      _ = (y - x) + (z - y) := add_comm _ _
      _ ≤ max (x - y) (y - x) + max (y - z) (z - y) :=
          add_le_add (le_max_right _ _) (le_max_right _ _)

end MxxSecurity
