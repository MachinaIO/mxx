import MxxFhe.Defs

/-!
# The hybrid argument

`secure_of_hops` reduces `Secure` to one distributional equality per game hop.
-/

namespace MxxFhe

open MxxSecurity


variable {S : Scheme}

/-- The game of use `u` on its real challenge, with reduction `reduce u.simulator A right`. -/
noncomputable def realHop (u : AssumptionUse S) (A : Adversary S) (right : Bool) : Comp Bool :=
  u.id.real u.params >>= reduce u.simulator A right

/-- The game of use `u` on its random challenge. -/
noncomputable def randomHop (u : AssumptionUse S) (A : Adversary S) (right : Bool) : Comp Bool :=
  u.id.random u.params >>= reduce u.simulator A right

/-- The hybrid argument: if the real world is the real challenge of the first use, the random
challenge of each use is the real challenge of the next, and the random challenge of the last
use no longer depends on the world, then the scheme is secure relative to the uses. -/
theorem secure_of_hops (uses : List (AssumptionUse S)) (hne : 0 < uses.length)
    (first : ∀ A right, world S A right =ᵈ realHop (uses.get ⟨0, hne⟩) A right)
    (step : ∀ A right (i : Nat) (hi : i + 1 < uses.length),
      randomHop (uses.get ⟨i, by omega⟩) A right =ᵈ realHop (uses.get ⟨i + 1, hi⟩) A right)
    (last : ∀ A, randomHop (uses.get ⟨uses.length - 1, by omega⟩) A false =ᵈ
      randomHop (uses.get ⟨uses.length - 1, by omega⟩) A true) :
    Secure S uses 0 := by
  intro A ε hε
  let ε' : Nat → ENNReal := fun j ↦ if h : j < uses.length then ε ⟨j, h⟩ else 0
  -- Every hybrid's random game is within the sum of the earlier bounds of the world.
  have chain : ∀ right (i : Nat) (hi : i < uses.length),
      distance (world S A right) (randomHop (uses.get ⟨i, hi⟩) A right) ≤
        ∑ j ∈ Finset.range (i + 1), ε' j := by
    intro right i
    induction i with
    | zero =>
      intro hi
      have hadv := hε ⟨0, hi⟩ right
      calc distance (world S A right) (randomHop (uses.get ⟨0, hi⟩) A right)
          ≤ distance (world S A right) (realHop (uses.get ⟨0, hi⟩) A right) +
              distance (realHop (uses.get ⟨0, hi⟩) A right)
                (randomHop (uses.get ⟨0, hi⟩) A right) := distance_triangle _ _ _
        _ = distance (realHop (uses.get ⟨0, hi⟩) A right)
                (randomHop (uses.get ⟨0, hi⟩) A right) := by
            rw [distance_congr (first A right) EvalDistEq.rfl, distance_self, zero_add]
        _ ≤ ∑ j ∈ Finset.range (0 + 1), ε' j := by
            simp only [zero_add, Finset.range_one, Finset.sum_singleton, ε', dif_pos hi]
            exact hadv
    | succ i ih =>
      intro hi
      have hadv := hε ⟨i + 1, hi⟩ right
      calc distance (world S A right) (randomHop (uses.get ⟨i + 1, hi⟩) A right)
          ≤ distance (world S A right) (randomHop (uses.get ⟨i, by omega⟩) A right) +
              distance (randomHop (uses.get ⟨i, by omega⟩) A right)
                (randomHop (uses.get ⟨i + 1, hi⟩) A right) := distance_triangle _ _ _
        _ ≤ distance (world S A right) (randomHop (uses.get ⟨i, by omega⟩) A right) +
              (distance (randomHop (uses.get ⟨i, by omega⟩) A right)
                (realHop (uses.get ⟨i + 1, hi⟩) A right) +
              distance (realHop (uses.get ⟨i + 1, hi⟩) A right)
                (randomHop (uses.get ⟨i + 1, hi⟩) A right)) :=
            add_le_add le_rfl (distance_triangle _ _ _)
        _ ≤ (∑ j ∈ Finset.range (i + 1), ε' j) + (0 + ε' (i + 1)) := by
            refine add_le_add (ih (by omega)) (add_le_add ?_ ?_)
            · rw [distance_congr (step A right i hi) EvalDistEq.rfl, distance_self]
            · simp only [ε', dif_pos hi]
              exact hadv
        _ = ∑ j ∈ Finset.range (i + 1 + 1), ε' j := by
            rw [Finset.sum_range_succ (n := i + 1), zero_add]
  have hsum : ∑ j ∈ Finset.range (uses.length - 1 + 1), ε' j = ∑ i, ε i := by
    rw [Nat.sub_add_cancel hne, Finset.sum_range]
    refine Finset.sum_congr rfl fun j _ ↦ ?_
    simp [ε', j.isLt]
  have hlast := chain false (uses.length - 1) (by omega)
  have hlast' := chain true (uses.length - 1) (by omega)
  rw [hsum] at hlast hlast'
  unfold advantage
  calc distance (world S A false) (world S A true)
      ≤ distance (world S A false) (randomHop (uses.get ⟨uses.length - 1, by omega⟩) A false) +
          distance (randomHop (uses.get ⟨uses.length - 1, by omega⟩) A false) (world S A true) :=
        distance_triangle _ _ _
    _ ≤ (∑ i, ε i) + (distance (randomHop (uses.get ⟨uses.length - 1, by omega⟩) A false)
            (randomHop (uses.get ⟨uses.length - 1, by omega⟩) A true) +
          distance (randomHop (uses.get ⟨uses.length - 1, by omega⟩) A true) (world S A true)) :=
        add_le_add hlast (distance_triangle _ _ _)
    _ ≤ (∑ i, ε i) + (0 + ∑ i, ε i) := by
        refine add_le_add le_rfl (add_le_add ?_ ?_)
        · rw [distance_congr (last A) EvalDistEq.rfl, distance_self]
        · rw [distance_comm]
          exact hlast'
    _ = 2 * ∑ i, ε i + 0 := by ring

end MxxFhe
