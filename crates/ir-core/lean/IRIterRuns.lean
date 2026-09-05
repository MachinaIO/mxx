namespace MxxIR

universe u

/- The sole execution derivation needed by a generated sequential loop. -/
inductive IterRuns (body : Nat → State → State → Prop) : Nat → State → State → Prop
  | zero (initial : State) : IterRuns body 0 initial initial
  | step {count : Nat} {initial current next : State} :
      IterRuns body count initial current → body count current next →
        IterRuns body (count + 1) initial next

theorem IterRuns.zero_iff {body : Nat → State → State → Prop} {state : State} :
    IterRuns body 0 state state :=
  .zero state

theorem IterRuns.step_of_succ {body : Nat → State → State → Prop}
    {count : Nat} {initial next : State} (h : IterRuns body (count + 1) initial next) :
    ∃ current, IterRuns body count initial current ∧ body count current next := by
  cases h with
  | step hprev hstep => exact ⟨_, hprev, hstep⟩

theorem IterRuns.invariant {body : Nat → State → State → Prop}
    {Invariant : Nat → State → Prop} {count : Nat} {initial final : State}
    (initialInvariant : Invariant 0 initial)
    (stepInvariant : ∀ i current next, Invariant i current → body i current next →
      Invariant (i + 1) next)
    (runs : IterRuns body count initial final) :
    Invariant count final := by
  induction runs with
  | zero => exact initialInvariant
  | step previous stepRun ih => exact stepInvariant _ _ _ (ih initialInvariant) stepRun

end MxxIR
