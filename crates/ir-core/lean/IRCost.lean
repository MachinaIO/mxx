import IRIterRuns
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Data.Fintype.Fin

namespace MxxIR

universe u

/-- Additive work attached to a state transition, independently of its application. -/
abbrev CostRel (State : Type u) := State → State → Nat → Prop

/-- Erasing work recovers the underlying transition relation. -/
def CostRel.erase {State : Type u} (body : CostRel State) : State → State → Prop :=
  fun initial final ↦ ∃ cost, body initial final cost

/-- Execution work is accumulated over iterations, without unrolling their graph bodies.

The body must count its execution schedule, not traverse operand references: reusing a value
does not execute its producer again. A separately constructed producer or another invocation
does contribute another body cost. Parallel work is additive too; it is not elapsed latency.
-/
inductive CostedIterRuns {State : Type u} (body : Nat → CostRel State) :
    Nat → State → State → Nat → Prop
  | zero (initial : State) : CostedIterRuns body 0 initial initial 0
  | step {count previousCost stepCost : Nat} {initial current next : State} :
      CostedIterRuns body count initial current previousCost →
      body count current next stepCost →
      CostedIterRuns body (count + 1) initial next (previousCost + stepCost)

/-- A counted execution is an execution of the same body after forgetting its work. -/
theorem CostedIterRuns.erase {State : Type u} {body : Nat → CostRel State}
    {count cost : Nat} {initial final : State}
    (run : CostedIterRuns body count initial final cost) :
    IterRuns (fun i ↦ (body i).erase) count initial final := by
  induction run with
  | zero initial => exact .zero initial
  | step previous stepRun ih => exact .step ih ⟨_, stepRun⟩

/-- Attaching costs neither removes nor invents executions of the erased body. -/
theorem CostedIterRuns.exists_iff {State : Type u} {body : Nat → CostRel State}
    {count : Nat} {initial final : State} :
    (∃ cost, CostedIterRuns body count initial final cost) ↔
      IterRuns (fun i ↦ (body i).erase) count initial final := by
  constructor
  · rintro ⟨cost, run⟩
    exact run.erase
  · intro run
    induction run with
    | zero initial => exact ⟨0, .zero initial⟩
    | step previous stepRun ih =>
      obtain ⟨previousCost, previousRun⟩ := ih
      obtain ⟨stepCost, stepRun⟩ := stepRun
      exact ⟨previousCost + stepCost, .step previousRun stepRun⟩

/-- A compact sum of iteration costs. The index permits nonuniform loop bodies. -/
def iterationWork (work : Nat → Nat) : Nat → Nat
  | 0 => 0
  | count + 1 => iterationWork work count + work count

/-- State-independent body costs determine total work for every admissible execution. -/
theorem CostedIterRuns.cost_eq {State : Type u} {body : Nat → CostRel State}
    {work : Nat → Nat} (bodyCost : ∀ i current next cost,
      body i current next cost → cost = work i)
    {count cost : Nat} {initial final : State}
    (run : CostedIterRuns body count initial final cost) :
    cost = iterationWork work count := by
  induction run with
  | zero initial => rfl
  | step previous stepRun ih =>
    rw [iterationWork, ih, bodyCost _ _ _ _ stepRun]

/-- Constant-cost bodies can be counted without enumerating iterations. -/
theorem iterationWork_const (work count : Nat) :
    iterationWork (fun _ ↦ work) count = count * work := by
  induction count with
  | zero => simp [iterationWork]
  | succ count ih => simp [iterationWork, ih, Nat.succ_mul]

/-- Costs can depend on the index; constant-cost multiplication needs an explicit premise. -/
theorem CostedIterRuns.constant_cost {State : Type u} {body : Nat → CostRel State}
    {work : Nat} (bodyCost : ∀ i current next cost,
      body i current next cost → cost = work)
    {count cost : Nat} {initial final : State}
    (run : CostedIterRuns body count initial final cost) : cost = count * work := by
  exact (run.cost_eq bodyCost).trans (iterationWork_const work count)

/-- Sequential execution composes work from two invocations, even of the same body. -/
def CostRel.comp {State : Type u} (left right : CostRel State) : CostRel State :=
  fun initial final cost ↦ ∃ middle leftCost rightCost,
    left initial middle leftCost ∧ right middle final rightCost ∧ cost = leftCost + rightCost

/-- Reading an already computed value has no producer-execution work. -/
def CostRel.reuse {State : Type u} : CostRel State :=
  fun initial final cost ↦ final = initial ∧ cost = 0

/-- Reusing the result preserves the producer's cost, without charging it per reference. -/
theorem CostRel.comp_reuse {State : Type u} {body : CostRel State}
    {initial final : State} {cost : Nat} :
    body.comp CostRel.reuse initial final cost ↔ body initial final cost := by
  constructor
  · rintro ⟨middle, leftCost, rightCost, run, ⟨rfl, rfl⟩, rfl⟩
    simpa using run
  · intro run
    exact ⟨final, cost, 0, run, ⟨rfl, rfl⟩, by simp⟩

/-- Two invocations are charged separately; equal inputs or equal values do not merge them. -/
theorem CostRel.comp_cost {State : Type u} {left right : CostRel State}
    {leftWork rightWork : Nat}
    (leftCost : ∀ initial final cost, left initial final cost → cost = leftWork)
    (rightCost : ∀ initial final cost, right initial final cost → cost = rightWork)
    {initial final : State} {cost : Nat}
    (run : left.comp right initial final cost) : cost = leftWork + rightWork := by
  obtain ⟨middle, a, b, leftRun, rightRun, rfl⟩ := run
  rw [leftCost _ _ _ leftRun, rightCost _ _ _ rightRun]

/-- A compositional instrumentation of a proposition, with an exact erasure certificate.

Generated scopes build this alongside their ordinary relations, using the same primitive
conditions, witnesses, calls and loops. The certificate alone does not establish a particular
cost: that requires `HasCost` and the corresponding compositional cost lemmas below.
-/
structure Costed (condition : Prop) where
  runs : Nat → Prop
  erase_iff : (∃ cost, runs cost) ↔ condition

namespace Costed

/-- Every counted derivation is an ordinary derivation. -/
theorem erase {condition : Prop} (execution : Costed condition) {cost : Nat}
    (run : execution.runs cost) : condition := execution.erase_iff.mp ⟨cost, run⟩

/-- Every ordinary derivation admits a counted derivation. -/
theorem lift {condition : Prop} (execution : Costed condition) (run : condition) :
    ∃ cost, execution.runs cost := execution.erase_iff.mpr run

/-- All admissible executions have this work; this does not assert execution existence. -/
def HasCost {condition : Prop} (execution : Costed condition) (work : Nat) : Prop :=
  ∀ cost, execution.runs cost → cost = work

/-- Exact work follows from the compositional cost proof, for precisely the original runs. -/
theorem runs_iff {condition : Prop} {execution : Costed condition} {work : Nat}
    (h : execution.HasCost work) (cost : Nat) :
    execution.runs cost ↔ condition ∧ cost = work := by
  constructor
  · intro run
    exact ⟨execution.erase run, h _ run⟩
  · rintro ⟨run, rfl⟩
    obtain ⟨actual, counted⟩ := execution.lift run
    rwa [← h _ counted]

/-- One producer executes once, independently of how many operands refer to its output. -/
def atom (condition : Prop) (work : Nat) : Costed condition where
  runs cost := condition ∧ cost = work
  erase_iff := ⟨fun ⟨_, run, _⟩ ↦ run, fun run ↦ ⟨work, run, rfl⟩⟩

theorem atom_cost (condition : Prop) (work : Nat) : (atom condition work).HasCost work :=
  fun _ run ↦ run.2

/-- Two scheduled operations contribute independently, including repeated invocations. -/
def both {left right : Prop} (a : Costed left) (b : Costed right) : Costed (left ∧ right) where
  runs cost := ∃ x y, a.runs x ∧ b.runs y ∧ cost = x + y
  erase_iff := by
    constructor
    · rintro ⟨_, x, y, ha, hb, _⟩
      exact ⟨a.erase ha, b.erase hb⟩
    · rintro ⟨ha, hb⟩
      obtain ⟨x, hx⟩ := a.lift ha
      obtain ⟨y, hy⟩ := b.lift hb
      exact ⟨x + y, x, y, hx, hy, rfl⟩

theorem both_cost {left right : Prop} {a : Costed left} {b : Costed right}
    {x y : Nat} (ha : a.HasCost x) (hb : b.HasCost y) : (both a b).HasCost (x + y) := by
  rintro cost ⟨u, v, hu, hv, rfl⟩
  rw [ha _ hu, hb _ hv]

/-- Add the work of the current scope's producers without changing its conditions. -/
def addWork {condition : Prop} (execution : Costed condition) (extra : Nat) :
    Costed condition where
  runs cost := ∃ previous, execution.runs previous ∧ cost = previous + extra
  erase_iff := by
    constructor
    · rintro ⟨_, previous, run, _⟩
      exact execution.erase run
    · intro run
      obtain ⟨previous, counted⟩ := execution.lift run
      exact ⟨previous + extra, previous, counted, rfl⟩

theorem addWork_cost {condition : Prop} {execution : Costed condition} {work : Nat}
    (h : execution.HasCost work) (extra : Nat) : (execution.addWork extra).HasCost (work + extra) := by
  rintro cost ⟨previous, run, rfl⟩
  rw [h _ run]

/-- Existential intermediate values preserve the actual execution witnesses. -/
def witness {α : Type u} {condition : α → Prop} (body : ∀ x, Costed (condition x)) :
    Costed (∃ x, condition x) where
  runs cost := ∃ x, (body x).runs cost
  erase_iff := by
    constructor
    · rintro ⟨cost, x, run⟩
      exact ⟨x, (body x).erase run⟩
    · rintro ⟨x, run⟩
      obtain ⟨cost, counted⟩ := (body x).lift run
      exact ⟨cost, x, counted⟩

theorem witness_cost {α : Type u} {condition : α → Prop}
    {body : ∀ x, Costed (condition x)} {work : Nat}
    (h : ∀ x, (body x).HasCost work) : (witness body).HasCost work := by
  rintro cost ⟨x, run⟩
  exact h x cost run

/-- Independent loop iterations all execute; the sum is work, not parallel latency. -/
noncomputable def all {count : Nat} {condition : Fin count → Prop}
    (body : ∀ i, Costed (condition i)) : Costed (∀ i, condition i) where
  runs cost := ∃ costs : Fin count → Nat,
    (∀ i, (body i).runs (costs i)) ∧ cost = ∑ i, costs i
  erase_iff := by
    classical
    constructor
    · rintro ⟨_, costs, runs, _⟩ i
      exact (body i).erase (runs i)
    · intro run
      let costs := fun i ↦ Classical.choose ((body i).lift (run i))
      exact ⟨∑ i, costs i, costs, fun i ↦ Classical.choose_spec ((body i).lift (run i)), rfl⟩

theorem all_cost {count : Nat} {condition : Fin count → Prop}
    {body : ∀ i, Costed (condition i)} {work : Fin count → Nat}
    (h : ∀ i, (body i).HasCost (work i)) : (all body).HasCost (∑ i, work i) := by
  rintro cost ⟨costs, runs, rfl⟩
  exact Finset.sum_congr rfl (fun i _ ↦ h i _ (runs i))

/-- Sequential-loop instrumentation retains the ordinary intermediate-state derivation. -/
def iter {State : Type u} {condition : Nat → State → State → Prop}
    (body : ∀ i current next, Costed (condition i current next))
    (count : Nat) (initial final : State) : Costed (IterRuns condition count initial final) where
  runs cost := CostedIterRuns (fun i current next ↦ (body i current next).runs)
    count initial final cost
  erase_iff := by
    constructor
    · rintro ⟨_, run⟩
      induction run with
      | zero initial => exact .zero initial
      | step previous stepRun ih => exact .step ih ((body _ _ _).erase stepRun)
    · intro run
      induction run with
      | zero initial => exact ⟨0, .zero initial⟩
      | step previous stepRun ih =>
        obtain ⟨previousCost, previousRun⟩ := ih
        obtain ⟨stepCost, countedStep⟩ := (body _ _ _).lift stepRun
        exact ⟨previousCost + stepCost, .step previousRun countedStep⟩

theorem iter_cost {State : Type u} {condition : Nat → State → State → Prop}
    {body : ∀ i current next, Costed (condition i current next)} {work : Nat → Nat}
    (h : ∀ i current next, (body i current next).HasCost (work i))
    (count : Nat) (initial final : State) :
    (iter body count initial final).HasCost (iterationWork work count) :=
  fun _ run ↦ run.cost_eq h

end Costed

end MxxIR
