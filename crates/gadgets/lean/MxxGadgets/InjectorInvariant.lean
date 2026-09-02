import MxxGadgets.InputInjector

namespace Mxx.Gadgets

open Mxx.Primitives

/-
  An injector state records exactly the information needed by the next right-preimage
  consumption.  `source` is the current public source, `value` is the current noisy
  state, and `left` is the exact coefficient which the state is intended to carry.
  The only approximation stored in the state is `value ≈ left * source`.
-/
structure InjectorStateInvariant
    {q n sourceRows columns resultRows : Nat}
    (source : ExactMatrix q n sourceRows columns)
    (left : ExactMatrix q n resultRows sourceRows)
    (value : ExactMatrix q n resultRows columns)
    (stateNoiseBound : Nat) where
  leftMagnitude : MagnitudeFact left
  approximation : ApproxWithin value (left * source) stateNoiseBound

namespace InjectorStateInvariant

variable {q n sourceRows columns resultRows : Nat}

def initial
    {source : ExactMatrix q n sourceRows columns}
    {left : ExactMatrix q n resultRows sourceRows}
    {value : ExactMatrix q n resultRows columns}
    {stateNoiseBound : Nat}
    (leftMagnitude : MagnitudeFact left)
    (approximation : ApproxWithin value (left * source) stateNoiseBound) :
    InjectorStateInvariant source left value stateNoiseBound :=
  { leftMagnitude := leftMagnitude, approximation := approximation }

/- The bound paid by one transition.  The first product is the propagated target error and
   the second product is the old state error multiplied by the sampled preimage. -/
def transitionNoiseBound (sourceRows n columns leftBound targetNoiseBound stateNoiseBound
    preimageBound : Nat)
    : Nat :=
  sourceRows * n * leftBound * targetNoiseBound +
    columns * n * stateNoiseBound * preimageBound

/- One transition is parameterized by the current source.  Its ideal target becomes the
   source of the next state; this prevents a caller from silently changing the large term. -/
structure Transition
    (source : ExactMatrix q n sourceRows columns) where
  actualPreimage : ExactMatrix q n columns columns
  actualTarget : ExactMatrix q n sourceRows columns
  nextSource : ExactMatrix q n sourceRows columns
  transitionLeft : ExactMatrix q n sourceRows sourceRows
  transitionLeftMagnitude : MagnitudeFact transitionLeft
  relation : RightPreimage source actualPreimage actualTarget
  preimageBound : Nat
  preimageLift : BoundedLift actualPreimage preimageBound
  targetNoiseBound : Nat
  targetApprox : ApproxWithin actualTarget (transitionLeft * nextSource) targetNoiseBound

noncomputable def consume
    {source : ExactMatrix q n sourceRows columns}
    {left : ExactMatrix q n resultRows sourceRows}
    {value : ExactMatrix q n resultRows columns}
    {stateNoiseBound : Nat}
    (hn : 0 < n)
    (state : InjectorStateInvariant source left value stateNoiseBound)
    (transition : Transition source) :
    InjectorStateInvariant transition.nextSource
      (left * transition.transitionLeft) (value * transition.actualPreimage)
      (transitionNoiseBound sourceRows n columns
        state.leftMagnitude.bound
        transition.targetNoiseBound stateNoiseBound transition.preimageBound) := by
  let nextLeftMagnitude := MagnitudeFact.mul (inner := sourceRows) hn state.leftMagnitude
    transition.transitionLeftMagnitude
  let consumed := input_injector_within hn source transition.actualPreimage
    transition.actualTarget left value (transition.transitionLeft * transition.nextSource)
    transition.relation state.leftMagnitude transition.preimageLift state.approximation
    transition.targetApprox
  refine { leftMagnitude := nextLeftMagnitude, approximation := ?_ }
  exact
    { toApprox :=
        { error := consumed.error
          equation := by
            simpa [Matrix.mul_assoc] using consumed.equation }
      norm_le := consumed.norm_le }

/- A packed state makes a sequential fold possible without existentially hiding the current
   source, value, or bound.  All transitions in one fold have the same rectangular shape. -/
structure PackedState
    {q n sourceRows resultRows : Nat}
    (columns : Nat) where
  left : ExactMatrix q n resultRows sourceRows
  source : ExactMatrix q n sourceRows columns
  value : ExactMatrix q n resultRows columns
  stateNoiseBound : Nat
  invariant : InjectorStateInvariant source left value stateNoiseBound

abbrev TransitionFactory
    {q n sourceRows columns : Nat} :=
  ∀ source : ExactMatrix q n sourceRows columns, Transition source

/- Consuming a transition changes all four packed fields by the same algebraic step:
   `(L, B, X, e) -> (L * S, B', X * K, e')`.  The transition is indexed by the
   state's actual source `B`, so no transition for an unrelated source is required. -/
noncomputable def PackedState.consumeTransition
    {q n sourceRows columns resultRows : Nat}
    (hn : 0 < n)
    (state : PackedState (q := q) (n := n) (sourceRows := sourceRows)
      (resultRows := resultRows) columns)
    (transition : Transition state.source) :
    PackedState (q := q) (n := n) (sourceRows := sourceRows)
      (resultRows := resultRows) columns := by
  let nextInvariant := InjectorStateInvariant.consume hn state.invariant transition
  exact
    { source := transition.nextSource
      value := state.value * transition.actualPreimage
      stateNoiseBound :=
        transitionNoiseBound sourceRows n columns
          state.invariant.leftMagnitude.bound
          transition.targetNoiseBound state.stateNoiseBound transition.preimageBound
      left := state.left * transition.transitionLeft
      invariant := nextInvariant }

/- The factory-based operation remains a generic convenience.  Its implementation immediately
   specializes the factory to the current source and delegates to the dependent operation above. -/
noncomputable def PackedState.consume
    {q n sourceRows columns resultRows : Nat}
    (hn : 0 < n)
    (state : PackedState (q := q) (n := n) (sourceRows := sourceRows)
      (resultRows := resultRows) columns)
    (factory : TransitionFactory (q := q) (n := n) (sourceRows := sourceRows)
      (columns := columns)) :
    PackedState (q := q) (n := n) (sourceRows := sourceRows) (resultRows := resultRows) columns :=
  state.consumeTransition hn (factory state.source)

/- A dependent transition chain is the mathematical counterpart of a successful sequential
   injector loop.  At iteration `i`, the transition is stated only for the source stored in the
   carried state at `i`; `step` records the exact next carried state. -/
structure IndexedTransitionChain
    {q n sourceRows columns resultRows : Nat} (hn : 0 < n) (count : Nat) where
  states : Fin (count + 1) → PackedState (q := q) (n := n) (sourceRows := sourceRows)
    (resultRows := resultRows) columns
  transition : ∀ i : Fin count, Transition (states i.castSucc).source
  step : ∀ i : Fin count,
    states i.succ = (states i.castSucc).consumeTransition hn (transition i)

namespace IndexedTransitionChain

def initial
    {q n sourceRows columns resultRows count : Nat} {hn : 0 < n}
    (chain : IndexedTransitionChain (q := q) (n := n) (sourceRows := sourceRows)
      (columns := columns) (resultRows := resultRows) hn count) :
    PackedState (q := q) (n := n) (sourceRows := sourceRows)
      (resultRows := resultRows) columns :=
  chain.states ⟨0, by omega⟩

def final
    {q n sourceRows columns resultRows count : Nat} {hn : 0 < n}
    (chain : IndexedTransitionChain (q := q) (n := n) (sourceRows := sourceRows)
      (columns := columns) (resultRows := resultRows) hn count) :
    PackedState (q := q) (n := n) (sourceRows := sourceRows)
      (resultRows := resultRows) columns :=
  chain.states (Fin.last count)

def stateInvariant
    {q n sourceRows columns resultRows count : Nat} {hn : 0 < n}
    (chain : IndexedTransitionChain (q := q) (n := n) (sourceRows := sourceRows)
      (columns := columns) (resultRows := resultRows) hn count)
    (index : Fin (count + 1)) :
    InjectorStateInvariant (chain.states index).source (chain.states index).left
      (chain.states index).value (chain.states index).stateNoiseBound :=
  (chain.states index).invariant

def finalInvariant
    {q n sourceRows columns resultRows count : Nat} {hn : 0 < n}
    (chain : IndexedTransitionChain (q := q) (n := n) (sourceRows := sourceRows)
      (columns := columns) (resultRows := resultRows) hn count) :
    InjectorStateInvariant chain.final.source chain.final.left chain.final.value
      chain.final.stateNoiseBound :=
  chain.final.invariant

/- Uniformity constrains only bounds, not sources, targets, or matrices.  Every transition remains
   tied to its own carried source while the three numerical constants are shared by the loop. -/
def UniformBounds
    {q n sourceRows columns resultRows count : Nat} {hn : 0 < n}
    (chain : IndexedTransitionChain (q := q) (n := n) (sourceRows := sourceRows)
      (columns := columns) (resultRows := resultRows) hn count)
    (transitionLeftBound preimageBound targetNoiseBound : Nat) : Prop :=
  ∀ i,
    (chain.transition i).transitionLeftMagnitude.bound = transitionLeftBound ∧
      (chain.transition i).preimageBound = preimageBound ∧
      (chain.transition i).targetNoiseBound = targetNoiseBound

def boundStep (sourceRows n columns transitionLeftBound preimageBound targetNoiseBound : Nat)
    (state : Nat × Nat) : Nat × Nat :=
  (sourceRows * n * state.1 * transitionLeftBound,
    transitionNoiseBound sourceRows n columns state.1 targetNoiseBound state.2 preimageBound)

def boundsFrom (sourceRows n columns transitionLeftBound preimageBound targetNoiseBound : Nat) :
    Nat → Nat × Nat → Nat × Nat
  | 0, state => state
  | steps + 1, state =>
      boundStep sourceRows n columns transitionLeftBound preimageBound targetNoiseBound
        (boundsFrom sourceRows n columns transitionLeftBound preimageBound targetNoiseBound
          steps state)

/- The tail-recursive form matches parameter-search recurrences that update the accumulator before
   recurring.  The equality below lets an application use either evaluation order. -/
def tailBoundsFrom
    (sourceRows n columns transitionLeftBound preimageBound targetNoiseBound : Nat) :
    Nat → Nat × Nat → Nat × Nat
  | 0, state => state
  | steps + 1, state =>
      tailBoundsFrom sourceRows n columns transitionLeftBound preimageBound targetNoiseBound steps
        (boundStep sourceRows n columns transitionLeftBound preimageBound targetNoiseBound state)

theorem boundsFrom_step
    (sourceRows n columns transitionLeftBound preimageBound targetNoiseBound steps : Nat)
    (state : Nat × Nat) :
    boundsFrom sourceRows n columns transitionLeftBound preimageBound targetNoiseBound steps
        (boundStep sourceRows n columns transitionLeftBound preimageBound targetNoiseBound state) =
      boundStep sourceRows n columns transitionLeftBound preimageBound targetNoiseBound
        (boundsFrom sourceRows n columns transitionLeftBound preimageBound targetNoiseBound
          steps state) := by
  induction steps with
  | zero => rfl
  | succ steps inductionHypothesis =>
      simp only [boundsFrom]
      rw [inductionHypothesis]

theorem boundsFrom_eq_tailBoundsFrom
    (sourceRows n columns transitionLeftBound preimageBound targetNoiseBound steps : Nat)
    (state : Nat × Nat) :
    boundsFrom sourceRows n columns transitionLeftBound preimageBound targetNoiseBound steps state =
      tailBoundsFrom sourceRows n columns transitionLeftBound preimageBound targetNoiseBound
        steps state := by
  induction steps generalizing state with
  | zero => rfl
  | succ steps inductionHypothesis =>
      rw [boundsFrom, tailBoundsFrom, ← inductionHypothesis,
        boundsFrom_step]

/- The two stored bounds follow the same forward recurrence as the exact carried states.  The
   first coordinate bounds `L`; the second bounds the error in `X = L * B + e`. -/
theorem boundsAt
    {q n sourceRows columns resultRows count : Nat} {hn : 0 < n}
    (chain : IndexedTransitionChain (q := q) (n := n) (sourceRows := sourceRows)
      (columns := columns) (resultRows := resultRows) hn count)
    (transitionLeftBound preimageBound targetNoiseBound : Nat)
    (uniform : chain.UniformBounds transitionLeftBound preimageBound targetNoiseBound)
    (steps : Nat) (stepsBound : steps ≤ count) :
    let state := chain.states ⟨steps, by omega⟩
    (state.invariant.leftMagnitude.bound, state.stateNoiseBound) =
      boundsFrom sourceRows n columns transitionLeftBound preimageBound targetNoiseBound steps
        (chain.initial.invariant.leftMagnitude.bound, chain.initial.stateNoiseBound) := by
  induction steps with
  | zero => rfl
  | succ steps inductionHypothesis =>
      have previousBound : steps ≤ count := by omega
      have indexBound : steps < count := by omega
      let index : Fin count := ⟨steps, indexBound⟩
      have stepEq := chain.step index
      have previousEq := inductionHypothesis previousBound
      obtain ⟨leftBoundEq, preimageBoundEq, targetNoiseBoundEq⟩ := uniform index
      have nextStateEq : chain.states ⟨steps + 1, by omega⟩ =
          (chain.states ⟨steps, by omega⟩).consumeTransition hn (chain.transition index) := by
        simpa [index] using stepEq
      dsimp only
      rw [nextStateEq]
      change
        (sourceRows * n * (chain.states ⟨steps, by omega⟩).invariant.leftMagnitude.bound *
            (chain.transition index).transitionLeftMagnitude.bound,
          transitionNoiseBound sourceRows n columns
            (chain.states ⟨steps, by omega⟩).invariant.leftMagnitude.bound
            (chain.transition index).targetNoiseBound
            (chain.states ⟨steps, by omega⟩).stateNoiseBound
            (chain.transition index).preimageBound) =
          boundStep sourceRows n columns transitionLeftBound preimageBound targetNoiseBound
            (boundsFrom sourceRows n columns transitionLeftBound preimageBound targetNoiseBound
              steps (chain.initial.invariant.leftMagnitude.bound,
                chain.initial.stateNoiseBound))
      rw [leftBoundEq, preimageBoundEq, targetNoiseBoundEq]
      exact congrArg
        (boundStep sourceRows n columns transitionLeftBound preimageBound targetNoiseBound)
        previousEq

theorem finalBounds
    {q n sourceRows columns resultRows count : Nat} {hn : 0 < n}
    (chain : IndexedTransitionChain (q := q) (n := n) (sourceRows := sourceRows)
      (columns := columns) (resultRows := resultRows) hn count)
    (transitionLeftBound preimageBound targetNoiseBound : Nat)
    (uniform : chain.UniformBounds transitionLeftBound preimageBound targetNoiseBound) :
    (chain.final.invariant.leftMagnitude.bound, chain.final.stateNoiseBound) =
      boundsFrom sourceRows n columns transitionLeftBound preimageBound targetNoiseBound count
        (chain.initial.invariant.leftMagnitude.bound, chain.initial.stateNoiseBound) := by
  simpa [final, Fin.last] using
    chain.boundsAt transitionLeftBound preimageBound targetNoiseBound uniform count (by omega)

end IndexedTransitionChain

/- The sequential operation is an ordinary left fold.  The invariant is preserved at every
   iteration because `PackedState.consume` constructs its next approximation from the same
   `RightPreimage` and `ApproxWithin` premises. -/
noncomputable def sequentialFold
    {q n sourceRows columns resultRows : Nat}
    (hn : 0 < n)
    (initial : PackedState (q := q) (n := n) (sourceRows := sourceRows)
      (resultRows := resultRows) columns)
    (factories : List (TransitionFactory (q := q) (n := n) (sourceRows := sourceRows)
      (columns := columns))) :
    PackedState (q := q) (n := n) (sourceRows := sourceRows) (resultRows := resultRows) columns :=
  factories.foldl (fun state factory => PackedState.consume hn state factory) initial

abbrev IndexedTransitionFactories
    {q n sourceRows columns : Nat} (count : Nat) :=
  Fin count → TransitionFactory (q := q) (n := n) (sourceRows := sourceRows)
    (columns := columns)

/- The indexed form mirrors a generated loop whose transition at iteration `i` is supplied by
   `factories i`.  `List.ofFn` is only the finite traversal order; no index is erased from the
   transition factory itself. -/
noncomputable def indexedSequentialFold
    {q n sourceRows columns resultRows count : Nat}
    (hn : 0 < n)
    (initial : PackedState (q := q) (n := n) (sourceRows := sourceRows)
      (resultRows := resultRows) columns)
    (factories : IndexedTransitionFactories (q := q) (n := n) (sourceRows := sourceRows)
      (columns := columns) count) :
    PackedState (q := q) (n := n) (sourceRows := sourceRows) (resultRows := resultRows) columns :=
  sequentialFold hn initial (List.ofFn factories)

noncomputable def indexedSequentialFoldInvariant
    {q n sourceRows columns resultRows count : Nat}
    (hn : 0 < n)
    (initial : PackedState (q := q) (n := n) (sourceRows := sourceRows)
      (resultRows := resultRows) columns)
    (factories : IndexedTransitionFactories (q := q) (n := n) (sourceRows := sourceRows)
      (columns := columns) count) :
    InjectorStateInvariant (indexedSequentialFold hn initial factories).source
      (indexedSequentialFold hn initial factories).left
      (indexedSequentialFold hn initial factories).value
      (indexedSequentialFold hn initial factories).stateNoiseBound :=
  (indexedSequentialFold hn initial factories).invariant

noncomputable def sequentialFoldInvariant
    {q n sourceRows columns resultRows : Nat}
    (hn : 0 < n)
    (initial : PackedState (q := q) (n := n) (sourceRows := sourceRows)
      (resultRows := resultRows) columns)
    (factories : List (TransitionFactory (q := q) (n := n) (sourceRows := sourceRows)
      (columns := columns))) :
    InjectorStateInvariant (sequentialFold hn initial factories).source
      (sequentialFold hn initial factories).left
      (sequentialFold hn initial factories).value
      (sequentialFold hn initial factories).stateNoiseBound :=
  (sequentialFold hn initial factories).invariant

/- A transition family is uniform when all selected transitions use common preimage and target
   bounds.  This is the form used by a generated input loop: the exact matrices may depend on
   the loop index, while the two sampler bounds and the left magnitude stay fixed. -/
def UniformBounds
    {q n sourceRows columns : Nat}
    (preimageBound targetNoiseBound : Nat)
    (factory : TransitionFactory (q := q) (n := n) (sourceRows := sourceRows)
      (columns := columns)) : Prop :=
  ∀ source, (factory source).preimageBound = preimageBound ∧
    (factory source).targetNoiseBound = targetNoiseBound

end InjectorStateInvariant

end Mxx.Gadgets
