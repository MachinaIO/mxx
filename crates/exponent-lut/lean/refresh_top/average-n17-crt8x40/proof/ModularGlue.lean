import Lean

/-!
# Glue steps as separate theorems

`node_have h := e` (or `node_have h : T := e`) elaborates the step `e` in the current context,
adds it as an auxiliary theorem over exactly the local variables it uses (`mkAuxTheorem`), and
introduces `h` as that theorem applied to them. Each glue step is then kernel-checked once, on its
own, and the enclosing proof term only applies the step theorems: no step's proof is carried
through (and instantiated into) the later steps.
-/

open Lean Elab Tactic Meta

syntax (name := nodeHave) "node_have " ident (" : " term)? " := " term : tactic

@[tactic nodeHave] def evalNodeHave : Tactic := fun stx => do
  let name := stx[1].getId
  withMainContext do
    -- the optional `: T` is a node with the atom and the type (empty when absent)
    let expected? ← if stx[2].getNumArgs == 2 then do
        let t ← Term.elabType stx[2][1]
        Term.synthesizeSyntheticMVarsNoPostponing
        pure (some (← instantiateMVars t))
      else pure none
    let e ← Term.withSynthesize <| elabTermEnsuringType stx[4] expected?
    let e ← instantiateMVars e
    let ty ← instantiateMVars (← match expected? with
      | some t => pure t
      | none => inferType e)
    if e.hasExprMVar || ty.hasExprMVar then
      throwError "node_have {name}: the step has unassigned metavariables"
    let proof ← mkAuxTheorem ty e (zetaDelta := true)
    let goal ← (← getMainGoal).assert name ty proof
    let (_, goal) ← goal.intro1P
    replaceMainGoal [goal]
