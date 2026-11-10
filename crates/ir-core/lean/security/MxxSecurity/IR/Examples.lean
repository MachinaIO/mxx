import MxxSecurity.IR.Lemmas

/-! Regression examples: operations of `MxxSecurity.IR.applyPure` on concrete arguments. -/

open MxxSecurity MxxSecurity.IR

variable (ctx : Context)

-- bit 2 of 13 = 0b1101 is 1, bit 1 is 0
example : applyPure ctx (fun _ ↦ 0) (.bitExtract (.const 2)) [.int 13] = some [.bool true] := by
  simp [applyPure, IExpr.evalNat, IExpr.eval]
example : applyPure ctx (fun _ ↦ 0) (.bitExtract (.const 1)) [.int 13] = some [.bool false] := by
  simp [applyPure, IExpr.evalNat, IExpr.eval]
-- [[1,2],[3,4]] · [5,6] = [17, 39]; transposed: [23, 34]
example : applyPure ctx (fun _ ↦ 0) (.intMatrixVectorProduct false)
    [.family [.int 1, .int 2, .int 3, .int 4], .family [.int 5, .int 6]] =
    some [.family [.int 17, .int 39]] := by
  simp [applyPure, Value.toInt?, MxxRuntime.intMatrixVectorProduct, List.ofFn_succ, intBound]
example : applyPure ctx (fun _ ↦ 0) (.intMatrixVectorProduct true)
    [.family [.int 1, .int 2, .int 3, .int 4], .family [.int 5, .int 6]] =
    some [.family [.int 23, .int 34]] := by
  simp [applyPure, Value.toInt?, MxxRuntime.intMatrixVectorProduct, List.ofFn_succ, intBound]
-- integer products beyond `intBound` fail
example : applyPure ctx (fun _ ↦ 0) (.intBinary .mul) [.int (2 ^ 200), .int (2 ^ 100)] = none := by
  simp [applyPure, intBound]
example : applyPure ctx (fun _ ↦ 0) (.intBinary .mul) [.int 6, .int 7] = some [.int 42] := by
  simp [applyPure, intBound]
example : applyPure ctx (fun _ ↦ 0) (.intMatrixVectorProduct false)
    [.family [.int (2 ^ 200)], .family [.int (2 ^ 100)]] = none := by
  simp [applyPure, Value.toInt?, MxxRuntime.intMatrixVectorProduct, intBound]
-- iteration 1 of a loop that zips its first argument and gathers by it from its second
example : iterationArgs [.zip, .gather 0] [.family [.int 0, .int 2], .family [.int 7, .int 8, .int 9]]
    1 = some [.int 2, .int 9] := by
  simp [iterationArgs, Value.toFamily?, Value.toInt?]
-- the runtime divides by the absolute divisor, rounding down
example : applyPure ctx (fun _ ↦ 0) (.intBinary .div) [.int 5, .int (-3)] = some [.int 1] := by
  simp [applyPure, intBound]
example : applyPure ctx (fun _ ↦ 0) (.intBinary .rem) [.int 5, .int (-3)] = some [.int 2] := by
  simp [applyPure, intBound]
-- constants beyond `intBound` fail
example : applyPure ctx (fun _ ↦ 0) (.constInt (2 ^ 300)) [] = none := by
  simp only [applyPure]
  exact if_neg (by unfold intBound; decide +kernel)
-- only the selected branch is evaluated: the other divides by zero
example : IExpr.eval (fun _ ↦ 1) (.select (.loopIndex 0) [.div (.const 1) (.const 0), .const 0]) =
    some 0 := by
  simp [IExpr.eval, IExpr.evalAt]
-- a residue charges every word of its limbs
example : limbWords [2 ^ 4096 + 1] = 65 := by decide +kernel
example : limbWords [2 ^ 8192 + 1] = 129 := by decide +kernel
-- expression products beyond `intBound` fail
example : IExpr.eval (fun _ ↦ 0) (.mul (.const (2 ^ 200)) (.const (2 ^ 100))) = none := by
  decide +kernel
