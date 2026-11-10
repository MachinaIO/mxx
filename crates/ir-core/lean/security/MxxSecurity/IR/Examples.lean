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
  simp [applyPure, Value.toInt?, MxxRuntime.intMatrixVectorProduct, List.ofFn_succ]
example : applyPure ctx (fun _ ↦ 0) (.intMatrixVectorProduct true)
    [.family [.int 1, .int 2, .int 3, .int 4], .family [.int 5, .int 6]] =
    some [.family [.int 23, .int 34]] := by
  simp [applyPure, Value.toInt?, MxxRuntime.intMatrixVectorProduct, List.ofFn_succ]
-- integer products beyond `intBound` fail
example : applyPure ctx (fun _ ↦ 0) (.intBinary .mul) [.int (2 ^ 200), .int (2 ^ 100)] = none := by
  simp [applyPure, intBound]
example : applyPure ctx (fun _ ↦ 0) (.intBinary .mul) [.int 6, .int 7] = some [.int 42] := by
  simp [applyPure, intBound]
