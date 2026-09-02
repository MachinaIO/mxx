import MxxIrCore.Types

namespace Mxx
namespace IR

noncomputable section

structure StructuralEnv where
  slots : Array (Nat × Int) := #[]
  axes : Array Int := #[]

def StructuralEnv.slot? (env : StructuralEnv) (slot : Nat) : Option Int :=
  (env.slots.find? (fun item => item.1 = slot)).map (fun item => item.2)

def intLog2Ceil (value : Int) : Option Int :=
  if value < 0 then none
  else if value ≤ 1 then some 0
  else
    let rec loop (n answer : Nat) : Nat :=
      if n ≤ 1 then answer else loop ((n + 1) / 2) (answer + 1)
    some (loop value.toNat 0)

/- Every structural integer is evaluated against the occurrence's slot environment.
   Division and logarithm failures remain explicit errors rather than fabricated values. -/
def StructuralIntExpr.eval (env : StructuralEnv) : StructuralIntExpr → Except String Int
  | .literal value => pure value
  | .structuralSlot slot =>
      match env.slot? slot with
      | some value => pure value
      | none => throw s!"structural slot {slot} is out of scope"
  | .add left right => do
      let l ← left.eval env
      let r ← right.eval env
      pure (l + r)
  | .subtract left right => do
      let l ← left.eval env
      let r ← right.eval env
      pure (l - r)
  | .multiply left right => do
      let l ← left.eval env
      let r ← right.eval env
      pure (l * r)
  | .exactDivide left right => do
      let numerator ← left.eval env
      let denominator ← right.eval env
      if denominator = 0 then throw "division by zero"
      else if numerator % denominator ≠ 0 then throw "non-exact division"
      else pure (numerator / denominator)
  | .roundDivide left right => do
      let numerator ← left.eval env
      let denominator ← right.eval env
      if denominator = 0 then throw "division by zero"
      else pure ((numerator + denominator / 2) / denominator)
  | .log2Ceil value => do
      let evaluated ← value.eval env
      match intLog2Ceil evaluated with
      | some result => pure result
      | none => throw "log2ceil input is not a natural number"

/- Index maps are evaluated at one concrete family/grid coordinate.  The fuel is
   derived from the closed expression size by the wrapper and is not a protocol input. -/
def IndexMapExpr.evalFuel (env : StructuralEnv) : Nat → IndexMapExpr → Except String Int
  | 0, _ => throw "index expression evaluation exhausted"
  | fuel + 1, .literal value => pure value
  | fuel + 1, .axis axisIndex =>
      match env.axes[axisIndex]? with
      | some value => pure value
      | none => throw s!"axis {axisIndex} is out of scope"
  | fuel + 1, .structuralSlot slot =>
      match env.slot? slot with
      | some value => pure value
      | none => throw s!"structural slot {slot} is out of scope"
  | fuel + 1, .add left right => do
      let l ← left.evalFuel env fuel
      let r ← right.evalFuel env fuel
      pure (l + r)
  | fuel + 1, .sub left right => do
      let l ← left.evalFuel env fuel
      let r ← right.evalFuel env fuel
      pure (l - r)
  | fuel + 1, .mul left right => do
      let l ← left.evalFuel env fuel
      let r ← right.evalFuel env fuel
      pure (l * r)
  | fuel + 1, .divide left right => do
      let numerator ← left.evalFuel env fuel
      let denominator ← right.evalFuel env fuel
      if denominator = 0 then throw "index division by zero"
      else pure (Int.ediv numerator denominator)
  | fuel + 1, .remainder left right => do
      let numerator ← left.evalFuel env fuel
      let denominator ← right.evalFuel env fuel
      if denominator = 0 then throw "index remainder by zero"
      else pure (Int.emod numerator denominator)
  | fuel + 1, .equal left right => do
      let l ← left.evalFuel env fuel
      let r ← right.evalFuel env fuel
      pure (if l = r then 1 else 0)
  | fuel + 1, .less left right => do
      let l ← left.evalFuel env fuel
      let r ← right.evalFuel env fuel
      pure (if l < r then 1 else 0)
  | fuel + 1, .lessEqual left right => do
      let l ← left.evalFuel env fuel
      let r ← right.evalFuel env fuel
      pure (if l ≤ r then 1 else 0)
  | fuel + 1, .log2Ceil value => do
      let evaluated ← value.evalFuel env fuel
      match intLog2Ceil evaluated with
      | some result => pure result
      | none => throw "log2ceil input is not a natural number"
  | fuel + 1, .select selector branches => do
      let choice ← selector.evalFuel env fuel
      if 0 ≤ choice ∧ choice < branches.size then
        match branches[choice.toNat]? with
        | some branch => branch.evalFuel env fuel
        | none => throw "select branch is missing"
      else throw "select index is out of range"

def IndexMapExpr.eval (env : StructuralEnv) (expression : IndexMapExpr) : Except String Int :=
  expression.evalFuel env (sizeOf expression + 1)

def Rational.toReal (value : Rational) : Real :=
  (value.numerator : Real) / (value.denominator : Real)

def RealExpr.eval (env : StructuralEnv) : RealExpr → Except String Real
  | .literal value =>
      if value.denominator = 0 then throw "rational denominator is zero" else pure value.toReal
  | .fromInt value => do
      let result ← value.eval env
      pure (result : Real)
  | .add left right => do
      let l ← left.eval env
      let r ← right.eval env
      pure (l + r)
  | .subtract left right => do
      let l ← left.eval env
      let r ← right.eval env
      pure (l - r)
  | .multiply left right => do
      let l ← left.eval env
      let r ← right.eval env
      pure (l * r)
  | .divide left right => do
      let numerator ← left.eval env
      let denominator ← right.eval env
      if denominator = 0 then throw "real division by zero" else pure (numerator / denominator)
  | .sqrt value => do
      let result ← value.eval env
      pure (Real.sqrt result)

end
end IR
end Mxx
