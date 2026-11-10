import MxxSecurity.IR.Syntax
import MxxSecurity.Sampling
import Mathlib.Analysis.SpecialFunctions.Sqrt

/-!
# The semantics of IR programs

`Program.run` gives a program its meaning, in an evaluation `Context`, as a sampled computation
that returns the values of the root outputs, or `none` when the program is ill-formed. Every
operation is a function of its arguments, or a draw from `MxxSecurity.sample` for a sampler; no
operation selects a value satisfying a relation, so a program can compute only what its
operations compute. The context holds what the runtime reads besides the program: the hash
function of the hash operations and the backend layouts of the gadget and trapdoor operations.

Three rules keep the cost of `MxxSecurity.IR.Cost` sound for the work evaluation does:

* every output of a node must have the type the node declares (`Value.conforms`), so the sizes
  the cost reads from types are the sizes of the values;
* integer arithmetic fails beyond `intBound`, so a unit-cost integer operation cannot build
  integers of exponential length;
* a trapdoor value keeps the secret its generation sampled, and only `gadgetTrapdoor` and
  `trapdoorSample` create one, so a program holds a trapdoor only for a matrix it generated.

Every operation has a semantics. A preimage under a sampled trapdoor is a draw of the ideal
preimage sampler (`MxxSecurity.PreimageRequest`), the law the runtime's Gaussian sampler of
Micciancio and Peikert is trusted to draw; a preimage under a public gadget trapdoor is the gadget
decomposition of the target.
-/

namespace MxxSecurity.IR

open Mxx.Primitives MxxRuntime

/-- The backend data evaluation reads: gadget layouts and default preimage widths. -/
structure Backend where
  /-- The gadget layout of every ring. -/
  layouts : BackendContext
  /-- The smallest preimage width the sampler accepts for a trapdoor of TrapGen width `sigma`
  over `Z_q[X]/(X^n + 1)` with `rows` rows, gadget base `base`, and `statisticalRows` columns of
  `ā` for a statistical trapdoor: the width of its preimages when the program names none. -/
  preimageWidth : (q n rows base : Nat) → (sigma : Rat) → (statisticalRows : Option Nat) → Rat

/-- What evaluation reads besides the program and its inputs: the hash function of the hash
operations and the backend of the gadget and trapdoor operations. A statement fixes it; a program
cannot choose it. -/
structure Context where
  hash : HashModel
  backend : Backend

/-- The values of the loop indices in scope, by slot. -/
abbrev LoopEnv := Nat → Int

mutual
/-- The value of an integer expression; structural, so that a cost or a value reduces by
computation. -/
def IExpr.eval (env : LoopEnv) : IExpr → Option Int
  | .const value => some value
  | .loopIndex slot => some (env slot)
  | .add left right => do pure ((← left.eval env) + (← right.eval env))
  | .sub left right => do pure ((← left.eval env) - (← right.eval env))
  | .mul left right => do pure ((← left.eval env) * (← right.eval env))
  | .div left right => do
      let x ← left.eval env
      let y ← right.eval env
      if y = 0 ∨ x % y ≠ 0 then none else pure (x / y)
  | .floorDiv left right => do
      let x ← left.eval env
      let y ← right.eval env
      if y = 0 then none else pure (Int.fdiv x y)
  | .rem left right => do
      let x ← left.eval env
      let y ← right.eval env
      if y = 0 then none else pure (Int.fmod x y)
  | .roundDiv left right => do
      let x ← left.eval env
      let y ← right.eval env
      if y ≤ 0 then none else pure (Int.fdiv (2 * x + y) (2 * y))
  | .log2Ceil value => do
      let x ← value.eval env
      if x < 1 then none else pure (Nat.clog 2 x.toNat)
  | .select selector branches => do
      let index ← selector.eval env
      let values ← IExpr.evalList env branches
      if index < 0 then none else values[index.toNat]?

def IExpr.evalList (env : LoopEnv) : List IExpr → Option (List Int)
  | [] => some []
  | e :: es => do pure ((← e.eval env) :: (← IExpr.evalList env es))
end

/-- A natural number: a nonnegative integer expression. -/
def IExpr.evalNat (env : LoopEnv) (e : IExpr) : Option Nat := do
  let x ← e.eval env
  if x < 0 then none else pure x.toNat

noncomputable def RExpr.eval (env : LoopEnv) : RExpr → Option ℝ
  | .rational value => some value
  | .ofInt value => do pure ((← value.eval env) : ℝ)
  | .add left right => do pure ((← left.eval env) + (← right.eval env))
  | .sub left right => do pure ((← left.eval env) - (← right.eval env))
  | .mul left right => do pure ((← left.eval env) * (← right.eval env))
  | .div left right => do
      let y ← right.eval env
      if y = 0 then none else pure ((← left.eval env) / y)
  | .sqrt value => do
      let x ← value.eval env
      if x < 0 then none else pure (Real.sqrt x)

/-- A rational expression, as a sampler's width needs it. -/
def RExpr.evalRat (env : LoopEnv) : RExpr → Option Rat
  | .rational value => some value
  | .ofInt value => do pure ((← value.eval env) : Rat)
  | .add left right => do pure ((← left.evalRat env) + (← right.evalRat env))
  | .sub left right => do pure ((← left.evalRat env) - (← right.evalRat env))
  | .mul left right => do pure ((← left.evalRat env) * (← right.evalRat env))
  | .div left right => do
      let y ← right.evalRat env
      if y = 0 then none else pure ((← left.evalRat env) / y)
  | .sqrt _ => none

/-- The CRT moduli and the dimension of a ring. -/
def RingE.eval (env : LoopEnv) : RingE → Option (List Nat × Nat)
  | .explicit moduli dimension => do
      let values ← moduli.mapM (·.evalNat env)
      pure (values, dimension)
  | .slice source start stop => do
      let (moduli, dimension) ← source.eval env
      let a ← start.evalNat env
      let b ← stop.evalNat env
      if a ≤ b ∧ b ≤ moduli.length then pure ((moduli.drop a).take (b - a), dimension) else none
  | .select source indices => do
      let (moduli, dimension) ← source.eval env
      let picked ← indices.mapM fun index ↦ do moduli[← index.evalNat env]?
      pure (picked, dimension)
  | .concat left right => do
      let (l, d) ← left.eval env
      let (r, e) ← right.eval env
      if d = e then pure (l ++ r, d) else none

/-- A matrix type: its moduli, dimension, rows, and columns. -/
def MatrixTy.eval (env : LoopEnv) (type : MatrixTy) : Option (List Nat × Nat × Nat × Nat) := do
  let (moduli, n) ← type.ring.eval env
  pure (moduli, n, ← type.rows.evalNat env, ← type.columns.evalNat env)

/-- The secret of a trapdoor value, which a trapdoor operation keeps with the public matrix: the
backend layout of a public gadget trapdoor, or the layout and the short matrices of a sampled one
(`r` and `e` of `trapGenPublic`, or `t` of `statTrapGenPublic`) with the width of its preimages. A program obtains one only from
`gadgetTrapdoor` or `trapdoorSample`, so it holds a secret only for a matrix it generated. -/
inductive TrapdoorSecret (q n rows : Nat) where
  | gadget (layout : RegularLayout q)
  | sampled (layout : RegularLayout q) (r e : ExactMatrix q n rows (rows * layout.digitCount))
      (width : Rat)
  | statistical (layout : RegularLayout q) (mBar : Nat)
      (t : ExactMatrix q n mBar (rows * layout.digitCount)) (width : Rat)

/-- A runtime value. A matrix carries the modulus `q` of its ring `Z_q[X]/(X^n + 1)`, which
determines the ring; a CRT basis is a choice of representation, which an operation that depends
on it reads from its types. Compact and preimage matrices are matrices whose coefficients the
runtime stores in a smaller form. -/
inductive Value where
  | int (value : Int)
  | real (value : ℝ)
  | bool (value : Bool)
  | bytes (value : List UInt8)
  | matrix (q n rows columns : Nat) (value : ExactMatrix q n rows columns)
  | family (members : List Value)
  | trapdoor (q n rows columns : Nat) (publicMatrix : ExactMatrix q n rows columns)
      (secret : TrapdoorSecret q n rows)

instance : Inhabited Value := ⟨.int 0⟩

namespace Value

def toInt? : Value → Option Int
  | .int value => some value
  | _ => none

def toBool? : Value → Option Bool
  | .bool value => some value
  | _ => none

def toFamily? : Value → Option (List Value)
  | .family members => some members
  | _ => none

/-- `f` on two matrices of the same ring and shape. -/
def zipMatrix (f : ∀ {q n r c : Nat}, ExactMatrix q n r c → ExactMatrix q n r c → ExactMatrix q n r c) :
    Value → Value → Option Value
  | .matrix m₁ n₁ r₁ c₁ a, .matrix m₂ n₂ r₂ c₂ b =>
      if h : m₂ = m₁ ∧ n₂ = n₁ ∧ r₂ = r₁ ∧ c₂ = c₁ then
        match m₂, n₂, r₂, c₂, h, b with
        | _, _, _, _, ⟨rfl, rfl, rfl, rfl⟩, b => some (.matrix m₁ n₁ r₁ c₁ (f a b))
      else none
  | _, _ => none

/-- `f` on one matrix. -/
def mapMatrix (f : ∀ {q n r c : Nat}, ExactMatrix q n r c → ExactMatrix q n r c) :
    Value → Option Value
  | .matrix m n r c a => some (.matrix m n r c (f a))
  | _ => none

/-- The product of two matrices, with a `1 × 1` factor multiplying every entry of the other as
the runtime does. -/
noncomputable def mulMatrix : Value → Value → Option Value
  | .matrix m₁ n₁ r₁ c₁ a, .matrix m₂ n₂ r₂ c₂ b =>
      if h : m₂ = m₁ ∧ n₂ = n₁ then
        match m₂, n₂, h, b with
        | _, _, ⟨rfl, rfl⟩, b =>
          if hl : r₁ = 1 ∧ c₁ = 1 then
            some (.matrix m₁ n₁ r₂ c₂ fun i j ↦ a ⟨0, by omega⟩ ⟨0, by omega⟩ * b i j)
          else if hr : r₂ = 1 ∧ c₂ = 1 then
            some (.matrix m₁ n₁ r₁ c₁ fun i j ↦ a i j * b ⟨0, by omega⟩ ⟨0, by omega⟩)
          else if hk : r₂ = c₁ then
            match r₂, hk, b with
            | _, rfl, b => some (.matrix m₁ n₁ r₁ c₂ (a * b))
          else none
      else none
  | _, _ => none

end Value

/-- The entries `[start, stop)` of `count` positions, as `Fin` indices of a slice. -/
def sliceIndex {count : Nat} (start : Nat) {width : Nat} (h : start + width ≤ count) (i : Fin width) :
    Fin count :=
  ⟨start + i.val, by omega⟩

/-- A slice of a matrix. -/
noncomputable def sliceValue (env : LoopEnv) (rows columns : Option (IExpr × IExpr)) : Value → Option Value
  | .matrix m n r c a => do
      let (r₀, r₁) ← match rows with
        | none => some (0, r)
        | some (start, stop) => do pure (← start.evalNat env, ← stop.evalNat env)
      let (c₀, c₁) ← match columns with
        | none => some (0, c)
        | some (start, stop) => do pure (← start.evalNat env, ← stop.evalNat env)
      if h : r₀ ≤ r₁ ∧ r₁ ≤ r ∧ c₀ ≤ c₁ ∧ c₁ ≤ c then
        pure (.matrix m n (r₁ - r₀) (c₁ - c₀) fun i j ↦
          a (sliceIndex r₀ (by omega) i) (sliceIndex c₀ (by omega) j))
      else none
  | _ => none

/-- Two matrices stacked along an axis. -/
noncomputable def concatValue (axis : Axis) : Value → Value → Option Value
  | .matrix m₁ n₁ r₁ c₁ a, .matrix m₂ n₂ r₂ c₂ b =>
      if h : m₂ = m₁ ∧ n₂ = n₁ then
        match m₂, n₂, h, b with
        | _, _, ⟨rfl, rfl⟩, b =>
          match axis with
          | .rows =>
            if hc : c₂ = c₁ then
              match c₂, hc, b with
              | _, rfl, b => some (.matrix m₁ n₁ (r₁ + r₂) c₁ fun i j ↦
                  if h : i.val < r₁ then a ⟨i.val, h⟩ j else b ⟨i.val - r₁, by omega⟩ j)
            else none
          | .columns =>
            if hr : r₂ = r₁ then
              match r₂, hr, b with
              | _, rfl, b => some (.matrix m₁ n₁ r₁ (c₁ + c₂) fun i j ↦
                  if h : j.val < c₁ then a i ⟨j.val, h⟩ else b i ⟨j.val - c₁, by omega⟩)
            else none
          | .diagonal => some (.matrix m₁ n₁ (r₁ + r₂) (c₁ + c₂) fun i j ↦
              if hi : i.val < r₁ then
                if hj : j.val < c₁ then a ⟨i.val, hi⟩ ⟨j.val, hj⟩ else 0
              else if hj : j.val < c₁ then 0
              else b ⟨i.val - r₁, by omega⟩ ⟨j.val - c₁, by omega⟩)
      else none
  | _, _ => none

/-- A matrix of `rows × columns` entries whose coefficients are drawn from `law` and reduced
modulo `q`. -/
noncomputable def sampleValue (law : SamplerLaw) (q n rows columns : Nat) : Comp Value := do
  let value ← sampleMatrix law q n rows columns
  pure (.matrix q n rows columns value)

/-- The small gadget matrix of a layout: `I_rows ⊗ (1, b, b², …)` with one entry per digit of a
CRT limb. -/
noncomputable def smallGadgetMatrix {q n rows : Nat} (layout : RegularLayout q) :
    ExactMatrix q n rows (rows * layout.digitsPerTower) :=
  fun row column ↦
    let (row', digit) := (finProdFinEquiv : Fin rows × Fin layout.digitsPerTower ≃ _).symm column
    if row = row' then ((layout.base ^ digit.val : Nat) : ExactPoly q n) else 0

/-- A constant matrix. The gadget matrices are those of the backend layout of their ring. -/
noncomputable def constMatrixValue (ctx : Context) (env : LoopEnv) (type : MatrixTy) :
    ConstMatrix → Option Value
  | .zero => do
      let (m, n, r, c) ← type.eval env
      pure (.matrix m.prod n r c 0)
  | .identity => do
      let (m, n, r, c) ← type.eval env
      if h : r = c then
        match c, h with
        | _, rfl => pure (.matrix m.prod n r r 1)
      else none
  | .polynomial coefficients => do
      let (m, n, r, c) ← type.eval env
      let values ← coefficients.mapM (·.eval env)
      if r = 1 ∧ c = 1 ∧ values.length ≤ n then
        pure (.matrix m.prod n r c fun _ _ ↦ polynomialOfCoefficients fun i ↦ values.getD i.val 0)
      else none
  | .packed width count table => do
      let (m, n, r, c) ← type.eval env
      if r = 1 ∧ c = 1 ∧ count ≤ n then
        pure (.matrix m.prod n r c fun _ _ ↦ polynomialOfCoefficients fun i ↦
          if i.val < count then (packedEntry width table i.val : Int) else 0)
      else none
  | .allCoefficients value => do
      let (m, n, r, c) ← type.eval env
      let x ← value.eval env
      pure (.matrix m.prod n r c fun _ _ ↦ polynomialOfCoefficients fun _ ↦ x)
  | .unitRow index => do
      let (m, n, r, c) ← type.eval env
      let i ← index.eval env
      if r = 1 then pure (.matrix m.prod n 1 c (unitRow i)) else none
  | .unitColumn index => do
      let (m, n, r, c) ← type.eval env
      let i ← index.eval env
      if c = 1 then pure (.matrix m.prod n r 1 (unitColumn i)) else none
  | .powerOfBase base exponent => do
      let (m, n, r, c) ← type.eval env
      let b ← base.eval env
      let e ← exponent.evalNat env
      if r = 1 ∧ c = 1 then pure (.matrix m.prod n 1 1 (liftInteger (b ^ e))) else none
  | .rotation exponent => do
      let (m, n, r, c) ← type.eval env
      let e ← exponent.eval env
      if r = 1 ∧ c = 1 ∧ 0 ≤ e then pure (.matrix m.prod n 1 1 (rotationPolynomial e)) else none
  | .gadget base small => do
      let (m, n, r, _) ← type.eval env
      let layout ← ctx.backend.layouts.regularLayout m.prod n
      let b ← base.eval env
      if b ≠ layout.base then none
      else if small then pure (.matrix m.prod n r (r * layout.digitsPerTower) (smallGadgetMatrix layout))
      else pure (.matrix m.prod n r (r * layout.digitCount) (regularGadgetMatrix layout))

/-- The slot that holds evaluation `i` of a transform of `2^bits` points: `bitReverse bits i`,
or `0` outside the transform. -/
def nttSlot (bits : Nat) {n : Nat} (i : Fin n) : Fin n :=
  if h : nttBitReverse bits i.val < n then ⟨_, h⟩ else i

/-- The root of the transform of `n` points over a squarefree modulus `q`: the CRT lift of the
native root (`nativeNttRoot`) of every prime factor, which is that root for a prime `q`. -/
noncomputable def nttRoot (q n : Nat) : Nat :=
  (∑ p ∈ q.primeFactors, nativeNttRoot p n * (q / p) * (((q / p : Nat) : ZMod p)⁻¹).val) % q

/-- The negacyclic NTT over `q` at its root `ω` (`nttRoot`): slot `bitReverse bits i` holds
`Σ_k c_k ω^((2i+1)k)`. Bit reversal is an involution, so slot `s` holds the evaluation
`bitReverse bits s`. Over a CRT basis this is the native transform of every limb. -/
@[irreducible] noncomputable def forwardNtt (p bits : Nat) {n : Nat} (coefficients : Fin n → ZMod p) :
    Fin n → ZMod p :=
  fun s ↦ ∑ k : Fin n, coefficients k *
    (nttRoot p n : ZMod p) ^ ((2 * (nttSlot bits s).val + 1) * k.val)

/-- The inverse of `forwardNtt`: `c_k = n⁻¹ Σ_i v_(bitReverse i) ω^(-(2i+1)k)`. -/
@[irreducible] noncomputable def inverseNtt (p bits : Nat) {n : Nat} (values : Fin n → ZMod p) :
    Fin n → ZMod p :=
  fun k ↦ (n : ZMod p)⁻¹ * ∑ i : Fin n, values (nttSlot bits i) *
    ((nttRoot p n : ZMod p)⁻¹) ^ ((2 * i.val + 1) * k.val)

/-- The number of bits of a transform of `n` points over `q` that has one: `q` is squarefree,
`n` is a power of two, and `2n` divides `p - 1` for every prime `p` of `q`. -/
@[irreducible] def nttBits? (q n : Nat) : Option Nat :=
  if 1 < q ∧ q.primeFactorsList.Nodup ∧ n = 2 ^ Nat.log2 n ∧ ∀ p ∈ q.primeFactors, 2 * n ∣ p - 1 then
    some (Nat.log2 n)
  else none

/-- The integers of a family of exactly `n` integers. -/
def intValues? (n : Nat) (members : List Value) : Option (Fin n → Int) := do
  let values ← members.mapM Value.toInt?
  if h : values.length = n then some fun i ↦ values[i.val]'(h ▸ i.isLt) else none

/-- The bound on the integers integer arithmetic returns. An integer operation costs one unit,
which is sound only for integers of bounded size: repeated squaring would otherwise reach
integers of exponential length at linear cost. -/
def intBound : Nat := 2 ^ 256

/-- The Kronecker product, left indices outer, as the runtime lays it out. -/
noncomputable def tensorMatrix {q n r₁ c₁ r₂ c₂ : Nat} (a : ExactMatrix q n r₁ c₁)
    (b : ExactMatrix q n r₂ c₂) : ExactMatrix q n (r₁ * r₂) (c₁ * c₂) :=
  fun i j ↦
    let (i₁, i₂) := (finProdFinEquiv : Fin r₁ × Fin r₂ ≃ _).symm i
    let (j₁, j₂) := (finProdFinEquiv : Fin c₁ × Fin c₂ ≃ _).symm j
    a i₁ j₁ * b i₂ j₂

/-- `count` copies of a matrix on the diagonal, zero elsewhere. -/
noncomputable def blockDiagonalMatrix {q n rows columns : Nat} (count : Nat)
    (a : ExactMatrix q n rows columns) : ExactMatrix q n (rows * count) (columns * count) :=
  fun i j ↦
    let (bi, ri) := (finProdFinEquiv : Fin count × Fin rows ≃ _).symm (finCongr (Nat.mul_comm _ _) i)
    let (bj, cj) :=
      (finProdFinEquiv : Fin count × Fin columns ≃ _).symm (finCongr (Nat.mul_comm _ _) j)
    if bi = bj then a ri cj else 0

/-- The CRT digits of every entry, digit-major (`rnsModUpRuns`). -/
noncomputable def rnsModUpMatrix {q p n rows columns : Nat} (basis : List Nat) (digitSize : Nat)
    (normalize : Bool) (digits : Nat) (a : ExactMatrix q n rows columns) :
    ExactMatrix p n (digits * rows) columns :=
  fun i j ↦
    let (digit, row) := (finProdFinEquiv : Fin digits × Fin rows ≃ _).symm i
    polynomialOfCoefficients fun k ↦
      rnsDigit basis digitSize digit.val normalize (Int.ofNat ((a row j).coeff k).val)

/-- The BGV modulus reduction by the dropped limbs (`rnsModDownRuns`). -/
noncomputable def rnsModDownMatrix {q p n rows columns : Nat} (basis : List Nat) (plaintext : Int)
    (a : ExactMatrix q n rows columns) : ExactMatrix p n rows columns :=
  fun row column ↦ polynomialOfCoefficients fun i ↦
    let dropped := basis.filter (fun prime ↦ p % prime != 0)
    let auxiliary := dropped.prod
    let value := Int.ofNat ((a row column).coeff i).val
    let correction := (dropped.map fun prime ↦
      let factor := auxiliary / prime
      Int.ofNat factor * rnsCentered prime
        (-value * rnsInverse prime plaintext * rnsInverse prime factor)).sum
    (value + plaintext * correction) * rnsInverse p auxiliary

/-- Exact block modulus switching (`blockModSwitchRuns`). -/
noncomputable def blockModSwitchMatrix {q p n rows columns : Nat} (basis : List Nat)
    (plaintext : Int) (a : ExactMatrix q n rows columns) : ExactMatrix p n rows columns :=
  fun row column ↦ polynomialOfCoefficients fun i ↦
    let auxiliary := (basis.filter (fun prime ↦ p % prime != 0)).prod
    let z := Int.ofNat ((a row column).coeff i).val
    let correction := rnsCentered auxiliary (z * rnsInverse auxiliary plaintext)
    ((z - plaintext * correction) / Int.ofNat auxiliary) % Int.ofNat p

/-- Whether `basis` is a valid CRT basis of `q` for rings of dimension `n`: distinct coprime
NTT-friendly primes with product `q`. -/
def crtBasisValid (basis : List Nat) (q n : Nat) : Prop :=
  basis.prod = q ∧ basis ≠ [] ∧ basis.Pairwise Nat.Coprime ∧
    ∀ prime ∈ basis, 2 < prime ∧ 2 * n ∣ prime - 1

instance (basis : List Nat) (q n : Nat) : Decidable (crtBasisValid basis q n) := by
  unfold crtBasisValid; infer_instance

/-- The decoded value of coefficient `k` (`thresholdDecode`). -/
noncomputable def thresholdValue {q n : Nat} (plaintext : Int) (a : ExactMatrix q n 1 1)
    (k : Fin n) : Int :=
  ((plaintext * (((a 0 0).coeff k).val : Int) + (q : Int) / 2) / (q : Int)) % plaintext

/-- The typed components of a hash tag; an operand component reads an integer argument. -/
def tagComponents (env : LoopEnv) (args : List Value) :
    List TagPart → Option (List HashTagComponent)
  | [] => some []
  | part :: rest => do
      let component ← match part with
        | .bytes value => some (.bytes value)
        | .integer value => do pure (.integer (← value.eval env))
        | .decimal value => do pure (.decimal (← value.eval env))
        | .u64le value => do
            let v ← value.eval env
            if 0 ≤ v ∧ v < 2 ^ 64 then pure (.u64Le v) else none
        | .operand index => do pure (.integer (← (← args[index]?).toInt?))
        | .operandU64le index => do
            let v ← (← args[index]?).toInt?
            if 0 ≤ v ∧ v < 2 ^ 64 then pure (.u64Le v) else none
      pure (component :: (← tagComponents env args rest))

/-- The 32-byte key of a hash operation. -/
def hashKey? : Value → Option ByteArray
  | .bytes value => if value.length = 32 then some ⟨value.toArray⟩ else none
  | _ => none

/-- One level of a CRT recomposition into `ExactMatrix d n 1 columns`. -/
noncomputable def crtTerm (d n columns : Nat) (plaintext coefficient : Int) :
    Value → Option (ExactMatrix d n 1 columns)
  | .matrix _ n' r c a =>
      if h : n' = n ∧ r = 1 ∧ c = columns then
        match n', r, c, h, a with
        | _, _, _, ⟨rfl, rfl, rfl⟩, a => some (crtRecomposeLevel plaintext coefficient a)
      else none
  | _ => none

/-- `bias + Σ_t c_t · left_t · right_t`, with the arguments `left_0, right_0, left_1, …, bias`. -/
noncomputable def mulAccumulate (env : LoopEnv) (coefficients : List IExpr) (hasBias : Bool)
    (args : List Value) : Option Value := do
  let scales ← coefficients.mapM (·.eval env)
  let count := scales.length
  if count = 0 ∨ args.length ≠ 2 * count + (if hasBias then 1 else 0) then none
  let products ← (List.range count).mapM fun t ↦ do
    let product ← Value.mulMatrix (← args[2 * t]?) (← args[2 * t + 1]?)
    Value.mapMatrix (matrixScale (← scales[t]?)) product
  let first ← products.head?
  let sum ← products.tail.foldlM (fun total term ↦ Value.zipMatrix (· + ·) total term) first
  if hasBias then Value.zipMatrix (· + ·) sum (← args[2 * count]?) else pure sum

/-- A gadget decomposition by the backend layout of its ring: the regular one, tower-major with
`digitCount` digits, or the small one, one row per digit. -/
noncomputable def decomposeValue (ctx : Context) (base digits : Int) (small : Bool) :
    Value → Option Value
  | .matrix q n r c a => do
      let layout ← ctx.backend.layouts.regularLayout q n
      if base ≠ layout.base then none
      else if small then
        if digits = layout.digitsPerTower then
          pure (.matrix q n (r * layout.digitsPerTower) c (smallDecomposeMatrix layout a))
        else none
      else if digits = layout.digitCount then
        pure (.matrix q n (r * layout.digitCount) c (regularDecomposeMatrix layout a))
      else none
  | _ => none

/-- The deterministic operations: their outputs from the values of their arguments. -/
noncomputable def applyPure (ctx : Context) (env : LoopEnv) (op : Op) (args : List Value) :
    Option (List Value) :=
  match op, args with
  | .constInt value, [] => some [.int value]
  | .evaluateInt value, [] => do pure [.int (← value.eval env)]
  | .constReal value, [] => do pure [.real (← value.eval env)]
  | .constBool value, [] => some [.bool value]
  | .constMatrix type value, [] => do pure [← constMatrixValue ctx env type value]
  | .intBinary op, [.int x, .int y] => do
      let z ← match op with
        | .add => some (x + y)
        | .sub => some (x - y)
        | .mul => some (x * y)
        | .div => if y = 0 then none else some (Int.fdiv x y)
        | .rem => if y = 0 then none else some (Int.fmod x y)
      if z.natAbs < intBound then some [.int z] else none
  | .intCompare op, [.int x, .int y] =>
      match op with
      | .eq => some [.bool (decide (x = y))]
      | .lt => some [.bool (decide (x < y))]
      | .le => some [.bool (decide (x ≤ y))]
  | .intToReal, [.int x] => some [.real x]
  | .boolToInt, [.bool b] => some [.int (if b then 1 else 0)]
  | .realBinary op, [.real x, .real y] =>
      match op with
      | .add => some [.real (x + y)]
      | .sub => some [.real (x - y)]
      | .mul => some [.real (x * y)]
      | .div => if y = 0 then none else some [.real (x / y)]
  | .realSqrt, [.real x] => if x < 0 then none else some [.real (Real.sqrt x)]
  | .matrixBinary .add, [a, b] => do pure [← Value.zipMatrix (· + ·) a b]
  | .matrixBinary .sub, [a, b] => do pure [← Value.zipMatrix (· - ·) a b]
  | .matrixBinary .mul, [a, b] => do pure [← Value.mulMatrix a b]
  | .matrixMulSmallRhs, [a, b] => do pure [← Value.mulMatrix a b]
  | .matrixNegate, [a] => do pure [← Value.mapMatrix (- ·) a]
  | .matrixScale scalar, [a] => do
      let s ← scalar.eval env
      pure [← Value.mapMatrix (matrixScale s) a]
  | .transpose, [.matrix m n r c a] => some [.matrix m n c r a.transpose]
  | .slice rows columns, [a] => do pure [← sliceValue env rows columns a]
  | .concat axis, [a, b] => do pure [← concatValue axis a b]
  | .modulusReduce destination, [.matrix m n r c a] => do
      let (m', n') ← destination.eval env
      if 1 < m'.prod ∧ m'.prod ∣ m ∧ n' = n then
        pure [.matrix m'.prod n r c (modulusReduce a)]
      else none
  | .centeredRebase destination, [.matrix _q n r c a] => do
      let (m', n') ← destination.eval env
      if 1 < m'.prod ∧ n' = n then pure [.matrix m'.prod n r c (centeredRebase a)] else none
  | .familyPack count, members => do
      let k ← count.evalNat env
      if members.length = k then pure [.family members] else none
  | .familyGetStatic index, [.family members] => do pure [← members[← index.evalNat env]?]
  | .familyGetDynamic, [.family members, .int index] =>
      if index < 0 then none else do pure [← members[index.toNat]?]
  | .select count, .int index :: choices => do
      let k ← count.evalNat env
      if choices.length = k ∧ 0 ≤ index then do pure [← choices[index.toNat]?] else none
  | .polynomialValues false, [.matrix _q n 1 1 a] =>
      some [.family (List.ofFn fun i : Fin n ↦ .int ((a 0 0).coeff i).val)]
  | .polynomialValues true, [.matrix q n 1 1 a] => do
      -- The transform is defined over a prime modulus; a CRT basis is not covered yet.
      let bits ← nttBits? q n
      let values := forwardNtt q bits fun k ↦ (((a 0 0).coeff k).val : ZMod q)
      pure [.family (List.ofFn fun s : Fin n ↦ .int (values s).val)]
  | .polynomialFromValues type true, [.family members] => do
      let (m, n, r, c) ← type.eval env
      let values ← intValues? n members
      let bits ← nttBits? m.prod n
      if r = 1 ∧ c = 1 then
        pure [.matrix m.prod n r c fun _ _ ↦ polynomialOfCoefficients fun k ↦
          ((inverseNtt m.prod bits fun s ↦ (values s : ZMod m.prod)) k).val]
      else none
  | .polynomialFromValues type false, [.family members] => do
      let (m, n, r, c) ← type.eval env
      let values ← members.mapM Value.toInt?
      if r = 1 ∧ c = 1 ∧ values.length = n then
        pure [.matrix m.prod n r c fun _ _ ↦ polynomialOfCoefficients fun i ↦ values.getD i.val 0]
      else none
  | .bitExtract bit, [.int x] => do
      let b ← bit.evalNat env
      pure [.bool (decide ((x / 2 ^ b) % 2 = 1))]
  | .matrixMulAccumulate coefficients hasBias, args => do
      pure [← mulAccumulate env coefficients hasBias args]
  | .ringAutomorphism index, [.matrix q n r c a] => do
      let k ← index.eval env
      if 0 < k ∧ k < 2 * (n : Int) ∧ k % 2 = 1 then
        pure [.matrix q n r c (ringAutomorphism k.toNat a)]
      else none
  | .multiplyMonomial, [.matrix q n r c a, .int k] => some [.matrix q n r c (multiplyMonomial a k)]
  | .modulusSwitch destination, [.matrix q n r c a] => do
      let (m', n') ← destination.eval env
      let p := m'.prod
      if 1 < p ∧ p ∣ q ∧ q % 2 = 1 ∧ p % 2 = 1 ∧ n' = n then
        pure [.matrix p n r c (modulusSwitch a)]
      else none
  | .centeredRoundDivide divisor, [.matrix q n r c a] => do
      let d ← divisor.eval env
      if 1 < q ∧ 0 < d then pure [.matrix q n r c (centeredRoundDivide d a)] else none
  | .rnsModUp source destination digitSize normalize, [.matrix q n r c a] => do
      let (basis, _) ← source.eval env
      let (m', n') ← destination.eval env
      let p := m'.prod
      let digits := (basis.length + digitSize - 1) / digitSize
      if crtBasisValid basis q n ∧ 0 < digitSize ∧ 1 < p ∧ q ∣ p ∧ n' = n then
        pure [.matrix p n (digits * r) c (rnsModUpMatrix basis digitSize normalize digits a)]
      else none
  | .rnsModDown destination plaintext, [.matrix q n r c a] => do
      let (m', n') ← destination.eval env
      let t ← plaintext.eval env
      let p := m'.prod
      let basis := q.primeFactorsList
      let auxiliary := (basis.filter (fun prime ↦ p % prime != 0)).prod
      if crtBasisValid basis q n ∧ 1 < p ∧ p < q ∧ p * auxiliary = q ∧ 2 ≤ t ∧
          Int.gcd t auxiliary = 1 ∧ n' = n then
        pure [.matrix p n r c (rnsModDownMatrix basis t a)]
      else none
  | .blockModSwitch destination plaintext, [.matrix q n r c a] => do
      let (m', n') ← destination.eval env
      let t ← plaintext.eval env
      let p := m'.prod
      let basis := q.primeFactorsList
      let kept := basis.filter (fun prime ↦ p % prime == 0)
      let dropped := basis.filter (fun prime ↦ p % prime != 0)
      if crtBasisValid basis q n ∧ 1 < p ∧ p < q ∧ kept ≠ [] ∧ dropped ≠ [] ∧
          p * dropped.prod = q ∧ 0 < t ∧ Int.gcd t dropped.prod = 1 ∧ n' = n then
        pure [.matrix p n r c (blockModSwitchMatrix basis t a)]
      else none
  | .tensor, [.matrix q n r₁ c₁ a, .matrix q' n' r₂ c₂ b] =>
      if h : q' = q ∧ n' = n then
        match q', n', h, b with
        | _, _, ⟨rfl, rfl⟩, b => some [.matrix q n (r₁ * r₂) (c₁ * c₂) (tensorMatrix a b)]
      else none
  | .blockDiagonal count, [.matrix q n r c a] =>
      if 0 < count then some [.matrix q n (r * count) (c * count) (blockDiagonalMatrix count a)]
      else none
  | .gadgetDecompose base small digitCount, [a] => do
      pure [← decomposeValue ctx (← base.eval env) (← digitCount.eval env) small a]
  | .extractCoefficient position, [.matrix _ n 1 1 a] => do
      let k ← position.evalNat env
      if h : k < n then pure [.int ((a 0 0).coeff ⟨k, h⟩).val] else none
  | .liftIntegerToConstantPolynomial type, [.int x] => do
      let (m, n, r, c) ← type.eval env
      if r = 1 ∧ c = 1 then pure [.matrix m.prod n 1 1 (liftInteger x)] else none
  | .thresholdDecode plaintext length outputBool family, [.matrix q n 1 1 a] => do
      let t ← plaintext.eval env
      let count ← length.evalNat env
      if h : 0 < q ∧ 0 < t ∧ count ≤ n then
        let values := List.ofFn fun k : Fin count ↦ thresholdValue t a ⟨k.val, by omega⟩
        if family then pure [.family (values.map .int)]
        else if outputBool then pure (values.map fun v ↦ .bool (decide (v ≠ 0)))
        else pure (values.map .int)
      else none
  | .crtRecompose modulus plaintexts coefficients, args => do
      let d ← modulus.evalNat env
      let ts ← plaintexts.mapM (·.eval env)
      let cs ← coefficients.mapM (·.eval env)
      let (n, columns) ← match args.head? with
        | some (.matrix _ n 1 c _) => some (n, c)
        | _ => none
      if ts.length = args.length ∧ cs.length = args.length ∧ ∀ t ∈ ts, 0 < t then
        let terms ← ((ts.zip cs).zip args).mapM fun ((t, c), arg) ↦ crtTerm d n columns t c arg
        pure [.matrix d n 1 columns terms.sum]
      else none
  | .packPolynomialCoefficients type bitsPerCoefficient, [.family members] => do
      let (m, n, r, c) ← type.eval env
      let width ← bitsPerCoefficient.eval env
      let bits ← members.mapM Value.toBool?
      let w := width.toNat
      let coefficient (i : Nat) : Nat := ∑ b ∈ Finset.range w,
        if bits.getD (i * w + b) false then 2 ^ b else 0
      if 0 < width ∧ r = 1 ∧ c = 1 ∧ bits.length = n * w ∧
          ∀ i < n, coefficient i < m.prod then
        pure [.matrix m.prod n 1 1 fun _ _ ↦
          polynomialOfCoefficients fun i ↦ Int.ofNat (coefficient i.val)]
      else none
  | .intMatrixVectorProduct transpose, [.family matrix, .family vector] => do
      let entries ← matrix.mapM Value.toInt?
      let values ← vector.mapM Value.toInt?
      let inner := values.length
      let outer := entries.length / inner
      if 0 < inner ∧ entries.length = outer * inner then
        let product := MxxRuntime.intMatrixVectorProduct (outer := outer) transpose
          (fun i : Fin entries.length ↦ entries[i]) (fun j : Fin inner ↦ values[j])
        pure [.family (List.ofFn fun i ↦ .int (product i))]
      else none
  | .sha256BitColumns seed rows columns indexBits, [] =>
      if seed.length = 32 ∧ 0 < rows ∧ 0 < columns ∧ 0 < indexBits ∧ rows ≤ 2 ^ indexBits then
        let result (i : Fin columns) := Sha256Columns.column seed rows columns indexBits i.val
        some [.family (List.ofFn fun i ↦ .int (result i).1),
          .family (List.ofFn fun i ↦ .int (result i).2)]
      else none
  | .hashSample type variant tagPrefix tag base digitCount, key :: _ => do
      let (m, n, r, c) ← type.eval env
      let q := m.prod
      let k ← hashKey? key
      let tagBytes := completeHashTag tagPrefix (← tagComponents env args tag)
      match variant, base, digitCount with
      | .plain, none, none => pure [.matrix q n r c (ctx.hash.sample q n r c k tagBytes)]
      | .decomposed, some b, some d | .smallDecomposed, some b, some d => do
          let digits ← d.evalNat env
          if digits = 0 ∨ r % digits ≠ 0 then none
          let source := Value.matrix q n (r / digits) c
            (ctx.hash.sample q n (r / digits) c k tagBytes)
          pure [← decomposeValue ctx (← b.eval env) digits (variant matches .smallDecomposed)
            source]
      | _, _, _ => none
  | .hashIntFamily count modulus tagPrefix tag, key :: _ => do
      let k ← count.evalNat env
      let modulusValue ← modulus.eval env
      let key ← hashKey? key
      let tagBytes := completeHashTag tagPrefix (← tagComponents env args tag)
      if 1 < modulusValue ∧ modulusValue = 2 ^ Nat.log2 modulusValue.toNat then
        pure [.family (List.ofFn fun i : Fin k ↦ .int
          ((ctx.hash.integers k modulusValue.toNat key tagBytes i % modulusValue.toNat : Nat) :
            Int))]
      else none
  | .gadgetTrapdoor type base, [] => do
      let (m, n, r, _) ← type.eval env
      let layout ← ctx.backend.layouts.regularLayout m.prod n
      if (← base.eval env) = layout.base then
        pure [.trapdoor m.prod n r (r * layout.digitCount) (regularGadgetMatrix layout)
          (.gadget layout)]
      else none
  | .trapdoorPublic, [.trapdoor q n r c publicMatrix _] => some [.matrix q n r c publicMatrix]
  | _, _ => none

/-- The sampling operations. -/
noncomputable def applySample (_ctx : Context) (env : LoopEnv) (op : Op) :
    Option (Comp Value) :=
  match op with
  | .uniformResidueSample type => do
      let (m, n, r, c) ← type.eval env
      pure (sampleValue (.residue m.prod) m.prod n r c)
  | .uniformIntervalSample type minimum maximum => do
      let (m, n, r, c) ← type.eval env
      pure (sampleValue (.interval (← minimum.eval env) (← maximum.eval env)) m.prod n r c)
  | .gaussianSample type sigma bound _ => do
      let (m, n, r, c) ← type.eval env
      pure (sampleValue (.gaussian (← sigma.evalRat env) (← bound.eval env)) m.prod n r c)
  | _ => none

/-- The value of `ref` among the outputs of the earlier nodes. -/
def lookup (outputs : List (List Value)) (ref : Ref) : Option Value := do
  (← outputs[ref.node]?)[ref.port]?

/-- The arguments of iteration `i` of a parallel loop that reads `args` by `modes`. -/
def iterationArgs (modes : List LoopMode) (args : List Value) (i : Nat) : Option (List Value) := do
  let read (mode : LoopMode) (arg : Value) : Option Value :=
    match mode with
    | .broadcast => some arg
    | .zip => do (← arg.toFamily?)[i]?
    | .zipOffset offset => do (← arg.toFamily?)[i + offset]?
    | .zipQuotient divisor => if divisor = 0 then none else do (← arg.toFamily?)[i / divisor]?
    | .gather _ => none
  let direct ← (List.zipWith read modes args).mapM id
  -- A gathered argument reads the member its index argument names for this iteration.
  (List.zipWith (fun mode pair ↦
      match mode, pair with
      | .gather indexArgument, (_, arg) => do
          let index ← (← direct[indexArgument]?).toInt?
          if index < 0 then none else (← arg.toFamily?)[index.toNat]?
      | _, (value, _) => some value) modes (direct.zip args)).mapM id

/-- Run the iterations `start, start + 1, ...` of a parallel loop, collecting each output. -/
noncomputable def parallelIterations (body : LoopEnv → List Value → OptionT Comp (List Value))
    (env : LoopEnv) (slot : Nat) (modes : List LoopMode) (args : List Value) :
    Nat → Nat → OptionT Comp (List (List Value))
  | _, 0 => pure []
  | start, count + 1 => do
      let arguments ← OptionT.mk (pure (iterationArgs modes args start))
      let head ← body (Function.update env slot start) arguments
      let tail ← parallelIterations body env slot modes args (start + 1) count
      pure (head :: tail)

/-- A parallel loop whose iterations `start, ..., start + count - 1` return `f i` without
sampling returns them in order. -/
theorem parallelIterations_pure {body : LoopEnv → List Value → OptionT Comp (List Value)}
    {env : LoopEnv} {slot : Nat} {modes : List LoopMode} {args : List Value}
    {f : Nat → List Value} :
    ∀ count start, (∀ i, start ≤ i → i < start + count → ∃ arguments,
      iterationArgs modes args i = some arguments ∧
        body (Function.update env slot i) arguments = pure (f i)) →
    parallelIterations body env slot modes args start count =
      pure ((List.range count).map fun j ↦ f (start + j))
  | 0, _, _ => rfl
  | count + 1, start, h => by
    obtain ⟨arguments, ha, hb⟩ := h start le_rfl (by omega)
    have hmk : (OptionT.mk (pure (iterationArgs modes args start)) : OptionT Comp (List Value)) =
        pure arguments := by
      rw [ha]; rfl
    rw [parallelIterations, hmk, pure_bind, hb, pure_bind,
      parallelIterations_pure count (start + 1) fun i h₁ h₂ ↦ h i (by omega) (by omega),
      pure_bind, List.range_succ_eq_map, List.map_cons, List.map_map]
    congr 2
    · simp
      intro j _
      congr 1
      omega

/-- Run `count` iterations of a sequential loop from iteration `start`, updating the carried
values. -/
noncomputable def sequentialIterations (body : LoopEnv → List Value → OptionT Comp (List Value))
    (env : LoopEnv) (slot : Nat) (invariant : List Value) :
    Nat → Nat → List Value → OptionT Comp (List Value)
  | _, 0, carried => pure carried
  | start, count + 1, carried => do
      let next ← body (Function.update env slot start) (carried ++ invariant)
      sequentialIterations body env slot invariant (start + 1) count next

-- The loop runners recurse on their iteration count; keep them folded so that unification never
-- unrolls a loop of thousands of iterations. Their equations remain available to rewriting.
theorem sequentialIterations_zero (body : LoopEnv → List Value → OptionT Comp (List Value))
    (env : LoopEnv) (slot : Nat) (invariant : List Value) (start : Nat) (carried : List Value) :
    sequentialIterations body env slot invariant start 0 carried = pure carried := by
  rw [sequentialIterations]

attribute [irreducible] parallelIterations sequentialIterations

/-- A trapdoor draw, as the backends generate it: the public matrix `[ā | I | G - (ā r + e)]` of
`trapGenPublic`, or `[ā | G - ā t]` of `statTrapGenPublic` with `statisticalRows` columns of
`ā`, for the backend layout of the ring, with `r`, `e`, `t` Gaussian of width `sigma` cut at
`trapGenCutoff` and `ā` uniform. It returns the public matrix and the trapdoor that keeps the
short matrices and the width of its preimages: `preimageSigma`, or the backend's smallest. -/
noncomputable def trapdoorDraw (ctx : Context) (env : LoopEnv) (type : MatrixTy) (sigma : RExpr)
    (base digits : IExpr) (statisticalRows : Option Nat) (preimageSigma : Option RExpr) :
    Option (Comp (List Value)) := do
  let (m, n, r, _) ← type.eval env
  let q := m.prod
  let layout ← ctx.backend.layouts.regularLayout q n
  let s ← sigma.evalRat env
  if (← base.eval env) ≠ layout.base ∨ (← digits.eval env) ≠ layout.digitCount then none
  let preimageWidth ← match preimageSigma with
    | some value => value.evalRat env
    | none => some (ctx.backend.preimageWidth q n r layout.base s statisticalRows)
  let width := r * layout.digitCount
  match statisticalRows with
  | none => some do
      let short ← sampleMatrix (.gaussian s trapGenCutoff) q n r width
      let error ← sampleMatrix (.gaussian s trapGenCutoff) q n r width
      let aBar ← sampleMatrix (.residue q) q n r r
      let publicMatrix := trapGenPublic layout aBar short error
      pure [.matrix q n r (r * (layout.digitCount + 2)) publicMatrix,
        .trapdoor q n r (r * (layout.digitCount + 2)) publicMatrix
          (.sampled layout short error preimageWidth)]
  | some mBar => some do
      let short ← sampleMatrix (.gaussian s trapGenCutoff) q n mBar width
      let aBar ← sampleMatrix (.residue q) q n r mBar
      let publicMatrix := statTrapGenPublic layout aBar short
      pure [.matrix q n r (mBar + width) publicMatrix,
        .trapdoor q n r (mBar + width) publicMatrix
          (.statistical layout mBar short preimageWidth)]

open Classical in
/-- A preimage of `target` under the public matrix of a trapdoor, with coefficients at most `bound`:
the gadget decomposition of the target for a public gadget trapdoor, and a draw of the ideal
preimage sampler (`PreimageRequest`) at the trapdoor's width for a sampled one. The public matrix
argument must be the trapdoor's; a trapdoor exists only for a matrix the program generated. -/
noncomputable def preimageDraw (env : LoopEnv) (bound : IExpr) :
    List Value → OptionT Comp (List Value)
  | [.matrix q₀ n₀ r₀ c₀ publicArg, .trapdoor q n r c publicMatrix secret,
      .matrix q' n' r' columns target] =>
    if h : q₀ = q ∧ n₀ = n ∧ r₀ = r ∧ c₀ = c ∧ q' = q ∧ n' = n ∧ r' = r then
      match q₀, n₀, r₀, c₀, q', n', r', h, publicArg, target with
      | _, _, _, _, _, _, _, ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩, publicArg, target =>
        if publicArg ≠ publicMatrix then OptionT.fail
        else match secret with
          | .gadget layout =>
              if layout.droppedModuli = 0 then
                pure [.matrix q n (r * layout.digitCount) columns
                  (regularDecomposeMatrix layout target)]
              else OptionT.fail
          | .sampled _ _ _ width | .statistical _ _ _ width => do
              let b ← OptionT.mk (pure (bound.evalNat env))
              let request : PreimageRequest := ⟨q, n, r, c, columns, publicMatrix, target, width, b⟩
              let x ← (liftM (OracleSpec.query (spec := samplerSpec) (.preimage request)) :
                OptionT Comp request.Coefficients)
              if x ∈ request.preimages then pure [.matrix q n c columns (request.matrixOf x)]
              else OptionT.fail
    else OptionT.fail
  | _ => OptionT.fail

/-- Whether an operation runs a scope, reads an input, or draws several outputs, rather than
computing one draw or its outputs from its arguments by `applySample` or `applyPure`. -/
def Op.structural : Op → Bool
  | .input _ | .call _ | .parallelLoop .. | .sequentialLoop .. | .trapdoorSample ..
  | .preimageSample .. => true
  | _ => false

/-- The outputs of one node with operation `op`, `outputCount` outputs, and arguments `args`, where
`scopeEval body env args` runs scope `body` one level of nesting down. -/
noncomputable def nodeResult (ctx : Context) (scopeEval : Nat → LoopEnv → List Value → OptionT Comp (List Value))
    (env : LoopEnv) (inputs : List Value) (op : Op) (outputCount : Nat) (args : List Value) :
    OptionT Comp (List Value) :=
  match op with
  | .input position => OptionT.mk (pure (do pure [← inputs[position]?]))
  | .call body => scopeEval body env args
  | .parallelLoop body count slot modes => do
      let k ← OptionT.mk (pure (count.evalNat env))
      let rows ← parallelIterations (scopeEval body) env slot modes args 0 k
      pure ((List.range outputCount).map fun port ↦
        .family (rows.map fun row ↦ row.getD port default))
  | .sequentialLoop body count slot carried => do
      let k ← OptionT.mk (pure (count.evalNat env))
      sequentialIterations (scopeEval body) env slot (args.drop carried) 0 k (args.take carried)
  | .trapdoorSample type sigma base digits _ statisticalRows preimageSigma =>
      match trapdoorDraw ctx env type sigma base digits statisticalRows preimageSigma with
      | some draw => liftM draw
      | none => OptionT.fail
  | .preimageSample _ bound => preimageDraw env bound args
  | op => match applySample ctx env op with
    | some draw => do pure [← (draw : OptionT Comp Value)]
    | none => OptionT.mk (pure (applyPure ctx env op args))

/-- Whether `value` has type `ty`: the same kind, a matrix of the ring and shape the type names,
a byte string of its length, a family of its count whose members have its element type. The
cost of a program is computed from the types its nodes declare, so evaluation checks every
output against them. -/
def Value.conforms (env : LoopEnv) : Ty → Value → Bool
  | .int, .int _ => true
  | .real, .real _ => true
  | .bool, .bool _ => true
  | .bytes length, .bytes value => length.evalNat env == some value.length
  | .blob _, .bytes value => decide (value.length ≤ 32)
  | .matrix type, .matrix q n r c _ | .small type _ _, .matrix q n r c _
  | .preimage type _ _, .matrix q n r c _ | .trapdoor type _ _ _ _ _ _, .trapdoor q n r c _ _ =>
      match type.eval env with
      | some (m, n', r', c') => decide (m.prod = q ∧ n' = n ∧ r' = r ∧ c' = c)
      | none => false
  | .family element count, .family members =>
      count.evalNat env == some members.length && members.all (Value.conforms env element)
  | _, _ => false

/-- Whether `values` have the types `types`, one by one. -/
def Value.conformsAll (env : LoopEnv) (types : List Ty) (values : List Value) : Bool :=
  types.length == values.length && (types.zip values).all fun (ty, value) ↦ value.conforms env ty

/-- One node of a scope: its outputs appended to those of the earlier nodes, or failure if they
do not have the types the node declares. -/
noncomputable def stepNode (ctx : Context) (scopeEval : Nat → LoopEnv → List Value → OptionT Comp (List Value))
    (env : LoopEnv) (inputs : List Value) (outputs : List (List Value)) (node : Node) :
    OptionT Comp (List (List Value)) := do
  let args ← OptionT.mk (pure (node.args.mapM (lookup outputs)))
  let result ← nodeResult ctx scopeEval env inputs node.op node.outputs.length args
  if Value.conformsAll env node.outputs result then pure (outputs ++ [result]) else OptionT.fail

/-- The outputs of `scope` of `program` on `inputs`, with `fuel` bounding the nesting of loops
and calls. -/
noncomputable def evalScope (ctx : Context) (program : Program) : Nat → Nat → LoopEnv → List Value →
    OptionT Comp (List Value)
  | 0, _, _, _ => OptionT.fail
  | fuel + 1, index, env, inputs => do
      let scope ← OptionT.mk (pure program.scopes[index]?)
      let outputs ← scope.nodes.foldlM (stepNode ctx (evalScope ctx program fuel) env inputs) []
      OptionT.mk (pure (scope.outputs.mapM (lookup outputs)))

/-- The depth of nesting a program may use: one level per scope. -/
def Program.fuel (program : Program) : Nat := program.scopes.length + 1

/-- The root outputs of `program` on its root inputs, in the order of `Program.outputs`. -/
noncomputable def Program.run (program : Program) (ctx : Context) (inputs : List Value) :
    Comp (Option (List Value)) :=
  (evalScope ctx program program.fuel 0 (fun _ ↦ 0) inputs).run

end MxxSecurity.IR
