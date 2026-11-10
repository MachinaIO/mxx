import MxxSecurity.IR.Eval

/-!
# The cost of IR programs

`Program.cost` charges every node of a program by the table `Op.cost`, multiplies a loop body by
its iterations (summing them, since a body may depend on its index), and charges a call by its
body. The unit is one operation on one 64-bit word of a coefficient residue, one integer (of at
most `intBound`), one real, or one sampler draw. Evaluation checks every value against the type
its node declares (`Value.conforms`), so the types the cost reads are the sizes evaluation
handles.

The table is part of the trusted statement: a statement that bounds the cost of a program is only
as meaningful as the table. Every entry is an upper bound for a straightforward implementation
over the operation's arguments and outputs, generous where an implementation may do more work
than its outputs suggest:

* reading, copying, or moving coefficients (inputs, constants, slices, concatenations, families,
  reductions, rebasing, scaling, addition) costs the coefficients of the arguments plus those of
  the outputs;
* a product of polynomials costs `n²` per product of entries and word of the CRT limbs
  (schoolbook multiplication), so `r × k` by `k × c` matrices cost `r k c n² words`;
* a number-theoretic transform costs `n²` per polynomial and word (the naive transform);
* changing a CRT basis costs the input coefficients times the source and destination words plus
  one;
* a sampler costs one draw per output coefficient;
* a Kronecker product costs `n²` per product of entries;
* SHA-256 selector columns cost one unit per selector bit;
* gadget decomposition, hashing, gadget trapdoors, trapdoor sampling, and preimages cost the
  square of their output words, which bounds the products and digit extractions of their
  algorithms; a preimage under a sampled trapdoor is one draw of the ideal preimage sampler,
  charged like the perturbation and gadget sampling the runtime performs for it.
-/

namespace MxxSecurity.IR

/-- The number of 64-bit words of `m`, at least one; `fuel` bounds the recursion. -/
def wordCount : Nat → Nat → Nat
  | 0, _ => 1
  | fuel + 1, m => if m < 2 ^ 64 then 1 else 1 + wordCount fuel (m / 2 ^ 64)

/-- The words of a residue in a CRT basis: the words of its limbs, at least one. A residue is
charged by its words, so a large modulus does not make arithmetic on long integers cheap. A limb
`m` has fewer than `m` words, so `m` is enough fuel. -/
def limbWords (moduli : List Nat) : Nat :=
  max 1 (moduli.map fun m ↦ wordCount m m).sum

/-- The number of coefficient residues, integers, reals, or bytes of a value of type `ty`. -/
def Ty.size (env : LoopEnv) : Ty → Option Nat
  | .int | .real | .bool => some 1
  | .bytes length => length.evalNat env
  | .blob _ => some 1
  | .matrix type | .trapdoor type .. | .small type .. | .preimage type .. => do
      let (moduli, n, r, c) ← type.eval env
      pure (r * c * n * limbWords moduli)
  | .family element count => do pure ((← count.evalNat env) * (← element.size env))

/-- The dimension and CRT limbs of a matrix type, or `(1, 1)` for any other type. -/
def Ty.ringSize (env : LoopEnv) : Ty → Option (Nat × Nat)
  | .matrix type | .trapdoor type .. | .small type .. | .preimage type .. => do
      let (moduli, n, _, _) ← type.eval env
      pure (n, limbWords moduli)
  | _ => some (1, 1)

/-- The rows and columns of a matrix type, or `(1, 1)` for any other type. -/
def Ty.shape (env : LoopEnv) : Ty → Option (Nat × Nat)
  | .matrix type | .trapdoor type .. | .small type .. | .preimage type .. => do
      let (_, _, r, c) ← type.eval env
      pure (r, c)
  | _ => some (1, 1)

def sumSizes (env : LoopEnv) (types : List Ty) : Option Nat := do
  let sizes ← types.mapM (·.size env)
  pure sizes.sum

/-- The cost of hashing one tag component: its bytes, or one unit for a bounded integer. -/
def TagPart.cost : TagPart → Nat
  | .bytes value => value.length
  | _ => 1

/-- The size of what an operation draws or builds as its own parameters and the types of its
arguments determine, before evaluation checks its outputs against their declared types: the
draws of a sampler or a trapdoor generation, the outputs of a hash, a constant, or a preimage. -/
def Op.intrinsic (env : LoopEnv) (op : Op) (args : List Ty) : Option Nat :=
  match op with
  | .uniformResidueSample type | .uniformIntervalSample type .. | .gaussianSample type ..
  | .hashSample type .. | .constMatrix type _ | .liftIntegerToConstantPolynomial type
  | .packPolynomialCoefficients type _ | .polynomialFromValues type _ => (Ty.matrix type).size env
  | .hashIntFamily count .. => count.evalNat env
  | .trapdoorSample type _ _ digitCount _ statisticalRows _ => do
      let (moduli, n, r, _) ← type.eval env
      let width := r * (← digitCount.evalNat env)
      -- `r`, `e`, and `ā` of `trapGenPublic`, or `t` and `ā` of `statTrapGenPublic`.
      let draws := match statisticalRows with
        | none => 2 * r * width + r * r
        | some rows => rows * width + r * rows
      pure (draws * n * limbWords moduli)
  | .preimageSample .. =>
      match args with
      | [publicMatrix, _, target] => do
          let (_, inner) ← publicMatrix.shape env
          let (_, columns) ← target.shape env
          let (n, limbs) ← publicMatrix.ringSize env
          pure (inner * columns * n * limbs)
      | _ => none
  | _ => some 0

/-- The cost of one node that is not a loop or a call, from the types of its arguments and
outputs and what it draws or builds by its own parameters (`Op.intrinsic`). -/
def Op.cost (env : LoopEnv) (op : Op) (args outputs : List Ty) : Option Nat := do
  let out := max (← sumSizes env outputs) (← op.intrinsic env args)
  let moved := (← sumSizes env args) + out
  let (n, limbs) ← match outputs.head? with
    | some ty => ty.ringSize env
    | none => some (1, 1)
  let product (left right : Ty) : Option Nat := do
    let (r, k) ← left.shape env
    let (k', c) ← right.shape env
    let (n, limbs) ← left.ringSize env
    -- A `1 × 1` factor multiplies every entry of the other.
    if r * k = 1 then pure (k' * c * n * n * limbs)
    else if k' * c = 1 then pure (r * k * n * n * limbs)
    else pure (r * k * c * n * n * limbs)
  match op with
  | .matrixBinary .mul | .matrixMulSmallRhs => do
      match args with
      | [left, right] => pure (moved + (← product left right))
      | _ => none
  | .matrixMulAccumulate coefficients _ => do
      -- Every product, with its own shapes.
      let products ← (List.range coefficients.length).mapM fun t ↦ do
        product (← args[2 * t]?) (← args[2 * t + 1]?)
      pure (moved + products.sum)
  | .tensor => do
      match args with
      | [left, right] => do
          let (r₁, c₁) ← left.shape env
          let (r₂, c₂) ← right.shape env
          let (n, words) ← left.ringSize env
          pure (moved + r₁ * c₁ * r₂ * c₂ * n * n * words)
      | _ => none
  | .sha256BitColumns _ rows columns _ =>
      -- One compression per 256 selector bits, charged per bit.
      pure (moved + rows * columns)
  | .ringAutomorphism _ | .multiplyMonomial => pure (moved + out * n)
  | .polynomialFromValues _ true => pure (moved + n * n * limbs)
  | .polynomialValues true => do
      -- The output is a family of integers, so the ring is the argument's.
      let (n, limbs) ← match args.head? with
        | some ty => ty.ringSize env
        | none => some (1, 1)
      pure (moved + n * n * limbs)
  | .rnsModUp .. | .rnsModDown .. | .modulusSwitch _ | .blockModSwitch .. | .crtRecompose .. => do
      let (_, sourceLimbs) ← match args.head? with
        | some ty => ty.ringSize env
        | none => some (1, 1)
      pure (moved + (← sumSizes env args) * (sourceLimbs + limbs + 1))
  | .hashSample _ _ tagPrefix tag .. | .hashIntFamily _ _ tagPrefix tag =>
      -- The runtime hashes the key and every tag byte.
      pure (moved + out * out + tagPrefix.length + (tag.map TagPart.cost).sum)
  | .gadgetDecompose .. | .trapdoorSample .. | .preimageSample .. | .gadgetTrapdoor .. =>
      pure (moved + out * out)
  | .uniformResidueSample .. | .uniformIntervalSample .. | .gaussianSample .. =>
      pure (moved + out)
  | _ => pure moved

/-- The total cost of the iterations `start, start + 1, ...` of a loop body. -/
def loopCost (body : LoopEnv → Option Nat) (env : LoopEnv) (slot : Nat) : Nat → Nat → Option Nat
  | _, 0 => some 0
  | start, count + 1 => do
      pure ((← body (Function.update env slot start)) + (← loopCost body env slot (start + 1) count))

/-- The cost of `scope` of `program`, with `fuel` bounding the nesting of loops and calls. -/
def scopeCost (program : Program) : Nat → Nat → LoopEnv → Option Nat
  | 0, _, _ => none
  | fuel + 1, index, env => do
      let scope ← program.scopes[index]?
      let costs ← scope.nodes.mapM fun node ↦ do
        let args ← node.args.mapM fun ref ↦ do (← scope.nodes[ref.node]?).outputs[ref.port]?
        let own ← Op.cost env node.op args node.outputs
        match node.op with
        | .call body => pure (own + (← scopeCost program fuel body env))
        | .parallelLoop body count slot _ | .sequentialLoop body count slot _ => do
            let k ← count.evalNat env
            pure (own + (← loopCost (scopeCost program fuel body) env slot 0 k))
        | _ => pure own
      pure costs.sum

/-- The cost of a program: the cost of its root. -/
def Program.cost (program : Program) : Option Nat :=
  scopeCost program program.fuel 0 (fun _ ↦ 0)

/-- A program whose cost is defined and at most `bound`. -/
def Program.CostAtMost (program : Program) (bound : Nat) : Prop :=
  ∃ cost, program.cost = some cost ∧ cost ≤ bound

end MxxSecurity.IR
