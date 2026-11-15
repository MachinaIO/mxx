# Layered architecture specification (proposal)

**Status: being implemented.** Implemented so far:

- `crates/core`: types, programs, validation, host execution, RV64IM flattening with a step bound,
  shared leaf libraries, call-shape counts and a measured host-time estimate, Lean export of code,
  and the claim generator; `crates/core/lean`: `MxxCore.Algorithm.run` and an example
  specification whose generated claim type-checks.
- The lattice DSL's lowering (`mxx_ir_core::lower`): every node kind. Under the gadget trapdoor a preimage is the target's gadget decomposition, as the
  executors compute it. Under a sampled computational trapdoor a preimage is drawn as the CPU
  backend draws it, by the leaf `trapdoor_preimage` (`mxx_lattice_leaves::preimage`: the GM18
  perturbation sampler as OpenFHE implements it, `SampleMat`, `ZSampleSigma2x2`, and `ZSampleF`
  over negacyclic binary64 transforms, then the arbitrary-base gadget sampler per tower and
  `x = p + [R; E; I] z`), from a SHAKE256 stream of 32 coin bytes with a word budget, so its step
  bound is finite; an `until` loop of at most 32 attempts redraws while a coefficient exceeds the
  bound or the stream ran out, and faults after the last. Host tests check `A x = u` in every
  tower and the width `s` of the draws; the host and the emulator agree bit for bit within the
  measured step bound; `mxx-backends`' lowering test maps preimages back to their targets on the
  CPU executor, the host, and the emulator. A statistical trapdoor (one row, per limb `A ‖ T`)
  is drawn as the CPU backend's statistical TrapGen: the leaf `statistical_trapgen` draws `T`
  by Karney's sampler (untruncated) and checks that every point's perturbation covariance is
  positive definite, an `until` loop of at most 8 attempts redraws (faulting after the last,
  where the CPU backend reports an error), and `A = [ā | g - ā T]`; its preimages take `p_1` by
  Peikert's convolution sampler (the Cholesky factors at the `n / 2` points, a conjugate-symmetric
  continuous part, and randomized rounding of width `σ`), tested like the computational ones. A trapdoor value is, per limb, its public matrix, followed for a
  sampled computational (MP12) trapdoor by `R` and `E`: `GadgetTrapdoor` is the gadget matrix,
  `TrapdoorPublic` its public matrix, and `TrapdoorSample` samples `R` and `E` Gaussian (truncated
  at 13 σ) and `ā` uniform and forms `A = [ā | I | G - (ā R + E)]`, so `A [R; E; I] = G`, which
  `mxx-backends`' lowering test checks in every tower. This covers arithmetic, shapes, constant
  matrices, every ring and modulus conversion (exact CRT maps, exact block switching, OpenFHE's
  fast RNS ModUp and ModDown), the CRT gadget decompositions (full, with dropped towers, and
  small per limb), polynomial coefficients and OpenFHE's evaluation slots (an NTT with OpenFHE's
  root of unity), CRT recomposition, bit packing, decoding, the uniform, interval, Gaussian, and
  Keccak hash samplers (plain and decomposed, with tags built at run time when they read loop
  indices or runtime integers, bit for bit the CPU executor's), hash integer families, SHA-256
  columns, integer arithmetic, binary64 reals (IEEE arithmetic in the leaves, bit for bit the
  host's), loops whose slices depend on the loop index, bindings of loops and subgraph calls
  (substituted when they read a loop index), families, and selection. Integers are two's complement of one width per program, sized for the product of
  two values below the graph's largest modulus (`mxx_ir_core::lower::int_words`); programs that
  exchange integers, such as one scheme's algorithms, are lowered at a common width
  (`lower_with_int_words`). Leaves are `no_std` Rust (`crates/lattice-leaves`), compiled to RV64IM
  by `scripts/build_rv64_leaves.sh`, plus hand-written ones. Each lowered node kind is checked
  against the CPU executor by host execution and by running its proof image on the emulator
  within its step bound (`crates/backends/src/executor/core_lowering_tests.rs`). Every graph of
  the BGV and TFHE fixtures lowers and flattens, and on the host the lowered BGV programs evaluate
  a two-gate circuit and the lowered TFHE programs one bootstrapped NAND gate on every input
  pair, from key generation to decryption, correctly.
- Proofs on the mathcal-A machine, checked by `scripts/check_core_lean.sh` with the standard
  axioms only. Code is specified once for every image that holds it at some instruction offset
  (`MxxCore.Sym.mrunAt`, a symbolic runner whose instruction facts come from hypotheses): the copy
  routine of proof images (`MxxCore.Copy`) and the addition leaf
  (`crates/ir-core/lean/leaves/MxxLatticeLeaves/AddMod.lean`, from the `At` structure each leaf
  code module now emits), each with a call lemma through `auipc`/`jalr` and back
  (`MxxCore.Glue`). Leaf specifications have the core's byte-level shape of section 4.3
  (`MxxCore.Leaf.Spec`: the descriptor, inputs held as bytes, outputs that hold `f args inputs`,
  every other byte and the kept registers unchanged), with one call lemma for every leaf
  (`Spec.call`); the elementwise leaves are its first instances (`MxxLatticeLeaves.AddMod`,
  `SubMod`, and `NegMod`, each with `spec` and `byteSpec`).
- Generated composition theorems. The flattener records the glue's events
  (`ProofImage::trace`): input queries, leaf calls, copies, counted loops with the events of one
  iteration, and `HALT`, with addresses affine in the loop indices (`flatten::Addr`). Index
  arithmetic in glue has two normal forms, an affine combination of indices or a table in
  read-only data indexed by one index; a loop of one iteration is inlined with its index bound to
  0, and windows of up to 16 runs are copies. From the trace and the leaves' Lean names alone,
  `mxx_core::compose::run_module` writes a module whose theorem `halts` states that the image
  halts within its step bound with the payload the leaves' functions describe
  (`L.f args inputs`), given the input lengths and, for every index of its enclosing loops, each
  leaf call's domain, output lengths, and step budget. Its proof chains one lemma of
  `MxxCore.Compose` per event, each handing on the bytes held in memory through a frame; a loop
  is `Compose.loop` with an invariant at its head (the index, the facts the loop leaves alone,
  each carried value as `Nat.rec` over the body's step, the members of each family it fills,
  `Compose.Members`, and those of enclosing loops' families), its body generated once for a
  symbolic index, with address conditions proved by `omega`; loops nest. A leaf argument affine
  in the indices, or a table over one index in read-only data (`Compose.table_getMem`), makes its
  descriptor a run-time store, which the segment before the call proves word by word. Runs are
  generated and checked for a nested loop (`Generated.Grid`), a loop with a carried value
  (`Generated.Fold`), and loops whose leaf argument reads the index, affinely
  (`Generated.Moduli`) and through a table (`Generated.Tables`). A member read at a run-time
  position and runtime selection (whose candidates' addresses the glue loads from a table in
  read-only data) are copies whose source the proof picks by the run-time value, bounded by a
  hypothesis of `halts` (`Generated.Gather`, `Generated.Choose`). Coins are drawn outside loops
  (`Compose.coins` over mathcal-A's `Prog.Runs.coins`: the result binds the draw,
  `coinsPC n >>= fun bs => …`) or once per iteration of a loop (`Compose.loopCoins`: the
  invariant also takes the draws so far, and the result is `Compose.coinsSeq n N >>= …`), with
  leaf hypotheses quantified over the draws (`Generated.Noise`, `Generated.Samples`). A loop with
  `until` is `Compose.loopUntil`: `halts` takes the stopping iteration `t` with hypotheses that the
  flag is 0 before it and 1 at it, and the body is generated for an iteration that continues and
  for iteration `t`, after which the rest follows (`Generated.Until`). Proofs stay
  linear in the trace: facts nothing reads any more are dropped, long expressions become
  definitions over the variables in scope, and large slices of the image stay slices. The run of
  every BGV and TFHE image is generated (17 KB to 3.5 MB of Lean each), which `mxx-fhe`'s
  lowering test checks; compiling them needs the leaves' Lean specifications. The generated correctness claims
  of three examples are proved this way: the identity
  (`crates/core/lean/MxxCore/Proofs/Identity.lean`) only decodes the payload; the coordinatewise
  sum modulo `q` by one `add_mod` call
  (`crates/ir-core/lean/leaves/MxxLatticeLeaves/Proofs/AddModSum.lean`) discharges the leaf's
  domain and identifies its function with the specification's sum; and the same claim on 12
  residues by a loop over three columns (`MxxLatticeLeaves/Proofs/Columns.lean`) shows that the
  columns' sums concatenate to the sum. The claim generator also
  writes a `Progs` module, a `MathcalA.Prog` per image, which lists all its code (checked for every
  BGV and TFHE image).
- The legacy marking of section 9, step 5.

Not yet: composition theorems of draws nested deeper than one loop, `until`
loops inside loops or filling families, drawing `until` loops other than one draw and one carried
value at the top level (which `Compose.loopUntilCoins` covers), and index expressions other than the
two normal forms (no BGV or TFHE image uses them); byte-level Lean specifications of the lattice leaves the generated BGV and TFHE runs call
other than the hand-written `add_mod`, `sub_mod`, `neg_mod`, `bool_to_int`, `int_constant`,
`scale_mod` (Shoup's multiplication after an in-leaf binary long division, with its arithmetic in
`MxxLatticeLeaves.Math.Shoup`), and `multiply_monomial` (gathered with a mask, two nested loops).
Matrix products and Kronecker products are loops of the glue (rows, columns, and a carried loop
over the inner dimension), each entry by the hand-written `poly_mul_acc` (a negacyclic
multiply-accumulate whose products go through a binary long division subroutine), and `Concat`
flattens a family of entries into a matrix; of the 21 leaves the runs call, 14 remain to be
proved (`poly_mul_acc` among them), each being rewritten as a small hand-written
RV64IM loop and proved, rather than proving rustc's output; moving GPU planning into the core (section 9, step 3); and restating `MxxFhe` as a
specification package and re-proving a BGV claim on generated proof images (section 9, step 4),
whose Lean pins (section 10, decision 4) are aligned.

This document specifies how mxx is split into a domain-agnostic core, DSLs on top of it,
unchanged applications, implementation-agnostic protocol specifications, and a separate library
of hardness assumptions. It is written against
the head of PR #177 (`claude/lean-security-proofs-5a8fe3`, commit `34baf8fa7`), which adds the
implementation-agnostic FHE specification (`MxxFhe`, `spec::Fhe`) and the closed list of
security assumptions (`MxxSecurity.Assumptions`).

Two names refer to outside work:

- **mathcal-A** is the repository `MachinaIO/mathcal-A`. Its trusted core defines an RV64IM
  machine (a wrapper over riscv-zkvm, proved equal per instruction to the Sail RISC-V model),
  games, `Target`, `Hard`, `Refuted`, and black-box reductions (`Reduces`). Its service
  specification is `docs/SERVICE-SPEC.md` there.
- **The assumption library** is the Lean package of section 7. mxx and the mathcal-A service
  both use it.

Paths are relative to this repository unless they name mathcal-A.

## 1. Layers

| Layer | Contents | Privilege |
| --- | --- | --- |
| **Core** | Structured binaries (section 3), their flattening to an RV64IM image with a Lean proof, cost estimation, planning, and execution. Knows nothing about lattices, rings, or moduli. | The only privileged layer: every other layer uses its public API only. |
| **DSL** | Typed front ends that build core programs, supply the leaves, and prove the leaves' specifications. The lattice DSL that mxx ships is one DSL among others. | None. mxx's lattice DSL has no access that a third-party DSL lacks. |
| **Protocol specifications** | Lean definitions of a primitive's correctness and security with the algorithms left as parameters, one package in each primitive's crate (section 6). Claims are generated by inserting a scheme's proof images and assumptions. | None. |
| **Assumption library** | The Lean definitions of the hardness assumptions a security claim may use (section 7). | None, but it is reviewed like trusted code, because claims and attacks both refer to it. |
| **Applications** | `mxx-khe`, `mxx-fhe`, `mxx-exponent-lut`, `mxx-pe`, `mxx-we`, and the interface crates. | None. They are unchanged by this specification, apart from the rename of `mxx-dsl` to `mxx-lattice-dsl` (section 5). |

### 1.1 Dependencies

```text
     primitive crates: Rust code (unchanged)     primitive crates: lean/spec
                     |                                   |
                     v                                   |
   lattice DSL (mxx-lattice-dsl, mxx-ir-core,            |      assumption library
   mxx-backends)   other DSLs                            |              |
         \           /                                   |              |
          v         v                                    v              v
           mxx-core (Rust + Lean) ---------------->  mathcal-A core (Lean)

   generated claims: import one lean/spec, the assumption library, and the generated images
```

- `mxx-core` depends on no DSL and no application.
- The Lean side of `mxx-core`, every DSL's leaf proofs, the specification packages, and the
  assumption library all build on the mathcal-A core. The mathcal-A core imports nothing from
  mxx.
- A specification package imports only the mathcal-A core and `MxxCore.Algorithm`
  (section 6.3): never a DSL, a scheme, or the assumption library. Only generated claims bring
  them together.

## 2. Concepts

**Parameter.** A named integer or real bound when a program is validated. A *compile
expression* is an expression over parameters and loop indices.

**Runtime value.** A value that exists only while a program runs. It never changes a buffer
size or a loop count.

**Buffer.** A contiguous byte range whose length is a compile expression. The core gives the
bytes no meaning and owns all memory of a run.

**View.** An affine window (origin, extent, stride) into a buffer. Leaves receive views.

**Leaf.** Code with a declared signature that computes outputs from inputs (section 3.2). Leaves
are the only nodes that compute.

**Structured binary.** A tree of loops whose leaves carry code (section 3.1). It is the
core's program format and the only thing a DSL hands to the core.

**Proof image.** The single, unstructured RV64IM binary that the core derives from a structured
binary (section 3.4). Lean claims are about proof images.

**Execution kernel.** The code that actually runs a leaf on a GPU or CPU. It is not what claims
are about; section 8 states how it relates to the leaf's RV64IM code.

## 3. Core layer

### 3.1 Structured binaries

Value types:

```rust
enum CoreType {
    /// `count` elements of `element_bytes` bytes. The core stores every value 8-byte aligned.
    Buffer { element_bytes: IntExpr, count: IntExpr },
    /// `count` members of one type; the element type may depend on the member index.
    Family { element: Box<CoreType>, count: IntExpr },
    /// A signed integer of `words` 64-bit words that the core may read.
    Int { words: u32 },
    Bool,
    /// An opaque value a store can hold, identified by its schema.
    TypedBlob { type_name: String, schema_hash: [u8; 32] },
}
```

`Int` and `Bool` exist because the core reads them: a selector or a dynamic index of a built-in
leaf, or the stop flag of a loop. A value that only DSL leaves read is a `Buffer`. `IntExpr`
keeps constants, variables, loop indices, arithmetic, `Log2Ceil`, and `Select`; the
ring-specific constructors (`RingModulus`, `RingCrtDepth`, `RingCrtModulus`) leave the core.

A structured binary has two kinds of node:

| Node | Meaning |
| --- | --- |
| `Leaf { leaf, arguments, inputs }` | Runs one leaf: a DSL's leaf (section 3.2) or one of the core's built-in leaves (below). |
| `Loop { count, carried, body, until }` | Runs `body` up to `count` times (below). |

**Loops.** `body` is a named definition: a list of nodes over the loop index `i` (a compile
expression), the carried values, and the loop's other inputs, with bindings for compile
parameters. Each iteration returns the next carried values and its per-iteration outputs. The
loop returns the final carried values and, for each per-iteration output, a family of `count`
members, member `i` coming from iteration `i`.

- A loop whose body reads only some members of a family receives the whole family and takes
  members with the built-in `get` and `get_dynamic` leaves.
- With `until = Some(flag)`, the loop stops after the first iteration whose Bool output `flag`
  is true; if no iteration of the `count` sets it, the run fails with a run-time error. Such a
  loop has no per-iteration outputs, only carried values.
- A loop with `count = 1` is a call of its body.

**Independent loops.** A loop without carried values and without `until` is *independent*: its
iterations read nothing that another iteration writes, and each writes only its own members.
Independence is a property the core reads off the loop, not a separate kind of node. The core
may run the iterations of an independent loop in any order and concurrently, and it plans how
many run together (section 3.6).

**Built-in leaves.** The core provides these leaves, with their RV64IM code, their execution
kernels, and the proofs of their specifications:

| Leaf | Result |
| --- | --- |
| `window { origin, extent, stride }` | An affine window of a value. |
| `concat`, `pack` | One value made of adjacent windows; a family made of values. |
| `get { index }` | A family member at a compile-expression index. |
| `get_dynamic` | A family member at a runtime `Int`; the members must share one type. |
| `select` | One of several values of one type, chosen by a runtime `Int`. Every candidate has already been computed. |
| `coins { bytes }` | Fresh uniform bytes. |
| `query` | One exchange with an external party (a request in bytes, a response in bytes). |
| `constant { ty, bytes }` | A value given by its bytes. A proof image holds them in read-only data and copies them, so a constant adds no code. |

`coins` and `query` are the only leaves with effects; every DSL leaf is a function of its
arguments and inputs (L4). A sampler is a DSL leaf that takes the output of a `coins` leaf as an
input.

**Program interface.** Inputs, outputs, and artifact imports and exports are declared on the
program, not as nodes.

Rules that hold for every structured binary:

- **S1.** No data-dependent control flow except `until`. A runtime value selects among computed
  values or family members; it never decides whether a node runs.
- **S2.** An iteration writes only its own values and member `i` of each per-iteration output.
  The core enforces this by construction.
- **S3.** A `coins` leaf draws fresh bytes every time it runs. Outside any loop it runs once per
  program run; inside a loop body it runs once per iteration, including each iteration of an
  `until` loop. This is today's sampler-placement rule, carried by the one leaf that produces
  randomness.
- **S4.** A `query` leaf is not allowed inside an independent loop, so the order of exchanges is
  fixed.

**Hints.** Many of today's node kinds exist to tell the GPU lowering something, not to change
what the program computes. Under this design the core derives those facts by analysis:

| Today | Derived fact |
| --- | --- |
| `Zip`, `ZipOffset`, `ZipQuotient`, `Gather` | Which member iteration `i` reads: the index of a `get` is a compile expression in `i`, and that of a `get_dynamic` is a member of an input family. |
| `View`, `Concat`, `FamilyGetStatic`, `FamilyPack` without copies | Whether a `window`, `concat`, `get`, or `pack` can be an alias instead of a copy, from the liveness of its operands. |
| `Reduce`, `parallel_reduction.rs` | Whether an independent loop's family is consumed only by a loop that folds it with one leaf, so the fold can accumulate without storing the family. |

A DSL may attach the same facts as hints to save the analysis; the core checks each hint against
its own analysis and rejects a program whose hint is wrong. One fact cannot be derived: whether a
fold may be regrouped, which needs the folding leaf to be associative. A DSL declares it with the
hint `associative` on the leaf and proves it in Lean as part of the leaf's specification
(section 4.3). Hints never change results, only how fast and in how much memory a program runs.

Compared with today's IR, `SubgraphCall`, `ParallelLoop`, and `SequentialLoop` become `Loop`;
`Select`, the family nodes, slices, and concatenation become built-in leaves; preimage retries
become `until` loops.

### 3.2 Leaves

```rust
struct LeafSignature {
    name: String,
    arguments: Vec<String>,       // names of the compile-expression arguments; sizes and the
                                  // step bound may refer to them
    inputs: Vec<CoreType>,
    outputs: Vec<CoreType>,
    scratch_bytes: IntExpr,
    in_place: Vec<(usize, usize)>,// an output may alias an input with no later reader
    associative: bool,            // hint: a two-input fold leaf that may be regrouped
}
```

Randomness reaches a leaf only as an input produced by a `coins` leaf.

A leaf carries some of these artifacts:

| Artifact | Purpose | Required for |
| --- | --- | --- |
| **RV64IM code** | Position-independent code for the proof image, with its ABI record (which registers hold the arguments and the address and stride of each view). Built from `no_std` Rust or written by hand, within the instruction set of the mathcal-A machine model. | Proof images and claims. |
| **Execution kernels** | A CUDA/HIP kernel against the native leaf interface (the successor of `crates/backends/gpu/include/SubgraphKernel.h`), and a host function. | Execution on that backend. |
| **Variants** | Alternative execution kernels with different scratch or launch shapes. | Planning (section 3.6). |

The **leaf contract** holds for every artifact of a leaf:

| | Requirement |
| --- | --- |
| **L1** | Running time depends on the signature's arguments and types, not on the bytes of the inputs. Data-dependent repetition is an `until` loop in the structure. |
| **L2** | The leaf reads only its input views and scratch, and writes only its output views and scratch. |
| **L3** | The leaf allocates nothing. |
| **L4** | The leaf keeps no state between calls: it is a function of its arguments and inputs. Only the built-in `coins` and `query` leaves have effects. |
| **L5** | The leaf does not synchronize with the host. |

The core never interprets what a leaf computes.

### 3.3 What the core derives from a structured binary

| Product | Section | Used by |
| --- | --- | --- |
| Proof image and its composition proof | 3.4 | Protocol claims (section 6), the mathcal-A service |
| Cost estimate | 3.5 | Benchmark estimates for any DSL |
| Plan and execution | 3.6 | Running programs on GPUs and CPUs |

### 3.4 Flattening to a proof image

For concrete parameter bindings, the core lays out every buffer at a fixed offset and emits one
RV64IM image:

- **Glue code.** Every `Loop` becomes a counted loop, with an early exit when it has `until`; an
  independent loop runs its iterations in index order. A loop with `count = 1` is inlined, and a
  window of at most 16 runs at constant positions is unrolled into copies. Leaf
  code, the built-in leaves' included, is placed once and reached with `JAL`/`JALR`. All code is
  one run of decodable instructions from the image's start, as mathcal-A's `Prog` lists it: leaf
  libraries without read-only data come first, alignment padding is `EBREAK` (which decodes and
  faults), and constants live in the image's read-only data rather than in leaves.
- **Environment calls.** The `coins` leaf draws bytes with the machine's `COINS` call; the
  `query` leaf uses `QUERY`; the end of the program uses `HALT`. These are the calls of the
  mathcal-A machine, so a proof image is an ordinary mathcal-A binary.
- **Cost.** The cost of a proof image is its number of machine steps. The core reports a bound as
  the structural sum of each leaf's step bound and the glue's steps.

The core also writes Lean:

- the image, its instruction list in the form of mathcal-A's `Prog`, and per-leaf decode checks;
- the structure term over opaque leaf functions;
- one theorem: given each leaf's specification (section 4.3), the proof image computes the
  structure term within the reported step bound.

That theorem is an instance of composition lemmas proved once in the core's Lean package: one for
sequencing, one for a loop with and one without `until`, plus a lemma that leaf code keeps its
specification at any load offset. The core's Lean package is untrusted and kernel-checked; only
the mathcal-A machine model is trusted.

### 3.5 Cost estimation

Without running the whole program, the core reports for a parameter binding:

- the number of executions of each leaf, in closed form over loop counts;
- the peak memory of each candidate plan;
- given a per-leaf cost (a measured time or a step bound), the total.

Measured times come from running each distinct (leaf, shape, variant) at one and at two
repetitions and extrapolating linearly, as the GPU planner does today
(`crates/backends/src/gpu_runtime_direct.rs`). L1 is what makes this valid. Counting follows the
structural traversal of today's `crates/ir-core/src/graph.rs` and `crates/ir-core/src/cost.rs`.

### 3.6 Planning and execution

Planning chooses, for each nesting depth of independent loops, how many iterations run
together, and for each leaf, a variant. It rejects candidates whose scheduled memory peak does
not fit and times the rest. Today's two numbers, the wave width W and the column tile width C,
become the widths of two depths of independent loops: a DSL expresses column tiling as an
independent loop over columns. An `until` loop is timed at one iteration and scaled by its
expected number of iterations (section 10, decision 9). Memory
scheduling, CUDA Graph regions, address patching, and multi-GPU placement keep their current
design (`crates/backends/src/gpu_graph_memory.rs`, `crates/backends/src/gpu_physical_lowering.rs`)
with leaf names in place of node kinds.

### 3.7 What the core excludes

Rings, CRT bases, moduli, NTT domains, sampler laws, coefficient bounds, gadget decomposition,
trapdoors, and every lattice cost model. Today these are spread over `mxx-ir-core` and
`mxx-backends`; section 4.4 lists where each goes.

## 4. DSL layer

### 4.1 Equal standing

A DSL is a crate that:

1. offers handles and operators to its users, with the checks that need its domain knowledge;
2. lowers its programs to structured binaries;
3. supplies leaves with their artifacts;
4. supplies a Lean package with each leaf's specification and its proof (section 4.3);
5. optionally performs optimizations before lowering (sharing, fusion, choice of
   representation).

The core offers no extension point that only mxx's lattice DSL uses. A third-party DSL can do
anything the lattice DSL does.

### 4.2 The lattice DSL

The lattice DSL is the three existing crates `mxx-lattice-dsl` (formerly `mxx-dsl`, renamed to
say which domain it serves), `mxx-ir-core`, and `mxx-backends`, with their public APIs kept, so
applications change only in the crate name they import. Each node kind
of today's IR lowers to leaves:

| Today | Lowered to |
| --- | --- |
| Control and family nodes, slices, concatenation, inputs, artifacts | Loops, built-in leaves, and the program interface |
| Matrix arithmetic, ring and modulus conversions, decomposition, decoding, CRT | Leaves, with today's CUDA kernels as their execution kernels |
| Samplers | Leaves that take bytes from a `coins` leaf; preimage retries become `until` loops |
| Domain tracking, fusions, and identity expansion of the GPU lowering | Lattice DSL optimizations before lowering |
| `crates/backends/src/gpu_cost_model.rs` | Lattice DSL per-leaf cost functions over the core's counts |

New for this DSL: an RV64IM implementation of every leaf, written as `no_std` Rust compiled for
the mathcal-A instruction set. Its constraints come from the model (header of mathcal-A's
`MathcalA/Trusted/Machine.lean`): no compressed encodings, no floating point, no atomics, and of
the RV64 word operations only `ADDIW`. Leaf code therefore uses 64-bit integer arithmetic; `MULHU`
is available for Shoup and Barrett multiplication.

### 4.3 Leaf specifications

For each leaf, the DSL's Lean package states and proves, in the form of mathcal-A's `Prog.Spec`:

> From the entry with the ABI's registers and input memory, the leaf returns within
> `steps args` steps without an environment call, its output views hold `f args inputs seed`, and
> no byte outside its output views and scratch changed.

`f` is the leaf's mathematical function and `steps` its step bound, both written by the DSL. The
statement's shape, the ABI, and the confinement clause are generated by the core. The proof
quantifies over all arguments in the leaf's declared range.

For the lattice DSL, `f` is the existing primitive model (`MxxPrimitives`, `MxxRuntime` in
`crates/ir-core/lean`). Those models stop being trusted once every leaf that uses them has its
proof.

### 4.4 What moves where

| Today | Destination |
| --- | --- |
| Graph structure, scopes, subgraph sealing, family types, artifacts, structural counting (`crates/ir-core/src/graph.rs`, `cost.rs`, `artifact.rs`) | `mxx-core` |
| Planner, memory scheduling, graph regions, native leaf interface, timing cache (`crates/backends/src/gpu_*`) | `mxx-core`, without node-kind knowledge |
| `NodeKind`, rings, `IntExpr` ring constructors, lattice validation, `parallel_reduction.rs` | lattice DSL (`mxx-ir-core`) |
| CUDA kernels under `crates/backends/gpu/src/matrix/` | lattice DSL (`mxx-backends`) as execution kernels |
| `crates/ir-core/lean` primitive and runtime models | lattice DSL Lean package |
| `crates/ir-core/lean/security` (`MxxSecurity`) | Assumptions to the assumption library (section 7); the rest stays with the legacy statements (section 6.8) |

Moves happen one subsystem at a time; no compatibility layer keeps two copies.

## 5. Application layer

`mxx-khe`, `mxx-fhe`, `mxx-exponent-lut`, `mxx-pe`, `mxx-we`, `mxx-func-enc`, and `mxx-io` keep
their source, apart from importing `mxx_lattice_dsl` instead of `mxx_dsl`. They use
`mxx-lattice-dsl`, `mxx-ir-core`, and `mxx-backends` directly, and those crates keep their public
APIs (section 4.2). `mxx-fhe` additionally receives a Rust descriptor
next to its existing Lean specification package (section 6). This is an addition next to the
existing code; the code of the algorithms does not change. The other primitive crates are
untouched until their specifications are written (section 6.1).

## 6. Protocol specifications

A DSL program describes one algorithm of a cryptographic primitive: for FHE, one of `KeyGen`,
`Enc`, `Eval`, and `Dec`. A **protocol specification** defines the primitive's correctness and
security once, with the algorithms left as parameters. A claim about a concrete scheme is then
generated by inserting two things into those definitions: the RISC-V proof images that the DSL
compilation produces for the scheme's algorithms, and the assumptions the scheme relies on. No
one writes a claim by hand.

### 6.1 Where specifications live

Each primitive's crate holds its specification as a Lean package of its own:

| Primitive | Crate | Specification package | Status |
| --- | --- | --- | --- |
| FHE | `mxx-fhe` | `crates/fhe/lean/spec` (`MxxFhe`) | exists since PR #177; restated by section 6.7 |

FHE is the only specification in scope. Predicate encryption (`mxx-pe`), iO (`mxx-io`),
functional encryption (`mxx-func-enc`), and witness encryption (`mxx-we`) are deferred. When one
is written, it goes to `crates/<crate>/lean/spec` under the same rules. `mxx-khe` and
`mxx-exponent-lut` are libraries, not primitives, and have no specification.

A specification package imports only the mathcal-A core and the calling convention of the core's
Lean package (section 6.3). It never imports a scheme, a DSL, a generated module, or the
assumption library. Proofs about a particular scheme live in sibling packages of the same crate,
as `crates/fhe/lean/bgv` and `crates/fhe/lean/tfhe` do today.

### 6.2 What a specification defines

| Definition | Contents |
| --- | --- |
| `Impl` | One field per algorithm, each an `Algorithm`: a proof image (`MathcalA.Bytes`) and its step bound (`MathcalA.Bounds`). Nothing about how the image was produced. |
| `Params` | The primitive-level parameters the definitions quantify over, for example FHE's message space and depth bound, or iO's circuit class and size. Not scheme parameters such as ring dimensions, which live inside the images. |
| Encodings | The byte encoding of every value the definitions inspect: messages, circuits, predicates, attributes. Keys, ciphertexts, and obfuscated programs stay opaque bytes. |
| `Correct` | `Correct (I : Impl) (p : Params) (δ : ℝ) : Prop`: every execution meets the primitive's correctness condition except with probability at most `δ` (`δ = 0` for perfect correctness). |
| Security notions | For each notion, a function from `Impl`, `Params`, the adversary's resources (`MathcalA.Bounds`), and a claimed advantage to a `MathcalA.Target`. Its game runs the algorithms as the challenger and the adversary as a mathcal-A binary. Examples: IND-CPA for FHE, attribute hiding for predicate encryption, indistinguishability of equivalent circuits for iO. |
| Manifest | `spec.toml`: the package's module, the `Impl` field names in order, and the constant names of `Params`, `Correct`, and each security notion. The generator reads it (section 6.4). |

Lemmas that help prove claims, such as hybrid arguments, may live in the package; they are
untrusted.

### 6.3 Running an algorithm

`MxxCore.Algorithm.run`, in the core's Lean package, fixes how a proof image is run as an
algorithm:

```lean
def Algorithm.run (A : Algorithm) (inputs : List MathcalA.Bytes) :
    ProbComp (Option (List MathcalA.Bytes))
```

It runs `A.image` on the mathcal-A machine within `A.bounds`, with these environment answers:

- `QUERY` with the request `0x00 ++ u64 i` returns `inputs[i]` (the empty string if `i` is out of
  range);
- `COINS` returns uniform bytes;
- any other `QUERY` request returns the empty string;
- `HALT` ends the run; its output, decoded as a length-prefixed list of byte strings, is the
  result;
- a fault, a timeout, or a payload that does not decode gives `none`.

The flattener (section 3.4) emits each program input as such a request and the program's outputs
as the `HALT` payload. Algorithms are not interactive: a `query` leaf in an algorithm makes every
run `none`. This definition is part of the meaning of every claim, so it is reviewed like trusted
code.

### 6.4 Generating the claims

`mxx-core` provides one generator for all primitives. Its inputs:

1. the specification's manifest;
2. for each algorithm, the scheme's structured binary at concrete bindings, which the core
   flattens to a proof image and a step bound;
3. the value of `Params`, written as a Lean term by the crate's Rust descriptor (section 6.6);
4. the claimed `δ`, and for each security notion the adversary's resources and the claimed
   advantage;
5. the assumptions: for each, an assumption of the library with its parameters, resources, and
   advantage, which the generator turns into a `MathcalA.Target`.

Its output, for FHE with one assumption:

```lean
-- Generated. Do not edit.
import MxxFhe.Spec
import MxxAssumptions.DecisionRlwe
import Generated.Images

namespace Generated

def impl : MxxFhe.Impl :=
  { keyGen := ⟨Images.keyGen, ⟨1048576, 65536, 32768⟩⟩, enc := …, eval := …, dec := … }

def params : MxxFhe.Params := { messageBits := 1, depth := .bounded 3 }

def CorrectnessClaim : Prop := MxxFhe.Correct impl params (2 ^ (-100 : ℤ))

def SecurityClaim : Prop :=
  MathcalA.Hard (MxxAssumptions.DecisionRlwe.target ⟨…⟩ ⟨…⟩ (2 ^ (-80 : ℤ))) →
  MathcalA.Hard (MxxFhe.indCpa impl params ⟨…⟩ (2 ^ (-78 : ℤ)))

end Generated
```

- `Generated.Images` defines the code of each distinct leaf once and each image as its glue code
  followed by references to leaf segments, so the module grows with the distinct code, not with
  the number of calls. Its decode checks are per segment (section 3.4).
- The generator also writes the composition theorem of every image (section 3.4), for the proofs
  to use.
- The output is deterministic: the same inputs give byte-identical modules. A manifest records
  the sha256 of every image and the revisions of the specification package, the assumption
  library, and the mathcal-A core.
- The assumptions are hypotheses of `SecurityClaim`, one per assumption, in the order given. A
  claim may name only assumptions of the library; the generator rejects anything else.
- As today, a proof package proves `Generated.CorrectnessClaim` and `Generated.SecurityClaim`,
  and a short certificate checks the proofs against the generated modules and checks their axioms.

The generator is not trusted. Apart from the image bytes, the generated modules are a few lines
that a reader can check directly: which specification, which parameters, which `δ`, which
resources and advantages, and which assumptions.

### 6.5 Proving a claim

A proof of a generated claim combines three things:

1. for each image, the composition theorem: the image computes its structure term within its step
   bound;
2. the DSL's leaf specifications (section 4.3), which give the structure term a mathematical
   meaning;
3. reasoning at the level of the specification, such as noise bounds or a hybrid argument with
   reductions given as binaries in the sense of mathcal-A's `Reduces`.

Only the third is specific to a scheme. The first is generated, and the second is proved once per
leaf by the DSL.

### 6.6 The Rust side

Each primitive crate adds, next to its existing API, a descriptor that depends only on `mxx-core`:
the algorithms in the order of `Impl`, the input and output roles of each, the encodings, and the
rendering of `Params` as a Lean term. A scheme supplies one structured binary per algorithm, built
with any DSL. DSL-specific conveniences such as `mxx-fhe`'s `spec::Fhe` trait, whose methods take
and return lattice DSL values, stay where they are and produce those binaries. The code of the
algorithms does not change.

### 6.7 FHE, the first instance

PR #177 already separates the FHE definitions (`MxxFhe` in `crates/fhe/lean/spec`) from the
schemes (BGV and TFHE implement `spec::Fhe`). Its definitions take each algorithm as a
computation of the relational semantics of the generated graphs (`generatedRoot.comp`, with
ideal samplers), and its bounded security measures simulators with
`MxxSecurity.IR.Program.cost`. Under this specification:

- `MxxFhe.Scheme` becomes an `Impl` of four algorithms, `Correct` is stated over
  `Algorithm.run`, and IND-CPA becomes a security notion that returns a `MathcalA.Target`;
- `crates/fhe/src/spec/lean` is replaced by the generic generator of section 6.4 and the crate's
  descriptor;
- the current claims stay valid until the leaves BGV and TFHE use have their proofs
  (section 9, phase 4).

### 6.8 Legacy correctness statements

Today a protocol states correctness by declaring an ideal functionality in the DSL
(`mxx_ir_core::protocol::ProtocolDecl`, `mxx_ir_core::lean::protocol::export`, specified in
`crates/ir-core/LEAN-SPEC.md`). This path stays as a **legacy implementation**: it keeps working
and its existing users are not changed, but new protocols use protocol specifications. Its users
at the base commit are `crates/lattice-dsl/examples/rlwe_encrypt.rs`, `crates/lattice-dsl/src/test_protocol.rs`,
`crates/lattice-dsl/src/test_lwe_protocol.rs`, `crates/exponent-lut/tests/nand_circuit.rs`,
`crates/exponent-lut/tests/public_ip.rs`, `crates/khe/tests/test_gpu_ring_gsw_boolean_bgg.rs`,
`crates/pe/tests/gpu_round_trip.rs`, and `crates/we/src/lean.rs`.

### 6.9 Randomness in claims about proof images

A proof image draws randomness only through `COINS`, as uniform bytes. A claim about it therefore
needs the distribution of each sampler leaf's output as a lemma about the leaf, not as an ideal
sampler assumed by the semantics. This is the main new proof obligation of the binary path:

- integer-only samplers (table-based Gaussians, uniform residues by rejection) have such lemmas
  at moderate cost;
- the trapdoor preimage sampler is assumed to draw its ideal law today (assumption R1 of
  `crates/ir-core/LEAN-SPEC.md`, on which `MxxSecurity.Sampling` relies) and has no such lemma
  yet.

Claims whose algorithms use a sampler without its lemma stay on the graph-semantics path.

## 7. Assumption library

The hardness assumptions a security claim may use live in one Lean package of their own, outside
the core, the DSLs, and the protocol specifications.

- **Contents.** Each assumption is a mathcal-A `Assumption` (a parameterized game and its
  baseline), with reference parameter sets. Initially: decision Ring-LWE and the circular
  key-switching assumption from `MxxSecurity.Assumptions`, and LWE and the lattice definitions
  that mathcal-A's specification moves out of mathcal-A. A claim builds the `Target`s it needs
  from these by choosing parameters, resources, and advantage.
- **Use.** A security claim may assume `Hard` only of targets built from library assumptions.
  The claim generator (section 6.4) rejects anything else. This replaces the Rust closed
  list `mxx_ir_core::security::AssumptionId`, which stays with the legacy path.
- **Shared with cryptanalysis.** The mathcal-A service accepts the library as an allowlisted
  dependency, so a claim's assumptions can be submitted as attack problems unchanged. A broken
  assumption invalidates every claim that lists it.
- **Changes.** The library is reviewed like trusted code: a claim means what its assumptions
  mean. Adding an assumption is a reviewed change.

## 8. Trust

| Stage | Trusted |
| --- | --- |
| A claim about proof images | The Lean kernel; the mathcal-A machine model, `Hard`, and `Reduces`; `MxxCore.Algorithm.run` (section 6.3); the specification package's definitions; the assumption library's definitions; and the few non-image lines of the generated claim, which a reader checks directly (section 6.4). Not the claim generator, not the core's flattener, not the DSLs, and not the primitive models, whose roles are proved. |
| That an execution computes what the claim is about | **E1:** each leaf's execution kernels compute what its RV64IM code computes. This is assumed. The core runs differential tests that execute both on random inputs (the RV64IM code in an emulator) and compare outputs. Proving E1 per kernel against a PTX semantics is future work. |
| A legacy or graph-semantics claim | As today: the statement generator, `MxxPrimitives`, `MxxRuntime`, and the ideal-sampler assumptions. |

## 9. Migration

1. **Core skeleton.** Create `mxx-core` with the types of section 3.1, the leaf signature, and
   flattening to a proof image. Port one elementwise lattice operation as a leaf with RV64IM
   code, prove its specification, and generate the composition theorem for a small program.
2. **Lattice DSL lowering for proof images.** Lower every node kind of today's IR to core nodes
   and leaves, with RV64IM code for every leaf. The existing GPU runtime keeps executing graphs
   unchanged in this phase.
3. **Estimation and planning in the core.** Move structural counting, the planner, memory
   scheduling, and the native kernel interface into `mxx-core` without node-kind knowledge, with
   today's CUDA kernels as execution kernels. Compare plan choice, predicted time, and peak
   memory with the current runtime on the same programs before switching.
4. **Protocol claims on proof images.** Align the Lean pins of mxx and mathcal-A. Add
   `MxxCore.Algorithm` and the claim generator of section 6.4. Restate `MxxFhe` as a
   specification package with a manifest, move its assumptions to the assumption library, add
   the FHE descriptor, and re-prove one BGV claim on generated proof images.
5. **Legacy marking.** Mark `ProtocolDecl` and `lean::protocol::export` as legacy in their rustdoc
   and in `crates/ir-core/LEAN-SPEC.md`.

## 10. Open decisions

1. **RV64 word operations.** Extend the mathcal-A machine model with `ADDW`, `MULW`, and the
   other word operations (each needs its Sail equivalence proof), or restrict leaf code to 64-bit
   arithmetic, which rustc does not enforce by itself.
2. **Execution kernels and E1.** Keep today's hand-written CUDA kernels with differential tests
   (this document), or generate execution kernels from the same source as the RV64IM code so
   that E1 can be checked per kernel.
3. **Barriers in proof images.** NTT kernels use shared memory and barriers. Their RV64IM code is
   sequential. Whether the core should recognize a block-level loop form (an independent loop
   whose iterations share scratch and synchronize between stages), so that a future PTX proof
   can follow the same structure, is open. It would be a third node kind or a hint, and this
   document has neither.
4. **Lean pins.** Resolved: mxx follows mathcal-A. `crates/ir-core/lean/security` pins VCVio at
   mathcal-A's `87bc189` (with its PolyFun and QuantumFoundations), on the same mathlib
   (`56a0058`, v4.35.0). `scripts/check_fhe_lean.py` compiles `MxxSecurity`, `MxxFhe`, and the
   BGV and TFHE proof packages against a built mathcal-A checkout, and one module imports
   `MathcalA`, `MxxFhe`, and `MxxCore.Compose` together.
5. **Assumption library home.** A repository of its own, or a package in mathcal-A's assumption
   repository.
6. **Reduction cost.** `MxxFhe`'s bounded security measures simulators with IR operation costs.
   Moving to machine steps changes the meaning of its bounds, so existing claims must be
   restated, not just re-proved.
7. **Size of images in Lean.** A proof image of a full scheme may be megabytes. Sharing leaf code
   between images (section 6.4) bounds the module size by the distinct code, but whether the
   kernel handles byte literals of that size, or the images must be split further, is untested.
8. **Deferred specifications.** Predicate encryption, iO, functional encryption, and witness
   encryption get specifications later. Each will have to choose its security notions, for
   example which iO definition and which predicate-encryption hiding notion.
9. **Expected count of an `until` loop.** Planning scales an `until` loop by its expected number
   of iterations. Whether the DSL declares it as a hint, or the core measures it during planning,
   is open.
