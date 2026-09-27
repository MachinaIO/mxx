# mxx-dsl API specification

This document lists every value type and operation of the DSL, grouped by purpose, with how to
call each one. It is written for readers who know lattice cryptography but not mxx. Read it top
to bottom: section 1 defines the terms that the rest uses.

## 1. Concepts

**Program (graph).** DSL code does not compute. Every operation adds a node to a directed graph
whose nodes are primitive operations (a matrix product, a Gaussian sample, a decoding) and whose
edges carry values. A backend in `mxx-backends` executes the finished graph later, on the CPU or
the GPU. So `&a * &s` returns a *handle* to the future product, not a matrix of numbers.

**Handle.** A Rust value such as `Mat` or `Int` that refers to one value of the graph. Cloning a
handle shares the value; it never recomputes or resamples it.

**Compile expression versus runtime value.** Two kinds of numbers appear:

- A *compile expression* (`IntExpr`, `RealExpr`) is known before the program runs: a Rust
  literal, or a named *parameter* such as `IntExpr::Var("n".into())` that gets its value when the
  graph is validated. Shapes, loop counts, moduli, sampler widths, and decomposition bases are
  compile expressions. Arguments typed `impl Into<IntExpr>` accept Rust integers directly.
- A *runtime value* (`Int`, `Bool`, a matrix) exists only while the program runs, for example
  an input or a decoded bit. A runtime value can choose between computed values or family
  members, but it can never change a shape or a loop count.

**Ring.** Matrices live over `R_Q = Z_Q[X]/(X^N + 1)`, where `Q` is a product of distinct
word-sized primes `q_i = 1 mod 2N`, its *CRT basis*. The order of the basis is part of the ring.

**Shape.** A matrix shape is written `(rows, columns)`; each entry is a compile expression. A
*scalar* polynomial is a `1 x 1` matrix.

**Coefficient bound (cutoff).** An integer `B` such that every centered coefficient lies in
`[-B, B]`. Samplers always take one, and resample any draw that exceeds it, so correctness can be
argued from worst-case bounds. Bounded matrix types carry their bound.

**Family.** An ordered, fixed-length collection of values of one type, such as the 1024
ciphertexts of a batch. `Family<T>` is to the graph what `Vec<T>` is to Rust.

**Schema.** The type description used to declare an input of that type, for example
`IntType` for an `Int` input or `MatType` for a matrix input.

**Sampler placement.** Where a sampler is written decides how often it draws: once per program
run outside any loop, and once per loop iteration inside a loop body. Every run of the program
draws fresh randomness automatically.

**Artifact.** A value one program exports so that another program, possibly run by another
party, can import it: for example public keys produced by setup. An artifact is identified by
the producing program and run (`ProductionId`) and a name.

## 2. Building and validating a graph

A graph is built with a `DslContext`: declare parameters and inputs, compute, name outputs, and
call `build`.

```rust
let context = DslContext::new("my-protocol").int_parameter("n");
let x = ring.input("x", (1, 1));
let built = context.output("y", &x + &x)?.build()?;
let validated = built.validate(&bindings)?; // bindings: ParamEnv with a value for "n"
```

| Call | Meaning |
| --- | --- |
| `DslContext::new(name)` | Starts a graph with a name. |
| `.int_parameter(name)`, `.real_parameter(name)` | Declares a named compile parameter, used through `IntExpr::Var(name)` or `RealExpr::Var(name)`. |
| `context.input::<V>(name, schema)` | Declares a runtime input of any value type `V`, described by its schema (section 3). |
| `context.int_family_input(name, count)` | Declares a runtime input `Family<Int>` of `count` integers. |
| `context.evaluate_int(expr)` | Turns a compile expression into an `Int` value. |
| `context.hash_int_family(key, tag, count, modulus)` | `count` integers uniform in `[0, modulus)` derived from a 32-byte key and a tag (`modulus` a power of two). |
| `.output(name, value)` | Names an output. A composite value (tuple, vector, record) is flattened into outputs `name.0`, `name.1`, ... |
| `.transferred_output(name, value)` | An output exported as an artifact that consumers receive from the producer. |
| `.cached_output(name, value)` | An output exported as an artifact that consumers could recompute from public data; the stored bytes are a cache. |
| `.transferred_trapdoor_output(name, trapdoor)`, `.transferred_trapdoor_family_output(name, trapdoors)` | Export a trapdoor, or a family of trapdoors, with its secret. |
| `.build()` | Freezes the reachable part of the graph and checks its structure; returns `BuiltGraph`. |
| `built.validate(&bindings)` | Binds every compile parameter (`ParamEnv`), resolves rings, and checks every node's types and shapes; returns the `ValidatedGraph` that backends execute. |
| `built.validate_with_manifests(&bindings, &manifests)` | As `validate`, also checking artifact inputs against the manifests of their producers. |
| `built.operation_counts()` | How many times each primitive operation runs, loops included. |

Output methods consume the context and return it, so they chain. Output names must be unique
after flattening.

## 3. Value types

| Type | What it is | Input schema |
| --- | --- | --- |
| `Mat` | A matrix over a ring. | `MatType(ring.matrix_type(shape))` |
| `SmallMatrix` | A matrix whose coefficients are bounded by a known `B`: a secret, an error, a decomposition. | `SmallMatrixType` |
| `Preimage` | A bounded matrix produced as a preimage (by trapdoor sampling or gadget decomposition). | `PreimageType` |
| `Trapdoor` | A lattice trapdoor together with its public matrix. | `TrapdoorType` |
| `Int` | A runtime integer of arbitrary size. | `IntType` |
| `Bool` | A runtime Boolean. | `BoolType` |
| `Bytes` | A fixed-length byte string, for example a 32-byte seed. | `BytesType` |
| `Family<T>` | An ordered collection of `count` values of type `T`. | `FamilyType { element, count }` |
| tuples, `Vec<T>`, records | Composite values of the types above; they implement `GraphValue`. | the tuple or vector of the element schemas |

Most inputs have a shorter declaration method on `Ring` (section 5). Families of families are
not supported, but a family can be a loop state, a `select` candidate, or a subgraph argument.

A *bound domain* (`CoefficientBoundDomain`) says how a bound is read: `Global` means one integer
bound on the centered coefficient; `PerCrtLimb` means a bound on the signed residue in each CRT
limb separately.

## 4. Rings

| Call | Meaning |
| --- | --- |
| `Ring::new(crt_bits, crt_depth, N)` | A ring with a generated basis of `crt_depth` primes of `crt_bits` bits. `crt_bits` and `crt_depth` may be parameters; `N` is a `u32`. |
| `Ring::from_crt_moduli(moduli, N)` | A ring with an explicit, ordered basis. |
| `ring.slice_crt(start, end)`, `ring.prefix(end)` | The ring of a contiguous part of the basis (for example a lower BGV level). |
| `ring.select_crt(indices)` | The ring of chosen basis primes, in the given order. |
| `ring.concat_crt(&other)` | The ring whose basis is this basis followed by `other`'s (for example `Q * P` in key switching). |
| `ring.modulus()`, `ring.crt_depth()`, `ring.crt_modulus(i)` | `Q`, the number of primes, and prime `i`, as compile expressions. |
| `ring.ring_dimension()` | `N`. |
| `ring.matrix_type(shape)` | The matrix type of that shape over this ring. |

## 5. Inputs

Runtime inputs, supplied by name when the program runs:

| Call | Returns |
| --- | --- |
| `ring.input(name, shape)` | `Mat` |
| `ring.small_matrix_input(name, shape, bound)` | `SmallMatrix` |
| `ring.preimage_input(name, shape, bound)` | `Preimage` |
| `ring.bool_input(name)` | `Bool` |
| `ring.bytes_input(name, length)` | `Bytes` |
| `ring.input_family(name, count, shape)` | `Family<Mat>` |
| `ring.small_matrix_input_family(name, count, shape, bound)` | `Family<SmallMatrix>` |
| `ring.preimage_input_family(name, count, shape, bound)` | `Family<Preimage>` |
| `context.input(name, IntType)` | `Int` |
| `context.int_family_input(name, count)` | `Family<Int>` |

Each bounded input also has a `*_with_domain` form that takes a `CoefficientBoundDomain`.

Artifact inputs, imported from a program run identified by a `ProductionId`: `artifact_input`,
`small_matrix_artifact_input`, `preimage_artifact_input`, `bytes_artifact_input`,
`family_artifact_input`, `small_matrix_family_artifact_input`,
`preimage_family_artifact_input`, `trapdoor_artifact_input`, and
`trapdoor_family_artifact_input`. Each takes the production id, the artifact name, the same
shape and bound arguments as the runtime form, and an `ArtifactAvailability` (`Transferred` or
`Cached`, as in section 2). A family artifact is loaded member by member as the program reads it.

## 6. Constants

| Call | Returns |
| --- | --- |
| `ring.zero(shape)` | The zero matrix. |
| `ring.identity(size)` | The `size x size` identity. |
| `ring.gadget(rows, base, digit_count)` | The gadget matrix `G = I_rows ⊗ (1, b, ..., b^(k-1))`. |
| `ring.polynomial([c_0, c_1, ...])` | The scalar polynomial `c_0 + c_1 X + ...` (compile expressions, for example `ring.modulus().floor_div(2)`). |
| `ring.constant(shape, ConstantMatrix::...)` | Any constant: `Zero`, `Identity`, `UnitRow { index }`, `UnitColumn { index }`, `Gadget { base, small }`, `PowerOfBase { base, exponent }`, `Rotation { exponent }`, or `Polynomial { coefficients }`. |
| `ring.from_coefficients(&values)` | The scalar polynomial whose coefficients are a runtime `Family<Int>` of length `N`, reduced modulo `Q`. |
| `ring.from_evaluations(&values)` | The scalar polynomial with the given NTT evaluation slots, in the backend's native order. |
| `ring.pack_polynomial_coefficients(bits, coefficient_bits)` | The scalar polynomial whose coefficients are given as bits, coefficient by coefficient, least significant bit first. |
| `Int::constant(v)`, `Bool::constant(b)` | Integer and Boolean constants. |

## 7. Sampling

Every sampler draws fresh randomness on every run of the program (section 1, sampler placement).

| Call | Returns |
| --- | --- |
| `ring.uniform_residue(shape)` | A uniformly random matrix over `R_Q`. |
| `ring.uniform_interval(shape, min, max)` | Coefficients uniform in `[-1, 1]` or `[0, 1]`, the two supported intervals (ternary and binary secrets). |
| `ring.gaussian(shape, sigma, cutoff)` | A discrete Gaussian of width `sigma`; any coefficient beyond `cutoff` is redrawn. `mxx_backends::sampler::bounds::hard_cutoff_from_sigma_bound` gives the standard cutoff `floor(6.5 * sigma)`. |
| `ring.sample_trapdoor(rows, sigma, gadget_base, digit_count, preimage_cutoff)` | A lattice trapdoor with its public matrix; its preimages are bounded by `preimage_cutoff`. |
| `trapdoor.sample_preimage(target, shape)` | A `Preimage` `u` with `A u = target` for the trapdoor's public matrix `A`, whose coefficients are bounded by the trapdoor's cutoff; a candidate beyond it is redrawn. |
| `ring.gadget_trapdoor(rows, base, digit_count)` | The public, secret-free trapdoor of the gadget matrix; its "preimage sampling" is deterministic gadget decomposition. |
| `ring.hash_matrix(key, tag, shape)` | A matrix derived from a 32-byte `key` and a `tag` with a hash function (a random oracle). |
| `ring.hash_decomposed(key, tag, shape, base, digit_count)`, `ring.hash_small_decomposed(...)` | The gadget decomposition (regular or compact, as in section 8.4) of a hash-derived matrix, as a `SmallMatrix`. |
| `context.hash_int_family(key, tag, count, modulus)` | Hash-derived integers (section 2). |

**Seeds instead of large public matrices.** A public uniform matrix can be produced with
`hash_matrix` instead of `uniform_residue`. Then a party sends only the 32-byte key, and every
other party recomputes the same matrix from it. Hash-derived values are identical on the CPU and
the GPU. A `HashTag` separates different matrices derived from one key:

| Call | Meaning |
| --- | --- |
| `HashTag::from(b"domain".as_slice())` | A tag starting with a fixed byte prefix. |
| `tag.push(part)`, `tag!(a, b, ...)` | Appends typed parts: strings, compile expressions, or runtime `Int`s. Each part is framed with its type and length, so `(1, 23)` and `(12, 3)` never collide. |
| `tag.push_decimal(index)` | Appends a runtime integer in decimal form. |

## 8. Matrix operations

### 8.1 Arithmetic

| Call | Meaning |
| --- | --- |
| `a + b`, `a - b`, `-a` | Entrywise addition, subtraction, negation. Operands may be owned or borrowed (`&a + &b`). |
| `a * b` | The matrix product over `R_Q`. A `1 x 1` operand multiplies as a scalar. |
| `a * c`, `c * a` | Scaling by an integer constant `c` (a compile expression, or an `i32` on the left). |
| `a.mul_small_rhs(small)` | The product `a * small` with a bounded right operand, which the backends compute faster. |
| `preimage.mul_small_rhs(a)` | The product `a * preimage`. |
| `Mat::multi_row_gemm_accumulate(vec![(c_i, a_i, b_i), ...], bias)` | `sum_i c_i * a_i * b_i + bias` as one fused operation. |
| `a.tensor(b)` | The tensor (Kronecker) product. |

### 8.2 Shape

| Call | Meaning |
| --- | --- |
| `a.transpose()`, `a.t()` | The transpose. |
| `a.slice(rows, columns)` | A sub-matrix; each argument is `None` (all) or `Some(IndexRange { start, end })`. |
| `Mat::concat(ConcatAxis::Rows \| Columns \| Diagonal, vec![...])` | Stacks matrices vertically, horizontally, or block-diagonally. |
| `concat_rows![a, b]`, `concat_cols![a, b]`, `concat_diag![a, b]` | Shorthands for `concat`. |

### 8.3 Ring structure

| Call | Meaning |
| --- | --- |
| `a.ring_automorphism(k)` | The automorphism `X -> X^k` applied to every entry (`k` odd); used for BGV slot rotations. |
| `a.multiply_monomial(k)` | Multiplication of every entry by `X^k` for a runtime `Int` `k` of any sign, taken modulo `2N`; used for TFHE blind rotation. |

### 8.4 Gadget decomposition

| Call | Meaning |
| --- | --- |
| `a.decompose(base, digit_count)` | The gadget decomposition `G^{-1}(a)`, with digits bounded by `base / 2`: a `Preimage` with `digit_count` times as many rows, so that `G * a.decompose(...) = a`. |
| `a.small_decompose(base, digit_count)` | The compact decomposition against the compact gadget matrix, valid when the coefficients of `a` are below the smallest CRT prime. Its bound applies per CRT limb. |

### 8.5 Moduli and rings

Each conversion takes the destination ring explicitly.

| Call | Meaning |
| --- | --- |
| `a.modulus_switch(&dest)` | Scales every coefficient by `Q_dest / Q` and rounds; `Q_dest` must be an odd divisor of `Q`. |
| `a.reduce_modulus(&dest)` | Reduces coefficients into a divisor ring without scaling them. |
| `a.centered_rebase(&dest)` | Takes the centered representative modulo the whole source basis and re-encodes it in another ring. |
| `a.centered_round_divide(d)` | Divides centered coefficients by a positive constant `d` and rounds. |
| `a.block_mod_switch(&dest, t)` | An exact CRT block modulus switch to a strict subset of the basis, with plaintext correction factor `t`. |
| `a.rns_mod_up(&dest, digit_size, normalize)` | Extends contiguous CRT digits of `a` into the larger basis `dest`, stacking the digits by rows (the first step of hybrid key switching). |
| `a.rns_mod_down(&dest, t)` | Removes the auxiliary basis `P` with the BGV correction `(x + t*U) / P` (the last step of hybrid key switching). |
| `Mat::crt_recompose(levels, plaintext_moduli, coefficients, &dest)` | Decodes each row-vector level `i` modulo its plaintext modulus `t_i` (rounding `t_i * x / Q_i`) and combines the results with the given CRT reconstruction coefficients into `dest`. |

### 8.6 Coefficients and decoding

| Call | Meaning |
| --- | --- |
| `a.coefficients()` | All `N` coefficients of a scalar polynomial, as a `Family<Int>`. |
| `a.evaluations()` | All `N` NTT evaluation slots, in the backend's native order. |
| `a.extract_coefficient(i)` | Coefficient `i` of a scalar polynomial, as an `Int`. |
| `a.canonical_coefficient_bits(N, bits)` | The coefficients as bits, least significant first. |
| `a.threshold_decode_ints(t, length)` | For the first `length` coefficients `c`, the integer `round(t * c / Q) mod t`: the standard decryption rounding. |
| `a.threshold_decode_bools(2, length)` | The same decoding with `t = 2`, returned as `Bool`s. |

### 8.7 Bounded matrices and trapdoors

| Call | Meaning |
| --- | --- |
| `small.max_coefficient_bound()`, `small.bound_domain()` | The bound and how it is read (section 3). Also on `Preimage`. |
| `small.centered_rebase(&dest)`, `preimage.centered_rebase(&dest)` | Re-encodes a bounded matrix in another ring, keeping its bound. |
| `trapdoor.public_matrix()` | The trapdoor's public matrix `A`. |
| `trapdoor.preimage_max_coefficient_bound()` | The bound of its preimages. |

## 9. Scalars

| Call | Meaning |
| --- | --- |
| `x + y`, `x - y`, `x * y` | Integer arithmetic. Either side may be a Rust integer. |
| `x / y`, `x % y` | `q = floor(x / \|y\|)` and `r = x - \|y\| q`, so `0 <= r < \|y\|`; a zero divisor fails at run time. |
| `x.equal(y)`, `x.less(y)`, `x.less_equal(y)` | Comparisons returning `Bool`. Rust `==` and `<` are not graph operations. |
| `x.bit(i)` | Bit `i` of `x`, as a `Bool`. |
| `x.lift_to_constant_polynomial(ring.matrix_type((1, 1)))` | The scalar polynomial whose constant term is `x`. |
| `Int::evaluate(expr)` | An `Int` holding a compile expression. |
| `x.expression()` | The compile expression behind `x`, if it has one. |
| `p & q`, `p \| q`, `p ^ q`, `!p` | Boolean operations. Rust `&&` and `\|\|` are not graph operations. |
| `p.to_int()` | `0` or `1`. |

## 10. Families

| Call | Meaning |
| --- | --- |
| `family.at(i)` | Member `i`. `i` may be a compile expression or a runtime `Int`. |
| `Family::pack(vec![...])` | A family of values of one type. |
| `family.count()` | The number of members, as a compile expression. |
| `family.field(\|record\| record.part)` | The family of one field of each member, without a loop. |
| `m.matrix_vector_product(&v)` | For a `Family<Int>` `m` read as a row-major matrix: `M v`, that is `out[i] = sum_j M[i, j] v[j]`. |
| `m.vector_matrix_product(&v)` | `v^T M`, that is `out[j] = sum_i v[i] M[i, j]`. |
| `trapdoors.public_matrices()` | The public matrices of a `Family<Trapdoor>`. |

## 11. Loops, selection, and reusable bodies

Loops and bodies are recorded once in the graph, not copied per iteration, so a loop with
thousands of iterations stays a small program. The Rust closure runs once, while the graph is
built.

**`parallel(count, |i| body)`: a loop over independent iterations.** The iterations do not depend
on one another, so a backend may run them at the same time. `i` is the iteration index as an
`Int`; the loop returns a `Family` of the body's results in index order.

```rust
let ciphertexts = parallel(count, |i| {
    let e = ring.gaussian((1, 1), sigma.clone(), cutoff.clone()); // fresh per iteration
    Ok(&a * &s + e + messages.at(i))                               // one member per iteration
})?;
```

Values from outside the loop that the body reads become loop inputs automatically: `family.at(i)`
gives each iteration its own member (and `family.at(i + c)` a shifted one), and any other outer
value is shared by all iterations.

**`iterate(count, initial, |i, state| body)`: a loop that carries a state.** Each iteration
receives the previous state and returns the next one, of the same type; the loop returns the
final state. Use it for rounds of an evaluation, such as the layers of a circuit. A zero count
returns `initial`.

**`select(selector, candidates)`: choosing a value at run time.** Returns one of several values of
the same type: a `Bool` selector picks candidate 0 (false) or 1 (true), and an `Int` selector picks
the candidate with that index. Every candidate is computed; `select` chooses among the results,
it does not skip work. It is the replacement for an `if` on a runtime value.

**`Subgraph::define(name, input_schema, |input| body)`: a reusable named body.** A subgraph is
defined once and called many times with `subgraph.call(input)`; the graph keeps one copy of its
body. A backend may also run a named subgraph with a specialized kernel.
`call_with_canonical_input_exclusive_uppers` additionally states upper bounds of the canonical
coefficients of its arguments.

## 12. Errors

Construction and validation errors are `DslError` and `ValidationBuildError`: a shape or type
mismatch, a duplicate output name, a runtime integer used where a compile expression is required,
a family count mismatch, or a failed structural or type check. Numerical failures (division by
zero, an index out of range, a missing artifact) are reported by the backend when the program
runs.
