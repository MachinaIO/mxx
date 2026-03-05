# Design: GGH15 Arbitrary-Polynomial Public Lookup

## Purpose

Define a concrete extension of `src/lookup/ggh15_eval.rs` from constant-input public lookups to finite-domain lookups over non-constant representative polynomials, while keeping `GGH15BGGPubKeyPltEvaluator` at one LUT preimage per row and one gate-preimage family per gate.

In this document, "arbitrary-polynomial" means "arbitrary within a finite representative set chosen by the caller," not "every mod-`q` polynomial in `R_q`." The representative set must remain inside the coefficient regime where the existing `small_decompose` / `small_gadget_matrix` relation is exact.

## Current Limitation

The present GGH15 lookup path mixes two different notions of "row input":

1. `GGH15BGGEncodingPltEvaluator::public_lookup` first computes the scalar selector key `rho(x) = x.to_const_int()` and calls `PublicLut::get(params, rho(x))`.
2. `GGH15BGGPubKeyPltEvaluator::sample_lut_preimages` hard-codes the row representative as the constant polynomial `idx * 1` through `M::P::from_usize_to_constant(params, idx)`.

That is harmless only when the admitted runtime polynomial for selector key `k` really is the constant polynomial `k * 1`. As soon as the lookup domain assigns key `k` to a different representative `x_k`, the gate-side cancellation no longer matches the LUT-side preimage.

The failure is algebraic, not only an indexing problem. The current row preimage encodes the compact decomposition of `idx * I`, while the gate-side evaluation multiplies by the compact decomposition of the actual plaintext polynomial `x`. If `x != x_k = idx * 1`, the residual term does not cancel.

## Core Decision

Bind every LUT row `k` to:

- an explicit selector key `rho_k`,
- an explicit representative input polynomial `x_k`,
- the row output `y_k`,

and use `x_k` in the LUT auxiliary preimage equation:

`B1 * K_k = W_I + W_gy * G^{-1}(G * y_k) + W_v * V_k + W_vx * D_sm(x_k) * V_k`

where:

- `x_k` is the exact lookup-domain representative for row `k`,
- `rho_k` is the scalar selector key consumed by the existing `PublicLut::get` API,
- `y_k` is the row output embedded in the same constant-polynomial form used today,
- `V_k` is the hashed matrix already derived from `(lut_id, k)`,
- `D_sm(x_k)` is the compact small decomposition of `x_k * I_m`,
- `G` is the existing gadget matrix,
- `B1` is the existing LUT trapdoor matrix.

The gate-side preimages remain one family per gate:

- `B0 * D_g^1 = S_g * B1 + e_1`
- `B0 * D_g^I = S_g * W_I + A_out + e_I`
- `B0 * D_g^gy = S_g * W_gy - G + e_gy`
- `B0 * D_g^v = S_g * W_v - A_x * U_g + e_v`
- `B0 * D_g^vx = S_g * W_vx + U_g * G_sm + e_vx`

Here `G_sm` is the existing compact gadget matrix used with `small_decompose()`. No new gate-side preimage family is introduced.

## Mathematical Construction

Let `R_q = Z_q[X] / (X^N - 1)` be the polynomial ring. Let `m = d * params.modulus_digits()`. For every row representative `x_k in R_q`, define:

`D_sm(x_k) = identity(m, x_k).small_decompose()`

By the `small_decompose()` contract already used in the matrix layer,

`G_sm * D_sm(x_k) = x_k * I_m`

The runtime encoding path already computes the same object for the live plaintext `x` through `small_decomposed_identity_chunk_from_scalar(...)`, except it does so chunk by chunk.

For a gate `g` selecting LUT row `k`, define the reconstructed constant part:

`R_g,k(x) = D_g^I + D_g^gy * G^{-1}(G * y_k) + D_g^v * V_k + D_g^vx * D_sm(x) * V_k - D_g^1 * K_k`

Applying `B0` gives:

`B0 * R_g,k(x)`

`= A_out - G * y_k - A_x * U_g * V_k + U_g * x * V_k`

`  + S_g * W_vx * (D_sm(x) - D_sm(x_k)) * V_k + noise`

If the row representative matches the runtime plaintext, `x = x_k`, the `W_vx` term cancels exactly and the remaining expression is:

`A_out - G * y_k - A_x * U_g * V_k + U_g * x_k * V_k + noise`

The runtime randomized term already computed in `GGH15BGGEncodingPltEvaluator` is:

`c_x * U_g^{dec} * V_k`

For a BGG encoding `c_x = s * A_x - x_k * s * G + err_x`, this expands to:

`s * A_x * U_g * V_k - s * x_k * U_g * V_k + noise`

Because multiplication is over the commutative ring `R_q`, the `A_x * U_g * V_k` and `x_k * U_g * V_k` terms cancel the matching pieces in `c_B0 * R_g,k(x_k)`, leaving an encryption under `A_out` of `y_k`.

This is the same cancellation pattern as the current constant-only construction, but with `x_k` as a full ring element instead of the special case `idx * 1`.

## Selector-Key Invariant

The lookup domain is still finite, so the runtime path may keep using a scalar selector key as long as that key is paired with a canonical lift back into the ring.

Write:

- `rho : Domain -> {0, ..., L - 1}` for the selector-key function used at runtime,
- `lift(k) = x_k` for the representative polynomial stored for row `k`.

The required invariant is:

`lift(rho(x)) = x` for every admissible runtime plaintext `x` in the lookup domain.

This means the implementation is not trying to support every polynomial in `R_q`; it supports a finite set of canonical representatives, each tagged by a scalar key. The current runtime call `x.to_const_int()` is one concrete choice for `rho`. The later implementation may keep that API shape, but only for domains where `to_const_int()` is a left inverse of the chosen representative map.

## Admissible Representative Domain

The proposal relies on the current compact decomposition identity

`G_sm * D_sm(x) = x * I_m`

as implemented by `identity(m, x).small_decompose()` and by `small_decomposed_identity_chunk_from_scalar(...)`.

That identity is not unconditional. The current matrix-layer contract states that `small_decompose()` is exact only when coefficients are bounded by the smallest CRT modulus. The runtime chunked helper inherits the same contract because it is defined by slicing `identity(m, x).small_decompose()`.

As a result, this design currently applies to finite representative sets `X_adm ⊂ R_q` such that every admissible input `x in X_adm` satisfies the `small_decompose` coefficient bound. Examples include LSB / bit-decoded representatives, sparse ternary representatives, and other small-coefficient encodings.

It does not yet justify unrestricted mod-`q` representatives. Supporting every polynomial in `R_q` would require a different exact decomposition relation for both the LUT-side stored representative and the runtime stage-5 linearization.

## Why This Avoids the Forbidden Blow-Up

The forbidden approach would split `x` into `N` coefficient-level subproblems and sample separate gate or LUT preimages per coefficient. That would multiply the number of trapdoor solves by the ring dimension.

This design does not do that.

- `D_sm(x_k)` has the same matrix shape for every polynomial `x_k`. Only the polynomial entries change.
- `D_g^vx` is a single universal linearizer per gate: after sampling it once, the runtime multiplies it by `D_sm(x)` to obtain `U_g * x`.
- `K_k` remains one preimage per LUT row, because each row already corresponds to one canonical representative `x_k`.
- The matrix dimensions grow with gadget digits exactly as they do today. They do not gain an extra factor of `params.ring_dimension()`.

In other words, the ring dimension is absorbed inside each polynomial entry, not by duplicating trapdoor solves.

## Required Row-Representative Invariant

The tuple `(row_idx, selector_key)` must identify a unique representative polynomial `x_k`.

If two different input polynomials share the same selected row but have different representatives, the residual term

`S_g * W_vx * (D_sm(x) - D_sm(x_k)) * V_k`

does not vanish for one of them. A later implementation therefore needs an explicit invariant:

- one row index,
- one selector key,
- one representative input polynomial,
- one output value.

Equivalently, once runtime selection chooses row `k`, the evaluator must know that the ciphertext plaintext is exactly `x_k`, not merely that it decodes to the same row number.

This agreement must also be stable across checkpoint/resume. LUT auxiliaries are persisted under identifiers of the form `..._idx{row_idx}`, and pending-row detection also filters by `row_idx`. Therefore the stored table must make `selector_key -> row_idx -> x_k` a single authoritative mapping during setup, resume, and evaluation.

If callers need multiple semantic aliases for the same output, they must either:

- canonicalize them to one representative before the lookup so that `lift(rho(x)) = x`, or
- allocate separate row indices with separate `K_k` rows.

## Data-Model Guidance

The later implementation should extend `PublicLut<P>` so that row representatives are explicit repository data, not reconstructed from `0..len`.

A concrete shape is:

- `PublicLutRow<P> { selector_key: u64, input_repr: P, row_idx: u64, output: P::Elem }`
- `PublicLut<P>` stores `rows: Vec<PublicLutRow<P>>`
- `PublicLut::get` remains the runtime selector over `u64`
- add a setup-time iterator or lookup helper that exposes `selector_key`, `input_repr`, `row_idx`, and `output` to `GGH15BGGPubKeyPltEvaluator`
- use that same stored row table as the resume/checkpoint source of truth for `row_idx`

This preserves the current selector shape in `GGH15BGGEncodingPltEvaluator::public_lookup`, but it gives `GGH15BGGPubKeyPltEvaluator` the missing representative polynomial `x_k` during auxiliary-matrix sampling.

For legacy constant-input call sites, a compatibility constructor can fill `selector_key = idx`, `input_repr = idx * 1`, and `row_idx = idx`. For arbitrary-polynomial domains, a new constructor should accept explicit row descriptors and document the required invariant `input_repr.to_const_int() == selector_key` whenever the runtime path still uses `to_const_int()`.

## Concrete Implementation Path

The later code change should follow this sequence.

1. Extend `src/lookup/mod.rs` so `PublicLut<P>` can enumerate explicit row descriptors `(selector_key, x_k, row_idx, y_k)`.
2. Keep a constant-domain compatibility constructor that derives those descriptors from `idx`.
3. In `GGH15BGGPubKeyPltEvaluator::sample_lut_preimages`, replace the current `(idx, y_poly)` batch input with row objects carrying `(selector_key, x_k, row_idx, y_k)`.
4. Replace `from_usize_to_constant(params, idx)` in the LUT preimage equation with the stored representative `x_k`.
5. In the resume path, compute pending rows from stored row descriptors rather than by scanning `0..plt.len()` and rebuilding constant polynomials, so resumed checkpoints reuse the same `row_idx -> x_k` mapping that produced `K_k`.
6. In `GGH15BGGEncodingPltEvaluator::public_lookup`, keep the existing runtime multiplication by `small_decomposed_identity_chunk_from_scalar(params, m, x, ...)`; it already works for admissible `x`, provided the selected row satisfies `lift(rho(x)) = x`.
7. Add CPU and GPU tests where the lookup input is a non-constant representative polynomial, for example `from_usize_to_lsb(params, k)`.

## Expected Code Impact

The design intentionally keeps most of `src/lookup/ggh15_eval.rs` unchanged.

- Gate-side stage 1 through stage 5 stay structurally identical.
- The runtime encoding equation stays structurally identical.
- The main change is that LUT-row setup must consume explicit `(selector_key, x_k)` row descriptors.
- Resume bookkeeping must skip completed rows by `row_idx`, not by a constant-polynomial reconstruction.

That narrow change surface is a direct consequence of treating each admissible input polynomial as one ring element per row instead of `N` coefficient fragments.

## Validation Plan For The Future Implementation

The later implementation should be accepted only if all of the following hold.

1. A new CPU test in `src/lookup/ggh15_eval.rs` uses non-constant plaintext inputs and shows the lookup result equals the row output value.
2. The existing constant-input tests still pass unchanged.
3. The GPU version of the same non-constant test passes when the `gpu` feature is enabled.
4. Auxiliary-matrix counts remain proportional to the number of rows and gates already present in the circuit. The implementation must not add loops that sample separate preimages per polynomial coefficient.

## Example Domain

For a binary-decoded domain, a valid row family is:

- `rho(x) = x.to_const_int()`
- `x_k = P::from_usize_to_lsb(params, k)`
- `row_idx = k`
- `selector_key = k`
- `y_k = f(x_k)`

Then `rho(x_k) = k`, so `lift(rho(x_k)) = x_k` holds on the admitted domain. The runtime selector may still recover `k` from `x_k.to_const_int()`, but the LUT preimage must use the full polynomial `x_k`, not the constant polynomial `k * 1`. That single change is what makes the GGH15 cancellation work for arbitrary polynomial representatives without adding coefficient-wise preimages.
