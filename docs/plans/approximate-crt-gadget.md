# Approximate CRT gadget implementation plan

## Contract

Implement Bernard--Joye, ePrint 2024/909, Section 3.2, followed by the existing
exact balanced radix decomposition. Keep `base_bits <= crt_bits / 2`.
The ordered CRT basis retains its full arithmetic modulus. `dropped_moduli = k`
selects the last `k` moduli as the low part `P`; require `0 <= k < crt_depth`.
For each coefficient `a`, compute

```
S(a) = sum_u (P / p_u) * centered((P / p_u)^(-1) * a mod p_u).
```

Decompose `centered(a - S(a) mod q_j)` for each retained tower using the
existing balanced radix rule. The gadget keeps the retained CRT idempotent
columns times radix powers. It has `(crt_depth-k)*ceil(crt_bits/base_bits)`
digits per source row. Thus `G D(A) = A - E(A)` modulo the unchanged full
modulus, with the same coefficient bound `base/2` on digits and
`||E(A)|| <= k*floor(P/2)`. The inclusive residual bound is represented
explicitly; a strict bound is this value plus one. At `k=0`, the residual is zero.
Residuals are deterministic and correlated across towers; they are not
independent Gaussian samples.

Compact/small decomposition retains its separate small-integer contract.
Exact trapdoor/preimage sampling must reject an approximate parameter set:
the shortened gadget is zero modulo `P`, so arbitrary exact syndromes are
not in its image. This work does not introduce approximate Gaussian sampling.

## API and proof boundary

Pass `Some(k)` as the final argument of `DCRTPolyParams::new` or
`GpuDCRTPolyParams::new`; `None` selects exact mode. Arithmetic modulus and CRT
ordering do not change. `PolyParams::modulus_digits()` reports the shortened
regular gadget digit count, and `gadget_error_bound()` reports the inclusive
residual bound. For a matrix with `d` rows the gadget has `d * modulus_digits()`
columns. The CPU standalone polynomial `decompose_base` retains its existing
full-integer radix contract; use matrix decomposition for the CRT gadget.

`mxx_bgg::encoding::multiplication_error_bound` computes the worst-case product
error, including the residual weighted by the secret and plaintext. It is an
explicit caller API, not an automatic change to every application simulator.
Existing application parameter searches construct exact-mode parameters.

Lean's `gadgetDecomposeRuns` binds the shortened, corrected decomposition and
the bound on its actual residual to the same layout and target. For `k > 0`,
this is an invocation-local primitive premise, as requested; the theorem does
not derive the paper's residual bound from the correction algorithm. The BGG
algebra and bound propagation are proved from that premise. For `k = 0`,
exact reconstruction and the zero residual bound are proved without that premise.

## Ordered stages

1. Extend CPU/GPU CRT parameters, shape calculations, and parameter identity.
2. Implement CPU and GPU gadget generation and corrected decomposition,
   including compact digit storage, ranges/chunks, and production runtime paths.
3. Propagate layout metadata and error bounds through runtime and exports;
   prevent exact-only samplers from silently consuming approximate layouts.
4. Extend Lean reconstruction contracts and noise propagation. BGG multiplication
   adds `x_L * s * E(A_R)` to its existing two error terms. Bind the residual
   and its bound to the same concrete layout/decomposition.
5. Add focused unit tests for exact mode, one/multiple dropped towers, output
   shape, digit bounds, residual bounds, and CPU/GPU agreement. Build Lean
   changes incrementally and check the resulting application obligations.
6. Use one GPT-6 medium review agent, address findings, run warning-free CPU
   and GPU release unit builds, then all non-ignored workspace unit tests
   (CPU and GPU) and relevant Python unit tests. Repeat new GPU round trips
   with the same built binary at least three times. Do not run integration tests.

## Validation record

The first full CPU run with the default test-harness concurrency was terminated
by SIGKILL in the gadget crate, without an assertion failure. The cause was not
established. The complete CPU rerun and GPU suite use
`RUST_TEST_THREADS=1`; this changes only concurrent test execution, not Rayon
parallelism inside the implementation or the test parameters.

- CPU/GPU CRT parameter fields, shortened regular gadget dimensions, CPU shared
  correction, GPU-resident correction, and compact digit output are implemented.
- `cargo test -r -p mxx-primitives --lib test_matrix_approximate_gadget_reconstruction`
  passed (one test covering `k=0,1,2`, digit/residual bounds and chunk/compact agreement).
- GPU release workspace unit binaries compiled successfully. The new
  `test_gpu_approximate_gadget_reconstruction` passed three consecutive executions
  of the same binary on the local RTX 4080 SUPER, comparing CPU/GPU results,
  coefficient/evaluation inputs, compact multiplication, and ranged gadget columns.
- Interim GPT-6 medium review identified the correction launch grid limit,
  standalone decomposition count regressions, and three existing tests with
  base widths above the requested half-width constraint. Those findings were
  addressed. The same reviewer accepted the implementation and conditional
  proof contract, and accepted the final localized Lean proof adjustments.
- Lean runtime now has concrete corrected/shortened decomposition definitions,
  an invocation-local bound on the actual residual, and an explicit exact-mode
  reconstruction corollary. The runtime module typechecked directly using the
  primitive package environment. The primitives, runtime, BGG, gadgets, and IR
  Lake packages and the generated exact/approximate runtime fixture compile.
  The production Diamond certificate unit test also passes (175.19 seconds),
  checking the generated IR through the final certificate with the current
  handwritten proofs. Its security estimator is a test double; this is not a
  GPU run or a practical security-parameter assessment.
- BGG's generic algebraic and integer-bound lemmas include the secret-weighted
  residual. The corresponding Rust worst-case bound is available to callers.
- All 41 runtime CPU unit tests pass. Python unit tests pass: seven repository
  validation tests and three Lean candidate selector tests.
- The final primitive CPU suite passes all 84 tests. The final GPU primitive
  suite passes all 167 tests, and the GPU runtime suite passes 74 tests with one
  existing ignored test. The final GPU round-trip binary passes 3/3 repeats.
- Both final CPU and GPU release workspace unit builds (`--no-run`) complete
  without warnings. `cargo +nightly fmt --all` and `git diff --check` pass.
- Axiom inspection of exact/approximate reconstruction, runtime approximation,
  BGG residual algebra, and its bound reports only `propext`, `Classical.choice`,
  and `Quot.sound`.
- Complete non-ignored release workspace unit suites pass, each covering all
  ten workspace crates and exiting with status zero:

  | Configuration | Passed | Failed | Ignored |
  | --- | ---: | ---: | ---: |
  | `RUST_TEST_THREADS=1 cargo test -r --workspace --lib` | 472 | 0 | 7 |
  | `RUST_TEST_THREADS=1 cargo test -r --workspace --lib --features gpu` | 617 | 0 | 9 |

  GPU execution used the local RTX 4080 SUPER outside the sandbox. The gadget
  crate dominated runtime: 5956.69 seconds in the CPU suite and 6000.66 seconds
  with the GPU feature. These counts exclude the separately run, normally ignored
  Diamond certificate test described above. Integration tests were not run.
- Implementation, layout/identity audit, the single-reviewer checks, and the
  requested unit-test validation are complete. Approximate Gaussian preimages
  and an unconditional Lean derivation of the paper's residual bound remain
  outside the implemented contract.
