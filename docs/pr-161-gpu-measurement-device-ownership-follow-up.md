# PR #161 GPU Measurement Device-Ownership Follow-up

## Executive recommendation

Forward-port the device-ownership fix described below onto the latest PR #161
head, `1911f1ff4` (`Preserve preimage resources for GPU measurement replay`).
Do not merge the current working-tree implementation as a wholesale replacement
for PR #161: the latest PR contains preimage `operations` and accepted-width
replay changes that must remain intact. After the forward-port, rerun the PR
unit tests and the four-device A100 measurement before merging.

The follow-up belongs in PR #161 only to the extent that it fixes runtime and
benchmark-estimator GPU measurement setup ownership, with the CUDA diagnostics
that make a device-specific provisioning failure actionable. It does not belong
in PR #161 as a vehicle for the unrelated Exponent-LUT work listed under
Non-goals.

## Scope

In scope:

- runtime GPU fleet setup and prepared-storage ownership;
- bench-estimator setup-claim aggregation and replay metadata;
- CUDA kernel-provisioning diagnostics in `Matrix.cu`;
- regression tests for partial device placement in a configured fleet.

Explicitly out of scope:

- Exponent-LUT Gaussian/noise distribution changes;
- NAND circuit, native readout, or exponent-decoding changes;
- lattice-security or parameter-search changes;
- Lean changes and ignored Lean tests;
- any unrelated PR branch or source-tree cleanup.

## Reproduction and observed evidence

The reproduction used a four-device NVIDIA A100-SXM4-80GB fleet and the
Exponent-LUT depth-one measurement workload only as a way to exercise the GPU
measurement path. During the first failing run, a `HashSample` measurement in a
parallel body reached preprocessing measurement with setup claims reported for
only device 0 and device 1:

```text
setup_claim_counts=[device_0:...:16, device_1:...:16]
plan_count=16
each plan=1x2
GPU fleet submission failed: prepared setup is missing a configured device
```

The configured fleet had four devices. The failure therefore occurred before
the intended measurement could complete; it was not evidence of a cryptographic
noise or NAND correctness failure.

After the device-completeness fix, the A100x4 run exited successfully with exit
code 0 and one passing integration test. The run used:

- ring dimension `16384`, NAND depth `1`, four rounds, cadence `2`;
- CRT depth `9`, `log2(q)=360`, `crt_bits=40`, `base_bits=20`;
- average-case noise, sigma `4.0`, hard bound `26`;
- lattice-security target `100`, reused evidence `107` from the completed local
  estimator search (no Sage or parameter simulation was run on the pod).

Measured and estimated values from the successful run were:

| Stage | Actual seconds | Work seconds | Estimated elapsed | Critical path | Peak memory | Transmitted |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Preprocess | 125.107928682 | 13.541880782672079 | 26.260297620221206 | 0.045492546 | 40,720,773,120 | 32 |
| Online | 55.656360572 | 7.315794933296733 | 12.532335742836574 | 0.034073123 | 38,751,547,392 | 4,955,071,232 |

The round trip completed with strict `q/4` residual acceptance and the audited
bound. The output remained native `X^0/X^1`, with `stride_log2=0` and no
production readout stage. The source archive was transferred by SCP; GitHub was
not pushed. The archive SHA-256 was
`7ec62ac8dd8198c0e079004b66e56717adc4db8517e174ca6bc4117bce904a87`.

Important provenance caveat: this successful A100 source was based on the
earlier `fc15` line plus integration refactors. It was not the exact latest
`1911f1ff4` tree. Treat the pass as evidence that the ownership diagnosis and
the proposed setup completion are plausible, not as sign-off for the current
PR head. The fix must be forward-ported onto `1911f1ff4` and rerun there.

### Exact reproduction command and retained evidence

The successful A100 run used the following command and essential environment.
The block uses a caller-selected working directory and relative artifact path
so it can be copied to another host without embedding a local absolute path:

```bash
cd "${MXX_RUN_DIR:?set MXX_RUN_DIR to the checked-out source}"
export RUST_LOG=debug
export CUDA_ARCH=80
export CUDA_MODULE_LOADING=EAGER
export MXX_EXPONENT_LUT_NAND_LARGE_PARAMETER_RUN=1
export MXX_PRIMITIVE_TEST_RING_DIMENSION=16384
export MXX_EXPONENT_LUT_NAND_DEPTH=1
export MXX_EXPONENT_LUT_NAND_CRT_BITS=40
export MXX_EXPONENT_LUT_NAND_MAX_CRT_DEPTH=10
export MXX_EXPONENT_LUT_NAND_SECURITY_BITS=100
# This explicit mode skips Sage and the target comparison; it does not consume
# or require a prevalidated-security environment variable.
export MXX_EXPONENT_LUT_NAND_SKIP_SAGE=1
export MXX_EXPONENT_LUT_NAND_NOISE_MODEL=average
export MXX_EXPONENT_LUT_REFRESH_SECURITY_BITS=1
export MXX_EXPONENT_LUT_BENCH_WARMUPS=1
export MXX_EXPONENT_LUT_BENCH_ITERATIONS=1
export MXX_EXPONENT_LUT_NAND_ARTIFACT_DIR=artifacts/native-final

cargo test -r -p mxx-exponent-lut --features gpu --test gpu_nand_circuit \
  test_gpu_nand_circuit_noise_search_and_artifact_round_trip \
  -- --exact --nocapture
```

The retained evidence is local and ignored by Git; it is not automatically
part of PR #161. Relative evidence names are:

- `logs/runpod-nand-a100x4-20260915/nand-a100x4-native-final.log`;
- `logs/runpod-nand-a100x4-20260915/nand-a100x4-native-final-metrics.csv`;
- `logs/runpod-nand-a100x4-20260915/nand-a100x4-native-final.exit`;
- `logs/runpod-nand-a100x4-20260915/nand-a100x4-native-final-manifest.bin`;
- `logs/runpod-nand-a100x4-20260915/nand-a100x4-native-final-artifact-inventory.txt`;
- `logs/runpod-nand-a100x4-20260915/source-manifest.txt`.

The retained final log SHA-256 is
`a8b9e47615fd3f0d0122bf1314ee7140922ed7aa3c6ba7a3ddf18aca5e56cf86`, and the
metrics CSV SHA-256 is
`263fde20c38f66e3fa67df9d948a3b497a7d5a57b5c6d31815906620b4dfed82`.

## Repetition and validation policy

The exact forward-port validation must first compile the GPU test surface using
the command required by `GPU.md`:

```bash
cargo test -r --workspace --lib --features gpu --no-run
```

It must then run the following exact filtered command three to five times with
identical parameters:

```bash
cargo test -r -p mxx-exponent-lut --features gpu --test gpu_nand_circuit \
  test_gpu_nand_circuit_noise_search_and_artifact_round_trip \
  -- --exact --nocapture
```

Record the resulting `round_trip_repetitions`, `passes`, and `failures` in the
validation handoff. The earlier
A100 evidence contains one successful run only; it is useful diagnosis and
performance evidence, but it is not exact-head sign-off. A repetition count of
300 is warranted only if the forward-port investigation discovers a
synchronization or nondeterministic race. The currently diagnosed failure is a
deterministic prepared-setup ownership omission, so 300 repetitions are not
required by the present evidence.

## Root cause

The original setup aggregation grouped prepared claims by concrete matrix type
only. That erased the device identity, even though a slot key is meaningful only
within its device and native context. The subsequent ownership-preserving fix
correctly retained device-specific claims and exposed the real partial-fleet
case: the selected workload used devices 0 and 1, while devices 2 and 3 were
still configured in the measurement fleet.

`prepare_memory` intentionally seals a complete fleet. It requires at least one
prepared storage owner for every configured device so that each execution owner
has an observed allocation epoch and a complete memory ledger. Consequently,
device-specific setup without a setup owner for unused configured devices
failed with `prepared setup is missing a configured device`.

This is a provisioning/ownership error, not a reason to copy active claims to
another GPU. Copying a claim across devices can substitute a slot from the
wrong native execution owner and invalidate the replay contract.

## Required forward-port changes

Forward-port the following behavior onto the latest PR #161 implementation.
Preserve all latest PR #161 preimage `operations` and accepted-width replay
logic; do not copy the current `gpu_admit.rs` wholesale or revert those changes.

1. Represent selected setup claims as:

   ```text
   (device, ConcreteMatrixType, Vec<GpuTracedClaim>)
   ```

2. Preserve the region/input device throughout setup selection. Deduplicate
   using device, native context, and slot identity together; a slot key without
   its device/context is not globally unique.

3. Propagate the device-bearing tuple through `GpuSelectedNode` and
   `BatchRequest` into measurement storage preparation and diagnostics.

4. Replicate shared workspace claims deliberately per device. Shared means
   shared by the sibling batch on one device; it does not mean that one device's
   native workspace may be submitted on another device.

5. In `prepare_measurement_storage`, filter claims by their owner device. For
   a configured device with no real selected storage, create only a minimal
   1x1 terminal-level evaluation placeholder using that device's own registered
   parameters. The placeholder exists solely to satisfy the complete-fleet
   setup seal and execution-owner ledger. Do not clone active matrix or
   workspace claims onto the unused device.

6. Keep the setup error context from `gpu_batch_measurement`: include device,
   concrete type, claim counts, plans, concrete argument types, and concrete
   output types so a future ownership mismatch identifies the responsible
   class.

7. Keep the `Matrix.cu` kernel-provisioning diagnostics that identify the
   inventory index and distinguish kernel inventory, `cuLaunchKernel`, and
   tensor-symbol lookup failures. These diagnostics are independent of the
   setup placeholder and should remain narrowly scoped.

## Regression tests required

Add or retain targeted tests covering:

- a four-device configured fleet where only a strict subset has active claims;
  setup sealing must succeed and produce one owner per configured device;
- no cross-device claim substitution: a claim selected for device 0 must not
  satisfy device 1, even when concrete types and slot keys are identical;
- equal slot keys in distinct native contexts remain distinct setup claims;
- shared resources are present on every device that must execute the sibling
  batch, with no accidental collapse across devices;
- preimage replay continues to preserve the latest PR #161 `operations` and
  accepted-width behavior;
- measurement error diagnostics include the device-bearing setup information;
- CUDA kernel provisioning reports the failing inventory index or symbol class.

The multi-device setup test should be runnable without a full Exponent-LUT
integration workload where possible. GPU-dependent tests may skip cleanly when
no CUDA device is available, but they must execute on the A100x4 validation
host.

## Validation and acceptance criteria

Before merge, run on the exact forward-ported `1911f1ff4` tree:

- PR #161 targeted runtime and bench-estimator unit tests;
- the device-ownership regression tests above;
- the preimage replay tests from PR #161;
- CUDA compile and kernel-provisioning checks;
- a four-device A100 measurement with setup claims active on only a subset of
  devices;
- the complete measurement path through preprocess and online replay;
- `cargo +nightly fmt --all` and `git diff --check`.

Acceptance requires that the four-device run no longer reports a missing
configured device, no setup claim crosses an execution owner, and the PR #161
preimage replay behavior remains unchanged. The measurement must finish with a
durable exit code and logs/metrics sufficient to distinguish setup, execution,
and correctness failures.

## Suggested commit split

1. Runtime: preserve device/context ownership in setup claims and add the
   per-unused-device setup placeholder.
2. Bench-estimator: propagate device-bearing setup tuples, deliberately
   replicate shared claims per device, and improve failure diagnostics.
3. CUDA: add kernel inventory and symbol-resolution diagnostics.
4. Tests and documentation: add the partial-fleet regression coverage and this
   follow-up report.

Keeping these commits separate makes it possible to review ownership semantics
without mixing them with preimage replay changes or application-level tests.

## Risks and compatibility

The placeholder allocates a minimal native matrix on each otherwise unused
configured device. It slightly increases setup memory, but it prevents an
incomplete fleet ledger and does not add active workload capacity or alter
placement. Its parameters must come from the same device, not from device 0.

The main compatibility risk is forward-porting into a changed `gpu_admit.rs`:
blindly applying the current branch can remove PR #161's preimage resource
operations or accepted-width replay behavior. Resolve conflicts by preserving
the latest PR logic and reapplying only the ownership tuple, filtering,
shared-resource, placeholder, and diagnostic changes described above.

## Non-goals

This follow-up does not change Exponent-LUT sampling, CBD/Gaussian contracts,
NAND output encoding, readout design, q/4 policy, lattice-estimator settings,
security claims, Lean tests, or parameter search. It does not authorize a
GitHub push or replace the required exact-head A100 validation.
