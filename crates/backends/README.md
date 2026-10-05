# mxx-backends

`mxx-backends` runs mxx programs. It implements the concrete lattice arithmetic (polynomials
and matrices over RNS rings, samplers, trapdoors, and codecs, built on OpenFHE) and executes
validated `mxx-ir-core` graphs on the CPU or, with the `gpu` feature, on one or more GPUs. It is
the only crate with concrete arithmetic, and it depends only on `mxx-ir-core`.

[SPEC.md](SPEC.md) describes every API a user needs to execute a program: ring parameters, the
CPU and GPU backends, inputs, planning, execution, outputs, and artifact stores.

## Contents

| Module or directory | What it provides |
| --- | --- |
| `poly`, `matrix`, `element`, `modulus` | Polynomial parameters, polynomials, and matrices over RNS rings. |
| `sampler` | Uniform, Gaussian, hash-derived, trapdoor, and preimage samplers, and their cutoffs. |
| `openfhe_guard` | CRT basis checks and OpenFHE table initialization. |
| `backend` | Runtime values and the CPU and GPU backends. |
| `executor` | The CPU executor. |
| `gpu_runtime` (`gpu` feature) | The GPU runtime: planning, execution, options, and limitations. |
| `artifact`, `session`, `transcript`, `authority` | Artifact stores, durable sessions, sampling transcripts, and the prepare-then-run boundary. |
| `host_control` | Structural node dispatch shared by execution and measurement. |
| `env` | Environment variables read by native primitives and backends. |
| `gpu/` | Shared native CUDA/HIP sources; `poly::dcrt::gpu` describes them. |
| `native/` | C++ adapters to OpenFHE, bridged with `cxx`. |
| `benches/` | Matrix-product and preimage-sampling benchmarks for the CPU and the GPU. |

## Design

```text
ValidatedGraph --> executor::execute(.., &mut CpuDcrtBackend, ..)   CPU, node by node
               '-> GpuRuntime::plan(graph, inputs) -> GpuExecutionPlan
                   GpuRuntime::execute(&mut plan, inputs)             GPU, replaying GPU graph regions
```

- **One graph, two executors.** The CPU executor walks each scope in order and batches the
  instances of a parallel loop into backend requests. The GPU runtime compiles the whole graph
  into GPU graph regions once and replays them. CUDA uses native conditional graphs;
  HIP schedules control regions at device-to-host predicate boundaries. Both take the same validated graph.
- **Plan once, execute many times.** GPU planning tries candidate degrees of parallelism,
  actually allocates and times each one, rejects those that exceed the memory budget, and
  freezes the fastest. Execution only rebinds inputs and replays the plan. There is no eager GPU
  matrix API: all GPU computation goes through a plan.
- **Values stay where they are produced.** GPU outputs stay on the device and can be bound as
  inputs of any plan. Artifacts stream to and from a store during execution, and a selected
  family member is loaded alone.
- **Trusted inputs.** Execution accepts complete inputs prepared for the exact validated graph.
  It checks what addressing needs, not the contents of resident values, and artifact stores must
  return intact payloads.
- **Multiple GPUs through copies.** A plan spreads matrix products, preimage tiles, and loop
  lanes over its devices, and moves data between devices only with copy nodes.

## GPU backend selection

The `gpu` Cargo feature keeps the same public Rust API. `MXX_GPU_BACKEND` selects one native
backend at build time: `cuda` by default, or `hip`. CPU builds need neither SDK. CUDA uses
`CUDA_ARCH` (default `89`). HIP requires ROCm, `HIPCC` (or the SDK's `hipcc`), `ROCM_PATH`, and
an explicit `HIP_ARCH` matching the target `gfx` architecture. Invalid selectors, missing
SDKs, and unsupported architecture values are build errors. Use separate target directories
when preserving backend artifacts; `scripts/run_tests.sh --gpu-compile` defaults to
`target/gpu-cuda` or `target/gpu-hip`.

Shared kernels live in `crates/backends/gpu/`; `GpuPlatform.h` contains the platform boundary.
Application-owned subgraph kernels include only the public
`crates/backends/gpu/include/SubgraphKernel.h`. Build metadata passes the selected backend,
SDK/compiler, architecture, and include directory to dependents. GPU plans identify the native
backend and device; canonical artifacts remain independent of GPU plan identity.

HIP control scheduling adds device-to-host control transfers, event waits, and region launches.
These costs belong in planning and execution measurements. Logical duplicate device mappings
are scheduling checks, not evidence of multiple physical GPUs or peer transfer support.

| Validation target | Current evidence for this change |
| --- | --- |
| CPU lib compile/test | Passed; 629 CPU tests, zero failures |
| CUDA lib compile and device regression | Current compile passed; three smoke repetitions in each logical mode passed; nested control 900 runs across three logical modes passed; matrix named-call regression passes after planner correction; latest long KHE probe pending |
| HIP lib compile and BGV/device execution | ROCm 7.0.0 compile passed for gfx1100/gfx942; 652 CPU-only tests passed; AMD device execution unverified |
| AMD wave32, wave64, and multiple physical GPUs | Unverified; separate hardware runs required |
| TFHE native GPU kernel on AMD | Deferred; the HIP accessor returns `None` |

Detailed logs, source evidence, and incomplete hardware gates are recorded in
[`AMD_GPU_VALIDATION.md`](../../AMD_GPU_VALIDATION.md). Compilation does not establish AMD
runtime support.

The ordinary CI jobs remain CPU-only. Manual GPU CI requires configured JSON runner labels in
`MXX_CUDA_RUNNER` or `MXX_HIP_RUNNER`, the SDK/OpenFHE/Rust installed on that runner, and
`MXX_HIP_ARCH` for HIP (`MXX_CUDA_ARCH` defaults to `89`). Device unit tests are a separate
workflow input. No runner is provisioned automatically and no integration tests are added.
