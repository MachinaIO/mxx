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
| `sampler` | Uniform, Gaussian (with a convolution sampler above `2^40`), hash-derived, trapdoor, and preimage samplers, and their widths and cutoffs. Trapdoors come in the computational instantiation `[ā \| I \| G - (ā R + E)]` and the statistical one `[ā \| g - ā T]`; a preimage is sampled at the width its trapdoor requests, at least the trapdoor's minimum. |
| `openfhe_guard` | CRT basis checks and OpenFHE table initialization. |
| `lattice_estimator` | Ring-LWE security estimates from the external `lattice-estimator-cli` (on `PATH` with its Sage environment). |
| `backend` | Runtime values and the CPU and GPU backends. |
| `executor` | The CPU executor. |
| `gpu_runtime` (`gpu` feature) | The GPU runtime: planning, execution, options, and limitations. |
| `gpu_cost_model` | A device-free prediction of the GPU work of a graph (NTTs, modular and wide multiplies, gadget digits), following the GPU lowering; available without the `gpu` feature. |
| `artifact`, `session`, `transcript`, `authority` | Artifact stores, durable sessions, sampling transcripts, and the prepare-then-run boundary. |
| `host_control` | Structural node dispatch shared by execution and measurement. |
| `env` | Environment variables read by native primitives and backends. |
| `cuda/` | Native CUDA sources; `poly::dcrt::gpu` describes them. |
| `native/` | C++ adapters to OpenFHE, bridged with `cxx`. |
| `benches/` | Matrix-product and preimage-sampling benchmarks for the CPU and the GPU. |

## Design

```text
ValidatedGraph --> executor::execute(.., &mut CpuDcrtBackend, ..)   CPU, node by node
               '-> GpuRuntime::plan(graph, inputs) -> GpuExecutionPlan
                   GpuRuntime::execute(&mut plan, inputs)             GPU, replaying CUDA Graphs
```

- **One graph, two executors.** The CPU executor walks each scope in order and batches the
  instances of a parallel loop into backend requests. The GPU runtime compiles the whole graph
  into CUDA Graph regions once and replays them. Both take the same validated graph.
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
- **Two trapdoor instantiations.** A trapdoor's type (`TrapdoorInstantiation` in `mxx-ir-core`)
  fixes its public width and secret layout. The statistical trapdoor keeps `T` in row chunks; its
  perturbation sampler never forms `T T*` or `T* T` but evaluates the conditional covariance with
  a Chebyshev expansion over the Fourier spectra of `T`, on the CPU and in CUDA. The samplers'
  floating-point error is heuristic, as in OpenFHE.
