# mxx-backends execution API

This document covers every API a user needs to execute a validated mxx program: choosing ring
parameters, creating a CPU or GPU backend, passing inputs, running, and reading outputs. It
assumes familiarity with lattice cryptography and with the DSL concepts in
`crates/dsl/SPEC.md` (programs, compile parameters, families, artifacts).

## 1. Concepts

**Validated program.** The `ValidatedGraph` returned by `BuiltGraph::validate(&bindings)`: a
program whose parameters are bound and whose types are checked. It is what both backends
execute.

**Backend parameters.** The concrete data a backend needs for each ring the program uses: the
ring dimension, the ordered CRT primes, and the gadget base. A backend is created with one
parameter set per ring, and looks each ring up by its ordered primes, so the primes must be
exactly those that validation resolved.

**Runtime value (`RuntimeValue`).** The type of every input and output: an integer, a Boolean, a
byte string, a matrix, a trapdoor, a family, a value on the GPU, or a reference to an artifact.
Inputs and outputs are passed by name, as a `BTreeMap<String, RuntimeValue>`.

**Resident value.** A value that lives in GPU memory. GPU outputs are resident, and they can be
passed to the next GPU program without copying them to the host.

**Plan (GPU only).** Before a program first runs on a GPU, the runtime builds a *plan*. The plan
fixes how much parallelism to use (how many loop iterations and matrix columns are processed at
once) and when GPU memory is allocated and freed, and compiles the program into GPU graph regions.
Planning tries several degrees of parallelism, discards those that do not fit in GPU memory, and
keeps the fastest. After that, the plan runs any number of times on new inputs without being
planned again.

**Artifact store and session.** An artifact store keeps exported artifacts (in memory or on
disk) so that later programs can import them. A session is a durable record of one producing run,
identified by the program and a 32-byte nonce, so that an interrupted run can resume with the
same inputs.

**Trusted inputs.** Execution assumes that inputs match the validated program exactly: shapes,
rings, bounds, and trapdoor data. It checks only what it needs to address them, not their
contents.

## 2. Ring parameters

| Call | Meaning |
| --- | --- |
| `mxx_ir_core::generate_crt_basis(N, crt_depth, crt_bits)` | The ordered primes that validation generates for `Ring::new(crt_bits, crt_depth, N)`. |
| `DCRTPolyParams::new(N, crt_depth, crt_bits, base_bits, moduli, dropped_moduli)` | CPU parameters. `base_bits` sets the gadget base `2^base_bits`; `moduli` is `None` to generate the basis or `Some(primes)` to use an explicit one; `dropped_moduli` is `None` for exact decomposition. |
| `GpuDCRTPolyParams::new(N, moduli, base_bits, dropped_moduli)` | GPU parameters, with explicit primes (for example from `generate_crt_basis`). |

For a program that uses several rings (for example a BGV tower or a key-switching basis), create
one parameter set per ring.

## 3. Running on the CPU

### 3.1 Backend

| Call | Meaning |
| --- | --- |
| `backend::poly::cpu_backend([params, ...])` | A `CpuDcrtBackend` for the given rings. |

### 3.2 Inputs

| Value | How to build it |
| --- | --- |
| integer | `RuntimeValue::Int(BigInt::from(x))` |
| Boolean | `RuntimeValue::Bool(b)` |
| real | `RuntimeValue::Real(x)` |
| bytes | `RuntimeValue::Bytes(Arc::from(bytes))` |
| matrix | `RuntimeValue::matrix(DCRTPolyMatrix)` |
| bounded matrix, preimage | `RuntimeValue::small_matrix(CpuSmallMatrix::new(matrix, bound)?)`, `RuntimeValue::preimage(...)` |
| trapdoor | `RuntimeValue::Trapdoor(TrapdoorValue::new(wire_type, public_matrix, Some(secret))?)` |
| integer family | `RuntimeValue::integer_values(vec![BigInt, ...])` |
| any family | `RuntimeValue::indexed_family(element_type, members)?` |
| composite (tuple, record) | its flattened leaves `name.0`, `name.1`, ... as separate entries |

### 3.3 Executing

| Call | Meaning |
| --- | --- |
| `execute(&validated, &mut backend, inputs, &mut store, SamplingMode::Fresh, ExecutionConfig::default())` | Runs the program once and returns an `ExecutionResult`. |
| `execute_in_session(&validated, &mut backend, inputs, &mut store, nonce, config)` | Runs as a durable session keyed by the program and `nonce`, for programs that export artifacts. Rerunning with the same nonce resumes the same inputs; changing the inputs needs a new nonce. |
| `execute_prepared(...)` | Same arguments as `execute_in_session`; uses a session only when the program exports artifacts. |
| `execute_with_trace(...)` | Same arguments as `execute`; also returns every intermediate value, for debugging. |

`SamplingMode` chooses where randomness comes from: `Fresh` draws new randomness,
`Record(&mut TranscriptRecorder)` draws new randomness and records it, and
`Replay(&TranscriptReplayer)` replays a recording (`recorder.into_replayer()`), so two runs see
the same samples.

`ExecutionConfig` has `max_parallel_instances` (default 64), the number of parallel-loop
iterations executed together, which bounds memory use; `preimage_progress`, optional progress
reporting for preimage sampling; and `release_fence_interval`.

### 3.4 Outputs

`ExecutionResult` has `outputs: BTreeMap<String, RuntimeValue>`, the `production_id` of a
session run, and `artifact_handles`. A matrix output is `RuntimeValue::Matrix(m)`; read it with
`m.as_cpu_full()`. A family output that was streamed to the store is a lazy reference until
`result.materialize_output(name, &backend, &mut store)` loads it; `result.cleanup_staged(&mut
store)` removes the staged members afterwards.

## 4. Running on GPUs (`gpu` feature)

The native backend is selected at build time by `MXX_GPU_BACKEND=cuda|hip`; CUDA is the
default. This leaves the `gpu` feature and Rust calls below unchanged. HIP requires ROCm/HIP
and an explicit `HIP_ARCH`; `HIPCC` and `ROCM_PATH` select its toolchain. CPU builds do not
require a GPU SDK. See [backend selection and validation evidence](README.md#gpu-backend-selection).
CUDA uses native conditional graphs. HIP schedules reusable graph regions after reading
GPU-produced control records at D2H boundaries, and measurements must include that cost.
Native subgraph callers use `crates/backends/gpu/include/SubgraphKernel.h`; backend-specific
plan identity must not be reused across CUDA/HIP builds, while canonical artifacts are portable.

### 4.1 Backend and runtime

| Call | Meaning |
| --- | --- |
| `backend::poly_gpu::gpu_backend([gpu_params, ...])` | A GPU backend on every detected GPU. |
| `backend::poly_gpu::gpu_backend_on([gpu_params, ...], device_ids)` | A GPU backend on the given devices. |
| `poly::dcrt::gpu::detected_gpu_device_ids()` | The available (logical) GPU ids. |
| `GpuRuntime::new(backend)?` | The runtime, with options read from the environment (section 4.6). |
| `runtime.options_mut()` | Changes options in code before planning. |

### 4.2 Inputs

GPU inputs use the same names and `RuntimeValue`s as on the CPU, with these differences:

- Integers (`Int`), integer families (`integer_values`), reals, and 32-byte byte strings can be
  passed from the host.
- A matrix must already be on the GPU:
  `RuntimeValue::gpu_matrix(ConcreteWireType::Matrix(ty), Arc::new(GpuDCRTPolyMatrix::from_cpu_matrix(&gpu_params, &cpu_matrix)))?`.
- A host `Bool` input is not supported yet; pass `0` or `1` as an `Int` instead.
- Any GPU output (`result[name]`) can be passed directly as an input of another plan, or of the
  same plan, and stays on the GPU.
- A host integer input gets a value range from `options_mut().integer_input_ranges` if one is
  declared, and otherwise from the values given at planning time (the full signed range of the
  fewest 64-bit words that hold them). Values outside that range fail at execution.

### 4.3 Planning and executing

| Call | Meaning |
| --- | --- |
| `runtime.plan(program, &inputs)?` | Builds a plan (section 1). `program` is a `ValidatedGraph`, or a `BuiltGraph` without parameters. `inputs` are example inputs of the right types and layouts; they are used only to build the plan. |
| `runtime.plan_with_store(validated, &inputs, &mut store)?` | As `plan`, for a program with artifact inputs; it reads their sizes (not their contents) from the store. |
| `runtime.execute(&mut plan, inputs)?` | Runs the plan with fresh randomness and returns a `GpuExecutionResult`. |
| `runtime.execute_with_artifacts(&mut plan, inputs, &mut store, nonce)?` | Runs a plan that imports or exports artifacts, as a session keyed by `nonce`. |

A plan runs only on the runtime that built it.

### 4.4 Outputs

| Call | Meaning |
| --- | --- |
| `result[name]` | The output as a `RuntimeValue`; it stays on the GPU. |
| `result.output(name)` | An optional `GpuOutputRef` for the typed downloads below. |
| `result.into_outputs()` | All outputs as a map. |
| `runtime.download_matrix(&value)`, `download_matrix_member(&value, i)` | A matrix, or member `i` of a matrix family, as a CPU `DCRTPolyMatrix`. |
| `runtime.download_integer_family(&value)` | An integer family as `Vec<BigInt>`. |
| `runtime.download_bool(&value)`, `download_real(&value)`, `download_bytes(&value)` | Scalars and bytes. |

Each download also has a `*_output` form that takes a `GpuOutputRef`. If the program is run again
while an output is still held, the new run writes into fresh memory, so held outputs stay valid.

### 4.5 Inspecting a plan

`plan.report()` gives the selected degrees of parallelism and their measured time;
`plan.compiled_region_count()` and `plan.compiled_launch_count()` give the size of the compiled
program.

### 4.6 Options

| Option (`runtime.options_mut()`) | Environment variable | Default | Meaning |
| --- | --- | --- | --- |
| `max_parallel_instances` | `MXX_GPU_MAX_PARALLEL_INSTANCES` | 64 | The most parallel-loop iterations one wave runs. |
| `measurement_warmups`, `measurement_iterations` | `MXX_GPU_MEASUREMENT_WARMUPS`, `MXX_GPU_MEASUREMENT_ITERATIONS` | 1, 2 | Trials per candidate during planning. |
| `io_trial_waves` | `MXX_GPU_IO_TRIAL_WAVES` | unset | Waves of each root wave group (and iterations of each root host-driven loop) that `plan_with_store` executes with artifact I/O to add an I/O-inclusive estimate to the report. A host or file store always runs this trial, with 2 waves when unset; a GPU-resident store runs it only when set. |
| `integer_input_ranges` | (code only) | empty | Declared value ranges of host integer inputs. |
| `subgraph_kernels` | (code only) | empty | Native kernels that replace named subgraphs on the GPU. |
| — | `MXX_GPU_MEMORY_FRACTION` | 0.8 | The fraction of each GPU's memory one plan may use. |
| — | `MXX_GPU_LOGICAL_DEVICES` | one per GPU | Maps logical devices to physical GPUs; `0,0` simulates two devices on GPU 0. |
| — | `MXX_GPU_SMALL_RHS_CHUNK_COLUMNS` | 1 | Columns of a bounded right operand one small-RHS product transforms at a time. Each extra column adds a workspace of the right operand's rows × one column, so a wider chunk means fewer launches and more memory. |

The small-RHS chunk width is chosen by hand for now: planning does not search it, so a setting
that does not fit in memory fails planning rather than falling back to a narrower chunk. TODO:
choose it automatically, as the wave width and tile width are.

### 4.7 Errors

Planning fails with `GpuPlanError`, for example when no degree of parallelism fits in memory.
Execution fails with `GpuRuntimeError`. A data-dependent failure detected on the GPU, such as
integer division by zero or exhausted preimage retries, is `DeviceStatus`; its outputs are
discarded and the plan can run again. `LaunchUncertain` means the GPU state is unknown and the
plan can no longer run. The current limitations of the GPU runtime are listed in the rustdoc of
`gpu_runtime`.

## 5. Artifact stores

| Call | Meaning |
| --- | --- |
| `MemoryArtifactStore::default()` | Artifacts kept in memory. |
| `FileArtifactStore::new(root)?` | Artifacts kept under a directory. |

Both implement `ArtifactStore` and `SessionStore`, and are passed to every `execute*` call. A
program that neither imports nor exports artifacts can use a fresh `MemoryArtifactStore`.

## 6. One interface for both backends

`ExecutionAuthority` is the prepare-then-run interface that applications use to stay independent
of the backend: `prepare(validated, &inputs)` followed by `run(&mut prepared, inputs, &mut store,
nonce)`. `CpuExecution::new(cpu_backend)` implements it with `execute_prepared`, and
`GpuRuntime` implements it with planning and execution.

## 7. Example

`crates/dsl/examples/rlwe_encrypt.rs` builds a Ring-LWE encryption and decryption program, binds
its parameters, creates a GPU backend from generated primes, plans once, and runs the plan on
several inputs.
