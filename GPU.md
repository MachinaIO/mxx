# Principles for GPU Implementation

Apply these requirements to GPU implementation and review. Passing tests does not establish the required ownership, asynchronous execution, or memory complexity; check the production path as well.

Native CUDA sources, GPU wrappers, and the concrete GPU runtime are owned by `mxx-backends` under `crates/backends/cuda/` and `crates/backends/src/`. Higher-level application graphs use its public APIs.

1. Minimize memory transfers (and transfer frequency) between the device and the host.
2. Minimize synchronization. Do not use `cudaDeviceSynchronize`. Use per-stream events and avoid `cudaStreamSynchronize` in asynchronous wrappers. Use `cudaMallocAsync`, `cudaFreeAsync`, and `cudaMemcpyAsync` rather than `cudaMalloc`, `cudaFree`, and `cudaMemcpy`.
3. Make effective use of streams. Assign separate streams whenever computations can proceed independently, and avoid introducing unnecessary blocking among device threads.
4. For any wrapper function that does **not** involve transferring data from device to host, guarantee that the host-side execution is not blocked by the device (i.e., the host-side wrapper function must not wait for completion of the device work it launches within the function).
5. Minimize the number of kernel launches. Do not repeatedly launch the same kernel in a loop over different inputs; instead, design a single kernel launch so that different device threads handle different data.
6. Do not wastefully loop on the host merely to prepare arrays of pointers (or similar structures) to pass into kernels. Instead, have each device thread compute the address of the data it is responsible for inside the kernel.
7. In general, minimize the number of host-side loops and kernel launches. However, this principle does not apply if reducing them would require excessive redundant recomputation on the device.
8. Keep the implementation as simple as possible, unless it violates the above principles.
9. Any GPU-specific implementation written in a language other than CUDA or another GPU-only language, and enabled only when the `gpu` feature is enabled, must be consolidated into files whose names include the word `gpu`. This applies to functions, modules, and tests.

## Dataflow and Memory

- Never fall back to CPU to work around a failing GPU path; a GPU failure is an error.
- Batch allocations and kernel launches, including across limbs. Avoid calls with implicit synchronization, such as `to_compact_bytes`, in hot paths.
- Multi-GPU: enumerate devices via `detected_gpu_device_ids`, not a fixed `gpu_id` in parameters; these are logical device ids (`docs/architecture.md`, section 6.4). Distribute work evenly, keep all limbs of a matrix on one device, and load shared data onto each device once before loops. Move data between devices only with copy nodes, and select devices in native code only through `mxx_set_device`.
- Matrices stay in evaluation format by default. Align NTT formats before comparing or concatenating them.
- Peak VRAM/RAM must scale with configured parallelism, not `num_slots` or total gate count. Matrices of order `d x m_b` or `d x m_g` are acceptable; `m_b^2`, `m_g^2`, and `m_b x m_g` are not. Chunk, stream, and store to disk; release large data promptly and pipeline load, compute, and store.
- CUDA headers (`.cuh`) declare only cross-file and Rust-facing functions; put bodies in `crates/backends/cuda/src/*.cu`.

## Runtime Validation

Run GPU-using tests outside the sandbox using the approved execution mechanism. Do not bypass a rejection; complete available checks and report the runtime gate as blocked. Compilation alone is not GPU validation.

To exercise multi-device code on one GPU, set `MXX_GPU_LOGICAL_DEVICES=0,0` (or `0,0,0`), which maps two (or three) logical devices onto physical GPU 0. For multi-device changes, run the affected GPU tests in identity mode (variable unset) and in `0,0` and `0,0,0` modes; this does not validate multiple physical GPUs or their memory capacity.

For CUDA/synchronization changes, compile once with `cargo test -r --workspace --lib --features gpu --no-run`, then run the relevant built test binary with the same exact test filter, parameters, and command for the full repetition set: 300 runs for synchronization bugs, or 3–5 for round-trip smoke checks. Continue through test failures to collect the failure count, unless the user cancels or a resource/permission limit prevents execution. For other GPU changes, choose enough repetitions to assess the relevant intermittent risk. Report the command, completed repetition count, and failures. Rebuild and rerun affected checks when code changes; do not repeat a completed successful set without a new reason.
