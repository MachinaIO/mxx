# Principles for GPU Implementation

The builder agent must implement the GPU version in accordance with the principles below. Even if the submitted GPU implementation passes all tests, the reviewer agent must additionally review it against these principles to ensure it meets a sufficient quality bar.

1. Minimize memory transfers (and transfer frequency) between the device and the host.
2. Minimize synchronization. Avoid `cudaDeviceSynchronize` and `cudaStreamSynchronize` as much as possible. Use `cudaMallocAsync`, `cudaFreeAsync`, and `cudaMemcpyAsync` rather than `cudaMalloc`, `cudaFree`, and `cudaMemcpy`.
3. Make effective use of streams. Assign separate streams whenever computations can proceed independently, and avoid introducing unnecessary blocking among device threads.
4. For any wrapper function that does **not** involve transferring data from device to host, guarantee that the host-side execution is not blocked by the device (i.e., the host-side wrapper function must not wait for completion of the device work it launches within the function).
5. Minimize the number of kernel launches. Do not repeatedly launch the same kernel in a loop over different inputs; instead, design a single kernel launch so that different device threads handle different data.
6. Do not wastefully loop on the host merely to prepare arrays of pointers (or similar structures) to pass into kernels. Instead, have each device thread compute the address of the data it is responsible for inside the kernel.
7. In general, minimize the number of host-side loops and kernel launches. However, this principle does not apply if reducing them would require excessive redundant recomputation on the device.
8. In GPU testing, do not run each unit test only once. Run tests sufficiently many times and confirm that they pass every time. This is to avoid missing intermittent errors that occur nondeterministically due to synchronization issues.
9. Run every test that uses the GPU outside the sandbox. Do not execute GPU-using tests inside the sandboxed environment.
10. Keep the implementation as simple as possible, unless it violates the above principals.
11. Any GPU-specific implementation written in a language other than CUDA or another GPU-only language, and enabled only when the `gpu` feature is enabled, must be consolidated into files whose names include the word `gpu`. This applies to functions, modules, and tests.
