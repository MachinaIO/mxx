# Accepted production VRAM budget policy

Status: accepted on 2026-09-11. This decision amends the
[fleet implementation plan](fleet-wide-gpu-column-sharding.md) and the
[prepared-storage accounting amendment](gpu-prepared-storage-accounting-amendment.md).
It records the approved behavior, not completed production implementation.

## Approved behavior

The configured VRAM budget controls initial setup and managed capacity planning.
During computation, actual VRAM consumption may exceed that budget. An otherwise
valid admitted computation continues while its resources remain available and
CUDA operations succeed. Observed excess does not stop new submissions, cancel
running work or trigger a retry. The user expects this
program normally to be the only program computing on each selected device.

For example, with a 12 GiB budget, an observed increase to 13 GiB does not itself
stop processing. Already prepared slots can still be used, and new managed
requests can proceed when their reservations fit. An actual CUDA allocation
failure or device failure remains an error. The implementation does not promise
to complete a workload that exceeds available device memory.

## Accounting and enforcement

1. Keep the initial acceptance checks `P_i <= B_i` and `E_i <= B_i`, where
   `P_i = total_i - free_i` and
   `E_i = P_i - pool_reserved_i + pool_used_i`. Initial provisioning may
   temporarily exceed `B_i`, as already approved.
2. Keep the accepted initial `E_i` as the ledger baseline. Add each live managed
   allocation reservation exactly once. Prepared backing already in that baseline
   is not charged again when a slot is used. Complete native source coverage and
   exact span/resource admission remain required before production activation.
3. Opaque CUDA-internal growth is outside the managed bound. Neither loading
   kernels nor a stable memory sample proves an upper bound on future driver
   consumption. A receipt must describe the managed activity it actually covers.
4. Later physical and allocator-adjusted observations are diagnostic only.
   They never increase or decrease managed charges and never stop valid work
   merely because `P_i` or `E_i` exceeds `B_i`. Do not add device-wide waits or
   per-wave pool queries for this policy.
5. Managed allocation capacity returns only after native release completion.
   CUDA retaining freed pool pages does not prevent reuse of that capacity.
   Prepared slot handoff still requires the existing producer/reader dependencies.
6. Report accepted setup residency, current managed reservations, native logical
   capacity, and observed physical/adjusted usage separately. Report observed
   budget excess as a diagnostic value, not an admission error.

This explicitly replaces the earlier proposal to stop further submissions after
observing physical-budget excess. The original setup checks and all resource
lifetime, layout, identity and managed-capacity checks remain applicable.

The existing default-pool diagnostic scheduler also treats the budget as a
planning target: it continues to propose at least the class minimum when target
headroom is exhausted. This candidate is not a managed reservation or a promise
of allocation success. The prepared planner still checks exact managed capacity. Estimator transfer
fixture allowances similarly remain diagnostics: exceeding the target does not
reject a full-shape fixture before its actual allocation is attempted.

## Remaining implementation and validation

Update ledger semantics, native receipts and reports consistently. Complete the
managed allocation/resource audit before enabling production setup; removing the
opaque-driver bound does not establish that audit. Finish default provisioning,
remaining invocation/import paths, fixed-operand redistribution and estimator
consumption. Verify continued admission after observed excess, safe capacity
reuse despite retained pool pages, and rejection of invalid claims and actual
native failures. Preserve the outstanding multi-GPU, timing and independent
review gates documented in the [implementation status](fleet-wide-gpu-implementation-status.md).
