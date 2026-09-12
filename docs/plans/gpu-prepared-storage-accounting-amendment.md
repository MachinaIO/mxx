# Accepted prepared-storage accounting amendment

Status: accepted, with production physical-budget semantics amended by the
[2026-09-11 decision](gpu-driver-memory-contract-decision.md). The approved contract is integrated into the authoritative
[fleet implementation plan](fleet-wide-gpu-column-sharding.md), particularly its
complete-invocation admission, calibration and validation sections. Acceptance
records the design decision; it does not claim completed implementation or
validation. Current evidence remains in the
[implementation status](fleet-wide-gpu-implementation-status.md).

This amendment records why finite prepared backing requires a distinct logical
occupancy calibration metric and two physical setup acceptance checks.

## Concrete incompatibility

Prepared ordinary-matrix storage consumes actual existing CUDA allocations.
Reusing its slots does not increase the default pool's used-memory counter. Extending
this ownership model to all scratch, compact and persistent
buffers is a way to enforce a finite live set without guessing undocumented pool
rounding or driver-resource sizes. However, a correctly bounded allocating pilot
can then have a zero default-pool increment. The superseded contract rejected
that result. Calling these operations allocation-free views would misrepresent
their logical output and scratch requirements.

Requested allocation bytes do not bound opaque CUDA consumption. The later
approved policy excludes that consumption from the managed bound and permits
observed budget excess during computation. Complete managed activity coverage
still requires a source audit; a green suite cannot replace that audit.

## Accepted contract

Retain the default asynchronous CUDA pool. At an explicit initial context setup
boundary, provision finite backing and finite event, stream and pinned resources.
Observe physical residency `P_i = total_bytes - free_bytes` and retain the
existing allocator-adjusted live-residency definition
`E_i = P_i - pool_reserved_i + pool_used_i`. These quantities are different:
`P_i = E_i + (pool_reserved_i - pool_used_i)`. Unused pool reservation still
occupies physical memory and cannot be omitted from a physical-budget claim.

Accept setup only when both `E_i <= B_i` and the separate physical check
`P_i <= B_i` hold. The latter is an additional acceptance requirement; `E_i <= B_i`
alone does not establish it. Unused pool reservation may be reclaimed at this
explicit setup boundary before repeating the coherent observation. Failed or
oversized setup publishes no executable permit. No new private CUDA pool is
introduced. External allocations can cause actual CUDA failures; the user expects
exclusive computation on each selected device in normal production use.

Provisioning may temporarily exceed the budget. After accepted setup, enforce
managed reservations and native span/resource fit. Opaque CUDA consumption may
exceed `B_i` during production without stopping otherwise valid work. A seal
covers managed allocation/resource sites and execution identities; it is not a
prospective bound on all driver consumption. Validate first use of supported
worker/launch classes and reject unexpected managed requests before allocation.

Maintain the accepted initial `E_i` as a fixed baseline and charge each additional
managed allocation reservation exactly once until native release completes.
Prepared backing in the baseline is not charged again for each output or scratch
claim. Separately bounded remaining managed demand must fit the managed budget.
Physical and adjusted observations remain separate diagnostics and never change
these charges or stop valid work merely because the budget is exceeded.

Native logical reservations cover fixed preparation, complete retained outputs
and simultaneous scratch. A recycled prepared slice returns logical capacity
only after establishing the next user's producer/reader dependency; it never
retires its baseline backing. A separately charged managed allocation returns
capacity on native release completion even if CUDA retains pool pages. Pinned
host reuse must preserve CPU-write/DMA ordering.

For prepared allocating classes, record logical occupied-byte high-water as the
calibration observation, with a metric identity distinct from default-pool bytes.
Include actual output/scratch claims, count aliases and reused spans once, and
keep fixed preparation outside the pilot's incremental observation. Reservation
and actual claim counters are distinct. A zero unexplained logical increment is
still an error; these operations remain allocating classes.

The measured logical bytes-per-column value is a candidate hint. Its trial
width uses available internal capacity after retained reservations, rather than
headroom in separately managed allocation capacity (`B_i - R_i` in the revised plan): a fully
charged arena can have zero physical headroom while still containing reusable
slots. The admission proof is the native fit predicate over available spans,
alignment, layout, typed resource slots, all retained outputs and simultaneous
scratch. Check every active owner independently, preserve separate role capacities,
and reserve the complete
output demand before searching scratch widths. Physical counter checks provide
setup observations and diagnostics for unexpected growth. Later samples can
detect growth but cannot prove its absence, especially transient or outside-pool
growth. The managed-resource argument and exact native fit establish managed
admission; no prospective physical cap is claimed during computation. Neither counter infers free logical slots or
retires ledger charges.

Include the memory metric, prepared-storage configuration, native layout/bound
identity and active role assumptions in the existing profile key. Reports expose
logical workspace pressure and physical residency separately. Existing timings
remain device-event time and coordinated fleet wall time under their current
measurement contracts.

## Required implementation and validation

1. Extend native prepared storage with stable backing/span/resource identities,
   transferable reservation tokens, thread-bound activation, and explicit child
   permits for existing Rayon subjobs. Cover ordinary/compact owners, scratch,
   caches, related parameter tables and pinned buffers.
2. Complete native activity instrumentation, private-stream retirement coverage,
   and a source audit before enabling coherent epoch certification.
3. Extend the current ledger and class-width search to the native multidimensional
   fit predicate. Preserve atomic fleet publication and independent owner leases.
4. Compile typed invocations into one range runner used by both isolated pilots
   and production. Keep production sampler state and source loaders separate;
   load staged preimage targets directly on their planned worker device. Migrate
   immutable import boundaries so every allocation consumes an admitted plan.
5. Connect estimator measurements to those exact plans and distinguish logical
   workspace observations from physical setup residency.
6. Validate execution with a fully charged physical arena but free logical slots;
   logical exhaustion despite physical headroom; full-output scratch reduction;
   wrong layouts, stale tokens and cross-device rollback; cross-thread/reader
   reuse; zero default-pool growth with nonzero logical pilot pressure; sampler
   and transcript invariance; first use of every permitted worker and launch class;
   and all existing final multi-GPU, synchronization, unit-test and matched-timing
   gates.
