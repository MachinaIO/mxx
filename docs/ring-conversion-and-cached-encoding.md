# Ring conversion and cached encoding

## Arithmetic contract

`ModulusSwitch` scales a coefficient by the destination/source modulus ratio
and rounds to nearest; `ModulusReduce` changes its modulus without scaling.
The executable IR checks the supported odd-modulus divisibility relation and
uses the destination ring in the output type. GPU conversions operate on
coefficients on the device and return evaluation-format matrices.

Explicit CRT bases retain their order. Selecting a subset must preserve
`base_bits <= crt_bits / 2`; unsupported selected bases are rejected by both
CPU and GPU parameter selection. Approximate gadget decomposition retains its
existing default, per-tower digit alignment and residual bound. A per-operation
digit count can select a supported exact or approximate layout without changing
the arithmetic ring. Unsupported distributed approximate layouts remain rejected.

CPU native transforms use immutable tables identified by ring dimension, modulus
and root of unity. They call OpenFHE's stateless butterflies instead of its mutable
modulus-only transform cache, so rings sharing primes can execute concurrently.
Per-worker cached table storage is bounded; oversized tables are temporary.

## Cached encoding circuit inputs

`PolyCircuitCompiler::compile_encodings` and
`compile_encodings_with_lowerings` require a decomposition provider. The provider
receives a `GateInstance`, including the nested call path, local gate and operation
occurrence. It returns the preprocessing-produced typed `Preimage` for each
multiplication: the public RHS decomposition for ordinary multiplication, or the
scalar-times-gadget decomposition for large scalar multiplication.

The ordinary encoding compiler consumes these inputs directly. It does not
construct public-key matrices or emit gadget decomposition in the online graph.
Public-key preprocessing and artifact production remain explicit separate steps.
The producer must bind each cached decomposition to the correct gate and public
target; shape and coefficient bounds alone do not establish that binding.

## Runtime storage and execution

`FileArtifactStore` complements the memory store with persistent artifacts,
family members and resumable sessions. Existing schema and hash validation remain
part of loading and committing artifacts.

GPU preimage targets are staged from existing shards without first assembling a
full device matrix. The column source loads the requested output range. This bounds
extra GPU allocation and per-shard pinned scratch, but the host representation
still contains the full logical target. CPU sampling does not provide a streaming
or cross-backend deterministic-replay guarantee.

Fleet arithmetic retains direct/peer-only device placement: unavailable peer
transport is an explicit error. Artifact serialization and explicit host staging
are separate from that arithmetic placement contract.

## Estimator interpretation

`measured_wave_workspace_bytes` records scratch for a bounded measured wave,
excluding resident inputs. `workspace_bytes` describes hypothetical workspace
when all independent waves run concurrently. Neither alone is the complete
runtime peak. Work, dependency latency, cumulative wave time and wave count are
reported separately; loop-index-dependent costs cannot be marked invariant.

## Proof and format changes

The IR wire format changes with the new operations and expression forms, so
previous graph identities and generated certificates are not reusable unchanged.
Exact integer division, floor division and remainder remain distinct in structural
indices. Public Lean exports are regenerated and checked against the current IR;
modulus-conversion bounds do not imply a complete cryptographic security proof.
