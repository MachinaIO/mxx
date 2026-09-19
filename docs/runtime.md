# Graph runtime

`mxx-runtime` executes a validated `mxx-ir-core` plan. It does not accept a
mutable builder or reinterpret symbolic annotations as executable operations.

The liveness plan releases intermediates after their last use unless trace mode
requests retention. Subgraph bodies share one validated plan while each call
has its own concrete instantiation path. Parallel loops execute in bounded
waves according to `max_parallel_instances`; each iteration receives its
concrete loop index and produces an ordered indexed-family result.

CPU-independent work uses Rayon where iterations are independent. GPU work may
intentionally use a smaller wave size to respect VRAM limits. DAG construction,
dependency-ordered traversal, and deterministic reductions remain ordered.

Artifacts are supplied and returned through the runtime artifact interfaces.
Final applications decide how artifact payloads are persisted.

## Caller and storage contracts

Execution accepts trusted, complete inputs prepared for the exact validated graph and parameter
bindings. Callers must supply every declared non-artifact input before starting execution, including
session execution. Matrix dimensions, ring dimension, ordered CRT basis/modulus, representation,
and bounded-matrix metadata must match the corresponding concrete wire type. For a host-staged
matrix, both its embedded metadata and its encoded payload must describe that same value; resident
CPU/GPU matrices have the same requirement.

A supplied trapdoor must have the validated matrix type, sigma, gadget base, digit count, small-gadget
mode, and preimage coefficient bound. Its public matrix and secret material must belong to the same
trapdoor construction. These requirements apply recursively to family elements. The runtime's
value-kind checks do not certify these caller obligations. Input producers enforce them when values
are constructed; execution does not copy resident matrices to the host or scan their contents to
revalidate them.

A session nonce binds an immutable, complete input map. Do not use session execution to probe input
validity. A failed execution can leave a durable session descriptor, and changing the inputs requires
a new nonce and, where applicable, a new stable alias. A retry with the original nonce is a resume of
the original inputs, not a correction of an incomplete invocation.

Artifact stores and transcript providers must return intact payloads produced by the matching backend
codec with the matching schema and parameters. Compact matrix decoding is not an untrusted-data
parser: malformed or truncated payloads violate its contract and can panic rather than return an
`ExecutionError`. Private-artifact manifests do not provide a content hash. Applications accepting
untrusted or potentially damaged storage must establish integrity before passing data to the runtime;
`Result` on the backend decoder is not a guarantee that all malformed byte strings are recoverable.
Serialized graphs must originate from the graph serializer and pass graph validation before execution;
structural consistency checks at deserialization do not authenticate their provenance.

## Hash-tag encoding

Hash samples preserve the insertion order of tag components. Byte strings and decimal integers use
length framing, and integer representations carry type markers, so decimal coordinates such as
`(1, 23)` and `(12, 3)` produce distinct tags. The raw prefix is a fixed namespace for a sampling
domain; callers use typed components for variable data.

This encoding changes hash-derived values from the former grouped tag format. Rebuild serialized
graphs and hash-derived preprocessing artifacts together; old cached values must not be reused with
the new tag encoding.
