# Diamond witness encryption

`mxx-we` implements Diamond witness encryption as declarative DSL graphs. The
same validated graphs drive symbolic noise simulation, cost estimation, and
CPU or GPU runtime execution. There is no parallel handwritten implementation
of the protocol equations.

## Graph structure

Encryption builds the public BGG+ keys, evaluates the public circuit, samples
the input-injection and final projection preimages, and exports the public
preprocessing matrices through a runtime artifact manifest. Trapdoors are
intermediate values and are never exported as public artifacts.

The BGG-independent Diamond input-injection gadget in `mxx-gadgets` is shared
with Diamond iO. For each input level and automaton state, it samples a base and
trapdoor. The initial `p` vector has the symbolic relation

```text
p_epsilon = [s_epsilon, k] B_0 + e.
```

Each transition target is a selector times the next base plus a fresh error,
and the previous trapdoor samples its preimage. The gadget returns the final
trapdoor family; Diamond WE uses it to sample its BGG+ projection preimages
outside the common gadget. Independent digit and state families are represented
by DSL parallel loops. Runtime wave size is bounded by
`ExecutionConfig::max_parallel_instances`, so GPU execution does not turn
protocol-level parallelism into unbounded VRAM use.

Decryption imports the exact encryption production's artifacts, chooses one
transition matrix per witness digit, advances the encoded state, evaluates the
same public circuit over BGG+ encodings, applies the one, witness, `k`, and
decoder projections, and threshold-decodes the resulting polynomial. Artifact
identity and manifest validation bind decryption to the corresponding
encryption production. The BGG hash key is also the production nonce, matching
the protocol's deterministic sampling session; decryption verifies both that
nonce and the encryption graph hash before consuming artifacts.

## Circuit lowerings

The default entry points compile ordinary arithmetic circuits. The generic
`build_encryption_with_lowerings` and `build_decryption_with_lowerings` entry
points additionally accept independent public-lookup and slot-operation
lowerings. This keeps LWE-based public lookup and slot transfer in their owning
BGG layer while allowing Diamond graphs to use them. Unsupported historical
GGH15 and WEE25 lookup implementations are not restored.

## Noise simulation and parameter search

`simulate_diamond_noise` elaborates the encryption graph, exports its symbolic
artifact manifest, elaborates the linked decryption graph, and evaluates the
actual output decode bound with `mxx-noise-simulator`. It does not use a
Diamond-specific replacement for the shared polynomial-matrix norm rules.
As in the previous Diamond simulator, correctness analysis fixes the witness
bits to the all-one satisfying witness. The simulator's fixed-selection API
evaluates those exact digit and bit branches; unrelated uses of the simulator
still take the conservative maximum over unspecified branches. Full gadget
matrices are modulus-scale signal values, while gadget decompositions remain
bounded factors.

`DiamondParameterSearch` searches CRT depth and ring dimension. For every
candidate it constructs the actual DCRT modulus, checks the requested lattice
security, builds the real protocol graphs, and accepts the candidate only when
the final simulated decode noise is below the decoding threshold. The search
therefore covers circuit evaluation and final decryption rather than using an
isolated closed-form estimate.

## Tests

`crates/we/tests/test_gpu_diamond_we.rs` contains the explicit ignored GPU
integration test. Its defaults deliberately use security level 1 and witness
size 1, but still run real lattice-estimator-backed parameter search, nonzero
noise, GPU encryption, and GPU decryption. Run it with:

```sh
conda run -n sage cargo test --release -p mxx-we --features gpu \
  --test test_gpu_diamond_we -- --ignored --nocapture
```

The `MXX_DIAMOND_TEST_*` environment variables can increase the security,
witness size, ring-dimension range, or bounded parallelism for deliberate
larger validation runs.
