# mxx-khe

`mxx-khe` implements key-homomorphic encodings with the mxx DSL. Its main scheme is BGG+:
public keys and encodings, the evaluation of polynomial and Boolean circuits over them, lookup
tables, slot operations, and Tall encodings. The crate also holds what these encodings evaluate
(scheme-independent circuit models, the framework that lowers a circuit to an encoding scheme,
and reusable gadgets such as nested-RNS arithmetic, NTT circuits, and Ring-GSW) and WEE25
commitments. It depends on `mxx-dsl`, `mxx-ir-core`, and `mxx-backends`.

## Contents

| Module | What it provides |
| --- | --- |
| `bgg` | BGG+ public keys and encodings, circuit compilation into public-key and encoding graphs, Boolean circuit evaluation, LWE lookup tables, slot transfer, and Tall encodings. |
| `wee25` | WEE25 commitment trees, public-parameter preprocessing, and openings with their verification. |
| `circuit` | Polynomial and Boolean circuits, public lookup programs, serialization, and the lowering traits an encoding scheme implements. |
| `circuit_gadgets` | Modular and nested-RNS arithmetic, negacyclic convolution, NTT circuits, nested-RNS modulus switching, Ring-GSW, a Goldreich PRG, and secret inner products. |
| `decoder` | Masked-decoder and PRG layout helpers. |
| `noise_refresh` | Noise-refresh circuits (decrypt, merge, PRG) and their material. |
| `input_injector` | Input-injection preprocessing shared by Diamond constructions. |
| `ring_from_params` | A DSL ring with the same ordered basis as backend parameters. |
| `test_utils` (`test-support` feature) | Test helpers for dependent crates. |
| `lean/` | The `MxxKhe` Lean package: gadget-matrix, decomposition, and BGG+ encoding facts. |

## Design

- **Circuits are independent of encodings.** A circuit describes gates only. `lower_circuit`
  walks it gate by gate and calls the traits an encoding scheme implements, giving each gate an
  instance identity (call path, local gate, and occurrence). `bgg` supplies the BGG+ meaning of
  each gate through those traits.
- **Preprocessing and online evaluation are separate graphs.** BGG+ public-key compilation
  produces the gadget decompositions of every multiplication as preimages. Encoding compilation
  takes them from a provider called with each gate instance, so the online graph never builds
  public-key matrices or decompositions. The producer must bind each cached decomposition to the
  right gate.
- **Gadgets stay above the IR.** Operations such as nested-RNS level switching are gadgets built
  from graph operations, not new graph nodes.
