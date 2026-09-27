# mxx-gadgets

`mxx-gadgets` holds reusable, BGG-independent building blocks for lattice constructions:
circuit models, the framework that lowers a circuit to an encoding scheme, and gadgets written
as circuits or DSL graphs, such as nested-RNS arithmetic, NTT circuits, and Ring-GSW. Encoding
schemes such as `mxx-bgg` implement its lowering traits, and applications reuse its gadgets. It
depends on `mxx-dsl`, `mxx-ir-core`, and `mxx-backends`.

## Contents

| Module | What it provides |
| --- | --- |
| `circuit` | Polynomial and Boolean circuits, public lookup programs, serialization, and the lowering traits an encoding scheme implements. |
| `circuit_gadgets` | Modular and nested-RNS arithmetic, negacyclic convolution, NTT circuits, nested-RNS modulus switching, Ring-GSW, a Goldreich PRG, and secret inner products. |
| `decoder` | Masked-decoder and PRG layout helpers. |
| `noise_refresh` | Noise-refresh circuits (decrypt, merge, PRG) and their material. |
| `input_injector` | Input-injection preprocessing shared by Diamond constructions. |
| `ring_from_params` | A DSL ring with the same ordered basis as backend parameters. |
| `test_utils` (`test-support` feature) | Test helpers for dependent crates. |

## Design

- **Circuits are independent of encodings.** A circuit describes gates only. `lower_circuit`
  walks it gate by gate and calls the traits a concrete encoding scheme implements, giving each
  gate an instance identity (call path, local gate, and occurrence).
- **Gadgets stay above the IR.** Operations such as nested-RNS level switching are gadgets built
  from graph operations, not new graph nodes.
- **Shared code lives here.** Anything two constructions need moves down to this crate, so
  application crates never depend on one another.
