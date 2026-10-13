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
| `bgg` | BGG+ public keys and encodings, circuit compilation into public-key and encoding graphs, Boolean circuit evaluation, the closed protocol that evaluates a layered Boolean circuit on encodings and decrypts its output bit (`boolean_protocol`), LWE lookup tables, slot transfer, and Tall encodings. |
| `wee25` | WEE25 commitment trees, public-parameter preprocessing, and openings with their verification. |
| `circuit` | Polynomial and Boolean circuits, public lookup programs, serialization, the lowering traits an encoding scheme implements, and the lowering of Boolean-idiom polynomial circuits to layered Boolean circuits with its depth computed without expansion (`boolean_lowering`). |
| `circuit_gadgets` | Modular arithmetic (carry and Montgomery arithmetic over polynomial entries, and nested RNS), negacyclic convolution, NTT circuits, nested-RNS modulus switching, Ring-GSW, a Goldreich PRG, and secret inner products. With one-bit Montgomery limbs, the Ring-GSW product is a Boolean circuit without lookup tables. |
| `decoder` | Masked-decoder and PRG layout helpers. |
| `noise_refresh` | Noise-refresh circuits (decrypt, merge, PRG) and their material. |
| `noise_simulate` | BGG+ error bounds that follow the evaluators term by term, the error of a layered Boolean evaluation and its decryption condition, and the selection of the ring dimension, CRT depth, and gadget base that decrypt at a Ring-LWE security target. |
| `input_injector` | Input-injection preprocessing shared by Diamond constructions. |
| `ring_from_params` | A DSL ring with the same ordered basis as backend parameters. |
| `test_utils` (`test-support` feature) | Test helpers for dependent crates. |
| `lean/` | The `MxxKhe` Lean package: gadget-matrix, decomposition, and BGG+ encoding facts; `lean/ring_gsw_bgg/` proves the correctness claim of `boolean_protocol` for one Ring-GSW multiplication ([README](lean/ring_gsw_bgg/README.md)). |

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
- **Layered Boolean evaluation keeps graphs small.** `boolean_protocol` evaluates a circuit given
  as runtime gate records with one `iterate` over its layers and one `parallel` over a layer's
  slots, so its graphs and their Lean statements do not grow with the circuit. A circuit enters
  this form through `lower_to_boolean_layers`; `boolean_layering_bound` gives the depth and a
  width bound of a circuit too large to expand.

## Tests

The unit tests run on the CPU. `tests/test_gpu_ring_gsw_boolean_bgg.rs` lowers one Ring-GSW
multiplication, selects the BGG+ parameters of `boolean_protocol` for it, and exports the Lean claim
of a recorded parameter row. It also prints the GPU work predicted by `mxx_backends::gpu_cost_model`
for each of the evaluate graph (which ends with the decomposition `G^(-1)(floor(q / 2) e_2)`), the
decrypt graph (the rounding), and the circuit evaluation alone (`evaluate_layers_graph`), whose
counters it totals, with `gadget_decomposed_words` the RNS residues decomposed into gadget digits;
every one of the `depth * max_layer_width`
slots is evaluated, active or not, and `select` evaluates every candidate. Its rustdoc lists the
`MXX_KHE_RING_GSW_BGG_*` variables. It needs
`lattice-estimator-cli` on `PATH` and about 40 GiB of host memory:

```bash
cargo test -r -p mxx-khe --features gpu --test test_gpu_ring_gsw_boolean_bgg
```
