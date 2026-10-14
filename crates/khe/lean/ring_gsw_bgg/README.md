# Correctness of one Ring-GSW multiplication on BGG+ encodings

These Lean packages prove the correctness claim of the closed protocol of
`mxx_khe::bgg::boolean_protocol` for the parameter rows of `parameters.csv`. The protocol encodes the
input bits of a layered Boolean circuit as BGG+ encodings `c_i = s (A_i - x_i G) + e_i` under
`s = (s_bar, 1)`, evaluates the layers, applies the decoding action `G^(-1)(floor(q / 2) e_2)` to the
output slot, and decrypts by rounding `2 (c' - s A') / q`. The circuit's gate records are
external inputs, so the claim covers every circuit of the row's shape; the row's shape is the one of
the Ring-GSW product `C1 G^(-1)(C2)` of `mxx_khe::circuit_gadgets::fhe::ring_gsw` with one-bit
Montgomery entries, lowered to layers by `mxx_khe::circuit::lower_to_boolean_layers`.

## What is proved

For the row `average-gsw2048x44-n524288-crt186x44`, the Ring-GSW ring has dimension 2048 and one
44-bit prime, the first prime of the BGG+ basis, which gives a circuit of 240 layers over
31,719,424 input bits; the BGG+ ring has dimension `2^19` and 186 primes of 44 bits, the gadget
base is 4, and encoding errors are discrete
Gaussians of width 3.19 truncated at 20. The generated statement `GeneratedClaim.CorrectnessClaim`
bounds by `2^-128` the measure of the sampling tapes on which some execution meeting the input
contracts and the circuit's validity predicate decrypts a bit other than the ideal circuit's output.
`RingGswBggProof.certificate` proves it with an empty failure set, using only the standard axioms:

- the samplers record their cutoffs, so every fresh error is at most the cutoff;
- the evaluation and the ideal circuit read the same gate records, and each slot of a layer encodes
  the ideal bit with the error bound `(2 m n (base / 2) + 4) E` of its preceding layer's bound `E`
  (`RingGswBggLayers.slot_layer`, the XOR row of `mxx_khe::noise_simulate::boolean`);
- the decoding action has constant digits, so the output error is at most `m (base / 2)` times the
  last layer's bound, and `4 m (base / 2) E + 4 < q` makes the rounding exact
  (`RingGswBggLayers.decode_output`, `decryption_bound`).

## Layout

| Path | Contents |
| --- | --- |
| `RingGswBggLayers.lean` | Parameter-independent lemmas: the encoding relation `Enc` through the six gate candidates, selection, the lock-step invariant of the evaluation and ideal loops, deterministic relations, and decoding. |
| `<id>/generated/` | The claim written by the integration test for row `<id>`; never edited by hand. |
| `<id>/RingGswBggStep.lean` | One layer of the evaluation against one layer of the ideal circuit. |
| `<id>/RingGswBggCertificate.lean` | The run of every stage and the certificate. |
| `generator/` | `gen.py` writes the proof files of a row from `templates/`, filling the claim's constants. |

The proofs reuse lemmas of `MxxKhe` (`../`), of `MxxExponentLut`
(`crates/exponent-lut/lean`: sampling, constant gadget digits, and threshold decoding), and of the
`mxx-ir-core` runtime.

## Building

Build `crates/ir-core/lean`, `crates/khe/lean`, and `crates/exponent-lut/lean` first (`lake build`
in each), then:

```bash
cd crates/khe/lean/ring_gsw_bgg/average-gsw2048x44-n524288-crt186x44 && lake build
```

The generated modules and the proof build in about a minute.

## A new parameter row

Add the row printed by `test_gpu_ring_gsw_boolean_bgg_parameters` to `parameters.csv`, export its
claim with `MXX_KHE_RING_GSW_BGG_PARAMETER_ID=<id>`, write its proof with
`python3 generator/gen.py <id>`, and build it.
