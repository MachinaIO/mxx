# The Diamond iO 2 correctness claim in Lean

`crates/io/tests/dio2_claim.rs` records the stages of one obfuscation and one evaluation of a
two-step function (`mxx_io::dio2::claim`) and exports the claim `GeneratedClaim.CorrectnessClaim`
(`mxx_ir_core::lean::protocol::export`, `crates/ir-core/LEAN-SPEC.md`). The obfuscation is keyed by
a fixed hash key, so every export writes the same modules.

| Directory | Contents |
| --- | --- |
| `worst-n8-p5x50/generated` | The worst-case claim: every execution decodes the ideal state. |
| `average-n8-p5x50/generated` | The average-case claim: failure probability `2^-40` over the sampling tape. |
| `average-n8-p5x50/worst` | The worst-case modules renamed `W_*` (`gen_bridge.py`). |
| `average-n8-p5x50/bridge` | Every average-case run is a worst-case run (`gen_bridge.py`). |
| `average-n8-p5x50/proof` | The proof of the worst-case claim and the certificate. |
| `*-n8-p5x50-v2` | The toy claim in the compact export format (symbolic label forests, smudge-digit loop, partitioned modules), with its proof. |
| `*-n65536-p27x60` | The production claim (ring dimension `2^16`, 27 60-bit primes, gadget base `2^6`, failure `2^-100`). |

The parameters are the toy ring of the GPU round trip: ring dimension 8, five 50-bit primes, gadget
base `2^12`, six levels of `Q_0`, the evaluator's chunks `x = (0, 1)`. Their simulated decoder error
(`mxx_io::dio2::noise`) is below `2^159` in the worst case against the spacing `2^200`, so every
execution decodes correctly and the average-case claim follows from the worst-case one with failure
probability zero, as for refresh_top (`crates/exponent-lut/lean/refresh_top`).

## Regenerating

From the repository root:

```bash
MXX_IO_DIO2_CLAIM_FAILURE_EXPONENT=worst MXX_IO_DIO2_CLAIM_EXPORT=$PWD/crates/io/lean/dio2/worst-n8-p5x50/generated cargo test -p mxx-io --test dio2_claim
```

```bash
MXX_IO_DIO2_CLAIM_EXPORT=$PWD/crates/io/lean/dio2/average-n8-p5x50/generated cargo test -p mxx-io --test dio2_claim
```

```bash
python3 crates/exponent-lut/lean/refresh_top/generator/gen_bridge.py crates/io/lean/dio2/average-n8-p5x50 crates/io/lean/dio2/worst-n8-p5x50 Dio2
```

Then `lake build Dio2Bridge Dio2Proof` in `average-n8-p5x50`. The package reads the
parameter-independent lemmas of refresh_top (`Dio2Generic`, from
`crates/exponent-lut/lean/refresh_top/generic`) and the libraries of `crates/exponent-lut/lean`
and `crates/ir-core/lean`, whose `.lake` directories must hold built Mathlib and ir-core
artifacts.

## Proof structure

1. Node-class lemmas (LUT branches without a modulus drop, the sign, CBD and smudge samplers, the
   automorphism LUTs, public products, `Gamma`) and stage summaries over the exported scopes.
2. One symbolic refresh part, quantified over the part index, with one application per prime; the
   CBD row of a part is shared by its primes, so the residues recombine to one row.
3. The step input invariant (every root row an encoding under the step's mask with error at most
   one) from the initial input, through step 0's refresh, to step 1's output decoder; the keys of
   the run on zero rows equal the keys of the evaluation.
4. The decode stage (`proof/Dio2Decode.lean`) and the ideal functionality
   (`proof/Dio2Ideal.lean`), then the certificate from the bridge.

The final theorem is `Dio2Proof.Complete.dio2_correctness_claim : GeneratedClaim.CorrectnessClaim`
(`proof/Dio2Complete.lean`), with no hypothesis: `Dio2Correct` reduces the claim to the step
pieces (`CorrectInputs`), and `Dio2Complete` builds them from the stage proofs, the requirement
and the validity contract of the run.

The generated proof modules are rewritten by the scripts in `crates/io/lean/dio2/generator`
(`gen_adapters_dio2.py`, `gen_requirement_dio2.py`, `gen_part_primes.py`, `gen_key_agree0.py`,
`gen_key_agree_prime.py`); rerun them after regenerating the export.

## The v2 proof

`average-n8-p5x50-v2/proof` proves `Dio2Proof.Complete.dio2_correctness_claim` for the compact
export without hypotheses. Every theorem quantifies over members, refresh parts, smudge digits and
coefficients, and every package-dependent module is written by a generator in `generator`
(metadata from `dio2_meta.py`), so the same generators target the production package. The label
forest has one leaf slot per label row, so the forest product equals the clear label transition
without a bound on the hashed selector table.
