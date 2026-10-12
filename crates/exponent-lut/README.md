# mxx-exponent-lut

`mxx-exponent-lut` is the shared Exponent-LUT library. Its programs keep private values in the
exponents of monomials and compute on them with private products and table lookups. It evaluates
these programs over RNS-BGG key-homomorphic encodings and simulates their noise. Protocols
built on it, such as [`mxx-pe`](../pe/README.md), depend on it as a library. It depends on
`mxx-dsl`, `mxx-ir-core`, and `mxx-backends`.

## Contents

| Item | What it provides |
| --- | --- |
| `program` | `ExponentLutProgram` and its builder: inputs, RHS inputs and families, additions, public and private products, lookups, parallel family bodies, and sums. |
| `api` | The stage lifecycle: preprocessing, encryption, online evaluation, and noise simulation, with their artifacts and executions. |
| `rns_bgg` | The RNS-BGG evaluator: parameters and level chains, public-key preprocessing, encryption, online evaluation, output verification, and the noise simulator. |
| `rns_bgg::setup`, `rns_bgg::dynamic` | The Dynamic setup schema and its shared packages, automorphism helpers, and public scalar actions. |
| `decoder` | Decoders of program outputs, including the smudged decoder. |
| `nand_circuit`, `program_gadgets` | NAND circuits compiled to programs, and gadgets such as the sparse LWR PRF and public inner products. |
| `lean/` | Lean proofs of the evaluator's correctness claims. |
| `tests/` | Integration tests: NAND circuits, the LWR PRF, decoders, and public inner products. |

## RHS modes

- **Static.** RHS ciphertexts and LUT switching material are sampled at preprocessing for one
  Program, and each level drops moduli as the program proceeds.
- **Dynamic.** A setup fixes input coordinates, RHS packages, and an automorphism group without
  naming a Program. An encryption publishes one package per RHS coordinate (member) and per
  automorphism, and any Program bound to the setup reads them at any level of an equal-basis
  chain. A lookup is `sum_k SMul_{D_k}(Aut_k(c))`, so table values enter only public actions,
  never encryption. Preprocessing publishes the public action of every root-scope product and
  lookup: its output key and its lookup branches' product keys. Online evaluation imports them
  and accumulates only the encoding rows. Products and lookups inside family bodies recompute the
  same action from the setup.
  A setup may decompose its cyclic group `H = <g>` of order `B^r` in radix `B`
  (`RnsBggDynamicSetup::with_lut_radix`): encryption then publishes switching packages for
  `K_B = {g^(a B^j) : 1 <= a < B, j < r}` only (`(B - 1) log_B |H|` instead of `|H| - 1`) and
  helpers for `J_B = K_B u K_B^-1`. A branch `k = g^e` is switched once per nonzero radix-`B`
  digit of `e`, from the branch without its top digit, so a lookup still performs `|H| - 1`
  switches, in a tree of depth at most `log_B |H|`; each switch adds one decomposition gain to the
  noise of the branches below it. `B = |H|` is the direct switching.
  A setup may also publish its fresh encodings compactly
  (`RnsBggDynamicSetup::with_compact_encodings`): each digit companion, automorphism helper, and
  private scalar input is a two-column row `s (Ã - p X I_2) + e` over `pq` for an extension `p` of
  the level's own size, and readers expand it to the full gadget width at `q` by
  `ModSwitch_{pq -> q}(c G̃)`. The expansion adds about `B q / p + (n + 3) / 2` to the error, and
  the simulator carries each expanded row as an opaque term. GSW ciphertexts stay at `q`.
  Every RHS gadget ciphertext `C = [a; c]` has a public top row `a` that the setup hashes
  (`rhs_public_top`); encryption samples only the bottom row `c` (`rhs_bottom`), in its own
  prelude from the bound plaintexts or in an enclosing scheme's prelude, and, as for an
  automorphism package's public top row, the digit rows of `a` need no companion, which halves
  the package's companions.
  A Dynamic Program may declare integer data families (`ExponentLutProgramBuilder::data_family`)
  whose values are supplied at run time (`ProgramData`) to preprocessing and online evaluation:
  a root-scope private product may read the RHS member at a data entry (`Operand::Selected`),
  and a lookup may take its table from data (`LutTable::from_data`). One graph then serves every
  value of the data; preprocessing binds the data's commitment, and online evaluation checks it.
  The noise simulator treats a selected member as member 0.

Sampling contracts (`RnsBggGaussian`) keep their sigma as a `BigDecimal` and their cutoff as a
`BigUint`, so a contract may be wide: above `2^40` the backends sample by convolution, as for the
noise-flooding contract of predicate encryption (sigmas up to `2^58`). The noise simulator reads
every contract exactly, without floating point.

Outputs are consumed by an explicit `RnsOutputUse`. `DirectDecode`, the default, applies a
coefficient output's decoding action. `Gadget` keeps the full-gadget carrier for protocols, such
as predicate encryption, that contract it further.
