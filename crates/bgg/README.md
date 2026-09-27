# mxx-bgg

`mxx-bgg` implements BGG+ encodings with the mxx DSL: public keys and encodings, the evaluation
of polynomial and Boolean circuits over them, lookup tables, slot operations, Tall encodings,
and WEE25 commitments. It implements the circuit lowering traits of `mxx-gadgets`, and
constructions such as Diamond witness encryption build on it. It depends on `mxx-ir-core`,
`mxx-dsl`, `mxx-gadgets`, and `mxx-backends`.

## Contents

| Module | What it provides |
| --- | --- |
| `public_key`, `encoding` | BGG+ public keys and encodings: wires, schemas, and samplers. |
| `circuit` | Compiling a polynomial circuit into public-key or encoding graphs, in naive and Tall variants. |
| `boolean` | Evaluation of dynamic Boolean circuit families. |
| `lwe_lookup` | LWE-based public lookup tables with preprocessing artifacts. |
| `naive_vec`, `slot_operation` | Per-slot vectors, slot transfer, and rotation. |
| `tall_encoding`, `tall_rotation_encoding` | Tall encodings with one row per slot and their linear-transform preprocessing. |
| `wee25_commitment`, `wee25_opening`, `wee25_public_parameters` | WEE25 commitments, openings, and public parameters. |

## Design

- **Preprocessing and online evaluation are separate graphs.** Public-key compilation produces
  the gadget decompositions of every multiplication as preimages. Encoding compilation takes
  them from a provider called with each gate instance, so the online graph never builds
  public-key matrices or decompositions. The producer must bind each cached decomposition to the
  right gate.
- **Circuits come from `mxx-gadgets`.** This crate supplies only the BGG+ meaning of each gate
  through the lowering traits.
