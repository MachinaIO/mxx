# V2 toy proof foundation (wp1)

## Scope and trust boundary

This is the foundation for porting the proof from
`crates/io/lean/dio2/average-n8-p5x50/proof` to
`crates/io/lean/dio2/average-n8-p5x50-v2/proof`. It is **not** a proof of the v2
correctness claim. No v2 `Dio2Complete.dio2_correctness_claim` is supplied here.
Neither the production package nor its build artifacts are modified or built.

The following 17 modules are copied byte-for-byte from the old proof:
`Dio2Rel`, `Dio2Cell`, `Dio2Lut`, `Dio2Label`, `Dio2Sharp`, `Dio2Diag`,
`Dio2PartRows`, `Dio2Part`, `Dio2Final`, `Dio2Sampler`, `Dio2Inv`,
`Dio2Decode`, `Dio2Ideal`, `Dio2CellBridge`, `Dio2LutBridge`, `Dio2Keys`,
and `Dio2KeysLoops`. Decode/Ideal/Inv use generic or unchanged claim-level names,
not numbered stage lemmas. The bridge/key modules consume `Dio2Comp`, which
regenerates byte-identically to the old toy's bundle file.

`Dio2Generic` retains the old parameterized proofs, including arbitrary-width
PRF cells (`Comp_1W`, `rel_1W`, `keys_1W`), CRT recombination and packing. Its
stage imports, the concrete toy CRT certificate, and theorems identifying old
stage-specific index definitions are removed. `Dio2SmudgeCore` extracts the
parameter-independent polynomial/half-offset lemmas from old
`Dio2SmudgeValues` and `Dio2SmudgeHalf`, breaking the import cycle through stage
proofs. The old one-digit and five-prime specializations that have no stage
dependencies remain usable; they do not establish a general multi-digit theorem.

When porting `Dio2SmudgeValues`, import `Dio2SmudgeCore` instead of redeclaring
`smudge_one_digit_bound`, `smudge_one_digit_two_mul`, `smudge_one_digit_value`,
`centeredSmudge`, `centeredSmudge_bound`, `smudge_coefficientwise_offset`,
`smudge_constant_coeff`, `scalarHalfResidual`, `scalarHalfResidual_eq`,
`smudge_scalar_offset`, and `smudge_scalar_offset_bound`. Likewise,
`Dio2SmudgeHalf` should reuse `allCoefficients_reduce` from that module.
The removed `crt0_certificate`, `primes0_prod_list`,
`part_row_five_of_certificate`, `readoutMemberN_eight`, `placedReadoutN_eight`,
`cellValN_eight`, `partIndexGen_val_toy`, `smudgePartColumnGen_val_toy`, and
`freshPartColumnGen_val_toy` need new stage/certificate instantiations, not an
import from this generic module back into the stage layer.

`Dio2Decode` and `Dio2Ideal` are unchanged **toy claim-level** proofs. Their
successful reuse here must not be read as a parameter-generic decoder/ideal
theorem for production.

## Regeneration

Run from the repository root:

```bash
python3 crates/io/lean/dio2/generator/gen_adapters_dio2.py crates/io/lean/dio2/average-n8-p5x50-v2
python3 crates/io/lean/dio2/generator/gen_requirement_dio2.py crates/io/lean/dio2/average-n8-p5x50-v2
python3 crates/io/lean/dio2/generator/xroles_dio2.py crates/io/lean/dio2/average-n8-p5x50-v2/worst
(cd crates/io/lean/dio2/average-n8-p5x50-v2; ulimit -v 40000000; LEAN_NUM_THREADS=4 lake build Dio2Proof)
```

Both generators extract `dio2_meta.py` metadata directly from their package;
there is no toy-sized modulus, dimension, digit count, or family-cardinality
substitution in their new paths. Adapter module discovery uses metadata sources;
physical `_Part_N` files are traversed through their logical module imports,
not treated as separate stages. The digit-loop count must equal metadata's
`views.worst.smudge_digits` (one for this toy). Python integers and Lean `Nat`/
`Int` preserve exported u128 cardinalities without a machine-width conversion.
A larger-cardinality or production build is **not** part of this validation.

The overlay resolves `InputType`/`OutputType` aliases across imported partitions
and supplies their matrix/numeral notations to the shared parser. The shared
parser already reads hoisted `<scope>.w_*` definitions and Facts projections;
the overlay does not guess helper-lemma numbers or substitute identifiers by
textual prefixes. A scratch split-module regression moves a port alias and its
notations into a preceding partition and compares all 90 online-0 scope
signatures and atom sets with the monolithic module.

The requirement wrapper extends the shared refresh_top generator in memory
(with exact, single-match source-contract guards). It adds parallel integer
cells, deterministic `sha256BitColumns` projections, and their finite-function
equalities. It does not edit the shared generator. An upstream generator
layout change fails closed rather than silently skipping an extension.
The v2 requirement has 21 loop scopes and 13 root facts; the old one has 16
loop scopes and 18 root facts. Porters must use the new `Facts` fields and
conditions, not old `cond_N` numbers.

## Role inventory and adapters

All counts below include every parsed stage scope, including roots and retained
unused definitions. Both exports have **zero unclassified scopes**. Every
non-skipped role produces `<scope>.sem` in `Gen_W_Stage_*_Scopes`.

- **Bundle** means a parameterized `Dio2Proof.Comp.<Role>.Comp[_N]`, produced by
  the existing `xbundle`/`ad_bundle` path. Bundle variants are byte-identical
  between these two toy exports.
- **Exact relation** means `ad_expose`: the adapter exposes the exact exported
  existential/constraint relation, proved by `exact h`. It does **not** assert
  an encoding invariant, a noise bound, or equivalence with an old scope.
  New forest and block-diagonal/digit stage semantics remain a porter obligation.
- **Equation** means `ad_equation`: an output equation with lets inlined.
- Roots and definitions with no loop parent (`unused.body`, the retained copy
  definitions) are classified but intentionally have no adapter. Stage roots
  are the responsibility of stage-summary proofs.

| Role | Old count | V2 count | Adapter |
| --- | ---: | ---: | --- |
| `decode.root` | 1 | 1 | Skipped: decode stage root |
| `decode.round.step` | 1 | 1 | Exact relation |
| `enc.rooted-branch-key` | 8 | 8 | Bundle |
| `enc.rooted-label-entry` | 2 | 2 | Bundle |
| `gather.scalar` | 0 | 20 | `ad_scalar_gather`: singleton-tuple `GatherUnpack.Spec` |
| `gather.unpack` | 24 | 24 | `ad_gather_unpack`: `GatherUnpack.Spec` |
| `import.copy` | 110 | 0 | Equation |
| `import.label` | 10 | 8 | Equation |
| `instance.body` | 6 | 6 | Exact relation |
| `label.forest.assemble` | 0 | 2 | Exact relation |
| `label.forest.branch` | 0 | 9 | Exact relation |
| `label.forest.entry` | 0 | 13 | Exact relation |
| `label.forest.evaluate` | 0 | 4 | Exact relation |
| `label.forest.gather` | 0 | 2 | Exact relation |
| `label.forest.public` | 0 | 4 | Exact relation |
| `label.zero` | 0 | 22 | Exact relation |
| `lut.key-member.flat` | 5 | 5 | Bundle |
| `lut.key-round.step` | 14 | 14 | Bundle |
| `lut.key-table.flat` | 15 | 15 | Bundle |
| `lut.key-table.index` | 15 | 15 | Bundle |
| `lut.key-table.step` | 15 | 15 | Bundle |
| `lut.member.flat` | 10 | 10 | Bundle |
| `lut.round.step` | 28 | 28 | Bundle |
| `preimage.sample` | 5 | 5 | Exact relation |
| `prf.cell-flat` | 16 | 16 | Bundle |
| `prf.rowsum` | 24 | 24 | `ad_rowsum`: `RowSum.Spec` |
| `sampler.gather-placement` | 0 | 9 | Exact relation |
| `sampler.monomial` | 3 | 3 | Equation |
| `sampler.placement` | 9 | 0 | `ad_placement`: monomial placement specification |
| `smudge.digit.step` | 0 | 17 | Exact relation |
| `smudge.digit.weight` | 0 | 4 | Exact relation |
| `stage.root` | 18 | 18 | Skipped: stage root |
| `sum.columns.slice-step` | 15 | 15 | `ad_slice_step`: sliced column-sum specification |
| `sum.columns.step` | 6 | 6 | `ad_columns_step`: column-sum specification |
| `sum.members.step` | 42 | 42 | `ad_sum_members_step`: `MemberStep.Spec` |
| `unused.body` | 0 | 110 | Skipped: retained definition with no loop parent |

### New shapes and semantic obligations

- `label.forest.branch`/`entry`: symbolic initial/zero label forests, including
  interval sampling, gadget matrices and (for branches) automorphisms.
- `label.zero`: zero-valued labels in preprocess and preimage stages.
- `label.forest.public`: table/forest-selected hash publics, including root routes.
- `label.forest.evaluate`: selected forest-row lookup and gadget decomposition.
- `label.forest.assemble`: the enclosing forest evaluation with nested parallel
  loops and selected/static rows.
- `label.forest.gather`: selected output label rows; no new encoding is assumed.
- `sampler.gather-placement`: fused dynamic lookup and multiplication by a
  monomial. It must not be treated as a pure `GatherUnpack.Spec`.
- `gather.scalar`: one family-to-row lookup, wrapped as `(inputs, ())` and
  `(outputs, ())` when calling the tuple-based generic gather specification.
- `smudge.digit.weight`: indexed weight lifting/scaling for the digit loop.
- `smudge.digit.step`: one accumulator iteration containing `blockDiagonalRuns`.
  There is no claim here that a block diagonal is a concatenation of old rows.
- `decode.round.step`: preserves the polynomial decoder relation, rather than
  using refresh_top's scalar-only rounding adapter.

The old explicit label-tree sequences become forest entry/evaluation/assembly
scopes, so no one-to-one numerical mapping exists for those subgraphs. Likewise
the old final-instance conjunctions have new sequential digit scopes. Use the
metadata forest/root-route selections and prove the new forest/digit invariants
before connecting old part, noise and final-correctness arguments.

## Complete scope migration index

The tables group scopes by **structural role**, not by an unproved semantic
equivalence. For unchanged bundle-role lists of equal length, the order gives
the corresponding variant occurrence; the common `Dio2Comp` file is stronger
reuse evidence than scope numbers alone. A dash means that role is absent.
`same` means the entire scope-name list is unchanged. Roots are always named
`generatedRoot` and omitted from these tables; their roles appear above.

Abbreviations (all names are relative to the table's `W_Stage_*` namespace):
`pN = parallel_generatedRoot_N`, `sN = sequential_generatedRoot_N`,
`ppN_K = parallel_parallel_generatedRoot_N_K`,
`spN_K = sequential_parallel_generatedRoot_N_K`,
`ssN_K = sequential_sequential_generatedRoot_N_K`.
Names such as `scope_88` are literal hoisted scope names. Each table preserves
export order within a role. Import `Gen_W_Stage_<stage>_Scopes` before using
these `.sem` lemmas.

### `W_Stage_anchor_publics`

Only the stage root in both exports.

### `W_Stage_anchors`

Only the stage root in both exports.

### `W_Stage_decode`

| Role | Old scopes | V2 scopes |
| --- | --- | --- |
| `decode.round.step` | `s3` | `same` |

### `W_Stage_import_0`

| Role | Old scopes | V2 scopes |
| --- | --- | --- |
| `import.copy` | `p22, p25, p28, p31, p34, p37, p40, p43, p46, p47, p48, p49, p50, p51, p52, p53, p54, p55, p56, p57, p58, p59, p60, p61, p62, p63, p64, p65, p66, p67, p68, p69, p70, p71, p72, p73, p74, p75, p76, p77, p78, p79, p80, p81, p82` | `—` |
| `import.label` | `p6, p11, p14, p19` | `p6, p11, p14` |
| `unused.body` | `—` | `p17, p20, p23, p26, p29, p32, p35, p38, p41, p42, p43, p44, p45, p46, p47, p48, p49, p50, p51, p52, p53, p54, p55, p56, p57, p58, p59, p60, p61, p62, p63, p64, p65, p66, p67, p68, p69, p70, p71, p72, p73, p74, p75, p76, p77` |

### `W_Stage_import_1`

| Role | Old scopes | V2 scopes |
| --- | --- | --- |
| `import.copy` | `p11, p14, p15, p16, p17, p18, p19, p20, p21, p22` | `—` |
| `import.label` | `p6` | `same` |
| `unused.body` | `—` | `p11, p14, p15, p16, p17, p18, p19, p20, p21, p22` |

### `W_Stage_initial`

| Role | Old scopes | V2 scopes |
| --- | --- | --- |
| `label.forest.branch` | `—` | `p10, p12, p14, p16, p18, p20, p22, p24, p26` |
| `label.forest.entry` | `—` | `p11, p13, p15, p17, p19, p21, p23, p25, p27, p29, p30, p31, p32` |

### `W_Stage_keys_import_0_0`

| Role | Old scopes | V2 scopes |
| --- | --- | --- |
| `import.copy` | `p22, p25, p28, p31, p34, p37, p40, p43, p46, p47, p48, p49, p50, p51, p52, p53, p54, p55, p56, p57, p58, p59, p60, p61, p62, p63, p64, p65, p66, p67, p68, p69, p70, p71, p72, p73, p74, p75, p76, p77, p78, p79, p80, p81, p82` | `—` |
| `import.label` | `p6, p11, p14, p19` | `p6, p11, p14` |
| `unused.body` | `—` | `p17, p20, p23, p26, p29, p32, p35, p38, p41, p42, p43, p44, p45, p46, p47, p48, p49, p50, p51, p52, p53, p54, p55, p56, p57, p58, p59, p60, p61, p62, p63, p64, p65, p66, p67, p68, p69, p70, p71, p72, p73, p74, p75, p76, p77` |

### `W_Stage_keys_import_1_1`

| Role | Old scopes | V2 scopes |
| --- | --- | --- |
| `import.copy` | `p11, p14, p15, p16, p17, p18, p19, p20, p21, p22` | `—` |
| `import.label` | `p6` | `same` |
| `unused.body` | `—` | `p11, p14, p15, p16, p17, p18, p19, p20, p21, p22` |

### `W_Stage_keys_online_0_0`

| Role | Old scopes | V2 scopes |
| --- | --- | --- |
| `gather.unpack` | `pp28_6, pp32_6, pp1077_6, pp1081_6, pp1131_6, pp1135_6` | `pp28_6, pp32_6, pp890_6, pp894_6, pp944_6, pp948_6` |
| `instance.body` | `p1267, p1268, p1269, p1270, p1271` | `p984, p985, p986, p987, p988` |
| `label.forest.assemble` | `—` | `p658` |
| `label.forest.evaluate` | `—` | `pp658_32, pp658_33` |
| `label.forest.public` | `—` | `pp658_31, p870` |
| `lut.member.flat` | `p33, p34, p1082, p1136` | `p33, p34, p895, p949` |
| `lut.round.step` | `sp33_18, sp34_18, s315, s675, s731, s787, s843, s899, s955, s1011, sp1082_18, sp1136_18` | `sp33_18, sp34_18, s161, s240, s298, s356, s414, s472, s530, s588, sp895_18, sp949_18` |
| `prf.cell-flat` | `p27, p31, p1076, p1080, p1130, p1134` | `p27, p31, p889, p893, p943, p947` |
| `prf.rowsum` | `p28, p32, p1077, p1081, p1131, p1135` | `p28, p32, p890, p894, p944, p948` |
| `sampler.gather-placement` | `—` | `p896, p950` |
| `sampler.monomial` | `p35` | `same` |
| `sampler.placement` | `p1083, p1137` | `—` |
| `smudge.digit.step` | `—` | `sp984_48, sp984_92, sp985_48, sp985_92, sp986_48, sp986_92, sp987_48, sp987_92, sp988_48, sp988_92` |
| `smudge.digit.weight` | `—` | `p983` |
| `sum.columns.slice-step` | `sp1267_41, sp1267_93, sp1268_41, sp1268_93, sp1269_41, sp1269_93, sp1270_41, sp1270_93, sp1271_41, sp1271_93` | `sp984_41, sp984_89, sp985_41, sp985_89, sp986_41, sp986_89, sp987_41, sp987_89, sp988_41, sp988_89` |
| `sum.columns.step` | `s1114, s1168` | `s927, s981` |
| `sum.members.step` | `sp28_25, sp32_25, s54, sp1077_25, sp1081_25, s1102, ss1114_28, sp1131_25, sp1135_25, s1156, ss1168_28` | `sp28_25, sp32_25, s54, sp890_25, sp894_25, s915, scope_80, sp944_25, sp948_25, s969, scope_81` |

### `W_Stage_keys_online_1_1`

| Role | Old scopes | V2 scopes |
| --- | --- | --- |
| `gather.unpack` | `pp143_6, pp147_6` | `pp144_6, pp148_6` |
| `lut.member.flat` | `p148` | `p149` |
| `lut.round.step` | `s87, sp148_18` | `s87, sp149_18` |
| `prf.cell-flat` | `p142, p146` | `p143, p147` |
| `prf.rowsum` | `p143, p147` | `p144, p148` |
| `sampler.gather-placement` | `—` | `p150` |
| `sampler.placement` | `p149` | `—` |
| `smudge.digit.step` | `—` | `s172` |
| `smudge.digit.weight` | `—` | `p171` |
| `sum.members.step` | `sp143_25, sp147_25, s168` | `sp144_25, sp148_25, s169` |

### `W_Stage_mask`

Only the stage root in both exports.

### `W_Stage_online_0`

| Role | Old scopes | V2 scopes |
| --- | --- | --- |
| `gather.scalar` | `—` | `p1128, p1129, p1130, p1131, p1132, p1133, p1134, p1135, p1136, p1137, p1138, p1139, p1140, p1141, p1142, p1143, p1144, p1145, p1148, p1149` |
| `gather.unpack` | `pp30_6, pp34_6, pp1182_6, pp1186_6, pp1236_6, pp1240_6` | `pp30_6, pp34_6, pp892_6, pp896_6, pp946_6, pp950_6` |
| `instance.body` | `p1281` | `p991` |
| `label.forest.assemble` | `—` | `p660` |
| `label.forest.evaluate` | `—` | `pp660_32, pp660_33` |
| `label.forest.gather` | `—` | `p1146, p1147` |
| `label.forest.public` | `—` | `pp660_31, p872` |
| `lut.member.flat` | `p35, p36, p1187, p1241` | `p35, p36, p897, p951` |
| `lut.round.step` | `sp35_18, sp36_18, s317, s766, s824, s882, s940, s998, s1056, s1114, sp1187_18, sp1241_18` | `sp35_18, sp36_18, s160, s239, s297, s355, s413, s471, s529, s587, sp897_18, sp951_18` |
| `prf.cell-flat` | `p29, p33, p1181, p1185, p1235, p1239` | `p29, p33, p891, p895, p945, p949` |
| `prf.rowsum` | `p30, p34, p1182, p1186, p1236, p1240` | `p30, p34, p892, p896, p946, p950` |
| `sampler.gather-placement` | `—` | `p898, p952` |
| `sampler.monomial` | `p37` | `same` |
| `sampler.placement` | `p1188, p1242` | `—` |
| `smudge.digit.step` | `—` | `sp991_52, scope_81, scope_83, scope_85, scope_87` |
| `smudge.digit.weight` | `—` | `p985` |
| `sum.columns.slice-step` | `sp1281_45, sp1281_103, sp1281_159, sp1281_215, sp1281_271` | `sp991_45, sp991_99, scope_82, scope_84, scope_86` |
| `sum.columns.step` | `s1219, s1273` | `s929, s983` |
| `sum.members.step` | `sp30_25, sp34_25, s56, sp1182_25, sp1186_25, s1207, ss1219_28, sp1236_25, sp1240_25, s1261, ss1273_28` | `sp30_25, sp34_25, s56, sp892_25, sp896_25, s917, scope_88, sp946_25, sp950_25, s971, scope_89` |

### `W_Stage_online_1`

| Role | Old scopes | V2 scopes |
| --- | --- | --- |
| `gather.unpack` | `pp149_6, pp153_6` | `pp150_6, pp154_6` |
| `lut.member.flat` | `p154` | `p155` |
| `lut.round.step` | `s91, sp154_18` | `s91, sp155_18` |
| `prf.cell-flat` | `p148, p152` | `p149, p153` |
| `prf.rowsum` | `p149, p153` | `p150, p154` |
| `sampler.gather-placement` | `—` | `p156` |
| `sampler.placement` | `p155` | `—` |
| `smudge.digit.step` | `—` | `s178` |
| `smudge.digit.weight` | `—` | `p177` |
| `sum.members.step` | `sp149_25, sp153_25, s174` | `sp150_25, sp154_25, s175` |

### `W_Stage_payload`

Only the stage root in both exports.

### `W_Stage_preimages_0_0`

| Role | Old scopes | V2 scopes |
| --- | --- | --- |
| `preimage.sample` | `p3, p5, p7, p9, p11` | `same` |

### `W_Stage_preimages_1_1`

Only the stage root in both exports.

### `W_Stage_preprocess_0_0`

| Role | Old scopes | V2 scopes |
| --- | --- | --- |
| `enc.rooted-branch-key` | `p79, p83, p116, p120, p161, p165` | `same` |
| `enc.rooted-label-entry` | `p36` | `same` |
| `gather.unpack` | `pp80_4, pp84_4, pp117_4, pp121_4, pp162_4, pp166_4` | `same` |
| `lut.key-member.flat` | `p85, p122, p123, p167` | `same` |
| `lut.key-round.step` | `s69, sp85_10, sp122_10, sp123_10, s139, s141, s143, s145, s147, s149, s151, sp167_10` | `same` |
| `lut.key-table.flat` | `p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15` | `same` |
| `lut.key-table.index` | `pp3_12, pp4_10, pp5_12, pp6_12, pp7_10, pp8_10, pp9_10, pp10_10, pp11_10, pp12_10, pp13_10, pp14_10, pp15_10` | `same` |
| `lut.key-table.step` | `sp3_13, sp4_13, sp5_13, sp6_13, sp7_13, sp8_13, sp9_13, sp10_13, sp11_13, sp12_13, sp13_13, sp14_13, sp15_13` | `same` |
| `prf.rowsum` | `p80, p84, p117, p121, p162, p166` | `same` |
| `sampler.gather-placement` | `—` | `p86, p168` |
| `sampler.monomial` | `p124` | `same` |
| `sampler.placement` | `p86, p168` | `—` |
| `sum.columns.step` | `s107, s189` | `same` |
| `sum.members.step` | `sp80_17, sp84_17, s99, ss107_20, sp117_17, sp121_17, s137, sp162_17, sp166_17, s181, ss189_20` | `sp80_17, sp84_17, s99, scope_89, sp117_17, sp121_17, s137, sp162_17, sp166_17, s181, scope_90` |

### `W_Stage_preprocess_1_1`

| Role | Old scopes | V2 scopes |
| --- | --- | --- |
| `enc.rooted-branch-key` | `p75, p79` | `same` |
| `enc.rooted-label-entry` | `p33` | `same` |
| `gather.unpack` | `pp76_4, pp80_4` | `same` |
| `lut.key-member.flat` | `p81` | `same` |
| `lut.key-round.step` | `s66, sp81_10` | `same` |
| `lut.key-table.flat` | `p3, p4` | `same` |
| `lut.key-table.index` | `pp3_12, pp4_10` | `same` |
| `lut.key-table.step` | `sp3_13, sp4_13` | `same` |
| `prf.rowsum` | `p76, p80` | `same` |
| `sampler.gather-placement` | `—` | `p82` |
| `sampler.placement` | `p82` | `—` |
| `sum.members.step` | `sp76_17, sp80_17, s95` | `same` |

### `W_Stage_zero`

| Role | Old scopes | V2 scopes |
| --- | --- | --- |
| `label.zero` | `—` | `p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p20, p21, p22` |

## Validation and remaining work

Validated on 2026-10-16 (Asia/Tokyo). All 40 `Dio2Proof` roots build with:

```bash
(cd crates/io/lean/dio2/average-n8-p5x50-v2; ulimit -v 40000000; LEAN_NUM_THREADS=4 lake build Dio2Proof)
```

Timings are `/usr/bin/time` elapsed wall seconds, not clean-build benchmarks.
The successful build reuses dependencies and foundation artifacts built in the
earlier passes. Logs are under the user-requested external scratch directory:

```text
/tmp/claude-1000/-home-sora-codes-mxx-677e85886d845177-diamond-io2-implementation-b649d0/15c4a271-8105-4727-a176-882ab48de01b/scratchpad/sol/wp1/
```

| Check | Seconds | Result / evidence |
| --- | ---: | --- |
| Initial foundation build | 908.96 | Failed generated adapters; unchanged CellBridge, LutBridge, Keys and KeysLoops built. `build-first.log`. |
| Connected 40-root build | 57.01 | Failed scalar lookup bounds, requirement numeric/Fin inference, and missing all-coefficients import. `build-second.log`. |
| Final 40-root build | 289.46 | Exit 0, 8153 Lake jobs including cached dependencies. `build-third.log`. |
| Kernel axiom audit | 92.31 | Exit 0; all 499 theorem declarations across the 40 roots audited. `AxiomAudit.lean`, `axiom-audit.log`. |
| Old-toy adapter regeneration in scratch | 14.17 | Exit 0, all 19 stages, 383 adapters, no failed adapters. `old-adapters-final.log`. |
| Old-toy requirement regeneration in scratch | 2.85 | Exit 0; byte-identical to the checked-in old requirement. `old-requirement-final.log`. |
| V2 adapter regeneration in scratch | 9.94 | Exit 0, all 19 stages, 368 adapters, no failed adapters. `v2-adapters-regression.log`. |
| V2 requirement regeneration in scratch | 2.02 | Exit 0; all 21 regenerated adapter/bundle/requirement files match the successfully built sources byte-for-byte. `v2-requirement-regression.log`. |
| Requirement into a fresh output directory | 2.08 | Exit 0 and byte-identical output; the wrapper creates `proof/` itself. `fresh-requirement.log`. |

The audit reports only `propext`, `Classical.choice`, and `Quot.sound` (or no
axioms); no `sorryAx` or additional axiom is present. All 40 Lean sources also
pass the textual check for `sorry` and axiom declarations. Existing unused
variable/simp/tactic linter warnings remain; they are not proof holes.

The final role inventories cover 402 old scopes and 497 v2 scopes, with zero
unclassified scopes in either export. V2's 497 comprise 368 adapted scopes,
19 stage/decode roots, and 110 retained definitions without loop parents. The
inventory and complete migration index above include every one of those roles.

Old-package regression is generator validation, **not** a fresh old-package
Lean build: `Dio2Comp` and `Dio2Requirement` are byte-identical; eight of the ten
previously checked-in stage adapters are byte-identical, while online-0 and
keys-online-0 only gain exact-relation instance adapters (no existing theorem
changes). The old proof directory is not rewritten.

Additional checks pass: Python syntax compilation; `git diff --check`; exact
manifest coverage of all 40 Lean files; byte comparison of the 17 copied
modules; the 90-scope partition/alias parser regression; and an exact
`2^80` family-cardinality parser round trip. The last two are parser checks,
not a production or large-cardinality Lean build. An early standalone
requirement check lacked the not-yet-built `GenReq` dependency; the final Lake
build includes it. A fresh scratch requirement run exposed a missing output
directory assumption, now fixed and covered by the fresh-directory check.

The stage summaries, concrete CRT/certificate links, forest/digit semantic
lemmas, step transition, and final v2 correctness theorem remain outside wp1.
Do not infer production readiness from the toy foundation build.
