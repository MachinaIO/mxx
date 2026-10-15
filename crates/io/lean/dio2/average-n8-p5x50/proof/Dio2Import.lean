import Dio2InitialTargets
import Dio2Initial
import Dio2Online0Label
import Dio2Online0State
import Gen_W_Stage_import_0_Scopes
import Gen_W_Stage_import_1_Scopes
import W_Stage_import_0_Facts
import W_Stage_import_1_Facts
import W_Claim

/-!
# Initial root rows through the two import stages

The import stages do not perform a modulus switch: `import.copy` preserves both components,
and `import.label` preserves both components and publishes `c X^e`. Consequently every imported
root relation retains its supplied error bound; initial root relations have bound `1`.
The statements quantify over family members; neither label families nor LUT-helper families
are expanded in their interfaces.

`Import0Port` and `Import1Port` name precisely the imported row pairs read by the online
theorems. The tables record `online pair <- import pair <- source pair`; step zero's source
is the initial graph and step one's source is the preceding online output. A suffix `[i]`
means the same member of the family, and singleton rows use member `0`.

## Step 0
* `labelState`: `(w_257_0[i], w_262_0[i])` <-
  `(w_6_0[i], w_6_1[i])` <- `(w_269_0[i], w_294_0[i])`.
* `labelSigma`: `(w_16_0[i], w_17_0[i])` <-
  `(w_11_0[i], w_11_1[i])` <- `(w_269_0[i], w_294_0[i])`.
* `labelFresh`: `(w_1169_0[i], w_1170_0[i])` <-
  `(w_14_0[i], w_14_1[i])` <- `(w_269_0[i], w_294_0[i])`.
* `labelInput`: `(w_372_0[i], w_377_0[i])` <-
  `(w_19_0[i], w_19_1[i])` <- `(w_367_0[i], w_416_0[i])`.
* `privateSigma`: `(w_21_0, w_25_0)` <-
  `(w_0_0, w_1_0)` <- `(w_431_0, w_444_0)`.
* `privateFresh`: `(w_1174_0, w_1178_0)` <-
  `(w_0_0, w_1_0)` <- `(w_431_0, w_444_0)`.
* `privateRoot`: `(w_1229_0, w_1232_0)` <-
  `(w_0_0, w_1_0)` <- `(w_431_0, w_444_0)`.
* `state`: `(w_275_0, w_290_0)` <-
  `(w_7_0, w_8_0)` <- `(w_462_0, w_474_0)`.
* `aut3`: `(w_764_0[i], w_765_0[i])` <-
  `(w_68_0[i], w_68_1[i])` <- `(w_108_0[i], w_121_0[i])`.
* `aut5`: `(w_822_0[i], w_823_0[i])` <-
  `(w_69_0[i], w_69_1[i])` <- `(w_136_0[i], w_149_0[i])`.
* `aut7`: `(w_880_0[i], w_881_0[i])` <-
  `(w_70_0[i], w_70_1[i])` <- `(w_164_0[i], w_177_0[i])`.
* `aut9`: `(w_938_0[i], w_939_0[i])` <-
  `(w_71_0[i], w_71_1[i])` <- `(w_192_0[i], w_205_0[i])`.
* `aut11`: `(w_996_0[i], w_997_0[i])` <-
  `(w_65_0[i], w_65_1[i])` <- `(w_24_0[i], w_37_0[i])`.
* `aut13`: `(w_1054_0[i], w_1055_0[i])` <-
  `(w_66_0[i], w_66_1[i])` <- `(w_52_0[i], w_65_0[i])`.
* `aut15`: `(w_1112_0[i], w_1113_0[i])` <-
  `(w_67_0[i], w_67_1[i])` <- `(w_80_0[i], w_93_0[i])`.
* `helperSigma`: `(w_6_0[i], w_8_0[i])` <-
  `(w_72_0[i], w_72_1[i])` <- `(w_220_0[i], w_233_0[i])`.
* `helperLabel`: `(w_13_0[i], w_14_0[i])` <-
  `(w_63_0[i], w_63_1[i])` <- `(w_220_0[i], w_233_0[i])`.
* `helperState`: `(w_255_0[i], w_256_0[i])` <-
  `(w_55_0[i], w_55_1[i])` <- `(w_234_0[i], w_235_0[i])`.

## Step 1

`import_1` reads the newly refreshed root families exported by `online_0`, not the original
initial rows. Use `import1_online_from_rows` or `import1_claim_online_rel` with their established
root relations; the initial-specialization theorem applies only when its initial-wiring
premise is available. The source wires in this table belong to `online_0`.
* `labelState`: `(w_16_0[i], w_27_0[i])` <-
  `(w_6_0[i], w_6_1[i])` <- `(w_1532_0[i], w_1535_0[i])`.
* `privateRoot`: `(w_142_0, w_145_0)` <-
  `(w_0_0, w_1_0)` <- `(w_1546_0, w_1547_0)`.
* `state`: `(w_46_0, w_63_0)` <-
  `(w_7_0, w_8_0)` <- `(w_1548_0, w_1549_0)`.
* `helperState`: `(w_13_0[i], w_15_0[i])` <-
  `(w_16_0[i], w_16_1[i])` <- `(w_1524_0[i], w_1529_0[i])`.
* `helperSmudge`: `(w_136_0[i], w_137_0[i])` <-
  `(w_17_0[i], w_17_1[i])` <- `(w_1510_0[i], w_1519_0[i])`.

The payload mappings are `online_0 w_276/w_22/w_1175/w_391 <- import_0
w_6_2/w_11_2/w_14_2/w_19_2`, and `online_1 w_47 <- import_1 w_6_2`.
Their coefficient/exponent inputs remain explicit: an import graph alone does not assert
that externally supplied plaintexts agree with the initial labels.

The two wiring hypotheses of `import0_online_rel` and `import1_online_rel` are equalities,
not encoding assumptions. They expose the named connections of `worst/W_Claim.lean` without
requiring its entire execution tuple. `initial_import_rel` supplies all root relations directly
from the initial graph, including the packaging of helper and label families.
Internal refreshed encodings (`henc`), switching rows, RHS relations, and the agreement of
preprocessed keys with root keys are not import-stage obligations.
-/

set_option maxHeartbeats 8000000
set_option maxRecDepth 16384

namespace Dio2Proof

open Mxx.Primitives MxxRuntime MxxExponentLut RefreshTopProof.Gen

local notation "Q" => W_Stage_initial.modulus_0
local notation "Row" => ExactMatrix Q 8 1 25

/-- The root-row families of the initial step input that are consumed by an import stage. -/
inductive ImportRoot where
  | label0 | label | one | state | aut3 | aut5 | aut7 | aut9 | aut11 | aut13 | aut15 | exp8 | exp4
  deriving DecidableEq

/-- The number of rows in an imported root family. -/
def ImportRoot.members : ImportRoot → Nat
  | .label0 => 2
  | .label => 4
  | .one => 1
  | .state => 1
  | .aut3 => 1
  | .aut5 => 1
  | .aut7 => 1
  | .aut9 => 1
  | .aut11 => 1
  | .aut13 => 1
  | .aut15 => 1
  | .exp8 => 8
  | .exp4 => 4

/-- The initial graph's exported row pair, before any import scope is applied. -/
noncomputable def initialImportRows {backend : BackendContext} {model : HashModel}
    {params : W_Stage_initial.Params} (ctx : W_Stage_initial.generatedRoot.Ctx backend model params)
      :
    (root : ImportRoot) → Fin root.members → Fin 2 → Row
  | .label0, member => ![W_Stage_initial.generatedRoot.w_269_0 ctx member,
      W_Stage_initial.generatedRoot.w_294_0 ctx member]
  | .label, member => ![W_Stage_initial.generatedRoot.w_367_0 ctx member,
      W_Stage_initial.generatedRoot.w_416_0 ctx member]
  | .one, _ => ![W_Stage_initial.generatedRoot.w_431_0 ctx,
      W_Stage_initial.generatedRoot.w_444_0 ctx]
  | .state, _ => ![W_Stage_initial.generatedRoot.w_462_0 ctx,
      W_Stage_initial.generatedRoot.w_474_0 ctx]
  | .aut3, member => ![W_Stage_initial.generatedRoot.w_108_0 ctx member,
      W_Stage_initial.generatedRoot.w_121_0 ctx member]
  | .aut5, member => ![W_Stage_initial.generatedRoot.w_136_0 ctx member,
      W_Stage_initial.generatedRoot.w_149_0 ctx member]
  | .aut7, member => ![W_Stage_initial.generatedRoot.w_164_0 ctx member,
      W_Stage_initial.generatedRoot.w_177_0 ctx member]
  | .aut9, member => ![W_Stage_initial.generatedRoot.w_192_0 ctx member,
      W_Stage_initial.generatedRoot.w_205_0 ctx member]
  | .aut11, member => ![W_Stage_initial.generatedRoot.w_24_0 ctx member,
      W_Stage_initial.generatedRoot.w_37_0 ctx member]
  | .aut13, member => ![W_Stage_initial.generatedRoot.w_52_0 ctx member,
      W_Stage_initial.generatedRoot.w_65_0 ctx member]
  | .aut15, member => ![W_Stage_initial.generatedRoot.w_80_0 ctx member,
      W_Stage_initial.generatedRoot.w_93_0 ctx member]
  | .exp8, member => ![W_Stage_initial.generatedRoot.w_220_0 ctx member,
      W_Stage_initial.generatedRoot.w_233_0 ctx member]
  | .exp4, member => ![W_Stage_initial.generatedRoot.w_234_0 ctx member,
      W_Stage_initial.generatedRoot.w_235_0 ctx member]

/-- The automorphism named by a helper member; non-helper roots use the identity here. -/
def ImportRoot.sigma : (root : ImportRoot) → Fin root.members → Nat
  | .label0, _ => 1
  | .label, _ => 1
  | .one, _ => 1
  | .state, _ => 1
  | .aut3, _ => 3
  | .aut5, _ => 5
  | .aut7, _ => 7
  | .aut9, _ => 9
  | .aut11, _ => 11
  | .aut13, _ => 13
  | .aut15, _ => 15
  | .exp8, member => 2 * member.val + 1
  | .exp4, member => ![1, 3, 5, 15] member

/-- Every helper automorphism is odd, including every member of the two layout families. -/
theorem ImportRoot.sigma_odd (root : ImportRoot) (member : Fin root.members) :
    Odd (root.sigma member) := by
  cases root <;> simp only [ImportRoot.sigma]
  all_goals first | decide | exact ⟨member.val, by omega⟩ | fin_cases member <;> decide

/-- The root tag prefix, identical to the prefix used by the corresponding online key. -/
def ImportRoot.prefix : ImportRoot → Blob
  | .label0 => labelPre
  | .label => labelInPre
  | .one => privatePrefix
  | .state => statePre
  | _ => helperPre

/-- The member or automorphism tag components of an imported root row. -/
def ImportRoot.tags : (root : ImportRoot) → Fin root.members → List HashTagComponent
  | .label0, member => [.bytes memberTagG, .u64Le member.val]
  | .label, member => [.bytes memberTagG, .u64Le member.val]
  | .one, _ => []
  | .state, _ => []
  | root, member => [.integer (root.sigma member)]

/-- The initial value encoded by a root member; helpers and the private unit encode one. -/
noncomputable def initialImportValue {backend : BackendContext} {model : HashModel}
    {params : W_Stage_initial.Params} (ctx : W_Stage_initial.generatedRoot.Ctx backend model params)
      :
    (root : ImportRoot) → Fin root.members → ErrorPoly 8
  | .label0, member => if member.val = 0 then message 1 (ctx.inputs.2.2.2.1 0) else 1
  | .label, member => message 1 (ctx.inputs.2.2.2.1 ⟨member.val + 1, by
      have hmember : member.val < 4 := member.isLt
      omega⟩)
  | .state, _ => message 1 (ctx.inputs.2.2.2.2.2.1 * 2)
  | _, _ => 1

/-- The root mask in each target: the initial mask, private payload, or substituted mask. -/
noncomputable def initialImportMask (sm tp : ErrorPoly 8) :
    (root : ImportRoot) → Fin root.members → ErrorPoly 8
  | .label0, _ => sm
  | .label, _ => sm
  | .one, _ => tp
  | .state, _ => tp
  | root, member => substitution (root.sigma member) (root.sigma_odd member) sm

/-- Packaging a family preserves the value at every member, with its length proof explicit. -/
theorem import_familyPack_get {α : Type} {size : Nat} {values : List α}
    {output : Fin size → α} (h : familyPack size values output) (member : Fin size) :
    output member = values[member.val]'(by have := h.2.1; omega) := by
  obtain ⟨position, hposition, hvalue⟩ := h.2.2 member
  simpa only [List.get_eq_getElem, hposition] using hvalue

/-- The root and online key definitions sample exactly the same four tagged blocks. -/
theorem import_rootKey_eq {model : HashModel} {key : ByteArray} {pre : Blob}
    {tags : List HashTagComponent} :
    rootKey (n := 8) model key pre tags Q 25 = keyPairG model key Q 25 pre tags := by
  funext row column
  fin_cases row <;> fin_cases column <;> rfl

/-- Every packaged initial root member has its closed-form target and error bound one. -/
theorem initial_import_rel {backend : BackendContext} {model : HashModel}
    {params : W_Stage_initial.Params} (ctx : W_Stage_initial.generatedRoot.Ctx backend model params)
    {outputs : _}
    (hbody : W_Stage_initial.generatedRoot.body backend model params ctx.inputs outputs ctx.witness)
    {sm tp : ErrorPoly 8} (hmask : ctx.inputs.1 0 0 = reducePoly Q 8 sm)
    (hpay : ctx.inputs.2.2.2.2.1 0 0 = reducePoly Q 8 tp)
    {G : Row} (hG : gadgetMatrixRuns backend 4096 25 G)
    (root : ImportRoot) (member : Fin root.members) :
    Rel sm (rootKey model ctx.inputs.2.2.1 root.prefix (root.tags member) Q 25)
      (fun component ↦ reducePoly Q 8
        (initialImportValue ctx root member * unitPair (initialImportMask sm tp root member)
          component)
        • G)
      (initialImportRows ctx root member) 1 := by
  have F := W_Stage_initial.generatedRoot.facts (ctx := ctx) hbody
  cases root with
  | label0 =>
    simp only [initialImportRows]
    rw [import_familyPack_get F.c3.h269_0 member]
    rw [import_familyPack_get F.c3.h294_0 member]
    fin_cases member
    · obtain ⟨hg, hrel⟩ := initial_rel_row_dio2_label_0_0 hbody hmask
      rw [gadgetMatrixRuns_unique hg hG] at hrel
      simpa [initialImportValue, initialImportMask, ImportRoot.prefix, ImportRoot.tags,
        ImportRoot.sigma, labelPre, labelInPre, privatePrefix, statePre, helperPre, memberTagG]
        using hrel
    · obtain ⟨hg, hrel⟩ := initial_rel_row_dio2_label_0_1 hbody hmask
      rw [gadgetMatrixRuns_unique hg hG] at hrel
      simpa [initialImportValue, initialImportMask, ImportRoot.prefix, ImportRoot.tags,
        ImportRoot.sigma, labelPre, labelInPre, privatePrefix, statePre, helperPre, memberTagG]
        using hrel
  | label =>
    simp only [initialImportRows]
    rw [import_familyPack_get F.c4.h367_0 member]
    rw [import_familyPack_get F.c5.h416_0 member]
    fin_cases member
    · obtain ⟨hg, hrel⟩ := initial_rel_row_dio2_label_0 hbody hmask
      rw [gadgetMatrixRuns_unique hg hG] at hrel
      simpa [initialImportValue, initialImportMask, ImportRoot.prefix, ImportRoot.tags,
        ImportRoot.sigma, labelPre, labelInPre, privatePrefix, statePre, helperPre, memberTagG]
        using hrel
    · obtain ⟨hg, hrel⟩ := initial_rel_row_dio2_label_1 hbody hmask
      rw [gadgetMatrixRuns_unique hg hG] at hrel
      simpa [initialImportValue, initialImportMask, ImportRoot.prefix, ImportRoot.tags,
        ImportRoot.sigma, labelPre, labelInPre, privatePrefix, statePre, helperPre, memberTagG]
        using hrel
    · obtain ⟨hg, hrel⟩ := initial_rel_row_dio2_label_2 hbody hmask
      rw [gadgetMatrixRuns_unique hg hG] at hrel
      simpa [initialImportValue, initialImportMask, ImportRoot.prefix, ImportRoot.tags,
        ImportRoot.sigma, labelPre, labelInPre, privatePrefix, statePre, helperPre, memberTagG]
        using hrel
    · obtain ⟨hg, hrel⟩ := initial_rel_row_dio2_label_3 hbody hmask
      rw [gadgetMatrixRuns_unique hg hG] at hrel
      simpa [initialImportValue, initialImportMask, ImportRoot.prefix, ImportRoot.tags,
        ImportRoot.sigma, labelPre, labelInPre, privatePrefix, statePre, helperPre, memberTagG]
        using hrel
  | one =>
    simp only [initialImportRows]
    fin_cases member
    · obtain ⟨hg, hrel⟩ := initial_rel_row_dio2_one_private_0 hbody hmask hpay
      rw [gadgetMatrixRuns_unique hg hG] at hrel
      simpa [initialImportValue, initialImportMask, ImportRoot.prefix, ImportRoot.tags,
        ImportRoot.sigma, labelPre, labelInPre, privatePrefix, statePre, helperPre, memberTagG]
        using hrel
  | state =>
    simp only [initialImportRows]
    fin_cases member
    · obtain ⟨hg, hrel⟩ := initial_rel_row_dio2_state_0 hbody hmask hpay
      rw [gadgetMatrixRuns_unique hg hG] at hrel
      simpa [initialImportValue, initialImportMask, ImportRoot.prefix, ImportRoot.tags,
        ImportRoot.sigma, labelPre, labelInPre, privatePrefix, statePre, helperPre, memberTagG]
        using hrel
  | aut3 =>
    simp only [initialImportRows]
    rw [import_familyPack_get F.c1.h108_0 member]
    rw [import_familyPack_get F.c1.h121_0 member]
    fin_cases member
    · obtain ⟨hg, hrel⟩ := initial_rel_helper_Aut_3_0 hbody hmask
      rw [gadgetMatrixRuns_unique hg hG] at hrel
      simpa [initialImportValue, initialImportMask, ImportRoot.prefix, ImportRoot.tags,
        ImportRoot.sigma, labelPre, labelInPre, privatePrefix, statePre, helperPre, memberTagG]
        using hrel
  | aut5 =>
    simp only [initialImportRows]
    rw [import_familyPack_get F.c1.h136_0 member]
    rw [import_familyPack_get F.c1.h149_0 member]
    fin_cases member
    · obtain ⟨hg, hrel⟩ := initial_rel_helper_Aut_5_0 hbody hmask
      rw [gadgetMatrixRuns_unique hg hG] at hrel
      simpa [initialImportValue, initialImportMask, ImportRoot.prefix, ImportRoot.tags,
        ImportRoot.sigma, labelPre, labelInPre, privatePrefix, statePre, helperPre, memberTagG]
        using hrel
  | aut7 =>
    simp only [initialImportRows]
    rw [import_familyPack_get F.c2.h164_0 member]
    rw [import_familyPack_get F.c2.h177_0 member]
    fin_cases member
    · obtain ⟨hg, hrel⟩ := initial_rel_helper_Aut_7_0 hbody hmask
      rw [gadgetMatrixRuns_unique hg hG] at hrel
      simpa [initialImportValue, initialImportMask, ImportRoot.prefix, ImportRoot.tags,
        ImportRoot.sigma, labelPre, labelInPre, privatePrefix, statePre, helperPre, memberTagG]
        using hrel
  | aut9 =>
    simp only [initialImportRows]
    rw [import_familyPack_get F.c2.h192_0 member]
    rw [import_familyPack_get F.c2.h205_0 member]
    fin_cases member
    · obtain ⟨hg, hrel⟩ := initial_rel_helper_Aut_9_0 hbody hmask
      rw [gadgetMatrixRuns_unique hg hG] at hrel
      simpa [initialImportValue, initialImportMask, ImportRoot.prefix, ImportRoot.tags,
        ImportRoot.sigma, labelPre, labelInPre, privatePrefix, statePre, helperPre, memberTagG]
        using hrel
  | aut11 =>
    simp only [initialImportRows]
    rw [import_familyPack_get F.c0.h24_0 member]
    rw [import_familyPack_get F.c0.h37_0 member]
    fin_cases member
    · obtain ⟨hg, hrel⟩ := initial_rel_helper_Aut_11_0 hbody hmask
      rw [gadgetMatrixRuns_unique hg hG] at hrel
      simpa [initialImportValue, initialImportMask, ImportRoot.prefix, ImportRoot.tags,
        ImportRoot.sigma, labelPre, labelInPre, privatePrefix, statePre, helperPre, memberTagG]
        using hrel
  | aut13 =>
    simp only [initialImportRows]
    rw [import_familyPack_get F.c0.h52_0 member]
    rw [import_familyPack_get F.c0.h65_0 member]
    fin_cases member
    · obtain ⟨hg, hrel⟩ := initial_rel_helper_Aut_13_0 hbody hmask
      rw [gadgetMatrixRuns_unique hg hG] at hrel
      simpa [initialImportValue, initialImportMask, ImportRoot.prefix, ImportRoot.tags,
        ImportRoot.sigma, labelPre, labelInPre, privatePrefix, statePre, helperPre, memberTagG]
        using hrel
  | aut15 =>
    simp only [initialImportRows]
    rw [import_familyPack_get F.c1.h80_0 member]
    rw [import_familyPack_get F.c1.h93_0 member]
    fin_cases member
    · obtain ⟨hg, hrel⟩ := initial_rel_helper_Aut_15_0 hbody hmask
      rw [gadgetMatrixRuns_unique hg hG] at hrel
      simpa [initialImportValue, initialImportMask, ImportRoot.prefix, ImportRoot.tags,
        ImportRoot.sigma, labelPre, labelInPre, privatePrefix, statePre, helperPre, memberTagG]
        using hrel
  | exp8 =>
    simp only [initialImportRows]
    rw [import_familyPack_get F.c2.h220_0 member]
    rw [import_familyPack_get F.c3.h233_0 member]
    fin_cases member
    · obtain ⟨hg, hrel⟩ := initial_rel_helper_ExpLayout_stride_log2_0_width_log2_3_0 hbody hmask
      rw [gadgetMatrixRuns_unique hg hG] at hrel
      simpa [initialImportValue, initialImportMask, ImportRoot.prefix, ImportRoot.tags,
        ImportRoot.sigma, labelPre, labelInPre, privatePrefix, statePre, helperPre, memberTagG]
        using hrel
    · obtain ⟨hg, hrel⟩ := initial_rel_helper_ExpLayout_stride_log2_0_width_log2_3_1 hbody hmask
      rw [gadgetMatrixRuns_unique hg hG] at hrel
      simpa [initialImportValue, initialImportMask, ImportRoot.prefix, ImportRoot.tags,
        ImportRoot.sigma, labelPre, labelInPre, privatePrefix, statePre, helperPre, memberTagG]
        using hrel
    · obtain ⟨hg, hrel⟩ := initial_rel_helper_ExpLayout_stride_log2_0_width_log2_3_2 hbody hmask
      rw [gadgetMatrixRuns_unique hg hG] at hrel
      simpa [initialImportValue, initialImportMask, ImportRoot.prefix, ImportRoot.tags,
        ImportRoot.sigma, labelPre, labelInPre, privatePrefix, statePre, helperPre, memberTagG]
        using hrel
    · obtain ⟨hg, hrel⟩ := initial_rel_helper_ExpLayout_stride_log2_0_width_log2_3_3 hbody hmask
      rw [gadgetMatrixRuns_unique hg hG] at hrel
      simpa [initialImportValue, initialImportMask, ImportRoot.prefix, ImportRoot.tags,
        ImportRoot.sigma, labelPre, labelInPre, privatePrefix, statePre, helperPre, memberTagG]
        using hrel
    · obtain ⟨hg, hrel⟩ := initial_rel_helper_ExpLayout_stride_log2_0_width_log2_3_4 hbody hmask
      rw [gadgetMatrixRuns_unique hg hG] at hrel
      simpa [initialImportValue, initialImportMask, ImportRoot.prefix, ImportRoot.tags,
        ImportRoot.sigma, labelPre, labelInPre, privatePrefix, statePre, helperPre, memberTagG]
        using hrel
    · obtain ⟨hg, hrel⟩ := initial_rel_helper_ExpLayout_stride_log2_0_width_log2_3_5 hbody hmask
      rw [gadgetMatrixRuns_unique hg hG] at hrel
      simpa [initialImportValue, initialImportMask, ImportRoot.prefix, ImportRoot.tags,
        ImportRoot.sigma, labelPre, labelInPre, privatePrefix, statePre, helperPre, memberTagG]
        using hrel
    · obtain ⟨hg, hrel⟩ := initial_rel_helper_ExpLayout_stride_log2_0_width_log2_3_6 hbody hmask
      rw [gadgetMatrixRuns_unique hg hG] at hrel
      simpa [initialImportValue, initialImportMask, ImportRoot.prefix, ImportRoot.tags,
        ImportRoot.sigma, labelPre, labelInPre, privatePrefix, statePre, helperPre, memberTagG]
        using hrel
    · obtain ⟨hg, hrel⟩ := initial_rel_helper_ExpLayout_stride_log2_0_width_log2_3_7 hbody hmask
      rw [gadgetMatrixRuns_unique hg hG] at hrel
      simpa [initialImportValue, initialImportMask, ImportRoot.prefix, ImportRoot.tags,
        ImportRoot.sigma, labelPre, labelInPre, privatePrefix, statePre, helperPre, memberTagG]
        using hrel
  | exp4 =>
    simp only [initialImportRows]
    rw [import_familyPack_get F.c3.h234_0 member]
    rw [import_familyPack_get F.c3.h235_0 member]
    fin_cases member
    · obtain ⟨hg, hrel⟩ := initial_rel_helper_ExpLayout_stride_log2_1_width_log2_2_0 hbody hmask
      rw [gadgetMatrixRuns_unique hg hG] at hrel
      simpa [initialImportValue, initialImportMask, ImportRoot.prefix, ImportRoot.tags,
        ImportRoot.sigma, labelPre, labelInPre, privatePrefix, statePre, helperPre, memberTagG]
        using hrel
    · obtain ⟨hg, hrel⟩ := initial_rel_helper_ExpLayout_stride_log2_1_width_log2_2_1 hbody hmask
      rw [gadgetMatrixRuns_unique hg hG] at hrel
      simpa [initialImportValue, initialImportMask, ImportRoot.prefix, ImportRoot.tags,
        ImportRoot.sigma, labelPre, labelInPre, privatePrefix, statePre, helperPre, memberTagG]
        using hrel
    · obtain ⟨hg, hrel⟩ := initial_rel_helper_ExpLayout_stride_log2_1_width_log2_2_2 hbody hmask
      rw [gadgetMatrixRuns_unique hg hG] at hrel
      simpa [initialImportValue, initialImportMask, ImportRoot.prefix, ImportRoot.tags,
        ImportRoot.sigma, labelPre, labelInPre, privatePrefix, statePre, helperPre, memberTagG]
        using hrel
    · obtain ⟨hg, hrel⟩ := initial_rel_helper_ExpLayout_stride_log2_1_width_log2_2_3 hbody hmask
      rw [gadgetMatrixRuns_unique hg hG] at hrel
      simpa [initialImportValue, initialImportMask, ImportRoot.prefix, ImportRoot.tags,
        ImportRoot.sigma, labelPre, labelInPre, privatePrefix, statePre, helperPre, memberTagG]
        using hrel

/-- The imported row pairs read by step 0's online proof modules. -/
inductive Import0Port where
  | labelState | labelSigma | labelFresh | labelInput | privateSigma | privateFresh | privateRoot |
    state | aut3 | aut5 | aut7 | aut9 | aut11 | aut13 | aut15 | helperSigma | helperLabel |
    helperState
  deriving DecidableEq

/-- The initial root family supplying an online input of step 0. -/
def Import0Port.root : Import0Port → ImportRoot
  | .labelState => .label0
  | .labelSigma => .label0
  | .labelFresh => .label0
  | .labelInput => .label
  | .privateSigma => .one
  | .privateFresh => .one
  | .privateRoot => .one
  | .state => .state
  | .aut3 => .aut3
  | .aut5 => .aut5
  | .aut7 => .aut7
  | .aut9 => .aut9
  | .aut11 => .aut11
  | .aut13 => .aut13
  | .aut15 => .aut15
  | .helperSigma => .exp8
  | .helperLabel => .exp8
  | .helperState => .exp4

/-- Family cardinality inherited from the initial root object. -/
abbrev Import0Port.members (port : Import0Port) : Nat := port.root.members

/-- The concrete input row pair of the import-to-online connection, member by member. -/
noncomputable def import0InputRows {params : W_Stage_import_0.Params}
    (ctx : W_Stage_import_0.generatedRoot.Ctx params) :
    (port : Import0Port) → Fin port.members → Fin 2 → Row
  | .labelState, member => ![W_Stage_import_0.generatedRoot.w_2_0 ctx member,
      W_Stage_import_0.generatedRoot.w_3_0 ctx member]
  | .labelSigma, member => ![W_Stage_import_0.generatedRoot.w_2_0 ctx member,
      W_Stage_import_0.generatedRoot.w_3_0 ctx member]
  | .labelFresh, member => ![W_Stage_import_0.generatedRoot.w_2_0 ctx member,
      W_Stage_import_0.generatedRoot.w_3_0 ctx member]
  | .labelInput, member => ![W_Stage_import_0.generatedRoot.w_15_0 ctx member,
      W_Stage_import_0.generatedRoot.w_16_0 ctx member]
  | .privateSigma, _ => ![W_Stage_import_0.generatedRoot.w_0_0 ctx,
      W_Stage_import_0.generatedRoot.w_1_0 ctx]
  | .privateFresh, _ => ![W_Stage_import_0.generatedRoot.w_0_0 ctx,
      W_Stage_import_0.generatedRoot.w_1_0 ctx]
  | .privateRoot, _ => ![W_Stage_import_0.generatedRoot.w_0_0 ctx,
      W_Stage_import_0.generatedRoot.w_1_0 ctx]
  | .state, _ => ![W_Stage_import_0.generatedRoot.w_7_0 ctx,
      W_Stage_import_0.generatedRoot.w_8_0 ctx]
  | .aut3, member => ![W_Stage_import_0.generatedRoot.w_29_0 ctx member,
      W_Stage_import_0.generatedRoot.w_30_0 ctx member]
  | .aut5, member => ![W_Stage_import_0.generatedRoot.w_32_0 ctx member,
      W_Stage_import_0.generatedRoot.w_33_0 ctx member]
  | .aut7, member => ![W_Stage_import_0.generatedRoot.w_35_0 ctx member,
      W_Stage_import_0.generatedRoot.w_36_0 ctx member]
  | .aut9, member => ![W_Stage_import_0.generatedRoot.w_38_0 ctx member,
      W_Stage_import_0.generatedRoot.w_39_0 ctx member]
  | .aut11, member => ![W_Stage_import_0.generatedRoot.w_20_0 ctx member,
      W_Stage_import_0.generatedRoot.w_21_0 ctx member]
  | .aut13, member => ![W_Stage_import_0.generatedRoot.w_23_0 ctx member,
      W_Stage_import_0.generatedRoot.w_24_0 ctx member]
  | .aut15, member => ![W_Stage_import_0.generatedRoot.w_26_0 ctx member,
      W_Stage_import_0.generatedRoot.w_27_0 ctx member]
  | .helperSigma, member => ![W_Stage_import_0.generatedRoot.w_41_0 ctx member,
      W_Stage_import_0.generatedRoot.w_42_0 ctx member]
  | .helperLabel, member => ![W_Stage_import_0.generatedRoot.w_41_0 ctx member,
      W_Stage_import_0.generatedRoot.w_42_0 ctx member]
  | .helperState, member => ![W_Stage_import_0.generatedRoot.w_44_0 ctx member,
      W_Stage_import_0.generatedRoot.w_45_0 ctx member]

/-- The concrete output row pair of the import-to-online connection, member by member. -/
noncomputable def import0OutputRows {params : W_Stage_import_0.Params}
    (ctx : W_Stage_import_0.generatedRoot.Ctx params) :
    (port : Import0Port) → Fin port.members → Fin 2 → Row
  | .labelState, member => ![W_Stage_import_0.generatedRoot.w_6_0 ctx member,
      W_Stage_import_0.generatedRoot.w_6_1 ctx member]
  | .labelSigma, member => ![W_Stage_import_0.generatedRoot.w_11_0 ctx member,
      W_Stage_import_0.generatedRoot.w_11_1 ctx member]
  | .labelFresh, member => ![W_Stage_import_0.generatedRoot.w_14_0 ctx member,
      W_Stage_import_0.generatedRoot.w_14_1 ctx member]
  | .labelInput, member => ![W_Stage_import_0.generatedRoot.w_19_0 ctx member,
      W_Stage_import_0.generatedRoot.w_19_1 ctx member]
  | .privateSigma, _ => ![W_Stage_import_0.generatedRoot.w_0_0 ctx,
      W_Stage_import_0.generatedRoot.w_1_0 ctx]
  | .privateFresh, _ => ![W_Stage_import_0.generatedRoot.w_0_0 ctx,
      W_Stage_import_0.generatedRoot.w_1_0 ctx]
  | .privateRoot, _ => ![W_Stage_import_0.generatedRoot.w_0_0 ctx,
      W_Stage_import_0.generatedRoot.w_1_0 ctx]
  | .state, _ => ![W_Stage_import_0.generatedRoot.w_7_0 ctx,
      W_Stage_import_0.generatedRoot.w_8_0 ctx]
  | .aut3, member => ![W_Stage_import_0.generatedRoot.w_68_0 ctx member,
      W_Stage_import_0.generatedRoot.w_68_1 ctx member]
  | .aut5, member => ![W_Stage_import_0.generatedRoot.w_69_0 ctx member,
      W_Stage_import_0.generatedRoot.w_69_1 ctx member]
  | .aut7, member => ![W_Stage_import_0.generatedRoot.w_70_0 ctx member,
      W_Stage_import_0.generatedRoot.w_70_1 ctx member]
  | .aut9, member => ![W_Stage_import_0.generatedRoot.w_71_0 ctx member,
      W_Stage_import_0.generatedRoot.w_71_1 ctx member]
  | .aut11, member => ![W_Stage_import_0.generatedRoot.w_65_0 ctx member,
      W_Stage_import_0.generatedRoot.w_65_1 ctx member]
  | .aut13, member => ![W_Stage_import_0.generatedRoot.w_66_0 ctx member,
      W_Stage_import_0.generatedRoot.w_66_1 ctx member]
  | .aut15, member => ![W_Stage_import_0.generatedRoot.w_67_0 ctx member,
      W_Stage_import_0.generatedRoot.w_67_1 ctx member]
  | .helperSigma, member => ![W_Stage_import_0.generatedRoot.w_72_0 ctx member,
      W_Stage_import_0.generatedRoot.w_72_1 ctx member]
  | .helperLabel, member => ![W_Stage_import_0.generatedRoot.w_63_0 ctx member,
      W_Stage_import_0.generatedRoot.w_63_1 ctx member]
  | .helperState, member => ![W_Stage_import_0.generatedRoot.w_55_0 ctx member,
      W_Stage_import_0.generatedRoot.w_55_1 ctx member]

/-- The concrete online row pair of the import-to-online connection, member by member. -/
noncomputable def import0OnlineRows {backend : BackendContext} {model : HashModel}
    {params : W_Stage_online_0.Params}
    (ctx : W_Stage_online_0.generatedRoot.Ctx backend model params) :
    (port : Import0Port) → Fin port.members → Fin 2 → Row
  | .labelState, member => ![W_Stage_online_0.generatedRoot.w_257_0 ctx member,
      W_Stage_online_0.generatedRoot.w_262_0 ctx member]
  | .labelSigma, member => ![W_Stage_online_0.generatedRoot.w_16_0 ctx member,
      W_Stage_online_0.generatedRoot.w_17_0 ctx member]
  | .labelFresh, member => ![W_Stage_online_0.generatedRoot.w_1169_0 ctx member,
      W_Stage_online_0.generatedRoot.w_1170_0 ctx member]
  | .labelInput, member => ![W_Stage_online_0.generatedRoot.w_372_0 ctx member,
      W_Stage_online_0.generatedRoot.w_377_0 ctx member]
  | .privateSigma, _ => ![W_Stage_online_0.generatedRoot.w_21_0 ctx,
      W_Stage_online_0.generatedRoot.w_25_0 ctx]
  | .privateFresh, _ => ![W_Stage_online_0.generatedRoot.w_1174_0 ctx,
      W_Stage_online_0.generatedRoot.w_1178_0 ctx]
  | .privateRoot, _ => ![W_Stage_online_0.generatedRoot.w_1229_0 ctx,
      W_Stage_online_0.generatedRoot.w_1232_0 ctx]
  | .state, _ => ![W_Stage_online_0.generatedRoot.w_275_0 ctx,
      W_Stage_online_0.generatedRoot.w_290_0 ctx]
  | .aut3, member => ![W_Stage_online_0.generatedRoot.w_764_0 ctx member,
      W_Stage_online_0.generatedRoot.w_765_0 ctx member]
  | .aut5, member => ![W_Stage_online_0.generatedRoot.w_822_0 ctx member,
      W_Stage_online_0.generatedRoot.w_823_0 ctx member]
  | .aut7, member => ![W_Stage_online_0.generatedRoot.w_880_0 ctx member,
      W_Stage_online_0.generatedRoot.w_881_0 ctx member]
  | .aut9, member => ![W_Stage_online_0.generatedRoot.w_938_0 ctx member,
      W_Stage_online_0.generatedRoot.w_939_0 ctx member]
  | .aut11, member => ![W_Stage_online_0.generatedRoot.w_996_0 ctx member,
      W_Stage_online_0.generatedRoot.w_997_0 ctx member]
  | .aut13, member => ![W_Stage_online_0.generatedRoot.w_1054_0 ctx member,
      W_Stage_online_0.generatedRoot.w_1055_0 ctx member]
  | .aut15, member => ![W_Stage_online_0.generatedRoot.w_1112_0 ctx member,
      W_Stage_online_0.generatedRoot.w_1113_0 ctx member]
  | .helperSigma, member => ![W_Stage_online_0.generatedRoot.w_6_0 ctx member,
      W_Stage_online_0.generatedRoot.w_8_0 ctx member]
  | .helperLabel, member => ![W_Stage_online_0.generatedRoot.w_13_0 ctx member,
      W_Stage_online_0.generatedRoot.w_14_0 ctx member]
  | .helperState, member => ![W_Stage_online_0.generatedRoot.w_255_0 ctx member,
      W_Stage_online_0.generatedRoot.w_256_0 ctx member]

/-- The generated copy and label adapters preserve the row pair at every consumed member. -/
theorem import0_rows_eq {params : W_Stage_import_0.Params}
    (ctx : W_Stage_import_0.generatedRoot.Ctx params) {outputs : _}
    (F : W_Stage_import_0.generatedRoot.Facts ctx outputs)
    (port : Import0Port) (member : Fin port.members) :
    import0OutputRows ctx port member = import0InputRows ctx port member := by
  cases port with
  | labelState =>
    exact congrArg (fun value ↦ ![value.1, value.2.1])
      (W_Stage_import_0.parallel_generatedRoot_6.sem (F.c0.h6_2 member))
  | labelSigma =>
    exact congrArg (fun value ↦ ![value.1, value.2.1])
      (W_Stage_import_0.parallel_generatedRoot_11.sem (F.c0.h11_2 member))
  | labelFresh =>
    exact congrArg (fun value ↦ ![value.1, value.2.1])
      (W_Stage_import_0.parallel_generatedRoot_14.sem (F.c0.h14_2 member))
  | labelInput =>
    exact congrArg (fun value ↦ ![value.1, value.2.1])
      (W_Stage_import_0.parallel_generatedRoot_19.sem (F.c0.h19_2 member))
  | privateSigma => rfl
  | privateFresh => rfl
  | privateRoot => rfl
  | state => rfl
  | aut3 =>
    exact congrArg (fun value ↦ ![value.1, value.2.1])
      (W_Stage_import_0.parallel_generatedRoot_68.sem (F.c2.h68_1 member))
  | aut5 =>
    exact congrArg (fun value ↦ ![value.1, value.2.1])
      (W_Stage_import_0.parallel_generatedRoot_69.sem (F.c2.h69_1 member))
  | aut7 =>
    exact congrArg (fun value ↦ ![value.1, value.2.1])
      (W_Stage_import_0.parallel_generatedRoot_70.sem (F.c2.h70_1 member))
  | aut9 =>
    exact congrArg (fun value ↦ ![value.1, value.2.1])
      (W_Stage_import_0.parallel_generatedRoot_71.sem (F.c2.h71_1 member))
  | aut11 =>
    exact congrArg (fun value ↦ ![value.1, value.2.1])
      (W_Stage_import_0.parallel_generatedRoot_65.sem (F.c1.h65_1 member))
  | aut13 =>
    exact congrArg (fun value ↦ ![value.1, value.2.1])
      (W_Stage_import_0.parallel_generatedRoot_66.sem (F.c2.h66_1 member))
  | aut15 =>
    exact congrArg (fun value ↦ ![value.1, value.2.1])
      (W_Stage_import_0.parallel_generatedRoot_67.sem (F.c2.h67_1 member))
  | helperSigma =>
    exact congrArg (fun value ↦ ![value.1, value.2.1])
      (W_Stage_import_0.parallel_generatedRoot_72.sem (F.c2.h72_1 member))
  | helperLabel =>
    exact congrArg (fun value ↦ ![value.1, value.2.1])
      (W_Stage_import_0.parallel_generatedRoot_63.sem (F.c1.h63_1 member))
  | helperState =>
    exact congrArg (fun value ↦ ![value.1, value.2.1])
      (W_Stage_import_0.parallel_generatedRoot_55.sem (F.c1.h55_1 member))

/-- Any initial `Rel` or unfolded `RootRow` passes through import zero without changing its
keys, targets, or bound; this also permits bounds other than the initial unit bound. -/
theorem import0_rel {params : W_Stage_import_0.Params}
    (ctx : W_Stage_import_0.generatedRoot.Ctx params) {outputs : _}
    (F : W_Stage_import_0.generatedRoot.Facts ctx outputs)
    {sm : ErrorPoly 8} (port : Import0Port) (member : Fin port.members)
    {keys : Fin 2 → Fin 2 → Row} {targets : Fin 2 → Row} {bound : Nat}
    (hroot : Rel sm keys targets (import0InputRows ctx port member) bound) :
    Rel sm keys targets (import0OutputRows ctx port member) bound := by
  rw [import0_rows_eq ctx F port member]
  exact hroot

/-- Initial root facts supply every consumed import output, retaining the unit error bound. -/
theorem import0_initial_rel {backend : BackendContext} {model : HashModel}
    {initialParams : W_Stage_initial.Params}
    (initial : W_Stage_initial.generatedRoot.Ctx backend model initialParams)
    {initialOutputs : _}
    (hbody : W_Stage_initial.generatedRoot.body backend model initialParams initial.inputs
      initialOutputs initial.witness)
    {params : W_Stage_import_0.Params} (ctx : W_Stage_import_0.generatedRoot.Ctx params)
    {outputs : _} (F : W_Stage_import_0.generatedRoot.Facts ctx outputs)
    (hwire : ∀ (port : Import0Port) (member : Fin port.members),
      import0InputRows ctx port member = initialImportRows initial port.root member)
    {sm tp : ErrorPoly 8} (hmask : initial.inputs.1 0 0 = reducePoly Q 8 sm)
    (hpay : initial.inputs.2.2.2.2.1 0 0 = reducePoly Q 8 tp)
    {G : Row} (hG : gadgetMatrixRuns backend 4096 25 G)
    (port : Import0Port) (member : Fin port.members) :
    Rel sm (rootKey model initial.inputs.2.2.1 port.root.prefix (port.root.tags member) Q 25)
      (fun component ↦ reducePoly Q 8
        (initialImportValue initial port.root member *
          unitPair (initialImportMask sm tp port.root member) component) • G)
      (import0OutputRows ctx port member) 1 := by
  rw [import0_rows_eq ctx F port member, hwire port member]
  exact initial_import_rel initial hbody hmask hpay hG port.root member

/-- Every named online input has its initial target under the online root key, with error one.
Specializing a label port gives `hlab`/`ha`/`hone`; the state and private ports give `hs`/`hpriv`;
the automorphism and layout ports give `hhelp`. No internal refreshed-row premise is assumed. -/
theorem import0_online_rel {backend : BackendContext} {model : HashModel}
    {initialParams : W_Stage_initial.Params}
    (initial : W_Stage_initial.generatedRoot.Ctx backend model initialParams)
    {initialOutputs : _}
    (hbody : W_Stage_initial.generatedRoot.body backend model initialParams initial.inputs
      initialOutputs initial.witness)
    {params : W_Stage_import_0.Params} (ctx : W_Stage_import_0.generatedRoot.Ctx params)
    {outputs : _} (F : W_Stage_import_0.generatedRoot.Facts ctx outputs)
    {onlineParams : W_Stage_online_0.Params}
    (online : W_Stage_online_0.generatedRoot.Ctx backend model onlineParams)
    (hinitial : ∀ (port : Import0Port) (member : Fin port.members),
      import0InputRows ctx port member = initialImportRows initial port.root member)
    (honline : ∀ (port : Import0Port) (member : Fin port.members),
      import0OnlineRows online port member = import0OutputRows ctx port member)
    (hkey : W_Stage_online_0.generatedRoot.w_7_0 online = initial.inputs.2.2.1)
    {sm tp : ErrorPoly 8} (hmask : initial.inputs.1 0 0 = reducePoly Q 8 sm)
    (hpay : initial.inputs.2.2.2.2.1 0 0 = reducePoly Q 8 tp)
    {G : Row} (hG : gadgetMatrixRuns backend 4096 25 G)
    (port : Import0Port) (member : Fin port.members) :
    Rel sm (keyPairG model (W_Stage_online_0.generatedRoot.w_7_0 online) Q 25
      port.root.prefix (port.root.tags member))
      (fun component ↦ reducePoly Q 8
        (initialImportValue initial port.root member *
          unitPair (initialImportMask sm tp port.root member) component) • G)
      (import0OnlineRows online port member) 1 := by
  rw [honline port member, hkey, ← import_rootKey_eq]
  exact import0_initial_rel initial hbody ctx F hinitial hmask hpay hG port member

/-- The imported row pairs read by step 1's online proof modules. -/
inductive Import1Port where
  | labelState | privateRoot | state | helperState | helperSmudge
  deriving DecidableEq

/-- The initial root family supplying an online input of step 1. -/
def Import1Port.root : Import1Port → ImportRoot
  | .labelState => .label0
  | .privateRoot => .one
  | .state => .state
  | .helperState => .exp4
  | .helperSmudge => .exp8

/-- Family cardinality inherited from the initial root object. -/
abbrev Import1Port.members (port : Import1Port) : Nat := port.root.members

/-- The concrete input row pair of the import-to-online connection, member by member. -/
noncomputable def import1InputRows {params : W_Stage_import_1.Params}
    (ctx : W_Stage_import_1.generatedRoot.Ctx params) :
    (port : Import1Port) → Fin port.members → Fin 2 → Row
  | .labelState, member => ![W_Stage_import_1.generatedRoot.w_2_0 ctx member,
      W_Stage_import_1.generatedRoot.w_3_0 ctx member]
  | .privateRoot, _ => ![W_Stage_import_1.generatedRoot.w_0_0 ctx,
      W_Stage_import_1.generatedRoot.w_1_0 ctx]
  | .state, _ => ![W_Stage_import_1.generatedRoot.w_7_0 ctx,
      W_Stage_import_1.generatedRoot.w_8_0 ctx]
  | .helperState, member => ![W_Stage_import_1.generatedRoot.w_12_0 ctx member,
      W_Stage_import_1.generatedRoot.w_13_0 ctx member]
  | .helperSmudge, member => ![W_Stage_import_1.generatedRoot.w_9_0 ctx member,
      W_Stage_import_1.generatedRoot.w_10_0 ctx member]

/-- The concrete output row pair of the import-to-online connection, member by member. -/
noncomputable def import1OutputRows {params : W_Stage_import_1.Params}
    (ctx : W_Stage_import_1.generatedRoot.Ctx params) :
    (port : Import1Port) → Fin port.members → Fin 2 → Row
  | .labelState, member => ![W_Stage_import_1.generatedRoot.w_6_0 ctx member,
      W_Stage_import_1.generatedRoot.w_6_1 ctx member]
  | .privateRoot, _ => ![W_Stage_import_1.generatedRoot.w_0_0 ctx,
      W_Stage_import_1.generatedRoot.w_1_0 ctx]
  | .state, _ => ![W_Stage_import_1.generatedRoot.w_7_0 ctx,
      W_Stage_import_1.generatedRoot.w_8_0 ctx]
  | .helperState, member => ![W_Stage_import_1.generatedRoot.w_16_0 ctx member,
      W_Stage_import_1.generatedRoot.w_16_1 ctx member]
  | .helperSmudge, member => ![W_Stage_import_1.generatedRoot.w_17_0 ctx member,
      W_Stage_import_1.generatedRoot.w_17_1 ctx member]

/-- The concrete online row pair of the import-to-online connection, member by member. -/
noncomputable def import1OnlineRows {backend : BackendContext} {model : HashModel}
    {params : W_Stage_online_1.Params}
    (ctx : W_Stage_online_1.generatedRoot.Ctx backend model params) :
    (port : Import1Port) → Fin port.members → Fin 2 → Row
  | .labelState, member => ![W_Stage_online_1.generatedRoot.w_16_0 ctx member,
      W_Stage_online_1.generatedRoot.w_27_0 ctx member]
  | .privateRoot, _ => ![W_Stage_online_1.generatedRoot.w_142_0 ctx,
      W_Stage_online_1.generatedRoot.w_145_0 ctx]
  | .state, _ => ![W_Stage_online_1.generatedRoot.w_46_0 ctx,
      W_Stage_online_1.generatedRoot.w_63_0 ctx]
  | .helperState, member => ![W_Stage_online_1.generatedRoot.w_13_0 ctx member,
      W_Stage_online_1.generatedRoot.w_15_0 ctx member]
  | .helperSmudge, member => ![W_Stage_online_1.generatedRoot.w_136_0 ctx member,
      W_Stage_online_1.generatedRoot.w_137_0 ctx member]

/-- The generated copy and label adapters preserve the row pair at every consumed member. -/
theorem import1_rows_eq {params : W_Stage_import_1.Params}
    (ctx : W_Stage_import_1.generatedRoot.Ctx params) {outputs : _}
    (F : W_Stage_import_1.generatedRoot.Facts ctx outputs)
    (port : Import1Port) (member : Fin port.members) :
    import1OutputRows ctx port member = import1InputRows ctx port member := by
  cases port with
  | labelState =>
    exact congrArg (fun value ↦ ![value.1, value.2.1])
      (W_Stage_import_1.parallel_generatedRoot_6.sem (F.c0.h6_2 member))
  | privateRoot => rfl
  | state => rfl
  | helperState =>
    exact congrArg (fun value ↦ ![value.1, value.2.1])
      (W_Stage_import_1.parallel_generatedRoot_16.sem (F.c0.h16_1 member))
  | helperSmudge =>
    exact congrArg (fun value ↦ ![value.1, value.2.1])
      (W_Stage_import_1.parallel_generatedRoot_17.sem (F.c0.h17_1 member))

/-- Any supplied `Rel` or unfolded `RootRow` passes through import one with the same bound.
In `W_Claim` these are the root relations supplied by the preceding online stage. -/
theorem import1_rel {params : W_Stage_import_1.Params}
    (ctx : W_Stage_import_1.generatedRoot.Ctx params) {outputs : _}
    (F : W_Stage_import_1.generatedRoot.Facts ctx outputs)
    {sm : ErrorPoly 8} (port : Import1Port) (member : Fin port.members)
    {keys : Fin 2 → Fin 2 → Row} {targets : Fin 2 → Row} {bound : Nat}
    (hroot : Rel sm keys targets (import1InputRows ctx port member) bound) :
    Rel sm keys targets (import1OutputRows ctx port member) bound := by
  rw [import1_rows_eq ctx F port member]
  exact hroot

/-- All step-one online input hypotheses follow from its supplied root relations and row
wiring. Targets and bounds remain arbitrary, so this applies to refreshed step-zero outputs
without identifying them with the original initial rows. -/
theorem import1_online_from_rows {backend : BackendContext} {model : HashModel}
    {params : W_Stage_import_1.Params} (ctx : W_Stage_import_1.generatedRoot.Ctx params)
    {outputs : _} (F : W_Stage_import_1.generatedRoot.Facts ctx outputs)
    {onlineParams : W_Stage_online_1.Params}
    (online : W_Stage_online_1.generatedRoot.Ctx backend model onlineParams)
    (hwire : ∀ (port : Import1Port) (member : Fin port.members),
      import1OnlineRows online port member = import1OutputRows ctx port member)
    {sm : ErrorPoly 8}
    (targets : (port : Import1Port) → Fin port.members → Fin 2 → Row)
    (bounds : (port : Import1Port) → Fin port.members → Nat)
    (hroot : ∀ (port : Import1Port) (member : Fin port.members),
      Rel sm (keyPairG model (W_Stage_online_1.generatedRoot.w_14_0 online) Q 25
        port.root.prefix (port.root.tags member))
        (targets port member) (import1InputRows ctx port member) (bounds port member)) :
    ∀ (port : Import1Port) (member : Fin port.members),
      Rel sm (keyPairG model (W_Stage_online_1.generatedRoot.w_14_0 online) Q 25
        port.root.prefix (port.root.tags member))
        (targets port member) (import1OnlineRows online port member) (bounds port member) := by
  intro port member
  rw [hwire port member]
  exact import1_rel ctx F port member (hroot port member)

/-- Initial root facts supply every consumed import output, retaining the unit error bound. -/
theorem import1_initial_rel {backend : BackendContext} {model : HashModel}
    {initialParams : W_Stage_initial.Params}
    (initial : W_Stage_initial.generatedRoot.Ctx backend model initialParams)
    {initialOutputs : _}
    (hbody : W_Stage_initial.generatedRoot.body backend model initialParams initial.inputs
      initialOutputs initial.witness)
    {params : W_Stage_import_1.Params} (ctx : W_Stage_import_1.generatedRoot.Ctx params)
    {outputs : _} (F : W_Stage_import_1.generatedRoot.Facts ctx outputs)
    (hwire : ∀ (port : Import1Port) (member : Fin port.members),
      import1InputRows ctx port member = initialImportRows initial port.root member)
    {sm tp : ErrorPoly 8} (hmask : initial.inputs.1 0 0 = reducePoly Q 8 sm)
    (hpay : initial.inputs.2.2.2.2.1 0 0 = reducePoly Q 8 tp)
    {G : Row} (hG : gadgetMatrixRuns backend 4096 25 G)
    (port : Import1Port) (member : Fin port.members) :
    Rel sm (rootKey model initial.inputs.2.2.1 port.root.prefix (port.root.tags member) Q 25)
      (fun component ↦ reducePoly Q 8
        (initialImportValue initial port.root member *
          unitPair (initialImportMask sm tp port.root member) component) • G)
      (import1OutputRows ctx port member) 1 := by
  rw [import1_rows_eq ctx F port member, hwire port member]
  exact initial_import_rel initial hbody hmask hpay hG port.root member

/-- Every named online input has its initial target under the online root key, with error one.
Specializing a label port gives `hlab`/`ha`/`hone`; the state and private ports give `hs`/`hpriv`;
the automorphism and layout ports give `hhelp`. No internal refreshed-row premise is assumed. -/
theorem import1_online_rel {backend : BackendContext} {model : HashModel}
    {initialParams : W_Stage_initial.Params}
    (initial : W_Stage_initial.generatedRoot.Ctx backend model initialParams)
    {initialOutputs : _}
    (hbody : W_Stage_initial.generatedRoot.body backend model initialParams initial.inputs
      initialOutputs initial.witness)
    {params : W_Stage_import_1.Params} (ctx : W_Stage_import_1.generatedRoot.Ctx params)
    {outputs : _} (F : W_Stage_import_1.generatedRoot.Facts ctx outputs)
    {onlineParams : W_Stage_online_1.Params}
    (online : W_Stage_online_1.generatedRoot.Ctx backend model onlineParams)
    (hinitial : ∀ (port : Import1Port) (member : Fin port.members),
      import1InputRows ctx port member = initialImportRows initial port.root member)
    (honline : ∀ (port : Import1Port) (member : Fin port.members),
      import1OnlineRows online port member = import1OutputRows ctx port member)
    (hkey : W_Stage_online_1.generatedRoot.w_14_0 online = initial.inputs.2.2.1)
    {sm tp : ErrorPoly 8} (hmask : initial.inputs.1 0 0 = reducePoly Q 8 sm)
    (hpay : initial.inputs.2.2.2.2.1 0 0 = reducePoly Q 8 tp)
    {G : Row} (hG : gadgetMatrixRuns backend 4096 25 G)
    (port : Import1Port) (member : Fin port.members) :
    Rel sm (keyPairG model (W_Stage_online_1.generatedRoot.w_14_0 online) Q 25
      port.root.prefix (port.root.tags member))
      (fun component ↦ reducePoly Q 8
        (initialImportValue initial port.root member *
          unitPair (initialImportMask sm tp port.root member) component) • G)
      (import1OnlineRows online port member) 1 := by
  rw [honline port member, hkey, ← import_rootKey_eq]
  exact import1_initial_rel initial hbody ctx F hinitial hmask hpay hG port member

/-- The four public label payload families constructed by import stage zero. -/
inductive Import0Label where
  | state | sigma | fresh | input
  deriving DecidableEq

/-- Label payload cardinality. -/
def Import0Label.members : Import0Label → Nat
  | .input => 4
  | _ => 2

/-- The concrete coefficient family on the public import path. -/
noncomputable def import0Coefficient {params : W_Stage_import_0.Params}
    (ctx : W_Stage_import_0.generatedRoot.Ctx params) :
    (label : Import0Label) → Fin label.members → Int
  | .state => W_Stage_import_0.generatedRoot.w_4_0 ctx
  | .sigma => W_Stage_import_0.generatedRoot.w_9_0 ctx
  | .fresh => W_Stage_import_0.generatedRoot.w_12_0 ctx
  | .input => W_Stage_import_0.generatedRoot.w_17_0 ctx

/-- The concrete exponent family on the public import path. -/
noncomputable def import0Exponent {params : W_Stage_import_0.Params}
    (ctx : W_Stage_import_0.generatedRoot.Ctx params) :
    (label : Import0Label) → Fin label.members → Int
  | .state => W_Stage_import_0.generatedRoot.w_5_0 ctx
  | .sigma => W_Stage_import_0.generatedRoot.w_10_0 ctx
  | .fresh => W_Stage_import_0.generatedRoot.w_13_0 ctx
  | .input => W_Stage_import_0.generatedRoot.w_18_0 ctx

/-- The concrete payload family on the public import path. -/
noncomputable def import0Payload {params : W_Stage_import_0.Params}
    (ctx : W_Stage_import_0.generatedRoot.Ctx params) :
    (label : Import0Label) → Fin label.members → ExactMatrix Q 8 1 1
  | .state => W_Stage_import_0.generatedRoot.w_6_2 ctx
  | .sigma => W_Stage_import_0.generatedRoot.w_11_2 ctx
  | .fresh => W_Stage_import_0.generatedRoot.w_14_2 ctx
  | .input => W_Stage_import_0.generatedRoot.w_19_2 ctx

/-- The concrete onlinepayload family on the public import path. -/
noncomputable def import0OnlinePayload {backend : BackendContext} {model : HashModel}
    {params : W_Stage_online_0.Params}
    (ctx : W_Stage_online_0.generatedRoot.Ctx backend model params) :
    (label : Import0Label) → Fin label.members → ExactMatrix Q 8 1 1
  | .state => W_Stage_online_0.generatedRoot.w_276_0 ctx
  | .sigma => W_Stage_online_0.generatedRoot.w_22_0 ctx
  | .fresh => W_Stage_online_0.generatedRoot.w_1175_0 ctx
  | .input => W_Stage_online_0.generatedRoot.w_391_0 ctx

/-- Every generated label payload is exactly its advertised plaintext monomial. -/
theorem import0_payload_eq {params : W_Stage_import_0.Params}
    (ctx : W_Stage_import_0.generatedRoot.Ctx params) {outputs : _}
    (F : W_Stage_import_0.generatedRoot.Facts ctx outputs)
    (label : Import0Label) (member : Fin label.members) :
    import0Payload ctx label member = fun _ _ ↦ reducePoly Q 8
      (message (import0Coefficient ctx label member)
        (import0Exponent ctx label member)) := by
  cases label with
  | state =>
    exact (congrArg (fun value ↦ value.2.2.1)
      (W_Stage_import_0.parallel_generatedRoot_6.sem (F.c0.h6_2 member))).trans
        (multiplyMonomial_liftInteger _ _)
  | sigma =>
    exact (congrArg (fun value ↦ value.2.2.1)
      (W_Stage_import_0.parallel_generatedRoot_11.sem (F.c0.h11_2 member))).trans
        (multiplyMonomial_liftInteger _ _)
  | fresh =>
    exact (congrArg (fun value ↦ value.2.2.1)
      (W_Stage_import_0.parallel_generatedRoot_14.sem (F.c0.h14_2 member))).trans
        (multiplyMonomial_liftInteger _ _)
  | input =>
    exact (congrArg (fun value ↦ value.2.2.1)
      (W_Stage_import_0.parallel_generatedRoot_19.sem (F.c0.h19_2 member))).trans
        (multiplyMonomial_liftInteger _ _)

/-- The online payload hypothesis follows from import wiring and plaintext agreement.
Plaintext agreement is explicit because the coefficient/exponent families are external inputs. -/
theorem import0_online_payload {backend : BackendContext} {model : HashModel}
    {params : W_Stage_import_0.Params} (ctx : W_Stage_import_0.generatedRoot.Ctx params)
    {outputs : _} (F : W_Stage_import_0.generatedRoot.Facts ctx outputs)
    {onlineParams : W_Stage_online_0.Params}
    (online : W_Stage_online_0.generatedRoot.Ctx backend model onlineParams)
    (label : Import0Label)
    {values : Fin label.members → ErrorPoly 8}
    (hwire : ∀ member, import0OnlinePayload online label member =
      import0Payload ctx label member)
    (hmessage : ∀ member, message (import0Coefficient ctx label member)
      (import0Exponent ctx label member) = values member) :
    ∀ member, import0OnlinePayload online label member =
      fun _ _ ↦ reducePoly Q 8 (values member) := by
  intro member
  rw [hwire member, import0_payload_eq ctx F label member, hmessage member]

/-- The concrete coefficient family on the public import path. -/
noncomputable def import1Coefficient {params : W_Stage_import_1.Params}
    (ctx : W_Stage_import_1.generatedRoot.Ctx params) :
    Fin 2 → Int :=
  W_Stage_import_1.generatedRoot.w_4_0 ctx

/-- The concrete exponent family on the public import path. -/
noncomputable def import1Exponent {params : W_Stage_import_1.Params}
    (ctx : W_Stage_import_1.generatedRoot.Ctx params) :
    Fin 2 → Int :=
  W_Stage_import_1.generatedRoot.w_5_0 ctx

/-- The concrete payload family on the public import path. -/
noncomputable def import1Payload {params : W_Stage_import_1.Params}
    (ctx : W_Stage_import_1.generatedRoot.Ctx params) :
    Fin 2 → ExactMatrix Q 8 1 1 :=
  W_Stage_import_1.generatedRoot.w_6_2 ctx

/-- The concrete onlinepayload family on the public import path. -/
noncomputable def import1OnlinePayload {backend : BackendContext} {model : HashModel}
    {params : W_Stage_online_1.Params}
    (ctx : W_Stage_online_1.generatedRoot.Ctx backend model params) :
    Fin 2 → ExactMatrix Q 8 1 1 :=
  W_Stage_online_1.generatedRoot.w_47_0 ctx

/-- Every generated label payload is exactly its advertised plaintext monomial. -/
theorem import1_payload_eq {params : W_Stage_import_1.Params}
    (ctx : W_Stage_import_1.generatedRoot.Ctx params) {outputs : _}
    (F : W_Stage_import_1.generatedRoot.Facts ctx outputs)
    (member : Fin 2) :
    import1Payload ctx member = fun _ _ ↦ reducePoly Q 8
      (message (import1Coefficient ctx member)
        (import1Exponent ctx member)) := by
  exact (congrArg (fun value ↦ value.2.2.1)
    (W_Stage_import_1.parallel_generatedRoot_6.sem (F.c0.h6_2 member))).trans
      (multiplyMonomial_liftInteger _ _)

/-- The online payload hypothesis follows from import wiring and plaintext agreement.
Plaintext agreement is explicit because the coefficient/exponent families are external inputs. -/
theorem import1_online_payload {backend : BackendContext} {model : HashModel}
    {params : W_Stage_import_1.Params} (ctx : W_Stage_import_1.generatedRoot.Ctx params)
    {outputs : _} (F : W_Stage_import_1.generatedRoot.Facts ctx outputs)
    {onlineParams : W_Stage_online_1.Params}
    (online : W_Stage_online_1.generatedRoot.Ctx backend model onlineParams)
    {values : Fin 2 → ErrorPoly 8}
    (hwire : ∀ member, import1OnlinePayload online member =
      import1Payload ctx member)
    (hmessage : ∀ member, message (import1Coefficient ctx member)
      (import1Exponent ctx member) = values member) :
    ∀ member, import1OnlinePayload online member =
      fun _ _ ↦ reducePoly Q 8 (values member) := by
  intro member
  rw [hwire member, import1_payload_eq ctx F member, hmessage member]

/-- The exact input tuple of `initial` in `W_Claim.Runs`; the run projection below
checks this transcription definitionally against the composed claim. -/
noncomputable def importClaimInitialInputs (external : GeneratedClaim.ExternalInputs)
    (execution : GeneratedClaim.Execution) :=
  (execution.«stage_3»,
    execution.«stage_2».1,
    external.input_2,
    external.input_0,
    execution.«stage_0»,
    external.input_37,
    ())

/-- The composed claim supplies the generated `initial` relation on its named inputs. -/
theorem importClaimInitialInputs_runs {model : HashModel} {external : GeneratedClaim.ExternalInputs}
    {execution : GeneratedClaim.Execution} (h : W_GeneratedClaim.Runs model external execution) :
    W_Stage_initial.generatedRoot Backend.backend model W_GeneratedClaim.stage_4_params
      (importClaimInitialInputs external execution) execution.«stage_4» :=
  h.2.2.2.2.2.1

/-- The exact input tuple of `import_0` in `W_Claim.Runs`; the run projection below
checks this transcription definitionally against the composed claim. -/
noncomputable def importClaim0Inputs (external : GeneratedClaim.ExternalInputs)
    (execution : GeneratedClaim.Execution) :=
  let stage4Tail24 := execution.«stage_4».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  (execution.«stage_4».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
    stage4Tail24.1,
    execution.«stage_4».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
    execution.«stage_4».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
    external.input_12,
    external.input_13,
    stage4Tail24.2.1,
    stage4Tail24.2.2.1,
    external.input_12,
    external.input_13,
    external.input_12,
    external.input_13,
    execution.«stage_4».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
    execution.«stage_4».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
    external.input_10,
    external.input_11,
    execution.«stage_4».2.1,
    execution.«stage_4».2.2.1,
    execution.«stage_4».2.2.2.1,
    execution.«stage_4».2.2.2.2.1,
    execution.«stage_4».2.2.2.2.2.1,
    execution.«stage_4».2.2.2.2.2.2.1,
    execution.«stage_4».2.2.2.2.2.2.2.1,
    execution.«stage_4».2.2.2.2.2.2.2.2.1,
    execution.«stage_4».2.2.2.2.2.2.2.2.2.1,
    execution.«stage_4».2.2.2.2.2.2.2.2.2.2.1,
    execution.«stage_4».2.2.2.2.2.2.2.2.2.2.2.1,
    execution.«stage_4».2.2.2.2.2.2.2.2.2.2.2.2.1,
    execution.«stage_4».2.2.2.2.2.2.2.2.2.2.2.2.2.1,
    execution.«stage_4».2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
    execution.«stage_4».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
    execution.«stage_4».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
    execution.«stage_4».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
    execution.«stage_4».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
    ())

/-- The composed claim supplies the generated `import_0` relation on its named inputs. -/
theorem importClaim0Inputs_runs {model : HashModel} {external : GeneratedClaim.ExternalInputs}
    {execution : GeneratedClaim.Execution} (h : W_GeneratedClaim.Runs model external execution) :
    W_Stage_import_0.generatedRoot W_GeneratedClaim.stage_10_params
      (importClaim0Inputs external execution) execution.«stage_10» :=
  h.2.2.2.2.2.2.2.2.2.2.2.1

/-- The exact input tuple of `online_0` in `W_Claim.Runs`; the run projection below
checks this transcription definitionally against the composed claim. -/
noncomputable def importClaimOnline0Inputs (external : GeneratedClaim.ExternalInputs)
    (execution : GeneratedClaim.Execution) :=
  let stage6Tail24 := execution.«stage_6».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  let stage6Tail48 := stage6Tail24.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  let stage6Tail72 := stage6Tail48.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  let stage6Tail96 := stage6Tail72.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  let stage6Tail120 := stage6Tail96.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  let stage10Tail24 := execution.«stage_10».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  let stage10Tail48 := stage10Tail24.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  let stage10Tail72 := stage10Tail48.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  (execution.«stage_4».1,
    execution.«stage_9».1,
    stage6Tail24.1,
    stage6Tail24.2.1,
    stage6Tail24.2.2.1,
    stage6Tail24.2.2.2.1,
    stage10Tail72.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
    external.input_2,
    stage10Tail72.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
    execution.«stage_6».2.2.2.2.2.2.2.2.2.2.2.2.1,
    execution.«stage_6».2.2.2.2.2.2.2.2.2.2.2.2.2.1,
    execution.«stage_6».2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
    execution.«stage_6».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
    stage10Tail48.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
    stage10Tail48.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
    external.input_27,
    execution.«stage_10».2.2.2.2.2.2.2.2.2.1,
    execution.«stage_10».2.2.2.2.2.2.2.2.2.2.1,
    external.input_26,
    execution.«stage_10».2.2.2.2.2.2.2.2.2.2.2.2.1,
    execution.«stage_10».2.2.2.2.2.2.2.2.2.2.2.1,
    execution.«stage_10».2.2.2.2.2.2.2.2.2.2.2.2.2.1,
    external.input_1,
    stage6Tail120.2.2.1,
    stage6Tail120.2.2.2.1,
    stage6Tail120.2.2.2.2.1,
    stage6Tail120.2.2.2.2.2.1,
    external.input_3,
    execution.«stage_2».2.1,
    execution.«stage_6».1,
    execution.«stage_6».2.1,
    execution.«stage_6».2.2.1,
    execution.«stage_6».2.2.2.1,
    stage10Tail48.2.2.2.2.2.2.1,
    stage10Tail48.2.2.2.2.2.2.2.1,
    execution.«stage_10».2.2.1,
    execution.«stage_10».2.2.2.1,
    execution.«stage_10».2.2.2.2.2.1,
    execution.«stage_10».2.2.2.2.1,
    execution.«stage_10».2.2.2.2.2.2.1,
    stage6Tail120.1,
    stage6Tail120.2.1,
    execution.«stage_10».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
    execution.«stage_10».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
    execution.«stage_10».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
    stage6Tail24.2.2.2.2.1,
    stage6Tail24.2.2.2.2.2.1,
    stage6Tail24.2.2.2.2.2.2.1,
    stage6Tail24.2.2.2.2.2.2.2.1,
    stage10Tail72.2.2.2.2.2.2.2.2.1,
    stage10Tail72.2.2.2.2.2.2.2.2.2.1,
    stage6Tail24.2.2.2.2.2.2.2.2.1,
    stage6Tail24.2.2.2.2.2.2.2.2.2.1,
    stage6Tail24.2.2.2.2.2.2.2.2.2.2.1,
    stage6Tail24.2.2.2.2.2.2.2.2.2.2.2.1,
    stage10Tail72.2.2.2.2.2.2.2.2.2.2.1,
    stage10Tail72.2.2.2.2.2.2.2.2.2.2.2.1,
    stage6Tail24.2.2.2.2.2.2.2.2.2.2.2.2.1,
    stage6Tail24.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
    stage6Tail24.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
    stage6Tail24.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
    stage10Tail72.2.2.2.2.2.2.2.2.2.2.2.2.1,
    stage10Tail72.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
    stage6Tail24.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
    stage6Tail24.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
    stage6Tail24.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
    stage6Tail24.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
    stage10Tail72.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
    stage10Tail72.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
    stage6Tail24.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
    stage6Tail24.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
    stage6Tail24.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
    stage6Tail24.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
    stage10Tail72.2.2.1,
    stage10Tail72.2.2.2.1,
    stage6Tail48.1,
    stage6Tail48.2.1,
    stage6Tail48.2.2.1,
    stage6Tail48.2.2.2.1,
    stage10Tail72.2.2.2.2.1,
    stage10Tail72.2.2.2.2.2.1,
    execution.«stage_6».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
    execution.«stage_6».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
    execution.«stage_6».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
    execution.«stage_6».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
    stage10Tail72.2.2.2.2.2.2.1,
    stage10Tail72.2.2.2.2.2.2.2.1,
    execution.«stage_6».2.2.2.2.2.2.2.2.1,
    execution.«stage_6».2.2.2.2.2.2.2.2.2.1,
    execution.«stage_6».2.2.2.2.2.2.2.2.2.2.1,
    execution.«stage_6».2.2.2.2.2.2.2.2.2.2.2.1,
    external.input_29,
    execution.«stage_10».2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
    execution.«stage_10».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
    external.input_28,
    execution.«stage_10».2.2.2.2.2.2.2.2.2.2.2.2.1,
    execution.«stage_10».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
    execution.«stage_10».2.2.2.2.2.2.2.2.2.2.2.2.2.1,
    stage6Tail120.2.2.2.2.2.2.1,
    stage6Tail120.2.2.2.2.2.2.2.1,
    stage6Tail120.2.2.2.2.2.2.2.2.1,
    stage6Tail120.2.2.2.2.2.2.2.2.2.1,
    execution.«stage_6».2.2.2.2.1,
    execution.«stage_6».2.2.2.2.2.1,
    execution.«stage_6».2.2.2.2.2.2.1,
    execution.«stage_6».2.2.2.2.2.2.2.1,
    external.input_25,
    external.input_24,
    execution.«stage_10».2.2.2.2.2.2.2.2.2.2.2.2.1,
    execution.«stage_10».2.2.2.2.2.2.2.2.2.2.2.2.2.1,
    stage6Tail96.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
    stage6Tail96.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
    stage6Tail96.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
    stage6Tail96.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
    execution.«stage_9».2.1,
    execution.«stage_9».2.2.1,
    execution.«stage_9».2.2.2.1,
    execution.«stage_9».2.2.2.2.1,
    ())

/-- The composed claim supplies the generated `online_0` relation on its named inputs. -/
theorem importClaimOnline0Inputs_runs {model : HashModel} {external : GeneratedClaim.ExternalInputs}
    {execution : GeneratedClaim.Execution} (h : W_GeneratedClaim.Runs model external execution) :
    W_Stage_online_0.generatedRoot Backend.backend model W_GeneratedClaim.stage_11_params
      (importClaimOnline0Inputs external execution) execution.«stage_11» :=
  h.2.2.2.2.2.2.2.2.2.2.2.2.1

/-- The exact input tuple of `import_1` in `W_Claim.Runs`; the run projection below
checks this transcription definitionally against the composed claim. -/
noncomputable def importClaim1Inputs (external : GeneratedClaim.ExternalInputs)
    (execution : GeneratedClaim.Execution) :=
  let stage11Tail24 := execution.«stage_11».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  let stage11Tail48 := stage11Tail24.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  let stage11Tail72 := stage11Tail48.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  let stage11Tail96 := stage11Tail72.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  let stage11Tail120 := stage11Tail96.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  (stage11Tail120.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
    stage11Tail120.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
    stage11Tail120.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
    stage11Tail120.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
    external.input_14,
    external.input_15,
    stage11Tail120.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
    stage11Tail120.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
    stage11Tail120.2.2.2.2.2.2.2.2.2.1,
    stage11Tail120.2.2.2.2.2.2.2.2.2.2.1,
    stage11Tail120.2.2.2.2.2.2.2.2.2.2.2.1,
    stage11Tail120.2.2.2.2.2.2.2.2.2.2.2.2.1,
    ())

/-- The composed claim supplies the generated `import_1` relation on its named inputs. -/
theorem importClaim1Inputs_runs {model : HashModel} {external : GeneratedClaim.ExternalInputs}
    {execution : GeneratedClaim.Execution} (h : W_GeneratedClaim.Runs model external execution) :
    W_Stage_import_1.generatedRoot W_GeneratedClaim.stage_16_params
      (importClaim1Inputs external execution) execution.«stage_16» :=
  h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

/-- The exact input tuple of `online_1` in `W_Claim.Runs`; the run projection below
checks this transcription definitionally against the composed claim. -/
noncomputable def importClaimOnline1Inputs (external : GeneratedClaim.ExternalInputs)
    (execution : GeneratedClaim.Execution) :=
  (execution.«stage_11».1,
    execution.«stage_15»,
    execution.«stage_12».1,
    execution.«stage_12».2.1,
    execution.«stage_12».2.2.1,
    execution.«stage_12».2.2.2.1,
    execution.«stage_16».2.2.2.2.2.2.2.2.2.2.2.2.2.1,
    external.input_3,
    execution.«stage_16».2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
    execution.«stage_16».2.2.1,
    execution.«stage_16».2.2.2.1,
    execution.«stage_16».2.2.2.2.2.1,
    execution.«stage_16».2.2.2.2.1,
    execution.«stage_16».2.2.2.2.2.2.1,
    external.input_1,
    execution.«stage_12».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
    execution.«stage_12».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
    execution.«stage_12».2.2.2.2.1,
    execution.«stage_12».2.2.2.2.2.1,
    execution.«stage_12».2.2.2.2.2.2.1,
    execution.«stage_12».2.2.2.2.2.2.2.1,
    execution.«stage_16».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
    execution.«stage_16».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
    external.input_31,
    external.input_30,
    execution.«stage_16».1,
    execution.«stage_16».2.1,
    execution.«stage_12».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
    execution.«stage_12».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
    execution.«stage_12».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
    execution.«stage_12».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
    ())

/-- The composed claim supplies the generated `online_1` relation on its named inputs. -/
theorem importClaimOnline1Inputs_runs {model : HashModel} {external : GeneratedClaim.ExternalInputs}
    {execution : GeneratedClaim.Execution} (h : W_GeneratedClaim.Runs model external execution) :
    W_Stage_online_1.generatedRoot Backend.backend model W_GeneratedClaim.stage_17_params
      (importClaimOnline1Inputs external execution) execution.«stage_17» :=
  h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

/-- The initial and online wiring equalities for step zero follow from the actual claim
tuples and generated output equations. Only whole-stage input equalities remain for callers;
these are reflexive when contexts are obtained using the three `*_runs` projections above. -/
theorem import0_claim_wiring {backend : BackendContext} {model : HashModel}
    {initialParams : W_Stage_initial.Params}
    (initial : W_Stage_initial.generatedRoot.Ctx backend model initialParams)
    {params : W_Stage_import_0.Params} (ctx : W_Stage_import_0.generatedRoot.Ctx params)
    {onlineParams : W_Stage_online_0.Params}
    (online : W_Stage_online_0.generatedRoot.Ctx backend model onlineParams)
    (external : GeneratedClaim.ExternalInputs) (execution : GeneratedClaim.Execution)
    (Fi : W_Stage_initial.generatedRoot.Facts initial execution.«stage_4»)
    (F : W_Stage_import_0.generatedRoot.Facts ctx execution.«stage_10»)
    (hinitial : initial.inputs = importClaimInitialInputs external execution)
    (himport : ctx.inputs = importClaim0Inputs external execution)
    (honline : online.inputs = importClaimOnline0Inputs external execution) :
    (∀ (port : Import0Port) (member : Fin port.members),
      import0InputRows ctx port member = initialImportRows initial port.root member) ∧
    (∀ (port : Import0Port) (member : Fin port.members),
      import0OnlineRows online port member = import0OutputRows ctx port member) ∧
    W_Stage_online_0.generatedRoot.w_7_0 online = initial.inputs.2.2.1 := by
  refine ⟨?_, ?_, ?_⟩
  · intro port member
    cases port <;>
      simp only [
        import0InputRows,
        Import0Port.root,
        initialImportRows,
        W_Stage_import_0.generatedRoot.w_0_0,
        W_Stage_import_0.generatedRoot.w_1_0,
        W_Stage_import_0.generatedRoot.w_2_0,
        W_Stage_import_0.generatedRoot.w_3_0,
        W_Stage_import_0.generatedRoot.w_7_0,
        W_Stage_import_0.generatedRoot.w_8_0,
        W_Stage_import_0.generatedRoot.w_15_0,
        W_Stage_import_0.generatedRoot.w_16_0,
        W_Stage_import_0.generatedRoot.w_20_0,
        W_Stage_import_0.generatedRoot.w_21_0,
        W_Stage_import_0.generatedRoot.w_23_0,
        W_Stage_import_0.generatedRoot.w_24_0,
        W_Stage_import_0.generatedRoot.w_26_0,
        W_Stage_import_0.generatedRoot.w_27_0,
        W_Stage_import_0.generatedRoot.w_29_0,
        W_Stage_import_0.generatedRoot.w_30_0,
        W_Stage_import_0.generatedRoot.w_32_0,
        W_Stage_import_0.generatedRoot.w_33_0,
        W_Stage_import_0.generatedRoot.w_35_0,
        W_Stage_import_0.generatedRoot.w_36_0,
        W_Stage_import_0.generatedRoot.w_38_0,
        W_Stage_import_0.generatedRoot.w_39_0,
        W_Stage_import_0.generatedRoot.w_41_0,
        W_Stage_import_0.generatedRoot.w_42_0,
        W_Stage_import_0.generatedRoot.w_44_0,
        W_Stage_import_0.generatedRoot.w_45_0,
        himport,
        importClaim0Inputs,
        Fi.c5.hout,
        W_Stage_initial.generatedRoot.outs]
  · intro port member
    cases port <;>
      simp only [
        import0OnlineRows,
        import0OutputRows,
        W_Stage_online_0.generatedRoot.w_6_0,
        W_Stage_online_0.generatedRoot.w_8_0,
        W_Stage_online_0.generatedRoot.w_13_0,
        W_Stage_online_0.generatedRoot.w_14_0,
        W_Stage_online_0.generatedRoot.w_16_0,
        W_Stage_online_0.generatedRoot.w_17_0,
        W_Stage_online_0.generatedRoot.w_21_0,
        W_Stage_online_0.generatedRoot.w_25_0,
        W_Stage_online_0.generatedRoot.w_255_0,
        W_Stage_online_0.generatedRoot.w_256_0,
        W_Stage_online_0.generatedRoot.w_257_0,
        W_Stage_online_0.generatedRoot.w_262_0,
        W_Stage_online_0.generatedRoot.w_275_0,
        W_Stage_online_0.generatedRoot.w_290_0,
        W_Stage_online_0.generatedRoot.w_372_0,
        W_Stage_online_0.generatedRoot.w_377_0,
        W_Stage_online_0.generatedRoot.w_764_0,
        W_Stage_online_0.generatedRoot.w_765_0,
        W_Stage_online_0.generatedRoot.w_822_0,
        W_Stage_online_0.generatedRoot.w_823_0,
        W_Stage_online_0.generatedRoot.w_880_0,
        W_Stage_online_0.generatedRoot.w_881_0,
        W_Stage_online_0.generatedRoot.w_938_0,
        W_Stage_online_0.generatedRoot.w_939_0,
        W_Stage_online_0.generatedRoot.w_996_0,
        W_Stage_online_0.generatedRoot.w_997_0,
        W_Stage_online_0.generatedRoot.w_1054_0,
        W_Stage_online_0.generatedRoot.w_1055_0,
        W_Stage_online_0.generatedRoot.w_1112_0,
        W_Stage_online_0.generatedRoot.w_1113_0,
        W_Stage_online_0.generatedRoot.w_1169_0,
        W_Stage_online_0.generatedRoot.w_1170_0,
        W_Stage_online_0.generatedRoot.w_1174_0,
        W_Stage_online_0.generatedRoot.w_1178_0,
        W_Stage_online_0.generatedRoot.w_1229_0,
        W_Stage_online_0.generatedRoot.w_1232_0,
        honline,
        importClaimOnline0Inputs,
        F.c3.hout,
        W_Stage_import_0.generatedRoot.outs]
  ·
    simp only [
      W_Stage_online_0.generatedRoot.w_7_0,
      honline,
      importClaimOnline0Inputs,
      hinitial,
      importClaimInitialInputs]

/-- The exact step-one online wiring preserves the preceding step's root rows. -/
theorem import1_claim_wiring {backend : BackendContext} {model : HashModel}
    {params : W_Stage_import_1.Params} (ctx : W_Stage_import_1.generatedRoot.Ctx params)
    {onlineParams : W_Stage_online_1.Params}
    (online : W_Stage_online_1.generatedRoot.Ctx backend model onlineParams)
    (external : GeneratedClaim.ExternalInputs) (execution : GeneratedClaim.Execution)
    (F : W_Stage_import_1.generatedRoot.Facts ctx execution.«stage_16»)
    (honline : online.inputs = importClaimOnline1Inputs external execution) :
    ∀ (port : Import1Port) (member : Fin port.members),
      import1OnlineRows online port member = import1OutputRows ctx port member := by
  intro port member
  cases port <;>
    simp only [
      import1OnlineRows,
      import1OutputRows,
      W_Stage_online_1.generatedRoot.w_13_0,
      W_Stage_online_1.generatedRoot.w_15_0,
      W_Stage_online_1.generatedRoot.w_16_0,
      W_Stage_online_1.generatedRoot.w_27_0,
      W_Stage_online_1.generatedRoot.w_46_0,
      W_Stage_online_1.generatedRoot.w_63_0,
      W_Stage_online_1.generatedRoot.w_136_0,
      W_Stage_online_1.generatedRoot.w_137_0,
      W_Stage_online_1.generatedRoot.w_142_0,
      W_Stage_online_1.generatedRoot.w_145_0,
      honline,
      importClaimOnline1Inputs,
      F.c0.hout,
      W_Stage_import_1.generatedRoot.outs]

/-- The payload connection to online step 0 is exactly the one in the composed claim. -/
theorem import0_claim_payload_wiring {backend : BackendContext} {model : HashModel}
    {params : W_Stage_import_0.Params} (ctx : W_Stage_import_0.generatedRoot.Ctx params)
    {onlineParams : W_Stage_online_0.Params}
    (online : W_Stage_online_0.generatedRoot.Ctx backend model onlineParams)
    (external : GeneratedClaim.ExternalInputs) (execution : GeneratedClaim.Execution)
    (F : W_Stage_import_0.generatedRoot.Facts ctx execution.«stage_10»)
    (honline : online.inputs = importClaimOnline0Inputs external execution)
    (label : Import0Label) (member : Fin label.members) :
    import0OnlinePayload online label member = import0Payload ctx label member := by
  cases label <;>
    simp only [
      import0OnlinePayload,
      import0Payload,
      W_Stage_online_0.generatedRoot.w_276_0,
      W_Stage_online_0.generatedRoot.w_22_0,
      W_Stage_online_0.generatedRoot.w_1175_0,
      W_Stage_online_0.generatedRoot.w_391_0,
      honline,
      importClaimOnline0Inputs,
      F.c3.hout,
      W_Stage_import_0.generatedRoot.outs]

/-- The payload connection to online step 1 is exactly the one in the composed claim. -/
theorem import1_claim_payload_wiring {backend : BackendContext} {model : HashModel}
    {params : W_Stage_import_1.Params} (ctx : W_Stage_import_1.generatedRoot.Ctx params)
    {onlineParams : W_Stage_online_1.Params}
    (online : W_Stage_online_1.generatedRoot.Ctx backend model onlineParams)
    (external : GeneratedClaim.ExternalInputs) (execution : GeneratedClaim.Execution)
    (F : W_Stage_import_1.generatedRoot.Facts ctx execution.«stage_16»)
    (honline : online.inputs = importClaimOnline1Inputs external execution)
    (member : Fin 2) :
    import1OnlinePayload online member = import1Payload ctx member := by
  simp only [
    import1OnlinePayload,
    import1Payload,
    W_Stage_online_1.generatedRoot.w_47_0,
    honline,
    importClaimOnline1Inputs,
    F.c0.hout,
    W_Stage_import_1.generatedRoot.outs]

/-- With the composed claim's concrete wiring, every step-zero imported online row satisfies
its initial closed-form relation with bound one. This discharges the imported `hlab`, `ha`,
`hone`, `hs`, `hpriv`, and `hhelp` premises by selecting their named port and family member. -/
theorem import0_claim_online_rel {backend : BackendContext} {model : HashModel}
    {initialParams : W_Stage_initial.Params}
    (initial : W_Stage_initial.generatedRoot.Ctx backend model initialParams)
    {params : W_Stage_import_0.Params} (ctx : W_Stage_import_0.generatedRoot.Ctx params)
    {onlineParams : W_Stage_online_0.Params}
    (online : W_Stage_online_0.generatedRoot.Ctx backend model onlineParams)
    (external : GeneratedClaim.ExternalInputs) (execution : GeneratedClaim.Execution)
    (hbody : W_Stage_initial.generatedRoot.body backend model initialParams initial.inputs
      execution.«stage_4» initial.witness)
    (F : W_Stage_import_0.generatedRoot.Facts ctx execution.«stage_10»)
    (hinitial : initial.inputs = importClaimInitialInputs external execution)
    (himport : ctx.inputs = importClaim0Inputs external execution)
    (honline : online.inputs = importClaimOnline0Inputs external execution)
    {sm tp : ErrorPoly 8} (hmask : initial.inputs.1 0 0 = reducePoly Q 8 sm)
    (hpay : initial.inputs.2.2.2.2.1 0 0 = reducePoly Q 8 tp)
    {G : Row} (hG : gadgetMatrixRuns backend 4096 25 G)
    (port : Import0Port) (member : Fin port.members) :
    Rel sm (keyPairG model (W_Stage_online_0.generatedRoot.w_7_0 online) Q 25
      port.root.prefix (port.root.tags member))
      (fun component ↦ reducePoly Q 8
        (initialImportValue initial port.root member *
          unitPair (initialImportMask sm tp port.root member) component) • G)
      (import0OnlineRows online port member) 1 := by
  have Fi := W_Stage_initial.generatedRoot.facts (ctx := initial) hbody
  obtain ⟨hinitialRows, honlineRows, hkey⟩ :=
    import0_claim_wiring initial ctx online external execution Fi F hinitial himport honline
  exact import0_online_rel initial hbody ctx F online hinitialRows honlineRows hkey
    hmask hpay hG port member

/-- The actual composed step-one import preserves every preceding root relation, including
its explicit bound; it does not replace the preceding step's rows with initial rows. -/
theorem import1_claim_online_rel {backend : BackendContext} {model : HashModel}
    {params : W_Stage_import_1.Params} (ctx : W_Stage_import_1.generatedRoot.Ctx params)
    {onlineParams : W_Stage_online_1.Params}
    (online : W_Stage_online_1.generatedRoot.Ctx backend model onlineParams)
    (external : GeneratedClaim.ExternalInputs) (execution : GeneratedClaim.Execution)
    (F : W_Stage_import_1.generatedRoot.Facts ctx execution.«stage_16»)
    (honline : online.inputs = importClaimOnline1Inputs external execution)
    {sm : ErrorPoly 8}
    (targets : (port : Import1Port) → Fin port.members → Fin 2 → Row)
    (bounds : (port : Import1Port) → Fin port.members → Nat)
    (hroot : ∀ (port : Import1Port) (member : Fin port.members),
      Rel sm (keyPairG model (W_Stage_online_1.generatedRoot.w_14_0 online) Q 25
        port.root.prefix (port.root.tags member))
        (targets port member) (import1InputRows ctx port member) (bounds port member)) :
    ∀ (port : Import1Port) (member : Fin port.members),
      Rel sm (keyPairG model (W_Stage_online_1.generatedRoot.w_14_0 online) Q 25
        port.root.prefix (port.root.tags member))
        (targets port member) (import1OnlineRows online port member) (bounds port member) := by
  exact import1_online_from_rows ctx F online
    (import1_claim_wiring ctx online external execution F honline) targets bounds hroot

end Dio2Proof
