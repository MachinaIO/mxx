import Dio2Inv
import Dio2Online0Row
import Dio2Import

/-!
# The transition from the refreshed parts to the next step input

`step_refresh_rel` packages the conclusion of `online0_part_row`: a part is
`β A'₀ + A'₁ - T G + E`, so its next mask is `(β, 1)`, its key is `A'`, and its
error is precisely `E`. Old target-row errors have already been cancelled by the refresh
preimage and decoder; they are not added a second time (DIO2-SPEC §§6.7, 6.8, 15).

There are two different families in `crates/io/src/dio2/refresh.rs`:
the encoded target rows, with forms supplied by `Dio2Online0Refresh`, `Dio2Online0Label`, and
`Dio2Online0Helper`, and the new root-key columns `a_primes`. The graph packs the latter in
`w_227`, and returns the refreshed parts in `w_1281`. In particular, `w_227` is not the family
of encoded targets. The `output-k.j` exports are internal Program encodings, not the next
step input: the latter is exported under the `rns-bgg/root-row/…`, `rns-bgg/root-helper/…`,
and `dio2/hint` names. `step_expand_rel` separately proves what substituting noisy row forms
into `A'` would give: explicit combined keys, target `β x₀ + x₁ + T`, and error
`β e₀ + e₁ + E`. Identifying that result with a prescribed root needs key and target
equalities; it does not follow just from the noisy forms.

`online0_step_transition` proves `Rel` for every paired part, including anchor blocks, without
unrolling the 36-part family. `online0_import_rows_eq` identifies every actual root export,
including the repackaged helper and label families. `online0_import1_rel` supplies exactly
the root-relation premise of `import1_online_from_rows`; `online0_online1_rel` carries it to
the five online ports. The new-root key agreement, refreshed target values, decoder success,
and claim wiring remain explicit hypotheses. No `StepInv` declaration exists in
`Dio2Inv.lean`; its available invariant is `RootRow`, whose bound is fixed to one.
-/

set_option maxHeartbeats 8000000
set_option maxRecDepth 16384
set_option autoImplicit false

namespace Dio2Proof

open Mxx.Primitives MxxRuntime MxxExponentLut RefreshTopProof.Gen
open Mxx.ExponentLut.Refresh
open RefreshTopProof GenDecode

section Algebra

variable {n q m : Nat}

/-- The exact output form of a refresh part gives the next-mask relation, with only fresh error. -/
theorem step_refresh_rel {ι : Type} {next : ErrorPoly n}
    {keys : ι → Fin 2 → Fin 2 → ExactMatrix q n 1 m}
    {target : ι → Fin 2 → ExactPoly q n} {gadget : ExactMatrix q n 1 m}
    {rows : ι → Fin 2 → ExactMatrix q n 1 m}
    {fresh : ι → Fin 2 → ErrorMatrix n 1 m} {bound : ι → Nat}
    (hfresh : ∀ index component, CoeffBound (fresh index component) (bound index))
    (hrow : ∀ index component, rows index component =
      reducePoly q n next • keys index 0 component + keys index 1 component -
        target index component • gadget + reduceMatrix q n 1 m (fresh index component)) :
    ∀ index, Rel next (keys index) (fun component ↦ target index component • gadget)
      (rows index) (bound index) := by
  intro index
  exact ⟨fresh index, hfresh index, hrow index⟩

/-- Equality with the next tag's hashed key and a unit fresh bound give the actual `RootRow`. -/
theorem step_refresh_root {ι : Type} {model : HashModel} {key : ByteArray}
    {pre : ι → Blob} {tags : ι → List HashTagComponent} {next : ErrorPoly n}
    {keys : ι → Fin 2 → Fin 2 → ExactMatrix q n 1 m}
    {target : ι → Fin 2 → ExactMatrix q n 1 1} {gadget : ExactMatrix q n 1 m}
    {rows : ι → Fin 2 → ExactMatrix q n 1 m}
    {fresh : ι → Fin 2 → ErrorMatrix n 1 m}
    (hkey : ∀ index, keys index = rootKey model key (pre index) (tags index) q m)
    (hfresh : ∀ index component, CoeffBound (fresh index component) 1)
    (hrow : ∀ index component, rows index component =
      reducePoly q n next • keys index 0 component + keys index 1 component -
        matrixMulScalarLeft (target index component) gadget +
        reduceMatrix q n 1 m (fresh index component)) :
    ∀ index, RootRow model key (pre index) (tags index) next gadget
      (target index) (rows index) := by
  intro index
  unfold RootRow
  rw [← hkey index]
  exact ⟨fresh index, hfresh index, hrow index⟩

/-- Substituting two noisy mask forms gives explicit combined keys and all three error pieces. -/
theorem step_expand_rel (hn : 0 < n) {ι : Type} {old next : ErrorPoly n}
    {maskBound : Nat} (hnext : polyNorm next ≤ maskBound)
    {pieces : ι → Fin 2 → Fin 2 → ExactMatrix q n 1 m}
    {keys : ι → Fin 2 → Fin 2 → Fin 2 → ExactMatrix q n 1 m}
    {values : ι → Fin 2 → Fin 2 → ExactPoly q n}
    {gadget : ExactMatrix q n 1 m} {target : ι → Fin 2 → ExactPoly q n}
    {errors : ι → Fin 2 → Fin 2 → ErrorMatrix n 1 m}
    {fresh : ι → Fin 2 → ErrorMatrix n 1 m}
    {pieceBounds : ι → Fin 2 → Nat} {freshBound : ι → Nat}
    (herrors : ∀ index block component,
      CoeffBound (errors index block component) (pieceBounds index block))
    (hfresh : ∀ index component, CoeffBound (fresh index component) (freshBound index))
    (hpieces : ∀ index block component, pieces index block component =
      maskPart (reducePoly q n old) (fun side ↦ keys index block side component) -
        values index block component • gadget +
        reduceMatrix q n 1 m (errors index block component))
    {rows : ι → Fin 2 → ExactMatrix q n 1 m}
    (hrow : ∀ index component, rows index component =
      reducePoly q n next • pieces index 0 component + pieces index 1 component -
        target index component • gadget + reduceMatrix q n 1 m (fresh index component)) :
    ∀ index, Rel next
      (fun block component ↦
        maskPart (reducePoly q n old) (fun side ↦ keys index block side component))
      (fun component ↦ (reducePoly q n next * values index 0 component +
        values index 1 component + target index component) • gadget)
      (rows index)
      (n * maskBound * pieceBounds index 0 + pieceBounds index 1 + freshBound index) := by
  intro index
  refine ⟨fun component ↦ next • errors index 0 component + errors index 1 component +
    fresh index component, fun component ↦ coeffBound_add
      (coeffBound_add (coeffBound_smul hn hnext (herrors index 0 component))
        (herrors index 1 component)) (hfresh index component), fun component ↦ ?_⟩
  rw [hrow index component, hpieces index 0 component, hpieces index 1 component]
  simp only [reduceMatrix_add, reduceMatrix_smul, smul_add, smul_sub, add_smul, smul_smul]
  abel

end Algebra

namespace StepTransition

open W_Stage_online_0

local notation "Q" => W_Stage_online_0.modulus_0
local notation "Row" => ExactMatrix Q 8 1 25

/-- Part `(target, component)` occupies position `2 * target + component` in the 36-part loop. -/
def partIndex (target : Fin 18) (component : Fin 2) : Fin 36 :=
  ⟨2 * target.val + component.val, by omega⟩

/-- The explicit four key blocks of a next row are the two rows of each packed `A'` column. -/
noncomputable def nextKeys {backend : BackendContext} {model : HashModel} {params : Params}
    (ctx : generatedRoot.Ctx backend model params) (target : Fin 18) :
    Fin 2 → Fin 2 → Row :=
  fun side component ↦ rowOf (generatedRoot.w_227_0 ctx (partIndex target component)) side

/-- Pair the CRT sums in their part order; targets 16 and 17 are next-anchor blocks. -/
noncomputable def nextRows {backend : BackendContext} {model : HashModel} {params : Params}
    (ctx : generatedRoot.Ctx backend model params) (target : Fin 18) : Fin 2 → Row :=
  fun component ↦ generatedRoot.w_1281_0 ctx (partIndex target component)

/-- The actual input tuple passed to one member of the generated refresh-part loop. -/
noncomputable def partInputs {backend : BackendContext} {model : HashModel} {params : Params}
    (ctx : generatedRoot.Ctx backend model params) (index : Fin 36) :=
  (generatedRoot.w_0_0 ctx, (generatedRoot.w_1_0 ctx index,
    (generatedRoot.w_63_0 ctx, (generatedRoot.w_70_0 ctx,
    (generatedRoot.w_227_0 ctx index, (generatedRoot.w_1162_0 ctx index,
    (generatedRoot.w_1219_5 ctx, (generatedRoot.w_57_0 ctx,
    (generatedRoot.w_1273_5 ctx, (generatedRoot.w_1274_0 ctx,
    (generatedRoot.w_1276_0 ctx, (generatedRoot.w_1277_0 ctx index,
    (generatedRoot.w_1278_0 ctx index, (generatedRoot.w_1279_0 ctx index,
    (generatedRoot.w_1280_0 ctx index, ())))))))))))))))

/-- The target number supplying each exported root member, including aliases in helper layouts. -/
def rootTarget : (root : ImportRoot) → Fin root.members → Fin 18
  | .label0, member => if member.val = 0 then 3 else 1
  | .label, member => ⟨4 + member.val, by have := member.isLt; change _ < 4 at this; omega⟩
  | .one, _ => 0
  | .state, _ => 2
  | .aut3, _ => 9
  | .aut5, _ => 10
  | .aut7, _ => 11
  | .aut9, _ => 12
  | .aut11, _ => 13
  | .aut13, _ => 14
  | .aut15, _ => 15
  | .exp8, member => ⟨8 + member.val, by have := member.isLt; change _ < 8 at this; omega⟩
  | .exp4, member => ![8, 9, 10, 15] member

/-- The named root outputs of `online_0`, exactly as consumed by `import_1` and `W_Claim`. -/
noncomputable def rootRows {backend : BackendContext} {model : HashModel} {params : Params}
    (ctx : generatedRoot.Ctx backend model params) :
    (root : ImportRoot) → Fin root.members → Fin 2 → Row
  | .label0, member => ![generatedRoot.w_1532_0 ctx member, generatedRoot.w_1535_0 ctx member]
  | .label, member => ![generatedRoot.w_1540_0 ctx member, generatedRoot.w_1545_0 ctx member]
  | .one, _ => ![generatedRoot.w_1546_0 ctx, generatedRoot.w_1547_0 ctx]
  | .state, _ => ![generatedRoot.w_1548_0 ctx, generatedRoot.w_1549_0 ctx]
  | .aut3, member => ![generatedRoot.w_1487_0 ctx member, generatedRoot.w_1489_0 ctx member]
  | .aut5, member => ![generatedRoot.w_1491_0 ctx member, generatedRoot.w_1493_0 ctx member]
  | .aut7, member => ![generatedRoot.w_1495_0 ctx member, generatedRoot.w_1497_0 ctx member]
  | .aut9, member => ![generatedRoot.w_1499_0 ctx member, generatedRoot.w_1501_0 ctx member]
  | .aut11, member => ![generatedRoot.w_1475_0 ctx member, generatedRoot.w_1477_0 ctx member]
  | .aut13, member => ![generatedRoot.w_1479_0 ctx member, generatedRoot.w_1481_0 ctx member]
  | .aut15, member => ![generatedRoot.w_1483_0 ctx member, generatedRoot.w_1485_0 ctx member]
  | .exp8, member => ![generatedRoot.w_1510_0 ctx member, generatedRoot.w_1519_0 ctx member]
  | .exp4, member => ![generatedRoot.w_1524_0 ctx member, generatedRoot.w_1529_0 ctx member]

/-- The next root's payload: private and state retain `tp`, helpers use `σ_k(next)`. -/
noncomputable def rootPayload (next tp : ErrorPoly 8) (root : ImportRoot)
    (member : Fin root.members) :
    ErrorPoly 8 :=
  match root with
  | .one | .state => tp
  | .label0 | .label => next
  | root => substitution (root.sigma member) (root.sigma_odd member) next

/-- The semantic next values: refreshed labels and state, with one for the private unit/helpers. -/
noncomputable def rootValue (labels : Fin 5 → ErrorPoly 8) (state : ErrorPoly 8) :
    (root : ImportRoot) → Fin root.members → ErrorPoly 8
  | .label0, member => if member.val = 0 then labels 0 else 1
  | .label, member => labels ⟨member.val + 1, by
      have := member.isLt
      change _ < 4 at this
      omega⟩
  | .state, _ => state
  | _, _ => 1

end StepTransition

open StepTransition W_Stage_online_0

local notation "Q" => W_Stage_online_0.modulus_0
local notation "Row" => ExactMatrix Q 8 1 25

/-- The five `online0_part_row` term hypotheses for each part imply the whole output family. -/
theorem online0_part_family {backend : BackendContext} {model : HashModel} {params : Params}
    (ctx : generatedRoot.Ctx backend model params) {outputs : _}
    (F : generatedRoot.Facts ctx outputs) {value : Fin 36 → Row}
    (hterms : ∀ (index : Fin 36)
      (part : parallel_generatedRoot_1281.Ctx backend model params index.val),
      part.inputs = partInputs ctx index →
      parallel_generatedRoot_1281.Facts part (generatedRoot.w_1281_0 ctx index) →
      TermOf modulus_1 (crt0 0) (parallel_generatedRoot_1281.w_64_0 part) (value index) ∧
      TermOf modulus_2 (crt0 1) (parallel_generatedRoot_1281.w_119_0 part) (value index) ∧
      TermOf modulus_3 (crt0 2) (parallel_generatedRoot_1281.w_175_0 part) (value index) ∧
      TermOf modulus_4 (crt0 3) (parallel_generatedRoot_1281.w_231_0 part) (value index) ∧
      TermOf modulus_5 (crt0 4) (parallel_generatedRoot_1281.w_287_0 part) (value index)) :
    ∀ index : Fin 36, generatedRoot.w_1281_0 ctx index = value index := by
  intro index
  obtain ⟨witness, hbody⟩ := F.c49.h1281_0 index
  let part : parallel_generatedRoot_1281.Ctx backend model params index.val :=
    ⟨partInputs ctx index, witness⟩
  have facts := parallel_generatedRoot_1281.facts part hbody
  obtain ⟨h0, h1, h2, h3, h4⟩ := hterms index part rfl facts
  exact facts.c14.hout.trans (online0_part_row part h0 h1 h2 h3 h4)

/-- All 36 successful part rows give the next-mask invariant for all 18 targets at once. -/
theorem online0_step_transition {backend : BackendContext} {model : HashModel} {params : Params}
    (ctx : generatedRoot.Ctx backend model params) {next : ErrorPoly 8}
    {gadget : Row} {target : Fin 36 → ExactPoly Q 8}
    {fresh : Fin 36 → ErrorMatrix 8 1 25} {bound : Nat}
    (hfresh : ∀ index, CoeffBound (fresh index) bound)
    (hparts : ∀ index : Fin 36, generatedRoot.w_1281_0 ctx index =
      reducePoly Q 8 next • rowOf (generatedRoot.w_227_0 ctx index) 0 +
        rowOf (generatedRoot.w_227_0 ctx index) 1 - target index • gadget +
        reduceMatrix Q 8 1 25 (fresh index)) :
    ∀ row : Fin 18, Rel next (nextKeys ctx row)
      (fun component ↦ target (partIndex row component) • gadget) (nextRows ctx row) bound := by
  exact step_refresh_rel (fun row component ↦ hfresh (partIndex row component))
    (fun row component ↦ hparts (partIndex row component))

/-- A generated family lookup at a valid natural index is the corresponding function value. -/
private theorem step_get {α : Type} {size : Nat} {family : Fin size → α} {value : α}
    (index : Fin size) (h : familyGetStatic family index.val value) : value = family index := by
  obtain ⟨position, hposition, hvalue⟩ := h
  have heq : position = index := Fin.ext (by omega)
  simpa only [heq] using hvalue

/-- The named root exports are exactly paired CRT sums, including family packaging. -/
theorem online0_import_rows_eq {backend : BackendContext} {model : HashModel} {params : Params}
    (ctx : generatedRoot.Ctx backend model params) {outputs : _}
    (F : generatedRoot.Facts ctx outputs) :
    ∀ (root : ImportRoot) (member : Fin root.members),
      rootRows ctx root member = nextRows ctx (rootTarget root member) := by
  have hget := @step_get Row 36 (generatedRoot.w_1281_0 ctx)
  have hpack := @import_familyPack_get Row
  intro root member
  funext component
  cases root <;> fin_cases component
  all_goals simp only [rootRows, nextRows, rootTarget, partIndex, Fin.isValue]
  all_goals first
    | exact hget 0 F.c70.h1546_0
    | exact hget 1 F.c70.h1547_0
    | exact hget 4 F.c70.h1548_0
    | exact hget 5 F.c70.h1549_0
    | skip
  all_goals fin_cases member
  all_goals simp only [Fin.isValue, Fin.reduceFinMk, Matrix.cons_val_zero, Matrix.cons_val_one,
    ↓reduceIte, Nat.reduceAdd, Nat.reduceMul]
  all_goals simp only [hpack F.c68.h1475_0, hpack F.c68.h1477_0,
    hpack F.c68.h1479_0, hpack F.c68.h1481_0, hpack F.c68.h1483_0,
    hpack F.c68.h1485_0, hpack F.c68.h1487_0, hpack F.c68.h1489_0,
    hpack F.c68.h1491_0, hpack F.c69.h1493_0, hpack F.c69.h1495_0,
    hpack F.c69.h1497_0, hpack F.c69.h1499_0, hpack F.c69.h1501_0,
    hpack F.c70.h1532_0, hpack F.c70.h1535_0, hpack F.c70.h1540_0,
    hpack F.c70.h1545_0, hpack F.c69.h1510_0, hpack F.c69.h1519_0,
    hpack F.c69.h1524_0, hpack F.c70.h1529_0,
    List.getElem_cons_zero, List.getElem_cons_succ]
  all_goals first
    | exact hget _ F.c68.h1474_0 | exact hget _ F.c68.h1476_0
    | exact hget _ F.c68.h1478_0 | exact hget _ F.c68.h1480_0
    | exact hget _ F.c68.h1482_0 | exact hget _ F.c68.h1484_0
    | exact hget _ F.c68.h1486_0 | exact hget _ F.c68.h1488_0
    | exact hget _ F.c68.h1490_0 | exact hget _ F.c68.h1492_0
    | exact hget _ F.c69.h1494_0 | exact hget _ F.c69.h1496_0
    | exact hget _ F.c69.h1498_0 | exact hget _ F.c69.h1500_0
    | exact hget _ F.c69.h1502_0 | exact hget _ F.c69.h1503_0
    | exact hget _ F.c69.h1504_0 | exact hget _ F.c69.h1505_0
    | exact hget _ F.c69.h1506_0 | exact hget _ F.c69.h1507_0
    | exact hget _ F.c69.h1508_0 | exact hget _ F.c69.h1509_0
    | exact hget _ F.c69.h1511_0 | exact hget _ F.c69.h1512_0
    | exact hget _ F.c69.h1513_0 | exact hget _ F.c69.h1514_0
    | exact hget _ F.c69.h1515_0 | exact hget _ F.c69.h1516_0
    | exact hget _ F.c69.h1517_0 | exact hget _ F.c69.h1518_0
    | exact hget _ F.c69.h1520_0 | exact hget _ F.c69.h1521_0
    | exact hget _ F.c69.h1522_0 | exact hget _ F.c69.h1523_0
    | exact hget _ F.c70.h1525_0 | exact hget _ F.c70.h1526_0
    | exact hget _ F.c70.h1527_0 | exact hget _ F.c70.h1528_0
    | exact hget _ F.c70.h1530_0 | exact hget _ F.c70.h1531_0
    | exact hget _ F.c70.h1533_0 | exact hget _ F.c70.h1534_0
    | exact hget _ F.c70.h1536_0 | exact hget _ F.c70.h1537_0
    | exact hget _ F.c70.h1538_0 | exact hget _ F.c70.h1539_0
    | exact hget _ F.c70.h1541_0 | exact hget _ F.c70.h1542_0
    | exact hget _ F.c70.h1543_0 | exact hget _ F.c70.h1544_0

/-- Every exported root has its specified new target and hashed key, retaining the fresh bound. -/
theorem online0_root_rel {backend : BackendContext} {model : HashModel} {params : Params}
    (ctx : generatedRoot.Ctx backend model params) {outputs : _}
    (F : generatedRoot.Facts ctx outputs) {next tp : ErrorPoly 8} {key : ByteArray}
    {gadget : Row} {target : Fin 36 → ExactPoly Q 8}
    {fresh : Fin 36 → ErrorMatrix 8 1 25} {bound : Nat}
    (hfresh : ∀ index, CoeffBound (fresh index) bound)
    (hparts : ∀ index : Fin 36, generatedRoot.w_1281_0 ctx index =
      reducePoly Q 8 next • rowOf (generatedRoot.w_227_0 ctx index) 0 +
        rowOf (generatedRoot.w_227_0 ctx index) 1 - target index • gadget +
        reduceMatrix Q 8 1 25 (fresh index))
    (value : (root : ImportRoot) → Fin root.members → ErrorPoly 8)
    (hkey : ∀ (root : ImportRoot) (member : Fin root.members),
      nextKeys ctx (rootTarget root member) =
        keyPairG model key Q 25 root.prefix (root.tags member))
    (hvalue : ∀ (root : ImportRoot) (member : Fin root.members) (component : Fin 2),
      target (partIndex (rootTarget root member) component) =
        reducePoly Q 8 (value root member * unitPair (rootPayload next tp root member) component)) :
    ∀ (root : ImportRoot) (member : Fin root.members),
      Rel next (keyPairG model key Q 25 root.prefix (root.tags member))
        (fun component ↦ reducePoly Q 8
          (value root member * unitPair (rootPayload next tp root member) component) • gadget)
        (rootRows ctx root member) bound := by
  intro root member
  rw [online0_import_rows_eq ctx F root member]
  have hrel := online0_step_transition ctx hfresh hparts (rootTarget root member)
  simpa only [hkey root member, hvalue root member] using hrel

/-- Unit fresh noise specializes every named next-root relation to `Dio2Inv.RootRow`. -/
theorem online0_root_invariant {backend : BackendContext} {model : HashModel} {params : Params}
    (ctx : generatedRoot.Ctx backend model params) {next tp : ErrorPoly 8} {key : ByteArray}
    {gadget : Row} (value : (root : ImportRoot) → Fin root.members → ErrorPoly 8)
    (hrel : ∀ (root : ImportRoot) (member : Fin root.members),
      Rel next (keyPairG model key Q 25 root.prefix (root.tags member))
        (fun component ↦ reducePoly Q 8
          (value root member * unitPair (rootPayload next tp root member) component) • gadget)
        (rootRows ctx root member) 1) :
    ∀ (root : ImportRoot) (member : Fin root.members),
      RootRow model key root.prefix (root.tags member) next gadget
        (fun component _ _ ↦ reducePoly Q 8
          (value root member * unitPair (rootPayload next tp root member) component))
        (rootRows ctx root member) := by
  intro root member
  unfold RootRow
  rw [import_rootKey_eq]
  simpa only [matrixMulScalarLeft_eq] using hrel root member

/-- The refreshed exports discharge every root-relation hypothesis of step one's import bridge. -/
theorem online0_import1_rel {backend : BackendContext} {model : HashModel}
    {params : W_Stage_online_0.Params} (ctx : generatedRoot.Ctx backend model params)
    {importParams : W_Stage_import_1.Params}
    (input : W_Stage_import_1.generatedRoot.Ctx importParams)
    (hwire : ∀ (port : Import1Port) (member : Fin port.members),
      import1InputRows input port member = rootRows ctx port.root member)
    {next tp : ErrorPoly 8} {key : ByteArray} {gadget : Row} {bound : Nat}
    (value : (root : ImportRoot) → Fin root.members → ErrorPoly 8)
    (hrel : ∀ (root : ImportRoot) (member : Fin root.members),
      Rel next (keyPairG model key Q 25 root.prefix (root.tags member))
        (fun component ↦ reducePoly Q 8
          (value root member * unitPair (rootPayload next tp root member) component) • gadget)
        (rootRows ctx root member) bound) :
    ∀ (port : Import1Port) (member : Fin port.members),
      Rel next (keyPairG model key Q 25 port.root.prefix (port.root.tags member))
        (fun component ↦ reducePoly Q 8
          (value port.root member * unitPair (rootPayload next tp port.root member) component) •
            gadget)
        (import1InputRows input port member) bound := by
  intro port member
  rw [hwire port member]
  exact hrel port.root member

/-- Refreshed root facts propagate to `hlab`/`ha`, `hs`, private `hs`, and both `hhelp` inputs. -/
theorem online0_online1_rel {backend : BackendContext} {model : HashModel}
    {params : W_Stage_online_0.Params} (ctx : generatedRoot.Ctx backend model params)
    {importParams : W_Stage_import_1.Params}
    (input : W_Stage_import_1.generatedRoot.Ctx importParams) {outputs : _}
    (F : W_Stage_import_1.generatedRoot.Facts input outputs)
    {onlineParams : W_Stage_online_1.Params}
    (online : W_Stage_online_1.generatedRoot.Ctx backend model onlineParams)
    (hinput : ∀ (port : Import1Port) (member : Fin port.members),
      import1InputRows input port member = rootRows ctx port.root member)
    (honline : ∀ (port : Import1Port) (member : Fin port.members),
      import1OnlineRows online port member = import1OutputRows input port member)
    {next tp : ErrorPoly 8} {gadget : Row} {bound : Nat}
    (value : (root : ImportRoot) → Fin root.members → ErrorPoly 8)
    (hrel : ∀ (root : ImportRoot) (member : Fin root.members),
      Rel next (keyPairG model (W_Stage_online_1.generatedRoot.w_14_0 online) Q 25
        root.prefix (root.tags member))
        (fun component ↦ reducePoly Q 8
          (value root member * unitPair (rootPayload next tp root member) component) • gadget)
        (rootRows ctx root member) bound) :
    ∀ (port : Import1Port) (member : Fin port.members),
      Rel next (keyPairG model (W_Stage_online_1.generatedRoot.w_14_0 online) Q 25
        port.root.prefix (port.root.tags member))
        (fun component ↦ reducePoly Q 8
          (value port.root member * unitPair (rootPayload next tp port.root member) component) •
            gadget)
        (import1OnlineRows online port member) bound := by
  exact import1_online_from_rows input F online honline
    (fun port member component ↦ reducePoly Q 8
      (value port.root member * unitPair (rootPayload next tp port.root member) component) •
        gadget)
    (fun _ _ ↦ bound) (online0_import1_rel ctx input hinput value hrel)

end Dio2Proof
