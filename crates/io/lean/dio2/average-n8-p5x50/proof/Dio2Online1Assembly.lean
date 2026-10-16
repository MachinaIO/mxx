import Dio2KeyAgree1
import Dio2PreprocessSwitch
import Dio2PreprocessRhs
import Dio2StepTransition

/-!
# The last online output, assembled at the claim boundary

The refreshed step-zero roots pass through `import_1` to both step-one PRF buckets and the
state LUT. Preprocessing supplies the exact selector RHSs and switching ciphertexts, with
scale `Δ' = 1` and zero preprocessing error. The last preimage and key agreement cancel the
state and smudge keys in `online1_output`.

`Online1Inputs` records the remaining value and transition obligations, including the refreshed
hint, the coefficientwise smudge residual, and the final strict noise inequality. In particular,
neither the state LUT equation nor the smudge PRF encoding is an input to this assembly.
-/

set_option maxHeartbeats 64000000
set_option maxRecDepth 16384

namespace Dio2Proof

open Mxx.Primitives MxxRuntime MxxExponentLut Mxx.ExponentLut.Refresh RefreshTopProof
  RefreshTopProof.Gen RefreshTopProof.GenIface RefreshTopProof.GenInst RefreshTopProof.GenFinal
  StepTransition W_Stage_online_1
open scoped BigOperators

namespace Online1Assembly

/-- The refreshed root values, retaining the unconsumed labels without restricting them. -/
noncomputable def values (labels : Fin 2 → ErrorPoly 8) (otherLabels : Fin 4 → ErrorPoly 8)
    (state : ErrorPoly 8) :
    (root : ImportRoot) → Fin root.members → ErrorPoly 8
  | .label0, member => labels member
  | .label, member => otherLabels member
  | .state, _ => state
  | _, _ => 1

/-- The error bound of a private/public cell with exact selector preprocessing. -/
def cellBound (labels state : Nat) : Nat :=
  (25 + 25) * 8 * ((25 + 25) * 8 * labels * 2048 + 8 * 1 * state) * 2048 + 8 * 1 * 0

/-- The first bucket starts from the refreshed private unit. -/
def bucket0Bound (bound : Nat) : Nat := 2 * cellBound bound bound

/-- The second bucket starts from the first bucket's accumulated relation. -/
def bucket1Bound (bound : Nat) : Nat := 2 * cellBound bound (bucket0Bound bound)

/-- The state LUT's four branches use refreshed state helpers and the state input cell. -/
def stateBound (bound : Nat) : Nat :=
  3 * ((25 + 25) * 8 * bound * 2048 + (25 + 25) * 8 * cellBound bound bound * 2048) +
    ((25 + 25) * 8 * bound * 2048 + (25 + 25) * 8 * cellBound bound bound * 2048) +
    4 * (8 * 1 * 0)

/-- Eight smudge readouts, each with eight LUT branches, are placed and summed. -/
def smudgeBound (bound : Nat) : Nat :=
  8 * (7 * ((25 + 25) * 8 * bound * 2048 +
      (25 + 25) * 8 * bucket1Bound bound * 2048) +
    ((25 + 25) * 8 * bound * 2048 + (25 + 25) * 8 * bucket1Bound bound * 2048) +
    8 * (8 * 1 * 0))

/-- The placed smudge plaintext before multiplication by the graph's actual half digits. -/
noncomputable def smudgeValue (values : Fin 8 → ErrorPoly 8) : ErrorPoly 8 :=
  ∑ member : Fin 8, Xe (n := 8) member * values member

/-- The exact smudge target consumed by the decoder, using its actual digit matrix. -/
noncomputable def smudgeTarget {backend : BackendContext} {model : HashModel} {params : Params}
    (online : generatedRoot.Ctx backend model params) (gadget : ExactMatrix modulus_0 8 1 25)
    (values : Fin 8 → ErrorPoly 8) : ExactMatrix modulus_0 8 1 1 :=
  (reducePoly modulus_0 8 (smudgeValue values) • gadget) * generatedRoot.w_182_0 online

/-- The preprocessing artifacts read by the last online stage, before any semantic reasoning. -/
structure PreprocessLink {backend : BackendContext} {model : HashModel}
    {params : Params} {preParams : W_Stage_preprocess_1_1.Params}
    (online : generatedRoot.Ctx backend model params)
    (pre : W_Stage_preprocess_1_1.generatedRoot.Ctx backend model preParams) : Prop where
  /-- The RHS top seed is shared. -/
  seed : generatedRoot.w_70_0 online = W_Stage_preprocess_1_1.generatedRoot.w_0_0 pre
  /-- The four state switching families are routed without modification. -/
  state : ![![generatedRoot.w_5_0 online, generatedRoot.w_7_0 online],
      ![generatedRoot.w_9_0 online, generatedRoot.w_11_0 online]] =
    LutKey.fam4 (W_Stage_preprocess_1_1.generatedRoot.w_3_0 pre)
      (W_Stage_preprocess_1_1.generatedRoot.w_3_1 pre)
      (W_Stage_preprocess_1_1.generatedRoot.w_3_2 pre)
      (W_Stage_preprocess_1_1.generatedRoot.w_3_3 pre)
  /-- The four smudge switching families are routed without modification. -/
  smudge : ![![generatedRoot.w_132_0 online, generatedRoot.w_133_0 online],
      ![generatedRoot.w_134_0 online, generatedRoot.w_135_0 online]] =
    LutKey.fam4 (W_Stage_preprocess_1_1.generatedRoot.w_4_0 pre)
      (W_Stage_preprocess_1_1.generatedRoot.w_4_1 pre)
      (W_Stage_preprocess_1_1.generatedRoot.w_4_2 pre)
      (W_Stage_preprocess_1_1.generatedRoot.w_4_3 pre)
  /-- The first bucket's two RHS bottoms. -/
  rhs0 : ![generatedRoot.w_146_0 online, generatedRoot.w_147_0 online] =
    ![W_Stage_preprocess_1_1.generatedRoot.w_75_6 pre,
      W_Stage_preprocess_1_1.generatedRoot.w_75_7 pre]
  /-- The second bucket's two RHS bottoms. -/
  rhs1 : ![generatedRoot.w_150_0 online, generatedRoot.w_151_0 online] =
    ![W_Stage_preprocess_1_1.generatedRoot.w_79_6 pre,
      W_Stage_preprocess_1_1.generatedRoot.w_79_7 pre]
  /-- The state cell's two RHS bottoms. -/
  rhs2 : ![generatedRoot.w_72_0 online, generatedRoot.w_82_0 online] =
    ![W_Stage_preprocess_1_1.generatedRoot.w_33_2 pre,
      W_Stage_preprocess_1_1.generatedRoot.w_33_3 pre]

end Online1Assembly

/-- The explicit transition, value, and noise obligations remaining at the last online step.
The switching and RHS equations are deliberately absent: they follow from preprocessing. -/
structure Online1Inputs {backend : BackendContext} {model : HashModel}
    {oldParams : W_Stage_online_0.Params} {params : Params}
    {importParams : W_Stage_import_1.Params} {preParams : W_Stage_preprocess_1_1.Params}
    (old : W_Stage_online_0.generatedRoot.Ctx backend model oldParams)
    (input : W_Stage_import_1.generatedRoot.Ctx importParams)
    (pre : W_Stage_preprocess_1_1.generatedRoot.Ctx backend model preParams)
    (online : generatedRoot.Ctx backend model params) (sm : ErrorPoly 8)
    (anchor : ExactMatrix modulus_0 8 2 54) where
  /-- The retained private payload. -/
  sp : ErrorPoly 8
  /-- The refreshed state value. -/
  state : ErrorPoly 8
  /-- The refreshed label values. -/
  labels : Fin 2 → ErrorPoly 8
  /-- The remaining refreshed labels, not consumed by the last program. -/
  otherLabels : Fin 4 → ErrorPoly 8
  /-- The common gadget of the root, selector, and switching relations. -/
  gadget : ExactMatrix modulus_0 8 1 25
  /-- The gadget follows the runtime layout. -/
  hgadget : gadgetMatrixRuns backend 4096 25 gadget
  /-- Fresh error budget in the step-zero refresh parts. -/
  bound : Nat
  /-- The fresh errors retained by step zero. -/
  fresh : Fin 36 → ErrorMatrix 8 1 25
  /-- The target of every step-zero part. -/
  target : Fin 36 → ExactPoly modulus_0 8
  /-- Step zero's decoder-success premise, including every fresh error. -/
  parts : ∀ index, W_Stage_online_0.generatedRoot.w_1281_0 old index =
    reducePoly modulus_0 8 sm • MxxExponentLut.rowOf
        (W_Stage_online_0.generatedRoot.w_227_0 old index) 0 +
      MxxExponentLut.rowOf (W_Stage_online_0.generatedRoot.w_227_0 old index) 1 -
      target index • gadget + reduceMatrix modulus_0 8 1 25 (fresh index)
  /-- The step-zero fresh errors satisfy the advertised common budget. -/
  freshBound : ∀ index, CoeffBound (fresh index) bound
  /-- Step zero's new root keys agree with the next step's hash seed. -/
  keys : ∀ (root : ImportRoot) (member : Fin root.members),
    nextKeys old (rootTarget root member) =
      keyPairG model (generatedRoot.w_14_0 online) modulus_0 25 root.prefix (root.tags member)
  /-- Step zero's targets are the values required by the last program. -/
  targets : ∀ (root : ImportRoot) (member : Fin root.members) (component : Fin 2),
    target (partIndex (rootTarget root member) component) = reducePoly modulus_0 8
      (Online1Assembly.values labels otherLabels state root member *
        unitPair (rootPayload sm sp root member) component)
  /-- The preprocessing payload is the retained private payload. -/
  preprocessSecret : W_Stage_preprocess_1_1.generatedRoot.w_1_0 pre =
    fun _ _ ↦ reducePoly modulus_0 8 sp
  /-- External import payloads agree with the refreshed label plaintexts. -/
  payload : ∀ member, message (import1Coefficient input member)
    (import1Exponent input member) = labels member
  /-- The common root mask is small. -/
  smallMask : polyNorm sm ≤ 1
  /-- Public labels are small. -/
  smallLabels : ∀ member, polyNorm (labels member) ≤ 1
  /-- The state cell's public monomial multiplier is small. -/
  smallStateLabel : polyNorm (Xe (n := 8) 2 * labels 1) ≤ 1
  /-- The product passed to the state cell is small. -/
  smallStateProduct : polyNorm (Xe (n := 8) 2 * labels 1 * state) ≤ 1
  /-- The selected coordinate in each first-bucket member. -/
  select0 : Fin 8 → Fin 2
  /-- The selected coordinate in each second-bucket member. -/
  select1 : Fin 8 → Fin 2
  /-- The label selected by each first-bucket member. -/
  key0 : Fin 8 → Fin 2
  /-- The label selected by each second-bucket member. -/
  key1 : Fin 8 → Fin 2
  /-- First-bucket gather indices agree with the advertised labels. -/
  gather0 : ∀ member, generatedRoot.w_139_0 online (cellAt member (select0 member)) =
    ((key0 member).val : Int)
  /-- Second-bucket gather indices agree with the advertised labels. -/
  gather1 : ∀ member, generatedRoot.w_138_0 online (cellAt member (select1 member)) =
    ((key1 member).val : Int)
  /-- The first selector RHS is the advertised one-hot selector. -/
  selector0 : ∀ member component,
    message (W_Stage_preprocess_1_1.generatedRoot.w_73_0 pre (cellAt member component))
      (W_Stage_preprocess_1_1.generatedRoot.w_74_0 pre (cellAt member component)) =
        if component = select0 member then (1 : ErrorPoly 8) else 0
  /-- The second selector RHS is the advertised one-hot selector. -/
  selector1 : ∀ member component,
    message (W_Stage_preprocess_1_1.generatedRoot.w_77_0 pre (cellAt member component))
      (W_Stage_preprocess_1_1.generatedRoot.w_78_0 pre (cellAt member component)) =
        if component = select1 member then (1 : ErrorPoly 8) else 0
  /-- The state RHS multiplies by the retained private unit. -/
  stateRhs : message (W_Stage_preprocess_1_1.generatedRoot.w_31_0 pre 0)
    (W_Stage_preprocess_1_1.generatedRoot.w_32_0 pre 0) = (1 : ErrorPoly 8)
  /-- All second-bucket products are small, including unselected labels. -/
  smallBucket : ∀ key member, polyNorm (labels key * labels (key0 member)) ≤ 1
  /-- The state LUT value. -/
  ν : ErrorPoly 8
  /-- The state LUT inputs are small under every branch automorphism. -/
  smallState : ∀ branch : Fin 4, polyNorm (substitution (sigmaOf sigmas4 branch).toNat
    (sigmas4_odd branch) (Xe (n := 8) 2 * labels 1 * state)) ≤ 1
  /-- The state value identity, with the actual encrypted table coefficients. -/
  stateWalsh : ∑ branch, reducePoly modulus_0 8
      (substitution (sigmaOf sigmas4 branch).toNat (sigmas4_odd branch)
        (Xe (n := 8) 2 * labels 1 * state)) * PreprocessSwitch.state1Coefficients branch =
    reducePoly modulus_0 8 ν
  /-- The eight smudge LUT values. -/
  smudgeValues : Fin 8 → ErrorPoly 8
  /-- The smudge LUT inputs are small under every branch automorphism. -/
  smallSmudge : ∀ member branch, polyNorm (substitution (sigmaOf sigmas8 branch).toNat
    (sigmas8_odd branch) (labels (key1 member) * labels (key0 member))) ≤ 1
  /-- Smudge values are the actual readout table values, including negacyclic signs. -/
  smudgeWalsh : ∀ member, ∑ branch, reducePoly modulus_0 8
      (substitution (sigmaOf sigmas8 branch).toNat (sigmas8_odd branch)
        (labels (key1 member) * labels (key0 member))) *
        PreprocessSwitch.smudgeCoefficients branch = reducePoly modulus_0 8 (smudgeValues member)
  /-- The step-zero refreshed hint's integer error. -/
  hintError : ErrorMatrix 8 1 54
  /-- The refreshed hint's error budget. -/
  hintBound : Nat
  /-- The hint belongs to the sampled next anchor; this is a step-zero transition premise. -/
  hint : W_Stage_online_0.generatedRoot.w_1287_0 old =
    maskPart (reducePoly modulus_0 8 sm) (MxxExponentLut.rowOf anchor) +
      reduceMatrix modulus_0 8 1 54 hintError
  /-- The refreshed hint error satisfies its budget. -/
  hintErrorBound : CoeffBound hintError hintBound
  /-- The smudge residual after subtracting a half from every coefficient. -/
  residual : ErrorMatrix 8 1 1
  /-- The smudge residual budget. -/
  residualBound : Nat
  /-- The explicit smudge obligation `hTH` of `online1_output`. -/
  hTH : Online1Assembly.smudgeTarget online gadget smudgeValues - generatedRoot.w_186_0 online =
    reduceMatrix modulus_0 8 1 1 residual
  /-- Every smudge residual coefficient is small. -/
  residualSmall : CoeffBound residual residualBound
  /-- The complete strict decoder inequality, with the derived state and smudge budgets. -/
  noise : 2 * (54 * 8 * hintBound * 22011706 +
    25 * 8 * Online1Assembly.stateBound bound * 2048 +
    25 * 8 * Online1Assembly.smudgeBound bound * 2048 + residualBound) <
      1606938044254428791916078459060479968194539524630727034176641

namespace Online1Inputs

variable {backend : BackendContext} {model : HashModel}
  {oldParams : W_Stage_online_0.Params} {params : Params}
  {importParams : W_Stage_import_1.Params} {preParams : W_Stage_preprocess_1_1.Params}
  {old : W_Stage_online_0.generatedRoot.Ctx backend model oldParams}
  {input : W_Stage_import_1.generatedRoot.Ctx importParams}
  {pre : W_Stage_preprocess_1_1.generatedRoot.Ctx backend model preParams}
  {online : generatedRoot.Ctx backend model params} {sm : ErrorPoly 8}
  {anchor : ExactMatrix modulus_0 8 2 54}
  (I : Online1Inputs old input pre online sm anchor)

/-- The exact remaining decoder inequality, with no synthetic or assumed encoding budgets. -/
def Noise1 : Prop :=
  2 * (54 * 8 * I.hintBound * 22011706 +
    25 * 8 * Online1Assembly.stateBound I.bound * 2048 +
    25 * 8 * Online1Assembly.smudgeBound I.bound * 2048 + I.residualBound) <
      1606938044254428791916078459060479968194539524630727034176641

/-- The step-zero part forms supply every imported step-one root relation. -/
theorem root_rel {outputs : _} (F : W_Stage_online_0.generatedRoot.Facts old outputs) :
    ∀ (root : ImportRoot) (member : Fin root.members),
      Rel sm (keyPairG model (generatedRoot.w_14_0 online) modulus_0 25
        root.prefix (root.tags member))
        (fun component ↦ reducePoly modulus_0 8
          (Online1Assembly.values I.labels I.otherLabels I.state root member *
            unitPair (rootPayload sm I.sp root member) component) • I.gadget)
        (rootRows old root member) I.bound :=
  online0_root_rel old F I.freshBound I.parts
    (Online1Assembly.values I.labels I.otherLabels I.state)
    I.keys I.targets

/-- The exact selectors and the refreshed private root assemble both smudge PRF buckets. -/
theorem buckets {outputs : _} (F : generatedRoot.Facts online outputs)
    {preOutputs : _} (Fp : W_Stage_preprocess_1_1.generatedRoot.Facts pre preOutputs)
    (L : Online1Assembly.PreprocessLink online pre)
    (hq : 1 < modulus_0) (layout : RegularLayout modulus_0)
    (hlayout : backend.regularLayout modulus_0 8 = some layout)
    (hexact : layout.droppedModuli = 0)
    (ha : ∀ member : Fin 2, Rel sm
      (labelKeysG model (generatedRoot.w_14_0 online) modulus_0 25 labelPre member)
      (fun component ↦ reducePoly modulus_0 8 (I.labels member * unitPair sm component) • I.gadget)
      ![generatedRoot.w_16_0 online member, generatedRoot.w_27_0 online member] I.bound)
    (hpay : ∀ member, generatedRoot.w_47_0 online member =
      fun _ _ ↦ reducePoly modulus_0 8 (I.labels member))
    (hpriv : Rel sm ![![generatedRoot.w_140_0 online, generatedRoot.w_143_0 online],
        ![generatedRoot.w_141_0 online, generatedRoot.w_144_0 online]]
      (fun component ↦ reducePoly modulus_0 8 (1 * unitPair I.sp component) • I.gadget)
      ![generatedRoot.w_142_0 online, generatedRoot.w_145_0 online] I.bound) :
    ∀ member : Fin 8, EncRel sm
      (generatedRoot.w_153_0 online member, (generatedRoot.w_153_1 online member,
        (generatedRoot.w_153_2 online member, (generatedRoot.w_153_3 online member,
          (generatedRoot.w_153_4 online member, (generatedRoot.w_153_5 online member, ()))))))
      (fun component ↦ reducePoly modulus_0 8
        (1 * (I.labels (I.key1 member) * I.labels (I.key0 member)) * unitPair I.sp component) •
          I.gadget) (Online1Assembly.bucket1Bound I.bound) := by
  have hg := centeredRebaseRuns_self hq (by decide : 0 < 8) I.gadget
  have hr0 := Preprocess1.rhs0 pre Fp I.preprocessSecret I.hgadget hg
  have hr1 := Preprocess1.rhs1 pre Fp I.preprocessSecret I.hgadget hg
  have hb0 := online1_bucket0 online F hq layout hlayout hexact I.smallLabels
    (fun member ↦ by simpa using I.smallLabels member) I.hgadget ha hpay I.hgadget hg hpriv
    (ν := fun index ↦ message (W_Stage_preprocess_1_1.generatedRoot.w_73_0 pre index)
      (W_Stage_preprocess_1_1.generatedRoot.w_74_0 pre index))
    (eB := fun _ _ ↦ 0) (BB := 0)
    (fun index component ↦ by
      have hrhs := congrArg (fun rows ↦ ![rows 0 index, rows 1 index]) L.rhs0
      simp only [Matrix.cons_val_zero, Matrix.cons_val_one] at hrhs
      rw [L.seed, congrFun hrhs component]
      simpa only [Nat.cast_one, Int.cast_one, one_mul] using hr0 index component)
    (fun _ _ _ _ _ ↦ by simp) I.select0 I.selector0
  have hfirst : ∀ member : Fin 8, EncRel sm
      (generatedRoot.w_149_0 online member, (generatedRoot.w_149_1 online member,
        (generatedRoot.w_149_2 online member, (generatedRoot.w_149_3 online member,
          (generatedRoot.w_149_4 online member, (generatedRoot.w_149_5 online member, ()))))))
      (fun component ↦ reducePoly modulus_0 8
        (1 * I.labels (I.key0 member) * unitPair I.sp component) • I.gadget)
        (Online1Assembly.bucket0Bound I.bound) := by
    intro member
    obtain ⟨key, hkey, hrel⟩ := hb0 member
    have heq : key = I.key0 member := Fin.ext (by have := I.gather0 member; omega)
    simpa only [heq, Nat.cast_one, Int.cast_one, mul_one, one_mul, Online1Assembly.bucket0Bound,
      Online1Assembly.cellBound] using hrel
  intro member
  obtain ⟨key, hkey, hrel⟩ := online1_bucket1 online F hq layout hlayout hexact
    I.smallLabels I.smallBucket I.hgadget ha hpay I.hgadget hg hfirst
    (ν := fun index ↦ message (W_Stage_preprocess_1_1.generatedRoot.w_77_0 pre index)
      (W_Stage_preprocess_1_1.generatedRoot.w_78_0 pre index))
    (eB := fun _ _ ↦ 0) (BB := 0)
    (fun index component ↦ by
      have hrhs := congrArg (fun rows ↦ ![rows 0 index, rows 1 index]) L.rhs1
      simp only [Matrix.cons_val_zero, Matrix.cons_val_one] at hrhs
      rw [L.seed, congrFun hrhs component]
      simpa only [Nat.cast_one, Int.cast_one, one_mul] using hr1 index component)
    (fun _ _ _ _ _ ↦ by simp) I.select1 I.selector1 member
  have heq : key = I.key1 member := Fin.ext (by have := I.gather1 member; omega)
  simpa only [heq, Online1Assembly.bucket1Bound, Online1Assembly.cellBound] using hrel

/-- Exact preprocessing and the imported state relation discharge the state LUT equation. -/
theorem state_lut {outputs : _} (F : generatedRoot.Facts online outputs)
    {preOutputs : _} (Fp : W_Stage_preprocess_1_1.generatedRoot.Facts pre preOutputs)
    (L : Online1Assembly.PreprocessLink online pre)
    (hq : 1 < modulus_0) (layout : RegularLayout modulus_0)
    (hlayout : backend.regularLayout modulus_0 8 = some layout)
    (hexact : layout.droppedModuli = 0)
    (ha : Rel sm (labelKeysG model (generatedRoot.w_14_0 online) modulus_0 25 labelPre 1)
      (fun component ↦ reducePoly modulus_0 8 (I.labels 1 * unitPair sm component) • I.gadget)
      ![generatedRoot.w_16_0 online 1, generatedRoot.w_27_0 online 1] I.bound)
    (hpay : generatedRoot.w_47_0 online 1 = fun _ _ ↦ reducePoly modulus_0 8 (I.labels 1))
    (hstate : Rel sm (keyPairG model (generatedRoot.w_14_0 online) modulus_0 25 statePre [])
      (fun component ↦ reducePoly modulus_0 8 (I.state * unitPair I.sp component) • I.gadget)
      ![generatedRoot.w_46_0 online, generatedRoot.w_63_0 online] I.bound)
    (hhelp : ∀ branch : Fin 4, Rel sm
      (keyPairG model (generatedRoot.w_14_0 online) modulus_0 25 helperPre
        [.integer (sigmaOf sigmas4 branch)])
      (fun component ↦ reducePoly modulus_0 8
        (1 * unitPair (substitution (sigmaOf sigmas4 branch).toNat (sigmas4_odd branch) sm)
          component) • I.gadget)
      ![generatedRoot.w_13_0 online branch, generatedRoot.w_15_0 online branch] I.bound) :
    ∃ error : ErrorMatrix 8 1 25, CoeffBound error (Online1Assembly.stateBound I.bound) ∧
      generatedRoot.w_124_0 online =
        maskPart (reducePoly modulus_0 8 sm) (stateKey1 online) -
          reducePoly modulus_0 8 (1 * I.ν) • I.gadget + reduceMatrix modulus_0 8 1 25 error := by
  have hg := centeredRebaseRuns_self hq (by decide : 0 < 8) I.gadget
  have hr := Preprocess1.rhs2 pre Fp I.preprocessSecret I.hgadget hg
  have hi := online1_lut_input online F hq layout hlayout hexact
    I.smallStateLabel I.smallStateProduct I.hgadget ha hpay I.hgadget hg hstate
    (T := fun component ↦ reducePoly modulus_0 8 (unitPair I.sp component) • I.gadget)
    (eB := fun _ ↦ 0) (BB := 0)
    (fun component ↦ by
      have hrhs := congrArg (fun rows ↦ ![rows 0 0, rows 1 0]) L.rhs2
      simp only [Matrix.cons_val_zero, Matrix.cons_val_one] at hrhs
      rw [L.seed, congrFun hrhs component]
      simpa only [I.stateRhs, one_mul] using hr 0 component)
    (fun _ _ _ _ ↦ by simp)
  have hs := PreprocessSwitch.preprocess1_state_switching pre Fp hq
    I.preprocessSecret I.hgadget
  rw [← L.state] at hs
  have hsw : ∀ branch : Fin 4, ∃ error : ErrorMatrix 8 1 25, CoeffBound error 0 ∧
      reducePoly modulus_0 8
          (substitution (sigmaOf sigmas4 branch).toNat (sigmas4_odd branch) I.sp) •
          generatedRoot.w_7_0 online branch + generatedRoot.w_11_0 online branch =
        (PreprocessSwitch.state1Coefficients branch * reducePoly modulus_0 8 1) • I.gadget +
          reduceMatrix modulus_0 8 1 25 error := by
    intro branch
    simpa [unitPair] using hs 1 branch
  have he := online1_state_lut online F hq layout hlayout hexact I.smallMask I.hgadget hg
    (by simpa only [smul_smul, ← map_mul, Nat.cast_one, Int.cast_one, one_mul] using hi)
    hhelp hsw
    I.smallState I.stateWalsh
  obtain ⟨error, hbound, heq⟩ := he
  exact ⟨error, hbound, by simpa [stateKey1, maskPart] using heq⟩

/-- The smudge readouts discharge the decoder's smudge equation with bounded digit noise. -/
theorem smudge {outputs : _} (F : generatedRoot.Facts online outputs)
    {preOutputs : _} (Fp : W_Stage_preprocess_1_1.generatedRoot.Facts pre preOutputs)
    (L : Online1Assembly.PreprocessLink online pre)
    (hq : 1 < modulus_0) (layout : RegularLayout modulus_0)
    (hlayout : backend.regularLayout modulus_0 8 = some layout)
    (hexact : layout.droppedModuli = 0)
    (hin : ∀ member : Fin 8, EncRel sm
      (generatedRoot.w_153_0 online member, (generatedRoot.w_153_1 online member,
        (generatedRoot.w_153_2 online member, (generatedRoot.w_153_3 online member,
          (generatedRoot.w_153_4 online member, (generatedRoot.w_153_5 online member, ()))))))
      (fun component ↦ reducePoly modulus_0 8
        (1 * (I.labels (I.key1 member) * I.labels (I.key0 member)) * unitPair I.sp component) •
          I.gadget) (Online1Assembly.bucket1Bound I.bound))
    (hhelp : ∀ branch : Fin 8, Rel sm
      (keyPairG model (generatedRoot.w_14_0 online) modulus_0 25 helperPre
        [.integer (sigmaOf sigmas8 branch)])
      (fun component ↦ reducePoly modulus_0 8
        (1 * unitPair (substitution (sigmaOf sigmas8 branch).toNat (sigmas8_odd branch) sm)
          component) • I.gadget)
      ![generatedRoot.w_136_0 online branch, generatedRoot.w_137_0 online branch] I.bound) :
    ∃ error : ErrorMatrix 8 1 1,
      CoeffBound error (25 * 8 * Online1Assembly.smudgeBound I.bound * 2048) ∧
      generatedRoot.w_183_0 online = maskPart (reducePoly modulus_0 8 sm) (smudgeKeyS1 online) -
        Online1Assembly.smudgeTarget online I.gadget I.smudgeValues +
          reduceMatrix modulus_0 8 1 1 error := by
  have hs := PreprocessSwitch.preprocess1_smudge_switching pre Fp hq
    I.preprocessSecret I.hgadget
  rw [← L.smudge] at hs
  have ht := online1_smudge_total online F hq layout hlayout hexact I.smallMask I.hgadget
    (centeredRebaseRuns_self hq (by decide) I.gadget) hin hhelp
    (PreprocessSwitch.switching_errors hs) I.smallSmudge I.smudgeWalsh
  obtain ⟨error, hbound, heq⟩ := Rel.comp ht 1
  obtain ⟨digits, hdigits, hdb⟩ := F.c7.h182_0.2.2.2
  refine ⟨error * digits, coeffBound_mul (by decide) hbound hdb, ?_⟩
  have htarget : (∑ member : Fin 8, fun component ↦
      reducePoly modulus_0 8 (Xe (n := 8) member) •
        (reducePoly modulus_0 8 (1 * I.smudgeValues member * unitPair I.sp component) •
          I.gadget)) 1 =
      reducePoly modulus_0 8 (Online1Assembly.smudgeValue I.smudgeValues) • I.gadget := by
    simp only [Finset.sum_apply, unitPair, Matrix.cons_val_one, Matrix.cons_val_zero,
      one_mul, mul_one, smul_smul, ← map_mul, ← Finset.sum_smul, ← map_sum,
      Online1Assembly.smudgeValue]
  have hrow : generatedRoot.w_175_0 online = generatedRoot.w_174_5 online :=
    key1_slice_full F.c7.h175_0
  change generatedRoot.w_175_0 online * generatedRoot.w_182_0 online = _
  rw [hrow]
  change generatedRoot.w_174_5 online = _ at heq
  rw [htarget] at heq
  rw [heq, Matrix.add_mul, Matrix.sub_mul, maskPart_mul, hdigits, ← reduceMatrix_mul, ← hdigits]
  rfl

/-- Refreshed roots, exact preprocessing, the solved preimage, and the explicit residual
discharge every semantic input to `online1_output`. -/
theorem output {oldOutputs : _} (F0 : W_Stage_online_0.generatedRoot.Facts old oldOutputs)
    {importOutputs : _} (Fi : W_Stage_import_1.generatedRoot.Facts input importOutputs)
    {preOutputs : _} (Fp : W_Stage_preprocess_1_1.generatedRoot.Facts pre preOutputs)
    {outputs : _} (F : generatedRoot.Facts online outputs)
    (L : Online1Assembly.PreprocessLink online pre)
    (hinput : ∀ (port : Import1Port) (member : Fin port.members),
      import1InputRows input port member = rootRows old port.root member)
    (honline : ∀ (port : Import1Port) (member : Fin port.members),
      import1OnlineRows online port member = import1OutputRows input port member)
    (hpayload : ∀ member, import1OnlinePayload online member = import1Payload input member)
    (hhint : generatedRoot.w_0_0 online = W_Stage_online_0.generatedRoot.w_1287_0 old)
    (layout : RegularLayout modulus_0)
    (hlayout : backend.regularLayout modulus_0 8 = some layout)
    (hexact : layout.droppedModuli = 0)
    {Pd : ErrorMatrix 8 54 1}
    (hP : generatedRoot.w_1_0 online = reduceMatrix modulus_0 8 54 1 Pd)
    (hPd : CoeffBound Pd 22011706)
    (hpre : ∀ component, MxxExponentLut.rowOf anchor component * generatedRoot.w_1_0 online =
      stateKey1 online component * generatedRoot.w_130_0 online + smudgeKeyS1 online component)
    (column : Fin 1) (coefficient : Fin 8) :
    (generatedRoot.w_188_0 online 0 column).coeff coefficient =
      (((reducePoly modulus_0 8 (1 * I.ν)).coeff coefficient).val : ZMod modulus_1) := by
  have hq : 1 < modulus_0 := by decide
  have hrel := online0_online1_rel old input Fi online hinput honline
    (Online1Assembly.values I.labels I.otherLabels I.state) (I.root_rel F0)
  have ha : ∀ member : Fin 2, Rel sm
      (labelKeysG model (generatedRoot.w_14_0 online) modulus_0 25 labelPre member)
      (fun component ↦ reducePoly modulus_0 8 (I.labels member * unitPair sm component) • I.gadget)
      ![generatedRoot.w_16_0 online member, generatedRoot.w_27_0 online member] I.bound := by
    intro member
    simpa [Import1Port.root, ImportRoot.prefix, ImportRoot.tags, import1OnlineRows,
      rootPayload, Online1Assembly.values, labelKeysG] using hrel .labelState member
  have hp := import1_online_payload input Fi online hpayload I.payload
  have hs : Rel sm (keyPairG model (generatedRoot.w_14_0 online) modulus_0 25 statePre [])
      (fun component ↦ reducePoly modulus_0 8 (I.state * unitPair I.sp component) • I.gadget)
      ![generatedRoot.w_46_0 online, generatedRoot.w_63_0 online] I.bound := by
    simpa [Import1Port.root, ImportRoot.prefix, ImportRoot.tags, import1OnlineRows,
      rootPayload, Online1Assembly.values] using hrel .state ⟨0, by decide⟩
  have h4 : ∀ branch : Fin 4, (ImportRoot.exp4.sigma branch : Int) = sigmaOf sigmas4 branch :=
    by decide
  have h8 : ∀ branch : Fin 8, (ImportRoot.exp8.sigma branch : Int) = sigmaOf sigmas8 branch :=
    by decide
  have hh4 : ∀ branch : Fin 4, Rel sm
      (keyPairG model (generatedRoot.w_14_0 online) modulus_0 25 helperPre
        [.integer (sigmaOf sigmas4 branch)])
      (fun component ↦ reducePoly modulus_0 8
        (1 * unitPair (substitution (sigmaOf sigmas4 branch).toNat (sigmas4_odd branch) sm)
          component) • I.gadget)
      ![generatedRoot.w_13_0 online branch, generatedRoot.w_15_0 online branch] I.bound := by
    intro branch
    simpa only [Import1Port.root, ImportRoot.prefix, ImportRoot.tags, import1OnlineRows,
      Online1Assembly.values, rootPayload, ← h4 branch, Int.toNat_natCast] using
        hrel .helperState branch
  have hh8 : ∀ branch : Fin 8, Rel sm
      (keyPairG model (generatedRoot.w_14_0 online) modulus_0 25 helperPre
        [.integer (sigmaOf sigmas8 branch)])
      (fun component ↦ reducePoly modulus_0 8
        (1 * unitPair (substitution (sigmaOf sigmas8 branch).toNat (sigmas8_odd branch) sm)
          component) • I.gadget)
      ![generatedRoot.w_136_0 online branch, generatedRoot.w_137_0 online branch] I.bound := by
    intro branch
    simpa only [Import1Port.root, ImportRoot.prefix, ImportRoot.tags, import1OnlineRows,
      Online1Assembly.values, rootPayload, ← h8 branch, Int.toNat_natCast] using
        hrel .helperSmudge branch
  have hk : keyPairG model (generatedRoot.w_14_0 online) modulus_0 25 privatePrefix [] =
      ![![generatedRoot.w_140_0 online, generatedRoot.w_143_0 online],
        ![generatedRoot.w_141_0 online, generatedRoot.w_144_0 online]] := by
    funext side component
    fin_cases side <;> fin_cases component
    · exact (keyPairG_of_hash 0 0 (c := 0) (by norm_num) F.c6.h140_0).symm
    · exact (keyPairG_of_hash 0 1 (c := 1) (by norm_num) F.c6.h143_0).symm
    · exact (keyPairG_of_hash 1 0 (c := 2) (by norm_num) F.c6.h141_0).symm
    · exact (keyPairG_of_hash 1 1 (c := 3) (by norm_num) F.c6.h144_0).symm
  have hpriv := hrel .privateRoot ⟨0, by decide⟩
  simp only [Import1Port.root, ImportRoot.prefix, ImportRoot.tags, import1OnlineRows,
    rootPayload, Online1Assembly.values, hk] at hpriv
  have hbucket := I.buckets F Fp L hq layout hlayout hexact ha hp hpriv
  obtain ⟨el, hel, hlut⟩ := I.state_lut F Fp L hq layout hlayout hexact (ha 1) (hp 1) hs hh4
  obtain ⟨es, hes, hsmudge⟩ := I.smudge F Fp L hq layout hlayout hexact hbucket hh8
  obtain ⟨digits, hdigits, hdb⟩ := F.c6.h130_0.2.2.2
  apply online1_output online F layout hlayout hexact (by decide) I.hgadget hel
    (K0 := stateKey1 online 0) (K1 := stateKey1 online 1)
    (by simpa [stateKey1, maskPart] using hlut)
    (by rw [hhint]; exact I.hint) I.hintErrorBound hP hPd
    (by intro component; simpa [stateKey1] using hpre component)
    hsmudge hes I.hTH I.residualSmall hdigits hdb I.noise column coefficient

end Online1Inputs

/-- The decoded `dio2/output` of the actual claim is coefficientwise the residue of the
step-one state LUT value. Preprocessing fixes the LUT scale to `Δ' = 1`; all remaining
transition, value, smudge, and noise premises are named in `Online1Inputs`. -/
theorem claim_online1_output {model : HashModel} {external : GeneratedClaim.ExternalInputs}
    {execution : GeneratedClaim.Execution} (h : W_GeneratedClaim.Runs model external execution)
    (sm : ErrorPoly 8) (layout : RegularLayout modulus_0)
    (hlayout : Backend.backend.regularLayout modulus_0 8 = some layout)
    (hexact : layout.droppedModuli = 0) :
    ∃ (old : W_Stage_online_0.generatedRoot.Ctx Backend.backend model
        W_GeneratedClaim.stage_11_params)
      (input : W_Stage_import_1.generatedRoot.Ctx W_GeneratedClaim.stage_16_params)
      (pre : W_Stage_preprocess_1_1.generatedRoot.Ctx Backend.backend model
        W_GeneratedClaim.stage_12_params)
      (online : generatedRoot.Ctx Backend.backend model W_GeneratedClaim.stage_17_params),
      W_Stage_online_0.generatedRoot.Facts old execution.«stage_11» ∧
      W_Stage_import_1.generatedRoot.Facts input execution.«stage_16» ∧
      W_Stage_preprocess_1_1.generatedRoot.Facts pre execution.«stage_12» ∧
      generatedRoot.Facts online execution.«stage_17» ∧
      ∀ I : Online1Inputs old input pre online sm execution.«stage_1».2.2.1,
        ∀ (column : Fin 1) (coefficient : Fin 8),
          (execution.«stage_17».1 0 column).coeff coefficient =
            (((reducePoly modulus_0 8 (1 * I.ν)).coeff coefficient).val : ZMod modulus_1) := by
  have hp := (claim_preimages h).2
  obtain ⟨w0, hw0⟩ := importClaimOnline0Inputs_runs h
  obtain ⟨wi, hwi⟩ := importClaim1Inputs_runs h
  obtain ⟨wo, hwo⟩ := importClaimOnline1Inputs_runs h
  have runs := h
  obtain ⟨-, -, -, -, -, -, -, -, -, -, -, -, -, h12, -, h14, -⟩ := runs
  obtain ⟨wp, hwp⟩ := h12
  obtain ⟨wk, hwk⟩ := h14
  let old : W_Stage_online_0.generatedRoot.Ctx Backend.backend model
      W_GeneratedClaim.stage_11_params := ⟨importClaimOnline0Inputs external execution, w0⟩
  let input : W_Stage_import_1.generatedRoot.Ctx W_GeneratedClaim.stage_16_params :=
    ⟨importClaim1Inputs external execution, wi⟩
  let pre : W_Stage_preprocess_1_1.generatedRoot.Ctx Backend.backend model
      W_GeneratedClaim.stage_12_params := ⟨(external.input_1, execution.stage_0,
        external.input_3, external.input_35, external.input_36, external.input_31,
        external.input_30, external.input_22, external.input_39, external.input_23,
        external.input_39, ()), wp⟩
  let online : generatedRoot.Ctx Backend.backend model W_GeneratedClaim.stage_17_params :=
    ⟨importClaimOnline1Inputs external execution, wo⟩
  let keys := (fun inputs (_ : W_Stage_keys_online_1_1.generatedRoot.body Backend.backend
      model W_GeneratedClaim.stage_14_params inputs execution.stage_14 wk) ↦
        (⟨inputs, wk⟩ : W_Stage_keys_online_1_1.generatedRoot.Ctx Backend.backend model
          W_GeneratedClaim.stage_14_params)) _ hwk
  have F0 := W_Stage_online_0.generatedRoot.facts old hw0
  have Fi := W_Stage_import_1.generatedRoot.facts input hwi
  have Fp := W_Stage_preprocess_1_1.generatedRoot.facts pre hwp
  have Fo := generatedRoot.facts online hwo
  have Fk := W_Stage_keys_online_1_1.generatedRoot.facts keys hwk
  have L : Online1Assembly.PreprocessLink online pre := by
    constructor <;>
      simp only [generatedRoot.w_70_0, generatedRoot.w_5_0, generatedRoot.w_7_0,
        generatedRoot.w_9_0, generatedRoot.w_11_0, generatedRoot.w_132_0,
        generatedRoot.w_133_0, generatedRoot.w_134_0, generatedRoot.w_135_0,
        generatedRoot.w_146_0, generatedRoot.w_147_0, generatedRoot.w_150_0,
        generatedRoot.w_151_0, generatedRoot.w_72_0, generatedRoot.w_82_0,
        online, importClaimOnline1Inputs, Fp.c3.hout,
        W_Stage_preprocess_1_1.generatedRoot.outs, LutKey.fam4,
        W_Stage_preprocess_1_1.generatedRoot.w_0_0, pre]
  have hinput : ∀ (port : Import1Port) (member : Fin port.members),
      import1InputRows input port member = rootRows old port.root member := by
    intro port member
    cases port <;>
      simp only [import1InputRows, Import1Port.root, rootRows, input, importClaim1Inputs,
        W_Stage_import_1.generatedRoot.w_0_0, W_Stage_import_1.generatedRoot.w_1_0,
        W_Stage_import_1.generatedRoot.w_2_0, W_Stage_import_1.generatedRoot.w_3_0,
        W_Stage_import_1.generatedRoot.w_7_0, W_Stage_import_1.generatedRoot.w_8_0,
        W_Stage_import_1.generatedRoot.w_9_0, W_Stage_import_1.generatedRoot.w_10_0,
        W_Stage_import_1.generatedRoot.w_12_0, W_Stage_import_1.generatedRoot.w_13_0,
        F0.c70.hout, W_Stage_online_0.generatedRoot.outs]
  have honline := import1_claim_wiring input online external execution Fi rfl
  have hpay := import1_claim_payload_wiring input online external execution Fi rfl
  have hhint : generatedRoot.w_0_0 online = W_Stage_online_0.generatedRoot.w_1287_0 old := by
    simp only [generatedRoot.w_0_0, online, importClaimOnline1Inputs,
      F0.c70.hout, W_Stage_online_0.generatedRoot.outs]
  have hsolved : execution.«stage_1».2.2.1 * generatedRoot.w_1_0 online =
      W_Stage_keys_online_1_1.generatedRoot.w_185_0 keys := by
    have heq := hp.1
    rw [Fk.c9.hout] at heq
    exact heq
  have hpre := online1_hpre Fo Fk (by constructor <;> rfl) (by decide)
    layout hlayout hexact hsolved
  obtain ⟨Pd, hPd, hbound⟩ := hp.2
  refine ⟨old, input, pre, online, F0, Fi, Fp, Fo, fun I column coefficient ↦ ?_⟩
  rw [Fo.c9.hout]
  exact I.output F0 Fi Fp Fo L hinput honline hpay hhint layout hlayout hexact
    hPd hbound hpre column coefficient

end Dio2Proof
