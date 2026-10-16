import Dio2Online0Assembly0
import Dio2Online0Row
import Dio2KeyAgree0P1
import Dio2KeyAgree0P2
import Dio2KeyAgree0P3
import Dio2KeyAgree0P4

/-!
# Step-zero refresh assembly for all five CRT primes

The existing `Online0Inputs` supplies the encoding, switching, plaintext, and norm inputs.
`Online0InputsP` adds only the smudge residues for primes 1 through 4, at cells
`125 i + 25 p + c`; the prime-zero residue is inherited unchanged. The noise conditions remain
explicit and prime-indexed. All decoder offsets are the corrected all-coefficient row `w_1276`.

Each CRT scalar times its decoder correction is 1 modulo its prime. Consequently the common
target is `T_i = reducePoly Q_0 8 (cxPlain i)`, rather than the prime-zero-specific lift `T0`.
The five stage-level terms and the claim-level `claim_online0_term1` through
`claim_online0_term4` therefore refer to the same `partRow`. `claim_online0_row` combines them
with `online0_part_row`, proving the complete CRT row `w_288` for every refresh part.

Key agreement is discharged by the stage-level `online0_hpre*_full` lemmas underlying
`claim_hpre*`, using the same linked claim contexts and bounded preimages. Encoding and
switching premises, smudge residues, and the strict noise inequalities are not assumed proven.
-/

set_option maxHeartbeats 64000000
set_option maxRecDepth 16384

namespace Dio2Proof

open Mxx.Primitives MxxRuntime MxxExponentLut Mxx.ExponentLut.Refresh RefreshTopProof
  RefreshTopProof.Gen RefreshTopProof.GenIface RefreshTopProof.GenInst
  RefreshTopProof.GenFinal GenDecode W_Stage_online_0

namespace Online0Inputs

variable {backend : BackendContext} {hashModel : HashModel} {params : Params}
  {Con : generatedRoot.Ctx backend hashModel params} {sm : ErrorPoly 8}

/-- The common target plaintext of part `i`, independent of the CRT prime. -/
noncomputable def T (I : Online0Inputs Con sm) (i : Fin 36) : ExactPoly modulus_0 8 :=
  reducePoly modulus_0 8 (I.cxPlain i)

/-- The full intended row `β A'₀ + A'₁ - T_i G' + E_i` of refresh part `i`. -/
noncomputable def partRow (I : Online0Inputs Con sm) (i : Fin 36) :
    ExactMatrix modulus_0 8 1 25 :=
  reducePoly modulus_0 8 I.xσ • MxxExponentLut.rowOf (generatedRoot.w_227_0 Con i) 0 +
    MxxExponentLut.rowOf (generatedRoot.w_227_0 Con i) 1 - I.T i •
      generatedRoot.w_57_0 Con + reduceMatrix modulus_0 8 1 25 (I.E0 i)

end Online0Inputs

/-- The additional smudge-residue premises for primes 1 through 4 of step zero. -/
structure Online0InputsP {backend : BackendContext} {hashModel : HashModel} {params : Params}
    (Con : generatedRoot.Ctx backend hashModel params) (sm : ErrorPoly 8)
    extends Online0Inputs Con sm where
  /-- Prime `p + 1` uses smudge cells `125 i + 25 (p + 1) + c`, with the same residue bound. -/
  hTHP : ∀ (p : Fin 4) (i : Fin 36), ∃ residue : ErrorMatrix 8 1 25,
    CoeffBound residue BM ∧
      Matrix.of (fun _ col ↦ reducePoly modulus_0 8
        (cellVal (C := 4500) (by norm_num) sΔ sν
          ⟨125 * i + 25 * (p.val + 1) + col.val,
            by have := i.isLt; have := p.isLt; have := col.isLt; omega⟩) * hconst) -
        generatedRoot.w_1276_0 Con = reduceMatrix modulus_0 8 1 25 residue

/-- The quotient `Q_0 / q_p` controlling the strict decoding cutoff at prime `p`. -/
def online0PrimeQuotient : Fin 5 → Nat :=
  ![1606938044254428791916078459060479968194539524630727034176641,
    1606938044254565807694577896670997561277593686540426466090209,
    1606938044254771331362327096896778694705812889646799470123121,
    1606938044255228050623992174508164557277769211170177223088561,
    1606938044255707605848740785410594412685698358681526360823809]

/-- The preimage input of part `i` at prime `p`. -/
noncomputable def online0PrimePreimage {backend : BackendContext} {hashModel : HashModel}
    {params : Params} (Con : generatedRoot.Ctx backend hashModel params) (p : Fin 5)
    (i : Fin 36) : ExactMatrix modulus_0 8 54 25 :=
  ![generatedRoot.w_1_0 Con, generatedRoot.w_1277_0 Con, generatedRoot.w_1278_0 Con,
    generatedRoot.w_1279_0 Con, generatedRoot.w_1280_0 Con] p i

/-- The keys stage's corresponding prime-indexed preimage right-hand side. -/
noncomputable def online0PrimeKeyImage {backend : BackendContext} {hashModel : HashModel}
    {params : W_Stage_keys_online_0_0.Params}
    (Ck : W_Stage_keys_online_0_0.generatedRoot.Ctx backend hashModel params) (p : Fin 5)
    (i : Fin 36) : ExactMatrix modulus_0 8 2 25 :=
  ![W_Stage_keys_online_0_0.generatedRoot.w_1267_0 Ck,
    W_Stage_keys_online_0_0.generatedRoot.w_1268_0 Ck,
    W_Stage_keys_online_0_0.generatedRoot.w_1269_0 Ck,
    W_Stage_keys_online_0_0.generatedRoot.w_1270_0 Ck,
    W_Stage_keys_online_0_0.generatedRoot.w_1271_0 Ck] p i

/-- The five term output wires of a step-zero refresh part. -/
noncomputable def online0PrimeTerm {backend : BackendContext} {hashModel : HashModel}
    {params : Params} {i : Nat} (ctx : parallel_generatedRoot_1281.Ctx backend hashModel params i)
    (p : Fin 5) : ExactMatrix modulus_0 8 1 25 :=
  ![parallel_generatedRoot_1281.w_64_0 ctx, parallel_generatedRoot_1281.w_119_0 ctx,
    parallel_generatedRoot_1281.w_175_0 ctx, parallel_generatedRoot_1281.w_231_0 ctx,
    parallel_generatedRoot_1281.w_287_0 ctx] p

/-- An inverse CRT correction gives the residue of the common target plaintext. -/
theorem online0_scalar_inverse {q : Nat} (hq : 1 < q) (hQ : 1 < modulus_0)
    (hqQ : q ∣ modulus_0) (scalar correction : Int)
    (hinverse : ((scalar * correction : Int) : ZMod q) = 1) (x : ErrorPoly 8) :
    reducePoly q 8 ((scalar : ErrorPoly 8) * x) * (1 : ExactPoly q 8) *
      (correction : ExactPoly q 8) = rho q (reducePoly modulus_0 8 x) := by
  rw [rho_reduce (by norm_num) hq hQ hqQ, map_mul, reducePoly_intCast, mul_one]
  calc
    (scalar : ExactPoly q 8) * reducePoly q 8 x * (correction : ExactPoly q 8) =
        ((scalar * correction : Int) : ExactPoly q 8) * reducePoly q 8 x := by
      push_cast
      ring
    _ = reducePoly q 8 x := by
      rw [intCast_eq_algebraMap, hinverse, map_one, one_mul]

namespace Online0InputsP

variable {backend : BackendContext} {hashModel : HashModel} {params : Params}
  {Con : generatedRoot.Ctx backend hashModel params} {sm : ErrorPoly 8}
  (I : Online0InputsP Con sm)

/-- The part's full decoding noise inequality at CRT prime `p`. -/
def Noise (p : Fin 5) (i : Fin 36) : Prop :=
  2 * (54 * 8 * 1 * 22011706 +
      (25 * 8 * I.toOnline0Inputs.gBound * 2048 +
        25 * 8 * I.toOnline0Inputs.c1Bound * 2048 +
        25 * 8 * I.toOnline0Inputs.cxBound i * 2048) +
      25 * 8 * ((25 * 25) * 8 * I.toOnline0Inputs.freshBound * 2048) * 2048 +
      (25 * 25) * 8 * I.toOnline0Inputs.smudgeBound * 2048 + I.BM) <
    online0PrimeQuotient p

/-- Prime zero's common noise condition is exactly the original `Noise0`. -/
theorem noise_zero (i : Fin 36) : I.Noise 0 i ↔ I.toOnline0Inputs.Noise0 i := by
  rfl

/-- The prime-0 CRT term of every part, with the common target `T_i = cxPlain i`.
The linked keys, row forms, fresh and smudge cells discharge the corresponding online premises. -/
theorem part_term0 {outputs : _} (Fon : generatedRoot.Facts Con outputs)
    {paramsK : W_Stage_keys_online_0_0.Params}
    {Ck : W_Stage_keys_online_0_0.generatedRoot.Ctx backend hashModel paramsK} {ok : _}
    (Fk : W_Stage_keys_online_0_0.generatedRoot.Facts Ck ok) (L : Link0 Con Ck)
    (hq1 : 1 < modulus_0) (layout : RegularLayout modulus_0)
    (hlayout : backend.regularLayout modulus_0 8 = some layout) (hexact : layout.droppedModuli = 0)
    {A : ExactMatrix modulus_0 8 2 54}
    (hsolve : ∀ j : Fin 36, A * generatedRoot.w_1_0 Con j =
      W_Stage_keys_online_0_0.generatedRoot.w_1267_0 Ck j)
    (hP : ∀ j : Fin 36, ∃ Pd : ErrorMatrix 8 54 25,
      generatedRoot.w_1_0 Con j = reduceMatrix modulus_0 8 54 25 Pd ∧ CoeffBound Pd 22011706)
    {eb : ErrorMatrix 8 1 54} (heb : CoeffBound eb 1)
    (hb : generatedRoot.w_0_0 Con = maskPart (reducePoly modulus_0 8 sm) (MxxExponentLut.rowOf A) +
      reduceMatrix modulus_0 8 1 54 eb)
    (hnoise : ∀ i, I.Noise 0 i)
    (i : Fin 36) (c : parallel_generatedRoot_1281.Ctx backend hashModel params i)
    (hc : c.inputs = Online0Inputs.PartInputs0 Con i) {oc : _}
    (Fc : parallel_generatedRoot_1281.Facts c oc) :
    TermOf modulus_1 (crt0 0)
      (parallel_generatedRoot_1281.w_64_0 c) (I.toOnline0Inputs.partRow i) := by
  have e0 : parallel_generatedRoot_1281.w_0_0 c = generatedRoot.w_0_0 Con := by
    unfold parallel_generatedRoot_1281.w_0_0; rw [hc]
  have e1 : parallel_generatedRoot_1281.w_1_0 c = generatedRoot.w_1_0 Con i := by
    unfold parallel_generatedRoot_1281.w_1_0; rw [hc]
  have e4 : parallel_generatedRoot_1281.w_4_0 c = generatedRoot.w_63_0 Con := by
    unfold parallel_generatedRoot_1281.w_4_0; rw [hc]
  have e5 : parallel_generatedRoot_1281.w_5_0 c = generatedRoot.w_70_0 Con := by
    unfold parallel_generatedRoot_1281.w_5_0; rw [hc]
  have e8 : parallel_generatedRoot_1281.w_8_0 c = generatedRoot.w_227_0 Con i := by
    unfold parallel_generatedRoot_1281.w_8_0; rw [hc]
  have e13 : parallel_generatedRoot_1281.w_13_0 c = generatedRoot.w_1162_0 Con i := by
    unfold parallel_generatedRoot_1281.w_13_0; rw [hc]
  have e29 : parallel_generatedRoot_1281.w_29_0 c = generatedRoot.w_1219_5 Con := by
    unfold parallel_generatedRoot_1281.w_29_0; rw [hc]
  have e31 : parallel_generatedRoot_1281.w_31_0 c = generatedRoot.w_57_0 Con := by
    unfold parallel_generatedRoot_1281.w_31_0; rw [hc]
  have e47 : parallel_generatedRoot_1281.w_47_0 c = generatedRoot.w_1273_5 Con := by
    unfold parallel_generatedRoot_1281.w_47_0; rw [hc]
  have e49 : parallel_generatedRoot_1281.w_49_0 c = generatedRoot.w_1274_0 Con := by
    unfold parallel_generatedRoot_1281.w_49_0; rw [hc]
  have e59 : parallel_generatedRoot_1281.w_59_0 c = generatedRoot.w_1276_0 Con := by
    unfold parallel_generatedRoot_1281.w_59_0; rw [hc]
  have e14 : parallel_generatedRoot_1281.w_14_0 c = generatedRoot.w_57_0 Con :=
    gadgetMatrixRuns_unique Fc.c0.h14_0 Fon.c1.h57_0
  have e15 : parallel_generatedRoot_1281.w_15_0 c = generatedRoot.w_57_0 Con :=
    (rebase_self hq1 Fc.c0.h15_0).trans e14
  have hG31 : gadgetMatrixRuns backend 4096 25 (parallel_generatedRoot_1281.w_31_0 c) := by
    rw [e31]; exact Fon.c1.h57_0
  obtain ⟨eg, heg, hg⟩ := I.toOnline0Inputs.g_form Fon hq1 layout hlayout hexact
  obtain ⟨e₁, he₁, hc₁⟩ := I.toOnline0Inputs.c1_form Fon hq1 layout hlayout hexact
  obtain ⟨ex, hex, hcx⟩ := I.toOnline0Inputs.cx_rows Fon hq1 layout hlayout hexact i
  rw [← online0_gadget_eq Con Fon] at hcx
  obtain ⟨Pd, hPe, hPd⟩ := hP i
  choose ef hef hz using I.toOnline0Inputs.fresh_cells Fon hq1 layout hlayout hexact
  obtain ⟨ef', hef', hf⟩ := online0_part_fresh0 c Fc i.isLt hq1 layout hlayout hexact hG31
    (K := freshCells0 Con) (t := I.toOnline0Inputs.freshCell) (e := ef)
    (fun cc ↦ by rw [e29, e31]; exact hz cc) hef
  choose es hes hzs using I.toOnline0Inputs.smudge_cells Fon hq1 layout hlayout hexact
  obtain ⟨es', hes', hs⟩ := online0_part_smudge0 c Fc i.isLt hq1 layout hlayout hexact hG31
    (K := smudgeCells0 Con) (t := I.toOnline0Inputs.smudgeCell) (e := es)
    (fun cc ↦ by rw [e47, e31]; exact hzs cc)
    hes (h := I.hconst) (by rw [e49]; exact I.hconst_eq)
  obtain ⟨M, hM, hTH⟩ := I.hTH i
  have hpre := online0_hpre0_full Fon Fk L hq1 layout hlayout hexact hsolve i c hc Fc
  have hRq : modulus_0 =
      1606938044254428791916078459060479968194539524630727034176641 * modulus_1 := by
    norm_num [modulus_0, modulus_1]
  have hq : 1 < modulus_1 := by norm_num [modulus_1]
  have hqR : modulus_1 ∣ modulus_0 :=
    ⟨1606938044254428791916078459060479968194539524630727034176641,
    by rw [hRq, Nat.mul_comm]⟩
  have hscalar := online0_scalar_inverse hq hq1 hqR 761964914453472 11560550400
    (by decide) (I.toOnline0Inputs.cxPlain i)
  have h := online0_part_prime0 c Fc hRq layout hlayout hexact (k := 761964914453472) (by norm_num)
    (s := reducePoly modulus_0 8 sm) (β := reducePoly modulus_0 8 I.toOnline0Inputs.xσ)
    (x := I.toOnline0Inputs.cxPlain i)
    heg he₁ hex (by rw [e4, e14]; exact hg) (by rw [e5, e14]; exact hc₁)
    (by rw [e13, e15]; exact hcx)
    hscalar (Bk := MxxExponentLut.rowOf A) (by rw [e0]; exact hb) heb (by rw [e1]; exact hPe) hPd
    hpre hf hef' hs hes' (by rw [e59]; exact hTH) hM (hnoise i)
  rw [e8, e15] at h
  exact h


/-- The prime-1 CRT term of every part, with the common target `T_i = cxPlain i`.
The linked keys, row forms, fresh and smudge cells discharge the corresponding online premises. -/
theorem part_term1 {outputs : _} (Fon : generatedRoot.Facts Con outputs)
    {paramsK : W_Stage_keys_online_0_0.Params}
    {Ck : W_Stage_keys_online_0_0.generatedRoot.Ctx backend hashModel paramsK} {ok : _}
    (Fk : W_Stage_keys_online_0_0.generatedRoot.Facts Ck ok) (L : Link0 Con Ck)
    (hq1 : 1 < modulus_0) (layout : RegularLayout modulus_0)
    (hlayout : backend.regularLayout modulus_0 8 = some layout) (hexact : layout.droppedModuli = 0)
    {A : ExactMatrix modulus_0 8 2 54}
    (hsolve : ∀ j : Fin 36, A * generatedRoot.w_1277_0 Con j =
      W_Stage_keys_online_0_0.generatedRoot.w_1268_0 Ck j)
    (hP : ∀ j : Fin 36, ∃ Pd : ErrorMatrix 8 54 25,
      generatedRoot.w_1277_0 Con j = reduceMatrix modulus_0 8 54 25 Pd ∧ CoeffBound Pd 22011706)
    {eb : ErrorMatrix 8 1 54} (heb : CoeffBound eb 1)
    (hb : generatedRoot.w_0_0 Con = maskPart (reducePoly modulus_0 8 sm) (MxxExponentLut.rowOf A) +
      reduceMatrix modulus_0 8 1 54 eb)
    (hnoise : ∀ i, I.Noise 1 i)
    (i : Fin 36) (c : parallel_generatedRoot_1281.Ctx backend hashModel params i)
    (hc : c.inputs = Online0Inputs.PartInputs0 Con i) {oc : _}
    (Fc : parallel_generatedRoot_1281.Facts c oc) :
    TermOf modulus_2 (crt0 1)
      (parallel_generatedRoot_1281.w_119_0 c) (I.toOnline0Inputs.partRow i) := by
  have e0 : parallel_generatedRoot_1281.w_0_0 c = generatedRoot.w_0_0 Con := by
    unfold parallel_generatedRoot_1281.w_0_0; rw [hc]
  have e1 : parallel_generatedRoot_1281.w_65_0 c = generatedRoot.w_1277_0 Con i := by
    unfold parallel_generatedRoot_1281.w_65_0; rw [hc]
  have e4 : parallel_generatedRoot_1281.w_4_0 c = generatedRoot.w_63_0 Con := by
    unfold parallel_generatedRoot_1281.w_4_0; rw [hc]
  have e5 : parallel_generatedRoot_1281.w_5_0 c = generatedRoot.w_70_0 Con := by
    unfold parallel_generatedRoot_1281.w_5_0; rw [hc]
  have e8 : parallel_generatedRoot_1281.w_8_0 c = generatedRoot.w_227_0 Con i := by
    unfold parallel_generatedRoot_1281.w_8_0; rw [hc]
  have e13 : parallel_generatedRoot_1281.w_13_0 c = generatedRoot.w_1162_0 Con i := by
    unfold parallel_generatedRoot_1281.w_13_0; rw [hc]
  have e29 : parallel_generatedRoot_1281.w_29_0 c = generatedRoot.w_1219_5 Con := by
    unfold parallel_generatedRoot_1281.w_29_0; rw [hc]
  have e31 : parallel_generatedRoot_1281.w_31_0 c = generatedRoot.w_57_0 Con := by
    unfold parallel_generatedRoot_1281.w_31_0; rw [hc]
  have e47 : parallel_generatedRoot_1281.w_47_0 c = generatedRoot.w_1273_5 Con := by
    unfold parallel_generatedRoot_1281.w_47_0; rw [hc]
  have e49 : parallel_generatedRoot_1281.w_49_0 c = generatedRoot.w_1274_0 Con := by
    unfold parallel_generatedRoot_1281.w_49_0; rw [hc]
  have e59 : parallel_generatedRoot_1281.w_59_0 c = generatedRoot.w_1276_0 Con := by
    unfold parallel_generatedRoot_1281.w_59_0; rw [hc]
  have e14 : parallel_generatedRoot_1281.w_74_0 c = generatedRoot.w_57_0 Con :=
    gadgetMatrixRuns_unique Fc.c3.h74_0 Fon.c1.h57_0
  have e15 : parallel_generatedRoot_1281.w_75_0 c = generatedRoot.w_57_0 Con :=
    (rebase_self hq1 Fc.c3.h75_0).trans e14
  have hG31 : gadgetMatrixRuns backend 4096 25 (parallel_generatedRoot_1281.w_31_0 c) := by
    rw [e31]; exact Fon.c1.h57_0
  obtain ⟨eg, heg, hg⟩ := I.toOnline0Inputs.g_form Fon hq1 layout hlayout hexact
  obtain ⟨e₁, he₁, hc₁⟩ := I.toOnline0Inputs.c1_form Fon hq1 layout hlayout hexact
  obtain ⟨ex, hex, hcx⟩ := I.toOnline0Inputs.cx_rows Fon hq1 layout hlayout hexact i
  rw [← online0_gadget_eq Con Fon] at hcx
  obtain ⟨Pd, hPe, hPd⟩ := hP i
  choose ef hef hz using I.toOnline0Inputs.fresh_cells Fon hq1 layout hlayout hexact
  obtain ⟨ef', hef', hf⟩ := online0_part_fresh1 c Fc i.isLt hq1 layout hlayout hexact hG31
    (K := freshCells0 Con) (t := I.toOnline0Inputs.freshCell) (e := ef)
    (fun cc ↦ by rw [e29, e31]; exact hz cc) hef
  choose es hes hzs using I.toOnline0Inputs.smudge_cells Fon hq1 layout hlayout hexact
  obtain ⟨es', hes', hs⟩ := online0_part_smudge1 c Fc i.isLt hq1 layout hlayout hexact hG31
    (K := smudgeCells0 Con) (t := I.toOnline0Inputs.smudgeCell) (e := es)
    (fun cc ↦ by rw [e47, e31]; exact hzs cc)
    hes (h := I.hconst) (by rw [e49]; exact I.hconst_eq)
  obtain ⟨M, hM, hTH⟩ := I.hTHP 0 i
  have hpre := online0_hpre1_full Fon Fk L hq1 layout hlayout hexact hsolve i c hc Fc
  have hRq : modulus_0 =
      1606938044254565807694577896670997561277593686540426466090209 * modulus_2 := by
    norm_num [modulus_0, modulus_2]
  have hq : 1 < modulus_2 := by norm_num [modulus_2]
  have hqR : modulus_2 ∣ modulus_0 :=
    ⟨1606938044254565807694577896670997561277593686540426466090209,
    by rw [hRq, Nat.mul_comm]⟩
  have hscalar := online0_scalar_inverse hq hq1 hqR 981575778075006 1125894775373377
    (by decide) (I.toOnline0Inputs.cxPlain i)
  have h := online0_part_prime1 c Fc hRq layout hlayout hexact (k := 981575778075006) (by norm_num)
    (s := reducePoly modulus_0 8 sm) (β := reducePoly modulus_0 8 I.toOnline0Inputs.xσ)
    (x := I.toOnline0Inputs.cxPlain i)
    heg he₁ hex (by rw [e4, e14]; exact hg) (by rw [e5, e14]; exact hc₁)
    (by rw [e13, e15]; exact hcx)
    hscalar (Bk := MxxExponentLut.rowOf A) (by rw [e0]; exact hb) heb (by rw [e1]; exact hPe) hPd
    hpre hf hef' hs hes' (by rw [e59]; exact hTH) hM (hnoise i)
  rw [e8, e15] at h
  exact h


/-- The prime-2 CRT term of every part, with the common target `T_i = cxPlain i`.
The linked keys, row forms, fresh and smudge cells discharge the corresponding online premises. -/
theorem part_term2 {outputs : _} (Fon : generatedRoot.Facts Con outputs)
    {paramsK : W_Stage_keys_online_0_0.Params}
    {Ck : W_Stage_keys_online_0_0.generatedRoot.Ctx backend hashModel paramsK} {ok : _}
    (Fk : W_Stage_keys_online_0_0.generatedRoot.Facts Ck ok) (L : Link0 Con Ck)
    (hq1 : 1 < modulus_0) (layout : RegularLayout modulus_0)
    (hlayout : backend.regularLayout modulus_0 8 = some layout) (hexact : layout.droppedModuli = 0)
    {A : ExactMatrix modulus_0 8 2 54}
    (hsolve : ∀ j : Fin 36, A * generatedRoot.w_1278_0 Con j =
      W_Stage_keys_online_0_0.generatedRoot.w_1269_0 Ck j)
    (hP : ∀ j : Fin 36, ∃ Pd : ErrorMatrix 8 54 25,
      generatedRoot.w_1278_0 Con j = reduceMatrix modulus_0 8 54 25 Pd ∧ CoeffBound Pd 22011706)
    {eb : ErrorMatrix 8 1 54} (heb : CoeffBound eb 1)
    (hb : generatedRoot.w_0_0 Con = maskPart (reducePoly modulus_0 8 sm) (MxxExponentLut.rowOf A) +
      reduceMatrix modulus_0 8 1 54 eb)
    (hnoise : ∀ i, I.Noise 2 i)
    (i : Fin 36) (c : parallel_generatedRoot_1281.Ctx backend hashModel params i)
    (hc : c.inputs = Online0Inputs.PartInputs0 Con i) {oc : _}
    (Fc : parallel_generatedRoot_1281.Facts c oc) :
    TermOf modulus_3 (crt0 2)
      (parallel_generatedRoot_1281.w_175_0 c) (I.toOnline0Inputs.partRow i) := by
  have e0 : parallel_generatedRoot_1281.w_0_0 c = generatedRoot.w_0_0 Con := by
    unfold parallel_generatedRoot_1281.w_0_0; rw [hc]
  have e1 : parallel_generatedRoot_1281.w_121_0 c = generatedRoot.w_1278_0 Con i := by
    unfold parallel_generatedRoot_1281.w_121_0; rw [hc]
  have e4 : parallel_generatedRoot_1281.w_4_0 c = generatedRoot.w_63_0 Con := by
    unfold parallel_generatedRoot_1281.w_4_0; rw [hc]
  have e5 : parallel_generatedRoot_1281.w_5_0 c = generatedRoot.w_70_0 Con := by
    unfold parallel_generatedRoot_1281.w_5_0; rw [hc]
  have e8 : parallel_generatedRoot_1281.w_8_0 c = generatedRoot.w_227_0 Con i := by
    unfold parallel_generatedRoot_1281.w_8_0; rw [hc]
  have e13 : parallel_generatedRoot_1281.w_13_0 c = generatedRoot.w_1162_0 Con i := by
    unfold parallel_generatedRoot_1281.w_13_0; rw [hc]
  have e29 : parallel_generatedRoot_1281.w_29_0 c = generatedRoot.w_1219_5 Con := by
    unfold parallel_generatedRoot_1281.w_29_0; rw [hc]
  have e31 : parallel_generatedRoot_1281.w_31_0 c = generatedRoot.w_57_0 Con := by
    unfold parallel_generatedRoot_1281.w_31_0; rw [hc]
  have e47 : parallel_generatedRoot_1281.w_47_0 c = generatedRoot.w_1273_5 Con := by
    unfold parallel_generatedRoot_1281.w_47_0; rw [hc]
  have e49 : parallel_generatedRoot_1281.w_49_0 c = generatedRoot.w_1274_0 Con := by
    unfold parallel_generatedRoot_1281.w_49_0; rw [hc]
  have e59 : parallel_generatedRoot_1281.w_59_0 c = generatedRoot.w_1276_0 Con := by
    unfold parallel_generatedRoot_1281.w_59_0; rw [hc]
  have e14 : parallel_generatedRoot_1281.w_130_0 c = generatedRoot.w_57_0 Con :=
    gadgetMatrixRuns_unique Fc.c6.h130_0 Fon.c1.h57_0
  have e15 : parallel_generatedRoot_1281.w_131_0 c = generatedRoot.w_57_0 Con :=
    (rebase_self hq1 Fc.c6.h131_0).trans e14
  have hG31 : gadgetMatrixRuns backend 4096 25 (parallel_generatedRoot_1281.w_31_0 c) := by
    rw [e31]; exact Fon.c1.h57_0
  obtain ⟨eg, heg, hg⟩ := I.toOnline0Inputs.g_form Fon hq1 layout hlayout hexact
  obtain ⟨e₁, he₁, hc₁⟩ := I.toOnline0Inputs.c1_form Fon hq1 layout hlayout hexact
  obtain ⟨ex, hex, hcx⟩ := I.toOnline0Inputs.cx_rows Fon hq1 layout hlayout hexact i
  rw [← online0_gadget_eq Con Fon] at hcx
  obtain ⟨Pd, hPe, hPd⟩ := hP i
  choose ef hef hz using I.toOnline0Inputs.fresh_cells Fon hq1 layout hlayout hexact
  obtain ⟨ef', hef', hf⟩ := online0_part_fresh2 c Fc i.isLt hq1 layout hlayout hexact hG31
    (K := freshCells0 Con) (t := I.toOnline0Inputs.freshCell) (e := ef)
    (fun cc ↦ by rw [e29, e31]; exact hz cc) hef
  choose es hes hzs using I.toOnline0Inputs.smudge_cells Fon hq1 layout hlayout hexact
  obtain ⟨es', hes', hs⟩ := online0_part_smudge2 c Fc i.isLt hq1 layout hlayout hexact hG31
    (K := smudgeCells0 Con) (t := I.toOnline0Inputs.smudgeCell) (e := es)
    (fun cc ↦ by rw [e47, e31]; exact hzs cc)
    hes (h := I.hconst) (by rw [e49]; exact I.hconst_eq)
  obtain ⟨M, hM, hTH⟩ := I.hTHP 1 i
  have hpre := online0_hpre2_full Fon Fk L hq1 layout hlayout hexact hsolve i c hc Fc
  have hRq : modulus_0 =
      1606938044254771331362327096896778694705812889646799470123121 * modulus_3 := by
    norm_num [modulus_0, modulus_3]
  have hq : 1 < modulus_3 := by norm_num [modulus_3]
  have hqR : modulus_3 ∣ modulus_0 :=
    ⟨1606938044254771331362327096896778694705812889646799470123121,
    by rw [hRq, Nat.mul_comm]⟩
  have hscalar := online0_scalar_inverse hq hq1 hqR 144137876894272 7254835200
    (by decide) (I.toOnline0Inputs.cxPlain i)
  have h := online0_part_prime2 c Fc hRq layout hlayout hexact (k := 144137876894272) (by norm_num)
    (s := reducePoly modulus_0 8 sm) (β := reducePoly modulus_0 8 I.toOnline0Inputs.xσ)
    (x := I.toOnline0Inputs.cxPlain i)
    heg he₁ hex (by rw [e4, e14]; exact hg) (by rw [e5, e14]; exact hc₁)
    (by rw [e13, e15]; exact hcx)
    hscalar (Bk := MxxExponentLut.rowOf A) (by rw [e0]; exact hb) heb (by rw [e1]; exact hPe) hPd
    hpre hf hef' hs hes' (by rw [e59]; exact hTH) hM (hnoise i)
  rw [e8, e15] at h
  exact h


/-- The prime-3 CRT term of every part, with the common target `T_i = cxPlain i`.
The linked keys, row forms, fresh and smudge cells discharge the corresponding online premises. -/
theorem part_term3 {outputs : _} (Fon : generatedRoot.Facts Con outputs)
    {paramsK : W_Stage_keys_online_0_0.Params}
    {Ck : W_Stage_keys_online_0_0.generatedRoot.Ctx backend hashModel paramsK} {ok : _}
    (Fk : W_Stage_keys_online_0_0.generatedRoot.Facts Ck ok) (L : Link0 Con Ck)
    (hq1 : 1 < modulus_0) (layout : RegularLayout modulus_0)
    (hlayout : backend.regularLayout modulus_0 8 = some layout) (hexact : layout.droppedModuli = 0)
    {A : ExactMatrix modulus_0 8 2 54}
    (hsolve : ∀ j : Fin 36, A * generatedRoot.w_1279_0 Con j =
      W_Stage_keys_online_0_0.generatedRoot.w_1270_0 Ck j)
    (hP : ∀ j : Fin 36, ∃ Pd : ErrorMatrix 8 54 25,
      generatedRoot.w_1279_0 Con j = reduceMatrix modulus_0 8 54 25 Pd ∧ CoeffBound Pd 22011706)
    {eb : ErrorMatrix 8 1 54} (heb : CoeffBound eb 1)
    (hb : generatedRoot.w_0_0 Con = maskPart (reducePoly modulus_0 8 sm) (MxxExponentLut.rowOf A) +
      reduceMatrix modulus_0 8 1 54 eb)
    (hnoise : ∀ i, I.Noise 3 i)
    (i : Fin 36) (c : parallel_generatedRoot_1281.Ctx backend hashModel params i)
    (hc : c.inputs = Online0Inputs.PartInputs0 Con i) {oc : _}
    (Fc : parallel_generatedRoot_1281.Facts c oc) :
    TermOf modulus_4 (crt0 3)
      (parallel_generatedRoot_1281.w_231_0 c) (I.toOnline0Inputs.partRow i) := by
  have e0 : parallel_generatedRoot_1281.w_0_0 c = generatedRoot.w_0_0 Con := by
    unfold parallel_generatedRoot_1281.w_0_0; rw [hc]
  have e1 : parallel_generatedRoot_1281.w_177_0 c = generatedRoot.w_1279_0 Con i := by
    unfold parallel_generatedRoot_1281.w_177_0; rw [hc]
  have e4 : parallel_generatedRoot_1281.w_4_0 c = generatedRoot.w_63_0 Con := by
    unfold parallel_generatedRoot_1281.w_4_0; rw [hc]
  have e5 : parallel_generatedRoot_1281.w_5_0 c = generatedRoot.w_70_0 Con := by
    unfold parallel_generatedRoot_1281.w_5_0; rw [hc]
  have e8 : parallel_generatedRoot_1281.w_8_0 c = generatedRoot.w_227_0 Con i := by
    unfold parallel_generatedRoot_1281.w_8_0; rw [hc]
  have e13 : parallel_generatedRoot_1281.w_13_0 c = generatedRoot.w_1162_0 Con i := by
    unfold parallel_generatedRoot_1281.w_13_0; rw [hc]
  have e29 : parallel_generatedRoot_1281.w_29_0 c = generatedRoot.w_1219_5 Con := by
    unfold parallel_generatedRoot_1281.w_29_0; rw [hc]
  have e31 : parallel_generatedRoot_1281.w_31_0 c = generatedRoot.w_57_0 Con := by
    unfold parallel_generatedRoot_1281.w_31_0; rw [hc]
  have e47 : parallel_generatedRoot_1281.w_47_0 c = generatedRoot.w_1273_5 Con := by
    unfold parallel_generatedRoot_1281.w_47_0; rw [hc]
  have e49 : parallel_generatedRoot_1281.w_49_0 c = generatedRoot.w_1274_0 Con := by
    unfold parallel_generatedRoot_1281.w_49_0; rw [hc]
  have e59 : parallel_generatedRoot_1281.w_59_0 c = generatedRoot.w_1276_0 Con := by
    unfold parallel_generatedRoot_1281.w_59_0; rw [hc]
  have e14 : parallel_generatedRoot_1281.w_186_0 c = generatedRoot.w_57_0 Con :=
    gadgetMatrixRuns_unique Fc.c8.h186_0 Fon.c1.h57_0
  have e15 : parallel_generatedRoot_1281.w_187_0 c = generatedRoot.w_57_0 Con :=
    (rebase_self hq1 Fc.c8.h187_0).trans e14
  have hG31 : gadgetMatrixRuns backend 4096 25 (parallel_generatedRoot_1281.w_31_0 c) := by
    rw [e31]; exact Fon.c1.h57_0
  obtain ⟨eg, heg, hg⟩ := I.toOnline0Inputs.g_form Fon hq1 layout hlayout hexact
  obtain ⟨e₁, he₁, hc₁⟩ := I.toOnline0Inputs.c1_form Fon hq1 layout hlayout hexact
  obtain ⟨ex, hex, hcx⟩ := I.toOnline0Inputs.cx_rows Fon hq1 layout hlayout hexact i
  rw [← online0_gadget_eq Con Fon] at hcx
  obtain ⟨Pd, hPe, hPd⟩ := hP i
  choose ef hef hz using I.toOnline0Inputs.fresh_cells Fon hq1 layout hlayout hexact
  obtain ⟨ef', hef', hf⟩ := online0_part_fresh3 c Fc i.isLt hq1 layout hlayout hexact hG31
    (K := freshCells0 Con) (t := I.toOnline0Inputs.freshCell) (e := ef)
    (fun cc ↦ by rw [e29, e31]; exact hz cc) hef
  choose es hes hzs using I.toOnline0Inputs.smudge_cells Fon hq1 layout hlayout hexact
  obtain ⟨es', hes', hs⟩ := online0_part_smudge3 c Fc i.isLt hq1 layout hlayout hexact hG31
    (K := smudgeCells0 Con) (t := I.toOnline0Inputs.smudgeCell) (e := es)
    (fun cc ↦ by rw [e47, e31]; exact hzs cc)
    hes (h := I.hconst) (by rw [e49]; exact I.hconst_eq)
  obtain ⟨M, hM, hTH⟩ := I.hTHP 2 i
  have hpre := online0_hpre3_full Fon Fk L hq1 layout hlayout hexact hsolve i c hc Fc
  have hRq : modulus_0 =
      1606938044255228050623992174508164557277769211170177223088561 * modulus_4 := by
    norm_num [modulus_0, modulus_4]
  have hq : 1 < modulus_4 := by norm_num [modulus_4]
  have hqR : modulus_4 ∣ modulus_0 :=
    ⟨1606938044255228050623992174508164557277769211170177223088561,
    by rw [hRq, Nat.mul_comm]⟩
  have hscalar := online0_scalar_inverse hq hq1 hqR 178232537817382 1125871968844913
    (by decide) (I.toOnline0Inputs.cxPlain i)
  have h := online0_part_prime3 c Fc hRq layout hlayout hexact (k := 178232537817382) (by norm_num)
    (s := reducePoly modulus_0 8 sm) (β := reducePoly modulus_0 8 I.toOnline0Inputs.xσ)
    (x := I.toOnline0Inputs.cxPlain i)
    heg he₁ hex (by rw [e4, e14]; exact hg) (by rw [e5, e14]; exact hc₁)
    (by rw [e13, e15]; exact hcx)
    hscalar (Bk := MxxExponentLut.rowOf A) (by rw [e0]; exact hb) heb (by rw [e1]; exact hPe) hPd
    hpre hf hef' hs hes' (by rw [e59]; exact hTH) hM (hnoise i)
  rw [e8, e15] at h
  exact h


/-- The prime-4 CRT term of every part, with the common target `T_i = cxPlain i`.
The linked keys, row forms, fresh and smudge cells discharge the corresponding online premises. -/
theorem part_term4 {outputs : _} (Fon : generatedRoot.Facts Con outputs)
    {paramsK : W_Stage_keys_online_0_0.Params}
    {Ck : W_Stage_keys_online_0_0.generatedRoot.Ctx backend hashModel paramsK} {ok : _}
    (Fk : W_Stage_keys_online_0_0.generatedRoot.Facts Ck ok) (L : Link0 Con Ck)
    (hq1 : 1 < modulus_0) (layout : RegularLayout modulus_0)
    (hlayout : backend.regularLayout modulus_0 8 = some layout) (hexact : layout.droppedModuli = 0)
    {A : ExactMatrix modulus_0 8 2 54}
    (hsolve : ∀ j : Fin 36, A * generatedRoot.w_1280_0 Con j =
      W_Stage_keys_online_0_0.generatedRoot.w_1271_0 Ck j)
    (hP : ∀ j : Fin 36, ∃ Pd : ErrorMatrix 8 54 25,
      generatedRoot.w_1280_0 Con j = reduceMatrix modulus_0 8 54 25 Pd ∧ CoeffBound Pd 22011706)
    {eb : ErrorMatrix 8 1 54} (heb : CoeffBound eb 1)
    (hb : generatedRoot.w_0_0 Con = maskPart (reducePoly modulus_0 8 sm) (MxxExponentLut.rowOf A) +
      reduceMatrix modulus_0 8 1 54 eb)
    (hnoise : ∀ i, I.Noise 4 i)
    (i : Fin 36) (c : parallel_generatedRoot_1281.Ctx backend hashModel params i)
    (hc : c.inputs = Online0Inputs.PartInputs0 Con i) {oc : _}
    (Fc : parallel_generatedRoot_1281.Facts c oc) :
    TermOf modulus_5 (crt0 4)
      (parallel_generatedRoot_1281.w_287_0 c) (I.toOnline0Inputs.partRow i) := by
  have e0 : parallel_generatedRoot_1281.w_0_0 c = generatedRoot.w_0_0 Con := by
    unfold parallel_generatedRoot_1281.w_0_0; rw [hc]
  have e1 : parallel_generatedRoot_1281.w_233_0 c = generatedRoot.w_1280_0 Con i := by
    unfold parallel_generatedRoot_1281.w_233_0; rw [hc]
  have e4 : parallel_generatedRoot_1281.w_4_0 c = generatedRoot.w_63_0 Con := by
    unfold parallel_generatedRoot_1281.w_4_0; rw [hc]
  have e5 : parallel_generatedRoot_1281.w_5_0 c = generatedRoot.w_70_0 Con := by
    unfold parallel_generatedRoot_1281.w_5_0; rw [hc]
  have e8 : parallel_generatedRoot_1281.w_8_0 c = generatedRoot.w_227_0 Con i := by
    unfold parallel_generatedRoot_1281.w_8_0; rw [hc]
  have e13 : parallel_generatedRoot_1281.w_13_0 c = generatedRoot.w_1162_0 Con i := by
    unfold parallel_generatedRoot_1281.w_13_0; rw [hc]
  have e29 : parallel_generatedRoot_1281.w_29_0 c = generatedRoot.w_1219_5 Con := by
    unfold parallel_generatedRoot_1281.w_29_0; rw [hc]
  have e31 : parallel_generatedRoot_1281.w_31_0 c = generatedRoot.w_57_0 Con := by
    unfold parallel_generatedRoot_1281.w_31_0; rw [hc]
  have e47 : parallel_generatedRoot_1281.w_47_0 c = generatedRoot.w_1273_5 Con := by
    unfold parallel_generatedRoot_1281.w_47_0; rw [hc]
  have e49 : parallel_generatedRoot_1281.w_49_0 c = generatedRoot.w_1274_0 Con := by
    unfold parallel_generatedRoot_1281.w_49_0; rw [hc]
  have e59 : parallel_generatedRoot_1281.w_59_0 c = generatedRoot.w_1276_0 Con := by
    unfold parallel_generatedRoot_1281.w_59_0; rw [hc]
  have e14 : parallel_generatedRoot_1281.w_242_0 c = generatedRoot.w_57_0 Con :=
    gadgetMatrixRuns_unique Fc.c11.h242_0 Fon.c1.h57_0
  have e15 : parallel_generatedRoot_1281.w_243_0 c = generatedRoot.w_57_0 Con :=
    (rebase_self hq1 Fc.c11.h243_0).trans e14
  have hG31 : gadgetMatrixRuns backend 4096 25 (parallel_generatedRoot_1281.w_31_0 c) := by
    rw [e31]; exact Fon.c1.h57_0
  obtain ⟨eg, heg, hg⟩ := I.toOnline0Inputs.g_form Fon hq1 layout hlayout hexact
  obtain ⟨e₁, he₁, hc₁⟩ := I.toOnline0Inputs.c1_form Fon hq1 layout hlayout hexact
  obtain ⟨ex, hex, hcx⟩ := I.toOnline0Inputs.cx_rows Fon hq1 layout hlayout hexact i
  rw [← online0_gadget_eq Con Fon] at hcx
  obtain ⟨Pd, hPe, hPd⟩ := hP i
  choose ef hef hz using I.toOnline0Inputs.fresh_cells Fon hq1 layout hlayout hexact
  obtain ⟨ef', hef', hf⟩ := online0_part_fresh4 c Fc i.isLt hq1 layout hlayout hexact hG31
    (K := freshCells0 Con) (t := I.toOnline0Inputs.freshCell) (e := ef)
    (fun cc ↦ by rw [e29, e31]; exact hz cc) hef
  choose es hes hzs using I.toOnline0Inputs.smudge_cells Fon hq1 layout hlayout hexact
  obtain ⟨es', hes', hs⟩ := online0_part_smudge4 c Fc i.isLt hq1 layout hlayout hexact hG31
    (K := smudgeCells0 Con) (t := I.toOnline0Inputs.smudgeCell) (e := es)
    (fun cc ↦ by rw [e47, e31]; exact hzs cc)
    hes (h := I.hconst) (by rw [e49]; exact I.hconst_eq)
  obtain ⟨M, hM, hTH⟩ := I.hTHP 3 i
  have hpre := online0_hpre4_full Fon Fk L hq1 layout hlayout hexact hsolve i c hc Fc
  have hRq : modulus_0 =
      1606938044255707605848740785410594412685698358681526360823809 * modulus_5 := by
    norm_num [modulus_0, modulus_5]
  have hq : 1 < modulus_5 := by norm_num [modulus_5]
  have hqR : modulus_5 ∣ modulus_0 :=
    ⟨1606938044255707605848740785410594412685698358681526360823809,
    by rw [hRq, Nat.mul_comm]⟩
  have hscalar := online0_scalar_inverse hq hq1 hqR 185888706444063 157994188800
    (by decide) (I.toOnline0Inputs.cxPlain i)
  have h := online0_part_prime4 c Fc hRq layout hlayout hexact (k := 185888706444063) (by norm_num)
    (s := reducePoly modulus_0 8 sm) (β := reducePoly modulus_0 8 I.toOnline0Inputs.xσ)
    (x := I.toOnline0Inputs.cxPlain i)
    heg he₁ hex (by rw [e4, e14]; exact hg) (by rw [e5, e14]; exact hc₁)
    (by rw [e13, e15]; exact hcx)
    hscalar (Bk := MxxExponentLut.rowOf A) (by rw [e0]; exact hb) heb (by rw [e1]; exact hPe) hPd
    hpre hf hef' hs hes' (by rw [e59]; exact hTH) hM (hnoise i)
  rw [e8, e15] at h
  exact h


end Online0InputsP

/-- All five CRT terms of every refresh part, at the claim level, with one common intended row.
The claim discharges hint and bounded preimage premises, linked key agreement, gadget agreement,
cell assembly, and CRT scalar identities. `Online0InputsP` and the noise inequalities remain. -/
theorem claim_online0_terms {hashModel : HashModel} {external : GeneratedClaim.ExternalInputs}
    {execution : GeneratedClaim.Execution}
    (h : W_GeneratedClaim.Runs hashModel external execution) {sm : ErrorPoly 8}
    (hmask : execution.«stage_3» 0 0 = reducePoly W_Stage_initial.modulus_0 8 sm)
    (hq1 : 1 < modulus_0) (layout : RegularLayout modulus_0)
    (hlayout : Backend.backend.regularLayout modulus_0 8 = some layout)
    (hexact : layout.droppedModuli = 0) :
    ∃ Con : generatedRoot.Ctx Backend.backend hashModel W_GeneratedClaim.stage_11_params,
      generatedRoot.Facts Con execution.«stage_11» ∧
      ∀ (I : Online0InputsP Con sm) (p : Fin 5), (∀ i, I.Noise p i) →
      ∀ (i : Fin 36) (c : parallel_generatedRoot_1281.Ctx Backend.backend hashModel
          W_GeneratedClaim.stage_11_params i),
        c.inputs = Online0Inputs.PartInputs0 Con i →
        ∀ {oc : _}, parallel_generatedRoot_1281.Facts c oc →
        TermOf (primes0 p) (crt0 p) (online0PrimeTerm c p)
          (I.toOnline0Inputs.partRow i) := by
  have hp := (claim_preimages h).1
  obtain ⟨eb, heb, hb⟩ := claim_hint0 h hmask
  obtain ⟨-, -, -, -, -, -, -, -, -, h8, -, -, h11, -⟩ := h
  obtain ⟨wk, hk⟩ := h8
  obtain ⟨wo, ho⟩ := h11
  let Con : generatedRoot.Ctx Backend.backend hashModel W_GeneratedClaim.stage_11_params :=
    (fun inputs (_ : generatedRoot.body Backend.backend hashModel
      W_GeneratedClaim.stage_11_params inputs execution.«stage_11» wo) ↦ ⟨inputs, wo⟩) _ ho
  let Ck : W_Stage_keys_online_0_0.generatedRoot.Ctx Backend.backend hashModel
      W_GeneratedClaim.stage_8_params :=
    (fun inputs (_ : W_Stage_keys_online_0_0.generatedRoot.body Backend.backend hashModel
      W_GeneratedClaim.stage_8_params inputs execution.«stage_8» wk) ↦ ⟨inputs, wk⟩) _ hk
  have Fo := generatedRoot.facts Con ho
  have Fk := W_Stage_keys_online_0_0.generatedRoot.facts Ck hk
  have hsolve : ∀ (p : Fin 5) (j : Fin 36),
      execution.«stage_1».1 * online0PrimePreimage Con p j =
        online0PrimeKeyImage Ck p j := by
    intro p j
    fin_cases p
    · have equation := (hp.1 j).1
      rw [Fk.c67.hout] at equation
      exact equation
    · have equation := (hp.2.1 j).1
      rw [Fk.c67.hout] at equation
      exact equation
    · have equation := (hp.2.2.1 j).1
      rw [Fk.c67.hout] at equation
      exact equation
    · have equation := (hp.2.2.2.1 j).1
      rw [Fk.c67.hout] at equation
      exact equation
    · have equation := (hp.2.2.2.2 j).1
      rw [Fk.c67.hout] at equation
      exact equation
  have hP : ∀ (p : Fin 5) (j : Fin 36), ∃ Pd : ErrorMatrix 8 54 25,
      online0PrimePreimage Con p j = reduceMatrix modulus_0 8 54 25 Pd ∧
        CoeffBound Pd 22011706 := by
    intro p j
    fin_cases p
    · exact (hp.1 j).2
    · exact (hp.2.1 j).2
    · exact (hp.2.2.1 j).2
    · exact (hp.2.2.2.1 j).2
    · exact (hp.2.2.2.2 j).2
  refine ⟨Con, Fo, fun I p hnoise i c hc _ Fc ↦ ?_⟩
  fin_cases p
  · exact I.part_term0 Fo Fk (by constructor <;> rfl) hq1 layout hlayout hexact
      (hsolve 0) (hP 0) heb hb hnoise i c hc Fc
  · exact I.part_term1 Fo Fk (by constructor <;> rfl) hq1 layout hlayout hexact
      (hsolve 1) (hP 1) heb hb hnoise i c hc Fc
  · exact I.part_term2 Fo Fk (by constructor <;> rfl) hq1 layout hlayout hexact
      (hsolve 2) (hP 2) heb hb hnoise i c hc Fc
  · exact I.part_term3 Fo Fk (by constructor <;> rfl) hq1 layout hlayout hexact
      (hsolve 3) (hP 3) heb hb hnoise i c hc Fc
  · exact I.part_term4 Fo Fk (by constructor <;> rfl) hq1 layout hlayout hexact
      (hsolve 4) (hP 4) heb hb hnoise i c hc Fc

/-- The prime-1 CRT term of every step-zero refresh part under the generated claim.
This specializes `claim_online0_terms` to the prime's actual output wire. -/
theorem claim_online0_term1 {hashModel : HashModel}
    {external : GeneratedClaim.ExternalInputs} {execution : GeneratedClaim.Execution}
    (h : W_GeneratedClaim.Runs hashModel external execution) {sm : ErrorPoly 8}
    (hmask : execution.«stage_3» 0 0 = reducePoly W_Stage_initial.modulus_0 8 sm)
    (hq1 : 1 < modulus_0) (layout : RegularLayout modulus_0)
    (hlayout : Backend.backend.regularLayout modulus_0 8 = some layout)
    (hexact : layout.droppedModuli = 0) :
    ∃ Con : generatedRoot.Ctx Backend.backend hashModel W_GeneratedClaim.stage_11_params,
      generatedRoot.Facts Con execution.«stage_11» ∧
      ∀ (I : Online0InputsP Con sm), (∀ i, I.Noise 1 i) →
      ∀ (i : Fin 36) (c : parallel_generatedRoot_1281.Ctx Backend.backend hashModel
          W_GeneratedClaim.stage_11_params i),
        c.inputs = Online0Inputs.PartInputs0 Con i →
        ∀ {oc : _}, parallel_generatedRoot_1281.Facts c oc →
        TermOf modulus_2 (crt0 1) (parallel_generatedRoot_1281.w_119_0 c)
          (I.toOnline0Inputs.partRow i) := by
  obtain ⟨Con, Fo, terms⟩ := claim_online0_terms h hmask hq1 layout hlayout hexact
  exact ⟨Con, Fo, fun I hnoise i c hc _ Fc ↦ terms I 1 hnoise i c hc Fc⟩

/-- The prime-2 CRT term of every step-zero refresh part under the generated claim.
This specializes `claim_online0_terms` to the prime's actual output wire. -/
theorem claim_online0_term2 {hashModel : HashModel}
    {external : GeneratedClaim.ExternalInputs} {execution : GeneratedClaim.Execution}
    (h : W_GeneratedClaim.Runs hashModel external execution) {sm : ErrorPoly 8}
    (hmask : execution.«stage_3» 0 0 = reducePoly W_Stage_initial.modulus_0 8 sm)
    (hq1 : 1 < modulus_0) (layout : RegularLayout modulus_0)
    (hlayout : Backend.backend.regularLayout modulus_0 8 = some layout)
    (hexact : layout.droppedModuli = 0) :
    ∃ Con : generatedRoot.Ctx Backend.backend hashModel W_GeneratedClaim.stage_11_params,
      generatedRoot.Facts Con execution.«stage_11» ∧
      ∀ (I : Online0InputsP Con sm), (∀ i, I.Noise 2 i) →
      ∀ (i : Fin 36) (c : parallel_generatedRoot_1281.Ctx Backend.backend hashModel
          W_GeneratedClaim.stage_11_params i),
        c.inputs = Online0Inputs.PartInputs0 Con i →
        ∀ {oc : _}, parallel_generatedRoot_1281.Facts c oc →
        TermOf modulus_3 (crt0 2) (parallel_generatedRoot_1281.w_175_0 c)
          (I.toOnline0Inputs.partRow i) := by
  obtain ⟨Con, Fo, terms⟩ := claim_online0_terms h hmask hq1 layout hlayout hexact
  exact ⟨Con, Fo, fun I hnoise i c hc _ Fc ↦ terms I 2 hnoise i c hc Fc⟩

/-- The prime-3 CRT term of every step-zero refresh part under the generated claim.
This specializes `claim_online0_terms` to the prime's actual output wire. -/
theorem claim_online0_term3 {hashModel : HashModel}
    {external : GeneratedClaim.ExternalInputs} {execution : GeneratedClaim.Execution}
    (h : W_GeneratedClaim.Runs hashModel external execution) {sm : ErrorPoly 8}
    (hmask : execution.«stage_3» 0 0 = reducePoly W_Stage_initial.modulus_0 8 sm)
    (hq1 : 1 < modulus_0) (layout : RegularLayout modulus_0)
    (hlayout : Backend.backend.regularLayout modulus_0 8 = some layout)
    (hexact : layout.droppedModuli = 0) :
    ∃ Con : generatedRoot.Ctx Backend.backend hashModel W_GeneratedClaim.stage_11_params,
      generatedRoot.Facts Con execution.«stage_11» ∧
      ∀ (I : Online0InputsP Con sm), (∀ i, I.Noise 3 i) →
      ∀ (i : Fin 36) (c : parallel_generatedRoot_1281.Ctx Backend.backend hashModel
          W_GeneratedClaim.stage_11_params i),
        c.inputs = Online0Inputs.PartInputs0 Con i →
        ∀ {oc : _}, parallel_generatedRoot_1281.Facts c oc →
        TermOf modulus_4 (crt0 3) (parallel_generatedRoot_1281.w_231_0 c)
          (I.toOnline0Inputs.partRow i) := by
  obtain ⟨Con, Fo, terms⟩ := claim_online0_terms h hmask hq1 layout hlayout hexact
  exact ⟨Con, Fo, fun I hnoise i c hc _ Fc ↦ terms I 3 hnoise i c hc Fc⟩

/-- The prime-4 CRT term of every step-zero refresh part under the generated claim.
This specializes `claim_online0_terms` to the prime's actual output wire. -/
theorem claim_online0_term4 {hashModel : HashModel}
    {external : GeneratedClaim.ExternalInputs} {execution : GeneratedClaim.Execution}
    (h : W_GeneratedClaim.Runs hashModel external execution) {sm : ErrorPoly 8}
    (hmask : execution.«stage_3» 0 0 = reducePoly W_Stage_initial.modulus_0 8 sm)
    (hq1 : 1 < modulus_0) (layout : RegularLayout modulus_0)
    (hlayout : Backend.backend.regularLayout modulus_0 8 = some layout)
    (hexact : layout.droppedModuli = 0) :
    ∃ Con : generatedRoot.Ctx Backend.backend hashModel W_GeneratedClaim.stage_11_params,
      generatedRoot.Facts Con execution.«stage_11» ∧
      ∀ (I : Online0InputsP Con sm), (∀ i, I.Noise 4 i) →
      ∀ (i : Fin 36) (c : parallel_generatedRoot_1281.Ctx Backend.backend hashModel
          W_GeneratedClaim.stage_11_params i),
        c.inputs = Online0Inputs.PartInputs0 Con i →
        ∀ {oc : _}, parallel_generatedRoot_1281.Facts c oc →
        TermOf modulus_5 (crt0 4) (parallel_generatedRoot_1281.w_287_0 c)
          (I.toOnline0Inputs.partRow i) := by
  obtain ⟨Con, Fo, terms⟩ := claim_online0_terms h hmask hq1 layout hlayout hexact
  exact ⟨Con, Fo, fun I hnoise i c hc _ Fc ↦ terms I 4 hnoise i c hc Fc⟩

/-- Every step-zero refresh part returns its full CRT row
`β A'_{i,0} + A'_{i,1} - T_i G' + E_i`, with `T_i = reducePoly Q_0 8 (cxPlain i)`.
CRT recombination adds no hypotheses beyond the assembled prime terms. -/
theorem claim_online0_row {hashModel : HashModel} {external : GeneratedClaim.ExternalInputs}
    {execution : GeneratedClaim.Execution}
    (h : W_GeneratedClaim.Runs hashModel external execution) {sm : ErrorPoly 8}
    (hmask : execution.«stage_3» 0 0 = reducePoly W_Stage_initial.modulus_0 8 sm)
    (hq1 : 1 < modulus_0) (layout : RegularLayout modulus_0)
    (hlayout : Backend.backend.regularLayout modulus_0 8 = some layout)
    (hexact : layout.droppedModuli = 0) :
    ∃ Con : generatedRoot.Ctx Backend.backend hashModel W_GeneratedClaim.stage_11_params,
      generatedRoot.Facts Con execution.«stage_11» ∧
      ∀ (I : Online0InputsP Con sm), (∀ p i, I.Noise p i) →
      ∀ (i : Fin 36) (c : parallel_generatedRoot_1281.Ctx Backend.backend hashModel
          W_GeneratedClaim.stage_11_params i),
        c.inputs = Online0Inputs.PartInputs0 Con i →
        ∀ {oc : _}, parallel_generatedRoot_1281.Facts c oc →
        parallel_generatedRoot_1281.w_288_0 c =
          reducePoly modulus_0 8 I.toOnline0Inputs.xσ •
              MxxExponentLut.rowOf (generatedRoot.w_227_0 Con i) 0 +
            MxxExponentLut.rowOf (generatedRoot.w_227_0 Con i) 1 -
            I.toOnline0Inputs.T i • generatedRoot.w_57_0 Con +
            reduceMatrix modulus_0 8 1 25 (I.toOnline0Inputs.E0 i) := by
  obtain ⟨Con, Fo, terms⟩ := claim_online0_terms h hmask hq1 layout hlayout hexact
  refine ⟨Con, Fo, fun I hnoise i c hc _ Fc ↦ ?_⟩
  have allTerms := fun p ↦ terms I p (hnoise p) i c hc Fc
  exact online0_part_row c (allTerms 0) (allTerms 1) (allTerms 2) (allTerms 3)
    (allTerms 4)

/-- The actual online stage output family satisfies the assembled full-row identity.
The generated scope facts discharge the part-context wiring and execution hypotheses. -/
theorem claim_online0_rows {hashModel : HashModel} {external : GeneratedClaim.ExternalInputs}
    {execution : GeneratedClaim.Execution}
    (h : W_GeneratedClaim.Runs hashModel external execution) {sm : ErrorPoly 8}
    (hmask : execution.«stage_3» 0 0 = reducePoly W_Stage_initial.modulus_0 8 sm)
    (hq1 : 1 < modulus_0) (layout : RegularLayout modulus_0)
    (hlayout : Backend.backend.regularLayout modulus_0 8 = some layout)
    (hexact : layout.droppedModuli = 0) :
    ∃ Con : generatedRoot.Ctx Backend.backend hashModel W_GeneratedClaim.stage_11_params,
      generatedRoot.Facts Con execution.«stage_11» ∧
      ∀ (I : Online0InputsP Con sm), (∀ p i, I.Noise p i) →
      ∀ i : Fin 36, generatedRoot.w_1281_0 Con i = I.toOnline0Inputs.partRow i := by
  obtain ⟨Con, Fo, rows⟩ := claim_online0_row h hmask hq1 layout hlayout hexact
  refine ⟨Con, Fo, fun I hnoise i ↦ ?_⟩
  obtain ⟨witness, facts⟩ := generatedRoot.h1281_0_at Fo i
  have row := rows I hnoise i ⟨Online0Inputs.PartInputs0 Con i, witness⟩ rfl facts
  exact facts.c14.hout.trans row

end Dio2Proof
