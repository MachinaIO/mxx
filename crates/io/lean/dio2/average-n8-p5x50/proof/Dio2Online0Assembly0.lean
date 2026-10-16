import Dio2KeyAgree0

/-!
# Step 0, prime 0: the refresh parts assembled

For every refresh part `i : Fin 36` of the online stage of step 0 (`parallel_generatedRoot_1281`),
`online0_part_prime0` (`Dio2Online0PartPrimes`, generic `prime_term`) is applied with all its
hypotheses discharged from the proven online pieces:

* `hg`: `online0_g` on `online0_sigma_total` (`Online0Inputs.g_form`);
* `hc₁`: `online0_c1` (`Online0Inputs.c1_form`);
* `hcx`: member `i` of the 36-family `w_1162` (`Online0Inputs.cx_rows`): `online0_target_priv`,
  `online0_target_pub`, `online0_state_lut`/`online0_target_state`, `online0_label_*`,
  `online0_target_sigma`, `online0_aut_*` and the four zero members, with the online keys
  `cxKeys0`;
* `hpre`: `online0_hpre0_full` (`Dio2KeyAgree0`);
* `hf`, `hs`: `online0_part_fresh0`/`online0_part_smudge0` on the cell forms of
  `online0_fresh_row`/`online0_smudge_row` (`cell_form`);
* `hb`, `hP`: `claim_hint0` and `claim_preimages` (at the claim level);
* `hscalar`: by the choice `T_i = k x_i s₉ s₁₀` (`Online0Inputs.T0`).

What the online proof does not derive (encoding relations of the stage inputs, switching
ciphertexts, LUT identities and norms, label values, the smudge constant and residue) is grouped
in `Online0Inputs`, each field in the exact shape of the consuming online lemma; the noise
condition is `Online0Inputs.Noise0` with the explicit bounds of the derivation.

* `Online0Inputs.part_term0`: the stage-level statement;
* `claim_online0_term0`: the claim-level statement under `W_GeneratedClaim.Runs`.
-/

set_option maxHeartbeats 32000000
set_option maxRecDepth 16384

namespace Dio2Proof

open Mxx.Primitives MxxRuntime MxxExponentLut Mxx.ExponentLut.Refresh RefreshTopProof
  RefreshTopProof.Gen RefreshTopProof.GenIface RefreshTopProof.GenInst RefreshTopProof.GenFinal GenDecode

/-- The plaintext of cell `cc` of a column total over the members `8 cc + j` (component `1`). -/
noncomputable def cellVal {F C : Nat} (hF : 8 * C ≤ F) (Δ : ErrorPoly 8) (ν : Fin F → ErrorPoly 8)
    (cc : Fin C) : ErrorPoly 8 :=
  ∑ j : Fin 8, Xe (n := 8) (j : Nat) * (Δ * ν ⟨8 * cc + j, by have := cc.isLt; have := j.isLt; omega⟩)

/-- **Component `1` of a cell of a column total**, in the masked form with the cell keys. -/
theorem cell_form {q C F : Nat} (hF : 8 * C ≤ F) {sm sp Δ : ErrorPoly 8} {ν : Fin F → ErrorPoly 8}
    {G : ExactMatrix q 8 1 25} {x : Tup (ExactMatrix q 8 (1 * 1) (C * 25)) 6} {B : Nat} (cc : Fin C)
    (h : EncRel sm (blkT (r := 1 * 1) (C := C) (m := 25) x cc)
      (∑ j : Fin 8, (fun (f : Fin F) (jj : Fin 2) ↦ reducePoly q 8 (Xe (n := 8) (f.val % 8)) •
          (reducePoly q 8 (Δ * ν f * unitPair sp jj) • G))
        ⟨8 * cc + j, by have := cc.isLt; have := j.isLt; omega⟩) B) :
    ∃ e : ErrorMatrix 8 1 25, CoeffBound e B ∧
      blk (C := C) (m := 25) (x.get 5) cc = maskPart (reducePoly q 8 sm)
        (fun k ↦ blk (C := C) (m := 25) (![x.get 1, x.get 3] k) cc) -
        reducePoly q 8 (cellVal hF Δ ν cc) • G + reduceMatrix q 8 1 25 e := by
  unfold EncRel at h
  obtain ⟨e, he, hc⟩ := Rel.comp h 1
  refine ⟨e, he, ?_⟩
  have hA : (fun k ↦ Enc.A (blkT (r := 1 * 1) (C := C) (m := 25) x cc) k 1) =
      fun k ↦ blk (C := C) (m := 25) (![x.get 1, x.get 3] k) cc := by
    funext k; fin_cases k <;> rfl
  have ht : (∑ j : Fin 8, (fun (f : Fin F) (jj : Fin 2) ↦ reducePoly q 8 (Xe (n := 8) (f.val % 8)) •
          (reducePoly q 8 (Δ * ν f * unitPair sp jj) • G))
        ⟨8 * cc + j, by have := cc.isLt; have := j.isLt; omega⟩) 1 =
      reducePoly q 8 (cellVal hF Δ ν cc) • G := by
    rw [Finset.sum_apply]
    simp only [smul_smul, ← map_mul]
    rw [← Finset.sum_smul, ← map_sum, cellVal]
    congr 2
    refine Finset.sum_congr rfl fun j _ ↦ ?_
    have hj : (8 * (cc : Nat) + j) % 8 = j := by omega
    simp only [hj, unitPair, Matrix.cons_val_one, Matrix.cons_val_zero, mul_one]
  rw [hA, ht] at hc
  exact hc

/-- `rns-bgg/root-input/dio2/one-private`, the tag prefix of the one-private root row. -/
abbrev privPre0 : Blob :=
  [114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100,
    105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]

set_option maxHeartbeats 16000000 in
open W_Stage_online_0 in
/-- **The inputs of step 0's refresh parts that the online proof does not derive.** Every field is
a hypothesis of an online lemma, stated exactly in that lemma's shape, at the online context `Con`
and the common mask `sm` (the hint's mask, `claim_hint0`); the gadgets `g`, `g'`, `g_q` of the
lemmas are the online gadget `w_57` (`w_229` where the lemma states it). They are discharged by
the value layer and the stage inputs:

* the encoding relations of the online stage's inputs (`σin`, `help`, `helpS`, `one`, `priv`,
  `stEnc`, `stHelp`, `lab`, `oneS`, `a*help`, `fIn`, `sIn`) and the label payloads (`pay`,
  `pay0`): `Dio2Import`, the LUT/PRF bucket chains;
* the switching ciphertexts (`*switch*`) and the LUT identities and norms (`*walsh*`, `*hμ*`):
  `Dio2PreprocessSwitch`;
* the smudge constant (`hconst_eq`) and the smudge/target residue (`hTH`): `Dio2SmudgeValues`;
* the label norms (`hα*`): the label values. -/
structure Online0Inputs {backend : BackendContext} {hashModel : HashModel} {params : Params}
    (Con : generatedRoot.Ctx backend hashModel params) (sm : ErrorPoly 8) where
  /-- The LUT secret. -/
  sp : ErrorPoly 8
  /-- The mask is small. -/
  hsm : polyNorm sm ≤ 1
  -- ### The σ sampler total (`online0_sigma_total`)
  /-- σ total: the member plaintexts. -/
  σμ : Fin 8 → ErrorPoly 8
  /-- σ total: the member bound. -/
  σB : Nat
  /-- σ total: the members (`hin`). -/
  σin : ∀ o : Fin 8, EncRel sm (generatedRoot.w_34_0 Con o, (generatedRoot.w_34_1 Con o,
      (generatedRoot.w_34_2 Con o, (generatedRoot.w_34_3 Con o, (generatedRoot.w_34_4 Con o,
        (generatedRoot.w_34_5 Con o, ()))))))
    (fun j ↦ reducePoly modulus_0 8 (((1 : Nat) : Int) * σμ o * unitPair sp j) • generatedRoot.w_57_0 Con) σB
  /-- The readout helpers' bound (`w_13`/`w_14`, shared by the σ readout, fresh and smudge LUTs). -/
  Bh : Nat
  /-- The readout helpers `w_13`/`w_14` (`hhelp` of `online0_sigma_total`, `online0_fresh_row`,
  `online0_smudge_row`). -/
  help : ∀ i : Fin 8, Rel sm (keyPairG hashModel (generatedRoot.w_7_0 Con) modulus_0 25 helperPre
      [.integer (sigmaOf sigmas8 i)])
    (fun j ↦ reducePoly modulus_0 8 (((1 : Nat) : Int) *
      unitPair (substitution (sigmaOf sigmas8 i).toNat (sigmas8_odd i) sm) j) • generatedRoot.w_57_0 Con)
    ![generatedRoot.w_13_0 Con i, generatedRoot.w_14_0 Con i] Bh
  /-- σ readout: the scale. -/
  σΔr : ErrorPoly 8
  /-- σ readout: the values. -/
  σνr : Fin 8 → ErrorPoly 8
  /-- σ readout: the switching coefficients. -/
  σCr : Fin 8 → ExactPoly modulus_0 8
  /-- σ readout: the switching error bound. -/
  σBBr : Nat
  /-- σ readout: the switching ciphertexts `w_9 … w_12` (`hswitch`). -/
  σswitch : ∀ i : Fin 8, ∃ e : Fin 2 → ErrorMatrix 8 1 25, (∀ j, CoeffBound (e j) σBBr) ∧ ∀ j,
    reducePoly modulus_0 8 (substitution (sigmaOf sigmas8 i).toNat (sigmas8_odd i) sp) •
        ![![generatedRoot.w_9_0 Con, generatedRoot.w_10_0 Con],
          ![generatedRoot.w_11_0 Con, generatedRoot.w_12_0 Con]] 0 j i +
      ![![generatedRoot.w_9_0 Con, generatedRoot.w_10_0 Con],
        ![generatedRoot.w_11_0 Con, generatedRoot.w_12_0 Con]] 1 j i =
    (σCr i * reducePoly modulus_0 8 (σΔr * unitPair sp j)) • generatedRoot.w_57_0 Con +
      reduceMatrix modulus_0 8 1 25 (e j)
  /-- σ readout: `hμ`. -/
  σhμ : ∀ o i, polyNorm (substitution (sigmaOf sigmas8 i).toNat (sigmas8_odd i) (σμ o)) ≤ 1
  /-- σ readout: `hwalsh`. -/
  σwalsh : ∀ o, ∑ i, reducePoly modulus_0 8 (substitution (sigmaOf sigmas8 i).toNat
    (sigmas8_odd i) (σμ o)) * σCr i = reducePoly modulus_0 8 (σνr o)
  /-- σ sampling: the helpers' bound. -/
  BhS : Nat
  /-- σ sampling: the helpers `w_6`/`w_8` (`hhelpS`). -/
  helpS : ∀ i : Fin 8, Rel sm (keyPairG hashModel (generatedRoot.w_7_0 Con) modulus_0 25 helperPre
      [.integer (sigmaOf sigmas8 i)])
    (fun j ↦ reducePoly modulus_0 8 (((1 : Nat) : Int) *
      unitPair (substitution (sigmaOf sigmas8 i).toNat (sigmas8_odd i) sm) j) • generatedRoot.w_57_0 Con)
    ![generatedRoot.w_6_0 Con i, generatedRoot.w_8_0 Con i] BhS
  /-- σ sampling: the scale `Δ'`. -/
  σΔ : ErrorPoly 8
  /-- σ sampling: the values `ν_o`. -/
  σν : Fin 8 → ErrorPoly 8
  /-- σ sampling: the switching coefficients. -/
  σCS : Fin 8 → ExactPoly modulus_0 8
  /-- σ sampling: the switching error bound. -/
  σBBS : Nat
  /-- σ sampling: the switching ciphertexts `w_2 … w_5` (`hswitchS`). -/
  σswitchS : ∀ i : Fin 8, ∃ e : Fin 2 → ErrorMatrix 8 1 25, (∀ j, CoeffBound (e j) σBBS) ∧ ∀ j,
    reducePoly modulus_0 8 (substitution (sigmaOf sigmas8 i).toNat (sigmas8_odd i) sp) •
        ![![generatedRoot.w_2_0 Con, generatedRoot.w_3_0 Con],
          ![generatedRoot.w_4_0 Con, generatedRoot.w_5_0 Con]] 0 j i +
      ![![generatedRoot.w_2_0 Con, generatedRoot.w_3_0 Con],
        ![generatedRoot.w_4_0 Con, generatedRoot.w_5_0 Con]] 1 j i =
    (σCS i * reducePoly modulus_0 8 (σΔ * unitPair sp j)) • generatedRoot.w_57_0 Con +
      reduceMatrix modulus_0 8 1 25 (e j)
  /-- σ sampling: `hμS`. -/
  σhμS : ∀ o i, polyNorm (substitution (sigmaOf sigmas8 i).toNat (sigmas8_odd i) (σΔr * σνr o)) ≤ 1
  /-- σ sampling: `hwalshS`. -/
  σwalshS : ∀ o, ∑ i, reducePoly modulus_0 8 (substitution (sigmaOf sigmas8 i).toNat
    (sigmas8_odd i) (σΔr * σνr o)) * σCS i = reducePoly modulus_0 8 (σν o)
  -- ### The one-public row (`online0_c1`)
  /-- One-public: component `0` of the root row. -/
  oneC0 : ExactMatrix modulus_0 8 1 25
  /-- One-public: the bound. -/
  oneB : Nat
  /-- One-public: the root row of label member `1` (`hone`). -/
  one : Rel sm (labelKeysG hashModel (generatedRoot.w_7_0 Con) modulus_0 25 labelPre 1)
    (fun j ↦ reducePoly modulus_0 8 (1 * unitPair sm j) • generatedRoot.w_57_0 Con)
    ![oneC0, generatedRoot.w_17_0 Con 1] oneB
  -- ### The one-private row (`online0_target_priv`)
  /-- One-private: the plaintext `t̄`. -/
  tp : ErrorPoly 8
  /-- One-private: the bound. -/
  privB : Nat
  /-- One-private: the root row (`hpriv`). -/
  priv : Rel sm (rootKey (n := 8) hashModel (generatedRoot.w_7_0 Con) privPre0 [] modulus_0 25)
    (fun j ↦ reducePoly modulus_0 8 (1 * unitPair tp j) • generatedRoot.w_229_0 Con)
    ![generatedRoot.w_21_0 Con, generatedRoot.w_25_0 Con] privB
  -- ### The state LUT (`online0_state_lut`)
  /-- State: the input plaintext. -/
  stμ : ErrorPoly 8
  /-- State: the input bound. -/
  stB : Nat
  /-- State: the input (`henc`). -/
  stEnc : Rel sm ![![generatedRoot.w_305_0 Con, generatedRoot.w_315_0 Con],
      ![generatedRoot.w_306_0 Con, generatedRoot.w_316_0 Con]]
    (fun j ↦ reducePoly modulus_0 8 (((1 : Nat) : Int) * stμ * unitPair sp j) • generatedRoot.w_57_0 Con)
    ![generatedRoot.w_304_0 Con, generatedRoot.w_314_0 Con] stB
  /-- State: the helpers' bound. -/
  stBh : Nat
  /-- State: the helpers `w_255`/`w_256` (`hhelp`). -/
  stHelp : ∀ i : Fin 4, Rel sm (keyPairG hashModel (generatedRoot.w_7_0 Con) modulus_0 25 helperPre
      [.integer (sigmaOf sigmas4 i)])
    (fun j ↦ reducePoly modulus_0 8 (((1 : Nat) : Int) *
      unitPair (substitution (sigmaOf sigmas4 i).toNat (sigmas4_odd i) sm) j) • generatedRoot.w_57_0 Con)
    ![generatedRoot.w_255_0 Con i, generatedRoot.w_256_0 Con i] stBh
  /-- State: the scale. -/
  stΔ : ErrorPoly 8
  /-- State: the value. -/
  stν : ErrorPoly 8
  /-- State: the switching coefficients. -/
  stC : Fin 4 → ExactPoly modulus_0 8
  /-- State: the switching error bound. -/
  stBB : Nat
  /-- State: the switching ciphertexts `w_247 … w_253` (`hswitch`, at the gadget `w_229`). -/
  stSwitch : ∀ (jc : Fin 2) (i : Fin 4), ∃ e : ErrorMatrix 8 1 25, CoeffBound e stBB ∧
    reducePoly modulus_0 8 (substitution (sigmaOf sigmas4 i).toNat (sigmas4_odd i) sp) •
        ![![generatedRoot.w_247_0 Con, generatedRoot.w_249_0 Con],
          ![generatedRoot.w_251_0 Con, generatedRoot.w_253_0 Con]] 0 jc i +
      ![![generatedRoot.w_247_0 Con, generatedRoot.w_249_0 Con],
        ![generatedRoot.w_251_0 Con, generatedRoot.w_253_0 Con]] 1 jc i =
      (stC i * reducePoly modulus_0 8 (stΔ * unitPair sp jc)) • generatedRoot.w_229_0 Con +
        reduceMatrix modulus_0 8 1 25 e
  /-- State: `hμ`. -/
  stHμ : ∀ i, polyNorm (substitution (sigmaOf sigmas4 i).toNat (sigmas4_odd i) stμ) ≤ 1
  /-- State: `hwalsh`. -/
  stWalsh : ∑ i, reducePoly modulus_0 8 (substitution (sigmaOf sigmas4 i).toNat
    (sigmas4_odd i) stμ) * stC i = reducePoly modulus_0 8 stν
  -- ### The labels (`online0_label_*`)
  /-- Labels: the values `α_i`. -/
  lv : Fin 4 → ErrorPoly 8
  /-- Labels: `hα`. -/
  hα : ∀ i, polyNorm (lv i) ≤ 1
  /-- Labels: the bound. -/
  Ba : Nat
  /-- Labels: the label inputs `w_372`/`w_377` (`hlab`). -/
  lab : ∀ i : Fin 4, Rel sm
    (labelKeysG hashModel (generatedRoot.w_7_0 Con) modulus_0 25 labelInPre (i : Nat))
    (fun j ↦ reducePoly modulus_0 8 (lv i * unitPair sm j) • generatedRoot.w_57_0 Con)
    ![generatedRoot.w_372_0 Con i, generatedRoot.w_377_0 Con i] Ba
  /-- Labels: the payloads `w_391` (`hpay`). -/
  pay : ∀ i : Fin 4, generatedRoot.w_391_0 Con i = fun _ _ ↦ reducePoly modulus_0 8 (lv i)
  /-- `online0_label_425`: `hα13`. -/
  hα13 : polyNorm (lv 1 * lv 3) ≤ 1
  /-- `online0_label_543`: `hα01`. -/
  hα01 : polyNorm (lv 0 * lv 1) ≤ 1
  /-- `online0_label_543`: `hα0123`. -/
  hα0123 : polyNorm (lv 0 * lv 1 * (lv 2 * lv 3)) ≤ 1
  /-- `online0_label_626`: `hα02`. -/
  hα02 : polyNorm (lv 0 * lv 2) ≤ 1
  /-- `online0_label_626`: `hα023`. -/
  hα023 : polyNorm (lv 0 * lv 2 * lv 3) ≤ 1
  /-- `online0_label_734`: `hα12`. -/
  hα12 : polyNorm (lv 1 * lv 2) ≤ 1
  /-- `online0_label_680`: the state label value `β`. -/
  βo : ErrorPoly 8
  /-- `online0_label_680`: its bound. -/
  Bo : Nat
  /-- `online0_label_680`: the state label row `(w_257 1, w_262 1)` (`hone`). -/
  oneS : Rel sm (labelKeysG hashModel (generatedRoot.w_7_0 Con) modulus_0 25 labelPre
      ((1 : Fin 2) : Nat))
    (fun j ↦ reducePoly modulus_0 8 (βo * unitPair sm j) • generatedRoot.w_57_0 Con)
    ![generatedRoot.w_257_0 Con 1, generatedRoot.w_262_0 Con 1] Bo
  /-- `online0_label_680`: its payload `w_276 1` (`hpay0`). -/
  pay0 : generatedRoot.w_276_0 Con 1 = fun _ _ ↦ reducePoly modulus_0 8 βo
  /-- `online0_label_680`: `hα1β`. -/
  hα1β : polyNorm (lv 1 * βo) ≤ 1
  -- ### The `Aut(k)` helper targets (`online0_aut_k`)
  /-- `Aut(3)`: the helper bound. -/
  a3Bh : Nat
  /-- `Aut(3)`: the helper `(w_764, w_765)` (`online0_aut_3`'s `hhelp`). -/
  a3help : Rel sm (keyPairG hashModel (generatedRoot.w_7_0 Con) modulus_0 25 helperPre [.integer 3])
    (fun j ↦ reducePoly modulus_0 8 (((1 : Nat) : Int) *
      unitPair (substitution 3 (⟨1, rfl⟩ : Odd 3) sm) j) • generatedRoot.w_57_0 Con)
    ![generatedRoot.w_764_0 Con 0, generatedRoot.w_765_0 Con 0] a3Bh
  /-- `Aut(3)`: the LUT scale `Δ'`. -/
  a3Δ : ErrorPoly 8
  /-- `Aut(3)`: the LUT value `ν`. -/
  a3ν : ErrorPoly 8
  /-- `Aut(3)`: the switching coefficient. -/
  a3C : ExactPoly modulus_0 8
  /-- `Aut(3)`: the switching error bound. -/
  a3BB : Nat
  /-- `Aut(3)`: the switching ciphertexts `(w_758, w_762)` (`hswitch`). -/
  a3switch : ∃ e : ErrorMatrix 8 1 25, CoeffBound e a3BB ∧
    reducePoly modulus_0 8 (substitution 3 (⟨1, rfl⟩ : Odd 3) sp) • generatedRoot.w_758_0 Con 0 +
      generatedRoot.w_762_0 Con 0 = (a3C * reducePoly modulus_0 8 (a3Δ * unitPair sp 1)) •
        generatedRoot.w_57_0 Con + reduceMatrix modulus_0 8 1 25 e
  /-- `Aut(3)`: the input plaintext (the σ sampler's `Δ' Σ X^o ν_o`) stays small under `σ_3`. -/
  a3hμ : polyNorm (substitution 3 (⟨1, rfl⟩ : Odd 3) (σΔ * ∑ o : Fin 8, Xe (n := 8) (o : Nat) * σν o)) ≤ 1
  /-- `Aut(3)`: the LUT identity (`hwalsh`). -/
  a3walsh : reducePoly modulus_0 8 (substitution 3 (⟨1, rfl⟩ : Odd 3) (σΔ * ∑ o : Fin 8, Xe (n := 8) (o : Nat) * σν o)) * a3C =
    reducePoly modulus_0 8 a3ν
  /-- `Aut(5)`: the helper bound. -/
  a5Bh : Nat
  /-- `Aut(5)`: the helper `(w_822, w_823)` (`online0_aut_5`'s `hhelp`). -/
  a5help : Rel sm (keyPairG hashModel (generatedRoot.w_7_0 Con) modulus_0 25 helperPre [.integer 5])
    (fun j ↦ reducePoly modulus_0 8 (((1 : Nat) : Int) *
      unitPair (substitution 5 (⟨2, rfl⟩ : Odd 5) sm) j) • generatedRoot.w_57_0 Con)
    ![generatedRoot.w_822_0 Con 0, generatedRoot.w_823_0 Con 0] a5Bh
  /-- `Aut(5)`: the LUT scale `Δ'`. -/
  a5Δ : ErrorPoly 8
  /-- `Aut(5)`: the LUT value `ν`. -/
  a5ν : ErrorPoly 8
  /-- `Aut(5)`: the switching coefficient. -/
  a5C : ExactPoly modulus_0 8
  /-- `Aut(5)`: the switching error bound. -/
  a5BB : Nat
  /-- `Aut(5)`: the switching ciphertexts `(w_816, w_820)` (`hswitch`). -/
  a5switch : ∃ e : ErrorMatrix 8 1 25, CoeffBound e a5BB ∧
    reducePoly modulus_0 8 (substitution 5 (⟨2, rfl⟩ : Odd 5) sp) • generatedRoot.w_816_0 Con 0 +
      generatedRoot.w_820_0 Con 0 = (a5C * reducePoly modulus_0 8 (a5Δ * unitPair sp 1)) •
        generatedRoot.w_57_0 Con + reduceMatrix modulus_0 8 1 25 e
  /-- `Aut(5)`: the input plaintext (the σ sampler's `Δ' Σ X^o ν_o`) stays small under `σ_5`. -/
  a5hμ : polyNorm (substitution 5 (⟨2, rfl⟩ : Odd 5) (σΔ * ∑ o : Fin 8, Xe (n := 8) (o : Nat) * σν o)) ≤ 1
  /-- `Aut(5)`: the LUT identity (`hwalsh`). -/
  a5walsh : reducePoly modulus_0 8 (substitution 5 (⟨2, rfl⟩ : Odd 5) (σΔ * ∑ o : Fin 8, Xe (n := 8) (o : Nat) * σν o)) * a5C =
    reducePoly modulus_0 8 a5ν
  /-- `Aut(7)`: the helper bound. -/
  a7Bh : Nat
  /-- `Aut(7)`: the helper `(w_880, w_881)` (`online0_aut_7`'s `hhelp`). -/
  a7help : Rel sm (keyPairG hashModel (generatedRoot.w_7_0 Con) modulus_0 25 helperPre [.integer 7])
    (fun j ↦ reducePoly modulus_0 8 (((1 : Nat) : Int) *
      unitPair (substitution 7 (⟨3, rfl⟩ : Odd 7) sm) j) • generatedRoot.w_57_0 Con)
    ![generatedRoot.w_880_0 Con 0, generatedRoot.w_881_0 Con 0] a7Bh
  /-- `Aut(7)`: the LUT scale `Δ'`. -/
  a7Δ : ErrorPoly 8
  /-- `Aut(7)`: the LUT value `ν`. -/
  a7ν : ErrorPoly 8
  /-- `Aut(7)`: the switching coefficient. -/
  a7C : ExactPoly modulus_0 8
  /-- `Aut(7)`: the switching error bound. -/
  a7BB : Nat
  /-- `Aut(7)`: the switching ciphertexts `(w_874, w_878)` (`hswitch`). -/
  a7switch : ∃ e : ErrorMatrix 8 1 25, CoeffBound e a7BB ∧
    reducePoly modulus_0 8 (substitution 7 (⟨3, rfl⟩ : Odd 7) sp) • generatedRoot.w_874_0 Con 0 +
      generatedRoot.w_878_0 Con 0 = (a7C * reducePoly modulus_0 8 (a7Δ * unitPair sp 1)) •
        generatedRoot.w_57_0 Con + reduceMatrix modulus_0 8 1 25 e
  /-- `Aut(7)`: the input plaintext (the σ sampler's `Δ' Σ X^o ν_o`) stays small under `σ_7`. -/
  a7hμ : polyNorm (substitution 7 (⟨3, rfl⟩ : Odd 7) (σΔ * ∑ o : Fin 8, Xe (n := 8) (o : Nat) * σν o)) ≤ 1
  /-- `Aut(7)`: the LUT identity (`hwalsh`). -/
  a7walsh : reducePoly modulus_0 8 (substitution 7 (⟨3, rfl⟩ : Odd 7) (σΔ * ∑ o : Fin 8, Xe (n := 8) (o : Nat) * σν o)) * a7C =
    reducePoly modulus_0 8 a7ν
  /-- `Aut(9)`: the helper bound. -/
  a9Bh : Nat
  /-- `Aut(9)`: the helper `(w_938, w_939)` (`online0_aut_9`'s `hhelp`). -/
  a9help : Rel sm (keyPairG hashModel (generatedRoot.w_7_0 Con) modulus_0 25 helperPre [.integer 9])
    (fun j ↦ reducePoly modulus_0 8 (((1 : Nat) : Int) *
      unitPair (substitution 9 (⟨4, rfl⟩ : Odd 9) sm) j) • generatedRoot.w_57_0 Con)
    ![generatedRoot.w_938_0 Con 0, generatedRoot.w_939_0 Con 0] a9Bh
  /-- `Aut(9)`: the LUT scale `Δ'`. -/
  a9Δ : ErrorPoly 8
  /-- `Aut(9)`: the LUT value `ν`. -/
  a9ν : ErrorPoly 8
  /-- `Aut(9)`: the switching coefficient. -/
  a9C : ExactPoly modulus_0 8
  /-- `Aut(9)`: the switching error bound. -/
  a9BB : Nat
  /-- `Aut(9)`: the switching ciphertexts `(w_932, w_936)` (`hswitch`). -/
  a9switch : ∃ e : ErrorMatrix 8 1 25, CoeffBound e a9BB ∧
    reducePoly modulus_0 8 (substitution 9 (⟨4, rfl⟩ : Odd 9) sp) • generatedRoot.w_932_0 Con 0 +
      generatedRoot.w_936_0 Con 0 = (a9C * reducePoly modulus_0 8 (a9Δ * unitPair sp 1)) •
        generatedRoot.w_57_0 Con + reduceMatrix modulus_0 8 1 25 e
  /-- `Aut(9)`: the input plaintext (the σ sampler's `Δ' Σ X^o ν_o`) stays small under `σ_9`. -/
  a9hμ : polyNorm (substitution 9 (⟨4, rfl⟩ : Odd 9) (σΔ * ∑ o : Fin 8, Xe (n := 8) (o : Nat) * σν o)) ≤ 1
  /-- `Aut(9)`: the LUT identity (`hwalsh`). -/
  a9walsh : reducePoly modulus_0 8 (substitution 9 (⟨4, rfl⟩ : Odd 9) (σΔ * ∑ o : Fin 8, Xe (n := 8) (o : Nat) * σν o)) * a9C =
    reducePoly modulus_0 8 a9ν
  /-- `Aut(11)`: the helper bound. -/
  a11Bh : Nat
  /-- `Aut(11)`: the helper `(w_996, w_997)` (`online0_aut_11`'s `hhelp`). -/
  a11help : Rel sm (keyPairG hashModel (generatedRoot.w_7_0 Con) modulus_0 25 helperPre [.integer 11])
    (fun j ↦ reducePoly modulus_0 8 (((1 : Nat) : Int) *
      unitPair (substitution 11 (⟨5, rfl⟩ : Odd 11) sm) j) • generatedRoot.w_57_0 Con)
    ![generatedRoot.w_996_0 Con 0, generatedRoot.w_997_0 Con 0] a11Bh
  /-- `Aut(11)`: the LUT scale `Δ'`. -/
  a11Δ : ErrorPoly 8
  /-- `Aut(11)`: the LUT value `ν`. -/
  a11ν : ErrorPoly 8
  /-- `Aut(11)`: the switching coefficient. -/
  a11C : ExactPoly modulus_0 8
  /-- `Aut(11)`: the switching error bound. -/
  a11BB : Nat
  /-- `Aut(11)`: the switching ciphertexts `(w_990, w_994)` (`hswitch`). -/
  a11switch : ∃ e : ErrorMatrix 8 1 25, CoeffBound e a11BB ∧
    reducePoly modulus_0 8 (substitution 11 (⟨5, rfl⟩ : Odd 11) sp) • generatedRoot.w_990_0 Con 0 +
      generatedRoot.w_994_0 Con 0 = (a11C * reducePoly modulus_0 8 (a11Δ * unitPair sp 1)) •
        generatedRoot.w_57_0 Con + reduceMatrix modulus_0 8 1 25 e
  /-- `Aut(11)`: the input plaintext (the σ sampler's `Δ' Σ X^o ν_o`) stays small under `σ_11`. -/
  a11hμ : polyNorm (substitution 11 (⟨5, rfl⟩ : Odd 11) (σΔ * ∑ o : Fin 8, Xe (n := 8) (o : Nat) * σν o)) ≤ 1
  /-- `Aut(11)`: the LUT identity (`hwalsh`). -/
  a11walsh : reducePoly modulus_0 8 (substitution 11 (⟨5, rfl⟩ : Odd 11) (σΔ * ∑ o : Fin 8, Xe (n := 8) (o : Nat) * σν o)) * a11C =
    reducePoly modulus_0 8 a11ν
  /-- `Aut(13)`: the helper bound. -/
  a13Bh : Nat
  /-- `Aut(13)`: the helper `(w_1054, w_1055)` (`online0_aut_13`'s `hhelp`). -/
  a13help : Rel sm (keyPairG hashModel (generatedRoot.w_7_0 Con) modulus_0 25 helperPre [.integer 13])
    (fun j ↦ reducePoly modulus_0 8 (((1 : Nat) : Int) *
      unitPair (substitution 13 (⟨6, rfl⟩ : Odd 13) sm) j) • generatedRoot.w_57_0 Con)
    ![generatedRoot.w_1054_0 Con 0, generatedRoot.w_1055_0 Con 0] a13Bh
  /-- `Aut(13)`: the LUT scale `Δ'`. -/
  a13Δ : ErrorPoly 8
  /-- `Aut(13)`: the LUT value `ν`. -/
  a13ν : ErrorPoly 8
  /-- `Aut(13)`: the switching coefficient. -/
  a13C : ExactPoly modulus_0 8
  /-- `Aut(13)`: the switching error bound. -/
  a13BB : Nat
  /-- `Aut(13)`: the switching ciphertexts `(w_1048, w_1052)` (`hswitch`). -/
  a13switch : ∃ e : ErrorMatrix 8 1 25, CoeffBound e a13BB ∧
    reducePoly modulus_0 8 (substitution 13 (⟨6, rfl⟩ : Odd 13) sp) • generatedRoot.w_1048_0 Con 0 +
      generatedRoot.w_1052_0 Con 0 = (a13C * reducePoly modulus_0 8 (a13Δ * unitPair sp 1)) •
        generatedRoot.w_57_0 Con + reduceMatrix modulus_0 8 1 25 e
  /-- `Aut(13)`: the input plaintext (the σ sampler's `Δ' Σ X^o ν_o`) stays small under `σ_13`. -/
  a13hμ : polyNorm (substitution 13 (⟨6, rfl⟩ : Odd 13) (σΔ * ∑ o : Fin 8, Xe (n := 8) (o : Nat) * σν o)) ≤ 1
  /-- `Aut(13)`: the LUT identity (`hwalsh`). -/
  a13walsh : reducePoly modulus_0 8 (substitution 13 (⟨6, rfl⟩ : Odd 13) (σΔ * ∑ o : Fin 8, Xe (n := 8) (o : Nat) * σν o)) * a13C =
    reducePoly modulus_0 8 a13ν
  /-- `Aut(15)`: the helper bound. -/
  a15Bh : Nat
  /-- `Aut(15)`: the helper `(w_1112, w_1113)` (`online0_aut_15`'s `hhelp`). -/
  a15help : Rel sm (keyPairG hashModel (generatedRoot.w_7_0 Con) modulus_0 25 helperPre [.integer 15])
    (fun j ↦ reducePoly modulus_0 8 (((1 : Nat) : Int) *
      unitPair (substitution 15 (⟨7, rfl⟩ : Odd 15) sm) j) • generatedRoot.w_57_0 Con)
    ![generatedRoot.w_1112_0 Con 0, generatedRoot.w_1113_0 Con 0] a15Bh
  /-- `Aut(15)`: the LUT scale `Δ'`. -/
  a15Δ : ErrorPoly 8
  /-- `Aut(15)`: the LUT value `ν`. -/
  a15ν : ErrorPoly 8
  /-- `Aut(15)`: the switching coefficient. -/
  a15C : ExactPoly modulus_0 8
  /-- `Aut(15)`: the switching error bound. -/
  a15BB : Nat
  /-- `Aut(15)`: the switching ciphertexts `(w_1106, w_1110)` (`hswitch`). -/
  a15switch : ∃ e : ErrorMatrix 8 1 25, CoeffBound e a15BB ∧
    reducePoly modulus_0 8 (substitution 15 (⟨7, rfl⟩ : Odd 15) sp) • generatedRoot.w_1106_0 Con 0 +
      generatedRoot.w_1110_0 Con 0 = (a15C * reducePoly modulus_0 8 (a15Δ * unitPair sp 1)) •
        generatedRoot.w_57_0 Con + reduceMatrix modulus_0 8 1 25 e
  /-- `Aut(15)`: the input plaintext (the σ sampler's `Δ' Σ X^o ν_o`) stays small under `σ_15`. -/
  a15hμ : polyNorm (substitution 15 (⟨7, rfl⟩ : Odd 15) (σΔ * ∑ o : Fin 8, Xe (n := 8) (o : Nat) * σν o)) ≤ 1
  /-- `Aut(15)`: the LUT identity (`hwalsh`). -/
  a15walsh : reducePoly modulus_0 8 (substitution 15 (⟨7, rfl⟩ : Odd 15) (σΔ * ∑ o : Fin 8, Xe (n := 8) (o : Nat) * σν o)) * a15C =
    reducePoly modulus_0 8 a15ν
  -- ### The fresh-noise row (`online0_fresh_row`)
  /-- Fresh: the member plaintexts. -/
  fμ : Fin 7200 → ErrorPoly 8
  /-- Fresh: the member bound. -/
  fB : Nat
  /-- Fresh: the members `w_1186` (`hin`). -/
  fIn : ∀ o : Fin 7200, EncRel sm (generatedRoot.w_1186_0 Con o, (generatedRoot.w_1186_1 Con o,
      (generatedRoot.w_1186_2 Con o, (generatedRoot.w_1186_3 Con o, (generatedRoot.w_1186_4 Con o,
        (generatedRoot.w_1186_5 Con o, ()))))))
    (fun j ↦ reducePoly modulus_0 8 (((1 : Nat) : Int) * fμ o * unitPair sp j) • generatedRoot.w_57_0 Con) fB
  /-- Fresh: the scale. -/
  fΔ : ErrorPoly 8
  /-- Fresh: the values. -/
  fν : Fin 7200 → ErrorPoly 8
  /-- Fresh: the switching coefficients. -/
  fC : Fin 8 → ExactPoly modulus_0 8
  /-- Fresh: the switching error bound. -/
  fBB : Nat
  /-- Fresh: the switching ciphertexts `w_1164 … w_1167` (`hswitch`). -/
  fSwitch : ∀ i : Fin 8, ∃ e : Fin 2 → ErrorMatrix 8 1 25, (∀ j, CoeffBound (e j) fBB) ∧ ∀ j,
    reducePoly modulus_0 8 (substitution (sigmaOf sigmas8 i).toNat (sigmas8_odd i) sp) •
        ![![generatedRoot.w_1164_0 Con, generatedRoot.w_1165_0 Con],
          ![generatedRoot.w_1166_0 Con, generatedRoot.w_1167_0 Con]] 0 j i +
      ![![generatedRoot.w_1164_0 Con, generatedRoot.w_1165_0 Con],
        ![generatedRoot.w_1166_0 Con, generatedRoot.w_1167_0 Con]] 1 j i =
    (fC i * reducePoly modulus_0 8 (fΔ * unitPair sp j)) • generatedRoot.w_57_0 Con +
      reduceMatrix modulus_0 8 1 25 (e j)
  /-- Fresh: `hμ`. -/
  fHμ : ∀ o i, polyNorm (substitution (sigmaOf sigmas8 i).toNat (sigmas8_odd i) (fμ o)) ≤ 1
  /-- Fresh: `hwalsh`. -/
  fWalsh : ∀ o, ∑ i, reducePoly modulus_0 8 (substitution (sigmaOf sigmas8 i).toNat
    (sigmas8_odd i) (fμ o)) * fC i = reducePoly modulus_0 8 (fν o)
  -- ### The smudge row (`online0_smudge_row`)
  /-- Smudge: the member plaintexts. -/
  sμ : Fin 36000 → ErrorPoly 8
  /-- Smudge: the member bound. -/
  sB : Nat
  /-- Smudge: the members `w_1240` (`hin`). -/
  sIn : ∀ o : Fin 36000, EncRel sm (generatedRoot.w_1240_0 Con o, (generatedRoot.w_1240_1 Con o,
      (generatedRoot.w_1240_2 Con o, (generatedRoot.w_1240_3 Con o, (generatedRoot.w_1240_4 Con o,
        (generatedRoot.w_1240_5 Con o, ()))))))
    (fun j ↦ reducePoly modulus_0 8 (((1 : Nat) : Int) * sμ o * unitPair sp j) • generatedRoot.w_57_0 Con) sB
  /-- Smudge: the scale. -/
  sΔ : ErrorPoly 8
  /-- Smudge: the values. -/
  sν : Fin 36000 → ErrorPoly 8
  /-- Smudge: the switching coefficients. -/
  sC : Fin 8 → ExactPoly modulus_0 8
  /-- Smudge: the switching error bound. -/
  sBB : Nat
  /-- Smudge: the switching ciphertexts `w_1221 … w_1224` (`hswitch`). -/
  sSwitch : ∀ i : Fin 8, ∃ e : Fin 2 → ErrorMatrix 8 1 25, (∀ j, CoeffBound (e j) sBB) ∧ ∀ j,
    reducePoly modulus_0 8 (substitution (sigmaOf sigmas8 i).toNat (sigmas8_odd i) sp) •
        ![![generatedRoot.w_1221_0 Con, generatedRoot.w_1222_0 Con],
          ![generatedRoot.w_1223_0 Con, generatedRoot.w_1224_0 Con]] 0 j i +
      ![![generatedRoot.w_1221_0 Con, generatedRoot.w_1222_0 Con],
        ![generatedRoot.w_1223_0 Con, generatedRoot.w_1224_0 Con]] 1 j i =
    (sC i * reducePoly modulus_0 8 (sΔ * unitPair sp j)) • generatedRoot.w_57_0 Con +
      reduceMatrix modulus_0 8 1 25 (e j)
  /-- Smudge: `hμ`. -/
  sHμ : ∀ o i, polyNorm (substitution (sigmaOf sigmas8 i).toNat (sigmas8_odd i) (sμ o)) ≤ 1
  /-- Smudge: `hwalsh`. -/
  sWalsh : ∀ o, ∑ i, reducePoly modulus_0 8 (substitution (sigmaOf sigmas8 i).toNat
    (sigmas8_odd i) (sμ o)) * sC i = reducePoly modulus_0 8 (sν o)
  -- ### The smudge constant and residue (`online0_part_smudge0`'s `hh`, `online0_part_prime0`'s `hTH`)
  /-- The smudge scale `h` (the `1 × 1` input `w_1274`). -/
  hconst : ExactPoly modulus_0 8
  /-- `w_1274` is the constant `h` (`hh`). -/
  hconst_eq : generatedRoot.w_1274_0 Con = fun _ _ ↦ hconst
  /-- The residue bound. -/
  BM : Nat
  /-- **`hTH`** of every part `i`: the smudge targets `(t_c h)_c` of cells `125 i + c` minus the
  stage's smudge offset `w_1275` are small. -/
  hTH : ∀ i : Fin 36, ∃ M : ErrorMatrix 8 1 25, CoeffBound M BM ∧
    Matrix.of (fun _ l ↦ reducePoly modulus_0 8
        (cellVal (C := 4500) (by norm_num) sΔ sν ⟨125 * i + l.val, by have := i.isLt; have := l.isLt; omega⟩) * hconst) -
      generatedRoot.w_1276_0 Con = reduceMatrix modulus_0 8 1 25 M

/-! ## The derived online rows -/

open W_Stage_online_0 in
/-- The masked form of a `cₓ` member of step 0: the row is `s̄ K - (1 · x) G + e` with the refresh
gadget `G = w_229` and `CoeffBound e B` (the form `hcx` of `online0_part_prime0`). -/
def CxRow {backend : BackendContext} {hashModel : HashModel} {params : Params}
    (Con : generatedRoot.Ctx backend hashModel params) (sm : ErrorPoly 8)
    (K : Fin 2 → ExactMatrix modulus_0 8 1 25) (x : ErrorPoly 8) (B : Nat)
    (row : ExactMatrix modulus_0 8 1 25) : Prop :=
  ∃ e : ErrorMatrix 8 1 25, CoeffBound e B ∧
    row = maskPart (reducePoly modulus_0 8 sm) K -
      reducePoly modulus_0 8 (((1 : Nat) : Int) * x) • generatedRoot.w_229_0 Con +
      reduceMatrix modulus_0 8 1 25 e

open W_Stage_online_0 in
/-- A component of an encoding relation with a scalar target is a `cₓ` member form. -/
theorem cxRow_of_rel {backend : BackendContext} {hashModel : HashModel} {params : Params}
    {Con : generatedRoot.Ctx backend hashModel params} {sm : ErrorPoly 8}
    {A : Fin 2 → Fin 2 → ExactMatrix modulus_0 8 1 25} {t c : Fin 2 → ExactMatrix modulus_0 8 1 25}
    {B : Nat} (h : Rel sm A t c B) (j : Fin 2) {x : ErrorPoly 8}
    (ht : t j = reducePoly modulus_0 8 (((1 : Nat) : Int) * x) • generatedRoot.w_229_0 Con) :
    CxRow Con sm (fun k ↦ A k j) x B (c j) := by
  obtain ⟨e, he, hc⟩ := Rel.comp h j
  exact ⟨e, he, by rw [hc, ht]⟩

open W_Stage_online_0 in
/-- A masked form with any scalar equal to `1 · x` and gadget `w_229` is a `cₓ` member form. -/
theorem cxRow_of_form {backend : BackendContext} {hashModel : HashModel} {params : Params}
    {Con : generatedRoot.Ctx backend hashModel params} {sm : ErrorPoly 8}
    {K : Fin 2 → ExactMatrix modulus_0 8 1 25} {y : ExactPoly modulus_0 8} {x : ErrorPoly 8}
    {B : Nat} {row : ExactMatrix modulus_0 8 1 25}
    (h : ∃ e : ErrorMatrix 8 1 25, CoeffBound e B ∧
      row = maskPart (reducePoly modulus_0 8 sm) K - y • generatedRoot.w_229_0 Con +
        reduceMatrix modulus_0 8 1 25 e)
    (hy : y = reducePoly modulus_0 8 (((1 : Nat) : Int) * x)) : CxRow Con sm K x B row := by
  rw [hy] at h
  exact h

/-- A matrix re-encoded at its own modulus: the self-rebase relation of a gadget. -/
theorem centeredRebaseRuns_self {q r c : Nat} (hq : 1 < q) (G : ExactMatrix q 8 r c) :
    centeredRebaseRuns G G :=
  ⟨hq, (rebase_self hq (⟨hq, rfl⟩ : centeredRebaseRuns G (centeredRebase G))).symm⟩

namespace Online0Inputs

open W_Stage_online_0

variable {backend : BackendContext} {hashModel : HashModel} {params : Params}
  {Con : generatedRoot.Ctx backend hashModel params} {sm : ErrorPoly 8}

/-- The σ carrier's plaintext `Δ' Σ_o X^o ν_o` (the scalar `β` of `online0_part_prime0`). -/
noncomputable def xσ (I : Online0Inputs Con sm) : ErrorPoly 8 :=
  I.σΔ * ∑ o : Fin 8, Xe (n := 8) (o : Nat) * I.σν o

/-- The σ sampler total's error bound (`online0_sigma_total`). -/
def σBound (I : Online0Inputs Con sm) : Nat :=
  8 * (7 * ((25 + 25) * 8 * I.BhS * 2048 + (25 + 25) * 8 *
      (7 * ((25 + 25) * 8 * I.Bh * 2048 + (25 + 25) * 8 * I.σB * 2048) +
        ((25 + 25) * 8 * I.Bh * 2048 + (25 + 25) * 8 * I.σB * 2048) + 8 * (8 * 1 * I.σBBr)) * 2048) +
    ((25 + 25) * 8 * I.BhS * 2048 + (25 + 25) * 8 *
      (7 * ((25 + 25) * 8 * I.Bh * 2048 + (25 + 25) * 8 * I.σB * 2048) +
        ((25 + 25) * 8 * I.Bh * 2048 + (25 + 25) * 8 * I.σB * 2048) + 8 * (8 * 1 * I.σBBr)) * 2048) +
    8 * (8 * 1 * I.σBBS))

/-- The σ carrier's error bound `Beg` (`online0_g`). -/
def gBound (I : Online0Inputs Con sm) : Nat := 25 * 8 * I.σBound * 2048

/-- The one-public row's error bound `Be1` (`online0_c1`). -/
def c1Bound (I : Online0Inputs Con sm) : Nat := 25 * 8 * I.oneB * 2048

/-- The state LUT's error bound (`online0_state_lut`). -/
def stBound (I : Online0Inputs Con sm) : Nat :=
  3 * ((25 + 25) * 8 * I.stBh * 2048 + (25 + 25) * 8 * I.stB * 2048) +
    ((25 + 25) * 8 * I.stBh * 2048 + (25 + 25) * 8 * I.stB * 2048) + 4 * (8 * 1 * I.stBB)

/-- The bound of `E^σ = (w_244, w_245)` as one encoding relation (`online0_target_pub`). -/
def pubBound (I : Online0Inputs Con sm) : Nat :=
  max (25 * 8 * I.gBound * 2048) (25 * 8 * I.c1Bound * 2048)

/-- The fresh row's cell error bound (`online0_fresh_row`). -/
def freshBound (I : Online0Inputs Con sm) : Nat :=
  8 * (7 * ((25 + 25) * 8 * I.Bh * 2048 + (25 + 25) * 8 * I.fB * 2048) +
    ((25 + 25) * 8 * I.Bh * 2048 + (25 + 25) * 8 * I.fB * 2048) + 8 * (8 * 1 * I.fBB))

/-- The smudge row's cell error bound (`online0_smudge_row`). -/
def smudgeBound (I : Online0Inputs Con sm) : Nat :=
  8 * (7 * ((25 + 25) * 8 * I.Bh * 2048 + (25 + 25) * 8 * I.sB * 2048) +
    ((25 + 25) * 8 * I.Bh * 2048 + (25 + 25) * 8 * I.sB * 2048) + 8 * (8 * 1 * I.sBB))

/-- The fresh value `E_c` of cell `cc` (component `1` of the fresh column total). -/
noncomputable def freshCell (I : Online0Inputs Con sm) (cc : Fin 900) : ErrorPoly 8 :=
  cellVal (C := 900) (by norm_num) I.fΔ I.fν cc

/-- The smudge value `t_c` of cell `cc` (component `1` of the smudge column total). -/
noncomputable def smudgeCell (I : Online0Inputs Con sm) (cc : Fin 4500) : ErrorPoly 8 :=
  cellVal (C := 4500) (by norm_num) I.sΔ I.sν cc

variable (I : Online0Inputs Con sm)

/-- **The σ sampler total of step 0** under the inputs (`online0_sigma_total` at `g = g_q = w_57`). -/
theorem sigma_total {outputs : _} (Fon : generatedRoot.Facts Con outputs)
    (hq1 : 1 < modulus_0) (layout : RegularLayout modulus_0)
    (hlayout : backend.regularLayout modulus_0 8 = some layout) (hexact : layout.droppedModuli = 0) :
    EncRel sm (generatedRoot.w_56_0 Con, (generatedRoot.w_56_1 Con, (generatedRoot.w_56_2 Con,
        (generatedRoot.w_56_3 Con, (generatedRoot.w_56_4 Con, (generatedRoot.w_56_5 Con, ()))))))
      (∑ o : Fin 8, fun j ↦ reducePoly modulus_0 8 (Xe (n := 8) o) •
        (reducePoly modulus_0 8 (I.σΔ * I.σν o * unitPair I.sp j) • generatedRoot.w_57_0 Con))
      I.σBound :=
  online0_sigma_total Con Fon hq1 layout hlayout hexact I.hsm Fon.c1.h57_0
    (centeredRebaseRuns_self hq1 _) I.σin I.help I.σswitch I.σhμ I.σwalsh I.helpS I.σswitchS
    I.σhμS I.σwalshS

/-- **The σ carrier of step 0** (`online0_g` on `sigma_total`): `g = s̄ K_g - β G + e_g` with the
online keys `sigmaKey0` and `β = Δ' Σ_o X^o ν_o`. -/
theorem g_form {outputs : _} (Fon : generatedRoot.Facts Con outputs)
    (hq1 : 1 < modulus_0) (layout : RegularLayout modulus_0)
    (hlayout : backend.regularLayout modulus_0 8 = some layout) (hexact : layout.droppedModuli = 0) :
    ∃ e : ErrorMatrix 8 1 25, CoeffBound e I.gBound ∧
      generatedRoot.w_63_0 Con = maskPart (reducePoly modulus_0 8 sm) (sigmaKey0 Con) -
        reducePoly modulus_0 8 I.xσ • generatedRoot.w_57_0 Con + reduceMatrix modulus_0 8 1 25 e :=
  online0_g Con Fon hq1 layout hlayout hexact (I.sigma_total Fon hq1 layout hlayout hexact)

/-- **The one-public row of step 0** (`online0_c1` on `one`): `c₁ = s̄ K₁ - G + e₁` with the online
keys `onePubKey0`. -/
theorem c1_form {outputs : _} (Fon : generatedRoot.Facts Con outputs)
    (hq1 : 1 < modulus_0) (layout : RegularLayout modulus_0)
    (hlayout : backend.regularLayout modulus_0 8 = some layout) (hexact : layout.droppedModuli = 0) :
    ∃ e : ErrorMatrix 8 1 25, CoeffBound e I.c1Bound ∧
      generatedRoot.w_70_0 Con = maskPart (reducePoly modulus_0 8 sm) (onePubKey0 Con) -
        generatedRoot.w_57_0 Con + reduceMatrix modulus_0 8 1 25 e :=
  online0_c1 Con Fon hq1 layout hlayout hexact I.one

/-- The σ sampler total as a relation for the single plaintext `β = Δ' Σ_o X^o ν_o` (`henc` of
the `Aut(k)` targets). -/
theorem sigma_rel {outputs : _} (Fon : generatedRoot.Facts Con outputs)
    (hq1 : 1 < modulus_0) (layout : RegularLayout modulus_0)
    (hlayout : backend.regularLayout modulus_0 8 = some layout) (hexact : layout.droppedModuli = 0) :
    Rel sm ![![generatedRoot.w_56_0 Con, generatedRoot.w_56_1 Con], ![generatedRoot.w_56_2 Con, generatedRoot.w_56_3 Con]]
      (fun j ↦ reducePoly modulus_0 8 (((1 : Nat) : Int) * I.xσ * unitPair I.sp j) • generatedRoot.w_57_0 Con)
      ![generatedRoot.w_56_4 Con, generatedRoot.w_56_5 Con] I.σBound := by
  have h := I.sigma_total Fon hq1 layout hlayout hexact
  have ht : (∑ o : Fin 8, fun j ↦ reducePoly modulus_0 8 (Xe (n := 8) o) •
        (reducePoly modulus_0 8 (I.σΔ * I.σν o * unitPair I.sp j) • generatedRoot.w_57_0 Con)) =
      fun j ↦ reducePoly modulus_0 8 (((1 : Nat) : Int) * I.xσ * unitPair I.sp j) • generatedRoot.w_57_0 Con := by
    funext j
    rw [Finset.sum_apply]
    simp only [smul_smul, ← map_mul]
    rw [← Finset.sum_smul, ← map_sum, xσ]
    congr 2
    simp only [Nat.cast_one, Int.cast_one, one_mul, Finset.mul_sum, Finset.sum_mul]
    exact Finset.sum_congr rfl fun o _ ↦ by ring
  unfold EncRel at h
  rw [ht] at h
  exact h

/-- **`E^σ = (w_244, w_245)` as one encoding relation** of `(β, 1) G` (`online0_target_pub`), with the
label stage's keys `w_402 … w_433` (the label lemmas' `hs`). -/
theorem pub_rel {outputs : _} (Fon : generatedRoot.Facts Con outputs)
    (hq1 : 1 < modulus_0) (layout : RegularLayout modulus_0)
    (hlayout : backend.regularLayout modulus_0 8 = some layout) (hexact : layout.droppedModuli = 0) :
    Rel sm ![![generatedRoot.w_402_0 Con, generatedRoot.w_428_0 Con],
        ![generatedRoot.w_417_0 Con, generatedRoot.w_433_0 Con]]
      (fun j ↦ reducePoly modulus_0 8 (unitPair I.xσ j) • generatedRoot.w_229_0 Con)
      ![generatedRoot.w_244_0 Con, generatedRoot.w_245_0 Con] I.pubBound := by
  obtain ⟨eg, heg, hg⟩ := I.g_form Fon hq1 layout hlayout hexact
  obtain ⟨e1, he1, hc⟩ := I.c1_form Fon hq1 layout hlayout hexact
  rw [online0_gadget_eq Con Fon] at hg hc
  obtain ⟨⟨f0, hf0, h0⟩, ⟨f1, hf1, h1⟩⟩ :=
    online0_target_pub Con Fon hq1 layout hlayout hexact heg he1 hg hc
  have k426 : generatedRoot.w_426_0 Con =
      labelKeysG hashModel (generatedRoot.w_7_0 Con) modulus_0 25 labelPre 1 0 1 := by
    rw [MxxExponentLut.hashSample_eq Fon.c15.h426_0]; rfl
  have k431 : generatedRoot.w_431_0 Con =
      labelKeysG hashModel (generatedRoot.w_7_0 Con) modulus_0 25 labelPre 1 1 1 := by
    rw [MxxExponentLut.hashSample_eq Fon.c15.h431_0]; rfl
  have k428 : generatedRoot.w_428_0 Con = onePubKey0 Con 0 * generatedRoot.w_243_0 Con := by
    unfold generatedRoot.w_428_0 generatedRoot.w_427_0; rw [k426]; rfl
  have k433 : generatedRoot.w_433_0 Con = onePubKey0 Con 1 * generatedRoot.w_243_0 Con := by
    unfold generatedRoot.w_433_0 generatedRoot.w_432_0; rw [k431]; rfl
  refine ⟨![f0, f1], fun j ↦ ?_, fun j ↦ ?_⟩
  · fin_cases j
    · exact coeffBound_mono hf0 (le_max_left _ _)
    · exact coeffBound_mono hf1 (le_max_right _ _)
  · fin_cases j
    · show generatedRoot.w_244_0 Con = _
      rw [h0]
      rfl
    · show generatedRoot.w_245_0 Con = _
      rw [h1, k428, k433]
      show _ = reducePoly modulus_0 8 sm • (onePubKey0 Con 0 * generatedRoot.w_243_0 Con) +
        onePubKey0 Con 1 * generatedRoot.w_243_0 Con -
        reducePoly modulus_0 8 (1 : ErrorPoly 8) • generatedRoot.w_229_0 Con +
        reduceMatrix modulus_0 8 1 25 f1
      rw [map_one, one_smul]
      rfl
/-- The error bound of the `cₓ` pair `lab425`. -/
def lab425Bound (I : Online0Inputs Con sm) : Nat :=
  25 * 8 * (25 * 8 * ((25 + 25) * 8 * I.Ba * 2048 + 8 * 1 * I.Ba) * (4096 / 2)) * (4096 / 2) +
    25 * 8 * (25 * 8 * ((25 + 25) * 8 * I.Ba * 2048 + 8 * 1 * I.Ba) * (4096 / 2)) * (4096 / 2) + 8 * 1 * I.pubBound

/-- The error bound of the `cₓ` pair `lab543`. -/
def lab543Bound (I : Online0Inputs Con sm) : Nat :=
  25 * 8 * (25 * 8 * ((25 + 25) * 8 * ((25 + 25) * 8 * I.Ba * 2048 + 8 * 1 * I.Ba) * 2048 +
      8 * 1 * ((25 + 25) * 8 * I.Ba * 2048 + 8 * 1 * I.Ba)) * (4096 / 2)) * (4096 / 2) +
    25 * 8 * (25 * 8 * ((25 + 25) * 8 * ((25 + 25) * 8 * I.Ba * 2048 + 8 * 1 * I.Ba) * 2048 +
      8 * 1 * ((25 + 25) * 8 * I.Ba * 2048 + 8 * 1 * I.Ba)) * (4096 / 2)) * (4096 / 2) + 8 * 1 * I.pubBound

/-- The error bound of the `cₓ` pair `lab626`. -/
def lab626Bound (I : Online0Inputs Con sm) : Nat :=
  25 * 8 * (25 * 8 * ((25 + 25) * 8 * ((25 + 25) * 8 * I.Ba * 2048 + 8 * 1 * I.Ba) * 2048 + 8 * 1 * I.Ba) * (4096 / 2)) * (4096 / 2) +
    25 * 8 * (25 * 8 * ((25 + 25) * 8 * ((25 + 25) * 8 * I.Ba * 2048 + 8 * 1 * I.Ba) * 2048 + 8 * 1 * I.Ba) * (4096 / 2)) * (4096 / 2) + 8 * 1 * I.pubBound

/-- The error bound of the `cₓ` pair `lab680`. -/
def lab680Bound (I : Online0Inputs Con sm) : Nat :=
  25 * 8 * (25 * 8 * ((25 + 25) * 8 * I.Ba * 2048 + 8 * 1 * I.Bo) * (4096 / 2)) * (4096 / 2) +
    25 * 8 * (25 * 8 * ((25 + 25) * 8 * I.Ba * 2048 + 8 * 1 * I.Bo) * (4096 / 2)) * (4096 / 2) + 8 * 1 * I.pubBound

/-- The error bound of the `cₓ` pair `lab734`. -/
def lab734Bound (I : Online0Inputs Con sm) : Nat :=
  25 * 8 * (25 * 8 * ((25 + 25) * 8 * I.Ba * 2048 + 8 * 1 * I.Ba) * (4096 / 2)) * (4096 / 2) +
    25 * 8 * (25 * 8 * ((25 + 25) * 8 * I.Ba * 2048 + 8 * 1 * I.Ba) * (4096 / 2)) * (4096 / 2) + 8 * 1 * I.pubBound

/-- The error bound of the `cₓ` pair `sig`. -/
def sigBound (I : Online0Inputs Con sm) : Nat :=
  max (25 * 8 * (25 * 8 * I.σBound * 2048) * 2048) (25 * 8 * I.c1Bound * 2048)

/-- The error bound of the `cₓ` pair `aut3`. -/
def aut3Bound (I : Online0Inputs Con sm) : Nat :=
  max (25 * 8 * (25 * 8 * ((25 + 25) * 8 * I.a3Bh * 2048 + (25 + 25) * 8 * I.σBound * 2048 +
    8 * 1 * I.a3BB) * 2048) * 2048) (25 * 8 * I.c1Bound * 2048)

/-- The error bound of the `cₓ` pair `aut5`. -/
def aut5Bound (I : Online0Inputs Con sm) : Nat :=
  max (25 * 8 * (25 * 8 * ((25 + 25) * 8 * I.a5Bh * 2048 + (25 + 25) * 8 * I.σBound * 2048 +
    8 * 1 * I.a5BB) * 2048) * 2048) (25 * 8 * I.c1Bound * 2048)

/-- The error bound of the `cₓ` pair `aut7`. -/
def aut7Bound (I : Online0Inputs Con sm) : Nat :=
  max (25 * 8 * (25 * 8 * ((25 + 25) * 8 * I.a7Bh * 2048 + (25 + 25) * 8 * I.σBound * 2048 +
    8 * 1 * I.a7BB) * 2048) * 2048) (25 * 8 * I.c1Bound * 2048)

/-- The error bound of the `cₓ` pair `aut9`. -/
def aut9Bound (I : Online0Inputs Con sm) : Nat :=
  max (25 * 8 * (25 * 8 * ((25 + 25) * 8 * I.a9Bh * 2048 + (25 + 25) * 8 * I.σBound * 2048 +
    8 * 1 * I.a9BB) * 2048) * 2048) (25 * 8 * I.c1Bound * 2048)

/-- The error bound of the `cₓ` pair `aut11`. -/
def aut11Bound (I : Online0Inputs Con sm) : Nat :=
  max (25 * 8 * (25 * 8 * ((25 + 25) * 8 * I.a11Bh * 2048 + (25 + 25) * 8 * I.σBound * 2048 +
    8 * 1 * I.a11BB) * 2048) * 2048) (25 * 8 * I.c1Bound * 2048)

/-- The error bound of the `cₓ` pair `aut13`. -/
def aut13Bound (I : Online0Inputs Con sm) : Nat :=
  max (25 * 8 * (25 * 8 * ((25 + 25) * 8 * I.a13Bh * 2048 + (25 + 25) * 8 * I.σBound * 2048 +
    8 * 1 * I.a13BB) * 2048) * 2048) (25 * 8 * I.c1Bound * 2048)

/-- The error bound of the `cₓ` pair `aut15`. -/
def aut15Bound (I : Online0Inputs Con sm) : Nat :=
  max (25 * 8 * (25 * 8 * ((25 + 25) * 8 * I.a15Bh * 2048 + (25 + 25) * 8 * I.σBound * 2048 +
    8 * 1 * I.a15BB) * 2048) * 2048) (25 * 8 * I.c1Bound * 2048)

/-- The plaintexts `x_m` of the 36 `cₓ` members of step 0 (member `2 t + j` is component `j` of
target `t`; members `32 … 35` are zero). -/
noncomputable def cxPlain (I : Online0Inputs Con sm) : Fin 36 → ErrorPoly 8 :=
  ![I.tp,
    1,
    I.xσ,
    1,
    I.stΔ * I.stν * I.sp,
    I.stΔ * I.stν,
    I.lv 1 * I.lv 3 * I.xσ,
    I.lv 1 * I.lv 3,
    I.lv 0 * I.lv 1 * (I.lv 2 * I.lv 3) * I.xσ,
    I.lv 0 * I.lv 1 * (I.lv 2 * I.lv 3),
    I.lv 0 * I.lv 2 * I.lv 3 * I.xσ,
    I.lv 0 * I.lv 2 * I.lv 3,
    I.lv 1 * I.βo * I.xσ,
    I.lv 1 * I.βo,
    I.lv 1 * I.lv 2 * I.xσ,
    I.lv 1 * I.lv 2,
    I.xσ,
    1,
    I.a3Δ * I.a3ν,
    1,
    I.a5Δ * I.a5ν,
    1,
    I.a7Δ * I.a7ν,
    1,
    I.a9Δ * I.a9ν,
    1,
    I.a11Δ * I.a11ν,
    1,
    I.a13Δ * I.a13ν,
    1,
    I.a15Δ * I.a15ν,
    1,
    0,
    0,
    0,
    0]

/-- The error bounds of the 36 `cₓ` members of step 0. -/
def cxBound (I : Online0Inputs Con sm) : Fin 36 → Nat :=
  ![25 * 8 * I.privB * 2048,
    25 * 8 * I.privB * 2048,
    I.pubBound,
    I.pubBound,
    25 * 8 * I.stBound * 2048,
    25 * 8 * I.stBound * 2048,
    I.lab425Bound,
    I.lab425Bound,
    I.lab543Bound,
    I.lab543Bound,
    I.lab626Bound,
    I.lab626Bound,
    I.lab680Bound,
    I.lab680Bound,
    I.lab734Bound,
    I.lab734Bound,
    I.sigBound,
    I.sigBound,
    I.aut3Bound,
    I.aut3Bound,
    I.aut5Bound,
    I.aut5Bound,
    I.aut7Bound,
    I.aut7Bound,
    I.aut9Bound,
    I.aut9Bound,
    I.aut11Bound,
    I.aut11Bound,
    I.aut13Bound,
    I.aut13Bound,
    I.aut15Bound,
    I.aut15Bound,
    0,
    0,
    0,
    0]
/-- A scalar times a component of a pair target, as `(1 · x) G`. -/
theorem _root_.Dio2Proof.smul_pair_scalar {q : Nat} (a y x : ErrorPoly 8) (j : Fin 2)
    (G : ExactMatrix q 8 1 25) (h : a * unitPair y j = x) :
    reducePoly q 8 a • (reducePoly q 8 (unitPair y j) • G) =
      reducePoly q 8 (((1 : Nat) : Int) * x) • G := by
  rw [smul_smul, ← map_mul, h]
  simp

/-- A `cₓ` form with equal key functions. -/
theorem _root_.Dio2Proof.CxRow.congr_keys {K K' : Fin 2 → ExactMatrix modulus_0 8 1 25} {x : ErrorPoly 8}
    {B : Nat} {row : ExactMatrix modulus_0 8 1 25} (h : CxRow Con sm K x B row) (hK : K = K') :
    CxRow Con sm K' x B row := hK ▸ h

/-- The one-private members `0`, `1` (`online0_target_priv`). -/
theorem cx_priv {outputs : _} (Fon : generatedRoot.Facts Con outputs)
    (hq1 : 1 < modulus_0) (layout : RegularLayout modulus_0)
    (hlayout : backend.regularLayout modulus_0 8 = some layout) (hexact : layout.droppedModuli = 0) (j : Fin 2) :
    CxRow Con sm (cxKeys0 Con ⟨0 + j, by omega⟩) (![I.tp, 1] j) (25 * 8 * I.privB * 2048)
      (![generatedRoot.w_237_0 Con, generatedRoot.w_238_0 Con] j) := by
  refine CxRow.congr_keys (cxRow_of_form (online0_target_priv Con Fon hq1 layout hlayout hexact I.priv j) ?_) ?_
  · fin_cases j <;> simp [unitPair]
  · funext k; fin_cases j <;> fin_cases k <;> rfl

/-- The keys `w_428`, `w_433` of `E^σ`'s second component are the one-public keys (hash
determinism). -/
theorem pub_keys {outputs : _} (Fon : generatedRoot.Facts Con outputs) :
    generatedRoot.w_428_0 Con = onePubKey0 Con 0 * generatedRoot.w_243_0 Con ∧
      generatedRoot.w_433_0 Con = onePubKey0 Con 1 * generatedRoot.w_243_0 Con := by
  have k426 : generatedRoot.w_426_0 Con =
      labelKeysG hashModel (generatedRoot.w_7_0 Con) modulus_0 25 labelPre 1 0 1 := by
    rw [MxxExponentLut.hashSample_eq Fon.c15.h426_0]; rfl
  have k431 : generatedRoot.w_431_0 Con =
      labelKeysG hashModel (generatedRoot.w_7_0 Con) modulus_0 25 labelPre 1 1 1 := by
    rw [MxxExponentLut.hashSample_eq Fon.c15.h431_0]; rfl
  constructor
  · unfold generatedRoot.w_428_0 generatedRoot.w_427_0; rw [k426]; rfl
  · unfold generatedRoot.w_433_0 generatedRoot.w_432_0; rw [k431]; rfl

/-- The `E^σ` members `2`, `3` (`online0_target_pub`). -/
theorem cx_pub {outputs : _} (Fon : generatedRoot.Facts Con outputs)
    (hq1 : 1 < modulus_0) (layout : RegularLayout modulus_0)
    (hlayout : backend.regularLayout modulus_0 8 = some layout) (hexact : layout.droppedModuli = 0) (j : Fin 2) :
    CxRow Con sm (cxKeys0 Con ⟨2 + j, by omega⟩) (![I.xσ, 1] j) I.pubBound
      (![generatedRoot.w_244_0 Con, generatedRoot.w_245_0 Con] j) := by
  obtain ⟨k428, k433⟩ := pub_keys (Con := Con) Fon
  refine CxRow.congr_keys (cxRow_of_rel (I.pub_rel Fon hq1 layout hlayout hexact) j ?_) ?_
  · fin_cases j <;> simp [unitPair]
  · funext k; fin_cases j <;> fin_cases k
    · rfl
    · rfl
    · exact k428
    · exact k433

/-- The state members `4`, `5` (`online0_state_lut`, `online0_target_state`). -/
theorem cx_state {outputs : _} (Fon : generatedRoot.Facts Con outputs)
    (hq1 : 1 < modulus_0) (layout : RegularLayout modulus_0)
    (hlayout : backend.regularLayout modulus_0 8 = some layout) (hexact : layout.droppedModuli = 0) (j : Fin 2) :
    CxRow Con sm (cxKeys0 Con ⟨4 + j, by omega⟩) (![I.stΔ * I.stν * I.sp, I.stΔ * I.stν] j)
      (25 * 8 * I.stBound * 2048)
      (![generatedRoot.w_356_0 Con, generatedRoot.w_371_0 Con] j) := by
  obtain ⟨⟨e0, he0, h0⟩, ⟨e1, he1, h1⟩⟩ := online0_state_lut Con Fon hq1 layout hlayout hexact I.hsm
    Fon.c1.h57_0 (centeredRebaseRuns_self hq1 _) I.stEnc I.stHelp I.stSwitch I.stHμ I.stWalsh
  have hT := online0_target_state Con Fon hq1 layout hlayout hexact (s := reducePoly modulus_0 8 sm)
    (K := fun k jj ↦ ![![generatedRoot.w_317_2 Con + MxxExponentLut.rowOf (matrixMul
        (generatedRoot.w_327_0 Con) (generatedRoot.w_345_0 Con)) 1,
      generatedRoot.w_317_4 Con + MxxExponentLut.rowOf (matrixMul (generatedRoot.w_327_0 Con)
        (generatedRoot.w_345_0 Con)) 2],
      ![generatedRoot.w_317_3 Con + MxxExponentLut.rowOf (matrixMul (generatedRoot.w_327_0 Con)
        (generatedRoot.w_365_0 Con)) 1,
      generatedRoot.w_317_5 Con + MxxExponentLut.rowOf (matrixMul (generatedRoot.w_327_0 Con)
        (generatedRoot.w_365_0 Con)) 2]] jj k)
    (x := fun jj ↦ reducePoly modulus_0 8 (I.stΔ * I.stν * unitPair I.sp jj))
    (e := ![e0, e1]) (B := I.stBound) (fun jj ↦ by fin_cases jj; exacts [he0, he1])
    (fun jj ↦ by fin_cases jj; exacts [h0, h1]) j
  refine CxRow.congr_keys (cxRow_of_form hT ?_) ?_
  · fin_cases j <;> simp [unitPair]
  · funext k; fin_cases j <;> fin_cases k <;> rfl

/-- The label members `6`, `7` (`online0_label_425`). -/
theorem cx_label_425 {outputs : _} (Fon : generatedRoot.Facts Con outputs)
    (hq1 : 1 < modulus_0) (layout : RegularLayout modulus_0)
    (hlayout : backend.regularLayout modulus_0 8 = some layout) (hexact : layout.droppedModuli = 0) (j : Fin 2) :
    CxRow Con sm (cxKeys0 Con ⟨6 + j, by omega⟩) (![I.lv 1 * I.lv 3 * I.xσ, I.lv 1 * I.lv 3] j) I.lab425Bound
      (![generatedRoot.w_425_0 Con, generatedRoot.w_438_0 Con] j) := by
  have hs := I.pub_rel Fon hq1 layout hlayout hexact
  refine CxRow.congr_keys (cxRow_of_rel (online0_label_425 Con Fon hq1 layout hlayout hexact I.hα I.hα13 I.lab I.pay hs) j ?_) ?_
  · fin_cases j <;> exact smul_pair_scalar _ _ _ _ _ (by simp [unitPair])
  · funext k; fin_cases j <;> fin_cases k <;> rfl

/-- The label members `8`, `9` (`online0_label_543`). -/
theorem cx_label_543 {outputs : _} (Fon : generatedRoot.Facts Con outputs)
    (hq1 : 1 < modulus_0) (layout : RegularLayout modulus_0)
    (hlayout : backend.regularLayout modulus_0 8 = some layout) (hexact : layout.droppedModuli = 0) (j : Fin 2) :
    CxRow Con sm (cxKeys0 Con ⟨8 + j, by omega⟩) (![I.lv 0 * I.lv 1 * (I.lv 2 * I.lv 3) * I.xσ, I.lv 0 * I.lv 1 * (I.lv 2 * I.lv 3)] j) I.lab543Bound
      (![generatedRoot.w_543_0 Con, generatedRoot.w_550_0 Con] j) := by
  have hs := I.pub_rel Fon hq1 layout hlayout hexact
  refine CxRow.congr_keys (cxRow_of_rel (online0_label_543 Con Fon hq1 layout hlayout hexact I.hα I.hα01 I.hα0123 I.lab I.pay hs) j ?_) ?_
  · fin_cases j <;> exact smul_pair_scalar _ _ _ _ _ (by simp [unitPair])
  · funext k; fin_cases j <;> fin_cases k <;> rfl

/-- The label members `10`, `11` (`online0_label_626`). -/
theorem cx_label_626 {outputs : _} (Fon : generatedRoot.Facts Con outputs)
    (hq1 : 1 < modulus_0) (layout : RegularLayout modulus_0)
    (hlayout : backend.regularLayout modulus_0 8 = some layout) (hexact : layout.droppedModuli = 0) (j : Fin 2) :
    CxRow Con sm (cxKeys0 Con ⟨10 + j, by omega⟩) (![I.lv 0 * I.lv 2 * I.lv 3 * I.xσ, I.lv 0 * I.lv 2 * I.lv 3] j) I.lab626Bound
      (![generatedRoot.w_626_0 Con, generatedRoot.w_633_0 Con] j) := by
  have hs := I.pub_rel Fon hq1 layout hlayout hexact
  refine CxRow.congr_keys (cxRow_of_rel (online0_label_626 Con Fon hq1 layout hlayout hexact I.hα I.lab I.pay hs I.hα02 I.hα023) j ?_) ?_
  · fin_cases j <;> exact smul_pair_scalar _ _ _ _ _ (by simp [unitPair])
  · funext k; fin_cases j <;> fin_cases k <;> rfl

/-- The label members `12`, `13` (`online0_label_680`). -/
theorem cx_label_680 {outputs : _} (Fon : generatedRoot.Facts Con outputs)
    (hq1 : 1 < modulus_0) (layout : RegularLayout modulus_0)
    (hlayout : backend.regularLayout modulus_0 8 = some layout) (hexact : layout.droppedModuli = 0) (j : Fin 2) :
    CxRow Con sm (cxKeys0 Con ⟨12 + j, by omega⟩) (![I.lv 1 * I.βo * I.xσ, I.lv 1 * I.βo] j) I.lab680Bound
      (![generatedRoot.w_680_0 Con, generatedRoot.w_687_0 Con] j) := by
  have hs := I.pub_rel Fon hq1 layout hlayout hexact
  refine CxRow.congr_keys (cxRow_of_rel (online0_label_680 Con Fon hq1 layout hlayout hexact I.hα I.lab I.pay hs I.oneS I.pay0 I.hα1β) j ?_) ?_
  · fin_cases j <;> exact smul_pair_scalar _ _ _ _ _ (by simp [unitPair])
  · funext k; fin_cases j <;> fin_cases k <;> rfl

/-- The label members `14`, `15` (`online0_label_734`). -/
theorem cx_label_734 {outputs : _} (Fon : generatedRoot.Facts Con outputs)
    (hq1 : 1 < modulus_0) (layout : RegularLayout modulus_0)
    (hlayout : backend.regularLayout modulus_0 8 = some layout) (hexact : layout.droppedModuli = 0) (j : Fin 2) :
    CxRow Con sm (cxKeys0 Con ⟨14 + j, by omega⟩) (![I.lv 1 * I.lv 2 * I.xσ, I.lv 1 * I.lv 2] j) I.lab734Bound
      (![generatedRoot.w_734_0 Con, generatedRoot.w_741_0 Con] j) := by
  have hs := I.pub_rel Fon hq1 layout hlayout hexact
  refine CxRow.congr_keys (cxRow_of_rel (online0_label_734 Con Fon hq1 layout hlayout hexact I.hα I.lab I.pay hs I.hα12) j ?_) ?_
  · fin_cases j <;> exact smul_pair_scalar _ _ _ _ _ (by simp [unitPair])
  · funext k; fin_cases j <;> fin_cases k <;> rfl

/-- The σ-carrier members `16`, `17` (`online0_target_sigma`). -/
theorem cx_sigma {outputs : _} (Fon : generatedRoot.Facts Con outputs)
    (hq1 : 1 < modulus_0) (layout : RegularLayout modulus_0)
    (hlayout : backend.regularLayout modulus_0 8 = some layout) (hexact : layout.droppedModuli = 0) (j : Fin 2) :
    CxRow Con sm (cxKeys0 Con ⟨16 + j, by omega⟩) (![I.xσ, 1] j) I.sigBound
      (![generatedRoot.w_753_0 Con, generatedRoot.w_754_0 Con] j) := by
  obtain ⟨e1, he1, hc⟩ := I.c1_form Fon hq1 layout hlayout hexact
  rw [online0_gadget_eq Con Fon] at hc
  refine CxRow.congr_keys (cxRow_of_rel (online0_target_sigma Con Fon hq1 layout hlayout hexact
    (I.sigma_total Fon hq1 layout hlayout hexact) he1 hc) j ?_) ?_
  · fin_cases j <;> simp [unitPair, xσ]
  · funext k; fin_cases j <;> fin_cases k <;> rfl

/-- The `Aut(3)` members `18`, `19` (`online0_aut_3`). -/
theorem cx_aut_3 {outputs : _} (Fon : generatedRoot.Facts Con outputs)
    (hq1 : 1 < modulus_0) (layout : RegularLayout modulus_0)
    (hlayout : backend.regularLayout modulus_0 8 = some layout) (hexact : layout.droppedModuli = 0) (j : Fin 2) :
    CxRow Con sm (cxKeys0 Con ⟨18 + j, by omega⟩) (![I.a3Δ * I.a3ν, 1] j) I.aut3Bound
      (![generatedRoot.w_811_0 Con, generatedRoot.w_812_0 Con] j) := by
  obtain ⟨e1, he1, hc⟩ := I.c1_form Fon hq1 layout hlayout hexact
  rw [online0_gadget_eq Con Fon] at hc
  refine CxRow.congr_keys (cxRow_of_rel (online0_aut_3 Con Fon hq1 layout hlayout hexact I.hsm
    (I.sigma_rel Fon hq1 layout hlayout hexact) I.a3help I.a3switch I.a3hμ I.a3walsh he1 hc) j ?_) ?_
  · fin_cases j <;> simp [unitPair]
  · funext k; fin_cases j <;> fin_cases k <;> rfl

/-- The `Aut(5)` members `20`, `21` (`online0_aut_5`). -/
theorem cx_aut_5 {outputs : _} (Fon : generatedRoot.Facts Con outputs)
    (hq1 : 1 < modulus_0) (layout : RegularLayout modulus_0)
    (hlayout : backend.regularLayout modulus_0 8 = some layout) (hexact : layout.droppedModuli = 0) (j : Fin 2) :
    CxRow Con sm (cxKeys0 Con ⟨20 + j, by omega⟩) (![I.a5Δ * I.a5ν, 1] j) I.aut5Bound
      (![generatedRoot.w_869_0 Con, generatedRoot.w_870_0 Con] j) := by
  obtain ⟨e1, he1, hc⟩ := I.c1_form Fon hq1 layout hlayout hexact
  rw [online0_gadget_eq Con Fon] at hc
  refine CxRow.congr_keys (cxRow_of_rel (online0_aut_5 Con Fon hq1 layout hlayout hexact I.hsm
    (I.sigma_rel Fon hq1 layout hlayout hexact) I.a5help I.a5switch I.a5hμ I.a5walsh he1 hc) j ?_) ?_
  · fin_cases j <;> simp [unitPair]
  · funext k; fin_cases j <;> fin_cases k <;> rfl

/-- The `Aut(7)` members `22`, `23` (`online0_aut_7`). -/
theorem cx_aut_7 {outputs : _} (Fon : generatedRoot.Facts Con outputs)
    (hq1 : 1 < modulus_0) (layout : RegularLayout modulus_0)
    (hlayout : backend.regularLayout modulus_0 8 = some layout) (hexact : layout.droppedModuli = 0) (j : Fin 2) :
    CxRow Con sm (cxKeys0 Con ⟨22 + j, by omega⟩) (![I.a7Δ * I.a7ν, 1] j) I.aut7Bound
      (![generatedRoot.w_927_0 Con, generatedRoot.w_928_0 Con] j) := by
  obtain ⟨e1, he1, hc⟩ := I.c1_form Fon hq1 layout hlayout hexact
  rw [online0_gadget_eq Con Fon] at hc
  refine CxRow.congr_keys (cxRow_of_rel (online0_aut_7 Con Fon hq1 layout hlayout hexact I.hsm
    (I.sigma_rel Fon hq1 layout hlayout hexact) I.a7help I.a7switch I.a7hμ I.a7walsh he1 hc) j ?_) ?_
  · fin_cases j <;> simp [unitPair]
  · funext k; fin_cases j <;> fin_cases k <;> rfl

/-- The `Aut(9)` members `24`, `25` (`online0_aut_9`). -/
theorem cx_aut_9 {outputs : _} (Fon : generatedRoot.Facts Con outputs)
    (hq1 : 1 < modulus_0) (layout : RegularLayout modulus_0)
    (hlayout : backend.regularLayout modulus_0 8 = some layout) (hexact : layout.droppedModuli = 0) (j : Fin 2) :
    CxRow Con sm (cxKeys0 Con ⟨24 + j, by omega⟩) (![I.a9Δ * I.a9ν, 1] j) I.aut9Bound
      (![generatedRoot.w_985_0 Con, generatedRoot.w_986_0 Con] j) := by
  obtain ⟨e1, he1, hc⟩ := I.c1_form Fon hq1 layout hlayout hexact
  rw [online0_gadget_eq Con Fon] at hc
  refine CxRow.congr_keys (cxRow_of_rel (online0_aut_9 Con Fon hq1 layout hlayout hexact I.hsm
    (I.sigma_rel Fon hq1 layout hlayout hexact) I.a9help I.a9switch I.a9hμ I.a9walsh he1 hc) j ?_) ?_
  · fin_cases j <;> simp [unitPair]
  · funext k; fin_cases j <;> fin_cases k <;> rfl

/-- The `Aut(11)` members `26`, `27` (`online0_aut_11`). -/
theorem cx_aut_11 {outputs : _} (Fon : generatedRoot.Facts Con outputs)
    (hq1 : 1 < modulus_0) (layout : RegularLayout modulus_0)
    (hlayout : backend.regularLayout modulus_0 8 = some layout) (hexact : layout.droppedModuli = 0) (j : Fin 2) :
    CxRow Con sm (cxKeys0 Con ⟨26 + j, by omega⟩) (![I.a11Δ * I.a11ν, 1] j) I.aut11Bound
      (![generatedRoot.w_1043_0 Con, generatedRoot.w_1044_0 Con] j) := by
  obtain ⟨e1, he1, hc⟩ := I.c1_form Fon hq1 layout hlayout hexact
  rw [online0_gadget_eq Con Fon] at hc
  refine CxRow.congr_keys (cxRow_of_rel (online0_aut_11 Con Fon hq1 layout hlayout hexact I.hsm
    (I.sigma_rel Fon hq1 layout hlayout hexact) I.a11help I.a11switch I.a11hμ I.a11walsh he1 hc) j ?_) ?_
  · fin_cases j <;> simp [unitPair]
  · funext k; fin_cases j <;> fin_cases k <;> rfl

/-- The `Aut(13)` members `28`, `29` (`online0_aut_13`). -/
theorem cx_aut_13 {outputs : _} (Fon : generatedRoot.Facts Con outputs)
    (hq1 : 1 < modulus_0) (layout : RegularLayout modulus_0)
    (hlayout : backend.regularLayout modulus_0 8 = some layout) (hexact : layout.droppedModuli = 0) (j : Fin 2) :
    CxRow Con sm (cxKeys0 Con ⟨28 + j, by omega⟩) (![I.a13Δ * I.a13ν, 1] j) I.aut13Bound
      (![generatedRoot.w_1101_0 Con, generatedRoot.w_1102_0 Con] j) := by
  obtain ⟨e1, he1, hc⟩ := I.c1_form Fon hq1 layout hlayout hexact
  rw [online0_gadget_eq Con Fon] at hc
  refine CxRow.congr_keys (cxRow_of_rel (online0_aut_13 Con Fon hq1 layout hlayout hexact I.hsm
    (I.sigma_rel Fon hq1 layout hlayout hexact) I.a13help I.a13switch I.a13hμ I.a13walsh he1 hc) j ?_) ?_
  · fin_cases j <;> simp [unitPair]
  · funext k; fin_cases j <;> fin_cases k <;> rfl

/-- The `Aut(15)` members `30`, `31` (`online0_aut_15`). -/
theorem cx_aut_15 {outputs : _} (Fon : generatedRoot.Facts Con outputs)
    (hq1 : 1 < modulus_0) (layout : RegularLayout modulus_0)
    (hlayout : backend.regularLayout modulus_0 8 = some layout) (hexact : layout.droppedModuli = 0) (j : Fin 2) :
    CxRow Con sm (cxKeys0 Con ⟨30 + j, by omega⟩) (![I.a15Δ * I.a15ν, 1] j) I.aut15Bound
      (![generatedRoot.w_1159_0 Con, generatedRoot.w_1160_0 Con] j) := by
  obtain ⟨e1, he1, hc⟩ := I.c1_form Fon hq1 layout hlayout hexact
  rw [online0_gadget_eq Con Fon] at hc
  refine CxRow.congr_keys (cxRow_of_rel (online0_aut_15 Con Fon hq1 layout hlayout hexact I.hsm
    (I.sigma_rel Fon hq1 layout hlayout hexact) I.a15help I.a15switch I.a15hμ I.a15walsh he1 hc) j ?_) ?_
  · fin_cases j <;> simp [unitPair]
  · funext k; fin_cases j <;> fin_cases k <;> rfl

/-- The zero members `32 … 35`. -/
theorem cx_zero : CxRow Con sm (fun _ ↦ 0) 0 0 (generatedRoot.w_1161_0 Con) := by
  refine ⟨0, coeffBound_zero, ?_⟩
  show (0 : ExactMatrix modulus_0 8 1 25) = _
  funext r l
  simp [maskPart, reduceMatrix]

/-- **The 36 `cₓ` members of step 0** (the family `w_1162` the refresh parts read as `cₓ`): member
`m` is `s̄ (cxKeys0 m) - (1 · x_m) G + e_m` with `CoeffBound e_m (cxBound m)`. -/
theorem cx_rows {outputs : _} (Fon : generatedRoot.Facts Con outputs)
    (hq1 : 1 < modulus_0) (layout : RegularLayout modulus_0)
    (hlayout : backend.regularLayout modulus_0 8 = some layout) (hexact : layout.droppedModuli = 0) (m : Fin 36) :
    CxRow Con sm (cxKeys0 Con m) (I.cxPlain m) (I.cxBound m) (generatedRoot.w_1162_0 Con m) := by
  obtain ⟨_, e⟩ := familyPack_get Fon.c45.h1162_0 m
  rw [e]
  fin_cases m
  · exact I.cx_priv Fon hq1 layout hlayout hexact 0
  · exact I.cx_priv Fon hq1 layout hlayout hexact 1
  · exact I.cx_pub Fon hq1 layout hlayout hexact 0
  · exact I.cx_pub Fon hq1 layout hlayout hexact 1
  · exact I.cx_state Fon hq1 layout hlayout hexact 0
  · exact I.cx_state Fon hq1 layout hlayout hexact 1
  · exact I.cx_label_425 Fon hq1 layout hlayout hexact 0
  · exact I.cx_label_425 Fon hq1 layout hlayout hexact 1
  · exact I.cx_label_543 Fon hq1 layout hlayout hexact 0
  · exact I.cx_label_543 Fon hq1 layout hlayout hexact 1
  · exact I.cx_label_626 Fon hq1 layout hlayout hexact 0
  · exact I.cx_label_626 Fon hq1 layout hlayout hexact 1
  · exact I.cx_label_680 Fon hq1 layout hlayout hexact 0
  · exact I.cx_label_680 Fon hq1 layout hlayout hexact 1
  · exact I.cx_label_734 Fon hq1 layout hlayout hexact 0
  · exact I.cx_label_734 Fon hq1 layout hlayout hexact 1
  · exact I.cx_sigma Fon hq1 layout hlayout hexact 0
  · exact I.cx_sigma Fon hq1 layout hlayout hexact 1
  · exact I.cx_aut_3 Fon hq1 layout hlayout hexact 0
  · exact I.cx_aut_3 Fon hq1 layout hlayout hexact 1
  · exact I.cx_aut_5 Fon hq1 layout hlayout hexact 0
  · exact I.cx_aut_5 Fon hq1 layout hlayout hexact 1
  · exact I.cx_aut_7 Fon hq1 layout hlayout hexact 0
  · exact I.cx_aut_7 Fon hq1 layout hlayout hexact 1
  · exact I.cx_aut_9 Fon hq1 layout hlayout hexact 0
  · exact I.cx_aut_9 Fon hq1 layout hlayout hexact 1
  · exact I.cx_aut_11 Fon hq1 layout hlayout hexact 0
  · exact I.cx_aut_11 Fon hq1 layout hlayout hexact 1
  · exact I.cx_aut_13 Fon hq1 layout hlayout hexact 0
  · exact I.cx_aut_13 Fon hq1 layout hlayout hexact 1
  · exact I.cx_aut_15 Fon hq1 layout hlayout hexact 0
  · exact I.cx_aut_15 Fon hq1 layout hlayout hexact 1
  · exact cx_zero
  · exact cx_zero
  · exact cx_zero
  · exact cx_zero

/-- **The fresh cells of step 0** (`online0_fresh_row`, component `1`): block `cc` of `w_1219_5` is
`s̄ (freshCells0 cc) - E_cc G + e`. -/
theorem fresh_cells {outputs : _} (Fon : generatedRoot.Facts Con outputs)
    (hq1 : 1 < modulus_0) (layout : RegularLayout modulus_0)
    (hlayout : backend.regularLayout modulus_0 8 = some layout) (hexact : layout.droppedModuli = 0) (cc : Fin 900) :
    ∃ e : ErrorMatrix 8 1 25, CoeffBound e I.freshBound ∧
      blk (C := 900) (m := 25) (generatedRoot.w_1219_5 Con) cc =
        maskPart (reducePoly modulus_0 8 sm) (freshCells0 Con cc) -
          reducePoly modulus_0 8 (I.freshCell cc) • generatedRoot.w_57_0 Con +
          reduceMatrix modulus_0 8 1 25 e :=
  cell_form (by norm_num) cc (online0_fresh_row Con Fon hq1 layout hlayout hexact I.hsm Fon.c1.h57_0
    (centeredRebaseRuns_self hq1 _) I.fIn I.help I.fSwitch I.fHμ I.fWalsh cc)

/-- **The smudge cells of step 0** (`online0_smudge_row`, component `1`). -/
theorem smudge_cells {outputs : _} (Fon : generatedRoot.Facts Con outputs)
    (hq1 : 1 < modulus_0) (layout : RegularLayout modulus_0)
    (hlayout : backend.regularLayout modulus_0 8 = some layout) (hexact : layout.droppedModuli = 0) (cc : Fin 4500) :
    ∃ e : ErrorMatrix 8 1 25, CoeffBound e I.smudgeBound ∧
      blk (C := 4500) (m := 25) (generatedRoot.w_1273_5 Con) cc =
        maskPart (reducePoly modulus_0 8 sm) (smudgeCells0 Con cc) -
          reducePoly modulus_0 8 (I.smudgeCell cc) • generatedRoot.w_57_0 Con +
          reduceMatrix modulus_0 8 1 25 e :=
  cell_form (by norm_num) cc (online0_smudge_row Con Fon hq1 layout hlayout hexact I.hsm Fon.c1.h57_0
    (centeredRebaseRuns_self hq1 _) I.sIn I.help I.sSwitch I.sHμ I.sWalsh cc)

/-- The target scalar `T_i` of part `i` at prime `0`: `k x_i s₉ s₁₀` lifted to `Q_0`
(`k = K₀ / (Q_0 / q_0)`, `s₉ = 1`, `s₁₀ = 11560550400`), so that `hscalar` holds. -/
noncomputable def T0 (I : Online0Inputs Con sm) (i : Fin 36) : ExactPoly modulus_0 8 :=
  reducePoly modulus_0 8 (((761964914453472 : Int) : ErrorPoly 8) * I.cxPlain i *
    ((11560550400 : Int) : ErrorPoly 8))

/-- The fresh error `E_i` of part `i`: the fresh values of cells `25 i + l`. -/
noncomputable def E0 (I : Online0Inputs Con sm) (i : Fin 36) : ErrorMatrix 8 1 25 :=
  fun _ l ↦ I.freshCell ⟨25 * i + l.val, by have := i.isLt; have := l.isLt; omega⟩

/-- **The noise condition of part `i` at prime `0`** (`hnoise` of `online0_part_prime0`) with the
bounds of the derivation: hint `1` (`claim_hint0`), preimage `22011706` (`claim_preimages`),
`gBound`, `c1Bound`, `cxBound i`, the fresh and smudge part rows, and the residue bound `BM`. -/
def Noise0 (I : Online0Inputs Con sm) (i : Fin 36) : Prop :=
  2 * (54 * 8 * 1 * 22011706 + (25 * 8 * I.gBound * 2048 + 25 * 8 * I.c1Bound * 2048 +
      25 * 8 * I.cxBound i * 2048) + 25 * 8 * ((25 * 25) * 8 * I.freshBound * 2048) * 2048 +
      (25 * 25) * 8 * I.smudgeBound * 2048 + I.BM) <
    1606938044254428791916078459060479968194539524630727034176641

/-- The inputs the online stage of step 0 passes refresh part `i` (`generatedRoot.constraints_49`). -/
noncomputable abbrev PartInputs0 (Con : generatedRoot.Ctx backend hashModel params) (i : Fin 36) :=
  (generatedRoot.w_0_0 Con, ((generatedRoot.w_1_0 Con i), (generatedRoot.w_63_0 Con,
    (generatedRoot.w_70_0 Con, ((generatedRoot.w_227_0 Con i), ((generatedRoot.w_1162_0 Con i),
    (generatedRoot.w_1219_5 Con, (generatedRoot.w_57_0 Con, (generatedRoot.w_1273_5 Con,
    (generatedRoot.w_1274_0 Con, (generatedRoot.w_1276_0 Con, ((generatedRoot.w_1277_0 Con i),
    ((generatedRoot.w_1278_0 Con i), ((generatedRoot.w_1279_0 Con i),
    ((generatedRoot.w_1280_0 Con i), ())))))))))))))))

set_option maxHeartbeats 64000000 in
/-- **The prime-0 CRT term of refresh part `i` of step 0, assembled at the stage level.** Under the
online facts `Fon`, the keys stage `Fk` linked by `L`, the preimages solving against `A`
(`hsolve`, `hP`), the hint under the rows of `A` (`hb`), the undischarged inputs `I` and the noise
condition, the prime-0 CRT term of every part `i` is the CRT term of
`β A'_i[0] + A'_i[1] - T_i G' + E_i` with `A'_i = w_227 i` (the next step's public matrix),
`β = Δ' Σ_o X^o ν_o` (`xσ`), `G' = w_57`, `T_i = T0 i` and `E_i = E0 i`. -/
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
    (hnoise : ∀ i, I.Noise0 i)
    (i : Fin 36) (c : parallel_generatedRoot_1281.Ctx backend hashModel params i)
    (hc : c.inputs = PartInputs0 Con i) {oc : _} (Fc : parallel_generatedRoot_1281.Facts c oc) :
    TermOf modulus_1 1224430409422355437355602951403039070751149016291967033667224370016023747552
      (parallel_generatedRoot_1281.w_64_0 c)
      (reducePoly modulus_0 8 I.xσ • MxxExponentLut.rowOf (generatedRoot.w_227_0 Con i) 0 +
        MxxExponentLut.rowOf (generatedRoot.w_227_0 Con i) 1 - I.T0 i • generatedRoot.w_57_0 Con +
        reduceMatrix modulus_0 8 1 25 (I.E0 i)) := by
  -- The part's inputs.
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
  -- The rows.
  obtain ⟨eg, heg, hg⟩ := I.g_form Fon hq1 layout hlayout hexact
  obtain ⟨e₁, he₁, hc₁⟩ := I.c1_form Fon hq1 layout hlayout hexact
  obtain ⟨ex, hex, hcx⟩ := I.cx_rows Fon hq1 layout hlayout hexact i
  rw [← online0_gadget_eq Con Fon] at hcx
  obtain ⟨Pd, hPe, hPd⟩ := hP i
  choose ef hef hz using I.fresh_cells Fon hq1 layout hlayout hexact
  obtain ⟨ef', hef', hf⟩ := online0_part_fresh0 c Fc i.isLt hq1 layout hlayout hexact hG31
    (K := freshCells0 Con) (t := I.freshCell) (e := ef) (fun cc ↦ by rw [e29, e31]; exact hz cc) hef
  choose es hes hzs using I.smudge_cells Fon hq1 layout hlayout hexact
  obtain ⟨es', hes', hs⟩ := online0_part_smudge0 c Fc i.isLt hq1 layout hlayout hexact hG31
    (K := smudgeCells0 Con) (t := I.smudgeCell) (e := es) (fun cc ↦ by rw [e47, e31]; exact hzs cc)
    hes (h := I.hconst) (by rw [e49]; exact I.hconst_eq)
  obtain ⟨M, hM, hTH⟩ := I.hTH i
  have hpre := online0_hpre0_full Fon Fk L hq1 layout hlayout hexact hsolve i c hc Fc
  -- The CRT scalar.
  have hRq : modulus_0 = 1606938044254428791916078459060479968194539524630727034176641 * modulus_1 := by
    norm_num [modulus_0, modulus_1]
  have hq : 1 < modulus_1 := by norm_num [modulus_1]
  have hqR : modulus_1 ∣ modulus_0 := ⟨1606938044254428791916078459060479968194539524630727034176641,
    by rw [hRq, Nat.mul_comm]⟩
  have hscalar : reducePoly modulus_1 8 ((((761964914453472 : Int) : Int) : ErrorPoly 8) * I.cxPlain i) *
      ((1 : Int) : ExactPoly modulus_1 8) * ((11560550400 : Int) : ExactPoly modulus_1 8) =
      GenFinal.rho modulus_1 (I.T0 i) := by
    rw [T0, rho_reduce (by norm_num) hq hq1 hqR, map_mul, ← reducePoly_intCast modulus_1 11560550400]
    simp
  have h := online0_part_prime0 c Fc hRq layout hlayout hexact (k := 761964914453472) (by norm_num)
    (s := reducePoly modulus_0 8 sm) (β := reducePoly modulus_0 8 I.xσ) (x := I.cxPlain i)
    heg he₁ hex (by rw [e4, e14]; exact hg) (by rw [e5, e14]; exact hc₁) (by rw [e13, e15]; exact hcx)
    hscalar (Bk := MxxExponentLut.rowOf A) (by rw [e0]; exact hb) heb (by rw [e1]; exact hPe) hPd
    hpre hf hef' hs hes' (by rw [e59]; exact hTH) hM (hnoise i)
  rw [e8, e15] at h
  exact h

end Online0Inputs

/-! ## The claim level -/

set_option maxHeartbeats 64000000 in
/-- **The prime-0 CRT term of every refresh part of step 0, at the claim level.** Under
`W_GeneratedClaim.Runs` with the hint mask `sm` (`hmask`), the online stage of step 0 (`Con`, the
execution's stage `11`) satisfies: for the undischarged inputs `I` and the noise condition, the
prime-0 CRT term `w_64` of every part `i` (its context `c` reading the inputs the stage passes it)
is the CRT term (basis element `K₀`, prime `q₀ = modulus_1`) of
`β A'_i[0] + A'_i[1] - T_i G' + E_i`, with `A'_i = w_227 i`, `β = xσ`, `G' = w_57`, `T_i = T0 i`,
`E_i = E0 i`. The hint (`claim_hint0`), the preimages (`claim_preimages`), the key agreement
(`online0_hpre0_full` with `Link0`) and the CRT scalar are discharged. -/
theorem claim_online0_term0 {hashModel : HashModel} {external : GeneratedClaim.ExternalInputs}
    {execution : GeneratedClaim.Execution}
    (h : W_GeneratedClaim.Runs hashModel external execution) {sm : ErrorPoly 8}
    (hmask : execution.«stage_3» 0 0 = reducePoly W_Stage_initial.modulus_0 8 sm)
    (hq1 : 1 < W_Stage_online_0.modulus_0) (layout : RegularLayout W_Stage_online_0.modulus_0)
    (hlayout : Backend.backend.regularLayout W_Stage_online_0.modulus_0 8 = some layout)
    (hexact : layout.droppedModuli = 0) :
    ∃ Con : W_Stage_online_0.generatedRoot.Ctx Backend.backend hashModel
        W_GeneratedClaim.stage_11_params,
      W_Stage_online_0.generatedRoot.Facts Con execution.«stage_11» ∧
      ∀ (I : Online0Inputs Con sm), (∀ i, I.Noise0 i) →
      ∀ (i : Fin 36) (c : W_Stage_online_0.parallel_generatedRoot_1281.Ctx Backend.backend hashModel
          W_GeneratedClaim.stage_11_params i),
        c.inputs = Online0Inputs.PartInputs0 Con i →
        ∀ {oc : _}, W_Stage_online_0.parallel_generatedRoot_1281.Facts c oc →
        TermOf W_Stage_online_0.modulus_1
          1224430409422355437355602951403039070751149016291967033667224370016023747552
          (W_Stage_online_0.parallel_generatedRoot_1281.w_64_0 c)
          (reducePoly W_Stage_online_0.modulus_0 8 I.xσ •
              MxxExponentLut.rowOf (W_Stage_online_0.generatedRoot.w_227_0 Con i) 0 +
            MxxExponentLut.rowOf (W_Stage_online_0.generatedRoot.w_227_0 Con i) 1 -
            I.T0 i • W_Stage_online_0.generatedRoot.w_57_0 Con +
            reduceMatrix W_Stage_online_0.modulus_0 8 1 25 (I.E0 i)) := by
  have hp := (claim_preimages h).1.1
  obtain ⟨eb, heb, hb⟩ := claim_hint0 h hmask
  obtain ⟨-, -, -, -, -, -, -, -, -, h8, -, -, h11, -⟩ := h
  obtain ⟨wk, hk⟩ := h8
  obtain ⟨wo, ho⟩ := h11
  have Fk := W_Stage_keys_online_0_0.generatedRoot.facts ⟨_, wk⟩ hk
  have Fo := W_Stage_online_0.generatedRoot.facts ⟨_, wo⟩ ho
  refine ⟨_, Fo, fun I hnoise i c hc _ Fc ↦
    I.part_term0 Fo Fk (by constructor <;> rfl) hq1 layout hlayout hexact (A := execution.«stage_1».1)
      (fun j ↦ ?_) (fun j ↦ (hp j).2) heb hb hnoise i c hc Fc⟩
  have e := (hp j).1
  rw [Fk.c67.hout] at e
  exact e

end Dio2Proof
