import Dio2LutKeyBridge
import Dio2LutBridge
import GenFinalValues
import Gen_W_Stage_preprocess_0_0_Scopes
import Gen_W_Stage_preprocess_1_1_Scopes
import W_Stage_preprocess_0_0_Facts
import W_Stage_preprocess_1_1_Facts

/-!
# The LUT switching ciphertexts produced by dio2 preprocessing

Every switching table is encrypted without Gaussian noise. Consequently the switching
hypothesis of the online LUT lemmas holds with `Δ' = 1` and `BB = 0`, independently of
the preimage and trapdoor stages. The only secret hypothesis below identifies the
preprocessing input with the reduction of `sp`; the gadget is identified by its runtime
relation, not by an assumed switching equation.

`W_Claim` routes the first four outputs of each preprocess stage (`w_3_0` through
`w_3_3`) to `artifact:lut-switching/1/2/LutId(1)/...`, the state LUT. In online step 0
these are inputs `w_247`, `w_249`, `w_251`, `w_253` (top column 0, top column 1,
bottom column 0, bottom column 1). Outputs `w_4_*` are `LutId(0)` (smudge).
In step 0, `w_5_*`, `w_6_*`, `w_7_*`, `w_8_*`, `w_9_*`, and `w_10_*` through
`w_15_*` are respectively `LutId(11)`, `LutId(2)`, `LutId(10)`, `LutId(12)`,
`LutId(3)`, and `LutId(4)` through `LutId(9)`. All these families are covered below.

The coefficient definitions are the actual finite table sums encrypted by the graph.
The Walsh lemmas identify them with Fourier coefficients and prove the value identity
on a monomial input. Negacyclic wrap contributes a sign; no unsigned table lookup is
asserted without the corresponding range hypothesis. Transport to an online input
family uses equality of the routed families, as supplied by the claim wiring.
-/

namespace Dio2Proof.PreprocessSwitch

open Mxx.Primitives MxxRuntime MxxExponentLut RefreshTopProof.Gen RefreshTopProof.GenFinal
  Dio2Proof.LutKey
open scoped BigOperators

variable {n : Nat}

/-- The explicit coefficient encrypted for an exponent-table branch. -/
noncomputable def exponentCoefficient {q W : Nat} (sigma stride outputStride inverse : Int)
    (table : Fin W → Int) : ExactPoly q n :=
  (keyTableG [1] [inverse] (∑ k ∈ Finset.range W,
    keyTermE sigma outputStride table (matrixPolynomial [1]) (-stride) k) :
      ExactMatrix q n 1 1) 0 0

/-- The explicit coefficient encrypted for a coefficient-table branch. -/
noncomputable def coefficientCoefficient {q W : Nat} (sigma stride inverse : Int)
    (table : Fin W → Int) : ExactPoly q n :=
  (keyTableG [1] [inverse] (∑ k ∈ Finset.range W,
    keyTermC sigma table (matrixPolynomial [1]) (-stride) k) : ExactMatrix q n 1 1) 0 0

/-- The online switching premise with the exact preprocessing error budget `BB = 0`. -/
def Switching {q m W : Nat} (sp : ErrorPoly n) (sigma : Fin W → Int)
    (odd : ∀ i, Odd (sigma i).toNat) (coeff : Fin W → ExactPoly q n)
    (family : Fin 2 → Fin 2 → Fin W → ExactMatrix q n 1 m)
    (gadget : ExactMatrix q n 1 m) : Prop :=
  ∀ column branch, ∃ err : ErrorMatrix n 1 m, CoeffBound err 0 ∧
    reducePoly q n (substitution (sigma branch).toNat (odd branch) sp) •
        family 0 column branch + family 1 column branch =
      (coeff branch * reducePoly q n (1 * unitPair sp column)) • gadget +
        reduceMatrix q n 1 m err

/-- Routing a preprocessing family preserves the online switching premise. -/
theorem switching_transport {q m W : Nat} {sp : ErrorPoly n} {sigma : Fin W → Int}
    {odd : ∀ i, Odd (sigma i).toNat} {coeff : Fin W → ExactPoly q n}
    {family routed : Fin 2 → Fin 2 → Fin W → ExactMatrix q n 1 m}
    {gadget : ExactMatrix q n 1 m} (h : Switching sp sigma odd coeff family gadget)
    (hroute : routed = family) : Switching sp sigma odd coeff routed gadget := by
  rw [hroute]
  exact h

/-- The same switching premise with one error family per branch, as in `LutMemberFlat.rel`. -/
theorem switching_errors {q m W : Nat} {sp : ErrorPoly n} {sigma : Fin W → Int}
    {odd : ∀ i, Odd (sigma i).toNat} {coeff : Fin W → ExactPoly q n}
    {family : Fin 2 → Fin 2 → Fin W → ExactMatrix q n 1 m}
    {gadget : ExactMatrix q n 1 m} (h : Switching sp sigma odd coeff family gadget) :
    ∀ branch, ∃ err : Fin 2 → ErrorMatrix n 1 m, (∀ column, CoeffBound (err column) 0) ∧
      ∀ column, reducePoly q n (substitution (sigma branch).toNat (odd branch) sp) •
          family 0 column branch + family 1 column branch =
        (coeff branch * reducePoly q n (1 * unitPair sp column)) • gadget +
          reduceMatrix q n 1 m (err column) := by
  intro branch
  choose err hbound heq using fun column ↦ h column branch
  exact ⟨err, hbound, heq⟩

/-- A coefficient table is its normalized Fourier coefficient at the branch automorphism. -/
theorem coefficientCoefficient_fourier {q W stride : Nat} (hq : 1 < q) (hn : 0 < n)
    (hperiod : n = stride * W) {sigma : Int} (hodd : Odd sigma.toNat)
    (inverse : Int) (table : Fin W → Int) :
    coefficientCoefficient (q := q) sigma stride inverse table =
      lutFourierCoefficient W (X q ^ stride) (inverse : ExactPoly q n)
        (fun k ↦ (table k : ExactPoly q n)) sigma.toNat := by
  simpa [coefficientCoefficient, keyTableG, keyTermC] using
    keyTableC_walsh hq hn hodd hperiod 1 inverse table

/-- An exponent table is its normalized Fourier coefficient, with its output stride. -/
theorem exponentCoefficient_fourier {q W stride outputStride : Nat} (hq : 1 < q)
    (hn : 0 < n) (hperiod : n = stride * W) {sigma : Int} (hodd : Odd sigma.toNat)
    (inverse : Int) (table : Fin W → Int) (hnonneg : ∀ k, 0 ≤ table k) :
    exponentCoefficient (q := q) sigma stride outputStride inverse table =
      lutFourierCoefficient W (X q ^ stride) (inverse : ExactPoly q n)
        (fun k ↦ reducePoly q n (Xe (n := n) (outputStride * (table k).toNat)))
        sigma.toNat := by
  have hcast : ∀ k, ((table k).toNat : Int) = table k :=
    fun k ↦ Int.toNat_of_nonneg (hnonneg k)
  have h := keyTableE_walsh hq hn hodd hperiod 1 inverse outputStride
    (fun k ↦ (table k).toNat)
  simp only [hcast] at h
  simpa [exponentCoefficient, keyTableG, keyTermE] using h

/-- Fourier coefficients discharge `hwalsh` on a monomial input, including the wrap sign. -/
theorem walsh_of_fourier {q inverse stride log : Nat} (hperiod : n = stride * 2 ^ log)
    (hn : 0 < n) (table : Fin (2 ^ log) → ErrorPoly n)
    (hinverse : (inverse : ExactPoly q n) * ((2 ^ log : Nat) : ExactPoly q n) = 1)
    (sigma : Fin (2 ^ log) → Int) (odd : ∀ i, Odd (sigma i).toNat)
    (perm : Fin (2 ^ log) ≃ Fin (2 ^ log))
    (hindex : ∀ i, (sigma i).toNat % (2 * 2 ^ log) = 1 + 2 * 1 * (perm i).val)
    {coeff : Fin (2 ^ log) → ExactPoly q n}
    (hcoeff : ∀ i, coeff i = lutFourierCoefficient (2 ^ log) (X q ^ stride)
      (inverse : ExactPoly q n) (fun k ↦ reducePoly q n (table k)) (sigma i).toNat)
    {message : ErrorPoly n} (exponent : Nat)
    (hmessage : reducePoly q n message = reducePoly q n (Xe (n := n) (stride * exponent))) :
    ∑ i, reducePoly q n (substitution (sigma i).toNat (odd i) message) * coeff i =
      reducePoly q n ((((-1 : Int) ^ (exponent / 2 ^ log) : Int) : ErrorPoly n) *
        table ⟨exponent % 2 ^ log, Nat.mod_lt _ (by positivity)⟩) := by
  calc
    _ = ∑ i, reducePoly q n (substitution (sigma i).toNat (odd i)
        (Xe (n := n) (stride * exponent))) *
          lutFourierCoefficient (2 ^ log) (X q ^ stride) (inverse : ExactPoly q n)
            (fun k ↦ reducePoly q n (table k)) (sigma i).toNat := by
      apply Finset.sum_congr rfl
      intro i hi
      rw [hcoeff i, reducePoly_substitution, reducePoly_substitution, hmessage]
    _ = _ := lut_gen_value hperiod hn table hinverse (fun i ↦ (sigma i).toNat)
      odd perm hindex exponent

/-- Reducing the zero error matrix gives the zero ciphertext matrix. -/
theorem zero_error {q rows columns : Nat} :
    reduceMatrix q n rows columns (0 : ErrorMatrix n rows columns) = 0 := by
  funext row column
  simp [reduceMatrix]

/-- The branch automorphisms of the `state` LUT. -/
abbrev stateSigmas : List Int := [1, 3, 5, 15]

/-- Every branch automorphism of the `state` LUT is odd. -/
theorem stateOdd : ∀ branch : Fin 4, Odd (sigmaOf stateSigmas branch).toNat := by
  decide

/-- The actual normalized table coefficients encrypted for the `state` LUT. -/
noncomputable def stateCoefficients (branch : Fin 4) :
    ExactPoly W_Stage_preprocess_0_0.modulus_0 8 :=
  exponentCoefficient (q := W_Stage_preprocess_0_0.modulus_0) (sigmaOf stateSigmas branch) 2 2
    1356938545745524308187737599098762797340022013559604731140326430414605958745 (fun index : Fin 4
    ↦ MxxIR.selectInt (Int.ofNat index) [1, 0, 1, 0])

set_option maxHeartbeats 16000000 in
/-- Preprocess step 0's `state` family discharges `hswitch` with `Δ' = 1` and `BB = 0`.
The graph output is `w_3_*`; the coefficient is the encrypted table sum, not a hypothesis. -/
theorem preprocess0_state_switching {backend : BackendContext} {hashModel : HashModel}
    {params : W_Stage_preprocess_0_0.Params}
    (ctx : W_Stage_preprocess_0_0.generatedRoot.Ctx backend hashModel params) {outputs : _}
    (F : W_Stage_preprocess_0_0.generatedRoot.Facts ctx outputs)
    (hq : 1 < W_Stage_preprocess_0_0.modulus_0) {sp : ErrorPoly 8}
    (hs : W_Stage_preprocess_0_0.generatedRoot.w_1_0 ctx = fun _ _ ↦
      reducePoly W_Stage_preprocess_0_0.modulus_0 8 sp)
    {gadget : ExactMatrix W_Stage_preprocess_0_0.modulus_0 8 1 25}
    (hgadget : gadgetMatrixRuns backend 4096 25 gadget) :
    Switching sp (sigmaOf stateSigmas) stateOdd stateCoefficients
      (fam4 (W_Stage_preprocess_0_0.generatedRoot.w_3_0 ctx)
        (W_Stage_preprocess_0_0.generatedRoot.w_3_1 ctx)
        (W_Stage_preprocess_0_0.generatedRoot.w_3_2 ctx)
        (W_Stage_preprocess_0_0.generatedRoot.w_3_3 ctx)) gadget := by
  intro column branch
  have hone : W_Stage_preprocess_0_0.generatedRoot.w_2_0 ctx =
      fun _ _ ↦ reducePoly W_Stage_preprocess_0_0.modulus_0 8 1 := by
    funext row column
    fin_cases row
    fin_cases column
    simp [W_Stage_preprocess_0_0.generatedRoot.w_2_0]
  have hbranch := LutKeyTableFlat.rel_index
    (W_Stage_preprocess_0_0.parallel_generatedRoot_3.sem (F.c0.h3_3 branch))
    (fun index value hindex ↦ W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_3_12.sem
      hindex)
    (fun table index cur nxt hstep ↦
      W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_3_13.sem hstep)
    hq (by decide) hs hone
  obtain ⟨_, _, source, target, hsource, hrebase, hodd, hpay⟩ := hbranch
  have hsourceEq := gadgetMatrixRuns_unique hsource hgadget
  subst source
  have htargetEq := centeredRebaseRuns_unique hrebase
    (centeredRebaseRuns_self hq (by decide) gadget)
  subst target
  refine ⟨0, fun _ _ _ ↦ by simp, ?_⟩
  rw [zero_error, add_zero]
  fin_cases column
  · simpa [stateCoefficients, stateSigmas, exponentCoefficient,
      fam4, sigmaOf, reduceMatrix] using hpay 0
  · simpa [stateCoefficients, stateSigmas, exponentCoefficient,
      fam4, sigmaOf, reduceMatrix] using hpay 1

/-- The branch automorphisms of the `smudge` LUT. -/
abbrev smudgeSigmas : List Int := [1, 3, 5, 7, 9, 11, 13, 15]

/-- Every branch automorphism of the `smudge` LUT is odd. -/
theorem smudgeOdd : ∀ branch : Fin 8, Odd (sigmaOf smudgeSigmas branch).toNat := by
  decide

/-- The actual normalized table coefficients encrypted for the `smudge` LUT. -/
noncomputable def smudgeCoefficients (branch : Fin 8) :
    ExactPoly W_Stage_preprocess_0_0.modulus_0 8 :=
  coefficientCoefficient (q := W_Stage_preprocess_0_0.modulus_0) (sigmaOf smudgeSigmas branch) 1
    1583094970036445026219027198948556596896692349152872186330380835483706951869 (fun index : Fin 8
    ↦ MxxIR.selectInt (Int.ofNat index) [1, 1, 3, 3, 5, 5, 7, 7])

set_option maxHeartbeats 16000000 in
/-- Preprocess step 0's `smudge` family discharges `hswitch` with `Δ' = 1` and `BB = 0`.
The graph output is `w_4_*`; the coefficient is the encrypted table sum, not a hypothesis. -/
theorem preprocess0_smudge_switching {backend : BackendContext} {hashModel : HashModel}
    {params : W_Stage_preprocess_0_0.Params}
    (ctx : W_Stage_preprocess_0_0.generatedRoot.Ctx backend hashModel params) {outputs : _}
    (F : W_Stage_preprocess_0_0.generatedRoot.Facts ctx outputs)
    (hq : 1 < W_Stage_preprocess_0_0.modulus_0) {sp : ErrorPoly 8}
    (hs : W_Stage_preprocess_0_0.generatedRoot.w_1_0 ctx = fun _ _ ↦
      reducePoly W_Stage_preprocess_0_0.modulus_0 8 sp)
    {gadget : ExactMatrix W_Stage_preprocess_0_0.modulus_0 8 1 25}
    (hgadget : gadgetMatrixRuns backend 4096 25 gadget) :
    Switching sp (sigmaOf smudgeSigmas) smudgeOdd smudgeCoefficients
      (fam4 (W_Stage_preprocess_0_0.generatedRoot.w_4_0 ctx)
        (W_Stage_preprocess_0_0.generatedRoot.w_4_1 ctx)
        (W_Stage_preprocess_0_0.generatedRoot.w_4_2 ctx)
        (W_Stage_preprocess_0_0.generatedRoot.w_4_3 ctx)) gadget := by
  intro column branch
  have hone : W_Stage_preprocess_0_0.generatedRoot.w_2_0 ctx =
      fun _ _ ↦ reducePoly W_Stage_preprocess_0_0.modulus_0 8 1 := by
    funext row column
    fin_cases row
    fin_cases column
    simp [W_Stage_preprocess_0_0.generatedRoot.w_2_0]
  have hbranch := LutKeyTableFlat.rel_1_index
    (W_Stage_preprocess_0_0.parallel_generatedRoot_4.sem (F.c0.h4_3 branch))
    (fun index value hindex ↦ W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_4_10.sem
      hindex)
    (fun table index cur nxt hstep ↦
      W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_4_13.sem hstep)
    hq (by decide) hs hone
  obtain ⟨_, _, target, htarget, hodd, hpay⟩ := hbranch
  have htargetEq := gadgetMatrixRuns_unique htarget hgadget
  subst target
  refine ⟨0, fun _ _ _ ↦ by simp, ?_⟩
  rw [zero_error, add_zero]
  fin_cases column
  · simpa [smudgeCoefficients, smudgeSigmas, coefficientCoefficient,
      fam4, sigmaOf, reduceMatrix] using hpay 0
  · simpa [smudgeCoefficients, smudgeSigmas, coefficientCoefficient,
      fam4, sigmaOf, reduceMatrix] using hpay 1

/-- The branch automorphisms of the `sigmaReadout` LUT. -/
abbrev sigmaReadoutSigmas : List Int := [1, 3, 5, 7, 9, 11, 13, 15]

/-- Every branch automorphism of the `sigmaReadout` LUT is odd. -/
theorem sigmaReadoutOdd : ∀ branch : Fin 8, Odd (sigmaOf sigmaReadoutSigmas branch).toNat := by
  decide

/-- The actual normalized table coefficients encrypted for the `sigmaReadout` LUT. -/
noncomputable def sigmaReadoutCoefficients (branch : Fin 8) :
    ExactPoly W_Stage_preprocess_0_0.modulus_0 8 :=
  exponentCoefficient (q := W_Stage_preprocess_0_0.modulus_0) (sigmaOf sigmaReadoutSigmas branch) 1
    1 1583094970036445026219027198948556596896692349152872186330380835483706951869 (fun index : Fin
    8 ↦ MxxIR.selectInt (Int.ofNat index) [1, 1, 3, 3, 5, 5, 7, 7])

set_option maxHeartbeats 16000000 in
/-- Preprocess step 0's `sigmaReadout` family discharges `hswitch` with `Δ' = 1` and `BB = 0`.
The graph output is `w_5_*`; the coefficient is the encrypted table sum, not a hypothesis. -/
theorem preprocess0_sigmaReadout_switching {backend : BackendContext} {hashModel : HashModel}
    {params : W_Stage_preprocess_0_0.Params}
    (ctx : W_Stage_preprocess_0_0.generatedRoot.Ctx backend hashModel params) {outputs : _}
    (F : W_Stage_preprocess_0_0.generatedRoot.Facts ctx outputs)
    (hq : 1 < W_Stage_preprocess_0_0.modulus_0) {sp : ErrorPoly 8}
    (hs : W_Stage_preprocess_0_0.generatedRoot.w_1_0 ctx = fun _ _ ↦
      reducePoly W_Stage_preprocess_0_0.modulus_0 8 sp)
    {gadget : ExactMatrix W_Stage_preprocess_0_0.modulus_0 8 1 25}
    (hgadget : gadgetMatrixRuns backend 4096 25 gadget) :
    Switching sp (sigmaOf sigmaReadoutSigmas) sigmaReadoutOdd sigmaReadoutCoefficients
      (fam4 (W_Stage_preprocess_0_0.generatedRoot.w_5_0 ctx)
        (W_Stage_preprocess_0_0.generatedRoot.w_5_1 ctx)
        (W_Stage_preprocess_0_0.generatedRoot.w_5_2 ctx)
        (W_Stage_preprocess_0_0.generatedRoot.w_5_3 ctx)) gadget := by
  intro column branch
  have hone : W_Stage_preprocess_0_0.generatedRoot.w_2_0 ctx =
      fun _ _ ↦ reducePoly W_Stage_preprocess_0_0.modulus_0 8 1 := by
    funext row column
    fin_cases row
    fin_cases column
    simp [W_Stage_preprocess_0_0.generatedRoot.w_2_0]
  have hbranch := LutKeyTableFlat.rel_index
    (W_Stage_preprocess_0_0.parallel_generatedRoot_5.sem (F.c0.h5_3 branch))
    (fun index value hindex ↦ W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_5_12.sem
      hindex)
    (fun table index cur nxt hstep ↦
      W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_5_13.sem hstep)
    hq (by decide) hs hone
  obtain ⟨_, _, source, target, hsource, hrebase, hodd, hpay⟩ := hbranch
  have hsourceEq := gadgetMatrixRuns_unique hsource hgadget
  subst source
  have htargetEq := centeredRebaseRuns_unique hrebase
    (centeredRebaseRuns_self hq (by decide) gadget)
  subst target
  refine ⟨0, fun _ _ _ ↦ by simp, ?_⟩
  rw [zero_error, add_zero]
  fin_cases column
  · simpa [sigmaReadoutCoefficients, sigmaReadoutSigmas, exponentCoefficient,
      fam4, sigmaOf, reduceMatrix] using hpay 0
  · simpa [sigmaReadoutCoefficients, sigmaReadoutSigmas, exponentCoefficient,
      fam4, sigmaOf, reduceMatrix] using hpay 1

/-- The branch automorphisms of the `readout` LUT. -/
abbrev readoutSigmas : List Int := [1, 3, 5, 7, 9, 11, 13, 15]

/-- Every branch automorphism of the `readout` LUT is odd. -/
theorem readoutOdd : ∀ branch : Fin 8, Odd (sigmaOf readoutSigmas branch).toNat := by
  decide

/-- The actual normalized table coefficients encrypted for the `readout` LUT. -/
noncomputable def readoutCoefficients (branch : Fin 8) :
    ExactPoly W_Stage_preprocess_0_0.modulus_0 8 :=
  exponentCoefficient (q := W_Stage_preprocess_0_0.modulus_0) (sigmaOf readoutSigmas branch) 1 1
    1583094970036445026219027198948556596896692349152872186330380835483706951869 (fun index : Fin 8
    ↦ MxxIR.selectInt (Int.ofNat index) [1, 1, 3, 3, 5, 5, 7, 7])

set_option maxHeartbeats 16000000 in
/-- Preprocess step 0's `readout` family discharges `hswitch` with `Δ' = 1` and `BB = 0`.
The graph output is `w_6_*`; the coefficient is the encrypted table sum, not a hypothesis. -/
theorem preprocess0_readout_switching {backend : BackendContext} {hashModel : HashModel}
    {params : W_Stage_preprocess_0_0.Params}
    (ctx : W_Stage_preprocess_0_0.generatedRoot.Ctx backend hashModel params) {outputs : _}
    (F : W_Stage_preprocess_0_0.generatedRoot.Facts ctx outputs)
    (hq : 1 < W_Stage_preprocess_0_0.modulus_0) {sp : ErrorPoly 8}
    (hs : W_Stage_preprocess_0_0.generatedRoot.w_1_0 ctx = fun _ _ ↦
      reducePoly W_Stage_preprocess_0_0.modulus_0 8 sp)
    {gadget : ExactMatrix W_Stage_preprocess_0_0.modulus_0 8 1 25}
    (hgadget : gadgetMatrixRuns backend 4096 25 gadget) :
    Switching sp (sigmaOf readoutSigmas) readoutOdd readoutCoefficients
      (fam4 (W_Stage_preprocess_0_0.generatedRoot.w_6_0 ctx)
        (W_Stage_preprocess_0_0.generatedRoot.w_6_1 ctx)
        (W_Stage_preprocess_0_0.generatedRoot.w_6_2 ctx)
        (W_Stage_preprocess_0_0.generatedRoot.w_6_3 ctx)) gadget := by
  intro column branch
  have hone : W_Stage_preprocess_0_0.generatedRoot.w_2_0 ctx =
      fun _ _ ↦ reducePoly W_Stage_preprocess_0_0.modulus_0 8 1 := by
    funext row column
    fin_cases row
    fin_cases column
    simp [W_Stage_preprocess_0_0.generatedRoot.w_2_0]
  have hbranch := LutKeyTableFlat.rel_index
    (W_Stage_preprocess_0_0.parallel_generatedRoot_6.sem (F.c0.h6_3 branch))
    (fun index value hindex ↦ W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_6_12.sem
      hindex)
    (fun table index cur nxt hstep ↦
      W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_6_13.sem hstep)
    hq (by decide) hs hone
  obtain ⟨_, _, source, target, hsource, hrebase, hodd, hpay⟩ := hbranch
  have hsourceEq := gadgetMatrixRuns_unique hsource hgadget
  subst source
  have htargetEq := centeredRebaseRuns_unique hrebase
    (centeredRebaseRuns_self hq (by decide) gadget)
  subst target
  refine ⟨0, fun _ _ _ ↦ by simp, ?_⟩
  rw [zero_error, add_zero]
  fin_cases column
  · simpa [readoutCoefficients, readoutSigmas, exponentCoefficient,
      fam4, sigmaOf, reduceMatrix] using hpay 0
  · simpa [readoutCoefficients, readoutSigmas, exponentCoefficient,
      fam4, sigmaOf, reduceMatrix] using hpay 1

/-- The branch automorphisms of the `aut15` LUT. -/
abbrev aut15Sigmas : List Int := [15]

/-- Every branch automorphism of the `aut15` LUT is odd. -/
theorem aut15Odd : ∀ branch : Fin 1, Odd (sigmaOf aut15Sigmas branch).toNat := by
  decide

/-- The actual normalized table coefficients encrypted for the `aut15` LUT. -/
noncomputable def aut15Coefficients (branch : Fin 1) :
    ExactPoly W_Stage_preprocess_0_0.modulus_0 8 :=
  coefficientCoefficient (q := W_Stage_preprocess_0_0.modulus_0) (sigmaOf aut15Sigmas branch) 1 1
    (fun index : Fin 1 ↦ MxxIR.selectInt (Int.ofNat index) [1])

set_option maxHeartbeats 16000000 in
/-- Preprocess step 0's `aut15` family discharges `hswitch` with `Δ' = 1` and `BB = 0`.
The graph output is `w_7_*`; the coefficient is the encrypted table sum, not a hypothesis. -/
theorem preprocess0_aut15_switching {backend : BackendContext} {hashModel : HashModel}
    {params : W_Stage_preprocess_0_0.Params}
    (ctx : W_Stage_preprocess_0_0.generatedRoot.Ctx backend hashModel params) {outputs : _}
    (F : W_Stage_preprocess_0_0.generatedRoot.Facts ctx outputs)
    (hq : 1 < W_Stage_preprocess_0_0.modulus_0) {sp : ErrorPoly 8}
    (hs : W_Stage_preprocess_0_0.generatedRoot.w_1_0 ctx = fun _ _ ↦
      reducePoly W_Stage_preprocess_0_0.modulus_0 8 sp)
    {gadget : ExactMatrix W_Stage_preprocess_0_0.modulus_0 8 1 25}
    (hgadget : gadgetMatrixRuns backend 4096 25 gadget) :
    Switching sp (sigmaOf aut15Sigmas) aut15Odd aut15Coefficients
      (fam4 (W_Stage_preprocess_0_0.generatedRoot.w_7_0 ctx)
        (W_Stage_preprocess_0_0.generatedRoot.w_7_1 ctx)
        (W_Stage_preprocess_0_0.generatedRoot.w_7_2 ctx)
        (W_Stage_preprocess_0_0.generatedRoot.w_7_3 ctx)) gadget := by
  intro column branch
  have hone : W_Stage_preprocess_0_0.generatedRoot.w_2_0 ctx =
      fun _ _ ↦ reducePoly W_Stage_preprocess_0_0.modulus_0 8 1 := by
    funext row column
    fin_cases row
    fin_cases column
    simp [W_Stage_preprocess_0_0.generatedRoot.w_2_0]
  have hbranch := LutKeyTableFlat.rel_1_index
    (W_Stage_preprocess_0_0.parallel_generatedRoot_7.sem (F.c0.h7_3 branch))
    (fun index value hindex ↦ W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_7_10.sem
      hindex)
    (fun table index cur nxt hstep ↦
      W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_7_13.sem hstep)
    hq (by decide) hs hone
  obtain ⟨_, _, target, htarget, hodd, hpay⟩ := hbranch
  have htargetEq := gadgetMatrixRuns_unique htarget hgadget
  subst target
  refine ⟨0, fun _ _ _ ↦ by simp, ?_⟩
  rw [zero_error, add_zero]
  fin_cases column
  · simpa [aut15Coefficients, aut15Sigmas, coefficientCoefficient,
      fam4, sigmaOf, reduceMatrix] using hpay 0
  · simpa [aut15Coefficients, aut15Sigmas, coefficientCoefficient,
      fam4, sigmaOf, reduceMatrix] using hpay 1

/-- The branch automorphisms of the `sigmaSampler` LUT. -/
abbrev sigmaSamplerSigmas : List Int := [1, 3, 5, 7, 9, 11, 13, 15]

/-- Every branch automorphism of the `sigmaSampler` LUT is odd. -/
theorem sigmaSamplerOdd : ∀ branch : Fin 8, Odd (sigmaOf sigmaSamplerSigmas branch).toNat := by
  decide

/-- The actual normalized table coefficients encrypted for the `sigmaSampler` LUT. -/
noncomputable def sigmaSamplerCoefficients (branch : Fin 8) :
    ExactPoly W_Stage_preprocess_0_0.modulus_0 8 :=
  coefficientCoefficient (q := W_Stage_preprocess_0_0.modulus_0) (sigmaOf sigmaSamplerSigmas branch)
    1 1583094970036445026219027198948556596896692349152872186330380835483706951869 (fun index : Fin
    8 ↦ MxxIR.selectInt (Int.ofNat index) [0, 0, 0, 0, 0, 1, 0, 1])

set_option maxHeartbeats 16000000 in
/-- Preprocess step 0's `sigmaSampler` family discharges `hswitch` with `Δ' = 1` and `BB = 0`.
The graph output is `w_8_*`; the coefficient is the encrypted table sum, not a hypothesis. -/
theorem preprocess0_sigmaSampler_switching {backend : BackendContext} {hashModel : HashModel}
    {params : W_Stage_preprocess_0_0.Params}
    (ctx : W_Stage_preprocess_0_0.generatedRoot.Ctx backend hashModel params) {outputs : _}
    (F : W_Stage_preprocess_0_0.generatedRoot.Facts ctx outputs)
    (hq : 1 < W_Stage_preprocess_0_0.modulus_0) {sp : ErrorPoly 8}
    (hs : W_Stage_preprocess_0_0.generatedRoot.w_1_0 ctx = fun _ _ ↦
      reducePoly W_Stage_preprocess_0_0.modulus_0 8 sp)
    {gadget : ExactMatrix W_Stage_preprocess_0_0.modulus_0 8 1 25}
    (hgadget : gadgetMatrixRuns backend 4096 25 gadget) :
    Switching sp (sigmaOf sigmaSamplerSigmas) sigmaSamplerOdd sigmaSamplerCoefficients
      (fam4 (W_Stage_preprocess_0_0.generatedRoot.w_8_0 ctx)
        (W_Stage_preprocess_0_0.generatedRoot.w_8_1 ctx)
        (W_Stage_preprocess_0_0.generatedRoot.w_8_2 ctx)
        (W_Stage_preprocess_0_0.generatedRoot.w_8_3 ctx)) gadget := by
  intro column branch
  have hone : W_Stage_preprocess_0_0.generatedRoot.w_2_0 ctx =
      fun _ _ ↦ reducePoly W_Stage_preprocess_0_0.modulus_0 8 1 := by
    funext row column
    fin_cases row
    fin_cases column
    simp [W_Stage_preprocess_0_0.generatedRoot.w_2_0]
  have hbranch := LutKeyTableFlat.rel_1_index
    (W_Stage_preprocess_0_0.parallel_generatedRoot_8.sem (F.c0.h8_3 branch))
    (fun index value hindex ↦ W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_8_10.sem
      hindex)
    (fun table index cur nxt hstep ↦
      W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_8_13.sem hstep)
    hq (by decide) hs hone
  obtain ⟨_, _, target, htarget, hodd, hpay⟩ := hbranch
  have htargetEq := gadgetMatrixRuns_unique htarget hgadget
  subst target
  refine ⟨0, fun _ _ _ ↦ by simp, ?_⟩
  rw [zero_error, add_zero]
  fin_cases column
  · simpa [sigmaSamplerCoefficients, sigmaSamplerSigmas, coefficientCoefficient,
      fam4, sigmaOf, reduceMatrix] using hpay 0
  · simpa [sigmaSamplerCoefficients, sigmaSamplerSigmas, coefficientCoefficient,
      fam4, sigmaOf, reduceMatrix] using hpay 1

/-- The branch automorphisms of the `sampler` LUT. -/
abbrev samplerSigmas : List Int := [1, 3, 5, 7, 9, 11, 13, 15]

/-- Every branch automorphism of the `sampler` LUT is odd. -/
theorem samplerOdd : ∀ branch : Fin 8, Odd (sigmaOf samplerSigmas branch).toNat := by
  decide

/-- The actual normalized table coefficients encrypted for the `sampler` LUT. -/
noncomputable def samplerCoefficients (branch : Fin 8) :
    ExactPoly W_Stage_preprocess_0_0.modulus_0 8 :=
  coefficientCoefficient (q := W_Stage_preprocess_0_0.modulus_0) (sigmaOf samplerSigmas branch) 1
    1583094970036445026219027198948556596896692349152872186330380835483706951869 (fun index : Fin 8
    ↦ MxxIR.selectInt (Int.ofNat index) [0, 1, 0, 1, 0, 1, 0, 1])

set_option maxHeartbeats 16000000 in
/-- Preprocess step 0's `sampler` family discharges `hswitch` with `Δ' = 1` and `BB = 0`.
The graph output is `w_9_*`; the coefficient is the encrypted table sum, not a hypothesis. -/
theorem preprocess0_sampler_switching {backend : BackendContext} {hashModel : HashModel}
    {params : W_Stage_preprocess_0_0.Params}
    (ctx : W_Stage_preprocess_0_0.generatedRoot.Ctx backend hashModel params) {outputs : _}
    (F : W_Stage_preprocess_0_0.generatedRoot.Facts ctx outputs)
    (hq : 1 < W_Stage_preprocess_0_0.modulus_0) {sp : ErrorPoly 8}
    (hs : W_Stage_preprocess_0_0.generatedRoot.w_1_0 ctx = fun _ _ ↦
      reducePoly W_Stage_preprocess_0_0.modulus_0 8 sp)
    {gadget : ExactMatrix W_Stage_preprocess_0_0.modulus_0 8 1 25}
    (hgadget : gadgetMatrixRuns backend 4096 25 gadget) :
    Switching sp (sigmaOf samplerSigmas) samplerOdd samplerCoefficients
      (fam4 (W_Stage_preprocess_0_0.generatedRoot.w_9_0 ctx)
        (W_Stage_preprocess_0_0.generatedRoot.w_9_1 ctx)
        (W_Stage_preprocess_0_0.generatedRoot.w_9_2 ctx)
        (W_Stage_preprocess_0_0.generatedRoot.w_9_3 ctx)) gadget := by
  intro column branch
  have hone : W_Stage_preprocess_0_0.generatedRoot.w_2_0 ctx =
      fun _ _ ↦ reducePoly W_Stage_preprocess_0_0.modulus_0 8 1 := by
    funext row column
    fin_cases row
    fin_cases column
    simp [W_Stage_preprocess_0_0.generatedRoot.w_2_0]
  have hbranch := LutKeyTableFlat.rel_1_index
    (W_Stage_preprocess_0_0.parallel_generatedRoot_9.sem (F.c0.h9_3 branch))
    (fun index value hindex ↦ W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_9_10.sem
      hindex)
    (fun table index cur nxt hstep ↦
      W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_9_13.sem hstep)
    hq (by decide) hs hone
  obtain ⟨_, _, target, htarget, hodd, hpay⟩ := hbranch
  have htargetEq := gadgetMatrixRuns_unique htarget hgadget
  subst target
  refine ⟨0, fun _ _ _ ↦ by simp, ?_⟩
  rw [zero_error, add_zero]
  fin_cases column
  · simpa [samplerCoefficients, samplerSigmas, coefficientCoefficient,
      fam4, sigmaOf, reduceMatrix] using hpay 0
  · simpa [samplerCoefficients, samplerSigmas, coefficientCoefficient,
      fam4, sigmaOf, reduceMatrix] using hpay 1

/-- The branch automorphisms of the `aut3` LUT. -/
abbrev aut3Sigmas : List Int := [3]

/-- Every branch automorphism of the `aut3` LUT is odd. -/
theorem aut3Odd : ∀ branch : Fin 1, Odd (sigmaOf aut3Sigmas branch).toNat := by
  decide

/-- The actual normalized table coefficients encrypted for the `aut3` LUT. -/
noncomputable def aut3Coefficients (branch : Fin 1) :
    ExactPoly W_Stage_preprocess_0_0.modulus_0 8 :=
  coefficientCoefficient (q := W_Stage_preprocess_0_0.modulus_0) (sigmaOf aut3Sigmas branch) 1 1
    (fun index : Fin 1 ↦ MxxIR.selectInt (Int.ofNat index) [1])

set_option maxHeartbeats 16000000 in
/-- Preprocess step 0's `aut3` family discharges `hswitch` with `Δ' = 1` and `BB = 0`.
The graph output is `w_10_*`; the coefficient is the encrypted table sum, not a hypothesis. -/
theorem preprocess0_aut3_switching {backend : BackendContext} {hashModel : HashModel}
    {params : W_Stage_preprocess_0_0.Params}
    (ctx : W_Stage_preprocess_0_0.generatedRoot.Ctx backend hashModel params) {outputs : _}
    (F : W_Stage_preprocess_0_0.generatedRoot.Facts ctx outputs)
    (hq : 1 < W_Stage_preprocess_0_0.modulus_0) {sp : ErrorPoly 8}
    (hs : W_Stage_preprocess_0_0.generatedRoot.w_1_0 ctx = fun _ _ ↦
      reducePoly W_Stage_preprocess_0_0.modulus_0 8 sp)
    {gadget : ExactMatrix W_Stage_preprocess_0_0.modulus_0 8 1 25}
    (hgadget : gadgetMatrixRuns backend 4096 25 gadget) :
    Switching sp (sigmaOf aut3Sigmas) aut3Odd aut3Coefficients
      (fam4 (W_Stage_preprocess_0_0.generatedRoot.w_10_0 ctx)
        (W_Stage_preprocess_0_0.generatedRoot.w_10_1 ctx)
        (W_Stage_preprocess_0_0.generatedRoot.w_10_2 ctx)
        (W_Stage_preprocess_0_0.generatedRoot.w_10_3 ctx)) gadget := by
  intro column branch
  have hone : W_Stage_preprocess_0_0.generatedRoot.w_2_0 ctx =
      fun _ _ ↦ reducePoly W_Stage_preprocess_0_0.modulus_0 8 1 := by
    funext row column
    fin_cases row
    fin_cases column
    simp [W_Stage_preprocess_0_0.generatedRoot.w_2_0]
  have hbranch := LutKeyTableFlat.rel_1_index
    (W_Stage_preprocess_0_0.parallel_generatedRoot_10.sem (F.c0.h10_3 branch))
    (fun index value hindex ↦ W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_10_10.sem
      hindex)
    (fun table index cur nxt hstep ↦
      W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_10_13.sem hstep)
    hq (by decide) hs hone
  obtain ⟨_, _, target, htarget, hodd, hpay⟩ := hbranch
  have htargetEq := gadgetMatrixRuns_unique htarget hgadget
  subst target
  refine ⟨0, fun _ _ _ ↦ by simp, ?_⟩
  rw [zero_error, add_zero]
  fin_cases column
  · simpa [aut3Coefficients, aut3Sigmas, coefficientCoefficient,
      fam4, sigmaOf, reduceMatrix] using hpay 0
  · simpa [aut3Coefficients, aut3Sigmas, coefficientCoefficient,
      fam4, sigmaOf, reduceMatrix] using hpay 1

/-- The branch automorphisms of the `aut5` LUT. -/
abbrev aut5Sigmas : List Int := [5]

/-- Every branch automorphism of the `aut5` LUT is odd. -/
theorem aut5Odd : ∀ branch : Fin 1, Odd (sigmaOf aut5Sigmas branch).toNat := by
  decide

/-- The actual normalized table coefficients encrypted for the `aut5` LUT. -/
noncomputable def aut5Coefficients (branch : Fin 1) :
    ExactPoly W_Stage_preprocess_0_0.modulus_0 8 :=
  coefficientCoefficient (q := W_Stage_preprocess_0_0.modulus_0) (sigmaOf aut5Sigmas branch) 1 1
    (fun index : Fin 1 ↦ MxxIR.selectInt (Int.ofNat index) [1])

set_option maxHeartbeats 16000000 in
/-- Preprocess step 0's `aut5` family discharges `hswitch` with `Δ' = 1` and `BB = 0`.
The graph output is `w_11_*`; the coefficient is the encrypted table sum, not a hypothesis. -/
theorem preprocess0_aut5_switching {backend : BackendContext} {hashModel : HashModel}
    {params : W_Stage_preprocess_0_0.Params}
    (ctx : W_Stage_preprocess_0_0.generatedRoot.Ctx backend hashModel params) {outputs : _}
    (F : W_Stage_preprocess_0_0.generatedRoot.Facts ctx outputs)
    (hq : 1 < W_Stage_preprocess_0_0.modulus_0) {sp : ErrorPoly 8}
    (hs : W_Stage_preprocess_0_0.generatedRoot.w_1_0 ctx = fun _ _ ↦
      reducePoly W_Stage_preprocess_0_0.modulus_0 8 sp)
    {gadget : ExactMatrix W_Stage_preprocess_0_0.modulus_0 8 1 25}
    (hgadget : gadgetMatrixRuns backend 4096 25 gadget) :
    Switching sp (sigmaOf aut5Sigmas) aut5Odd aut5Coefficients
      (fam4 (W_Stage_preprocess_0_0.generatedRoot.w_11_0 ctx)
        (W_Stage_preprocess_0_0.generatedRoot.w_11_1 ctx)
        (W_Stage_preprocess_0_0.generatedRoot.w_11_2 ctx)
        (W_Stage_preprocess_0_0.generatedRoot.w_11_3 ctx)) gadget := by
  intro column branch
  have hone : W_Stage_preprocess_0_0.generatedRoot.w_2_0 ctx =
      fun _ _ ↦ reducePoly W_Stage_preprocess_0_0.modulus_0 8 1 := by
    funext row column
    fin_cases row
    fin_cases column
    simp [W_Stage_preprocess_0_0.generatedRoot.w_2_0]
  have hbranch := LutKeyTableFlat.rel_1_index
    (W_Stage_preprocess_0_0.parallel_generatedRoot_11.sem (F.c0.h11_3 branch))
    (fun index value hindex ↦ W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_11_10.sem
      hindex)
    (fun table index cur nxt hstep ↦
      W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_11_13.sem hstep)
    hq (by decide) hs hone
  obtain ⟨_, _, target, htarget, hodd, hpay⟩ := hbranch
  have htargetEq := gadgetMatrixRuns_unique htarget hgadget
  subst target
  refine ⟨0, fun _ _ _ ↦ by simp, ?_⟩
  rw [zero_error, add_zero]
  fin_cases column
  · simpa [aut5Coefficients, aut5Sigmas, coefficientCoefficient,
      fam4, sigmaOf, reduceMatrix] using hpay 0
  · simpa [aut5Coefficients, aut5Sigmas, coefficientCoefficient,
      fam4, sigmaOf, reduceMatrix] using hpay 1

/-- The branch automorphisms of the `aut7` LUT. -/
abbrev aut7Sigmas : List Int := [7]

/-- Every branch automorphism of the `aut7` LUT is odd. -/
theorem aut7Odd : ∀ branch : Fin 1, Odd (sigmaOf aut7Sigmas branch).toNat := by
  decide

/-- The actual normalized table coefficients encrypted for the `aut7` LUT. -/
noncomputable def aut7Coefficients (branch : Fin 1) :
    ExactPoly W_Stage_preprocess_0_0.modulus_0 8 :=
  coefficientCoefficient (q := W_Stage_preprocess_0_0.modulus_0) (sigmaOf aut7Sigmas branch) 1 1
    (fun index : Fin 1 ↦ MxxIR.selectInt (Int.ofNat index) [1])

set_option maxHeartbeats 16000000 in
/-- Preprocess step 0's `aut7` family discharges `hswitch` with `Δ' = 1` and `BB = 0`.
The graph output is `w_12_*`; the coefficient is the encrypted table sum, not a hypothesis. -/
theorem preprocess0_aut7_switching {backend : BackendContext} {hashModel : HashModel}
    {params : W_Stage_preprocess_0_0.Params}
    (ctx : W_Stage_preprocess_0_0.generatedRoot.Ctx backend hashModel params) {outputs : _}
    (F : W_Stage_preprocess_0_0.generatedRoot.Facts ctx outputs)
    (hq : 1 < W_Stage_preprocess_0_0.modulus_0) {sp : ErrorPoly 8}
    (hs : W_Stage_preprocess_0_0.generatedRoot.w_1_0 ctx = fun _ _ ↦
      reducePoly W_Stage_preprocess_0_0.modulus_0 8 sp)
    {gadget : ExactMatrix W_Stage_preprocess_0_0.modulus_0 8 1 25}
    (hgadget : gadgetMatrixRuns backend 4096 25 gadget) :
    Switching sp (sigmaOf aut7Sigmas) aut7Odd aut7Coefficients
      (fam4 (W_Stage_preprocess_0_0.generatedRoot.w_12_0 ctx)
        (W_Stage_preprocess_0_0.generatedRoot.w_12_1 ctx)
        (W_Stage_preprocess_0_0.generatedRoot.w_12_2 ctx)
        (W_Stage_preprocess_0_0.generatedRoot.w_12_3 ctx)) gadget := by
  intro column branch
  have hone : W_Stage_preprocess_0_0.generatedRoot.w_2_0 ctx =
      fun _ _ ↦ reducePoly W_Stage_preprocess_0_0.modulus_0 8 1 := by
    funext row column
    fin_cases row
    fin_cases column
    simp [W_Stage_preprocess_0_0.generatedRoot.w_2_0]
  have hbranch := LutKeyTableFlat.rel_1_index
    (W_Stage_preprocess_0_0.parallel_generatedRoot_12.sem (F.c0.h12_3 branch))
    (fun index value hindex ↦ W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_12_10.sem
      hindex)
    (fun table index cur nxt hstep ↦
      W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_12_13.sem hstep)
    hq (by decide) hs hone
  obtain ⟨_, _, target, htarget, hodd, hpay⟩ := hbranch
  have htargetEq := gadgetMatrixRuns_unique htarget hgadget
  subst target
  refine ⟨0, fun _ _ _ ↦ by simp, ?_⟩
  rw [zero_error, add_zero]
  fin_cases column
  · simpa [aut7Coefficients, aut7Sigmas, coefficientCoefficient,
      fam4, sigmaOf, reduceMatrix] using hpay 0
  · simpa [aut7Coefficients, aut7Sigmas, coefficientCoefficient,
      fam4, sigmaOf, reduceMatrix] using hpay 1

/-- The branch automorphisms of the `aut9` LUT. -/
abbrev aut9Sigmas : List Int := [9]

/-- Every branch automorphism of the `aut9` LUT is odd. -/
theorem aut9Odd : ∀ branch : Fin 1, Odd (sigmaOf aut9Sigmas branch).toNat := by
  decide

/-- The actual normalized table coefficients encrypted for the `aut9` LUT. -/
noncomputable def aut9Coefficients (branch : Fin 1) :
    ExactPoly W_Stage_preprocess_0_0.modulus_0 8 :=
  coefficientCoefficient (q := W_Stage_preprocess_0_0.modulus_0) (sigmaOf aut9Sigmas branch) 1 1
    (fun index : Fin 1 ↦ MxxIR.selectInt (Int.ofNat index) [1])

set_option maxHeartbeats 16000000 in
/-- Preprocess step 0's `aut9` family discharges `hswitch` with `Δ' = 1` and `BB = 0`.
The graph output is `w_13_*`; the coefficient is the encrypted table sum, not a hypothesis. -/
theorem preprocess0_aut9_switching {backend : BackendContext} {hashModel : HashModel}
    {params : W_Stage_preprocess_0_0.Params}
    (ctx : W_Stage_preprocess_0_0.generatedRoot.Ctx backend hashModel params) {outputs : _}
    (F : W_Stage_preprocess_0_0.generatedRoot.Facts ctx outputs)
    (hq : 1 < W_Stage_preprocess_0_0.modulus_0) {sp : ErrorPoly 8}
    (hs : W_Stage_preprocess_0_0.generatedRoot.w_1_0 ctx = fun _ _ ↦
      reducePoly W_Stage_preprocess_0_0.modulus_0 8 sp)
    {gadget : ExactMatrix W_Stage_preprocess_0_0.modulus_0 8 1 25}
    (hgadget : gadgetMatrixRuns backend 4096 25 gadget) :
    Switching sp (sigmaOf aut9Sigmas) aut9Odd aut9Coefficients
      (fam4 (W_Stage_preprocess_0_0.generatedRoot.w_13_0 ctx)
        (W_Stage_preprocess_0_0.generatedRoot.w_13_1 ctx)
        (W_Stage_preprocess_0_0.generatedRoot.w_13_2 ctx)
        (W_Stage_preprocess_0_0.generatedRoot.w_13_3 ctx)) gadget := by
  intro column branch
  have hone : W_Stage_preprocess_0_0.generatedRoot.w_2_0 ctx =
      fun _ _ ↦ reducePoly W_Stage_preprocess_0_0.modulus_0 8 1 := by
    funext row column
    fin_cases row
    fin_cases column
    simp [W_Stage_preprocess_0_0.generatedRoot.w_2_0]
  have hbranch := LutKeyTableFlat.rel_1_index
    (W_Stage_preprocess_0_0.parallel_generatedRoot_13.sem (F.c0.h13_3 branch))
    (fun index value hindex ↦ W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_13_10.sem
      hindex)
    (fun table index cur nxt hstep ↦
      W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_13_13.sem hstep)
    hq (by decide) hs hone
  obtain ⟨_, _, target, htarget, hodd, hpay⟩ := hbranch
  have htargetEq := gadgetMatrixRuns_unique htarget hgadget
  subst target
  refine ⟨0, fun _ _ _ ↦ by simp, ?_⟩
  rw [zero_error, add_zero]
  fin_cases column
  · simpa [aut9Coefficients, aut9Sigmas, coefficientCoefficient,
      fam4, sigmaOf, reduceMatrix] using hpay 0
  · simpa [aut9Coefficients, aut9Sigmas, coefficientCoefficient,
      fam4, sigmaOf, reduceMatrix] using hpay 1

/-- The branch automorphisms of the `aut11` LUT. -/
abbrev aut11Sigmas : List Int := [11]

/-- Every branch automorphism of the `aut11` LUT is odd. -/
theorem aut11Odd : ∀ branch : Fin 1, Odd (sigmaOf aut11Sigmas branch).toNat := by
  decide

/-- The actual normalized table coefficients encrypted for the `aut11` LUT. -/
noncomputable def aut11Coefficients (branch : Fin 1) :
    ExactPoly W_Stage_preprocess_0_0.modulus_0 8 :=
  coefficientCoefficient (q := W_Stage_preprocess_0_0.modulus_0) (sigmaOf aut11Sigmas branch) 1 1
    (fun index : Fin 1 ↦ MxxIR.selectInt (Int.ofNat index) [1])

set_option maxHeartbeats 16000000 in
/-- Preprocess step 0's `aut11` family discharges `hswitch` with `Δ' = 1` and `BB = 0`.
The graph output is `w_14_*`; the coefficient is the encrypted table sum, not a hypothesis. -/
theorem preprocess0_aut11_switching {backend : BackendContext} {hashModel : HashModel}
    {params : W_Stage_preprocess_0_0.Params}
    (ctx : W_Stage_preprocess_0_0.generatedRoot.Ctx backend hashModel params) {outputs : _}
    (F : W_Stage_preprocess_0_0.generatedRoot.Facts ctx outputs)
    (hq : 1 < W_Stage_preprocess_0_0.modulus_0) {sp : ErrorPoly 8}
    (hs : W_Stage_preprocess_0_0.generatedRoot.w_1_0 ctx = fun _ _ ↦
      reducePoly W_Stage_preprocess_0_0.modulus_0 8 sp)
    {gadget : ExactMatrix W_Stage_preprocess_0_0.modulus_0 8 1 25}
    (hgadget : gadgetMatrixRuns backend 4096 25 gadget) :
    Switching sp (sigmaOf aut11Sigmas) aut11Odd aut11Coefficients
      (fam4 (W_Stage_preprocess_0_0.generatedRoot.w_14_0 ctx)
        (W_Stage_preprocess_0_0.generatedRoot.w_14_1 ctx)
        (W_Stage_preprocess_0_0.generatedRoot.w_14_2 ctx)
        (W_Stage_preprocess_0_0.generatedRoot.w_14_3 ctx)) gadget := by
  intro column branch
  have hone : W_Stage_preprocess_0_0.generatedRoot.w_2_0 ctx =
      fun _ _ ↦ reducePoly W_Stage_preprocess_0_0.modulus_0 8 1 := by
    funext row column
    fin_cases row
    fin_cases column
    simp [W_Stage_preprocess_0_0.generatedRoot.w_2_0]
  have hbranch := LutKeyTableFlat.rel_1_index
    (W_Stage_preprocess_0_0.parallel_generatedRoot_14.sem (F.c0.h14_3 branch))
    (fun index value hindex ↦ W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_14_10.sem
      hindex)
    (fun table index cur nxt hstep ↦
      W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_14_13.sem hstep)
    hq (by decide) hs hone
  obtain ⟨_, _, target, htarget, hodd, hpay⟩ := hbranch
  have htargetEq := gadgetMatrixRuns_unique htarget hgadget
  subst target
  refine ⟨0, fun _ _ _ ↦ by simp, ?_⟩
  rw [zero_error, add_zero]
  fin_cases column
  · simpa [aut11Coefficients, aut11Sigmas, coefficientCoefficient,
      fam4, sigmaOf, reduceMatrix] using hpay 0
  · simpa [aut11Coefficients, aut11Sigmas, coefficientCoefficient,
      fam4, sigmaOf, reduceMatrix] using hpay 1

/-- The branch automorphisms of the `aut13` LUT. -/
abbrev aut13Sigmas : List Int := [13]

/-- Every branch automorphism of the `aut13` LUT is odd. -/
theorem aut13Odd : ∀ branch : Fin 1, Odd (sigmaOf aut13Sigmas branch).toNat := by
  decide

/-- The actual normalized table coefficients encrypted for the `aut13` LUT. -/
noncomputable def aut13Coefficients (branch : Fin 1) :
    ExactPoly W_Stage_preprocess_0_0.modulus_0 8 :=
  coefficientCoefficient (q := W_Stage_preprocess_0_0.modulus_0) (sigmaOf aut13Sigmas branch) 1 1
    (fun index : Fin 1 ↦ MxxIR.selectInt (Int.ofNat index) [1])

set_option maxHeartbeats 16000000 in
/-- Preprocess step 0's `aut13` family discharges `hswitch` with `Δ' = 1` and `BB = 0`.
The graph output is `w_15_*`; the coefficient is the encrypted table sum, not a hypothesis. -/
theorem preprocess0_aut13_switching {backend : BackendContext} {hashModel : HashModel}
    {params : W_Stage_preprocess_0_0.Params}
    (ctx : W_Stage_preprocess_0_0.generatedRoot.Ctx backend hashModel params) {outputs : _}
    (F : W_Stage_preprocess_0_0.generatedRoot.Facts ctx outputs)
    (hq : 1 < W_Stage_preprocess_0_0.modulus_0) {sp : ErrorPoly 8}
    (hs : W_Stage_preprocess_0_0.generatedRoot.w_1_0 ctx = fun _ _ ↦
      reducePoly W_Stage_preprocess_0_0.modulus_0 8 sp)
    {gadget : ExactMatrix W_Stage_preprocess_0_0.modulus_0 8 1 25}
    (hgadget : gadgetMatrixRuns backend 4096 25 gadget) :
    Switching sp (sigmaOf aut13Sigmas) aut13Odd aut13Coefficients
      (fam4 (W_Stage_preprocess_0_0.generatedRoot.w_15_0 ctx)
        (W_Stage_preprocess_0_0.generatedRoot.w_15_1 ctx)
        (W_Stage_preprocess_0_0.generatedRoot.w_15_2 ctx)
        (W_Stage_preprocess_0_0.generatedRoot.w_15_3 ctx)) gadget := by
  intro column branch
  have hone : W_Stage_preprocess_0_0.generatedRoot.w_2_0 ctx =
      fun _ _ ↦ reducePoly W_Stage_preprocess_0_0.modulus_0 8 1 := by
    funext row column
    fin_cases row
    fin_cases column
    simp [W_Stage_preprocess_0_0.generatedRoot.w_2_0]
  have hbranch := LutKeyTableFlat.rel_1_index
    (W_Stage_preprocess_0_0.parallel_generatedRoot_15.sem (F.c0.h15_3 branch))
    (fun index value hindex ↦ W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_15_10.sem
      hindex)
    (fun table index cur nxt hstep ↦
      W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_15_13.sem hstep)
    hq (by decide) hs hone
  obtain ⟨_, _, target, htarget, hodd, hpay⟩ := hbranch
  have htargetEq := gadgetMatrixRuns_unique htarget hgadget
  subst target
  refine ⟨0, fun _ _ _ ↦ by simp, ?_⟩
  rw [zero_error, add_zero]
  fin_cases column
  · simpa [aut13Coefficients, aut13Sigmas, coefficientCoefficient,
      fam4, sigmaOf, reduceMatrix] using hpay 0
  · simpa [aut13Coefficients, aut13Sigmas, coefficientCoefficient,
      fam4, sigmaOf, reduceMatrix] using hpay 1

set_option maxHeartbeats 16000000 in
/-- The state table differs in step 1: its entries are `0, 1, 1, 0`. -/
noncomputable def state1Coefficients (branch : Fin 4) :
    ExactPoly W_Stage_preprocess_1_1.modulus_0 8 :=
  exponentCoefficient (sigmaOf stateSigmas branch) 2 2
    1356938545745524308187737599098762797340022013559604731140326430414605958745
    (fun index : Fin 4 ↦ MxxIR.selectInt (Int.ofNat index) [0, 1, 1, 0])

set_option maxHeartbeats 16000000 in
/-- Preprocess step 1's `state` family discharges `hswitch` with `Δ' = 1` and `BB = 0`.
The graph output is `w_3_*`; the coefficient is the encrypted table sum, not a hypothesis. -/
theorem preprocess1_state_switching {backend : BackendContext} {hashModel : HashModel}
    {params : W_Stage_preprocess_1_1.Params}
    (ctx : W_Stage_preprocess_1_1.generatedRoot.Ctx backend hashModel params) {outputs : _}
    (F : W_Stage_preprocess_1_1.generatedRoot.Facts ctx outputs)
    (hq : 1 < W_Stage_preprocess_1_1.modulus_0) {sp : ErrorPoly 8}
    (hs : W_Stage_preprocess_1_1.generatedRoot.w_1_0 ctx = fun _ _ ↦
      reducePoly W_Stage_preprocess_1_1.modulus_0 8 sp)
    {gadget : ExactMatrix W_Stage_preprocess_1_1.modulus_0 8 1 25}
    (hgadget : gadgetMatrixRuns backend 4096 25 gadget) :
    Switching sp (sigmaOf stateSigmas) stateOdd state1Coefficients
      (fam4 (W_Stage_preprocess_1_1.generatedRoot.w_3_0 ctx)
        (W_Stage_preprocess_1_1.generatedRoot.w_3_1 ctx)
        (W_Stage_preprocess_1_1.generatedRoot.w_3_2 ctx)
        (W_Stage_preprocess_1_1.generatedRoot.w_3_3 ctx)) gadget := by
  intro column branch
  have hone : W_Stage_preprocess_1_1.generatedRoot.w_2_0 ctx =
      fun _ _ ↦ reducePoly W_Stage_preprocess_1_1.modulus_0 8 1 := by
    funext row column
    fin_cases row
    fin_cases column
    simp [W_Stage_preprocess_1_1.generatedRoot.w_2_0]
  have hbranch := LutKeyTableFlat.rel_index
    (W_Stage_preprocess_1_1.parallel_generatedRoot_3.sem (F.c0.h3_3 branch))
    (fun index value hindex ↦ W_Stage_preprocess_1_1.parallel_parallel_generatedRoot_3_12.sem
      hindex)
    (fun table index cur nxt hstep ↦
      W_Stage_preprocess_1_1.sequential_parallel_generatedRoot_3_13.sem hstep)
    hq (by decide) hs hone
  obtain ⟨_, _, source, target, hsource, hrebase, hodd, hpay⟩ := hbranch
  have hsourceEq := gadgetMatrixRuns_unique hsource hgadget
  subst source
  have htargetEq := centeredRebaseRuns_unique hrebase
    (centeredRebaseRuns_self hq (by decide) gadget)
  subst target
  refine ⟨0, fun _ _ _ ↦ by simp, ?_⟩
  rw [zero_error, add_zero]
  fin_cases column
  · simpa [state1Coefficients, stateSigmas, exponentCoefficient,
      fam4, sigmaOf, reduceMatrix] using hpay 0
  · simpa [state1Coefficients, stateSigmas, exponentCoefficient,
      fam4, sigmaOf, reduceMatrix] using hpay 1

set_option maxHeartbeats 16000000 in
/-- Preprocess step 1's `smudge` family discharges `hswitch` with `Δ' = 1` and `BB = 0`.
The graph output is `w_4_*`; the coefficient is the encrypted table sum, not a hypothesis. -/
theorem preprocess1_smudge_switching {backend : BackendContext} {hashModel : HashModel}
    {params : W_Stage_preprocess_1_1.Params}
    (ctx : W_Stage_preprocess_1_1.generatedRoot.Ctx backend hashModel params) {outputs : _}
    (F : W_Stage_preprocess_1_1.generatedRoot.Facts ctx outputs)
    (hq : 1 < W_Stage_preprocess_1_1.modulus_0) {sp : ErrorPoly 8}
    (hs : W_Stage_preprocess_1_1.generatedRoot.w_1_0 ctx = fun _ _ ↦
      reducePoly W_Stage_preprocess_1_1.modulus_0 8 sp)
    {gadget : ExactMatrix W_Stage_preprocess_1_1.modulus_0 8 1 25}
    (hgadget : gadgetMatrixRuns backend 4096 25 gadget) :
    Switching sp (sigmaOf smudgeSigmas) smudgeOdd smudgeCoefficients
      (fam4 (W_Stage_preprocess_1_1.generatedRoot.w_4_0 ctx)
        (W_Stage_preprocess_1_1.generatedRoot.w_4_1 ctx)
        (W_Stage_preprocess_1_1.generatedRoot.w_4_2 ctx)
        (W_Stage_preprocess_1_1.generatedRoot.w_4_3 ctx)) gadget := by
  intro column branch
  have hone : W_Stage_preprocess_1_1.generatedRoot.w_2_0 ctx =
      fun _ _ ↦ reducePoly W_Stage_preprocess_1_1.modulus_0 8 1 := by
    funext row column
    fin_cases row
    fin_cases column
    simp [W_Stage_preprocess_1_1.generatedRoot.w_2_0]
  have hbranch := LutKeyTableFlat.rel_1_index
    (W_Stage_preprocess_1_1.parallel_generatedRoot_4.sem (F.c0.h4_3 branch))
    (fun index value hindex ↦ W_Stage_preprocess_1_1.parallel_parallel_generatedRoot_4_10.sem
      hindex)
    (fun table index cur nxt hstep ↦
      W_Stage_preprocess_1_1.sequential_parallel_generatedRoot_4_13.sem hstep)
    hq (by decide) hs hone
  obtain ⟨_, _, target, htarget, hodd, hpay⟩ := hbranch
  have htargetEq := gadgetMatrixRuns_unique htarget hgadget
  subst target
  refine ⟨0, fun _ _ _ ↦ by simp, ?_⟩
  rw [zero_error, add_zero]
  fin_cases column
  · simpa [smudgeCoefficients, smudgeSigmas, coefficientCoefficient,
      fam4, sigmaOf, reduceMatrix] using hpay 0
  · simpa [smudgeCoefficients, smudgeSigmas, coefficientCoefficient,
      fam4, sigmaOf, reduceMatrix] using hpay 1
/-- The coefficient-table Walsh premise, proved from the table and its normalization. -/
theorem coefficient_walsh {q inverse stride log : Nat} (hq : 1 < q)
    (hperiod : n = stride * 2 ^ log) (hn : 0 < n) (table : Fin (2 ^ log) → Int)
    (hinverse : (inverse : ExactPoly q n) * ((2 ^ log : Nat) : ExactPoly q n) = 1)
    (sigma : Fin (2 ^ log) → Int) (odd : ∀ i, Odd (sigma i).toNat)
    (perm : Fin (2 ^ log) ≃ Fin (2 ^ log))
    (hindex : ∀ i, (sigma i).toNat % (2 * 2 ^ log) = 1 + 2 * 1 * (perm i).val)
    {message : ErrorPoly n} (exponent : Nat)
    (hmessage : reducePoly q n message = reducePoly q n (Xe (n := n) (stride * exponent))) :
    ∑ i, reducePoly q n (substitution (sigma i).toNat (odd i) message) *
        coefficientCoefficient (sigma i) stride inverse table =
      reducePoly q n ((((-1 : Int) ^ (exponent / 2 ^ log) : Int) : ErrorPoly n) *
        (table ⟨exponent % 2 ^ log, Nat.mod_lt _ (by positivity)⟩ : ErrorPoly n)) := by
  apply walsh_of_fourier hperiod hn (fun k ↦ (table k : ErrorPoly n)) hinverse
    sigma odd perm hindex (exponent := exponent) (hmessage := hmessage)
  intro i
  simpa using coefficientCoefficient_fourier hq hn hperiod (odd i) inverse table

/-- The exponent-table Walsh premise, proved from the table and its normalization. -/
theorem exponent_walsh {q inverse stride outputStride log : Nat} (hq : 1 < q)
    (hperiod : n = stride * 2 ^ log) (hn : 0 < n) (table : Fin (2 ^ log) → Int)
    (hnonneg : ∀ k, 0 ≤ table k)
    (hinverse : (inverse : ExactPoly q n) * ((2 ^ log : Nat) : ExactPoly q n) = 1)
    (sigma : Fin (2 ^ log) → Int) (odd : ∀ i, Odd (sigma i).toNat)
    (perm : Fin (2 ^ log) ≃ Fin (2 ^ log))
    (hindex : ∀ i, (sigma i).toNat % (2 * 2 ^ log) = 1 + 2 * 1 * (perm i).val)
    {message : ErrorPoly n} (exponent : Nat)
    (hmessage : reducePoly q n message = reducePoly q n (Xe (n := n) (stride * exponent))) :
    ∑ i, reducePoly q n (substitution (sigma i).toNat (odd i) message) *
        exponentCoefficient (sigma i) stride outputStride inverse table =
      reducePoly q n ((((-1 : Int) ^ (exponent / 2 ^ log) : Int) : ErrorPoly n) *
        Xe (n := n) (outputStride *
          (table ⟨exponent % 2 ^ log, Nat.mod_lt _ (by positivity)⟩).toNat)) := by
  apply walsh_of_fourier hperiod hn
    (fun k ↦ Xe (n := n) (outputStride * (table k).toNat)) hinverse
    sigma odd perm hindex (exponent := exponent) (hmessage := hmessage)
  intro i
  simpa using exponentCoefficient_fourier hq hn hperiod (odd i) inverse table hnonneg

/-- The singleton `Aut` table encrypts the coefficient one, for every odd automorphism. -/
theorem aut_coefficient {q : Nat} (hq : 1 < q) (hn : 0 < n) {sigma : Int}
    (hodd : Odd sigma.toNat) :
    coefficientCoefficient (q := q) (n := n) sigma 1 1 (fun _ : Fin 1 ↦ 1) = 1 := by
  simp only [coefficientCoefficient, Finset.sum_range_one, keyTermC, dif_pos Nat.zero_lt_one]
  change keyTableG [1] [1] (matrixMulScalarLeft (liftInteger 1)
    (ringAutomorphism sigma.toNat
      (multiplyMonomial (matrixPolynomial [1] : ExactMatrix q n 1 1) 0))) 0 0 = 1
  have hmon : multiplyMonomial (matrixPolynomial [1] : ExactMatrix q n 1 1) 0 =
      matrixPolynomial [1] := by
    funext row column
    simp [multiplyMonomial]
  rw [hmon, ringAutomorphism_eq hq hn _ hodd]
  simp [keyTableG, matrixMulScalarLeft, matrixPolynomial, liftInteger]

/-- The singleton `Aut` Walsh premise holds for arbitrary inputs, not only monomials. -/
theorem aut_walsh {q : Nat} (hq : 1 < q) (hn : 0 < n) {sigma : Int}
    (hodd : Odd sigma.toNat) (message : ErrorPoly n) :
    reducePoly q n (substitution sigma.toNat hodd message) *
        coefficientCoefficient sigma 1 1 (fun _ : Fin 1 ↦ 1) =
      reducePoly q n (substitution sigma.toNat hodd message) := by
  rw [aut_coefficient hq hn hodd, mul_one]

/-- The graph's singleton table index family is identically one. -/
theorem aut_table_index :
    (fun index : Fin 1 ↦ MxxIR.selectInt (Int.ofNat index) [1]) = fun _ ↦ (1 : Int) := by
  funext index
  fin_cases index
  rfl

/-- The exact singleton table used by every generated `Aut` LUT has coefficient one. -/
theorem aut_table_coefficient {q : Nat} (hq : 1 < q) (hn : 0 < n) {sigma : Int}
    (hodd : Odd sigma.toNat) :
    coefficientCoefficient (q := q) (n := n) sigma 1 1
      (fun index : Fin 1 ↦ MxxIR.selectInt (Int.ofNat index) [1]) = 1 := by
  rw [aut_table_index]
  exact aut_coefficient hq hn hodd

/-- Every generated `Aut` table discharges `hwalsh` on an arbitrary message. -/
theorem aut_table_walsh {q : Nat} (hq : 1 < q) (hn : 0 < n) {sigma : Int}
    (hodd : Odd sigma.toNat) (message : ErrorPoly n) :
    reducePoly q n (substitution sigma.toNat hodd message) *
        coefficientCoefficient sigma 1 1
          (fun index : Fin 1 ↦ MxxIR.selectInt (Int.ofNat index) [1]) =
      reducePoly q n (substitution sigma.toNat hodd message) := by
  rw [aut_table_coefficient hq hn hodd, mul_one]

/-- The signed, negacyclic extension of a nonempty finite LUT table. -/
noncomputable def signedValue {W : Nat} (hwidth : 0 < W) (table : Fin W → ErrorPoly n)
    (exponent : Nat) : ErrorPoly n :=
  (((-1 : Int) ^ (exponent / W) : Int) : ErrorPoly n) *
    table ⟨exponent % W, Nat.mod_lt _ hwidth⟩

/-- The `state` table value, with its mandatory negacyclic wrap sign. -/
noncomputable def stateValue (exponent : Nat) : ErrorPoly 8 :=
  signedValue (by decide : 0 < 4)
    (fun index : Fin 4 ↦ Xe (n := 8) (2 * (MxxIR.selectInt (Int.ofNat index) [1, 0, 1, 0]).toNat))
      exponent

/-- The actual `state` coefficients discharge `hwalsh` at every monomial input.
No switching equation or Walsh identity is assumed; only the input's monomial form is needed. -/
theorem state_walsh {message : ErrorPoly 8} (exponent : Nat)
    (hmessage : reducePoly W_Stage_preprocess_0_0.modulus_0 8 message =
      reducePoly W_Stage_preprocess_0_0.modulus_0 8 (Xe (n := 8) (2 * exponent))) :
    ∑ branch, reducePoly W_Stage_preprocess_0_0.modulus_0 8
        (substitution (sigmaOf stateSigmas branch).toNat (stateOdd branch) message) *
        stateCoefficients branch =
      reducePoly W_Stage_preprocess_0_0.modulus_0 8 (stateValue exponent) := by
  have hinverse : (1356938545745524308187737599098762797340022013559604731140326430414605958745 :
      ExactPoly W_Stage_preprocess_0_0.modulus_0 8) * (4 :
        ExactPoly W_Stage_preprocess_0_0.modulus_0 8) = 1 :=
    cast_inverse (by decide) (by decide)
  simpa [stateCoefficients, stateValue, signedValue] using
    exponent_walsh
      (q := W_Stage_preprocess_0_0.modulus_0) (inverse :=
        1356938545745524308187737599098762797340022013559604731140326430414605958745)
      (stride := 2) (outputStride := 2) (log := 2)
      (by decide) (by decide) (by decide)
      (fun index : Fin 4 ↦ MxxIR.selectInt (Int.ofNat index) [1, 0, 1, 0]) (by decide)
      hinverse (sigmaOf stateSigmas) stateOdd (Equiv.refl _)
      (by decide) exponent hmessage

/-- The `state1` table value, with its mandatory negacyclic wrap sign. -/
noncomputable def state1Value (exponent : Nat) : ErrorPoly 8 :=
  signedValue (by decide : 0 < 4)
    (fun index : Fin 4 ↦ Xe (n := 8) (2 * (MxxIR.selectInt (Int.ofNat index) [0, 1, 1, 0]).toNat))
      exponent

/-- The actual `state1` coefficients discharge `hwalsh` at every monomial input.
No switching equation or Walsh identity is assumed; only the input's monomial form is needed. -/
theorem state1_walsh {message : ErrorPoly 8} (exponent : Nat)
    (hmessage : reducePoly W_Stage_preprocess_1_1.modulus_0 8 message =
      reducePoly W_Stage_preprocess_1_1.modulus_0 8 (Xe (n := 8) (2 * exponent))) :
    ∑ branch, reducePoly W_Stage_preprocess_1_1.modulus_0 8
        (substitution (sigmaOf stateSigmas branch).toNat (stateOdd branch) message) *
        state1Coefficients branch =
      reducePoly W_Stage_preprocess_1_1.modulus_0 8 (state1Value exponent) := by
  have hinverse : (1356938545745524308187737599098762797340022013559604731140326430414605958745 :
      ExactPoly W_Stage_preprocess_1_1.modulus_0 8) * (4 :
        ExactPoly W_Stage_preprocess_1_1.modulus_0 8) = 1 :=
    cast_inverse (by decide) (by decide)
  simpa [state1Coefficients, state1Value, signedValue] using
    exponent_walsh
      (q := W_Stage_preprocess_1_1.modulus_0) (inverse :=
        1356938545745524308187737599098762797340022013559604731140326430414605958745)
      (stride := 2) (outputStride := 2) (log := 2)
      (by decide) (by decide) (by decide)
      (fun index : Fin 4 ↦ MxxIR.selectInt (Int.ofNat index) [0, 1, 1, 0]) (by decide)
      hinverse (sigmaOf stateSigmas) stateOdd (Equiv.refl _)
      (by decide) exponent hmessage

/-- The `smudge` table value, with its mandatory negacyclic wrap sign. -/
noncomputable def smudgeValue (exponent : Nat) : ErrorPoly 8 :=
  signedValue (by decide : 0 < 8)
    (fun index : Fin 8 ↦ (MxxIR.selectInt (Int.ofNat index) [1, 1, 3, 3, 5, 5, 7, 7] : ErrorPoly 8))
      exponent

/-- The actual `smudge` coefficients discharge `hwalsh` at every monomial input.
No switching equation or Walsh identity is assumed; only the input's monomial form is needed. -/
theorem smudge_walsh {message : ErrorPoly 8} (exponent : Nat)
    (hmessage : reducePoly W_Stage_preprocess_0_0.modulus_0 8 message =
      reducePoly W_Stage_preprocess_0_0.modulus_0 8 (Xe (n := 8) (1 * exponent))) :
    ∑ branch, reducePoly W_Stage_preprocess_0_0.modulus_0 8
        (substitution (sigmaOf smudgeSigmas branch).toNat (smudgeOdd branch) message) *
        smudgeCoefficients branch =
      reducePoly W_Stage_preprocess_0_0.modulus_0 8 (smudgeValue exponent) := by
  have hinverse : (1583094970036445026219027198948556596896692349152872186330380835483706951869 :
      ExactPoly W_Stage_preprocess_0_0.modulus_0 8) * (8 :
        ExactPoly W_Stage_preprocess_0_0.modulus_0 8) = 1 :=
    cast_inverse (by decide) (by decide)
  simpa [smudgeCoefficients, smudgeValue, signedValue] using
    coefficient_walsh
      (q := W_Stage_preprocess_0_0.modulus_0) (inverse :=
        1583094970036445026219027198948556596896692349152872186330380835483706951869)
      (stride := 1) (log := 3)
      (by decide) (by decide) (by decide)
      (fun index : Fin 8 ↦ MxxIR.selectInt (Int.ofNat index) [1, 1, 3, 3, 5, 5, 7, 7]) 
      hinverse (sigmaOf smudgeSigmas) smudgeOdd (Equiv.refl _)
      (by decide) exponent hmessage

/-- The `sigmaReadout` table value, with its mandatory negacyclic wrap sign. -/
noncomputable def sigmaReadoutValue (exponent : Nat) : ErrorPoly 8 :=
  signedValue (by decide : 0 < 8)
    (fun index : Fin 8 ↦ Xe (n := 8) (1 * (MxxIR.selectInt (Int.ofNat index) [1, 1, 3, 3, 5, 5, 7,
      7]).toNat)) exponent

/-- The actual `sigmaReadout` coefficients discharge `hwalsh` at every monomial input.
No switching equation or Walsh identity is assumed; only the input's monomial form is needed. -/
theorem sigmaReadout_walsh {message : ErrorPoly 8} (exponent : Nat)
    (hmessage : reducePoly W_Stage_preprocess_0_0.modulus_0 8 message =
      reducePoly W_Stage_preprocess_0_0.modulus_0 8 (Xe (n := 8) (1 * exponent))) :
    ∑ branch, reducePoly W_Stage_preprocess_0_0.modulus_0 8
        (substitution (sigmaOf sigmaReadoutSigmas branch).toNat (sigmaReadoutOdd branch) message) *
        sigmaReadoutCoefficients branch =
      reducePoly W_Stage_preprocess_0_0.modulus_0 8 (sigmaReadoutValue exponent) := by
  have hinverse : (1583094970036445026219027198948556596896692349152872186330380835483706951869 :
      ExactPoly W_Stage_preprocess_0_0.modulus_0 8) * (8 :
        ExactPoly W_Stage_preprocess_0_0.modulus_0 8) = 1 :=
    cast_inverse (by decide) (by decide)
  simpa [sigmaReadoutCoefficients, sigmaReadoutValue, signedValue] using
    exponent_walsh
      (q := W_Stage_preprocess_0_0.modulus_0) (inverse :=
        1583094970036445026219027198948556596896692349152872186330380835483706951869)
      (stride := 1) (outputStride := 1) (log := 3)
      (by decide) (by decide) (by decide)
      (fun index : Fin 8 ↦ MxxIR.selectInt (Int.ofNat index) [1, 1, 3, 3, 5, 5, 7, 7]) (by decide)
      hinverse (sigmaOf sigmaReadoutSigmas) sigmaReadoutOdd (Equiv.refl _)
      (by decide) exponent hmessage

/-- The `readout` table value, with its mandatory negacyclic wrap sign. -/
noncomputable def readoutValue (exponent : Nat) : ErrorPoly 8 :=
  signedValue (by decide : 0 < 8)
    (fun index : Fin 8 ↦ Xe (n := 8) (1 * (MxxIR.selectInt (Int.ofNat index) [1, 1, 3, 3, 5, 5, 7,
      7]).toNat)) exponent

/-- The actual `readout` coefficients discharge `hwalsh` at every monomial input.
No switching equation or Walsh identity is assumed; only the input's monomial form is needed. -/
theorem readout_walsh {message : ErrorPoly 8} (exponent : Nat)
    (hmessage : reducePoly W_Stage_preprocess_0_0.modulus_0 8 message =
      reducePoly W_Stage_preprocess_0_0.modulus_0 8 (Xe (n := 8) (1 * exponent))) :
    ∑ branch, reducePoly W_Stage_preprocess_0_0.modulus_0 8
        (substitution (sigmaOf readoutSigmas branch).toNat (readoutOdd branch) message) *
        readoutCoefficients branch =
      reducePoly W_Stage_preprocess_0_0.modulus_0 8 (readoutValue exponent) := by
  have hinverse : (1583094970036445026219027198948556596896692349152872186330380835483706951869 :
      ExactPoly W_Stage_preprocess_0_0.modulus_0 8) * (8 :
        ExactPoly W_Stage_preprocess_0_0.modulus_0 8) = 1 :=
    cast_inverse (by decide) (by decide)
  simpa [readoutCoefficients, readoutValue, signedValue] using
    exponent_walsh
      (q := W_Stage_preprocess_0_0.modulus_0) (inverse :=
        1583094970036445026219027198948556596896692349152872186330380835483706951869)
      (stride := 1) (outputStride := 1) (log := 3)
      (by decide) (by decide) (by decide)
      (fun index : Fin 8 ↦ MxxIR.selectInt (Int.ofNat index) [1, 1, 3, 3, 5, 5, 7, 7]) (by decide)
      hinverse (sigmaOf readoutSigmas) readoutOdd (Equiv.refl _)
      (by decide) exponent hmessage

/-- The `sigmaSampler` table value, with its mandatory negacyclic wrap sign. -/
noncomputable def sigmaSamplerValue (exponent : Nat) : ErrorPoly 8 :=
  signedValue (by decide : 0 < 8)
    (fun index : Fin 8 ↦ (MxxIR.selectInt (Int.ofNat index) [0, 0, 0, 0, 0, 1, 0, 1] : ErrorPoly 8))
      exponent

/-- The actual `sigmaSampler` coefficients discharge `hwalsh` at every monomial input.
No switching equation or Walsh identity is assumed; only the input's monomial form is needed. -/
theorem sigmaSampler_walsh {message : ErrorPoly 8} (exponent : Nat)
    (hmessage : reducePoly W_Stage_preprocess_0_0.modulus_0 8 message =
      reducePoly W_Stage_preprocess_0_0.modulus_0 8 (Xe (n := 8) (1 * exponent))) :
    ∑ branch, reducePoly W_Stage_preprocess_0_0.modulus_0 8
        (substitution (sigmaOf sigmaSamplerSigmas branch).toNat (sigmaSamplerOdd branch) message) *
        sigmaSamplerCoefficients branch =
      reducePoly W_Stage_preprocess_0_0.modulus_0 8 (sigmaSamplerValue exponent) := by
  have hinverse : (1583094970036445026219027198948556596896692349152872186330380835483706951869 :
      ExactPoly W_Stage_preprocess_0_0.modulus_0 8) * (8 :
        ExactPoly W_Stage_preprocess_0_0.modulus_0 8) = 1 :=
    cast_inverse (by decide) (by decide)
  simpa [sigmaSamplerCoefficients, sigmaSamplerValue, signedValue] using
    coefficient_walsh
      (q := W_Stage_preprocess_0_0.modulus_0) (inverse :=
        1583094970036445026219027198948556596896692349152872186330380835483706951869)
      (stride := 1) (log := 3)
      (by decide) (by decide) (by decide)
      (fun index : Fin 8 ↦ MxxIR.selectInt (Int.ofNat index) [0, 0, 0, 0, 0, 1, 0, 1]) 
      hinverse (sigmaOf sigmaSamplerSigmas) sigmaSamplerOdd (Equiv.refl _)
      (by decide) exponent hmessage

/-- The `sampler` table value, with its mandatory negacyclic wrap sign. -/
noncomputable def samplerValue (exponent : Nat) : ErrorPoly 8 :=
  signedValue (by decide : 0 < 8)
    (fun index : Fin 8 ↦ (MxxIR.selectInt (Int.ofNat index) [0, 1, 0, 1, 0, 1, 0, 1] : ErrorPoly 8))
      exponent

/-- The actual `sampler` coefficients discharge `hwalsh` at every monomial input.
No switching equation or Walsh identity is assumed; only the input's monomial form is needed. -/
theorem sampler_walsh {message : ErrorPoly 8} (exponent : Nat)
    (hmessage : reducePoly W_Stage_preprocess_0_0.modulus_0 8 message =
      reducePoly W_Stage_preprocess_0_0.modulus_0 8 (Xe (n := 8) (1 * exponent))) :
    ∑ branch, reducePoly W_Stage_preprocess_0_0.modulus_0 8
        (substitution (sigmaOf samplerSigmas branch).toNat (samplerOdd branch) message) *
        samplerCoefficients branch =
      reducePoly W_Stage_preprocess_0_0.modulus_0 8 (samplerValue exponent) := by
  have hinverse : (1583094970036445026219027198948556596896692349152872186330380835483706951869 :
      ExactPoly W_Stage_preprocess_0_0.modulus_0 8) * (8 :
        ExactPoly W_Stage_preprocess_0_0.modulus_0 8) = 1 :=
    cast_inverse (by decide) (by decide)
  simpa [samplerCoefficients, samplerValue, signedValue] using
    coefficient_walsh
      (q := W_Stage_preprocess_0_0.modulus_0) (inverse :=
        1583094970036445026219027198948556596896692349152872186330380835483706951869)
      (stride := 1) (log := 3)
      (by decide) (by decide) (by decide)
      (fun index : Fin 8 ↦ MxxIR.selectInt (Int.ofNat index) [0, 1, 0, 1, 0, 1, 0, 1]) 
      hinverse (sigmaOf samplerSigmas) samplerOdd (Equiv.refl _)
      (by decide) exponent hmessage
end Dio2Proof.PreprocessSwitch
