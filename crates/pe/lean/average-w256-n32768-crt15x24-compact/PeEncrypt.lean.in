import PePrelude
import Stage_encrypt
import Stage_encrypt_Facts

/-!
# Encryption

Under the secret `s`, the encryption publishes the helpers `h_k = s H_k^0 + H_k^1 - (σ_k(s), 1) g + e`
of the `45` helper automorphisms, the compact digit companions of the switching ciphertexts and of
the attribute ciphertexts, and the initial state encoding.
-/

set_option maxRecDepth 200000

namespace PeProof

open Mxx.Primitives MxxRuntime MxxExponentLut

/-- The law of the encryption samples. -/
abbrev encLaw : SamplerLaw := .gaussian (319 / 100) 31

/-- The helper automorphisms: the first-depth switches `kSw b` (members `b < 30`) and the
inverses of the internal switches (members `30 ≤ i < 45`). -/
abbrev helperList : List Int := [257, 513, 769, 1025, 1281, 1537, 1793, 2049, 2305, 2561, 2817, 3073, 3329, 3585, 3841, 4097, 8193, 12289, 16385, 20481, 24577, 28673, 32769, 36865, 40961, 45057, 49153, 53249, 57345, 61441, 61697, 61953, 62209, 62465, 62721, 62977, 63233, 63489, 63745, 64001, 64257, 64513, 64769, 65025, 65281]

/-- The automorphism of helper member `i`. -/
def kHelp (i : Nat) : Nat := (MxxIR.selectInt (Int.ofNat i) helperList).toNat

theorem kHelp_odd (i : Fin 45) : Odd (kHelp i) := by
  fin_cases i <;> decide

theorem kHelp_cast (i : Nat) (hi : i < 45) :
    ((kHelp i : Nat) : Int) = MxxIR.selectInt (Int.ofNat i) helperList := by
  interval_cases i <;> rfl

theorem kHelp_sw (b : Nat) (hb : b < 30) : kHelp b = kSw b := by
  interval_cases b <;> rfl

theorem kHelp_inv (b : Nat) (hb : b < 30) : kHelp (44 - b) = kInv b := by
  interval_cases b <;> rfl

/-- The error of component `j` of the helper of member `i`. -/
noncomputable def helperErr (tape : SampleTape) (path : List Nat) (i : Nat) (j : Fin 2) : ErrorMatrix N 1 30 :=
  tapeWitness tape (path ++ [4, i] ++ [![12, 20] j]) encLaw

/-- The routed secret. -/
theorem routed_secret {k : Int} {sm : ErrorPoly N} {y : ExactMatrix Q N 1 1}
    (h : ringAutomorphismRuns k (fun _ _ ↦ reducePoly Q N sm) y) :
    ∃ hodd : Odd k.toNat, y = fun _ _ ↦ reducePoly Q N (substitution k.toNat hodd sm) := by
  obtain ⟨hodd, hy⟩ := ringAutomorphismRuns_eq hQ hN h
  exact ⟨hodd, hy.trans (by funext a b; rw [reducePoly_substitution])⟩

set_option maxHeartbeats 8000000 in
/-- The helper of member `i` encodes `(σ_k(s), 1) g` under `s` and the helper key `H_k`. -/
theorem encrypt_helper {model : HashModel} {tape : SampleTape} {path : List Nat} {i : Nat} (hi : i < 45)
    {key : ByteArray} {sm : ErrorPoly N} {outputs : ExactMatrix Q N 1 30 × ExactMatrix Q N 1 30 × Unit}
    (h : Stage_prelude.parallel_generatedRoot_4 Backend.backend model tape (path ++ [4, i])
      GeneratedClaim.stage_3_params i ((fun _ _ ↦ reducePoly Q N sm), (key, (1, (g0, ())))) outputs) :
    Rel sm (helperKey model key (kHelp i))
      (fun j ↦ reducePoly Q N ((((1 : Nat) : Int) : ErrorPoly N) *
        unitPair (substitution (kHelp i) (kHelp_odd ⟨i, hi⟩) sm) j) • g0)
      ![outputs.1, outputs.2.1] 31 := by
  obtain ⟨w2, w5, w8, w12, w14, w16, w20, -, -, h2, -, -, h5, -, -, h8, h12, -, -, h14, -, -, h16, h20,
    hout⟩ := h
  have hk := kHelp_cast i hi
  rw [← hk] at h2 h5 h8 h14 h16
  have e2 : w2 = helperKey model key (kHelp i) 0 0 := helperKey_read (c := 0) (by norm_num) h2
  have e5 : w5 = helperKey model key (kHelp i) 1 0 := helperKey_read (c := 2) (by norm_num) h5
  have e14 : w14 = helperKey model key (kHelp i) 0 1 := helperKey_read (c := 1) (by norm_num) h14
  have e16 : w16 = helperKey model key (kHelp i) 1 1 := helperKey_read (c := 3) (by norm_num) h16
  obtain ⟨hodd, e8⟩ := routed_secret h8
  simp only [Int.toNat_natCast] at e8 hodd
  obtain ⟨he12, hb12⟩ := gaussianSampleAt_explicit hN h12
  obtain ⟨he20, hb20⟩ := gaussianSampleAt_explicit hN h20
  subst hout e2 e5 e14 e16 e8 he12 he20
  refine ⟨helperErr tape path i, fun j ↦ ?_, fun j ↦ ?_⟩
  · fin_cases j
    · exact hb12
    · exact hb20
  · funext r c
    rw [Subsingleton.elim r 0]
    fin_cases j <;>
      simp only [Fin.zero_eta, Fin.mk_one, Fin.isValue, Matrix.cons_val_zero, Matrix.cons_val_one,
        matrixAdd, matrixSub, matrixMulScalarLeft_eq, unitPair, helperErr,
        Matrix.add_apply, Matrix.sub_apply, Matrix.smul_apply, smul_eq_mul, map_one,
        Nat.cast_one, Int.cast_one, one_mul, Matrix.one_apply_eq]

set_option maxHeartbeats 16000000 in
/-- The streamed helper ciphertexts are published by the prelude. -/
theorem prelude_helpers {model : HashModel} {tape : SampleTape} {path : List Nat}
    {secret : ExactMatrix Q N 1 1} {key : ByteArray} {x : Fin 2 → Int}
    {BQ : ExactMatrix PQ N 1 295} {message : Fin 1 → Int} {outputs : PreludeOutputs}
    (h : Stage_prelude.generatedRoot Backend.backend model tape path GeneratedClaim.stage_3_params
      (secret, key, x, BQ, message, ()) outputs)
    {sm : ErrorPoly N} (hsecret : secret = fun _ _ ↦ reducePoly Q N sm) :
    ∀ index : Fin 45, Rel sm (helperKey model key (kHelp index))
      (fun component ↦ reducePoly Q N ((((1 : Nat) : Int) : ErrorPoly N) *
        unitPair (substitution (kHelp index) (kHelp_odd index) sm) component) • g0)
      ![outputs.1 index, outputs.2.1 index] 31 := by
  obtain ⟨witness, body⟩ := h
  let ctx : Stage_prelude.generatedRoot.Ctx Backend.backend model tape path
      GeneratedClaim.stage_3_params := ⟨(secret, key, x, BQ, message, ()), witness⟩
  have facts := Stage_prelude.generatedRoot.facts ctx body
  have gadget := gadgetMatrixRuns_unique facts.c0.h3_0 g0_runs
  rw [facts.c0.hout]
  intro index
  have helper := facts.c0.h4_1 index
  change Stage_prelude.parallel_generatedRoot_4 _ _ _ _ _ _
    (secret, key, 1, Stage_prelude.generatedRoot.w_3_0 ctx, ()) _ at helper
  rw [hsecret, gadget] at helper
  exact encrypt_helper index.isLt helper

/-- `companion-key` -/
def compKeyBytes : Blob := [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121]
/-- `automorphism` -/
def autBytes : Blob := [97, 117, 116, 111, 109, 111, 114, 112, 104, 105, 115, 109]
/-- `column` -/
def columnBytes : Blob := [99, 111, 108, 117, 109, 110]
/-- `digit-row` -/
def digitRowBytes : Blob := [100, 105, 103, 105, 116, 45, 114, 111, 119]

/-- The compact companion key of automorphism `k`, column `jj`, and digit row `row`. -/
noncomputable def lutCompKey (model : HashModel) (key : ByteArray) (k jj row : Int) :
    Fin 2 → Fin 2 → ExactMatrix PQ N 1 1 :=
  fun r j ↦ model.sample PQ N 1 1 key (completeHashTag dynamicPrefix [.bytes schemaBytes,
    .bytes compKeyBytes, .bytes autBytes, .u64Le k, .bytes columnBytes, .u64Le jj,
    .bytes digitRowBytes, .u64Le row, .bytes componentBytes, .u64Le (2 * (r : Int) + (j : Int))])

/-- Digit `a` of the entry `y`, as an integer polynomial. -/
noncomputable def entryDigit (y : ExactPoly Q N) (a : Fin 30) : ErrorPoly N :=
  centeredIntLift ((entryDigits 30 Backend.backend 4096 30 (fun _ _ ↦ y) : ExactMatrix Q N 30 1) a 0)

theorem entryDigit_spec (y : ExactPoly Q N) (a : Fin 30) :
    reducePoly Q N (entryDigit y a) =
        (entryDigits 30 Backend.backend 4096 30 (fun _ _ ↦ y) : ExactMatrix Q N 30 1) a 0 ∧
      ∀ k, ((entryDigit y a).coeff k).natAbs ≤ 2048 := by
  obtain ⟨w, hw, hb⟩ := (decompose_entry hQ hN Backend.layout0 layoutQ rfl g0_runs
    (entryDigits_runs hQ hN g0_runs Backend.layout0 layoutQ rfl (fun _ _ ↦ y))).2
  have he : (entryDigits 30 Backend.backend 4096 30 (fun _ _ ↦ y) : ExactMatrix Q N 30 1) a 0 =
      reducePoly Q N (w a 0) := by rw [hw]; rfl
  have hlift : entryDigit y a = w a 0 := by
    rw [entryDigit, he]
    exact centeredIntLift_reducePoly hQ hN _ fun i ↦ by
      have h1 := hb a 0 i
      have h2 : Backend.layout0.base / 2 = 2048 := rfl
      rw [h2] at h1
      have : (2048 : Nat) * 2 < Q := by norm_num
      omega
  refine ⟨by rw [hlift, he], fun k ↦ ?_⟩
  rw [hlift]
  exact hb a 0 k

/-- The error of component `j` of LUT companion `i`. -/
noncomputable def compErr (tape : SampleTape) (path : List Nat) (i : Nat) (j : Fin 2) : ErrorMatrix N 1 1 :=
  tapeWitness tape (path ++ [5, i] ++ [![33, 43] j]) encLaw

/-- The centered rebase of the reduction of a polynomial with small coefficients is its reduction. -/
theorem rebase_lift {q p : Nat} (hq : 1 < q) {x : ErrorPoly N} (hx : ∀ i, 2 * (x.coeff i).natAbs < q)
    {out : ExactMatrix p N 1 1} (h : centeredRebaseRuns (fun _ _ ↦ reducePoly q N x) out) :
    out = fun _ _ ↦ reducePoly p N x := by
  rw [centeredRebaseRuns_reduce h]
  funext r c
  rw [centeredIntLift_reducePoly hq hN _ hx]

theorem rebase_one {q p : Nat} (hq : 2 < q) {out : ExactMatrix p N 1 1}
    (h : centeredRebaseRuns (1 : ExactMatrix q N 1 1) out) : out = 1 := by
  have h1 : (1 : ExactMatrix q N 1 1) = fun _ _ ↦ reducePoly q N 1 := by
    funext r c
    rw [Subsingleton.elim r 0, Subsingleton.elim c 0]
    simp
  rw [h1] at h
  rw [rebase_small hq (by simpa using polyNorm_Xe (n := N) hN1 0) h]
  funext r c
  rw [Subsingleton.elim r 0, Subsingleton.elim c 0]
  simp

/-- Component `j` of the bottom row of the switching ciphertext of first-depth branch `b`. -/
noncomputable def switchBot (model : HashModel) (key : ByteArray) (hf0 hf1 : Fin 45 → ExactMatrix Q N 1 30)
    (b : Fin 30) (j : Fin 2) : ExactMatrix Q N 1 30 :=
  substituteMatrix (kSw b) (kSw_odd b) (helperKey model key (kInv b) 1 j) -
    substituteMatrix (kSw b) (kSw_odd b) (![hf0, hf1] j ⟨44 - b, by omega⟩)

set_option maxHeartbeats 32000000 in
/-- LUT companion `i` (branch `b = i / 1800`, digit row `30 + a` for `a = (i / 60) % 30`, column
`jj = i % 60`) is a compact encoding over `R_{PQ}` under the routed mask `σ_k(s)` of
`P d (σ_k(s), 1)`, for the digit `d` of entry `jj` of the bottom row of the switching ciphertext. -/
theorem encrypt_companion {model : HashModel} {tape : SampleTape} {path : List Nat} {i : Nat}
    (hi : i < 54000) {key : ByteArray} {sm : ErrorPoly N} (hs : polyNorm sm ≤ 1)
    {hf0 hf1 : Fin 45 → ExactMatrix Q N 1 30} {outputs : ExactMatrix PQ N 1 1 × ExactMatrix PQ N 1 1 × Unit}
    (h : Stage_encrypt.parallel_generatedRoot_5 Backend.backend model tape (path ++ [5, i])
      GeneratedClaim.stage_4_params i ((fun _ _ ↦ reducePoly Q N sm), (key, (1, (hf0, (hf1, ()))))) outputs) :
    RelE (substitution (kSw (i / 1800)) (kSw_odd ⟨i / 1800, by omega⟩) sm)
      (lutCompKey model key (kSw (i / 1800)) ((i % 60 : Nat) : Int) ((30 + (i / 60) % 30 : Nat) : Int))
      (fun j _ _ ↦ reducePoly PQ N (((P : Nat) : Int) *
        (entryDigit (joinedEntry ![switchBot model key hf0 hf1 ⟨i / 1800, by omega⟩ 0, switchBot model key hf0 hf1 ⟨i / 1800, by omega⟩ 1] ⟨i % 60, by omega⟩)
          ⟨(i / 60) % 30, by omega⟩ *
          unitPair (substitution (kSw (i / 1800)) (kSw_odd ⟨i / 1800, by omega⟩) sm) j)))
      ![outputs.1, outputs.2.1] (compErr tape path i) ∧
    ∀ j, CoeffBound (compErr tape path i j) 31 := by
  obtain ⟨w1, w2, w4, w7, w8, w13, w14, w17, w19, w20, w22, w24, w25, w26, w27, w29, w33, w35, w37, w43, ⟨-, -, -, h1, h2, -, -, -, -, -, -, h4, h7, -, -, -, -, -, -, h8, -, -, -, -, -, -, -, -, -, -, -, -⟩, ⟨-, -, h13, -, -, -, h14, -, -, -, -, -, h17, -, -, -, h19, -, -, -, h20, -, -, h22, -, -, -, h24, h25, -, -, -⟩, ⟨-, -, -, -, -, h26, h27, h29, h33, -, -, -, -, -, -, h35, -, -, -, -, -, -, h37, h43, hout⟩⟩ := h
  have hb : i / 1800 < 30 := by omega
  have hfb : Int.fdiv (Int.ofNat i) 1800 = Int.ofNat (i / 1800) := by
    rw [Int.fdiv_eq_ediv_of_nonneg _ (by norm_num)]; simp only [Int.ofNat_eq_natCast]; omega
  have hjj : Int.fmod (Int.ofNat i) 60 = ((i % 60 : Nat) : Int) := by
    rw [Int.fmod_eq_emod_of_nonneg _ (by norm_num)]; simp only [Int.ofNat_eq_natCast]; omega
  have ha : Int.fmod (Int.fdiv (Int.ofNat i) 60) 30 = (((i / 60) % 30 : Nat) : Int) := by
    rw [Int.fdiv_eq_ediv_of_nonneg _ (by norm_num), Int.fmod_eq_emod_of_nonneg _ (by omega)]
    simp only [Int.ofNat_eq_natCast]; omega
  have hrow : (30 : Int) + Int.fmod (Int.fdiv (Int.ofNat i) 60) 30 = ((30 + (i / 60) % 30 : Nat) : Int) := by
    rw [ha]; push_cast; ring
  have hone : Int.fdiv (30 + Int.fmod (Int.fdiv (Int.ofNat i) 60) 30) 30 = 1 := by
    rw [ha, Int.fdiv_eq_ediv_of_nonneg _ (by omega)]; omega
  rw [hfb] at h1 h4 h8 h35 h37
  rw [hjj, hrow] at h4 h8 h35 h37
  have hk := kSw_cast (i / 1800) hb
  rw [← hk] at h4 h8 h35 h37
  -- The mask.
  have e1 := routing_kSw hb h1
  set σs := substitution (kSw (i / 1800)) (kSw_odd ⟨i / 1800, hb⟩) sm with hσs
  have e1' : w1 = fun _ _ ↦ reducePoly Q N σs := by
    rw [e1]; funext a c
    simp only [substituteMatrix]
    exact (reducePoly_substitution _ _ _).symm
  rw [e1'] at h2
  have hσn : polyNorm σs ≤ 1 :=
    (polyNorm_substitution_le hN _ (kSw_coprime ⟨i / 1800, hb⟩) sm).trans hs
  have e2 := rebase_small (q := Q) (by norm_num) hσn h2
  have e7 := rebase_one (q := Q) (by norm_num) h7
  -- The keys.
  have e4 : w4 = lutCompKey model key (kSw (i / 1800)) ((i % 60 : Nat) : Int) ((30 + (i / 60) % 30 : Nat) : Int) 0 0 :=
    hashSample_eq h4
  have e8 : w8 = lutCompKey model key (kSw (i / 1800)) ((i % 60 : Nat) : Int) ((30 + (i / 60) % 30 : Nat) : Int) 1 0 :=
    hashSample_eq h8
  have e35 : w35 = lutCompKey model key (kSw (i / 1800)) ((i % 60 : Nat) : Int) ((30 + (i / 60) % 30 : Nat) : Int) 0 1 :=
    hashSample_eq h35
  have e37 : w37 = lutCompKey model key (kSw (i / 1800)) ((i % 60 : Nat) : Int) ((30 + (i / 60) % 30 : Nat) : Int) 1 1 :=
    hashSample_eq h37
  have hidx : MxxIR.selectInt (Int.ofNat (i / 1800)) switchInvMembers =
      ((44 - i / 1800 : Nat) : Int) := switchInvMembers_eq _ hb
  rw [hfb] at h14 h19 h20 h24
  have inverse_cast := kInv_cast (i / 1800) hb
  rw [← inverse_cast] at h14 h20
  rw [hfb, hidx] at h17 h22
  obtain ⟨_, e17⟩ := familyGet_eq h17
  obtain ⟨_, e22⟩ := familyGet_eq h22
  have e14 : w14 = helperKey model key (kInv (i / 1800)) 1 0 :=
    helperKey_read (c := 2) (by norm_num) h14
  have e20 : w20 = helperKey model key (kInv (i / 1800)) 1 1 :=
    helperKey_read (c := 3) (by norm_num) h20
  have e19 := routing_kSw hb h19
  have e24 := routing_kSw hb h24
  rw [e14, e17] at e19
  rw [e20, e22] at e24
  have e25 := concat_joinRow (r := ![_, _]) h25
  rw [e19, e24] at e25
  rw [hjj] at h26
  obtain ⟨_, e26⟩ := sliceMatrix_column h26
  have hy : w26 = fun _ _ ↦ joinedEntry
      ![switchBot model key hf0 hf1 ⟨i / 1800, hb⟩ 0,
        switchBot model key hf0 hf1 ⟨i / 1800, hb⟩ 1] ⟨i % 60, by omega⟩ := by
    funext row column
    rw [Subsingleton.elim row 0, Subsingleton.elim column 0, e26, e25]
    have index_cast : ((i : Int) % 60).toNat = i % 60 := by omega
    simp [switchBot, joinRow, joinedEntry, substituteMatrix, matrixSub, index_cast]
  have e27 := entryDigits_eq h27
  rw [hy] at e27
  set y := joinedEntry ![switchBot model key hf0 hf1 ⟨i / 1800, hb⟩ 0, switchBot model key hf0 hf1 ⟨i / 1800, hb⟩ 1] ⟨i % 60, by omega⟩ with hydef
  set a : Fin 30 := ⟨(i / 60) % 30, by omega⟩ with hadef
  have hrow2 : Int.fmod (30 + Int.fmod (Int.fdiv (Int.ofNat i) 60) 30) 30 = (((i / 60) % 30 : Nat) : Int) := by
    rw [ha, Int.fmod_eq_emod_of_nonneg _ (by omega)]; omega
  rw [hrow2] at h13
  have e13 := unitRow_window (c := (((i / 60) % 30 : Nat) : Int)) (by ring) h13
  have e28 : (matrixMul w13 w27 : ExactMatrix Q N 1 1) = fun _ _ ↦ reducePoly Q N (entryDigit y a) := by
    funext r c
    rw [Subsingleton.elim r 0, Subsingleton.elim c 0, matrixMul_eq_mul, e13, e27,
      unitRow_mul_entry _ (by omega), (entryDigit_spec y a).1]
  have hδ : ∀ k, 2 * ((entryDigit y a).coeff k).natAbs < Q := fun k ↦ by
    have := (entryDigit_spec y a).2 k
    have : (2048 : Nat) * 2 < Q := by norm_num
    omega
  rw [e28] at h29
  have e29 := rebase_lift hQ hδ h29
  obtain ⟨he33, hb33⟩ := gaussianSampleAt_explicit hN h33
  obtain ⟨he43, hb43⟩ := gaussianSampleAt_explicit hN h43
  refine ⟨fun j ↦ ?_, fun j ↦ ?_⟩
  · subst hout
    funext r c
    rw [Subsingleton.elim r 0, Subsingleton.elim c 0]
    fin_cases j
    · simp only [Fin.zero_eta, Fin.isValue, Matrix.cons_val_zero, matrixAdd, matrixSub,
        matrixMulScalarLeft_eq, e2, e4, e7, e8, e29, he33, compErr, Matrix.add_apply, Matrix.sub_apply,
        Matrix.smul_apply, smul_eq_mul, unitPair, Matrix.one_apply_eq, one_mul, map_mul]
      simp [matrixPolynomial, map_ofNat]
    · simp only [Fin.mk_one, Fin.isValue, Matrix.cons_val_one, matrixAdd, matrixSub,
        matrixMulScalarLeft_eq, e2, e7, e35, e37, e29, he43, compErr, Matrix.add_apply, Matrix.sub_apply,
        Matrix.smul_apply, smul_eq_mul, unitPair, Matrix.one_apply_eq, map_mul]
      simp [matrixPolynomial, map_ofNat]
  · fin_cases j
    · exact hb33
    · exact hb43

/-! ## The compact digit companions of the attribute ciphertexts -/

/-- `rhs` -/
def rhsBytes : Blob := [114, 104, 115]

/-- The compact companion key of attribute position `x`, column `jj`, and digit row `row`. -/
noncomputable def rhsCompKey (model : HashModel) (key : ByteArray) (x jj row : Int) :
    Fin 2 → Fin 2 → ExactMatrix PQ N 1 1 :=
  fun r j ↦ model.sample PQ N 1 1 key (completeHashTag dynamicPrefix [.bytes schemaBytes,
    .bytes compKeyBytes, .bytes rhsBytes, .u64Le 0, .bytes memberBytes, .u64Le x, .bytes columnBytes,
    .u64Le jj, .bytes digitRowBytes, .u64Le row, .bytes componentBytes, .u64Le (2 * (r : Int) + (j : Int))])

/-- The error of component `j` of product companion `i`. -/
noncomputable def rhsCompErr (tape : SampleTape) (path : List Nat) (i : Nat) (j : Fin 2) : ErrorMatrix N 1 1 :=
  tapeWitness tape (path ++ [37, i] ++ [![26, 36] j]) encLaw

set_option maxHeartbeats 32000000 in
/-- Product companion `i` (position `x = i / 1800`, digit row `30 + a` for `a = (i / 60) % 30`,
column `jj = i % 60`) is a compact encoding over `R_{PQ}` under `s` of `P d (s, 1)`, for the digit
`d` of entry `jj` of the bottom row `Y` of the attribute ciphertext of position `x`. -/
theorem encrypt_rhsCompanion {model : HashModel} {tape : SampleTape} {path : List Nat} {i : Nat}
    (hi : i < 3600) {key : ByteArray} {sm : ErrorPoly N} (hs : polyNorm sm ≤ 1)
    {Y0 Y1 : Fin 2 → ExactMatrix Q N 1 30}
    {outputs : ExactMatrix PQ N 1 1 × ExactMatrix PQ N 1 1 × Unit}
    (h : Stage_encrypt.parallel_generatedRoot_37 Backend.backend model tape (path ++ [37, i])
      GeneratedClaim.stage_4_params i ((fun _ _ ↦ reducePoly Q N sm), (key, (1, (Y0, (Y1, ()))))) outputs) :
    RelE sm (rhsCompKey model key ((i / 1800 : Nat) : Int) ((i % 60 : Nat) : Int) ((30 + (i / 60) % 30 : Nat) : Int))
      (fun j _ _ ↦ reducePoly PQ N (((P : Nat) : Int) *
        (entryDigit (joinedEntry ![Y0 ⟨i / 1800, by omega⟩, Y1 ⟨i / 1800, by omega⟩] ⟨i % 60, by omega⟩) ⟨(i / 60) % 30, by omega⟩ * unitPair sm j)))
      ![outputs.1, outputs.2.1] (rhsCompErr tape path i) ∧
    ∀ j, CoeffBound (rhsCompErr tape path i j) 31 := by
  obtain ⟨w1, w3, w6, w7, w12, w15, w17, w18, w19, w20, w22, w26, w28, w30, w36, ⟨h1, -, -, -, -, h3, h6, -, -, -, -, h7, -, -, -, -, -, -, -, -, -, -, -, -, -, -, h12, -, -, -, h15, -⟩, ⟨-, h17, h18, -, -, -, -, -, -, -, -, h19, h20, h22, h26, -, -, -, -, h28, -, -, -, -, h30, h36, hout⟩⟩ := h
  have hfb : Int.fdiv (Int.ofNat i) 1800 = ((i / 1800 : Nat) : Int) := by
    rw [Int.fdiv_eq_ediv_of_nonneg _ (by norm_num)]; simp only [Int.ofNat_eq_natCast]; omega
  have hjj : Int.fmod (Int.ofNat i) 60 = ((i % 60 : Nat) : Int) := by
    rw [Int.fmod_eq_emod_of_nonneg _ (by norm_num)]; simp only [Int.ofNat_eq_natCast]; omega
  have ha : Int.fmod (Int.fdiv (Int.ofNat i) 60) 30 = (((i / 60) % 30 : Nat) : Int) := by
    rw [Int.fdiv_eq_ediv_of_nonneg _ (by norm_num), Int.fmod_eq_emod_of_nonneg _ (by omega)]
    simp only [Int.ofNat_eq_natCast]; omega
  have hrow : (30 : Int) + Int.fmod (Int.fdiv (Int.ofNat i) 60) 30 = ((30 + (i / 60) % 30 : Nat) : Int) := by
    rw [ha]; push_cast; ring
  have hone : Int.fdiv (30 + Int.fmod (Int.fdiv (Int.ofNat i) 60) 30) 30 = 1 := by
    rw [ha, Int.fdiv_eq_ediv_of_nonneg _ (by omega)]; omega
  rw [hfb, hjj, hrow] at h3 h7 h28 h30
  -- The mask.
  have e1 := rebase_small (q := Q) (by norm_num) hs h1
  have e6 := rebase_one (q := Q) (by norm_num) h6
  -- The keys.
  have e3 : w3 = rhsCompKey model key ((i / 1800 : Nat) : Int) ((i % 60 : Nat) : Int) ((30 + (i / 60) % 30 : Nat) : Int) 0 0 :=
    hashSample_eq h3
  have e7 : w7 = rhsCompKey model key ((i / 1800 : Nat) : Int) ((i % 60 : Nat) : Int) ((30 + (i / 60) % 30 : Nat) : Int) 1 0 :=
    hashSample_eq h7
  have e28 : w28 = rhsCompKey model key ((i / 1800 : Nat) : Int) ((i % 60 : Nat) : Int) ((30 + (i / 60) % 30 : Nat) : Int) 0 1 :=
    hashSample_eq h28
  have e30 : w30 = rhsCompKey model key ((i / 1800 : Nat) : Int) ((i % 60 : Nat) : Int) ((30 + (i / 60) % 30 : Nat) : Int) 1 1 :=
    hashSample_eq h30
  rw [hfb] at h15 h17
  obtain ⟨_, e15⟩ := familyGet_eq h15
  obtain ⟨_, e17⟩ := familyGet_eq h17
  rw [hjj] at h19
  obtain ⟨_, e19⟩ := sliceMatrix_column h19
  rw [concatColumns_entry h18, e15, e17] at e19
  have hy : w19 = fun _ _ ↦ joinedEntry
      ![Y0 ⟨i / 1800, by omega⟩, Y1 ⟨i / 1800, by omega⟩] ⟨i % 60, by omega⟩ := by
    funext row column
    rw [Subsingleton.elim row 0, Subsingleton.elim column 0, e19]
    rfl
  have e20 := entryDigits_eq h20
  rw [hy] at e20
  set y := joinedEntry ![Y0 ⟨i / 1800, by omega⟩, Y1 ⟨i / 1800, by omega⟩] ⟨i % 60, by omega⟩ with hydef
  set a : Fin 30 := ⟨(i / 60) % 30, by omega⟩ with hadef
  have hrow2 : Int.fmod (30 + Int.fmod (Int.fdiv (Int.ofNat i) 60) 30) 30 = (((i / 60) % 30 : Nat) : Int) := by
    rw [ha, Int.fmod_eq_emod_of_nonneg _ (by omega)]; omega
  rw [hrow2] at h12
  have e12 := unitRow_window (c := (((i / 60) % 30 : Nat) : Int)) (by ring) h12
  have e21 : (matrixMul w12 w20 : ExactMatrix Q N 1 1) = fun _ _ ↦ reducePoly Q N (entryDigit y a) := by
    funext r c
    rw [Subsingleton.elim r 0, Subsingleton.elim c 0, matrixMul_eq_mul, e12, e20,
      unitRow_mul_entry _ (by omega), (entryDigit_spec y a).1]
  have hδ : ∀ k, 2 * ((entryDigit y a).coeff k).natAbs < Q := fun k ↦ by
    have := (entryDigit_spec y a).2 k
    have : (2048 : Nat) * 2 < Q := by norm_num
    omega
  rw [e21] at h22
  have e22 := rebase_lift hQ hδ h22
  obtain ⟨he26, hb26⟩ := gaussianSampleAt_explicit hN h26
  obtain ⟨he36, hb36⟩ := gaussianSampleAt_explicit hN h36
  refine ⟨fun j ↦ ?_, fun j ↦ ?_⟩
  · subst hout
    funext r c
    rw [Subsingleton.elim r 0, Subsingleton.elim c 0]
    fin_cases j
    · simp only [Fin.zero_eta, Fin.isValue, Matrix.cons_val_zero, matrixAdd, matrixSub,
        matrixMulScalarLeft_eq, e1, e3, e6, e7, e22, he26, rhsCompErr, Matrix.add_apply, Matrix.sub_apply,
        Matrix.smul_apply, smul_eq_mul, unitPair, Matrix.one_apply_eq, one_mul, map_mul]
      simp [matrixPolynomial, map_ofNat]
    · simp only [Fin.mk_one, Fin.isValue, Matrix.cons_val_one, matrixAdd, matrixSub,
        matrixMulScalarLeft_eq, e1, e6, e28, e30, e22, he36, rhsCompErr, Matrix.add_apply, Matrix.sub_apply,
        Matrix.smul_apply, smul_eq_mul, unitPair, Matrix.one_apply_eq, map_mul]
      simp [matrixPolynomial, map_ofNat]
  · fin_cases j
    · exact hb26
    · exact hb36

/-! ## The encryption -/

/-- `input-key` -/
def inputKeyBytes : Blob := [105, 110, 112, 117, 116, 45, 107, 101, 121]

/-- The hashed key `A` of the initial state encoding. -/
noncomputable def inputKey (model : HashModel) (key : ByteArray) : Fin 2 → Fin 2 → ExactMatrix Q N 1 30 :=
  fun r j ↦ model.sample Q N 1 30 key (completeHashTag dynamicPrefix [.bytes schemaBytes,
    .bytes inputKeyBytes, .u64Le 0, .bytes componentBytes, .u64Le (2 * (r : Int) + (j : Int))])

/-- The error of component `j` of the initial state encoding. -/
noncomputable def stateErr (tape : SampleTape) (path : List Nat) (j : Fin 2) : ErrorMatrix N 1 30 :=
  tapeWitness tape (path ++ [![23, 33] j]) encLaw

abbrev EncryptOutputs := (Fin 54000 → ExactMatrix PQ N 1 1) × (Fin 54000 → ExactMatrix PQ N 1 1) ×
  (Fin 45 → ExactMatrix Q N 1 30) × (Fin 45 → ExactMatrix Q N 1 30) × ExactMatrix Q N 1 30 ×
  ExactMatrix Q N 1 30 × (Fin 3600 → ExactMatrix PQ N 1 1) × (Fin 3600 → ExactMatrix PQ N 1 1) × Unit

theorem encrypt_helpers {model : HashModel} {tape : SampleTape} {path : List Nat}
    {secret : ExactMatrix Q N 1 1} {key : ByteArray}
    {hf0 hf1 : Fin 45 → ExactMatrix Q N 1 30} {c e : Int}
    {Y0 Y1 : Fin 2 → ExactMatrix Q N 1 30} {outputs : EncryptOutputs}
    (h : Stage_encrypt.generatedRoot Backend.backend model tape path GeneratedClaim.stage_4_params
      (secret, key, hf0, hf1, c, e, Y0, Y1, ()) outputs) :
    outputs.2.2.1 = hf0 ∧ outputs.2.2.2.1 = hf1 := by
  obtain ⟨w, hbody⟩ := h
  let ctx : Stage_encrypt.generatedRoot.Ctx Backend.backend model tape path
      GeneratedClaim.stage_4_params := ⟨(secret, key, hf0, hf1, c, e, Y0, Y1, ()), w⟩
  have F := Stage_encrypt.generatedRoot.facts ctx hbody
  rw [F.c0.hout]
  exact ⟨rfl, rfl⟩

set_option maxHeartbeats 32000000 in
/-- The encryption: the helpers, the compact companions of the switches and of the attribute
ciphertexts, and the initial state encoding of `c X^e`. -/
theorem encrypt_explicit {model : HashModel} {tape : SampleTape} {path : List Nat}
    {secret : ExactMatrix Q N 1 1} {key : ByteArray}
    {hf0 hf1 : Fin 45 → ExactMatrix Q N 1 30} {c e : Int} {Y0 Y1 : Fin 2 → ExactMatrix Q N 1 30}
    {outputs : EncryptOutputs}
    (h : Stage_encrypt.generatedRoot Backend.backend model tape path GeneratedClaim.stage_4_params
      (secret, key, hf0, hf1, c, e, Y0, Y1, ()) outputs)
    {sm : ErrorPoly N} (hsecret : secret = fun _ _ ↦ reducePoly Q N sm) (hs : polyNorm sm ≤ 1)
    (helpers : ∀ index : Fin 45, Rel sm (helperKey model key (kHelp index))
      (fun component ↦ reducePoly Q N ((((1 : Nat) : Int) : ErrorPoly N) *
        unitPair (substitution (kHelp index) (kHelp_odd index) sm) component) • g0)
      ![hf0 index, hf1 index] 31) :
    (∀ index : Fin 45, Rel sm (helperKey model key (kHelp index))
      (fun component ↦ reducePoly Q N ((((1 : Nat) : Int) : ErrorPoly N) *
        unitPair (substitution (kHelp index) (kHelp_odd index) sm) component) • g0)
      ![outputs.2.2.1 index, outputs.2.2.2.1 index] 31) ∧
    (∀ i : Fin 54000,
      RelE (substitution (kSw (i / 1800)) (kSw_odd ⟨i / 1800, by omega⟩) sm)
        (lutCompKey model key (kSw (i / 1800)) ((i % 60 : Nat) : Int) ((30 + (i / 60) % 30 : Nat) : Int))
        (fun j _ _ ↦ reducePoly PQ N (((P : Nat) : Int) *
          (entryDigit (joinedEntry ![switchBot model key hf0 hf1 ⟨i / 1800, by omega⟩ 0,
              switchBot model key hf0 hf1 ⟨i / 1800, by omega⟩ 1] ⟨i % 60, by omega⟩)
            ⟨(i / 60) % 30, by omega⟩ *
            unitPair (substitution (kSw (i / 1800)) (kSw_odd ⟨i / 1800, by omega⟩) sm) j)))
        ![outputs.1 i, outputs.2.1 i] (compErr tape path i) ∧
      ∀ j, CoeffBound (compErr tape path i j) 31) ∧
    (∀ i : Fin 3600,
      RelE sm (rhsCompKey model key ((i / 1800 : Nat) : Int) ((i % 60 : Nat) : Int) ((30 + (i / 60) % 30 : Nat) : Int))
        (fun j _ _ ↦ reducePoly PQ N (((P : Nat) : Int) *
          (entryDigit (joinedEntry ![Y0 ⟨i / 1800, by omega⟩, Y1 ⟨i / 1800, by omega⟩] ⟨i % 60, by omega⟩)
            ⟨(i / 60) % 30, by omega⟩ * unitPair sm j)))
        ![outputs.2.2.2.2.2.2.1 i, outputs.2.2.2.2.2.2.2.1 i] (rhsCompErr tape path i) ∧
      ∀ j, CoeffBound (rhsCompErr tape path i j) 31) ∧
    RelE sm (inputKey model key)
      (fun j ↦ reducePoly Q N ((((1 : Nat) : Int) : ErrorPoly N) * message c e * unitPair sm j) • g0)
      ![outputs.2.2.2.2.1, outputs.2.2.2.2.2.1] (stateErr tape path) ∧
    ∀ j, CoeffBound (stateErr tape path j) 31 := by
  obtain ⟨w, hbody⟩ := h
  let ctx : Stage_encrypt.generatedRoot.Ctx Backend.backend model tape path GeneratedClaim.stage_4_params :=
    ⟨(secret, key, hf0, hf1, c, e, Y0, Y1, ()), w⟩
  have F := Stage_encrypt.generatedRoot.facts ctx hbody
  have hw0 : Stage_encrypt.generatedRoot.w_0_0 ctx = fun _ _ ↦ reducePoly Q N sm := hsecret
  rw [F.c0.hout]
  refine ⟨fun index ↦ ?_, fun i ↦ ?_, fun i ↦ ?_, ?_, fun j ↦ ?_⟩
  · exact helpers index
  · have hi5 := F.c0.h5_1 i
    rw [hw0] at hi5
    exact encrypt_companion i.isLt hs hi5
  · have hi37 := F.c0.h37_1 i
    rw [hw0] at hi37
    exact encrypt_rhsCompanion i.isLt hs hi37
  · have e7 : Stage_encrypt.generatedRoot.w_7_0 ctx = inputKey model key 0 0 := hashSample_eq F.c0.h7_0
    have e9 : Stage_encrypt.generatedRoot.w_9_0 ctx = inputKey model key 1 0 := hashSample_eq F.c0.h9_0
    have e25 : Stage_encrypt.generatedRoot.w_25_0 ctx = inputKey model key 0 1 := hashSample_eq F.c0.h25_0
    have e27 : Stage_encrypt.generatedRoot.w_27_0 ctx = inputKey model key 1 1 := hashSample_eq F.c0.h27_0
    have hg12 := gadgetMatrixRuns_unique F.c0.h12_0 g0_runs
    have h13 := F.c0.h13_0
    rw [hg12] at h13
    have e13 : Stage_encrypt.generatedRoot.w_13_0 ctx = g0 := centeredRebaseRuns_unique h13 hg0
    obtain ⟨he23, -⟩ := gaussianSampleAt_explicit hN F.c0.h23_0
    obtain ⟨he33, -⟩ := gaussianSampleAt_explicit hN F.c0.h33_0
    have e20 : Stage_encrypt.generatedRoot.w_20_0 ctx = fun _ _ ↦ reducePoly Q N (message c e) := by
      show matrixMulScalarLeft (matrixPolynomial [1]) (multiplyMonomial (liftInteger c) e) = _
      rw [multiplyMonomial_liftInteger]
      funext r c'
      rw [Subsingleton.elim r 0, Subsingleton.elim c' 0]
      simp [Matrix.mul_apply, matrixPolynomial]
    intro j
    funext r c'
    rw [Subsingleton.elim r 0]
    fin_cases j
    · show (matrixAdd (matrixSub (matrixAdd (matrixMulScalarLeft secret (Stage_encrypt.generatedRoot.w_7_0 ctx))
        (matrixMulScalarLeft 1 (Stage_encrypt.generatedRoot.w_9_0 ctx)))
        (matrixMulScalarRight (matrixMulScalarLeft secret (Stage_encrypt.generatedRoot.w_13_0 ctx))
          (Stage_encrypt.generatedRoot.w_20_0 ctx))) (Stage_encrypt.generatedRoot.w_23_0 ctx)) 0 c' = _
      rw [e7, e9, e13, e20, he23, hsecret]
      simp only [matrixAdd, matrixSub, matrixMulScalarLeft_eq, matrixMulScalarRight_eq, Matrix.add_apply,
        Matrix.sub_apply, Matrix.smul_apply, smul_eq_mul, Matrix.one_apply_eq, one_mul, unitPair, stateErr,
        Fin.zero_eta, Fin.isValue, Matrix.cons_val_zero, map_mul, Nat.cast_one, Int.cast_one]
      ring
    · show (matrixAdd (matrixSub (matrixAdd (matrixMulScalarLeft secret (Stage_encrypt.generatedRoot.w_25_0 ctx))
        (matrixMulScalarLeft 1 (Stage_encrypt.generatedRoot.w_27_0 ctx)))
        (matrixMulScalarRight (matrixMulScalarLeft 1 (Stage_encrypt.generatedRoot.w_13_0 ctx))
          (Stage_encrypt.generatedRoot.w_20_0 ctx))) (Stage_encrypt.generatedRoot.w_33_0 ctx)) 0 c' = _
      rw [e25, e27, e13, e20, he33, hsecret]
      simp only [matrixAdd, matrixSub, matrixMulScalarLeft_eq, matrixMulScalarRight_eq, Matrix.add_apply,
        Matrix.sub_apply, Matrix.smul_apply, smul_eq_mul, Matrix.one_apply_eq, one_mul, unitPair, stateErr,
        Fin.mk_one, Fin.isValue, Matrix.cons_val_one, map_mul, Nat.cast_one, Int.cast_one]
      simp only [Matrix.cons_val_zero, map_one, mul_one]
  · obtain ⟨-, hb23⟩ := gaussianSampleAt_explicit hN F.c0.h23_0
    obtain ⟨-, hb33⟩ := gaussianSampleAt_explicit hN F.c0.h33_0
    fin_cases j
    · exact hb23
    · exact hb33

end PeProof
