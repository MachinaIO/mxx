import PePrelude
import Stage_encrypt
import Stage_encrypt_Facts

/-!
# Encryption

Under the secret `s`, the encryption publishes the helpers `h_k = s H_k^0 + H_k^1 - (σ_k(s), 1) g + e`
of the `37` helper automorphisms, the compact digit companions of the switching ciphertexts and of
the attribute ciphertexts, and the initial state encoding.
-/

namespace PeProof

open Mxx.Primitives MxxRuntime MxxExponentLut

/-- The law of the encryption samples. -/
abbrev encLaw : SamplerLaw := .gaussian (319 / 100) 31

/-- The helper automorphisms: the first-depth switches `kSw b` (members `b < 22`) and the
inverses of the internal switches (members `22 ≤ i < 37`). -/
abbrev helperList : List Int := [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689), (115713), (116737), (117761), (118785), (119809), (120833), (121857), (122881), (123905), (124929), (125953), (126977), (128001), (129025), (130049)]

/-- The automorphism of helper member `i`. -/
def kHelp (i : Nat) : Nat := (MxxIR.selectInt (Int.ofNat i) helperList).toNat

theorem kHelp_odd (i : Fin 37) : Odd (kHelp i) := by
  fin_cases i <;> decide

theorem kHelp_cast (i : Nat) (hi : i < 37) :
    ((kHelp i : Nat) : Int) = MxxIR.selectInt (Int.ofNat i) helperList := by
  interval_cases i <;> rfl

theorem kHelp_sw (b : Nat) (hb : b < 22) : kHelp b = kSw b := by
  interval_cases b <;> rfl

theorem kHelp_inv (b : Nat) (hb : b < 22) : kHelp (36 - b) = kInv b := by
  interval_cases b <;> rfl

/-- The error of component `j` of the helper of member `i`. -/
noncomputable def helperErr (tape : SampleTape) (path : List Nat) (i : Nat) (j : Fin 2) : ErrorMatrix N 1 44 :=
  tapeWitness tape (path ++ [4, i] ++ [![12, 20] j]) encLaw

/-- The routed secret. -/
theorem routed_secret {k : Int} {sm : ErrorPoly N} {y : ExactMatrix Q N 1 1}
    (h : ringAutomorphismRuns k (fun _ _ ↦ reducePoly Q N sm) y) :
    ∃ hodd : Odd k.toNat, y = fun _ _ ↦ reducePoly Q N (substitution k.toNat hodd sm) := by
  obtain ⟨hodd, hy⟩ := ringAutomorphismRuns_eq hQ hN h
  exact ⟨hodd, hy.trans (by funext a b; rw [reducePoly_substitution])⟩

set_option maxHeartbeats 8000000 in
/-- The helper of member `i` encodes `(σ_k(s), 1) g` under `s` and the helper key `H_k`. -/
theorem encrypt_helper {model : HashModel} {tape : SampleTape} {path : List Nat} {i : Nat} (hi : i < 37)
    {key : ByteArray} {sm : ErrorPoly N} {outputs : ExactMatrix Q N 1 44 × ExactMatrix Q N 1 44 × Unit}
    (h : Stage_encrypt.parallel_generatedRoot_4 Backend.backend model tape (path ++ [4, i])
      GeneratedClaim.stage_4_params i ((fun _ _ ↦ reducePoly Q N sm), (key, (1, (g0, ())))) outputs) :
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

set_option maxHeartbeats 8000000 in
/-- The switching ciphertext `C_k` of first-depth branch `b`: the routed helper key of `k⁻¹`
(top row) and the routed helper key minus the routed helper of `k⁻¹` (bottom row). -/
theorem encrypt_switch {model : HashModel} {tape : SampleTape} {path : List Nat} {b : Nat} (hb : b < 22)
    {key : ByteArray} {hf0 hf1 : Fin 37 → ExactMatrix Q N 1 44}
    {outputs : ExactMatrix Q N 1 44 × ExactMatrix Q N 1 44 × ExactMatrix Q N 1 44 ×
      ExactMatrix Q N 1 44 × Unit}
    (h : Stage_encrypt.parallel_generatedRoot_5 Backend.backend model tape (path ++ [5, b])
      GeneratedClaim.stage_4_params b (key, (hf0, (hf1, ()))) outputs) :
    outputs.1 = substituteMatrix (kSw b) (kSw_odd ⟨b, hb⟩) (helperKey model key (kInv b) 0 0) ∧
    outputs.2.1 = substituteMatrix (kSw b) (kSw_odd ⟨b, hb⟩) (helperKey model key (kInv b) 0 1) ∧
    outputs.2.2.1 = substituteMatrix (kSw b) (kSw_odd ⟨b, hb⟩) (helperKey model key (kInv b) 1 0) -
      substituteMatrix (kSw b) (kSw_odd ⟨b, hb⟩) (hf0 ⟨36 - b, by omega⟩) ∧
    outputs.2.2.2.1 = substituteMatrix (kSw b) (kSw_odd ⟨b, hb⟩) (helperKey model key (kInv b) 1 1) -
      substituteMatrix (kSw b) (kSw_odd ⟨b, hb⟩) (hf1 ⟨36 - b, by omega⟩) := by
  obtain ⟨w1, w2, w3, w4, w5, w6, w9, w10, w12, w13, w15, w16, ⟨-, -, h1, -, -, h2, -, -, h3, -, -, h4, -, -, h5, -, -, h6, -, -, -, -, h9, -, -, h10, -, -, h12, -, -, h13⟩, -, -, h15, -, -, h16, hout⟩ := h
  have hk := kInv_cast b hb
  rw [← hk] at h1 h3 h5 h12
  have e1 : w1 = helperKey model key (kInv b) 0 0 := helperKey_read (c := 0) (by norm_num) h1
  have e3 : w3 = helperKey model key (kInv b) 0 1 := helperKey_read (c := 1) (by norm_num) h3
  have e5 : w5 = helperKey model key (kInv b) 1 0 := helperKey_read (c := 2) (by norm_num) h5
  have e12 : w12 = helperKey model key (kInv b) 1 1 := helperKey_read (c := 3) (by norm_num) h12
  have hm := switchInvMembers_eq b hb
  rw [hm] at h9 h15
  obtain ⟨_, e9⟩ := familyGet_eq h9
  obtain ⟨_, e15⟩ := familyGet_eq h15
  rw [routing_kSw hb h2, routing_kSw hb h4, routing_kSw hb h6, routing_kSw hb h10, routing_kSw hb h13,
    routing_kSw hb h16] at hout
  subst hout e1 e3 e5 e12 e9 e15
  exact ⟨rfl, rfl, rfl, rfl⟩

/-! ## The compact digit companions of the switches -/

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
noncomputable def entryDigit (y : ExactPoly Q N) (a : Fin 44) : ErrorPoly N :=
  centeredIntLift ((entryDigits 44 Backend.backend 65536 44 (fun _ _ ↦ y) : ExactMatrix Q N 44 1) a 0)

theorem entryDigit_spec (y : ExactPoly Q N) (a : Fin 44) :
    reducePoly Q N (entryDigit y a) =
        (entryDigits 44 Backend.backend 65536 44 (fun _ _ ↦ y) : ExactMatrix Q N 44 1) a 0 ∧
      ∀ k, ((entryDigit y a).coeff k).natAbs ≤ 32768 := by
  obtain ⟨w, hw, hb⟩ := (decompose_entry hQ hN Backend.layout0 layoutQ rfl g0_runs
    (entryDigits_runs hQ hN g0_runs Backend.layout0 layoutQ rfl (fun _ _ ↦ y))).2
  have he : (entryDigits 44 Backend.backend 65536 44 (fun _ _ ↦ y) : ExactMatrix Q N 44 1) a 0 =
      reducePoly Q N (w a 0) := by rw [hw]; rfl
  have hlift : entryDigit y a = w a 0 := by
    rw [entryDigit, he]
    exact centeredIntLift_reducePoly hQ hN _ fun i ↦ by
      have h1 := hb a 0 i
      have h2 : Backend.layout0.base / 2 = 32768 := rfl
      rw [h2] at h1
      have : (32768 : Nat) * 2 < Q := by norm_num
      omega
  refine ⟨by rw [hlift, he], fun k ↦ ?_⟩
  rw [hlift]
  exact hb a 0 k

/-- The error of component `j` of LUT companion `i`. -/
noncomputable def compErr (tape : SampleTape) (path : List Nat) (i : Nat) (j : Fin 2) : ErrorMatrix N 1 1 :=
  tapeWitness tape (path ++ [7, i] ++ [![35, 45] j]) encLaw

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

set_option maxHeartbeats 32000000 in
/-- LUT companion `i` (branch `b = i / 3872`, digit row `44 + a` for `a = (i / 88) % 44`, column
`jj = i % 88`) is a compact encoding over `R_{PQ}` under the routed mask `σ_k(s)` of
`P d (σ_k(s), 1)`, for the digit `d` of entry `jj` of the bottom row of the switching ciphertext. -/
theorem encrypt_companion {model : HashModel} {tape : SampleTape} {path : List Nat} {i : Nat}
    (hi : i < 85184) {key : ByteArray} {sm : ErrorPoly N} (hs : polyNorm sm ≤ 1)
    {T0 T1 B0 B1 : Fin 22 → ExactMatrix Q N 1 44} {outputs : ExactMatrix PQ N 1 1 × ExactMatrix PQ N 1 1 × Unit}
    (h : Stage_encrypt.parallel_generatedRoot_7 Backend.backend model tape (path ++ [7, i])
      GeneratedClaim.stage_4_params i ((fun _ _ ↦ reducePoly Q N sm), (key, (1, (T0,
        (((i / 3872 : Nat) : Int), (T1, (B0, (B1, ())))))))) outputs) :
    RelE (substitution (kSw (i / 3872)) (kSw_odd ⟨i / 3872, by omega⟩) sm)
      (lutCompKey model key (kSw (i / 3872)) ((i % 88 : Nat) : Int) ((44 + (i / 88) % 44 : Nat) : Int))
      (fun j _ _ ↦ reducePoly PQ N (((P : Nat) : Int) *
        (entryDigit (joinedEntry ![B0 ⟨i / 3872, by omega⟩, B1 ⟨i / 3872, by omega⟩] ⟨i % 88, by omega⟩)
          ⟨(i / 88) % 44, by omega⟩ *
          unitPair (substitution (kSw (i / 3872)) (kSw_odd ⟨i / 3872, by omega⟩) sm) j)))
      ![outputs.1, outputs.2.1] (compErr tape path i) ∧
    ∀ j, CoeffBound (compErr tape path i j) 31 := by
  obtain ⟨w1, w2, w4, w7, w8, w13, w16, w18, w19, w20, w22, w24, w25, w26, w27, w28, w29, w31, w35, w37, w39, w45, ⟨-, -, -, h1, h2, -, -, -, -, -, -, h4, h7, -, -, -, -, -, -, h8, -, -, -, -, -, -, -, -, -, -, -, -⟩, ⟨-, -, h13, -, -, h16, -, -, h18, h19, -, -, -, -, -, -, -, -, h20, -, -, h22, -, -, h24, h25, -, -, -, -, -, -⟩, ⟨-, -, h26, h27, -, -, -, -, -, -, -, -, -, -, -, -, h28, h29, h31, h35, -, -, -, -, -, -, h37, -, -, -, -, -⟩, -, h39, h45, hout⟩ := h
  have hb : i / 3872 < 22 := by omega
  have hfb : Int.fdiv (Int.ofNat i) 3872 = Int.ofNat (i / 3872) := by
    rw [Int.fdiv_eq_ediv_of_nonneg _ (by norm_num)]; simp only [Int.ofNat_eq_natCast]; omega
  have hjj : Int.fmod (Int.ofNat i) 88 = ((i % 88 : Nat) : Int) := by
    rw [Int.fmod_eq_emod_of_nonneg _ (by norm_num)]; simp only [Int.ofNat_eq_natCast]; omega
  have ha : Int.fmod (Int.fdiv (Int.ofNat i) 88) 44 = (((i / 88) % 44 : Nat) : Int) := by
    rw [Int.fdiv_eq_ediv_of_nonneg _ (by norm_num), Int.fmod_eq_emod_of_nonneg _ (by omega)]
    simp only [Int.ofNat_eq_natCast]; omega
  have hrow : (44 : Int) + Int.fmod (Int.fdiv (Int.ofNat i) 88) 44 = ((44 + (i / 88) % 44 : Nat) : Int) := by
    rw [ha]; push_cast; ring
  have hone : Int.fdiv (44 + Int.fmod (Int.fdiv (Int.ofNat i) 88) 44) 44 = 1 := by
    rw [ha, Int.fdiv_eq_ediv_of_nonneg _ (by omega)]; omega
  rw [hfb] at h1 h4 h8 h37 h39
  rw [hjj, hrow] at h4 h8 h37 h39
  have hk := kSw_cast (i / 3872) hb
  rw [← hk] at h4 h8 h37 h39
  -- The mask.
  have e1 := routing_kSw hb h1
  set σs := substitution (kSw (i / 3872)) (kSw_odd ⟨i / 3872, hb⟩) sm with hσs
  have e1' : w1 = fun _ _ ↦ reducePoly Q N σs := by
    rw [e1]; funext a c
    simp only [substituteMatrix]
    exact (reducePoly_substitution _ _ _).symm
  rw [e1'] at h2
  have hσn : polyNorm σs ≤ 1 :=
    (polyNorm_substitution_le hN _ (kSw_coprime ⟨i / 3872, hb⟩) sm).trans hs
  have e2 := rebase_small (q := Q) (by norm_num) hσn h2
  have e7 := rebase_one (q := Q) (by norm_num) h7
  -- The keys.
  have e4 : w4 = lutCompKey model key (kSw (i / 3872)) ((i % 88 : Nat) : Int) ((44 + (i / 88) % 44 : Nat) : Int) 0 0 :=
    hashSample_eq h4
  have e8 : w8 = lutCompKey model key (kSw (i / 3872)) ((i % 88 : Nat) : Int) ((44 + (i / 88) % 44 : Nat) : Int) 1 0 :=
    hashSample_eq h8
  have e37 : w37 = lutCompKey model key (kSw (i / 3872)) ((i % 88 : Nat) : Int) ((44 + (i / 88) % 44 : Nat) : Int) 0 1 :=
    hashSample_eq h37
  have e39 : w39 = lutCompKey model key (kSw (i / 3872)) ((i % 88 : Nat) : Int) ((44 + (i / 88) % 44 : Nat) : Int) 1 1 :=
    hashSample_eq h39
  -- The digit.
  obtain ⟨_, e22⟩ := familyGet_eq h22
  obtain ⟨_, e24⟩ := familyGet_eq h24
  rw [hjj] at h26
  obtain ⟨_, e26⟩ := sliceMatrix_column h26
  rw [hone] at h28
  obtain ⟨_, e28⟩ := sliceMatrix_column h28
  rw [concatColumns_entry h27] at e28
  have e28' : w28 0 0 = w26 0 0 := by
    rw [e28]
    exact joinedEntry_pair (m := 1) ![w20, w26] 1 0 (by omega)
  rw [concatColumns_entry h25, e22, e24] at e26
  have hy : w28 = fun _ _ ↦ joinedEntry ![B0 ⟨i / 3872, hb⟩, B1 ⟨i / 3872, hb⟩] ⟨i % 88, by omega⟩ := by
    funext r c
    rw [Subsingleton.elim r 0, Subsingleton.elim c 0, e28', e26]
    rfl
  have e29 := entryDigits_eq h29
  rw [hy] at e29
  set y := joinedEntry ![B0 ⟨i / 3872, hb⟩, B1 ⟨i / 3872, hb⟩] ⟨i % 88, by omega⟩ with hydef
  set a : Fin 44 := ⟨(i / 88) % 44, by omega⟩ with hadef
  have hrow2 : Int.fmod (44 + Int.fmod (Int.fdiv (Int.ofNat i) 88) 44) 44 = (((i / 88) % 44 : Nat) : Int) := by
    rw [ha, Int.fmod_eq_emod_of_nonneg _ (by omega)]; omega
  rw [hrow2] at h13
  have e13 := unitRow_window (c := (((i / 88) % 44 : Nat) : Int)) (by ring) h13
  have e30 : (matrixMul w13 w29 : ExactMatrix Q N 1 1) = fun _ _ ↦ reducePoly Q N (entryDigit y a) := by
    funext r c
    rw [Subsingleton.elim r 0, Subsingleton.elim c 0, matrixMul_eq_mul, e13, e29,
      unitRow_mul_entry _ (by omega), (entryDigit_spec y a).1]
  have hδ : ∀ k, 2 * ((entryDigit y a).coeff k).natAbs < Q := fun k ↦ by
    have := (entryDigit_spec y a).2 k
    have : (32768 : Nat) * 2 < Q := by norm_num
    omega
  rw [e30] at h31
  have e31 := rebase_lift hQ hδ h31
  obtain ⟨he35, hb35⟩ := gaussianSampleAt_explicit hN h35
  obtain ⟨he45, hb45⟩ := gaussianSampleAt_explicit hN h45
  refine ⟨fun j ↦ ?_, fun j ↦ ?_⟩
  · subst hout
    funext r c
    rw [Subsingleton.elim r 0, Subsingleton.elim c 0]
    fin_cases j
    · simp only [Fin.zero_eta, Fin.isValue, Matrix.cons_val_zero, matrixAdd, matrixSub,
        matrixMulScalarLeft_eq, e2, e4, e7, e8, e31, he35, compErr, Matrix.add_apply, Matrix.sub_apply,
        Matrix.smul_apply, smul_eq_mul, unitPair, Matrix.one_apply_eq, one_mul, map_mul]
      simp [matrixPolynomial, map_ofNat]
    · simp only [Fin.mk_one, Fin.isValue, Matrix.cons_val_one, matrixAdd, matrixSub,
        matrixMulScalarLeft_eq, e2, e7, e37, e39, e31, he45, compErr, Matrix.add_apply, Matrix.sub_apply,
        Matrix.smul_apply, smul_eq_mul, unitPair, Matrix.one_apply_eq, map_mul]
      simp [matrixPolynomial, map_ofNat]
  · fin_cases j
    · exact hb35
    · exact hb45

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
  tapeWitness tape (path ++ [40, i] ++ [![32, 42] j]) encLaw

set_option maxHeartbeats 32000000 in
/-- Product companion `i` (position `x = i / 3872`, digit row `44 + a` for `a = (i / 88) % 44`,
column `jj = i % 88`) is a compact encoding over `R_{PQ}` under `s` of `P d (s, 1)`, for the digit
`d` of entry `jj` of the bottom row `Y` of the attribute ciphertext of position `x`. -/
theorem encrypt_rhsCompanion {model : HashModel} {tape : SampleTape} {path : List Nat} {i : Nat}
    (_hi : i < 15488) {key : ByteArray} {sm : ErrorPoly N} (hs : polyNorm sm ≤ 1)
    {R0 R1 : Fin 4 → ExactMatrix Q N 1 44} {Y0 Y1 : ExactMatrix Q N 1 44}
    {outputs : ExactMatrix PQ N 1 1 × ExactMatrix PQ N 1 1 × Unit}
    (h : Stage_encrypt.parallel_generatedRoot_40 Backend.backend model tape (path ++ [40, i])
      GeneratedClaim.stage_4_params i ((fun _ _ ↦ reducePoly Q N sm), (key, (1, (R0,
        (((i / 3872 : Nat) : Int), (R1, (Y0, (Y1, ())))))))) outputs) :
    RelE sm (rhsCompKey model key ((i / 3872 : Nat) : Int) ((i % 88 : Nat) : Int) ((44 + (i / 88) % 44 : Nat) : Int))
      (fun j _ _ ↦ reducePoly PQ N (((P : Nat) : Int) *
        (entryDigit (joinedEntry ![Y0, Y1] ⟨i % 88, by omega⟩) ⟨(i / 88) % 44, by omega⟩ * unitPair sm j)))
      ![outputs.1, outputs.2.1] (rhsCompErr tape path i) ∧
    ∀ j, CoeffBound (rhsCompErr tape path i j) 31 := by
  obtain ⟨w1, w3, w6, w7, w12, w15, w17, w18, w19, w22, w23, w24, w25, w26, w28, w32, w34, w36, w42, ⟨h1, -, -, -, -, h3, h6, -, -, -, -, h7, -, -, -, -, -, -, -, -, -, -, -, -, -, -, h12, -, -, h15, -, -⟩, ⟨h17, h18, -, -, -, -, -, -, -, -, h19, h22, -, -, -, -, -, -, -, -, h23, h24, -, -, -, -, -, -, -, -, -, -⟩, -, -, h25, h26, h28, h32, -, -, -, -, h34, -, -, -, -, h36, h42, hout⟩ := h
  have hfb : Int.fdiv (Int.ofNat i) 3872 = ((i / 3872 : Nat) : Int) := by
    rw [Int.fdiv_eq_ediv_of_nonneg _ (by norm_num)]; simp only [Int.ofNat_eq_natCast]; omega
  have hjj : Int.fmod (Int.ofNat i) 88 = ((i % 88 : Nat) : Int) := by
    rw [Int.fmod_eq_emod_of_nonneg _ (by norm_num)]; simp only [Int.ofNat_eq_natCast]; omega
  have ha : Int.fmod (Int.fdiv (Int.ofNat i) 88) 44 = (((i / 88) % 44 : Nat) : Int) := by
    rw [Int.fdiv_eq_ediv_of_nonneg _ (by norm_num), Int.fmod_eq_emod_of_nonneg _ (by omega)]
    simp only [Int.ofNat_eq_natCast]; omega
  have hrow : (44 : Int) + Int.fmod (Int.fdiv (Int.ofNat i) 88) 44 = ((44 + (i / 88) % 44 : Nat) : Int) := by
    rw [ha]; push_cast; ring
  have hone : Int.fdiv (44 + Int.fmod (Int.fdiv (Int.ofNat i) 88) 44) 44 = 1 := by
    rw [ha, Int.fdiv_eq_ediv_of_nonneg _ (by omega)]; omega
  rw [hfb, hjj, hrow] at h3 h7 h34 h36
  -- The mask.
  have e1 := rebase_small (q := Q) (by norm_num) hs h1
  have e6 := rebase_one (q := Q) (by norm_num) h6
  -- The keys.
  have e3 : w3 = rhsCompKey model key ((i / 3872 : Nat) : Int) ((i % 88 : Nat) : Int) ((44 + (i / 88) % 44 : Nat) : Int) 0 0 :=
    hashSample_eq h3
  have e7 : w7 = rhsCompKey model key ((i / 3872 : Nat) : Int) ((i % 88 : Nat) : Int) ((44 + (i / 88) % 44 : Nat) : Int) 1 0 :=
    hashSample_eq h7
  have e34 : w34 = rhsCompKey model key ((i / 3872 : Nat) : Int) ((i % 88 : Nat) : Int) ((44 + (i / 88) % 44 : Nat) : Int) 0 1 :=
    hashSample_eq h34
  have e36 : w36 = rhsCompKey model key ((i / 3872 : Nat) : Int) ((i % 88 : Nat) : Int) ((44 + (i / 88) % 44 : Nat) : Int) 1 1 :=
    hashSample_eq h36
  -- The digit.
  rw [hjj] at h23
  obtain ⟨_, e23⟩ := sliceMatrix_column h23
  rw [hone] at h25
  obtain ⟨_, e25⟩ := sliceMatrix_column h25
  rw [concatColumns_entry h24] at e25
  have e25' : w25 0 0 = w23 0 0 := by
    rw [e25]
    exact joinedEntry_pair (m := 1) ![w19, w23] 1 0 (by omega)
  rw [concatColumns_entry h22] at e23
  have hy : w25 = fun _ _ ↦ joinedEntry ![Y0, Y1] ⟨i % 88, by omega⟩ := by
    funext r c
    rw [Subsingleton.elim r 0, Subsingleton.elim c 0, e25', e23]
    rfl
  have e26 := entryDigits_eq h26
  rw [hy] at e26
  set y := joinedEntry ![Y0, Y1] ⟨i % 88, by omega⟩ with hydef
  set a : Fin 44 := ⟨(i / 88) % 44, by omega⟩ with hadef
  have hrow2 : Int.fmod (44 + Int.fmod (Int.fdiv (Int.ofNat i) 88) 44) 44 = (((i / 88) % 44 : Nat) : Int) := by
    rw [ha, Int.fmod_eq_emod_of_nonneg _ (by omega)]; omega
  rw [hrow2] at h12
  have e12 := unitRow_window (c := (((i / 88) % 44 : Nat) : Int)) (by ring) h12
  have e27 : (matrixMul w12 w26 : ExactMatrix Q N 1 1) = fun _ _ ↦ reducePoly Q N (entryDigit y a) := by
    funext r c
    rw [Subsingleton.elim r 0, Subsingleton.elim c 0, matrixMul_eq_mul, e12, e26,
      unitRow_mul_entry _ (by omega), (entryDigit_spec y a).1]
  have hδ : ∀ k, 2 * ((entryDigit y a).coeff k).natAbs < Q := fun k ↦ by
    have := (entryDigit_spec y a).2 k
    have : (32768 : Nat) * 2 < Q := by norm_num
    omega
  rw [e27] at h28
  have e28 := rebase_lift hQ hδ h28
  obtain ⟨he32, hb32⟩ := gaussianSampleAt_explicit hN h32
  obtain ⟨he42, hb42⟩ := gaussianSampleAt_explicit hN h42
  refine ⟨fun j ↦ ?_, fun j ↦ ?_⟩
  · subst hout
    funext r c
    rw [Subsingleton.elim r 0, Subsingleton.elim c 0]
    fin_cases j
    · simp only [Fin.zero_eta, Fin.isValue, Matrix.cons_val_zero, matrixAdd, matrixSub,
        matrixMulScalarLeft_eq, e1, e3, e6, e7, e28, he32, rhsCompErr, Matrix.add_apply, Matrix.sub_apply,
        Matrix.smul_apply, smul_eq_mul, unitPair, Matrix.one_apply_eq, one_mul, map_mul]
      simp [matrixPolynomial, map_ofNat]
    · simp only [Fin.mk_one, Fin.isValue, Matrix.cons_val_one, matrixAdd, matrixSub,
        matrixMulScalarLeft_eq, e1, e6, e34, e36, e28, he42, rhsCompErr, Matrix.add_apply, Matrix.sub_apply,
        Matrix.smul_apply, smul_eq_mul, unitPair, Matrix.one_apply_eq, map_mul]
      simp [matrixPolynomial, map_ofNat]
  · fin_cases j
    · exact hb32
    · exact hb42

/-! ## The encryption -/

/-- `input-key` -/
def inputKeyBytes : Blob := [105, 110, 112, 117, 116, 45, 107, 101, 121]

/-- The hashed key `A` of the initial state encoding. -/
noncomputable def inputKey (model : HashModel) (key : ByteArray) : Fin 2 → Fin 2 → ExactMatrix Q N 1 44 :=
  fun r j ↦ model.sample Q N 1 44 key (completeHashTag dynamicPrefix [.bytes schemaBytes,
    .bytes inputKeyBytes, .u64Le 0, .bytes componentBytes, .u64Le (2 * (r : Int) + (j : Int))])

/-- Component `j` of the bottom row of the switching ciphertext of first-depth branch `b`. -/
noncomputable def switchBot (model : HashModel) (key : ByteArray) (hf0 hf1 : Fin 37 → ExactMatrix Q N 1 44)
    (b : Fin 22) (j : Fin 2) : ExactMatrix Q N 1 44 :=
  substituteMatrix (kSw b) (kSw_odd b) (helperKey model key (kInv b) 1 j) -
    substituteMatrix (kSw b) (kSw_odd b) (![hf0, hf1] j ⟨36 - b, by omega⟩)

/-- The error of component `j` of the initial state encoding. -/
noncomputable def stateErr (tape : SampleTape) (path : List Nat) (j : Fin 2) : ErrorMatrix N 1 44 :=
  tapeWitness tape (path ++ [![24, 34] j]) encLaw

abbrev EncryptOutputs := (Fin 85184 → ExactMatrix PQ N 1 1) × (Fin 85184 → ExactMatrix PQ N 1 1) ×
  (Fin 37 → ExactMatrix Q N 1 44) × (Fin 37 → ExactMatrix Q N 1 44) × ExactMatrix Q N 1 44 ×
  ExactMatrix Q N 1 44 × (Fin 15488 → ExactMatrix PQ N 1 1) × (Fin 15488 → ExactMatrix PQ N 1 1) × Unit

set_option maxHeartbeats 32000000 in
/-- The encryption: the helpers, the compact companions of the switches and of the attribute
ciphertexts, and the initial state encoding of `c X^e`. -/
theorem encrypt_explicit {model : HashModel} {tape : SampleTape} {path : List Nat}
    {secret : ExactMatrix Q N 1 1} {key : ByteArray} {c e : Int} {Y0 Y1 : Fin 4 → ExactMatrix Q N 1 44}
    {outputs : EncryptOutputs}
    (h : Stage_encrypt.generatedRoot Backend.backend model tape path GeneratedClaim.stage_4_params
      (secret, key, c, e, Y0, Y1, ()) outputs)
    {sm : ErrorPoly N} (hsecret : secret = fun _ _ ↦ reducePoly Q N sm) (hs : polyNorm sm ≤ 1) :
    (∀ i : Fin 37, Rel sm (helperKey model key (kHelp i))
      (fun j ↦ reducePoly Q N ((((1 : Nat) : Int) : ErrorPoly N) *
        unitPair (substitution (kHelp i) (kHelp_odd i) sm) j) • g0)
      ![outputs.2.2.1 i, outputs.2.2.2.1 i] 31) ∧
    (∀ i : Fin 85184,
      RelE (substitution (kSw (i / 3872)) (kSw_odd ⟨i / 3872, by omega⟩) sm)
        (lutCompKey model key (kSw (i / 3872)) ((i % 88 : Nat) : Int) ((44 + (i / 88) % 44 : Nat) : Int))
        (fun j _ _ ↦ reducePoly PQ N (((P : Nat) : Int) *
          (entryDigit (joinedEntry ![switchBot model key outputs.2.2.1 outputs.2.2.2.1 ⟨i / 3872, by omega⟩ 0,
              switchBot model key outputs.2.2.1 outputs.2.2.2.1 ⟨i / 3872, by omega⟩ 1] ⟨i % 88, by omega⟩)
            ⟨(i / 88) % 44, by omega⟩ *
            unitPair (substitution (kSw (i / 3872)) (kSw_odd ⟨i / 3872, by omega⟩) sm) j)))
        ![outputs.1 i, outputs.2.1 i] (compErr tape path i) ∧
      ∀ j, CoeffBound (compErr tape path i j) 31) ∧
    (∀ i : Fin 15488,
      RelE sm (rhsCompKey model key ((i / 3872 : Nat) : Int) ((i % 88 : Nat) : Int) ((44 + (i / 88) % 44 : Nat) : Int))
        (fun j _ _ ↦ reducePoly PQ N (((P : Nat) : Int) *
          (entryDigit (joinedEntry ![Y0 ⟨i / 3872, by omega⟩, Y1 ⟨i / 3872, by omega⟩] ⟨i % 88, by omega⟩)
            ⟨(i / 88) % 44, by omega⟩ * unitPair sm j)))
        ![outputs.2.2.2.2.2.2.1 i, outputs.2.2.2.2.2.2.2.1 i] (rhsCompErr tape path i) ∧
      ∀ j, CoeffBound (rhsCompErr tape path i j) 31) ∧
    RelE sm (inputKey model key)
      (fun j ↦ reducePoly Q N ((((1 : Nat) : Int) : ErrorPoly N) * message c e * unitPair sm j) • g0)
      ![outputs.2.2.2.2.1, outputs.2.2.2.2.2.1] (stateErr tape path) ∧
    ∀ j, CoeffBound (stateErr tape path j) 31 := by
  obtain ⟨w, hbody⟩ := h
  let ctx : Stage_encrypt.generatedRoot.Ctx Backend.backend model tape path GeneratedClaim.stage_4_params :=
    ⟨(secret, key, c, e, Y0, Y1, ()), w⟩
  have F := Stage_encrypt.generatedRoot.facts ctx hbody
  have hg3 : Stage_encrypt.generatedRoot.w_3_0 ctx = g0 := gadgetMatrixRuns_unique F.c0.h3_0 g0_runs
  have hw0 : Stage_encrypt.generatedRoot.w_0_0 ctx = fun _ _ ↦ reducePoly Q N sm := hsecret
  -- The helpers.
  have hhelp : ∀ i : Fin 37, Rel sm (helperKey model key (kHelp i))
      (fun j ↦ reducePoly Q N ((((1 : Nat) : Int) : ErrorPoly N) *
        unitPair (substitution (kHelp i) (kHelp_odd i) sm) j) • g0)
      ![Stage_encrypt.generatedRoot.w_4_0 ctx i, Stage_encrypt.generatedRoot.w_4_1 ctx i] 31 := by
    intro i
    have hi := F.c0.h4_1 i
    rw [hw0, hg3] at hi
    exact encrypt_helper i.isLt hi
  -- The switches.
  have hsw : ∀ (b : Fin 22) (j : Fin 2),
      ![Stage_encrypt.generatedRoot.w_5_2 ctx b, Stage_encrypt.generatedRoot.w_5_3 ctx b] j =
        switchBot model key (Stage_encrypt.generatedRoot.w_4_0 ctx) (Stage_encrypt.generatedRoot.w_4_1 ctx) b j := by
    intro b j
    obtain ⟨-, -, e2, e3⟩ := encrypt_switch b.isLt (F.c0.h5_3 b)
    fin_cases j
    · exact e2
    · exact e3
  rw [F.c0.hout]
  refine ⟨hhelp, fun i ↦ ?_, fun i ↦ ?_, ?_, fun j ↦ ?_⟩
  · have h6 := F.c0.h6_0 i
    obtain ⟨-, e6⟩ := h6
    have hi7 := F.c0.h7_1 i
    have hidx : Stage_encrypt.generatedRoot.w_6_0 ctx i = ((i / 3872 : Nat) : Int) := by
      rw [e6, add_zero, Int.fdiv_eq_ediv_of_nonneg _ (by norm_num)]
      simp only [Int.ofNat_eq_natCast]; omega
    rw [hidx, hw0] at hi7
    have := encrypt_companion i.isLt hs hi7
    have e0 := hsw ⟨i / 3872, by omega⟩ 0
    have e1 := hsw ⟨i / 3872, by omega⟩ 1
    simp only [Matrix.cons_val_zero, Matrix.cons_val_one] at e0 e1
    rw [e0, e1] at this
    exact this
  · obtain ⟨y0, y1, -, -, hy0, -, -, hy1, hi40⟩ := F.c0.h40_1 i
    obtain ⟨-, e37⟩ := F.c0.h37_0 i
    have hidx : Stage_encrypt.generatedRoot.w_37_0 ctx i = ((i / 3872 : Nat) : Int) := by
      rw [e37, add_zero, Int.fdiv_eq_ediv_of_nonneg _ (by norm_num)]
      simp only [Int.ofNat_eq_natCast]; omega
    rw [hidx] at hy0 hy1 hi40
    obtain ⟨_, ey0⟩ := familyGet_eq hy0
    obtain ⟨_, ey1⟩ := familyGet_eq hy1
    rw [hw0, ey0, ey1] at hi40
    exact encrypt_rhsCompanion i.isLt hs hi40
  · have e8 : Stage_encrypt.generatedRoot.w_8_0 ctx = inputKey model key 0 0 := hashSample_eq F.c0.h8_0
    have e10 : Stage_encrypt.generatedRoot.w_10_0 ctx = inputKey model key 1 0 := hashSample_eq F.c0.h10_0
    have e26 : Stage_encrypt.generatedRoot.w_26_0 ctx = inputKey model key 0 1 := hashSample_eq F.c0.h26_0
    have e28 : Stage_encrypt.generatedRoot.w_28_0 ctx = inputKey model key 1 1 := hashSample_eq F.c0.h28_0
    have hg13 := gadgetMatrixRuns_unique F.c0.h13_0 g0_runs
    have h14 := F.c0.h14_0
    rw [hg13] at h14
    have e14 : Stage_encrypt.generatedRoot.w_14_0 ctx = g0 := centeredRebaseRuns_unique h14 hg0
    obtain ⟨he24, -⟩ := gaussianSampleAt_explicit hN F.c0.h24_0
    obtain ⟨he34, -⟩ := gaussianSampleAt_explicit hN F.c0.h34_0
    have e21 : Stage_encrypt.generatedRoot.w_21_0 ctx = fun _ _ ↦ reducePoly Q N (message c e) := by
      show matrixMulScalarLeft (matrixPolynomial [1]) (multiplyMonomial (liftInteger c) e) = _
      rw [multiplyMonomial_liftInteger]
      funext r c'
      rw [Subsingleton.elim r 0, Subsingleton.elim c' 0]
      simp [Matrix.mul_apply, matrixPolynomial]
    intro j
    funext r c'
    rw [Subsingleton.elim r 0]
    fin_cases j
    · show (matrixAdd (matrixSub (matrixAdd (matrixMulScalarLeft secret (Stage_encrypt.generatedRoot.w_8_0 ctx))
        (matrixMulScalarLeft 1 (Stage_encrypt.generatedRoot.w_10_0 ctx)))
        (matrixMulScalarRight (matrixMulScalarLeft secret (Stage_encrypt.generatedRoot.w_14_0 ctx))
          (Stage_encrypt.generatedRoot.w_21_0 ctx))) (Stage_encrypt.generatedRoot.w_24_0 ctx)) 0 c' = _
      rw [e8, e10, e14, e21, he24, hsecret]
      simp only [matrixAdd, matrixSub, matrixMulScalarLeft_eq, matrixMulScalarRight_eq, Matrix.add_apply,
        Matrix.sub_apply, Matrix.smul_apply, smul_eq_mul, Matrix.one_apply_eq, one_mul, unitPair, stateErr,
        Fin.zero_eta, Fin.isValue, Matrix.cons_val_zero, map_mul, Nat.cast_one, Int.cast_one]
      ring
    · show (matrixAdd (matrixSub (matrixAdd (matrixMulScalarLeft secret (Stage_encrypt.generatedRoot.w_26_0 ctx))
        (matrixMulScalarLeft 1 (Stage_encrypt.generatedRoot.w_28_0 ctx)))
        (matrixMulScalarRight (matrixMulScalarLeft 1 (Stage_encrypt.generatedRoot.w_14_0 ctx))
          (Stage_encrypt.generatedRoot.w_21_0 ctx))) (Stage_encrypt.generatedRoot.w_34_0 ctx)) 0 c' = _
      rw [e26, e28, e14, e21, he34, hsecret]
      simp only [matrixAdd, matrixSub, matrixMulScalarLeft_eq, matrixMulScalarRight_eq, Matrix.add_apply,
        Matrix.sub_apply, Matrix.smul_apply, smul_eq_mul, Matrix.one_apply_eq, one_mul, unitPair, stateErr,
        Fin.mk_one, Fin.isValue, Matrix.cons_val_one, map_mul, Nat.cast_one, Int.cast_one]
      simp only [Matrix.cons_val_zero, map_one, mul_one]
  · obtain ⟨-, hb24⟩ := gaussianSampleAt_explicit hN F.c0.h24_0
    obtain ⟨-, hb34⟩ := gaussianSampleAt_explicit hN F.c0.h34_0
    fin_cases j
    · exact hb24
    · exact hb34

end PeProof
