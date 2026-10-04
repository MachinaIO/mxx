import NandSetup

/-!
# The encryption stage

Dynamic encryption samples every ciphertext under the single secret `s`, which is both the payload
and the mask. The state input is encrypted at level `0` (modulus `Q7`, scale one, gadget of `Q7`
with fourteen columns) under the key sampled from the hash key, and the LUT helpers of every level
are encryptions of the substituted mask.
-/

namespace NandProof

open Mxx.Primitives MxxRuntime MxxExponentLut

/-- `rns-bgg/input/WireId(0)/0` -/
def inputPrefix : Blob :=
  [114, 110, 115, 45, 98, 103, 103, 47, 105, 110, 112, 117, 116, 47, 87, 105, 114, 101, 73, 100,
    40, 48, 41, 47, 48]

/-- The public key of the state input. -/
noncomputable def inputKey (model : HashModel) (key : ByteArray) : Fin 2 → Fin 2 → ExactMatrix Q7 N 1 14 :=
  fun k j ↦ model.sample Q7 N 1 14 key (completeHashTag inputPrefix
    [.bytes componentBytes, .u64Le (2 * (k : Int) + (j : Int))])

set_option maxHeartbeats 4000000 in
/-- The state input is an encoding of `c X^e` at level `0`. -/
theorem encrypt_input {model : HashModel} {secret : ExactMatrix Q7 N 1 1} {key : ByteArray}
    {c e c0 e0 c1 e1 c2 e2 c3 e3 : Int} {outputs}
    (h : Stage_encrypt.generatedRoot Backend.backend model GeneratedClaim.stage_1_params
      (secret, key, c, e, c0, e0, c1, e1, c2, e2, c3, e3, ()) outputs)
    {sm : ErrorPoly N} (hsecret : secret = fun _ _ ↦ reducePoly Q7 N sm)
    {g' g : ExactMatrix Q7 N 1 14} (hg' : gadgetMatrixRuns Backend.backend 1048576 14 g')
    (hg : centeredRebaseRuns g' g) :
    Rel sm (inputKey model key)
      (fun j ↦ reducePoly Q7 N (((1 : Nat) : Int) * message (n := N) c e * unitPair sm j) • g)
      ![outputs.1, outputs.2.1] 26 := by
  obtain ⟨w, h2, h5, h8, h9, h19, h21, h23, h29, hrest⟩ := h
  obtain ⟨-, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, hc1⟩ := hrest
  obtain ⟨-, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, hc2⟩ := hc1
  obtain ⟨-, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, hout⟩ := hc2
  have hgg' := gadgetMatrixRuns_unique h8 hg'
  subst hgg'
  have hgg := centeredRebaseRuns_unique h9 hg
  subst hgg
  obtain ⟨e0', he0, hb0⟩ := gaussian_bound h19
  obtain ⟨e1', he1, hb1⟩ := gaussian_bound h29
  subst hout hsecret
  refine ⟨![e0', e1'], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩
  have hkey : ∀ k j, inputKey model key k j = ![![w.w_2_0, w.w_21_0], ![w.w_5_0, w.w_23_0]] k j := by
    intro k j
    fin_cases k <;> fin_cases j <;>
      simp [inputKey, inputPrefix, componentBytes, hashSample_eq h2, hashSample_eq h5,
        hashSample_eq h21, hashSample_eq h23]
  simp only [hkey]
  fin_cases j <;> funext r c' <;> obtain rfl : r = 0 := Subsingleton.elim _ _ <;>
    (simp [matrixAdd, matrixSub, matrixMulScalarRight_eq, Matrix.mul_apply,
      multiplyMonomial_liftInteger, unitPair, he0, he1, reduceMatrix, matrixPolynomial_single]
     try ring)

set_option maxHeartbeats 4000000 in
/-- The level-one helpers: encryptions at `Q7` of `P7 σ_i(s)` against the gadget of `Q6`. -/
theorem encrypt_helper1 {model : HashModel} {secret : ExactMatrix Q7 N 1 1}
    {key : ByteArray} {c e c0 e0 c1 e1 c2 e2 c3 e3 : Int} {outputs}
    (h : Stage_encrypt.generatedRoot Backend.backend model GeneratedClaim.stage_1_params
      (secret, key, c, e, c0, e0, c1, e1, c2, e2, c3, e3, ()) outputs)
    {sm : ErrorPoly N} (hsecret : secret = fun _ _ ↦ reducePoly Q7 N sm)
    {g' : ExactMatrix Q6 N 1 12} {g : ExactMatrix Q7 N 1 12}
    (hg' : gadgetMatrixRuns Backend.backend 1048576 12 g') (hg : centeredRebaseRuns g' g)
    (i : Fin 16) :
    Rel sm (lutA model key Q7 12 1 (sigma i))
      (fun j ↦ reducePoly Q7 N (((P7 : Nat) : Int) *
        unitPair (substitution (sigma i).toNat (sigma_odd i) sm) j) • g)
      ![outputs.2.2.2.2.1 i, outputs.2.2.2.2.2.1 i] 26 := by
  obtain ⟨w, -, -, -, -, -, -, -, -, -, -, -, h34, hrest⟩ := h
  obtain ⟨-, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, hc1⟩ := hrest
  obtain ⟨-, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, hc2⟩ := hc1
  obtain ⟨-, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, hout⟩ := hc2
  subst hout hsecret
  obtain ⟨w3, w6, w10, w12, w13, w16, w18, w20, w23, w27, -, -, h3, h6, -, -, h10, h12, h13, h16,
    h18, h20, -, -, h23, h27, hout⟩ := h34 i
  have hgg' := gadgetMatrixRuns_unique h12 hg'
  subst hgg'
  have hgg := centeredRebaseRuns_unique h13 hg
  subst hgg
  obtain ⟨e0, he0, hb0⟩ := gaussian_bound h16
  obtain ⟨e1, he1, hb1⟩ := gaussian_bound h27
  obtain ⟨_, hw10⟩ := ringAutomorphismRuns_eq hQ7 hN h10
  obtain ⟨_, hw23⟩ := ringAutomorphismRuns_eq hQ7 hN h23
  simp only [Prod.mk.injEq] at hout
  obtain ⟨ho0, ho1, -⟩ := hout
  refine ⟨![e0, e1], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩
  have hkey : ∀ k j, lutA model key Q7 12 1 (sigma i) k j = ![![w3, w18], ![w6, w20]] k j := by
    intro k j
    fin_cases k <;> fin_cases j <;>
      simp [lutA, lutAPrefix, componentBytes, sigma, hashSample_eq h3, hashSample_eq h6,
        hashSample_eq h18, hashSample_eq h20]
  simp only [hkey]
  fin_cases j <;> funext r c <;> obtain rfl : r = 0 := Subsingleton.elim _ _ <;>
    simp [matrixAdd, matrixSub, Matrix.mul_apply, unitPair, he0, he1, reduceMatrix,
      matrixPolynomial_single, ho0, ho1, hw10, hw23, reducePoly_substitution, sigma,
      map_ofNat]

set_option maxHeartbeats 4000000 in
/-- The level-3 helpers: encryptions at `Q6` of `P6 σ_i(s)` against the gadget of `Q5`. -/
theorem encrypt_helper3 {model : HashModel} {secret : ExactMatrix Q7 N 1 1}
    {key : ByteArray} {c e c0 e0 c1 e1 c2 e2 c3 e3 : Int} {outputs}
    (h : Stage_encrypt.generatedRoot Backend.backend model GeneratedClaim.stage_1_params
      (secret, key, c, e, c0, e0, c1, e1, c2, e2, c3, e3, ()) outputs)
    {sm : ErrorPoly N} (hsecret : secret = fun _ _ ↦ reducePoly Q7 N sm)
    {g' : ExactMatrix Q5 N 1 10} {g : ExactMatrix Q6 N 1 10}
    (hg' : gadgetMatrixRuns Backend.backend 1048576 10 g') (hg : centeredRebaseRuns g' g)
    (i : Fin 16) :
    Rel sm (lutA model key Q6 10 3 (sigma i))
      (fun j ↦ reducePoly Q6 N (((P6 : Nat) : Int) *
        unitPair (substitution (sigma i).toNat (sigma_odd i) sm) j) • g)
      ![outputs.2.2.2.2.2.2.2.2.1 i, outputs.2.2.2.2.2.2.2.2.2.1 i] 26 := by
  obtain ⟨w, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, hloop, hrest⟩ := h
  obtain ⟨-, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, hc1⟩ := hrest
  obtain ⟨-, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, hc2⟩ := hc1
  obtain ⟨-, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, hout⟩ := hc2
  subst hout hsecret
  obtain ⟨w1, w4, w7, w8, w12, w14, w15, w18, w20, w22, w25, w29, h1, -, -, h4, h7, h8, -, -, h12,
    h14, h15, h18, h20, h22, -, -, h25, h29, hout⟩ := hloop i
  have hw1 := modulusReduce_poly hQ7 hQ6 hN h1
  have hw7 := modulusReduce_one hQ7 hQ6 hN h7
  subst hw1 hw7
  have hgg' := gadgetMatrixRuns_unique h14 hg'
  subst hgg'
  have hgg := centeredRebaseRuns_unique h15 hg
  subst hgg
  obtain ⟨e0, he0, hb0⟩ := gaussian_bound h18
  obtain ⟨e1, he1, hb1⟩ := gaussian_bound h29
  obtain ⟨_, hw12⟩ := ringAutomorphismRuns_eq hQ6 hN h12
  obtain ⟨_, hw25⟩ := ringAutomorphismRuns_eq hQ6 hN h25
  simp only [Prod.mk.injEq] at hout
  obtain ⟨ho0, ho1, -⟩ := hout
  refine ⟨![e0, e1], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩
  have hkey : ∀ k j, lutA model key Q6 10 3 (sigma i) k j = ![![w4, w20], ![w8, w22]] k j := by
    intro k j
    fin_cases k <;> fin_cases j <;>
      simp [lutA, lutAPrefix, componentBytes, sigma, hashSample_eq h4, hashSample_eq h8,
        hashSample_eq h20, hashSample_eq h22]
  simp only [hkey]
  fin_cases j <;> funext r c <;> obtain rfl : r = 0 := Subsingleton.elim _ _ <;>
    simp [matrixAdd, matrixSub, Matrix.mul_apply, unitPair, he0, he1, reduceMatrix,
      matrixPolynomial_single, ho0, ho1, hw12, hw25, reducePoly_substitution, sigma,
      map_ofNat]

set_option maxHeartbeats 4000000 in
/-- The level-5 helpers: encryptions at `Q5` of `P5 σ_i(s)` against the gadget of `Q4`. -/
theorem encrypt_helper5 {model : HashModel} {secret : ExactMatrix Q7 N 1 1}
    {key : ByteArray} {c e c0 e0 c1 e1 c2 e2 c3 e3 : Int} {outputs}
    (h : Stage_encrypt.generatedRoot Backend.backend model GeneratedClaim.stage_1_params
      (secret, key, c, e, c0, e0, c1, e1, c2, e2, c3, e3, ()) outputs)
    {sm : ErrorPoly N} (hsecret : secret = fun _ _ ↦ reducePoly Q7 N sm)
    {g' : ExactMatrix Q4 N 1 8} {g : ExactMatrix Q5 N 1 8}
    (hg' : gadgetMatrixRuns Backend.backend 1048576 8 g') (hg : centeredRebaseRuns g' g)
    (i : Fin 16) :
    Rel sm (lutA model key Q5 8 5 (sigma i))
      (fun j ↦ reducePoly Q5 N (((P5 : Nat) : Int) *
        unitPair (substitution (sigma i).toNat (sigma_odd i) sm) j) • g)
      ![outputs.2.2.2.2.2.2.2.2.2.2.2.2.1 i, outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.1 i] 26 := by
  obtain ⟨w, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, hloop, hrest⟩ := h
  obtain ⟨-, -, -, -, -, -, -, -, -, -, -, -, hc1⟩ := hrest
  obtain ⟨-, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, hc2⟩ := hc1
  obtain ⟨-, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, hout⟩ := hc2
  subst hout hsecret
  obtain ⟨w1, w4, w7, w8, w12, w14, w15, w18, w20, w22, w25, w29, h1, -, -, h4, h7, h8, -, -, h12,
    h14, h15, h18, h20, h22, -, -, h25, h29, hout⟩ := hloop i
  have hw1 := modulusReduce_poly hQ7 hQ5 hN h1
  have hw7 := modulusReduce_one hQ7 hQ5 hN h7
  subst hw1 hw7
  have hgg' := gadgetMatrixRuns_unique h14 hg'
  subst hgg'
  have hgg := centeredRebaseRuns_unique h15 hg
  subst hgg
  obtain ⟨e0, he0, hb0⟩ := gaussian_bound h18
  obtain ⟨e1, he1, hb1⟩ := gaussian_bound h29
  obtain ⟨_, hw12⟩ := ringAutomorphismRuns_eq hQ5 hN h12
  obtain ⟨_, hw25⟩ := ringAutomorphismRuns_eq hQ5 hN h25
  simp only [Prod.mk.injEq] at hout
  obtain ⟨ho0, ho1, -⟩ := hout
  refine ⟨![e0, e1], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩
  have hkey : ∀ k j, lutA model key Q5 8 5 (sigma i) k j = ![![w4, w20], ![w8, w22]] k j := by
    intro k j
    fin_cases k <;> fin_cases j <;>
      simp [lutA, lutAPrefix, componentBytes, sigma, hashSample_eq h4, hashSample_eq h8,
        hashSample_eq h20, hashSample_eq h22]
  simp only [hkey]
  fin_cases j <;> funext r c <;> obtain rfl : r = 0 := Subsingleton.elim _ _ <;>
    simp [matrixAdd, matrixSub, Matrix.mul_apply, unitPair, he0, he1, reduceMatrix,
      matrixPolynomial_single, ho0, ho1, hw12, hw25, reducePoly_substitution, sigma,
      map_ofNat]

set_option maxHeartbeats 4000000 in
/-- The level-7 helpers: encryptions at `Q4` of `P4 σ_i(s)` against the gadget of `Q3`. -/
theorem encrypt_helper7 {model : HashModel} {secret : ExactMatrix Q7 N 1 1}
    {key : ByteArray} {c e c0 e0 c1 e1 c2 e2 c3 e3 : Int} {outputs}
    (h : Stage_encrypt.generatedRoot Backend.backend model GeneratedClaim.stage_1_params
      (secret, key, c, e, c0, e0, c1, e1, c2, e2, c3, e3, ()) outputs)
    {sm : ErrorPoly N} (hsecret : secret = fun _ _ ↦ reducePoly Q7 N sm)
    {g' : ExactMatrix Q3 N 1 6} {g : ExactMatrix Q4 N 1 6}
    (hg' : gadgetMatrixRuns Backend.backend 1048576 6 g') (hg : centeredRebaseRuns g' g)
    (i : Fin 16) :
    Rel sm (lutA model key Q4 6 7 (sigma i))
      (fun j ↦ reducePoly Q4 N (((P4 : Nat) : Int) *
        unitPair (substitution (sigma i).toNat (sigma_odd i) sm) j) • g)
      ![outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 i, outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 i] 26 := by
  obtain ⟨w, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, hloop, hrest⟩ := h
  obtain ⟨-, -, -, -, -, -, -, -, hc1⟩ := hrest
  obtain ⟨-, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, hc2⟩ := hc1
  obtain ⟨-, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, hout⟩ := hc2
  subst hout hsecret
  obtain ⟨w1, w4, w7, w8, w12, w14, w15, w18, w20, w22, w25, w29, h1, -, -, h4, h7, h8, -, -, h12,
    h14, h15, h18, h20, h22, -, -, h25, h29, hout⟩ := hloop i
  have hw1 := modulusReduce_poly hQ7 hQ4 hN h1
  have hw7 := modulusReduce_one hQ7 hQ4 hN h7
  subst hw1 hw7
  have hgg' := gadgetMatrixRuns_unique h14 hg'
  subst hgg'
  have hgg := centeredRebaseRuns_unique h15 hg
  subst hgg
  obtain ⟨e0, he0, hb0⟩ := gaussian_bound h18
  obtain ⟨e1, he1, hb1⟩ := gaussian_bound h29
  obtain ⟨_, hw12⟩ := ringAutomorphismRuns_eq hQ4 hN h12
  obtain ⟨_, hw25⟩ := ringAutomorphismRuns_eq hQ4 hN h25
  simp only [Prod.mk.injEq] at hout
  obtain ⟨ho0, ho1, -⟩ := hout
  refine ⟨![e0, e1], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩
  have hkey : ∀ k j, lutA model key Q4 6 7 (sigma i) k j = ![![w4, w20], ![w8, w22]] k j := by
    intro k j
    fin_cases k <;> fin_cases j <;>
      simp [lutA, lutAPrefix, componentBytes, sigma, hashSample_eq h4, hashSample_eq h8,
        hashSample_eq h20, hashSample_eq h22]
  simp only [hkey]
  fin_cases j <;> funext r c <;> obtain rfl : r = 0 := Subsingleton.elim _ _ <;>
    simp [matrixAdd, matrixSub, Matrix.mul_apply, unitPair, he0, he1, reduceMatrix,
      matrixPolynomial_single, ho0, ho1, hw12, hw25, reducePoly_substitution, sigma,
      map_ofNat]

set_option maxHeartbeats 8000000 in
/-- The switching ciphertexts of round `0` at level `2`: `σ_i(s) B_0j + B_1j = (C_i r_j) g + e`. -/
theorem encrypt_switch0 {model : HashModel} {key : ByteArray} {secret : ExactMatrix Q7 N 1 1}
    {c e c0 e0 c1 e1 c2 e2 c3 e3 : Int} {outputs}
    (h : Stage_encrypt.generatedRoot Backend.backend model GeneratedClaim.stage_1_params
      (secret, key, c, e, c0, e0, c1, e1, c2, e2, c3, e3, ()) outputs)
    {sp : ErrorPoly N} (hpayload : secret = fun _ _ ↦ reducePoly Q7 N sp)
    {g' g : ExactMatrix Q6 N 1 12}
    (hg' : gadgetMatrixRuns Backend.backend 1048576 12 g') (hg : centeredRebaseRuns g' g)
    (i : Fin 16) :
    ∃ e : Fin 2 → ErrorMatrix N 1 12, (∀ j, CoeffBound (e j) 26) ∧ ∀ j,
      reducePoly Q6 N (substitution (sigma i).toNat (sigma_odd i) sp) •
          ![outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 i, outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 i] j + ![outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 i, outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 i] j =
        (lutCoefficient Q6 inverse6 (MxxExponentLutNand.table 0) (sigma i).toNat *
          reducePoly Q6 N (1 * unitPair sp j)) • g + reduceMatrix Q6 N 1 12 (e j) := by
  obtain ⟨w, h2, h5, h8, h9, h19, h21, h23, h29, -, hpar31, -, hpar32, -, hpar33, -, hpar34, -, hpar35, -, hpar36, -, hpar37, -, hpar38, -, hpar39, h40, -, hpar41, -, hpar42, h43, -, hpar44, -, hpar45, h46, -, hpar47, -, hpar48, h49, -, hpar50, h51, h52, h53, h54, h65, h72, h74, -, hpar75, h76, h77, h78, h79, h80, h88, h92, h94, h100, h102, -, hpar103, h104, h105, h106, h107, h108, h116, h120, h122, h128, h130, -, hpar131, h132, h133, h134, h135, h136, h144, h148, h150, h156, h158, -, hpar159, hout⟩ := h
  subst hout hpayload
  obtain ⟨w0, w1, w3, w4, w11, w16, w22, w27, w34, w39, w45, w50, w58, w63, w69, w74, w81, w86, w92, w97, w106, w107, w110, w116, w120, h0, h1, h3, -, -, h4, -, -, -, -, -, -, h11, -, -, -, -, -, -, h16, -, -, -, -, -, -, h22, -, -, -, -, -, -, h27, -, -, -, -, -, -, h34, -, -, -, -, -, -, h39, -, -, -, -, -, -, h45, -, -, -, -, -, -, h50, -, -, -, -, -, -, h58, -, -, -, -, -, -, h63, -, -, -, -, -, -, h69, -, -, -, -, -, -, h74, -, -, -, -, -, -, h81, -, -, -, -, -, -, h86, -, -, -, -, -, -, h92, -, -, -, -, -, -, h97, h106, h107, h110, h116, h120, hout⟩ := hpar39 i
  obtain ⟨_, hw5⟩ := ringAutomorphismRuns_eq hQ6 hN h4
  have hw4 := modulusReduce_poly hQ7 hQ6 hN h3
  have hgg' := gadgetMatrixRuns_unique h106 hg'
  subst hgg'
  have hgg := centeredRebaseRuns_unique h107 hg
  subst hgg
  obtain ⟨e0', he0, hb0⟩ := gaussian_bound h110
  obtain ⟨e1', he1, hb1⟩ := gaussian_bound h120
  have hw117 := modulusReduce_one hQ7 hQ6 hN h116
  obtain ⟨_, a0⟩ := ringAutomorphismRuns_eq hQ6 hN h11
  obtain ⟨_, a1⟩ := ringAutomorphismRuns_eq hQ6 hN h16
  obtain ⟨_, a2⟩ := ringAutomorphismRuns_eq hQ6 hN h22
  obtain ⟨_, a3⟩ := ringAutomorphismRuns_eq hQ6 hN h27
  obtain ⟨_, a4⟩ := ringAutomorphismRuns_eq hQ6 hN h34
  obtain ⟨_, a5⟩ := ringAutomorphismRuns_eq hQ6 hN h39
  obtain ⟨_, a6⟩ := ringAutomorphismRuns_eq hQ6 hN h45
  obtain ⟨_, a7⟩ := ringAutomorphismRuns_eq hQ6 hN h50
  obtain ⟨_, a8⟩ := ringAutomorphismRuns_eq hQ6 hN h58
  obtain ⟨_, a9⟩ := ringAutomorphismRuns_eq hQ6 hN h63
  obtain ⟨_, a10⟩ := ringAutomorphismRuns_eq hQ6 hN h69
  obtain ⟨_, a11⟩ := ringAutomorphismRuns_eq hQ6 hN h74
  obtain ⟨_, a12⟩ := ringAutomorphismRuns_eq hQ6 hN h81
  obtain ⟨_, a13⟩ := ringAutomorphismRuns_eq hQ6 hN h86
  obtain ⟨_, a14⟩ := ringAutomorphismRuns_eq hQ6 hN h92
  obtain ⟨_, a15⟩ := ringAutomorphismRuns_eq hQ6 hN h97
  simp only [Prod.mk.injEq] at hout
  obtain ⟨ho0, ho1, ho2, ho3, -⟩ := hout
  refine ⟨![e0', e1'], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩
  rw [lutCoefficient_eq (sigma_odd i)]
  fin_cases j <;> funext r c <;> obtain rfl : r = 0 := Subsingleton.elim _ _ <;>
    simp [ho0, ho1, ho2, ho3, matrixAdd, matrixNeg, Matrix.mul_apply, unitPair, he0, he1, hw4, hw5,
      hw117, reduceMatrix, matrixPolynomial_single, Fin.sum_univ_succ,
      MxxExponentLutNand.table, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15,
      rotationPolynomial, substitution_root_pow, reducePoly_substitution, sigma] <;>
    ring

set_option maxHeartbeats 8000000 in
/-- The switching ciphertexts of round `1` at level `4`: `σ_i(s) B_0j + B_1j = (C_i r_j) g + e`. -/
theorem encrypt_switch1 {model : HashModel} {key : ByteArray} {secret : ExactMatrix Q7 N 1 1}
    {c e c0 e0 c1 e1 c2 e2 c3 e3 : Int} {outputs}
    (h : Stage_encrypt.generatedRoot Backend.backend model GeneratedClaim.stage_1_params
      (secret, key, c, e, c0, e0, c1, e1, c2, e2, c3, e3, ()) outputs)
    {sp : ErrorPoly N} (hpayload : secret = fun _ _ ↦ reducePoly Q7 N sp)
    {g' g : ExactMatrix Q5 N 1 10}
    (hg' : gadgetMatrixRuns Backend.backend 1048576 10 g') (hg : centeredRebaseRuns g' g)
    (i : Fin 16) :
    ∃ e : Fin 2 → ErrorMatrix N 1 10, (∀ j, CoeffBound (e j) 26) ∧ ∀ j,
      reducePoly Q5 N (substitution (sigma i).toNat (sigma_odd i) sp) •
          ![outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 i, outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 i] j + ![outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 i, outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 i] j =
        (lutCoefficient Q5 inverse5 (MxxExponentLutNand.table 1) (sigma i).toNat *
          reducePoly Q5 N (1 * unitPair sp j)) • g + reduceMatrix Q5 N 1 10 (e j) := by
  obtain ⟨w, h2, h5, h8, h9, h19, h21, h23, h29, -, hpar31, -, hpar32, -, hpar33, -, hpar34, -, hpar35, -, hpar36, -, hpar37, -, hpar38, -, hpar39, h40, -, hpar41, -, hpar42, h43, -, hpar44, -, hpar45, h46, -, hpar47, -, hpar48, h49, -, hpar50, h51, h52, h53, h54, h65, h72, h74, -, hpar75, h76, h77, h78, h79, h80, h88, h92, h94, h100, h102, -, hpar103, h104, h105, h106, h107, h108, h116, h120, h122, h128, h130, -, hpar131, h132, h133, h134, h135, h136, h144, h148, h150, h156, h158, -, hpar159, hout⟩ := h
  subst hout hpayload
  obtain ⟨w0, w1, w3, w4, w11, w16, w22, w27, w34, w39, w45, w50, w58, w63, w69, w74, w81, w86, w92, w97, w106, w107, w110, w116, w120, h0, h1, h3, -, -, h4, -, -, -, -, -, -, h11, -, -, -, -, -, -, h16, -, -, -, -, -, -, h22, -, -, -, -, -, -, h27, -, -, -, -, -, -, h34, -, -, -, -, -, -, h39, -, -, -, -, -, -, h45, -, -, -, -, -, -, h50, -, -, -, -, -, -, h58, -, -, -, -, -, -, h63, -, -, -, -, -, -, h69, -, -, -, -, -, -, h74, -, -, -, -, -, -, h81, -, -, -, -, -, -, h86, -, -, -, -, -, -, h92, -, -, -, -, -, -, h97, h106, h107, h110, h116, h120, hout⟩ := hpar42 i
  obtain ⟨_, hw5⟩ := ringAutomorphismRuns_eq hQ5 hN h4
  have hw4 := modulusReduce_poly hQ7 hQ5 hN h3
  have hgg' := gadgetMatrixRuns_unique h106 hg'
  subst hgg'
  have hgg := centeredRebaseRuns_unique h107 hg
  subst hgg
  obtain ⟨e0', he0, hb0⟩ := gaussian_bound h110
  obtain ⟨e1', he1, hb1⟩ := gaussian_bound h120
  have hw117 := modulusReduce_one hQ7 hQ5 hN h116
  obtain ⟨_, a0⟩ := ringAutomorphismRuns_eq hQ5 hN h11
  obtain ⟨_, a1⟩ := ringAutomorphismRuns_eq hQ5 hN h16
  obtain ⟨_, a2⟩ := ringAutomorphismRuns_eq hQ5 hN h22
  obtain ⟨_, a3⟩ := ringAutomorphismRuns_eq hQ5 hN h27
  obtain ⟨_, a4⟩ := ringAutomorphismRuns_eq hQ5 hN h34
  obtain ⟨_, a5⟩ := ringAutomorphismRuns_eq hQ5 hN h39
  obtain ⟨_, a6⟩ := ringAutomorphismRuns_eq hQ5 hN h45
  obtain ⟨_, a7⟩ := ringAutomorphismRuns_eq hQ5 hN h50
  obtain ⟨_, a8⟩ := ringAutomorphismRuns_eq hQ5 hN h58
  obtain ⟨_, a9⟩ := ringAutomorphismRuns_eq hQ5 hN h63
  obtain ⟨_, a10⟩ := ringAutomorphismRuns_eq hQ5 hN h69
  obtain ⟨_, a11⟩ := ringAutomorphismRuns_eq hQ5 hN h74
  obtain ⟨_, a12⟩ := ringAutomorphismRuns_eq hQ5 hN h81
  obtain ⟨_, a13⟩ := ringAutomorphismRuns_eq hQ5 hN h86
  obtain ⟨_, a14⟩ := ringAutomorphismRuns_eq hQ5 hN h92
  obtain ⟨_, a15⟩ := ringAutomorphismRuns_eq hQ5 hN h97
  simp only [Prod.mk.injEq] at hout
  obtain ⟨ho0, ho1, ho2, ho3, -⟩ := hout
  refine ⟨![e0', e1'], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩
  rw [lutCoefficient_eq (sigma_odd i)]
  fin_cases j <;> funext r c <;> obtain rfl : r = 0 := Subsingleton.elim _ _ <;>
    simp [ho0, ho1, ho2, ho3, matrixAdd, matrixNeg, Matrix.mul_apply, unitPair, he0, he1, hw4, hw5,
      hw117, reduceMatrix, matrixPolynomial_single, Fin.sum_univ_succ,
      MxxExponentLutNand.table, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15,
      rotationPolynomial, substitution_root_pow, reducePoly_substitution, sigma] <;>
    ring

set_option maxHeartbeats 8000000 in
/-- The switching ciphertexts of round `2` at level `6`: `σ_i(s) B_0j + B_1j = (C_i r_j) g + e`. -/
theorem encrypt_switch2 {model : HashModel} {key : ByteArray} {secret : ExactMatrix Q7 N 1 1}
    {c e c0 e0 c1 e1 c2 e2 c3 e3 : Int} {outputs}
    (h : Stage_encrypt.generatedRoot Backend.backend model GeneratedClaim.stage_1_params
      (secret, key, c, e, c0, e0, c1, e1, c2, e2, c3, e3, ()) outputs)
    {sp : ErrorPoly N} (hpayload : secret = fun _ _ ↦ reducePoly Q7 N sp)
    {g' g : ExactMatrix Q4 N 1 8}
    (hg' : gadgetMatrixRuns Backend.backend 1048576 8 g') (hg : centeredRebaseRuns g' g)
    (i : Fin 16) :
    ∃ e : Fin 2 → ErrorMatrix N 1 8, (∀ j, CoeffBound (e j) 26) ∧ ∀ j,
      reducePoly Q4 N (substitution (sigma i).toNat (sigma_odd i) sp) •
          ![outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 i, outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 i] j + ![outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 i, outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 i] j =
        (lutCoefficient Q4 inverse4 (MxxExponentLutNand.table 2) (sigma i).toNat *
          reducePoly Q4 N (1 * unitPair sp j)) • g + reduceMatrix Q4 N 1 8 (e j) := by
  obtain ⟨w, h2, h5, h8, h9, h19, h21, h23, h29, -, hpar31, -, hpar32, -, hpar33, -, hpar34, -, hpar35, -, hpar36, -, hpar37, -, hpar38, -, hpar39, h40, -, hpar41, -, hpar42, h43, -, hpar44, -, hpar45, h46, -, hpar47, -, hpar48, h49, -, hpar50, h51, h52, h53, h54, h65, h72, h74, -, hpar75, h76, h77, h78, h79, h80, h88, h92, h94, h100, h102, -, hpar103, h104, h105, h106, h107, h108, h116, h120, h122, h128, h130, -, hpar131, h132, h133, h134, h135, h136, h144, h148, h150, h156, h158, -, hpar159, hout⟩ := h
  subst hout hpayload
  obtain ⟨w0, w1, w3, w4, w11, w16, w22, w27, w34, w39, w45, w50, w58, w63, w69, w74, w81, w86, w92, w97, w106, w107, w110, w116, w120, h0, h1, h3, -, -, h4, -, -, -, -, -, -, h11, -, -, -, -, -, -, h16, -, -, -, -, -, -, h22, -, -, -, -, -, -, h27, -, -, -, -, -, -, h34, -, -, -, -, -, -, h39, -, -, -, -, -, -, h45, -, -, -, -, -, -, h50, -, -, -, -, -, -, h58, -, -, -, -, -, -, h63, -, -, -, -, -, -, h69, -, -, -, -, -, -, h74, -, -, -, -, -, -, h81, -, -, -, -, -, -, h86, -, -, -, -, -, -, h92, -, -, -, -, -, -, h97, h106, h107, h110, h116, h120, hout⟩ := hpar45 i
  obtain ⟨_, hw5⟩ := ringAutomorphismRuns_eq hQ4 hN h4
  have hw4 := modulusReduce_poly hQ7 hQ4 hN h3
  have hgg' := gadgetMatrixRuns_unique h106 hg'
  subst hgg'
  have hgg := centeredRebaseRuns_unique h107 hg
  subst hgg
  obtain ⟨e0', he0, hb0⟩ := gaussian_bound h110
  obtain ⟨e1', he1, hb1⟩ := gaussian_bound h120
  have hw117 := modulusReduce_one hQ7 hQ4 hN h116
  obtain ⟨_, a0⟩ := ringAutomorphismRuns_eq hQ4 hN h11
  obtain ⟨_, a1⟩ := ringAutomorphismRuns_eq hQ4 hN h16
  obtain ⟨_, a2⟩ := ringAutomorphismRuns_eq hQ4 hN h22
  obtain ⟨_, a3⟩ := ringAutomorphismRuns_eq hQ4 hN h27
  obtain ⟨_, a4⟩ := ringAutomorphismRuns_eq hQ4 hN h34
  obtain ⟨_, a5⟩ := ringAutomorphismRuns_eq hQ4 hN h39
  obtain ⟨_, a6⟩ := ringAutomorphismRuns_eq hQ4 hN h45
  obtain ⟨_, a7⟩ := ringAutomorphismRuns_eq hQ4 hN h50
  obtain ⟨_, a8⟩ := ringAutomorphismRuns_eq hQ4 hN h58
  obtain ⟨_, a9⟩ := ringAutomorphismRuns_eq hQ4 hN h63
  obtain ⟨_, a10⟩ := ringAutomorphismRuns_eq hQ4 hN h69
  obtain ⟨_, a11⟩ := ringAutomorphismRuns_eq hQ4 hN h74
  obtain ⟨_, a12⟩ := ringAutomorphismRuns_eq hQ4 hN h81
  obtain ⟨_, a13⟩ := ringAutomorphismRuns_eq hQ4 hN h86
  obtain ⟨_, a14⟩ := ringAutomorphismRuns_eq hQ4 hN h92
  obtain ⟨_, a15⟩ := ringAutomorphismRuns_eq hQ4 hN h97
  simp only [Prod.mk.injEq] at hout
  obtain ⟨ho0, ho1, ho2, ho3, -⟩ := hout
  refine ⟨![e0', e1'], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩
  rw [lutCoefficient_eq (sigma_odd i)]
  fin_cases j <;> funext r c <;> obtain rfl : r = 0 := Subsingleton.elim _ _ <;>
    simp [ho0, ho1, ho2, ho3, matrixAdd, matrixNeg, Matrix.mul_apply, unitPair, he0, he1, hw4, hw5,
      hw117, reduceMatrix, matrixPolynomial_single, Fin.sum_univ_succ,
      MxxExponentLutNand.table, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15,
      rotationPolynomial, substitution_root_pow, reducePoly_substitution, sigma] <;>
    ring

set_option maxHeartbeats 8000000 in
/-- The switching ciphertexts of round `3` at level `8`: `σ_i(s) B_0j + B_1j = (C_i r_j) g + e`. -/
theorem encrypt_switch3 {model : HashModel} {key : ByteArray} {secret : ExactMatrix Q7 N 1 1}
    {c e c0 e0 c1 e1 c2 e2 c3 e3 : Int} {outputs}
    (h : Stage_encrypt.generatedRoot Backend.backend model GeneratedClaim.stage_1_params
      (secret, key, c, e, c0, e0, c1, e1, c2, e2, c3, e3, ()) outputs)
    {sp : ErrorPoly N} (hpayload : secret = fun _ _ ↦ reducePoly Q7 N sp)
    (i : Fin 16) :
    ∃ e : Fin 2 → ErrorMatrix N 1 1, (∀ j, CoeffBound (e j) 26) ∧ ∀ j,
      reducePoly Q3 N (substitution (sigma i).toNat (sigma_odd i) sp) •
          ![outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 i, outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 i] j + ![outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 i, outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 i] j =
        (lutCoefficient Q3 inverse3 (MxxExponentLutNand.table 3) (sigma i).toNat *
          reducePoly Q3 N ((((Q3 / 2 : Nat) : Int) : ErrorPoly N) * unitPair sp j)) •
          (1 : ExactMatrix Q3 N 1 1) + reduceMatrix Q3 N 1 1 (e j) := by
  obtain ⟨w, h2, h5, h8, h9, h19, h21, h23, h29, -, hpar31, -, hpar32, -, hpar33, -, hpar34, -, hpar35, -, hpar36, -, hpar37, -, hpar38, -, hpar39, h40, -, hpar41, -, hpar42, h43, -, hpar44, -, hpar45, h46, -, hpar47, -, hpar48, h49, -, hpar50, h51, h52, h53, h54, h65, h72, h74, -, hpar75, h76, h77, h78, h79, h80, h88, h92, h94, h100, h102, -, hpar103, h104, h105, h106, h107, h108, h116, h120, h122, h128, h130, -, hpar131, h132, h133, h134, h135, h136, h144, h148, h150, h156, h158, -, hpar159, hout⟩ := h
  subst hout hpayload
  obtain ⟨w0, w1, w3, w4, w11, w16, w22, w27, w34, w39, w45, w50, w58, w63, w69, w74, w81, w86, w92, w97, w109, w115, w119, h0, h1, h3, -, -, h4, -, -, -, -, -, -, h11, -, -, -, -, -, -, h16, -, -, -, -, -, -, h22, -, -, -, -, -, -, h27, -, -, -, -, -, -, h34, -, -, -, -, -, -, h39, -, -, -, -, -, -, h45, -, -, -, -, -, -, h50, -, -, -, -, -, -, h58, -, -, -, -, -, -, h63, -, -, -, -, -, -, h69, -, -, -, -, -, -, h74, -, -, -, -, -, -, h81, -, -, -, -, -, -, h86, -, -, -, -, -, -, h92, -, -, -, -, -, -, h97, h109, h115, h119, hout⟩ := hpar48 i
  obtain ⟨_, hw5⟩ := ringAutomorphismRuns_eq hQ3 hN h4
  have hw4 := modulusReduce_poly hQ7 hQ3 hN h3
  obtain ⟨e0', he0, hb0⟩ := gaussian_bound h109
  obtain ⟨e1', he1, hb1⟩ := gaussian_bound h119
  have hw116 := modulusReduce_one hQ7 hQ3 hN h115
  obtain ⟨_, a0⟩ := ringAutomorphismRuns_eq hQ3 hN h11
  obtain ⟨_, a1⟩ := ringAutomorphismRuns_eq hQ3 hN h16
  obtain ⟨_, a2⟩ := ringAutomorphismRuns_eq hQ3 hN h22
  obtain ⟨_, a3⟩ := ringAutomorphismRuns_eq hQ3 hN h27
  obtain ⟨_, a4⟩ := ringAutomorphismRuns_eq hQ3 hN h34
  obtain ⟨_, a5⟩ := ringAutomorphismRuns_eq hQ3 hN h39
  obtain ⟨_, a6⟩ := ringAutomorphismRuns_eq hQ3 hN h45
  obtain ⟨_, a7⟩ := ringAutomorphismRuns_eq hQ3 hN h50
  obtain ⟨_, a8⟩ := ringAutomorphismRuns_eq hQ3 hN h58
  obtain ⟨_, a9⟩ := ringAutomorphismRuns_eq hQ3 hN h63
  obtain ⟨_, a10⟩ := ringAutomorphismRuns_eq hQ3 hN h69
  obtain ⟨_, a11⟩ := ringAutomorphismRuns_eq hQ3 hN h74
  obtain ⟨_, a12⟩ := ringAutomorphismRuns_eq hQ3 hN h81
  obtain ⟨_, a13⟩ := ringAutomorphismRuns_eq hQ3 hN h86
  obtain ⟨_, a14⟩ := ringAutomorphismRuns_eq hQ3 hN h92
  obtain ⟨_, a15⟩ := ringAutomorphismRuns_eq hQ3 hN h97
  simp only [Prod.mk.injEq] at hout
  obtain ⟨ho0, ho1, ho2, ho3, -⟩ := hout
  refine ⟨![e0', e1'], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩
  rw [lutCoefficient_eq (sigma_odd i)]
  fin_cases j <;> funext r c <;> obtain rfl : r = 0 := Subsingleton.elim _ _ <;>
    obtain rfl : c = 0 := Subsingleton.elim _ _ <;>
    simp [ho0, ho1, ho2, ho3, matrixAdd, matrixNeg, Matrix.mul_apply, unitPair, he0, he1, hw4, hw5,
      hw116, reduceMatrix, matrixPolynomial_single, Fin.sum_univ_succ,
      MxxExponentLutNand.table, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15,
      rotationPolynomial, substitution_root_pow, map_ofNat, reducePoly_substitution, sigma,
      Matrix.one_apply_eq] <;>
    ring


set_option maxHeartbeats 8000000 in
/-- The RHS ciphertext of round `0`, sampled by encryption with a uniform top row:
`s top_j + bot_j = (P7 ν r_j) g + e_j`. -/
theorem encrypt_rhs0 {model : HashModel} {key : ByteArray} {secret : ExactMatrix Q7 N 1 1}
    {c e c0 e0 c1 e1 c2 e2 c3 e3 : Int} {outputs}
    (h : Stage_encrypt.generatedRoot Backend.backend model GeneratedClaim.stage_1_params
      (secret, key, c, e, c0, e0, c1, e1, c2, e2, c3, e3, ()) outputs)
    {sm : ErrorPoly N} (hsecret : secret = fun _ _ ↦ reducePoly Q7 N sm)
    {g' : ExactMatrix Q6 N 1 12} {g : ExactMatrix Q7 N 1 12}
    (hg' : gadgetMatrixRuns Backend.backend 1048576 12 g') (hg : centeredRebaseRuns g' g) :
    ∃ e : Fin 2 → ErrorMatrix N 1 12, (∀ j, CoeffBound (e j) 26) ∧ ∀ j,
      reducePoly Q7 N sm • ![outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1] j +
          ![outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1] j =
        reducePoly Q7 N (((P7 : Nat) : Int) * message (n := N) c0 e0 * unitPair sm j) • g +
          reduceMatrix Q7 N 1 12 (e j) := by
  obtain ⟨w, h2, h5, h8, h9, h19, h21, h23, h29, -, hpar31, -, hpar32, -, hpar33, -, hpar34, -, hpar35, -, hpar36, -, hpar37, -, hpar38, -, hpar39, h40, -, hpar41, -, hpar42, h43, -, hpar44, -, hpar45, h46, -, hpar47, -, hpar48, h49, -, hpar50, h51, h52, h53, h54, h65, h72, h74, -, hpar75, h76, h77, h78, h79, h80, h88, h92, h94, h100, h102, -, hpar103, h104, h105, h106, h107, h108, h116, h120, h122, h128, h130, -, hpar131, h132, h133, h134, h135, h136, h144, h148, h150, h156, h158, -, hpar159, hout⟩ := h
  subst hout hsecret
  have hgg' := gadgetMatrixRuns_unique h53 hg'
  subst hgg'
  have hgg := centeredRebaseRuns_unique h54 hg
  subst hgg
  obtain ⟨e0', he0, hb0⟩ := gaussian_bound h65
  obtain ⟨e1', he1, hb1⟩ := gaussian_bound h72
  refine ⟨![e0', e1'], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩
  fin_cases j <;> funext r c' <;> obtain rfl : r = 0 := Subsingleton.elim _ _ <;>
    simp (config := { zetaDelta := true }) [matrixAdd, matrixSub, matrixMulScalarRight_eq,
      Matrix.mul_apply, unitPair, he0, he1, reduceMatrix, matrixPolynomial_single,
      map_ofNat, multiplyMonomial_liftInteger] <;>
    ring

set_option maxHeartbeats 8000000 in
/-- The RHS ciphertext of round `1`, sampled by encryption with a uniform top row:
`s top_j + bot_j = (P6 ν r_j) g + e_j`. -/
theorem encrypt_rhs1 {model : HashModel} {key : ByteArray} {secret : ExactMatrix Q7 N 1 1}
    {c e c0 e0 c1 e1 c2 e2 c3 e3 : Int} {outputs}
    (h : Stage_encrypt.generatedRoot Backend.backend model GeneratedClaim.stage_1_params
      (secret, key, c, e, c0, e0, c1, e1, c2, e2, c3, e3, ()) outputs)
    {sm : ErrorPoly N} (hsecret : secret = fun _ _ ↦ reducePoly Q7 N sm)
    {g' : ExactMatrix Q5 N 1 10} {g : ExactMatrix Q6 N 1 10}
    (hg' : gadgetMatrixRuns Backend.backend 1048576 10 g') (hg : centeredRebaseRuns g' g) :
    ∃ e : Fin 2 → ErrorMatrix N 1 10, (∀ j, CoeffBound (e j) 26) ∧ ∀ j,
      reducePoly Q6 N sm • ![outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1] j +
          ![outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1] j =
        reducePoly Q6 N (((P6 : Nat) : Int) * message (n := N) c1 e1 * unitPair sm j) • g +
          reduceMatrix Q6 N 1 10 (e j) := by
  obtain ⟨w, h2, h5, h8, h9, h19, h21, h23, h29, -, hpar31, -, hpar32, -, hpar33, -, hpar34, -, hpar35, -, hpar36, -, hpar37, -, hpar38, -, hpar39, h40, -, hpar41, -, hpar42, h43, -, hpar44, -, hpar45, h46, -, hpar47, -, hpar48, h49, -, hpar50, h51, h52, h53, h54, h65, h72, h74, -, hpar75, h76, h77, h78, h79, h80, h88, h92, h94, h100, h102, -, hpar103, h104, h105, h106, h107, h108, h116, h120, h122, h128, h130, -, hpar131, h132, h133, h134, h135, h136, h144, h148, h150, h156, h158, -, hpar159, hout⟩ := h
  subst hout hsecret
  dsimp (config := { zetaDelta := true }) only at h78 h94 h88
  have hpay := modulusReduce_poly hQ7 hQ6 hN h78
  have hone := modulusReduce_one hQ7 hQ6 hN h94
  rw [multiplyMonomial_liftInteger] at h88
  have hnu := modulusReduce_poly hQ7 hQ6 hN h88
  have hgg' := gadgetMatrixRuns_unique h79 hg'
  subst hgg'
  have hgg := centeredRebaseRuns_unique h80 hg
  subst hgg
  obtain ⟨e0', he0, hb0⟩ := gaussian_bound h92
  obtain ⟨e1', he1, hb1⟩ := gaussian_bound h100
  refine ⟨![e0', e1'], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩
  fin_cases j <;> funext r c' <;> obtain rfl : r = 0 := Subsingleton.elim _ _ <;>
    simp (config := { zetaDelta := true }) [matrixAdd, matrixSub, matrixMulScalarRight_eq,
      Matrix.mul_apply, unitPair, he0, he1, hpay, hone, hnu, reduceMatrix, matrixPolynomial_single,
      map_ofNat] <;>
    ring

set_option maxHeartbeats 8000000 in
/-- The RHS ciphertext of round `2`, sampled by encryption with a uniform top row:
`s top_j + bot_j = (P5 ν r_j) g + e_j`. -/
theorem encrypt_rhs2 {model : HashModel} {key : ByteArray} {secret : ExactMatrix Q7 N 1 1}
    {c e c0 e0 c1 e1 c2 e2 c3 e3 : Int} {outputs}
    (h : Stage_encrypt.generatedRoot Backend.backend model GeneratedClaim.stage_1_params
      (secret, key, c, e, c0, e0, c1, e1, c2, e2, c3, e3, ()) outputs)
    {sm : ErrorPoly N} (hsecret : secret = fun _ _ ↦ reducePoly Q7 N sm)
    {g' : ExactMatrix Q4 N 1 8} {g : ExactMatrix Q5 N 1 8}
    (hg' : gadgetMatrixRuns Backend.backend 1048576 8 g') (hg : centeredRebaseRuns g' g) :
    ∃ e : Fin 2 → ErrorMatrix N 1 8, (∀ j, CoeffBound (e j) 26) ∧ ∀ j,
      reducePoly Q5 N sm • ![outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1] j +
          ![outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1] j =
        reducePoly Q5 N (((P5 : Nat) : Int) * message (n := N) c2 e2 * unitPair sm j) • g +
          reduceMatrix Q5 N 1 8 (e j) := by
  obtain ⟨w, h2, h5, h8, h9, h19, h21, h23, h29, -, hpar31, -, hpar32, -, hpar33, -, hpar34, -, hpar35, -, hpar36, -, hpar37, -, hpar38, -, hpar39, h40, -, hpar41, -, hpar42, h43, -, hpar44, -, hpar45, h46, -, hpar47, -, hpar48, h49, -, hpar50, h51, h52, h53, h54, h65, h72, h74, -, hpar75, h76, h77, h78, h79, h80, h88, h92, h94, h100, h102, -, hpar103, h104, h105, h106, h107, h108, h116, h120, h122, h128, h130, -, hpar131, h132, h133, h134, h135, h136, h144, h148, h150, h156, h158, -, hpar159, hout⟩ := h
  subst hout hsecret
  dsimp (config := { zetaDelta := true }) only at h106 h122 h116
  have hpay := modulusReduce_poly hQ7 hQ5 hN h106
  have hone := modulusReduce_one hQ7 hQ5 hN h122
  rw [multiplyMonomial_liftInteger] at h116
  have hnu := modulusReduce_poly hQ7 hQ5 hN h116
  have hgg' := gadgetMatrixRuns_unique h107 hg'
  subst hgg'
  have hgg := centeredRebaseRuns_unique h108 hg
  subst hgg
  obtain ⟨e0', he0, hb0⟩ := gaussian_bound h120
  obtain ⟨e1', he1, hb1⟩ := gaussian_bound h128
  refine ⟨![e0', e1'], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩
  fin_cases j <;> funext r c' <;> obtain rfl : r = 0 := Subsingleton.elim _ _ <;>
    simp (config := { zetaDelta := true }) [matrixAdd, matrixSub, matrixMulScalarRight_eq,
      Matrix.mul_apply, unitPair, he0, he1, hpay, hone, hnu, reduceMatrix, matrixPolynomial_single,
      map_ofNat] <;>
    ring

set_option maxHeartbeats 8000000 in
/-- The RHS ciphertext of round `3`, sampled by encryption with a uniform top row:
`s top_j + bot_j = (P4 ν r_j) g + e_j`. -/
theorem encrypt_rhs3 {model : HashModel} {key : ByteArray} {secret : ExactMatrix Q7 N 1 1}
    {c e c0 e0 c1 e1 c2 e2 c3 e3 : Int} {outputs}
    (h : Stage_encrypt.generatedRoot Backend.backend model GeneratedClaim.stage_1_params
      (secret, key, c, e, c0, e0, c1, e1, c2, e2, c3, e3, ()) outputs)
    {sm : ErrorPoly N} (hsecret : secret = fun _ _ ↦ reducePoly Q7 N sm)
    {g' : ExactMatrix Q3 N 1 6} {g : ExactMatrix Q4 N 1 6}
    (hg' : gadgetMatrixRuns Backend.backend 1048576 6 g') (hg : centeredRebaseRuns g' g) :
    ∃ e : Fin 2 → ErrorMatrix N 1 6, (∀ j, CoeffBound (e j) 26) ∧ ∀ j,
      reducePoly Q4 N sm • ![outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1] j +
          ![outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1] j =
        reducePoly Q4 N (((P4 : Nat) : Int) * message (n := N) c3 e3 * unitPair sm j) • g +
          reduceMatrix Q4 N 1 6 (e j) := by
  obtain ⟨w, h2, h5, h8, h9, h19, h21, h23, h29, -, hpar31, -, hpar32, -, hpar33, -, hpar34, -, hpar35, -, hpar36, -, hpar37, -, hpar38, -, hpar39, h40, -, hpar41, -, hpar42, h43, -, hpar44, -, hpar45, h46, -, hpar47, -, hpar48, h49, -, hpar50, h51, h52, h53, h54, h65, h72, h74, -, hpar75, h76, h77, h78, h79, h80, h88, h92, h94, h100, h102, -, hpar103, h104, h105, h106, h107, h108, h116, h120, h122, h128, h130, -, hpar131, h132, h133, h134, h135, h136, h144, h148, h150, h156, h158, -, hpar159, hout⟩ := h
  subst hout hsecret
  dsimp (config := { zetaDelta := true }) only at h134 h150 h144
  have hpay := modulusReduce_poly hQ7 hQ4 hN h134
  have hone := modulusReduce_one hQ7 hQ4 hN h150
  rw [multiplyMonomial_liftInteger] at h144
  have hnu := modulusReduce_poly hQ7 hQ4 hN h144
  have hgg' := gadgetMatrixRuns_unique h135 hg'
  subst hgg'
  have hgg := centeredRebaseRuns_unique h136 hg
  subst hgg
  obtain ⟨e0', he0, hb0⟩ := gaussian_bound h148
  obtain ⟨e1', he1, hb1⟩ := gaussian_bound h156
  refine ⟨![e0', e1'], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩
  fin_cases j <;> funext r c' <;> obtain rfl : r = 0 := Subsingleton.elim _ _ <;>
    simp (config := { zetaDelta := true }) [matrixAdd, matrixSub, matrixMulScalarRight_eq,
      Matrix.mul_apply, unitPair, he0, he1, hpay, hone, hnu, reduceMatrix, matrixPolynomial_single,
      map_ofNat] <;>
    ring

end NandProof
