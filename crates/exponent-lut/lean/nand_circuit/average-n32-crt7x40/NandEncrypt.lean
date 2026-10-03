import NandSetup

/-!
# The encryption stage

The state input is encrypted at level `0` (modulus `Q7`, scale one, gadget of `Q7` with fourteen
columns) under the key sampled from the hash key, and the LUT helpers of every level are
encryptions of the substituted mask.
-/

namespace NandProof

open Mxx.Primitives MxxRuntime MxxExponentLut

variable {tape : SampleTape} {path loopPath : List Nat}

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
theorem encrypt_input {model : HashModel} {mask payload : ExactMatrix Q7 N 1 1} {key : ByteArray}
    {coefficient exponent : Int} {outputs}
    (h : Stage_encrypt.generatedRoot Backend.backend model tape path GeneratedClaim.stage_3_params
      (mask, key, payload, coefficient, exponent, ()) outputs)
    {sm sp : ErrorPoly N} (hmask : mask = fun _ _ ↦ reducePoly Q7 N sm)
    (hpayload : payload = fun _ _ ↦ reducePoly Q7 N sp)
    {g' g : ExactMatrix Q7 N 1 14} (hg' : gadgetMatrixRuns Backend.backend 1048576 14 g')
    (hg : centeredRebaseRuns g' g) :
    Rel sm (inputKey model key)
      (fun j ↦ reducePoly Q7 N (((1 : Nat) : Int) * message (n := N) coefficient exponent * unitPair sp j) • g)
      ![outputs.1, outputs.2.1] 26 := by
  obtain ⟨w, h2, h5, h9, h10, h20, h22, h24, h31, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -,
    hout⟩ := h
  have hgg' := gadgetMatrixRuns_unique h9 hg'
  subst hgg'
  have hgg := centeredRebaseRuns_unique h10 hg
  subst hgg
  obtain ⟨e0, he0, hb0⟩ := gaussian_bound_at hN h20
  obtain ⟨e1, he1, hb1⟩ := gaussian_bound_at hN h31
  subst hout hmask hpayload
  refine ⟨![e0, e1], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩
  have hkey : ∀ k j, inputKey model key k j = ![![w.w_2_0, w.w_22_0], ![w.w_5_0, w.w_24_0]] k j := by
    intro k j
    fin_cases k <;> fin_cases j <;>
      simp [inputKey, inputPrefix, componentBytes, hashSample_eq h2, hashSample_eq h5,
        hashSample_eq h22, hashSample_eq h24]
  simp only [hkey]
  fin_cases j <;> funext r c <;> obtain rfl : r = 0 := Subsingleton.elim _ _ <;>
    (simp [matrixAdd, matrixSub, matrixMulScalarRight_eq, Matrix.mul_apply,
      multiplyMonomial_liftInteger, unitPair, he0, he1, reduceMatrix, matrixPolynomial_single]
     try ring)

set_option maxHeartbeats 4000000 in
/-- The level-one helpers: encryptions at `Q7` of `P7 σ_i(s)` against the gadget of `Q6`. -/
theorem encrypt_helper1 {model : HashModel} {mask payload : ExactMatrix Q7 N 1 1}
    {key : ByteArray} {coefficient exponent : Int} {outputs}
    (h : Stage_encrypt.generatedRoot Backend.backend model tape path GeneratedClaim.stage_3_params
      (mask, key, payload, coefficient, exponent, ()) outputs)
    {sm : ErrorPoly N} (hmask : mask = fun _ _ ↦ reducePoly Q7 N sm)
    {g' : ExactMatrix Q6 N 1 12} {g : ExactMatrix Q7 N 1 12}
    (hg' : gadgetMatrixRuns Backend.backend 1048576 12 g') (hg : centeredRebaseRuns g' g)
    (i : Fin 16) :
    Rel sm (lutA model key Q7 12 1 (sigma i))
      (fun j ↦ reducePoly Q7 N (((P7 : Nat) : Int) *
        unitPair (substitution (sigma i).toNat (sigma_odd i) sm) j) • g)
      ![outputs.2.2.2.2.1 i, outputs.2.2.2.2.2.1 i] 26 := by
  obtain ⟨w, -, -, -, -, -, -, -, -, -, -, -, h34, -, -, -, -, -, -, -, -, -, -, -, -, hout⟩ := h
  subst hout hmask
  obtain ⟨w3, w6, w10, w12, w13, w16, w18, w20, w23, w27, -, -, h3, h6, -, -, h10, h12, h13, h16,
    h18, h20, -, -, h23, h27, hout⟩ := h34 i
  have hgg' := gadgetMatrixRuns_unique h12 hg'
  subst hgg'
  have hgg := centeredRebaseRuns_unique h13 hg
  subst hgg
  obtain ⟨e0, he0, hb0⟩ := gaussian_bound_at hN h16
  obtain ⟨e1, he1, hb1⟩ := gaussian_bound_at hN h27
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
theorem encrypt_helper3 {model : HashModel} {mask payload : ExactMatrix Q7 N 1 1}
    {key : ByteArray} {coefficient exponent : Int} {outputs}
    (h : Stage_encrypt.generatedRoot Backend.backend model tape path GeneratedClaim.stage_3_params
      (mask, key, payload, coefficient, exponent, ()) outputs)
    {sm : ErrorPoly N} (hmask : mask = fun _ _ ↦ reducePoly Q7 N sm)
    {g' : ExactMatrix Q5 N 1 10} {g : ExactMatrix Q6 N 1 10}
    (hg' : gadgetMatrixRuns Backend.backend 1048576 10 g') (hg : centeredRebaseRuns g' g)
    (i : Fin 16) :
    Rel sm (lutA model key Q6 10 3 (sigma i))
      (fun j ↦ reducePoly Q6 N (((P6 : Nat) : Int) *
        unitPair (substitution (sigma i).toNat (sigma_odd i) sm) j) • g)
      ![outputs.2.2.2.2.2.2.2.2.1 i, outputs.2.2.2.2.2.2.2.2.2.1 i] 26 := by
  obtain ⟨w, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, hloop, -, -, -, -, -, -, -, -, hout⟩ := h
  subst hout hmask
  obtain ⟨w1, w4, w7, w8, w12, w14, w15, w18, w20, w22, w25, w29, h1, -, -, h4, h7, h8, -, -, h12,
    h14, h15, h18, h20, h22, -, -, h25, h29, hout⟩ := hloop i
  have hw1 := modulusReduce_poly hQ7 hQ6 hN h1
  have hw7 := modulusReduce_one hQ7 hQ6 hN h7
  subst hw1 hw7
  have hgg' := gadgetMatrixRuns_unique h14 hg'
  subst hgg'
  have hgg := centeredRebaseRuns_unique h15 hg
  subst hgg
  obtain ⟨e0, he0, hb0⟩ := gaussian_bound_at hN h18
  obtain ⟨e1, he1, hb1⟩ := gaussian_bound_at hN h29
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
theorem encrypt_helper5 {model : HashModel} {mask payload : ExactMatrix Q7 N 1 1}
    {key : ByteArray} {coefficient exponent : Int} {outputs}
    (h : Stage_encrypt.generatedRoot Backend.backend model tape path GeneratedClaim.stage_3_params
      (mask, key, payload, coefficient, exponent, ()) outputs)
    {sm : ErrorPoly N} (hmask : mask = fun _ _ ↦ reducePoly Q7 N sm)
    {g' : ExactMatrix Q4 N 1 8} {g : ExactMatrix Q5 N 1 8}
    (hg' : gadgetMatrixRuns Backend.backend 1048576 8 g') (hg : centeredRebaseRuns g' g)
    (i : Fin 16) :
    Rel sm (lutA model key Q5 8 5 (sigma i))
      (fun j ↦ reducePoly Q5 N (((P5 : Nat) : Int) *
        unitPair (substitution (sigma i).toNat (sigma_odd i) sm) j) • g)
      ![outputs.2.2.2.2.2.2.2.2.2.2.2.2.1 i, outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.1 i] 26 := by
  obtain ⟨w, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, hloop, -, -, -, -, hout⟩ := h
  subst hout hmask
  obtain ⟨w1, w4, w7, w8, w12, w14, w15, w18, w20, w22, w25, w29, h1, -, -, h4, h7, h8, -, -, h12,
    h14, h15, h18, h20, h22, -, -, h25, h29, hout⟩ := hloop i
  have hw1 := modulusReduce_poly hQ7 hQ5 hN h1
  have hw7 := modulusReduce_one hQ7 hQ5 hN h7
  subst hw1 hw7
  have hgg' := gadgetMatrixRuns_unique h14 hg'
  subst hgg'
  have hgg := centeredRebaseRuns_unique h15 hg
  subst hgg
  obtain ⟨e0, he0, hb0⟩ := gaussian_bound_at hN h18
  obtain ⟨e1, he1, hb1⟩ := gaussian_bound_at hN h29
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
theorem encrypt_helper7 {model : HashModel} {mask payload : ExactMatrix Q7 N 1 1}
    {key : ByteArray} {coefficient exponent : Int} {outputs}
    (h : Stage_encrypt.generatedRoot Backend.backend model tape path GeneratedClaim.stage_3_params
      (mask, key, payload, coefficient, exponent, ()) outputs)
    {sm : ErrorPoly N} (hmask : mask = fun _ _ ↦ reducePoly Q7 N sm)
    {g' : ExactMatrix Q3 N 1 6} {g : ExactMatrix Q4 N 1 6}
    (hg' : gadgetMatrixRuns Backend.backend 1048576 6 g') (hg : centeredRebaseRuns g' g)
    (i : Fin 16) :
    Rel sm (lutA model key Q4 6 7 (sigma i))
      (fun j ↦ reducePoly Q4 N (((P4 : Nat) : Int) *
        unitPair (substitution (sigma i).toNat (sigma_odd i) sm) j) • g)
      ![outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 i, outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 i] 26 := by
  obtain ⟨w, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, hloop, hout⟩ := h
  subst hout hmask
  obtain ⟨w1, w4, w7, w8, w12, w14, w15, w18, w20, w22, w25, w29, h1, -, -, h4, h7, h8, -, -, h12,
    h14, h15, h18, h20, h22, -, -, h25, h29, hout⟩ := hloop i
  have hw1 := modulusReduce_poly hQ7 hQ4 hN h1
  have hw7 := modulusReduce_one hQ7 hQ4 hN h7
  subst hw1 hw7
  have hgg' := gadgetMatrixRuns_unique h14 hg'
  subst hgg'
  have hgg := centeredRebaseRuns_unique h15 hg
  subst hgg
  obtain ⟨e0, he0, hb0⟩ := gaussian_bound_at hN h18
  obtain ⟨e1, he1, hb1⟩ := gaussian_bound_at hN h29
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

end NandProof
