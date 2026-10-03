import LwrSetup

/-!
# The encryption stage

The state input is encrypted at `Q7` with scale `P7` against the gadget of `Q6`; every member of
the public vector `a` is a public encoding at `Q7` with the native gadget, whose payload is the
mask, together with its public payload; and the readout LUT's helpers are encryptions at `Q2` of
`P2 σ_i(s)` against the gadget of `Q1`.
-/

namespace LwrProof

open Mxx.Primitives MxxRuntime MxxExponentLut

variable {tape : SampleTape} {path loopPath cellPath : List Nat} {loopPaths : Nat → List Nat}

set_option maxHeartbeats 4000000 in
/-- The state input is an encoding of `c X^e` at `Q7` with scale `P7`. -/
theorem encrypt_state {model : HashModel} {mask payload : ExactMatrix Q7 N 1 1} {key : ByteArray}
    {coefficient exponent : Int} {ac ae : Fin 17 → Int} {outputs}
    (h : Stage_encrypt.generatedRoot Backend.backend model tape path GeneratedClaim.stage_3_params
      (mask, key, payload, coefficient, exponent, ac, ae, ()) outputs)
    {sm sp : ErrorPoly N} (hmask : mask = fun _ _ ↦ reducePoly Q7 N sm)
    (hpayload : payload = fun _ _ ↦ reducePoly Q7 N sp)
    {g' : ExactMatrix Q6 N 1 12} {g : ExactMatrix Q7 N 1 12}
    (hg' : gadgetMatrixRuns Backend.backend 1048576 12 g') (hg : centeredRebaseRuns g' g) :
    Rel sm (stateKey model key)
      (fun j ↦ reducePoly Q7 N (((P7 : Nat) : Int) * message (n := N) coefficient exponent * unitPair sp j) • g)
      ![outputs.1, outputs.2.1] 26 := by
  obtain ⟨w, h2, h5, h9, h10, h20, h22, h24, h31, -, -, -, -, -, -, -, -, -, -, -, -, -, -,
    hout⟩ := h
  have hgg' := gadgetMatrixRuns_unique h9 hg'
  subst hgg'
  have hgg := centeredRebaseRuns_unique h10 hg
  subst hgg
  obtain ⟨e0, hb0, he0⟩ := gaussian_bound_at hN h20
  obtain ⟨e1, hb1, he1⟩ := gaussian_bound_at hN h31
  subst hout hmask hpayload
  refine ⟨![e0, e1], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩
  have hkey : ∀ k j, stateKey model key k j = ![![w.w_2_0, w.w_22_0], ![w.w_5_0, w.w_24_0]] k j := by
    intro k j
    fin_cases k <;> fin_cases j <;>
      simp [stateKey, statePrefix, componentBytes, hashSample_eq h2, hashSample_eq h5,
        hashSample_eq h22, hashSample_eq h24]
  simp only [hkey]
  fin_cases j <;> funext r c <;> obtain rfl : r = 0 := Subsingleton.elim _ _ <;>
    (simp [matrixAdd, matrixSub, matrixMulScalarRight_smul, Matrix.mul_apply,
      multiplyMonomial_liftInteger, unitPair, hb0, hb1, reduceMatrix, matrixPolynomial_single,
      map_ofNat]
     try ring)

set_option maxHeartbeats 4000000 in
/-- Member `i` of `a` is a public encoding of `c_i X^(e_i)` at `Q7` with the native gadget, and
its public payload is that plaintext. -/
theorem encrypt_member {model : HashModel} {mask payload : ExactMatrix Q7 N 1 1} {key : ByteArray}
    {coefficient exponent : Int} {ac ae : Fin 17 → Int} {outputs}
    (h : Stage_encrypt.generatedRoot Backend.backend model tape path GeneratedClaim.stage_3_params
      (mask, key, payload, coefficient, exponent, ac, ae, ()) outputs)
    {sm : ErrorPoly N} (hmask : mask = fun _ _ ↦ reducePoly Q7 N sm)
    {G : ExactMatrix Q7 N 1 14} (hG : gadgetMatrixRuns Backend.backend 1048576 14 G)
    (i : Fin 17) :
    Rel sm (memberKey model key i)
      (fun j ↦ reducePoly Q7 N (message (n := N) (ac i) (ae i) * unitPair sm j) • G)
      ![outputs.2.2.1 i, outputs.2.2.2.1 i] 26 ∧
      outputs.2.2.2.2.1 i = fun _ _ ↦ reducePoly Q7 N (message (n := N) (ac i) (ae i)) := by
  obtain ⟨w, -, -, -, -, -, -, -, -, -, hloop, -, -, -, -, -, -, -, -, -, -, -, -, hout⟩ := h
  subst hout hmask
  obtain ⟨w2, w5, w12, w16, w18, w20, w25, h2, h5, h12, h16, h18, h20, h25, hout⟩ := hloop i
  have hgg := gadgetMatrixRuns_unique h12 hG
  subst hgg
  obtain ⟨e0, hb0, he0⟩ := gaussian_bound_at hN h16
  obtain ⟨e1, hb1, he1⟩ := gaussian_bound_at hN h25
  simp only [Prod.mk.injEq] at hout
  obtain ⟨ho0, ho1, ho2, -⟩ := hout
  refine ⟨⟨![e0, e1], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩, ?_⟩
  · have hkey : ∀ k j, memberKey model key i k j = ![![w2, w18], ![w5, w20]] k j := by
      intro k j
      fin_cases k <;> fin_cases j <;>
        simp [memberKey, memberPrefix, memberBytes, componentBytes, hashSample_eq h2,
          hashSample_eq h5, hashSample_eq h18, hashSample_eq h20]
    simp only [hkey]
    fin_cases j <;> funext r c <;> obtain rfl : r = 0 := Subsingleton.elim _ _ <;>
      (simp [matrixAdd, matrixSub, Matrix.mul_apply, multiplyMonomial_liftInteger, unitPair,
        hb0, hb1, reduceMatrix, ho0, ho1]
       try ring)
  · simp [ho2, multiplyMonomial_liftInteger]

set_option maxHeartbeats 4000000 in
/-- The level-5 helpers: encryptions at `Q2` of `P2 σ_i(s)` against the gadget of `Q1`. -/
theorem encrypt_helper5 {model : HashModel} {mask payload : ExactMatrix Q7 N 1 1}
    {key : ByteArray} {coefficient exponent : Int} {ac ae : Fin 17 → Int} {outputs}
    (h : Stage_encrypt.generatedRoot Backend.backend model tape path GeneratedClaim.stage_3_params
      (mask, key, payload, coefficient, exponent, ac, ae, ()) outputs)
    {sm : ErrorPoly N} (hmask : mask = fun _ _ ↦ reducePoly Q7 N sm)
    {g' : ExactMatrix Q1 N 1 2} {g : ExactMatrix Q2 N 1 2}
    (hg' : gadgetMatrixRuns Backend.backend 1048576 2 g') (hg : centeredRebaseRuns g' g)
    (i : Fin 16) :
    Rel sm (lutA model key Q2 2 5 (sigma i))
      (fun j ↦ reducePoly Q2 N (((P2 : Nat) : Int) *
        unitPair (substitution (sigma i).toNat (sigma_odd i) sm) j) • g)
      ![outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 i, outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 i] 26 := by
  obtain ⟨w, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, hloop, hout⟩ := h
  subst hout hmask
  obtain ⟨w1, w4, w7, w8, w12, w14, w15, w18, w20, w22, w25, w29, h1, -, -, h4, h7, h8, -, -, h12,
    h14, h15, h18, h20, h22, -, -, h25, h29, hout⟩ := hloop i
  have hw1 := modulusReduce_poly hQ7 hQ2 hN h1
  have hw7 := modulusReduce_one hQ7 hQ2 hN h7
  subst hw1 hw7
  have hgg' := gadgetMatrixRuns_unique h14 hg'
  subst hgg'
  have hgg := centeredRebaseRuns_unique h15 hg
  subst hgg
  obtain ⟨e0, hb0, he0⟩ := gaussian_bound_at hN h18
  obtain ⟨e1, hb1, he1⟩ := gaussian_bound_at hN h29
  obtain ⟨_, hw12⟩ := ringAutomorphismRuns_eq hQ2 hN h12
  obtain ⟨_, hw25⟩ := ringAutomorphismRuns_eq hQ2 hN h25
  simp only [Prod.mk.injEq] at hout
  obtain ⟨ho0, ho1, -⟩ := hout
  refine ⟨![e0, e1], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩
  have hkey : ∀ k j, lutA model key Q2 2 5 (sigma i) k j = ![![w4, w20], ![w8, w22]] k j := by
    intro k j
    fin_cases k <;> fin_cases j <;>
      simp [lutA, lutAPrefix, componentBytes, sigma, hashSample_eq h4, hashSample_eq h8,
        hashSample_eq h20, hashSample_eq h22]
  simp only [hkey]
  fin_cases j <;> funext r c <;> obtain rfl : r = 0 := Subsingleton.elim _ _ <;>
    simp [matrixAdd, matrixSub, Matrix.mul_apply, unitPair, hb0, hb1, reduceMatrix,
      matrixPolynomial_single, ho0, ho1, hw12, hw25, reducePoly_substitution, sigma,
      map_ofNat]

end LwrProof
