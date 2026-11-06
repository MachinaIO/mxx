import PublicIpSetup

/-!
# The encryption stage

Every member of the input family is a public encoding at `Q` with the native gadget, whose payload
is the mask, together with its public payload.
-/

namespace PublicIpProof

open Mxx.Primitives MxxRuntime MxxExponentLut

variable {tape : SampleTape} {path : List Nat}

set_option maxHeartbeats 4000000 in
/-- Member `i` is a public encoding of `c_i X^(e_i)` at `Q` with the native gadget, and its public
payload is that plaintext. -/
theorem encrypt_member {model : HashModel} {mask : ExactMatrix Q N 1 1} {key : ByteArray}
    {ac ae : Fin 5 → Int} {outputs}
    (h : Stage_encrypt.generatedRoot Backend.backend model tape path GeneratedClaim.stage_1_params
      (mask, key, ac, ae, ()) outputs)
    {sm : ErrorPoly N} (hmask : mask = fun _ _ ↦ reducePoly Q N sm)
    {G : ExactMatrix Q N 1 4} (hG : gadgetMatrixRuns Backend.backend 1048576 4 G) (i : Fin 5) :
    Rel sm (memberKey model key i)
      (fun j ↦ reducePoly Q N (message (n := N) (ac i) (ae i) * unitPair sm j) • G)
      ![outputs.1 i, outputs.2.1 i] 26 ∧
      outputs.2.2.1 i = fun _ _ ↦ reducePoly Q N (message (n := N) (ac i) (ae i)) := by
  obtain ⟨w, -, hloop, hout⟩ := h
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

end PublicIpProof
