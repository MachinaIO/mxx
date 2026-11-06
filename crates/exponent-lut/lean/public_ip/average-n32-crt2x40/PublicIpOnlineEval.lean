import PublicIpEncrypt

/-!
# Online evaluation

The selected coordinates `0, 1, 3` are multiplied in a balanced tree: the one-member family of the
first level multiplies members `0` and `1`, and the root multiplies that product by member `3`,
the odd member carried to the second level. Each product is a public multiplication at the native
gadget of `Q`, so the output is a public encoding of `α₀ α₁ α₃`.
-/

namespace PublicIpProof

open Mxx.Primitives MxxRuntime MxxExponentLut

variable {tape : SampleTape} {path : List Nat}

/-- The error bound of a public product of two encodings of errors `Ba` and `Bs`. -/
abbrev productBound (Ba Bs : Nat) : Nat := (4 + 4) * N * Ba * 524288 + N * 1 * Bs

/-- The key of member `k` from its four hash samples. -/
theorem memberKey_eq {model : HashModel} {key : ByteArray} {k : Nat}
    {a00 a01 a10 a11 : ExactMatrix Q N 1 4}
    (h00 : hashSample model memberPrefix [.bytes memberBytes, .u64Le (Int.ofNat k),
      .bytes componentBytes, .u64Le 0] key a00)
    (h01 : hashSample model memberPrefix [.bytes memberBytes, .u64Le (Int.ofNat k),
      .bytes componentBytes, .u64Le 1] key a01)
    (h10 : hashSample model memberPrefix [.bytes memberBytes, .u64Le (Int.ofNat k),
      .bytes componentBytes, .u64Le 2] key a10)
    (h11 : hashSample model memberPrefix [.bytes memberBytes, .u64Le (Int.ofNat k),
      .bytes componentBytes, .u64Le 3] key a11) :
    memberKey model key (Int.ofNat k) = ![![a00, a01], ![a10, a11]] := by
  funext r j
  fin_cases r <;> fin_cases j
  · exact (hashSample_eq h00).symm
  · exact (hashSample_eq h01).symm
  · exact (hashSample_eq h10).symm
  · exact (hashSample_eq h11).symm

set_option maxHeartbeats 16000000 in
/-- The first level: member `0` times member `1`, an encoding of `α₀ α₁` with that public payload. -/
theorem pair {model : HashModel} {key : ByteArray} {c0 c1 : Fin 5 → ExactMatrix Q N 1 4}
    {pay : Fin 5 → ExactMatrix Q N 1 1} {i : Fin 1} {outputs}
    (h : Stage_online_eval.parallel_generatedRoot_4 Backend.backend model tape path
      GeneratedClaim.stage_2_params i (c0, key, c1, pay, ()) outputs)
    {sm : ErrorPoly N} {α : Fin 5 → ErrorPoly N} (hα : ∀ k, polyNorm (α k) ≤ 1)
    {G : ExactMatrix Q N 1 4} (hG : gadgetMatrixRuns Backend.backend 1048576 4 G)
    (ha : ∀ k : Fin 5, Rel sm (memberKey model key k)
      (fun j ↦ reducePoly Q N (α k * unitPair sm j) • G) ![c0 k, c1 k] 26)
    (hpay : ∀ k : Fin 5, pay k = fun _ _ ↦ reducePoly Q N (α k)) :
    Rel sm ![![outputs.1, outputs.2.1], ![outputs.2.2.1, outputs.2.2.2.1]]
      (fun j ↦ reducePoly Q N (α 0 * α 1 * unitPair sm j) • G)
      ![outputs.2.2.2.2.1, outputs.2.2.2.2.2.1] (productBound 26 26) ∧
      outputs.2.2.2.2.2.2.1 = fun _ _ ↦ reducePoly Q N (α 0 * α 1) := by
  obtain rfl : i = 0 := Subsingleton.elim _ _
  obtain ⟨w2, w4, w5, w6, w6_concat_1, w8, w9, w10, w11, w11_concat_1, w12, w13, w14, w15, w16, w17, w19, w20, w21, w22, w23, w24, w26, w27, w28, w29, w31, w33, w36, w37, w40, -, -, -, -, h2, -, -, h4, -, -, h5, h6_concat_1, h6, -, -, h8, -, -, h9, -, -, h10, h11_concat_1, h11, h12, -, -, h13, -, -, h14, h15, h16, h17, -, -, -, -, -, -, h19, -, -, h20, -, -, h21, h22, h23, h24, -, -, -, -, -, -, h26, -, -, -, -, -, -, h27, -, -, -, -, -, -, h28, -, -, -, -, -, -, h29, -, -, -, -, h31, -, -, h33, -, -, -, -, -, -, h36, -, -, h37, -, -, h40, hout⟩ := h
  have s0 : MxxIR.selectInt (Int.ofNat ((0 : Fin 1) : Nat)) [(0)] = Int.ofNat 0 := rfl
  have s1 : MxxIR.selectInt (Int.ofNat ((0 : Fin 1) : Nat)) [(1)] = Int.ofNat 1 := rfl
  simp only [s0, s1] at *
  obtain ⟨_, rfl⟩ := familyGet_eq h2
  obtain ⟨_, rfl⟩ := familyGet_eq h8
  obtain ⟨_, rfl⟩ := familyGet_eq h31
  obtain ⟨_, rfl⟩ := familyGet_eq h33
  obtain ⟨_, rfl⟩ := familyGet_eq h37
  obtain ⟨_, rfl⟩ := familyGet_eq h40
  have hl := ha 0
  rw [show ((0 : Fin 5) : Int) = Int.ofNat 0 from rfl, memberKey_eq h4 h9 h5 h10] at hl
  have hr := ha 1
  rw [show ((1 : Fin 5) : Int) = Int.ofNat 1 from rfl, memberKey_eq h13 h20 h14 h21] at hr
  have hpub := mul_public (q := Q) (m := 4) (m' := 4) (bound := 524288) hN hQ
    Backend.layout0 layoutQ rfl hG (hα 0) hl hr h6_concat_1 h6 h11_concat_1 h11 h12
    (Xs := ![w15, w22]) (fun j ↦ by fin_cases j; exacts [h15, h22])
    (D' := ![w16, w23]) (fun j ↦ by fin_cases j; exacts [h16, h23])
    (D := ![w17, w24]) (fun j ↦ by fin_cases j; exacts [h17, h24])
    (hpay 0) (Y := ![w29, w36]) (A' := ![![w19, w26], ![w27, w28]])
    (fun j ↦ by fin_cases j; exacts [h29, h36]) (fun j ↦ by fin_cases j; exacts [h19, h26])
    (fun j ↦ by fin_cases j; exacts [h27, h28])
    (c' := ![_, _]) (fun j ↦ by fin_cases j <;> rfl)
  subst hout
  refine ⟨hpub.target_eq (by funext j; rw [smul_smul, ← map_mul, mul_assoc]), ?_⟩
  funext r c
  simp [matrixMulScalarLeft, hpay, map_mul]

set_option maxHeartbeats 16000000 in
/-- The root: the first-level product times member `3`, a public encoding of `α₀ α₁ α₃`. -/
theorem root {model : HashModel} {key : ByteArray} {c0 c1 : Fin 5 → ExactMatrix Q N 1 4}
    {pay : Fin 5 → ExactMatrix Q N 1 1} {outputs}
    (h : Stage_online_eval.generatedRoot Backend.backend model tape path
      GeneratedClaim.stage_2_params (c0, key, c1, pay, ()) outputs)
    {sm : ErrorPoly N} {α : Fin 5 → ErrorPoly N} (hα : ∀ k, polyNorm (α k) ≤ 1)
    (hα01 : polyNorm (α 0 * α 1) ≤ 1)
    {G : ExactMatrix Q N 1 4} (hG : gadgetMatrixRuns Backend.backend 1048576 4 G)
    (ha : ∀ k : Fin 5, Rel sm (memberKey model key k)
      (fun j ↦ reducePoly Q N (α k * unitPair sm j) • G) ![c0 k, c1 k] 26)
    (hpay : ∀ k : Fin 5, pay k = fun _ _ ↦ reducePoly Q N (α k)) :
    Rel sm ![![outputs.1, outputs.2.1], ![outputs.2.2.1, outputs.2.2.2.1]]
      (fun j ↦ reducePoly Q N (α 0 * α 1 * α 3 * unitPair sm j) • G)
      ![outputs.2.2.2.2.1, outputs.2.2.2.2.2.1] (productBound (productBound 26 26) 26) := by
  obtain ⟨w, hb⟩ := h
  unfold Stage_online_eval.generatedRoot.body at hb
  obtain ⟨-, hloop, h5, h6, h7, h8_concat_1, h8, h9, h10, h11, h12_concat_1, h12, h13, h14, h15, h16, h17, h18, -, -, -, -, -, -, h20, h21, h22, h23, h24, h25, -, -, -, -, -, -, h27, -, -, -, -, -, -, h28, -, -, -, -, -, -, h29, -, -, -, -, -, -, h30, h31, h32, -, -, -, -, -, -, h35, h36, h39, hout⟩ := hb
  obtain ⟨hp, hpp⟩ := pair (hloop 0) hα hG ha hpay
  obtain ⟨_, e5⟩ := familyGetStatic_eq (k := 0) h5
  obtain ⟨_, e6⟩ := familyGetStatic_eq (k := 0) h6
  obtain ⟨_, e7⟩ := familyGetStatic_eq (k := 0) h7
  obtain ⟨_, e9⟩ := familyGetStatic_eq (k := 0) h9
  obtain ⟨_, e10⟩ := familyGetStatic_eq (k := 0) h10
  obtain ⟨_, e11⟩ := familyGetStatic_eq (k := 0) h11
  obtain ⟨_, e31⟩ := familyGetStatic_eq (k := 3) h31
  obtain ⟨_, e32⟩ := familyGetStatic_eq (k := 0) h32
  obtain ⟨_, e36⟩ := familyGetStatic_eq (k := 3) h36
  rw [e5, e6] at h8_concat_1
  rw [e7] at h8
  rw [e9, e10] at h12_concat_1
  rw [e11] at h12
  rw [e31, e32, e36] at hout
  have hr := ha 3
  rw [show ((3 : Fin 5) : Int) = Int.ofNat 3 from rfl,
    memberKey_eq (k := 3) h14 h21 h15 h22] at hr
  have hpub := mul_public (q := Q) (m := 4) (m' := 4) (bound := 524288) hN hQ
    Backend.layout0 layoutQ rfl hG hα01 hp hr h8_concat_1 h8 h12_concat_1 h12 h13
    (Xs := ![w.w_16_0, w.w_23_0]) (fun j ↦ by fin_cases j; exacts [h16, h23])
    (D' := ![w.w_17_0, w.w_24_0]) (fun j ↦ by fin_cases j; exacts [h17, h24])
    (D := ![w.w_18_0, w.w_25_0]) (fun j ↦ by fin_cases j; exacts [h18, h25])
    (payload := w.w_4_6 0) hpp (Y := ![w.w_30_0, w.w_35_0])
    (A' := ![![w.w_20_0, w.w_27_0], ![w.w_28_0, w.w_29_0]])
    (fun j ↦ by fin_cases j; exacts [h30, h35]) (fun j ↦ by fin_cases j; exacts [h20, h27])
    (fun j ↦ by fin_cases j; exacts [h28, h29])
    (c' := ![_, _]) (fun j ↦ by fin_cases j <;> rfl)
  subst hout
  exact hpub.target_eq (by funext j; rw [smul_smul, ← map_mul]; congr 2; ring)

end PublicIpProof
