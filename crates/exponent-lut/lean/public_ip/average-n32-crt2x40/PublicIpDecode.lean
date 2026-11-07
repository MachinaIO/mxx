import PublicIpOnlineEval

/-!
# The decode stage

The decode stage subtracts the mask part `s₀ A₀ⱼ + A₁ⱼ` from each row component of the output and
adds back the claimed plaintext `μ` times `sⱼ G`. For the plaintext of the output encoding, both
residuals are its error, so every coefficient of every column threshold-decodes to zero when the
error is below a quarter of `Q`, and the decoded residual count is zero.
-/

namespace PublicIpProof

open Mxx.Primitives MxxRuntime MxxExponentLut

variable {tape : SampleTape} {path : List Nat}

set_option maxHeartbeats 4000000 in
/-- Both residuals of a public encoding of `μ` against `μ` are its error. -/
theorem residuals {sm μ : ErrorPoly N} {A00 A01 A10 A11 c0 c1 G : ExactMatrix Q N 1 4} {B : Nat}
    (hrel : Rel sm ![![A00, A01], ![A10, A11]] (fun j ↦ reducePoly Q N (μ * unitPair sm j) • G)
      ![c0, c1] B)
    {mask message : ExactMatrix Q N 1 1} (hmask : mask = fun _ _ ↦ reducePoly Q N sm)
    (hmessage : message = fun _ _ ↦ reducePoly Q N μ) :
    ∃ e : Fin 2 → ErrorMatrix N 1 4, (∀ j, CoeffBound (e j) B) ∧
      matrixAdd (matrixSub c0 (matrixAdd (matrixMulScalarLeft mask A00)
          (matrixMulScalarLeft (1 : ExactMatrix Q N 1 1) A10)))
        (matrixMulScalarLeft mask (matrixMulScalarLeft message G)) = reduceMatrix Q N 1 4 (e 0) ∧
      matrixAdd (matrixSub c1 (matrixAdd (matrixMulScalarLeft mask A01)
          (matrixMulScalarLeft (1 : ExactMatrix Q N 1 1) A11)))
        (matrixMulScalarLeft (1 : ExactMatrix Q N 1 1) (matrixMulScalarLeft message G)) =
          reduceMatrix Q N 1 4 (e 1) := by
  obtain ⟨e, he, hc⟩ := hrel
  refine ⟨e, he, ?_, ?_⟩
  · have h0 : c0 = reducePoly Q N sm • A00 + A10 - reducePoly Q N (μ * sm) • G +
        reduceMatrix Q N 1 4 (e 0) := hc 0
    rw [h0, hmask, hmessage]
    funext r c
    simp only [matrixAdd, matrixSub, matrixMulScalarLeft, Matrix.add_apply, Matrix.sub_apply,
      Matrix.smul_apply, smul_eq_mul, Matrix.one_apply_eq, one_mul, map_mul]
    ring
  · have h1 : c1 = reducePoly Q N sm • A01 + A11 - reducePoly Q N (μ * 1) • G +
        reduceMatrix Q N 1 4 (e 1) := hc 1
    rw [h1, hmask, hmessage]
    funext r c
    simp only [matrixAdd, matrixSub, matrixMulScalarLeft, Matrix.add_apply, Matrix.sub_apply,
      Matrix.smul_apply, smul_eq_mul, Matrix.one_apply_eq, one_mul, mul_one]
    ring

/-- Every decoded coefficient of a column of a residual with error below a quarter of `Q` is zero. -/
theorem decode_zero {X : ExactMatrix Q N 1 4} {E : ErrorMatrix N 1 4} {B : Nat}
    (hX : X = reduceMatrix Q N 1 4 E) (hE : CoeffBound E B) (hB : 4 * B + 4 < Q)
    {r0 r1 c0 c1 : Int} {s : ExactMatrix Q N 1 1} (hs : sliceMatrix X r0 r1 c0 c1 s)
    {length position : Int} {out : Int} (hdecode : thresholdDecode 2 length position s out) :
    out = 0 := by
  obtain ⟨hr0, -, hr1, hc0, -, hc1, hrows, hcols, hval⟩ := hs
  obtain ⟨_, _, _, _, index, -, -, hout⟩ := hdecode
  simp only [Int.ofNat_eq_natCast] at hr1 hc1 hrows hcols
  obtain ⟨i, j, hij⟩ : ∃ i j, s 0 0 = reducePoly Q N (E i j) :=
    ⟨_, _, (hval 0 0 (by simp; omega) (by simp; omega)).trans (by rw [hX]; rfl)⟩
  have heabs : 4 * |(E i j).coeff index| + 4 < Q := by
    have := hE i j index
    rw [Int.abs_eq_natAbs]
    omega
  have hv := threshold_decode_value hQ hQodd (Or.inl rfl) heabs
  rw [mul_zero, neg_zero, zero_add] at hv
  rw [hout, hij, reducePoly_coeff_eq hQ hN]
  exact hv

set_option maxRecDepth 100000 in
set_option maxHeartbeats 64000000 in
/-- The decode stage outputs zero for the output encoding of the claimed plaintext `c X^e`. -/
theorem decode_value {mask : ExactMatrix Q N 1 1} {A00 A01 A10 A11 c0 c1 G : ExactMatrix Q N 1 4}
    {ec ee : Int} {out : Int}
    (h : Stage_decode.generatedRoot Backend.backend tape path GeneratedClaim.stage_3_params
      (c0, mask, A00, A10, ec, ee, c1, A01, A11, ()) out)
    {sm : ErrorPoly N} (hmask : mask = fun _ _ ↦ reducePoly Q N sm)
    (hG : gadgetMatrixRuns Backend.backend 1048576 4 G) {B : Nat}
    (hrel : Rel sm ![![A00, A01], ![A10, A11]]
      (fun j ↦ reducePoly Q N (message (n := N) ec ee * unitPair sm j) • G) ![c0, c1] B)
    (hB : 4 * B + 4 < Q) : out = 0 := by
  obtain ⟨w, hb⟩ := h
  unfold Stage_decode.generatedRoot.body at hb
  dsimp only at hb
  obtain ⟨h14, -, -, -, -, -, -, h18, -, h19, h19_1, h19_2, h19_3, h19_4, h19_5, h19_6, h19_7, h19_8, h19_9, h19_10, h19_11, h19_12, h19_13, h19_14, h19_15, h19_16, h19_17, h19_18, h19_19, h19_20, h19_21, h19_22, h19_23, h19_24, h19_25, h19_26, h19_27, h19_28, h19_29, h19_30, h19_31, -, -, -, -, -, -, h52, -, h53, h53_1, h53_2, h53_3, h53_4, h53_5, h53_6, h53_7, h53_8, h53_9, h53_10, h53_11, h53_12, h53_13, h53_14, h53_15, h53_16, h53_17, h53_18, h53_19, h53_20, h53_21, h53_22, h53_23, h53_24, h53_25, h53_26, h53_27, h53_28, h53_29, h53_30, h53_31, -, -, -, -, -, -, h86, -, h87, h87_1, h87_2, h87_3, h87_4, h87_5, h87_6, h87_7, h87_8, h87_9, h87_10, h87_11, h87_12, h87_13, h87_14, h87_15, h87_16, h87_17, h87_18, h87_19, h87_20, h87_21, h87_22, h87_23, h87_24, h87_25, h87_26, h87_27, h87_28, h87_29, h87_30, h87_31, -, -, -, -, -, -, h120, -, h121, h121_1, h121_2, h121_3, h121_4, h121_5, h121_6, h121_7, h121_8, h121_9, h121_10, h121_11, h121_12, h121_13, h121_14, h121_15, h121_16, h121_17, h121_18, h121_19, h121_20, h121_21, h121_22, h121_23, h121_24, h121_25, h121_26, h121_27, h121_28, h121_29, h121_30, h121_31, -, -, -, -, -, -, h163, -, h164, h164_1, h164_2, h164_3, h164_4, h164_5, h164_6, h164_7, h164_8, h164_9, h164_10, h164_11, h164_12, h164_13, h164_14, h164_15, h164_16, h164_17, h164_18, h164_19, h164_20, h164_21, h164_22, h164_23, h164_24, h164_25, h164_26, h164_27, h164_28, h164_29, h164_30, h164_31, -, -, -, -, -, -, h197, -, h198, h198_1, h198_2, h198_3, h198_4, h198_5, h198_6, h198_7, h198_8, h198_9, h198_10, h198_11, h198_12, h198_13, h198_14, h198_15, h198_16, h198_17, h198_18, h198_19, h198_20, h198_21, h198_22, h198_23, h198_24, h198_25, h198_26, h198_27, h198_28, h198_29, h198_30, h198_31, -, -, -, -, -, -, h231, -, h232, h232_1, h232_2, h232_3, h232_4, h232_5, h232_6, h232_7, h232_8, h232_9, h232_10, h232_11, h232_12, h232_13, h232_14, h232_15, h232_16, h232_17, h232_18, h232_19, h232_20, h232_21, h232_22, h232_23, h232_24, h232_25, h232_26, h232_27, h232_28, h232_29, h232_30, h232_31, -, -, -, -, -, -, h265, -, h266, h266_1, h266_2, h266_3, h266_4, h266_5, h266_6, h266_7, h266_8, h266_9, h266_10, h266_11, h266_12, h266_13, h266_14, h266_15, h266_16, h266_17, h266_18, h266_19, h266_20, h266_21, h266_22, h266_23, h266_24, h266_25, h266_26, h266_27, h266_28, h266_29, h266_30, h266_31, hout⟩ := hb
  have hgg := gadgetMatrixRuns_unique h14 hG
  rw [← hgg] at hrel
  obtain ⟨e, he, hr0, hr1⟩ := residuals hrel hmask (multiplyMonomial_liftInteger ec ee)
  have z266_31 := decode_zero hr1 (he 1) hB h265 h266_31
  have z266_0 := decode_zero hr1 (he 1) hB h265 h266
  have z266_1 := decode_zero hr1 (he 1) hB h265 h266_1
  have z266_2 := decode_zero hr1 (he 1) hB h265 h266_2
  have z266_3 := decode_zero hr1 (he 1) hB h265 h266_3
  have z266_4 := decode_zero hr1 (he 1) hB h265 h266_4
  have z266_5 := decode_zero hr1 (he 1) hB h265 h266_5
  have z266_6 := decode_zero hr1 (he 1) hB h265 h266_6
  have z266_7 := decode_zero hr1 (he 1) hB h265 h266_7
  have z266_8 := decode_zero hr1 (he 1) hB h265 h266_8
  have z266_9 := decode_zero hr1 (he 1) hB h265 h266_9
  have z266_10 := decode_zero hr1 (he 1) hB h265 h266_10
  have z266_11 := decode_zero hr1 (he 1) hB h265 h266_11
  have z266_12 := decode_zero hr1 (he 1) hB h265 h266_12
  have z266_13 := decode_zero hr1 (he 1) hB h265 h266_13
  have z266_14 := decode_zero hr1 (he 1) hB h265 h266_14
  have z266_15 := decode_zero hr1 (he 1) hB h265 h266_15
  have z266_16 := decode_zero hr1 (he 1) hB h265 h266_16
  have z266_17 := decode_zero hr1 (he 1) hB h265 h266_17
  have z266_18 := decode_zero hr1 (he 1) hB h265 h266_18
  have z266_19 := decode_zero hr1 (he 1) hB h265 h266_19
  have z266_20 := decode_zero hr1 (he 1) hB h265 h266_20
  have z266_21 := decode_zero hr1 (he 1) hB h265 h266_21
  have z266_22 := decode_zero hr1 (he 1) hB h265 h266_22
  have z266_23 := decode_zero hr1 (he 1) hB h265 h266_23
  have z266_24 := decode_zero hr1 (he 1) hB h265 h266_24
  have z266_25 := decode_zero hr1 (he 1) hB h265 h266_25
  have z266_26 := decode_zero hr1 (he 1) hB h265 h266_26
  have z266_27 := decode_zero hr1 (he 1) hB h265 h266_27
  have z266_28 := decode_zero hr1 (he 1) hB h265 h266_28
  have z266_29 := decode_zero hr1 (he 1) hB h265 h266_29
  have z266_30 := decode_zero hr1 (he 1) hB h265 h266_30
  have z232_7 := decode_zero hr1 (he 1) hB h231 h232_7
  have z232_8 := decode_zero hr1 (he 1) hB h231 h232_8
  have z232_9 := decode_zero hr1 (he 1) hB h231 h232_9
  have z232_10 := decode_zero hr1 (he 1) hB h231 h232_10
  have z232_11 := decode_zero hr1 (he 1) hB h231 h232_11
  have z232_12 := decode_zero hr1 (he 1) hB h231 h232_12
  have z232_13 := decode_zero hr1 (he 1) hB h231 h232_13
  have z232_14 := decode_zero hr1 (he 1) hB h231 h232_14
  have z232_15 := decode_zero hr1 (he 1) hB h231 h232_15
  have z232_16 := decode_zero hr1 (he 1) hB h231 h232_16
  have z232_17 := decode_zero hr1 (he 1) hB h231 h232_17
  have z232_18 := decode_zero hr1 (he 1) hB h231 h232_18
  have z232_19 := decode_zero hr1 (he 1) hB h231 h232_19
  have z232_20 := decode_zero hr1 (he 1) hB h231 h232_20
  have z232_21 := decode_zero hr1 (he 1) hB h231 h232_21
  have z232_22 := decode_zero hr1 (he 1) hB h231 h232_22
  have z232_23 := decode_zero hr1 (he 1) hB h231 h232_23
  have z232_24 := decode_zero hr1 (he 1) hB h231 h232_24
  have z232_25 := decode_zero hr1 (he 1) hB h231 h232_25
  have z232_26 := decode_zero hr1 (he 1) hB h231 h232_26
  have z232_27 := decode_zero hr1 (he 1) hB h231 h232_27
  have z232_28 := decode_zero hr1 (he 1) hB h231 h232_28
  have z232_29 := decode_zero hr1 (he 1) hB h231 h232_29
  have z232_30 := decode_zero hr1 (he 1) hB h231 h232_30
  have z232_31 := decode_zero hr1 (he 1) hB h231 h232_31
  have z198_15 := decode_zero hr1 (he 1) hB h197 h198_15
  have z198_16 := decode_zero hr1 (he 1) hB h197 h198_16
  have z198_17 := decode_zero hr1 (he 1) hB h197 h198_17
  have z198_18 := decode_zero hr1 (he 1) hB h197 h198_18
  have z198_19 := decode_zero hr1 (he 1) hB h197 h198_19
  have z198_20 := decode_zero hr1 (he 1) hB h197 h198_20
  have z198_21 := decode_zero hr1 (he 1) hB h197 h198_21
  have z198_22 := decode_zero hr1 (he 1) hB h197 h198_22
  have z198_23 := decode_zero hr1 (he 1) hB h197 h198_23
  have z198_24 := decode_zero hr1 (he 1) hB h197 h198_24
  have z198_25 := decode_zero hr1 (he 1) hB h197 h198_25
  have z198_26 := decode_zero hr1 (he 1) hB h197 h198_26
  have z198_27 := decode_zero hr1 (he 1) hB h197 h198_27
  have z198_28 := decode_zero hr1 (he 1) hB h197 h198_28
  have z198_29 := decode_zero hr1 (he 1) hB h197 h198_29
  have z198_30 := decode_zero hr1 (he 1) hB h197 h198_30
  have z198_31 := decode_zero hr1 (he 1) hB h197 h198_31
  have z232_0 := decode_zero hr1 (he 1) hB h231 h232
  have z232_1 := decode_zero hr1 (he 1) hB h231 h232_1
  have z232_2 := decode_zero hr1 (he 1) hB h231 h232_2
  have z232_3 := decode_zero hr1 (he 1) hB h231 h232_3
  have z232_4 := decode_zero hr1 (he 1) hB h231 h232_4
  have z232_5 := decode_zero hr1 (he 1) hB h231 h232_5
  have z232_6 := decode_zero hr1 (he 1) hB h231 h232_6
  have z164_23 := decode_zero hr1 (he 1) hB h163 h164_23
  have z164_24 := decode_zero hr1 (he 1) hB h163 h164_24
  have z164_25 := decode_zero hr1 (he 1) hB h163 h164_25
  have z164_26 := decode_zero hr1 (he 1) hB h163 h164_26
  have z164_27 := decode_zero hr1 (he 1) hB h163 h164_27
  have z164_28 := decode_zero hr1 (he 1) hB h163 h164_28
  have z164_29 := decode_zero hr1 (he 1) hB h163 h164_29
  have z164_30 := decode_zero hr1 (he 1) hB h163 h164_30
  have z164_31 := decode_zero hr1 (he 1) hB h163 h164_31
  have z198_0 := decode_zero hr1 (he 1) hB h197 h198
  have z198_1 := decode_zero hr1 (he 1) hB h197 h198_1
  have z198_2 := decode_zero hr1 (he 1) hB h197 h198_2
  have z198_3 := decode_zero hr1 (he 1) hB h197 h198_3
  have z198_4 := decode_zero hr1 (he 1) hB h197 h198_4
  have z198_5 := decode_zero hr1 (he 1) hB h197 h198_5
  have z198_6 := decode_zero hr1 (he 1) hB h197 h198_6
  have z198_7 := decode_zero hr1 (he 1) hB h197 h198_7
  have z198_8 := decode_zero hr1 (he 1) hB h197 h198_8
  have z198_9 := decode_zero hr1 (he 1) hB h197 h198_9
  have z198_10 := decode_zero hr1 (he 1) hB h197 h198_10
  have z198_11 := decode_zero hr1 (he 1) hB h197 h198_11
  have z198_12 := decode_zero hr1 (he 1) hB h197 h198_12
  have z198_13 := decode_zero hr1 (he 1) hB h197 h198_13
  have z198_14 := decode_zero hr1 (he 1) hB h197 h198_14
  have z121_31 := decode_zero hr0 (he 0) hB h120 h121_31
  have z164_0 := decode_zero hr1 (he 1) hB h163 h164
  have z164_1 := decode_zero hr1 (he 1) hB h163 h164_1
  have z164_2 := decode_zero hr1 (he 1) hB h163 h164_2
  have z164_3 := decode_zero hr1 (he 1) hB h163 h164_3
  have z164_4 := decode_zero hr1 (he 1) hB h163 h164_4
  have z164_5 := decode_zero hr1 (he 1) hB h163 h164_5
  have z164_6 := decode_zero hr1 (he 1) hB h163 h164_6
  have z164_7 := decode_zero hr1 (he 1) hB h163 h164_7
  have z164_8 := decode_zero hr1 (he 1) hB h163 h164_8
  have z164_9 := decode_zero hr1 (he 1) hB h163 h164_9
  have z164_10 := decode_zero hr1 (he 1) hB h163 h164_10
  have z164_11 := decode_zero hr1 (he 1) hB h163 h164_11
  have z164_12 := decode_zero hr1 (he 1) hB h163 h164_12
  have z164_13 := decode_zero hr1 (he 1) hB h163 h164_13
  have z164_14 := decode_zero hr1 (he 1) hB h163 h164_14
  have z164_15 := decode_zero hr1 (he 1) hB h163 h164_15
  have z164_16 := decode_zero hr1 (he 1) hB h163 h164_16
  have z164_17 := decode_zero hr1 (he 1) hB h163 h164_17
  have z164_18 := decode_zero hr1 (he 1) hB h163 h164_18
  have z164_19 := decode_zero hr1 (he 1) hB h163 h164_19
  have z164_20 := decode_zero hr1 (he 1) hB h163 h164_20
  have z164_21 := decode_zero hr1 (he 1) hB h163 h164_21
  have z164_22 := decode_zero hr1 (he 1) hB h163 h164_22
  have z121_0 := decode_zero hr0 (he 0) hB h120 h121
  have z121_1 := decode_zero hr0 (he 0) hB h120 h121_1
  have z121_2 := decode_zero hr0 (he 0) hB h120 h121_2
  have z121_3 := decode_zero hr0 (he 0) hB h120 h121_3
  have z121_4 := decode_zero hr0 (he 0) hB h120 h121_4
  have z121_5 := decode_zero hr0 (he 0) hB h120 h121_5
  have z121_6 := decode_zero hr0 (he 0) hB h120 h121_6
  have z121_7 := decode_zero hr0 (he 0) hB h120 h121_7
  have z121_8 := decode_zero hr0 (he 0) hB h120 h121_8
  have z121_9 := decode_zero hr0 (he 0) hB h120 h121_9
  have z121_10 := decode_zero hr0 (he 0) hB h120 h121_10
  have z121_11 := decode_zero hr0 (he 0) hB h120 h121_11
  have z121_12 := decode_zero hr0 (he 0) hB h120 h121_12
  have z121_13 := decode_zero hr0 (he 0) hB h120 h121_13
  have z121_14 := decode_zero hr0 (he 0) hB h120 h121_14
  have z121_15 := decode_zero hr0 (he 0) hB h120 h121_15
  have z121_16 := decode_zero hr0 (he 0) hB h120 h121_16
  have z121_17 := decode_zero hr0 (he 0) hB h120 h121_17
  have z121_18 := decode_zero hr0 (he 0) hB h120 h121_18
  have z121_19 := decode_zero hr0 (he 0) hB h120 h121_19
  have z121_20 := decode_zero hr0 (he 0) hB h120 h121_20
  have z121_21 := decode_zero hr0 (he 0) hB h120 h121_21
  have z121_22 := decode_zero hr0 (he 0) hB h120 h121_22
  have z121_23 := decode_zero hr0 (he 0) hB h120 h121_23
  have z121_24 := decode_zero hr0 (he 0) hB h120 h121_24
  have z121_25 := decode_zero hr0 (he 0) hB h120 h121_25
  have z121_26 := decode_zero hr0 (he 0) hB h120 h121_26
  have z121_27 := decode_zero hr0 (he 0) hB h120 h121_27
  have z121_28 := decode_zero hr0 (he 0) hB h120 h121_28
  have z121_29 := decode_zero hr0 (he 0) hB h120 h121_29
  have z121_30 := decode_zero hr0 (he 0) hB h120 h121_30
  have z87_7 := decode_zero hr0 (he 0) hB h86 h87_7
  have z87_8 := decode_zero hr0 (he 0) hB h86 h87_8
  have z87_9 := decode_zero hr0 (he 0) hB h86 h87_9
  have z87_10 := decode_zero hr0 (he 0) hB h86 h87_10
  have z87_11 := decode_zero hr0 (he 0) hB h86 h87_11
  have z87_12 := decode_zero hr0 (he 0) hB h86 h87_12
  have z87_13 := decode_zero hr0 (he 0) hB h86 h87_13
  have z87_14 := decode_zero hr0 (he 0) hB h86 h87_14
  have z87_15 := decode_zero hr0 (he 0) hB h86 h87_15
  have z87_16 := decode_zero hr0 (he 0) hB h86 h87_16
  have z87_17 := decode_zero hr0 (he 0) hB h86 h87_17
  have z87_18 := decode_zero hr0 (he 0) hB h86 h87_18
  have z87_19 := decode_zero hr0 (he 0) hB h86 h87_19
  have z87_20 := decode_zero hr0 (he 0) hB h86 h87_20
  have z87_21 := decode_zero hr0 (he 0) hB h86 h87_21
  have z87_22 := decode_zero hr0 (he 0) hB h86 h87_22
  have z87_23 := decode_zero hr0 (he 0) hB h86 h87_23
  have z87_24 := decode_zero hr0 (he 0) hB h86 h87_24
  have z87_25 := decode_zero hr0 (he 0) hB h86 h87_25
  have z87_26 := decode_zero hr0 (he 0) hB h86 h87_26
  have z87_27 := decode_zero hr0 (he 0) hB h86 h87_27
  have z87_28 := decode_zero hr0 (he 0) hB h86 h87_28
  have z87_29 := decode_zero hr0 (he 0) hB h86 h87_29
  have z87_30 := decode_zero hr0 (he 0) hB h86 h87_30
  have z87_31 := decode_zero hr0 (he 0) hB h86 h87_31
  have z53_15 := decode_zero hr0 (he 0) hB h52 h53_15
  have z53_16 := decode_zero hr0 (he 0) hB h52 h53_16
  have z53_17 := decode_zero hr0 (he 0) hB h52 h53_17
  have z53_18 := decode_zero hr0 (he 0) hB h52 h53_18
  have z53_19 := decode_zero hr0 (he 0) hB h52 h53_19
  have z53_20 := decode_zero hr0 (he 0) hB h52 h53_20
  have z53_21 := decode_zero hr0 (he 0) hB h52 h53_21
  have z53_22 := decode_zero hr0 (he 0) hB h52 h53_22
  have z53_23 := decode_zero hr0 (he 0) hB h52 h53_23
  have z53_24 := decode_zero hr0 (he 0) hB h52 h53_24
  have z53_25 := decode_zero hr0 (he 0) hB h52 h53_25
  have z53_26 := decode_zero hr0 (he 0) hB h52 h53_26
  have z53_27 := decode_zero hr0 (he 0) hB h52 h53_27
  have z53_28 := decode_zero hr0 (he 0) hB h52 h53_28
  have z53_29 := decode_zero hr0 (he 0) hB h52 h53_29
  have z53_30 := decode_zero hr0 (he 0) hB h52 h53_30
  have z53_31 := decode_zero hr0 (he 0) hB h52 h53_31
  have z87_0 := decode_zero hr0 (he 0) hB h86 h87
  have z87_1 := decode_zero hr0 (he 0) hB h86 h87_1
  have z87_2 := decode_zero hr0 (he 0) hB h86 h87_2
  have z87_3 := decode_zero hr0 (he 0) hB h86 h87_3
  have z87_4 := decode_zero hr0 (he 0) hB h86 h87_4
  have z87_5 := decode_zero hr0 (he 0) hB h86 h87_5
  have z87_6 := decode_zero hr0 (he 0) hB h86 h87_6
  have z19_23 := decode_zero hr0 (he 0) hB h18 h19_23
  have z19_24 := decode_zero hr0 (he 0) hB h18 h19_24
  have z19_25 := decode_zero hr0 (he 0) hB h18 h19_25
  have z19_26 := decode_zero hr0 (he 0) hB h18 h19_26
  have z19_27 := decode_zero hr0 (he 0) hB h18 h19_27
  have z19_28 := decode_zero hr0 (he 0) hB h18 h19_28
  have z19_29 := decode_zero hr0 (he 0) hB h18 h19_29
  have z19_30 := decode_zero hr0 (he 0) hB h18 h19_30
  have z19_31 := decode_zero hr0 (he 0) hB h18 h19_31
  have z53_0 := decode_zero hr0 (he 0) hB h52 h53
  have z53_1 := decode_zero hr0 (he 0) hB h52 h53_1
  have z53_2 := decode_zero hr0 (he 0) hB h52 h53_2
  have z53_3 := decode_zero hr0 (he 0) hB h52 h53_3
  have z53_4 := decode_zero hr0 (he 0) hB h52 h53_4
  have z53_5 := decode_zero hr0 (he 0) hB h52 h53_5
  have z53_6 := decode_zero hr0 (he 0) hB h52 h53_6
  have z53_7 := decode_zero hr0 (he 0) hB h52 h53_7
  have z53_8 := decode_zero hr0 (he 0) hB h52 h53_8
  have z53_9 := decode_zero hr0 (he 0) hB h52 h53_9
  have z53_10 := decode_zero hr0 (he 0) hB h52 h53_10
  have z53_11 := decode_zero hr0 (he 0) hB h52 h53_11
  have z53_12 := decode_zero hr0 (he 0) hB h52 h53_12
  have z53_13 := decode_zero hr0 (he 0) hB h52 h53_13
  have z53_14 := decode_zero hr0 (he 0) hB h52 h53_14
  have z19_0 := decode_zero hr0 (he 0) hB h18 h19
  have z19_1 := decode_zero hr0 (he 0) hB h18 h19_1
  have z19_2 := decode_zero hr0 (he 0) hB h18 h19_2
  have z19_3 := decode_zero hr0 (he 0) hB h18 h19_3
  have z19_4 := decode_zero hr0 (he 0) hB h18 h19_4
  have z19_5 := decode_zero hr0 (he 0) hB h18 h19_5
  have z19_6 := decode_zero hr0 (he 0) hB h18 h19_6
  have z19_7 := decode_zero hr0 (he 0) hB h18 h19_7
  have z19_8 := decode_zero hr0 (he 0) hB h18 h19_8
  have z19_9 := decode_zero hr0 (he 0) hB h18 h19_9
  have z19_10 := decode_zero hr0 (he 0) hB h18 h19_10
  have z19_11 := decode_zero hr0 (he 0) hB h18 h19_11
  have z19_12 := decode_zero hr0 (he 0) hB h18 h19_12
  have z19_13 := decode_zero hr0 (he 0) hB h18 h19_13
  have z19_14 := decode_zero hr0 (he 0) hB h18 h19_14
  have z19_15 := decode_zero hr0 (he 0) hB h18 h19_15
  have z19_16 := decode_zero hr0 (he 0) hB h18 h19_16
  have z19_17 := decode_zero hr0 (he 0) hB h18 h19_17
  have z19_18 := decode_zero hr0 (he 0) hB h18 h19_18
  have z19_19 := decode_zero hr0 (he 0) hB h18 h19_19
  have z19_20 := decode_zero hr0 (he 0) hB h18 h19_20
  have z19_21 := decode_zero hr0 (he 0) hB h18 h19_21
  have z19_22 := decode_zero hr0 (he 0) hB h18 h19_22
  rw [hout]
  rw [z266_31, z266_0, z266_1, z266_2, z266_3, z266_4, z266_5, z266_6, z266_7, z266_8, z266_9, z266_10, z266_11, z266_12, z266_13, z266_14]
  rw [z266_15, z266_16, z266_17, z266_18, z266_19, z266_20, z266_21, z266_22, z266_23, z266_24, z266_25, z266_26, z266_27, z266_28, z266_29, z266_30]
  rw [z232_7, z232_8, z232_9, z232_10, z232_11, z232_12, z232_13, z232_14, z232_15, z232_16, z232_17, z232_18, z232_19, z232_20, z232_21, z232_22]
  rw [z232_23, z232_24, z232_25, z232_26, z232_27, z232_28, z232_29, z232_30, z232_31, z198_15, z198_16, z198_17, z198_18, z198_19, z198_20, z198_21]
  rw [z198_22, z198_23, z198_24, z198_25, z198_26, z198_27, z198_28, z198_29, z198_30, z198_31, z232_0, z232_1, z232_2, z232_3, z232_4, z232_5]
  rw [z232_6, z164_23, z164_24, z164_25, z164_26, z164_27, z164_28, z164_29, z164_30, z164_31, z198_0, z198_1, z198_2, z198_3, z198_4, z198_5]
  rw [z198_6, z198_7, z198_8, z198_9, z198_10, z198_11, z198_12, z198_13, z198_14, z121_31, z164_0, z164_1, z164_2, z164_3, z164_4, z164_5]
  rw [z164_6, z164_7, z164_8, z164_9, z164_10, z164_11, z164_12, z164_13, z164_14, z164_15, z164_16, z164_17, z164_18, z164_19, z164_20, z164_21]
  rw [z164_22, z121_0, z121_1, z121_2, z121_3, z121_4, z121_5, z121_6, z121_7, z121_8, z121_9, z121_10, z121_11, z121_12, z121_13, z121_14]
  rw [z121_15, z121_16, z121_17, z121_18, z121_19, z121_20, z121_21, z121_22, z121_23, z121_24, z121_25, z121_26, z121_27, z121_28, z121_29, z121_30]
  rw [z87_7, z87_8, z87_9, z87_10, z87_11, z87_12, z87_13, z87_14, z87_15, z87_16, z87_17, z87_18, z87_19, z87_20, z87_21, z87_22]
  rw [z87_23, z87_24, z87_25, z87_26, z87_27, z87_28, z87_29, z87_30, z87_31, z53_15, z53_16, z53_17, z53_18, z53_19, z53_20, z53_21]
  rw [z53_22, z53_23, z53_24, z53_25, z53_26, z53_27, z53_28, z53_29, z53_30, z53_31, z87_0, z87_1, z87_2, z87_3, z87_4, z87_5]
  rw [z87_6, z19_23, z19_24, z19_25, z19_26, z19_27, z19_28, z19_29, z19_30, z19_31, z53_0, z53_1, z53_2, z53_3, z53_4, z53_5]
  rw [z53_6, z53_7, z53_8, z53_9, z53_10, z53_11, z53_12, z53_13, z53_14, z19_0, z19_1, z19_2, z19_3, z19_4, z19_5, z19_6]
  rw [z19_7, z19_8, z19_9, z19_10, z19_11, z19_12, z19_13, z19_14, z19_15, z19_16, z19_17, z19_18, z19_19, z19_20, z19_21, z19_22]
  norm_num

end PublicIpProof
