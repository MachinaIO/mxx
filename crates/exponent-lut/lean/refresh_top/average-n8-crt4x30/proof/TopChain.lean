import TopRefresh
import TopLinkC
import TopColumn
import TopGlue
import TopFinal

/-!
# The refresh combination over any decoding ring

`refresh_column` for the ring `R` of a chain (`Q₁` for chains `0`–`2`, `q₀ q₁ q₃` for chain `3`).
-/

namespace RefreshTopProof

open Mxx.Primitives MxxRuntime MxxExponentLut Mxx.ExponentLut.Refresh

/-- The action constant `k = c' / d`. -/
abbrev actionConst' (c : Int) (d : Nat) : Int := c / d

set_option maxHeartbeats 4000000 in
/-- One output column of a decoder before the modulus switch. -/
theorem refresh_columnR {R : Nat} {s₀ β Qq : ExactPoly R N}
    {g c₁ G : ExactMatrix R N 1 6} {gc : ExactMatrix R N 1 (6 + 6)} (hgc : concatColumns g c₁ gc)
    {D : ExactMatrix R N (6 + 6) 6} {A₀ A₁ : ExactMatrix R N 1 6}
    (hX₀ : G * topRows D = Qq • A₀) (hX₁ : G * bottomRows D = Qq • A₁)
    {cx P : ExactMatrix R N 1 4} {Y : ExactMatrix R N 4 6} {Tx : ExactMatrix R N 1 6}
    (hY : P * Y = Qq • Tx)
    {fw e' ew : ExactMatrix R N 1 6}
    {Kg K₁ Kw : Fin 2 → ExactMatrix R N 1 6} {Kx : Fin 2 → ExactMatrix R N 1 4}
    {eg e₁ : ExactMatrix R N 1 6} {ex : ExactMatrix R N 1 4}
    (hg : g = maskPart s₀ Kg - β • G + eg) (hc₁ : c₁ = maskPart s₀ K₁ - G + e₁)
    (hcx : cx = maskPart s₀ Kx - P + ex) (hfw : fw = maskPart s₀ Kw - Qq • e' + ew)
    {b : ExactMatrix R N 1 16} {B : Fin 2 → ExactMatrix R N 1 16} {eb : ExactMatrix R N 1 16}
    (hb : b = maskPart s₀ B + eb)
    (k : Fin 6) {Kpre : ExactMatrix R N 16 1} {Ksm : Fin 2 → ExactMatrix R N 1 1}
    (hpre : ∀ i, B i * Kpre =
      (Kg i * topRows D + K₁ i * bottomRows D - Kx i * Y + Kw i) * unitCol (q := R) k + Ksm i)
    {csm S esm H : ExactMatrix R N 1 1} (hcsm : csm = maskPart s₀ Ksm - S + esm)
    {dk : ExactMatrix R N 1 1}
    (hdk : dk = (gc * D - cx * Y + fw) * unitCol (q := R) k) :
    b * Kpre - (dk + csm) - H =
      Qq • ((β • A₀ + A₁ - Tx + e') * unitCol (q := R) k) + (S - H) +
        (eb * Kpre - ((eg * topRows D + e₁ * bottomRows D - ex * Y + ew) * unitCol (q := R) k) - esm) := by
  have h := decoded_value (R := ExactPoly R N) (m := 6) (w := 4) (columns := 1) (hint := 16)
    s₀ β Qq G P Kg K₁ Kx (fun i ↦ Kw i * unitCol (q := R) k) eg e₁ ex (ew * unitCol (q := R) k)
    (topRows D * unitCol (q := R) k) (bottomRows D * unitCol (q := R) k) (Y * unitCol (q := R) k)
    (A₀ * unitCol (q := R) k) (A₁ * unitCol (q := R) k) (Tx * unitCol (q := R) k) (e' * unitCol (q := R) k)
    (by rw [← Matrix.mul_assoc, hX₀, Matrix.smul_mul]) (by rw [← Matrix.mul_assoc, hX₁, Matrix.smul_mul])
    (by rw [← Matrix.mul_assoc, hY, Matrix.smul_mul]) B eb Kpre Ksm (S - H) H esm
    (fun i ↦ by rw [hpre i]; simp only [Matrix.add_mul, Matrix.sub_mul, Matrix.mul_assoc])
  simp only at h
  have hd : dk = (maskPart s₀ Kg - β • G + eg) * (topRows D * unitCol (q := R) k) +
      (maskPart s₀ K₁ - G + e₁) * (bottomRows D * unitCol (q := R) k) -
      (maskPart s₀ Kx - P + ex) * (Y * unitCol (q := R) k) +
      (maskPart s₀ (fun i ↦ Kw i * unitCol (q := R) k) - Qq • (e' * unitCol (q := R) k) + ew * unitCol (q := R) k) := by
    rw [hdk, concatColumns_mul hgc, ← hg, ← hc₁, ← hcx, hfw]
    simp only [maskPart, Matrix.add_mul, Matrix.sub_mul, Matrix.mul_assoc, Matrix.smul_mul]
    try abel
  rw [hd, hb, hcsm]
  rw [show S - H + H = S by abel] at h
  convert h using 1
  · try abel
  · simp only [Matrix.add_mul, Matrix.sub_mul, Matrix.smul_mul, Matrix.mul_assoc, smul_add, smul_sub]
    try abel

/-- The gadget of `R` times the halves of the re-encoded digits of a two-row `z` are its rows. -/
theorem decompose_rowsR {R : Nat} (LR : RegularLayout R)
    (hLR : Backend.backend.regularLayout R N = some LR) (hxR : LR.droppedModuli = 0) (hR : 1 < R)
    (hbR : (32768 : Int) = LR.base) (hdR : (6 : Int) = LR.digitCount) (hiR : 6 = 1 * LR.digitCount)
    {z : ExactMatrix R N (1 + 1) 6} {Dz' Dz : ExactMatrix R N (6 + 6) 6}
    (hDz : gadgetDecomposeRuns Backend.backend 32768 6 z Dz')
    {bound : Nat} (hc : compactCenteredRebaseRuns bound Dz' Dz)
    {G : ExactMatrix R N 1 6} (hG : gadgetMatrixRuns Backend.backend 32768 6 G) :
    G * topRows Dz = (fun _ c ↦ z 0 c) ∧ G * bottomRows Dz = (fun _ c ↦ z 1 c) := by
  rw [compactRebase_self hR hc]
  have hz : concatRows (fun _ c ↦ z 0 c : ExactMatrix R N 1 6) (fun _ c ↦ z 1 c) z := by
    intro row column
    fin_cases row <;> rfl
  obtain ⟨Dx, hDx⟩ := gadgetDecomposeRuns_exists (inner := 6) (backend := Backend.backend)
    (base := 32768) (digits := 6) LR hLR hxR hR hbR hdR hiR (fun _ c ↦ z 0 c : ExactMatrix R N 1 6)
  obtain ⟨Dy, hDy⟩ := gadgetDecomposeRuns_exists (inner := 6) (backend := Backend.backend)
    (base := 32768) (digits := 6) LR hLR hxR hR hbR hdR hiR (fun _ c ↦ z 1 c : ExactMatrix R N 1 6)
  obtain ⟨ht, hb⟩ := decompose_stack LR hLR hxR hz hDz hDx hDy
  rw [ht, hb]
  exact ⟨gadget_mul_decompose_one hR LR hLR hxR hG hDx, gadget_mul_decompose_one hR LR hLR hxR hG hDy⟩

set_option maxHeartbeats 4000000 in
/-- The source plaintext times the action digits is `Q_q T` with `T = k x W`, `p c' = Q_q k`. -/
theorem action_mulR {R Mx p : Nat} (hRp : R = p * Mx) (hMx : 1 < Mx) (LX : RegularLayout Mx)
    (hLX : Backend.backend.regularLayout Mx N = some LX) (hxX : LX.droppedModuli = 0)
    {Qq c' k : Int} (hc : (p : Int) * c' = Qq * k)
    {g' : ExactMatrix Mx N 1 4} (hg' : gadgetMatrixRuns Backend.backend 32768 4 g')
    {g : ExactMatrix R N 1 4} (hg : centeredRebaseRuns g' g)
    {W : ErrorMatrix N 1 6} {X : ExactMatrix Mx N 1 6}
    (hX : X = reduceMatrix Mx N 1 6 (((c' : Int) : ErrorPoly N) • W))
    {Y' : ExactMatrix Mx N 4 6} (hY' : gadgetDecomposeRuns Backend.backend 32768 4 X Y')
    {Y : ExactMatrix R N 4 6} {bound : Nat} (hY : compactCenteredRebaseRuns bound Y' Y)
    (x : ErrorPoly N) :
    (reducePoly R N (((p : Nat) : Int) * x) • g) * Y =
      ((Qq : Int) : ExactPoly R N) • reduceMatrix R N 1 6 ((((k : Int) : ErrorPoly N) * x) • W) := by
  rw [gadget_mul_action_scaleUp_one hRp hMx LX hLX hxX hg' hg hY' hY x, hX, ← reduceMatrix_smul]
  funext r c
  simp only [scaleUpMatrix, reduceMatrix, Matrix.smul_apply, smul_eq_mul]
  rw [scaleUp_reducePoly hRp hMx hN, ← reducePoly_intCast, ← map_mul]
  congr 1
  rw [← mul_assoc, ← mul_assoc, ← mul_assoc]
  congr 1
  have h := congrArg (fun z : Int ↦ (z : ErrorPoly N)) hc
  simp only [Int.cast_mul, Int.cast_natCast] at h
  linear_combination (x) * h

set_option maxHeartbeats 16000000 in
/-- One output column of a decoder of a chain, from the runtime relations of the combination and
the semantic forms of its rows. -/
theorem column_valueR {R Mx qc p : Nat} {Qq c' k : Int} (LR : RegularLayout R)
    (hLR : Backend.backend.regularLayout R N = some LR) (hxR : LR.droppedModuli = 0) (hR : 1 < R)
    (hbR : (32768 : Int) = LR.base) (hdR : (6 : Int) = LR.digitCount) (hiR : 6 = 1 * LR.digitCount)
    (hRp : R = p * Mx) (hMx : 1 < Mx) (LX : RegularLayout Mx)
    (hLX : Backend.backend.regularLayout Mx N = some LX) (hxX : LX.droppedModuli = 0)
    (hc : (p : Int) * c' = Qq * k)
    {g c₁ : ExactMatrix R N 1 6} {gc : ExactMatrix R N 1 (6 + 6)}
    (hgc : concatColumns g c₁ gc)
    {A' : ExactMatrix R N (1 + 1) 6} {z : ExactMatrix R N (1 + 1) 6}
    (hz : z = matrixMulScalarLeft (matrixPolynomial [Qq]) A')
    {Dz' D : ExactMatrix R N (6 + 6) 6} (hDz : gadgetDecomposeRuns Backend.backend 32768 6 z Dz')
    (hD : compactCenteredRebaseRuns 16384 Dz' D)
    {G : ExactMatrix R N 1 6} (hG : gadgetMatrixRuns Backend.backend 32768 6 G)
    {w350 : ExactMatrix qc N 1 6} {w351 : ExactMatrix Mx N 1 6} (h351 : centeredRebaseRuns w350 w351)
    {X : ExactMatrix Mx N 1 6} (hX : X = matrixMulScalarRight w351 (matrixPolynomial [c']))
    {Y' : ExactMatrix Mx N 4 6} (hY' : gadgetDecomposeRuns Backend.backend 32768 4 X Y')
    {Y : ExactMatrix R N 4 6} (hY : compactCenteredRebaseRuns 16384 Y' Y)
    {g4' : ExactMatrix Mx N 1 4} (hg4' : gadgetMatrixRuns Backend.backend 32768 4 g4')
    {g4 : ExactMatrix R N 1 4} (hg4 : centeredRebaseRuns g4' g4)
    {cx : ExactMatrix R N 1 4} {fw : ExactMatrix R N 1 6} {comb : ExactMatrix R N 1 6}
    (hcomb : comb = matrixAdd (matrixSub (matrixMul gc D) (matrixMul cx Y)) fw)
    {a b' : Int} (ha : a.toNat < 6) {dk : ExactMatrix R N 1 1} (hslice : sliceMatrix comb 0 1 a b' dk)
    {s₀ β : ExactPoly R N} {x : ErrorPoly N}
    {Kg K₁ Kw : Fin 2 → ExactMatrix R N 1 6} {Kx : Fin 2 → ExactMatrix R N 1 4}
    {eg e₁ e' ew : ExactMatrix R N 1 6} {ex : ExactMatrix R N 1 4}
    (hg : g = maskPart s₀ Kg - β • G + eg) (hc₁ : c₁ = maskPart s₀ K₁ - G + e₁)
    (hcx : cx = maskPart s₀ Kx - reducePoly R N (((p : Nat) : Int) * x) • g4 + ex)
    (hfw : fw = maskPart s₀ Kw - ((Qq : Int) : ExactPoly R N) • e' + ew)
    {b : ExactMatrix R N 1 16} {B : Fin 2 → ExactMatrix R N 1 16} {eb : ExactMatrix R N 1 16}
    (hb : b = maskPart s₀ B + eb)
    {Kpre : ExactMatrix R N 16 1} {Ksm : Fin 2 → ExactMatrix R N 1 1}
    (hpre : ∀ i, B i * Kpre =
      (Kg i * topRows D + K₁ i * bottomRows D - Kx i * Y + Kw i) * unitCol (q := R) ⟨a.toNat, ha⟩ + Ksm i)
    {csm S esm H : ExactMatrix R N 1 1} (hcsm : csm = maskPart s₀ Ksm - S + esm)
    {out : ExactMatrix R N 1 1}
    (hout : out = matrixSub (matrixSub (matrixMul b Kpre) (matrixAdd dk csm)) H) :
    out = ((Qq : Int) : ExactPoly R N) • ((β • rowOf A' 0 + rowOf A' 1 - actionValueC k x w350 + e') *
        unitCol (q := R) ⟨a.toNat, ha⟩) + (S - H) +
      (eb * Kpre - ((eg * topRows D + e₁ * bottomRows D - ex * Y + ew) *
        unitCol (q := R) ⟨a.toNat, ha⟩) - esm) := by
  obtain ⟨hX₀, hX₁⟩ := decompose_rowsR LR hLR hxR hR hbR hdR hiR hDz hD hG
  have hzr : ∀ r : Fin (1 + 1), (fun _ c ↦ z r c : ExactMatrix R N 1 6) =
      ((Qq : Int) : ExactPoly R N) • rowOf A' r := by
    intro r
    funext i c
    rw [hz, matrixMulScalarLeft_eq]
    simp only [Matrix.smul_apply, smul_eq_mul, matrixPolynomial_single, rowOf]
  rw [hzr 0] at hX₀
  rw [hzr 1] at hX₁
  have hXw : X = reduceMatrix Mx N 1 6 (((c' : Int) : ErrorPoly N) •
      fun r c ↦ centeredIntLift (w350 r c)) := by
    rw [hX, matrixMulScalarRight_eq, centeredRebaseRuns_reduce h351, reduceMatrix_smul]
    funext r c
    simp only [Matrix.smul_apply, smul_eq_mul, matrixPolynomial_single, reduceMatrix,
      reducePoly_intCast]
  have hYx := action_mulR hRp hMx LX hLX hxX hc hg4' hg4 hXw hY' hY x
  have hdk : dk = (gc * D - cx * Y + fw) * unitCol (q := R) ⟨a.toNat, ha⟩ := by
    rw [slice_col hslice ha, hcomb]
    rfl
  have h := refresh_columnR (H := H) hgc hX₀ hX₁ hYx hg hc₁ hcx hfw hb ⟨a.toNat, ha⟩ hpre hcsm hdk
  rw [hout]
  exact h

set_option maxHeartbeats 4000000 in
/-- A decoder column is `Q_q y` plus a small integer polynomial. -/
theorem column_noiseQ {Qq : Int} {y : ExactMatrix Q1 N 1 6} (k : Fin 6) {M : ErrorMatrix N 1 1} {BM : Nat}
    (hM : CoeffBound M BM) {ebI : ErrorMatrix N 1 16} {Beb : Nat} (heb : CoeffBound ebI Beb)
    {KI : ErrorMatrix N 16 1} {Bpre : Nat} (hK : CoeffBound KI Bpre)
    {egI e1I ewI : ErrorMatrix N 1 6} {Beg Be1 Bew : Nat} (heg : CoeffBound egI Beg)
    (he1 : CoeffBound e1I Be1) (hew : CoeffBound ewI Bew)
    {X0I X1I : ErrorMatrix N 6 6} (hX0 : CoeffBound X0I 16384) (hX1 : CoeffBound X1I 16384)
    {exI : ErrorMatrix N 1 4} {Bex : Nat} (hex : CoeffBound exI Bex)
    {YI : ErrorMatrix N 4 6} (hY : CoeffBound YI 16384) {esmI : ErrorMatrix N 1 1} {Besm : Nat}
    (hesm : CoeffBound esmI Besm) {out : ExactMatrix Q1 N 1 1}
    (hout : out = ((Qq : Int) : ExactPoly Q1 N) • (y * unitCol (q := Q1) k) + reduceMatrix Q1 N 1 1 M +
      (reduceMatrix Q1 N 1 16 ebI * reduceMatrix Q1 N 16 1 KI -
        (reduceMatrix Q1 N 1 6 egI * reduceMatrix Q1 N 6 6 X0I +
          reduceMatrix Q1 N 1 6 e1I * reduceMatrix Q1 N 6 6 X1I -
          reduceMatrix Q1 N 1 4 exI * reduceMatrix Q1 N 4 6 YI + reduceMatrix Q1 N 1 6 ewI) *
          unitCol (q := Q1) k - reduceMatrix Q1 N 1 1 esmI)) :
    ∃ ν : ErrorMatrix N 1 1, CoeffBound ν (BM + 16 * N * Beb * Bpre +
        (6 * N * Beg * 16384 + 6 * N * Be1 * 16384 + 4 * N * Bex * 16384 + Bew) + Besm) ∧
      out 0 0 = ((Qq : Int) : ExactPoly Q1 N) * y 0 k + reducePoly Q1 N (ν 0 0) := by
  set ν : ErrorMatrix N 1 1 := M + ebI * KI - (egI * X0I + e1I * X1I - exI * YI + ewI) * unitColI k -
    esmI with hν
  refine ⟨ν, coeffBound_sub (coeffBound_sub (coeffBound_add hM (coeffBound_mul hN heb hK)) ?_) hesm, ?_⟩
  · exact coeffBound_col (coeffBound_add (coeffBound_sub (coeffBound_add
      (coeffBound_mul hN heg hX0) (coeffBound_mul hN he1 hX1)) (coeffBound_mul hN hex hY)) hew) k
  · have hr : reduceMatrix Q1 N 1 1 ν = reduceMatrix Q1 N 1 1 M +
        (reduceMatrix Q1 N 1 16 ebI * reduceMatrix Q1 N 16 1 KI -
          (reduceMatrix Q1 N 1 6 egI * reduceMatrix Q1 N 6 6 X0I +
            reduceMatrix Q1 N 1 6 e1I * reduceMatrix Q1 N 6 6 X1I -
            reduceMatrix Q1 N 1 4 exI * reduceMatrix Q1 N 4 6 YI + reduceMatrix Q1 N 1 6 ewI) *
            unitCol (q := Q1) k - reduceMatrix Q1 N 1 1 esmI) := by
      rw [hν, unitCol_reduce]
      simp only [reduceMatrix_add, reduceMatrix_sub, reduceMatrix_mul]
      abel
    rw [hout, add_assoc, ← hr, Matrix.add_apply, Matrix.smul_apply, mul_unitCol, smul_eq_mul]
    rfl

/-- Six `1 × 1` rows of target `t_k Q_q` side by side: a row of target `Q_q t`. -/
theorem six_rowsQ {Qq : Int} {s : ExactPoly Q1 N} {z : ExactMatrix Q1 N 1 6} {x : Fin 6 → ExactMatrix Q1 N 1 1}
    (hz : ∀ k, z 0 k = x k 0 0) {K : Fin 6 → Fin 2 → ExactMatrix Q1 N 1 1} {t : Fin 6 → ErrorPoly N}
    {e : Fin 6 → ErrorMatrix N 1 1}
    (hx : ∀ k, x k = maskPart s (K k) - reducePoly Q1 N (t k) •
      (matrixPolynomial [Qq] : ExactMatrix Q1 N 1 1) + reduceMatrix Q1 N 1 1 (e k)) :
    z = maskPart s (fun i ↦ rowMat (fun k ↦ K k i 0 0)) - ((Qq : Int) : ExactPoly Q1 N) • rowMat (fun k ↦ reducePoly Q1 N (t k)) +
      reduceMatrix Q1 N 1 6 (fun _ k ↦ e k 0 0) := by
  funext r k
  obtain rfl : r = 0 := Subsingleton.elim _ _
  rw [hz, hx]
  simp [maskPart, matrixPolynomial_single, reduceMatrix, rowMat, mul_comm]

/-- A row of target `t G` times the digits of `Q_q` has target `Q_q t`. -/
theorem row_mul_qqQ {Qq : Int} {sm : ErrorPoly N} {K : Fin 2 → ExactMatrix Q1 N 1 6} {t : ErrorPoly N}
    {gq : ExactMatrix Q1 N 1 6} (hgq : gadgetMatrixRuns Backend.backend 32768 6 gq)
    {e : ErrorMatrix N 1 6} {B : Nat} (he : CoeffBound e B) {c : ExactMatrix Q1 N 1 6}
    (hc : c = maskPart (reducePoly Q1 N sm) K - reducePoly Q1 N t • gq + reduceMatrix Q1 N 1 6 e)
    {Dq' Dq : ExactMatrix Q1 N 6 1}
    (hD' : gadgetDecomposeRuns Backend.backend 32768 6
      (matrixPolynomial [Qq] : ExactMatrix Q1 N 1 1) Dq')
    (hD : compactCenteredRebaseRuns 16384 Dq' Dq) :
    ∃ e' : ErrorMatrix N 1 1, CoeffBound e' (6 * N * B * 16384) ∧
      c * Dq = maskPart (reducePoly Q1 N sm) (fun k ↦ K k * Dq) -
        reducePoly Q1 N t • (matrixPolynomial [Qq] : ExactMatrix Q1 N 1 1) +
          reduceMatrix Q1 N 1 1 e' := by
  obtain ⟨W, hW, hWb⟩ := hD.2.2.2
  have hg := gadget_mul_decompose_one (by norm_num) Backend.layout4 layoutQ1 rfl hgq hD'
  rw [← compactRebase_self (by norm_num) hD] at hg
  refine ⟨e * W, coeffBound_mul hN he hWb, ?_⟩
  rw [hc, Matrix.add_mul, Matrix.sub_mul, maskPart_mul, Matrix.smul_mul, hg, hW,
    ← reduceMatrix_mul]

/-- The fresh columns side by side: a row of target `Q_q e'`. -/
theorem fresh_glueQ {Qq : Int} {sm : ErrorPoly N} {B : Nat} {x0 x1 x2 x3 x4 x5 : ExactMatrix Q1 N 1 1}
    {K0 K1 K2 K3 K4 K5 : Fin 2 → ExactMatrix Q1 N 1 1} {t0 t1 t2 t3 t4 t5 : ErrorPoly N}
    (h0 : ∃ e : ErrorMatrix N 1 1, CoeffBound e B ∧ x0 = maskPart (reducePoly Q1 N sm) K0 -
      reducePoly Q1 N t0 • (matrixPolynomial [Qq] : ExactMatrix Q1 N 1 1) + reduceMatrix Q1 N 1 1 e)
    (h1 : ∃ e : ErrorMatrix N 1 1, CoeffBound e B ∧ x1 = maskPart (reducePoly Q1 N sm) K1 -
      reducePoly Q1 N t1 • (matrixPolynomial [Qq] : ExactMatrix Q1 N 1 1) + reduceMatrix Q1 N 1 1 e)
    (h2 : ∃ e : ErrorMatrix N 1 1, CoeffBound e B ∧ x2 = maskPart (reducePoly Q1 N sm) K2 -
      reducePoly Q1 N t2 • (matrixPolynomial [Qq] : ExactMatrix Q1 N 1 1) + reduceMatrix Q1 N 1 1 e)
    (h3 : ∃ e : ErrorMatrix N 1 1, CoeffBound e B ∧ x3 = maskPart (reducePoly Q1 N sm) K3 -
      reducePoly Q1 N t3 • (matrixPolynomial [Qq] : ExactMatrix Q1 N 1 1) + reduceMatrix Q1 N 1 1 e)
    (h4 : ∃ e : ErrorMatrix N 1 1, CoeffBound e B ∧ x4 = maskPart (reducePoly Q1 N sm) K4 -
      reducePoly Q1 N t4 • (matrixPolynomial [Qq] : ExactMatrix Q1 N 1 1) + reduceMatrix Q1 N 1 1 e)
    (h5 : ∃ e : ErrorMatrix N 1 1, CoeffBound e B ∧ x5 = maskPart (reducePoly Q1 N sm) K5 -
      reducePoly Q1 N t5 • (matrixPolynomial [Qq] : ExactMatrix Q1 N 1 1) + reduceMatrix Q1 N 1 1 e)
    {z1 : ExactMatrix Q1 N 1 2} {z2 : ExactMatrix Q1 N 1 3} {z3 : ExactMatrix Q1 N 1 4}
    {z4 : ExactMatrix Q1 N 1 5} {z : ExactMatrix Q1 N 1 6} (c1 : concatColumns x0 x1 z1)
    (c2 : concatColumns z1 x2 z2) (c3 : concatColumns z2 x3 z3) (c4 : concatColumns z3 x4 z4)
    (c5 : concatColumns z4 x5 z) :
    ∃ ewI : ErrorMatrix N 1 6, CoeffBound ewI B ∧
      z = maskPart (reducePoly Q1 N sm) (fun i ↦ rowMat (fun k ↦ ![K0, K1, K2, K3, K4, K5] k i 0 0)) -
        ((Qq : Int) : ExactPoly Q1 N) • rowMat (fun k ↦ reducePoly Q1 N (![t0, t1, t2, t3, t4, t5] k)) +
          reduceMatrix Q1 N 1 6 ewI := by
  obtain ⟨e0, b0, h0⟩ := h0
  obtain ⟨e1, b1, h1⟩ := h1
  obtain ⟨e2, b2, h2⟩ := h2
  obtain ⟨e3, b3, h3⟩ := h3
  obtain ⟨e4, b4, h4⟩ := h4
  obtain ⟨e5, b5, h5⟩ := h5
  refine ⟨fun _ k ↦ ![e0, e1, e2, e3, e4, e5] k 0 0, fun r k i ↦ ?_, ?_⟩
  · fin_cases k
    · exact b0 0 0 i
    · exact b1 0 0 i
    · exact b2 0 0 i
    · exact b3 0 0 i
    · exact b4 0 0 i
    · exact b5 0 0 i
  · refine six_rowsQ (x := ![x0, x1, x2, x3, x4, x5]) (fun k ↦ ?_) (K := ![K0, K1, K2, K3, K4, K5])
      (t := ![t0, t1, t2, t3, t4, t5]) (e := ![e0, e1, e2, e3, e4, e5]) (fun k ↦ ?_)
    · exact (concat_six c1 c2 c3 c4 c5 k).trans (by fin_cases k <;> rfl)
    · fin_cases k
      · exact h0
      · exact h1
      · exact h2
      · exact h3
      · exact h4
      · exact h5

/-- The action digits `Y` that two stages compute from the gadget agree. -/
theorem y_uniqueC {qc Mx : Nat} {g g' : ExactMatrix Q1 N 1 6} (hg : gadgetMatrixRuns Backend.backend 32768 6 g)
    (hg' : gadgetMatrixRuns Backend.backend 32768 6 g') {r r' : ExactMatrix Q0 N 1 6}
    (hr : centeredRebaseRuns g r) (hr' : centeredRebaseRuns g' r') {m m' : ExactMatrix qc N 1 6}
    (hm : modulusReduceRuns r m) (hm' : modulusReduceRuns r' m') {c3 c1 : ExactMatrix qc N 1 1}
    {s s' t t' : ExactMatrix qc N 1 6} (hs : s = matrixMulScalarRight m c3)
    (hs' : s' = matrixMulScalarRight m' c3) (ht : t = matrixMulScalarRight s c1)
    (ht' : t' = matrixMulScalarRight s' c1) {u u' : ExactMatrix Mx N 1 6}
    (hu : centeredRebaseRuns t u) (hu' : centeredRebaseRuns t' u') {c : ExactMatrix Mx N 1 1}
    {v v' : ExactMatrix Mx N 1 6} (hv : v = matrixMulScalarRight u c)
    (hv' : v' = matrixMulScalarRight u' c) {Yz Yz' : ExactMatrix Mx N 4 6}
    (hy : gadgetDecomposeRuns Backend.backend 32768 4 v Yz)
    (hy' : gadgetDecomposeRuns Backend.backend 32768 4 v' Yz') {Y Y' : ExactMatrix Q1 N 4 6}
    (hY : compactCenteredRebaseRuns 16384 Yz Y) (hY' : compactCenteredRebaseRuns 16384 Yz' Y') :
    Y = Y' := by
  obtain rfl := gadgetMatrixRuns_unique hg hg'
  obtain rfl := centeredRebaseRuns_unique hr hr'
  obtain rfl := modulusReduceRuns_unique hm hm'
  subst hs hs' ht ht'
  obtain rfl := centeredRebaseRuns_unique hu hu'
  subst hv hv'
  exact dec_rebase_unique hy hy' hY hY'

/-- The sampling table of the mask at a chain with prime `qc`: `(0, h, 0, h)` with `2 h = qc + 1`. -/
noncomputable def maskTableC (h : Int) (k : Fin 4) : ErrorPoly N :=
  ![((0 : Int) : ErrorPoly N), ((h : Int) : ErrorPoly N), ((0 : Int) : ErrorPoly N),
    ((h : Int) : ErrorPoly N)] k

theorem sampled_maskC (h : Int) (z : Nat) :
    Part_I.sampled (maskTableC h) (Part_I.readoutExp z) = (((-1 : Int) ^ (z / 4) * h : Int) : ErrorPoly N) := by
  obtain ⟨hdiv, hmod⟩ := readoutExp_split z
  unfold Part_I.sampled
  rw [hdiv]
  have ht : maskTableC h ⟨Part_I.readoutExp z % 4, Nat.mod_lt _ (by norm_num)⟩ = ((h : Int) : ErrorPoly N) := by
    rcases hmod with hm | hm
    · rw [show (⟨Part_I.readoutExp z % 4, Nat.mod_lt _ (by norm_num)⟩ : Fin 4) = 1 from Fin.ext hm]; rfl
    · rw [show (⟨Part_I.readoutExp z % 4, Nat.mod_lt _ (by norm_num)⟩ : Fin 4) = 3 from Fin.ext hm]; rfl
  rw [ht]
  push_cast
  ring

theorem sampled_one_maskC (h : Int) : Part_I.sampled (maskTableC h) 1 = ((h : Int) : ErrorPoly N) := by
  unfold Part_I.sampled
  rw [show (⟨1 % 4, Nat.mod_lt _ (by norm_num)⟩ : Fin 4) = 1 from rfl]
  simp [maskTableC]

/-- A member's value is its `σ` modulo `qc`, when `2 h ≡ 1`. -/
theorem member_sigmaC {qc : Nat} {h : Int} (h2 : ((2 * h : Int) : ZMod qc) = 1) (z : Nat) :
    reducePoly qc N (Part_I.sampled (maskTableC h) (Part_I.readoutExp z) + Part_I.sampled (maskTableC h) 1) =
      reducePoly qc N (((1 - ((z : Int) % 8) / 4 : Int)) : ErrorPoly N) := by
  rw [sampled_maskC, sampled_one_maskC, ← Int.cast_add, reducePoly_intCast, reducePoly_intCast,
    intCast_eq_algebraMap, intCast_eq_algebraMap]
  congr 1
  have h8 : ((z : Int) % 8) / 4 = ((z / 4 : Nat) : Int) % 2 := by omega
  rw [h8]
  rcases Nat.even_or_odd (z / 4) with ⟨m, hm⟩ | ⟨m, hm⟩
  · rw [hm, show ((m + m : Nat) : Int) % 2 = 0 by omega, Even.neg_one_pow ⟨m, rfl⟩]
    push_cast at h2 ⊢
    linear_combination h2
  · rw [hm, show ((2 * m + 1 : Nat) : Int) % 2 = 1 by omega, Odd.neg_one_pow ⟨m, rfl⟩]
    push_cast
    ring

/-- The mask's value is `σ` modulo `qc`. -/
theorem mask_sigmaC {qc : Nat} {h : Int} (h2 : ((2 * h : Int) : ZMod qc) = 1)
    {s s' : Fin (8 * 5) → Int} {sel sel' : Fin 8 → Fin 5}
    (hsel : ∀ o (k : Fin 5), s (cellIndex o k) = if k = sel o then 1 else 0)
    (hsel' : ∀ o (k : Fin 5), s' (cellIndex o k) = if k = sel' o then 1 else 0) (sp : ErrorPoly N) :
    reducePoly qc N (((1 : Int) : ErrorPoly N) * (∑ o : Fin 8, Xe (n := N) o *
      (Part_I.sampled (fun k ↦ ![((0 : Int) : ErrorPoly N), ((h : Int) : ErrorPoly N), ((0 : Int) : ErrorPoly N),
        ((h : Int) : ErrorPoly N)] k) (Part_I.readoutExp (labelExp (sel' o) + labelExp (sel o))) +
       Part_I.sampled (fun k ↦ ![((0 : Int) : ErrorPoly N), ((h : Int) : ErrorPoly N), ((0 : Int) : ErrorPoly N),
        ((h : Int) : ErrorPoly N)] k) 1)) * unitPair sp 1) =
      reducePoly qc N (∑ o : Fin 8, ((sigmaOf s s' o : Int) : ErrorPoly N) * Xe (n := N) o) := by
  rw [Int.cast_one, show unitPair sp 1 = 1 from rfl, one_mul, mul_one, map_sum, map_sum]
  refine Finset.sum_congr rfl fun o _ ↦ ?_
  rw [map_mul, map_mul, mul_comm]
  congr 1
  rw [sigmaOf, zOf_eq hsel hsel']
  exact member_sigmaC h2 _

end RefreshTopProof
