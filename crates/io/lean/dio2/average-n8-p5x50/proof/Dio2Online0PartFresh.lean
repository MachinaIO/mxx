import Dio2PartRows
import Dio2Online0Part
import Gen_W_Stage_online_0_Scopes
import GenFinalFresh

/-!
# The fresh and smudge rows of a refresh part of step 0

Refresh part `i` of step 0 (`parallel_generatedRoot_1281`) builds, for every prime `p` of `Q_0`,

* the **fresh row**: the `25` cells `25 i + c` of the fresh column total's `c₁` row (`900` cells)
  sliced out, times the digits of the block-diagonal gadget `diag(G, …, G)` and then, per cell,
  times the digits of `Q_q` (`sum.columns.slice-step`): `maskPart s Kf - Q_q • E + ef` with
  `E_c` the cell's target scalar;
* the **smudge row**: the `25` cells `125 i + 25 p + c` of the smudge column total's `c₁` row
  (`4500` cells) times the digits of `diag(h, …, h)` (`h` the `1 × 1` input):
  `maskPart s Ks - (t_c h)_c + es`.

These are the forms `hf` and `hs` of `online0_part_prime0 … 4` (`Dio2Online0PartPrimes`), proved
from the cell forms of the two column totals (`Dio2Online0Fresh`).
-/

set_option maxHeartbeats 32000000
set_option maxRecDepth 16384

namespace Dio2Proof

open Mxx.Primitives MxxRuntime MxxExponentLut Mxx.ExponentLut.Refresh RefreshTopProof
  RefreshTopProof.Gen RefreshTopProof.GenInst RefreshTopProof.GenFinal GenDecode

section Generic

variable {n : Nat}

/-- A matrix re-encoded at its own modulus and scaled by one on the right is itself (any rows). -/
theorem rebase_one_rows {q r m : Nat} (hq : 1 < q) (hn : 0 < n) {G G' : ExactMatrix q n r m}
    (h : centeredRebaseRuns G G') : matrixMulScalarRight G' (matrixPolynomial [(1 : Int)]) = G := by
  have e : (fun row column ↦ reducePoly q n (centeredIntLift (G row column))) = G :=
    reduceMatrix_centeredIntLift hq hn G
  rw [centeredRebaseRuns_reduce h, e]
  funext r c
  simp [matrixMulScalarRight, matrixPolynomial]

/-- The key row of a row of cells: entry `l` is entry `l mod m` of the key `k` of cell `l / m`. -/
def cellsKey {q C m : Nat} (K : Fin C → Fin 2 → ExactMatrix q n 1 m) (k : Fin 2) :
    ExactMatrix q n 1 (C * m) :=
  fun _ l ↦ K (divFin l) k 0 (modFin l)

/-- Block `c` of the row of blocks `(t_c x)_c` is `t_c x`. -/
theorem blk_blockRow {q C m : Nat} (t : Fin C → ExactPoly q n) (x : ExactMatrix q n 1 m)
    (c : Fin C) : blk (blockRow t x) c = t c • x := by
  funext r l
  have hr : r = 0 := Subsingleton.elim _ _
  subst hr
  have hm : 0 < m := Nat.pos_of_ne_zero (by rintro rfl; exact l.elim0)
  have hd : (c.val * m + l.val) / m = c.val := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ hm, Nat.div_eq_of_lt l.isLt, zero_add]
  have hmd : (c.val * m + l.val) % m = l.val := by
    rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt l.isLt]
  simp only [blk, blockRow, Matrix.smul_apply, smul_eq_mul]
  congr 2
  · exact Fin.ext hd
  · exact Fin.ext hmd

/-- **The cells of a slice.** A one-row slice of `C'` blocks at block `k₀` of a row of cells
`s K_c - w_c G + e_c` has the cells `k₀ + c`. -/
theorem slice_cells {q C C' m : Nat} {z : ExactMatrix q n 1 (C * m)} {s₀ e₀ : Int}
    {x : ExactMatrix q n 1 (C' * m)} (h : sliceMatrix z 0 1 s₀ e₀ x) (k₀ : Nat)
    (hs : s₀ = (m : Int) * (k₀ : Int)) (hk : k₀ + C' ≤ C) {s : ExactPoly q n}
    {K : Fin C → Fin 2 → ExactMatrix q n 1 m} {w : Fin C → ErrorPoly n} {G : ExactMatrix q n 1 m}
    {e : Fin C → ErrorMatrix n 1 m}
    (hz : ∀ cc, blk z cc = maskPart s (K cc) - reducePoly q n (w cc) • G + reduceMatrix q n 1 m (e cc))
    (c : Fin C') :
    blk x c = maskPart s (K ⟨k₀ + c.val, by omega⟩) - reducePoly q n (w ⟨k₀ + c.val, by omega⟩) • G +
      reduceMatrix q n 1 m (e ⟨k₀ + c.val, by omega⟩) := by
  rw [slice_blocks h k₀ hs c (by omega), hz]

set_option maxHeartbeats 4000000 in
/-- **A row of cells times the digits of a block-diagonal matrix.** Cells
`blk x c = s K_c - w_c G + e_c` times the digits `D` of `diag(x_b, …, x_b)` give
`s (K D) - (w_c x_b)_c + e'`. -/
theorem cells_digits {q m C b B bound : Nat} (hq : 1 < q) (hn : 0 < n) {backend : BackendContext}
    {base digits : Int} (layout : RegularLayout q) (hlayout : backend.regularLayout q n = some layout)
    (hexact : layout.droppedModuli = 0) {G : ExactMatrix q n 1 m}
    (hG : gadgetMatrixRuns backend base digits G) {x : ExactMatrix q n 1 (C * m)} {s : ExactPoly q n}
    {K : Fin C → Fin 2 → ExactMatrix q n 1 m} {w : Fin C → ErrorPoly n} {e : Fin C → ErrorMatrix n 1 m}
    (hx : ∀ c, blk x c = maskPart s (K c) - reducePoly q n (w c) • G + reduceMatrix q n 1 m (e c))
    (he : ∀ c, CoeffBound (e c) B)
    {xb : ExactMatrix q n 1 b} {X : ExactMatrix q n C (C * b)} (hX : IsDiag xb X)
    {D' D : ExactMatrix q n (C * m) (C * b)} (hD' : gadgetDecomposeRuns backend base digits X D')
    (hD : compactCenteredRebaseRuns bound D' D) :
    ∃ e' : ErrorMatrix n 1 (C * b), CoeffBound e' ((C * m) * n * B * (base.toNat / 2)) ∧
      x * D = maskPart s (fun k ↦ cellsKey K k * D) -
        blockRow (fun c ↦ reducePoly q n (w c)) xb + reduceMatrix q n 1 (C * b) e' := by
  have hy := row_of_cells (t := fun c ↦ reducePoly q n (w c)) hx
  exact part_block_row hq hn layout hlayout hexact hG hy (fun _ l i ↦ he _ 0 _ i) hX hD' hD

set_option maxHeartbeats 8000000 in
/-- **The fresh row of a part** (generic). Cells `blk x c = s K_c - w_c G + e_c` times the digits
`D` of `diag(G, …, G)`, then block `c` times the digits of `Q_q` (`hblk`, the slice column loop):
the row is `s Kf - Q_q • (w_c)_c + ef`. -/
theorem fresh_cells_gen {q m C B bound bound' : Nat} (hq : 1 < q) (hn : 0 < n)
    {backend : BackendContext} {base digits : Int} (layout : RegularLayout q)
    (hlayout : backend.regularLayout q n = some layout) (hexact : layout.droppedModuli = 0)
    {G : ExactMatrix q n 1 m} (hG : gadgetMatrixRuns backend base digits G)
    {x : ExactMatrix q n 1 (C * m)} {s : ExactPoly q n}
    {K : Fin C → Fin 2 → ExactMatrix q n 1 m} {w : Fin C → ErrorPoly n} {e : Fin C → ErrorMatrix n 1 m}
    (hx : ∀ c, blk x c = maskPart s (K c) - reducePoly q n (w c) • G + reduceMatrix q n 1 m (e c))
    (he : ∀ c, CoeffBound (e c) B)
    {X : ExactMatrix q n C (C * m)} (hX : IsDiag G X)
    {D' D : ExactMatrix q n (C * m) (C * m)} (hD' : gadgetDecomposeRuns backend base digits X D')
    (hD : compactCenteredRebaseRuns bound D' D)
    {Qq : Int} {Dq' Dq : ExactMatrix q n m 1}
    (hDq' : gadgetDecomposeRuns backend base digits (matrixPolynomial [Qq]) Dq')
    (hDq : compactCenteredRebaseRuns bound' Dq' Dq)
    {final : ExactMatrix q n (1 * 1) (C * 1)}
    (hblk : ∀ cc, blk final cc = matrixMul (blk (x * D) cc) Dq) :
    ∃ ef : ErrorMatrix n 1 (C * 1),
      CoeffBound ef (m * n * ((C * m) * n * B * (base.toNat / 2)) * (base.toNat / 2)) ∧
      final = maskPart s (fun k _ l ↦ (blk (cellsKey K k * D) (RefreshTopProof.GenFinal.colOf l) * Dq) 0 0) -
        ((Qq : Int) : ExactPoly q n) • reduceMatrix q n 1 (C * 1) (fun _ l ↦ w (RefreshTopProof.GenFinal.colOf l)) +
        reduceMatrix q n 1 (C * 1) ef := by
  obtain ⟨e', he', hy⟩ := cells_digits hq hn layout hlayout hexact hG hx he hX hD' hD
  have hDD : Dq = Dq' := compactRebase_self_gen hn hq hDq
  subst hDD
  have hGD : G * Dq = matrixPolynomial [Qq] := gadget_mul_decompose hq hn layout hlayout hexact hG hDq'
  obtain ⟨d, hd, hdb⟩ := gadgetDecomposeRuns_small hq hn hDq'
  let ec : Fin C → ErrorMatrix n 1 m := fun c _ k ↦
    e' 0 ⟨c.val * m + k.val, by have := c.isLt; have := k.isLt; nlinarith⟩
  have hb : ∀ c, blk (x * D) c = maskPart s (fun k ↦ blk (cellsKey K k * D) c) -
      reducePoly q n (w c) • G + reduceMatrix q n 1 m (ec c) := by
    intro c
    rw [hy, ← blk_blockRow (fun c ↦ reducePoly q n (w c)) G c]
    funext r l
    obtain rfl : r = 0 := Subsingleton.elim _ _
    rfl
  refine ⟨fun _ l ↦ (ec (RefreshTopProof.GenFinal.colOf l) * d) 0 0,
    fun _ l i ↦ coeffBound_mul hn (fun _ k j ↦ he' _ _ j) hdb 0 0 i, ?_⟩
  funext r l
  have hr : r = 0 := Fin.ext (by have := r.isLt; omega)
  subst hr
  rw [entry_blk final l, hblk, matrixMul, hb, Matrix.add_mul, Matrix.sub_mul, maskPart_mul,
    Matrix.smul_mul, hGD, hd, ← reduceMatrix_mul]
  simp only [maskPart, Matrix.add_apply, Matrix.sub_apply, Matrix.smul_apply, smul_eq_mul,
    reduceMatrix, matrixPolynomial_single]
  ring

/-- **The smudge row of a part** (generic). Cells `blk x c = s K_c - w_c G + e_c` times the digits
`D` of `diag(h, …, h)` (`h` a `1 × 1` constant): `s (K D) - (w_c h)_c + es`. -/
theorem smudge_cells_gen {q m C B bound : Nat} (hq : 1 < q) (hn : 0 < n)
    {backend : BackendContext} {base digits : Int} (layout : RegularLayout q)
    (hlayout : backend.regularLayout q n = some layout) (hexact : layout.droppedModuli = 0)
    {G : ExactMatrix q n 1 m} (hG : gadgetMatrixRuns backend base digits G)
    {x : ExactMatrix q n 1 (C * m)} {s : ExactPoly q n}
    {K : Fin C → Fin 2 → ExactMatrix q n 1 m} {w : Fin C → ErrorPoly n} {e : Fin C → ErrorMatrix n 1 m}
    (hx : ∀ c, blk x c = maskPart s (K c) - reducePoly q n (w c) • G + reduceMatrix q n 1 m (e c))
    (he : ∀ c, CoeffBound (e c) B) {h : ExactPoly q n}
    {X : ExactMatrix q n C (C * 1)} (hX : IsDiag (fun _ _ ↦ h : ExactMatrix q n 1 1) X)
    {D' D : ExactMatrix q n (C * m) (C * 1)} (hD' : gadgetDecomposeRuns backend base digits X D')
    (hD : compactCenteredRebaseRuns bound D' D) :
    ∃ es : ErrorMatrix n 1 (C * 1), CoeffBound es ((C * m) * n * B * (base.toNat / 2)) ∧
      x * D = maskPart s (fun k ↦ cellsKey K k * D) -
        Matrix.of (fun _ l ↦ reducePoly q n (w (RefreshTopProof.GenFinal.colOf l)) * h) + reduceMatrix q n 1 (C * 1) es := by
  obtain ⟨es, hes, hy⟩ := cells_digits hq hn layout hlayout hexact hG hx he hX hD' hD
  refine ⟨es, hes, ?_⟩
  rw [hy]
  congr 2
  funext r l
  simp only [blockRow, divFin, RefreshTopProof.GenFinal.colOf, Nat.div_one, Matrix.of_apply]

end Generic

section Instances

open W_Stage_online_0

set_option maxHeartbeats 16000000 in
/-- **The fresh row of prime 0 of refresh part `i` of step 0** (`w_45`, the form `hf` of
`online0_part_prime0`): the cells `25 i + c` of the fresh column total's `c₁` row (`w_29`)
times the digits of `diag(G, …, G)` and, per cell, of `Q_q`. -/
theorem online0_part_fresh0 {backend : BackendContext} {hashModel : HashModel} {params : Params}
    {i : Nat} (ctx : parallel_generatedRoot_1281.Ctx backend hashModel params i) {outputs : _}
    (F : parallel_generatedRoot_1281.Facts ctx outputs) (hi : i < 36)
    (hq1 : 1 < modulus_0) (L : RegularLayout modulus_0) (hL : backend.regularLayout modulus_0 8 = some L)
    (hLx : L.droppedModuli = 0)
    (hG : gadgetMatrixRuns backend 4096 25 (parallel_generatedRoot_1281.w_31_0 ctx))
    {s : ExactPoly modulus_0 8} {K : Fin 900 → Fin 2 → ExactMatrix modulus_0 8 1 25}
    {t : Fin 900 → ErrorPoly 8} {e : Fin 900 → ErrorMatrix 8 1 25} {B : Nat}
    (hz : ∀ cc, blk (C := 900) (m := 25) (parallel_generatedRoot_1281.w_29_0 ctx) cc =
      maskPart s (K cc) - reducePoly modulus_0 8 (t cc) • (parallel_generatedRoot_1281.w_31_0 ctx) + reduceMatrix modulus_0 8 1 25 (e cc))
    (he : ∀ cc, CoeffBound (e cc) B) :
    ∃ ef : ErrorMatrix 8 1 25, CoeffBound ef (25 * 8 * ((25 * 25) * 8 * B * 2048) * 2048) ∧
      (parallel_generatedRoot_1281.w_45_0 ctx) = maskPart s (fun k _ l ↦
          (blk (C := 25) (m := 25) (cellsKey (fun c : Fin 25 ↦ K ⟨25 * i + c.val, by omega⟩) k * (parallel_generatedRoot_1281.w_37_0 ctx))
            (RefreshTopProof.GenFinal.colOf (C := 25) l) * (parallel_generatedRoot_1281.w_42_0 ctx)) 0 0) -
        ((1606938044254428791916078459060479968194539524630727034176641 : Int) : ExactPoly modulus_0 8) •
          reduceMatrix modulus_0 8 1 25 (fun _ l ↦ t ⟨25 * i + l.val, by omega⟩) +
        reduceMatrix modulus_0 8 1 25 ef := by
  have d0 := isDiag_self (parallel_generatedRoot_1281.w_31_0 ctx)
  have d1 := isDiag_step (k := 1) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d0 F.c0.h31_0
  have d2 := isDiag_step (k := 2) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d1 F.c0.h31_0_1
  have d3 := isDiag_step (k := 3) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d2 F.c0.h31_0_2
  have d4 := isDiag_step (k := 4) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d3 F.c0.h31_0_3
  have d5 := isDiag_step (k := 5) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d4 F.c0.h31_0_4
  have d6 := isDiag_step (k := 6) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d5 F.c0.h31_0_5
  have d7 := isDiag_step (k := 7) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d6 F.c0.h31_0_6
  have d8 := isDiag_step (k := 8) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d7 F.c0.h31_0_7
  have d9 := isDiag_step (k := 9) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d8 F.c0.h31_0_8
  have d10 := isDiag_step (k := 10) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d9 F.c0.h31_0_9
  have d11 := isDiag_step (k := 11) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d10 F.c0.h31_0_10
  have d12 := isDiag_step (k := 12) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d11 F.c0.h31_0_11
  have d13 := isDiag_step (k := 13) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d12 F.c0.h31_0_12
  have d14 := isDiag_step (k := 14) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d13 F.c1.h31_0_13
  have d15 := isDiag_step (k := 15) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d14 F.c1.h31_0_14
  have d16 := isDiag_step (k := 16) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d15 F.c1.h31_0_15
  have d17 := isDiag_step (k := 17) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d16 F.c1.h31_0_16
  have d18 := isDiag_step (k := 18) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d17 F.c1.h31_0_17
  have d19 := isDiag_step (k := 19) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d18 F.c1.h31_0_18
  have d20 := isDiag_step (k := 20) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d19 F.c1.h31_0_19
  have d21 := isDiag_step (k := 21) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d20 F.c1.h31_0_20
  have d22 := isDiag_step (k := 22) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d21 F.c1.h31_0_21
  have d23 := isDiag_step (k := 23) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d22 F.c1.h31_0_22
  have d24 := isDiag_step (k := 24) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d23 F.c1.h32_0
  have hX' : (parallel_generatedRoot_1281.w_35_0 ctx) = (parallel_generatedRoot_1281.w_32_0 ctx) := rebase_one_rows hq1 (by norm_num) F.c1.h33_0
  have hdig := F.c1.h36_0
  rw [hX'] at hdig
  have hcells := slice_cells (C := 900) (C' := 25) (m := 25) F.c0.h30_0 (25 * i)
    (by simp only [Int.ofNat_eq_natCast]; push_cast; ring) (by omega) hz
  have hblk := slice_column_total (C := 25) (m := 25) (by norm_num) F.c1.h39_0 rfl F.c1.h44_0 F.c1.h45_0
    (fun _ _ _ hb ↦ sequential_parallel_generatedRoot_1281_45.sem hb)
  obtain ⟨ef, hef, hfin⟩ := fresh_cells_gen hq1 (by norm_num) L hL hLx hG hcells (fun _ ↦ he _) d24
    hdig F.c1.h37_0 F.c1.h41_0 F.c1.h42_0 hblk
  exact ⟨ef, hef, hfin⟩

set_option maxHeartbeats 16000000 in
/-- **The smudge row of prime 0 of refresh part `i` of step 0** (`w_56`, the form `hs` of
`online0_part_prime0`): the cells `125 * i + c` of the smudge column total's `c₁` row (`w_47`)
times the digits of `diag(h, …, h)`. -/
theorem online0_part_smudge0 {backend : BackendContext} {hashModel : HashModel} {params : Params}
    {i : Nat} (ctx : parallel_generatedRoot_1281.Ctx backend hashModel params i) {outputs : _}
    (F : parallel_generatedRoot_1281.Facts ctx outputs) (hi : i < 36)
    (hq1 : 1 < modulus_0) (L : RegularLayout modulus_0) (hL : backend.regularLayout modulus_0 8 = some L)
    (hLx : L.droppedModuli = 0)
    (hG : gadgetMatrixRuns backend 4096 25 (parallel_generatedRoot_1281.w_31_0 ctx))
    {s : ExactPoly modulus_0 8} {K : Fin 4500 → Fin 2 → ExactMatrix modulus_0 8 1 25}
    {t : Fin 4500 → ErrorPoly 8} {e : Fin 4500 → ErrorMatrix 8 1 25} {B : Nat}
    (hz : ∀ cc, blk (C := 4500) (m := 25) (parallel_generatedRoot_1281.w_47_0 ctx) cc =
      maskPart s (K cc) - reducePoly modulus_0 8 (t cc) • (parallel_generatedRoot_1281.w_31_0 ctx) + reduceMatrix modulus_0 8 1 25 (e cc))
    (he : ∀ cc, CoeffBound (e cc) B) {h : ExactPoly modulus_0 8} (hh : (parallel_generatedRoot_1281.w_49_0 ctx) = fun _ _ ↦ h) :
    ∃ es : ErrorMatrix 8 1 25, CoeffBound es ((25 * 25) * 8 * B * 2048) ∧
      (parallel_generatedRoot_1281.w_56_0 ctx) = maskPart s (fun k ↦
          cellsKey (fun c : Fin 25 ↦ K ⟨125 * i + c.val, by omega⟩) k * (parallel_generatedRoot_1281.w_55_0 ctx)) -
        Matrix.of (fun _ l ↦ reducePoly modulus_0 8 (t ⟨125 * i + l.val, by omega⟩) * h) +
        reduceMatrix modulus_0 8 1 25 es := by
  have d0 := isDiag_self (parallel_generatedRoot_1281.w_49_0 ctx)
  have d1 := isDiag_step (k := 1) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d0 F.c2.h49_0
  have d2 := isDiag_step (k := 2) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d1 F.c2.h49_0_1
  have d3 := isDiag_step (k := 3) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d2 F.c2.h49_0_2
  have d4 := isDiag_step (k := 4) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d3 F.c2.h49_0_3
  have d5 := isDiag_step (k := 5) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d4 F.c2.h49_0_4
  have d6 := isDiag_step (k := 6) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d5 F.c2.h49_0_5
  have d7 := isDiag_step (k := 7) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d6 F.c2.h49_0_6
  have d8 := isDiag_step (k := 8) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d7 F.c2.h49_0_7
  have d9 := isDiag_step (k := 9) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d8 F.c2.h49_0_8
  have d10 := isDiag_step (k := 10) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d9 F.c2.h49_0_9
  have d11 := isDiag_step (k := 11) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d10 F.c2.h49_0_10
  have d12 := isDiag_step (k := 12) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d11 F.c2.h49_0_11
  have d13 := isDiag_step (k := 13) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d12 F.c2.h49_0_12
  have d14 := isDiag_step (k := 14) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d13 F.c2.h49_0_13
  have d15 := isDiag_step (k := 15) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d14 F.c2.h49_0_14
  have d16 := isDiag_step (k := 16) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d15 F.c2.h49_0_15
  have d17 := isDiag_step (k := 17) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d16 F.c2.h49_0_16
  have d18 := isDiag_step (k := 18) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d17 F.c2.h49_0_17
  have d19 := isDiag_step (k := 19) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d18 F.c2.h49_0_18
  have d20 := isDiag_step (k := 20) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d19 F.c2.h49_0_19
  have d21 := isDiag_step (k := 21) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d20 F.c2.h49_0_20
  have d22 := isDiag_step (k := 22) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d21 F.c2.h49_0_21
  have d23 := isDiag_step (k := 23) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d22 F.c2.h49_0_22
  have d24 := isDiag_step (k := 24) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d23 F.c2.h50_0
  rw [hh] at d24
  have hX' : (parallel_generatedRoot_1281.w_53_0 ctx) = (parallel_generatedRoot_1281.w_50_0 ctx) := rebase_one_rows hq1 (by norm_num) F.c2.h51_0
  have hdig := F.c2.h54_0
  rw [hX'] at hdig
  have hcells := slice_cells (C := 4500) (C' := 25) (m := 25) F.c2.h48_0 (125 * i)
    (by simp only [Int.ofNat_eq_natCast]; push_cast; ring) (by omega) hz
  obtain ⟨es, hes, hfin⟩ := smudge_cells_gen hq1 (by norm_num) L hL hLx hG hcells (fun _ ↦ he _) d24
    hdig F.c2.h55_0
  exact ⟨es, hes, hfin⟩

set_option maxHeartbeats 16000000 in
/-- **The fresh row of prime 1 of refresh part `i` of step 0** (`w_103`, the form `hf` of
`online0_part_prime1`): the cells `25 i + c` of the fresh column total's `c₁` row (`w_29`)
times the digits of `diag(G, …, G)` and, per cell, of `Q_q`. -/
theorem online0_part_fresh1 {backend : BackendContext} {hashModel : HashModel} {params : Params}
    {i : Nat} (ctx : parallel_generatedRoot_1281.Ctx backend hashModel params i) {outputs : _}
    (F : parallel_generatedRoot_1281.Facts ctx outputs) (hi : i < 36)
    (hq1 : 1 < modulus_0) (L : RegularLayout modulus_0) (hL : backend.regularLayout modulus_0 8 = some L)
    (hLx : L.droppedModuli = 0)
    (hG : gadgetMatrixRuns backend 4096 25 (parallel_generatedRoot_1281.w_31_0 ctx))
    {s : ExactPoly modulus_0 8} {K : Fin 900 → Fin 2 → ExactMatrix modulus_0 8 1 25}
    {t : Fin 900 → ErrorPoly 8} {e : Fin 900 → ErrorMatrix 8 1 25} {B : Nat}
    (hz : ∀ cc, blk (C := 900) (m := 25) (parallel_generatedRoot_1281.w_29_0 ctx) cc =
      maskPart s (K cc) - reducePoly modulus_0 8 (t cc) • (parallel_generatedRoot_1281.w_31_0 ctx) + reduceMatrix modulus_0 8 1 25 (e cc))
    (he : ∀ cc, CoeffBound (e cc) B) :
    ∃ ef : ErrorMatrix 8 1 25, CoeffBound ef (25 * 8 * ((25 * 25) * 8 * B * 2048) * 2048) ∧
      (parallel_generatedRoot_1281.w_103_0 ctx) = maskPart s (fun k _ l ↦
          (blk (C := 25) (m := 25) (cellsKey (fun c : Fin 25 ↦ K ⟨25 * i + c.val, by omega⟩) k * (parallel_generatedRoot_1281.w_95_0 ctx))
            (RefreshTopProof.GenFinal.colOf (C := 25) l) * (parallel_generatedRoot_1281.w_100_0 ctx)) 0 0) -
        ((1606938044254565807694577896670997561277593686540426466090209 : Int) : ExactPoly modulus_0 8) •
          reduceMatrix modulus_0 8 1 25 (fun _ l ↦ t ⟨25 * i + l.val, by omega⟩) +
        reduceMatrix modulus_0 8 1 25 ef := by
  have d0 := isDiag_self (parallel_generatedRoot_1281.w_31_0 ctx)
  have d1 := isDiag_step (k := 1) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d0 F.c3.h31_0_23
  have d2 := isDiag_step (k := 2) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d1 F.c3.h31_0_24
  have d3 := isDiag_step (k := 3) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d2 F.c3.h31_0_25
  have d4 := isDiag_step (k := 4) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d3 F.c3.h31_0_26
  have d5 := isDiag_step (k := 5) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d4 F.c3.h31_0_27
  have d6 := isDiag_step (k := 6) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d5 F.c3.h31_0_28
  have d7 := isDiag_step (k := 7) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d6 F.c3.h31_0_29
  have d8 := isDiag_step (k := 8) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d7 F.c3.h31_0_30
  have d9 := isDiag_step (k := 9) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d8 F.c3.h31_0_31
  have d10 := isDiag_step (k := 10) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d9 F.c3.h31_0_32
  have d11 := isDiag_step (k := 11) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d10 F.c3.h31_0_33
  have d12 := isDiag_step (k := 12) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d11 F.c3.h31_0_34
  have d13 := isDiag_step (k := 13) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d12 F.c3.h31_0_35
  have d14 := isDiag_step (k := 14) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d13 F.c3.h31_0_36
  have d15 := isDiag_step (k := 15) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d14 F.c3.h31_0_37
  have d16 := isDiag_step (k := 16) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d15 F.c4.h31_0_38
  have d17 := isDiag_step (k := 17) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d16 F.c4.h31_0_39
  have d18 := isDiag_step (k := 18) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d17 F.c4.h31_0_40
  have d19 := isDiag_step (k := 19) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d18 F.c4.h31_0_41
  have d20 := isDiag_step (k := 20) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d19 F.c4.h31_0_42
  have d21 := isDiag_step (k := 21) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d20 F.c4.h31_0_43
  have d22 := isDiag_step (k := 22) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d21 F.c4.h31_0_44
  have d23 := isDiag_step (k := 23) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d22 F.c4.h31_0_45
  have d24 := isDiag_step (k := 24) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d23 F.c4.h90_0
  have hX' : (parallel_generatedRoot_1281.w_93_0 ctx) = (parallel_generatedRoot_1281.w_90_0 ctx) := rebase_one_rows hq1 (by norm_num) F.c4.h91_0
  have hdig := F.c4.h94_0
  rw [hX'] at hdig
  have hcells := slice_cells (C := 900) (C' := 25) (m := 25) F.c3.h89_0 (25 * i)
    (by simp only [Int.ofNat_eq_natCast]; push_cast; ring) (by omega) hz
  have hblk := slice_column_total (C := 25) (m := 25) (by norm_num) F.c4.h97_0 rfl F.c4.h102_0 F.c4.h103_0
    (fun _ _ _ hb ↦ sequential_parallel_generatedRoot_1281_103.sem hb)
  obtain ⟨ef, hef, hfin⟩ := fresh_cells_gen hq1 (by norm_num) L hL hLx hG hcells (fun _ ↦ he _) d24
    hdig F.c4.h95_0 F.c4.h99_0 F.c4.h100_0 hblk
  exact ⟨ef, hef, hfin⟩

set_option maxHeartbeats 16000000 in
/-- **The smudge row of prime 1 of refresh part `i` of step 0** (`w_112`, the form `hs` of
`online0_part_prime1`): the cells `125 * i + 25 + c` of the smudge column total's `c₁` row (`w_47`)
times the digits of `diag(h, …, h)`. -/
theorem online0_part_smudge1 {backend : BackendContext} {hashModel : HashModel} {params : Params}
    {i : Nat} (ctx : parallel_generatedRoot_1281.Ctx backend hashModel params i) {outputs : _}
    (F : parallel_generatedRoot_1281.Facts ctx outputs) (hi : i < 36)
    (hq1 : 1 < modulus_0) (L : RegularLayout modulus_0) (hL : backend.regularLayout modulus_0 8 = some L)
    (hLx : L.droppedModuli = 0)
    (hG : gadgetMatrixRuns backend 4096 25 (parallel_generatedRoot_1281.w_31_0 ctx))
    {s : ExactPoly modulus_0 8} {K : Fin 4500 → Fin 2 → ExactMatrix modulus_0 8 1 25}
    {t : Fin 4500 → ErrorPoly 8} {e : Fin 4500 → ErrorMatrix 8 1 25} {B : Nat}
    (hz : ∀ cc, blk (C := 4500) (m := 25) (parallel_generatedRoot_1281.w_47_0 ctx) cc =
      maskPart s (K cc) - reducePoly modulus_0 8 (t cc) • (parallel_generatedRoot_1281.w_31_0 ctx) + reduceMatrix modulus_0 8 1 25 (e cc))
    (he : ∀ cc, CoeffBound (e cc) B) {h : ExactPoly modulus_0 8} (hh : (parallel_generatedRoot_1281.w_49_0 ctx) = fun _ _ ↦ h) :
    ∃ es : ErrorMatrix 8 1 25, CoeffBound es ((25 * 25) * 8 * B * 2048) ∧
      (parallel_generatedRoot_1281.w_112_0 ctx) = maskPart s (fun k ↦
          cellsKey (fun c : Fin 25 ↦ K ⟨125 * i + 25 + c.val, by omega⟩) k * (parallel_generatedRoot_1281.w_111_0 ctx)) -
        Matrix.of (fun _ l ↦ reducePoly modulus_0 8 (t ⟨125 * i + 25 + l.val, by omega⟩) * h) +
        reduceMatrix modulus_0 8 1 25 es := by
  have d0 := isDiag_self (parallel_generatedRoot_1281.w_49_0 ctx)
  have d1 := isDiag_step (k := 1) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d0 F.c4.h49_0_23
  have d2 := isDiag_step (k := 2) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d1 F.c5.h49_0_24
  have d3 := isDiag_step (k := 3) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d2 F.c5.h49_0_25
  have d4 := isDiag_step (k := 4) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d3 F.c5.h49_0_26
  have d5 := isDiag_step (k := 5) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d4 F.c5.h49_0_27
  have d6 := isDiag_step (k := 6) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d5 F.c5.h49_0_28
  have d7 := isDiag_step (k := 7) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d6 F.c5.h49_0_29
  have d8 := isDiag_step (k := 8) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d7 F.c5.h49_0_30
  have d9 := isDiag_step (k := 9) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d8 F.c5.h49_0_31
  have d10 := isDiag_step (k := 10) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d9 F.c5.h49_0_32
  have d11 := isDiag_step (k := 11) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d10 F.c5.h49_0_33
  have d12 := isDiag_step (k := 12) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d11 F.c5.h49_0_34
  have d13 := isDiag_step (k := 13) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d12 F.c5.h49_0_35
  have d14 := isDiag_step (k := 14) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d13 F.c5.h49_0_36
  have d15 := isDiag_step (k := 15) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d14 F.c5.h49_0_37
  have d16 := isDiag_step (k := 16) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d15 F.c5.h49_0_38
  have d17 := isDiag_step (k := 17) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d16 F.c5.h49_0_39
  have d18 := isDiag_step (k := 18) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d17 F.c5.h49_0_40
  have d19 := isDiag_step (k := 19) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d18 F.c5.h49_0_41
  have d20 := isDiag_step (k := 20) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d19 F.c5.h49_0_42
  have d21 := isDiag_step (k := 21) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d20 F.c5.h49_0_43
  have d22 := isDiag_step (k := 22) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d21 F.c5.h49_0_44
  have d23 := isDiag_step (k := 23) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d22 F.c5.h49_0_45
  have d24 := isDiag_step (k := 24) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d23 F.c5.h106_0
  rw [hh] at d24
  have hX' : (parallel_generatedRoot_1281.w_109_0 ctx) = (parallel_generatedRoot_1281.w_106_0 ctx) := rebase_one_rows hq1 (by norm_num) F.c5.h107_0
  have hdig := F.c5.h110_0
  rw [hX'] at hdig
  have hcells := slice_cells (C := 4500) (C' := 25) (m := 25) F.c4.h105_0 (125 * i + 25)
    (by simp only [Int.ofNat_eq_natCast]; push_cast; ring) (by omega) hz
  obtain ⟨es, hes, hfin⟩ := smudge_cells_gen hq1 (by norm_num) L hL hLx hG hcells (fun _ ↦ he _) d24
    hdig F.c5.h111_0
  exact ⟨es, hes, hfin⟩

set_option maxHeartbeats 16000000 in
/-- **The fresh row of prime 2 of refresh part `i` of step 0** (`w_159`, the form `hf` of
`online0_part_prime2`): the cells `25 i + c` of the fresh column total's `c₁` row (`w_29`)
times the digits of `diag(G, …, G)` and, per cell, of `Q_q`. -/
theorem online0_part_fresh2 {backend : BackendContext} {hashModel : HashModel} {params : Params}
    {i : Nat} (ctx : parallel_generatedRoot_1281.Ctx backend hashModel params i) {outputs : _}
    (F : parallel_generatedRoot_1281.Facts ctx outputs) (hi : i < 36)
    (hq1 : 1 < modulus_0) (L : RegularLayout modulus_0) (hL : backend.regularLayout modulus_0 8 = some L)
    (hLx : L.droppedModuli = 0)
    (hG : gadgetMatrixRuns backend 4096 25 (parallel_generatedRoot_1281.w_31_0 ctx))
    {s : ExactPoly modulus_0 8} {K : Fin 900 → Fin 2 → ExactMatrix modulus_0 8 1 25}
    {t : Fin 900 → ErrorPoly 8} {e : Fin 900 → ErrorMatrix 8 1 25} {B : Nat}
    (hz : ∀ cc, blk (C := 900) (m := 25) (parallel_generatedRoot_1281.w_29_0 ctx) cc =
      maskPart s (K cc) - reducePoly modulus_0 8 (t cc) • (parallel_generatedRoot_1281.w_31_0 ctx) + reduceMatrix modulus_0 8 1 25 (e cc))
    (he : ∀ cc, CoeffBound (e cc) B) :
    ∃ ef : ErrorMatrix 8 1 25, CoeffBound ef (25 * 8 * ((25 * 25) * 8 * B * 2048) * 2048) ∧
      (parallel_generatedRoot_1281.w_159_0 ctx) = maskPart s (fun k _ l ↦
          (blk (C := 25) (m := 25) (cellsKey (fun c : Fin 25 ↦ K ⟨25 * i + c.val, by omega⟩) k * (parallel_generatedRoot_1281.w_151_0 ctx))
            (RefreshTopProof.GenFinal.colOf (C := 25) l) * (parallel_generatedRoot_1281.w_156_0 ctx)) 0 0) -
        ((1606938044254771331362327096896778694705812889646799470123121 : Int) : ExactPoly modulus_0 8) •
          reduceMatrix modulus_0 8 1 25 (fun _ l ↦ t ⟨25 * i + l.val, by omega⟩) +
        reduceMatrix modulus_0 8 1 25 ef := by
  have d0 := isDiag_self (parallel_generatedRoot_1281.w_31_0 ctx)
  have d1 := isDiag_step (k := 1) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d0 F.c6.h31_0_46
  have d2 := isDiag_step (k := 2) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d1 F.c6.h31_0_47
  have d3 := isDiag_step (k := 3) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d2 F.c6.h31_0_48
  have d4 := isDiag_step (k := 4) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d3 F.c6.h31_0_49
  have d5 := isDiag_step (k := 5) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d4 F.c6.h31_0_50
  have d6 := isDiag_step (k := 6) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d5 F.c6.h31_0_51
  have d7 := isDiag_step (k := 7) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d6 F.c6.h31_0_52
  have d8 := isDiag_step (k := 8) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d7 F.c6.h31_0_53
  have d9 := isDiag_step (k := 9) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d8 F.c6.h31_0_54
  have d10 := isDiag_step (k := 10) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d9 F.c6.h31_0_55
  have d11 := isDiag_step (k := 11) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d10 F.c6.h31_0_56
  have d12 := isDiag_step (k := 12) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d11 F.c6.h31_0_57
  have d13 := isDiag_step (k := 13) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d12 F.c6.h31_0_58
  have d14 := isDiag_step (k := 14) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d13 F.c6.h31_0_59
  have d15 := isDiag_step (k := 15) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d14 F.c6.h31_0_60
  have d16 := isDiag_step (k := 16) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d15 F.c6.h31_0_61
  have d17 := isDiag_step (k := 17) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d16 F.c6.h31_0_62
  have d18 := isDiag_step (k := 18) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d17 F.c7.h31_0_63
  have d19 := isDiag_step (k := 19) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d18 F.c7.h31_0_64
  have d20 := isDiag_step (k := 20) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d19 F.c7.h31_0_65
  have d21 := isDiag_step (k := 21) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d20 F.c7.h31_0_66
  have d22 := isDiag_step (k := 22) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d21 F.c7.h31_0_67
  have d23 := isDiag_step (k := 23) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d22 F.c7.h31_0_68
  have d24 := isDiag_step (k := 24) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d23 F.c7.h146_0
  have hX' : (parallel_generatedRoot_1281.w_149_0 ctx) = (parallel_generatedRoot_1281.w_146_0 ctx) := rebase_one_rows hq1 (by norm_num) F.c7.h147_0
  have hdig := F.c7.h150_0
  rw [hX'] at hdig
  have hcells := slice_cells (C := 900) (C' := 25) (m := 25) F.c6.h145_0 (25 * i)
    (by simp only [Int.ofNat_eq_natCast]; push_cast; ring) (by omega) hz
  have hblk := slice_column_total (C := 25) (m := 25) (by norm_num) F.c7.h153_0 rfl F.c7.h158_0 F.c7.h159_0
    (fun _ _ _ hb ↦ sequential_parallel_generatedRoot_1281_159.sem hb)
  obtain ⟨ef, hef, hfin⟩ := fresh_cells_gen hq1 (by norm_num) L hL hLx hG hcells (fun _ ↦ he _) d24
    hdig F.c7.h151_0 F.c7.h155_0 F.c7.h156_0 hblk
  exact ⟨ef, hef, hfin⟩

set_option maxHeartbeats 16000000 in
/-- **The smudge row of prime 2 of refresh part `i` of step 0** (`w_168`, the form `hs` of
`online0_part_prime2`): the cells `125 * i + 50 + c` of the smudge column total's `c₁` row (`w_47`)
times the digits of `diag(h, …, h)`. -/
theorem online0_part_smudge2 {backend : BackendContext} {hashModel : HashModel} {params : Params}
    {i : Nat} (ctx : parallel_generatedRoot_1281.Ctx backend hashModel params i) {outputs : _}
    (F : parallel_generatedRoot_1281.Facts ctx outputs) (hi : i < 36)
    (hq1 : 1 < modulus_0) (L : RegularLayout modulus_0) (hL : backend.regularLayout modulus_0 8 = some L)
    (hLx : L.droppedModuli = 0)
    (hG : gadgetMatrixRuns backend 4096 25 (parallel_generatedRoot_1281.w_31_0 ctx))
    {s : ExactPoly modulus_0 8} {K : Fin 4500 → Fin 2 → ExactMatrix modulus_0 8 1 25}
    {t : Fin 4500 → ErrorPoly 8} {e : Fin 4500 → ErrorMatrix 8 1 25} {B : Nat}
    (hz : ∀ cc, blk (C := 4500) (m := 25) (parallel_generatedRoot_1281.w_47_0 ctx) cc =
      maskPart s (K cc) - reducePoly modulus_0 8 (t cc) • (parallel_generatedRoot_1281.w_31_0 ctx) + reduceMatrix modulus_0 8 1 25 (e cc))
    (he : ∀ cc, CoeffBound (e cc) B) {h : ExactPoly modulus_0 8} (hh : (parallel_generatedRoot_1281.w_49_0 ctx) = fun _ _ ↦ h) :
    ∃ es : ErrorMatrix 8 1 25, CoeffBound es ((25 * 25) * 8 * B * 2048) ∧
      (parallel_generatedRoot_1281.w_168_0 ctx) = maskPart s (fun k ↦
          cellsKey (fun c : Fin 25 ↦ K ⟨125 * i + 50 + c.val, by omega⟩) k * (parallel_generatedRoot_1281.w_167_0 ctx)) -
        Matrix.of (fun _ l ↦ reducePoly modulus_0 8 (t ⟨125 * i + 50 + l.val, by omega⟩) * h) +
        reduceMatrix modulus_0 8 1 25 es := by
  have d0 := isDiag_self (parallel_generatedRoot_1281.w_49_0 ctx)
  have d1 := isDiag_step (k := 1) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d0 F.c7.h49_0_46
  have d2 := isDiag_step (k := 2) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d1 F.c7.h49_0_47
  have d3 := isDiag_step (k := 3) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d2 F.c7.h49_0_48
  have d4 := isDiag_step (k := 4) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d3 F.c8.h49_0_49
  have d5 := isDiag_step (k := 5) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d4 F.c8.h49_0_50
  have d6 := isDiag_step (k := 6) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d5 F.c8.h49_0_51
  have d7 := isDiag_step (k := 7) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d6 F.c8.h49_0_52
  have d8 := isDiag_step (k := 8) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d7 F.c8.h49_0_53
  have d9 := isDiag_step (k := 9) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d8 F.c8.h49_0_54
  have d10 := isDiag_step (k := 10) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d9 F.c8.h49_0_55
  have d11 := isDiag_step (k := 11) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d10 F.c8.h49_0_56
  have d12 := isDiag_step (k := 12) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d11 F.c8.h49_0_57
  have d13 := isDiag_step (k := 13) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d12 F.c8.h49_0_58
  have d14 := isDiag_step (k := 14) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d13 F.c8.h49_0_59
  have d15 := isDiag_step (k := 15) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d14 F.c8.h49_0_60
  have d16 := isDiag_step (k := 16) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d15 F.c8.h49_0_61
  have d17 := isDiag_step (k := 17) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d16 F.c8.h49_0_62
  have d18 := isDiag_step (k := 18) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d17 F.c8.h49_0_63
  have d19 := isDiag_step (k := 19) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d18 F.c8.h49_0_64
  have d20 := isDiag_step (k := 20) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d19 F.c8.h49_0_65
  have d21 := isDiag_step (k := 21) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d20 F.c8.h49_0_66
  have d22 := isDiag_step (k := 22) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d21 F.c8.h49_0_67
  have d23 := isDiag_step (k := 23) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d22 F.c8.h49_0_68
  have d24 := isDiag_step (k := 24) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d23 F.c8.h162_0
  rw [hh] at d24
  have hX' : (parallel_generatedRoot_1281.w_165_0 ctx) = (parallel_generatedRoot_1281.w_162_0 ctx) := rebase_one_rows hq1 (by norm_num) F.c8.h163_0
  have hdig := F.c8.h166_0
  rw [hX'] at hdig
  have hcells := slice_cells (C := 4500) (C' := 25) (m := 25) F.c7.h161_0 (125 * i + 50)
    (by simp only [Int.ofNat_eq_natCast]; push_cast; ring) (by omega) hz
  obtain ⟨es, hes, hfin⟩ := smudge_cells_gen hq1 (by norm_num) L hL hLx hG hcells (fun _ ↦ he _) d24
    hdig F.c8.h167_0
  exact ⟨es, hes, hfin⟩

set_option maxHeartbeats 16000000 in
/-- **The fresh row of prime 3 of refresh part `i` of step 0** (`w_215`, the form `hf` of
`online0_part_prime3`): the cells `25 i + c` of the fresh column total's `c₁` row (`w_29`)
times the digits of `diag(G, …, G)` and, per cell, of `Q_q`. -/
theorem online0_part_fresh3 {backend : BackendContext} {hashModel : HashModel} {params : Params}
    {i : Nat} (ctx : parallel_generatedRoot_1281.Ctx backend hashModel params i) {outputs : _}
    (F : parallel_generatedRoot_1281.Facts ctx outputs) (hi : i < 36)
    (hq1 : 1 < modulus_0) (L : RegularLayout modulus_0) (hL : backend.regularLayout modulus_0 8 = some L)
    (hLx : L.droppedModuli = 0)
    (hG : gadgetMatrixRuns backend 4096 25 (parallel_generatedRoot_1281.w_31_0 ctx))
    {s : ExactPoly modulus_0 8} {K : Fin 900 → Fin 2 → ExactMatrix modulus_0 8 1 25}
    {t : Fin 900 → ErrorPoly 8} {e : Fin 900 → ErrorMatrix 8 1 25} {B : Nat}
    (hz : ∀ cc, blk (C := 900) (m := 25) (parallel_generatedRoot_1281.w_29_0 ctx) cc =
      maskPart s (K cc) - reducePoly modulus_0 8 (t cc) • (parallel_generatedRoot_1281.w_31_0 ctx) + reduceMatrix modulus_0 8 1 25 (e cc))
    (he : ∀ cc, CoeffBound (e cc) B) :
    ∃ ef : ErrorMatrix 8 1 25, CoeffBound ef (25 * 8 * ((25 * 25) * 8 * B * 2048) * 2048) ∧
      (parallel_generatedRoot_1281.w_215_0 ctx) = maskPart s (fun k _ l ↦
          (blk (C := 25) (m := 25) (cellsKey (fun c : Fin 25 ↦ K ⟨25 * i + c.val, by omega⟩) k * (parallel_generatedRoot_1281.w_207_0 ctx))
            (RefreshTopProof.GenFinal.colOf (C := 25) l) * (parallel_generatedRoot_1281.w_212_0 ctx)) 0 0) -
        ((1606938044255228050623992174508164557277769211170177223088561 : Int) : ExactPoly modulus_0 8) •
          reduceMatrix modulus_0 8 1 25 (fun _ l ↦ t ⟨25 * i + l.val, by omega⟩) +
        reduceMatrix modulus_0 8 1 25 ef := by
  have d0 := isDiag_self (parallel_generatedRoot_1281.w_31_0 ctx)
  have d1 := isDiag_step (k := 1) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d0 F.c9.h31_0_69
  have d2 := isDiag_step (k := 2) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d1 F.c9.h31_0_70
  have d3 := isDiag_step (k := 3) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d2 F.c9.h31_0_71
  have d4 := isDiag_step (k := 4) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d3 F.c9.h31_0_72
  have d5 := isDiag_step (k := 5) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d4 F.c9.h31_0_73
  have d6 := isDiag_step (k := 6) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d5 F.c9.h31_0_74
  have d7 := isDiag_step (k := 7) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d6 F.c9.h31_0_75
  have d8 := isDiag_step (k := 8) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d7 F.c9.h31_0_76
  have d9 := isDiag_step (k := 9) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d8 F.c9.h31_0_77
  have d10 := isDiag_step (k := 10) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d9 F.c9.h31_0_78
  have d11 := isDiag_step (k := 11) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d10 F.c9.h31_0_79
  have d12 := isDiag_step (k := 12) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d11 F.c9.h31_0_80
  have d13 := isDiag_step (k := 13) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d12 F.c9.h31_0_81
  have d14 := isDiag_step (k := 14) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d13 F.c9.h31_0_82
  have d15 := isDiag_step (k := 15) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d14 F.c9.h31_0_83
  have d16 := isDiag_step (k := 16) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d15 F.c9.h31_0_84
  have d17 := isDiag_step (k := 17) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d16 F.c9.h31_0_85
  have d18 := isDiag_step (k := 18) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d17 F.c9.h31_0_86
  have d19 := isDiag_step (k := 19) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d18 F.c9.h31_0_87
  have d20 := isDiag_step (k := 20) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d19 F.c10.h31_0_88
  have d21 := isDiag_step (k := 21) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d20 F.c10.h31_0_89
  have d22 := isDiag_step (k := 22) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d21 F.c10.h31_0_90
  have d23 := isDiag_step (k := 23) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d22 F.c10.h31_0_91
  have d24 := isDiag_step (k := 24) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d23 F.c10.h202_0
  have hX' : (parallel_generatedRoot_1281.w_205_0 ctx) = (parallel_generatedRoot_1281.w_202_0 ctx) := rebase_one_rows hq1 (by norm_num) F.c10.h203_0
  have hdig := F.c10.h206_0
  rw [hX'] at hdig
  have hcells := slice_cells (C := 900) (C' := 25) (m := 25) F.c9.h201_0 (25 * i)
    (by simp only [Int.ofNat_eq_natCast]; push_cast; ring) (by omega) hz
  have hblk := slice_column_total (C := 25) (m := 25) (by norm_num) F.c10.h209_0 rfl F.c10.h214_0 F.c10.h215_0
    (fun _ _ _ hb ↦ sequential_parallel_generatedRoot_1281_215.sem hb)
  obtain ⟨ef, hef, hfin⟩ := fresh_cells_gen hq1 (by norm_num) L hL hLx hG hcells (fun _ ↦ he _) d24
    hdig F.c10.h207_0 F.c10.h211_0 F.c10.h212_0 hblk
  exact ⟨ef, hef, hfin⟩

set_option maxHeartbeats 16000000 in
/-- **The smudge row of prime 3 of refresh part `i` of step 0** (`w_224`, the form `hs` of
`online0_part_prime3`): the cells `125 * i + 75 + c` of the smudge column total's `c₁` row (`w_47`)
times the digits of `diag(h, …, h)`. -/
theorem online0_part_smudge3 {backend : BackendContext} {hashModel : HashModel} {params : Params}
    {i : Nat} (ctx : parallel_generatedRoot_1281.Ctx backend hashModel params i) {outputs : _}
    (F : parallel_generatedRoot_1281.Facts ctx outputs) (hi : i < 36)
    (hq1 : 1 < modulus_0) (L : RegularLayout modulus_0) (hL : backend.regularLayout modulus_0 8 = some L)
    (hLx : L.droppedModuli = 0)
    (hG : gadgetMatrixRuns backend 4096 25 (parallel_generatedRoot_1281.w_31_0 ctx))
    {s : ExactPoly modulus_0 8} {K : Fin 4500 → Fin 2 → ExactMatrix modulus_0 8 1 25}
    {t : Fin 4500 → ErrorPoly 8} {e : Fin 4500 → ErrorMatrix 8 1 25} {B : Nat}
    (hz : ∀ cc, blk (C := 4500) (m := 25) (parallel_generatedRoot_1281.w_47_0 ctx) cc =
      maskPart s (K cc) - reducePoly modulus_0 8 (t cc) • (parallel_generatedRoot_1281.w_31_0 ctx) + reduceMatrix modulus_0 8 1 25 (e cc))
    (he : ∀ cc, CoeffBound (e cc) B) {h : ExactPoly modulus_0 8} (hh : (parallel_generatedRoot_1281.w_49_0 ctx) = fun _ _ ↦ h) :
    ∃ es : ErrorMatrix 8 1 25, CoeffBound es ((25 * 25) * 8 * B * 2048) ∧
      (parallel_generatedRoot_1281.w_224_0 ctx) = maskPart s (fun k ↦
          cellsKey (fun c : Fin 25 ↦ K ⟨125 * i + 75 + c.val, by omega⟩) k * (parallel_generatedRoot_1281.w_223_0 ctx)) -
        Matrix.of (fun _ l ↦ reducePoly modulus_0 8 (t ⟨125 * i + 75 + l.val, by omega⟩) * h) +
        reduceMatrix modulus_0 8 1 25 es := by
  have d0 := isDiag_self (parallel_generatedRoot_1281.w_49_0 ctx)
  have d1 := isDiag_step (k := 1) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d0 F.c10.h49_0_69
  have d2 := isDiag_step (k := 2) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d1 F.c10.h49_0_70
  have d3 := isDiag_step (k := 3) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d2 F.c10.h49_0_71
  have d4 := isDiag_step (k := 4) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d3 F.c10.h49_0_72
  have d5 := isDiag_step (k := 5) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d4 F.c10.h49_0_73
  have d6 := isDiag_step (k := 6) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d5 F.c11.h49_0_74
  have d7 := isDiag_step (k := 7) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d6 F.c11.h49_0_75
  have d8 := isDiag_step (k := 8) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d7 F.c11.h49_0_76
  have d9 := isDiag_step (k := 9) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d8 F.c11.h49_0_77
  have d10 := isDiag_step (k := 10) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d9 F.c11.h49_0_78
  have d11 := isDiag_step (k := 11) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d10 F.c11.h49_0_79
  have d12 := isDiag_step (k := 12) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d11 F.c11.h49_0_80
  have d13 := isDiag_step (k := 13) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d12 F.c11.h49_0_81
  have d14 := isDiag_step (k := 14) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d13 F.c11.h49_0_82
  have d15 := isDiag_step (k := 15) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d14 F.c11.h49_0_83
  have d16 := isDiag_step (k := 16) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d15 F.c11.h49_0_84
  have d17 := isDiag_step (k := 17) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d16 F.c11.h49_0_85
  have d18 := isDiag_step (k := 18) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d17 F.c11.h49_0_86
  have d19 := isDiag_step (k := 19) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d18 F.c11.h49_0_87
  have d20 := isDiag_step (k := 20) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d19 F.c11.h49_0_88
  have d21 := isDiag_step (k := 21) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d20 F.c11.h49_0_89
  have d22 := isDiag_step (k := 22) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d21 F.c11.h49_0_90
  have d23 := isDiag_step (k := 23) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d22 F.c11.h49_0_91
  have d24 := isDiag_step (k := 24) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d23 F.c11.h218_0
  rw [hh] at d24
  have hX' : (parallel_generatedRoot_1281.w_221_0 ctx) = (parallel_generatedRoot_1281.w_218_0 ctx) := rebase_one_rows hq1 (by norm_num) F.c11.h219_0
  have hdig := F.c11.h222_0
  rw [hX'] at hdig
  have hcells := slice_cells (C := 4500) (C' := 25) (m := 25) F.c10.h217_0 (125 * i + 75)
    (by simp only [Int.ofNat_eq_natCast]; push_cast; ring) (by omega) hz
  obtain ⟨es, hes, hfin⟩ := smudge_cells_gen hq1 (by norm_num) L hL hLx hG hcells (fun _ ↦ he _) d24
    hdig F.c11.h223_0
  exact ⟨es, hes, hfin⟩

set_option maxHeartbeats 16000000 in
/-- **The fresh row of prime 4 of refresh part `i` of step 0** (`w_271`, the form `hf` of
`online0_part_prime4`): the cells `25 i + c` of the fresh column total's `c₁` row (`w_29`)
times the digits of `diag(G, …, G)` and, per cell, of `Q_q`. -/
theorem online0_part_fresh4 {backend : BackendContext} {hashModel : HashModel} {params : Params}
    {i : Nat} (ctx : parallel_generatedRoot_1281.Ctx backend hashModel params i) {outputs : _}
    (F : parallel_generatedRoot_1281.Facts ctx outputs) (hi : i < 36)
    (hq1 : 1 < modulus_0) (L : RegularLayout modulus_0) (hL : backend.regularLayout modulus_0 8 = some L)
    (hLx : L.droppedModuli = 0)
    (hG : gadgetMatrixRuns backend 4096 25 (parallel_generatedRoot_1281.w_31_0 ctx))
    {s : ExactPoly modulus_0 8} {K : Fin 900 → Fin 2 → ExactMatrix modulus_0 8 1 25}
    {t : Fin 900 → ErrorPoly 8} {e : Fin 900 → ErrorMatrix 8 1 25} {B : Nat}
    (hz : ∀ cc, blk (C := 900) (m := 25) (parallel_generatedRoot_1281.w_29_0 ctx) cc =
      maskPart s (K cc) - reducePoly modulus_0 8 (t cc) • (parallel_generatedRoot_1281.w_31_0 ctx) + reduceMatrix modulus_0 8 1 25 (e cc))
    (he : ∀ cc, CoeffBound (e cc) B) :
    ∃ ef : ErrorMatrix 8 1 25, CoeffBound ef (25 * 8 * ((25 * 25) * 8 * B * 2048) * 2048) ∧
      (parallel_generatedRoot_1281.w_271_0 ctx) = maskPart s (fun k _ l ↦
          (blk (C := 25) (m := 25) (cellsKey (fun c : Fin 25 ↦ K ⟨25 * i + c.val, by omega⟩) k * (parallel_generatedRoot_1281.w_263_0 ctx))
            (RefreshTopProof.GenFinal.colOf (C := 25) l) * (parallel_generatedRoot_1281.w_268_0 ctx)) 0 0) -
        ((1606938044255707605848740785410594412685698358681526360823809 : Int) : ExactPoly modulus_0 8) •
          reduceMatrix modulus_0 8 1 25 (fun _ l ↦ t ⟨25 * i + l.val, by omega⟩) +
        reduceMatrix modulus_0 8 1 25 ef := by
  have d0 := isDiag_self (parallel_generatedRoot_1281.w_31_0 ctx)
  have d1 := isDiag_step (k := 1) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d0 F.c12.h31_0_92
  have d2 := isDiag_step (k := 2) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d1 F.c12.h31_0_93
  have d3 := isDiag_step (k := 3) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d2 F.c12.h31_0_94
  have d4 := isDiag_step (k := 4) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d3 F.c12.h31_0_95
  have d5 := isDiag_step (k := 5) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d4 F.c12.h31_0_96
  have d6 := isDiag_step (k := 6) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d5 F.c12.h31_0_97
  have d7 := isDiag_step (k := 7) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d6 F.c12.h31_0_98
  have d8 := isDiag_step (k := 8) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d7 F.c12.h31_0_99
  have d9 := isDiag_step (k := 9) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d8 F.c12.h31_0_100
  have d10 := isDiag_step (k := 10) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d9 F.c12.h31_0_101
  have d11 := isDiag_step (k := 11) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d10 F.c12.h31_0_102
  have d12 := isDiag_step (k := 12) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d11 F.c12.h31_0_103
  have d13 := isDiag_step (k := 13) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d12 F.c12.h31_0_104
  have d14 := isDiag_step (k := 14) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d13 F.c12.h31_0_105
  have d15 := isDiag_step (k := 15) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d14 F.c12.h31_0_106
  have d16 := isDiag_step (k := 16) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d15 F.c12.h31_0_107
  have d17 := isDiag_step (k := 17) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d16 F.c12.h31_0_108
  have d18 := isDiag_step (k := 18) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d17 F.c12.h31_0_109
  have d19 := isDiag_step (k := 19) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d18 F.c12.h31_0_110
  have d20 := isDiag_step (k := 20) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d19 F.c12.h31_0_111
  have d21 := isDiag_step (k := 21) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d20 F.c12.h31_0_112
  have d22 := isDiag_step (k := 22) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d21 F.c13.h31_0_113
  have d23 := isDiag_step (k := 23) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d22 F.c13.h31_0_114
  have d24 := isDiag_step (k := 24) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d23 F.c13.h258_0
  have hX' : (parallel_generatedRoot_1281.w_261_0 ctx) = (parallel_generatedRoot_1281.w_258_0 ctx) := rebase_one_rows hq1 (by norm_num) F.c13.h259_0
  have hdig := F.c13.h262_0
  rw [hX'] at hdig
  have hcells := slice_cells (C := 900) (C' := 25) (m := 25) F.c12.h257_0 (25 * i)
    (by simp only [Int.ofNat_eq_natCast]; push_cast; ring) (by omega) hz
  have hblk := slice_column_total (C := 25) (m := 25) (by norm_num) F.c13.h265_0 rfl F.c13.h270_0 F.c13.h271_0
    (fun _ _ _ hb ↦ sequential_parallel_generatedRoot_1281_271.sem hb)
  obtain ⟨ef, hef, hfin⟩ := fresh_cells_gen hq1 (by norm_num) L hL hLx hG hcells (fun _ ↦ he _) d24
    hdig F.c13.h263_0 F.c13.h267_0 F.c13.h268_0 hblk
  exact ⟨ef, hef, hfin⟩

set_option maxHeartbeats 16000000 in
/-- **The smudge row of prime 4 of refresh part `i` of step 0** (`w_280`, the form `hs` of
`online0_part_prime4`): the cells `125 * i + 100 + c` of the smudge column total's `c₁` row (`w_47`)
times the digits of `diag(h, …, h)`. -/
theorem online0_part_smudge4 {backend : BackendContext} {hashModel : HashModel} {params : Params}
    {i : Nat} (ctx : parallel_generatedRoot_1281.Ctx backend hashModel params i) {outputs : _}
    (F : parallel_generatedRoot_1281.Facts ctx outputs) (hi : i < 36)
    (hq1 : 1 < modulus_0) (L : RegularLayout modulus_0) (hL : backend.regularLayout modulus_0 8 = some L)
    (hLx : L.droppedModuli = 0)
    (hG : gadgetMatrixRuns backend 4096 25 (parallel_generatedRoot_1281.w_31_0 ctx))
    {s : ExactPoly modulus_0 8} {K : Fin 4500 → Fin 2 → ExactMatrix modulus_0 8 1 25}
    {t : Fin 4500 → ErrorPoly 8} {e : Fin 4500 → ErrorMatrix 8 1 25} {B : Nat}
    (hz : ∀ cc, blk (C := 4500) (m := 25) (parallel_generatedRoot_1281.w_47_0 ctx) cc =
      maskPart s (K cc) - reducePoly modulus_0 8 (t cc) • (parallel_generatedRoot_1281.w_31_0 ctx) + reduceMatrix modulus_0 8 1 25 (e cc))
    (he : ∀ cc, CoeffBound (e cc) B) {h : ExactPoly modulus_0 8} (hh : (parallel_generatedRoot_1281.w_49_0 ctx) = fun _ _ ↦ h) :
    ∃ es : ErrorMatrix 8 1 25, CoeffBound es ((25 * 25) * 8 * B * 2048) ∧
      (parallel_generatedRoot_1281.w_280_0 ctx) = maskPart s (fun k ↦
          cellsKey (fun c : Fin 25 ↦ K ⟨125 * i + 100 + c.val, by omega⟩) k * (parallel_generatedRoot_1281.w_279_0 ctx)) -
        Matrix.of (fun _ l ↦ reducePoly modulus_0 8 (t ⟨125 * i + 100 + l.val, by omega⟩) * h) +
        reduceMatrix modulus_0 8 1 25 es := by
  have d0 := isDiag_self (parallel_generatedRoot_1281.w_49_0 ctx)
  have d1 := isDiag_step (k := 1) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d0 F.c13.h49_0_92
  have d2 := isDiag_step (k := 2) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d1 F.c13.h49_0_93
  have d3 := isDiag_step (k := 3) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d2 F.c13.h49_0_94
  have d4 := isDiag_step (k := 4) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d3 F.c13.h49_0_95
  have d5 := isDiag_step (k := 5) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d4 F.c13.h49_0_96
  have d6 := isDiag_step (k := 6) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d5 F.c13.h49_0_97
  have d7 := isDiag_step (k := 7) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d6 F.c13.h49_0_98
  have d8 := isDiag_step (k := 8) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d7 F.c14.h49_0_99
  have d9 := isDiag_step (k := 9) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d8 F.c14.h49_0_100
  have d10 := isDiag_step (k := 10) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d9 F.c14.h49_0_101
  have d11 := isDiag_step (k := 11) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d10 F.c14.h49_0_102
  have d12 := isDiag_step (k := 12) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d11 F.c14.h49_0_103
  have d13 := isDiag_step (k := 13) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d12 F.c14.h49_0_104
  have d14 := isDiag_step (k := 14) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d13 F.c14.h49_0_105
  have d15 := isDiag_step (k := 15) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d14 F.c14.h49_0_106
  have d16 := isDiag_step (k := 16) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d15 F.c14.h49_0_107
  have d17 := isDiag_step (k := 17) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d16 F.c14.h49_0_108
  have d18 := isDiag_step (k := 18) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d17 F.c14.h49_0_109
  have d19 := isDiag_step (k := 19) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d18 F.c14.h49_0_110
  have d20 := isDiag_step (k := 20) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d19 F.c14.h49_0_111
  have d21 := isDiag_step (k := 21) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d20 F.c14.h49_0_112
  have d22 := isDiag_step (k := 22) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d21 F.c14.h49_0_113
  have d23 := isDiag_step (k := 23) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d22 F.c14.h49_0_114
  have d24 := isDiag_step (k := 24) (by norm_num) (by norm_num) (by norm_num) (by norm_num) d23 F.c14.h274_0
  rw [hh] at d24
  have hX' : (parallel_generatedRoot_1281.w_277_0 ctx) = (parallel_generatedRoot_1281.w_274_0 ctx) := rebase_one_rows hq1 (by norm_num) F.c14.h275_0
  have hdig := F.c14.h278_0
  rw [hX'] at hdig
  have hcells := slice_cells (C := 4500) (C' := 25) (m := 25) F.c13.h273_0 (125 * i + 100)
    (by simp only [Int.ofNat_eq_natCast]; push_cast; ring) (by omega) hz
  obtain ⟨es, hes, hfin⟩ := smudge_cells_gen hq1 (by norm_num) L hL hLx hG hcells (fun _ ↦ he _) d24
    hdig F.c14.h279_0
  exact ⟨es, hes, hfin⟩


end Instances

end Dio2Proof
