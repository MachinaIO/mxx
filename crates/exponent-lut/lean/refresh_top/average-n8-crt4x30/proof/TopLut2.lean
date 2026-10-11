import TopLut

/-!
# Width-`2` LUT loops

The source LUT of the refresh has width `2`: automorphisms `σ_i ∈ {1, 9}` and the LUT variable
`X` (`X^8 = -1`). Its loop accumulates the first branch, and the second is in the loop's caller.
-/

namespace RefreshTopProof

open Mxx.Primitives MxxRuntime MxxExponentLut

/-- The automorphism indices of a width-`2` LUT. -/
abbrev sigmas2 : List Int := [(1), (9)]

/-- The automorphism of branch `i`. -/
abbrev sigma2 (i : Nat) : Int := MxxIR.selectInt (Int.ofNat i) sigmas2

theorem sigma2_odd (i : Fin 2) : Odd (sigma2 i).toNat := by
  fin_cases i <;> decide

/-- The target of branch `i` of a width-`2` LUT over the switching families `fam`. -/
noncomputable def branchTarget2 {q q' p m : Nat} (μ sp : ErrorPoly N)
    (fam : Fin 2 → Fin 2 → Fin 2 → ExactMatrix q' N 1 m) (i : Fin 2) :
    Fin 2 → ExactMatrix q N 1 m :=
  fun j ↦ scaleUpMatrix q p (reducePoly q' N (substitution (sigma2 i).toNat (sigma2_odd i) μ) •
    (reducePoly q' N (substitution (sigma2 i).toNat (sigma2_odd i) sp) • fam 0 j i + fam 1 j i))

/-- Branch `i` of a width-`2` LUT, zero past the two branches. -/
noncomputable def lutBranchTarget2 {q q' p m : Nat} (μ sp : ErrorPoly N)
    (fam : Fin 2 → Fin 2 → Fin 2 → ExactMatrix q' N 1 m) (i : Nat) : Fin 2 → ExactMatrix q N 1 m :=
  if h : i < 2 then branchTarget2 (q := q) (p := p) μ sp fam ⟨i, h⟩ else 0

/-! ## The interpolation -/

/-- `σ_i ≡ 1 + 8 i (mod 16)`. -/
theorem sigma2_mod (i : Fin 2) :
    (sigma2 i).toNat % (2 * 8) = 1 + 2 * 4 * ((Equiv.refl (Fin 2)) i).val := by
  fin_cases i <;> decide

/-- The coefficient `C_σ = 2⁻¹ Σ_k T(k) σ(X^(-k))` of a width-`2` LUT with table `T`. -/
noncomputable def lutCoefficient2 (q inverse : Nat) (T : Fin 2 → ExactPoly q N) (σ : Nat) :
    ExactPoly q N :=
  lutFourierCoefficient 8 (X q) (inverse : ExactPoly q N) T σ

theorem polyNorm_substitution2_Xe (i : Fin 2) (k : Nat) :
    polyNorm (substitution (sigma2 i).toNat (sigma2_odd i) (Xe (n := N) k)) ≤ 1 := by
  rw [substitution_root_pow]
  exact polyNorm_Xe hN1 _

/-- A width-`2` LUT on the state `X^z`: `X^(z - z mod 2) T(z mod 2)`. -/
theorem lut2_walsh_value {q inverse : Nat} (T : Fin 2 → ErrorPoly N)
    (hinverse : (inverse : ExactPoly q N) * ((2 : Nat) : ExactPoly q N) = 1) (z : Nat) :
    ∑ i : Fin 2, reducePoly q N (substitution (sigma2 i).toNat (sigma2_odd i) (Xe z)) *
        lutCoefficient2 q inverse (fun k ↦ reducePoly q N (T k)) (sigma2 i).toNat =
      reducePoly q N (Xe (z - z % 2) * T ⟨z % 2, Nat.mod_lt _ (by norm_num)⟩) := by
  have h := lut_fourier_interpolation (R := ExactPoly q N) 1 8 4 z (X q) (inverse : ExactPoly q N)
    (fun k ↦ reducePoly q N (T k)) (by norm_num) (by norm_num) Negacyclic.root_pow_n hinverse
    (fun i ↦ (sigma2 i).toNat) (Equiv.refl _) sigma2_mod
  simp only [pow_one] at h
  rw [map_mul, reducePoly_Xe]
  simp only [lutCoefficient2]
  rw [← h]
  apply Finset.sum_congr rfl
  intro i _
  rw [substitution_root_pow, reducePoly_Xe, mul_comm z]
  rfl

/-! ## A LUT round -/

set_option maxHeartbeats 64000000 in
/-- The two branches of a width-`2` LUT before its move down: the loop's first (`hloop`) and the
fourth. -/
theorem lut_branches2 {q q' p m m' : Nat} (hq : q = p * q') (hq1 : 1 < q) (hq' : 1 < q') (hp : 0 < p)
    (layout : RegularLayout q') (hlayout : Backend.backend.regularLayout q' N = some layout)
    (hexact : layout.droppedModuli = 0)
    {model : HashModel} {key : ByteArray} {level : Int}
    {sm sp : ErrorPoly N} (hsm : polyNorm sm ≤ 1) {z : Nat}
    {g' : ExactMatrix q' N 1 m} {g : ExactMatrix q N 1 m}
    (hg' : gadgetMatrixRuns Backend.backend 32768 m g') (hg : centeredRebaseRuns g' g)
    {c : Fin 2 → ExactMatrix q N 1 m} {A : Fin 2 → Fin 2 → ExactMatrix q N 1 m} {B : Nat}
    (hinput : Rel sm A (fun j ↦ reducePoly q N (((p : Nat) : Int) * Xe (n := N) z *
      unitPair sp j) • g) c B)
    {fam : Fin 2 → Fin 2 → Fin 2 → ExactMatrix q' N 1 m'} {final : LoopState q q' m'} {Bl : Nat}
    (hloop : Rel sm final.key (∑ i ∈ Finset.range 1, lutBranchTarget2 (q := q) (p := p)
      (Xe z) sp fam i) final.row Bl)
    (hloopSwitch : final.switch = fun k j ↦ fam k j 1)
    {h3 : Fin 2 → ExactMatrix q N 1 m} {Bh : Nat}
    (hhelp : Rel sm (lutA model key q m level (sigma2 1))
      (fun j ↦ reducePoly q N (((p : Nat) : Int) *
        unitPair (substitution (sigma2 1).toNat (sigma2_odd 1) sm) j) • g) h3 Bh)
    {rc0 rA00 rA10 rc1 rA01 rA11 : ExactMatrix q N 1 m}
    (hr0 : ringAutomorphismRuns (sigma2 1) (c 0) rc0)
    (hr1 : ringAutomorphismRuns (sigma2 1) (A 0 0) rA00)
    (hr2 : ringAutomorphismRuns (sigma2 1) (A 1 0) rA10)
    (hr3 : ringAutomorphismRuns (sigma2 1) (c 1) rc1)
    (hr4 : ringAutomorphismRuns (sigma2 1) (A 0 1) rA01)
    (hr5 : ringAutomorphismRuns (sigma2 1) (A 1 1) rA11)
    {l01 r01 : ExactMatrix q N (1 + 1) m} {l r : ExactMatrix q N (1 + 1 + 1) m}
    {both : ExactMatrix q N (1 + 1 + 1) (m + m)}
    (hl01 : concatRows rc0 rA00 l01) (hl : concatRows l01 rA10 l)
    (hr01 : concatRows rc1 rA01 r01) (hr : concatRows r01 rA11 r) (hboth : concatColumns l r both)
    {Xb : Fin 2 → ExactMatrix q' N (1 + 1) m'}
    (hXb : ∀ j, concatRows (final.switch 0 j) (final.switch 1 j) (Xb j))
    {D' : Fin 2 → ExactMatrix q' N (m + m) m'}
    (hD' : ∀ j, gadgetDecomposeRuns Backend.backend 32768 m (Xb j) (D' j))
    {D : Fin 2 → ExactMatrix q N (m + m) m'} (hD : ∀ j, compactCenteredRebaseRuns 16384 (D' j) (D j))
    {M : Fin 2 → Fin 2 → ExactMatrix q N 1 m'}
    (hM0 : ∀ j, sliceMatrix (matrixMul both (D j)) 1 2 0 m' (M 0 j))
    (hM1 : ∀ j, sliceMatrix (matrixMul both (D j)) 2 3 0 m' (M 1 j))
    {basis : List Nat} {Asw : Fin 2 → Fin 2 → ExactMatrix q' N 1 m'}
    (hAsw : ∀ k j, blockModSwitchRuns basis 1 (M k j) (Asw k j))
    {Z : Fin 2 → ExactMatrix q' N (1 + 1) m'} (hZ : ∀ j, concatRows (Asw 0 j) (Asw 1 j) (Z j))
    {E' : Fin 2 → ExactMatrix q' N (m + m) m'}
    (hE' : ∀ j, gadgetDecomposeRuns Backend.backend 32768 m (Z j) (E' j))
    {E : Fin 2 → ExactMatrix q N (m + m) m'} (hE : ∀ j, compactCenteredRebaseRuns 16384 (E' j) (E j))
    {hl01' hr01' : ExactMatrix q N (1 + 1) m} {hl' hr' : ExactMatrix q N (1 + 1 + 1) m}
    {H : ExactMatrix q N (1 + 1 + 1) (m + m)}
    (hhl01 : concatRows (h3 0) (lutA model key q m level (sigma2 1) 0 0) hl01')
    (hhl : concatRows hl01' (lutA model key q m level (sigma2 1) 1 0) hl')
    (hhr01 : concatRows (h3 1) (lutA model key q m level (sigma2 1) 0 1) hr01')
    (hhr : concatRows hr01' (lutA model key q m level (sigma2 1) 1 1) hr')
    (hH : concatColumns hl' hr' H)
    {Y X : Fin 2 → ExactMatrix q N 1 m'} {K : Fin 2 → Fin 2 → ExactMatrix q N 1 m'}
    (hY : ∀ j, sliceMatrix (matrixMul H (E j)) 0 1 0 m' (Y j))
    (hX : ∀ j, sliceMatrix (matrixMul both (D j)) 0 1 0 m' (X j))
    (hK0 : ∀ j, sliceMatrix (matrixMul H (E j)) 1 2 0 m' (K 0 j))
    (hK1 : ∀ j, sliceMatrix (matrixMul H (E j)) 2 3 0 m' (K 1 j))
    :
    Rel sm (fun k j ↦ final.key k j + K k j) (fun j ↦ ∑ i : Fin 2, scaleUpMatrix q p
      (reducePoly q' N (substitution (sigma2 i).toNat (sigma2_odd i) (Xe z)) •
        (reducePoly q' N (substitution (sigma2 i).toNat (sigma2_odd i) sp) • fam 0 j i +
          fam 1 j i))) (fun j ↦ final.row j + (Y j + X j))
      (Bl + ((m + m) * N * Bh * 16384 + (m + m) * N * B * 16384 + (N * 1 * (p / 2) + p / 2))) := by
  have hbranch := lut_branch (n := N) (q := q) (q' := q') (p := p) (m := m) (m' := m')
    (log := 3) (bound := 16384) (index := sigma2 1) hN hpow hq hq1 hq'
    layout hlayout hexact hg' hg hsm hinput hr0 hr1 hr2 hr3 hr4 hr5 hl01 hl hr01 hr hboth
    (bsw := final.switch) hXb hD' hD hM0 hM1 hAsw hZ hE' hE (h := h3) (sigma2_odd 1) hhelp
    hhl01 hhl hhr01 hhr hH hY hX hK0 hK1
  rw [hloopSwitch] at hbranch
  have hsum := hloop.add hbranch
  refine hsum.target_eq ?_
  funext j
  rw [Fin.sum_univ_castSucc, Pi.add_apply, Finset.sum_apply, Finset.sum_range]
  refine congrArg₂ (· + ·) (Finset.sum_congr rfl fun i _ ↦ ?_) ?_
  · rw [lutBranchTarget2, dif_pos (by omega)]
    rfl
  · rw [show (Fin.last 1 : Fin 2) = 1 from rfl]
    rfl

set_option maxHeartbeats 64000000 in
/-- A width-`2` LUT round: the loop's first branches (`hloop`), the second branch, and the move down
to `q'`. An encoding of `X^z` at `q` with scale `p` becomes an encoding of
`X^(z - z mod 2) T(z mod 2)` at `q'` with scale `Δ'` against `gq`, given the switching ciphertexts
`hswitch` of the coefficients of `T`. -/
theorem lut_round2 {q q' p m m' : Nat} (hq : q = p * q') (hq1 : 1 < q) (hq' : 1 < q') (hp : 0 < p)
    (layout : RegularLayout q') (hlayout : Backend.backend.regularLayout q' N = some layout)
    (hexact : layout.droppedModuli = 0)
    {model : HashModel} {key : ByteArray} {level : Int}
    {sm sp : ErrorPoly N} (hsm : polyNorm sm ≤ 1) {z : Nat}
    {g' : ExactMatrix q' N 1 m} {g : ExactMatrix q N 1 m}
    (hg' : gadgetMatrixRuns Backend.backend 32768 m g') (hg : centeredRebaseRuns g' g)
    {c : Fin 2 → ExactMatrix q N 1 m} {A : Fin 2 → Fin 2 → ExactMatrix q N 1 m} {B : Nat}
    (hinput : Rel sm A (fun j ↦ reducePoly q N (((p : Nat) : Int) * Xe (n := N) z *
      unitPair sp j) • g) c B)
    {fam : Fin 2 → Fin 2 → Fin 2 → ExactMatrix q' N 1 m'} {final : LoopState q q' m'} {Bl : Nat}
    (hloop : Rel sm final.key (∑ i ∈ Finset.range 1, lutBranchTarget2 (q := q) (p := p)
      (Xe z) sp fam i) final.row Bl)
    (hloopSwitch : final.switch = fun k j ↦ fam k j 1)
    {h3 : Fin 2 → ExactMatrix q N 1 m} {Bh : Nat}
    (hhelp : Rel sm (lutA model key q m level (sigma2 1))
      (fun j ↦ reducePoly q N (((p : Nat) : Int) *
        unitPair (substitution (sigma2 1).toNat (sigma2_odd 1) sm) j) • g) h3 Bh)
    {rc0 rA00 rA10 rc1 rA01 rA11 : ExactMatrix q N 1 m}
    (hr0 : ringAutomorphismRuns (sigma2 1) (c 0) rc0)
    (hr1 : ringAutomorphismRuns (sigma2 1) (A 0 0) rA00)
    (hr2 : ringAutomorphismRuns (sigma2 1) (A 1 0) rA10)
    (hr3 : ringAutomorphismRuns (sigma2 1) (c 1) rc1)
    (hr4 : ringAutomorphismRuns (sigma2 1) (A 0 1) rA01)
    (hr5 : ringAutomorphismRuns (sigma2 1) (A 1 1) rA11)
    {l01 r01 : ExactMatrix q N (1 + 1) m} {l r : ExactMatrix q N (1 + 1 + 1) m}
    {both : ExactMatrix q N (1 + 1 + 1) (m + m)}
    (hl01 : concatRows rc0 rA00 l01) (hl : concatRows l01 rA10 l)
    (hr01 : concatRows rc1 rA01 r01) (hr : concatRows r01 rA11 r) (hboth : concatColumns l r both)
    {Xb : Fin 2 → ExactMatrix q' N (1 + 1) m'}
    (hXb : ∀ j, concatRows (final.switch 0 j) (final.switch 1 j) (Xb j))
    {D' : Fin 2 → ExactMatrix q' N (m + m) m'}
    (hD' : ∀ j, gadgetDecomposeRuns Backend.backend 32768 m (Xb j) (D' j))
    {D : Fin 2 → ExactMatrix q N (m + m) m'} (hD : ∀ j, compactCenteredRebaseRuns 16384 (D' j) (D j))
    {M : Fin 2 → Fin 2 → ExactMatrix q N 1 m'}
    (hM0 : ∀ j, sliceMatrix (matrixMul both (D j)) 1 2 0 m' (M 0 j))
    (hM1 : ∀ j, sliceMatrix (matrixMul both (D j)) 2 3 0 m' (M 1 j))
    {basis : List Nat} {Asw : Fin 2 → Fin 2 → ExactMatrix q' N 1 m'}
    (hAsw : ∀ k j, blockModSwitchRuns basis 1 (M k j) (Asw k j))
    {Z : Fin 2 → ExactMatrix q' N (1 + 1) m'} (hZ : ∀ j, concatRows (Asw 0 j) (Asw 1 j) (Z j))
    {E' : Fin 2 → ExactMatrix q' N (m + m) m'}
    (hE' : ∀ j, gadgetDecomposeRuns Backend.backend 32768 m (Z j) (E' j))
    {E : Fin 2 → ExactMatrix q N (m + m) m'} (hE : ∀ j, compactCenteredRebaseRuns 16384 (E' j) (E j))
    {hl01' hr01' : ExactMatrix q N (1 + 1) m} {hl' hr' : ExactMatrix q N (1 + 1 + 1) m}
    {H : ExactMatrix q N (1 + 1 + 1) (m + m)}
    (hhl01 : concatRows (h3 0) (lutA model key q m level (sigma2 1) 0 0) hl01')
    (hhl : concatRows hl01' (lutA model key q m level (sigma2 1) 1 0) hl')
    (hhr01 : concatRows (h3 1) (lutA model key q m level (sigma2 1) 0 1) hr01')
    (hhr : concatRows hr01' (lutA model key q m level (sigma2 1) 1 1) hr')
    (hH : concatColumns hl' hr' H)
    {Y X : Fin 2 → ExactMatrix q N 1 m'} {K : Fin 2 → Fin 2 → ExactMatrix q N 1 m'}
    (hY : ∀ j, sliceMatrix (matrixMul H (E j)) 0 1 0 m' (Y j))
    (hX : ∀ j, sliceMatrix (matrixMul both (D j)) 0 1 0 m' (X j))
    (hK0 : ∀ j, sliceMatrix (matrixMul H (E j)) 1 2 0 m' (K 0 j))
    (hK1 : ∀ j, sliceMatrix (matrixMul H (E j)) 2 3 0 m' (K 1 j))
    {T : Fin 2 → ErrorPoly N} {Δ' : ErrorPoly N} {gq : ExactMatrix q' N 1 m'} {inverse : Nat}
    (hinverse : (inverse : ExactPoly q' N) * ((2 : Nat) : ExactPoly q' N) = 1) {BB : Nat}
    (hswitch : ∀ i : Fin 2, ∃ e : Fin 2 → ErrorMatrix N 1 m', (∀ j, CoeffBound (e j) BB) ∧ ∀ j,
      reducePoly q' N (substitution (sigma2 i).toNat (sigma2_odd i) sp) • fam 0 j i + fam 1 j i =
        (lutCoefficient2 q' inverse (fun k ↦ reducePoly q' N (T k)) (sigma2 i).toNat *
          reducePoly q' N (Δ' * unitPair sp j)) • gq + reduceMatrix q' N 1 m' (e j))
    {basis' : List Nat} {K' : Fin 2 → Fin 2 → ExactMatrix q' N 1 m'} {R' : Fin 2 → ExactMatrix q' N 1 m'}
    (hK' : ∀ k j, blockModSwitchRuns basis' 1 (final.key k j + K k j) (K' k j))
    (hR' : ∀ j, blockModSwitchRuns basis' 1 (final.row j + (Y j + X j)) (R' j)) :
    Rel sm K' (fun j ↦ reducePoly q' N (Δ' * (Xe (z - z % 2) * T ⟨z % 2, Nat.mod_lt _ (by norm_num)⟩) * unitPair sp j) • gq) R'
      ((Bl + ((m + m) * N * Bh * 16384 + (m + m) * N * B * 16384 + (N * 1 * (p / 2) + p / 2)) +
        (N + 2) * (p / 2)) / p + 2 * (N * 1 * BB)) := by
  have hsum' := lut_branches2 hq hq1 hq' hp layout hlayout hexact hsm hg' hg hinput hloop
    hloopSwitch hhelp hr0 hr1 hr2 hr3 hr4 hr5 hl01 hl hr01 hr hboth hXb hD' hD hM0 hM1 hAsw hZ hE' hE
    hhl01 hhl hhr01 hhr hH hY hX hK0 hK1
  choose eS heS hS using hswitch
  have hwalsh := lut2_walsh_value (q := q') T hinverse z
  exact lut_finish (W := 2) (q := q) (q' := q') (p := p) (m' := m') hN hq hq1 hq' hp
    (sp := sp) (μ := Xe z) (Δ' := Δ') hsm (fun i ↦ substitution (sigma2 i).toNat (sigma2_odd i))
    (fun i ↦ polyNorm_substitution2_Xe i z) (bsw := fun i k j ↦ fam k j i) hsum' (g' := gq)
    (C := fun i ↦ lutCoefficient2 q' inverse (fun k ↦ reducePoly q' N (T k)) (sigma2 i).toNat)
    (fun i j ↦ hS i j) (fun i j ↦ heS i j) hwalsh hK' hR'

end RefreshTopProof
