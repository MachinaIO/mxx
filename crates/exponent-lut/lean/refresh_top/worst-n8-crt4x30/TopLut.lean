import TopLemmas
import ExponentWalsh
import ExponentAutomorphism

/-!
# LUT loops

A LUT of width `4` is the sum of four branches, one per automorphism `σ_i ∈ {1, 3, 5, 15}`: the
generated loop accumulates the first three in a state of six sums and carries the switching
ciphertext of the next branch, and the fourth is in the loop's caller.
-/

namespace RefreshTopProof

open Mxx.Primitives MxxRuntime MxxExponentLut

/-- The six accumulated sums and the carried switching ciphertext of a LUT loop state. -/
abbrev LoopState (q q' m : Nat) :=
  ExactMatrix q N 1 m × ExactMatrix q N 1 m × ExactMatrix q N 1 m × ExactMatrix q N 1 m ×
    ExactMatrix q N 1 m × ExactMatrix q N 1 m × ExactMatrix q' N 1 m × ExactMatrix q' N 1 m ×
      ExactMatrix q' N 1 m × ExactMatrix q' N 1 m × Unit

/-- The accumulated row. -/
def LoopState.row {q q' m : Nat} (s : LoopState q q' m) : Fin 2 → ExactMatrix q N 1 m :=
  ![s.1, s.2.1]

/-- The accumulated key. -/
def LoopState.key {q q' m : Nat} (s : LoopState q q' m) : Fin 2 → Fin 2 → ExactMatrix q N 1 m :=
  ![![s.2.2.1, s.2.2.2.1], ![s.2.2.2.2.1, s.2.2.2.2.2.1]]

/-- The carried switching ciphertext. -/
def LoopState.switch {q q' m : Nat} (s : LoopState q q' m) : Fin 2 → Fin 2 → ExactMatrix q' N 1 m :=
  ![![s.2.2.2.2.2.2.1, s.2.2.2.2.2.2.2.1], ![s.2.2.2.2.2.2.2.2.1, s.2.2.2.2.2.2.2.2.2.1]]

/-- The automorphism indices of a width-`4` LUT: one odd residue modulo `8` each. -/
abbrev sigmas : List Int := [(1), (3), (5), (15)]

/-- The automorphism of branch `i`. -/
abbrev sigma (i : Nat) : Int := MxxIR.selectInt (Int.ofNat i) sigmas

theorem sigma_odd (i : Fin 4) : Odd (sigma i).toNat := by
  fin_cases i <;> decide

/-- `rns-bgg/lut-a`, the tag of the LUT helpers. -/
def lutAPrefix : Blob := [114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]

/-- The public matrix `A_σ` of a LUT helper at `level` for the automorphism `sigma`. -/
noncomputable def lutA (model : HashModel) (key : ByteArray) (q m : Nat) (level : Int)
    (sigma : Int) : Fin 2 → Fin 2 → ExactMatrix q N 1 m :=
  fun k j ↦ model.sample q N 1 m key (completeHashTag lutAPrefix
    [.u64Le level, .integer sigma, .bytes componentBytes, .u64Le (2 * (k : Int) + (j : Int))])

/-- The target of branch `i` of a LUT over the switching families `fam`. -/
noncomputable def branchTarget {q q' p m : Nat} (μ sp : ErrorPoly N)
    (fam : Fin 2 → Fin 2 → Fin 4 → ExactMatrix q' N 1 m) (i : Fin 4) :
    Fin 2 → ExactMatrix q N 1 m :=
  fun j ↦ scaleUpMatrix q p (reducePoly q' N (substitution (sigma i).toNat (sigma_odd i) μ) •
    (reducePoly q' N (substitution (sigma i).toNat (sigma_odd i) sp) • fam 0 j i + fam 1 j i))

/-- Branch `i` of a LUT, zero past the four branches. -/
noncomputable def lutBranchTarget {q q' p m : Nat} (μ sp : ErrorPoly N)
    (fam : Fin 2 → Fin 2 → Fin 4 → ExactMatrix q' N 1 m) (i : Nat) : Fin 2 → ExactMatrix q N 1 m :=
  if h : i < 4 then branchTarget (q := q) (p := p) μ sp fam ⟨i, h⟩ else 0

/-! ## The interpolation -/

/-- The LUT variable `Y = X^2` of stride `2`; `Y^4 = -1`. -/
noncomputable abbrev Y (q : Nat) : ExactPoly q N := X q ^ 2

theorem Y_pow_4 {q : Nat} : Y q ^ 4 = -1 := by
  rw [Y, ← pow_mul]
  exact Negacyclic.root_pow_n

/-- `σ_i ≡ 1 + 2 i (mod 8)`. -/
theorem sigma_mod (i : Fin 4) : (sigma i).toNat % (2 * 4) = 1 + 2 * 1 * ((Equiv.refl (Fin 4)) i).val := by
  fin_cases i <;> decide

/-- The coefficient `C_σ = 4⁻¹ Σ_k T(k) σ(Y^(-k))` of a width-`4` LUT with table `T`. -/
noncomputable def lutCoefficient (q inverse : Nat) (T : Fin 4 → ExactPoly q N) (σ : Nat) :
    ExactPoly q N :=
  lutFourierCoefficient 4 (Y q) (inverse : ExactPoly q N) T σ

/-- A width-`4` LUT on every exponent `z`: `Σ_i σ_i(Y^z) C_i = Y^(z - z mod 4) T(z mod 4)`. -/
theorem lut_walsh {q inverse : Nat} (T : Fin 4 → ExactPoly q N)
    (hinverse : (inverse : ExactPoly q N) * ((4 : Nat) : ExactPoly q N) = 1) (z : Nat) :
    ∑ i : Fin 4, Y q ^ ((sigma i).toNat * z) * lutCoefficient q inverse T (sigma i).toNat =
      Y q ^ (z - z % 4) * T ⟨z % 4, Nat.mod_lt _ (by norm_num)⟩ := by
  have h := lut_fourier_interpolation (R := ExactPoly q N) 2 4 1 z (Y q) (inverse : ExactPoly q N)
    T (by norm_num) (by norm_num) Y_pow_4 hinverse
    (fun i ↦ (sigma i).toNat) (Equiv.refl _) sigma_mod
  simpa [lutCoefficient] using h

theorem polyNorm_substitution_Xe (i : Fin 4) (k : Nat) :
    polyNorm (substitution (sigma i).toNat (sigma_odd i) (Xe (n := N) k)) ≤ 1 := by
  rw [substitution_root_pow]
  exact polyNorm_Xe hN1 _

/-- A width-`4` LUT on the state `X^(2z)`: `(-1)^⌊z/4⌋ T(z mod 4)`. -/
theorem lut_walsh_value {q inverse : Nat} (T : Fin 4 → ErrorPoly N)
    (hinverse : (inverse : ExactPoly q N) * ((4 : Nat) : ExactPoly q N) = 1) (z : Nat) :
    ∑ i : Fin 4, reducePoly q N (substitution (sigma i).toNat (sigma_odd i) (Xe (2 * z))) *
        lutCoefficient q inverse (fun k ↦ reducePoly q N (T k)) (sigma i).toNat =
      reducePoly q N ((((-1 : Int) ^ (z / 4) : Int) : ErrorPoly N) *
        T ⟨z % 4, Nat.mod_lt _ (by norm_num)⟩) := by
  have h := lut_walsh (q := q) (fun k ↦ reducePoly q N (T k)) hinverse z
  have hY : Y q ^ (z - z % 4) = (-1) ^ (z / 4) := by
    rw [show z - z % 4 = 4 * (z / 4) by have := Nat.div_add_mod z 4; omega, pow_mul, Y_pow_4]
  rw [hY] at h
  have hc : reducePoly q N ((((-1 : Int) ^ (z / 4) : Int) : ErrorPoly N)) = (-1) ^ (z / 4) := by
    simp
  rw [map_mul, hc, ← h]
  · apply Finset.sum_congr rfl
    intro i _
    rw [substitution_root_pow, reducePoly_Xe, Y, ← pow_mul]
    congr 2
    ring

/-! ## A LUT round -/

set_option maxHeartbeats 64000000 in
/-- The four branches of a width-`4` LUT before its move down: the loop's three (`hloop`) and the
fourth. -/
theorem lut_branches {q q' p m m' : Nat} (hq : q = p * q') (hq1 : 1 < q) (hq' : 1 < q') (hp : 0 < p)
    (layout : RegularLayout q') (hlayout : Backend.backend.regularLayout q' N = some layout)
    (hexact : layout.droppedModuli = 0)
    {model : HashModel} {key : ByteArray} {level : Int}
    {sm sp : ErrorPoly N} (hsm : polyNorm sm ≤ 1) {z : Nat}
    {g' : ExactMatrix q' N 1 m} {g : ExactMatrix q N 1 m}
    (hg' : gadgetMatrixRuns Backend.backend 32768 m g') (hg : centeredRebaseRuns g' g)
    {c : Fin 2 → ExactMatrix q N 1 m} {A : Fin 2 → Fin 2 → ExactMatrix q N 1 m} {B : Nat}
    (hinput : Rel sm A (fun j ↦ reducePoly q N (((p : Nat) : Int) * Xe (n := N) (2 * z) *
      unitPair sp j) • g) c B)
    {fam : Fin 2 → Fin 2 → Fin 4 → ExactMatrix q' N 1 m'} {final : LoopState q q' m'} {Bl : Nat}
    (hloop : Rel sm final.key (∑ i ∈ Finset.range 3, lutBranchTarget (q := q) (p := p)
      (Xe (2 * z)) sp fam i) final.row Bl)
    (hloopSwitch : final.switch = fun k j ↦ fam k j 3)
    {h3 : Fin 2 → ExactMatrix q N 1 m} {Bh : Nat}
    (hhelp : Rel sm (lutA model key q m level (sigma 3))
      (fun j ↦ reducePoly q N (((p : Nat) : Int) *
        unitPair (substitution (sigma 3).toNat (sigma_odd 3) sm) j) • g) h3 Bh)
    {rc0 rA00 rA10 rc1 rA01 rA11 : ExactMatrix q N 1 m}
    (hr0 : ringAutomorphismRuns (sigma 3) (c 0) rc0)
    (hr1 : ringAutomorphismRuns (sigma 3) (A 0 0) rA00)
    (hr2 : ringAutomorphismRuns (sigma 3) (A 1 0) rA10)
    (hr3 : ringAutomorphismRuns (sigma 3) (c 1) rc1)
    (hr4 : ringAutomorphismRuns (sigma 3) (A 0 1) rA01)
    (hr5 : ringAutomorphismRuns (sigma 3) (A 1 1) rA11)
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
    (hhl01 : concatRows (h3 0) (lutA model key q m level (sigma 3) 0 0) hl01')
    (hhl : concatRows hl01' (lutA model key q m level (sigma 3) 1 0) hl')
    (hhr01 : concatRows (h3 1) (lutA model key q m level (sigma 3) 0 1) hr01')
    (hhr : concatRows hr01' (lutA model key q m level (sigma 3) 1 1) hr')
    (hH : concatColumns hl' hr' H)
    {Y X : Fin 2 → ExactMatrix q N 1 m'} {K : Fin 2 → Fin 2 → ExactMatrix q N 1 m'}
    (hY : ∀ j, sliceMatrix (matrixMul H (E j)) 0 1 0 m' (Y j))
    (hX : ∀ j, sliceMatrix (matrixMul both (D j)) 0 1 0 m' (X j))
    (hK0 : ∀ j, sliceMatrix (matrixMul H (E j)) 1 2 0 m' (K 0 j))
    (hK1 : ∀ j, sliceMatrix (matrixMul H (E j)) 2 3 0 m' (K 1 j))
    :
    Rel sm (fun k j ↦ final.key k j + K k j) (fun j ↦ ∑ i : Fin 4, scaleUpMatrix q p
      (reducePoly q' N (substitution (sigma i).toNat (sigma_odd i) (Xe (2 * z))) •
        (reducePoly q' N (substitution (sigma i).toNat (sigma_odd i) sp) • fam 0 j i +
          fam 1 j i))) (fun j ↦ final.row j + (Y j + X j))
      (Bl + ((m + m) * N * Bh * 16384 + (m + m) * N * B * 16384 + (N * 1 * (p / 2) + p / 2))) := by
  have hbranch := lut_branch (n := N) (q := q) (q' := q') (p := p) (m := m) (m' := m')
    (log := 3) (bound := 16384) (index := sigma 3) hN hpow hq hq1 hq'
    layout hlayout hexact hg' hg hsm hinput hr0 hr1 hr2 hr3 hr4 hr5 hl01 hl hr01 hr hboth
    (bsw := final.switch) hXb hD' hD hM0 hM1 hAsw hZ hE' hE (h := h3) (sigma_odd 3) hhelp
    hhl01 hhl hhr01 hhr hH hY hX hK0 hK1
  rw [hloopSwitch] at hbranch
  have hsum := hloop.add hbranch
  refine hsum.target_eq ?_
  funext j
  rw [Fin.sum_univ_castSucc, Pi.add_apply, Finset.sum_apply, Finset.sum_range]
  refine congrArg₂ (· + ·) (Finset.sum_congr rfl fun i _ ↦ ?_) ?_
  · rw [lutBranchTarget, dif_pos (by omega)]
    rfl
  · rw [show (Fin.last 3 : Fin 4) = 3 from rfl]
    rfl

set_option maxHeartbeats 64000000 in
/-- A width-`4` LUT round: the loop's three branches (`hloop`), the fourth branch, and the move down
to `q'`. An encoding of `X^(2z)` at `q` with scale `p` becomes an encoding of
`(-1)^⌊z/4⌋ T(z mod 4)` at `q'` with scale `Δ'` against `gq`, given the switching ciphertexts
`hswitch` of the coefficients of `T`. -/
theorem lut_round {q q' p m m' : Nat} (hq : q = p * q') (hq1 : 1 < q) (hq' : 1 < q') (hp : 0 < p)
    (layout : RegularLayout q') (hlayout : Backend.backend.regularLayout q' N = some layout)
    (hexact : layout.droppedModuli = 0)
    {model : HashModel} {key : ByteArray} {level : Int}
    {sm sp : ErrorPoly N} (hsm : polyNorm sm ≤ 1) {z : Nat}
    {g' : ExactMatrix q' N 1 m} {g : ExactMatrix q N 1 m}
    (hg' : gadgetMatrixRuns Backend.backend 32768 m g') (hg : centeredRebaseRuns g' g)
    {c : Fin 2 → ExactMatrix q N 1 m} {A : Fin 2 → Fin 2 → ExactMatrix q N 1 m} {B : Nat}
    (hinput : Rel sm A (fun j ↦ reducePoly q N (((p : Nat) : Int) * Xe (n := N) (2 * z) *
      unitPair sp j) • g) c B)
    {fam : Fin 2 → Fin 2 → Fin 4 → ExactMatrix q' N 1 m'} {final : LoopState q q' m'} {Bl : Nat}
    (hloop : Rel sm final.key (∑ i ∈ Finset.range 3, lutBranchTarget (q := q) (p := p)
      (Xe (2 * z)) sp fam i) final.row Bl)
    (hloopSwitch : final.switch = fun k j ↦ fam k j 3)
    {h3 : Fin 2 → ExactMatrix q N 1 m} {Bh : Nat}
    (hhelp : Rel sm (lutA model key q m level (sigma 3))
      (fun j ↦ reducePoly q N (((p : Nat) : Int) *
        unitPair (substitution (sigma 3).toNat (sigma_odd 3) sm) j) • g) h3 Bh)
    {rc0 rA00 rA10 rc1 rA01 rA11 : ExactMatrix q N 1 m}
    (hr0 : ringAutomorphismRuns (sigma 3) (c 0) rc0)
    (hr1 : ringAutomorphismRuns (sigma 3) (A 0 0) rA00)
    (hr2 : ringAutomorphismRuns (sigma 3) (A 1 0) rA10)
    (hr3 : ringAutomorphismRuns (sigma 3) (c 1) rc1)
    (hr4 : ringAutomorphismRuns (sigma 3) (A 0 1) rA01)
    (hr5 : ringAutomorphismRuns (sigma 3) (A 1 1) rA11)
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
    (hhl01 : concatRows (h3 0) (lutA model key q m level (sigma 3) 0 0) hl01')
    (hhl : concatRows hl01' (lutA model key q m level (sigma 3) 1 0) hl')
    (hhr01 : concatRows (h3 1) (lutA model key q m level (sigma 3) 0 1) hr01')
    (hhr : concatRows hr01' (lutA model key q m level (sigma 3) 1 1) hr')
    (hH : concatColumns hl' hr' H)
    {Y X : Fin 2 → ExactMatrix q N 1 m'} {K : Fin 2 → Fin 2 → ExactMatrix q N 1 m'}
    (hY : ∀ j, sliceMatrix (matrixMul H (E j)) 0 1 0 m' (Y j))
    (hX : ∀ j, sliceMatrix (matrixMul both (D j)) 0 1 0 m' (X j))
    (hK0 : ∀ j, sliceMatrix (matrixMul H (E j)) 1 2 0 m' (K 0 j))
    (hK1 : ∀ j, sliceMatrix (matrixMul H (E j)) 2 3 0 m' (K 1 j))
    {T : Fin 4 → ErrorPoly N} {Δ' : ErrorPoly N} {gq : ExactMatrix q' N 1 m'} {inverse : Nat}
    (hinverse : (inverse : ExactPoly q' N) * ((4 : Nat) : ExactPoly q' N) = 1) {BB : Nat}
    (hswitch : ∀ i : Fin 4, ∃ e : Fin 2 → ErrorMatrix N 1 m', (∀ j, CoeffBound (e j) BB) ∧ ∀ j,
      reducePoly q' N (substitution (sigma i).toNat (sigma_odd i) sp) • fam 0 j i + fam 1 j i =
        (lutCoefficient q' inverse (fun k ↦ reducePoly q' N (T k)) (sigma i).toNat *
          reducePoly q' N (Δ' * unitPair sp j)) • gq + reduceMatrix q' N 1 m' (e j))
    {basis' : List Nat} {K' : Fin 2 → Fin 2 → ExactMatrix q' N 1 m'} {R' : Fin 2 → ExactMatrix q' N 1 m'}
    (hK' : ∀ k j, blockModSwitchRuns basis' 1 (final.key k j + K k j) (K' k j))
    (hR' : ∀ j, blockModSwitchRuns basis' 1 (final.row j + (Y j + X j)) (R' j)) :
    Rel sm K' (fun j ↦ reducePoly q' N (Δ' * ((((-1 : Int) ^ (z / 4) : Int) : ErrorPoly N) *
        T ⟨z % 4, Nat.mod_lt _ (by norm_num)⟩) * unitPair sp j) • gq) R'
      ((Bl + ((m + m) * N * Bh * 16384 + (m + m) * N * B * 16384 + (N * 1 * (p / 2) + p / 2)) +
        (N + 2) * (p / 2)) / p + 4 * (N * 1 * BB)) := by
  have hsum' := lut_branches hq hq1 hq' hp layout hlayout hexact hsm hg' hg hinput hloop
    hloopSwitch hhelp hr0 hr1 hr2 hr3 hr4 hr5 hl01 hl hr01 hr hboth hXb hD' hD hM0 hM1 hAsw hZ hE' hE
    hhl01 hhl hhr01 hhr hH hY hX hK0 hK1
  choose eS heS hS using hswitch
  have hwalsh := lut_walsh_value (q := q') T hinverse z
  exact lut_finish (W := 4) (q := q) (q' := q') (p := p) (m' := m') hN hq hq1 hq' hp
    (sp := sp) (μ := Xe (2 * z)) (Δ' := Δ') hsm (fun i ↦ substitution (sigma i).toNat (sigma_odd i))
    (fun i ↦ polyNorm_substitution_Xe i (2 * z)) (bsw := fun i k j ↦ fam k j i) hsum' (g' := gq)
    (C := fun i ↦ lutCoefficient q' inverse (fun k ↦ reducePoly q' N (T k)) (sigma i).toNat)
    (fun i j ↦ hS i j) (fun i j ↦ heS i j) hwalsh hK' hR'

end RefreshTopProof
