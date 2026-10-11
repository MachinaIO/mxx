import TopKeys
import TopFinish
import ExponentPublic
import ExponentFunctional

/-!
# Canonical key forms

The keys of a LUT branch, written with canonical digits: `dec X` is the decomposition the runtime
relation determines, `rdec X` its centered rebase to the evaluation modulus.
-/

namespace RefreshTopProof

open Mxx.Primitives MxxRuntime MxxExponentLut

theorem preimageWithin_zero {q rows columns : Nat} (bound : Nat) :
    PreimageWithin (0 : ExactMatrix q N rows columns) bound :=
  ⟨0, by funext r c; simp [reduceMatrix], coeffBound_mono coeffBound_zero (Nat.zero_le _)⟩

/-- Every target has its regular decomposition at an exact layout. -/
theorem gadgetDecomposeRuns_exists {q rows columns inner : Nat} {backend : BackendContext}
    {base digits : Int} (layout : RegularLayout q) (hlayout : backend.regularLayout q N = some layout)
    (hexact : layout.droppedModuli = 0) (hq : 1 < q) (hbase : base = layout.base)
    (hdigits : digits = layout.digitCount) (hinner : inner = rows * layout.digitCount)
    (X : ExactMatrix q N rows columns) :
    ∃ D : ExactMatrix q N inner columns, gadgetDecomposeRuns backend base digits X D := by
  refine ⟨castMatrixRows hinner.symm (regularDecomposeMatrix layout X),
    layout, hlayout, hbase, hdigits, hinner, rfl, ?_⟩
  rw [regularGadgetMatrix_reconstruct layout X hq hN hexact, sub_self]
  exact preimageWithin_zero _

/-- The decomposition the runtime relation determines. -/
noncomputable def dec {q rows columns inner : Nat} (digits : Int) (X : ExactMatrix q N rows columns) :
    ExactMatrix q N inner columns :=
  Classical.epsilon (fun D ↦ gadgetDecomposeRuns Backend.backend 32768 digits X D)

theorem dec_eq {q rows columns inner : Nat} {digits : Int} {X : ExactMatrix q N rows columns}
    {D : ExactMatrix q N inner columns} (h : gadgetDecomposeRuns Backend.backend 32768 digits X D) :
    D = dec digits X :=
  gadgetDecomposeRuns_unique h (Classical.epsilon_spec ⟨D, h⟩)

/-- The rebased digits of a one-row target. -/
noncomputable def rdec {q' q columns inner : Nat} (digits : Int) (X : ExactMatrix q' N 1 columns) :
    ExactMatrix q N inner columns :=
  centeredRebase (dec digits X : ExactMatrix q' N inner columns)

/-- The keys of a LUT branch with automorphism `σ`: the input keys `A` routed by `σ` and switched
by the branch's ciphertexts `sw` to the next modulus, then multiplied by the helper keys `Ad`. -/
noncomputable def branchKey {q q' m m' : Nat} (digits : Int) (basis : List Nat) (σ : Nat)
    (A Ad : Fin 2 → Fin 2 → ExactMatrix q N 1 m) (sw : Fin 2 → Fin 2 → ExactMatrix q' N 1 m') :
    Fin 2 → Fin 2 → ExactMatrix q N 1 m' :=
  fun k j ↦
    let Asw : Fin 2 → ExactMatrix q' N 1 m' := fun k' ↦ blockModSwitch basis 1
      (ringAutomorphism σ (A k' 0) * rdec (inner := m) digits (sw 0 j) +
        ringAutomorphism σ (A k' 1) * rdec (inner := m) digits (sw 1 j))
    Ad k 0 * rdec (inner := m) digits (Asw 0) + Ad k 1 * rdec (inner := m) digits (Asw 1)

/-- The canonical digits decompose: given any decomposition at the same modulus and digit count. -/
theorem dec_of_runs {q rows rows' columns columns' inner inner' : Nat} {digits : Int}
    (layout : RegularLayout q) (hlayout : Backend.backend.regularLayout q N = some layout)
    (hexact : layout.droppedModuli = 0) (hq : 1 < q) {Y : ExactMatrix q N rows' columns'}
    {DY : ExactMatrix q N inner' columns'}
    (hY : gadgetDecomposeRuns Backend.backend 32768 digits Y DY)
    (hinner : inner = rows * layout.digitCount) (X : ExactMatrix q N rows columns) :
    gadgetDecomposeRuns Backend.backend 32768 digits X (dec (inner := inner) digits X) := by
  obtain ⟨l, hl, hb, hd, _⟩ := hY
  rw [hlayout] at hl
  cases hl
  obtain ⟨D, hD⟩ := gadgetDecomposeRuns_exists layout hlayout hexact hq hb hd hinner X
  rw [← dec_eq hD]
  exact hD

theorem rebase_top {q p m columns bound : Nat} {D' : ExactMatrix q N (m + m) columns}
    {D : ExactMatrix p N (m + m) columns} (hD : compactCenteredRebaseRuns bound D' D) :
    topRows D = centeredRebase (topRows D') ∧ bottomRows D = centeredRebase (bottomRows D') := by
  rw [hD.2.2.1.2]
  exact ⟨rfl, rfl⟩

set_option maxHeartbeats 4000000 in
/-- The keys of a LUT branch as the online evaluation computes them (three-row stacks). -/
theorem branchKey_of_stacks {q q' m m' : Nat} (hq' : 1 < q') (layout : RegularLayout q')
    (hlayout : Backend.backend.regularLayout q' N = some layout) (hexact : layout.droppedModuli = 0)
    (hm : m = 1 * layout.digitCount) {digits : Int}
    {c : Fin 2 → ExactMatrix q N 1 m} {A : Fin 2 → Fin 2 → ExactMatrix q N 1 m}
    {index : Int} {rc0 rA00 rA10 rc1 rA01 rA11 : ExactMatrix q N 1 m}
    (hr0 : ringAutomorphismRuns index (c 0) rc0) (hr1 : ringAutomorphismRuns index (A 0 0) rA00)
    (hr2 : ringAutomorphismRuns index (A 1 0) rA10) (hr3 : ringAutomorphismRuns index (c 1) rc1)
    (hr4 : ringAutomorphismRuns index (A 0 1) rA01) (hr5 : ringAutomorphismRuns index (A 1 1) rA11)
    {l01 r01 : ExactMatrix q N (1 + 1) m} {l r : ExactMatrix q N (1 + 1 + 1) m}
    {both : ExactMatrix q N (1 + 1 + 1) (m + m)}
    (hl01 : concatRows rc0 rA00 l01) (hl : concatRows l01 rA10 l)
    (hr01 : concatRows rc1 rA01 r01) (hr : concatRows r01 rA11 r) (hboth : concatColumns l r both)
    {sw : Fin 2 → Fin 2 → ExactMatrix q' N 1 m'} {Xb : Fin 2 → ExactMatrix q' N (1 + 1) m'}
    (hXb : ∀ j, concatRows (sw 0 j) (sw 1 j) (Xb j))
    {D' : Fin 2 → ExactMatrix q' N (m + m) m'}
    (hD' : ∀ j, gadgetDecomposeRuns Backend.backend 32768 digits (Xb j) (D' j))
    {D : Fin 2 → ExactMatrix q N (m + m) m'} (hD : ∀ j, compactCenteredRebaseRuns 16384 (D' j) (D j))
    {M : Fin 2 → Fin 2 → ExactMatrix q N 1 m'}
    (hM0 : ∀ j, sliceMatrix (matrixMul both (D j)) 1 2 0 m' (M 0 j))
    (hM1 : ∀ j, sliceMatrix (matrixMul both (D j)) 2 3 0 m' (M 1 j))
    {basis : List Nat} {Asw : Fin 2 → Fin 2 → ExactMatrix q' N 1 m'}
    (hAsw : ∀ k j, blockModSwitchRuns basis 1 (M k j) (Asw k j))
    {Z : Fin 2 → ExactMatrix q' N (1 + 1) m'} (hZ : ∀ j, concatRows (Asw 0 j) (Asw 1 j) (Z j))
    {E' : Fin 2 → ExactMatrix q' N (m + m) m'}
    (hE' : ∀ j, gadgetDecomposeRuns Backend.backend 32768 digits (Z j) (E' j))
    {E : Fin 2 → ExactMatrix q N (m + m) m'} (hE : ∀ j, compactCenteredRebaseRuns 16384 (E' j) (E j))
    {h : Fin 2 → ExactMatrix q N 1 m} {Ad : Fin 2 → Fin 2 → ExactMatrix q N 1 m}
    {hl01' hr01' : ExactMatrix q N (1 + 1) m} {hl' hr' : ExactMatrix q N (1 + 1 + 1) m}
    {H : ExactMatrix q N (1 + 1 + 1) (m + m)}
    (hhl01 : concatRows (h 0) (Ad 0 0) hl01') (hhl : concatRows hl01' (Ad 1 0) hl')
    (hhr01 : concatRows (h 1) (Ad 0 1) hr01') (hhr : concatRows hr01' (Ad 1 1) hr')
    (hH : concatColumns hl' hr' H)
    {K : Fin 2 → Fin 2 → ExactMatrix q N 1 m'}
    (hK0 : ∀ j, sliceMatrix (matrixMul H (E j)) 1 2 0 m' (K 0 j))
    (hK1 : ∀ j, sliceMatrix (matrixMul H (E j)) 2 3 0 m' (K 1 j)) :
    K = branchKey digits basis index.toNat A Ad sw := by
  have hdec := fun (X : ExactMatrix q' N 1 m') ↦
    dec_of_runs (inner := m) layout hlayout hexact hq' (hD' 0) hm X
  -- The routed keys.
  have hσ : ∀ {x y : ExactMatrix q N 1 m}, ringAutomorphismRuns index x y →
      y = ringAutomorphism index.toNat x := fun hxy ↦ hxy.2.2.2
  -- The digits of the switching ciphertexts and of the switched keys.
  have hDtop : ∀ j, topRows (D j) = rdec (inner := m) digits (sw 0 j) ∧
      bottomRows (D j) = rdec (inner := m) digits (sw 1 j) := by
    intro j
    obtain ⟨t, b⟩ := decompose_stack layout hlayout hexact (hXb j) (hD' j)
      (hdec (sw 0 j)) (hdec (sw 1 j))
    obtain ⟨t', b'⟩ := rebase_top (hD j)
    exact ⟨by rw [t', t]; rfl, by rw [b', b]; rfl⟩
  have hEtop : ∀ j, topRows (E j) = rdec (inner := m) digits (Asw 0 j) ∧
      bottomRows (E j) = rdec (inner := m) digits (Asw 1 j) := by
    intro j
    obtain ⟨t, b⟩ := decompose_stack layout hlayout hexact (hZ j) (hE' j)
      (hdec (Asw 0 j)) (hdec (Asw 1 j))
    obtain ⟨t', b'⟩ := rebase_top (hE j)
    exact ⟨by rw [t', t]; rfl, by rw [b', b]; rfl⟩
  -- The switched keys.
  have hM : ∀ k j, M k j = ringAutomorphism index.toNat (A k 0) * rdec (inner := m) digits (sw 0 j) +
      ringAutomorphism index.toNat (A k 1) * rdec (inner := m) digits (sw 1 j) := by
    intro k j
    obtain ⟨-, s1, s2⟩ := stacked_row hl01 hl hr01 hr hboth (D j)
    obtain ⟨dt, db⟩ := hDtop j
    fin_cases k
    · show M 0 j = ringAutomorphism index.toNat (A 0 0) * _ + ringAutomorphism index.toNat (A 0 1) * _
      rw [slice_rowOf (by norm_num) (hM0 j)]
      show rowOf (both * D j) 1 = _
      rw [s1, pairMul, dt, db, hσ hr1, hσ hr4]
      rfl
    · show M 1 j = ringAutomorphism index.toNat (A 1 0) * _ + ringAutomorphism index.toNat (A 1 1) * _
      rw [slice_rowOf (by norm_num) (hM1 j)]
      show rowOf (both * D j) 2 = _
      rw [s2, pairMul, dt, db, hσ hr2, hσ hr5]
      rfl
  have hA : ∀ k j, Asw k j = blockModSwitch basis 1 (M k j) := fun k j ↦
    blockModSwitchRuns_unique (hAsw k j) (blockModSwitchRuns_fn (hAsw k j) _)
  funext k j
  obtain ⟨-, s1, s2⟩ := stacked_row hhl01 hhl hhr01 hhr hH (E j)
  obtain ⟨et, eb⟩ := hEtop j
  simp only [branchKey]
  fin_cases k
  · show K 0 j = Ad 0 0 * _ + Ad 0 1 * _
    rw [slice_rowOf (by norm_num) (hK0 j)]
    show rowOf (H * E j) 1 = _
    rw [s1, pairMul, et, eb, hA, hA, hM, hM]
    rfl
  · show K 1 j = Ad 1 0 * _ + Ad 1 1 * _
    rw [slice_rowOf (by norm_num) (hK1 j)]
    show rowOf (H * E j) 2 = _
    rw [s2, pairMul, et, eb, hA, hA, hM, hM]
    rfl

set_option maxHeartbeats 4000000 in
/-- The keys of a LUT branch as the preprocessing computes them (two-row stacks, one-row
decompositions). -/
theorem branchKey_of_pre {q q' m m' : Nat} (hq' : 1 < q') (layout : RegularLayout q')
    (hlayout : Backend.backend.regularLayout q' N = some layout) (hexact : layout.droppedModuli = 0)
    (hm : m = 1 * layout.digitCount) {digits : Int} {basis : List Nat} {index : Int}
    {A Ad : Fin 2 → Fin 2 → ExactMatrix q N 1 m}
    {a00 a10 a01 a11 : ExactMatrix q N 1 m}
    (hσ00 : ringAutomorphismRuns index (A 0 0) a00) (hσ10 : ringAutomorphismRuns index (A 1 0) a10)
    (hσ01 : ringAutomorphismRuns index (A 0 1) a01) (hσ11 : ringAutomorphismRuns index (A 1 1) a11)
    {S0 S1 T0 T1 : ExactMatrix q N (1 + 1) m}
    (hS0 : concatRows a00 a10 S0) (hS1 : concatRows a01 a11 S1)
    (hT0 : concatRows (Ad 0 0) (Ad 1 0) T0) (hT1 : concatRows (Ad 0 1) (Ad 1 1) T1)
    {sw : Fin 2 → Fin 2 → ExactMatrix q' N 1 m'}
    {dA dB dC dE : Fin 2 → ExactMatrix q' N m m'} {rA rB rC rE : Fin 2 → ExactMatrix q N m m'}
    (hdA : ∀ j, gadgetDecomposeRuns Backend.backend 32768 digits (sw 0 j) (dA j))
    (hrA : ∀ j, compactCenteredRebaseRuns 16384 (dA j) (rA j))
    (hdB : ∀ j, gadgetDecomposeRuns Backend.backend 32768 digits (sw 1 j) (dB j))
    (hrB : ∀ j, compactCenteredRebaseRuns 16384 (dB j) (rB j))
    {M : Fin 2 → Fin 2 → ExactMatrix q N 1 m'}
    (hM0 : ∀ j, sliceMatrix (matrixAdd (matrixMul S0 (rA j)) (matrixMul S1 (rB j))) 0 1 0 m' (M 0 j))
    (hM1 : ∀ j, sliceMatrix (matrixAdd (matrixMul S0 (rA j)) (matrixMul S1 (rB j))) 1 2 0 m' (M 1 j))
    {Asw : Fin 2 → Fin 2 → ExactMatrix q' N 1 m'}
    (hAsw : ∀ k j, blockModSwitchRuns basis 1 (M k j) (Asw k j))
    (hdC : ∀ j, gadgetDecomposeRuns Backend.backend 32768 digits (Asw 0 j) (dC j))
    (hrC : ∀ j, compactCenteredRebaseRuns 16384 (dC j) (rC j))
    (hdE : ∀ j, gadgetDecomposeRuns Backend.backend 32768 digits (Asw 1 j) (dE j))
    (hrE : ∀ j, compactCenteredRebaseRuns 16384 (dE j) (rE j))
    {K : Fin 2 → Fin 2 → ExactMatrix q N 1 m'}
    (hK0 : ∀ j, sliceMatrix (matrixAdd (matrixMul T0 (rC j)) (matrixMul T1 (rE j))) 0 1 0 m' (K 0 j))
    (hK1 : ∀ j, sliceMatrix (matrixAdd (matrixMul T0 (rC j)) (matrixMul T1 (rE j))) 1 2 0 m' (K 1 j)) :
    K = branchKey digits basis index.toNat A Ad sw := by
  have hr : ∀ {d : ExactMatrix q' N m m'} {r : ExactMatrix q N m m'} {X : ExactMatrix q' N 1 m'},
      gadgetDecomposeRuns Backend.backend 32768 digits X d → compactCenteredRebaseRuns 16384 d r →
      r = rdec (inner := m) digits X := by
    intro d r X hd hrr
    rw [hrr.2.2.1.2, dec_eq hd]
    rfl
  have hσ : ∀ {x y : ExactMatrix q N 1 m}, ringAutomorphismRuns index x y →
      y = ringAutomorphism index.toNat x := fun hxy ↦ hxy.2.2.2
  have hA : ∀ k j, Asw k j = blockModSwitch basis 1 (M k j) := fun k j ↦
    blockModSwitchRuns_unique (hAsw k j) (blockModSwitchRuns_fn (hAsw k j) _)
  have hM : ∀ k j, M k j = ringAutomorphism index.toNat (A k 0) * rdec (inner := m) digits (sw 0 j) +
      ringAutomorphism index.toNat (A k 1) * rdec (inner := m) digits (sw 1 j) := by
    intro k j
    obtain ⟨s0a, s0b⟩ := stack2_mul_rows hS0 (rA j)
    obtain ⟨s1a, s1b⟩ := stack2_mul_rows hS1 (rB j)
    fin_cases k
    · show M 0 j = ringAutomorphism index.toNat (A 0 0) * _ + ringAutomorphism index.toNat (A 0 1) * _
      rw [slice_rowOf (by norm_num) (hM0 j)]
      show rowOf (S0 * rA j + S1 * rB j) 0 = _
      rw [rowOf_add, s0a, s1a, hr (hdA j) (hrA j), hr (hdB j) (hrB j), hσ hσ00, hσ hσ01]
    · show M 1 j = ringAutomorphism index.toNat (A 1 0) * _ + ringAutomorphism index.toNat (A 1 1) * _
      rw [slice_rowOf (by norm_num) (hM1 j)]
      show rowOf (S0 * rA j + S1 * rB j) 1 = _
      rw [rowOf_add, s0b, s1b, hr (hdA j) (hrA j), hr (hdB j) (hrB j), hσ hσ10, hσ hσ11]
  funext k j
  obtain ⟨t0a, t0b⟩ := stack2_mul_rows hT0 (rC j)
  obtain ⟨t1a, t1b⟩ := stack2_mul_rows hT1 (rE j)
  simp only [branchKey]
  fin_cases k
  · show K 0 j = Ad 0 0 * _ + Ad 0 1 * _
    rw [slice_rowOf (by norm_num) (hK0 j)]
    show rowOf (T0 * rC j + T1 * rE j) 0 = _
    rw [rowOf_add, t0a, t1a, hr (hdC j) (hrC j), hr (hdE j) (hrE j), hA, hA, hM, hM]
  · show K 1 j = Ad 1 0 * _ + Ad 1 1 * _
    rw [slice_rowOf (by norm_num) (hK1 j)]
    show rowOf (T0 * rC j + T1 * rE j) 1 = _
    rw [rowOf_add, t0b, t1b, hr (hdC j) (hrC j), hr (hdE j) (hrE j), hA, hA, hM, hM]

/-- The four keys of a pair-indexed key matrix as a tuple. -/
def keyTuple {q m : Nat} (K : Fin 2 → Fin 2 → ExactMatrix q N 1 m) : Keys q m :=
  (K 0 0, K 0 1, K 1 0, K 1 1, ())

theorem keyTuple_add {q m : Nat} (K L : Fin 2 → Fin 2 → ExactMatrix q N 1 m) :
    keyTuple (K + L) = keyTuple K + keyTuple L := rfl

/-- The keys of branch `i` of a width-`W` LUT with automorphisms `σf` (zero past the branches). -/
noncomputable def lutKeys {q q' m m' W : Nat} (digits : Int) (basis : List Nat) (model : HashModel)
    (key : ByteArray) (level : Int) (σf : Nat → Int) (A : Fin 2 → Fin 2 → ExactMatrix q N 1 m)
    (fam : Fin 2 → Fin 2 → Fin W → ExactMatrix q' N 1 m') (i : Nat) : Keys q m' :=
  if h : i < W then
    keyTuple (branchKey digits basis (σf i).toNat A (lutA model key q m level (σf i))
      (fun k j ↦ fam k j ⟨i, h⟩))
  else 0

/-- A map applied to each of the four keys. -/
def keyMap {q m q' m' : Nat} (f : ExactMatrix q N 1 m → ExactMatrix q' N 1 m') (x : Keys q m) :
    Keys q' m' :=
  (f x.1, f x.2.1, f x.2.2.1, f x.2.2.2.1, ())

/-- The keys after a private multiplication by an RHS with top `top` and bottom `bot`, moved to
the next modulus. -/
noncomputable def privKey {q q' m m' : Nat} (digits : Int) (basis : List Nat)
    (A : Fin 2 → Fin 2 → ExactMatrix q N 1 m) (top bot : Fin 2 → ExactMatrix q' N 1 m') :
    Fin 2 → Fin 2 → ExactMatrix q' N 1 m' :=
  fun k j ↦ blockModSwitch basis 1
    (A k 0 * rdec (inner := m) digits (top j) + A k 1 * rdec (inner := m) digits (bot j))

set_option maxHeartbeats 4000000 in
/-- The keys of a private multiplication as the online evaluation computes them. -/
theorem privKey_of_stacks {q q' m m' : Nat} (hq' : 1 < q') (layout : RegularLayout q')
    (hlayout : Backend.backend.regularLayout q' N = some layout) (hexact : layout.droppedModuli = 0)
    (hm : m = 1 * layout.digitCount) {digits : Int} {basis : List Nat}
    {c : Fin 2 → ExactMatrix q N 1 m} {A : Fin 2 → Fin 2 → ExactMatrix q N 1 m}
    {l01 r01 : ExactMatrix q N (1 + 1) m} {l r : ExactMatrix q N (1 + 1 + 1) m}
    {both : ExactMatrix q N (1 + 1 + 1) (m + m)}
    (hl01 : concatRows (c 0) (A 0 0) l01) (hl : concatRows l01 (A 1 0) l)
    (hr01 : concatRows (c 1) (A 0 1) r01) (hr : concatRows r01 (A 1 1) r)
    (hboth : concatColumns l r both)
    {top bot : Fin 2 → ExactMatrix q' N 1 m'} {Xb : Fin 2 → ExactMatrix q' N (1 + 1) m'}
    (hXb : ∀ j, concatRows (top j) (bot j) (Xb j))
    {D' : Fin 2 → ExactMatrix q' N (m + m) m'}
    (hD' : ∀ j, gadgetDecomposeRuns Backend.backend 32768 digits (Xb j) (D' j))
    {D : Fin 2 → ExactMatrix q N (m + m) m'} (hD : ∀ j, compactCenteredRebaseRuns 16384 (D' j) (D j))
    {M : Fin 2 → Fin 2 → ExactMatrix q N 1 m'}
    (hM0 : ∀ j, sliceMatrix (matrixMul both (D j)) 1 2 0 m' (M 0 j))
    (hM1 : ∀ j, sliceMatrix (matrixMul both (D j)) 2 3 0 m' (M 1 j))
    {A' : Fin 2 → Fin 2 → ExactMatrix q' N 1 m'}
    (hA' : ∀ k j, blockModSwitchRuns basis 1 (M k j) (A' k j)) :
    A' = privKey digits basis A top bot := by
  have hdec := fun (X : ExactMatrix q' N 1 m') ↦
    dec_of_runs (inner := m) layout hlayout hexact hq' (hD' 0) hm X
  funext k j
  obtain ⟨t, b⟩ := decompose_stack layout hlayout hexact (hXb j) (hD' j) (hdec (top j)) (hdec (bot j))
  obtain ⟨t', b'⟩ := rebase_top (hD j)
  obtain ⟨-, s1, s2⟩ := stacked_row hl01 hl hr01 hr hboth (D j)
  rw [blockModSwitchRuns_unique (hA' k j) (blockModSwitchRuns_fn (hA' k j) _)]
  simp only [privKey]
  congr 1
  fin_cases k
  · show M 0 j = A 0 0 * _ + A 0 1 * _
    rw [slice_rowOf (by norm_num) (hM0 j)]
    show rowOf (both * D j) 1 = _
    rw [s1, pairMul, t', b', t, b]
    rfl
  · show M 1 j = A 1 0 * _ + A 1 1 * _
    rw [slice_rowOf (by norm_num) (hM1 j)]
    show rowOf (both * D j) 2 = _
    rw [s2, pairMul, t', b', t, b]
    rfl

set_option maxHeartbeats 4000000 in
/-- The keys of a private multiplication as the preprocessing computes them. -/
theorem privKey_of_pre {q q' m m' : Nat} {digits : Int} {basis : List Nat}
    {A : Fin 2 → Fin 2 → ExactMatrix q N 1 m} {S0 S1 : ExactMatrix q N (1 + 1) m}
    (hS0 : concatRows (A 0 0) (A 1 0) S0) (hS1 : concatRows (A 0 1) (A 1 1) S1)
    {top bot : Fin 2 → ExactMatrix q' N 1 m'}
    {dT dB : Fin 2 → ExactMatrix q' N m m'} {rT rB : Fin 2 → ExactMatrix q N m m'}
    (hdT : ∀ j, gadgetDecomposeRuns Backend.backend 32768 digits (top j) (dT j))
    (hrT : ∀ j, compactCenteredRebaseRuns 16384 (dT j) (rT j))
    (hdB : ∀ j, gadgetDecomposeRuns Backend.backend 32768 digits (bot j) (dB j))
    (hrB : ∀ j, compactCenteredRebaseRuns 16384 (dB j) (rB j))
    {M : Fin 2 → Fin 2 → ExactMatrix q N 1 m'}
    (hM0 : ∀ j, sliceMatrix (matrixAdd (matrixMul S0 (rT j)) (matrixMul S1 (rB j))) 0 1 0 m' (M 0 j))
    (hM1 : ∀ j, sliceMatrix (matrixAdd (matrixMul S0 (rT j)) (matrixMul S1 (rB j))) 1 2 0 m' (M 1 j))
    {A' : Fin 2 → Fin 2 → ExactMatrix q' N 1 m'}
    (hA' : ∀ k j, blockModSwitchRuns basis 1 (M k j) (A' k j)) :
    A' = privKey digits basis A top bot := by
  have hr : ∀ {d : ExactMatrix q' N m m'} {r : ExactMatrix q N m m'} {X : ExactMatrix q' N 1 m'},
      gadgetDecomposeRuns Backend.backend 32768 digits X d → compactCenteredRebaseRuns 16384 d r →
      r = rdec (inner := m) digits X := by
    intro d r X hd hrr
    rw [hrr.2.2.1.2, dec_eq hd]
    rfl
  funext k j
  obtain ⟨s0a, s0b⟩ := stack2_mul_rows hS0 (rT j)
  obtain ⟨s1a, s1b⟩ := stack2_mul_rows hS1 (rB j)
  rw [blockModSwitchRuns_unique (hA' k j) (blockModSwitchRuns_fn (hA' k j) _)]
  simp only [privKey]
  congr 1
  fin_cases k
  · show M 0 j = A 0 0 * _ + A 0 1 * _
    rw [slice_rowOf (by norm_num) (hM0 j)]
    show rowOf (S0 * rT j + S1 * rB j) 0 = _
    rw [rowOf_add, s0a, s1a, hr (hdT j) (hrT j), hr (hdB j) (hrB j)]
  · show M 1 j = A 1 0 * _ + A 1 1 * _
    rw [slice_rowOf (by norm_num) (hM1 j)]
    show rowOf (S0 * rT j + S1 * rB j) 1 = _
    rw [rowOf_add, s0b, s1b, hr (hdT j) (hrT j), hr (hdB j) (hrB j)]

end RefreshTopProof
