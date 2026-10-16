import Dio2CellBridge
import Dio2LutBridge
import GenKey

/-!
# The keys of dio2 encodings

Key agreement between the online stage and the keys stage (which runs the same graph on zero rows
and its own label payloads): the keys of every computation bundle are a function of its input keys
alone (refresh_top's `GenKey` forms, without the modulus switch of a dio2 chain), so both stages
compute the same keys from the same input keys.
-/

namespace Dio2Proof

open Mxx.Primitives MxxRuntime MxxExponentLut RefreshTopProof.Gen

variable {n : Nat}

/-- The keys after a private multiplication by an RHS with tops `top` and bottoms `bot`, at one
modulus. -/
noncomputable def privKeyF {q m m' : Nat} (backend : BackendContext) (base digits : Int)
    (A : Fin 2 → Fin 2 → ExactMatrix q n 1 m) (top bot : Fin 2 → ExactMatrix q n 1 m') :
    Fin 2 → Fin 2 → ExactMatrix q n 1 m' :=
  fun k j ↦ A k 0 * rdecG (inner := m) backend base digits (top j) +
    A k 1 * rdecG (inner := m) backend base digits (bot j)

/-- The keys of a dio2 PRF cell. -/
noncomputable def cellKeyF {q m m' m'' : Nat} (backend : BackendContext)
    (base digits base' digits' : Int)
    (A : Fin 2 → Fin 2 → ExactMatrix q n 1 m) (As : Fin 2 → Fin 2 → ExactMatrix q n 1 m')
    (top bot : Fin 2 → ExactMatrix q n 1 m'') : Fin 2 → Fin 2 → ExactMatrix q n 1 m'' :=
  privKeyF backend base' digits' (pubKeyG backend base digits A As) top bot

/-- The width of a decomposed two-row stack is that of the layout's digits. -/
theorem width_of_stack {q m cols : Nat} {backend : BackendContext} {base digits : Int}
    {layout : RegularLayout q} (hlayout : backend.regularLayout q n = some layout)
    {X : ExactMatrix q n (1 + 1) cols} {D : ExactMatrix q n (m + m) cols}
    (h : gadgetDecomposeRuns backend base digits X D) : m = 1 * layout.digitCount := by
  obtain ⟨layout', hl, _, _, hw, _⟩ := h
  rw [hlayout] at hl
  cases hl
  omega

set_option maxHeartbeats 4000000 in
/-- The keys of a private multiplication at one modulus as the online evaluation computes them. -/
theorem privKeyF_of_stacks (hn : 0 < n) {q m m' bound : Nat} {backend : BackendContext}
    {base digits : Int} (hq : 1 < q) (layout : RegularLayout q)
    (hlayout : backend.regularLayout q n = some layout) (hexact : layout.droppedModuli = 0)
    {c : Fin 2 → ExactMatrix q n 1 m} {A : Fin 2 → Fin 2 → ExactMatrix q n 1 m}
    {l01 r01 : ExactMatrix q n (1 + 1) m} {l r : ExactMatrix q n (1 + 1 + 1) m}
    {both : ExactMatrix q n (1 + 1 + 1) (m + m)}
    (hl01 : concatRows (c 0) (A 0 0) l01) (hl : concatRows l01 (A 1 0) l)
    (hr01 : concatRows (c 1) (A 0 1) r01) (hr : concatRows r01 (A 1 1) r)
    (hboth : concatColumns l r both)
    {top bot : Fin 2 → ExactMatrix q n 1 m'} {Xb : Fin 2 → ExactMatrix q n (1 + 1) m'}
    (hXb : ∀ j, concatRows (top j) (bot j) (Xb j))
    {D' : Fin 2 → ExactMatrix q n (m + m) m'}
    (hD' : ∀ j, gadgetDecomposeRuns backend base digits (Xb j) (D' j))
    {D : Fin 2 → ExactMatrix q n (m + m) m'} (hD : ∀ j, compactCenteredRebaseRuns bound (D' j) (D j))
    {M : Fin 2 → Fin 2 → ExactMatrix q n 1 m'}
    (hM0 : ∀ j, sliceMatrix (matrixMul both (D j)) 1 2 0 m' (M 0 j))
    (hM1 : ∀ j, sliceMatrix (matrixMul both (D j)) 2 3 0 m' (M 1 j)) :
    M = privKeyF backend base digits A top bot := by
  have hm := width_of_stack hlayout (hD' 0)
  funext k j
  obtain ⟨t, b⟩ := stack_rdecG hn hq layout hlayout hexact hm (hXb j) (hD' j) (hD j)
  obtain ⟨-, s1, s2⟩ := stacked_row hl01 hl hr01 hr hboth (D j)
  simp only [privKeyF]
  fin_cases k
  · show M 0 j = A 0 0 * _ + A 0 1 * _
    rw [slice_rowOfG (by norm_num) (hM0 j)]
    show rowOf (both * D j) 1 = _
    rw [s1, pairMul, t, b]
    rfl
  · show M 1 j = A 1 0 * _ + A 1 1 * _
    rw [slice_rowOfG (by norm_num) (hM1 j)]
    show rowOf (both * D j) 2 = _
    rw [s2, pairMul, t, b]
    rfl

set_option maxHeartbeats 8000000 in
/-- **The keys of a dio2 PRF cell** (variant 0): a function of the label member's root keys, the
state keys and the selector RHS alone — whatever the rows and the payload. -/
theorem PrfCellFlat.keys (hn : 0 < n) {q m m' m'' bound : Nat} (hq1 : 1 < q)
    {backend : BackendContext} {hashModel : HashModel} {base digits base' digits' : Int}
    (layout : RegularLayout q) (hlayout : backend.regularLayout q n = some layout)
    (hexact : layout.droppedModuli = 0)
    {lo two width lo' two' width' lo'' two'' width'' lo''' two''' width''' : Nat}
    {pre rhsPre : Blob} {a b : Int} {i : Nat}
    {idx : Int} {x0 x1 : ExactMatrix q n 1 m} {seed seed' : ByteArray}
    {As00 As10 As01 As11 cs0 cs1 : ExactMatrix q n 1 m'} {pay : ExactMatrix q n 1 1}
    {bot0 bot1 : ExactMatrix q n 1 m''}
    {outputs : ExactMatrix q n 1 m'' × ExactMatrix q n 1 m'' × ExactMatrix q n 1 m'' ×
      ExactMatrix q n 1 m'' × ExactMatrix q n 1 m'' × ExactMatrix q n 1 m'' × Unit}
    (h : Dio2Proof.Comp.PrfCellFlat.Comp q n m' m' m' m' m'' m'' m m q q q q m'' m'' m'' m'' m''
      m'' lo two width pre lo' two' width' pre lo'' two'' width'' pre lo''' two''' width''' pre
      base digits bound m' m' m' base digits bound m' m' m' a b rhsPre base' digits' bound m''
      a b rhsPre base' digits' bound m'' m'' m'' m'' m'' backend hashModel i
      (idx, x0, seed, x1, As00, As10, cs0, pay, As01, As11, cs1, seed', bot0, bot1, ()) outputs) :
    ![![outputs.1, outputs.2.1], ![outputs.2.2.1, outputs.2.2.2.1]] =
      cellKeyF backend base digits base' digits' (labelKeysG hashModel seed q m pre idx)
        ![![As00, As01], ![As10, As11]] (topPairG hashModel seed' q m'' rhsPre (prfTopTags a b i))
        ![bot0, bot1] := by
  obtain ⟨v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18,
    v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, v36, v37,
    v38, v39, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19,
    c20, c21, c22, c23, c24, c25, c26, c27, c28, c29, c30, c31, c32, c33, c34, c35, c36, c37, c38,
    c39, c40, c41, c42, c43, c44, c45, c46, c47, c48, h⟩ := h
  subst h
  have hkeys : ![![v0, v4], ![v1, v5]] = labelKeysG hashModel seed q m pre idx := by
    funext k j
    fin_cases k <;> fin_cases j
    · exact keyPairG_of_hash (pre := [.bytes memberTagG, .u64Le idx]) 0 0 (by norm_num) c3
    · exact keyPairG_of_hash (pre := [.bytes memberTagG, .u64Le idx]) 0 1 (by norm_num) c11
    · exact keyPairG_of_hash (pre := [.bytes memberTagG, .u64Le idx]) 1 0 (by norm_num) c6
    · exact keyPairG_of_hash (pre := [.bytes memberTagG, .u64Le idx]) 1 1 (by norm_num) c14
  have htop := prf_tops c35 c40
  have hm := width_of_stack hlayout c19
  have hpub := pubKeyG_of_stacks hn hq1 layout hlayout hexact hm
    (A := ![![v0, v4], ![v1, v5]]) c7 c8 c15 c16 c17
    (As := ![![As00, As01], ![As10, As11]])
    (Xs := ![v9, v17]) (fun j ↦ by fin_cases j; exacts [c18, c26])
    (D' := ![v10, v18]) (fun j ↦ by fin_cases j; exacts [c19, c27])
    (D := ![v11, v19]) (fun j ↦ by fin_cases j; exacts [c20, c28])
    (M := ![![v13, v21], ![v14, v22]]) (fun j ↦ by fin_cases j; exacts [c22, c30])
    (fun j ↦ by fin_cases j; exacts [c23, c31])
  have hpriv := privKeyF_of_stacks hn hq1 layout hlayout hexact
    (c := ![matrixAdd v12 (matrixMulScalarRight cs0 pay),
      matrixAdd v20 (matrixMulScalarRight cs1 pay)])
    (A := ![![v13, v21], ![v14, v22]]) c24 c25 c32 c33 c34
    (top := ![v26, v31]) (bot := ![bot0, bot1])
    (Xb := ![v27, v32]) (fun j ↦ by fin_cases j; exacts [c36, c41])
    (D' := ![v28, v33]) (fun j ↦ by fin_cases j; exacts [c37, c42])
    (D := ![v29, v34]) (fun j ↦ by fin_cases j; exacts [c38, c43])
    (M := ![![v30, v35], ![v36, v37]]) (fun j ↦ by fin_cases j; exacts [c39, c44])
    (fun j ↦ by fin_cases j; exacts [c45, c46])
  rw [cellKeyF, ← hkeys, ← htop, ← hpub, ← hpriv]

/-- The keys of a dio2 LUT branch with automorphism `σ`: the input keys `A` routed by `σ` and
switched by the branch's ciphertexts `sw`, then multiplied by the helper keys `Ad`. -/
noncomputable def branchKeyF {q m m' : Nat} (backend : BackendContext) (base digits : Int)
    (σ : Nat) (A Ad : Fin 2 → Fin 2 → ExactMatrix q n 1 m)
    (sw : Fin 2 → Fin 2 → ExactMatrix q n 1 m') : Fin 2 → Fin 2 → ExactMatrix q n 1 m' :=
  fun k j ↦
    let Asw : Fin 2 → ExactMatrix q n 1 m' := fun k' ↦
      ringAutomorphism σ (A k' 0) * rdecG (inner := m) backend base digits (sw 0 j) +
        ringAutomorphism σ (A k' 1) * rdecG (inner := m) backend base digits (sw 1 j)
    Ad k 0 * rdecG (inner := m) backend base digits (Asw 0) +
      Ad k 1 * rdecG (inner := m) backend base digits (Asw 1)

set_option maxHeartbeats 4000000 in
/-- The keys of a dio2 LUT branch as the online evaluation computes them (`lut_branch_flat`'s
atoms): a function of the input keys, the switching ciphertexts and the helper keys. -/
theorem branchKeyF_of_stacks (hn : 0 < n) {q m m' bound : Nat} {backend : BackendContext}
    {base digits : Int} (hq : 1 < q) (layout : RegularLayout q)
    (hlayout : backend.regularLayout q n = some layout) (hexact : layout.droppedModuli = 0)
    {c : Fin 2 → ExactMatrix q n 1 m} {A : Fin 2 → Fin 2 → ExactMatrix q n 1 m}
    {index : Int} {rc0 rA00 rA10 rc1 rA01 rA11 : ExactMatrix q n 1 m}
    (hr0 : ringAutomorphismRuns index (c 0) rc0) (hr1 : ringAutomorphismRuns index (A 0 0) rA00)
    (hr2 : ringAutomorphismRuns index (A 1 0) rA10) (hr3 : ringAutomorphismRuns index (c 1) rc1)
    (hr4 : ringAutomorphismRuns index (A 0 1) rA01) (hr5 : ringAutomorphismRuns index (A 1 1) rA11)
    {l01 r01 : ExactMatrix q n (1 + 1) m} {l r : ExactMatrix q n (1 + 1 + 1) m}
    {both : ExactMatrix q n (1 + 1 + 1) (m + m)}
    (hl01 : concatRows rc0 rA00 l01) (hl : concatRows l01 rA10 l)
    (hr01 : concatRows rc1 rA01 r01) (hr : concatRows r01 rA11 r) (hboth : concatColumns l r both)
    {sw : Fin 2 → Fin 2 → ExactMatrix q n 1 m'} {Xb : Fin 2 → ExactMatrix q n (1 + 1) m'}
    (hXb : ∀ j, concatRows (sw 0 j) (sw 1 j) (Xb j))
    {D' : Fin 2 → ExactMatrix q n (m + m) m'}
    (hD' : ∀ j, gadgetDecomposeRuns backend base digits (Xb j) (D' j))
    {D : Fin 2 → ExactMatrix q n (m + m) m'} (hD : ∀ j, compactCenteredRebaseRuns bound (D' j) (D j))
    {M : Fin 2 → Fin 2 → ExactMatrix q n 1 m'}
    (hM0 : ∀ j, sliceMatrix (matrixMul both (D j)) 1 2 0 m' (M 0 j))
    (hM1 : ∀ j, sliceMatrix (matrixMul both (D j)) 2 3 0 m' (M 1 j))
    {Z : Fin 2 → ExactMatrix q n (1 + 1) m'} (hZ : ∀ j, concatRows (M 0 j) (M 1 j) (Z j))
    {E' : Fin 2 → ExactMatrix q n (m + m) m'}
    (hE' : ∀ j, gadgetDecomposeRuns backend base digits (Z j) (E' j))
    {E : Fin 2 → ExactMatrix q n (m + m) m'} (hE : ∀ j, compactCenteredRebaseRuns bound (E' j) (E j))
    {h : Fin 2 → ExactMatrix q n 1 m} {Ad : Fin 2 → Fin 2 → ExactMatrix q n 1 m}
    {hl01' hr01' : ExactMatrix q n (1 + 1) m} {hl' hr' : ExactMatrix q n (1 + 1 + 1) m}
    {H : ExactMatrix q n (1 + 1 + 1) (m + m)}
    (hhl01 : concatRows (h 0) (Ad 0 0) hl01') (hhl : concatRows hl01' (Ad 1 0) hl')
    (hhr01 : concatRows (h 1) (Ad 0 1) hr01') (hhr : concatRows hr01' (Ad 1 1) hr')
    (hH : concatColumns hl' hr' H)
    {K : Fin 2 → Fin 2 → ExactMatrix q n 1 m'}
    (hK0 : ∀ j, sliceMatrix (matrixMul H (E j)) 1 2 0 m' (K 0 j))
    (hK1 : ∀ j, sliceMatrix (matrixMul H (E j)) 2 3 0 m' (K 1 j)) :
    K = branchKeyF backend base digits index.toNat A Ad sw := by
  have hm := width_of_stack hlayout (hD' 0)
  have hσ : ∀ {x y : ExactMatrix q n 1 m}, ringAutomorphismRuns index x y →
      y = ringAutomorphism index.toNat x := fun hxy ↦ hxy.2.2.2
  have hDtop := fun j ↦ stack_rdecG hn hq layout hlayout hexact hm (hXb j) (hD' j) (hD j)
  have hEtop := fun j ↦ stack_rdecG hn hq layout hlayout hexact hm (hZ j) (hE' j) (hE j)
  have hM : ∀ k j, M k j =
      ringAutomorphism index.toNat (A k 0) * rdecG (inner := m) backend base digits (sw 0 j) +
      ringAutomorphism index.toNat (A k 1) * rdecG (inner := m) backend base digits (sw 1 j) := by
    intro k j
    obtain ⟨-, s1, s2⟩ := stacked_row hl01 hl hr01 hr hboth (D j)
    obtain ⟨dt, db⟩ := hDtop j
    fin_cases k
    · show M 0 j = ringAutomorphism index.toNat (A 0 0) * _ + ringAutomorphism index.toNat (A 0 1) * _
      rw [slice_rowOfG (by norm_num) (hM0 j)]
      show rowOf (both * D j) 1 = _
      rw [s1, pairMul, dt, db, hσ hr1, hσ hr4]
      rfl
    · show M 1 j = ringAutomorphism index.toNat (A 1 0) * _ + ringAutomorphism index.toNat (A 1 1) * _
      rw [slice_rowOfG (by norm_num) (hM1 j)]
      show rowOf (both * D j) 2 = _
      rw [s2, pairMul, dt, db, hσ hr2, hσ hr5]
      rfl
  funext k j
  obtain ⟨-, s1, s2⟩ := stacked_row hhl01 hhl hhr01 hhr hH (E j)
  obtain ⟨et, eb⟩ := hEtop j
  simp only [branchKeyF]
  fin_cases k
  · show K 0 j = Ad 0 0 * _ + Ad 0 1 * _
    rw [slice_rowOfG (by norm_num) (hK0 j)]
    show rowOf (H * E j) 1 = _
    rw [s1, pairMul, et, eb, hM, hM]
    rfl
  · show K 1 j = Ad 1 0 * _ + Ad 1 1 * _
    rw [slice_rowOfG (by norm_num) (hK1 j)]
    show rowOf (H * E j) 2 = _
    rw [s2, pairMul, et, eb, hM, hM]
    rfl

set_option maxHeartbeats 8000000 in
/-- **The keys of one LUT loop step** (variant 0): the current keys gain the branch keys
`branchKeyF` of branch `i` (input keys `A`, helper keys tagged `σ_i`, carried switching
ciphertexts), whatever the rows. -/
theorem LutRoundStep.keys (hn : 0 < n) {q m m' bound W : Nat} (hq1 : 1 < q)
    {backend : BackendContext} {hashModel : HashModel} {base digits : Int}
    (layout : RegularLayout q) (hlayout : backend.regularLayout q n = some layout)
    (hexact : layout.droppedModuli = 0)
    {lo0 hi0 lo1 hi1 l2 h2 l3 h3 l4 h4 l5 h5 : Int} {pre : Blob} {sigmas : List Int}
    {seed : ByteArray} {i : Nat}
    {r0 r1 k00 k01 k10 k11 b00 b01 b10 b11 : ExactMatrix q n 1 m'}
    {hf0 hf1 : Fin W → ExactMatrix q n 1 m} {x0 a00 a10 x1 a01 a11 : ExactMatrix q n 1 m}
    {sw00 sw01 sw10 sw11 : Fin W → ExactMatrix q n 1 m'}
    {outputs : ExactMatrix q n 1 m' × ExactMatrix q n 1 m' × ExactMatrix q n 1 m' ×
      ExactMatrix q n 1 m' × ExactMatrix q n 1 m' × ExactMatrix q n 1 m' × ExactMatrix q n 1 m' ×
      ExactMatrix q n 1 m' × ExactMatrix q n 1 m' × ExactMatrix q n 1 m' × Unit}
    (h : Dio2Proof.Comp.LutRoundStep.Comp q n m' m' m' m' m' m' m' m' m' m' q q W m W m m m W
      q n m' W q n m' W q n m' W q n m' lo0 hi0 pre sigmas pre sigmas lo1 hi1 pre sigmas pre
      sigmas sigmas sigmas sigmas sigmas sigmas sigmas base digits bound m' m' base digits bound
      m' m' base digits bound m' m' base digits bound m' m' m' m' m' m' l2 h2 l3 h3 l4 h4 l5 h5
      backend hashModel i
      (r0, r1, k00, k01, k10, k11, b00, b01, b10, b11, hf0, seed, hf1, x0, a00, a10, x1, a01, a11,
        sw00, sw01, sw10, sw11, ()) outputs) :
    ![![outputs.2.2.1, outputs.2.2.2.1], ![outputs.2.2.2.2.1, outputs.2.2.2.2.2.1]] =
        ![![k00, k01], ![k10, k11]] + branchKeyF backend base digits
          (MxxIR.selectInt (Int.ofNat i) sigmas).toNat ![![a00, a01], ![a10, a11]]
          (keyPairG hashModel seed q m pre [.integer (MxxIR.selectInt (Int.ofNat i) sigmas)])
          ![![b00, b01], ![b10, b11]] ∧
      ∀ hi1 : i + 1 < W, outputs.2.2.2.2.2.2.1 = sw00 ⟨i + 1, hi1⟩ ∧
        outputs.2.2.2.2.2.2.2.1 = sw01 ⟨i + 1, hi1⟩ ∧
        outputs.2.2.2.2.2.2.2.2.1 = sw10 ⟨i + 1, hi1⟩ ∧
        outputs.2.2.2.2.2.2.2.2.2.1 = sw11 ⟨i + 1, hi1⟩ := by
  obtain ⟨v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18,
    v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, v36, v37,
    v38, v39, v40, v41, v42, v43, v44, v45, v46, v47, v48, v49,
    c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c20,
    c21, c22, c23, c24, c25, c26, c27, c28, c29, c30, c31, c32, c33, c34, c35, c36, c37, c38, c39,
    c40, c41, c42, c43, c44, c45, c46, c47, c48, c49, c50, c51, c52, c53, c54, c55, c56, c57, c58,
    c59, c60, c61, c62, h⟩ := h
  subst h
  have hkeys : ![![v1, v6], ![v2, v7]] = keyPairG hashModel seed q m pre
      [.integer (MxxIR.selectInt (Int.ofNat i) sigmas)] := by
    funext k j
    fin_cases k <;> fin_cases j
    · exact keyPairG_of_hash (pre := [.integer (MxxIR.selectInt (Int.ofNat i) sigmas)]) 0 0
        (by norm_num) c4
    · exact keyPairG_of_hash (pre := [.integer (MxxIR.selectInt (Int.ofNat i) sigmas)]) 0 1
        (by norm_num) c11
    · exact keyPairG_of_hash (pre := [.integer (MxxIR.selectInt (Int.ofNat i) sigmas)]) 1 0
        (by norm_num) c5
    · exact keyPairG_of_hash (pre := [.integer (MxxIR.selectInt (Int.ofNat i) sigmas)]) 1 1
        (by norm_num) c12
  have hb := branchKeyF_of_stacks hn hq1 layout hlayout hexact
    (c := ![x0, x1]) (A := ![![a00, a01], ![a10, a11]])
    (rc0 := v11) (rA00 := v12) (rA10 := v13) (rc1 := v16) (rA01 := v17) (rA11 := v18)
    c16 c17 c18 c21 c22 c23 c19 c20 c24 c25 c26
    (sw := ![![b00, b01], ![b10, b11]]) (Xb := ![v22, v32])
    (fun j ↦ by fin_cases j; exacts [c27, c37])
    (D' := ![v23, v33]) (fun j ↦ by fin_cases j; exacts [c28, c38])
    (D := ![v24, v34]) (fun j ↦ by fin_cases j; exacts [c29, c39])
    (M := ![![v25, v35], ![v26, v36]]) (fun j ↦ by fin_cases j; exacts [c30, c40])
    (fun j ↦ by fin_cases j; exacts [c31, c41])
    (Z := ![v27, v37]) (fun j ↦ by fin_cases j; exacts [c32, c42])
    (E' := ![v28, v38]) (fun j ↦ by fin_cases j; exacts [c33, c43])
    (E := ![v29, v39]) (fun j ↦ by fin_cases j; exacts [c34, c44])
    (h := ![v0, v5]) (Ad := ![![v1, v6], ![v2, v7]]) c6 c7 c13 c14 c15
    (K := ![![v42, v43], ![v44, v45]])
    (fun j ↦ by fin_cases j; exacts [c47, c48])
    (fun j ↦ by fin_cases j; exacts [c49, c50])
  refine ⟨?_, fun hi1 ↦ ⟨familyGetDynamic_succ c53 hi1, familyGetDynamic_succ c56 hi1,
    familyGetDynamic_succ c59 hi1, familyGetDynamic_succ c62 hi1⟩⟩
  rw [← hkeys, ← hb]
  funext k j
  fin_cases k <;> fin_cases j <;> rfl

end Dio2Proof
