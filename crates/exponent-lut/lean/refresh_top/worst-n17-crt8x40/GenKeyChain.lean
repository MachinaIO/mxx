import GenInstKeys

/-!
# Key chains of the online evaluation (WP D)

The keys (the `A` parts) of the online encodings carry no semantic hypothesis: they are functions
of the public keys only, the same functions the preprocessing evaluates. The lemmas here state the
keys of the online root pieces in the preprocessing's closed forms (`lutKeysG`, `keyBmsG`), so the
part theorem can identify the final combination's keys with the preprocessing wires.
-/

namespace RefreshTopProof.GenIface

open Mxx.Primitives MxxRuntime MxxExponentLut RefreshTopProof.Gen BrLut BrLutKey

variable {n : Nat}

/-- A function applied to both entries of a pair, read at `j`. -/
theorem fin2_map {α β : Type} (f : α → β) (a b : α) (j : Fin 2) : ![f a, f b] j = f (![a, b] j) := by
  fin_cases j <;> rfl

set_option maxHeartbeats 16000000 in
/-- **The keys of a LUT's branches** (the key part of `lut_branches_gen`): `L` looped branches (key
step `hchild`) plus the unrolled last branch are the preprocessing's summed branch keys `lutKeysG`
of all `L + 1` branches. -/
theorem branch_keys_core {q q' m m' W : Nat} {backend : BackendContext} {hashModel : HashModel}
    {base digits : Int} {tab : List Int} {key : ByteArray} {level : Int} {tag : Blob}
    {hf0 hf1 : Fin W → ExactMatrix q n 1 m} {x0 x1 x2 x3 x4 x5 : ExactMatrix q n 1 m}
    {f00 f01 f10 f11 : Fin W → ExactMatrix q' n 1 m'} {basis : List Nat}
    {child : Nat → StepIn n q q' m m' W → LoopStateG n q q' m' → Prop}
    (hchild : LutKeyStepHyp child hf0 key hf1 x0 x1 x2 x3 x4 x5 f00 f01 f10 f11 backend base digits
      basis hashModel tag level tab)
    {L : Nat} (hL : L + 1 = W) {v0 v1 v2 v3 : ExactMatrix q' n 1 m'}
    (h0 : familyGetStatic f00 0 v0) (h1 : familyGetStatic f01 0 v1)
    (h2 : familyGetStatic f10 0 v2) (h3 : familyGetStatic f11 0 v3) {F : LoopStateG n q q' m'}
    (hruns : MxxIR.IterRuns (fun i current next ↦
      child i (stepIn current hf0 key hf1 x0 x1 x2 x3 x4 x5 f00 f01 f10 f11) next) L
      ((0 : ExactMatrix q n 1 m'), (0 : ExactMatrix q n 1 m'), (0 : ExactMatrix q n 1 m'),
        (0 : ExactMatrix q n 1 m'), (0 : ExactMatrix q n 1 m'), (0 : ExactMatrix q n 1 m'),
        v0, v1, v2, v3, ()) F)
    (hn : 0 < n) {bound : Nat} (hq' : 1 < q') (layout : RegularLayout q')
    (hlayout : backend.regularLayout q' n = some layout) (hexact : layout.droppedModuli = 0)
    (hm : m = 1 * layout.digitCount)
    {rc0 rA00 rA10 rc1 rA01 rA11 : ExactMatrix q n 1 m}
    (hr0 : ringAutomorphismRuns (MxxIR.selectInt (Int.ofNat L) tab) (inC x0 x1 x2 x3 x4 x5 0) rc0)
    (hr1 : ringAutomorphismRuns (MxxIR.selectInt (Int.ofNat L) tab) (inA x0 x1 x2 x3 x4 x5 0 0) rA00)
    (hr2 : ringAutomorphismRuns (MxxIR.selectInt (Int.ofNat L) tab) (inA x0 x1 x2 x3 x4 x5 1 0) rA10)
    (hr3 : ringAutomorphismRuns (MxxIR.selectInt (Int.ofNat L) tab) (inC x0 x1 x2 x3 x4 x5 1) rc1)
    (hr4 : ringAutomorphismRuns (MxxIR.selectInt (Int.ofNat L) tab) (inA x0 x1 x2 x3 x4 x5 0 1) rA01)
    (hr5 : ringAutomorphismRuns (MxxIR.selectInt (Int.ofNat L) tab) (inA x0 x1 x2 x3 x4 x5 1 1) rA11)
    {l01 r01 : ExactMatrix q n (1 + 1) m} {l r : ExactMatrix q n (1 + 1 + 1) m}
    {both : ExactMatrix q n (1 + 1 + 1) (m + m)}
    (hl01 : concatRows rc0 rA00 l01) (hl : concatRows l01 rA10 l)
    (hr01 : concatRows rc1 rA01 r01) (hr : concatRows r01 rA11 r) (hboth : concatColumns l r both)
    {Xb : Fin 2 → ExactMatrix q' n (1 + 1) m'}
    (hXb : ∀ j, concatRows (F.switch 0 j) (F.switch 1 j) (Xb j))
    {D' : Fin 2 → ExactMatrix q' n (m + m) m'}
    (hD' : ∀ j, gadgetDecomposeRuns backend base digits (Xb j) (D' j))
    {D : Fin 2 → ExactMatrix q n (m + m) m'} (hD : ∀ j, compactCenteredRebaseRuns bound (D' j) (D j))
    {M : Fin 2 → Fin 2 → ExactMatrix q n 1 m'}
    (hM0 : ∀ j, sliceMatrix (matrixMul both (D j)) 1 2 0 m' (M 0 j))
    (hM1 : ∀ j, sliceMatrix (matrixMul both (D j)) 2 3 0 m' (M 1 j))
    {Asw : Fin 2 → Fin 2 → ExactMatrix q' n 1 m'}
    (hAsw : ∀ k j, blockModSwitchRuns basis 1 (M k j) (Asw k j))
    {Z : Fin 2 → ExactMatrix q' n (1 + 1) m'} (hZ : ∀ j, concatRows (Asw 0 j) (Asw 1 j) (Z j))
    {E' : Fin 2 → ExactMatrix q' n (m + m) m'}
    (hE' : ∀ j, gadgetDecomposeRuns backend base digits (Z j) (E' j))
    {E : Fin 2 → ExactMatrix q n (m + m) m'} (hE : ∀ j, compactCenteredRebaseRuns bound (E' j) (E j))
    {h : Fin 2 → ExactMatrix q n 1 m} {Ad : Fin 2 → Fin 2 → ExactMatrix q n 1 m}
    (hAd : Ad = lutAG hashModel key q m tag level (MxxIR.selectInt (Int.ofNat L) tab))
    {hl01' hr01' : ExactMatrix q n (1 + 1) m} {hl' hr' : ExactMatrix q n (1 + 1 + 1) m}
    {H : ExactMatrix q n (1 + 1 + 1) (m + m)}
    (hhl01 : concatRows (h 0) (Ad 0 0) hl01') (hhl : concatRows hl01' (Ad 1 0) hl')
    (hhr01 : concatRows (h 1) (Ad 0 1) hr01') (hhr : concatRows hr01' (Ad 1 1) hr')
    (hH : concatColumns hl' hr' H)
    {K : Fin 2 → Fin 2 → ExactMatrix q n 1 m'}
    (hK0 : ∀ j, sliceMatrix (matrixMul H (E j)) 1 2 0 m' (K 0 j))
    (hK1 : ∀ j, sliceMatrix (matrixMul H (E j)) 2 3 0 m' (K 1 j)) :
    keyTupleG (fun k j ↦ F.key k j + K k j) = ∑ k ∈ Finset.range (L + 1), lutKeysG backend base digits basis
      hashModel key tag level (sigN tab) (inA x0 x1 x2 x3 x4 x5) (fam4 f00 f01 f10 f11) k := by
  subst hL
  obtain ⟨_, rfl⟩ := familyGet_lit 0 h0
  obtain ⟨_, rfl⟩ := familyGet_lit 0 h1
  obtain ⟨_, rfl⟩ := familyGet_lit 0 h2
  obtain ⟨_, rfl⟩ := familyGet_lit 0 h3
  let famF : Nat → Fin 2 → Fin 2 → ExactMatrix q' n 1 m' := fun k ↦
    if hk : k < L + 1 then fun a b ↦ fam4 f00 f01 f10 f11 a b ⟨k, hk⟩ else 0
  let G : Nat → (Fin 2 → Fin 2 → ExactMatrix q' n 1 m') → Fin 2 → Fin 2 → ExactMatrix q n 1 m' :=
    fun k sw ↦ branchKeyG backend base digits basis (MxxIR.selectInt (Int.ofNat k) tab).toNat
      (inA x0 x1 x2 x3 x4 x5) (lutAG hashModel key q m tag level (MxxIR.selectInt (Int.ofNat k) tab)) sw
  have hC := carried_loop_gen LoopStateG.key LoopStateG.switch famF G
    (by funext a b; simp only [famF, dif_pos (show 0 < L + 1 by omega)]; fin_cases a <;> fin_cases b <;> rfl)
    (fun k cur nxt _ hb ↦ by
      obtain ⟨hk, hk1, hs⟩ := hchild k cur nxt hb
      exact ⟨hk, by rw [hs]; simp only [famF, dif_pos hk1]⟩) hruns
  have hkey0 : LoopStateG.key ((0 : ExactMatrix q n 1 m'), (0 : ExactMatrix q n 1 m'),
      (0 : ExactMatrix q n 1 m'), (0 : ExactMatrix q n 1 m'), (0 : ExactMatrix q n 1 m'),
      (0 : ExactMatrix q n 1 m'), f00 ⟨0, by omega⟩, f01 ⟨0, by omega⟩, f10 ⟨0, by omega⟩,
      f11 ⟨0, by omega⟩, ()) = 0 := by
    funext a b; fin_cases a <;> fin_cases b <;> rfl
  rw [hkey0, zero_add] at hC
  obtain ⟨hLk, hLs⟩ := hC
  subst hAd
  have hK := branchKeyG_of_stacks (c := inC x0 x1 x2 x3 x4 x5) (A := inA x0 x1 x2 x3 x4 x5)
    hn hq' layout hlayout hexact hm hr0 hr1 hr2 hr3 hr4 hr5 hl01 hl hr01 hr hboth hXb hD' hD
    hM0 hM1 hAsw hZ hE' hE hhl01 hhl hhr01 hhr hH hK0 hK1
  have htot : F.key + K = ∑ k ∈ Finset.range (L + 1), G k (famF k) := by
    rw [Finset.sum_range_succ, hLk, hK, hLs]
  have hsum : keyTupleG (F.key + K) =
      ∑ k ∈ Finset.range (L + 1), lutKeysG backend base digits basis hashModel key tag level (sigN tab)
        (inA x0 x1 x2 x3 x4 x5) (fam4 f00 f01 f10 f11) k := by
    rw [htot, keyTupleG_sum]
    exact lut_keys_link_gen le_rfl famF (fun k hk ↦ by simp only [famF, dif_pos hk])
  exact hsum


/-- **The keys of a root LUT** (the key part of `lut_round_gen`): the branch keys
(`branch_keys_core`) moved down by `basis'`. -/
theorem lut_root_keys {q q' m m' W : Nat} {backend : BackendContext} {hashModel : HashModel}
    {base digits : Int} {tab : List Int} {key : ByteArray} {level : Int} {tag : Blob}
    {x0 x1 x2 x3 x4 x5 : ExactMatrix q n 1 m} {f00 f01 f10 f11 : Fin W → ExactMatrix q' n 1 m'}
    {basis : List Nat} {L : Nat} {Fk K : Fin 2 → Fin 2 → ExactMatrix q n 1 m'}
    (hcore : keyTupleG (fun k j ↦ Fk k j + K k j) = ∑ k ∈ Finset.range (L + 1), lutKeysG backend base
      digits basis hashModel key tag level (sigN tab) (inA x0 x1 x2 x3 x4 x5) (fam4 f00 f01 f10 f11) k)
    {q'' : Nat} {basis' : List Nat} {K' : Fin 2 → Fin 2 → ExactMatrix q'' n 1 m'}
    (hK' : ∀ k j, blockModSwitchRuns basis' 1 (Fk k j + K k j) (K' k j)) :
    keyTupleG K' = keyBmsG basis' (∑ k ∈ Finset.range (L + 1), lutKeysG backend base digits basis
      hashModel key tag level (sigN tab) (inA x0 x1 x2 x3 x4 x5) (fam4 f00 f01 f10 f11) k) := by
  have hK'e : K' = fun k j ↦ blockModSwitchG basis' 1 (Fk k j + K k j) :=
    funext fun k ↦ funext fun j ↦ blockModSwitchRuns_eqG (hK' k j)
  rw [← hcore, hK'e]
  rfl

set_option maxHeartbeats 64000000 in
/-- **The keys of a readout LUT member** (`lut.member.readout`; the key part of `LutMemberReadout.rel`,
no semantic hypothesis): the moved-down summed branch keys, `LutKeyMemberReadout.rel`'s form. -/
theorem LutMemberReadout.keys {q q' m m' W : Nat} {backend : BackendContext}
    {hashModel : HashModel} {base digits : Int} {tab : List Int} {key : ByteArray} {level : Int}
    {tag : Blob} {hf0 hf1 : Fin W → ExactMatrix q n 1 m} {x0 x1 x2 x3 x4 x5 : ExactMatrix q n 1 m}
    {f00 f01 f10 f11 : Fin W → ExactMatrix q' n 1 m'}
    {child : Nat → StepIn n q q' m m' W → LoopStateG n q q' m' → Prop} {i : Nat} {ℓ : Int}
    {bound : Nat} {basis : List Nat} {mw : Int} {basis' : List Nat}
    {outputs : ExactMatrix q' n 1 m' × ExactMatrix q' n 1 m' × ExactMatrix q' n 1 m' ×
      ExactMatrix q' n 1 m' × ExactMatrix q' n 1 m' × ExactMatrix q' n 1 m' × Unit}
    (h : LutMemberReadout.Comp q n m' q' m' q' m' W m W m m m W W W W m' m' q' q' q' q' q' q' q' q'
      0 0 0 0 ℓ ℓ level tag ℓ tab level tag ℓ tab ℓ level tag ℓ tab level tag ℓ tab
      ℓ tab ℓ tab ℓ tab ℓ tab ℓ tab ℓ tab
      base digits bound mw basis mw basis base digits bound mw basis' base digits bound mw basis mw
      basis base digits bound mw basis' mw basis' mw basis' mw mw basis' mw mw basis'
      backend hashModel i child (f00, f01, f10, f11, hf0, key, hf1, x0, x1, x2, x3, x4, x5, ())
      outputs)
    (hmw : mw = (m' : Int)) (L : Nat) (hℓ : ℓ = (L : Int)) (hL : L + 1 = W) (hn : 0 < n)
    (hq' : 1 < q') (layout : RegularLayout q') (hlayout : backend.regularLayout q' n = some layout)
    (hexact : layout.droppedModuli = 0) (hm : m = 1 * layout.digitCount)
    (hchild : LutKeyStepHyp child hf0 key hf1 x0 x1 x2 x3 x4 x5 f00 f01 f10 f11 backend base digits
      basis hashModel tag level tab) :
    ((outputs.1, outputs.2.1, outputs.2.2.1, outputs.2.2.2.1, ()) : KeysG n q' m') =
      keyBmsG basis' (∑ k ∈ Finset.range (L + 1), lutKeysG backend base digits basis hashModel key tag level
        (sigN tab) (inA x0 x1 x2 x3 x4 x5) (fam4 f00 f01 f10 f11) k) := by
  subst hmw hℓ
  refine Exists.elim h (fun v0 h ↦ Exists.elim h (fun v1 h ↦ Exists.elim h (fun v2 h ↦ Exists.elim h (fun v3 h ↦ Exists.elim h (fun v4 h ↦ Exists.elim h (fun v5 h ↦ Exists.elim h (fun v6 h ↦ Exists.elim h (fun v7 h ↦ Exists.elim h (fun v8 h ↦ Exists.elim h (fun v9 h ↦ Exists.elim h (fun v10 h ↦ Exists.elim h (fun v11 h ↦ Exists.elim h (fun v12 h ↦ Exists.elim h (fun v13 h ↦ Exists.elim h (fun v14 h ↦ Exists.elim h (fun v15 h ↦ Exists.elim h (fun v16 h ↦ Exists.elim h (fun v17 h ↦ Exists.elim h (fun v18 h ↦ Exists.elim h (fun v19 h ↦ Exists.elim h (fun v20 h ↦ Exists.elim h (fun v21 h ↦ Exists.elim h (fun v22 h ↦ Exists.elim h (fun v23 h ↦ Exists.elim h (fun v24 h ↦ Exists.elim h (fun v25 h ↦ Exists.elim h (fun v26 h ↦ Exists.elim h (fun v27 h ↦ Exists.elim h (fun v28 h ↦ Exists.elim h (fun v29 h ↦ Exists.elim h (fun v30 h ↦ Exists.elim h (fun v31 h ↦ Exists.elim h (fun v32 h ↦ Exists.elim h (fun v33 h ↦ Exists.elim h (fun v34 h ↦ Exists.elim h (fun v35 h ↦ Exists.elim h (fun v36 h ↦ Exists.elim h (fun v37 h ↦ Exists.elim h (fun v38 h ↦ Exists.elim h (fun v39 h ↦ Exists.elim h (fun v40 h ↦ Exists.elim h (fun v41 h ↦ Exists.elim h (fun v42 h ↦ Exists.elim h (fun v43 h ↦ Exists.elim h (fun v44 h ↦ Exists.elim h (fun v45 h ↦ Exists.elim h (fun v46 h ↦ Exists.elim h (fun v47 h ↦ Exists.elim h (fun v48 h ↦ Exists.elim h (fun v49 h ↦ Exists.elim h (fun v50 h ↦ Exists.elim h (fun v51 h ↦ Exists.elim h (fun v52 h ↦ Exists.elim h (fun v53 h ↦ Exists.elim h (fun v54 h ↦ Exists.elim h (fun v55 h ↦ Exists.elim h (fun v56 h ↦ Exists.elim h (fun v57 h ↦ Exists.elim h (fun v58 h ↦ Exists.elim h (fun v59 h ↦ Exists.elim h (fun v60 h ↦ Exists.elim h (fun v61 h ↦ Exists.elim h (fun v62 h ↦ Exists.elim h (fun v63 h ↦ Exists.elim h (fun v64 h ↦ Exists.elim h (fun v65 h ↦ Exists.elim h (fun v66 h ↦ Exists.elim h (fun v67 h ↦ Exists.elim h (fun v68 h ↦ Exists.elim h (fun v69 h ↦ ?_))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))
  obtain ⟨a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a30, a31, a32, a33, a34, a35, a36, a37, a38, a39, a40, a41, a42, a43, a44, a45, a46, a47, a48, a49, a50, a51, a52, a53, a54, a55, a56, a57, a58, a59, a60, rfl⟩ := h
  obtain ⟨_, rfl⟩ := familyGet_lit L a5
  obtain ⟨_, rfl⟩ := familyGet_lit L a10
  obtain ⟨rfl, rfl, rfl, rfl⟩ := lutA_of_samples a6 a11 a7 a12
  exact lut_root_keys (branch_keys_core hchild (L := L) hL a0 a1 a2 a3 a4 hn hq' layout hlayout hexact hm
    a16 a17 a18 a21 a22 a23 a19 a20 a24 a25 a26
    (Xb := ![v36, v48]) (Fin.forall_fin_two.2 ⟨a27, a39⟩)
    (D' := ![v37, v49]) (Fin.forall_fin_two.2 ⟨a28, a40⟩)
    (D := ![v38, v50]) (Fin.forall_fin_two.2 ⟨a29, a41⟩)
    (M := ![![v39, v51], ![v41, v53]]) (Fin.forall_fin_two.2 ⟨a30, a42⟩) (Fin.forall_fin_two.2 ⟨a32, a44⟩)
    (Asw := ![![v40, v52], ![v42, v54]]) (Fin.forall_fin_two.2 ⟨Fin.forall_fin_two.2 ⟨a31, a43⟩, Fin.forall_fin_two.2 ⟨a33, a45⟩⟩)
    (Z := ![v43, v55]) (Fin.forall_fin_two.2 ⟨a34, a46⟩)
    (E' := ![v44, v56]) (Fin.forall_fin_two.2 ⟨a35, a47⟩)
    (E := ![v45, v57]) (Fin.forall_fin_two.2 ⟨a36, a48⟩)
    (h := ![hf0 ⟨L, by omega⟩, hf1 ⟨L, by omega⟩]) (Ad := lutAG hashModel key q m tag level (MxxIR.selectInt (Int.ofNat L) tab)) rfl
    a8 a9 a13 a14 a15
    (K := ![![v46, v58], ![v60, v62]]) (Fin.forall_fin_two.2 ⟨a37, a49⟩) (Fin.forall_fin_two.2 ⟨a51, a53⟩))
    (K' := ![![v47, v59], ![v61, v63]])
    (Fin.forall_fin_two.2 ⟨Fin.forall_fin_two.2 ⟨a38, a50⟩, Fin.forall_fin_two.2 ⟨a52, a54⟩⟩)

set_option maxHeartbeats 64000000 in
/-- **The keys of a sampling LUT member** (`lut.member.sampler`; the key part of `LutMemberSampler.rel`):
the summed branch keys, `LutKeyMemberSampler.rel`'s form without the offset. -/
theorem LutMemberSampler.keys {q q' m m' W : Nat} {backend : BackendContext}
    {hashModel : HashModel} {base digits : Int} {tab : List Int} {key : ByteArray} {level : Int}
    {tag : Blob} {hf0 hf1 : Fin W → ExactMatrix q n 1 m} {x0 x1 x2 x3 x4 x5 : ExactMatrix q n 1 m}
    {f00 f01 f10 f11 : Fin W → ExactMatrix q' n 1 m'}
    {child : Nat → StepIn n q q' m m' W → LoopStateG n q q' m' → Prop} {i : Nat} {ℓ : Int}
    {bound : Nat} {basis : List Nat} {mw : Int}
    {outputs : ExactMatrix q n 1 m' × ExactMatrix q n 1 m' × ExactMatrix q n 1 m' ×
      ExactMatrix q n 1 m' × ExactMatrix q n 1 m' × ExactMatrix q n 1 m' × Unit}
    (h : LutMemberSampler.Comp q n m' q' m' q' m' W m W m m m W W W W m' m' q' q'
      0 0 0 0 ℓ ℓ level tag ℓ tab level tag ℓ tab ℓ level tag ℓ tab level tag ℓ tab
      ℓ tab ℓ tab ℓ tab ℓ tab ℓ tab ℓ tab
      base digits bound mw basis mw basis base digits bound mw base digits bound mw basis mw basis
      base digits bound mw mw mw mw mw mw mw
      backend hashModel i child (f00, f01, f10, f11, hf0, key, hf1, x0, x1, x2, x3, x4, x5, ())
      outputs)
    (hmw : mw = (m' : Int)) (L : Nat) (hℓ : ℓ = (L : Int)) (hL : L + 1 = W) (hn : 0 < n)
    (hq' : 1 < q') (layout : RegularLayout q') (hlayout : backend.regularLayout q' n = some layout)
    (hexact : layout.droppedModuli = 0) (hm : m = 1 * layout.digitCount)
    (hchild : LutKeyStepHyp child hf0 key hf1 x0 x1 x2 x3 x4 x5 f00 f01 f10 f11 backend base digits
      basis hashModel tag level tab) :
    ((outputs.1, outputs.2.1, outputs.2.2.1, outputs.2.2.2.1, ()) : KeysG n q m') = ∑ k ∈ Finset.range (L + 1), lutKeysG backend base digits basis hashModel key tag level
        (sigN tab) (inA x0 x1 x2 x3 x4 x5) (fam4 f00 f01 f10 f11) k := by
  subst hmw hℓ
  refine Exists.elim h (fun v0 h ↦ Exists.elim h (fun v1 h ↦ Exists.elim h (fun v2 h ↦ Exists.elim h (fun v3 h ↦ Exists.elim h (fun v4 h ↦ Exists.elim h (fun v5 h ↦ Exists.elim h (fun v6 h ↦ Exists.elim h (fun v7 h ↦ Exists.elim h (fun v8 h ↦ Exists.elim h (fun v9 h ↦ Exists.elim h (fun v10 h ↦ Exists.elim h (fun v11 h ↦ Exists.elim h (fun v12 h ↦ Exists.elim h (fun v13 h ↦ Exists.elim h (fun v14 h ↦ Exists.elim h (fun v15 h ↦ Exists.elim h (fun v16 h ↦ Exists.elim h (fun v17 h ↦ Exists.elim h (fun v18 h ↦ Exists.elim h (fun v19 h ↦ Exists.elim h (fun v20 h ↦ Exists.elim h (fun v21 h ↦ Exists.elim h (fun v22 h ↦ Exists.elim h (fun v23 h ↦ Exists.elim h (fun v24 h ↦ Exists.elim h (fun v25 h ↦ Exists.elim h (fun v26 h ↦ Exists.elim h (fun v27 h ↦ Exists.elim h (fun v28 h ↦ Exists.elim h (fun v29 h ↦ Exists.elim h (fun v30 h ↦ Exists.elim h (fun v31 h ↦ Exists.elim h (fun v32 h ↦ Exists.elim h (fun v33 h ↦ Exists.elim h (fun v34 h ↦ Exists.elim h (fun v35 h ↦ Exists.elim h (fun v36 h ↦ Exists.elim h (fun v37 h ↦ Exists.elim h (fun v38 h ↦ Exists.elim h (fun v39 h ↦ Exists.elim h (fun v40 h ↦ Exists.elim h (fun v41 h ↦ Exists.elim h (fun v42 h ↦ Exists.elim h (fun v43 h ↦ Exists.elim h (fun v44 h ↦ Exists.elim h (fun v45 h ↦ Exists.elim h (fun v46 h ↦ Exists.elim h (fun v47 h ↦ Exists.elim h (fun v48 h ↦ Exists.elim h (fun v49 h ↦ Exists.elim h (fun v50 h ↦ Exists.elim h (fun v51 h ↦ Exists.elim h (fun v52 h ↦ Exists.elim h (fun v53 h ↦ Exists.elim h (fun v54 h ↦ Exists.elim h (fun v55 h ↦ Exists.elim h (fun v56 h ↦ Exists.elim h (fun v57 h ↦ Exists.elim h (fun v58 h ↦ Exists.elim h (fun v59 h ↦ Exists.elim h (fun v60 h ↦ Exists.elim h (fun v61 h ↦ Exists.elim h (fun v62 h ↦ Exists.elim h (fun v63 h ↦ ?_))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))
  obtain ⟨a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a30, a31, a32, a33, a34, a35, a36, a37, a38, a39, a40, a41, a42, a43, a44, a45, a46, a47, a48, a49, a50, a51, a52, a53, a54, rfl⟩ := h
  obtain ⟨_, rfl⟩ := familyGet_lit L a5
  obtain ⟨_, rfl⟩ := familyGet_lit L a10
  obtain ⟨rfl, rfl, rfl, rfl⟩ := lutA_of_samples a6 a11 a7 a12
  exact branch_keys_core hchild (L := L) hL a0 a1 a2 a3 a4 hn hq' layout hlayout hexact hm
    a16 a17 a18 a21 a22 a23 a19 a20 a24 a25 a26
    (Xb := ![v36, v47]) (Fin.forall_fin_two.2 ⟨a27, a38⟩)
    (D' := ![v37, v48]) (Fin.forall_fin_two.2 ⟨a28, a39⟩)
    (D := ![v38, v49]) (Fin.forall_fin_two.2 ⟨a29, a40⟩)
    (M := ![![v39, v50], ![v41, v52]]) (Fin.forall_fin_two.2 ⟨a30, a41⟩) (Fin.forall_fin_two.2 ⟨a32, a43⟩)
    (Asw := ![![v40, v51], ![v42, v53]]) (Fin.forall_fin_two.2 ⟨Fin.forall_fin_two.2 ⟨a31, a42⟩, Fin.forall_fin_two.2 ⟨a33, a44⟩⟩)
    (Z := ![v43, v54]) (Fin.forall_fin_two.2 ⟨a34, a45⟩)
    (E' := ![v44, v55]) (Fin.forall_fin_two.2 ⟨a35, a46⟩)
    (E := ![v45, v56]) (Fin.forall_fin_two.2 ⟨a36, a47⟩)
    (h := ![hf0 ⟨L, by omega⟩, hf1 ⟨L, by omega⟩]) (Ad := lutAG hashModel key q m tag level (MxxIR.selectInt (Int.ofNat L) tab)) rfl
    a8 a9 a13 a14 a15
    (K := ![![v46, v57], ![v58, v59]]) (Fin.forall_fin_two.2 ⟨a37, a48⟩) (Fin.forall_fin_two.2 ⟨a49, a50⟩)

set_option maxHeartbeats 64000000 in
/-- **The keys of a sampling LUT member with the offset** (`lut.member.sampler-offset`; the key part
of `LutMemberSamplerOffset.rel`): the summed branch keys plus the offset keys
(`LutKeyMemberSampler.rel`'s form). -/
theorem LutMemberSamplerOffset.keys {q q' m m' W : Nat} {backend : BackendContext}
    {hashModel : HashModel} {base digits : Int} {tab : List Int} {key : ByteArray} {level : Int}
    {tag : Blob} {hf0 hf1 : Fin W → ExactMatrix q n 1 m} {x0 x1 x2 x3 x4 x5 : ExactMatrix q n 1 m}
    {f00 f01 f10 f11 : Fin W → ExactMatrix q' n 1 m'}
    {child : Nat → StepIn n q q' m m' W → LoopStateG n q q' m' → Prop} {i : Nat} {ℓ : Int}
    {bound : Nat} {basis : List Nat} {mw : Int} {o0 o1 o2 o3 o4 o5 : ExactMatrix q n 1 m'}
    {outputs : ExactMatrix q n 1 m' × ExactMatrix q n 1 m' × ExactMatrix q n 1 m' ×
      ExactMatrix q n 1 m' × ExactMatrix q n 1 m' × ExactMatrix q n 1 m' × Unit}
    (h : LutMemberSamplerOffset.Comp q n m' q' m' q' m' W m W m m m W W W W m' m' q' q'
      0 0 0 0 ℓ ℓ level tag ℓ tab level tag ℓ tab ℓ level tag ℓ tab level tag ℓ tab
      ℓ tab ℓ tab ℓ tab ℓ tab ℓ tab ℓ tab
      base digits bound mw basis mw basis base digits bound mw base digits bound mw basis mw basis
      base digits bound mw mw mw mw mw mw mw
      backend hashModel i child
      (f00, f01, f10, f11, hf0, key, hf1, x0, x1, x2, x3, x4, x5, o0, o1, o2, o3, o4, o5, ()) outputs)
    (hmw : mw = (m' : Int)) (L : Nat) (hℓ : ℓ = (L : Int)) (hL : L + 1 = W) (hn : 0 < n)
    (hq' : 1 < q') (layout : RegularLayout q') (hlayout : backend.regularLayout q' n = some layout)
    (hexact : layout.droppedModuli = 0) (hm : m = 1 * layout.digitCount)
    (hchild : LutKeyStepHyp child hf0 key hf1 x0 x1 x2 x3 x4 x5 f00 f01 f10 f11 backend base digits
      basis hashModel tag level tab) :
    ((outputs.1, outputs.2.1, outputs.2.2.1, outputs.2.2.2.1, ()) : KeysG n q m') =
      (∑ k ∈ Finset.range (L + 1), lutKeysG backend base digits basis hashModel key tag level
        (sigN tab) (inA x0 x1 x2 x3 x4 x5) (fam4 f00 f01 f10 f11) k) + (o0, o1, o2, o3, ()) := by
  subst hmw hℓ
  refine Exists.elim h (fun v0 h ↦ Exists.elim h (fun v1 h ↦ Exists.elim h (fun v2 h ↦ Exists.elim h (fun v3 h ↦ Exists.elim h (fun v4 h ↦ Exists.elim h (fun v5 h ↦ Exists.elim h (fun v6 h ↦ Exists.elim h (fun v7 h ↦ Exists.elim h (fun v8 h ↦ Exists.elim h (fun v9 h ↦ Exists.elim h (fun v10 h ↦ Exists.elim h (fun v11 h ↦ Exists.elim h (fun v12 h ↦ Exists.elim h (fun v13 h ↦ Exists.elim h (fun v14 h ↦ Exists.elim h (fun v15 h ↦ Exists.elim h (fun v16 h ↦ Exists.elim h (fun v17 h ↦ Exists.elim h (fun v18 h ↦ Exists.elim h (fun v19 h ↦ Exists.elim h (fun v20 h ↦ Exists.elim h (fun v21 h ↦ Exists.elim h (fun v22 h ↦ Exists.elim h (fun v23 h ↦ Exists.elim h (fun v24 h ↦ Exists.elim h (fun v25 h ↦ Exists.elim h (fun v26 h ↦ Exists.elim h (fun v27 h ↦ Exists.elim h (fun v28 h ↦ Exists.elim h (fun v29 h ↦ Exists.elim h (fun v30 h ↦ Exists.elim h (fun v31 h ↦ Exists.elim h (fun v32 h ↦ Exists.elim h (fun v33 h ↦ Exists.elim h (fun v34 h ↦ Exists.elim h (fun v35 h ↦ Exists.elim h (fun v36 h ↦ Exists.elim h (fun v37 h ↦ Exists.elim h (fun v38 h ↦ Exists.elim h (fun v39 h ↦ Exists.elim h (fun v40 h ↦ Exists.elim h (fun v41 h ↦ Exists.elim h (fun v42 h ↦ Exists.elim h (fun v43 h ↦ Exists.elim h (fun v44 h ↦ Exists.elim h (fun v45 h ↦ Exists.elim h (fun v46 h ↦ Exists.elim h (fun v47 h ↦ Exists.elim h (fun v48 h ↦ Exists.elim h (fun v49 h ↦ Exists.elim h (fun v50 h ↦ Exists.elim h (fun v51 h ↦ Exists.elim h (fun v52 h ↦ Exists.elim h (fun v53 h ↦ Exists.elim h (fun v54 h ↦ Exists.elim h (fun v55 h ↦ Exists.elim h (fun v56 h ↦ Exists.elim h (fun v57 h ↦ Exists.elim h (fun v58 h ↦ Exists.elim h (fun v59 h ↦ Exists.elim h (fun v60 h ↦ Exists.elim h (fun v61 h ↦ Exists.elim h (fun v62 h ↦ Exists.elim h (fun v63 h ↦ ?_))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))
  obtain ⟨a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a30, a31, a32, a33, a34, a35, a36, a37, a38, a39, a40, a41, a42, a43, a44, a45, a46, a47, a48, a49, a50, a51, a52, a53, a54, rfl⟩ := h
  obtain ⟨_, rfl⟩ := familyGet_lit L a5
  obtain ⟨_, rfl⟩ := familyGet_lit L a10
  obtain ⟨rfl, rfl, rfl, rfl⟩ := lutA_of_samples a6 a11 a7 a12
  rw [← branch_keys_core hchild (L := L) hL a0 a1 a2 a3 a4 hn hq' layout hlayout hexact hm
      a16 a17 a18 a21 a22 a23 a19 a20 a24 a25 a26
      (Xb := ![v36, v47]) (Fin.forall_fin_two.2 ⟨a27, a38⟩)
      (D' := ![v37, v48]) (Fin.forall_fin_two.2 ⟨a28, a39⟩)
      (D := ![v38, v49]) (Fin.forall_fin_two.2 ⟨a29, a40⟩)
      (M := ![![v39, v50], ![v41, v52]]) (Fin.forall_fin_two.2 ⟨a30, a41⟩) (Fin.forall_fin_two.2 ⟨a32, a43⟩)
      (Asw := ![![v40, v51], ![v42, v53]]) (Fin.forall_fin_two.2 ⟨Fin.forall_fin_two.2 ⟨a31, a42⟩, Fin.forall_fin_two.2 ⟨a33, a44⟩⟩)
      (Z := ![v43, v54]) (Fin.forall_fin_two.2 ⟨a34, a45⟩)
      (E' := ![v44, v55]) (Fin.forall_fin_two.2 ⟨a35, a46⟩)
      (E := ![v45, v56]) (Fin.forall_fin_two.2 ⟨a36, a47⟩)
      (h := ![hf0 ⟨L, by omega⟩, hf1 ⟨L, by omega⟩]) (Ad := lutAG hashModel key q m tag level (MxxIR.selectInt (Int.ofNat L) tab)) rfl
      a8 a9 a13 a14 a15
      (K := ![![v46, v57], ![v58, v59]]) (Fin.forall_fin_two.2 ⟨a37, a48⟩) (Fin.forall_fin_two.2 ⟨a49, a50⟩)]
  rfl

end RefreshTopProof.GenIface
