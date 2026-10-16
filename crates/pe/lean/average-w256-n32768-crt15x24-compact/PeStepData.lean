import PeStepSum

/-!
# The branches of a step from the online runtime data

The rows that the online evaluation computes for its branches (`BranchData.row`,
`Leaf2Data.row`) are the branch rows of `branch1_rel`.
-/

set_option maxRecDepth 200000

namespace PeProof

open Mxx.Primitives MxxRuntime MxxExponentLut MxxPe

/-- The key of first-depth branch `b`. -/
noncomputable def BranchData.key1 (d : BranchData) (model : HashModel) (b : Fin 30) :
    Fin 2 → Fin 2 → ExactMatrix Q N 1 30 :=
  fun r j ↦ pairMul (helperKey model d.key (kSw b) r) (actionDigits (d.Tb0 b b.isLt) (d.Tb1 b b.isLt) j)

set_option maxHeartbeats 64000000 in
/-- First-depth branch `b` of the online data. -/
theorem data_branch1 {model : HashModel} (d : BranchData) {s μ : ErrorPoly N} (hs : polyNorm s ≤ 1)
    {A : Fin 2 → Fin 2 → ExactMatrix Q N 1 30} {B : Nat}
    (henc : Rel s A (fun j ↦ reducePoly Q N ((((1 : Nat) : Int) : ErrorPoly N) * μ * unitPair s j) • g0)
      ![d.c0, d.c1] B)
    (b : Fin 30) (hμ : SmulGain (substitution (kSw b) (kSw_odd b) μ) 1)
    (hhelp : ∀ (i : Fin 45) (k : Nat) (hk : Odd k), kHelp i = k → Rel s (helperKey model d.key k)
      (fun j ↦ reducePoly Q N ((((1 : Nat) : Int) : ErrorPoly N) * unitPair (substitution k hk s) j) • g0)
      ![d.hf0 i, d.hf1 i] 31)
    {zc0 zc1 : Fin 54000 → ExactMatrix PQ N 1 1} {ez : Fin 54000 → Fin 2 → ErrorMatrix N 1 1}
    (hcomp : ∀ i : Fin 54000,
      RelE (substitution (kSw (i / 1800)) (kSw_odd ⟨i / 1800, by omega⟩) s)
        (lutCompKey model d.key (kSw (i / 1800)) ((i % 60 : Nat) : Int) ((30 + (i / 60) % 30 : Nat) : Int))
        (fun j _ _ ↦ reducePoly PQ N (((P : Nat) : Int) *
          (entryDigit (joinedEntry ![switchBot model d.key d.hf0 d.hf1 ⟨i / 1800, by omega⟩ 0,
              switchBot model d.key d.hf0 d.hf1 ⟨i / 1800, by omega⟩ 1] ⟨i % 60, by omega⟩)
            ⟨(i / 60) % 30, by omega⟩ *
            unitPair (substitution (kSw (i / 1800)) (kSw_odd ⟨i / 1800, by omega⟩) s) j)))
        ![zc0 i, zc1 i] (ez i) ∧ ∀ j, CoeffBound (ez i j) 31)
    (hE : ∀ jj, d.Eb b b.isLt 0 jj = ∑ a, lutCompanionSum A (expandFam zc0) (expandFam zc1) b jj a)
    (hT : ∀ r jj, ![d.Tb0 b b.isLt, d.Tb1 b b.isLt] r 0 jj = ∑ a, lutKeyCompanionSum model d.key b A r jj a +
      ∑ a : Fin 30, substitution (kSw b) (kSw_odd b) (A r 0 0 a) * rowDigits (switchTopRow model d.key b) a jj)
    (hruns : d.Runs model b b.isLt) :
    Rel s (d.key1 model b)
      (fun j ↦ reducePoly Q N ((((1 : Nat) : Int) : ErrorPoly N) * substitution (kSw b) (kSw_odd b) μ *
        unitPair s j) • g0)
      (fun j ↦ d.row model b b.isLt j) (switchBound B 31 companionBound 1) := by
  have h := branch1_rel hs henc b hμ hhelp hcomp hE hT hruns.2 hruns.1
  have hrow : (fun j ↦ d.row model b b.isLt j) = fun j ↦
      pairMul ![d.hf0 ⟨b, by omega⟩, d.hf1 ⟨b, by omega⟩] (actionDigits (d.Tb0 b b.isLt) (d.Tb1 b b.isLt) j) +
        (pairMul (fun i ↦ substituteMatrix (kSw b) (kSw_odd b) (![d.c0, d.c1] i))
          (actionDigits (joinRow fun j ↦ substituteMatrix (kSw b) (kSw_odd b) (helperKey model d.key (kInv b) 0 j))
            (joinRow fun j ↦ substituteMatrix (kSw b) (kSw_odd b) (helperKey model d.key (kInv b) 1 j) -
              substituteMatrix (kSw b) (kSw_odd b) (![d.hf0 ⟨44 - b, by omega⟩, d.hf1 ⟨44 - b, by omega⟩] j)) j) +
          halves (d.Eb b b.isLt) j) := by
    funext j
    unfold BranchData.row
    congr 2
  rw [hrow]
  exact h

/-- The key of second-depth leaf `l`. -/
noncomputable def Leaf2Data.key2 (d : Leaf2Data) (model : HashModel) (l : Fin 225) :
    Fin 2 → Fin 2 → ExactMatrix Q N 1 30 :=
  fun r j ↦ pairMul (helperKey model d.key (kSw (15 + l / 15)) r)
    (actionDigits (d.Tb0 l l.isLt) (d.Tb1 l l.isLt) j)

set_option maxHeartbeats 64000000 in
/-- Second-depth leaf `l` of the online data, from the encoding of its parent. -/
theorem data_branch2 {model : HashModel} (d : Leaf2Data) {s μ : ErrorPoly N} (hs : polyNorm s ≤ 1)
    (l : Fin 225) {A : Fin 2 → Fin 2 → ExactMatrix Q N 1 30} {B : Nat}
    (henc : Rel s A (fun j ↦ reducePoly Q N ((((1 : Nat) : Int) : ErrorPoly N) * μ * unitPair s j) • g0)
      ![d.c0 ⟨l % 15, by omega⟩, d.c1 ⟨l % 15, by omega⟩] B)
    (hμ : SmulGain (substitution (kSw (15 + l / 15)) (kSw_odd (member2 l l.isLt)) μ) 1)
    (hhelp : ∀ (i : Fin 45) (k : Nat) (hk : Odd k), kHelp i = k → Rel s (helperKey model d.key k)
      (fun j ↦ reducePoly Q N ((((1 : Nat) : Int) : ErrorPoly N) * unitPair (substitution k hk s) j) • g0)
      ![d.hf0 i, d.hf1 i] 31)
    {zc0 zc1 : Fin 54000 → ExactMatrix PQ N 1 1} {ez : Fin 54000 → Fin 2 → ErrorMatrix N 1 1}
    (hcomp : ∀ i : Fin 54000,
      RelE (substitution (kSw (i / 1800)) (kSw_odd ⟨i / 1800, by omega⟩) s)
        (lutCompKey model d.key (kSw (i / 1800)) ((i % 60 : Nat) : Int) ((30 + (i / 60) % 30 : Nat) : Int))
        (fun j _ _ ↦ reducePoly PQ N (((P : Nat) : Int) *
          (entryDigit (joinedEntry ![switchBot model d.key d.hf0 d.hf1 ⟨i / 1800, by omega⟩ 0,
              switchBot model d.key d.hf0 d.hf1 ⟨i / 1800, by omega⟩ 1] ⟨i % 60, by omega⟩)
            ⟨(i / 60) % 30, by omega⟩ *
            unitPair (substitution (kSw (i / 1800)) (kSw_odd ⟨i / 1800, by omega⟩) s) j)))
        ![zc0 i, zc1 i] (ez i) ∧ ∀ j, CoeffBound (ez i j) 31)
    (hE : ∀ jj, d.Eb l l.isLt 0 jj = ∑ a, lutCompanionSum A (expandFam zc0) (expandFam zc1) (member2 l l.isLt) jj a)
    (hT : ∀ r jj, ![d.Tb0 l l.isLt, d.Tb1 l l.isLt] r 0 jj =
      ∑ a, lutKeyCompanionSum model d.key (member2 l l.isLt) A r jj a +
      ∑ a : Fin 30, substitution (kSw (15 + l / 15)) (kSw_odd (member2 l l.isLt)) (A r 0 0 a) *
        rowDigits (switchTopRow model d.key (member2 l l.isLt)) a jj)
    (hruns : d.Runs model l l.isLt) :
    Rel s (d.key2 model l)
      (fun j ↦ reducePoly Q N ((((1 : Nat) : Int) : ErrorPoly N) *
        substitution (kSw (15 + l / 15)) (kSw_odd (member2 l l.isLt)) μ * unitPair s j) • g0)
      (fun j ↦ d.row model l l.isLt j) (switchBound B 31 companionBound 1) := by
  have h := branch1_rel hs henc (member2 l l.isLt) hμ hhelp hcomp hE hT hruns.2 hruns.1
  have hrow : (fun j ↦ d.row model l l.isLt j) = fun j ↦
      pairMul ![d.hf0 ⟨member2 l l.isLt, by simp [member2]; omega⟩, d.hf1 ⟨member2 l l.isLt, by simp [member2]; omega⟩]
          (actionDigits (d.Tb0 l l.isLt) (d.Tb1 l l.isLt) j) +
        (pairMul (fun i ↦ substituteMatrix (kSw (member2 l l.isLt)) (kSw_odd (member2 l l.isLt))
            (![d.c0 ⟨l % 15, by omega⟩, d.c1 ⟨l % 15, by omega⟩] i))
          (actionDigits (joinRow fun j ↦ substituteMatrix (kSw (member2 l l.isLt)) (kSw_odd (member2 l l.isLt))
              (helperKey model d.key (kInv (member2 l l.isLt)) 0 j))
            (joinRow fun j ↦ substituteMatrix (kSw (member2 l l.isLt)) (kSw_odd (member2 l l.isLt))
                (helperKey model d.key (kInv (member2 l l.isLt)) 1 j) -
              substituteMatrix (kSw (member2 l l.isLt)) (kSw_odd (member2 l l.isLt))
                (![d.hf0 ⟨44 - member2 l l.isLt, by omega⟩, d.hf1 ⟨44 - member2 l l.isLt, by omega⟩] j)) j) +
          halves (d.Eb l l.isLt) j) := by
    funext j
    unfold Leaf2Data.row
    congr 2
  rw [hrow]
  exact h

/-- A routed monomial has gain one. -/
theorem smulGain_subst_Xe (k : Nat) (hk : Odd k) (e : Nat) : SmulGain (substitution k hk (Xe (n := N) e)) 1 := by
  rw [Xe, substitution_root_pow]
  exact smulGain_root_pow hN1 _

theorem smulGain_subst2_Xe (k k' : Nat) (hk : Odd k) (hk' : Odd k') (e : Nat) :
    SmulGain (substitution k hk (substitution k' hk' (Xe (n := N) e))) 1 := by
  rw [Xe, substitution_root_pow, substitution_root_pow]
  exact smulGain_root_pow hN1 _

/-- The hypotheses of the branches of a step, for the online branch data `d1`, `d2` built on the
product rows. -/
structure BranchHyps (model : HashModel) (s : ErrorPoly N) (e : Nat) (M : Fin 2 → Fin 2 → ExactMatrix Q N 1 30)
    (d1 : BranchData) (d2 : Leaf2Data) (zc0 zc1 : Fin 54000 → ExactMatrix PQ N 1 1)
    (ez : Fin 54000 → Fin 2 → ErrorMatrix N 1 1) : Prop where
  hhelp : ∀ (i : Fin 45) (k : Nat) (hk : Odd k), kHelp i = k → Rel s (helperKey model d1.key k)
      (fun j ↦ reducePoly Q N ((((1 : Nat) : Int) : ErrorPoly N) * unitPair (substitution k hk s) j) • g0)
      ![d1.hf0 i, d1.hf1 i] 31
  hcomp : ∀ i : Fin 54000,
      RelE (substitution (kSw (i / 1800)) (kSw_odd ⟨i / 1800, by omega⟩) s)
        (lutCompKey model d1.key (kSw (i / 1800)) ((i % 60 : Nat) : Int) ((30 + (i / 60) % 30 : Nat) : Int))
        (fun j _ _ ↦ reducePoly PQ N (((P : Nat) : Int) *
          (entryDigit (joinedEntry ![switchBot model d1.key d1.hf0 d1.hf1 ⟨i / 1800, by omega⟩ 0,
              switchBot model d1.key d1.hf0 d1.hf1 ⟨i / 1800, by omega⟩ 1] ⟨i % 60, by omega⟩)
            ⟨(i / 60) % 30, by omega⟩ *
            unitPair (substitution (kSw (i / 1800)) (kSw_odd ⟨i / 1800, by omega⟩) s) j)))
        ![zc0 i, zc1 i] (ez i) ∧ ∀ j, CoeffBound (ez i j) 31
  hE1 : ∀ (b : Fin 30) jj, d1.Eb b b.isLt 0 jj = ∑ a, lutCompanionSum M (expandFam zc0) (expandFam zc1) b jj a
  hT1 : ∀ (b : Fin 30) r jj, ![d1.Tb0 b b.isLt, d1.Tb1 b b.isLt] r 0 jj =
      ∑ a, lutKeyCompanionSum model d1.key b M r jj a +
      ∑ a : Fin 30, substitution (kSw b) (kSw_odd b) (M r 0 0 a) * rowDigits (switchTopRow model d1.key b) a jj
  hruns1 : ∀ b : Fin 30, d1.Runs model b b.isLt
  hkey2 : d2.key = d1.key
  hf2 : d2.hf0 = d1.hf0 ∧ d2.hf1 = d1.hf1
  hc2 : ∀ p : Fin 15, d2.c0 p = d1.row model p (by omega) 0 ∧ d2.c1 p = d1.row model p (by omega) 1
  hE2 : ∀ (l : Fin 225) jj, d2.Eb l l.isLt 0 jj =
      ∑ a, lutCompanionSum (d1.key1 model ⟨l % 15, by omega⟩) (expandFam zc0) (expandFam zc1) (member2 l l.isLt) jj a
  hT2 : ∀ (l : Fin 225) r jj, ![d2.Tb0 l l.isLt, d2.Tb1 l l.isLt] r 0 jj =
      ∑ a, lutKeyCompanionSum model d1.key (member2 l l.isLt) (d1.key1 model ⟨l % 15, by omega⟩) r jj a +
      ∑ a : Fin 30, substitution (kSw (15 + l / 15)) (kSw_odd (member2 l l.isLt))
        (d1.key1 model ⟨l % 15, by omega⟩ r 0 0 a) *
        rowDigits (switchTopRow model d1.key (member2 l l.isLt)) a jj
  hruns2 : ∀ l : Fin 225, d2.Runs model l l.isLt

/-- The bound of a first-depth branch. -/
def bound1 (Bp : Nat) : Nat := switchBound Bp 31 companionBound 1

/-- The bound of a second-depth branch. -/
def bound2 (Bp : Nat) : Nat := switchBound (bound1 Bp) 31 companionBound 1

set_option maxRecDepth 200000 in
set_option maxHeartbeats 64000000 in
/-- One step: the product encoding of `X^e` (with `e < 256`) is carried through the 256 branches of
the online data, and their scalar-acted sum encodes the table value `T e`. -/
theorem step_correct {model : HashModel} {s : ErrorPoly N} (hs : polyNorm s ≤ 1) {e : Nat} (he : e < 256)
    {M : Fin 2 → Fin 2 → ExactMatrix Q N 1 30} {d1 : BranchData} {d2 : Leaf2Data} {Bp : Nat}
    (hP : Rel s M (fun j ↦ reducePoly Q N ((((1 : Nat) : Int) : ErrorPoly N) * Xe e * unitPair s j) • g0)
      ![d1.c0, d1.c1] Bp)
    {zc0 zc1 : Fin 54000 → ExactMatrix PQ N 1 1} {ez : Fin 54000 → Fin 2 → ErrorMatrix N 1 1}
    (H : BranchHyps model s e M d1 d2 zc0 zc1 ez)
    (table : Fin (2 ^ 8) → ExactPoly Q N) (coef : Nat → ExactMatrix Q N 1 1)
    (hcoef : ∀ k, Odd k → (Winv : ExactPoly Q N) * coef k 0 0 =
      lutFourierCoefficient N (X Q) (Winv : ExactPoly Q N) table k) :
    Rel s (fun r j ↦ M r j * scalarV (coef 1) +
        ∑ b : Fin 15, d1.key1 model ⟨b, by omega⟩ r j * scalarV (coef (kSw b)) +
        ∑ i : Fin 15, d1.key1 model ⟨15 + i, by omega⟩ r j * scalarV (coef (kSw (15 + i))) +
        ∑ l : Fin 225, d2.key2 model l r j * scalarV (coef (kLeaf2 l)))
      (fun j ↦ (reducePoly Q N (unitPair s j) * table ⟨e, by norm_num; omega⟩) • g0)
      (fun j ↦ ![d1.c0, d1.c1] j * scalarV (coef 1) +
        ∑ b : Fin 15, d1.row model b (by omega) j * scalarV (coef (kSw b)) +
        ∑ i : Fin 15, d1.row model (15 + i) (by omega) j * scalarV (coef (kSw (15 + i))) +
        ∑ l : Fin 225, d2.row model l l.isLt j * scalarV (coef (kLeaf2 l)))
      (stepBound Bp (bound1 Bp) (bound2 Bp)) := by
  have hb1 : ∀ b : Fin 30, Rel s (d1.key1 model b)
      (fun j ↦ reducePoly Q N ((((1 : Nat) : Int) : ErrorPoly N) * substitution (kSw b) (kSw_odd b) (Xe e) *
        unitPair s j) • g0) (fun j ↦ d1.row model b b.isLt j) (bound1 Bp) := fun b ↦
    data_branch1 d1 hs hP b (smulGain_subst_Xe _ _ e) H.hhelp H.hcomp (H.hE1 b) (H.hT1 b) (H.hruns1 b)
  have hhelp2 : ∀ (i : Fin 45) (k : Nat) (hk : Odd k), kHelp i = k → Rel s (helperKey model d2.key k)
      (fun j ↦ reducePoly Q N ((((1 : Nat) : Int) : ErrorPoly N) * unitPair (substitution k hk s) j) • g0)
      ![d2.hf0 i, d2.hf1 i] 31 := by
    rw [H.hkey2, H.hf2.1, H.hf2.2]; exact H.hhelp
  have hcomp2 := H.hcomp
  rw [← H.hkey2, ← H.hf2.1, ← H.hf2.2] at hcomp2
  have hl2 : ∀ l : Fin 225, Rel s (d2.key2 model l)
      (fun j ↦ reducePoly Q N ((((1 : Nat) : Int) : ErrorPoly N) *
        substitution (kSw (15 + l / 15)) (kSw_odd (member2 l l.isLt))
          (substitution (kSw (l % 15)) (kSw_odd ⟨l % 15, by omega⟩) (Xe e)) * unitPair s j) • g0)
      (fun j ↦ d2.row model l l.isLt j) (bound2 Bp) := by
    intro l
    have hpar := hb1 ⟨l % 15, by omega⟩
    have hrows : (fun j ↦ d1.row model (l % 15) (by omega) j) =
        ![d2.c0 ⟨l % 15, by omega⟩, d2.c1 ⟨l % 15, by omega⟩] := by
      funext j
      fin_cases j
      · exact (H.hc2 ⟨l % 15, by omega⟩).1.symm
      · exact (H.hc2 ⟨l % 15, by omega⟩).2.symm
    rw [hrows] at hpar
    have hT2 := H.hT2 l
    rw [← H.hkey2] at hT2
    exact data_branch2 d2 hs l hpar (smulGain_subst2_Xe _ _ _ _ e) hhelp2 hcomp2 (H.hE2 l) hT2 (H.hruns2 l)
  exact step_sum he table coef hcoef hP (fun b ↦ hb1 ⟨b, by omega⟩) (fun i ↦ hb1 ⟨15 + i, by omega⟩) hl2

end PeProof
