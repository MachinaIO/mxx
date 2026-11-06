import PeProduct

/-!
# The first-depth branches

Branch `b < 30` of a step switches the product encoding with the switching ciphertext of `kSw b`,
whose bottom row comes from the helper of `kInv b` (`switchBot`), the LUT companions of `b`, and the
imported key block `T_b`.
-/

set_option maxRecDepth 200000

namespace PeProof

open Mxx.Primitives MxxRuntime MxxExponentLut MxxPe

/-- The index arithmetic of a LUT companion. -/
theorem lutIndex_div (b : Fin 30) (a : Fin 30) (jj : Fin 60) :
    (lutIndex b a jj : Nat) / 1800 = b ∧ (lutIndex b a jj : Nat) % 60 = jj ∧
      (lutIndex b a jj : Nat) / 60 % 30 = a := by
  simp only [lutIndex]
  omega

set_option maxHeartbeats 64000000 in
/-- First-depth branch `b`: the branch row encodes `σ_b(μ)` under the branch key `H_b Dec(T_b)`. -/
theorem branch1_rel {model : HashModel} {key : ByteArray} {s μ : ErrorPoly N} (hs : polyNorm s ≤ 1)
    {A : Fin 2 → Fin 2 → ExactMatrix Q N 1 30} {c : Fin 2 → ExactMatrix Q N 1 30} {B : Nat}
    (henc : Rel s A (fun j ↦ reducePoly Q N ((((1 : Nat) : Int) : ErrorPoly N) * μ * unitPair s j) • g0) c B)
    (b : Fin 30) (hμ : SmulGain (substitution (kSw b) (kSw_odd b) μ) 1)
    {hf0 hf1 : Fin 45 → ExactMatrix Q N 1 30}
    (hhelp : ∀ (i : Fin 45) (k : Nat) (hk : Odd k), kHelp i = k → Rel s (helperKey model key k)
      (fun j ↦ reducePoly Q N ((((1 : Nat) : Int) : ErrorPoly N) * unitPair (substitution k hk s) j) • g0)
      ![hf0 i, hf1 i] 31)
    {zc0 zc1 : Fin 54000 → ExactMatrix PQ N 1 1} {ez : Fin 54000 → Fin 2 → ErrorMatrix N 1 1}
    (hcomp : ∀ i : Fin 54000,
      RelE (substitution (kSw (i / 1800)) (kSw_odd ⟨i / 1800, by omega⟩) s)
        (lutCompKey model key (kSw (i / 1800)) ((i % 60 : Nat) : Int) ((30 + (i / 60) % 30 : Nat) : Int))
        (fun j _ _ ↦ reducePoly PQ N (((P : Nat) : Int) *
          (entryDigit (joinedEntry ![switchBot model key hf0 hf1 ⟨i / 1800, by omega⟩ 0,
              switchBot model key hf0 hf1 ⟨i / 1800, by omega⟩ 1] ⟨i % 60, by omega⟩)
            ⟨(i / 60) % 30, by omega⟩ *
            unitPair (substitution (kSw (i / 1800)) (kSw_odd ⟨i / 1800, by omega⟩) s) j)))
        ![zc0 i, zc1 i] (ez i) ∧ ∀ j, CoeffBound (ez i j) 31)
    {E : ExactMatrix Q N 1 60}
    (hE : ∀ jj, E 0 jj = ∑ a, lutCompanionSum A (expandFam zc0) (expandFam zc1) b jj a)
    {Tb0 Tb1 : ExactMatrix Q N 1 60}
    (hT : ∀ r jj, ![Tb0, Tb1] r 0 jj = ∑ a, lutKeyCompanionSum model key b A r jj a +
      ∑ a : Fin 30, substitution (kSw b) (kSw_odd b) (A r 0 0 a) * rowDigits (switchTopRow model key b) a jj)
    (hD : ∀ (j : Fin 2) (l : Fin 30), ∃ D' : ExactMatrix Q N (30 + 30) 1,
      gadgetDecomposeRuns Backend.backend 4096 30
        (pairColumn (substituteMatrix (kSw b) (kSw_odd b) (helperKey model key (kInv b) 0 j) 0 l)
          ((substituteMatrix (kSw b) (kSw_odd b) (helperKey model key (kInv b) 1 j) -
            substituteMatrix (kSw b) (kSw_odd b) (![hf0 ⟨44 - b, by omega⟩, hf1 ⟨44 - b, by omega⟩] j)) 0 l)) D' ∧
        compactCenteredRebaseRuns 2048 D' (fun a _ ↦ actionDigits
          (joinRow fun j ↦ substituteMatrix (kSw b) (kSw_odd b) (helperKey model key (kInv b) 0 j))
          (joinRow fun j ↦ substituteMatrix (kSw b) (kSw_odd b) (helperKey model key (kInv b) 1 j) -
            substituteMatrix (kSw b) (kSw_odd b) (![hf0 ⟨44 - b, by omega⟩, hf1 ⟨44 - b, by omega⟩] j)) j a l))
    (hDt : ∀ (j : Fin 2) (l : Fin 30), ∃ D' : ExactMatrix Q N (30 + 30) 1,
      gadgetDecomposeRuns Backend.backend 4096 30 (pairColumn (halves Tb0 j 0 l) (halves Tb1 j 0 l)) D' ∧
        compactCenteredRebaseRuns 2048 D' (fun a _ ↦ actionDigits Tb0 Tb1 j a l)) :
    Rel s (fun r j ↦ pairMul (helperKey model key (kSw b) r) (actionDigits Tb0 Tb1 j))
      (fun j ↦ reducePoly Q N ((((1 : Nat) : Int) : ErrorPoly N) * substitution (kSw b) (kSw_odd b) μ *
        unitPair s j) • g0)
      (fun j ↦ pairMul ![hf0 ⟨b, by omega⟩, hf1 ⟨b, by omega⟩] (actionDigits Tb0 Tb1 j) +
        (pairMul (fun i ↦ substituteMatrix (kSw b) (kSw_odd b) (c i))
          (actionDigits (joinRow fun j ↦ substituteMatrix (kSw b) (kSw_odd b) (helperKey model key (kInv b) 0 j))
            (joinRow fun j ↦ substituteMatrix (kSw b) (kSw_odd b) (helperKey model key (kInv b) 1 j) -
              substituteMatrix (kSw b) (kSw_odd b) (![hf0 ⟨44 - b, by omega⟩, hf1 ⟨44 - b, by omega⟩] j)) j) +
          halves E j))
      (switchBound B 31 companionBound 1) := by
  have hσn : polyNorm (substitution (kSw b) (kSw_odd b) s) ≤ 1 :=
    (polyNorm_substitution_le hN _ (kSw_coprime b) s).trans hs
  have hhi := hhelp ⟨44 - b, by omega⟩ (kInv b) (kInv_odd b) (kHelp_inv b b.isLt)
  have hhk := hhelp ⟨b, by omega⟩ (kSw b) (kSw_odd b) (kHelp_sw b b.isLt)
  set top : Fin 2 → ExactMatrix Q N 1 30 := fun j ↦ substituteMatrix (kSw b) (kSw_odd b) (helperKey model key (kInv b) 0 j)
    with htop
  set bot : Fin 2 → ExactMatrix Q N 1 30 := fun j ↦ substituteMatrix (kSw b) (kSw_odd b) (helperKey model key (kInv b) 1 j) -
    substituteMatrix (kSw b) (kSw_odd b) (![hf0 ⟨44 - b, by omega⟩, hf1 ⟨44 - b, by omega⟩] j) with hbot
  have hbotsw : bot = ![switchBot model key hf0 hf1 b 0, switchBot model key hf0 hf1 b 1] := by
    funext j
    fin_cases j <;> rfl
  have htopsw : joinRow top = switchTopRow model key b := by
    unfold switchTopRow
    congr 1
    funext j
    fin_cases j <;> rfl
  -- The companions of the switch.
  have hz : ∀ (j : Fin 2) (l : Fin 30) (a : Fin 30), Rel (substitution (kSw b) (kSw_odd b) s)
      (expandKey (lutCompKey model key (kSw b) ((30 * (j : Nat) + l : Nat) : Int) ((30 + a : Nat) : Int)))
      (fun r ↦ (actionDigits (joinRow top) (joinRow bot) j (Fin.natAdd 30 a) l *
        reducePoly Q N (unitPair (substitution (kSw b) (kSw_odd b) s) r)) • g0)
      (fun r ↦ expand (![zc0 (lutIndex b a ⟨30 * (j : Nat) + l, by omega⟩),
        zc1 (lutIndex b a ⟨30 * (j : Nat) + l, by omega⟩)] r)) companionBound := by
    intro j l a
    obtain ⟨hr, hb⟩ := hcomp (lutIndex b a ⟨30 * (j : Nat) + l, by omega⟩)
    obtain ⟨h1, h2, h3⟩ := lutIndex_div b a ⟨30 * (j : Nat) + l, by omega⟩
    simp only [h1, h2, h3, Fin.eta] at hr
    have := companion_expand hσn hr hb
    refine Rel.target_eq this ?_
    funext r
    rw [actionDigits_bottom, hbotsw]
    rfl
  have h := switch_rel henc (kSw_odd b) (kInv_odd b) (kSw_kInv b) (kSw_coprime b) hhi hD hμ
    (Bc := fun j l a ↦ expandKey (lutCompKey model key (kSw b) ((30 * (j : Nat) + l : Nat) : Int) ((30 + a : Nat) : Int)))
    (z := fun j l a r ↦ expand (![zc0 (lutIndex b a ⟨30 * (j : Nat) + l, by omega⟩),
        zc1 (lutIndex b a ⟨30 * (j : Nat) + l, by omega⟩)] r))
    hz (E := fun j ↦ halves E j)
    (fun j l ↦ by
      show E 0 ⟨30 * (j : Nat) + l, by omega⟩ = _
      rw [hE]
      rfl)
    (Mc := fun r j ↦ fun _ l ↦ (∑ a : Fin 30, (expandKey (lutCompKey model key (kSw b) ((30 * (j : Nat) + l : Nat) : Int)
        ((30 + a : Nat) : Int)) r 0 * keyDigits (routedKey (kSw b) (kSw_odd b) A) (Fin.natAdd 30 a) 0 +
      expandKey (lutCompKey model key (kSw b) ((30 * (j : Nat) + l : Nat) : Int) ((30 + a : Nat) : Int)) r 1 *
        keyDigits (routedKey (kSw b) (kSw_odd b) A) (Fin.natAdd 30 a) 1)) 0 0)
    (fun r j l ↦ rfl) hhk (Tb0 := Tb0) (Tb1 := Tb1)
    (fun r j ↦ by
      funext x l
      rw [Subsingleton.elim x 0]
      show ![Tb0, Tb1] r 0 ⟨30 * (j : Nat) + l, by omega⟩ = _
      rw [hT]
      simp only [Matrix.add_apply, pairMulTop, Matrix.mul_apply, topRows, Matrix.submatrix_apply, id]
      refine congrArg₂ (· + ·) ?_ (Finset.sum_congr rfl fun a _ ↦ ?_)
      · rw [Matrix.sum_apply]
        rfl
      · rw [actionDigits_top, htopsw]
        rfl)
    hDt
  exact h

end PeProof
