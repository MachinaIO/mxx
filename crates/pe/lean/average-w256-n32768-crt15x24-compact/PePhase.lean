import PeSlots

/-!
# The phase

With the anchor encoding, the slot targets of the key, the helper encodings, and the online output
substituted, every secret-dependent term of the phase cancels except the small `σ_i(s) v_ι,i`:
`phase = e_m + ⌊Q/2⌋ m - (Σ_i σ_i(e_b k̃_i) + Σ_i σ_i(e'') v''_i - Σ_i σ_i(s) v_ι,i
  + Σ_(i,k) σ_i(e_h,k) v_(i,k) + e_F v_F)`.
The helper gadget terms cancel by re-indexing the pairs: `σ_i ∘ σ_(kHelp k) = σ_(i + idx k)`.
-/

set_option maxRecDepth 200000

namespace PeProof

open Mxx.Primitives MxxRuntime MxxExponentLut

/-- `σ_i` of slot `i` on matrices. -/
noncomputable abbrev σs {r c : Nat} (i : Fin 256) (x : ExactMatrix Q N r c) : ExactMatrix Q N r c :=
  substituteMatrix (slotAut i) (slotAut_odd i) x

/-- `σ_i` of slot `i` on polynomials. -/
noncomputable abbrev σsp (i : Fin 256) (x : ExactPoly Q N) : ExactPoly Q N :=
  substitution (slotAut i) (slotAut_odd i) x

/-- The pairing row of pair `(i, k)`. -/
def prow (i : Fin 256) (k : Fin 45) : Fin 256 := ⟨pairRow (45 * i + k), pairRow_lt _⟩

theorem sum_pairs {M : Type} [AddCommMonoid M] (F : Fin 11520 → M) :
    ∑ q, F q = ∑ i : Fin 256, ∑ k : Fin 45, F ⟨45 * i + k, by omega⟩ := by
  calc ∑ q, F q = ∑ x : Fin 256 × Fin 45, F (finProdFinEquiv x) :=
        (Equiv.sum_comp (finProdFinEquiv (m := 256) (n := 45)) F).symm
    _ = ∑ i : Fin 256, ∑ k : Fin 45, F (finProdFinEquiv (i, k)) := Fintype.sum_prod_type _
    _ = _ := by
      refine Finset.sum_congr rfl fun i _ ↦ Finset.sum_congr rfl fun k _ ↦ ?_
      congr 1
      ext
      simp only [finProdFinEquiv_apply_val]
      ring

/-- The pairing rows of member `k` are a permutation of the slots. -/
def prowEquiv (k : Fin 45) : Fin 256 ≃ Fin 256 where
  toFun i := ⟨((i : Nat) + 256 - helperIdx k) % 256, Nat.mod_lt _ (by norm_num)⟩
  invFun j := ⟨((j : Nat) + helperIdx k) % 256, Nat.mod_lt _ (by norm_num)⟩
  left_inv i := by
    have := helperIdx_lt k k.isLt
    ext
    simp only
    omega
  right_inv j := by
    have := helperIdx_lt k k.isLt
    ext
    simp only
    omega

theorem prow_eq (i : Fin 256) (k : Fin 45) : prow i k = prowEquiv k i :=
  Fin.ext (pairRow_eq i k i.isLt k.isLt)

/-- The helper gadget terms re-indexed by the pairing rows. -/
theorem helper_reindex (s : ExactPoly Q N) (X : Fin 45 → Fin 256 → ExactMatrix Q N 1 1) :
    ∑ i : Fin 256, ∑ k : Fin 45, σsp i s • X k (prow i k) =
      ∑ i : Fin 256, ∑ k : Fin 45, σsp i (substitution (kHelp k) (kHelp_odd k) s) • X k i := by
  rw [Finset.sum_comm, Finset.sum_comm (γ := Fin 256)]
  refine Finset.sum_congr rfl fun k _ ↦ ?_
  have h : ∀ i : Fin 256, σsp i s • X k (prow i k) =
      (fun j ↦ σsp j (substitution (kHelp k) (kHelp_odd k) s) • X k j) (prowEquiv k i) := by
    intro i
    simp only
    rw [← prow_eq]
    simp only [σsp]
    rw [substitution_comp (slotAut_odd (prow i k)) (kHelp_odd k) (slotAut_odd i) (slot_pair i k)]
  simp only [h]
  exact Equiv.sum_comp (prowEquiv k) (fun j ↦ σsp j (substitution (kHelp k) (kHelp_odd k) s) • X k j)

theorem subM_congr {a b : Nat} (ha : Odd a) (hb : Odd b) (h : a = b) {r c : Nat} (x : ExactMatrix Q N r c) :
    substituteMatrix a ha x = substituteMatrix b hb x := by
  subst h
  rfl

theorem slotAut_zero : slotAut (0 : Fin 256) = 1 := rfl

theorem σs_zero {r c : Nat} (x : ExactMatrix Q N r c) : σs 0 x = x := by
  have h : ∀ (k : Nat) (hk : Odd k), k = 1 → substituteMatrix k hk x = x := by
    intro k hk e
    subst e
    exact subM_one_index x
  exact h _ _ slotAut_zero

theorem σsp_zero (x : ExactPoly Q N) : σsp 0 x = x := by
  have h : ∀ (k : Nat) (hk : Odd k), k = 1 → substitution k hk x = x := by
    intro k hk e
    subst e
    exact substitution_one x
  exact h _ _ slotAut_zero

set_option maxRecDepth 200000 in
set_option maxHeartbeats 16000000 in
/-- The phase identity. -/
theorem phase_eq {model : HashModel} {key : ByteArray} {s : ExactPoly Q N}
    {cm d0 d1 Em : ExactMatrix Q N 1 1}
    {W Bb Eb : ExactMatrix Q N 1 295} {p : Fin 256 → ExactMatrix Q N 295 1}
    {Cc Bc Ec : ExactMatrix Q N 1 30} {vC : Fin 256 → ExactMatrix Q N 30 1}
    {vI : Fin 256 → ExactMatrix Q N 1 1} {v : Fin 11520 → ExactMatrix Q N 60 1}
    {AF0 CF EF : ExactMatrix Q N 1 60} {kF : ExactMatrix Q N 60 1}
    {D Eh : Fin 45 → ExactMatrix Q N 1 60}
    (hcm : cm - d1 = s • d0 + Em)
    (hW : W = s • Bb + Eb)
    (hp : ∀ i, σs i (Bb * p i) = slotResidual model key (d0 - AF0 * kF) v Bc (vC i) (vI i) i)
    (hC : Cc = s • Bc + Ec)
    (hD : ∀ k : Fin 45, D k = s • joinRow (helperKey model key (kHelp k) 0) -
      substitution (kHelp k) (kHelp_odd k) s • gZero - zeroG + Eh k)
    (hCF : CF = s • AF0 + EF) :
    cm - d1 - (∑ i, σs i (W * p i) + ∑ i, σs i Cc * vC i +
        ∑ q : Fin 11520, (substituteMatrix (slotAut (q / 45)) (slotAut_odd ⟨q / 45, by omega⟩)
          (D ⟨q % 45, by omega⟩) + zeroG) * v q + CF * kF) =
      Em - (∑ i, σs i (Eb * p i) + ∑ i, σs i Ec * vC i - ∑ i, σsp i s • vI i +
        ∑ q : Fin 11520, substituteMatrix (slotAut (q / 45)) (slotAut_odd ⟨q / 45, by omega⟩)
          (Eh ⟨q % 45, by omega⟩) * v q + EF * kF) := by
  -- The helper rows of slot `i` and member `k`.
  set H1 : Fin 45 → ExactMatrix Q N 1 60 := fun k ↦ joinRow (helperKey model key (kHelp k) 0) with hH1
  set T1 := ∑ i : Fin 256, ∑ k : Fin 45, σsp i s • (σs i (H1 k) * v ⟨45 * i + k, by omega⟩)
  set T2 := ∑ i : Fin 256, ∑ k : Fin 45,
    σsp i (substitution (kHelp k) (kHelp_odd k) s) • (gZero * v ⟨45 * i + k, by omega⟩)
  set T3 := ∑ i : Fin 256, ∑ k : Fin 45, σs i (Eh k) * v ⟨45 * i + k, by omega⟩ with hT3
  set Bt := ∑ i : Fin 256, σsp i s • (σs i Bc * vC i)
  set It := ∑ i : Fin 256, σsp i s • vI i
  -- The pair sums over `q` as double sums.
  have hq : ∀ (F : Fin 45 → ExactMatrix Q N 1 60), ∑ q : Fin 11520,
      substituteMatrix (slotAut (q / 45)) (slotAut_odd ⟨q / 45, by omega⟩) (F ⟨q % 45, by omega⟩) * v q =
      ∑ i : Fin 256, ∑ k : Fin 45, σs i (F k) * v ⟨45 * i + k, by omega⟩ := by
    intro F
    rw [sum_pairs]
    refine Finset.sum_congr rfl fun i _ ↦ Finset.sum_congr rfl fun k _ ↦ ?_
    have h1 : (45 * (i : Nat) + k) / 45 = i := by omega
    have h2 : (45 * (i : Nat) + k) % 45 = k := by omega
    simp only [h2, Fin.eta]
    rw [subM_congr _ (slotAut_odd i) (congrArg slotAut h1)]
  -- The anchor blocks.
  have SW : ∑ i, σs i (W * p i) = s • (d0 - AF0 * kF) - (T1 - T2 + Bt + It) +
      ∑ i, σs i (Eb * p i) := by
    have e1 : ∀ i, σs i (W * p i) = σsp i s • σs i (Bb * p i) + σs i (Eb * p i) := by
      intro i
      simp only [σs, σsp]
      rw [hW, Matrix.add_mul, Matrix.smul_mul, subM_add, subM_smul]
    simp only [e1, Finset.sum_add_distrib, hp]
    congr 1
    unfold slotResidual helperTerm
    simp only [smul_sub, smul_add, Finset.sum_sub_distrib, Finset.sum_add_distrib, Finset.smul_sum]
    have e0 : ∑ i : Fin 256, σsp i s • (if (i : Nat) = 0 then d0 - AF0 * kF else 0) = s • (d0 - AF0 * kF) := by
      rw [Finset.sum_eq_single (0 : Fin 256)]
      · simp only [Fin.val_zero, if_true, σsp_zero]
      · intro b _ hb
        rw [if_neg (show ¬ ((b : Nat) = 0) from fun h ↦ hb (Fin.ext h)), smul_zero]
      · simp
    have e2 : ∑ i : Fin 256, ∑ k : Fin 45, σsp i s • (gZero *
        v ⟨k + 45 * pairRow (45 * i + k), by have := pairRow_lt (45 * i + k); omega⟩) = T2 := by
      refine Eq.trans ?_ (helper_reindex s (fun k j ↦ gZero * v ⟨45 * j + k, by omega⟩))
      refine Finset.sum_congr rfl fun i _ ↦ Finset.sum_congr rfl fun k _ ↦ ?_
      simp only [prow]
      rw [show (⟨k + 45 * pairRow (45 * i + k), by have := pairRow_lt (45 * i + k); omega⟩ : Fin 11520) =
        ⟨45 * pairRow (45 * i + k) + k, by have := pairRow_lt (45 * i + k); omega⟩ from
        Fin.ext (Nat.add_comm _ _)]
    rw [e0, e2, smul_sub]
  -- The compact anchor blocks.
  have SC : ∑ i, σs i Cc * vC i = Bt + ∑ i, σs i Ec * vC i := by
    simp only [hC, σs, subM_add, subM_smul, Matrix.add_mul, Matrix.smul_mul, Finset.sum_add_distrib]
    rfl
  -- The helper blocks.
  have SH : ∑ q : Fin 11520, (substituteMatrix (slotAut (q / 45)) (slotAut_odd ⟨q / 45, by omega⟩)
      (D ⟨q % 45, by omega⟩) + zeroG) * v q = T1 - T2 + T3 := by
    rw [sum_pairs]
    have e : ∀ (i : Fin 256) (k : Fin 45),
        (substituteMatrix (slotAut ((⟨45 * i + k, by omega⟩ : Fin 11520) / 45))
          (slotAut_odd ⟨(⟨45 * i + k, by omega⟩ : Fin 11520) / 45, by omega⟩)
          (D ⟨(⟨45 * i + k, by omega⟩ : Fin 11520) % 45, by omega⟩) + zeroG) * v ⟨45 * i + k, by omega⟩ =
        σsp i s • (σs i (H1 k) * v ⟨45 * i + k, by omega⟩) -
          σsp i (substitution (kHelp k) (kHelp_odd k) s) • (gZero * v ⟨45 * i + k, by omega⟩) +
          σs i (Eh k) * v ⟨45 * i + k, by omega⟩ := by
      intro i k
      have h1 : (45 * (i : Nat) + k) / 45 = i := by omega
      have h2 : (45 * (i : Nat) + k) % 45 = k := by omega
      simp only [h2, Fin.eta]
      rw [subM_congr _ (slotAut_odd i) (congrArg slotAut h1), hD k]
      simp only [σs, σsp, subM_add, subM_sub, subM_smul, subM_gZero, subM_zeroG, Matrix.add_mul,
        Matrix.sub_mul, Matrix.smul_mul]
      abel
    simp only [e, Finset.sum_add_distrib, Finset.sum_sub_distrib]
    rfl
  rw [hcm, SW, SC, SH, hCF, Matrix.add_mul, Matrix.smul_mul, hq Eh, ← hT3, smul_sub]
  abel

end PeProof
