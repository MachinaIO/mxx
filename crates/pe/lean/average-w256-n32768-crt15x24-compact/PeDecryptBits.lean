import PeDecrypt

/-!
# The decrypted bits

The phase of `decode_phase`, with every encoding and key substituted, is `⌊Q/2⌋ m` plus an integer
error whose coefficients are bounded; under `4 (31 + decErr + 1) + 4 < Q` every bit decodes to the
message bit.
-/

set_option maxRecDepth 200000

namespace PeProof

open Mxx.Primitives MxxRuntime MxxExponentLut

theorem subM_reduce {k : Nat} (hk : Odd k) {r c : Nat} (X : ErrorMatrix N r c) :
    substituteMatrix k hk (reduceMatrix Q N r c X) = reduceMatrix Q N r c (substituteMatrix k hk X) := by
  funext i j
  simp only [substituteMatrix, reduceMatrix]
  exact (reducePoly_substitution k hk (X i j)).symm

theorem reduceMatrix_sum' {ι : Type} (s : Finset ι) {r c : Nat} (X : ι → ErrorMatrix N r c) :
    ∑ i ∈ s, reduceMatrix Q N r c (X i) = reduceMatrix Q N r c (∑ i ∈ s, X i) := by
  funext a b
  simp [reduceMatrix, Matrix.sum_apply, map_sum]

theorem reduceMatrix_add' {r c : Nat} (X Y : ErrorMatrix N r c) :
    reduceMatrix Q N r c X + reduceMatrix Q N r c Y = reduceMatrix Q N r c (X + Y) := by
  funext a b
  simp [reduceMatrix]

theorem reduceMatrix_sub' {r c : Nat} (X Y : ErrorMatrix N r c) :
    reduceMatrix Q N r c X - reduceMatrix Q N r c Y = reduceMatrix Q N r c (X - Y) := by
  funext a b
  simp [reduceMatrix]

theorem coprime_odd {k : Nat} (hk : Odd k) : N.Coprime k := by
  show Nat.Coprime (2 ^ 15) k
  exact Nat.Coprime.pow_left 15 (Nat.coprime_two_left.mpr hk)

theorem coeffBound_subst {k : Nat} (hk : Odd k) {r c : Nat} {X : ErrorMatrix N r c} {B : Nat}
    (h : CoeffBound X B) : CoeffBound (substituteMatrix k hk X) B := by
  intro a b i
  exact (coeff_natAbs_le_polyNorm _ i).trans
    ((polyNorm_substitution_le hN hk (coprime_odd hk) _).trans (coeffBound_polyNorm h a b))

/-- The bound of the decryption error besides `e_m`, for the output error bound `Bf`. -/
def decErr (Bf : Nat) : Nat :=
  256 * (295 * N * roundBound * keyCutoff) + 256 * (30 * N * companionBound * keyCutoff) +
    256 * (N * 1 * keyCutoff) + 11520 * (60 * N * 31 * keyCutoff) + 60 * N * Bf * keyCutoff

set_option maxRecDepth 200000 in
set_option maxHeartbeats 0 in
/-- Decryption: every bit of the phase of `decode_phase` is the message bit. -/
theorem decrypt_bits {model : HashModel} {key : ByteArray} {sm : ErrorPoly N} (hs : polyNorm sm ≤ 1)
    {m : Fin N → Int} (hm : ∀ k, m k = 0 ∨ m k = 1)
    {cm : ExactMatrix Q N 1 1} {em : ErrorMatrix N 1 1} (hem : CoeffBound em 31)
    (hcm : cm = reducePoly Q N sm • messageRow model key 0 + messageRow model key 1 +
      reduceMatrix Q N 1 1 em + ((halfQ : ExactPoly Q N) • (fun _ _ ↦ polynomialOfCoefficients m) :
        ExactMatrix Q N 1 1))
    {bQ BQ : ExactMatrix PQ N 1 295} {Bbar : ExactMatrix Q N 1 295}
    {Eb : ErrorMatrix N 1 295} (hEb : CoeffBound Eb roundBound)
    (hW : roundedAnchor model key bQ = reducePoly Q N sm • Bbar + reduceMatrix Q N 1 295 Eb)
    {Ec : ErrorMatrix N 1 30} (hEc : CoeffBound Ec companionBound)
    (hC : compressedAnchor model key bQ =
      reducePoly Q N sm • expand (fun _ _ ↦ BQ 0 0) + reduceMatrix Q N 1 30 Ec)
    {p' : Fin 256 → ErrorMatrix N 295 1} (hp' : ∀ i, CoeffBound (p' i) keyCutoff)
    {vC' : Fin 256 → ErrorMatrix N 30 1} (hvC : ∀ i, CoeffBound (vC' i) keyCutoff)
    {vI' : Fin 256 → ErrorMatrix N 1 1} (hvI : ∀ i, CoeffBound (vI' i) keyCutoff)
    {v' : Fin 11520 → ErrorMatrix N 60 1} (hv : ∀ q, CoeffBound (v' q) keyCutoff)
    {kF' : ErrorMatrix N 60 1} (hkF : CoeffBound kF' keyCutoff)
    {A : Fin 2 → Fin 2 → ExactMatrix Q N 1 30}
    (hp : ∀ i, σs i (Bbar * reduceMatrix Q N 295 1 (p' i)) =
      slotResidual model key (messageRow model key 0 - joinRow ![A 0 0, A 0 1] * reduceMatrix Q N 60 1 kF')
        (fun q ↦ reduceMatrix Q N 60 1 (v' q)) (expand (fun _ _ ↦ BQ 0 0))
        (reduceMatrix Q N 30 1 (vC' i)) (reduceMatrix Q N 1 1 (vI' i)) i)
    {hf0 hf1 : Fin 45 → ExactMatrix Q N 1 30}
    (hhelp : ∀ k : Fin 45, Rel sm (helperKey model key (kHelp k))
      (fun j ↦ reducePoly Q N ((((1 : Nat) : Int) : ErrorPoly N) *
        unitPair (substitution (kHelp k) (kHelp_odd k) sm) j) • g0) ![hf0 k, hf1 k] 31)
    {c0 c1 : ExactMatrix Q N 1 30} {Bf : Nat} (hout : Rel sm A (fun _ ↦ 0) ![c0, c1] Bf)
    (hT : 4 * (31 + decErr Bf + 1) + 4 < Q) :
    ∀ k : Fin N, ((2 * (((((cm - messageRow model key 1) -
      (∑ i : Fin 256, σs i (roundedAnchor model key bQ * reduceMatrix Q N 295 1 (p' i)) +
        ∑ i : Fin 256, σs i (compressedAnchor model key bQ) * reduceMatrix Q N 30 1 (vC' i) +
        ∑ q : Fin 11520, helperValue model key hf0 hf1 q * reduceMatrix Q N 60 1 (v' q) +
        (joinRow ![c0, c1] - joinRow ![A 1 0, A 1 1]) * reduceMatrix Q N 60 1 kF')) 0 0).coeff k).val : Int) +
      (Q : Int) / 2) / (Q : Int)) % 2 = m k := by
  set s := reducePoly Q N sm with hsdef
  choose eh heh hD using fun k : Fin 45 ↦ helper_form (kHelp_odd k) (hhelp k)
  obtain ⟨eF, heF, hCF⟩ := out_form hout
  have hcm' : cm - messageRow model key 1 = s • messageRow model key 0 +
      (reduceMatrix Q N 1 1 em + ((halfQ : ExactPoly Q N) • (fun _ _ ↦ polynomialOfCoefficients m) :
        ExactMatrix Q N 1 1)) := by
    rw [hcm]
    abel
  have hCF' : joinRow ![c0, c1] - joinRow ![A 1 0, A 1 1] =
      s • joinRow ![A 0 0, A 0 1] + reduceMatrix Q N 1 60 (jZ eF) := by
    rw [joinRow_pair (A 1), joinRow_pair (A 0)]
    exact hCF
  have hpe := phase_eq (model := model) (key := key) (s := s)
    (D := fun k ↦ joinRow ![hf0 k, hf1 k] - joinRow (helperKey model key (kHelp k) 1))
    (Eh := fun k ↦ reduceMatrix Q N 1 60 (jZ (eh k)))
    hcm' hW hp hC (fun k ↦ hD k) hCF'
  refine (congrArg (fun P : ExactMatrix Q N 1 1 ↦ ∀ k : Fin N,
    ((2 * (((P 0 0).coeff k).val : Int) + (Q : Int) / 2) / (Q : Int)) % 2 = m k) hpe).mpr ?_
  -- The integer error.
  set mM : ErrorMatrix N 1 1 := fun _ _ ↦ intPoly m with hmM
  set Err : ErrorMatrix N 1 1 :=
    ∑ i : Fin 256, substituteMatrix (slotAut i) (slotAut_odd i) (Eb * p' i) +
      ∑ i : Fin 256, substituteMatrix (slotAut i) (slotAut_odd i) Ec * vC' i -
      ∑ i : Fin 256, substitution (slotAut i) (slotAut_odd i) sm • vI' i +
      ∑ q : Fin 11520, substituteMatrix (slotAut (q / 45)) (slotAut_odd ⟨q / 45, by omega⟩)
        (jZ (eh ⟨q % 45, by omega⟩)) * v' q +
      jZ eF * kF' with hErr
  have e1 : ∀ i : Fin 256, σs i (reduceMatrix Q N 1 295 Eb * reduceMatrix Q N 295 1 (p' i)) =
      reduceMatrix Q N 1 1 (substituteMatrix (slotAut i) (slotAut_odd i) (Eb * p' i)) := by
    intro i
    rw [← reduceMatrix_mul]
    exact subM_reduce _ _
  have e2 : ∀ i : Fin 256, σs i (reduceMatrix Q N 1 30 Ec) * reduceMatrix Q N 30 1 (vC' i) =
      reduceMatrix Q N 1 1 (substituteMatrix (slotAut i) (slotAut_odd i) Ec * vC' i) := by
    intro i
    rw [show σs i (reduceMatrix Q N 1 30 Ec) =
      reduceMatrix Q N 1 30 (substituteMatrix (slotAut i) (slotAut_odd i) Ec) from subM_reduce _ _,
      ← reduceMatrix_mul]
  have e3 : ∀ i : Fin 256, σsp i s • reduceMatrix Q N 1 1 (vI' i) =
      reduceMatrix Q N 1 1 (substitution (slotAut i) (slotAut_odd i) sm • vI' i) := by
    intro i
    rw [reduceMatrix_smul, hsdef, reducePoly_substitution]
  have e4 : ∀ q : Fin 11520, substituteMatrix (slotAut (q / 45)) (slotAut_odd ⟨q / 45, by omega⟩)
      (reduceMatrix Q N 1 60 (jZ (eh ⟨q % 45, by omega⟩))) * reduceMatrix Q N 60 1 (v' q) =
      reduceMatrix Q N 1 1 (substituteMatrix (slotAut (q / 45)) (slotAut_odd ⟨q / 45, by omega⟩)
        (jZ (eh ⟨q % 45, by omega⟩)) * v' q) := by
    intro q
    rw [subM_reduce, ← reduceMatrix_mul]
  have e5 : reduceMatrix Q N 1 60 (jZ eF) * reduceMatrix Q N 60 1 kF' = reduceMatrix Q N 1 1 (jZ eF * kF') :=
    (reduceMatrix_mul _ _ _ _ _ _ _).symm
  have eEm : reduceMatrix Q N 1 1 em + ((halfQ : ExactPoly Q N) • (fun _ _ ↦ polynomialOfCoefficients m) :
      ExactMatrix Q N 1 1) =
      reduceMatrix Q N 1 1 (em + (halfQ : ErrorPoly N) • mM) := by
    funext a b
    simp only [reduceMatrix, Matrix.add_apply, Matrix.smul_apply, smul_eq_mul, map_add, map_mul,
      map_intCast, polynomialOfCoefficients_eq_reduce]
    rfl
  have hred : reduceMatrix Q N 1 1 em + ((halfQ : ExactPoly Q N) • (fun _ _ ↦ polynomialOfCoefficients m) :
      ExactMatrix Q N 1 1) -
      (∑ i, σs i (reduceMatrix Q N 1 295 Eb * reduceMatrix Q N 295 1 (p' i)) +
        ∑ i, σs i (reduceMatrix Q N 1 30 Ec) * reduceMatrix Q N 30 1 (vC' i) -
        ∑ i, σsp i s • reduceMatrix Q N 1 1 (vI' i) +
        ∑ q : Fin 11520, substituteMatrix (slotAut (q / 45)) (slotAut_odd ⟨q / 45, by omega⟩)
          (reduceMatrix Q N 1 60 (jZ (eh ⟨q % 45, by omega⟩))) * reduceMatrix Q N 60 1 (v' q) +
        reduceMatrix Q N 1 60 (jZ eF) * reduceMatrix Q N 60 1 kF') =
      reduceMatrix Q N 1 1 (em + (halfQ : ErrorPoly N) • mM - Err) := by
    simp only [e1, e2, e3, e4, e5, eEm, reduceMatrix_sum', reduceMatrix_add', reduceMatrix_sub']
    rfl
  rw [hred]
  -- The bound of the integer error.
  have hErrB : CoeffBound Err (decErr Bf) := by
    have b1 := coeffBound_sum (Finset.univ : Finset (Fin 256))
      (fun i _ ↦ coeffBound_subst (slotAut_odd i) (coeffBound_mul hN hEb (hp' i)))
    have b2 := coeffBound_sum (Finset.univ : Finset (Fin 256))
      (fun i _ ↦ coeffBound_mul hN (coeffBound_subst (slotAut_odd i) hEc) (hvC i))
    have b3 := coeffBound_sum (Finset.univ : Finset (Fin 256))
      (fun i _ ↦ coeffBound_smul hN ((polyNorm_substitution_le hN (slotAut_odd i)
        (coprime_odd (slotAut_odd i)) sm).trans hs) (hvI i))
    have b4 := coeffBound_sum (Finset.univ : Finset (Fin 11520))
      (fun q _ ↦ coeffBound_mul hN (coeffBound_subst (slotAut_odd ⟨q / 45, by omega⟩)
        (jZ_bound (heh ⟨q % 45, by omega⟩))) (hv q))
    have b5 := coeffBound_mul hN (jZ_bound heF) hkF
    refine coeffBound_mono (coeffBound_add (coeffBound_add (coeffBound_sub (coeffBound_add b1 b2) b3) b4) b5)
      (le_of_eq ?_)
    simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, smul_eq_mul, decErr]
  intro k
  rw [show (reduceMatrix Q N 1 1 (em + (halfQ : ErrorPoly N) • mM - Err)) 0 0 =
    reducePoly Q N (em 0 0 + (halfQ : ErrorPoly N) * intPoly m - Err 0 0) from rfl, reducePoly_coeff hQ hN]
  have hc : (em 0 0 + (halfQ : ErrorPoly N) * intPoly m - Err 0 0).coeff k =
      (em 0 0).coeff k + halfQ * m k - (Err 0 0).coeff k := by
    rw [Negacyclic.coeff_sub, Negacyclic.coeff_add, intCast_eq_algebraMap, Negacyclic.coeff_smul,
      intPoly_coeff hN]
    rfl
  rw [hc]
  set a := (em 0 0).coeff k
  set b := (Err 0 0).coeff k
  have ha : |a| ≤ 31 := by
    have := hem 0 0 k
    rw [Int.abs_eq_natAbs]
    exact_mod_cast this
  have hb : |b| ≤ (decErr Bf : Int) := by
    have := hErrB 0 0 k
    rw [Int.abs_eq_natAbs]
    exact_mod_cast this
  have hq2 : ((Q / 2 : Nat) : Int) = halfQ := by norm_num
  have hcast : ((a + halfQ * m k - b : Int) : ZMod Q) =
      ((-(((Q / 2 : Nat) : Int) * m k) + (a - b - m k) : Int) : ZMod Q) := by
    rw [ZMod.intCast_eq_intCast_iff_dvd_sub]
    refine ⟨-(m k), ?_⟩
    rw [hq2]
    norm_num
    ring
  rw [hcast]
  apply threshold_decode_value hQ hQodd (hm k)
  have hmk := hm k
  have hmb : |m k| ≤ 1 := by rcases hmk with h | h <;> rw [h] <;> norm_num
  have habs : |a - b - m k| ≤ 31 + (decErr Bf : Int) + 1 := by
    have h1 := abs_sub (a - b) (m k)
    have h2 := abs_sub a b
    linarith only [h1, h2, ha, hb, hmb]
  have hT' : ((4 * (31 + decErr Bf + 1) + 4 : Nat) : Int) < (Q : Int) := Int.ofNat_lt.mpr hT
  have hc4 : ((4 * (31 + decErr Bf + 1) + 4 : Nat) : Int) = 4 * (31 + (decErr Bf : Int) + 1) + 4 := by
    push_cast
    ring
  rw [hc4] at hT'
  linarith only [habs, hT']

end PeProof
