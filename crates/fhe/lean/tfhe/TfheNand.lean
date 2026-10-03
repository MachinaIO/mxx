import TfheStages

/-!
The bootstrapped NAND stage: the phase of its output under the LWE secret is `+Δ` or `-Δ`, by
the NAND of the input bits, plus a bounded error. The proof follows the stage: the input
combination, rounding to `Z_4096`, blind rotation, sample extraction, the switch from `Q` to `q`,
and key switching.
-/

namespace MxxFheTfhe

open Mxx.Primitives MxxRuntime

/-- The NAND lookup-table coefficient `floor(Q / 8)`. -/
abbrev lutValue : Int := 576406878024605184

/-- A packed table whose entries all equal `floor(Q / 8)` is the reduction of that constant
integer table. -/
theorem packed_lut {table : Nat}
    (h : checkBelow (fun i ↦ packedEntry 59 table i == 576406878024605184) 2048 = true) :
    (packedPolynomial (q := Q) (n := N) 59 2048 table) 0 0 =
      reducePoly Q N (intPoly fun _ ↦ lutValue) := by
  letI : Fact (1 < Q) := ⟨by decide⟩
  apply negacyclic_ext (by decide)
  intro i
  rw [packedPolynomial_coeff (by decide) (by decide), reducePoly_coeff (by decide) (by decide),
    intPoly_coeff (by decide), if_pos i.isLt]
  have hi := checkBelow_spec h i.val i.isLt
  simp only [beq_iff_eq] at hi
  rw [hi]
  rfl

theorem cast_emod_q (a : Int) : ((a % 4294967296 : Int) : ZMod q) = a := ZMod.intCast_mod a q

theorem cast_emod_Q (a : Int) : ((a % 4611255024196841473 : Int) : ZMod Q) = a :=
  ZMod.intCast_mod a Q

theorem cast_emod_4096 (a : Int) : ((a % 4096 : Int) : ZMod 4096) = a := ZMod.intCast_mod a 4096

/-- Equal residues differ by a multiple of the modulus. -/
theorem exists_of_cast_eq {m : Nat} {a b : Int} (h : (a : ZMod m) = b) : ∃ k : Int, a = b + m * k := by
  obtain ⟨k, hk⟩ := (ZMod.intCast_eq_intCast_iff_dvd_sub b a m).mp h.symm
  exact ⟨k, by linarith⟩

theorem abs_sum_le_card {ι : Type} [Fintype ι] (x : ι → Int) (b : Int) (h : ∀ i, |x i| ≤ b) :
    |∑ i, x i| ≤ Fintype.card ι * b := by
  refine (Finset.abs_sum_le_sum_abs _ _).trans ?_
  simpa using Finset.sum_le_sum fun i (_ : i ∈ Finset.univ) ↦ h i

theorem abs_binary_mul_le {s x b : Int} (hs : s = 0 ∨ s = 1) (h : |x| ≤ b) (hb : 0 ≤ b) :
    |s * x| ≤ b := by
  rcases hs with rfl | rfl <;> simpa

/-- Rounding `x q / Q` to the nearest integer moves `Q` times the result by at most `Q / 2`. -/
theorem switch_bound (x : Int) :
    |4611255024196841473 * ((x * 4294967296 + 2305627512098420736) / 4611255024196841473) -
      4294967296 * x| ≤ 2305627512098420736 := by
  rw [abs_le]
  omega

/-- The key-switching digit gadget `4^k` selected by `k = i % 8`. -/
theorem select_powers {i : Nat} {w : Int}
    (h : select ((i : Int) % 8) [1 + 0, 4 + 0, 16 + 0, 64 + 0, 256 + 0, 1024 + 0, 4096 + 0,
      16384 + 0] w) : w = 4 ^ (i % 8) := by
  obtain ⟨p, hp, rfl⟩ := h
  have hp' : i % 8 = p.val := by
    have := p.isLt
    simp only [List.length_cons, List.length_nil] at this hp
    omega
  rw [hp']
  fin_cases p <;> rfl

/-- Eight base-four digits rebuild a 16-bit value. -/
theorem digits_four {t : Int} (h0 : 0 ≤ t) (h1 : t < 65536) :
    ∑ k : Fin 8, t / 4 ^ k.val % 4 * 4 ^ k.val = t := by
  simp only [Fin.sum_univ_eight, Fin.isValue, Fin.val_zero, Fin.val_one, Fin.val_two]
  norm_num
  omega

/-- Regroup a sum over flat key-switching digits by ring coefficient. -/
theorem sum_digits_reindex (f : Fin 16384 → Int) (g : Fin 2048 → Int) :
    ∑ i : Fin 16384, f i * (g ⟨i.val / 8, by omega⟩ * 4 ^ (i.val % 8)) =
      ∑ j : Fin 2048, g j * ∑ k : Fin 8, f ⟨j.val * 8 + k.val, by omega⟩ * 4 ^ k.val := by
  rw [← (finProdFinEquiv : Fin 2048 × Fin 8 ≃ Fin 16384).sum_comp, Fintype.sum_prod_type]
  refine Finset.sum_congr rfl fun j _ ↦ ?_
  rw [Finset.mul_sum]
  refine Finset.sum_congr rfl fun k _ ↦ ?_
  have hjk : (finProdFinEquiv (j, k) : Fin 16384) = ⟨j.val * 8 + k.val, by omega⟩ := by
    apply Fin.ext
    simp only [finProdFinEquiv_apply_val]
    ring
  rw [hjk]
  have hdiv : (⟨(j.val * 8 + k.val) / 8, by omega⟩ : Fin 2048) = j := Fin.ext (by simp only; omega)
  have hmod : (j.val * 8 + k.val) % 8 = k.val := by omega
  simp only [hdiv, hmod]
  ring

/-- Key switching cancels the key masks against the output mask: only the encrypted digit
values remain. -/
theorem ks_algebra {R : Type} [CommRing R] {I J : Type} [Fintype I] [Fintype J] (b : R)
    (a : I → J → R) (s : J → R) (d c e : I → R) :
    b - ∑ i, (∑ j, a i j * s j + c i + e i) * d i - ∑ j, (0 - ∑ i, a i j * d i) * s j =
      b - ∑ i, d i * (c i + e i) := by
  have h : ∑ j, (∑ i, a i j * d i) * s j = ∑ i, (∑ j, a i j * s j) * d i := by
    simp only [Finset.sum_mul]
    rw [Finset.sum_comm]
    exact Finset.sum_congr rfl fun i _ ↦ Finset.sum_congr rfl fun j _ ↦ by ring
  simp only [zero_sub, neg_mul, Finset.sum_neg_distrib, add_mul, Finset.sum_add_distrib,
    sub_neg_eq_add, h]
  rw [show ∑ i, c i * d i = ∑ i, d i * c i from Finset.sum_congr rfl fun _ _ ↦ mul_comm _ _,
    show ∑ i, e i * d i = ∑ i, d i * e i from Finset.sum_congr rfl fun _ _ ↦ mul_comm _ _]
  simp only [mul_add, Finset.sum_add_distrib]
  ring

set_option maxHeartbeats 8000000 in
/-- The bootstrapped NAND output phase. -/
theorem nand_spec {S : Fin lweN → Int} (hS : LweSecretFacts S) {Z : ErrorPoly N}
    (hZ : ∀ c, Z.coeff c = 0 ∨ Z.coeff c = 1) {A B : Fin lweN → ExactMatrix Q N 1 16}
    (hgsw : ∀ i, GswFacts (S i) (reducePoly Q N Z) (A i) (B i))
    {kska : Fin 10321920 → Int} {kskb : Fin 16384 → Int} (hks : KeySwitchFacts S Z kska kskb)
    {m1 m2 : Int} (hm1 : m1 = 0 ∨ m1 = 1) (hm2 : m2 = 0 ∨ m2 = 1)
    {left right : (Fin lweN → Int) × Int × Unit} (hl : EncryptionFacts S m1 left)
    (hr : EncryptionFacts S m2 right) {outputs}
    (h : Stage_nand.generatedRoot FheBackend.backend { unit := () }
      (kska, left.2.1, right.2.1, left.1, right.1, A, B, kskb, ()) outputs) :
    ∃ η : Int, |η| ≤ 167780352 ∧
      ((outputs.2.1 - ∑ j : Fin lweN, outputs.1 j * S j : Int) : ZMod q) =
        (((1 - m1 * m2) * 2 - 1) * Δ + η : Int) := by
  obtain ⟨witness, _, _, _, _, _, _, _, hsum, _, hneg, hbr, hav, _, hext, _, hdig, _, houta, hbv,
    hb0, _, _, _, hksb, _, hout⟩ := h
  subst hout
  dsimp only at hsum hbr houta hksb ⊢
  obtain ⟨_, e1, he1, hb1⟩ := hl
  obtain ⟨_, e2, he2, hb2⟩ := hr
  simp only [q, Δ, Nat.cast_ofNat] at hb1 hb2
  have hw31 (i : Fin lweN) : witness.w_31_0 i = (left.1 i + right.1 i) % 4294967296 := (hsum i).2
  have hw32 (i : Fin lweN) : witness.w_32_0 i = (0 - witness.w_31_0 i) % 4294967296 := (hneg i).2
  -- The combined input phase `(3 - 2 m1 - 2 m2) Δ - e1 - e2`.
  set bp : Int := ((0 - (left.2.1 + right.2.1) % 4294967296) % 4294967296 + 536870912) %
    4294967296 with hbp
  have hbp_range : 0 ≤ bp ∧ bp < 4294967296 := by omega
  have hap_range (i : Fin lweN) : 0 ≤ witness.w_32_0 i ∧ witness.w_32_0 i < 4294967296 := by
    rw [hw32]; omega
  have hμ : ((bp - ∑ j, witness.w_32_0 j * S j : Int) : ZMod q) =
      (((3 - 2 * m1 - 2 * m2) * 536870912 - e1 - e2 : Int) : ZMod q) := by
    simp only [hbp, hw32, hw31, hb1, hb2, Int.cast_sub, Int.cast_add, Int.cast_mul, Int.cast_sum,
      cast_emod_q, Int.cast_zero]
    simp only [zero_sub, neg_add, neg_mul, add_mul, Finset.sum_neg_distrib, Finset.sum_add_distrib]
    push_cast
    ring
  -- Rounding to `Z_4096`: `2^20` times each rounded value is within `2^19` of the input.
  set rb : Int := (bp * 4096 + 2147483648) / 4294967296 with hrb
  have hrb_bound : -524288 < 1048576 * rb - bp ∧ 1048576 * rb - bp ≤ 524288 := by omega
  have hra_bound (i : Fin lweN) :
      -524288 < 1048576 * ((witness.w_32_0 i * 4096 + 2147483648) / 4294967296) - witness.w_32_0 i ∧
      1048576 * ((witness.w_32_0 i * 4096 + 2147483648) / 4294967296) - witness.w_32_0 i ≤
        524288 := by
    have := hap_range i
    omega
  -- Blind rotation.
  obtain ⟨E, hE, hphase⟩ := blind_rotation_spec hS hgsw (by omega) hbr
  rw [packed_lut ?hlut] at hphase
  case hlut => decide +kernel
  dsimp only at hphase
  -- The rotation exponent is minus the rounded phase modulo `2 N`.
  set T : Nat := ((0 - rb % 4096) % 4096).toNat + ∑ i, (S i).toNat *
    ((witness.w_32_0 i * 4096 + 2147483648) / 4294967296 % 4096).toNat with hT
  have hTint : (T : Int) = (0 - rb % 4096) % 4096 + ∑ i, S i *
      ((witness.w_32_0 i * 4096 + 2147483648) / 4294967296 % 4096) := by
    rw [hT, Nat.cast_add, Nat.cast_sum, Int.toNat_of_nonneg (by omega)]
    refine congrArg₂ (· + ·) rfl (Finset.sum_congr rfl fun i _ ↦ ?_)
    rw [Nat.cast_mul, Int.toNat_of_nonneg (by rcases hS i with h | h <;> omega),
      Int.toNat_of_nonneg (by omega)]
  set sr : Int := ∑ i, S i * ((witness.w_32_0 i * 4096 + 2147483648) / 4294967296) with hsr
  have hTmod : ((T % (2 * N) : Nat) : Int) = (-(rb - sr)) % 4096 := by
    have hcast : ((T : Int) : ZMod 4096) = ((-(rb - sr) : Int) : ZMod 4096) := by
      simp only [hTint, hsr, Int.cast_add, Int.cast_sum, Int.cast_mul, Int.cast_sub, Int.cast_neg,
        Int.cast_zero, cast_emod_4096]
      ring
    have := (ZMod.intCast_eq_intCast_iff' _ _ _).mp hcast
    simp only [N, Nat.cast_ofNat] at this ⊢
    omega
  set sw : Int := ∑ j, witness.w_32_0 j * S j with hsw
  have hρ : |1048576 * sr - sw| ≤ 630 * 524288 := by
    have : 1048576 * sr - sw = ∑ i, S i * (1048576 *
        ((witness.w_32_0 i * 4096 + 2147483648) / 4294967296) - witness.w_32_0 i) := by
      rw [hsr, hsw, Finset.mul_sum, ← Finset.sum_sub_distrib]
      exact Finset.sum_congr rfl fun i _ ↦ by ring
    rw [this]
    refine (abs_sum_le_card _ 524288 fun i ↦ abs_binary_mul_le (hS i)
      (abs_le.mpr ⟨(hra_bound i).1.le, (hra_bound i).2⟩) (by norm_num)).trans ?_
    simp [lweN]
  obtain ⟨k, hk⟩ := exists_of_cast_eq hμ
  have hsign : (if T % (2 * N) = 0 ∨ N < T % (2 * N) then lutValue else -lutValue) =
      ((1 - m1 * m2) * 2 - 1) * lutValue := by
    have hρ' := abs_le.mp hρ
    clear_value T sr sw rb bp
    simp only [q, Nat.cast_ofNat] at hk
    simp only [N] at hTmod ⊢
    rcases hm1 with rfl | rfl <;> rcases hm2 with rfl | rfl <;> split_ifs with hc <;>
      first | (exfalso; omega) | norm_num
  -- Sample extraction: the constant coefficient of the accumulator phase.
  have hα (i : Fin N) : witness.w_36_0 i = ((witness.w_35_0 0 0).coeff i).val := by
    simp only [polynomialValues, Bool.false_eq_true, if_false] at hav
    exact hav i
  have hβ (i : Fin N) : witness.w_51_0 i = ((witness.w_35_1 0 0).coeff i).val := by
    simp only [polynomialValues, Bool.false_eq_true, if_false] at hbv
    exact hbv i
  have hw52 : witness.w_52_0 = witness.w_51_0 ⟨0, by decide⟩ := by
    obtain ⟨p, hp, h52⟩ := hb0
    have hp0 : p = 0 := by exact_mod_cast hp
    rw [h52, hp0]
    rfl
  have hcoef : ((witness.w_52_0 - ∑ j : Fin N, (if j.val = 0 then 1 else -1) *
      witness.w_36_0 ⟨(N - j.val) % N, Nat.mod_lt _ (by decide)⟩ * Z.coeff j : Int) : ZMod Q) =
      ((((1 - m1 * m2) * 2 - 1) * lutValue + E.coeff ⟨0, by decide⟩ : Int) : ZMod Q) := by
    letI : Fact (1 < Q) := ⟨by decide⟩
    have hred : reducePoly Q N (canonicalLift (witness.w_35_1 0 0) -
        canonicalLift (witness.w_35_0 0 0) * Z) = reducePoly Q N
          (AdjoinRoot.root (negacyclicModulus N Int) ^ T * intPoly (fun _ ↦ lutValue) + E) := by
      rw [map_sub, map_mul, reducePoly_canonicalLift (by decide) (by decide),
        reducePoly_canonicalLift (by decide) (by decide), mul_comm (witness.w_35_0 0 0), hphase,
        map_add, map_mul, reducePoly_root_pow]
    have h0 := congrArg (fun p ↦ Negacyclic.coeff p ⟨0, by decide⟩) hred
    simp only [reducePoly_coeff (by decide : 1 < Q) (by decide : 0 < N), Negacyclic.coeff_sub,
      Negacyclic.coeff_add, coeff_zero_root_pow_mul_const_mod (by decide : 0 < N), hsign] at h0
    simp only [coeff_zero_mul (by decide : 0 < N), canonicalLift_coeff (by decide : 0 < N)] at h0
    rw [hw52, hβ]
    simp only [hα]
    exact h0
  -- Switch from `Q` to `q`.
  have hw38 (i : Fin N) : witness.w_38_0 i = ((witness.w_36_0 ⟨(N - i.val) % N,
      Nat.mod_lt _ (by decide)⟩ * (if i.val = 0 then 1 else -1)) % 4611255024196841473 *
        4294967296 + 2305627512098420736) / 4611255024196841473 % 4294967296 := by
    obtain ⟨w6, _, _, _, ⟨p, hp, hw6⟩, _, _, _, hout⟩ := hext i
    have hp' : p = ⟨(N - i.val) % N, Nat.mod_lt _ (by decide)⟩ := by
      apply Fin.ext
      have := i.isLt
      simp only [Int.ofNat_eq_natCast, N] at hp this ⊢
      omega
    subst hp'
    rw [hout, hw6]
    by_cases hi : i.val = 0 <;> simp [hi]
  set γ : Fin N → Int := fun j ↦ (witness.w_36_0 ⟨(N - j.val) % N, Nat.mod_lt _ (by decide)⟩ *
    (if j.val = 0 then 1 else -1)) % 4611255024196841473 with hγ
  set rγ : Fin N → Int := fun j ↦ (γ j * 4294967296 + 2305627512098420736) / 4611255024196841473
    with hrγ
  have hβ0 : 0 ≤ witness.w_52_0 ∧ witness.w_52_0 < 4611255024196841473 := by
    rw [hw52, hβ]
    have := ZMod.val_lt ((witness.w_35_1 0 0).coeff ⟨0, by decide⟩)
    omega
  have hcoefγ : ((witness.w_52_0 - ∑ j, γ j * Z.coeff j : Int) : ZMod Q) =
      ((((1 - m1 * m2) * 2 - 1) * lutValue + E.coeff ⟨0, by decide⟩ : Int) : ZMod Q) := by
    rw [← hcoef]
    simp only [hγ, Int.cast_sub, Int.cast_sum, Int.cast_mul, cast_emod_Q]
    exact congrArg₂ (· - ·) rfl (Finset.sum_congr rfl fun j _ ↦ by ring)
  obtain ⟨l, hl⟩ := exists_of_cast_eq hcoefγ
  set rbQ : Int := (witness.w_52_0 * 4294967296 + 2305627512098420736) / 4611255024196841473
    with hrbQ
  set P : Int := rbQ - ∑ j, rγ j * Z.coeff j - 4294967296 * l with hP
  have hPbound : |P - ((1 - m1 * m2) * 2 - 1) * 536870912| ≤ 8191 := by
    have hσ : |∑ j, (4611255024196841473 * rγ j - 4294967296 * γ j) * Z.coeff j| ≤
        2048 * 2305627512098420736 := by
      refine (abs_sum_le_card _ 2305627512098420736 fun j ↦ ?_).trans (by simp [N])
      rw [mul_comm]
      exact abs_binary_mul_le (hZ j) (switch_bound _) (by norm_num)
    have hσdef : ∑ j, (4611255024196841473 * rγ j - 4294967296 * γ j) * Z.coeff j =
        4611255024196841473 * ∑ j, rγ j * Z.coeff j - 4294967296 * ∑ j, γ j * Z.coeff j := by
      rw [Finset.mul_sum, Finset.mul_sum, ← Finset.sum_sub_distrib]
      exact Finset.sum_congr rfl fun j _ ↦ by ring
    have hE0 : |E.coeff ⟨0, by decide⟩| ≤ 630 * stepBound := by
      have := natAbs_coeff_le_of_polyNorm hE ⟨0, by decide⟩
      rw [Int.abs_eq_natAbs]
      exact_mod_cast this
    have hb := switch_bound witness.w_52_0
    rw [hσdef] at hσ
    simp only [Q, Nat.cast_ofNat, lutValue] at hl
    simp only [stepBound, N] at hE0
    clear_value P rbQ rγ γ T sr sw rb bp
    rw [abs_le] at hσ hE0 hb ⊢
    rcases hm1 with rfl | rfl <;> rcases hm2 with rfl | rfl <;> constructor <;> omega
  -- Key switching: the rounded top 16 bits of each extracted mask entry, in base-four digits.
  set t : Fin N → Int := fun j ↦ (witness.w_38_0 j + 32768) / 65536 % 65536 with ht
  have hw47 (i : Fin 16384) :
      witness.w_47_0 i = t ⟨i.val / 8, by simp only [N]; omega⟩ / 4 ^ (i.val % 8) % 4 := by
    obtain ⟨w4, w37, _, _, _, ⟨p, hp, hw4⟩, _, _, _, _, _, _, hsel, _, _, hout⟩ := hdig i
    have hp' : p = ⟨i.val / 8, by omega⟩ := by
      apply Fin.ext
      show p.val = i.val / 8
      rw [Int.ofNat_eq_natCast] at hp
      omega
    rw [hp'] at hw4
    rw [hout, hw4, select_powers hsel]
  have hw47_range (i : Fin 16384) : 0 ≤ witness.w_47_0 i ∧ witness.w_47_0 i < 4 := by
    rw [hw47]
    omega
  have hdigits (j : Fin N) :
      ∑ k : Fin 8, witness.w_47_0 ⟨j.val * 8 + k.val, by simp only [N] at *; omega⟩ * 4 ^ k.val = t j := by
    have hterm (k : Fin 8) : witness.w_47_0 ⟨j.val * 8 + k.val, by simp only [N] at *; omega⟩ =
        t j / 4 ^ k.val % 4 := by
      rw [hw47]
      have hdiv : (⟨(j.val * 8 + k.val) / 8, by simp only [N] at *; omega⟩ : Fin N) = j :=
        Fin.ext (by simp only; omega)
      have hmod : (j.val * 8 + k.val) % 8 = k.val := by omega
      simp only [hdiv, hmod]
    simp only [hterm]
    exact digits_four (by simp only [ht]; omega) (by simp only [ht]; omega)
  have hw50 (j : Fin lweN) : witness.w_50_0 j =
      (0 - ∑ i : Fin 16384, kska ⟨i.val * 630 + j.val, by simp only [lweN] at *; omega⟩ * witness.w_47_0 i) %
        4294967296 := by
    rw [(houta j).2, intMatrixVectorProduct_columns (by rfl)]
  have hw64 : witness.w_64_0 = ∑ i : Fin 16384, kskb i * witness.w_47_0 i := by
    obtain ⟨p, hp, h64⟩ := hksb
    have hp0 : p = 0 := Fin.ext (by have := p.isLt; omega)
    rw [h64, hp0, intMatrixVectorProduct_columns (by rfl)]
    exact Finset.sum_congr rfl fun i _ ↦ by simp
  choose err herr hkb using hks.2
  simp only [q, Nat.cast_ofNat] at hkb
  have hq0 : ((4294967296 : Int) : ZMod q) = 0 := by exact_mod_cast ZMod.natCast_self q
  -- Rounding each mask entry to its top 16 bits.
  set ρ : Fin N → Int := fun j ↦ 65536 * ((witness.w_38_0 j + 32768) / 65536) - witness.w_38_0 j
    with hρdef
  have hρbound (j : Fin N) : |ρ j| ≤ 32768 := by
    have : 0 ≤ witness.w_38_0 j ∧ witness.w_38_0 j < 4294967296 := by rw [hw38 j]; omega
    simp only [hρdef]
    rw [abs_le]
    omega
  have ht_cast (j : Fin N) : ((65536 * t j : Int) : ZMod q) = ((rγ j + ρ j : Int) : ZMod q) := by
    have : 65536 * t j = witness.w_38_0 j + ρ j -
        4294967296 * ((witness.w_38_0 j + 32768) / 65536 / 65536) := by
      simp only [ht, hρdef]
      omega
    rw [this, Int.cast_sub, Int.cast_mul, hq0, zero_mul, sub_zero, Int.cast_add, Int.cast_add,
      hw38 j, cast_emod_q]
  -- The output phase.
  have hreindex : ∑ i : Fin 16384, witness.w_47_0 i *
      (Z.coeff ⟨i.val / 8, by simp only [N]; omega⟩ * (4 ^ (i.val % 8) * 65536) + err i) =
      ∑ j : Fin N, Z.coeff j * (65536 * t j) + ∑ i, witness.w_47_0 i * err i := by
    have h := sum_digits_reindex witness.w_47_0 (fun j ↦ Z.coeff j)
    simp only [hdigits] at h
    calc _ = (∑ i : Fin 16384, witness.w_47_0 i *
          (Z.coeff ⟨i.val / 8, by simp only [N]; omega⟩ * 4 ^ (i.val % 8))) * 65536 +
          ∑ i, witness.w_47_0 i * err i := by
          rw [Finset.sum_mul, ← Finset.sum_add_distrib]
          exact Finset.sum_congr rfl fun i _ ↦ by ring
      _ = _ := by
          rw [h, Finset.sum_mul]
          exact congrArg₂ (· + ·) (Finset.sum_congr rfl fun j _ ↦ by ring) rfl
  rw [Int.emod_eq_of_lt hβ0.1 hβ0.2, ← hrbQ]
  refine ⟨P - ((1 - m1 * m2) * 2 - 1) * 536870912 - ∑ j, ρ j * Z.coeff j -
    ∑ i, witness.w_47_0 i * err i, ?_, ?_⟩
  · have hZρ : |∑ j, ρ j * Z.coeff j| ≤ 2048 * 32768 := by
      refine (abs_sum_le_card _ 32768 fun j ↦ ?_).trans (by simp [N])
      rw [mul_comm]
      exact abs_binary_mul_le (hZ j) (hρbound j) (by norm_num)
    have hde : |∑ i, witness.w_47_0 i * err i| ≤ 16384 * 6144 := by
      refine (abs_sum_le_card _ 6144 fun i ↦ ?_).trans (by simp)
      have he : |err i| ≤ 2048 := by
        rw [Int.abs_eq_natAbs]
        exact_mod_cast herr i
      rw [abs_mul, abs_of_nonneg (hw47_range i).1]
      exact (mul_le_mul (show witness.w_47_0 i ≤ 3 by have := (hw47_range i).2; omega) he
        (abs_nonneg _) (by norm_num)).trans (by norm_num)
    rw [abs_le] at hZρ hde hPbound ⊢
    constructor <;> omega
  · have hL : (((rbQ % 4294967296 - witness.w_64_0) % 4294967296 -
        ∑ j, witness.w_50_0 j * S j : Int) : ZMod q) = ((rbQ - ∑ i : Fin 16384, witness.w_47_0 i *
          (Z.coeff ⟨i.val / 8, by simp only [N]; omega⟩ * (4 ^ (i.val % 8) * 65536) + err i) :
            Int) : ZMod q) := by
      simp only [hw64, hw50, Int.cast_sub, Int.cast_sum, Int.cast_mul, cast_emod_q, Int.cast_zero,
        hkb, Int.cast_add]
      exact ks_algebra _ _ _ _ _ _
    have hsumt : ((∑ j, Z.coeff j * (65536 * t j) : Int) : ZMod q) =
        ∑ j, ((rγ j : Int) : ZMod q) * (Z.coeff j : ZMod q) +
          ∑ j, ((ρ j : Int) : ZMod q) * (Z.coeff j : ZMod q) := by
      rw [Int.cast_sum, ← Finset.sum_add_distrib]
      exact Finset.sum_congr rfl fun j _ ↦ by rw [Int.cast_mul, ht_cast, Int.cast_add]; ring
    have hq0' : (4294967296 : ZMod q) = 0 := by simpa using hq0
    have hPcast : ((P : Int) : ZMod q) =
        (rbQ : ZMod q) - ∑ j, ((rγ j : Int) : ZMod q) * (Z.coeff j : ZMod q) := by
      rw [hP]
      push_cast
      rw [hq0']
      ring
    rw [hL, hreindex, Int.cast_sub, Int.cast_add, hsumt]
    simp only [Δ, Nat.cast_ofNat]
    push_cast
    rw [hPcast]
    ring

end MxxFheTfhe
