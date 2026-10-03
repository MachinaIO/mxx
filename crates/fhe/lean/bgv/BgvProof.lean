import Claim
import BgvStages
import Mathlib.Tactic.NormNum.Prime

/-!
The generated BGV round-trip claim: the decryption phase is the integer polynomial `W'` that
the stage lemmas determine, its coefficients stay below the decoder radius, and the decoded
slots are the slotwise products of the two encrypted inputs.
-/

namespace MxxFheBgv

open Mxx.Primitives MxxRuntime GeneratedClaim


theorem polyNorm_sub_le (x y : ErrorPoly n) : polyNorm (x - y) ≤ polyNorm x + polyNorm y := by
  rw [sub_eq_add_neg]
  exact (polyNorm_add_le _ _).trans (by rw [polyNorm_neg])

/-- The fresh-encryption phase `c1 - s c0 = t (e u + e2 - s e1) + m~` over the integers. -/
theorem fresh_phase {A : ExactPoly Q n} {S E U E1 E2 M : ErrorPoly n} {c0 c1 : ExactPoly Q n}
    (hc0 : c0 = A * reducePoly Q n U + ((t : Int) : ExactPoly Q n) * reducePoly Q n E1)
    (hc1 : c1 = (reducePoly Q n S * A + ((t : Int) : ExactPoly Q n) * reducePoly Q n E) *
      reducePoly Q n U + ((t : Int) : ExactPoly Q n) * reducePoly Q n E2 + reducePoly Q n M) :
    c1 - reducePoly Q n S * c0 =
      reducePoly Q n (((t : Int) : ErrorPoly n) * (E * U + E2 - S * E1) + M) := by
  subst hc0 hc1
  simp only [map_add, map_mul, map_sub, reducePoly_intCast]
  ring

/-- The static bound on a fresh phase. -/
theorem fresh_phase_norm {S E U E1 E2 M : ErrorPoly n} (hS : polyNorm S ≤ 1)
    (hE : polyNorm E ≤ 20) (hU : polyNorm U ≤ 1) (hE1 : polyNorm E1 ≤ 20)
    (hE2 : polyNorm E2 ≤ 20) (hM : polyNorm M ≤ t / 2) :
    polyNorm (((t : Int) : ErrorPoly n) * (E * U + E2 - S * E1) + M) ≤
      t * (n * 20 * 1 + 20 + n * 1 * 20) + t / 2 := by
  have hEU := (polyNorm_mul_le hn E U).trans (Nat.mul_le_mul (Nat.mul_le_mul_left _ hE) hU)
  have hSE := (polyNorm_mul_le hn S E1).trans (Nat.mul_le_mul (Nat.mul_le_mul_left _ hS) hE1)
  have hinner : polyNorm (E * U + E2 - S * E1) ≤ n * 20 * 1 + 20 + n * 1 * 20 :=
    (polyNorm_sub_le _ _).trans (Nat.add_le_add ((polyNorm_add_le _ _).trans
      (Nat.add_le_add hEU hE2)) hSE)
  refine (polyNorm_add_le _ _).trans (Nat.add_le_add ?_ hM)
  refine (polyNorm_intCast_mul_le _ _).trans ?_
  exact Nat.mul_le_mul_left _ hinner


theorem gadget_eq : gadget 0 = (P : Int) * ((q2 * q3 : Nat) : Int) ∧
    gadget 1 = (P : Int) * ((q1 * q3 : Nat) : Int) ∧ gadget 2 = (P : Int) * ((q1 * q2 : Nat) : Int) := by
  refine ⟨?_, ?_, ?_⟩ <;> simp [gadget]

theorem intCast_errorPoly_mul (a b : Int) :
    (((a * b : Int)) : ErrorPoly n) = (a : ErrorPoly n) * (b : ErrorPoly n) := by push_cast; ring

/-- Relinearization moves `s^2 d0` into the pair and adds `t X` for a key-switching error `X`
with `P X = sum_j e_j d_j + c1 - s c0`. -/
theorem relin_phase {S D : ErrorPoly n} {Erk : Fin 3 → ErrorPoly n} {rk : ExactMatrix QP n 2 3}
    {digits : Fin 3 → ErrorPoly n} {C K : Fin 2 → ErrorPoly n} {d0 : ExactPoly Q n}
    (hD : ∀ i, D.coeff i ≡ (S * S).coeff i [ZMOD Q])
    (hrk : ∀ j : Fin 3, rk 1 j = reducePoly QP n S * rk 0 j +
      ((t : Int) : ExactPoly QP n) * reducePoly QP n (Erk j) +
      reducePoly QP n D * ((gadget j : Int) : ExactPoly QP n))
    (hcrt : ∀ i, ((q2 * q3 : Nat) : Int) * (digits 0).coeff i +
      ((q1 * q3 : Nat) : Int) * (digits 1).coeff i + ((q1 * q2 : Nat) : Int) * (digits 2).coeff i ≡
        (canonicalLift d0).coeff i [ZMOD Q])
    (hK : ∀ r : Fin 2, ((P : Int) : ErrorPoly n) * K r =
      canonicalLift (∑ j : Fin 3, rk r j * reducePoly QP n (digits j)) +
        ((t : Int) : ErrorPoly n) * C r) :
    ∃ X : ErrorPoly n, ((P : Int) : ErrorPoly n) * X =
        Erk 0 * digits 0 + Erk 1 * digits 1 + Erk 2 * digits 2 + C 1 - S * C 0 ∧
      reducePoly Q n (K 1) - reducePoly Q n S * reducePoly Q n (K 0) =
        reducePoly Q n (S * S) * d0 + ((t : Int) : ExactPoly Q n) * reducePoly Q n X := by
  have hA (r : Fin 2) : reducePoly QP n (canonicalLift (∑ j : Fin 3, rk r j *
      reducePoly QP n (digits j))) = ∑ j : Fin 3, rk r j * reducePoly QP n (digits j) :=
    reducePoly_canonicalLift (by decide) hn _
  obtain ⟨g0, g1, g2⟩ := gadget_eq
  -- The CRT-normalized digits recombine to the leading component modulo `Q`.
  have hM : reducePoly Q n (((q2 * q3 : Nat) : Int) * digits 0 + ((q1 * q3 : Nat) : Int) * digits 1 +
      ((q1 * q2 : Nat) : Int) * digits 2) = reducePoly Q n (canonicalLift d0) := by
    rw [reducePoly_eq_iff (by decide) hn]
    intro i
    simp only [Negacyclic.coeff_add, intCast_mul_coeff]
    exact hcrt i
  obtain ⟨Y, hY⟩ := exists_of_reducePoly_eq (q := Q) (by decide) hn hM
  -- Over `Q P`, the key rows leave only the encrypted error and the scaled target.
  have hdiff : reducePoly QP n (canonicalLift (∑ j : Fin 3, rk 1 j * reducePoly QP n (digits j)) -
      S * canonicalLift (∑ j : Fin 3, rk 0 j * reducePoly QP n (digits j))) = reducePoly QP n
      (((t : Int) : ErrorPoly n) * (Erk 0 * digits 0 + Erk 1 * digits 1 + Erk 2 * digits 2) +
        ((P : Int) : ErrorPoly n) * (D * (((q2 * q3 : Nat) : Int) * digits 0 +
          ((q1 * q3 : Nat) : Int) * digits 1 + ((q1 * q2 : Nat) : Int) * digits 2))) := by
    rw [map_sub, map_mul, hA, hA]
    simp only [Fin.sum_univ_three, hrk, map_add, map_mul, reducePoly_intCast, g0, g1, g2]
    push_cast
    ring
  obtain ⟨Z, hZ⟩ := exists_of_reducePoly_eq (q := QP) (by decide) hn hdiff
  have hQP : ((QP : Int) : ErrorPoly n) = ((Q : Int) : ErrorPoly n) * ((P : Int) : ErrorPoly n) := by
    rw [← intCast_errorPoly_mul]; norm_num
  have key : ((t : Int) : ErrorPoly n) *
      (Erk 0 * digits 0 + Erk 1 * digits 1 + Erk 2 * digits 2 + C 1 - S * C 0) =
      ((P : Int) : ErrorPoly n) * (K 1 - S * K 0 - D * canonicalLift d0 -
        ((Q : Int) : ErrorPoly n) * (D * Y) - ((Q : Int) : ErrorPoly n) * Z) := by
    rw [hQP] at hZ
    linear_combination -(hK 1) + S * (hK 0) - hZ - ((P : Int) : ErrorPoly n) * D * hY
  obtain ⟨X, hX⟩ := exists_of_coprime_mul hn (by decide) key
  refine ⟨X, hX.symm, ?_⟩
  have hcancel := intCast_mul_cancel hn (by decide : ((P : Int)) ≠ 0)
    (show ((P : Int) : ErrorPoly n) * (((t : Int) : ErrorPoly n) * X) =
      ((P : Int) : ErrorPoly n) * (K 1 - S * K 0 - D * canonicalLift d0 -
        ((Q : Int) : ErrorPoly n) * (D * Y) - ((Q : Int) : ErrorPoly n) * Z) by
      rw [← key, hX]; ring)
  have hred := congrArg (reducePoly Q n) hcancel
  have hDS : reducePoly Q n D = reducePoly Q n (S * S) :=
    (reducePoly_eq_iff (by decide) hn _ _).mpr hD
  have hQ0 : reducePoly Q n ((Q : Int) : ErrorPoly n) = 0 := by
    have := reducePoly_modulus_mul (n := n) Q 1
    rwa [mul_one] at this
  simp only [map_sub, map_mul] at hred
  rw [hQ0, hDS, reducePoly_canonicalLift (q := Q) (by decide) hn, reducePoly_intCast] at hred
  rw [map_mul] at hred ⊢
  linear_combination -hred


/-- Exact division by a positive integer divides the norm. -/
theorem polyNorm_le_div_of_mul {c : Nat} (hc : 0 < c) {X Y : ErrorPoly n} {bound : Nat}
    (h : ((c : Int) : ErrorPoly n) * X = Y) (hY : polyNorm Y ≤ bound) :
    polyNorm X ≤ bound / c := by
  apply polyNorm_le_of_coeff
  intro i
  rw [Nat.le_div_iff_mul_le hc]
  have hcoeff := congrArg (fun z : ErrorPoly n ↦ z.coeff i) h
  simp only [intCast_mul_coeff] at hcoeff
  have := natAbs_coeff_le_of_polyNorm hY i
  rw [← hcoeff, Int.natAbs_mul] at this
  simpa [mul_comm] using this

theorem relin_error_norm {S X : ErrorPoly n} {Erk digits : Fin 3 → ErrorPoly n}
    {C : Fin 2 → ErrorPoly n} (hS : polyNorm S ≤ 1) (hErk : ∀ j, polyNorm (Erk j) ≤ 20)
    (hδ0 : polyNorm (digits 0) ≤ q1 / 2) (hδ1 : polyNorm (digits 1) ≤ q2 / 2)
    (hδ2 : polyNorm (digits 2) ≤ q3 / 2) (hC : ∀ r, polyNorm (C r) ≤ P / 2)
    (hX : ((P : Int) : ErrorPoly n) * X =
      Erk 0 * digits 0 + Erk 1 * digits 1 + Erk 2 * digits 2 + C 1 - S * C 0) :
    polyNorm X ≤ (n * 20 * (q1 / 2) + n * 20 * (q2 / 2) + n * 20 * (q3 / 2) + P / 2 +
      n * 1 * (P / 2)) / P := by
  apply polyNorm_le_div_of_mul (by decide) hX
  have m (a b : ErrorPoly n) {x y : Nat} (ha : polyNorm a ≤ x) (hb : polyNorm b ≤ y) :
      polyNorm (a * b) ≤ n * x * y :=
    (polyNorm_mul_le hn a b).trans (Nat.mul_le_mul (Nat.mul_le_mul_left _ ha) hb)
  exact (polyNorm_sub_le _ _).trans (Nat.add_le_add ((polyNorm_add_le _ _).trans
    (Nat.add_le_add ((polyNorm_add_le _ _).trans (Nat.add_le_add ((polyNorm_add_le _ _).trans
      (Nat.add_le_add (m _ _ (hErk 0) hδ0) (m _ _ (hErk 1) hδ1))) (m _ _ (hErk 2) hδ2)))
      (hC 1))) (m _ _ hS (hC 0)))

/-- Modulus switching divides the phase by `q3` exactly after its `t`-correction. -/
theorem modswitch_phase {S W : ErrorPoly n} {c : ExactMatrix Q n 2 1}
    {C' K' : Fin 2 → ErrorPoly n}
    (hW : c 1 0 - reducePoly Q n S * c 0 0 = reducePoly Q n W)
    (hK' : ∀ r : Fin 2, ((q3 : Int) : ErrorPoly n) * K' r =
      canonicalLift (c r 0) + ((t : Int) : ErrorPoly n) * C' r) :
    ∃ W' : ErrorPoly n, ((q3 : Int) : ErrorPoly n) * W' =
        W + ((t : Int) : ErrorPoly n) * (C' 1 - S * C' 0) ∧
      reducePoly QL n (K' 1) - reducePoly QL n S * reducePoly QL n (K' 0) = reducePoly QL n W' := by
  have hlift : reducePoly Q n (canonicalLift (c 1 0) - S * canonicalLift (c 0 0)) =
      reducePoly Q n W := by
    rw [map_sub, map_mul, reducePoly_canonicalLift (by decide) hn,
      reducePoly_canonicalLift (by decide) hn, hW]
  obtain ⟨Z, hZ⟩ := exists_of_reducePoly_eq (q := Q) (by decide) hn hlift
  have hQ : ((Q : Int) : ErrorPoly n) = ((q3 : Int) : ErrorPoly n) * ((QL : Int) : ErrorPoly n) := by
    rw [← intCast_errorPoly_mul]; norm_num
  refine ⟨K' 1 - S * K' 0 - ((QL : Int) : ErrorPoly n) * Z, ?_, ?_⟩
  · rw [hQ] at hZ
    linear_combination (hK' 1) - S * (hK' 0) + hZ
  · rw [map_sub, map_mul (reducePoly QL n) ((QL : Int) : ErrorPoly n), reducePoly_intCast,
      show ((QL : Int) : ExactPoly QL n) = 0 by
        have := reducePoly_modulus_mul (n := n) QL 1
        rwa [mul_one, reducePoly_intCast] at this, map_sub, map_mul]
    ring


set_option maxRecDepth 100000 in
theorem tables_consistent (k : Nat) (hk : k < n) :
    packedEntry 13 BgvTables.encode (packedEntry 13 BgvTables.decode k) = k := by
  have := checkBelow_spec (test := fun k ↦
    packedEntry 13 BgvTables.encode (packedEntry 13 BgvTables.decode k) == k)
    (bound := 8192) (by decide +kernel) k hk
  simpa using this

set_option maxRecDepth 100000 in
theorem bitReverse_involutive (k : Nat) (hk : k < n) :
    nttBitReverse 13 (nttBitReverse 13 k) = k ∧ nttBitReverse 13 k < n := by
  have := checkBelow_spec (test := fun k ↦
    bitReverseFast 13 (bitReverseFast 13 k 0) 0 == k && decide (bitReverseFast 13 k 0 < 8192))
    (bound := 8192) (by decide +kernel) k hk
  simp only [bitReverseFast_eq, zero_mul, zero_add, Bool.and_eq_true, beq_iff_eq,
    decide_eq_true_eq] at this
  exact this

theorem packedEntry_lt (table index : Nat) : packedEntry 13 table index < n :=
  Nat.mod_lt _ (by decide)

theorem bits_eq {bits : Nat} (h : n = 2 ^ bits) : bits = 13 :=
  Nat.pow_right_injective (le_refl 2) (h.symm.trans (by decide))

theorem zmod_val_cast {x : ZMod t} : (((x.val : Nat) : Int) : ZMod t) = x := by
  rw [Int.cast_natCast, ZMod.natCast_zmod_val]

/-- The decoded slots are the slotwise products of the two encoded inputs. -/
theorem slots_correct {x y : Fin n → Int} {mx my plain : ExactMatrix t n 1 1}
    {nx ny nout out ideal : Fin n → Int}
    (hfx : polynomialFromValues true nx mx)
    (htx : ∀ j : Fin n, ∃ k : Fin n, k.val = packedEntry 13 BgvTables.encode j.val ∧ nx j = x k)
    (hfy : polynomialFromValues true ny my)
    (hty : ∀ j : Fin n, ∃ k : Fin n, k.val = packedEntry 13 BgvTables.encode j.val ∧ ny j = y k)
    (hplain : plain 0 0 = mx 0 0 * my 0 0) (hvals : polynomialValues true plain nout)
    (hslots : ∀ i : Fin n, ∃ k : Fin n, k.val = packedEntry 13 BgvTables.decode i.val ∧
      out i = nout k)
    (hideal : ∀ i, ideal i = x i * y i % t) : out = ideal := by
  haveI : Fact (Nat.Prime t) := ⟨by norm_num⟩
  simp only [polynomialValues, polynomialFromValues, if_true] at hvals hfx hfy
  obtain ⟨hprim, hcanon, bits, hbits, hev⟩ := polynomialNttRuns_eval hvals
  obtain ⟨_, _, bitsx, hbitsx, hevx⟩ := polynomialNttRuns_eval hfx
  obtain ⟨_, _, bitsy, hbitsy, hevy⟩ := polynomialNttRuns_eval hfy
  rw [bits_eq hbits] at hev
  rw [bits_eq hbitsx] at hevx
  rw [bits_eq hbitsy] at hevy
  have hroot := nttPrimitiveRoot_pow_n hn hprim
  funext i
  obtain ⟨k, hk, hout⟩ := hslots i
  obtain ⟨hinv, hlt⟩ := bitReverse_involutive k.val k.isLt
  let index : Fin n := ⟨nttBitReverse 13 k.val, hlt⟩
  have hslot {slot : Fin n} (h : slot.val = nttBitReverse 13 index.val) : slot = k :=
    Fin.ext (h.trans hinv)
  obtain ⟨s0, hs0, hv0⟩ := hev index
  obtain ⟨s1, hs1, hv1⟩ := hevx index
  obtain ⟨s2, hs2, hv2⟩ := hevy index
  rw [hslot hs0] at hv0
  rw [hslot hs1] at hv1
  rw [hslot hs2] at hv2
  simp only [Bool.false_eq_true, if_false, if_true, zmod_val_cast] at hv0 hv1 hv2
  change (nout k : ZMod t) = oddPowerSum _ index.val (plain 0 0) at hv0
  change (nx k : ZMod t) = oddPowerSum _ index.val (mx 0 0) at hv1
  change (ny k : ZMod t) = oddPowerSum _ index.val (my 0 0) at hv2
  rw [hplain, oddPowerSum_mul hn hroot, ← hv1, ← hv2] at hv0
  obtain ⟨kx, hkx, hnx⟩ := htx k
  obtain ⟨ky, hky, hny⟩ := hty k
  have hidx : packedEntry 13 BgvTables.encode k.val = i.val := by
    rw [hk]; exact tables_consistent i.val i.isLt
  have hkxi : kx = i := Fin.ext (hkx.trans hidx)
  have hkyi : ky = i := Fin.ext (hky.trans hidx)
  rw [hnx, hny, hkxi, hkyi] at hv0
  rw [hout, hideal i]
  have hcast : ((nout k : Int) : ZMod t) = ((x i * y i : Int) : ZMod t) := by
    push_cast; exact hv0
  have hmod := (ZMod.intCast_eq_intCast_iff' _ _ t).mp hcast
  rw [← hmod]
  exact (Int.emod_eq_of_lt (hcanon k).1 (hcanon k).2).symm

theorem correctness : CorrectnessClaim := by
  intro hashModel external execution hruns
  obtain ⟨hvalid, hk, hx, hy, hm, hr, hs, hd, hi⟩ := hruns
  obtain ⟨S, E, Erk, D, hS, hE, hErk, hD, hsk, hpk, hrk⟩ := keygen_spec hk
  obtain ⟨Ux, E1x, E2x, mx, nx, hUx, hE1x, hE2x, hfx, htx, hcx0, hcx1⟩ := encrypt_x_spec hx
  obtain ⟨Uy, E1y, E2y, my, ny, hUy, hE1y, hE2y, hfy, hty, hcy0, hcy1⟩ := encrypt_y_spec hy
  obtain ⟨hd0, hd1, hd2⟩ := multiply_spec hm
  obtain ⟨digits, C, K, hδ0, hδ1, hδ2, hcrt, hC, hK, hr0, hr1⟩ := relinearize_spec hr
  obtain ⟨C', K', hC', hK', hms⟩ := modswitch_spec hs
  obtain ⟨hphase, plain, nout, hplain, hvals, hslots⟩ := decrypt_spec hsk hd
  have hideal := ideal_spec hi
  -- Fresh phases over the integers.
  set Mx := centeredIntLift (mx 0 0)
  set My := centeredIntLift (my 0 0)
  set Vx := ((t : Int) : ErrorPoly n) * (E * Ux + E2x - S * E1x) + Mx
  set Vy := ((t : Int) : ErrorPoly n) * (E * Uy + E2y - S * E1y) + My
  have phx := fresh_phase (S := S) (E := E) (E2 := E2x) (M := Mx)
    (c1 := execution.«stage_1» 1 0) hcx0 (by rw [hcx1, hpk])
  have phy := fresh_phase (S := S) (E := E) (E2 := E2y) (M := My)
    (c1 := execution.«stage_2» 1 0) hcy0 (by rw [hcy1, hpk])
  -- Relinearization keeps the product phase and adds the key-switching error.
  obtain ⟨X, hX, hrel⟩ := relin_phase hD hrk hcrt hK
  set W := Vx * Vy + ((t : Int) : ErrorPoly n) * X
  have phrel : execution.«stage_4» 1 0 - reducePoly Q n S * execution.«stage_4» 0 0 =
      reducePoly Q n W := by
    rw [hr0, hr1, hd1, hd2]
    rw [hd0] at hrel
    simp only [W, map_add, map_mul, reducePoly_intCast] at hrel ⊢
    linear_combination hrel + (execution.«stage_2» 1 0 - reducePoly Q n S * execution.«stage_2» 0 0) *
      phx + reducePoly Q n Vx * phy
  obtain ⟨W', hW', hphaseL⟩ := modswitch_phase phrel hK'
  have hfinal : execution.«stage_6».1 0 0 = reducePoly QL n W' := by
    rw [hphase, hms 0, hms 1, hphaseL]
  -- The static bound on the final phase.
  have hVx := fresh_phase_norm hS hE hUx hE1x hE2x (polyNorm_centeredIntLift (by decide) (mx 0 0))
  have hVy := fresh_phase_norm hS hE hUy hE1y hE2y (polyNorm_centeredIntLift (by decide) (my 0 0))
  have hXn := relin_error_norm hS hErk hδ0 hδ1 hδ2 hC hX
  have hWn : polyNorm W ≤ n * (t * (n * 20 * 1 + 20 + n * 1 * 20) + t / 2) *
      (t * (n * 20 * 1 + 20 + n * 1 * 20) + t / 2) + t * ((n * 20 * (q1 / 2) + n * 20 * (q2 / 2) +
        n * 20 * (q3 / 2) + P / 2 + n * 1 * (P / 2)) / P) :=
    (polyNorm_add_le _ _).trans (Nat.add_le_add ((polyNorm_mul_le hn _ _).trans
      (Nat.mul_le_mul (Nat.mul_le_mul_left _ hVx) hVy))
      ((polyNorm_intCast_mul_le _ _).trans (Nat.mul_le_mul_left _ hXn)))
  have hΔn : polyNorm (C' 1 - S * C' 0) ≤ q3 / 2 + n * 1 * (q3 / 2) :=
    (polyNorm_sub_le _ _).trans (Nat.add_le_add (hC' 1)
      ((polyNorm_mul_le hn _ _).trans (Nat.mul_le_mul (Nat.mul_le_mul_left _ hS) (hC' 0))))
  have hW'n := polyNorm_le_div_of_mul (by decide) hW'
    ((polyNorm_add_le _ _).trans (Nat.add_le_add hWn
      ((polyNorm_intCast_mul_le _ _).trans (Nat.mul_le_mul_left _ hΔn))))
  have hbound : polyNorm W' < 56258131176 := hW'n.trans_lt (by decide +kernel)
  have hsmall (i : Fin n) : 2 * (W'.coeff i).natAbs < QL := by
    have := natAbs_coeff_le_of_polyNorm hbound.le i
    unfold QL
    omega
  refine ⟨fun index ↦ ?_, ?_⟩
  · unfold observedResidual
    simp only [BgvSemantics.messageCenter, Int.cast_zero, sub_zero, hfinal,
      reducePoly_coeff (by decide : 1 < QL) hn]
    rw [centeredLift_intCast (by decide) (hsmall index)]
    exact (natAbs_coeff_le_of_polyNorm le_rfl index).trans_lt hbound
  · -- The decrypted plaintext is the product of the two encoded plaintexts.
    have ht0 : ((t : Int) : ExactPoly t n) = 0 := by
      have := reducePoly_modulus_mul (n := n) t 1
      rwa [mul_one, reducePoly_intCast] at this
    have hfactor : ((998911 : Int) : ExactPoly t n) = ((q3 : Int) : ExactPoly t n) := by
      rw [intCast_eq_algebraMap, intCast_eq_algebraMap (value := (q3 : Int))]
      congr 1
    have hred_t := congrArg (reducePoly t n) hW'
    rw [map_mul, map_add, map_mul, reducePoly_intCast, reducePoly_intCast, ht0, zero_mul,
      add_zero] at hred_t
    have hVt (V : ErrorPoly n) (m : ExactMatrix t n 1 1) (A : ErrorPoly n)
        (hV : V = ((t : Int) : ErrorPoly n) * A + centeredIntLift (m 0 0)) :
        reducePoly t n V = m 0 0 := by
      rw [hV, map_add, map_mul, reducePoly_intCast, ht0, zero_mul, zero_add,
        reducePoly_centeredIntLift (q := t) (by decide) hn]
    have hplain' : plain 0 0 = mx 0 0 * my 0 0 := by
      rw [hplain, hfinal, centeredIntLift_reducePoly (q := QL) (by decide) hn _ hsmall, hfactor,
        mul_comm, hred_t]
      change reducePoly t n (Vx * Vy + ((t : Int) : ErrorPoly n) * X) = _
      rw [map_add, map_mul, map_mul (reducePoly t n) ((t : Int) : ErrorPoly n), reducePoly_intCast,
        ht0, zero_mul, add_zero, hVt Vx mx _ rfl, hVt Vy my _ rfl]
    exact slots_correct hfx htx hfy hty hplain' hvals hslots hideal

end MxxFheBgv
