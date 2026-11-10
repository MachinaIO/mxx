import TfheFresh

/-!
# The bootstrapped NAND gate

The blind rotation starts from the lookup table rotated by `v = initialExp b`, where `b` is the
body of the NAND input. For every `v`, `brFinal v` is the accumulator, `brErr v` the constant
coefficient of its integer error, and `outDigits v` the key-switching digits of its extraction;
the actual run is the one at `v = initialExp b`. `nand_output_phase` gives the phase of the gate's
output: the lookup-table sign `±⌊Q/8⌋` switched to `q`, plus `(q/Q) brErr v - Σ dᵢ εᵢ`.
-/

set_option maxRecDepth 16384

namespace MxxFheTfhe

open Mxx.Primitives MxxRuntime MxxSecurity

/-- A packed table whose entries all equal `⌊Q / 8⌋` is the lookup table. -/
theorem packed_lut {table : Nat}
    (h : checkBelow (fun i ↦ packedEntry 30 table i == 654329600) 1024 = true) :
    (packedPolynomial (q := Q) (n := N) 30 1024 table) 0 0 =
      reducePoly Q N (intPoly fun _ ↦ 654329600) := by
  letI : Fact (1 < Q) := ⟨by decide⟩
  apply negacyclic_ext (by decide)
  intro i
  rw [packedPolynomial_coeff (by decide) (by decide), reducePoly_coeff (by decide) (by decide),
    intPoly_coeff (by decide)]
  rw [if_pos (show i.val < 1024 from i.isLt)]
  have hi := checkBelow_spec h i.val i.isLt
  simp only [beq_iff_eq] at hi
  rw [hi]
  rfl

theorem lut_eq : lut 0 0 = reducePoly Q N (intPoly fun _ ↦ 654329600) :=
  packed_lut (by decide +kernel)

section Gate

variable (ω : Key → ℤ)

/-- The rotation of every bootstrapping step for the NAND input mask `a`. -/
def rotOf (a : Fin 720 → Int) (i : Fin 720) : Int := roundRing (a i)

/-- The initial accumulator `(0, X^v L)`. -/
noncomputable def brInit (v : Int) : ExactMatrix Q N 2 1 :=
  concatRowsOf (0 : ExactMatrix Q N 1 1) (multiplyMonomial lut v)

/-- The accumulator after blind rotation from `X^v L`, with the NAND input mask `a`. -/
noncomputable def brFinal (a : Fin 720 → Int) (v : Int) : ExactMatrix Q N 2 1 :=
  accOf (smp (.residue Q) 0 ω) (smp ringLaw 3 ω) (smp bitLaw 2 ω) (secretOf ω) (rotOf a) (brInit v)
    720

/-- The constant coefficient of the integer error of `brFinal`. -/
noncomputable def brErr (a : Fin 720 → Int) (v : Int) : Int :=
  (errSeq (smp (.residue Q) 0 ω) (smp ringLaw 3 ω) (smpInt ringLaw 3 ω) (smp bitLaw 2 ω)
    (secretOf ω) (rotOf a) (brInit v) 720).coeff 0

/-- The total rotation of the blind rotation. -/
noncomputable def brShift (a : Fin 720 → Int) : Nat := expSeq (secretOf ω) (rotOf a) 720

/-- The key-switching digits of the extraction of `brFinal`. -/
noncomputable def outDigits (a : Fin 720 → Int) (v : Int) : Fin 8192 → Int :=
  digitAt (extractAt (coeffVals (rowOf (brFinal ω a v) 0)))

end Gate

theorem secretOf_eq (ω : Key → ℤ) (j : Fin 720) : secretOf ω j = bitLaw.clamp (ω (1, j.val)) := by
  rw [secretOf, coeffVals_smp_bit]
  rcases bit_clamp (ω (1, j.val)) with h | h <;> simp [h]

theorem secretOf_binary (ω : Key → ℤ) (j : Fin 720) : secretOf ω j = 0 ∨ secretOf ω j = 1 := by
  rw [secretOf_eq]
  exact bit_clamp _

theorem roundRing_range (x : Int) : 0 ≤ roundRing x ∧ roundRing x < 2048 := by
  unfold roundRing; omega

/-- The sign of the lookup table after a total rotation `T`. -/
def lutSign (T : Nat) : Int := if T % 2048 = 0 ∨ 1024 < T % 2048 then 1 else -1

theorem rowOf_apply {q n : Nat} (x : ExactMatrix q n 2 1) (r : Fin 2) : rowOf x r 0 0 = x r 0 :=
  rfl

/-- The constant phase of the accumulator after blind rotation from `X^v L`. -/
theorem br_constant_phase (ω : Key → ℤ) (a : Fin 720 → Int) (v : Int) :
    ((lutSign (brShift ω a + (v % 2048).toNat) * 654329600 + brErr ω a v : Int) : ZMod Q) =
      (rowOf (brFinal ω a v) 1 0 0).coeff 0 -
        (rowOf (brFinal ω a v) 0 0 0 * (smp bitLaw 2 ω : ExactMatrix Q N 1 1) 0 0).coeff 0 := by
  letI : Fact (1 < Q) := ⟨by decide⟩
  have hphase := accOf_phase (smp (.residue Q) 0 ω) (smp ringLaw 3 ω) (smpInt ringLaw 3 ω)
    (smp_eq_reduce _ _ _) (smp bitLaw 2 ω) (secretOf ω) (secretOf_binary ω) (rotOf a)
    (fun i ↦ roundRing_range _) (brInit v) 720
  have hinit1 : brInit v 1 0 = reducePoly Q N (intPoly fun _ ↦ 654329600) *
      rootQ ^ (v % 2048).toNat := by
    show lut 0 0 * rootQ ^ (v % (2 * (N : Int))).toNat = _
    rw [lut_eq]
    rfl
  have hinit0 : brInit v 0 0 = 0 := rfl
  rw [hinit1, hinit0, mul_zero, sub_zero] at hphase
  have hfinal : rowOf (brFinal ω a v) 1 0 0 -
      rowOf (brFinal ω a v) 0 0 0 * (smp bitLaw 2 ω : ExactMatrix Q N 1 1) 0 0 =
      reducePoly Q N (rootZ ^ (brShift ω a + (v % 2048).toNat) * intPoly (fun _ ↦ 654329600) +
        errSeq (smp (.residue Q) 0 ω) (smp ringLaw 3 ω) (smpInt ringLaw 3 ω) (smp bitLaw 2 ω)
          (secretOf ω) (rotOf a) (brInit v) 720) := by
    rw [rowOf_apply, rowOf_apply, mul_comm (brFinal ω a v 0 0)]
    unfold brFinal
    rw [hphase, map_add, map_mul, reducePoly_rootZ_pow, pow_add]
    simp only [brShift]
    ring
  rw [← Negacyclic.coeff_sub, hfinal, reducePoly_coeff (by decide) (by decide),
    Negacyclic.coeff_add]
  rw [show (0 : Fin N) = ⟨0, by decide⟩ from rfl,
    coeff_zero_root_pow_mul_const_mod (by decide : 0 < N)]
  simp only [lutSign, brErr, N]
  push_cast
  split_ifs <;> simp

theorem nand_output_phase (messages : Fin 2 → Int) (ω : Key → ℤ) (a : Fin 720 → Int) (v : Int)
    (ha : a = nandMask (evalInputsOf messages ω).2.2.2.1 (evalInputsOf messages ω).2.2.2.2.1)
    (hv : v = initialExp (nandBody (evalInputsOf messages ω).2.1 (evalInputsOf messages ω).2.2.1)) :
    ∃ η : Int, (((runOf messages ω).2.1 -
        ∑ j, (runOf messages ω).1 j * secretOf ω j : Int) : ZMod q) = η ∧
      |(η : ℝ) - ((4294967296 / 5234636801 : ℝ) *
        ((lutSign (brShift ω a + (v % 2048).toNat) * 654329600 + brErr ω a v : Int) : ℝ) -
        ∑ i, (outDigits ω a v i : ℝ) * kskErr ω i)| ≤ allowance := by
  have hzp : ∀ k, ((smp bitLaw 2 ω : ExactMatrix Q N 1 1) 0 0).coeff k =
      (ringSecret ω k : ZMod Q) := by
    intro k
    simp [ringSecret, coeffVals]
  have hrun : runOf messages ω = (ksMask (kskAOf ω) (outDigits ω a v),
      ksBody (kskBOf ω) (rescale (coeffVals (rowOf (brFinal ω a v) 1) 0)) (outDigits ω a v), ()) := by
    subst ha hv
    rfl
  rw [hrun]
  exact ks_phase (s := secretOf ω) (ringSecret_binary ω) (A := kskAOf ω) (B := kskBOf ω)
    (ε := kskErr ω) (fun i ↦ rfl) (rowOf (brFinal ω a v) 0) (rowOf (brFinal ω a v) 1) hzp
    (br_constant_phase ω a v)

/-! ## Selecting the lookup table -/

/-- The rounding to `2 N` of the phase selects the right half of the lookup table when the
rounded phase `2^21 K` is within `2^29 - 2^21` of `(2 m₀ + 2 m₁ - 3) 2^29` modulo `q`. -/
theorem lut_select {m0 m1 : Int} (hm0 : m0 = 0 ∨ m0 = 1) (hm1 : m1 = 0 ∨ m1 = 1) {K W j : Int}
    (hj : 2097152 * K = W + 4294967296 * j)
    (hW : |W - (2 * m0 + 2 * m1 - 3) * 536870912| < 534773760) {T : Nat}
    (hT : (T : Int) % 2048 = K % 2048) : lutSign T = (1 - m0 * m1) * 2 - 1 := by
  rw [abs_lt] at hW
  unfold lutSign
  have hT' : ((T % 2048 : Nat) : Int) = K % 2048 := by rw [← hT]; push_cast; rfl
  rcases hm0 with rfl | rfl <;> rcases hm1 with rfl | rfl <;> norm_num at hW ⊢
  all_goals first | omega | (intro h; omega) | (split_ifs <;> omega)

/-- Decryption of a phase within `2^29` of `±2^29`. -/
theorem decOf_of_phase {b : Int} {a s : Fin 720 → Int} {η sg : Int} (hsg : sg = 1 ∨ sg = -1)
    (hη : ((b - ∑ j, a j * s j : Int) : ZMod q) = η) (hb : |η - sg * 536870912| < 536870912) :
    decOf b a s = (sg + 1) / 2 := by
  obtain ⟨k, hk⟩ := exists_of_cast_eq hη
  unfold decOf
  simp only
  generalize ∑ j, a j * s j = S at hk ⊢
  rw [abs_lt] at hb
  simp only [q, Nat.cast_ofNat] at hk
  rcases hsg with rfl | rfl <;> norm_num at hb ⊢
  all_goals first | omega | (split_ifs <;> omega) | (intro h; omega)

/-! ## One run -/

section Run

variable (messages : Fin 2 → Int) (ω : Key → ℤ)

/-- The NAND input mask `-(a₀ + a₁)`. -/
noncomputable def nandA : Fin 720 → Int :=
  nandMask (freshOf ω 15 (messages 0)).1 (freshOf ω 18 (messages 1)).1

/-- The initial rotation `-round(b)` of the NAND input. -/
noncomputable def nandV : Int :=
  initialExp (nandBody (freshOf ω 15 (messages 0)).2.1 (freshOf ω 18 (messages 1)).2.1)

/-- The key-switching digits of the two fresh encryptions. -/
noncomputable def digits0 : Fin 8192 → Int :=
  encDigits (encInputsOf ω (messages 0)) (smp bitLaw 15 ω) (smp ringLaw 16 ω)

noncomputable def digits1 : Fin 8192 → Int :=
  encDigits (encInputsOf ω (messages 1)) (smp bitLaw 18 ω) (smp ringLaw 19 ω)

/-- The rounding errors of the switch of the NAND input mask to `2 N`. -/
noncomputable def msErr : Int :=
  ∑ j, secretOf ω j * (2097152 * ((nandA messages ω j * 2048 + 2147483648) / 4294967296) -
    nandA messages ω j)

/-- The input noise that decides the lookup-table half. -/
noncomputable def inNoise : ℝ :=
  (msErr messages ω : ℝ) - ∑ i, ((digits0 messages ω i + digits1 messages ω i : Int) : ℝ) * kskErr ω i

/-- The output noise of the gate when the blind rotation starts from `X^v L`. -/
noncomputable def outNoise (v : Int) : ℝ :=
  (4294967296 / 5234636801 : ℝ) * (brErr ω (nandA messages ω) v : ℝ) -
    ∑ i, (outDigits ω (nandA messages ω) v i : ℝ) * kskErr ω i

end Run

/-- The input-noise threshold. -/
abbrev inThreshold : ℝ := 458000000
/-- The output-noise threshold. -/
abbrev outThreshold : ℝ := 503000000

theorem cast_emod_2048 (x : Int) : ((x % 2048 : Int) : ZMod 2048) = x := ZMod.intCast_mod x 2048

theorem nandV_range (messages : Fin 2 → Int) (ω : Key → ℤ) :
    0 ≤ nandV messages ω ∧ nandV messages ω < 2048 := by
  unfold nandV initialExp
  omega

theorem expSeq_cast (s rot : Fin 720 → Int) (hs : ∀ j, 0 ≤ s j) (hrot : ∀ j, 0 ≤ rot j) :
    ((expSeq s rot 720 : Nat) : Int) = ∑ j, s j * rot j := by
  rw [expSeq, Finset.sum_range (fun j ↦ if h : j < 720 then stepShift s rot ⟨j, h⟩ else 0),
    Nat.cast_sum]
  refine Finset.sum_congr rfl fun j _ ↦ ?_
  rw [dif_pos j.isLt]
  simp only [Fin.eta, stepShift, Nat.cast_mul]
  rw [Int.toNat_of_nonneg (hs j), Int.toNat_of_nonneg (hrot j)]

/-- With small input and output noise the run decrypts to the NAND of the messages. -/
theorem run_correct {messages : Fin 2 → Int} (hm0 : messages 0 = 0 ∨ messages 0 = 1)
    (hm1 : messages 1 = 0 ∨ messages 1 = 1) (ω : Key → ℤ)
    (hin : |inNoise messages ω| ≤ inThreshold)
    (hout : |outNoise messages ω (nandV messages ω)| ≤ outThreshold) :
    failR messages ω = false := by
  set m0 := messages 0
  set m1 := messages 1
  set s := secretOf ω
  set a := nandA messages ω with ha
  set v := nandV messages ω with hvdef
  set b0 := (freshOf ω 15 m0).2.1
  set b1 := (freshOf ω 18 m1).2.1
  set A0 := (freshOf ω 15 m0).1
  set A1 := (freshOf ω 18 m1).1
  obtain ⟨η0, hη0, hb0⟩ := fresh_phase ω 15 m0
  obtain ⟨η1, hη1, hb1⟩ := fresh_phase ω 18 m1
  obtain ⟨ηo, hηo, hbo⟩ := nand_output_phase messages ω a v rfl rfl
  have hs : ∀ j, s j = 0 ∨ s j = 1 := secretOf_binary ω
  -- The rounding of the NAND input to `2 N`.
  set bN := nandBody b0 b1 with hbN
  set rb := (bN * 2048 + 2147483648) / 4294967296 with hrb
  set ra : Fin 720 → Int := fun j ↦ (a j * 2048 + 2147483648) / 4294967296 with hra
  set K := ∑ j, s j * ra j - rb with hK
  set rbE := 2097152 * rb - bN with hrbE
  have hbN_range : 0 ≤ bN ∧ bN < 4294967296 := by simp only [hbN, nandBody]; omega
  have hrbEb : -1048576 < rbE ∧ rbE ≤ 1048576 := by simp only [hrbE, hrb]; omega
  have hv : v = (0 - rb % 2048) % 2048 := rfl
  have hT : ((brShift ω a + (v % 2048).toNat : Nat) : Int) % 2048 = K % 2048 := by
    have hcast : (((brShift ω a + (v % 2048).toNat : Nat) : Int) : ZMod 2048) = (K : ZMod 2048) := by
      have hv0 := nandV_range messages ω
      rw [Nat.cast_add, brShift, expSeq_cast s (rotOf a) (fun j ↦ by rcases hs j with h | h <;> rw [h] <;> norm_num)
        (fun j ↦ (roundRing_range _).1), Int.toNat_of_nonneg (by omega),
        Int.emod_eq_of_lt hv0.1 hv0.2, show nandV messages ω = (0 - rb % 2048) % 2048 from rfl, hK]
      simp only [rotOf, roundRing, Int.cast_add, Int.cast_sum, Int.cast_mul, Int.cast_sub,
        cast_emod_2048, Int.cast_zero, hra]
      ring
    exact (ZMod.intCast_eq_intCast_iff' _ _ _).mp hcast
  -- The rounded phase modulo `q`.
  have hms : msErr messages ω = ∑ j, s j * (2097152 * ra j - a j) := rfl
  set W : Int := η0 + η1 - 536870912 + msErr messages ω - rbE with hWdef
  have hint : 2097152 * K = ∑ j, a j * s j + msErr messages ω - bN - rbE := by
    have hsplit : ∑ j, 2097152 * (s j * ra j) =
        ∑ j, a j * s j + ∑ j, s j * (2097152 * ra j - a j) := by
      rw [← Finset.sum_add_distrib]
      exact Finset.sum_congr rfl fun j _ ↦ by ring
    rw [hms, hK, hrbE, mul_sub, Finset.mul_sum, hsplit]
    ring
  have hcong : ((2097152 * K : Int) : ZMod q) = (W : ZMod q) := by
    have hbN' : ((bN : Int) : ZMod q) = 536870912 - ((b0 : ZMod q) + (b1 : ZMod q)) := by
      simp only [hbN, nandBody, cast_emod_q, Int.cast_add, Int.cast_sub, Int.cast_zero]
      push_cast
      ring
    have ha' (j : Fin 720) : ((a j : Int) : ZMod q) = -((A0 j : ZMod q) + (A1 j : ZMod q)) := by
      show (((0 - (A0 j + A1 j) % 4294967296) % 4294967296 : Int) : ZMod q) = _
      simp only [cast_emod_q, Int.cast_sub, Int.cast_zero, Int.cast_add]
      ring
    rw [hint, hWdef]
    push_cast
    rw [← hη0, ← hη1, hbN']
    push_cast
    simp only [ha', neg_mul, add_mul, Finset.sum_neg_distrib, Finset.sum_add_distrib]
    ring
  obtain ⟨j, hj⟩ := exists_of_cast_eq hcong
  simp only [q, Nat.cast_ofNat] at hj
  -- The rounded phase is near `(2 m₀ + 2 m₁ - 3) 2^29`.
  set S0 := ∑ i, (digits0 messages ω i : ℝ) * kskErr ω i with hS0
  set S1 := ∑ i, (digits1 messages ω i : ℝ) * kskErr ω i with hS1
  have hb0' : |(η0 : ℝ) - ((4294967296 / 5234636801 : ℝ) *
      ((freshNoise ω 15 : ℝ) + ((m0 : ℝ) * 2 - 1) * 654329600) - S0)| ≤ allowance := by
    have h := hb0
    push_cast at h
    exact h
  have hb1' : |(η1 : ℝ) - ((4294967296 / 5234636801 : ℝ) *
      ((freshNoise ω 18 : ℝ) + ((m1 : ℝ) * 2 - 1) * 654329600) - S1)| ≤ allowance := by
    have h := hb1
    push_cast at h
    exact h
  have hinS : inNoise messages ω = (msErr messages ω : ℝ) - S0 - S1 := by
    have hsum : ∑ i, ((digits0 messages ω i + digits1 messages ω i : Int) : ℝ) * kskErr ω i =
        S0 + S1 := by
      rw [hS0, hS1, ← Finset.sum_add_distrib]
      exact Finset.sum_congr rfl fun i _ ↦ by push_cast; ring
    unfold inNoise
    rw [hsum]
    ring
  have hn0 : |(freshNoise ω 15 : ℝ)| ≤ 5114304 := by exact_mod_cast freshNoise_bound ω 15
  have hn1 : |(freshNoise ω 18 : ℝ)| ≤ 5114304 := by exact_mod_cast freshNoise_bound ω 18
  have hrbr : -1048576 < (rbE : ℝ) ∧ (rbE : ℝ) ≤ 1048576 := by exact_mod_cast hrbEb
  have hWr : |(W : ℝ) - (2 * m0 + 2 * m1 - 3) * 536870912| < 534773760 := by
    rw [hinS] at hin
    rw [abs_le] at hb0' hb1' hin hn0 hn1
    rw [abs_lt]
    simp only [hWdef]
    push_cast
    rcases hm0 with h0 | h0 <;> rcases hm1 with h1 | h1 <;> rw [h0] at hb0' ⊢ <;>
      rw [h1] at hb1' ⊢ <;> norm_num at hb0' hb1' ⊢ <;> constructor <;>
      linarith [hb0'.1, hb0'.2, hb1'.1, hb1'.2, hin.1, hin.2, hn0.1, hn0.2, hn1.1, hn1.2, hrbr.1,
        hrbr.2]
  have hW : |W - (2 * m0 + 2 * m1 - 3) * 536870912| < 534773760 := by exact_mod_cast hWr
  have hsign := lut_select hm0 hm1 hj hW hT
  -- Decryption of the output.
  set sg : Int := (1 - m0 * m1) * 2 - 1 with hsgdef
  have hsg : sg = 1 ∨ sg = -1 := by
    rcases hm0 with h | h <;> rcases hm1 with h' | h' <;> simp [hsgdef, h, h']
  have hbor : |(ηo : ℝ) - sg * 536870912| < 536870912 := by
    rw [hsign] at hbo
    push_cast at hbo
    unfold outNoise at hout
    rw [← ha] at hout
    simp only [outThreshold, allowance] at hout hbo
    rw [abs_le] at hbo hout
    rw [abs_lt]
    have hsgr : (sg : ℝ) = 1 ∨ (sg : ℝ) = -1 := by
      rcases hsg with h | h <;> [left; right] <;> exact_mod_cast h
    rcases hsgr with h | h <;> rw [h] at hbo ⊢ <;> constructor <;>
      linarith [hbo.1, hbo.2, hout.1, hout.2]
  have hbo' : |ηo - sg * 536870912| < 536870912 := by exact_mod_cast hbor
  have hdec := decOf_of_phase hsg hηo hbo'
  rw [failR_eq]
  simp only [decide_eq_false_iff_not, not_not]
  rw [hdec, hsgdef, show messages 0 = m0 from rfl, show messages 1 = m1 from rfl]
  rcases hm0 with h | h <;> rcases hm1 with h' | h' <;> rw [h, h'] <;> norm_num

end MxxFheTfhe
