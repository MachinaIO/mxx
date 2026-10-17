import PeDecodeRoot

/-!
# The slot automorphisms

`σ_k` on matrices distributes over the ring operations and fixes constant rows; the slot
automorphisms `σ_{1 + 256 i}` compose additively in `i` modulo 256; and the pairing row of pair
`(i, k)` is `i - idx(k)` modulo 256 for the helper member `k = σ_{1 + 256 idx(k)}`.
-/

set_option maxRecDepth 200000

namespace PeProof

open Mxx.Primitives MxxRuntime MxxExponentLut

/-- Substituting `X ↦ X^a` after `X ↦ X^b` is substituting `X ↦ X^c` for `a b ≡ c (mod 2n)`. -/
theorem substitution_comp {n : Nat} {R : Type} [CommRing R] {a b c : Nat} (ha : Odd a) (hb : Odd b)
    (hc : Odd c) (habc : a * b % (2 * n) = c % (2 * n)) (x : Negacyclic n R) :
    substitution a ha (substitution b hb x) = substitution c hc x := by
  have h : (substitution (n := n) (R := R) a ha).comp (substitution b hb) = substitution c hc := by
    apply AdjoinRoot.ringHom_ext
    · ext r
      simp [substitution]
    · simp only [RingHom.comp_apply, substitution_root, map_pow]
      have hroot : AdjoinRoot.root (negacyclicModulus n R) ^ (2 * n) = 1 := by
        rw [pow_mul', Negacyclic.root_pow_n]
        simp
      have key : ∀ m, AdjoinRoot.root (negacyclicModulus n R) ^ m =
          AdjoinRoot.root (negacyclicModulus n R) ^ (m % (2 * n)) := fun m ↦ by
        conv_lhs => rw [← Nat.div_add_mod m (2 * n), pow_add, pow_mul, hroot, one_pow, one_mul]
      rw [← pow_mul, key (a * b), key c, habc]
  exact RingHom.congr_fun h x

theorem substitution_one {n : Nat} {R : Type} [CommRing R] (x : Negacyclic n R) :
    substitution 1 odd_one x = x := by
  have h : (substitution (n := n) (R := R) 1 odd_one) = RingHom.id _ := by
    apply AdjoinRoot.ringHom_ext
    · ext r
      simp [substitution]
    · simp
  exact RingHom.congr_fun h x

section Matrices

variable {k : Nat} (hk : Odd k)

theorem subM_add {r c : Nat} (x y : ExactMatrix Q N r c) :
    substituteMatrix k hk (x + y) = substituteMatrix k hk x + substituteMatrix k hk y := by
  funext i j
  simp [substituteMatrix]

theorem subM_sub {r c : Nat} (x y : ExactMatrix Q N r c) :
    substituteMatrix k hk (x - y) = substituteMatrix k hk x - substituteMatrix k hk y := by
  funext i j
  simp [substituteMatrix]

theorem subM_neg {r c : Nat} (x : ExactMatrix Q N r c) :
    substituteMatrix k hk (-x) = -substituteMatrix k hk x := by
  funext i j
  simp [substituteMatrix]

theorem subM_zero {r c : Nat} : substituteMatrix k hk (0 : ExactMatrix Q N r c) = 0 := by
  funext i j
  simp [substituteMatrix]

theorem subM_smul {r c : Nat} (a : ExactPoly Q N) (x : ExactMatrix Q N r c) :
    substituteMatrix k hk (a • x) = substitution k hk a • substituteMatrix k hk x := by
  funext i j
  simp [substituteMatrix]

theorem subM_mul {r m c : Nat} (x : ExactMatrix Q N r m) (y : ExactMatrix Q N m c) :
    substituteMatrix k hk (x * y) = substituteMatrix k hk x * substituteMatrix k hk y := by
  funext i j
  simp [substituteMatrix, Matrix.mul_apply, map_sum]

theorem subM_sum {ι : Type} (s : Finset ι) {r c : Nat} (f : ι → ExactMatrix Q N r c) :
    substituteMatrix k hk (∑ i ∈ s, f i) = ∑ i ∈ s, substituteMatrix k hk (f i) := by
  funext i j
  simp [substituteMatrix, Matrix.sum_apply, map_sum]

theorem subM_one_index {r c : Nat} (x : ExactMatrix Q N r c) : substituteMatrix 1 odd_one x = x := by
  funext i j
  exact substitution_one _

theorem subM_comp {a b c' : Nat} (ha : Odd a) (hb : Odd b) (hc : Odd c') (habc : a * b % (2 * N) = c' % (2 * N))
    {r c : Nat} (x : ExactMatrix Q N r c) :
    substituteMatrix a ha (substituteMatrix b hb x) = substituteMatrix c' hc x := by
  funext i j
  exact substitution_comp ha hb hc habc _

theorem subM_g0 : substituteMatrix k hk g0 = g0 := by
  funext i j
  obtain ⟨x, hx⟩ := gadgetMatrixRuns_constant g0_runs i j
  simp only [substituteMatrix, hx, substitution_algebraMap]

theorem subM_joinRow (x : Fin 2 → ExactMatrix Q N 1 30) :
    substituteMatrix k hk (joinRow x) = joinRow fun j ↦ substituteMatrix k hk (x j) := by
  funext i p
  simp only [substituteMatrix, joinRow, joinedEntry_split]
  split <;> rfl

theorem subM_gZero : substituteMatrix k hk gZero = gZero := by
  rw [gZero, subM_joinRow]
  congr 1
  funext j
  fin_cases j
  · exact subM_g0 hk
  · exact subM_zero hk

theorem subM_zeroG : substituteMatrix k hk zeroG = zeroG := by
  rw [zeroG, subM_joinRow]
  congr 1
  funext j
  fin_cases j
  · exact subM_zero hk
  · exact subM_g0 hk

end Matrices

/-! ## Slot indices -/

/-- `σ_{1 + 256 a} σ_{1 + 256 b} = σ_{1 + 256 ((a + b) mod 256)}`. -/
theorem hIndex_mul (a b : Nat) :
    hIndex a * hIndex b % (2 * N) = hIndex ((a + b) % 256) % (2 * N) := by
  unfold hIndex
  have e : (1 + 256 * a) * (1 + 256 * b) = 1 + 256 * ((a + b) % 256) +
      65536 * ((a + b) / 256 + a * b) := by
    have := Nat.div_add_mod (a + b) 256
    nlinarith
  rw [e, show 2 * N = 65536 from rfl, Nat.add_mul_mod_self_left]

/-- The slot of the pairing row composed with the helper member is the slot itself. -/
theorem slot_pair (i : Fin 256) (k : Fin 45) :
    slotAut (pairRow (45 * i + k)) * kHelp k % (2 * N) = slotAut i % (2 * N) := by
  rw [pairRow_eq i k i.isLt k.isLt, slotAut_eq _ (Nat.mod_lt _ (by norm_num)), kHelp_eq k k.isLt,
    hIndex_mul, slotAut_eq i i.isLt]
  have := helperIdx_lt k k.isLt
  congr 2
  omega

end PeProof
