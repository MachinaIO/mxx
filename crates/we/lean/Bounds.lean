import Mathlib

namespace MxxWe

/-! Capped nonnegative arithmetic for the application-owned bound checker.

These definitions are numeric helpers only. They do not assert anything about a Diamond
execution or a decoder residual.
-/

def cap (C x : Nat) : Nat := min x C

def cadd (C x y : Nat) : Nat := cap C (x + y)

def cmul (C x y : Nat) : Nat := cap C (x * y)

theorem cap_add_cap (C x y : Nat) : cap C (x + y) = cap C (cap C x + cap C y) := by
  by_cases hx : C ≤ x
  · by_cases hy : C ≤ y
    · have hxy : C ≤ x + y := by omega
      simp [cap, Nat.min_eq_right hx, Nat.min_eq_right hy, Nat.min_eq_right hxy]
    · have hy' : y ≤ C := Nat.le_of_not_ge hy
      have hxy : C ≤ x + y := by omega
      simp [cap, Nat.min_eq_right hx, Nat.min_eq_left hy', Nat.min_eq_right hxy]
  · have hx' : x ≤ C := Nat.le_of_not_ge hx
    by_cases hy : C ≤ y
    · have hxy : C ≤ x + y := by omega
      simp [cap, Nat.min_eq_left hx', Nat.min_eq_right hy, Nat.min_eq_right hxy]
    · have hy' : y ≤ C := Nat.le_of_not_ge hy
      simp [cap, Nat.min_eq_left hx', Nat.min_eq_left hy']

theorem cap_mul_cap (C x y : Nat) : cap C (x * y) = cap C (cap C x * cap C y) := by
  by_cases hx0 : x = 0
  · simp [cap, hx0]
  by_cases hy0 : y = 0
  · simp [cap, hy0]
  have hxpos : 0 < x := Nat.pos_of_ne_zero hx0
  have hypos : 0 < y := Nat.pos_of_ne_zero hy0
  by_cases hx : C ≤ x
  · have hxy : C ≤ x * y := hx.trans (Nat.le_mul_of_pos_right x hypos)
    by_cases hy : C ≤ y
    · have hcc : C ≤ C * C := by
        by_cases hC : C = 0
        · simp [hC]
        · exact Nat.le_mul_of_pos_right C (Nat.pos_of_ne_zero hC)
      simp [cap, Nat.min_eq_right hx, Nat.min_eq_right hy, Nat.min_eq_right hxy,
        Nat.min_eq_right hcc]
    · have hy' : y ≤ C := Nat.le_of_not_ge hy
      have hcy : C ≤ C * y := Nat.le_mul_of_pos_right C hypos
      simp [cap, Nat.min_eq_right hx, Nat.min_eq_left hy', Nat.min_eq_right hxy,
        Nat.min_eq_right hcy]
  · have hx' : x ≤ C := Nat.le_of_not_ge hx
    by_cases hy : C ≤ y
    · have hxy : C ≤ x * y := hy.trans (Nat.le_mul_of_pos_left y hxpos)
      have hcx : C ≤ C * x := Nat.le_mul_of_pos_right C hxpos
      have hxc : C ≤ x * C := by simpa [Nat.mul_comm] using hcx
      simp [cap, Nat.min_eq_left hx', Nat.min_eq_right hy, Nat.min_eq_right hxy,
        Nat.min_eq_right hxc]
    · have hy' : y ≤ C := Nat.le_of_not_ge hy
      simp [cap, Nat.min_eq_left hx', Nat.min_eq_left hy']

theorem cap_lt_iff (C x : Nat) : cap C x < C ↔ x < C := by
  simp [cap]

theorem cap_pow_add (C a m n : Nat) :
    cap C (a ^ (m + n)) = cmul C (cap C (a ^ m)) (cap C (a ^ n)) := by
  simp only [cmul, Nat.pow_add]
  exact cap_mul_cap C (a ^ m) (a ^ n)

def binaryPow (a n : Nat) : Nat :=
  if n = 0 then 1
  else
    let half := binaryPow a (n / 2)
    if n % 2 = 0 then half * half else half * half * a
termination_by n

def cappedBinaryPow (C a n : Nat) : Nat :=
  if h : n = 0 then cap C 1
  else
    let half := cappedBinaryPow C a (n / 2)
    if n % 2 = 0 then cmul C half half else cmul C (cmul C half half) (cap C a)
termination_by n

theorem binaryPow_eq_pow (a n : Nat) : binaryPow a n = a ^ n := by
  induction n using Nat.strong_induction_on with
  | h n ih =>
      by_cases hzero : n = 0
      · simp [binaryPow, hzero]
      · have hhalf : n / 2 < n := Nat.div_lt_self (Nat.pos_of_ne_zero hzero) (by omega)
        have hdecomp := Nat.div_add_mod n 2
        rw [binaryPow]
        simp only [hzero, ↓reduceIte]
        rw [ih (n / 2) hhalf]
        split <;> rename_i hparity
        · have hexp : n / 2 + n / 2 = n := by omega
          calc
            a ^ (n / 2) * a ^ (n / 2) = a ^ (n / 2 + n / 2) := by rw [Nat.pow_add]
            _ = a ^ n := by rw [hexp]
        · have hmod : n % 2 = 1 := by
            rcases Nat.mod_two_eq_zero_or_one n with hmod | hmod
            · exact False.elim (hparity hmod)
            · exact hmod
          have hexp : n / 2 + n / 2 + 1 = n := by omega
          calc
            a ^ (n / 2) * a ^ (n / 2) * a =
                a ^ (n / 2 + n / 2) * a := by rw [Nat.pow_add]
            _ = a ^ (n / 2 + n / 2) * a ^ 1 := by rw [Nat.pow_one]
            _ = a ^ (n / 2 + n / 2 + 1) := by
              simp [Nat.pow_add, mul_assoc]
            _ = a ^ n := by rw [hexp]

theorem cappedBinaryPow_eq_cap_pow (C a n : Nat) :
    cappedBinaryPow C a n = cap C (a ^ n) := by
  induction n using Nat.strong_induction_on with
  | h n ih =>
      by_cases hzero : n = 0
      · simp [cappedBinaryPow, hzero]
      · have hhalf : n / 2 < n := Nat.div_lt_self (Nat.pos_of_ne_zero hzero) (by omega)
        have hdecomp := Nat.div_add_mod n 2
        rw [cappedBinaryPow]
        simp only [hzero, ↓reduceDIte]
        rw [ih (n / 2) hhalf]
        split <;> rename_i hparity
        · have hexp : n / 2 + n / 2 = n := by omega
          calc
            cmul C (cap C (a ^ (n / 2))) (cap C (a ^ (n / 2))) =
                cap C (a ^ (n / 2 + n / 2)) :=
              (cap_pow_add C a (n / 2) (n / 2)).symm
            _ = cap C (a ^ n) := by rw [hexp]
        · rw [← cap_pow_add]
          calc
            cmul C (cap C (a ^ (n / 2 + n / 2))) (cap C a) =
                cap C (a ^ ((n / 2 + n / 2) + 1)) := by
              simpa [Nat.pow_one] using (cap_pow_add C a (n / 2 + n / 2) 1).symm
            _ = cap C (a ^ n) := by
              have hmod : n % 2 = 1 := by
                rcases Nat.mod_two_eq_zero_or_one n with hmod | hmod
                · exact False.elim (hparity hmod)
                · exact hmod
              have hexp : n / 2 + n / 2 + 1 = n := by omega
              simp [hexp]

end MxxWe

namespace MxxWe

structure BoundMatrix2 where
  a00 : Nat
  a01 : Nat
  a10 : Nat
  a11 : Nat
  deriving DecidableEq, Repr

def capMatrix (C : Nat) (A : BoundMatrix2) : BoundMatrix2 :=
  { a00 := cap C A.a00
    a01 := cap C A.a01
    a10 := cap C A.a10
    a11 := cap C A.a11 }

def matrixMul (A B : BoundMatrix2) : BoundMatrix2 :=
  { a00 := A.a00 * B.a00 + A.a01 * B.a10
    a01 := A.a00 * B.a01 + A.a01 * B.a11
    a10 := A.a10 * B.a00 + A.a11 * B.a10
    a11 := A.a10 * B.a01 + A.a11 * B.a11 }

def matrixMulCap (C : Nat) (A B : BoundMatrix2) : BoundMatrix2 :=
  { a00 := cadd C (cmul C A.a00 B.a00) (cmul C A.a01 B.a10)
    a01 := cadd C (cmul C A.a00 B.a01) (cmul C A.a01 B.a11)
    a10 := cadd C (cmul C A.a10 B.a00) (cmul C A.a11 B.a10)
    a11 := cadd C (cmul C A.a10 B.a01) (cmul C A.a11 B.a11) }

theorem boundMatrix2_ext {A B : BoundMatrix2}
    (h00 : A.a00 = B.a00) (h01 : A.a01 = B.a01)
    (h10 : A.a10 = B.a10) (h11 : A.a11 = B.a11) : A = B := by
  cases A
  cases B
  simp_all

theorem cap_two_products (C x y u v : Nat) :
    cap C (x * y + u * v) =
      cadd C (cmul C (cap C x) (cap C y)) (cmul C (cap C u) (cap C v)) := by
  simp only [cadd, cmul]
  rw [cap_add_cap]
  apply congrArg (cap C)
  exact congrArg₂ Nat.add (cap_mul_cap C x y) (cap_mul_cap C u v)

theorem capMatrix_mul (C : Nat) (A B : BoundMatrix2) :
    capMatrix C (matrixMul A B) = matrixMulCap C (capMatrix C A) (capMatrix C B) := by
  apply boundMatrix2_ext
  · exact cap_two_products C A.a00 B.a00 A.a01 B.a10
  · exact cap_two_products C A.a00 B.a01 A.a01 B.a11
  · exact cap_two_products C A.a10 B.a00 A.a11 B.a10
  · exact cap_two_products C A.a10 B.a01 A.a11 B.a11

def matrixOne : BoundMatrix2 :=
  { a00 := 1, a01 := 0, a10 := 0, a11 := 1 }

def matrixPow (A : BoundMatrix2) (n : Nat) : BoundMatrix2 :=
  if h : n = 0 then matrixOne
  else
    let half := matrixPow A (n / 2)
    if n % 2 = 0 then matrixMul half half else matrixMul (matrixMul half half) A
termination_by n

def matrixPowCap (C : Nat) (A : BoundMatrix2) (n : Nat) : BoundMatrix2 :=
  if h : n = 0 then capMatrix C matrixOne
  else
    let half := matrixPowCap C A (n / 2)
    if n % 2 = 0 then matrixMulCap C half half
    else matrixMulCap C (matrixMulCap C half half) (capMatrix C A)
termination_by n

theorem matrixPowCap_eq_cap_matrixPow (C : Nat) (A : BoundMatrix2) (n : Nat) :
    matrixPowCap C A n = capMatrix C (matrixPow A n) := by
  induction n using Nat.strong_induction_on with
  | h n ih =>
      by_cases hzero : n = 0
      · simp [matrixPowCap, matrixPow, hzero]
      · have hhalf : n / 2 < n := Nat.div_lt_self (Nat.pos_of_ne_zero hzero) (by omega)
        rw [matrixPowCap, matrixPow]
        simp only [hzero, ↓reduceDIte]
        rw [ih (n / 2) hhalf]
        split <;> rename_i hparity
        · rw [capMatrix_mul]
        · rw [capMatrix_mul, capMatrix_mul]

/-! Fixed-dimensional majorants used by the later Diamond application proof.

These are definitions and arithmetic identities only.  They do not provide the graph invariants
needed to connect a generated execution to these numbers.
-/

def injectorMatrix (n ell E K : Nat) : BoundMatrix2 :=
  { a00 := n
    a01 := 0
    a10 := 2 * n * E
    a11 := 2 * (ell + 2) * n * K }

def initialErrorBound (m n K N : Nat) : Nat := m * n * K * N

def layerFactor (ell n D : Nat) : Nat := 2 * (ell * n * D) + 4

def rawLayerBound (factor B0 H : Nat) : Nat := factor ^ H * B0

def cappedLayerBound (C factor B0 H : Nat) : Nat :=
  cmul C (cappedBinaryPow C factor H) (cap C B0)

def rawFinalBound (B0 a factor H : Nat) : Nat :=
  2 * B0 + a * (B0 + rawLayerBound factor B0 H)

def cappedFinalBound (C B0 a factor H : Nat) : Nat :=
  cadd C (cmul C 2 (cap C B0))
    (cmul C (cap C a) (cadd C (cap C B0) (cappedLayerBound C factor B0 H)))

theorem cap_add_capped (C x y : Nat) :
    cap C (cap C x + cap C y) = cap C (x + y) := (cap_add_cap C x y).symm

theorem cap_mul_capped (C x y : Nat) :
    cap C (cap C x * cap C y) = cap C (x * y) := (cap_mul_cap C x y).symm

theorem cap_mul_add_capped (C a x y : Nat) :
    cap C (cap C a * cap C (cap C x + cap C y)) = cap C (a * (x + y)) := by
  calc
    cap C (cap C a * cap C (cap C x + cap C y)) =
        cap C (cap C a * cap C (x + y)) := by rw [cap_add_capped]
    _ = cap C (a * (x + y)) := cap_mul_capped C a (x + y)

theorem cappedLayerBound_eq_cap_raw (C factor B0 H : Nat) :
    cappedLayerBound C factor B0 H = cap C (rawLayerBound factor B0 H) := by
  simp only [cappedLayerBound, rawLayerBound, cmul]
  rw [cappedBinaryPow_eq_cap_pow]
  exact (cap_mul_cap C (factor ^ H) B0).symm

theorem cappedFinalBound_eq_cap_raw (C B0 a factor H : Nat) :
    cappedFinalBound C B0 a factor H = cap C (rawFinalBound B0 a factor H) := by
  unfold cappedFinalBound
  rw [cappedLayerBound_eq_cap_raw]
  have htwo : cap C (2 * cap C B0) = cap C (2 * B0) := by
    rw [cap_mul_cap C 2 (cap C B0)]
    simpa only [cap, min_self, min_assoc] using (cap_mul_cap C 2 B0).symm
  simp only [rawFinalBound, cadd, cmul]
  rw [htwo, cap_mul_add_capped, cap_add_capped]

end MxxWe
