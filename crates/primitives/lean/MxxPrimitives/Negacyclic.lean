import Mathlib.RingTheory.AdjoinRoot
import Mathlib.Data.ZMod.Basic

namespace Mxx.Primitives

/- The negacyclic ring is the polynomial quotient by `X^n + 1`.  Unlike a
coefficient-level multiplication, this definition inherits all ring laws from
the quotient and therefore cannot silently lose associativity or commutativity.
The coefficient view is only a canonical degree-`< n` view used by bounds. -/
noncomputable def negacyclicModulus (n : Nat) (R : Type u) [CommRing R] : Polynomial R :=
  Polynomial.X ^ n + Polynomial.C 1

abbrev Negacyclic (n : Nat) (R : Type u) [CommRing R] :=
  AdjoinRoot (negacyclicModulus n R)

namespace Negacyclic

variable {n : Nat} {R : Type u} [CommRing R]

private theorem modulus_monic (h : n ≠ 0) : (negacyclicModulus n R).Monic := by
  exact Polynomial.monic_X_pow_add_C 1 h

theorem root_pow_n :
    (AdjoinRoot.root (negacyclicModulus n R)) ^ n = -1 := by
  have hroot : AdjoinRoot.mk (negacyclicModulus n R) (Polynomial.X ^ n) = -1 := by
    have hzero := (AdjoinRoot.mk_self (f := negacyclicModulus n R))
    simpa [negacyclicModulus, map_add, AdjoinRoot.mk_X] using congrArg
      (fun z => z - AdjoinRoot.mk (negacyclicModulus n R) (Polynomial.C 1)) hzero
  rw [← AdjoinRoot.mk_X, ← (AdjoinRoot.mk (negacyclicModulus n R)).map_pow]
  simpa using hroot

theorem root_pow_add_lt (i j : Fin n) :
    (AdjoinRoot.root (negacyclicModulus n R)) ^ (i : Nat) *
        (AdjoinRoot.root (negacyclicModulus n R)) ^ (j : Nat) =
      (AdjoinRoot.root (negacyclicModulus n R)) ^ ((i : Nat) + (j : Nat)) := by
  rw [← pow_add]

theorem root_pow_add_ge (i j : Fin n)
    (hij : n ≤ (i : Nat) + (j : Nat)) :
    (AdjoinRoot.root (negacyclicModulus n R)) ^ (i : Nat) *
        (AdjoinRoot.root (negacyclicModulus n R)) ^ (j : Nat) =
      - (AdjoinRoot.root (negacyclicModulus n R)) ^ ((i : Nat) + (j : Nat) - n) := by
  rw [← pow_add]
  have hsplit : (i : Nat) + (j : Nat) = n + ((i : Nat) + (j : Nat) - n) := by
    omega
  rw [hsplit, pow_add, root_pow_n]
  rw [Nat.add_sub_cancel_left n ((i : Nat) + (j : Nat) - n)]
  ring_nf

theorem root_pow_mul (i j : Fin n) :
    (AdjoinRoot.root (negacyclicModulus n R)) ^ (i : Nat) *
        (AdjoinRoot.root (negacyclicModulus n R)) ^ (j : Nat) =
      if (i : Nat) + (j : Nat) < n then
        (AdjoinRoot.root (negacyclicModulus n R)) ^ ((i : Nat) + (j : Nat))
      else
        - (AdjoinRoot.root (negacyclicModulus n R)) ^ ((i : Nat) + (j : Nat) - n) := by
  by_cases h : (i : Nat) + (j : Nat) < n
  · rw [if_pos h, root_pow_add_lt i j]
  · rw [if_neg h, root_pow_add_ge i j (Nat.le_of_not_gt h)]

/- Canonical coefficient extraction.  The zero-dimensional case has no indices;
the quotient remains a perfectly valid ring, while no coefficient is exposed. -/
noncomputable def coeff (x : Negacyclic n R) (i : Fin n) : R :=
  if h : n = 0 then Fin.elim0 (h ▸ i) else
    (AdjoinRoot.modByMonicHom (modulus_monic (R := R) h) x).coeff i.val

theorem expansion [Nontrivial R] (hn : 0 < n) (x : Negacyclic n R) :
    x = ∑ i : Fin n,
      algebraMap R (Negacyclic n R) (x.coeff i) *
        (AdjoinRoot.root (negacyclicModulus n R)) ^ (i : Nat) := by
  let hmonic := modulus_monic (R := R) (Nat.ne_of_gt hn)
  have hdegree : (negacyclicModulus n R).degree ≤ n := by
    rw [negacyclicModulus, Polynomial.degree_X_pow_add_C hn]
  induction x using AdjoinRoot.induction_on with
  | ih p =>
    have hsum := Polynomial.sum_modByMonic_coeff
      (p := p) (q := negacyclicModulus n R) hmonic hdegree
    calc
      AdjoinRoot.mk (negacyclicModulus n R) p =
          AdjoinRoot.mk (negacyclicModulus n R) (p %ₘ negacyclicModulus n R) := by
        rw [← AdjoinRoot.modByMonicHom_mk]
        exact (AdjoinRoot.mk_leftInverse hmonic (AdjoinRoot.mk _ p)).symm
      _ = AdjoinRoot.mk (negacyclicModulus n R)
          (∑ i : Fin n, Polynomial.monomial i
            ((p %ₘ negacyclicModulus n R).coeff i)) := by rw [hsum]
      _ = ∑ i : Fin n,
          algebraMap R (Negacyclic n R)
              (Negacyclic.coeff (AdjoinRoot.mk (negacyclicModulus n R) p) i) *
            AdjoinRoot.root (negacyclicModulus n R) ^ (i : Nat) := by
        rw [map_sum]
        apply Finset.sum_congr rfl
        intro i hi
        rw [← Polynomial.C_mul_X_pow_eq_monomial]
        rw [map_mul, map_pow, AdjoinRoot.mk_X]
        congr 1
        simp [coeff, hn.ne']

@[simp] theorem coeff_zero (i : Fin n) : (0 : Negacyclic n R).coeff i = 0 := by
  unfold coeff
  split
  · subst n
    exact Fin.elim0 i
  · simp

@[simp] theorem coeff_add (x y : Negacyclic n R) (i : Fin n) :
    (x + y).coeff i = x.coeff i + y.coeff i := by
  unfold coeff
  split
  · subst n
    exact Fin.elim0 i
  · simp

@[simp] theorem coeff_neg (x : Negacyclic n R) (i : Fin n) :
    (-x).coeff i = -x.coeff i := by
  unfold coeff
  split
  · subst n
    exact Fin.elim0 i
  · simp

@[simp] theorem coeff_sub (x y : Negacyclic n R) (i : Fin n) :
    (x - y).coeff i = x.coeff i - y.coeff i := by
  simp [sub_eq_add_neg]

theorem coeff_smul (a : R) (x : Negacyclic n R) (i : Fin n) :
    (algebraMap R (Negacyclic n R) a * x).coeff i = a * x.coeff i := by
  unfold coeff
  split
  · subst n
    exact Fin.elim0 i
  · rw [← Algebra.smul_def]
    rw [(AdjoinRoot.modByMonicHom _).map_smul]
    rfl

theorem coeff_root_pow {n : Nat} {R : Type u} [CommRing R] [Nontrivial R]
    (hn : 0 < n) (i k : Fin n) :
    Negacyclic.coeff (AdjoinRoot.root (negacyclicModulus n R) ^ (i : Nat)) k =
      if i = k then 1 else 0 := by
  let hmonic := modulus_monic (R := R) (Nat.ne_of_gt hn)
  unfold coeff
  rw [dif_neg hn.ne']
  rw [← AdjoinRoot.mk_X, ← (AdjoinRoot.mk (negacyclicModulus n R)).map_pow]
  rw [AdjoinRoot.modByMonicHom_mk]
  have hdeg : ((Polynomial.X : Polynomial R) ^ (i : Nat)).degree <
      (negacyclicModulus n R).degree := by
    rw [negacyclicModulus, Polynomial.degree_X_pow, Polynomial.degree_X_pow_add_C hn]
    exact_mod_cast i.isLt
  rw [(Polynomial.modByMonic_eq_self_iff hmonic).mpr hdeg]
  simp [Fin.ext_iff, eq_comm]

/- Small concrete instances exercise both the non-wrapping and signed-wrapping branches. -/
example :
    (AdjoinRoot.root (negacyclicModulus 1 Int)) ^ (0 : Nat) *
        (AdjoinRoot.root (negacyclicModulus 1 Int)) ^ (0 : Nat) =
      (AdjoinRoot.root (negacyclicModulus 1 Int)) ^ (0 : Nat) := by
  simpa using root_pow_mul (R := Int) (n := 1)
    (⟨0, by omega⟩ : Fin 1) ⟨0, by omega⟩

example :
    (AdjoinRoot.root (negacyclicModulus 2 Int)) ^ (1 : Nat) *
        (AdjoinRoot.root (negacyclicModulus 2 Int)) ^ (1 : Nat) = -1 := by
  simpa using root_pow_mul (R := Int) (n := 2)
    (⟨1, by omega⟩ : Fin 2) ⟨1, by omega⟩

example :
    (AdjoinRoot.root (negacyclicModulus 3 Int)) ^ (2 : Nat) *
        (AdjoinRoot.root (negacyclicModulus 3 Int)) ^ (2 : Nat) =
      -(AdjoinRoot.root (negacyclicModulus 3 Int)) := by
  simpa using root_pow_mul (R := Int) (n := 3)
    (⟨2, by omega⟩ : Fin 3) ⟨2, by omega⟩

end Negacyclic

abbrev ExactPoly (q n : Nat) := Negacyclic n (ZMod q)
abbrev ErrorPoly (n : Nat) := Negacyclic n Int

end Mxx.Primitives
