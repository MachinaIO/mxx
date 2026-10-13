import ExponentDynamic
import PeDynamic

/-!
# Dynamic products with a public top row

Every RHS gadget ciphertext of the scheme has a public top row `C_{0,:}`: its digits (the digit
rows `a < m` of `Dec(C)`) are known when the key is computed, so they need no companion. The
ciphertext supplies companions only for the digit rows `m + a` of its bottom row, and the key
takes the public rows' term `A_{:,0} Dec(C_{0,:})` directly:

`M_{k,j} = Σ_a B_{j,a,k} Dec(A_{k',m+a}) + A_{k,0} topRows(Dec(C)_j)`.

`companion_rows_bottom` sums the bottom companions, and `mul_public_top` is the product.
-/

namespace MxxPe

open Mxx.Primitives MxxRuntime MxxExponentLut
open scoped BigOperators

variable {n : Nat}

/-- The bottom half `x_1 * bottomRows d` of `pairMul x d`: the private digit rows. -/
def pairMulBottom {α : Type} [CommRing α] {m m' : Nat} (x : Fin 2 → Matrix (Fin 1) (Fin m) α)
    (d : Matrix (Fin (m + m)) (Fin m') α) : Matrix (Fin 1) (Fin m') α :=
  x 1 * bottomRows d

/-- The top half `x_0 * topRows d` of `pairMul x d`: the public digit rows. -/
def pairMulTop {α : Type} [CommRing α] {m m' : Nat} (x : Fin 2 → Matrix (Fin 1) (Fin m) α)
    (d : Matrix (Fin (m + m)) (Fin m') α) : Matrix (Fin 1) (Fin m') α :=
  x 0 * topRows d

theorem pairMul_split {α : Type} [CommRing α] {m m' : Nat} (x : Fin 2 → Matrix (Fin 1) (Fin m) α)
    (d : Matrix (Fin (m + m)) (Fin m') α) : pairMul x d = pairMulTop x d + pairMulBottom x d := rfl

theorem pairMulBottom_entry {α : Type} [CommRing α] {m m' : Nat}
    (x : Fin 2 → Matrix (Fin 1) (Fin m) α) (d : Matrix (Fin (m + m)) (Fin m') α) (l : Fin m') :
    pairMulBottom x d 0 l = ∑ a : Fin m, x 1 0 a * d (Fin.natAdd m a) l := by
  simp [pairMulBottom, Matrix.mul_apply, bottomRows]

/-- The companions of the bottom digit rows, each paired with its key column's digits: the
companion keys replace the companions, and the bottom half of the key action is removed. -/
theorem companion_rows_bottom {q m m' Dg : Nat} (hn : 0 < n) {sm : ErrorPoly n}
    {g : ExactMatrix q n 1 Dg} {A : Fin 2 → Fin 2 → ExactMatrix q n 1 m}
    {D : Fin 2 → ExactMatrix q n (m + m) m'}
    {B : Fin 2 → Fin m' → Fin m → Fin 2 → Fin 2 → ExactMatrix q n 1 Dg}
    {z : Fin 2 → Fin m' → Fin m → Fin 2 → ExactMatrix q n 1 Dg} {bound : Nat}
    (hz : ∀ j l a, Rel sm (B j l a)
      (fun r ↦ (D j (Fin.natAdd m a) l * reducePoly q n (unitPair sm r)) • g) (z j l a) bound)
    {u : Fin m → Fin 2 → ExactMatrix q n Dg 1}
    (hgu : ∀ a r, g * u a r = fun _ _ ↦ A r 1 0 a)
    {u' : Fin m → Fin 2 → ErrorMatrix n Dg 1} {β : Nat}
    (hu : ∀ a r, u a r = reduceMatrix q n Dg 1 (u' a r)) (hβ : ∀ a r, CoeffBound (u' a r) β)
    {E : Fin 2 → ExactMatrix q n 1 m'} {M : Fin 2 → Fin 2 → ExactMatrix q n 1 m'}
    (hErow : ∀ j l, E j 0 l = (∑ a, (z j l a 0 * u a 0 + z j l a 1 * u a 1)) 0 0)
    (hMrow : ∀ k j l, M k j 0 l = (∑ a, (B j l a k 0 * u a 0 + B j l a k 1 * u a 1)) 0 0) :
    ∃ f : Fin 2 → ErrorMatrix n 1 m',
      (∀ j, CoeffBound (f j) (m * (Dg * n * bound * β + Dg * n * bound * β))) ∧
      ∀ j, E j = reducePoly q n sm • (M 0 j - pairMulBottom (A 0) (D j)) +
        (M 1 j - pairMulBottom (A 1) (D j)) + reduceMatrix q n 1 m' (f j) := by
  choose f hf heq using fun j l ↦ companion_sum (ι := Fin m) hn (hz j l) hgu hu hβ
  refine ⟨fun j _ l ↦ f j l 0 0, fun j row l c ↦ ?_, fun j ↦ ?_⟩
  · simpa using hf j l 0 0 c
  · funext row l
    obtain rfl : row = 0 := Subsingleton.elim _ _
    have h := congrFun (congrFun (heq j l) 0) 0
    rw [hErow j l, h]
    simp only [Matrix.add_apply, Matrix.sub_apply, Matrix.smul_apply, smul_eq_mul, Matrix.sum_apply,
      hMrow, pairMulBottom_entry, reduceMatrix]
    have hdigits :
        ∑ a : Fin m, D j (Fin.natAdd m a) l * (reducePoly q n sm * A 0 1 0 a + A 1 1 0 a) =
        reducePoly q n sm * ∑ a : Fin m, A 0 1 0 a * D j (Fin.natAdd m a) l +
          ∑ a : Fin m, A 1 1 0 a * D j (Fin.natAdd m a) l := by
      rw [Finset.mul_sum, ← Finset.sum_add_distrib]
      exact Finset.sum_congr rfl fun a _ ↦ by ring
    rw [hdigits]
    ring

set_option maxHeartbeats 1000000 in
/-- A Dynamic product whose RHS ciphertext has a public top row `top` and companions only for its
bottom digit rows: the row `c Dec(C) + E` encodes `μ T` under the key
`M + A_{:,0} topRows(Dec(C))`. -/
theorem mul_public_top {q m m' bound : Nat} (hn : 0 < n) (hq1 : 1 < q)
    {backend : BackendContext} {base digits : Int} (layout : RegularLayout q)
    (hlayout : backend.regularLayout q n = some layout) (hexact : layout.droppedModuli = 0)
    {g' : ExactMatrix q n 1 m} (hg' : gadgetMatrixRuns backend base digits g')
    {g : ExactMatrix q n 1 m} (hg : centeredRebaseRuns g' g)
    {sm sp μ : ErrorPoly n} {A : Fin 2 → Fin 2 → ExactMatrix q n 1 m}
    {c : Fin 2 → ExactMatrix q n 1 m} {B : Nat}
    (henc : Rel sm A (fun j ↦ reducePoly q n (((1 : Nat) : Int) * μ * unitPair sp j) • g) c B)
    {top bot : Fin 2 → ExactMatrix q n 1 m'} {D : Fin 2 → ExactMatrix q n (m + m) m'}
    (hD : ∀ j l, ∃ D' : ExactMatrix q n (m + m) 1,
      gadgetDecomposeRuns backend base digits (pairColumn (top j 0 l) (bot j 0 l)) D' ∧
        compactCenteredRebaseRuns bound D' (fun a _ ↦ D j a l))
    {T : Fin 2 → ExactMatrix q n 1 m'} {eB : Fin 2 → ErrorMatrix n 1 m'} {BB : Nat}
    (hrhs : ∀ j, reducePoly q n sp • top j + bot j = T j + reduceMatrix q n 1 m' (eB j))
    (heB : ∀ j, CoeffBound (eB j) BB) {K : Nat} (hμ : SmulGain μ K)
    {E : Fin 2 → ExactMatrix q n 1 m'} {M : Fin 2 → Fin 2 → ExactMatrix q n 1 m'}
    {f : Fin 2 → ErrorMatrix n 1 m'} {F : Nat}
    (hE : ∀ j, E j = reducePoly q n sm • (M 0 j - pairMulBottom (A 0) (D j)) +
      (M 1 j - pairMulBottom (A 1) (D j)) + reduceMatrix q n 1 m' (f j))
    (hf : ∀ j, CoeffBound (f j) F) :
    Rel sm (fun k j ↦ M k j + pairMulTop (A k) (D j)) (fun j ↦ reducePoly q n μ • T j)
      (fun j ↦ pairMul c (D j) + E j) ((m + m) * n * B * bound + F + K * BB) := by
  refine mul_private_dynamic hn hq1 layout hlayout hexact hg' hg henc hD hrhs heB hμ
    (M := fun k j ↦ M k j + pairMulTop (A k) (D j)) (fun j ↦ ?_) hf
  have hcancel : ∀ x y z : ExactMatrix q n 1 m', x + y - (y + z) = x - z := fun _ _ _ ↦ by abel
  rw [hE j, pairMul_split, pairMul_split, hcancel, hcancel]

set_option maxHeartbeats 1000000 in
/-- `Aut_k` before its mask switch, for a switching ciphertext `C_k` whose top row
`σ_k(H_{k⁻¹}^{(0)})` is public: as `MxxPe.aut_switch`, with companions only for the bottom digit
rows and the public rows' term in the key. -/
theorem aut_switch_public {q m bound : Nat} (hn : 0 < n) (hq1 : 1 < q)
    {backend : BackendContext} {base digits : Int} (layout : RegularLayout q)
    (hlayout : backend.regularLayout q n = some layout) (hexact : layout.droppedModuli = 0)
    {g' : ExactMatrix q n 1 m} (hg' : gadgetMatrixRuns backend base digits g')
    {g : ExactMatrix q n 1 m} (hg : centeredRebaseRuns g' g)
    {s μ : ErrorPoly n} {A : Fin 2 → Fin 2 → ExactMatrix q n 1 m}
    {c : Fin 2 → ExactMatrix q n 1 m} {B : Nat}
    (henc : MxxExponentLut.Rel s A (fun j ↦ reducePoly q n ((((1 : Nat) : Int) : ErrorPoly n) * μ *
      unitPair s j) • g) c B)
    {k kinv : Nat} (hk : Odd k) (hkinv : Odd kinv) (hinv : k * kinv % (2 * n) = 1)
    (hcoprime : n.Coprime k) {H : Fin 2 → Fin 2 → ExactMatrix q n 1 m}
    {h : Fin 2 → ExactMatrix q n 1 m} {Bh : Nat}
    (hh : MxxExponentLut.Rel s H (fun j ↦ reducePoly q n ((((1 : Nat) : Int) : ErrorPoly n) *
      unitPair (substitution kinv hkinv s) j) • g) h Bh)
    {top bot : Fin 2 → ExactMatrix q n 1 m}
    (htop : ∀ j, top j = substituteMatrix k hk (H 0 j))
    (hbot : ∀ j, bot j = substituteMatrix k hk (H 1 j) - substituteMatrix k hk (h j))
    {D : Fin 2 → ExactMatrix q n (m + m) m}
    (hD : ∀ j l, ∃ D' : ExactMatrix q n (m + m) 1,
      gadgetDecomposeRuns backend base digits (pairColumn (top j 0 l) (bot j 0 l)) D' ∧
        compactCenteredRebaseRuns bound D' (fun a _ ↦ D j a l))
    {K : Nat} (hμ : SmulGain (substitution k hk μ) K)
    {E : Fin 2 → ExactMatrix q n 1 m} {M : Fin 2 → Fin 2 → ExactMatrix q n 1 m}
    {f : Fin 2 → ErrorMatrix n 1 m} {F : Nat}
    (hE : ∀ j, E j = reducePoly q n (substitution k hk s) •
      (M 0 j - pairMulBottom (fun i ↦ substituteMatrix k hk (A 0 i)) (D j)) +
      (M 1 j - pairMulBottom (fun i ↦ substituteMatrix k hk (A 1 i)) (D j)) +
        reduceMatrix q n 1 m (f j))
    (hf : ∀ j, CoeffBound (f j) F) :
    MxxExponentLut.Rel (substitution k hk s)
      (fun r j ↦ M r j + pairMulTop (fun i ↦ substituteMatrix k hk (A r i)) (D j))
      (fun j ↦ reducePoly q n (substitution k hk μ) • (reducePoly q n (unitPair s j) • g))
      (fun j ↦ pairMul (fun i ↦ substituteMatrix k hk (c i)) (D j) + E j)
      ((m + m) * n * B * bound + F + K * Bh) := by
  have hcancel : ∀ x y z : ExactMatrix q n 1 m, x + y - (y + z) = x - z := fun _ _ _ ↦ by abel
  exact aut_switch hn hq1 layout hlayout hexact hg' hg henc hk hkinv hinv hcoprime hh htop hbot hD
    hμ (M := fun r j ↦ M r j + pairMulTop (fun i ↦ substituteMatrix k hk (A r i)) (D j))
    (fun j ↦ by rw [hE j, pairMul_split, pairMul_split, hcancel, hcancel]) hf

end MxxPe
