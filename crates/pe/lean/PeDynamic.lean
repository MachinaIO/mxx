import ExponentDynamic
import ExponentLut
import ExponentDecode
import ExponentFunctional

/-!
# The Dynamic LUT branch of the predicate encryption scheme

A Dynamic LUT `L` on an encoding `c` of `X^u` under the secret `s = (s̄, 1)` is
`Σ_{k ∈ H} SMul_{D_k}(Aut_k(c))`. For `k ≠ 1`:

- the routed encoding `σ_k(c)` is under `s_k = (σ_k(s̄), 1)`;
- the switching ciphertext `C_k = σ_k(H_{k⁻¹}) - [0; σ_k(h_{k⁻¹})]`, derived from the helper
  `h_{k⁻¹} = s H_{k⁻¹} - s_{k⁻¹} G + e`, satisfies `s_k C_k = s G - σ_k(e)`
  (`switching_rhs`), so the Dynamic product of `σ_k(c)` by `C_k` with its companions is an
  encoding of `σ_k(X^u)` under `s_k`, with the payload gadget of `s` (`aut_switch`);
- the helper `h_k = s H_k - s_k G + e_h` acting through the digits of the switched key `T_k`
  moves the mask back to `s` (`mask_switch`);
- the public scalar action `V_D = Dec(D g)`, block-diagonal on the two components, scales the
  payload by `D` (`Rel.blockAction`, `gadget_scalar_action`).
-/

namespace MxxPe

open Mxx.Primitives MxxRuntime MxxExponentLut

variable {n : Nat}

/-- The helper of `k⁻¹`, routed by `σ_k`, opens the switching ciphertext `C_k`:
`σ_k(s̄) top_j + bot_j = (s G)_j - σ_k(e)_j` for `top = σ_k(H⁰)` and `bot = σ_k(H¹) - σ_k(h)`. -/
theorem switching_rhs {q m : Nat} (hn : 0 < n) {backend : BackendContext} {base digits : Int}
    {g' : ExactMatrix q n 1 m} (hg' : gadgetMatrixRuns backend base digits g')
    {g : ExactMatrix q n 1 m} (hg : centeredRebaseRuns g' g)
    {k kinv : Nat} (hk : Odd k) (hkinv : Odd kinv) (hinv : k * kinv % (2 * n) = 1)
    (hcoprime : n.Coprime k) {s : ErrorPoly n} {H : Fin 2 → Fin 2 → ExactMatrix q n 1 m}
    {h : Fin 2 → ExactMatrix q n 1 m} {Bh : Nat}
    (hh : MxxExponentLut.Rel s H (fun j ↦ reducePoly q n ((((1 : Nat) : Int) : ErrorPoly n) *
      unitPair (substitution kinv hkinv s) j) • g) h Bh) :
    ∃ e : Fin 2 → ErrorMatrix n 1 m, (∀ j, CoeffBound (e j) Bh) ∧ ∀ j,
      reducePoly q n (substitution k hk s) • substituteMatrix k hk (H 0 j) +
          (substituteMatrix k hk (H 1 j) - substituteMatrix k hk (h j)) =
        reducePoly q n (unitPair s j) • g + reduceMatrix q n 1 m (e j) := by
  obtain ⟨e, he, hc⟩ := hh.substitute hn hk hcoprime
  refine ⟨fun j ↦ -e j, fun j ↦ coeffBound_neg (he j), fun j ↦ ?_⟩
  have htarget : substituteMatrix k hk (reducePoly q n ((((1 : Nat) : Int) : ErrorPoly n) *
      unitPair (substitution kinv hkinv s) j) • g) = reducePoly q n (unitPair s j) • g := by
    have h1 := substituteMatrix_target (n := n) (q := q) hn hg' hg hk 1
      (unitPair (substitution kinv hkinv s) j) 1
    simp only [one_mul, mul_one, Nat.cast_one, Int.cast_one, map_one] at h1 ⊢
    rw [h1]
    congr 2
    fin_cases j
    · simp [unitPair, substitution_inverse hk hkinv hinv]
    · simp [unitPair]
  have hj := hc j
  simp only [htarget] at hj
  have hneg : reduceMatrix q n 1 m (-e j) = -reduceMatrix q n 1 m (e j) := by
    funext row column
    simp [reduceMatrix]
  rw [hj, hneg]
  abel

set_option maxHeartbeats 1000000 in
/-- `Aut_k` before its mask switch: the routed encoding `σ_k(c)` times the digits `D` of the
switching ciphertext `C_k` (columns of `top = σ_k(H⁰)`, `bot = σ_k(H¹) - σ_k(h)`, from the helper
of `k⁻¹`), plus the companion row `E` of the automorphism package, is an encoding under the
routed mask `σ_k(s)` and the companion keys `M` of the payload `σ_k(μ)` against the gadget of
`s`. -/
theorem aut_switch {q m bound : Nat} (hn : 0 < n) (hq1 : 1 < q)
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
      (M 0 j - pairMul (fun i ↦ substituteMatrix k hk (A 0 i)) (D j)) +
      (M 1 j - pairMul (fun i ↦ substituteMatrix k hk (A 1 i)) (D j)) + reduceMatrix q n 1 m (f j))
    (hf : ∀ j, CoeffBound (f j) F) :
    MxxExponentLut.Rel (substitution k hk s) M
      (fun j ↦ reducePoly q n (substitution k hk μ) • (reducePoly q n (unitPair s j) • g))
      (fun j ↦ pairMul (fun i ↦ substituteMatrix k hk (c i)) (D j) + E j)
      ((m + m) * n * B * bound + F + K * Bh) := by
  have hsub := henc.substitute hn hk hcoprime
  have htarget : (fun j ↦ substituteMatrix k hk (reducePoly q n ((((1 : Nat) : Int) : ErrorPoly n) *
      μ * unitPair s j) • g)) = fun j ↦ reducePoly q n ((((1 : Nat) : Int) : ErrorPoly n) *
        substitution k hk μ * unitPair (substitution k hk s) j) • g := by
    funext j
    rw [substituteMatrix_target hn hg' hg hk 1 μ (unitPair s j)]
    congr 3
    fin_cases j <;> simp [unitPair]
  rw [htarget] at hsub
  obtain ⟨eB, heB, hrhs⟩ := switching_rhs hn hg' hg hk hkinv hinv hcoprime hh
  exact mul_private_dynamic hn hq1 layout hlayout hexact hg' hg hsub hD
    (T := fun j ↦ reducePoly q n (unitPair s j) • g) (eB := eB)
    (fun j ↦ by rw [htop j, hbot j]; exact hrhs j) heB hμ hE hf

/-- The mask switch of `Aut_k`: the helper `h = s H - s' G + e` of `k`, acting through the digits
`Dt` of the columns of the switched key `T`, moves an encoding under the routed mask `s'` and the
key `T` to the mask `s` and the key `H Dt`, with the same target. -/
theorem mask_switch {q m m' bound : Nat} (hn : 0 < n) (hq1 : 1 < q)
    {backend : BackendContext} {base digits : Int} (layout : RegularLayout q)
    (hlayout : backend.regularLayout q n = some layout) (hexact : layout.droppedModuli = 0)
    {g' : ExactMatrix q n 1 m} (hg' : gadgetMatrixRuns backend base digits g')
    {g : ExactMatrix q n 1 m} (hg : centeredRebaseRuns g' g)
    {s s' : ErrorPoly n} {H : Fin 2 → Fin 2 → ExactMatrix q n 1 m}
    {h : Fin 2 → ExactMatrix q n 1 m} {Bh : Nat}
    (hh : MxxExponentLut.Rel s H (fun j ↦ reducePoly q n ((((1 : Nat) : Int) : ErrorPoly n) * unitPair s' j) • g)
      h Bh)
    {T : Fin 2 → Fin 2 → ExactMatrix q n 1 m'} {t r : Fin 2 → ExactMatrix q n 1 m'} {Br : Nat}
    (hr : MxxExponentLut.Rel s' T t r Br) {Dt : Fin 2 → ExactMatrix q n (m + m) m'}
    (hDt : ∀ j l, ∃ D' : ExactMatrix q n (m + m) 1,
      gadgetDecomposeRuns backend base digits (pairColumn (T 0 j 0 l) (T 1 j 0 l)) D' ∧
        compactCenteredRebaseRuns bound D' (fun a _ ↦ Dt j a l)) :
    MxxExponentLut.Rel s (fun k j ↦ pairMul (H k) (Dt j)) t (fun j ↦ pairMul h (Dt j) + r j)
      ((m + m) * n * Bh * bound + Br) := by
  choose d hd using fun j ↦ columns_bounded fun l ↦ (hDt j l).choose_spec.2.2.2.2
  have hDd : ∀ j, Dt j = reduceMatrix q n (m + m) m' (d j) := fun j ↦ (hd j).1
  have hact := hh.action hn d (fun j ↦ (hd j).2)
  simp only [← hDd] at hact
  obtain ⟨eh, heh, hY⟩ := hact
  obtain ⟨er, her, hX⟩ := hr
  have hq : q = 1 * q := (one_mul q).symm
  have hgad : ∀ j, pairMul (fun i ↦ reducePoly q n ((((1 : Nat) : Int) : ErrorPoly n) *
      unitPair s' i) • g) (Dt j) = reducePoly q n s' • T 0 j + T 1 j := by
    intro j
    rw [gadget_action_columns hq hq1 hn layout hlayout hexact hg' hg (hDt j), scaleUpMatrix_one hq1 hn]
    simp [unitPair]
  refine ⟨fun j ↦ eh j + er j, fun j ↦ coeffBound_add (heh j) (her j), fun j ↦ ?_⟩
  have hYj := hY j
  have hXj := hX j
  dsimp only at hYj hXj ⊢
  rw [hYj, hXj, hgad j, reduceMatrix_add]
  abel

/-! ## The public scalar action -/

/-- A right action by one small `m × m'` matrix on each component: the error bound gains the
factor `m n β`. -/
theorem Rel.blockAction {q m m' : Nat} (hn : 0 < n) {s : ErrorPoly n}
    {A : Fin 2 → Fin 2 → ExactMatrix q n 1 m} {t c : Fin 2 → ExactMatrix q n 1 m} {B β : Nat}
    (h : MxxExponentLut.Rel s A t c B) (v : ErrorMatrix n m m') (hv : CoeffBound v β) :
    MxxExponentLut.Rel s (fun k j ↦ A k j * reduceMatrix q n m m' v)
      (fun j ↦ t j * reduceMatrix q n m m' v) (fun j ↦ c j * reduceMatrix q n m m' v)
      (m * n * B * β) := by
  obtain ⟨e, he, hc⟩ := h
  refine ⟨fun j ↦ e j * v, fun j ↦ coeffBound_mul hn (he j) hv, fun j ↦ ?_⟩
  dsimp only
  rw [hc j, reduceMatrix_mul]
  simp only [Matrix.add_mul, Matrix.sub_mul, Matrix.smul_mul]

/-- The digits of a gadget row reconstruct it and are at most `base / 2`. -/
theorem decompose_row {q D m' : Nat} (hq : 1 < q) (hn : 0 < n) {backend : BackendContext}
    {base digits : Int} (layout : RegularLayout q)
    (hlayout : backend.regularLayout q n = some layout) (hexact : layout.droppedModuli = 0)
    {g : ExactMatrix q n 1 D} (hg : gadgetMatrixRuns backend base digits g)
    {x : ExactMatrix q n 1 m'} {u : ExactMatrix q n D m'}
    (hu : gadgetDecomposeRuns backend base digits x u) :
    g * u = x ∧ PreimageWithin u (layout.base / 2) := by
  obtain ⟨layout₁, hl₁, _, _, hw₁, hg⟩ := hg
  obtain ⟨layout₂, hl₂, _, _, hw₂, hu, _⟩ := hu
  rw [hlayout] at hl₁ hl₂
  cases hl₁
  cases hl₂
  have hd : D = layout.digitCount := by simpa using hw₁
  subst hd
  subst hg hu
  constructor
  · have hrec := regularGadgetMatrix_reconstruct (rows := 1) layout x hq hn hexact
    funext row column
    rw [← congrFun (congrFun hrec row) column]
    simp only [Matrix.mul_apply, castMatrixColumns_apply, castMatrixRows_apply]
    exact Fintype.sum_equiv (finCongr (by simp)) _ _ (fun _ ↦ rfl)
  · obtain ⟨w, hw, hb⟩ := regularDecomposeMatrix_bounded (rows := 1) layout x hq hn
    refine ⟨fun row column ↦ w (Fin.cast (by simp) row) column, ?_, fun row column ↦ hb _ _⟩
    funext row column
    simp only [castMatrixRows_apply, hw, reduceMatrix]

end MxxPe
