import ExponentRuntime
import ExponentEncoding

/-!
# The PBC sparse inner product for any parameter set

A multi-member sparse-LWR PRF (`SparseLwrPrf::build` with a key shape) runs `B` buckets. Bucket
`b` has `M W` cells `f = W o + j` (member `o`, slot `j`): cell `f` reads member `tab_b f` of the
label family `a` (entry `ν` is the zero member, `a_ν = 0`), multiplies member `o`'s running state
by `x^(a (tab_b f))` and weights it by selector bit `s_b f`; member `o`'s next state is the sum of
its row of `W` cells. With one-hot rows consistent with the table (row `o` of bucket `b` has its
one at the slot `sel_b o` whose table entry is the key's coordinate `key_b o`), the state after
`B` buckets is `x^(Σ_b a (key_b o))`, and `Σ_b a (key_b o)` is the inner product `⟨a, k_o⟩` of the
label with the key's sparse vector (`k_o[i]` the number of buckets whose coordinate is `i`).

Nothing here depends on `N`, `ν`, `h`, `W`, `B` or `M`:

* `cellAt`: the cell index `W o + j` (the toy's `cellIndex`).
* `onehot_sum`, `onehot_exp_sum`, `bucket_step`: one row of one bucket.
* `pbcState`, `pbcState_eq`: the state after any number of buckets.
* `bucket_sum_inner`: the accumulated exponent is the sparse inner product.
* `pow_mod_period`, `Xe_stride_mod`: the exponent only matters modulo the exponent modulus.
* `onehot_collapse_gen`: the matrix-level row sum of the encodings' targets.
-/

namespace RefreshTopProof.Gen

open Mxx.Primitives MxxRuntime MxxExponentLut
open scoped BigOperators

/-- The index `W o + j` of cell `j` of member `o` in a family of `W` cells per member. -/
def cellAt {M W : Nat} (o : Fin M) (j : Fin W) : Fin (M * W) :=
  ⟨W * o.val + j.val, by
    have ho := o.isLt
    have hj := j.isLt
    calc W * o.val + j.val < W * o.val + W := by omega
      _ = W * (o.val + 1) := by ring
      _ ≤ W * M := Nat.mul_le_mul_left _ ho
      _ = M * W := by ring⟩

theorem cellAt_mod {M W : Nat} (o : Fin M) (j : Fin W) : (cellAt o j).val % W = j.val := by
  simp only [cellAt]
  rw [Nat.add_comm, Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt j.isLt]

theorem cellAt_div {M W : Nat} (o : Fin M) (j : Fin W) : (cellAt o j).val / W = o.val := by
  simp only [cellAt]
  have hW : 0 < W := Nat.lt_of_le_of_lt (Nat.zero_le _) j.isLt
  rw [Nat.add_comm, Nat.add_mul_div_left _ _ hW, Nat.div_eq_of_lt j.isLt, Nat.zero_add]

/-! ## One row of one bucket -/

/-- A one-hot row sums to the selected cell. -/
theorem onehot_sum {R : Type} [NonAssocSemiring R] {W : Nat} (g : Fin W → R) (s : Fin W → R)
    (sel : Fin W) (hs : ∀ j, s j = if j = sel then 1 else 0) :
    ∑ j, g j * s j = g sel := by
  simp only [hs, mul_ite, mul_one, mul_zero, Finset.sum_ite_eq', Finset.mem_univ, if_true]

/-- The exponent a one-hot row consistent with the table reads: the label entry at the key's
coordinate. -/
theorem onehot_exp_sum {R : Type} [NonAssocSemiring R] {M W ν : Nat} (a : Fin (ν + 1) → R)
    (tab : Fin (M * W) → Fin (ν + 1)) (s : Fin (M * W) → R) (sel : Fin M → Fin W)
    (key : Fin M → Fin (ν + 1)) (hsel : ∀ o j, s (cellAt o j) = if j = sel o then 1 else 0)
    (htab : ∀ o, tab (cellAt o (sel o)) = key o) (o : Fin M) :
    ∑ j : Fin W, s (cellAt o j) * a (tab (cellAt o j)) = a (key o) := by
  simp only [hsel, ite_mul, one_mul, zero_mul, Finset.sum_ite_eq', Finset.mem_univ, if_true, htab]

/-- One bucket step of member `o`: the row sum of the cells `x^(a (tab f)) · state · s f` is
`x^(a (key o)) · state`. -/
theorem bucket_step {R : Type} [CommSemiring R] {M W ν : Nat} (x : R) (a : Fin (ν + 1) → Nat)
    (tab : Fin (M * W) → Fin (ν + 1)) (s : Fin (M * W) → R) (sel : Fin M → Fin W)
    (key : Fin M → Fin (ν + 1)) (hsel : ∀ o j, s (cellAt o j) = if j = sel o then 1 else 0)
    (htab : ∀ o, tab (cellAt o (sel o)) = key o) (state : Fin M → R) (o : Fin M) :
    ∑ j : Fin W, x ^ a (tab (cellAt o j)) * state o * s (cellAt o j) = x ^ a (key o) * state o := by
  rw [onehot_sum (fun j ↦ x ^ a (tab (cellAt o j)) * state o) (fun j ↦ s (cellAt o j)) (sel o)
    (fun j ↦ hsel o j), htab]

/-! ## All buckets -/

/-- The state of every member after the first `b` of `B` buckets (start `x^0 = 1`, the private
input `mono(0)`); past `B` buckets it stays. -/
def pbcState {R : Type} [CommSemiring R] {B M W ν : Nat} (x : R) (a : Fin (ν + 1) → Nat)
    (tab : Fin B → Fin (M * W) → Fin (ν + 1)) (s : Fin B → Fin (M * W) → R) : Nat → Fin M → R
  | 0, _ => 1
  | b + 1, o => if h : b < B then
      ∑ j : Fin W, x ^ a (tab ⟨b, h⟩ (cellAt o j)) * pbcState x a tab s b o * s ⟨b, h⟩ (cellAt o j)
    else pbcState x a tab s b o

/-- The state after the first `b ≤ B` buckets. -/
theorem pbcState_prefix {R : Type} [CommSemiring R] {B M W ν : Nat} (x : R)
    (a : Fin (ν + 1) → Nat) (tab : Fin B → Fin (M * W) → Fin (ν + 1)) (s : Fin B → Fin (M * W) → R)
    (sel : Fin B → Fin M → Fin W) (key : Fin B → Fin M → Fin (ν + 1))
    (hsel : ∀ b o j, s b (cellAt o j) = if j = sel b o then 1 else 0)
    (htab : ∀ b o, tab b (cellAt o (sel b o)) = key b o) (o : Fin M) (b : Nat) (hb : b ≤ B) :
    pbcState x a tab s b o =
      x ^ (∑ i ∈ Finset.range b, if h : i < B then a (key ⟨i, h⟩ o) else 0) := by
  induction b with
  | zero => simp [pbcState]
  | succ b ih =>
    have hb' : b < B := by omega
    rw [pbcState, dif_pos hb', ih (by omega),
      bucket_step x a (tab ⟨b, hb'⟩) (s ⟨b, hb'⟩) (sel ⟨b, hb'⟩) (key ⟨b, hb'⟩)
        (hsel ⟨b, hb'⟩) (htab ⟨b, hb'⟩) (fun _ ↦ _) o,
      Finset.sum_range_succ, dif_pos hb', pow_add, mul_comm]

/-- **The PBC accumulation.** After all `B` buckets, member `o`'s state is
`x^(Σ_b a (key_b o))`. -/
theorem pbcState_eq {R : Type} [CommSemiring R] {B M W ν : Nat} (x : R) (a : Fin (ν + 1) → Nat)
    (tab : Fin B → Fin (M * W) → Fin (ν + 1)) (s : Fin B → Fin (M * W) → R)
    (sel : Fin B → Fin M → Fin W) (key : Fin B → Fin M → Fin (ν + 1))
    (hsel : ∀ b o j, s b (cellAt o j) = if j = sel b o then 1 else 0)
    (htab : ∀ b o, tab b (cellAt o (sel b o)) = key b o) (o : Fin M) :
    pbcState x a tab s B o = x ^ (∑ b : Fin B, a (key b o)) := by
  rw [pbcState_prefix x a tab s sel key hsel htab o B le_rfl, Finset.sum_range]
  congr 1
  exact Finset.sum_congr rfl fun i _ ↦ by rw [dif_pos i.isLt]

/-- **The sparse inner product.** With the zero member `a_ν = 0`, the accumulated exponent is
`Σ_{i < ν} a_i k[i]`, where `k[i]` counts the buckets whose coordinate is `i`. -/
theorem bucket_sum_inner {B ν : Nat} (a : Fin (ν + 1) → Nat) (ha : a (Fin.last ν) = 0)
    (key : Fin B → Fin (ν + 1)) :
    ∑ b, a (key b) =
      ∑ i : Fin ν, a i.castSucc * (Finset.univ.filter fun b ↦ key b = i.castSucc).card := by
  rw [← Finset.sum_fiberwise_of_maps_to (s := Finset.univ) (t := Finset.univ) (g := key)
    (fun _ _ ↦ Finset.mem_univ _)]
  have hfib : ∀ i : Fin (ν + 1), ∑ b ∈ Finset.univ.filter (fun b ↦ key b = i), a (key b) =
      a i * (Finset.univ.filter fun b ↦ key b = i).card := by
    intro i
    rw [Finset.sum_congr rfl (fun b hb ↦ by rw [(Finset.mem_filter.1 hb).2]), Finset.sum_const,
      smul_eq_mul, mul_comm]
  simp only [hfib]
  rw [Fin.sum_univ_castSucc, ha, zero_mul, add_zero]

/-- Both together: the state is `x^⟨a, k_o⟩`. -/
theorem pbcState_inner {R : Type} [CommSemiring R] {B M W ν : Nat} (x : R) (a : Fin (ν + 1) → Nat)
    (ha : a (Fin.last ν) = 0)
    (tab : Fin B → Fin (M * W) → Fin (ν + 1)) (s : Fin B → Fin (M * W) → R)
    (sel : Fin B → Fin M → Fin W) (key : Fin B → Fin M → Fin (ν + 1))
    (hsel : ∀ b o j, s b (cellAt o j) = if j = sel b o then 1 else 0)
    (htab : ∀ b o, tab b (cellAt o (sel b o)) = key b o) (o : Fin M) :
    pbcState x a tab s B o = x ^ (∑ i : Fin ν,
      a i.castSucc * (Finset.univ.filter fun b ↦ key b o = i.castSucc).card) := by
  rw [pbcState_eq x a tab s sel key hsel htab o, bucket_sum_inner a ha (fun b ↦ key b o)]

/-! ## The exponent modulus -/

/-- A power of an element of period `Q` only depends on the exponent modulo `Q`. -/
theorem pow_mod_period {R : Type} [Monoid R] {x : R} {Q : Nat} (hx : x ^ Q = 1) (k : Nat) :
    x ^ k = x ^ (k % Q) := by
  conv_lhs => rw [← Nat.div_add_mod k Q, pow_add, pow_mul, hx, one_pow, one_mul]

/-- The PRF's exponent layout: `X^(s k)` with `s Q = 2 n` only depends on `k` modulo `Q`. -/
theorem Xe_stride_mod {n s Q : Nat} (hsQ : s * Q = 2 * n) (k : Nat) :
    Xe (n := n) (s * k) = Xe (n := n) (s * (k % Q)) := by
  have hx : (Xe (n := n) s) ^ Q = 1 := by
    rw [Xe, ← pow_mul, hsQ, Nat.mul_comm, pow_mul, Negacyclic.root_pow_n]; norm_num
  simp only [Xe, pow_mul]
  exact pow_mod_period hx k

/-- The accumulated state in the ring: `X^(s Σ_b a (key_b o))`, and modulo the exponent modulus. -/
theorem pbcState_Xe {n s Q B M W ν : Nat} (hsQ : s * Q = 2 * n) (a : Fin (ν + 1) → Nat)
    (tab : Fin B → Fin (M * W) → Fin (ν + 1)) (sb : Fin B → Fin (M * W) → ErrorPoly n)
    (sel : Fin B → Fin M → Fin W) (key : Fin B → Fin M → Fin (ν + 1))
    (hsel : ∀ b o j, sb b (cellAt o j) = if j = sel b o then 1 else 0)
    (htab : ∀ b o, tab b (cellAt o (sel b o)) = key b o) (o : Fin M) :
    pbcState (Xe (n := n) s) a tab sb B o = Xe (n := n) (s * ((∑ b : Fin B, a (key b o)) % Q)) := by
  rw [pbcState_eq _ a tab sb sel key hsel htab o, ← Xe_stride_mod hsQ]
  simp only [Xe, pow_mul]

/-! ## The row sum of the encodings' targets -/

/-- The cells of a member with one-hot selectors sum to the selected cell (any ring dimension). -/
theorem onehot_collapse_gen {n q m w : Nat} (h : ExactMatrix q n 1 m) (sp : ErrorPoly n) (p : Nat)
    (β : Fin w → ErrorPoly n) (s : Fin w → Int) (sel : Fin w)
    (hs : ∀ k, s k = if k = sel then 1 else 0) :
    (∑ k : Fin w, fun j ↦ reducePoly q n (((p : Nat) : Int) * (β k * ((s k : Int) : ErrorPoly n)) *
      unitPair sp j) • h) = fun j ↦ reducePoly q n (((p : Nat) : Int) * β sel * unitPair sp j) • h := by
  rw [Finset.sum_eq_single sel]
  · simp [hs]
  · intro k _ hk
    funext j
    simp [hs, hk]
  · simp

/-- The row of member `o` of a bucket whose cells read the table: the targets
`p β(tab f) s_f` sum to `p β(key o)`. -/
theorem onehot_collapse_table {n q m M W ν : Nat} (h : ExactMatrix q n 1 m) (sp : ErrorPoly n)
    (p : Nat) (β : Fin (ν + 1) → ErrorPoly n) (tab : Fin (M * W) → Fin (ν + 1))
    (s : Fin (M * W) → Int) (sel : Fin M → Fin W) (key : Fin M → Fin (ν + 1))
    (hsel : ∀ o j, s (cellAt o j) = if j = sel o then 1 else 0)
    (htab : ∀ o, tab (cellAt o (sel o)) = key o) (o : Fin M) :
    (∑ j : Fin W, fun i ↦ reducePoly q n (((p : Nat) : Int) *
      (β (tab (cellAt o j)) * ((s (cellAt o j) : Int) : ErrorPoly n)) * unitPair sp i) • h) =
      fun i ↦ reducePoly q n (((p : Nat) : Int) * β (key o) * unitPair sp i) • h := by
  rw [onehot_collapse_gen h sp p (fun j ↦ β (tab (cellAt o j))) (fun j ↦ s (cellAt o j)) (sel o)
    (hsel o), htab]

end RefreshTopProof.Gen
