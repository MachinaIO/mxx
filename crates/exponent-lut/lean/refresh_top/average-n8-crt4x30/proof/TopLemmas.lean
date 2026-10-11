import TopSetup

/-!
# Lemmas shared by every part

Hash tags of the RHS tops, and the slot and member of a PRF cell.
-/

namespace RefreshTopProof

open Mxx.Primitives MxxRuntime MxxExponentLut

/-- The member tag of a selector RHS top. -/
def memberBytes : Blob := [109, 101, 109, 98, 101, 114]
/-- The component tag. -/
def componentBytes : Blob := [99, 111, 109, 112, 111, 110, 101, 110, 116]

/-- The RHS tops of member `i` of a selector family. -/
noncomputable def rhsTop (model : HashModel) (key : ByteArray) (q m : Nat) (tagPrefix : Blob)
    (i : Int) : Fin 2 → ExactMatrix q N 1 m :=
  fun j ↦ model.sample q N 1 m key (completeHashTag tagPrefix
    [.bytes memberBytes, .u64Le i, .bytes componentBytes, .u64Le (j : Int)])

theorem select_slot (i : Nat) :
    MxxIR.selectInt (Int.fmod (Int.ofNat i) 5) [(0), (1), (2), (3), (4)] = ((i % 5 : Nat) : Int) := by
  have hi : Int.fmod (Int.ofNat i) 5 = Int.ofNat (i % 5) := by
    rw [Int.fmod_eq_emod_of_nonneg _ (by norm_num)]; rfl
  rw [hi]
  have := Nat.mod_lt i (show 0 < 5 by norm_num)
  have h : ([(0), (1), (2), (3), (4)] : List Int) = [0, 1, 2, 3, 4].map Int.ofNat := rfl
  rw [h, selectInt_map_ofNat]
  interval_cases (i % 5) <;> rfl

theorem fdiv_slot (i : Nat) : Int.fdiv (Int.ofNat i) 5 = ((i / 5 : Nat) : Int) := by
  rw [Int.fdiv_eq_ediv_of_nonneg _ (by norm_num)]
  rfl

/-- A loop whose iteration `i` adds `f i` sums `f` over its iterations. -/
theorem iter_sum {α : Type} [AddCommMonoid α] {body : Nat → α → α → Prop} {count : Nat}
    {init final : α} (f : Nat → α)
    (hstep : ∀ i current next, body i current next → next = current + f i)
    (h : MxxIR.IterRuns body count init final) : final = init + ∑ i ∈ Finset.range count, f i := by
  have := MxxIR.IterRuns.invariant
    (Invariant := fun k (s : α) ↦ s = init + ∑ i ∈ Finset.range k, f i)
    (by simp) (fun i current next hcur hbody ↦ by
      show next = init + ∑ j ∈ Finset.range (i + 1), f j
      rw [hstep i current next hbody, hcur, Finset.sum_range_succ, add_assoc])
    h
  exact this

/-- The index `w o + k` of cell `k` of member `o` in a family of `w` cells per member. -/
def cellIndex {members w : Nat} (o : Fin members) (k : Fin w) : Fin (members * w) :=
  ⟨w * o.val + k.val, by
    have ho := o.isLt
    have hk := k.isLt
    calc w * o.val + k.val < w * o.val + w := by omega
      _ = w * (o.val + 1) := by ring
      _ ≤ w * members := Nat.mul_le_mul_left _ ho
      _ = members * w := by ring⟩

/-- The cells `0` and `1` and the loop's cells `2, …, w - 1` of a member sum. -/
theorem sum_cells {α : Type} [AddCommMonoid α] {w : Nat} (f : Fin (w + 2) → α) :
    f 0 + f 1 + ∑ i ∈ Finset.range w, (if h : 2 + i < w + 2 then f ⟨2 + i, h⟩ else 0) =
      ∑ k : Fin (w + 2), f k := by
  rw [Fin.sum_univ_succ, Fin.sum_univ_succ, ← add_assoc]
  congr 1
  rw [← Fin.sum_univ_eq_sum_range]
  apply Finset.sum_congr rfl
  intro i _
  rw [dif_pos (by omega)]
  congr 1
  ext
  simp [Fin.succ]
  omega

/-- Member `k` of a family, zero past it. -/
noncomputable def cellAt {α : Type} [Zero α] {w : Nat} (f : Fin w → α) (k : Nat) : α :=
  if h : k < w then f ⟨k, h⟩ else 0

/-- A member read at a shifted index. -/
theorem familyGet_cellAt {α : Type} [Zero α] {w : Nat} {f : Fin w → α} {i : Nat} {out : α}
    (h : familyGetDynamic f (2 + Int.ofNat i) out) : out = cellAt f (2 + i) := by
  obtain ⟨hk, hout⟩ := familyGet_eq (k := 2 + i) (by
    rw [show (Int.ofNat (2 + i)) = 2 + Int.ofNat i by simp only [Int.ofNat_eq_natCast]; push_cast; ring]
    exact h)
  rw [hout, cellAt, dif_pos hk]

/-- The six blocks `(A₀₀, A₀₁, A₁₀, A₁₁, c₀, c₁)` of an encoding. -/
abbrev Blocks (q m : Nat) :=
  ExactMatrix q N 1 m × ExactMatrix q N 1 m × ExactMatrix q N 1 m × ExactMatrix q N 1 m ×
    ExactMatrix q N 1 m × ExactMatrix q N 1 m × Unit

/-- The four key blocks `(A₀₀, A₀₁, A₁₀, A₁₁)` of an encoding. -/
abbrev Keys (q m : Nat) :=
  ExactMatrix q N 1 m × ExactMatrix q N 1 m × ExactMatrix q N 1 m × ExactMatrix q N 1 m × Unit

theorem slice_row {q rows columns : Nat} {A : ExactMatrix q N rows columns}
    {x : ExactMatrix q N 1 columns} {r r' c c' : Int}
    (h : sliceMatrix A r r' c c' x) (hc : c = 0) (hr : r.toNat < rows) :
    x = fun _ k ↦ A ⟨r.toNat, hr⟩ k := by
  obtain ⟨_, _, _, _, _, _, _, _, hx⟩ := h
  funext row k
  obtain rfl : row = 0 := Subsingleton.elim _ _
  rw [hx 0 k (by simpa using hr) (by subst hc; simpa using k.isLt)]
  subst hc
  simp

theorem slice_entry {q columns : Nat} {D : ExactMatrix q N 1 columns} {x : ExactMatrix q N 1 1}
    {a b : Int} (h : sliceMatrix D 0 1 a b x) (ha : a.toNat < columns) :
    x 0 0 = D 0 ⟨a.toNat, ha⟩ := by
  obtain ⟨_, _, _, _, _, _, _, _, hx⟩ := h
  rw [hx 0 0 (by simp) (by simpa using ha)]
  simp

end RefreshTopProof
