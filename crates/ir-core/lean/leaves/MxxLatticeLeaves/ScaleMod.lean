import MxxLatticeLeaves.Code.ScaleMod
import MxxLatticeLeaves.Math.Shoup
import MxxLatticeLeaves.Math.Words
import MxxCore.Glue
import MxxCore.Leaf
import MxxCore.Sym

/-!
# The specification of the leaf `lattice.scale_mod`

For any program `P` that holds the leaf's code at instruction `c` (`Code.ScaleMod.At P c`), from
that entry, with `a0` pointing at the descriptor `[n, q, k, a, o, scratch]`, the leaf reaches its
`ret` (instruction `c + 33`) within `524 + 14 n` steps, having written `k A i mod q` to the
doubleword at `o + 8 i` for every `i < n`, changed no other doubleword, and kept every register
the calling convention saves. The input is canonical residues (`A i < q`), the scalar `k < q`,
and `q < 2^63`; the arrays are aligned and confined, and apart.

The leaf first computes `k' = floor(k 2^64 / q)` by 64 steps of binary long division (instructions
8 to 17), then per residue Shoup's product (`MxxLatticeLeaves.Math.Shoup`).
-/

namespace MxxLatticeLeaves.ScaleMod

open MathcalA RiscvZkvm.Rv64 MxxLatticeLeaves.Code.ScaleMod MxxLatticeLeaves.Math.Shoup
  MxxLatticeLeaves.Math.Words

/-- The registers a leaf keeps: `ra`, `sp`, and `s0`–`s11`. -/
def Saved (s0 s : MachineState) : Prop :=
  ∀ r ∈ [Reg.x1, .x2, .x8, .x9, .x18, .x19, .x20, .x21, .x22, .x23, .x24, .x25, .x26, .x27],
    s.getReg r = s0.getReg r

/-- What the leaf assumes on entry. -/
structure Pre (P : Prog) (d n q k a o : Nat) (A : Nat → Nat) (s : MachineState) : Prop where
  hd : s.getReg .x10 = W d
  dn : s.getMem (W d) = W n
  dq : s.getMem (W (d + 8)) = W q
  dk : s.getMem (W (d + 16)) = W k
  da : s.getMem (W (d + 24)) = W a
  dout : s.getMem (W (d + 32)) = W o
  hA : ∀ i < n, s.getMem (W (a + 8 * i)) = W (A i)
  hAq : ∀ i < n, A i < q
  hk : k < q
  hq : q < 2 ^ 63
  hmem : BASE + P.mem < 2 ^ 64
  dal : d % 8 = 0
  aal : a % 8 = 0
  oal : o % 8 = 0
  drange : BASE ≤ d ∧ d + 40 ≤ BASE + P.mem
  arange : BASE ≤ a ∧ a + 8 * n ≤ BASE + P.mem
  orange : BASE + P.bin.length ≤ o ∧ o + 8 * n ≤ BASE + P.mem
  disjA : o + 8 * n ≤ a ∨ a + 8 * n ≤ o

/-- What holds at `ret`. -/
def Post (s0 : MachineState) (n q k o : Nat) (A : Nat → Nat) (s : MachineState) : Prop :=
  (∀ i < n, s.getMem (W (o + 8 * i)) = W (k * A i % q)) ∧
    (∀ x, (∀ i < n, x ≠ W (o + 8 * i)) → s.getMem x = s0.getMem x) ∧ Saved s0 s

/-- The division's head (instruction 8) after `i` steps: the remainder and quotient of `k 2^i`. -/
def DivInv (s0 : MachineState) (n q k a o : Nat) (i : Nat) (s : MachineState) : Prop :=
  s.getReg .x5 = W n ∧ s.getReg .x6 = W q ∧ s.getReg .x13 = W k ∧ s.getReg .x7 = W a ∧
    s.getReg .x12 = W o ∧ s.getReg .x16 = W (64 - i) ∧
    (∃ qt r, s.getReg .x14 = W qt ∧ s.getReg .x15 = W r ∧ k * 2 ^ i = qt * q + r ∧ r < q ∧
      qt < 2 ^ i) ∧
    (∀ x, s.getMem x = s0.getMem x) ∧ Saved s0 s

/-- The residue loop's head (instruction 18) after `j` residues, with `k' = kq`. -/
def Inv (s0 : MachineState) (n q k a o kq : Nat) (A : Nat → Nat) (j : Nat) (s : MachineState) :
    Prop :=
  s.getReg .x5 = W (n - j) ∧ s.getReg .x6 = W q ∧ s.getReg .x13 = W k ∧ s.getReg .x14 = W kq ∧
    s.getReg .x7 = W (a + 8 * j) ∧ s.getReg .x12 = W (o + 8 * j) ∧
    (∀ i < j, s.getMem (W (o + 8 * i)) = W (k * A i % q)) ∧
    (∀ x, (∀ i < j, x ≠ W (o + 8 * i)) → s.getMem x = s0.getMem x) ∧ Saved s0 s

-- The goals every segment leaves: address bounds, kept memory, and kept registers.
set_option hygiene false in
macro "leaf_close" : tactic => `(tactic| first
  | (simp only [BASE] at *; omega)
  | (intro x; simp; done)
  | (intro r hr
     have hs := hsaved r hr
     simp only [List.mem_cons, List.mem_nil_iff, or_false] at hr
     rcases hr with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
       simpa using hs)
  | bv_omega
  | (simp; done))

/-- The loads and the division's setup: instructions 0 to 8. -/
theorem prologue (P : Prog) (c : Nat) (hcode : At P c) (_hc : BASE + 4 * (c + 34) < 2 ^ 64)
    (d n q k a o : Nat) (A : Nat → Nat) (s : MachineState) (h : Pre P d n q k a o A s) :
    P.Spec c s 8 (c + 8) (DivInv s n q k a o 0) := by
  obtain ⟨hd, dn, dq, dk, da, dout, -, -, hk, hq, hmem, dal, -, -, ⟨d1, d2⟩, -, -, -⟩ := h
  have hsaved : Saved s s := fun _ _ => rfl
  cases hcode
  mrunAt [DivInv, Saved, hd, dn, dq, dk, da, dout, W_add]
  all_goals and_intros
  all_goals first
    | exact ⟨0, k, rfl, by simp, by simp, hk, by simp⟩
    | (intro _; rfl)
    | leaf_close

/-- The division's head in step `i < 64`: instruction 8 to 10. -/
theorem divBranch (P : Prog) (c : Nat) (hcode : At P c) (_hc : BASE + 4 * (c + 34) < 2 ^ 64)
    (n q k a o : Nat) (s0 : MachineState) (i : Nat) (s : MachineState) (hi : i < 64)
    (hI : DivInv s0 n q k a o i s) : P.Spec (c + 8) s 1 (c + 10) (fun t => t = s) := by
  obtain ⟨-, -, -, -, -, x16, -⟩ := hI
  cases hcode
  mrunAt [x16]
  all_goals first
    | rfl
    | (exfalso
       have h0 : W (64 - i) = W 0 := by simpa using hbr
       have := Mem.ofNat_inj (by omega) (by omega) h0
       omega)

/-- After the doubling and the subtraction of step `i`: the remainder and quotient of `k 2^(i+1)`,
the counter not yet decremented. -/
def DivMid (s0 : MachineState) (n q k a o : Nat) (i : Nat) (s : MachineState) : Prop :=
  s.getReg .x5 = W n ∧ s.getReg .x6 = W q ∧ s.getReg .x13 = W k ∧ s.getReg .x7 = W a ∧
    s.getReg .x12 = W o ∧ s.getReg .x16 = W (64 - i) ∧
    (∃ qt r, s.getReg .x14 = W qt ∧ s.getReg .x15 = W r ∧ k * 2 ^ (i + 1) = qt * q + r ∧ r < q ∧
      qt < 2 ^ (i + 1)) ∧
    (∀ x, s.getMem x = s0.getMem x) ∧ Saved s0 s

/-- One division step: instructions 10 to 16. -/
theorem divStep (P : Prog) (c : Nat) (hcode : At P c) (_hc : BASE + 4 * (c + 34) < 2 ^ 64)
    (n q k a o : Nat) (s0 : MachineState) (i : Nat) (s : MachineState) (hq : q < 2 ^ 63)
    (hi : i < 64) (hI : DivInv s0 n q k a o i s) :
    P.Spec (c + 10) s 5 (c + 16) (DivMid s0 n q k a o i) := by
  obtain ⟨x5, x6, x13, x7, x12, x16, ⟨qt, r, x14, x15, hdiv, hr, hqt⟩, hmem, hsaved⟩ := hI
  have hstep := div_step hdiv hr
  have hqt2 : qt < 2 ^ 63 := lt_of_lt_of_le hqt (Nat.pow_le_pow_right (by omega) (by omega))
  have hpow : 2 ^ (i + 1) = 2 ^ i * 2 := pow_succ 2 i
  cases hcode
  mrunAt [DivMid, x5, x6, x13, x7, x12, x14, x15, x16]
  all_goals and_intros
  all_goals first
    | (have hge : 2 * r ≥ q := by simp [BitVec.ult] at hbr; omega
       simp only [hge, if_true] at hstep
       refine ⟨2 * qt + 1, 2 * r - q, ?_, ?_, ?_, ?_, ?_⟩
       · bv_omega
       · bv_omega
       · exact hstep.1
       · exact hstep.2
       · omega)
    | (have hlt : ¬ 2 * r ≥ q := by simp [BitVec.ult] at hbr; omega
       simp only [hlt, if_false] at hstep
       refine ⟨2 * qt, 2 * r, ?_, ?_, ?_, ?_, ?_⟩
       · bv_omega
       · bv_omega
       · exact hstep.1
       · exact hstep.2
       · omega)
    | leaf_close

/-- The counter's decrement: instruction 16. -/
theorem divNext (P : Prog) (c : Nat) (hcode : At P c) (_hc : BASE + 4 * (c + 34) < 2 ^ 64)
    (n q k a o : Nat) (s0 : MachineState) (i : Nat) (s : MachineState) (hi : i < 64)
    (hM : DivMid s0 n q k a o i s) : P.Spec (c + 16) s 1 (c + 17) (DivInv s0 n q k a o (i + 1)) := by
  obtain ⟨x5, x6, x13, x7, x12, x16, ⟨qt, r, x14, x15, hdiv, hr, hqt⟩, hmem, hsaved⟩ := hM
  cases hcode
  mrunAt [DivInv, x5, x6, x13, x7, x12, x14, x15, x16]
  all_goals and_intros
  all_goals first
    | exact W_mod (by omega)
    | exact ⟨qt, r, by simpa using x14, by simpa using x15, hdiv, hr, hqt⟩
    | (intro x; simpa using hmem x)
    | leaf_close

/-- The jump back: instruction 17. -/
theorem divJump (P : Prog) (c : Nat) (hcode : At P c) (_hc : BASE + 4 * (c + 34) < 2 ^ 64)
    (n q k a o : Nat) (s0 : MachineState) (i : Nat) (s : MachineState)
    (hI : DivInv s0 n q k a o i s) : P.Spec (c + 17) s 1 (c + 8) (DivInv s0 n q k a o i) := by
  cases hcode
  mrunAt
  exact hI

/-- One division step from the head back to the head. -/
theorem divBody (P : Prog) (c : Nat) (hcode : At P c) (hc : BASE + 4 * (c + 34) < 2 ^ 64)
    (n q k a o : Nat) (s0 : MachineState) (hq : q < 2 ^ 63) (i : Nat) (s : MachineState)
    (hi : i < 64) (hI : DivInv s0 n q k a o i s) :
    P.Spec (c + 8) s 8 (c + 8) (DivInv s0 n q k a o (i + 1)) :=
  Prog.Spec.seq (divBranch P c hcode hc n q k a o s0 i s hi hI) fun s' h' => by
    subst h'
    exact Prog.Spec.seq (divStep P c hcode hc n q k a o s0 i s' hq hi hI) fun s'' h'' =>
      Prog.Spec.seq (divNext P c hcode hc n q k a o s0 i s'' hi h'') fun s3 h3 =>
        divJump P c hcode hc n q k a o s0 (i + 1) s3 h3

/-- After the division: `k 2^64 = k' q + ρ` with `ρ < q`, and the residue loop's first head. -/
def Divided (s0 : MachineState) (n q k a o : Nat) (A : Nat → Nat) (s : MachineState) : Prop :=
  ∃ kq ρ, k * 2 ^ 64 = kq * q + ρ ∧ ρ < q ∧ kq < 2 ^ 64 ∧ Inv s0 n q k a o kq A 0 s

/-- The division's exit: instructions 8, 9, and the jump to 18. -/
theorem divExit (P : Prog) (c : Nat) (hcode : At P c) (_hc : BASE + 4 * (c + 34) < 2 ^ 64)
    (n q k a o : Nat) (A : Nat → Nat) (s0 : MachineState) (s : MachineState)
    (hI : DivInv s0 n q k a o 64 s) : P.Spec (c + 8) s 2 (c + 18) (Divided s0 n q k a o A) := by
  obtain ⟨x5, x6, x13, x7, x12, x16, ⟨qt, r, x14, x15, hdiv, hr, hqt⟩, hmem, hsaved⟩ := hI
  have x16' : s.getReg .x16 = W 0 := by simpa using x16
  cases hcode
  mrunAt [Divided, Inv, x5, x6, x13, x7, x12, x14, x16']
  all_goals first
    | (refine ⟨qt, r, ?_, hr, ?_, ?_, rfl, ?_, ?_, fun i hi => absurd hi (Nat.not_lt_zero i),
         fun x _ => hmem x, hsaved⟩
       · simpa using hdiv
       · simpa using hqt
       · simp
       · simp
       · simp)
    | (exfalso; revert hbr; simp; done)

/-- Inside an iteration: the loop registers, `x15 = v`, and the memory of `s1`. -/
def Mid (s0 s1 : MachineState) (n q k a o kq : Nat) (j v : Nat) (s : MachineState) : Prop :=
  s.getReg .x5 = W (n - j) ∧ s.getReg .x6 = W q ∧ s.getReg .x13 = W k ∧ s.getReg .x14 = W kq ∧
    s.getReg .x7 = W (a + 8 * j) ∧ s.getReg .x12 = W (o + 8 * j) ∧ s.getReg .x15 = W v ∧
    (∀ x, s.getMem x = s1.getMem x) ∧ Saved s0 s

/-- The products: instructions 20 to 25. -/
theorem compute (P : Prog) (c : Nat) (hcode : At P c) (_hc : BASE + 4 * (c + 34) < 2 ^ 64)
    (d n q k a o : Nat) (A : Nat → Nat) (s0 : MachineState) (h : Pre P d n q k a o A s0)
    (kq ρ : Nat) (hdiv : k * 2 ^ 64 = kq * q + ρ) (hρ : ρ < q) (hkq : kq < 2 ^ 64) (j : Nat)
    (s : MachineState) (hj : j < n) (hI : Inv s0 n q k a o kq A j s) :
    P.Spec (c + 20) s 5 (c + 25)
      (Mid s0 s n q k a o kq j (k * A j - A j * kq / 2 ^ 64 * q)) := by
  obtain ⟨x5, x6, x13, x14, x7, x12, -, hframe, hsaved⟩ := hI
  have hAj : s.getMem (W (a + 8 * j)) = W (A j) := by
    rw [hframe _ ?_, h.hA j hj]
    intro i hi heq
    have := Mem.ofNat_inj (by have := h.arange; have := h.hmem; omega)
      (by have := h.orange; have := h.hmem; omega) heq
    have := h.disjA; omega
  have hAq := h.hAq j hj
  have hq := h.hq
  have hk := h.hk
  obtain ⟨hlo, hhi⟩ := shoup (x := A j) (by omega) hAq hdiv hρ
  cases hcode
  have hmul : rv64_mulhu (W (A j)) (W kq) = W (A j * kq / 2 ^ 64) := mulhu_W (by omega) hkq
  have haal := h.aal
  have ha := h.arange
  have hm := h.hmem
  mrunAt [Mid, x5, x6, x13, x14, x7, x12, hAj, hmul]
  all_goals and_intros
  all_goals first
    | (rw [BitVec.ofNat_mul_ofNat, BitVec.ofNat_mul_ofNat, sub_W (by rw [Nat.mul_comm (A j) k]; exact hlo),
         Nat.mul_comm (A j) k]; done)
    | (intro x; simp; done)
    | (simp only [BASE] at *; omega)
    | leaf_close

/-- The residue loop's head in iteration `j < n`: instruction 18 to 20. -/
theorem elemBranch (P : Prog) (c : Nat) (hcode : At P c) (_hc : BASE + 4 * (c + 34) < 2 ^ 64)
    (d n q k a o : Nat) (A : Nat → Nat) (s0 : MachineState) (h : Pre P d n q k a o A s0)
    (kq j : Nat) (s : MachineState) (hj : j < n) (hI : Inv s0 n q k a o kq A j s) :
    P.Spec (c + 18) s 1 (c + 20) (fun t => t = s) := by
  obtain ⟨x5, -⟩ := hI
  have ha := h.arange
  have hm := h.hmem
  cases hcode
  mrunAt [x5]
  all_goals first
    | rfl
    | (exfalso
       have h0 : W (n - j) = W 0 := by simpa using hbr
       have := Mem.ofNat_inj (by omega) (by omega) h0
       omega)

/-- The conditional subtraction: instructions 25 to 28. -/
theorem reduce (P : Prog) (c : Nat) (hcode : At P c) (_hc : BASE + 4 * (c + 34) < 2 ^ 64)
    (d n q k a o : Nat) (A : Nat → Nat) (s0 s1 : MachineState) (h : Pre P d n q k a o A s0)
    (kq ρ : Nat) (hdiv : k * 2 ^ 64 = kq * q + ρ) (hρ : ρ < q) (j : Nat) (s : MachineState)
    (hj : j < n) (hM : Mid s0 s1 n q k a o kq j (k * A j - A j * kq / 2 ^ 64 * q) s) :
    P.Spec (c + 25) s 2 (c + 28) (Mid s0 s1 n q k a o kq j (k * A j % q)) := by
  obtain ⟨x5, x6, x13, x14, x7, x12, x15, hmem, hsaved⟩ := hM
  have hAq := h.hAq j hj
  have hq := h.hq
  obtain ⟨hlo, hhi⟩ := shoup (x := A j) (c := k) (by omega) hAq hdiv hρ
  have hmod := shoup_mod (x := A j) (c := k) (by omega) hAq hdiv hρ
  have hv : k * A j - A j * kq / 2 ^ 64 * q < 2 * q := by omega
  have e64 : (2 : ℕ) ^ 64 = 18446744073709551616 := by norm_num
  rw [e64] at hv hmod
  cases hcode
  mrunAt [Mid, x5, x6, x13, x14, x7, x12, x15]
  all_goals and_intros
  all_goals first
    | (have hge : k * A j - A j * kq / 18446744073709551616 * q ≥ q := by
         simp [BitVec.ult] at hbr; omega
       rw [if_pos hge] at hmod
       rw [sub_W hge, hmod]; done)
    | (have hlt : ¬ k * A j - A j * kq / 18446744073709551616 * q ≥ q := by
         simp [BitVec.ult] at hbr; omega
       rw [if_neg hlt] at hmod
       rw [hmod]; done)
    | leaf_close

/-- After the store of residue `j`. -/
def Stored (s0 : MachineState) (n q k a o kq : Nat) (A : Nat → Nat) (j : Nat) (s : MachineState) :
    Prop :=
  s.getReg .x5 = W (n - j) ∧ s.getReg .x6 = W q ∧ s.getReg .x13 = W k ∧ s.getReg .x14 = W kq ∧
    s.getReg .x7 = W (a + 8 * j) ∧ s.getReg .x12 = W (o + 8 * j) ∧
    (∀ i < j + 1, s.getMem (W (o + 8 * i)) = W (k * A i % q)) ∧
    (∀ x, (∀ i < j + 1, x ≠ W (o + 8 * i)) → s.getMem x = s0.getMem x) ∧ Saved s0 s

/-- The store: instruction 28. -/
theorem store (P : Prog) (c : Nat) (hcode : At P c) (_hc : BASE + 4 * (c + 34) < 2 ^ 64)
    (d n q k a o : Nat) (A : Nat → Nat) (s0 s1 : MachineState) (h : Pre P d n q k a o A s0)
    (kq j : Nat) (s : MachineState) (hj : j < n) (hI : Inv s0 n q k a o kq A j s1)
    (hM : Mid s0 s1 n q k a o kq j (k * A j % q) s) :
    P.Spec (c + 28) s 1 (c + 29) (Stored s0 n q k a o kq A j) := by
  obtain ⟨-, -, -, -, -, -, hout, hframe, -⟩ := hI
  obtain ⟨x5, x6, x13, x14, x7, x12, x15, hmem, hsaved⟩ := hM
  have ho := h.orange
  have hm := h.hmem
  have hoal := h.oal
  cases hcode
  mrunAt [Stored, x5, x6, x13, x14, x7, x12, x15]
  all_goals first
    | (simp only [BASE] at *; omega)
    | (refine ⟨?_, ?_, ?_⟩
       · intro i hi
         rcases Nat.lt_succ_iff_lt_or_eq.mp hi with hi | rfl
         · rw [MachineState.getMem_setMem_ne (Mem.ofNat_ne (by simp only [BASE] at *; omega)
             (by simp only [BASE] at *; omega) (by omega))]
           simpa [hmem] using hout i hi
         · rw [MachineState.getMem_setMem_eq]
       · intro x hx
         rw [MachineState.getMem_setMem_ne (hx _ (Nat.lt_succ_self _))]
         simpa [hmem] using hframe x fun i hi => hx i (by omega)
       · intro r hr
         simpa using hsaved r hr)

/-- The pointer updates: instructions 29 to 31. -/
theorem next (P : Prog) (c : Nat) (hcode : At P c) (_hc : BASE + 4 * (c + 34) < 2 ^ 64)
    (n q k a o kq : Nat) (A : Nat → Nat) (s0 : MachineState) (j : Nat) (s : MachineState)
    (hj : j < n) (hS : Stored s0 n q k a o kq A j s) :
    P.Spec (c + 29) s 3 (c + 32) (Inv s0 n q k a o kq A (j + 1)) := by
  obtain ⟨x5, x6, x13, x14, x7, x12, hout, hframe, hsaved⟩ := hS
  cases hcode
  mrunAt [Inv, x5, x6, x13, x14, x7, x12, W_add]
  all_goals and_intros
  all_goals first
    | exact W_mod (by omega)
    | (intro i hi; simpa using hout i hi)
    | (intro x hx; simpa using hframe x hx)
    | leaf_close

/-- The jump back: instruction 32. -/
theorem jump (P : Prog) (c : Nat) (hcode : At P c) (_hc : BASE + 4 * (c + 34) < 2 ^ 64)
    (n q k a o kq : Nat) (A : Nat → Nat) (s0 : MachineState) (j : Nat) (s : MachineState)
    (hI : Inv s0 n q k a o kq A j s) : P.Spec (c + 32) s 1 (c + 18) (Inv s0 n q k a o kq A j) := by
  cases hcode
  mrunAt
  exact hI

/-- One residue, from the head back to the head. -/
theorem body (P : Prog) (c : Nat) (hcode : At P c) (hc : BASE + 4 * (c + 34) < 2 ^ 64)
    (d n q k a o : Nat) (A : Nat → Nat) (s0 : MachineState) (h : Pre P d n q k a o A s0)
    (kq ρ : Nat) (hdiv : k * 2 ^ 64 = kq * q + ρ) (hρ : ρ < q) (hkq : kq < 2 ^ 64) (j : Nat)
    (s : MachineState) (hj : j < n) (hI : Inv s0 n q k a o kq A j s) :
    P.Spec (c + 18) s 13 (c + 18) (Inv s0 n q k a o kq A (j + 1)) :=
  Prog.Spec.seq (elemBranch P c hcode hc d n q k a o A s0 h kq j s hj hI) fun s' h' => by
    subst h'
    exact Prog.Spec.seq (compute P c hcode hc d n q k a o A s0 h kq ρ hdiv hρ hkq j s' hj hI)
      fun s2 h2 =>
        Prog.Spec.seq (reduce P c hcode hc d n q k a o A s0 s' h kq ρ hdiv hρ j s2 hj h2)
          fun s3 h3 =>
            Prog.Spec.seq (store P c hcode hc d n q k a o A s0 s' h kq j s3 hj hI h3) fun s4 h4 =>
              Prog.Spec.seq (next P c hcode hc n q k a o kq A s0 j s4 hj h4) fun s5 h5 =>
                jump P c hcode hc n q k a o kq A s0 (j + 1) s5 h5

/-- The exit: instructions 18, 19, and the jump to `ret`. -/
theorem exit (P : Prog) (c : Nat) (hcode : At P c) (_hc : BASE + 4 * (c + 34) < 2 ^ 64)
    (n q k a o kq : Nat) (A : Nat → Nat) (s0 : MachineState) (s : MachineState)
    (hI : Inv s0 n q k a o kq A n s) : P.Spec (c + 18) s 2 (c + 33) (Post s0 n q k o A) := by
  obtain ⟨x5, -, -, -, -, -, hout, hframe, hsaved⟩ := hI
  have x5' : s.getReg .x5 = W 0 := by simpa using x5
  cases hcode
  mrunAt [Post, x5']
  all_goals first
    | exact ⟨hout, hframe, hsaved⟩
    | (exfalso; revert hbr; simp; done)

/-- **The specification.** From entry, the leaf reaches `ret` within `524 + 14 n` steps. -/
theorem spec (P : Prog) (c : Nat) (hcode : At P c) (hc : BASE + 4 * (c + 34) < 2 ^ 64)
    (d n q k a o : Nat) (A : Nat → Nat) (s : MachineState) (h : Pre P d n q k a o A s) :
    P.Spec c s (524 + 14 * n) (c + 33) (Post s n q k o A) := by
  refine Prog.Spec.mono (Prog.Spec.seq (prologue P c hcode hc d n q k a o A s h) fun s1 h1 =>
    Prog.Spec.seq (Prog.Spec.loop (N := 64)
      (fun i s' hi hI => divBody P c hcode hc n q k a o s h.hq i s' hi hI)
      (fun s' hI => divExit P c hcode hc n q k a o A s s' hI) 0 s1 (Nat.zero_le _) h1)
    fun s2 h2 => by
      obtain ⟨kq, ρ, hdiv, hρ, hkq, h2⟩ := h2
      exact Prog.Spec.loop (N := n)
        (fun j s' hj hI => body P c hcode hc d n q k a o A s h kq ρ hdiv hρ hkq j s' hj hI)
        (fun s' hI => exit P c hcode hc n q k a o kq A s s' hI) 0 s2 (Nat.zero_le _) h2) ?_
    fun _ hp => hp
  omega

open MxxCore.Leaf

/-- The leaf's function on bytes: the encoded products `k x i mod q` of the words of its input. -/
def bytesF (args : List Nat) (ins : List Bytes) : List Bytes :=
  match args, ins with
  | [n, q, k], [x] => [Bytes.u64s ((List.range n).map fun i => k * (words x).getD i 0 % q)]
  | _, _ => []

/-- The leaf's domain: arguments `[n, q, k]` with `k < q < 2^63`, and one input of `n` words below
`q`. -/
def bytesDom (args : List Nat) (ins : List Bytes) : Prop :=
  ∃ n q k x, args = [n, q, k] ∧ ins = [x] ∧ x.length = 8 * n ∧ k < q ∧ q < 2 ^ 63 ∧
    ∀ v ∈ words x, v < q

theorem sound_bytes {P : Prog} {c : Nat} (hcode : At P c) (hc : BASE + 4 * (c + 34) < 2 ^ 64)
    (kk : Call) (s : MachineState) (he : Entry P kk s) (hdom : bytesDom kk.args (kk.ins.map (·.2)))
    (hlen : kk.outs.map (·.2) = (bytesF kk.args (kk.ins.map (·.2))).map List.length) :
    P.Spec c s (524 + 14 * kk.args.headD 0) (c + 33)
      (Exit kk (bytesF kk.args (kk.ins.map (·.2))) s) := by
  obtain ⟨n, q, k, x, hargs, hins, hx, hk, hq, hxq⟩ := hdom
  obtain ⟨d, args, ins, outs, sc⟩ := kk
  simp only at hargs hins hlen ⊢
  subst hargs
  rcases ins with _ | ⟨⟨a, x'⟩, _ | _⟩ <;> simp at hins
  subst hins
  simp only [List.map, bytesF, Mem.length_u64s, List.length_map, List.length_range] at hlen
  rcases outs with _ | ⟨⟨o, lo⟩, _ | _⟩ <;> simp at hlen
  subst hlen
  have hw : Call.words ⟨d, [n, q, k], [(a, x')], [(o, 8 * n)], sc⟩ = [n, q, k, a, o, sc.1] := rfl
  have hmem := he.hmem
  have ⟨aal, a1, a2⟩ := he.ial (a, x') (by simp)
  have ⟨oal, o1, o2⟩ := he.wal (o, 8 * n) (by simp [Call.written])
  simp only at aal a1 a2 oal o1 o2
  have dr := he.drange
  rw [hw] at dr
  simp only [List.length_cons, List.length_nil] at dr
  have dj : ∀ i (h : i < 6), s.getMem (W (d + 8 * i)) = W ([n, q, k, a, o, sc.1][i]) :=
    fun i h => he.desc i (by rw [hw]; simpa using h)
  have hlx : (words x').length = n := by rw [length_words, hx]; omega
  have hgetD : ∀ (l : List Nat) (i : Nat) (h : i < l.length), l.getD i 0 = l[i] :=
    fun l i h => by simp [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem h]
  have hA : ∀ i < n, s.getMem (W (a + 8 * i)) = W ((words x').getD i 0) := fun i hi => by
    have hx' : Holds s a x' := he.ins (a, x') (by simp)
    rw [getMem_of_holds s a x' hx' i (by omega) aal (by omega), hgetD _ _ (by omega)]
  have d0 : s.getMem (W d) = W n := by
    have := dj 0 (by decide)
    simp only [Nat.mul_zero, Nat.add_zero] at this
    exact this
  have d1 : s.getMem (W (d + 8)) = W q := by
    have := dj 1 (by decide); simpa only [List.getElem_cons_succ, List.getElem_cons_zero] using this
  have d2 : s.getMem (W (d + 16)) = W k := by
    have := dj 2 (by decide); simpa only [List.getElem_cons_succ, List.getElem_cons_zero] using this
  have d3 : s.getMem (W (d + 24)) = W a := by
    have := dj 3 (by decide); simpa only [List.getElem_cons_succ, List.getElem_cons_zero] using this
  have d4 : s.getMem (W (d + 32)) = W o := by
    have := dj 4 (by decide); simpa only [List.getElem_cons_succ, List.getElem_cons_zero] using this
  have pre : Pre P d n q k a o (fun i => (words x').getD i 0) s :=
    { hd := he.hd
      dn := d0
      dq := d1
      dk := d2
      da := d3
      dout := d4
      hA := hA
      hAq := fun i hi => by rw [hgetD _ _ (by omega)]; exact hxq _ (List.getElem_mem _)
      hk := hk
      hq := hq
      hmem := hmem
      dal := he.dal
      aal := aal
      oal := oal
      drange := ⟨dr.1, by omega⟩
      arange := ⟨a1, by omega⟩
      orange := ⟨o1, o2⟩
      disjA := by
        have := he.apartIn (o, 8 * n) (by simp [Call.written]) (a, x') (by simp)
        simp only [Apart] at this; omega }
  refine Prog.Spec.mono (spec P c hcode hc d n q k a o _ s pre) (by simp) ?_
  rintro s' ⟨hout, hframe, hsaved⟩
  refine ⟨?_, ?_, hsaved⟩
  · intro i h h'
    simp only [List.length_cons, List.length_nil] at h
    obtain rfl : i = 0 := by omega
    simp only [bytesF, List.getElem_cons_zero]
    exact holds_u64s s' o _ oal (by simp; omega) fun i hi => by
      simp only [List.length_map, List.length_range] at hi
      rw [hout i hi]
      simp
  · intro c' hc' hout'
    have := hout' (o, 8 * n) (by simp [Call.written])
    exact getByte_of_frame s s' o n oal hframe (by omega) c' hc' this

/-- The specification in the shape of the core (`MxxCore.Leaf.Spec`), for every program that holds
the leaf's code at instruction `c`. -/
def byteSpec {P : Prog} {c : Nat} (hcode : At P c) (hc : BASE + 4 * (c + 34) < 2 ^ 64) :
    MxxCore.Leaf.Spec P c where
  ret := 33
  hret := hcode.i33
  steps args := 524 + 14 * args.headD 0
  dom := bytesDom
  f := bytesF
  sound k s he hdom hlen := sound_bytes hcode hc k s he hdom hlen

end MxxLatticeLeaves.ScaleMod
