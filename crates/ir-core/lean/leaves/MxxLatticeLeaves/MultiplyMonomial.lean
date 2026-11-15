import MxxLatticeLeaves.Code.MultiplyMonomial
import MxxLatticeLeaves.Math.Monomial
import MxxLatticeLeaves.Math.Words
import MxxCore.Glue
import MxxCore.Leaf
import MxxCore.Sym

/-!
# The specification of the leaf `lattice.multiply_monomial`

For any program `P` that holds the leaf's code at instruction `c`, from that entry, with `a0`
pointing at the descriptor `[N, q, E, a, ka, o, scratch]` (`N = 2^m`), the leaf reaches its `ret`
(instruction `c + 39`) within `14 + E (16 N + 6)` steps, having written to the doubleword at
`o + 8 j` (`j < E N`) the word `F j`: coefficient `j % N` of entry `j / N` of the product by
`X^k`, gathered — with `t = (p - k) mod 2 N` (computed as `(p + kr) mod 2 N` for
`kr = (2 N - k mod 2 N) mod 2 N`), coefficient `p` is `A[t]` when `t < N`, else `-A[t - N] mod q`.
Here `k` is the word at `ka`. It changes no other doubleword and keeps every register the calling
convention saves.
-/

namespace MxxLatticeLeaves.MultiplyMonomial

open MathcalA RiscvZkvm.Rv64 MxxLatticeLeaves.Code.MultiplyMonomial MxxLatticeLeaves.Math.Monomial
  MxxLatticeLeaves.Math.Words

/-- The registers a leaf keeps: `ra`, `sp`, and `s0`–`s11`. -/
def Saved (s0 s : MachineState) : Prop :=
  ∀ r ∈ [Reg.x1, .x2, .x8, .x9, .x18, .x19, .x20, .x21, .x22, .x23, .x24, .x25, .x26, .x27],
    s.getReg r = s0.getReg r

/-- `-k mod 2 N`. -/
def negk (N k : Nat) : Nat := (2 * N - k % (2 * N)) % (2 * N)

/-- The input coefficient of output coefficient `p`: `(p - k) mod 2 N`. -/
def shift (N k p : Nat) : Nat := (p + negk N k) % (2 * N)

/-- Coefficient `p` of entry `e` of the product by `X^k`. -/
def val (N q k : Nat) (A : Nat → Nat) (e p : Nat) : Nat :=
  if shift N k p < N then A (e * N + shift N k p) else (q - A (e * N + (shift N k p - N))) % q

/-- Word `j` of the output. -/
def F (N q k : Nat) (A : Nat → Nat) (j : Nat) : Nat := val N q k A (j / N) (j % N)

/-- What the leaf assumes on entry. -/
structure Pre (P : Prog) (d N q E a ka o k m : Nat) (A : Nat → Nat) (s : MachineState) : Prop where
  hd : s.getReg .x10 = W d
  dN : s.getMem (W d) = W N
  dq : s.getMem (W (d + 8)) = W q
  dE : s.getMem (W (d + 16)) = W E
  da : s.getMem (W (d + 24)) = W a
  dka : s.getMem (W (d + 32)) = W ka
  dout : s.getMem (W (d + 40)) = W o
  hk : s.getMem (W ka) = W k
  hk64 : k < 2 ^ 64
  hA : ∀ j < E * N, s.getMem (W (a + 8 * j)) = W (A j)
  hAq : ∀ j < E * N, A j < q
  hq : q < 2 ^ 63
  hN : N = 2 ^ m
  hm : m ≤ 58
  hmem : BASE + P.mem < 2 ^ 64
  dal : d % 8 = 0
  aal : a % 8 = 0
  kal : ka % 8 = 0
  oal : o % 8 = 0
  drange : BASE ≤ d ∧ d + 48 ≤ BASE + P.mem
  arange : BASE ≤ a ∧ a + 8 * (E * N) ≤ BASE + P.mem
  krange : BASE ≤ ka ∧ ka + 8 ≤ BASE + P.mem
  orange : BASE + P.bin.length ≤ o ∧ o + 8 * (E * N) ≤ BASE + P.mem
  disjA : o + 8 * (E * N) ≤ a ∨ a + 8 * (E * N) ≤ o
  disjK : o + 8 * (E * N) ≤ ka ∨ ka + 8 ≤ o

/-- What holds at `ret`. -/
def Post (s0 : MachineState) (N q E o k : Nat) (A : Nat → Nat) (s : MachineState) : Prop :=
  (∀ j < E * N, s.getMem (W (o + 8 * j)) = W (F N q k A j)) ∧
    (∀ x, (∀ j < E * N, x ≠ W (o + 8 * j)) → s.getMem x = s0.getMem x) ∧ Saved s0 s

/-- The registers that hold throughout the loops. -/
def Fixed (N q k : Nat) (s : MachineState) : Prop :=
  s.getReg .x5 = W N ∧ s.getReg .x6 = W q ∧ s.getReg .x13 = W (negk N k) ∧
    s.getReg .x14 = W (2 * N - 1) ∧ s.getReg .x28 = W (8 * N)

/-- The written prefix: `J` words, and the frame. -/
def Written (s0 : MachineState) (N q o k : Nat) (A : Nat → Nat) (J : Nat) (s : MachineState) :
    Prop :=
  (∀ j < J, s.getMem (W (o + 8 * j)) = W (F N q k A j)) ∧
    (∀ x, (∀ j < J, x ≠ W (o + 8 * j)) → s.getMem x = s0.getMem x)

/-- The entry loop's head (instruction 12) before entry `e`. -/
def OInv (s0 : MachineState) (N q E a o k : Nat) (A : Nat → Nat) (e : Nat) (s : MachineState) :
    Prop :=
  Fixed N q k s ∧ s.getReg .x7 = W (E - e) ∧ s.getReg .x11 = W (a + 8 * (e * N)) ∧
    s.getReg .x12 = W (o + 8 * (e * N)) ∧ Written s0 N q o k A (e * N) s ∧ Saved s0 s

/-- The coefficient loop's head (instruction 15) before coefficient `p` of entry `e`. -/
def IInv (s0 : MachineState) (N q E a o k : Nat) (A : Nat → Nat) (e p : Nat) (s : MachineState) :
    Prop :=
  Fixed N q k s ∧ s.getReg .x7 = W (E - e) ∧ s.getReg .x11 = W (a + 8 * (e * N)) ∧
    s.getReg .x12 = W (o + 8 * (e * N + p)) ∧ s.getReg .x15 = W p ∧
    Written s0 N q o k A (e * N + p) s ∧ Saved s0 s

/-- Inside a coefficient: the head's registers and the memory of `s1`. -/
def Base (s0 s1 : MachineState) (N q E a o k e p : Nat) (s : MachineState) : Prop :=
  Fixed N q k s ∧ s.getReg .x7 = W (E - e) ∧ s.getReg .x11 = W (a + 8 * (e * N)) ∧
    s.getReg .x12 = W (o + 8 * (e * N + p)) ∧ s.getReg .x15 = W p ∧
    (∀ x, s.getMem x = s1.getMem x) ∧ Saved s0 s

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

theorem two_N {N m : Nat} (hN : N = 2 ^ m) : 2 * N = 2 ^ (m + 1) := by rw [hN, pow_succ]; ring

theorem negk_lt {N m k : Nat} (hN : N = 2 ^ m) : negk N k < 2 * N := by
  unfold negk; exact Nat.mod_lt _ (by rw [two_N hN]; positivity)

/-- The word `-k mod 2 N`. -/
theorem negk_W {N m k : Nat} (hN : N = 2 ^ m) (hm : m ≤ 58) (hk : k < 2 ^ 64) :
    (0#64 - BitVec.ofNat 64 k) &&& BitVec.ofNat 64 (2 * N - 1) = W (negk N k) := by
  apply BitVec.eq_of_toNat_eq
  have h59 : 2 ^ (m + 1) ≤ 2 ^ 59 := Nat.pow_le_pow_right (by norm_num) (by omega)
  have hpos : 0 < 2 ^ (m + 1) := by positivity
  simp only [W, BitVec.toNat_ofNat]
  rw [two_N hN, neg_mask (by omega) hk]
  unfold negk
  rw [two_N hN]
  exact (Nat.mod_eq_of_lt (lt_of_lt_of_le (Nat.mod_lt _ hpos) (by omega))).symm

/-- The word `(p - k) mod 2 N` for `p < N`. -/
theorem shift_W {N m k p : Nat} (hN : N = 2 ^ m) (hm : m ≤ 58) (hp : p < N) :
    (BitVec.ofNat 64 p + BitVec.ofNat 64 (negk N k)) &&& BitVec.ofNat 64 (2 * N - 1) =
      W (shift N k p) := by
  apply BitVec.eq_of_toNat_eq
  have h59 : 2 ^ (m + 1) ≤ 2 ^ 59 := Nat.pow_le_pow_right (by norm_num) (by omega)
  have hpos : 0 < 2 ^ (m + 1) := by positivity
  rw [BitVec.ofNat_add_ofNat, two_N hN, and_mask (by omega), mod_mod_pow (by omega)]
  simp only [W, BitVec.toNat_ofNat]
  unfold shift
  rw [two_N hN]
  exact (Nat.mod_eq_of_lt (lt_of_lt_of_le (Nat.mod_lt _ hpos) (by omega))).symm

theorem shift_lt {N m k p : Nat} (hN : N = 2 ^ m) : shift N k p < 2 * N := by
  unfold shift; exact Nat.mod_lt _ (by rw [two_N hN]; positivity)

/-- After the loads: `x5 = N`, `x6 = q`, `x7 = E`, `x11 = a`, `x13 = k`, `x12 = o`. -/
def Loaded (s0 : MachineState) (N q E a o k : Nat) (s : MachineState) : Prop :=
  s.getReg .x5 = W N ∧ s.getReg .x6 = W q ∧ s.getReg .x7 = W E ∧ s.getReg .x11 = W a ∧
    s.getReg .x13 = W k ∧ s.getReg .x12 = W o ∧ (∀ x, s.getMem x = s0.getMem x) ∧ Saved s0 s

/-- The loads: instructions 0 to 7. -/
theorem pro1 (P : Prog) (c : Nat) (hcode : At P c) (_hc : BASE + 4 * (c + 40) < 2 ^ 64)
    (d N q E a ka o k m : Nat) (A : Nat → Nat) (s : MachineState)
    (h : Pre P d N q E a ka o k m A s) : P.Spec c s 7 (c + 7) (Loaded s N q E a o k) := by
  obtain ⟨hd, dN, dq, dE, da, dka, dout, hk, -, -, -, -, -, -, hmem, dal, -, kal, -, ⟨d1, d2⟩, -,
    ⟨k1, k2⟩, -, -, -⟩ := h
  have hsaved : Saved s s := fun _ _ => rfl
  cases hcode
  mrunAt [Loaded, Saved, hd, dN, dq, dE, da, dka, dout, hk, W_add]
  all_goals and_intros
  all_goals first
    | (intro _; rfl)
    | leaf_close

/-- The mask: instructions 7 and 8. -/
theorem pro2 (P : Prog) (c : Nat) (hcode : At P c) (_hc : BASE + 4 * (c + 40) < 2 ^ 64)
    (d N q E a ka o k m : Nat) (A : Nat → Nat) (s0 : MachineState)
    (h : Pre P d N q E a ka o k m A s0) (s : MachineState) (hL : Loaded s0 N q E a o k s) :
    P.Spec (c + 7) s 2 (c + 9)
      (fun t => Loaded s0 N q E a o k t ∧ t.getReg .x14 = W (2 * N - 1)) := by
  obtain ⟨x5, x6, x7, x11, x13, x12, hmem, hsaved⟩ := hL
  have hN2 : 2 * N ≤ 2 ^ 59 := by
    rw [two_N h.hN]; exact Nat.pow_le_pow_right (by norm_num) (by have := h.hm; omega)
  have hN1 : 1 ≤ N := by rw [h.hN]; exact Nat.one_le_two_pow
  cases hcode
  mrunAt [Loaded, x5, x6, x7, x11, x13, x12]
  all_goals and_intros
  all_goals first
    | (intro x; simpa using hmem x)
    | leaf_close

/-- `-k mod 2 N` and `8 N`: instructions 9 to 12. -/
theorem pro3 (P : Prog) (c : Nat) (hcode : At P c) (_hc : BASE + 4 * (c + 40) < 2 ^ 64)
    (d N q E a ka o k m : Nat) (A : Nat → Nat) (s0 : MachineState)
    (h : Pre P d N q E a ka o k m A s0) (s : MachineState)
    (hL : Loaded s0 N q E a o k s ∧ s.getReg .x14 = W (2 * N - 1)) :
    P.Spec (c + 9) s 3 (c + 12) (OInv s0 N q E a o k A 0) := by
  obtain ⟨⟨x5, x6, x7, x11, x13, x12, hmem, hsaved⟩, x14⟩ := hL
  have hN2 : 2 * N ≤ 2 ^ 59 := by
    rw [two_N h.hN]; exact Nat.pow_le_pow_right (by norm_num) (by have := h.hm; omega)
  have hneg := negk_W h.hN h.hm h.hk64
  have hshl := shl3_W (x := N) (by omega)
  cases hcode
  mrunAt [OInv, Fixed, Written, x5, x6, x7, x11, x13, x12, x14, hneg, hshl]
  all_goals and_intros
  all_goals first
    | (intro x _; simpa using hmem x; done)
    | (intro j hj; simp at hj; done)
    | leaf_close

theorem N_pos {N m : Nat} (hN : N = 2 ^ m) : 0 < N := by rw [hN]; positivity

/-- The entry loop's head for an entry `e < E`: instruction 12 to 14. -/
theorem oBranch (P : Prog) (c : Nat) (hcode : At P c) (_hc : BASE + 4 * (c + 40) < 2 ^ 64)
    (d N q E a ka o k m : Nat) (A : Nat → Nat) (s0 : MachineState)
    (h : Pre P d N q E a ka o k m A s0) (e : Nat) (s : MachineState) (he : e < E)
    (hI : OInv s0 N q E a o k A e s) : P.Spec (c + 12) s 1 (c + 14) (fun t => t = s) := by
  obtain ⟨-, x7, -⟩ := hI
  have ha := h.arange
  have hm := h.hmem
  have hN := N_pos h.hN
  have hEN : E ≤ E * N := Nat.le_mul_of_pos_right E hN
  cases hcode
  mrunAt [x7]
  all_goals first
    | rfl
    | (exfalso
       have h0 : W (E - e) = W 0 := by simpa using hbr
       have := Mem.ofNat_inj (by omega) (by omega) h0
       omega)

/-- The start of an entry: instruction 14. -/
theorem oStart (P : Prog) (c : Nat) (hcode : At P c) (_hc : BASE + 4 * (c + 40) < 2 ^ 64)
    (N q E a o k : Nat) (A : Nat → Nat) (s0 : MachineState) (e : Nat) (s : MachineState)
    (hI : OInv s0 N q E a o k A e s) : P.Spec (c + 14) s 1 (c + 15) (IInv s0 N q E a o k A e 0) := by
  obtain ⟨hfix, x7, x11, x12, hw, hsaved⟩ := hI
  obtain ⟨x5, x6, x13, x14, x28⟩ := hfix
  cases hcode
  mrunAt [IInv, Fixed, x5, x6, x13, x14, x28, x7, x11, x12]
  all_goals and_intros
  all_goals first
    | (simpa using hw.1; done)
    | (simpa using hw.2; done)
    | (exact hw; done)
    | leaf_close

/-- The coefficient loop's head for `p < N`: instruction 15 to 17. -/
theorem iBranch (P : Prog) (c : Nat) (hcode : At P c) (_hc : BASE + 4 * (c + 40) < 2 ^ 64)
    (d N q E a ka o k m : Nat) (A : Nat → Nat) (s0 : MachineState)
    (h : Pre P d N q E a ka o k m A s0) (e p : Nat) (s : MachineState) (hp : p < N)
    (hI : IInv s0 N q E a o k A e p s) : P.Spec (c + 15) s 1 (c + 17) (fun t => t = s) := by
  obtain ⟨⟨x5, -⟩, -, -, -, x15, -⟩ := hI
  have hN2 : 2 * N ≤ 2 ^ 59 := by
    rw [two_N h.hN]; exact Nat.pow_le_pow_right (by norm_num) (by have := h.hm; omega)
  cases hcode
  mrunAt [x5, x15]
  all_goals first
    | rfl
    | (exfalso
       have h0 : W p = W N := by simpa using hbr
       have := Mem.ofNat_inj (by omega) (by omega) h0
       omega)

/-- `t = (p - k) mod 2 N`: instructions 17 and 18. -/
theorem iShift (P : Prog) (c : Nat) (hcode : At P c) (_hc : BASE + 4 * (c + 40) < 2 ^ 64)
    (d N q E a ka o k m : Nat) (A : Nat → Nat) (s0 : MachineState)
    (h : Pre P d N q E a ka o k m A s0) (e p : Nat) (s : MachineState) (hp : p < N)
    (hI : IInv s0 N q E a o k A e p s) :
    P.Spec (c + 17) s 2 (c + 19)
      (fun t => Base s0 s N q E a o k e p t ∧ t.getReg .x17 = W (shift N k p)) := by
  obtain ⟨⟨x5, x6, x13, x14, x28⟩, x7, x11, x12, x15, -, hsaved⟩ := hI
  have hsh := shift_W (k := k) h.hN h.hm hp
  cases hcode
  mrunAt [Base, Fixed, x5, x6, x13, x14, x28, x7, x11, x12, x15, hsh]
  all_goals and_intros
  all_goals first
    | (intro x; rfl)
    | leaf_close

set_option maxHeartbeats 2000000 in
/-- The upper half: `t ≥ N`, `-A[e N + t - N] mod q`: instructions 19, 21 to 28, to 32. -/
theorem iHigh (P : Prog) (c : Nat) (hcode : At P c) (_hc : BASE + 4 * (c + 40) < 2 ^ 64)
    (d N q E a ka o k m : Nat) (A : Nat → Nat) (s0 s1 : MachineState)
    (h : Pre P d N q E a ka o k m A s0) (e p : Nat) (s : MachineState) (he : e < E) (hp : p < N)
    (hload : s1.getMem (W (a + 8 * (e * N + (shift N k p - N)))) =
      W (A (e * N + (shift N k p - N))))
    (ht : N ≤ shift N k p)
    (hB : Base s0 s1 N q E a o k e p s ∧ s.getReg .x17 = W (shift N k p)) :
    P.Spec (c + 19) s 9 (c + 32)
      (fun t => Base s0 s1 N q E a o k e p t ∧ t.getReg .x16 = W (val N q k A e p)) := by
  obtain ⟨⟨⟨x5, x6, x13, x14, x28⟩, x7, x11, x12, x15, hmem, hsaved⟩, x17⟩ := hB
  have hN2 : 2 * N ≤ 2 ^ 59 := by
    rw [two_N h.hN]; exact Nat.pow_le_pow_right (by norm_num) (by have := h.hm; omega)
  have htl := shift_lt (k := k) (p := p) h.hN
  have hj : e * N + (shift N k p - N) < E * N := by
    have : (e + 1) * N ≤ E * N := Nat.mul_le_mul_right N he
    rw [Nat.add_mul, Nat.one_mul] at this; omega
  have hv := h.hAq _ hj
  have hq := h.hq
  have ha := h.arange
  have hm := h.hmem
  have haal := h.aal
  have hsub : W (shift N k p) - W N = W (shift N k p - N) := sub_W ht
  have hshl : BitVec.ofNat 64 (shift N k p - N) <<< 3 = W (8 * (shift N k p - N)) :=
    shl3_W (by omega)
  have hadd : W (a + 8 * (e * N)) + W (8 * (shift N k p - N)) =
      W (a + 8 * (e * N + (shift N k p - N))) := by
    rw [W, W, BitVec.ofNat_add_ofNat]; congr 1; ring
  have hload' : s.getMem (W (a + 8 * (e * N + (shift N k p - N)))) =
      W (A (e * N + (shift N k p - N))) := by rw [hmem]; exact hload
  have hneg : W q - W (A (e * N + (shift N k p - N))) = W (q - A (e * N + (shift N k p - N))) :=
    sub_W hv.le
  have hval : val N q k A e p = (q - A (e * N + (shift N k p - N))) % q := by
    unfold val; simp only [show ¬ shift N k p < N by omega, ↓reduceIte]
  have hult : (W (shift N k p)).ult (W N) = false := by simp [BitVec.ult]; omega
  cases hcode
  mrunAt [Base, Fixed, x5, x6, x13, x14, x28, x7, x11, x12, x15, x17, hult, hsub, hshl, hadd,
    hload', hneg]
  all_goals and_intros
  all_goals first
    | (rw [hval]
       have h0 : A (e * N + (shift N k p - N)) = 0 := by
         simp only [bne_eq_false_iff_eq] at hbr
         bv_omega
       rw [h0]; simp; done)
    | (rw [hval]
       have h0 : A (e * N + (shift N k p - N)) ≠ 0 := by
         intro h0; simp [h0] at hbr
       rw [Nat.mod_eq_of_lt (by omega)]; done)
    | (intro x; simpa using hmem x; done)
    | leaf_close

set_option maxHeartbeats 2000000 in
/-- The lower half: `t < N`, `A[e N + t]`: instructions 19, 20, 29 to 31, to 32. -/
theorem iLow (P : Prog) (c : Nat) (hcode : At P c) (_hc : BASE + 4 * (c + 40) < 2 ^ 64)
    (d N q E a ka o k m : Nat) (A : Nat → Nat) (s0 s1 : MachineState)
    (h : Pre P d N q E a ka o k m A s0) (e p : Nat) (s : MachineState) (he : e < E) (hp : p < N)
    (hload : s1.getMem (W (a + 8 * (e * N + shift N k p))) = W (A (e * N + shift N k p)))
    (ht : shift N k p < N)
    (hB : Base s0 s1 N q E a o k e p s ∧ s.getReg .x17 = W (shift N k p)) :
    P.Spec (c + 19) s 9 (c + 32)
      (fun t => Base s0 s1 N q E a o k e p t ∧ t.getReg .x16 = W (val N q k A e p)) := by
  obtain ⟨⟨⟨x5, x6, x13, x14, x28⟩, x7, x11, x12, x15, hmem, hsaved⟩, x17⟩ := hB
  have hN2 : 2 * N ≤ 2 ^ 59 := by
    rw [two_N h.hN]; exact Nat.pow_le_pow_right (by norm_num) (by have := h.hm; omega)
  have hj : e * N + shift N k p < E * N := by
    have : (e + 1) * N ≤ E * N := Nat.mul_le_mul_right N he
    rw [Nat.add_mul, Nat.one_mul] at this; omega
  have ha := h.arange
  have hm := h.hmem
  have haal := h.aal
  have hshl : BitVec.ofNat 64 (shift N k p) <<< 3 = W (8 * shift N k p) := shl3_W (by omega)
  have hadd : W (a + 8 * (e * N)) + W (8 * shift N k p) = W (a + 8 * (e * N + shift N k p)) := by
    rw [W, W, BitVec.ofNat_add_ofNat]; congr 1; ring
  have hload' : s.getMem (W (a + 8 * (e * N + shift N k p))) = W (A (e * N + shift N k p)) := by
    rw [hmem]; exact hload
  have hval : val N q k A e p = A (e * N + shift N k p) := by unfold val; simp only [ht, ↓reduceIte]
  have hult : (W (shift N k p)).ult (W N) = true := by simp [BitVec.ult]; omega
  cases hcode
  mrunAt [Base, Fixed, x5, x6, x13, x14, x28, x7, x11, x12, x15, x17, hult, hshl, hadd, hload']
  all_goals and_intros
  all_goals first
    | (rw [hval]; done)
    | (intro x; simpa using hmem x; done)
    | leaf_close

theorem F_flat {N q k : Nat} {A : Nat → Nat} {e p : Nat} (hp : p < N) :
    F N q k A (e * N + p) = val N q k A e p := by
  unfold F; rw [(flat hp).1, (flat hp).2]

/-- After the store of coefficient `p` of entry `e`. -/
def Stored (s0 : MachineState) (N q E a o k : Nat) (A : Nat → Nat) (e p : Nat) (s : MachineState) :
    Prop :=
  Fixed N q k s ∧ s.getReg .x7 = W (E - e) ∧ s.getReg .x11 = W (a + 8 * (e * N)) ∧
    s.getReg .x12 = W (o + 8 * (e * N + p)) ∧ s.getReg .x15 = W p ∧
    Written s0 N q o k A (e * N + p + 1) s ∧ Saved s0 s

/-- The store: instruction 32. -/
theorem iStore (P : Prog) (c : Nat) (hcode : At P c) (_hc : BASE + 4 * (c + 40) < 2 ^ 64)
    (d N q E a ka o k m : Nat) (A : Nat → Nat) (s0 s1 : MachineState)
    (h : Pre P d N q E a ka o k m A s0) (e p : Nat) (s : MachineState) (he : e < E) (hp : p < N)
    (hI : IInv s0 N q E a o k A e p s1)
    (hB : Base s0 s1 N q E a o k e p s ∧ s.getReg .x16 = W (val N q k A e p)) :
    P.Spec (c + 32) s 1 (c + 33) (Stored s0 N q E a o k A e p) := by
  obtain ⟨-, -, -, -, -, ⟨hout, hframe⟩, -⟩ := hI
  obtain ⟨⟨⟨x5, x6, x13, x14, x28⟩, x7, x11, x12, x15, hmem, hsaved⟩, x16⟩ := hB
  have ho := h.orange
  have hm := h.hmem
  have hoal := h.oal
  have hj : e * N + p < E * N := by
    have : (e + 1) * N ≤ E * N := Nat.mul_le_mul_right N he
    rw [Nat.add_mul, Nat.one_mul] at this; omega
  cases hcode
  mrunAt [Stored, Fixed, Written, x5, x6, x13, x14, x28, x7, x11, x12, x15, x16]
  all_goals first
    | (simp only [BASE] at *; omega)
    | (refine ⟨⟨?_, ?_⟩, ?_⟩
       · intro j hj'
         rcases Nat.lt_succ_iff_lt_or_eq.mp hj' with hj' | rfl
         · rw [MachineState.getMem_setMem_ne (Mem.ofNat_ne (by simp only [BASE] at *; omega)
             (by simp only [BASE] at *; omega) (by omega))]
           simpa [hmem] using hout j hj'
         · rw [MachineState.getMem_setMem_eq, F_flat hp]
       · intro x hx
         rw [MachineState.getMem_setMem_ne (hx _ (Nat.lt_succ_self _))]
         simpa [hmem] using hframe x fun j hj' => hx j (by omega)
       · intro r hr
         simpa using hsaved r hr)

/-- The pointer updates: instructions 33 and 34. -/
theorem iNext (P : Prog) (c : Nat) (hcode : At P c) (_hc : BASE + 4 * (c + 40) < 2 ^ 64)
    (N q E a o k : Nat) (A : Nat → Nat) (s0 : MachineState) (e p : Nat) (s : MachineState)
    (hS : Stored s0 N q E a o k A e p s) :
    P.Spec (c + 33) s 2 (c + 35) (IInv s0 N q E a o k A e (p + 1)) := by
  obtain ⟨⟨x5, x6, x13, x14, x28⟩, x7, x11, x12, x15, ⟨hout, hframe⟩, hsaved⟩ := hS
  cases hcode
  mrunAt [IInv, Fixed, Written, x5, x6, x13, x14, x28, x7, x11, x12, x15, W_add]
  all_goals and_intros
  all_goals first
    | exact W_mod (by omega)
    | (intro j hj; simpa using hout j (by omega); done)
    | (intro x hx; simpa using hframe x (fun j hj => hx j (by omega)); done)
    | leaf_close

/-- The jump back to the coefficient loop's head: instruction 35. -/
theorem iJump (P : Prog) (c : Nat) (hcode : At P c) (_hc : BASE + 4 * (c + 40) < 2 ^ 64)
    (N q E a o k : Nat) (A : Nat → Nat) (s0 : MachineState) (e p : Nat) (s : MachineState)
    (hI : IInv s0 N q E a o k A e p s) : P.Spec (c + 35) s 1 (c + 15) (IInv s0 N q E a o k A e p) := by
  cases hcode
  mrunAt
  exact hI

/-- One coefficient, from the head back to the head. -/
theorem iBody (P : Prog) (c : Nat) (hcode : At P c) (hc : BASE + 4 * (c + 40) < 2 ^ 64)
    (d N q E a ka o k m : Nat) (A : Nat → Nat) (s0 : MachineState)
    (h : Pre P d N q E a ka o k m A s0) (e : Nat) (he : e < E) (p : Nat) (s : MachineState)
    (hp : p < N) (hI : IInv s0 N q E a o k A e p s) :
    P.Spec (c + 15) s 16 (c + 15) (IInv s0 N q E a o k A e (p + 1)) := by
  suffices H : P.Spec (c + 15) s (1 + (2 + (9 + (1 + (2 + 1))))) (c + 15)
      (IInv s0 N q E a o k A e (p + 1)) from Prog.Spec.mono H (by omega) fun _ h => h
  refine Prog.Spec.seq (iBranch P c hcode hc d N q E a ka o k m A s0 h e p s hp hI) fun s' h' => ?_
  subst h'
  refine Prog.Spec.seq (iShift P c hcode hc d N q E a ka o k m A s0 h e p s' hp hI) fun s2 h2 => ?_
  have htl := shift_lt (k := k) (p := p) h.hN
  have hlt : e * N + shift N k p < E * N ∨ N ≤ shift N k p := by
    by_cases ht : shift N k p < N
    · left
      have : (e + 1) * N ≤ E * N := Nat.mul_le_mul_right N he
      rw [Nat.add_mul, Nat.one_mul] at this; omega
    · right; omega
  have hjp : e * N + p < E * N := by
    have : (e + 1) * N ≤ E * N := Nat.mul_le_mul_right N he
    rw [Nat.add_mul, Nat.one_mul] at this; omega
  have hframe : ∀ j < E * N, s'.getMem (W (a + 8 * j)) = W (A j) := fun j hj => by
    rw [hI.2.2.2.2.2.1.2 _ ?_, h.hA j hj]
    intro i hi heq
    have hiE : i < E * N := by omega
    have := Mem.ofNat_inj (by have := h.arange; have := h.hmem; omega)
      (by have := h.orange; have := h.hmem; omega) heq
    have := h.disjA
    omega
  have hval : P.Spec (c + 19) s2 9 (c + 32)
      (fun t => Base s0 s' N q E a o k e p t ∧ t.getReg .x16 = W (val N q k A e p)) := by
    by_cases ht : shift N k p < N
    · have hj : e * N + shift N k p < E * N := by
        have : (e + 1) * N ≤ E * N := Nat.mul_le_mul_right N he
        rw [Nat.add_mul, Nat.one_mul] at this; omega
      exact iLow P c hcode hc d N q E a ka o k m A s0 s' h e p s2 he hp (hframe _ hj) ht h2
    · have hj : e * N + (shift N k p - N) < E * N := by
        have : (e + 1) * N ≤ E * N := Nat.mul_le_mul_right N he
        rw [Nat.add_mul, Nat.one_mul] at this; omega
      exact iHigh P c hcode hc d N q E a ka o k m A s0 s' h e p s2 he hp (hframe _ hj)
        (by omega) h2
  exact Prog.Spec.seq hval fun s3 h3 =>
    Prog.Spec.seq (iStore P c hcode hc d N q E a ka o k m A s0 s' h e p s3 he hp hI h3)
      fun s4 h4 =>
        Prog.Spec.seq (iNext P c hcode hc N q E a o k A s0 e p s4 h4) fun s5 h5 =>
          iJump P c hcode hc N q E a o k A s0 e (p + 1) s5 h5

/-- The end of an entry: instructions 15 and 16, to 36. -/
theorem iExit (P : Prog) (c : Nat) (hcode : At P c) (_hc : BASE + 4 * (c + 40) < 2 ^ 64)
    (N q E a o k : Nat) (A : Nat → Nat) (s0 : MachineState) (e : Nat) (s : MachineState)
    (hI : IInv s0 N q E a o k A e N s) :
    P.Spec (c + 15) s 2 (c + 36) (fun t => t = s) := by
  obtain ⟨⟨x5, -⟩, -, -, -, x15, -⟩ := hI
  cases hcode
  mrunAt [x5, x15]
  all_goals first
    | rfl
    | (exfalso; revert hbr; simp; done)

/-- The next entry: instructions 36 and 37. -/
theorem oNext (P : Prog) (c : Nat) (hcode : At P c) (_hc : BASE + 4 * (c + 40) < 2 ^ 64)
    (d N q E a ka o k m : Nat) (A : Nat → Nat) (s0 : MachineState)
    (h : Pre P d N q E a ka o k m A s0) (e : Nat) (s : MachineState) (he : e < E)
    (hI : IInv s0 N q E a o k A e N s) :
    P.Spec (c + 36) s 2 (c + 38) (OInv s0 N q E a o k A (e + 1)) := by
  obtain ⟨⟨x5, x6, x13, x14, x28⟩, x7, x11, x12, x15, ⟨hout, hframe⟩, hsaved⟩ := hI
  have e1 : e * N + N = (e + 1) * N := by ring
  rw [e1] at hout hframe x12
  cases hcode
  mrunAt [OInv, Fixed, Written, x5, x6, x13, x14, x28, x7, x11, x12, W_add]
  all_goals and_intros
  all_goals first
    | exact W_mod (by omega)
    | (rw [W_add]; congr 1; ring)
    | (intro j hj; simpa using hout j hj; done)
    | (intro x hx; simpa using hframe x hx; done)
    | leaf_close

/-- The jump back to the entry loop's head: instruction 38. -/
theorem oJump (P : Prog) (c : Nat) (hcode : At P c) (_hc : BASE + 4 * (c + 40) < 2 ^ 64)
    (N q E a o k : Nat) (A : Nat → Nat) (s0 : MachineState) (e : Nat) (s : MachineState)
    (hI : OInv s0 N q E a o k A e s) : P.Spec (c + 38) s 1 (c + 12) (OInv s0 N q E a o k A e) := by
  cases hcode
  mrunAt
  exact hI

/-- One entry, from the entry loop's head back to it. -/
theorem oBody (P : Prog) (c : Nat) (hcode : At P c) (hc : BASE + 4 * (c + 40) < 2 ^ 64)
    (d N q E a ka o k m : Nat) (A : Nat → Nat) (s0 : MachineState)
    (h : Pre P d N q E a ka o k m A s0) (e : Nat) (s : MachineState) (he : e < E)
    (hI : OInv s0 N q E a o k A e s) :
    P.Spec (c + 12) s (16 * N + 7) (c + 12) (OInv s0 N q E a o k A (e + 1)) := by
  suffices H : P.Spec (c + 12) s (1 + (1 + (((N - 0) * 16 + 2) + (2 + 1)))) (c + 12)
      (OInv s0 N q E a o k A (e + 1)) from Prog.Spec.mono H (by omega) fun _ h => h
  refine Prog.Spec.seq (oBranch P c hcode hc d N q E a ka o k m A s0 h e s he hI) fun s' h' => ?_
  subst h'
  exact Prog.Spec.seq (oStart P c hcode hc N q E a o k A s0 e s' hI) fun s2 h2 =>
    Prog.Spec.seq (Prog.Spec.loop (N := N)
      (fun p s3 hp hI' => iBody P c hcode hc d N q E a ka o k m A s0 h e he p s3 hp hI')
      (fun s3 hI' => Prog.Spec.mono (iExit P c hcode hc N q E a o k A s0 e s3 hI') le_rfl
        fun t ht => by subst ht; exact hI') 0 s2 (Nat.zero_le _) h2)
      fun s4 h4 =>
        Prog.Spec.seq (oNext P c hcode hc d N q E a ka o k m A s0 h e s4 he h4) fun s5 h5 =>
          oJump P c hcode hc N q E a o k A s0 (e + 1) s5 h5

/-- The exit: instructions 12, 13, and the jump to `ret`. -/
theorem oExit (P : Prog) (c : Nat) (hcode : At P c) (_hc : BASE + 4 * (c + 40) < 2 ^ 64)
    (N q E a o k : Nat) (A : Nat → Nat) (s0 : MachineState) (s : MachineState)
    (hI : OInv s0 N q E a o k A E s) : P.Spec (c + 12) s 2 (c + 39) (Post s0 N q E o k A) := by
  obtain ⟨-, x7, -, -, ⟨hout, hframe⟩, hsaved⟩ := hI
  have x7' : s.getReg .x7 = W 0 := by simpa using x7
  cases hcode
  mrunAt [Post, x7']
  all_goals first
    | exact ⟨hout, hframe, hsaved⟩
    | (exfalso; revert hbr; simp; done)

/-- **The specification.** From entry, the leaf reaches `ret` within `14 + E (16 N + 7)` steps. -/
theorem spec (P : Prog) (c : Nat) (hcode : At P c) (hc : BASE + 4 * (c + 40) < 2 ^ 64)
    (d N q E a ka o k m : Nat) (A : Nat → Nat) (s : MachineState)
    (h : Pre P d N q E a ka o k m A s) :
    P.Spec c s (14 + E * (16 * N + 7)) (c + 39) (Post s N q E o k A) := by
  refine Prog.Spec.mono (Prog.Spec.seq (pro1 P c hcode hc d N q E a ka o k m A s h) fun s1 h1 =>
    Prog.Spec.seq (pro2 P c hcode hc d N q E a ka o k m A s h s1 h1) fun s2 h2 =>
      Prog.Spec.seq (pro3 P c hcode hc d N q E a ka o k m A s h s2 h2) fun s3 h3 =>
        Prog.Spec.loop (N := E)
          (fun e s' he hI => oBody P c hcode hc d N q E a ka o k m A s h e s' he hI)
          (fun s' hI => oExit P c hcode hc N q E a o k A s s' hI) 0 s3 (Nat.zero_le _) h3) ?_
    fun _ hp => hp
  simp only [Nat.sub_zero]; omega

open MxxCore.Leaf

/-- The leaf's function on bytes: the product of each entry by `X^k`, `k` the first word of the
second input. -/
def bytesF (args : List Nat) (ins : List Bytes) : List Bytes :=
  match args, ins with
  | [N, q, E], [x, kb] =>
    [Bytes.u64s ((List.range (E * N)).map fun j =>
      F N q ((words kb).getD 0 0) (fun i => (words x).getD i 0) j)]
  | _, _ => []

/-- The leaf's domain: arguments `[N, q, E]` with `N = 2^m` (`m ≤ 58`) and `q < 2^63`, an input of
`E N` words below `q`, and an exponent of at least one word. -/
def bytesDom (args : List Nat) (ins : List Bytes) : Prop :=
  ∃ N q E x kb m, args = [N, q, E] ∧ ins = [x, kb] ∧ x.length = 8 * (E * N) ∧ 8 ≤ kb.length ∧
    N = 2 ^ m ∧ m ≤ 58 ∧ q < 2 ^ 63 ∧ ∀ v ∈ words x, v < q

theorem sound_bytes {P : Prog} {c : Nat} (hcode : At P c) (hc : BASE + 4 * (c + 40) < 2 ^ 64)
    (kk : Call) (s : MachineState) (he : Entry P kk s) (hdom : bytesDom kk.args (kk.ins.map (·.2)))
    (hlen : kk.outs.map (·.2) = (bytesF kk.args (kk.ins.map (·.2))).map List.length) :
    P.Spec c s (14 + kk.args.getD 2 0 * (16 * kk.args.headD 0 + 7)) (c + 39)
      (Exit kk (bytesF kk.args (kk.ins.map (·.2))) s) := by
  obtain ⟨N, q, E, x, kb, m, hargs, hins, hx, hkb, hN, hm, hq, hxq⟩ := hdom
  obtain ⟨d, args, ins, outs, sc⟩ := kk
  simp only at hargs hins hlen ⊢
  subst hargs
  rcases ins with _ | ⟨⟨a, x'⟩, _ | ⟨⟨ka, kb'⟩, _ | _⟩⟩ <;> simp at hins
  obtain ⟨rfl, rfl⟩ := hins
  simp only [List.map, bytesF, Mem.length_u64s, List.length_map, List.length_range] at hlen
  rcases outs with _ | ⟨⟨o, lo⟩, _ | _⟩ <;> simp at hlen
  subst hlen
  have hw : Call.words ⟨d, [N, q, E], [(a, x'), (ka, kb')], [(o, 8 * (E * N))], sc⟩ =
      [N, q, E, a, ka, o, sc.1] := rfl
  have hmem := he.hmem
  have ⟨aal, a1, a2⟩ := he.ial (a, x') (by simp)
  have ⟨kal, k1, k2⟩ := he.ial (ka, kb') (by simp)
  have ⟨oal, o1, o2⟩ := he.wal (o, 8 * (E * N)) (by simp [Call.written])
  simp only at aal a1 a2 kal k1 k2 oal o1 o2
  have dr := he.drange
  rw [hw] at dr
  simp only [List.length_cons, List.length_nil] at dr
  have dj : ∀ i (h : i < 7), s.getMem (W (d + 8 * i)) = W ([N, q, E, a, ka, o, sc.1][i]) :=
    fun i h => he.desc i (by rw [hw]; simpa using h)
  have hlx : (words x').length = E * N := by rw [length_words, hx]; omega
  have hgetD : ∀ (l : List Nat) (i : Nat) (h : i < l.length), l.getD i 0 = l[i] :=
    fun l i h => by simp [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem h]
  have hA : ∀ i < E * N, s.getMem (W (a + 8 * i)) = W ((words x').getD i 0) := fun i hi => by
    have hx' : Holds s a x' := he.ins (a, x') (by simp)
    rw [getMem_of_holds s a x' hx' i (by omega) aal (by omega), hgetD _ _ (by omega)]
  have hlk : 0 < (words kb').length := by rw [length_words]; omega
  have hk : s.getMem (W ka) = W ((words kb').getD 0 0) := by
    have hk' : Holds s ka kb' := he.ins (ka, kb') (by simp)
    have := getMem_of_holds s ka kb' hk' 0 (by omega) kal (by omega)
    simp only [Nat.mul_zero, Nat.add_zero] at this
    rw [this, hgetD _ _ hlk]
  have hk64 : (words kb').getD 0 0 < 2 ^ 64 := by
    rw [hgetD _ _ hlk]
    simp only [words, List.getElem_map, List.getElem_range]
    have h1 := leToNat_lt ((kb'.drop (8 * 0)).take 8)
    have h2 : ((kb'.drop (8 * 0)).take 8).length ≤ 8 := by simp
    calc _ < 256 ^ ((kb'.drop (8 * 0)).take 8).length := h1
      _ ≤ 256 ^ 8 := Nat.pow_le_pow_right (by norm_num) h2
      _ = 2 ^ 64 := by norm_num
  have dword : ∀ i (h : i < 7), s.getMem (W (d + 8 * i)) = W ([N, q, E, a, ka, o, sc.1][i]) := dj
  have d0 : s.getMem (W d) = W N := by
    have := dword 0 (by decide); simp only [Nat.mul_zero, Nat.add_zero] at this; exact this
  have d1 : s.getMem (W (d + 8)) = W q := by
    have := dword 1 (by decide); simpa only [List.getElem_cons_succ, List.getElem_cons_zero] using this
  have d2 : s.getMem (W (d + 16)) = W E := by
    have := dword 2 (by decide); simpa only [List.getElem_cons_succ, List.getElem_cons_zero] using this
  have d3 : s.getMem (W (d + 24)) = W a := by
    have := dword 3 (by decide); simpa only [List.getElem_cons_succ, List.getElem_cons_zero] using this
  have d4 : s.getMem (W (d + 32)) = W ka := by
    have := dword 4 (by decide); simpa only [List.getElem_cons_succ, List.getElem_cons_zero] using this
  have d5 : s.getMem (W (d + 40)) = W o := by
    have := dword 5 (by decide); simpa only [List.getElem_cons_succ, List.getElem_cons_zero] using this
  have pre : Pre P d N q E a ka o ((words kb').getD 0 0) m (fun i => (words x').getD i 0) s :=
    { hd := he.hd
      dN := d0
      dq := d1
      dE := d2
      da := d3
      dka := d4
      dout := d5
      hk := hk
      hk64 := hk64
      hA := hA
      hAq := fun i hi => by rw [hgetD _ _ (by omega)]; exact hxq _ (List.getElem_mem _)
      hq := hq
      hN := hN
      hm := hm
      hmem := hmem
      dal := he.dal
      aal := aal
      kal := kal
      oal := oal
      drange := ⟨dr.1, by omega⟩
      arange := ⟨a1, by omega⟩
      krange := ⟨k1, by omega⟩
      orange := ⟨o1, o2⟩
      disjA := by
        have := he.apartIn (o, 8 * (E * N)) (by simp [Call.written]) (a, x') (by simp)
        simp only [Apart] at this; omega
      disjK := by
        have := he.apartIn (o, 8 * (E * N)) (by simp [Call.written]) (ka, kb') (by simp)
        simp only [Apart] at this; omega }
  refine Prog.Spec.mono (spec P c hcode hc d N q E a ka o _ m _ s pre) (by simp) ?_
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
    have := hout' (o, 8 * (E * N)) (by simp [Call.written])
    exact getByte_of_frame s s' o (E * N) oal hframe (by omega) c' hc' this

/-- The specification in the shape of the core (`MxxCore.Leaf.Spec`), for every program that holds
the leaf's code at instruction `c`. -/
def byteSpec {P : Prog} {c : Nat} (hcode : At P c) (hc : BASE + 4 * (c + 40) < 2 ^ 64) :
    MxxCore.Leaf.Spec P c where
  ret := 39
  hret := hcode.i39
  steps args := 14 + args.getD 2 0 * (16 * args.headD 0 + 7)
  dom := bytesDom
  f := bytesF
  sound k s he hdom hlen := sound_bytes hcode hc k s he hdom hlen

end MxxLatticeLeaves.MultiplyMonomial
