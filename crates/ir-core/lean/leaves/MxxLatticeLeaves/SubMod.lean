import MxxLatticeLeaves.Code.SubMod
import MxxCore.Glue
import MxxCore.Leaf
import MxxCore.Sym

/-!
# The specification of the leaf `lattice.sub_mod`

For any program `P` that holds the leaf's code at instruction `c` (`Code.AddMod.At P c`), from that
entry, with `a0` pointing at the descriptor `[n, q, a, b, o, scratch]`, the leaf reaches its `ret`
(instruction `c + 19`) within `7 + 12 n` steps, having written `(A k - B k) mod q` to the
doubleword at `o + 8 k` for every `k < n`, with `(A k + q - B k) mod q` the difference, changed no other doubleword, and kept every register
the calling convention saves. The inputs are canonical residues (`A k, B k < q < 2^63`) in
aligned, confined arrays that the output array does not overlap.
-/

namespace MxxLatticeLeaves.SubMod

open MathcalA RiscvZkvm.Rv64 MxxLatticeLeaves.Code.SubMod

/-- The registers a leaf keeps: `ra`, `sp`, and `s0`–`s11`. -/
def Saved (s0 s : MachineState) : Prop :=
  ∀ r ∈ [Reg.x1, .x2, .x8, .x9, .x18, .x19, .x20, .x21, .x22, .x23, .x24, .x25, .x26, .x27],
    s.getReg r = s0.getReg r

/-- What the leaf assumes on entry, in a machine of `mem` bytes. -/
structure Pre (P : Prog) (d n q a b o : Nat) (A B : Nat → Nat) (s : MachineState) : Prop where
  hd : s.getReg .x10 = W d
  dn : s.getMem (W d) = W n
  dq : s.getMem (W (d + 8)) = W q
  da : s.getMem (W (d + 16)) = W a
  db : s.getMem (W (d + 24)) = W b
  dout : s.getMem (W (d + 32)) = W o
  hA : ∀ k < n, s.getMem (W (a + 8 * k)) = W (A k)
  hB : ∀ k < n, s.getMem (W (b + 8 * k)) = W (B k)
  hAq : ∀ k < n, A k < q
  hBq : ∀ k < n, B k < q
  hq : q < 2 ^ 63
  hmem : BASE + P.mem < 2 ^ 64
  dal : d % 8 = 0
  aal : a % 8 = 0
  bal : b % 8 = 0
  oal : o % 8 = 0
  drange : BASE ≤ d ∧ d + 40 ≤ BASE + P.mem
  arange : BASE ≤ a ∧ a + 8 * n ≤ BASE + P.mem
  brange : BASE ≤ b ∧ b + 8 * n ≤ BASE + P.mem
  orange : BASE + P.bin.length ≤ o ∧ o + 8 * n ≤ BASE + P.mem
  disjA : o + 8 * n ≤ a ∨ a + 8 * n ≤ o
  disjB : o + 8 * n ≤ b ∨ b + 8 * n ≤ o

/-- What holds at `ret`. -/
def Post (s0 : MachineState) (n q o : Nat) (A B : Nat → Nat) (s : MachineState) : Prop :=
  (∀ k < n, s.getMem (W (o + 8 * k)) = W ((A k + q - B k) % q)) ∧
    (∀ x, (∀ k < n, x ≠ W (o + 8 * k)) → s.getMem x = s0.getMem x) ∧ Saved s0 s

/-- The state at the loop head (instruction 5) after `j` residues. -/
def Inv (s0 : MachineState) (n q a b o : Nat) (A B : Nat → Nat) (j : Nat) (s : MachineState) :
    Prop :=
  s.getReg .x5 = W (n - j) ∧ s.getReg .x6 = W q ∧ s.getReg .x7 = W (a + 8 * j) ∧
    s.getReg .x11 = W (b + 8 * j) ∧ s.getReg .x12 = W (o + 8 * j) ∧
    (∀ k < j, s.getMem (W (o + 8 * k)) = W ((A k + q - B k) % q)) ∧
    (∀ x, (∀ k < j, x ≠ W (o + 8 * k)) → s.getMem x = s0.getMem x) ∧ Saved s0 s

theorem prologue (P : Prog) (c : Nat) (hcode : At P c) (_hc : BASE + 4 * (c + 20) < 2 ^ 64)
    (d n q a b o : Nat) (A B : Nat → Nat) (s : MachineState)
    (h : Pre P d n q a b o A B s) :
    P.Spec c s 5 (c + 5) (Inv s n q a b o A B 0) := by
  obtain ⟨hd, dn, dq, da, db, dout, -, -, -, -, -, hmem, dal, -, -, -, ⟨d1, d2⟩, -, -, -, -, -⟩ := h
  cases hcode
  mrunAt [Inv, Saved, hd, dn, dq, da, db, dout, W_add]
  all_goals first
    | (refine ⟨by simp, by simp, by simp, by simp, fun k hk => absurd hk (Nat.not_lt_zero k),
        fun _ _ => trivial, ?_⟩
       intro r hr
       simp only [List.mem_cons, List.mem_nil_iff, or_false] at hr
       rcases hr with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
         simp)
    | (simp only [BASE] at *; omega)

/-- The loop registers at residue `j`. -/
def Regs (n q a b o j : Nat) (s : MachineState) : Prop :=
  s.getReg .x5 = W (n - j) ∧ s.getReg .x6 = W q ∧ s.getReg .x7 = W (a + 8 * j) ∧
    s.getReg .x11 = W (b + 8 * j) ∧ s.getReg .x12 = W (o + 8 * j)

/-- Inside an iteration: the loop registers, `x15 = v`, and the memory of `s1`, the state at the
loop head. The iteration is split here so each symbolic run stays small for the kernel. -/
def Mid (s0 s1 : MachineState) (n q a b o : Nat) (A B : Nat → Nat) (j v : Nat)
    (s : MachineState) : Prop :=
  Regs n q a b o j s ∧ s.getReg .x15 = W v ∧ s.getReg .x13 = W (A j) ∧ s.getReg .x14 = W (B j) ∧
    (∀ x, s.getMem x = s1.getMem x) ∧ Saved s0 s

/-- Loads and difference: instructions 5 to 10. -/
theorem seg1 (P : Prog) (c : Nat) (hcode : At P c) (_hc : BASE + 4 * (c + 20) < 2 ^ 64)
    (d n q a b o : Nat) (A B : Nat → Nat) (s0 : MachineState)
    (h : Pre P d n q a b o A B s0) (j : Nat) (s : MachineState) (hj : j < n)
    (hI : Inv s0 n q a b o A B j s) :
    P.Spec (c + 5) s 4 (c + 10) (Mid s0 s n q a b o A B j (A j + 2 ^ 64 - B j)) := by
  obtain ⟨x5, x6, x7, x11, x12, -, hframe, hsaved⟩ := hI
  have hAj : s.getMem (W (a + 8 * j)) = W (A j) := by
    rw [hframe _ ?_, h.hA j hj]
    intro k hk heq
    have := Mem.ofNat_inj (by have := h.arange; have := h.hmem; omega)
      (by have := h.orange; have := h.hmem; omega) heq
    have := h.disjA; omega
  have hBj : s.getMem (W (b + 8 * j)) = W (B j) := by
    rw [hframe _ ?_, h.hB j hj]
    intro k hk heq
    have := Mem.ofNat_inj (by have := h.brange; have := h.hmem; omega)
      (by have := h.orange; have := h.hmem; omega) heq
    have := h.disjB; omega
  have hm := h.hmem
  have ha := h.arange
  have hb := h.brange
  have haal := h.aal
  have hbal := h.bal
  have hAq := h.hAq j hj
  have hBq := h.hBq j hj
  have hq := h.hq
  cases hcode
  mrunAt [Mid, Regs, x5, x6, x7, x11, x12, hAj, hBj, W_add]
  all_goals and_intros
  all_goals first
    | (simp only [BASE] at *; omega)
    | (exfalso
       have h0 : W (n - j) = W 0 := by simpa using hbr
       have := Mem.ofNat_inj (by omega) (by omega) h0
       omega)
    | (intro x; simp; done)
    | (intro r hr
       have hs := hsaved r hr
       simp only [List.mem_cons, List.mem_nil_iff, or_false] at hr
       rcases hr with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
         simpa using hs)
    | (intro hr
       rcases hr with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
         simpa using hsaved _ (by simp))
    | bv_omega
    | simp

/-- The reduction: instructions 10 to 13. -/
theorem seg2 (P : Prog) (c : Nat) (hcode : At P c) (_hc : BASE + 4 * (c + 20) < 2 ^ 64)
    (d n q a b o : Nat) (A B : Nat → Nat) (s0 s1 : MachineState)
    (h : Pre P d n q a b o A B s0) (j : Nat) (s : MachineState) (hj : j < n)
    (hM : Mid s0 s1 n q a b o A B j (A j + 2 ^ 64 - B j) s) :
    P.Spec (c + 10) s 2 (c + 13) (Mid s0 s1 n q a b o A B j ((A j + q - B j) % q)) := by
  obtain ⟨⟨x5, x6, x7, x11, x12⟩, x15, x13, x14, hmem, hsaved⟩ := hM
  have hAq := h.hAq j hj
  have hBq := h.hBq j hj
  have hq := h.hq
  have e1 : A j < B j → (A j + q - B j) % q = A j + q - B j := fun h => Nat.mod_eq_of_lt (by omega)
  have e2 : B j ≤ A j → (A j + q - B j) % q = A j - B j := fun h => by
    rw [show A j + q - B j = (A j - B j) + q by omega, Nat.add_mod_right,
      Nat.mod_eq_of_lt (by omega)]
  cases hcode
  mrunAt [Mid, Regs, x5, x6, x7, x11, x12, x15, x13, x14]
  all_goals and_intros
  all_goals first
    | (intro x; simpa using hmem x)
    | (intro r hr
       have hs := hsaved r hr
       simp only [List.mem_cons, List.mem_nil_iff, or_false] at hr
       rcases hr with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
         simpa using hs)
    | (simp [BitVec.ult] at hbr
       first
         | (rw [e1 (by omega)]; bv_omega)
         | (rw [e2 (by omega)]; bv_omega))
    | simp
    | bv_omega

/-- After the store of residue `j`: the loop registers, `j + 1` outputs, and the frame. -/
def Stored (s0 : MachineState) (n q a b o : Nat) (A B : Nat → Nat) (j : Nat) (s : MachineState) :
    Prop :=
  Regs n q a b o j s ∧ (∀ k < j + 1, s.getMem (W (o + 8 * k)) = W ((A k + q - B k) % q)) ∧
    (∀ x, (∀ k < j + 1, x ≠ W (o + 8 * k)) → s.getMem x = s0.getMem x) ∧ Saved s0 s

/-- The store: instruction 13. -/
theorem seg3 (P : Prog) (c : Nat) (hcode : At P c) (_hc : BASE + 4 * (c + 20) < 2 ^ 64)
    (d n q a b o : Nat) (A B : Nat → Nat) (s0 s1 : MachineState)
    (h : Pre P d n q a b o A B s0) (j : Nat) (s : MachineState) (hj : j < n)
    (hI : Inv s0 n q a b o A B j s1)
    (hM : Mid s0 s1 n q a b o A B j ((A j + q - B j) % q) s) :
    P.Spec (c + 13) s 1 (c + 14) (Stored s0 n q a b o A B j) := by
  obtain ⟨-, -, -, -, -, hout, hframe, -⟩ := hI
  obtain ⟨⟨x5, x6, x7, x11, x12⟩, x15, -, -, hmem, hsaved⟩ := hM
  have ho := h.orange
  have hm := h.hmem
  have hoal := h.oal
  cases hcode
  mrunAt [Stored, Regs, x5, x6, x7, x11, x12, x15]
  all_goals first
    | (simp only [BASE] at *; omega)
    | (refine ⟨?_, ?_, ?_⟩
       · intro k hk
         rcases Nat.lt_succ_iff_lt_or_eq.mp hk with hk | rfl
         · rw [MachineState.getMem_setMem_ne (Mem.ofNat_ne (by simp only [BASE] at *; omega)
             (by simp only [BASE] at *; omega) (by omega))]
           simpa [hmem] using hout k hk
         · rw [MachineState.getMem_setMem_eq]
       · intro x hx
         rw [MachineState.getMem_setMem_ne (hx _ (Nat.lt_succ_self _))]
         simpa [hmem] using hframe x fun k hk => hx k (by omega)
       · intro r hr
         simpa using hsaved r hr)

/-- The pointer updates: instructions 14 to 18. -/
theorem seg4 (P : Prog) (c : Nat) (hcode : At P c) (_hc : BASE + 4 * (c + 20) < 2 ^ 64)
    (n q a b o : Nat) (A B : Nat → Nat) (s0 : MachineState)
    (j : Nat) (s : MachineState) (hj : j < n) (hS : Stored s0 n q a b o A B j s) :
    P.Spec (c + 14) s 4 (c + 18) (Inv s0 n q a b o A B (j + 1)) := by
  obtain ⟨⟨x5, x6, x7, x11, x12⟩, hout, hframe, hsaved⟩ := hS
  cases hcode
  mrunAt [Inv, x5, x6, x7, x11, x12, W_add]
  refine ⟨W_mod (by omega), W_mod (by omega), W_mod (by omega), W_mod (by omega),
    fun k hk => by simpa using hout k hk, fun x hx => by simpa using hframe x hx, fun r hr => ?_⟩
  have hs := hsaved r hr
  simp only [List.mem_cons, List.mem_nil_iff, or_false] at hr
  rcases hr with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
    simpa using hs

/-- The jump back: instruction 18. -/
theorem seg5 (P : Prog) (c : Nat) (hcode : At P c) (hc : BASE + 4 * (c + 20) < 2 ^ 64)
    (n q a b o : Nat) (A B : Nat → Nat) (s0 : MachineState) (j : Nat)
    (s : MachineState) (hI : Inv s0 n q a b o A B j s) :
    P.Spec (c + 18) s 1 (c + 5) (Inv s0 n q a b o A B j) := by
  cases hcode
  mrunAt
  exact hI

theorem body (P : Prog) (c : Nat) (hcode : At P c) (hc : BASE + 4 * (c + 20) < 2 ^ 64)
    (d n q a b o : Nat) (A B : Nat → Nat) (s0 : MachineState)
    (h : Pre P d n q a b o A B s0) (j : Nat) (s : MachineState) (hj : j < n)
    (hI : Inv s0 n q a b o A B j s) :
    P.Spec (c + 5) s 12 (c + 5) (Inv s0 n q a b o A B (j + 1)) :=
  Prog.Spec.seq (seg1 P c hcode hc d n q a b o A B s0 h j s hj hI) fun s' h' =>
    Prog.Spec.seq (seg2 P c hcode hc d n q a b o A B s0 s h j s' hj h') fun s'' h'' =>
      Prog.Spec.seq (seg3 P c hcode hc d n q a b o A B s0 s h j s'' hj hI h'') fun s3 h3 =>
        Prog.Spec.seq (seg4 P c hcode hc n q a b o A B s0 j s3 hj h3) fun s4 h4 =>
          seg5 P c hcode hc n q a b o A B s0 (j + 1) s4 h4

/-- The exit: instructions 5, 6, and the jump to `ret`. -/
theorem exit (P : Prog) (c : Nat) (hcode : At P c) (_hc : BASE + 4 * (c + 20) < 2 ^ 64)
    (n q o : Nat) (A B : Nat → Nat) (s0 : MachineState) (a b : Nat)
    (s : MachineState) (hI : Inv s0 n q a b o A B n s) :
    P.Spec (c + 5) s 2 (c + 19) (Post s0 n q o A B) := by
  obtain ⟨x5, -, -, -, -, hout, hframe, hsaved⟩ := hI
  cases hcode
  mrunAt [Post, x5]
  all_goals first
    | exact ⟨hout, hframe, hsaved⟩
    | (exfalso; simp at hbr)

/-- **The specification.** From entry, the leaf reaches `ret` within `7 + 12 n` steps. -/
theorem spec (P : Prog) (c : Nat) (hcode : At P c) (hc : BASE + 4 * (c + 20) < 2 ^ 64)
    (d n q a b o : Nat) (A B : Nat → Nat) (s : MachineState)
    (h : Pre P d n q a b o A B s) :
    P.Spec c s (7 + 12 * n) (c + 19) (Post s n q o A B) := by
  refine Prog.Spec.mono (Prog.Spec.seq (prologue P c hcode hc d n q a b o A B s h) fun s1 h1 =>
    Prog.Spec.loop (fun j s' hj hI => body P c hcode hc d n q a b o A B s h j s' hj hI)
      (fun s' hI => exit P c hcode hc n q o A B s a b s' hI) 0 s1 (Nat.zero_le n) h1) ?_ fun _ hp => hp
  omega

/-- A call of the leaf at `c` from instruction `j` (`auipc ra, 0`; `jalr ra, off(ra)`), back at
`j + 2` through its `ret`. -/
theorem call {P : Prog} {c : Nat} (hcode : At P c) (hc : BASE + 4 * (c + 20) < 2 ^ 64)
    (j : Nat) (off : BitVec 12) (s : MachineState) (d n q a b o : Nat) (A B : Nat → Nat)
    (hauipc : P.instrs[j]? = some (.AUIPC .x1 0x0#20))
    (hjalr : P.instrs[j + 1]? = some (.JALR .x1 .x1 off))
    (ht : (pcOf j + ((0x0#20 : BitVec 20).zeroExtend 32 <<< 12 : BitVec 32).signExtend 64 +
      signExtend12 off) &&& ~~~1#64 = pcOf c)
    (hback : (pcOf (j + 1 + 1) + signExtend12 0x0#12) &&& ~~~1#64 = pcOf (j + 2))
    (h : Pre P d n q a b o A B s) :
    P.Spec j s (2 + ((7 + 12 * n) + 1)) (j + 2)
      (Post (s.setReg .x1 (pcOf (j + 1 + 1))) n q o A B) := by
  rw [show 2 + ((7 + 12 * n) + 1) = ((7 + 12 * n) + 1) + 1 + 1 by omega]
  refine MxxCore.Glue.auipc hauipc ?_
  refine MxxCore.Glue.jalr (t := c) hjalr (by simpa using ht) ?_
  rw [Sym.setReg_setReg_same]
  have hpre : Pre P d n q a b o A B (s.setReg .x1 (pcOf (j + 1 + 1))) := by
    obtain ⟨hd, dn, dq, da, db, dout, hA, hB, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12,
      r13, r14⟩ := h
    exact ⟨by simpa using hd, by simpa using dn, by simpa using dq, by simpa using da,
      by simpa using db, by simpa using dout, fun k hk => by simpa using hA k hk,
      fun k hk => by simpa using hB k hk, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13,
      r14⟩
  refine Prog.Spec.seq (spec P c hcode hc d n q a b o A B _ hpre) fun s' hs' => ?_
  have hra : s'.getReg .x1 = pcOf (j + 1 + 1) := by
    rw [hs'.2.2 .x1 (by simp)]
    simp
  refine MxxCore.Glue.jalr (t := j + 2) hcode.i19 (by rw [hra]; exact hback) ?_
  exact Prog.Spec.done (by simpa using hs')

open MxxCore.Leaf

/-- The leaf's function on bytes: the encoded differences `(x k + q - y k) mod q` of the words
of its inputs. -/
def bytesF (args : List Nat) (ins : List Bytes) : List Bytes :=
  match args, ins with
  | [n, q], [x, y] =>
    [Bytes.u64s ((List.range n).map fun k => ((words x).getD k 0 + q - (words y).getD k 0) % q)]
  | _, _ => []

/-- The leaf's domain: arguments `[n, q]` with `q < 2^63`, and two inputs of `n` words below
`q`. -/
def bytesDom (args : List Nat) (ins : List Bytes) : Prop :=
  ∃ n q x y, args = [n, q] ∧ ins = [x, y] ∧ x.length = 8 * n ∧ y.length = 8 * n ∧ q < 2 ^ 63 ∧
    (∀ v ∈ words x, v < q) ∧ (∀ v ∈ words y, v < q)

theorem sound_bytes {P : Prog} {c : Nat} (hcode : At P c) (hc : BASE + 4 * (c + 20) < 2 ^ 64)
    (k : Call) (s : MachineState) (he : Entry P k s) (hdom : bytesDom k.args (k.ins.map (·.2)))
    (hlen : k.outs.map (·.2) = (bytesF k.args (k.ins.map (·.2))).map List.length) :
    P.Spec c s (7 + 12 * k.args.headD 0) (c + 19)
      (Exit k (bytesF k.args (k.ins.map (·.2))) s) := by
  obtain ⟨n, q, x, y, hargs, hins, hx, hy, hq, hxq, hyq⟩ := hdom
  obtain ⟨d, args, ins, outs, sc⟩ := k
  simp only at hargs hins hlen ⊢
  subst hargs
  rcases ins with _ | ⟨⟨a, x'⟩, _ | ⟨⟨b, y'⟩, _ | _⟩⟩ <;> simp at hins
  obtain ⟨h1, h2⟩ := hins
  subst x' y'
  simp only [List.map, bytesF, Mem.length_u64s,
    List.length_map, List.length_range] at hlen
  rcases outs with _ | ⟨⟨o, lo⟩, _ | _⟩ <;> simp at hlen
  subst hlen
  have hw : Call.words ⟨d, [n, q], [(a, x), (b, y)], [(o, 8 * n)], sc⟩ = [n, q, a, b, o, sc.1] :=
    rfl
  have hmem := he.hmem
  have ⟨aal, a1, a2⟩ := he.ial (a, x) (by simp)
  have ⟨bal, b1, b2⟩ := he.ial (b, y) (by simp)
  have ⟨oal, o1, o2⟩ := he.wal (o, 8 * n) (by simp [Call.written])
  simp only at aal a1 a2 bal b1 b2 oal o1 o2
  have dr := he.drange
  rw [hw] at dr
  simp only [List.length_cons, List.length_nil] at dr
  have dj : ∀ i (h : i < 6), s.getMem (W (d + 8 * i)) = W ([n, q, a, b, o, sc.1][i]) :=
    fun i h => he.desc i (by rw [hw]; simpa using h)
  have hlx : (words x).length = n := by rw [length_words, hx]; omega
  have hly : (words y).length = n := by rw [length_words, hy]; omega
  have hgetD : ∀ (l : List Nat) (i : Nat) (h : i < l.length), l.getD i 0 = l[i] :=
    fun l i h => by simp [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem h]
  have hA : ∀ i < n, s.getMem (W (a + 8 * i)) = W ((words x).getD i 0) := fun i hi => by
    have hx' : Holds s a x := he.ins (a, x) (by simp)
    rw [getMem_of_holds s a x hx' i (by omega) aal (by omega), hgetD _ _ (by omega)]
  have hB : ∀ i < n, s.getMem (W (b + 8 * i)) = W ((words y).getD i 0) := fun i hi => by
    have hy' : Holds s b y := he.ins (b, y) (by simp)
    rw [getMem_of_holds s b y hy' i (by omega) bal (by omega), hgetD _ _ (by omega)]
  have d0 : s.getMem (W d) = W n := by
    have := dj 0 (by decide)
    simp only [Nat.mul_zero, Nat.add_zero] at this
    exact this
  have d1 : s.getMem (W (d + 8)) = W q := by
    have := dj 1 (by decide); simpa only [List.getElem_cons_succ, List.getElem_cons_zero] using this
  have d2 : s.getMem (W (d + 16)) = W a := by
    have := dj 2 (by decide); simpa only [List.getElem_cons_succ, List.getElem_cons_zero] using this
  have d3 : s.getMem (W (d + 24)) = W b := by
    have := dj 3 (by decide); simpa only [List.getElem_cons_succ, List.getElem_cons_zero] using this
  have d4 : s.getMem (W (d + 32)) = W o := by
    have := dj 4 (by decide); simpa only [List.getElem_cons_succ, List.getElem_cons_zero] using this
  have pre : Pre P d n q a b o (fun i => (words x).getD i 0) (fun i => (words y).getD i 0) s :=
    { hd := he.hd
      dn := d0
      dq := d1
      da := d2
      db := d3
      dout := d4
      hA := hA
      hB := hB
      hAq := fun i hi => by rw [hgetD _ _ (by omega)]; exact hxq _ (List.getElem_mem _)
      hBq := fun i hi => by rw [hgetD _ _ (by omega)]; exact hyq _ (List.getElem_mem _)
      hq := hq
      hmem := hmem
      dal := he.dal
      aal := aal
      bal := bal
      oal := oal
      drange := ⟨dr.1, by omega⟩
      arange := ⟨a1, by omega⟩
      brange := ⟨b1, by omega⟩
      orange := ⟨o1, o2⟩
      disjA := by
        have := he.apartIn (o, 8 * n) (by simp [Call.written]) (a, x) (by simp)
        simp only [Apart] at this; omega
      disjB := by
        have := he.apartIn (o, 8 * n) (by simp [Call.written]) (b, y) (by simp)
        simp only [Apart] at this; omega }
  refine Prog.Spec.mono (spec P c hcode hc d n q a b o _ _ s pre) (by simp) ?_
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
def byteSpec {P : Prog} {c : Nat} (hcode : At P c) (hc : BASE + 4 * (c + 20) < 2 ^ 64) :
    MxxCore.Leaf.Spec P c where
  ret := 19
  hret := hcode.i19
  steps args := 7 + 12 * args.headD 0
  dom := bytesDom
  f := bytesF
  sound k s he hdom hlen := sound_bytes hcode hc k s he hdom hlen

end MxxLatticeLeaves.SubMod
