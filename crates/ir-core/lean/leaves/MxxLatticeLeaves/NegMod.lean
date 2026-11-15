import MxxLatticeLeaves.Code.NegMod
import MxxCore.Glue
import MxxCore.Leaf
import MxxCore.Sym

/-!
# The specification of the leaf `lattice.neg_mod`

For any program `P` that holds the leaf's code at instruction `c` (`Code.NegMod.At P c`), from that
entry, with `a0` pointing at the descriptor `[n, q, a, o, scratch]`, the leaf reaches its `ret`
(instruction `c + 16`) within `6 + 10 n` steps, having written `(q - A k) mod q`, the negation, to
the doubleword at `o + 8 k` for every `k < n`, changed no other doubleword, and kept every register
the calling convention saves. The input is canonical residues (`A k < q < 2^63`) in an aligned,
confined array that the output array does not overlap. The proof follows `lattice.add_mod`'s
(`MxxLatticeLeaves.AddMod`), with one input.
-/

namespace MxxLatticeLeaves.NegMod

open MathcalA RiscvZkvm.Rv64 MxxLatticeLeaves.Code.NegMod

/-- The registers a leaf keeps: `ra`, `sp`, and `s0`–`s11`. -/
def Saved (s0 s : MachineState) : Prop :=
  ∀ r ∈ [Reg.x1, .x2, .x8, .x9, .x18, .x19, .x20, .x21, .x22, .x23, .x24, .x25, .x26, .x27],
    s.getReg r = s0.getReg r

/-- What the leaf assumes on entry. -/
structure Pre (P : Prog) (d n q a o : Nat) (A : Nat → Nat) (s : MachineState) : Prop where
  hd : s.getReg .x10 = W d
  dn : s.getMem (W d) = W n
  dq : s.getMem (W (d + 8)) = W q
  da : s.getMem (W (d + 16)) = W a
  dout : s.getMem (W (d + 24)) = W o
  hA : ∀ k < n, s.getMem (W (a + 8 * k)) = W (A k)
  hAq : ∀ k < n, A k < q
  hq : q < 2 ^ 63
  hmem : BASE + P.mem < 2 ^ 64
  dal : d % 8 = 0
  aal : a % 8 = 0
  oal : o % 8 = 0
  drange : BASE ≤ d ∧ d + 32 ≤ BASE + P.mem
  arange : BASE ≤ a ∧ a + 8 * n ≤ BASE + P.mem
  orange : BASE + P.bin.length ≤ o ∧ o + 8 * n ≤ BASE + P.mem
  disjA : o + 8 * n ≤ a ∨ a + 8 * n ≤ o

/-- What holds at `ret`. -/
def Post (s0 : MachineState) (n q o : Nat) (A : Nat → Nat) (s : MachineState) : Prop :=
  (∀ k < n, s.getMem (W (o + 8 * k)) = W ((q - A k) % q)) ∧
    (∀ x, (∀ k < n, x ≠ W (o + 8 * k)) → s.getMem x = s0.getMem x) ∧ Saved s0 s

/-- The state at the loop head (instruction 4) after `j` residues. -/
def Inv (s0 : MachineState) (n q a o : Nat) (A : Nat → Nat) (j : Nat) (s : MachineState) : Prop :=
  s.getReg .x5 = W (n - j) ∧ s.getReg .x6 = W q ∧ s.getReg .x7 = W (a + 8 * j) ∧
    s.getReg .x12 = W (o + 8 * j) ∧
    (∀ k < j, s.getMem (W (o + 8 * k)) = W ((q - A k) % q)) ∧
    (∀ x, (∀ k < j, x ≠ W (o + 8 * k)) → s.getMem x = s0.getMem x) ∧ Saved s0 s

-- The goals every segment leaves: address bounds, kept memory, and kept registers.
set_option hygiene false in
macro "leaf_close" : tactic => `(tactic| first
  | (simp only [BASE] at *; omega)
  | (intro x; simp; done)
  | (intro x; simpa using hmem x)
  | (intro r hr
     have hs := hsaved r hr
     simp only [List.mem_cons, List.mem_nil_iff, or_false] at hr
     rcases hr with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
       simpa using hs)
  | (intro hr
     rcases hr with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
       simpa using hsaved _ (by simp))
  | bv_omega
  | (simp; done))

theorem prologue (P : Prog) (c : Nat) (hcode : At P c) (_hc : BASE + 4 * (c + 17) < 2 ^ 64)
    (d n q a o : Nat) (A : Nat → Nat) (s : MachineState) (h : Pre P d n q a o A s) :
    P.Spec c s 4 (c + 4) (Inv s n q a o A 0) := by
  obtain ⟨hd, dn, dq, da, dout, -, -, -, hmem, dal, -, -, ⟨d1, d2⟩, -, -, -⟩ := h
  have hsaved : Saved s s := fun _ _ => rfl
  cases hcode
  mrunAt [Inv, Saved, hd, dn, dq, da, dout, W_add]
  all_goals and_intros
  all_goals first
    | (intro k hk; exact absurd hk (Nat.not_lt_zero k))
    | (intro _ _; trivial)
    | leaf_close

/-- The loop registers at residue `j`. -/
def Regs (n q a o j : Nat) (s : MachineState) : Prop :=
  s.getReg .x5 = W (n - j) ∧ s.getReg .x6 = W q ∧ s.getReg .x7 = W (a + 8 * j) ∧
    s.getReg .x12 = W (o + 8 * j)

/-- Inside an iteration: the loop registers, `x15 = v`, `x13 = A j`, and the memory of `s1`. -/
def Mid (s0 s1 : MachineState) (n q a o : Nat) (A : Nat → Nat) (j v : Nat) (s : MachineState) :
    Prop :=
  Regs n q a o j s ∧ s.getReg .x15 = W v ∧ s.getReg .x13 = W (A j) ∧
    (∀ x, s.getMem x = s1.getMem x) ∧ Saved s0 s

/-- Load and difference: instructions 4 to 8. -/
theorem seg1 (P : Prog) (c : Nat) (hcode : At P c) (_hc : BASE + 4 * (c + 17) < 2 ^ 64)
    (d n q a o : Nat) (A : Nat → Nat) (s0 : MachineState) (h : Pre P d n q a o A s0) (j : Nat)
    (s : MachineState) (hj : j < n) (hI : Inv s0 n q a o A j s) :
    P.Spec (c + 4) s 3 (c + 8) (Mid s0 s n q a o A j (q - A j)) := by
  obtain ⟨x5, x6, x7, x12, -, hframe, hsaved⟩ := hI
  have hAj : s.getMem (W (a + 8 * j)) = W (A j) := by
    rw [hframe _ ?_, h.hA j hj]
    intro k hk heq
    have := Mem.ofNat_inj (by have := h.arange; have := h.hmem; omega)
      (by have := h.orange; have := h.hmem; omega) heq
    have := h.disjA; omega
  have hm := h.hmem
  have ha := h.arange
  have haal := h.aal
  have hAq := h.hAq j hj
  have hq := h.hq
  cases hcode
  mrunAt [Mid, Regs, x5, x6, x7, x12, hAj, W_add]
  all_goals and_intros
  all_goals first
    | (exfalso
       have h0 : W (n - j) = W 0 := by simpa using hbr
       have := Mem.ofNat_inj (by omega) (by omega) h0
       omega)
    | leaf_close

/-- The reduction: instructions 8 to 11. -/
theorem seg2 (P : Prog) (c : Nat) (hcode : At P c) (_hc : BASE + 4 * (c + 17) < 2 ^ 64)
    (d n q a o : Nat) (A : Nat → Nat) (s0 s1 : MachineState) (h : Pre P d n q a o A s0)
    (j : Nat) (s : MachineState) (hj : j < n) (hM : Mid s0 s1 n q a o A j (q - A j) s) :
    P.Spec (c + 8) s 2 (c + 11) (Mid s0 s1 n q a o A j ((q - A j) % q)) := by
  obtain ⟨⟨x5, x6, x7, x12⟩, x15, x13, hmem, hsaved⟩ := hM
  have hAq := h.hAq j hj
  have hq := h.hq
  have e1 : A j = 0 → (q - A j) % q = 0 := fun h => by rw [h, Nat.sub_zero, Nat.mod_self]
  have e2 : A j ≠ 0 → (q - A j) % q = q - A j := fun h => Nat.mod_eq_of_lt (by omega)
  cases hcode
  mrunAt [Mid, Regs, x5, x6, x7, x12, x15, x13]
  all_goals and_intros
  all_goals first
    | (have hz : A j = 0 := by have := beq_iff_eq.mp hbr; bv_omega
       rw [e1 hz])
    | (have hz : A j ≠ 0 := by intro h0; simp [h0] at hbr
       rw [e2 hz])
    | leaf_close

/-- After the store of residue `j`: the loop registers, `j + 1` outputs, and the frame. -/
def Stored (s0 : MachineState) (n q a o : Nat) (A : Nat → Nat) (j : Nat) (s : MachineState) :
    Prop :=
  Regs n q a o j s ∧ (∀ k < j + 1, s.getMem (W (o + 8 * k)) = W ((q - A k) % q)) ∧
    (∀ x, (∀ k < j + 1, x ≠ W (o + 8 * k)) → s.getMem x = s0.getMem x) ∧ Saved s0 s

/-- The store: instruction 11. -/
theorem seg3 (P : Prog) (c : Nat) (hcode : At P c) (_hc : BASE + 4 * (c + 17) < 2 ^ 64)
    (d n q a o : Nat) (A : Nat → Nat) (s0 s1 : MachineState) (h : Pre P d n q a o A s0)
    (j : Nat) (s : MachineState) (hj : j < n) (hI : Inv s0 n q a o A j s1)
    (hM : Mid s0 s1 n q a o A j ((q - A j) % q) s) :
    P.Spec (c + 11) s 1 (c + 12) (Stored s0 n q a o A j) := by
  obtain ⟨-, -, -, -, hout, hframe, -⟩ := hI
  obtain ⟨⟨x5, x6, x7, x12⟩, x15, -, hmem, hsaved⟩ := hM
  have ho := h.orange
  have hm := h.hmem
  have hoal := h.oal
  cases hcode
  mrunAt [Stored, Regs, x5, x6, x7, x12, x15]
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

/-- The pointer updates: instructions 12 to 15. -/
theorem seg4 (P : Prog) (c : Nat) (hcode : At P c) (_hc : BASE + 4 * (c + 17) < 2 ^ 64)
    (n q a o : Nat) (A : Nat → Nat) (s0 : MachineState) (j : Nat) (s : MachineState) (hj : j < n)
    (hS : Stored s0 n q a o A j s) :
    P.Spec (c + 12) s 3 (c + 15) (Inv s0 n q a o A (j + 1)) := by
  obtain ⟨⟨x5, x6, x7, x12⟩, hout, hframe, hsaved⟩ := hS
  cases hcode
  mrunAt [Inv, x5, x6, x7, x12, W_add]
  refine ⟨W_mod (by omega), W_mod (by omega), W_mod (by omega),
    fun k hk => by simpa using hout k hk, fun x hx => by simpa using hframe x hx, fun r hr => ?_⟩
  have hs := hsaved r hr
  simp only [List.mem_cons, List.mem_nil_iff, or_false] at hr
  rcases hr with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
    simpa using hs

/-- The jump back: instruction 15. -/
theorem seg5 (P : Prog) (c : Nat) (hcode : At P c) (hc : BASE + 4 * (c + 17) < 2 ^ 64)
    (n q a o : Nat) (A : Nat → Nat) (s0 : MachineState) (j : Nat) (s : MachineState)
    (hI : Inv s0 n q a o A j s) : P.Spec (c + 15) s 1 (c + 4) (Inv s0 n q a o A j) := by
  cases hcode
  mrunAt
  exact hI

theorem body (P : Prog) (c : Nat) (hcode : At P c) (hc : BASE + 4 * (c + 17) < 2 ^ 64)
    (d n q a o : Nat) (A : Nat → Nat) (s0 : MachineState) (h : Pre P d n q a o A s0) (j : Nat)
    (s : MachineState) (hj : j < n) (hI : Inv s0 n q a o A j s) :
    P.Spec (c + 4) s 10 (c + 4) (Inv s0 n q a o A (j + 1)) :=
  Prog.Spec.seq (seg1 P c hcode hc d n q a o A s0 h j s hj hI) fun s' h' =>
    Prog.Spec.seq (seg2 P c hcode hc d n q a o A s0 s h j s' hj h') fun s'' h'' =>
      Prog.Spec.seq (seg3 P c hcode hc d n q a o A s0 s h j s'' hj hI h'') fun s3 h3 =>
        Prog.Spec.seq (seg4 P c hcode hc n q a o A s0 j s3 hj h3) fun s4 h4 =>
          seg5 P c hcode hc n q a o A s0 (j + 1) s4 h4

/-- The exit: instructions 4, 5, and the jump to `ret`. -/
theorem exit (P : Prog) (c : Nat) (hcode : At P c) (_hc : BASE + 4 * (c + 17) < 2 ^ 64)
    (n q o : Nat) (A : Nat → Nat) (s0 : MachineState) (a : Nat) (s : MachineState)
    (hI : Inv s0 n q a o A n s) : P.Spec (c + 4) s 2 (c + 16) (Post s0 n q o A) := by
  obtain ⟨x5, -, -, -, hout, hframe, hsaved⟩ := hI
  cases hcode
  mrunAt [Post, x5]
  all_goals first
    | exact ⟨hout, hframe, hsaved⟩
    | (exfalso; simp at hbr)

/-- **The specification.** From entry, the leaf reaches `ret` within `6 + 10 n` steps. -/
theorem spec (P : Prog) (c : Nat) (hcode : At P c) (hc : BASE + 4 * (c + 17) < 2 ^ 64)
    (d n q a o : Nat) (A : Nat → Nat) (s : MachineState) (h : Pre P d n q a o A s) :
    P.Spec c s (6 + 10 * n) (c + 16) (Post s n q o A) := by
  refine Prog.Spec.mono (Prog.Spec.seq (prologue P c hcode hc d n q a o A s h) fun s1 h1 =>
    Prog.Spec.loop (fun j s' hj hI => body P c hcode hc d n q a o A s h j s' hj hI)
      (fun s' hI => exit P c hcode hc n q o A s a s' hI) 0 s1 (Nat.zero_le n) h1) ?_ fun _ hp => hp
  omega

open MxxCore.Leaf

/-- The leaf's function on bytes: the encoded negations `(q - x k) mod q` of the words of its
input. -/
def bytesF (args : List Nat) (ins : List Bytes) : List Bytes :=
  match args, ins with
  | [n, q], [x] => [Bytes.u64s ((List.range n).map fun k => (q - (words x).getD k 0) % q)]
  | _, _ => []

/-- The leaf's domain: arguments `[n, q]` with `q < 2^63`, and one input of `n` words below `q`. -/
def bytesDom (args : List Nat) (ins : List Bytes) : Prop :=
  ∃ n q x, args = [n, q] ∧ ins = [x] ∧ x.length = 8 * n ∧ q < 2 ^ 63 ∧ ∀ v ∈ words x, v < q

theorem sound_bytes {P : Prog} {c : Nat} (hcode : At P c) (hc : BASE + 4 * (c + 17) < 2 ^ 64)
    (k : Call) (s : MachineState) (he : Entry P k s) (hdom : bytesDom k.args (k.ins.map (·.2)))
    (hlen : k.outs.map (·.2) = (bytesF k.args (k.ins.map (·.2))).map List.length) :
    P.Spec c s (6 + 10 * k.args.headD 0) (c + 16)
      (Exit k (bytesF k.args (k.ins.map (·.2))) s) := by
  obtain ⟨n, q, x, hargs, hins, hx, hq, hxq⟩ := hdom
  obtain ⟨d, args, ins, outs, sc⟩ := k
  simp only at hargs hins hlen ⊢
  subst hargs
  rcases ins with _ | ⟨⟨a, x'⟩, _ | _⟩ <;> simp at hins
  subst hins
  simp only [List.map, bytesF, Mem.length_u64s, List.length_map, List.length_range] at hlen
  rcases outs with _ | ⟨⟨o, lo⟩, _ | _⟩ <;> simp at hlen
  subst hlen
  have hw : Call.words ⟨d, [n, q], [(a, x')], [(o, 8 * n)], sc⟩ = [n, q, a, o, sc.1] := rfl
  have hmem := he.hmem
  have ⟨aal, a1, a2⟩ := he.ial (a, x') (by simp)
  have ⟨oal, o1, o2⟩ := he.wal (o, 8 * n) (by simp [Call.written])
  simp only at aal a1 a2 oal o1 o2
  have dr := he.drange
  rw [hw] at dr
  simp only [List.length_cons, List.length_nil] at dr
  have dj : ∀ i (h : i < 5), s.getMem (W (d + 8 * i)) = W ([n, q, a, o, sc.1][i]) :=
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
  have d2 : s.getMem (W (d + 16)) = W a := by
    have := dj 2 (by decide); simpa only [List.getElem_cons_succ, List.getElem_cons_zero] using this
  have d3 : s.getMem (W (d + 24)) = W o := by
    have := dj 3 (by decide); simpa only [List.getElem_cons_succ, List.getElem_cons_zero] using this
  have pre : Pre P d n q a o (fun i => (words x').getD i 0) s :=
    { hd := he.hd
      dn := d0
      dq := d1
      da := d2
      dout := d3
      hA := hA
      hAq := fun i hi => by rw [hgetD _ _ (by omega)]; exact hxq _ (List.getElem_mem _)
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
  refine Prog.Spec.mono (spec P c hcode hc d n q a o _ s pre) (by simp) ?_
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
def byteSpec {P : Prog} {c : Nat} (hcode : At P c) (hc : BASE + 4 * (c + 17) < 2 ^ 64) :
    MxxCore.Leaf.Spec P c where
  ret := 16
  hret := hcode.i16
  steps args := 6 + 10 * args.headD 0
  dom := bytesDom
  f := bytesF
  sound k s he hdom hlen := sound_bytes hcode hc k s he hdom hlen

end MxxLatticeLeaves.NegMod
