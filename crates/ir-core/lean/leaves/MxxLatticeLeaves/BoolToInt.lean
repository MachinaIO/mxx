import MxxLatticeLeaves.Code.BoolToInt
import MxxCore.Glue
import MxxCore.Leaf
import MxxCore.Sym

/-!
# The specification of the leaf `lattice.bool_to_int`

For any program `P` that holds the leaf's code at instruction `c` (`Code.BoolToInt.At P c`), from
that entry, with `a0` pointing at the descriptor `[w, a, o, scratch]` (`w ≥ 1`), the leaf reaches
its `ret` (instruction `c + 14`) within `5 + 5 w` steps, having written the word `v` at `a` to
the doubleword at `o` and zero to the doublewords at `o + 8 k` for `0 < k < w` (the `w`-word
integer of the word), changed no other doubleword, and kept every register the calling
convention saves. The output array is aligned, confined, and apart from the input word.
-/

namespace MxxLatticeLeaves.BoolToInt

open MathcalA RiscvZkvm.Rv64 MxxLatticeLeaves.Code.BoolToInt

/-- The registers a leaf keeps: `ra`, `sp`, and `s0`–`s11`. -/
def Saved (s0 s : MachineState) : Prop :=
  ∀ r ∈ [Reg.x1, .x2, .x8, .x9, .x18, .x19, .x20, .x21, .x22, .x23, .x24, .x25, .x26, .x27],
    s.getReg r = s0.getReg r

/-- Word `k` of the output: `v`, then zeros. -/
def out (v k : Nat) : Nat := if k = 0 then v else 0

/-- What the leaf assumes on entry. -/
structure Pre (P : Prog) (d w a o v : Nat) (s : MachineState) : Prop where
  hd : s.getReg .x10 = W d
  dw : s.getMem (W d) = W w
  da : s.getMem (W (d + 8)) = W a
  dout : s.getMem (W (d + 16)) = W o
  hv : s.getMem (W a) = W v
  hw : 1 ≤ w
  hmem : BASE + P.mem < 2 ^ 64
  dal : d % 8 = 0
  aal : a % 8 = 0
  oal : o % 8 = 0
  drange : BASE ≤ d ∧ d + 24 ≤ BASE + P.mem
  arange : BASE ≤ a ∧ a + 8 ≤ BASE + P.mem
  orange : BASE + P.bin.length ≤ o ∧ o + 8 * w ≤ BASE + P.mem

/-- What holds at `ret`. -/
def Post (s0 : MachineState) (w o v : Nat) (s : MachineState) : Prop :=
  (∀ k < w, s.getMem (W (o + 8 * k)) = W (out v k)) ∧
    (∀ x, (∀ k < w, x ≠ W (o + 8 * k)) → s.getMem x = s0.getMem x) ∧ Saved s0 s

/-- The state at the loop head (instruction 8) after `i` iterations: `i + 1` words written. -/
def Inv (s0 : MachineState) (w o v : Nat) (i : Nat) (s : MachineState) : Prop :=
  s.getReg .x5 = W (w - 1 - i) ∧ s.getReg .x12 = W (o + 8 * (i + 1)) ∧ s.getReg .x15 = W 0 ∧
    (∀ k < i + 1, s.getMem (W (o + 8 * k)) = W (out v k)) ∧
    (∀ x, (∀ k < i + 1, x ≠ W (o + 8 * k)) → s.getMem x = s0.getMem x) ∧ Saved s0 s

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

/-- After the loads: `x5 = w`, `x6 = a`, `x15 = v`, `x12 = o`, memory and saved registers as on
entry. -/
def Loaded (s0 : MachineState) (w a o v : Nat) (s : MachineState) : Prop :=
  s.getReg .x5 = W w ∧ s.getReg .x6 = W a ∧ s.getReg .x15 = W v ∧ s.getReg .x12 = W o ∧
    (∀ x, s.getMem x = s0.getMem x) ∧ Saved s0 s

/-- The loads: instructions 0 to 4. -/
theorem pro1 (P : Prog) (c : Nat) (hcode : At P c) (_hc : BASE + 4 * (c + 15) < 2 ^ 64)
    (d w a o v : Nat) (s : MachineState) (h : Pre P d w a o v s) :
    P.Spec c s 4 (c + 4) (Loaded s w a o v) := by
  obtain ⟨hd, dw, da, dout, hv, -, hmem, dal, aal, -, ⟨d1, d2⟩, ⟨a1, a2⟩, -⟩ := h
  have hsaved : Saved s s := fun _ _ => rfl
  cases hcode
  mrunAt [Loaded, Saved, hd, dw, da, dout, hv, W_add]
  all_goals and_intros
  all_goals first
    | (intro _; rfl)
    | (intro _ _; rfl)
    | leaf_close

/-- The first word and the setup of the loop: instructions 4 to 8. -/
theorem pro2 (P : Prog) (c : Nat) (hcode : At P c) (_hc : BASE + 4 * (c + 15) < 2 ^ 64)
    (d w a o v : Nat) (s0 : MachineState) (h : Pre P d w a o v s0) (s : MachineState)
    (hL : Loaded s0 w a o v s) : P.Spec (c + 4) s 4 (c + 8) (Inv s0 w o v 0) := by
  obtain ⟨x5, -, x15, x12, hmem, hsaved⟩ := hL
  have ho := h.orange
  have hm := h.hmem
  have hoal := h.oal
  have hw := h.hw
  cases hcode
  mrunAt [Inv, x5, x15, x12, W_add]
  all_goals and_intros
  all_goals first
    | (simp only [BASE] at *; omega)
    | exact W_mod (by omega)
    | (intro k hk
       rw [Nat.lt_one_iff.mp hk]
       simp [out, MachineState.getMem_setMem_eq])
    | (intro x hx
       have hx0 := hx 0 Nat.zero_lt_one
       rw [Nat.mul_zero, Nat.add_zero] at hx0
       rw [MachineState.getMem_setMem_ne hx0]
       exact hmem x)
    | leaf_close

/-- The head's branch in iteration `i < w - 1`: instruction 8 to 10. -/
theorem segBranch (P : Prog) (c : Nat) (hcode : At P c) (_hc : BASE + 4 * (c + 15) < 2 ^ 64)
    (d w a o v : Nat) (s0 : MachineState) (h : Pre P d w a o v s0) (i : Nat) (s : MachineState)
    (hi : i < w - 1) (hI : Inv s0 w o v i s) : P.Spec (c + 8) s 1 (c + 10) (fun t => t = s) := by
  obtain ⟨x5, -, -, -, -, -⟩ := hI
  have ho := h.orange
  have hm := h.hmem
  cases hcode
  mrunAt [x5]
  all_goals first
    | rfl
    | (exfalso
       have h0 : W (w - 1 - i) = W 0 := by simpa using hbr
       have := Mem.ofNat_inj (by omega) (by omega) h0
       omega)

/-- After the store of word `i + 1`. -/
def Stored (s0 : MachineState) (w o v : Nat) (i : Nat) (s : MachineState) : Prop :=
  s.getReg .x5 = W (w - 1 - i) ∧ s.getReg .x12 = W (o + 8 * (i + 1)) ∧ s.getReg .x15 = W 0 ∧
    (∀ k < i + 2, s.getMem (W (o + 8 * k)) = W (out v k)) ∧
    (∀ x, (∀ k < i + 2, x ≠ W (o + 8 * k)) → s.getMem x = s0.getMem x) ∧ Saved s0 s

/-- The store: instruction 10. -/
theorem segStore (P : Prog) (c : Nat) (hcode : At P c) (_hc : BASE + 4 * (c + 15) < 2 ^ 64)
    (d w a o v : Nat) (s0 : MachineState) (h : Pre P d w a o v s0) (i : Nat) (s : MachineState)
    (hi : i < w - 1) (hI : Inv s0 w o v i s) : P.Spec (c + 10) s 1 (c + 11) (Stored s0 w o v i) := by
  obtain ⟨x5, x12, x15, hout, hframe, hsaved⟩ := hI
  have ho := h.orange
  have hm := h.hmem
  have hoal := h.oal
  cases hcode
  mrunAt [Stored, x5, x12, x15]
  all_goals and_intros
  all_goals first
    | (simp only [BASE] at *; omega)
    | (intro k hk
       rcases Nat.lt_succ_iff_lt_or_eq.mp hk with hk | rfl
       · rw [MachineState.getMem_setMem_ne (Mem.ofNat_ne (by simp only [BASE] at *; omega)
           (by simp only [BASE] at *; omega) (by omega))]
         exact hout k hk
       · simp [out, MachineState.getMem_setMem_eq])
    | (intro x hx
       rw [MachineState.getMem_setMem_ne (hx _ (Nat.lt_succ_self _))]
       exact hframe x fun k hk => hx k (by omega))
    | leaf_close

/-- The pointer updates: instructions 11 and 12. -/
theorem segNext (P : Prog) (c : Nat) (hcode : At P c) (_hc : BASE + 4 * (c + 15) < 2 ^ 64)
    (w o v : Nat) (s0 : MachineState) (i : Nat) (s : MachineState) (hi : i < w - 1)
    (hS : Stored s0 w o v i s) : P.Spec (c + 11) s 2 (c + 13) (Inv s0 w o v (i + 1)) := by
  obtain ⟨x5, x12, x15, hout, hframe, hsaved⟩ := hS
  cases hcode
  mrunAt [Inv, x5, x12, x15, W_add]
  all_goals and_intros
  all_goals first
    | exact W_mod (by omega)
    | (intro k hk; simpa using hout k hk)
    | (intro x hx; simpa using hframe x hx)
    | leaf_close

/-- The jump back: instruction 13. -/
theorem segJump (P : Prog) (c : Nat) (hcode : At P c) (_hc : BASE + 4 * (c + 15) < 2 ^ 64)
    (w o v : Nat) (s0 : MachineState) (i : Nat) (s : MachineState) (hI : Inv s0 w o v i s) :
    P.Spec (c + 13) s 1 (c + 8) (Inv s0 w o v i) := by
  cases hcode
  mrunAt
  exact hI

/-- One iteration: instructions 8, 10 to 13, back to 8. -/
theorem body (P : Prog) (c : Nat) (hcode : At P c) (hc : BASE + 4 * (c + 15) < 2 ^ 64)
    (d w a o v : Nat) (s0 : MachineState) (h : Pre P d w a o v s0) (i : Nat) (s : MachineState)
    (hi : i < w - 1) (hI : Inv s0 w o v i s) :
    P.Spec (c + 8) s 5 (c + 8) (Inv s0 w o v (i + 1)) :=
  Prog.Spec.seq (segBranch P c hcode hc d w a o v s0 h i s hi hI) fun s' h' => by
    subst h'
    exact Prog.Spec.seq (segStore P c hcode hc d w a o v s0 h i s' hi hI) fun s'' h'' =>
      Prog.Spec.seq (segNext P c hcode hc w o v s0 i s'' hi h'') fun s3 h3 =>
        segJump P c hcode hc w o v s0 (i + 1) s3 h3

/-- The exit: instructions 8, 9, and the jump to `ret`. -/
theorem exit (P : Prog) (c : Nat) (hcode : At P c) (_hc : BASE + 4 * (c + 15) < 2 ^ 64)
    (d w a o v : Nat) (s0 : MachineState) (h : Pre P d w a o v s0) (s : MachineState)
    (hI : Inv s0 w o v (w - 1) s) : P.Spec (c + 8) s 2 (c + 14) (Post s0 w o v) := by
  obtain ⟨x5, -, -, hout, hframe, hsaved⟩ := hI
  have hw := h.hw
  have e : w - 1 + 1 = w := by omega
  rw [e] at hout hframe
  cases hcode
  mrunAt [Post, x5]
  all_goals first
    | exact ⟨hout, hframe, hsaved⟩
    | (exfalso; simp at hbr)

/-- **The specification.** From entry, the leaf reaches `ret` within `5 + 5 w` steps. -/
theorem spec (P : Prog) (c : Nat) (hcode : At P c) (hc : BASE + 4 * (c + 15) < 2 ^ 64)
    (d w a o v : Nat) (s : MachineState) (h : Pre P d w a o v s) :
    P.Spec c s (5 + 5 * w) (c + 14) (Post s w o v) := by
  have hw := h.hw
  refine Prog.Spec.mono (Prog.Spec.seq (pro1 P c hcode hc d w a o v s h) fun s1 h1 =>
    Prog.Spec.seq (pro2 P c hcode hc d w a o v s h s1 h1) fun s2 h2 =>
      Prog.Spec.loop (N := w - 1) (fun i s' hi hI => body P c hcode hc d w a o v s h i s' hi hI)
        (fun s' hI => exit P c hcode hc d w a o v s h s' hI) 0 s2 (Nat.zero_le _) h2) ?_
    fun _ hp => hp
  omega

open MxxCore.Leaf

/-- The leaf's function on bytes: the `w`-word integer of the first word of its input. -/
def bytesF (args : List Nat) (ins : List Bytes) : List Bytes :=
  match args, ins with
  | [w], [x] => [Bytes.u64s ((List.range w).map fun k => out ((words x).getD 0 0) k)]
  | _, _ => []

/-- The leaf's domain: an argument `[w]` with `w ≥ 1`, and one input of one word. -/
def bytesDom (args : List Nat) (ins : List Bytes) : Prop :=
  ∃ w x, args = [w] ∧ ins = [x] ∧ x.length = 8 ∧ 1 ≤ w

theorem sound_bytes {P : Prog} {c : Nat} (hcode : At P c) (hc : BASE + 4 * (c + 15) < 2 ^ 64)
    (k : Call) (s : MachineState) (he : Entry P k s) (hdom : bytesDom k.args (k.ins.map (·.2)))
    (hlen : k.outs.map (·.2) = (bytesF k.args (k.ins.map (·.2))).map List.length) :
    P.Spec c s (5 + 5 * k.args.headD 0) (c + 14)
      (Exit k (bytesF k.args (k.ins.map (·.2))) s) := by
  obtain ⟨w, x, hargs, hins, hx, hw⟩ := hdom
  obtain ⟨d, args, ins, outs, sc⟩ := k
  simp only at hargs hins hlen ⊢
  subst hargs
  rcases ins with _ | ⟨⟨a, x'⟩, _ | _⟩ <;> simp at hins
  subst hins
  simp only [List.map, bytesF, Mem.length_u64s, List.length_map, List.length_range] at hlen
  rcases outs with _ | ⟨⟨o, lo⟩, _ | _⟩ <;> simp at hlen
  subst hlen
  have hw' : Call.words ⟨d, [w], [(a, x')], [(o, 8 * w)], sc⟩ = [w, a, o, sc.1] := rfl
  have hmem := he.hmem
  have ⟨aal, a1, a2⟩ := he.ial (a, x') (by simp)
  have ⟨oal, o1, o2⟩ := he.wal (o, 8 * w) (by simp [Call.written])
  simp only at aal a1 a2 oal o1 o2
  have dr := he.drange
  rw [hw'] at dr
  simp only [List.length_cons, List.length_nil] at dr
  have dj : ∀ i (h : i < 4), s.getMem (W (d + 8 * i)) = W ([w, a, o, sc.1][i]) :=
    fun i h => he.desc i (by rw [hw']; simpa using h)
  have hlx : (words x').length = 1 := by rw [length_words, hx]
  have hv : s.getMem (W a) = W ((words x').getD 0 0) := by
    have hx' : Holds s a x' := he.ins (a, x') (by simp)
    have := getMem_of_holds s a x' hx' 0 (by omega) aal (by omega)
    simp only [Nat.mul_zero, Nat.add_zero] at this
    rw [this, List.getD_eq_getElem?_getD, List.getElem?_eq_getElem (by omega), Option.getD_some]
  have d0 : s.getMem (W d) = W w := by
    have := dj 0 (by decide)
    simp only [Nat.mul_zero, Nat.add_zero] at this
    exact this
  have d1 : s.getMem (W (d + 8)) = W a := by
    have := dj 1 (by decide); simpa only [List.getElem_cons_succ, List.getElem_cons_zero] using this
  have d2 : s.getMem (W (d + 16)) = W o := by
    have := dj 2 (by decide); simpa only [List.getElem_cons_succ, List.getElem_cons_zero] using this
  have pre : Pre P d w a o ((words x').getD 0 0) s :=
    { hd := he.hd
      dw := d0
      da := d1
      dout := d2
      hv := hv
      hw := hw
      hmem := hmem
      dal := he.dal
      aal := aal
      oal := oal
      drange := ⟨dr.1, by omega⟩
      arange := ⟨a1, by omega⟩
      orange := ⟨o1, o2⟩ }
  refine Prog.Spec.mono (spec P c hcode hc d w a o _ s pre) (by simp) ?_
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
    have := hout' (o, 8 * w) (by simp [Call.written])
    exact getByte_of_frame s s' o w oal hframe (by omega) c' hc' this

/-- The specification in the shape of the core (`MxxCore.Leaf.Spec`), for every program that holds
the leaf's code at instruction `c`. -/
def byteSpec {P : Prog} {c : Nat} (hcode : At P c) (hc : BASE + 4 * (c + 15) < 2 ^ 64) :
    MxxCore.Leaf.Spec P c where
  ret := 14
  hret := hcode.i14
  steps args := 5 + 5 * args.headD 0
  dom := bytesDom
  f := bytesF
  sound k s he hdom hlen := sound_bytes hcode hc k s he hdom hlen

end MxxLatticeLeaves.BoolToInt
