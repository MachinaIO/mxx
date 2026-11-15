import MxxLatticeLeaves.Math.MulMod
import MxxLatticeLeaves.Math.Words
import MxxCore.Glue
import MxxCore.Leaf
import MxxCore.Sym

/-!
# The inline modular multiplication

The 15 instructions `ir_core::lower::leaves::emit_mulmod` writes compute `a5 = a3 a4 mod q` (`q`
in `t1`) and fall through. `spec` holds wherever a leaf places them (`At P c`): from `c`, with
`x13 = W x`, `x14 = W y`, `x6 = W q` (`x, y < q < 2^63`), the code reaches `c + 15` within 582
steps with `x15 = W (x y mod q)`, every other register but `x28`, `x29`, `x30` and all of memory
unchanged.
-/

namespace MxxLatticeLeaves.MulMod

open MathcalA RiscvZkvm.Rv64 MxxLatticeLeaves.Math.MulMod MxxLatticeLeaves.Math.Words

/-- The code at instruction `c`. -/
structure At (P : Prog) (c : Nat) : Prop where
  i0 : P.instrs[c]? = some (.MULHU .x28 .x13 .x14)
  i1 : P.instrs[c + 1]? = some (.MUL .x29 .x13 .x14)
  i2 : P.instrs[c + 2]? = some (.ADDI .x30 .x0 0x40#12)
  i3 : P.instrs[c + 3]? = some (.BNE .x30 .x0 0x8#13)
  i4 : P.instrs[c + 4]? = some (.JAL .x0 0x28#21)
  i5 : P.instrs[c + 5]? = some (.SRLI .x15 .x29 0x3f#6)
  i6 : P.instrs[c + 6]? = some (.SLLI .x29 .x29 0x1#6)
  i7 : P.instrs[c + 7]? = some (.ADD .x28 .x28 .x28)
  i8 : P.instrs[c + 8]? = some (.ADD .x28 .x28 .x15)
  i9 : P.instrs[c + 9]? = some (.BGEU .x28 .x6 0x8#13)
  i10 : P.instrs[c + 10]? = some (.JAL .x0 0x8#21)
  i11 : P.instrs[c + 11]? = some (.SUB .x28 .x28 .x6)
  i12 : P.instrs[c + 12]? = some (.ADDI .x30 .x30 0xfff#12)
  i13 : P.instrs[c + 13]? = some (.JAL .x0 0x1fffd8#21)
  i14 : P.instrs[c + 14]? = some (.ADDI .x15 .x28 0x0#12)

/-- Every register but `x15`, `x28`, `x29`, `x30` as in `s0`, and memory as in `s0`. -/
def Keep (s0 s : MachineState) : Prop :=
  (∀ r, r ≠ .x15 → r ≠ .x28 → r ≠ .x29 → r ≠ .x30 → s.getReg r = s0.getReg r) ∧
    ∀ a, s.getMem a = s0.getMem a

/-- The loop's head (instruction 3) after `i` steps. -/
def Head (s0 : MachineState) (q hi lo : Nat) (i : Nat) (s : MachineState) : Prop :=
  s.getReg .x30 = W (64 - i) ∧ s.getReg .x28 = W (rem q hi lo i) ∧
    s.getReg .x29 = W (lo * 2 ^ i % 2 ^ 64) ∧ Keep s0 s

set_option hygiene false in
macro "keep_close" : tactic => `(tactic| first
  | (intro r h1 h2 h3 h4; simp [h1, h2, h3, h4, hkeep.1 r h1 h2 h3 h4]; done)
  | (intro a; simp [hkeep.2 a]; done)
  | (intro a; simp [hmem a]; done))

/-- The products and the counter: instructions 0 to 3. -/
theorem start (P : Prog) (c : Nat) (hcode : At P c) (_hc : BASE + 4 * (c + 16) < 2 ^ 64)
    (s : MachineState) (x y q : Nat) (hx : s.getReg .x13 = W x) (hy : s.getReg .x14 = W y)
    (hxq : x < q) (hyq : y < q) (hq : q < 2 ^ 63) :
    P.Spec c s 3 (c + 3) (Head s q (x * y / 2 ^ 64) (x * y % 2 ^ 64) 0) := by
  have hmul : rv64_mulhu (W x) (W y) = W (x * y / 2 ^ 64) := mulhu_W (by omega) (by omega)
  have hlo : W x * W y = W (x * y % 2 ^ 64 * 2 ^ 0 % 2 ^ 64) := by
    rw [W_mul]; exact W_mod (by simp)
  have hhi : W (x * y / 2 ^ 64) = W (rem q (x * y / 2 ^ 64) (x * y % 2 ^ 64) 0) := by
    rw [rem_zero (Nat.mod_lt _ (by positivity)) (hi_lt hxq hyq (by omega))]
  have hkeep : Keep s s := ⟨fun _ _ _ _ _ => rfl, fun _ => rfl⟩
  cases hcode
  mrunAt [Head, Keep, hx, hy, hmul, hlo]
  all_goals and_intros
  all_goals first
    | exact hhi
    | keep_close
    | rfl

/-- After the doubling of step `i`: `x28 = 2 r + b`, `x29` doubled, `x15 = b`. -/
def Mid (s0 : MachineState) (q hi lo : Nat) (i : Nat) (s : MachineState) : Prop :=
  s.getReg .x30 = W (64 - i) ∧
    s.getReg .x28 = W (2 * rem q hi lo i + lo * 2 ^ i % 2 ^ 64 / 2 ^ 63) ∧
    s.getReg .x29 = W (lo * 2 ^ (i + 1) % 2 ^ 64) ∧ Keep s0 s

/-- The head's branch in step `i < 64`: instruction 3 to 5. -/
theorem branch (P : Prog) (c : Nat) (hcode : At P c) (_hc : BASE + 4 * (c + 16) < 2 ^ 64)
    (s0 : MachineState) (q hi lo i : Nat) (s : MachineState) (hi64 : i < 64)
    (hH : Head s0 q hi lo i s) : P.Spec (c + 3) s 1 (c + 5) (fun t => t = s) := by
  obtain ⟨x30, -⟩ := hH
  cases hcode
  mrunAt [x30]
  all_goals first
    | rfl
    | (exfalso
       have h0 : W (64 - i) = W 0 := by simpa using hbr
       have := Mem.ofNat_inj (by omega) (by omega) h0
       omega)

/-- The doubling: instructions 5 to 9. -/
theorem double (P : Prog) (c : Nat) (hcode : At P c) (_hc : BASE + 4 * (c + 16) < 2 ^ 64)
    (s0 : MachineState) (q hi lo i : Nat) (s : MachineState) (hq : q < 2 ^ 63) (hq0 : 0 < q)
    (hH : Head s0 q hi lo i s) : P.Spec (c + 5) s 4 (c + 9) (Mid s0 q hi lo i) := by
  obtain ⟨x30, x28, x29, hkeep⟩ := hH
  have hv : lo * 2 ^ i % 2 ^ 64 < 2 ^ 64 := Nat.mod_lt _ (by positivity)
  have hsrl := srl_W (k := 63) hv
  have hshl : BitVec.ofNat 64 (lo * 2 ^ i % 2 ^ 64) <<< 1 = W (lo * 2 ^ (i + 1) % 2 ^ 64) := by
    rw [shl_W]; apply W_mod
    rw [Nat.mod_mod, Nat.mod_mul_mod, Nat.mul_assoc, ← pow_add]
  have hadd : W (rem q hi lo i) + W (rem q hi lo i) + W (lo * 2 ^ i % 2 ^ 64 / 2 ^ 63) =
      W (2 * rem q hi lo i + lo * 2 ^ i % 2 ^ 64 / 2 ^ 63) := by
    rw [W_add, W_add, two_mul]
  unfold Mid
  generalize lo * 2 ^ i % 2 ^ 64 / 2 ^ 63 = B at hsrl hadd ⊢
  generalize lo * 2 ^ (i + 1) % 2 ^ 64 = V' at hshl ⊢
  generalize lo * 2 ^ i % 2 ^ 64 = V at x29 hsrl hshl hv
  generalize rem q hi lo i = R at x28 hadd ⊢
  cases hcode
  mrunAt [x30, x28, x29, hsrl, hshl, hadd]
  all_goals and_intros
  all_goals first
    | keep_close
    | rfl

/-- The reduction: instructions 9 to 12. -/
theorem reduce (P : Prog) (c : Nat) (hcode : At P c) (_hc : BASE + 4 * (c + 16) < 2 ^ 64)
    (s0 : MachineState) (q hi lo i : Nat) (s : MachineState) (hq : q < 2 ^ 63) (hq0 : 0 < q)
    (hi64 : i < 64) (h6 : s0.getReg .x6 = W q) (hM : Mid s0 q hi lo i s) :
    P.Spec (c + 9) s 2 (c + 12) (fun t => t.getReg .x30 = W (64 - i) ∧
      t.getReg .x28 = W (rem q hi lo (i + 1)) ∧ t.getReg .x29 = W (lo * 2 ^ (i + 1) % 2 ^ 64) ∧
      Keep s0 t) := by
  obtain ⟨x30, x28, x29, hkeep⟩ := hM
  have x6 : s.getReg .x6 = W q := by rw [hkeep.1 .x6 (by decide) (by decide) (by decide) (by decide), h6]
  have hr := rem_lt (hi := hi) (lo := lo) (i := i) hq0
  have hb : lo * 2 ^ i % 2 ^ 64 / 2 ^ 63 ≤ 1 := by
    have : lo * 2 ^ i % 2 ^ 64 < 2 ^ 64 := Nat.mod_lt _ (by positivity)
    omega
  have hstep := rem_step (hi := hi) (lo := lo) hq0 hi64
  generalize lo * 2 ^ i % 2 ^ 64 / 2 ^ 63 = B at x28 hb hstep
  generalize rem q hi lo i = R at x28 hr hstep
  generalize rem q hi lo (i + 1) = R' at hstep ⊢
  cases hcode
  mrunAt [x30, x28, x29, x6]
  all_goals and_intros
  all_goals first
    | (have hge : 2 * R + B ≥ q := by simp [BitVec.ult] at hbr; omega
       rw [sub_W hge, ← hstep, Nat.mod_eq_sub_mod hge, Nat.mod_eq_of_lt (by omega)]; done)
    | (have hlt : 2 * R + B < q := by simp [BitVec.ult] at hbr; omega
       rw [← hstep, Nat.mod_eq_of_lt hlt]; done)
    | keep_close
    | rfl

/-- The counter and the jump back: instructions 12 and 13, back to 3. -/
theorem next (P : Prog) (c : Nat) (hcode : At P c) (_hc : BASE + 4 * (c + 16) < 2 ^ 64)
    (s0 : MachineState) (q hi lo i : Nat) (s : MachineState) (hi64 : i < 64)
    (hR : s.getReg .x30 = W (64 - i) ∧ s.getReg .x28 = W (rem q hi lo (i + 1)) ∧
      s.getReg .x29 = W (lo * 2 ^ (i + 1) % 2 ^ 64) ∧ Keep s0 s) :
    P.Spec (c + 12) s 1 (c + 13) (Head s0 q hi lo (i + 1)) := by
  obtain ⟨x30, x28, x29, hkeep⟩ := hR
  unfold Head
  generalize lo * 2 ^ (i + 1) % 2 ^ 64 = V at x29 ⊢
  generalize rem q hi lo (i + 1) = R at x28 ⊢
  cases hcode
  mrunAt [x30, x28, x29, W_add]
  all_goals and_intros
  all_goals first
    | exact W_mod (by omega)
    | keep_close
    | bv_omega

theorem jump (P : Prog) (c : Nat) (hcode : At P c) (_hc : BASE + 4 * (c + 16) < 2 ^ 64)
    (s0 : MachineState) (q hi lo i : Nat) (s : MachineState) (hH : Head s0 q hi lo i s) :
    P.Spec (c + 13) s 1 (c + 3) (Head s0 q hi lo i) := by
  cases hcode
  mrunAt
  exact hH

/-- The exit: instructions 3, 4, 14. -/
theorem exit (P : Prog) (c : Nat) (hcode : At P c) (_hc : BASE + 4 * (c + 16) < 2 ^ 64)
    (s0 : MachineState) (q x y : Nat) (s : MachineState)
    (hH : Head s0 q (x * y / 2 ^ 64) (x * y % 2 ^ 64) 64 s) :
    P.Spec (c + 3) s 3 (c + 15) (fun t => t.getReg .x15 = W (x * y % q) ∧
      (∀ r, r ≠ .x15 → r ≠ .x28 → r ≠ .x29 → r ≠ .x30 → t.getReg r = s0.getReg r) ∧
      ∀ a, t.getMem a = s0.getMem a) := by
  obtain ⟨x30, x28, -, hkeep⟩ := hH
  have x30' : s.getReg .x30 = W 0 := by simpa using x30
  rw [rem_last] at x28
  cases hcode
  mrunAt [x30', x28]
  all_goals and_intros
  all_goals first
    | (exfalso; revert hbr; simp; done)
    | (simp; done)
    | keep_close

/-- **The specification** of the inline modular multiplication. -/
theorem spec (P : Prog) (c : Nat) (hcode : At P c) (hc : BASE + 4 * (c + 16) < 2 ^ 64)
    (s : MachineState) (x y q : Nat) (hx : s.getReg .x13 = W x) (hy : s.getReg .x14 = W y)
    (h6 : s.getReg .x6 = W q) (hxq : x < q) (hyq : y < q) (hq : q < 2 ^ 63) :
    P.Spec c s 582 (c + 15) (fun t => t.getReg .x15 = W (x * y % q) ∧
      (∀ r, r ≠ .x15 → r ≠ .x28 → r ≠ .x29 → r ≠ .x30 → t.getReg r = s.getReg r) ∧
      ∀ a, t.getMem a = s.getMem a) := by
  have hq0 : 0 < q := by omega
  refine Prog.Spec.mono (Prog.Spec.seq (start P c hcode hc s x y q hx hy hxq hyq hq) fun s1 h1 =>
    Prog.Spec.loop (N := 64)
      (fun i s' hi hH => Prog.Spec.seq (branch P c hcode hc s _ _ _ i s' hi hH) fun s2 h2 => by
        subst s2
        exact Prog.Spec.seq (double P c hcode hc s _ _ _ i s' hq hq0 hH) fun s3 h3 =>
          Prog.Spec.seq (reduce P c hcode hc s _ _ _ i s3 hq hq0 hi h6 h3) fun s4 h4 =>
            Prog.Spec.seq (next P c hcode hc s _ _ _ i s4 hi h4) fun s5 h5 =>
              jump P c hcode hc s _ _ _ (i + 1) s5 h5)
      (fun s' hH => exit P c hcode hc s q x y s' hH) 0 s1 (Nat.zero_le _) h1) (by omega)
    fun _ hp => hp

end MxxLatticeLeaves.MulMod
