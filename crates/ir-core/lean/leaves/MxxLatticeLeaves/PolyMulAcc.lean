import MxxLatticeLeaves.Code.PolyMulAcc
import MxxLatticeLeaves.MulMod
import MxxCore.Glue
import MxxCore.Leaf
import MxxCore.Sym

/-!
# The specification of the leaf `lattice.poly_mul_acc`

For any program `P` that holds the leaf's code at instruction `c`, from that entry, with `a0`
pointing at the descriptor `[N, q, acc, a, b, o, scratch]`, the leaf reaches its `ret`
(instruction `c + 57`) within `9 + N (8 + N (25 + 582))` steps, having written `F k` to the
doubleword at `o + 8 k` for every `k < N`: coefficient `k` of `acc + a b` in `Z_q[X]/(X^N + 1)`,
as the partial sums `part k i` it accumulates (`part k 0 = acc[k]`; term `i` adds
`a[i] b[k - i] mod q` for `i ≤ k` and subtracts `a[i] b[N + k - i] mod q` for `i > k`). The
products are the inline modular multiplication of `MxxLatticeLeaves.MulMod` at instruction
`c + 24`.
-/

namespace MxxLatticeLeaves.PolyMulAcc

open MathcalA RiscvZkvm.Rv64 MxxLatticeLeaves.Code.PolyMulAcc MxxLatticeLeaves.Math.Words

/-- The registers a leaf keeps: `ra`, `sp`, and `s0`–`s11`. -/
def Saved (s0 s : MachineState) : Prop :=
  ∀ r ∈ [Reg.x1, .x2, .x8, .x9, .x18, .x19, .x20, .x21, .x22, .x23, .x24, .x25, .x26, .x27],
    s.getReg r = s0.getReg r

/-- The index of `b` that term `i` of coefficient `k` reads. -/
def idx (N k i : Nat) : Nat := if k < i then N + k - i else k - i

/-- The partial sums of coefficient `k`. -/
def part (N q : Nat) (C A B : Nat → Nat) (k : Nat) : Nat → Nat
  | 0 => C k
  | i + 1 =>
    if k < i then (part N q C A B k i + q - A i * B (idx N k i) % q) % q
    else (part N q C A B k i + A i * B (idx N k i) % q) % q

/-- Coefficient `k` of the result. -/
def F (N q : Nat) (C A B : Nat → Nat) (k : Nat) : Nat := part N q C A B k N

theorem part_lt {N q : Nat} {C A B : Nat → Nat} {k : Nat} (hq : 0 < q) (hC : C k < q) :
    ∀ i, part N q C A B k i < q
  | 0 => hC
  | i + 1 => by unfold part; split <;> exact Nat.mod_lt _ hq

/-- The fragment's code inside the leaf's. -/
theorem mulmodAt {P : Prog} {c : Nat} (hcode : At P c) : MulMod.At P (c + 24) :=
  ⟨hcode.i24, hcode.i25, hcode.i26, hcode.i27, hcode.i28, hcode.i29, hcode.i30, hcode.i31,
    hcode.i32, hcode.i33, hcode.i34, hcode.i35, hcode.i36, hcode.i37, hcode.i38⟩

/-- What the leaf assumes on entry. -/
structure Pre (P : Prog) (d N q ac a b o : Nat) (C A B : Nat → Nat) (s : MachineState) : Prop where
  hd : s.getReg .x10 = W d
  dN : s.getMem (W d) = W N
  dq : s.getMem (W (d + 8)) = W q
  dc : s.getMem (W (d + 16)) = W ac
  da : s.getMem (W (d + 24)) = W a
  db : s.getMem (W (d + 32)) = W b
  dout : s.getMem (W (d + 40)) = W o
  hC : ∀ k < N, s.getMem (W (ac + 8 * k)) = W (C k)
  hA : ∀ k < N, s.getMem (W (a + 8 * k)) = W (A k)
  hB : ∀ k < N, s.getMem (W (b + 8 * k)) = W (B k)
  hCq : ∀ k < N, C k < q
  hAq : ∀ k < N, A k < q
  hBq : ∀ k < N, B k < q
  hq : q < 2 ^ 63
  hmem : BASE + P.mem < 2 ^ 64
  dal : d % 8 = 0
  cal : ac % 8 = 0
  aal : a % 8 = 0
  bal : b % 8 = 0
  oal : o % 8 = 0
  drange : BASE ≤ d ∧ d + 48 ≤ BASE + P.mem
  crange : BASE ≤ ac ∧ ac + 8 * N ≤ BASE + P.mem
  arange : BASE ≤ a ∧ a + 8 * N ≤ BASE + P.mem
  brange : BASE ≤ b ∧ b + 8 * N ≤ BASE + P.mem
  orange : BASE + P.bin.length ≤ o ∧ o + 8 * N ≤ BASE + P.mem
  disjC : o + 8 * N ≤ ac ∨ ac + 8 * N ≤ o
  disjA : o + 8 * N ≤ a ∨ a + 8 * N ≤ o
  disjB : o + 8 * N ≤ b ∨ b + 8 * N ≤ o

/-- What holds at `ret`. -/
def Post (s0 : MachineState) (N q o : Nat) (C A B : Nat → Nat) (s : MachineState) : Prop :=
  (∀ k < N, s.getMem (W (o + 8 * k)) = W (F N q C A B k)) ∧
    (∀ x, (∀ k < N, x ≠ W (o + 8 * k)) → s.getMem x = s0.getMem x) ∧ Saved s0 s

/-- The registers that hold throughout. -/
def Fixed (N q a b : Nat) (s : MachineState) : Prop :=
  s.getReg .x5 = W N ∧ s.getReg .x6 = W q ∧ s.getReg .x16 = W a ∧ s.getReg .x17 = W b

/-- The written prefix: `K` coefficients, and the frame. -/
def Written (s0 : MachineState) (N q o : Nat) (C A B : Nat → Nat) (K : Nat) (s : MachineState) :
    Prop :=
  (∀ k < K, s.getMem (W (o + 8 * k)) = W (F N q C A B k)) ∧
    (∀ x, (∀ k < K, x ≠ W (o + 8 * k)) → s.getMem x = s0.getMem x)

/-- The coefficient loop's head (instruction 7) before coefficient `k`. -/
def OInv (s0 : MachineState) (N q ac a b o : Nat) (C A B : Nat → Nat) (k : Nat) (s : MachineState) :
    Prop :=
  Fixed N q a b s ∧ s.getReg .x7 = W k ∧ s.getReg .x11 = W (ac + 8 * k) ∧
    s.getReg .x12 = W (o + 8 * k) ∧ Written s0 N q o C A B k s ∧ Saved s0 s

/-- The term loop's head (instruction 11) before term `i` of coefficient `k`. -/
def IInv (s0 : MachineState) (N q ac a b o : Nat) (C A B : Nat → Nat) (k i : Nat)
    (s : MachineState) : Prop :=
  Fixed N q a b s ∧ s.getReg .x7 = W k ∧ s.getReg .x11 = W (ac + 8 * k) ∧
    s.getReg .x12 = W (o + 8 * k) ∧ s.getReg .x3 = W i ∧
    s.getReg .x10 = W (part N q C A B k i) ∧ Written s0 N q o C A B k s ∧ Saved s0 s

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

end MxxLatticeLeaves.PolyMulAcc
