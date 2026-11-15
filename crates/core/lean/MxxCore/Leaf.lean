import MxxCore.Algorithm
import MxxCore.Copy
/-!
# The specification shape of leaves

A leaf is called with `a0` at a descriptor of words: its arguments, then the address of each
input, of each output, and of its scratch area (`mxx_core::leaf`). Its specification, in the
shape the core fixes for every DSL (`ARCHITECTURE-SPEC.md`, section 4.3), is about bytes:

- **Entry.** The descriptor holds the arguments and addresses; each input holds its bytes; every
  address is 8-byte aligned; each output and the scratch lie past the image and inside memory,
  apart from each other, from the inputs, and from the descriptor; the arguments and inputs lie in
  the leaf's domain.
- **Exit.** At its `ret`, within its step bound, each output holds `f args inputs`, every byte
  outside the outputs and the scratch is unchanged, and the registers the ABI keeps are kept.

A generated composition proof chains such specifications through a program's glue without
knowing what each leaf computes: it tracks which bytes each range holds (\`Holds\`) through
input queries (\`holds_write\`), leaf calls (\`Spec.call\`, \`holds_exit\`), copies (\`holds_copy\`,
\`holds_copy_far\`), and the loaded image (\`holds_init\`), and reads the \`HALT\` payload as adjacent
held ranges (\`holds_append\`).
-/

namespace MxxCore.Leaf

open MathcalA RiscvZkvm.Rv64

/-- The bytes at `a` in `s` are `bs`. -/
def Holds (s : MachineState) (a : Nat) (bs : Bytes) : Prop :=
  Prog.Runs.read s a bs.length = bs

/-- The registers a leaf keeps: `ra`, `sp`, and `s0`–`s11`. -/
def Saved (s0 s : MachineState) : Prop :=
  ∀ r ∈ [Reg.x1, .x2, .x8, .x9, .x18, .x19, .x20, .x21, .x22, .x23, .x24, .x25, .x26, .x27],
    s.getReg r = s0.getReg r

/-- Two byte ranges `[a, a + m)` and `[b, b + n)` do not overlap. -/
def Apart (a m b n : Nat) : Prop := a + m ≤ b ∨ b + n ≤ a

/-- One call: the descriptor's address, the arguments, the inputs (address and bytes), the
outputs (address and length), and the scratch (address and length). -/
structure Call where
  d : Nat
  args : List Nat
  ins : List (Nat × Bytes)
  outs : List (Nat × Nat)
  scratch : Nat × Nat

/-- The descriptor's words. -/
def Call.words (k : Call) : List Nat :=
  k.args ++ k.ins.map (·.1) ++ k.outs.map (·.1) ++ [k.scratch.1]

/-- The written ranges: the outputs and the scratch. -/
def Call.written (k : Call) : List (Nat × Nat) := k.outs ++ [k.scratch]

/-- What a call assumes on entry, in program `P`. -/
structure Entry (P : Prog) (k : Call) (s : MachineState) : Prop where
  hd : s.getReg .x10 = W k.d
  desc : ∀ i (h : i < k.words.length), s.getMem (W (k.d + 8 * i)) = W (k.words[i])
  words : ∀ w ∈ k.words, w < 2 ^ 64
  ins : ∀ x ∈ k.ins, Holds s x.1 x.2
  hmem : BASE + P.mem < 2 ^ 64
  dal : k.d % 8 = 0
  drange : BASE ≤ k.d ∧ k.d + 8 * k.words.length ≤ BASE + P.mem
  ial : ∀ x ∈ k.ins, x.1 % 8 = 0 ∧ BASE ≤ x.1 ∧ x.1 + x.2.length ≤ BASE + P.mem
  wal : ∀ x ∈ k.written, x.1 % 8 = 0 ∧ BASE + P.bin.length ≤ x.1 ∧ x.1 + x.2 ≤ BASE + P.mem
  apartIn : ∀ x ∈ k.written, ∀ y ∈ k.ins, Apart x.1 x.2 y.1 y.2.length
  apartDesc : ∀ x ∈ k.written, Apart x.1 x.2 k.d (8 * k.words.length)
  apartOut : k.written.Pairwise fun x y => Apart x.1 x.2 y.1 y.2

/-- What holds at the leaf's `ret`: the outputs hold `outs`, every other byte is unchanged, and
the kept registers are kept. -/
def Exit (k : Call) (outs : List Bytes) (s0 s : MachineState) : Prop :=
  (∀ i (h : i < k.outs.length) (h' : i < outs.length), Holds s k.outs[i].1 outs[i]) ∧
    (∀ c, c < 2 ^ 64 → (∀ x ∈ k.written, c < x.1 ∨ x.1 + x.2 ≤ c) →
      s.getByte (BitVec.ofNat 64 c) = s0.getByte (BitVec.ofNat 64 c)) ∧ Saved s0 s

/-- The specification of a leaf whose code sits at instruction `c` of `P`: from its entry, with
arguments and inputs in its domain, it reaches its `ret` at `c + ret` within `steps args` steps
with each output holding `f args inputs`. -/
structure Spec (P : Prog) (c : Nat) where
  ret : Nat
  hret : P.instrs[c + ret]? = some (.JALR .x0 .x1 0x0#12)
  steps : List Nat → Nat
  dom : List Nat → List Bytes → Prop
  f : List Nat → List Bytes → List Bytes
  sound : ∀ (k : Call) (s : MachineState), Entry P k s → dom k.args (k.ins.map (·.2)) →
    k.outs.map (·.2) = (f k.args (k.ins.map (·.2))).map List.length →
    P.Spec c s (steps k.args) (c + ret) (Exit k (f k.args (k.ins.map (·.2))) s)

/-! ### Words -/

/-- The 64-bit little-endian words of `bs`, its length a multiple of 8. -/
def words (bs : Bytes) : List Nat :=
  (List.range (bs.length / 8)).map fun k => Bytes.leToNat ((bs.drop (8 * k)).take 8)

theorem length_words (bs : Bytes) : (words bs).length = bs.length / 8 := by simp [words]

theorem leToNat_lt (bs : Bytes) : Bytes.leToNat bs < 256 ^ bs.length := by
  induction bs with
  | nil => simp [Bytes.leToNat]
  | cons b bs ih =>
    simp only [Bytes.leToNat, List.length_cons, Nat.pow_succ]
    have := b.isLt
    omega

/-- Byte `j` of an 8-byte string is the `j`-th base-256 digit of its value. -/
theorem getElem_of_leToNat (bs : Bytes) (j : Nat) (hj : j < bs.length) :
    bs[j] = BitVec.ofNat 8 (Bytes.leToNat bs / 256 ^ j) := by
  induction bs generalizing j with
  | nil => simp at hj
  | cons b bs ih =>
    cases j with
    | zero =>
      apply BitVec.eq_of_toNat_eq
      simp only [List.getElem_cons_zero, Bytes.leToNat, pow_zero, Nat.div_one, BitVec.toNat_ofNat]
      have := b.isLt
      omega
    | succ j =>
      simp only [List.getElem_cons_succ, Bytes.leToNat]
      rw [ih j (by simpa using hj), Nat.pow_succ', ← Nat.div_div_eq_div_mul]
      congr 2
      have := b.isLt
      omega

/-- An aligned doubleword is the word its 8 held bytes encode. -/
theorem getMem_of_holds (s : MachineState) (a : Nat) (bs : Bytes) (h : Holds s a bs)
    (k : Nat) (hk : 8 * k + 8 ≤ bs.length) (ha : a % 8 = 0) (hend : a + bs.length < 2 ^ 64) :
    s.getMem (W (a + 8 * k)) = W ((words bs)[k]'(by rw [length_words]; omega)) := by
  apply Mem.getMem_of_bytes _ (by omega) (by omega)
  intro j hj
  have hb := congrArg (fun l => l[8 * k + j]?) h
  rw [List.getElem?_eq_getElem (by simp [Prog.Runs.read]; omega),
    List.getElem?_eq_getElem (by omega)] at hb
  simp only [Option.some.injEq, Prog.Runs.read, List.getElem_map, List.getElem_range] at hb
  rw [show a + 8 * k + j = a + (8 * k + j) by omega, hb]
  simp only [words, List.getElem_map, List.getElem_range]
  have := getElem_of_leToNat ((bs.drop (8 * k)).take 8) j (by simp; omega)
  simp only [List.getElem_take, List.getElem_drop] at this
  exact this


/-- Consecutive aligned doublewords hold their `u64` encodings. -/
theorem holds_u64s (s : MachineState) (a : Nat) (vs : List Nat) (ha : a % 8 = 0)
    (hend : a + 8 * vs.length < 2 ^ 64)
    (h : ∀ k (hk : k < vs.length), s.getMem (W (a + 8 * k)) = W vs[k]) :
    Holds s a (Bytes.u64s vs) := by
  unfold Holds
  rw [Mem.length_u64s]
  apply List.ext_getElem (by simp [Prog.Runs.read, Mem.length_u64s])
  intro i h1 h2
  simp only [Prog.Runs.read, List.length_map, List.length_range] at h1
  simp only [Prog.Runs.read, List.getElem_map, List.getElem_range]
  have hk : i / 8 < vs.length := by omega
  rw [show (Bytes.u64s vs)[i] = (Bytes.u64s vs)[8 * (i / 8) + i % 8]'(by
      rw [Mem.length_u64s]; omega) from by congr 1; omega,
    Mem.getElem_u64s vs (i / 8) (i % 8) hk (by omega)]
  unfold MachineState.getByte
  rw [Mem.alignToDword_ofNat (by omega), Mem.byteOffset_ofNat (by omega),
    show (a + i) / 8 * 8 = a + 8 * (i / 8) by omega, show (a + i) % 8 = i % 8 by omega, h _ hk]
  exact Mem.extractByte_ofNat _ _ (by omega)

/-- A byte outside the aligned doublewords `[o, o + 8 n)` is kept when every other doubleword is. -/
theorem getByte_of_frame (s0 s : MachineState) (o n : Nat) (ho : o % 8 = 0)
    (hframe : ∀ x, (∀ k < n, x ≠ W (o + 8 * k)) → s.getMem x = s0.getMem x)
    (hend : o + 8 * n < 2 ^ 64) (c : Nat) (hc : c < 2 ^ 64) (hout : c < o ∨ o + 8 * n ≤ c) :
    s.getByte (BitVec.ofNat 64 c) = s0.getByte (BitVec.ofNat 64 c) := by
  unfold MachineState.getByte
  rw [Mem.alignToDword_ofNat hc, hframe]
  intro k hk e
  have := (W_eq _ _ (by omega) (by omega)).1 e
  omega


/-- A call of the leaf `L` from instruction `j` (`auipc ra, 0`; `jalr ra, off(ra)`), back at
`j + 2` through its `ret`. -/
theorem Spec.call {P : Prog} {c : Nat} (L : Spec P c) (j : Nat) (off : BitVec 12)
    (k : Call) (s : MachineState)
    (hauipc : P.instrs[j]? = some (.AUIPC .x1 0x0#20))
    (hjalr : P.instrs[j + 1]? = some (.JALR .x1 .x1 off))
    (ht : (pcOf j + ((0x0#20 : BitVec 20).zeroExtend 32 <<< 12 : BitVec 32).signExtend 64 +
      signExtend12 off) &&& ~~~1#64 = pcOf c)
    (hback : (pcOf (j + 1 + 1) + signExtend12 0x0#12) &&& ~~~1#64 = pcOf (j + 2))
    (h : Entry P k s) (hdom : L.dom k.args (k.ins.map (·.2)))
    (hlen : k.outs.map (·.2) = (L.f k.args (k.ins.map (·.2))).map List.length) :
    P.Spec j s (2 + (L.steps k.args + 1)) (j + 2)
      (Exit k (L.f k.args (k.ins.map (·.2))) (s.setReg .x1 (pcOf (j + 1 + 1)))) := by
  rw [show 2 + (L.steps k.args + 1) = (L.steps k.args + 1) + 1 + 1 by omega]
  refine MxxCore.Glue.auipc hauipc ?_
  refine MxxCore.Glue.jalr (t := c) hjalr (by simpa using ht) ?_
  rw [Sym.setReg_setReg_same]
  have hentry : Entry P k (s.setReg .x1 (pcOf (j + 1 + 1))) := by
    refine { h with hd := by simpa using h.hd, desc := fun i hi => by simpa using h.desc i hi,
                    ins := fun x hx => ?_ }
    have := h.ins x hx
    unfold Holds Prog.Runs.read at this ⊢
    simpa using this
  refine Prog.Spec.seq (L.sound k _ hentry hdom hlen) fun s' hs' => ?_
  have hra : s'.getReg .x1 = pcOf (j + 1 + 1) := by
    rw [hs'.2.2 .x1 (by simp)]
    simp
  refine MxxCore.Glue.jalr (t := j + 2) L.hret (by rw [hra]; exact hback) ?_
  exact Prog.Spec.done (by simpa using hs')

/-! ### Holding bytes through glue -/

theorem holds_write (s : MachineState) (a : Nat) (bs : Bytes) (h : a + bs.length ≤ 2 ^ 64) :
    Holds (Prog.Runs.write s a bs) a bs := by
  unfold Holds
  apply List.ext_getElem (by simp [Prog.Runs.read])
  intro i h1 h2
  simp only [Prog.Runs.read, List.getElem_map, List.getElem_range]
  exact Prog.Runs.write_getByte_in _ _ _ _ h2 h

/-- Holds only reads bytes. -/
theorem holds_congr {s t : MachineState} {a : Nat} {bs : Bytes}
    (h : ∀ i < bs.length, t.getByte (BitVec.ofNat 64 (a + i)) = s.getByte (BitVec.ofNat 64 (a + i)))
    (hs : Holds s a bs) : Holds t a bs := by
  unfold Holds at *
  have : Prog.Runs.read t a bs.length = Prog.Runs.read s a bs.length := by
    unfold Prog.Runs.read
    refine List.map_congr_left fun i hi => ?_
    simp only [List.mem_range] at hi
    exact h i hi
  rw [this, hs]

theorem holds_setReg {s : MachineState} {a : Nat} {bs : Bytes} (r : Reg) (v : Word)
    (h : Holds s a bs) : Holds (s.setReg r v) a bs :=
  holds_congr (fun i _ => by simp) h

theorem holds_write_far {s : MachineState} {r : Nat} {bs : Bytes} (a : Nat) (xs : Bytes)
    (h : Holds s r bs) (hend : a + xs.length ≤ 2 ^ 64) (hr : r + bs.length < 2 ^ 64)
    (hout : Apart r bs.length a xs.length) : Holds (Prog.Runs.write s a xs) r bs :=
  holds_congr (fun i hi => Prog.Runs.write_getByte_out _ _ _ _ (by omega) hend
    (by unfold Apart at hout; omega)) h

/-- After a leaf, inputs and every range apart from what it writes still hold. -/
theorem holds_exit {k : Call} {outs : List Bytes} {s0 s : MachineState} {r : Nat} {bs : Bytes}
    (he : Exit k outs s0 s) (h : Holds s0 r bs) (hr : r + bs.length < 2 ^ 64)
    (hout : ∀ x ∈ k.written, Apart r bs.length x.1 x.2) : Holds s r bs :=
  holds_congr (fun i hi => he.2.1 _ (by omega) fun x hx => by
    have := hout x hx; unfold Apart at this; omega) h

theorem read_split (s : MachineState) (a m n : Nat) :
    Prog.Runs.read s a (m + n) = Prog.Runs.read s a m ++ Prog.Runs.read s (a + m) n := by
  simp only [Prog.Runs.read, List.range_add, List.map_append, List.map_map]
  congr 1
  refine List.map_congr_left fun i _ => ?_
  simp only [Function.comp, Nat.add_assoc]

/-- Adjacent holds make one. -/
theorem holds_append {s : MachineState} {a : Nat} {x y : Bytes} (hx : Holds s a x)
    (hy : Holds s (a + x.length) y) : Holds s a (x ++ y) := by
  unfold Holds at *
  rw [List.length_append, read_split, hx, hy]


/-- A byte through the doubleword holding it. -/
theorem getByte_word (s : MachineState) (a i : Nat) (ha : a % 8 = 0) (h : a + i < 2 ^ 64) :
    s.getByte (BitVec.ofNat 64 (a + i)) =
      extractByte (s.getMem (W (a + 8 * (i / 8)))) (i % 8) := by
  unfold MachineState.getByte
  rw [Mem.alignToDword_ofNat h, Mem.byteOffset_ofNat h,
    show (a + i) / 8 * 8 = a + 8 * (i / 8) by omega, show (a + i) % 8 = i % 8 by omega]

/-- An aligned copy moves what its source holds to its destination. -/
theorem holds_copy {s0 s : MachineState} {dst src N : Nat} {bs : Bytes}
    (hpost : MxxCore.Copy.Post s0 dst src N s) (h : Holds s0 src bs) (hlen : bs.length = 8 * N)
    (hd : dst % 8 = 0) (hs : src % 8 = 0) (hdst : dst + 8 * N < 2 ^ 64)
    (hsrc : src + 8 * N < 2 ^ 64) : Holds s dst bs := by
  unfold Holds at *
  have : Prog.Runs.read s dst bs.length = Prog.Runs.read s0 src bs.length := by
    unfold Prog.Runs.read
    refine List.map_congr_left fun i hi => ?_
    simp only [List.mem_range] at hi
    rw [getByte_word s dst i hd (by omega), getByte_word s0 src i hs (by omega),
      hpost.1 (i / 8) (by omega)]
  rw [this, h]

/-- An aligned copy keeps what lies apart from its destination. -/
theorem holds_copy_far {s0 s : MachineState} {dst src N r : Nat} {bs : Bytes}
    (hpost : MxxCore.Copy.Post s0 dst src N s) (h : Holds s0 r bs) (hd : dst % 8 = 0)
    (hdst : dst + 8 * N < 2 ^ 64) (hr : r + bs.length < 2 ^ 64)
    (hout : Apart r bs.length dst (8 * N)) : Holds s r bs :=
  holds_congr (fun i hi => by
    unfold MachineState.getByte
    rw [Mem.alignToDword_ofNat (by omega), hpost.2.1]
    intro k hk e
    have := (W_eq _ _ (by omega) (by omega)).1 e
    unfold Apart at hout
    omega) h

/-- The loaded image holds its bytes. -/
theorem holds_init (P : Prog) (c k : Nat) (h : c + k ≤ P.bin.length)
    (hend : BASE + P.bin.length ≤ 2 ^ 64) :
    Holds P.init (BASE + c) ((P.bin.drop c).take k) := by
  unfold Holds
  rw [show ((P.bin.drop c).take k).length = k by simp; omega]
  apply List.ext_getElem (by simp [Prog.Runs.read]; omega)
  intro i h1 h2
  simp only [Prog.Runs.read, List.getElem_map, List.getElem_range, List.getElem_take,
    List.getElem_drop]
  simp only [Prog.Runs.read, List.length_map, List.length_range] at h1
  rw [show BASE + c + i = BASE + (c + i) by omega]
  unfold Prog.init
  exact Prog.Runs.write_getByte_in _ _ _ _ (by omega) hend

theorem leToNat_leOfNat (w n : Nat) : Bytes.leToNat (Bytes.leOfNat w n) = n % 256 ^ w := by
  induction w generalizing n with
  | zero => simp [Bytes.leOfNat, Bytes.leToNat, Nat.mod_one]
  | succ w ih =>
    show (BitVec.ofNat 8 n).toNat + 256 * Bytes.leToNat (Bytes.leOfNat w (n / 256)) = _
    rw [ih, BitVec.toNat_ofNat, show 256 ^ (w + 1) = 256 * 256 ^ w by ring,
      Nat.mod_mul (a := 256)]

/-- A segment that keeps memory keeps what it holds. -/
theorem holds_of_mem {s t : MachineState} {a : Nat} {bs : Bytes} (h : ∀ w, t.getMem w = s.getMem w)
    (hs : Holds s a bs) : Holds t a bs :=
  holds_congr (fun i _ => by unfold MachineState.getByte; rw [h]) hs

/-- The `QUERY` for input `i` of an algorithm (`MxxCore.inputGame`) at instruction `j`: the request
`0x00 ++ u64 i` held at `req`, the response written at `buf`, its length in `a0`. -/
theorem query_input (P : Prog) (inputs : List Bytes) (i j req buf cap : Nat) (s : MachineState)
    (hi : P.instrs[j]? = some .ECALL) (h17 : s.getReg .x17 = W 1)
    (h10 : s.getReg .x10 = W req) (h11 : s.getReg .x11 = W 9) (h12 : s.getReg .x12 = W buf)
    (h13 : s.getReg .x13 = W cap) (hreq : Holds s req (0 :: Bytes.u64 i))
    (hlen : (inputs.getD i []).length ≤ cap)
    (hr : BASE ≤ req ∧ req + 9 ≤ BASE + P.mem) (hb : BASE + P.bin.length ≤ buf ∧ buf + cap ≤ BASE + P.mem)
    (hmem : BASE + P.mem < 2 ^ 64) (hidx : i < 2 ^ 64)
    {n : Nat} {Y : ProbComp (Outcome × Unit)}
    (h : P.Runs (MxxCore.inputGame inputs) (j + 1)
      ((Prog.Runs.write s buf (inputs.getD i [])).setReg .x10
        (BitVec.ofNat 64 (inputs.getD i []).length)) () n Y) :
    P.Runs (MxxCore.inputGame inputs) j s () (n + (inputs.getD i []).length + 9 + 1) Y := by
  have hreq' : Prog.Runs.read s req 9 = 0 :: Bytes.u64 i := by
    have := hreq
    unfold Holds at this
    simpa [Mem.length_u64] using this
  have hidx' : i % 18446744073709551616 = i := Nat.mod_eq_of_lt (by simpa using hidx)
  refine Prog.Runs.query' (P := P) (j := j) (g := MxxCore.inputGame inputs) (st := ())
    (st' := ()) (resp := inputs.getD i []) (Y := Y) (m := n) (a := req) (len := 9) (buf := buf)
    (cap := cap) hi h17 (by rw [h10]; exact W_toNat _ (by omega)) (by rw [h11]; rfl)
    (by rw [h12]; exact W_toNat _ (by omega)) (by rw [h13]; exact W_toNat _ (by omega)) hr hb
    (by rw [hreq']; simp [MxxCore.inputGame, Bytes.u64, leToNat_leOfNat, hidx'])
    hlen h

end MxxCore.Leaf
