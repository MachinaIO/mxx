import MxxCore.Glue
import MxxCore.Sym

/-!
# The copy routine of proof images

Every proof image (`mxx_core::flatten`) contains one copy routine, at an instruction index that
depends on its glue. Its aligned path (destination, source, and length multiples of 8) is proved
here once for every program that contains it at some offset `c` (`At P c`): `aligned` from its
entry to its `ret`, and `call` for a call through `auipc`/`jalr` and back.
-/

namespace MxxCore.Copy

open MathcalA RiscvZkvm.Rv64

/-- The copy routine of proof images (`mxx_core::flatten`): `a0` destination, `a1` source, `a2`
length; doublewords when all three are multiples of 8, bytes otherwise; then `ret`. -/
def code : List Instr :=
  [.OR .x5 .x10 .x11, .OR .x5 .x5 .x12, .ANDI .x5 .x5 0x7#12, .BEQ .x5 .x0 0x8#13,
   .JAL .x0 0x24#21, .BNE .x12 .x0 0x8#13, .JAL .x0 0x3c#21, .LD .x6 .x11 0x0#12,
   .SD .x10 .x6 0x0#12, .ADDI .x10 .x10 0x8#12, .ADDI .x11 .x11 0x8#12,
   .ADDI .x12 .x12 0xff8#12, .JAL .x0 0x1fffe4#21, .BNE .x12 .x0 0x8#13, .JAL .x0 0x1c#21,
   .LBU .x6 .x11 0x0#12, .SB .x10 .x6 0x0#12, .ADDI .x10 .x10 0x1#12, .ADDI .x11 .x11 0x1#12,
   .ADDI .x12 .x12 0xfff#12, .JAL .x0 0x1fffe4#21, .JALR .x0 .x1 0x0#12]

/-- The routine at instruction `c` of `P`. -/
def At (P : Prog) (c : Nat) : Prop :=
  ∀ i (h : i < code.length), P.instrs[c + i]? = some code[i]

variable {P : Prog} {c : Nat}

theorem At.get (h : At P c) (i : Nat) (hi : i < code.length) : P.instrs[c + i]? = some code[i] :=
  h i hi

theorem W_or_or_and7 (a b d : Nat) (ha : a % 8 = 0) (hb : b % 8 = 0) (hd : d % 8 = 0) :
    (BitVec.ofNat 64 a ||| BitVec.ofNat 64 b ||| BitVec.ofNat 64 d) &&& 7#64 = 0#64 := by
  apply BitVec.eq_of_toNat_eq
  have h7 : (7#64).toNat = 2 ^ 3 - 1 := by decide
  simp only [BitVec.toNat_and, BitVec.toNat_or, BitVec.toNat_ofNat, h7]
  rw [Nat.and_two_pow_sub_one_eq_mod, Nat.or_mod_two_pow, Nat.or_mod_two_pow,
    Nat.mod_mod_of_dvd _ (by decide : 2 ^ 3 ∣ 2 ^ 64), Nat.mod_mod_of_dvd _ (by decide : 2 ^ 3 ∣ 2 ^ 64),
    Nat.mod_mod_of_dvd _ (by decide : 2 ^ 3 ∣ 2 ^ 64)]
  simp [ha, hb, hd]

/-- The registers the routine keeps: all but `t0`, `t1`, `a0`–`a2`. -/
def Kept (s0 s : MachineState) : Prop :=
  ∀ r : Reg, r ≠ .x5 → r ≠ .x6 → r ≠ .x10 → r ≠ .x11 → r ≠ .x12 → s.getReg r = s0.getReg r

/-- What an aligned copy of `8 N` bytes from `src` to `dst` in `P` assumes. -/
structure Pre (P : Prog) (s : MachineState) (dst src N : Nat) : Prop where
  h10 : s.getReg .x10 = W dst
  h11 : s.getReg .x11 = W src
  h12 : s.getReg .x12 = W (8 * N)
  dal : dst % 8 = 0
  sal : src % 8 = 0
  drange : BASE + P.bin.length ≤ dst ∧ dst + 8 * N ≤ BASE + P.mem
  srange : BASE ≤ src ∧ src + 8 * N ≤ BASE + P.mem
  disj : dst + 8 * N ≤ src ∨ src + 8 * N ≤ dst
  hmem : BASE + P.mem < 2 ^ 64

/-- The loop head after `i` doublewords. -/
def Inv (s0 : MachineState) (dst src N i : Nat) (s : MachineState) : Prop :=
  s.getReg .x10 = W (dst + 8 * i) ∧ s.getReg .x11 = W (src + 8 * i) ∧
    s.getReg .x12 = W (8 * N - 8 * i) ∧
    (∀ k < i, s.getMem (W (dst + 8 * k)) = s0.getMem (W (src + 8 * k))) ∧
    (∀ w, (∀ k < i, w ≠ W (dst + 8 * k)) → s.getMem w = s0.getMem w) ∧ Kept s0 s

theorem prologue (hcode : At P c) (_hc : BASE + 4 * (c + 22) < 2 ^ 64) (s : MachineState)
    (dst src N : Nat) (h : Pre P s dst src N) : P.Spec c s 4 (c + 5) (Inv s dst src N 0) := by
  obtain ⟨h10, h11, h12, dal, sal, -, -, -, -⟩ := h
  have h0 := hcode.get 0 (by decide); have h1 := hcode.get 1 (by decide)
  have h2 := hcode.get 2 (by decide); have h3 := hcode.get 3 (by decide)
  simp only [code, List.getElem_cons_succ, List.getElem_cons_zero, Nat.add_zero] at h0 h1 h2 h3
  mrunAt [Inv, Kept, h10, h11, h12, W_or_or_and7 dst src (8 * N) dal sal (by omega)]
  all_goals first
    | (refine ⟨by simp, by simp, by simp, fun k hk => absurd hk (Nat.not_lt_zero k),
        fun _ _ => trivial, ?_⟩
       intro r h5 h6 h10' h11' h12'
       simp [*])
    | (simp at hbr)

/-- The state after the load and store of iteration \`i\`. -/
def Stored (s : MachineState) (dst src i : Nat) : MachineState :=
  (s.setReg .x6 (s.getMem (W (src + 8 * i)))).setMem (W (dst + 8 * i)) (s.getMem (W (src + 8 * i)))

theorem segA (hcode : At P c) (_hc : BASE + 4 * (c + 22) < 2 ^ 64) (s : MachineState) (v : Nat)
    (h12 : s.getReg .x12 = W v) (hv : 0 < v) (hv' : v < 2 ^ 64) :
    P.Spec (c + 5) s 1 (c + 7) (· = s) := by
  have h5 := hcode.get 5 (by decide)
  simp only [code, List.getElem_cons_succ, List.getElem_cons_zero] at h5
  have hz : (BitVec.ofNat 64 v == 0#64) = false := by
    simp only [beq_eq_false_iff_ne, ne_eq]
    intro e
    have := (W_eq v 0 hv' (by norm_num)).1 e
    omega
  mrunAt [h12, hz]
  simp [bne, hz] at hbr

theorem segB (hcode : At P c) (hc : BASE + 4 * (c + 22) < 2 ^ 64) (s : MachineState)
    (dst src i : Nat) (h10 : s.getReg .x10 = W (dst + 8 * i))
    (h11 : s.getReg .x11 = W (src + 8 * i)) (dal : dst % 8 = 0) (sal : src % 8 = 0)
    (hd : BASE + P.bin.length ≤ dst + 8 * i ∧ dst + 8 * i + 8 ≤ BASE + P.mem)
    (hs : BASE ≤ src + 8 * i ∧ src + 8 * i + 8 ≤ BASE + P.mem) (hmem : BASE + P.mem < 2 ^ 64) :
    P.Spec (c + 7) s 2 (c + 9) (· = Stored s dst src i) := by
  have h7 := hcode.get 7 (by decide); have h8 := hcode.get 8 (by decide)
  simp only [code, List.getElem_cons_succ, List.getElem_cons_zero] at h7 h8
  mrunAt [Stored, h10, h11]
  all_goals omega

/-- The pointer updates of one iteration. -/
def Advanced (t : MachineState) (a b d : Nat) : MachineState :=
  ((t.setReg .x10 (W (a + 8))).setReg .x11 (W (b + 8))).setReg .x12 (W (d - 8))

theorem segC (hcode : At P c) (_hc : BASE + 4 * (c + 22) < 2 ^ 64) (t : MachineState)
    (a b d : Nat) (h10 : t.getReg .x10 = W a) (h11 : t.getReg .x11 = W b)
    (h12 : t.getReg .x12 = W d) (hd : 8 ≤ d) :
    P.Spec (c + 9) t 3 (c + 12) (· = Advanced t a b d) := by
  have h9 := hcode.get 9 (by decide); have h10' := hcode.get 10 (by decide)
  have h11' := hcode.get 11 (by decide)
  simp only [code, List.getElem_cons_succ, List.getElem_cons_zero] at h9 h10' h11'
  mrunAt [Advanced, h10, h11, h12]
  rw [← BitVec.ofNat_add, ← BitVec.ofNat_add, ← BitVec.ofNat_add,
    show BitVec.ofNat 64 (d + 18446744073709551608) = BitVec.ofNat 64 (d - 8) from
      W_mod (by omega)]

theorem segD (hcode : At P c) (hc : BASE + 4 * (c + 22) < 2 ^ 64) (t : MachineState) :
    P.Spec (c + 12) t 1 (c + 5) (· = t) := by
  have h12 := hcode.get 12 (by decide)
  simp only [code, List.getElem_cons_succ, List.getElem_cons_zero] at h12
  mrunAt []

theorem getMem_setMem' (s : MachineState) (a b v : Word) :
    (s.setMem a v).getMem b = if b == a then v else s.getMem b := rfl

theorem step (s0 s : MachineState) (dst src N i : Nat) (h : Pre P s0 dst src N)
    (hi : i < N) (hI : Inv s0 dst src N i s) :
    Inv s0 dst src N (i + 1)
      (Advanced (Stored s dst src i) (dst + 8 * i) (src + 8 * i) (8 * N - 8 * i)) := by
  obtain ⟨-, -, -, dal, sal, ⟨d1, d2⟩, ⟨s1, s2⟩, disj, hmem⟩ := h
  obtain ⟨h10, h11, h12, hcopy, hframe, hkept⟩ := hI
  have inj : ∀ a b : Nat, a < 2 ^ 64 → b < 2 ^ 64 → (W a = W b ↔ a = b) :=
    fun a b ha hb => W_eq a b ha hb
  have hsrc : s.getMem (W (src + 8 * i)) = s0.getMem (W (src + 8 * i)) :=
    hframe _ fun k hk e => by
      have := (inj _ _ (by omega) (by omega)).1 e
      omega
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_⟩
  · simp only [Advanced, Stored, Sym.getReg_setReg, Sym.getReg_setMem]; congr 1
  · simp only [Advanced, Stored, Sym.getReg_setReg, Sym.getReg_setMem]; congr 1
  · simp only [Advanced, Stored, Sym.getReg_setReg, Sym.getReg_setMem]; congr 1
  · intro k hk
    simp only [Advanced, Stored, MachineState.getMem_setReg, getMem_setMem']
    by_cases hki : k = i
    · subst hki; simp [hsrc]
    · have hne : (W (dst + 8 * k) == W (dst + 8 * i)) = false := by
        simp only [beq_eq_false_iff_ne, ne_eq]
        intro e; exact hki (by have := (inj _ _ (by omega) (by omega)).1 e; omega)
      simp only [hne, Bool.false_eq_true, ↓reduceIte]
      exact hcopy k (by omega)
  · intro w hw
    simp only [Advanced, Stored, MachineState.getMem_setReg, getMem_setMem']
    have hne : (w == W (dst + 8 * i)) = false := by
      simp only [beq_eq_false_iff_ne, ne_eq]; exact hw i (by omega)
    simp only [hne, Bool.false_eq_true, ↓reduceIte]
    exact hframe w fun k hk => hw k (by omega)
  · intro r h5 h6 h10' h11' h12'
    simp only [Advanced, Stored, Sym.getReg_setReg, Sym.getReg_setMem]
    simp [*]
    exact hkept r h5 h6 h10' h11' h12'

set_option maxRecDepth 20000 in
theorem body (hcode : At P c) (hc : BASE + 4 * (c + 22) < 2 ^ 64) (s0 s : MachineState)
    (dst src N i : Nat) (h : Pre P s0 dst src N) (hi : i < N) (hI : Inv s0 dst src N i s) :
    P.Spec (c + 5) s 7 (c + 5) (Inv s0 dst src N (i + 1)) := by
  obtain ⟨-, -, -, dal, sal, ⟨d1, d2⟩, ⟨s1, s2⟩, -, hmem⟩ := id h
  obtain ⟨h10, h11, h12, -⟩ := id hI
  refine Prog.Spec.seq (n := 1) (n' := 6)
    (segA hcode hc s (8 * N - 8 * i) h12 (by omega) (by omega)) fun t e => ?_
  subst e
  refine Prog.Spec.seq (n := 2) (n' := 4)
    (segB hcode hc t dst src i h10 h11 dal sal ⟨by omega, by omega⟩ ⟨by omega, by omega⟩ hmem)
    fun t' e => ?_
  subst e
  refine Prog.Spec.seq (n := 3) (n' := 1)
    (segC hcode hc (Stored t dst src i) (dst + 8 * i) (src + 8 * i) (8 * N - 8 * i)
      (by simp [Stored, h10]) (by simp [Stored, h11]) (by simp [Stored, h12]) (by omega))
    fun t'' e => ?_
  subst e
  exact Prog.Spec.mono (segD hcode hc _) le_rfl fun _ e => e ▸ step s0 t dst src N i h hi hI

/-- What the routine leaves after an aligned copy: the doublewords copied, every other doubleword
unchanged, and the registers it keeps. -/
def Post (s0 : MachineState) (dst src N : Nat) (s : MachineState) : Prop :=
  (∀ k < N, s.getMem (W (dst + 8 * k)) = s0.getMem (W (src + 8 * k))) ∧
    (∀ w, (∀ k < N, w ≠ W (dst + 8 * k)) → s.getMem w = s0.getMem w) ∧ Kept s0 s

theorem exit (hcode : At P c) (_hc : BASE + 4 * (c + 22) < 2 ^ 64) (s0 s : MachineState)
    (dst src N : Nat) (hI : Inv s0 dst src N N s) :
    P.Spec (c + 5) s 2 (c + 21) (Post s0 dst src N) := by
  have h5 := hcode.get 5 (by decide); have h6 := hcode.get 6 (by decide)
  have h7 := hcode.get 7 (by decide)
  simp only [code, List.getElem_cons_succ, List.getElem_cons_zero] at h5 h6 h7
  obtain ⟨-, -, h12, hcopy, hframe, hkept⟩ := hI
  have h0 : s.getReg .x12 = W 0 := by rw [h12]; congr 1; omega
  mrunAt [Post, h0]
  all_goals first | exact ⟨hcopy, hframe, hkept⟩ | (simp at hbr)

/-- The aligned copy of \`8 N\` bytes, from the routine's entry to its \`ret\`. -/
theorem aligned (hcode : At P c) (hc : BASE + 4 * (c + 22) < 2 ^ 64) (s0 : MachineState)
    (dst src N : Nat) (h : Pre P s0 dst src N) :
    P.Spec c s0 (4 + ((N - 0) * 7 + 2)) (c + 21) (Post s0 dst src N) :=
  Prog.Spec.seq (prologue hcode hc s0 dst src N h) fun s hs =>
    Prog.Spec.loop (I := Inv s0 dst src N)
      (fun i s hi hI => body hcode hc s0 s dst src N i h hi hI)
      (fun s hI => exit hcode hc s0 s dst src N hI) 0 s (Nat.zero_le _) hs

/-- A call of the routine at `c` from instruction `j` (`auipc ra, 0`; `jalr ra, off(ra)`), back at
`j + 2` through the routine's `ret`. -/
theorem call (hcode : At P c) (hc : BASE + 4 * (c + 22) < 2 ^ 64) (j : Nat) (off : BitVec 12)
    (s : MachineState) (dst src N : Nat)
    (hauipc : P.instrs[j]? = some (.AUIPC .x1 0x0#20))
    (hjalr : P.instrs[j + 1]? = some (.JALR .x1 .x1 off))
    (ht : (pcOf j + ((0x0#20 : BitVec 20).zeroExtend 32 <<< 12 : BitVec 32).signExtend 64 +
      signExtend12 off) &&& ~~~1#64 = pcOf c)
    (hback : (pcOf (j + 1 + 1) + signExtend12 0x0#12) &&& ~~~1#64 = pcOf (j + 2))
    (h : Pre P s dst src N) :
    P.Spec j s (2 + ((4 + ((N - 0) * 7 + 2)) + 1)) (j + 2)
      (Post (s.setReg .x1 (pcOf (j + 1 + 1))) dst src N) := by
  rw [show 2 + ((4 + ((N - 0) * 7 + 2)) + 1) = ((4 + ((N - 0) * 7 + 2)) + 1) + 1 + 1 by omega]
  refine MxxCore.Glue.auipc hauipc ?_
  refine MxxCore.Glue.jalr (t := c) hjalr (by simpa using ht) ?_
  rw [Sym.setReg_setReg_same]
  have hpre : Pre P (s.setReg .x1 (pcOf (j + 1 + 1))) dst src N := by
    obtain ⟨h10, h11, h12, r1, r2, r3, r4, r5, r6⟩ := h
    exact ⟨by simpa using h10, by simpa using h11, by simpa using h12, r1, r2, r3, r4, r5, r6⟩
  refine Prog.Spec.seq (aligned hcode hc _ dst src N hpre) fun s' hs' => ?_
  have hra : s'.getReg .x1 = pcOf (j + 1 + 1) := by
    rw [hs'.2.2 .x1 (by decide) (by decide) (by decide) (by decide) (by decide)]
    simp
  have hret := hcode.get 21 (by decide)
  simp only [code, List.getElem_cons_succ, List.getElem_cons_zero] at hret
  refine MxxCore.Glue.jalr (t := j + 2) hret (by rw [hra]; exact hback) ?_
  exact Prog.Spec.done (by simpa using hs')

end MxxCore.Copy
