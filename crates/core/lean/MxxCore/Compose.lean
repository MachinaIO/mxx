import MxxCore.Leaf
import MathcalA.Lib.Coins
/-!
# Composing a straight-line glue

A proof image whose glue is straight-line code (`mxx_core::flatten::ProofImage::trace`) runs as a
sequence of events: input queries, leaf calls, copies, and one `HALT`, with glue segments between
them that only set registers. The generated run proof of such an image (`mxx_core::compose`)
chains one lemma of this module per event. Each lemma hands its continuation the state after the
event, what the event wrote (`Holds` facts), and a `Frame`: every range apart from the written
ones holds after the event as before it. Every side condition about addresses is a closed
proposition about numerals (`LeafNum`, `CopyNum`, `Frame` arguments), proved by `decide`.
-/

namespace MxxCore.Compose

open MathcalA RiscvZkvm.Rv64 MxxCore.Leaf

instance (a m b n : Nat) : Decidable (Apart a m b n) :=
  inferInstanceAs (Decidable (_ ∨ _))

/-- An unsigned comparison of words of naturals below `2^64`. -/
theorem ult_W {a b : Nat} (ha : a < 2 ^ 64) (hb : b < 2 ^ 64) :
    (BitVec.ofNat 64 a).ult (BitVec.ofNat 64 b) = decide (a < b) := by
  simp [BitVec.ult]
  omega

/-- The memory and the image size of `P`, as numerals. -/
structure Sizes (P : Prog) (mem len : Nat) : Prop where
  hmem : P.mem = mem
  hlen : P.bin.length = len
  hfit : BASE + mem < 2 ^ 64
  hbin : len ≤ mem

/-- Every range of `s` apart from the ranges `ws` holds in `t` what it holds in `s`. -/
def Frame (ws : List (Nat × Nat)) (s t : MachineState) : Prop :=
  ∀ (r n : Nat) (bs : Bytes), bs.length = n → r + n < 2 ^ 64 → (∀ x ∈ ws, Apart r n x.1 x.2) →
    Holds s r bs → Holds t r bs

/-- The loaded image holds its bytes. -/
theorem init {P : Prog} {mem len : Nat} (hP : Sizes P mem len) : Holds P.init BASE P.bin := by
  have := holds_init P 0 P.bin.length (by simp)
    (by have := hP.hfit; have := hP.hlen; have := hP.hbin; omega)
  simpa using this

/-- A held range holds each of its slices. -/
theorem slice {s : MachineState} {a : Nat} {bs : Bytes} (h : Holds s a bs) (off n b : Nat)
    (v : Bytes) (hb : b = a + off) (hn : off + n ≤ bs.length) (hv : (bs.drop off).take n = v) :
    Holds s b v := by
  subst hb hv
  unfold Holds at *
  have hl : ((bs.drop off).take n).length = n := by simp; omega
  rw [hl]
  apply List.ext_getElem (by simp [Prog.Runs.read]; omega)
  intro i h1 h2
  have := congrArg (fun l => l[off + i]?) h
  simp only [Prog.Runs.read, List.length_map, List.length_range] at h1
  rw [List.getElem?_eq_getElem (by simp [Prog.Runs.read]; omega),
    List.getElem?_eq_getElem (by omega)] at this
  simp only [Option.some.injEq, Prog.Runs.read, List.getElem_map, List.getElem_range] at this
  simp only [Prog.Runs.read, List.getElem_map, List.getElem_range, List.getElem_take,
    List.getElem_drop, ← this, Nat.add_assoc]

/-- Adjacent held ranges make one. -/
theorem append {s : MachineState} {a b : Nat} {x y : Bytes} {n : Nat} (hx : Holds s a x)
    (hl : x.length = n) (hy : Holds s b y) (hb : b = a + n) : Holds s a (x ++ y) :=
  holds_append hx (by rw [hl, ← hb]; exact hy)

/-- A glue segment that keeps memory and sets registers (`R`). -/
theorem phase {P : Prog} {g : Game} {j k : Nat} {s : MachineState} {st : g.State} {n m : Nat}
    {R : MachineState → Prop} {Y : ProbComp (Outcome × g.State)}
    (h : P.Spec j s n k fun t => (∀ w, t.getMem w = s.getMem w) ∧ R t)
    (h' : ∀ t, R t → Frame [] s t → P.Runs g k t st m Y) : P.Runs g j s st (n + m) Y :=
  Prog.Runs.spec h fun t ht => h' t ht.2 fun _ _ _ _ _ _ hs => holds_of_mem ht.1 hs

/-! ### Segments that store words -/

/-- `w` is none of the doublewords at `as`. -/
def Avoid : List Nat → Word → Prop
  | [], _ => True
  | a :: as, w => w ≠ W a ∧ Avoid as w

/-- `t` holds the words `x.2` at the addresses `x.1`. -/
def Stored : List (Nat × Nat) → MachineState → Prop
  | [], _ => True
  | x :: xs, t => t.getMem (W x.1) = W x.2 ∧ Stored xs t

theorem avoid_of_apart {as : List Nat} {c : Nat} (hc : c < 2 ^ 64)
    (h : ∀ a ∈ as, a % 8 = 0 ∧ a + 8 < 2 ^ 64 ∧ (c < a ∨ a + 8 ≤ c)) :
    Avoid as (alignToDword (BitVec.ofNat 64 c)) := by
  induction as with
  | nil => trivial
  | cons a as ih =>
    refine ⟨fun e => ?_, ih fun b hb => h b (List.mem_cons_of_mem _ hb)⟩
    obtain ⟨ha, ha', hout⟩ := h a (List.mem_cons_self ..)
    rw [Mem.alignToDword_ofNat hc] at e
    have := (W_eq _ _ (by omega) (by omega)).1 e
    omega

theorem stored_getMem {xs : List (Nat × Nat)} {t : MachineState} (h : Stored xs t) :
    ∀ x ∈ xs, t.getMem (W x.1) = W x.2 := by
  induction xs with
  | nil => intro x hx; cases hx
  | cons y ys ih =>
    intro x hx
    rcases List.mem_cons.1 hx with rfl | hx
    · exact h.1
    · exact ih h.2 x hx

/-- A glue segment that sets registers (`R`) and stores the words of `stores`, at aligned
addresses, keeping every other doubleword. -/
theorem phaseStore {P : Prog} {g : Game} {j k : Nat} {s : MachineState} {st : g.State}
    {n m : Nat} {R : MachineState → Prop} {Y : ProbComp (Outcome × g.State)}
    (stores : List (Nat × Nat)) (hal : ∀ x ∈ stores, x.1 % 8 = 0 ∧ x.1 + 8 < 2 ^ 64)
    (h : P.Spec j s n k fun t => (∀ w, Avoid (stores.map (·.1)) w → t.getMem w = s.getMem w) ∧
      Stored stores t ∧ R t)
    (h' : ∀ t, R t → Stored stores t → Frame (stores.map fun x => (x.1, 8)) s t →
      P.Runs g k t st m Y) : P.Runs g j s st (n + m) Y := by
  refine Prog.Runs.spec h fun t ht => h' t ht.2.2 ht.2.1 fun r n bs hbs hr hap hs => ?_
  refine holds_congr (fun i hi => ?_) hs
  unfold MachineState.getByte
  rw [ht.1]
  refine avoid_of_apart (by omega) fun a ha => ?_
  obtain ⟨x, hx, rfl⟩ := List.mem_map.1 ha
  have := hap (x.1, 8) (List.mem_map.2 ⟨x, hx, rfl⟩)
  have := hal x hx
  unfold Apart at *
  omega

/-! ### Loops -/

/-- A counted loop from its head at `top`: `N` iterations, each from a state satisfying `I i`
back to the head in a state satisfying `I (i + 1)` within `c` steps, then the exit from a state
satisfying `I N` (the last argument, so a generated proof continues there). -/
theorem loop {P : Prog} {g : Game} {st : g.State} {Y : ProbComp (Outcome × g.State)}
    (top N c : Nat) {M : Nat} (I : Nat → MachineState → Prop)
    (hbody : ∀ i < N, ∀ s, I i s → ∀ r, (∀ s', I (i + 1) s' → P.Runs g top s' st r Y) →
      P.Runs g top s st (c + r) Y)
    (s : MachineState) (h : I 0 s) (hexit : ∀ s, I N s → P.Runs g top s st M Y) :
    P.Runs g top s st (N * c + M) Y := by
  have key : ∀ k i, i + k = N → ∀ s, I i s → P.Runs g top s st (k * c + M) Y := by
    intro k
    induction k with
    | zero => intro i hi s h; simp only [Nat.add_zero] at hi; subst hi; simpa using hexit s h
    | succ k ih =>
      intro i hi s h
      have := hbody i (by omega) s h (k * c + M) fun s' h' => ih (i + 1) (by omega) s' h'
      rwa [show c + (k * c + M) = (k + 1) * c + M by ring] at this
  exact key N 0 (by omega) s h

/-- A counted loop that stops early, at iteration `t`: the iterations before `t` go back to the
head in a state satisfying the next invariant within `c` steps each, and iteration `t` (its stop
included) and everything after it run from a state satisfying `I t`. -/
theorem loopUntil {P : Prog} {g : Game} {st : g.State} {Y : ProbComp (Outcome × g.State)}
    (top c t : Nat) {K : Nat} (I : Nat → MachineState → Prop)
    (hcont : ∀ i < t, ∀ s, I i s → ∀ r, (∀ s', I (i + 1) s' → P.Runs g top s' st r Y) →
      P.Runs g top s st (c + r) Y)
    (s : MachineState) (h : I 0 s) (hlast : ∀ s, I t s → P.Runs g top s st K Y) :
    P.Runs g top s st (t * c + K) Y := by
  have key : ∀ k i, i + k = t → ∀ s, I i s → P.Runs g top s st (k * c + K) Y := by
    intro k
    induction k with
    | zero => intro i hi s h; simp only [Nat.add_zero] at hi; subst hi; simpa using hlast s h
    | succ k ih =>
      intro i hi s h
      have := hcont i (by omega) s h (k * c + K) fun s' h' => ih (i + 1) (by omega) s' h'
      rwa [show c + (k * c + K) = (k + 1) * c + K by ring] at this
  exact key t 0 (by omega) s h

/-- The members `[0, i)` of a family of `n`-byte members at `a`: member `j` has `n` bytes and
holds `O j`. -/
def Members (s : MachineState) (a n : Nat) (O : Nat → Bytes) (i : Nat) : Prop :=
  ∀ j < i, (O j).length = n ∧ Holds s (a + n * j) (O j)

theorem members_zero (s : MachineState) (a n : Nat) (O : Nat → Bytes) : Members s a n O 0 :=
  fun _ h => absurd h (Nat.not_lt_zero _)

/-- Members held apart from what a frame writes still hold. -/
theorem members_frame {ws : List (Nat × Nat)} {s t : MachineState} {a n i : Nat}
    {O : Nat → Bytes} (F : Frame ws s t)
    (hap : ∀ j < i, a + n * j + n < 2 ^ 64 ∧ ∀ x ∈ ws, Apart (a + n * j) n x.1 x.2)
    (h : Members s a n O i) : Members t a n O i :=
  fun j hj => ⟨(h j hj).1, F _ _ _ (h j hj).1 (hap j hj).1 (hap j hj).2 (h j hj).2⟩

/-- One more member. -/
theorem members_succ {s : MachineState} {a n i : Nat} {O : Nat → Bytes} (h : Members s a n O i)
    (hl : (O i).length = n) (hi : Holds s (a + n * i) (O i)) : Members s a n O (i + 1) :=
  fun j hj => by
    rcases Nat.lt_succ_iff_lt_or_eq.1 hj with hj | rfl
    · exact h j hj
    · exact ⟨hl, hi⟩

theorem length_flatMap_range {n N : Nat} {O : Nat → Bytes} (hl : ∀ j < N, (O j).length = n) :
    ((List.range N).flatMap O).length = n * N := by
  induction N with
  | zero => simp
  | succ N ih =>
    rw [List.range_succ, List.flatMap_append, List.flatMap_singleton, List.length_append,
      ih fun j hj => hl j (by omega), hl N (by omega)]
    ring

/-- All `N` members, as one range. -/
theorem members_holds {s : MachineState} {a n N : Nat} {O : Nat → Bytes}
    (h : Members s a n O N) : Holds s a ((List.range N).flatMap O) := by
  induction N with
  | zero => simp [Holds, Prog.Runs.read]
  | succ N ih =>
    rw [List.range_succ, List.flatMap_append, List.flatMap_singleton]
    refine holds_append (ih fun j hj => h j (by omega)) ?_
    rw [length_flatMap_range fun j hj => (h j (by omega)).1]
    exact (h N (by omega)).2

/-- The `QUERY` of input `i` at instruction `j`, its response of `cap` bytes landing at `buf`. -/
theorem query {P : Prog} {mem len : Nat} (hP : Sizes P mem len) (inputs : List Bytes)
    (i j req buf cap : Nat) (s : MachineState) {n : Nat} {Y : ProbComp (Outcome × Unit)}
    (hi : P.instrs[j]? = some .ECALL) (h10 : s.getReg .x10 = W req) (h11 : s.getReg .x11 = W 9)
    (h12 : s.getReg .x12 = W buf) (h13 : s.getReg .x13 = W cap) (h17 : s.getReg .x17 = W 1)
    (hreq : Holds s req (0 :: Bytes.u64 i)) (hlen : (inputs.getD i []).length = cap)
    (hnum : BASE ≤ req ∧ req + 9 ≤ BASE + mem ∧ BASE + len ≤ buf ∧ buf + cap ≤ BASE + mem ∧
      i < 2 ^ 64)
    (h' : ∀ t, Holds t buf (inputs.getD i []) → t.getReg .x10 = W cap →
      Frame [(buf, cap)] s t → P.Runs (MxxCore.inputGame inputs) (j + 1) t () n Y) :
    P.Runs (MxxCore.inputGame inputs) j s () (n + cap + 9 + 1) Y := by
  obtain ⟨hm, hl, hfit, -⟩ := hP
  obtain ⟨r1, r2, b1, b2, hidx⟩ := hnum
  have := query_input P inputs i j req buf cap s hi h17 h10 h11 h12 h13 hreq (by omega)
    ⟨r1, by rw [hm]; exact r2⟩ ⟨by rw [hl]; exact b1, by rw [hm]; exact b2⟩ (by rw [hm]; exact hfit)
    hidx (h' _ (holds_setReg _ _ (holds_write _ _ _ (by omega))) (by subst hlen; simp)
      fun r m bs hbs hr hap hs => holds_setReg _ _ (holds_write_far _ _ hs (by omega) (by omega)
        (by have := hap _ (List.mem_singleton_self _); rw [hbs, hlen]; exact this)))
  rwa [hlen] at this

/-- The closed conditions on a leaf call's addresses: the memory fits, the descriptor's words are
words, the descriptor is aligned and in memory, the inputs are aligned and in memory, the written
ranges are aligned, past the image and in memory, and nothing written overlaps an input, the
descriptor, or another written range. -/
def LeafNum (mem len d : Nat) (words : List Nat) (ins written : List (Nat × Nat)) : Prop :=
  (∀ w ∈ words, w < 2 ^ 64) ∧ d % 8 = 0 ∧ BASE ≤ d ∧ d + 8 * words.length ≤ BASE + mem ∧
    (∀ x ∈ ins, x.1 % 8 = 0 ∧ BASE ≤ x.1 ∧ x.1 + x.2 ≤ BASE + mem) ∧
    (∀ x ∈ written, x.1 % 8 = 0 ∧ BASE + len ≤ x.1 ∧ x.1 + x.2 ≤ BASE + mem) ∧
    (∀ x ∈ written, ∀ y ∈ ins, Apart x.1 x.2 y.1 y.2) ∧
    (∀ x ∈ written, Apart x.1 x.2 d (8 * words.length)) ∧
    written.Pairwise fun x y => Apart x.1 x.2 y.1 y.2

instance (mem len d : Nat) (words : List Nat) (ins written : List (Nat × Nat)) :
    Decidable (LeafNum mem len d words ins written) := by
  unfold LeafNum; infer_instance

theorem words_u64s (vs : List Nat) (h : ∀ v ∈ vs, v < 2 ^ 64) : words (Bytes.u64s vs) = vs := by
  apply List.ext_getElem (by rw [length_words, Mem.length_u64s]; omega)
  intro k h1 h2
  simp only [words, List.getElem_map, List.getElem_range]
  have hk : ((Bytes.u64s vs).drop (8 * k)).take 8 = Bytes.u64 vs[k] := by
    apply List.ext_getElem (by simp [Mem.length_u64s, Mem.length_u64]; omega)
    intro i i1 i2
    simp only [List.getElem_take, List.getElem_drop]
    simp only [List.length_take, List.length_drop, Mem.length_u64s] at i1
    rw [Mem.getElem_u64s vs k i h2 (by omega)]
    exact (Mem.getElem_leOfNat 8 _ i (by simpa [Bytes.u64] using i2)).symm
  rw [hk, Bytes.u64, leToNat_leOfNat, Nat.mod_eq_of_lt (by have := h _ (List.getElem_mem h2); simpa using this)]

/-- A held `u64` is the doubleword. -/
theorem u64_getMem {t : MachineState} {a v : Nat} (h : Holds t a (Bytes.u64 v)) (ha : a % 8 = 0)
    (hend : a + 8 < 2 ^ 64) (hv : v < 2 ^ 64) : t.getMem (W a) = W v := by
  have hw : words (Bytes.u64 v) = [v] := by
    have := words_u64s [v] (by simpa using hv)
    simpa [Bytes.u64s] using this
  have := getMem_of_holds t a _ h 0 (by rw [Mem.length_u64]) ha (by rw [Mem.length_u64]; omega)
  simp only [Nat.mul_zero, Nat.add_zero] at this
  rw [this]
  congr 1
  simp [hw]

/-- An entry of a list of bounded values is bounded. -/
theorem getD_lt {vs : List Nat} {B : Nat} (h : ∀ v ∈ vs, v < B) (hB : 0 < B) (i : Nat) :
    vs.getD i 0 < B := by
  rw [List.getD_eq_getElem?_getD]
  cases hv : vs[i]? with
  | none => simpa using hB
  | some v => simpa using h v (List.mem_of_getElem? hv)

/-- Word `i` of a held table of words. -/
theorem table_getMem {t : MachineState} {a : Nat} {vs : List Nat} (h : Holds t a (Bytes.u64s vs))
    (ha : a % 8 = 0) (hend : a + 8 * vs.length < 2 ^ 64) (hv : ∀ v ∈ vs, v < 2 ^ 64) {i : Nat}
    (hi : i < vs.length) : t.getMem (W (a + 8 * i)) = W (vs.getD i 0) := by
  have := getMem_of_holds t a _ h i (by rw [Mem.length_u64s]; omega) ha
    (by rw [Mem.length_u64s]; omega)
  rw [this]
  congr 1
  simp [words_u64s vs hv, List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hi]

/-- The words of held bytes, one doubleword each. -/
theorem int_words {t : MachineState} {a : Nat} {E : Bytes} {n : Nat} (h : Holds t a E)
    (hl : E.length = 8 * n) (ha : a % 8 = 0) (hend : a + 8 * n < 2 ^ 64) (j : Nat) (hj : j < n) :
    t.getMem (W (a + 8 * j)) = W ((words E).getD j 0) := by
  rw [getMem_of_holds t a E h j (by omega) ha (by omega)]
  congr 1
  rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem (by rw [length_words]; omega)]
  rfl

/-- Every word of bytes is below `2^64`. -/
theorem words_getD_lt (E : Bytes) (j : Nat) : (words E).getD j 0 < 2 ^ 64 := by
  rw [List.getD_eq_getElem?_getD]
  cases hv : (words E)[j]? with
  | none => simp
  | some v =>
    obtain ⟨hj, rfl⟩ := List.getElem?_eq_some_iff.1 hv
    simp only [words, List.getElem_map, List.getElem_range, Option.getD_some]
    have := leToNat_lt ((E.drop (8 * j)).take 8)
    have hl : ((E.drop (8 * j)).take 8).length ≤ 8 := by simp
    calc _ < 256 ^ ((E.drop (8 * j)).take 8).length := this
      _ ≤ 256 ^ 8 := Nat.pow_le_pow_right (by norm_num) hl
      _ = 2 ^ 64 := by norm_num

/-- A doubleword is a held `u64`. -/
theorem u64_holds {t : MachineState} {a v : Nat} (h : t.getMem (W a) = W v) (ha : a % 8 = 0)
    (hend : a + 8 < 2 ^ 64) : Holds t a (Bytes.u64 v) := by
  have := holds_u64s t a [v] ha (by simpa using hend) fun k hk => by
    obtain rfl : k = 0 := by simpa using hk
    simpa using h
  simpa [Bytes.u64s] using this

/-- A call of the leaf `L` from instruction `j`, its descriptor and inputs held. -/
theorem leaf {P : Prog} {c : Nat} {g : Game} {st : g.State} {mem len : Nat} (hP : Sizes P mem len)
    (L : Spec P c) (j : Nat) (off : BitVec 12) (d : Nat) (args : List Nat) (ins : List (Nat × Bytes))
    (outs : List (Nat × Nat)) (sc : Nat × Nat) (s : MachineState) {n : Nat}
    {Y : ProbComp (Outcome × g.State)}
    (hauipc : P.instrs[j]? = some (.AUIPC .x1 0x0#20))
    (hjalr : P.instrs[j + 1]? = some (.JALR .x1 .x1 off))
    (ht : (pcOf j + ((0x0#20 : BitVec 20).zeroExtend 32 <<< 12 : BitVec 32).signExtend 64 +
      signExtend12 off) &&& ~~~1#64 = pcOf c)
    (hback : (pcOf (j + 1 + 1) + signExtend12 0x0#12) &&& ~~~1#64 = pcOf (j + 2))
    (hd : s.getReg .x10 = W d) (insNum : List (Nat × Nat))
    (hdesc : Holds s d (Bytes.u64s (args ++ insNum.map (·.1) ++ outs.map (·.1) ++ [sc.1])))
    (hinsNum : ins.map (fun x => (x.1, x.2.length)) = insNum)
    (hins : ∀ x ∈ ins, Holds s x.1 x.2)
    (hnum : LeafNum mem len d (args ++ insNum.map (·.1) ++ outs.map (·.1) ++ [sc.1]) insNum
      (outs ++ [sc]))
    (hdom : L.dom args (ins.map (·.2)))
    (hlen : outs.map (·.2) = (L.f args (ins.map (·.2))).map List.length)
    (h' : ∀ t, (∀ i (hi : i < outs.length), Holds t outs[i].1 ((L.f args (ins.map (·.2))).getD i [])) →
      Frame (outs ++ [sc]) s t → P.Runs g (j + 2) t st n Y) :
    P.Runs g j s st (2 + (L.steps args + 1) + n) Y := by
  obtain ⟨hm, hl, hfit, -⟩ := hP
  have hws : Call.words ⟨d, args, ins, outs, sc⟩ =
      args ++ insNum.map (·.1) ++ outs.map (·.1) ++ [sc.1] := by
    rw [← hinsNum]; simp [Call.words, List.map_map, Function.comp_def]
  rw [← hws] at hdesc hnum
  obtain ⟨hw, dal, d1, d2, hi, hwr, hai, had, hpw⟩ := hnum
  have hiw : ∀ x ∈ ins, x.1 % 8 = 0 ∧ BASE ≤ x.1 ∧ x.1 + x.2.length ≤ BASE + mem := fun x hx =>
    hi (x.1, x.2.length) (by rw [← hinsNum]; exact List.mem_map_of_mem hx)
  have entry : Entry P ⟨d, args, ins, outs, sc⟩ s :=
    { hd := hd
      desc := fun i hi' => by
        have := getMem_of_holds s d _ hdesc i (by rw [Mem.length_u64s]; omega) dal
          (by rw [Mem.length_u64s]; omega)
        rw [this]
        congr 1
        simp only [words_u64s _ hw]
      words := hw
      ins := hins
      hmem := by rw [hm]; exact hfit
      dal := dal
      drange := ⟨d1, by rw [hm]; exact d2⟩
      ial := fun x hx => by rw [hm]; exact hiw x hx
      wal := fun x hx => by rw [hl, hm]; exact hwr x hx
      apartIn := fun x hx y hy => hai x hx (y.1, y.2.length)
        (by rw [← hinsNum]; exact List.mem_map_of_mem hy)
      apartDesc := had
      apartOut := hpw }
  refine Prog.Runs.spec (L.call j off ⟨d, args, ins, outs, sc⟩ s hauipc hjalr ht hback entry hdom hlen) fun t ht' => ?_
  refine h' t (fun i hi' => ?_) fun r m bs hbs hr hap hs => ?_
  · have hlen' := congrArg List.length hlen
    simp only [List.length_map] at hlen'
    have hi2 : i < (L.f args (ins.map (·.2))).length := by omega
    rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hi2, Option.getD_some]
    exact ht'.1 i hi' hi2
  · exact holds_exit ht' (holds_setReg _ _ hs) (by omega) fun x hx => by rw [hbs]; exact hap x hx

theorem length_getD {l : List Bytes} {ns : List Nat} (h : l.map List.length = ns) (i : Nat) :
    (l.getD i []).length = ns.getD i 0 := by
  subst h
  by_cases hi : i < l.length
  · simp [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hi]
  · simp [List.getD_eq_getElem?_getD, List.getElem?_eq_none (by omega : l.length ≤ i)]

/-- The closed conditions on an aligned copy of `8 N` bytes by the routine at instruction `c`. -/
def CopyNum (mem len c dst src N : Nat) : Prop :=
  BASE + 4 * (c + 22) < 2 ^ 64 ∧ dst % 8 = 0 ∧ src % 8 = 0 ∧ BASE + len ≤ dst ∧
    dst + 8 * N ≤ BASE + mem ∧ BASE ≤ src ∧ src + 8 * N ≤ BASE + mem ∧ Apart dst (8 * N) src (8 * N)

instance (mem len c dst src N : Nat) : Decidable (CopyNum mem len c dst src N) := by
  unfold CopyNum; infer_instance

/-! ### Runtime selection -/

theorem getD_mem {α : Type} {l : List α} {d : α} {v : Nat} (hv : v < l.length) :
    l.getD v d ∈ l := by
  rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hv]
  exact List.getElem_mem hv

/-- Candidate `v` of held candidates. -/
theorem holds_select {s : MachineState} (addrs : List Nat) (cands : List Bytes) {v : Nat}
    (h : ∀ k < cands.length, Holds s (addrs.getD k 0) (cands.getD k [])) (hv : v < cands.length) :
    Holds s (addrs.getD v 0) (cands.getD v []) := h v hv

theorem select_length {cands : List Bytes} {L v : Nat} (h : ∀ c ∈ cands, c.length = L)
    (hv : v < cands.length) : (cands.getD v []).length = L := h _ (getD_mem hv)

theorem copy_num_select {mem len c dst N v : Nat} {addrs : List Nat}
    (h : ∀ a ∈ addrs, CopyNum mem len c dst a N) (hv : v < addrs.length) :
    CopyNum mem len c dst (addrs.getD v 0) N := h _ (getD_mem hv)

/-- A call of the copy routine at `c` from instruction `j`, its source held. -/
theorem copy {P : Prog} {c : Nat} {g : Game} {st : g.State} {mem len : Nat} (hP : Sizes P mem len)
    (hcode : MxxCore.Copy.At P c) (j : Nat) (off : BitVec 12) (s : MachineState)
    (dst src N : Nat) (bs : Bytes) {n : Nat} {Y : ProbComp (Outcome × g.State)}
    (hauipc : P.instrs[j]? = some (.AUIPC .x1 0x0#20))
    (hjalr : P.instrs[j + 1]? = some (.JALR .x1 .x1 off))
    (ht : (pcOf j + ((0x0#20 : BitVec 20).zeroExtend 32 <<< 12 : BitVec 32).signExtend 64 +
      signExtend12 off) &&& ~~~1#64 = pcOf c)
    (hback : (pcOf (j + 1 + 1) + signExtend12 0x0#12) &&& ~~~1#64 = pcOf (j + 2))
    (h10 : s.getReg .x10 = W dst) (h11 : s.getReg .x11 = W src) (h12 : s.getReg .x12 = W (8 * N))
    (hsrc : Holds s src bs) (hbs : bs.length = 8 * N) (hnum : CopyNum mem len c dst src N)
    (h' : ∀ t, Holds t dst bs → Frame [(dst, 8 * N)] s t → P.Runs g (j + 2) t st n Y) :
    P.Runs g j s st (2 + ((4 + ((N - 0) * 7 + 2)) + 1) + n) Y := by
  obtain ⟨hm, hl, hfit, -⟩ := hP
  obtain ⟨hc, dal, sal, d1, d2, s1, s2, disj⟩ := hnum
  have pre : MxxCore.Copy.Pre P s dst src N :=
    ⟨h10, h11, h12, dal, sal, ⟨by rw [hl]; exact d1, by rw [hm]; exact d2⟩,
      ⟨s1, by rw [hm]; exact s2⟩, by unfold Apart at disj; omega, by rw [hm]; exact hfit⟩
  refine Prog.Runs.spec (MxxCore.Copy.call hcode hc j off s dst src N hauipc hjalr ht hback pre)
    fun t ht' => h' t ?_ fun r m xs hxs hr hap hs => ?_
  · exact holds_copy ht' (holds_setReg _ _ hsrc) hbs dal sal (by omega) (by omega)
  · exact holds_copy_far ht' (holds_setReg _ _ hs) dal (by omega) (by omega)
      (by rw [hxs]; exact hap _ (List.mem_singleton_self _))

/-- `HALT` with the held bytes `bs` at `a` as its payload. -/
theorem halt {P : Prog} {g : Game} {st : g.State} {mem len : Nat} (hP : Sizes P mem len)
    (j a l : Nat) (bs : Bytes) (s : MachineState) (hi : P.instrs[j]? = some .ECALL)
    (h10 : s.getReg .x10 = W a) (h11 : s.getReg .x11 = W l) (h17 : s.getReg .x17 = W 0)
    (h : Holds s a bs) (hbs : bs.length = l) (hnum : BASE ≤ a ∧ a + l ≤ BASE + mem) :
    P.Runs g j s st (l + 1) (pure (.halted bs, st)) := by
  obtain ⟨hm, -, hfit, -⟩ := hP
  have := Prog.Runs.halt (P := P) (g := g) (st := st) (a := a) (len := l) hi h17
    (by rw [h10]; exact W_toNat _ (by omega)) (by rw [h11]; exact W_toNat _ (by omega))
    ⟨hnum.1, by rw [hm]; exact hnum.2⟩
  unfold Holds at h
  subst hbs
  rwa [h] at this

/-! ### Coins -/

/-- `COINS` at instruction `j`: `n` uniform bytes land at `a`. -/
theorem coins {P : Prog} {g : Game} {st : g.State} {mem len : Nat} (hP : Sizes P mem len)
    (j a n : Nat) (s : MachineState) {m : Nat} {Y : Bytes → ProbComp (Outcome × g.State)}
    (hi : P.instrs[j]? = some .ECALL) (h10 : s.getReg .x10 = W a) (h11 : s.getReg .x11 = W n)
    (h17 : s.getReg .x17 = W 2) (hnum : BASE + len ≤ a ∧ a + n ≤ BASE + mem)
    (h' : ∀ bs t, bs.length = n → Holds t a bs → Frame [(a, n)] s t →
      P.Runs g (j + 1) t st m (Y bs)) :
    P.Runs g j s st (m + n + 1) (coinsPC n >>= Y) := by
  obtain ⟨hm, hl, hfit, -⟩ := hP
  refine Prog.Runs.coins (a := a) (len := n) hi h17 (by rw [h10]; exact W_toNat _ (by omega))
    (by rw [h11]; exact W_toNat _ (by omega)) ⟨by rw [hl]; exact hnum.1, by rw [hm]; exact hnum.2⟩
    fun bs hbs => h' bs _ hbs (holds_write _ _ _ (by omega)) fun r k xs hxs hr hap hx =>
      holds_write_far _ _ hx (by omega) (by omega)
        (by have := hap _ (List.mem_singleton_self _); rw [hxs, hbs]; exact this)

/-- `k` draws of `n` uniform bytes each, in order. -/
def coinsSeq (n : Nat) : Nat → ProbComp (List Bytes)
  | 0 => pure []
  | k + 1 => do
    let b ← coinsPC n
    let rest ← coinsSeq n k
    pure (b :: rest)

/-- A counted loop whose iterations each draw `n` uniform bytes: the invariant at the head also
takes the draws so far; after `N` iterations the run continues with all of them. -/
theorem loopCoins {P : Prog} {g : Game} {st : g.State} (top N c n : Nat) {M : Nat}
    (I : Nat → List Bytes → MachineState → Prop) {K : List Bytes → ProbComp (Outcome × g.State)}
    (hbody : ∀ i < N, ∀ (pre : List Bytes) s, pre.length = i → I i pre s →
      ∀ r (Kc : Bytes → ProbComp (Outcome × g.State)),
      (∀ b s', b.length = n → I (i + 1) (pre ++ [b]) s' → P.Runs g top s' st r (Kc b)) →
      P.Runs g top s st (c + r) (coinsPC n >>= Kc))
    (s : MachineState) (h : I 0 [] s)
    (hexit : ∀ bss s, bss.length = N → I N bss s → P.Runs g top s st M (K bss)) :
    P.Runs g top s st (N * c + M) (coinsSeq n N >>= K) := by
  have key : ∀ k i (pre : List Bytes), i + k = N → pre.length = i → ∀ s, I i pre s →
      P.Runs g top s st (k * c + M) (coinsSeq n k >>= fun rest => K (pre ++ rest)) := by
    intro k
    induction k with
    | zero =>
      intro i pre hi hpre s h
      simp only [Nat.add_zero] at hi
      subst hi
      simpa [coinsSeq] using hexit pre s hpre h
    | succ k ih =>
      intro i pre hi hpre s h
      have := hbody i (by omega) pre s hpre h (k * c + M)
        (fun b => coinsSeq n k >>= fun rest => K (pre ++ b :: rest)) fun b s' hb h' => by
          have := ih (i + 1) (pre ++ [b]) (by omega) (by simp [hpre]) s' h'
          simpa using this
      rw [show c + (k * c + M) = (k + 1) * c + M by ring] at this
      simpa [coinsSeq, bind_assoc] using this
  simpa using key N 0 [] (by omega) rfl s h

/-- An environment call the machine does not know (`a7 = 255`, the glue's fault handler)
faults. -/
theorem fault {P : Prog} {g : Game} {st : g.State} {j : Nat} {s : MachineState}
    (hi : P.instrs[j]? = some .ECALL) (h17 : s.getReg .x17 = W 255) :
    P.Runs g j s st 1 (pure (.fault, st)) := by
  intro f hf
  obtain ⟨f, rfl⟩ : ∃ f', f = f' + 1 := ⟨f - 1, by omega⟩
  rw [Machine.run_ecall (Prog.Runs.ecall_mach hi)]
  simp [Machine.handleEcall, Machine.getReg, h17]

/-- Draws of `n` bytes, each making the next carried value by `step`, until `stop` holds of the
carried value and the draw: the carried value then, or `none` when `k` draws did not stop. -/
def untilCoins {α : Type} (n : Nat) (step : α → Bytes → α) (stop : α → Bytes → Bool) :
    Nat → α → ProbComp (Option α)
  | 0, _ => pure none
  | k + 1, x => coinsPC n >>= fun b =>
      if stop x b then pure (some (step x b)) else untilCoins n step stop k (step x b)

/-- A counted loop from its head at `top` that draws `n` bytes per iteration and stops early: an
iteration from a state satisfying `I i x` (the carried value `x`) draws `b` and, within `c` steps,
either goes back to the head satisfying `I (i + 1) (step x b)` or, when `stop x b`, leaves for
`bot` satisfying `E (step x b)` (both continuations take the iteration's result as the body
leaves it, before the case on `stop`); after `N` iterations the head faults within `h` steps.
The run is that of `untilCoins`, then the exit or the fault, within the steps of every
iteration. -/
theorem loopUntilCoins {P : Prog} {g : Game} {st : g.State} (top bot N c h n : Nat) {M : Nat}
    {α : Type} (step : α → Bytes → α) (stop : α → Bytes → Bool)
    (I : Nat → α → MachineState → Prop) (E : α → MachineState → Prop)
    {K : α → ProbComp (Outcome × g.State)}
    (hbody : ∀ i < N, ∀ x s, I i x s → ∀ r (Kc Kx : Bytes → ProbComp (Outcome × g.State)),
      (∀ b s', b.length = n → stop x b = false → I (i + 1) (step x b) s' →
        P.Runs g top s' st r (if stop x b then Kx b else Kc b)) →
      (∀ b s', b.length = n → stop x b = true → E (step x b) s' →
        P.Runs g bot s' st r (if stop x b then Kx b else Kc b)) →
      P.Runs g top s st (c + r) (coinsPC n >>= fun b => if stop x b then Kx b else Kc b))
    (hhead : ∀ x s, I N x s → P.Runs g top s st h (pure (.fault, st)))
    (x : α) (s : MachineState) (hs : I 0 x s)
    (hexit : ∀ x s, E x s → P.Runs g bot s st M (K x)) :
    P.Runs g top s st (N * c + h + M)
      (untilCoins n step stop N x >>= fun o => o.elim (pure (.fault, st)) K) := by
  have key : ∀ k i x s, i + k = N → I i x s →
      P.Runs g top s st (k * c + h + M)
        (untilCoins n step stop k x >>= fun o => o.elim (pure (.fault, st)) K) := by
    intro k
    induction k with
    | zero =>
      intro i x s hi hs
      simp only [Nat.add_zero] at hi
      subst hi
      simpa [untilCoins] using Prog.Runs.mono (hhead x s hs) (by omega)
    | succ k ih =>
      intro i x s hi hs
      have := hbody i (by omega) x s hs (k * c + h + M)
        (fun b => untilCoins n step stop k (step x b) >>= fun o => o.elim (pure (.fault, st)) K)
        (fun b => K (step x b))
        (fun b s' _ hst h' => by
          simp only [hst, Bool.false_eq_true, ↓reduceIte]
          exact ih (i + 1) (step x b) s' (by omega) h')
        (fun b s' _ hst h' => by
          simp only [hst, ↓reduceIte]; exact Prog.Runs.mono (hexit _ s' h') (by omega))
      rw [show c + (k * c + h + M) = (k + 1) * c + h + M by ring] at this
      refine (congrArg _ ?_).mpr this
      simp only [untilCoins, bind_assoc]
      congr 1
      funext b
      split <;> simp
  simpa using key N 0 x s (by omega) hs

/-- Members whose values are a function of the member number and that iteration's draw, with
one more draw. -/
theorem members_snoc {s : MachineState} {a n i : Nat} {pre : List Bytes} {b : Bytes}
    (F : Nat → Bytes → Bytes) (hpre : pre.length = i)
    (h : Members s a n (fun j => F j (pre.getD j [])) i) (hl : (F i b).length = n)
    (hb : Holds s (a + n * i) (F i b)) :
    Members s a n (fun j => F j ((pre ++ [b]).getD j [])) (i + 1) := fun j hj => by
  rcases Nat.lt_succ_iff_lt_or_eq.1 hj with hj | rfl
  · have e : (pre ++ [b]).getD j [] = pre.getD j [] := by
      simp [List.getD_eq_getElem?_getD, List.getElem?_append_left (show j < pre.length by omega)]
    show (F j ((pre ++ [b]).getD j [])).length = n ∧
      Holds s (a + n * j) (F j ((pre ++ [b]).getD j []))
    rw [e]; exact h j hj
  · have e : (pre ++ [b]).getD j [] = b := by
      simp [List.getD_eq_getElem?_getD, hpre]
    show (F j ((pre ++ [b]).getD j [])).length = n ∧
      Holds s (a + n * j) (F j ((pre ++ [b]).getD j []))
    rw [e]; exact ⟨hl, hb⟩

end MxxCore.Compose
