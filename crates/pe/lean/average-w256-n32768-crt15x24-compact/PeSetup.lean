import Claim
import ExponentLut
import ExponentDecode
import ExponentRuntime
import ExponentDynamic
import ExponentFunctional
import ExponentWalsh
import PeDynamic
import PeCompact

/-!
# The date parameter set

The ring `Z_Q[X]/(X^32768 + 1)` uses 15 24-bit CRT primes (`log2 Q ≈ 354.97`).
The gadget base is `2^12`, with two digits per limb, so `D = 30` and the joined row has
60 columns. The lookup group has order 256 and a two-level radix-16 plan: 15 internal
branches, 15 first-depth leaves, and 225 second-depth leaves. Each leaf member shares
its companion decompositions across groups of at most four coefficient slots.

Compact companions use `R_PQ`, with 15 extension primes (`log2 P ≈ 338.85`).
The statistical TrapGen anchor has 265 sampled columns plus the 30 gadget columns.
-/

set_option maxRecDepth 200000

namespace PeProof

open Mxx.Primitives MxxRuntime MxxExponentLut

abbrev N : Nat := 32768
abbrev Q : Nat := 71693316984649896843904253443489203902220097476285449024776635275984659786482860843809654567991896462655489

/-- The auxiliary modulus of the compressed digit encodings. -/
abbrev P : Nat := 1009041927918404748912066953523025233623263629931203392403660676397045184266783237099217072052453834753

/-- The modulus of the compressed digit encodings and of the anchor. -/
abbrev PQ : Nat := 72341562789056444115684979687416806180395197053951206217914228266014512572833794748552875371413453242085860536342419038795864559652387809846630825946845083145910068784615984026523621685186503403424255574409217

/-- `W⁻¹ mod Q` for the lookup width `W = 256`. -/
abbrev Winv : Nat := 71413264965178608184357752453475574199477050220518708989523601544437844709191912168638523104835678117098241

theorem hN : 0 < N := by decide
theorem hN1 : 1 < N := by decide
theorem hpow : N = 2 ^ 15 := rfl
theorem hQ : 1 < Q := by norm_num
theorem hQodd : Q % 2 = 1 := by norm_num
theorem hP : 1 < P := by norm_num
theorem hPodd : P % 2 = 1 := by norm_num
theorem hPQ : PQ = P * Q := by norm_num
theorem hWinv : 256 * Winv % Q = 1 := by norm_num

theorem layoutQ : Backend.backend.regularLayout Q N = some Backend.layout0 := by
  simp [Backend.backend]

/-- `matrixMul` is the matrix product, for every ring dimension: rewriting with it keeps the kernel
from unfolding the instances at the concrete dimension. -/
theorem matrixMul_eq_mul {q n r k c : Nat} (a : ExactMatrix q n r k) (b : ExactMatrix q n k c) :
    matrixMul a b = a * b := rfl

/-! ## The gadget and the expansion -/

/-- The gadget row. -/
noncomputable def g0 : ExactMatrix Q N 1 30 :=
  Classical.choose (gadget_exists (m := 30) (base := 4096) layoutQ rfl (by decide) (by decide))

theorem g0_runs : gadgetMatrixRuns Backend.backend 4096 30 g0 :=
  Classical.choose_spec (gadget_exists (m := 30) (base := 4096) layoutQ rfl (by decide) (by decide))

theorem gadget_lift_exists : ∃ z : Fin 30 → Int, g0 = MxxPe.constRow Q z ∧
    centeredRebase (p := PQ) g0 = MxxPe.constRow PQ z ∧ ∀ c, (z c).natAbs ≤ Q / 2 := by
  obtain ⟨w, hw⟩ := MxxPe.gadgetMatrixRuns_const g0_runs
  exact MxxPe.centeredRebase_constRow hN hQ hw

/-- The integer constants of the gadget, each at most `⌊Q / 2⌋`. -/
noncomputable def gz : Fin 30 → Int := Classical.choose gadget_lift_exists

theorem g0_eq : g0 = MxxPe.constRow Q gz := (Classical.choose_spec gadget_lift_exists).1

theorem gtilde_eq : centeredRebase (p := PQ) g0 = MxxPe.constRow PQ gz :=
  (Classical.choose_spec gadget_lift_exists).2.1

theorem gz_bound (c : Fin 30) : (gz c).natAbs ≤ Q / 2 :=
  (Classical.choose_spec gadget_lift_exists).2.2 c

/-- The integer lift `G̃` of the gadget at `PQ`. -/
noncomputable abbrev Gt : ExactMatrix PQ N 1 30 := MxxPe.constRow PQ gz

theorem expand_exists (x : ExactMatrix PQ N 1 1) :
    ∃ y : ExactMatrix Q N 1 30, y = modulusSwitch (matrixMulScalarLeft x Gt) := ⟨_, rfl⟩

/-- The expansion `ModSwitch_{PQ → Q}(x G̃)` of a single entry over `R_{PQ}`. It is chosen, so that
neither the elaborator nor the kernel unfolds it into its sum over every coefficient; `expand_def`
states it. -/
noncomputable def expand (x : ExactMatrix PQ N 1 1) : ExactMatrix Q N 1 30 :=
  Classical.choose (expand_exists x)

theorem expand_def (x : ExactMatrix PQ N 1 1) : expand x = modulusSwitch (matrixMulScalarLeft x Gt) :=
  Classical.choose_spec (expand_exists x)

theorem expand_runs (x : ExactMatrix PQ N 1 1) : modulusSwitchRuns (matrixMulScalarLeft x Gt) (expand x) :=
  ⟨hQ, ⟨P, by rw [hPQ, Nat.mul_comm]⟩, by norm_num, hQodd, expand_def x⟩

/-- A runtime switch of `x G̃` is the expansion. -/
theorem expand_eq {x : ExactMatrix PQ N 1 1} {y : ExactMatrix Q N 1 30}
    (h : modulusSwitchRuns (matrixMulScalarLeft x Gt) y) : y = expand x :=
  h.2.2.2.2.trans (expand_def x).symm

/-- A runtime expansion: the gadget, its rebase to `PQ`, and the switch of `x G̃`. -/
theorem expand_read {x : ExactMatrix PQ N 1 1} {g : ExactMatrix Q N 1 30}
    {gt : ExactMatrix PQ N 1 30} {y : ExactMatrix Q N 1 30}
    (hg : gadgetMatrixRuns Backend.backend 4096 30 g) (ht : centeredRebaseRuns g gt)
    (hy : modulusSwitchRuns (matrixMulScalarLeft x gt) y) : y = expand x := by
  have hgt : gt = Gt := by
    rw [ht.2, gadgetMatrixRuns_unique hg g0_runs]
    exact gtilde_eq
  subst hgt
  exact expand_eq hy

/-- A runtime expansion through the state gadget: the gadget, its rebase to `Q` (the identity),
its rebase to `PQ`, and the switch of `x G̃`. -/
theorem expand_read2 {x : ExactMatrix PQ N 1 1} {g g' : ExactMatrix Q N 1 30}
    {gt : ExactMatrix PQ N 1 30} {y : ExactMatrix Q N 1 30}
    (hg : gadgetMatrixRuns Backend.backend 4096 30 g) (hself : centeredRebaseRuns g g')
    (ht : centeredRebaseRuns g' gt) (hy : modulusSwitchRuns (matrixMulScalarLeft x gt) y) :
    y = expand x := by
  have hg' : g' = g := centeredRebaseRuns_unique hself (centeredRebaseRuns_self hQ hN g)
  subst hg'
  exact expand_read hg ht hy

/-- The expansion of a compact key. -/
noncomputable def expandKey (A : Fin 2 → Fin 2 → ExactMatrix PQ N 1 1) :
    Fin 2 → Fin 2 → ExactMatrix Q N 1 30 :=
  fun k j ↦ expand (A k j)

/-- The error of the expansion of a compact encoding (`MxxPe.expandErr`). -/
noncomputable def expandErrC (sm : ErrorPoly N) (At : Fin 2 → Fin 2 → ExactMatrix PQ N 1 1)
    (ct : Fin 2 → ExactMatrix PQ N 1 1) (e : Fin 2 → ErrorMatrix N 1 1) : Fin 2 → ErrorMatrix N 1 30 :=
  MxxPe.expandErr P Q sm gz At ct e

/-- The expansion of a compact encoding of `P x_j` under `Ã` is an encoding of `x_j g` under the
expanded key, with the error `expandErrC` within `(B ⌊Q / 2⌋ + (N + 2) ⌊P / 2⌋) / P`
(`MxxPe.RelE.expandCompact`). -/
theorem expand_rel {sm : ErrorPoly N} (hsm : polyNorm sm ≤ 1) {At : Fin 2 → Fin 2 → ExactMatrix PQ N 1 1}
    {x : Fin 2 → ErrorPoly N} {ct : Fin 2 → ExactMatrix PQ N 1 1} {e : Fin 2 → ErrorMatrix N 1 1}
    {B : Nat} (he : ∀ j, CoeffBound (e j) B)
    (hc : RelE sm At (fun j _ _ ↦ reducePoly PQ N (((P : Nat) : Int) * x j)) ct e) :
    RelE sm (expandKey At) (fun j ↦ reducePoly Q N (x j) • g0) (fun j ↦ expand (ct j))
        (expandErrC sm At ct e) ∧
      ∀ j, CoeffBound (expandErrC sm At ct e j) ((B * (Q / 2) + (N + 2) * (P / 2)) / P) := by
  have h := MxxPe.RelE.expandCompact (p := P) (q := Q) hN hP hPodd hQ hsm he hc gz_bound
    (fun k j ↦ expand_runs (At k j)) (fun j ↦ expand_runs (ct j))
  rw [← g0_eq] at h
  exact h

/-- The digits of entry `a` of the joined key row `[A_r0 | A_r1]`. -/
noncomputable def keyDigits (A : Fin 2 → Fin 2 → ExactMatrix Q N 1 30) (a : Fin (30 + 30)) (r : Fin 2) :
    ExactMatrix Q N 30 1 :=
  entryDigits 30 Backend.backend 4096 30 (fun _ _ ↦ joinedEntry (A r) a)

/-! ## Hash tags -/

/-- `component` -/
def componentBytes : Blob := [99, 111, 109, 112, 111, 110, 101, 110, 116]

/-- `rns-bgg/dynamic` -/
def dynamicPrefix : Blob := [114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]

/-- The hex digest of the Dynamic setup schema. -/
def schemaBytes : Blob := [55, 51, 50, 48, 54, 98, 48, 49, 52, 57, 53, 52, 99, 53, 53, 100, 101, 48, 99, 53, 52, 55, 54, 57, 100, 102, 98, 56, 50, 52, 51, 53, 98, 52, 101, 54, 100, 98, 100, 55, 49, 50, 53, 102, 54, 50, 53, 53, 50, 52, 50, 56, 53, 52, 49, 102, 102, 51, 50, 57, 48, 49, 51, 52]

/-- `helper-key` -/
def helperKeyBytes : Blob := [104, 101, 108, 112, 101, 114, 45, 107, 101, 121]

/-- The public helper key `H_k` of automorphism `k`: four `1 × 30` blocks hashed under the setup
schema. -/
noncomputable def helperKey (model : HashModel) (key : ByteArray) (k : Int) :
    Fin 2 → Fin 2 → ExactMatrix Q N 1 30 :=
  fun r j ↦ model.sample Q N 1 30 key (completeHashTag dynamicPrefix
    [.bytes schemaBytes, .bytes helperKeyBytes, .u64Le k, .bytes componentBytes,
      .u64Le (2 * (r : Int) + (j : Int))])

end PeProof
