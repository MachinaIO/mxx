import Claim
import ExponentLut
import ExponentDecode
import ExponentRuntime
import ExponentDynamic
import ExponentFunctional
import PeDynamic
import PeCompact

/-!
# The parameter set

The ring `Z_Q[X]/(X^{{N}} + 1)` with {{DEPTH}} {{CRT_BITS}}-bit CRT primes (`Q ≈ 2^{{QBITS}}`), gadget base `2^{{BASE_BITS}}` with
two digits per prime, so a gadget row has `D = {{D}}` columns and an encoding `m = {{M}}`. The setup
automorphism group is `H = {1, {{K}}}`, and `{{K}}` is its own inverse modulo `2N`. Every Dynamic
evaluation stays on this one ring.

Encryption publishes its fresh encodings compactly, over `R_{PQ}` for the auxiliary modulus `P` of
{{DEPTH}} further primes: a compact key is a `2 × 2` matrix of single entries hashed at `PQ` under the
tag of its full key, and every reader expands it to the full key
`ModSwitch_{PQ → Q}(Ã G̃)` (`expandKey`) for the integer lift `G̃` of the gadget.

The public matrices are expanded from the hash key under tags that name the setup schema
(`schemaBytes`) or the parameters (`peContextBytes`).
-/

namespace PeProof

open Mxx.Primitives MxxRuntime MxxExponentLut

abbrev N : Nat := {{N}}
abbrev Q : Nat := {{Q}}

/-- The auxiliary modulus of the compact encodings. -/
abbrev P : Nat := {{P}}

/-- The modulus of the compact encodings. -/
abbrev PQ : Nat := {{PQ}}

theorem hN : 0 < N := by decide
theorem hN1 : 1 < N := by decide
theorem hpow : N = 2 ^ {{LOGN}} := rfl
theorem hQ : 1 < Q := by norm_num
theorem hQodd : Q % 2 = 1 := by norm_num
theorem hP : 1 < P := by norm_num
theorem hPodd : P % 2 = 1 := by norm_num
theorem hPQ : PQ = P * Q := by norm_num

theorem layoutQ : Backend.backend.regularLayout Q N = some Backend.layout0 := by
  simp [Backend.backend]

/-- The nonidentity automorphism of the setup group. -/
abbrev K : Nat := {{K}}

theorem hK : Odd K := by decide
theorem hKinv : K * K % (2 * N) = 1 := by decide
theorem hKcoprime : N.Coprime K := by decide

/-! ## Hash tags -/

/-- `component` -/
def componentBytes : Blob := [99, 111, 109, 112, 111, 110, 101, 110, 116]

/-- `rns-bgg/dynamic` -/
def dynamicPrefix : Blob := [114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]

/-- The hex digest of the Dynamic setup schema. -/
def schemaBytes : Blob := {{SCHEMA_BYTES}}

/-- `mxx-pe` -/
def pePrefix : Blob := [109, 120, 120, 45, 112, 101]

/-- The hex digest of the setup context, which names the parameters. -/
def peContextBytes : Blob := {{PECTX_BYTES}}

/-- `matrixMul` is the matrix product, for every ring dimension: rewriting with it keeps the kernel
from unfolding the instances at the concrete dimension. -/
theorem matrixMul_eq_mul {q n r k c : Nat} (a : ExactMatrix q n r k) (b : ExactMatrix q n k c) :
    matrixMul a b = a * b := rfl

/-! ## The gadget and the expansion -/

/-- The gadget row. -/
noncomputable def g0 : ExactMatrix Q N 1 {{D}} :=
  Classical.choose (gadget_exists (m := {{D}}) (base := {{BASE}}) layoutQ rfl (by decide) (by decide))

theorem g0_runs : gadgetMatrixRuns Backend.backend {{BASE}} {{D}} g0 :=
  Classical.choose_spec (gadget_exists (m := {{D}}) (base := {{BASE}}) layoutQ rfl (by decide) (by decide))

theorem gadget_lift_exists : ∃ z : Fin {{D}} → Int, g0 = MxxPe.constRow Q z ∧
    centeredRebase (p := PQ) g0 = MxxPe.constRow PQ z ∧ ∀ c, (z c).natAbs ≤ Q / 2 := by
  obtain ⟨w, hw⟩ := MxxPe.gadgetMatrixRuns_const g0_runs
  exact MxxPe.centeredRebase_constRow hN hQ hw

/-- The integer constants of the gadget, each at most `⌊Q / 2⌋`. -/
noncomputable def gz : Fin {{D}} → Int := Classical.choose gadget_lift_exists

theorem g0_eq : g0 = MxxPe.constRow Q gz := (Classical.choose_spec gadget_lift_exists).1

theorem gtilde_eq : centeredRebase (p := PQ) g0 = MxxPe.constRow PQ gz :=
  (Classical.choose_spec gadget_lift_exists).2.1

theorem gz_bound (c : Fin {{D}}) : (gz c).natAbs ≤ Q / 2 :=
  (Classical.choose_spec gadget_lift_exists).2.2 c

/-- The integer lift `G̃` of the gadget at `PQ`. -/
noncomputable abbrev Gt : ExactMatrix PQ N 1 {{D}} := MxxPe.constRow PQ gz

theorem expand_exists (x : ExactMatrix PQ N 1 1) :
    ∃ y : ExactMatrix Q N 1 {{D}}, y = modulusSwitch (matrixMulScalarLeft x Gt) := ⟨_, rfl⟩

/-- The expansion `ModSwitch_{PQ → Q}(x G̃)` of a single entry over `R_{PQ}`. It is chosen, so that
neither the elaborator nor the kernel unfolds it into its sum over every coefficient; `expand_def`
states it. -/
noncomputable def expand (x : ExactMatrix PQ N 1 1) : ExactMatrix Q N 1 {{D}} :=
  Classical.choose (expand_exists x)

theorem expand_def (x : ExactMatrix PQ N 1 1) : expand x = modulusSwitch (matrixMulScalarLeft x Gt) :=
  Classical.choose_spec (expand_exists x)

theorem expand_runs (x : ExactMatrix PQ N 1 1) : modulusSwitchRuns (matrixMulScalarLeft x Gt) (expand x) :=
  ⟨hQ, ⟨P, by rw [hPQ, Nat.mul_comm]⟩, by norm_num, hQodd, expand_def x⟩

/-- A runtime switch of `x G̃` is the expansion. -/
theorem expand_eq {x : ExactMatrix PQ N 1 1} {y : ExactMatrix Q N 1 {{D}}}
    (h : modulusSwitchRuns (matrixMulScalarLeft x Gt) y) : y = expand x :=
  h.2.2.2.2.trans (expand_def x).symm

/-- A runtime expansion: the gadget, its rebase to `PQ`, and the switch of `x G̃`. -/
theorem expand_read {x : ExactMatrix PQ N 1 1} {g : ExactMatrix Q N 1 {{D}}}
    {gt : ExactMatrix PQ N 1 {{D}}} {y : ExactMatrix Q N 1 {{D}}}
    (hg : gadgetMatrixRuns Backend.backend {{BASE}} {{D}} g) (ht : centeredRebaseRuns g gt)
    (hy : modulusSwitchRuns (matrixMulScalarLeft x gt) y) : y = expand x := by
  have hgt : gt = Gt := by
    rw [ht.2, gadgetMatrixRuns_unique hg g0_runs]
    exact gtilde_eq
  subst hgt
  exact expand_eq hy

/-- A runtime expansion through the state gadget: the gadget, its rebase to `Q` (the identity),
its rebase to `PQ`, and the switch of `x G̃`. -/
theorem expand_read2 {x : ExactMatrix PQ N 1 1} {g g' : ExactMatrix Q N 1 {{D}}}
    {gt : ExactMatrix PQ N 1 {{D}}} {y : ExactMatrix Q N 1 {{D}}}
    (hg : gadgetMatrixRuns Backend.backend {{BASE}} {{D}} g) (hself : centeredRebaseRuns g g')
    (ht : centeredRebaseRuns g' gt) (hy : modulusSwitchRuns (matrixMulScalarLeft x gt) y) :
    y = expand x := by
  have hg' : g' = g := centeredRebaseRuns_unique hself (centeredRebaseRuns_self hQ hN g)
  subst hg'
  exact expand_read hg ht hy

/-- The expansion of a compact key. -/
noncomputable def expandKey (A : Fin 2 → Fin 2 → ExactMatrix PQ N 1 1) :
    Fin 2 → Fin 2 → ExactMatrix Q N 1 {{D}} :=
  fun k j ↦ expand (A k j)

/-- The error of the expansion of a compact encoding (`MxxPe.expandErr`). -/
noncomputable def expandErrC (sm : ErrorPoly N) (At : Fin 2 → Fin 2 → ExactMatrix PQ N 1 1)
    (ct : Fin 2 → ExactMatrix PQ N 1 1) (e : Fin 2 → ErrorMatrix N 1 1) : Fin 2 → ErrorMatrix N 1 {{D}} :=
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

/-- The compact public key of the initial state. -/
noncomputable def inputKeyC (model : HashModel) (key : ByteArray) : Fin 2 → Fin 2 → ExactMatrix PQ N 1 1 :=
  fun k j ↦ model.sample PQ N 1 1 key (completeHashTag dynamicPrefix
    [.bytes schemaBytes, .bytes [105, 110, 112, 117, 116, 45, 107, 101, 121], .u64Le 0,
      .bytes componentBytes, .u64Le (2 * (k : Int) + (j : Int))])

/-- The public key `A_0` of the initial state: the expansion of its compact key. -/
noncomputable def inputKey (model : HashModel) (key : ByteArray) : Fin 2 → Fin 2 → ExactMatrix Q N 1 {{D}} :=
  expandKey (inputKeyC model key)

/-- The compact helper key of `K`. -/
noncomputable def helperKeyC (model : HashModel) (key : ByteArray) : Fin 2 → Fin 2 → ExactMatrix PQ N 1 1 :=
  fun k j ↦ model.sample PQ N 1 1 key (completeHashTag dynamicPrefix
    [.bytes schemaBytes, .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le (K : Int),
      .bytes componentBytes, .u64Le (2 * (k : Int) + (j : Int))])

/-- The helper key `H_K`: the expansion of its compact key. -/
noncomputable def helperKey (model : HashModel) (key : ByteArray) : Fin 2 → Fin 2 → ExactMatrix Q N 1 {{D}} :=
  expandKey (helperKeyC model key)

/-- The message row `d = (d_1, d_2)`. -/
noncomputable def messageRow (model : HashModel) (key : ByteArray) (index : Fin 2) : ExactMatrix Q N 1 1 :=
  model.sample Q N 1 1 key (completeHashTag pePrefix
    [.bytes peContextBytes, .bytes [109, 101, 115, 115, 97, 103, 101, 45, 114, 111, 119],
      .u64Le (index : Int)])

/-- The second anchor row `B^(2)`. -/
noncomputable def anchorSecond (model : HashModel) (key : ByteArray) : ExactMatrix Q N 1 {{MB}} :=
  model.sample Q N 1 {{MB}} key (completeHashTag pePrefix
    [.bytes peContextBytes, .bytes [97, 110, 99, 104, 111, 114, 45, 115, 101, 99, 111, 110, 100, 45,
      114, 111, 119]])

/-- The digits of entry `a` of the joined key row `[A_r0 | A_r1]`. -/
noncomputable def keyDigits (A : Fin 2 → Fin 2 → ExactMatrix Q N 1 {{D}}) (a : Fin ({{D}} + {{D}})) (r : Fin 2) :
    ExactMatrix Q N {{D}} 1 :=
  entryDigits {{D}} Backend.backend {{BASE}} {{D}} (fun _ _ ↦ joinedEntry (A r) a)

end PeProof
