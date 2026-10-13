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
# The parameter set

The ring `Z_Q[X]/(X^65536 + 1)` with 22 32-bit CRT primes (`Q ≈ 2^704`), gadget base `2^16` with
two digits per prime, so a gadget row has `D = 44` columns and an encoding `m = 88`. The lookup
group `H` has order `W = 128`; its LUTs are evaluated in radix 16, the last round in radix 8.

The digit companions are published compactly over `R_PQ` for the auxiliary modulus `P` of 22
further primes, and the anchor is defined modulo `PQ` (the manuscript's Appendix C).
-/

namespace PeProof

open Mxx.Primitives MxxRuntime MxxExponentLut

abbrev N : Nat := 65536
abbrev Q : Nat := 78079523927896127712009307999974412043135143141543941570119110821624764686494582648421232630165133436275519032254680972082541026053657846674805178959586229420520814546267827301225593318349477307602951695899164673

/-- The auxiliary modulus of the compressed digit encodings. -/
abbrev P : Nat := 67598865718973452911259282735194974463966008168342023559927392187609292957345997883634735633837025843140326117569327049993577673305918340947654355600136203031252562778990577685529163020701990351591297285624954881

/-- The modulus of the compressed digit encodings and of the anchor. -/
abbrev PQ : Nat := 5278087253403224991333645099610003021785206966119564254171942298247993190952980821552692773158302798199939315804484105262715072792529872674621368544490941390506983498620144494214222200022752952511718466872729239793477806438096725442738627984346925320228738380148171996792888816066148390509478988805065937247347127262069050805334354391385613933775075315634469785233518480876460832181155068060520232362595000272223451214118913

/-- `W⁻¹ mod Q` for the lookup width `W = 128`. -/
abbrev Winv : Nat := 77469527647209439214259235281224611949048149835750629526602555268330821212381343721480441750241968331304616539815191276988146174287613644747658263498964462003172995682625109900434768370549872016137303635774952449

theorem hN : 0 < N := by decide
theorem hN1 : 1 < N := by decide
theorem hpow : N = 2 ^ 16 := rfl
theorem hQ : 1 < Q := by norm_num
theorem hQodd : Q % 2 = 1 := by norm_num
theorem hP : 1 < P := by norm_num
theorem hPodd : P % 2 = 1 := by norm_num
theorem hPQ : PQ = P * Q := by norm_num
theorem hWinv : 128 * Winv % Q = 1 := by norm_num

theorem layoutQ : Backend.backend.regularLayout Q N = some Backend.layout0 := by
  simp [Backend.backend]

/-- `matrixMul` is the matrix product, for every ring dimension: rewriting with it keeps the kernel
from unfolding the instances at the concrete dimension. -/
theorem matrixMul_eq_mul {q n r k c : Nat} (a : ExactMatrix q n r k) (b : ExactMatrix q n k c) :
    matrixMul a b = a * b := rfl

/-! ## The gadget and the expansion -/

/-- The gadget row. -/
noncomputable def g0 : ExactMatrix Q N 1 44 :=
  Classical.choose (gadget_exists (m := 44) (base := 65536) layoutQ rfl (by decide) (by decide))

theorem g0_runs : gadgetMatrixRuns Backend.backend 65536 44 g0 :=
  Classical.choose_spec (gadget_exists (m := 44) (base := 65536) layoutQ rfl (by decide) (by decide))

theorem gadget_lift_exists : ∃ z : Fin 44 → Int, g0 = MxxPe.constRow Q z ∧
    centeredRebase (p := PQ) g0 = MxxPe.constRow PQ z ∧ ∀ c, (z c).natAbs ≤ Q / 2 := by
  obtain ⟨w, hw⟩ := MxxPe.gadgetMatrixRuns_const g0_runs
  exact MxxPe.centeredRebase_constRow hN hQ hw

/-- The integer constants of the gadget, each at most `⌊Q / 2⌋`. -/
noncomputable def gz : Fin 44 → Int := Classical.choose gadget_lift_exists

theorem g0_eq : g0 = MxxPe.constRow Q gz := (Classical.choose_spec gadget_lift_exists).1

theorem gtilde_eq : centeredRebase (p := PQ) g0 = MxxPe.constRow PQ gz :=
  (Classical.choose_spec gadget_lift_exists).2.1

theorem gz_bound (c : Fin 44) : (gz c).natAbs ≤ Q / 2 :=
  (Classical.choose_spec gadget_lift_exists).2.2 c

/-- The integer lift `G̃` of the gadget at `PQ`. -/
noncomputable abbrev Gt : ExactMatrix PQ N 1 44 := MxxPe.constRow PQ gz

theorem expand_exists (x : ExactMatrix PQ N 1 1) :
    ∃ y : ExactMatrix Q N 1 44, y = modulusSwitch (matrixMulScalarLeft x Gt) := ⟨_, rfl⟩

/-- The expansion `ModSwitch_{PQ → Q}(x G̃)` of a single entry over `R_{PQ}`. It is chosen, so that
neither the elaborator nor the kernel unfolds it into its sum over every coefficient; `expand_def`
states it. -/
noncomputable def expand (x : ExactMatrix PQ N 1 1) : ExactMatrix Q N 1 44 :=
  Classical.choose (expand_exists x)

theorem expand_def (x : ExactMatrix PQ N 1 1) : expand x = modulusSwitch (matrixMulScalarLeft x Gt) :=
  Classical.choose_spec (expand_exists x)

theorem expand_runs (x : ExactMatrix PQ N 1 1) : modulusSwitchRuns (matrixMulScalarLeft x Gt) (expand x) :=
  ⟨hQ, ⟨P, by rw [hPQ, Nat.mul_comm]⟩, by norm_num, hQodd, expand_def x⟩

/-- A runtime switch of `x G̃` is the expansion. -/
theorem expand_eq {x : ExactMatrix PQ N 1 1} {y : ExactMatrix Q N 1 44}
    (h : modulusSwitchRuns (matrixMulScalarLeft x Gt) y) : y = expand x :=
  h.2.2.2.2.trans (expand_def x).symm

/-- A runtime expansion: the gadget, its rebase to `PQ`, and the switch of `x G̃`. -/
theorem expand_read {x : ExactMatrix PQ N 1 1} {g : ExactMatrix Q N 1 44}
    {gt : ExactMatrix PQ N 1 44} {y : ExactMatrix Q N 1 44}
    (hg : gadgetMatrixRuns Backend.backend 65536 44 g) (ht : centeredRebaseRuns g gt)
    (hy : modulusSwitchRuns (matrixMulScalarLeft x gt) y) : y = expand x := by
  have hgt : gt = Gt := by
    rw [ht.2, gadgetMatrixRuns_unique hg g0_runs]
    exact gtilde_eq
  subst hgt
  exact expand_eq hy

/-- A runtime expansion through the state gadget: the gadget, its rebase to `Q` (the identity),
its rebase to `PQ`, and the switch of `x G̃`. -/
theorem expand_read2 {x : ExactMatrix PQ N 1 1} {g g' : ExactMatrix Q N 1 44}
    {gt : ExactMatrix PQ N 1 44} {y : ExactMatrix Q N 1 44}
    (hg : gadgetMatrixRuns Backend.backend 65536 44 g) (hself : centeredRebaseRuns g g')
    (ht : centeredRebaseRuns g' gt) (hy : modulusSwitchRuns (matrixMulScalarLeft x gt) y) :
    y = expand x := by
  have hg' : g' = g := centeredRebaseRuns_unique hself (centeredRebaseRuns_self hQ hN g)
  subst hg'
  exact expand_read hg ht hy

/-- The expansion of a compact key. -/
noncomputable def expandKey (A : Fin 2 → Fin 2 → ExactMatrix PQ N 1 1) :
    Fin 2 → Fin 2 → ExactMatrix Q N 1 44 :=
  fun k j ↦ expand (A k j)

/-- The error of the expansion of a compact encoding (`MxxPe.expandErr`). -/
noncomputable def expandErrC (sm : ErrorPoly N) (At : Fin 2 → Fin 2 → ExactMatrix PQ N 1 1)
    (ct : Fin 2 → ExactMatrix PQ N 1 1) (e : Fin 2 → ErrorMatrix N 1 1) : Fin 2 → ErrorMatrix N 1 44 :=
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
noncomputable def keyDigits (A : Fin 2 → Fin 2 → ExactMatrix Q N 1 44) (a : Fin (44 + 44)) (r : Fin 2) :
    ExactMatrix Q N 44 1 :=
  entryDigits 44 Backend.backend 65536 44 (fun _ _ ↦ joinedEntry (A r) a)

/-! ## Hash tags -/

/-- `component` -/
def componentBytes : Blob := [99, 111, 109, 112, 111, 110, 101, 110, 116]

/-- `rns-bgg/dynamic` -/
def dynamicPrefix : Blob := [114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]

/-- The hex digest of the Dynamic setup schema. -/
def schemaBytes : Blob := [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49]

/-- `helper-key` -/
def helperKeyBytes : Blob := [104, 101, 108, 112, 101, 114, 45, 107, 101, 121]

/-- The public helper key `H_k` of automorphism `k`: four `1 × 44` blocks hashed under the setup
schema. -/
noncomputable def helperKey (model : HashModel) (key : ByteArray) (k : Int) :
    Fin 2 → Fin 2 → ExactMatrix Q N 1 44 :=
  fun r j ↦ model.sample Q N 1 44 key (completeHashTag dynamicPrefix
    [.bytes schemaBytes, .bytes helperKeyBytes, .u64Le k, .bytes componentBytes,
      .u64Le (2 * (r : Int) + (j : Int))])

end PeProof
