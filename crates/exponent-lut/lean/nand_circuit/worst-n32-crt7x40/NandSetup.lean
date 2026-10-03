import Claim
import ExponentLut
import ExponentDecode
import ExponentWalsh
import NandProgram

/-!
# The parameter set

Moduli, dropped primes, gadget layouts, and hash tags of the generated NAND claim at this parameter
set. Levels `0` to `8` use the moduli `Q7, Q7, Q6, Q6, Q5, Q5, Q4, Q4, Q3` (the index counts CRT
primes): private multiplications keep the modulus and every LUT drops one prime `P_k = Q_k / Q_{k-1}`.
-/

namespace NandProof

open Mxx.Primitives MxxRuntime MxxExponentLut

abbrev N : Nat := 32
abbrev Q7 : Nat := 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921
abbrev Q6 : Nat := 1766847023753256444127058376275403912430991675449840265546213289008802881
abbrev Q5 : Nat := 1606938016486074846153923224048467814823508072370602761938369
abbrev Q4 : Nat := 1461501619046042689013583379439969400432371472961
abbrev Q3 : Nat := 1329227985420794846003957960673218177
abbrev P7 : Nat := 1099511619841
abbrev P6 : Nat := 1099511621249
abbrev P5 : Nat := 1099511622529
abbrev P4 : Nat := 1099511622593

theorem hN : 0 < N := by decide
theorem hpow : N = 2 ^ 5 := rfl
theorem q7_eq : Q7 = P7 * Q6 := by norm_num
theorem q6_eq : Q6 = P6 * Q5 := by norm_num
theorem q5_eq : Q5 = P5 * Q4 := by norm_num
theorem q4_eq : Q4 = P4 * Q3 := by norm_num
theorem hQ7 : 1 < Q7 := by norm_num
theorem hQ6 : 1 < Q6 := by norm_num
theorem hQ5 : 1 < Q5 := by norm_num
theorem hQ4 : 1 < Q4 := by norm_num
theorem hQ3 : 1 < Q3 := by norm_num

theorem layout7 : Backend.backend.regularLayout Q7 N = some Backend.layout4 := by
  simp [Backend.backend]
theorem layout6 : Backend.backend.regularLayout Q6 N = some Backend.layout3 := by
  simp [Backend.backend]
theorem layout5 : Backend.backend.regularLayout Q5 N = some Backend.layout2 := by
  simp [Backend.backend]
theorem layout4 : Backend.backend.regularLayout Q4 N = some Backend.layout1 := by
  simp [Backend.backend]
theorem layout3 : Backend.backend.regularLayout Q3 N = some Backend.layout0 := by
  simp [Backend.backend]

/-! ## Relations of the generated graphs as equations -/

theorem hashSample_eq {q rows columns : Nat} {model : HashModel} {tagPrefix : Blob}
    {components : List HashTagComponent} {key : ByteArray}
    {output : ExactMatrix q N rows columns} (h : hashSample model tagPrefix components key output) :
    output = model.sample q N rows columns key (completeHashTag tagPrefix components) := h.2.2

theorem matrixMulScalarLeft_eq {q rows columns : Nat} (a : ExactMatrix q N 1 1)
    (x : ExactMatrix q N rows columns) : matrixMulScalarLeft a x = a 0 0 • x := rfl

theorem matrixMulScalarRight_eq {q rows columns : Nat} (x : ExactMatrix q N rows columns)
    (a : ExactMatrix q N 1 1) : matrixMulScalarRight x a = a 0 0 • x := by
  funext i j
  simp [matrixMulScalarRight, mul_comm]

/-- Reduction of a reduced integer polynomial to a divisor modulus. -/
theorem modulusReduce_poly {q p : Nat} (hq : 1 < q) (hp : 1 < p) {s : ErrorPoly N}
    {out : ExactMatrix p N 1 1} (h : modulusReduceRuns (fun _ _ ↦ reducePoly q N s : ExactMatrix q N 1 1) out) :
    out = fun _ _ ↦ reducePoly p N s :=
  modulusReduceRuns_reduce hq hp hN h (fun _ _ ↦ s) rfl

theorem modulusReduce_one {q p : Nat} (hq : 1 < q) (hp : 1 < p) {out : ExactMatrix p N 1 1}
    (h : modulusReduceRuns (1 : ExactMatrix q N 1 1) out) : out = 1 := by
  have hone : (1 : ExactMatrix q N 1 1) = fun _ _ ↦ reducePoly q N 1 := by
    funext i j
    rw [Subsingleton.elim i 0, Subsingleton.elim j 0]
    simp
  rw [hone] at h
  rw [modulusReduce_poly hq hp h]
  funext i j
  rw [Subsingleton.elim i 0, Subsingleton.elim j 0]
  simp

/-- A ternary sample as one integer polynomial. -/
theorem ternary_poly {q : Nat} {output : ExactMatrix q N 1 1}
    (h : uniformIntervalSample (-1) 1 output) :
    ∃ s : ErrorPoly N, output = (fun _ _ ↦ reducePoly q N s) ∧ polyNorm s ≤ 1 := by
  obtain ⟨w, hw, hb⟩ := uniformIntervalSample_ternary h
  refine ⟨w 0 0, ?_, hb 0 0⟩
  funext i j
  rw [hw, Subsingleton.elim i 0, Subsingleton.elim j 0]
  rfl

/-- A Gaussian sample as an integer matrix with a coefficient bound. -/
theorem gaussian_bound {q rows columns : Nat} {output : ExactMatrix q N rows columns}
    (h : gaussianSample (4 / 1) 26 output) :
    ∃ e : ErrorMatrix N rows columns, output = reduceMatrix q N rows columns e ∧ CoeffBound e 26 := by
  obtain ⟨w, hw, hb⟩ := gaussianSample_bounded h
  exact ⟨w, hw, fun i j k ↦ (coeff_natAbs_le_polyNorm _ k).trans (by simpa using hb i j)⟩

/-- The message `c X^e` of a plaintext with coefficient `c` and exponent `e`. -/
noncomputable def message (coefficient exponent : Int) : ErrorPoly N :=
  (coefficient : ErrorPoly N) * AdjoinRoot.root (negacyclicModulus N Int) ^ (exponent % 64).toNat

theorem multiplyMonomial_liftInteger {q : Nat} (coefficient exponent : Int) :
    multiplyMonomial (liftInteger coefficient : ExactMatrix q N 1 1) exponent =
      fun _ _ ↦ reducePoly q N (message coefficient exponent) := by
  funext i j
  simp [multiplyMonomial, liftInteger, message, reducePoly]

/-! ## Hash tags -/

/-- `component` -/
def componentBytes : Blob := [99, 111, 109, 112, 111, 110, 101, 110, 116]

/-- `rns-bgg/lut-a` -/
def lutAPrefix : Blob := [114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]

/-- The public matrix `A_σ` of a LUT helper at `level` for the automorphism `sigma`. -/
noncomputable def lutA (model : HashModel) (key : ByteArray) (q m : Nat) (level : Int) (sigma : Int) :
    Fin 2 → Fin 2 → ExactMatrix q N 1 m :=
  fun k j ↦ model.sample q N 1 m key (completeHashTag lutAPrefix
    [.u64Le level, .integer sigma, .bytes componentBytes, .u64Le (2 * (k : Int) + (j : Int))])

/-- The automorphism indices `1 + 4 i` of the LUT layout. -/
abbrev sigmas : List Int := [(1), (5), (9), (13), (17), (21), (25), (29), (33), (37), (41), (45), (49),
  (53), (57), (61)]

/-- The automorphism of branch `i`. -/
abbrev sigma (i : Nat) : Int := MxxIR.selectInt (Int.ofNat i) sigmas

theorem sigma_odd (i : Fin 16) : Odd (sigma i).toNat := by
  fin_cases i <;> decide

theorem sigma_mod (i : Fin 16) : (sigma i).toNat % 64 = 1 + 2 * 2 * i.val := by
  fin_cases i <;> decide

end NandProof
