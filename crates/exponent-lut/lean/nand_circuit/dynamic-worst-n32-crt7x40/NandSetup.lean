import Claim
import ExponentLut
import ExponentDecode
import ExponentWalsh
import ExponentRuntime
import ExponentDynamic
import NandProgram

/-!
# The parameter set

Moduli, dropped primes, gadget layouts, and hash tags of the generated Dynamic NAND claim at this
parameter set. Levels `0` to `8` use the moduli `Q7, Q7, Q6, Q6, Q5, Q5, Q4, Q4, Q3` (the index
counts CRT primes): private multiplications keep the modulus and every LUT drops one prime
`P_k = Q_k / Q_{k-1}`.
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
theorem hN1 : 1 < N := by decide
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

/-- The inverses `σ_i⁻¹ mod 2N` of the automorphisms of the LUT layout. -/
abbrev sigmasInv : List Int := [(1), (13), (57), (5), (49), (61), (41), (53), (33), (45), (25),
  (37), (17), (29), (9), (21)]

/-- The inverse automorphism of branch `i`. -/
abbrev sigmaInv (i : Nat) : Int := MxxIR.selectInt (Int.ofNat i) sigmasInv

theorem sigmaInv_odd (i : Fin 16) : Odd (sigmaInv i).toNat := by
  fin_cases i <;> decide

theorem sigma_mul_sigmaInv (i : Fin 16) : (sigma i).toNat * (sigmaInv i).toNat % (2 * N) = 1 := by
  fin_cases i <;> decide

/-- The automorphism of branch `k` depends only on `k`. -/
theorem sigma_congr {q rows columns k k' : Nat} (hk : k < 16) (hk' : k' < 16) (h : k = k')
    (X : ExactMatrix q N rows columns) :
    substituteMatrix (sigma k).toNat (sigma_odd ⟨k, hk⟩) X =
      substituteMatrix (sigma k').toNat (sigma_odd ⟨k', hk'⟩) X := by
  subst h
  rfl

/-! ## Digit keys of the Dynamic companions -/

/-- `rns-bgg/dynamic-rhs-digit-key` -/
def digitKeyPrefix : Blob := [114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99,
  45, 114, 104, 115, 45, 100, 105, 103, 105, 116, 45, 107, 101, 121]

/-- The protocol hashes that open every digit-key tag. -/
def digitKeyHead : List HashTagComponent :=
  [.bytes [56, 52, 102, 53, 102, 51, 54, 98, 56, 101, 52, 100, 56, 56, 99, 52, 57, 48, 56, 97, 97,
      97, 52, 56, 97, 102, 52, 49, 51, 57, 49, 49, 53, 102, 49, 57, 98, 51, 102, 56, 57, 53, 56, 97,
      56, 97, 53, 55, 54, 50, 49, 56, 97, 53, 48, 99, 48, 102, 101, 52, 102, 55, 50, 99],
    .bytes [52, 101, 52, 101, 52, 101, 52, 101, 52, 101, 52, 101, 52, 101, 52, 101, 52, 101, 52,
      101, 52, 101, 52, 101, 52, 101, 52, 101, 52, 101, 52, 101, 52, 101, 52, 101, 52, 101, 52, 101,
      52, 101, 52, 101, 52, 101, 52, 101, 52, 101, 52, 101, 52, 101, 52, 101, 52, 101, 52, 101, 52,
      101, 52, 101],
    .bytes [57, 52, 51, 48, 52, 54, 50, 49, 53, 102, 52, 98, 100, 56, 52, 55, 102, 51, 102, 56, 98,
      57, 51, 99, 102, 97, 48, 97, 54, 54, 101, 98, 57, 100, 57, 49, 49, 102, 52, 52, 52, 50, 53,
      56, 100, 100, 57, 57, 97, 102, 100, 99, 102, 52, 56, 56, 53, 55, 97, 98, 49, 54, 99, 54]]

/-- The tag components naming RHS `RhsId(id - 48)` at `level`. -/
def rhsOwner (id : UInt8) (level : Int) : List HashTagComponent :=
  [.bytes [114, 104, 115, 47, 82, 104, 115, 73, 100, 40, id, 41], .bytes [108, 101, 118, 101, 108],
    .u64Le level]

/-- The tag components naming member `member` of LUT `LutId(id - 48)` at `level`. -/
def lutOwner (id : UInt8) (member level : Int) : List HashTagComponent :=
  [.bytes [108, 117, 116, 47, 76, 117, 116, 73, 100, 40, id, 41],
    .bytes [109, 101, 109, 98, 101, 114], .u64Le member, .bytes [108, 101, 118, 101, 108],
    .u64Le level]

/-- The keys `B` of the digit companions of row `row` of column `column`, owned by `owner`. -/
noncomputable def digitKey (model : HashModel) (key : ByteArray) (q D : Nat)
    (owner : List HashTagComponent) (column row : Int) : Fin 2 → Fin 2 → ExactMatrix q N 1 D :=
  fun k j ↦ model.sample q N 1 D key (completeHashTag digitKeyPrefix (digitKeyHead ++ owner ++
    [.bytes [99, 111, 108, 117, 109, 110], .u64Le column,
      .bytes [100, 105, 103, 105, 116, 45, 114, 111, 119], .u64Le row, .bytes componentBytes,
      .u64Le (2 * (k : Int) + (j : Int))]))

/-- The digits of entry `a` of the joined key row `[A_r0 | A_r1]`. -/
noncomputable def keyDigits {q m : Nat} (D : Nat) (A : Fin 2 → Fin 2 → ExactMatrix q N 1 m)
    (a : Fin (m + m)) (r : Fin 2) : ExactMatrix q N D 1 :=
  entryDigits D Backend.backend 1048576 D (fun _ _ ↦ joinedEntry (A r) a)

/-! ## LUT coefficients -/

theorem polyNorm_substitution_Xe (i : Fin 16) (k : Nat) :
    polyNorm (substitution (sigma i).toNat (sigma_odd i) (Xe (n := N) k)) ≤ 1 := by
  rw [substitution_root_pow]
  exact polyNorm_Xe hN1 _

/-- The coefficient `C_σ = W⁻¹ Σ_k X^(T k) σ(X^(-k))` of a width-16 LUT with table `T`. -/
noncomputable def lutCoefficient (q inverse : Nat) (table : Fin 16 → Nat) (σ : Nat) : ExactPoly q N :=
  lutFourierCoefficient 32 (X q) (inverse : ExactPoly q N) (fun k ↦ X q ^ table k) σ

theorem X_pow_period {q : Nat} (k : Nat) : X (n := N) q ^ (64 * k) = 1 := by
  have hroot : X (n := N) q ^ 32 = -1 := Negacyclic.root_pow_n
  rw [pow_mul, show X (n := N) q ^ 64 = (X q ^ 32) ^ 2 by rw [← pow_mul], hroot]
  simp

/-- `σ(X^(-k)) = X^(σ (64 - k))` is `-X^(σ (32 - k))` for an odd `σ`. -/
theorem X_pow_neg_odd {q σ : Nat} (hσ : Odd σ) (k : Nat) (hk : k ≤ 32) :
    X (n := N) q ^ (σ * (64 - k)) = -X q ^ (σ * (32 - k)) := by
  have hsplit : σ * (64 - k) = σ * (32 - k) + 32 * σ := by
    rw [show 64 - k = (32 - k) + 32 by omega, Nat.mul_add, Nat.mul_comm σ 32]
  have hroot : X (n := N) q ^ 32 = -1 := Negacyclic.root_pow_n
  rw [hsplit, pow_add, pow_mul (X q) 32 σ, hroot, hσ.neg_one_pow]
  ring

/-- The coefficient in the form the graph computes it: `σ(X^0) = 1` and
`σ(X^(-k)) = σ(-X^(32 - k)) = -X^(σ (32 - k))`. -/
theorem lutCoefficient_eq {q inverse σ : Nat} (hσ : Odd σ) (table : Fin 16 → Nat) :
    lutCoefficient q inverse table σ = (inverse : ExactPoly q N) *
      ∑ k : Fin 16, X q ^ table k * (if k.val = 0 then 1 else -X q ^ (σ * (32 - k.val))) := by
  unfold lutCoefficient lutFourierCoefficient
  congr 1
  apply Finset.sum_congr rfl
  intro k _
  congr 1
  split
  · rename_i hk
    rw [hk, show σ * (2 * 32 - 0) = 64 * σ by ring, X_pow_period]
  · exact X_pow_neg_odd hσ k.val (by omega)

/-- `σ` applied to a rotation constant. -/
theorem substitution_rotation {q σ : Nat} (hσ : Odd σ) (e : Nat) :
    substitution σ hσ ((rotationPolynomial (e : Int) : ExactMatrix q N 1 1) 0 0) = X q ^ (e * σ) := by
  simp [rotationPolynomial, substitution_root_pow]

/-- LUT interpolation on the lookup domain. -/
theorem lut_walsh {q inverse : Nat} (hinverse : (inverse : ExactPoly q N) * ((16 : Nat) : ExactPoly q N) = 1)
    (table : Fin 16 → Nat) (e : Fin 16) :
    ∑ i : Fin 16, X q ^ ((sigma i).toNat * e.val) * lutCoefficient q inverse table (sigma i).toNat =
      X q ^ table e := by
  have h := lut_fourier_table_value (R := ExactPoly q N) 4 32 2 (X q) (inverse : ExactPoly q N)
    (fun k ↦ X q ^ table k) e (by norm_num) (by norm_num) Negacyclic.root_pow_n hinverse
    (fun i ↦ (sigma i).toNat) (Equiv.refl _) (fun i ↦ by simpa using sigma_mod i)
  simpa [lutCoefficient] using h

/-- The inverses of the LUT width `16` at the LUT output moduli. -/
abbrev inverse6 : Nat := 1656419084768677916369117227758191167904054695734225248949574958445752701
abbrev inverse5 : Nat := 1506504390455695168269303022545438576397038817847440089317221
abbrev inverse4 : Nat := 1370157767855665020950234418224971312905348255901
abbrev inverse3 : Nat := 1246151236331995168128710588131142041

theorem inverse6_mul : (inverse6 : ExactPoly Q6 N) * ((16 : Nat) : ExactPoly Q6 N) = 1 :=
  inverse_mul (k := 15) (by norm_num)
theorem inverse5_mul : (inverse5 : ExactPoly Q5 N) * ((16 : Nat) : ExactPoly Q5 N) = 1 :=
  inverse_mul (k := 15) (by norm_num)
theorem inverse4_mul : (inverse4 : ExactPoly Q4 N) * ((16 : Nat) : ExactPoly Q4 N) = 1 :=
  inverse_mul (k := 15) (by norm_num)
theorem inverse3_mul : (inverse3 : ExactPoly Q3 N) * ((16 : Nat) : ExactPoly Q3 N) = 1 :=
  inverse_mul (k := 15) (by norm_num)

end NandProof
