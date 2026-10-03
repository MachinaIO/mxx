import Claim
import ExponentLut
import ExponentDecode
import ExponentWalsh
import ExponentPublic
import ExponentRuntime
import LwrProgram
import ExponentIterate

/-!
# The parameter set

Moduli, dropped primes, gadget layouts, the LUT automorphisms, and the hash tags of the generated
sparse-LWR claim at this parameter set. The CRT moduli are `Q7 ⊃ Q6 ⊃ ... ⊃ Q1` (the index counts
primes): block `j` of the accumulation multiplies at `Q_(7-j)` and its private multiplication drops
`P_(7-j) = Q_(7-j) / Q_(6-j)`; the readout LUT moves from `Q2` to `Q1`. A level-`ℓ` encoding at `Q_k`
uses the gadget of `Q_(k-1)` re-encoded at `Q_k` with scale `P_k`; the public vector `a` is encoded
at `Q7` with the native gadget and is aligned to the modulus of each block.
-/

namespace LwrProof

open Mxx.Primitives MxxRuntime MxxExponentLut

abbrev N : Nat := 32
abbrev Q7 : Nat := 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921
abbrev Q6 : Nat := 1766847023753256444127058376275403912430991675449840265546213289008802881
abbrev Q5 : Nat := 1606938016486074846153923224048467814823508072370602761938369
abbrev Q4 : Nat := 1461501619046042689013583379439969400432371472961
abbrev Q3 : Nat := 1329227985420794846003957960673218177
abbrev Q2 : Nat := 1208925815113228574679041
abbrev Q1 : Nat := 1099511626049
abbrev P7 : Nat := 1099511619841
abbrev P6 : Nat := 1099511621249
abbrev P5 : Nat := 1099511622529
abbrev P4 : Nat := 1099511622593
abbrev P3 : Nat := 1099511623297
abbrev P2 : Nat := 1099511625409

theorem hN : 0 < N := by decide
theorem hN1 : 1 < N := by decide
theorem hpow : N = 2 ^ 5 := rfl
theorem q7_eq : Q7 = P7 * Q6 := by norm_num
theorem q6_eq : Q6 = P6 * Q5 := by norm_num
theorem q5_eq : Q5 = P5 * Q4 := by norm_num
theorem q4_eq : Q4 = P4 * Q3 := by norm_num
theorem q3_eq : Q3 = P3 * Q2 := by norm_num
theorem q2_eq : Q2 = P2 * Q1 := by norm_num
/-- The factors `Q7 / Q_k` dropped by aligning `a` from `Q7` to `Q_k`. -/
abbrev A6 : Nat := 1099511619841
abbrev A5 : Nat := 1208925803713492065601409
abbrev A4 : Nat := 1329227971958197034497972917778543361
abbrev A3 : Nat := 1461501604243759924967666999996056787893117755073

theorem q7_eq6 : Q7 = A6 * Q6 := by norm_num
theorem q7_eq5 : Q7 = A5 * Q5 := by norm_num
theorem q7_eq4 : Q7 = A4 * Q4 := by norm_num
theorem q7_eq3 : Q7 = A3 * Q3 := by norm_num
theorem hQ7 : 1 < Q7 := by norm_num
theorem hQ6 : 1 < Q6 := by norm_num
theorem hQ5 : 1 < Q5 := by norm_num
theorem hQ4 : 1 < Q4 := by norm_num
theorem hQ3 : 1 < Q3 := by norm_num
theorem hQ2 : 1 < Q2 := by norm_num
theorem hQ1 : 1 < Q1 := by norm_num

theorem layout7 : Backend.backend.regularLayout Q7 N = some Backend.layout6 := by
  simp [Backend.backend]
theorem layout6 : Backend.backend.regularLayout Q6 N = some Backend.layout5 := by
  simp [Backend.backend]
theorem layout5 : Backend.backend.regularLayout Q5 N = some Backend.layout4 := by
  simp [Backend.backend]
theorem layout4 : Backend.backend.regularLayout Q4 N = some Backend.layout3 := by
  simp [Backend.backend]
theorem layout3 : Backend.backend.regularLayout Q3 N = some Backend.layout2 := by
  simp [Backend.backend]
theorem layout2 : Backend.backend.regularLayout Q2 N = some Backend.layout1 := by
  simp [Backend.backend]
theorem layout1 : Backend.backend.regularLayout Q1 N = some Backend.layout0 := by
  simp [Backend.backend]

/-! ## The readout LUT -/

/-- The automorphism indices of the readout LUT: one odd residue modulo `32` each. -/
abbrev sigmas : List Int := [(1), (3), (5), (7), (9), (11), (15), (17), (21), (25), (27), (45),
  (51), (55), (61), (63)]

/-- The automorphism of branch `i`. -/
abbrev sigma (i : Nat) : Int := MxxIR.selectInt (Int.ofNat i) sigmas

theorem sigma_odd (i : Fin 16) : Odd (sigma i).toNat := by
  fin_cases i <;> decide

/-- `σ_i ≡ 1 + 2 π(i) (mod 32)` for the permutation `π` of the residues. -/
def sigmaResidue : Fin 16 ≃ Fin 16 where
  toFun := ![0, 1, 2, 3, 4, 5, 7, 8, 10, 12, 13, 6, 9, 11, 14, 15]
  invFun := ![0, 1, 2, 3, 4, 5, 11, 6, 7, 12, 8, 13, 9, 10, 14, 15]
  left_inv := by decide
  right_inv := by decide

theorem sigma_mod (i : Fin 16) : (sigma i).toNat % 32 = 1 + 2 * (sigmaResidue i).val := by
  fin_cases i <;> decide

/-- The LUT variable `Y = X^2` of stride `2`; `Y^16 = -1`. -/
noncomputable abbrev Y (q : Nat) : ExactPoly q N := X q ^ 2

theorem Y_pow_16 {q : Nat} : Y q ^ 16 = -1 := by
  rw [Y, ← pow_mul]
  exact Negacyclic.root_pow_n

/-- The readout value `τ(k) = ⌊2 k / 16⌋ + 1` as a constant. -/
noncomputable def readoutPoly (q : Nat) (k : Fin 16) : ExactPoly q N :=
  ((MxxExponentLutLwr.readout 2 16 k.val : Nat) : ExactPoly q N)

/-- The coefficient `C_σ = W⁻¹ Σ_k τ(k) σ(Y^(-k))` of the readout LUT. -/
noncomputable def lutCoefficient (q inverse : Nat) (σ : Nat) : ExactPoly q N :=
  lutFourierCoefficient 16 (Y q) (inverse : ExactPoly q N) (readoutPoly q) σ

/-- The readout LUT on every exponent `z`: `Σ_i σ_i(Y^z) C_i = Y^(z - z mod 16) τ(z mod 16)`. -/
theorem lut_walsh {q inverse : Nat}
    (hinverse : (inverse : ExactPoly q N) * ((16 : Nat) : ExactPoly q N) = 1) (z : Nat) :
    ∑ i : Fin 16, Y q ^ ((sigma i).toNat * z) * lutCoefficient q inverse (sigma i).toNat =
      Y q ^ (z - z % 16) * readoutPoly q ⟨z % 16, Nat.mod_lt _ (by norm_num)⟩ := by
  have h := lut_fourier_interpolation (R := ExactPoly q N) 4 16 1 z (Y q) (inverse : ExactPoly q N)
    (readoutPoly q) (by norm_num) (by norm_num) Y_pow_16 hinverse
    (fun i ↦ (sigma i).toNat) sigmaResidue (fun i ↦ by simpa using sigma_mod i)
  simpa [lutCoefficient] using h

/-- `X^(64 k) = 1`. -/
theorem X_pow_period {q : Nat} (k : Nat) : X (n := N) q ^ (64 * k) = 1 := by
  have hroot : X (n := N) q ^ 32 = -1 := Negacyclic.root_pow_n
  rw [pow_mul, show X (n := N) q ^ 64 = (X q ^ 32) ^ 2 by rw [← pow_mul], hroot]
  simp

/-- The coefficient in the form the graph computes it: `σ(Y^0) = 1` and
`σ(Y^(-k)) = σ(-X^(32 - 2k)) = -X^(σ (32 - 2k))`. -/
theorem lutCoefficient_eq {q inverse σ : Nat} (hσ : Odd σ) :
    lutCoefficient q inverse σ = (inverse : ExactPoly q N) *
      ∑ k : Fin 16, readoutPoly q k * (if k.val = 0 then 1 else -X q ^ (σ * (32 - 2 * k.val))) := by
  unfold lutCoefficient lutFourierCoefficient
  congr 1
  apply Finset.sum_congr rfl
  intro k _
  congr 1
  rw [Y, ← pow_mul]
  split
  · rename_i hk
    rw [hk, show 2 * (σ * (2 * 16 - 0)) = 64 * σ by ring, X_pow_period]
  · have hsplit : 2 * (σ * (2 * 16 - k.val)) = σ * (32 - 2 * k.val) + 32 * σ := by
      have := k.isLt
      rw [Nat.mul_comm 32 σ, ← Nat.mul_add, Nat.mul_left_comm]
      congr 1
      omega
    have hroot : X (n := N) q ^ 32 = -1 := Negacyclic.root_pow_n
    rw [hsplit, pow_add, pow_mul (X q) 32 σ, hroot, hσ.neg_one_pow]
    ring

theorem polyNorm_substitution_Xe (i : Fin 16) (k : Nat) :
    polyNorm (substitution (sigma i).toNat (sigma_odd i) (Xe (n := N) k)) ≤ 1 := by
  rw [substitution_root_pow]
  exact polyNorm_Xe hN1 _

/-- The readout of the exponent `z`: `(-1)^⌊z / 16⌋ τ(z mod 16)`. -/
noncomputable def lutValue (z : Nat) : ErrorPoly N :=
  (((-1 : Int) ^ (z / 16) * (MxxExponentLutLwr.readout 2 16 (z % 16) : Int) : Int) : ErrorPoly N)

/-- The readout LUT on the state `X^(2z)`. -/
theorem lut_walsh_value {q inverse : Nat}
    (hinverse : (inverse : ExactPoly q N) * ((16 : Nat) : ExactPoly q N) = 1) (z : Nat) :
    ∑ i : Fin 16, reducePoly q N (substitution (sigma i).toNat (sigma_odd i) (Xe (2 * z))) *
        lutCoefficient q inverse (sigma i).toNat = reducePoly q N (lutValue z) := by
  have h := lut_walsh hinverse z
  have hY : Y q ^ (z - z % 16) = (-1) ^ (z / 16) := by
    rw [show z - z % 16 = 16 * (z / 16) by have := Nat.div_add_mod z 16; omega, pow_mul, Y_pow_16]
  rw [hY] at h
  have hvalue : reducePoly q N (lutValue z) =
      (-1) ^ (z / 16) * readoutPoly q ⟨z % 16, Nat.mod_lt _ (by norm_num)⟩ := by
    simp [lutValue, readoutPoly]
  rw [hvalue, ← h]
  apply Finset.sum_congr rfl
  intro i _
  rw [substitution_root_pow, reducePoly_Xe, Y, ← pow_mul]
  congr 2
  ring

/-- The inverse of the LUT width `16` modulo `Q1`. -/
abbrev inverse1 : Nat := 1030792149421

theorem inverse1_mul : (inverse1 : ExactPoly Q1 N) * ((16 : Nat) : ExactPoly Q1 N) = 1 :=
  inverse_mul (k := 15) (by norm_num)

/-! ## The key layout of output 0 -/

/-- The coordinates of `a` read by the cells of every block; dummy and padding cells read member
`16`. -/
def coordsTable : Fin 5 → List Nat :=
  ![
    [4, 5, 10, 11, 12, 14, 15, 16, 16, 16, 16, 16, 16, 16],
    [0, 1, 2, 4, 6, 7, 8, 9, 12, 13, 14, 15, 16, 16],
    [0, 1, 2, 3, 5, 8, 9, 10, 11, 12, 15, 16, 16, 16],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 16],
    [3, 6, 7, 13, 14, 16, 16, 16, 16, 16, 16, 16, 16, 16]]

/-- The coordinate of `a` read by cell `i` of block `b`. -/
def coords (b : Fin 5) (i : Fin 14) : Nat := (coordsTable b).getD i 0

theorem coords_lt (b : Fin 5) (i : Fin 14) : coords b i < 17 :=
  getD_lt_of_all (by norm_num) (by revert b; decide) i

theorem selectInt_coords0 (i : Fin 14) :
    MxxIR.selectInt (Int.ofNat i) [(4), (5), (10), (11), (12), (14), (15), (16), (16), (16), (16), (16), (16), (16)] = Int.ofNat (coords 0 i) :=
  selectInt_map_ofNat (coordsTable 0) i

theorem selectInt_coords1 (i : Fin 14) :
    MxxIR.selectInt (Int.ofNat i) [(0), (1), (2), (4), (6), (7), (8), (9), (12), (13), (14), (15), (16), (16)] = Int.ofNat (coords 1 i) :=
  selectInt_map_ofNat (coordsTable 1) i

theorem selectInt_coords2 (i : Fin 14) :
    MxxIR.selectInt (Int.ofNat i) [(0), (1), (2), (3), (5), (8), (9), (10), (11), (12), (15), (16), (16), (16)] = Int.ofNat (coords 2 i) :=
  selectInt_map_ofNat (coordsTable 2) i

theorem selectInt_coords3 (i : Fin 14) :
    MxxIR.selectInt (Int.ofNat i) [(0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (13), (16)] = Int.ofNat (coords 3 i) :=
  selectInt_map_ofNat (coordsTable 3) i

theorem selectInt_coords4 (i : Fin 14) :
    MxxIR.selectInt (Int.ofNat i) [(3), (6), (7), (13), (14), (16), (16), (16), (16), (16), (16), (16), (16), (16)] = Int.ofNat (coords 4 i) :=
  selectInt_map_ofNat (coordsTable 4) i

/-! ## Hash tags -/

/-- `component` -/
def componentBytes : Blob := [99, 111, 109, 112, 111, 110, 101, 110, 116]

/-- `member` -/
def memberBytes : Blob := [109, 101, 109, 98, 101, 114]

/-- `rns-bgg/lut-a` -/
def lutAPrefix : Blob := [114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]

/-- `rns-bgg/input/WireId(0)/0`: the state input. -/
def statePrefix : Blob :=
  [114, 110, 115, 45, 98, 103, 103, 47, 105, 110, 112, 117, 116, 47, 87, 105, 114, 101, 73, 100,
    40, 48, 41, 47, 48]

/-- `rns-bgg/input/WireId(1)/0`: the public vector `a`. -/
def memberPrefix : Blob :=
  [114, 110, 115, 45, 98, 103, 103, 47, 105, 110, 112, 117, 116, 47, 87, 105, 114, 101, 73, 100,
    40, 49, 41, 47, 48]

/-- The public matrix of the state input. -/
noncomputable def stateKey (model : HashModel) (key : ByteArray) :
    Fin 2 → Fin 2 → ExactMatrix Q7 N 1 12 :=
  fun k j ↦ model.sample Q7 N 1 12 key (completeHashTag statePrefix
    [.bytes componentBytes, .u64Le (2 * (k : Int) + (j : Int))])

/-- The public matrix of member `i` of `a`. -/
noncomputable def memberKey (model : HashModel) (key : ByteArray) (i : Int) :
    Fin 2 → Fin 2 → ExactMatrix Q7 N 1 14 :=
  fun k j ↦ model.sample Q7 N 1 14 key (completeHashTag memberPrefix
    [.bytes memberBytes, .u64Le i, .bytes componentBytes, .u64Le (2 * (k : Int) + (j : Int))])

/-- The public matrix `A_σ` of a LUT helper at `level` for the automorphism `sigma`. -/
noncomputable def lutA (model : HashModel) (key : ByteArray) (q m : Nat) (level : Int) (sigma : Int) :
    Fin 2 → Fin 2 → ExactMatrix q N 1 m :=
  fun k j ↦ model.sample q N 1 m key (completeHashTag lutAPrefix
    [.u64Le level, .integer sigma, .bytes componentBytes, .u64Le (2 * (k : Int) + (j : Int))])

/-- The RHS tops of member `i` of a selector family. -/
noncomputable def rhsTop (model : HashModel) (key : ByteArray) (q m : Nat) (tagPrefix : Blob)
    (i : Int) : Fin 2 → ExactMatrix q N 1 m :=
  fun j ↦ model.sample q N 1 m key (completeHashTag tagPrefix
    [.bytes memberBytes, .u64Le i, .bytes componentBytes, .u64Le (j : Int)])

/-- `rns-bgg/rhs/63db44534261096922a2cfd1dcb584cf7eeca2c9e015bdaba3deded0c2e3c398/ExponentLutProgramId([80, 246, 244, 43, 167, 61, 62, 12, 162, 195, 241, 155, 192, 180, 214, 73, 55, 186, 131, 190, 145, 17, 234, 200, 111, 54, 53, 201, 176, 224, 227, 80])/RhsId(0)/0/1` -/
def rhsPrefix0 : Blob :=
  [114, 110, 115, 45, 98, 103, 103, 47, 114, 104, 115, 47, 54, 51, 100, 98, 52, 52, 53, 51, 52, 50, 54, 49, 48, 57, 54, 57, 50, 50, 97, 50, 99, 102, 100, 49, 100, 99, 98, 53, 56, 52, 99, 102, 55, 101, 101, 99, 97, 50, 99, 57, 101, 48, 49, 53, 98, 100, 97, 98, 97, 51, 100, 101, 100, 101, 100, 48, 99, 50, 101, 51, 99, 51, 57, 56, 47, 69, 120, 112, 111, 110, 101, 110, 116, 76, 117, 116, 80, 114, 111, 103, 114, 97, 109, 73, 100, 40, 91, 56, 48, 44, 32, 50, 52, 54, 44, 32, 50, 52, 52, 44, 32, 52, 51, 44, 32, 49, 54, 55, 44, 32, 54, 49, 44, 32, 54, 50, 44, 32, 49, 50, 44, 32, 49, 54, 50, 44, 32, 49, 57, 53, 44, 32, 50, 52, 49, 44, 32, 49, 53, 53, 44, 32, 49, 57, 50, 44, 32, 49, 56, 48, 44, 32, 50, 49, 52, 44, 32, 55, 51, 44, 32, 53, 53, 44, 32, 49, 56, 54, 44, 32, 49, 51, 49, 44, 32, 49, 57, 48, 44, 32, 49, 52, 53, 44, 32, 49, 55, 44, 32, 50, 51, 52, 44, 32, 50, 48, 48, 44, 32, 49, 49, 49, 44, 32, 53, 52, 44, 32, 53, 51, 44, 32, 50, 48, 49, 44, 32, 49, 55, 54, 44, 32, 50, 50, 52, 44, 32, 50, 50, 55, 44, 32, 56, 48, 93, 41, 47, 82, 104, 115, 73, 100, 40, 48, 41, 47, 48, 47, 49]

/-- `rns-bgg/rhs/63db44534261096922a2cfd1dcb584cf7eeca2c9e015bdaba3deded0c2e3c398/ExponentLutProgramId([80, 246, 244, 43, 167, 61, 62, 12, 162, 195, 241, 155, 192, 180, 214, 73, 55, 186, 131, 190, 145, 17, 234, 200, 111, 54, 53, 201, 176, 224, 227, 80])/RhsId(1)/1/2` -/
def rhsPrefix1 : Blob :=
  [114, 110, 115, 45, 98, 103, 103, 47, 114, 104, 115, 47, 54, 51, 100, 98, 52, 52, 53, 51, 52, 50, 54, 49, 48, 57, 54, 57, 50, 50, 97, 50, 99, 102, 100, 49, 100, 99, 98, 53, 56, 52, 99, 102, 55, 101, 101, 99, 97, 50, 99, 57, 101, 48, 49, 53, 98, 100, 97, 98, 97, 51, 100, 101, 100, 101, 100, 48, 99, 50, 101, 51, 99, 51, 57, 56, 47, 69, 120, 112, 111, 110, 101, 110, 116, 76, 117, 116, 80, 114, 111, 103, 114, 97, 109, 73, 100, 40, 91, 56, 48, 44, 32, 50, 52, 54, 44, 32, 50, 52, 52, 44, 32, 52, 51, 44, 32, 49, 54, 55, 44, 32, 54, 49, 44, 32, 54, 50, 44, 32, 49, 50, 44, 32, 49, 54, 50, 44, 32, 49, 57, 53, 44, 32, 50, 52, 49, 44, 32, 49, 53, 53, 44, 32, 49, 57, 50, 44, 32, 49, 56, 48, 44, 32, 50, 49, 52, 44, 32, 55, 51, 44, 32, 53, 53, 44, 32, 49, 56, 54, 44, 32, 49, 51, 49, 44, 32, 49, 57, 48, 44, 32, 49, 52, 53, 44, 32, 49, 55, 44, 32, 50, 51, 52, 44, 32, 50, 48, 48, 44, 32, 49, 49, 49, 44, 32, 53, 52, 44, 32, 53, 51, 44, 32, 50, 48, 49, 44, 32, 49, 55, 54, 44, 32, 50, 50, 52, 44, 32, 50, 50, 55, 44, 32, 56, 48, 93, 41, 47, 82, 104, 115, 73, 100, 40, 49, 41, 47, 49, 47, 50]

/-- `rns-bgg/rhs/63db44534261096922a2cfd1dcb584cf7eeca2c9e015bdaba3deded0c2e3c398/ExponentLutProgramId([80, 246, 244, 43, 167, 61, 62, 12, 162, 195, 241, 155, 192, 180, 214, 73, 55, 186, 131, 190, 145, 17, 234, 200, 111, 54, 53, 201, 176, 224, 227, 80])/RhsId(2)/2/3` -/
def rhsPrefix2 : Blob :=
  [114, 110, 115, 45, 98, 103, 103, 47, 114, 104, 115, 47, 54, 51, 100, 98, 52, 52, 53, 51, 52, 50, 54, 49, 48, 57, 54, 57, 50, 50, 97, 50, 99, 102, 100, 49, 100, 99, 98, 53, 56, 52, 99, 102, 55, 101, 101, 99, 97, 50, 99, 57, 101, 48, 49, 53, 98, 100, 97, 98, 97, 51, 100, 101, 100, 101, 100, 48, 99, 50, 101, 51, 99, 51, 57, 56, 47, 69, 120, 112, 111, 110, 101, 110, 116, 76, 117, 116, 80, 114, 111, 103, 114, 97, 109, 73, 100, 40, 91, 56, 48, 44, 32, 50, 52, 54, 44, 32, 50, 52, 52, 44, 32, 52, 51, 44, 32, 49, 54, 55, 44, 32, 54, 49, 44, 32, 54, 50, 44, 32, 49, 50, 44, 32, 49, 54, 50, 44, 32, 49, 57, 53, 44, 32, 50, 52, 49, 44, 32, 49, 53, 53, 44, 32, 49, 57, 50, 44, 32, 49, 56, 48, 44, 32, 50, 49, 52, 44, 32, 55, 51, 44, 32, 53, 53, 44, 32, 49, 56, 54, 44, 32, 49, 51, 49, 44, 32, 49, 57, 48, 44, 32, 49, 52, 53, 44, 32, 49, 55, 44, 32, 50, 51, 52, 44, 32, 50, 48, 48, 44, 32, 49, 49, 49, 44, 32, 53, 52, 44, 32, 53, 51, 44, 32, 50, 48, 49, 44, 32, 49, 55, 54, 44, 32, 50, 50, 52, 44, 32, 50, 50, 55, 44, 32, 56, 48, 93, 41, 47, 82, 104, 115, 73, 100, 40, 50, 41, 47, 50, 47, 51]

/-- `rns-bgg/rhs/63db44534261096922a2cfd1dcb584cf7eeca2c9e015bdaba3deded0c2e3c398/ExponentLutProgramId([80, 246, 244, 43, 167, 61, 62, 12, 162, 195, 241, 155, 192, 180, 214, 73, 55, 186, 131, 190, 145, 17, 234, 200, 111, 54, 53, 201, 176, 224, 227, 80])/RhsId(3)/3/4` -/
def rhsPrefix3 : Blob :=
  [114, 110, 115, 45, 98, 103, 103, 47, 114, 104, 115, 47, 54, 51, 100, 98, 52, 52, 53, 51, 52, 50, 54, 49, 48, 57, 54, 57, 50, 50, 97, 50, 99, 102, 100, 49, 100, 99, 98, 53, 56, 52, 99, 102, 55, 101, 101, 99, 97, 50, 99, 57, 101, 48, 49, 53, 98, 100, 97, 98, 97, 51, 100, 101, 100, 101, 100, 48, 99, 50, 101, 51, 99, 51, 57, 56, 47, 69, 120, 112, 111, 110, 101, 110, 116, 76, 117, 116, 80, 114, 111, 103, 114, 97, 109, 73, 100, 40, 91, 56, 48, 44, 32, 50, 52, 54, 44, 32, 50, 52, 52, 44, 32, 52, 51, 44, 32, 49, 54, 55, 44, 32, 54, 49, 44, 32, 54, 50, 44, 32, 49, 50, 44, 32, 49, 54, 50, 44, 32, 49, 57, 53, 44, 32, 50, 52, 49, 44, 32, 49, 53, 53, 44, 32, 49, 57, 50, 44, 32, 49, 56, 48, 44, 32, 50, 49, 52, 44, 32, 55, 51, 44, 32, 53, 53, 44, 32, 49, 56, 54, 44, 32, 49, 51, 49, 44, 32, 49, 57, 48, 44, 32, 49, 52, 53, 44, 32, 49, 55, 44, 32, 50, 51, 52, 44, 32, 50, 48, 48, 44, 32, 49, 49, 49, 44, 32, 53, 52, 44, 32, 53, 51, 44, 32, 50, 48, 49, 44, 32, 49, 55, 54, 44, 32, 50, 50, 52, 44, 32, 50, 50, 55, 44, 32, 56, 48, 93, 41, 47, 82, 104, 115, 73, 100, 40, 51, 41, 47, 51, 47, 52]

/-- `rns-bgg/rhs/63db44534261096922a2cfd1dcb584cf7eeca2c9e015bdaba3deded0c2e3c398/ExponentLutProgramId([80, 246, 244, 43, 167, 61, 62, 12, 162, 195, 241, 155, 192, 180, 214, 73, 55, 186, 131, 190, 145, 17, 234, 200, 111, 54, 53, 201, 176, 224, 227, 80])/RhsId(4)/4/5` -/
def rhsPrefix4 : Blob :=
  [114, 110, 115, 45, 98, 103, 103, 47, 114, 104, 115, 47, 54, 51, 100, 98, 52, 52, 53, 51, 52, 50, 54, 49, 48, 57, 54, 57, 50, 50, 97, 50, 99, 102, 100, 49, 100, 99, 98, 53, 56, 52, 99, 102, 55, 101, 101, 99, 97, 50, 99, 57, 101, 48, 49, 53, 98, 100, 97, 98, 97, 51, 100, 101, 100, 101, 100, 48, 99, 50, 101, 51, 99, 51, 57, 56, 47, 69, 120, 112, 111, 110, 101, 110, 116, 76, 117, 116, 80, 114, 111, 103, 114, 97, 109, 73, 100, 40, 91, 56, 48, 44, 32, 50, 52, 54, 44, 32, 50, 52, 52, 44, 32, 52, 51, 44, 32, 49, 54, 55, 44, 32, 54, 49, 44, 32, 54, 50, 44, 32, 49, 50, 44, 32, 49, 54, 50, 44, 32, 49, 57, 53, 44, 32, 50, 52, 49, 44, 32, 49, 53, 53, 44, 32, 49, 57, 50, 44, 32, 49, 56, 48, 44, 32, 50, 49, 52, 44, 32, 55, 51, 44, 32, 53, 53, 44, 32, 49, 56, 54, 44, 32, 49, 51, 49, 44, 32, 49, 57, 48, 44, 32, 49, 52, 53, 44, 32, 49, 55, 44, 32, 50, 51, 52, 44, 32, 50, 48, 48, 44, 32, 49, 49, 49, 44, 32, 53, 52, 44, 32, 53, 51, 44, 32, 50, 48, 49, 44, 32, 49, 55, 54, 44, 32, 50, 50, 52, 44, 32, 50, 50, 55, 44, 32, 56, 48, 93, 41, 47, 82, 104, 115, 73, 100, 40, 52, 41, 47, 52, 47, 53]

end LwrProof
