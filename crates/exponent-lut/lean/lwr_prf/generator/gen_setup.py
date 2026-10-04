"""LwrSetup.lean: moduli, layouts, the readout LUT, the key layout, and the hash tags."""


def blob(bytes_text, width=96):
    items = bytes_text.split(', ')
    lines, cur = [], ''
    for x in items:
        piece = x + ', '
        if len(cur) + len(piece) > width:
            lines.append(cur.rstrip())
            cur = ''
        cur += piece
    lines.append(cur.rstrip().rstrip(','))
    return '[' + '\n    '.join(lines) + ']'


def setup(I):
    L, N, W = I.L, I.n, I.W
    o = []
    o.append(f'''import Claim
import ExponentLut
import ExponentDecode
import ExponentWalsh
import ExponentPublic
import ExponentRuntime
import ExponentIterate
import LwrProgram

/-!
# The parameter set

Moduli, dropped primes, gadget layouts, the LUT automorphisms, the key layout, and the hash tags of
the generated sparse-LWR claim at this parameter set. The CRT moduli are `Q{L} ⊃ Q{L - 1} ⊃ ... ⊃ Q1`
(the index counts primes): block `j` of the accumulation multiplies at `Q_({L}-j)` and its private
multiplication drops `P_({L}-j) = Q_({L}-j) / Q_({L - 1}-j)`; the readout LUT moves from `Q2` to `Q1`. A
level-`ℓ` encoding at `Q_k` uses the gadget of `Q_(k-1)` re-encoded at `Q_k` with scale `P_k`; the
public vector `a` is encoded at `Q{L}` with the native gadget and is aligned to the modulus of each
block.
-/

namespace LwrProof

open Mxx.Primitives MxxRuntime MxxExponentLut

set_option maxRecDepth 100000

abbrev N : Nat := {N}
''')
    for k in range(L, 0, -1):
        o.append(f'abbrev Q{k} : Nat := {I.Q[k]}\n')
    for k in range(L, 1, -1):
        o.append(f'abbrev P{k} : Nat := {I.P[k]}\n')
    o.append(f'''
theorem hN : 0 < N := by decide
theorem hN1 : 1 < N := by decide
theorem hpow : N = 2 ^ {I.log_n} := rfl
''')
    for k in range(L, 1, -1):
        o.append(f'theorem q{k}_eq : Q{k} = P{k} * Q{k - 1} := by norm_num\n')
    o.append(f'/-- The factors `Q{L} / Q_k` dropped by aligning `a` from `Q{L}` to `Q_k`. -/\n')
    for k in range(L - 1, 2, -1):
        o.append(f'abbrev A{k} : Nat := {I.Q[L] // I.Q[k]}\n')
    o.append('\n')
    for k in range(L - 1, 2, -1):
        o.append(f'theorem q{L}_eq{k} : Q{L} = A{k} * Q{k} := by norm_num\n')
    for k in range(L, 0, -1):
        o.append(f'theorem hQ{k} : 1 < Q{k} := by norm_num\n')
    o.append('\n')
    for k in range(L, 0, -1):
        o.append(f'''theorem layout{k} : Backend.backend.regularLayout Q{k} N = some Backend.layout{k - 1} := by
  simp [Backend.backend]
''')
    # The readout LUT.
    s = I.stride
    perm = [((x % (2 * W)) - 1) // 2 for x in I.sigmas]
    assert sorted(perm) == list(range(W))
    inv = [0] * W
    for i, p in enumerate(perm):
        inv[p] = i
    vec = lambda v: '![' + ', '.join(map(str, v)) + ']'
    sig = ', '.join(f'({x})' for x in I.sigmas)
    o.append(f'''
/-! ## The readout LUT -/

/-- The automorphism indices of the readout LUT: one odd residue modulo `{2 * W}` each. -/
abbrev sigmas : List Int := [{sig}]

/-- The automorphism of branch `i`. -/
abbrev sigma (i : Nat) : Int := MxxIR.selectInt (Int.ofNat i) sigmas

theorem sigma_odd (i : Fin {W}) : Odd (sigma i).toNat := by
  fin_cases i <;> decide

/-- `σ_i ≡ 1 + 2 π(i) (mod {2 * W})` for the permutation `π` of the residues. -/
def sigmaResidue : Fin {W} ≃ Fin {W} where
  toFun := {vec(perm)}
  invFun := {vec(inv)}
  left_inv := by decide
  right_inv := by decide

theorem sigma_mod (i : Fin {W}) : (sigma i).toNat % {2 * W} = 1 + 2 * (sigmaResidue i).val := by
  fin_cases i <;> decide

/-- The LUT variable `Y = X^{s}` of stride `{s}`; `Y^{W} = -1`. -/
noncomputable abbrev Y (q : Nat) : ExactPoly q N := X q ^ {s}

theorem Y_pow_{W} {{q : Nat}} : Y q ^ {W} = -1 := by
  rw [Y, ← pow_mul]
  exact Negacyclic.root_pow_n

/-- The readout value `τ(k) = ⌊{I.pprime} k / {W}⌋ + 1` as a constant. -/
noncomputable def readoutPoly (q : Nat) (k : Fin {W}) : ExactPoly q N :=
  ((MxxExponentLutLwr.readout {I.pprime} {W} k.val : Nat) : ExactPoly q N)

/-- The coefficient `C_σ = W⁻¹ Σ_k τ(k) σ(Y^(-k))` of the readout LUT. -/
noncomputable def lutCoefficient (q inverse : Nat) (σ : Nat) : ExactPoly q N :=
  lutFourierCoefficient {W} (Y q) (inverse : ExactPoly q N) (readoutPoly q) σ

set_option maxHeartbeats 4000000 in
/-- The readout LUT on every exponent `z`: `Σ_i σ_i(Y^z) C_i = Y^(z - z mod {W}) τ(z mod {W})`. -/
theorem lut_walsh {{q inverse : Nat}}
    (hinverse : (inverse : ExactPoly q N) * (({W} : Nat) : ExactPoly q N) = 1) (z : Nat) :
    ∑ i : Fin {W}, Y q ^ ((sigma i).toNat * z) * lutCoefficient q inverse (sigma i).toNat =
      Y q ^ (z - z % {W}) * readoutPoly q ⟨z % {W}, Nat.mod_lt _ (by norm_num)⟩ := by
  have h := lut_fourier_interpolation (R := ExactPoly q N) {I.log_W} {W} 1 z (Y q) (inverse : ExactPoly q N)
    (readoutPoly q) (by norm_num) (by norm_num) Y_pow_{W} hinverse
    (fun i ↦ (sigma i).toNat) sigmaResidue (fun i ↦ by simpa using sigma_mod i)
  simpa [lutCoefficient] using h

/-- `X^({2 * N} k) = 1`. -/
theorem X_pow_period {{q : Nat}} (k : Nat) : X (n := N) q ^ ({2 * N} * k) = 1 := by
  have hroot : X (n := N) q ^ {N} = -1 := Negacyclic.root_pow_n
  rw [pow_mul, show X (n := N) q ^ {2 * N} = (X q ^ {N}) ^ 2 by rw [← pow_mul], hroot]
  simp

set_option maxHeartbeats 4000000 in
/-- The coefficient in the form the graph computes it: `σ(Y^0) = 1` and
`σ(Y^(-k)) = σ(-X^({N} - {s}k)) = -X^(σ ({N} - {s}k))`. -/
theorem lutCoefficient_eq {{q inverse σ : Nat}} (hσ : Odd σ) :
    lutCoefficient q inverse σ = (inverse : ExactPoly q N) *
      ∑ k : Fin {W}, readoutPoly q k * (if k.val = 0 then 1 else -X q ^ (σ * ({N} - {s} * k.val))) := by
  unfold lutCoefficient lutFourierCoefficient
  congr 1
  apply Finset.sum_congr rfl
  intro k _
  congr 1
  rw [Y, ← pow_mul]
  split
  · rename_i hk
    rw [hk, show {s} * (σ * (2 * {W} - 0)) = {2 * N} * σ by ring, X_pow_period]
  · have hsplit : {s} * (σ * (2 * {W} - k.val)) = σ * ({N} - {s} * k.val) + {N} * σ := by
      have := k.isLt
      rw [Nat.mul_comm {N} σ, ← Nat.mul_add, Nat.mul_left_comm]
      congr 1
      omega
    have hroot : X (n := N) q ^ {N} = -1 := Negacyclic.root_pow_n
    rw [hsplit, pow_add, pow_mul (X q) {N} σ, hroot, hσ.neg_one_pow]
    ring

/-- `lutCoefficient_eq` as a sum over `Finset.range {W}`, whose terms have literal indices once
expanded. -/
theorem lutCoefficient_range {{q inverse σ : Nat}} (hσ : Odd σ) :
    lutCoefficient q inverse σ = (inverse : ExactPoly q N) *
      ∑ k ∈ Finset.range {W}, ((MxxExponentLutLwr.readout {I.pprime} {W} k : Nat) : ExactPoly q N) *
        (if k = 0 then 1 else -X q ^ (σ * ({N} - {s} * k))) := by
  rw [lutCoefficient_eq hσ, ← Fin.sum_univ_eq_sum_range (fun k ↦
    ((MxxExponentLutLwr.readout {I.pprime} {W} k : Nat) : ExactPoly q N) *
      (if k = 0 then 1 else -X q ^ (σ * ({N} - {s} * k)))) {W}]
  rfl

theorem polyNorm_substitution_Xe (i : Fin {W}) (k : Nat) :
    polyNorm (substitution (sigma i).toNat (sigma_odd i) (Xe (n := N) k)) ≤ 1 := by
  rw [substitution_root_pow]
  exact polyNorm_Xe hN1 _

/-- The readout of the exponent `z`: `(-1)^⌊z / {W}⌋ τ(z mod {W})`. -/
noncomputable def lutValue (z : Nat) : ErrorPoly N :=
  (((-1 : Int) ^ (z / {W}) * (MxxExponentLutLwr.readout {I.pprime} {W} (z % {W}) : Int) : Int) : ErrorPoly N)

set_option maxHeartbeats 4000000 in
/-- The readout LUT on the state `X^({s}z)`. -/
theorem lut_walsh_value {{q inverse : Nat}}
    (hinverse : (inverse : ExactPoly q N) * (({W} : Nat) : ExactPoly q N) = 1) (z : Nat) :
    ∑ i : Fin {W}, reducePoly q N (substitution (sigma i).toNat (sigma_odd i) (Xe ({s} * z))) *
        lutCoefficient q inverse (sigma i).toNat = reducePoly q N (lutValue z) := by
  have h := lut_walsh hinverse z
  have hY : Y q ^ (z - z % {W}) = (-1) ^ (z / {W}) := by
    rw [show z - z % {W} = {W} * (z / {W}) by have := Nat.div_add_mod z {W}; omega, pow_mul, Y_pow_{W}]
  rw [hY] at h
  have hvalue : reducePoly q N (lutValue z) =
      (-1) ^ (z / {W}) * readoutPoly q ⟨z % {W}, Nat.mod_lt _ (by norm_num)⟩ := by
    simp [lutValue, readoutPoly]
  rw [hvalue, ← h]
  apply Finset.sum_congr rfl
  intro i _
  rw [substitution_root_pow, reducePoly_Xe, Y, ← pow_mul]
  congr 2
  ring

/-- The inverse of the LUT width `{W}` modulo `Q1`. -/
abbrev inverse1 : Nat := {I.inverse}

theorem inverse1_mul : (inverse1 : ExactPoly Q1 N) * (({W} : Nat) : ExactPoly Q1 N) = 1 :=
  inverse_mul (k := {I.inverse_k}) (by norm_num)
''')
    # The key layout.
    B, C, M = I.B, I.C, I.M
    tab = ',\n'.join(f'    {c}' for c in I.coords)
    o.append(f'''
/-! ## The key layout of output 0 -/

/-- The coordinates of `a` read by the cells of every block; dummy and padding cells read member
`{M - 1}`. -/
def coordsTable : Fin {B} → List Nat :=
  ![
{tab}]

/-- The coordinate of `a` read by cell `i` of block `b`. -/
def coords (b : Fin {B}) (i : Fin {C}) : Nat := (coordsTable b).getD i 0

theorem coords_lt (b : Fin {B}) (i : Fin {C}) : coords b i < {M} :=
  getD_lt_of_all (by norm_num) (by revert b; decide) i

''')
    for b in range(B):
        lits = ', '.join(f'({x})' for x in I.coords[b])
        o.append(f'''theorem selectInt_coords{b} (i : Fin {C}) :
    MxxIR.selectInt (Int.ofNat i) [{lits}] = Int.ofNat (coords {b} i) :=
  selectInt_map_ofNat (coordsTable {b}) i

''')
    mL, mL1 = I.m(L), I.m(L - 1)
    o.append(f'''/-! ## Hash tags -/

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
    Fin 2 → Fin 2 → ExactMatrix Q{L} N 1 {mL1} :=
  fun k j ↦ model.sample Q{L} N 1 {mL1} key (completeHashTag statePrefix
    [.bytes componentBytes, .u64Le (2 * (k : Int) + (j : Int))])

/-- The public matrix of member `i` of `a`. -/
noncomputable def memberKey (model : HashModel) (key : ByteArray) (i : Int) :
    Fin 2 → Fin 2 → ExactMatrix Q{L} N 1 {mL} :=
  fun k j ↦ model.sample Q{L} N 1 {mL} key (completeHashTag memberPrefix
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
''')
    for b, (raw, text) in enumerate(I.rhs_prefixes):
        o.append(f'''
/-- `{text}` -/
def rhsPrefix{b} : Blob :=
  {blob(raw)}
''')
    o.append('\nend LwrProof\n')
    return ''.join(o)
