import RuntimePrimitives

namespace MxxRuntime

open Mxx.Primitives
open scoped BigOperators

noncomputable def matrixAdd {q n r c : Nat} (left right : ExactMatrix q n r c) := left + right
noncomputable def matrixSub {q n r c : Nat} (left right : ExactMatrix q n r c) := left - right
noncomputable def matrixMul {q n r k c : Nat}
    (left : ExactMatrix q n r k) (right : ExactMatrix q n k c) := left * right
noncomputable def matrixMulScalarLeft {q n r c : Nat}
    (left : ExactMatrix q n 1 1) (right : ExactMatrix q n r c) : ExactMatrix q n r c :=
  fun i j ↦ left 0 0 * right i j
noncomputable def matrixMulScalarRight {q n r c : Nat}
    (left : ExactMatrix q n r c) (right : ExactMatrix q n 1 1) : ExactMatrix q n r c :=
  fun i j ↦ left i j * right 0 0

@[simp] theorem matrixMulScalarLeft_one {q n c : Nat}
    (left : ExactMatrix q n 1 1) (right : ExactMatrix q n 1 c) :
    matrixMulScalarLeft left right = left * right := by
  ext i j
  fin_cases i
  simp [matrixMulScalarLeft, Matrix.mul_apply]

@[simp] theorem matrixMulScalarRight_one {q n r : Nat}
    (left : ExactMatrix q n r 1) (right : ExactMatrix q n 1 1) :
    matrixMulScalarRight left right = left * right := by
  ext i j
  fin_cases j
  simp [matrixMulScalarRight, Matrix.mul_apply]
noncomputable def matrixNeg {q n r c : Nat} (value : ExactMatrix q n r c) := -value
noncomputable def matrixScale {q n r c : Nat} (scalar : Int) (value : ExactMatrix q n r c) :
    ExactMatrix q n r c := fun i j ↦ (scalar : ExactPoly q n) * value i j
def transpose {q n r c : Nat} (value : ExactMatrix q n r c) := value.transpose
def trapdoorPublic {Public Token : Type} (value : TrapdoorValue Public Token) := value.publicMatrix

def familyGetDynamic {α : Type} {count : Nat} (family : Fin count → α) (index : Int)
    (output : α) : Prop :=
  ∃ position : Fin count, (position.val : Int) = index ∧ output = family position

def familyGetStatic {α : Type} {count : Nat} := @familyGetDynamic α count

def select {α : Type} (index : Int) (choices : List α) (output : α) : Prop :=
  ∃ position : Fin choices.length, (position.val : Int) = index ∧ output = choices.get position

def familyPack {α : Type} {size : Nat} (count : Int) (values : List α)
    (output : Fin size → α) : Prop :=
  count = size ∧ values.length = size ∧
  ∀ index : Fin size, ∃ position : Fin values.length,
    position.val = index.val ∧ output index = values.get position

def uniformResidueSample {q n rows columns : Nat}
    (output : ExactMatrix q n rows columns) : Prop := output = output

def uniformIntervalSample {q n rows columns : Nat} (minimum maximum : Int)
    (output : ExactMatrix q n rows columns) : Prop :=
  minimum ≤ maximum ∧ ∃ witness : ErrorMatrix n rows columns,
    output = reduceMatrix q n rows columns witness ∧
    ∀ row column coefficient, minimum ≤ (witness row column).coeff coefficient ∧
      (witness row column).coeff coefficient ≤ maximum

def gaussianSample {q n rows columns : Nat} (sigma : Rat) (cutoff : Int)
    (output : ExactMatrix q n rows columns) : Prop :=
  0 ≤ sigma ∧ 0 ≤ cutoff ∧ PreimageWithin output cutoff.toNat ∧ (sigma = 0 → output = 0)

/-- The backend returns the canonical residue, not a centered integer. -/
noncomputable def extractCoefficient {q n : Nat} (position : Int)
    (input : ExactMatrix q n 1 1) (output : Int) : Prop :=
  ∃ index : Fin n, (index.val : Int) = position ∧
    output = ((input 0 0).coeff index).val

/-- Decode one of the first `length` canonical coefficients using the runtime's nearest-integer
rounding, followed by reduction modulo the plaintext modulus. Boolean ports separately test whether
this integer is nonzero. Positive divisors make integer division agree with the backend formula. -/
noncomputable def thresholdDecode {q n : Nat} (plaintextModulus length position : Int)
    (input : ExactMatrix q n 1 1) (output : Int) : Prop :=
  0 < q ∧ 0 < plaintextModulus ∧ 0 ≤ length ∧ length ≤ (n : Int) ∧
  ∃ index : Fin n, (index.val : Int) = position ∧ position < length ∧
    output = ((plaintextModulus * (((input 0 0).coeff index).val : Int) + (q : Int) / 2) /
      (q : Int)) % plaintextModulus

-- Boundary regressions exercise the relation itself, including the half-step tie and wraparound.
example (input : ExactMatrix 256 1 1 1) (h : ((input 0 0).coeff 0).val = 63) (out : Int) :
    thresholdDecode 2 1 0 input out ↔ out = 0 := by
  simp only [thresholdDecode, Fin.exists_fin_one, Fin.val_zero, h]
  norm_num

example (input : ExactMatrix 256 1 1 1) (h : ((input 0 0).coeff 0).val = 64) (out : Int) :
    thresholdDecode 2 1 0 input out ↔ out = 1 := by
  simp only [thresholdDecode, Fin.exists_fin_one, Fin.val_zero, h]
  norm_num

example (input : ExactMatrix 256 1 1 1) (h : ((input 0 0).coeff 0).val = 191) (out : Int) :
    thresholdDecode 2 1 0 input out ↔ out = 1 := by
  simp only [thresholdDecode, Fin.exists_fin_one, Fin.val_zero, h]
  norm_num

example (input : ExactMatrix 256 1 1 1) (h : ((input 0 0).coeff 0).val = 192) (out : Int) :
    thresholdDecode 2 1 0 input out ↔ out = 0 := by
  simp only [thresholdDecode, Fin.exists_fin_one, Fin.val_zero, h]
  norm_num

-- For plaintext modulus three, integer two must also decode to Boolean true.
example (input : ExactMatrix 256 1 1 1) (h : ((input 0 0).coeff 0).val = 171) (out : Int)
    (hrun : thresholdDecode 3 1 0 input out) : decide (out ≠ 0) = true := by
  simp only [thresholdDecode, Fin.exists_fin_one, Fin.val_zero, h] at hrun
  norm_num at hrun
  simp [hrun]

/-- A literal coefficient list in increasing degree order, reduced in the exact residue ring.
The exporter validates the scalar shape and the list length against the ring dimension. -/
noncomputable def matrixPolynomial {q n : Nat} (coefficients : List Int) :
    ExactMatrix q n 1 1 :=
  fun _ _ ↦ coefficients.foldr
    (fun coefficient tail ↦ (coefficient : ExactPoly q n) +
      AdjoinRoot.root (negacyclicModulus n (ZMod q)) * tail) 0

/- Pure matrix relations used by generic frozen-IR extraction.  The dimensions are carried by the
   matrix types; the relations only describe the coefficient-wise operation and its range guards. -/
def sliceMatrix {q n rows columns outRows outColumns : Nat}
    (input : ExactMatrix q n rows columns)
    (rowStart rowEnd columnStart columnEnd : Int)
    (output : ExactMatrix q n outRows outColumns) : Prop :=
  0 ≤ rowStart ∧ rowStart ≤ rowEnd ∧ rowEnd ≤ Int.ofNat rows ∧
  0 ≤ columnStart ∧ columnStart ≤ columnEnd ∧ columnEnd ≤ Int.ofNat columns ∧
  rowEnd - rowStart = Int.ofNat outRows ∧
  columnEnd - columnStart = Int.ofNat outColumns ∧
  ∀ row column hRow hColumn,
    output row column =
      input
        ⟨rowStart.toNat + row.val, hRow⟩
        ⟨columnStart.toNat + column.val, hColumn⟩

def concatRows {q n leftRows rightRows columns : Nat}
    (left : ExactMatrix q n leftRows columns)
    (right : ExactMatrix q n rightRows columns)
    (output : ExactMatrix q n (leftRows + rightRows) columns) : Prop :=
  ∀ row column,
    output row column =
      if h : row.val < leftRows then
        left ⟨row.val, h⟩ column
      else
        right ⟨row.val - leftRows, by omega⟩ column

def concatColumns {q n rows leftColumns rightColumns : Nat}
    (left : ExactMatrix q n rows leftColumns)
    (right : ExactMatrix q n rows rightColumns)
    (output : ExactMatrix q n rows (leftColumns + rightColumns)) : Prop :=
  ∀ row column,
    output row column =
      if h : column.val < leftColumns then
        left row ⟨column.val, h⟩
      else
        right row ⟨column.val - leftColumns, by omega⟩

def concatDiagonal {q n leftRows leftColumns rightRows rightColumns : Nat}
    (left : ExactMatrix q n leftRows leftColumns)
    (right : ExactMatrix q n rightRows rightColumns)
    (output : ExactMatrix q n (leftRows + rightRows) (leftColumns + rightColumns)) : Prop :=
  ∀ row column,
    output row column =
      if hRow : row.val < leftRows then
        if hColumn : column.val < leftColumns then
          left ⟨row.val, hRow⟩ ⟨column.val, hColumn⟩
        else
          0
      else if hColumn : column.val < leftColumns then
        0
      else
        right ⟨row.val - leftRows, by omega⟩ ⟨column.val - leftColumns, by omega⟩

/-- Reverse exactly `bits` low bits, matching the native NTT evaluation-slot order. -/
def nttBitReverse (bits index : Nat) : Nat :=
  ∑ bit ∈ Finset.range bits, (index / 2 ^ bit % 2) * 2 ^ (bits - 1 - bit)

/-- A canonical primitive root with the order used by the native negacyclic transform. -/
def nttPrimitiveRoot (q n root : Nat) : Prop :=
  0 < root ∧ root < q ∧ root ^ (2 * n) % q = 1 ∧
  ∀ exponent : Nat, 0 < exponent → exponent < 2 * n → root ^ exponent % q ≠ 1

/-- Evaluate at odd powers of `root`, writing evaluations in bit-reversed order.
Both sides are reduced, so inverse inputs may use noncanonical integer representatives. -/
def nttEvaluations {n : Nat} (q bits root : Nat)
    (coefficients evaluations : Fin n → Int) : Prop :=
  ∀ index : Fin n, ∃ slot : Fin n,
    slot.val = nttBitReverse bits index.val ∧
    evaluations slot % (q : Int) =
      (∑ coefficient : Fin n,
        coefficients coefficient * (root : Int) ^ ((2 * index.val + 1) * coefficient.val)) %
        (q : Int)

/-- Native negacyclic NTT semantics, including CRT rings with distinct prime limbs. Each
prime limb chooses its smallest primitive `2*n`-th root; `root` is their canonical CRT lift.
Forward maps coefficients to bit-reversed evaluations; inverse reverses that same relation.
In either direction the returned integers are canonical residues modulo the full modulus. -/
def polynomialNttRuns {n : Nat} (q : Nat) (inverse : Bool)
    (input output : Fin n → Int) : Prop :=
  1 < q ∧
  (∀ prime : Nat, prime.Prime → ¬ (prime * prime) ∣ q) ∧
  ∃ bits root : Nat, n = 2 ^ bits ∧ root < q ∧
    (∀ prime : Nat, prime.Prime → prime ∣ q →
      (2 * n) ∣ (prime - 1) ∧ nttPrimitiveRoot prime n (root % prime) ∧
      ∀ candidate : Nat, nttPrimitiveRoot prime n candidate → root % prime ≤ candidate) ∧
    (∀ index, 0 ≤ output index ∧ output index < (q : Int)) ∧
    (if inverse then nttEvaluations q bits root output input
     else nttEvaluations q bits root input output)

/-- Canonical coefficients reconstructed in the quotient by `X^n + 1`. -/
noncomputable def polynomialOfCoefficients {q n : Nat} (values : Fin n → Int) :
    ExactPoly q n :=
  ∑ i : Fin n, (values i : ExactPoly q n) *
    AdjoinRoot.root (negacyclicModulus n (ZMod q)) ^ i.val

/-- Runtime import reduces arbitrary integers; evaluation imports use the native inverse NTT. -/
noncomputable def polynomialFromValues {q n : Nat} (evaluation : Bool)
    (input : Fin n → Int) (output : ExactMatrix q n 1 1) : Prop :=
  if evaluation then polynomialNttRuns q true input (fun i ↦ ((output 0 0).coeff i).val)
  else output = fun _ _ ↦ polynomialOfCoefficients input

/-- Runtime export returns canonical residues, in coefficient or native evaluation order. -/
noncomputable def polynomialValues {q n : Nat} (evaluation : Bool)
    (input : ExactMatrix q n 1 1) (output : Fin n → Int) : Prop :=
  if evaluation then polynomialNttRuns q false (fun i ↦ ((input 0 0).coeff i).val) output
  else ∀ i, output i = ((input 0 0).coeff i).val

theorem polynomialOfCoefficients_coeff {q n : Nat} (hq : 1 < q) (hn : 0 < n)
    (values : Fin n → Int) (i : Fin n) :
    (polynomialOfCoefficients (q := q) values).coeff i = (values i : ZMod q) := by
  letI : Fact (1 < q) := ⟨hq⟩
  unfold polynomialOfCoefficients
  rw [Negacyclic.coeff_sum]
  have hcast (value : Int) : (value : ExactPoly q n) =
      algebraMap (ZMod q) (ExactPoly q n) (value : ZMod q) := by simp
  simp_rw [hcast, Negacyclic.coeff_smul, Negacyclic.coeff_root_pow hn]
  simp

/-- The runtime scales canonical residues and rounds half upward before reduction. -/
noncomputable def modulusSwitch {q p n rows columns : Nat}
    (input : ExactMatrix q n rows columns) : ExactMatrix p n rows columns :=
  fun row column ↦ polynomialOfCoefficients fun i ↦
    ((Int.ofNat ((input row column).coeff i).val * Int.ofNat p + Int.ofNat (q / 2)) /
      Int.ofNat q) % Int.ofNat p

noncomputable def modulusSwitchRuns {q p n rows columns : Nat}
    (input : ExactMatrix q n rows columns) (output : ExactMatrix p n rows columns) : Prop :=
  1 < p ∧ p ∣ q ∧ q % 2 = 1 ∧ p % 2 = 1 ∧ output = modulusSwitch input

/-- Ordinary reduction uses the canonical representative without scaling. -/
noncomputable def modulusReduce {q p n rows columns : Nat}
    (input : ExactMatrix q n rows columns) : ExactMatrix p n rows columns :=
  fun row column ↦ polynomialOfCoefficients fun i ↦ Int.ofNat ((input row column).coeff i).val

noncomputable def modulusReduceRuns {q p n rows columns : Nat}
    (input : ExactMatrix q n rows columns) (output : ExactMatrix p n rows columns) : Prop :=
  1 < p ∧ p ∣ q ∧ output = modulusReduce input

/-- Centered representatives are re-encoded without requiring a divisor ring. -/
noncomputable def centeredRebase {q p n rows columns : Nat}
    (input : ExactMatrix q n rows columns) : ExactMatrix p n rows columns :=
  fun row column ↦ polynomialOfCoefficients fun i ↦
    let u := ((input row column).coeff i).val
    if u ≤ q / 2 then Int.ofNat u else Int.ofNat u - Int.ofNat q

noncomputable def centeredRebaseRuns {q p n rows columns : Nat}
    (input : ExactMatrix q n rows columns) (output : ExactMatrix p n rows columns) : Prop :=
  1 < p ∧ output = centeredRebase input

/-- The inverse is taken in the native residue ring, before centered lifting. -/
noncomputable def rnsInverse (modulus : Nat) (value : Int) : Int :=
  Int.ofNat (((value : ZMod modulus)⁻¹).val)

noncomputable def rnsCentered (modulus : Nat) (value : Int) : Int :=
  let residue := value % Int.ofNat modulus
  if residue ≤ Int.ofNat (modulus / 2) then residue else residue - Int.ofNat modulus

/-- One contiguous CRT digit, using the same unreduced centered sum as the backend. -/
noncomputable def rnsDigit (basis : List Nat) (digitSize digit : Nat)
    (normalize : Bool) (value : Int) : Int :=
  let group := (basis.drop (digit * digitSize)).take digitSize
  let digitModulus := group.prod
  (group.map fun prime ↦
    let factor := digitModulus / prime
    let normalization := if normalize then rnsInverse prime (basis.prod / digitModulus) else 1
    Int.ofNat factor * rnsCentered prime
      (value * rnsInverse prime factor * normalization)).sum

/-- Group-major row stacking fixes both the CRT formula and the digit layout. -/
noncomputable def rnsModUpRuns {q p n rows columns outputRows : Nat}
    (basis : List Nat) (digitSize : Nat) (normalize : Bool)
    (input : ExactMatrix q n rows columns)
    (output : ExactMatrix p n outputRows columns) : Prop :=
  basis.prod = q ∧ basis ≠ [] ∧ basis.Pairwise Nat.Coprime ∧
  (∀ prime ∈ basis, 2 < prime ∧ (2 * n) ∣ (prime - 1)) ∧
  0 < digitSize ∧ 1 < p ∧ q ∣ p ∧
  outputRows = rows * ((basis.length + digitSize - 1) / digitSize) ∧
  ∀ (digit : Fin ((basis.length + digitSize - 1) / digitSize))
    (row : Fin rows) (outRow : Fin outputRows) (column : Fin columns),
    outRow.val = digit.val * rows + row.val →
    output outRow column = polynomialOfCoefficients (fun i ↦
      rnsDigit basis digitSize digit.val normalize
        (Int.ofNat ((input row column).coeff i).val))

/-- The dropped basis computes the centered BGV correction without exact CRT reduction. -/
noncomputable def rnsModDownRuns {q p n rows columns : Nat}
    (basis : List Nat) (plaintext : Int)
    (input : ExactMatrix q n rows columns) (output : ExactMatrix p n rows columns) : Prop :=
  let dropped := basis.filter (fun prime ↦ p % prime != 0)
  let auxiliary := dropped.prod
  basis.prod = q ∧ basis ≠ [] ∧ basis.Pairwise Nat.Coprime ∧
  (∀ prime ∈ basis, 2 < prime ∧ (2 * n) ∣ (prime - 1)) ∧
  1 < p ∧ p < q ∧ p * auxiliary = q ∧ 2 ≤ plaintext ∧
  Int.gcd plaintext (Int.ofNat auxiliary) = 1 ∧
  ∀ row column, output row column = polynomialOfCoefficients (fun i ↦
    let value := Int.ofNat ((input row column).coeff i).val
    let correction := (dropped.map fun prime ↦
      let factor := auxiliary / prime
      Int.ofNat factor * rnsCentered prime
        (-value * rnsInverse prime plaintext * rnsInverse prime factor)).sum
    (value + plaintext * correction) * rnsInverse p auxiliary)

/-- Substitution in the negacyclic quotient incorporates the runtime's wraparound sign. -/
noncomputable def ringAutomorphism {q n rows columns : Nat} (index : Nat)
    (input : ExactMatrix q n rows columns) : ExactMatrix q n rows columns :=
  fun row column ↦ ∑ i : Fin n,
    (Int.ofNat ((input row column).coeff i).val : ExactPoly q n) *
      AdjoinRoot.root (negacyclicModulus n (ZMod q)) ^ (i.val * index)

noncomputable def ringAutomorphismRuns {q n rows columns : Nat} (index : Int)
    (input output : ExactMatrix q n rows columns) : Prop :=
  0 < index ∧ index < 2 * Int.ofNat n ∧ index % 2 = 1 ∧
  output = ringAutomorphism index.toNat input

/-- One heterogeneous CRT level: round using its own source modulus, reduce to the
plaintext modulus, lift the canonical digit to the destination, and multiply by the
reconstruction coefficient. This follows `crt_recompose_cpu` term for term. -/
noncomputable def crtRecomposeLevel {source destination n columns : Nat}
    (plaintext coefficient : Int) (input : ExactMatrix source n 1 columns) :
    ExactMatrix destination n 1 columns :=
  fun row column ↦ (coefficient : ExactPoly destination n) *
    polynomialOfCoefficients fun i ↦
      ((plaintext * Int.ofNat ((input row column).coeff i).val + Int.ofNat (source / 2)) /
        Int.ofNat source) % plaintext

noncomputable def unitRow {q n columns : Nat} (index : Int) : ExactMatrix q n 1 columns :=
  fun _ column ↦ if (column.val : Int) = index then 1 else 0

noncomputable def unitColumn {q n rows : Nat} (index : Int) : ExactMatrix q n rows 1 :=
  fun row _ ↦ if (row.val : Int) = index then 1 else 0

noncomputable def rotationPolynomial {q n : Nat} (exponent : Int) : ExactMatrix q n 1 1 :=
  fun _ _ ↦ AdjoinRoot.root (negacyclicModulus n (ZMod q)) ^ exponent.toNat

noncomputable def liftInteger {q n : Nat} (value : Int) : ExactMatrix q n 1 1 :=
  fun _ _ ↦ (value : ExactPoly q n)

/-- Tensor layout uses left indices as the outer blocks, matching the runtime. -/
def tensorRuns {q n r₁ r₂ c₁ c₂ : Nat}
    (left : ExactMatrix q n r₁ c₁) (right : ExactMatrix q n r₂ c₂)
    (output : ExactMatrix q n (r₁ * r₂) (c₁ * c₂)) : Prop :=
  ∀ i₁ i₂ j₁ j₂,
    output ⟨i₁.val * r₂ + i₂.val, by have := i₁.isLt; have := i₂.isLt; nlinarith⟩
      ⟨j₁.val * c₂ + j₂.val, by have := j₁.isLt; have := j₂.isLt; nlinarith⟩ = left i₁ j₁ * right i₂ j₂

/-- Packed bits are coefficient-major, little-endian, and must encode canonical residues. -/
noncomputable def packPolynomial {q n count : Nat} (coefficientBits : Int)
    (bits : Fin count → Bool) (output : ExactMatrix q n 1 1) : Prop :=
  0 < coefficientBits ∧ count = n * coefficientBits.toNat ∧
  ∃ coefficients : Fin n → Nat,
    (∀ i, coefficients i < q ∧ coefficients i =
      ∑ bit : Fin coefficientBits.toNat,
        if h : i.val * coefficientBits.toNat + bit.val < count then
          if bits ⟨i.val * coefficientBits.toNat + bit.val, h⟩ then 2 ^ bit.val else 0
        else 0) ∧
    output = fun _ _ ↦ polynomialOfCoefficients (fun i ↦ Int.ofNat (coefficients i))

end MxxRuntime
