import RuntimePrimitives

namespace MxxRuntime

abbrev Blob := List UInt8

/-- One fixed interpretation for all plain hash requests in a linked execution. The backend
algorithm is an explicit abstract boundary; key, complete encoded tag and geometry are not. -/
structure HashModel where
  sample : (q n rows columns : Nat) → ByteArray → Blob →
    Mxx.Primitives.ExactMatrix q n rows columns
  /-- Integer families keyed like `sample`: the runtime truncates the digest stream of each
  coefficient to `log2 modulus` bits, so the result need not relate to `sample`'s residues. -/
  integers : (count modulus : Nat) → ByteArray → Blob → Fin count → Nat

def u64LittleEndian (value : Nat) : Blob :=
  (List.range 8).map (fun index ↦ UInt8.ofNat (value / 256 ^ index))

-- num-bigint 0.4 to_bytes_be returns a one-byte magnitude [0] for zero.
def signedIntegerTag (value : Int) : Blob :=
  let magnitude := if value.natAbs = 0 then [0] else (Nat.digits 256 value.natAbs).reverse
  [if value < 0 then 1 else 0] ++ (u64LittleEndian magnitude.length).reverse ++
    magnitude.map UInt8.ofNat

def decimalIntegerTag (value : Int) : Blob :=
  let digits := if value.natAbs = 0 then [0] else (Nat.digits 10 value.natAbs).reverse
  (if value < 0 then [45] else []) ++ digits.map (fun digit ↦ UInt8.ofNat (48 + digit))

/-- Ordered, typed tag components. The explicit namespace prefix is chosen by the caller. -/
inductive HashTagComponent where
  | bytes : Blob → HashTagComponent
  | integer : Int → HashTagComponent
  | decimal : Int → HashTagComponent
  | u64Le : Int → HashTagComponent

def encodeHashTagComponent : HashTagComponent → Blob
  | .bytes bytes => [0] ++ (u64LittleEndian bytes.length).reverse ++ bytes
  | .integer value => [1] ++ signedIntegerTag value
  | .decimal value =>
    let bytes := decimalIntegerTag value
    [2] ++ (u64LittleEndian bytes.length).reverse ++ bytes
  | .u64Le value => [3] ++ u64LittleEndian value.toNat

def completeHashTag (tagPrefix : Blob) (components : List HashTagComponent) : Blob :=
  tagPrefix ++ components.flatMap encodeHashTagComponent

noncomputable def hashSample {q n rows columns : Nat} (model : HashModel)
    (tagPrefix : Blob) (components : List HashTagComponent) (key : ByteArray)
    (output : Mxx.Primitives.ExactMatrix q n rows columns) : Prop :=
  key.size = 32 ∧ (∀ value, .u64Le value ∈ components → 0 ≤ value ∧ value < 2 ^ 64) ∧
  output = model.sample q n rows columns key (completeHashTag tagPrefix components)

/-- `count` integers on `[0, modulus)` for a power-of-two modulus above one. The reduction is the
identity on the runtime's truncated values and keeps every model inhabited. -/
def hashIntFamily {count : Nat} (model : HashModel) (modulus : Int)
    (tagPrefix : Blob) (components : List HashTagComponent) (key : ByteArray)
    (output : Fin count → Int) : Prop :=
  key.size = 32 ∧ (∀ value, .u64Le value ∈ components → 0 ≤ value ∧ value < 2 ^ 64) ∧
  (∃ bits : Nat, 0 < bits ∧ modulus = 2 ^ bits) ∧
  output = fun index ↦
    ((model.integers count modulus.toNat key (completeHashTag tagPrefix components) index %
      modulus.toNat : Nat) : Int)

namespace Sha256Columns

def initial : Array UInt32 := #[
  0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
  0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19]

def constants : Array UInt32 := #[
  0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
  0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
  0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
  0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
  0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
  0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
  0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
  0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2]

def rotate (value : UInt32) (bits : UInt32) : UInt32 :=
  (value >>> bits) ||| (value <<< (32 - bits))

/-- SHA-256 of exactly a 32-byte seed followed by a little-endian 64-bit counter.
UInt32 arithmetic supplies the SHA-256 additions modulo 2^32. -/
def digest (seed : Blob) (counter : Nat) : Array UInt32 := Id.run do
  let bytes := (seed ++ u64LittleEndian counter ++ [128] ++ List.replicate 15 0 ++
    [0, 0, 0, 0, 0, 0, 1, 64]).toArray
  let mut words : Array UInt32 := #[]
  for index in List.range 16 do
    words := words.push ((List.range 4).foldl (fun value byte ↦
      value * 256 + UInt32.ofNat (bytes.getD (index * 4 + byte) 0).toNat) 0)
  for offset in List.range 48 do
    let index := offset + 16
    let left := words.getD (index - 15) 0
    let right := words.getD (index - 2) 0
    let sigma0 := rotate left 7 ^^^ rotate left 18 ^^^ (left >>> 3)
    let sigma1 := rotate right 17 ^^^ rotate right 19 ^^^ (right >>> 10)
    words := words.push (words.getD (index - 16) 0 + sigma0 + words.getD (index - 7) 0 + sigma1)
  let mut state := initial
  for index in List.range 64 do
    let stateA := state.getD 0 0
    let stateB := state.getD 1 0
    let stateC := state.getD 2 0
    let stateD := state.getD 3 0
    let stateE := state.getD 4 0
    let stateF := state.getD 5 0
    let stateG := state.getD 6 0
    let stateH := state.getD 7 0
    let upper := rotate stateE 6 ^^^ rotate stateE 11 ^^^ rotate stateE 25
    let choose := (stateE &&& stateF) ^^^ ((~~~stateE) &&& stateG)
    let first := stateH + upper + choose + constants.getD index 0 + words.getD index 0
    let lower := rotate stateA 2 ^^^ rotate stateA 13 ^^^ rotate stateA 22
    let majority := (stateA &&& stateB) ^^^ (stateA &&& stateC) ^^^ (stateB &&& stateC)
    state := #[first + lower + majority, stateA, stateB, stateC, stateD + first,
      stateE, stateF, stateG]
  return (List.range 8).toArray.map (fun index ↦ initial.getD index 0 + state.getD index 0)

def bit (seed : Blob) (index : Nat) : Bool :=
  let words := digest seed (index / 256)
  let byte := index % 256 / 8
  (words.getD (byte / 4) 0).toNat.testBit ((3 - byte % 4) * 8 + index % 8)

/-- Ascending selected rows, packed into fixed-width fields, and the number of selected rows. -/
def column (seed : Blob) (rows columns indexBits selectedColumn : Nat) : Nat × Nat :=
  (List.range rows).foldl (fun result row ↦
    if bit seed (row * columns + selectedColumn) then
      (result.1 + row * 2 ^ (result.2 * indexBits), result.2 + 1)
    else result) (0, 0)

end Sha256Columns

/-- Exact public SHA-256 selector families, not an abstract hash-model assumption. -/
def sha256BitColumns (seed : Blob) (rows columns indexBits : Nat)
    (packed counts : Fin columns → Int) : Prop :=
  seed.length = 32 ∧ 0 < rows ∧ 0 < columns ∧ 0 < indexBits ∧ rows ≤ 2 ^ indexBits ∧
  ∀ index, let result := Sha256Columns.column seed rows columns indexBits index.val
    packed index = (result.1 : Int) ∧ counts index = (result.2 : Int)

end MxxRuntime
