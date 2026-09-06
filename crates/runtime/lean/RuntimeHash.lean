import RuntimePrimitives

namespace MxxRuntime

abbrev Blob := List UInt8

/-- One fixed interpretation for all plain hash requests in a linked execution. The backend
algorithm is an explicit abstract boundary; key, complete encoded tag and geometry are not. -/
structure HashModel where
  sample : (q n rows columns : Nat) → ByteArray → Blob →
    Mxx.Primitives.ExactMatrix q n rows columns

def u64LittleEndian (value : Nat) : Blob :=
  (List.range 8).map (fun index ↦ UInt8.ofNat (value / 256 ^ index))

def signedIntegerTag (value : Int) : Blob :=
  let magnitude := if value.natAbs = 0 then [0] else (Nat.digits 256 value.natAbs).reverse
  [if value < 0 then 1 else 0] ++ (u64LittleEndian magnitude.length).reverse ++
    magnitude.map UInt8.ofNat

def decimalIntegerTag (value : Int) : Blob :=
  let digits := if value.natAbs = 0 then [0] else (Nat.digits 10 value.natAbs).reverse
  (if value < 0 then [45] else []) ++ digits.map (fun digit ↦ UInt8.ofNat (48 + digit))

/-- Byte order agrees with the executor: prefix, signed compile-time integers, decimal
integers, u64 little-endian integers, and finally signed integer wire operands. -/
def completeHashTag (tagPrefix : Blob) (integers decimals u64s operands : List Int) : Blob :=
  tagPrefix ++ integers.flatMap signedIntegerTag ++
    decimals.flatMap decimalIntegerTag ++
    u64s.flatMap (fun value ↦ u64LittleEndian value.toNat) ++
    operands.flatMap signedIntegerTag

noncomputable def hashSample {q n rows columns : Nat} (model : HashModel)
    (tagPrefix : Blob) (integers decimals u64s operands : List Int) (key : ByteArray)
    (output : Mxx.Primitives.ExactMatrix q n rows columns) : Prop :=
  key.size = 32 ∧ (∀ value ∈ u64s, 0 ≤ value ∧ value < 2 ^ 64) ∧
  output = model.sample q n rows columns key
    (completeHashTag tagPrefix integers decimals u64s operands)

end MxxRuntime
