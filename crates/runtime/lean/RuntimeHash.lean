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

end MxxRuntime
