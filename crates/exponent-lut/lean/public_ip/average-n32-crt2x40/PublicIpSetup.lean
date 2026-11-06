import Claim
import ExponentDecode
import ExponentPublic
import ExponentRuntime
import ExponentIterate
import PublicIpProgram

/-!
# The parameter set

The modulus `Q = q₀ q₁` of two 40-bit primes, its gadget layout, and the hash tags of the generated
public inner-product claim. The Program has one level: every encoding lives at `Q` with the native
gadget of `Q` (two digits of `2^20` per prime, four columns). The public vector has `m = 4`
coordinates and the hash-sampled `t = (1, 1, 0, 1)` selects coordinates `0`, `1`, and `3`; member `4`
of the input family is the encoding of zero.
-/

namespace PublicIpProof

open Mxx.Primitives MxxRuntime MxxExponentLut MxxExponentLutPublicIp

abbrev N : Nat := 32
abbrev Q : Nat := 1208925815113228574679041

theorem hN : 0 < N := by decide
theorem hN1 : 1 < N := by decide
theorem hQ : 1 < Q := by norm_num
theorem hQodd : Q % 2 = 1 := by norm_num

theorem layoutQ : Backend.backend.regularLayout Q N = some Backend.layout0 := by
  simp [Backend.backend]

/-- A family read at a static index. -/
theorem familyGetStatic_eq {α : Type} {count : Nat} {f : Fin count → α} {k : Nat} {out : α}
    (h : familyGetStatic f (Int.ofNat k) out) : ∃ hk : k < count, out = f ⟨k, hk⟩ :=
  familyGet_eq h

/-! ## Hash tags -/

/-- `component` -/
def componentBytes : Blob := [99, 111, 109, 112, 111, 110, 101, 110, 116]

/-- `member` -/
def memberBytes : Blob := [109, 101, 109, 98, 101, 114]

/-- `rns-bgg/input/WireId(0)/0`: the public input family. -/
def memberPrefix : Blob :=
  [114, 110, 115, 45, 98, 103, 103, 47, 105, 110, 112, 117, 116, 47, 87, 105, 114, 101, 73, 100,
    40, 48, 41, 47, 48]

/-- The public matrix of member `i` of the input family. -/
noncomputable def memberKey (model : HashModel) (key : ByteArray) (i : Int) :
    Fin 2 → Fin 2 → ExactMatrix Q N 1 4 :=
  fun k j ↦ model.sample Q N 1 4 key (completeHashTag memberPrefix
    [.bytes memberBytes, .u64Le i, .bytes componentBytes, .u64Le (2 * (k : Int) + (j : Int))])

end PublicIpProof
