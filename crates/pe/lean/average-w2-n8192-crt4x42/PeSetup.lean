import Claim
import ExponentLut
import ExponentDecode
import ExponentRuntime
import ExponentDynamic
import ExponentFunctional
import PeDynamic

/-!
# The parameter set

The ring `Z_Q[X]/(X^8192 + 1)` with 4 42-bit CRT primes (`Q ≈ 2^168`), gadget base `2^21` with
two digits per prime, so a gadget row has `D = 8` columns and an encoding `m = 16`. The setup
automorphism group is `H = {1, 8193}`, and `8193` is its own inverse modulo `2N`. Every Dynamic
evaluation stays on this one ring.

The public matrices are expanded from the hash key under tags that name the setup schema
(`schemaBytes`) or the parameters (`peContextBytes`).
-/

namespace PeProof

open Mxx.Primitives MxxRuntime MxxExponentLut

abbrev N : Nat := 8192
abbrev Q : Nat := 374144139004015249260017124107828540725582728380417

theorem hN : 0 < N := by decide
theorem hN1 : 1 < N := by decide
theorem hpow : N = 2 ^ 13 := rfl
theorem hQ : 1 < Q := by norm_num
theorem hQodd : Q % 2 = 1 := by norm_num

theorem layoutQ : Backend.backend.regularLayout Q N = some Backend.layout0 := by
  simp [Backend.backend]

/-- The nonidentity automorphism of the setup group. -/
abbrev K : Nat := 8193

theorem hK : Odd K := by decide
theorem hKinv : K * K % (2 * N) = 1 := by decide
theorem hKcoprime : N.Coprime K := by decide

/-! ## Hash tags -/

/-- `component` -/
def componentBytes : Blob := [99, 111, 109, 112, 111, 110, 101, 110, 116]

/-- `rns-bgg/dynamic` -/
def dynamicPrefix : Blob := [114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]

/-- The hex digest of the Dynamic setup schema. -/
def schemaBytes : Blob := [100, 51, 55, 54, 53, 97, 53, 48, 99, 56, 57, 48, 98, 100, 57, 49, 55, 102, 56, 98, 54, 55, 53, 53, 57, 100, 57, 54, 49, 99, 54, 53, 98, 54, 48, 49, 52, 56, 56, 53, 101, 99, 53, 54, 98, 48, 53, 49, 102, 99, 97, 102, 52, 97, 53, 101, 52, 57, 48, 97, 55, 50, 50, 49]

/-- `mxx-pe` -/
def pePrefix : Blob := [109, 120, 120, 45, 112, 101]

/-- The hex digest of the setup context, which names the parameters. -/
def peContextBytes : Blob := [54, 55, 98, 97, 98, 99, 52, 54, 48, 98, 53, 49, 57, 55, 101, 102, 100, 56, 54, 97, 99, 56, 52, 98, 53, 50, 56, 55, 50, 54, 99, 50, 98, 97, 52, 55, 102, 53, 56, 56, 48, 48, 55, 102, 101, 101, 97, 56, 48, 57, 99, 52, 55, 54, 101, 49, 57, 100, 99, 98, 56, 56, 56, 55]

/-- The public key `A_0` of the initial state. -/
noncomputable def inputKey (model : HashModel) (key : ByteArray) : Fin 2 → Fin 2 → ExactMatrix Q N 1 8 :=
  fun k j ↦ model.sample Q N 1 8 key (completeHashTag dynamicPrefix
    [.bytes schemaBytes, .bytes [105, 110, 112, 117, 116, 45, 107, 101, 121], .u64Le 0,
      .bytes componentBytes, .u64Le (2 * (k : Int) + (j : Int))])

/-- The helper key `H_K`. -/
noncomputable def helperKey (model : HashModel) (key : ByteArray) : Fin 2 → Fin 2 → ExactMatrix Q N 1 8 :=
  fun k j ↦ model.sample Q N 1 8 key (completeHashTag dynamicPrefix
    [.bytes schemaBytes, .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le (K : Int),
      .bytes componentBytes, .u64Le (2 * (k : Int) + (j : Int))])

/-- The message row `d = (d_1, d_2)`. -/
noncomputable def messageRow (model : HashModel) (key : ByteArray) (index : Fin 2) : ExactMatrix Q N 1 1 :=
  model.sample Q N 1 1 key (completeHashTag pePrefix
    [.bytes peContextBytes, .bytes [109, 101, 115, 115, 97, 103, 101, 45, 114, 111, 119],
      .u64Le (index : Int)])

/-- The second anchor row `B^(2)`. -/
noncomputable def anchorSecond (model : HashModel) (key : ByteArray) : ExactMatrix Q N 1 10 :=
  model.sample Q N 1 10 key (completeHashTag pePrefix
    [.bytes peContextBytes, .bytes [97, 110, 99, 104, 111, 114, 45, 115, 101, 99, 111, 110, 100, 45,
      114, 111, 119]])

/-- The digits of entry `a` of the joined key row `[A_r0 | A_r1]`. -/
noncomputable def keyDigits (A : Fin 2 → Fin 2 → ExactMatrix Q N 1 8) (a : Fin (8 + 8)) (r : Fin 2) :
    ExactMatrix Q N 8 1 :=
  entryDigits 8 Backend.backend 2097152 8 (fun _ _ ↦ joinedEntry (A r) a)

end PeProof
