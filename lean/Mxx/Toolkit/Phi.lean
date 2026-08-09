import Mxx.Toolkit.Norms

namespace Mxx.Toolkit

def scaledBit (modulus bit : Int) : Int := (modulus / 2) * bit

/-- Decoder for an integer lift of the two representatives `0` and `q/2`.
The Graph IR decoder additionally reduces its input modulo `q`; protocol proofs can discharge
that reduction separately and reuse this radius lemma. -/
def decodeScaledBitLift (modulus value : Int) : Bool :=
  decide (modulus / 4 ≤ value)

theorem scaledBit_zero (modulus : Int) : scaledBit modulus 0 = 0 := by simp [scaledBit]

theorem scaledBit_one (modulus : Int) : scaledBit modulus 1 = modulus / 2 := by
  simp [scaledBit]

theorem decode_scaledBit_add_error
    (modulus error : Int) (positive : 0 < modulus)
    (divisible : modulus % 4 = 0) (bounded : error.natAbs < (modulus / 4).toNat)
    (message : Bool) :
    decodeScaledBitLift modulus (scaledBit modulus (if message then 1 else 0) + error) =
      message := by
  have radius_positive : 0 < modulus / 4 := by omega
  have radius_cast : ((modulus / 4).toNat : Int) = modulus / 4 :=
    Int.toNat_of_nonneg radius_positive.le
  have cast_bound : (error.natAbs : Int) < ((modulus / 4).toNat : Int) := by
    exact_mod_cast bounded
  rw [radius_cast] at cast_bound
  have lower : -(modulus / 4) < error := by
    omega
  have upper : error < modulus / 4 := by
    omega
  cases message <;> simp [decodeScaledBitLift, scaledBit] <;> omega

/-- A protocol-independent interface for gadget and GGH15-style encoding maps.  The robust
decode law is data attached to the concrete map, rather than a new global axiom. -/
structure RobustEncodingMap (Message Encoded Error : Type) where
  encode : Message → Encoded
  addError : Encoded → Error → Encoded
  decode : Encoded → Message
  errorNorm : Error → Nat
  decodingRadius : Nat
  decodeWithin :
    ∀ message error, errorNorm error < decodingRadius →
      decode (addError (encode message) error) = message

theorem RobustEncodingMap.decode_of_error_bound
    {Message Encoded Error : Type} (encoding : RobustEncodingMap Message Encoded Error)
    (message : Message) (error : Error)
    (bounded : encoding.errorNorm error < encoding.decodingRadius) :
    encoding.decode (encoding.addError (encoding.encode message) error) = message :=
  encoding.decodeWithin message error bounded

/-- The executable `x * G` map used by gadget encodings. -/
def gadgetMap (base : Int) (digits : Nat) (value : Int) : List Int :=
  (List.range digits).map fun digit => value * base ^ digit

/-- The executable `T * A` map used by the GGH15 preimage-product relation. -/
def ggh15Map (multiply : Mxx.Matrix → Mxx.Matrix → Mxx.Matrix)
    (trapdoor publicMatrix : Mxx.Matrix) : Mxx.Matrix :=
  multiply trapdoor publicMatrix

theorem robust_gadget_decode
    (encoding : RobustEncodingMap Int (List Int) (List Int))
    (value : Int) (error : List Int)
    (bounded : encoding.errorNorm error < encoding.decodingRadius) :
    encoding.decode (encoding.addError (encoding.encode value) error) = value :=
  encoding.decode_of_error_bound value error bounded

theorem robust_ggh15_decode
    (encoding : RobustEncodingMap Mxx.Matrix Mxx.Matrix Mxx.Matrix)
    (message error : Mxx.Matrix)
    (bounded : encoding.errorNorm error < encoding.decodingRadius) :
    encoding.decode (encoding.addError (encoding.encode message) error) = message :=
  encoding.decode_of_error_bound message error bounded

end Mxx.Toolkit
