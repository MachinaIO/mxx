import MathcalA.Trusted.Game

/-!
# Running a proof image as an algorithm

`Algorithm.run` fixes how a proof image is run as one algorithm of a protocol: its inputs are
answered to `QUERY` requests `0x00 ++ u64 i` with input `i` (the empty string when `i` is out of
range), any other request is answered with the empty string, `COINS` is uniform, and the output is
the `HALT` payload decoded as a list of byte strings, each a little-endian `u64` length followed by
its bytes. A fault, a timeout, or a payload that does not decode gives `none`.

This definition is part of the meaning of every claim about proof images, so it is reviewed like
trusted code (`ARCHITECTURE-SPEC.md`, section 6.3).
-/

namespace MxxCore

open MathcalA

/-- A proof image and the resources it runs with. -/
structure Algorithm where
  image : Bytes
  bounds : Bounds

/-- The environment of an algorithm: it answers input requests from `inputs`. -/
def inputGame (inputs : List Bytes) : Game where
  State := Unit
  init := pure ()
  respond := fun req => pure (match req with
    | 0x00 :: index => inputs.getD (Bytes.leToNat index) []
    | _ => [])
  win := fun _ _ => true

/-- A payload of length-prefixed byte strings. -/
def decodeOutputs (payload : Bytes) : Option (List Bytes) :=
  if _h : payload.length < 8 then
    if payload.isEmpty then some [] else none
  else
    let length := Bytes.leToNat (payload.take 8)
    let rest := payload.drop 8
    if length ≤ rest.length then
      (decodeOutputs (rest.drop length)).map (rest.take length :: ·)
    else none
termination_by payload.length
decreasing_by simp only [List.length_drop]; omega

/-- Runs `A` on `inputs`. -/
def Algorithm.run (A : Algorithm) (inputs : List Bytes) : ProbComp (Option (List Bytes)) := do
  let (outcome, _) ← (simulateQ (inputGame inputs).impl (runBinary A.image A.bounds)).run ()
  pure (match outcome with
    | .halted payload => decodeOutputs payload
    | _ => none)

end MxxCore
