import Mxx.Certificate.Derivation
import MxxWe.Generated.DiamondWeFamily.Derivation

open MxxWe.Generated.DiamondWeFamily

private def describeError : Mxx.Certificate.DerivationError → String
  | .missingNode _ => "missing frozen node"
  | .unexpectedInstruction _ => "unexpected instruction"
  | .sourceNodeMismatch _ _ => "source node order mismatch"
  | .operandMismatch _ => "operand mismatch"
  | .forwardOperand _ _ => "forward operand"
  | .ruleMismatch _ _ => "rule mismatch"
  | .invalidRelationOperand _ _ => "invalid relation operand"
  | .definitionMismatch _ _ => "definition mismatch"
  | .missingDefinition _ => "missing definition"
  | .unexpectedDefinition _ => "unexpected definition"

private def checkOne
    (name : String)
    (result : Except Mxx.Certificate.DerivationError Unit) : IO (Except String Unit) := do
  IO.eprintln s!"Diamond derivation check: {name} started"
  let task ← IO.asTask (prio := .dedicated) (pure result)
  let mut elapsedSeconds := 0
  while (← IO.getTaskState task) != .finished do
    IO.sleep 1000
    elapsedSeconds := elapsedSeconds + 1
    if elapsedSeconds % 30 == 0 && (← IO.getTaskState task) != .finished then
      IO.eprintln s!"Diamond derivation check: {name} still running ({elapsedSeconds}s elapsed)"
  match ← IO.wait task with
  | .ok (.ok ()) =>
      IO.eprintln s!"Diamond derivation check: {name} completed after {elapsedSeconds}s"
      pure (.ok ())
  | .ok (.error error) => pure (.error s!"{name}: {describeError error}")
  | .error error => pure (.error s!"{name}: interrupted: {error}")

private def checkAll : IO (Except String Unit) := do
  match ← checkOne "stage:encrypt" (Mxx.Certificate.checkProgramDerivation
      DiamondWeFamily_stage_encrypt DiamondWeFamily_stage_encrypt_derivation) with
  | .error error => return .error error
  | .ok () => pure ()
  match ← checkOne "stage:decrypt" (Mxx.Certificate.checkProgramDerivation
      DiamondWeFamily_stage_decrypt DiamondWeFamily_stage_decrypt_derivation) with
  | .error error => return .error error
  | .ok () => pure ()
  match ← checkOne "ideal" (Mxx.Certificate.checkProgramDerivation
      DiamondWeFamily_ideal DiamondWeFamily_ideal_derivation) with
  | .error error => return .error error
  | .ok () => pure ()
  match ← checkOne "requirement-0" (Mxx.Certificate.checkProgramDerivation
      DiamondWeFamily_requirement_0 DiamondWeFamily_requirement_0_derivation) with
  | .error error => return .error error
  | .ok () => pure ()
  match ← checkOne "requirement-1" (Mxx.Certificate.checkProgramDerivation
      DiamondWeFamily_requirement_1 DiamondWeFamily_requirement_1_derivation) with
  | .error error => return .error error
  | .ok () => pure ()
  checkOne "requirement-2" (Mxx.Certificate.checkProgramDerivation
    DiamondWeFamily_requirement_2 DiamondWeFamily_requirement_2_derivation)

def main (_args : List String) : IO UInt32 := do
  IO.eprintln s!"Diamond derivation hash: {DiamondWeFamily_derivationHash}"
  match ← checkAll with
  | .ok () =>
      IO.println "true"
      return 0
  | .error error =>
      IO.eprintln s!"Diamond derivation check failed: {error}"
      return 1
