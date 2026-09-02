import MxxIrCore.Structural

namespace Mxx.Bgg

open Mxx.IR

inductive TraceStep where
  | zeroPlaintext | zeroVector | zeroPublicKey
  | notPlaintext | notVector | notPublicKey
  | productPublicKeyDecompose | productPublicKeyMaterialize | productPublicKeyMultiply
  | productVectorDecompose | productVectorApplyPreimage | productVectorMultiply | productVectorOutput
  | productPlaintextOutput | sumPlaintext | sumVector | sumPublicKey
  | twoProductPublicKey | twoProductVector | twoProductPlaintext
  | xorPlaintext | xorVector | xorPublicKey
  | candidateVectorSelect | candidatePublicKeySelect | candidatePlaintextSelect
  | activeVectorSelect | activePublicKeySelect | activePlaintextSelect | layerOutput
  deriving Repr, DecidableEq, Ord, Inhabited

/- These tags mirror the retained operation metadata emitted by the Rust BGG lowering. -/
inductive TraceLane where
  | vector | publicKey | plaintext
  deriving Repr, DecidableEq, Ord, Inhabited

inductive TraceSubrole where
  | decompose | materializeExact | multiply | applyPreimage | select | gateOutput
  deriving Repr, DecidableEq, Inhabited

inductive TraceRole where
  | decomposition | materializePreimageExact | applyPreimage | matrixMultiply
  | candidateSelect | activeSelect | gateOutput
  deriving Repr, DecidableEq, Inhabited

inductive TraceAnchor where
  | one | left | right | scalar | selector | active
  deriving Repr, DecidableEq, Ord

inductive OperandSource where
  | external (role : TraceAnchor) (handle : WireRef) (route : StructuralValueRoute)
  | prior (step : TraceStep) (route : StructuralValueRoute)
  deriving Repr, DecidableEq

inductive OperandSourceDescriptor where
  | external (role : TraceAnchor)
  | prior (step : TraceStep)
  deriving Repr, DecidableEq

def OperandSource.descriptor : OperandSource → OperandSourceDescriptor
  | .external role _ _ => .external role
  | .prior step _ => .prior step

structure TraceEntry where
  step : TraceStep
  operands : Array WireRef
  sources : Array OperandSource
  deriving Repr, DecidableEq

def OperandSource.valid (stage : Stage) (entry : TraceEntry)
    (prior : TraceStep → Option WireRef) : Prop :=
  ∃ hsize : entry.operands.size = entry.sources.size, ∀ (i : Nat) (h : i < entry.sources.size),
    let source : OperandSource := entry.sources[i]'h
    let operand := entry.operands[i]'(by omega)
    match source with
    | .external _ handle route => followsStructuralValueRoute stage handle operand route
    | .prior step route => match prior step with
      | some handle => followsStructuralValueRoute stage handle operand route
      | none => False

end Mxx.Bgg
