import Mxx.Certificate.Rules.BggThreeTrace
import Mxx.Certificate.Rules.PointwiseFormulaSemantics

namespace Mxx.Certificate

open scoped Matrix

/-!
# Quotient-level transport for frame-indexed BGG semantics

The pointwise elaborator normalizes executable matrices only up to `MatrixModEq`.  Its
`SemanticResultAt.matrixValue_eq` theorem converts that relation to equality in the exact
negacyclic quotient.  The helpers in this file transport the compact BGG induction payload across
those equalities without ever claiming equality of the stored integer representatives.
-/

/-- Transport a BGG lane across quotient equalities for its public key, encoding vector, and
plaintext scalar. -/
def QuotientBggLane.transport
    {R : Type} [CommRing R]
    {outputRows secretColumns publicColumns : Type} [Fintype secretColumns]
    {secret : _root_.Matrix outputRows secretColumns R}
    {gadget : _root_.Matrix secretColumns publicColumns R}
    {sourcePublicKey targetPublicKey : _root_.Matrix secretColumns publicColumns R}
    {sourceVector targetVector : _root_.Matrix outputRows publicColumns R}
    {sourcePlaintext targetPlaintext : R}
    (publicKeyEq : sourcePublicKey = targetPublicKey)
    (vectorEq : sourceVector = targetVector)
    (plaintextEq : sourcePlaintext = targetPlaintext)
    (source : QuotientBggLane secret gadget sourcePublicKey sourceVector sourcePlaintext) :
    QuotientBggLane secret gadget targetPublicKey targetVector targetPlaintext := by
  subst targetPublicKey
  subst targetVector
  subst targetPlaintext
  exact source

/-- Transport only the executable representatives of a BGG gate result.  The Boolean plaintext
is fixed by the gate and therefore cannot be replaced by a caller-provided scalar equality. -/
def QuotientBggGateResult.transport
    {R : Type} [CommRing R]
    {outputRows secretColumns publicColumns : Type} [Fintype secretColumns]
    {secret : _root_.Matrix outputRows secretColumns R}
    {gadget : _root_.Matrix secretColumns publicColumns R}
    {booleanValue : Bool}
    (source : QuotientBggGateResult secret gadget booleanValue)
    (targetPublicKey : _root_.Matrix secretColumns publicColumns R)
    (targetVector : _root_.Matrix outputRows publicColumns R)
    (publicKeyEq : source.publicKey = targetPublicKey)
    (vectorEq : source.vector = targetVector) :
    QuotientBggGateResult secret gadget booleanValue := {
  publicKey := targetPublicKey
  vector := targetVector
  lane := source.lane.transport publicKeyEq vectorEq rfl
}

end Mxx.Certificate
