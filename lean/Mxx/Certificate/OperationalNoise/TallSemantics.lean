import Mxx.Certificate.OperationalNoise.TallSecurity0ABI

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSemantics

open Mxx.Certificate.OperationalNoise
open SchemaV1
open TallSecurity0ABI

/-! Pure semantic statements shared by the fixed toy and Tall certificate ABIs. -/

/-- A semantic environment assigns values only to factors that occur in monomial keys. -/
abbrev Env (Factor : Type) := Factor → Int

def evalMonomial {Factor : Type} (env : Env Factor) (key : MonomialKey Factor) : Int :=
  (key.centralFactors.map env).prod * (key.orderedFactors.map env).prod

def evalPolynomial {Factor : Type} (env : Env Factor) (terms : Polynomial Factor) : Int :=
  (terms.map (fun term ↦ term.coefficient * evalMonomial env term.key)).sum

def boundInterprets (modulus : Nat) (bound : Bound) (value : Int) : Prop :=
  match bound with
  | .exactZero => centeredNorm modulus value = 0
  | .finite maximum => centeredNorm modulus value ≤ maximum
  | .large => True
  | .missing => False

/-- An event-level claim; coefficient results do not create entries in `Env`. -/
inductive ValueClaim (Factor : Type) where
  | exact (terms : Polynomial Factor) (summary : Bound)
  | coefficient (bound : Bound)

def ValueClaim.Interprets {Factor : Type} (modulus : Nat) (env : Env Factor)
    (actual : Int) : ValueClaim Factor → Prop
  | .exact terms summary =>
      ∃ remainder,
        (actual - evalPolynomial env terms) % Int.ofNat modulus = remainder % Int.ofNat modulus ∧
          boundInterprets modulus summary remainder
  | .coefficient bound => boundInterprets modulus bound actual

/-- A finite raw bound is supplied constructively; no decimal parser belongs to this layer. -/
def rawCoefficientClassInterprets (modulus : Nat) (value : Int) :
    RawCoefficientClass → Prop
  | .exactZero => centeredNorm modulus value = 0
  | .finite encoded =>
      ∃ maximum : Nat, encoded = toString maximum ∧ centeredNorm modulus value ≤ maximum
  | .large => True

def rawValueContractInterprets (modulus : Nat) (value : Int)
    (contract : RawValueContract) : Prop :=
  ∃ coefficientClass,
    contract.coefficientClass = some coefficientClass ∧
      rawCoefficientClassInterprets modulus value coefficientClass

def ownerAtSelector (document : TallDocument) (selector : Option Nat) (owner : Owner) : Prop :=
  match document.residualRoot, owner.scope, selector with
  | .closed root, .closed ownerRoot, none => root = ownerRoot
  | .family program domain, .program ownerProgram, some selected =>
      program = ownerProgram ∧ domain.minimum ≤ selected ∧ selected < domain.maximumExclusive
  | _, _, _ => False

def sourceRef? : ExpressionDescriptor → Option SourceRef
  | .source (.direct source) | .source (.family source _) => some source
  | _ => none

def sourceContract? : SchemaV1.SourceRow → Option RawValueContract
  | .direct _ _ contract | .family _ contract => contract
  | .constant _ => none

def expressionEventRefs : ExpressionDescriptor → List EventRef
  | .event (.sample event) | .event (.sampler event) => [event]
  | .event (.gadgetDecompose events) => events
  | .operation (.event (.sample event)) _ | .operation (.event (.sampler event)) _ => [event]
  | .operation (.event (.gadgetDecompose events)) _ => events
  | _ => []

def eventContract? : SchemaV1.EventRow → Option RawValueContract
  | .sample _ _ contract | .sampler _ _ contract | .gadgetDecompose _ _ _ _ _ _ _ contract =>
      contract

def sourceRawContract? (document : TallDocument) (source : SourceRef) :
    Option RawValueContract :=
  (document.sources.lookup source.row).bind sourceContract?

def eventRawContract? (document : TallDocument) (event : EventRef) : Option RawValueContract :=
  (document.events.lookup event.row).bind eventContract?

def monomialContains (monomial : Monomial) (owner : Owner) : Prop :=
  owner ∈ monomial.centralFactors ∨ owner ∈ monomial.orderedFactors

def termContains (term : Term) (owner : Owner) : Prop := monomialContains term.monomial owner

def eventContainsFactor (event : Event) (owner : Owner) : Prop :=
  match event with
  | .resultExact _ terms _ | .invocationEndExact _ _ terms _ |
      .preFoldPolynomial _ terms _ _ => ∃ term ∈ terms, termContains term owner
  | .appliedRelation _ sourceMonomial _ _ _ rule =>
      monomialContains sourceMonomial owner ∨ match rule with
        | .universal _ lhs _ _ => monomialContains lhs owner
        | .gadget gadget decomposition _ _ => gadget = owner ∨ decomposition = owner
  | .boundTransfer _ (.monomialProduct monomial _) => monomialContains monomial owner
  | .coefficientMerge merge => monomialContains merge.output owner
  | _ => False

def FactorAtomAt (history : EventHistory) (event : Nat) (owner : Owner) : Prop :=
  ∃ entry, history.lookup event = some entry ∧ eventContainsFactor entry.event owner

def SourceFactorAt (document : TallDocument) (history : EventHistory) (selector : Option Nat)
    (owner : Owner) (factorEvent : Nat) (source : SourceRef) : Prop :=
  FactorAtomAt history factorEvent owner ∧ ownerAtSelector document selector owner ∧
    ∃ expression sourceRow,
      document.expressions.lookup owner.expression.row = some expression ∧
        sourceRef? expression.descriptor = some source ∧
        document.sources.lookup source.row = some sourceRow

def SamplerFactorAt (document : TallDocument) (history : EventHistory) (selector : Option Nat)
    (owner : Owner) (factorEvent : Nat) (event : EventRef) : Prop :=
  FactorAtomAt history factorEvent owner ∧ ownerAtSelector document selector owner ∧
    ∃ expression eventRow,
      document.expressions.lookup owner.expression.row = some expression ∧
        event ∈ expressionEventRefs expression.descriptor ∧
        document.events.lookup event.row = some eventRow

def RelationApplicationAt (document : TallDocument) (history : EventHistory)
    (selector : Option Nat) (application : Nat) : Prop :=
  ∃ entry owner sourceMonomial outerCoefficient orderedStart orderedEndExclusive rule,
    history.lookup application = some entry ∧
      entry.event = .appliedRelation owner sourceMonomial outerCoefficient orderedStart
        orderedEndExclusive rule ∧
      ownerAtSelector document selector owner ∧
      TallSecurity0ABI.ownerValid document owner = true

def exactTermsAt? (history : EventHistory) (event : Nat) : Option (List Term) :=
  match TallSecurity0ABI.eventAt? history event with
  | some (.resultExact _ terms _) | some (.invocationEndExact _ _ terms _) => some terms
  | _ => none

def RelationCongruent (modulus : Nat) (history : EventHistory) (env : Env Owner)
    (application : Nat) : Prop :=
  match TallSecurity0ABI.eventAt? history application with
  | some (.appliedRelation _ _ _ _ _ (.universal computed lhs _ rhsResult)) =>
      (∃ computedOwner dispatch source,
        TallSecurity0ABI.eventAt? history computed =
            some (Event.specializationComputed computedOwner dispatch source) ∧
          source.end = computed ∧ source.start ≤ rhsResult ∧ rhsResult < source.end) ∧
        ∃ rhsTerms,
          exactTermsAt? history rhsResult = some rhsTerms ∧
            evalMonomial env lhs.toKey % Int.ofNat modulus =
              evalPolynomial env (rhsTerms.map Term.toExact) % Int.ofNat modulus
  | some (.appliedRelation _ _ _ _ _ (.gadget gadget decomposition _ inputResult)) =>
      ∃ inputTerms,
        exactTermsAt? history inputResult = some inputTerms ∧
          (env gadget * env decomposition) % Int.ofNat modulus =
            evalPolynomial env (inputTerms.map Term.toExact) % Int.ofNat modulus
  | _ => False

/-- Honest primitive contracts and all reached relation congruences for one selector. -/
structure Witness (document : TallDocument) (history : EventHistory) (selector : Option Nat)
    (modulus : Nat) where
  env : Env Owner
  sourceBound : ∀ owner factorEvent source,
    SourceFactorAt document history selector owner factorEvent source →
      ∃ contract, sourceRawContract? document source = some contract ∧
        rawValueContractInterprets modulus (env owner) contract
  samplerBound : ∀ owner factorEvent event,
    SamplerFactorAt document history selector owner factorEvent event →
      ∃ contract, eventRawContract? document event = some contract ∧
        rawValueContractInterprets modulus (env owner) contract
  relationCongruence : ∀ application,
    RelationApplicationAt document history selector application →
      RelationCongruent modulus history env application

abbrev TallEnv := Env Owner

abbrev TallValueClaim := ValueClaim Owner

def ForStatement (root : SchemaV1.ResidualRoot) (claim : Option Nat → Prop) : Prop :=
  match root with
  | .closed _ => claim none
  | .family _ domain =>
      ∀ selector, domain.minimum ≤ selector → selector < domain.maximumExclusive →
        claim (some selector)

def rootMatchesOwner (root : SchemaV1.ResidualRoot) (owner : Owner) : Prop :=
  match root, owner.scope with
  | .closed expression, .closed ownerExpression => expression = ownerExpression
  | .family program _, .program ownerProgram => program = ownerProgram
  | _, _ => False

/-- The fixed Security0 statement; later checkpoints construct, rather than assume, its proof. -/
def Security0Accepted (document : TallDocument) (history : EventHistory)
    (plaintextModulus ciphertextModulus ringDimension finalEvent preFoldEvent finalBound : Nat)
    (finalOwner : Owner) (finalTerms : List Term) (finalSummary : Bound)
    (residual : Option Nat → Env Owner → Int) : Prop :=
  TallSecurity0ABI.Valid document history ∧
    document.plaintextModulus = toString plaintextModulus ∧
    document.ciphertextModulus = toString ciphertextModulus ∧
    document.ringDimension = ringDimension ∧
    rootMatchesOwner document.residualRoot finalOwner ∧
    finalSummary = .finite finalBound ∧
    (∃ frameStart,
      history.lookup finalEvent = some
        ⟨.invocationEndExact finalOwner preFoldEvent finalTerms finalSummary, frameStart⟩) ∧
    ForStatement document.residualRoot fun selector ↦
      ∀ witness : Witness document history selector ciphertextModulus,
        ValueClaim.Interprets ciphertextModulus witness.env (residual selector witness.env)
            (.exact (finalTerms.map Term.toExact) finalSummary) ∧
          2 * plaintextModulus * centeredNorm ciphertextModulus
            (residual selector witness.env) < ciphertextModulus

end Mxx.Certificate.OperationalNoise.TallSemantics
