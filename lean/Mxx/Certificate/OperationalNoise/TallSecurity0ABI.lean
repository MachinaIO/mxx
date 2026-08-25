import Mxx.Certificate.OperationalNoise.RelationReplay
import Mxx.Certificate.OperationalNoise.SchemaV1

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0ABI

open Mxx.Certificate.OperationalNoise
open SchemaV1

/-! Reached-only chronological ABI for the fixed Tall Security0 proof payload. -/

structure Owner where
  scope : StatementScope
  expression : ExpressionRef
deriving DecidableEq, Repr

structure Monomial where
  centralFactors : List Owner
  orderedFactors : List Owner
deriving DecidableEq, Repr

structure Term where
  monomial : Monomial
  coefficient : Int
deriving DecidableEq, Repr

inductive Bound where
  | exactZero
  | finite (maximumAbsoluteCoefficient : Nat)
  | large
  | missing
deriving DecidableEq, Repr

inductive Projection where | coefficient | summary
deriving DecidableEq, Repr

inductive ValueRef where
  | predecessor (inputPosition : Nat) (projection : Projection)
  | result (event : Nat) (projection : Projection)
  | transfer (event : Nat)
deriving DecidableEq, Repr

inductive Scale where
  | value (value : ValueRef)
  | magnitude (magnitude : Nat)
deriving DecidableEq, Repr

structure FactorEvidence where
  bound : ValueRef
  isConstantPolynomial : Bool
  supportUpper : Option Nat
deriving DecidableEq, Repr

structure ProductFacts where
  leftIsConstantPolynomial : Bool
  rightIsConstantPolynomial : Bool
  rightKnownZeroRows : Option Nat
  leftSupportUpper : Option Nat
  rightSupportUpper : Option Nat
deriving DecidableEq, Repr

inductive Authority where
  | factStore
  | programFamilyFact
  | operator
  | relationPreimageSource (source : ExpressionRef)
deriving DecidableEq, Repr

inductive BoundRule where
  | authority (authority : Authority)
  | identity (input : ValueRef)
  | sum (inputs : List ValueRef)
  | scale (value : ValueRef) (scale : Scale)
  | monomialProduct (monomial : Monomial) (factors : List FactorEvidence)
  | product (left right : ValueRef) (facts : ProductFacts)
  | tensor (left right : ValueRef) (leftConstantPolynomial rightConstantPolynomial : Bool)
deriving DecidableEq, Repr

structure UniversalDispatch where
  preimageFamily : ProgramRef
  preimageSource : ExpressionRef
  trapdoorSource : ExpressionRef
deriving DecidableEq, Repr

structure EventRange where
  start : Nat
  «end» : Nat
deriving DecidableEq, Repr

inductive RelationRule where
  | universal (computed : Nat) (lhs : Monomial) (lhsLayout : Option Layout)
      (rhsResult : Nat)
  | gadget (gadget decomposition : Owner) (input : ExpressionRef) (inputResult : Nat)
deriving DecidableEq, Repr

structure TermRef where
  valueEvent : Nat
  termOrdinal : Nat
deriving DecidableEq, Repr

inductive MergeSource where
  | operator (inputs : TermRef × TermRef)
  | relation (application sourceTermOrdinal : Nat)
deriving DecidableEq, Repr

structure Merge where
  owner : Owner
  source : MergeSource
  output : Monomial
  signedContribution : Int
deriving DecidableEq, Repr

inductive Event where
  | invocationStart (root : Owner)
  | predecessor (consumer : Owner) (inputPosition : Nat) (predecessor : ExpressionRef)
      (sourceResult : Nat)
  | resultExact (owner : Owner) (terms : List Term) (summary : Bound)
  | resultCoefficient (owner : Owner) (bound : Bound)
  | invocationEndExact (root : Owner) (terms : List Term) (summary : Bound)
  | specializationComputed (owner : Owner) (dispatch : UniversalDispatch) (source : EventRange)
  | appliedRelation (owner : Owner) (sourceMonomial : Monomial) (outerCoefficient : Int)
      (orderedStart orderedEndExclusive : Nat) (rule : RelationRule)
  | boundTransfer (owner : Owner) (rule : BoundRule)
  | coefficientMerge (merge : Merge)
  | preFoldPolynomial (terms : List Term) (summary : Bound)
      (summaryEvidence : Option ValueRef)
  | survivorFold (coefficient : Int) (bound : Nat)
deriving DecidableEq, Repr, Inhabited

structure Frame where
  root : Owner
  start : Nat
  lastExact : Option (List Term × Bound)
  preFolded : Bool
deriving DecidableEq, Repr

structure ReplayState where
  history : List Event
  frameStarts : List Nat
  frames : List Frame
deriving DecidableEq, Repr

def initialState : ReplayState := ⟨[], [], []⟩

def rowAt? {α : Type} (rows : List α) (row : Nat) : Option α := rows[row]?

def ownerValid (document : Document) (owner : Owner) : Bool :=
  (rowAt? document.expressions owner.expression.row).isSome &&
    match owner.scope with
    | .closed root => decide (document.residualRoot = .closed root)
    | .program program => (rowAt? document.programs program.row).isSome

def monomialValid (document : Document) (monomial : Monomial) : Bool :=
  (monomial.centralFactors ++ monomial.orderedFactors).all (ownerValid document)

def currentScope (state : ReplayState) (owner : Owner) : Bool :=
  match state.frames.head? with
  | some frame => decide (frame.root.scope = owner.scope)
  | none => false

def prior (state : ReplayState) (event : Nat) : Bool := event < state.history.length

def sameFrame (state : ReplayState) (owner : Owner) (event : Nat) : Bool :=
  prior state event && match state.frames.head?, state.frameStarts[event]? with
    | some frame, some start =>
        decide (frame.root.scope = owner.scope) && decide (frame.start = start)
    | _, _ => false

def exactFrameRange (state : ReplayState) (range : EventRange) : Bool :=
  decide (range.start < range.end) && decide (range.end ≤ state.history.length) &&
    ((state.frameStarts.drop range.start).take (range.end - range.start)).all
      (fun start => decide (start = range.start))

def resultOwner? : Event → Option Owner
  | .resultExact owner _ _ | .resultCoefficient owner _ => some owner
  | _ => none

def transferOwner? : Event → Option Owner
  | .boundTransfer owner _ => some owner
  | .appliedRelation owner _ _ _ _ _ => some owner
  | _ => none

def relationOwner? : Event → Option Owner
  | .appliedRelation owner _ _ _ _ _ => some owner
  | _ => none

def projectionAvailable : Event → Projection → Bool
  | .resultExact _ _ summary, .coefficient => decide (summary ≠ .missing)
  | .resultExact .., .summary => true
  | .resultCoefficient _ bound, .coefficient => decide (bound ≠ .missing)
  | _, _ => false

def exactTermExists (state : ReplayState) (event ordinal : Nat) : Bool :=
  prior state event && match state.history[event]! with
    | .resultExact _ terms _ | .invocationEndExact _ terms _ =>
        decide (ordinal < terms.length)
    | _ => false

def relationSourceValid (state : ReplayState) (owner : Owner) (application ordinal : Nat) : Bool :=
  sameFrame state owner application && match state.history[application]! with
    | .appliedRelation relationOwner _ _ _ _ rule =>
        decide (relationOwner = owner) && match rule with
          | .universal computed _ _ rhsResult =>
              match state.history[computed]? with
              | some (.specializationComputed computedOwner _ source) =>
                  decide (computedOwner = owner) && decide (source.end = computed) &&
                    decide (rhsResult + 1 = source.end) && exactFrameRange state source &&
                    exactTermExists state rhsResult ordinal
              | _ => false
          | .gadget _ _ _ inputResult =>
              sameFrame state owner inputResult && exactTermExists state inputResult ordinal
    | _ => false

def valueRefValid (state : ReplayState) (owner : Owner) : ValueRef → Bool
  | .predecessor position projection => state.history.reverse.any fun event => match event with
      | .predecessor consumer inputPosition predecessor sourceResult =>
          decide (consumer = owner) && decide (inputPosition = position) &&
            sameFrame state owner sourceResult &&
            decide
              (resultOwner? state.history[sourceResult]! = some ⟨owner.scope, predecessor⟩) &&
            projectionAvailable state.history[sourceResult]! projection
      | _ => false
  | .result event projection => sameFrame state owner event &&
      projectionAvailable state.history[event]! projection &&
      match resultOwner? state.history[event]! with
      | some resultOwner => decide (resultOwner.scope = owner.scope)
      | none => false
  | .transfer event =>
      sameFrame state owner event && decide (transferOwner? state.history[event]! = some owner)

def scaleValid (state : ReplayState) (owner : Owner) : Scale → Bool
  | .value value => valueRefValid state owner value
  | .magnitude _ => true

def preimageSourceValid (document : Document) (source : ExpressionRef) : Bool :=
  let event? := match rowAt? document.expressions source.row with
    | some { descriptor := .event (.sampler event), .. }
    | some { descriptor := .operation (.event (.sampler event)) _, .. } => some event
    | _ => none
  match event? with
  | some event => match rowAt? document.events event.row with
      | some (.sampler _ (.preimage ..) _) => true
      | _ => false
  | none => false

def ruleValid (document : Document) (state : ReplayState) (owner : Owner) : BoundRule → Bool
  | .authority (.relationPreimageSource source) =>
      preimageSourceValid document source
  | .authority _ => true
  | .identity input => valueRefValid state owner input
  | .sum inputs => inputs.all (valueRefValid state owner)
  | .scale value scale => valueRefValid state owner value && scaleValid state owner scale
  | .monomialProduct monomial factors =>
      monomialValid document monomial &&
        factors.all (fun factor => valueRefValid state owner factor.bound)
  | .product left right _ | .tensor left right _ _ =>
      valueRefValid state owner left && valueRefValid state owner right

def termRefValid (state : ReplayState) (reference : TermRef) : Bool :=
  prior state reference.valueEvent &&
    match state.history[reference.valueEvent]! with
    | .resultExact _ terms _ => reference.termOrdinal < terms.length
    | _ => false

def termRefOwnerLocal (state : ReplayState) (owner : Owner) (reference : TermRef) : Bool :=
  sameFrame state owner reference.valueEvent && termRefValid state reference &&
    match resultOwner? state.history[reference.valueEvent]! with
    | some sourceOwner => decide (sourceOwner.scope = owner.scope)
    | none => false

def summaryEvidenceValid (state : ReplayState) (frame : Frame) : Option ValueRef → Bool
  | none => true
  | some (.result event .summary) => sameFrame state frame.root event &&
      match state.history[event]! with | .resultExact .. => true | _ => false
  | some (.transfer event) => sameFrame state frame.root event &&
      match transferOwner? state.history[event]! with
      | some owner => decide (owner.scope = frame.root.scope)
      | none => false
  | _ => false

def append (state : ReplayState) (event : Event) (frames : List Frame) : ReplayState :=
  let frameStart := match event, state.frames.head? with
    | .invocationStart _, _ => state.history.length
    | _, some frame => frame.start
    | _, none => 0
  ⟨state.history ++ [event], state.frameStarts ++ [frameStart], frames⟩

/-- One honest chronological replay step. Failed structural references return `none`. -/
def step (document : Document) (state : ReplayState) (event : Event) : Option ReplayState :=
  let accept (frames : List Frame) := some (append state event frames)
  match event with
  | .invocationStart root =>
      if ownerValid document root then
        accept (⟨root, state.history.length, none, false⟩ :: state.frames)
      else none
  | .predecessor consumer _ predecessor sourceResult =>
      if currentScope state consumer && ownerValid document consumer && prior state sourceResult &&
          decide
            (resultOwner? state.history[sourceResult]! = some ⟨consumer.scope, predecessor⟩)
      then
        accept state.frames
      else none
  | .resultExact owner terms summary =>
      if currentScope state owner && ownerValid document owner &&
          terms.all (fun term => monomialValid document term.monomial) then
        match state.frames with
        | [] => none
        | frame :: frames => accept ({ frame with lastExact := some (terms, summary) } :: frames)
      else none
  | .resultCoefficient owner _ =>
      if currentScope state owner && ownerValid document owner then accept state.frames else none
  | .invocationEndExact root terms summary =>
      match state.frames with
      | [] => none
      | frame :: frames =>
          if decide (frame.root = root) && decide (frame.lastExact = some (terms, summary)) &&
              frame.preFolded then
            accept frames
          else none
  | .specializationComputed owner dispatch source =>
      if currentScope state owner && ownerValid document owner &&
          (rowAt? document.programs dispatch.preimageFamily.row).isSome &&
          (rowAt? document.expressions dispatch.preimageSource.row).isSome &&
          (rowAt? document.expressions dispatch.trapdoorSource.row).isSome &&
          exactFrameRange state source && decide (source.end = state.history.length) &&
          (match state.history[source.start]?, state.history[source.end - 1]? with
          | some (.invocationStart root), some (.invocationEndExact ended _ _) =>
              decide (root = ended) && decide (root.scope = owner.scope)
          | _, _ => false) then accept state.frames else none
  | .appliedRelation owner sourceMonomial _ orderedStart orderedEnd rule =>
      let ruleOk := match rule with
        | .universal computed lhs _ rhsResult =>
            sameFrame state owner computed && monomialValid document lhs &&
              (match state.history[computed]! with
              | .specializationComputed computedOwner _ source =>
                  decide (computedOwner = owner) && decide (source.end = computed) &&
                    decide (rhsResult + 1 = source.end) && exactFrameRange state source &&
                    (match state.history[rhsResult]? with
                    | some (.invocationEndExact rhsOwner _ _) =>
                        decide (rhsOwner.scope = owner.scope)
                    | _ => false)
              | _ => false)
        | .gadget gadget decomposition input inputResult =>
            ownerValid document gadget && ownerValid document decomposition &&
              decide (gadget.scope = owner.scope) &&
              decide (decomposition.scope = owner.scope) &&
              (rowAt? document.expressions input.row).isSome &&
              sameFrame state owner inputResult &&
              decide (resultOwner? state.history[inputResult]! = some ⟨owner.scope, input⟩)
      if currentScope state owner && ownerValid document owner &&
          monomialValid document sourceMonomial && decide (orderedStart ≤ orderedEnd) &&
          decide (orderedEnd ≤ sourceMonomial.orderedFactors.length) && ruleOk then
        accept state.frames
      else none
  | .boundTransfer owner rule =>
      if currentScope state owner && ownerValid document owner &&
          ruleValid document state owner rule
      then accept state.frames else none
  | .coefficientMerge merge =>
      let sourceOk := match merge.source with
        | .operator inputs =>
            termRefOwnerLocal state merge.owner inputs.1 &&
              termRefOwnerLocal state merge.owner inputs.2
        | .relation application ordinal =>
            relationSourceValid state merge.owner application ordinal
      if currentScope state merge.owner && ownerValid document merge.owner &&
          monomialValid document merge.output && sourceOk then accept state.frames else none
  | .preFoldPolynomial terms summary evidence =>
      match state.frames with
      | [] => none
      | frame :: frames =>
          if decide (frame.lastExact = some (terms, summary)) &&
              summaryEvidenceValid state frame evidence then
            accept ({ frame with preFolded := true } :: frames)
          else none
  | .survivorFold _ bound =>
      if prior state bound then
        match transferOwner? state.history[bound]! with
        | some owner =>
            if sameFrame state owner bound then accept state.frames else none
        | none => none
      else none

def replayFrom (document : Document) : ReplayState → List Event → Option ReplayState
  | state, [] => some state
  | state, event :: events => do
      let next ← step document state event
      replayFrom document next events

def replay (document : Document) (events : List Event) : Option ReplayState :=
  replayFrom document initialState events

def Valid (document : Document) (events : List Event) : Prop :=
  ∃ state, replay document events = some state ∧ state.frames = []

instance (document : Document) (events : List Event) : Decidable (Valid document events) := by
  unfold Valid
  infer_instance

end Mxx.Certificate.OperationalNoise.TallSecurity0ABI
