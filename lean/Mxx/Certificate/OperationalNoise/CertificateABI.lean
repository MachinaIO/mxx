import Mxx.Certificate.OperationalNoise.RelationReplay
import Mxx.Certificate.OperationalNoise.SchemaV1

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.CertificateABI

open Mxx.Certificate.OperationalNoise
open SchemaV1

/-! Chronological ABI for an operational-noise proof payload. -/

def rowTableNodeCount {α : Type} : RowTable α → Nat
  | .empty => 0
  | .node _ _ left right => rowTableNodeCount left + rowTableNodeCount right + 1

def orderedLeafSize : Nat := 16

structure OrderedLeaves (α : Type) where
  leaves : RowTable (Array α)
  size : Nat
deriving Repr

def OrderedLeaves.leafCount {α : Type} (rows : OrderedLeaves α) : Nat :=
  (rows.size + orderedLeafSize - 1) / orderedLeafSize

def OrderedLeaves.expectedLeafSize {α : Type} (rows : OrderedLeaves α) (leaf : Nat) : Nat :=
  Nat.min orderedLeafSize (rows.size - leaf * orderedLeafSize)

def OrderedLeaves.wellFormed {α : Type} (rows : OrderedLeaves α) : Bool :=
  rows.leaves.wellFormed &&
    decide (rowTableNodeCount rows.leaves = rows.leafCount) &&
    rows.leaves.allBool fun leaf values =>
      decide (leaf < rows.leafCount) &&
        decide (values.size = rows.expectedLeafSize leaf)

def OrderedLeaves.get? {α : Type} (rows : OrderedLeaves α) (position : Nat) : Option α :=
  if position < rows.size then
    (rows.leaves.lookup (position / orderedLeafSize)).bind fun leaf =>
      leaf[position % orderedLeafSize]?
  else none

def OrderedLeaves.Valid {α : Type} (rows : OrderedLeaves α) : Prop :=
  rows.wellFormed = true

abbrev ExpressionInputs := OrderedLeaves ExpressionRef

inductive IndexExpression where
  | binding (expression : ExpressionRef)
  | constant (expression : ExpressionRef) (value : Int)
  | add (expression : ExpressionRef) (left right : IndexExpression)
  | subtract (expression : ExpressionRef) (left right : IndexExpression)
  | multiply (expression : ExpressionRef) (left right : IndexExpression)
  | divide (expression : ExpressionRef) (left right : IndexExpression)
  | remainder (expression : ExpressionRef) (left right : IndexExpression)
  | negate (expression : ExpressionRef) (value : IndexExpression)
deriving Repr

def IndexExpression.reference : IndexExpression → ExpressionRef
  | .binding expression | .constant expression _ | .add expression _ _ |
      .subtract expression _ _ | .multiply expression _ _ | .divide expression _ _ |
      .remainder expression _ _ | .negate expression _ => expression

def IndexExpression.lookup (bindings : List (ExpressionRef × Int))
    (expression : ExpressionRef) : Option Int :=
  (bindings.find? fun binding => binding.1 = expression).map Prod.snd

def IndexExpression.euclideanDivRem (numerator denominator : Int) : Option (Int × Int) :=
  if denominator = 0 then none
  else
    let positiveDenominator : Int := Int.ofNat denominator.natAbs
    some (numerator / positiveDenominator, numerator % positiveDenominator)

def IndexExpression.evaluate (bindings : List (ExpressionRef × Int)) :
    IndexExpression → Option Int
  | .binding expression => IndexExpression.lookup bindings expression
  | .constant _ value => some value
  | .add _ left right => return (← left.evaluate bindings) + (← right.evaluate bindings)
  | .subtract _ left right => return (← left.evaluate bindings) - (← right.evaluate bindings)
  | .multiply _ left right => return (← left.evaluate bindings) * (← right.evaluate bindings)
  | .divide _ left right => do
      let leftValue ← left.evaluate bindings
      let rightValue ← right.evaluate bindings
      (IndexExpression.euclideanDivRem leftValue rightValue).map Prod.fst
  | .remainder _ left right => do
      let leftValue ← left.evaluate bindings
      let rightValue ← right.evaluate bindings
      (IndexExpression.euclideanDivRem leftValue rightValue).map Prod.snd
  | .negate _ value => return -(← value.evaluate bindings)

structure IndexLutRows where
  expression : IndexExpression
deriving Repr

def frontierAxisRange : SchemaV1.FrontierAxis → Nat × Nat
  | .argument _ _ _ domain | .extractedCoefficient _ _ domain => domain

def frontierAxisExpression : SchemaV1.FrontierAxis → Option ExpressionRef
  | .argument _ (.expression expression) _ _ |
      .extractedCoefficient _ (.expression expression) _ => some expression
  | .argument _ (.family _) _ _ | .extractedCoefficient _ (.family _) _ => none

def frontierSize : List SchemaV1.FrontierAxis → Nat
  | [] => 1
  | axis :: rest =>
      let domain := frontierAxisRange axis
      (domain.2 - domain.1) * frontierSize rest

def frontierBindingsAt : List SchemaV1.FrontierAxis → Nat →
    Option (List (ExpressionRef × Int))
  | [], position => if position = 0 then some [] else none
  | axis :: rest, position => do
      let expression ← frontierAxisExpression axis
      let tailSize := frontierSize rest
      if tailSize = 0 then none
      else
        let domain := frontierAxisRange axis
        let width := domain.2 - domain.1
        let digit := position / tailSize
        if digit < width then
          let tail ← frontierBindingsAt rest (position % tailSize)
          some ((expression, Int.ofNat (domain.1 + digit)) :: tail)
        else none

def IndexLutRows.get? (rows : IndexLutRows) (frontier : List SchemaV1.FrontierAxis)
    (position : Nat) : Option SchemaV1.IndexLutRow := do
  if position < frontierSize frontier then
    let bindings ← frontierBindingsAt frontier position
    let output ← rows.expression.evaluate bindings
    some ⟨bindings.map fun binding => toString binding.2, toString output⟩
  else none

structure ExpressionRow where
  descriptor : SchemaV1.ExpressionDescriptor
  inputs : ExpressionInputs
  program : Option ProgramRef
deriving Repr

structure IndexUseRow where
  owner : SchemaV1.ObservedWire
  result : Option SchemaV1.PlanRef
  consumed : Option SchemaV1.PlanRef
  kind : SchemaV1.IndexUseKind
  outputRange : Option SchemaV1.Range
  outputType : SchemaV1.ValueType
  frontier : List SchemaV1.FrontierAxis
  rows : IndexLutRows
deriving Repr

def IndexUseRow.index (row : IndexUseRow) : SchemaV1.PlanRef :=
  .expression row.rows.expression.reference

structure CertificateDocument where
  schemaId : String
  schemaVersion : Nat
  plaintextModulus : String
  ciphertextModulus : String
  ringDimension : Nat
  expressions : RowTable ExpressionRow
  programs : RowTable SchemaV1.ProgramRow
  sources : RowTable SchemaV1.SourceRow
  events : RowTable SchemaV1.EventRow
  indexUses : RowTable IndexUseRow
  sliceGroups : RowTable SchemaV1.SliceGroupRow
  residualRoot : SchemaV1.ResidualRoot
deriving Repr

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

def Monomial.toKey (monomial : Monomial) : MonomialKey Owner :=
  { centralFactors := monomial.centralFactors
    orderedFactors := monomial.orderedFactors }

def Term.toExact (term : Term) : ExactTerm Owner :=
  { coefficient := term.coefficient
    key := term.monomial.toKey }

inductive Bound where
  | exactZero
  | finite (maximumAbsoluteCoefficient : Nat)
  | large
  | missing
deriving DecidableEq, Repr

inductive Projection where | coefficient | summary
deriving DecidableEq, Repr

inductive ValueRef where
  | predecessor (inputPosition bindingEvent : Nat) (projection : Projection)
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
  | resultExact (owner : Owner) (terms : List Term)
      (coefficientBound : Bound) (coefficientProducer : Nat)
      (summary : Bound) (summaryProducer : Option Nat)
  | resultCoefficient (owner : Owner) (bound : Bound)
  | invocationEndExact (root : Owner) (preFoldEvent : Nat) (terms : List Term)
      (coefficientBound : Bound) (coefficientProducer : Nat)
      (summary : Bound) (summaryProducer : Option Nat)
  | specializationComputed (owner : Owner) (dispatch : UniversalDispatch) (source : EventRange)
  | appliedRelation (owner : Owner) (sourceMonomial : Monomial) (outerCoefficient : Int)
      (orderedStart orderedEndExclusive : Nat) (rule : RelationRule)
  | boundTransfer (owner : Owner) (rule : BoundRule)
  | coefficientMerge (merge : Merge)
  | preFoldPolynomial (resultEvent : Nat) (terms : List Term) (summary : Bound)
      (summaryEvidence : Option ValueRef)
  | survivorFold (coefficient : Int) (bound : Nat)
deriving DecidableEq, Repr, Inhabited

structure Frame where
  root : Owner
  start : Nat
deriving DecidableEq, Repr

structure AnnotatedEvent where
  event : Event
  frameStart : Nat
deriving DecidableEq, Repr

def eventLeafSize : Nat := 16

structure EventHistory where
  leaves : RowTable (Array AnnotatedEvent)
  size : Nat
deriving Repr

def EventHistory.leafCount (history : EventHistory) : Nat :=
  (history.size + eventLeafSize - 1) / eventLeafSize

def EventHistory.expectedLeafSize (history : EventHistory) (leaf : Nat) : Nat :=
  Nat.min eventLeafSize (history.size - leaf * eventLeafSize)

def EventHistory.wellFormed (history : EventHistory) : Bool :=
  history.leaves.wellFormed &&
    decide (rowTableNodeCount history.leaves = history.leafCount) &&
    history.leaves.allBool fun leaf events =>
      decide (leaf < history.leafCount) &&
        decide (events.size = history.expectedLeafSize leaf)

def EventHistory.lookup (history : EventHistory) (event : Nat) : Option AnnotatedEvent :=
  if event < history.size then
    (history.leaves.lookup (event / eventLeafSize)).bind fun leaf =>
      leaf[event % eventLeafSize]?
  else none

structure ReplayState where
  cursor : Nat
  frames : List Frame
deriving DecidableEq, Repr

def initialState : ReplayState := ⟨0, []⟩

def ownerValid (document : CertificateDocument) (owner : Owner) : Bool :=
  match document.expressions.lookup owner.expression.row with
  | some expression => expression.inputs.wellFormed && match owner.scope with
      | .closed root => decide (document.residualRoot = .closed root)
      | .program program => (document.programs.lookup program.row).isSome
  | none => false

def monomialValid (document : CertificateDocument) (monomial : Monomial) : Bool :=
  (monomial.centralFactors ++ monomial.orderedFactors).all (ownerValid document)

def currentScope (state : ReplayState) (owner : Owner) : Bool :=
  match state.frames.head? with
  | some frame => decide (frame.root.scope = owner.scope)
  | none => false

def prior (state : ReplayState) (event : Nat) : Bool := event < state.cursor

def eventAt? (history : EventHistory) (event : Nat) : Option Event :=
  (history.lookup event).map AnnotatedEvent.event

def sameFrame (history : EventHistory) (state : ReplayState) (owner : Owner)
    (event : Nat) : Bool :=
  prior state event && match state.frames.head?, history.lookup event with
    | some frame, some entry =>
        decide (frame.root.scope = owner.scope) && decide (frame.start = entry.frameStart)
    | _, _ => false

def specializationRangeValid (history : EventHistory) (state : ReplayState)
    (range : EventRange) : Bool :=
  decide (0 < range.start) && decide (range.start < range.end) &&
    decide (range.end = state.cursor) &&
    match state.frames.head?, history.lookup (range.start - 1), history.lookup range.start with
    | some parent, some before, some first =>
        decide (parent.start < range.start) && decide (before.frameStart = parent.start) &&
          decide (first.frameStart = range.start) &&
          (match first.event with
          | .invocationStart child => decide (child.scope = parent.root.scope)
          | _ => false)
    | _, _, _ => false

def completedInvocationInRange (history : EventHistory) (range : EventRange)
    (result : Nat) : Option Owner :=
  if range.start ≤ result && result < range.end then
    match history.lookup result with
    | some { event := .invocationEndExact ended _ _ _ _ _ _, frameStart := childStart } =>
        if range.start ≤ childStart && childStart ≤ result && childStart < range.end then
          match history.lookup childStart with
          | some { event := .invocationStart started, frameStart := annotatedStart } =>
              if annotatedStart = childStart && started = ended then some ended else none
          | _ => none
        else none
    | _ => none
  else none

def resultOwner? : Event → Option Owner
  | .resultExact owner _ _ _ _ _ | .resultCoefficient owner _ => some owner
  | _ => none

def transferOwner? : Event → Option Owner
  | .boundTransfer owner _ => some owner
  | .appliedRelation owner _ _ _ _ _ => some owner
  | _ => none

def relationOwner? : Event → Option Owner
  | .appliedRelation owner _ _ _ _ _ => some owner
  | _ => none

def projectionAvailable : Event → Projection → Bool
  | .resultExact _ _ coefficientBound _ _ _, .coefficient =>
      decide (coefficientBound ≠ .missing)
  | .resultExact .., .summary => true
  | .resultCoefficient _ bound, .coefficient => decide (bound ≠ .missing)
  | _, _ => false

def termExists (terms : List Term) (ordinal : Nat) : Bool :=
  (List.get?Internal terms ordinal).isSome

def exactTermExists (history : EventHistory) (state : ReplayState)
    (event ordinal : Nat) : Bool :=
  prior state event && match eventAt? history event with
    | some (.resultExact _ terms _ _ _ _) | some (.invocationEndExact _ _ terms _ _ _ _) =>
        termExists terms ordinal
    | _ => false

def relationSourceValid (history : EventHistory) (state : ReplayState) (owner : Owner)
    (application ordinal : Nat) : Bool :=
  sameFrame history state owner application && match eventAt? history application with
    | some (.appliedRelation relationOwner _ _ _ _ rule) =>
        decide (relationOwner = owner) && match rule with
          | .universal computed _ _ rhsResult =>
              match eventAt? history computed with
              | some (.specializationComputed computedOwner _ source) =>
                  decide (computedOwner.scope = owner.scope) && decide (source.end = computed) &&
                    specializationRangeValid history { state with cursor := computed } source &&
                    (match completedInvocationInRange history source rhsResult with
                    | some rhsOwner => decide (rhsOwner.scope = owner.scope)
                    | none => false) && exactTermExists history state rhsResult ordinal
              | _ => false
          | .gadget _ _ _ inputResult =>
              sameFrame history state owner inputResult &&
                exactTermExists history state inputResult ordinal
    | _ => false

def valueRefValid (history : EventHistory) (state : ReplayState) (owner : Owner) :
    ValueRef → Bool
  | .predecessor position bindingEvent projection =>
      sameFrame history state owner bindingEvent &&
        match eventAt? history bindingEvent with
        | some (.predecessor consumer inputPosition predecessor sourceResult) =>
            decide (consumer = owner) && decide (inputPosition = position) &&
              sameFrame history state owner sourceResult &&
              match eventAt? history sourceResult with
              | some source =>
                  decide (resultOwner? source = some ⟨owner.scope, predecessor⟩) &&
                    projectionAvailable source projection
              | none => false
        | _ => false
  | .result event projection => sameFrame history state owner event &&
      match eventAt? history event with
      | some source => projectionAvailable source projection && match resultOwner? source with
          | some resultOwner => decide (resultOwner.scope = owner.scope)
          | none => false
      | none => false
  | .transfer event =>
      sameFrame history state owner event && match eventAt? history event with
      | some source => decide (transferOwner? source = some owner)
      | none => false

def scaleValid (history : EventHistory) (state : ReplayState) (owner : Owner) :
    Scale → Bool
  | .value value => valueRefValid history state owner value
  | .magnitude _ => true

def preimageSourceValid (document : CertificateDocument) (source : ExpressionRef) : Bool :=
  let event? := match document.expressions.lookup source.row with
    | some { descriptor := .event (.sampler event), .. }
    | some { descriptor := .operation (.event (.sampler event)) _, .. } => some event
    | _ => none
  match event? with
  | some event => match document.events.lookup event.row with
      | some (.sampler _ (.preimage ..) _) => true
      | _ => false
  | none => false

def ruleValid (document : CertificateDocument) (history : EventHistory) (state : ReplayState)
    (owner : Owner) : BoundRule → Bool
  | .authority (.relationPreimageSource source) =>
      preimageSourceValid document source
  | .authority _ => true
  | .identity input => valueRefValid history state owner input
  | .sum inputs => inputs.all (valueRefValid history state owner)
  | .scale value scale =>
      valueRefValid history state owner value && scaleValid history state owner scale
  | .monomialProduct monomial factors =>
      monomialValid document monomial &&
        factors.all (fun factor => valueRefValid history state owner factor.bound)
  | .product left right _ | .tensor left right _ _ =>
      valueRefValid history state owner left && valueRefValid history state owner right

def termRefValid (history : EventHistory) (state : ReplayState)
    (reference : TermRef) : Bool :=
  prior state reference.valueEvent &&
    match eventAt? history reference.valueEvent with
    | some (.resultExact _ terms _ _ _ _) => termExists terms reference.termOrdinal
    | _ => false

def termRefOwnerLocal (history : EventHistory) (state : ReplayState) (owner : Owner)
    (reference : TermRef) : Bool :=
  sameFrame history state owner reference.valueEvent && termRefValid history state reference &&
    match eventAt? history reference.valueEvent with
    | some source => match resultOwner? source with
        | some sourceOwner => decide (sourceOwner.scope = owner.scope)
        | none => false
    | none => false

def summaryEvidenceValid (history : EventHistory) (state : ReplayState) (frame : Frame) :
    Option ValueRef → Bool
  | none => true
  | some (.result event .summary) => sameFrame history state frame.root event &&
      match eventAt? history event with | some (.resultExact ..) => true | _ => false
  | some (.transfer event) => sameFrame history state frame.root event &&
      match eventAt? history event with
      | some source => match transferOwner? source with
          | some owner => decide (owner.scope = frame.root.scope)
          | none => false
      | none => false
  | _ => false

def boundProducerValid (history : EventHistory) (state : ReplayState) (owner : Owner)
    (producer : Nat) : Bool :=
  sameFrame history state owner producer && match eventAt? history producer with
    | some (.boundTransfer producerOwner _) => decide (producerOwner = owner)
    | _ => false

def summaryProducerValid (history : EventHistory) (state : ReplayState) (owner : Owner) :
    Bound → Option Nat → Bool
  | .exactZero, none => true
  -- `none` records that no single transfer is the summary endpoint. Semantic replay must
  -- establish any nonzero summary separately; structural validity grants no such theorem.
  | .finite _, none | .large, none => true
  | .finite _, some producer | .large, some producer =>
      boundProducerValid history state owner producer
  | _, _ => false

def append (state : ReplayState) (frames : List Frame) : ReplayState :=
  ⟨state.cursor + 1, frames⟩

/-- One honest chronological replay step. Failed structural references return `none`. -/
def stepAt (document : CertificateDocument) (history : EventHistory) (state : ReplayState) :
    Option ReplayState := do
  let entry ← history.lookup state.cursor
  let event := entry.event
  let expectedFrameStart := match event, state.frames.head? with
    | .invocationStart _, _ => state.cursor
    | _, some frame => frame.start
    | _, none => 0
  if entry.frameStart != expectedFrameStart then none else
  let accept (frames : List Frame) := some (append state frames)
  match event with
  | .invocationStart root =>
      if ownerValid document root then match document.expressions.lookup root.expression.row with
        | some _ => accept (⟨root, state.cursor⟩ :: state.frames)
        | none => none
      else none
  | .predecessor consumer inputPosition predecessor sourceResult =>
      if currentScope state consumer && ownerValid document consumer &&
          (match document.expressions.lookup consumer.expression.row with
          | some expression => expression.inputs.get? inputPosition = some predecessor
          | none => false) &&
          sameFrame history state consumer sourceResult &&
          (match eventAt? history sourceResult with
          | some source =>
              decide (resultOwner? source = some ⟨consumer.scope, predecessor⟩)
          | none => false)
      then
        accept state.frames
      else none
  | .resultExact owner terms coefficientBound coefficientProducer summary summaryProducer =>
      if currentScope state owner && ownerValid document owner &&
          terms.all (fun term => monomialValid document term.monomial) &&
          decide (coefficientBound ≠ .missing) &&
          boundProducerValid history state owner coefficientProducer &&
          summaryProducerValid history state owner summary summaryProducer then
        accept state.frames
      else none
  | .resultCoefficient owner _ =>
      if currentScope state owner && ownerValid document owner then accept state.frames else none
  | .invocationEndExact root preFoldEvent terms coefficientBound coefficientProducer summary
      summaryProducer =>
      match state.frames with
      | [] => none
      | frame :: frames =>
          if decide (frame.root = root) && sameFrame history state root preFoldEvent &&
              (match eventAt? history preFoldEvent with
              | some (.preFoldPolynomial resultEvent preFoldTerms preFoldSummary _) =>
                  decide (preFoldTerms = terms) && decide (preFoldSummary = summary) &&
                    sameFrame history state root resultEvent &&
                    (match eventAt? history resultEvent with
                    | some (.resultExact resultOwner resultTerms resultCoefficientBound
                        resultCoefficientProducer resultSummary resultSummaryProducer) =>
                        decide (resultOwner = root) && decide (resultTerms = terms) &&
                          decide (resultCoefficientBound = coefficientBound) &&
                          decide (resultCoefficientProducer = coefficientProducer) &&
                          decide (resultSummary = summary) &&
                          decide (resultSummaryProducer = summaryProducer)
                    | _ => false)
              | _ => false) &&
              boundProducerValid history state root coefficientProducer &&
              summaryProducerValid history state root summary summaryProducer then
            accept frames
          else none
  | .specializationComputed owner dispatch source =>
      if currentScope state owner && ownerValid document owner &&
          (document.programs.lookup dispatch.preimageFamily.row).isSome &&
          (document.expressions.lookup dispatch.preimageSource.row).isSome &&
          (document.expressions.lookup dispatch.trapdoorSource.row).isSome &&
          specializationRangeValid history state source then accept state.frames else none
  | .appliedRelation owner sourceMonomial _ orderedStart orderedEnd rule =>
      let ruleOk := match rule with
        | .universal computed lhs _ rhsResult =>
            sameFrame history state owner computed && monomialValid document lhs &&
              (match eventAt? history computed with
              | some (.specializationComputed computedOwner _ source) =>
                  decide (computedOwner.scope = owner.scope) && decide (source.end = computed) &&
                    specializationRangeValid history { state with cursor := computed } source &&
                    (match completedInvocationInRange history source rhsResult with
                    | some rhsOwner => decide (rhsOwner.scope = owner.scope)
                    | none => false)
              | _ => false)
        | .gadget gadget decomposition input inputResult =>
            ownerValid document gadget && ownerValid document decomposition &&
              decide (gadget.scope = owner.scope) &&
              decide (decomposition.scope = owner.scope) &&
              (document.expressions.lookup input.row).isSome &&
              sameFrame history state owner inputResult &&
              (match eventAt? history inputResult with
              | some source => decide (resultOwner? source = some ⟨owner.scope, input⟩)
              | none => false)
      if currentScope state owner && ownerValid document owner &&
          monomialValid document sourceMonomial && decide (orderedStart ≤ orderedEnd) &&
          decide (orderedEnd ≤ sourceMonomial.orderedFactors.length) && ruleOk then
        accept state.frames
      else none
  | .boundTransfer owner rule =>
      if currentScope state owner && ownerValid document owner &&
          ruleValid document history state owner rule
      then accept state.frames else none
  | .coefficientMerge merge =>
      let sourceOk := match merge.source with
        | .operator inputs =>
            termRefOwnerLocal history state merge.owner inputs.1 &&
              termRefOwnerLocal history state merge.owner inputs.2
        | .relation application ordinal =>
            relationSourceValid history state merge.owner application ordinal
      if currentScope state merge.owner && ownerValid document merge.owner &&
          monomialValid document merge.output && sourceOk then accept state.frames else none
  | .preFoldPolynomial resultEvent terms _ evidence =>
      match state.frames with
      | [] => none
      | frame :: _ =>
          if terms.all (fun term => monomialValid document term.monomial) &&
              sameFrame history state frame.root resultEvent &&
              (match eventAt? history resultEvent with
              | some (.resultExact owner _ _ _ _ _) => decide (owner = frame.root)
              | _ => false) && summaryEvidenceValid history state frame evidence
          then accept state.frames else none
  | .survivorFold _ bound =>
      if prior state bound then
        match eventAt? history bound with
        | some source => match transferOwner? source with
            | some owner =>
                if sameFrame history state owner bound then accept state.frames else none
            | none => none
        | none => none
      else none

/-- Replay one exact-result event from its recorded row and checked branch guard. -/
theorem stepAt_resultExact
    (document : CertificateDocument) (history : EventHistory) (state : ReplayState)
    (owner : Owner) (terms : List Term) (coefficientBound : Bound)
    (coefficientProducer : Nat) (summary : Bound) (summaryProducer : Option Nat)
    (frameStart : Nat)
    (entryAt : history.lookup state.cursor =
      some ⟨.resultExact owner terms coefficientBound coefficientProducer summary summaryProducer,
        frameStart⟩)
    (frameAt : (frameStart != match
        (Event.resultExact owner terms coefficientBound coefficientProducer summary summaryProducer),
        state.frames.head? with
      | Event.invocationStart _, _ => state.cursor
      | _, some frame => frame.start
      | _, none => 0) = false)
    (guardAt : (currentScope state owner && ownerValid document owner &&
        terms.all (fun term => monomialValid document term.monomial) &&
        decide (coefficientBound ≠ .missing) &&
        boundProducerValid history state owner coefficientProducer &&
        summaryProducerValid history state owner summary summaryProducer) = true) :
    stepAt document history state = some (append state state.frames) := by
  simp only [stepAt, entryAt, Option.bind_eq_bind, Option.bind_some]
  rw [frameAt]
  simp only [Bool.false_eq_true, ↓reduceIte]
  rw [guardAt]
  rfl

def replayBlock (document : CertificateDocument) (history : EventHistory) (endExclusive : Nat)
    (state : ReplayState) : Option ReplayState :=
  let steps := Nat.min (endExclusive - state.cursor) 4
  let rec run : Nat → ReplayState → Option ReplayState
    | 0, current => some current
    | remaining + 1, current => do
        let next ← stepAt document history current
        run remaining next
  run steps state

def replayRange (document : CertificateDocument) (history : EventHistory) (endExclusive : Nat)
    (state : ReplayState) : Option ReplayState := do
  let steps := endExclusive - state.cursor
  if state.cursor ≤ endExclusive && endExclusive ≤ history.size && steps ≤ 4 then
    let finish ← replayBlock document history endExclusive state
    if finish.cursor = endExclusive then some finish else none
  else none

def replay (document : CertificateDocument) (history : EventHistory) : Option ReplayState :=
  replayRange document history history.size initialState

inductive ReplayChain (document : CertificateDocument) (history : EventHistory) :
    ReplayState → ReplayState → Prop
  | chunk {start finish : ReplayState} (endExclusive : Nat)
      (replayed : replayRange document history endExclusive start = some finish) :
      ReplayChain document history start finish
  | trans {start middle finish : ReplayState}
      (left : ReplayChain document history start middle)
      (right : ReplayChain document history middle finish) :
      ReplayChain document history start finish

def Valid (document : CertificateDocument) (history : EventHistory) : Prop :=
  history.wellFormed = true ∧
    ∃ state, ReplayChain document history initialState state ∧ state.cursor = history.size ∧
      state.frames = []

theorem validOfReplay (document : CertificateDocument) (history : EventHistory)
    (state : ReplayState)
    (replayed : replayRange document history history.size initialState = some state)
    (historyValid : history.wellFormed = true)
    (finished : state.cursor = history.size) (closed : state.frames = []) :
    Valid document history :=
  ⟨historyValid, state, .chunk history.size replayed, finished, closed⟩

end Mxx.Certificate.OperationalNoise.CertificateABI
