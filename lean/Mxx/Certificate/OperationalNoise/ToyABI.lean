import Mxx.Certificate.OperationalNoise.RelationReplay
import Mxx.Certificate.OperationalNoise.SchemaV1

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.ToyABI

open Mxx.Certificate.OperationalNoise
open EventReplay
open SchemaV1

/-! The fixed Lean ABI for the Rust `singleton-preimage-gaussian-v1` audit slice. -/

structure ToyRequest where
  targetId : String
  environment : List Unit
  layouts : List Unit
deriving DecidableEq, Repr

structure ToyParameters where
  plaintextModulus : String
  ciphertextModulus : String
  ringDimension : Nat
  trapdoorRows : String
  trapdoorSigma : String
  gadgetBase : String
  digitCount : String
  preimageMaximumAbsoluteCoefficient : String
  gaussianSigma : String
  gaussianMaximumAbsoluteCoefficient : String
deriving DecidableEq, Repr

structure ToySource where
  schemaId : String
  schemaVersion : Nat
  abi : String
  rustProjectionVersion : String
  leanAbiVersion : String
  request : ToyRequest
  parameters : ToyParameters
deriving DecidableEq, Repr

def expectedSource : ToySource where
  schemaId := "mxx.operational-noise.toy-source"
  schemaVersion := 1
  abi := "singleton-preimage-gaussian-v1"
  rustProjectionVersion := "operational-noise-certificate-v1"
  leanAbiVersion := "toy-replay-v1"
  request := ⟨"singleton-preimage-gaussian", [], []⟩
  parameters := ⟨"2", "257", 1, "1", "3", "4", "2", "8", "1", "1"⟩

structure ToyRows where
  expressions : List ExpressionRef
  program : ProgramRef
  sources : List SourceRef
  events : List EventRef
  root : ExpressionRef
deriving DecidableEq, Repr

def expectedRows : ToyRows where
  expressions := (List.range 13).map ExpressionRef.mk
  program := ⟨0⟩
  sources := [⟨0⟩, ⟨1⟩]
  events := [⟨0⟩, ⟨1⟩, ⟨2⟩, ⟨3⟩]
  root := ⟨12⟩

inductive ToyScope where
  | closed (rootExpressionRow : Nat)
  | program (programRow : Nat)
deriving DecidableEq, Repr

structure ToyOwner where
  scope : ToyScope
  expressionRow : Nat
deriving DecidableEq, Repr

structure ToyMonomial where
  centralFactors : List ToyOwner
  orderedFactors : List ToyOwner
deriving DecidableEq, Repr

structure ToyTerm where
  monomial : ToyMonomial
  coefficient : Int
deriving DecidableEq, Repr

inductive ToySummary where
  | exactZero
  | finite (maximumAbsoluteCoefficient : Nat)
deriving DecidableEq, Repr

inductive ToyValue where
  | exact (terms : List ToyTerm) (summary : ToySummary)
  | coefficient (bound : ToySummary)
deriving DecidableEq, Repr

inductive ToyProjection where | coefficient | summary
deriving DecidableEq, Repr

inductive ToyValueRef where
  | predecessor (inputPosition : Nat) (projection : ToyProjection)
  | result (event : Nat) (projection : ToyProjection)
  | transfer (event : Nat)
deriving DecidableEq, Repr

inductive ToyScale where
  | value (value : ToyValueRef)
  | magnitude (magnitude : Nat)
deriving DecidableEq, Repr

structure ToyFactorEvidence where
  bound : ToyValueRef
  isConstantPolynomial : Bool
  supportUpper : Option Nat
deriving DecidableEq, Repr

structure ToyProductFacts where
  leftIsConstantPolynomial : Bool
  rightIsConstantPolynomial : Bool
  rightKnownZeroRows : Option Nat
  leftSupportUpper : Option Nat
  rightSupportUpper : Option Nat
deriving DecidableEq, Repr

inductive ToyAuthority where
  | operator
  | relationPreimageSource (source : Nat)
deriving DecidableEq, Repr

inductive ToyBoundRule where
  | authority (authority : ToyAuthority)
  | sum (inputs : List ToyValueRef)
  | scale (value : ToyValueRef) (scale : ToyScale)
  | monomialProduct (monomial : ToyMonomial) (factors : List ToyFactorEvidence)
  | product (left right : ToyValueRef) (facts : ToyProductFacts)
deriving DecidableEq, Repr

structure ToyDispatch where
  preimageFamily : Nat
  preimageSource : Nat
  trapdoorSource : Nat
deriving DecidableEq, Repr

structure ToyRange where
  start : Nat
  «end» : Nat
deriving DecidableEq, Repr

structure ToyLayout where
  name : String
  rowStride : Nat
  columnStride : Nat
deriving DecidableEq, Repr

structure ToyTermRef where
  valueEvent : Nat
  termOrdinal : Nat
deriving DecidableEq, Repr

inductive ToyMergeSource where
  | operator (inputs : ToyTermRef × ToyTermRef)
  | relation (application sourceTermOrdinal : Nat)
deriving DecidableEq, Repr

structure ToyMerge where
  owner : ToyOwner
  source : ToyMergeSource
  output : ToyMonomial
  signedContribution : Int
deriving DecidableEq, Repr

inductive ToyEvent where
  | invocationStart (root : ToyOwner)
  | predecessor (consumer : ToyOwner) (inputPosition predecessor sourceResult : Nat)
  | result (owner : ToyOwner) (value : ToyValue)
  | invocationEnd (root : ToyOwner) (result : ToyValue)
  | specializationComputed (owner : ToyOwner) (dispatch : ToyDispatch) (source : ToyRange)
  | appliedUniversal (owner : ToyOwner) (sourceMonomial : ToyMonomial)
      (outerCoefficient : Int) (orderedStart orderedEndExclusive computed : Nat)
      (lhs : ToyMonomial) (lhsLayout : Option ToyLayout) (rhsResult : Nat)
  | boundTransfer (owner : ToyOwner) (rule : ToyBoundRule)
  | coefficientMerge (merge : ToyMerge)
  | preFoldPolynomial (terms : List ToyTerm) (summary : ToySummary)
      (summaryEvidence : Option ToyValueRef)
  | survivorFold (coefficient : Int) (bound : Nat)
deriving DecidableEq, Repr

abbrev ToyEnv := ToyOwner -> Int

def ToyMonomial.eval (env : ToyEnv) (value : ToyMonomial) : Int :=
  (value.centralFactors.map env).prod * (value.orderedFactors.map env).prod

def ToyTerms.eval (env : ToyEnv) (terms : List ToyTerm) : Int :=
  (terms.map (fun term => term.coefficient * term.monomial.eval env)).sum

def ToySummary.Interprets (modulus : Nat) (summary : ToySummary) (value : Int) : Prop :=
  match summary with
  | .exactZero => centeredNorm modulus value = 0
  | .finite maximum => centeredNorm modulus value <= maximum

def ToyValue.Interprets (modulus : Nat) (env : ToyEnv) (actual : Int) : ToyValue -> Prop
  | .exact terms summary =>
      ∃ remainder, (actual - ToyTerms.eval env terms) % Int.ofNat modulus = remainder %
        Int.ofNat modulus ∧ summary.Interprets modulus remainder
  | .coefficient summary => summary.Interprets modulus actual

def ToyOwner.toKeyFactor (owner : ToyOwner) : Nat := owner.expressionRow

def ToyMonomial.toCore (value : ToyMonomial) : MonomialKey :=
  { centralFactors := value.centralFactors.map ToyOwner.toKeyFactor
    orderedFactors := value.orderedFactors.map ToyOwner.toKeyFactor }

def ToyTerm.toCore (value : ToyTerm) : ExactTerm :=
  { coefficient := value.coefficient
    key := value.monomial.toCore }

def toyMatrix (rows columns : Nat) : ValueType := .matrix "257" 1 rows columns

def toyWire (node : Nat) : ObservedWire :=
  { stage := "consumer"
    definition := .root
    path := 0
    node
    port := 0 }

def toyProgramInput : ProgramInput where
  valueType := .int
  trustedIndexRange := some ⟨0, 1⟩

def toyFamily : Family where
  domain := ⟨0, 1⟩
  elementType := toyMatrix 4 1
  reducible := false
  artifact := none

def toyProgram : SchemaV1.ProgramRow where
  signature := [toyProgramInput]
  output := toyMatrix 4 1
  family := some toyFamily
  root := ⟨2⟩

def zeroConstant : Constant where
  valueType := .int
  value := .int "0"

def oneConstant : Constant where
  valueType := .int
  value := .int "1"

def zeroSource : SchemaV1.SourceRow := .constant zeroConstant
def oneSource : SchemaV1.SourceRow := .constant oneConstant

def expectedDocument : Document :=
  { schemaId := "mxx.operational-noise.certificate"
    schemaVersion := 1
    plaintextModulus := "2"
    ciphertextModulus := "257"
    ringDimension := 1
    expressions :=
      [ { descriptor := .operation (.event (.sampler ⟨0⟩)) (toyMatrix 1 4)
          inputs := []
          program := none },
        { descriptor := .operation (.event (.sampler ⟨1⟩)) (toyMatrix 1 1)
          inputs := []
          program := none },
        { descriptor := .operation (.event (.sampler ⟨2⟩)) (toyMatrix 4 1)
          inputs := []
          program := none },
        { descriptor := .operation (.event (.sampler ⟨3⟩)) (toyMatrix 1 1)
          inputs := []
          program := none },
        { descriptor := .operation (.stable (.trapdoor (.generate "trapdoor-sample" [4, 2]
            (some ⟨0⟩) "value"))) .trapdoor
          inputs := []
          program := none },
        { descriptor := .source (.direct ⟨0⟩)
          inputs := []
          program := none },
        { descriptor := .source (.direct ⟨1⟩)
          inputs := []
          program := none },
        { descriptor := .operation (.stable .programCall) (toyMatrix 4 1)
          inputs := [⟨5⟩]
          program := some ⟨0⟩ },
        { descriptor := .operation (.stable (.matrix .multiply)) (toyMatrix 1 1)
          inputs := [⟨0⟩, ⟨7⟩]
          program := none },
        { descriptor := .operation (.stable (.matrix .scale)) (toyMatrix 4 1)
          inputs := [⟨7⟩, ⟨6⟩]
          program := none },
        { descriptor := .operation (.stable (.matrix .multiply)) (toyMatrix 1 1)
          inputs := [⟨0⟩, ⟨9⟩]
          program := none },
        { descriptor := .operation (.stable (.matrix .subtract)) (toyMatrix 1 1)
          inputs := [⟨10⟩, ⟨1⟩]
          program := none },
        { descriptor := .operation (.stable (.matrix .add)) (toyMatrix 1 1)
          inputs := [⟨11⟩, ⟨3⟩]
          program := none } ]
    programs := [toyProgram]
    sources := [zeroSource, oneSource]
    events :=
      [ .sampler (toyWire 0) (.trapdoor (toyMatrix 1 4)
          "{\"tag\":\"Rational\",\"value\":{\"numerator\":\"3\",\"denominator\":\"1\"}}"
          4 2 "8") none,
        .sampler (toyWire 1) (.uniformResidue (toyMatrix 1 1)) none,
        .sampler (toyWire 2) (.preimage (toyMatrix 4 1) "8") none,
        .sampler (toyWire 6) (.gaussian (toyMatrix 1 1)
          "{\"tag\":\"Rational\",\"value\":{\"numerator\":\"1\",\"denominator\":\"1\"}}"
          "1") none ]
    indexUses := []
    sliceGroups := []
    residualRoot := .closed ⟨12⟩ }

def ToyDocumentValid (document : Document) : Prop := document = expectedDocument

def o (row : Nat) : ToyOwner := ⟨.closed 12, row⟩
def m (rows : List Nat) : ToyMonomial := ⟨[], rows.map o⟩
def t (coefficient : Int) (rows : List Nat) : ToyTerm := ⟨m rows, coefficient⟩
def ez (rows : List Nat) : ToyValue := .exact [t 1 rows] .exactZero
def c0 : ToyValue := .coefficient .exactZero
def c1 : ToyValue := .coefficient (.finite 1)
def empty0 : ToyValue := .exact [] .exactZero
def empty1 : ToyValue := .exact [] (.finite 1)
def predC (position : Nat) : ToyValueRef := .predecessor position .coefficient
def noFacts : ToyProductFacts := ⟨false, false, none, none, none⟩

def expectedEvents : List ToyEvent :=
  [ .invocationStart (o 12),
    .boundTransfer (o 3) (.authority .operator), .result (o 3) (ez [3]),
    .boundTransfer (o 1) (.authority .operator), .result (o 1) (ez [1]),
    .boundTransfer (o 6) (.authority .operator), .result (o 6) c1,
    .boundTransfer (o 5) (.authority .operator), .result (o 5) c0,
    .predecessor (o 7) 0 5 8,
    .boundTransfer (o 7) (.authority (.relationPreimageSource 2)),
    .result (o 7) (ez [7]),
    .predecessor (o 9) 0 7 11, .predecessor (o 9) 1 6 6,
    .boundTransfer (o 9) (.scale (predC 0) (.value (predC 1))),
    .result (o 9) (ez [7]),
    .boundTransfer (o 0) (.authority .operator), .result (o 0) (ez [0]),
    .predecessor (o 10) 0 0 17, .predecessor (o 10) 1 9 15,
    .boundTransfer (o 10) (.product (predC 0) (predC 1) noFacts),
    .coefficientMerge ⟨o 10, .operator (⟨17, 0⟩, ⟨15, 0⟩), m [0, 7], 1⟩,
    .invocationStart (o 8),
    .boundTransfer (o 5) (.authority .operator), .result (o 5) c0,
    .predecessor (o 7) 0 5 24,
    .boundTransfer (o 7) (.authority (.relationPreimageSource 2)),
    .result (o 7) (ez [7]),
    .boundTransfer (o 0) (.authority .operator), .result (o 0) (ez [0]),
    .predecessor (o 8) 0 0 29, .predecessor (o 8) 1 7 27,
    .boundTransfer (o 8) (.product (predC 0) (predC 1) noFacts),
    .coefficientMerge ⟨o 8, .operator (⟨29, 0⟩, ⟨27, 0⟩), m [0, 7], 1⟩,
    .result (o 8) (ez [0, 7]),
    .preFoldPolynomial [t 1 [0, 7]] .exactZero none,
    .invocationEnd (o 8) (ez [0, 7]),
    .invocationStart (o 1),
    .boundTransfer (o 1) (.authority .operator), .result (o 1) (ez [1]),
    .preFoldPolynomial [t 1 [1]] .exactZero none,
    .invocationEnd (o 1) (ez [1]),
    .specializationComputed (o 5) ⟨0, 2, 4⟩ ⟨22, 42⟩,
    .appliedUniversal (o 10) (m [0, 7]) 1 0 2 42 (m [0, 7]) none 41,
    .coefficientMerge ⟨o 10, .relation 43 0, m [1], 1⟩,
    .result (o 10) (ez [1]),
    .predecessor (o 11) 0 10 45, .predecessor (o 11) 1 1 4,
    .boundTransfer (o 11) (.sum [predC 0, predC 1]),
    .coefficientMerge ⟨o 11, .operator (⟨45, 0⟩, ⟨4, 0⟩), m [1], -1⟩,
    .result (o 11) empty0,
    .predecessor (o 12) 0 11 50, .predecessor (o 12) 1 3 2,
    .boundTransfer (o 12) (.sum [predC 0, predC 1]),
    .boundTransfer (o 12) (.monomialProduct (m [3])
      [⟨.result 2 .coefficient, false, none⟩]),
    .survivorFold 1 54,
    .result (o 12) empty1,
    .preFoldPolynomial [] (.finite 1) (some (.result 56 .summary)),
    .invocationEnd (o 12) empty1 ]

def eventAt? (events : List ToyEvent) (index : Nat) : Option ToyEvent := events[index]?

def ToyEventsValid (events : List ToyEvent) : Prop :=
  events.length = 59 ∧
    ∀ index, index < 59 → eventAt? events index = eventAt? expectedEvents index

def ToyValid (source : ToySource) (document : Document) (rows : ToyRows)
    (events : List ToyEvent) : Prop :=
  source = expectedSource ∧ document = expectedDocument ∧ rows = expectedRows ∧
    ToyEventsValid events

instance (source : ToySource) (document : Document) (rows : ToyRows)
    (events : List ToyEvent) : Decidable (ToyValid source document rows events) := by
  unfold ToyValid ToyEventsValid
  infer_instance

def finalValue (events : List ToyEvent) : ToyValue :=
  match eventAt? events 58 with | some (.invocationEnd _ value) => value | _ => empty0

def ToyResidual (_events : List ToyEvent) (env : ToyEnv) : Int :=
  (m [0, 7]).eval env - (m [1]).eval env + env (o 3)

structure ToyReplayWitness (events : List ToyEvent) where
  env : ToyEnv
  gaussianEvent : eventAt? events 2 = some (.result (o 3) (ez [3]))
  gaussianBound : centeredNorm 257 (env (o 3)) ≤ 1
  universalEvent : eventAt? events 43 = some (.appliedUniversal (o 10) (m [0, 7])
    1 0 2 42 (m [0, 7]) none 41)
  universalRelation : (m [0, 7]).eval env % Int.ofNat 257 =
    (m [1]).eval env % Int.ofNat 257

def toyValuation (env : ToyEnv) (key : MonomialKey) : Int :=
  (key.orderedFactors.map (fun row => env (o row))).prod

def universalContext : MonomialContext := ⟨[], [], []⟩

/-- Context and its outer coefficient preserve an exact modular base relation. -/
theorem relationReplacement_modular (modulus : Nat) (valuation : MonomialKey → Int)
    (context : MonomialContext) (contextMultiplier outerCoefficient : Int)
    (left right : Polynomial)
    (contextSound : ∀ key, valuation (context.plug key) = contextMultiplier * valuation key)
    (baseRelation : evaluatePolynomial valuation left % Int.ofNat modulus =
      evaluatePolynomial valuation right % Int.ofNat modulus) :
    evaluatePolynomial valuation (relationReplacement context outerCoefficient left) %
        Int.ofNat modulus =
      evaluatePolynomial valuation (relationReplacement context outerCoefficient right) %
        Int.ofNat modulus := by
  rw [evaluate_relationReplacement valuation context contextMultiplier outerCoefficient left
      contextSound,
    evaluate_relationReplacement valuation context contextMultiplier outerCoefficient right
      contextSound]
  calc
    (outerCoefficient * contextMultiplier * evaluatePolynomial valuation left) %
        Int.ofNat modulus =
      ((outerCoefficient * contextMultiplier) % Int.ofNat modulus *
        (evaluatePolynomial valuation left % Int.ofNat modulus)) % Int.ofNat modulus :=
      Int.mul_emod _ _ _
    _ = ((outerCoefficient * contextMultiplier) % Int.ofNat modulus *
        (evaluatePolynomial valuation right % Int.ofNat modulus)) % Int.ofNat modulus := by
      rw [baseRelation]
    _ = (outerCoefficient * contextMultiplier * evaluatePolynomial valuation right) %
        Int.ofNat modulus := (Int.mul_emod _ _ _).symm

theorem fixed_relation_replay {events : List ToyEvent} (witness : ToyReplayWitness events) :
    evaluatePolynomial (toyValuation witness.env)
        (relationReplacement universalContext 1 [(t 1 [0, 7]).toCore]) % 257 =
      evaluatePolynomial (toyValuation witness.env)
        (relationReplacement universalContext 1 [(t 1 [1]).toCore]) % 257 := by
  apply relationReplacement_modular 257 (toyValuation witness.env) universalContext 1 1
  · intro key
    simp [universalContext, MonomialContext.plug, toyValuation]
  · simpa [evaluatePolynomial, ToyTerm.toCore, ToyMonomial.toCore, ToyOwner.toKeyFactor,
      toyValuation, t, m, ToyMonomial.eval, o] using witness.universalRelation

theorem fixed_merge_cancels :
    coefficient (t 1 [1]).toCore.key [(t 1 [1]).toCore, (t (-1) [1]).toCore] = 0 := by
  decide

theorem fixed_product_merge :
    operatorProductContribution (t 1 [0]).toCore (t 1 [7]).toCore false false =
      (t 1 [0, 7]).toCore := by
  decide

def finiteOne : CoeffClass := .finite ⟨1, by decide⟩

theorem fixed_bound_replay {events : List ToyEvent} (witness : ToyReplayWitness events) :
    centeredNorm 257 (ToyResidual events witness.env) ≤ 1 := by
  have relationReplay := fixed_relation_replay witness
  have mergeReplay := fixed_merge_cancels
  have productReplay := operatorProductContribution_natAbs_le
    (t 1 [0]).toCore (t 1 [7]).toCore false false 1 1 (by decide) (by decide)
  rw [fixed_product_merge] at productReplay
  have cancellationMagnitude :
      (coefficient (t 1 [1]).toCore.key
        [(t 1 [1]).toCore, (t (-1) [1]).toCore]).natAbs = 0 := by
    rw [mergeReplay]
    decide
  have scaleReplay :
      (coefficient (t 1 [1]).toCore.key
          [(t 1 [1]).toCore, (t (-1) [1]).toCore]).natAbs * (1 * 1) ≤
        (coefficient (t 1 [1]).toCore.key
          [(t 1 [1]).toCore, (t (-1) [1]).toCore]).natAbs * (1 * 1) :=
    boundTransfer_scale productReplay
  rw [cancellationMagnitude] at scaleReplay
  have cancelledReplay : 0 ≤ 0 := by exact scaleReplay
  have sumReplay : 0 + centeredNorm 257 (witness.env (o 3)) ≤ 0 + 1 :=
    boundTransfer_sum cancelledReplay witness.gaussianBound
  have gaussianReplay : centeredNorm 257 (witness.env (o 3)) ≤ 1 := by
    simpa using sumReplay
  have gaussianClass : finiteOne.Interprets (centeredNorm 257 (witness.env (o 3))) := by
    simpa [finiteOne, CoeffClass.Interprets] using gaussianReplay
  have monomialReplay : (productNonempty finiteOne []).Interprets
      (centeredNorm 257 (witness.env (o 3))) := by
    simpa using productNonempty_sound gaussianClass (.nil : List.Forall₂ _ [] [])
  have gaussianProductReplay : centeredNorm 257 (witness.env (o 3)) ≤ 1 := by
    simpa [productNonempty, finiteOne, CoeffClass.Interprets] using monomialReplay
  have survivorReplay : [centeredNorm 257 (witness.env (o 3))].sum ≤ [1].sum :=
    survivorFold_sound (.cons gaussianProductReplay .nil)
  have finalReplay : 0 + [centeredNorm 257 (witness.env (o 3))].sum ≤ 0 + [1].sum :=
    Nat.add_le_add cancelledReplay survivorReplay
  have invocationReplay : 0 + [1].sum ≤ 0 + [1].sum :=
    preFold_to_invocationEnd cancelledReplay (.cons (Nat.le_refl 1) .nil)
  have relationModular :
      (m [0, 7]).eval witness.env % Int.ofNat 257 =
        (m [1]).eval witness.env % Int.ofNat 257 := by
    simpa [evaluatePolynomial, relationReplacement, universalContext, contextualize,
      scalePolynomial, MonomialContext.plug, toyValuation, ToyTerm.toCore, ToyMonomial.toCore,
      ToyOwner.toKeyFactor, t, m, o, ToyMonomial.eval] using relationReplay
  have residualRemainder : ToyResidual events witness.env % Int.ofNat 257 =
      witness.env (o 3) % Int.ofNat 257 := by
    unfold ToyResidual
    rw [Int.add_emod, Int.sub_emod, relationModular]
    simp
  have centeredEquality : centeredNorm 257 (ToyResidual events witness.env) =
      centeredNorm 257 (witness.env (o 3)) := by
    unfold centeredNorm centeredCoefficient
    simp only [show (257 : Nat) ≠ 0 by decide, ↓reduceIte]
    rw [residualRemainder]
  rw [centeredEquality]
  exact Nat.le_trans (by simpa using finalReplay) (by exact invocationReplay)

theorem replay_sound {source : ToySource} {document : Document} {rows : ToyRows}
    {events : List ToyEvent} (valid : ToyValid source document rows events)
    (witness : ToyReplayWitness events) :
    (finalValue events).Interprets 257 witness.env (ToyResidual events witness.env) := by
  rcases valid with ⟨_, _, _, _, positions⟩
  have finalEvent := positions 58 (by decide)
  simp [expectedEvents, eventAt?] at finalEvent
  unfold finalValue
  change (match events[58]? with
    | some (ToyEvent.invocationEnd _ value) => value
    | _ => empty0).Interprets 257 witness.env (ToyResidual events witness.env)
  rw [finalEvent]
  change ∃ remainder, _ = _ ∧ centeredNorm 257 remainder ≤ 1
  refine ⟨witness.env (o 3), ?_, witness.gaussianBound⟩
  simp only [ToyTerms.eval, List.map_nil, List.sum_nil, Int.sub_zero]
  unfold ToyResidual
  rw [Int.add_emod, Int.sub_emod, witness.universalRelation]
  simp

def ToyOperationalClaim (events : List ToyEvent) (witness : ToyReplayWitness events) : Prop :=
  (finalValue events).Interprets 257 witness.env (ToyResidual events witness.env) ∧
    2 * 2 * centeredNorm 257 (ToyResidual events witness.env) < 257

theorem operationalProof {source : ToySource} {document : Document} {rows : ToyRows}
    {events : List ToyEvent} (valid : ToyValid source document rows events) :
    ∀ witness : ToyReplayWitness events, ToyOperationalClaim events witness := by
  intro witness
  refine ⟨replay_sound valid witness, ?_⟩
  have replayBound := fixed_bound_replay witness
  calc
    2 * 2 * centeredNorm 257 (ToyResidual events witness.env) ≤ 2 * 2 * 1 := by omega
    _ < 257 := by decide

end Mxx.Certificate.OperationalNoise.ToyABI
