import Mxx.Certificate
import MxxWe.Generated.DiamondWeFamily.Ir

open MxxWe.Generated.DiamondWeFamily
open Lean

namespace MxxWe.AnalysisFacts

private def matrixExprShape : Mxx.Certificate.MatrixExpr → String
  | .wire _ => "wire"
  | .zero _ => "zero"
  | .identity _ => "identity"
  | .gadget _ _ => "gadget"
  | .add left right => s!"add({matrixExprShape left},{matrixExprShape right})"
  | .negate value => s!"negate({matrixExprShape value})"
  | .multiply left right => s!"multiply({matrixExprShape left},{matrixExprShape right})"
  | .scalarMultiply _ value => s!"scale({matrixExprShape value})"
  | .rowSlice value _ _ => s!"rowSlice({matrixExprShape value})"
  | .columnSlice value _ _ => s!"columnSlice({matrixExprShape value})"
  | .rowConcat _ => "rowConcat"
  | .columnConcat _ => "columnConcat"
  | .diagonalConcat _ => "diagonalConcat"
  | .rowCoefficientEmbed _ _ value => s!"rowCoefficientEmbed({matrixExprShape value})"
  | .columnBasisEmbed _ _ value => s!"columnBasisEmbed({matrixExprShape value})"
  | .diagonalCoefficientEmbed _ _ value => s!"diagonalCoefficientEmbed({matrixExprShape value})"
  | .diagonalBasisEmbed _ _ value => s!"diagonalBasisEmbed({matrixExprShape value})"
  | .select _ _ => "select"
  | .loopResult _ _ _ => "loopResult"
  | .carriedInput _ _ => "carriedInput"

private def object (fields : List (String × Json)) : Json := Json.mkObj fields

private def array (values : List Json) : Json := .arr values.toArray

private def intExprJson : Mxx.Ir.IntExpr → Json
  | .constant value => object [("kind", "constant"), ("value", toJson value)]
  | .parameter name => object [("kind", "parameter"), ("name", name)]
  | .loopIndex slot => object [("kind", "loopIndex"), ("slot", slot)]
  | .add left right =>
      object [("kind", "add"), ("left", intExprJson left), ("right", intExprJson right)]
  | .subtract left right =>
      object [("kind", "subtract"), ("left", intExprJson left), ("right", intExprJson right)]
  | .multiply left right =>
      object [("kind", "multiply"), ("left", intExprJson left), ("right", intExprJson right)]
  | .divide left right =>
      object [("kind", "divide"), ("left", intExprJson left), ("right", intExprJson right)]
  | .roundDivide left right =>
      object [("kind", "roundDivide"), ("left", intExprJson left),
        ("right", intExprJson right)]
  | .log2Ceil value => object [("kind", "log2Ceil"), ("value", intExprJson value)]

private def matrixTypeJson (type : Mxx.Certificate.MatrixTypeExpr) : Json := object [
  ("modulus", intExprJson type.modulus),
  ("ringDimension", intExprJson type.ringDimension),
  ("rows", intExprJson type.rows),
  ("columns", intExprJson type.columns)
]

private def coreNodeRefJson (reference : Mxx.Certificate.CoreNodeRef) : Json := object [
  ("stage", reference.stage.name),
  ("scope", array (reference.scope.path.map toJson)),
  ("node", reference.node.value)
]

private def coreWireRefJson (reference : Mxx.Certificate.CoreWireRef) : Json := object [
  ("stage", reference.stage.name),
  ("scope", array (reference.scope.path.map toJson)),
  ("node", reference.node.value),
  ("port", reference.port)
]

private def templateWireRefJson (reference : Mxx.Certificate.TemplateWireRef) : Json := object [
  ("stage", reference.definition.stage.name),
  ("definition", reference.definition.name),
  ("bodyScope", array (reference.bodyScope.path.map toJson)),
  ("node", reference.node.value),
  ("port", reference.port)
]

private def runtimeRefJson {type : Mxx.Certificate.RuntimeScalarType}
    (reference : Mxx.Certificate.RuntimeExprRef type) : Json :=
  object [("id", reference.id)]

private def instanceFrameJson : Mxx.Certificate.InstanceFrame → Json
  | .subgraphCall callSite =>
      object [("kind", "subgraphCall"), ("callSite", coreNodeRefJson callSite)]
  | .parallelLane loopSite index => object [
      ("kind", "parallelLane"),
      ("loopSite", coreNodeRefJson loopSite),
      ("index", runtimeRefJson index)
    ]
  | .sequentialIteration loopSite index => object [
      ("kind", "sequentialIteration"),
      ("loopSite", coreNodeRefJson loopSite),
      ("index", runtimeRefJson index)
    ]

private def instancePathJson (path : Mxx.Certificate.InstancePathExpr) : Json :=
  array (path.map instanceFrameJson)

private def familyAggregateRefJson : Mxx.Certificate.FamilyAggregateRef → Json
  | .joint joint outputSlot path => object [
      ("kind", "joint"),
      ("joint", joint.name),
      ("outputSlot", outputSlot),
      ("path", instancePathJson path)
    ]
  | .carriedInput carriedSlot => object [
      ("kind", "carriedInput"),
      ("carriedSlot", carriedSlot)
    ]
  | .recurrenceResult recurrence path slot => object [
      ("kind", "recurrenceResult"),
      ("recurrence", coreNodeRefJson recurrence.site),
      ("path", instancePathJson path),
      ("slot", slot)
    ]
  | .familyElement parent index => object [
      ("kind", "familyElement"),
      ("parent", familyAggregateRefJson parent),
      ("index", runtimeRefJson index)
    ]

private def recurrenceInstanceRefJson
    (reference : Mxx.Certificate.SequentialRecurrenceInstanceRef) : Json := object [
  ("recurrence", coreNodeRefJson reference.recurrence.site),
  ("path", instancePathJson reference.path)
]

private def valueInstanceRefJson : Mxx.Certificate.ValueInstanceRef → Json
  | .protocolInput input =>
      object [("kind", "protocolInput"), ("input", input.name)]
  | .concrete wire => object [("kind", "concrete"), ("wire", coreWireRefJson wire)]
  | .template wire => object [("kind", "template"), ("wire", templateWireRefJson wire)]
  | .instantiatedTemplate wire path => object [
      ("kind", "instantiatedTemplate"),
      ("wire", templateWireRefJson wire),
      ("path", array (path.map instanceFrameJson))
    ]
  | .familyElement aggregate index => object [
      ("kind", "familyElement"),
      ("aggregate", familyAggregateRefJson aggregate),
      ("index", runtimeRefJson index)
    ]
  | .recurrenceResult recurrence slot => object [
      ("kind", "recurrenceResult"),
      ("instance", recurrenceInstanceRefJson recurrence),
      ("slot", slot)
    ]

private def matrixFactPathJson : Mxx.Certificate.MatrixFactPath → Json
  | .exactExpression carriedSlot =>
      object [("kind", "exactExpression"), ("carriedSlot", carriedSlot)]
  | .affineCoefficient carriedSlot termIndex => object [
      ("kind", "affineCoefficient"),
      ("carriedSlot", carriedSlot),
      ("termIndex", termIndex)
    ]
  | .affineBasis carriedSlot termIndex => object [
      ("kind", "affineBasis"),
      ("carriedSlot", carriedSlot),
      ("termIndex", termIndex)
    ]
  | .familyElement carriedSlot index nested => object [
      ("kind", "familyElement"),
      ("carriedSlot", carriedSlot),
      ("index", runtimeRefJson index),
      ("nested", matrixFactPathJson nested)
    ]

private def boundFactPathJson : Mxx.Certificate.BoundFactPath → Json
  | .affineCoefficientBound carriedSlot termIndex => object [
      ("kind", "affineCoefficientBound"),
      ("carriedSlot", carriedSlot),
      ("termIndex", termIndex)
    ]
  | .affineNoiseBound carriedSlot =>
      object [("kind", "affineNoiseBound"), ("carriedSlot", carriedSlot)]
  | .matrixTotalBound carriedSlot =>
      object [("kind", "matrixTotalBound"), ("carriedSlot", carriedSlot)]
  | .familyElement carriedSlot index nested => object [
      ("kind", "familyElement"),
      ("carriedSlot", carriedSlot),
      ("index", runtimeRefJson index),
      ("nested", boundFactPathJson nested)
    ]

private def boundExprJson : Mxx.Certificate.BoundExpr → Json
  | .constant value => object [("kind", "constant"), ("value", value)]
  | .parameter value => object [("kind", "parameter"), ("value", intExprJson value)]
  | .add left right =>
      object [("kind", "add"), ("left", boundExprJson left), ("right", boundExprJson right)]
  | .multiply left right => object [
      ("kind", "multiply"),
      ("left", boundExprJson left),
      ("right", boundExprJson right)
    ]
  | .maximum left right => object [
      ("kind", "maximum"),
      ("left", boundExprJson left),
      ("right", boundExprJson right)
    ]
  | .absolute value => object [("kind", "absolute"), ("value", intExprJson value)]
  | .floorDivide value positiveDivisor => object [
      ("kind", "floorDivide"),
      ("value", boundExprJson value),
      ("positiveDivisor", positiveDivisor)
    ]
  | .matrixProduct ringDimension innerDimension left right => object [
      ("kind", "matrixProduct"),
      ("ringDimension", intExprJson ringDimension),
      ("innerDimension", intExprJson innerDimension),
      ("left", boundExprJson left),
      ("right", boundExprJson right)
    ]
  | .minimum left right => object [
      ("kind", "minimum"),
      ("left", boundExprJson left),
      ("right", boundExprJson right)
    ]
  | .recurrenceResult recurrence path => object [
      ("kind", "recurrenceResult"),
      ("instance", recurrenceInstanceRefJson recurrence),
      ("path", boundFactPathJson path)
    ]
  | .carriedInput path => object [
      ("kind", "carriedInput"),
      ("path", boundFactPathJson path)
    ]

private def concatPartJson (part : Mxx.Certificate.ConcatPart) : Json := object [
  ("matrixType", matrixTypeJson part.matrixType),
  ("rowOffset", intExprJson part.rowOffset),
  ("columnOffset", intExprJson part.columnOffset)
]

private def concatAxisName : Mxx.Certificate.ConcatAxis → String
  | .rows => "rows"
  | .columns => "columns"
  | .diagonal => "diagonal"

private def concatLayoutJson (layout : Mxx.Certificate.ConcatLayout) : Json := object [
  ("axis", concatAxisName layout.axis),
  ("parts", array (layout.parts.map concatPartJson)),
  ("output", matrixTypeJson layout.output)
]

private def matrixExprJson : Mxx.Certificate.MatrixExpr → Json
  | .wire reference => object [
      ("kind", "wire"),
      ("valueInstanceRef", valueInstanceRefJson reference.value),
      ("matrixType", matrixTypeJson reference.type)
    ]
  | .zero type => object [("kind", "zero"), ("matrixType", matrixTypeJson type)]
  | .identity type => object [("kind", "identity"), ("matrixType", matrixTypeJson type)]
  | .gadget type base => object [
      ("kind", "gadget"),
      ("matrixType", matrixTypeJson type),
      ("base", intExprJson base)
    ]
  | .add left right =>
      object [("kind", "add"), ("left", matrixExprJson left), ("right", matrixExprJson right)]
  | .negate value => object [("kind", "negate"), ("value", matrixExprJson value)]
  | .multiply left right => object [
      ("kind", "multiply"),
      ("left", matrixExprJson left),
      ("right", matrixExprJson right)
    ]
  | .scalarMultiply scalar value => object [
      ("kind", "scalarMultiply"),
      ("scalar", intExprJson scalar),
      ("value", matrixExprJson value)
    ]
  | .rowSlice value start stop => object [
      ("kind", "rowSlice"),
      ("value", matrixExprJson value),
      ("start", intExprJson start),
      ("stop", intExprJson stop)
    ]
  | .rowConcat parts => object [("kind", "rowConcat"), ("parts", array (parts.map matrixExprJson))]
  | .columnSlice value start stop => object [
      ("kind", "columnSlice"),
      ("value", matrixExprJson value),
      ("start", intExprJson start),
      ("stop", intExprJson stop)
    ]
  | .columnConcat parts =>
      object [("kind", "columnConcat"), ("parts", array (parts.map matrixExprJson))]
  | .diagonalConcat parts =>
      object [("kind", "diagonalConcat"), ("parts", array (parts.map matrixExprJson))]
  | .rowCoefficientEmbed layout part value => object [
      ("kind", "rowCoefficientEmbed"),
      ("layout", concatLayoutJson layout),
      ("part", part),
      ("value", matrixExprJson value)
    ]
  | .columnBasisEmbed layout part value => object [
      ("kind", "columnBasisEmbed"),
      ("layout", concatLayoutJson layout),
      ("part", part),
      ("value", matrixExprJson value)
    ]
  | .diagonalCoefficientEmbed layout part value => object [
      ("kind", "diagonalCoefficientEmbed"),
      ("layout", concatLayoutJson layout),
      ("part", part),
      ("value", matrixExprJson value)
    ]
  | .diagonalBasisEmbed layout part value => object [
      ("kind", "diagonalBasisEmbed"),
      ("layout", concatLayoutJson layout),
      ("part", part),
      ("value", matrixExprJson value)
    ]
  | .select _ branches =>
      object [("kind", "select"), ("branches", array (branches.map matrixExprJson))]
  | .loopResult type summary path => object [
      ("kind", "loopResult"),
      ("type", matrixTypeJson type),
      ("instance", recurrenceInstanceRefJson summary),
      ("path", matrixFactPathJson path)
    ]
  | .carriedInput type path => object [
      ("kind", "carriedInput"),
      ("type", matrixTypeJson type),
      ("path", matrixFactPathJson path)
    ]

private def signalModeName : Mxx.Certificate.SignalProductMode → String
  | .ordinaryMatrixProduct => "ordinaryMatrixProduct"
  | .leftPolynomialScalarBroadcast => "leftPolynomialScalarBroadcast"
  | .rightPolynomialScalarBroadcast => "rightPolynomialScalarBroadcast"
  | .swappedRowVectorScalarProduct => "swappedRowVectorScalarProduct"

private def signalTermJson (term : Mxx.Certificate.SignalTerm) : Json := object [
  ("coefficient", object [
    ("expression", matrixExprJson term.coefficient.expression),
    ("normBound", boundExprJson term.coefficient.normBound)
  ]),
  ("basis", matrixExprJson term.basis),
  ("mode", signalModeName term.mode)
]

private def matrixPrimaryName : Mxx.Certificate.MatrixPrimaryForm → String
  | .exact _ => "exact"
  | .affine form => if form.terms.isEmpty then "bounded" else "affine"

private def matrixPrimaryJson : Mxx.Certificate.MatrixPrimaryForm → Json
  | .exact expression => object [("kind", "exact"), ("expression", matrixExprJson expression)]
  | .affine form => object [
      ("kind", if form.terms.isEmpty then "bounded" else "affine"),
      ("terms", array (form.terms.map signalTermJson)),
      ("noiseBound", boundExprJson form.noiseBound)
    ]

private def matrixRelationJson : Mxx.Certificate.MatrixRelation → Json
  | .preimage subject source target trapdoor => object [
      ("kind", "preimage"),
      ("subject", valueInstanceRefJson subject),
      ("source", object [
        ("valueInstanceRef", valueInstanceRefJson source.value),
        ("matrixType", matrixTypeJson source.type)
      ]),
      ("target", object [
        ("valueInstanceRef", valueInstanceRefJson target.value),
        ("matrixType", matrixTypeJson target.type)
      ]),
      ("trapdoor", valueInstanceRefJson trapdoor)
    ]
  | .gadgetDecomposition subject target base digitCount => object [
      ("kind", "gadgetDecomposition"),
      ("subject", valueInstanceRefJson subject),
      ("target", object [
        ("valueInstanceRef", valueInstanceRefJson target.value),
        ("matrixType", matrixTypeJson target.type)
      ]),
      ("base", intExprJson base),
      ("digitCount", intExprJson digitCount)
    ]

private def matrixFactJson (fact : Mxx.Certificate.MatrixFact) : Json := object [
  ("valueInstanceRef", valueInstanceRefJson fact.subject),
  ("primary", matrixPrimaryName fact.primary),
  ("primaryForm", matrixPrimaryJson fact.primary),
  ("relations", array (fact.relations.map matrixRelationJson)),
  ("totalNormBound", boundExprJson fact.totalNormBound)
]

private def valueFactJson : Mxx.Certificate.ValueFact → Json
  | .matrix fact => object [("kind", "matrix"), ("matrixFact", matrixFactJson fact)]
  | .trapdoor fact => object [
      ("kind", "trapdoor"),
      ("privatePort", valueInstanceRefJson fact.privatePort),
      ("publicPort", valueInstanceRefJson fact.publicPort),
      ("publicMatrix", matrixExprJson fact.publicMatrix)
    ]
  | .integer _ => object [("kind", "integer")]
  | .boolean _ => object [("kind", "boolean")]
  | .bytes wire => object [("kind", "bytes"), ("valueInstanceRef", valueInstanceRefJson wire)]
  | .family fact => object [
      ("kind", "family"),
      ("aggregate", familyAggregateRefJson fact.aggregate),
      ("count", intExprJson fact.count)
    ]

private def programName (stage : String) : String :=
  if stage = "encrypt" then "DiamondWeFamily_stage_encrypt"
  else if stage = "decrypt" then "DiamondWeFamily_stage_decrypt"
  else if stage = "$ideal" then "DiamondWeFamily_ideal"
  else if stage.startsWith "$requirement:" then
    s!"DiamondWeFamily_requirement_{stage.drop "$requirement:".length}"
  else if stage = "$comparator" then "DiamondWeFamily_comparator"
  else stage

private def scopeName (scope : Mxx.Certificate.StaticScopeId) : String :=
  match scope.path.reverse with
  | [] => "__root"
  | name :: _ => name

private def wireFactJson (entry : Mxx.Certificate.ScopedWireFact) : Json :=
  let common : List (String × Json) := [
    ("program", programName entry.wire.stage.name),
    ("stage", entry.wire.stage.name),
    ("scope", scopeName entry.wire.scope),
    ("scopePath", array (entry.wire.scope.path.map toJson)),
    ("node", toJson entry.wire.node.value),
    ("port", toJson entry.wire.port),
    ("fact", valueFactJson entry.fact)
  ]
  match entry.fact with
  | .matrix fact => Json.mkObj <| common ++ ([
      ("primary", matrixPrimaryName fact.primary),
      ("valueInstanceRef", valueInstanceRefJson fact.subject),
      ("matrixFact", matrixFactJson fact)
    ] : List (String × Json))
  | .trapdoor fact => Json.mkObj <| common ++ [
      ("valueInstanceRef", valueInstanceRefJson fact.privatePort)
    ]
  | .bytes wire => Json.mkObj <| common ++ [
      ("valueInstanceRef", valueInstanceRefJson wire)
    ]
  | .family fact => Json.mkObj <| common ++ [
      ("valueInstanceRef", object [
        ("kind", "family"), ("aggregate", familyAggregateRefJson fact.aggregate)
      ])
    ]
  | .integer _ | .boolean _ => Json.mkObj common

private def matrixWireFacts (facts : List Mxx.Certificate.ScopedWireFact) : List Json :=
  facts.filterMap fun entry =>
    match entry.fact with
    | .matrix _ => some (wireFactJson entry)
    | _ => none

private def familyJson (entry : Mxx.Certificate.JointFamilyId ×
    Mxx.Certificate.JointFamilyFact) : Json :=
  let (key, fact) := entry
  object [
    ("key", key.name),
    ("id", fact.id.name),
    ("count", intExprJson fact.count),
    ("indexVariable", fact.indexVariable.slot),
    ("outputs", array (fact.outputFamilies.map coreWireRefJson)),
    ("elements", array (fact.elementTuple.toList.map fun template => object [
      ("fact", valueFactJson template.fact)
    ]))
  ]

private def recurrenceJson (transfer : Mxx.Certificate.SymbolicRecurrenceTransfer) : Json :=
  let key := transfer.identity
  let fact := transfer.source
  object [
    ("key", recurrenceInstanceRefJson key),
    ("loopSite", coreNodeRefJson fact.loop.site),
    ("count", intExprJson fact.count),
    ("carriedArity", fact.carriedArity),
    ("initial", array (fact.initial.toList.map fun template => object [
      ("fact", valueFactJson template.fact)
    ])),
    ("bodyInputs", array (fact.bodyInputs.toList.map templateWireRefJson)),
    ("bodyOutputs", array (fact.bodyOutputs.toList.map fun template => object [
      ("fact", valueFactJson template.fact)
    ])),
    ("invariantInputs", array (fact.invariantInputs.map fun input => object [
      ("wire", coreWireRefJson input.wire),
      ("fact", valueFactJson input.template.fact)
    ])),
    ("iterationVariable", fact.iterationVariable.slot)
  ]

private def findMatrixFact (value : Mxx.Certificate.ValueInstanceRef) :
    List Mxx.Certificate.ScopedWireFact → Option Mxx.Certificate.MatrixFact
  | [] => none
  | entry :: tail =>
      match entry.fact with
      | .matrix fact => if fact.subject = value then some fact else findMatrixFact value tail
      | _ => findMatrixFact value tail

private def findWireFact (wire : Mxx.Certificate.CoreWireRef) :
    List Mxx.Certificate.ScopedWireFact → Option Mxx.Certificate.ScopedWireFact
  | [] => none
  | entry :: tail => if entry.wire = wire then some entry else findWireFact wire tail

private def anchorInputFacts
    (analysis : Mxx.Certificate.AnalysisResult)
    (wire : Mxx.Certificate.CoreWireRef) : List Json :=
  let program := DiamondWeFamily_protocol.bundle.workflow.stages.find?
    (fun stage => stage.id = wire.stage.name) |>.map (·.program)
  let scope := program.bind fun program =>
    match wire.scope.path.reverse with
    | [] => some program.root
    | definition :: _ => program.definitions.find? (fun entry => entry.1 = definition) |>.map (·.2)
  let arguments := scope.bind (fun scope => scope.nodes[wire.node.value]?) |>.map (·.arguments)
  match arguments with
  | none => []
  | some arguments => arguments.map fun argument =>
      let inputWire : Mxx.Certificate.CoreWireRef := {
        stage := wire.stage
        scope := wire.scope
        node := ⟨argument.node⟩
        port := argument.port
      }
      object [
        ("wire", coreWireRefJson inputWire),
        ("matrixFact", match findWireFact inputWire analysis.facts with
          | some { fact := .matrix fact, .. } => matrixFactJson fact
          | _ => Json.null)
      ]

private def semanticAnchorJson
    (analysis : Mxx.Certificate.AnalysisResult)
    (binding : Mxx.Certificate.SemanticAnchorBinding) :
    Except Mxx.Certificate.VerifyError Json := do
  let values ← Mxx.Certificate.resolveSemanticAnchor
    DiamondWeFamily_protocol.bundle binding.anchor
  let facts := values.map fun value => object [
    ("valueInstanceRef", valueInstanceRefJson value),
    ("matrixFact", match findMatrixFact value analysis.facts with
      | some fact => matrixFactJson fact
      | none => Json.null)
  ]
  return object [
    ("stage", binding.anchor.stage.name),
    ("label", binding.anchor.label),
    ("values", array facts),
    ("inputs", array (binding.wires.flatMap (anchorInputFacts analysis)))
  ]

private def analysisJson (analysis : Mxx.Certificate.AnalysisResult) :
    Except Mxx.Certificate.VerifyError Json := do
  let anchors ← DiamondWeFamily_protocol.bundle.anchorBindings.mapM (semanticAnchorJson analysis)
  return object [
    ("schema", "mxx-analysis-facts-v1"),
    ("workflowHash", DiamondWeFamily_workflowHash),
    ("wireFacts", array (matrixWireFacts analysis.facts)),
    ("families", array (analysis.families.map familyJson)),
    ("symbolicRecurrences", array (analysis.symbolicRecurrences.map recurrenceJson)),
    ("semanticAnchors", array anchors)
  ]

private def verifyErrorMessage : Mxx.Certificate.VerifyError → String
  | .disabledRule rule => s!"disabledRule {repr rule}"
  | .unsupportedNode stage node => s!"unsupportedNode stage={stage.name} node={node.value}"
  | .unsupportedNodeInScope stage scope node =>
      s!"unsupportedNode stage={stage.name} scope={repr scope.path} node={node.value}"
  | .unsupportedDefinition stage name =>
      s!"unsupportedDefinition stage={stage.name} name={name}"
  | .missingInputFact stage node input =>
      s!"missingInputFact stage={stage.name} node={node.value} " ++
        s!"inputNode={input.node} inputPort={input.port}"
  | .expectedMatrixFact wire => s!"expectedMatrixFact wire={repr wire}"
  | .expectedTrapdoorFact wire => s!"expectedTrapdoorFact wire={repr wire}"
  | .trapdoorPublicMismatch wire expected actual sourcePrimary =>
      s!"trapdoorPublicMismatch wire={repr wire} expected={repr expected} actual={repr actual} " ++
        s!"sourcePrimary={repr sourcePrimary}"
  | .missingAnchorBinding anchor =>
      s!"missingAnchorBinding stage={anchor.stage.name} label={anchor.label}"
  | .invalidAnchorWire anchor wire =>
      s!"invalidAnchorWire stage={anchor.stage.name} label={anchor.label} " ++
        s!"wireNode={wire.node.value} wirePort={wire.port}"
  | .unsupportedOverride anchor =>
      s!"unsupportedOverride stage={anchor.stage.name} label={anchor.label}"
  | .mismatchedMatrixTypes left right =>
      s!"mismatchedMatrixTypes left={(matrixTypeJson left).compress} " ++
        s!"right={(matrixTypeJson right).compress}"
  | .expectedIntegerFact wire => s!"expectedIntegerFact wire={repr wire}"
  | .expectedBooleanFact wire => s!"expectedBooleanFact wire={repr wire}"
  | .missingInputContract name => s!"missingInputContract name={name}"
  | .missingProgramInput stage name => s!"missingProgramInput stage={stage.name} name={name}"
  | .missingArtifactOutput stage name =>
      s!"missingArtifactOutput stage={stage.name} name={name}"
  | .invalidInputCoverage input => s!"invalidInputCoverage input={input.name}"
  | .invalidInputDestination input => s!"invalidInputDestination input={input.name}"
  | .invalidEndpointCoverage endpoint => s!"invalidEndpointCoverage endpoint={repr endpoint}"
  | .invalidEndpointConnection endpoint =>
      s!"invalidEndpointConnection endpoint={repr endpoint}"
  | .diamondEndpoint error => s!"diamondEndpoint {repr error}"
  | .frozenRecurrenceInterface recurrence error =>
      s!"frozenRecurrenceInterface recurrence={repr recurrence} error={repr error}"
  | .bggRecurrencePrefilter error => s!"bggRecurrencePrefilter error={repr error}"
  | .bggCarriedRoleInference error => s!"bggCarriedRoleInference error={repr error}"
  | .bggThreeTraceInterface error => s!"bggThreeTraceInterface error={repr error}"
  | .invalidPreconditionSpec => "invalidPreconditionSpec"
  | .duplicateInputId input => s!"duplicateInputId input={input.name}"
  | .duplicateInputName name => s!"duplicateInputName name={name}"
  | .duplicateInputDestination destination =>
      s!"duplicateInputDestination destination={repr destination}"
  | .unboundProgramInput stage name =>
      s!"unboundProgramInput stage={stage.name} name={name}"
  | .duplicateEndpointSpec endpoint => s!"duplicateEndpointSpec endpoint={repr endpoint}"
  | .invalidComparatorPolarity endpoint =>
      s!"invalidComparatorPolarity endpoint={repr endpoint}"
  | .nonBooleanOutput stage name => s!"nonBooleanOutput stage={stage.name} name={name}"
  | .invalidEndpointAnchorArity endpoint => s!"invalidEndpointAnchorArity {repr endpoint}"
  | .missingOrInvalidOutputTypes stage node =>
      s!"missingOrInvalidOutputTypes stage={stage.name} node={node.value}"
  | .inputContractTypeMismatch input stage name =>
      s!"inputContractTypeMismatch input={input.name} stage={stage.name} name={name}"
  | .duplicateParameter name => s!"duplicateParameter name={name}"
  | .missingParameterDeclaration name => s!"missingParameterDeclaration name={name}"
  | .parameterKindMismatch name => s!"parameterKindMismatch name={name}"
  | .typing .unknownExpressionType => "typing unknownExpressionType"
  | .typing (.incompatibleMatrixProduct left right) =>
      s!"typing incompatibleMatrixProduct left={(matrixTypeJson left).compress} " ++
        s!"right={(matrixTypeJson right).compress}"
  | .exactLeftAffineRightProduct stage node =>
      s!"exactLeftAffineRightProduct stage={stage.name} node={node.value}"
  | .generalAffineProduct stage node =>
      s!"generalAffineProduct stage={stage.name} node={node.value}"
  | .invalidLoopDefinition stage name =>
      s!"invalidLoopDefinition stage={stage.name} name={name}"
  | .invalidLoopArity stage node =>
      s!"invalidLoopArity stage={stage.name} node={node.value}"
  | .invalidLoopArityInScope stage scope node =>
      s!"invalidLoopArity stage={stage.name} scope={reprStr scope.path} node={node.value}"
  | .unsupportedSequentialRecurrence stage node =>
      s!"unsupportedSequentialRecurrence stage={stage.name} node={node.value}"
  | .unsupportedCarriedKind stage node slot =>
      s!"unsupportedCarriedKind stage={stage.name} node={node.value} slot={slot}"
  | .nonUniformNestedRecurrenceInput stage node slot =>
      s!"nonUniformNestedRecurrenceInput stage={stage.name} node={node.value} slot={slot}"
  | .relationBearingCarriedMatrix stage node slot =>
      s!"relationBearingCarriedMatrix stage={stage.name} node={node.value} slot={slot}"
  | .escapedCarriedInput stage node slot =>
      s!"escapedCarriedInput stage={stage.name} node={node.value} slot={slot}"
  | .invalidExpressionReference detail => s!"invalidExpressionReference detail={detail}"
  | .missingFamily joint => s!"missingFamily joint={joint.name}"
  | .invalidFamilySlot joint slot => s!"invalidFamilySlot joint={joint.name} slot={slot}"
  | .scalarControl (.unsupportedNodeKind _) => "scalarControl unsupportedNodeKind"
  | .matrixAffine stage scope node (.typing .unknownExpressionType) =>
      s!"matrixAffine stage={stage.name} scope={repr scope.path} node={node.value} " ++
        "typing=unknownExpressionType"
  | .matrixAffine stage scope node (.typing (.incompatibleMatrixProduct _ _)) =>
      s!"matrixAffine stage={stage.name} scope={repr scope.path} node={node.value} " ++
        "typing=incompatibleMatrixProduct"
  | .matrixAffine stage scope node (.unsupportedScale _) =>
      s!"matrixAffine stage={stage.name} scope={repr scope.path} node={node.value} " ++
        "unsupportedScale"
  | .matrixAffine stage scope node .generalAffineProduct =>
      s!"matrixAffine stage={stage.name} scope={repr scope.path} node={node.value} " ++
        "generalAffineProduct"
  | .matrixAffine stage scope node (.unknownCoefficientType _) =>
      s!"matrixAffine stage={stage.name} scope={repr scope.path} node={node.value} " ++
        "unknownCoefficientType"
  | .matrixAffine stage scope node (.unknownBasisType expression) =>
      s!"matrixAffine stage={stage.name} scope={repr scope.path} node={node.value} " ++
        s!"unknownBasisType={matrixExprShape expression}"
  | .matrixSelect wire .emptyBranches => s!"matrixSelect empty branches at {reprStr wire}"
  | .matrixSelect wire (.mismatchedBranchType _ _) =>
      s!"matrixSelect mismatched branch type at {reprStr wire}"
  | .matrixSelect wire .unsupportedBasis =>
      s!"matrixSelect unsupported basis at {reprStr wire}"
  | .matrixSelect wire .duplicateBasis =>
      s!"matrixSelect duplicate basis at {reprStr wire}"
  | .matrixSelect wire .incompatibleBasisCoefficient =>
      s!"matrixSelect incompatible basis/coefficient at {reprStr wire}"
  | .matrixSelect wire (.invalidConcreteIndex index) =>
      s!"matrixSelect invalid concrete index {index} at {reprStr wire}"
  | .matrixSelect wire .unknownCoefficientType =>
      s!"matrixSelect unknown input coefficient type at {reprStr wire}"
  | .matrixSelect wire (.exactEmbeddingTyping _) =>
      s!"matrixSelect invalid exact embedding at {reprStr wire}"
  | .transform (.unsupportedNodeKind _) => "transform unsupportedNodeKind"
  | .affineNormalize wire .unknownCoefficientType =>
      s!"affineNormalize failed at {reprStr wire}: unknown coefficient type"
  | .affineNormalize wire .incompatibleBasisCoefficient =>
      s!"affineNormalize failed at {reprStr wire}: incompatible basis/coefficient"
  | .affineNormalize wire (.typing .unknownExpressionType) =>
      s!"affineNormalize failed at {reprStr wire}: unknown expression type"
  | .affineNormalize wire (.typing (.incompatibleMatrixProduct _ _)) =>
      s!"affineNormalize failed at {reprStr wire}: incompatible matrix product"
  | .transform .relationBearingInput => "transform relationBearingInput"
  | .transform .affineSignalInput => "transform affineSignalInput"
  | .transform .emptyConcat => "transform emptyConcat"
  | .symbolicEvaluation error => s!"symbolicEvaluation {repr error}"
  | .symbolicRecurrence error => s!"symbolicRecurrence {repr error}"

def run (args : List String) : IO UInt32 := do
  let output ← match args with
    | [path] => pure (System.FilePath.mk path)
    | _ =>
        IO.eprintln "usage: mxx_analysis_facts <output.json>"
        return 2
  let certificate : Mxx.Certificate.SparseCertificate := { overrides := [] }
  let analysis ← match Mxx.Certificate.analyzeProtocol DiamondWeFamily_protocol certificate with
    | .ok analysis => pure analysis
    | .error error =>
        IO.eprintln s!"Diamond analyzer failed: {verifyErrorMessage error}"
        return 1
  if (matrixWireFacts analysis.facts).isEmpty then
    IO.eprintln "Diamond analyzer fact export failed: analysis contains no matrix wire facts"
    return 1
  let payload ← match analysisJson analysis with
    | .ok payload => pure payload
    | .error error =>
        IO.eprintln s!"Diamond analyzer fact export failed: {verifyErrorMessage error}"
        return 1
  if let some parent := output.parent then
    IO.FS.createDirAll parent
  IO.FS.writeFile output payload.pretty
  IO.println <| s!"wrote {(matrixWireFacts analysis.facts).length} matrix wire facts, " ++
    s!"{analysis.families.length} families, and " ++
    s!"{analysis.symbolicRecurrences.length} symbolic recurrences to {output}"
  return 0

end MxxWe.AnalysisFacts

def main (args : List String) : IO UInt32 :=
  MxxWe.AnalysisFacts.run args
