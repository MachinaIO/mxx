import MxxWe.GenericBooleanLayers
import MxxWe.GenericDiamondExecution

namespace MxxWe.Certificate

deriving instance DecidableEq for Mxx.Ir.IntExpr

/-- An exact scope in one emitted executable stage.  This mirrors
`mxx_ir_core::FrozenGraphScopeId`; structural bodies retain their exact parent owner. -/
inductive ScopeRef where
  | root
  | subgraph (canonicalName : String)
  | parallelBody (parent : ScopeRef) (owner : Nat)
  | sequentialBody (parent : ScopeRef) (owner : Nat)
  deriving DecidableEq, Repr

structure CoreNodeRef where
  stage : String
  scope : ScopeRef
  node : Nat
  deriving DecidableEq

structure CoreWireRef where
  node : CoreNodeRef
  port : Nat
  deriving DecidableEq

structure CoreOperandRef where
  node : CoreNodeRef
  operand : Nat
  wire : CoreWireRef
  deriving DecidableEq

inductive CoreNodeParameter where
  | gadgetDecomposeBase
  | gadgetDecomposeDigitCount
  deriving DecidableEq

structure CoreNodeParameterRef where
  node : CoreNodeRef
  parameter : CoreNodeParameter
  deriving DecidableEq

structure StageInputLayout where
  name : String
  node : CoreNodeRef
  deriving DecidableEq

structure StageOutputLayout where
  name : String
  wire : CoreWireRef
  deriving DecidableEq

structure StageInterfaceLayout where
  stage : String
  inputs : List StageInputLayout
  outputs : List StageOutputLayout
  deriving DecidableEq

structure ArtifactProvenance where
  producerStage : String
  producerOutput : StageOutputLayout
  consumerStage : String
  consumerInput : StageInputLayout
  deriving DecidableEq

structure DiamondWorkflowLayout where
  encryption : StageInterfaceLayout
  decryption : StageInterfaceLayout
  artifacts : List ArtifactProvenance

structure OperationRef where
  operation : CoreNodeRef
  inputs : List CoreOperandRef
  outputs : List CoreWireRef
  deriving DecidableEq

structure DynamicFamilyGetRef where
  operation : CoreNodeRef
  family : CoreOperandRef
  index : CoreOperandRef
  output : CoreWireRef
  deriving DecidableEq

inductive CertifiedLoopInputMode where
  | broadcast
  | zip
  | zipOffset (offset : Nat)
  deriving DecidableEq

structure ParallelLoopRef where
  operation : CoreNodeRef
  bodyScope : ScopeRef
  count : Mxx.Ir.IntExpr
  indexSlot : Nat
  bindings : List (String × Mxx.Ir.IntExpr)
  inputModes : List CertifiedLoopInputMode
  arguments : List CoreOperandRef
  bodyInputs : List CoreWireRef
  bodyOutputs : List CoreWireRef
  outputs : List CoreWireRef
  deriving DecidableEq

structure SequentialLoopRef where
  operation : CoreNodeRef
  bodyScope : ScopeRef
  count : Mxx.Ir.IntExpr
  indexSlot : Nat
  bindings : List (String × Mxx.Ir.IntExpr)
  carriedCount : Nat
  arguments : List CoreOperandRef
  bodyInputs : List CoreWireRef
  bodyOutputs : List CoreWireRef
  outputs : List CoreWireRef
  deriving DecidableEq

structure SequentialOperationRef where
  sequentialLoop : SequentialLoopRef
  body : OperationRef
  deriving DecidableEq

structure UnaryNodeRef where
  operation : CoreNodeRef
  input : CoreOperandRef
  output : CoreWireRef
  deriving DecidableEq

structure BinaryNodeRef where
  operation : CoreNodeRef
  left : CoreOperandRef
  right : CoreOperandRef
  output : CoreWireRef
  deriving DecidableEq

structure MatrixBinaryRef where
  operation : CoreNodeRef
  left : CoreOperandRef
  right : CoreOperandRef
  output : CoreWireRef
  deriving DecidableEq

structure ParallelMatrixBinaryRef where
  parallelLoop : CoreNodeRef
  bodyScope : ScopeRef
  leftFamily : CoreOperandRef
  rightFamily : CoreOperandRef
  bodyLeft : CoreWireRef
  bodyRight : CoreWireRef
  operation : MatrixBinaryRef
  bodyOutput : CoreWireRef
  outputFamily : CoreWireRef
  deriving DecidableEq

structure EvaluateIntRef where
  operation : CoreNodeRef
  expression : Mxx.Ir.IntExpr
  evaluated : CoreWireRef
  materialization : Option BinaryNodeRef
  output : CoreWireRef
  deriving DecidableEq

structure ParallelFamilyGetRef where
  parallelLoop : ParallelLoopRef
  indexFamily : CoreOperandRef
  sourceFamily : CoreOperandRef
  bodyIndex : CoreWireRef
  bodySource : CoreWireRef
  get : DynamicFamilyGetRef
  outputFamily : CoreWireRef
  deriving DecidableEq

structure ParallelOperationRef where
  parallelLoop : ParallelLoopRef
  body : OperationRef
  deriving DecidableEq

structure ParallelIndexFormulaRef where
  parallelLoop : ParallelLoopRef
  bodyOutput : CoreWireRef
  deriving DecidableEq

structure InitialStateExpansionRef where
  parallelLoop : ParallelLoopRef
  bodyOutput : CoreWireRef
  deriving DecidableEq

structure WitnessDigitPackingRef where
  parallelLoop : ParallelLoopRef
  bodyOutput : CoreWireRef
  bitScan : SequentialLoopRef
  deriving DecidableEq

structure MessageConstructionLayout where
  toInt : OperationRef
  zero : OperationRef
  one : OperationRef
  select : OperationRef
  deriving DecidableEq

structure BggPublicKeySamplingLayout where
  publicKeysArtifact : ArtifactProvenance
  packedHash : OperationRef
  slices : ParallelOperationRef
  deriving DecidableEq

structure ParallelPackedPublicKeyLayout where
  parallelLoop : ParallelLoopRef
  inRange : OperationRef
  padded : OperationRef
  deriving DecidableEq

structure ParallelCircuitInputPublicKeyLayout where
  parallelLoop : ParallelLoopRef
  selectedInstance : OperationRef
  selectedSource : OperationRef
  deriving DecidableEq

structure ParallelGatherRef where
  parallelLoop : ParallelLoopRef
  indexFamily : CoreOperandRef
  sourceFamilies : List CoreOperandRef
  bodyIndex : CoreWireRef
  bodySources : List CoreWireRef
  gets : List DynamicFamilyGetRef
  outputFamilies : List CoreWireRef
  deriving DecidableEq

structure EncryptionInitialPublicKeysLayout where
  onePublicKey : OperationRef
  zeroPublicKey : OperationRef
  instanceWidth : EvaluateIntRef
  publicIndices : ParallelLoopRef
  publicCandidates : ParallelGatherRef
  packedInputs : ParallelPackedPublicKeyLayout
  circuitInputs : ParallelCircuitInputPublicKeyLayout
  deriving DecidableEq

structure PreimageRef where
  sample : OperationRef
  materialize : OperationRef
  deriving DecidableEq

structure ParallelPreimageRef where
  parallelLoop : ParallelLoopRef
  body : PreimageRef
  deriving DecidableEq

structure TransitionSelectorBitLayout where
  bitExtract : OperationRef
  bitToInt : OperationRef
  bitZero : OperationRef
  bitOne : OperationRef
  bitSelect : OperationRef
  specialProduct : OperationRef
  specialTop : OperationRef
  specialBottom : OperationRef
  special : OperationRef
  stateMatch : OperationRef
  stateMatchToInt : OperationRef
  selector : OperationRef
  deriving DecidableEq

structure TransitionSelectorLayout where
  regular : OperationRef
  kIdentity : OperationRef
  k : OperationRef
  initialSelect : OperationRef
  bitScan : SequentialLoopRef
  bitBody : TransitionSelectorBitLayout
  deriving DecidableEq

structure TransitionTargetRef where
  digitSecret : CoreWireRef
  targetPublic : CoreWireRef
  selector : CoreWireRef
  selectorConstruction : TransitionSelectorLayout
  errorSample : OperationRef
  selectorProduct : OperationRef
  targetSum : OperationRef
  deriving DecidableEq

structure ParallelTransitionTargetRef where
  parallelLoop : ParallelLoopRef
  body : TransitionTargetRef
  deriving DecidableEq

structure DiamondInputPreprocessingLayout where
  initialStateArtifact : ArtifactProvenance
  transitionsArtifact : ArtifactProvenance
  trapdoorSamples : ParallelOperationRef
  secretSample : OperationRef
  messageSelector : OperationRef
  initialErrorSample : OperationRef
  initialPublicProduct : OperationRef
  initialState : OperationRef
  transitionSourceIndices : ParallelIndexFormulaRef
  transitionTargetIndices : ParallelIndexFormulaRef
  digitSecretIndices : ParallelIndexFormulaRef
  digitSecretSamples : ParallelOperationRef
  digitSecrets : ParallelGatherRef
  transitionSources : ParallelGatherRef
  targetPublicMatrices : ParallelGatherRef
  transitionTargets : ParallelTransitionTargetRef
  transitionPreimages : ParallelPreimageRef
  finalIndices : ParallelLoopRef
  finalTrapdoors : ParallelGatherRef
  deriving DecidableEq

structure InputInjectionLayout where
  stateScan : CoreNodeRef
  bodyScope : ScopeRef
  initialStatesExpansion : InitialStateExpansionRef
  initialStates : CoreOperandRef
  packedDigits : CoreOperandRef
  transitionFamily : CoreOperandRef
  finalStates : CoreWireRef
  bodyInitialStates : CoreWireRef
  bodyPackedDigits : CoreWireRef
  bodyTransitionFamily : CoreWireRef
  selectedDigit : DynamicFamilyGetRef
  sourceIndices : ParallelIndexFormulaRef
  sourceStates : ParallelFamilyGetRef
  transitionIndices : ParallelIndexFormulaRef
  selectedTransitions : ParallelFamilyGetRef
  bodyFinalStates : CoreWireRef
  stateProduct : ParallelMatrixBinaryRef

structure OneTargetLayout where
  gadget : OperationRef
  difference : OperationRef
  zeroRow : OperationRef
  target : OperationRef
  deriving DecidableEq

structure StaticTrapdoorLayout where
  publicOperation : OperationRef
  secret : OperationRef
  deriving DecidableEq

structure ParallelWitnessTargetLayout where
  parallelLoop : ParallelLoopRef
  negatedGadget : OperationRef
  target : OperationRef
  deriving DecidableEq

structure KTargetLayout where
  publicKeyHash : OperationRef
  firstColumn : OperationRef
  halfModulus : OperationRef
  target : OperationRef
  deriving DecidableEq

structure DecoderTargetLayout where
  publicKeyDifference : OperationRef
  projectedDifference : OperationRef
  publicKeySum : OperationRef
  zero : OperationRef
  target : OperationRef
  deriving DecidableEq

structure DiamondArtifactPreprocessingLayout where
  onePreimageArtifact : ArtifactProvenance
  witnessPreimagesArtifact : ArtifactProvenance
  kPreimageArtifact : ArtifactProvenance
  rDecomposedArtifact : ArtifactProvenance
  decoderPreimageArtifact : ArtifactProvenance
  projectionTrapdoor : StaticTrapdoorLayout
  oneTarget : OneTargetLayout
  onePreimage : PreimageRef
  witnessIndices : ParallelLoopRef
  witnessTrapdoors : ParallelGatherRef
  witnessPublicKeys : ParallelGatherRef
  witnessTargets : ParallelWitnessTargetLayout
  witnessPreimages : ParallelPreimageRef
  kTarget : KTargetLayout
  kPreimage : PreimageRef
  rHash : OperationRef
  rSlice : OperationRef
  rDecomposition : OperationRef
  rMaterialization : OperationRef
  rReshape : OperationRef
  decoderTarget : DecoderTargetLayout
  decoderPreimage : PreimageRef
  deriving DecidableEq

structure EncodingComponentOperationsLayout where
  vectors : ParallelOperationRef
  publicKeys : ParallelOperationRef
  plaintexts : ParallelOperationRef
  deriving DecidableEq

structure DecryptionInitialEncodingsLayout where
  initialStateArtifact : ArtifactProvenance
  onePreimageArtifact : ArtifactProvenance
  witnessPreimagesArtifact : ArtifactProvenance
  publicKeysArtifact : ArtifactProvenance
  witnessIndices : ParallelLoopRef
  witnessBits : ParallelGatherRef
  witnessDigits : WitnessDigitPackingRef
  initialProjectionState : OperationRef
  onePublicKey : OperationRef
  onePlaintext : OperationRef
  zeroEncoding : List OperationRef
  witnessStateIndices : ParallelLoopRef
  witnessStates : ParallelGatherRef
  witnessVectors : ParallelMatrixBinaryRef
  witnessPublicIndices : ParallelLoopRef
  witnessPublicKeys : ParallelGatherRef
  witnessPlaintextConstants : List ParallelLoopRef
  witnessPlaintexts : ParallelOperationRef
  instanceWidth : EvaluateIntRef
  packedIndices : ParallelLoopRef
  packedVectors : ParallelGatherRef
  packedPublicKeys : ParallelGatherRef
  packedPlaintexts : ParallelGatherRef
  activeWitness : ParallelLoopRef
  activeWitnessZeroes : List ParallelLoopRef
  activeWitnessSelection : EncodingComponentOperationsLayout
  instanceConstants : List (List ParallelLoopRef)
  selectedInstance : EncodingComponentOperationsLayout
  activeInstance : ParallelLoopRef
  circuitInputs : EncodingComponentOperationsLayout
  deriving DecidableEq

structure LayerScalarMetadataRef where
  sourceInputName : String
  rootInput : CoreWireRef
  sequentialOperand : CoreOperandRef
  bodySource : CoreWireRef
  layerIndex : EvaluateIntRef
  selected : DynamicFamilyGetRef
  deriving DecidableEq

structure LayerFamilyMetadataRef where
  sourceInputName : String
  rootInput : CoreWireRef
  sequentialOperand : CoreOperandRef
  bodySource : CoreWireRef
  flattenedIndices : ParallelLoopRef
  flattenedIndex : EvaluateIntRef
  gathered : ParallelFamilyGetRef
  deriving DecidableEq

structure BooleanLayerMetadataLayout where
  activeGateCount : LayerScalarMetadataRef
  opcode : LayerFamilyMetadataRef
  leftSource : LayerFamilyMetadataRef
  rightSource : LayerFamilyMetadataRef
  deriving DecidableEq

structure PublicKeyBooleanLoopLayout where
  layerScan : CoreNodeRef
  bodyScope : ScopeRef
  initialPublicKeys : CoreOperandRef
  activeGateCounts : CoreOperandRef
  gateKinds : CoreOperandRef
  leftSources : CoreOperandRef
  rightSources : CoreOperandRef
  onePublicKey : CoreOperandRef
  finalPublicKeys : CoreWireRef
  bodyInitialPublicKeys : CoreWireRef
  bodyActiveGateCounts : CoreWireRef
  bodyGateKinds : CoreWireRef
  bodyLeftSources : CoreWireRef
  bodyRightSources : CoreWireRef
  bodyOnePublicKey : CoreWireRef
  bodyFinalPublicKeys : CoreWireRef
  metadata : BooleanLayerMetadataLayout
  selectedOutput : DynamicFamilyGetRef
  deriving DecidableEq

structure EncodingBooleanLoopLayout where
  layerScan : CoreNodeRef
  bodyScope : ScopeRef
  initialVectors : CoreOperandRef
  initialPublicKeys : CoreOperandRef
  initialPlaintexts : CoreOperandRef
  activeGateCounts : CoreOperandRef
  gateKinds : CoreOperandRef
  leftSources : CoreOperandRef
  rightSources : CoreOperandRef
  oneVector : CoreOperandRef
  onePublicKey : CoreOperandRef
  onePlaintext : CoreOperandRef
  finalVectors : CoreWireRef
  finalPublicKeys : CoreWireRef
  finalPlaintexts : CoreWireRef
  bodyInitialVectors : CoreWireRef
  bodyInitialPublicKeys : CoreWireRef
  bodyInitialPlaintexts : CoreWireRef
  bodyActiveGateCounts : CoreWireRef
  bodyGateKinds : CoreWireRef
  bodyLeftSources : CoreWireRef
  bodyRightSources : CoreWireRef
  bodyOneVector : CoreWireRef
  bodyOnePublicKey : CoreWireRef
  bodyOnePlaintext : CoreWireRef
  bodyFinalVectors : CoreWireRef
  bodyFinalPublicKeys : CoreWireRef
  bodyFinalPlaintexts : CoreWireRef
  metadata : BooleanLayerMetadataLayout
  selectedVector : DynamicFamilyGetRef
  deriving DecidableEq

structure LocalGadgetDecompositionRef where
  decompositionNode : CoreNodeRef
  rightPublicKey : CoreOperandRef
  base : CoreNodeParameterRef
  digitCount : CoreNodeParameterRef
  decomposition : CoreWireRef
  materialized : CoreWireRef
  deriving DecidableEq

structure EncryptPublicKeyRhsDecomposition where
  rightSelection : ParallelFamilyGetRef
  enclosingParallelLoop : CoreNodeRef
  bodyScope : ScopeRef
  rightPublicKeyFamily : CoreOperandRef
  bodyRightPublicKey : CoreWireRef
  localDecomposition : LocalGadgetDecompositionRef
  multiplicationConsumer : CoreOperandRef
  deriving DecidableEq

structure ParallelDecompositionConsumer where
  consumerLoop : CoreNodeRef
  decompositionFamily : CoreOperandRef
  bodyScope : ScopeRef
  bodyDecomposition : CoreWireRef
  multiplicationConsumer : CoreOperandRef
  deriving DecidableEq

structure DecryptEncodingRhsDecomposition where
  rightSelection : ParallelFamilyGetRef
  decompositionLoop : CoreNodeRef
  bodyScope : ScopeRef
  rightPublicKeyFamily : CoreOperandRef
  bodyRightPublicKey : CoreWireRef
  localDecomposition : LocalGadgetDecompositionRef
  bodyOutput : CoreWireRef
  decompositionFamily : CoreWireRef
  publicKeyConsumer : ParallelDecompositionConsumer
  vectorConsumer : ParallelDecompositionConsumer
  deriving DecidableEq

structure SixWaySelectRef where
  operation : CoreNodeRef
  selector : CoreOperandRef
  branches : Fin 6 → CoreOperandRef
  output : CoreWireRef

structure TwoWaySelectRef where
  operation : CoreNodeRef
  selector : CoreOperandRef
  branches : Fin 2 → CoreOperandRef
  output : CoreWireRef

structure ParallelSixWaySelectRef where
  parallelLoop : CoreNodeRef
  bodyScope : ScopeRef
  selectorFamily : CoreOperandRef
  branchFamilies : Fin 6 → CoreOperandRef
  bodySelector : CoreWireRef
  bodyBranches : Fin 6 → CoreWireRef
  select : SixWaySelectRef
  bodyOutput : CoreWireRef
  outputFamily : CoreWireRef

structure ParallelTwoWaySelectRef where
  parallelLoop : CoreNodeRef
  bodyScope : ScopeRef
  selectorFamily : CoreOperandRef
  branchFamilies : Fin 2 → CoreOperandRef
  bodySelector : CoreWireRef
  bodyBranches : Fin 2 → CoreWireRef
  select : TwoWaySelectRef
  bodyOutput : CoreWireRef
  outputFamily : CoreWireRef

structure LocalBooleanGateLayout where
  bodyScope : ScopeRef
  parentLoop : ParallelLoopRef
  opcodeFamily : CoreOperandRef
  leftFamily : CoreOperandRef
  rightFamily : CoreOperandRef
  onePublicKey : CoreOperandRef
  activeGateCount : CoreOperandRef
  leftSelection : ParallelFamilyGetRef
  bodyOpcode : CoreWireRef
  bodyLeft : CoreWireRef
  bodyRight : CoreWireRef
  bodyOnePublicKey : CoreWireRef
  bodyActiveGateCount : CoreWireRef
  zero : MatrixBinaryRef
  one : CoreWireRef
  copy : CoreWireRef
  not : MatrixBinaryRef
  product : MatrixBinaryRef
  sum : MatrixBinaryRef
  twoProduct : MatrixBinaryRef
  xor : MatrixBinaryRef
  candidateSelect : SixWaySelectRef
  activeSelect : TwoWaySelectRef

inductive FamilyProductRef where
  | direct (operation : ParallelMatrixBinaryRef)
  | encodingVector
      (leftTimesRightDecomposition : ParallelMatrixBinaryRef)
      (rightTimesLeftPlaintext : ParallelMatrixBinaryRef)
      (sum : ParallelMatrixBinaryRef)

def FamilyProductRef.outputFamily : FamilyProductRef → CoreWireRef
  | .direct operation => operation.outputFamily
  | .encodingVector _ _ sum => sum.outputFamily

structure FamilyBooleanGateLayout where
  stateInput : CoreWireRef
  stateOutput : CoreWireRef
  leftSelection : ParallelFamilyGetRef
  rightSelection : ParallelFamilyGetRef
  opcodeFamily : CoreWireRef
  activeFamily : CoreWireRef
  zero : ParallelMatrixBinaryRef
  oneRepetition : ParallelLoopRef
  oneFamily : CoreWireRef
  copyFamily : CoreWireRef
  not : ParallelMatrixBinaryRef
  product : FamilyProductRef
  sum : ParallelMatrixBinaryRef
  twoProduct : ParallelMatrixBinaryRef
  xor : ParallelMatrixBinaryRef
  candidateSelect : ParallelSixWaySelectRef
  activeMask : ParallelLoopRef
  activeSelect : ParallelTwoWaySelectRef

structure BooleanLayersLayout where
  publicKeysArtifact : ArtifactProvenance
  encryption : PublicKeyBooleanLoopLayout
  decryption : EncodingBooleanLoopLayout
  encryptPublicKeyRhsDecomposition : EncryptPublicKeyRhsDecomposition
  decryptEncodingRhsDecomposition : DecryptEncodingRhsDecomposition
  encryptionGate : LocalBooleanGateLayout
  decryptionVectors : FamilyBooleanGateLayout
  decryptionPublicKeys : FamilyBooleanGateLayout
  decryptionPlaintexts : FamilyBooleanGateLayout

structure DecoderLayout where
  oneVector : MatrixBinaryRef
  kVector : MatrixBinaryRef
  decoderVector : MatrixBinaryRef
  onePreimage : CoreWireRef
  kPreimage : CoreWireRef
  decoderPreimage : CoreWireRef
  rDecomposed : CoreWireRef
  selectedCircuitVector : CoreWireRef
  oneMinusCircuit : MatrixBinaryRef
  projectedDifference : MatrixBinaryRef
  kPlusProjection : MatrixBinaryRef
  residual : MatrixBinaryRef
  extractCoefficient : UnaryNodeRef
  threshold : EvaluateIntRef
  lowerCompare : BinaryNodeRef
  upperScale : BinaryNodeRef
  upperCompare : BinaryNodeRef
  lowerToInt : UnaryNodeRef
  upperToInt : UnaryNodeRef
  comparisonSum : BinaryNodeRef
  equalsTwo : BinaryNodeRef
  decoded : CoreWireRef

structure DiamondCertificate where
  workflow : DiamondWorkflowLayout
  message : MessageConstructionLayout
  inputPreprocessing : DiamondInputPreprocessingLayout
  publicKeySampling : BggPublicKeySamplingLayout
  encryptionInitialPublicKeys : EncryptionInitialPublicKeysLayout
  artifactPreprocessing : DiamondArtifactPreprocessingLayout
  inputInjection : InputInjectionLayout
  decryptionInitialEncodings : DecryptionInitialEncodingsLayout
  booleanLayers : BooleanLayersLayout
  decoder : DecoderLayout

end MxxWe.Certificate
