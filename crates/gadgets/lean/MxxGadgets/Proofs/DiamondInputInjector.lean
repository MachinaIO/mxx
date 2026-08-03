import MxxGadgets.Generated.DiamondInputInjector.Statement
import Mxx.Toolkit.Norms
import Mathlib.Tactic

open MxxGadgets.Generated.DiamondInputInjector
open Mxx

def dParams (rows columns bound : Nat) : SamplerParams := {
  maxCoefficientBound := bound
  modulus := 65537
  ringDimension := 1
  rows
  columns
}

def dTyped (rows columns bound : Nat) (matrix : Matrix) : Matrix :=
  matrix.withSamplerParams (dParams rows columns bound)

def dNormalize (matrix : Matrix) : Matrix :=
  { matrix with coefficients := matrix.coefficients.map (reduceCoefficient matrix.modulus) }

def dRuntimeNormalize (matrix : Matrix) : Matrix :=
  { matrix with coefficients :=
      matrix.coefficients.map fun coefficient => reduceCoefficient matrix.modulus (1 * coefficient) }

def dMessageMatrix (message : Bool) : Matrix :=
  { coefficients := [if message then 1 else 0], modulus := 65537,
    ringDimension := 1, rows := 1, columns := 1 }

def dInitial (secret b error : Matrix) (message : Bool) : Matrix :=
  let vector := matrixConcatColumns [secret, dMessageMatrix message]
  let product := matrixMul vector b
  let coefficients := List.map (reduceCoefficient product.modulus)
    (Ir.addCoefficients product.coefficients error.coefficients)
  { product with coefficients }

def dIdentity : Matrix :=
  { coefficients := [1], modulus := 65537, ringDimension := 1, rows := 1, columns := 1 }

def dRuntimeIdentity : Matrix :=
  ({ coefficients := [1] } : Matrix).withSamplerParams (dParams 1 1 0)

def dTransitionTarget (selector b error : Matrix) : Matrix :=
  let diagonal := matrixConcatDiagonal [selector, dIdentity]
  let product := matrixMul diagonal b
  let coefficients := List.map (reduceCoefficient product.modulus)
    (Ir.addCoefficients product.coefficients error.coefficients)
  { product with coefficients }

def dRuntimeTransitionTarget (selector b error : Matrix) : Matrix :=
  let diagonal := matrixConcatDiagonal [selector, dRuntimeIdentity]
  let product := matrixMul diagonal b
  let coefficients := List.map (reduceCoefficient product.modulus)
    (Ir.addCoefficients product.coefficients error.coefficients)
  { product with coefficients }

def dZero12 : Matrix :=
  { coefficients := [0, 0], modulus := 65537, ringDimension := 1, rows := 1, columns := 2 }

def dZero11 : Matrix :=
  { coefficients := [0], modulus := 65537, ringDimension := 1, rows := 1, columns := 1 }

def dAuxiliaryTarget (left right b error : Matrix) : Matrix :=
  let top := matrixConcatColumns [left, matrixMul left right]
  let target := matrixConcatRows [top, dZero12]
  let product := matrixMul target b
  let coefficients := List.map (reduceCoefficient product.modulus)
    (Ir.addCoefficients product.coefficients error.coefficients)
  { product with coefficients }

@[simp] theorem dRuntimeNormalize_eq (matrix : Matrix) :
    dRuntimeNormalize matrix = dNormalize matrix := by
  simp [dRuntimeNormalize, dNormalize]

@[simp] theorem dRuntimeIdentity_eq : dRuntimeIdentity = dIdentity := by
  rfl

@[simp] theorem dRuntimeTransitionTarget_eq (selector b error : Matrix) :
    dRuntimeTransitionTarget selector b error = dTransitionTarget selector b error := by
  simp [dRuntimeTransitionTarget, dTransitionTarget]

def dProjectionTarget : Matrix :=
  { coefficients := [0, 32768], modulus := 65537, ringDimension := 1, rows := 2, columns := 1 }

def dOutput (initial transition projection : Matrix) : Ir.Environment :=
  let value := matrixMul (matrixMul initial transition) projection
  let result := match value.coefficients.head? with
    | some coefficient => .boolean (Ir.thresholdDecodeBool 65537 2 coefficient)
    | none => .invalid "missing output result"
  [("result", result)]

theorem dThresholdOutput (coefficients : List Int) :
    (Ir.lookupWire { node := 15, port := 0 }
      (List.map (fun x => (⟨15, x.2⟩, x.1))
        (List.take 1
          (List.map (fun coefficient =>
            Ir.Value.boolean (Ir.thresholdDecodeBool 65537 2 coefficient)) coefficients)).zipIdx)).getD
        (.invalid "missing output result") =
      match coefficients.head? with
      | some coefficient => .boolean (Ir.thresholdDecodeBool 65537 2 coefficient)
      | none => .invalid "missing output result" := by
  cases coefficients <;> simp [Ir.lookupWire]

theorem dLookupWireDifferentNode
    (source target port start : Nat) (different : source ≠ target)
    (values : List Ir.Value) :
    Ir.lookupWire { node := target, port := port }
        (List.map (fun x => (⟨source, x.2⟩, x.1)) (values.zipIdx start)) = none := by
  induction values generalizing start with
  | nil => simp [Ir.lookupWire]
  | cons head tail inductionHypothesis =>
      simp [List.zipIdx_cons, Ir.lookupWire, different, inductionHypothesis]

theorem dLookupWireAppend
    (wire : Ir.WireRef) (left right : Ir.WireEnvironment) :
    Ir.lookupWire wire (left ++ right) =
      match Ir.lookupWire wire left with
      | some value => some value
      | none => Ir.lookupWire wire right := by
  induction left with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      simp only [List.cons_append]
      rcases head with ⟨candidate, value⟩
      simp only [Ir.lookupWire]
      split <;> simp_all

set_option maxRecDepth 10000
set_option maxHeartbeats 1000000

theorem dPreprocessSelectorChild
    (samplers : MxxSamplerFamily) (params : Ir.ParamEnvironment) :
    Ir.childRunnerWithFuel samplers DiamondInputInjector_stage_preprocess 3
        "parallel:__root:17" params [] =
      (Ir.uniformMatrixSupport (dParams 1 1 0) (-1) 1).map
        (fun selector => [.matrix selector]) := by
  simp [Ir.childRunnerWithFuel, DiamondInputInjector_stage_preprocess,
    Ir.lookupDefinition, Ir.denoteScopeWithFuel_succ, Ir.evaluateNodes,
    Ir.evaluateNode, Ir.lookupWire, Ir.collectOutputs,
    Ir.bindOutputs, Ir.IntExpr.evaluate, Ir.MatrixTypeExpr.evaluate,
    dParams]

theorem dPreprocessTransitionChild
    (samplers : MxxSamplerFamily) (params : Ir.ParamEnvironment)
    (b0 selector b1 : Mxx.Matrix) :
    Ir.childRunnerWithFuel samplers DiamondInputInjector_stage_preprocess 3
        "parallel:__root:18" params
        [.matrix selector, .matrix b0, .trapdoor b0, .matrix b1] =
      (samplers.gaussianSample (dParams 2 6 1)).flatMap fun errorRaw =>
        (samplers.samplePreimage (dParams 6 6 2) b0
          (dRuntimeTransitionTarget selector b1 (dTyped 2 6 1 errorRaw))).map fun preimageRaw =>
            [.matrix (dRuntimeNormalize (dTyped 6 6 2 preimageRaw))] := by
  simp (config := { maxSteps := 10000000 }) [Ir.childRunnerWithFuel,
    DiamondInputInjector_stage_preprocess,
    Ir.lookupDefinition, Ir.denoteScopeWithFuel_succ, Ir.evaluateNodes,
    Ir.evaluateNode, Ir.arguments, Ir.lookupWire, Ir.collectOutputs,
    Ir.bindOutputs, Ir.IntExpr.evaluate, Ir.MatrixTypeExpr.evaluate,
    Ir.lookupEnvironment, dParams, dTyped, dRuntimeNormalize, dRuntimeIdentity,
    dRuntimeTransitionTarget, List.map_flatMap, List.flatMap_map, List.map_map,
    List.flatMap_assoc, Function.comp_def]
  apply List.flatMap_congr
  intro errorRaw _
  change List.flatMap _
      (samplers.samplePreimage (dParams 6 6 2) b0
        (dRuntimeTransitionTarget selector b1 (dTyped 2 6 1 errorRaw))) =
    List.map _
      (samplers.samplePreimage (dParams 6 6 2) b0
        (dRuntimeTransitionTarget selector b1 (dTyped 2 6 1 errorRaw)))
  exact List.flatMap_pure_eq_map _ _

theorem dEvaluateMultiplyChild
    (samplers : MxxSamplerFamily) (params : Ir.ParamEnvironment)
    (left right : Mxx.Matrix) :
    Ir.childRunnerWithFuel samplers DiamondInputInjector_stage_evaluate 1
        "parallel:__root:11" params [.matrix left, .matrix right] =
      [[.matrix (Mxx.matrixMul left right)]] := by
  simp [Ir.childRunnerWithFuel, DiamondInputInjector_stage_evaluate,
    Ir.lookupDefinition, Ir.denoteScopeWithFuel_succ, Ir.evaluateNodes,
    Ir.evaluateNode, Ir.arguments, Ir.lookupWire, Ir.collectOutputs,
    Ir.bindOutputs]
  simp [Ir.lookupWire, Ir.lookupEnvironment]

def dMultiplyValue (left : Mxx.Matrix) : Ir.Value → Ir.Value
  | .matrix right => .matrix (Mxx.matrixMul left right)
  | _ => .invalid "matrix multiplication argument mismatch"

theorem dEvaluateMultiplyAnyChild
    (samplers : MxxSamplerFamily) (params : Ir.ParamEnvironment)
    (left : Mxx.Matrix) (right : Ir.Value) :
    Ir.childRunnerWithFuel samplers DiamondInputInjector_stage_evaluate 1
        "parallel:__root:11" params [.matrix left, right] =
      [[dMultiplyValue left right]] := by
  cases right <;>
    simp [dMultiplyValue, Ir.childRunnerWithFuel, DiamondInputInjector_stage_evaluate,
      Ir.lookupDefinition, Ir.denoteScopeWithFuel_succ, Ir.evaluateNodes,
      Ir.evaluateNode, Ir.arguments, Ir.lookupWire, Ir.lookupEnvironment,
      Ir.collectOutputs, Ir.bindOutputs]

theorem dPreprocessAuxiliaryChild
    (samplers : MxxSamplerFamily) (params : Ir.ParamEnvironment)
    (left right b0 b1 : Mxx.Matrix) :
    Ir.childRunnerWithFuel samplers DiamondInputInjector_stage_preprocess 3
        "parallel:__root:24" params
        [.matrix left, .matrix right, .matrix b0, .trapdoor b0, .matrix b1] =
      (samplers.gaussianSample (dParams 2 6 1)).flatMap fun errorRaw =>
        (samplers.samplePreimage (dParams 6 6 2) b0
          (dAuxiliaryTarget left right b1 (dTyped 2 6 1 errorRaw))).map fun preimageRaw =>
            [.matrix (dRuntimeNormalize (dTyped 6 6 2 preimageRaw))] := by
  simp (config := { maxSteps := 10000000 }) [Ir.childRunnerWithFuel,
    DiamondInputInjector_stage_preprocess, Ir.lookupDefinition,
    Ir.denoteScopeWithFuel_succ, Ir.evaluateNodes, Ir.evaluateNode,
    Ir.arguments, Ir.lookupWire, Ir.collectOutputs, Ir.bindOutputs,
    Ir.IntExpr.evaluate, Ir.MatrixTypeExpr.evaluate, Ir.lookupEnvironment,
    dParams, dTyped, dRuntimeNormalize, dAuxiliaryTarget, dZero12,
    List.map_flatMap, List.flatMap_map, List.map_map, List.flatMap_assoc,
    Function.comp_def]
  apply List.flatMap_congr
  intro errorRaw _
  change List.flatMap _
      (samplers.samplePreimage (dParams 6 6 2) b0
        (dAuxiliaryTarget left right b1 (dTyped 2 6 1 errorRaw))) =
    List.map _
      (samplers.samplePreimage (dParams 6 6 2) b0
        (dAuxiliaryTarget left right b1 (dTyped 2 6 1 errorRaw)))
  exact List.flatMap_pure_eq_map _ _

theorem dPreprocessSelectorLoop
    (samplers : MxxSamplerFamily) (params : Ir.ParamEnvironment) :
    Ir.evaluateParallelIterations
        (Ir.childRunnerWithFuel samplers DiamondInputInjector_stage_preprocess 3)
        "parallel:__root:17" params 0 [] [] [] 1 (List.range 2) [[[]]] =
      (Ir.uniformMatrixSupport (dParams 1 1 0) (-1) 1).flatMap fun selector0 =>
        (Ir.uniformMatrixSupport (dParams 1 1 0) (-1) 1).map fun selector1 =>
          [[.matrix selector0, .matrix selector1]] := by
  rw [show List.range 2 = [0, 1] by decide]
  simp [Ir.evaluateParallelIterations, Ir.evaluateBindings, Ir.appendPortValues,
    dPreprocessSelectorChild, Function.comp_def, List.flatMap_map]

theorem dPreprocessTransitionLoop
    (samplers : MxxSamplerFamily) (params : Ir.ParamEnvironment)
    (b0 selector0 selector1 b1 : Mxx.Matrix) :
    Ir.evaluateParallelIterations
        (Ir.childRunnerWithFuel samplers DiamondInputInjector_stage_preprocess 3)
        "parallel:__root:18" params 0 []
        [.zip, .broadcast, .broadcast, .broadcast]
        [.family [.matrix selector0, .matrix selector1], .matrix b0, .trapdoor b0, .matrix b1]
        1 (List.range 2) [[[]]] =
      (samplers.gaussianSample (dParams 2 6 1)).flatMap fun error0 =>
        (samplers.samplePreimage (dParams 6 6 2) b0
          (dRuntimeTransitionTarget selector0 b1 (dTyped 2 6 1 error0))).flatMap fun transition0 =>
          (samplers.gaussianSample (dParams 2 6 1)).flatMap fun error1 =>
            (samplers.samplePreimage (dParams 6 6 2) b0
              (dRuntimeTransitionTarget selector1 b1 (dTyped 2 6 1 error1))).map fun transition1 =>
                [[.matrix (dRuntimeNormalize (dTyped 6 6 2 transition0)),
                  .matrix (dRuntimeNormalize (dTyped 6 6 2 transition1))]] := by
  rw [show List.range 2 = [0, 1] by decide]
  simp [Ir.evaluateParallelIterations, Ir.evaluateBindings, Ir.loopArgument,
    Ir.appendPortValues, dPreprocessTransitionChild, Function.comp_def,
    List.flatMap_assoc, List.flatMap_map, List.map_flatMap, List.map_map]

theorem dEvaluateMultiplyLoop
    (samplers : MxxSamplerFamily) (params : Ir.ParamEnvironment)
    (left0 left1 right0 : Mxx.Matrix) (right1 : Ir.Value) :
    Ir.evaluateParallelIterations
        (Ir.childRunnerWithFuel samplers DiamondInputInjector_stage_evaluate 1)
        "parallel:__root:11" params 0 [] [.zip, .zip]
        [.family [.matrix left0, .matrix left1], .family [.matrix right0, right1]]
        1 (List.range 2) [[[]]] =
      [[[.matrix (Mxx.matrixMul left0 right0), dMultiplyValue left1 right1]]] := by
  rw [show List.range 2 = [0, 1] by decide]
  simp [Ir.evaluateParallelIterations, Ir.evaluateBindings, Ir.loopArgument,
    Ir.appendPortValues, dEvaluateMultiplyAnyChild, dMultiplyValue]

theorem dEvaluateSelectedLoop
    (samplers : MxxSamplerFamily) (digit initial transition0 transition1 : Mxx.Matrix)
    (unused0 unused1 : Ir.Value)
    (digitCase : digit.coefficients.getD 0 0 = 0 ∨ digit.coefficients.getD 0 0 = 1) :
    Ir.evaluateParallelIterations
        (Ir.childRunnerWithFuel samplers DiamondInputInjector_stage_evaluate 1)
        "parallel:__root:11" [] 0 [] [.zip, .zip]
        [.family [.matrix initial, .matrix initial],
          .family [
            (getElem? ([Ir.Value.matrix transition0, Ir.Value.matrix transition1] : List Ir.Value)
              (digit.coefficients.getD 0 0).toNat).getD
                (.invalid "Select index out of range"),
            (getElem? ([unused0, unused1] : List Ir.Value)
              (digit.coefficients.getD 0 0).toNat).getD
              (.invalid "Select index out of range")]]
        1 (List.range 2) [[[]]] =
      [[[.matrix (Mxx.matrixMul initial
          (if digit.coefficients.getD 0 0 = 0 then transition0 else transition1)),
        dMultiplyValue initial
          (if digit.coefficients.getD 0 0 = 0 then unused0 else unused1)]]] := by
  rcases digitCase with digitZero | digitOne
  all_goals try simp only [List.getD_eq_getElem?_getD] at digitZero
  all_goals try simp only [List.getD_eq_getElem?_getD] at digitOne
  all_goals simp only [List.getD_eq_getElem?_getD]
  case inl =>
    simp only [digitZero]
    exact dEvaluateMultiplyLoop samplers [] initial initial transition0 unused0
  case inr =>
    simp only [digitOne]
    exact dEvaluateMultiplyLoop samplers [] initial initial transition1 unused1

theorem dPreprocessAuxiliaryLoop
    (samplers : MxxSamplerFamily) (params : Ir.ParamEnvironment)
    (selector0 selector1 b0 b1 : Mxx.Matrix) :
    Ir.evaluateParallelIterations
        (Ir.childRunnerWithFuel samplers DiamondInputInjector_stage_preprocess 3)
        "parallel:__root:24" params 0 []
        [.zip, .zip, .broadcast, .broadcast, .broadcast]
        [.family [.matrix selector0, .matrix selector1],
          .family [.matrix dZero11, .matrix dIdentity], .matrix b0, .trapdoor b0, .matrix b1]
        1 (List.range 2) [[[]]] =
      (samplers.gaussianSample (dParams 2 6 1)).flatMap fun error0 =>
        (samplers.samplePreimage (dParams 6 6 2) b0
          (dAuxiliaryTarget selector0 dZero11 b1 (dTyped 2 6 1 error0))).flatMap
          fun auxiliary0 =>
            (samplers.gaussianSample (dParams 2 6 1)).flatMap fun error1 =>
              (samplers.samplePreimage (dParams 6 6 2) b0
                (dAuxiliaryTarget selector1 dIdentity b1 (dTyped 2 6 1 error1))).map
                fun auxiliary1 =>
                  [[.matrix (dRuntimeNormalize (dTyped 6 6 2 auxiliary0)),
                    .matrix (dRuntimeNormalize (dTyped 6 6 2 auxiliary1))]] := by
  rw [show List.range 2 = [0, 1] by decide]
  simp [Ir.evaluateParallelIterations, Ir.evaluateBindings, Ir.loopArgument,
    Ir.appendPortValues, dPreprocessAuxiliaryChild, List.flatMap_assoc,
    List.flatMap_map, List.map_flatMap, List.map_map, Function.comp_def]

attribute [local irreducible] Ir.evaluateParallelIterations

theorem dPreprocessOutcome
    (samplers : MxxSamplerFamily) (message : Bool) (wires : Ir.WireEnvironment)
    (member : wires ∈
      Ir.evaluateNodes
        (Ir.childRunnerWithFuel samplers DiamondInputInjector_stage_preprocess
          DiamondInputInjector_stage_preprocess.definitions.length)
        samplers [] [("message", .boolean message)]
        DiamondInputInjector_stage_preprocess.root.nodes 0 [[]]) :
    ∃ secret ∈ Ir.uniformMatrixSupport (dParams 1 1 0) (-1) 1,
    ∃ b0raw ∈ samplers.trapdoorSample (dParams 2 6 2),
    ∃ e0raw ∈ samplers.gaussianSample (dParams 1 6 1),
    ∃ b1raw ∈ samplers.trapdoorSample (dParams 2 6 2),
    ∃ projectionRaw ∈ samplers.samplePreimage (dParams 6 1 2)
        (dTyped 2 6 2 b1raw) dProjectionTarget,
    ∃ selector0 ∈ Ir.uniformMatrixSupport (dParams 1 1 0) (-1) 1,
    ∃ selector1 ∈ Ir.uniformMatrixSupport (dParams 1 1 0) (-1) 1,
    ∃ transitionError0Raw ∈ samplers.gaussianSample (dParams 2 6 1),
    ∃ transition0Raw ∈ samplers.samplePreimage (dParams 6 6 2)
        (dTyped 2 6 2 b0raw)
        (dTransitionTarget selector0 (dTyped 2 6 2 b1raw)
          (dTyped 2 6 1 transitionError0Raw)),
    ∃ transitionError1Raw ∈ samplers.gaussianSample (dParams 2 6 1),
    ∃ transition1Raw ∈ samplers.samplePreimage (dParams 6 6 2)
        (dTyped 2 6 2 b0raw)
        (dTransitionTarget selector1 (dTyped 2 6 2 b1raw)
          (dTyped 2 6 1 transitionError1Raw)),
    ∃ unused0 unused1 : Ir.Value,
      Ir.collectOutputs DiamondInputInjector_stage_preprocess.root.outputs wires =
        [("initial", .matrix
            (dInitial secret (dTyped 2 6 2 b0raw) (dTyped 1 6 1 e0raw) message)),
          ("projection", .matrix (dNormalize (dTyped 6 1 2 projectionRaw))),
          ("transition-0-0", .matrix (dNormalize (dTyped 6 6 2 transition0Raw))),
          ("transition-0-1", unused0),
          ("transition-1-0", .matrix (dNormalize (dTyped 6 6 2 transition1Raw))),
          ("transition-1-1", unused1)] := by
  have selectorLoop := dPreprocessSelectorLoop samplers []
  have transitionLoop := @dPreprocessTransitionLoop samplers
  have auxiliaryLoop := @dPreprocessAuxiliaryLoop samplers
  dsimp only [DiamondInputInjector_stage_preprocess] at member selectorLoop transitionLoop auxiliaryLoop ⊢
  cases message
  all_goals
    have rangeTwo : List.range 2 = [0, 1] := by decide
    simp (config := { maxSteps := 10000000 }) [Ir.evaluateNodes, Ir.evaluateNode,
      Ir.arguments, Ir.lookupWire, Ir.lookupEnvironment,
      Ir.collectOutputs, Ir.bindOutputs, Ir.IntExpr.evaluate, Ir.MatrixTypeExpr.evaluate,
      dParams, dTyped, dNormalize, dMessageMatrix, dInitial, dIdentity,
      dTransitionTarget, dProjectionTarget, rangeTwo] at member ⊢
  all_goals
    try simp (config := { maxSteps := 10000000 }) [Ir.arguments, Ir.lookupWire,
      Ir.lookupEnvironment,
      Ir.collectOutputs, Ir.bindOutputs, Ir.IntExpr.evaluate, Ir.MatrixTypeExpr.evaluate,
      dParams, dTyped, dNormalize, dIdentity, dTransitionTarget, dProjectionTarget, dZero11,
      rangeTwo, selectorLoop, transitionLoop, auxiliaryLoop] at member
    rcases member with
      ⟨secret, b0raw, e0raw, b1raw, projectionRaw, selectorPorts, transitionPorts,
        transition0Port, auxiliaryPublicRaw, auxiliaryPorts, auxiliary0Port,
        transition1Port, auxiliary1Port, conditions⟩
    rcases conditions with ⟨conditions, wiresIdentity⟩
    rcases conditions with ⟨conditions, auxiliary1Member⟩
    rcases conditions with ⟨conditions, transition1Member⟩
    rcases conditions with ⟨conditions, auxiliary0Member⟩
    rcases conditions with ⟨conditions, auxiliaryLoopMember⟩
    rcases conditions with ⟨conditions, auxiliaryPublicMember⟩
    rcases conditions with ⟨conditions, transition0Member⟩
    rcases conditions with ⟨conditions, transitionLoopMember⟩
    rcases conditions with ⟨conditions, selectorLoopMember⟩
    rcases conditions with ⟨conditions, projectionMember⟩
    rcases conditions with ⟨baseMembers, b1Member⟩
    rcases baseMembers with ⟨secretMember, b0Member, e0Member⟩
    have selectorMembership : selectorPorts ∈
        List.flatMap
          (fun selector0 =>
            List.map (fun selector1 => [[.matrix selector0, .matrix selector1]])
              (Ir.uniformMatrixSupport (dParams 1 1 0) (-1) 1))
          (Ir.uniformMatrixSupport (dParams 1 1 0) (-1) 1) := by
      exact (congrArg (fun outcomes => selectorPorts ∈ outcomes) selectorLoop).mp
        selectorLoopMember
    simp only [List.mem_flatMap, List.mem_map] at selectorMembership
    rcases selectorMembership with
      ⟨selector0, selector0Member, selector1, selector1Member, rfl⟩
    simp [Ir.lookupWire, transitionLoop] at transitionLoopMember
    rcases transitionLoopMember with
      ⟨transitionError0Raw, transition0Raw, transitionError1Raw, transition1Raw,
        transitionMembers, rfl⟩
    rcases transitionMembers with
      ⟨transitionError0Member, transition0RawMember, transitionError1Member,
        transition1RawMember⟩
    simp [Ir.lookupWire] at transition0Member
    subst transition0Port
    simp [Ir.lookupWire] at auxiliaryPublicMember
    subst auxiliaryPublicRaw
    simp [Ir.lookupWire] at auxiliary0Member
    rcases auxiliary0Member with ⟨auxiliaryValues, _auxiliaryValuesMember, rfl⟩
    simp [Ir.lookupWire] at auxiliary1Member
    subst auxiliary1Port
    rcases wiresIdentity with ⟨finalPort, _finalPortMember, rfl⟩
    refine ⟨secret, ?_, b0raw, ?_, e0raw, ?_, b1raw, ?_, projectionRaw, ?_,
      selector0, ?_, selector1, ?_, transitionError0Raw, ?_, transition0Raw, ?_,
      transitionError1Raw, ?_, transition1Raw, ?_, ?_⟩
    · simpa [dParams] using secretMember
    · simpa [dParams] using b0Member
    · simpa [dParams] using e0Member
    · simpa [dParams] using b1Member
    · simpa [dParams, dTyped, dProjectionTarget, Mxx.matrixConcatRows,
        Mxx.Matrix.withSamplerParams, Mxx.reduceCoefficient] using projectionMember
    · simpa [dParams] using selector0Member
    · simpa [dParams] using selector1Member
    · simpa [dParams] using transitionError0Member
    · simpa [dParams, dTyped, dTransitionTarget, dIdentity] using transition0RawMember
    · simpa [dParams] using transitionError1Member
    · simpa [dParams, dTyped, dTransitionTarget, dIdentity] using transition1RawMember
    · simp [Ir.lookupWire, Mxx.Matrix.withSamplerParams,
        dLookupWireAppend, dLookupWireDifferentNode, dParams, dTyped, dNormalize]

theorem dEvaluateOutcome
    (samplers : MxxSamplerFamily) (digit : Mxx.Matrix)
    (initial transition0 transition1 projection : Mxx.Matrix)
    (unused0 unused1 : Ir.Value)
    (digitCase : digit.coefficients.getD 0 0 = 0 ∨ digit.coefficients.getD 0 0 = 1)
    (wires : Ir.WireEnvironment)
    (member : wires ∈
      Ir.evaluateNodes
        (Ir.childRunnerWithFuel samplers DiamondInputInjector_stage_evaluate
          DiamondInputInjector_stage_evaluate.definitions.length)
        samplers []
        [("initial", .matrix initial), ("digit", .matrix digit),
          ("transition-0-0", .matrix transition0),
          ("transition-1-0", .matrix transition1),
          ("transition-0-1", unused0), ("transition-1-1", unused1),
          ("projection", .matrix projection)]
        DiamondInputInjector_stage_evaluate.root.nodes 0 [[]]) :
    Ir.collectOutputs DiamondInputInjector_stage_evaluate.root.outputs wires =
      dOutput initial
        (if digit.coefficients.getD 0 0 = 0 then transition0 else transition1)
        projection := by
  rcases digitCase with digitZero | digitOne
  all_goals try simp only [List.getD_eq_getElem?_getD] at digitZero
  all_goals try simp only [List.getD_eq_getElem?_getD] at digitOne
  all_goals
    have selectedLoop := dEvaluateSelectedLoop samplers digit initial transition0 transition1
      unused0 unused1 (by aesop)
    dsimp only [DiamondInputInjector_stage_evaluate] at member selectedLoop ⊢
    try simp [List.getD_eq_getElem?_getD, digitZero] at selectedLoop
    try simp [List.getD_eq_getElem?_getD, digitOne] at selectedLoop
    simp (config := { maxSteps := 10000000 }) [Ir.evaluateNodes, Ir.evaluateNode,
      Ir.arguments, Ir.lookupWire, Ir.lookupEnvironment, Ir.collectOutputs,
      Ir.bindOutputs, Ir.IntExpr.evaluate, dOutput, List.getD_eq_getElem?_getD, *] at member ⊢
    subst wires
    exact dThresholdOutput _

theorem traceOutcome
    (samplers : MxxSamplerFamily) (p : DiamondInputInjectorParams)
    (x : DiamondInputInjectorInputs p) (output : Ir.Environment)
    (digitCase : x.digit.coefficients.getD 0 0 = 0 ∨
      x.digit.coefficients.getD 0 0 = 1)
    (member : output ∈ DiamondInputInjectorConcreteOutcomes samplers p x) :
    ∃ secret ∈ Ir.uniformMatrixSupport (dParams 1 1 0) (-1) 1,
    ∃ b0raw ∈ samplers.trapdoorSample (dParams 2 6 2),
    ∃ e0raw ∈ samplers.gaussianSample (dParams 1 6 1),
    ∃ b1raw ∈ samplers.trapdoorSample (dParams 2 6 2),
    ∃ projectionRaw ∈ samplers.samplePreimage (dParams 6 1 2)
        (dTyped 2 6 2 b1raw) dProjectionTarget,
    ∃ selector0 ∈ Ir.uniformMatrixSupport (dParams 1 1 0) (-1) 1,
    ∃ selector1 ∈ Ir.uniformMatrixSupport (dParams 1 1 0) (-1) 1,
    ∃ transitionError0Raw ∈ samplers.gaussianSample (dParams 2 6 1),
    ∃ transition0Raw ∈ samplers.samplePreimage (dParams 6 6 2)
        (dTyped 2 6 2 b0raw)
        (dTransitionTarget selector0 (dTyped 2 6 2 b1raw)
          (dTyped 2 6 1 transitionError0Raw)),
    ∃ transitionError1Raw ∈ samplers.gaussianSample (dParams 2 6 1),
    ∃ transition1Raw ∈ samplers.samplePreimage (dParams 6 6 2)
        (dTyped 2 6 2 b0raw)
        (dTransitionTarget selector1 (dTyped 2 6 2 b1raw)
          (dTyped 2 6 1 transitionError1Raw)),
      output = dOutput
        (dInitial secret (dTyped 2 6 2 b0raw) (dTyped 1 6 1 e0raw) x.message)
        (if x.digit.coefficients.getD 0 0 = 0 then dNormalize (dTyped 6 6 2 transition0Raw)
          else dNormalize (dTyped 6 6 2 transition1Raw))
        (dNormalize (dTyped 6 1 2 projectionRaw)) := by
  rcases x with ⟨message, digit⟩
  simp [DiamondInputInjectorConcreteOutcomes,
    DiamondInputInjector_workflow, DiamondInputInjectorParamEnvironment,
    DiamondInputInjectorInputEnvironment, Ir.denoteWorkflow, Ir.evaluateStages,
    Ir.stageInputs, Ir.resolveStageInput, Ir.lookupStage, Ir.lookupEnvironment,
    Ir.denote, Ir.denoteScopeWithFuel_succ] at member
  rcases member with ⟨preprocessWires, evaluateWires, stageMembers, outputIdentity⟩
  rcases stageMembers with ⟨preprocessMember, evaluateMember⟩
  obtain ⟨secret, secretMember, b0raw, b0Member, e0raw, e0Member, b1raw, b1Member,
    projectionRaw, projectionMember, selector0, selector0Member, selector1, selector1Member,
    transitionError0Raw, transitionError0Member, transition0Raw, transition0Member,
    transitionError1Raw, transitionError1Member, transition1Raw, transition1Member,
    unused0, unused1, preprocessOutputs⟩ :=
    dPreprocessOutcome samplers message preprocessWires preprocessMember
  rw [preprocessOutputs] at evaluateMember
  simp [Ir.lookupEnvironment] at evaluateMember
  have evaluateOutputs := dEvaluateOutcome samplers digit
    (dInitial secret (dTyped 2 6 2 b0raw) (dTyped 1 6 1 e0raw) message)
    (dNormalize (dTyped 6 6 2 transition0Raw))
    (dNormalize (dTyped 6 6 2 transition1Raw))
    (dNormalize (dTyped 6 1 2 projectionRaw)) unused0 unused1 digitCase
    evaluateWires evaluateMember
  refine ⟨secret, secretMember, b0raw, b0Member, e0raw, e0Member, b1raw, b1Member,
    projectionRaw, projectionMember, selector0, selector0Member, selector1, selector1Member,
    transitionError0Raw, transitionError0Member, transition0Raw, transition0Member,
    transitionError1Raw, transitionError1Member, transition1Raw, transition1Member, ?_⟩
  exact outputIdentity.symm.trans evaluateOutputs

abbrev DZMatrix (rows columns : Nat) :=
  _root_.Matrix (Fin rows) (Fin columns) (ZMod 65537)

def dZ (rows columns : Nat) (matrix : Mxx.Matrix) : DZMatrix rows columns :=
  fun row column => (matrix.coefficient row column 0 : ZMod 65537)

theorem dCastEmod (value : Int) :
    ((value % 65537 : Int) : ZMod 65537) = (value : ZMod 65537) := by
  rw [ZMod.intCast_eq_intCast_iff']
  omega

theorem dOptionReduce (value : Option Int) :
    (((value.map (reduceCoefficient 65537)).getD 0 : Int) : ZMod 65537) =
      ((value.getD 0 : Int) : ZMod 65537) := by
  cases value with
  | none => rfl
  | some value => exact dCastEmod value

theorem dAddCoefficientsGetD (left right : List Int) (index : Nat) :
    (Ir.addCoefficients left right).getD index 0 =
      left.getD index 0 + right.getD index 0 := by
  induction left generalizing right index with
  | nil =>
      cases right <;> cases index <;> simp [Ir.addCoefficients]
  | cons left leftTail inductionHypothesis =>
      cases right with
      | nil => cases index <;> simp [Ir.addCoefficients]
      | cons right rightTail =>
          cases index with
          | zero => simp [Ir.addCoefficients]
          | succ index =>
              simpa only [Ir.addCoefficients, List.getD_cons_succ] using
                inductionHypothesis rightTail index

theorem dReducedAddCoefficient (left right : List Int) (index : Nat) :
    ((((Ir.addCoefficients left right)[index]?.map (reduceCoefficient 65537)).getD 0 :
        Int) : ZMod 65537) =
      ((left[index]?.getD 0 : Int) : ZMod 65537) +
        ((right[index]?.getD 0 : Int) : ZMod 65537) := by
  rw [dOptionReduce]
  change ((Ir.addCoefficients left right).getD index 0 : ZMod 65537) =
    (left.getD index 0 : ZMod 65537) + (right.getD index 0 : ZMod 65537)
  rw [dAddCoefficientsGetD, Int.cast_add]

theorem dZMul126 (left right : Mxx.Matrix)
    (leftModulus : left.modulus = 65537) (leftRing : left.ringDimension = 1)
    (leftRows : left.rows = 1) (leftColumns : left.columns = 2)
    (rightModulus : right.modulus = 65537) (rightRing : right.ringDimension = 1)
    (rightRows : right.rows = 2) (rightColumns : right.columns = 6) :
    dZ 1 6 (Mxx.matrixMul left right) = dZ 1 2 left * dZ 2 6 right := by
  rcases left with ⟨leftCoefficients, leftModulusValue, leftRingValue, leftRowsValue,
    leftColumnsValue⟩
  rcases right with ⟨rightCoefficients, rightModulusValue, rightRingValue, rightRowsValue,
    rightColumnsValue⟩
  change leftModulusValue = 65537 at leftModulus
  change leftRingValue = 1 at leftRing
  change leftRowsValue = 1 at leftRows
  change leftColumnsValue = 2 at leftColumns
  change rightModulusValue = 65537 at rightModulus
  change rightRingValue = 1 at rightRing
  change rightRowsValue = 2 at rightRows
  change rightColumnsValue = 6 at rightColumns
  subst leftModulusValue
  subst leftRingValue
  subst leftRowsValue
  subst leftColumnsValue
  subst rightModulusValue
  subst rightRingValue
  subst rightRowsValue
  subst rightColumnsValue
  ext row column
  fin_cases row <;> fin_cases column
  all_goals
    have rangeOne : List.range 1 = [0] := by decide
    have rangeTwo : List.range 2 = [0, 1] := by decide
    have rangeSix : List.range 6 = [0, 1, 2, 3, 4, 5] := by decide
    simp [dZ, Mxx.matrixMul, Mxx.Matrix.coefficient, Mxx.negacyclicCoefficient,
      Mxx.reduceCoefficient, _root_.Matrix.mul_apply, Fin.sum_univ_succ, dCastEmod,
      rangeOne, rangeTwo, rangeSix]

theorem dZMul226 (left right : Mxx.Matrix)
    (leftModulus : left.modulus = 65537) (leftRing : left.ringDimension = 1)
    (leftRows : left.rows = 2) (leftColumns : left.columns = 2)
    (rightModulus : right.modulus = 65537) (rightRing : right.ringDimension = 1)
    (rightRows : right.rows = 2) (rightColumns : right.columns = 6) :
    dZ 2 6 (Mxx.matrixMul left right) = dZ 2 2 left * dZ 2 6 right := by
  rcases left with ⟨leftCoefficients, leftModulusValue, leftRingValue, leftRowsValue,
    leftColumnsValue⟩
  rcases right with ⟨rightCoefficients, rightModulusValue, rightRingValue, rightRowsValue,
    rightColumnsValue⟩
  change leftModulusValue = 65537 at leftModulus
  change leftRingValue = 1 at leftRing
  change leftRowsValue = 2 at leftRows
  change leftColumnsValue = 2 at leftColumns
  change rightModulusValue = 65537 at rightModulus
  change rightRingValue = 1 at rightRing
  change rightRowsValue = 2 at rightRows
  change rightColumnsValue = 6 at rightColumns
  subst leftModulusValue
  subst leftRingValue
  subst leftRowsValue
  subst leftColumnsValue
  subst rightModulusValue
  subst rightRingValue
  subst rightRowsValue
  subst rightColumnsValue
  ext row column
  fin_cases row <;> fin_cases column
  all_goals
    have rangeOne : List.range 1 = [0] := by decide
    have rangeTwo : List.range 2 = [0, 1] := by decide
    have rangeSix : List.range 6 = [0, 1, 2, 3, 4, 5] := by decide
    simp [dZ, Mxx.matrixMul, Mxx.Matrix.coefficient, Mxx.negacyclicCoefficient,
      Mxx.reduceCoefficient, _root_.Matrix.mul_apply, Fin.sum_univ_succ, dCastEmod,
      rangeOne, rangeTwo, rangeSix]

theorem dZMul266 (left right : Mxx.Matrix)
    (leftModulus : left.modulus = 65537) (leftRing : left.ringDimension = 1)
    (leftRows : left.rows = 2) (leftColumns : left.columns = 6)
    (rightModulus : right.modulus = 65537) (rightRing : right.ringDimension = 1)
    (rightRows : right.rows = 6) (rightColumns : right.columns = 6) :
    dZ 2 6 (Mxx.matrixMul left right) = dZ 2 6 left * dZ 6 6 right := by
  rcases left with ⟨leftCoefficients, leftModulusValue, leftRingValue, leftRowsValue,
    leftColumnsValue⟩
  rcases right with ⟨rightCoefficients, rightModulusValue, rightRingValue, rightRowsValue,
    rightColumnsValue⟩
  change leftModulusValue = 65537 at leftModulus
  change leftRingValue = 1 at leftRing
  change leftRowsValue = 2 at leftRows
  change leftColumnsValue = 6 at leftColumns
  change rightModulusValue = 65537 at rightModulus
  change rightRingValue = 1 at rightRing
  change rightRowsValue = 6 at rightRows
  change rightColumnsValue = 6 at rightColumns
  subst leftModulusValue
  subst leftRingValue
  subst leftRowsValue
  subst leftColumnsValue
  subst rightModulusValue
  subst rightRingValue
  subst rightRowsValue
  subst rightColumnsValue
  ext row column
  fin_cases row <;> fin_cases column
  all_goals
    have rangeOne : List.range 1 = [0] := by decide
    have rangeTwo : List.range 2 = [0, 1] := by decide
    have rangeSix : List.range 6 = [0, 1, 2, 3, 4, 5] := by decide
    simp [dZ, Mxx.matrixMul, Mxx.Matrix.coefficient, Mxx.negacyclicCoefficient,
      Mxx.reduceCoefficient, _root_.Matrix.mul_apply, Fin.sum_univ_succ, dCastEmod,
      rangeOne, rangeTwo, rangeSix] <;> ring

theorem dZMul261 (left right : Mxx.Matrix)
    (leftModulus : left.modulus = 65537) (leftRing : left.ringDimension = 1)
    (leftRows : left.rows = 2) (leftColumns : left.columns = 6)
    (rightModulus : right.modulus = 65537) (rightRing : right.ringDimension = 1)
    (rightRows : right.rows = 6) (rightColumns : right.columns = 1) :
    dZ 2 1 (Mxx.matrixMul left right) = dZ 2 6 left * dZ 6 1 right := by
  rcases left with ⟨leftCoefficients, leftModulusValue, leftRingValue, leftRowsValue,
    leftColumnsValue⟩
  rcases right with ⟨rightCoefficients, rightModulusValue, rightRingValue, rightRowsValue,
    rightColumnsValue⟩
  change leftModulusValue = 65537 at leftModulus
  change leftRingValue = 1 at leftRing
  change leftRowsValue = 2 at leftRows
  change leftColumnsValue = 6 at leftColumns
  change rightModulusValue = 65537 at rightModulus
  change rightRingValue = 1 at rightRing
  change rightRowsValue = 6 at rightRows
  change rightColumnsValue = 1 at rightColumns
  subst leftModulusValue
  subst leftRingValue
  subst leftRowsValue
  subst leftColumnsValue
  subst rightModulusValue
  subst rightRingValue
  subst rightRowsValue
  subst rightColumnsValue
  ext row column
  fin_cases row <;> fin_cases column
  all_goals
    have rangeOne : List.range 1 = [0] := by decide
    have rangeTwo : List.range 2 = [0, 1] := by decide
    have rangeSix : List.range 6 = [0, 1, 2, 3, 4, 5] := by decide
    simp [dZ, Mxx.matrixMul, Mxx.Matrix.coefficient, Mxx.negacyclicCoefficient,
      Mxx.reduceCoefficient, _root_.Matrix.mul_apply, Fin.sum_univ_succ, dCastEmod,
      rangeOne, rangeTwo, rangeSix] <;> ring

theorem dZMul166 (left right : Mxx.Matrix)
    (leftModulus : left.modulus = 65537) (leftRing : left.ringDimension = 1)
    (leftRows : left.rows = 1) (leftColumns : left.columns = 6)
    (rightModulus : right.modulus = 65537) (rightRing : right.ringDimension = 1)
    (rightRows : right.rows = 6) (rightColumns : right.columns = 6) :
    dZ 1 6 (Mxx.matrixMul left right) = dZ 1 6 left * dZ 6 6 right := by
  rcases left with ⟨leftCoefficients, leftModulusValue, leftRingValue, leftRowsValue,
    leftColumnsValue⟩
  rcases right with ⟨rightCoefficients, rightModulusValue, rightRingValue, rightRowsValue,
    rightColumnsValue⟩
  change leftModulusValue = 65537 at leftModulus
  change leftRingValue = 1 at leftRing
  change leftRowsValue = 1 at leftRows
  change leftColumnsValue = 6 at leftColumns
  change rightModulusValue = 65537 at rightModulus
  change rightRingValue = 1 at rightRing
  change rightRowsValue = 6 at rightRows
  change rightColumnsValue = 6 at rightColumns
  subst leftModulusValue
  subst leftRingValue
  subst leftRowsValue
  subst leftColumnsValue
  subst rightModulusValue
  subst rightRingValue
  subst rightRowsValue
  subst rightColumnsValue
  ext row column
  fin_cases row <;> fin_cases column
  all_goals
    have rangeOne : List.range 1 = [0] := by decide
    have rangeTwo : List.range 2 = [0, 1] := by decide
    have rangeSix : List.range 6 = [0, 1, 2, 3, 4, 5] := by decide
    simp [dZ, Mxx.matrixMul, Mxx.Matrix.coefficient, Mxx.negacyclicCoefficient,
      Mxx.reduceCoefficient, _root_.Matrix.mul_apply, Fin.sum_univ_succ, dCastEmod,
      rangeOne, rangeTwo, rangeSix] <;> ring

theorem dZMul161 (left right : Mxx.Matrix)
    (leftModulus : left.modulus = 65537) (leftRing : left.ringDimension = 1)
    (leftRows : left.rows = 1) (leftColumns : left.columns = 6)
    (rightModulus : right.modulus = 65537) (rightRing : right.ringDimension = 1)
    (rightRows : right.rows = 6) (rightColumns : right.columns = 1) :
    dZ 1 1 (Mxx.matrixMul left right) = dZ 1 6 left * dZ 6 1 right := by
  rcases left with ⟨leftCoefficients, leftModulusValue, leftRingValue, leftRowsValue,
    leftColumnsValue⟩
  rcases right with ⟨rightCoefficients, rightModulusValue, rightRingValue, rightRowsValue,
    rightColumnsValue⟩
  change leftModulusValue = 65537 at leftModulus
  change leftRingValue = 1 at leftRing
  change leftRowsValue = 1 at leftRows
  change leftColumnsValue = 6 at leftColumns
  change rightModulusValue = 65537 at rightModulus
  change rightRingValue = 1 at rightRing
  change rightRowsValue = 6 at rightRows
  change rightColumnsValue = 1 at rightColumns
  subst leftModulusValue
  subst leftRingValue
  subst leftRowsValue
  subst leftColumnsValue
  subst rightModulusValue
  subst rightRingValue
  subst rightRowsValue
  subst rightColumnsValue
  ext row column
  fin_cases row <;> fin_cases column
  all_goals
    have rangeOne : List.range 1 = [0] := by decide
    have rangeTwo : List.range 2 = [0, 1] := by decide
    have rangeSix : List.range 6 = [0, 1, 2, 3, 4, 5] := by decide
    simp [dZ, Mxx.matrixMul, Mxx.Matrix.coefficient, Mxx.negacyclicCoefficient,
      Mxx.reduceCoefficient, _root_.Matrix.mul_apply, Fin.sum_univ_succ, dCastEmod,
      rangeOne, rangeTwo, rangeSix] <;> ring

def dVectorZ (secret : Mxx.Matrix) (message : Bool) : DZMatrix 1 2 :=
  fun _ column => if column = 0 then dZ 1 1 secret 0 0 else if message then 1 else 0

def dDiagonalZ (selector : Mxx.Matrix) : DZMatrix 2 2 :=
  fun row column =>
    if row = column then if row = 0 then dZ 1 1 selector 0 0 else 1 else 0

def dProjectionZ : DZMatrix 2 1 :=
  fun row _ => if row = 0 then 0 else 32768

theorem dZNormalize (rows columns : Nat) (matrix : Mxx.Matrix)
    (matrixModulus : matrix.modulus = 65537) :
    dZ rows columns (dNormalize matrix) = dZ rows columns matrix := by
  ext row column
  simp only [dZ, dNormalize, Mxx.Matrix.coefficient, matrixModulus]
  rw [show (0 : Int) = reduceCoefficient 65537 0 by
    norm_num [reduceCoefficient]]
  rw [List.getD_map]
  exact dCastEmod _

theorem dZConcatVector (secret : Mxx.Matrix) (message : Bool)
    (secretModulus : secret.modulus = 65537) (secretRing : secret.ringDimension = 1)
    (secretRows : secret.rows = 1) (secretColumns : secret.columns = 1) :
    dZ 1 2 (Mxx.matrixConcatColumns [secret, dMessageMatrix message]) =
      dVectorZ secret message := by
  rcases secret with ⟨coefficients, modulus, ringDimension, rows, columns⟩
  simp only at secretModulus secretRing secretRows secretColumns
  subst modulus
  subst ringDimension
  subst rows
  subst columns
  ext row column
  fin_cases row <;> fin_cases column <;>
    simp [dZ, dVectorZ, dMessageMatrix, Mxx.matrixConcatColumns,
      Mxx.Matrix.coefficient]

theorem dZConcatDiagonal (selector : Mxx.Matrix)
    (selectorModulus : selector.modulus = 65537)
    (selectorRing : selector.ringDimension = 1)
    (selectorRows : selector.rows = 1) (selectorColumns : selector.columns = 1) :
    dZ 2 2 (Mxx.matrixConcatDiagonal [selector, dIdentity]) = dDiagonalZ selector := by
  rcases selector with ⟨coefficients, modulus, ringDimension, rows, columns⟩
  simp only at selectorModulus selectorRing selectorRows selectorColumns
  subst modulus
  subst ringDimension
  subst rows
  subst columns
  ext row column
  fin_cases row <;> fin_cases column
  all_goals
    have rangeTwo : List.range 2 = [0, 1] := by decide
    simp [dZ, dDiagonalZ, dIdentity, Mxx.matrixConcatDiagonal,
      Mxx.diagonalCoefficient, Mxx.Matrix.coefficient, rangeTwo]

theorem dZProjectionTarget : dZ 2 1 dProjectionTarget = dProjectionZ := by
  ext row column
  fin_cases row <;> fin_cases column <;>
    simp [dZ, dProjectionZ, dProjectionTarget, Mxx.Matrix.coefficient]

theorem dZInitial (secret b error : Mxx.Matrix) (message : Bool)
    (secretModulus : secret.modulus = 65537) (secretRing : secret.ringDimension = 1)
    (secretRows : secret.rows = 1) (secretColumns : secret.columns = 1)
    (bModulus : b.modulus = 65537) (bRing : b.ringDimension = 1)
    (bRows : b.rows = 2) (bColumns : b.columns = 6)
    (errorModulus : error.modulus = 65537) (errorRing : error.ringDimension = 1)
    (errorRows : error.rows = 1) (errorColumns : error.columns = 6) :
    dZ 1 6 (dInitial secret b error message) =
      dVectorZ secret message * dZ 2 6 b + dZ 1 6 error := by
  have vectorShape :
      let vector := Mxx.matrixConcatColumns [secret, dMessageMatrix message]
      vector.modulus = 65537 ∧ vector.ringDimension = 1 ∧
        vector.rows = 1 ∧ vector.columns = 2 := by
    simp [Mxx.matrixConcatColumns, secretModulus, secretRing, secretRows, secretColumns,
      dMessageMatrix]
  have productModulus :
      (Mxx.matrixMul (Mxx.matrixConcatColumns [secret, dMessageMatrix message]) b).modulus =
        65537 := by
    simp [Mxx.matrixMul, secretModulus, secretRing, secretRows, secretColumns,
      bModulus, bRing, bRows, bColumns, Mxx.matrixConcatColumns, dMessageMatrix]
  have productRing :
      (Mxx.matrixMul (Mxx.matrixConcatColumns [secret, dMessageMatrix message]) b).ringDimension =
        1 := by
    simp [Mxx.matrixMul, secretModulus, secretRing, secretRows, secretColumns,
      bModulus, bRing, bRows, bColumns, Mxx.matrixConcatColumns, dMessageMatrix]
  have productZ :
      dZ 1 6 (Mxx.matrixMul (Mxx.matrixConcatColumns [secret, dMessageMatrix message]) b) =
        dVectorZ secret message * dZ 2 6 b := by
    rw [dZMul126 (Mxx.matrixConcatColumns [secret, dMessageMatrix message]) b
      (by simp_all) (by simp_all) (by simp_all) (by simp_all)
      bModulus bRing bRows bColumns]
    rw [dZConcatVector secret message secretModulus secretRing secretRows secretColumns]
  rw [dInitial]
  ext row column
  simp only [_root_.Matrix.add_apply]
  rw [← congrFun (congrFun productZ row) column]
  simpa [dZ, Mxx.Matrix.coefficient, productModulus, productRing,
    bModulus, bRing, bRows, bColumns,
    errorModulus, errorRing, errorRows, errorColumns] using
    dReducedAddCoefficient
      (Mxx.matrixMul (Mxx.matrixConcatColumns [secret, dMessageMatrix message]) b).coefficients
      error.coefficients (↑row * 6 + ↑column)

theorem dZTransitionTarget (selector b error : Mxx.Matrix)
    (selectorModulus : selector.modulus = 65537)
    (selectorRing : selector.ringDimension = 1)
    (selectorRows : selector.rows = 1) (selectorColumns : selector.columns = 1)
    (bModulus : b.modulus = 65537) (bRing : b.ringDimension = 1)
    (bRows : b.rows = 2) (bColumns : b.columns = 6)
    (errorModulus : error.modulus = 65537) (errorRing : error.ringDimension = 1)
    (errorRows : error.rows = 2) (errorColumns : error.columns = 6) :
    dZ 2 6 (dTransitionTarget selector b error) =
      dDiagonalZ selector * dZ 2 6 b + dZ 2 6 error := by
  have diagonalShape :
      let diagonal := Mxx.matrixConcatDiagonal [selector, dIdentity]
      diagonal.modulus = 65537 ∧ diagonal.ringDimension = 1 ∧
        diagonal.rows = 2 ∧ diagonal.columns = 2 := by
    simp [Mxx.matrixConcatDiagonal, selectorModulus, selectorRing, selectorRows,
      selectorColumns, dIdentity]
  have productModulus :
      (Mxx.matrixMul (Mxx.matrixConcatDiagonal [selector, dIdentity]) b).modulus =
        65537 := by
    simp [Mxx.matrixMul, selectorModulus, selectorRing, selectorRows, selectorColumns,
      bModulus, bRing, bRows, bColumns, Mxx.matrixConcatDiagonal, dIdentity]
  have productRing :
      (Mxx.matrixMul (Mxx.matrixConcatDiagonal [selector, dIdentity]) b).ringDimension =
        1 := by
    simp [Mxx.matrixMul, selectorModulus, selectorRing, selectorRows, selectorColumns,
      bModulus, bRing, bRows, bColumns, Mxx.matrixConcatDiagonal, dIdentity]
  have productColumns :
      (Mxx.matrixMul (Mxx.matrixConcatDiagonal [selector, dIdentity]) b).columns = 6 := by
    simp [Mxx.matrixMul, selectorModulus, selectorRing, selectorRows, selectorColumns,
      bModulus, bRing, bRows, bColumns, Mxx.matrixConcatDiagonal, dIdentity]
  have productZ :
      dZ 2 6 (Mxx.matrixMul (Mxx.matrixConcatDiagonal [selector, dIdentity]) b) =
        dDiagonalZ selector * dZ 2 6 b := by
    rw [dZMul226 (Mxx.matrixConcatDiagonal [selector, dIdentity]) b
      (by simp_all) (by simp_all) (by simp_all) (by simp_all)
      bModulus bRing bRows bColumns]
    rw [dZConcatDiagonal selector selectorModulus selectorRing selectorRows selectorColumns]
  rw [dTransitionTarget]
  ext row column
  simp only [_root_.Matrix.add_apply]
  rw [← congrFun (congrFun productZ row) column]
  simpa [dZ, Mxx.Matrix.coefficient, productModulus, productRing, productColumns,
    bModulus, bRing, bRows, bColumns,
    errorModulus, errorRing, errorRows, errorColumns] using
    dReducedAddCoefficient
      (Mxx.matrixMul (Mxx.matrixConcatDiagonal [selector, dIdentity]) b).coefficients
      error.coefficients (↑row * 6 + ↑column)

def dC (rows columns : Nat) (matrix : Mxx.Matrix) :
    _root_.Matrix (Fin rows) (Fin columns) Int :=
  fun row column => Mxx.centeredCoefficient matrix.modulus (matrix.coefficient row column 0)

def dVectorC (secret : Mxx.Matrix) (message : Bool) :
    _root_.Matrix (Fin 1) (Fin 2) Int :=
  fun _ column =>
    if column = 0 then dC 1 1 secret 0 0 else if message then 1 else 0

def dCastInt {rows columns : Nat}
    (matrix : _root_.Matrix (Fin rows) (Fin columns) Int) : DZMatrix rows columns :=
  fun row column => (matrix row column : ZMod 65537)

theorem dCastCentered (value : Int) :
    ((Mxx.centeredCoefficient 65537 value : Int) : ZMod 65537) =
      (value : ZMod 65537) := by
  rw [ZMod.intCast_eq_intCast_iff']
  simp only [Mxx.centeredCoefficient, Mxx.reduceCoefficient]
  split <;> omega

theorem dCastC (rows columns : Nat) (matrix : Mxx.Matrix)
    (matrixModulus : matrix.modulus = 65537) :
    dCastInt (dC rows columns matrix) = dZ rows columns matrix := by
  ext row column
  simp [dCastInt, dC, dZ, matrixModulus, dCastCentered]

theorem dCastVectorC (secret : Mxx.Matrix) (message : Bool)
    (secretModulus : secret.modulus = 65537) :
    dCastInt (dVectorC secret message) = dVectorZ secret message := by
  ext row column
  fin_cases row <;> fin_cases column <;>
    simp [dCastInt, dVectorC, dVectorZ, dC, dZ, secretModulus, dCastCentered]

theorem dCEntryBound (rows columns : Nat) (matrix : Mxx.Matrix) (bound : Nat)
    (matrixBound : Mxx.maxCenteredCoefficientNorm matrix ≤ bound) :
    ∀ row column, (dC rows columns matrix row column).natAbs ≤ bound := by
  intro row column
  exact le_trans (Mxx.Toolkit.centeredEntry_natAbs_le_norm matrix row column 0) matrixBound

theorem dVectorCEntryBound (secret : Mxx.Matrix) (message : Bool)
    (secretBound : Mxx.maxCenteredCoefficientNorm secret ≤ 1) :
    ∀ row column, (dVectorC secret message row column).natAbs ≤ 1 := by
  intro row column
  fin_cases row <;> fin_cases column
  · simpa [dVectorC, dC] using dCEntryBound 1 1 secret 1 secretBound 0 0
  · cases message <;> simp [dVectorC]

def dNoise (secret initialError transitionError transition projection : Mxx.Matrix)
    (message : Bool) : Int :=
  (((dVectorC secret message * dC 2 6 transitionError) * dC 6 1 projection) +
    ((dC 1 6 initialError * dC 6 6 transition) * dC 6 1 projection)) 0 0

theorem dNoiseBound (secret initialError transitionError transition projection : Mxx.Matrix)
    (message : Bool)
    (secretBound : Mxx.maxCenteredCoefficientNorm secret ≤ 1)
    (initialErrorBound : Mxx.maxCenteredCoefficientNorm initialError ≤ 1)
    (transitionErrorBound : Mxx.maxCenteredCoefficientNorm transitionError ≤ 1)
    (transitionBound : Mxx.maxCenteredCoefficientNorm transition ≤ 2)
    (projectionBound : Mxx.maxCenteredCoefficientNorm projection ≤ 2) :
    (dNoise secret initialError transitionError transition projection message).natAbs ≤ 168 := by
  have vectorBound := dVectorCEntryBound secret message secretBound
  have initialBound := dCEntryBound 1 6 initialError 1 initialErrorBound
  have errorBound := dCEntryBound 2 6 transitionError 1 transitionErrorBound
  have transitionEntryBound := dCEntryBound 6 6 transition 2 transitionBound
  have projectionEntryBound := dCEntryBound 6 1 projection 2 projectionBound
  have vectorTimesError : ∀ row column,
      ((dVectorC secret message * dC 2 6 transitionError) row column).natAbs ≤ 2 := by
    intro row column
    simpa using Mxx.Toolkit.matrixMulEntry_natAbs_le
      (dVectorC secret message) (dC 2 6 transitionError) 1 1 vectorBound errorBound
      row column
  have signalNoiseBound :
      (((dVectorC secret message * dC 2 6 transitionError) *
        dC 6 1 projection) 0 0).natAbs ≤ 24 := by
    simpa using Mxx.Toolkit.matrixMulEntry_natAbs_le
      (dVectorC secret message * dC 2 6 transitionError) (dC 6 1 projection)
      2 2 vectorTimesError projectionEntryBound 0 0
  have initialTimesTransition : ∀ row column,
      ((dC 1 6 initialError * dC 6 6 transition) row column).natAbs ≤ 12 := by
    intro row column
    simpa using Mxx.Toolkit.matrixMulEntry_natAbs_le
      (dC 1 6 initialError) (dC 6 6 transition) 1 2 initialBound
      transitionEntryBound row column
  have propagatedInitialBound :
      (((dC 1 6 initialError * dC 6 6 transition) *
        dC 6 1 projection) 0 0).natAbs ≤ 144 := by
    simpa using Mxx.Toolkit.matrixMulEntry_natAbs_le
      (dC 1 6 initialError * dC 6 6 transition) (dC 6 1 projection)
      12 2 initialTimesTransition projectionEntryBound 0 0
  unfold dNoise
  exact le_trans (Int.natAbs_add_le _ _) (by omega)

theorem dCastIntAdd {rows columns : Nat}
    (left right : _root_.Matrix (Fin rows) (Fin columns) Int) :
    dCastInt (left + right) = dCastInt left + dCastInt right := by
  ext row column
  simp [dCastInt]

theorem dCastIntMul {rows inner columns : Nat}
    (left : _root_.Matrix (Fin rows) (Fin inner) Int)
    (right : _root_.Matrix (Fin inner) (Fin columns) Int) :
    dCastInt (left * right) = dCastInt left * dCastInt right := by
  ext row column
  simp [dCastInt, _root_.Matrix.mul_apply]

theorem dCastNoise (secret initialError transitionError transition projection : Mxx.Matrix)
    (message : Bool)
    (secretModulus : secret.modulus = 65537)
    (initialErrorModulus : initialError.modulus = 65537)
    (transitionErrorModulus : transitionError.modulus = 65537)
    (transitionModulus : transition.modulus = 65537)
    (projectionModulus : projection.modulus = 65537) :
    ((dNoise secret initialError transitionError transition projection message : Int) :
        ZMod 65537) =
      (((dVectorZ secret message * dZ 2 6 transitionError) * dZ 6 1 projection) +
        ((dZ 1 6 initialError * dZ 6 6 transition) * dZ 6 1 projection)) 0 0 := by
  change dCastInt
      (((dVectorC secret message * dC 2 6 transitionError) * dC 6 1 projection) +
        ((dC 1 6 initialError * dC 6 6 transition) * dC 6 1 projection)) 0 0 = _
  rw [dCastIntAdd, dCastIntMul, dCastIntMul, dCastIntMul, dCastIntMul,
    dCastVectorC secret message secretModulus,
    dCastC 2 6 transitionError transitionErrorModulus,
    dCastC 6 1 projection projectionModulus,
    dCastC 1 6 initialError initialErrorModulus,
    dCastC 6 6 transition transitionModulus]

structure DShape (matrix : Mxx.Matrix) (rows columns : Nat) : Prop where
  modulus : matrix.modulus = 65537
  ringDimension : matrix.ringDimension = 1
  rows : matrix.rows = rows
  columns : matrix.columns = columns

theorem dTypedShape (rows columns bound : Nat) (matrix : Mxx.Matrix) :
    DShape (dTyped rows columns bound matrix) rows columns := by
  exact ⟨rfl, rfl, rfl, rfl⟩

theorem dNormalizeShape (matrix : Mxx.Matrix) (rows columns : Nat)
    (shape : DShape matrix rows columns) : DShape (dNormalize matrix) rows columns := by
  exact ⟨shape.modulus, shape.ringDimension, shape.rows, shape.columns⟩

theorem dInitialShape (secret b error : Mxx.Matrix) (message : Bool)
    (secretShape : DShape secret 1 1) (bShape : DShape b 2 6)
    (errorShape : DShape error 1 6) :
    DShape (dInitial secret b error message) 1 6 := by
  refine ⟨?_, ?_, ?_, ?_⟩ <;>
    simp [dInitial, Mxx.matrixConcatColumns, dMessageMatrix, Mxx.matrixMul,
      secretShape.modulus, secretShape.ringDimension, secretShape.rows, secretShape.columns,
      bShape.modulus, bShape.ringDimension, bShape.rows, bShape.columns]

theorem dTransitionTargetShape (selector b error : Mxx.Matrix)
    (selectorShape : DShape selector 1 1) (bShape : DShape b 2 6)
    (errorShape : DShape error 2 6) :
    DShape (dTransitionTarget selector b error) 2 6 := by
  refine ⟨?_, ?_, ?_, ?_⟩ <;>
    simp [dTransitionTarget, Mxx.matrixConcatDiagonal, dIdentity, Mxx.matrixMul,
      selectorShape.modulus, selectorShape.ringDimension, selectorShape.rows,
      selectorShape.columns, bShape.modulus, bShape.ringDimension, bShape.rows,
      bShape.columns]

theorem dMulShape (left right : Mxx.Matrix) (rows inner columns : Nat)
    (leftShape : DShape left rows inner) (rightShape : DShape right inner columns) :
    DShape (Mxx.matrixMul left right) rows columns := by
  refine ⟨?_, ?_, ?_, ?_⟩ <;>
    simp [Mxx.matrixMul, leftShape.modulus, leftShape.ringDimension,
      leftShape.rows, leftShape.columns, rightShape.modulus, rightShape.ringDimension,
      rightShape.rows, rightShape.columns]

theorem dSignalProduct (secret selector : Mxx.Matrix) (message : Bool) :
    ((dVectorZ secret message * dDiagonalZ selector) * dProjectionZ) 0 0 =
      if message then 32768 else 0 := by
  cases message <;>
    simp [dVectorZ, dDiagonalZ, dProjectionZ, _root_.Matrix.mul_apply,
      Fin.sum_univ_succ]

theorem dOutputResidue (secret b0 initialError b1 selector transitionError transition projection :
    Mxx.Matrix) (message : Bool)
    (secretShape : DShape secret 1 1)
    (b0Shape : DShape b0 2 6)
    (initialErrorShape : DShape initialError 1 6)
    (b1Shape : DShape b1 2 6)
    (selectorShape : DShape selector 1 1)
    (transitionErrorShape : DShape transitionError 2 6)
    (transitionShape : DShape transition 6 6)
    (projectionShape : DShape projection 6 1)
    (transitionRelation : Mxx.matrixMul b0 transition =
      dTransitionTarget selector b1 transitionError)
    (projectionRelation : Mxx.matrixMul b1 projection = dProjectionTarget) :
    ((Mxx.matrixMul
      (Mxx.matrixMul (dInitial secret b0 initialError message) (dNormalize transition))
      (dNormalize projection)).coefficient 0 0 0 : ZMod 65537) =
      (((if message then 32768 else 0) +
        dNoise secret initialError transitionError transition projection message : Int) :
          ZMod 65537) := by
  let initial := dInitial secret b0 initialError message
  let normalizedTransition := dNormalize transition
  let normalizedProjection := dNormalize projection
  have initialShape := dInitialShape secret b0 initialError message secretShape b0Shape
    initialErrorShape
  have normalizedTransitionShape := dNormalizeShape transition 6 6 transitionShape
  have normalizedProjectionShape := dNormalizeShape projection 6 1 projectionShape
  have firstProductShape := dMulShape initial normalizedTransition 1 6 6 initialShape
    normalizedTransitionShape
  have initialZ : dZ 1 6 initial =
      dVectorZ secret message * dZ 2 6 b0 + dZ 1 6 initialError := by
    exact dZInitial secret b0 initialError message secretShape.modulus
      secretShape.ringDimension secretShape.rows secretShape.columns b0Shape.modulus
      b0Shape.ringDimension b0Shape.rows b0Shape.columns initialErrorShape.modulus
      initialErrorShape.ringDimension initialErrorShape.rows initialErrorShape.columns
  have transitionZ :
      dZ 2 6 b0 * dZ 6 6 transition =
        dDiagonalZ selector * dZ 2 6 b1 + dZ 2 6 transitionError := by
    calc
      dZ 2 6 b0 * dZ 6 6 transition = dZ 2 6 (Mxx.matrixMul b0 transition) :=
        (dZMul266 b0 transition b0Shape.modulus b0Shape.ringDimension b0Shape.rows
          b0Shape.columns transitionShape.modulus transitionShape.ringDimension
          transitionShape.rows transitionShape.columns).symm
      _ = dZ 2 6 (dTransitionTarget selector b1 transitionError) := by
        rw [transitionRelation]
      _ = dDiagonalZ selector * dZ 2 6 b1 + dZ 2 6 transitionError :=
        dZTransitionTarget selector b1 transitionError selectorShape.modulus
          selectorShape.ringDimension selectorShape.rows selectorShape.columns
          b1Shape.modulus b1Shape.ringDimension b1Shape.rows b1Shape.columns
          transitionErrorShape.modulus transitionErrorShape.ringDimension
          transitionErrorShape.rows transitionErrorShape.columns
  have projectionZ : dZ 2 6 b1 * dZ 6 1 projection = dProjectionZ := by
    calc
      dZ 2 6 b1 * dZ 6 1 projection = dZ 2 1 (Mxx.matrixMul b1 projection) :=
        (dZMul261 b1 projection b1Shape.modulus b1Shape.ringDimension b1Shape.rows
          b1Shape.columns projectionShape.modulus projectionShape.ringDimension
          projectionShape.rows projectionShape.columns).symm
      _ = dZ 2 1 dProjectionTarget := by rw [projectionRelation]
      _ = dProjectionZ := dZProjectionTarget
  have outputZ :
      dZ 1 1 (Mxx.matrixMul (Mxx.matrixMul initial normalizedTransition)
        normalizedProjection) =
        (dZ 1 6 initial * dZ 6 6 transition) * dZ 6 1 projection := by
    rw [dZMul161 (Mxx.matrixMul initial normalizedTransition) normalizedProjection
      firstProductShape.modulus firstProductShape.ringDimension firstProductShape.rows
      firstProductShape.columns normalizedProjectionShape.modulus
      normalizedProjectionShape.ringDimension normalizedProjectionShape.rows
      normalizedProjectionShape.columns]
    rw [dZMul166 initial normalizedTransition initialShape.modulus
      initialShape.ringDimension initialShape.rows initialShape.columns
      normalizedTransitionShape.modulus normalizedTransitionShape.ringDimension
      normalizedTransitionShape.rows normalizedTransitionShape.columns]
    rw [dZNormalize 6 6 transition transitionShape.modulus,
      dZNormalize 6 1 projection projectionShape.modulus]
  have matrixAlgebra :
      ((dVectorZ secret message * dZ 2 6 b0 + dZ 1 6 initialError) *
          dZ 6 6 transition) * dZ 6 1 projection =
        ((dVectorZ secret message * dDiagonalZ selector) * dProjectionZ) +
          ((dVectorZ secret message * dZ 2 6 transitionError) *
            dZ 6 1 projection) +
          ((dZ 1 6 initialError * dZ 6 6 transition) * dZ 6 1 projection) := by
    rw [_root_.Matrix.add_mul, _root_.Matrix.mul_assoc]
    rw [transitionZ]
    rw [_root_.Matrix.mul_add]
    rw [_root_.Matrix.add_mul, _root_.Matrix.add_mul]
    rw [← _root_.Matrix.mul_assoc (dVectorZ secret message) (dDiagonalZ selector)
      (dZ 2 6 b1)]
    rw [_root_.Matrix.mul_assoc (dVectorZ secret message * dDiagonalZ selector)
      (dZ 2 6 b1) (dZ 6 1 projection)]
    rw [projectionZ, add_assoc]
  change dZ 1 1 (Mxx.matrixMul (Mxx.matrixMul initial normalizedTransition)
    normalizedProjection) 0 0 = _
  rw [outputZ, initialZ, matrixAlgebra]
  simp only [_root_.Matrix.add_apply]
  rw [dSignalProduct]
  rw [Int.cast_add]
  rw [dCastNoise secret initialError transitionError transition projection message
    secretShape.modulus initialErrorShape.modulus transitionErrorShape.modulus
    transitionShape.modulus projectionShape.modulus]
  simp only [_root_.Matrix.add_apply]
  cases message <;> norm_num <;> ring

theorem dThresholdCongruent (left right : Int)
    (congruent : (left : ZMod 65537) = (right : ZMod 65537)) :
    Mxx.Ir.thresholdDecodeBool 65537 2 left =
      Mxx.Ir.thresholdDecodeBool 65537 2 right := by
  have residues : left % (65537 : Int) = right % (65537 : Int) :=
    (ZMod.intCast_eq_intCast_iff' left right 65537).mp congruent
  unfold Mxx.Ir.thresholdDecodeBool Mxx.Ir.centeredRepresentative
  norm_num
  simp only [residues]

theorem dDecodeCorrect (message : Bool) (noise : Int) (bound : noise.natAbs < 16384) :
    Mxx.Ir.thresholdDecodeBool 65537 2
      ((if message then 32768 else 0) + noise) = message := by
  have boundInt : (noise.natAbs : Int) < 16384 := by exact_mod_cast bound
  have lower : -16384 < noise := by
    by_cases nonnegative : 0 ≤ noise
    · omega
    · have boundNeg : ((-noise).natAbs : Int) < 16384 := by
        exact_mod_cast (show (-noise).natAbs < 16384 by simpa using bound)
      rw [Int.natAbs_of_nonneg (by omega)] at boundNeg
      omega
  have upper : noise < 16384 := by
    by_cases nonnegative : 0 ≤ noise
    · rw [Int.natAbs_of_nonneg nonnegative] at boundInt
      omega
    · omega
  cases message
  · by_cases nonnegative : 0 ≤ noise
    · have reduced : noise % 65537 = noise := Int.emod_eq_of_lt nonnegative (by omega)
      simp [Mxx.Ir.thresholdDecodeBool, Mxx.Ir.centeredRepresentative, reduced]
      omega
    · have shifted : (noise + 65537) % 65537 = noise + 65537 :=
        Int.emod_eq_of_lt (by omega) (by omega)
      have reduced : noise % 65537 = noise + 65537 := by
        rw [← shifted]
        omega
      simp [Mxx.Ir.thresholdDecodeBool, Mxx.Ir.centeredRepresentative, reduced]
      omega
  · have reduced : (32768 + noise) % 65537 = 32768 + noise :=
      Int.emod_eq_of_lt (by omega) (by omega)
    simp [Mxx.Ir.thresholdDecodeBool, Mxx.Ir.centeredRepresentative, reduced]
    omega

theorem dOutputDecodes (secret b0 initialError b1 selector transitionError transition projection :
    Mxx.Matrix) (message : Bool)
    (secretShape : DShape secret 1 1)
    (b0Shape : DShape b0 2 6)
    (initialErrorShape : DShape initialError 1 6)
    (b1Shape : DShape b1 2 6)
    (selectorShape : DShape selector 1 1)
    (transitionErrorShape : DShape transitionError 2 6)
    (transitionShape : DShape transition 6 6)
    (projectionShape : DShape projection 6 1)
    (transitionRelation : Mxx.matrixMul b0 transition =
      dTransitionTarget selector b1 transitionError)
    (projectionRelation : Mxx.matrixMul b1 projection = dProjectionTarget)
    (secretBound : Mxx.maxCenteredCoefficientNorm secret ≤ 1)
    (initialErrorBound : Mxx.maxCenteredCoefficientNorm initialError ≤ 1)
    (transitionErrorBound : Mxx.maxCenteredCoefficientNorm transitionError ≤ 1)
    (transitionBound : Mxx.maxCenteredCoefficientNorm transition ≤ 2)
    (projectionBound : Mxx.maxCenteredCoefficientNorm projection ≤ 2) :
    Mxx.Ir.thresholdDecodeBool 65537 2
      ((Mxx.matrixMul
        (Mxx.matrixMul (dInitial secret b0 initialError message) (dNormalize transition))
        (dNormalize projection)).coefficient 0 0 0) = message := by
  have residue := dOutputResidue secret b0 initialError b1 selector transitionError
    transition projection message secretShape b0Shape initialErrorShape b1Shape selectorShape
    transitionErrorShape transitionShape projectionShape transitionRelation projectionRelation
  rw [dThresholdCongruent _ _ residue]
  apply dDecodeCorrect
  exact lt_of_le_of_lt
    (dNoiseBound secret initialError transitionError transition projection message secretBound
      initialErrorBound transitionErrorBound transitionBound projectionBound)
    (by decide)

theorem dUniformShapeAndBound (matrix : Mxx.Matrix)
    (member : matrix ∈ Mxx.Ir.uniformMatrixSupport (dParams 1 1 0) (-1) 1) :
    DShape matrix 1 1 ∧ Mxx.maxCenteredCoefficientNorm matrix ≤ 1 := by
  have range : Mxx.Ir.integerRange (-1) 1 = [-1, 0, 1] := by decide
  simp [Mxx.Ir.uniformMatrixSupport, Mxx.Ir.coefficientVectors, range, dParams,
    Mxx.Matrix.withSamplerParams] at member
  rcases member with rfl | rfl | rfl
  all_goals
    constructor
    · exact ⟨rfl, rfl, rfl, rfl⟩
    · decide

theorem dDigitCases (p : DiamondInputInjectorParams) (x : DiamondInputInjectorInputs p)
    (wellFormed : DiamondInputInjectorInputsWF p x)
    (precondition : DiamondInputInjectorPreconditions p x) :
    x.digit.coefficients.getD 0 0 = 0 ∨ x.digit.coefficients.getD 0 0 = 1 := by
  rcases wellFormed with ⟨_, length, coefficients⟩
  have headMember : x.digit.coefficients.getD 0 0 ∈ x.digit.coefficients := by
    cases values : x.digit.coefficients with
    | nil => simp [values] at length
    | cons head tail => simp
  have range := coefficients _ headMember
  have upper : x.digit.coefficients.getD 0 0 ≤ 1 := by
    simpa [DiamondInputInjectorPreconditions, DiamondInputInjector_requirement_0,
      DiamondInputInjectorParamEnvironment, DiamondInputInjectorInputEnvironment,
      Mxx.Ir.denotePure, Mxx.Ir.denote, Mxx.Ir.denoteScopeWithFuel,
      Mxx.Ir.lookupDefinition, Mxx.Ir.evaluateNodes, Mxx.Ir.evaluateNode,
      Mxx.Ir.arguments, Mxx.Ir.lookupWire, Mxx.Ir.bindOutputs, Mxx.Ir.collectOutputs,
      Mxx.Ir.IntExpr.evaluate, Mxx.Ir.lookupEnvironment, Mxx.Ir.singleBooleanOutput,
      Mxx.Ir.environmentValues, Mxx.Ir.evaluateIntCompare] using precondition
  omega

theorem dProductHead
    (initial transition projection : Mxx.Matrix)
    (initialShape : DShape initial 1 6)
    (transitionShape : DShape transition 6 6)
    (projectionShape : DShape projection 6 1) :
    (Mxx.matrixMul (Mxx.matrixMul initial transition) projection).coefficients.head? =
      some ((Mxx.matrixMul (Mxx.matrixMul initial transition) projection).coefficient 0 0 0) := by
  simp [Mxx.matrixMul, Mxx.Matrix.coefficient, initialShape.modulus,
    initialShape.ringDimension, initialShape.rows, initialShape.columns,
    transitionShape.modulus, transitionShape.ringDimension, transitionShape.rows,
    transitionShape.columns, projectionShape.modulus, projectionShape.ringDimension,
    projectionShape.rows, projectionShape.columns]

theorem dFailureBoolSafe (p : DiamondInputInjectorParams)
    (x : DiamondInputInjectorInputs p) (initial transition projection : Mxx.Matrix)
    (headPresent :
      (Mxx.matrixMul (Mxx.matrixMul initial transition) projection).coefficients.head? =
        some ((Mxx.matrixMul (Mxx.matrixMul initial transition) projection).coefficient 0 0 0))
    (decoded :
      Mxx.Ir.thresholdDecodeBool 65537 2
        ((Mxx.matrixMul (Mxx.matrixMul initial transition) projection).coefficient 0 0 0) =
          x.message) :
    DiamondInputInjectorFailureBool p x (dOutput initial transition projection) = false := by
  rcases x with ⟨message, digit⟩
  cases message <;>
    simpa [DiamondInputInjectorFailureBool, DiamondInputInjectorIdealOutput,
      DiamondInputInjector_ideal, DiamondInputInjectorParamEnvironment,
      DiamondInputInjectorInputEnvironment, Mxx.Ir.denotePure, Mxx.Ir.denote,
      Mxx.Ir.denoteScopeWithFuel, Mxx.Ir.lookupDefinition, Mxx.Ir.evaluateNodes,
      Mxx.Ir.evaluateNode, Mxx.Ir.arguments, Mxx.Ir.lookupWire, Mxx.Ir.bindOutputs,
      Mxx.Ir.collectOutputs, Mxx.Ir.IntExpr.evaluate, Mxx.Ir.lookupEnvironment,
      Mxx.Ir.projectOutputs, Mxx.Ir.environmentValues, Mxx.Ir.environmentValid,
      Mxx.Ir.Value.isValid, Mxx.Ir.valuesEqual, Mxx.Ir.Value.equal, dOutput,
      headPresent] using decoded

def diamondInputInjectorChecker : DiamondInputInjectorParams → Bool := fun _ => true

theorem diamondInputInjector_correct :
    DiamondInputInjectorCorrectStatement diamondInputInjectorChecker := by
  constructor
  intro samplers contract p _checker _paramsValid x wellFormed precondition
  unfold DiamondInputInjectorFailureProbability
  apply Mxx.booleanFailureProbability_eq_zero
  intro output outputMember
  have digitCase := dDigitCases p x wellFormed precondition
  obtain ⟨secret, secretMember, b0raw, b0Member, initialErrorRaw, initialErrorMember,
    b1raw, b1Member, projectionRaw, projectionMember, selector0, selector0Member,
    selector1, selector1Member, transitionError0Raw, transitionError0Member,
    transition0Raw, transition0Member, transitionError1Raw, transitionError1Member,
    transition1Raw, transition1Member, outputIdentity⟩ :=
    traceOutcome samplers p x output digitCase outputMember
  obtain ⟨secretShape, secretBound⟩ := dUniformShapeAndBound secret secretMember
  obtain ⟨selector0Shape, _selector0Bound⟩ :=
    dUniformShapeAndBound selector0 selector0Member
  obtain ⟨selector1Shape, _selector1Bound⟩ :=
    dUniformShapeAndBound selector1 selector1Member
  have initialErrorBound := contract.gaussianHardSupport (dParams 1 6 1)
    initialErrorRaw initialErrorMember
  have transitionError0Bound := contract.gaussianHardSupport (dParams 2 6 1)
    transitionError0Raw transitionError0Member
  have transitionError1Bound := contract.gaussianHardSupport (dParams 2 6 1)
    transitionError1Raw transitionError1Member
  obtain ⟨projectionRelation, projectionBound⟩ := contract.preimageContract
    (dParams 6 1 2) (dTyped 2 6 2 b1raw) dProjectionTarget projectionRaw
    projectionMember
  obtain ⟨transition0Relation, transition0Bound⟩ := contract.preimageContract
    (dParams 6 6 2) (dTyped 2 6 2 b0raw)
    (dTransitionTarget selector0 (dTyped 2 6 2 b1raw)
      (dTyped 2 6 1 transitionError0Raw)) transition0Raw transition0Member
  obtain ⟨transition1Relation, transition1Bound⟩ := contract.preimageContract
    (dParams 6 6 2) (dTyped 2 6 2 b0raw)
    (dTransitionTarget selector1 (dTyped 2 6 2 b1raw)
      (dTyped 2 6 1 transitionError1Raw)) transition1Raw transition1Member
  have initialShape := dInitialShape secret (dTyped 2 6 2 b0raw)
    (dTyped 1 6 1 initialErrorRaw) x.message secretShape (dTypedShape 2 6 2 b0raw)
    (dTypedShape 1 6 1 initialErrorRaw)
  have projectionShape := dNormalizeShape (dTyped 6 1 2 projectionRaw) 6 1
    (dTypedShape 6 1 2 projectionRaw)
  rcases digitCase with digitZero | digitOne
  · have decoded := dOutputDecodes secret (dTyped 2 6 2 b0raw)
      (dTyped 1 6 1 initialErrorRaw) (dTyped 2 6 2 b1raw) selector0
      (dTyped 2 6 1 transitionError0Raw) (dTyped 6 6 2 transition0Raw)
      (dTyped 6 1 2 projectionRaw) x.message secretShape
      (dTypedShape 2 6 2 b0raw) (dTypedShape 1 6 1 initialErrorRaw)
      (dTypedShape 2 6 2 b1raw) selector0Shape
      (dTypedShape 2 6 1 transitionError0Raw) (dTypedShape 6 6 2 transition0Raw)
      (dTypedShape 6 1 2 projectionRaw) transition0Relation projectionRelation secretBound
      initialErrorBound transitionError0Bound transition0Bound projectionBound
    have transitionShape := dNormalizeShape (dTyped 6 6 2 transition0Raw) 6 6
      (dTypedShape 6 6 2 transition0Raw)
    have headPresent := dProductHead
      (dInitial secret (dTyped 2 6 2 b0raw) (dTyped 1 6 1 initialErrorRaw) x.message)
      (dNormalize (dTyped 6 6 2 transition0Raw))
      (dNormalize (dTyped 6 1 2 projectionRaw)) initialShape transitionShape projectionShape
    have safe := dFailureBoolSafe p x
      (dInitial secret (dTyped 2 6 2 b0raw) (dTyped 1 6 1 initialErrorRaw) x.message)
      (dNormalize (dTyped 6 6 2 transition0Raw))
      (dNormalize (dTyped 6 1 2 projectionRaw)) headPresent decoded
    rw [outputIdentity]
    rw [if_pos digitZero]
    exact safe
  · have decoded := dOutputDecodes secret (dTyped 2 6 2 b0raw)
      (dTyped 1 6 1 initialErrorRaw) (dTyped 2 6 2 b1raw) selector1
      (dTyped 2 6 1 transitionError1Raw) (dTyped 6 6 2 transition1Raw)
      (dTyped 6 1 2 projectionRaw) x.message secretShape
      (dTypedShape 2 6 2 b0raw) (dTypedShape 1 6 1 initialErrorRaw)
      (dTypedShape 2 6 2 b1raw) selector1Shape
      (dTypedShape 2 6 1 transitionError1Raw) (dTypedShape 6 6 2 transition1Raw)
      (dTypedShape 6 1 2 projectionRaw) transition1Relation projectionRelation secretBound
      initialErrorBound transitionError1Bound transition1Bound projectionBound
    have transitionShape := dNormalizeShape (dTyped 6 6 2 transition1Raw) 6 6
      (dTypedShape 6 6 2 transition1Raw)
    have headPresent := dProductHead
      (dInitial secret (dTyped 2 6 2 b0raw) (dTyped 1 6 1 initialErrorRaw) x.message)
      (dNormalize (dTyped 6 6 2 transition1Raw))
      (dNormalize (dTyped 6 1 2 projectionRaw)) initialShape transitionShape projectionShape
    have safe := dFailureBoolSafe p x
      (dInitial secret (dTyped 2 6 2 b0raw) (dTyped 1 6 1 initialErrorRaw) x.message)
      (dNormalize (dTyped 6 6 2 transition1Raw))
      (dNormalize (dTyped 6 1 2 projectionRaw)) headPresent decoded
    rw [outputIdentity]
    have nonzero : x.digit.coefficients.getD 0 0 ≠ 0 := by omega
    rw [if_neg nonzero]
    exact safe
