import Mxx.Certificate.FrozenDependencySlice

namespace Mxx.Certificate

deriving instance ReflBEq for Mxx.Ir.IntExpr
deriving instance LawfulBEq for Mxx.Ir.IntExpr
deriving instance ReflBEq for Mxx.Ir.RealExpr
deriving instance LawfulBEq for Mxx.Ir.RealExpr
deriving instance ReflBEq for Mxx.Ir.MatrixTypeExpr
deriving instance LawfulBEq for Mxx.Ir.MatrixTypeExpr
deriving instance ReflBEq for Mxx.Ir.WireRef
deriving instance LawfulBEq for Mxx.Ir.WireRef
deriving instance ReflBEq for Mxx.Ir.IntBinaryOp
deriving instance LawfulBEq for Mxx.Ir.IntBinaryOp
deriving instance ReflBEq for Mxx.Ir.IntCompareOp
deriving instance LawfulBEq for Mxx.Ir.IntCompareOp
deriving instance ReflBEq for Mxx.Ir.RealBinaryOp
deriving instance LawfulBEq for Mxx.Ir.RealBinaryOp
deriving instance ReflBEq for Mxx.Ir.ConcatAxis
deriving instance LawfulBEq for Mxx.Ir.ConcatAxis
deriving instance ReflBEq for Mxx.Ir.LoopInputMode
deriving instance LawfulBEq for Mxx.Ir.LoopInputMode
deriving instance ReflBEq for Mxx.Ir.NodeKind
deriving instance LawfulBEq for Mxx.Ir.NodeKind

/-!
# Closed pointwise-formula validation inversion

These helpers invert the Boolean provenance validator into typed facts about the exact frozen
scope node selected by a formula.  They do not trust a certificate-supplied node: both the scope
and node are recovered from `FrozenPointwiseMatrixProgramFormula.validIn`.
-/

/-- Exact frozen node facts shared by the arithmetic validation inversions below. -/
structure ValidatedPointwiseNode
    (program : Mxx.Ir.Prog)
    (scopeId : StaticScopeId)
    (wire : Mxx.Ir.WireRef)
    (expectedKind : Mxx.Ir.NodeKind)
    (expectedArguments : List Mxx.Ir.WireRef) : Prop where
  outputPortZero : wire.port = 0
  nodeFound : ∃ scope outputCount outputTypes,
    scopeAtStaticPath? program scopeId = some scope ∧
      scope.nodes[wire.node]? = some {
        kind := expectedKind
        arguments := expectedArguments
        outputCount
        outputTypes
      }

private theorem validatedPointwiseNode_of_check
    {program : Mxx.Ir.Prog}
    {scopeId : StaticScopeId}
    {wire : Mxx.Ir.WireRef}
    {expectedKind : Mxx.Ir.NodeKind}
    {expectedArguments : List Mxx.Ir.WireRef}
    {check : Mxx.Ir.Scope → Mxx.Ir.Node → Bool}
    (valid : pointwiseFormulaNodeValid program scopeId wire check = true)
    (invert : ∀ scope node, check scope node = true →
      wire.port = 0 ∧ node.kind = expectedKind ∧ node.arguments = expectedArguments) :
    ValidatedPointwiseNode program scopeId wire expectedKind expectedArguments := by
  unfold pointwiseFormulaNodeValid at valid
  split at valid <;> try simp_all
  rename_i scope scopeFound
  split at valid <;> try simp_all
  rename_i node nodeFound
  obtain ⟨outputPortZero, kindEq, argumentsEq⟩ := invert scope node valid.2
  constructor
  · exact outputPortZero
  · refine ⟨scope, node.outputCount, node.outputTypes, scopeFound, ?_⟩
    cases node
    simp_all

theorem FrozenPointwiseMatrixProgramFormula.validZeroNode
    {program : Mxx.Ir.Prog}
    {substitutions : List FrozenPointwiseMatrixProgramFormula}
    {scopeId : StaticScopeId}
    {wire : Mxx.Ir.WireRef}
    {matrixType : Mxx.Ir.MatrixTypeExpr}
    (valid : (FrozenPointwiseMatrixProgramFormula.zero scopeId wire matrixType).validIn
      program substitutions = true) :
    ValidatedPointwiseNode program scopeId wire (.zeroMatrix matrixType) [] := by
  unfold FrozenPointwiseMatrixProgramFormula.validIn at valid
  apply validatedPointwiseNode_of_check valid
  intro scope node checked
  rcases node with ⟨kind, arguments, outputCount, outputTypes⟩
  cases kind <;> cases arguments <;> simp_all

theorem FrozenPointwiseMatrixProgramFormula.validIdentityNode
    {program : Mxx.Ir.Prog}
    {substitutions : List FrozenPointwiseMatrixProgramFormula}
    {scopeId : StaticScopeId}
    {wire : Mxx.Ir.WireRef}
    {matrixType : Mxx.Ir.MatrixTypeExpr}
    (valid : (FrozenPointwiseMatrixProgramFormula.identity scopeId wire matrixType).validIn
      program substitutions = true) :
    ValidatedPointwiseNode program scopeId wire (.identityMatrix matrixType) [] := by
  unfold FrozenPointwiseMatrixProgramFormula.validIn at valid
  apply validatedPointwiseNode_of_check valid
  intro scope node checked
  rcases node with ⟨kind, arguments, outputCount, outputTypes⟩
  cases kind <;> cases arguments <;> simp_all

theorem FrozenPointwiseMatrixProgramFormula.validConstantNode
    {program : Mxx.Ir.Prog}
    {substitutions : List FrozenPointwiseMatrixProgramFormula}
    {scopeId : StaticScopeId}
    {wire : Mxx.Ir.WireRef}
    {matrixType : Mxx.Ir.MatrixTypeExpr}
    {coefficients : List Mxx.Ir.IntExpr}
    (valid : (FrozenPointwiseMatrixProgramFormula.constant scopeId wire matrixType
      coefficients).validIn program substitutions = true) :
    ValidatedPointwiseNode program scopeId wire
      (.constantMatrix matrixType coefficients) [] := by
  unfold FrozenPointwiseMatrixProgramFormula.validIn at valid
  apply validatedPointwiseNode_of_check valid
  intro scope node checked
  rcases node with ⟨kind, arguments, outputCount, outputTypes⟩
  cases kind <;> cases arguments <;> simp_all

theorem FrozenPointwiseMatrixProgramFormula.validGadgetNode
    {program : Mxx.Ir.Prog}
    {substitutions : List FrozenPointwiseMatrixProgramFormula}
    {scopeId : StaticScopeId}
    {wire : Mxx.Ir.WireRef}
    {matrixType : Mxx.Ir.MatrixTypeExpr}
    {base : Mxx.Ir.IntExpr}
    (valid : (FrozenPointwiseMatrixProgramFormula.gadget scopeId wire matrixType base).validIn
      program substitutions = true) :
    ValidatedPointwiseNode program scopeId wire (.gadgetMatrix matrixType base) [] := by
  unfold FrozenPointwiseMatrixProgramFormula.validIn at valid
  apply validatedPointwiseNode_of_check valid
  intro scope node checked
  rcases node with ⟨kind, arguments, outputCount, outputTypes⟩
  cases kind <;> cases arguments <;> simp_all

theorem FrozenPointwiseMatrixProgramFormula.validDecomposeNode
    {program : Mxx.Ir.Prog}
    {substitutions : List FrozenPointwiseMatrixProgramFormula}
    {scopeId : StaticScopeId}
    {wire : Mxx.Ir.WireRef}
    {matrixType : Mxx.Ir.MatrixTypeExpr}
    {base digitCount : Mxx.Ir.IntExpr}
    {input : FrozenPointwiseMatrixProgramFormula}
    (valid : (FrozenPointwiseMatrixProgramFormula.decompose scopeId wire matrixType base
      digitCount input).validIn program substitutions = true) :
    ValidatedPointwiseNode program scopeId wire
      (.gadgetDecompose matrixType base digitCount) [input.source.2] := by
  unfold FrozenPointwiseMatrixProgramFormula.validIn at valid
  apply validatedPointwiseNode_of_check valid
  intro scope node checked
  rcases node with ⟨kind, arguments, outputCount, outputTypes⟩
  cases kind <;> simp_all [pointwiseFormulaArgumentsMatch]

theorem FrozenPointwiseMatrixProgramFormula.validPreimageNode
    {program : Mxx.Ir.Prog}
    {substitutions : List FrozenPointwiseMatrixProgramFormula}
    {scopeId : StaticScopeId}
    {wire publicWire trapdoor target : Mxx.Ir.WireRef}
    {matrixType : Mxx.Ir.MatrixTypeExpr}
    {cutoff : Mxx.Ir.IntExpr}
    (valid : (FrozenPointwiseMatrixProgramFormula.preimage scopeId wire matrixType cutoff
      publicWire trapdoor target).validIn program substitutions = true) :
    ValidatedPointwiseNode program scopeId wire (.preimageSample matrixType cutoff)
      [publicWire, trapdoor, target] := by
  unfold FrozenPointwiseMatrixProgramFormula.validIn at valid
  apply validatedPointwiseNode_of_check valid
  intro scope node checked
  rcases node with ⟨kind, arguments, outputCount, outputTypes⟩
  cases kind <;> simp_all

theorem FrozenPointwiseMatrixProgramFormula.validReshapeNode
    {program : Mxx.Ir.Prog} {substitutions : List FrozenPointwiseMatrixProgramFormula}
    {scopeId : StaticScopeId} {wire : Mxx.Ir.WireRef}
    {rows columns : Mxx.Ir.IntExpr} {input : FrozenPointwiseMatrixProgramFormula}
    (valid : (FrozenPointwiseMatrixProgramFormula.reshape scopeId wire rows columns input).validIn
      program substitutions = true) :
    ValidatedPointwiseNode program scopeId wire (.reshape rows columns) [input.source.2] := by
  unfold FrozenPointwiseMatrixProgramFormula.validIn at valid
  apply validatedPointwiseNode_of_check valid
  intro scope node checked
  rcases node with ⟨kind, arguments, outputCount, outputTypes⟩
  cases kind <;> simp_all [pointwiseFormulaArgumentsMatch]

theorem FrozenPointwiseMatrixProgramFormula.validSliceNode
    {program : Mxx.Ir.Prog} {substitutions : List FrozenPointwiseMatrixProgramFormula}
    {scopeId : StaticScopeId} {wire : Mxx.Ir.WireRef}
    {rows columns : Option (Mxx.Ir.IntExpr × Mxx.Ir.IntExpr)}
    {input : FrozenPointwiseMatrixProgramFormula}
    (valid : (FrozenPointwiseMatrixProgramFormula.slice scopeId wire rows columns input).validIn
      program substitutions = true) :
    ValidatedPointwiseNode program scopeId wire (.slice rows columns) [input.source.2] := by
  unfold FrozenPointwiseMatrixProgramFormula.validIn at valid
  apply validatedPointwiseNode_of_check valid
  intro scope node checked
  rcases node with ⟨kind, arguments, outputCount, outputTypes⟩
  cases kind <;> simp_all [pointwiseFormulaArgumentsMatch]

theorem FrozenPointwiseMatrixProgramFormula.validConcatRowsNode
    {program : Mxx.Ir.Prog} {substitutions : List FrozenPointwiseMatrixProgramFormula}
    {scopeId : StaticScopeId} {wire : Mxx.Ir.WireRef}
    {left right : FrozenPointwiseMatrixProgramFormula}
    (valid : (FrozenPointwiseMatrixProgramFormula.concatRows scopeId wire left right).validIn
      program substitutions = true) :
    ValidatedPointwiseNode program scopeId wire (.concat .rows)
      [left.source.2, right.source.2] := by
  unfold FrozenPointwiseMatrixProgramFormula.validIn at valid
  apply validatedPointwiseNode_of_check valid
  intro scope node checked
  rcases node with ⟨kind, arguments, outputCount, outputTypes⟩
  cases kind <;> simp_all [pointwiseFormulaArgumentsMatch]

theorem FrozenPointwiseMatrixProgramFormula.validAddNode
    {program : Mxx.Ir.Prog}
    {substitutions : List FrozenPointwiseMatrixProgramFormula}
    {scopeId : StaticScopeId}
    {wire : Mxx.Ir.WireRef}
    {left right : FrozenPointwiseMatrixProgramFormula}
    (valid : (FrozenPointwiseMatrixProgramFormula.add scopeId wire left right).validIn
      program substitutions = true) :
    ValidatedPointwiseNode program scopeId wire .matrixAdd [left.source.2, right.source.2] := by
  unfold FrozenPointwiseMatrixProgramFormula.validIn at valid
  apply validatedPointwiseNode_of_check valid
  intro scope node checked
  rcases node with ⟨kind, arguments, outputCount, outputTypes⟩
  cases kind <;> simp_all [pointwiseFormulaArgumentsMatch]

theorem FrozenPointwiseMatrixProgramFormula.validSubtractNode
    {program : Mxx.Ir.Prog}
    {substitutions : List FrozenPointwiseMatrixProgramFormula}
    {scopeId : StaticScopeId}
    {wire : Mxx.Ir.WireRef}
    {left right : FrozenPointwiseMatrixProgramFormula}
    (valid : (FrozenPointwiseMatrixProgramFormula.subtract scopeId wire left right).validIn
      program substitutions = true) :
    ValidatedPointwiseNode program scopeId wire .matrixSubtract
      [left.source.2, right.source.2] := by
  unfold FrozenPointwiseMatrixProgramFormula.validIn at valid
  apply validatedPointwiseNode_of_check valid
  intro scope node checked
  rcases node with ⟨kind, arguments, outputCount, outputTypes⟩
  cases kind <;> simp_all [pointwiseFormulaArgumentsMatch]

theorem FrozenPointwiseMatrixProgramFormula.validMultiplyNode
    {program : Mxx.Ir.Prog}
    {substitutions : List FrozenPointwiseMatrixProgramFormula}
    {scopeId : StaticScopeId}
    {wire : Mxx.Ir.WireRef}
    {left right : FrozenPointwiseMatrixProgramFormula}
    (valid : (FrozenPointwiseMatrixProgramFormula.multiply scopeId wire left right).validIn
      program substitutions = true) :
    ValidatedPointwiseNode program scopeId wire .matrixMultiply
      [left.source.2, right.source.2] := by
  unfold FrozenPointwiseMatrixProgramFormula.validIn at valid
  apply validatedPointwiseNode_of_check valid
  intro scope node checked
  rcases node with ⟨kind, arguments, outputCount, outputTypes⟩
  cases kind <;> simp_all [pointwiseFormulaArgumentsMatch]

theorem FrozenPointwiseMatrixProgramFormula.validNegateNode
    {program : Mxx.Ir.Prog}
    {substitutions : List FrozenPointwiseMatrixProgramFormula}
    {scopeId : StaticScopeId}
    {wire : Mxx.Ir.WireRef}
    {input : FrozenPointwiseMatrixProgramFormula}
    (valid : (FrozenPointwiseMatrixProgramFormula.negate scopeId wire input).validIn
      program substitutions = true) :
    ValidatedPointwiseNode program scopeId wire .matrixNegate [input.source.2] := by
  unfold FrozenPointwiseMatrixProgramFormula.validIn at valid
  apply validatedPointwiseNode_of_check valid
  intro scope node checked
  rcases node with ⟨kind, arguments, outputCount, outputTypes⟩
  cases kind <;> simp_all [pointwiseFormulaArgumentsMatch]

theorem FrozenPointwiseMatrixProgramFormula.validScaleNode
    {program : Mxx.Ir.Prog}
    {substitutions : List FrozenPointwiseMatrixProgramFormula}
    {scopeId : StaticScopeId}
    {wire : Mxx.Ir.WireRef}
    {scalar : Mxx.Ir.IntExpr}
    {input : FrozenPointwiseMatrixProgramFormula}
    (valid : (FrozenPointwiseMatrixProgramFormula.scale scopeId wire scalar input).validIn
      program substitutions = true) :
    ValidatedPointwiseNode program scopeId wire (.matrixScale scalar) [input.source.2] := by
  unfold FrozenPointwiseMatrixProgramFormula.validIn at valid
  apply validatedPointwiseNode_of_check valid
  intro scope node checked
  rcases node with ⟨kind, arguments, outputCount, outputTypes⟩
  cases kind <;> simp_all [pointwiseFormulaArgumentsMatch]

theorem FrozenPointwiseMatrixProgramFormula.validScaleOneNode
    {program : Mxx.Ir.Prog}
    {substitutions : List FrozenPointwiseMatrixProgramFormula}
    {scopeId : StaticScopeId}
    {wire : Mxx.Ir.WireRef}
    {input : FrozenPointwiseMatrixProgramFormula}
    (valid : (FrozenPointwiseMatrixProgramFormula.scaleOne scopeId wire input).validIn
      program substitutions = true) :
    ValidatedPointwiseNode program scopeId wire (.matrixScale (.constant 1))
      [input.source.2] := by
  unfold FrozenPointwiseMatrixProgramFormula.validIn at valid
  apply validatedPointwiseNode_of_check valid
  intro scope node checked
  rcases node with ⟨kind, arguments, outputCount, outputTypes⟩
  cases kind <;> simp_all [pointwiseFormulaArgumentsMatch]

theorem FrozenPointwiseMatrixProgramFormula.validSelectNode
    {program : Mxx.Ir.Prog}
    {substitutions : List FrozenPointwiseMatrixProgramFormula}
    {scopeId : StaticScopeId}
    {wire index : Mxx.Ir.WireRef}
    {branches : List FrozenPointwiseMatrixProgramFormula}
    (valid : (FrozenPointwiseMatrixProgramFormula.select scopeId wire index branches).validIn
      program substitutions = true) :
    ValidatedPointwiseNode program scopeId wire .select
      (index :: branches.map (fun branch => branch.source.2)) := by
  unfold FrozenPointwiseMatrixProgramFormula.validIn at valid
  apply validatedPointwiseNode_of_check valid
  intro scope node checked
  rcases node with ⟨kind, arguments, outputCount, outputTypes⟩
  cases kind <;> simp_all [pointwiseFormulaArgumentsMatch]

end Mxx.Certificate
