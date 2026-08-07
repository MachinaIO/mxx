import Mxx.Certificate.Analyzer
import Mxx.Certificate.Checker

namespace Mxx.Certificate

/-!
# Mandatory integrated toy recurrence gate

This fixture is intentionally an executable IR program.  It reaches the symbolic recurrence only
through `analyzeProgramState`, and Phase B consumes only the recurrence emitted by that analysis.
The quotient-ring preimage relation represented by the program is

```text
c = s * B + e
B * K = S' * B + E                              (mod R_q)
```

Using the same `B` as the next public matrix makes the one-step graph a closed GGH15 recurrence,
so a parameter count of three exercises the recurrence machinery without statically unrolling
three distinct public-key/preimage stages.
-/

private def toyQ : IntExpr := .constant 100000
private def toyN : IntExpr := .constant 4

private def toyMatrixType (rows columns : Int) : MatrixTypeExpr where
  modulus := toyQ
  ringDimension := toyN
  rows := .constant rows
  columns := .constant columns

private def toyBType : MatrixTypeExpr := toyMatrixType 2 3
private def toySecretType : MatrixTypeExpr := toyMatrixType 1 2
private def toySecretStepType : MatrixTypeExpr := toyMatrixType 2 2
private def toyErrorType : MatrixTypeExpr := toyMatrixType 1 3
private def toyRelationErrorType : MatrixTypeExpr := toyMatrixType 2 3
private def toyPreimageType : MatrixTypeExpr := toyMatrixType 3 3

private def toyTrapdoorType : Mxx.Ir.WireTypeExpr :=
  .trapdoor toyBType (.parameter "toy_sigma") (.constant 2) (.constant 1) (.constant 11)

private def toyStep : Mxx.Ir.Scope := {
  nodes := [
    {
      kind := .input "c"
      arguments := []
      outputTypes := [.matrix toyErrorType]
    },
    {
      kind := .input "k"
      arguments := []
      outputTypes := [.preimage toyPreimageType]
    },
    {
      kind := .matrixMultiply
      arguments := [⟨0, 0⟩, ⟨1, 0⟩]
      outputTypes := [.matrix toyErrorType]
    }
  ]
  outputs := [("next_c", ⟨2, 0⟩)]
  inputNames := ["c", "k"]
}

private def toyProgram : Mxx.Ir.Prog := {
  root := {
    nodes := [
      {
        kind := .trapdoorSample toyBType (.constant 11)
        arguments := []
        outputCount := 2
        outputTypes := [.matrix toyBType, toyTrapdoorType]
      },
      {
        kind := .gaussianSample toySecretStepType (.constant 5)
        arguments := []
        outputTypes := [.matrix toySecretStepType]
      },
      {
        kind := .matrixMultiply
        arguments := [⟨1, 0⟩, ⟨0, 0⟩]
        outputTypes := [.matrix toyRelationErrorType]
      },
      {
        kind := .gaussianSample toyRelationErrorType (.constant 7)
        arguments := []
        outputTypes := [.matrix toyRelationErrorType]
      },
      {
        kind := .matrixAdd
        arguments := [⟨2, 0⟩, ⟨3, 0⟩]
        outputTypes := [.matrix toyRelationErrorType]
      },
      {
        kind := .preimageSample toyPreimageType (.constant 11)
        arguments := [⟨0, 0⟩, ⟨0, 1⟩, ⟨4, 0⟩]
        outputTypes := [.preimage toyPreimageType]
      },
      {
        kind := .gaussianSample toySecretType (.constant 2)
        arguments := []
        outputTypes := [.matrix toySecretType]
      },
      {
        kind := .matrixMultiply
        arguments := [⟨6, 0⟩, ⟨0, 0⟩]
        outputTypes := [.matrix toyErrorType]
      },
      {
        kind := .gaussianSample toyErrorType (.constant 3)
        arguments := []
        outputTypes := [.matrix toyErrorType]
      },
      {
        kind := .matrixAdd
        arguments := [⟨7, 0⟩, ⟨8, 0⟩]
        outputTypes := [.matrix toyErrorType]
      },
      {
        kind := .sequentialLoop "toy-ggh15-step" (.parameter "toy_chain_depth") 0 [] 1
        arguments := [⟨9, 0⟩, ⟨5, 0⟩]
        outputTypes := [.matrix toyErrorType]
      }
    ]
    outputs := [("final_c", ⟨10, 0⟩)]
    inputNames := []
  }
  definitions := [("toy-ggh15-step", toyStep)]
}

private def toyStage : StageId := ⟨"toy-recurrence-gate"⟩

private def toyAnalyzed : Except VerifyError AnalysisState :=
  analyzeProgramState toyStage toyProgram { facts := [] }

/-- The executable program must be accepted through the analyzer's sequential-recurrence path. -/
example : toyAnalyzed.isOk = true := by
  decide

private def toyAnalysisState : AnalysisState :=
  toyAnalyzed.toOption.get (by decide)

/-- Phase A keeps one recurrence transfer and one whole matrix body form.  In particular, it does
not make three symbolic copies of the body when Phase B later instantiates the count at three. -/
example :
    toyAnalysisState.symbolicRecurrences.length = 1 ∧
      toyAnalysisState.symbolicFormArena.entries.size = 1 := by
  decide

private def toyRecurrence : SymbolicRecurrenceTransfer :=
  toyAnalysisState.symbolicRecurrences[0]'(by decide)

/-- The recurrence is owned by the actual root loop and retains the parameter expression. -/
example :
    toyRecurrence.source.loop.site = { stage := toyStage, scope := ⟨[]⟩, node := ⟨10⟩ } ∧
      toyRecurrence.source.count = .parameter "toy_chain_depth" ∧
      toyRecurrence.carriedSchemas = [.matrix toyErrorType .unknown] := by
  decide

private def isSingleWholeMatrixOutput :
    {schemas : List CarriedValueSchema} → SymbolicCarriedOutputVector schemas → Bool
  | [_], .cons (.matrix _) .nil => true
  | _, _ => false

/-- One matrix carried slot is represented by one schema-indexed whole-form result. -/
example : isSingleWholeMatrixOutput toyRecurrence.bodyOutputs = true := by
  decide

private def toyParameters : Mxx.Ir.ParamEnvironment := [
  (.parameter "toy_chain_depth", .integer 3)
]

private def toyAnalysis : AnalysisResult where
  expressionArena := toyAnalysisState.expressionArena
  symbolicFormArena := toyAnalysisState.symbolicFormArena
  boundWitnessArena := toyAnalysisState.boundWitnessArena
  symbolicMatrixFacts := toyAnalysisState.symbolicMatrixFacts
  facts := toyAnalysisState.facts
  families := toyAnalysisState.families
  parallelFamilyDerivations := toyAnalysisState.parallelFamilyDerivations
  symbolicRecurrences := toyAnalysisState.symbolicRecurrences
  staticObligations := toyAnalysisState.staticObligations
  inputObligations := []
  semanticObligations := []
  endpointFacts := []
  usedRules := []

/-!
The exact hard-bound calculation expected from the analyzer-produced transition is:

```text
q/2 = 50000, n = 4, secret rows = 2, public columns = 3
a_0 = 2, eta_0 = 3, total_0 = 50000

a_(i+1)   = min(50000, 4 * 2 * a_i * 5)
eta_(i+1) = min(50000, 4 * 2 * a_i * 7 + 4 * 3 * eta_i * 11)
total_(i+1) = min(50000, 4 * 2 * a_(i+1) * 50000 + eta_(i+1))

i=1: a_1=80,    eta_1=508,   total_1=50000
i=2: a_2=3200,  eta_2=50000, total_2=50000
i=3: a_3=50000, eta_3=50000, total_3=50000
```

No CLT or probabilistic estimate participates in this computation.
-/
private def toyResolvedBounds : Option (Bool × Nat × Nat × Nat) :=
  match checkStaticParameters toyAnalysis toyParameters with
  | .ok { symbolicRecurrenceStates := { entries := [{
      schemas := [.matrixSummary]
      values := .cons (.matrix signal coefficient noise total) .nil
      ..
    }] }, .. } => some (signal, coefficient, noise, total)
  | _ => none

example : toyResolvedBounds = some (true, 50000, 50000, 50000) := by
  -- `native_decide` is used only for this closed Phase-B checker evaluation.
  native_decide

private def toySchemaBreakingStep : Mxx.Ir.Scope := {
  toyStep with
  nodes := toyStep.nodes.take 2 ++ [{
    kind := .extractCoefficient (.constant 0)
    arguments := [⟨0, 0⟩]
    outputTypes := [.integer]
  }]
  outputs := [("next_c", ⟨2, 0⟩)]
}

private def toySchemaBreakingProgram : Mxx.Ir.Prog := {
  toyProgram with definitions := [("toy-ggh15-step", toySchemaBreakingStep)]
}

/-- A body that changes the retained carried value kind fails closed in recurrence construction. -/
example :
    (analyzeProgramState toyStage toySchemaBreakingProgram { facts := [] }).isOk = false := by
  decide

end Mxx.Certificate
