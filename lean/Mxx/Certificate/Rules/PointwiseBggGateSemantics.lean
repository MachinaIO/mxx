import Mxx.Certificate.Rules.PointwiseFormulaSemantics
import Mxx.Certificate.Rules.RecurrenceCoupling

namespace Mxx.Certificate

/-!
# Program-preserving BGG gate formula semantics

The BGG role checker intentionally compares erased arithmetic formulas.  Runtime soundness,
however, must remain attached to the exact provenance-preserving program formulas.  This module
connects those two views without treating erasure as injective: transparent call, loop, input, and
`scaleOne` boundaries remain present in every semantic witness.
-/

/-- The exact six retained program formulas corresponding, by position, to one checked erased
six-way gate skeleton. -/
structure CheckedSixWayMatrixProgramSkeleton
    (programFormulas : List FrozenPointwiseMatrixProgramFormula)
    (skeleton : CheckedSixWayMatrixSkeleton) where
  zeroProgram : FrozenPointwiseMatrixProgramFormula
  oneProgram : FrozenPointwiseMatrixProgramFormula
  leftProgram : FrozenPointwiseMatrixProgramFormula
  notProgram : FrozenPointwiseMatrixProgramFormula
  andProgram : FrozenPointwiseMatrixProgramFormula
  xorProgram : FrozenPointwiseMatrixProgramFormula
  programsMatch : programFormulas =
    [zeroProgram, oneProgram, leftProgram, notProgram, andProgram, xorProgram]
  zeroErases : zeroProgram.erase = skeleton.zeroFormula
  oneErases : oneProgram.erase = skeleton.oneFormula
  leftErases : leftProgram.erase = skeleton.leftFormula
  notErases : notProgram.erase = skeleton.notFormula
  andErases : andProgram.erase = skeleton.andFormula
  xorErases : xorProgram.erase = skeleton.xorFormula

/-- Positional lifting from the analyzer-retained program formulas to a checked erased skeleton.
This does not invert erasure and therefore does not discard transparent execution boundaries. -/
theorem CheckedSixWayMatrixProgramSkeleton.ofErasedList
    (programFormulas : List FrozenPointwiseMatrixProgramFormula)
    (skeleton : CheckedSixWayMatrixSkeleton)
    (erases : programFormulas.map FrozenPointwiseMatrixProgramFormula.erase =
      skeleton.formulas) :
    Nonempty (CheckedSixWayMatrixProgramSkeleton programFormulas skeleton) := by
  rw [skeleton.formulasMatch] at erases
  rcases programFormulas with _ | ⟨zeroProgram, programs⟩ <;> simp at erases
  rcases programs with _ | ⟨oneProgram, programs⟩ <;> simp at erases
  rcases programs with _ | ⟨leftProgram, programs⟩ <;> simp at erases
  rcases programs with _ | ⟨notProgram, programs⟩ <;> simp at erases
  rcases programs with _ | ⟨andProgram, programs⟩ <;> simp at erases
  rcases programs with _ | ⟨xorProgram, programs⟩ <;> simp at erases
  rcases programs with _ | ⟨extra, programs⟩ <;> simp at erases
  refine ⟨{
    zeroProgram
    oneProgram
    leftProgram
    notProgram
    andProgram
    xorProgram
    programsMatch := rfl
    zeroErases := ?_
    oneErases := ?_
    leftErases := ?_
    notErases := ?_
    andErases := ?_
    xorErases := ?_
  }⟩ <;> simp_all

/-- One exact denotation whose runtime frame is existentially hidden.  This is used only when a
transparent program boundary moves an arithmetic subexpression into a child frame. -/
structure FrozenPointwiseMatrixProgramFormula.DenotationSomewhere
    {samplers : Mxx.MxxSamplerFamily}
    {program : Mxx.Ir.Prog}
    (formula : FrozenPointwiseMatrixProgramFormula)
    (matrix : Mxx.Matrix) : Type where
  current : ExecutedScope samplers program
  frame : FormulaExecutionFrame samplers program current
  denotes : formula.DenotesAt frame matrix

/-- Exact multiplication found below zero or more transparent program boundaries. -/
structure FrozenPointwiseMatrixProgramFormula.MultiplyWitness
    {samplers : Mxx.MxxSamplerFamily}
    {program : Mxx.Ir.Prog}
    (leftErase rightErase : FrozenPointwiseMatrixFormula)
    (matrix : Mxx.Matrix) : Type where
  leftFormula : FrozenPointwiseMatrixProgramFormula
  rightFormula : FrozenPointwiseMatrixProgramFormula
  leftValue : Mxx.Matrix
  rightValue : Mxx.Matrix
  leftErases : leftFormula.erase = leftErase
  rightErases : rightFormula.erase = rightErase
  leftDenotes : @FrozenPointwiseMatrixProgramFormula.DenotationSomewhere samplers program
    leftFormula leftValue
  rightDenotes : @FrozenPointwiseMatrixProgramFormula.DenotationSomewhere samplers program
    rightFormula rightValue
  valueEq : matrix = Mxx.matrixMultiply leftValue rightValue

/-- Exact addition found below zero or more transparent program boundaries. -/
structure FrozenPointwiseMatrixProgramFormula.AddWitness
    {samplers : Mxx.MxxSamplerFamily}
    {program : Mxx.Ir.Prog}
    (leftErase rightErase : FrozenPointwiseMatrixFormula)
    (matrix : Mxx.Matrix) : Type where
  leftFormula : FrozenPointwiseMatrixProgramFormula
  rightFormula : FrozenPointwiseMatrixProgramFormula
  leftValue : Mxx.Matrix
  rightValue : Mxx.Matrix
  leftErases : leftFormula.erase = leftErase
  rightErases : rightFormula.erase = rightErase
  leftDenotes : @FrozenPointwiseMatrixProgramFormula.DenotationSomewhere samplers program
    leftFormula leftValue
  rightDenotes : @FrozenPointwiseMatrixProgramFormula.DenotationSomewhere samplers program
    rightFormula rightValue
  valueEq : matrix = Mxx.matrixAdd leftValue rightValue

/-- Exact gadget decomposition found below zero or more transparent program boundaries. -/
structure FrozenPointwiseMatrixProgramFormula.DecomposeWitness
    {samplers : Mxx.MxxSamplerFamily}
    {program : Mxx.Ir.Prog}
    (matrixType : MatrixTypeExpr)
    (base digitCount : Mxx.Ir.IntExpr)
    (inputErase : FrozenPointwiseMatrixFormula)
    (matrix : Mxx.Matrix) : Type where
  inputFormula : FrozenPointwiseMatrixProgramFormula
  inputValue : Mxx.Matrix
  matrixParams : Mxx.SamplerParams
  baseValue : Int
  digitCountValue : Int
  inputErases : inputFormula.erase = inputErase
  inputDenotes : @FrozenPointwiseMatrixProgramFormula.DenotationSomewhere samplers program
    inputFormula inputValue
  decompositionRelation : Mxx.MatrixModEq
    (Mxx.matrixMul
      (Mxx.gadgetMatrix {
        matrixParams with
        rows := inputValue.rows
        columns := inputValue.rows * digitCountValue.toNat
      } baseValue digitCountValue.toNat)
      matrix)
    inputValue

/-- Sound inversion of multiplication through only the four constructors erased as transparent
program boundaries. -/
def FrozenPointwiseMatrixProgramFormula.DenotesAt.exposeMultiply
    {samplers : Mxx.MxxSamplerFamily}
    {program : Mxx.Ir.Prog}
    {current : ExecutedScope samplers program}
    {frame : FormulaExecutionFrame samplers program current}
    {formula : FrozenPointwiseMatrixProgramFormula}
    {matrix : Mxx.Matrix}
    {leftErase rightErase : FrozenPointwiseMatrixFormula}
    (denotes : formula.DenotesAt frame matrix)
    (erases : formula.erase = .multiply leftErase rightErase) :
    FrozenPointwiseMatrixProgramFormula.MultiplyWitness (samplers := samplers)
      (program := program) leftErase rightErase matrix := by
  cases denotes with
  | inputSubstitutionSubgraph parentDenotes =>
      exact FrozenPointwiseMatrixProgramFormula.DenotesAt.exposeMultiply parentDenotes erases
  | inputSubstitutionParallel parentDenotes =>
      exact FrozenPointwiseMatrixProgramFormula.DenotesAt.exposeMultiply parentDenotes erases
  | scaleOne inputDenotes =>
      exact FrozenPointwiseMatrixProgramFormula.DenotesAt.exposeMultiply inputDenotes erases
  | subgraphCall _ _ _ _ _ _ _ _ outputDenotes =>
      exact FrozenPointwiseMatrixProgramFormula.DenotesAt.exposeMultiply outputDenotes erases
  | parallelLoop _ _ _ _ _ _ _ _ outputDenotes =>
      exact FrozenPointwiseMatrixProgramFormula.DenotesAt.exposeMultiply outputDenotes erases
  | multiply leftDenotes rightDenotes =>
      simp only [FrozenPointwiseMatrixProgramFormula.erase,
        FrozenPointwiseMatrixFormula.multiply.injEq] at erases
      exact {
        leftFormula := _
        rightFormula := _
        leftValue := _
        rightValue := _
        leftErases := erases.1
        rightErases := erases.2
        leftDenotes := ⟨_, _, leftDenotes⟩
        rightDenotes := ⟨_, _, rightDenotes⟩
        valueEq := rfl
      }
  | _ => simp_all [FrozenPointwiseMatrixProgramFormula.erase]
termination_by sizeOf formula

/-- Sound inversion of addition through only transparent program boundaries. -/
def FrozenPointwiseMatrixProgramFormula.DenotesAt.exposeAdd
    {samplers : Mxx.MxxSamplerFamily}
    {program : Mxx.Ir.Prog}
    {current : ExecutedScope samplers program}
    {frame : FormulaExecutionFrame samplers program current}
    {formula : FrozenPointwiseMatrixProgramFormula}
    {matrix : Mxx.Matrix}
    {leftErase rightErase : FrozenPointwiseMatrixFormula}
    (denotes : formula.DenotesAt frame matrix)
    (erases : formula.erase = .add leftErase rightErase) :
    FrozenPointwiseMatrixProgramFormula.AddWitness (samplers := samplers)
      (program := program) leftErase rightErase matrix := by
  cases denotes with
  | inputSubstitutionSubgraph parentDenotes =>
      exact FrozenPointwiseMatrixProgramFormula.DenotesAt.exposeAdd parentDenotes erases
  | inputSubstitutionParallel parentDenotes =>
      exact FrozenPointwiseMatrixProgramFormula.DenotesAt.exposeAdd parentDenotes erases
  | scaleOne inputDenotes =>
      exact FrozenPointwiseMatrixProgramFormula.DenotesAt.exposeAdd inputDenotes erases
  | subgraphCall _ _ _ _ _ _ _ _ outputDenotes =>
      exact FrozenPointwiseMatrixProgramFormula.DenotesAt.exposeAdd outputDenotes erases
  | parallelLoop _ _ _ _ _ _ _ _ outputDenotes =>
      exact FrozenPointwiseMatrixProgramFormula.DenotesAt.exposeAdd outputDenotes erases
  | add leftDenotes rightDenotes =>
      simp only [FrozenPointwiseMatrixProgramFormula.erase,
        FrozenPointwiseMatrixFormula.add.injEq] at erases
      exact {
        leftFormula := _
        rightFormula := _
        leftValue := _
        rightValue := _
        leftErases := erases.1
        rightErases := erases.2
        leftDenotes := ⟨_, _, leftDenotes⟩
        rightDenotes := ⟨_, _, rightDenotes⟩
        valueEq := rfl
      }
  | _ => simp_all [FrozenPointwiseMatrixProgramFormula.erase]
termination_by sizeOf formula

/-- Sound inversion of decomposition through only transparent program boundaries. -/
def FrozenPointwiseMatrixProgramFormula.DenotesAt.exposeDecompose
    {samplers : Mxx.MxxSamplerFamily}
    {program : Mxx.Ir.Prog}
    {current : ExecutedScope samplers program}
    {frame : FormulaExecutionFrame samplers program current}
    {formula : FrozenPointwiseMatrixProgramFormula}
    {matrix : Mxx.Matrix}
    {matrixType : MatrixTypeExpr}
    {base digitCount : Mxx.Ir.IntExpr}
    {inputErase : FrozenPointwiseMatrixFormula}
    (denotes : formula.DenotesAt frame matrix)
    (erases : formula.erase = .decompose matrixType base digitCount inputErase) :
    FrozenPointwiseMatrixProgramFormula.DecomposeWitness (samplers := samplers)
      (program := program) matrixType base digitCount inputErase matrix := by
  cases denotes with
  | inputSubstitutionSubgraph parentDenotes =>
      exact FrozenPointwiseMatrixProgramFormula.DenotesAt.exposeDecompose parentDenotes erases
  | inputSubstitutionParallel parentDenotes =>
      exact FrozenPointwiseMatrixProgramFormula.DenotesAt.exposeDecompose parentDenotes erases
  | scaleOne inputDenotes =>
      exact FrozenPointwiseMatrixProgramFormula.DenotesAt.exposeDecompose inputDenotes erases
  | subgraphCall _ _ _ _ _ _ _ _ outputDenotes =>
      exact FrozenPointwiseMatrixProgramFormula.DenotesAt.exposeDecompose outputDenotes erases
  | parallelLoop _ _ _ _ _ _ _ _ outputDenotes =>
      exact FrozenPointwiseMatrixProgramFormula.DenotesAt.exposeDecompose outputDenotes erases
  | decompose _ _ _ _ _ _ _ _ _ _ _ inputDenotes _ _ _ decompositionRelation =>
      simp only [FrozenPointwiseMatrixProgramFormula.erase,
        FrozenPointwiseMatrixFormula.decompose.injEq] at erases
      rcases erases with ⟨rfl, rfl, rfl, inputErases⟩
      exact {
        inputFormula := _
        inputValue := _
        matrixParams := _
        baseValue := _
        digitCountValue := _
        inputErases
        inputDenotes := ⟨_, _, inputDenotes⟩
        decompositionRelation
      }
  | _ => simp_all [FrozenPointwiseMatrixProgramFormula.erase]
termination_by sizeOf formula

/-- Exact retained candidates for a public-key gate role.  The decomposition shape is certified
on their erased arithmetic view, while all runtime semantics continue to mention the exact
program formulas. -/
structure CheckedPublicKeyGateProgramFormula
    {erasedFormulas : List FrozenPointwiseMatrixFormula}
    (role : CheckedPublicKeyGateFormula erasedFormulas)
    (programFormulas : List FrozenPointwiseMatrixProgramFormula) where
  erases : programFormulas.map FrozenPointwiseMatrixProgramFormula.erase = erasedFormulas
  programs : CheckedSixWayMatrixProgramSkeleton programFormulas role.skeleton

theorem CheckedPublicKeyGateProgramFormula.ofErasedList
    {erasedFormulas : List FrozenPointwiseMatrixFormula}
    (role : CheckedPublicKeyGateFormula erasedFormulas)
    (programFormulas : List FrozenPointwiseMatrixProgramFormula)
    (erases : programFormulas.map FrozenPointwiseMatrixProgramFormula.erase = erasedFormulas) :
    Nonempty (CheckedPublicKeyGateProgramFormula role programFormulas) := by
  have skeletonErases : programFormulas.map FrozenPointwiseMatrixProgramFormula.erase =
      role.skeleton.formulas :=
    erases.trans (role.skeleton.formulas_eq_input role.skeletonFound).symm
  have lifted := CheckedSixWayMatrixProgramSkeleton.ofErasedList programFormulas role.skeleton
    skeletonErases
  exact lifted.map fun programs => { erases, programs }

/-- Exact retained candidates for an ordinary plaintext matrix gate role. -/
structure CheckedPlaintextGateProgramFormula
    {erasedFormulas : List FrozenPointwiseMatrixFormula}
    (role : CheckedPlaintextGateFormula erasedFormulas)
    (programFormulas : List FrozenPointwiseMatrixProgramFormula) where
  erases : programFormulas.map FrozenPointwiseMatrixProgramFormula.erase = erasedFormulas
  programs : CheckedSixWayMatrixProgramSkeleton programFormulas role.skeleton

theorem CheckedPlaintextGateProgramFormula.ofErasedList
    {erasedFormulas : List FrozenPointwiseMatrixFormula}
    (role : CheckedPlaintextGateFormula erasedFormulas)
    (programFormulas : List FrozenPointwiseMatrixProgramFormula)
    (erases : programFormulas.map FrozenPointwiseMatrixProgramFormula.erase = erasedFormulas) :
    Nonempty (CheckedPlaintextGateProgramFormula role programFormulas) := by
  have skeletonErases : programFormulas.map FrozenPointwiseMatrixProgramFormula.erase =
      role.skeleton.formulas :=
    erases.trans (role.skeleton.formulas_eq_input role.skeletonFound).symm
  have lifted := CheckedSixWayMatrixProgramSkeleton.ofErasedList programFormulas role.skeleton
    skeletonErases
  exact lifted.map fun programs => { erases, programs }

/-- Frame-indexed semantic evidence for the exact public-key `AND` candidate selected by the
checked six-way role. -/
structure CheckedPublicKeyGateProgramFormula.AndSemanticAt
    {erasedFormulas : List FrozenPointwiseMatrixFormula}
    {role : CheckedPublicKeyGateFormula erasedFormulas}
    {programFormulas : List FrozenPointwiseMatrixProgramFormula}
    (checked : CheckedPublicKeyGateProgramFormula role programFormulas)
    {samplers : Mxx.MxxSamplerFamily}
    {program : Mxx.Ir.Prog}
    {current : ExecutedScope samplers program}
    (frame : FormulaExecutionFrame samplers program current)
    (q ringDimension rows columns : Nat)
    [Fact (1 < q)] [NeZero ringDimension]
    (runtimeValue : Mxx.Matrix) : Type where
  result : checked.programs.andProgram.SemanticResultAt
    frame q ringDimension rows columns runtimeValue

/-- Frame-indexed semantic evidence for the exact plaintext `AND` candidate selected by the
checked six-way role. -/
structure CheckedPlaintextGateProgramFormula.AndSemanticAt
    {erasedFormulas : List FrozenPointwiseMatrixFormula}
    {role : CheckedPlaintextGateFormula erasedFormulas}
    {programFormulas : List FrozenPointwiseMatrixProgramFormula}
    (checked : CheckedPlaintextGateProgramFormula role programFormulas)
    {samplers : Mxx.MxxSamplerFamily}
    {program : Mxx.Ir.Prog}
    {current : ExecutedScope samplers program}
    (frame : FormulaExecutionFrame samplers program current)
    (q ringDimension rows columns : Nat)
    [Fact (1 < q)] [NeZero ringDimension]
    (runtimeValue : Mxx.Matrix) : Type where
  result : checked.programs.andProgram.SemanticResultAt
    frame q ringDimension rows columns runtimeValue

/-- Arithmetic content extracted from a checked public-key `AND` denotation. -/
structure CheckedPublicKeyGateProgramFormula.AndArithmeticWitness
    {erasedFormulas : List FrozenPointwiseMatrixFormula}
    {role : CheckedPublicKeyGateFormula erasedFormulas}
    {programFormulas : List FrozenPointwiseMatrixProgramFormula}
    (checked : CheckedPublicKeyGateProgramFormula role programFormulas)
    {samplers : Mxx.MxxSamplerFamily}
    {program : Mxx.Ir.Prog}
    (matrix : Mxx.Matrix) : Type where
  multiplication : FrozenPointwiseMatrixProgramFormula.MultiplyWitness
    (samplers := samplers) (program := program)
    role.skeleton.leftFormula
    (.decompose role.decompositionType role.base role.digitCount role.skeleton.rightFormula)
    matrix
  decomposition : FrozenPointwiseMatrixProgramFormula.DecomposeWitness
    (samplers := samplers) (program := program)
    role.decompositionType role.base role.digitCount role.skeleton.rightFormula
    multiplication.rightValue

/-- Extract the public-key multiplication and its exact quotient-ring decomposition relation from
the exact normalized denotation. -/
def CheckedPublicKeyGateProgramFormula.AndSemanticAt.arithmeticWitness
    {erasedFormulas : List FrozenPointwiseMatrixFormula}
    {role : CheckedPublicKeyGateFormula erasedFormulas}
    {programFormulas : List FrozenPointwiseMatrixProgramFormula}
    {checked : CheckedPublicKeyGateProgramFormula role programFormulas}
    {samplers : Mxx.MxxSamplerFamily}
    {program : Mxx.Ir.Prog}
    {current : ExecutedScope samplers program}
    {frame : FormulaExecutionFrame samplers program current}
    {q ringDimension rows columns : Nat}
    [Fact (1 < q)] [NeZero ringDimension]
    {runtimeValue : Mxx.Matrix}
    (semantic : checked.AndSemanticAt frame q ringDimension rows columns runtimeValue) :
    checked.AndArithmeticWitness (samplers := samplers) (program := program)
      semantic.result.normalizedValue := by
  let multiplication :=
    FrozenPointwiseMatrixProgramFormula.DenotesAt.exposeMultiply
      semantic.result.normalizedDenotes (checked.programs.andErases.trans role.andMatches)
  let decomposition :=
    FrozenPointwiseMatrixProgramFormula.DenotesAt.exposeDecompose
      multiplication.rightDenotes.denotes multiplication.rightErases
  exact { multiplication, decomposition }

/-- Arithmetic content extracted from a checked plaintext `AND` denotation. -/
structure CheckedPlaintextGateProgramFormula.AndArithmeticWitness
    {erasedFormulas : List FrozenPointwiseMatrixFormula}
    {role : CheckedPlaintextGateFormula erasedFormulas}
    {programFormulas : List FrozenPointwiseMatrixProgramFormula}
    (checked : CheckedPlaintextGateProgramFormula role programFormulas)
    {samplers : Mxx.MxxSamplerFamily}
    {program : Mxx.Ir.Prog}
    (matrix : Mxx.Matrix) : Type where
  multiplication : FrozenPointwiseMatrixProgramFormula.MultiplyWitness
    (samplers := samplers) (program := program)
    role.skeleton.leftFormula role.skeleton.rightFormula matrix

/-- Extract ordinary matrix multiplication from the exact normalized plaintext denotation. -/
def CheckedPlaintextGateProgramFormula.AndSemanticAt.arithmeticWitness
    {erasedFormulas : List FrozenPointwiseMatrixFormula}
    {role : CheckedPlaintextGateFormula erasedFormulas}
    {programFormulas : List FrozenPointwiseMatrixProgramFormula}
    {checked : CheckedPlaintextGateProgramFormula role programFormulas}
    {samplers : Mxx.MxxSamplerFamily}
    {program : Mxx.Ir.Prog}
    {current : ExecutedScope samplers program}
    {frame : FormulaExecutionFrame samplers program current}
    {q ringDimension rows columns : Nat}
    [Fact (1 < q)] [NeZero ringDimension]
    {runtimeValue : Mxx.Matrix}
    (semantic : checked.AndSemanticAt frame q ringDimension rows columns runtimeValue) :
    checked.AndArithmeticWitness (samplers := samplers) (program := program)
      semantic.result.normalizedValue := {
  multiplication := FrozenPointwiseMatrixProgramFormula.DenotesAt.exposeMultiply
    semantic.result.normalizedDenotes (checked.programs.andErases.trans role.andMatches)
}

/-- The four exact program-level six-way candidate lists checked by one BGG gate coupling. -/
structure CheckedBggGateProgramCoupling
    {encryptionPublicKeyErased encodingVectorErased decryptionPublicKeyErased
      plaintextErased : List FrozenPointwiseMatrixFormula}
    (coupling : CheckedBggGateFormulaCoupling encryptionPublicKeyErased encodingVectorErased
      decryptionPublicKeyErased plaintextErased)
    (encryptionPublicKeyPrograms encodingVectorPrograms decryptionPublicKeyPrograms
      plaintextPrograms : List FrozenPointwiseMatrixProgramFormula) where
  encryptionPublicKey : CheckedPublicKeyGateProgramFormula coupling.encryptionPublicKey
    encryptionPublicKeyPrograms
  encodingVectorErases : encodingVectorPrograms.map
    FrozenPointwiseMatrixProgramFormula.erase = encodingVectorErased
  encodingVector : CheckedSixWayMatrixProgramSkeleton encodingVectorPrograms
    coupling.encodingVector
  decryptionPublicKey : CheckedPublicKeyGateProgramFormula coupling.decryptionPublicKey
    decryptionPublicKeyPrograms
  plaintext : CheckedPlaintextGateProgramFormula coupling.plaintext plaintextPrograms

theorem CheckedBggGateProgramCoupling.ofErasedLists
    {encryptionPublicKeyErased encodingVectorErased decryptionPublicKeyErased
      plaintextErased : List FrozenPointwiseMatrixFormula}
    (coupling : CheckedBggGateFormulaCoupling encryptionPublicKeyErased encodingVectorErased
      decryptionPublicKeyErased plaintextErased)
    (encryptionPublicKeyPrograms encodingVectorPrograms decryptionPublicKeyPrograms
      plaintextPrograms : List FrozenPointwiseMatrixProgramFormula)
    (encryptionErases : encryptionPublicKeyPrograms.map
      FrozenPointwiseMatrixProgramFormula.erase = encryptionPublicKeyErased)
    (vectorErases : encodingVectorPrograms.map FrozenPointwiseMatrixProgramFormula.erase =
      encodingVectorErased)
    (decryptionErases : decryptionPublicKeyPrograms.map
      FrozenPointwiseMatrixProgramFormula.erase = decryptionPublicKeyErased)
    (plaintextErases : plaintextPrograms.map FrozenPointwiseMatrixProgramFormula.erase =
      plaintextErased) :
    Nonempty (CheckedBggGateProgramCoupling coupling encryptionPublicKeyPrograms
      encodingVectorPrograms decryptionPublicKeyPrograms plaintextPrograms) := by
  have encryption := CheckedPublicKeyGateProgramFormula.ofErasedList
    coupling.encryptionPublicKey encryptionPublicKeyPrograms encryptionErases
  have vectorSkeletonErases : encodingVectorPrograms.map
      FrozenPointwiseMatrixProgramFormula.erase = coupling.encodingVector.formulas :=
    vectorErases.trans
      (coupling.encodingVector.formulas_eq_input coupling.encodingVectorFound).symm
  have vector := CheckedSixWayMatrixProgramSkeleton.ofErasedList encodingVectorPrograms
    coupling.encodingVector vectorSkeletonErases
  have decryption := CheckedPublicKeyGateProgramFormula.ofErasedList
    coupling.decryptionPublicKey decryptionPublicKeyPrograms decryptionErases
  have plaintext := CheckedPlaintextGateProgramFormula.ofErasedList
    coupling.plaintext plaintextPrograms plaintextErases
  rcases encryption with ⟨encryptionPublicKey⟩
  rcases vector with ⟨encodingVector⟩
  rcases decryption with ⟨decryptionPublicKey⟩
  rcases plaintext with ⟨plaintext⟩
  exact ⟨{
    encryptionPublicKey
    encodingVectorErases := vectorErases
    encodingVector
    decryptionPublicKey
    plaintext
  }⟩

/-- Frame-indexed semantic evidence for the exact encoding-vector `AND` candidate.  Its erased
formula is the checked `v_L * D(pk_R) + m_L * v_R` role, but its denotation retains every actual
program boundary. -/
structure CheckedBggGateProgramCoupling.VectorAndSemanticAt
    {encryptionPublicKeyErased encodingVectorErased decryptionPublicKeyErased
      plaintextErased : List FrozenPointwiseMatrixFormula}
    {coupling : CheckedBggGateFormulaCoupling encryptionPublicKeyErased encodingVectorErased
      decryptionPublicKeyErased plaintextErased}
    {encryptionPublicKeyPrograms encodingVectorPrograms decryptionPublicKeyPrograms
      plaintextPrograms : List FrozenPointwiseMatrixProgramFormula}
    (checked : CheckedBggGateProgramCoupling coupling encryptionPublicKeyPrograms
      encodingVectorPrograms decryptionPublicKeyPrograms plaintextPrograms)
    {samplers : Mxx.MxxSamplerFamily}
    {program : Mxx.Ir.Prog}
    {current : ExecutedScope samplers program}
    (frame : FormulaExecutionFrame samplers program current)
    (q ringDimension rows columns : Nat)
    [Fact (1 < q)] [NeZero ringDimension]
    (runtimeValue : Mxx.Matrix) : Type where
  result : checked.encodingVector.andProgram.SemanticResultAt
    frame q ringDimension rows columns runtimeValue

/-- Arithmetic content of the checked vector `AND`: `v_L * D(pk_R) + v_R * m_L`, including the
exact decomposition relation produced by the executable decomposition node. -/
structure CheckedBggGateProgramCoupling.VectorAndArithmeticWitness
    {encryptionPublicKeyErased encodingVectorErased decryptionPublicKeyErased
      plaintextErased : List FrozenPointwiseMatrixFormula}
    {coupling : CheckedBggGateFormulaCoupling encryptionPublicKeyErased encodingVectorErased
      decryptionPublicKeyErased plaintextErased}
    {encryptionPublicKeyPrograms encodingVectorPrograms decryptionPublicKeyPrograms
      plaintextPrograms : List FrozenPointwiseMatrixProgramFormula}
    (checked : CheckedBggGateProgramCoupling coupling encryptionPublicKeyPrograms
      encodingVectorPrograms decryptionPublicKeyPrograms plaintextPrograms)
    {samplers : Mxx.MxxSamplerFamily}
    {program : Mxx.Ir.Prog}
    (matrix : Mxx.Matrix) : Type where
  addition : FrozenPointwiseMatrixProgramFormula.AddWitness
    (samplers := samplers) (program := program)
    (.multiply coupling.encodingVector.leftFormula
      (.decompose coupling.decryptionPublicKey.decompositionType
        coupling.decryptionPublicKey.base coupling.decryptionPublicKey.digitCount
        coupling.decryptionPublicKey.skeleton.rightFormula))
    (.multiply coupling.encodingVector.rightFormula coupling.plaintext.skeleton.leftFormula)
    matrix
  encodingProduct : FrozenPointwiseMatrixProgramFormula.MultiplyWitness
    (samplers := samplers) (program := program)
    coupling.encodingVector.leftFormula
    (.decompose coupling.decryptionPublicKey.decompositionType
      coupling.decryptionPublicKey.base coupling.decryptionPublicKey.digitCount
      coupling.decryptionPublicKey.skeleton.rightFormula)
    addition.leftValue
  plaintextProduct : FrozenPointwiseMatrixProgramFormula.MultiplyWitness
    (samplers := samplers) (program := program)
    coupling.encodingVector.rightFormula coupling.plaintext.skeleton.leftFormula
    addition.rightValue
  decomposition : FrozenPointwiseMatrixProgramFormula.DecomposeWitness
    (samplers := samplers) (program := program)
    coupling.decryptionPublicKey.decompositionType coupling.decryptionPublicKey.base
    coupling.decryptionPublicKey.digitCount
    coupling.decryptionPublicKey.skeleton.rightFormula encodingProduct.rightValue

/-- Extract the exact vector BGG arithmetic witnesses from its normalized frame-indexed
denotation. -/
def CheckedBggGateProgramCoupling.VectorAndSemanticAt.arithmeticWitness
    {encryptionPublicKeyErased encodingVectorErased decryptionPublicKeyErased
      plaintextErased : List FrozenPointwiseMatrixFormula}
    {coupling : CheckedBggGateFormulaCoupling encryptionPublicKeyErased encodingVectorErased
      decryptionPublicKeyErased plaintextErased}
    {encryptionPublicKeyPrograms encodingVectorPrograms decryptionPublicKeyPrograms
      plaintextPrograms : List FrozenPointwiseMatrixProgramFormula}
    {checked : CheckedBggGateProgramCoupling coupling encryptionPublicKeyPrograms
      encodingVectorPrograms decryptionPublicKeyPrograms plaintextPrograms}
    {samplers : Mxx.MxxSamplerFamily}
    {program : Mxx.Ir.Prog}
    {current : ExecutedScope samplers program}
    {frame : FormulaExecutionFrame samplers program current}
    {q ringDimension rows columns : Nat}
    [Fact (1 < q)] [NeZero ringDimension]
    {runtimeValue : Mxx.Matrix}
    (semantic : checked.VectorAndSemanticAt frame q ringDimension rows columns runtimeValue) :
    checked.VectorAndArithmeticWitness (samplers := samplers) (program := program)
      semantic.result.normalizedValue := by
  let addition := FrozenPointwiseMatrixProgramFormula.DenotesAt.exposeAdd
    semantic.result.normalizedDenotes
    (checked.encodingVector.andErases.trans coupling.vectorAndMatches)
  let encodingProduct := FrozenPointwiseMatrixProgramFormula.DenotesAt.exposeMultiply
    addition.leftDenotes.denotes addition.leftErases
  let plaintextProduct := FrozenPointwiseMatrixProgramFormula.DenotesAt.exposeMultiply
    addition.rightDenotes.denotes addition.rightErases
  let decomposition := FrozenPointwiseMatrixProgramFormula.DenotesAt.exposeDecompose
    encodingProduct.rightDenotes.denotes encodingProduct.rightErases
  exact { addition, encodingProduct, plaintextProduct, decomposition }

/-- Recover the exact program-level four-lane coupling directly from the static three-trace
matcher.  All four erasure equalities are fields of the matched lanes; theorem callers do not
supply formula lists or positional correspondence. -/
theorem CheckedBggThreeTraceInterface.programGateCoupling
    {bundle : ClosedProtocolBundle}
    (checked : CheckedBggThreeTraceInterface bundle) :
    Nonempty (CheckedBggGateProgramCoupling checked.gateFormulaCoupling
      checked.encryptionLaneControl.lane.gateCandidateProgramFormulas
      checked.encodingVectorLane.binding.lane.gateCandidateProgramFormulas
      checked.decryptionPublicKeyLane.binding.lane.gateCandidateProgramFormulas
      checked.plaintextLane.binding.lane.gateCandidateProgramFormulas) :=
  CheckedBggGateProgramCoupling.ofErasedLists checked.gateFormulaCoupling
    checked.encryptionLaneControl.lane.gateCandidateProgramFormulas
    checked.encodingVectorLane.binding.lane.gateCandidateProgramFormulas
    checked.decryptionPublicKeyLane.binding.lane.gateCandidateProgramFormulas
    checked.plaintextLane.binding.lane.gateCandidateProgramFormulas
    checked.encryptionLaneControl.lane.gateCandidateFormulasMatch.symm
    checked.encodingVectorLane.binding.lane.gateCandidateFormulasMatch.symm
    checked.decryptionPublicKeyLane.binding.lane.gateCandidateFormulasMatch.symm
    checked.plaintextLane.binding.lane.gateCandidateFormulasMatch.symm

end Mxx.Certificate
