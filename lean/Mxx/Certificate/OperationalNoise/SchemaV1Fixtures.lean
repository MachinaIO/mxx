import Mxx.Certificate.OperationalNoise.SchemaV1
import Lean.Elab.Command

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.SchemaV1

open Lean

/-! This decoder intentionally covers the constructors used by the committed small Rust v1
    golden. It demonstrates the JSON-to-DTO boundary only. It is not a production decoder and it
    does not invoke semantic validation or acceptance. -/

private def field (json : Json) (name : String) : Except String Json :=
  json.getObjVal? name

private def stringField (json : Json) (name : String) : Except String String :=
  do
  (← field json name).getStr?

private def natField (json : Json) (name : String) : Except String Nat :=
  do
  (← field json name).getNat?

private def boolField (json : Json) (name : String) : Except String Bool :=
  do
  (← field json name).getBool?

private def decodeList {α : Type} (decode : Json → Except String α) (json : Json) :
    Except String (List α) := do
  (← json.getArr?).toList.mapM decode

private def decodeOption {α : Type} (decode : Json → Except String α) (json : Json) :
    Except String (Option α) :=
  if json.isNull then pure none else some <$> decode json

private def decodeExpressionRef (json : Json) : Except String ExpressionRef :=
  do
  pure ⟨← natField json "row"⟩

private def decodeProgramRef (json : Json) : Except String ProgramRef :=
  do
  pure ⟨← natField json "row"⟩

private def decodeSourceRef (json : Json) : Except String SourceRef :=
  do
  pure ⟨← natField json "source"⟩

private def decodeEventRef (json : Json) : Except String EventRef :=
  do
  pure ⟨← natField json "row"⟩

private def decodeRange (json : Json) : Except String Range :=
  do
  pure { minimum := ← natField json "minimum"
         maximumExclusive := ← natField json "maximum_exclusive" }

private def decodeValueType (json : Json) : Except String ValueType := do
  match ← stringField json "kind" with
  | "bool" => pure .bool
  | "int" => pure .int
  | "real" => pure .real
  | "bytes" => pure .bytes
  | "matrix" =>
      pure <| .matrix (← stringField json "modulus") (← natField json "ring_dimension")
        (← natField json "rows") (← natField json "columns")
  | "trapdoor" => pure .trapdoor
  | kind => throw s!"unsupported v1 value type {kind}"

private def decodeConstantValue (json : Json) : Except String ConstantValue := do
  match ← stringField json "kind" with
  | "bool" => pure <| .bool (← boolField json "value")
  | "int" => pure <| .int (← stringField json "value")
  | "real" => pure <| .real (← stringField json "value")
  | "bytes" => pure <| .bytes (← decodeList Json.getNat? (← field json "value"))
  | kind => throw s!"unsupported v1 constant value {kind}"

private def decodeConstant (json : Json) : Except String Constant :=
  do
  pure { valueType := ← decodeValueType (← field json "value_type")
         value := ← decodeConstantValue (← field json "value") }

private def decodeScope (json : Json) : Except String Scope := do
  match ← stringField json "kind" with
  | "root" => pure .root
  | "subgraph" => pure <| .subgraph (← stringField json "canonical_name")
  | "parallel_body" =>
      throw "small v1 fixture does not contain a parallel-body scope"
  | "sequential_body" =>
      throw "small v1 fixture does not contain a sequential-body scope"
  | kind => throw s!"unsupported v1 scope {kind}"

private def decodeObservedWire (json : Json) : Except String ObservedWire :=
  do
  pure { stage := ← stringField json "stage"
         definition := ← decodeScope (← field json "definition")
         path := ← natField json "path"
         node := ← natField json "node"
         port := ← natField json "port" }

private def decodeSignedRange (json : Json) : Except String SignedRange :=
  do
  pure { minimum := ← stringField json "minimum"
         maxExclusive := ← stringField json "maxExclusive" }

private def decodeRawCoefficientClass (json : Json) : Except String RawCoefficientClass := do
  match ← stringField json "kind" with
  | "exact_zero" => pure .exactZero
  | "finite" => pure <| .finite (← stringField json "maximumAbsoluteCoefficient")
  | "large" => pure .large
  | kind => throw s!"unsupported v1 raw coefficient class {kind}"

private def decodeRawValueContract (json : Json) : Except String RawValueContract :=
  do
  pure { signedRange := ← decodeOption decodeSignedRange (← field json "signedRange")
         coefficientClass :=
           ← decodeOption decodeRawCoefficientClass (← field json "coefficientClass")
         canonicalCoefficientExclusiveUpper :=
           ← decodeOption Json.getStr? (← field json "canonicalCoefficientExclusiveUpper")
         polynomialSupportUpper :=
           ← decodeOption Json.getNat? (← field json "polynomialSupportUpper") }

private def decodeSampleDescriptor (json : Json) : Except String SampleDescriptor :=
  do
  pure { definition := ← stringField json "definition"
         parameters := ← decodeList Json.getNat? (← field json "parameters")
         outputType := ← decodeValueType (← field json "output_type")
         gadgetBase := ← decodeOption Json.getStr? (← field json "gadget_base")
         digitCount := ← decodeOption Json.getNat? (← field json "digit_count")
         decomposition := ← decodeOption Json.getStr? (← field json "decomposition") }

private def decodeHashVariant (json : Json) : Except String HashVariant := do
  match ← json.getStr? with
  | "plain" => pure .plain
  | "decomposed" => pure .decomposed
  | "small_decomposed" => pure .smallDecomposed
  | kind => throw s!"unsupported v1 hash variant {kind}"

private def decodeExpressionRefs (json : Json) : Except String (List ExpressionRef) :=
  decodeList (fun value => do pure ⟨← value.getNat?⟩) json

private def decodeSamplerOperation (json : Json) : Except String SamplerOperation := do
  let output ← decodeValueType (← field json "output")
  match ← stringField json "kind" with
  | "uniform_residue" => pure <| .uniformResidue output
  | "uniform_interval" =>
      pure <| .uniformInterval output (← stringField json "minimum")
        (← stringField json "maximum")
  | "gaussian" =>
      pure <| .gaussian output (← stringField json "sigma")
        (← stringField json "max_coefficient_bound")
  | "hash" =>
      pure <| .hash output (← decodeHashVariant (← field json "variant"))
        (← decodeList Json.getNat? (← field json "tag_prefix"))
        (← decodeExpressionRefs (← field json "tag_expressions"))
        (← decodeExpressionRefs (← field json "tag_decimal_expressions"))
        (← decodeExpressionRefs (← field json "tag_u64_le_expressions"))
        (← decodeOption Json.getNat? (← field json "base"))
        (← decodeOption Json.getNat? (← field json "digit_count"))
  | "trapdoor" =>
      pure <| .trapdoor output (← stringField json "sigma") (← natField json "gadget_base")
        (← natField json "digit_count") (← stringField json "preimage_max_coefficient_bound")
  | "preimage" => pure <| .preimage output (← stringField json "max_coefficient_bound")
  | kind => throw s!"unsupported v1 sampler operation {kind}"

private def decodeStatementScope (json : Json) : Except String StatementScope := do
  match ← stringField json "kind" with
  | "closed" => pure <| .closed ⟨← natField json "root"⟩
  | "program" => pure <| .program ⟨← natField json "program"⟩
  | kind => throw s!"unsupported v1 statement scope {kind}"

private def decodeEventRow (json : Json) : Except String EventRow := do
  match ← stringField json "kind" with
  | "sample" =>
      pure <| .sample (← decodeObservedWire (← field json "owner"))
        (← decodeSampleDescriptor (← field json "descriptor"))
        (← decodeOption decodeRawValueContract (← field json "contract"))
  | "sampler" =>
      pure <| .sampler (← decodeObservedWire (← field json "owner"))
        (← decodeSamplerOperation (← field json "operation"))
        (← decodeOption decodeRawValueContract (← field json "contract"))
  | "gadget_decompose" =>
      pure <| .gadgetDecompose (← decodeStatementScope (← field json "scope"))
        ⟨← natField json "expression"⟩ (← decodeValueType (← field json "output"))
        (← natField json "base") (← boolField json "small") (← natField json "digit_count")
        ⟨← natField json "input"⟩
        (← decodeOption decodeRawValueContract (← field json "contract"))
  | kind => throw s!"unsupported v1 event row {kind}"

private def decodeSourceRow (json : Json) : Except String SourceRow := do
  match ← stringField json "kind" with
  | "constant" => pure <| .constant (← decodeConstant (← field json "value"))
  | kind => throw s!"small v1 fixture does not contain source row {kind}"

private def decodeExpressionSource (json : Json) : Except String ExpressionSource := do
  match ← stringField json "kind" with
  | "direct" => pure <| .direct (← decodeSourceRef json)
  | "family" =>
      pure <| .family (← decodeSourceRef json) ⟨← natField json "selector"⟩
  | kind => throw s!"unsupported v1 expression source {kind}"

private def decodeExpressionEventOperator (json : Json) : Except String ExpressionEventOperator :=
  do
    match ← stringField json "kind" with
    | "sample" => pure <| .sample (← decodeEventRef (← field json "event"))
    | "sampler" => pure <| .sampler (← decodeEventRef (← field json "event"))
    | "gadget_decompose" =>
        pure <| .gadgetDecompose (← decodeList decodeEventRef (← field json "events"))
    | kind => throw s!"unsupported v1 expression event operator {kind}"

private def decodeMatrixOperation (json : Json) : Except String MatrixOperation := do
  match ← stringField json "kind" with
  | "add" => pure .add
  | "subtract" => pure .subtract
  | "multiply" => pure .multiply
  | "negate" => pure .negate
  | "scale" => pure .scale
  | "transpose" => pure .transpose
  | kind => throw s!"small v1 fixture does not contain matrix operation {kind}"

private def decodeTrapdoorOperation (json : Json) : Except String TrapdoorOperation := do
  match ← stringField json "kind" with
  | "generate" =>
      pure <| .generate (← stringField json "descriptor")
        (← decodeList Json.getNat? (← field json "parameters"))
        (← decodeOption decodeEventRef (← field json "paired_public_event"))
        (← stringField json "paired_public_output_role")
  | kind => throw s!"small v1 fixture does not contain trapdoor operation {kind}"

private def decodeStableOperator (json : Json) : Except String StableOperator := do
  match ← stringField json "kind" with
  | "program_call" => pure .programCall
  | "matrix" => pure <| .matrix (← decodeMatrixOperation (← field json "operation"))
  | "trapdoor" => pure <| .trapdoor (← decodeTrapdoorOperation (← field json "operation"))
  | kind => throw s!"small v1 fixture does not contain stable operator {kind}"

private def decodeExpressionOperator (json : Json) : Except String ExpressionOperator := do
  match ← stringField json "kind" with
  | "sample" | "sampler" | "gadget_decompose" =>
      pure <| .event (← decodeExpressionEventOperator json)
  | _ => pure <| .stable (← decodeStableOperator json)

private def decodeExpressionDescriptor (json : Json) : Except String ExpressionDescriptor := do
  match ← stringField json "kind" with
  | "source" => pure <| .source (← decodeExpressionSource (← field json "source"))
  | "event" => pure <| .event (← decodeExpressionEventOperator (← field json "operator"))
  | "operation" =>
      pure <| .operation (← decodeExpressionOperator (← field json "operator"))
        (← decodeValueType (← field json "value_type"))
  | kind => throw s!"unsupported v1 expression descriptor {kind}"

private def decodeExpressionRow (json : Json) : Except String ExpressionRow :=
  do
  pure { descriptor := ← decodeExpressionDescriptor (← field json "descriptor")
         inputs := ← decodeList (fun value => do pure ⟨← value.getNat?⟩)
           (← field json "inputs")
         program := ← decodeOption (fun value => do pure ⟨← value.getNat?⟩)
           (← field json "program") }

private def decodeProgramInput (json : Json) : Except String ProgramInput :=
  do
  pure { valueType := ← decodeValueType (← field json "value_type")
         trustedIndexRange := ← decodeOption decodeRange (← field json "trusted_index_range") }

private def decodeFamily (json : Json) : Except String Family := do
  let artifact ← field json "artifact"
  if !artifact.isNull then
    throw "small v1 fixture does not contain a family artifact"
  pure { domain := ← decodeRange (← field json "domain")
         elementType := ← decodeValueType (← field json "element_type")
         reducible := ← boolField json "reducible"
         artifact := none }

private def decodeProgramRow (json : Json) : Except String ProgramRow :=
  do
  pure { signature := ← decodeList decodeProgramInput (← field json "signature")
         output := ← decodeValueType (← field json "output")
         family := ← decodeOption decodeFamily (← field json "family")
         root := ⟨← natField json "root"⟩ }

private def decodeResidualRoot (json : Json) : Except String ResidualRoot := do
  match ← stringField json "kind" with
  | "closed" => pure <| .closed ⟨← natField json "expression"⟩
  | "family" =>
      pure <| .family ⟨← natField json "program"⟩ (← decodeRange (← field json "domain"))
  | kind => throw s!"unsupported v1 residual root {kind}"

private def requireEmptyRows {α : Type} (label : String) (json : Json) :
    Except String (List α) := do
  if (← json.getArr?).isEmpty then pure [] else throw s!"small v1 fixture has nonempty {label}"

def decodeSmallDocument (json : Json) : Except String Document :=
  do
  pure { schemaId := ← stringField json "schemaId"
         schemaVersion := ← natField json "schemaVersion"
         plaintextModulus := ← stringField json "plaintextModulus"
         ciphertextModulus := ← stringField json "ciphertextModulus"
         ringDimension := ← natField json "ringDimension"
         expressions := ← decodeList decodeExpressionRow (← field json "expressions")
         programs := ← decodeList decodeProgramRow (← field json "programs")
         sources := ← decodeList decodeSourceRow (← field json "sources")
         events := ← decodeList decodeEventRow (← field json "events")
         indexUses := ← requireEmptyRows "indexUses" (← field json "indexUses")
         sliceGroups := ← requireEmptyRows "sliceGroups" (← field json "sliceGroups")
         residualRoot := ← decodeResidualRoot (← field json "residualRoot") }

def rustV1Golden : String :=
  include_str "../../../../crates/correctness/testdata/operational-noise-certificate-v1.json"

def decodedRustV1Golden : Except String Document :=
  Json.parse rustV1Golden >>= decodeSmallDocument

def expectedRawContract : RawValueContract :=
  { signedRange := some { minimum := "-3", maxExclusive := "5" }
    coefficientClass := some (.finite "7")
    canonicalCoefficientExclusiveUpper := some "257"
    polynomialSupportUpper := some 2 }

def decodedRawContract : Except String RawValueContract :=
  Json.parse
      "{\"signedRange\":{\"minimum\":\"-3\",\"maxExclusive\":\"5\"},\
      \"coefficientClass\":{\"kind\":\"finite\",\"maximumAbsoluteCoefficient\":\"7\"},\
      \"canonicalCoefficientExclusiveUpper\":\"257\",\"polynomialSupportUpper\":2}" >>=
    decodeRawValueContract

private def goldenMatrix (rows columns : Nat) : ValueType :=
  .matrix "257" 1 rows columns

private def goldenOwner (node : Nat) : ObservedWire :=
  { stage := "consumer", definition := .root, path := 0, node, port := 0 }

def expectedDocument : Document :=
  { schemaId := "mxx.operational-noise.certificate"
    schemaVersion := 1
    plaintextModulus := "2"
    ciphertextModulus := "257"
    ringDimension := 1
    expressions :=
      [ { descriptor := .operation (.event (.sampler ⟨0⟩)) (goldenMatrix 1 4)
          inputs := [], program := none },
        { descriptor := .operation (.event (.sampler ⟨1⟩)) (goldenMatrix 1 1)
          inputs := [], program := none },
        { descriptor := .operation (.event (.sampler ⟨2⟩)) (goldenMatrix 4 1)
          inputs := [], program := none },
        { descriptor :=
            .operation
              (.stable
                (.trapdoor
                  (.generate "trapdoor-sample" [4, 2] (some ⟨0⟩) "value")))
              .trapdoor
          inputs := [], program := none },
        { descriptor := .source (.direct ⟨0⟩), inputs := [], program := none },
        { descriptor := .source (.direct ⟨1⟩), inputs := [], program := none },
        { descriptor := .operation (.stable .programCall) (goldenMatrix 4 1)
          inputs := [⟨4⟩], program := some ⟨0⟩ },
        { descriptor := .operation (.stable (.matrix .multiply)) (goldenMatrix 1 1)
          inputs := [⟨0⟩, ⟨6⟩], program := none },
        { descriptor := .operation (.stable (.matrix .scale)) (goldenMatrix 4 1)
          inputs := [⟨6⟩, ⟨5⟩], program := none },
        { descriptor := .operation (.stable (.matrix .multiply)) (goldenMatrix 1 1)
          inputs := [⟨0⟩, ⟨8⟩], program := none },
        { descriptor := .operation (.stable (.matrix .subtract)) (goldenMatrix 1 1)
          inputs := [⟨9⟩, ⟨1⟩], program := none } ]
    programs :=
      [ { signature := [ { valueType := .int
                           trustedIndexRange := some { minimum := 0, maximumExclusive := 1 } } ]
          output := goldenMatrix 4 1
          family :=
            some
              { domain := { minimum := 0, maximumExclusive := 1 }
                elementType := goldenMatrix 4 1
                reducible := false
                artifact := none }
          root := ⟨2⟩ } ]
    sources :=
      [ .constant { valueType := .int, value := .int "0" },
        .constant { valueType := .int, value := .int "1" } ]
    events :=
      [ .sampler (goldenOwner 0)
          (.trapdoor (goldenMatrix 1 4)
            "{\"tag\":\"Rational\",\"value\":{\"numerator\":\"3\",\"denominator\":\"1\"}}"
            4 2 "8") none,
        .sampler (goldenOwner 1) (.uniformResidue (goldenMatrix 1 1)) none,
        .sampler (goldenOwner 2) (.preimage (goldenMatrix 4 1) "8") none ]
    indexUses := []
    sliceGroups := []
    residualRoot := .closed ⟨10⟩ }

theorem rust_v1_golden_representation_is_typed :
    expectedDocument.schemaVersion = 1 ∧
      expectedDocument.expressions.length = 11 ∧
      expectedDocument.events.length = 3 ∧
      expectedDocument.residualRoot = .closed ⟨10⟩ := by
  exact ⟨rfl, rfl, rfl, rfl⟩

run_cmd
  match decodedRustV1Golden with
  | .error message => throwError "Rust v1 golden decode failed: {message}"
  | .ok document =>
      if document = expectedDocument then pure ()
      else throwError "Rust v1 golden document mismatch: {repr document}"

run_cmd
  match decodedRawContract with
  | .error message => throwError "raw v1 contract decode failed: {message}"
  | .ok contract =>
      if contract = expectedRawContract then pure ()
      else throwError "raw v1 contract mismatch: {repr contract}"

def typedGadgetOperator : ExpressionEventOperator :=
  .gadgetDecompose [⟨2⟩, ⟨5⟩]

def typedGadgetEvent : EventRow :=
  .gadgetDecompose (.program ⟨3⟩) ⟨7⟩ (.matrix "257" 1 4 1) 4 false 2 ⟨6⟩ none

theorem gadget_refs_and_scope_are_typed :
    typedGadgetOperator = .gadgetDecompose [⟨2⟩, ⟨5⟩] ∧
      typedGadgetEvent =
        .gadgetDecompose
          (.program ⟨3⟩) ⟨7⟩ (.matrix "257" 1 4 1) 4 false 2 ⟨6⟩ none := by
  exact ⟨rfl, rfl⟩

theorem raw_contract_fields_are_distinct :
    expectedRawContract.signedRange = some { minimum := "-3", maxExclusive := "5" } ∧
      expectedRawContract.coefficientClass = some (.finite "7") ∧
      expectedRawContract.canonicalCoefficientExclusiveUpper = some "257" ∧
      expectedRawContract.polynomialSupportUpper = some 2 := by
  exact ⟨rfl, rfl, rfl, rfl⟩

#print axioms rust_v1_golden_representation_is_typed
#print axioms gadget_refs_and_scope_are_typed
#print axioms raw_contract_fields_are_distinct

end Mxx.Certificate.OperationalNoise.SchemaV1
