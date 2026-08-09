import Mxx.Ir
import Mxx.Certificate.Derivation

namespace Mxx.Ir

def binaryFormatVersion : UInt8 := 1

inductive DecodeError where
  | truncated (offset : Nat)
  | wrongVersion (offset : Nat) (actual : UInt8)
  | wrongDocumentKind (offset : Nat) (expected actual : UInt8)
  | unknownTag (offset : Nat) (domain : String) (tag : UInt8)
  | invalidBoolean (offset : Nat) (actual : UInt8)
  | invalidLength (offset : Nat)
  | invalidInteger (offset : Nat)
  | invalidRational (offset : Nat)
  | invalidUtf8 (offset : Nat)
  | invalidHex (offset : Nat)
  | invalidStringIndex (offset index : Nat)
  | invalidWire (offset node port : Nat)
  | trailingBytes (offset : Nat)
  deriving BEq, DecidableEq, Repr

private structure Reader where
  bytes : ByteArray
  position : Nat
  limit : Nat
  strings : Array String := #[]

private abbrev DecodeM := StateT Reader (Except DecodeError)

private def failAt {α : Type} (error : DecodeError) : DecodeM α :=
  fun _ => .error error

private def readByte : DecodeM UInt8 := fun state =>
  if state.position < state.limit then
    match state.bytes[state.position]? with
    | some value => .ok (value, { state with position := state.position + 1 })
    | none => .error (.truncated state.position)
  else
    .error (.truncated state.position)

private def readU32 : DecodeM Nat := do
  let offset ← get
  let b0 ← readByte
  let b1 ← readByte
  let b2 ← readByte
  let b3 ← readByte
  let value := b0.toNat + b1.toNat * 256 + b2.toNat * 65536 + b3.toNat * 16777216
  if value ≤ UInt32.size - 1 then pure value else failAt (.invalidLength offset.position)

private def readBool : DecodeM Bool := do
  let offset := (← get).position
  let actual ← readByte
  match actual.toNat with
  | 0 => pure false
  | 1 => pure true
  | _ => failAt (.invalidBoolean offset actual)

private def readBytes (count : Nat) : DecodeM ByteArray := do
  let state ← get
  let endOffset := state.position + count
  if endOffset < state.position ∨ endOffset > state.limit then
    failAt (.truncated state.position)
  else
    set { state with position := endOffset }
    pure (state.bytes.extract state.position endOffset)

private def readString : DecodeM String := do
  let offset := (← get).position
  let index ← readU32
  match (← get).strings[index]? with
  | some value => pure value
  | none => failAt (.invalidStringIndex offset index)

private def readArray {α : Type} (read : DecodeM α) : DecodeM (List α) := do
  let count ← readU32
  (List.range count).mapM fun _ => read

private def readOption {α : Type} (read : DecodeM α) : DecodeM (Option α) := do
  let offset := (← get).position
  let actual ← readByte
  match actual.toNat with
  | 0 => pure none
  | 1 => some <$> read
  | _ => failAt (.unknownTag offset "Option" actual)

private def readRecord {α : Type} (_domain : String) (read : UInt8 → DecodeM α) : DecodeM α := do
  let tag ← readByte
  let lengthOffset := (← get).position
  let length ← readU32
  let state ← get
  let recordEnd := state.position + length
  if recordEnd < state.position ∨ recordEnd > state.limit then
    failAt (.invalidLength lengthOffset)
  let outerLimit := state.limit
  set { state with limit := recordEnd }
  let value ← read tag
  let final ← get
  if final.position != recordEnd then
    failAt (.trailingBytes final.position)
  set { final with limit := outerLimit }
  pure value

private def readInt : DecodeM Int := do
  let offset := (← get).position
  let length ← readU32
  if length = 0 then failAt (.invalidInteger offset)
  let bytes ← readBytes length
  let last ← match bytes[length - 1]? with
    | some value => pure value
    | none => failAt (.invalidInteger offset)
  if length > 1 then
    let previous ← match bytes[length - 2]? with
      | some value => pure value
      | none => failAt (.invalidInteger offset)
    if (last = 0 && previous.toNat < 128) || (last = 255 && previous.toNat ≥ 128) then
      failAt (.invalidInteger offset)
  let magnitude ← (List.range length).foldlM (init := 0) fun value index =>
    match bytes[index]? with
    | some byte => pure (value + byte.toNat * 2 ^ (8 * index))
    | none => failAt (.invalidInteger offset)
  if last.toNat < 128 then
    pure (Int.ofNat magnitude)
  else
    pure (Int.ofNat magnitude - Int.ofNat (2 ^ (8 * length)))

private def readIntExpr : Nat → DecodeM IntExpr
  | 0 => do failAt (.invalidLength (← get).position)
  | fuel + 1 => readRecord "IntExpr" fun tag =>
      match tag.toNat with
      | 0 => .constant <$> readInt
      | 1 => .parameter <$> readString
      | 2 => .loopIndex <$> readU32
      | 3 => .add <$> readIntExpr fuel <*> readIntExpr fuel
      | 4 => .subtract <$> readIntExpr fuel <*> readIntExpr fuel
      | 5 => .multiply <$> readIntExpr fuel <*> readIntExpr fuel
      | 6 => .divide <$> readIntExpr fuel <*> readIntExpr fuel
      | 7 => .roundDivide <$> readIntExpr fuel <*> readIntExpr fuel
      | 8 => .log2Ceil <$> readIntExpr fuel
      | _ => do failAt (.unknownTag ((← get).position - 5) "IntExpr" tag)

private def readRat : DecodeM Rat := do
  let offset := (← get).position
  let numerator ← readInt
  let denominator ← readInt
  if h : denominator ≤ 0 then failAt (.invalidRational offset)
  else
    let denominatorNat := denominator.toNat
    if Int.gcd numerator denominator != 1 then failAt (.invalidRational offset)
    have denominatorNatNonzero : denominatorNat ≠ 0 := by
      intro isZero
      have nonpositive := Int.toNat_eq_zero.mp isZero
      exact h nonpositive
    pure (Rat.normalize numerator denominatorNat denominatorNatNonzero)

private def readRealExpr : Nat → DecodeM RealExpr
  | 0 => do failAt (.invalidLength (← get).position)
  | fuel + 1 => readRecord "RealExpr" fun tag =>
      match tag.toNat with
      | 0 => .rational <$> readRat
      | 1 => .parameter <$> readString
      | 2 => .fromInt <$> readIntExpr fuel
      | 3 => .add <$> readRealExpr fuel <*> readRealExpr fuel
      | 4 => .subtract <$> readRealExpr fuel <*> readRealExpr fuel
      | 5 => .multiply <$> readRealExpr fuel <*> readRealExpr fuel
      | 6 => .divide <$> readRealExpr fuel <*> readRealExpr fuel
      | 7 => .sqrt <$> readRealExpr fuel
      | _ => do failAt (.unknownTag ((← get).position - 5) "RealExpr" tag)

private def readMatrixType (fuel : Nat) : DecodeM MatrixTypeExpr := do
  pure {
    modulus := ← readIntExpr fuel
    ringDimension := ← readIntExpr fuel
    rows := ← readIntExpr fuel
    columns := ← readIntExpr fuel
  }

private def readWireType : Nat → DecodeM WireTypeExpr
  | 0 => do failAt (.invalidLength (← get).position)
  | fuel + 1 => readRecord "WireTypeExpr" fun tag =>
      match tag.toNat with
      | 0 => pure .constantInt
      | 1 => pure .constantReal
      | 2 => pure .constantBool
      | 3 => pure .integer
      | 4 => pure .real
      | 5 => pure .boolean
      | 6 => .bytes <$> readIntExpr fuel
      | 7 => .typedBlob <$> readString <*> (readArray (UInt8.toNat <$> readByte))
      | 8 => .matrix <$> readMatrixType fuel
      | 9 => .trapdoor <$> readMatrixType fuel <*> readRealExpr fuel <*>
          readIntExpr fuel <*> readIntExpr fuel <*> readIntExpr fuel
      | 10 => .preimage <$> readMatrixType fuel
      | 11 => .indexedFamily <$> readWireType fuel <*> readIntExpr fuel
      | _ => do failAt (.unknownTag ((← get).position - 5) "WireTypeExpr" tag)

private def readWireRef : DecodeM WireRef := do
  pure { node := ← readU32, port := ← readU32 }

private def readIntBinaryOp : DecodeM IntBinaryOp := do
  let offset := (← get).position
  let tag ← readByte
  match tag.toNat with
  | 0 => pure .add | 1 => pure .subtract | 2 => pure .multiply
  | 3 => pure .divide | 4 => pure .remainder
  | _ => failAt (.unknownTag offset "IntBinaryOp" tag)

private def readRealBinaryOp : DecodeM RealBinaryOp := do
  let offset := (← get).position
  let tag ← readByte
  match tag.toNat with
  | 0 => pure .add | 1 => pure .subtract | 2 => pure .multiply | 3 => pure .divide
  | _ => failAt (.unknownTag offset "RealBinaryOp" tag)

private def readIntCompareOp : DecodeM IntCompareOp := do
  let offset := (← get).position
  let tag ← readByte
  match tag.toNat with
  | 0 => pure .equal | 1 => pure .less | 2 => pure .lessEqual
  | _ => failAt (.unknownTag offset "IntCompareOp" tag)

private def readConcatAxis : DecodeM ConcatAxis := do
  let offset := (← get).position
  let tag ← readByte
  match tag.toNat with
  | 0 => pure .rows | 1 => pure .columns | 2 => pure .diagonal
  | _ => failAt (.unknownTag offset "ConcatAxis" tag)

private def readLoopInputMode : DecodeM LoopInputMode := do
  let offset := (← get).position
  let tag ← readByte
  match tag.toNat with
  | 0 => pure .broadcast | 1 => pure .zip | 2 => .zipOffset <$> readU32
  | _ => failAt (.unknownTag offset "LoopInputMode" tag)

private def readHashVariant : DecodeM Mxx.HashVariant := do
  let offset := (← get).position
  let tag ← readByte
  match tag.toNat with
  | 0 => pure .plain | 1 => pure .decomposed | 2 => pure .smallDecomposed
  | _ => failAt (.unknownTag offset "HashVariant" tag)

private def readBindings (fuel : Nat) : DecodeM (List (String × IntExpr)) :=
  readArray do pure (← readString, ← readIntExpr fuel)

private def readRange (fuel : Nat) : DecodeM (Option (IntExpr × IntExpr)) :=
  readOption do pure (← readIntExpr fuel, ← readIntExpr fuel)

private def readNodeKind (fuel : Nat) : DecodeM NodeKind :=
  readRecord "NodeKind" fun tag =>
    match tag.toNat with
    | 0 => .input <$> readString
    | 1 => .constantInt <$> readInt
    | 2 => .evaluateInt <$> readIntExpr fuel
    | 3 => .constantReal <$> readRealExpr fuel
    | 4 => .constantBool <$> readBool
    | 5 => .zeroMatrix <$> readMatrixType fuel
    | 6 => .identityMatrix <$> readMatrixType fuel
    | 7 => .constantMatrix <$> readMatrixType fuel <*> readArray (readIntExpr fuel)
    | 8 => .unitRowMatrix <$> readMatrixType fuel <*> readIntExpr fuel
    | 9 => .unitColumnMatrix <$> readMatrixType fuel <*> readIntExpr fuel
    | 10 => .gadgetMatrix <$> readMatrixType fuel <*> readIntExpr fuel
    | 11 => .smallGadgetMatrix <$> readMatrixType fuel <*> readIntExpr fuel
    | 12 => .powerOfBaseMatrix <$> readMatrixType fuel <*> readIntExpr fuel <*> readIntExpr fuel
    | 13 => .rotationMatrix <$> readMatrixType fuel <*> readIntExpr fuel
    | 14 => .gadgetTrapdoor <$> readMatrixType fuel <*> readIntExpr fuel
    | 15 => pure .boolToInt
    | 16 => pure .intToReal
    | 17 => .intBinary <$> readIntBinaryOp
    | 18 => .realBinary <$> readRealBinaryOp
    | 19 => pure .realSqrt
    | 20 => .intCompare <$> readIntCompareOp
    | 21 => .bitExtract <$> readIntExpr fuel
    | 22 => .extractCoefficient <$> readIntExpr fuel
    | 23 => .constantCoefficient <$> readIntExpr fuel
    | 24 => pure .select
    | 25 => .uniformResidueSample <$> readMatrixType fuel
    | 26 => .uniformIntervalSample <$> readMatrixType fuel <*> readIntExpr fuel <*> readIntExpr fuel
    | 27 => .gaussianSample <$> readMatrixType fuel <*> readIntExpr fuel
    | 28 => .hashSample <$> readMatrixType fuel <*> readHashVariant <*>
        readArray (UInt8.toNat <$> readByte) <*> readArray (readIntExpr fuel) <*> readArray (readIntExpr fuel) <*>
        readArray (readIntExpr fuel) <*> readOption (readIntExpr fuel) <*> readOption (readIntExpr fuel)
    | 29 => .gadgetDecompose <$> readMatrixType fuel <*> readIntExpr fuel <*> readBool <*> readIntExpr fuel
    | 30 => .trapdoorSample <$> readMatrixType fuel <*> readIntExpr fuel
    | 31 => pure .trapdoorPublic
    | 32 => .preimageSample <$> readMatrixType fuel <*> readIntExpr fuel
    | 33 => pure .matrixAdd | 34 => pure .matrixSubtract | 35 => pure .matrixMultiply
    | 36 => pure .matrixNegate | 37 => .matrixScale <$> readIntExpr fuel
    | 38 => pure .transpose
    | 39 => .slice <$> readRange fuel <*> readRange fuel
    | 40 => pure .tensor
    | 41 => .reshape <$> readIntExpr fuel <*> readIntExpr fuel
    | 42 => .concat <$> readConcatAxis
    | 43 => .thresholdDecodeBool <$> readIntExpr fuel <*> readIntExpr fuel <*> readIntExpr fuel
    | 44 => .thresholdDecodeInt <$> readIntExpr fuel <*> readIntExpr fuel <*> readIntExpr fuel
    | 45 => .crtRecompose <$> readArray (readIntExpr fuel) <*> readArray (readIntExpr fuel)
    | 46 => .packPolynomialCoefficients <$> readMatrixType fuel <*> readIntExpr fuel
    | 47 => pure .familyPack
    | 48 => .familyGetStatic <$> readIntExpr fuel
    | 49 => pure .familyGetDynamic
    | 50 => .subgraphCall <$> readString <*> readBindings fuel
    | 51 => .parallelLoop <$> readString <*> readIntExpr fuel <*> readU32 <*>
        readBindings fuel <*> readArray readLoopInputMode
    | 52 => .sequentialLoop <$> readString <*> readIntExpr fuel <*> readU32 <*>
        readBindings fuel <*> readU32
    | _ => do failAt (.unknownTag ((← get).position - 5) "NodeKind" tag)

private def readNode (fuel : Nat) : DecodeM Node := do
  let kind ← readNodeKind fuel
  let arguments ← readArray readWireRef
  let outputCount ← readU32
  let outputTypes ← readArray (readWireType fuel)
  pure { kind, arguments, outputCount, outputTypes }

private def validateWire (nodes : Array Node) (offset : Nat) (wire : WireRef) : Except DecodeError Unit :=
  match nodes[wire.node]? with
  | none => .error (.invalidWire offset wire.node wire.port)
  | some node => if wire.port < node.outputCount then .ok () else .error (.invalidWire offset wire.node wire.port)

private def readScope (fuel : Nat) : DecodeM Scope := do
  let wireOffset := (← get).position
  let nodes := (← readArray (readNode fuel)).toArray
  let outputs ← readArray do pure (← readString, ← readWireRef)
  let inputNames ← readArray readString
  for node in nodes do
    for wire in node.arguments do
      match validateWire nodes wireOffset wire with
      | .ok () => pure ()
      | .error error => failAt error
  for (_, wire) in outputs do
    match validateWire nodes wireOffset wire with
    | .ok () => pure ()
    | .error error => failAt error
  pure { nodes, outputs, inputNames }

private def readProgPayload (fuel : Nat) : DecodeM Prog := do
  let root ← readScope fuel
  let definitions ← readArray do pure (← readString, ← readScope fuel)
  pure { root, definitions }

private def readDerivationRule : DecodeM Mxx.Certificate.DerivationRule :=
  readRecord "DerivationRule" fun tag =>
    match tag.toNat with
    | 0 => pure .input | 1 => pure .constantInt | 2 => pure .evaluateInt
    | 3 => pure .constantReal | 4 => pure .constantBool | 5 => pure .zeroMatrix
    | 6 => pure .identityMatrix | 7 => pure .constantMatrix | 8 => pure .unitRowMatrix
    | 9 => pure .unitColumnMatrix | 10 => pure .gadgetMatrix | 11 => pure .smallGadgetMatrix
    | 12 => pure .powerOfBaseMatrix | 13 => pure .rotationMatrix | 14 => pure .gadgetTrapdoor
    | 15 => pure .intToReal | 16 => pure .boolToInt | 17 => pure .intBinary
    | 18 => pure .realBinary | 19 => pure .realSqrt | 20 => pure .intCompare
    | 21 => pure .bitExtract | 22 => pure .extractCoefficient | 23 => pure .constantCoefficient
    | 24 => pure .select | 25 => pure .uniformResidueSample | 26 => pure .uniformIntervalSample
    | 27 => pure .gaussianSample | 28 => pure .hashSample | 29 => pure .gadgetDecompose
    | 30 => pure .trapdoorSample | 31 => pure .trapdoorPublic | 32 => pure .preimageSample
    | 33 => pure .matrixAdd | 34 => pure .matrixSubtract | 35 => pure .matrixMultiplyBound
    | 36 => .matrixMultiplyRelation <$> readWireRef
    | 37 => pure .matrixNegate | 38 => pure .matrixScale | 39 => pure .transpose
    | 40 => pure .slice | 41 => pure .tensor | 42 => pure .reshape | 43 => pure .concat
    | 44 => pure .thresholdDecodeBool | 45 => pure .thresholdDecodeInt
    | 46 => pure .crtRecompose | 47 => pure .packPolynomialCoefficients
    | 48 => pure .familyPack | 49 => pure .familyGetStatic | 50 => pure .familyGetDynamic
    | 51 => pure .subgraphCall | 52 => pure .parallelLoop | 53 => pure .sequentialLoop
    | _ => do failAt (.unknownTag ((← get).position - 5) "DerivationRule" tag)

private def readNodeDerivation : DecodeM Mxx.Certificate.NodeDerivation := do
  pure {
    sourceNode := ← readU32
    rule := ← readDerivationRule
    arguments := ← readArray readWireRef
  }

private def readDerivationAttachment : DecodeM Mxx.Certificate.DerivationAttachment := do
  pure {
    ownerNamespace := ← readString
    ruleName := ← readString
    roles := ← readArray do pure (← readString, ← readWireRef)
  }

private def readScopeDerivation : DecodeM Mxx.Certificate.ScopeDerivation := do
  pure {
    steps := (← readArray readNodeDerivation).toArray
    attachments := ← readArray readDerivationAttachment
  }

private def readProgramDerivationPayload (_fuel : Nat) : DecodeM Mxx.Certificate.ProgramDerivation := do
  pure {
    root := ← readScopeDerivation
    definitions := ← readArray do pure (← readString, ← readScopeDerivation)
  }

private def readEnvelope {α : Type}
    (bytes : ByteArray)
    (expectedKind : UInt8)
    (readPayload : Nat → DecodeM α) : Except DecodeError α := do
  let initial : Reader := { bytes, position := 0, limit := bytes.size }
  let (version, state) ← readByte.run initial
  if version != binaryFormatVersion then throw (.wrongVersion 0 version)
  let (kind, state) ← readByte.run state
  if kind != expectedKind then throw (.wrongDocumentKind 1 expectedKind kind)
  let (payloadLength, state) ← readU32.run state
  let (stringCount, state) ← readU32.run state
  let (blobLength, state) ← readU32.run state
  let (offsets, state) ← ((List.range (stringCount + 1)).mapM fun _ => readU32).run state
  if offsets.head? != some 0 ∨ offsets.getLast? != some blobLength then
    throw (.invalidLength state.position)
  if !(offsets.zip offsets.tail).all fun (left, right) => left ≤ right then
    throw (.invalidLength state.position)
  let blobOffset := state.position
  let (blob, state) ← (readBytes blobLength).run state
  let strings ← (offsets.zip offsets.tail).mapM fun (startOffset, endOffset) =>
    match String.fromUTF8? (blob.extract startOffset endOffset) with
    | some value => pure value
    | none => throw (.invalidUtf8 (blobOffset + startOffset))
  let payloadStart := state.position
  let payloadEnd := payloadStart + payloadLength
  if payloadEnd < payloadStart ∨ payloadEnd != bytes.size then
    throw (.trailingBytes payloadEnd)
  let payloadState := { state with limit := payloadEnd, strings := strings.toArray }
  let (value, final) ← (readPayload (payloadLength + 1)).run payloadState
  if final.position != payloadEnd then throw (.trailingBytes final.position)
  pure value

def decodeProg (bytes : ByteArray) : Except DecodeError Prog :=
  readEnvelope bytes 1 readProgPayload

def decodeProgramDerivation
    (bytes : ByteArray) : Except DecodeError Mxx.Certificate.ProgramDerivation :=
  readEnvelope bytes 2 readProgramDerivationPayload

private def hexNibble (offset : Nat) (value : UInt8) : Except DecodeError UInt8 :=
  if 48 ≤ value.toNat ∧ value.toNat ≤ 57 then
    pure (UInt8.ofNat (value.toNat - 48))
  else if 97 ≤ value.toNat ∧ value.toNat ≤ 102 then
    pure (UInt8.ofNat (value.toNat - 87))
  else
    throw (.invalidHex offset)

def decodeHexChunks (chunks : Array String) : Except DecodeError ByteArray := do
  let mut output := ByteArray.empty
  let mut sourceOffset := 0
  for chunk in chunks do
    let bytes := chunk.toUTF8
    if bytes.size % 2 != 0 then throw (.invalidHex sourceOffset)
    for pairIndex in List.range (bytes.size / 2) do
      let index := pairIndex * 2
      let high ← match bytes[index]? with
        | some value => hexNibble (sourceOffset + index) value
        | none => throw (.invalidHex (sourceOffset + index))
      let low ← match bytes[index + 1]? with
        | some value => hexNibble (sourceOffset + index + 1) value
        | none => throw (.invalidHex (sourceOffset + index + 1))
      output := output.push (UInt8.ofNat (high.toNat * 16 + low.toNat))
    sourceOffset := sourceOffset + bytes.size
  pure output

end Mxx.Ir
