# Graph IR Binary Format v1

## Purpose

This format transports Graph IR and its untrusted derivation from the Rust generator to Lean. It
does not change the trust boundary: all checkers and theorems consume the decoded `Mxx.Ir.Prog`
and `Mxx.Certificate.ProgramDerivation`, and Rust asserts no semantic fact. Logical workflow and
derivation hashes remain computed over the existing canonical logical representation, not these
transport bytes.

The Rust constant `IR_BINARY_FORMAT_VERSION` and Lean constant
`Mxx.Ir.binaryFormatVersion` both equal `1`.

## Primitives and envelope

- `u8` is one byte. A Boolean is exactly `0` or `1`.
- `u32` is unsigned little-endian and encodes counts, lengths, indices, ports, slots, and arities.
  The encoder rejects values above `u32::MAX`.
- `int` is a `u32` length and the unique shortest non-empty two's-complement little-endian byte
  sequence. Redundant sign extension is invalid.
- `rat` is a canonical `int` numerator and canonical positive `int` denominator, in lowest terms.
- `option<T>` is tag `0`, or tag `1` followed by `T`.
- `array<T>` is a `u32` count followed by its elements. A pair stores its fields in order.
- Every algebraic value is `u8 tag | u32 payload_length | payload`. A known record must consume its
  exact payload; unknown tags and leftover or truncated payload bytes are errors.

Each independently decoded document is:

```text
u8 version (= 1)
u8 document_kind (Prog = 1, ProgramDerivation = 2, ClosedProtocolTransport = 3)
u32 payload_length
u32 string_count
u32 string_blob_length
u32 offsets[string_count + 1]
u8 utf8_blob[string_blob_length]
u8 payload[payload_length]
```

Offsets are monotone, begin at zero, end at `string_blob_length`, and delimit valid UTF-8. A string
in the payload is its `u32` table index. Strings are interned in first-occurrence order during the
canonical traversal; duplicate strings use their first index and unused entries are invalid. The
decoder rejects a wrong version or document kind and trailing bytes. Decoder errors carry the
zero-based document offset at which the failure was detected.

## Expressions and types

Tags and payloads are:

- `IntExpr`: `constant=0(int)`, `parameter=1(string)`, `loopIndex=2(u32)`,
  `add=3(left,right)`, `subtract=4(left,right)`, `multiply=5(left,right)`,
  `divide=6(left,right)`, `roundDivide=7(left,right)`, `log2Ceil=8(value)`.
- `RealExpr`: `rational=0(rat)`, `parameter=1(string)`, `fromInt=2(IntExpr)`,
  `add=3(left,right)`, `subtract=4(left,right)`, `multiply=5(left,right)`,
  `divide=6(left,right)`, `sqrt=7(value)`.
- `MatrixTypeExpr`: `modulus, ringDimension, rows, columns` in that order.
- `WireTypeExpr`: `constantInt=0`, `constantReal=1`, `constantBool=2`, `integer=3`,
  `real=4`, `boolean=5`, `bytes=6(IntExpr)`,
  `typedBlob=7(string,array<u8>)`, `matrix=8(MatrixTypeExpr)`,
  `trapdoor=9(MatrixTypeExpr,RealExpr,IntExpr gadgetBase,IntExpr digitCount,
  IntExpr preimageMaxCoefficientBound)`, `preimage=10(MatrixTypeExpr)`, and
  `indexedFamily=11(WireTypeExpr,IntExpr count)`.

Small enums use declaration-order tags: `IntBinaryOp=0..4`, `IntCompareOp=0..2`,
`RealBinaryOp=0..3`, `ConcatAxis=0..2`; `LoopInputMode` is `broadcast=0`, `zip=1`, or
`zipOffset=2(u32 offset)`. `HashVariant` also uses its declaration-order `u8` tag. Changing any
such declaration requires a new format version.

## Node kinds

`NodeKind` tags follow the declaration order in `lean/Mxx/Ir.lean`:

| Tags | Variants |
| --- | --- |
| 0--4 | `input(string)`, `constantInt(int)`, `evaluateInt(IntExpr)`, `constantReal(RealExpr)`, `constantBool(bool)` |
| 5--14 | `zeroMatrix(type)`, `identityMatrix(type)`, `constantMatrix(type,array<IntExpr>)`, `unitRowMatrix(type,index)`, `unitColumnMatrix(type,index)`, `gadgetMatrix(type,base)`, `smallGadgetMatrix(type,base)`, `powerOfBaseMatrix(type,base,exponent)`, `rotationMatrix(type,exponent)`, `gadgetTrapdoor(type,base)` |
| 15--24 | `boolToInt`, `intToReal`, `intBinary(op)`, `realBinary(op)`, `realSqrt`, `intCompare(op)`, `bitExtract(bit)`, `extractCoefficient(position)`, `constantCoefficient(position)`, `select` |
| 25--32 | `uniformResidueSample(type)`, `uniformIntervalSample(type,min,max)`, `gaussianSample(type,bound)`, `hashSample(type,variant,array<u8> prefix,array<IntExpr> expressions,array<IntExpr> decimalExpressions,array<IntExpr> u64LeExpressions,option<IntExpr> base,option<IntExpr> digits)`, `gadgetDecompose(type,base,bool small,digits)`, `trapdoorSample(type,bound)`, `trapdoorPublic`, `preimageSample(type,bound)` |
| 33--42 | `matrixAdd`, `matrixSubtract`, `matrixMultiply`, `matrixNegate`, `matrixScale(scalar)`, `transpose`, `slice(option<(IntExpr,IntExpr)> rows,option<(IntExpr,IntExpr)> columns)`, `tensor`, `reshape(rows,columns)`, `concat(axis)` |
| 43--52 | `thresholdDecodeBool(q,p,length)`, `thresholdDecodeInt(q,p,length)`, `crtRecompose(array<IntExpr>,array<IntExpr>)`, `packPolynomialCoefficients(type,bits)`, `familyPack`, `familyGetStatic(index)`, `familyGetDynamic`, `subgraphCall(string,array<(string,IntExpr)>)`, `parallelLoop(string,count,indexSlot,bindings,inputModes)`, `sequentialLoop(string,count,indexSlot,bindings,carriedCount)` |

Variants without a displayed payload have an empty TLV payload. An encoder must fail on an active
Rust variant not represented above; it must never use fallback serialization.

## Programs

- `WireRef`: `u32 node, u32 port`.
- `Node`: `NodeKind, array<WireRef> arguments, u32 outputCount, array<WireTypeExpr> outputTypes`.
- `Scope`: `array<Node> nodes, array<(string,WireRef)> outputs, array<string> inputNames`.
- `Prog`: `Scope root, array<(string,Scope)> definitions`.

All arrays preserve source order. Numeric overflow and structurally out-of-range wire or port
indices are decode errors. Existing Graph IR validation remains responsible for semantic validity.

## Derivations

`DerivationRule` uses declaration-order tags `0..51`. `matrixMultiplyRelation` is tag `36` and
carries its `WireRef rightOperand`; all other rule payloads are empty.

- `NodeDerivation`: `u32 sourceNode, DerivationRule, array<WireRef> arguments`.
- `DerivationAttachment`: `string ownerNamespace, string ruleName,
  array<(string,WireRef)> roles`.
- `ScopeDerivation`: `array<NodeDerivation> steps, array<DerivationAttachment> attachments`.
- `ProgramDerivation`: `ScopeDerivation root, array<(string,ScopeDerivation)> definitions`.

Program/derivation correspondence is checked by `checkProgramDerivation`; it is not trusted or
asserted by the decoder.

## Workflow and input contracts

The combined document follows the field order of the existing normalized closed-protocol logical
serialization. It contains parameter declarations, input contracts, workflow stages, artifact
bindings, protocol-input bindings, the ideal program, requirement programs, comparator program,
comparator endpoints, endpoint anchors, semantic anchors, and the corresponding stage, ideal,
requirement, and comparator derivations. Collections retain their normalized logical order.

Leaf structures (`ParameterDecl`, `InputContract`, `Stage`, `ArtifactBinding`,
`ProtocolInputBinding`, `ComparatorEndpointBinding`, `EndpointAnchors`, and
`SemanticAnchorBinding`) store fields in their Lean declaration order. Their enum fields use
declaration-order TLV tags and declared payloads. Before the v1 combined encoder is enabled, its
implementation must enumerate every active leaf variant and a fixture must demonstrate exact
round-trip equality. This requirement prevents a partially covered workflow format from shipping.

## Canonical order and determinism

The encoder preserves the existing normalized logical field and collection order wherever it
exists, traverses program records in the order above, interns strings on first occurrence, emits
minimal integers and exact lengths, and adds no padding. Consequently a supported logical value
has exactly one v1 encoding. Regeneration of unchanged input must be byte-identical and must leave
the existing logical workflow and derivation hashes unchanged.
