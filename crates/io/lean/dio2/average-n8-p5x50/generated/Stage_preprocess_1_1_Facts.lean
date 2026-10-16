import Stage_preprocess_1_1

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000
set_option linter.unusedVariables false

-- The wire types repeat the ring modulus `X ^ n + 1`; unfolding it while checking a
-- projection against its field type expands it at the ring dimension.
attribute [local irreducible] Mxx.Primitives.negacyclicModulus

noncomputable section

namespace Stage_preprocess_1_1

local notation "mxxNat0" => 1356938545745524308187737599098762797340022013559604731140326430414605958745
local notation "mxxNat1" => 1583094970036445026219027198948556596896692349152872186330380835483706951869
local notation "mxxMat0" => Mxx.Primitives.ExactMatrix modulus_0 8 1 1
local notation "mxxMat1" => Mxx.Primitives.ExactMatrix modulus_0 8 1 25
local notation "mxxMat2" => Mxx.Primitives.ExactMatrix modulus_0 8 2 25
local notation "mxxMat3" => Mxx.Primitives.ExactMatrix modulus_0 8 25 25

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_3.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) where
  inputs : ByteArray × mxxMat0 × mxxMat0 × Unit
  witness : parallel_generatedRoot_3.Witness

def parallel_generatedRoot_3.w_0_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_3.Ctx backend hashModel tape path params i_0) : ByteArray :=
  ctx.inputs.1

def parallel_generatedRoot_3.w_3_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_3.Ctx backend hashModel tape path params i_0) : mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_3.w_26_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_3.Ctx backend hashModel tape path params i_0) : mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_3.w_1_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_3.Ctx backend hashModel tape path params i_0) : mxxMat1 :=
  ctx.witness.w_1_0

def parallel_generatedRoot_3.w_2_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_3.Ctx backend hashModel tape path params i_0) : mxxMat1 :=
  ctx.witness.w_2_0

def parallel_generatedRoot_3.w_4_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_3.Ctx backend hashModel tape path params i_0) : mxxMat0 :=
  ctx.witness.w_4_0

def parallel_generatedRoot_3.w_5_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_3.Ctx backend hashModel tape path params i_0) : mxxMat0 :=
  MxxRuntime.matrixNeg (parallel_generatedRoot_3.w_4_0 ctx)

def parallel_generatedRoot_3.w_6_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_3.Ctx backend hashModel tape path params i_0) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_3.w_5_0 ctx) (parallel_generatedRoot_3.w_1_0 ctx)

def parallel_generatedRoot_3.w_7_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_3.Ctx backend hashModel tape path params i_0) : mxxMat0 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat0)

def parallel_generatedRoot_3.w_8_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_3.Ctx backend hashModel tape path params i_0) : mxxMat0 :=
  (MxxRuntime.matrixPolynomial [(mxxNat0)] : mxxMat0)

def parallel_generatedRoot_3.w_9_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_3.Ctx backend hashModel tape path params i_0) : mxxMat0 :=
  (0 : mxxMat0)

def parallel_generatedRoot_3.w_10_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_3.Ctx backend hashModel tape path params i_0) : mxxMat0 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat0)

def parallel_generatedRoot_3.w_11_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_3.Ctx backend hashModel tape path params i_0) : Int :=
  -2

def parallel_generatedRoot_3.w_12_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_3.Ctx backend hashModel tape path params i_0) : Fin 4 → Int :=
  ctx.witness.w_12_0

def parallel_generatedRoot_3.w_13_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_3.Ctx backend hashModel tape path params i_0) : mxxMat0 :=
  ctx.witness.w_13_0

def parallel_generatedRoot_3.w_14_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_3.Ctx backend hashModel tape path params i_0) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_3.w_8_0 ctx) (parallel_generatedRoot_3.w_13_0 ctx)

def parallel_generatedRoot_3.w_15_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_3.Ctx backend hashModel tape path params i_0) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_3.w_7_0 ctx) (parallel_generatedRoot_3.w_14_0 ctx)

def parallel_generatedRoot_3.w_16_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_3.Ctx backend hashModel tape path params i_0) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_3.w_15_0 ctx) (parallel_generatedRoot_3.w_3_0 ctx)

def parallel_generatedRoot_3.w_17_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_3.Ctx backend hashModel tape path params i_0) : mxxMat1 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_3.w_18_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_3.Ctx backend hashModel tape path params i_0) : mxxMat1 :=
  ctx.witness.w_18_0

def parallel_generatedRoot_3.w_19_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_3.Ctx backend hashModel tape path params i_0) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_3.w_16_0 ctx) (parallel_generatedRoot_3.w_18_0 ctx)

def parallel_generatedRoot_3.w_20_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_3.Ctx backend hashModel tape path params i_0) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_3.w_6_0 ctx) (parallel_generatedRoot_3.w_19_0 ctx)

def parallel_generatedRoot_3.w_21_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_3.Ctx backend hashModel tape path params i_0) : mxxMat1 :=
  (0 : mxxMat1)

def parallel_generatedRoot_3.w_22_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_3.Ctx backend hashModel tape path params i_0) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_3.w_20_0 ctx) (parallel_generatedRoot_3.w_21_0 ctx)

def parallel_generatedRoot_3.w_23_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_3.Ctx backend hashModel tape path params i_0) : mxxMat0 :=
  MxxRuntime.matrixNeg (parallel_generatedRoot_3.w_4_0 ctx)

def parallel_generatedRoot_3.w_24_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_3.Ctx backend hashModel tape path params i_0) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_3.w_23_0 ctx) (parallel_generatedRoot_3.w_2_0 ctx)

def parallel_generatedRoot_3.w_25_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_3.Ctx backend hashModel tape path params i_0) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_3.w_7_0 ctx) (parallel_generatedRoot_3.w_14_0 ctx)

def parallel_generatedRoot_3.w_27_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_3.Ctx backend hashModel tape path params i_0) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_3.w_25_0 ctx) (parallel_generatedRoot_3.w_26_0 ctx)

def parallel_generatedRoot_3.w_28_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_3.Ctx backend hashModel tape path params i_0) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_3.w_27_0 ctx) (parallel_generatedRoot_3.w_18_0 ctx)

def parallel_generatedRoot_3.w_29_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_3.Ctx backend hashModel tape path params i_0) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_3.w_24_0 ctx) (parallel_generatedRoot_3.w_28_0 ctx)

def parallel_generatedRoot_3.w_30_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_3.Ctx backend hashModel tape path params i_0) : mxxMat1 :=
  (0 : mxxMat1)

def parallel_generatedRoot_3.w_31_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_3.Ctx backend hashModel tape path params i_0) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_3.w_29_0 ctx) (parallel_generatedRoot_3.w_30_0 ctx)

def parallel_generatedRoot_3.outs {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_3.Ctx backend hashModel tape path params i_0) : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit :=
  ((parallel_generatedRoot_3.w_1_0 ctx), ((parallel_generatedRoot_3.w_2_0 ctx), ((parallel_generatedRoot_3.w_22_0 ctx), ((parallel_generatedRoot_3.w_31_0 ctx), ()))))

structure parallel_generatedRoot_3.Facts_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_3.Ctx backend hashModel tape path params i_0) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  t0 : 0 ≤ ((Int.ofNat i_0))
  t1 : ((Int.ofNat i_0)) < 4
  h1_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 98, 47, 76, 117, 116, 73, 100, 40, 49, 41, 47, 49, 47, 50]) ([.bytes [115, 105, 103, 109, 97], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(1), (3), (5), (15)])), .bytes [98, 114, 97, 110, 99, 104], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((parallel_generatedRoot_3.w_0_0 ctx)) (parallel_generatedRoot_3.w_1_0 ctx)
  t3 : 0 ≤ ((Int.ofNat i_0))
  t4 : ((Int.ofNat i_0)) < 4
  h2_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 98, 47, 76, 117, 116, 73, 100, 40, 49, 41, 47, 49, 47, 50]) ([.bytes [115, 105, 103, 109, 97], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(1), (3), (5), (15)])), .bytes [98, 114, 97, 110, 99, 104], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((parallel_generatedRoot_3.w_0_0 ctx)) (parallel_generatedRoot_3.w_2_0 ctx)
  t6 : 0 ≤ ((Int.ofNat i_0))
  t7 : ((Int.ofNat i_0)) < 4
  h4_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(1), (3), (5), (15)])) (parallel_generatedRoot_3.w_3_0 ctx) (parallel_generatedRoot_3.w_4_0 ctx)
  t9 : (4) = 4
  h12_0 : (∀ i : Fin 4, parallel_parallel_generatedRoot_3_12 backend hashModel tape (path ++ [12, i.val]) params i_0 i () ((parallel_generatedRoot_3.w_12_0 ctx) i))
  t11 : 0 ≤ (4)
  h13_0 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0) => sequential_parallel_generatedRoot_3_13 backend hashModel tape (path ++ [13, i]) params i_0 i (current, ((parallel_generatedRoot_3.w_10_0 ctx), ((parallel_generatedRoot_3.w_11_0 ctx), ((parallel_generatedRoot_3.w_12_0 ctx), ())))) next) (Int.toNat (4)) (parallel_generatedRoot_3.w_9_0 ctx) (parallel_generatedRoot_3.w_13_0 ctx)
  h17_0 : MxxRuntime.gadgetMatrixRuns backend 4096 25 (parallel_generatedRoot_3.w_17_0 ctx)
  h18_0 : MxxRuntime.centeredRebaseRuns (parallel_generatedRoot_3.w_17_0 ctx) (parallel_generatedRoot_3.w_18_0 ctx)
  hout : outputs = parallel_generatedRoot_3.outs ctx

theorem parallel_generatedRoot_3.facts_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} {ctx : parallel_generatedRoot_3.Ctx backend hashModel tape path params i_0} {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_3.constraints_0 backend hashModel tape path params i_0 (parallel_generatedRoot_3.w_0_0 ctx) (parallel_generatedRoot_3.w_3_0 ctx) (parallel_generatedRoot_3.w_1_0 ctx) (parallel_generatedRoot_3.w_2_0 ctx) (parallel_generatedRoot_3.w_4_0 ctx) (parallel_generatedRoot_3.w_9_0 ctx) (parallel_generatedRoot_3.w_10_0 ctx) (parallel_generatedRoot_3.w_11_0 ctx) (parallel_generatedRoot_3.w_12_0 ctx) (parallel_generatedRoot_3.w_13_0 ctx) (parallel_generatedRoot_3.w_17_0 ctx) (parallel_generatedRoot_3.w_18_0 ctx) (parallel_generatedRoot_3.w_22_0 ctx) (parallel_generatedRoot_3.w_31_0 ctx) outputs) :
    parallel_generatedRoot_3.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_3.Facts {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_3.Ctx backend hashModel tape path params i_0) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  c0 : parallel_generatedRoot_3.Facts_0 ctx outputs

theorem parallel_generatedRoot_3.facts {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_3.Ctx backend hashModel tape path params i_0) {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_3.body backend hashModel tape path params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_3.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_3.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_4.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) where
  inputs : ByteArray × mxxMat0 × mxxMat0 × Unit
  witness : parallel_generatedRoot_4.Witness

def parallel_generatedRoot_4.w_0_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_4.Ctx backend hashModel tape path params i_0) : ByteArray :=
  ctx.inputs.1

def parallel_generatedRoot_4.w_3_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_4.Ctx backend hashModel tape path params i_0) : mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_4.w_25_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_4.Ctx backend hashModel tape path params i_0) : mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_4.w_1_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_4.Ctx backend hashModel tape path params i_0) : mxxMat1 :=
  ctx.witness.w_1_0

def parallel_generatedRoot_4.w_2_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_4.Ctx backend hashModel tape path params i_0) : mxxMat1 :=
  ctx.witness.w_2_0

def parallel_generatedRoot_4.w_4_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_4.Ctx backend hashModel tape path params i_0) : mxxMat0 :=
  ctx.witness.w_4_0

def parallel_generatedRoot_4.w_5_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_4.Ctx backend hashModel tape path params i_0) : mxxMat0 :=
  MxxRuntime.matrixNeg (parallel_generatedRoot_4.w_4_0 ctx)

def parallel_generatedRoot_4.w_6_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_4.Ctx backend hashModel tape path params i_0) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_4.w_5_0 ctx) (parallel_generatedRoot_4.w_1_0 ctx)

def parallel_generatedRoot_4.w_7_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_4.Ctx backend hashModel tape path params i_0) : mxxMat0 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat0)

def parallel_generatedRoot_4.w_8_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_4.Ctx backend hashModel tape path params i_0) : mxxMat0 :=
  (MxxRuntime.matrixPolynomial [(mxxNat1)] : mxxMat0)

def parallel_generatedRoot_4.w_9_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_4.Ctx backend hashModel tape path params i_0) : mxxMat0 :=
  (0 : mxxMat0)

def parallel_generatedRoot_4.w_10_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_4.Ctx backend hashModel tape path params i_0) : Fin 8 → Int :=
  ctx.witness.w_10_0

def parallel_generatedRoot_4.w_11_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_4.Ctx backend hashModel tape path params i_0) : mxxMat0 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat0)

def parallel_generatedRoot_4.w_12_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_4.Ctx backend hashModel tape path params i_0) : Int :=
  -1

def parallel_generatedRoot_4.w_13_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_4.Ctx backend hashModel tape path params i_0) : mxxMat0 :=
  ctx.witness.w_13_0

def parallel_generatedRoot_4.w_14_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_4.Ctx backend hashModel tape path params i_0) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_4.w_8_0 ctx) (parallel_generatedRoot_4.w_13_0 ctx)

def parallel_generatedRoot_4.w_15_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_4.Ctx backend hashModel tape path params i_0) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_4.w_7_0 ctx) (parallel_generatedRoot_4.w_14_0 ctx)

def parallel_generatedRoot_4.w_16_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_4.Ctx backend hashModel tape path params i_0) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_4.w_15_0 ctx) (parallel_generatedRoot_4.w_3_0 ctx)

def parallel_generatedRoot_4.w_17_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_4.Ctx backend hashModel tape path params i_0) : mxxMat1 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_4.w_18_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_4.Ctx backend hashModel tape path params i_0) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_4.w_16_0 ctx) (parallel_generatedRoot_4.w_17_0 ctx)

def parallel_generatedRoot_4.w_19_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_4.Ctx backend hashModel tape path params i_0) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_4.w_6_0 ctx) (parallel_generatedRoot_4.w_18_0 ctx)

def parallel_generatedRoot_4.w_20_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_4.Ctx backend hashModel tape path params i_0) : mxxMat1 :=
  (0 : mxxMat1)

def parallel_generatedRoot_4.w_21_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_4.Ctx backend hashModel tape path params i_0) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_4.w_19_0 ctx) (parallel_generatedRoot_4.w_20_0 ctx)

def parallel_generatedRoot_4.w_22_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_4.Ctx backend hashModel tape path params i_0) : mxxMat0 :=
  MxxRuntime.matrixNeg (parallel_generatedRoot_4.w_4_0 ctx)

def parallel_generatedRoot_4.w_23_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_4.Ctx backend hashModel tape path params i_0) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_4.w_22_0 ctx) (parallel_generatedRoot_4.w_2_0 ctx)

def parallel_generatedRoot_4.w_24_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_4.Ctx backend hashModel tape path params i_0) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_4.w_7_0 ctx) (parallel_generatedRoot_4.w_14_0 ctx)

def parallel_generatedRoot_4.w_26_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_4.Ctx backend hashModel tape path params i_0) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_4.w_24_0 ctx) (parallel_generatedRoot_4.w_25_0 ctx)

def parallel_generatedRoot_4.w_27_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_4.Ctx backend hashModel tape path params i_0) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_4.w_26_0 ctx) (parallel_generatedRoot_4.w_17_0 ctx)

def parallel_generatedRoot_4.w_28_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_4.Ctx backend hashModel tape path params i_0) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_4.w_23_0 ctx) (parallel_generatedRoot_4.w_27_0 ctx)

def parallel_generatedRoot_4.w_29_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_4.Ctx backend hashModel tape path params i_0) : mxxMat1 :=
  (0 : mxxMat1)

def parallel_generatedRoot_4.w_30_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_4.Ctx backend hashModel tape path params i_0) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_4.w_28_0 ctx) (parallel_generatedRoot_4.w_29_0 ctx)

def parallel_generatedRoot_4.outs {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_4.Ctx backend hashModel tape path params i_0) : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit :=
  ((parallel_generatedRoot_4.w_1_0 ctx), ((parallel_generatedRoot_4.w_2_0 ctx), ((parallel_generatedRoot_4.w_21_0 ctx), ((parallel_generatedRoot_4.w_30_0 ctx), ()))))

structure parallel_generatedRoot_4.Facts_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_4.Ctx backend hashModel tape path params i_0) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  t0 : 0 ≤ ((Int.ofNat i_0))
  t1 : ((Int.ofNat i_0)) < 8
  h1_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 98, 47, 76, 117, 116, 73, 100, 40, 48, 41, 47, 50, 47, 51]) ([.bytes [115, 105, 103, 109, 97], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(1), (3), (5), (7), (9), (11), (13), (15)])), .bytes [98, 114, 97, 110, 99, 104], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((parallel_generatedRoot_4.w_0_0 ctx)) (parallel_generatedRoot_4.w_1_0 ctx)
  t3 : 0 ≤ ((Int.ofNat i_0))
  t4 : ((Int.ofNat i_0)) < 8
  h2_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 98, 47, 76, 117, 116, 73, 100, 40, 48, 41, 47, 50, 47, 51]) ([.bytes [115, 105, 103, 109, 97], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(1), (3), (5), (7), (9), (11), (13), (15)])), .bytes [98, 114, 97, 110, 99, 104], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((parallel_generatedRoot_4.w_0_0 ctx)) (parallel_generatedRoot_4.w_2_0 ctx)
  t6 : 0 ≤ ((Int.ofNat i_0))
  t7 : ((Int.ofNat i_0)) < 8
  h4_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(1), (3), (5), (7), (9), (11), (13), (15)])) (parallel_generatedRoot_4.w_3_0 ctx) (parallel_generatedRoot_4.w_4_0 ctx)
  t9 : (8) = 8
  h10_0 : (∀ i : Fin 8, parallel_parallel_generatedRoot_4_10 backend hashModel tape (path ++ [10, i.val]) params i_0 i () ((parallel_generatedRoot_4.w_10_0 ctx) i))
  t11 : 0 ≤ (8)
  h13_0 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0) => sequential_parallel_generatedRoot_4_13 backend hashModel tape (path ++ [13, i]) params i_0 i (current, ((parallel_generatedRoot_4.w_10_0 ctx), ((parallel_generatedRoot_4.w_11_0 ctx), ((parallel_generatedRoot_4.w_12_0 ctx), ())))) next) (Int.toNat (8)) (parallel_generatedRoot_4.w_9_0 ctx) (parallel_generatedRoot_4.w_13_0 ctx)
  h17_0 : MxxRuntime.gadgetMatrixRuns backend 4096 25 (parallel_generatedRoot_4.w_17_0 ctx)
  hout : outputs = parallel_generatedRoot_4.outs ctx

theorem parallel_generatedRoot_4.facts_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} {ctx : parallel_generatedRoot_4.Ctx backend hashModel tape path params i_0} {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_4.constraints_0 backend hashModel tape path params i_0 (parallel_generatedRoot_4.w_0_0 ctx) (parallel_generatedRoot_4.w_3_0 ctx) (parallel_generatedRoot_4.w_1_0 ctx) (parallel_generatedRoot_4.w_2_0 ctx) (parallel_generatedRoot_4.w_4_0 ctx) (parallel_generatedRoot_4.w_9_0 ctx) (parallel_generatedRoot_4.w_10_0 ctx) (parallel_generatedRoot_4.w_11_0 ctx) (parallel_generatedRoot_4.w_12_0 ctx) (parallel_generatedRoot_4.w_13_0 ctx) (parallel_generatedRoot_4.w_17_0 ctx) (parallel_generatedRoot_4.w_21_0 ctx) (parallel_generatedRoot_4.w_30_0 ctx) outputs) :
    parallel_generatedRoot_4.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_4.Facts {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_4.Ctx backend hashModel tape path params i_0) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  c0 : parallel_generatedRoot_4.Facts_0 ctx outputs

theorem parallel_generatedRoot_4.facts {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_4.Ctx backend hashModel tape path params i_0) {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_4.body backend hashModel tape path params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_4.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_4.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_76.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) where
  inputs : (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × Unit
  witness : parallel_generatedRoot_76.Witness

def parallel_generatedRoot_76.w_0_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_76.Ctx backend hashModel tape path params i_0) : Fin 16 → mxxMat1 :=
  ctx.inputs.1

def parallel_generatedRoot_76.w_1_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_76.Ctx backend hashModel tape path params i_0) : Fin 16 → mxxMat1 :=
  ctx.inputs.2.1

def parallel_generatedRoot_76.w_2_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_76.Ctx backend hashModel tape path params i_0) : Fin 16 → mxxMat1 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_76.w_3_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_76.Ctx backend hashModel tape path params i_0) : Fin 16 → mxxMat1 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_76.w_4_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_76.Ctx backend hashModel tape path params i_0) : Fin 2 → mxxMat1 :=
  ctx.witness.w_4_0

def parallel_generatedRoot_76.w_4_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_76.Ctx backend hashModel tape path params i_0) : Fin 2 → mxxMat1 :=
  ctx.witness.w_4_1

def parallel_generatedRoot_76.w_4_2 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_76.Ctx backend hashModel tape path params i_0) : Fin 2 → mxxMat1 :=
  ctx.witness.w_4_2

def parallel_generatedRoot_76.w_4_3 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_76.Ctx backend hashModel tape path params i_0) : Fin 2 → mxxMat1 :=
  ctx.witness.w_4_3

def parallel_generatedRoot_76.w_5_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_76.Ctx backend hashModel tape path params i_0) : mxxMat1 :=
  ctx.witness.w_5_0

def parallel_generatedRoot_76.w_6_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_76.Ctx backend hashModel tape path params i_0) : mxxMat1 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_76.w_7_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_76.Ctx backend hashModel tape path params i_0) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_76.w_5_0 ctx) (parallel_generatedRoot_76.w_6_0 ctx)

def parallel_generatedRoot_76.w_8_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_76.Ctx backend hashModel tape path params i_0) : mxxMat1 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_76.w_9_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_76.Ctx backend hashModel tape path params i_0) : mxxMat1 :=
  ctx.witness.w_9_0

def parallel_generatedRoot_76.w_10_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_76.Ctx backend hashModel tape path params i_0) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_76.w_8_0 ctx) (parallel_generatedRoot_76.w_9_0 ctx)

def parallel_generatedRoot_76.w_11_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_76.Ctx backend hashModel tape path params i_0) : mxxMat1 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_76.w_12_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_76.Ctx backend hashModel tape path params i_0) : mxxMat1 :=
  ctx.witness.w_12_0

def parallel_generatedRoot_76.w_13_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_76.Ctx backend hashModel tape path params i_0) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_76.w_11_0 ctx) (parallel_generatedRoot_76.w_12_0 ctx)

def parallel_generatedRoot_76.w_14_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_76.Ctx backend hashModel tape path params i_0) : mxxMat1 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_76.w_15_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_76.Ctx backend hashModel tape path params i_0) : mxxMat1 :=
  ctx.witness.w_15_0

def parallel_generatedRoot_76.w_16_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_76.Ctx backend hashModel tape path params i_0) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_76.w_14_0 ctx) (parallel_generatedRoot_76.w_15_0 ctx)

def parallel_generatedRoot_76.w_17_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_76.Ctx backend hashModel tape path params i_0) : mxxMat1 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_76.w_17_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_76.Ctx backend hashModel tape path params i_0) : mxxMat1 :=
  ctx.witness.w_17_1

def parallel_generatedRoot_76.w_17_2 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_76.Ctx backend hashModel tape path params i_0) : mxxMat1 :=
  ctx.witness.w_17_2

def parallel_generatedRoot_76.w_17_3 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_76.Ctx backend hashModel tape path params i_0) : mxxMat1 :=
  ctx.witness.w_17_3

def parallel_generatedRoot_76.outs {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_76.Ctx backend hashModel tape path params i_0) : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit :=
  ((parallel_generatedRoot_76.w_17_0 ctx), ((parallel_generatedRoot_76.w_17_1 ctx), ((parallel_generatedRoot_76.w_17_2 ctx), ((parallel_generatedRoot_76.w_17_3 ctx), ()))))

structure parallel_generatedRoot_76.Facts_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_76.Ctx backend hashModel tape path params i_0) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  t0 : (2) = 2
  h4_3 : (∀ i : Fin 2, parallel_parallel_generatedRoot_76_4 backend hashModel tape (path ++ [4, i.val]) params i_0 i ((parallel_generatedRoot_76.w_0_0 ctx), ((parallel_generatedRoot_76.w_1_0 ctx), ((parallel_generatedRoot_76.w_2_0 ctx), ((parallel_generatedRoot_76.w_3_0 ctx), ())))) (((parallel_generatedRoot_76.w_4_0 ctx) i), (((parallel_generatedRoot_76.w_4_1 ctx) i), (((parallel_generatedRoot_76.w_4_2 ctx) i), (((parallel_generatedRoot_76.w_4_3 ctx) i), ())))))
  h5_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_76.w_4_0 ctx) (0) (parallel_generatedRoot_76.w_5_0 ctx)
  h6_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_76.w_4_0 ctx) (1) (parallel_generatedRoot_76.w_6_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_76.w_4_1 ctx) (0) (parallel_generatedRoot_76.w_8_0 ctx)
  h9_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_76.w_4_1 ctx) (1) (parallel_generatedRoot_76.w_9_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_76.w_4_2 ctx) (0) (parallel_generatedRoot_76.w_11_0 ctx)
  h12_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_76.w_4_2 ctx) (1) (parallel_generatedRoot_76.w_12_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_76.w_4_3 ctx) (0) (parallel_generatedRoot_76.w_14_0 ctx)
  h15_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_76.w_4_3 ctx) (1) (parallel_generatedRoot_76.w_15_0 ctx)
  t10 : 0 ≤ (0)
  h17_3 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_parallel_generatedRoot_76_17 backend hashModel tape (path ++ [17, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, ((parallel_generatedRoot_76.w_4_0 ctx), ((parallel_generatedRoot_76.w_4_1 ctx), ((parallel_generatedRoot_76.w_4_2 ctx), ((parallel_generatedRoot_76.w_4_3 ctx), ())))))))) next) (Int.toNat (0)) ((parallel_generatedRoot_76.w_7_0 ctx), ((parallel_generatedRoot_76.w_10_0 ctx), ((parallel_generatedRoot_76.w_13_0 ctx), ((parallel_generatedRoot_76.w_16_0 ctx), ())))) ((parallel_generatedRoot_76.w_17_0 ctx), ((parallel_generatedRoot_76.w_17_1 ctx), ((parallel_generatedRoot_76.w_17_2 ctx), ((parallel_generatedRoot_76.w_17_3 ctx), ()))))
  hout : outputs = parallel_generatedRoot_76.outs ctx

theorem parallel_generatedRoot_76.facts_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} {ctx : parallel_generatedRoot_76.Ctx backend hashModel tape path params i_0} {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_76.constraints_0 backend hashModel tape path params i_0 (parallel_generatedRoot_76.w_0_0 ctx) (parallel_generatedRoot_76.w_1_0 ctx) (parallel_generatedRoot_76.w_2_0 ctx) (parallel_generatedRoot_76.w_3_0 ctx) (parallel_generatedRoot_76.w_4_0 ctx) (parallel_generatedRoot_76.w_4_1 ctx) (parallel_generatedRoot_76.w_4_2 ctx) (parallel_generatedRoot_76.w_4_3 ctx) (parallel_generatedRoot_76.w_5_0 ctx) (parallel_generatedRoot_76.w_6_0 ctx) (parallel_generatedRoot_76.w_7_0 ctx) (parallel_generatedRoot_76.w_8_0 ctx) (parallel_generatedRoot_76.w_9_0 ctx) (parallel_generatedRoot_76.w_10_0 ctx) (parallel_generatedRoot_76.w_11_0 ctx) (parallel_generatedRoot_76.w_12_0 ctx) (parallel_generatedRoot_76.w_13_0 ctx) (parallel_generatedRoot_76.w_14_0 ctx) (parallel_generatedRoot_76.w_15_0 ctx) (parallel_generatedRoot_76.w_16_0 ctx) (parallel_generatedRoot_76.w_17_0 ctx) (parallel_generatedRoot_76.w_17_1 ctx) (parallel_generatedRoot_76.w_17_2 ctx) (parallel_generatedRoot_76.w_17_3 ctx) outputs) :
    parallel_generatedRoot_76.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_76.Facts {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_76.Ctx backend hashModel tape path params i_0) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  c0 : parallel_generatedRoot_76.Facts_0 ctx outputs

theorem parallel_generatedRoot_76.facts {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_76.Ctx backend hashModel tape path params i_0) {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_76.body backend hashModel tape path params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_76.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_76.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_80.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) where
  inputs : (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × Unit
  witness : parallel_generatedRoot_80.Witness

def parallel_generatedRoot_80.w_0_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_80.Ctx backend hashModel tape path params i_0) : Fin 16 → mxxMat1 :=
  ctx.inputs.1

def parallel_generatedRoot_80.w_1_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_80.Ctx backend hashModel tape path params i_0) : Fin 16 → mxxMat1 :=
  ctx.inputs.2.1

def parallel_generatedRoot_80.w_2_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_80.Ctx backend hashModel tape path params i_0) : Fin 16 → mxxMat1 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_80.w_3_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_80.Ctx backend hashModel tape path params i_0) : Fin 16 → mxxMat1 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_80.w_4_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_80.Ctx backend hashModel tape path params i_0) : Fin 2 → mxxMat1 :=
  ctx.witness.w_4_0

def parallel_generatedRoot_80.w_4_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_80.Ctx backend hashModel tape path params i_0) : Fin 2 → mxxMat1 :=
  ctx.witness.w_4_1

def parallel_generatedRoot_80.w_4_2 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_80.Ctx backend hashModel tape path params i_0) : Fin 2 → mxxMat1 :=
  ctx.witness.w_4_2

def parallel_generatedRoot_80.w_4_3 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_80.Ctx backend hashModel tape path params i_0) : Fin 2 → mxxMat1 :=
  ctx.witness.w_4_3

def parallel_generatedRoot_80.w_5_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_80.Ctx backend hashModel tape path params i_0) : mxxMat1 :=
  ctx.witness.w_5_0

def parallel_generatedRoot_80.w_6_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_80.Ctx backend hashModel tape path params i_0) : mxxMat1 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_80.w_7_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_80.Ctx backend hashModel tape path params i_0) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_80.w_5_0 ctx) (parallel_generatedRoot_80.w_6_0 ctx)

def parallel_generatedRoot_80.w_8_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_80.Ctx backend hashModel tape path params i_0) : mxxMat1 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_80.w_9_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_80.Ctx backend hashModel tape path params i_0) : mxxMat1 :=
  ctx.witness.w_9_0

def parallel_generatedRoot_80.w_10_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_80.Ctx backend hashModel tape path params i_0) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_80.w_8_0 ctx) (parallel_generatedRoot_80.w_9_0 ctx)

def parallel_generatedRoot_80.w_11_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_80.Ctx backend hashModel tape path params i_0) : mxxMat1 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_80.w_12_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_80.Ctx backend hashModel tape path params i_0) : mxxMat1 :=
  ctx.witness.w_12_0

def parallel_generatedRoot_80.w_13_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_80.Ctx backend hashModel tape path params i_0) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_80.w_11_0 ctx) (parallel_generatedRoot_80.w_12_0 ctx)

def parallel_generatedRoot_80.w_14_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_80.Ctx backend hashModel tape path params i_0) : mxxMat1 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_80.w_15_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_80.Ctx backend hashModel tape path params i_0) : mxxMat1 :=
  ctx.witness.w_15_0

def parallel_generatedRoot_80.w_16_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_80.Ctx backend hashModel tape path params i_0) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_80.w_14_0 ctx) (parallel_generatedRoot_80.w_15_0 ctx)

def parallel_generatedRoot_80.w_17_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_80.Ctx backend hashModel tape path params i_0) : mxxMat1 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_80.w_17_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_80.Ctx backend hashModel tape path params i_0) : mxxMat1 :=
  ctx.witness.w_17_1

def parallel_generatedRoot_80.w_17_2 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_80.Ctx backend hashModel tape path params i_0) : mxxMat1 :=
  ctx.witness.w_17_2

def parallel_generatedRoot_80.w_17_3 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_80.Ctx backend hashModel tape path params i_0) : mxxMat1 :=
  ctx.witness.w_17_3

def parallel_generatedRoot_80.outs {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_80.Ctx backend hashModel tape path params i_0) : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit :=
  ((parallel_generatedRoot_80.w_17_0 ctx), ((parallel_generatedRoot_80.w_17_1 ctx), ((parallel_generatedRoot_80.w_17_2 ctx), ((parallel_generatedRoot_80.w_17_3 ctx), ()))))

structure parallel_generatedRoot_80.Facts_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_80.Ctx backend hashModel tape path params i_0) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  t0 : (2) = 2
  h4_3 : (∀ i : Fin 2, parallel_parallel_generatedRoot_80_4 backend hashModel tape (path ++ [4, i.val]) params i_0 i ((parallel_generatedRoot_80.w_0_0 ctx), ((parallel_generatedRoot_80.w_1_0 ctx), ((parallel_generatedRoot_80.w_2_0 ctx), ((parallel_generatedRoot_80.w_3_0 ctx), ())))) (((parallel_generatedRoot_80.w_4_0 ctx) i), (((parallel_generatedRoot_80.w_4_1 ctx) i), (((parallel_generatedRoot_80.w_4_2 ctx) i), (((parallel_generatedRoot_80.w_4_3 ctx) i), ())))))
  h5_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_80.w_4_0 ctx) (0) (parallel_generatedRoot_80.w_5_0 ctx)
  h6_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_80.w_4_0 ctx) (1) (parallel_generatedRoot_80.w_6_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_80.w_4_1 ctx) (0) (parallel_generatedRoot_80.w_8_0 ctx)
  h9_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_80.w_4_1 ctx) (1) (parallel_generatedRoot_80.w_9_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_80.w_4_2 ctx) (0) (parallel_generatedRoot_80.w_11_0 ctx)
  h12_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_80.w_4_2 ctx) (1) (parallel_generatedRoot_80.w_12_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_80.w_4_3 ctx) (0) (parallel_generatedRoot_80.w_14_0 ctx)
  h15_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_80.w_4_3 ctx) (1) (parallel_generatedRoot_80.w_15_0 ctx)
  t10 : 0 ≤ (0)
  h17_3 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_parallel_generatedRoot_80_17 backend hashModel tape (path ++ [17, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, ((parallel_generatedRoot_80.w_4_0 ctx), ((parallel_generatedRoot_80.w_4_1 ctx), ((parallel_generatedRoot_80.w_4_2 ctx), ((parallel_generatedRoot_80.w_4_3 ctx), ())))))))) next) (Int.toNat (0)) ((parallel_generatedRoot_80.w_7_0 ctx), ((parallel_generatedRoot_80.w_10_0 ctx), ((parallel_generatedRoot_80.w_13_0 ctx), ((parallel_generatedRoot_80.w_16_0 ctx), ())))) ((parallel_generatedRoot_80.w_17_0 ctx), ((parallel_generatedRoot_80.w_17_1 ctx), ((parallel_generatedRoot_80.w_17_2 ctx), ((parallel_generatedRoot_80.w_17_3 ctx), ()))))
  hout : outputs = parallel_generatedRoot_80.outs ctx

theorem parallel_generatedRoot_80.facts_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} {ctx : parallel_generatedRoot_80.Ctx backend hashModel tape path params i_0} {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_80.constraints_0 backend hashModel tape path params i_0 (parallel_generatedRoot_80.w_0_0 ctx) (parallel_generatedRoot_80.w_1_0 ctx) (parallel_generatedRoot_80.w_2_0 ctx) (parallel_generatedRoot_80.w_3_0 ctx) (parallel_generatedRoot_80.w_4_0 ctx) (parallel_generatedRoot_80.w_4_1 ctx) (parallel_generatedRoot_80.w_4_2 ctx) (parallel_generatedRoot_80.w_4_3 ctx) (parallel_generatedRoot_80.w_5_0 ctx) (parallel_generatedRoot_80.w_6_0 ctx) (parallel_generatedRoot_80.w_7_0 ctx) (parallel_generatedRoot_80.w_8_0 ctx) (parallel_generatedRoot_80.w_9_0 ctx) (parallel_generatedRoot_80.w_10_0 ctx) (parallel_generatedRoot_80.w_11_0 ctx) (parallel_generatedRoot_80.w_12_0 ctx) (parallel_generatedRoot_80.w_13_0 ctx) (parallel_generatedRoot_80.w_14_0 ctx) (parallel_generatedRoot_80.w_15_0 ctx) (parallel_generatedRoot_80.w_16_0 ctx) (parallel_generatedRoot_80.w_17_0 ctx) (parallel_generatedRoot_80.w_17_1 ctx) (parallel_generatedRoot_80.w_17_2 ctx) (parallel_generatedRoot_80.w_17_3 ctx) outputs) :
    parallel_generatedRoot_80.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_80.Facts {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_80.Ctx backend hashModel tape path params i_0) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  c0 : parallel_generatedRoot_80.Facts_0 ctx outputs

theorem parallel_generatedRoot_80.facts {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_80.Ctx backend hashModel tape path params i_0) {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_80.body backend hashModel tape path params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_80.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_80.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_81.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) where
  inputs : ByteArray × mxxMat1 × mxxMat1 × (Fin 8 → mxxMat1) × mxxMat1 × mxxMat1 × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × Unit
  witness : parallel_generatedRoot_81.Witness

def parallel_generatedRoot_81.w_1_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_81.Ctx backend hashModel tape path params i_0) : ByteArray :=
  ctx.inputs.1

def parallel_generatedRoot_81.w_2_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_81.Ctx backend hashModel tape path params i_0) : mxxMat1 :=
  ctx.inputs.2.1

def parallel_generatedRoot_81.w_3_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_81.Ctx backend hashModel tape path params i_0) : mxxMat1 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_81.w_4_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_81.Ctx backend hashModel tape path params i_0) : Fin 8 → mxxMat1 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_81.w_5_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_81.Ctx backend hashModel tape path params i_0) : mxxMat1 :=
  ctx.inputs.2.2.2.2.1

def parallel_generatedRoot_81.w_6_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_81.Ctx backend hashModel tape path params i_0) : mxxMat1 :=
  ctx.inputs.2.2.2.2.2.1

def parallel_generatedRoot_81.w_7_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_81.Ctx backend hashModel tape path params i_0) : Fin 8 → mxxMat1 :=
  ctx.inputs.2.2.2.2.2.2.1

def parallel_generatedRoot_81.w_8_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_81.Ctx backend hashModel tape path params i_0) : Fin 8 → mxxMat1 :=
  ctx.inputs.2.2.2.2.2.2.2.1

def parallel_generatedRoot_81.w_9_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_81.Ctx backend hashModel tape path params i_0) : Fin 8 → mxxMat1 :=
  ctx.inputs.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_81.w_0_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_81.Ctx backend hashModel tape path params i_0) : mxxMat1 :=
  (0 : mxxMat1)

def parallel_generatedRoot_81.w_10_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_81.Ctx backend hashModel tape path params i_0) : mxxMat1 :=
  ctx.witness.w_10_0

def parallel_generatedRoot_81.w_10_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_81.Ctx backend hashModel tape path params i_0) : mxxMat1 :=
  ctx.witness.w_10_1

def parallel_generatedRoot_81.w_10_2 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_81.Ctx backend hashModel tape path params i_0) : mxxMat1 :=
  ctx.witness.w_10_2

def parallel_generatedRoot_81.w_10_3 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_81.Ctx backend hashModel tape path params i_0) : mxxMat1 :=
  ctx.witness.w_10_3

def parallel_generatedRoot_81.outs {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_81.Ctx backend hashModel tape path params i_0) : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit :=
  ((parallel_generatedRoot_81.w_10_0 ctx), ((parallel_generatedRoot_81.w_10_1 ctx), ((parallel_generatedRoot_81.w_10_2 ctx), ((parallel_generatedRoot_81.w_10_3 ctx), ()))))

structure parallel_generatedRoot_81.Facts_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_81.Ctx backend hashModel tape path params i_0) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  t0 : 0 ≤ (8)
  h10_3 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_parallel_generatedRoot_81_10 backend hashModel tape (path ++ [10, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, ((parallel_generatedRoot_81.w_1_0 ctx), ((parallel_generatedRoot_81.w_2_0 ctx), ((parallel_generatedRoot_81.w_3_0 ctx), ((parallel_generatedRoot_81.w_4_0 ctx), ((parallel_generatedRoot_81.w_5_0 ctx), ((parallel_generatedRoot_81.w_6_0 ctx), ((parallel_generatedRoot_81.w_7_0 ctx), ((parallel_generatedRoot_81.w_8_0 ctx), ((parallel_generatedRoot_81.w_9_0 ctx), ()))))))))))))) next) (Int.toNat (8)) ((parallel_generatedRoot_81.w_0_0 ctx), ((parallel_generatedRoot_81.w_0_0 ctx), ((parallel_generatedRoot_81.w_0_0 ctx), ((parallel_generatedRoot_81.w_0_0 ctx), ())))) ((parallel_generatedRoot_81.w_10_0 ctx), ((parallel_generatedRoot_81.w_10_1 ctx), ((parallel_generatedRoot_81.w_10_2 ctx), ((parallel_generatedRoot_81.w_10_3 ctx), ()))))
  hout : outputs = parallel_generatedRoot_81.outs ctx

theorem parallel_generatedRoot_81.facts_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} {ctx : parallel_generatedRoot_81.Ctx backend hashModel tape path params i_0} {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_81.constraints_0 backend hashModel tape path params i_0 (parallel_generatedRoot_81.w_1_0 ctx) (parallel_generatedRoot_81.w_2_0 ctx) (parallel_generatedRoot_81.w_3_0 ctx) (parallel_generatedRoot_81.w_4_0 ctx) (parallel_generatedRoot_81.w_5_0 ctx) (parallel_generatedRoot_81.w_6_0 ctx) (parallel_generatedRoot_81.w_7_0 ctx) (parallel_generatedRoot_81.w_8_0 ctx) (parallel_generatedRoot_81.w_9_0 ctx) (parallel_generatedRoot_81.w_0_0 ctx) (parallel_generatedRoot_81.w_10_0 ctx) (parallel_generatedRoot_81.w_10_1 ctx) (parallel_generatedRoot_81.w_10_2 ctx) (parallel_generatedRoot_81.w_10_3 ctx) outputs) :
    parallel_generatedRoot_81.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure parallel_generatedRoot_81.Facts {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_81.Ctx backend hashModel tape path params i_0) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  c0 : parallel_generatedRoot_81.Facts_0 ctx outputs

theorem parallel_generatedRoot_81.facts {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_81.Ctx backend hashModel tape path params i_0) {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_81.body backend hashModel tape path params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_81.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_81.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) where
  inputs : ByteArray × mxxMat0 × ByteArray × (Fin 1 → Int) × (Fin 1 → Int) × (Fin 16 → Int) × (Fin 16 → Int) × (Fin 16 → Int) × (Fin 16 → Int) × (Fin 16 → Int) × (Fin 16 → Int) × Unit
  witness : generatedRoot.Witness

def generatedRoot.w_0_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : ByteArray :=
  ctx.inputs.1

def generatedRoot.w_1_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.inputs.2.1

def generatedRoot.w_6_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : ByteArray :=
  ctx.inputs.2.2.1

def generatedRoot.w_31_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 1 → Int :=
  ctx.inputs.2.2.2.1

def generatedRoot.w_32_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 1 → Int :=
  ctx.inputs.2.2.2.2.1

def generatedRoot.w_67_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 16 → Int :=
  ctx.inputs.2.2.2.2.2.1

def generatedRoot.w_68_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 16 → Int :=
  ctx.inputs.2.2.2.2.2.2.1

def generatedRoot.w_73_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 16 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.1

def generatedRoot.w_74_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 16 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.1

def generatedRoot.w_77_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 16 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_78_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 16 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_2_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  (1 : mxxMat0)

def generatedRoot.w_3_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 4 → mxxMat1 :=
  ctx.witness.w_3_0

def generatedRoot.w_3_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 4 → mxxMat1 :=
  ctx.witness.w_3_1

def generatedRoot.w_3_2 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 4 → mxxMat1 :=
  ctx.witness.w_3_2

def generatedRoot.w_3_3 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 4 → mxxMat1 :=
  ctx.witness.w_3_3

def generatedRoot.w_4_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → mxxMat1 :=
  ctx.witness.w_4_0

def generatedRoot.w_4_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → mxxMat1 :=
  ctx.witness.w_4_1

def generatedRoot.w_4_2 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → mxxMat1 :=
  ctx.witness.w_4_2

def generatedRoot.w_4_3 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → mxxMat1 :=
  ctx.witness.w_4_3

def generatedRoot.w_5_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat1 :=
  (0 : mxxMat1)

def generatedRoot.w_7_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat1 :=
  ctx.witness.w_7_0

def generatedRoot.w_8_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Int :=
  2

def generatedRoot.w_9_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat1 :=
  MxxRuntime.multiplyMonomial (generatedRoot.w_7_0 ctx) (generatedRoot.w_8_0 ctx)

def generatedRoot.w_10_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat1 :=
  ctx.witness.w_10_0

def generatedRoot.w_11_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Int :=
  2

def generatedRoot.w_12_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat1 :=
  MxxRuntime.multiplyMonomial (generatedRoot.w_10_0 ctx) (generatedRoot.w_11_0 ctx)

def generatedRoot.w_13_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.witness.w_13_0

def generatedRoot.w_14_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat1 :=
  ctx.witness.w_14_0

def generatedRoot.w_15_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat3 :=
  ctx.witness.w_15_0

def generatedRoot.w_16_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  MxxRuntime.matrixMul (generatedRoot.w_13_0 ctx) (generatedRoot.w_15_0 ctx)

def generatedRoot.w_17_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat1 :=
  ctx.witness.w_17_0

def generatedRoot.w_18_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Int :=
  2

def generatedRoot.w_19_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat1 :=
  MxxRuntime.multiplyMonomial (generatedRoot.w_17_0 ctx) (generatedRoot.w_18_0 ctx)

def generatedRoot.w_20_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat1 :=
  ctx.witness.w_20_0

def generatedRoot.w_21_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Int :=
  2

def generatedRoot.w_22_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat1 :=
  MxxRuntime.multiplyMonomial (generatedRoot.w_20_0 ctx) (generatedRoot.w_21_0 ctx)

def generatedRoot.w_23_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.witness.w_23_0

def generatedRoot.w_24_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat1 :=
  ctx.witness.w_24_0

def generatedRoot.w_25_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat3 :=
  ctx.witness.w_25_0

def generatedRoot.w_26_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  MxxRuntime.matrixMul (generatedRoot.w_23_0 ctx) (generatedRoot.w_25_0 ctx)

def generatedRoot.w_27_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  MxxRuntime.matrixAdd (generatedRoot.w_16_0 ctx) (generatedRoot.w_26_0 ctx)

def generatedRoot.w_28_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat1 :=
  ctx.witness.w_28_0

def generatedRoot.w_29_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat1 :=
  ctx.witness.w_29_0

def generatedRoot.w_30_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.witness.w_30_0

def generatedRoot.w_33_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 1 → mxxMat1 :=
  ctx.witness.w_33_0

def generatedRoot.w_33_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 1 → mxxMat1 :=
  ctx.witness.w_33_1

def generatedRoot.w_33_2 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 1 → mxxMat1 :=
  ctx.witness.w_33_2

def generatedRoot.w_33_3 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 1 → mxxMat1 :=
  ctx.witness.w_33_3

def generatedRoot.w_34_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat1 :=
  ctx.witness.w_34_0

def generatedRoot.w_35_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat3 :=
  ctx.witness.w_35_0

def generatedRoot.w_36_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat3 :=
  ctx.witness.w_36_0

def generatedRoot.w_37_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  MxxRuntime.matrixMul (generatedRoot.w_30_0 ctx) (generatedRoot.w_36_0 ctx)

def generatedRoot.w_38_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat1 :=
  ctx.witness.w_38_0

def generatedRoot.w_39_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat3 :=
  ctx.witness.w_39_0

def generatedRoot.w_40_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  MxxRuntime.matrixMul (generatedRoot.w_13_0 ctx) (generatedRoot.w_39_0 ctx)

def generatedRoot.w_41_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat1 :=
  ctx.witness.w_41_0

def generatedRoot.w_42_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat3 :=
  ctx.witness.w_42_0

def generatedRoot.w_43_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  MxxRuntime.matrixMul (generatedRoot.w_23_0 ctx) (generatedRoot.w_42_0 ctx)

def generatedRoot.w_44_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  MxxRuntime.matrixAdd (generatedRoot.w_40_0 ctx) (generatedRoot.w_43_0 ctx)

def generatedRoot.w_45_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat1 :=
  ctx.witness.w_45_0

def generatedRoot.w_46_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat1 :=
  ctx.witness.w_46_0

def generatedRoot.w_47_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.witness.w_47_0

def generatedRoot.w_48_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat1 :=
  ctx.witness.w_48_0

def generatedRoot.w_49_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat3 :=
  ctx.witness.w_49_0

def generatedRoot.w_50_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat3 :=
  ctx.witness.w_50_0

def generatedRoot.w_51_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  MxxRuntime.matrixMul (generatedRoot.w_47_0 ctx) (generatedRoot.w_50_0 ctx)

def generatedRoot.w_52_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  MxxRuntime.matrixAdd (generatedRoot.w_37_0 ctx) (generatedRoot.w_51_0 ctx)

def generatedRoot.w_53_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat1 :=
  ctx.witness.w_53_0

def generatedRoot.w_54_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat1 :=
  ctx.witness.w_54_0

def generatedRoot.w_55_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat1 :=
  ctx.witness.w_55_0

def generatedRoot.w_56_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat3 :=
  ctx.witness.w_56_0

def generatedRoot.w_57_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat3 :=
  ctx.witness.w_57_0

def generatedRoot.w_58_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  MxxRuntime.matrixMul (generatedRoot.w_30_0 ctx) (generatedRoot.w_57_0 ctx)

def generatedRoot.w_59_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat1 :=
  ctx.witness.w_59_0

def generatedRoot.w_60_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat3 :=
  ctx.witness.w_60_0

def generatedRoot.w_61_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat3 :=
  ctx.witness.w_61_0

def generatedRoot.w_62_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  MxxRuntime.matrixMul (generatedRoot.w_47_0 ctx) (generatedRoot.w_61_0 ctx)

def generatedRoot.w_63_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  MxxRuntime.matrixAdd (generatedRoot.w_58_0 ctx) (generatedRoot.w_62_0 ctx)

def generatedRoot.w_64_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat1 :=
  ctx.witness.w_64_0

def generatedRoot.w_65_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat1 :=
  ctx.witness.w_65_0

def generatedRoot.w_66_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat1 :=
  ctx.witness.w_66_0

def generatedRoot.w_66_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat1 :=
  ctx.witness.w_66_1

def generatedRoot.w_66_2 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat1 :=
  ctx.witness.w_66_2

def generatedRoot.w_66_3 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat1 :=
  ctx.witness.w_66_3

def generatedRoot.w_69_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat1 :=
  ctx.witness.w_69_0

def generatedRoot.w_70_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat1 :=
  ctx.witness.w_70_0

def generatedRoot.w_71_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat1 :=
  ctx.witness.w_71_0

def generatedRoot.w_72_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat1 :=
  ctx.witness.w_72_0

def generatedRoot.w_75_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 16 → mxxMat1 :=
  ctx.witness.w_75_0

def generatedRoot.w_75_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 16 → mxxMat1 :=
  ctx.witness.w_75_1

def generatedRoot.w_75_2 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 16 → mxxMat1 :=
  ctx.witness.w_75_2

def generatedRoot.w_75_3 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 16 → mxxMat1 :=
  ctx.witness.w_75_3

def generatedRoot.w_75_4 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 16 → mxxMat1 :=
  ctx.witness.w_75_4

def generatedRoot.w_75_5 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 16 → mxxMat1 :=
  ctx.witness.w_75_5

def generatedRoot.w_75_6 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 16 → mxxMat1 :=
  ctx.witness.w_75_6

def generatedRoot.w_75_7 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 16 → mxxMat1 :=
  ctx.witness.w_75_7

def generatedRoot.w_76_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → mxxMat1 :=
  ctx.witness.w_76_0

def generatedRoot.w_76_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → mxxMat1 :=
  ctx.witness.w_76_1

def generatedRoot.w_76_2 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → mxxMat1 :=
  ctx.witness.w_76_2

def generatedRoot.w_76_3 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → mxxMat1 :=
  ctx.witness.w_76_3

def generatedRoot.w_79_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 16 → mxxMat1 :=
  ctx.witness.w_79_0

def generatedRoot.w_79_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 16 → mxxMat1 :=
  ctx.witness.w_79_1

def generatedRoot.w_79_2 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 16 → mxxMat1 :=
  ctx.witness.w_79_2

def generatedRoot.w_79_3 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 16 → mxxMat1 :=
  ctx.witness.w_79_3

def generatedRoot.w_79_4 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 16 → mxxMat1 :=
  ctx.witness.w_79_4

def generatedRoot.w_79_5 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 16 → mxxMat1 :=
  ctx.witness.w_79_5

def generatedRoot.w_79_6 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 16 → mxxMat1 :=
  ctx.witness.w_79_6

def generatedRoot.w_79_7 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 16 → mxxMat1 :=
  ctx.witness.w_79_7

def generatedRoot.w_80_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → mxxMat1 :=
  ctx.witness.w_80_0

def generatedRoot.w_80_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → mxxMat1 :=
  ctx.witness.w_80_1

def generatedRoot.w_80_2 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → mxxMat1 :=
  ctx.witness.w_80_2

def generatedRoot.w_80_3 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → mxxMat1 :=
  ctx.witness.w_80_3

def generatedRoot.w_81_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → mxxMat1 :=
  ctx.witness.w_81_0

def generatedRoot.w_81_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → mxxMat1 :=
  ctx.witness.w_81_1

def generatedRoot.w_81_2 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → mxxMat1 :=
  ctx.witness.w_81_2

def generatedRoot.w_81_3 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → mxxMat1 :=
  ctx.witness.w_81_3

def generatedRoot.w_82_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → mxxMat1 :=
  ctx.witness.w_82_0

def generatedRoot.w_82_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → mxxMat1 :=
  ctx.witness.w_82_1

def generatedRoot.w_82_2 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → mxxMat1 :=
  ctx.witness.w_82_2

def generatedRoot.w_82_3 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → mxxMat1 :=
  ctx.witness.w_82_3

def generatedRoot.w_83_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat1 :=
  ctx.witness.w_83_0

def generatedRoot.w_84_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat1 :=
  ctx.witness.w_84_0

def generatedRoot.w_85_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat1 :=
  MxxRuntime.matrixAdd (generatedRoot.w_83_0 ctx) (generatedRoot.w_84_0 ctx)

def generatedRoot.w_86_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat1 :=
  ctx.witness.w_86_0

def generatedRoot.w_87_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat1 :=
  ctx.witness.w_87_0

def generatedRoot.w_88_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat1 :=
  MxxRuntime.matrixAdd (generatedRoot.w_86_0 ctx) (generatedRoot.w_87_0 ctx)

def generatedRoot.w_89_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat1 :=
  ctx.witness.w_89_0

def generatedRoot.w_90_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat1 :=
  ctx.witness.w_90_0

def generatedRoot.w_91_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat1 :=
  MxxRuntime.matrixAdd (generatedRoot.w_89_0 ctx) (generatedRoot.w_90_0 ctx)

def generatedRoot.w_92_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat1 :=
  ctx.witness.w_92_0

def generatedRoot.w_93_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat1 :=
  ctx.witness.w_93_0

def generatedRoot.w_94_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat1 :=
  MxxRuntime.matrixAdd (generatedRoot.w_92_0 ctx) (generatedRoot.w_93_0 ctx)

def generatedRoot.w_95_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat1 :=
  ctx.witness.w_95_0

def generatedRoot.w_95_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat1 :=
  ctx.witness.w_95_1

def generatedRoot.w_95_2 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat1 :=
  ctx.witness.w_95_2

def generatedRoot.w_95_3 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat1 :=
  ctx.witness.w_95_3

def generatedRoot.outs {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × Unit :=
  ((generatedRoot.w_3_0 ctx), ((generatedRoot.w_3_1 ctx), ((generatedRoot.w_3_2 ctx), ((generatedRoot.w_3_3 ctx), ((generatedRoot.w_4_0 ctx), ((generatedRoot.w_4_1 ctx), ((generatedRoot.w_4_2 ctx), ((generatedRoot.w_4_3 ctx), ((generatedRoot.w_66_0 ctx), ((generatedRoot.w_66_1 ctx), ((generatedRoot.w_66_2 ctx), ((generatedRoot.w_66_3 ctx), ((generatedRoot.w_95_0 ctx), ((generatedRoot.w_95_1 ctx), ((generatedRoot.w_95_2 ctx), ((generatedRoot.w_95_3 ctx), ((generatedRoot.w_75_6 ctx), ((generatedRoot.w_75_7 ctx), ((generatedRoot.w_79_6 ctx), ((generatedRoot.w_79_7 ctx), ((generatedRoot.w_33_2 ctx), ((generatedRoot.w_33_3 ctx), ()))))))))))))))))))))))

structure generatedRoot.Facts_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Prop where
  t0 : (4) = 4
  h3_3 : (∀ i : Fin 4, parallel_generatedRoot_3 backend hashModel tape (path ++ [3, i.val]) params i ((generatedRoot.w_0_0 ctx), ((generatedRoot.w_1_0 ctx), ((generatedRoot.w_2_0 ctx), ()))) (((generatedRoot.w_3_0 ctx) i), (((generatedRoot.w_3_1 ctx) i), (((generatedRoot.w_3_2 ctx) i), (((generatedRoot.w_3_3 ctx) i), ())))))
  t2 : (8) = 8
  h4_3 : (∀ i : Fin 8, parallel_generatedRoot_4 backend hashModel tape (path ++ [4, i.val]) params i ((generatedRoot.w_0_0 ctx), ((generatedRoot.w_1_0 ctx), ((generatedRoot.w_2_0 ctx), ()))) (((generatedRoot.w_4_0 ctx) i), (((generatedRoot.w_4_1 ctx) i), (((generatedRoot.w_4_2 ctx) i), (((generatedRoot.w_4_3 ctx) i), ())))))
  h7_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((generatedRoot.w_6_0 ctx)) (generatedRoot.w_7_0 ctx)
  h10_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) ((generatedRoot.w_6_0 ctx)) (generatedRoot.w_10_0 ctx)
  h13_0 : MxxRuntime.concatRows (generatedRoot.w_9_0 ctx) (generatedRoot.w_12_0 ctx) (generatedRoot.w_13_0 ctx)
  h14_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 115, 116, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((generatedRoot.w_6_0 ctx)) (generatedRoot.w_14_0 ctx)
  h15_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_14_0 ctx) (generatedRoot.w_15_0 ctx)
  h17_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((generatedRoot.w_6_0 ctx)) (generatedRoot.w_17_0 ctx)
  h20_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) ((generatedRoot.w_6_0 ctx)) (generatedRoot.w_20_0 ctx)
  h23_0 : MxxRuntime.concatRows (generatedRoot.w_19_0 ctx) (generatedRoot.w_22_0 ctx) (generatedRoot.w_23_0 ctx)
  h24_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 115, 116, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) ((generatedRoot.w_6_0 ctx)) (generatedRoot.w_24_0 ctx)
  h25_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_24_0 ctx) (generatedRoot.w_25_0 ctx)
  t14 : 0 ≤ 0
  t15 : 0 < 1
  t16 : 1 ≤ 2
  t17 : 0 ≤ 0
  t18 : 0 < 25
  t19 : 25 ≤ 25
  h28_0 : MxxRuntime.sliceMatrix (generatedRoot.w_27_0 ctx) 0 1 0 25 (generatedRoot.w_28_0 ctx)
  t21 : 0 ≤ 1
  t22 : 1 < 2
  t23 : 2 ≤ 2
  t24 : 0 ≤ 0
  t25 : 0 < 25
  t26 : 25 ≤ 25
  h29_0 : MxxRuntime.sliceMatrix (generatedRoot.w_27_0 ctx) 1 2 0 25 (generatedRoot.w_29_0 ctx)
  h30_0 : MxxRuntime.concatRows (generatedRoot.w_28_0 ctx) (generatedRoot.w_29_0 ctx) (generatedRoot.w_30_0 ctx)
  t29 : (1) = 1
  h33_3 : (∀ i : Fin 1, parallel_generatedRoot_33 backend hashModel tape (path ++ [33, i.val]) params i ((generatedRoot.w_0_0 ctx), ((generatedRoot.w_1_0 ctx), (((generatedRoot.w_31_0 ctx) i), (((generatedRoot.w_32_0 ctx) i), ((generatedRoot.w_2_0 ctx), ()))))) (((generatedRoot.w_33_0 ctx) i), (((generatedRoot.w_33_1 ctx) i), (((generatedRoot.w_33_2 ctx) i), (((generatedRoot.w_33_3 ctx) i), ())))))
  h34_0 : MxxRuntime.familyGetStatic (generatedRoot.w_33_0 ctx) (0) (generatedRoot.w_34_0 ctx)

theorem generatedRoot.facts_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {ctx : generatedRoot.Ctx backend hashModel tape path params} {outputs : (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × Unit}
    (h : generatedRoot.constraints_0 backend hashModel tape path params (generatedRoot.w_0_0 ctx) (generatedRoot.w_1_0 ctx) (generatedRoot.w_6_0 ctx) (generatedRoot.w_31_0 ctx) (generatedRoot.w_32_0 ctx) (generatedRoot.w_2_0 ctx) (generatedRoot.w_3_0 ctx) (generatedRoot.w_3_1 ctx) (generatedRoot.w_3_2 ctx) (generatedRoot.w_3_3 ctx) (generatedRoot.w_4_0 ctx) (generatedRoot.w_4_1 ctx) (generatedRoot.w_4_2 ctx) (generatedRoot.w_4_3 ctx) (generatedRoot.w_7_0 ctx) (generatedRoot.w_9_0 ctx) (generatedRoot.w_10_0 ctx) (generatedRoot.w_12_0 ctx) (generatedRoot.w_13_0 ctx) (generatedRoot.w_14_0 ctx) (generatedRoot.w_15_0 ctx) (generatedRoot.w_17_0 ctx) (generatedRoot.w_19_0 ctx) (generatedRoot.w_20_0 ctx) (generatedRoot.w_22_0 ctx) (generatedRoot.w_23_0 ctx) (generatedRoot.w_24_0 ctx) (generatedRoot.w_25_0 ctx) (generatedRoot.w_27_0 ctx) (generatedRoot.w_28_0 ctx) (generatedRoot.w_29_0 ctx) (generatedRoot.w_30_0 ctx) (generatedRoot.w_33_0 ctx) (generatedRoot.w_33_1 ctx) (generatedRoot.w_33_2 ctx) (generatedRoot.w_33_3 ctx) (generatedRoot.w_34_0 ctx)) :
    generatedRoot.Facts_0 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Prop where
  h35_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_34_0 ctx) (generatedRoot.w_35_0 ctx)
  h36_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_35_0 ctx) (generatedRoot.w_36_0 ctx)
  h38_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 115, 116, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((generatedRoot.w_6_0 ctx)) (generatedRoot.w_38_0 ctx)
  h39_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_38_0 ctx) (generatedRoot.w_39_0 ctx)
  h41_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 115, 116, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) ((generatedRoot.w_6_0 ctx)) (generatedRoot.w_41_0 ctx)
  h42_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_41_0 ctx) (generatedRoot.w_42_0 ctx)
  t38 : 0 ≤ 0
  t39 : 0 < 1
  t40 : 1 ≤ 2
  t41 : 0 ≤ 0
  t42 : 0 < 25
  t43 : 25 ≤ 25
  h45_0 : MxxRuntime.sliceMatrix (generatedRoot.w_44_0 ctx) 0 1 0 25 (generatedRoot.w_45_0 ctx)
  t45 : 0 ≤ 1
  t46 : 1 < 2
  t47 : 2 ≤ 2
  t48 : 0 ≤ 0
  t49 : 0 < 25
  t50 : 25 ≤ 25
  h46_0 : MxxRuntime.sliceMatrix (generatedRoot.w_44_0 ctx) 1 2 0 25 (generatedRoot.w_46_0 ctx)
  h47_0 : MxxRuntime.concatRows (generatedRoot.w_45_0 ctx) (generatedRoot.w_46_0 ctx) (generatedRoot.w_47_0 ctx)
  h48_0 : MxxRuntime.familyGetStatic (generatedRoot.w_33_2 ctx) (0) (generatedRoot.w_48_0 ctx)
  h49_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_48_0 ctx) (generatedRoot.w_49_0 ctx)
  h50_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_49_0 ctx) (generatedRoot.w_50_0 ctx)
  t56 : 0 ≤ 0
  t57 : 0 < 1
  t58 : 1 ≤ 2
  t59 : 0 ≤ 0
  t60 : 0 < 25
  t61 : 25 ≤ 25
  h53_0 : MxxRuntime.sliceMatrix (generatedRoot.w_52_0 ctx) 0 1 0 25 (generatedRoot.w_53_0 ctx)
  t63 : 0 ≤ 1

theorem generatedRoot.facts_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {ctx : generatedRoot.Ctx backend hashModel tape path params} {outputs : (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × Unit}
    (h : generatedRoot.constraints_1 backend hashModel (generatedRoot.w_6_0 ctx) (generatedRoot.w_33_2 ctx) (generatedRoot.w_34_0 ctx) (generatedRoot.w_35_0 ctx) (generatedRoot.w_36_0 ctx) (generatedRoot.w_38_0 ctx) (generatedRoot.w_39_0 ctx) (generatedRoot.w_41_0 ctx) (generatedRoot.w_42_0 ctx) (generatedRoot.w_44_0 ctx) (generatedRoot.w_45_0 ctx) (generatedRoot.w_46_0 ctx) (generatedRoot.w_47_0 ctx) (generatedRoot.w_48_0 ctx) (generatedRoot.w_49_0 ctx) (generatedRoot.w_50_0 ctx) (generatedRoot.w_52_0 ctx) (generatedRoot.w_53_0 ctx)) :
    generatedRoot.Facts_1 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_2 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Prop where
  t64 : 1 < 2
  t65 : 2 ≤ 2
  t66 : 0 ≤ 0
  t67 : 0 < 25
  t68 : 25 ≤ 25
  h54_0 : MxxRuntime.sliceMatrix (generatedRoot.w_52_0 ctx) 1 2 0 25 (generatedRoot.w_54_0 ctx)
  h55_0 : MxxRuntime.familyGetStatic (generatedRoot.w_33_1 ctx) (0) (generatedRoot.w_55_0 ctx)
  h56_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_55_0 ctx) (generatedRoot.w_56_0 ctx)
  h57_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_56_0 ctx) (generatedRoot.w_57_0 ctx)
  h59_0 : MxxRuntime.familyGetStatic (generatedRoot.w_33_3 ctx) (0) (generatedRoot.w_59_0 ctx)
  h60_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_59_0 ctx) (generatedRoot.w_60_0 ctx)
  h61_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_60_0 ctx) (generatedRoot.w_61_0 ctx)
  t76 : 0 ≤ 0
  t77 : 0 < 1
  t78 : 1 ≤ 2
  t79 : 0 ≤ 0
  t80 : 0 < 25
  t81 : 25 ≤ 25
  h64_0 : MxxRuntime.sliceMatrix (generatedRoot.w_63_0 ctx) 0 1 0 25 (generatedRoot.w_64_0 ctx)
  t83 : 0 ≤ 1
  t84 : 1 < 2
  t85 : 2 ≤ 2
  t86 : 0 ≤ 0
  t87 : 0 < 25
  t88 : 25 ≤ 25
  h65_0 : MxxRuntime.sliceMatrix (generatedRoot.w_63_0 ctx) 1 2 0 25 (generatedRoot.w_65_0 ctx)
  t90 : 0 ≤ (4)
  h66_3 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_generatedRoot_66 backend hashModel tape (path ++ [66, i]) params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, ((generatedRoot.w_6_0 ctx), ((generatedRoot.w_53_0 ctx), ((generatedRoot.w_54_0 ctx), ((generatedRoot.w_3_0 ctx), ((generatedRoot.w_64_0 ctx), ((generatedRoot.w_65_0 ctx), ((generatedRoot.w_3_2 ctx), ((generatedRoot.w_3_1 ctx), ((generatedRoot.w_3_3 ctx), ()))))))))))))) next) (Int.toNat (4)) ((generatedRoot.w_5_0 ctx), ((generatedRoot.w_5_0 ctx), ((generatedRoot.w_5_0 ctx), ((generatedRoot.w_5_0 ctx), ())))) ((generatedRoot.w_66_0 ctx), ((generatedRoot.w_66_1 ctx), ((generatedRoot.w_66_2 ctx), ((generatedRoot.w_66_3 ctx), ()))))
  h69_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((generatedRoot.w_6_0 ctx)) (generatedRoot.w_69_0 ctx)
  h70_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) ((generatedRoot.w_6_0 ctx)) (generatedRoot.w_70_0 ctx)
  h71_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((generatedRoot.w_6_0 ctx)) (generatedRoot.w_71_0 ctx)
  h72_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) ((generatedRoot.w_6_0 ctx)) (generatedRoot.w_72_0 ctx)

theorem generatedRoot.facts_2 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {ctx : generatedRoot.Ctx backend hashModel tape path params} {outputs : (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × Unit}
    (h : generatedRoot.constraints_2 backend hashModel tape path params (generatedRoot.w_6_0 ctx) (generatedRoot.w_3_0 ctx) (generatedRoot.w_3_1 ctx) (generatedRoot.w_3_2 ctx) (generatedRoot.w_3_3 ctx) (generatedRoot.w_5_0 ctx) (generatedRoot.w_33_1 ctx) (generatedRoot.w_33_3 ctx) (generatedRoot.w_52_0 ctx) (generatedRoot.w_53_0 ctx) (generatedRoot.w_54_0 ctx) (generatedRoot.w_55_0 ctx) (generatedRoot.w_56_0 ctx) (generatedRoot.w_57_0 ctx) (generatedRoot.w_59_0 ctx) (generatedRoot.w_60_0 ctx) (generatedRoot.w_61_0 ctx) (generatedRoot.w_63_0 ctx) (generatedRoot.w_64_0 ctx) (generatedRoot.w_65_0 ctx) (generatedRoot.w_66_0 ctx) (generatedRoot.w_66_1 ctx) (generatedRoot.w_66_2 ctx) (generatedRoot.w_66_3 ctx) (generatedRoot.w_69_0 ctx) (generatedRoot.w_70_0 ctx) (generatedRoot.w_71_0 ctx) (generatedRoot.w_72_0 ctx)) :
    generatedRoot.Facts_2 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_3 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) (outputs : (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × Unit) : Prop where
  t96 : (16) = 16
  h75_7 : (∀ i : Fin 16, parallel_generatedRoot_75 backend hashModel tape (path ++ [75, i.val]) params i ((generatedRoot.w_6_0 ctx), (((generatedRoot.w_68_0 ctx) i), ((generatedRoot.w_69_0 ctx), ((generatedRoot.w_70_0 ctx), ((generatedRoot.w_0_0 ctx), ((generatedRoot.w_71_0 ctx), ((generatedRoot.w_72_0 ctx), ((generatedRoot.w_1_0 ctx), (((generatedRoot.w_73_0 ctx) i), (((generatedRoot.w_74_0 ctx) i), ((generatedRoot.w_2_0 ctx), ()))))))))))) (((generatedRoot.w_75_0 ctx) i), (((generatedRoot.w_75_1 ctx) i), (((generatedRoot.w_75_2 ctx) i), (((generatedRoot.w_75_3 ctx) i), (((generatedRoot.w_75_4 ctx) i), (((generatedRoot.w_75_5 ctx) i), (((generatedRoot.w_75_6 ctx) i), (((generatedRoot.w_75_7 ctx) i), ())))))))))
  t98 : (8) = 8
  h76_3 : (∀ i : Fin 8, parallel_generatedRoot_76 backend hashModel tape (path ++ [76, i.val]) params i ((generatedRoot.w_75_0 ctx), ((generatedRoot.w_75_1 ctx), ((generatedRoot.w_75_2 ctx), ((generatedRoot.w_75_3 ctx), ())))) (((generatedRoot.w_76_0 ctx) i), (((generatedRoot.w_76_1 ctx) i), (((generatedRoot.w_76_2 ctx) i), (((generatedRoot.w_76_3 ctx) i), ())))))
  t100 : (16) = 16
  h79_7 : (∀ i : Fin 16, parallel_generatedRoot_79 backend hashModel tape (path ++ [79, i.val]) params i ((generatedRoot.w_6_0 ctx), (((generatedRoot.w_67_0 ctx) i), ((generatedRoot.w_76_0 ctx), ((generatedRoot.w_76_2 ctx), ((generatedRoot.w_0_0 ctx), ((generatedRoot.w_76_1 ctx), ((generatedRoot.w_76_3 ctx), ((generatedRoot.w_1_0 ctx), (((generatedRoot.w_77_0 ctx) i), (((generatedRoot.w_78_0 ctx) i), ((generatedRoot.w_2_0 ctx), ()))))))))))) (((generatedRoot.w_79_0 ctx) i), (((generatedRoot.w_79_1 ctx) i), (((generatedRoot.w_79_2 ctx) i), (((generatedRoot.w_79_3 ctx) i), (((generatedRoot.w_79_4 ctx) i), (((generatedRoot.w_79_5 ctx) i), (((generatedRoot.w_79_6 ctx) i), (((generatedRoot.w_79_7 ctx) i), ())))))))))
  t102 : (8) = 8
  h80_3 : (∀ i : Fin 8, parallel_generatedRoot_80 backend hashModel tape (path ++ [80, i.val]) params i ((generatedRoot.w_79_0 ctx), ((generatedRoot.w_79_1 ctx), ((generatedRoot.w_79_2 ctx), ((generatedRoot.w_79_3 ctx), ())))) (((generatedRoot.w_80_0 ctx) i), (((generatedRoot.w_80_1 ctx) i), (((generatedRoot.w_80_2 ctx) i), (((generatedRoot.w_80_3 ctx) i), ())))))
  t104 : (8) = 8
  h81_3 : (∀ i : Fin 8, parallel_generatedRoot_81 backend hashModel tape (path ++ [81, i.val]) params i ((generatedRoot.w_6_0 ctx), (((generatedRoot.w_80_0 ctx) i), (((generatedRoot.w_80_2 ctx) i), ((generatedRoot.w_4_0 ctx), (((generatedRoot.w_80_1 ctx) i), (((generatedRoot.w_80_3 ctx) i), ((generatedRoot.w_4_2 ctx), ((generatedRoot.w_4_1 ctx), ((generatedRoot.w_4_3 ctx), ()))))))))) (((generatedRoot.w_81_0 ctx) i), (((generatedRoot.w_81_1 ctx) i), (((generatedRoot.w_81_2 ctx) i), (((generatedRoot.w_81_3 ctx) i), ())))))
  t106 : (8) = 8
  h82_3 : (∀ i : Fin 8, parallel_generatedRoot_82 backend hashModel tape (path ++ [82, i.val]) params i (((generatedRoot.w_81_0 ctx) i), (((generatedRoot.w_81_1 ctx) i), (((generatedRoot.w_81_2 ctx) i), (((generatedRoot.w_81_3 ctx) i), ())))) (((generatedRoot.w_82_0 ctx) i), (((generatedRoot.w_82_1 ctx) i), (((generatedRoot.w_82_2 ctx) i), (((generatedRoot.w_82_3 ctx) i), ())))))
  h83_0 : MxxRuntime.familyGetStatic (generatedRoot.w_82_0 ctx) (0) (generatedRoot.w_83_0 ctx)
  h84_0 : MxxRuntime.familyGetStatic (generatedRoot.w_82_0 ctx) (1) (generatedRoot.w_84_0 ctx)
  h86_0 : MxxRuntime.familyGetStatic (generatedRoot.w_82_1 ctx) (0) (generatedRoot.w_86_0 ctx)
  h87_0 : MxxRuntime.familyGetStatic (generatedRoot.w_82_1 ctx) (1) (generatedRoot.w_87_0 ctx)
  h89_0 : MxxRuntime.familyGetStatic (generatedRoot.w_82_2 ctx) (0) (generatedRoot.w_89_0 ctx)
  h90_0 : MxxRuntime.familyGetStatic (generatedRoot.w_82_2 ctx) (1) (generatedRoot.w_90_0 ctx)
  h92_0 : MxxRuntime.familyGetStatic (generatedRoot.w_82_3 ctx) (0) (generatedRoot.w_92_0 ctx)
  h93_0 : MxxRuntime.familyGetStatic (generatedRoot.w_82_3 ctx) (1) (generatedRoot.w_93_0 ctx)
  t116 : 0 ≤ (6)
  h95_3 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_generatedRoot_95 backend hashModel tape (path ++ [95, i]) params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, ((generatedRoot.w_82_0 ctx), ((generatedRoot.w_82_1 ctx), ((generatedRoot.w_82_2 ctx), ((generatedRoot.w_82_3 ctx), ())))))))) next) (Int.toNat (6)) ((generatedRoot.w_85_0 ctx), ((generatedRoot.w_88_0 ctx), ((generatedRoot.w_91_0 ctx), ((generatedRoot.w_94_0 ctx), ())))) ((generatedRoot.w_95_0 ctx), ((generatedRoot.w_95_1 ctx), ((generatedRoot.w_95_2 ctx), ((generatedRoot.w_95_3 ctx), ()))))
  hout : outputs = generatedRoot.outs ctx

theorem generatedRoot.facts_3 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {ctx : generatedRoot.Ctx backend hashModel tape path params} {outputs : (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × Unit}
    (h : generatedRoot.constraints_3 backend hashModel tape path params (generatedRoot.w_0_0 ctx) (generatedRoot.w_1_0 ctx) (generatedRoot.w_6_0 ctx) (generatedRoot.w_67_0 ctx) (generatedRoot.w_68_0 ctx) (generatedRoot.w_73_0 ctx) (generatedRoot.w_74_0 ctx) (generatedRoot.w_77_0 ctx) (generatedRoot.w_78_0 ctx) (generatedRoot.w_2_0 ctx) (generatedRoot.w_3_0 ctx) (generatedRoot.w_3_1 ctx) (generatedRoot.w_3_2 ctx) (generatedRoot.w_3_3 ctx) (generatedRoot.w_4_0 ctx) (generatedRoot.w_4_1 ctx) (generatedRoot.w_4_2 ctx) (generatedRoot.w_4_3 ctx) (generatedRoot.w_33_2 ctx) (generatedRoot.w_33_3 ctx) (generatedRoot.w_66_0 ctx) (generatedRoot.w_66_1 ctx) (generatedRoot.w_66_2 ctx) (generatedRoot.w_66_3 ctx) (generatedRoot.w_69_0 ctx) (generatedRoot.w_70_0 ctx) (generatedRoot.w_71_0 ctx) (generatedRoot.w_72_0 ctx) (generatedRoot.w_75_0 ctx) (generatedRoot.w_75_1 ctx) (generatedRoot.w_75_2 ctx) (generatedRoot.w_75_3 ctx) (generatedRoot.w_75_4 ctx) (generatedRoot.w_75_5 ctx) (generatedRoot.w_75_6 ctx) (generatedRoot.w_75_7 ctx) (generatedRoot.w_76_0 ctx) (generatedRoot.w_76_1 ctx) (generatedRoot.w_76_2 ctx) (generatedRoot.w_76_3 ctx) (generatedRoot.w_79_0 ctx) (generatedRoot.w_79_1 ctx) (generatedRoot.w_79_2 ctx) (generatedRoot.w_79_3 ctx) (generatedRoot.w_79_4 ctx) (generatedRoot.w_79_5 ctx) (generatedRoot.w_79_6 ctx) (generatedRoot.w_79_7 ctx) (generatedRoot.w_80_0 ctx) (generatedRoot.w_80_1 ctx) (generatedRoot.w_80_2 ctx) (generatedRoot.w_80_3 ctx) (generatedRoot.w_81_0 ctx) (generatedRoot.w_81_1 ctx) (generatedRoot.w_81_2 ctx) (generatedRoot.w_81_3 ctx) (generatedRoot.w_82_0 ctx) (generatedRoot.w_82_1 ctx) (generatedRoot.w_82_2 ctx) (generatedRoot.w_82_3 ctx) (generatedRoot.w_83_0 ctx) (generatedRoot.w_84_0 ctx) (generatedRoot.w_85_0 ctx) (generatedRoot.w_86_0 ctx) (generatedRoot.w_87_0 ctx) (generatedRoot.w_88_0 ctx) (generatedRoot.w_89_0 ctx) (generatedRoot.w_90_0 ctx) (generatedRoot.w_91_0 ctx) (generatedRoot.w_92_0 ctx) (generatedRoot.w_93_0 ctx) (generatedRoot.w_94_0 ctx) (generatedRoot.w_95_0 ctx) (generatedRoot.w_95_1 ctx) (generatedRoot.w_95_2 ctx) (generatedRoot.w_95_3 ctx) outputs) :
    generatedRoot.Facts_3 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) (outputs : (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × Unit) : Prop where
  c0 : generatedRoot.Facts_0 ctx
  c1 : generatedRoot.Facts_1 ctx
  c2 : generatedRoot.Facts_2 ctx
  c3 : generatedRoot.Facts_3 ctx outputs

theorem generatedRoot.facts {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) {outputs : (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × Unit}
    (h : generatedRoot.body backend hashModel tape path params ctx.inputs outputs ctx.witness) :
    generatedRoot.Facts ctx outputs :=
  have t0 := h
  have r0 := generatedRoot.facts_0 (ctx := ctx) (outputs := outputs) t0.1
  have t1 := t0.2
  have r1 := generatedRoot.facts_1 (ctx := ctx) (outputs := outputs) t1.1
  have t2 := t1.2
  have r2 := generatedRoot.facts_2 (ctx := ctx) (outputs := outputs) t2.1
  have t3 := t2.2
  have r3 := generatedRoot.facts_3 (ctx := ctx) (outputs := outputs) t3
  ⟨r0, r1, r2, r3⟩

theorem generatedRoot.h3_3_at {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {ctx : generatedRoot.Ctx backend hashModel tape path params} {outputs : (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × Unit}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 4) :
    ∃ w : parallel_generatedRoot_3.Witness, parallel_generatedRoot_3.Facts (⟨((generatedRoot.w_0_0 ctx), ((generatedRoot.w_1_0 ctx), ((generatedRoot.w_2_0 ctx), ()))), w⟩ : parallel_generatedRoot_3.Ctx backend hashModel tape (path ++ [3, i.val]) params i) (((generatedRoot.w_3_0 ctx) i), (((generatedRoot.w_3_1 ctx) i), (((generatedRoot.w_3_2 ctx) i), (((generatedRoot.w_3_3 ctx) i), ())))) :=
  let ⟨w, h⟩ := F.c0.h3_3 i
  ⟨w, parallel_generatedRoot_3.facts _ h⟩

theorem generatedRoot.h4_3_at {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {ctx : generatedRoot.Ctx backend hashModel tape path params} {outputs : (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × Unit}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 8) :
    ∃ w : parallel_generatedRoot_4.Witness, parallel_generatedRoot_4.Facts (⟨((generatedRoot.w_0_0 ctx), ((generatedRoot.w_1_0 ctx), ((generatedRoot.w_2_0 ctx), ()))), w⟩ : parallel_generatedRoot_4.Ctx backend hashModel tape (path ++ [4, i.val]) params i) (((generatedRoot.w_4_0 ctx) i), (((generatedRoot.w_4_1 ctx) i), (((generatedRoot.w_4_2 ctx) i), (((generatedRoot.w_4_3 ctx) i), ())))) :=
  let ⟨w, h⟩ := F.c0.h4_3 i
  ⟨w, parallel_generatedRoot_4.facts _ h⟩

theorem generatedRoot.h76_3_at {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {ctx : generatedRoot.Ctx backend hashModel tape path params} {outputs : (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × Unit}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 8) :
    ∃ w : parallel_generatedRoot_76.Witness, parallel_generatedRoot_76.Facts (⟨((generatedRoot.w_75_0 ctx), ((generatedRoot.w_75_1 ctx), ((generatedRoot.w_75_2 ctx), ((generatedRoot.w_75_3 ctx), ())))), w⟩ : parallel_generatedRoot_76.Ctx backend hashModel tape (path ++ [76, i.val]) params i) (((generatedRoot.w_76_0 ctx) i), (((generatedRoot.w_76_1 ctx) i), (((generatedRoot.w_76_2 ctx) i), (((generatedRoot.w_76_3 ctx) i), ())))) :=
  let ⟨w, h⟩ := F.c3.h76_3 i
  ⟨w, parallel_generatedRoot_76.facts _ h⟩

theorem generatedRoot.h80_3_at {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {ctx : generatedRoot.Ctx backend hashModel tape path params} {outputs : (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × Unit}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 8) :
    ∃ w : parallel_generatedRoot_80.Witness, parallel_generatedRoot_80.Facts (⟨((generatedRoot.w_79_0 ctx), ((generatedRoot.w_79_1 ctx), ((generatedRoot.w_79_2 ctx), ((generatedRoot.w_79_3 ctx), ())))), w⟩ : parallel_generatedRoot_80.Ctx backend hashModel tape (path ++ [80, i.val]) params i) (((generatedRoot.w_80_0 ctx) i), (((generatedRoot.w_80_1 ctx) i), (((generatedRoot.w_80_2 ctx) i), (((generatedRoot.w_80_3 ctx) i), ())))) :=
  let ⟨w, h⟩ := F.c3.h80_3 i
  ⟨w, parallel_generatedRoot_80.facts _ h⟩

theorem generatedRoot.h81_3_at {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {ctx : generatedRoot.Ctx backend hashModel tape path params} {outputs : (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × Unit}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 8) :
    ∃ w : parallel_generatedRoot_81.Witness, parallel_generatedRoot_81.Facts (⟨((generatedRoot.w_6_0 ctx), (((generatedRoot.w_80_0 ctx) i), (((generatedRoot.w_80_2 ctx) i), ((generatedRoot.w_4_0 ctx), (((generatedRoot.w_80_1 ctx) i), (((generatedRoot.w_80_3 ctx) i), ((generatedRoot.w_4_2 ctx), ((generatedRoot.w_4_1 ctx), ((generatedRoot.w_4_3 ctx), ()))))))))), w⟩ : parallel_generatedRoot_81.Ctx backend hashModel tape (path ++ [81, i.val]) params i) (((generatedRoot.w_81_0 ctx) i), (((generatedRoot.w_81_1 ctx) i), (((generatedRoot.w_81_2 ctx) i), (((generatedRoot.w_81_3 ctx) i), ())))) :=
  let ⟨w, h⟩ := F.c3.h81_3 i
  ⟨w, parallel_generatedRoot_81.facts _ h⟩

end Stage_preprocess_1_1

end
