import Stage_preprocess_1_1

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000
set_option linter.unusedVariables false

-- The wire types repeat the ring modulus `X ^ n + 1`; unfolding it while checking a
-- projection against its field type expands it at the ring dimension.
attribute [local irreducible] Mxx.Primitives.negacyclicModulus

noncomputable section

namespace Stage_preprocess_1_1

local notation "mxxNat0" => 34966667090521891691697534320578980119233289572661749111322581056293999333631863706749951226205037258731813033975235880421886188355853612691760742635757734639777110088925268398855525802971213260863115946668447253110267847342147835648991308630750798284189280777490844226542736101225275634281796633314346229417268180970144094013151705539447909488293483742477879658230930231096823578792363056833678376101314143030229295656153923770515332945564395417379854024076729782890141983050429450944513
local notation "mxxNat1" => 46257986671836252550474863111599275782735705997167105595187164522388936618450486362054622976333747206863960992863072466808120270012431341790141815778554503117205135221807386319319289343514000876350163804446800011927125173046383074243978085376097410230125402695222262674697161300579270891185293462822103865999927697741753124371565277119894630260554921200986361631201334784888506192777396960602887018384030168383740839045120294988077575875902898104242098552684840441948416998410463961145345
local notation "mxxMat0" => Mxx.Primitives.ExactMatrix modulus_0 65536 1 1
local notation "mxxMat1" => Mxx.Primitives.ExactMatrix modulus_0 65536 1 270
local notation "mxxMat2" => Mxx.Primitives.ExactMatrix modulus_0 65536 1 82620
local notation "mxxMat3" => Mxx.Primitives.ExactMatrix modulus_0 65536 1 306

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_3.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) where
  inputs : ByteArray × mxxMat0 × mxxMat0 × Unit
  witness : parallel_generatedRoot_3.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_3Ctx" => (parallel_generatedRoot_3.Ctx backend hashModel tape path params i_0)

def parallel_generatedRoot_3.w_0_0  (ctx : parallel_generatedRoot_3Ctx) : ByteArray :=
  ctx.inputs.1

def parallel_generatedRoot_3.w_3_0  (ctx : parallel_generatedRoot_3Ctx) : mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_3.w_26_0  (ctx : parallel_generatedRoot_3Ctx) : mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_3.w_1_0  (ctx : parallel_generatedRoot_3Ctx) : mxxMat1 :=
  ctx.witness.w_1_0

def parallel_generatedRoot_3.w_2_0  (ctx : parallel_generatedRoot_3Ctx) : mxxMat1 :=
  ctx.witness.w_2_0

def parallel_generatedRoot_3.w_4_0  (ctx : parallel_generatedRoot_3Ctx) : mxxMat0 :=
  ctx.witness.w_4_0

def parallel_generatedRoot_3.w_5_0  (ctx : parallel_generatedRoot_3Ctx) : mxxMat0 :=
  MxxRuntime.matrixNeg (parallel_generatedRoot_3.w_4_0 ctx)

def parallel_generatedRoot_3.w_6_0  (ctx : parallel_generatedRoot_3Ctx) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_3.w_5_0 ctx) (parallel_generatedRoot_3.w_1_0 ctx)

def parallel_generatedRoot_3.w_7_0  (ctx : parallel_generatedRoot_3Ctx) : mxxMat0 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat0)

def parallel_generatedRoot_3.w_8_0  (ctx : parallel_generatedRoot_3Ctx) : mxxMat0 :=
  (MxxRuntime.matrixPolynomial [(mxxNat0)] : mxxMat0)

def parallel_generatedRoot_3.w_9_0  (ctx : parallel_generatedRoot_3Ctx) : mxxMat0 :=
  (0 : mxxMat0)

def parallel_generatedRoot_3.w_10_0  (ctx : parallel_generatedRoot_3Ctx) : mxxMat0 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat0)

def parallel_generatedRoot_3.w_11_0  (ctx : parallel_generatedRoot_3Ctx) : Int :=
  -2

def parallel_generatedRoot_3.w_12_0  (ctx : parallel_generatedRoot_3Ctx) : Fin 4 → Int :=
  ctx.witness.w_12_0

def parallel_generatedRoot_3.w_13_0  (ctx : parallel_generatedRoot_3Ctx) : mxxMat0 :=
  ctx.witness.w_13_0

def parallel_generatedRoot_3.w_14_0  (ctx : parallel_generatedRoot_3Ctx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_3.w_8_0 ctx) (parallel_generatedRoot_3.w_13_0 ctx)

def parallel_generatedRoot_3.w_15_0  (ctx : parallel_generatedRoot_3Ctx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_3.w_7_0 ctx) (parallel_generatedRoot_3.w_14_0 ctx)

def parallel_generatedRoot_3.w_16_0  (ctx : parallel_generatedRoot_3Ctx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_3.w_15_0 ctx) (parallel_generatedRoot_3.w_3_0 ctx)

def parallel_generatedRoot_3.w_17_0  (ctx : parallel_generatedRoot_3Ctx) : mxxMat1 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_3.w_18_0  (ctx : parallel_generatedRoot_3Ctx) : mxxMat1 :=
  ctx.witness.w_18_0

def parallel_generatedRoot_3.w_19_0  (ctx : parallel_generatedRoot_3Ctx) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_3.w_16_0 ctx) (parallel_generatedRoot_3.w_18_0 ctx)

def parallel_generatedRoot_3.w_20_0  (ctx : parallel_generatedRoot_3Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_3.w_6_0 ctx) (parallel_generatedRoot_3.w_19_0 ctx)

def parallel_generatedRoot_3.w_21_0  (ctx : parallel_generatedRoot_3Ctx) : mxxMat1 :=
  ctx.witness.w_21_0

def parallel_generatedRoot_3.w_22_0  (ctx : parallel_generatedRoot_3Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_3.w_20_0 ctx) (parallel_generatedRoot_3.w_21_0 ctx)

def parallel_generatedRoot_3.w_23_0  (ctx : parallel_generatedRoot_3Ctx) : mxxMat0 :=
  MxxRuntime.matrixNeg (parallel_generatedRoot_3.w_4_0 ctx)

def parallel_generatedRoot_3.w_24_0  (ctx : parallel_generatedRoot_3Ctx) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_3.w_23_0 ctx) (parallel_generatedRoot_3.w_2_0 ctx)

def parallel_generatedRoot_3.w_25_0  (ctx : parallel_generatedRoot_3Ctx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_3.w_7_0 ctx) (parallel_generatedRoot_3.w_14_0 ctx)

def parallel_generatedRoot_3.w_27_0  (ctx : parallel_generatedRoot_3Ctx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_3.w_25_0 ctx) (parallel_generatedRoot_3.w_26_0 ctx)

def parallel_generatedRoot_3.w_28_0  (ctx : parallel_generatedRoot_3Ctx) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_3.w_27_0 ctx) (parallel_generatedRoot_3.w_18_0 ctx)

def parallel_generatedRoot_3.w_29_0  (ctx : parallel_generatedRoot_3Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_3.w_24_0 ctx) (parallel_generatedRoot_3.w_28_0 ctx)

def parallel_generatedRoot_3.w_30_0  (ctx : parallel_generatedRoot_3Ctx) : mxxMat1 :=
  ctx.witness.w_30_0

def parallel_generatedRoot_3.w_31_0  (ctx : parallel_generatedRoot_3Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_3.w_29_0 ctx) (parallel_generatedRoot_3.w_30_0 ctx)

def parallel_generatedRoot_3.outs  (ctx : parallel_generatedRoot_3Ctx) : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit :=
  ((parallel_generatedRoot_3.w_1_0 ctx), ((parallel_generatedRoot_3.w_2_0 ctx), ((parallel_generatedRoot_3.w_22_0 ctx), ((parallel_generatedRoot_3.w_31_0 ctx), ()))))

structure parallel_generatedRoot_3.Facts_0  (ctx : parallel_generatedRoot_3Ctx) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  t0 : 0 ≤ ((Int.ofNat i_0))
  t1 : ((Int.ofNat i_0)) < 4
  h1_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 98, 47, 76, 117, 116, 73, 100, 40, 49, 41, 47, 49, 47, 50]) ([.bytes [115, 105, 103, 109, 97], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(1), (16385), (32769), (49153)])), .bytes [98, 114, 97, 110, 99, 104], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((parallel_generatedRoot_3.w_0_0 ctx)) (parallel_generatedRoot_3.w_1_0 ctx)
  t3 : 0 ≤ ((Int.ofNat i_0))
  t4 : ((Int.ofNat i_0)) < 4
  h2_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 98, 47, 76, 117, 116, 73, 100, 40, 49, 41, 47, 49, 47, 50]) ([.bytes [115, 105, 103, 109, 97], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(1), (16385), (32769), (49153)])), .bytes [98, 114, 97, 110, 99, 104], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((parallel_generatedRoot_3.w_0_0 ctx)) (parallel_generatedRoot_3.w_2_0 ctx)
  t6 : 0 ≤ ((Int.ofNat i_0))
  t7 : ((Int.ofNat i_0)) < 4
  h4_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(1), (16385), (32769), (49153)])) (parallel_generatedRoot_3.w_3_0 ctx) (parallel_generatedRoot_3.w_4_0 ctx)
  t9 : (4) = 4
  h12_0 : (∀ i : Fin 4, parallel_parallel_generatedRoot_3_12 backend hashModel tape (path ++ [12, i.val]) params i_0 i () ((parallel_generatedRoot_3.w_12_0 ctx) i))
  t11 : 0 ≤ (4)
  h13_0 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0) => sequential_parallel_generatedRoot_3_13 backend hashModel tape (path ++ [13, i]) params i_0 i (current, ((parallel_generatedRoot_3.w_10_0 ctx), ((parallel_generatedRoot_3.w_11_0 ctx), ((parallel_generatedRoot_3.w_12_0 ctx), ())))) next) (Int.toNat (4)) (parallel_generatedRoot_3.w_9_0 ctx) (parallel_generatedRoot_3.w_13_0 ctx)
  h17_0 : MxxRuntime.gadgetMatrixRuns backend 64 270 (parallel_generatedRoot_3.w_17_0 ctx)
  h18_0 : MxxRuntime.centeredRebaseRuns (parallel_generatedRoot_3.w_17_0 ctx) (parallel_generatedRoot_3.w_18_0 ctx)
  h21_0 : MxxRuntime.gaussianSampleAt tape (path ++ [21]) ((4/1)) (26) (parallel_generatedRoot_3.w_21_0 ctx)
  h30_0 : MxxRuntime.gaussianSampleAt tape (path ++ [30]) ((4/1)) (26) (parallel_generatedRoot_3.w_30_0 ctx)
  hout : outputs = parallel_generatedRoot_3.outs ctx

theorem parallel_generatedRoot_3.facts_0  {ctx : parallel_generatedRoot_3Ctx} {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_3.constraints_0 backend hashModel tape path params i_0 (parallel_generatedRoot_3.w_0_0 ctx) (parallel_generatedRoot_3.w_3_0 ctx) (parallel_generatedRoot_3.w_1_0 ctx) (parallel_generatedRoot_3.w_2_0 ctx) (parallel_generatedRoot_3.w_4_0 ctx) (parallel_generatedRoot_3.w_9_0 ctx) (parallel_generatedRoot_3.w_10_0 ctx) (parallel_generatedRoot_3.w_11_0 ctx) (parallel_generatedRoot_3.w_12_0 ctx) (parallel_generatedRoot_3.w_13_0 ctx) (parallel_generatedRoot_3.w_17_0 ctx) (parallel_generatedRoot_3.w_18_0 ctx) (parallel_generatedRoot_3.w_21_0 ctx) (parallel_generatedRoot_3.w_22_0 ctx) (parallel_generatedRoot_3.w_30_0 ctx) (parallel_generatedRoot_3.w_31_0 ctx) outputs) :
    parallel_generatedRoot_3.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_3.Facts  (ctx : parallel_generatedRoot_3Ctx) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  c0 : parallel_generatedRoot_3.Facts_0 ctx outputs

theorem parallel_generatedRoot_3.facts  (ctx : parallel_generatedRoot_3Ctx) {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_3.body backend hashModel tape path params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_3.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_3.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_4.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) where
  inputs : ByteArray × mxxMat0 × mxxMat0 × Unit
  witness : parallel_generatedRoot_4.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_4Ctx" => (parallel_generatedRoot_4.Ctx backend hashModel tape path params i_0)

def parallel_generatedRoot_4.w_0_0  (ctx : parallel_generatedRoot_4Ctx) : ByteArray :=
  ctx.inputs.1

def parallel_generatedRoot_4.w_3_0  (ctx : parallel_generatedRoot_4Ctx) : mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_4.w_25_0  (ctx : parallel_generatedRoot_4Ctx) : mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_4.w_1_0  (ctx : parallel_generatedRoot_4Ctx) : mxxMat1 :=
  ctx.witness.w_1_0

def parallel_generatedRoot_4.w_2_0  (ctx : parallel_generatedRoot_4Ctx) : mxxMat1 :=
  ctx.witness.w_2_0

def parallel_generatedRoot_4.w_4_0  (ctx : parallel_generatedRoot_4Ctx) : mxxMat0 :=
  ctx.witness.w_4_0

def parallel_generatedRoot_4.w_5_0  (ctx : parallel_generatedRoot_4Ctx) : mxxMat0 :=
  MxxRuntime.matrixNeg (parallel_generatedRoot_4.w_4_0 ctx)

def parallel_generatedRoot_4.w_6_0  (ctx : parallel_generatedRoot_4Ctx) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_4.w_5_0 ctx) (parallel_generatedRoot_4.w_1_0 ctx)

def parallel_generatedRoot_4.w_7_0  (ctx : parallel_generatedRoot_4Ctx) : mxxMat0 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat0)

def parallel_generatedRoot_4.w_8_0  (ctx : parallel_generatedRoot_4Ctx) : mxxMat0 :=
  (MxxRuntime.matrixPolynomial [(mxxNat1)] : mxxMat0)

def parallel_generatedRoot_4.w_9_0  (ctx : parallel_generatedRoot_4Ctx) : mxxMat0 :=
  (0 : mxxMat0)

def parallel_generatedRoot_4.w_10_0  (ctx : parallel_generatedRoot_4Ctx) : Fin 128 → Int :=
  ctx.witness.w_10_0

def parallel_generatedRoot_4.w_11_0  (ctx : parallel_generatedRoot_4Ctx) : mxxMat0 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat0)

def parallel_generatedRoot_4.w_12_0  (ctx : parallel_generatedRoot_4Ctx) : Int :=
  -512

def parallel_generatedRoot_4.w_13_0  (ctx : parallel_generatedRoot_4Ctx) : mxxMat0 :=
  ctx.witness.w_13_0

def parallel_generatedRoot_4.w_14_0  (ctx : parallel_generatedRoot_4Ctx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_4.w_8_0 ctx) (parallel_generatedRoot_4.w_13_0 ctx)

def parallel_generatedRoot_4.w_15_0  (ctx : parallel_generatedRoot_4Ctx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_4.w_7_0 ctx) (parallel_generatedRoot_4.w_14_0 ctx)

def parallel_generatedRoot_4.w_16_0  (ctx : parallel_generatedRoot_4Ctx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_4.w_15_0 ctx) (parallel_generatedRoot_4.w_3_0 ctx)

def parallel_generatedRoot_4.w_17_0  (ctx : parallel_generatedRoot_4Ctx) : mxxMat1 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_4.w_18_0  (ctx : parallel_generatedRoot_4Ctx) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_4.w_16_0 ctx) (parallel_generatedRoot_4.w_17_0 ctx)

def parallel_generatedRoot_4.w_19_0  (ctx : parallel_generatedRoot_4Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_4.w_6_0 ctx) (parallel_generatedRoot_4.w_18_0 ctx)

def parallel_generatedRoot_4.w_20_0  (ctx : parallel_generatedRoot_4Ctx) : mxxMat1 :=
  ctx.witness.w_20_0

def parallel_generatedRoot_4.w_21_0  (ctx : parallel_generatedRoot_4Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_4.w_19_0 ctx) (parallel_generatedRoot_4.w_20_0 ctx)

def parallel_generatedRoot_4.w_22_0  (ctx : parallel_generatedRoot_4Ctx) : mxxMat0 :=
  MxxRuntime.matrixNeg (parallel_generatedRoot_4.w_4_0 ctx)

def parallel_generatedRoot_4.w_23_0  (ctx : parallel_generatedRoot_4Ctx) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_4.w_22_0 ctx) (parallel_generatedRoot_4.w_2_0 ctx)

def parallel_generatedRoot_4.w_24_0  (ctx : parallel_generatedRoot_4Ctx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_4.w_7_0 ctx) (parallel_generatedRoot_4.w_14_0 ctx)

def parallel_generatedRoot_4.w_26_0  (ctx : parallel_generatedRoot_4Ctx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_4.w_24_0 ctx) (parallel_generatedRoot_4.w_25_0 ctx)

def parallel_generatedRoot_4.w_27_0  (ctx : parallel_generatedRoot_4Ctx) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_4.w_26_0 ctx) (parallel_generatedRoot_4.w_17_0 ctx)

def parallel_generatedRoot_4.w_28_0  (ctx : parallel_generatedRoot_4Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_4.w_23_0 ctx) (parallel_generatedRoot_4.w_27_0 ctx)

def parallel_generatedRoot_4.w_29_0  (ctx : parallel_generatedRoot_4Ctx) : mxxMat1 :=
  ctx.witness.w_29_0

def parallel_generatedRoot_4.w_30_0  (ctx : parallel_generatedRoot_4Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_4.w_28_0 ctx) (parallel_generatedRoot_4.w_29_0 ctx)

def parallel_generatedRoot_4.outs  (ctx : parallel_generatedRoot_4Ctx) : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit :=
  ((parallel_generatedRoot_4.w_1_0 ctx), ((parallel_generatedRoot_4.w_2_0 ctx), ((parallel_generatedRoot_4.w_21_0 ctx), ((parallel_generatedRoot_4.w_30_0 ctx), ()))))

structure parallel_generatedRoot_4.Facts_0  (ctx : parallel_generatedRoot_4Ctx) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  t0 : 0 ≤ ((Int.ofNat i_0))
  t1 : ((Int.ofNat i_0)) < 128
  h1_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 98, 47, 76, 117, 116, 73, 100, 40, 48, 41, 47, 51, 52, 47, 51, 53]) ([.bytes [115, 105, 103, 109, 97], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (11473), (12771), (14561), (15147), (16575), (17415), (18103), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (34419), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (47627), (49725), (50407), (51253), (57365), (57915), (58037), (59211), (60493), (63855), (66041), (67051), (68061), (70487), (72369), (72805), (73111), (75465), (75735), (83399), (86035), (87251), (87343), (95323), (96525), (98685), (99701), (103257), (107063), (109395), (110877), (114939), (115181), (116465), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049)])), .bytes [98, 114, 97, 110, 99, 104], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((parallel_generatedRoot_4.w_0_0 ctx)) (parallel_generatedRoot_4.w_1_0 ctx)
  t3 : 0 ≤ ((Int.ofNat i_0))
  t4 : ((Int.ofNat i_0)) < 128
  h2_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 98, 47, 76, 117, 116, 73, 100, 40, 48, 41, 47, 51, 52, 47, 51, 53]) ([.bytes [115, 105, 103, 109, 97], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (11473), (12771), (14561), (15147), (16575), (17415), (18103), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (34419), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (47627), (49725), (50407), (51253), (57365), (57915), (58037), (59211), (60493), (63855), (66041), (67051), (68061), (70487), (72369), (72805), (73111), (75465), (75735), (83399), (86035), (87251), (87343), (95323), (96525), (98685), (99701), (103257), (107063), (109395), (110877), (114939), (115181), (116465), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049)])), .bytes [98, 114, 97, 110, 99, 104], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((parallel_generatedRoot_4.w_0_0 ctx)) (parallel_generatedRoot_4.w_2_0 ctx)
  t6 : 0 ≤ ((Int.ofNat i_0))
  t7 : ((Int.ofNat i_0)) < 128
  h4_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (11473), (12771), (14561), (15147), (16575), (17415), (18103), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (34419), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (47627), (49725), (50407), (51253), (57365), (57915), (58037), (59211), (60493), (63855), (66041), (67051), (68061), (70487), (72369), (72805), (73111), (75465), (75735), (83399), (86035), (87251), (87343), (95323), (96525), (98685), (99701), (103257), (107063), (109395), (110877), (114939), (115181), (116465), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049)])) (parallel_generatedRoot_4.w_3_0 ctx) (parallel_generatedRoot_4.w_4_0 ctx)
  t9 : (128) = 128
  h10_0 : (∀ i : Fin 128, parallel_parallel_generatedRoot_4_10 backend hashModel tape (path ++ [10, i.val]) params i_0 i () ((parallel_generatedRoot_4.w_10_0 ctx) i))
  t11 : 0 ≤ (128)
  h13_0 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0) => sequential_parallel_generatedRoot_4_13 backend hashModel tape (path ++ [13, i]) params i_0 i (current, ((parallel_generatedRoot_4.w_10_0 ctx), ((parallel_generatedRoot_4.w_11_0 ctx), ((parallel_generatedRoot_4.w_12_0 ctx), ())))) next) (Int.toNat (128)) (parallel_generatedRoot_4.w_9_0 ctx) (parallel_generatedRoot_4.w_13_0 ctx)
  h17_0 : MxxRuntime.gadgetMatrixRuns backend 64 270 (parallel_generatedRoot_4.w_17_0 ctx)
  h20_0 : MxxRuntime.gaussianSampleAt tape (path ++ [20]) ((4/1)) (26) (parallel_generatedRoot_4.w_20_0 ctx)
  h29_0 : MxxRuntime.gaussianSampleAt tape (path ++ [29]) ((4/1)) (26) (parallel_generatedRoot_4.w_29_0 ctx)
  hout : outputs = parallel_generatedRoot_4.outs ctx

theorem parallel_generatedRoot_4.facts_0  {ctx : parallel_generatedRoot_4Ctx} {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_4.constraints_0 backend hashModel tape path params i_0 (parallel_generatedRoot_4.w_0_0 ctx) (parallel_generatedRoot_4.w_3_0 ctx) (parallel_generatedRoot_4.w_1_0 ctx) (parallel_generatedRoot_4.w_2_0 ctx) (parallel_generatedRoot_4.w_4_0 ctx) (parallel_generatedRoot_4.w_9_0 ctx) (parallel_generatedRoot_4.w_10_0 ctx) (parallel_generatedRoot_4.w_11_0 ctx) (parallel_generatedRoot_4.w_12_0 ctx) (parallel_generatedRoot_4.w_13_0 ctx) (parallel_generatedRoot_4.w_17_0 ctx) (parallel_generatedRoot_4.w_20_0 ctx) (parallel_generatedRoot_4.w_21_0 ctx) (parallel_generatedRoot_4.w_29_0 ctx) (parallel_generatedRoot_4.w_30_0 ctx) outputs) :
    parallel_generatedRoot_4.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_4.Facts  (ctx : parallel_generatedRoot_4Ctx) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  c0 : parallel_generatedRoot_4.Facts_0 ctx outputs

theorem parallel_generatedRoot_4.facts  (ctx : parallel_generatedRoot_4Ctx) {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_4.body backend hashModel tape path params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_4.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_4.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_109.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) where
  inputs : (Fin 2607022080 → mxxMat1) × (Fin 2607022080 → mxxMat1) × (Fin 2607022080 → mxxMat1) × (Fin 2607022080 → mxxMat1) × Unit
  witness : parallel_generatedRoot_109.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_109Ctx" => (parallel_generatedRoot_109.Ctx backend hashModel tape path params i_0)

def parallel_generatedRoot_109.w_0_0  (ctx : parallel_generatedRoot_109Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.1

def parallel_generatedRoot_109.w_1_0  (ctx : parallel_generatedRoot_109Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.2.1

def parallel_generatedRoot_109.w_2_0  (ctx : parallel_generatedRoot_109Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_109.w_3_0  (ctx : parallel_generatedRoot_109Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_109.w_4_0  (ctx : parallel_generatedRoot_109Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_0

def parallel_generatedRoot_109.w_4_1  (ctx : parallel_generatedRoot_109Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_1

def parallel_generatedRoot_109.w_4_2  (ctx : parallel_generatedRoot_109Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_2

def parallel_generatedRoot_109.w_4_3  (ctx : parallel_generatedRoot_109Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_3

def parallel_generatedRoot_109.w_5_0  (ctx : parallel_generatedRoot_109Ctx) : mxxMat1 :=
  ctx.witness.w_5_0

def parallel_generatedRoot_109.w_6_0  (ctx : parallel_generatedRoot_109Ctx) : mxxMat1 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_109.w_7_0  (ctx : parallel_generatedRoot_109Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_109.w_5_0 ctx) (parallel_generatedRoot_109.w_6_0 ctx)

def parallel_generatedRoot_109.w_8_0  (ctx : parallel_generatedRoot_109Ctx) : mxxMat1 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_109.w_9_0  (ctx : parallel_generatedRoot_109Ctx) : mxxMat1 :=
  ctx.witness.w_9_0

def parallel_generatedRoot_109.w_10_0  (ctx : parallel_generatedRoot_109Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_109.w_8_0 ctx) (parallel_generatedRoot_109.w_9_0 ctx)

def parallel_generatedRoot_109.w_11_0  (ctx : parallel_generatedRoot_109Ctx) : mxxMat1 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_109.w_12_0  (ctx : parallel_generatedRoot_109Ctx) : mxxMat1 :=
  ctx.witness.w_12_0

def parallel_generatedRoot_109.w_13_0  (ctx : parallel_generatedRoot_109Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_109.w_11_0 ctx) (parallel_generatedRoot_109.w_12_0 ctx)

def parallel_generatedRoot_109.w_14_0  (ctx : parallel_generatedRoot_109Ctx) : mxxMat1 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_109.w_15_0  (ctx : parallel_generatedRoot_109Ctx) : mxxMat1 :=
  ctx.witness.w_15_0

def parallel_generatedRoot_109.w_16_0  (ctx : parallel_generatedRoot_109Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_109.w_14_0 ctx) (parallel_generatedRoot_109.w_15_0 ctx)

def parallel_generatedRoot_109.w_17_0  (ctx : parallel_generatedRoot_109Ctx) : mxxMat1 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_109.w_17_1  (ctx : parallel_generatedRoot_109Ctx) : mxxMat1 :=
  ctx.witness.w_17_1

def parallel_generatedRoot_109.w_17_2  (ctx : parallel_generatedRoot_109Ctx) : mxxMat1 :=
  ctx.witness.w_17_2

def parallel_generatedRoot_109.w_17_3  (ctx : parallel_generatedRoot_109Ctx) : mxxMat1 :=
  ctx.witness.w_17_3

def parallel_generatedRoot_109.outs  (ctx : parallel_generatedRoot_109Ctx) : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit :=
  ((parallel_generatedRoot_109.w_17_0 ctx), ((parallel_generatedRoot_109.w_17_1 ctx), ((parallel_generatedRoot_109.w_17_2 ctx), ((parallel_generatedRoot_109.w_17_3 ctx), ()))))

structure parallel_generatedRoot_109.Facts_0  (ctx : parallel_generatedRoot_109Ctx) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  t0 : (130) = 130
  h4_3 : (∀ i : Fin 130, parallel_parallel_generatedRoot_109_4 backend hashModel tape (path ++ [4, i.val]) params i_0 i ((parallel_generatedRoot_109.w_0_0 ctx), ((parallel_generatedRoot_109.w_1_0 ctx), ((parallel_generatedRoot_109.w_2_0 ctx), ((parallel_generatedRoot_109.w_3_0 ctx), ())))) (((parallel_generatedRoot_109.w_4_0 ctx) i), (((parallel_generatedRoot_109.w_4_1 ctx) i), (((parallel_generatedRoot_109.w_4_2 ctx) i), (((parallel_generatedRoot_109.w_4_3 ctx) i), ())))))
  h5_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_109.w_4_0 ctx) (0) (parallel_generatedRoot_109.w_5_0 ctx)
  h6_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_109.w_4_0 ctx) (1) (parallel_generatedRoot_109.w_6_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_109.w_4_1 ctx) (0) (parallel_generatedRoot_109.w_8_0 ctx)
  h9_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_109.w_4_1 ctx) (1) (parallel_generatedRoot_109.w_9_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_109.w_4_2 ctx) (0) (parallel_generatedRoot_109.w_11_0 ctx)
  h12_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_109.w_4_2 ctx) (1) (parallel_generatedRoot_109.w_12_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_109.w_4_3 ctx) (0) (parallel_generatedRoot_109.w_14_0 ctx)
  h15_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_109.w_4_3 ctx) (1) (parallel_generatedRoot_109.w_15_0 ctx)
  t10 : 0 ≤ (128)
  h17_3 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_parallel_generatedRoot_109_17 backend hashModel tape (path ++ [17, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, ((parallel_generatedRoot_109.w_4_0 ctx), ((parallel_generatedRoot_109.w_4_1 ctx), ((parallel_generatedRoot_109.w_4_2 ctx), ((parallel_generatedRoot_109.w_4_3 ctx), ())))))))) next) (Int.toNat (128)) ((parallel_generatedRoot_109.w_7_0 ctx), ((parallel_generatedRoot_109.w_10_0 ctx), ((parallel_generatedRoot_109.w_13_0 ctx), ((parallel_generatedRoot_109.w_16_0 ctx), ())))) ((parallel_generatedRoot_109.w_17_0 ctx), ((parallel_generatedRoot_109.w_17_1 ctx), ((parallel_generatedRoot_109.w_17_2 ctx), ((parallel_generatedRoot_109.w_17_3 ctx), ()))))
  hout : outputs = parallel_generatedRoot_109.outs ctx

theorem parallel_generatedRoot_109.facts_0  {ctx : parallel_generatedRoot_109Ctx} {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_109.constraints_0 backend hashModel tape path params i_0 (parallel_generatedRoot_109.w_0_0 ctx) (parallel_generatedRoot_109.w_1_0 ctx) (parallel_generatedRoot_109.w_2_0 ctx) (parallel_generatedRoot_109.w_3_0 ctx) (parallel_generatedRoot_109.w_4_0 ctx) (parallel_generatedRoot_109.w_4_1 ctx) (parallel_generatedRoot_109.w_4_2 ctx) (parallel_generatedRoot_109.w_4_3 ctx) (parallel_generatedRoot_109.w_5_0 ctx) (parallel_generatedRoot_109.w_6_0 ctx) (parallel_generatedRoot_109.w_7_0 ctx) (parallel_generatedRoot_109.w_8_0 ctx) (parallel_generatedRoot_109.w_9_0 ctx) (parallel_generatedRoot_109.w_10_0 ctx) (parallel_generatedRoot_109.w_11_0 ctx) (parallel_generatedRoot_109.w_12_0 ctx) (parallel_generatedRoot_109.w_13_0 ctx) (parallel_generatedRoot_109.w_14_0 ctx) (parallel_generatedRoot_109.w_15_0 ctx) (parallel_generatedRoot_109.w_16_0 ctx) (parallel_generatedRoot_109.w_17_0 ctx) (parallel_generatedRoot_109.w_17_1 ctx) (parallel_generatedRoot_109.w_17_2 ctx) (parallel_generatedRoot_109.w_17_3 ctx) outputs) :
    parallel_generatedRoot_109.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_109.Facts  (ctx : parallel_generatedRoot_109Ctx) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  c0 : parallel_generatedRoot_109.Facts_0 ctx outputs

theorem parallel_generatedRoot_109.facts  (ctx : parallel_generatedRoot_109Ctx) {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_109.body backend hashModel tape path params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_109.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_109.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_113.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) where
  inputs : (Fin 2607022080 → mxxMat1) × (Fin 2607022080 → mxxMat1) × (Fin 2607022080 → mxxMat1) × (Fin 2607022080 → mxxMat1) × Unit
  witness : parallel_generatedRoot_113.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_113Ctx" => (parallel_generatedRoot_113.Ctx backend hashModel tape path params i_0)

def parallel_generatedRoot_113.w_0_0  (ctx : parallel_generatedRoot_113Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.1

def parallel_generatedRoot_113.w_1_0  (ctx : parallel_generatedRoot_113Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.2.1

def parallel_generatedRoot_113.w_2_0  (ctx : parallel_generatedRoot_113Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_113.w_3_0  (ctx : parallel_generatedRoot_113Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_113.w_4_0  (ctx : parallel_generatedRoot_113Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_0

def parallel_generatedRoot_113.w_4_1  (ctx : parallel_generatedRoot_113Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_1

def parallel_generatedRoot_113.w_4_2  (ctx : parallel_generatedRoot_113Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_2

def parallel_generatedRoot_113.w_4_3  (ctx : parallel_generatedRoot_113Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_3

def parallel_generatedRoot_113.w_5_0  (ctx : parallel_generatedRoot_113Ctx) : mxxMat1 :=
  ctx.witness.w_5_0

def parallel_generatedRoot_113.w_6_0  (ctx : parallel_generatedRoot_113Ctx) : mxxMat1 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_113.w_7_0  (ctx : parallel_generatedRoot_113Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_113.w_5_0 ctx) (parallel_generatedRoot_113.w_6_0 ctx)

def parallel_generatedRoot_113.w_8_0  (ctx : parallel_generatedRoot_113Ctx) : mxxMat1 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_113.w_9_0  (ctx : parallel_generatedRoot_113Ctx) : mxxMat1 :=
  ctx.witness.w_9_0

def parallel_generatedRoot_113.w_10_0  (ctx : parallel_generatedRoot_113Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_113.w_8_0 ctx) (parallel_generatedRoot_113.w_9_0 ctx)

def parallel_generatedRoot_113.w_11_0  (ctx : parallel_generatedRoot_113Ctx) : mxxMat1 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_113.w_12_0  (ctx : parallel_generatedRoot_113Ctx) : mxxMat1 :=
  ctx.witness.w_12_0

def parallel_generatedRoot_113.w_13_0  (ctx : parallel_generatedRoot_113Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_113.w_11_0 ctx) (parallel_generatedRoot_113.w_12_0 ctx)

def parallel_generatedRoot_113.w_14_0  (ctx : parallel_generatedRoot_113Ctx) : mxxMat1 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_113.w_15_0  (ctx : parallel_generatedRoot_113Ctx) : mxxMat1 :=
  ctx.witness.w_15_0

def parallel_generatedRoot_113.w_16_0  (ctx : parallel_generatedRoot_113Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_113.w_14_0 ctx) (parallel_generatedRoot_113.w_15_0 ctx)

def parallel_generatedRoot_113.w_17_0  (ctx : parallel_generatedRoot_113Ctx) : mxxMat1 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_113.w_17_1  (ctx : parallel_generatedRoot_113Ctx) : mxxMat1 :=
  ctx.witness.w_17_1

def parallel_generatedRoot_113.w_17_2  (ctx : parallel_generatedRoot_113Ctx) : mxxMat1 :=
  ctx.witness.w_17_2

def parallel_generatedRoot_113.w_17_3  (ctx : parallel_generatedRoot_113Ctx) : mxxMat1 :=
  ctx.witness.w_17_3

def parallel_generatedRoot_113.outs  (ctx : parallel_generatedRoot_113Ctx) : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit :=
  ((parallel_generatedRoot_113.w_17_0 ctx), ((parallel_generatedRoot_113.w_17_1 ctx), ((parallel_generatedRoot_113.w_17_2 ctx), ((parallel_generatedRoot_113.w_17_3 ctx), ()))))

structure parallel_generatedRoot_113.Facts_0  (ctx : parallel_generatedRoot_113Ctx) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  t0 : (130) = 130
  h4_3 : (∀ i : Fin 130, parallel_parallel_generatedRoot_113_4 backend hashModel tape (path ++ [4, i.val]) params i_0 i ((parallel_generatedRoot_113.w_0_0 ctx), ((parallel_generatedRoot_113.w_1_0 ctx), ((parallel_generatedRoot_113.w_2_0 ctx), ((parallel_generatedRoot_113.w_3_0 ctx), ())))) (((parallel_generatedRoot_113.w_4_0 ctx) i), (((parallel_generatedRoot_113.w_4_1 ctx) i), (((parallel_generatedRoot_113.w_4_2 ctx) i), (((parallel_generatedRoot_113.w_4_3 ctx) i), ())))))
  h5_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_113.w_4_0 ctx) (0) (parallel_generatedRoot_113.w_5_0 ctx)
  h6_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_113.w_4_0 ctx) (1) (parallel_generatedRoot_113.w_6_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_113.w_4_1 ctx) (0) (parallel_generatedRoot_113.w_8_0 ctx)
  h9_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_113.w_4_1 ctx) (1) (parallel_generatedRoot_113.w_9_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_113.w_4_2 ctx) (0) (parallel_generatedRoot_113.w_11_0 ctx)
  h12_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_113.w_4_2 ctx) (1) (parallel_generatedRoot_113.w_12_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_113.w_4_3 ctx) (0) (parallel_generatedRoot_113.w_14_0 ctx)
  h15_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_113.w_4_3 ctx) (1) (parallel_generatedRoot_113.w_15_0 ctx)
  t10 : 0 ≤ (128)
  h17_3 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_parallel_generatedRoot_113_17 backend hashModel tape (path ++ [17, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, ((parallel_generatedRoot_113.w_4_0 ctx), ((parallel_generatedRoot_113.w_4_1 ctx), ((parallel_generatedRoot_113.w_4_2 ctx), ((parallel_generatedRoot_113.w_4_3 ctx), ())))))))) next) (Int.toNat (128)) ((parallel_generatedRoot_113.w_7_0 ctx), ((parallel_generatedRoot_113.w_10_0 ctx), ((parallel_generatedRoot_113.w_13_0 ctx), ((parallel_generatedRoot_113.w_16_0 ctx), ())))) ((parallel_generatedRoot_113.w_17_0 ctx), ((parallel_generatedRoot_113.w_17_1 ctx), ((parallel_generatedRoot_113.w_17_2 ctx), ((parallel_generatedRoot_113.w_17_3 ctx), ()))))
  hout : outputs = parallel_generatedRoot_113.outs ctx

theorem parallel_generatedRoot_113.facts_0  {ctx : parallel_generatedRoot_113Ctx} {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_113.constraints_0 backend hashModel tape path params i_0 (parallel_generatedRoot_113.w_0_0 ctx) (parallel_generatedRoot_113.w_1_0 ctx) (parallel_generatedRoot_113.w_2_0 ctx) (parallel_generatedRoot_113.w_3_0 ctx) (parallel_generatedRoot_113.w_4_0 ctx) (parallel_generatedRoot_113.w_4_1 ctx) (parallel_generatedRoot_113.w_4_2 ctx) (parallel_generatedRoot_113.w_4_3 ctx) (parallel_generatedRoot_113.w_5_0 ctx) (parallel_generatedRoot_113.w_6_0 ctx) (parallel_generatedRoot_113.w_7_0 ctx) (parallel_generatedRoot_113.w_8_0 ctx) (parallel_generatedRoot_113.w_9_0 ctx) (parallel_generatedRoot_113.w_10_0 ctx) (parallel_generatedRoot_113.w_11_0 ctx) (parallel_generatedRoot_113.w_12_0 ctx) (parallel_generatedRoot_113.w_13_0 ctx) (parallel_generatedRoot_113.w_14_0 ctx) (parallel_generatedRoot_113.w_15_0 ctx) (parallel_generatedRoot_113.w_16_0 ctx) (parallel_generatedRoot_113.w_17_0 ctx) (parallel_generatedRoot_113.w_17_1 ctx) (parallel_generatedRoot_113.w_17_2 ctx) (parallel_generatedRoot_113.w_17_3 ctx) outputs) :
    parallel_generatedRoot_113.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_113.Facts  (ctx : parallel_generatedRoot_113Ctx) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  c0 : parallel_generatedRoot_113.Facts_0 ctx outputs

theorem parallel_generatedRoot_113.facts  (ctx : parallel_generatedRoot_113Ctx) {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_113.body backend hashModel tape path params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_113.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_113.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_117.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) where
  inputs : (Fin 2607022080 → mxxMat1) × (Fin 2607022080 → mxxMat1) × (Fin 2607022080 → mxxMat1) × (Fin 2607022080 → mxxMat1) × Unit
  witness : parallel_generatedRoot_117.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_117Ctx" => (parallel_generatedRoot_117.Ctx backend hashModel tape path params i_0)

def parallel_generatedRoot_117.w_0_0  (ctx : parallel_generatedRoot_117Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.1

def parallel_generatedRoot_117.w_1_0  (ctx : parallel_generatedRoot_117Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.2.1

def parallel_generatedRoot_117.w_2_0  (ctx : parallel_generatedRoot_117Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_117.w_3_0  (ctx : parallel_generatedRoot_117Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_117.w_4_0  (ctx : parallel_generatedRoot_117Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_0

def parallel_generatedRoot_117.w_4_1  (ctx : parallel_generatedRoot_117Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_1

def parallel_generatedRoot_117.w_4_2  (ctx : parallel_generatedRoot_117Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_2

def parallel_generatedRoot_117.w_4_3  (ctx : parallel_generatedRoot_117Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_3

def parallel_generatedRoot_117.w_5_0  (ctx : parallel_generatedRoot_117Ctx) : mxxMat1 :=
  ctx.witness.w_5_0

def parallel_generatedRoot_117.w_6_0  (ctx : parallel_generatedRoot_117Ctx) : mxxMat1 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_117.w_7_0  (ctx : parallel_generatedRoot_117Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_117.w_5_0 ctx) (parallel_generatedRoot_117.w_6_0 ctx)

def parallel_generatedRoot_117.w_8_0  (ctx : parallel_generatedRoot_117Ctx) : mxxMat1 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_117.w_9_0  (ctx : parallel_generatedRoot_117Ctx) : mxxMat1 :=
  ctx.witness.w_9_0

def parallel_generatedRoot_117.w_10_0  (ctx : parallel_generatedRoot_117Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_117.w_8_0 ctx) (parallel_generatedRoot_117.w_9_0 ctx)

def parallel_generatedRoot_117.w_11_0  (ctx : parallel_generatedRoot_117Ctx) : mxxMat1 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_117.w_12_0  (ctx : parallel_generatedRoot_117Ctx) : mxxMat1 :=
  ctx.witness.w_12_0

def parallel_generatedRoot_117.w_13_0  (ctx : parallel_generatedRoot_117Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_117.w_11_0 ctx) (parallel_generatedRoot_117.w_12_0 ctx)

def parallel_generatedRoot_117.w_14_0  (ctx : parallel_generatedRoot_117Ctx) : mxxMat1 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_117.w_15_0  (ctx : parallel_generatedRoot_117Ctx) : mxxMat1 :=
  ctx.witness.w_15_0

def parallel_generatedRoot_117.w_16_0  (ctx : parallel_generatedRoot_117Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_117.w_14_0 ctx) (parallel_generatedRoot_117.w_15_0 ctx)

def parallel_generatedRoot_117.w_17_0  (ctx : parallel_generatedRoot_117Ctx) : mxxMat1 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_117.w_17_1  (ctx : parallel_generatedRoot_117Ctx) : mxxMat1 :=
  ctx.witness.w_17_1

def parallel_generatedRoot_117.w_17_2  (ctx : parallel_generatedRoot_117Ctx) : mxxMat1 :=
  ctx.witness.w_17_2

def parallel_generatedRoot_117.w_17_3  (ctx : parallel_generatedRoot_117Ctx) : mxxMat1 :=
  ctx.witness.w_17_3

def parallel_generatedRoot_117.outs  (ctx : parallel_generatedRoot_117Ctx) : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit :=
  ((parallel_generatedRoot_117.w_17_0 ctx), ((parallel_generatedRoot_117.w_17_1 ctx), ((parallel_generatedRoot_117.w_17_2 ctx), ((parallel_generatedRoot_117.w_17_3 ctx), ()))))

structure parallel_generatedRoot_117.Facts_0  (ctx : parallel_generatedRoot_117Ctx) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  t0 : (130) = 130
  h4_3 : (∀ i : Fin 130, parallel_parallel_generatedRoot_117_4 backend hashModel tape (path ++ [4, i.val]) params i_0 i ((parallel_generatedRoot_117.w_0_0 ctx), ((parallel_generatedRoot_117.w_1_0 ctx), ((parallel_generatedRoot_117.w_2_0 ctx), ((parallel_generatedRoot_117.w_3_0 ctx), ())))) (((parallel_generatedRoot_117.w_4_0 ctx) i), (((parallel_generatedRoot_117.w_4_1 ctx) i), (((parallel_generatedRoot_117.w_4_2 ctx) i), (((parallel_generatedRoot_117.w_4_3 ctx) i), ())))))
  h5_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_117.w_4_0 ctx) (0) (parallel_generatedRoot_117.w_5_0 ctx)
  h6_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_117.w_4_0 ctx) (1) (parallel_generatedRoot_117.w_6_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_117.w_4_1 ctx) (0) (parallel_generatedRoot_117.w_8_0 ctx)
  h9_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_117.w_4_1 ctx) (1) (parallel_generatedRoot_117.w_9_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_117.w_4_2 ctx) (0) (parallel_generatedRoot_117.w_11_0 ctx)
  h12_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_117.w_4_2 ctx) (1) (parallel_generatedRoot_117.w_12_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_117.w_4_3 ctx) (0) (parallel_generatedRoot_117.w_14_0 ctx)
  h15_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_117.w_4_3 ctx) (1) (parallel_generatedRoot_117.w_15_0 ctx)
  t10 : 0 ≤ (128)
  h17_3 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_parallel_generatedRoot_117_17 backend hashModel tape (path ++ [17, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, ((parallel_generatedRoot_117.w_4_0 ctx), ((parallel_generatedRoot_117.w_4_1 ctx), ((parallel_generatedRoot_117.w_4_2 ctx), ((parallel_generatedRoot_117.w_4_3 ctx), ())))))))) next) (Int.toNat (128)) ((parallel_generatedRoot_117.w_7_0 ctx), ((parallel_generatedRoot_117.w_10_0 ctx), ((parallel_generatedRoot_117.w_13_0 ctx), ((parallel_generatedRoot_117.w_16_0 ctx), ())))) ((parallel_generatedRoot_117.w_17_0 ctx), ((parallel_generatedRoot_117.w_17_1 ctx), ((parallel_generatedRoot_117.w_17_2 ctx), ((parallel_generatedRoot_117.w_17_3 ctx), ()))))
  hout : outputs = parallel_generatedRoot_117.outs ctx

theorem parallel_generatedRoot_117.facts_0  {ctx : parallel_generatedRoot_117Ctx} {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_117.constraints_0 backend hashModel tape path params i_0 (parallel_generatedRoot_117.w_0_0 ctx) (parallel_generatedRoot_117.w_1_0 ctx) (parallel_generatedRoot_117.w_2_0 ctx) (parallel_generatedRoot_117.w_3_0 ctx) (parallel_generatedRoot_117.w_4_0 ctx) (parallel_generatedRoot_117.w_4_1 ctx) (parallel_generatedRoot_117.w_4_2 ctx) (parallel_generatedRoot_117.w_4_3 ctx) (parallel_generatedRoot_117.w_5_0 ctx) (parallel_generatedRoot_117.w_6_0 ctx) (parallel_generatedRoot_117.w_7_0 ctx) (parallel_generatedRoot_117.w_8_0 ctx) (parallel_generatedRoot_117.w_9_0 ctx) (parallel_generatedRoot_117.w_10_0 ctx) (parallel_generatedRoot_117.w_11_0 ctx) (parallel_generatedRoot_117.w_12_0 ctx) (parallel_generatedRoot_117.w_13_0 ctx) (parallel_generatedRoot_117.w_14_0 ctx) (parallel_generatedRoot_117.w_15_0 ctx) (parallel_generatedRoot_117.w_16_0 ctx) (parallel_generatedRoot_117.w_17_0 ctx) (parallel_generatedRoot_117.w_17_1 ctx) (parallel_generatedRoot_117.w_17_2 ctx) (parallel_generatedRoot_117.w_17_3 ctx) outputs) :
    parallel_generatedRoot_117.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_117.Facts  (ctx : parallel_generatedRoot_117Ctx) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  c0 : parallel_generatedRoot_117.Facts_0 ctx outputs

theorem parallel_generatedRoot_117.facts  (ctx : parallel_generatedRoot_117Ctx) {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_117.body backend hashModel tape path params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_117.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_117.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_121.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) where
  inputs : (Fin 2607022080 → mxxMat1) × (Fin 2607022080 → mxxMat1) × (Fin 2607022080 → mxxMat1) × (Fin 2607022080 → mxxMat1) × Unit
  witness : parallel_generatedRoot_121.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_121Ctx" => (parallel_generatedRoot_121.Ctx backend hashModel tape path params i_0)

def parallel_generatedRoot_121.w_0_0  (ctx : parallel_generatedRoot_121Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.1

def parallel_generatedRoot_121.w_1_0  (ctx : parallel_generatedRoot_121Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.2.1

def parallel_generatedRoot_121.w_2_0  (ctx : parallel_generatedRoot_121Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_121.w_3_0  (ctx : parallel_generatedRoot_121Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_121.w_4_0  (ctx : parallel_generatedRoot_121Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_0

def parallel_generatedRoot_121.w_4_1  (ctx : parallel_generatedRoot_121Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_1

def parallel_generatedRoot_121.w_4_2  (ctx : parallel_generatedRoot_121Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_2

def parallel_generatedRoot_121.w_4_3  (ctx : parallel_generatedRoot_121Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_3

def parallel_generatedRoot_121.w_5_0  (ctx : parallel_generatedRoot_121Ctx) : mxxMat1 :=
  ctx.witness.w_5_0

def parallel_generatedRoot_121.w_6_0  (ctx : parallel_generatedRoot_121Ctx) : mxxMat1 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_121.w_7_0  (ctx : parallel_generatedRoot_121Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_121.w_5_0 ctx) (parallel_generatedRoot_121.w_6_0 ctx)

def parallel_generatedRoot_121.w_8_0  (ctx : parallel_generatedRoot_121Ctx) : mxxMat1 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_121.w_9_0  (ctx : parallel_generatedRoot_121Ctx) : mxxMat1 :=
  ctx.witness.w_9_0

def parallel_generatedRoot_121.w_10_0  (ctx : parallel_generatedRoot_121Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_121.w_8_0 ctx) (parallel_generatedRoot_121.w_9_0 ctx)

def parallel_generatedRoot_121.w_11_0  (ctx : parallel_generatedRoot_121Ctx) : mxxMat1 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_121.w_12_0  (ctx : parallel_generatedRoot_121Ctx) : mxxMat1 :=
  ctx.witness.w_12_0

def parallel_generatedRoot_121.w_13_0  (ctx : parallel_generatedRoot_121Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_121.w_11_0 ctx) (parallel_generatedRoot_121.w_12_0 ctx)

def parallel_generatedRoot_121.w_14_0  (ctx : parallel_generatedRoot_121Ctx) : mxxMat1 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_121.w_15_0  (ctx : parallel_generatedRoot_121Ctx) : mxxMat1 :=
  ctx.witness.w_15_0

def parallel_generatedRoot_121.w_16_0  (ctx : parallel_generatedRoot_121Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_121.w_14_0 ctx) (parallel_generatedRoot_121.w_15_0 ctx)

def parallel_generatedRoot_121.w_17_0  (ctx : parallel_generatedRoot_121Ctx) : mxxMat1 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_121.w_17_1  (ctx : parallel_generatedRoot_121Ctx) : mxxMat1 :=
  ctx.witness.w_17_1

def parallel_generatedRoot_121.w_17_2  (ctx : parallel_generatedRoot_121Ctx) : mxxMat1 :=
  ctx.witness.w_17_2

def parallel_generatedRoot_121.w_17_3  (ctx : parallel_generatedRoot_121Ctx) : mxxMat1 :=
  ctx.witness.w_17_3

def parallel_generatedRoot_121.outs  (ctx : parallel_generatedRoot_121Ctx) : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit :=
  ((parallel_generatedRoot_121.w_17_0 ctx), ((parallel_generatedRoot_121.w_17_1 ctx), ((parallel_generatedRoot_121.w_17_2 ctx), ((parallel_generatedRoot_121.w_17_3 ctx), ()))))

structure parallel_generatedRoot_121.Facts_0  (ctx : parallel_generatedRoot_121Ctx) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  t0 : (130) = 130
  h4_3 : (∀ i : Fin 130, parallel_parallel_generatedRoot_121_4 backend hashModel tape (path ++ [4, i.val]) params i_0 i ((parallel_generatedRoot_121.w_0_0 ctx), ((parallel_generatedRoot_121.w_1_0 ctx), ((parallel_generatedRoot_121.w_2_0 ctx), ((parallel_generatedRoot_121.w_3_0 ctx), ())))) (((parallel_generatedRoot_121.w_4_0 ctx) i), (((parallel_generatedRoot_121.w_4_1 ctx) i), (((parallel_generatedRoot_121.w_4_2 ctx) i), (((parallel_generatedRoot_121.w_4_3 ctx) i), ())))))
  h5_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_121.w_4_0 ctx) (0) (parallel_generatedRoot_121.w_5_0 ctx)
  h6_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_121.w_4_0 ctx) (1) (parallel_generatedRoot_121.w_6_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_121.w_4_1 ctx) (0) (parallel_generatedRoot_121.w_8_0 ctx)
  h9_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_121.w_4_1 ctx) (1) (parallel_generatedRoot_121.w_9_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_121.w_4_2 ctx) (0) (parallel_generatedRoot_121.w_11_0 ctx)
  h12_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_121.w_4_2 ctx) (1) (parallel_generatedRoot_121.w_12_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_121.w_4_3 ctx) (0) (parallel_generatedRoot_121.w_14_0 ctx)
  h15_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_121.w_4_3 ctx) (1) (parallel_generatedRoot_121.w_15_0 ctx)
  t10 : 0 ≤ (128)
  h17_3 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_parallel_generatedRoot_121_17 backend hashModel tape (path ++ [17, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, ((parallel_generatedRoot_121.w_4_0 ctx), ((parallel_generatedRoot_121.w_4_1 ctx), ((parallel_generatedRoot_121.w_4_2 ctx), ((parallel_generatedRoot_121.w_4_3 ctx), ())))))))) next) (Int.toNat (128)) ((parallel_generatedRoot_121.w_7_0 ctx), ((parallel_generatedRoot_121.w_10_0 ctx), ((parallel_generatedRoot_121.w_13_0 ctx), ((parallel_generatedRoot_121.w_16_0 ctx), ())))) ((parallel_generatedRoot_121.w_17_0 ctx), ((parallel_generatedRoot_121.w_17_1 ctx), ((parallel_generatedRoot_121.w_17_2 ctx), ((parallel_generatedRoot_121.w_17_3 ctx), ()))))
  hout : outputs = parallel_generatedRoot_121.outs ctx

theorem parallel_generatedRoot_121.facts_0  {ctx : parallel_generatedRoot_121Ctx} {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_121.constraints_0 backend hashModel tape path params i_0 (parallel_generatedRoot_121.w_0_0 ctx) (parallel_generatedRoot_121.w_1_0 ctx) (parallel_generatedRoot_121.w_2_0 ctx) (parallel_generatedRoot_121.w_3_0 ctx) (parallel_generatedRoot_121.w_4_0 ctx) (parallel_generatedRoot_121.w_4_1 ctx) (parallel_generatedRoot_121.w_4_2 ctx) (parallel_generatedRoot_121.w_4_3 ctx) (parallel_generatedRoot_121.w_5_0 ctx) (parallel_generatedRoot_121.w_6_0 ctx) (parallel_generatedRoot_121.w_7_0 ctx) (parallel_generatedRoot_121.w_8_0 ctx) (parallel_generatedRoot_121.w_9_0 ctx) (parallel_generatedRoot_121.w_10_0 ctx) (parallel_generatedRoot_121.w_11_0 ctx) (parallel_generatedRoot_121.w_12_0 ctx) (parallel_generatedRoot_121.w_13_0 ctx) (parallel_generatedRoot_121.w_14_0 ctx) (parallel_generatedRoot_121.w_15_0 ctx) (parallel_generatedRoot_121.w_16_0 ctx) (parallel_generatedRoot_121.w_17_0 ctx) (parallel_generatedRoot_121.w_17_1 ctx) (parallel_generatedRoot_121.w_17_2 ctx) (parallel_generatedRoot_121.w_17_3 ctx) outputs) :
    parallel_generatedRoot_121.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_121.Facts  (ctx : parallel_generatedRoot_121Ctx) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  c0 : parallel_generatedRoot_121.Facts_0 ctx outputs

theorem parallel_generatedRoot_121.facts  (ctx : parallel_generatedRoot_121Ctx) {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_121.body backend hashModel tape path params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_121.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_121.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_125.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) where
  inputs : (Fin 2607022080 → mxxMat1) × (Fin 2607022080 → mxxMat1) × (Fin 2607022080 → mxxMat1) × (Fin 2607022080 → mxxMat1) × Unit
  witness : parallel_generatedRoot_125.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_125Ctx" => (parallel_generatedRoot_125.Ctx backend hashModel tape path params i_0)

def parallel_generatedRoot_125.w_0_0  (ctx : parallel_generatedRoot_125Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.1

def parallel_generatedRoot_125.w_1_0  (ctx : parallel_generatedRoot_125Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.2.1

def parallel_generatedRoot_125.w_2_0  (ctx : parallel_generatedRoot_125Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_125.w_3_0  (ctx : parallel_generatedRoot_125Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_125.w_4_0  (ctx : parallel_generatedRoot_125Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_0

def parallel_generatedRoot_125.w_4_1  (ctx : parallel_generatedRoot_125Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_1

def parallel_generatedRoot_125.w_4_2  (ctx : parallel_generatedRoot_125Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_2

def parallel_generatedRoot_125.w_4_3  (ctx : parallel_generatedRoot_125Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_3

def parallel_generatedRoot_125.w_5_0  (ctx : parallel_generatedRoot_125Ctx) : mxxMat1 :=
  ctx.witness.w_5_0

def parallel_generatedRoot_125.w_6_0  (ctx : parallel_generatedRoot_125Ctx) : mxxMat1 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_125.w_7_0  (ctx : parallel_generatedRoot_125Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_125.w_5_0 ctx) (parallel_generatedRoot_125.w_6_0 ctx)

def parallel_generatedRoot_125.w_8_0  (ctx : parallel_generatedRoot_125Ctx) : mxxMat1 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_125.w_9_0  (ctx : parallel_generatedRoot_125Ctx) : mxxMat1 :=
  ctx.witness.w_9_0

def parallel_generatedRoot_125.w_10_0  (ctx : parallel_generatedRoot_125Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_125.w_8_0 ctx) (parallel_generatedRoot_125.w_9_0 ctx)

def parallel_generatedRoot_125.w_11_0  (ctx : parallel_generatedRoot_125Ctx) : mxxMat1 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_125.w_12_0  (ctx : parallel_generatedRoot_125Ctx) : mxxMat1 :=
  ctx.witness.w_12_0

def parallel_generatedRoot_125.w_13_0  (ctx : parallel_generatedRoot_125Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_125.w_11_0 ctx) (parallel_generatedRoot_125.w_12_0 ctx)

def parallel_generatedRoot_125.w_14_0  (ctx : parallel_generatedRoot_125Ctx) : mxxMat1 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_125.w_15_0  (ctx : parallel_generatedRoot_125Ctx) : mxxMat1 :=
  ctx.witness.w_15_0

def parallel_generatedRoot_125.w_16_0  (ctx : parallel_generatedRoot_125Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_125.w_14_0 ctx) (parallel_generatedRoot_125.w_15_0 ctx)

def parallel_generatedRoot_125.w_17_0  (ctx : parallel_generatedRoot_125Ctx) : mxxMat1 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_125.w_17_1  (ctx : parallel_generatedRoot_125Ctx) : mxxMat1 :=
  ctx.witness.w_17_1

def parallel_generatedRoot_125.w_17_2  (ctx : parallel_generatedRoot_125Ctx) : mxxMat1 :=
  ctx.witness.w_17_2

def parallel_generatedRoot_125.w_17_3  (ctx : parallel_generatedRoot_125Ctx) : mxxMat1 :=
  ctx.witness.w_17_3

def parallel_generatedRoot_125.outs  (ctx : parallel_generatedRoot_125Ctx) : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit :=
  ((parallel_generatedRoot_125.w_17_0 ctx), ((parallel_generatedRoot_125.w_17_1 ctx), ((parallel_generatedRoot_125.w_17_2 ctx), ((parallel_generatedRoot_125.w_17_3 ctx), ()))))

structure parallel_generatedRoot_125.Facts_0  (ctx : parallel_generatedRoot_125Ctx) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  t0 : (130) = 130
  h4_3 : (∀ i : Fin 130, parallel_parallel_generatedRoot_125_4 backend hashModel tape (path ++ [4, i.val]) params i_0 i ((parallel_generatedRoot_125.w_0_0 ctx), ((parallel_generatedRoot_125.w_1_0 ctx), ((parallel_generatedRoot_125.w_2_0 ctx), ((parallel_generatedRoot_125.w_3_0 ctx), ())))) (((parallel_generatedRoot_125.w_4_0 ctx) i), (((parallel_generatedRoot_125.w_4_1 ctx) i), (((parallel_generatedRoot_125.w_4_2 ctx) i), (((parallel_generatedRoot_125.w_4_3 ctx) i), ())))))
  h5_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_125.w_4_0 ctx) (0) (parallel_generatedRoot_125.w_5_0 ctx)
  h6_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_125.w_4_0 ctx) (1) (parallel_generatedRoot_125.w_6_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_125.w_4_1 ctx) (0) (parallel_generatedRoot_125.w_8_0 ctx)
  h9_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_125.w_4_1 ctx) (1) (parallel_generatedRoot_125.w_9_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_125.w_4_2 ctx) (0) (parallel_generatedRoot_125.w_11_0 ctx)
  h12_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_125.w_4_2 ctx) (1) (parallel_generatedRoot_125.w_12_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_125.w_4_3 ctx) (0) (parallel_generatedRoot_125.w_14_0 ctx)
  h15_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_125.w_4_3 ctx) (1) (parallel_generatedRoot_125.w_15_0 ctx)
  t10 : 0 ≤ (128)
  h17_3 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_parallel_generatedRoot_125_17 backend hashModel tape (path ++ [17, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, ((parallel_generatedRoot_125.w_4_0 ctx), ((parallel_generatedRoot_125.w_4_1 ctx), ((parallel_generatedRoot_125.w_4_2 ctx), ((parallel_generatedRoot_125.w_4_3 ctx), ())))))))) next) (Int.toNat (128)) ((parallel_generatedRoot_125.w_7_0 ctx), ((parallel_generatedRoot_125.w_10_0 ctx), ((parallel_generatedRoot_125.w_13_0 ctx), ((parallel_generatedRoot_125.w_16_0 ctx), ())))) ((parallel_generatedRoot_125.w_17_0 ctx), ((parallel_generatedRoot_125.w_17_1 ctx), ((parallel_generatedRoot_125.w_17_2 ctx), ((parallel_generatedRoot_125.w_17_3 ctx), ()))))
  hout : outputs = parallel_generatedRoot_125.outs ctx

theorem parallel_generatedRoot_125.facts_0  {ctx : parallel_generatedRoot_125Ctx} {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_125.constraints_0 backend hashModel tape path params i_0 (parallel_generatedRoot_125.w_0_0 ctx) (parallel_generatedRoot_125.w_1_0 ctx) (parallel_generatedRoot_125.w_2_0 ctx) (parallel_generatedRoot_125.w_3_0 ctx) (parallel_generatedRoot_125.w_4_0 ctx) (parallel_generatedRoot_125.w_4_1 ctx) (parallel_generatedRoot_125.w_4_2 ctx) (parallel_generatedRoot_125.w_4_3 ctx) (parallel_generatedRoot_125.w_5_0 ctx) (parallel_generatedRoot_125.w_6_0 ctx) (parallel_generatedRoot_125.w_7_0 ctx) (parallel_generatedRoot_125.w_8_0 ctx) (parallel_generatedRoot_125.w_9_0 ctx) (parallel_generatedRoot_125.w_10_0 ctx) (parallel_generatedRoot_125.w_11_0 ctx) (parallel_generatedRoot_125.w_12_0 ctx) (parallel_generatedRoot_125.w_13_0 ctx) (parallel_generatedRoot_125.w_14_0 ctx) (parallel_generatedRoot_125.w_15_0 ctx) (parallel_generatedRoot_125.w_16_0 ctx) (parallel_generatedRoot_125.w_17_0 ctx) (parallel_generatedRoot_125.w_17_1 ctx) (parallel_generatedRoot_125.w_17_2 ctx) (parallel_generatedRoot_125.w_17_3 ctx) outputs) :
    parallel_generatedRoot_125.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_125.Facts  (ctx : parallel_generatedRoot_125Ctx) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  c0 : parallel_generatedRoot_125.Facts_0 ctx outputs

theorem parallel_generatedRoot_125.facts  (ctx : parallel_generatedRoot_125Ctx) {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_125.body backend hashModel tape path params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_125.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_125.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_129.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) where
  inputs : (Fin 2607022080 → mxxMat1) × (Fin 2607022080 → mxxMat1) × (Fin 2607022080 → mxxMat1) × (Fin 2607022080 → mxxMat1) × Unit
  witness : parallel_generatedRoot_129.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_129Ctx" => (parallel_generatedRoot_129.Ctx backend hashModel tape path params i_0)

def parallel_generatedRoot_129.w_0_0  (ctx : parallel_generatedRoot_129Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.1

def parallel_generatedRoot_129.w_1_0  (ctx : parallel_generatedRoot_129Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.2.1

def parallel_generatedRoot_129.w_2_0  (ctx : parallel_generatedRoot_129Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_129.w_3_0  (ctx : parallel_generatedRoot_129Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_129.w_4_0  (ctx : parallel_generatedRoot_129Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_0

def parallel_generatedRoot_129.w_4_1  (ctx : parallel_generatedRoot_129Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_1

def parallel_generatedRoot_129.w_4_2  (ctx : parallel_generatedRoot_129Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_2

def parallel_generatedRoot_129.w_4_3  (ctx : parallel_generatedRoot_129Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_3

def parallel_generatedRoot_129.w_5_0  (ctx : parallel_generatedRoot_129Ctx) : mxxMat1 :=
  ctx.witness.w_5_0

def parallel_generatedRoot_129.w_6_0  (ctx : parallel_generatedRoot_129Ctx) : mxxMat1 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_129.w_7_0  (ctx : parallel_generatedRoot_129Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_129.w_5_0 ctx) (parallel_generatedRoot_129.w_6_0 ctx)

def parallel_generatedRoot_129.w_8_0  (ctx : parallel_generatedRoot_129Ctx) : mxxMat1 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_129.w_9_0  (ctx : parallel_generatedRoot_129Ctx) : mxxMat1 :=
  ctx.witness.w_9_0

def parallel_generatedRoot_129.w_10_0  (ctx : parallel_generatedRoot_129Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_129.w_8_0 ctx) (parallel_generatedRoot_129.w_9_0 ctx)

def parallel_generatedRoot_129.w_11_0  (ctx : parallel_generatedRoot_129Ctx) : mxxMat1 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_129.w_12_0  (ctx : parallel_generatedRoot_129Ctx) : mxxMat1 :=
  ctx.witness.w_12_0

def parallel_generatedRoot_129.w_13_0  (ctx : parallel_generatedRoot_129Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_129.w_11_0 ctx) (parallel_generatedRoot_129.w_12_0 ctx)

def parallel_generatedRoot_129.w_14_0  (ctx : parallel_generatedRoot_129Ctx) : mxxMat1 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_129.w_15_0  (ctx : parallel_generatedRoot_129Ctx) : mxxMat1 :=
  ctx.witness.w_15_0

def parallel_generatedRoot_129.w_16_0  (ctx : parallel_generatedRoot_129Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_129.w_14_0 ctx) (parallel_generatedRoot_129.w_15_0 ctx)

def parallel_generatedRoot_129.w_17_0  (ctx : parallel_generatedRoot_129Ctx) : mxxMat1 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_129.w_17_1  (ctx : parallel_generatedRoot_129Ctx) : mxxMat1 :=
  ctx.witness.w_17_1

def parallel_generatedRoot_129.w_17_2  (ctx : parallel_generatedRoot_129Ctx) : mxxMat1 :=
  ctx.witness.w_17_2

def parallel_generatedRoot_129.w_17_3  (ctx : parallel_generatedRoot_129Ctx) : mxxMat1 :=
  ctx.witness.w_17_3

def parallel_generatedRoot_129.outs  (ctx : parallel_generatedRoot_129Ctx) : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit :=
  ((parallel_generatedRoot_129.w_17_0 ctx), ((parallel_generatedRoot_129.w_17_1 ctx), ((parallel_generatedRoot_129.w_17_2 ctx), ((parallel_generatedRoot_129.w_17_3 ctx), ()))))

structure parallel_generatedRoot_129.Facts_0  (ctx : parallel_generatedRoot_129Ctx) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  t0 : (130) = 130
  h4_3 : (∀ i : Fin 130, parallel_parallel_generatedRoot_129_4 backend hashModel tape (path ++ [4, i.val]) params i_0 i ((parallel_generatedRoot_129.w_0_0 ctx), ((parallel_generatedRoot_129.w_1_0 ctx), ((parallel_generatedRoot_129.w_2_0 ctx), ((parallel_generatedRoot_129.w_3_0 ctx), ())))) (((parallel_generatedRoot_129.w_4_0 ctx) i), (((parallel_generatedRoot_129.w_4_1 ctx) i), (((parallel_generatedRoot_129.w_4_2 ctx) i), (((parallel_generatedRoot_129.w_4_3 ctx) i), ())))))
  h5_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_129.w_4_0 ctx) (0) (parallel_generatedRoot_129.w_5_0 ctx)
  h6_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_129.w_4_0 ctx) (1) (parallel_generatedRoot_129.w_6_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_129.w_4_1 ctx) (0) (parallel_generatedRoot_129.w_8_0 ctx)
  h9_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_129.w_4_1 ctx) (1) (parallel_generatedRoot_129.w_9_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_129.w_4_2 ctx) (0) (parallel_generatedRoot_129.w_11_0 ctx)
  h12_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_129.w_4_2 ctx) (1) (parallel_generatedRoot_129.w_12_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_129.w_4_3 ctx) (0) (parallel_generatedRoot_129.w_14_0 ctx)
  h15_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_129.w_4_3 ctx) (1) (parallel_generatedRoot_129.w_15_0 ctx)
  t10 : 0 ≤ (128)
  h17_3 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_parallel_generatedRoot_129_17 backend hashModel tape (path ++ [17, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, ((parallel_generatedRoot_129.w_4_0 ctx), ((parallel_generatedRoot_129.w_4_1 ctx), ((parallel_generatedRoot_129.w_4_2 ctx), ((parallel_generatedRoot_129.w_4_3 ctx), ())))))))) next) (Int.toNat (128)) ((parallel_generatedRoot_129.w_7_0 ctx), ((parallel_generatedRoot_129.w_10_0 ctx), ((parallel_generatedRoot_129.w_13_0 ctx), ((parallel_generatedRoot_129.w_16_0 ctx), ())))) ((parallel_generatedRoot_129.w_17_0 ctx), ((parallel_generatedRoot_129.w_17_1 ctx), ((parallel_generatedRoot_129.w_17_2 ctx), ((parallel_generatedRoot_129.w_17_3 ctx), ()))))
  hout : outputs = parallel_generatedRoot_129.outs ctx

theorem parallel_generatedRoot_129.facts_0  {ctx : parallel_generatedRoot_129Ctx} {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_129.constraints_0 backend hashModel tape path params i_0 (parallel_generatedRoot_129.w_0_0 ctx) (parallel_generatedRoot_129.w_1_0 ctx) (parallel_generatedRoot_129.w_2_0 ctx) (parallel_generatedRoot_129.w_3_0 ctx) (parallel_generatedRoot_129.w_4_0 ctx) (parallel_generatedRoot_129.w_4_1 ctx) (parallel_generatedRoot_129.w_4_2 ctx) (parallel_generatedRoot_129.w_4_3 ctx) (parallel_generatedRoot_129.w_5_0 ctx) (parallel_generatedRoot_129.w_6_0 ctx) (parallel_generatedRoot_129.w_7_0 ctx) (parallel_generatedRoot_129.w_8_0 ctx) (parallel_generatedRoot_129.w_9_0 ctx) (parallel_generatedRoot_129.w_10_0 ctx) (parallel_generatedRoot_129.w_11_0 ctx) (parallel_generatedRoot_129.w_12_0 ctx) (parallel_generatedRoot_129.w_13_0 ctx) (parallel_generatedRoot_129.w_14_0 ctx) (parallel_generatedRoot_129.w_15_0 ctx) (parallel_generatedRoot_129.w_16_0 ctx) (parallel_generatedRoot_129.w_17_0 ctx) (parallel_generatedRoot_129.w_17_1 ctx) (parallel_generatedRoot_129.w_17_2 ctx) (parallel_generatedRoot_129.w_17_3 ctx) outputs) :
    parallel_generatedRoot_129.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_129.Facts  (ctx : parallel_generatedRoot_129Ctx) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  c0 : parallel_generatedRoot_129.Facts_0 ctx outputs

theorem parallel_generatedRoot_129.facts  (ctx : parallel_generatedRoot_129Ctx) {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_129.body backend hashModel tape path params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_129.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_129.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_133.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) where
  inputs : (Fin 2607022080 → mxxMat1) × (Fin 2607022080 → mxxMat1) × (Fin 2607022080 → mxxMat1) × (Fin 2607022080 → mxxMat1) × Unit
  witness : parallel_generatedRoot_133.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_133Ctx" => (parallel_generatedRoot_133.Ctx backend hashModel tape path params i_0)

def parallel_generatedRoot_133.w_0_0  (ctx : parallel_generatedRoot_133Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.1

def parallel_generatedRoot_133.w_1_0  (ctx : parallel_generatedRoot_133Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.2.1

def parallel_generatedRoot_133.w_2_0  (ctx : parallel_generatedRoot_133Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_133.w_3_0  (ctx : parallel_generatedRoot_133Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_133.w_4_0  (ctx : parallel_generatedRoot_133Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_0

def parallel_generatedRoot_133.w_4_1  (ctx : parallel_generatedRoot_133Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_1

def parallel_generatedRoot_133.w_4_2  (ctx : parallel_generatedRoot_133Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_2

def parallel_generatedRoot_133.w_4_3  (ctx : parallel_generatedRoot_133Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_3

def parallel_generatedRoot_133.w_5_0  (ctx : parallel_generatedRoot_133Ctx) : mxxMat1 :=
  ctx.witness.w_5_0

def parallel_generatedRoot_133.w_6_0  (ctx : parallel_generatedRoot_133Ctx) : mxxMat1 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_133.w_7_0  (ctx : parallel_generatedRoot_133Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_133.w_5_0 ctx) (parallel_generatedRoot_133.w_6_0 ctx)

def parallel_generatedRoot_133.w_8_0  (ctx : parallel_generatedRoot_133Ctx) : mxxMat1 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_133.w_9_0  (ctx : parallel_generatedRoot_133Ctx) : mxxMat1 :=
  ctx.witness.w_9_0

def parallel_generatedRoot_133.w_10_0  (ctx : parallel_generatedRoot_133Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_133.w_8_0 ctx) (parallel_generatedRoot_133.w_9_0 ctx)

def parallel_generatedRoot_133.w_11_0  (ctx : parallel_generatedRoot_133Ctx) : mxxMat1 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_133.w_12_0  (ctx : parallel_generatedRoot_133Ctx) : mxxMat1 :=
  ctx.witness.w_12_0

def parallel_generatedRoot_133.w_13_0  (ctx : parallel_generatedRoot_133Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_133.w_11_0 ctx) (parallel_generatedRoot_133.w_12_0 ctx)

def parallel_generatedRoot_133.w_14_0  (ctx : parallel_generatedRoot_133Ctx) : mxxMat1 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_133.w_15_0  (ctx : parallel_generatedRoot_133Ctx) : mxxMat1 :=
  ctx.witness.w_15_0

def parallel_generatedRoot_133.w_16_0  (ctx : parallel_generatedRoot_133Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_133.w_14_0 ctx) (parallel_generatedRoot_133.w_15_0 ctx)

def parallel_generatedRoot_133.w_17_0  (ctx : parallel_generatedRoot_133Ctx) : mxxMat1 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_133.w_17_1  (ctx : parallel_generatedRoot_133Ctx) : mxxMat1 :=
  ctx.witness.w_17_1

def parallel_generatedRoot_133.w_17_2  (ctx : parallel_generatedRoot_133Ctx) : mxxMat1 :=
  ctx.witness.w_17_2

def parallel_generatedRoot_133.w_17_3  (ctx : parallel_generatedRoot_133Ctx) : mxxMat1 :=
  ctx.witness.w_17_3

def parallel_generatedRoot_133.outs  (ctx : parallel_generatedRoot_133Ctx) : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit :=
  ((parallel_generatedRoot_133.w_17_0 ctx), ((parallel_generatedRoot_133.w_17_1 ctx), ((parallel_generatedRoot_133.w_17_2 ctx), ((parallel_generatedRoot_133.w_17_3 ctx), ()))))

structure parallel_generatedRoot_133.Facts_0  (ctx : parallel_generatedRoot_133Ctx) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  t0 : (130) = 130
  h4_3 : (∀ i : Fin 130, parallel_parallel_generatedRoot_133_4 backend hashModel tape (path ++ [4, i.val]) params i_0 i ((parallel_generatedRoot_133.w_0_0 ctx), ((parallel_generatedRoot_133.w_1_0 ctx), ((parallel_generatedRoot_133.w_2_0 ctx), ((parallel_generatedRoot_133.w_3_0 ctx), ())))) (((parallel_generatedRoot_133.w_4_0 ctx) i), (((parallel_generatedRoot_133.w_4_1 ctx) i), (((parallel_generatedRoot_133.w_4_2 ctx) i), (((parallel_generatedRoot_133.w_4_3 ctx) i), ())))))
  h5_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_133.w_4_0 ctx) (0) (parallel_generatedRoot_133.w_5_0 ctx)
  h6_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_133.w_4_0 ctx) (1) (parallel_generatedRoot_133.w_6_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_133.w_4_1 ctx) (0) (parallel_generatedRoot_133.w_8_0 ctx)
  h9_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_133.w_4_1 ctx) (1) (parallel_generatedRoot_133.w_9_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_133.w_4_2 ctx) (0) (parallel_generatedRoot_133.w_11_0 ctx)
  h12_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_133.w_4_2 ctx) (1) (parallel_generatedRoot_133.w_12_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_133.w_4_3 ctx) (0) (parallel_generatedRoot_133.w_14_0 ctx)
  h15_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_133.w_4_3 ctx) (1) (parallel_generatedRoot_133.w_15_0 ctx)
  t10 : 0 ≤ (128)
  h17_3 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_parallel_generatedRoot_133_17 backend hashModel tape (path ++ [17, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, ((parallel_generatedRoot_133.w_4_0 ctx), ((parallel_generatedRoot_133.w_4_1 ctx), ((parallel_generatedRoot_133.w_4_2 ctx), ((parallel_generatedRoot_133.w_4_3 ctx), ())))))))) next) (Int.toNat (128)) ((parallel_generatedRoot_133.w_7_0 ctx), ((parallel_generatedRoot_133.w_10_0 ctx), ((parallel_generatedRoot_133.w_13_0 ctx), ((parallel_generatedRoot_133.w_16_0 ctx), ())))) ((parallel_generatedRoot_133.w_17_0 ctx), ((parallel_generatedRoot_133.w_17_1 ctx), ((parallel_generatedRoot_133.w_17_2 ctx), ((parallel_generatedRoot_133.w_17_3 ctx), ()))))
  hout : outputs = parallel_generatedRoot_133.outs ctx

theorem parallel_generatedRoot_133.facts_0  {ctx : parallel_generatedRoot_133Ctx} {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_133.constraints_0 backend hashModel tape path params i_0 (parallel_generatedRoot_133.w_0_0 ctx) (parallel_generatedRoot_133.w_1_0 ctx) (parallel_generatedRoot_133.w_2_0 ctx) (parallel_generatedRoot_133.w_3_0 ctx) (parallel_generatedRoot_133.w_4_0 ctx) (parallel_generatedRoot_133.w_4_1 ctx) (parallel_generatedRoot_133.w_4_2 ctx) (parallel_generatedRoot_133.w_4_3 ctx) (parallel_generatedRoot_133.w_5_0 ctx) (parallel_generatedRoot_133.w_6_0 ctx) (parallel_generatedRoot_133.w_7_0 ctx) (parallel_generatedRoot_133.w_8_0 ctx) (parallel_generatedRoot_133.w_9_0 ctx) (parallel_generatedRoot_133.w_10_0 ctx) (parallel_generatedRoot_133.w_11_0 ctx) (parallel_generatedRoot_133.w_12_0 ctx) (parallel_generatedRoot_133.w_13_0 ctx) (parallel_generatedRoot_133.w_14_0 ctx) (parallel_generatedRoot_133.w_15_0 ctx) (parallel_generatedRoot_133.w_16_0 ctx) (parallel_generatedRoot_133.w_17_0 ctx) (parallel_generatedRoot_133.w_17_1 ctx) (parallel_generatedRoot_133.w_17_2 ctx) (parallel_generatedRoot_133.w_17_3 ctx) outputs) :
    parallel_generatedRoot_133.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_133.Facts  (ctx : parallel_generatedRoot_133Ctx) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  c0 : parallel_generatedRoot_133.Facts_0 ctx outputs

theorem parallel_generatedRoot_133.facts  (ctx : parallel_generatedRoot_133Ctx) {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_133.body backend hashModel tape path params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_133.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_133.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_137.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) where
  inputs : (Fin 2607022080 → mxxMat1) × (Fin 2607022080 → mxxMat1) × (Fin 2607022080 → mxxMat1) × (Fin 2607022080 → mxxMat1) × Unit
  witness : parallel_generatedRoot_137.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_137Ctx" => (parallel_generatedRoot_137.Ctx backend hashModel tape path params i_0)

def parallel_generatedRoot_137.w_0_0  (ctx : parallel_generatedRoot_137Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.1

def parallel_generatedRoot_137.w_1_0  (ctx : parallel_generatedRoot_137Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.2.1

def parallel_generatedRoot_137.w_2_0  (ctx : parallel_generatedRoot_137Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_137.w_3_0  (ctx : parallel_generatedRoot_137Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_137.w_4_0  (ctx : parallel_generatedRoot_137Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_0

def parallel_generatedRoot_137.w_4_1  (ctx : parallel_generatedRoot_137Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_1

def parallel_generatedRoot_137.w_4_2  (ctx : parallel_generatedRoot_137Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_2

def parallel_generatedRoot_137.w_4_3  (ctx : parallel_generatedRoot_137Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_3

def parallel_generatedRoot_137.w_5_0  (ctx : parallel_generatedRoot_137Ctx) : mxxMat1 :=
  ctx.witness.w_5_0

def parallel_generatedRoot_137.w_6_0  (ctx : parallel_generatedRoot_137Ctx) : mxxMat1 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_137.w_7_0  (ctx : parallel_generatedRoot_137Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_137.w_5_0 ctx) (parallel_generatedRoot_137.w_6_0 ctx)

def parallel_generatedRoot_137.w_8_0  (ctx : parallel_generatedRoot_137Ctx) : mxxMat1 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_137.w_9_0  (ctx : parallel_generatedRoot_137Ctx) : mxxMat1 :=
  ctx.witness.w_9_0

def parallel_generatedRoot_137.w_10_0  (ctx : parallel_generatedRoot_137Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_137.w_8_0 ctx) (parallel_generatedRoot_137.w_9_0 ctx)

def parallel_generatedRoot_137.w_11_0  (ctx : parallel_generatedRoot_137Ctx) : mxxMat1 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_137.w_12_0  (ctx : parallel_generatedRoot_137Ctx) : mxxMat1 :=
  ctx.witness.w_12_0

def parallel_generatedRoot_137.w_13_0  (ctx : parallel_generatedRoot_137Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_137.w_11_0 ctx) (parallel_generatedRoot_137.w_12_0 ctx)

def parallel_generatedRoot_137.w_14_0  (ctx : parallel_generatedRoot_137Ctx) : mxxMat1 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_137.w_15_0  (ctx : parallel_generatedRoot_137Ctx) : mxxMat1 :=
  ctx.witness.w_15_0

def parallel_generatedRoot_137.w_16_0  (ctx : parallel_generatedRoot_137Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_137.w_14_0 ctx) (parallel_generatedRoot_137.w_15_0 ctx)

def parallel_generatedRoot_137.w_17_0  (ctx : parallel_generatedRoot_137Ctx) : mxxMat1 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_137.w_17_1  (ctx : parallel_generatedRoot_137Ctx) : mxxMat1 :=
  ctx.witness.w_17_1

def parallel_generatedRoot_137.w_17_2  (ctx : parallel_generatedRoot_137Ctx) : mxxMat1 :=
  ctx.witness.w_17_2

def parallel_generatedRoot_137.w_17_3  (ctx : parallel_generatedRoot_137Ctx) : mxxMat1 :=
  ctx.witness.w_17_3

def parallel_generatedRoot_137.outs  (ctx : parallel_generatedRoot_137Ctx) : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit :=
  ((parallel_generatedRoot_137.w_17_0 ctx), ((parallel_generatedRoot_137.w_17_1 ctx), ((parallel_generatedRoot_137.w_17_2 ctx), ((parallel_generatedRoot_137.w_17_3 ctx), ()))))

structure parallel_generatedRoot_137.Facts_0  (ctx : parallel_generatedRoot_137Ctx) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  t0 : (130) = 130
  h4_3 : (∀ i : Fin 130, parallel_parallel_generatedRoot_137_4 backend hashModel tape (path ++ [4, i.val]) params i_0 i ((parallel_generatedRoot_137.w_0_0 ctx), ((parallel_generatedRoot_137.w_1_0 ctx), ((parallel_generatedRoot_137.w_2_0 ctx), ((parallel_generatedRoot_137.w_3_0 ctx), ())))) (((parallel_generatedRoot_137.w_4_0 ctx) i), (((parallel_generatedRoot_137.w_4_1 ctx) i), (((parallel_generatedRoot_137.w_4_2 ctx) i), (((parallel_generatedRoot_137.w_4_3 ctx) i), ())))))
  h5_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_137.w_4_0 ctx) (0) (parallel_generatedRoot_137.w_5_0 ctx)
  h6_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_137.w_4_0 ctx) (1) (parallel_generatedRoot_137.w_6_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_137.w_4_1 ctx) (0) (parallel_generatedRoot_137.w_8_0 ctx)
  h9_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_137.w_4_1 ctx) (1) (parallel_generatedRoot_137.w_9_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_137.w_4_2 ctx) (0) (parallel_generatedRoot_137.w_11_0 ctx)
  h12_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_137.w_4_2 ctx) (1) (parallel_generatedRoot_137.w_12_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_137.w_4_3 ctx) (0) (parallel_generatedRoot_137.w_14_0 ctx)
  h15_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_137.w_4_3 ctx) (1) (parallel_generatedRoot_137.w_15_0 ctx)
  t10 : 0 ≤ (128)
  h17_3 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_parallel_generatedRoot_137_17 backend hashModel tape (path ++ [17, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, ((parallel_generatedRoot_137.w_4_0 ctx), ((parallel_generatedRoot_137.w_4_1 ctx), ((parallel_generatedRoot_137.w_4_2 ctx), ((parallel_generatedRoot_137.w_4_3 ctx), ())))))))) next) (Int.toNat (128)) ((parallel_generatedRoot_137.w_7_0 ctx), ((parallel_generatedRoot_137.w_10_0 ctx), ((parallel_generatedRoot_137.w_13_0 ctx), ((parallel_generatedRoot_137.w_16_0 ctx), ())))) ((parallel_generatedRoot_137.w_17_0 ctx), ((parallel_generatedRoot_137.w_17_1 ctx), ((parallel_generatedRoot_137.w_17_2 ctx), ((parallel_generatedRoot_137.w_17_3 ctx), ()))))
  hout : outputs = parallel_generatedRoot_137.outs ctx

theorem parallel_generatedRoot_137.facts_0  {ctx : parallel_generatedRoot_137Ctx} {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_137.constraints_0 backend hashModel tape path params i_0 (parallel_generatedRoot_137.w_0_0 ctx) (parallel_generatedRoot_137.w_1_0 ctx) (parallel_generatedRoot_137.w_2_0 ctx) (parallel_generatedRoot_137.w_3_0 ctx) (parallel_generatedRoot_137.w_4_0 ctx) (parallel_generatedRoot_137.w_4_1 ctx) (parallel_generatedRoot_137.w_4_2 ctx) (parallel_generatedRoot_137.w_4_3 ctx) (parallel_generatedRoot_137.w_5_0 ctx) (parallel_generatedRoot_137.w_6_0 ctx) (parallel_generatedRoot_137.w_7_0 ctx) (parallel_generatedRoot_137.w_8_0 ctx) (parallel_generatedRoot_137.w_9_0 ctx) (parallel_generatedRoot_137.w_10_0 ctx) (parallel_generatedRoot_137.w_11_0 ctx) (parallel_generatedRoot_137.w_12_0 ctx) (parallel_generatedRoot_137.w_13_0 ctx) (parallel_generatedRoot_137.w_14_0 ctx) (parallel_generatedRoot_137.w_15_0 ctx) (parallel_generatedRoot_137.w_16_0 ctx) (parallel_generatedRoot_137.w_17_0 ctx) (parallel_generatedRoot_137.w_17_1 ctx) (parallel_generatedRoot_137.w_17_2 ctx) (parallel_generatedRoot_137.w_17_3 ctx) outputs) :
    parallel_generatedRoot_137.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_137.Facts  (ctx : parallel_generatedRoot_137Ctx) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  c0 : parallel_generatedRoot_137.Facts_0 ctx outputs

theorem parallel_generatedRoot_137.facts  (ctx : parallel_generatedRoot_137Ctx) {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_137.body backend hashModel tape path params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_137.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_137.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_141.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) where
  inputs : (Fin 2607022080 → mxxMat1) × (Fin 2607022080 → mxxMat1) × (Fin 2607022080 → mxxMat1) × (Fin 2607022080 → mxxMat1) × Unit
  witness : parallel_generatedRoot_141.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_141Ctx" => (parallel_generatedRoot_141.Ctx backend hashModel tape path params i_0)

def parallel_generatedRoot_141.w_0_0  (ctx : parallel_generatedRoot_141Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.1

def parallel_generatedRoot_141.w_1_0  (ctx : parallel_generatedRoot_141Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.2.1

def parallel_generatedRoot_141.w_2_0  (ctx : parallel_generatedRoot_141Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_141.w_3_0  (ctx : parallel_generatedRoot_141Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_141.w_4_0  (ctx : parallel_generatedRoot_141Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_0

def parallel_generatedRoot_141.w_4_1  (ctx : parallel_generatedRoot_141Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_1

def parallel_generatedRoot_141.w_4_2  (ctx : parallel_generatedRoot_141Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_2

def parallel_generatedRoot_141.w_4_3  (ctx : parallel_generatedRoot_141Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_3

def parallel_generatedRoot_141.w_5_0  (ctx : parallel_generatedRoot_141Ctx) : mxxMat1 :=
  ctx.witness.w_5_0

def parallel_generatedRoot_141.w_6_0  (ctx : parallel_generatedRoot_141Ctx) : mxxMat1 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_141.w_7_0  (ctx : parallel_generatedRoot_141Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_141.w_5_0 ctx) (parallel_generatedRoot_141.w_6_0 ctx)

def parallel_generatedRoot_141.w_8_0  (ctx : parallel_generatedRoot_141Ctx) : mxxMat1 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_141.w_9_0  (ctx : parallel_generatedRoot_141Ctx) : mxxMat1 :=
  ctx.witness.w_9_0

def parallel_generatedRoot_141.w_10_0  (ctx : parallel_generatedRoot_141Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_141.w_8_0 ctx) (parallel_generatedRoot_141.w_9_0 ctx)

def parallel_generatedRoot_141.w_11_0  (ctx : parallel_generatedRoot_141Ctx) : mxxMat1 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_141.w_12_0  (ctx : parallel_generatedRoot_141Ctx) : mxxMat1 :=
  ctx.witness.w_12_0

def parallel_generatedRoot_141.w_13_0  (ctx : parallel_generatedRoot_141Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_141.w_11_0 ctx) (parallel_generatedRoot_141.w_12_0 ctx)

def parallel_generatedRoot_141.w_14_0  (ctx : parallel_generatedRoot_141Ctx) : mxxMat1 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_141.w_15_0  (ctx : parallel_generatedRoot_141Ctx) : mxxMat1 :=
  ctx.witness.w_15_0

def parallel_generatedRoot_141.w_16_0  (ctx : parallel_generatedRoot_141Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_141.w_14_0 ctx) (parallel_generatedRoot_141.w_15_0 ctx)

def parallel_generatedRoot_141.w_17_0  (ctx : parallel_generatedRoot_141Ctx) : mxxMat1 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_141.w_17_1  (ctx : parallel_generatedRoot_141Ctx) : mxxMat1 :=
  ctx.witness.w_17_1

def parallel_generatedRoot_141.w_17_2  (ctx : parallel_generatedRoot_141Ctx) : mxxMat1 :=
  ctx.witness.w_17_2

def parallel_generatedRoot_141.w_17_3  (ctx : parallel_generatedRoot_141Ctx) : mxxMat1 :=
  ctx.witness.w_17_3

def parallel_generatedRoot_141.outs  (ctx : parallel_generatedRoot_141Ctx) : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit :=
  ((parallel_generatedRoot_141.w_17_0 ctx), ((parallel_generatedRoot_141.w_17_1 ctx), ((parallel_generatedRoot_141.w_17_2 ctx), ((parallel_generatedRoot_141.w_17_3 ctx), ()))))

structure parallel_generatedRoot_141.Facts_0  (ctx : parallel_generatedRoot_141Ctx) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  t0 : (130) = 130
  h4_3 : (∀ i : Fin 130, parallel_parallel_generatedRoot_141_4 backend hashModel tape (path ++ [4, i.val]) params i_0 i ((parallel_generatedRoot_141.w_0_0 ctx), ((parallel_generatedRoot_141.w_1_0 ctx), ((parallel_generatedRoot_141.w_2_0 ctx), ((parallel_generatedRoot_141.w_3_0 ctx), ())))) (((parallel_generatedRoot_141.w_4_0 ctx) i), (((parallel_generatedRoot_141.w_4_1 ctx) i), (((parallel_generatedRoot_141.w_4_2 ctx) i), (((parallel_generatedRoot_141.w_4_3 ctx) i), ())))))
  h5_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_141.w_4_0 ctx) (0) (parallel_generatedRoot_141.w_5_0 ctx)
  h6_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_141.w_4_0 ctx) (1) (parallel_generatedRoot_141.w_6_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_141.w_4_1 ctx) (0) (parallel_generatedRoot_141.w_8_0 ctx)
  h9_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_141.w_4_1 ctx) (1) (parallel_generatedRoot_141.w_9_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_141.w_4_2 ctx) (0) (parallel_generatedRoot_141.w_11_0 ctx)
  h12_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_141.w_4_2 ctx) (1) (parallel_generatedRoot_141.w_12_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_141.w_4_3 ctx) (0) (parallel_generatedRoot_141.w_14_0 ctx)
  h15_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_141.w_4_3 ctx) (1) (parallel_generatedRoot_141.w_15_0 ctx)
  t10 : 0 ≤ (128)
  h17_3 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_parallel_generatedRoot_141_17 backend hashModel tape (path ++ [17, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, ((parallel_generatedRoot_141.w_4_0 ctx), ((parallel_generatedRoot_141.w_4_1 ctx), ((parallel_generatedRoot_141.w_4_2 ctx), ((parallel_generatedRoot_141.w_4_3 ctx), ())))))))) next) (Int.toNat (128)) ((parallel_generatedRoot_141.w_7_0 ctx), ((parallel_generatedRoot_141.w_10_0 ctx), ((parallel_generatedRoot_141.w_13_0 ctx), ((parallel_generatedRoot_141.w_16_0 ctx), ())))) ((parallel_generatedRoot_141.w_17_0 ctx), ((parallel_generatedRoot_141.w_17_1 ctx), ((parallel_generatedRoot_141.w_17_2 ctx), ((parallel_generatedRoot_141.w_17_3 ctx), ()))))
  hout : outputs = parallel_generatedRoot_141.outs ctx

theorem parallel_generatedRoot_141.facts_0  {ctx : parallel_generatedRoot_141Ctx} {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_141.constraints_0 backend hashModel tape path params i_0 (parallel_generatedRoot_141.w_0_0 ctx) (parallel_generatedRoot_141.w_1_0 ctx) (parallel_generatedRoot_141.w_2_0 ctx) (parallel_generatedRoot_141.w_3_0 ctx) (parallel_generatedRoot_141.w_4_0 ctx) (parallel_generatedRoot_141.w_4_1 ctx) (parallel_generatedRoot_141.w_4_2 ctx) (parallel_generatedRoot_141.w_4_3 ctx) (parallel_generatedRoot_141.w_5_0 ctx) (parallel_generatedRoot_141.w_6_0 ctx) (parallel_generatedRoot_141.w_7_0 ctx) (parallel_generatedRoot_141.w_8_0 ctx) (parallel_generatedRoot_141.w_9_0 ctx) (parallel_generatedRoot_141.w_10_0 ctx) (parallel_generatedRoot_141.w_11_0 ctx) (parallel_generatedRoot_141.w_12_0 ctx) (parallel_generatedRoot_141.w_13_0 ctx) (parallel_generatedRoot_141.w_14_0 ctx) (parallel_generatedRoot_141.w_15_0 ctx) (parallel_generatedRoot_141.w_16_0 ctx) (parallel_generatedRoot_141.w_17_0 ctx) (parallel_generatedRoot_141.w_17_1 ctx) (parallel_generatedRoot_141.w_17_2 ctx) (parallel_generatedRoot_141.w_17_3 ctx) outputs) :
    parallel_generatedRoot_141.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_141.Facts  (ctx : parallel_generatedRoot_141Ctx) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  c0 : parallel_generatedRoot_141.Facts_0 ctx outputs

theorem parallel_generatedRoot_141.facts  (ctx : parallel_generatedRoot_141Ctx) {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_141.body backend hashModel tape path params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_141.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_141.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_145.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) where
  inputs : (Fin 2607022080 → mxxMat1) × (Fin 2607022080 → mxxMat1) × (Fin 2607022080 → mxxMat1) × (Fin 2607022080 → mxxMat1) × Unit
  witness : parallel_generatedRoot_145.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_145Ctx" => (parallel_generatedRoot_145.Ctx backend hashModel tape path params i_0)

def parallel_generatedRoot_145.w_0_0  (ctx : parallel_generatedRoot_145Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.1

def parallel_generatedRoot_145.w_1_0  (ctx : parallel_generatedRoot_145Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.2.1

def parallel_generatedRoot_145.w_2_0  (ctx : parallel_generatedRoot_145Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_145.w_3_0  (ctx : parallel_generatedRoot_145Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_145.w_4_0  (ctx : parallel_generatedRoot_145Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_0

def parallel_generatedRoot_145.w_4_1  (ctx : parallel_generatedRoot_145Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_1

def parallel_generatedRoot_145.w_4_2  (ctx : parallel_generatedRoot_145Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_2

def parallel_generatedRoot_145.w_4_3  (ctx : parallel_generatedRoot_145Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_3

def parallel_generatedRoot_145.w_5_0  (ctx : parallel_generatedRoot_145Ctx) : mxxMat1 :=
  ctx.witness.w_5_0

def parallel_generatedRoot_145.w_6_0  (ctx : parallel_generatedRoot_145Ctx) : mxxMat1 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_145.w_7_0  (ctx : parallel_generatedRoot_145Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_145.w_5_0 ctx) (parallel_generatedRoot_145.w_6_0 ctx)

def parallel_generatedRoot_145.w_8_0  (ctx : parallel_generatedRoot_145Ctx) : mxxMat1 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_145.w_9_0  (ctx : parallel_generatedRoot_145Ctx) : mxxMat1 :=
  ctx.witness.w_9_0

def parallel_generatedRoot_145.w_10_0  (ctx : parallel_generatedRoot_145Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_145.w_8_0 ctx) (parallel_generatedRoot_145.w_9_0 ctx)

def parallel_generatedRoot_145.w_11_0  (ctx : parallel_generatedRoot_145Ctx) : mxxMat1 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_145.w_12_0  (ctx : parallel_generatedRoot_145Ctx) : mxxMat1 :=
  ctx.witness.w_12_0

def parallel_generatedRoot_145.w_13_0  (ctx : parallel_generatedRoot_145Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_145.w_11_0 ctx) (parallel_generatedRoot_145.w_12_0 ctx)

def parallel_generatedRoot_145.w_14_0  (ctx : parallel_generatedRoot_145Ctx) : mxxMat1 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_145.w_15_0  (ctx : parallel_generatedRoot_145Ctx) : mxxMat1 :=
  ctx.witness.w_15_0

def parallel_generatedRoot_145.w_16_0  (ctx : parallel_generatedRoot_145Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_145.w_14_0 ctx) (parallel_generatedRoot_145.w_15_0 ctx)

def parallel_generatedRoot_145.w_17_0  (ctx : parallel_generatedRoot_145Ctx) : mxxMat1 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_145.w_17_1  (ctx : parallel_generatedRoot_145Ctx) : mxxMat1 :=
  ctx.witness.w_17_1

def parallel_generatedRoot_145.w_17_2  (ctx : parallel_generatedRoot_145Ctx) : mxxMat1 :=
  ctx.witness.w_17_2

def parallel_generatedRoot_145.w_17_3  (ctx : parallel_generatedRoot_145Ctx) : mxxMat1 :=
  ctx.witness.w_17_3

def parallel_generatedRoot_145.outs  (ctx : parallel_generatedRoot_145Ctx) : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit :=
  ((parallel_generatedRoot_145.w_17_0 ctx), ((parallel_generatedRoot_145.w_17_1 ctx), ((parallel_generatedRoot_145.w_17_2 ctx), ((parallel_generatedRoot_145.w_17_3 ctx), ()))))

structure parallel_generatedRoot_145.Facts_0  (ctx : parallel_generatedRoot_145Ctx) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  t0 : (130) = 130
  h4_3 : (∀ i : Fin 130, parallel_parallel_generatedRoot_145_4 backend hashModel tape (path ++ [4, i.val]) params i_0 i ((parallel_generatedRoot_145.w_0_0 ctx), ((parallel_generatedRoot_145.w_1_0 ctx), ((parallel_generatedRoot_145.w_2_0 ctx), ((parallel_generatedRoot_145.w_3_0 ctx), ())))) (((parallel_generatedRoot_145.w_4_0 ctx) i), (((parallel_generatedRoot_145.w_4_1 ctx) i), (((parallel_generatedRoot_145.w_4_2 ctx) i), (((parallel_generatedRoot_145.w_4_3 ctx) i), ())))))
  h5_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_145.w_4_0 ctx) (0) (parallel_generatedRoot_145.w_5_0 ctx)
  h6_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_145.w_4_0 ctx) (1) (parallel_generatedRoot_145.w_6_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_145.w_4_1 ctx) (0) (parallel_generatedRoot_145.w_8_0 ctx)
  h9_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_145.w_4_1 ctx) (1) (parallel_generatedRoot_145.w_9_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_145.w_4_2 ctx) (0) (parallel_generatedRoot_145.w_11_0 ctx)
  h12_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_145.w_4_2 ctx) (1) (parallel_generatedRoot_145.w_12_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_145.w_4_3 ctx) (0) (parallel_generatedRoot_145.w_14_0 ctx)
  h15_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_145.w_4_3 ctx) (1) (parallel_generatedRoot_145.w_15_0 ctx)
  t10 : 0 ≤ (128)
  h17_3 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_parallel_generatedRoot_145_17 backend hashModel tape (path ++ [17, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, ((parallel_generatedRoot_145.w_4_0 ctx), ((parallel_generatedRoot_145.w_4_1 ctx), ((parallel_generatedRoot_145.w_4_2 ctx), ((parallel_generatedRoot_145.w_4_3 ctx), ())))))))) next) (Int.toNat (128)) ((parallel_generatedRoot_145.w_7_0 ctx), ((parallel_generatedRoot_145.w_10_0 ctx), ((parallel_generatedRoot_145.w_13_0 ctx), ((parallel_generatedRoot_145.w_16_0 ctx), ())))) ((parallel_generatedRoot_145.w_17_0 ctx), ((parallel_generatedRoot_145.w_17_1 ctx), ((parallel_generatedRoot_145.w_17_2 ctx), ((parallel_generatedRoot_145.w_17_3 ctx), ()))))
  hout : outputs = parallel_generatedRoot_145.outs ctx

theorem parallel_generatedRoot_145.facts_0  {ctx : parallel_generatedRoot_145Ctx} {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_145.constraints_0 backend hashModel tape path params i_0 (parallel_generatedRoot_145.w_0_0 ctx) (parallel_generatedRoot_145.w_1_0 ctx) (parallel_generatedRoot_145.w_2_0 ctx) (parallel_generatedRoot_145.w_3_0 ctx) (parallel_generatedRoot_145.w_4_0 ctx) (parallel_generatedRoot_145.w_4_1 ctx) (parallel_generatedRoot_145.w_4_2 ctx) (parallel_generatedRoot_145.w_4_3 ctx) (parallel_generatedRoot_145.w_5_0 ctx) (parallel_generatedRoot_145.w_6_0 ctx) (parallel_generatedRoot_145.w_7_0 ctx) (parallel_generatedRoot_145.w_8_0 ctx) (parallel_generatedRoot_145.w_9_0 ctx) (parallel_generatedRoot_145.w_10_0 ctx) (parallel_generatedRoot_145.w_11_0 ctx) (parallel_generatedRoot_145.w_12_0 ctx) (parallel_generatedRoot_145.w_13_0 ctx) (parallel_generatedRoot_145.w_14_0 ctx) (parallel_generatedRoot_145.w_15_0 ctx) (parallel_generatedRoot_145.w_16_0 ctx) (parallel_generatedRoot_145.w_17_0 ctx) (parallel_generatedRoot_145.w_17_1 ctx) (parallel_generatedRoot_145.w_17_2 ctx) (parallel_generatedRoot_145.w_17_3 ctx) outputs) :
    parallel_generatedRoot_145.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_145.Facts  (ctx : parallel_generatedRoot_145Ctx) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  c0 : parallel_generatedRoot_145.Facts_0 ctx outputs

theorem parallel_generatedRoot_145.facts  (ctx : parallel_generatedRoot_145Ctx) {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_145.body backend hashModel tape path params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_145.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_145.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_149.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) where
  inputs : (Fin 2607022080 → mxxMat1) × (Fin 2607022080 → mxxMat1) × (Fin 2607022080 → mxxMat1) × (Fin 2607022080 → mxxMat1) × Unit
  witness : parallel_generatedRoot_149.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_149Ctx" => (parallel_generatedRoot_149.Ctx backend hashModel tape path params i_0)

def parallel_generatedRoot_149.w_0_0  (ctx : parallel_generatedRoot_149Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.1

def parallel_generatedRoot_149.w_1_0  (ctx : parallel_generatedRoot_149Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.2.1

def parallel_generatedRoot_149.w_2_0  (ctx : parallel_generatedRoot_149Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_149.w_3_0  (ctx : parallel_generatedRoot_149Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_149.w_4_0  (ctx : parallel_generatedRoot_149Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_0

def parallel_generatedRoot_149.w_4_1  (ctx : parallel_generatedRoot_149Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_1

def parallel_generatedRoot_149.w_4_2  (ctx : parallel_generatedRoot_149Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_2

def parallel_generatedRoot_149.w_4_3  (ctx : parallel_generatedRoot_149Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_3

def parallel_generatedRoot_149.w_5_0  (ctx : parallel_generatedRoot_149Ctx) : mxxMat1 :=
  ctx.witness.w_5_0

def parallel_generatedRoot_149.w_6_0  (ctx : parallel_generatedRoot_149Ctx) : mxxMat1 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_149.w_7_0  (ctx : parallel_generatedRoot_149Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_149.w_5_0 ctx) (parallel_generatedRoot_149.w_6_0 ctx)

def parallel_generatedRoot_149.w_8_0  (ctx : parallel_generatedRoot_149Ctx) : mxxMat1 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_149.w_9_0  (ctx : parallel_generatedRoot_149Ctx) : mxxMat1 :=
  ctx.witness.w_9_0

def parallel_generatedRoot_149.w_10_0  (ctx : parallel_generatedRoot_149Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_149.w_8_0 ctx) (parallel_generatedRoot_149.w_9_0 ctx)

def parallel_generatedRoot_149.w_11_0  (ctx : parallel_generatedRoot_149Ctx) : mxxMat1 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_149.w_12_0  (ctx : parallel_generatedRoot_149Ctx) : mxxMat1 :=
  ctx.witness.w_12_0

def parallel_generatedRoot_149.w_13_0  (ctx : parallel_generatedRoot_149Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_149.w_11_0 ctx) (parallel_generatedRoot_149.w_12_0 ctx)

def parallel_generatedRoot_149.w_14_0  (ctx : parallel_generatedRoot_149Ctx) : mxxMat1 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_149.w_15_0  (ctx : parallel_generatedRoot_149Ctx) : mxxMat1 :=
  ctx.witness.w_15_0

def parallel_generatedRoot_149.w_16_0  (ctx : parallel_generatedRoot_149Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_149.w_14_0 ctx) (parallel_generatedRoot_149.w_15_0 ctx)

def parallel_generatedRoot_149.w_17_0  (ctx : parallel_generatedRoot_149Ctx) : mxxMat1 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_149.w_17_1  (ctx : parallel_generatedRoot_149Ctx) : mxxMat1 :=
  ctx.witness.w_17_1

def parallel_generatedRoot_149.w_17_2  (ctx : parallel_generatedRoot_149Ctx) : mxxMat1 :=
  ctx.witness.w_17_2

def parallel_generatedRoot_149.w_17_3  (ctx : parallel_generatedRoot_149Ctx) : mxxMat1 :=
  ctx.witness.w_17_3

def parallel_generatedRoot_149.outs  (ctx : parallel_generatedRoot_149Ctx) : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit :=
  ((parallel_generatedRoot_149.w_17_0 ctx), ((parallel_generatedRoot_149.w_17_1 ctx), ((parallel_generatedRoot_149.w_17_2 ctx), ((parallel_generatedRoot_149.w_17_3 ctx), ()))))

structure parallel_generatedRoot_149.Facts_0  (ctx : parallel_generatedRoot_149Ctx) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  t0 : (130) = 130
  h4_3 : (∀ i : Fin 130, parallel_parallel_generatedRoot_149_4 backend hashModel tape (path ++ [4, i.val]) params i_0 i ((parallel_generatedRoot_149.w_0_0 ctx), ((parallel_generatedRoot_149.w_1_0 ctx), ((parallel_generatedRoot_149.w_2_0 ctx), ((parallel_generatedRoot_149.w_3_0 ctx), ())))) (((parallel_generatedRoot_149.w_4_0 ctx) i), (((parallel_generatedRoot_149.w_4_1 ctx) i), (((parallel_generatedRoot_149.w_4_2 ctx) i), (((parallel_generatedRoot_149.w_4_3 ctx) i), ())))))
  h5_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_149.w_4_0 ctx) (0) (parallel_generatedRoot_149.w_5_0 ctx)
  h6_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_149.w_4_0 ctx) (1) (parallel_generatedRoot_149.w_6_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_149.w_4_1 ctx) (0) (parallel_generatedRoot_149.w_8_0 ctx)
  h9_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_149.w_4_1 ctx) (1) (parallel_generatedRoot_149.w_9_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_149.w_4_2 ctx) (0) (parallel_generatedRoot_149.w_11_0 ctx)
  h12_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_149.w_4_2 ctx) (1) (parallel_generatedRoot_149.w_12_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_149.w_4_3 ctx) (0) (parallel_generatedRoot_149.w_14_0 ctx)
  h15_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_149.w_4_3 ctx) (1) (parallel_generatedRoot_149.w_15_0 ctx)
  t10 : 0 ≤ (128)
  h17_3 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_parallel_generatedRoot_149_17 backend hashModel tape (path ++ [17, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, ((parallel_generatedRoot_149.w_4_0 ctx), ((parallel_generatedRoot_149.w_4_1 ctx), ((parallel_generatedRoot_149.w_4_2 ctx), ((parallel_generatedRoot_149.w_4_3 ctx), ())))))))) next) (Int.toNat (128)) ((parallel_generatedRoot_149.w_7_0 ctx), ((parallel_generatedRoot_149.w_10_0 ctx), ((parallel_generatedRoot_149.w_13_0 ctx), ((parallel_generatedRoot_149.w_16_0 ctx), ())))) ((parallel_generatedRoot_149.w_17_0 ctx), ((parallel_generatedRoot_149.w_17_1 ctx), ((parallel_generatedRoot_149.w_17_2 ctx), ((parallel_generatedRoot_149.w_17_3 ctx), ()))))
  hout : outputs = parallel_generatedRoot_149.outs ctx

theorem parallel_generatedRoot_149.facts_0  {ctx : parallel_generatedRoot_149Ctx} {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_149.constraints_0 backend hashModel tape path params i_0 (parallel_generatedRoot_149.w_0_0 ctx) (parallel_generatedRoot_149.w_1_0 ctx) (parallel_generatedRoot_149.w_2_0 ctx) (parallel_generatedRoot_149.w_3_0 ctx) (parallel_generatedRoot_149.w_4_0 ctx) (parallel_generatedRoot_149.w_4_1 ctx) (parallel_generatedRoot_149.w_4_2 ctx) (parallel_generatedRoot_149.w_4_3 ctx) (parallel_generatedRoot_149.w_5_0 ctx) (parallel_generatedRoot_149.w_6_0 ctx) (parallel_generatedRoot_149.w_7_0 ctx) (parallel_generatedRoot_149.w_8_0 ctx) (parallel_generatedRoot_149.w_9_0 ctx) (parallel_generatedRoot_149.w_10_0 ctx) (parallel_generatedRoot_149.w_11_0 ctx) (parallel_generatedRoot_149.w_12_0 ctx) (parallel_generatedRoot_149.w_13_0 ctx) (parallel_generatedRoot_149.w_14_0 ctx) (parallel_generatedRoot_149.w_15_0 ctx) (parallel_generatedRoot_149.w_16_0 ctx) (parallel_generatedRoot_149.w_17_0 ctx) (parallel_generatedRoot_149.w_17_1 ctx) (parallel_generatedRoot_149.w_17_2 ctx) (parallel_generatedRoot_149.w_17_3 ctx) outputs) :
    parallel_generatedRoot_149.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_149.Facts  (ctx : parallel_generatedRoot_149Ctx) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  c0 : parallel_generatedRoot_149.Facts_0 ctx outputs

theorem parallel_generatedRoot_149.facts  (ctx : parallel_generatedRoot_149Ctx) {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_149.body backend hashModel tape path params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_149.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_149.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_153.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) where
  inputs : (Fin 2607022080 → mxxMat1) × (Fin 2607022080 → mxxMat1) × (Fin 2607022080 → mxxMat1) × (Fin 2607022080 → mxxMat1) × Unit
  witness : parallel_generatedRoot_153.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_153Ctx" => (parallel_generatedRoot_153.Ctx backend hashModel tape path params i_0)

def parallel_generatedRoot_153.w_0_0  (ctx : parallel_generatedRoot_153Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.1

def parallel_generatedRoot_153.w_1_0  (ctx : parallel_generatedRoot_153Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.2.1

def parallel_generatedRoot_153.w_2_0  (ctx : parallel_generatedRoot_153Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_153.w_3_0  (ctx : parallel_generatedRoot_153Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_153.w_4_0  (ctx : parallel_generatedRoot_153Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_0

def parallel_generatedRoot_153.w_4_1  (ctx : parallel_generatedRoot_153Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_1

def parallel_generatedRoot_153.w_4_2  (ctx : parallel_generatedRoot_153Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_2

def parallel_generatedRoot_153.w_4_3  (ctx : parallel_generatedRoot_153Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_3

def parallel_generatedRoot_153.w_5_0  (ctx : parallel_generatedRoot_153Ctx) : mxxMat1 :=
  ctx.witness.w_5_0

def parallel_generatedRoot_153.w_6_0  (ctx : parallel_generatedRoot_153Ctx) : mxxMat1 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_153.w_7_0  (ctx : parallel_generatedRoot_153Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_153.w_5_0 ctx) (parallel_generatedRoot_153.w_6_0 ctx)

def parallel_generatedRoot_153.w_8_0  (ctx : parallel_generatedRoot_153Ctx) : mxxMat1 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_153.w_9_0  (ctx : parallel_generatedRoot_153Ctx) : mxxMat1 :=
  ctx.witness.w_9_0

def parallel_generatedRoot_153.w_10_0  (ctx : parallel_generatedRoot_153Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_153.w_8_0 ctx) (parallel_generatedRoot_153.w_9_0 ctx)

def parallel_generatedRoot_153.w_11_0  (ctx : parallel_generatedRoot_153Ctx) : mxxMat1 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_153.w_12_0  (ctx : parallel_generatedRoot_153Ctx) : mxxMat1 :=
  ctx.witness.w_12_0

def parallel_generatedRoot_153.w_13_0  (ctx : parallel_generatedRoot_153Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_153.w_11_0 ctx) (parallel_generatedRoot_153.w_12_0 ctx)

def parallel_generatedRoot_153.w_14_0  (ctx : parallel_generatedRoot_153Ctx) : mxxMat1 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_153.w_15_0  (ctx : parallel_generatedRoot_153Ctx) : mxxMat1 :=
  ctx.witness.w_15_0

def parallel_generatedRoot_153.w_16_0  (ctx : parallel_generatedRoot_153Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_153.w_14_0 ctx) (parallel_generatedRoot_153.w_15_0 ctx)

def parallel_generatedRoot_153.w_17_0  (ctx : parallel_generatedRoot_153Ctx) : mxxMat1 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_153.w_17_1  (ctx : parallel_generatedRoot_153Ctx) : mxxMat1 :=
  ctx.witness.w_17_1

def parallel_generatedRoot_153.w_17_2  (ctx : parallel_generatedRoot_153Ctx) : mxxMat1 :=
  ctx.witness.w_17_2

def parallel_generatedRoot_153.w_17_3  (ctx : parallel_generatedRoot_153Ctx) : mxxMat1 :=
  ctx.witness.w_17_3

def parallel_generatedRoot_153.outs  (ctx : parallel_generatedRoot_153Ctx) : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit :=
  ((parallel_generatedRoot_153.w_17_0 ctx), ((parallel_generatedRoot_153.w_17_1 ctx), ((parallel_generatedRoot_153.w_17_2 ctx), ((parallel_generatedRoot_153.w_17_3 ctx), ()))))

structure parallel_generatedRoot_153.Facts_0  (ctx : parallel_generatedRoot_153Ctx) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  t0 : (130) = 130
  h4_3 : (∀ i : Fin 130, parallel_parallel_generatedRoot_153_4 backend hashModel tape (path ++ [4, i.val]) params i_0 i ((parallel_generatedRoot_153.w_0_0 ctx), ((parallel_generatedRoot_153.w_1_0 ctx), ((parallel_generatedRoot_153.w_2_0 ctx), ((parallel_generatedRoot_153.w_3_0 ctx), ())))) (((parallel_generatedRoot_153.w_4_0 ctx) i), (((parallel_generatedRoot_153.w_4_1 ctx) i), (((parallel_generatedRoot_153.w_4_2 ctx) i), (((parallel_generatedRoot_153.w_4_3 ctx) i), ())))))
  h5_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_153.w_4_0 ctx) (0) (parallel_generatedRoot_153.w_5_0 ctx)
  h6_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_153.w_4_0 ctx) (1) (parallel_generatedRoot_153.w_6_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_153.w_4_1 ctx) (0) (parallel_generatedRoot_153.w_8_0 ctx)
  h9_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_153.w_4_1 ctx) (1) (parallel_generatedRoot_153.w_9_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_153.w_4_2 ctx) (0) (parallel_generatedRoot_153.w_11_0 ctx)
  h12_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_153.w_4_2 ctx) (1) (parallel_generatedRoot_153.w_12_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_153.w_4_3 ctx) (0) (parallel_generatedRoot_153.w_14_0 ctx)
  h15_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_153.w_4_3 ctx) (1) (parallel_generatedRoot_153.w_15_0 ctx)
  t10 : 0 ≤ (128)
  h17_3 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_parallel_generatedRoot_153_17 backend hashModel tape (path ++ [17, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, ((parallel_generatedRoot_153.w_4_0 ctx), ((parallel_generatedRoot_153.w_4_1 ctx), ((parallel_generatedRoot_153.w_4_2 ctx), ((parallel_generatedRoot_153.w_4_3 ctx), ())))))))) next) (Int.toNat (128)) ((parallel_generatedRoot_153.w_7_0 ctx), ((parallel_generatedRoot_153.w_10_0 ctx), ((parallel_generatedRoot_153.w_13_0 ctx), ((parallel_generatedRoot_153.w_16_0 ctx), ())))) ((parallel_generatedRoot_153.w_17_0 ctx), ((parallel_generatedRoot_153.w_17_1 ctx), ((parallel_generatedRoot_153.w_17_2 ctx), ((parallel_generatedRoot_153.w_17_3 ctx), ()))))
  hout : outputs = parallel_generatedRoot_153.outs ctx

theorem parallel_generatedRoot_153.facts_0  {ctx : parallel_generatedRoot_153Ctx} {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_153.constraints_0 backend hashModel tape path params i_0 (parallel_generatedRoot_153.w_0_0 ctx) (parallel_generatedRoot_153.w_1_0 ctx) (parallel_generatedRoot_153.w_2_0 ctx) (parallel_generatedRoot_153.w_3_0 ctx) (parallel_generatedRoot_153.w_4_0 ctx) (parallel_generatedRoot_153.w_4_1 ctx) (parallel_generatedRoot_153.w_4_2 ctx) (parallel_generatedRoot_153.w_4_3 ctx) (parallel_generatedRoot_153.w_5_0 ctx) (parallel_generatedRoot_153.w_6_0 ctx) (parallel_generatedRoot_153.w_7_0 ctx) (parallel_generatedRoot_153.w_8_0 ctx) (parallel_generatedRoot_153.w_9_0 ctx) (parallel_generatedRoot_153.w_10_0 ctx) (parallel_generatedRoot_153.w_11_0 ctx) (parallel_generatedRoot_153.w_12_0 ctx) (parallel_generatedRoot_153.w_13_0 ctx) (parallel_generatedRoot_153.w_14_0 ctx) (parallel_generatedRoot_153.w_15_0 ctx) (parallel_generatedRoot_153.w_16_0 ctx) (parallel_generatedRoot_153.w_17_0 ctx) (parallel_generatedRoot_153.w_17_1 ctx) (parallel_generatedRoot_153.w_17_2 ctx) (parallel_generatedRoot_153.w_17_3 ctx) outputs) :
    parallel_generatedRoot_153.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_153.Facts  (ctx : parallel_generatedRoot_153Ctx) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  c0 : parallel_generatedRoot_153.Facts_0 ctx outputs

theorem parallel_generatedRoot_153.facts  (ctx : parallel_generatedRoot_153Ctx) {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_153.body backend hashModel tape path params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_153.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_153.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_157.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) where
  inputs : (Fin 2607022080 → mxxMat1) × (Fin 2607022080 → mxxMat1) × (Fin 2607022080 → mxxMat1) × (Fin 2607022080 → mxxMat1) × Unit
  witness : parallel_generatedRoot_157.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_157Ctx" => (parallel_generatedRoot_157.Ctx backend hashModel tape path params i_0)

def parallel_generatedRoot_157.w_0_0  (ctx : parallel_generatedRoot_157Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.1

def parallel_generatedRoot_157.w_1_0  (ctx : parallel_generatedRoot_157Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.2.1

def parallel_generatedRoot_157.w_2_0  (ctx : parallel_generatedRoot_157Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_157.w_3_0  (ctx : parallel_generatedRoot_157Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_157.w_4_0  (ctx : parallel_generatedRoot_157Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_0

def parallel_generatedRoot_157.w_4_1  (ctx : parallel_generatedRoot_157Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_1

def parallel_generatedRoot_157.w_4_2  (ctx : parallel_generatedRoot_157Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_2

def parallel_generatedRoot_157.w_4_3  (ctx : parallel_generatedRoot_157Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_3

def parallel_generatedRoot_157.w_5_0  (ctx : parallel_generatedRoot_157Ctx) : mxxMat1 :=
  ctx.witness.w_5_0

def parallel_generatedRoot_157.w_6_0  (ctx : parallel_generatedRoot_157Ctx) : mxxMat1 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_157.w_7_0  (ctx : parallel_generatedRoot_157Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_157.w_5_0 ctx) (parallel_generatedRoot_157.w_6_0 ctx)

def parallel_generatedRoot_157.w_8_0  (ctx : parallel_generatedRoot_157Ctx) : mxxMat1 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_157.w_9_0  (ctx : parallel_generatedRoot_157Ctx) : mxxMat1 :=
  ctx.witness.w_9_0

def parallel_generatedRoot_157.w_10_0  (ctx : parallel_generatedRoot_157Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_157.w_8_0 ctx) (parallel_generatedRoot_157.w_9_0 ctx)

def parallel_generatedRoot_157.w_11_0  (ctx : parallel_generatedRoot_157Ctx) : mxxMat1 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_157.w_12_0  (ctx : parallel_generatedRoot_157Ctx) : mxxMat1 :=
  ctx.witness.w_12_0

def parallel_generatedRoot_157.w_13_0  (ctx : parallel_generatedRoot_157Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_157.w_11_0 ctx) (parallel_generatedRoot_157.w_12_0 ctx)

def parallel_generatedRoot_157.w_14_0  (ctx : parallel_generatedRoot_157Ctx) : mxxMat1 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_157.w_15_0  (ctx : parallel_generatedRoot_157Ctx) : mxxMat1 :=
  ctx.witness.w_15_0

def parallel_generatedRoot_157.w_16_0  (ctx : parallel_generatedRoot_157Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_157.w_14_0 ctx) (parallel_generatedRoot_157.w_15_0 ctx)

def parallel_generatedRoot_157.w_17_0  (ctx : parallel_generatedRoot_157Ctx) : mxxMat1 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_157.w_17_1  (ctx : parallel_generatedRoot_157Ctx) : mxxMat1 :=
  ctx.witness.w_17_1

def parallel_generatedRoot_157.w_17_2  (ctx : parallel_generatedRoot_157Ctx) : mxxMat1 :=
  ctx.witness.w_17_2

def parallel_generatedRoot_157.w_17_3  (ctx : parallel_generatedRoot_157Ctx) : mxxMat1 :=
  ctx.witness.w_17_3

def parallel_generatedRoot_157.outs  (ctx : parallel_generatedRoot_157Ctx) : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit :=
  ((parallel_generatedRoot_157.w_17_0 ctx), ((parallel_generatedRoot_157.w_17_1 ctx), ((parallel_generatedRoot_157.w_17_2 ctx), ((parallel_generatedRoot_157.w_17_3 ctx), ()))))

structure parallel_generatedRoot_157.Facts_0  (ctx : parallel_generatedRoot_157Ctx) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  t0 : (130) = 130
  h4_3 : (∀ i : Fin 130, parallel_parallel_generatedRoot_157_4 backend hashModel tape (path ++ [4, i.val]) params i_0 i ((parallel_generatedRoot_157.w_0_0 ctx), ((parallel_generatedRoot_157.w_1_0 ctx), ((parallel_generatedRoot_157.w_2_0 ctx), ((parallel_generatedRoot_157.w_3_0 ctx), ())))) (((parallel_generatedRoot_157.w_4_0 ctx) i), (((parallel_generatedRoot_157.w_4_1 ctx) i), (((parallel_generatedRoot_157.w_4_2 ctx) i), (((parallel_generatedRoot_157.w_4_3 ctx) i), ())))))
  h5_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_157.w_4_0 ctx) (0) (parallel_generatedRoot_157.w_5_0 ctx)
  h6_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_157.w_4_0 ctx) (1) (parallel_generatedRoot_157.w_6_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_157.w_4_1 ctx) (0) (parallel_generatedRoot_157.w_8_0 ctx)
  h9_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_157.w_4_1 ctx) (1) (parallel_generatedRoot_157.w_9_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_157.w_4_2 ctx) (0) (parallel_generatedRoot_157.w_11_0 ctx)
  h12_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_157.w_4_2 ctx) (1) (parallel_generatedRoot_157.w_12_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_157.w_4_3 ctx) (0) (parallel_generatedRoot_157.w_14_0 ctx)
  h15_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_157.w_4_3 ctx) (1) (parallel_generatedRoot_157.w_15_0 ctx)
  t10 : 0 ≤ (128)
  h17_3 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_parallel_generatedRoot_157_17 backend hashModel tape (path ++ [17, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, ((parallel_generatedRoot_157.w_4_0 ctx), ((parallel_generatedRoot_157.w_4_1 ctx), ((parallel_generatedRoot_157.w_4_2 ctx), ((parallel_generatedRoot_157.w_4_3 ctx), ())))))))) next) (Int.toNat (128)) ((parallel_generatedRoot_157.w_7_0 ctx), ((parallel_generatedRoot_157.w_10_0 ctx), ((parallel_generatedRoot_157.w_13_0 ctx), ((parallel_generatedRoot_157.w_16_0 ctx), ())))) ((parallel_generatedRoot_157.w_17_0 ctx), ((parallel_generatedRoot_157.w_17_1 ctx), ((parallel_generatedRoot_157.w_17_2 ctx), ((parallel_generatedRoot_157.w_17_3 ctx), ()))))
  hout : outputs = parallel_generatedRoot_157.outs ctx

theorem parallel_generatedRoot_157.facts_0  {ctx : parallel_generatedRoot_157Ctx} {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_157.constraints_0 backend hashModel tape path params i_0 (parallel_generatedRoot_157.w_0_0 ctx) (parallel_generatedRoot_157.w_1_0 ctx) (parallel_generatedRoot_157.w_2_0 ctx) (parallel_generatedRoot_157.w_3_0 ctx) (parallel_generatedRoot_157.w_4_0 ctx) (parallel_generatedRoot_157.w_4_1 ctx) (parallel_generatedRoot_157.w_4_2 ctx) (parallel_generatedRoot_157.w_4_3 ctx) (parallel_generatedRoot_157.w_5_0 ctx) (parallel_generatedRoot_157.w_6_0 ctx) (parallel_generatedRoot_157.w_7_0 ctx) (parallel_generatedRoot_157.w_8_0 ctx) (parallel_generatedRoot_157.w_9_0 ctx) (parallel_generatedRoot_157.w_10_0 ctx) (parallel_generatedRoot_157.w_11_0 ctx) (parallel_generatedRoot_157.w_12_0 ctx) (parallel_generatedRoot_157.w_13_0 ctx) (parallel_generatedRoot_157.w_14_0 ctx) (parallel_generatedRoot_157.w_15_0 ctx) (parallel_generatedRoot_157.w_16_0 ctx) (parallel_generatedRoot_157.w_17_0 ctx) (parallel_generatedRoot_157.w_17_1 ctx) (parallel_generatedRoot_157.w_17_2 ctx) (parallel_generatedRoot_157.w_17_3 ctx) outputs) :
    parallel_generatedRoot_157.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_157.Facts  (ctx : parallel_generatedRoot_157Ctx) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  c0 : parallel_generatedRoot_157.Facts_0 ctx outputs

theorem parallel_generatedRoot_157.facts  (ctx : parallel_generatedRoot_157Ctx) {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_157.body backend hashModel tape path params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_157.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_157.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_161.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) where
  inputs : (Fin 2607022080 → mxxMat1) × (Fin 2607022080 → mxxMat1) × (Fin 2607022080 → mxxMat1) × (Fin 2607022080 → mxxMat1) × Unit
  witness : parallel_generatedRoot_161.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_161Ctx" => (parallel_generatedRoot_161.Ctx backend hashModel tape path params i_0)

def parallel_generatedRoot_161.w_0_0  (ctx : parallel_generatedRoot_161Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.1

def parallel_generatedRoot_161.w_1_0  (ctx : parallel_generatedRoot_161Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.2.1

def parallel_generatedRoot_161.w_2_0  (ctx : parallel_generatedRoot_161Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_161.w_3_0  (ctx : parallel_generatedRoot_161Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_161.w_4_0  (ctx : parallel_generatedRoot_161Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_0

def parallel_generatedRoot_161.w_4_1  (ctx : parallel_generatedRoot_161Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_1

def parallel_generatedRoot_161.w_4_2  (ctx : parallel_generatedRoot_161Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_2

def parallel_generatedRoot_161.w_4_3  (ctx : parallel_generatedRoot_161Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_3

def parallel_generatedRoot_161.w_5_0  (ctx : parallel_generatedRoot_161Ctx) : mxxMat1 :=
  ctx.witness.w_5_0

def parallel_generatedRoot_161.w_6_0  (ctx : parallel_generatedRoot_161Ctx) : mxxMat1 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_161.w_7_0  (ctx : parallel_generatedRoot_161Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_161.w_5_0 ctx) (parallel_generatedRoot_161.w_6_0 ctx)

def parallel_generatedRoot_161.w_8_0  (ctx : parallel_generatedRoot_161Ctx) : mxxMat1 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_161.w_9_0  (ctx : parallel_generatedRoot_161Ctx) : mxxMat1 :=
  ctx.witness.w_9_0

def parallel_generatedRoot_161.w_10_0  (ctx : parallel_generatedRoot_161Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_161.w_8_0 ctx) (parallel_generatedRoot_161.w_9_0 ctx)

def parallel_generatedRoot_161.w_11_0  (ctx : parallel_generatedRoot_161Ctx) : mxxMat1 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_161.w_12_0  (ctx : parallel_generatedRoot_161Ctx) : mxxMat1 :=
  ctx.witness.w_12_0

def parallel_generatedRoot_161.w_13_0  (ctx : parallel_generatedRoot_161Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_161.w_11_0 ctx) (parallel_generatedRoot_161.w_12_0 ctx)

def parallel_generatedRoot_161.w_14_0  (ctx : parallel_generatedRoot_161Ctx) : mxxMat1 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_161.w_15_0  (ctx : parallel_generatedRoot_161Ctx) : mxxMat1 :=
  ctx.witness.w_15_0

def parallel_generatedRoot_161.w_16_0  (ctx : parallel_generatedRoot_161Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_161.w_14_0 ctx) (parallel_generatedRoot_161.w_15_0 ctx)

def parallel_generatedRoot_161.w_17_0  (ctx : parallel_generatedRoot_161Ctx) : mxxMat1 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_161.w_17_1  (ctx : parallel_generatedRoot_161Ctx) : mxxMat1 :=
  ctx.witness.w_17_1

def parallel_generatedRoot_161.w_17_2  (ctx : parallel_generatedRoot_161Ctx) : mxxMat1 :=
  ctx.witness.w_17_2

def parallel_generatedRoot_161.w_17_3  (ctx : parallel_generatedRoot_161Ctx) : mxxMat1 :=
  ctx.witness.w_17_3

def parallel_generatedRoot_161.outs  (ctx : parallel_generatedRoot_161Ctx) : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit :=
  ((parallel_generatedRoot_161.w_17_0 ctx), ((parallel_generatedRoot_161.w_17_1 ctx), ((parallel_generatedRoot_161.w_17_2 ctx), ((parallel_generatedRoot_161.w_17_3 ctx), ()))))

structure parallel_generatedRoot_161.Facts_0  (ctx : parallel_generatedRoot_161Ctx) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  t0 : (130) = 130
  h4_3 : (∀ i : Fin 130, parallel_parallel_generatedRoot_161_4 backend hashModel tape (path ++ [4, i.val]) params i_0 i ((parallel_generatedRoot_161.w_0_0 ctx), ((parallel_generatedRoot_161.w_1_0 ctx), ((parallel_generatedRoot_161.w_2_0 ctx), ((parallel_generatedRoot_161.w_3_0 ctx), ())))) (((parallel_generatedRoot_161.w_4_0 ctx) i), (((parallel_generatedRoot_161.w_4_1 ctx) i), (((parallel_generatedRoot_161.w_4_2 ctx) i), (((parallel_generatedRoot_161.w_4_3 ctx) i), ())))))
  h5_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_161.w_4_0 ctx) (0) (parallel_generatedRoot_161.w_5_0 ctx)
  h6_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_161.w_4_0 ctx) (1) (parallel_generatedRoot_161.w_6_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_161.w_4_1 ctx) (0) (parallel_generatedRoot_161.w_8_0 ctx)
  h9_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_161.w_4_1 ctx) (1) (parallel_generatedRoot_161.w_9_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_161.w_4_2 ctx) (0) (parallel_generatedRoot_161.w_11_0 ctx)
  h12_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_161.w_4_2 ctx) (1) (parallel_generatedRoot_161.w_12_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_161.w_4_3 ctx) (0) (parallel_generatedRoot_161.w_14_0 ctx)
  h15_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_161.w_4_3 ctx) (1) (parallel_generatedRoot_161.w_15_0 ctx)
  t10 : 0 ≤ (128)
  h17_3 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_parallel_generatedRoot_161_17 backend hashModel tape (path ++ [17, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, ((parallel_generatedRoot_161.w_4_0 ctx), ((parallel_generatedRoot_161.w_4_1 ctx), ((parallel_generatedRoot_161.w_4_2 ctx), ((parallel_generatedRoot_161.w_4_3 ctx), ())))))))) next) (Int.toNat (128)) ((parallel_generatedRoot_161.w_7_0 ctx), ((parallel_generatedRoot_161.w_10_0 ctx), ((parallel_generatedRoot_161.w_13_0 ctx), ((parallel_generatedRoot_161.w_16_0 ctx), ())))) ((parallel_generatedRoot_161.w_17_0 ctx), ((parallel_generatedRoot_161.w_17_1 ctx), ((parallel_generatedRoot_161.w_17_2 ctx), ((parallel_generatedRoot_161.w_17_3 ctx), ()))))
  hout : outputs = parallel_generatedRoot_161.outs ctx

theorem parallel_generatedRoot_161.facts_0  {ctx : parallel_generatedRoot_161Ctx} {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_161.constraints_0 backend hashModel tape path params i_0 (parallel_generatedRoot_161.w_0_0 ctx) (parallel_generatedRoot_161.w_1_0 ctx) (parallel_generatedRoot_161.w_2_0 ctx) (parallel_generatedRoot_161.w_3_0 ctx) (parallel_generatedRoot_161.w_4_0 ctx) (parallel_generatedRoot_161.w_4_1 ctx) (parallel_generatedRoot_161.w_4_2 ctx) (parallel_generatedRoot_161.w_4_3 ctx) (parallel_generatedRoot_161.w_5_0 ctx) (parallel_generatedRoot_161.w_6_0 ctx) (parallel_generatedRoot_161.w_7_0 ctx) (parallel_generatedRoot_161.w_8_0 ctx) (parallel_generatedRoot_161.w_9_0 ctx) (parallel_generatedRoot_161.w_10_0 ctx) (parallel_generatedRoot_161.w_11_0 ctx) (parallel_generatedRoot_161.w_12_0 ctx) (parallel_generatedRoot_161.w_13_0 ctx) (parallel_generatedRoot_161.w_14_0 ctx) (parallel_generatedRoot_161.w_15_0 ctx) (parallel_generatedRoot_161.w_16_0 ctx) (parallel_generatedRoot_161.w_17_0 ctx) (parallel_generatedRoot_161.w_17_1 ctx) (parallel_generatedRoot_161.w_17_2 ctx) (parallel_generatedRoot_161.w_17_3 ctx) outputs) :
    parallel_generatedRoot_161.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_161.Facts  (ctx : parallel_generatedRoot_161Ctx) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  c0 : parallel_generatedRoot_161.Facts_0 ctx outputs

theorem parallel_generatedRoot_161.facts  (ctx : parallel_generatedRoot_161Ctx) {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_161.body backend hashModel tape path params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_161.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_161.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_165.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) where
  inputs : (Fin 2607022080 → mxxMat1) × (Fin 2607022080 → mxxMat1) × (Fin 2607022080 → mxxMat1) × (Fin 2607022080 → mxxMat1) × Unit
  witness : parallel_generatedRoot_165.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_165Ctx" => (parallel_generatedRoot_165.Ctx backend hashModel tape path params i_0)

def parallel_generatedRoot_165.w_0_0  (ctx : parallel_generatedRoot_165Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.1

def parallel_generatedRoot_165.w_1_0  (ctx : parallel_generatedRoot_165Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.2.1

def parallel_generatedRoot_165.w_2_0  (ctx : parallel_generatedRoot_165Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_165.w_3_0  (ctx : parallel_generatedRoot_165Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_165.w_4_0  (ctx : parallel_generatedRoot_165Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_0

def parallel_generatedRoot_165.w_4_1  (ctx : parallel_generatedRoot_165Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_1

def parallel_generatedRoot_165.w_4_2  (ctx : parallel_generatedRoot_165Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_2

def parallel_generatedRoot_165.w_4_3  (ctx : parallel_generatedRoot_165Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_3

def parallel_generatedRoot_165.w_5_0  (ctx : parallel_generatedRoot_165Ctx) : mxxMat1 :=
  ctx.witness.w_5_0

def parallel_generatedRoot_165.w_6_0  (ctx : parallel_generatedRoot_165Ctx) : mxxMat1 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_165.w_7_0  (ctx : parallel_generatedRoot_165Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_165.w_5_0 ctx) (parallel_generatedRoot_165.w_6_0 ctx)

def parallel_generatedRoot_165.w_8_0  (ctx : parallel_generatedRoot_165Ctx) : mxxMat1 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_165.w_9_0  (ctx : parallel_generatedRoot_165Ctx) : mxxMat1 :=
  ctx.witness.w_9_0

def parallel_generatedRoot_165.w_10_0  (ctx : parallel_generatedRoot_165Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_165.w_8_0 ctx) (parallel_generatedRoot_165.w_9_0 ctx)

def parallel_generatedRoot_165.w_11_0  (ctx : parallel_generatedRoot_165Ctx) : mxxMat1 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_165.w_12_0  (ctx : parallel_generatedRoot_165Ctx) : mxxMat1 :=
  ctx.witness.w_12_0

def parallel_generatedRoot_165.w_13_0  (ctx : parallel_generatedRoot_165Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_165.w_11_0 ctx) (parallel_generatedRoot_165.w_12_0 ctx)

def parallel_generatedRoot_165.w_14_0  (ctx : parallel_generatedRoot_165Ctx) : mxxMat1 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_165.w_15_0  (ctx : parallel_generatedRoot_165Ctx) : mxxMat1 :=
  ctx.witness.w_15_0

def parallel_generatedRoot_165.w_16_0  (ctx : parallel_generatedRoot_165Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_165.w_14_0 ctx) (parallel_generatedRoot_165.w_15_0 ctx)

def parallel_generatedRoot_165.w_17_0  (ctx : parallel_generatedRoot_165Ctx) : mxxMat1 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_165.w_17_1  (ctx : parallel_generatedRoot_165Ctx) : mxxMat1 :=
  ctx.witness.w_17_1

def parallel_generatedRoot_165.w_17_2  (ctx : parallel_generatedRoot_165Ctx) : mxxMat1 :=
  ctx.witness.w_17_2

def parallel_generatedRoot_165.w_17_3  (ctx : parallel_generatedRoot_165Ctx) : mxxMat1 :=
  ctx.witness.w_17_3

def parallel_generatedRoot_165.outs  (ctx : parallel_generatedRoot_165Ctx) : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit :=
  ((parallel_generatedRoot_165.w_17_0 ctx), ((parallel_generatedRoot_165.w_17_1 ctx), ((parallel_generatedRoot_165.w_17_2 ctx), ((parallel_generatedRoot_165.w_17_3 ctx), ()))))

structure parallel_generatedRoot_165.Facts_0  (ctx : parallel_generatedRoot_165Ctx) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  t0 : (130) = 130
  h4_3 : (∀ i : Fin 130, parallel_parallel_generatedRoot_165_4 backend hashModel tape (path ++ [4, i.val]) params i_0 i ((parallel_generatedRoot_165.w_0_0 ctx), ((parallel_generatedRoot_165.w_1_0 ctx), ((parallel_generatedRoot_165.w_2_0 ctx), ((parallel_generatedRoot_165.w_3_0 ctx), ())))) (((parallel_generatedRoot_165.w_4_0 ctx) i), (((parallel_generatedRoot_165.w_4_1 ctx) i), (((parallel_generatedRoot_165.w_4_2 ctx) i), (((parallel_generatedRoot_165.w_4_3 ctx) i), ())))))
  h5_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_165.w_4_0 ctx) (0) (parallel_generatedRoot_165.w_5_0 ctx)
  h6_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_165.w_4_0 ctx) (1) (parallel_generatedRoot_165.w_6_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_165.w_4_1 ctx) (0) (parallel_generatedRoot_165.w_8_0 ctx)
  h9_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_165.w_4_1 ctx) (1) (parallel_generatedRoot_165.w_9_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_165.w_4_2 ctx) (0) (parallel_generatedRoot_165.w_11_0 ctx)
  h12_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_165.w_4_2 ctx) (1) (parallel_generatedRoot_165.w_12_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_165.w_4_3 ctx) (0) (parallel_generatedRoot_165.w_14_0 ctx)
  h15_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_165.w_4_3 ctx) (1) (parallel_generatedRoot_165.w_15_0 ctx)
  t10 : 0 ≤ (128)
  h17_3 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_parallel_generatedRoot_165_17 backend hashModel tape (path ++ [17, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, ((parallel_generatedRoot_165.w_4_0 ctx), ((parallel_generatedRoot_165.w_4_1 ctx), ((parallel_generatedRoot_165.w_4_2 ctx), ((parallel_generatedRoot_165.w_4_3 ctx), ())))))))) next) (Int.toNat (128)) ((parallel_generatedRoot_165.w_7_0 ctx), ((parallel_generatedRoot_165.w_10_0 ctx), ((parallel_generatedRoot_165.w_13_0 ctx), ((parallel_generatedRoot_165.w_16_0 ctx), ())))) ((parallel_generatedRoot_165.w_17_0 ctx), ((parallel_generatedRoot_165.w_17_1 ctx), ((parallel_generatedRoot_165.w_17_2 ctx), ((parallel_generatedRoot_165.w_17_3 ctx), ()))))
  hout : outputs = parallel_generatedRoot_165.outs ctx

theorem parallel_generatedRoot_165.facts_0  {ctx : parallel_generatedRoot_165Ctx} {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_165.constraints_0 backend hashModel tape path params i_0 (parallel_generatedRoot_165.w_0_0 ctx) (parallel_generatedRoot_165.w_1_0 ctx) (parallel_generatedRoot_165.w_2_0 ctx) (parallel_generatedRoot_165.w_3_0 ctx) (parallel_generatedRoot_165.w_4_0 ctx) (parallel_generatedRoot_165.w_4_1 ctx) (parallel_generatedRoot_165.w_4_2 ctx) (parallel_generatedRoot_165.w_4_3 ctx) (parallel_generatedRoot_165.w_5_0 ctx) (parallel_generatedRoot_165.w_6_0 ctx) (parallel_generatedRoot_165.w_7_0 ctx) (parallel_generatedRoot_165.w_8_0 ctx) (parallel_generatedRoot_165.w_9_0 ctx) (parallel_generatedRoot_165.w_10_0 ctx) (parallel_generatedRoot_165.w_11_0 ctx) (parallel_generatedRoot_165.w_12_0 ctx) (parallel_generatedRoot_165.w_13_0 ctx) (parallel_generatedRoot_165.w_14_0 ctx) (parallel_generatedRoot_165.w_15_0 ctx) (parallel_generatedRoot_165.w_16_0 ctx) (parallel_generatedRoot_165.w_17_0 ctx) (parallel_generatedRoot_165.w_17_1 ctx) (parallel_generatedRoot_165.w_17_2 ctx) (parallel_generatedRoot_165.w_17_3 ctx) outputs) :
    parallel_generatedRoot_165.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_165.Facts  (ctx : parallel_generatedRoot_165Ctx) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  c0 : parallel_generatedRoot_165.Facts_0 ctx outputs

theorem parallel_generatedRoot_165.facts  (ctx : parallel_generatedRoot_165Ctx) {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_165.body backend hashModel tape path params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_165.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_165.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_169.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) where
  inputs : (Fin 2607022080 → mxxMat1) × (Fin 2607022080 → mxxMat1) × (Fin 2607022080 → mxxMat1) × (Fin 2607022080 → mxxMat1) × Unit
  witness : parallel_generatedRoot_169.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_169Ctx" => (parallel_generatedRoot_169.Ctx backend hashModel tape path params i_0)

def parallel_generatedRoot_169.w_0_0  (ctx : parallel_generatedRoot_169Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.1

def parallel_generatedRoot_169.w_1_0  (ctx : parallel_generatedRoot_169Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.2.1

def parallel_generatedRoot_169.w_2_0  (ctx : parallel_generatedRoot_169Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_169.w_3_0  (ctx : parallel_generatedRoot_169Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_169.w_4_0  (ctx : parallel_generatedRoot_169Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_0

def parallel_generatedRoot_169.w_4_1  (ctx : parallel_generatedRoot_169Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_1

def parallel_generatedRoot_169.w_4_2  (ctx : parallel_generatedRoot_169Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_2

def parallel_generatedRoot_169.w_4_3  (ctx : parallel_generatedRoot_169Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_3

def parallel_generatedRoot_169.w_5_0  (ctx : parallel_generatedRoot_169Ctx) : mxxMat1 :=
  ctx.witness.w_5_0

def parallel_generatedRoot_169.w_6_0  (ctx : parallel_generatedRoot_169Ctx) : mxxMat1 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_169.w_7_0  (ctx : parallel_generatedRoot_169Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_169.w_5_0 ctx) (parallel_generatedRoot_169.w_6_0 ctx)

def parallel_generatedRoot_169.w_8_0  (ctx : parallel_generatedRoot_169Ctx) : mxxMat1 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_169.w_9_0  (ctx : parallel_generatedRoot_169Ctx) : mxxMat1 :=
  ctx.witness.w_9_0

def parallel_generatedRoot_169.w_10_0  (ctx : parallel_generatedRoot_169Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_169.w_8_0 ctx) (parallel_generatedRoot_169.w_9_0 ctx)

def parallel_generatedRoot_169.w_11_0  (ctx : parallel_generatedRoot_169Ctx) : mxxMat1 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_169.w_12_0  (ctx : parallel_generatedRoot_169Ctx) : mxxMat1 :=
  ctx.witness.w_12_0

def parallel_generatedRoot_169.w_13_0  (ctx : parallel_generatedRoot_169Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_169.w_11_0 ctx) (parallel_generatedRoot_169.w_12_0 ctx)

def parallel_generatedRoot_169.w_14_0  (ctx : parallel_generatedRoot_169Ctx) : mxxMat1 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_169.w_15_0  (ctx : parallel_generatedRoot_169Ctx) : mxxMat1 :=
  ctx.witness.w_15_0

def parallel_generatedRoot_169.w_16_0  (ctx : parallel_generatedRoot_169Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_169.w_14_0 ctx) (parallel_generatedRoot_169.w_15_0 ctx)

def parallel_generatedRoot_169.w_17_0  (ctx : parallel_generatedRoot_169Ctx) : mxxMat1 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_169.w_17_1  (ctx : parallel_generatedRoot_169Ctx) : mxxMat1 :=
  ctx.witness.w_17_1

def parallel_generatedRoot_169.w_17_2  (ctx : parallel_generatedRoot_169Ctx) : mxxMat1 :=
  ctx.witness.w_17_2

def parallel_generatedRoot_169.w_17_3  (ctx : parallel_generatedRoot_169Ctx) : mxxMat1 :=
  ctx.witness.w_17_3

def parallel_generatedRoot_169.outs  (ctx : parallel_generatedRoot_169Ctx) : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit :=
  ((parallel_generatedRoot_169.w_17_0 ctx), ((parallel_generatedRoot_169.w_17_1 ctx), ((parallel_generatedRoot_169.w_17_2 ctx), ((parallel_generatedRoot_169.w_17_3 ctx), ()))))

structure parallel_generatedRoot_169.Facts_0  (ctx : parallel_generatedRoot_169Ctx) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  t0 : (130) = 130
  h4_3 : (∀ i : Fin 130, parallel_parallel_generatedRoot_169_4 backend hashModel tape (path ++ [4, i.val]) params i_0 i ((parallel_generatedRoot_169.w_0_0 ctx), ((parallel_generatedRoot_169.w_1_0 ctx), ((parallel_generatedRoot_169.w_2_0 ctx), ((parallel_generatedRoot_169.w_3_0 ctx), ())))) (((parallel_generatedRoot_169.w_4_0 ctx) i), (((parallel_generatedRoot_169.w_4_1 ctx) i), (((parallel_generatedRoot_169.w_4_2 ctx) i), (((parallel_generatedRoot_169.w_4_3 ctx) i), ())))))
  h5_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_169.w_4_0 ctx) (0) (parallel_generatedRoot_169.w_5_0 ctx)
  h6_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_169.w_4_0 ctx) (1) (parallel_generatedRoot_169.w_6_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_169.w_4_1 ctx) (0) (parallel_generatedRoot_169.w_8_0 ctx)
  h9_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_169.w_4_1 ctx) (1) (parallel_generatedRoot_169.w_9_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_169.w_4_2 ctx) (0) (parallel_generatedRoot_169.w_11_0 ctx)
  h12_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_169.w_4_2 ctx) (1) (parallel_generatedRoot_169.w_12_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_169.w_4_3 ctx) (0) (parallel_generatedRoot_169.w_14_0 ctx)
  h15_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_169.w_4_3 ctx) (1) (parallel_generatedRoot_169.w_15_0 ctx)
  t10 : 0 ≤ (128)
  h17_3 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_parallel_generatedRoot_169_17 backend hashModel tape (path ++ [17, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, ((parallel_generatedRoot_169.w_4_0 ctx), ((parallel_generatedRoot_169.w_4_1 ctx), ((parallel_generatedRoot_169.w_4_2 ctx), ((parallel_generatedRoot_169.w_4_3 ctx), ())))))))) next) (Int.toNat (128)) ((parallel_generatedRoot_169.w_7_0 ctx), ((parallel_generatedRoot_169.w_10_0 ctx), ((parallel_generatedRoot_169.w_13_0 ctx), ((parallel_generatedRoot_169.w_16_0 ctx), ())))) ((parallel_generatedRoot_169.w_17_0 ctx), ((parallel_generatedRoot_169.w_17_1 ctx), ((parallel_generatedRoot_169.w_17_2 ctx), ((parallel_generatedRoot_169.w_17_3 ctx), ()))))
  hout : outputs = parallel_generatedRoot_169.outs ctx

theorem parallel_generatedRoot_169.facts_0  {ctx : parallel_generatedRoot_169Ctx} {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_169.constraints_0 backend hashModel tape path params i_0 (parallel_generatedRoot_169.w_0_0 ctx) (parallel_generatedRoot_169.w_1_0 ctx) (parallel_generatedRoot_169.w_2_0 ctx) (parallel_generatedRoot_169.w_3_0 ctx) (parallel_generatedRoot_169.w_4_0 ctx) (parallel_generatedRoot_169.w_4_1 ctx) (parallel_generatedRoot_169.w_4_2 ctx) (parallel_generatedRoot_169.w_4_3 ctx) (parallel_generatedRoot_169.w_5_0 ctx) (parallel_generatedRoot_169.w_6_0 ctx) (parallel_generatedRoot_169.w_7_0 ctx) (parallel_generatedRoot_169.w_8_0 ctx) (parallel_generatedRoot_169.w_9_0 ctx) (parallel_generatedRoot_169.w_10_0 ctx) (parallel_generatedRoot_169.w_11_0 ctx) (parallel_generatedRoot_169.w_12_0 ctx) (parallel_generatedRoot_169.w_13_0 ctx) (parallel_generatedRoot_169.w_14_0 ctx) (parallel_generatedRoot_169.w_15_0 ctx) (parallel_generatedRoot_169.w_16_0 ctx) (parallel_generatedRoot_169.w_17_0 ctx) (parallel_generatedRoot_169.w_17_1 ctx) (parallel_generatedRoot_169.w_17_2 ctx) (parallel_generatedRoot_169.w_17_3 ctx) outputs) :
    parallel_generatedRoot_169.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_169.Facts  (ctx : parallel_generatedRoot_169Ctx) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  c0 : parallel_generatedRoot_169.Facts_0 ctx outputs

theorem parallel_generatedRoot_169.facts  (ctx : parallel_generatedRoot_169Ctx) {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_169.body backend hashModel tape path params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_169.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_169.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_173.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) where
  inputs : (Fin 2607022080 → mxxMat1) × (Fin 2607022080 → mxxMat1) × (Fin 2607022080 → mxxMat1) × (Fin 2607022080 → mxxMat1) × Unit
  witness : parallel_generatedRoot_173.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_173Ctx" => (parallel_generatedRoot_173.Ctx backend hashModel tape path params i_0)

def parallel_generatedRoot_173.w_0_0  (ctx : parallel_generatedRoot_173Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.1

def parallel_generatedRoot_173.w_1_0  (ctx : parallel_generatedRoot_173Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.2.1

def parallel_generatedRoot_173.w_2_0  (ctx : parallel_generatedRoot_173Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_173.w_3_0  (ctx : parallel_generatedRoot_173Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_173.w_4_0  (ctx : parallel_generatedRoot_173Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_0

def parallel_generatedRoot_173.w_4_1  (ctx : parallel_generatedRoot_173Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_1

def parallel_generatedRoot_173.w_4_2  (ctx : parallel_generatedRoot_173Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_2

def parallel_generatedRoot_173.w_4_3  (ctx : parallel_generatedRoot_173Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_3

def parallel_generatedRoot_173.w_5_0  (ctx : parallel_generatedRoot_173Ctx) : mxxMat1 :=
  ctx.witness.w_5_0

def parallel_generatedRoot_173.w_6_0  (ctx : parallel_generatedRoot_173Ctx) : mxxMat1 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_173.w_7_0  (ctx : parallel_generatedRoot_173Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_173.w_5_0 ctx) (parallel_generatedRoot_173.w_6_0 ctx)

def parallel_generatedRoot_173.w_8_0  (ctx : parallel_generatedRoot_173Ctx) : mxxMat1 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_173.w_9_0  (ctx : parallel_generatedRoot_173Ctx) : mxxMat1 :=
  ctx.witness.w_9_0

def parallel_generatedRoot_173.w_10_0  (ctx : parallel_generatedRoot_173Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_173.w_8_0 ctx) (parallel_generatedRoot_173.w_9_0 ctx)

def parallel_generatedRoot_173.w_11_0  (ctx : parallel_generatedRoot_173Ctx) : mxxMat1 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_173.w_12_0  (ctx : parallel_generatedRoot_173Ctx) : mxxMat1 :=
  ctx.witness.w_12_0

def parallel_generatedRoot_173.w_13_0  (ctx : parallel_generatedRoot_173Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_173.w_11_0 ctx) (parallel_generatedRoot_173.w_12_0 ctx)

def parallel_generatedRoot_173.w_14_0  (ctx : parallel_generatedRoot_173Ctx) : mxxMat1 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_173.w_15_0  (ctx : parallel_generatedRoot_173Ctx) : mxxMat1 :=
  ctx.witness.w_15_0

def parallel_generatedRoot_173.w_16_0  (ctx : parallel_generatedRoot_173Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_173.w_14_0 ctx) (parallel_generatedRoot_173.w_15_0 ctx)

def parallel_generatedRoot_173.w_17_0  (ctx : parallel_generatedRoot_173Ctx) : mxxMat1 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_173.w_17_1  (ctx : parallel_generatedRoot_173Ctx) : mxxMat1 :=
  ctx.witness.w_17_1

def parallel_generatedRoot_173.w_17_2  (ctx : parallel_generatedRoot_173Ctx) : mxxMat1 :=
  ctx.witness.w_17_2

def parallel_generatedRoot_173.w_17_3  (ctx : parallel_generatedRoot_173Ctx) : mxxMat1 :=
  ctx.witness.w_17_3

def parallel_generatedRoot_173.outs  (ctx : parallel_generatedRoot_173Ctx) : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit :=
  ((parallel_generatedRoot_173.w_17_0 ctx), ((parallel_generatedRoot_173.w_17_1 ctx), ((parallel_generatedRoot_173.w_17_2 ctx), ((parallel_generatedRoot_173.w_17_3 ctx), ()))))

structure parallel_generatedRoot_173.Facts_0  (ctx : parallel_generatedRoot_173Ctx) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  t0 : (130) = 130
  h4_3 : (∀ i : Fin 130, parallel_parallel_generatedRoot_173_4 backend hashModel tape (path ++ [4, i.val]) params i_0 i ((parallel_generatedRoot_173.w_0_0 ctx), ((parallel_generatedRoot_173.w_1_0 ctx), ((parallel_generatedRoot_173.w_2_0 ctx), ((parallel_generatedRoot_173.w_3_0 ctx), ())))) (((parallel_generatedRoot_173.w_4_0 ctx) i), (((parallel_generatedRoot_173.w_4_1 ctx) i), (((parallel_generatedRoot_173.w_4_2 ctx) i), (((parallel_generatedRoot_173.w_4_3 ctx) i), ())))))
  h5_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_173.w_4_0 ctx) (0) (parallel_generatedRoot_173.w_5_0 ctx)
  h6_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_173.w_4_0 ctx) (1) (parallel_generatedRoot_173.w_6_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_173.w_4_1 ctx) (0) (parallel_generatedRoot_173.w_8_0 ctx)
  h9_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_173.w_4_1 ctx) (1) (parallel_generatedRoot_173.w_9_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_173.w_4_2 ctx) (0) (parallel_generatedRoot_173.w_11_0 ctx)
  h12_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_173.w_4_2 ctx) (1) (parallel_generatedRoot_173.w_12_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_173.w_4_3 ctx) (0) (parallel_generatedRoot_173.w_14_0 ctx)
  h15_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_173.w_4_3 ctx) (1) (parallel_generatedRoot_173.w_15_0 ctx)
  t10 : 0 ≤ (128)
  h17_3 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_parallel_generatedRoot_173_17 backend hashModel tape (path ++ [17, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, ((parallel_generatedRoot_173.w_4_0 ctx), ((parallel_generatedRoot_173.w_4_1 ctx), ((parallel_generatedRoot_173.w_4_2 ctx), ((parallel_generatedRoot_173.w_4_3 ctx), ())))))))) next) (Int.toNat (128)) ((parallel_generatedRoot_173.w_7_0 ctx), ((parallel_generatedRoot_173.w_10_0 ctx), ((parallel_generatedRoot_173.w_13_0 ctx), ((parallel_generatedRoot_173.w_16_0 ctx), ())))) ((parallel_generatedRoot_173.w_17_0 ctx), ((parallel_generatedRoot_173.w_17_1 ctx), ((parallel_generatedRoot_173.w_17_2 ctx), ((parallel_generatedRoot_173.w_17_3 ctx), ()))))
  hout : outputs = parallel_generatedRoot_173.outs ctx

theorem parallel_generatedRoot_173.facts_0  {ctx : parallel_generatedRoot_173Ctx} {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_173.constraints_0 backend hashModel tape path params i_0 (parallel_generatedRoot_173.w_0_0 ctx) (parallel_generatedRoot_173.w_1_0 ctx) (parallel_generatedRoot_173.w_2_0 ctx) (parallel_generatedRoot_173.w_3_0 ctx) (parallel_generatedRoot_173.w_4_0 ctx) (parallel_generatedRoot_173.w_4_1 ctx) (parallel_generatedRoot_173.w_4_2 ctx) (parallel_generatedRoot_173.w_4_3 ctx) (parallel_generatedRoot_173.w_5_0 ctx) (parallel_generatedRoot_173.w_6_0 ctx) (parallel_generatedRoot_173.w_7_0 ctx) (parallel_generatedRoot_173.w_8_0 ctx) (parallel_generatedRoot_173.w_9_0 ctx) (parallel_generatedRoot_173.w_10_0 ctx) (parallel_generatedRoot_173.w_11_0 ctx) (parallel_generatedRoot_173.w_12_0 ctx) (parallel_generatedRoot_173.w_13_0 ctx) (parallel_generatedRoot_173.w_14_0 ctx) (parallel_generatedRoot_173.w_15_0 ctx) (parallel_generatedRoot_173.w_16_0 ctx) (parallel_generatedRoot_173.w_17_0 ctx) (parallel_generatedRoot_173.w_17_1 ctx) (parallel_generatedRoot_173.w_17_2 ctx) (parallel_generatedRoot_173.w_17_3 ctx) outputs) :
    parallel_generatedRoot_173.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_173.Facts  (ctx : parallel_generatedRoot_173Ctx) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  c0 : parallel_generatedRoot_173.Facts_0 ctx outputs

theorem parallel_generatedRoot_173.facts  (ctx : parallel_generatedRoot_173Ctx) {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_173.body backend hashModel tape path params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_173.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_173.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_177.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) where
  inputs : (Fin 2607022080 → mxxMat1) × (Fin 2607022080 → mxxMat1) × (Fin 2607022080 → mxxMat1) × (Fin 2607022080 → mxxMat1) × Unit
  witness : parallel_generatedRoot_177.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_177Ctx" => (parallel_generatedRoot_177.Ctx backend hashModel tape path params i_0)

def parallel_generatedRoot_177.w_0_0  (ctx : parallel_generatedRoot_177Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.1

def parallel_generatedRoot_177.w_1_0  (ctx : parallel_generatedRoot_177Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.2.1

def parallel_generatedRoot_177.w_2_0  (ctx : parallel_generatedRoot_177Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_177.w_3_0  (ctx : parallel_generatedRoot_177Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_177.w_4_0  (ctx : parallel_generatedRoot_177Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_0

def parallel_generatedRoot_177.w_4_1  (ctx : parallel_generatedRoot_177Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_1

def parallel_generatedRoot_177.w_4_2  (ctx : parallel_generatedRoot_177Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_2

def parallel_generatedRoot_177.w_4_3  (ctx : parallel_generatedRoot_177Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_3

def parallel_generatedRoot_177.w_5_0  (ctx : parallel_generatedRoot_177Ctx) : mxxMat1 :=
  ctx.witness.w_5_0

def parallel_generatedRoot_177.w_6_0  (ctx : parallel_generatedRoot_177Ctx) : mxxMat1 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_177.w_7_0  (ctx : parallel_generatedRoot_177Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_177.w_5_0 ctx) (parallel_generatedRoot_177.w_6_0 ctx)

def parallel_generatedRoot_177.w_8_0  (ctx : parallel_generatedRoot_177Ctx) : mxxMat1 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_177.w_9_0  (ctx : parallel_generatedRoot_177Ctx) : mxxMat1 :=
  ctx.witness.w_9_0

def parallel_generatedRoot_177.w_10_0  (ctx : parallel_generatedRoot_177Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_177.w_8_0 ctx) (parallel_generatedRoot_177.w_9_0 ctx)

def parallel_generatedRoot_177.w_11_0  (ctx : parallel_generatedRoot_177Ctx) : mxxMat1 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_177.w_12_0  (ctx : parallel_generatedRoot_177Ctx) : mxxMat1 :=
  ctx.witness.w_12_0

def parallel_generatedRoot_177.w_13_0  (ctx : parallel_generatedRoot_177Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_177.w_11_0 ctx) (parallel_generatedRoot_177.w_12_0 ctx)

def parallel_generatedRoot_177.w_14_0  (ctx : parallel_generatedRoot_177Ctx) : mxxMat1 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_177.w_15_0  (ctx : parallel_generatedRoot_177Ctx) : mxxMat1 :=
  ctx.witness.w_15_0

def parallel_generatedRoot_177.w_16_0  (ctx : parallel_generatedRoot_177Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_177.w_14_0 ctx) (parallel_generatedRoot_177.w_15_0 ctx)

def parallel_generatedRoot_177.w_17_0  (ctx : parallel_generatedRoot_177Ctx) : mxxMat1 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_177.w_17_1  (ctx : parallel_generatedRoot_177Ctx) : mxxMat1 :=
  ctx.witness.w_17_1

def parallel_generatedRoot_177.w_17_2  (ctx : parallel_generatedRoot_177Ctx) : mxxMat1 :=
  ctx.witness.w_17_2

def parallel_generatedRoot_177.w_17_3  (ctx : parallel_generatedRoot_177Ctx) : mxxMat1 :=
  ctx.witness.w_17_3

def parallel_generatedRoot_177.outs  (ctx : parallel_generatedRoot_177Ctx) : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit :=
  ((parallel_generatedRoot_177.w_17_0 ctx), ((parallel_generatedRoot_177.w_17_1 ctx), ((parallel_generatedRoot_177.w_17_2 ctx), ((parallel_generatedRoot_177.w_17_3 ctx), ()))))

structure parallel_generatedRoot_177.Facts_0  (ctx : parallel_generatedRoot_177Ctx) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  t0 : (130) = 130
  h4_3 : (∀ i : Fin 130, parallel_parallel_generatedRoot_177_4 backend hashModel tape (path ++ [4, i.val]) params i_0 i ((parallel_generatedRoot_177.w_0_0 ctx), ((parallel_generatedRoot_177.w_1_0 ctx), ((parallel_generatedRoot_177.w_2_0 ctx), ((parallel_generatedRoot_177.w_3_0 ctx), ())))) (((parallel_generatedRoot_177.w_4_0 ctx) i), (((parallel_generatedRoot_177.w_4_1 ctx) i), (((parallel_generatedRoot_177.w_4_2 ctx) i), (((parallel_generatedRoot_177.w_4_3 ctx) i), ())))))
  h5_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_177.w_4_0 ctx) (0) (parallel_generatedRoot_177.w_5_0 ctx)
  h6_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_177.w_4_0 ctx) (1) (parallel_generatedRoot_177.w_6_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_177.w_4_1 ctx) (0) (parallel_generatedRoot_177.w_8_0 ctx)
  h9_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_177.w_4_1 ctx) (1) (parallel_generatedRoot_177.w_9_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_177.w_4_2 ctx) (0) (parallel_generatedRoot_177.w_11_0 ctx)
  h12_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_177.w_4_2 ctx) (1) (parallel_generatedRoot_177.w_12_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_177.w_4_3 ctx) (0) (parallel_generatedRoot_177.w_14_0 ctx)
  h15_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_177.w_4_3 ctx) (1) (parallel_generatedRoot_177.w_15_0 ctx)
  t10 : 0 ≤ (128)
  h17_3 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_parallel_generatedRoot_177_17 backend hashModel tape (path ++ [17, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, ((parallel_generatedRoot_177.w_4_0 ctx), ((parallel_generatedRoot_177.w_4_1 ctx), ((parallel_generatedRoot_177.w_4_2 ctx), ((parallel_generatedRoot_177.w_4_3 ctx), ())))))))) next) (Int.toNat (128)) ((parallel_generatedRoot_177.w_7_0 ctx), ((parallel_generatedRoot_177.w_10_0 ctx), ((parallel_generatedRoot_177.w_13_0 ctx), ((parallel_generatedRoot_177.w_16_0 ctx), ())))) ((parallel_generatedRoot_177.w_17_0 ctx), ((parallel_generatedRoot_177.w_17_1 ctx), ((parallel_generatedRoot_177.w_17_2 ctx), ((parallel_generatedRoot_177.w_17_3 ctx), ()))))
  hout : outputs = parallel_generatedRoot_177.outs ctx

theorem parallel_generatedRoot_177.facts_0  {ctx : parallel_generatedRoot_177Ctx} {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_177.constraints_0 backend hashModel tape path params i_0 (parallel_generatedRoot_177.w_0_0 ctx) (parallel_generatedRoot_177.w_1_0 ctx) (parallel_generatedRoot_177.w_2_0 ctx) (parallel_generatedRoot_177.w_3_0 ctx) (parallel_generatedRoot_177.w_4_0 ctx) (parallel_generatedRoot_177.w_4_1 ctx) (parallel_generatedRoot_177.w_4_2 ctx) (parallel_generatedRoot_177.w_4_3 ctx) (parallel_generatedRoot_177.w_5_0 ctx) (parallel_generatedRoot_177.w_6_0 ctx) (parallel_generatedRoot_177.w_7_0 ctx) (parallel_generatedRoot_177.w_8_0 ctx) (parallel_generatedRoot_177.w_9_0 ctx) (parallel_generatedRoot_177.w_10_0 ctx) (parallel_generatedRoot_177.w_11_0 ctx) (parallel_generatedRoot_177.w_12_0 ctx) (parallel_generatedRoot_177.w_13_0 ctx) (parallel_generatedRoot_177.w_14_0 ctx) (parallel_generatedRoot_177.w_15_0 ctx) (parallel_generatedRoot_177.w_16_0 ctx) (parallel_generatedRoot_177.w_17_0 ctx) (parallel_generatedRoot_177.w_17_1 ctx) (parallel_generatedRoot_177.w_17_2 ctx) (parallel_generatedRoot_177.w_17_3 ctx) outputs) :
    parallel_generatedRoot_177.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_177.Facts  (ctx : parallel_generatedRoot_177Ctx) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  c0 : parallel_generatedRoot_177.Facts_0 ctx outputs

theorem parallel_generatedRoot_177.facts  (ctx : parallel_generatedRoot_177Ctx) {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_177.body backend hashModel tape path params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_177.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_177.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_181.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) where
  inputs : (Fin 2607022080 → mxxMat1) × (Fin 2607022080 → mxxMat1) × (Fin 2607022080 → mxxMat1) × (Fin 2607022080 → mxxMat1) × Unit
  witness : parallel_generatedRoot_181.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_181Ctx" => (parallel_generatedRoot_181.Ctx backend hashModel tape path params i_0)

def parallel_generatedRoot_181.w_0_0  (ctx : parallel_generatedRoot_181Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.1

def parallel_generatedRoot_181.w_1_0  (ctx : parallel_generatedRoot_181Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.2.1

def parallel_generatedRoot_181.w_2_0  (ctx : parallel_generatedRoot_181Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_181.w_3_0  (ctx : parallel_generatedRoot_181Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_181.w_4_0  (ctx : parallel_generatedRoot_181Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_0

def parallel_generatedRoot_181.w_4_1  (ctx : parallel_generatedRoot_181Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_1

def parallel_generatedRoot_181.w_4_2  (ctx : parallel_generatedRoot_181Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_2

def parallel_generatedRoot_181.w_4_3  (ctx : parallel_generatedRoot_181Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_3

def parallel_generatedRoot_181.w_5_0  (ctx : parallel_generatedRoot_181Ctx) : mxxMat1 :=
  ctx.witness.w_5_0

def parallel_generatedRoot_181.w_6_0  (ctx : parallel_generatedRoot_181Ctx) : mxxMat1 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_181.w_7_0  (ctx : parallel_generatedRoot_181Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_181.w_5_0 ctx) (parallel_generatedRoot_181.w_6_0 ctx)

def parallel_generatedRoot_181.w_8_0  (ctx : parallel_generatedRoot_181Ctx) : mxxMat1 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_181.w_9_0  (ctx : parallel_generatedRoot_181Ctx) : mxxMat1 :=
  ctx.witness.w_9_0

def parallel_generatedRoot_181.w_10_0  (ctx : parallel_generatedRoot_181Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_181.w_8_0 ctx) (parallel_generatedRoot_181.w_9_0 ctx)

def parallel_generatedRoot_181.w_11_0  (ctx : parallel_generatedRoot_181Ctx) : mxxMat1 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_181.w_12_0  (ctx : parallel_generatedRoot_181Ctx) : mxxMat1 :=
  ctx.witness.w_12_0

def parallel_generatedRoot_181.w_13_0  (ctx : parallel_generatedRoot_181Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_181.w_11_0 ctx) (parallel_generatedRoot_181.w_12_0 ctx)

def parallel_generatedRoot_181.w_14_0  (ctx : parallel_generatedRoot_181Ctx) : mxxMat1 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_181.w_15_0  (ctx : parallel_generatedRoot_181Ctx) : mxxMat1 :=
  ctx.witness.w_15_0

def parallel_generatedRoot_181.w_16_0  (ctx : parallel_generatedRoot_181Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_181.w_14_0 ctx) (parallel_generatedRoot_181.w_15_0 ctx)

def parallel_generatedRoot_181.w_17_0  (ctx : parallel_generatedRoot_181Ctx) : mxxMat1 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_181.w_17_1  (ctx : parallel_generatedRoot_181Ctx) : mxxMat1 :=
  ctx.witness.w_17_1

def parallel_generatedRoot_181.w_17_2  (ctx : parallel_generatedRoot_181Ctx) : mxxMat1 :=
  ctx.witness.w_17_2

def parallel_generatedRoot_181.w_17_3  (ctx : parallel_generatedRoot_181Ctx) : mxxMat1 :=
  ctx.witness.w_17_3

def parallel_generatedRoot_181.outs  (ctx : parallel_generatedRoot_181Ctx) : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit :=
  ((parallel_generatedRoot_181.w_17_0 ctx), ((parallel_generatedRoot_181.w_17_1 ctx), ((parallel_generatedRoot_181.w_17_2 ctx), ((parallel_generatedRoot_181.w_17_3 ctx), ()))))

structure parallel_generatedRoot_181.Facts_0  (ctx : parallel_generatedRoot_181Ctx) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  t0 : (130) = 130
  h4_3 : (∀ i : Fin 130, parallel_parallel_generatedRoot_181_4 backend hashModel tape (path ++ [4, i.val]) params i_0 i ((parallel_generatedRoot_181.w_0_0 ctx), ((parallel_generatedRoot_181.w_1_0 ctx), ((parallel_generatedRoot_181.w_2_0 ctx), ((parallel_generatedRoot_181.w_3_0 ctx), ())))) (((parallel_generatedRoot_181.w_4_0 ctx) i), (((parallel_generatedRoot_181.w_4_1 ctx) i), (((parallel_generatedRoot_181.w_4_2 ctx) i), (((parallel_generatedRoot_181.w_4_3 ctx) i), ())))))
  h5_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_181.w_4_0 ctx) (0) (parallel_generatedRoot_181.w_5_0 ctx)
  h6_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_181.w_4_0 ctx) (1) (parallel_generatedRoot_181.w_6_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_181.w_4_1 ctx) (0) (parallel_generatedRoot_181.w_8_0 ctx)
  h9_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_181.w_4_1 ctx) (1) (parallel_generatedRoot_181.w_9_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_181.w_4_2 ctx) (0) (parallel_generatedRoot_181.w_11_0 ctx)
  h12_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_181.w_4_2 ctx) (1) (parallel_generatedRoot_181.w_12_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_181.w_4_3 ctx) (0) (parallel_generatedRoot_181.w_14_0 ctx)
  h15_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_181.w_4_3 ctx) (1) (parallel_generatedRoot_181.w_15_0 ctx)
  t10 : 0 ≤ (128)
  h17_3 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_parallel_generatedRoot_181_17 backend hashModel tape (path ++ [17, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, ((parallel_generatedRoot_181.w_4_0 ctx), ((parallel_generatedRoot_181.w_4_1 ctx), ((parallel_generatedRoot_181.w_4_2 ctx), ((parallel_generatedRoot_181.w_4_3 ctx), ())))))))) next) (Int.toNat (128)) ((parallel_generatedRoot_181.w_7_0 ctx), ((parallel_generatedRoot_181.w_10_0 ctx), ((parallel_generatedRoot_181.w_13_0 ctx), ((parallel_generatedRoot_181.w_16_0 ctx), ())))) ((parallel_generatedRoot_181.w_17_0 ctx), ((parallel_generatedRoot_181.w_17_1 ctx), ((parallel_generatedRoot_181.w_17_2 ctx), ((parallel_generatedRoot_181.w_17_3 ctx), ()))))
  hout : outputs = parallel_generatedRoot_181.outs ctx

theorem parallel_generatedRoot_181.facts_0  {ctx : parallel_generatedRoot_181Ctx} {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_181.constraints_0 backend hashModel tape path params i_0 (parallel_generatedRoot_181.w_0_0 ctx) (parallel_generatedRoot_181.w_1_0 ctx) (parallel_generatedRoot_181.w_2_0 ctx) (parallel_generatedRoot_181.w_3_0 ctx) (parallel_generatedRoot_181.w_4_0 ctx) (parallel_generatedRoot_181.w_4_1 ctx) (parallel_generatedRoot_181.w_4_2 ctx) (parallel_generatedRoot_181.w_4_3 ctx) (parallel_generatedRoot_181.w_5_0 ctx) (parallel_generatedRoot_181.w_6_0 ctx) (parallel_generatedRoot_181.w_7_0 ctx) (parallel_generatedRoot_181.w_8_0 ctx) (parallel_generatedRoot_181.w_9_0 ctx) (parallel_generatedRoot_181.w_10_0 ctx) (parallel_generatedRoot_181.w_11_0 ctx) (parallel_generatedRoot_181.w_12_0 ctx) (parallel_generatedRoot_181.w_13_0 ctx) (parallel_generatedRoot_181.w_14_0 ctx) (parallel_generatedRoot_181.w_15_0 ctx) (parallel_generatedRoot_181.w_16_0 ctx) (parallel_generatedRoot_181.w_17_0 ctx) (parallel_generatedRoot_181.w_17_1 ctx) (parallel_generatedRoot_181.w_17_2 ctx) (parallel_generatedRoot_181.w_17_3 ctx) outputs) :
    parallel_generatedRoot_181.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_181.Facts  (ctx : parallel_generatedRoot_181Ctx) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  c0 : parallel_generatedRoot_181.Facts_0 ctx outputs

theorem parallel_generatedRoot_181.facts  (ctx : parallel_generatedRoot_181Ctx) {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_181.body backend hashModel tape path params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_181.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_181.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_185.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) where
  inputs : (Fin 2607022080 → mxxMat1) × (Fin 2607022080 → mxxMat1) × (Fin 2607022080 → mxxMat1) × (Fin 2607022080 → mxxMat1) × Unit
  witness : parallel_generatedRoot_185.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_185Ctx" => (parallel_generatedRoot_185.Ctx backend hashModel tape path params i_0)

def parallel_generatedRoot_185.w_0_0  (ctx : parallel_generatedRoot_185Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.1

def parallel_generatedRoot_185.w_1_0  (ctx : parallel_generatedRoot_185Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.2.1

def parallel_generatedRoot_185.w_2_0  (ctx : parallel_generatedRoot_185Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_185.w_3_0  (ctx : parallel_generatedRoot_185Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_185.w_4_0  (ctx : parallel_generatedRoot_185Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_0

def parallel_generatedRoot_185.w_4_1  (ctx : parallel_generatedRoot_185Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_1

def parallel_generatedRoot_185.w_4_2  (ctx : parallel_generatedRoot_185Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_2

def parallel_generatedRoot_185.w_4_3  (ctx : parallel_generatedRoot_185Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_3

def parallel_generatedRoot_185.w_5_0  (ctx : parallel_generatedRoot_185Ctx) : mxxMat1 :=
  ctx.witness.w_5_0

def parallel_generatedRoot_185.w_6_0  (ctx : parallel_generatedRoot_185Ctx) : mxxMat1 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_185.w_7_0  (ctx : parallel_generatedRoot_185Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_185.w_5_0 ctx) (parallel_generatedRoot_185.w_6_0 ctx)

def parallel_generatedRoot_185.w_8_0  (ctx : parallel_generatedRoot_185Ctx) : mxxMat1 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_185.w_9_0  (ctx : parallel_generatedRoot_185Ctx) : mxxMat1 :=
  ctx.witness.w_9_0

def parallel_generatedRoot_185.w_10_0  (ctx : parallel_generatedRoot_185Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_185.w_8_0 ctx) (parallel_generatedRoot_185.w_9_0 ctx)

def parallel_generatedRoot_185.w_11_0  (ctx : parallel_generatedRoot_185Ctx) : mxxMat1 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_185.w_12_0  (ctx : parallel_generatedRoot_185Ctx) : mxxMat1 :=
  ctx.witness.w_12_0

def parallel_generatedRoot_185.w_13_0  (ctx : parallel_generatedRoot_185Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_185.w_11_0 ctx) (parallel_generatedRoot_185.w_12_0 ctx)

def parallel_generatedRoot_185.w_14_0  (ctx : parallel_generatedRoot_185Ctx) : mxxMat1 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_185.w_15_0  (ctx : parallel_generatedRoot_185Ctx) : mxxMat1 :=
  ctx.witness.w_15_0

def parallel_generatedRoot_185.w_16_0  (ctx : parallel_generatedRoot_185Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_185.w_14_0 ctx) (parallel_generatedRoot_185.w_15_0 ctx)

def parallel_generatedRoot_185.w_17_0  (ctx : parallel_generatedRoot_185Ctx) : mxxMat1 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_185.w_17_1  (ctx : parallel_generatedRoot_185Ctx) : mxxMat1 :=
  ctx.witness.w_17_1

def parallel_generatedRoot_185.w_17_2  (ctx : parallel_generatedRoot_185Ctx) : mxxMat1 :=
  ctx.witness.w_17_2

def parallel_generatedRoot_185.w_17_3  (ctx : parallel_generatedRoot_185Ctx) : mxxMat1 :=
  ctx.witness.w_17_3

def parallel_generatedRoot_185.outs  (ctx : parallel_generatedRoot_185Ctx) : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit :=
  ((parallel_generatedRoot_185.w_17_0 ctx), ((parallel_generatedRoot_185.w_17_1 ctx), ((parallel_generatedRoot_185.w_17_2 ctx), ((parallel_generatedRoot_185.w_17_3 ctx), ()))))

structure parallel_generatedRoot_185.Facts_0  (ctx : parallel_generatedRoot_185Ctx) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  t0 : (130) = 130
  h4_3 : (∀ i : Fin 130, parallel_parallel_generatedRoot_185_4 backend hashModel tape (path ++ [4, i.val]) params i_0 i ((parallel_generatedRoot_185.w_0_0 ctx), ((parallel_generatedRoot_185.w_1_0 ctx), ((parallel_generatedRoot_185.w_2_0 ctx), ((parallel_generatedRoot_185.w_3_0 ctx), ())))) (((parallel_generatedRoot_185.w_4_0 ctx) i), (((parallel_generatedRoot_185.w_4_1 ctx) i), (((parallel_generatedRoot_185.w_4_2 ctx) i), (((parallel_generatedRoot_185.w_4_3 ctx) i), ())))))
  h5_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_185.w_4_0 ctx) (0) (parallel_generatedRoot_185.w_5_0 ctx)
  h6_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_185.w_4_0 ctx) (1) (parallel_generatedRoot_185.w_6_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_185.w_4_1 ctx) (0) (parallel_generatedRoot_185.w_8_0 ctx)
  h9_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_185.w_4_1 ctx) (1) (parallel_generatedRoot_185.w_9_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_185.w_4_2 ctx) (0) (parallel_generatedRoot_185.w_11_0 ctx)
  h12_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_185.w_4_2 ctx) (1) (parallel_generatedRoot_185.w_12_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_185.w_4_3 ctx) (0) (parallel_generatedRoot_185.w_14_0 ctx)
  h15_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_185.w_4_3 ctx) (1) (parallel_generatedRoot_185.w_15_0 ctx)
  t10 : 0 ≤ (128)
  h17_3 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_parallel_generatedRoot_185_17 backend hashModel tape (path ++ [17, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, ((parallel_generatedRoot_185.w_4_0 ctx), ((parallel_generatedRoot_185.w_4_1 ctx), ((parallel_generatedRoot_185.w_4_2 ctx), ((parallel_generatedRoot_185.w_4_3 ctx), ())))))))) next) (Int.toNat (128)) ((parallel_generatedRoot_185.w_7_0 ctx), ((parallel_generatedRoot_185.w_10_0 ctx), ((parallel_generatedRoot_185.w_13_0 ctx), ((parallel_generatedRoot_185.w_16_0 ctx), ())))) ((parallel_generatedRoot_185.w_17_0 ctx), ((parallel_generatedRoot_185.w_17_1 ctx), ((parallel_generatedRoot_185.w_17_2 ctx), ((parallel_generatedRoot_185.w_17_3 ctx), ()))))
  hout : outputs = parallel_generatedRoot_185.outs ctx

theorem parallel_generatedRoot_185.facts_0  {ctx : parallel_generatedRoot_185Ctx} {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_185.constraints_0 backend hashModel tape path params i_0 (parallel_generatedRoot_185.w_0_0 ctx) (parallel_generatedRoot_185.w_1_0 ctx) (parallel_generatedRoot_185.w_2_0 ctx) (parallel_generatedRoot_185.w_3_0 ctx) (parallel_generatedRoot_185.w_4_0 ctx) (parallel_generatedRoot_185.w_4_1 ctx) (parallel_generatedRoot_185.w_4_2 ctx) (parallel_generatedRoot_185.w_4_3 ctx) (parallel_generatedRoot_185.w_5_0 ctx) (parallel_generatedRoot_185.w_6_0 ctx) (parallel_generatedRoot_185.w_7_0 ctx) (parallel_generatedRoot_185.w_8_0 ctx) (parallel_generatedRoot_185.w_9_0 ctx) (parallel_generatedRoot_185.w_10_0 ctx) (parallel_generatedRoot_185.w_11_0 ctx) (parallel_generatedRoot_185.w_12_0 ctx) (parallel_generatedRoot_185.w_13_0 ctx) (parallel_generatedRoot_185.w_14_0 ctx) (parallel_generatedRoot_185.w_15_0 ctx) (parallel_generatedRoot_185.w_16_0 ctx) (parallel_generatedRoot_185.w_17_0 ctx) (parallel_generatedRoot_185.w_17_1 ctx) (parallel_generatedRoot_185.w_17_2 ctx) (parallel_generatedRoot_185.w_17_3 ctx) outputs) :
    parallel_generatedRoot_185.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_185.Facts  (ctx : parallel_generatedRoot_185Ctx) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  c0 : parallel_generatedRoot_185.Facts_0 ctx outputs

theorem parallel_generatedRoot_185.facts  (ctx : parallel_generatedRoot_185Ctx) {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_185.body backend hashModel tape path params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_185.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_185.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_189.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) where
  inputs : (Fin 2607022080 → mxxMat1) × (Fin 2607022080 → mxxMat1) × (Fin 2607022080 → mxxMat1) × (Fin 2607022080 → mxxMat1) × Unit
  witness : parallel_generatedRoot_189.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_189Ctx" => (parallel_generatedRoot_189.Ctx backend hashModel tape path params i_0)

def parallel_generatedRoot_189.w_0_0  (ctx : parallel_generatedRoot_189Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.1

def parallel_generatedRoot_189.w_1_0  (ctx : parallel_generatedRoot_189Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.2.1

def parallel_generatedRoot_189.w_2_0  (ctx : parallel_generatedRoot_189Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_189.w_3_0  (ctx : parallel_generatedRoot_189Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_189.w_4_0  (ctx : parallel_generatedRoot_189Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_0

def parallel_generatedRoot_189.w_4_1  (ctx : parallel_generatedRoot_189Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_1

def parallel_generatedRoot_189.w_4_2  (ctx : parallel_generatedRoot_189Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_2

def parallel_generatedRoot_189.w_4_3  (ctx : parallel_generatedRoot_189Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_3

def parallel_generatedRoot_189.w_5_0  (ctx : parallel_generatedRoot_189Ctx) : mxxMat1 :=
  ctx.witness.w_5_0

def parallel_generatedRoot_189.w_6_0  (ctx : parallel_generatedRoot_189Ctx) : mxxMat1 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_189.w_7_0  (ctx : parallel_generatedRoot_189Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_189.w_5_0 ctx) (parallel_generatedRoot_189.w_6_0 ctx)

def parallel_generatedRoot_189.w_8_0  (ctx : parallel_generatedRoot_189Ctx) : mxxMat1 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_189.w_9_0  (ctx : parallel_generatedRoot_189Ctx) : mxxMat1 :=
  ctx.witness.w_9_0

def parallel_generatedRoot_189.w_10_0  (ctx : parallel_generatedRoot_189Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_189.w_8_0 ctx) (parallel_generatedRoot_189.w_9_0 ctx)

def parallel_generatedRoot_189.w_11_0  (ctx : parallel_generatedRoot_189Ctx) : mxxMat1 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_189.w_12_0  (ctx : parallel_generatedRoot_189Ctx) : mxxMat1 :=
  ctx.witness.w_12_0

def parallel_generatedRoot_189.w_13_0  (ctx : parallel_generatedRoot_189Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_189.w_11_0 ctx) (parallel_generatedRoot_189.w_12_0 ctx)

def parallel_generatedRoot_189.w_14_0  (ctx : parallel_generatedRoot_189Ctx) : mxxMat1 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_189.w_15_0  (ctx : parallel_generatedRoot_189Ctx) : mxxMat1 :=
  ctx.witness.w_15_0

def parallel_generatedRoot_189.w_16_0  (ctx : parallel_generatedRoot_189Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_189.w_14_0 ctx) (parallel_generatedRoot_189.w_15_0 ctx)

def parallel_generatedRoot_189.w_17_0  (ctx : parallel_generatedRoot_189Ctx) : mxxMat1 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_189.w_17_1  (ctx : parallel_generatedRoot_189Ctx) : mxxMat1 :=
  ctx.witness.w_17_1

def parallel_generatedRoot_189.w_17_2  (ctx : parallel_generatedRoot_189Ctx) : mxxMat1 :=
  ctx.witness.w_17_2

def parallel_generatedRoot_189.w_17_3  (ctx : parallel_generatedRoot_189Ctx) : mxxMat1 :=
  ctx.witness.w_17_3

def parallel_generatedRoot_189.outs  (ctx : parallel_generatedRoot_189Ctx) : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit :=
  ((parallel_generatedRoot_189.w_17_0 ctx), ((parallel_generatedRoot_189.w_17_1 ctx), ((parallel_generatedRoot_189.w_17_2 ctx), ((parallel_generatedRoot_189.w_17_3 ctx), ()))))

structure parallel_generatedRoot_189.Facts_0  (ctx : parallel_generatedRoot_189Ctx) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  t0 : (130) = 130
  h4_3 : (∀ i : Fin 130, parallel_parallel_generatedRoot_189_4 backend hashModel tape (path ++ [4, i.val]) params i_0 i ((parallel_generatedRoot_189.w_0_0 ctx), ((parallel_generatedRoot_189.w_1_0 ctx), ((parallel_generatedRoot_189.w_2_0 ctx), ((parallel_generatedRoot_189.w_3_0 ctx), ())))) (((parallel_generatedRoot_189.w_4_0 ctx) i), (((parallel_generatedRoot_189.w_4_1 ctx) i), (((parallel_generatedRoot_189.w_4_2 ctx) i), (((parallel_generatedRoot_189.w_4_3 ctx) i), ())))))
  h5_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_189.w_4_0 ctx) (0) (parallel_generatedRoot_189.w_5_0 ctx)
  h6_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_189.w_4_0 ctx) (1) (parallel_generatedRoot_189.w_6_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_189.w_4_1 ctx) (0) (parallel_generatedRoot_189.w_8_0 ctx)
  h9_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_189.w_4_1 ctx) (1) (parallel_generatedRoot_189.w_9_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_189.w_4_2 ctx) (0) (parallel_generatedRoot_189.w_11_0 ctx)
  h12_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_189.w_4_2 ctx) (1) (parallel_generatedRoot_189.w_12_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_189.w_4_3 ctx) (0) (parallel_generatedRoot_189.w_14_0 ctx)
  h15_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_189.w_4_3 ctx) (1) (parallel_generatedRoot_189.w_15_0 ctx)
  t10 : 0 ≤ (128)
  h17_3 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_parallel_generatedRoot_189_17 backend hashModel tape (path ++ [17, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, ((parallel_generatedRoot_189.w_4_0 ctx), ((parallel_generatedRoot_189.w_4_1 ctx), ((parallel_generatedRoot_189.w_4_2 ctx), ((parallel_generatedRoot_189.w_4_3 ctx), ())))))))) next) (Int.toNat (128)) ((parallel_generatedRoot_189.w_7_0 ctx), ((parallel_generatedRoot_189.w_10_0 ctx), ((parallel_generatedRoot_189.w_13_0 ctx), ((parallel_generatedRoot_189.w_16_0 ctx), ())))) ((parallel_generatedRoot_189.w_17_0 ctx), ((parallel_generatedRoot_189.w_17_1 ctx), ((parallel_generatedRoot_189.w_17_2 ctx), ((parallel_generatedRoot_189.w_17_3 ctx), ()))))
  hout : outputs = parallel_generatedRoot_189.outs ctx

theorem parallel_generatedRoot_189.facts_0  {ctx : parallel_generatedRoot_189Ctx} {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_189.constraints_0 backend hashModel tape path params i_0 (parallel_generatedRoot_189.w_0_0 ctx) (parallel_generatedRoot_189.w_1_0 ctx) (parallel_generatedRoot_189.w_2_0 ctx) (parallel_generatedRoot_189.w_3_0 ctx) (parallel_generatedRoot_189.w_4_0 ctx) (parallel_generatedRoot_189.w_4_1 ctx) (parallel_generatedRoot_189.w_4_2 ctx) (parallel_generatedRoot_189.w_4_3 ctx) (parallel_generatedRoot_189.w_5_0 ctx) (parallel_generatedRoot_189.w_6_0 ctx) (parallel_generatedRoot_189.w_7_0 ctx) (parallel_generatedRoot_189.w_8_0 ctx) (parallel_generatedRoot_189.w_9_0 ctx) (parallel_generatedRoot_189.w_10_0 ctx) (parallel_generatedRoot_189.w_11_0 ctx) (parallel_generatedRoot_189.w_12_0 ctx) (parallel_generatedRoot_189.w_13_0 ctx) (parallel_generatedRoot_189.w_14_0 ctx) (parallel_generatedRoot_189.w_15_0 ctx) (parallel_generatedRoot_189.w_16_0 ctx) (parallel_generatedRoot_189.w_17_0 ctx) (parallel_generatedRoot_189.w_17_1 ctx) (parallel_generatedRoot_189.w_17_2 ctx) (parallel_generatedRoot_189.w_17_3 ctx) outputs) :
    parallel_generatedRoot_189.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_189.Facts  (ctx : parallel_generatedRoot_189Ctx) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  c0 : parallel_generatedRoot_189.Facts_0 ctx outputs

theorem parallel_generatedRoot_189.facts  (ctx : parallel_generatedRoot_189Ctx) {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_189.body backend hashModel tape path params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_189.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_189.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_193.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) where
  inputs : (Fin 2607022080 → mxxMat1) × (Fin 2607022080 → mxxMat1) × (Fin 2607022080 → mxxMat1) × (Fin 2607022080 → mxxMat1) × Unit
  witness : parallel_generatedRoot_193.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_193Ctx" => (parallel_generatedRoot_193.Ctx backend hashModel tape path params i_0)

def parallel_generatedRoot_193.w_0_0  (ctx : parallel_generatedRoot_193Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.1

def parallel_generatedRoot_193.w_1_0  (ctx : parallel_generatedRoot_193Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.2.1

def parallel_generatedRoot_193.w_2_0  (ctx : parallel_generatedRoot_193Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_193.w_3_0  (ctx : parallel_generatedRoot_193Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_193.w_4_0  (ctx : parallel_generatedRoot_193Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_0

def parallel_generatedRoot_193.w_4_1  (ctx : parallel_generatedRoot_193Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_1

def parallel_generatedRoot_193.w_4_2  (ctx : parallel_generatedRoot_193Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_2

def parallel_generatedRoot_193.w_4_3  (ctx : parallel_generatedRoot_193Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_3

def parallel_generatedRoot_193.w_5_0  (ctx : parallel_generatedRoot_193Ctx) : mxxMat1 :=
  ctx.witness.w_5_0

def parallel_generatedRoot_193.w_6_0  (ctx : parallel_generatedRoot_193Ctx) : mxxMat1 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_193.w_7_0  (ctx : parallel_generatedRoot_193Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_193.w_5_0 ctx) (parallel_generatedRoot_193.w_6_0 ctx)

def parallel_generatedRoot_193.w_8_0  (ctx : parallel_generatedRoot_193Ctx) : mxxMat1 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_193.w_9_0  (ctx : parallel_generatedRoot_193Ctx) : mxxMat1 :=
  ctx.witness.w_9_0

def parallel_generatedRoot_193.w_10_0  (ctx : parallel_generatedRoot_193Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_193.w_8_0 ctx) (parallel_generatedRoot_193.w_9_0 ctx)

def parallel_generatedRoot_193.w_11_0  (ctx : parallel_generatedRoot_193Ctx) : mxxMat1 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_193.w_12_0  (ctx : parallel_generatedRoot_193Ctx) : mxxMat1 :=
  ctx.witness.w_12_0

def parallel_generatedRoot_193.w_13_0  (ctx : parallel_generatedRoot_193Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_193.w_11_0 ctx) (parallel_generatedRoot_193.w_12_0 ctx)

def parallel_generatedRoot_193.w_14_0  (ctx : parallel_generatedRoot_193Ctx) : mxxMat1 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_193.w_15_0  (ctx : parallel_generatedRoot_193Ctx) : mxxMat1 :=
  ctx.witness.w_15_0

def parallel_generatedRoot_193.w_16_0  (ctx : parallel_generatedRoot_193Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_193.w_14_0 ctx) (parallel_generatedRoot_193.w_15_0 ctx)

def parallel_generatedRoot_193.w_17_0  (ctx : parallel_generatedRoot_193Ctx) : mxxMat1 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_193.w_17_1  (ctx : parallel_generatedRoot_193Ctx) : mxxMat1 :=
  ctx.witness.w_17_1

def parallel_generatedRoot_193.w_17_2  (ctx : parallel_generatedRoot_193Ctx) : mxxMat1 :=
  ctx.witness.w_17_2

def parallel_generatedRoot_193.w_17_3  (ctx : parallel_generatedRoot_193Ctx) : mxxMat1 :=
  ctx.witness.w_17_3

def parallel_generatedRoot_193.outs  (ctx : parallel_generatedRoot_193Ctx) : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit :=
  ((parallel_generatedRoot_193.w_17_0 ctx), ((parallel_generatedRoot_193.w_17_1 ctx), ((parallel_generatedRoot_193.w_17_2 ctx), ((parallel_generatedRoot_193.w_17_3 ctx), ()))))

structure parallel_generatedRoot_193.Facts_0  (ctx : parallel_generatedRoot_193Ctx) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  t0 : (130) = 130
  h4_3 : (∀ i : Fin 130, parallel_parallel_generatedRoot_193_4 backend hashModel tape (path ++ [4, i.val]) params i_0 i ((parallel_generatedRoot_193.w_0_0 ctx), ((parallel_generatedRoot_193.w_1_0 ctx), ((parallel_generatedRoot_193.w_2_0 ctx), ((parallel_generatedRoot_193.w_3_0 ctx), ())))) (((parallel_generatedRoot_193.w_4_0 ctx) i), (((parallel_generatedRoot_193.w_4_1 ctx) i), (((parallel_generatedRoot_193.w_4_2 ctx) i), (((parallel_generatedRoot_193.w_4_3 ctx) i), ())))))
  h5_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_193.w_4_0 ctx) (0) (parallel_generatedRoot_193.w_5_0 ctx)
  h6_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_193.w_4_0 ctx) (1) (parallel_generatedRoot_193.w_6_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_193.w_4_1 ctx) (0) (parallel_generatedRoot_193.w_8_0 ctx)
  h9_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_193.w_4_1 ctx) (1) (parallel_generatedRoot_193.w_9_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_193.w_4_2 ctx) (0) (parallel_generatedRoot_193.w_11_0 ctx)
  h12_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_193.w_4_2 ctx) (1) (parallel_generatedRoot_193.w_12_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_193.w_4_3 ctx) (0) (parallel_generatedRoot_193.w_14_0 ctx)
  h15_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_193.w_4_3 ctx) (1) (parallel_generatedRoot_193.w_15_0 ctx)
  t10 : 0 ≤ (128)
  h17_3 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_parallel_generatedRoot_193_17 backend hashModel tape (path ++ [17, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, ((parallel_generatedRoot_193.w_4_0 ctx), ((parallel_generatedRoot_193.w_4_1 ctx), ((parallel_generatedRoot_193.w_4_2 ctx), ((parallel_generatedRoot_193.w_4_3 ctx), ())))))))) next) (Int.toNat (128)) ((parallel_generatedRoot_193.w_7_0 ctx), ((parallel_generatedRoot_193.w_10_0 ctx), ((parallel_generatedRoot_193.w_13_0 ctx), ((parallel_generatedRoot_193.w_16_0 ctx), ())))) ((parallel_generatedRoot_193.w_17_0 ctx), ((parallel_generatedRoot_193.w_17_1 ctx), ((parallel_generatedRoot_193.w_17_2 ctx), ((parallel_generatedRoot_193.w_17_3 ctx), ()))))
  hout : outputs = parallel_generatedRoot_193.outs ctx

theorem parallel_generatedRoot_193.facts_0  {ctx : parallel_generatedRoot_193Ctx} {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_193.constraints_0 backend hashModel tape path params i_0 (parallel_generatedRoot_193.w_0_0 ctx) (parallel_generatedRoot_193.w_1_0 ctx) (parallel_generatedRoot_193.w_2_0 ctx) (parallel_generatedRoot_193.w_3_0 ctx) (parallel_generatedRoot_193.w_4_0 ctx) (parallel_generatedRoot_193.w_4_1 ctx) (parallel_generatedRoot_193.w_4_2 ctx) (parallel_generatedRoot_193.w_4_3 ctx) (parallel_generatedRoot_193.w_5_0 ctx) (parallel_generatedRoot_193.w_6_0 ctx) (parallel_generatedRoot_193.w_7_0 ctx) (parallel_generatedRoot_193.w_8_0 ctx) (parallel_generatedRoot_193.w_9_0 ctx) (parallel_generatedRoot_193.w_10_0 ctx) (parallel_generatedRoot_193.w_11_0 ctx) (parallel_generatedRoot_193.w_12_0 ctx) (parallel_generatedRoot_193.w_13_0 ctx) (parallel_generatedRoot_193.w_14_0 ctx) (parallel_generatedRoot_193.w_15_0 ctx) (parallel_generatedRoot_193.w_16_0 ctx) (parallel_generatedRoot_193.w_17_0 ctx) (parallel_generatedRoot_193.w_17_1 ctx) (parallel_generatedRoot_193.w_17_2 ctx) (parallel_generatedRoot_193.w_17_3 ctx) outputs) :
    parallel_generatedRoot_193.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_193.Facts  (ctx : parallel_generatedRoot_193Ctx) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  c0 : parallel_generatedRoot_193.Facts_0 ctx outputs

theorem parallel_generatedRoot_193.facts  (ctx : parallel_generatedRoot_193Ctx) {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_193.body backend hashModel tape path params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_193.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_193.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_197.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) where
  inputs : (Fin 2607022080 → mxxMat1) × (Fin 2607022080 → mxxMat1) × (Fin 2607022080 → mxxMat1) × (Fin 2607022080 → mxxMat1) × Unit
  witness : parallel_generatedRoot_197.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_197Ctx" => (parallel_generatedRoot_197.Ctx backend hashModel tape path params i_0)

def parallel_generatedRoot_197.w_0_0  (ctx : parallel_generatedRoot_197Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.1

def parallel_generatedRoot_197.w_1_0  (ctx : parallel_generatedRoot_197Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.2.1

def parallel_generatedRoot_197.w_2_0  (ctx : parallel_generatedRoot_197Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_197.w_3_0  (ctx : parallel_generatedRoot_197Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_197.w_4_0  (ctx : parallel_generatedRoot_197Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_0

def parallel_generatedRoot_197.w_4_1  (ctx : parallel_generatedRoot_197Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_1

def parallel_generatedRoot_197.w_4_2  (ctx : parallel_generatedRoot_197Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_2

def parallel_generatedRoot_197.w_4_3  (ctx : parallel_generatedRoot_197Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_3

def parallel_generatedRoot_197.w_5_0  (ctx : parallel_generatedRoot_197Ctx) : mxxMat1 :=
  ctx.witness.w_5_0

def parallel_generatedRoot_197.w_6_0  (ctx : parallel_generatedRoot_197Ctx) : mxxMat1 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_197.w_7_0  (ctx : parallel_generatedRoot_197Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_197.w_5_0 ctx) (parallel_generatedRoot_197.w_6_0 ctx)

def parallel_generatedRoot_197.w_8_0  (ctx : parallel_generatedRoot_197Ctx) : mxxMat1 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_197.w_9_0  (ctx : parallel_generatedRoot_197Ctx) : mxxMat1 :=
  ctx.witness.w_9_0

def parallel_generatedRoot_197.w_10_0  (ctx : parallel_generatedRoot_197Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_197.w_8_0 ctx) (parallel_generatedRoot_197.w_9_0 ctx)

def parallel_generatedRoot_197.w_11_0  (ctx : parallel_generatedRoot_197Ctx) : mxxMat1 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_197.w_12_0  (ctx : parallel_generatedRoot_197Ctx) : mxxMat1 :=
  ctx.witness.w_12_0

def parallel_generatedRoot_197.w_13_0  (ctx : parallel_generatedRoot_197Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_197.w_11_0 ctx) (parallel_generatedRoot_197.w_12_0 ctx)

def parallel_generatedRoot_197.w_14_0  (ctx : parallel_generatedRoot_197Ctx) : mxxMat1 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_197.w_15_0  (ctx : parallel_generatedRoot_197Ctx) : mxxMat1 :=
  ctx.witness.w_15_0

def parallel_generatedRoot_197.w_16_0  (ctx : parallel_generatedRoot_197Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_197.w_14_0 ctx) (parallel_generatedRoot_197.w_15_0 ctx)

def parallel_generatedRoot_197.w_17_0  (ctx : parallel_generatedRoot_197Ctx) : mxxMat1 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_197.w_17_1  (ctx : parallel_generatedRoot_197Ctx) : mxxMat1 :=
  ctx.witness.w_17_1

def parallel_generatedRoot_197.w_17_2  (ctx : parallel_generatedRoot_197Ctx) : mxxMat1 :=
  ctx.witness.w_17_2

def parallel_generatedRoot_197.w_17_3  (ctx : parallel_generatedRoot_197Ctx) : mxxMat1 :=
  ctx.witness.w_17_3

def parallel_generatedRoot_197.outs  (ctx : parallel_generatedRoot_197Ctx) : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit :=
  ((parallel_generatedRoot_197.w_17_0 ctx), ((parallel_generatedRoot_197.w_17_1 ctx), ((parallel_generatedRoot_197.w_17_2 ctx), ((parallel_generatedRoot_197.w_17_3 ctx), ()))))

structure parallel_generatedRoot_197.Facts_0  (ctx : parallel_generatedRoot_197Ctx) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  t0 : (130) = 130
  h4_3 : (∀ i : Fin 130, parallel_parallel_generatedRoot_197_4 backend hashModel tape (path ++ [4, i.val]) params i_0 i ((parallel_generatedRoot_197.w_0_0 ctx), ((parallel_generatedRoot_197.w_1_0 ctx), ((parallel_generatedRoot_197.w_2_0 ctx), ((parallel_generatedRoot_197.w_3_0 ctx), ())))) (((parallel_generatedRoot_197.w_4_0 ctx) i), (((parallel_generatedRoot_197.w_4_1 ctx) i), (((parallel_generatedRoot_197.w_4_2 ctx) i), (((parallel_generatedRoot_197.w_4_3 ctx) i), ())))))
  h5_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_197.w_4_0 ctx) (0) (parallel_generatedRoot_197.w_5_0 ctx)
  h6_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_197.w_4_0 ctx) (1) (parallel_generatedRoot_197.w_6_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_197.w_4_1 ctx) (0) (parallel_generatedRoot_197.w_8_0 ctx)
  h9_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_197.w_4_1 ctx) (1) (parallel_generatedRoot_197.w_9_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_197.w_4_2 ctx) (0) (parallel_generatedRoot_197.w_11_0 ctx)
  h12_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_197.w_4_2 ctx) (1) (parallel_generatedRoot_197.w_12_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_197.w_4_3 ctx) (0) (parallel_generatedRoot_197.w_14_0 ctx)
  h15_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_197.w_4_3 ctx) (1) (parallel_generatedRoot_197.w_15_0 ctx)
  t10 : 0 ≤ (128)
  h17_3 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_parallel_generatedRoot_197_17 backend hashModel tape (path ++ [17, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, ((parallel_generatedRoot_197.w_4_0 ctx), ((parallel_generatedRoot_197.w_4_1 ctx), ((parallel_generatedRoot_197.w_4_2 ctx), ((parallel_generatedRoot_197.w_4_3 ctx), ())))))))) next) (Int.toNat (128)) ((parallel_generatedRoot_197.w_7_0 ctx), ((parallel_generatedRoot_197.w_10_0 ctx), ((parallel_generatedRoot_197.w_13_0 ctx), ((parallel_generatedRoot_197.w_16_0 ctx), ())))) ((parallel_generatedRoot_197.w_17_0 ctx), ((parallel_generatedRoot_197.w_17_1 ctx), ((parallel_generatedRoot_197.w_17_2 ctx), ((parallel_generatedRoot_197.w_17_3 ctx), ()))))
  hout : outputs = parallel_generatedRoot_197.outs ctx

theorem parallel_generatedRoot_197.facts_0  {ctx : parallel_generatedRoot_197Ctx} {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_197.constraints_0 backend hashModel tape path params i_0 (parallel_generatedRoot_197.w_0_0 ctx) (parallel_generatedRoot_197.w_1_0 ctx) (parallel_generatedRoot_197.w_2_0 ctx) (parallel_generatedRoot_197.w_3_0 ctx) (parallel_generatedRoot_197.w_4_0 ctx) (parallel_generatedRoot_197.w_4_1 ctx) (parallel_generatedRoot_197.w_4_2 ctx) (parallel_generatedRoot_197.w_4_3 ctx) (parallel_generatedRoot_197.w_5_0 ctx) (parallel_generatedRoot_197.w_6_0 ctx) (parallel_generatedRoot_197.w_7_0 ctx) (parallel_generatedRoot_197.w_8_0 ctx) (parallel_generatedRoot_197.w_9_0 ctx) (parallel_generatedRoot_197.w_10_0 ctx) (parallel_generatedRoot_197.w_11_0 ctx) (parallel_generatedRoot_197.w_12_0 ctx) (parallel_generatedRoot_197.w_13_0 ctx) (parallel_generatedRoot_197.w_14_0 ctx) (parallel_generatedRoot_197.w_15_0 ctx) (parallel_generatedRoot_197.w_16_0 ctx) (parallel_generatedRoot_197.w_17_0 ctx) (parallel_generatedRoot_197.w_17_1 ctx) (parallel_generatedRoot_197.w_17_2 ctx) (parallel_generatedRoot_197.w_17_3 ctx) outputs) :
    parallel_generatedRoot_197.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_197.Facts  (ctx : parallel_generatedRoot_197Ctx) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  c0 : parallel_generatedRoot_197.Facts_0 ctx outputs

theorem parallel_generatedRoot_197.facts  (ctx : parallel_generatedRoot_197Ctx) {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_197.body backend hashModel tape path params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_197.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_197.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_201.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) where
  inputs : (Fin 2607022080 → mxxMat1) × (Fin 2607022080 → mxxMat1) × (Fin 2607022080 → mxxMat1) × (Fin 2607022080 → mxxMat1) × Unit
  witness : parallel_generatedRoot_201.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_201Ctx" => (parallel_generatedRoot_201.Ctx backend hashModel tape path params i_0)

def parallel_generatedRoot_201.w_0_0  (ctx : parallel_generatedRoot_201Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.1

def parallel_generatedRoot_201.w_1_0  (ctx : parallel_generatedRoot_201Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.2.1

def parallel_generatedRoot_201.w_2_0  (ctx : parallel_generatedRoot_201Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_201.w_3_0  (ctx : parallel_generatedRoot_201Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_201.w_4_0  (ctx : parallel_generatedRoot_201Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_0

def parallel_generatedRoot_201.w_4_1  (ctx : parallel_generatedRoot_201Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_1

def parallel_generatedRoot_201.w_4_2  (ctx : parallel_generatedRoot_201Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_2

def parallel_generatedRoot_201.w_4_3  (ctx : parallel_generatedRoot_201Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_3

def parallel_generatedRoot_201.w_5_0  (ctx : parallel_generatedRoot_201Ctx) : mxxMat1 :=
  ctx.witness.w_5_0

def parallel_generatedRoot_201.w_6_0  (ctx : parallel_generatedRoot_201Ctx) : mxxMat1 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_201.w_7_0  (ctx : parallel_generatedRoot_201Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_201.w_5_0 ctx) (parallel_generatedRoot_201.w_6_0 ctx)

def parallel_generatedRoot_201.w_8_0  (ctx : parallel_generatedRoot_201Ctx) : mxxMat1 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_201.w_9_0  (ctx : parallel_generatedRoot_201Ctx) : mxxMat1 :=
  ctx.witness.w_9_0

def parallel_generatedRoot_201.w_10_0  (ctx : parallel_generatedRoot_201Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_201.w_8_0 ctx) (parallel_generatedRoot_201.w_9_0 ctx)

def parallel_generatedRoot_201.w_11_0  (ctx : parallel_generatedRoot_201Ctx) : mxxMat1 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_201.w_12_0  (ctx : parallel_generatedRoot_201Ctx) : mxxMat1 :=
  ctx.witness.w_12_0

def parallel_generatedRoot_201.w_13_0  (ctx : parallel_generatedRoot_201Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_201.w_11_0 ctx) (parallel_generatedRoot_201.w_12_0 ctx)

def parallel_generatedRoot_201.w_14_0  (ctx : parallel_generatedRoot_201Ctx) : mxxMat1 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_201.w_15_0  (ctx : parallel_generatedRoot_201Ctx) : mxxMat1 :=
  ctx.witness.w_15_0

def parallel_generatedRoot_201.w_16_0  (ctx : parallel_generatedRoot_201Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_201.w_14_0 ctx) (parallel_generatedRoot_201.w_15_0 ctx)

def parallel_generatedRoot_201.w_17_0  (ctx : parallel_generatedRoot_201Ctx) : mxxMat1 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_201.w_17_1  (ctx : parallel_generatedRoot_201Ctx) : mxxMat1 :=
  ctx.witness.w_17_1

def parallel_generatedRoot_201.w_17_2  (ctx : parallel_generatedRoot_201Ctx) : mxxMat1 :=
  ctx.witness.w_17_2

def parallel_generatedRoot_201.w_17_3  (ctx : parallel_generatedRoot_201Ctx) : mxxMat1 :=
  ctx.witness.w_17_3

def parallel_generatedRoot_201.outs  (ctx : parallel_generatedRoot_201Ctx) : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit :=
  ((parallel_generatedRoot_201.w_17_0 ctx), ((parallel_generatedRoot_201.w_17_1 ctx), ((parallel_generatedRoot_201.w_17_2 ctx), ((parallel_generatedRoot_201.w_17_3 ctx), ()))))

structure parallel_generatedRoot_201.Facts_0  (ctx : parallel_generatedRoot_201Ctx) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  t0 : (130) = 130
  h4_3 : (∀ i : Fin 130, parallel_parallel_generatedRoot_201_4 backend hashModel tape (path ++ [4, i.val]) params i_0 i ((parallel_generatedRoot_201.w_0_0 ctx), ((parallel_generatedRoot_201.w_1_0 ctx), ((parallel_generatedRoot_201.w_2_0 ctx), ((parallel_generatedRoot_201.w_3_0 ctx), ())))) (((parallel_generatedRoot_201.w_4_0 ctx) i), (((parallel_generatedRoot_201.w_4_1 ctx) i), (((parallel_generatedRoot_201.w_4_2 ctx) i), (((parallel_generatedRoot_201.w_4_3 ctx) i), ())))))
  h5_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_201.w_4_0 ctx) (0) (parallel_generatedRoot_201.w_5_0 ctx)
  h6_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_201.w_4_0 ctx) (1) (parallel_generatedRoot_201.w_6_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_201.w_4_1 ctx) (0) (parallel_generatedRoot_201.w_8_0 ctx)
  h9_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_201.w_4_1 ctx) (1) (parallel_generatedRoot_201.w_9_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_201.w_4_2 ctx) (0) (parallel_generatedRoot_201.w_11_0 ctx)
  h12_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_201.w_4_2 ctx) (1) (parallel_generatedRoot_201.w_12_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_201.w_4_3 ctx) (0) (parallel_generatedRoot_201.w_14_0 ctx)
  h15_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_201.w_4_3 ctx) (1) (parallel_generatedRoot_201.w_15_0 ctx)
  t10 : 0 ≤ (128)
  h17_3 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_parallel_generatedRoot_201_17 backend hashModel tape (path ++ [17, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, ((parallel_generatedRoot_201.w_4_0 ctx), ((parallel_generatedRoot_201.w_4_1 ctx), ((parallel_generatedRoot_201.w_4_2 ctx), ((parallel_generatedRoot_201.w_4_3 ctx), ())))))))) next) (Int.toNat (128)) ((parallel_generatedRoot_201.w_7_0 ctx), ((parallel_generatedRoot_201.w_10_0 ctx), ((parallel_generatedRoot_201.w_13_0 ctx), ((parallel_generatedRoot_201.w_16_0 ctx), ())))) ((parallel_generatedRoot_201.w_17_0 ctx), ((parallel_generatedRoot_201.w_17_1 ctx), ((parallel_generatedRoot_201.w_17_2 ctx), ((parallel_generatedRoot_201.w_17_3 ctx), ()))))
  hout : outputs = parallel_generatedRoot_201.outs ctx

theorem parallel_generatedRoot_201.facts_0  {ctx : parallel_generatedRoot_201Ctx} {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_201.constraints_0 backend hashModel tape path params i_0 (parallel_generatedRoot_201.w_0_0 ctx) (parallel_generatedRoot_201.w_1_0 ctx) (parallel_generatedRoot_201.w_2_0 ctx) (parallel_generatedRoot_201.w_3_0 ctx) (parallel_generatedRoot_201.w_4_0 ctx) (parallel_generatedRoot_201.w_4_1 ctx) (parallel_generatedRoot_201.w_4_2 ctx) (parallel_generatedRoot_201.w_4_3 ctx) (parallel_generatedRoot_201.w_5_0 ctx) (parallel_generatedRoot_201.w_6_0 ctx) (parallel_generatedRoot_201.w_7_0 ctx) (parallel_generatedRoot_201.w_8_0 ctx) (parallel_generatedRoot_201.w_9_0 ctx) (parallel_generatedRoot_201.w_10_0 ctx) (parallel_generatedRoot_201.w_11_0 ctx) (parallel_generatedRoot_201.w_12_0 ctx) (parallel_generatedRoot_201.w_13_0 ctx) (parallel_generatedRoot_201.w_14_0 ctx) (parallel_generatedRoot_201.w_15_0 ctx) (parallel_generatedRoot_201.w_16_0 ctx) (parallel_generatedRoot_201.w_17_0 ctx) (parallel_generatedRoot_201.w_17_1 ctx) (parallel_generatedRoot_201.w_17_2 ctx) (parallel_generatedRoot_201.w_17_3 ctx) outputs) :
    parallel_generatedRoot_201.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_201.Facts  (ctx : parallel_generatedRoot_201Ctx) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  c0 : parallel_generatedRoot_201.Facts_0 ctx outputs

theorem parallel_generatedRoot_201.facts  (ctx : parallel_generatedRoot_201Ctx) {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_201.body backend hashModel tape path params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_201.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_201.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_205.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) where
  inputs : (Fin 2607022080 → mxxMat1) × (Fin 2607022080 → mxxMat1) × (Fin 2607022080 → mxxMat1) × (Fin 2607022080 → mxxMat1) × Unit
  witness : parallel_generatedRoot_205.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_205Ctx" => (parallel_generatedRoot_205.Ctx backend hashModel tape path params i_0)

def parallel_generatedRoot_205.w_0_0  (ctx : parallel_generatedRoot_205Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.1

def parallel_generatedRoot_205.w_1_0  (ctx : parallel_generatedRoot_205Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.2.1

def parallel_generatedRoot_205.w_2_0  (ctx : parallel_generatedRoot_205Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_205.w_3_0  (ctx : parallel_generatedRoot_205Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_205.w_4_0  (ctx : parallel_generatedRoot_205Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_0

def parallel_generatedRoot_205.w_4_1  (ctx : parallel_generatedRoot_205Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_1

def parallel_generatedRoot_205.w_4_2  (ctx : parallel_generatedRoot_205Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_2

def parallel_generatedRoot_205.w_4_3  (ctx : parallel_generatedRoot_205Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_3

def parallel_generatedRoot_205.w_5_0  (ctx : parallel_generatedRoot_205Ctx) : mxxMat1 :=
  ctx.witness.w_5_0

def parallel_generatedRoot_205.w_6_0  (ctx : parallel_generatedRoot_205Ctx) : mxxMat1 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_205.w_7_0  (ctx : parallel_generatedRoot_205Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_205.w_5_0 ctx) (parallel_generatedRoot_205.w_6_0 ctx)

def parallel_generatedRoot_205.w_8_0  (ctx : parallel_generatedRoot_205Ctx) : mxxMat1 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_205.w_9_0  (ctx : parallel_generatedRoot_205Ctx) : mxxMat1 :=
  ctx.witness.w_9_0

def parallel_generatedRoot_205.w_10_0  (ctx : parallel_generatedRoot_205Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_205.w_8_0 ctx) (parallel_generatedRoot_205.w_9_0 ctx)

def parallel_generatedRoot_205.w_11_0  (ctx : parallel_generatedRoot_205Ctx) : mxxMat1 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_205.w_12_0  (ctx : parallel_generatedRoot_205Ctx) : mxxMat1 :=
  ctx.witness.w_12_0

def parallel_generatedRoot_205.w_13_0  (ctx : parallel_generatedRoot_205Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_205.w_11_0 ctx) (parallel_generatedRoot_205.w_12_0 ctx)

def parallel_generatedRoot_205.w_14_0  (ctx : parallel_generatedRoot_205Ctx) : mxxMat1 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_205.w_15_0  (ctx : parallel_generatedRoot_205Ctx) : mxxMat1 :=
  ctx.witness.w_15_0

def parallel_generatedRoot_205.w_16_0  (ctx : parallel_generatedRoot_205Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_205.w_14_0 ctx) (parallel_generatedRoot_205.w_15_0 ctx)

def parallel_generatedRoot_205.w_17_0  (ctx : parallel_generatedRoot_205Ctx) : mxxMat1 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_205.w_17_1  (ctx : parallel_generatedRoot_205Ctx) : mxxMat1 :=
  ctx.witness.w_17_1

def parallel_generatedRoot_205.w_17_2  (ctx : parallel_generatedRoot_205Ctx) : mxxMat1 :=
  ctx.witness.w_17_2

def parallel_generatedRoot_205.w_17_3  (ctx : parallel_generatedRoot_205Ctx) : mxxMat1 :=
  ctx.witness.w_17_3

def parallel_generatedRoot_205.outs  (ctx : parallel_generatedRoot_205Ctx) : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit :=
  ((parallel_generatedRoot_205.w_17_0 ctx), ((parallel_generatedRoot_205.w_17_1 ctx), ((parallel_generatedRoot_205.w_17_2 ctx), ((parallel_generatedRoot_205.w_17_3 ctx), ()))))

structure parallel_generatedRoot_205.Facts_0  (ctx : parallel_generatedRoot_205Ctx) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  t0 : (130) = 130
  h4_3 : (∀ i : Fin 130, parallel_parallel_generatedRoot_205_4 backend hashModel tape (path ++ [4, i.val]) params i_0 i ((parallel_generatedRoot_205.w_0_0 ctx), ((parallel_generatedRoot_205.w_1_0 ctx), ((parallel_generatedRoot_205.w_2_0 ctx), ((parallel_generatedRoot_205.w_3_0 ctx), ())))) (((parallel_generatedRoot_205.w_4_0 ctx) i), (((parallel_generatedRoot_205.w_4_1 ctx) i), (((parallel_generatedRoot_205.w_4_2 ctx) i), (((parallel_generatedRoot_205.w_4_3 ctx) i), ())))))
  h5_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_205.w_4_0 ctx) (0) (parallel_generatedRoot_205.w_5_0 ctx)
  h6_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_205.w_4_0 ctx) (1) (parallel_generatedRoot_205.w_6_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_205.w_4_1 ctx) (0) (parallel_generatedRoot_205.w_8_0 ctx)
  h9_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_205.w_4_1 ctx) (1) (parallel_generatedRoot_205.w_9_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_205.w_4_2 ctx) (0) (parallel_generatedRoot_205.w_11_0 ctx)
  h12_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_205.w_4_2 ctx) (1) (parallel_generatedRoot_205.w_12_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_205.w_4_3 ctx) (0) (parallel_generatedRoot_205.w_14_0 ctx)
  h15_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_205.w_4_3 ctx) (1) (parallel_generatedRoot_205.w_15_0 ctx)
  t10 : 0 ≤ (128)
  h17_3 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_parallel_generatedRoot_205_17 backend hashModel tape (path ++ [17, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, ((parallel_generatedRoot_205.w_4_0 ctx), ((parallel_generatedRoot_205.w_4_1 ctx), ((parallel_generatedRoot_205.w_4_2 ctx), ((parallel_generatedRoot_205.w_4_3 ctx), ())))))))) next) (Int.toNat (128)) ((parallel_generatedRoot_205.w_7_0 ctx), ((parallel_generatedRoot_205.w_10_0 ctx), ((parallel_generatedRoot_205.w_13_0 ctx), ((parallel_generatedRoot_205.w_16_0 ctx), ())))) ((parallel_generatedRoot_205.w_17_0 ctx), ((parallel_generatedRoot_205.w_17_1 ctx), ((parallel_generatedRoot_205.w_17_2 ctx), ((parallel_generatedRoot_205.w_17_3 ctx), ()))))
  hout : outputs = parallel_generatedRoot_205.outs ctx

theorem parallel_generatedRoot_205.facts_0  {ctx : parallel_generatedRoot_205Ctx} {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_205.constraints_0 backend hashModel tape path params i_0 (parallel_generatedRoot_205.w_0_0 ctx) (parallel_generatedRoot_205.w_1_0 ctx) (parallel_generatedRoot_205.w_2_0 ctx) (parallel_generatedRoot_205.w_3_0 ctx) (parallel_generatedRoot_205.w_4_0 ctx) (parallel_generatedRoot_205.w_4_1 ctx) (parallel_generatedRoot_205.w_4_2 ctx) (parallel_generatedRoot_205.w_4_3 ctx) (parallel_generatedRoot_205.w_5_0 ctx) (parallel_generatedRoot_205.w_6_0 ctx) (parallel_generatedRoot_205.w_7_0 ctx) (parallel_generatedRoot_205.w_8_0 ctx) (parallel_generatedRoot_205.w_9_0 ctx) (parallel_generatedRoot_205.w_10_0 ctx) (parallel_generatedRoot_205.w_11_0 ctx) (parallel_generatedRoot_205.w_12_0 ctx) (parallel_generatedRoot_205.w_13_0 ctx) (parallel_generatedRoot_205.w_14_0 ctx) (parallel_generatedRoot_205.w_15_0 ctx) (parallel_generatedRoot_205.w_16_0 ctx) (parallel_generatedRoot_205.w_17_0 ctx) (parallel_generatedRoot_205.w_17_1 ctx) (parallel_generatedRoot_205.w_17_2 ctx) (parallel_generatedRoot_205.w_17_3 ctx) outputs) :
    parallel_generatedRoot_205.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_205.Facts  (ctx : parallel_generatedRoot_205Ctx) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  c0 : parallel_generatedRoot_205.Facts_0 ctx outputs

theorem parallel_generatedRoot_205.facts  (ctx : parallel_generatedRoot_205Ctx) {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_205.body backend hashModel tape path params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_205.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_205.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_209.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) where
  inputs : (Fin 2607022080 → mxxMat1) × (Fin 2607022080 → mxxMat1) × (Fin 2607022080 → mxxMat1) × (Fin 2607022080 → mxxMat1) × Unit
  witness : parallel_generatedRoot_209.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_209Ctx" => (parallel_generatedRoot_209.Ctx backend hashModel tape path params i_0)

def parallel_generatedRoot_209.w_0_0  (ctx : parallel_generatedRoot_209Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.1

def parallel_generatedRoot_209.w_1_0  (ctx : parallel_generatedRoot_209Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.2.1

def parallel_generatedRoot_209.w_2_0  (ctx : parallel_generatedRoot_209Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_209.w_3_0  (ctx : parallel_generatedRoot_209Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_209.w_4_0  (ctx : parallel_generatedRoot_209Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_0

def parallel_generatedRoot_209.w_4_1  (ctx : parallel_generatedRoot_209Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_1

def parallel_generatedRoot_209.w_4_2  (ctx : parallel_generatedRoot_209Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_2

def parallel_generatedRoot_209.w_4_3  (ctx : parallel_generatedRoot_209Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_3

def parallel_generatedRoot_209.w_5_0  (ctx : parallel_generatedRoot_209Ctx) : mxxMat1 :=
  ctx.witness.w_5_0

def parallel_generatedRoot_209.w_6_0  (ctx : parallel_generatedRoot_209Ctx) : mxxMat1 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_209.w_7_0  (ctx : parallel_generatedRoot_209Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_209.w_5_0 ctx) (parallel_generatedRoot_209.w_6_0 ctx)

def parallel_generatedRoot_209.w_8_0  (ctx : parallel_generatedRoot_209Ctx) : mxxMat1 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_209.w_9_0  (ctx : parallel_generatedRoot_209Ctx) : mxxMat1 :=
  ctx.witness.w_9_0

def parallel_generatedRoot_209.w_10_0  (ctx : parallel_generatedRoot_209Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_209.w_8_0 ctx) (parallel_generatedRoot_209.w_9_0 ctx)

def parallel_generatedRoot_209.w_11_0  (ctx : parallel_generatedRoot_209Ctx) : mxxMat1 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_209.w_12_0  (ctx : parallel_generatedRoot_209Ctx) : mxxMat1 :=
  ctx.witness.w_12_0

def parallel_generatedRoot_209.w_13_0  (ctx : parallel_generatedRoot_209Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_209.w_11_0 ctx) (parallel_generatedRoot_209.w_12_0 ctx)

def parallel_generatedRoot_209.w_14_0  (ctx : parallel_generatedRoot_209Ctx) : mxxMat1 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_209.w_15_0  (ctx : parallel_generatedRoot_209Ctx) : mxxMat1 :=
  ctx.witness.w_15_0

def parallel_generatedRoot_209.w_16_0  (ctx : parallel_generatedRoot_209Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_209.w_14_0 ctx) (parallel_generatedRoot_209.w_15_0 ctx)

def parallel_generatedRoot_209.w_17_0  (ctx : parallel_generatedRoot_209Ctx) : mxxMat1 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_209.w_17_1  (ctx : parallel_generatedRoot_209Ctx) : mxxMat1 :=
  ctx.witness.w_17_1

def parallel_generatedRoot_209.w_17_2  (ctx : parallel_generatedRoot_209Ctx) : mxxMat1 :=
  ctx.witness.w_17_2

def parallel_generatedRoot_209.w_17_3  (ctx : parallel_generatedRoot_209Ctx) : mxxMat1 :=
  ctx.witness.w_17_3

def parallel_generatedRoot_209.outs  (ctx : parallel_generatedRoot_209Ctx) : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit :=
  ((parallel_generatedRoot_209.w_17_0 ctx), ((parallel_generatedRoot_209.w_17_1 ctx), ((parallel_generatedRoot_209.w_17_2 ctx), ((parallel_generatedRoot_209.w_17_3 ctx), ()))))

structure parallel_generatedRoot_209.Facts_0  (ctx : parallel_generatedRoot_209Ctx) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  t0 : (130) = 130
  h4_3 : (∀ i : Fin 130, parallel_parallel_generatedRoot_209_4 backend hashModel tape (path ++ [4, i.val]) params i_0 i ((parallel_generatedRoot_209.w_0_0 ctx), ((parallel_generatedRoot_209.w_1_0 ctx), ((parallel_generatedRoot_209.w_2_0 ctx), ((parallel_generatedRoot_209.w_3_0 ctx), ())))) (((parallel_generatedRoot_209.w_4_0 ctx) i), (((parallel_generatedRoot_209.w_4_1 ctx) i), (((parallel_generatedRoot_209.w_4_2 ctx) i), (((parallel_generatedRoot_209.w_4_3 ctx) i), ())))))
  h5_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_209.w_4_0 ctx) (0) (parallel_generatedRoot_209.w_5_0 ctx)
  h6_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_209.w_4_0 ctx) (1) (parallel_generatedRoot_209.w_6_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_209.w_4_1 ctx) (0) (parallel_generatedRoot_209.w_8_0 ctx)
  h9_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_209.w_4_1 ctx) (1) (parallel_generatedRoot_209.w_9_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_209.w_4_2 ctx) (0) (parallel_generatedRoot_209.w_11_0 ctx)
  h12_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_209.w_4_2 ctx) (1) (parallel_generatedRoot_209.w_12_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_209.w_4_3 ctx) (0) (parallel_generatedRoot_209.w_14_0 ctx)
  h15_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_209.w_4_3 ctx) (1) (parallel_generatedRoot_209.w_15_0 ctx)
  t10 : 0 ≤ (128)
  h17_3 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_parallel_generatedRoot_209_17 backend hashModel tape (path ++ [17, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, ((parallel_generatedRoot_209.w_4_0 ctx), ((parallel_generatedRoot_209.w_4_1 ctx), ((parallel_generatedRoot_209.w_4_2 ctx), ((parallel_generatedRoot_209.w_4_3 ctx), ())))))))) next) (Int.toNat (128)) ((parallel_generatedRoot_209.w_7_0 ctx), ((parallel_generatedRoot_209.w_10_0 ctx), ((parallel_generatedRoot_209.w_13_0 ctx), ((parallel_generatedRoot_209.w_16_0 ctx), ())))) ((parallel_generatedRoot_209.w_17_0 ctx), ((parallel_generatedRoot_209.w_17_1 ctx), ((parallel_generatedRoot_209.w_17_2 ctx), ((parallel_generatedRoot_209.w_17_3 ctx), ()))))
  hout : outputs = parallel_generatedRoot_209.outs ctx

theorem parallel_generatedRoot_209.facts_0  {ctx : parallel_generatedRoot_209Ctx} {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_209.constraints_0 backend hashModel tape path params i_0 (parallel_generatedRoot_209.w_0_0 ctx) (parallel_generatedRoot_209.w_1_0 ctx) (parallel_generatedRoot_209.w_2_0 ctx) (parallel_generatedRoot_209.w_3_0 ctx) (parallel_generatedRoot_209.w_4_0 ctx) (parallel_generatedRoot_209.w_4_1 ctx) (parallel_generatedRoot_209.w_4_2 ctx) (parallel_generatedRoot_209.w_4_3 ctx) (parallel_generatedRoot_209.w_5_0 ctx) (parallel_generatedRoot_209.w_6_0 ctx) (parallel_generatedRoot_209.w_7_0 ctx) (parallel_generatedRoot_209.w_8_0 ctx) (parallel_generatedRoot_209.w_9_0 ctx) (parallel_generatedRoot_209.w_10_0 ctx) (parallel_generatedRoot_209.w_11_0 ctx) (parallel_generatedRoot_209.w_12_0 ctx) (parallel_generatedRoot_209.w_13_0 ctx) (parallel_generatedRoot_209.w_14_0 ctx) (parallel_generatedRoot_209.w_15_0 ctx) (parallel_generatedRoot_209.w_16_0 ctx) (parallel_generatedRoot_209.w_17_0 ctx) (parallel_generatedRoot_209.w_17_1 ctx) (parallel_generatedRoot_209.w_17_2 ctx) (parallel_generatedRoot_209.w_17_3 ctx) outputs) :
    parallel_generatedRoot_209.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_209.Facts  (ctx : parallel_generatedRoot_209Ctx) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  c0 : parallel_generatedRoot_209.Facts_0 ctx outputs

theorem parallel_generatedRoot_209.facts  (ctx : parallel_generatedRoot_209Ctx) {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_209.body backend hashModel tape path params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_209.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_209.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_213.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) where
  inputs : (Fin 2607022080 → mxxMat1) × (Fin 2607022080 → mxxMat1) × (Fin 2607022080 → mxxMat1) × (Fin 2607022080 → mxxMat1) × Unit
  witness : parallel_generatedRoot_213.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_213Ctx" => (parallel_generatedRoot_213.Ctx backend hashModel tape path params i_0)

def parallel_generatedRoot_213.w_0_0  (ctx : parallel_generatedRoot_213Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.1

def parallel_generatedRoot_213.w_1_0  (ctx : parallel_generatedRoot_213Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.2.1

def parallel_generatedRoot_213.w_2_0  (ctx : parallel_generatedRoot_213Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_213.w_3_0  (ctx : parallel_generatedRoot_213Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_213.w_4_0  (ctx : parallel_generatedRoot_213Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_0

def parallel_generatedRoot_213.w_4_1  (ctx : parallel_generatedRoot_213Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_1

def parallel_generatedRoot_213.w_4_2  (ctx : parallel_generatedRoot_213Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_2

def parallel_generatedRoot_213.w_4_3  (ctx : parallel_generatedRoot_213Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_3

def parallel_generatedRoot_213.w_5_0  (ctx : parallel_generatedRoot_213Ctx) : mxxMat1 :=
  ctx.witness.w_5_0

def parallel_generatedRoot_213.w_6_0  (ctx : parallel_generatedRoot_213Ctx) : mxxMat1 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_213.w_7_0  (ctx : parallel_generatedRoot_213Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_213.w_5_0 ctx) (parallel_generatedRoot_213.w_6_0 ctx)

def parallel_generatedRoot_213.w_8_0  (ctx : parallel_generatedRoot_213Ctx) : mxxMat1 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_213.w_9_0  (ctx : parallel_generatedRoot_213Ctx) : mxxMat1 :=
  ctx.witness.w_9_0

def parallel_generatedRoot_213.w_10_0  (ctx : parallel_generatedRoot_213Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_213.w_8_0 ctx) (parallel_generatedRoot_213.w_9_0 ctx)

def parallel_generatedRoot_213.w_11_0  (ctx : parallel_generatedRoot_213Ctx) : mxxMat1 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_213.w_12_0  (ctx : parallel_generatedRoot_213Ctx) : mxxMat1 :=
  ctx.witness.w_12_0

def parallel_generatedRoot_213.w_13_0  (ctx : parallel_generatedRoot_213Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_213.w_11_0 ctx) (parallel_generatedRoot_213.w_12_0 ctx)

def parallel_generatedRoot_213.w_14_0  (ctx : parallel_generatedRoot_213Ctx) : mxxMat1 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_213.w_15_0  (ctx : parallel_generatedRoot_213Ctx) : mxxMat1 :=
  ctx.witness.w_15_0

def parallel_generatedRoot_213.w_16_0  (ctx : parallel_generatedRoot_213Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_213.w_14_0 ctx) (parallel_generatedRoot_213.w_15_0 ctx)

def parallel_generatedRoot_213.w_17_0  (ctx : parallel_generatedRoot_213Ctx) : mxxMat1 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_213.w_17_1  (ctx : parallel_generatedRoot_213Ctx) : mxxMat1 :=
  ctx.witness.w_17_1

def parallel_generatedRoot_213.w_17_2  (ctx : parallel_generatedRoot_213Ctx) : mxxMat1 :=
  ctx.witness.w_17_2

def parallel_generatedRoot_213.w_17_3  (ctx : parallel_generatedRoot_213Ctx) : mxxMat1 :=
  ctx.witness.w_17_3

def parallel_generatedRoot_213.outs  (ctx : parallel_generatedRoot_213Ctx) : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit :=
  ((parallel_generatedRoot_213.w_17_0 ctx), ((parallel_generatedRoot_213.w_17_1 ctx), ((parallel_generatedRoot_213.w_17_2 ctx), ((parallel_generatedRoot_213.w_17_3 ctx), ()))))

structure parallel_generatedRoot_213.Facts_0  (ctx : parallel_generatedRoot_213Ctx) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  t0 : (130) = 130
  h4_3 : (∀ i : Fin 130, parallel_parallel_generatedRoot_213_4 backend hashModel tape (path ++ [4, i.val]) params i_0 i ((parallel_generatedRoot_213.w_0_0 ctx), ((parallel_generatedRoot_213.w_1_0 ctx), ((parallel_generatedRoot_213.w_2_0 ctx), ((parallel_generatedRoot_213.w_3_0 ctx), ())))) (((parallel_generatedRoot_213.w_4_0 ctx) i), (((parallel_generatedRoot_213.w_4_1 ctx) i), (((parallel_generatedRoot_213.w_4_2 ctx) i), (((parallel_generatedRoot_213.w_4_3 ctx) i), ())))))
  h5_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_213.w_4_0 ctx) (0) (parallel_generatedRoot_213.w_5_0 ctx)
  h6_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_213.w_4_0 ctx) (1) (parallel_generatedRoot_213.w_6_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_213.w_4_1 ctx) (0) (parallel_generatedRoot_213.w_8_0 ctx)
  h9_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_213.w_4_1 ctx) (1) (parallel_generatedRoot_213.w_9_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_213.w_4_2 ctx) (0) (parallel_generatedRoot_213.w_11_0 ctx)
  h12_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_213.w_4_2 ctx) (1) (parallel_generatedRoot_213.w_12_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_213.w_4_3 ctx) (0) (parallel_generatedRoot_213.w_14_0 ctx)
  h15_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_213.w_4_3 ctx) (1) (parallel_generatedRoot_213.w_15_0 ctx)
  t10 : 0 ≤ (128)
  h17_3 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_parallel_generatedRoot_213_17 backend hashModel tape (path ++ [17, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, ((parallel_generatedRoot_213.w_4_0 ctx), ((parallel_generatedRoot_213.w_4_1 ctx), ((parallel_generatedRoot_213.w_4_2 ctx), ((parallel_generatedRoot_213.w_4_3 ctx), ())))))))) next) (Int.toNat (128)) ((parallel_generatedRoot_213.w_7_0 ctx), ((parallel_generatedRoot_213.w_10_0 ctx), ((parallel_generatedRoot_213.w_13_0 ctx), ((parallel_generatedRoot_213.w_16_0 ctx), ())))) ((parallel_generatedRoot_213.w_17_0 ctx), ((parallel_generatedRoot_213.w_17_1 ctx), ((parallel_generatedRoot_213.w_17_2 ctx), ((parallel_generatedRoot_213.w_17_3 ctx), ()))))
  hout : outputs = parallel_generatedRoot_213.outs ctx

theorem parallel_generatedRoot_213.facts_0  {ctx : parallel_generatedRoot_213Ctx} {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_213.constraints_0 backend hashModel tape path params i_0 (parallel_generatedRoot_213.w_0_0 ctx) (parallel_generatedRoot_213.w_1_0 ctx) (parallel_generatedRoot_213.w_2_0 ctx) (parallel_generatedRoot_213.w_3_0 ctx) (parallel_generatedRoot_213.w_4_0 ctx) (parallel_generatedRoot_213.w_4_1 ctx) (parallel_generatedRoot_213.w_4_2 ctx) (parallel_generatedRoot_213.w_4_3 ctx) (parallel_generatedRoot_213.w_5_0 ctx) (parallel_generatedRoot_213.w_6_0 ctx) (parallel_generatedRoot_213.w_7_0 ctx) (parallel_generatedRoot_213.w_8_0 ctx) (parallel_generatedRoot_213.w_9_0 ctx) (parallel_generatedRoot_213.w_10_0 ctx) (parallel_generatedRoot_213.w_11_0 ctx) (parallel_generatedRoot_213.w_12_0 ctx) (parallel_generatedRoot_213.w_13_0 ctx) (parallel_generatedRoot_213.w_14_0 ctx) (parallel_generatedRoot_213.w_15_0 ctx) (parallel_generatedRoot_213.w_16_0 ctx) (parallel_generatedRoot_213.w_17_0 ctx) (parallel_generatedRoot_213.w_17_1 ctx) (parallel_generatedRoot_213.w_17_2 ctx) (parallel_generatedRoot_213.w_17_3 ctx) outputs) :
    parallel_generatedRoot_213.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_213.Facts  (ctx : parallel_generatedRoot_213Ctx) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  c0 : parallel_generatedRoot_213.Facts_0 ctx outputs

theorem parallel_generatedRoot_213.facts  (ctx : parallel_generatedRoot_213Ctx) {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_213.body backend hashModel tape path params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_213.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_213.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_217.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) where
  inputs : (Fin 2607022080 → mxxMat1) × (Fin 2607022080 → mxxMat1) × (Fin 2607022080 → mxxMat1) × (Fin 2607022080 → mxxMat1) × Unit
  witness : parallel_generatedRoot_217.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_217Ctx" => (parallel_generatedRoot_217.Ctx backend hashModel tape path params i_0)

def parallel_generatedRoot_217.w_0_0  (ctx : parallel_generatedRoot_217Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.1

def parallel_generatedRoot_217.w_1_0  (ctx : parallel_generatedRoot_217Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.2.1

def parallel_generatedRoot_217.w_2_0  (ctx : parallel_generatedRoot_217Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_217.w_3_0  (ctx : parallel_generatedRoot_217Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_217.w_4_0  (ctx : parallel_generatedRoot_217Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_0

def parallel_generatedRoot_217.w_4_1  (ctx : parallel_generatedRoot_217Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_1

def parallel_generatedRoot_217.w_4_2  (ctx : parallel_generatedRoot_217Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_2

def parallel_generatedRoot_217.w_4_3  (ctx : parallel_generatedRoot_217Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_3

def parallel_generatedRoot_217.w_5_0  (ctx : parallel_generatedRoot_217Ctx) : mxxMat1 :=
  ctx.witness.w_5_0

def parallel_generatedRoot_217.w_6_0  (ctx : parallel_generatedRoot_217Ctx) : mxxMat1 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_217.w_7_0  (ctx : parallel_generatedRoot_217Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_217.w_5_0 ctx) (parallel_generatedRoot_217.w_6_0 ctx)

def parallel_generatedRoot_217.w_8_0  (ctx : parallel_generatedRoot_217Ctx) : mxxMat1 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_217.w_9_0  (ctx : parallel_generatedRoot_217Ctx) : mxxMat1 :=
  ctx.witness.w_9_0

def parallel_generatedRoot_217.w_10_0  (ctx : parallel_generatedRoot_217Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_217.w_8_0 ctx) (parallel_generatedRoot_217.w_9_0 ctx)

def parallel_generatedRoot_217.w_11_0  (ctx : parallel_generatedRoot_217Ctx) : mxxMat1 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_217.w_12_0  (ctx : parallel_generatedRoot_217Ctx) : mxxMat1 :=
  ctx.witness.w_12_0

def parallel_generatedRoot_217.w_13_0  (ctx : parallel_generatedRoot_217Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_217.w_11_0 ctx) (parallel_generatedRoot_217.w_12_0 ctx)

def parallel_generatedRoot_217.w_14_0  (ctx : parallel_generatedRoot_217Ctx) : mxxMat1 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_217.w_15_0  (ctx : parallel_generatedRoot_217Ctx) : mxxMat1 :=
  ctx.witness.w_15_0

def parallel_generatedRoot_217.w_16_0  (ctx : parallel_generatedRoot_217Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_217.w_14_0 ctx) (parallel_generatedRoot_217.w_15_0 ctx)

def parallel_generatedRoot_217.w_17_0  (ctx : parallel_generatedRoot_217Ctx) : mxxMat1 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_217.w_17_1  (ctx : parallel_generatedRoot_217Ctx) : mxxMat1 :=
  ctx.witness.w_17_1

def parallel_generatedRoot_217.w_17_2  (ctx : parallel_generatedRoot_217Ctx) : mxxMat1 :=
  ctx.witness.w_17_2

def parallel_generatedRoot_217.w_17_3  (ctx : parallel_generatedRoot_217Ctx) : mxxMat1 :=
  ctx.witness.w_17_3

def parallel_generatedRoot_217.outs  (ctx : parallel_generatedRoot_217Ctx) : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit :=
  ((parallel_generatedRoot_217.w_17_0 ctx), ((parallel_generatedRoot_217.w_17_1 ctx), ((parallel_generatedRoot_217.w_17_2 ctx), ((parallel_generatedRoot_217.w_17_3 ctx), ()))))

structure parallel_generatedRoot_217.Facts_0  (ctx : parallel_generatedRoot_217Ctx) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  t0 : (130) = 130
  h4_3 : (∀ i : Fin 130, parallel_parallel_generatedRoot_217_4 backend hashModel tape (path ++ [4, i.val]) params i_0 i ((parallel_generatedRoot_217.w_0_0 ctx), ((parallel_generatedRoot_217.w_1_0 ctx), ((parallel_generatedRoot_217.w_2_0 ctx), ((parallel_generatedRoot_217.w_3_0 ctx), ())))) (((parallel_generatedRoot_217.w_4_0 ctx) i), (((parallel_generatedRoot_217.w_4_1 ctx) i), (((parallel_generatedRoot_217.w_4_2 ctx) i), (((parallel_generatedRoot_217.w_4_3 ctx) i), ())))))
  h5_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_217.w_4_0 ctx) (0) (parallel_generatedRoot_217.w_5_0 ctx)
  h6_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_217.w_4_0 ctx) (1) (parallel_generatedRoot_217.w_6_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_217.w_4_1 ctx) (0) (parallel_generatedRoot_217.w_8_0 ctx)
  h9_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_217.w_4_1 ctx) (1) (parallel_generatedRoot_217.w_9_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_217.w_4_2 ctx) (0) (parallel_generatedRoot_217.w_11_0 ctx)
  h12_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_217.w_4_2 ctx) (1) (parallel_generatedRoot_217.w_12_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_217.w_4_3 ctx) (0) (parallel_generatedRoot_217.w_14_0 ctx)
  h15_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_217.w_4_3 ctx) (1) (parallel_generatedRoot_217.w_15_0 ctx)
  t10 : 0 ≤ (128)
  h17_3 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_parallel_generatedRoot_217_17 backend hashModel tape (path ++ [17, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, ((parallel_generatedRoot_217.w_4_0 ctx), ((parallel_generatedRoot_217.w_4_1 ctx), ((parallel_generatedRoot_217.w_4_2 ctx), ((parallel_generatedRoot_217.w_4_3 ctx), ())))))))) next) (Int.toNat (128)) ((parallel_generatedRoot_217.w_7_0 ctx), ((parallel_generatedRoot_217.w_10_0 ctx), ((parallel_generatedRoot_217.w_13_0 ctx), ((parallel_generatedRoot_217.w_16_0 ctx), ())))) ((parallel_generatedRoot_217.w_17_0 ctx), ((parallel_generatedRoot_217.w_17_1 ctx), ((parallel_generatedRoot_217.w_17_2 ctx), ((parallel_generatedRoot_217.w_17_3 ctx), ()))))
  hout : outputs = parallel_generatedRoot_217.outs ctx

theorem parallel_generatedRoot_217.facts_0  {ctx : parallel_generatedRoot_217Ctx} {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_217.constraints_0 backend hashModel tape path params i_0 (parallel_generatedRoot_217.w_0_0 ctx) (parallel_generatedRoot_217.w_1_0 ctx) (parallel_generatedRoot_217.w_2_0 ctx) (parallel_generatedRoot_217.w_3_0 ctx) (parallel_generatedRoot_217.w_4_0 ctx) (parallel_generatedRoot_217.w_4_1 ctx) (parallel_generatedRoot_217.w_4_2 ctx) (parallel_generatedRoot_217.w_4_3 ctx) (parallel_generatedRoot_217.w_5_0 ctx) (parallel_generatedRoot_217.w_6_0 ctx) (parallel_generatedRoot_217.w_7_0 ctx) (parallel_generatedRoot_217.w_8_0 ctx) (parallel_generatedRoot_217.w_9_0 ctx) (parallel_generatedRoot_217.w_10_0 ctx) (parallel_generatedRoot_217.w_11_0 ctx) (parallel_generatedRoot_217.w_12_0 ctx) (parallel_generatedRoot_217.w_13_0 ctx) (parallel_generatedRoot_217.w_14_0 ctx) (parallel_generatedRoot_217.w_15_0 ctx) (parallel_generatedRoot_217.w_16_0 ctx) (parallel_generatedRoot_217.w_17_0 ctx) (parallel_generatedRoot_217.w_17_1 ctx) (parallel_generatedRoot_217.w_17_2 ctx) (parallel_generatedRoot_217.w_17_3 ctx) outputs) :
    parallel_generatedRoot_217.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_217.Facts  (ctx : parallel_generatedRoot_217Ctx) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  c0 : parallel_generatedRoot_217.Facts_0 ctx outputs

theorem parallel_generatedRoot_217.facts  (ctx : parallel_generatedRoot_217Ctx) {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_217.body backend hashModel tape path params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_217.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_217.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_221.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) where
  inputs : (Fin 2607022080 → mxxMat1) × (Fin 2607022080 → mxxMat1) × (Fin 2607022080 → mxxMat1) × (Fin 2607022080 → mxxMat1) × Unit
  witness : parallel_generatedRoot_221.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_221Ctx" => (parallel_generatedRoot_221.Ctx backend hashModel tape path params i_0)

def parallel_generatedRoot_221.w_0_0  (ctx : parallel_generatedRoot_221Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.1

def parallel_generatedRoot_221.w_1_0  (ctx : parallel_generatedRoot_221Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.2.1

def parallel_generatedRoot_221.w_2_0  (ctx : parallel_generatedRoot_221Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_221.w_3_0  (ctx : parallel_generatedRoot_221Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_221.w_4_0  (ctx : parallel_generatedRoot_221Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_0

def parallel_generatedRoot_221.w_4_1  (ctx : parallel_generatedRoot_221Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_1

def parallel_generatedRoot_221.w_4_2  (ctx : parallel_generatedRoot_221Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_2

def parallel_generatedRoot_221.w_4_3  (ctx : parallel_generatedRoot_221Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_3

def parallel_generatedRoot_221.w_5_0  (ctx : parallel_generatedRoot_221Ctx) : mxxMat1 :=
  ctx.witness.w_5_0

def parallel_generatedRoot_221.w_6_0  (ctx : parallel_generatedRoot_221Ctx) : mxxMat1 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_221.w_7_0  (ctx : parallel_generatedRoot_221Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_221.w_5_0 ctx) (parallel_generatedRoot_221.w_6_0 ctx)

def parallel_generatedRoot_221.w_8_0  (ctx : parallel_generatedRoot_221Ctx) : mxxMat1 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_221.w_9_0  (ctx : parallel_generatedRoot_221Ctx) : mxxMat1 :=
  ctx.witness.w_9_0

def parallel_generatedRoot_221.w_10_0  (ctx : parallel_generatedRoot_221Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_221.w_8_0 ctx) (parallel_generatedRoot_221.w_9_0 ctx)

def parallel_generatedRoot_221.w_11_0  (ctx : parallel_generatedRoot_221Ctx) : mxxMat1 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_221.w_12_0  (ctx : parallel_generatedRoot_221Ctx) : mxxMat1 :=
  ctx.witness.w_12_0

def parallel_generatedRoot_221.w_13_0  (ctx : parallel_generatedRoot_221Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_221.w_11_0 ctx) (parallel_generatedRoot_221.w_12_0 ctx)

def parallel_generatedRoot_221.w_14_0  (ctx : parallel_generatedRoot_221Ctx) : mxxMat1 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_221.w_15_0  (ctx : parallel_generatedRoot_221Ctx) : mxxMat1 :=
  ctx.witness.w_15_0

def parallel_generatedRoot_221.w_16_0  (ctx : parallel_generatedRoot_221Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_221.w_14_0 ctx) (parallel_generatedRoot_221.w_15_0 ctx)

def parallel_generatedRoot_221.w_17_0  (ctx : parallel_generatedRoot_221Ctx) : mxxMat1 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_221.w_17_1  (ctx : parallel_generatedRoot_221Ctx) : mxxMat1 :=
  ctx.witness.w_17_1

def parallel_generatedRoot_221.w_17_2  (ctx : parallel_generatedRoot_221Ctx) : mxxMat1 :=
  ctx.witness.w_17_2

def parallel_generatedRoot_221.w_17_3  (ctx : parallel_generatedRoot_221Ctx) : mxxMat1 :=
  ctx.witness.w_17_3

def parallel_generatedRoot_221.outs  (ctx : parallel_generatedRoot_221Ctx) : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit :=
  ((parallel_generatedRoot_221.w_17_0 ctx), ((parallel_generatedRoot_221.w_17_1 ctx), ((parallel_generatedRoot_221.w_17_2 ctx), ((parallel_generatedRoot_221.w_17_3 ctx), ()))))

structure parallel_generatedRoot_221.Facts_0  (ctx : parallel_generatedRoot_221Ctx) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  t0 : (130) = 130
  h4_3 : (∀ i : Fin 130, parallel_parallel_generatedRoot_221_4 backend hashModel tape (path ++ [4, i.val]) params i_0 i ((parallel_generatedRoot_221.w_0_0 ctx), ((parallel_generatedRoot_221.w_1_0 ctx), ((parallel_generatedRoot_221.w_2_0 ctx), ((parallel_generatedRoot_221.w_3_0 ctx), ())))) (((parallel_generatedRoot_221.w_4_0 ctx) i), (((parallel_generatedRoot_221.w_4_1 ctx) i), (((parallel_generatedRoot_221.w_4_2 ctx) i), (((parallel_generatedRoot_221.w_4_3 ctx) i), ())))))
  h5_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_221.w_4_0 ctx) (0) (parallel_generatedRoot_221.w_5_0 ctx)
  h6_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_221.w_4_0 ctx) (1) (parallel_generatedRoot_221.w_6_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_221.w_4_1 ctx) (0) (parallel_generatedRoot_221.w_8_0 ctx)
  h9_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_221.w_4_1 ctx) (1) (parallel_generatedRoot_221.w_9_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_221.w_4_2 ctx) (0) (parallel_generatedRoot_221.w_11_0 ctx)
  h12_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_221.w_4_2 ctx) (1) (parallel_generatedRoot_221.w_12_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_221.w_4_3 ctx) (0) (parallel_generatedRoot_221.w_14_0 ctx)
  h15_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_221.w_4_3 ctx) (1) (parallel_generatedRoot_221.w_15_0 ctx)
  t10 : 0 ≤ (128)
  h17_3 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_parallel_generatedRoot_221_17 backend hashModel tape (path ++ [17, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, ((parallel_generatedRoot_221.w_4_0 ctx), ((parallel_generatedRoot_221.w_4_1 ctx), ((parallel_generatedRoot_221.w_4_2 ctx), ((parallel_generatedRoot_221.w_4_3 ctx), ())))))))) next) (Int.toNat (128)) ((parallel_generatedRoot_221.w_7_0 ctx), ((parallel_generatedRoot_221.w_10_0 ctx), ((parallel_generatedRoot_221.w_13_0 ctx), ((parallel_generatedRoot_221.w_16_0 ctx), ())))) ((parallel_generatedRoot_221.w_17_0 ctx), ((parallel_generatedRoot_221.w_17_1 ctx), ((parallel_generatedRoot_221.w_17_2 ctx), ((parallel_generatedRoot_221.w_17_3 ctx), ()))))
  hout : outputs = parallel_generatedRoot_221.outs ctx

theorem parallel_generatedRoot_221.facts_0  {ctx : parallel_generatedRoot_221Ctx} {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_221.constraints_0 backend hashModel tape path params i_0 (parallel_generatedRoot_221.w_0_0 ctx) (parallel_generatedRoot_221.w_1_0 ctx) (parallel_generatedRoot_221.w_2_0 ctx) (parallel_generatedRoot_221.w_3_0 ctx) (parallel_generatedRoot_221.w_4_0 ctx) (parallel_generatedRoot_221.w_4_1 ctx) (parallel_generatedRoot_221.w_4_2 ctx) (parallel_generatedRoot_221.w_4_3 ctx) (parallel_generatedRoot_221.w_5_0 ctx) (parallel_generatedRoot_221.w_6_0 ctx) (parallel_generatedRoot_221.w_7_0 ctx) (parallel_generatedRoot_221.w_8_0 ctx) (parallel_generatedRoot_221.w_9_0 ctx) (parallel_generatedRoot_221.w_10_0 ctx) (parallel_generatedRoot_221.w_11_0 ctx) (parallel_generatedRoot_221.w_12_0 ctx) (parallel_generatedRoot_221.w_13_0 ctx) (parallel_generatedRoot_221.w_14_0 ctx) (parallel_generatedRoot_221.w_15_0 ctx) (parallel_generatedRoot_221.w_16_0 ctx) (parallel_generatedRoot_221.w_17_0 ctx) (parallel_generatedRoot_221.w_17_1 ctx) (parallel_generatedRoot_221.w_17_2 ctx) (parallel_generatedRoot_221.w_17_3 ctx) outputs) :
    parallel_generatedRoot_221.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_221.Facts  (ctx : parallel_generatedRoot_221Ctx) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  c0 : parallel_generatedRoot_221.Facts_0 ctx outputs

theorem parallel_generatedRoot_221.facts  (ctx : parallel_generatedRoot_221Ctx) {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_221.body backend hashModel tape path params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_221.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_221.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_225.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) where
  inputs : (Fin 2607022080 → mxxMat1) × (Fin 2607022080 → mxxMat1) × (Fin 2607022080 → mxxMat1) × (Fin 2607022080 → mxxMat1) × Unit
  witness : parallel_generatedRoot_225.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_225Ctx" => (parallel_generatedRoot_225.Ctx backend hashModel tape path params i_0)

def parallel_generatedRoot_225.w_0_0  (ctx : parallel_generatedRoot_225Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.1

def parallel_generatedRoot_225.w_1_0  (ctx : parallel_generatedRoot_225Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.2.1

def parallel_generatedRoot_225.w_2_0  (ctx : parallel_generatedRoot_225Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_225.w_3_0  (ctx : parallel_generatedRoot_225Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_225.w_4_0  (ctx : parallel_generatedRoot_225Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_0

def parallel_generatedRoot_225.w_4_1  (ctx : parallel_generatedRoot_225Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_1

def parallel_generatedRoot_225.w_4_2  (ctx : parallel_generatedRoot_225Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_2

def parallel_generatedRoot_225.w_4_3  (ctx : parallel_generatedRoot_225Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_3

def parallel_generatedRoot_225.w_5_0  (ctx : parallel_generatedRoot_225Ctx) : mxxMat1 :=
  ctx.witness.w_5_0

def parallel_generatedRoot_225.w_6_0  (ctx : parallel_generatedRoot_225Ctx) : mxxMat1 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_225.w_7_0  (ctx : parallel_generatedRoot_225Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_225.w_5_0 ctx) (parallel_generatedRoot_225.w_6_0 ctx)

def parallel_generatedRoot_225.w_8_0  (ctx : parallel_generatedRoot_225Ctx) : mxxMat1 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_225.w_9_0  (ctx : parallel_generatedRoot_225Ctx) : mxxMat1 :=
  ctx.witness.w_9_0

def parallel_generatedRoot_225.w_10_0  (ctx : parallel_generatedRoot_225Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_225.w_8_0 ctx) (parallel_generatedRoot_225.w_9_0 ctx)

def parallel_generatedRoot_225.w_11_0  (ctx : parallel_generatedRoot_225Ctx) : mxxMat1 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_225.w_12_0  (ctx : parallel_generatedRoot_225Ctx) : mxxMat1 :=
  ctx.witness.w_12_0

def parallel_generatedRoot_225.w_13_0  (ctx : parallel_generatedRoot_225Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_225.w_11_0 ctx) (parallel_generatedRoot_225.w_12_0 ctx)

def parallel_generatedRoot_225.w_14_0  (ctx : parallel_generatedRoot_225Ctx) : mxxMat1 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_225.w_15_0  (ctx : parallel_generatedRoot_225Ctx) : mxxMat1 :=
  ctx.witness.w_15_0

def parallel_generatedRoot_225.w_16_0  (ctx : parallel_generatedRoot_225Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_225.w_14_0 ctx) (parallel_generatedRoot_225.w_15_0 ctx)

def parallel_generatedRoot_225.w_17_0  (ctx : parallel_generatedRoot_225Ctx) : mxxMat1 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_225.w_17_1  (ctx : parallel_generatedRoot_225Ctx) : mxxMat1 :=
  ctx.witness.w_17_1

def parallel_generatedRoot_225.w_17_2  (ctx : parallel_generatedRoot_225Ctx) : mxxMat1 :=
  ctx.witness.w_17_2

def parallel_generatedRoot_225.w_17_3  (ctx : parallel_generatedRoot_225Ctx) : mxxMat1 :=
  ctx.witness.w_17_3

def parallel_generatedRoot_225.outs  (ctx : parallel_generatedRoot_225Ctx) : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit :=
  ((parallel_generatedRoot_225.w_17_0 ctx), ((parallel_generatedRoot_225.w_17_1 ctx), ((parallel_generatedRoot_225.w_17_2 ctx), ((parallel_generatedRoot_225.w_17_3 ctx), ()))))

structure parallel_generatedRoot_225.Facts_0  (ctx : parallel_generatedRoot_225Ctx) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  t0 : (130) = 130
  h4_3 : (∀ i : Fin 130, parallel_parallel_generatedRoot_225_4 backend hashModel tape (path ++ [4, i.val]) params i_0 i ((parallel_generatedRoot_225.w_0_0 ctx), ((parallel_generatedRoot_225.w_1_0 ctx), ((parallel_generatedRoot_225.w_2_0 ctx), ((parallel_generatedRoot_225.w_3_0 ctx), ())))) (((parallel_generatedRoot_225.w_4_0 ctx) i), (((parallel_generatedRoot_225.w_4_1 ctx) i), (((parallel_generatedRoot_225.w_4_2 ctx) i), (((parallel_generatedRoot_225.w_4_3 ctx) i), ())))))
  h5_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_225.w_4_0 ctx) (0) (parallel_generatedRoot_225.w_5_0 ctx)
  h6_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_225.w_4_0 ctx) (1) (parallel_generatedRoot_225.w_6_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_225.w_4_1 ctx) (0) (parallel_generatedRoot_225.w_8_0 ctx)
  h9_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_225.w_4_1 ctx) (1) (parallel_generatedRoot_225.w_9_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_225.w_4_2 ctx) (0) (parallel_generatedRoot_225.w_11_0 ctx)
  h12_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_225.w_4_2 ctx) (1) (parallel_generatedRoot_225.w_12_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_225.w_4_3 ctx) (0) (parallel_generatedRoot_225.w_14_0 ctx)
  h15_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_225.w_4_3 ctx) (1) (parallel_generatedRoot_225.w_15_0 ctx)
  t10 : 0 ≤ (128)
  h17_3 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_parallel_generatedRoot_225_17 backend hashModel tape (path ++ [17, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, ((parallel_generatedRoot_225.w_4_0 ctx), ((parallel_generatedRoot_225.w_4_1 ctx), ((parallel_generatedRoot_225.w_4_2 ctx), ((parallel_generatedRoot_225.w_4_3 ctx), ())))))))) next) (Int.toNat (128)) ((parallel_generatedRoot_225.w_7_0 ctx), ((parallel_generatedRoot_225.w_10_0 ctx), ((parallel_generatedRoot_225.w_13_0 ctx), ((parallel_generatedRoot_225.w_16_0 ctx), ())))) ((parallel_generatedRoot_225.w_17_0 ctx), ((parallel_generatedRoot_225.w_17_1 ctx), ((parallel_generatedRoot_225.w_17_2 ctx), ((parallel_generatedRoot_225.w_17_3 ctx), ()))))
  hout : outputs = parallel_generatedRoot_225.outs ctx

theorem parallel_generatedRoot_225.facts_0  {ctx : parallel_generatedRoot_225Ctx} {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_225.constraints_0 backend hashModel tape path params i_0 (parallel_generatedRoot_225.w_0_0 ctx) (parallel_generatedRoot_225.w_1_0 ctx) (parallel_generatedRoot_225.w_2_0 ctx) (parallel_generatedRoot_225.w_3_0 ctx) (parallel_generatedRoot_225.w_4_0 ctx) (parallel_generatedRoot_225.w_4_1 ctx) (parallel_generatedRoot_225.w_4_2 ctx) (parallel_generatedRoot_225.w_4_3 ctx) (parallel_generatedRoot_225.w_5_0 ctx) (parallel_generatedRoot_225.w_6_0 ctx) (parallel_generatedRoot_225.w_7_0 ctx) (parallel_generatedRoot_225.w_8_0 ctx) (parallel_generatedRoot_225.w_9_0 ctx) (parallel_generatedRoot_225.w_10_0 ctx) (parallel_generatedRoot_225.w_11_0 ctx) (parallel_generatedRoot_225.w_12_0 ctx) (parallel_generatedRoot_225.w_13_0 ctx) (parallel_generatedRoot_225.w_14_0 ctx) (parallel_generatedRoot_225.w_15_0 ctx) (parallel_generatedRoot_225.w_16_0 ctx) (parallel_generatedRoot_225.w_17_0 ctx) (parallel_generatedRoot_225.w_17_1 ctx) (parallel_generatedRoot_225.w_17_2 ctx) (parallel_generatedRoot_225.w_17_3 ctx) outputs) :
    parallel_generatedRoot_225.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_225.Facts  (ctx : parallel_generatedRoot_225Ctx) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  c0 : parallel_generatedRoot_225.Facts_0 ctx outputs

theorem parallel_generatedRoot_225.facts  (ctx : parallel_generatedRoot_225Ctx) {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_225.body backend hashModel tape path params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_225.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_225.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_229.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) where
  inputs : (Fin 2607022080 → mxxMat1) × (Fin 2607022080 → mxxMat1) × (Fin 2607022080 → mxxMat1) × (Fin 2607022080 → mxxMat1) × Unit
  witness : parallel_generatedRoot_229.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_229Ctx" => (parallel_generatedRoot_229.Ctx backend hashModel tape path params i_0)

def parallel_generatedRoot_229.w_0_0  (ctx : parallel_generatedRoot_229Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.1

def parallel_generatedRoot_229.w_1_0  (ctx : parallel_generatedRoot_229Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.2.1

def parallel_generatedRoot_229.w_2_0  (ctx : parallel_generatedRoot_229Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_229.w_3_0  (ctx : parallel_generatedRoot_229Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_229.w_4_0  (ctx : parallel_generatedRoot_229Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_0

def parallel_generatedRoot_229.w_4_1  (ctx : parallel_generatedRoot_229Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_1

def parallel_generatedRoot_229.w_4_2  (ctx : parallel_generatedRoot_229Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_2

def parallel_generatedRoot_229.w_4_3  (ctx : parallel_generatedRoot_229Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_3

def parallel_generatedRoot_229.w_5_0  (ctx : parallel_generatedRoot_229Ctx) : mxxMat1 :=
  ctx.witness.w_5_0

def parallel_generatedRoot_229.w_6_0  (ctx : parallel_generatedRoot_229Ctx) : mxxMat1 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_229.w_7_0  (ctx : parallel_generatedRoot_229Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_229.w_5_0 ctx) (parallel_generatedRoot_229.w_6_0 ctx)

def parallel_generatedRoot_229.w_8_0  (ctx : parallel_generatedRoot_229Ctx) : mxxMat1 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_229.w_9_0  (ctx : parallel_generatedRoot_229Ctx) : mxxMat1 :=
  ctx.witness.w_9_0

def parallel_generatedRoot_229.w_10_0  (ctx : parallel_generatedRoot_229Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_229.w_8_0 ctx) (parallel_generatedRoot_229.w_9_0 ctx)

def parallel_generatedRoot_229.w_11_0  (ctx : parallel_generatedRoot_229Ctx) : mxxMat1 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_229.w_12_0  (ctx : parallel_generatedRoot_229Ctx) : mxxMat1 :=
  ctx.witness.w_12_0

def parallel_generatedRoot_229.w_13_0  (ctx : parallel_generatedRoot_229Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_229.w_11_0 ctx) (parallel_generatedRoot_229.w_12_0 ctx)

def parallel_generatedRoot_229.w_14_0  (ctx : parallel_generatedRoot_229Ctx) : mxxMat1 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_229.w_15_0  (ctx : parallel_generatedRoot_229Ctx) : mxxMat1 :=
  ctx.witness.w_15_0

def parallel_generatedRoot_229.w_16_0  (ctx : parallel_generatedRoot_229Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_229.w_14_0 ctx) (parallel_generatedRoot_229.w_15_0 ctx)

def parallel_generatedRoot_229.w_17_0  (ctx : parallel_generatedRoot_229Ctx) : mxxMat1 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_229.w_17_1  (ctx : parallel_generatedRoot_229Ctx) : mxxMat1 :=
  ctx.witness.w_17_1

def parallel_generatedRoot_229.w_17_2  (ctx : parallel_generatedRoot_229Ctx) : mxxMat1 :=
  ctx.witness.w_17_2

def parallel_generatedRoot_229.w_17_3  (ctx : parallel_generatedRoot_229Ctx) : mxxMat1 :=
  ctx.witness.w_17_3

def parallel_generatedRoot_229.outs  (ctx : parallel_generatedRoot_229Ctx) : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit :=
  ((parallel_generatedRoot_229.w_17_0 ctx), ((parallel_generatedRoot_229.w_17_1 ctx), ((parallel_generatedRoot_229.w_17_2 ctx), ((parallel_generatedRoot_229.w_17_3 ctx), ()))))

structure parallel_generatedRoot_229.Facts_0  (ctx : parallel_generatedRoot_229Ctx) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  t0 : (130) = 130
  h4_3 : (∀ i : Fin 130, parallel_parallel_generatedRoot_229_4 backend hashModel tape (path ++ [4, i.val]) params i_0 i ((parallel_generatedRoot_229.w_0_0 ctx), ((parallel_generatedRoot_229.w_1_0 ctx), ((parallel_generatedRoot_229.w_2_0 ctx), ((parallel_generatedRoot_229.w_3_0 ctx), ())))) (((parallel_generatedRoot_229.w_4_0 ctx) i), (((parallel_generatedRoot_229.w_4_1 ctx) i), (((parallel_generatedRoot_229.w_4_2 ctx) i), (((parallel_generatedRoot_229.w_4_3 ctx) i), ())))))
  h5_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_229.w_4_0 ctx) (0) (parallel_generatedRoot_229.w_5_0 ctx)
  h6_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_229.w_4_0 ctx) (1) (parallel_generatedRoot_229.w_6_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_229.w_4_1 ctx) (0) (parallel_generatedRoot_229.w_8_0 ctx)
  h9_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_229.w_4_1 ctx) (1) (parallel_generatedRoot_229.w_9_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_229.w_4_2 ctx) (0) (parallel_generatedRoot_229.w_11_0 ctx)
  h12_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_229.w_4_2 ctx) (1) (parallel_generatedRoot_229.w_12_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_229.w_4_3 ctx) (0) (parallel_generatedRoot_229.w_14_0 ctx)
  h15_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_229.w_4_3 ctx) (1) (parallel_generatedRoot_229.w_15_0 ctx)
  t10 : 0 ≤ (128)
  h17_3 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_parallel_generatedRoot_229_17 backend hashModel tape (path ++ [17, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, ((parallel_generatedRoot_229.w_4_0 ctx), ((parallel_generatedRoot_229.w_4_1 ctx), ((parallel_generatedRoot_229.w_4_2 ctx), ((parallel_generatedRoot_229.w_4_3 ctx), ())))))))) next) (Int.toNat (128)) ((parallel_generatedRoot_229.w_7_0 ctx), ((parallel_generatedRoot_229.w_10_0 ctx), ((parallel_generatedRoot_229.w_13_0 ctx), ((parallel_generatedRoot_229.w_16_0 ctx), ())))) ((parallel_generatedRoot_229.w_17_0 ctx), ((parallel_generatedRoot_229.w_17_1 ctx), ((parallel_generatedRoot_229.w_17_2 ctx), ((parallel_generatedRoot_229.w_17_3 ctx), ()))))
  hout : outputs = parallel_generatedRoot_229.outs ctx

theorem parallel_generatedRoot_229.facts_0  {ctx : parallel_generatedRoot_229Ctx} {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_229.constraints_0 backend hashModel tape path params i_0 (parallel_generatedRoot_229.w_0_0 ctx) (parallel_generatedRoot_229.w_1_0 ctx) (parallel_generatedRoot_229.w_2_0 ctx) (parallel_generatedRoot_229.w_3_0 ctx) (parallel_generatedRoot_229.w_4_0 ctx) (parallel_generatedRoot_229.w_4_1 ctx) (parallel_generatedRoot_229.w_4_2 ctx) (parallel_generatedRoot_229.w_4_3 ctx) (parallel_generatedRoot_229.w_5_0 ctx) (parallel_generatedRoot_229.w_6_0 ctx) (parallel_generatedRoot_229.w_7_0 ctx) (parallel_generatedRoot_229.w_8_0 ctx) (parallel_generatedRoot_229.w_9_0 ctx) (parallel_generatedRoot_229.w_10_0 ctx) (parallel_generatedRoot_229.w_11_0 ctx) (parallel_generatedRoot_229.w_12_0 ctx) (parallel_generatedRoot_229.w_13_0 ctx) (parallel_generatedRoot_229.w_14_0 ctx) (parallel_generatedRoot_229.w_15_0 ctx) (parallel_generatedRoot_229.w_16_0 ctx) (parallel_generatedRoot_229.w_17_0 ctx) (parallel_generatedRoot_229.w_17_1 ctx) (parallel_generatedRoot_229.w_17_2 ctx) (parallel_generatedRoot_229.w_17_3 ctx) outputs) :
    parallel_generatedRoot_229.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_229.Facts  (ctx : parallel_generatedRoot_229Ctx) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  c0 : parallel_generatedRoot_229.Facts_0 ctx outputs

theorem parallel_generatedRoot_229.facts  (ctx : parallel_generatedRoot_229Ctx) {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_229.body backend hashModel tape path params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_229.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_229.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_233.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) where
  inputs : (Fin 2607022080 → mxxMat1) × (Fin 2607022080 → mxxMat1) × (Fin 2607022080 → mxxMat1) × (Fin 2607022080 → mxxMat1) × Unit
  witness : parallel_generatedRoot_233.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_233Ctx" => (parallel_generatedRoot_233.Ctx backend hashModel tape path params i_0)

def parallel_generatedRoot_233.w_0_0  (ctx : parallel_generatedRoot_233Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.1

def parallel_generatedRoot_233.w_1_0  (ctx : parallel_generatedRoot_233Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.2.1

def parallel_generatedRoot_233.w_2_0  (ctx : parallel_generatedRoot_233Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_233.w_3_0  (ctx : parallel_generatedRoot_233Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_233.w_4_0  (ctx : parallel_generatedRoot_233Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_0

def parallel_generatedRoot_233.w_4_1  (ctx : parallel_generatedRoot_233Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_1

def parallel_generatedRoot_233.w_4_2  (ctx : parallel_generatedRoot_233Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_2

def parallel_generatedRoot_233.w_4_3  (ctx : parallel_generatedRoot_233Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_3

def parallel_generatedRoot_233.w_5_0  (ctx : parallel_generatedRoot_233Ctx) : mxxMat1 :=
  ctx.witness.w_5_0

def parallel_generatedRoot_233.w_6_0  (ctx : parallel_generatedRoot_233Ctx) : mxxMat1 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_233.w_7_0  (ctx : parallel_generatedRoot_233Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_233.w_5_0 ctx) (parallel_generatedRoot_233.w_6_0 ctx)

def parallel_generatedRoot_233.w_8_0  (ctx : parallel_generatedRoot_233Ctx) : mxxMat1 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_233.w_9_0  (ctx : parallel_generatedRoot_233Ctx) : mxxMat1 :=
  ctx.witness.w_9_0

def parallel_generatedRoot_233.w_10_0  (ctx : parallel_generatedRoot_233Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_233.w_8_0 ctx) (parallel_generatedRoot_233.w_9_0 ctx)

def parallel_generatedRoot_233.w_11_0  (ctx : parallel_generatedRoot_233Ctx) : mxxMat1 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_233.w_12_0  (ctx : parallel_generatedRoot_233Ctx) : mxxMat1 :=
  ctx.witness.w_12_0

def parallel_generatedRoot_233.w_13_0  (ctx : parallel_generatedRoot_233Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_233.w_11_0 ctx) (parallel_generatedRoot_233.w_12_0 ctx)

def parallel_generatedRoot_233.w_14_0  (ctx : parallel_generatedRoot_233Ctx) : mxxMat1 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_233.w_15_0  (ctx : parallel_generatedRoot_233Ctx) : mxxMat1 :=
  ctx.witness.w_15_0

def parallel_generatedRoot_233.w_16_0  (ctx : parallel_generatedRoot_233Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_233.w_14_0 ctx) (parallel_generatedRoot_233.w_15_0 ctx)

def parallel_generatedRoot_233.w_17_0  (ctx : parallel_generatedRoot_233Ctx) : mxxMat1 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_233.w_17_1  (ctx : parallel_generatedRoot_233Ctx) : mxxMat1 :=
  ctx.witness.w_17_1

def parallel_generatedRoot_233.w_17_2  (ctx : parallel_generatedRoot_233Ctx) : mxxMat1 :=
  ctx.witness.w_17_2

def parallel_generatedRoot_233.w_17_3  (ctx : parallel_generatedRoot_233Ctx) : mxxMat1 :=
  ctx.witness.w_17_3

def parallel_generatedRoot_233.outs  (ctx : parallel_generatedRoot_233Ctx) : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit :=
  ((parallel_generatedRoot_233.w_17_0 ctx), ((parallel_generatedRoot_233.w_17_1 ctx), ((parallel_generatedRoot_233.w_17_2 ctx), ((parallel_generatedRoot_233.w_17_3 ctx), ()))))

structure parallel_generatedRoot_233.Facts_0  (ctx : parallel_generatedRoot_233Ctx) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  t0 : (130) = 130
  h4_3 : (∀ i : Fin 130, parallel_parallel_generatedRoot_233_4 backend hashModel tape (path ++ [4, i.val]) params i_0 i ((parallel_generatedRoot_233.w_0_0 ctx), ((parallel_generatedRoot_233.w_1_0 ctx), ((parallel_generatedRoot_233.w_2_0 ctx), ((parallel_generatedRoot_233.w_3_0 ctx), ())))) (((parallel_generatedRoot_233.w_4_0 ctx) i), (((parallel_generatedRoot_233.w_4_1 ctx) i), (((parallel_generatedRoot_233.w_4_2 ctx) i), (((parallel_generatedRoot_233.w_4_3 ctx) i), ())))))
  h5_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_233.w_4_0 ctx) (0) (parallel_generatedRoot_233.w_5_0 ctx)
  h6_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_233.w_4_0 ctx) (1) (parallel_generatedRoot_233.w_6_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_233.w_4_1 ctx) (0) (parallel_generatedRoot_233.w_8_0 ctx)
  h9_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_233.w_4_1 ctx) (1) (parallel_generatedRoot_233.w_9_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_233.w_4_2 ctx) (0) (parallel_generatedRoot_233.w_11_0 ctx)
  h12_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_233.w_4_2 ctx) (1) (parallel_generatedRoot_233.w_12_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_233.w_4_3 ctx) (0) (parallel_generatedRoot_233.w_14_0 ctx)
  h15_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_233.w_4_3 ctx) (1) (parallel_generatedRoot_233.w_15_0 ctx)
  t10 : 0 ≤ (128)
  h17_3 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_parallel_generatedRoot_233_17 backend hashModel tape (path ++ [17, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, ((parallel_generatedRoot_233.w_4_0 ctx), ((parallel_generatedRoot_233.w_4_1 ctx), ((parallel_generatedRoot_233.w_4_2 ctx), ((parallel_generatedRoot_233.w_4_3 ctx), ())))))))) next) (Int.toNat (128)) ((parallel_generatedRoot_233.w_7_0 ctx), ((parallel_generatedRoot_233.w_10_0 ctx), ((parallel_generatedRoot_233.w_13_0 ctx), ((parallel_generatedRoot_233.w_16_0 ctx), ())))) ((parallel_generatedRoot_233.w_17_0 ctx), ((parallel_generatedRoot_233.w_17_1 ctx), ((parallel_generatedRoot_233.w_17_2 ctx), ((parallel_generatedRoot_233.w_17_3 ctx), ()))))
  hout : outputs = parallel_generatedRoot_233.outs ctx

theorem parallel_generatedRoot_233.facts_0  {ctx : parallel_generatedRoot_233Ctx} {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_233.constraints_0 backend hashModel tape path params i_0 (parallel_generatedRoot_233.w_0_0 ctx) (parallel_generatedRoot_233.w_1_0 ctx) (parallel_generatedRoot_233.w_2_0 ctx) (parallel_generatedRoot_233.w_3_0 ctx) (parallel_generatedRoot_233.w_4_0 ctx) (parallel_generatedRoot_233.w_4_1 ctx) (parallel_generatedRoot_233.w_4_2 ctx) (parallel_generatedRoot_233.w_4_3 ctx) (parallel_generatedRoot_233.w_5_0 ctx) (parallel_generatedRoot_233.w_6_0 ctx) (parallel_generatedRoot_233.w_7_0 ctx) (parallel_generatedRoot_233.w_8_0 ctx) (parallel_generatedRoot_233.w_9_0 ctx) (parallel_generatedRoot_233.w_10_0 ctx) (parallel_generatedRoot_233.w_11_0 ctx) (parallel_generatedRoot_233.w_12_0 ctx) (parallel_generatedRoot_233.w_13_0 ctx) (parallel_generatedRoot_233.w_14_0 ctx) (parallel_generatedRoot_233.w_15_0 ctx) (parallel_generatedRoot_233.w_16_0 ctx) (parallel_generatedRoot_233.w_17_0 ctx) (parallel_generatedRoot_233.w_17_1 ctx) (parallel_generatedRoot_233.w_17_2 ctx) (parallel_generatedRoot_233.w_17_3 ctx) outputs) :
    parallel_generatedRoot_233.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_233.Facts  (ctx : parallel_generatedRoot_233Ctx) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  c0 : parallel_generatedRoot_233.Facts_0 ctx outputs

theorem parallel_generatedRoot_233.facts  (ctx : parallel_generatedRoot_233Ctx) {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_233.body backend hashModel tape path params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_233.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_233.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_237.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) where
  inputs : (Fin 2607022080 → mxxMat1) × (Fin 2607022080 → mxxMat1) × (Fin 2607022080 → mxxMat1) × (Fin 2607022080 → mxxMat1) × Unit
  witness : parallel_generatedRoot_237.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_237Ctx" => (parallel_generatedRoot_237.Ctx backend hashModel tape path params i_0)

def parallel_generatedRoot_237.w_0_0  (ctx : parallel_generatedRoot_237Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.1

def parallel_generatedRoot_237.w_1_0  (ctx : parallel_generatedRoot_237Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.2.1

def parallel_generatedRoot_237.w_2_0  (ctx : parallel_generatedRoot_237Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_237.w_3_0  (ctx : parallel_generatedRoot_237Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_237.w_4_0  (ctx : parallel_generatedRoot_237Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_0

def parallel_generatedRoot_237.w_4_1  (ctx : parallel_generatedRoot_237Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_1

def parallel_generatedRoot_237.w_4_2  (ctx : parallel_generatedRoot_237Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_2

def parallel_generatedRoot_237.w_4_3  (ctx : parallel_generatedRoot_237Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_3

def parallel_generatedRoot_237.w_5_0  (ctx : parallel_generatedRoot_237Ctx) : mxxMat1 :=
  ctx.witness.w_5_0

def parallel_generatedRoot_237.w_6_0  (ctx : parallel_generatedRoot_237Ctx) : mxxMat1 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_237.w_7_0  (ctx : parallel_generatedRoot_237Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_237.w_5_0 ctx) (parallel_generatedRoot_237.w_6_0 ctx)

def parallel_generatedRoot_237.w_8_0  (ctx : parallel_generatedRoot_237Ctx) : mxxMat1 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_237.w_9_0  (ctx : parallel_generatedRoot_237Ctx) : mxxMat1 :=
  ctx.witness.w_9_0

def parallel_generatedRoot_237.w_10_0  (ctx : parallel_generatedRoot_237Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_237.w_8_0 ctx) (parallel_generatedRoot_237.w_9_0 ctx)

def parallel_generatedRoot_237.w_11_0  (ctx : parallel_generatedRoot_237Ctx) : mxxMat1 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_237.w_12_0  (ctx : parallel_generatedRoot_237Ctx) : mxxMat1 :=
  ctx.witness.w_12_0

def parallel_generatedRoot_237.w_13_0  (ctx : parallel_generatedRoot_237Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_237.w_11_0 ctx) (parallel_generatedRoot_237.w_12_0 ctx)

def parallel_generatedRoot_237.w_14_0  (ctx : parallel_generatedRoot_237Ctx) : mxxMat1 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_237.w_15_0  (ctx : parallel_generatedRoot_237Ctx) : mxxMat1 :=
  ctx.witness.w_15_0

def parallel_generatedRoot_237.w_16_0  (ctx : parallel_generatedRoot_237Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_237.w_14_0 ctx) (parallel_generatedRoot_237.w_15_0 ctx)

def parallel_generatedRoot_237.w_17_0  (ctx : parallel_generatedRoot_237Ctx) : mxxMat1 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_237.w_17_1  (ctx : parallel_generatedRoot_237Ctx) : mxxMat1 :=
  ctx.witness.w_17_1

def parallel_generatedRoot_237.w_17_2  (ctx : parallel_generatedRoot_237Ctx) : mxxMat1 :=
  ctx.witness.w_17_2

def parallel_generatedRoot_237.w_17_3  (ctx : parallel_generatedRoot_237Ctx) : mxxMat1 :=
  ctx.witness.w_17_3

def parallel_generatedRoot_237.outs  (ctx : parallel_generatedRoot_237Ctx) : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit :=
  ((parallel_generatedRoot_237.w_17_0 ctx), ((parallel_generatedRoot_237.w_17_1 ctx), ((parallel_generatedRoot_237.w_17_2 ctx), ((parallel_generatedRoot_237.w_17_3 ctx), ()))))

structure parallel_generatedRoot_237.Facts_0  (ctx : parallel_generatedRoot_237Ctx) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  t0 : (130) = 130
  h4_3 : (∀ i : Fin 130, parallel_parallel_generatedRoot_237_4 backend hashModel tape (path ++ [4, i.val]) params i_0 i ((parallel_generatedRoot_237.w_0_0 ctx), ((parallel_generatedRoot_237.w_1_0 ctx), ((parallel_generatedRoot_237.w_2_0 ctx), ((parallel_generatedRoot_237.w_3_0 ctx), ())))) (((parallel_generatedRoot_237.w_4_0 ctx) i), (((parallel_generatedRoot_237.w_4_1 ctx) i), (((parallel_generatedRoot_237.w_4_2 ctx) i), (((parallel_generatedRoot_237.w_4_3 ctx) i), ())))))
  h5_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_237.w_4_0 ctx) (0) (parallel_generatedRoot_237.w_5_0 ctx)
  h6_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_237.w_4_0 ctx) (1) (parallel_generatedRoot_237.w_6_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_237.w_4_1 ctx) (0) (parallel_generatedRoot_237.w_8_0 ctx)
  h9_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_237.w_4_1 ctx) (1) (parallel_generatedRoot_237.w_9_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_237.w_4_2 ctx) (0) (parallel_generatedRoot_237.w_11_0 ctx)
  h12_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_237.w_4_2 ctx) (1) (parallel_generatedRoot_237.w_12_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_237.w_4_3 ctx) (0) (parallel_generatedRoot_237.w_14_0 ctx)
  h15_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_237.w_4_3 ctx) (1) (parallel_generatedRoot_237.w_15_0 ctx)
  t10 : 0 ≤ (128)
  h17_3 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_parallel_generatedRoot_237_17 backend hashModel tape (path ++ [17, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, ((parallel_generatedRoot_237.w_4_0 ctx), ((parallel_generatedRoot_237.w_4_1 ctx), ((parallel_generatedRoot_237.w_4_2 ctx), ((parallel_generatedRoot_237.w_4_3 ctx), ())))))))) next) (Int.toNat (128)) ((parallel_generatedRoot_237.w_7_0 ctx), ((parallel_generatedRoot_237.w_10_0 ctx), ((parallel_generatedRoot_237.w_13_0 ctx), ((parallel_generatedRoot_237.w_16_0 ctx), ())))) ((parallel_generatedRoot_237.w_17_0 ctx), ((parallel_generatedRoot_237.w_17_1 ctx), ((parallel_generatedRoot_237.w_17_2 ctx), ((parallel_generatedRoot_237.w_17_3 ctx), ()))))
  hout : outputs = parallel_generatedRoot_237.outs ctx

theorem parallel_generatedRoot_237.facts_0  {ctx : parallel_generatedRoot_237Ctx} {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_237.constraints_0 backend hashModel tape path params i_0 (parallel_generatedRoot_237.w_0_0 ctx) (parallel_generatedRoot_237.w_1_0 ctx) (parallel_generatedRoot_237.w_2_0 ctx) (parallel_generatedRoot_237.w_3_0 ctx) (parallel_generatedRoot_237.w_4_0 ctx) (parallel_generatedRoot_237.w_4_1 ctx) (parallel_generatedRoot_237.w_4_2 ctx) (parallel_generatedRoot_237.w_4_3 ctx) (parallel_generatedRoot_237.w_5_0 ctx) (parallel_generatedRoot_237.w_6_0 ctx) (parallel_generatedRoot_237.w_7_0 ctx) (parallel_generatedRoot_237.w_8_0 ctx) (parallel_generatedRoot_237.w_9_0 ctx) (parallel_generatedRoot_237.w_10_0 ctx) (parallel_generatedRoot_237.w_11_0 ctx) (parallel_generatedRoot_237.w_12_0 ctx) (parallel_generatedRoot_237.w_13_0 ctx) (parallel_generatedRoot_237.w_14_0 ctx) (parallel_generatedRoot_237.w_15_0 ctx) (parallel_generatedRoot_237.w_16_0 ctx) (parallel_generatedRoot_237.w_17_0 ctx) (parallel_generatedRoot_237.w_17_1 ctx) (parallel_generatedRoot_237.w_17_2 ctx) (parallel_generatedRoot_237.w_17_3 ctx) outputs) :
    parallel_generatedRoot_237.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_237.Facts  (ctx : parallel_generatedRoot_237Ctx) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  c0 : parallel_generatedRoot_237.Facts_0 ctx outputs

theorem parallel_generatedRoot_237.facts  (ctx : parallel_generatedRoot_237Ctx) {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_237.body backend hashModel tape path params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_237.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_237.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_241.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) where
  inputs : (Fin 2607022080 → mxxMat1) × (Fin 2607022080 → mxxMat1) × (Fin 2607022080 → mxxMat1) × (Fin 2607022080 → mxxMat1) × Unit
  witness : parallel_generatedRoot_241.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_241Ctx" => (parallel_generatedRoot_241.Ctx backend hashModel tape path params i_0)

def parallel_generatedRoot_241.w_0_0  (ctx : parallel_generatedRoot_241Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.1

def parallel_generatedRoot_241.w_1_0  (ctx : parallel_generatedRoot_241Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.2.1

def parallel_generatedRoot_241.w_2_0  (ctx : parallel_generatedRoot_241Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_241.w_3_0  (ctx : parallel_generatedRoot_241Ctx) : Fin 2607022080 → mxxMat1 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_241.w_4_0  (ctx : parallel_generatedRoot_241Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_0

def parallel_generatedRoot_241.w_4_1  (ctx : parallel_generatedRoot_241Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_1

def parallel_generatedRoot_241.w_4_2  (ctx : parallel_generatedRoot_241Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_2

def parallel_generatedRoot_241.w_4_3  (ctx : parallel_generatedRoot_241Ctx) : Fin 130 → mxxMat1 :=
  ctx.witness.w_4_3

def parallel_generatedRoot_241.w_5_0  (ctx : parallel_generatedRoot_241Ctx) : mxxMat1 :=
  ctx.witness.w_5_0

def parallel_generatedRoot_241.w_6_0  (ctx : parallel_generatedRoot_241Ctx) : mxxMat1 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_241.w_7_0  (ctx : parallel_generatedRoot_241Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_241.w_5_0 ctx) (parallel_generatedRoot_241.w_6_0 ctx)

def parallel_generatedRoot_241.w_8_0  (ctx : parallel_generatedRoot_241Ctx) : mxxMat1 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_241.w_9_0  (ctx : parallel_generatedRoot_241Ctx) : mxxMat1 :=
  ctx.witness.w_9_0

def parallel_generatedRoot_241.w_10_0  (ctx : parallel_generatedRoot_241Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_241.w_8_0 ctx) (parallel_generatedRoot_241.w_9_0 ctx)

def parallel_generatedRoot_241.w_11_0  (ctx : parallel_generatedRoot_241Ctx) : mxxMat1 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_241.w_12_0  (ctx : parallel_generatedRoot_241Ctx) : mxxMat1 :=
  ctx.witness.w_12_0

def parallel_generatedRoot_241.w_13_0  (ctx : parallel_generatedRoot_241Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_241.w_11_0 ctx) (parallel_generatedRoot_241.w_12_0 ctx)

def parallel_generatedRoot_241.w_14_0  (ctx : parallel_generatedRoot_241Ctx) : mxxMat1 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_241.w_15_0  (ctx : parallel_generatedRoot_241Ctx) : mxxMat1 :=
  ctx.witness.w_15_0

def parallel_generatedRoot_241.w_16_0  (ctx : parallel_generatedRoot_241Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_241.w_14_0 ctx) (parallel_generatedRoot_241.w_15_0 ctx)

def parallel_generatedRoot_241.w_17_0  (ctx : parallel_generatedRoot_241Ctx) : mxxMat1 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_241.w_17_1  (ctx : parallel_generatedRoot_241Ctx) : mxxMat1 :=
  ctx.witness.w_17_1

def parallel_generatedRoot_241.w_17_2  (ctx : parallel_generatedRoot_241Ctx) : mxxMat1 :=
  ctx.witness.w_17_2

def parallel_generatedRoot_241.w_17_3  (ctx : parallel_generatedRoot_241Ctx) : mxxMat1 :=
  ctx.witness.w_17_3

def parallel_generatedRoot_241.outs  (ctx : parallel_generatedRoot_241Ctx) : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit :=
  ((parallel_generatedRoot_241.w_17_0 ctx), ((parallel_generatedRoot_241.w_17_1 ctx), ((parallel_generatedRoot_241.w_17_2 ctx), ((parallel_generatedRoot_241.w_17_3 ctx), ()))))

structure parallel_generatedRoot_241.Facts_0  (ctx : parallel_generatedRoot_241Ctx) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  t0 : (130) = 130
  h4_3 : (∀ i : Fin 130, parallel_parallel_generatedRoot_241_4 backend hashModel tape (path ++ [4, i.val]) params i_0 i ((parallel_generatedRoot_241.w_0_0 ctx), ((parallel_generatedRoot_241.w_1_0 ctx), ((parallel_generatedRoot_241.w_2_0 ctx), ((parallel_generatedRoot_241.w_3_0 ctx), ())))) (((parallel_generatedRoot_241.w_4_0 ctx) i), (((parallel_generatedRoot_241.w_4_1 ctx) i), (((parallel_generatedRoot_241.w_4_2 ctx) i), (((parallel_generatedRoot_241.w_4_3 ctx) i), ())))))
  h5_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_241.w_4_0 ctx) (0) (parallel_generatedRoot_241.w_5_0 ctx)
  h6_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_241.w_4_0 ctx) (1) (parallel_generatedRoot_241.w_6_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_241.w_4_1 ctx) (0) (parallel_generatedRoot_241.w_8_0 ctx)
  h9_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_241.w_4_1 ctx) (1) (parallel_generatedRoot_241.w_9_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_241.w_4_2 ctx) (0) (parallel_generatedRoot_241.w_11_0 ctx)
  h12_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_241.w_4_2 ctx) (1) (parallel_generatedRoot_241.w_12_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_241.w_4_3 ctx) (0) (parallel_generatedRoot_241.w_14_0 ctx)
  h15_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_241.w_4_3 ctx) (1) (parallel_generatedRoot_241.w_15_0 ctx)
  t10 : 0 ≤ (128)
  h17_3 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_parallel_generatedRoot_241_17 backend hashModel tape (path ++ [17, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, ((parallel_generatedRoot_241.w_4_0 ctx), ((parallel_generatedRoot_241.w_4_1 ctx), ((parallel_generatedRoot_241.w_4_2 ctx), ((parallel_generatedRoot_241.w_4_3 ctx), ())))))))) next) (Int.toNat (128)) ((parallel_generatedRoot_241.w_7_0 ctx), ((parallel_generatedRoot_241.w_10_0 ctx), ((parallel_generatedRoot_241.w_13_0 ctx), ((parallel_generatedRoot_241.w_16_0 ctx), ())))) ((parallel_generatedRoot_241.w_17_0 ctx), ((parallel_generatedRoot_241.w_17_1 ctx), ((parallel_generatedRoot_241.w_17_2 ctx), ((parallel_generatedRoot_241.w_17_3 ctx), ()))))
  hout : outputs = parallel_generatedRoot_241.outs ctx

theorem parallel_generatedRoot_241.facts_0  {ctx : parallel_generatedRoot_241Ctx} {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_241.constraints_0 backend hashModel tape path params i_0 (parallel_generatedRoot_241.w_0_0 ctx) (parallel_generatedRoot_241.w_1_0 ctx) (parallel_generatedRoot_241.w_2_0 ctx) (parallel_generatedRoot_241.w_3_0 ctx) (parallel_generatedRoot_241.w_4_0 ctx) (parallel_generatedRoot_241.w_4_1 ctx) (parallel_generatedRoot_241.w_4_2 ctx) (parallel_generatedRoot_241.w_4_3 ctx) (parallel_generatedRoot_241.w_5_0 ctx) (parallel_generatedRoot_241.w_6_0 ctx) (parallel_generatedRoot_241.w_7_0 ctx) (parallel_generatedRoot_241.w_8_0 ctx) (parallel_generatedRoot_241.w_9_0 ctx) (parallel_generatedRoot_241.w_10_0 ctx) (parallel_generatedRoot_241.w_11_0 ctx) (parallel_generatedRoot_241.w_12_0 ctx) (parallel_generatedRoot_241.w_13_0 ctx) (parallel_generatedRoot_241.w_14_0 ctx) (parallel_generatedRoot_241.w_15_0 ctx) (parallel_generatedRoot_241.w_16_0 ctx) (parallel_generatedRoot_241.w_17_0 ctx) (parallel_generatedRoot_241.w_17_1 ctx) (parallel_generatedRoot_241.w_17_2 ctx) (parallel_generatedRoot_241.w_17_3 ctx) outputs) :
    parallel_generatedRoot_241.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_241.Facts  (ctx : parallel_generatedRoot_241Ctx) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  c0 : parallel_generatedRoot_241.Facts_0 ctx outputs

theorem parallel_generatedRoot_241.facts  (ctx : parallel_generatedRoot_241Ctx) {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_241.body backend hashModel tape path params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_241.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_241.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_242.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) where
  inputs : ByteArray × mxxMat1 × mxxMat1 × (Fin 128 → mxxMat1) × mxxMat1 × mxxMat1 × (Fin 128 → mxxMat1) × (Fin 128 → mxxMat1) × (Fin 128 → mxxMat1) × Unit
  witness : parallel_generatedRoot_242.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_242Ctx" => (parallel_generatedRoot_242.Ctx backend hashModel tape path params i_0)

def parallel_generatedRoot_242.w_1_0  (ctx : parallel_generatedRoot_242Ctx) : ByteArray :=
  ctx.inputs.1

def parallel_generatedRoot_242.w_2_0  (ctx : parallel_generatedRoot_242Ctx) : mxxMat1 :=
  ctx.inputs.2.1

def parallel_generatedRoot_242.w_3_0  (ctx : parallel_generatedRoot_242Ctx) : mxxMat1 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_242.w_4_0  (ctx : parallel_generatedRoot_242Ctx) : Fin 128 → mxxMat1 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_242.w_5_0  (ctx : parallel_generatedRoot_242Ctx) : mxxMat1 :=
  ctx.inputs.2.2.2.2.1

def parallel_generatedRoot_242.w_6_0  (ctx : parallel_generatedRoot_242Ctx) : mxxMat1 :=
  ctx.inputs.2.2.2.2.2.1

def parallel_generatedRoot_242.w_7_0  (ctx : parallel_generatedRoot_242Ctx) : Fin 128 → mxxMat1 :=
  ctx.inputs.2.2.2.2.2.2.1

def parallel_generatedRoot_242.w_8_0  (ctx : parallel_generatedRoot_242Ctx) : Fin 128 → mxxMat1 :=
  ctx.inputs.2.2.2.2.2.2.2.1

def parallel_generatedRoot_242.w_9_0  (ctx : parallel_generatedRoot_242Ctx) : Fin 128 → mxxMat1 :=
  ctx.inputs.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_242.w_0_0  (ctx : parallel_generatedRoot_242Ctx) : mxxMat1 :=
  (0 : mxxMat1)

def parallel_generatedRoot_242.w_10_0  (ctx : parallel_generatedRoot_242Ctx) : mxxMat1 :=
  ctx.witness.w_10_0

def parallel_generatedRoot_242.w_10_1  (ctx : parallel_generatedRoot_242Ctx) : mxxMat1 :=
  ctx.witness.w_10_1

def parallel_generatedRoot_242.w_10_2  (ctx : parallel_generatedRoot_242Ctx) : mxxMat1 :=
  ctx.witness.w_10_2

def parallel_generatedRoot_242.w_10_3  (ctx : parallel_generatedRoot_242Ctx) : mxxMat1 :=
  ctx.witness.w_10_3

def parallel_generatedRoot_242.outs  (ctx : parallel_generatedRoot_242Ctx) : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit :=
  ((parallel_generatedRoot_242.w_10_0 ctx), ((parallel_generatedRoot_242.w_10_1 ctx), ((parallel_generatedRoot_242.w_10_2 ctx), ((parallel_generatedRoot_242.w_10_3 ctx), ()))))

structure parallel_generatedRoot_242.Facts_0  (ctx : parallel_generatedRoot_242Ctx) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  t0 : 0 ≤ (128)
  h10_3 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_parallel_generatedRoot_242_10 backend hashModel tape (path ++ [10, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, ((parallel_generatedRoot_242.w_1_0 ctx), ((parallel_generatedRoot_242.w_2_0 ctx), ((parallel_generatedRoot_242.w_3_0 ctx), ((parallel_generatedRoot_242.w_4_0 ctx), ((parallel_generatedRoot_242.w_5_0 ctx), ((parallel_generatedRoot_242.w_6_0 ctx), ((parallel_generatedRoot_242.w_7_0 ctx), ((parallel_generatedRoot_242.w_8_0 ctx), ((parallel_generatedRoot_242.w_9_0 ctx), ()))))))))))))) next) (Int.toNat (128)) ((parallel_generatedRoot_242.w_0_0 ctx), ((parallel_generatedRoot_242.w_0_0 ctx), ((parallel_generatedRoot_242.w_0_0 ctx), ((parallel_generatedRoot_242.w_0_0 ctx), ())))) ((parallel_generatedRoot_242.w_10_0 ctx), ((parallel_generatedRoot_242.w_10_1 ctx), ((parallel_generatedRoot_242.w_10_2 ctx), ((parallel_generatedRoot_242.w_10_3 ctx), ()))))
  hout : outputs = parallel_generatedRoot_242.outs ctx

theorem parallel_generatedRoot_242.facts_0  {ctx : parallel_generatedRoot_242Ctx} {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_242.constraints_0 backend hashModel tape path params i_0 (parallel_generatedRoot_242.w_1_0 ctx) (parallel_generatedRoot_242.w_2_0 ctx) (parallel_generatedRoot_242.w_3_0 ctx) (parallel_generatedRoot_242.w_4_0 ctx) (parallel_generatedRoot_242.w_5_0 ctx) (parallel_generatedRoot_242.w_6_0 ctx) (parallel_generatedRoot_242.w_7_0 ctx) (parallel_generatedRoot_242.w_8_0 ctx) (parallel_generatedRoot_242.w_9_0 ctx) (parallel_generatedRoot_242.w_0_0 ctx) (parallel_generatedRoot_242.w_10_0 ctx) (parallel_generatedRoot_242.w_10_1 ctx) (parallel_generatedRoot_242.w_10_2 ctx) (parallel_generatedRoot_242.w_10_3 ctx) outputs) :
    parallel_generatedRoot_242.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure parallel_generatedRoot_242.Facts  (ctx : parallel_generatedRoot_242Ctx) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  c0 : parallel_generatedRoot_242.Facts_0 ctx outputs

theorem parallel_generatedRoot_242.facts  (ctx : parallel_generatedRoot_242Ctx) {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_242.body backend hashModel tape path params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_242.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_242.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_264.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) where
  inputs : mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × (Fin 20054016 → mxxMat1) × (Fin 20054016 → mxxMat1) × (Fin 20054016 → mxxMat1) × (Fin 20054016 → mxxMat1) × Unit
  witness : sequential_generatedRoot_264.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_264Ctx" => (sequential_generatedRoot_264.Ctx backend hashModel tape path params i_0)

def sequential_generatedRoot_264.w_0_0  (ctx : sequential_generatedRoot_264Ctx) : mxxMat2 :=
  ctx.inputs.1

def sequential_generatedRoot_264.w_23_0  (ctx : sequential_generatedRoot_264Ctx) : mxxMat2 :=
  ctx.inputs.2.1

def sequential_generatedRoot_264.w_27_0  (ctx : sequential_generatedRoot_264Ctx) : mxxMat2 :=
  ctx.inputs.2.2.1

def sequential_generatedRoot_264.w_31_0  (ctx : sequential_generatedRoot_264Ctx) : mxxMat2 :=
  ctx.inputs.2.2.2.1

def sequential_generatedRoot_264.w_2_0  (ctx : sequential_generatedRoot_264Ctx) : Fin 20054016 → mxxMat1 :=
  ctx.inputs.2.2.2.2.1

def sequential_generatedRoot_264.w_8_0  (ctx : sequential_generatedRoot_264Ctx) : Fin 20054016 → mxxMat1 :=
  ctx.inputs.2.2.2.2.2.1

def sequential_generatedRoot_264.w_12_0  (ctx : sequential_generatedRoot_264Ctx) : Fin 20054016 → mxxMat1 :=
  ctx.inputs.2.2.2.2.2.2.1

def sequential_generatedRoot_264.w_16_0  (ctx : sequential_generatedRoot_264Ctx) : Fin 20054016 → mxxMat1 :=
  ctx.inputs.2.2.2.2.2.2.2.1

def sequential_generatedRoot_264.w_1_0  (ctx : sequential_generatedRoot_264Ctx) : mxxMat3 :=
  (MxxRuntime.unitRow ((1 + (Int.ofNat i_0))) : mxxMat3)

def sequential_generatedRoot_264.w_3_0  (ctx : sequential_generatedRoot_264Ctx) : Int :=
  (65536 + (65536 * (Int.ofNat i_0)))

def sequential_generatedRoot_264.w_4_0  (ctx : sequential_generatedRoot_264Ctx) : mxxMat1 :=
  ctx.witness.w_4_0

def sequential_generatedRoot_264.w_5_0  (ctx : sequential_generatedRoot_264Ctx) : Int :=
  (65537 + (65536 * (Int.ofNat i_0)))

def sequential_generatedRoot_264.w_6_0  (ctx : sequential_generatedRoot_264Ctx) : mxxMat1 :=
  ctx.witness.w_6_0

def sequential_generatedRoot_264.w_7_0  (ctx : sequential_generatedRoot_264Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (sequential_generatedRoot_264.w_4_0 ctx) (sequential_generatedRoot_264.w_6_0 ctx)

def sequential_generatedRoot_264.w_9_0  (ctx : sequential_generatedRoot_264Ctx) : mxxMat1 :=
  ctx.witness.w_9_0

def sequential_generatedRoot_264.w_10_0  (ctx : sequential_generatedRoot_264Ctx) : mxxMat1 :=
  ctx.witness.w_10_0

def sequential_generatedRoot_264.w_11_0  (ctx : sequential_generatedRoot_264Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (sequential_generatedRoot_264.w_9_0 ctx) (sequential_generatedRoot_264.w_10_0 ctx)

def sequential_generatedRoot_264.w_13_0  (ctx : sequential_generatedRoot_264Ctx) : mxxMat1 :=
  ctx.witness.w_13_0

def sequential_generatedRoot_264.w_14_0  (ctx : sequential_generatedRoot_264Ctx) : mxxMat1 :=
  ctx.witness.w_14_0

def sequential_generatedRoot_264.w_15_0  (ctx : sequential_generatedRoot_264Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (sequential_generatedRoot_264.w_13_0 ctx) (sequential_generatedRoot_264.w_14_0 ctx)

def sequential_generatedRoot_264.w_17_0  (ctx : sequential_generatedRoot_264Ctx) : mxxMat1 :=
  ctx.witness.w_17_0

def sequential_generatedRoot_264.w_18_0  (ctx : sequential_generatedRoot_264Ctx) : mxxMat1 :=
  ctx.witness.w_18_0

def sequential_generatedRoot_264.w_19_0  (ctx : sequential_generatedRoot_264Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (sequential_generatedRoot_264.w_17_0 ctx) (sequential_generatedRoot_264.w_18_0 ctx)

def sequential_generatedRoot_264.w_20_0  (ctx : sequential_generatedRoot_264Ctx) : mxxMat1 :=
  ctx.witness.w_20_0

def sequential_generatedRoot_264.w_20_1  (ctx : sequential_generatedRoot_264Ctx) : mxxMat1 :=
  ctx.witness.w_20_1

def sequential_generatedRoot_264.w_20_2  (ctx : sequential_generatedRoot_264Ctx) : mxxMat1 :=
  ctx.witness.w_20_2

def sequential_generatedRoot_264.w_20_3  (ctx : sequential_generatedRoot_264Ctx) : mxxMat1 :=
  ctx.witness.w_20_3

def sequential_generatedRoot_264.w_21_0  (ctx : sequential_generatedRoot_264Ctx) : mxxMat2 :=
  ctx.witness.w_21_0

def sequential_generatedRoot_264.w_22_0  (ctx : sequential_generatedRoot_264Ctx) : mxxMat2 :=
  MxxRuntime.matrixAdd (sequential_generatedRoot_264.w_0_0 ctx) (sequential_generatedRoot_264.w_21_0 ctx)

def sequential_generatedRoot_264.w_24_0  (ctx : sequential_generatedRoot_264Ctx) : mxxMat3 :=
  (MxxRuntime.unitRow ((1 + (Int.ofNat i_0))) : mxxMat3)

def sequential_generatedRoot_264.w_25_0  (ctx : sequential_generatedRoot_264Ctx) : mxxMat2 :=
  ctx.witness.w_25_0

def sequential_generatedRoot_264.w_26_0  (ctx : sequential_generatedRoot_264Ctx) : mxxMat2 :=
  MxxRuntime.matrixAdd (sequential_generatedRoot_264.w_23_0 ctx) (sequential_generatedRoot_264.w_25_0 ctx)

def sequential_generatedRoot_264.w_28_0  (ctx : sequential_generatedRoot_264Ctx) : mxxMat3 :=
  (MxxRuntime.unitRow ((1 + (Int.ofNat i_0))) : mxxMat3)

def sequential_generatedRoot_264.w_29_0  (ctx : sequential_generatedRoot_264Ctx) : mxxMat2 :=
  ctx.witness.w_29_0

def sequential_generatedRoot_264.w_30_0  (ctx : sequential_generatedRoot_264Ctx) : mxxMat2 :=
  MxxRuntime.matrixAdd (sequential_generatedRoot_264.w_27_0 ctx) (sequential_generatedRoot_264.w_29_0 ctx)

def sequential_generatedRoot_264.w_32_0  (ctx : sequential_generatedRoot_264Ctx) : mxxMat3 :=
  (MxxRuntime.unitRow ((1 + (Int.ofNat i_0))) : mxxMat3)

def sequential_generatedRoot_264.w_33_0  (ctx : sequential_generatedRoot_264Ctx) : mxxMat2 :=
  ctx.witness.w_33_0

def sequential_generatedRoot_264.w_34_0  (ctx : sequential_generatedRoot_264Ctx) : mxxMat2 :=
  MxxRuntime.matrixAdd (sequential_generatedRoot_264.w_31_0 ctx) (sequential_generatedRoot_264.w_33_0 ctx)

def sequential_generatedRoot_264.outs  (ctx : sequential_generatedRoot_264Ctx) : mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × Unit :=
  ((sequential_generatedRoot_264.w_22_0 ctx), ((sequential_generatedRoot_264.w_26_0 ctx), ((sequential_generatedRoot_264.w_30_0 ctx), ((sequential_generatedRoot_264.w_34_0 ctx), ()))))

structure sequential_generatedRoot_264.Facts_0  (ctx : sequential_generatedRoot_264Ctx) : Prop where
  t0 : 0 ≤ ((1 + (Int.ofNat i_0)))
  t1 : ((1 + (Int.ofNat i_0))) < 306
  h3_0 : 0 ≤ (sequential_generatedRoot_264.w_3_0 ctx)
  h3_0_1 : (sequential_generatedRoot_264.w_3_0 ctx) < 20054016
  h4_0 : MxxRuntime.familyGetDynamic (sequential_generatedRoot_264.w_2_0 ctx) (sequential_generatedRoot_264.w_3_0 ctx) (sequential_generatedRoot_264.w_4_0 ctx)
  h5_0 : 0 ≤ (sequential_generatedRoot_264.w_5_0 ctx)
  h5_0_1 : (sequential_generatedRoot_264.w_5_0 ctx) < 20054016
  h6_0 : MxxRuntime.familyGetDynamic (sequential_generatedRoot_264.w_2_0 ctx) (sequential_generatedRoot_264.w_5_0 ctx) (sequential_generatedRoot_264.w_6_0 ctx)
  h3_0_2 : 0 ≤ (sequential_generatedRoot_264.w_3_0 ctx)
  h3_0_3 : (sequential_generatedRoot_264.w_3_0 ctx) < 20054016
  h9_0 : MxxRuntime.familyGetDynamic (sequential_generatedRoot_264.w_8_0 ctx) (sequential_generatedRoot_264.w_3_0 ctx) (sequential_generatedRoot_264.w_9_0 ctx)
  h5_0_2 : 0 ≤ (sequential_generatedRoot_264.w_5_0 ctx)
  h5_0_3 : (sequential_generatedRoot_264.w_5_0 ctx) < 20054016
  h10_0 : MxxRuntime.familyGetDynamic (sequential_generatedRoot_264.w_8_0 ctx) (sequential_generatedRoot_264.w_5_0 ctx) (sequential_generatedRoot_264.w_10_0 ctx)
  h3_0_4 : 0 ≤ (sequential_generatedRoot_264.w_3_0 ctx)
  h3_0_5 : (sequential_generatedRoot_264.w_3_0 ctx) < 20054016
  h13_0 : MxxRuntime.familyGetDynamic (sequential_generatedRoot_264.w_12_0 ctx) (sequential_generatedRoot_264.w_3_0 ctx) (sequential_generatedRoot_264.w_13_0 ctx)
  h5_0_4 : 0 ≤ (sequential_generatedRoot_264.w_5_0 ctx)
  h5_0_5 : (sequential_generatedRoot_264.w_5_0 ctx) < 20054016
  h14_0 : MxxRuntime.familyGetDynamic (sequential_generatedRoot_264.w_12_0 ctx) (sequential_generatedRoot_264.w_5_0 ctx) (sequential_generatedRoot_264.w_14_0 ctx)
  h3_0_6 : 0 ≤ (sequential_generatedRoot_264.w_3_0 ctx)
  h3_0_7 : (sequential_generatedRoot_264.w_3_0 ctx) < 20054016
  h17_0 : MxxRuntime.familyGetDynamic (sequential_generatedRoot_264.w_16_0 ctx) (sequential_generatedRoot_264.w_3_0 ctx) (sequential_generatedRoot_264.w_17_0 ctx)
  h5_0_6 : 0 ≤ (sequential_generatedRoot_264.w_5_0 ctx)
  h5_0_7 : (sequential_generatedRoot_264.w_5_0 ctx) < 20054016
  h18_0 : MxxRuntime.familyGetDynamic (sequential_generatedRoot_264.w_16_0 ctx) (sequential_generatedRoot_264.w_5_0 ctx) (sequential_generatedRoot_264.w_18_0 ctx)
  t26 : 0 ≤ (65534)
  h20_3 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => scope_150 backend hashModel tape (path ++ [20, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, ((sequential_generatedRoot_264.w_2_0 ctx), ((sequential_generatedRoot_264.w_8_0 ctx), ((sequential_generatedRoot_264.w_12_0 ctx), ((sequential_generatedRoot_264.w_16_0 ctx), ())))))))) next) (Int.toNat (65534)) ((sequential_generatedRoot_264.w_7_0 ctx), ((sequential_generatedRoot_264.w_11_0 ctx), ((sequential_generatedRoot_264.w_15_0 ctx), ((sequential_generatedRoot_264.w_19_0 ctx), ())))) ((sequential_generatedRoot_264.w_20_0 ctx), ((sequential_generatedRoot_264.w_20_1 ctx), ((sequential_generatedRoot_264.w_20_2 ctx), ((sequential_generatedRoot_264.w_20_3 ctx), ()))))
  h21_0 : MxxRuntime.tensorRuns (sequential_generatedRoot_264.w_1_0 ctx) (sequential_generatedRoot_264.w_20_0 ctx) (sequential_generatedRoot_264.w_21_0 ctx)
  t29 : 0 ≤ ((1 + (Int.ofNat i_0)))
  t30 : ((1 + (Int.ofNat i_0))) < 306
  h25_0 : MxxRuntime.tensorRuns (sequential_generatedRoot_264.w_24_0 ctx) (sequential_generatedRoot_264.w_20_1 ctx) (sequential_generatedRoot_264.w_25_0 ctx)

theorem sequential_generatedRoot_264.facts_0  {ctx : sequential_generatedRoot_264Ctx} {outputs : mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × Unit}
    (h : sequential_generatedRoot_264.constraints_0 backend hashModel tape path params i_0 (sequential_generatedRoot_264.w_2_0 ctx) (sequential_generatedRoot_264.w_8_0 ctx) (sequential_generatedRoot_264.w_12_0 ctx) (sequential_generatedRoot_264.w_16_0 ctx) (sequential_generatedRoot_264.w_1_0 ctx) (sequential_generatedRoot_264.w_3_0 ctx) (sequential_generatedRoot_264.w_4_0 ctx) (sequential_generatedRoot_264.w_5_0 ctx) (sequential_generatedRoot_264.w_6_0 ctx) (sequential_generatedRoot_264.w_7_0 ctx) (sequential_generatedRoot_264.w_9_0 ctx) (sequential_generatedRoot_264.w_10_0 ctx) (sequential_generatedRoot_264.w_11_0 ctx) (sequential_generatedRoot_264.w_13_0 ctx) (sequential_generatedRoot_264.w_14_0 ctx) (sequential_generatedRoot_264.w_15_0 ctx) (sequential_generatedRoot_264.w_17_0 ctx) (sequential_generatedRoot_264.w_18_0 ctx) (sequential_generatedRoot_264.w_19_0 ctx) (sequential_generatedRoot_264.w_20_0 ctx) (sequential_generatedRoot_264.w_20_1 ctx) (sequential_generatedRoot_264.w_20_2 ctx) (sequential_generatedRoot_264.w_20_3 ctx) (sequential_generatedRoot_264.w_21_0 ctx) (sequential_generatedRoot_264.w_24_0 ctx) (sequential_generatedRoot_264.w_25_0 ctx)) :
    sequential_generatedRoot_264.Facts_0 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure sequential_generatedRoot_264.Facts_1  (ctx : sequential_generatedRoot_264Ctx) (outputs : mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × Unit) : Prop where
  t32 : 0 ≤ ((1 + (Int.ofNat i_0)))
  t33 : ((1 + (Int.ofNat i_0))) < 306
  h29_0 : MxxRuntime.tensorRuns (sequential_generatedRoot_264.w_28_0 ctx) (sequential_generatedRoot_264.w_20_2 ctx) (sequential_generatedRoot_264.w_29_0 ctx)
  t35 : 0 ≤ ((1 + (Int.ofNat i_0)))
  t36 : ((1 + (Int.ofNat i_0))) < 306
  h33_0 : MxxRuntime.tensorRuns (sequential_generatedRoot_264.w_32_0 ctx) (sequential_generatedRoot_264.w_20_3 ctx) (sequential_generatedRoot_264.w_33_0 ctx)
  hout : outputs = sequential_generatedRoot_264.outs ctx

theorem sequential_generatedRoot_264.facts_1  {ctx : sequential_generatedRoot_264Ctx} {outputs : mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × Unit}
    (h : sequential_generatedRoot_264.constraints_1 i_0 (sequential_generatedRoot_264.w_20_2 ctx) (sequential_generatedRoot_264.w_20_3 ctx) (sequential_generatedRoot_264.w_22_0 ctx) (sequential_generatedRoot_264.w_26_0 ctx) (sequential_generatedRoot_264.w_28_0 ctx) (sequential_generatedRoot_264.w_29_0 ctx) (sequential_generatedRoot_264.w_30_0 ctx) (sequential_generatedRoot_264.w_32_0 ctx) (sequential_generatedRoot_264.w_33_0 ctx) (sequential_generatedRoot_264.w_34_0 ctx) outputs) :
    sequential_generatedRoot_264.Facts_1 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2⟩

structure sequential_generatedRoot_264.Facts  (ctx : sequential_generatedRoot_264Ctx) (outputs : mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × Unit) : Prop where
  c0 : sequential_generatedRoot_264.Facts_0 ctx
  c1 : sequential_generatedRoot_264.Facts_1 ctx outputs

theorem sequential_generatedRoot_264.facts  (ctx : sequential_generatedRoot_264Ctx) {outputs : mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × Unit}
    (h : sequential_generatedRoot_264.body backend hashModel tape path params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_264.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_264.facts_0 (ctx := ctx) (outputs := outputs) t0.1
  have t1 := t0.2
  have r1 := sequential_generatedRoot_264.facts_1 (ctx := ctx) (outputs := outputs) t1
  ⟨r0, r1⟩

end

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params}
local notation "generatedRootCtx" => (generatedRoot.Ctx backend hashModel tape path params)

structure generatedRoot.Facts_0  (ctx : generatedRootCtx) : Prop where
  t0 : (4) = 4
  h3_3 : (∀ i : Fin 4, parallel_generatedRoot_3 backend hashModel tape (path ++ [3, i.val]) params i ((generatedRoot.w_0_0 ctx), ((generatedRoot.w_1_0 ctx), ((generatedRoot.w_2_0 ctx), ()))) (((generatedRoot.w_3_0 ctx) i), (((generatedRoot.w_3_1 ctx) i), (((generatedRoot.w_3_2 ctx) i), (((generatedRoot.w_3_3 ctx) i), ())))))
  t2 : (128) = 128
  h4_3 : (∀ i : Fin 128, parallel_generatedRoot_4 backend hashModel tape (path ++ [4, i.val]) params i ((generatedRoot.w_0_0 ctx), ((generatedRoot.w_1_0 ctx), ((generatedRoot.w_2_0 ctx), ()))) (((generatedRoot.w_4_0 ctx) i), (((generatedRoot.w_4_1 ctx) i), (((generatedRoot.w_4_2 ctx) i), (((generatedRoot.w_4_3 ctx) i), ())))))
  h7_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1024), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((generatedRoot.w_6_0 ctx)) (generatedRoot.w_7_0 ctx)
  h10_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1024), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) ((generatedRoot.w_6_0 ctx)) (generatedRoot.w_10_0 ctx)
  h13_0 : MxxRuntime.concatRows (generatedRoot.w_9_0 ctx) (generatedRoot.w_12_0 ctx) (generatedRoot.w_13_0 ctx)
  h14_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 115, 116, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((generatedRoot.w_6_0 ctx)) (generatedRoot.w_14_0 ctx)
  h15_0 : MxxRuntime.gadgetDecomposeRuns backend 64 270 (generatedRoot.w_14_0 ctx) (generatedRoot.w_15_0 ctx)
  h17_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1024), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((generatedRoot.w_6_0 ctx)) (generatedRoot.w_17_0 ctx)
  h20_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1024), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) ((generatedRoot.w_6_0 ctx)) (generatedRoot.w_20_0 ctx)
  h23_0 : MxxRuntime.concatRows (generatedRoot.w_19_0 ctx) (generatedRoot.w_22_0 ctx) (generatedRoot.w_23_0 ctx)
  h24_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 115, 116, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) ((generatedRoot.w_6_0 ctx)) (generatedRoot.w_24_0 ctx)
  h25_0 : MxxRuntime.gadgetDecomposeRuns backend 64 270 (generatedRoot.w_24_0 ctx) (generatedRoot.w_25_0 ctx)
  t14 : 0 ≤ 0
  t15 : 0 < 1
  t16 : 1 ≤ 2
  t17 : 0 ≤ 0
  t18 : 0 < 270
  t19 : 270 ≤ 270
  h28_0 : MxxRuntime.sliceMatrix (generatedRoot.w_27_0 ctx) 0 1 0 270 (generatedRoot.w_28_0 ctx)
  t21 : 0 ≤ 1
  t22 : 1 < 2
  t23 : 2 ≤ 2
  t24 : 0 ≤ 0
  t25 : 0 < 270
  t26 : 270 ≤ 270
  h29_0 : MxxRuntime.sliceMatrix (generatedRoot.w_27_0 ctx) 1 2 0 270 (generatedRoot.w_29_0 ctx)
  h30_0 : MxxRuntime.concatRows (generatedRoot.w_28_0 ctx) (generatedRoot.w_29_0 ctx) (generatedRoot.w_30_0 ctx)
  t29 : (1) = 1
  h33_3 : (∀ i : Fin 1, parallel_generatedRoot_33 backend hashModel tape (path ++ [33, i.val]) params i ((generatedRoot.w_0_0 ctx), ((generatedRoot.w_1_0 ctx), (((generatedRoot.w_31_0 ctx) i), (((generatedRoot.w_32_0 ctx) i), ((generatedRoot.w_2_0 ctx), ()))))) (((generatedRoot.w_33_0 ctx) i), (((generatedRoot.w_33_1 ctx) i), (((generatedRoot.w_33_2 ctx) i), (((generatedRoot.w_33_3 ctx) i), ())))))
  h34_0 : MxxRuntime.familyGetStatic (generatedRoot.w_33_0 ctx) (0) (generatedRoot.w_34_0 ctx)

theorem generatedRoot.facts_0  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_0 backend hashModel tape path params (generatedRoot.w_0_0 ctx) (generatedRoot.w_1_0 ctx) (generatedRoot.w_6_0 ctx) (generatedRoot.w_31_0 ctx) (generatedRoot.w_32_0 ctx) (generatedRoot.w_2_0 ctx) (generatedRoot.w_3_0 ctx) (generatedRoot.w_3_1 ctx) (generatedRoot.w_3_2 ctx) (generatedRoot.w_3_3 ctx) (generatedRoot.w_4_0 ctx) (generatedRoot.w_4_1 ctx) (generatedRoot.w_4_2 ctx) (generatedRoot.w_4_3 ctx) (generatedRoot.w_7_0 ctx) (generatedRoot.w_9_0 ctx) (generatedRoot.w_10_0 ctx) (generatedRoot.w_12_0 ctx) (generatedRoot.w_13_0 ctx) (generatedRoot.w_14_0 ctx) (generatedRoot.w_15_0 ctx) (generatedRoot.w_17_0 ctx) (generatedRoot.w_19_0 ctx) (generatedRoot.w_20_0 ctx) (generatedRoot.w_22_0 ctx) (generatedRoot.w_23_0 ctx) (generatedRoot.w_24_0 ctx) (generatedRoot.w_25_0 ctx) (generatedRoot.w_27_0 ctx) (generatedRoot.w_28_0 ctx) (generatedRoot.w_29_0 ctx) (generatedRoot.w_30_0 ctx) (generatedRoot.w_33_0 ctx) (generatedRoot.w_33_1 ctx) (generatedRoot.w_33_2 ctx) (generatedRoot.w_33_3 ctx) (generatedRoot.w_34_0 ctx)) :
    generatedRoot.Facts_0 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_1  (ctx : generatedRootCtx) : Prop where
  h35_0 : MxxRuntime.gadgetDecomposeRuns backend 64 270 (generatedRoot.w_34_0 ctx) (generatedRoot.w_35_0 ctx)
  h36_0 : MxxRuntime.compactCenteredRebaseRuns 32 (generatedRoot.w_35_0 ctx) (generatedRoot.w_36_0 ctx)
  h38_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 115, 116, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((generatedRoot.w_6_0 ctx)) (generatedRoot.w_38_0 ctx)
  h39_0 : MxxRuntime.gadgetDecomposeRuns backend 64 270 (generatedRoot.w_38_0 ctx) (generatedRoot.w_39_0 ctx)
  h41_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 115, 116, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) ((generatedRoot.w_6_0 ctx)) (generatedRoot.w_41_0 ctx)
  h42_0 : MxxRuntime.gadgetDecomposeRuns backend 64 270 (generatedRoot.w_41_0 ctx) (generatedRoot.w_42_0 ctx)
  t38 : 0 ≤ 0
  t39 : 0 < 1
  t40 : 1 ≤ 2
  t41 : 0 ≤ 0
  t42 : 0 < 270
  t43 : 270 ≤ 270
  h45_0 : MxxRuntime.sliceMatrix (generatedRoot.w_44_0 ctx) 0 1 0 270 (generatedRoot.w_45_0 ctx)
  t45 : 0 ≤ 1
  t46 : 1 < 2
  t47 : 2 ≤ 2
  t48 : 0 ≤ 0
  t49 : 0 < 270
  t50 : 270 ≤ 270
  h46_0 : MxxRuntime.sliceMatrix (generatedRoot.w_44_0 ctx) 1 2 0 270 (generatedRoot.w_46_0 ctx)
  h47_0 : MxxRuntime.concatRows (generatedRoot.w_45_0 ctx) (generatedRoot.w_46_0 ctx) (generatedRoot.w_47_0 ctx)
  h48_0 : MxxRuntime.familyGetStatic (generatedRoot.w_33_2 ctx) (0) (generatedRoot.w_48_0 ctx)
  h49_0 : MxxRuntime.gadgetDecomposeRuns backend 64 270 (generatedRoot.w_48_0 ctx) (generatedRoot.w_49_0 ctx)
  h50_0 : MxxRuntime.compactCenteredRebaseRuns 32 (generatedRoot.w_49_0 ctx) (generatedRoot.w_50_0 ctx)
  t56 : 0 ≤ 0
  t57 : 0 < 1
  t58 : 1 ≤ 2
  t59 : 0 ≤ 0
  t60 : 0 < 270
  t61 : 270 ≤ 270
  h53_0 : MxxRuntime.sliceMatrix (generatedRoot.w_52_0 ctx) 0 1 0 270 (generatedRoot.w_53_0 ctx)
  t63 : 0 ≤ 1

theorem generatedRoot.facts_1  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_1 backend hashModel (generatedRoot.w_6_0 ctx) (generatedRoot.w_33_2 ctx) (generatedRoot.w_34_0 ctx) (generatedRoot.w_35_0 ctx) (generatedRoot.w_36_0 ctx) (generatedRoot.w_38_0 ctx) (generatedRoot.w_39_0 ctx) (generatedRoot.w_41_0 ctx) (generatedRoot.w_42_0 ctx) (generatedRoot.w_44_0 ctx) (generatedRoot.w_45_0 ctx) (generatedRoot.w_46_0 ctx) (generatedRoot.w_47_0 ctx) (generatedRoot.w_48_0 ctx) (generatedRoot.w_49_0 ctx) (generatedRoot.w_50_0 ctx) (generatedRoot.w_52_0 ctx) (generatedRoot.w_53_0 ctx)) :
    generatedRoot.Facts_1 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_2  (ctx : generatedRootCtx) : Prop where
  t64 : 1 < 2
  t65 : 2 ≤ 2
  t66 : 0 ≤ 0
  t67 : 0 < 270
  t68 : 270 ≤ 270
  h54_0 : MxxRuntime.sliceMatrix (generatedRoot.w_52_0 ctx) 1 2 0 270 (generatedRoot.w_54_0 ctx)
  h55_0 : MxxRuntime.familyGetStatic (generatedRoot.w_33_1 ctx) (0) (generatedRoot.w_55_0 ctx)
  h56_0 : MxxRuntime.gadgetDecomposeRuns backend 64 270 (generatedRoot.w_55_0 ctx) (generatedRoot.w_56_0 ctx)
  h57_0 : MxxRuntime.compactCenteredRebaseRuns 32 (generatedRoot.w_56_0 ctx) (generatedRoot.w_57_0 ctx)
  h59_0 : MxxRuntime.familyGetStatic (generatedRoot.w_33_3 ctx) (0) (generatedRoot.w_59_0 ctx)
  h60_0 : MxxRuntime.gadgetDecomposeRuns backend 64 270 (generatedRoot.w_59_0 ctx) (generatedRoot.w_60_0 ctx)
  h61_0 : MxxRuntime.compactCenteredRebaseRuns 32 (generatedRoot.w_60_0 ctx) (generatedRoot.w_61_0 ctx)
  t76 : 0 ≤ 0
  t77 : 0 < 1
  t78 : 1 ≤ 2
  t79 : 0 ≤ 0
  t80 : 0 < 270
  t81 : 270 ≤ 270
  h64_0 : MxxRuntime.sliceMatrix (generatedRoot.w_63_0 ctx) 0 1 0 270 (generatedRoot.w_64_0 ctx)
  t83 : 0 ≤ 1
  t84 : 1 < 2
  t85 : 2 ≤ 2
  t86 : 0 ≤ 0
  t87 : 0 < 270
  t88 : 270 ≤ 270
  h65_0 : MxxRuntime.sliceMatrix (generatedRoot.w_63_0 ctx) 1 2 0 270 (generatedRoot.w_65_0 ctx)
  t90 : 0 ≤ (4)
  h66_3 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_generatedRoot_66 backend hashModel tape (path ++ [66, i]) params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, ((generatedRoot.w_6_0 ctx), ((generatedRoot.w_53_0 ctx), ((generatedRoot.w_54_0 ctx), ((generatedRoot.w_3_0 ctx), ((generatedRoot.w_64_0 ctx), ((generatedRoot.w_65_0 ctx), ((generatedRoot.w_3_2 ctx), ((generatedRoot.w_3_1 ctx), ((generatedRoot.w_3_3 ctx), ()))))))))))))) next) (Int.toNat (4)) ((generatedRoot.w_5_0 ctx), ((generatedRoot.w_5_0 ctx), ((generatedRoot.w_5_0 ctx), ((generatedRoot.w_5_0 ctx), ())))) ((generatedRoot.w_66_0 ctx), ((generatedRoot.w_66_1 ctx), ((generatedRoot.w_66_2 ctx), ((generatedRoot.w_66_3 ctx), ()))))
  t92 : 0 ≤ (0)
  t93 : (0) < 306
  h102_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((generatedRoot.w_6_0 ctx)) (generatedRoot.w_102_0 ctx)
  h103_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) ((generatedRoot.w_6_0 ctx)) (generatedRoot.w_103_0 ctx)

theorem generatedRoot.facts_2  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_2 backend hashModel tape path params (generatedRoot.w_6_0 ctx) (generatedRoot.w_3_0 ctx) (generatedRoot.w_3_1 ctx) (generatedRoot.w_3_2 ctx) (generatedRoot.w_3_3 ctx) (generatedRoot.w_5_0 ctx) (generatedRoot.w_33_1 ctx) (generatedRoot.w_33_3 ctx) (generatedRoot.w_52_0 ctx) (generatedRoot.w_53_0 ctx) (generatedRoot.w_54_0 ctx) (generatedRoot.w_55_0 ctx) (generatedRoot.w_56_0 ctx) (generatedRoot.w_57_0 ctx) (generatedRoot.w_59_0 ctx) (generatedRoot.w_60_0 ctx) (generatedRoot.w_61_0 ctx) (generatedRoot.w_63_0 ctx) (generatedRoot.w_64_0 ctx) (generatedRoot.w_65_0 ctx) (generatedRoot.w_66_0 ctx) (generatedRoot.w_66_1 ctx) (generatedRoot.w_66_2 ctx) (generatedRoot.w_66_3 ctx) (generatedRoot.w_102_0 ctx) (generatedRoot.w_103_0 ctx)) :
    generatedRoot.Facts_2 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_3  (ctx : generatedRootCtx) : Prop where
  h104_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((generatedRoot.w_6_0 ctx)) (generatedRoot.w_104_0 ctx)
  h105_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) ((generatedRoot.w_6_0 ctx)) (generatedRoot.w_105_0 ctx)
  t98 : (2607022080) = 2607022080
  h108_7 : (∀ i : Fin 2607022080, parallel_generatedRoot_108 backend hashModel tape (path ++ [108, i.val]) params i ((generatedRoot.w_6_0 ctx), (((generatedRoot.w_101_0 ctx) i), ((generatedRoot.w_102_0 ctx), ((generatedRoot.w_103_0 ctx), ((generatedRoot.w_0_0 ctx), ((generatedRoot.w_104_0 ctx), ((generatedRoot.w_105_0 ctx), ((generatedRoot.w_1_0 ctx), (((generatedRoot.w_106_0 ctx) i), (((generatedRoot.w_107_0 ctx) i), ((generatedRoot.w_2_0 ctx), ()))))))))))) (((generatedRoot.w_108_0 ctx) i), (((generatedRoot.w_108_1 ctx) i), (((generatedRoot.w_108_2 ctx) i), (((generatedRoot.w_108_3 ctx) i), (((generatedRoot.w_108_4 ctx) i), (((generatedRoot.w_108_5 ctx) i), (((generatedRoot.w_108_6 ctx) i), (((generatedRoot.w_108_7 ctx) i), ())))))))))
  t100 : (20054016) = 20054016
  h109_3 : (∀ i : Fin 20054016, parallel_generatedRoot_109 backend hashModel tape (path ++ [109, i.val]) params i ((generatedRoot.w_108_0 ctx), ((generatedRoot.w_108_1 ctx), ((generatedRoot.w_108_2 ctx), ((generatedRoot.w_108_3 ctx), ())))) (((generatedRoot.w_109_0 ctx) i), (((generatedRoot.w_109_1 ctx) i), (((generatedRoot.w_109_2 ctx) i), (((generatedRoot.w_109_3 ctx) i), ())))))
  t102 : (2607022080) = 2607022080
  h112_7 : (∀ i : Fin 2607022080, parallel_generatedRoot_112 backend hashModel tape (path ++ [112, i.val]) params i ((generatedRoot.w_6_0 ctx), (((generatedRoot.w_100_0 ctx) i), ((generatedRoot.w_109_0 ctx), ((generatedRoot.w_109_2 ctx), ((generatedRoot.w_0_0 ctx), ((generatedRoot.w_109_1 ctx), ((generatedRoot.w_109_3 ctx), ((generatedRoot.w_1_0 ctx), (((generatedRoot.w_110_0 ctx) i), (((generatedRoot.w_111_0 ctx) i), ((generatedRoot.w_2_0 ctx), ()))))))))))) (((generatedRoot.w_112_0 ctx) i), (((generatedRoot.w_112_1 ctx) i), (((generatedRoot.w_112_2 ctx) i), (((generatedRoot.w_112_3 ctx) i), (((generatedRoot.w_112_4 ctx) i), (((generatedRoot.w_112_5 ctx) i), (((generatedRoot.w_112_6 ctx) i), (((generatedRoot.w_112_7 ctx) i), ())))))))))
  t104 : (20054016) = 20054016
  h113_3 : (∀ i : Fin 20054016, parallel_generatedRoot_113 backend hashModel tape (path ++ [113, i.val]) params i ((generatedRoot.w_112_0 ctx), ((generatedRoot.w_112_1 ctx), ((generatedRoot.w_112_2 ctx), ((generatedRoot.w_112_3 ctx), ())))) (((generatedRoot.w_113_0 ctx) i), (((generatedRoot.w_113_1 ctx) i), (((generatedRoot.w_113_2 ctx) i), (((generatedRoot.w_113_3 ctx) i), ())))))
  t106 : (2607022080) = 2607022080
  h116_7 : (∀ i : Fin 2607022080, parallel_generatedRoot_116 backend hashModel tape (path ++ [116, i.val]) params i ((generatedRoot.w_6_0 ctx), (((generatedRoot.w_99_0 ctx) i), ((generatedRoot.w_113_0 ctx), ((generatedRoot.w_113_2 ctx), ((generatedRoot.w_0_0 ctx), ((generatedRoot.w_113_1 ctx), ((generatedRoot.w_113_3 ctx), ((generatedRoot.w_1_0 ctx), (((generatedRoot.w_114_0 ctx) i), (((generatedRoot.w_115_0 ctx) i), ((generatedRoot.w_2_0 ctx), ()))))))))))) (((generatedRoot.w_116_0 ctx) i), (((generatedRoot.w_116_1 ctx) i), (((generatedRoot.w_116_2 ctx) i), (((generatedRoot.w_116_3 ctx) i), (((generatedRoot.w_116_4 ctx) i), (((generatedRoot.w_116_5 ctx) i), (((generatedRoot.w_116_6 ctx) i), (((generatedRoot.w_116_7 ctx) i), ())))))))))
  t108 : (20054016) = 20054016
  h117_3 : (∀ i : Fin 20054016, parallel_generatedRoot_117 backend hashModel tape (path ++ [117, i.val]) params i ((generatedRoot.w_116_0 ctx), ((generatedRoot.w_116_1 ctx), ((generatedRoot.w_116_2 ctx), ((generatedRoot.w_116_3 ctx), ())))) (((generatedRoot.w_117_0 ctx) i), (((generatedRoot.w_117_1 ctx) i), (((generatedRoot.w_117_2 ctx) i), (((generatedRoot.w_117_3 ctx) i), ())))))
  t110 : (2607022080) = 2607022080
  h120_7 : (∀ i : Fin 2607022080, parallel_generatedRoot_120 backend hashModel tape (path ++ [120, i.val]) params i ((generatedRoot.w_6_0 ctx), (((generatedRoot.w_98_0 ctx) i), ((generatedRoot.w_117_0 ctx), ((generatedRoot.w_117_2 ctx), ((generatedRoot.w_0_0 ctx), ((generatedRoot.w_117_1 ctx), ((generatedRoot.w_117_3 ctx), ((generatedRoot.w_1_0 ctx), (((generatedRoot.w_118_0 ctx) i), (((generatedRoot.w_119_0 ctx) i), ((generatedRoot.w_2_0 ctx), ()))))))))))) (((generatedRoot.w_120_0 ctx) i), (((generatedRoot.w_120_1 ctx) i), (((generatedRoot.w_120_2 ctx) i), (((generatedRoot.w_120_3 ctx) i), (((generatedRoot.w_120_4 ctx) i), (((generatedRoot.w_120_5 ctx) i), (((generatedRoot.w_120_6 ctx) i), (((generatedRoot.w_120_7 ctx) i), ())))))))))
  t112 : (20054016) = 20054016
  h121_3 : (∀ i : Fin 20054016, parallel_generatedRoot_121 backend hashModel tape (path ++ [121, i.val]) params i ((generatedRoot.w_120_0 ctx), ((generatedRoot.w_120_1 ctx), ((generatedRoot.w_120_2 ctx), ((generatedRoot.w_120_3 ctx), ())))) (((generatedRoot.w_121_0 ctx) i), (((generatedRoot.w_121_1 ctx) i), (((generatedRoot.w_121_2 ctx) i), (((generatedRoot.w_121_3 ctx) i), ())))))
  t114 : (2607022080) = 2607022080
  h124_7 : (∀ i : Fin 2607022080, parallel_generatedRoot_124 backend hashModel tape (path ++ [124, i.val]) params i ((generatedRoot.w_6_0 ctx), (((generatedRoot.w_97_0 ctx) i), ((generatedRoot.w_121_0 ctx), ((generatedRoot.w_121_2 ctx), ((generatedRoot.w_0_0 ctx), ((generatedRoot.w_121_1 ctx), ((generatedRoot.w_121_3 ctx), ((generatedRoot.w_1_0 ctx), (((generatedRoot.w_122_0 ctx) i), (((generatedRoot.w_123_0 ctx) i), ((generatedRoot.w_2_0 ctx), ()))))))))))) (((generatedRoot.w_124_0 ctx) i), (((generatedRoot.w_124_1 ctx) i), (((generatedRoot.w_124_2 ctx) i), (((generatedRoot.w_124_3 ctx) i), (((generatedRoot.w_124_4 ctx) i), (((generatedRoot.w_124_5 ctx) i), (((generatedRoot.w_124_6 ctx) i), (((generatedRoot.w_124_7 ctx) i), ())))))))))
  t116 : (20054016) = 20054016
  h125_3 : (∀ i : Fin 20054016, parallel_generatedRoot_125 backend hashModel tape (path ++ [125, i.val]) params i ((generatedRoot.w_124_0 ctx), ((generatedRoot.w_124_1 ctx), ((generatedRoot.w_124_2 ctx), ((generatedRoot.w_124_3 ctx), ())))) (((generatedRoot.w_125_0 ctx) i), (((generatedRoot.w_125_1 ctx) i), (((generatedRoot.w_125_2 ctx) i), (((generatedRoot.w_125_3 ctx) i), ())))))
  t118 : (2607022080) = 2607022080
  h128_7 : (∀ i : Fin 2607022080, parallel_generatedRoot_128 backend hashModel tape (path ++ [128, i.val]) params i ((generatedRoot.w_6_0 ctx), (((generatedRoot.w_96_0 ctx) i), ((generatedRoot.w_125_0 ctx), ((generatedRoot.w_125_2 ctx), ((generatedRoot.w_0_0 ctx), ((generatedRoot.w_125_1 ctx), ((generatedRoot.w_125_3 ctx), ((generatedRoot.w_1_0 ctx), (((generatedRoot.w_126_0 ctx) i), (((generatedRoot.w_127_0 ctx) i), ((generatedRoot.w_2_0 ctx), ()))))))))))) (((generatedRoot.w_128_0 ctx) i), (((generatedRoot.w_128_1 ctx) i), (((generatedRoot.w_128_2 ctx) i), (((generatedRoot.w_128_3 ctx) i), (((generatedRoot.w_128_4 ctx) i), (((generatedRoot.w_128_5 ctx) i), (((generatedRoot.w_128_6 ctx) i), (((generatedRoot.w_128_7 ctx) i), ())))))))))
  t120 : (20054016) = 20054016
  h129_3 : (∀ i : Fin 20054016, parallel_generatedRoot_129 backend hashModel tape (path ++ [129, i.val]) params i ((generatedRoot.w_128_0 ctx), ((generatedRoot.w_128_1 ctx), ((generatedRoot.w_128_2 ctx), ((generatedRoot.w_128_3 ctx), ())))) (((generatedRoot.w_129_0 ctx) i), (((generatedRoot.w_129_1 ctx) i), (((generatedRoot.w_129_2 ctx) i), (((generatedRoot.w_129_3 ctx) i), ())))))
  t122 : (2607022080) = 2607022080
  h132_7 : (∀ i : Fin 2607022080, parallel_generatedRoot_132 backend hashModel tape (path ++ [132, i.val]) params i ((generatedRoot.w_6_0 ctx), (((generatedRoot.w_95_0 ctx) i), ((generatedRoot.w_129_0 ctx), ((generatedRoot.w_129_2 ctx), ((generatedRoot.w_0_0 ctx), ((generatedRoot.w_129_1 ctx), ((generatedRoot.w_129_3 ctx), ((generatedRoot.w_1_0 ctx), (((generatedRoot.w_130_0 ctx) i), (((generatedRoot.w_131_0 ctx) i), ((generatedRoot.w_2_0 ctx), ()))))))))))) (((generatedRoot.w_132_0 ctx) i), (((generatedRoot.w_132_1 ctx) i), (((generatedRoot.w_132_2 ctx) i), (((generatedRoot.w_132_3 ctx) i), (((generatedRoot.w_132_4 ctx) i), (((generatedRoot.w_132_5 ctx) i), (((generatedRoot.w_132_6 ctx) i), (((generatedRoot.w_132_7 ctx) i), ())))))))))
  t124 : (20054016) = 20054016
  h133_3 : (∀ i : Fin 20054016, parallel_generatedRoot_133 backend hashModel tape (path ++ [133, i.val]) params i ((generatedRoot.w_132_0 ctx), ((generatedRoot.w_132_1 ctx), ((generatedRoot.w_132_2 ctx), ((generatedRoot.w_132_3 ctx), ())))) (((generatedRoot.w_133_0 ctx) i), (((generatedRoot.w_133_1 ctx) i), (((generatedRoot.w_133_2 ctx) i), (((generatedRoot.w_133_3 ctx) i), ())))))
  t126 : (2607022080) = 2607022080
  h136_7 : (∀ i : Fin 2607022080, parallel_generatedRoot_136 backend hashModel tape (path ++ [136, i.val]) params i ((generatedRoot.w_6_0 ctx), (((generatedRoot.w_94_0 ctx) i), ((generatedRoot.w_133_0 ctx), ((generatedRoot.w_133_2 ctx), ((generatedRoot.w_0_0 ctx), ((generatedRoot.w_133_1 ctx), ((generatedRoot.w_133_3 ctx), ((generatedRoot.w_1_0 ctx), (((generatedRoot.w_134_0 ctx) i), (((generatedRoot.w_135_0 ctx) i), ((generatedRoot.w_2_0 ctx), ()))))))))))) (((generatedRoot.w_136_0 ctx) i), (((generatedRoot.w_136_1 ctx) i), (((generatedRoot.w_136_2 ctx) i), (((generatedRoot.w_136_3 ctx) i), (((generatedRoot.w_136_4 ctx) i), (((generatedRoot.w_136_5 ctx) i), (((generatedRoot.w_136_6 ctx) i), (((generatedRoot.w_136_7 ctx) i), ())))))))))

theorem generatedRoot.facts_3  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_3 backend hashModel tape path params (generatedRoot.w_0_0 ctx) (generatedRoot.w_1_0 ctx) (generatedRoot.w_6_0 ctx) (generatedRoot.w_94_0 ctx) (generatedRoot.w_95_0 ctx) (generatedRoot.w_96_0 ctx) (generatedRoot.w_97_0 ctx) (generatedRoot.w_98_0 ctx) (generatedRoot.w_99_0 ctx) (generatedRoot.w_100_0 ctx) (generatedRoot.w_101_0 ctx) (generatedRoot.w_106_0 ctx) (generatedRoot.w_107_0 ctx) (generatedRoot.w_110_0 ctx) (generatedRoot.w_111_0 ctx) (generatedRoot.w_114_0 ctx) (generatedRoot.w_115_0 ctx) (generatedRoot.w_118_0 ctx) (generatedRoot.w_119_0 ctx) (generatedRoot.w_122_0 ctx) (generatedRoot.w_123_0 ctx) (generatedRoot.w_126_0 ctx) (generatedRoot.w_127_0 ctx) (generatedRoot.w_130_0 ctx) (generatedRoot.w_131_0 ctx) (generatedRoot.w_134_0 ctx) (generatedRoot.w_135_0 ctx) (generatedRoot.w_2_0 ctx) (generatedRoot.w_102_0 ctx) (generatedRoot.w_103_0 ctx) (generatedRoot.w_104_0 ctx) (generatedRoot.w_105_0 ctx) (generatedRoot.w_108_0 ctx) (generatedRoot.w_108_1 ctx) (generatedRoot.w_108_2 ctx) (generatedRoot.w_108_3 ctx) (generatedRoot.w_108_4 ctx) (generatedRoot.w_108_5 ctx) (generatedRoot.w_108_6 ctx) (generatedRoot.w_108_7 ctx) (generatedRoot.w_109_0 ctx) (generatedRoot.w_109_1 ctx) (generatedRoot.w_109_2 ctx) (generatedRoot.w_109_3 ctx) (generatedRoot.w_112_0 ctx) (generatedRoot.w_112_1 ctx) (generatedRoot.w_112_2 ctx) (generatedRoot.w_112_3 ctx) (generatedRoot.w_112_4 ctx) (generatedRoot.w_112_5 ctx) (generatedRoot.w_112_6 ctx) (generatedRoot.w_112_7 ctx) (generatedRoot.w_113_0 ctx) (generatedRoot.w_113_1 ctx) (generatedRoot.w_113_2 ctx) (generatedRoot.w_113_3 ctx) (generatedRoot.w_116_0 ctx) (generatedRoot.w_116_1 ctx) (generatedRoot.w_116_2 ctx) (generatedRoot.w_116_3 ctx) (generatedRoot.w_116_4 ctx) (generatedRoot.w_116_5 ctx) (generatedRoot.w_116_6 ctx) (generatedRoot.w_116_7 ctx) (generatedRoot.w_117_0 ctx) (generatedRoot.w_117_1 ctx) (generatedRoot.w_117_2 ctx) (generatedRoot.w_117_3 ctx) (generatedRoot.w_120_0 ctx) (generatedRoot.w_120_1 ctx) (generatedRoot.w_120_2 ctx) (generatedRoot.w_120_3 ctx) (generatedRoot.w_120_4 ctx) (generatedRoot.w_120_5 ctx) (generatedRoot.w_120_6 ctx) (generatedRoot.w_120_7 ctx) (generatedRoot.w_121_0 ctx) (generatedRoot.w_121_1 ctx) (generatedRoot.w_121_2 ctx) (generatedRoot.w_121_3 ctx) (generatedRoot.w_124_0 ctx) (generatedRoot.w_124_1 ctx) (generatedRoot.w_124_2 ctx) (generatedRoot.w_124_3 ctx) (generatedRoot.w_124_4 ctx) (generatedRoot.w_124_5 ctx) (generatedRoot.w_124_6 ctx) (generatedRoot.w_124_7 ctx) (generatedRoot.w_125_0 ctx) (generatedRoot.w_125_1 ctx) (generatedRoot.w_125_2 ctx) (generatedRoot.w_125_3 ctx) (generatedRoot.w_128_0 ctx) (generatedRoot.w_128_1 ctx) (generatedRoot.w_128_2 ctx) (generatedRoot.w_128_3 ctx) (generatedRoot.w_128_4 ctx) (generatedRoot.w_128_5 ctx) (generatedRoot.w_128_6 ctx) (generatedRoot.w_128_7 ctx) (generatedRoot.w_129_0 ctx) (generatedRoot.w_129_1 ctx) (generatedRoot.w_129_2 ctx) (generatedRoot.w_129_3 ctx) (generatedRoot.w_132_0 ctx) (generatedRoot.w_132_1 ctx) (generatedRoot.w_132_2 ctx) (generatedRoot.w_132_3 ctx) (generatedRoot.w_132_4 ctx) (generatedRoot.w_132_5 ctx) (generatedRoot.w_132_6 ctx) (generatedRoot.w_132_7 ctx) (generatedRoot.w_133_0 ctx) (generatedRoot.w_133_1 ctx) (generatedRoot.w_133_2 ctx) (generatedRoot.w_133_3 ctx) (generatedRoot.w_136_0 ctx) (generatedRoot.w_136_1 ctx) (generatedRoot.w_136_2 ctx) (generatedRoot.w_136_3 ctx) (generatedRoot.w_136_4 ctx) (generatedRoot.w_136_5 ctx) (generatedRoot.w_136_6 ctx) (generatedRoot.w_136_7 ctx)) :
    generatedRoot.Facts_3 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_4  (ctx : generatedRootCtx) : Prop where
  t128 : (20054016) = 20054016
  h137_3 : (∀ i : Fin 20054016, parallel_generatedRoot_137 backend hashModel tape (path ++ [137, i.val]) params i ((generatedRoot.w_136_0 ctx), ((generatedRoot.w_136_1 ctx), ((generatedRoot.w_136_2 ctx), ((generatedRoot.w_136_3 ctx), ())))) (((generatedRoot.w_137_0 ctx) i), (((generatedRoot.w_137_1 ctx) i), (((generatedRoot.w_137_2 ctx) i), (((generatedRoot.w_137_3 ctx) i), ())))))
  t130 : (2607022080) = 2607022080
  h140_7 : (∀ i : Fin 2607022080, parallel_generatedRoot_140 backend hashModel tape (path ++ [140, i.val]) params i ((generatedRoot.w_6_0 ctx), (((generatedRoot.w_93_0 ctx) i), ((generatedRoot.w_137_0 ctx), ((generatedRoot.w_137_2 ctx), ((generatedRoot.w_0_0 ctx), ((generatedRoot.w_137_1 ctx), ((generatedRoot.w_137_3 ctx), ((generatedRoot.w_1_0 ctx), (((generatedRoot.w_138_0 ctx) i), (((generatedRoot.w_139_0 ctx) i), ((generatedRoot.w_2_0 ctx), ()))))))))))) (((generatedRoot.w_140_0 ctx) i), (((generatedRoot.w_140_1 ctx) i), (((generatedRoot.w_140_2 ctx) i), (((generatedRoot.w_140_3 ctx) i), (((generatedRoot.w_140_4 ctx) i), (((generatedRoot.w_140_5 ctx) i), (((generatedRoot.w_140_6 ctx) i), (((generatedRoot.w_140_7 ctx) i), ())))))))))
  t132 : (20054016) = 20054016
  h141_3 : (∀ i : Fin 20054016, parallel_generatedRoot_141 backend hashModel tape (path ++ [141, i.val]) params i ((generatedRoot.w_140_0 ctx), ((generatedRoot.w_140_1 ctx), ((generatedRoot.w_140_2 ctx), ((generatedRoot.w_140_3 ctx), ())))) (((generatedRoot.w_141_0 ctx) i), (((generatedRoot.w_141_1 ctx) i), (((generatedRoot.w_141_2 ctx) i), (((generatedRoot.w_141_3 ctx) i), ())))))
  t134 : (2607022080) = 2607022080
  h144_7 : (∀ i : Fin 2607022080, parallel_generatedRoot_144 backend hashModel tape (path ++ [144, i.val]) params i ((generatedRoot.w_6_0 ctx), (((generatedRoot.w_92_0 ctx) i), ((generatedRoot.w_141_0 ctx), ((generatedRoot.w_141_2 ctx), ((generatedRoot.w_0_0 ctx), ((generatedRoot.w_141_1 ctx), ((generatedRoot.w_141_3 ctx), ((generatedRoot.w_1_0 ctx), (((generatedRoot.w_142_0 ctx) i), (((generatedRoot.w_143_0 ctx) i), ((generatedRoot.w_2_0 ctx), ()))))))))))) (((generatedRoot.w_144_0 ctx) i), (((generatedRoot.w_144_1 ctx) i), (((generatedRoot.w_144_2 ctx) i), (((generatedRoot.w_144_3 ctx) i), (((generatedRoot.w_144_4 ctx) i), (((generatedRoot.w_144_5 ctx) i), (((generatedRoot.w_144_6 ctx) i), (((generatedRoot.w_144_7 ctx) i), ())))))))))
  t136 : (20054016) = 20054016
  h145_3 : (∀ i : Fin 20054016, parallel_generatedRoot_145 backend hashModel tape (path ++ [145, i.val]) params i ((generatedRoot.w_144_0 ctx), ((generatedRoot.w_144_1 ctx), ((generatedRoot.w_144_2 ctx), ((generatedRoot.w_144_3 ctx), ())))) (((generatedRoot.w_145_0 ctx) i), (((generatedRoot.w_145_1 ctx) i), (((generatedRoot.w_145_2 ctx) i), (((generatedRoot.w_145_3 ctx) i), ())))))
  t138 : (2607022080) = 2607022080
  h148_7 : (∀ i : Fin 2607022080, parallel_generatedRoot_148 backend hashModel tape (path ++ [148, i.val]) params i ((generatedRoot.w_6_0 ctx), (((generatedRoot.w_91_0 ctx) i), ((generatedRoot.w_145_0 ctx), ((generatedRoot.w_145_2 ctx), ((generatedRoot.w_0_0 ctx), ((generatedRoot.w_145_1 ctx), ((generatedRoot.w_145_3 ctx), ((generatedRoot.w_1_0 ctx), (((generatedRoot.w_146_0 ctx) i), (((generatedRoot.w_147_0 ctx) i), ((generatedRoot.w_2_0 ctx), ()))))))))))) (((generatedRoot.w_148_0 ctx) i), (((generatedRoot.w_148_1 ctx) i), (((generatedRoot.w_148_2 ctx) i), (((generatedRoot.w_148_3 ctx) i), (((generatedRoot.w_148_4 ctx) i), (((generatedRoot.w_148_5 ctx) i), (((generatedRoot.w_148_6 ctx) i), (((generatedRoot.w_148_7 ctx) i), ())))))))))
  t140 : (20054016) = 20054016
  h149_3 : (∀ i : Fin 20054016, parallel_generatedRoot_149 backend hashModel tape (path ++ [149, i.val]) params i ((generatedRoot.w_148_0 ctx), ((generatedRoot.w_148_1 ctx), ((generatedRoot.w_148_2 ctx), ((generatedRoot.w_148_3 ctx), ())))) (((generatedRoot.w_149_0 ctx) i), (((generatedRoot.w_149_1 ctx) i), (((generatedRoot.w_149_2 ctx) i), (((generatedRoot.w_149_3 ctx) i), ())))))
  t142 : (2607022080) = 2607022080
  h152_7 : (∀ i : Fin 2607022080, parallel_generatedRoot_152 backend hashModel tape (path ++ [152, i.val]) params i ((generatedRoot.w_6_0 ctx), (((generatedRoot.w_90_0 ctx) i), ((generatedRoot.w_149_0 ctx), ((generatedRoot.w_149_2 ctx), ((generatedRoot.w_0_0 ctx), ((generatedRoot.w_149_1 ctx), ((generatedRoot.w_149_3 ctx), ((generatedRoot.w_1_0 ctx), (((generatedRoot.w_150_0 ctx) i), (((generatedRoot.w_151_0 ctx) i), ((generatedRoot.w_2_0 ctx), ()))))))))))) (((generatedRoot.w_152_0 ctx) i), (((generatedRoot.w_152_1 ctx) i), (((generatedRoot.w_152_2 ctx) i), (((generatedRoot.w_152_3 ctx) i), (((generatedRoot.w_152_4 ctx) i), (((generatedRoot.w_152_5 ctx) i), (((generatedRoot.w_152_6 ctx) i), (((generatedRoot.w_152_7 ctx) i), ())))))))))
  t144 : (20054016) = 20054016
  h153_3 : (∀ i : Fin 20054016, parallel_generatedRoot_153 backend hashModel tape (path ++ [153, i.val]) params i ((generatedRoot.w_152_0 ctx), ((generatedRoot.w_152_1 ctx), ((generatedRoot.w_152_2 ctx), ((generatedRoot.w_152_3 ctx), ())))) (((generatedRoot.w_153_0 ctx) i), (((generatedRoot.w_153_1 ctx) i), (((generatedRoot.w_153_2 ctx) i), (((generatedRoot.w_153_3 ctx) i), ())))))
  t146 : (2607022080) = 2607022080
  h156_7 : (∀ i : Fin 2607022080, parallel_generatedRoot_156 backend hashModel tape (path ++ [156, i.val]) params i ((generatedRoot.w_6_0 ctx), (((generatedRoot.w_89_0 ctx) i), ((generatedRoot.w_153_0 ctx), ((generatedRoot.w_153_2 ctx), ((generatedRoot.w_0_0 ctx), ((generatedRoot.w_153_1 ctx), ((generatedRoot.w_153_3 ctx), ((generatedRoot.w_1_0 ctx), (((generatedRoot.w_154_0 ctx) i), (((generatedRoot.w_155_0 ctx) i), ((generatedRoot.w_2_0 ctx), ()))))))))))) (((generatedRoot.w_156_0 ctx) i), (((generatedRoot.w_156_1 ctx) i), (((generatedRoot.w_156_2 ctx) i), (((generatedRoot.w_156_3 ctx) i), (((generatedRoot.w_156_4 ctx) i), (((generatedRoot.w_156_5 ctx) i), (((generatedRoot.w_156_6 ctx) i), (((generatedRoot.w_156_7 ctx) i), ())))))))))
  t148 : (20054016) = 20054016
  h157_3 : (∀ i : Fin 20054016, parallel_generatedRoot_157 backend hashModel tape (path ++ [157, i.val]) params i ((generatedRoot.w_156_0 ctx), ((generatedRoot.w_156_1 ctx), ((generatedRoot.w_156_2 ctx), ((generatedRoot.w_156_3 ctx), ())))) (((generatedRoot.w_157_0 ctx) i), (((generatedRoot.w_157_1 ctx) i), (((generatedRoot.w_157_2 ctx) i), (((generatedRoot.w_157_3 ctx) i), ())))))
  t150 : (2607022080) = 2607022080
  h160_7 : (∀ i : Fin 2607022080, parallel_generatedRoot_160 backend hashModel tape (path ++ [160, i.val]) params i ((generatedRoot.w_6_0 ctx), (((generatedRoot.w_88_0 ctx) i), ((generatedRoot.w_157_0 ctx), ((generatedRoot.w_157_2 ctx), ((generatedRoot.w_0_0 ctx), ((generatedRoot.w_157_1 ctx), ((generatedRoot.w_157_3 ctx), ((generatedRoot.w_1_0 ctx), (((generatedRoot.w_158_0 ctx) i), (((generatedRoot.w_159_0 ctx) i), ((generatedRoot.w_2_0 ctx), ()))))))))))) (((generatedRoot.w_160_0 ctx) i), (((generatedRoot.w_160_1 ctx) i), (((generatedRoot.w_160_2 ctx) i), (((generatedRoot.w_160_3 ctx) i), (((generatedRoot.w_160_4 ctx) i), (((generatedRoot.w_160_5 ctx) i), (((generatedRoot.w_160_6 ctx) i), (((generatedRoot.w_160_7 ctx) i), ())))))))))
  t152 : (20054016) = 20054016
  h161_3 : (∀ i : Fin 20054016, parallel_generatedRoot_161 backend hashModel tape (path ++ [161, i.val]) params i ((generatedRoot.w_160_0 ctx), ((generatedRoot.w_160_1 ctx), ((generatedRoot.w_160_2 ctx), ((generatedRoot.w_160_3 ctx), ())))) (((generatedRoot.w_161_0 ctx) i), (((generatedRoot.w_161_1 ctx) i), (((generatedRoot.w_161_2 ctx) i), (((generatedRoot.w_161_3 ctx) i), ())))))
  t154 : (2607022080) = 2607022080
  h164_7 : (∀ i : Fin 2607022080, parallel_generatedRoot_164 backend hashModel tape (path ++ [164, i.val]) params i ((generatedRoot.w_6_0 ctx), (((generatedRoot.w_87_0 ctx) i), ((generatedRoot.w_161_0 ctx), ((generatedRoot.w_161_2 ctx), ((generatedRoot.w_0_0 ctx), ((generatedRoot.w_161_1 ctx), ((generatedRoot.w_161_3 ctx), ((generatedRoot.w_1_0 ctx), (((generatedRoot.w_162_0 ctx) i), (((generatedRoot.w_163_0 ctx) i), ((generatedRoot.w_2_0 ctx), ()))))))))))) (((generatedRoot.w_164_0 ctx) i), (((generatedRoot.w_164_1 ctx) i), (((generatedRoot.w_164_2 ctx) i), (((generatedRoot.w_164_3 ctx) i), (((generatedRoot.w_164_4 ctx) i), (((generatedRoot.w_164_5 ctx) i), (((generatedRoot.w_164_6 ctx) i), (((generatedRoot.w_164_7 ctx) i), ())))))))))
  t156 : (20054016) = 20054016
  h165_3 : (∀ i : Fin 20054016, parallel_generatedRoot_165 backend hashModel tape (path ++ [165, i.val]) params i ((generatedRoot.w_164_0 ctx), ((generatedRoot.w_164_1 ctx), ((generatedRoot.w_164_2 ctx), ((generatedRoot.w_164_3 ctx), ())))) (((generatedRoot.w_165_0 ctx) i), (((generatedRoot.w_165_1 ctx) i), (((generatedRoot.w_165_2 ctx) i), (((generatedRoot.w_165_3 ctx) i), ())))))
  t158 : (2607022080) = 2607022080
  h168_7 : (∀ i : Fin 2607022080, parallel_generatedRoot_168 backend hashModel tape (path ++ [168, i.val]) params i ((generatedRoot.w_6_0 ctx), (((generatedRoot.w_86_0 ctx) i), ((generatedRoot.w_165_0 ctx), ((generatedRoot.w_165_2 ctx), ((generatedRoot.w_0_0 ctx), ((generatedRoot.w_165_1 ctx), ((generatedRoot.w_165_3 ctx), ((generatedRoot.w_1_0 ctx), (((generatedRoot.w_166_0 ctx) i), (((generatedRoot.w_167_0 ctx) i), ((generatedRoot.w_2_0 ctx), ()))))))))))) (((generatedRoot.w_168_0 ctx) i), (((generatedRoot.w_168_1 ctx) i), (((generatedRoot.w_168_2 ctx) i), (((generatedRoot.w_168_3 ctx) i), (((generatedRoot.w_168_4 ctx) i), (((generatedRoot.w_168_5 ctx) i), (((generatedRoot.w_168_6 ctx) i), (((generatedRoot.w_168_7 ctx) i), ())))))))))

theorem generatedRoot.facts_4  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_4 backend hashModel tape path params (generatedRoot.w_0_0 ctx) (generatedRoot.w_1_0 ctx) (generatedRoot.w_6_0 ctx) (generatedRoot.w_86_0 ctx) (generatedRoot.w_87_0 ctx) (generatedRoot.w_88_0 ctx) (generatedRoot.w_89_0 ctx) (generatedRoot.w_90_0 ctx) (generatedRoot.w_91_0 ctx) (generatedRoot.w_92_0 ctx) (generatedRoot.w_93_0 ctx) (generatedRoot.w_138_0 ctx) (generatedRoot.w_139_0 ctx) (generatedRoot.w_142_0 ctx) (generatedRoot.w_143_0 ctx) (generatedRoot.w_146_0 ctx) (generatedRoot.w_147_0 ctx) (generatedRoot.w_150_0 ctx) (generatedRoot.w_151_0 ctx) (generatedRoot.w_154_0 ctx) (generatedRoot.w_155_0 ctx) (generatedRoot.w_158_0 ctx) (generatedRoot.w_159_0 ctx) (generatedRoot.w_162_0 ctx) (generatedRoot.w_163_0 ctx) (generatedRoot.w_166_0 ctx) (generatedRoot.w_167_0 ctx) (generatedRoot.w_2_0 ctx) (generatedRoot.w_136_0 ctx) (generatedRoot.w_136_1 ctx) (generatedRoot.w_136_2 ctx) (generatedRoot.w_136_3 ctx) (generatedRoot.w_137_0 ctx) (generatedRoot.w_137_1 ctx) (generatedRoot.w_137_2 ctx) (generatedRoot.w_137_3 ctx) (generatedRoot.w_140_0 ctx) (generatedRoot.w_140_1 ctx) (generatedRoot.w_140_2 ctx) (generatedRoot.w_140_3 ctx) (generatedRoot.w_140_4 ctx) (generatedRoot.w_140_5 ctx) (generatedRoot.w_140_6 ctx) (generatedRoot.w_140_7 ctx) (generatedRoot.w_141_0 ctx) (generatedRoot.w_141_1 ctx) (generatedRoot.w_141_2 ctx) (generatedRoot.w_141_3 ctx) (generatedRoot.w_144_0 ctx) (generatedRoot.w_144_1 ctx) (generatedRoot.w_144_2 ctx) (generatedRoot.w_144_3 ctx) (generatedRoot.w_144_4 ctx) (generatedRoot.w_144_5 ctx) (generatedRoot.w_144_6 ctx) (generatedRoot.w_144_7 ctx) (generatedRoot.w_145_0 ctx) (generatedRoot.w_145_1 ctx) (generatedRoot.w_145_2 ctx) (generatedRoot.w_145_3 ctx) (generatedRoot.w_148_0 ctx) (generatedRoot.w_148_1 ctx) (generatedRoot.w_148_2 ctx) (generatedRoot.w_148_3 ctx) (generatedRoot.w_148_4 ctx) (generatedRoot.w_148_5 ctx) (generatedRoot.w_148_6 ctx) (generatedRoot.w_148_7 ctx) (generatedRoot.w_149_0 ctx) (generatedRoot.w_149_1 ctx) (generatedRoot.w_149_2 ctx) (generatedRoot.w_149_3 ctx) (generatedRoot.w_152_0 ctx) (generatedRoot.w_152_1 ctx) (generatedRoot.w_152_2 ctx) (generatedRoot.w_152_3 ctx) (generatedRoot.w_152_4 ctx) (generatedRoot.w_152_5 ctx) (generatedRoot.w_152_6 ctx) (generatedRoot.w_152_7 ctx) (generatedRoot.w_153_0 ctx) (generatedRoot.w_153_1 ctx) (generatedRoot.w_153_2 ctx) (generatedRoot.w_153_3 ctx) (generatedRoot.w_156_0 ctx) (generatedRoot.w_156_1 ctx) (generatedRoot.w_156_2 ctx) (generatedRoot.w_156_3 ctx) (generatedRoot.w_156_4 ctx) (generatedRoot.w_156_5 ctx) (generatedRoot.w_156_6 ctx) (generatedRoot.w_156_7 ctx) (generatedRoot.w_157_0 ctx) (generatedRoot.w_157_1 ctx) (generatedRoot.w_157_2 ctx) (generatedRoot.w_157_3 ctx) (generatedRoot.w_160_0 ctx) (generatedRoot.w_160_1 ctx) (generatedRoot.w_160_2 ctx) (generatedRoot.w_160_3 ctx) (generatedRoot.w_160_4 ctx) (generatedRoot.w_160_5 ctx) (generatedRoot.w_160_6 ctx) (generatedRoot.w_160_7 ctx) (generatedRoot.w_161_0 ctx) (generatedRoot.w_161_1 ctx) (generatedRoot.w_161_2 ctx) (generatedRoot.w_161_3 ctx) (generatedRoot.w_164_0 ctx) (generatedRoot.w_164_1 ctx) (generatedRoot.w_164_2 ctx) (generatedRoot.w_164_3 ctx) (generatedRoot.w_164_4 ctx) (generatedRoot.w_164_5 ctx) (generatedRoot.w_164_6 ctx) (generatedRoot.w_164_7 ctx) (generatedRoot.w_165_0 ctx) (generatedRoot.w_165_1 ctx) (generatedRoot.w_165_2 ctx) (generatedRoot.w_165_3 ctx) (generatedRoot.w_168_0 ctx) (generatedRoot.w_168_1 ctx) (generatedRoot.w_168_2 ctx) (generatedRoot.w_168_3 ctx) (generatedRoot.w_168_4 ctx) (generatedRoot.w_168_5 ctx) (generatedRoot.w_168_6 ctx) (generatedRoot.w_168_7 ctx)) :
    generatedRoot.Facts_4 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_5  (ctx : generatedRootCtx) : Prop where
  t160 : (20054016) = 20054016
  h169_3 : (∀ i : Fin 20054016, parallel_generatedRoot_169 backend hashModel tape (path ++ [169, i.val]) params i ((generatedRoot.w_168_0 ctx), ((generatedRoot.w_168_1 ctx), ((generatedRoot.w_168_2 ctx), ((generatedRoot.w_168_3 ctx), ())))) (((generatedRoot.w_169_0 ctx) i), (((generatedRoot.w_169_1 ctx) i), (((generatedRoot.w_169_2 ctx) i), (((generatedRoot.w_169_3 ctx) i), ())))))
  t162 : (2607022080) = 2607022080
  h172_7 : (∀ i : Fin 2607022080, parallel_generatedRoot_172 backend hashModel tape (path ++ [172, i.val]) params i ((generatedRoot.w_6_0 ctx), (((generatedRoot.w_85_0 ctx) i), ((generatedRoot.w_169_0 ctx), ((generatedRoot.w_169_2 ctx), ((generatedRoot.w_0_0 ctx), ((generatedRoot.w_169_1 ctx), ((generatedRoot.w_169_3 ctx), ((generatedRoot.w_1_0 ctx), (((generatedRoot.w_170_0 ctx) i), (((generatedRoot.w_171_0 ctx) i), ((generatedRoot.w_2_0 ctx), ()))))))))))) (((generatedRoot.w_172_0 ctx) i), (((generatedRoot.w_172_1 ctx) i), (((generatedRoot.w_172_2 ctx) i), (((generatedRoot.w_172_3 ctx) i), (((generatedRoot.w_172_4 ctx) i), (((generatedRoot.w_172_5 ctx) i), (((generatedRoot.w_172_6 ctx) i), (((generatedRoot.w_172_7 ctx) i), ())))))))))
  t164 : (20054016) = 20054016
  h173_3 : (∀ i : Fin 20054016, parallel_generatedRoot_173 backend hashModel tape (path ++ [173, i.val]) params i ((generatedRoot.w_172_0 ctx), ((generatedRoot.w_172_1 ctx), ((generatedRoot.w_172_2 ctx), ((generatedRoot.w_172_3 ctx), ())))) (((generatedRoot.w_173_0 ctx) i), (((generatedRoot.w_173_1 ctx) i), (((generatedRoot.w_173_2 ctx) i), (((generatedRoot.w_173_3 ctx) i), ())))))
  t166 : (2607022080) = 2607022080
  h176_7 : (∀ i : Fin 2607022080, parallel_generatedRoot_176 backend hashModel tape (path ++ [176, i.val]) params i ((generatedRoot.w_6_0 ctx), (((generatedRoot.w_84_0 ctx) i), ((generatedRoot.w_173_0 ctx), ((generatedRoot.w_173_2 ctx), ((generatedRoot.w_0_0 ctx), ((generatedRoot.w_173_1 ctx), ((generatedRoot.w_173_3 ctx), ((generatedRoot.w_1_0 ctx), (((generatedRoot.w_174_0 ctx) i), (((generatedRoot.w_175_0 ctx) i), ((generatedRoot.w_2_0 ctx), ()))))))))))) (((generatedRoot.w_176_0 ctx) i), (((generatedRoot.w_176_1 ctx) i), (((generatedRoot.w_176_2 ctx) i), (((generatedRoot.w_176_3 ctx) i), (((generatedRoot.w_176_4 ctx) i), (((generatedRoot.w_176_5 ctx) i), (((generatedRoot.w_176_6 ctx) i), (((generatedRoot.w_176_7 ctx) i), ())))))))))
  t168 : (20054016) = 20054016
  h177_3 : (∀ i : Fin 20054016, parallel_generatedRoot_177 backend hashModel tape (path ++ [177, i.val]) params i ((generatedRoot.w_176_0 ctx), ((generatedRoot.w_176_1 ctx), ((generatedRoot.w_176_2 ctx), ((generatedRoot.w_176_3 ctx), ())))) (((generatedRoot.w_177_0 ctx) i), (((generatedRoot.w_177_1 ctx) i), (((generatedRoot.w_177_2 ctx) i), (((generatedRoot.w_177_3 ctx) i), ())))))
  t170 : (2607022080) = 2607022080
  h180_7 : (∀ i : Fin 2607022080, parallel_generatedRoot_180 backend hashModel tape (path ++ [180, i.val]) params i ((generatedRoot.w_6_0 ctx), (((generatedRoot.w_83_0 ctx) i), ((generatedRoot.w_177_0 ctx), ((generatedRoot.w_177_2 ctx), ((generatedRoot.w_0_0 ctx), ((generatedRoot.w_177_1 ctx), ((generatedRoot.w_177_3 ctx), ((generatedRoot.w_1_0 ctx), (((generatedRoot.w_178_0 ctx) i), (((generatedRoot.w_179_0 ctx) i), ((generatedRoot.w_2_0 ctx), ()))))))))))) (((generatedRoot.w_180_0 ctx) i), (((generatedRoot.w_180_1 ctx) i), (((generatedRoot.w_180_2 ctx) i), (((generatedRoot.w_180_3 ctx) i), (((generatedRoot.w_180_4 ctx) i), (((generatedRoot.w_180_5 ctx) i), (((generatedRoot.w_180_6 ctx) i), (((generatedRoot.w_180_7 ctx) i), ())))))))))
  t172 : (20054016) = 20054016
  h181_3 : (∀ i : Fin 20054016, parallel_generatedRoot_181 backend hashModel tape (path ++ [181, i.val]) params i ((generatedRoot.w_180_0 ctx), ((generatedRoot.w_180_1 ctx), ((generatedRoot.w_180_2 ctx), ((generatedRoot.w_180_3 ctx), ())))) (((generatedRoot.w_181_0 ctx) i), (((generatedRoot.w_181_1 ctx) i), (((generatedRoot.w_181_2 ctx) i), (((generatedRoot.w_181_3 ctx) i), ())))))
  t174 : (2607022080) = 2607022080
  h184_7 : (∀ i : Fin 2607022080, parallel_generatedRoot_184 backend hashModel tape (path ++ [184, i.val]) params i ((generatedRoot.w_6_0 ctx), (((generatedRoot.w_82_0 ctx) i), ((generatedRoot.w_181_0 ctx), ((generatedRoot.w_181_2 ctx), ((generatedRoot.w_0_0 ctx), ((generatedRoot.w_181_1 ctx), ((generatedRoot.w_181_3 ctx), ((generatedRoot.w_1_0 ctx), (((generatedRoot.w_182_0 ctx) i), (((generatedRoot.w_183_0 ctx) i), ((generatedRoot.w_2_0 ctx), ()))))))))))) (((generatedRoot.w_184_0 ctx) i), (((generatedRoot.w_184_1 ctx) i), (((generatedRoot.w_184_2 ctx) i), (((generatedRoot.w_184_3 ctx) i), (((generatedRoot.w_184_4 ctx) i), (((generatedRoot.w_184_5 ctx) i), (((generatedRoot.w_184_6 ctx) i), (((generatedRoot.w_184_7 ctx) i), ())))))))))
  t176 : (20054016) = 20054016
  h185_3 : (∀ i : Fin 20054016, parallel_generatedRoot_185 backend hashModel tape (path ++ [185, i.val]) params i ((generatedRoot.w_184_0 ctx), ((generatedRoot.w_184_1 ctx), ((generatedRoot.w_184_2 ctx), ((generatedRoot.w_184_3 ctx), ())))) (((generatedRoot.w_185_0 ctx) i), (((generatedRoot.w_185_1 ctx) i), (((generatedRoot.w_185_2 ctx) i), (((generatedRoot.w_185_3 ctx) i), ())))))
  t178 : (2607022080) = 2607022080
  h188_7 : (∀ i : Fin 2607022080, parallel_generatedRoot_188 backend hashModel tape (path ++ [188, i.val]) params i ((generatedRoot.w_6_0 ctx), (((generatedRoot.w_81_0 ctx) i), ((generatedRoot.w_185_0 ctx), ((generatedRoot.w_185_2 ctx), ((generatedRoot.w_0_0 ctx), ((generatedRoot.w_185_1 ctx), ((generatedRoot.w_185_3 ctx), ((generatedRoot.w_1_0 ctx), (((generatedRoot.w_186_0 ctx) i), (((generatedRoot.w_187_0 ctx) i), ((generatedRoot.w_2_0 ctx), ()))))))))))) (((generatedRoot.w_188_0 ctx) i), (((generatedRoot.w_188_1 ctx) i), (((generatedRoot.w_188_2 ctx) i), (((generatedRoot.w_188_3 ctx) i), (((generatedRoot.w_188_4 ctx) i), (((generatedRoot.w_188_5 ctx) i), (((generatedRoot.w_188_6 ctx) i), (((generatedRoot.w_188_7 ctx) i), ())))))))))
  t180 : (20054016) = 20054016
  h189_3 : (∀ i : Fin 20054016, parallel_generatedRoot_189 backend hashModel tape (path ++ [189, i.val]) params i ((generatedRoot.w_188_0 ctx), ((generatedRoot.w_188_1 ctx), ((generatedRoot.w_188_2 ctx), ((generatedRoot.w_188_3 ctx), ())))) (((generatedRoot.w_189_0 ctx) i), (((generatedRoot.w_189_1 ctx) i), (((generatedRoot.w_189_2 ctx) i), (((generatedRoot.w_189_3 ctx) i), ())))))
  t182 : (2607022080) = 2607022080
  h192_7 : (∀ i : Fin 2607022080, parallel_generatedRoot_192 backend hashModel tape (path ++ [192, i.val]) params i ((generatedRoot.w_6_0 ctx), (((generatedRoot.w_80_0 ctx) i), ((generatedRoot.w_189_0 ctx), ((generatedRoot.w_189_2 ctx), ((generatedRoot.w_0_0 ctx), ((generatedRoot.w_189_1 ctx), ((generatedRoot.w_189_3 ctx), ((generatedRoot.w_1_0 ctx), (((generatedRoot.w_190_0 ctx) i), (((generatedRoot.w_191_0 ctx) i), ((generatedRoot.w_2_0 ctx), ()))))))))))) (((generatedRoot.w_192_0 ctx) i), (((generatedRoot.w_192_1 ctx) i), (((generatedRoot.w_192_2 ctx) i), (((generatedRoot.w_192_3 ctx) i), (((generatedRoot.w_192_4 ctx) i), (((generatedRoot.w_192_5 ctx) i), (((generatedRoot.w_192_6 ctx) i), (((generatedRoot.w_192_7 ctx) i), ())))))))))
  t184 : (20054016) = 20054016
  h193_3 : (∀ i : Fin 20054016, parallel_generatedRoot_193 backend hashModel tape (path ++ [193, i.val]) params i ((generatedRoot.w_192_0 ctx), ((generatedRoot.w_192_1 ctx), ((generatedRoot.w_192_2 ctx), ((generatedRoot.w_192_3 ctx), ())))) (((generatedRoot.w_193_0 ctx) i), (((generatedRoot.w_193_1 ctx) i), (((generatedRoot.w_193_2 ctx) i), (((generatedRoot.w_193_3 ctx) i), ())))))
  t186 : (2607022080) = 2607022080
  h196_7 : (∀ i : Fin 2607022080, parallel_generatedRoot_196 backend hashModel tape (path ++ [196, i.val]) params i ((generatedRoot.w_6_0 ctx), (((generatedRoot.w_79_0 ctx) i), ((generatedRoot.w_193_0 ctx), ((generatedRoot.w_193_2 ctx), ((generatedRoot.w_0_0 ctx), ((generatedRoot.w_193_1 ctx), ((generatedRoot.w_193_3 ctx), ((generatedRoot.w_1_0 ctx), (((generatedRoot.w_194_0 ctx) i), (((generatedRoot.w_195_0 ctx) i), ((generatedRoot.w_2_0 ctx), ()))))))))))) (((generatedRoot.w_196_0 ctx) i), (((generatedRoot.w_196_1 ctx) i), (((generatedRoot.w_196_2 ctx) i), (((generatedRoot.w_196_3 ctx) i), (((generatedRoot.w_196_4 ctx) i), (((generatedRoot.w_196_5 ctx) i), (((generatedRoot.w_196_6 ctx) i), (((generatedRoot.w_196_7 ctx) i), ())))))))))
  t188 : (20054016) = 20054016
  h197_3 : (∀ i : Fin 20054016, parallel_generatedRoot_197 backend hashModel tape (path ++ [197, i.val]) params i ((generatedRoot.w_196_0 ctx), ((generatedRoot.w_196_1 ctx), ((generatedRoot.w_196_2 ctx), ((generatedRoot.w_196_3 ctx), ())))) (((generatedRoot.w_197_0 ctx) i), (((generatedRoot.w_197_1 ctx) i), (((generatedRoot.w_197_2 ctx) i), (((generatedRoot.w_197_3 ctx) i), ())))))
  t190 : (2607022080) = 2607022080
  h200_7 : (∀ i : Fin 2607022080, parallel_generatedRoot_200 backend hashModel tape (path ++ [200, i.val]) params i ((generatedRoot.w_6_0 ctx), (((generatedRoot.w_78_0 ctx) i), ((generatedRoot.w_197_0 ctx), ((generatedRoot.w_197_2 ctx), ((generatedRoot.w_0_0 ctx), ((generatedRoot.w_197_1 ctx), ((generatedRoot.w_197_3 ctx), ((generatedRoot.w_1_0 ctx), (((generatedRoot.w_198_0 ctx) i), (((generatedRoot.w_199_0 ctx) i), ((generatedRoot.w_2_0 ctx), ()))))))))))) (((generatedRoot.w_200_0 ctx) i), (((generatedRoot.w_200_1 ctx) i), (((generatedRoot.w_200_2 ctx) i), (((generatedRoot.w_200_3 ctx) i), (((generatedRoot.w_200_4 ctx) i), (((generatedRoot.w_200_5 ctx) i), (((generatedRoot.w_200_6 ctx) i), (((generatedRoot.w_200_7 ctx) i), ())))))))))

theorem generatedRoot.facts_5  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_5 backend hashModel tape path params (generatedRoot.w_0_0 ctx) (generatedRoot.w_1_0 ctx) (generatedRoot.w_6_0 ctx) (generatedRoot.w_78_0 ctx) (generatedRoot.w_79_0 ctx) (generatedRoot.w_80_0 ctx) (generatedRoot.w_81_0 ctx) (generatedRoot.w_82_0 ctx) (generatedRoot.w_83_0 ctx) (generatedRoot.w_84_0 ctx) (generatedRoot.w_85_0 ctx) (generatedRoot.w_170_0 ctx) (generatedRoot.w_171_0 ctx) (generatedRoot.w_174_0 ctx) (generatedRoot.w_175_0 ctx) (generatedRoot.w_178_0 ctx) (generatedRoot.w_179_0 ctx) (generatedRoot.w_182_0 ctx) (generatedRoot.w_183_0 ctx) (generatedRoot.w_186_0 ctx) (generatedRoot.w_187_0 ctx) (generatedRoot.w_190_0 ctx) (generatedRoot.w_191_0 ctx) (generatedRoot.w_194_0 ctx) (generatedRoot.w_195_0 ctx) (generatedRoot.w_198_0 ctx) (generatedRoot.w_199_0 ctx) (generatedRoot.w_2_0 ctx) (generatedRoot.w_168_0 ctx) (generatedRoot.w_168_1 ctx) (generatedRoot.w_168_2 ctx) (generatedRoot.w_168_3 ctx) (generatedRoot.w_169_0 ctx) (generatedRoot.w_169_1 ctx) (generatedRoot.w_169_2 ctx) (generatedRoot.w_169_3 ctx) (generatedRoot.w_172_0 ctx) (generatedRoot.w_172_1 ctx) (generatedRoot.w_172_2 ctx) (generatedRoot.w_172_3 ctx) (generatedRoot.w_172_4 ctx) (generatedRoot.w_172_5 ctx) (generatedRoot.w_172_6 ctx) (generatedRoot.w_172_7 ctx) (generatedRoot.w_173_0 ctx) (generatedRoot.w_173_1 ctx) (generatedRoot.w_173_2 ctx) (generatedRoot.w_173_3 ctx) (generatedRoot.w_176_0 ctx) (generatedRoot.w_176_1 ctx) (generatedRoot.w_176_2 ctx) (generatedRoot.w_176_3 ctx) (generatedRoot.w_176_4 ctx) (generatedRoot.w_176_5 ctx) (generatedRoot.w_176_6 ctx) (generatedRoot.w_176_7 ctx) (generatedRoot.w_177_0 ctx) (generatedRoot.w_177_1 ctx) (generatedRoot.w_177_2 ctx) (generatedRoot.w_177_3 ctx) (generatedRoot.w_180_0 ctx) (generatedRoot.w_180_1 ctx) (generatedRoot.w_180_2 ctx) (generatedRoot.w_180_3 ctx) (generatedRoot.w_180_4 ctx) (generatedRoot.w_180_5 ctx) (generatedRoot.w_180_6 ctx) (generatedRoot.w_180_7 ctx) (generatedRoot.w_181_0 ctx) (generatedRoot.w_181_1 ctx) (generatedRoot.w_181_2 ctx) (generatedRoot.w_181_3 ctx) (generatedRoot.w_184_0 ctx) (generatedRoot.w_184_1 ctx) (generatedRoot.w_184_2 ctx) (generatedRoot.w_184_3 ctx) (generatedRoot.w_184_4 ctx) (generatedRoot.w_184_5 ctx) (generatedRoot.w_184_6 ctx) (generatedRoot.w_184_7 ctx) (generatedRoot.w_185_0 ctx) (generatedRoot.w_185_1 ctx) (generatedRoot.w_185_2 ctx) (generatedRoot.w_185_3 ctx) (generatedRoot.w_188_0 ctx) (generatedRoot.w_188_1 ctx) (generatedRoot.w_188_2 ctx) (generatedRoot.w_188_3 ctx) (generatedRoot.w_188_4 ctx) (generatedRoot.w_188_5 ctx) (generatedRoot.w_188_6 ctx) (generatedRoot.w_188_7 ctx) (generatedRoot.w_189_0 ctx) (generatedRoot.w_189_1 ctx) (generatedRoot.w_189_2 ctx) (generatedRoot.w_189_3 ctx) (generatedRoot.w_192_0 ctx) (generatedRoot.w_192_1 ctx) (generatedRoot.w_192_2 ctx) (generatedRoot.w_192_3 ctx) (generatedRoot.w_192_4 ctx) (generatedRoot.w_192_5 ctx) (generatedRoot.w_192_6 ctx) (generatedRoot.w_192_7 ctx) (generatedRoot.w_193_0 ctx) (generatedRoot.w_193_1 ctx) (generatedRoot.w_193_2 ctx) (generatedRoot.w_193_3 ctx) (generatedRoot.w_196_0 ctx) (generatedRoot.w_196_1 ctx) (generatedRoot.w_196_2 ctx) (generatedRoot.w_196_3 ctx) (generatedRoot.w_196_4 ctx) (generatedRoot.w_196_5 ctx) (generatedRoot.w_196_6 ctx) (generatedRoot.w_196_7 ctx) (generatedRoot.w_197_0 ctx) (generatedRoot.w_197_1 ctx) (generatedRoot.w_197_2 ctx) (generatedRoot.w_197_3 ctx) (generatedRoot.w_200_0 ctx) (generatedRoot.w_200_1 ctx) (generatedRoot.w_200_2 ctx) (generatedRoot.w_200_3 ctx) (generatedRoot.w_200_4 ctx) (generatedRoot.w_200_5 ctx) (generatedRoot.w_200_6 ctx) (generatedRoot.w_200_7 ctx)) :
    generatedRoot.Facts_5 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_6  (ctx : generatedRootCtx) : Prop where
  t192 : (20054016) = 20054016
  h201_3 : (∀ i : Fin 20054016, parallel_generatedRoot_201 backend hashModel tape (path ++ [201, i.val]) params i ((generatedRoot.w_200_0 ctx), ((generatedRoot.w_200_1 ctx), ((generatedRoot.w_200_2 ctx), ((generatedRoot.w_200_3 ctx), ())))) (((generatedRoot.w_201_0 ctx) i), (((generatedRoot.w_201_1 ctx) i), (((generatedRoot.w_201_2 ctx) i), (((generatedRoot.w_201_3 ctx) i), ())))))
  t194 : (2607022080) = 2607022080
  h204_7 : (∀ i : Fin 2607022080, parallel_generatedRoot_204 backend hashModel tape (path ++ [204, i.val]) params i ((generatedRoot.w_6_0 ctx), (((generatedRoot.w_77_0 ctx) i), ((generatedRoot.w_201_0 ctx), ((generatedRoot.w_201_2 ctx), ((generatedRoot.w_0_0 ctx), ((generatedRoot.w_201_1 ctx), ((generatedRoot.w_201_3 ctx), ((generatedRoot.w_1_0 ctx), (((generatedRoot.w_202_0 ctx) i), (((generatedRoot.w_203_0 ctx) i), ((generatedRoot.w_2_0 ctx), ()))))))))))) (((generatedRoot.w_204_0 ctx) i), (((generatedRoot.w_204_1 ctx) i), (((generatedRoot.w_204_2 ctx) i), (((generatedRoot.w_204_3 ctx) i), (((generatedRoot.w_204_4 ctx) i), (((generatedRoot.w_204_5 ctx) i), (((generatedRoot.w_204_6 ctx) i), (((generatedRoot.w_204_7 ctx) i), ())))))))))
  t196 : (20054016) = 20054016
  h205_3 : (∀ i : Fin 20054016, parallel_generatedRoot_205 backend hashModel tape (path ++ [205, i.val]) params i ((generatedRoot.w_204_0 ctx), ((generatedRoot.w_204_1 ctx), ((generatedRoot.w_204_2 ctx), ((generatedRoot.w_204_3 ctx), ())))) (((generatedRoot.w_205_0 ctx) i), (((generatedRoot.w_205_1 ctx) i), (((generatedRoot.w_205_2 ctx) i), (((generatedRoot.w_205_3 ctx) i), ())))))
  t198 : (2607022080) = 2607022080
  h208_7 : (∀ i : Fin 2607022080, parallel_generatedRoot_208 backend hashModel tape (path ++ [208, i.val]) params i ((generatedRoot.w_6_0 ctx), (((generatedRoot.w_76_0 ctx) i), ((generatedRoot.w_205_0 ctx), ((generatedRoot.w_205_2 ctx), ((generatedRoot.w_0_0 ctx), ((generatedRoot.w_205_1 ctx), ((generatedRoot.w_205_3 ctx), ((generatedRoot.w_1_0 ctx), (((generatedRoot.w_206_0 ctx) i), (((generatedRoot.w_207_0 ctx) i), ((generatedRoot.w_2_0 ctx), ()))))))))))) (((generatedRoot.w_208_0 ctx) i), (((generatedRoot.w_208_1 ctx) i), (((generatedRoot.w_208_2 ctx) i), (((generatedRoot.w_208_3 ctx) i), (((generatedRoot.w_208_4 ctx) i), (((generatedRoot.w_208_5 ctx) i), (((generatedRoot.w_208_6 ctx) i), (((generatedRoot.w_208_7 ctx) i), ())))))))))
  t200 : (20054016) = 20054016
  h209_3 : (∀ i : Fin 20054016, parallel_generatedRoot_209 backend hashModel tape (path ++ [209, i.val]) params i ((generatedRoot.w_208_0 ctx), ((generatedRoot.w_208_1 ctx), ((generatedRoot.w_208_2 ctx), ((generatedRoot.w_208_3 ctx), ())))) (((generatedRoot.w_209_0 ctx) i), (((generatedRoot.w_209_1 ctx) i), (((generatedRoot.w_209_2 ctx) i), (((generatedRoot.w_209_3 ctx) i), ())))))
  t202 : (2607022080) = 2607022080
  h212_7 : (∀ i : Fin 2607022080, parallel_generatedRoot_212 backend hashModel tape (path ++ [212, i.val]) params i ((generatedRoot.w_6_0 ctx), (((generatedRoot.w_75_0 ctx) i), ((generatedRoot.w_209_0 ctx), ((generatedRoot.w_209_2 ctx), ((generatedRoot.w_0_0 ctx), ((generatedRoot.w_209_1 ctx), ((generatedRoot.w_209_3 ctx), ((generatedRoot.w_1_0 ctx), (((generatedRoot.w_210_0 ctx) i), (((generatedRoot.w_211_0 ctx) i), ((generatedRoot.w_2_0 ctx), ()))))))))))) (((generatedRoot.w_212_0 ctx) i), (((generatedRoot.w_212_1 ctx) i), (((generatedRoot.w_212_2 ctx) i), (((generatedRoot.w_212_3 ctx) i), (((generatedRoot.w_212_4 ctx) i), (((generatedRoot.w_212_5 ctx) i), (((generatedRoot.w_212_6 ctx) i), (((generatedRoot.w_212_7 ctx) i), ())))))))))
  t204 : (20054016) = 20054016
  h213_3 : (∀ i : Fin 20054016, parallel_generatedRoot_213 backend hashModel tape (path ++ [213, i.val]) params i ((generatedRoot.w_212_0 ctx), ((generatedRoot.w_212_1 ctx), ((generatedRoot.w_212_2 ctx), ((generatedRoot.w_212_3 ctx), ())))) (((generatedRoot.w_213_0 ctx) i), (((generatedRoot.w_213_1 ctx) i), (((generatedRoot.w_213_2 ctx) i), (((generatedRoot.w_213_3 ctx) i), ())))))
  t206 : (2607022080) = 2607022080
  h216_7 : (∀ i : Fin 2607022080, parallel_generatedRoot_216 backend hashModel tape (path ++ [216, i.val]) params i ((generatedRoot.w_6_0 ctx), (((generatedRoot.w_74_0 ctx) i), ((generatedRoot.w_213_0 ctx), ((generatedRoot.w_213_2 ctx), ((generatedRoot.w_0_0 ctx), ((generatedRoot.w_213_1 ctx), ((generatedRoot.w_213_3 ctx), ((generatedRoot.w_1_0 ctx), (((generatedRoot.w_214_0 ctx) i), (((generatedRoot.w_215_0 ctx) i), ((generatedRoot.w_2_0 ctx), ()))))))))))) (((generatedRoot.w_216_0 ctx) i), (((generatedRoot.w_216_1 ctx) i), (((generatedRoot.w_216_2 ctx) i), (((generatedRoot.w_216_3 ctx) i), (((generatedRoot.w_216_4 ctx) i), (((generatedRoot.w_216_5 ctx) i), (((generatedRoot.w_216_6 ctx) i), (((generatedRoot.w_216_7 ctx) i), ())))))))))
  t208 : (20054016) = 20054016
  h217_3 : (∀ i : Fin 20054016, parallel_generatedRoot_217 backend hashModel tape (path ++ [217, i.val]) params i ((generatedRoot.w_216_0 ctx), ((generatedRoot.w_216_1 ctx), ((generatedRoot.w_216_2 ctx), ((generatedRoot.w_216_3 ctx), ())))) (((generatedRoot.w_217_0 ctx) i), (((generatedRoot.w_217_1 ctx) i), (((generatedRoot.w_217_2 ctx) i), (((generatedRoot.w_217_3 ctx) i), ())))))
  t210 : (2607022080) = 2607022080
  h220_7 : (∀ i : Fin 2607022080, parallel_generatedRoot_220 backend hashModel tape (path ++ [220, i.val]) params i ((generatedRoot.w_6_0 ctx), (((generatedRoot.w_73_0 ctx) i), ((generatedRoot.w_217_0 ctx), ((generatedRoot.w_217_2 ctx), ((generatedRoot.w_0_0 ctx), ((generatedRoot.w_217_1 ctx), ((generatedRoot.w_217_3 ctx), ((generatedRoot.w_1_0 ctx), (((generatedRoot.w_218_0 ctx) i), (((generatedRoot.w_219_0 ctx) i), ((generatedRoot.w_2_0 ctx), ()))))))))))) (((generatedRoot.w_220_0 ctx) i), (((generatedRoot.w_220_1 ctx) i), (((generatedRoot.w_220_2 ctx) i), (((generatedRoot.w_220_3 ctx) i), (((generatedRoot.w_220_4 ctx) i), (((generatedRoot.w_220_5 ctx) i), (((generatedRoot.w_220_6 ctx) i), (((generatedRoot.w_220_7 ctx) i), ())))))))))
  t212 : (20054016) = 20054016
  h221_3 : (∀ i : Fin 20054016, parallel_generatedRoot_221 backend hashModel tape (path ++ [221, i.val]) params i ((generatedRoot.w_220_0 ctx), ((generatedRoot.w_220_1 ctx), ((generatedRoot.w_220_2 ctx), ((generatedRoot.w_220_3 ctx), ())))) (((generatedRoot.w_221_0 ctx) i), (((generatedRoot.w_221_1 ctx) i), (((generatedRoot.w_221_2 ctx) i), (((generatedRoot.w_221_3 ctx) i), ())))))
  t214 : (2607022080) = 2607022080
  h224_7 : (∀ i : Fin 2607022080, parallel_generatedRoot_224 backend hashModel tape (path ++ [224, i.val]) params i ((generatedRoot.w_6_0 ctx), (((generatedRoot.w_72_0 ctx) i), ((generatedRoot.w_221_0 ctx), ((generatedRoot.w_221_2 ctx), ((generatedRoot.w_0_0 ctx), ((generatedRoot.w_221_1 ctx), ((generatedRoot.w_221_3 ctx), ((generatedRoot.w_1_0 ctx), (((generatedRoot.w_222_0 ctx) i), (((generatedRoot.w_223_0 ctx) i), ((generatedRoot.w_2_0 ctx), ()))))))))))) (((generatedRoot.w_224_0 ctx) i), (((generatedRoot.w_224_1 ctx) i), (((generatedRoot.w_224_2 ctx) i), (((generatedRoot.w_224_3 ctx) i), (((generatedRoot.w_224_4 ctx) i), (((generatedRoot.w_224_5 ctx) i), (((generatedRoot.w_224_6 ctx) i), (((generatedRoot.w_224_7 ctx) i), ())))))))))
  t216 : (20054016) = 20054016
  h225_3 : (∀ i : Fin 20054016, parallel_generatedRoot_225 backend hashModel tape (path ++ [225, i.val]) params i ((generatedRoot.w_224_0 ctx), ((generatedRoot.w_224_1 ctx), ((generatedRoot.w_224_2 ctx), ((generatedRoot.w_224_3 ctx), ())))) (((generatedRoot.w_225_0 ctx) i), (((generatedRoot.w_225_1 ctx) i), (((generatedRoot.w_225_2 ctx) i), (((generatedRoot.w_225_3 ctx) i), ())))))
  t218 : (2607022080) = 2607022080
  h228_7 : (∀ i : Fin 2607022080, parallel_generatedRoot_228 backend hashModel tape (path ++ [228, i.val]) params i ((generatedRoot.w_6_0 ctx), (((generatedRoot.w_71_0 ctx) i), ((generatedRoot.w_225_0 ctx), ((generatedRoot.w_225_2 ctx), ((generatedRoot.w_0_0 ctx), ((generatedRoot.w_225_1 ctx), ((generatedRoot.w_225_3 ctx), ((generatedRoot.w_1_0 ctx), (((generatedRoot.w_226_0 ctx) i), (((generatedRoot.w_227_0 ctx) i), ((generatedRoot.w_2_0 ctx), ()))))))))))) (((generatedRoot.w_228_0 ctx) i), (((generatedRoot.w_228_1 ctx) i), (((generatedRoot.w_228_2 ctx) i), (((generatedRoot.w_228_3 ctx) i), (((generatedRoot.w_228_4 ctx) i), (((generatedRoot.w_228_5 ctx) i), (((generatedRoot.w_228_6 ctx) i), (((generatedRoot.w_228_7 ctx) i), ())))))))))
  t220 : (20054016) = 20054016
  h229_3 : (∀ i : Fin 20054016, parallel_generatedRoot_229 backend hashModel tape (path ++ [229, i.val]) params i ((generatedRoot.w_228_0 ctx), ((generatedRoot.w_228_1 ctx), ((generatedRoot.w_228_2 ctx), ((generatedRoot.w_228_3 ctx), ())))) (((generatedRoot.w_229_0 ctx) i), (((generatedRoot.w_229_1 ctx) i), (((generatedRoot.w_229_2 ctx) i), (((generatedRoot.w_229_3 ctx) i), ())))))
  t222 : (2607022080) = 2607022080
  h232_7 : (∀ i : Fin 2607022080, parallel_generatedRoot_232 backend hashModel tape (path ++ [232, i.val]) params i ((generatedRoot.w_6_0 ctx), (((generatedRoot.w_70_0 ctx) i), ((generatedRoot.w_229_0 ctx), ((generatedRoot.w_229_2 ctx), ((generatedRoot.w_0_0 ctx), ((generatedRoot.w_229_1 ctx), ((generatedRoot.w_229_3 ctx), ((generatedRoot.w_1_0 ctx), (((generatedRoot.w_230_0 ctx) i), (((generatedRoot.w_231_0 ctx) i), ((generatedRoot.w_2_0 ctx), ()))))))))))) (((generatedRoot.w_232_0 ctx) i), (((generatedRoot.w_232_1 ctx) i), (((generatedRoot.w_232_2 ctx) i), (((generatedRoot.w_232_3 ctx) i), (((generatedRoot.w_232_4 ctx) i), (((generatedRoot.w_232_5 ctx) i), (((generatedRoot.w_232_6 ctx) i), (((generatedRoot.w_232_7 ctx) i), ())))))))))

theorem generatedRoot.facts_6  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_6 backend hashModel tape path params (generatedRoot.w_0_0 ctx) (generatedRoot.w_1_0 ctx) (generatedRoot.w_6_0 ctx) (generatedRoot.w_70_0 ctx) (generatedRoot.w_71_0 ctx) (generatedRoot.w_72_0 ctx) (generatedRoot.w_73_0 ctx) (generatedRoot.w_74_0 ctx) (generatedRoot.w_75_0 ctx) (generatedRoot.w_76_0 ctx) (generatedRoot.w_77_0 ctx) (generatedRoot.w_202_0 ctx) (generatedRoot.w_203_0 ctx) (generatedRoot.w_206_0 ctx) (generatedRoot.w_207_0 ctx) (generatedRoot.w_210_0 ctx) (generatedRoot.w_211_0 ctx) (generatedRoot.w_214_0 ctx) (generatedRoot.w_215_0 ctx) (generatedRoot.w_218_0 ctx) (generatedRoot.w_219_0 ctx) (generatedRoot.w_222_0 ctx) (generatedRoot.w_223_0 ctx) (generatedRoot.w_226_0 ctx) (generatedRoot.w_227_0 ctx) (generatedRoot.w_230_0 ctx) (generatedRoot.w_231_0 ctx) (generatedRoot.w_2_0 ctx) (generatedRoot.w_200_0 ctx) (generatedRoot.w_200_1 ctx) (generatedRoot.w_200_2 ctx) (generatedRoot.w_200_3 ctx) (generatedRoot.w_201_0 ctx) (generatedRoot.w_201_1 ctx) (generatedRoot.w_201_2 ctx) (generatedRoot.w_201_3 ctx) (generatedRoot.w_204_0 ctx) (generatedRoot.w_204_1 ctx) (generatedRoot.w_204_2 ctx) (generatedRoot.w_204_3 ctx) (generatedRoot.w_204_4 ctx) (generatedRoot.w_204_5 ctx) (generatedRoot.w_204_6 ctx) (generatedRoot.w_204_7 ctx) (generatedRoot.w_205_0 ctx) (generatedRoot.w_205_1 ctx) (generatedRoot.w_205_2 ctx) (generatedRoot.w_205_3 ctx) (generatedRoot.w_208_0 ctx) (generatedRoot.w_208_1 ctx) (generatedRoot.w_208_2 ctx) (generatedRoot.w_208_3 ctx) (generatedRoot.w_208_4 ctx) (generatedRoot.w_208_5 ctx) (generatedRoot.w_208_6 ctx) (generatedRoot.w_208_7 ctx) (generatedRoot.w_209_0 ctx) (generatedRoot.w_209_1 ctx) (generatedRoot.w_209_2 ctx) (generatedRoot.w_209_3 ctx) (generatedRoot.w_212_0 ctx) (generatedRoot.w_212_1 ctx) (generatedRoot.w_212_2 ctx) (generatedRoot.w_212_3 ctx) (generatedRoot.w_212_4 ctx) (generatedRoot.w_212_5 ctx) (generatedRoot.w_212_6 ctx) (generatedRoot.w_212_7 ctx) (generatedRoot.w_213_0 ctx) (generatedRoot.w_213_1 ctx) (generatedRoot.w_213_2 ctx) (generatedRoot.w_213_3 ctx) (generatedRoot.w_216_0 ctx) (generatedRoot.w_216_1 ctx) (generatedRoot.w_216_2 ctx) (generatedRoot.w_216_3 ctx) (generatedRoot.w_216_4 ctx) (generatedRoot.w_216_5 ctx) (generatedRoot.w_216_6 ctx) (generatedRoot.w_216_7 ctx) (generatedRoot.w_217_0 ctx) (generatedRoot.w_217_1 ctx) (generatedRoot.w_217_2 ctx) (generatedRoot.w_217_3 ctx) (generatedRoot.w_220_0 ctx) (generatedRoot.w_220_1 ctx) (generatedRoot.w_220_2 ctx) (generatedRoot.w_220_3 ctx) (generatedRoot.w_220_4 ctx) (generatedRoot.w_220_5 ctx) (generatedRoot.w_220_6 ctx) (generatedRoot.w_220_7 ctx) (generatedRoot.w_221_0 ctx) (generatedRoot.w_221_1 ctx) (generatedRoot.w_221_2 ctx) (generatedRoot.w_221_3 ctx) (generatedRoot.w_224_0 ctx) (generatedRoot.w_224_1 ctx) (generatedRoot.w_224_2 ctx) (generatedRoot.w_224_3 ctx) (generatedRoot.w_224_4 ctx) (generatedRoot.w_224_5 ctx) (generatedRoot.w_224_6 ctx) (generatedRoot.w_224_7 ctx) (generatedRoot.w_225_0 ctx) (generatedRoot.w_225_1 ctx) (generatedRoot.w_225_2 ctx) (generatedRoot.w_225_3 ctx) (generatedRoot.w_228_0 ctx) (generatedRoot.w_228_1 ctx) (generatedRoot.w_228_2 ctx) (generatedRoot.w_228_3 ctx) (generatedRoot.w_228_4 ctx) (generatedRoot.w_228_5 ctx) (generatedRoot.w_228_6 ctx) (generatedRoot.w_228_7 ctx) (generatedRoot.w_229_0 ctx) (generatedRoot.w_229_1 ctx) (generatedRoot.w_229_2 ctx) (generatedRoot.w_229_3 ctx) (generatedRoot.w_232_0 ctx) (generatedRoot.w_232_1 ctx) (generatedRoot.w_232_2 ctx) (generatedRoot.w_232_3 ctx) (generatedRoot.w_232_4 ctx) (generatedRoot.w_232_5 ctx) (generatedRoot.w_232_6 ctx) (generatedRoot.w_232_7 ctx)) :
    generatedRoot.Facts_6 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_7  (ctx : generatedRootCtx) : Prop where
  t224 : (20054016) = 20054016
  h233_3 : (∀ i : Fin 20054016, parallel_generatedRoot_233 backend hashModel tape (path ++ [233, i.val]) params i ((generatedRoot.w_232_0 ctx), ((generatedRoot.w_232_1 ctx), ((generatedRoot.w_232_2 ctx), ((generatedRoot.w_232_3 ctx), ())))) (((generatedRoot.w_233_0 ctx) i), (((generatedRoot.w_233_1 ctx) i), (((generatedRoot.w_233_2 ctx) i), (((generatedRoot.w_233_3 ctx) i), ())))))
  t226 : (2607022080) = 2607022080
  h236_7 : (∀ i : Fin 2607022080, parallel_generatedRoot_236 backend hashModel tape (path ++ [236, i.val]) params i ((generatedRoot.w_6_0 ctx), (((generatedRoot.w_69_0 ctx) i), ((generatedRoot.w_233_0 ctx), ((generatedRoot.w_233_2 ctx), ((generatedRoot.w_0_0 ctx), ((generatedRoot.w_233_1 ctx), ((generatedRoot.w_233_3 ctx), ((generatedRoot.w_1_0 ctx), (((generatedRoot.w_234_0 ctx) i), (((generatedRoot.w_235_0 ctx) i), ((generatedRoot.w_2_0 ctx), ()))))))))))) (((generatedRoot.w_236_0 ctx) i), (((generatedRoot.w_236_1 ctx) i), (((generatedRoot.w_236_2 ctx) i), (((generatedRoot.w_236_3 ctx) i), (((generatedRoot.w_236_4 ctx) i), (((generatedRoot.w_236_5 ctx) i), (((generatedRoot.w_236_6 ctx) i), (((generatedRoot.w_236_7 ctx) i), ())))))))))
  t228 : (20054016) = 20054016
  h237_3 : (∀ i : Fin 20054016, parallel_generatedRoot_237 backend hashModel tape (path ++ [237, i.val]) params i ((generatedRoot.w_236_0 ctx), ((generatedRoot.w_236_1 ctx), ((generatedRoot.w_236_2 ctx), ((generatedRoot.w_236_3 ctx), ())))) (((generatedRoot.w_237_0 ctx) i), (((generatedRoot.w_237_1 ctx) i), (((generatedRoot.w_237_2 ctx) i), (((generatedRoot.w_237_3 ctx) i), ())))))
  t230 : (2607022080) = 2607022080
  h240_7 : (∀ i : Fin 2607022080, parallel_generatedRoot_240 backend hashModel tape (path ++ [240, i.val]) params i ((generatedRoot.w_6_0 ctx), (((generatedRoot.w_68_0 ctx) i), ((generatedRoot.w_237_0 ctx), ((generatedRoot.w_237_2 ctx), ((generatedRoot.w_0_0 ctx), ((generatedRoot.w_237_1 ctx), ((generatedRoot.w_237_3 ctx), ((generatedRoot.w_1_0 ctx), (((generatedRoot.w_238_0 ctx) i), (((generatedRoot.w_239_0 ctx) i), ((generatedRoot.w_2_0 ctx), ()))))))))))) (((generatedRoot.w_240_0 ctx) i), (((generatedRoot.w_240_1 ctx) i), (((generatedRoot.w_240_2 ctx) i), (((generatedRoot.w_240_3 ctx) i), (((generatedRoot.w_240_4 ctx) i), (((generatedRoot.w_240_5 ctx) i), (((generatedRoot.w_240_6 ctx) i), (((generatedRoot.w_240_7 ctx) i), ())))))))))
  t232 : (20054016) = 20054016
  h241_3 : (∀ i : Fin 20054016, parallel_generatedRoot_241 backend hashModel tape (path ++ [241, i.val]) params i ((generatedRoot.w_240_0 ctx), ((generatedRoot.w_240_1 ctx), ((generatedRoot.w_240_2 ctx), ((generatedRoot.w_240_3 ctx), ())))) (((generatedRoot.w_241_0 ctx) i), (((generatedRoot.w_241_1 ctx) i), (((generatedRoot.w_241_2 ctx) i), (((generatedRoot.w_241_3 ctx) i), ())))))
  t234 : (20054016) = 20054016
  h242_3 : (∀ i : Fin 20054016, parallel_generatedRoot_242 backend hashModel tape (path ++ [242, i.val]) params i ((generatedRoot.w_6_0 ctx), (((generatedRoot.w_241_0 ctx) i), (((generatedRoot.w_241_2 ctx) i), ((generatedRoot.w_4_0 ctx), (((generatedRoot.w_241_1 ctx) i), (((generatedRoot.w_241_3 ctx) i), ((generatedRoot.w_4_2 ctx), ((generatedRoot.w_4_1 ctx), ((generatedRoot.w_4_3 ctx), ()))))))))) (((generatedRoot.w_242_0 ctx) i), (((generatedRoot.w_242_1 ctx) i), (((generatedRoot.w_242_2 ctx) i), (((generatedRoot.w_242_3 ctx) i), ())))))
  t236 : (20054016) = 20054016
  h243_3 : (∀ i : Fin 20054016, parallel_generatedRoot_243 backend hashModel tape (path ++ [243, i.val]) params i ((generatedRoot.w_242_0 ctx), ((generatedRoot.w_242_1 ctx), ((generatedRoot.w_242_2 ctx), ((generatedRoot.w_242_3 ctx), ())))) (((generatedRoot.w_243_0 ctx) i), (((generatedRoot.w_243_1 ctx) i), (((generatedRoot.w_243_2 ctx) i), (((generatedRoot.w_243_3 ctx) i), ())))))
  h244_0 : MxxRuntime.familyGetStatic (generatedRoot.w_243_0 ctx) (0) (generatedRoot.w_244_0 ctx)
  h245_0 : MxxRuntime.familyGetStatic (generatedRoot.w_243_0 ctx) (1) (generatedRoot.w_245_0 ctx)
  h247_0 : MxxRuntime.familyGetStatic (generatedRoot.w_243_1 ctx) (0) (generatedRoot.w_247_0 ctx)
  h248_0 : MxxRuntime.familyGetStatic (generatedRoot.w_243_1 ctx) (1) (generatedRoot.w_248_0 ctx)
  h250_0 : MxxRuntime.familyGetStatic (generatedRoot.w_243_2 ctx) (0) (generatedRoot.w_250_0 ctx)
  h251_0 : MxxRuntime.familyGetStatic (generatedRoot.w_243_2 ctx) (1) (generatedRoot.w_251_0 ctx)
  h253_0 : MxxRuntime.familyGetStatic (generatedRoot.w_243_3 ctx) (0) (generatedRoot.w_253_0 ctx)
  h254_0 : MxxRuntime.familyGetStatic (generatedRoot.w_243_3 ctx) (1) (generatedRoot.w_254_0 ctx)
  t246 : 0 ≤ (65534)
  h256_3 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_generatedRoot_256 backend hashModel tape (path ++ [256, i]) params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, ((generatedRoot.w_243_0 ctx), ((generatedRoot.w_243_1 ctx), ((generatedRoot.w_243_2 ctx), ((generatedRoot.w_243_3 ctx), ())))))))) next) (Int.toNat (65534)) ((generatedRoot.w_246_0 ctx), ((generatedRoot.w_249_0 ctx), ((generatedRoot.w_252_0 ctx), ((generatedRoot.w_255_0 ctx), ())))) ((generatedRoot.w_256_0 ctx), ((generatedRoot.w_256_1 ctx), ((generatedRoot.w_256_2 ctx), ((generatedRoot.w_256_3 ctx), ()))))
  h257_0 : MxxRuntime.tensorRuns (generatedRoot.w_67_0 ctx) (generatedRoot.w_256_0 ctx) (generatedRoot.w_257_0 ctx)
  t249 : 0 ≤ (0)
  t250 : (0) < 306
  h259_0 : MxxRuntime.tensorRuns (generatedRoot.w_258_0 ctx) (generatedRoot.w_256_1 ctx) (generatedRoot.w_259_0 ctx)
  t252 : 0 ≤ (0)
  t253 : (0) < 306
  h261_0 : MxxRuntime.tensorRuns (generatedRoot.w_260_0 ctx) (generatedRoot.w_256_2 ctx) (generatedRoot.w_261_0 ctx)
  t255 : 0 ≤ (0)

theorem generatedRoot.facts_7  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_7 backend hashModel tape path params (generatedRoot.w_0_0 ctx) (generatedRoot.w_1_0 ctx) (generatedRoot.w_6_0 ctx) (generatedRoot.w_68_0 ctx) (generatedRoot.w_69_0 ctx) (generatedRoot.w_234_0 ctx) (generatedRoot.w_235_0 ctx) (generatedRoot.w_238_0 ctx) (generatedRoot.w_239_0 ctx) (generatedRoot.w_2_0 ctx) (generatedRoot.w_4_0 ctx) (generatedRoot.w_4_1 ctx) (generatedRoot.w_4_2 ctx) (generatedRoot.w_4_3 ctx) (generatedRoot.w_67_0 ctx) (generatedRoot.w_232_0 ctx) (generatedRoot.w_232_1 ctx) (generatedRoot.w_232_2 ctx) (generatedRoot.w_232_3 ctx) (generatedRoot.w_233_0 ctx) (generatedRoot.w_233_1 ctx) (generatedRoot.w_233_2 ctx) (generatedRoot.w_233_3 ctx) (generatedRoot.w_236_0 ctx) (generatedRoot.w_236_1 ctx) (generatedRoot.w_236_2 ctx) (generatedRoot.w_236_3 ctx) (generatedRoot.w_236_4 ctx) (generatedRoot.w_236_5 ctx) (generatedRoot.w_236_6 ctx) (generatedRoot.w_236_7 ctx) (generatedRoot.w_237_0 ctx) (generatedRoot.w_237_1 ctx) (generatedRoot.w_237_2 ctx) (generatedRoot.w_237_3 ctx) (generatedRoot.w_240_0 ctx) (generatedRoot.w_240_1 ctx) (generatedRoot.w_240_2 ctx) (generatedRoot.w_240_3 ctx) (generatedRoot.w_240_4 ctx) (generatedRoot.w_240_5 ctx) (generatedRoot.w_240_6 ctx) (generatedRoot.w_240_7 ctx) (generatedRoot.w_241_0 ctx) (generatedRoot.w_241_1 ctx) (generatedRoot.w_241_2 ctx) (generatedRoot.w_241_3 ctx) (generatedRoot.w_242_0 ctx) (generatedRoot.w_242_1 ctx) (generatedRoot.w_242_2 ctx) (generatedRoot.w_242_3 ctx) (generatedRoot.w_243_0 ctx) (generatedRoot.w_243_1 ctx) (generatedRoot.w_243_2 ctx) (generatedRoot.w_243_3 ctx) (generatedRoot.w_244_0 ctx) (generatedRoot.w_245_0 ctx) (generatedRoot.w_246_0 ctx) (generatedRoot.w_247_0 ctx) (generatedRoot.w_248_0 ctx) (generatedRoot.w_249_0 ctx) (generatedRoot.w_250_0 ctx) (generatedRoot.w_251_0 ctx) (generatedRoot.w_252_0 ctx) (generatedRoot.w_253_0 ctx) (generatedRoot.w_254_0 ctx) (generatedRoot.w_255_0 ctx) (generatedRoot.w_256_0 ctx) (generatedRoot.w_256_1 ctx) (generatedRoot.w_256_2 ctx) (generatedRoot.w_256_3 ctx) (generatedRoot.w_257_0 ctx) (generatedRoot.w_258_0 ctx) (generatedRoot.w_259_0 ctx) (generatedRoot.w_260_0 ctx) (generatedRoot.w_261_0 ctx)) :
    generatedRoot.Facts_7 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_8  (ctx : generatedRootCtx) (outputs : (generatedRoot.OutputType)) : Prop where
  t256 : (0) < 306
  h263_0 : MxxRuntime.tensorRuns (generatedRoot.w_262_0 ctx) (generatedRoot.w_256_3 ctx) (generatedRoot.w_263_0 ctx)
  t258 : 0 ≤ (305)
  h264_3 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × Unit) => sequential_generatedRoot_264 backend hashModel tape (path ++ [264, i]) params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, ((generatedRoot.w_243_0 ctx), ((generatedRoot.w_243_1 ctx), ((generatedRoot.w_243_2 ctx), ((generatedRoot.w_243_3 ctx), ())))))))) next) (Int.toNat (305)) ((generatedRoot.w_257_0 ctx), ((generatedRoot.w_259_0 ctx), ((generatedRoot.w_261_0 ctx), ((generatedRoot.w_263_0 ctx), ())))) ((generatedRoot.w_264_0 ctx), ((generatedRoot.w_264_1 ctx), ((generatedRoot.w_264_2 ctx), ((generatedRoot.w_264_3 ctx), ()))))
  hout : outputs = generatedRoot.outs ctx

theorem generatedRoot.facts_8  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_8 backend hashModel tape path params (generatedRoot.w_3_0 ctx) (generatedRoot.w_3_1 ctx) (generatedRoot.w_3_2 ctx) (generatedRoot.w_3_3 ctx) (generatedRoot.w_4_0 ctx) (generatedRoot.w_4_1 ctx) (generatedRoot.w_4_2 ctx) (generatedRoot.w_4_3 ctx) (generatedRoot.w_33_2 ctx) (generatedRoot.w_33_3 ctx) (generatedRoot.w_66_0 ctx) (generatedRoot.w_66_1 ctx) (generatedRoot.w_66_2 ctx) (generatedRoot.w_66_3 ctx) (generatedRoot.w_108_6 ctx) (generatedRoot.w_108_7 ctx) (generatedRoot.w_112_6 ctx) (generatedRoot.w_112_7 ctx) (generatedRoot.w_116_6 ctx) (generatedRoot.w_116_7 ctx) (generatedRoot.w_120_6 ctx) (generatedRoot.w_120_7 ctx) (generatedRoot.w_124_6 ctx) (generatedRoot.w_124_7 ctx) (generatedRoot.w_128_6 ctx) (generatedRoot.w_128_7 ctx) (generatedRoot.w_132_6 ctx) (generatedRoot.w_132_7 ctx) (generatedRoot.w_136_6 ctx) (generatedRoot.w_136_7 ctx) (generatedRoot.w_140_6 ctx) (generatedRoot.w_140_7 ctx) (generatedRoot.w_144_6 ctx) (generatedRoot.w_144_7 ctx) (generatedRoot.w_148_6 ctx) (generatedRoot.w_148_7 ctx) (generatedRoot.w_152_6 ctx) (generatedRoot.w_152_7 ctx) (generatedRoot.w_156_6 ctx) (generatedRoot.w_156_7 ctx) (generatedRoot.w_160_6 ctx) (generatedRoot.w_160_7 ctx) (generatedRoot.w_164_6 ctx) (generatedRoot.w_164_7 ctx) (generatedRoot.w_168_6 ctx) (generatedRoot.w_168_7 ctx) (generatedRoot.w_172_6 ctx) (generatedRoot.w_172_7 ctx) (generatedRoot.w_176_6 ctx) (generatedRoot.w_176_7 ctx) (generatedRoot.w_180_6 ctx) (generatedRoot.w_180_7 ctx) (generatedRoot.w_184_6 ctx) (generatedRoot.w_184_7 ctx) (generatedRoot.w_188_6 ctx) (generatedRoot.w_188_7 ctx) (generatedRoot.w_192_6 ctx) (generatedRoot.w_192_7 ctx) (generatedRoot.w_196_6 ctx) (generatedRoot.w_196_7 ctx) (generatedRoot.w_200_6 ctx) (generatedRoot.w_200_7 ctx) (generatedRoot.w_204_6 ctx) (generatedRoot.w_204_7 ctx) (generatedRoot.w_208_6 ctx) (generatedRoot.w_208_7 ctx) (generatedRoot.w_212_6 ctx) (generatedRoot.w_212_7 ctx) (generatedRoot.w_216_6 ctx) (generatedRoot.w_216_7 ctx) (generatedRoot.w_220_6 ctx) (generatedRoot.w_220_7 ctx) (generatedRoot.w_224_6 ctx) (generatedRoot.w_224_7 ctx) (generatedRoot.w_228_6 ctx) (generatedRoot.w_228_7 ctx) (generatedRoot.w_232_6 ctx) (generatedRoot.w_232_7 ctx) (generatedRoot.w_236_6 ctx) (generatedRoot.w_236_7 ctx) (generatedRoot.w_240_6 ctx) (generatedRoot.w_240_7 ctx) (generatedRoot.w_243_0 ctx) (generatedRoot.w_243_1 ctx) (generatedRoot.w_243_2 ctx) (generatedRoot.w_243_3 ctx) (generatedRoot.w_256_3 ctx) (generatedRoot.w_257_0 ctx) (generatedRoot.w_259_0 ctx) (generatedRoot.w_261_0 ctx) (generatedRoot.w_262_0 ctx) (generatedRoot.w_263_0 ctx) (generatedRoot.w_264_0 ctx) (generatedRoot.w_264_1 ctx) (generatedRoot.w_264_2 ctx) (generatedRoot.w_264_3 ctx) outputs) :
    generatedRoot.Facts_8 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2⟩

structure generatedRoot.Facts  (ctx : generatedRootCtx) (outputs : (generatedRoot.OutputType)) : Prop where
  c0 : generatedRoot.Facts_0 ctx
  c1 : generatedRoot.Facts_1 ctx
  c2 : generatedRoot.Facts_2 ctx
  c3 : generatedRoot.Facts_3 ctx
  c4 : generatedRoot.Facts_4 ctx
  c5 : generatedRoot.Facts_5 ctx
  c6 : generatedRoot.Facts_6 ctx
  c7 : generatedRoot.Facts_7 ctx
  c8 : generatedRoot.Facts_8 ctx outputs

theorem generatedRoot.facts  (ctx : generatedRootCtx) {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.body backend hashModel tape path params ctx.inputs outputs ctx.witness) :
    generatedRoot.Facts ctx outputs :=
  ⟨(generatedRoot.facts_0 (ctx := ctx) (outputs := outputs) h.1.1.1), (generatedRoot.facts_1 (ctx := ctx) (outputs := outputs) h.1.1.2), (generatedRoot.facts_2 (ctx := ctx) (outputs := outputs) h.1.2.1), (generatedRoot.facts_3 (ctx := ctx) (outputs := outputs) h.1.2.2), (generatedRoot.facts_4 (ctx := ctx) (outputs := outputs) h.2.1.1), (generatedRoot.facts_5 (ctx := ctx) (outputs := outputs) h.2.1.2), (generatedRoot.facts_6 (ctx := ctx) (outputs := outputs) h.2.2.1), (generatedRoot.facts_7 (ctx := ctx) (outputs := outputs) h.2.2.2.1), (generatedRoot.facts_8 (ctx := ctx) (outputs := outputs) h.2.2.2.2)⟩

theorem generatedRoot.h3_3_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 4) :
    ∃ w : parallel_generatedRoot_3.Witness, parallel_generatedRoot_3.Facts (⟨((generatedRoot.w_0_0 ctx), ((generatedRoot.w_1_0 ctx), ((generatedRoot.w_2_0 ctx), ()))), w⟩ : parallel_generatedRoot_3.Ctx backend hashModel tape (path ++ [3, i.val]) params i) (((generatedRoot.w_3_0 ctx) i), (((generatedRoot.w_3_1 ctx) i), (((generatedRoot.w_3_2 ctx) i), (((generatedRoot.w_3_3 ctx) i), ())))) :=
  let ⟨w, h⟩ := F.c0.h3_3 i
  ⟨w, parallel_generatedRoot_3.facts _ h⟩

theorem generatedRoot.h4_3_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 128) :
    ∃ w : parallel_generatedRoot_4.Witness, parallel_generatedRoot_4.Facts (⟨((generatedRoot.w_0_0 ctx), ((generatedRoot.w_1_0 ctx), ((generatedRoot.w_2_0 ctx), ()))), w⟩ : parallel_generatedRoot_4.Ctx backend hashModel tape (path ++ [4, i.val]) params i) (((generatedRoot.w_4_0 ctx) i), (((generatedRoot.w_4_1 ctx) i), (((generatedRoot.w_4_2 ctx) i), (((generatedRoot.w_4_3 ctx) i), ())))) :=
  let ⟨w, h⟩ := F.c0.h4_3 i
  ⟨w, parallel_generatedRoot_4.facts _ h⟩

theorem generatedRoot.h109_3_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 20054016) :
    ∃ w : parallel_generatedRoot_109.Witness, parallel_generatedRoot_109.Facts (⟨((generatedRoot.w_108_0 ctx), ((generatedRoot.w_108_1 ctx), ((generatedRoot.w_108_2 ctx), ((generatedRoot.w_108_3 ctx), ())))), w⟩ : parallel_generatedRoot_109.Ctx backend hashModel tape (path ++ [109, i.val]) params i) (((generatedRoot.w_109_0 ctx) i), (((generatedRoot.w_109_1 ctx) i), (((generatedRoot.w_109_2 ctx) i), (((generatedRoot.w_109_3 ctx) i), ())))) :=
  let ⟨w, h⟩ := F.c3.h109_3 i
  ⟨w, parallel_generatedRoot_109.facts _ h⟩

theorem generatedRoot.h113_3_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 20054016) :
    ∃ w : parallel_generatedRoot_113.Witness, parallel_generatedRoot_113.Facts (⟨((generatedRoot.w_112_0 ctx), ((generatedRoot.w_112_1 ctx), ((generatedRoot.w_112_2 ctx), ((generatedRoot.w_112_3 ctx), ())))), w⟩ : parallel_generatedRoot_113.Ctx backend hashModel tape (path ++ [113, i.val]) params i) (((generatedRoot.w_113_0 ctx) i), (((generatedRoot.w_113_1 ctx) i), (((generatedRoot.w_113_2 ctx) i), (((generatedRoot.w_113_3 ctx) i), ())))) :=
  let ⟨w, h⟩ := F.c3.h113_3 i
  ⟨w, parallel_generatedRoot_113.facts _ h⟩

theorem generatedRoot.h117_3_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 20054016) :
    ∃ w : parallel_generatedRoot_117.Witness, parallel_generatedRoot_117.Facts (⟨((generatedRoot.w_116_0 ctx), ((generatedRoot.w_116_1 ctx), ((generatedRoot.w_116_2 ctx), ((generatedRoot.w_116_3 ctx), ())))), w⟩ : parallel_generatedRoot_117.Ctx backend hashModel tape (path ++ [117, i.val]) params i) (((generatedRoot.w_117_0 ctx) i), (((generatedRoot.w_117_1 ctx) i), (((generatedRoot.w_117_2 ctx) i), (((generatedRoot.w_117_3 ctx) i), ())))) :=
  let ⟨w, h⟩ := F.c3.h117_3 i
  ⟨w, parallel_generatedRoot_117.facts _ h⟩

theorem generatedRoot.h121_3_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 20054016) :
    ∃ w : parallel_generatedRoot_121.Witness, parallel_generatedRoot_121.Facts (⟨((generatedRoot.w_120_0 ctx), ((generatedRoot.w_120_1 ctx), ((generatedRoot.w_120_2 ctx), ((generatedRoot.w_120_3 ctx), ())))), w⟩ : parallel_generatedRoot_121.Ctx backend hashModel tape (path ++ [121, i.val]) params i) (((generatedRoot.w_121_0 ctx) i), (((generatedRoot.w_121_1 ctx) i), (((generatedRoot.w_121_2 ctx) i), (((generatedRoot.w_121_3 ctx) i), ())))) :=
  let ⟨w, h⟩ := F.c3.h121_3 i
  ⟨w, parallel_generatedRoot_121.facts _ h⟩

theorem generatedRoot.h125_3_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 20054016) :
    ∃ w : parallel_generatedRoot_125.Witness, parallel_generatedRoot_125.Facts (⟨((generatedRoot.w_124_0 ctx), ((generatedRoot.w_124_1 ctx), ((generatedRoot.w_124_2 ctx), ((generatedRoot.w_124_3 ctx), ())))), w⟩ : parallel_generatedRoot_125.Ctx backend hashModel tape (path ++ [125, i.val]) params i) (((generatedRoot.w_125_0 ctx) i), (((generatedRoot.w_125_1 ctx) i), (((generatedRoot.w_125_2 ctx) i), (((generatedRoot.w_125_3 ctx) i), ())))) :=
  let ⟨w, h⟩ := F.c3.h125_3 i
  ⟨w, parallel_generatedRoot_125.facts _ h⟩

theorem generatedRoot.h129_3_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 20054016) :
    ∃ w : parallel_generatedRoot_129.Witness, parallel_generatedRoot_129.Facts (⟨((generatedRoot.w_128_0 ctx), ((generatedRoot.w_128_1 ctx), ((generatedRoot.w_128_2 ctx), ((generatedRoot.w_128_3 ctx), ())))), w⟩ : parallel_generatedRoot_129.Ctx backend hashModel tape (path ++ [129, i.val]) params i) (((generatedRoot.w_129_0 ctx) i), (((generatedRoot.w_129_1 ctx) i), (((generatedRoot.w_129_2 ctx) i), (((generatedRoot.w_129_3 ctx) i), ())))) :=
  let ⟨w, h⟩ := F.c3.h129_3 i
  ⟨w, parallel_generatedRoot_129.facts _ h⟩

theorem generatedRoot.h133_3_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 20054016) :
    ∃ w : parallel_generatedRoot_133.Witness, parallel_generatedRoot_133.Facts (⟨((generatedRoot.w_132_0 ctx), ((generatedRoot.w_132_1 ctx), ((generatedRoot.w_132_2 ctx), ((generatedRoot.w_132_3 ctx), ())))), w⟩ : parallel_generatedRoot_133.Ctx backend hashModel tape (path ++ [133, i.val]) params i) (((generatedRoot.w_133_0 ctx) i), (((generatedRoot.w_133_1 ctx) i), (((generatedRoot.w_133_2 ctx) i), (((generatedRoot.w_133_3 ctx) i), ())))) :=
  let ⟨w, h⟩ := F.c3.h133_3 i
  ⟨w, parallel_generatedRoot_133.facts _ h⟩

theorem generatedRoot.h137_3_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 20054016) :
    ∃ w : parallel_generatedRoot_137.Witness, parallel_generatedRoot_137.Facts (⟨((generatedRoot.w_136_0 ctx), ((generatedRoot.w_136_1 ctx), ((generatedRoot.w_136_2 ctx), ((generatedRoot.w_136_3 ctx), ())))), w⟩ : parallel_generatedRoot_137.Ctx backend hashModel tape (path ++ [137, i.val]) params i) (((generatedRoot.w_137_0 ctx) i), (((generatedRoot.w_137_1 ctx) i), (((generatedRoot.w_137_2 ctx) i), (((generatedRoot.w_137_3 ctx) i), ())))) :=
  let ⟨w, h⟩ := F.c4.h137_3 i
  ⟨w, parallel_generatedRoot_137.facts _ h⟩

theorem generatedRoot.h141_3_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 20054016) :
    ∃ w : parallel_generatedRoot_141.Witness, parallel_generatedRoot_141.Facts (⟨((generatedRoot.w_140_0 ctx), ((generatedRoot.w_140_1 ctx), ((generatedRoot.w_140_2 ctx), ((generatedRoot.w_140_3 ctx), ())))), w⟩ : parallel_generatedRoot_141.Ctx backend hashModel tape (path ++ [141, i.val]) params i) (((generatedRoot.w_141_0 ctx) i), (((generatedRoot.w_141_1 ctx) i), (((generatedRoot.w_141_2 ctx) i), (((generatedRoot.w_141_3 ctx) i), ())))) :=
  let ⟨w, h⟩ := F.c4.h141_3 i
  ⟨w, parallel_generatedRoot_141.facts _ h⟩

theorem generatedRoot.h145_3_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 20054016) :
    ∃ w : parallel_generatedRoot_145.Witness, parallel_generatedRoot_145.Facts (⟨((generatedRoot.w_144_0 ctx), ((generatedRoot.w_144_1 ctx), ((generatedRoot.w_144_2 ctx), ((generatedRoot.w_144_3 ctx), ())))), w⟩ : parallel_generatedRoot_145.Ctx backend hashModel tape (path ++ [145, i.val]) params i) (((generatedRoot.w_145_0 ctx) i), (((generatedRoot.w_145_1 ctx) i), (((generatedRoot.w_145_2 ctx) i), (((generatedRoot.w_145_3 ctx) i), ())))) :=
  let ⟨w, h⟩ := F.c4.h145_3 i
  ⟨w, parallel_generatedRoot_145.facts _ h⟩

theorem generatedRoot.h149_3_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 20054016) :
    ∃ w : parallel_generatedRoot_149.Witness, parallel_generatedRoot_149.Facts (⟨((generatedRoot.w_148_0 ctx), ((generatedRoot.w_148_1 ctx), ((generatedRoot.w_148_2 ctx), ((generatedRoot.w_148_3 ctx), ())))), w⟩ : parallel_generatedRoot_149.Ctx backend hashModel tape (path ++ [149, i.val]) params i) (((generatedRoot.w_149_0 ctx) i), (((generatedRoot.w_149_1 ctx) i), (((generatedRoot.w_149_2 ctx) i), (((generatedRoot.w_149_3 ctx) i), ())))) :=
  let ⟨w, h⟩ := F.c4.h149_3 i
  ⟨w, parallel_generatedRoot_149.facts _ h⟩

theorem generatedRoot.h153_3_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 20054016) :
    ∃ w : parallel_generatedRoot_153.Witness, parallel_generatedRoot_153.Facts (⟨((generatedRoot.w_152_0 ctx), ((generatedRoot.w_152_1 ctx), ((generatedRoot.w_152_2 ctx), ((generatedRoot.w_152_3 ctx), ())))), w⟩ : parallel_generatedRoot_153.Ctx backend hashModel tape (path ++ [153, i.val]) params i) (((generatedRoot.w_153_0 ctx) i), (((generatedRoot.w_153_1 ctx) i), (((generatedRoot.w_153_2 ctx) i), (((generatedRoot.w_153_3 ctx) i), ())))) :=
  let ⟨w, h⟩ := F.c4.h153_3 i
  ⟨w, parallel_generatedRoot_153.facts _ h⟩

theorem generatedRoot.h157_3_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 20054016) :
    ∃ w : parallel_generatedRoot_157.Witness, parallel_generatedRoot_157.Facts (⟨((generatedRoot.w_156_0 ctx), ((generatedRoot.w_156_1 ctx), ((generatedRoot.w_156_2 ctx), ((generatedRoot.w_156_3 ctx), ())))), w⟩ : parallel_generatedRoot_157.Ctx backend hashModel tape (path ++ [157, i.val]) params i) (((generatedRoot.w_157_0 ctx) i), (((generatedRoot.w_157_1 ctx) i), (((generatedRoot.w_157_2 ctx) i), (((generatedRoot.w_157_3 ctx) i), ())))) :=
  let ⟨w, h⟩ := F.c4.h157_3 i
  ⟨w, parallel_generatedRoot_157.facts _ h⟩

theorem generatedRoot.h161_3_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 20054016) :
    ∃ w : parallel_generatedRoot_161.Witness, parallel_generatedRoot_161.Facts (⟨((generatedRoot.w_160_0 ctx), ((generatedRoot.w_160_1 ctx), ((generatedRoot.w_160_2 ctx), ((generatedRoot.w_160_3 ctx), ())))), w⟩ : parallel_generatedRoot_161.Ctx backend hashModel tape (path ++ [161, i.val]) params i) (((generatedRoot.w_161_0 ctx) i), (((generatedRoot.w_161_1 ctx) i), (((generatedRoot.w_161_2 ctx) i), (((generatedRoot.w_161_3 ctx) i), ())))) :=
  let ⟨w, h⟩ := F.c4.h161_3 i
  ⟨w, parallel_generatedRoot_161.facts _ h⟩

theorem generatedRoot.h165_3_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 20054016) :
    ∃ w : parallel_generatedRoot_165.Witness, parallel_generatedRoot_165.Facts (⟨((generatedRoot.w_164_0 ctx), ((generatedRoot.w_164_1 ctx), ((generatedRoot.w_164_2 ctx), ((generatedRoot.w_164_3 ctx), ())))), w⟩ : parallel_generatedRoot_165.Ctx backend hashModel tape (path ++ [165, i.val]) params i) (((generatedRoot.w_165_0 ctx) i), (((generatedRoot.w_165_1 ctx) i), (((generatedRoot.w_165_2 ctx) i), (((generatedRoot.w_165_3 ctx) i), ())))) :=
  let ⟨w, h⟩ := F.c4.h165_3 i
  ⟨w, parallel_generatedRoot_165.facts _ h⟩

theorem generatedRoot.h169_3_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 20054016) :
    ∃ w : parallel_generatedRoot_169.Witness, parallel_generatedRoot_169.Facts (⟨((generatedRoot.w_168_0 ctx), ((generatedRoot.w_168_1 ctx), ((generatedRoot.w_168_2 ctx), ((generatedRoot.w_168_3 ctx), ())))), w⟩ : parallel_generatedRoot_169.Ctx backend hashModel tape (path ++ [169, i.val]) params i) (((generatedRoot.w_169_0 ctx) i), (((generatedRoot.w_169_1 ctx) i), (((generatedRoot.w_169_2 ctx) i), (((generatedRoot.w_169_3 ctx) i), ())))) :=
  let ⟨w, h⟩ := F.c5.h169_3 i
  ⟨w, parallel_generatedRoot_169.facts _ h⟩

theorem generatedRoot.h173_3_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 20054016) :
    ∃ w : parallel_generatedRoot_173.Witness, parallel_generatedRoot_173.Facts (⟨((generatedRoot.w_172_0 ctx), ((generatedRoot.w_172_1 ctx), ((generatedRoot.w_172_2 ctx), ((generatedRoot.w_172_3 ctx), ())))), w⟩ : parallel_generatedRoot_173.Ctx backend hashModel tape (path ++ [173, i.val]) params i) (((generatedRoot.w_173_0 ctx) i), (((generatedRoot.w_173_1 ctx) i), (((generatedRoot.w_173_2 ctx) i), (((generatedRoot.w_173_3 ctx) i), ())))) :=
  let ⟨w, h⟩ := F.c5.h173_3 i
  ⟨w, parallel_generatedRoot_173.facts _ h⟩

theorem generatedRoot.h177_3_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 20054016) :
    ∃ w : parallel_generatedRoot_177.Witness, parallel_generatedRoot_177.Facts (⟨((generatedRoot.w_176_0 ctx), ((generatedRoot.w_176_1 ctx), ((generatedRoot.w_176_2 ctx), ((generatedRoot.w_176_3 ctx), ())))), w⟩ : parallel_generatedRoot_177.Ctx backend hashModel tape (path ++ [177, i.val]) params i) (((generatedRoot.w_177_0 ctx) i), (((generatedRoot.w_177_1 ctx) i), (((generatedRoot.w_177_2 ctx) i), (((generatedRoot.w_177_3 ctx) i), ())))) :=
  let ⟨w, h⟩ := F.c5.h177_3 i
  ⟨w, parallel_generatedRoot_177.facts _ h⟩

theorem generatedRoot.h181_3_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 20054016) :
    ∃ w : parallel_generatedRoot_181.Witness, parallel_generatedRoot_181.Facts (⟨((generatedRoot.w_180_0 ctx), ((generatedRoot.w_180_1 ctx), ((generatedRoot.w_180_2 ctx), ((generatedRoot.w_180_3 ctx), ())))), w⟩ : parallel_generatedRoot_181.Ctx backend hashModel tape (path ++ [181, i.val]) params i) (((generatedRoot.w_181_0 ctx) i), (((generatedRoot.w_181_1 ctx) i), (((generatedRoot.w_181_2 ctx) i), (((generatedRoot.w_181_3 ctx) i), ())))) :=
  let ⟨w, h⟩ := F.c5.h181_3 i
  ⟨w, parallel_generatedRoot_181.facts _ h⟩

theorem generatedRoot.h185_3_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 20054016) :
    ∃ w : parallel_generatedRoot_185.Witness, parallel_generatedRoot_185.Facts (⟨((generatedRoot.w_184_0 ctx), ((generatedRoot.w_184_1 ctx), ((generatedRoot.w_184_2 ctx), ((generatedRoot.w_184_3 ctx), ())))), w⟩ : parallel_generatedRoot_185.Ctx backend hashModel tape (path ++ [185, i.val]) params i) (((generatedRoot.w_185_0 ctx) i), (((generatedRoot.w_185_1 ctx) i), (((generatedRoot.w_185_2 ctx) i), (((generatedRoot.w_185_3 ctx) i), ())))) :=
  let ⟨w, h⟩ := F.c5.h185_3 i
  ⟨w, parallel_generatedRoot_185.facts _ h⟩

theorem generatedRoot.h189_3_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 20054016) :
    ∃ w : parallel_generatedRoot_189.Witness, parallel_generatedRoot_189.Facts (⟨((generatedRoot.w_188_0 ctx), ((generatedRoot.w_188_1 ctx), ((generatedRoot.w_188_2 ctx), ((generatedRoot.w_188_3 ctx), ())))), w⟩ : parallel_generatedRoot_189.Ctx backend hashModel tape (path ++ [189, i.val]) params i) (((generatedRoot.w_189_0 ctx) i), (((generatedRoot.w_189_1 ctx) i), (((generatedRoot.w_189_2 ctx) i), (((generatedRoot.w_189_3 ctx) i), ())))) :=
  let ⟨w, h⟩ := F.c5.h189_3 i
  ⟨w, parallel_generatedRoot_189.facts _ h⟩

theorem generatedRoot.h193_3_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 20054016) :
    ∃ w : parallel_generatedRoot_193.Witness, parallel_generatedRoot_193.Facts (⟨((generatedRoot.w_192_0 ctx), ((generatedRoot.w_192_1 ctx), ((generatedRoot.w_192_2 ctx), ((generatedRoot.w_192_3 ctx), ())))), w⟩ : parallel_generatedRoot_193.Ctx backend hashModel tape (path ++ [193, i.val]) params i) (((generatedRoot.w_193_0 ctx) i), (((generatedRoot.w_193_1 ctx) i), (((generatedRoot.w_193_2 ctx) i), (((generatedRoot.w_193_3 ctx) i), ())))) :=
  let ⟨w, h⟩ := F.c5.h193_3 i
  ⟨w, parallel_generatedRoot_193.facts _ h⟩

theorem generatedRoot.h197_3_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 20054016) :
    ∃ w : parallel_generatedRoot_197.Witness, parallel_generatedRoot_197.Facts (⟨((generatedRoot.w_196_0 ctx), ((generatedRoot.w_196_1 ctx), ((generatedRoot.w_196_2 ctx), ((generatedRoot.w_196_3 ctx), ())))), w⟩ : parallel_generatedRoot_197.Ctx backend hashModel tape (path ++ [197, i.val]) params i) (((generatedRoot.w_197_0 ctx) i), (((generatedRoot.w_197_1 ctx) i), (((generatedRoot.w_197_2 ctx) i), (((generatedRoot.w_197_3 ctx) i), ())))) :=
  let ⟨w, h⟩ := F.c5.h197_3 i
  ⟨w, parallel_generatedRoot_197.facts _ h⟩

theorem generatedRoot.h201_3_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 20054016) :
    ∃ w : parallel_generatedRoot_201.Witness, parallel_generatedRoot_201.Facts (⟨((generatedRoot.w_200_0 ctx), ((generatedRoot.w_200_1 ctx), ((generatedRoot.w_200_2 ctx), ((generatedRoot.w_200_3 ctx), ())))), w⟩ : parallel_generatedRoot_201.Ctx backend hashModel tape (path ++ [201, i.val]) params i) (((generatedRoot.w_201_0 ctx) i), (((generatedRoot.w_201_1 ctx) i), (((generatedRoot.w_201_2 ctx) i), (((generatedRoot.w_201_3 ctx) i), ())))) :=
  let ⟨w, h⟩ := F.c6.h201_3 i
  ⟨w, parallel_generatedRoot_201.facts _ h⟩

theorem generatedRoot.h205_3_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 20054016) :
    ∃ w : parallel_generatedRoot_205.Witness, parallel_generatedRoot_205.Facts (⟨((generatedRoot.w_204_0 ctx), ((generatedRoot.w_204_1 ctx), ((generatedRoot.w_204_2 ctx), ((generatedRoot.w_204_3 ctx), ())))), w⟩ : parallel_generatedRoot_205.Ctx backend hashModel tape (path ++ [205, i.val]) params i) (((generatedRoot.w_205_0 ctx) i), (((generatedRoot.w_205_1 ctx) i), (((generatedRoot.w_205_2 ctx) i), (((generatedRoot.w_205_3 ctx) i), ())))) :=
  let ⟨w, h⟩ := F.c6.h205_3 i
  ⟨w, parallel_generatedRoot_205.facts _ h⟩

theorem generatedRoot.h209_3_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 20054016) :
    ∃ w : parallel_generatedRoot_209.Witness, parallel_generatedRoot_209.Facts (⟨((generatedRoot.w_208_0 ctx), ((generatedRoot.w_208_1 ctx), ((generatedRoot.w_208_2 ctx), ((generatedRoot.w_208_3 ctx), ())))), w⟩ : parallel_generatedRoot_209.Ctx backend hashModel tape (path ++ [209, i.val]) params i) (((generatedRoot.w_209_0 ctx) i), (((generatedRoot.w_209_1 ctx) i), (((generatedRoot.w_209_2 ctx) i), (((generatedRoot.w_209_3 ctx) i), ())))) :=
  let ⟨w, h⟩ := F.c6.h209_3 i
  ⟨w, parallel_generatedRoot_209.facts _ h⟩

theorem generatedRoot.h213_3_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 20054016) :
    ∃ w : parallel_generatedRoot_213.Witness, parallel_generatedRoot_213.Facts (⟨((generatedRoot.w_212_0 ctx), ((generatedRoot.w_212_1 ctx), ((generatedRoot.w_212_2 ctx), ((generatedRoot.w_212_3 ctx), ())))), w⟩ : parallel_generatedRoot_213.Ctx backend hashModel tape (path ++ [213, i.val]) params i) (((generatedRoot.w_213_0 ctx) i), (((generatedRoot.w_213_1 ctx) i), (((generatedRoot.w_213_2 ctx) i), (((generatedRoot.w_213_3 ctx) i), ())))) :=
  let ⟨w, h⟩ := F.c6.h213_3 i
  ⟨w, parallel_generatedRoot_213.facts _ h⟩

theorem generatedRoot.h217_3_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 20054016) :
    ∃ w : parallel_generatedRoot_217.Witness, parallel_generatedRoot_217.Facts (⟨((generatedRoot.w_216_0 ctx), ((generatedRoot.w_216_1 ctx), ((generatedRoot.w_216_2 ctx), ((generatedRoot.w_216_3 ctx), ())))), w⟩ : parallel_generatedRoot_217.Ctx backend hashModel tape (path ++ [217, i.val]) params i) (((generatedRoot.w_217_0 ctx) i), (((generatedRoot.w_217_1 ctx) i), (((generatedRoot.w_217_2 ctx) i), (((generatedRoot.w_217_3 ctx) i), ())))) :=
  let ⟨w, h⟩ := F.c6.h217_3 i
  ⟨w, parallel_generatedRoot_217.facts _ h⟩

theorem generatedRoot.h221_3_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 20054016) :
    ∃ w : parallel_generatedRoot_221.Witness, parallel_generatedRoot_221.Facts (⟨((generatedRoot.w_220_0 ctx), ((generatedRoot.w_220_1 ctx), ((generatedRoot.w_220_2 ctx), ((generatedRoot.w_220_3 ctx), ())))), w⟩ : parallel_generatedRoot_221.Ctx backend hashModel tape (path ++ [221, i.val]) params i) (((generatedRoot.w_221_0 ctx) i), (((generatedRoot.w_221_1 ctx) i), (((generatedRoot.w_221_2 ctx) i), (((generatedRoot.w_221_3 ctx) i), ())))) :=
  let ⟨w, h⟩ := F.c6.h221_3 i
  ⟨w, parallel_generatedRoot_221.facts _ h⟩

theorem generatedRoot.h225_3_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 20054016) :
    ∃ w : parallel_generatedRoot_225.Witness, parallel_generatedRoot_225.Facts (⟨((generatedRoot.w_224_0 ctx), ((generatedRoot.w_224_1 ctx), ((generatedRoot.w_224_2 ctx), ((generatedRoot.w_224_3 ctx), ())))), w⟩ : parallel_generatedRoot_225.Ctx backend hashModel tape (path ++ [225, i.val]) params i) (((generatedRoot.w_225_0 ctx) i), (((generatedRoot.w_225_1 ctx) i), (((generatedRoot.w_225_2 ctx) i), (((generatedRoot.w_225_3 ctx) i), ())))) :=
  let ⟨w, h⟩ := F.c6.h225_3 i
  ⟨w, parallel_generatedRoot_225.facts _ h⟩

theorem generatedRoot.h229_3_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 20054016) :
    ∃ w : parallel_generatedRoot_229.Witness, parallel_generatedRoot_229.Facts (⟨((generatedRoot.w_228_0 ctx), ((generatedRoot.w_228_1 ctx), ((generatedRoot.w_228_2 ctx), ((generatedRoot.w_228_3 ctx), ())))), w⟩ : parallel_generatedRoot_229.Ctx backend hashModel tape (path ++ [229, i.val]) params i) (((generatedRoot.w_229_0 ctx) i), (((generatedRoot.w_229_1 ctx) i), (((generatedRoot.w_229_2 ctx) i), (((generatedRoot.w_229_3 ctx) i), ())))) :=
  let ⟨w, h⟩ := F.c6.h229_3 i
  ⟨w, parallel_generatedRoot_229.facts _ h⟩

theorem generatedRoot.h233_3_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 20054016) :
    ∃ w : parallel_generatedRoot_233.Witness, parallel_generatedRoot_233.Facts (⟨((generatedRoot.w_232_0 ctx), ((generatedRoot.w_232_1 ctx), ((generatedRoot.w_232_2 ctx), ((generatedRoot.w_232_3 ctx), ())))), w⟩ : parallel_generatedRoot_233.Ctx backend hashModel tape (path ++ [233, i.val]) params i) (((generatedRoot.w_233_0 ctx) i), (((generatedRoot.w_233_1 ctx) i), (((generatedRoot.w_233_2 ctx) i), (((generatedRoot.w_233_3 ctx) i), ())))) :=
  let ⟨w, h⟩ := F.c7.h233_3 i
  ⟨w, parallel_generatedRoot_233.facts _ h⟩

theorem generatedRoot.h237_3_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 20054016) :
    ∃ w : parallel_generatedRoot_237.Witness, parallel_generatedRoot_237.Facts (⟨((generatedRoot.w_236_0 ctx), ((generatedRoot.w_236_1 ctx), ((generatedRoot.w_236_2 ctx), ((generatedRoot.w_236_3 ctx), ())))), w⟩ : parallel_generatedRoot_237.Ctx backend hashModel tape (path ++ [237, i.val]) params i) (((generatedRoot.w_237_0 ctx) i), (((generatedRoot.w_237_1 ctx) i), (((generatedRoot.w_237_2 ctx) i), (((generatedRoot.w_237_3 ctx) i), ())))) :=
  let ⟨w, h⟩ := F.c7.h237_3 i
  ⟨w, parallel_generatedRoot_237.facts _ h⟩

theorem generatedRoot.h241_3_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 20054016) :
    ∃ w : parallel_generatedRoot_241.Witness, parallel_generatedRoot_241.Facts (⟨((generatedRoot.w_240_0 ctx), ((generatedRoot.w_240_1 ctx), ((generatedRoot.w_240_2 ctx), ((generatedRoot.w_240_3 ctx), ())))), w⟩ : parallel_generatedRoot_241.Ctx backend hashModel tape (path ++ [241, i.val]) params i) (((generatedRoot.w_241_0 ctx) i), (((generatedRoot.w_241_1 ctx) i), (((generatedRoot.w_241_2 ctx) i), (((generatedRoot.w_241_3 ctx) i), ())))) :=
  let ⟨w, h⟩ := F.c7.h241_3 i
  ⟨w, parallel_generatedRoot_241.facts _ h⟩

theorem generatedRoot.h242_3_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 20054016) :
    ∃ w : parallel_generatedRoot_242.Witness, parallel_generatedRoot_242.Facts (⟨((generatedRoot.w_6_0 ctx), (((generatedRoot.w_241_0 ctx) i), (((generatedRoot.w_241_2 ctx) i), ((generatedRoot.w_4_0 ctx), (((generatedRoot.w_241_1 ctx) i), (((generatedRoot.w_241_3 ctx) i), ((generatedRoot.w_4_2 ctx), ((generatedRoot.w_4_1 ctx), ((generatedRoot.w_4_3 ctx), ()))))))))), w⟩ : parallel_generatedRoot_242.Ctx backend hashModel tape (path ++ [242, i.val]) params i) (((generatedRoot.w_242_0 ctx) i), (((generatedRoot.w_242_1 ctx) i), (((generatedRoot.w_242_2 ctx) i), (((generatedRoot.w_242_3 ctx) i), ())))) :=
  let ⟨w, h⟩ := F.c7.h242_3 i
  ⟨w, parallel_generatedRoot_242.facts _ h⟩

end

end Stage_preprocess_1_1

end
