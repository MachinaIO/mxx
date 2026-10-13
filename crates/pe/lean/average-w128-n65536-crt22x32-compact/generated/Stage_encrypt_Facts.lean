import Stage_encrypt

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000
set_option linter.unusedVariables false

noncomputable section

namespace Stage_encrypt

local notation "mxxNat0" => 78079523927896127712009307999974412043135143141543941570119110821624764686494582648421232630165133436275519032254680972082541026053657846674805178959586229420520814546267827301225593318349477307602951695899164673
local notation "mxxNat1" => 5278087253403224991333645099610003021785206966119564254171942298247993190952980821552692773158302798199939315804484105262715072792529872674621368544490941390506983498620144494214222200022752952511718466872729239793477806438096725442738627984346925320228738380148171996792888816066148390509478988805065937247347127262069050805334354391385613933775075315634469785233518480876460832181155068060520232362595000272223451214118913
local notation "mxxMat0" => Mxx.Primitives.ExactMatrix mxxNat0 65536 1 1
local notation "mxxMat1" => Mxx.Primitives.ExactMatrix mxxNat0 65536 1 44
local notation "mxxMat2" => Mxx.Primitives.ExactMatrix mxxNat1 65536 1 1

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) where
  inputs : mxxMat0 × ByteArray × Int × Int × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × Unit
  witness : generatedRoot.Witness

def generatedRoot.w_0_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.inputs.1

def generatedRoot.w_1_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : ByteArray :=
  ctx.inputs.2.1

def generatedRoot.w_17_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Int :=
  ctx.inputs.2.2.1

def generatedRoot.w_19_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Int :=
  ctx.inputs.2.2.2.1

def generatedRoot.w_38_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 4 → mxxMat1 :=
  ctx.inputs.2.2.2.2.1

def generatedRoot.w_39_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 4 → mxxMat1 :=
  ctx.inputs.2.2.2.2.2.1

def generatedRoot.w_2_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  (1 : mxxMat0)

def generatedRoot.w_3_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat1 :=
  ctx.witness.w_3_0

def generatedRoot.w_4_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 37 → mxxMat1 :=
  ctx.witness.w_4_0

def generatedRoot.w_4_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 37 → mxxMat1 :=
  ctx.witness.w_4_1

def generatedRoot.w_5_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 22 → mxxMat1 :=
  ctx.witness.w_5_0

def generatedRoot.w_5_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 22 → mxxMat1 :=
  ctx.witness.w_5_1

def generatedRoot.w_5_2 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 22 → mxxMat1 :=
  ctx.witness.w_5_2

def generatedRoot.w_5_3 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 22 → mxxMat1 :=
  ctx.witness.w_5_3

def generatedRoot.w_6_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 85184 → Int :=
  ctx.witness.w_6_0

def generatedRoot.w_7_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 85184 → mxxMat2 :=
  ctx.witness.w_7_0

def generatedRoot.w_7_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 85184 → mxxMat2 :=
  ctx.witness.w_7_1

def generatedRoot.w_8_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat1 :=
  ctx.witness.w_8_0

def generatedRoot.w_9_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_0_0 ctx) (generatedRoot.w_8_0 ctx)

def generatedRoot.w_10_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat1 :=
  ctx.witness.w_10_0

def generatedRoot.w_11_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_2_0 ctx) (generatedRoot.w_10_0 ctx)

def generatedRoot.w_12_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat1 :=
  MxxRuntime.matrixAdd (generatedRoot.w_9_0 ctx) (generatedRoot.w_11_0 ctx)

def generatedRoot.w_13_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat1 :=
  ctx.witness.w_13_0

def generatedRoot.w_14_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat1 :=
  ctx.witness.w_14_0

def generatedRoot.w_15_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_0_0 ctx) (generatedRoot.w_14_0 ctx)

def generatedRoot.w_16_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat0)

def generatedRoot.w_18_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.liftInteger (generatedRoot.w_17_0 ctx)

def generatedRoot.w_20_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.multiplyMonomial (generatedRoot.w_18_0 ctx) (generatedRoot.w_19_0 ctx)

def generatedRoot.w_21_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_16_0 ctx) (generatedRoot.w_20_0 ctx)

def generatedRoot.w_22_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat1 :=
  MxxRuntime.matrixMulScalarRight (generatedRoot.w_15_0 ctx) (generatedRoot.w_21_0 ctx)

def generatedRoot.w_23_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat1 :=
  MxxRuntime.matrixSub (generatedRoot.w_12_0 ctx) (generatedRoot.w_22_0 ctx)

def generatedRoot.w_24_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat1 :=
  ctx.witness.w_24_0

def generatedRoot.w_25_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat1 :=
  MxxRuntime.matrixAdd (generatedRoot.w_23_0 ctx) (generatedRoot.w_24_0 ctx)

def generatedRoot.w_26_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat1 :=
  ctx.witness.w_26_0

def generatedRoot.w_27_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_0_0 ctx) (generatedRoot.w_26_0 ctx)

def generatedRoot.w_28_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat1 :=
  ctx.witness.w_28_0

def generatedRoot.w_29_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_2_0 ctx) (generatedRoot.w_28_0 ctx)

def generatedRoot.w_30_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat1 :=
  MxxRuntime.matrixAdd (generatedRoot.w_27_0 ctx) (generatedRoot.w_29_0 ctx)

def generatedRoot.w_31_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_2_0 ctx) (generatedRoot.w_14_0 ctx)

def generatedRoot.w_32_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat1 :=
  MxxRuntime.matrixMulScalarRight (generatedRoot.w_31_0 ctx) (generatedRoot.w_21_0 ctx)

def generatedRoot.w_33_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat1 :=
  MxxRuntime.matrixSub (generatedRoot.w_30_0 ctx) (generatedRoot.w_32_0 ctx)

def generatedRoot.w_34_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat1 :=
  ctx.witness.w_34_0

def generatedRoot.w_35_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat1 :=
  MxxRuntime.matrixAdd (generatedRoot.w_33_0 ctx) (generatedRoot.w_34_0 ctx)

def generatedRoot.w_36_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 4 → mxxMat1 :=
  ctx.witness.w_36_0

def generatedRoot.w_36_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 4 → mxxMat1 :=
  ctx.witness.w_36_1

def generatedRoot.w_37_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 15488 → Int :=
  ctx.witness.w_37_0

def generatedRoot.w_40_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 15488 → mxxMat2 :=
  ctx.witness.w_40_0

def generatedRoot.w_40_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 15488 → mxxMat2 :=
  ctx.witness.w_40_1

def generatedRoot.outs {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : (Fin 85184 → mxxMat2) × (Fin 85184 → mxxMat2) × (Fin 37 → mxxMat1) × (Fin 37 → mxxMat1) × mxxMat1 × mxxMat1 × (Fin 15488 → mxxMat2) × (Fin 15488 → mxxMat2) × Unit :=
  ((generatedRoot.w_7_0 ctx), ((generatedRoot.w_7_1 ctx), ((generatedRoot.w_4_0 ctx), ((generatedRoot.w_4_1 ctx), ((generatedRoot.w_25_0 ctx), ((generatedRoot.w_35_0 ctx), ((generatedRoot.w_40_0 ctx), ((generatedRoot.w_40_1 ctx), ()))))))))

structure generatedRoot.Facts_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) (outputs : (Fin 85184 → mxxMat2) × (Fin 85184 → mxxMat2) × (Fin 37 → mxxMat1) × (Fin 37 → mxxMat1) × mxxMat1 × mxxMat1 × (Fin 15488 → mxxMat2) × (Fin 15488 → mxxMat2) × Unit) : Prop where
  h3_0 : MxxRuntime.gadgetMatrixRuns backend 65536 44 (generatedRoot.w_3_0 ctx)
  t1 : (37) = 37
  h4_1 : (∀ i : Fin 37, parallel_generatedRoot_4 backend hashModel tape (path ++ [4, i.val]) params i ((generatedRoot.w_0_0 ctx), ((generatedRoot.w_1_0 ctx), ((generatedRoot.w_2_0 ctx), ((generatedRoot.w_3_0 ctx), ())))) (((generatedRoot.w_4_0 ctx) i), (((generatedRoot.w_4_1 ctx) i), ())))
  t3 : (22) = 22
  h5_3 : (∀ i : Fin 22, parallel_generatedRoot_5 backend hashModel tape (path ++ [5, i.val]) params i ((generatedRoot.w_1_0 ctx), ((generatedRoot.w_4_0 ctx), ((generatedRoot.w_4_1 ctx), ()))) (((generatedRoot.w_5_0 ctx) i), (((generatedRoot.w_5_1 ctx) i), (((generatedRoot.w_5_2 ctx) i), (((generatedRoot.w_5_3 ctx) i), ())))))
  t5 : (85184) = 85184
  h6_0 : (∀ i : Fin 85184, parallel_generatedRoot_6 backend hashModel tape (path ++ [6, i.val]) params i () ((generatedRoot.w_6_0 ctx) i))
  t7 : (85184) = 85184
  h7_1 : (∀ i : Fin 85184, parallel_generatedRoot_7 backend hashModel tape (path ++ [7, i.val]) params i ((generatedRoot.w_0_0 ctx), ((generatedRoot.w_1_0 ctx), ((generatedRoot.w_2_0 ctx), ((generatedRoot.w_5_0 ctx), (((generatedRoot.w_6_0 ctx) i), ((generatedRoot.w_5_1 ctx), ((generatedRoot.w_5_2 ctx), ((generatedRoot.w_5_3 ctx), ())))))))) (((generatedRoot.w_7_0 ctx) i), (((generatedRoot.w_7_1 ctx) i), ())))
  h8_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [105, 110, 112, 117, 116, 45, 107, 101, 121], .u64Le (0), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((generatedRoot.w_1_0 ctx)) (generatedRoot.w_8_0 ctx)
  h10_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [105, 110, 112, 117, 116, 45, 107, 101, 121], .u64Le (0), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) ((generatedRoot.w_1_0 ctx)) (generatedRoot.w_10_0 ctx)
  h13_0 : MxxRuntime.gadgetMatrixRuns backend 65536 44 (generatedRoot.w_13_0 ctx)
  h14_0 : MxxRuntime.centeredRebaseRuns (generatedRoot.w_13_0 ctx) (generatedRoot.w_14_0 ctx)
  h24_0 : MxxRuntime.gaussianSampleAt tape (path ++ [24]) ((319/100)) (31) (generatedRoot.w_24_0 ctx)
  h26_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [105, 110, 112, 117, 116, 45, 107, 101, 121], .u64Le (0), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((generatedRoot.w_1_0 ctx)) (generatedRoot.w_26_0 ctx)
  h28_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [105, 110, 112, 117, 116, 45, 107, 101, 121], .u64Le (0), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) ((generatedRoot.w_1_0 ctx)) (generatedRoot.w_28_0 ctx)
  h34_0 : MxxRuntime.gaussianSampleAt tape (path ++ [34]) ((319/100)) (31) (generatedRoot.w_34_0 ctx)
  t17 : (4) = 4
  h36_1 : (∀ i : Fin 4, parallel_generatedRoot_36 backend hashModel tape (path ++ [36, i.val]) params i (generatedRoot.w_1_0 ctx) (((generatedRoot.w_36_0 ctx) i), (((generatedRoot.w_36_1 ctx) i), ())))
  t19 : (15488) = 15488
  h37_0 : (∀ i : Fin 15488, parallel_generatedRoot_37 backend hashModel tape (path ++ [37, i.val]) params i () ((generatedRoot.w_37_0 ctx) i))
  t21 : (15488) = 15488
  h40_1 : (∀ i : Fin 15488, ∃ w_38_0_gathered : mxxMat1, ∃ w_39_0_gathered : mxxMat1, 0 ≤ ((generatedRoot.w_37_0 ctx) i) ∧ ((generatedRoot.w_37_0 ctx) i) < 4 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_38_0 ctx) ((generatedRoot.w_37_0 ctx) i) w_38_0_gathered ∧ 0 ≤ ((generatedRoot.w_37_0 ctx) i) ∧ ((generatedRoot.w_37_0 ctx) i) < 4 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_39_0 ctx) ((generatedRoot.w_37_0 ctx) i) w_39_0_gathered ∧ parallel_generatedRoot_40 backend hashModel tape (path ++ [40, i.val]) params i ((generatedRoot.w_0_0 ctx), ((generatedRoot.w_1_0 ctx), ((generatedRoot.w_2_0 ctx), ((generatedRoot.w_36_0 ctx), (((generatedRoot.w_37_0 ctx) i), ((generatedRoot.w_36_1 ctx), (w_38_0_gathered, (w_39_0_gathered, ())))))))) (((generatedRoot.w_40_0 ctx) i), (((generatedRoot.w_40_1 ctx) i), ())))
  hout : outputs = generatedRoot.outs ctx

theorem generatedRoot.facts_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {ctx : generatedRoot.Ctx backend hashModel tape path params} {outputs : (Fin 85184 → mxxMat2) × (Fin 85184 → mxxMat2) × (Fin 37 → mxxMat1) × (Fin 37 → mxxMat1) × mxxMat1 × mxxMat1 × (Fin 15488 → mxxMat2) × (Fin 15488 → mxxMat2) × Unit}
    (h : generatedRoot.constraints_0 backend hashModel tape path params (generatedRoot.w_0_0 ctx) (generatedRoot.w_1_0 ctx) (generatedRoot.w_38_0 ctx) (generatedRoot.w_39_0 ctx) (generatedRoot.w_2_0 ctx) (generatedRoot.w_3_0 ctx) (generatedRoot.w_4_0 ctx) (generatedRoot.w_4_1 ctx) (generatedRoot.w_5_0 ctx) (generatedRoot.w_5_1 ctx) (generatedRoot.w_5_2 ctx) (generatedRoot.w_5_3 ctx) (generatedRoot.w_6_0 ctx) (generatedRoot.w_7_0 ctx) (generatedRoot.w_7_1 ctx) (generatedRoot.w_8_0 ctx) (generatedRoot.w_10_0 ctx) (generatedRoot.w_13_0 ctx) (generatedRoot.w_14_0 ctx) (generatedRoot.w_24_0 ctx) (generatedRoot.w_25_0 ctx) (generatedRoot.w_26_0 ctx) (generatedRoot.w_28_0 ctx) (generatedRoot.w_34_0 ctx) (generatedRoot.w_35_0 ctx) (generatedRoot.w_36_0 ctx) (generatedRoot.w_36_1 ctx) (generatedRoot.w_37_0 ctx) (generatedRoot.w_40_0 ctx) (generatedRoot.w_40_1 ctx) outputs) :
    generatedRoot.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) (outputs : (Fin 85184 → mxxMat2) × (Fin 85184 → mxxMat2) × (Fin 37 → mxxMat1) × (Fin 37 → mxxMat1) × mxxMat1 × mxxMat1 × (Fin 15488 → mxxMat2) × (Fin 15488 → mxxMat2) × Unit) : Prop where
  c0 : generatedRoot.Facts_0 ctx outputs

theorem generatedRoot.facts {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) {outputs : (Fin 85184 → mxxMat2) × (Fin 85184 → mxxMat2) × (Fin 37 → mxxMat1) × (Fin 37 → mxxMat1) × mxxMat1 × mxxMat1 × (Fin 15488 → mxxMat2) × (Fin 15488 → mxxMat2) × Unit}
    (h : generatedRoot.body backend hashModel tape path params ctx.inputs outputs ctx.witness) :
    generatedRoot.Facts ctx outputs :=
  have r0 := generatedRoot.facts_0 (ctx := ctx) (outputs := outputs) h
  ⟨r0⟩

end Stage_encrypt

end
