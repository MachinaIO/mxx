import Stage_encrypt

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000
set_option linter.unusedVariables false

-- The wire types repeat the ring modulus `X ^ n + 1`; unfolding it while checking a
-- projection against its field type expands it at the ring dimension.
attribute [local irreducible] Mxx.Primitives.negacyclicModulus

noncomputable section

namespace Stage_encrypt

local notation "mxxMat0" => Mxx.Primitives.ExactMatrix modulus_0 32768 1 1
local notation "mxxMat1" => Mxx.Primitives.ExactMatrix modulus_0 32768 1 30
local notation "mxxMat2" => Mxx.Primitives.ExactMatrix modulus_1 32768 1 1

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) where
  inputs : mxxMat0 × ByteArray × (Fin 45 → mxxMat1) × (Fin 45 → mxxMat1) × Int × Int × (Fin 2 → mxxMat1) × (Fin 2 → mxxMat1) × Unit
  witness : generatedRoot.Witness

def generatedRoot.w_0_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.inputs.1

def generatedRoot.w_1_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : ByteArray :=
  ctx.inputs.2.1

def generatedRoot.w_3_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 45 → mxxMat1 :=
  ctx.inputs.2.2.1

def generatedRoot.w_4_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 45 → mxxMat1 :=
  ctx.inputs.2.2.2.1

def generatedRoot.w_16_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Int :=
  ctx.inputs.2.2.2.2.1

def generatedRoot.w_18_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Int :=
  ctx.inputs.2.2.2.2.2.1

def generatedRoot.w_35_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 2 → mxxMat1 :=
  ctx.inputs.2.2.2.2.2.2.1

def generatedRoot.w_36_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 2 → mxxMat1 :=
  ctx.inputs.2.2.2.2.2.2.2.1

def generatedRoot.w_2_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  (1 : mxxMat0)

def generatedRoot.w_5_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 54000 → mxxMat2 :=
  ctx.witness.w_5_0

def generatedRoot.w_5_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 54000 → mxxMat2 :=
  ctx.witness.w_5_1

def generatedRoot.w_6_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 45 → mxxMat1 :=
  ctx.witness.w_6_0

def generatedRoot.w_6_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 45 → mxxMat1 :=
  ctx.witness.w_6_1

def generatedRoot.w_7_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat1 :=
  ctx.witness.w_7_0

def generatedRoot.w_8_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_0_0 ctx) (generatedRoot.w_7_0 ctx)

def generatedRoot.w_9_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat1 :=
  ctx.witness.w_9_0

def generatedRoot.w_10_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_2_0 ctx) (generatedRoot.w_9_0 ctx)

def generatedRoot.w_11_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat1 :=
  MxxRuntime.matrixAdd (generatedRoot.w_8_0 ctx) (generatedRoot.w_10_0 ctx)

def generatedRoot.w_12_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat1 :=
  ctx.witness.w_12_0

def generatedRoot.w_13_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat1 :=
  ctx.witness.w_13_0

def generatedRoot.w_14_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_0_0 ctx) (generatedRoot.w_13_0 ctx)

def generatedRoot.w_15_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat0)

def generatedRoot.w_17_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.liftInteger (generatedRoot.w_16_0 ctx)

def generatedRoot.w_19_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.multiplyMonomial (generatedRoot.w_17_0 ctx) (generatedRoot.w_18_0 ctx)

def generatedRoot.w_20_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_15_0 ctx) (generatedRoot.w_19_0 ctx)

def generatedRoot.w_21_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat1 :=
  MxxRuntime.matrixMulScalarRight (generatedRoot.w_14_0 ctx) (generatedRoot.w_20_0 ctx)

def generatedRoot.w_22_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat1 :=
  MxxRuntime.matrixSub (generatedRoot.w_11_0 ctx) (generatedRoot.w_21_0 ctx)

def generatedRoot.w_23_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat1 :=
  ctx.witness.w_23_0

def generatedRoot.w_24_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat1 :=
  MxxRuntime.matrixAdd (generatedRoot.w_22_0 ctx) (generatedRoot.w_23_0 ctx)

def generatedRoot.w_25_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat1 :=
  ctx.witness.w_25_0

def generatedRoot.w_26_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_0_0 ctx) (generatedRoot.w_25_0 ctx)

def generatedRoot.w_27_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat1 :=
  ctx.witness.w_27_0

def generatedRoot.w_28_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_2_0 ctx) (generatedRoot.w_27_0 ctx)

def generatedRoot.w_29_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat1 :=
  MxxRuntime.matrixAdd (generatedRoot.w_26_0 ctx) (generatedRoot.w_28_0 ctx)

def generatedRoot.w_30_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_2_0 ctx) (generatedRoot.w_13_0 ctx)

def generatedRoot.w_31_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat1 :=
  MxxRuntime.matrixMulScalarRight (generatedRoot.w_30_0 ctx) (generatedRoot.w_20_0 ctx)

def generatedRoot.w_32_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat1 :=
  MxxRuntime.matrixSub (generatedRoot.w_29_0 ctx) (generatedRoot.w_31_0 ctx)

def generatedRoot.w_33_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat1 :=
  ctx.witness.w_33_0

def generatedRoot.w_34_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat1 :=
  MxxRuntime.matrixAdd (generatedRoot.w_32_0 ctx) (generatedRoot.w_33_0 ctx)

def generatedRoot.w_37_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 3600 → mxxMat2 :=
  ctx.witness.w_37_0

def generatedRoot.w_37_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 3600 → mxxMat2 :=
  ctx.witness.w_37_1

def generatedRoot.outs {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : (Fin 54000 → mxxMat2) × (Fin 54000 → mxxMat2) × (Fin 45 → mxxMat1) × (Fin 45 → mxxMat1) × mxxMat1 × mxxMat1 × (Fin 3600 → mxxMat2) × (Fin 3600 → mxxMat2) × Unit :=
  ((generatedRoot.w_5_0 ctx), ((generatedRoot.w_5_1 ctx), ((generatedRoot.w_6_0 ctx), ((generatedRoot.w_6_1 ctx), ((generatedRoot.w_24_0 ctx), ((generatedRoot.w_34_0 ctx), ((generatedRoot.w_37_0 ctx), ((generatedRoot.w_37_1 ctx), ()))))))))

structure generatedRoot.Facts_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) (outputs : (Fin 54000 → mxxMat2) × (Fin 54000 → mxxMat2) × (Fin 45 → mxxMat1) × (Fin 45 → mxxMat1) × mxxMat1 × mxxMat1 × (Fin 3600 → mxxMat2) × (Fin 3600 → mxxMat2) × Unit) : Prop where
  t0 : (54000) = 54000
  h5_1 : (∀ i : Fin 54000, parallel_generatedRoot_5 backend hashModel tape (path ++ [5, i.val]) params i ((generatedRoot.w_0_0 ctx), ((generatedRoot.w_1_0 ctx), ((generatedRoot.w_2_0 ctx), ((generatedRoot.w_3_0 ctx), ((generatedRoot.w_4_0 ctx), ()))))) (((generatedRoot.w_5_0 ctx) i), (((generatedRoot.w_5_1 ctx) i), ())))
  t2 : (45) = 45
  h6_1 : (∀ i : Fin 45, parallel_generatedRoot_6 backend hashModel tape (path ++ [6, i.val]) params i (((generatedRoot.w_3_0 ctx) i), (((generatedRoot.w_4_0 ctx) i), ())) (((generatedRoot.w_6_0 ctx) i), (((generatedRoot.w_6_1 ctx) i), ())))
  h7_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [55, 51, 50, 48, 54, 98, 48, 49, 52, 57, 53, 52, 99, 53, 53, 100, 101, 48, 99, 53, 52, 55, 54, 57, 100, 102, 98, 56, 50, 52, 51, 53, 98, 52, 101, 54, 100, 98, 100, 55, 49, 50, 53, 102, 54, 50, 53, 53, 50, 52, 50, 56, 53, 52, 49, 102, 102, 51, 50, 57, 48, 49, 51, 52], .bytes [105, 110, 112, 117, 116, 45, 107, 101, 121], .u64Le (0), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((generatedRoot.w_1_0 ctx)) (generatedRoot.w_7_0 ctx)
  h9_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [55, 51, 50, 48, 54, 98, 48, 49, 52, 57, 53, 52, 99, 53, 53, 100, 101, 48, 99, 53, 52, 55, 54, 57, 100, 102, 98, 56, 50, 52, 51, 53, 98, 52, 101, 54, 100, 98, 100, 55, 49, 50, 53, 102, 54, 50, 53, 53, 50, 52, 50, 56, 53, 52, 49, 102, 102, 51, 50, 57, 48, 49, 51, 52], .bytes [105, 110, 112, 117, 116, 45, 107, 101, 121], .u64Le (0), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) ((generatedRoot.w_1_0 ctx)) (generatedRoot.w_9_0 ctx)
  h12_0 : MxxRuntime.gadgetMatrixRuns backend 4096 30 (generatedRoot.w_12_0 ctx)
  h13_0 : MxxRuntime.centeredRebaseRuns (generatedRoot.w_12_0 ctx) (generatedRoot.w_13_0 ctx)
  h23_0 : MxxRuntime.gaussianSampleAt tape (path ++ [23]) ((319/100)) (31) (generatedRoot.w_23_0 ctx)
  h25_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [55, 51, 50, 48, 54, 98, 48, 49, 52, 57, 53, 52, 99, 53, 53, 100, 101, 48, 99, 53, 52, 55, 54, 57, 100, 102, 98, 56, 50, 52, 51, 53, 98, 52, 101, 54, 100, 98, 100, 55, 49, 50, 53, 102, 54, 50, 53, 53, 50, 52, 50, 56, 53, 52, 49, 102, 102, 51, 50, 57, 48, 49, 51, 52], .bytes [105, 110, 112, 117, 116, 45, 107, 101, 121], .u64Le (0), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((generatedRoot.w_1_0 ctx)) (generatedRoot.w_25_0 ctx)
  h27_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [55, 51, 50, 48, 54, 98, 48, 49, 52, 57, 53, 52, 99, 53, 53, 100, 101, 48, 99, 53, 52, 55, 54, 57, 100, 102, 98, 56, 50, 52, 51, 53, 98, 52, 101, 54, 100, 98, 100, 55, 49, 50, 53, 102, 54, 50, 53, 53, 50, 52, 50, 56, 53, 52, 49, 102, 102, 51, 50, 57, 48, 49, 51, 52], .bytes [105, 110, 112, 117, 116, 45, 107, 101, 121], .u64Le (0), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) ((generatedRoot.w_1_0 ctx)) (generatedRoot.w_27_0 ctx)
  h33_0 : MxxRuntime.gaussianSampleAt tape (path ++ [33]) ((319/100)) (31) (generatedRoot.w_33_0 ctx)
  t12 : (3600) = 3600
  h37_1 : (∀ i : Fin 3600, parallel_generatedRoot_37 backend hashModel tape (path ++ [37, i.val]) params i ((generatedRoot.w_0_0 ctx), ((generatedRoot.w_1_0 ctx), ((generatedRoot.w_2_0 ctx), ((generatedRoot.w_35_0 ctx), ((generatedRoot.w_36_0 ctx), ()))))) (((generatedRoot.w_37_0 ctx) i), (((generatedRoot.w_37_1 ctx) i), ())))
  hout : outputs = generatedRoot.outs ctx

theorem generatedRoot.facts_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {ctx : generatedRoot.Ctx backend hashModel tape path params} {outputs : (Fin 54000 → mxxMat2) × (Fin 54000 → mxxMat2) × (Fin 45 → mxxMat1) × (Fin 45 → mxxMat1) × mxxMat1 × mxxMat1 × (Fin 3600 → mxxMat2) × (Fin 3600 → mxxMat2) × Unit}
    (h : generatedRoot.constraints_0 backend hashModel tape path params (generatedRoot.w_0_0 ctx) (generatedRoot.w_1_0 ctx) (generatedRoot.w_3_0 ctx) (generatedRoot.w_4_0 ctx) (generatedRoot.w_35_0 ctx) (generatedRoot.w_36_0 ctx) (generatedRoot.w_2_0 ctx) (generatedRoot.w_5_0 ctx) (generatedRoot.w_5_1 ctx) (generatedRoot.w_6_0 ctx) (generatedRoot.w_6_1 ctx) (generatedRoot.w_7_0 ctx) (generatedRoot.w_9_0 ctx) (generatedRoot.w_12_0 ctx) (generatedRoot.w_13_0 ctx) (generatedRoot.w_23_0 ctx) (generatedRoot.w_24_0 ctx) (generatedRoot.w_25_0 ctx) (generatedRoot.w_27_0 ctx) (generatedRoot.w_33_0 ctx) (generatedRoot.w_34_0 ctx) (generatedRoot.w_37_0 ctx) (generatedRoot.w_37_1 ctx) outputs) :
    generatedRoot.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) (outputs : (Fin 54000 → mxxMat2) × (Fin 54000 → mxxMat2) × (Fin 45 → mxxMat1) × (Fin 45 → mxxMat1) × mxxMat1 × mxxMat1 × (Fin 3600 → mxxMat2) × (Fin 3600 → mxxMat2) × Unit) : Prop where
  c0 : generatedRoot.Facts_0 ctx outputs

theorem generatedRoot.facts {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) {outputs : (Fin 54000 → mxxMat2) × (Fin 54000 → mxxMat2) × (Fin 45 → mxxMat1) × (Fin 45 → mxxMat1) × mxxMat1 × mxxMat1 × (Fin 3600 → mxxMat2) × (Fin 3600 → mxxMat2) × Unit}
    (h : generatedRoot.body backend hashModel tape path params ctx.inputs outputs ctx.witness) :
    generatedRoot.Facts ctx outputs :=
  have t0 := h
  have r0 := generatedRoot.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end Stage_encrypt

end
