import Stage_decode

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000
set_option linter.unusedVariables false

noncomputable section

namespace Stage_decode

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) where
  inputs : Mxx.Primitives.ExactMatrix 1073741441 8 1 6 × Mxx.Primitives.ExactMatrix 1073741329 8 1 6 × Mxx.Primitives.ExactMatrix 1073740609 8 1 6 × Mxx.Primitives.ExactMatrix 1073740529 8 1 6 × (Fin 8 → Int) × ByteArray × Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 × Int × Int × Mxx.Primitives.ExactMatrix 1073741441 8 1 6 × Mxx.Primitives.ExactMatrix 1073741329 8 1 6 × Mxx.Primitives.ExactMatrix 1073740609 8 1 6 × Mxx.Primitives.ExactMatrix 1073740529 8 1 6 × Unit
  witness : generatedRoot.Witness

def generatedRoot.w_1_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Mxx.Primitives.ExactMatrix 1073741441 8 1 6 :=
  ctx.inputs.1

def generatedRoot.w_5_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Mxx.Primitives.ExactMatrix 1073741329 8 1 6 :=
  ctx.inputs.2.1

def generatedRoot.w_10_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Mxx.Primitives.ExactMatrix 1073740609 8 1 6 :=
  ctx.inputs.2.2.1

def generatedRoot.w_15_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Mxx.Primitives.ExactMatrix 1073740529 8 1 6 :=
  ctx.inputs.2.2.2.1

def generatedRoot.w_20_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 8 → Int :=
  ctx.inputs.2.2.2.2.1

def generatedRoot.w_60_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : ByteArray :=
  ctx.inputs.2.2.2.2.2.1

def generatedRoot.w_67_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 :=
  ctx.inputs.2.2.2.2.2.2.1

def generatedRoot.w_69_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ctx.inputs.2.2.2.2.2.2.2.1

def generatedRoot.w_71_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.1

def generatedRoot.w_139_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Mxx.Primitives.ExactMatrix 1073741441 8 1 6 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_143_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Mxx.Primitives.ExactMatrix 1073741329 8 1 6 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_148_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Mxx.Primitives.ExactMatrix 1073740609 8 1 6 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_153_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Mxx.Primitives.ExactMatrix 1073740529 8 1 6 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_0_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  0

def generatedRoot.w_2_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6 :=
  ctx.witness.w_2_0

def generatedRoot.w_3_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 :=
  (MxxRuntime.matrixPolynomial [(890511653275736298117428745327756462)] : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1)

def generatedRoot.w_4_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6 :=
  MxxRuntime.matrixMulScalarRight (generatedRoot.w_2_0 ctx) (generatedRoot.w_3_0 ctx)

def generatedRoot.w_6_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6 :=
  ctx.witness.w_6_0

def generatedRoot.w_7_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 :=
  (MxxRuntime.matrixPolynomial [(673373482644736804462909142109450233)] : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1)

def generatedRoot.w_8_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6 :=
  MxxRuntime.matrixMulScalarRight (generatedRoot.w_6_0 ctx) (generatedRoot.w_7_0 ctx)

def generatedRoot.w_9_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6 :=
  MxxRuntime.matrixAdd (generatedRoot.w_4_0 ctx) (generatedRoot.w_8_0 ctx)

def generatedRoot.w_11_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6 :=
  ctx.witness.w_11_0

def generatedRoot.w_12_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 :=
  (MxxRuntime.matrixPolynomial [(676801805491843546832002660143739331)] : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1)

def generatedRoot.w_13_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6 :=
  MxxRuntime.matrixMulScalarRight (generatedRoot.w_11_0 ctx) (generatedRoot.w_12_0 ctx)

def generatedRoot.w_14_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6 :=
  MxxRuntime.matrixAdd (generatedRoot.w_9_0 ctx) (generatedRoot.w_13_0 ctx)

def generatedRoot.w_16_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6 :=
  ctx.witness.w_16_0

def generatedRoot.w_17_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 :=
  (MxxRuntime.matrixPolynomial [(417760661884955689660474101877988633)] : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1)

def generatedRoot.w_18_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6 :=
  MxxRuntime.matrixMulScalarRight (generatedRoot.w_16_0 ctx) (generatedRoot.w_17_0 ctx)

def generatedRoot.w_19_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6 :=
  MxxRuntime.matrixAdd (generatedRoot.w_14_0 ctx) (generatedRoot.w_18_0 ctx)

def generatedRoot.w_21_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ctx.witness.w_21_0

def generatedRoot.w_22_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 :=
  MxxRuntime.liftInteger (generatedRoot.w_21_0 ctx)

def generatedRoot.w_23_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  0

def generatedRoot.w_24_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 :=
  MxxRuntime.multiplyMonomial (generatedRoot.w_22_0 ctx) (generatedRoot.w_23_0 ctx)

def generatedRoot.w_25_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ctx.witness.w_25_0

def generatedRoot.w_26_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 :=
  MxxRuntime.liftInteger (generatedRoot.w_25_0 ctx)

def generatedRoot.w_27_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  1

def generatedRoot.w_28_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 :=
  MxxRuntime.multiplyMonomial (generatedRoot.w_26_0 ctx) (generatedRoot.w_27_0 ctx)

def generatedRoot.w_29_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 :=
  MxxRuntime.matrixAdd (generatedRoot.w_24_0 ctx) (generatedRoot.w_28_0 ctx)

def generatedRoot.w_30_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ctx.witness.w_30_0

def generatedRoot.w_31_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 :=
  MxxRuntime.liftInteger (generatedRoot.w_30_0 ctx)

def generatedRoot.w_32_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  2

def generatedRoot.w_33_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 :=
  MxxRuntime.multiplyMonomial (generatedRoot.w_31_0 ctx) (generatedRoot.w_32_0 ctx)

def generatedRoot.w_34_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 :=
  MxxRuntime.matrixAdd (generatedRoot.w_29_0 ctx) (generatedRoot.w_33_0 ctx)

def generatedRoot.w_35_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ctx.witness.w_35_0

def generatedRoot.w_36_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 :=
  MxxRuntime.liftInteger (generatedRoot.w_35_0 ctx)

def generatedRoot.w_37_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  3

def generatedRoot.w_38_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 :=
  MxxRuntime.multiplyMonomial (generatedRoot.w_36_0 ctx) (generatedRoot.w_37_0 ctx)

def generatedRoot.w_39_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 :=
  MxxRuntime.matrixAdd (generatedRoot.w_34_0 ctx) (generatedRoot.w_38_0 ctx)

def generatedRoot.w_40_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ctx.witness.w_40_0

def generatedRoot.w_41_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 :=
  MxxRuntime.liftInteger (generatedRoot.w_40_0 ctx)

def generatedRoot.w_42_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  4

def generatedRoot.w_43_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 :=
  MxxRuntime.multiplyMonomial (generatedRoot.w_41_0 ctx) (generatedRoot.w_42_0 ctx)

def generatedRoot.w_44_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 :=
  MxxRuntime.matrixAdd (generatedRoot.w_39_0 ctx) (generatedRoot.w_43_0 ctx)

def generatedRoot.w_45_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ctx.witness.w_45_0

def generatedRoot.w_46_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 :=
  MxxRuntime.liftInteger (generatedRoot.w_45_0 ctx)

def generatedRoot.w_47_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  5

def generatedRoot.w_48_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 :=
  MxxRuntime.multiplyMonomial (generatedRoot.w_46_0 ctx) (generatedRoot.w_47_0 ctx)

def generatedRoot.w_49_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 :=
  MxxRuntime.matrixAdd (generatedRoot.w_44_0 ctx) (generatedRoot.w_48_0 ctx)

def generatedRoot.w_50_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ctx.witness.w_50_0

def generatedRoot.w_51_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 :=
  MxxRuntime.liftInteger (generatedRoot.w_50_0 ctx)

def generatedRoot.w_52_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  6

def generatedRoot.w_53_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 :=
  MxxRuntime.multiplyMonomial (generatedRoot.w_51_0 ctx) (generatedRoot.w_52_0 ctx)

def generatedRoot.w_54_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 :=
  MxxRuntime.matrixAdd (generatedRoot.w_49_0 ctx) (generatedRoot.w_53_0 ctx)

def generatedRoot.w_55_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ctx.witness.w_55_0

def generatedRoot.w_56_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 :=
  MxxRuntime.liftInteger (generatedRoot.w_55_0 ctx)

def generatedRoot.w_57_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  7

def generatedRoot.w_58_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 :=
  MxxRuntime.multiplyMonomial (generatedRoot.w_56_0 ctx) (generatedRoot.w_57_0 ctx)

def generatedRoot.w_59_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 :=
  MxxRuntime.matrixAdd (generatedRoot.w_54_0 ctx) (generatedRoot.w_58_0 ctx)

def generatedRoot.w_61_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 2 6 :=
  ctx.witness.w_61_0

def generatedRoot.w_62_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6 :=
  ctx.witness.w_62_0

def generatedRoot.w_63_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_59_0 ctx) (generatedRoot.w_62_0 ctx)

def generatedRoot.w_64_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6 :=
  ctx.witness.w_64_0

def generatedRoot.w_65_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6 :=
  MxxRuntime.matrixAdd (generatedRoot.w_63_0 ctx) (generatedRoot.w_64_0 ctx)

def generatedRoot.w_66_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 :=
  (MxxRuntime.matrixPolynomial [(1073740529)] : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1)

def generatedRoot.w_68_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_66_0 ctx) (generatedRoot.w_67_0 ctx)

def generatedRoot.w_70_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 :=
  MxxRuntime.liftInteger (generatedRoot.w_69_0 ctx)

def generatedRoot.w_72_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 :=
  MxxRuntime.multiplyMonomial (generatedRoot.w_70_0 ctx) (generatedRoot.w_71_0 ctx)

def generatedRoot.w_73_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_68_0 ctx) (generatedRoot.w_72_0 ctx)

def generatedRoot.w_74_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Mxx.Primitives.ExactMatrix 1237937626222020133447349201 8 1 6 :=
  ctx.witness.w_74_0

def generatedRoot.w_75_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6 :=
  ctx.witness.w_75_0

def generatedRoot.w_76_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_73_0 ctx) (generatedRoot.w_75_0 ctx)

def generatedRoot.w_77_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6 :=
  MxxRuntime.matrixSub (generatedRoot.w_65_0 ctx) (generatedRoot.w_76_0 ctx)

def generatedRoot.w_78_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6 :=
  MxxRuntime.matrixSub (generatedRoot.w_19_0 ctx) (generatedRoot.w_77_0 ctx)

def generatedRoot.w_79_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 :=
  ctx.witness.w_79_0

def generatedRoot.w_80_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ctx.witness.w_80_0

def generatedRoot.w_80_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ctx.witness.w_80_1

def generatedRoot.w_80_2 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ctx.witness.w_80_2

def generatedRoot.w_80_3 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ctx.witness.w_80_3

def generatedRoot.w_80_4 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ctx.witness.w_80_4

def generatedRoot.w_80_5 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ctx.witness.w_80_5

def generatedRoot.w_80_6 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ctx.witness.w_80_6

def generatedRoot.w_80_7 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ctx.witness.w_80_7

def generatedRoot.w_81_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ((generatedRoot.w_0_0 ctx) + (generatedRoot.w_80_0 ctx))

def generatedRoot.w_82_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ((generatedRoot.w_81_0 ctx) + (generatedRoot.w_80_1 ctx))

def generatedRoot.w_83_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ((generatedRoot.w_82_0 ctx) + (generatedRoot.w_80_2 ctx))

def generatedRoot.w_84_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ((generatedRoot.w_83_0 ctx) + (generatedRoot.w_80_3 ctx))

def generatedRoot.w_85_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ((generatedRoot.w_84_0 ctx) + (generatedRoot.w_80_4 ctx))

def generatedRoot.w_86_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ((generatedRoot.w_85_0 ctx) + (generatedRoot.w_80_5 ctx))

def generatedRoot.w_87_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ((generatedRoot.w_86_0 ctx) + (generatedRoot.w_80_6 ctx))

def generatedRoot.w_88_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ((generatedRoot.w_87_0 ctx) + (generatedRoot.w_80_7 ctx))

def generatedRoot.w_89_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 :=
  ctx.witness.w_89_0

def generatedRoot.w_90_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ctx.witness.w_90_0

def generatedRoot.w_90_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ctx.witness.w_90_1

def generatedRoot.w_90_2 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ctx.witness.w_90_2

def generatedRoot.w_90_3 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ctx.witness.w_90_3

def generatedRoot.w_90_4 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ctx.witness.w_90_4

def generatedRoot.w_90_5 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ctx.witness.w_90_5

def generatedRoot.w_90_6 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ctx.witness.w_90_6

def generatedRoot.w_90_7 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ctx.witness.w_90_7

def generatedRoot.w_91_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ((generatedRoot.w_88_0 ctx) + (generatedRoot.w_90_0 ctx))

def generatedRoot.w_92_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ((generatedRoot.w_91_0 ctx) + (generatedRoot.w_90_1 ctx))

def generatedRoot.w_93_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ((generatedRoot.w_92_0 ctx) + (generatedRoot.w_90_2 ctx))

def generatedRoot.w_94_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ((generatedRoot.w_93_0 ctx) + (generatedRoot.w_90_3 ctx))

def generatedRoot.w_95_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ((generatedRoot.w_94_0 ctx) + (generatedRoot.w_90_4 ctx))

def generatedRoot.w_96_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ((generatedRoot.w_95_0 ctx) + (generatedRoot.w_90_5 ctx))

def generatedRoot.w_97_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ((generatedRoot.w_96_0 ctx) + (generatedRoot.w_90_6 ctx))

def generatedRoot.w_98_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ((generatedRoot.w_97_0 ctx) + (generatedRoot.w_90_7 ctx))

def generatedRoot.w_99_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 :=
  ctx.witness.w_99_0

def generatedRoot.w_100_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ctx.witness.w_100_0

def generatedRoot.w_100_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ctx.witness.w_100_1

def generatedRoot.w_100_2 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ctx.witness.w_100_2

def generatedRoot.w_100_3 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ctx.witness.w_100_3

def generatedRoot.w_100_4 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ctx.witness.w_100_4

def generatedRoot.w_100_5 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ctx.witness.w_100_5

def generatedRoot.w_100_6 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ctx.witness.w_100_6

def generatedRoot.w_100_7 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ctx.witness.w_100_7

def generatedRoot.w_101_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ((generatedRoot.w_98_0 ctx) + (generatedRoot.w_100_0 ctx))

def generatedRoot.w_102_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ((generatedRoot.w_101_0 ctx) + (generatedRoot.w_100_1 ctx))

def generatedRoot.w_103_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ((generatedRoot.w_102_0 ctx) + (generatedRoot.w_100_2 ctx))

def generatedRoot.w_104_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ((generatedRoot.w_103_0 ctx) + (generatedRoot.w_100_3 ctx))

def generatedRoot.w_105_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ((generatedRoot.w_104_0 ctx) + (generatedRoot.w_100_4 ctx))

def generatedRoot.w_106_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ((generatedRoot.w_105_0 ctx) + (generatedRoot.w_100_5 ctx))

def generatedRoot.w_107_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ((generatedRoot.w_106_0 ctx) + (generatedRoot.w_100_6 ctx))

def generatedRoot.w_108_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ((generatedRoot.w_107_0 ctx) + (generatedRoot.w_100_7 ctx))

def generatedRoot.w_109_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 :=
  ctx.witness.w_109_0

def generatedRoot.w_110_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ctx.witness.w_110_0

def generatedRoot.w_110_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ctx.witness.w_110_1

def generatedRoot.w_110_2 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ctx.witness.w_110_2

def generatedRoot.w_110_3 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ctx.witness.w_110_3

def generatedRoot.w_110_4 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ctx.witness.w_110_4

def generatedRoot.w_110_5 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ctx.witness.w_110_5

def generatedRoot.w_110_6 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ctx.witness.w_110_6

def generatedRoot.w_110_7 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ctx.witness.w_110_7

def generatedRoot.w_111_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ((generatedRoot.w_108_0 ctx) + (generatedRoot.w_110_0 ctx))

def generatedRoot.w_112_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ((generatedRoot.w_111_0 ctx) + (generatedRoot.w_110_1 ctx))

def generatedRoot.w_113_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ((generatedRoot.w_112_0 ctx) + (generatedRoot.w_110_2 ctx))

def generatedRoot.w_114_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ((generatedRoot.w_113_0 ctx) + (generatedRoot.w_110_3 ctx))

def generatedRoot.w_115_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ((generatedRoot.w_114_0 ctx) + (generatedRoot.w_110_4 ctx))

def generatedRoot.w_116_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ((generatedRoot.w_115_0 ctx) + (generatedRoot.w_110_5 ctx))

def generatedRoot.w_117_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ((generatedRoot.w_116_0 ctx) + (generatedRoot.w_110_6 ctx))

def generatedRoot.w_118_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ((generatedRoot.w_117_0 ctx) + (generatedRoot.w_110_7 ctx))

def generatedRoot.w_119_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 :=
  ctx.witness.w_119_0

def generatedRoot.w_120_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ctx.witness.w_120_0

def generatedRoot.w_120_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ctx.witness.w_120_1

def generatedRoot.w_120_2 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ctx.witness.w_120_2

def generatedRoot.w_120_3 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ctx.witness.w_120_3

def generatedRoot.w_120_4 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ctx.witness.w_120_4

def generatedRoot.w_120_5 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ctx.witness.w_120_5

def generatedRoot.w_120_6 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ctx.witness.w_120_6

def generatedRoot.w_120_7 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ctx.witness.w_120_7

def generatedRoot.w_121_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ((generatedRoot.w_118_0 ctx) + (generatedRoot.w_120_0 ctx))

def generatedRoot.w_122_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ((generatedRoot.w_121_0 ctx) + (generatedRoot.w_120_1 ctx))

def generatedRoot.w_123_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ((generatedRoot.w_122_0 ctx) + (generatedRoot.w_120_2 ctx))

def generatedRoot.w_124_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ((generatedRoot.w_123_0 ctx) + (generatedRoot.w_120_3 ctx))

def generatedRoot.w_125_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ((generatedRoot.w_124_0 ctx) + (generatedRoot.w_120_4 ctx))

def generatedRoot.w_126_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ((generatedRoot.w_125_0 ctx) + (generatedRoot.w_120_5 ctx))

def generatedRoot.w_127_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ((generatedRoot.w_126_0 ctx) + (generatedRoot.w_120_6 ctx))

def generatedRoot.w_128_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ((generatedRoot.w_127_0 ctx) + (generatedRoot.w_120_7 ctx))

def generatedRoot.w_129_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 :=
  ctx.witness.w_129_0

def generatedRoot.w_130_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ctx.witness.w_130_0

def generatedRoot.w_130_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ctx.witness.w_130_1

def generatedRoot.w_130_2 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ctx.witness.w_130_2

def generatedRoot.w_130_3 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ctx.witness.w_130_3

def generatedRoot.w_130_4 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ctx.witness.w_130_4

def generatedRoot.w_130_5 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ctx.witness.w_130_5

def generatedRoot.w_130_6 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ctx.witness.w_130_6

def generatedRoot.w_130_7 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ctx.witness.w_130_7

def generatedRoot.w_131_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ((generatedRoot.w_128_0 ctx) + (generatedRoot.w_130_0 ctx))

def generatedRoot.w_132_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ((generatedRoot.w_131_0 ctx) + (generatedRoot.w_130_1 ctx))

def generatedRoot.w_133_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ((generatedRoot.w_132_0 ctx) + (generatedRoot.w_130_2 ctx))

def generatedRoot.w_134_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ((generatedRoot.w_133_0 ctx) + (generatedRoot.w_130_3 ctx))

def generatedRoot.w_135_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ((generatedRoot.w_134_0 ctx) + (generatedRoot.w_130_4 ctx))

def generatedRoot.w_136_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ((generatedRoot.w_135_0 ctx) + (generatedRoot.w_130_5 ctx))

def generatedRoot.w_137_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ((generatedRoot.w_136_0 ctx) + (generatedRoot.w_130_6 ctx))

def generatedRoot.w_138_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ((generatedRoot.w_137_0 ctx) + (generatedRoot.w_130_7 ctx))

def generatedRoot.w_140_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6 :=
  ctx.witness.w_140_0

def generatedRoot.w_141_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 :=
  (MxxRuntime.matrixPolynomial [(890511653275736298117428745327756462)] : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1)

def generatedRoot.w_142_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6 :=
  MxxRuntime.matrixMulScalarRight (generatedRoot.w_140_0 ctx) (generatedRoot.w_141_0 ctx)

def generatedRoot.w_144_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6 :=
  ctx.witness.w_144_0

def generatedRoot.w_145_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 :=
  (MxxRuntime.matrixPolynomial [(673373482644736804462909142109450233)] : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1)

def generatedRoot.w_146_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6 :=
  MxxRuntime.matrixMulScalarRight (generatedRoot.w_144_0 ctx) (generatedRoot.w_145_0 ctx)

def generatedRoot.w_147_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6 :=
  MxxRuntime.matrixAdd (generatedRoot.w_142_0 ctx) (generatedRoot.w_146_0 ctx)

def generatedRoot.w_149_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6 :=
  ctx.witness.w_149_0

def generatedRoot.w_150_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 :=
  (MxxRuntime.matrixPolynomial [(676801805491843546832002660143739331)] : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1)

def generatedRoot.w_151_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6 :=
  MxxRuntime.matrixMulScalarRight (generatedRoot.w_149_0 ctx) (generatedRoot.w_150_0 ctx)

def generatedRoot.w_152_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6 :=
  MxxRuntime.matrixAdd (generatedRoot.w_147_0 ctx) (generatedRoot.w_151_0 ctx)

def generatedRoot.w_154_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6 :=
  ctx.witness.w_154_0

def generatedRoot.w_155_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 :=
  (MxxRuntime.matrixPolynomial [(417760661884955689660474101877988633)] : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1)

def generatedRoot.w_156_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6 :=
  MxxRuntime.matrixMulScalarRight (generatedRoot.w_154_0 ctx) (generatedRoot.w_155_0 ctx)

def generatedRoot.w_157_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6 :=
  MxxRuntime.matrixAdd (generatedRoot.w_152_0 ctx) (generatedRoot.w_156_0 ctx)

def generatedRoot.w_158_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 2 6 :=
  ctx.witness.w_158_0

def generatedRoot.w_159_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6 :=
  ctx.witness.w_159_0

def generatedRoot.w_160_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_59_0 ctx) (generatedRoot.w_159_0 ctx)

def generatedRoot.w_161_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6 :=
  ctx.witness.w_161_0

def generatedRoot.w_162_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6 :=
  MxxRuntime.matrixAdd (generatedRoot.w_160_0 ctx) (generatedRoot.w_161_0 ctx)

def generatedRoot.w_163_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 :=
  (1 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1)

def generatedRoot.w_164_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_66_0 ctx) (generatedRoot.w_163_0 ctx)

def generatedRoot.w_165_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_164_0 ctx) (generatedRoot.w_72_0 ctx)

def generatedRoot.w_166_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_165_0 ctx) (generatedRoot.w_75_0 ctx)

def generatedRoot.w_167_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6 :=
  MxxRuntime.matrixSub (generatedRoot.w_162_0 ctx) (generatedRoot.w_166_0 ctx)

def generatedRoot.w_168_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6 :=
  MxxRuntime.matrixSub (generatedRoot.w_157_0 ctx) (generatedRoot.w_167_0 ctx)

def generatedRoot.w_169_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 :=
  ctx.witness.w_169_0

def generatedRoot.w_170_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ctx.witness.w_170_0

def generatedRoot.w_170_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ctx.witness.w_170_1

def generatedRoot.w_170_2 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ctx.witness.w_170_2

def generatedRoot.w_170_3 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ctx.witness.w_170_3

def generatedRoot.w_170_4 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ctx.witness.w_170_4

def generatedRoot.w_170_5 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ctx.witness.w_170_5

def generatedRoot.w_170_6 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ctx.witness.w_170_6

def generatedRoot.w_170_7 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ctx.witness.w_170_7

def generatedRoot.w_171_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ((generatedRoot.w_138_0 ctx) + (generatedRoot.w_170_0 ctx))

def generatedRoot.w_172_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ((generatedRoot.w_171_0 ctx) + (generatedRoot.w_170_1 ctx))

def generatedRoot.w_173_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ((generatedRoot.w_172_0 ctx) + (generatedRoot.w_170_2 ctx))

def generatedRoot.w_174_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ((generatedRoot.w_173_0 ctx) + (generatedRoot.w_170_3 ctx))

def generatedRoot.w_175_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ((generatedRoot.w_174_0 ctx) + (generatedRoot.w_170_4 ctx))

def generatedRoot.w_176_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ((generatedRoot.w_175_0 ctx) + (generatedRoot.w_170_5 ctx))

def generatedRoot.w_177_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ((generatedRoot.w_176_0 ctx) + (generatedRoot.w_170_6 ctx))

def generatedRoot.w_178_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ((generatedRoot.w_177_0 ctx) + (generatedRoot.w_170_7 ctx))

def generatedRoot.w_179_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 :=
  ctx.witness.w_179_0

def generatedRoot.w_180_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ctx.witness.w_180_0

def generatedRoot.w_180_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ctx.witness.w_180_1

def generatedRoot.w_180_2 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ctx.witness.w_180_2

def generatedRoot.w_180_3 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ctx.witness.w_180_3

def generatedRoot.w_180_4 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ctx.witness.w_180_4

def generatedRoot.w_180_5 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ctx.witness.w_180_5

def generatedRoot.w_180_6 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ctx.witness.w_180_6

def generatedRoot.w_180_7 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ctx.witness.w_180_7

def generatedRoot.w_181_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ((generatedRoot.w_178_0 ctx) + (generatedRoot.w_180_0 ctx))

def generatedRoot.w_182_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ((generatedRoot.w_181_0 ctx) + (generatedRoot.w_180_1 ctx))

def generatedRoot.w_183_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ((generatedRoot.w_182_0 ctx) + (generatedRoot.w_180_2 ctx))

def generatedRoot.w_184_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ((generatedRoot.w_183_0 ctx) + (generatedRoot.w_180_3 ctx))

def generatedRoot.w_185_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ((generatedRoot.w_184_0 ctx) + (generatedRoot.w_180_4 ctx))

def generatedRoot.w_186_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ((generatedRoot.w_185_0 ctx) + (generatedRoot.w_180_5 ctx))

def generatedRoot.w_187_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ((generatedRoot.w_186_0 ctx) + (generatedRoot.w_180_6 ctx))

def generatedRoot.w_188_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ((generatedRoot.w_187_0 ctx) + (generatedRoot.w_180_7 ctx))

def generatedRoot.w_189_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 :=
  ctx.witness.w_189_0

def generatedRoot.w_190_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ctx.witness.w_190_0

def generatedRoot.w_190_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ctx.witness.w_190_1

def generatedRoot.w_190_2 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ctx.witness.w_190_2

def generatedRoot.w_190_3 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ctx.witness.w_190_3

def generatedRoot.w_190_4 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ctx.witness.w_190_4

def generatedRoot.w_190_5 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ctx.witness.w_190_5

def generatedRoot.w_190_6 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ctx.witness.w_190_6

def generatedRoot.w_190_7 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ctx.witness.w_190_7

def generatedRoot.w_191_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ((generatedRoot.w_188_0 ctx) + (generatedRoot.w_190_0 ctx))

def generatedRoot.w_192_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ((generatedRoot.w_191_0 ctx) + (generatedRoot.w_190_1 ctx))

def generatedRoot.w_193_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ((generatedRoot.w_192_0 ctx) + (generatedRoot.w_190_2 ctx))

def generatedRoot.w_194_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ((generatedRoot.w_193_0 ctx) + (generatedRoot.w_190_3 ctx))

def generatedRoot.w_195_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ((generatedRoot.w_194_0 ctx) + (generatedRoot.w_190_4 ctx))

def generatedRoot.w_196_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ((generatedRoot.w_195_0 ctx) + (generatedRoot.w_190_5 ctx))

def generatedRoot.w_197_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ((generatedRoot.w_196_0 ctx) + (generatedRoot.w_190_6 ctx))

def generatedRoot.w_198_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ((generatedRoot.w_197_0 ctx) + (generatedRoot.w_190_7 ctx))

def generatedRoot.w_199_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 :=
  ctx.witness.w_199_0

def generatedRoot.w_200_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ctx.witness.w_200_0

def generatedRoot.w_200_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ctx.witness.w_200_1

def generatedRoot.w_200_2 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ctx.witness.w_200_2

def generatedRoot.w_200_3 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ctx.witness.w_200_3

def generatedRoot.w_200_4 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ctx.witness.w_200_4

def generatedRoot.w_200_5 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ctx.witness.w_200_5

def generatedRoot.w_200_6 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ctx.witness.w_200_6

def generatedRoot.w_200_7 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ctx.witness.w_200_7

def generatedRoot.w_201_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ((generatedRoot.w_198_0 ctx) + (generatedRoot.w_200_0 ctx))

def generatedRoot.w_202_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ((generatedRoot.w_201_0 ctx) + (generatedRoot.w_200_1 ctx))

def generatedRoot.w_203_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ((generatedRoot.w_202_0 ctx) + (generatedRoot.w_200_2 ctx))

def generatedRoot.w_204_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ((generatedRoot.w_203_0 ctx) + (generatedRoot.w_200_3 ctx))

def generatedRoot.w_205_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ((generatedRoot.w_204_0 ctx) + (generatedRoot.w_200_4 ctx))

def generatedRoot.w_206_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ((generatedRoot.w_205_0 ctx) + (generatedRoot.w_200_5 ctx))

def generatedRoot.w_207_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ((generatedRoot.w_206_0 ctx) + (generatedRoot.w_200_6 ctx))

def generatedRoot.w_208_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ((generatedRoot.w_207_0 ctx) + (generatedRoot.w_200_7 ctx))

def generatedRoot.w_209_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 :=
  ctx.witness.w_209_0

def generatedRoot.w_210_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ctx.witness.w_210_0

def generatedRoot.w_210_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ctx.witness.w_210_1

def generatedRoot.w_210_2 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ctx.witness.w_210_2

def generatedRoot.w_210_3 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ctx.witness.w_210_3

def generatedRoot.w_210_4 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ctx.witness.w_210_4

def generatedRoot.w_210_5 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ctx.witness.w_210_5

def generatedRoot.w_210_6 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ctx.witness.w_210_6

def generatedRoot.w_210_7 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ctx.witness.w_210_7

def generatedRoot.w_211_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ((generatedRoot.w_208_0 ctx) + (generatedRoot.w_210_0 ctx))

def generatedRoot.w_212_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ((generatedRoot.w_211_0 ctx) + (generatedRoot.w_210_1 ctx))

def generatedRoot.w_213_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ((generatedRoot.w_212_0 ctx) + (generatedRoot.w_210_2 ctx))

def generatedRoot.w_214_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ((generatedRoot.w_213_0 ctx) + (generatedRoot.w_210_3 ctx))

def generatedRoot.w_215_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ((generatedRoot.w_214_0 ctx) + (generatedRoot.w_210_4 ctx))

def generatedRoot.w_216_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ((generatedRoot.w_215_0 ctx) + (generatedRoot.w_210_5 ctx))

def generatedRoot.w_217_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ((generatedRoot.w_216_0 ctx) + (generatedRoot.w_210_6 ctx))

def generatedRoot.w_218_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ((generatedRoot.w_217_0 ctx) + (generatedRoot.w_210_7 ctx))

def generatedRoot.w_219_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 :=
  ctx.witness.w_219_0

def generatedRoot.w_220_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ctx.witness.w_220_0

def generatedRoot.w_220_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ctx.witness.w_220_1

def generatedRoot.w_220_2 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ctx.witness.w_220_2

def generatedRoot.w_220_3 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ctx.witness.w_220_3

def generatedRoot.w_220_4 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ctx.witness.w_220_4

def generatedRoot.w_220_5 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ctx.witness.w_220_5

def generatedRoot.w_220_6 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ctx.witness.w_220_6

def generatedRoot.w_220_7 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ctx.witness.w_220_7

def generatedRoot.w_221_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ((generatedRoot.w_218_0 ctx) + (generatedRoot.w_220_0 ctx))

def generatedRoot.w_222_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ((generatedRoot.w_221_0 ctx) + (generatedRoot.w_220_1 ctx))

def generatedRoot.w_223_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ((generatedRoot.w_222_0 ctx) + (generatedRoot.w_220_2 ctx))

def generatedRoot.w_224_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ((generatedRoot.w_223_0 ctx) + (generatedRoot.w_220_3 ctx))

def generatedRoot.w_225_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ((generatedRoot.w_224_0 ctx) + (generatedRoot.w_220_4 ctx))

def generatedRoot.w_226_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ((generatedRoot.w_225_0 ctx) + (generatedRoot.w_220_5 ctx))

def generatedRoot.w_227_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ((generatedRoot.w_226_0 ctx) + (generatedRoot.w_220_6 ctx))

def generatedRoot.w_228_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ((generatedRoot.w_227_0 ctx) + (generatedRoot.w_220_7 ctx))

def generatedRoot.outs {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  (generatedRoot.w_228_0 ctx)

structure generatedRoot.Facts_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Prop where
  h2_0 : MxxRuntime.centeredRebaseRuns (generatedRoot.w_1_0 ctx) (generatedRoot.w_2_0 ctx)
  h6_0 : MxxRuntime.centeredRebaseRuns (generatedRoot.w_5_0 ctx) (generatedRoot.w_6_0 ctx)
  h11_0 : MxxRuntime.centeredRebaseRuns (generatedRoot.w_10_0 ctx) (generatedRoot.w_11_0 ctx)
  h16_0 : MxxRuntime.centeredRebaseRuns (generatedRoot.w_15_0 ctx) (generatedRoot.w_16_0 ctx)
  h21_0 : MxxRuntime.familyGetStatic (generatedRoot.w_20_0 ctx) (0) (generatedRoot.w_21_0 ctx)
  h25_0 : MxxRuntime.familyGetStatic (generatedRoot.w_20_0 ctx) (1) (generatedRoot.w_25_0 ctx)
  h30_0 : MxxRuntime.familyGetStatic (generatedRoot.w_20_0 ctx) (2) (generatedRoot.w_30_0 ctx)
  h35_0 : MxxRuntime.familyGetStatic (generatedRoot.w_20_0 ctx) (3) (generatedRoot.w_35_0 ctx)
  h40_0 : MxxRuntime.familyGetStatic (generatedRoot.w_20_0 ctx) (4) (generatedRoot.w_40_0 ctx)
  h45_0 : MxxRuntime.familyGetStatic (generatedRoot.w_20_0 ctx) (5) (generatedRoot.w_45_0 ctx)
  h50_0 : MxxRuntime.familyGetStatic (generatedRoot.w_20_0 ctx) (6) (generatedRoot.w_50_0 ctx)
  h55_0 : MxxRuntime.familyGetStatic (generatedRoot.w_20_0 ctx) (7) (generatedRoot.w_55_0 ctx)
  h61_0 : MxxRuntime.hashSample (hashModel) ([114, 101, 102, 114, 101, 115, 104, 47, 118, 49, 47, 65, 116, 111, 112, 45, 114, 101, 102, 114, 101, 115, 104, 45, 114, 111, 117, 110, 100, 45, 116, 114, 105, 112]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((generatedRoot.w_60_0 ctx)) (generatedRoot.w_61_0 ctx)
  t13 : 0 ≤ 0
  t14 : 0 < 1
  t15 : 1 ≤ 2
  t16 : 0 ≤ 0
  t17 : 0 < 6
  t18 : 6 ≤ 6
  h62_0 : MxxRuntime.sliceMatrix (generatedRoot.w_61_0 ctx) 0 1 0 6 (generatedRoot.w_62_0 ctx)
  t20 : 0 ≤ 1
  t21 : 1 < 2
  t22 : 2 ≤ 2
  t23 : 0 ≤ 0
  t24 : 0 < 6
  t25 : 6 ≤ 6
  h64_0 : MxxRuntime.sliceMatrix (generatedRoot.w_61_0 ctx) 1 2 0 6 (generatedRoot.w_64_0 ctx)
  h74_0 : MxxRuntime.gadgetMatrixRuns backend 32768 6 (generatedRoot.w_74_0 ctx)
  h75_0 : MxxRuntime.centeredRebaseRuns (generatedRoot.w_74_0 ctx) (generatedRoot.w_75_0 ctx)
  t29 : 0 ≤ 0
  t30 : 0 < 1
  t31 : 1 ≤ 1

theorem generatedRoot.facts_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {ctx : generatedRoot.Ctx backend hashModel params} {outputs : Int}
    (h : generatedRoot.constraints_0 backend hashModel (generatedRoot.w_1_0 ctx) (generatedRoot.w_5_0 ctx) (generatedRoot.w_10_0 ctx) (generatedRoot.w_15_0 ctx) (generatedRoot.w_20_0 ctx) (generatedRoot.w_60_0 ctx) (generatedRoot.w_139_0 ctx) (generatedRoot.w_143_0 ctx) (generatedRoot.w_148_0 ctx) (generatedRoot.w_153_0 ctx) (generatedRoot.w_2_0 ctx) (generatedRoot.w_6_0 ctx) (generatedRoot.w_11_0 ctx) (generatedRoot.w_16_0 ctx) (generatedRoot.w_21_0 ctx) (generatedRoot.w_25_0 ctx) (generatedRoot.w_30_0 ctx) (generatedRoot.w_35_0 ctx) (generatedRoot.w_40_0 ctx) (generatedRoot.w_45_0 ctx) (generatedRoot.w_50_0 ctx) (generatedRoot.w_55_0 ctx) (generatedRoot.w_61_0 ctx) (generatedRoot.w_62_0 ctx) (generatedRoot.w_64_0 ctx) (generatedRoot.w_74_0 ctx) (generatedRoot.w_75_0 ctx) (generatedRoot.w_78_0 ctx) (generatedRoot.w_79_0 ctx) (generatedRoot.w_80_0 ctx) (generatedRoot.w_80_1 ctx) (generatedRoot.w_80_2 ctx) (generatedRoot.w_80_3 ctx) (generatedRoot.w_80_4 ctx) (generatedRoot.w_80_5 ctx) (generatedRoot.w_80_6 ctx) (generatedRoot.w_80_7 ctx) (generatedRoot.w_89_0 ctx) (generatedRoot.w_90_0 ctx) (generatedRoot.w_90_1 ctx) (generatedRoot.w_90_2 ctx) (generatedRoot.w_90_3 ctx) (generatedRoot.w_90_4 ctx) (generatedRoot.w_90_5 ctx) (generatedRoot.w_90_6 ctx) (generatedRoot.w_90_7 ctx) (generatedRoot.w_99_0 ctx) (generatedRoot.w_100_0 ctx) (generatedRoot.w_100_1 ctx) (generatedRoot.w_100_2 ctx) (generatedRoot.w_100_3 ctx) (generatedRoot.w_100_4 ctx) (generatedRoot.w_100_5 ctx) (generatedRoot.w_100_6 ctx) (generatedRoot.w_100_7 ctx) (generatedRoot.w_109_0 ctx) (generatedRoot.w_110_0 ctx) (generatedRoot.w_110_1 ctx) (generatedRoot.w_110_2 ctx) (generatedRoot.w_110_3 ctx) (generatedRoot.w_110_4 ctx) (generatedRoot.w_110_5 ctx) (generatedRoot.w_110_6 ctx) (generatedRoot.w_110_7 ctx) (generatedRoot.w_119_0 ctx) (generatedRoot.w_120_0 ctx) (generatedRoot.w_120_1 ctx) (generatedRoot.w_120_2 ctx) (generatedRoot.w_120_3 ctx) (generatedRoot.w_120_4 ctx) (generatedRoot.w_120_5 ctx) (generatedRoot.w_120_6 ctx) (generatedRoot.w_120_7 ctx) (generatedRoot.w_129_0 ctx) (generatedRoot.w_130_0 ctx) (generatedRoot.w_130_1 ctx) (generatedRoot.w_130_2 ctx) (generatedRoot.w_130_3 ctx) (generatedRoot.w_130_4 ctx) (generatedRoot.w_130_5 ctx) (generatedRoot.w_130_6 ctx) (generatedRoot.w_130_7 ctx) (generatedRoot.w_140_0 ctx) (generatedRoot.w_144_0 ctx) (generatedRoot.w_149_0 ctx) (generatedRoot.w_154_0 ctx) (generatedRoot.w_158_0 ctx) (generatedRoot.w_159_0 ctx) (generatedRoot.w_161_0 ctx) (generatedRoot.w_168_0 ctx) (generatedRoot.w_169_0 ctx) (generatedRoot.w_170_0 ctx) (generatedRoot.w_170_1 ctx) (generatedRoot.w_170_2 ctx) (generatedRoot.w_170_3 ctx) (generatedRoot.w_170_4 ctx) (generatedRoot.w_170_5 ctx) (generatedRoot.w_170_6 ctx) (generatedRoot.w_170_7 ctx) (generatedRoot.w_179_0 ctx) (generatedRoot.w_180_0 ctx) (generatedRoot.w_180_1 ctx) (generatedRoot.w_180_2 ctx) (generatedRoot.w_180_3 ctx) (generatedRoot.w_180_4 ctx) (generatedRoot.w_180_5 ctx) (generatedRoot.w_180_6 ctx) (generatedRoot.w_180_7 ctx) (generatedRoot.w_189_0 ctx) (generatedRoot.w_190_0 ctx) (generatedRoot.w_190_1 ctx) (generatedRoot.w_190_2 ctx) (generatedRoot.w_190_3 ctx) (generatedRoot.w_190_4 ctx) (generatedRoot.w_190_5 ctx) (generatedRoot.w_190_6 ctx) (generatedRoot.w_190_7 ctx) (generatedRoot.w_199_0 ctx) (generatedRoot.w_200_0 ctx) (generatedRoot.w_200_1 ctx) (generatedRoot.w_200_2 ctx) (generatedRoot.w_200_3 ctx) (generatedRoot.w_200_4 ctx) (generatedRoot.w_200_5 ctx) (generatedRoot.w_200_6 ctx) (generatedRoot.w_200_7 ctx) (generatedRoot.w_209_0 ctx) (generatedRoot.w_210_0 ctx) (generatedRoot.w_210_1 ctx) (generatedRoot.w_210_2 ctx) (generatedRoot.w_210_3 ctx) (generatedRoot.w_210_4 ctx) (generatedRoot.w_210_5 ctx) (generatedRoot.w_210_6 ctx) (generatedRoot.w_210_7 ctx) (generatedRoot.w_219_0 ctx) (generatedRoot.w_220_0 ctx) (generatedRoot.w_220_1 ctx) (generatedRoot.w_220_2 ctx) (generatedRoot.w_220_3 ctx) (generatedRoot.w_220_4 ctx) (generatedRoot.w_220_5 ctx) (generatedRoot.w_220_6 ctx) (generatedRoot.w_220_7 ctx) (generatedRoot.w_228_0 ctx) outputs) :
    generatedRoot.Facts_0 ctx ∧ generatedRoot.constraints_1 hashModel (generatedRoot.w_60_0 ctx) (generatedRoot.w_139_0 ctx) (generatedRoot.w_143_0 ctx) (generatedRoot.w_148_0 ctx) (generatedRoot.w_153_0 ctx) (generatedRoot.w_78_0 ctx) (generatedRoot.w_79_0 ctx) (generatedRoot.w_80_0 ctx) (generatedRoot.w_80_1 ctx) (generatedRoot.w_80_2 ctx) (generatedRoot.w_80_3 ctx) (generatedRoot.w_80_4 ctx) (generatedRoot.w_80_5 ctx) (generatedRoot.w_80_6 ctx) (generatedRoot.w_80_7 ctx) (generatedRoot.w_89_0 ctx) (generatedRoot.w_90_0 ctx) (generatedRoot.w_90_1 ctx) (generatedRoot.w_90_2 ctx) (generatedRoot.w_90_3 ctx) (generatedRoot.w_90_4 ctx) (generatedRoot.w_90_5 ctx) (generatedRoot.w_90_6 ctx) (generatedRoot.w_90_7 ctx) (generatedRoot.w_99_0 ctx) (generatedRoot.w_100_0 ctx) (generatedRoot.w_100_1 ctx) (generatedRoot.w_100_2 ctx) (generatedRoot.w_100_3 ctx) (generatedRoot.w_100_4 ctx) (generatedRoot.w_100_5 ctx) (generatedRoot.w_100_6 ctx) (generatedRoot.w_100_7 ctx) (generatedRoot.w_109_0 ctx) (generatedRoot.w_110_0 ctx) (generatedRoot.w_110_1 ctx) (generatedRoot.w_110_2 ctx) (generatedRoot.w_110_3 ctx) (generatedRoot.w_110_4 ctx) (generatedRoot.w_110_5 ctx) (generatedRoot.w_110_6 ctx) (generatedRoot.w_110_7 ctx) (generatedRoot.w_119_0 ctx) (generatedRoot.w_120_0 ctx) (generatedRoot.w_120_1 ctx) (generatedRoot.w_120_2 ctx) (generatedRoot.w_120_3 ctx) (generatedRoot.w_120_4 ctx) (generatedRoot.w_120_5 ctx) (generatedRoot.w_120_6 ctx) (generatedRoot.w_120_7 ctx) (generatedRoot.w_129_0 ctx) (generatedRoot.w_130_0 ctx) (generatedRoot.w_130_1 ctx) (generatedRoot.w_130_2 ctx) (generatedRoot.w_130_3 ctx) (generatedRoot.w_130_4 ctx) (generatedRoot.w_130_5 ctx) (generatedRoot.w_130_6 ctx) (generatedRoot.w_130_7 ctx) (generatedRoot.w_140_0 ctx) (generatedRoot.w_144_0 ctx) (generatedRoot.w_149_0 ctx) (generatedRoot.w_154_0 ctx) (generatedRoot.w_158_0 ctx) (generatedRoot.w_159_0 ctx) (generatedRoot.w_161_0 ctx) (generatedRoot.w_168_0 ctx) (generatedRoot.w_169_0 ctx) (generatedRoot.w_170_0 ctx) (generatedRoot.w_170_1 ctx) (generatedRoot.w_170_2 ctx) (generatedRoot.w_170_3 ctx) (generatedRoot.w_170_4 ctx) (generatedRoot.w_170_5 ctx) (generatedRoot.w_170_6 ctx) (generatedRoot.w_170_7 ctx) (generatedRoot.w_179_0 ctx) (generatedRoot.w_180_0 ctx) (generatedRoot.w_180_1 ctx) (generatedRoot.w_180_2 ctx) (generatedRoot.w_180_3 ctx) (generatedRoot.w_180_4 ctx) (generatedRoot.w_180_5 ctx) (generatedRoot.w_180_6 ctx) (generatedRoot.w_180_7 ctx) (generatedRoot.w_189_0 ctx) (generatedRoot.w_190_0 ctx) (generatedRoot.w_190_1 ctx) (generatedRoot.w_190_2 ctx) (generatedRoot.w_190_3 ctx) (generatedRoot.w_190_4 ctx) (generatedRoot.w_190_5 ctx) (generatedRoot.w_190_6 ctx) (generatedRoot.w_190_7 ctx) (generatedRoot.w_199_0 ctx) (generatedRoot.w_200_0 ctx) (generatedRoot.w_200_1 ctx) (generatedRoot.w_200_2 ctx) (generatedRoot.w_200_3 ctx) (generatedRoot.w_200_4 ctx) (generatedRoot.w_200_5 ctx) (generatedRoot.w_200_6 ctx) (generatedRoot.w_200_7 ctx) (generatedRoot.w_209_0 ctx) (generatedRoot.w_210_0 ctx) (generatedRoot.w_210_1 ctx) (generatedRoot.w_210_2 ctx) (generatedRoot.w_210_3 ctx) (generatedRoot.w_210_4 ctx) (generatedRoot.w_210_5 ctx) (generatedRoot.w_210_6 ctx) (generatedRoot.w_210_7 ctx) (generatedRoot.w_219_0 ctx) (generatedRoot.w_220_0 ctx) (generatedRoot.w_220_1 ctx) (generatedRoot.w_220_2 ctx) (generatedRoot.w_220_3 ctx) (generatedRoot.w_220_4 ctx) (generatedRoot.w_220_5 ctx) (generatedRoot.w_220_6 ctx) (generatedRoot.w_220_7 ctx) (generatedRoot.w_228_0 ctx) outputs :=
  ⟨⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1⟩, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Prop where
  t32 : 0 ≤ 0
  t33 : 0 < 1
  t34 : 1 ≤ 6
  h79_0 : MxxRuntime.sliceMatrix (generatedRoot.w_78_0 ctx) 0 1 0 1 (generatedRoot.w_79_0 ctx)
  t36 : (8) = 8
  h80_0 : MxxRuntime.thresholdDecode (2) (8) 0 (generatedRoot.w_79_0 ctx) (generatedRoot.w_80_0 ctx)
  h80_1 : MxxRuntime.thresholdDecode (2) (8) 1 (generatedRoot.w_79_0 ctx) (generatedRoot.w_80_1 ctx)
  h80_2 : MxxRuntime.thresholdDecode (2) (8) 2 (generatedRoot.w_79_0 ctx) (generatedRoot.w_80_2 ctx)
  h80_3 : MxxRuntime.thresholdDecode (2) (8) 3 (generatedRoot.w_79_0 ctx) (generatedRoot.w_80_3 ctx)
  h80_4 : MxxRuntime.thresholdDecode (2) (8) 4 (generatedRoot.w_79_0 ctx) (generatedRoot.w_80_4 ctx)
  h80_5 : MxxRuntime.thresholdDecode (2) (8) 5 (generatedRoot.w_79_0 ctx) (generatedRoot.w_80_5 ctx)
  h80_6 : MxxRuntime.thresholdDecode (2) (8) 6 (generatedRoot.w_79_0 ctx) (generatedRoot.w_80_6 ctx)
  h80_7 : MxxRuntime.thresholdDecode (2) (8) 7 (generatedRoot.w_79_0 ctx) (generatedRoot.w_80_7 ctx)
  t45 : 0 ≤ 0
  t46 : 0 < 1
  t47 : 1 ≤ 1
  t48 : 0 ≤ 1
  t49 : 1 < 2
  t50 : 2 ≤ 6
  h89_0 : MxxRuntime.sliceMatrix (generatedRoot.w_78_0 ctx) 0 1 1 2 (generatedRoot.w_89_0 ctx)
  t52 : (8) = 8
  h90_0 : MxxRuntime.thresholdDecode (2) (8) 0 (generatedRoot.w_89_0 ctx) (generatedRoot.w_90_0 ctx)
  h90_1 : MxxRuntime.thresholdDecode (2) (8) 1 (generatedRoot.w_89_0 ctx) (generatedRoot.w_90_1 ctx)
  h90_2 : MxxRuntime.thresholdDecode (2) (8) 2 (generatedRoot.w_89_0 ctx) (generatedRoot.w_90_2 ctx)
  h90_3 : MxxRuntime.thresholdDecode (2) (8) 3 (generatedRoot.w_89_0 ctx) (generatedRoot.w_90_3 ctx)
  h90_4 : MxxRuntime.thresholdDecode (2) (8) 4 (generatedRoot.w_89_0 ctx) (generatedRoot.w_90_4 ctx)
  h90_5 : MxxRuntime.thresholdDecode (2) (8) 5 (generatedRoot.w_89_0 ctx) (generatedRoot.w_90_5 ctx)
  h90_6 : MxxRuntime.thresholdDecode (2) (8) 6 (generatedRoot.w_89_0 ctx) (generatedRoot.w_90_6 ctx)
  h90_7 : MxxRuntime.thresholdDecode (2) (8) 7 (generatedRoot.w_89_0 ctx) (generatedRoot.w_90_7 ctx)
  t61 : 0 ≤ 0
  t62 : 0 < 1
  t63 : 1 ≤ 1

theorem generatedRoot.facts_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {ctx : generatedRoot.Ctx backend hashModel params} {outputs : Int}
    (h : generatedRoot.constraints_1 hashModel (generatedRoot.w_60_0 ctx) (generatedRoot.w_139_0 ctx) (generatedRoot.w_143_0 ctx) (generatedRoot.w_148_0 ctx) (generatedRoot.w_153_0 ctx) (generatedRoot.w_78_0 ctx) (generatedRoot.w_79_0 ctx) (generatedRoot.w_80_0 ctx) (generatedRoot.w_80_1 ctx) (generatedRoot.w_80_2 ctx) (generatedRoot.w_80_3 ctx) (generatedRoot.w_80_4 ctx) (generatedRoot.w_80_5 ctx) (generatedRoot.w_80_6 ctx) (generatedRoot.w_80_7 ctx) (generatedRoot.w_89_0 ctx) (generatedRoot.w_90_0 ctx) (generatedRoot.w_90_1 ctx) (generatedRoot.w_90_2 ctx) (generatedRoot.w_90_3 ctx) (generatedRoot.w_90_4 ctx) (generatedRoot.w_90_5 ctx) (generatedRoot.w_90_6 ctx) (generatedRoot.w_90_7 ctx) (generatedRoot.w_99_0 ctx) (generatedRoot.w_100_0 ctx) (generatedRoot.w_100_1 ctx) (generatedRoot.w_100_2 ctx) (generatedRoot.w_100_3 ctx) (generatedRoot.w_100_4 ctx) (generatedRoot.w_100_5 ctx) (generatedRoot.w_100_6 ctx) (generatedRoot.w_100_7 ctx) (generatedRoot.w_109_0 ctx) (generatedRoot.w_110_0 ctx) (generatedRoot.w_110_1 ctx) (generatedRoot.w_110_2 ctx) (generatedRoot.w_110_3 ctx) (generatedRoot.w_110_4 ctx) (generatedRoot.w_110_5 ctx) (generatedRoot.w_110_6 ctx) (generatedRoot.w_110_7 ctx) (generatedRoot.w_119_0 ctx) (generatedRoot.w_120_0 ctx) (generatedRoot.w_120_1 ctx) (generatedRoot.w_120_2 ctx) (generatedRoot.w_120_3 ctx) (generatedRoot.w_120_4 ctx) (generatedRoot.w_120_5 ctx) (generatedRoot.w_120_6 ctx) (generatedRoot.w_120_7 ctx) (generatedRoot.w_129_0 ctx) (generatedRoot.w_130_0 ctx) (generatedRoot.w_130_1 ctx) (generatedRoot.w_130_2 ctx) (generatedRoot.w_130_3 ctx) (generatedRoot.w_130_4 ctx) (generatedRoot.w_130_5 ctx) (generatedRoot.w_130_6 ctx) (generatedRoot.w_130_7 ctx) (generatedRoot.w_140_0 ctx) (generatedRoot.w_144_0 ctx) (generatedRoot.w_149_0 ctx) (generatedRoot.w_154_0 ctx) (generatedRoot.w_158_0 ctx) (generatedRoot.w_159_0 ctx) (generatedRoot.w_161_0 ctx) (generatedRoot.w_168_0 ctx) (generatedRoot.w_169_0 ctx) (generatedRoot.w_170_0 ctx) (generatedRoot.w_170_1 ctx) (generatedRoot.w_170_2 ctx) (generatedRoot.w_170_3 ctx) (generatedRoot.w_170_4 ctx) (generatedRoot.w_170_5 ctx) (generatedRoot.w_170_6 ctx) (generatedRoot.w_170_7 ctx) (generatedRoot.w_179_0 ctx) (generatedRoot.w_180_0 ctx) (generatedRoot.w_180_1 ctx) (generatedRoot.w_180_2 ctx) (generatedRoot.w_180_3 ctx) (generatedRoot.w_180_4 ctx) (generatedRoot.w_180_5 ctx) (generatedRoot.w_180_6 ctx) (generatedRoot.w_180_7 ctx) (generatedRoot.w_189_0 ctx) (generatedRoot.w_190_0 ctx) (generatedRoot.w_190_1 ctx) (generatedRoot.w_190_2 ctx) (generatedRoot.w_190_3 ctx) (generatedRoot.w_190_4 ctx) (generatedRoot.w_190_5 ctx) (generatedRoot.w_190_6 ctx) (generatedRoot.w_190_7 ctx) (generatedRoot.w_199_0 ctx) (generatedRoot.w_200_0 ctx) (generatedRoot.w_200_1 ctx) (generatedRoot.w_200_2 ctx) (generatedRoot.w_200_3 ctx) (generatedRoot.w_200_4 ctx) (generatedRoot.w_200_5 ctx) (generatedRoot.w_200_6 ctx) (generatedRoot.w_200_7 ctx) (generatedRoot.w_209_0 ctx) (generatedRoot.w_210_0 ctx) (generatedRoot.w_210_1 ctx) (generatedRoot.w_210_2 ctx) (generatedRoot.w_210_3 ctx) (generatedRoot.w_210_4 ctx) (generatedRoot.w_210_5 ctx) (generatedRoot.w_210_6 ctx) (generatedRoot.w_210_7 ctx) (generatedRoot.w_219_0 ctx) (generatedRoot.w_220_0 ctx) (generatedRoot.w_220_1 ctx) (generatedRoot.w_220_2 ctx) (generatedRoot.w_220_3 ctx) (generatedRoot.w_220_4 ctx) (generatedRoot.w_220_5 ctx) (generatedRoot.w_220_6 ctx) (generatedRoot.w_220_7 ctx) (generatedRoot.w_228_0 ctx) outputs) :
    generatedRoot.Facts_1 ctx ∧ generatedRoot.constraints_2 hashModel (generatedRoot.w_60_0 ctx) (generatedRoot.w_139_0 ctx) (generatedRoot.w_143_0 ctx) (generatedRoot.w_148_0 ctx) (generatedRoot.w_153_0 ctx) (generatedRoot.w_78_0 ctx) (generatedRoot.w_99_0 ctx) (generatedRoot.w_100_0 ctx) (generatedRoot.w_100_1 ctx) (generatedRoot.w_100_2 ctx) (generatedRoot.w_100_3 ctx) (generatedRoot.w_100_4 ctx) (generatedRoot.w_100_5 ctx) (generatedRoot.w_100_6 ctx) (generatedRoot.w_100_7 ctx) (generatedRoot.w_109_0 ctx) (generatedRoot.w_110_0 ctx) (generatedRoot.w_110_1 ctx) (generatedRoot.w_110_2 ctx) (generatedRoot.w_110_3 ctx) (generatedRoot.w_110_4 ctx) (generatedRoot.w_110_5 ctx) (generatedRoot.w_110_6 ctx) (generatedRoot.w_110_7 ctx) (generatedRoot.w_119_0 ctx) (generatedRoot.w_120_0 ctx) (generatedRoot.w_120_1 ctx) (generatedRoot.w_120_2 ctx) (generatedRoot.w_120_3 ctx) (generatedRoot.w_120_4 ctx) (generatedRoot.w_120_5 ctx) (generatedRoot.w_120_6 ctx) (generatedRoot.w_120_7 ctx) (generatedRoot.w_129_0 ctx) (generatedRoot.w_130_0 ctx) (generatedRoot.w_130_1 ctx) (generatedRoot.w_130_2 ctx) (generatedRoot.w_130_3 ctx) (generatedRoot.w_130_4 ctx) (generatedRoot.w_130_5 ctx) (generatedRoot.w_130_6 ctx) (generatedRoot.w_130_7 ctx) (generatedRoot.w_140_0 ctx) (generatedRoot.w_144_0 ctx) (generatedRoot.w_149_0 ctx) (generatedRoot.w_154_0 ctx) (generatedRoot.w_158_0 ctx) (generatedRoot.w_159_0 ctx) (generatedRoot.w_161_0 ctx) (generatedRoot.w_168_0 ctx) (generatedRoot.w_169_0 ctx) (generatedRoot.w_170_0 ctx) (generatedRoot.w_170_1 ctx) (generatedRoot.w_170_2 ctx) (generatedRoot.w_170_3 ctx) (generatedRoot.w_170_4 ctx) (generatedRoot.w_170_5 ctx) (generatedRoot.w_170_6 ctx) (generatedRoot.w_170_7 ctx) (generatedRoot.w_179_0 ctx) (generatedRoot.w_180_0 ctx) (generatedRoot.w_180_1 ctx) (generatedRoot.w_180_2 ctx) (generatedRoot.w_180_3 ctx) (generatedRoot.w_180_4 ctx) (generatedRoot.w_180_5 ctx) (generatedRoot.w_180_6 ctx) (generatedRoot.w_180_7 ctx) (generatedRoot.w_189_0 ctx) (generatedRoot.w_190_0 ctx) (generatedRoot.w_190_1 ctx) (generatedRoot.w_190_2 ctx) (generatedRoot.w_190_3 ctx) (generatedRoot.w_190_4 ctx) (generatedRoot.w_190_5 ctx) (generatedRoot.w_190_6 ctx) (generatedRoot.w_190_7 ctx) (generatedRoot.w_199_0 ctx) (generatedRoot.w_200_0 ctx) (generatedRoot.w_200_1 ctx) (generatedRoot.w_200_2 ctx) (generatedRoot.w_200_3 ctx) (generatedRoot.w_200_4 ctx) (generatedRoot.w_200_5 ctx) (generatedRoot.w_200_6 ctx) (generatedRoot.w_200_7 ctx) (generatedRoot.w_209_0 ctx) (generatedRoot.w_210_0 ctx) (generatedRoot.w_210_1 ctx) (generatedRoot.w_210_2 ctx) (generatedRoot.w_210_3 ctx) (generatedRoot.w_210_4 ctx) (generatedRoot.w_210_5 ctx) (generatedRoot.w_210_6 ctx) (generatedRoot.w_210_7 ctx) (generatedRoot.w_219_0 ctx) (generatedRoot.w_220_0 ctx) (generatedRoot.w_220_1 ctx) (generatedRoot.w_220_2 ctx) (generatedRoot.w_220_3 ctx) (generatedRoot.w_220_4 ctx) (generatedRoot.w_220_5 ctx) (generatedRoot.w_220_6 ctx) (generatedRoot.w_220_7 ctx) (generatedRoot.w_228_0 ctx) outputs :=
  ⟨⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1⟩, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_2 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Prop where
  t64 : 0 ≤ 2
  t65 : 2 < 3
  t66 : 3 ≤ 6
  h99_0 : MxxRuntime.sliceMatrix (generatedRoot.w_78_0 ctx) 0 1 2 3 (generatedRoot.w_99_0 ctx)
  t68 : (8) = 8
  h100_0 : MxxRuntime.thresholdDecode (2) (8) 0 (generatedRoot.w_99_0 ctx) (generatedRoot.w_100_0 ctx)
  h100_1 : MxxRuntime.thresholdDecode (2) (8) 1 (generatedRoot.w_99_0 ctx) (generatedRoot.w_100_1 ctx)
  h100_2 : MxxRuntime.thresholdDecode (2) (8) 2 (generatedRoot.w_99_0 ctx) (generatedRoot.w_100_2 ctx)
  h100_3 : MxxRuntime.thresholdDecode (2) (8) 3 (generatedRoot.w_99_0 ctx) (generatedRoot.w_100_3 ctx)
  h100_4 : MxxRuntime.thresholdDecode (2) (8) 4 (generatedRoot.w_99_0 ctx) (generatedRoot.w_100_4 ctx)
  h100_5 : MxxRuntime.thresholdDecode (2) (8) 5 (generatedRoot.w_99_0 ctx) (generatedRoot.w_100_5 ctx)
  h100_6 : MxxRuntime.thresholdDecode (2) (8) 6 (generatedRoot.w_99_0 ctx) (generatedRoot.w_100_6 ctx)
  h100_7 : MxxRuntime.thresholdDecode (2) (8) 7 (generatedRoot.w_99_0 ctx) (generatedRoot.w_100_7 ctx)
  t77 : 0 ≤ 0
  t78 : 0 < 1
  t79 : 1 ≤ 1
  t80 : 0 ≤ 3
  t81 : 3 < 4
  t82 : 4 ≤ 6
  h109_0 : MxxRuntime.sliceMatrix (generatedRoot.w_78_0 ctx) 0 1 3 4 (generatedRoot.w_109_0 ctx)
  t84 : (8) = 8
  h110_0 : MxxRuntime.thresholdDecode (2) (8) 0 (generatedRoot.w_109_0 ctx) (generatedRoot.w_110_0 ctx)
  h110_1 : MxxRuntime.thresholdDecode (2) (8) 1 (generatedRoot.w_109_0 ctx) (generatedRoot.w_110_1 ctx)
  h110_2 : MxxRuntime.thresholdDecode (2) (8) 2 (generatedRoot.w_109_0 ctx) (generatedRoot.w_110_2 ctx)
  h110_3 : MxxRuntime.thresholdDecode (2) (8) 3 (generatedRoot.w_109_0 ctx) (generatedRoot.w_110_3 ctx)
  h110_4 : MxxRuntime.thresholdDecode (2) (8) 4 (generatedRoot.w_109_0 ctx) (generatedRoot.w_110_4 ctx)
  h110_5 : MxxRuntime.thresholdDecode (2) (8) 5 (generatedRoot.w_109_0 ctx) (generatedRoot.w_110_5 ctx)
  h110_6 : MxxRuntime.thresholdDecode (2) (8) 6 (generatedRoot.w_109_0 ctx) (generatedRoot.w_110_6 ctx)
  h110_7 : MxxRuntime.thresholdDecode (2) (8) 7 (generatedRoot.w_109_0 ctx) (generatedRoot.w_110_7 ctx)
  t93 : 0 ≤ 0
  t94 : 0 < 1
  t95 : 1 ≤ 1

theorem generatedRoot.facts_2 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {ctx : generatedRoot.Ctx backend hashModel params} {outputs : Int}
    (h : generatedRoot.constraints_2 hashModel (generatedRoot.w_60_0 ctx) (generatedRoot.w_139_0 ctx) (generatedRoot.w_143_0 ctx) (generatedRoot.w_148_0 ctx) (generatedRoot.w_153_0 ctx) (generatedRoot.w_78_0 ctx) (generatedRoot.w_99_0 ctx) (generatedRoot.w_100_0 ctx) (generatedRoot.w_100_1 ctx) (generatedRoot.w_100_2 ctx) (generatedRoot.w_100_3 ctx) (generatedRoot.w_100_4 ctx) (generatedRoot.w_100_5 ctx) (generatedRoot.w_100_6 ctx) (generatedRoot.w_100_7 ctx) (generatedRoot.w_109_0 ctx) (generatedRoot.w_110_0 ctx) (generatedRoot.w_110_1 ctx) (generatedRoot.w_110_2 ctx) (generatedRoot.w_110_3 ctx) (generatedRoot.w_110_4 ctx) (generatedRoot.w_110_5 ctx) (generatedRoot.w_110_6 ctx) (generatedRoot.w_110_7 ctx) (generatedRoot.w_119_0 ctx) (generatedRoot.w_120_0 ctx) (generatedRoot.w_120_1 ctx) (generatedRoot.w_120_2 ctx) (generatedRoot.w_120_3 ctx) (generatedRoot.w_120_4 ctx) (generatedRoot.w_120_5 ctx) (generatedRoot.w_120_6 ctx) (generatedRoot.w_120_7 ctx) (generatedRoot.w_129_0 ctx) (generatedRoot.w_130_0 ctx) (generatedRoot.w_130_1 ctx) (generatedRoot.w_130_2 ctx) (generatedRoot.w_130_3 ctx) (generatedRoot.w_130_4 ctx) (generatedRoot.w_130_5 ctx) (generatedRoot.w_130_6 ctx) (generatedRoot.w_130_7 ctx) (generatedRoot.w_140_0 ctx) (generatedRoot.w_144_0 ctx) (generatedRoot.w_149_0 ctx) (generatedRoot.w_154_0 ctx) (generatedRoot.w_158_0 ctx) (generatedRoot.w_159_0 ctx) (generatedRoot.w_161_0 ctx) (generatedRoot.w_168_0 ctx) (generatedRoot.w_169_0 ctx) (generatedRoot.w_170_0 ctx) (generatedRoot.w_170_1 ctx) (generatedRoot.w_170_2 ctx) (generatedRoot.w_170_3 ctx) (generatedRoot.w_170_4 ctx) (generatedRoot.w_170_5 ctx) (generatedRoot.w_170_6 ctx) (generatedRoot.w_170_7 ctx) (generatedRoot.w_179_0 ctx) (generatedRoot.w_180_0 ctx) (generatedRoot.w_180_1 ctx) (generatedRoot.w_180_2 ctx) (generatedRoot.w_180_3 ctx) (generatedRoot.w_180_4 ctx) (generatedRoot.w_180_5 ctx) (generatedRoot.w_180_6 ctx) (generatedRoot.w_180_7 ctx) (generatedRoot.w_189_0 ctx) (generatedRoot.w_190_0 ctx) (generatedRoot.w_190_1 ctx) (generatedRoot.w_190_2 ctx) (generatedRoot.w_190_3 ctx) (generatedRoot.w_190_4 ctx) (generatedRoot.w_190_5 ctx) (generatedRoot.w_190_6 ctx) (generatedRoot.w_190_7 ctx) (generatedRoot.w_199_0 ctx) (generatedRoot.w_200_0 ctx) (generatedRoot.w_200_1 ctx) (generatedRoot.w_200_2 ctx) (generatedRoot.w_200_3 ctx) (generatedRoot.w_200_4 ctx) (generatedRoot.w_200_5 ctx) (generatedRoot.w_200_6 ctx) (generatedRoot.w_200_7 ctx) (generatedRoot.w_209_0 ctx) (generatedRoot.w_210_0 ctx) (generatedRoot.w_210_1 ctx) (generatedRoot.w_210_2 ctx) (generatedRoot.w_210_3 ctx) (generatedRoot.w_210_4 ctx) (generatedRoot.w_210_5 ctx) (generatedRoot.w_210_6 ctx) (generatedRoot.w_210_7 ctx) (generatedRoot.w_219_0 ctx) (generatedRoot.w_220_0 ctx) (generatedRoot.w_220_1 ctx) (generatedRoot.w_220_2 ctx) (generatedRoot.w_220_3 ctx) (generatedRoot.w_220_4 ctx) (generatedRoot.w_220_5 ctx) (generatedRoot.w_220_6 ctx) (generatedRoot.w_220_7 ctx) (generatedRoot.w_228_0 ctx) outputs) :
    generatedRoot.Facts_2 ctx ∧ generatedRoot.constraints_3 hashModel (generatedRoot.w_60_0 ctx) (generatedRoot.w_139_0 ctx) (generatedRoot.w_143_0 ctx) (generatedRoot.w_148_0 ctx) (generatedRoot.w_153_0 ctx) (generatedRoot.w_78_0 ctx) (generatedRoot.w_119_0 ctx) (generatedRoot.w_120_0 ctx) (generatedRoot.w_120_1 ctx) (generatedRoot.w_120_2 ctx) (generatedRoot.w_120_3 ctx) (generatedRoot.w_120_4 ctx) (generatedRoot.w_120_5 ctx) (generatedRoot.w_120_6 ctx) (generatedRoot.w_120_7 ctx) (generatedRoot.w_129_0 ctx) (generatedRoot.w_130_0 ctx) (generatedRoot.w_130_1 ctx) (generatedRoot.w_130_2 ctx) (generatedRoot.w_130_3 ctx) (generatedRoot.w_130_4 ctx) (generatedRoot.w_130_5 ctx) (generatedRoot.w_130_6 ctx) (generatedRoot.w_130_7 ctx) (generatedRoot.w_140_0 ctx) (generatedRoot.w_144_0 ctx) (generatedRoot.w_149_0 ctx) (generatedRoot.w_154_0 ctx) (generatedRoot.w_158_0 ctx) (generatedRoot.w_159_0 ctx) (generatedRoot.w_161_0 ctx) (generatedRoot.w_168_0 ctx) (generatedRoot.w_169_0 ctx) (generatedRoot.w_170_0 ctx) (generatedRoot.w_170_1 ctx) (generatedRoot.w_170_2 ctx) (generatedRoot.w_170_3 ctx) (generatedRoot.w_170_4 ctx) (generatedRoot.w_170_5 ctx) (generatedRoot.w_170_6 ctx) (generatedRoot.w_170_7 ctx) (generatedRoot.w_179_0 ctx) (generatedRoot.w_180_0 ctx) (generatedRoot.w_180_1 ctx) (generatedRoot.w_180_2 ctx) (generatedRoot.w_180_3 ctx) (generatedRoot.w_180_4 ctx) (generatedRoot.w_180_5 ctx) (generatedRoot.w_180_6 ctx) (generatedRoot.w_180_7 ctx) (generatedRoot.w_189_0 ctx) (generatedRoot.w_190_0 ctx) (generatedRoot.w_190_1 ctx) (generatedRoot.w_190_2 ctx) (generatedRoot.w_190_3 ctx) (generatedRoot.w_190_4 ctx) (generatedRoot.w_190_5 ctx) (generatedRoot.w_190_6 ctx) (generatedRoot.w_190_7 ctx) (generatedRoot.w_199_0 ctx) (generatedRoot.w_200_0 ctx) (generatedRoot.w_200_1 ctx) (generatedRoot.w_200_2 ctx) (generatedRoot.w_200_3 ctx) (generatedRoot.w_200_4 ctx) (generatedRoot.w_200_5 ctx) (generatedRoot.w_200_6 ctx) (generatedRoot.w_200_7 ctx) (generatedRoot.w_209_0 ctx) (generatedRoot.w_210_0 ctx) (generatedRoot.w_210_1 ctx) (generatedRoot.w_210_2 ctx) (generatedRoot.w_210_3 ctx) (generatedRoot.w_210_4 ctx) (generatedRoot.w_210_5 ctx) (generatedRoot.w_210_6 ctx) (generatedRoot.w_210_7 ctx) (generatedRoot.w_219_0 ctx) (generatedRoot.w_220_0 ctx) (generatedRoot.w_220_1 ctx) (generatedRoot.w_220_2 ctx) (generatedRoot.w_220_3 ctx) (generatedRoot.w_220_4 ctx) (generatedRoot.w_220_5 ctx) (generatedRoot.w_220_6 ctx) (generatedRoot.w_220_7 ctx) (generatedRoot.w_228_0 ctx) outputs :=
  ⟨⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1⟩, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_3 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Prop where
  t96 : 0 ≤ 4
  t97 : 4 < 5
  t98 : 5 ≤ 6
  h119_0 : MxxRuntime.sliceMatrix (generatedRoot.w_78_0 ctx) 0 1 4 5 (generatedRoot.w_119_0 ctx)
  t100 : (8) = 8
  h120_0 : MxxRuntime.thresholdDecode (2) (8) 0 (generatedRoot.w_119_0 ctx) (generatedRoot.w_120_0 ctx)
  h120_1 : MxxRuntime.thresholdDecode (2) (8) 1 (generatedRoot.w_119_0 ctx) (generatedRoot.w_120_1 ctx)
  h120_2 : MxxRuntime.thresholdDecode (2) (8) 2 (generatedRoot.w_119_0 ctx) (generatedRoot.w_120_2 ctx)
  h120_3 : MxxRuntime.thresholdDecode (2) (8) 3 (generatedRoot.w_119_0 ctx) (generatedRoot.w_120_3 ctx)
  h120_4 : MxxRuntime.thresholdDecode (2) (8) 4 (generatedRoot.w_119_0 ctx) (generatedRoot.w_120_4 ctx)
  h120_5 : MxxRuntime.thresholdDecode (2) (8) 5 (generatedRoot.w_119_0 ctx) (generatedRoot.w_120_5 ctx)
  h120_6 : MxxRuntime.thresholdDecode (2) (8) 6 (generatedRoot.w_119_0 ctx) (generatedRoot.w_120_6 ctx)
  h120_7 : MxxRuntime.thresholdDecode (2) (8) 7 (generatedRoot.w_119_0 ctx) (generatedRoot.w_120_7 ctx)
  t109 : 0 ≤ 0
  t110 : 0 < 1
  t111 : 1 ≤ 1
  t112 : 0 ≤ 5
  t113 : 5 < 6
  t114 : 6 ≤ 6
  h129_0 : MxxRuntime.sliceMatrix (generatedRoot.w_78_0 ctx) 0 1 5 6 (generatedRoot.w_129_0 ctx)
  t116 : (8) = 8
  h130_0 : MxxRuntime.thresholdDecode (2) (8) 0 (generatedRoot.w_129_0 ctx) (generatedRoot.w_130_0 ctx)
  h130_1 : MxxRuntime.thresholdDecode (2) (8) 1 (generatedRoot.w_129_0 ctx) (generatedRoot.w_130_1 ctx)
  h130_2 : MxxRuntime.thresholdDecode (2) (8) 2 (generatedRoot.w_129_0 ctx) (generatedRoot.w_130_2 ctx)
  h130_3 : MxxRuntime.thresholdDecode (2) (8) 3 (generatedRoot.w_129_0 ctx) (generatedRoot.w_130_3 ctx)
  h130_4 : MxxRuntime.thresholdDecode (2) (8) 4 (generatedRoot.w_129_0 ctx) (generatedRoot.w_130_4 ctx)
  h130_5 : MxxRuntime.thresholdDecode (2) (8) 5 (generatedRoot.w_129_0 ctx) (generatedRoot.w_130_5 ctx)
  h130_6 : MxxRuntime.thresholdDecode (2) (8) 6 (generatedRoot.w_129_0 ctx) (generatedRoot.w_130_6 ctx)
  h130_7 : MxxRuntime.thresholdDecode (2) (8) 7 (generatedRoot.w_129_0 ctx) (generatedRoot.w_130_7 ctx)
  h140_0 : MxxRuntime.centeredRebaseRuns (generatedRoot.w_139_0 ctx) (generatedRoot.w_140_0 ctx)
  h144_0 : MxxRuntime.centeredRebaseRuns (generatedRoot.w_143_0 ctx) (generatedRoot.w_144_0 ctx)
  h149_0 : MxxRuntime.centeredRebaseRuns (generatedRoot.w_148_0 ctx) (generatedRoot.w_149_0 ctx)

theorem generatedRoot.facts_3 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {ctx : generatedRoot.Ctx backend hashModel params} {outputs : Int}
    (h : generatedRoot.constraints_3 hashModel (generatedRoot.w_60_0 ctx) (generatedRoot.w_139_0 ctx) (generatedRoot.w_143_0 ctx) (generatedRoot.w_148_0 ctx) (generatedRoot.w_153_0 ctx) (generatedRoot.w_78_0 ctx) (generatedRoot.w_119_0 ctx) (generatedRoot.w_120_0 ctx) (generatedRoot.w_120_1 ctx) (generatedRoot.w_120_2 ctx) (generatedRoot.w_120_3 ctx) (generatedRoot.w_120_4 ctx) (generatedRoot.w_120_5 ctx) (generatedRoot.w_120_6 ctx) (generatedRoot.w_120_7 ctx) (generatedRoot.w_129_0 ctx) (generatedRoot.w_130_0 ctx) (generatedRoot.w_130_1 ctx) (generatedRoot.w_130_2 ctx) (generatedRoot.w_130_3 ctx) (generatedRoot.w_130_4 ctx) (generatedRoot.w_130_5 ctx) (generatedRoot.w_130_6 ctx) (generatedRoot.w_130_7 ctx) (generatedRoot.w_140_0 ctx) (generatedRoot.w_144_0 ctx) (generatedRoot.w_149_0 ctx) (generatedRoot.w_154_0 ctx) (generatedRoot.w_158_0 ctx) (generatedRoot.w_159_0 ctx) (generatedRoot.w_161_0 ctx) (generatedRoot.w_168_0 ctx) (generatedRoot.w_169_0 ctx) (generatedRoot.w_170_0 ctx) (generatedRoot.w_170_1 ctx) (generatedRoot.w_170_2 ctx) (generatedRoot.w_170_3 ctx) (generatedRoot.w_170_4 ctx) (generatedRoot.w_170_5 ctx) (generatedRoot.w_170_6 ctx) (generatedRoot.w_170_7 ctx) (generatedRoot.w_179_0 ctx) (generatedRoot.w_180_0 ctx) (generatedRoot.w_180_1 ctx) (generatedRoot.w_180_2 ctx) (generatedRoot.w_180_3 ctx) (generatedRoot.w_180_4 ctx) (generatedRoot.w_180_5 ctx) (generatedRoot.w_180_6 ctx) (generatedRoot.w_180_7 ctx) (generatedRoot.w_189_0 ctx) (generatedRoot.w_190_0 ctx) (generatedRoot.w_190_1 ctx) (generatedRoot.w_190_2 ctx) (generatedRoot.w_190_3 ctx) (generatedRoot.w_190_4 ctx) (generatedRoot.w_190_5 ctx) (generatedRoot.w_190_6 ctx) (generatedRoot.w_190_7 ctx) (generatedRoot.w_199_0 ctx) (generatedRoot.w_200_0 ctx) (generatedRoot.w_200_1 ctx) (generatedRoot.w_200_2 ctx) (generatedRoot.w_200_3 ctx) (generatedRoot.w_200_4 ctx) (generatedRoot.w_200_5 ctx) (generatedRoot.w_200_6 ctx) (generatedRoot.w_200_7 ctx) (generatedRoot.w_209_0 ctx) (generatedRoot.w_210_0 ctx) (generatedRoot.w_210_1 ctx) (generatedRoot.w_210_2 ctx) (generatedRoot.w_210_3 ctx) (generatedRoot.w_210_4 ctx) (generatedRoot.w_210_5 ctx) (generatedRoot.w_210_6 ctx) (generatedRoot.w_210_7 ctx) (generatedRoot.w_219_0 ctx) (generatedRoot.w_220_0 ctx) (generatedRoot.w_220_1 ctx) (generatedRoot.w_220_2 ctx) (generatedRoot.w_220_3 ctx) (generatedRoot.w_220_4 ctx) (generatedRoot.w_220_5 ctx) (generatedRoot.w_220_6 ctx) (generatedRoot.w_220_7 ctx) (generatedRoot.w_228_0 ctx) outputs) :
    generatedRoot.Facts_3 ctx ∧ generatedRoot.constraints_4 hashModel (generatedRoot.w_60_0 ctx) (generatedRoot.w_153_0 ctx) (generatedRoot.w_154_0 ctx) (generatedRoot.w_158_0 ctx) (generatedRoot.w_159_0 ctx) (generatedRoot.w_161_0 ctx) (generatedRoot.w_168_0 ctx) (generatedRoot.w_169_0 ctx) (generatedRoot.w_170_0 ctx) (generatedRoot.w_170_1 ctx) (generatedRoot.w_170_2 ctx) (generatedRoot.w_170_3 ctx) (generatedRoot.w_170_4 ctx) (generatedRoot.w_170_5 ctx) (generatedRoot.w_170_6 ctx) (generatedRoot.w_170_7 ctx) (generatedRoot.w_179_0 ctx) (generatedRoot.w_180_0 ctx) (generatedRoot.w_180_1 ctx) (generatedRoot.w_180_2 ctx) (generatedRoot.w_180_3 ctx) (generatedRoot.w_180_4 ctx) (generatedRoot.w_180_5 ctx) (generatedRoot.w_180_6 ctx) (generatedRoot.w_180_7 ctx) (generatedRoot.w_189_0 ctx) (generatedRoot.w_190_0 ctx) (generatedRoot.w_190_1 ctx) (generatedRoot.w_190_2 ctx) (generatedRoot.w_190_3 ctx) (generatedRoot.w_190_4 ctx) (generatedRoot.w_190_5 ctx) (generatedRoot.w_190_6 ctx) (generatedRoot.w_190_7 ctx) (generatedRoot.w_199_0 ctx) (generatedRoot.w_200_0 ctx) (generatedRoot.w_200_1 ctx) (generatedRoot.w_200_2 ctx) (generatedRoot.w_200_3 ctx) (generatedRoot.w_200_4 ctx) (generatedRoot.w_200_5 ctx) (generatedRoot.w_200_6 ctx) (generatedRoot.w_200_7 ctx) (generatedRoot.w_209_0 ctx) (generatedRoot.w_210_0 ctx) (generatedRoot.w_210_1 ctx) (generatedRoot.w_210_2 ctx) (generatedRoot.w_210_3 ctx) (generatedRoot.w_210_4 ctx) (generatedRoot.w_210_5 ctx) (generatedRoot.w_210_6 ctx) (generatedRoot.w_210_7 ctx) (generatedRoot.w_219_0 ctx) (generatedRoot.w_220_0 ctx) (generatedRoot.w_220_1 ctx) (generatedRoot.w_220_2 ctx) (generatedRoot.w_220_3 ctx) (generatedRoot.w_220_4 ctx) (generatedRoot.w_220_5 ctx) (generatedRoot.w_220_6 ctx) (generatedRoot.w_220_7 ctx) (generatedRoot.w_228_0 ctx) outputs :=
  ⟨⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1⟩, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_4 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Prop where
  h154_0 : MxxRuntime.centeredRebaseRuns (generatedRoot.w_153_0 ctx) (generatedRoot.w_154_0 ctx)
  h158_0 : MxxRuntime.hashSample (hashModel) ([114, 101, 102, 114, 101, 115, 104, 47, 118, 49, 47, 65, 116, 111, 112, 45, 114, 101, 102, 114, 101, 115, 104, 45, 114, 111, 117, 110, 100, 45, 116, 114, 105, 112]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((generatedRoot.w_60_0 ctx)) (generatedRoot.w_158_0 ctx)
  t130 : 0 ≤ 0
  t131 : 0 < 1
  t132 : 1 ≤ 2
  t133 : 0 ≤ 0
  t134 : 0 < 6
  t135 : 6 ≤ 6
  h159_0 : MxxRuntime.sliceMatrix (generatedRoot.w_158_0 ctx) 0 1 0 6 (generatedRoot.w_159_0 ctx)
  t137 : 0 ≤ 1
  t138 : 1 < 2
  t139 : 2 ≤ 2
  t140 : 0 ≤ 0
  t141 : 0 < 6
  t142 : 6 ≤ 6
  h161_0 : MxxRuntime.sliceMatrix (generatedRoot.w_158_0 ctx) 1 2 0 6 (generatedRoot.w_161_0 ctx)
  t144 : 0 ≤ 0
  t145 : 0 < 1
  t146 : 1 ≤ 1
  t147 : 0 ≤ 0
  t148 : 0 < 1
  t149 : 1 ≤ 6
  h169_0 : MxxRuntime.sliceMatrix (generatedRoot.w_168_0 ctx) 0 1 0 1 (generatedRoot.w_169_0 ctx)
  t151 : (8) = 8
  h170_0 : MxxRuntime.thresholdDecode (2) (8) 0 (generatedRoot.w_169_0 ctx) (generatedRoot.w_170_0 ctx)
  h170_1 : MxxRuntime.thresholdDecode (2) (8) 1 (generatedRoot.w_169_0 ctx) (generatedRoot.w_170_1 ctx)
  h170_2 : MxxRuntime.thresholdDecode (2) (8) 2 (generatedRoot.w_169_0 ctx) (generatedRoot.w_170_2 ctx)
  h170_3 : MxxRuntime.thresholdDecode (2) (8) 3 (generatedRoot.w_169_0 ctx) (generatedRoot.w_170_3 ctx)
  h170_4 : MxxRuntime.thresholdDecode (2) (8) 4 (generatedRoot.w_169_0 ctx) (generatedRoot.w_170_4 ctx)
  h170_5 : MxxRuntime.thresholdDecode (2) (8) 5 (generatedRoot.w_169_0 ctx) (generatedRoot.w_170_5 ctx)
  h170_6 : MxxRuntime.thresholdDecode (2) (8) 6 (generatedRoot.w_169_0 ctx) (generatedRoot.w_170_6 ctx)
  h170_7 : MxxRuntime.thresholdDecode (2) (8) 7 (generatedRoot.w_169_0 ctx) (generatedRoot.w_170_7 ctx)

theorem generatedRoot.facts_4 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {ctx : generatedRoot.Ctx backend hashModel params} {outputs : Int}
    (h : generatedRoot.constraints_4 hashModel (generatedRoot.w_60_0 ctx) (generatedRoot.w_153_0 ctx) (generatedRoot.w_154_0 ctx) (generatedRoot.w_158_0 ctx) (generatedRoot.w_159_0 ctx) (generatedRoot.w_161_0 ctx) (generatedRoot.w_168_0 ctx) (generatedRoot.w_169_0 ctx) (generatedRoot.w_170_0 ctx) (generatedRoot.w_170_1 ctx) (generatedRoot.w_170_2 ctx) (generatedRoot.w_170_3 ctx) (generatedRoot.w_170_4 ctx) (generatedRoot.w_170_5 ctx) (generatedRoot.w_170_6 ctx) (generatedRoot.w_170_7 ctx) (generatedRoot.w_179_0 ctx) (generatedRoot.w_180_0 ctx) (generatedRoot.w_180_1 ctx) (generatedRoot.w_180_2 ctx) (generatedRoot.w_180_3 ctx) (generatedRoot.w_180_4 ctx) (generatedRoot.w_180_5 ctx) (generatedRoot.w_180_6 ctx) (generatedRoot.w_180_7 ctx) (generatedRoot.w_189_0 ctx) (generatedRoot.w_190_0 ctx) (generatedRoot.w_190_1 ctx) (generatedRoot.w_190_2 ctx) (generatedRoot.w_190_3 ctx) (generatedRoot.w_190_4 ctx) (generatedRoot.w_190_5 ctx) (generatedRoot.w_190_6 ctx) (generatedRoot.w_190_7 ctx) (generatedRoot.w_199_0 ctx) (generatedRoot.w_200_0 ctx) (generatedRoot.w_200_1 ctx) (generatedRoot.w_200_2 ctx) (generatedRoot.w_200_3 ctx) (generatedRoot.w_200_4 ctx) (generatedRoot.w_200_5 ctx) (generatedRoot.w_200_6 ctx) (generatedRoot.w_200_7 ctx) (generatedRoot.w_209_0 ctx) (generatedRoot.w_210_0 ctx) (generatedRoot.w_210_1 ctx) (generatedRoot.w_210_2 ctx) (generatedRoot.w_210_3 ctx) (generatedRoot.w_210_4 ctx) (generatedRoot.w_210_5 ctx) (generatedRoot.w_210_6 ctx) (generatedRoot.w_210_7 ctx) (generatedRoot.w_219_0 ctx) (generatedRoot.w_220_0 ctx) (generatedRoot.w_220_1 ctx) (generatedRoot.w_220_2 ctx) (generatedRoot.w_220_3 ctx) (generatedRoot.w_220_4 ctx) (generatedRoot.w_220_5 ctx) (generatedRoot.w_220_6 ctx) (generatedRoot.w_220_7 ctx) (generatedRoot.w_228_0 ctx) outputs) :
    generatedRoot.Facts_4 ctx ∧ generatedRoot.constraints_5 (generatedRoot.w_168_0 ctx) (generatedRoot.w_179_0 ctx) (generatedRoot.w_180_0 ctx) (generatedRoot.w_180_1 ctx) (generatedRoot.w_180_2 ctx) (generatedRoot.w_180_3 ctx) (generatedRoot.w_180_4 ctx) (generatedRoot.w_180_5 ctx) (generatedRoot.w_180_6 ctx) (generatedRoot.w_180_7 ctx) (generatedRoot.w_189_0 ctx) (generatedRoot.w_190_0 ctx) (generatedRoot.w_190_1 ctx) (generatedRoot.w_190_2 ctx) (generatedRoot.w_190_3 ctx) (generatedRoot.w_190_4 ctx) (generatedRoot.w_190_5 ctx) (generatedRoot.w_190_6 ctx) (generatedRoot.w_190_7 ctx) (generatedRoot.w_199_0 ctx) (generatedRoot.w_200_0 ctx) (generatedRoot.w_200_1 ctx) (generatedRoot.w_200_2 ctx) (generatedRoot.w_200_3 ctx) (generatedRoot.w_200_4 ctx) (generatedRoot.w_200_5 ctx) (generatedRoot.w_200_6 ctx) (generatedRoot.w_200_7 ctx) (generatedRoot.w_209_0 ctx) (generatedRoot.w_210_0 ctx) (generatedRoot.w_210_1 ctx) (generatedRoot.w_210_2 ctx) (generatedRoot.w_210_3 ctx) (generatedRoot.w_210_4 ctx) (generatedRoot.w_210_5 ctx) (generatedRoot.w_210_6 ctx) (generatedRoot.w_210_7 ctx) (generatedRoot.w_219_0 ctx) (generatedRoot.w_220_0 ctx) (generatedRoot.w_220_1 ctx) (generatedRoot.w_220_2 ctx) (generatedRoot.w_220_3 ctx) (generatedRoot.w_220_4 ctx) (generatedRoot.w_220_5 ctx) (generatedRoot.w_220_6 ctx) (generatedRoot.w_220_7 ctx) (generatedRoot.w_228_0 ctx) outputs :=
  ⟨⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1⟩, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_5 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Prop where
  t160 : 0 ≤ 0
  t161 : 0 < 1
  t162 : 1 ≤ 1
  t163 : 0 ≤ 1
  t164 : 1 < 2
  t165 : 2 ≤ 6
  h179_0 : MxxRuntime.sliceMatrix (generatedRoot.w_168_0 ctx) 0 1 1 2 (generatedRoot.w_179_0 ctx)
  t167 : (8) = 8
  h180_0 : MxxRuntime.thresholdDecode (2) (8) 0 (generatedRoot.w_179_0 ctx) (generatedRoot.w_180_0 ctx)
  h180_1 : MxxRuntime.thresholdDecode (2) (8) 1 (generatedRoot.w_179_0 ctx) (generatedRoot.w_180_1 ctx)
  h180_2 : MxxRuntime.thresholdDecode (2) (8) 2 (generatedRoot.w_179_0 ctx) (generatedRoot.w_180_2 ctx)
  h180_3 : MxxRuntime.thresholdDecode (2) (8) 3 (generatedRoot.w_179_0 ctx) (generatedRoot.w_180_3 ctx)
  h180_4 : MxxRuntime.thresholdDecode (2) (8) 4 (generatedRoot.w_179_0 ctx) (generatedRoot.w_180_4 ctx)
  h180_5 : MxxRuntime.thresholdDecode (2) (8) 5 (generatedRoot.w_179_0 ctx) (generatedRoot.w_180_5 ctx)
  h180_6 : MxxRuntime.thresholdDecode (2) (8) 6 (generatedRoot.w_179_0 ctx) (generatedRoot.w_180_6 ctx)
  h180_7 : MxxRuntime.thresholdDecode (2) (8) 7 (generatedRoot.w_179_0 ctx) (generatedRoot.w_180_7 ctx)
  t176 : 0 ≤ 0
  t177 : 0 < 1
  t178 : 1 ≤ 1
  t179 : 0 ≤ 2
  t180 : 2 < 3
  t181 : 3 ≤ 6
  h189_0 : MxxRuntime.sliceMatrix (generatedRoot.w_168_0 ctx) 0 1 2 3 (generatedRoot.w_189_0 ctx)
  t183 : (8) = 8
  h190_0 : MxxRuntime.thresholdDecode (2) (8) 0 (generatedRoot.w_189_0 ctx) (generatedRoot.w_190_0 ctx)
  h190_1 : MxxRuntime.thresholdDecode (2) (8) 1 (generatedRoot.w_189_0 ctx) (generatedRoot.w_190_1 ctx)
  h190_2 : MxxRuntime.thresholdDecode (2) (8) 2 (generatedRoot.w_189_0 ctx) (generatedRoot.w_190_2 ctx)
  h190_3 : MxxRuntime.thresholdDecode (2) (8) 3 (generatedRoot.w_189_0 ctx) (generatedRoot.w_190_3 ctx)
  h190_4 : MxxRuntime.thresholdDecode (2) (8) 4 (generatedRoot.w_189_0 ctx) (generatedRoot.w_190_4 ctx)
  h190_5 : MxxRuntime.thresholdDecode (2) (8) 5 (generatedRoot.w_189_0 ctx) (generatedRoot.w_190_5 ctx)
  h190_6 : MxxRuntime.thresholdDecode (2) (8) 6 (generatedRoot.w_189_0 ctx) (generatedRoot.w_190_6 ctx)
  h190_7 : MxxRuntime.thresholdDecode (2) (8) 7 (generatedRoot.w_189_0 ctx) (generatedRoot.w_190_7 ctx)

theorem generatedRoot.facts_5 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {ctx : generatedRoot.Ctx backend hashModel params} {outputs : Int}
    (h : generatedRoot.constraints_5 (generatedRoot.w_168_0 ctx) (generatedRoot.w_179_0 ctx) (generatedRoot.w_180_0 ctx) (generatedRoot.w_180_1 ctx) (generatedRoot.w_180_2 ctx) (generatedRoot.w_180_3 ctx) (generatedRoot.w_180_4 ctx) (generatedRoot.w_180_5 ctx) (generatedRoot.w_180_6 ctx) (generatedRoot.w_180_7 ctx) (generatedRoot.w_189_0 ctx) (generatedRoot.w_190_0 ctx) (generatedRoot.w_190_1 ctx) (generatedRoot.w_190_2 ctx) (generatedRoot.w_190_3 ctx) (generatedRoot.w_190_4 ctx) (generatedRoot.w_190_5 ctx) (generatedRoot.w_190_6 ctx) (generatedRoot.w_190_7 ctx) (generatedRoot.w_199_0 ctx) (generatedRoot.w_200_0 ctx) (generatedRoot.w_200_1 ctx) (generatedRoot.w_200_2 ctx) (generatedRoot.w_200_3 ctx) (generatedRoot.w_200_4 ctx) (generatedRoot.w_200_5 ctx) (generatedRoot.w_200_6 ctx) (generatedRoot.w_200_7 ctx) (generatedRoot.w_209_0 ctx) (generatedRoot.w_210_0 ctx) (generatedRoot.w_210_1 ctx) (generatedRoot.w_210_2 ctx) (generatedRoot.w_210_3 ctx) (generatedRoot.w_210_4 ctx) (generatedRoot.w_210_5 ctx) (generatedRoot.w_210_6 ctx) (generatedRoot.w_210_7 ctx) (generatedRoot.w_219_0 ctx) (generatedRoot.w_220_0 ctx) (generatedRoot.w_220_1 ctx) (generatedRoot.w_220_2 ctx) (generatedRoot.w_220_3 ctx) (generatedRoot.w_220_4 ctx) (generatedRoot.w_220_5 ctx) (generatedRoot.w_220_6 ctx) (generatedRoot.w_220_7 ctx) (generatedRoot.w_228_0 ctx) outputs) :
    generatedRoot.Facts_5 ctx ∧ generatedRoot.constraints_6 (generatedRoot.w_168_0 ctx) (generatedRoot.w_199_0 ctx) (generatedRoot.w_200_0 ctx) (generatedRoot.w_200_1 ctx) (generatedRoot.w_200_2 ctx) (generatedRoot.w_200_3 ctx) (generatedRoot.w_200_4 ctx) (generatedRoot.w_200_5 ctx) (generatedRoot.w_200_6 ctx) (generatedRoot.w_200_7 ctx) (generatedRoot.w_209_0 ctx) (generatedRoot.w_210_0 ctx) (generatedRoot.w_210_1 ctx) (generatedRoot.w_210_2 ctx) (generatedRoot.w_210_3 ctx) (generatedRoot.w_210_4 ctx) (generatedRoot.w_210_5 ctx) (generatedRoot.w_210_6 ctx) (generatedRoot.w_210_7 ctx) (generatedRoot.w_219_0 ctx) (generatedRoot.w_220_0 ctx) (generatedRoot.w_220_1 ctx) (generatedRoot.w_220_2 ctx) (generatedRoot.w_220_3 ctx) (generatedRoot.w_220_4 ctx) (generatedRoot.w_220_5 ctx) (generatedRoot.w_220_6 ctx) (generatedRoot.w_220_7 ctx) (generatedRoot.w_228_0 ctx) outputs :=
  ⟨⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1⟩, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_6 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Prop where
  t192 : 0 ≤ 0
  t193 : 0 < 1
  t194 : 1 ≤ 1
  t195 : 0 ≤ 3
  t196 : 3 < 4
  t197 : 4 ≤ 6
  h199_0 : MxxRuntime.sliceMatrix (generatedRoot.w_168_0 ctx) 0 1 3 4 (generatedRoot.w_199_0 ctx)
  t199 : (8) = 8
  h200_0 : MxxRuntime.thresholdDecode (2) (8) 0 (generatedRoot.w_199_0 ctx) (generatedRoot.w_200_0 ctx)
  h200_1 : MxxRuntime.thresholdDecode (2) (8) 1 (generatedRoot.w_199_0 ctx) (generatedRoot.w_200_1 ctx)
  h200_2 : MxxRuntime.thresholdDecode (2) (8) 2 (generatedRoot.w_199_0 ctx) (generatedRoot.w_200_2 ctx)
  h200_3 : MxxRuntime.thresholdDecode (2) (8) 3 (generatedRoot.w_199_0 ctx) (generatedRoot.w_200_3 ctx)
  h200_4 : MxxRuntime.thresholdDecode (2) (8) 4 (generatedRoot.w_199_0 ctx) (generatedRoot.w_200_4 ctx)
  h200_5 : MxxRuntime.thresholdDecode (2) (8) 5 (generatedRoot.w_199_0 ctx) (generatedRoot.w_200_5 ctx)
  h200_6 : MxxRuntime.thresholdDecode (2) (8) 6 (generatedRoot.w_199_0 ctx) (generatedRoot.w_200_6 ctx)
  h200_7 : MxxRuntime.thresholdDecode (2) (8) 7 (generatedRoot.w_199_0 ctx) (generatedRoot.w_200_7 ctx)
  t208 : 0 ≤ 0
  t209 : 0 < 1
  t210 : 1 ≤ 1
  t211 : 0 ≤ 4
  t212 : 4 < 5
  t213 : 5 ≤ 6
  h209_0 : MxxRuntime.sliceMatrix (generatedRoot.w_168_0 ctx) 0 1 4 5 (generatedRoot.w_209_0 ctx)
  t215 : (8) = 8
  h210_0 : MxxRuntime.thresholdDecode (2) (8) 0 (generatedRoot.w_209_0 ctx) (generatedRoot.w_210_0 ctx)
  h210_1 : MxxRuntime.thresholdDecode (2) (8) 1 (generatedRoot.w_209_0 ctx) (generatedRoot.w_210_1 ctx)
  h210_2 : MxxRuntime.thresholdDecode (2) (8) 2 (generatedRoot.w_209_0 ctx) (generatedRoot.w_210_2 ctx)
  h210_3 : MxxRuntime.thresholdDecode (2) (8) 3 (generatedRoot.w_209_0 ctx) (generatedRoot.w_210_3 ctx)
  h210_4 : MxxRuntime.thresholdDecode (2) (8) 4 (generatedRoot.w_209_0 ctx) (generatedRoot.w_210_4 ctx)
  h210_5 : MxxRuntime.thresholdDecode (2) (8) 5 (generatedRoot.w_209_0 ctx) (generatedRoot.w_210_5 ctx)
  h210_6 : MxxRuntime.thresholdDecode (2) (8) 6 (generatedRoot.w_209_0 ctx) (generatedRoot.w_210_6 ctx)
  h210_7 : MxxRuntime.thresholdDecode (2) (8) 7 (generatedRoot.w_209_0 ctx) (generatedRoot.w_210_7 ctx)

theorem generatedRoot.facts_6 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {ctx : generatedRoot.Ctx backend hashModel params} {outputs : Int}
    (h : generatedRoot.constraints_6 (generatedRoot.w_168_0 ctx) (generatedRoot.w_199_0 ctx) (generatedRoot.w_200_0 ctx) (generatedRoot.w_200_1 ctx) (generatedRoot.w_200_2 ctx) (generatedRoot.w_200_3 ctx) (generatedRoot.w_200_4 ctx) (generatedRoot.w_200_5 ctx) (generatedRoot.w_200_6 ctx) (generatedRoot.w_200_7 ctx) (generatedRoot.w_209_0 ctx) (generatedRoot.w_210_0 ctx) (generatedRoot.w_210_1 ctx) (generatedRoot.w_210_2 ctx) (generatedRoot.w_210_3 ctx) (generatedRoot.w_210_4 ctx) (generatedRoot.w_210_5 ctx) (generatedRoot.w_210_6 ctx) (generatedRoot.w_210_7 ctx) (generatedRoot.w_219_0 ctx) (generatedRoot.w_220_0 ctx) (generatedRoot.w_220_1 ctx) (generatedRoot.w_220_2 ctx) (generatedRoot.w_220_3 ctx) (generatedRoot.w_220_4 ctx) (generatedRoot.w_220_5 ctx) (generatedRoot.w_220_6 ctx) (generatedRoot.w_220_7 ctx) (generatedRoot.w_228_0 ctx) outputs) :
    generatedRoot.Facts_6 ctx ∧ generatedRoot.constraints_7 (generatedRoot.w_168_0 ctx) (generatedRoot.w_219_0 ctx) (generatedRoot.w_220_0 ctx) (generatedRoot.w_220_1 ctx) (generatedRoot.w_220_2 ctx) (generatedRoot.w_220_3 ctx) (generatedRoot.w_220_4 ctx) (generatedRoot.w_220_5 ctx) (generatedRoot.w_220_6 ctx) (generatedRoot.w_220_7 ctx) (generatedRoot.w_228_0 ctx) outputs :=
  ⟨⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1⟩, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_7 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) (outputs : Int) : Prop where
  t224 : 0 ≤ 0
  t225 : 0 < 1
  t226 : 1 ≤ 1
  t227 : 0 ≤ 5
  t228 : 5 < 6
  t229 : 6 ≤ 6
  h219_0 : MxxRuntime.sliceMatrix (generatedRoot.w_168_0 ctx) 0 1 5 6 (generatedRoot.w_219_0 ctx)
  t231 : (8) = 8
  h220_0 : MxxRuntime.thresholdDecode (2) (8) 0 (generatedRoot.w_219_0 ctx) (generatedRoot.w_220_0 ctx)
  h220_1 : MxxRuntime.thresholdDecode (2) (8) 1 (generatedRoot.w_219_0 ctx) (generatedRoot.w_220_1 ctx)
  h220_2 : MxxRuntime.thresholdDecode (2) (8) 2 (generatedRoot.w_219_0 ctx) (generatedRoot.w_220_2 ctx)
  h220_3 : MxxRuntime.thresholdDecode (2) (8) 3 (generatedRoot.w_219_0 ctx) (generatedRoot.w_220_3 ctx)
  h220_4 : MxxRuntime.thresholdDecode (2) (8) 4 (generatedRoot.w_219_0 ctx) (generatedRoot.w_220_4 ctx)
  h220_5 : MxxRuntime.thresholdDecode (2) (8) 5 (generatedRoot.w_219_0 ctx) (generatedRoot.w_220_5 ctx)
  h220_6 : MxxRuntime.thresholdDecode (2) (8) 6 (generatedRoot.w_219_0 ctx) (generatedRoot.w_220_6 ctx)
  h220_7 : MxxRuntime.thresholdDecode (2) (8) 7 (generatedRoot.w_219_0 ctx) (generatedRoot.w_220_7 ctx)
  hout : outputs = generatedRoot.outs ctx

theorem generatedRoot.facts_7 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {ctx : generatedRoot.Ctx backend hashModel params} {outputs : Int}
    (h : generatedRoot.constraints_7 (generatedRoot.w_168_0 ctx) (generatedRoot.w_219_0 ctx) (generatedRoot.w_220_0 ctx) (generatedRoot.w_220_1 ctx) (generatedRoot.w_220_2 ctx) (generatedRoot.w_220_3 ctx) (generatedRoot.w_220_4 ctx) (generatedRoot.w_220_5 ctx) (generatedRoot.w_220_6 ctx) (generatedRoot.w_220_7 ctx) (generatedRoot.w_228_0 ctx) outputs) :
    generatedRoot.Facts_7 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) (outputs : Int) : Prop where
  c0 : generatedRoot.Facts_0 ctx
  c1 : generatedRoot.Facts_1 ctx
  c2 : generatedRoot.Facts_2 ctx
  c3 : generatedRoot.Facts_3 ctx
  c4 : generatedRoot.Facts_4 ctx
  c5 : generatedRoot.Facts_5 ctx
  c6 : generatedRoot.Facts_6 ctx
  c7 : generatedRoot.Facts_7 ctx outputs

theorem generatedRoot.facts {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) {outputs : Int}
    (h : generatedRoot.body backend hashModel params ctx.inputs outputs ctx.witness) :
    generatedRoot.Facts ctx outputs :=
  have r0 := generatedRoot.facts_0 (ctx := ctx) (outputs := outputs) h
  have r1 := generatedRoot.facts_1 (ctx := ctx) (outputs := outputs) r0.2
  have r2 := generatedRoot.facts_2 (ctx := ctx) (outputs := outputs) r1.2
  have r3 := generatedRoot.facts_3 (ctx := ctx) (outputs := outputs) r2.2
  have r4 := generatedRoot.facts_4 (ctx := ctx) (outputs := outputs) r3.2
  have r5 := generatedRoot.facts_5 (ctx := ctx) (outputs := outputs) r4.2
  have r6 := generatedRoot.facts_6 (ctx := ctx) (outputs := outputs) r5.2
  have r7 := generatedRoot.facts_7 (ctx := ctx) (outputs := outputs) r6.2
  ⟨r0.1, r1.1, r2.1, r3.1, r4.1, r5.1, r6.1, r7⟩

end Stage_decode

end
