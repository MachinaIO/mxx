import W_Stage_initial

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000
set_option linter.unusedVariables false

-- The wire types repeat the ring modulus `X ^ n + 1`; unfolding it while checking a
-- projection against its field type expands it at the ring dimension.
attribute [local irreducible] Mxx.Primitives.negacyclicModulus

noncomputable section

namespace W_Stage_initial

local notation "mxxMat0" => Mxx.Primitives.ExactMatrix modulus_0 8 1 1
local notation "mxxMat1" => Mxx.Primitives.ExactMatrix modulus_0 8 2 54
local notation "mxxMat2" => Mxx.Primitives.ExactMatrix modulus_0 8 1 2
local notation "mxxMat3" => Mxx.Primitives.ExactMatrix modulus_0 8 1 54
local notation "mxxMat4" => Mxx.Primitives.ExactMatrix modulus_0 8 1 25

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) where
  inputs : mxxMat0 × mxxMat1 × ByteArray × (Fin 5 → Int) × mxxMat0 × Int × Unit
  witness : generatedRoot.Witness

def generatedRoot.w_0_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat0 :=
  ctx.inputs.1

def generatedRoot.w_3_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.inputs.2.1

def generatedRoot.w_9_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : ByteArray :=
  ctx.inputs.2.2.1

def generatedRoot.w_242_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 5 → Int :=
  ctx.inputs.2.2.2.1

def generatedRoot.w_423_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat0 :=
  ctx.inputs.2.2.2.2.1

def generatedRoot.w_451_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ctx.inputs.2.2.2.2.2.1

def generatedRoot.w_1_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat0 :=
  (1 : mxxMat0)

def generatedRoot.w_2_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat2 :=
  ctx.witness.w_2_0

def generatedRoot.w_4_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat3 :=
  MxxRuntime.matrixMul (generatedRoot.w_2_0 ctx) (generatedRoot.w_3_0 ctx)

def generatedRoot.w_5_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat3 :=
  ctx.witness.w_5_0

def generatedRoot.w_6_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat3 :=
  ctx.witness.w_6_0

def generatedRoot.w_7_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat3 :=
  MxxRuntime.matrixSub (generatedRoot.w_5_0 ctx) (generatedRoot.w_6_0 ctx)

def generatedRoot.w_8_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat3 :=
  MxxRuntime.matrixAdd (generatedRoot.w_4_0 ctx) (generatedRoot.w_7_0 ctx)

def generatedRoot.w_10_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  11

def generatedRoot.w_11_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_11_0

def generatedRoot.w_12_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_0_0 ctx) (generatedRoot.w_11_0 ctx)

def generatedRoot.w_13_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_13_0

def generatedRoot.w_14_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_1_0 ctx) (generatedRoot.w_13_0 ctx)

def generatedRoot.w_15_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixAdd (generatedRoot.w_12_0 ctx) (generatedRoot.w_14_0 ctx)

def generatedRoot.w_16_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat0 :=
  ctx.witness.w_16_0

def generatedRoot.w_17_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_17_0

def generatedRoot.w_18_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_16_0 ctx) (generatedRoot.w_17_0 ctx)

def generatedRoot.w_19_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixSub (generatedRoot.w_15_0 ctx) (generatedRoot.w_18_0 ctx)

def generatedRoot.w_20_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_20_0

def generatedRoot.w_21_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_21_0

def generatedRoot.w_22_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixSub (generatedRoot.w_20_0 ctx) (generatedRoot.w_21_0 ctx)

def generatedRoot.w_23_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixAdd (generatedRoot.w_19_0 ctx) (generatedRoot.w_22_0 ctx)

def generatedRoot.w_24_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 1 → mxxMat4 :=
  ctx.witness.w_24_0

def generatedRoot.w_25_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_25_0

def generatedRoot.w_26_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_0_0 ctx) (generatedRoot.w_25_0 ctx)

def generatedRoot.w_27_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_27_0

def generatedRoot.w_28_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_1_0 ctx) (generatedRoot.w_27_0 ctx)

def generatedRoot.w_29_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixAdd (generatedRoot.w_26_0 ctx) (generatedRoot.w_28_0 ctx)

def generatedRoot.w_30_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat0 :=
  (1 : mxxMat0)

def generatedRoot.w_31_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_30_0 ctx) (generatedRoot.w_17_0 ctx)

def generatedRoot.w_32_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixSub (generatedRoot.w_29_0 ctx) (generatedRoot.w_31_0 ctx)

def generatedRoot.w_33_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_33_0

def generatedRoot.w_34_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_34_0

def generatedRoot.w_35_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixSub (generatedRoot.w_33_0 ctx) (generatedRoot.w_34_0 ctx)

def generatedRoot.w_36_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixAdd (generatedRoot.w_32_0 ctx) (generatedRoot.w_35_0 ctx)

def generatedRoot.w_37_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 1 → mxxMat4 :=
  ctx.witness.w_37_0

def generatedRoot.w_38_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  13

def generatedRoot.w_39_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_39_0

def generatedRoot.w_40_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_0_0 ctx) (generatedRoot.w_39_0 ctx)

def generatedRoot.w_41_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_41_0

def generatedRoot.w_42_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_1_0 ctx) (generatedRoot.w_41_0 ctx)

def generatedRoot.w_43_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixAdd (generatedRoot.w_40_0 ctx) (generatedRoot.w_42_0 ctx)

def generatedRoot.w_44_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat0 :=
  ctx.witness.w_44_0

def generatedRoot.w_45_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_45_0

def generatedRoot.w_46_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_44_0 ctx) (generatedRoot.w_45_0 ctx)

def generatedRoot.w_47_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixSub (generatedRoot.w_43_0 ctx) (generatedRoot.w_46_0 ctx)

def generatedRoot.w_48_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_48_0

def generatedRoot.w_49_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_49_0

def generatedRoot.w_50_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixSub (generatedRoot.w_48_0 ctx) (generatedRoot.w_49_0 ctx)

def generatedRoot.w_51_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixAdd (generatedRoot.w_47_0 ctx) (generatedRoot.w_50_0 ctx)

def generatedRoot.w_52_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 1 → mxxMat4 :=
  ctx.witness.w_52_0

def generatedRoot.w_53_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_53_0

def generatedRoot.w_54_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_0_0 ctx) (generatedRoot.w_53_0 ctx)

def generatedRoot.w_55_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_55_0

def generatedRoot.w_56_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_1_0 ctx) (generatedRoot.w_55_0 ctx)

def generatedRoot.w_57_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixAdd (generatedRoot.w_54_0 ctx) (generatedRoot.w_56_0 ctx)

def generatedRoot.w_58_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat0 :=
  (1 : mxxMat0)

def generatedRoot.w_59_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_58_0 ctx) (generatedRoot.w_45_0 ctx)

def generatedRoot.w_60_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixSub (generatedRoot.w_57_0 ctx) (generatedRoot.w_59_0 ctx)

def generatedRoot.w_61_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_61_0

def generatedRoot.w_62_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_62_0

def generatedRoot.w_63_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixSub (generatedRoot.w_61_0 ctx) (generatedRoot.w_62_0 ctx)

def generatedRoot.w_64_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixAdd (generatedRoot.w_60_0 ctx) (generatedRoot.w_63_0 ctx)

def generatedRoot.w_65_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 1 → mxxMat4 :=
  ctx.witness.w_65_0

def generatedRoot.w_66_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  15

def generatedRoot.w_67_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_67_0

def generatedRoot.w_68_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_0_0 ctx) (generatedRoot.w_67_0 ctx)

def generatedRoot.w_69_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_69_0

def generatedRoot.w_70_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_1_0 ctx) (generatedRoot.w_69_0 ctx)

def generatedRoot.w_71_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixAdd (generatedRoot.w_68_0 ctx) (generatedRoot.w_70_0 ctx)

def generatedRoot.w_72_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat0 :=
  ctx.witness.w_72_0

def generatedRoot.w_73_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_73_0

def generatedRoot.w_74_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_72_0 ctx) (generatedRoot.w_73_0 ctx)

def generatedRoot.w_75_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixSub (generatedRoot.w_71_0 ctx) (generatedRoot.w_74_0 ctx)

def generatedRoot.w_76_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_76_0

def generatedRoot.w_77_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_77_0

def generatedRoot.w_78_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixSub (generatedRoot.w_76_0 ctx) (generatedRoot.w_77_0 ctx)

def generatedRoot.w_79_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixAdd (generatedRoot.w_75_0 ctx) (generatedRoot.w_78_0 ctx)

def generatedRoot.w_80_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 1 → mxxMat4 :=
  ctx.witness.w_80_0

def generatedRoot.w_81_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_81_0

def generatedRoot.w_82_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_0_0 ctx) (generatedRoot.w_81_0 ctx)

def generatedRoot.w_83_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_83_0

def generatedRoot.w_84_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_1_0 ctx) (generatedRoot.w_83_0 ctx)

def generatedRoot.w_85_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixAdd (generatedRoot.w_82_0 ctx) (generatedRoot.w_84_0 ctx)

def generatedRoot.w_86_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat0 :=
  (1 : mxxMat0)

def generatedRoot.w_87_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_86_0 ctx) (generatedRoot.w_73_0 ctx)

def generatedRoot.w_88_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixSub (generatedRoot.w_85_0 ctx) (generatedRoot.w_87_0 ctx)

def generatedRoot.w_89_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_89_0

def generatedRoot.w_90_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_90_0

def generatedRoot.w_91_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixSub (generatedRoot.w_89_0 ctx) (generatedRoot.w_90_0 ctx)

def generatedRoot.w_92_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixAdd (generatedRoot.w_88_0 ctx) (generatedRoot.w_91_0 ctx)

def generatedRoot.w_93_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 1 → mxxMat4 :=
  ctx.witness.w_93_0

def generatedRoot.w_94_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  3

def generatedRoot.w_95_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_95_0

def generatedRoot.w_96_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_0_0 ctx) (generatedRoot.w_95_0 ctx)

def generatedRoot.w_97_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_97_0

def generatedRoot.w_98_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_1_0 ctx) (generatedRoot.w_97_0 ctx)

def generatedRoot.w_99_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixAdd (generatedRoot.w_96_0 ctx) (generatedRoot.w_98_0 ctx)

def generatedRoot.w_100_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat0 :=
  ctx.witness.w_100_0

def generatedRoot.w_101_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_101_0

def generatedRoot.w_102_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_100_0 ctx) (generatedRoot.w_101_0 ctx)

def generatedRoot.w_103_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixSub (generatedRoot.w_99_0 ctx) (generatedRoot.w_102_0 ctx)

def generatedRoot.w_104_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_104_0

def generatedRoot.w_105_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_105_0

def generatedRoot.w_106_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixSub (generatedRoot.w_104_0 ctx) (generatedRoot.w_105_0 ctx)

def generatedRoot.w_107_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixAdd (generatedRoot.w_103_0 ctx) (generatedRoot.w_106_0 ctx)

def generatedRoot.w_108_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 1 → mxxMat4 :=
  ctx.witness.w_108_0

def generatedRoot.w_109_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_109_0

def generatedRoot.w_110_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_0_0 ctx) (generatedRoot.w_109_0 ctx)

def generatedRoot.w_111_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_111_0

def generatedRoot.w_112_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_1_0 ctx) (generatedRoot.w_111_0 ctx)

def generatedRoot.w_113_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixAdd (generatedRoot.w_110_0 ctx) (generatedRoot.w_112_0 ctx)

def generatedRoot.w_114_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat0 :=
  (1 : mxxMat0)

def generatedRoot.w_115_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_114_0 ctx) (generatedRoot.w_101_0 ctx)

def generatedRoot.w_116_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixSub (generatedRoot.w_113_0 ctx) (generatedRoot.w_115_0 ctx)

def generatedRoot.w_117_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_117_0

def generatedRoot.w_118_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_118_0

def generatedRoot.w_119_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixSub (generatedRoot.w_117_0 ctx) (generatedRoot.w_118_0 ctx)

def generatedRoot.w_120_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixAdd (generatedRoot.w_116_0 ctx) (generatedRoot.w_119_0 ctx)

def generatedRoot.w_121_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 1 → mxxMat4 :=
  ctx.witness.w_121_0

def generatedRoot.w_122_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  5

def generatedRoot.w_123_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_123_0

def generatedRoot.w_124_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_0_0 ctx) (generatedRoot.w_123_0 ctx)

def generatedRoot.w_125_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_125_0

def generatedRoot.w_126_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_1_0 ctx) (generatedRoot.w_125_0 ctx)

def generatedRoot.w_127_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixAdd (generatedRoot.w_124_0 ctx) (generatedRoot.w_126_0 ctx)

def generatedRoot.w_128_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat0 :=
  ctx.witness.w_128_0

def generatedRoot.w_129_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_129_0

def generatedRoot.w_130_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_128_0 ctx) (generatedRoot.w_129_0 ctx)

def generatedRoot.w_131_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixSub (generatedRoot.w_127_0 ctx) (generatedRoot.w_130_0 ctx)

def generatedRoot.w_132_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_132_0

def generatedRoot.w_133_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_133_0

def generatedRoot.w_134_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixSub (generatedRoot.w_132_0 ctx) (generatedRoot.w_133_0 ctx)

def generatedRoot.w_135_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixAdd (generatedRoot.w_131_0 ctx) (generatedRoot.w_134_0 ctx)

def generatedRoot.w_136_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 1 → mxxMat4 :=
  ctx.witness.w_136_0

def generatedRoot.w_137_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_137_0

def generatedRoot.w_138_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_0_0 ctx) (generatedRoot.w_137_0 ctx)

def generatedRoot.w_139_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_139_0

def generatedRoot.w_140_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_1_0 ctx) (generatedRoot.w_139_0 ctx)

def generatedRoot.w_141_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixAdd (generatedRoot.w_138_0 ctx) (generatedRoot.w_140_0 ctx)

def generatedRoot.w_142_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat0 :=
  (1 : mxxMat0)

def generatedRoot.w_143_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_142_0 ctx) (generatedRoot.w_129_0 ctx)

def generatedRoot.w_144_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixSub (generatedRoot.w_141_0 ctx) (generatedRoot.w_143_0 ctx)

def generatedRoot.w_145_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_145_0

def generatedRoot.w_146_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_146_0

def generatedRoot.w_147_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixSub (generatedRoot.w_145_0 ctx) (generatedRoot.w_146_0 ctx)

def generatedRoot.w_148_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixAdd (generatedRoot.w_144_0 ctx) (generatedRoot.w_147_0 ctx)

def generatedRoot.w_149_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 1 → mxxMat4 :=
  ctx.witness.w_149_0

def generatedRoot.w_150_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  7

def generatedRoot.w_151_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_151_0

def generatedRoot.w_152_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_0_0 ctx) (generatedRoot.w_151_0 ctx)

def generatedRoot.w_153_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_153_0

def generatedRoot.w_154_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_1_0 ctx) (generatedRoot.w_153_0 ctx)

def generatedRoot.w_155_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixAdd (generatedRoot.w_152_0 ctx) (generatedRoot.w_154_0 ctx)

def generatedRoot.w_156_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat0 :=
  ctx.witness.w_156_0

def generatedRoot.w_157_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_157_0

def generatedRoot.w_158_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_156_0 ctx) (generatedRoot.w_157_0 ctx)

def generatedRoot.w_159_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixSub (generatedRoot.w_155_0 ctx) (generatedRoot.w_158_0 ctx)

def generatedRoot.w_160_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_160_0

def generatedRoot.w_161_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_161_0

def generatedRoot.w_162_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixSub (generatedRoot.w_160_0 ctx) (generatedRoot.w_161_0 ctx)

def generatedRoot.w_163_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixAdd (generatedRoot.w_159_0 ctx) (generatedRoot.w_162_0 ctx)

def generatedRoot.w_164_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 1 → mxxMat4 :=
  ctx.witness.w_164_0

def generatedRoot.w_165_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_165_0

def generatedRoot.w_166_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_0_0 ctx) (generatedRoot.w_165_0 ctx)

def generatedRoot.w_167_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_167_0

def generatedRoot.w_168_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_1_0 ctx) (generatedRoot.w_167_0 ctx)

def generatedRoot.w_169_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixAdd (generatedRoot.w_166_0 ctx) (generatedRoot.w_168_0 ctx)

def generatedRoot.w_170_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat0 :=
  (1 : mxxMat0)

def generatedRoot.w_171_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_170_0 ctx) (generatedRoot.w_157_0 ctx)

def generatedRoot.w_172_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixSub (generatedRoot.w_169_0 ctx) (generatedRoot.w_171_0 ctx)

def generatedRoot.w_173_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_173_0

def generatedRoot.w_174_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_174_0

def generatedRoot.w_175_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixSub (generatedRoot.w_173_0 ctx) (generatedRoot.w_174_0 ctx)

def generatedRoot.w_176_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixAdd (generatedRoot.w_172_0 ctx) (generatedRoot.w_175_0 ctx)

def generatedRoot.w_177_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 1 → mxxMat4 :=
  ctx.witness.w_177_0

def generatedRoot.w_178_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  9

def generatedRoot.w_179_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_179_0

def generatedRoot.w_180_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_0_0 ctx) (generatedRoot.w_179_0 ctx)

def generatedRoot.w_181_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_181_0

def generatedRoot.w_182_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_1_0 ctx) (generatedRoot.w_181_0 ctx)

def generatedRoot.w_183_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixAdd (generatedRoot.w_180_0 ctx) (generatedRoot.w_182_0 ctx)

def generatedRoot.w_184_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat0 :=
  ctx.witness.w_184_0

def generatedRoot.w_185_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_185_0

def generatedRoot.w_186_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_184_0 ctx) (generatedRoot.w_185_0 ctx)

def generatedRoot.w_187_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixSub (generatedRoot.w_183_0 ctx) (generatedRoot.w_186_0 ctx)

def generatedRoot.w_188_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_188_0

def generatedRoot.w_189_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_189_0

def generatedRoot.w_190_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixSub (generatedRoot.w_188_0 ctx) (generatedRoot.w_189_0 ctx)

def generatedRoot.w_191_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixAdd (generatedRoot.w_187_0 ctx) (generatedRoot.w_190_0 ctx)

def generatedRoot.w_192_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 1 → mxxMat4 :=
  ctx.witness.w_192_0

def generatedRoot.w_193_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_193_0

def generatedRoot.w_194_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_0_0 ctx) (generatedRoot.w_193_0 ctx)

def generatedRoot.w_195_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_195_0

def generatedRoot.w_196_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_1_0 ctx) (generatedRoot.w_195_0 ctx)

def generatedRoot.w_197_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixAdd (generatedRoot.w_194_0 ctx) (generatedRoot.w_196_0 ctx)

def generatedRoot.w_198_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat0 :=
  (1 : mxxMat0)

def generatedRoot.w_199_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_198_0 ctx) (generatedRoot.w_185_0 ctx)

def generatedRoot.w_200_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixSub (generatedRoot.w_197_0 ctx) (generatedRoot.w_199_0 ctx)

def generatedRoot.w_201_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_201_0

def generatedRoot.w_202_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_202_0

def generatedRoot.w_203_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixSub (generatedRoot.w_201_0 ctx) (generatedRoot.w_202_0 ctx)

def generatedRoot.w_204_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixAdd (generatedRoot.w_200_0 ctx) (generatedRoot.w_203_0 ctx)

def generatedRoot.w_205_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 1 → mxxMat4 :=
  ctx.witness.w_205_0

def generatedRoot.w_206_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  1

def generatedRoot.w_207_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_207_0

def generatedRoot.w_208_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_0_0 ctx) (generatedRoot.w_207_0 ctx)

def generatedRoot.w_209_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_209_0

def generatedRoot.w_210_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_1_0 ctx) (generatedRoot.w_209_0 ctx)

def generatedRoot.w_211_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixAdd (generatedRoot.w_208_0 ctx) (generatedRoot.w_210_0 ctx)

def generatedRoot.w_212_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat0 :=
  ctx.witness.w_212_0

def generatedRoot.w_213_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_213_0

def generatedRoot.w_214_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_212_0 ctx) (generatedRoot.w_213_0 ctx)

def generatedRoot.w_215_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixSub (generatedRoot.w_211_0 ctx) (generatedRoot.w_214_0 ctx)

def generatedRoot.w_216_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_216_0

def generatedRoot.w_217_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_217_0

def generatedRoot.w_218_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixSub (generatedRoot.w_216_0 ctx) (generatedRoot.w_217_0 ctx)

def generatedRoot.w_219_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixAdd (generatedRoot.w_215_0 ctx) (generatedRoot.w_218_0 ctx)

def generatedRoot.w_220_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 8 → mxxMat4 :=
  ctx.witness.w_220_0

def generatedRoot.w_221_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_221_0

def generatedRoot.w_222_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_0_0 ctx) (generatedRoot.w_221_0 ctx)

def generatedRoot.w_223_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_223_0

def generatedRoot.w_224_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_1_0 ctx) (generatedRoot.w_223_0 ctx)

def generatedRoot.w_225_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixAdd (generatedRoot.w_222_0 ctx) (generatedRoot.w_224_0 ctx)

def generatedRoot.w_226_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat0 :=
  (1 : mxxMat0)

def generatedRoot.w_227_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_226_0 ctx) (generatedRoot.w_213_0 ctx)

def generatedRoot.w_228_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixSub (generatedRoot.w_225_0 ctx) (generatedRoot.w_227_0 ctx)

def generatedRoot.w_229_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_229_0

def generatedRoot.w_230_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_230_0

def generatedRoot.w_231_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixSub (generatedRoot.w_229_0 ctx) (generatedRoot.w_230_0 ctx)

def generatedRoot.w_232_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixAdd (generatedRoot.w_228_0 ctx) (generatedRoot.w_231_0 ctx)

def generatedRoot.w_233_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 8 → mxxMat4 :=
  ctx.witness.w_233_0

def generatedRoot.w_234_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 4 → mxxMat4 :=
  ctx.witness.w_234_0

def generatedRoot.w_235_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 4 → mxxMat4 :=
  ctx.witness.w_235_0

def generatedRoot.w_236_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_236_0

def generatedRoot.w_237_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_0_0 ctx) (generatedRoot.w_236_0 ctx)

def generatedRoot.w_238_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_238_0

def generatedRoot.w_239_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_1_0 ctx) (generatedRoot.w_238_0 ctx)

def generatedRoot.w_240_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixAdd (generatedRoot.w_237_0 ctx) (generatedRoot.w_239_0 ctx)

def generatedRoot.w_241_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat0 :=
  (1 : mxxMat0)

def generatedRoot.w_243_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ctx.witness.w_243_0

def generatedRoot.w_244_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  1

def generatedRoot.w_245_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ((generatedRoot.w_243_0 ctx) * (generatedRoot.w_244_0 ctx))

def generatedRoot.w_246_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat0 :=
  MxxRuntime.multiplyMonomial (generatedRoot.w_241_0 ctx) (generatedRoot.w_245_0 ctx)

def generatedRoot.w_247_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_246_0 ctx) (generatedRoot.w_0_0 ctx)

def generatedRoot.w_248_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_248_0

def generatedRoot.w_249_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_247_0 ctx) (generatedRoot.w_248_0 ctx)

def generatedRoot.w_250_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixSub (generatedRoot.w_240_0 ctx) (generatedRoot.w_249_0 ctx)

def generatedRoot.w_251_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_251_0

def generatedRoot.w_252_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_252_0

def generatedRoot.w_253_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixSub (generatedRoot.w_251_0 ctx) (generatedRoot.w_252_0 ctx)

def generatedRoot.w_254_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixAdd (generatedRoot.w_250_0 ctx) (generatedRoot.w_253_0 ctx)

def generatedRoot.w_255_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_255_0

def generatedRoot.w_256_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_0_0 ctx) (generatedRoot.w_255_0 ctx)

def generatedRoot.w_257_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_257_0

def generatedRoot.w_258_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_1_0 ctx) (generatedRoot.w_257_0 ctx)

def generatedRoot.w_259_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixAdd (generatedRoot.w_256_0 ctx) (generatedRoot.w_258_0 ctx)

def generatedRoot.w_260_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat0 :=
  (1 : mxxMat0)

def generatedRoot.w_261_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_260_0 ctx) (generatedRoot.w_0_0 ctx)

def generatedRoot.w_262_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_262_0

def generatedRoot.w_263_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_261_0 ctx) (generatedRoot.w_262_0 ctx)

def generatedRoot.w_264_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixSub (generatedRoot.w_259_0 ctx) (generatedRoot.w_263_0 ctx)

def generatedRoot.w_265_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_265_0

def generatedRoot.w_266_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_266_0

def generatedRoot.w_267_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixSub (generatedRoot.w_265_0 ctx) (generatedRoot.w_266_0 ctx)

def generatedRoot.w_268_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixAdd (generatedRoot.w_264_0 ctx) (generatedRoot.w_267_0 ctx)

def generatedRoot.w_269_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 2 → mxxMat4 :=
  ctx.witness.w_269_0

def generatedRoot.w_270_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_270_0

def generatedRoot.w_271_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_0_0 ctx) (generatedRoot.w_270_0 ctx)

def generatedRoot.w_272_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_272_0

def generatedRoot.w_273_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_1_0 ctx) (generatedRoot.w_272_0 ctx)

def generatedRoot.w_274_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixAdd (generatedRoot.w_271_0 ctx) (generatedRoot.w_273_0 ctx)

def generatedRoot.w_275_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_246_0 ctx) (generatedRoot.w_1_0 ctx)

def generatedRoot.w_276_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_275_0 ctx) (generatedRoot.w_248_0 ctx)

def generatedRoot.w_277_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixSub (generatedRoot.w_274_0 ctx) (generatedRoot.w_276_0 ctx)

def generatedRoot.w_278_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_278_0

def generatedRoot.w_279_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_279_0

def generatedRoot.w_280_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixSub (generatedRoot.w_278_0 ctx) (generatedRoot.w_279_0 ctx)

def generatedRoot.w_281_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixAdd (generatedRoot.w_277_0 ctx) (generatedRoot.w_280_0 ctx)

def generatedRoot.w_282_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_282_0

def generatedRoot.w_283_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_0_0 ctx) (generatedRoot.w_282_0 ctx)

def generatedRoot.w_284_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_284_0

def generatedRoot.w_285_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_1_0 ctx) (generatedRoot.w_284_0 ctx)

def generatedRoot.w_286_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixAdd (generatedRoot.w_283_0 ctx) (generatedRoot.w_285_0 ctx)

def generatedRoot.w_287_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_260_0 ctx) (generatedRoot.w_1_0 ctx)

def generatedRoot.w_288_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_287_0 ctx) (generatedRoot.w_262_0 ctx)

def generatedRoot.w_289_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixSub (generatedRoot.w_286_0 ctx) (generatedRoot.w_288_0 ctx)

def generatedRoot.w_290_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_290_0

def generatedRoot.w_291_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_291_0

def generatedRoot.w_292_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixSub (generatedRoot.w_290_0 ctx) (generatedRoot.w_291_0 ctx)

def generatedRoot.w_293_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixAdd (generatedRoot.w_289_0 ctx) (generatedRoot.w_292_0 ctx)

def generatedRoot.w_294_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 2 → mxxMat4 :=
  ctx.witness.w_294_0

def generatedRoot.w_295_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_295_0

def generatedRoot.w_296_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_0_0 ctx) (generatedRoot.w_295_0 ctx)

def generatedRoot.w_297_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_297_0

def generatedRoot.w_298_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_1_0 ctx) (generatedRoot.w_297_0 ctx)

def generatedRoot.w_299_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixAdd (generatedRoot.w_296_0 ctx) (generatedRoot.w_298_0 ctx)

def generatedRoot.w_300_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat0 :=
  (1 : mxxMat0)

def generatedRoot.w_301_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ctx.witness.w_301_0

def generatedRoot.w_302_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  1

def generatedRoot.w_303_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ((generatedRoot.w_301_0 ctx) * (generatedRoot.w_302_0 ctx))

def generatedRoot.w_304_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat0 :=
  MxxRuntime.multiplyMonomial (generatedRoot.w_300_0 ctx) (generatedRoot.w_303_0 ctx)

def generatedRoot.w_305_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_304_0 ctx) (generatedRoot.w_0_0 ctx)

def generatedRoot.w_306_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_306_0

def generatedRoot.w_307_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_305_0 ctx) (generatedRoot.w_306_0 ctx)

def generatedRoot.w_308_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixSub (generatedRoot.w_299_0 ctx) (generatedRoot.w_307_0 ctx)

def generatedRoot.w_309_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_309_0

def generatedRoot.w_310_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_310_0

def generatedRoot.w_311_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixSub (generatedRoot.w_309_0 ctx) (generatedRoot.w_310_0 ctx)

def generatedRoot.w_312_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixAdd (generatedRoot.w_308_0 ctx) (generatedRoot.w_311_0 ctx)

def generatedRoot.w_313_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_313_0

def generatedRoot.w_314_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_0_0 ctx) (generatedRoot.w_313_0 ctx)

def generatedRoot.w_315_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_315_0

def generatedRoot.w_316_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_1_0 ctx) (generatedRoot.w_315_0 ctx)

def generatedRoot.w_317_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixAdd (generatedRoot.w_314_0 ctx) (generatedRoot.w_316_0 ctx)

def generatedRoot.w_318_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat0 :=
  (1 : mxxMat0)

def generatedRoot.w_319_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ctx.witness.w_319_0

def generatedRoot.w_320_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  1

def generatedRoot.w_321_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ((generatedRoot.w_319_0 ctx) * (generatedRoot.w_320_0 ctx))

def generatedRoot.w_322_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat0 :=
  MxxRuntime.multiplyMonomial (generatedRoot.w_318_0 ctx) (generatedRoot.w_321_0 ctx)

def generatedRoot.w_323_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_322_0 ctx) (generatedRoot.w_0_0 ctx)

def generatedRoot.w_324_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_324_0

def generatedRoot.w_325_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_323_0 ctx) (generatedRoot.w_324_0 ctx)

def generatedRoot.w_326_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixSub (generatedRoot.w_317_0 ctx) (generatedRoot.w_325_0 ctx)

def generatedRoot.w_327_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_327_0

def generatedRoot.w_328_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_328_0

def generatedRoot.w_329_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixSub (generatedRoot.w_327_0 ctx) (generatedRoot.w_328_0 ctx)

def generatedRoot.w_330_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixAdd (generatedRoot.w_326_0 ctx) (generatedRoot.w_329_0 ctx)

def generatedRoot.w_331_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_331_0

def generatedRoot.w_332_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_0_0 ctx) (generatedRoot.w_331_0 ctx)

def generatedRoot.w_333_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_333_0

def generatedRoot.w_334_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_1_0 ctx) (generatedRoot.w_333_0 ctx)

def generatedRoot.w_335_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixAdd (generatedRoot.w_332_0 ctx) (generatedRoot.w_334_0 ctx)

def generatedRoot.w_336_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat0 :=
  (1 : mxxMat0)

def generatedRoot.w_337_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ctx.witness.w_337_0

def generatedRoot.w_338_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  1

def generatedRoot.w_339_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ((generatedRoot.w_337_0 ctx) * (generatedRoot.w_338_0 ctx))

def generatedRoot.w_340_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat0 :=
  MxxRuntime.multiplyMonomial (generatedRoot.w_336_0 ctx) (generatedRoot.w_339_0 ctx)

def generatedRoot.w_341_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_340_0 ctx) (generatedRoot.w_0_0 ctx)

def generatedRoot.w_342_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_342_0

def generatedRoot.w_343_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_341_0 ctx) (generatedRoot.w_342_0 ctx)

def generatedRoot.w_344_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixSub (generatedRoot.w_335_0 ctx) (generatedRoot.w_343_0 ctx)

def generatedRoot.w_345_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_345_0

def generatedRoot.w_346_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_346_0

def generatedRoot.w_347_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixSub (generatedRoot.w_345_0 ctx) (generatedRoot.w_346_0 ctx)

def generatedRoot.w_348_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixAdd (generatedRoot.w_344_0 ctx) (generatedRoot.w_347_0 ctx)

def generatedRoot.w_349_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_349_0

def generatedRoot.w_350_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_0_0 ctx) (generatedRoot.w_349_0 ctx)

def generatedRoot.w_351_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_351_0

def generatedRoot.w_352_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_1_0 ctx) (generatedRoot.w_351_0 ctx)

def generatedRoot.w_353_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixAdd (generatedRoot.w_350_0 ctx) (generatedRoot.w_352_0 ctx)

def generatedRoot.w_354_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat0 :=
  (1 : mxxMat0)

def generatedRoot.w_355_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ctx.witness.w_355_0

def generatedRoot.w_356_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  1

def generatedRoot.w_357_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ((generatedRoot.w_355_0 ctx) * (generatedRoot.w_356_0 ctx))

def generatedRoot.w_358_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat0 :=
  MxxRuntime.multiplyMonomial (generatedRoot.w_354_0 ctx) (generatedRoot.w_357_0 ctx)

def generatedRoot.w_359_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_358_0 ctx) (generatedRoot.w_0_0 ctx)

def generatedRoot.w_360_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_360_0

def generatedRoot.w_361_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_359_0 ctx) (generatedRoot.w_360_0 ctx)

def generatedRoot.w_362_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixSub (generatedRoot.w_353_0 ctx) (generatedRoot.w_361_0 ctx)

def generatedRoot.w_363_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_363_0

def generatedRoot.w_364_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_364_0

def generatedRoot.w_365_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixSub (generatedRoot.w_363_0 ctx) (generatedRoot.w_364_0 ctx)

def generatedRoot.w_366_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixAdd (generatedRoot.w_362_0 ctx) (generatedRoot.w_365_0 ctx)

def generatedRoot.w_367_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 4 → mxxMat4 :=
  ctx.witness.w_367_0

def generatedRoot.w_368_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_368_0

def generatedRoot.w_369_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_0_0 ctx) (generatedRoot.w_368_0 ctx)

def generatedRoot.w_370_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_370_0

def generatedRoot.w_371_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_1_0 ctx) (generatedRoot.w_370_0 ctx)

def generatedRoot.w_372_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixAdd (generatedRoot.w_369_0 ctx) (generatedRoot.w_371_0 ctx)

def generatedRoot.w_373_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_304_0 ctx) (generatedRoot.w_1_0 ctx)

def generatedRoot.w_374_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_373_0 ctx) (generatedRoot.w_306_0 ctx)

def generatedRoot.w_375_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixSub (generatedRoot.w_372_0 ctx) (generatedRoot.w_374_0 ctx)

def generatedRoot.w_376_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_376_0

def generatedRoot.w_377_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_377_0

def generatedRoot.w_378_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixSub (generatedRoot.w_376_0 ctx) (generatedRoot.w_377_0 ctx)

def generatedRoot.w_379_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixAdd (generatedRoot.w_375_0 ctx) (generatedRoot.w_378_0 ctx)

def generatedRoot.w_380_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_380_0

def generatedRoot.w_381_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_0_0 ctx) (generatedRoot.w_380_0 ctx)

def generatedRoot.w_382_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_382_0

def generatedRoot.w_383_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_1_0 ctx) (generatedRoot.w_382_0 ctx)

def generatedRoot.w_384_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixAdd (generatedRoot.w_381_0 ctx) (generatedRoot.w_383_0 ctx)

def generatedRoot.w_385_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_322_0 ctx) (generatedRoot.w_1_0 ctx)

def generatedRoot.w_386_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_385_0 ctx) (generatedRoot.w_324_0 ctx)

def generatedRoot.w_387_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixSub (generatedRoot.w_384_0 ctx) (generatedRoot.w_386_0 ctx)

def generatedRoot.w_388_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_388_0

def generatedRoot.w_389_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_389_0

def generatedRoot.w_390_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixSub (generatedRoot.w_388_0 ctx) (generatedRoot.w_389_0 ctx)

def generatedRoot.w_391_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixAdd (generatedRoot.w_387_0 ctx) (generatedRoot.w_390_0 ctx)

def generatedRoot.w_392_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_392_0

def generatedRoot.w_393_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_0_0 ctx) (generatedRoot.w_392_0 ctx)

def generatedRoot.w_394_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_394_0

def generatedRoot.w_395_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_1_0 ctx) (generatedRoot.w_394_0 ctx)

def generatedRoot.w_396_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixAdd (generatedRoot.w_393_0 ctx) (generatedRoot.w_395_0 ctx)

def generatedRoot.w_397_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_340_0 ctx) (generatedRoot.w_1_0 ctx)

def generatedRoot.w_398_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_397_0 ctx) (generatedRoot.w_342_0 ctx)

def generatedRoot.w_399_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixSub (generatedRoot.w_396_0 ctx) (generatedRoot.w_398_0 ctx)

def generatedRoot.w_400_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_400_0

def generatedRoot.w_401_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_401_0

def generatedRoot.w_402_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixSub (generatedRoot.w_400_0 ctx) (generatedRoot.w_401_0 ctx)

def generatedRoot.w_403_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixAdd (generatedRoot.w_399_0 ctx) (generatedRoot.w_402_0 ctx)

def generatedRoot.w_404_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_404_0

def generatedRoot.w_405_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_0_0 ctx) (generatedRoot.w_404_0 ctx)

def generatedRoot.w_406_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_406_0

def generatedRoot.w_407_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_1_0 ctx) (generatedRoot.w_406_0 ctx)

def generatedRoot.w_408_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixAdd (generatedRoot.w_405_0 ctx) (generatedRoot.w_407_0 ctx)

def generatedRoot.w_409_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_358_0 ctx) (generatedRoot.w_1_0 ctx)

def generatedRoot.w_410_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_409_0 ctx) (generatedRoot.w_360_0 ctx)

def generatedRoot.w_411_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixSub (generatedRoot.w_408_0 ctx) (generatedRoot.w_410_0 ctx)

def generatedRoot.w_412_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_412_0

def generatedRoot.w_413_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_413_0

def generatedRoot.w_414_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixSub (generatedRoot.w_412_0 ctx) (generatedRoot.w_413_0 ctx)

def generatedRoot.w_415_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixAdd (generatedRoot.w_411_0 ctx) (generatedRoot.w_414_0 ctx)

def generatedRoot.w_416_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 4 → mxxMat4 :=
  ctx.witness.w_416_0

def generatedRoot.w_417_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_417_0

def generatedRoot.w_418_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_0_0 ctx) (generatedRoot.w_417_0 ctx)

def generatedRoot.w_419_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_419_0

def generatedRoot.w_420_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_1_0 ctx) (generatedRoot.w_419_0 ctx)

def generatedRoot.w_421_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixAdd (generatedRoot.w_418_0 ctx) (generatedRoot.w_420_0 ctx)

def generatedRoot.w_422_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat0 :=
  (1 : mxxMat0)

def generatedRoot.w_424_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_422_0 ctx) (generatedRoot.w_423_0 ctx)

def generatedRoot.w_425_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_425_0

def generatedRoot.w_426_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_424_0 ctx) (generatedRoot.w_425_0 ctx)

def generatedRoot.w_427_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixSub (generatedRoot.w_421_0 ctx) (generatedRoot.w_426_0 ctx)

def generatedRoot.w_428_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_428_0

def generatedRoot.w_429_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_429_0

def generatedRoot.w_430_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixSub (generatedRoot.w_428_0 ctx) (generatedRoot.w_429_0 ctx)

def generatedRoot.w_431_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixAdd (generatedRoot.w_427_0 ctx) (generatedRoot.w_430_0 ctx)

def generatedRoot.w_432_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_432_0

def generatedRoot.w_433_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_0_0 ctx) (generatedRoot.w_432_0 ctx)

def generatedRoot.w_434_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_434_0

def generatedRoot.w_435_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_1_0 ctx) (generatedRoot.w_434_0 ctx)

def generatedRoot.w_436_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixAdd (generatedRoot.w_433_0 ctx) (generatedRoot.w_435_0 ctx)

def generatedRoot.w_437_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat0 :=
  (1 : mxxMat0)

def generatedRoot.w_438_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_422_0 ctx) (generatedRoot.w_437_0 ctx)

def generatedRoot.w_439_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_438_0 ctx) (generatedRoot.w_425_0 ctx)

def generatedRoot.w_440_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixSub (generatedRoot.w_436_0 ctx) (generatedRoot.w_439_0 ctx)

def generatedRoot.w_441_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_441_0

def generatedRoot.w_442_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_442_0

def generatedRoot.w_443_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixSub (generatedRoot.w_441_0 ctx) (generatedRoot.w_442_0 ctx)

def generatedRoot.w_444_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixAdd (generatedRoot.w_440_0 ctx) (generatedRoot.w_443_0 ctx)

def generatedRoot.w_445_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_445_0

def generatedRoot.w_446_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_0_0 ctx) (generatedRoot.w_445_0 ctx)

def generatedRoot.w_447_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_447_0

def generatedRoot.w_448_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_1_0 ctx) (generatedRoot.w_447_0 ctx)

def generatedRoot.w_449_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixAdd (generatedRoot.w_446_0 ctx) (generatedRoot.w_448_0 ctx)

def generatedRoot.w_450_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat0 :=
  (1 : mxxMat0)

def generatedRoot.w_452_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  2

def generatedRoot.w_453_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Int :=
  ((generatedRoot.w_451_0 ctx) * (generatedRoot.w_452_0 ctx))

def generatedRoot.w_454_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat0 :=
  MxxRuntime.multiplyMonomial (generatedRoot.w_450_0 ctx) (generatedRoot.w_453_0 ctx)

def generatedRoot.w_455_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_454_0 ctx) (generatedRoot.w_423_0 ctx)

def generatedRoot.w_456_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_456_0

def generatedRoot.w_457_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_455_0 ctx) (generatedRoot.w_456_0 ctx)

def generatedRoot.w_458_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixSub (generatedRoot.w_449_0 ctx) (generatedRoot.w_457_0 ctx)

def generatedRoot.w_459_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_459_0

def generatedRoot.w_460_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_460_0

def generatedRoot.w_461_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixSub (generatedRoot.w_459_0 ctx) (generatedRoot.w_460_0 ctx)

def generatedRoot.w_462_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixAdd (generatedRoot.w_458_0 ctx) (generatedRoot.w_461_0 ctx)

def generatedRoot.w_463_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_463_0

def generatedRoot.w_464_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_0_0 ctx) (generatedRoot.w_463_0 ctx)

def generatedRoot.w_465_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_465_0

def generatedRoot.w_466_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_1_0 ctx) (generatedRoot.w_465_0 ctx)

def generatedRoot.w_467_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixAdd (generatedRoot.w_464_0 ctx) (generatedRoot.w_466_0 ctx)

def generatedRoot.w_468_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_454_0 ctx) (generatedRoot.w_437_0 ctx)

def generatedRoot.w_469_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_468_0 ctx) (generatedRoot.w_456_0 ctx)

def generatedRoot.w_470_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixSub (generatedRoot.w_467_0 ctx) (generatedRoot.w_469_0 ctx)

def generatedRoot.w_471_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_471_0

def generatedRoot.w_472_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_472_0

def generatedRoot.w_473_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixSub (generatedRoot.w_471_0 ctx) (generatedRoot.w_472_0 ctx)

def generatedRoot.w_474_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  MxxRuntime.matrixAdd (generatedRoot.w_470_0 ctx) (generatedRoot.w_473_0 ctx)

def generatedRoot.outs {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat3 × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 8 → mxxMat4) × (Fin 8 → mxxMat4) × (Fin 4 → mxxMat4) × (Fin 4 → mxxMat4) × (Fin 2 → mxxMat4) × (Fin 2 → mxxMat4) × (Fin 4 → mxxMat4) × (Fin 4 → mxxMat4) × mxxMat4 × mxxMat4 × mxxMat4 × mxxMat4 × Unit :=
  ((generatedRoot.w_8_0 ctx), ((generatedRoot.w_24_0 ctx), ((generatedRoot.w_37_0 ctx), ((generatedRoot.w_52_0 ctx), ((generatedRoot.w_65_0 ctx), ((generatedRoot.w_80_0 ctx), ((generatedRoot.w_93_0 ctx), ((generatedRoot.w_108_0 ctx), ((generatedRoot.w_121_0 ctx), ((generatedRoot.w_136_0 ctx), ((generatedRoot.w_149_0 ctx), ((generatedRoot.w_164_0 ctx), ((generatedRoot.w_177_0 ctx), ((generatedRoot.w_192_0 ctx), ((generatedRoot.w_205_0 ctx), ((generatedRoot.w_220_0 ctx), ((generatedRoot.w_233_0 ctx), ((generatedRoot.w_234_0 ctx), ((generatedRoot.w_235_0 ctx), ((generatedRoot.w_269_0 ctx), ((generatedRoot.w_294_0 ctx), ((generatedRoot.w_367_0 ctx), ((generatedRoot.w_416_0 ctx), ((generatedRoot.w_431_0 ctx), ((generatedRoot.w_444_0 ctx), ((generatedRoot.w_462_0 ctx), ((generatedRoot.w_474_0 ctx), ())))))))))))))))))))))))))))

structure generatedRoot.Facts_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Prop where
  h2_0 : MxxRuntime.concatColumns (generatedRoot.w_0_0 ctx) (generatedRoot.w_1_0 ctx) (generatedRoot.w_2_0 ctx)
  h5_0 : MxxRuntime.uniformIntervalSample (0) (1) (generatedRoot.w_5_0 ctx)
  h6_0 : MxxRuntime.uniformIntervalSample (0) (1) (generatedRoot.w_6_0 ctx)
  h11_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_10_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((generatedRoot.w_9_0 ctx)) (generatedRoot.w_11_0 ctx)
  h13_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_10_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) ((generatedRoot.w_9_0 ctx)) (generatedRoot.w_13_0 ctx)
  h16_0 : MxxRuntime.ringAutomorphismRuns (11) (generatedRoot.w_0_0 ctx) (generatedRoot.w_16_0 ctx)
  h17_0 : MxxRuntime.gadgetMatrixRuns backend 4096 25 (generatedRoot.w_17_0 ctx)
  h20_0 : MxxRuntime.uniformIntervalSample (0) (1) (generatedRoot.w_20_0 ctx)
  h21_0 : MxxRuntime.uniformIntervalSample (0) (1) (generatedRoot.w_21_0 ctx)
  h24_0 : MxxRuntime.familyPack (1) [(generatedRoot.w_23_0 ctx)] (generatedRoot.w_24_0 ctx)
  h25_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_10_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((generatedRoot.w_9_0 ctx)) (generatedRoot.w_25_0 ctx)
  h27_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_10_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) ((generatedRoot.w_9_0 ctx)) (generatedRoot.w_27_0 ctx)
  h33_0 : MxxRuntime.uniformIntervalSample (0) (1) (generatedRoot.w_33_0 ctx)
  h34_0 : MxxRuntime.uniformIntervalSample (0) (1) (generatedRoot.w_34_0 ctx)
  h37_0 : MxxRuntime.familyPack (1) [(generatedRoot.w_36_0 ctx)] (generatedRoot.w_37_0 ctx)
  h39_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_38_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((generatedRoot.w_9_0 ctx)) (generatedRoot.w_39_0 ctx)
  h41_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_38_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) ((generatedRoot.w_9_0 ctx)) (generatedRoot.w_41_0 ctx)
  h44_0 : MxxRuntime.ringAutomorphismRuns (13) (generatedRoot.w_0_0 ctx) (generatedRoot.w_44_0 ctx)
  h45_0 : MxxRuntime.gadgetMatrixRuns backend 4096 25 (generatedRoot.w_45_0 ctx)
  h48_0 : MxxRuntime.uniformIntervalSample (0) (1) (generatedRoot.w_48_0 ctx)
  h49_0 : MxxRuntime.uniformIntervalSample (0) (1) (generatedRoot.w_49_0 ctx)
  h52_0 : MxxRuntime.familyPack (1) [(generatedRoot.w_51_0 ctx)] (generatedRoot.w_52_0 ctx)
  h53_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_38_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((generatedRoot.w_9_0 ctx)) (generatedRoot.w_53_0 ctx)
  h55_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_38_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) ((generatedRoot.w_9_0 ctx)) (generatedRoot.w_55_0 ctx)
  h61_0 : MxxRuntime.uniformIntervalSample (0) (1) (generatedRoot.w_61_0 ctx)
  h62_0 : MxxRuntime.uniformIntervalSample (0) (1) (generatedRoot.w_62_0 ctx)
  h65_0 : MxxRuntime.familyPack (1) [(generatedRoot.w_64_0 ctx)] (generatedRoot.w_65_0 ctx)
  h67_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_66_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((generatedRoot.w_9_0 ctx)) (generatedRoot.w_67_0 ctx)
  h69_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_66_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) ((generatedRoot.w_9_0 ctx)) (generatedRoot.w_69_0 ctx)
  h72_0 : MxxRuntime.ringAutomorphismRuns (15) (generatedRoot.w_0_0 ctx) (generatedRoot.w_72_0 ctx)
  h73_0 : MxxRuntime.gadgetMatrixRuns backend 4096 25 (generatedRoot.w_73_0 ctx)
  h76_0 : MxxRuntime.uniformIntervalSample (0) (1) (generatedRoot.w_76_0 ctx)

theorem generatedRoot.facts_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {ctx : generatedRoot.Ctx backend hashModel params} {outputs : mxxMat3 × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 8 → mxxMat4) × (Fin 8 → mxxMat4) × (Fin 4 → mxxMat4) × (Fin 4 → mxxMat4) × (Fin 2 → mxxMat4) × (Fin 2 → mxxMat4) × (Fin 4 → mxxMat4) × (Fin 4 → mxxMat4) × mxxMat4 × mxxMat4 × mxxMat4 × mxxMat4 × Unit}
    (h : generatedRoot.constraints_0 backend hashModel (generatedRoot.w_0_0 ctx) (generatedRoot.w_9_0 ctx) (generatedRoot.w_1_0 ctx) (generatedRoot.w_2_0 ctx) (generatedRoot.w_5_0 ctx) (generatedRoot.w_6_0 ctx) (generatedRoot.w_10_0 ctx) (generatedRoot.w_11_0 ctx) (generatedRoot.w_13_0 ctx) (generatedRoot.w_16_0 ctx) (generatedRoot.w_17_0 ctx) (generatedRoot.w_20_0 ctx) (generatedRoot.w_21_0 ctx) (generatedRoot.w_23_0 ctx) (generatedRoot.w_24_0 ctx) (generatedRoot.w_25_0 ctx) (generatedRoot.w_27_0 ctx) (generatedRoot.w_33_0 ctx) (generatedRoot.w_34_0 ctx) (generatedRoot.w_36_0 ctx) (generatedRoot.w_37_0 ctx) (generatedRoot.w_38_0 ctx) (generatedRoot.w_39_0 ctx) (generatedRoot.w_41_0 ctx) (generatedRoot.w_44_0 ctx) (generatedRoot.w_45_0 ctx) (generatedRoot.w_48_0 ctx) (generatedRoot.w_49_0 ctx) (generatedRoot.w_51_0 ctx) (generatedRoot.w_52_0 ctx) (generatedRoot.w_53_0 ctx) (generatedRoot.w_55_0 ctx) (generatedRoot.w_61_0 ctx) (generatedRoot.w_62_0 ctx) (generatedRoot.w_64_0 ctx) (generatedRoot.w_65_0 ctx) (generatedRoot.w_66_0 ctx) (generatedRoot.w_67_0 ctx) (generatedRoot.w_69_0 ctx) (generatedRoot.w_72_0 ctx) (generatedRoot.w_73_0 ctx) (generatedRoot.w_76_0 ctx)) :
    generatedRoot.Facts_0 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Prop where
  h77_0 : MxxRuntime.uniformIntervalSample (0) (1) (generatedRoot.w_77_0 ctx)
  h80_0 : MxxRuntime.familyPack (1) [(generatedRoot.w_79_0 ctx)] (generatedRoot.w_80_0 ctx)
  h81_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_66_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((generatedRoot.w_9_0 ctx)) (generatedRoot.w_81_0 ctx)
  h83_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_66_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) ((generatedRoot.w_9_0 ctx)) (generatedRoot.w_83_0 ctx)
  h89_0 : MxxRuntime.uniformIntervalSample (0) (1) (generatedRoot.w_89_0 ctx)
  h90_0 : MxxRuntime.uniformIntervalSample (0) (1) (generatedRoot.w_90_0 ctx)
  h93_0 : MxxRuntime.familyPack (1) [(generatedRoot.w_92_0 ctx)] (generatedRoot.w_93_0 ctx)
  h95_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_94_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((generatedRoot.w_9_0 ctx)) (generatedRoot.w_95_0 ctx)
  h97_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_94_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) ((generatedRoot.w_9_0 ctx)) (generatedRoot.w_97_0 ctx)
  h100_0 : MxxRuntime.ringAutomorphismRuns (3) (generatedRoot.w_0_0 ctx) (generatedRoot.w_100_0 ctx)
  h101_0 : MxxRuntime.gadgetMatrixRuns backend 4096 25 (generatedRoot.w_101_0 ctx)
  h104_0 : MxxRuntime.uniformIntervalSample (0) (1) (generatedRoot.w_104_0 ctx)
  h105_0 : MxxRuntime.uniformIntervalSample (0) (1) (generatedRoot.w_105_0 ctx)
  h108_0 : MxxRuntime.familyPack (1) [(generatedRoot.w_107_0 ctx)] (generatedRoot.w_108_0 ctx)
  h109_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_94_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((generatedRoot.w_9_0 ctx)) (generatedRoot.w_109_0 ctx)
  h111_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_94_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) ((generatedRoot.w_9_0 ctx)) (generatedRoot.w_111_0 ctx)
  h117_0 : MxxRuntime.uniformIntervalSample (0) (1) (generatedRoot.w_117_0 ctx)
  h118_0 : MxxRuntime.uniformIntervalSample (0) (1) (generatedRoot.w_118_0 ctx)
  h121_0 : MxxRuntime.familyPack (1) [(generatedRoot.w_120_0 ctx)] (generatedRoot.w_121_0 ctx)
  h123_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_122_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((generatedRoot.w_9_0 ctx)) (generatedRoot.w_123_0 ctx)
  h125_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_122_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) ((generatedRoot.w_9_0 ctx)) (generatedRoot.w_125_0 ctx)
  h128_0 : MxxRuntime.ringAutomorphismRuns (5) (generatedRoot.w_0_0 ctx) (generatedRoot.w_128_0 ctx)
  h129_0 : MxxRuntime.gadgetMatrixRuns backend 4096 25 (generatedRoot.w_129_0 ctx)
  h132_0 : MxxRuntime.uniformIntervalSample (0) (1) (generatedRoot.w_132_0 ctx)
  h133_0 : MxxRuntime.uniformIntervalSample (0) (1) (generatedRoot.w_133_0 ctx)
  h136_0 : MxxRuntime.familyPack (1) [(generatedRoot.w_135_0 ctx)] (generatedRoot.w_136_0 ctx)
  h137_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_122_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((generatedRoot.w_9_0 ctx)) (generatedRoot.w_137_0 ctx)
  h139_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_122_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) ((generatedRoot.w_9_0 ctx)) (generatedRoot.w_139_0 ctx)
  h145_0 : MxxRuntime.uniformIntervalSample (0) (1) (generatedRoot.w_145_0 ctx)
  h146_0 : MxxRuntime.uniformIntervalSample (0) (1) (generatedRoot.w_146_0 ctx)
  h149_0 : MxxRuntime.familyPack (1) [(generatedRoot.w_148_0 ctx)] (generatedRoot.w_149_0 ctx)
  h151_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_150_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((generatedRoot.w_9_0 ctx)) (generatedRoot.w_151_0 ctx)

theorem generatedRoot.facts_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {ctx : generatedRoot.Ctx backend hashModel params} {outputs : mxxMat3 × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 8 → mxxMat4) × (Fin 8 → mxxMat4) × (Fin 4 → mxxMat4) × (Fin 4 → mxxMat4) × (Fin 2 → mxxMat4) × (Fin 2 → mxxMat4) × (Fin 4 → mxxMat4) × (Fin 4 → mxxMat4) × mxxMat4 × mxxMat4 × mxxMat4 × mxxMat4 × Unit}
    (h : generatedRoot.constraints_1 backend hashModel (generatedRoot.w_0_0 ctx) (generatedRoot.w_9_0 ctx) (generatedRoot.w_66_0 ctx) (generatedRoot.w_77_0 ctx) (generatedRoot.w_79_0 ctx) (generatedRoot.w_80_0 ctx) (generatedRoot.w_81_0 ctx) (generatedRoot.w_83_0 ctx) (generatedRoot.w_89_0 ctx) (generatedRoot.w_90_0 ctx) (generatedRoot.w_92_0 ctx) (generatedRoot.w_93_0 ctx) (generatedRoot.w_94_0 ctx) (generatedRoot.w_95_0 ctx) (generatedRoot.w_97_0 ctx) (generatedRoot.w_100_0 ctx) (generatedRoot.w_101_0 ctx) (generatedRoot.w_104_0 ctx) (generatedRoot.w_105_0 ctx) (generatedRoot.w_107_0 ctx) (generatedRoot.w_108_0 ctx) (generatedRoot.w_109_0 ctx) (generatedRoot.w_111_0 ctx) (generatedRoot.w_117_0 ctx) (generatedRoot.w_118_0 ctx) (generatedRoot.w_120_0 ctx) (generatedRoot.w_121_0 ctx) (generatedRoot.w_122_0 ctx) (generatedRoot.w_123_0 ctx) (generatedRoot.w_125_0 ctx) (generatedRoot.w_128_0 ctx) (generatedRoot.w_129_0 ctx) (generatedRoot.w_132_0 ctx) (generatedRoot.w_133_0 ctx) (generatedRoot.w_135_0 ctx) (generatedRoot.w_136_0 ctx) (generatedRoot.w_137_0 ctx) (generatedRoot.w_139_0 ctx) (generatedRoot.w_145_0 ctx) (generatedRoot.w_146_0 ctx) (generatedRoot.w_148_0 ctx) (generatedRoot.w_149_0 ctx) (generatedRoot.w_150_0 ctx) (generatedRoot.w_151_0 ctx)) :
    generatedRoot.Facts_1 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_2 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Prop where
  h153_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_150_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) ((generatedRoot.w_9_0 ctx)) (generatedRoot.w_153_0 ctx)
  h156_0 : MxxRuntime.ringAutomorphismRuns (7) (generatedRoot.w_0_0 ctx) (generatedRoot.w_156_0 ctx)
  h157_0 : MxxRuntime.gadgetMatrixRuns backend 4096 25 (generatedRoot.w_157_0 ctx)
  h160_0 : MxxRuntime.uniformIntervalSample (0) (1) (generatedRoot.w_160_0 ctx)
  h161_0 : MxxRuntime.uniformIntervalSample (0) (1) (generatedRoot.w_161_0 ctx)
  h164_0 : MxxRuntime.familyPack (1) [(generatedRoot.w_163_0 ctx)] (generatedRoot.w_164_0 ctx)
  h165_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_150_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((generatedRoot.w_9_0 ctx)) (generatedRoot.w_165_0 ctx)
  h167_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_150_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) ((generatedRoot.w_9_0 ctx)) (generatedRoot.w_167_0 ctx)
  h173_0 : MxxRuntime.uniformIntervalSample (0) (1) (generatedRoot.w_173_0 ctx)
  h174_0 : MxxRuntime.uniformIntervalSample (0) (1) (generatedRoot.w_174_0 ctx)
  h177_0 : MxxRuntime.familyPack (1) [(generatedRoot.w_176_0 ctx)] (generatedRoot.w_177_0 ctx)
  h179_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_178_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((generatedRoot.w_9_0 ctx)) (generatedRoot.w_179_0 ctx)
  h181_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_178_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) ((generatedRoot.w_9_0 ctx)) (generatedRoot.w_181_0 ctx)
  h184_0 : MxxRuntime.ringAutomorphismRuns (9) (generatedRoot.w_0_0 ctx) (generatedRoot.w_184_0 ctx)
  h185_0 : MxxRuntime.gadgetMatrixRuns backend 4096 25 (generatedRoot.w_185_0 ctx)
  h188_0 : MxxRuntime.uniformIntervalSample (0) (1) (generatedRoot.w_188_0 ctx)
  h189_0 : MxxRuntime.uniformIntervalSample (0) (1) (generatedRoot.w_189_0 ctx)
  h192_0 : MxxRuntime.familyPack (1) [(generatedRoot.w_191_0 ctx)] (generatedRoot.w_192_0 ctx)
  h193_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_178_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((generatedRoot.w_9_0 ctx)) (generatedRoot.w_193_0 ctx)
  h195_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_178_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) ((generatedRoot.w_9_0 ctx)) (generatedRoot.w_195_0 ctx)
  h201_0 : MxxRuntime.uniformIntervalSample (0) (1) (generatedRoot.w_201_0 ctx)
  h202_0 : MxxRuntime.uniformIntervalSample (0) (1) (generatedRoot.w_202_0 ctx)
  h205_0 : MxxRuntime.familyPack (1) [(generatedRoot.w_204_0 ctx)] (generatedRoot.w_205_0 ctx)
  h207_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_206_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((generatedRoot.w_9_0 ctx)) (generatedRoot.w_207_0 ctx)
  h209_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_206_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) ((generatedRoot.w_9_0 ctx)) (generatedRoot.w_209_0 ctx)
  h212_0 : MxxRuntime.ringAutomorphismRuns (1) (generatedRoot.w_0_0 ctx) (generatedRoot.w_212_0 ctx)
  h213_0 : MxxRuntime.gadgetMatrixRuns backend 4096 25 (generatedRoot.w_213_0 ctx)
  h216_0 : MxxRuntime.uniformIntervalSample (0) (1) (generatedRoot.w_216_0 ctx)
  h217_0 : MxxRuntime.uniformIntervalSample (0) (1) (generatedRoot.w_217_0 ctx)
  h220_0 : MxxRuntime.familyPack (8) [(generatedRoot.w_219_0 ctx), (generatedRoot.w_107_0 ctx), (generatedRoot.w_135_0 ctx), (generatedRoot.w_163_0 ctx), (generatedRoot.w_191_0 ctx), (generatedRoot.w_23_0 ctx), (generatedRoot.w_51_0 ctx), (generatedRoot.w_79_0 ctx)] (generatedRoot.w_220_0 ctx)
  h221_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_206_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((generatedRoot.w_9_0 ctx)) (generatedRoot.w_221_0 ctx)
  h223_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_206_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) ((generatedRoot.w_9_0 ctx)) (generatedRoot.w_223_0 ctx)

theorem generatedRoot.facts_2 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {ctx : generatedRoot.Ctx backend hashModel params} {outputs : mxxMat3 × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 8 → mxxMat4) × (Fin 8 → mxxMat4) × (Fin 4 → mxxMat4) × (Fin 4 → mxxMat4) × (Fin 2 → mxxMat4) × (Fin 2 → mxxMat4) × (Fin 4 → mxxMat4) × (Fin 4 → mxxMat4) × mxxMat4 × mxxMat4 × mxxMat4 × mxxMat4 × Unit}
    (h : generatedRoot.constraints_2 backend hashModel (generatedRoot.w_0_0 ctx) (generatedRoot.w_9_0 ctx) (generatedRoot.w_23_0 ctx) (generatedRoot.w_51_0 ctx) (generatedRoot.w_79_0 ctx) (generatedRoot.w_107_0 ctx) (generatedRoot.w_135_0 ctx) (generatedRoot.w_150_0 ctx) (generatedRoot.w_153_0 ctx) (generatedRoot.w_156_0 ctx) (generatedRoot.w_157_0 ctx) (generatedRoot.w_160_0 ctx) (generatedRoot.w_161_0 ctx) (generatedRoot.w_163_0 ctx) (generatedRoot.w_164_0 ctx) (generatedRoot.w_165_0 ctx) (generatedRoot.w_167_0 ctx) (generatedRoot.w_173_0 ctx) (generatedRoot.w_174_0 ctx) (generatedRoot.w_176_0 ctx) (generatedRoot.w_177_0 ctx) (generatedRoot.w_178_0 ctx) (generatedRoot.w_179_0 ctx) (generatedRoot.w_181_0 ctx) (generatedRoot.w_184_0 ctx) (generatedRoot.w_185_0 ctx) (generatedRoot.w_188_0 ctx) (generatedRoot.w_189_0 ctx) (generatedRoot.w_191_0 ctx) (generatedRoot.w_192_0 ctx) (generatedRoot.w_193_0 ctx) (generatedRoot.w_195_0 ctx) (generatedRoot.w_201_0 ctx) (generatedRoot.w_202_0 ctx) (generatedRoot.w_204_0 ctx) (generatedRoot.w_205_0 ctx) (generatedRoot.w_206_0 ctx) (generatedRoot.w_207_0 ctx) (generatedRoot.w_209_0 ctx) (generatedRoot.w_212_0 ctx) (generatedRoot.w_213_0 ctx) (generatedRoot.w_216_0 ctx) (generatedRoot.w_217_0 ctx) (generatedRoot.w_219_0 ctx) (generatedRoot.w_220_0 ctx) (generatedRoot.w_221_0 ctx) (generatedRoot.w_223_0 ctx)) :
    generatedRoot.Facts_2 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_3 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Prop where
  h229_0 : MxxRuntime.uniformIntervalSample (0) (1) (generatedRoot.w_229_0 ctx)
  h230_0 : MxxRuntime.uniformIntervalSample (0) (1) (generatedRoot.w_230_0 ctx)
  h233_0 : MxxRuntime.familyPack (8) [(generatedRoot.w_232_0 ctx), (generatedRoot.w_120_0 ctx), (generatedRoot.w_148_0 ctx), (generatedRoot.w_176_0 ctx), (generatedRoot.w_204_0 ctx), (generatedRoot.w_36_0 ctx), (generatedRoot.w_64_0 ctx), (generatedRoot.w_92_0 ctx)] (generatedRoot.w_233_0 ctx)
  h234_0 : MxxRuntime.familyPack (4) [(generatedRoot.w_219_0 ctx), (generatedRoot.w_107_0 ctx), (generatedRoot.w_135_0 ctx), (generatedRoot.w_79_0 ctx)] (generatedRoot.w_234_0 ctx)
  h235_0 : MxxRuntime.familyPack (4) [(generatedRoot.w_232_0 ctx), (generatedRoot.w_120_0 ctx), (generatedRoot.w_148_0 ctx), (generatedRoot.w_92_0 ctx)] (generatedRoot.w_235_0 ctx)
  h236_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (0), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((generatedRoot.w_9_0 ctx)) (generatedRoot.w_236_0 ctx)
  h238_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (0), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) ((generatedRoot.w_9_0 ctx)) (generatedRoot.w_238_0 ctx)
  h243_0 : MxxRuntime.familyGetStatic (generatedRoot.w_242_0 ctx) (0) (generatedRoot.w_243_0 ctx)
  h248_0 : MxxRuntime.gadgetMatrixRuns backend 4096 25 (generatedRoot.w_248_0 ctx)
  h251_0 : MxxRuntime.uniformIntervalSample (0) (1) (generatedRoot.w_251_0 ctx)
  h252_0 : MxxRuntime.uniformIntervalSample (0) (1) (generatedRoot.w_252_0 ctx)
  h255_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((generatedRoot.w_9_0 ctx)) (generatedRoot.w_255_0 ctx)
  h257_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) ((generatedRoot.w_9_0 ctx)) (generatedRoot.w_257_0 ctx)
  h262_0 : MxxRuntime.gadgetMatrixRuns backend 4096 25 (generatedRoot.w_262_0 ctx)
  h265_0 : MxxRuntime.uniformIntervalSample (0) (1) (generatedRoot.w_265_0 ctx)
  h266_0 : MxxRuntime.uniformIntervalSample (0) (1) (generatedRoot.w_266_0 ctx)
  h269_0 : MxxRuntime.familyPack (2) [(generatedRoot.w_254_0 ctx), (generatedRoot.w_268_0 ctx)] (generatedRoot.w_269_0 ctx)
  h270_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (0), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((generatedRoot.w_9_0 ctx)) (generatedRoot.w_270_0 ctx)
  h272_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (0), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) ((generatedRoot.w_9_0 ctx)) (generatedRoot.w_272_0 ctx)
  h278_0 : MxxRuntime.uniformIntervalSample (0) (1) (generatedRoot.w_278_0 ctx)
  h279_0 : MxxRuntime.uniformIntervalSample (0) (1) (generatedRoot.w_279_0 ctx)
  h282_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((generatedRoot.w_9_0 ctx)) (generatedRoot.w_282_0 ctx)
  h284_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) ((generatedRoot.w_9_0 ctx)) (generatedRoot.w_284_0 ctx)
  h290_0 : MxxRuntime.uniformIntervalSample (0) (1) (generatedRoot.w_290_0 ctx)
  h291_0 : MxxRuntime.uniformIntervalSample (0) (1) (generatedRoot.w_291_0 ctx)
  h294_0 : MxxRuntime.familyPack (2) [(generatedRoot.w_281_0 ctx), (generatedRoot.w_293_0 ctx)] (generatedRoot.w_294_0 ctx)
  h295_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (0), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((generatedRoot.w_9_0 ctx)) (generatedRoot.w_295_0 ctx)
  h297_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (0), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) ((generatedRoot.w_9_0 ctx)) (generatedRoot.w_297_0 ctx)
  h301_0 : MxxRuntime.familyGetStatic (generatedRoot.w_242_0 ctx) (1) (generatedRoot.w_301_0 ctx)
  h306_0 : MxxRuntime.gadgetMatrixRuns backend 4096 25 (generatedRoot.w_306_0 ctx)
  h309_0 : MxxRuntime.uniformIntervalSample (0) (1) (generatedRoot.w_309_0 ctx)
  h310_0 : MxxRuntime.uniformIntervalSample (0) (1) (generatedRoot.w_310_0 ctx)

theorem generatedRoot.facts_3 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {ctx : generatedRoot.Ctx backend hashModel params} {outputs : mxxMat3 × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 8 → mxxMat4) × (Fin 8 → mxxMat4) × (Fin 4 → mxxMat4) × (Fin 4 → mxxMat4) × (Fin 2 → mxxMat4) × (Fin 2 → mxxMat4) × (Fin 4 → mxxMat4) × (Fin 4 → mxxMat4) × mxxMat4 × mxxMat4 × mxxMat4 × mxxMat4 × Unit}
    (h : generatedRoot.constraints_3 backend hashModel (generatedRoot.w_9_0 ctx) (generatedRoot.w_242_0 ctx) (generatedRoot.w_36_0 ctx) (generatedRoot.w_64_0 ctx) (generatedRoot.w_79_0 ctx) (generatedRoot.w_92_0 ctx) (generatedRoot.w_107_0 ctx) (generatedRoot.w_120_0 ctx) (generatedRoot.w_135_0 ctx) (generatedRoot.w_148_0 ctx) (generatedRoot.w_176_0 ctx) (generatedRoot.w_204_0 ctx) (generatedRoot.w_219_0 ctx) (generatedRoot.w_229_0 ctx) (generatedRoot.w_230_0 ctx) (generatedRoot.w_232_0 ctx) (generatedRoot.w_233_0 ctx) (generatedRoot.w_234_0 ctx) (generatedRoot.w_235_0 ctx) (generatedRoot.w_236_0 ctx) (generatedRoot.w_238_0 ctx) (generatedRoot.w_243_0 ctx) (generatedRoot.w_248_0 ctx) (generatedRoot.w_251_0 ctx) (generatedRoot.w_252_0 ctx) (generatedRoot.w_254_0 ctx) (generatedRoot.w_255_0 ctx) (generatedRoot.w_257_0 ctx) (generatedRoot.w_262_0 ctx) (generatedRoot.w_265_0 ctx) (generatedRoot.w_266_0 ctx) (generatedRoot.w_268_0 ctx) (generatedRoot.w_269_0 ctx) (generatedRoot.w_270_0 ctx) (generatedRoot.w_272_0 ctx) (generatedRoot.w_278_0 ctx) (generatedRoot.w_279_0 ctx) (generatedRoot.w_281_0 ctx) (generatedRoot.w_282_0 ctx) (generatedRoot.w_284_0 ctx) (generatedRoot.w_290_0 ctx) (generatedRoot.w_291_0 ctx) (generatedRoot.w_293_0 ctx) (generatedRoot.w_294_0 ctx) (generatedRoot.w_295_0 ctx) (generatedRoot.w_297_0 ctx) (generatedRoot.w_301_0 ctx) (generatedRoot.w_306_0 ctx) (generatedRoot.w_309_0 ctx) (generatedRoot.w_310_0 ctx)) :
    generatedRoot.Facts_3 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_4 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Prop where
  h313_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((generatedRoot.w_9_0 ctx)) (generatedRoot.w_313_0 ctx)
  h315_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) ((generatedRoot.w_9_0 ctx)) (generatedRoot.w_315_0 ctx)
  h319_0 : MxxRuntime.familyGetStatic (generatedRoot.w_242_0 ctx) (2) (generatedRoot.w_319_0 ctx)
  h324_0 : MxxRuntime.gadgetMatrixRuns backend 4096 25 (generatedRoot.w_324_0 ctx)
  h327_0 : MxxRuntime.uniformIntervalSample (0) (1) (generatedRoot.w_327_0 ctx)
  h328_0 : MxxRuntime.uniformIntervalSample (0) (1) (generatedRoot.w_328_0 ctx)
  h331_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (2), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((generatedRoot.w_9_0 ctx)) (generatedRoot.w_331_0 ctx)
  h333_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (2), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) ((generatedRoot.w_9_0 ctx)) (generatedRoot.w_333_0 ctx)
  h337_0 : MxxRuntime.familyGetStatic (generatedRoot.w_242_0 ctx) (3) (generatedRoot.w_337_0 ctx)
  h342_0 : MxxRuntime.gadgetMatrixRuns backend 4096 25 (generatedRoot.w_342_0 ctx)
  h345_0 : MxxRuntime.uniformIntervalSample (0) (1) (generatedRoot.w_345_0 ctx)
  h346_0 : MxxRuntime.uniformIntervalSample (0) (1) (generatedRoot.w_346_0 ctx)
  h349_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (3), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((generatedRoot.w_9_0 ctx)) (generatedRoot.w_349_0 ctx)
  h351_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (3), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) ((generatedRoot.w_9_0 ctx)) (generatedRoot.w_351_0 ctx)
  h355_0 : MxxRuntime.familyGetStatic (generatedRoot.w_242_0 ctx) (4) (generatedRoot.w_355_0 ctx)
  h360_0 : MxxRuntime.gadgetMatrixRuns backend 4096 25 (generatedRoot.w_360_0 ctx)
  h363_0 : MxxRuntime.uniformIntervalSample (0) (1) (generatedRoot.w_363_0 ctx)
  h364_0 : MxxRuntime.uniformIntervalSample (0) (1) (generatedRoot.w_364_0 ctx)
  h367_0 : MxxRuntime.familyPack (4) [(generatedRoot.w_312_0 ctx), (generatedRoot.w_330_0 ctx), (generatedRoot.w_348_0 ctx), (generatedRoot.w_366_0 ctx)] (generatedRoot.w_367_0 ctx)
  h368_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (0), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((generatedRoot.w_9_0 ctx)) (generatedRoot.w_368_0 ctx)
  h370_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (0), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) ((generatedRoot.w_9_0 ctx)) (generatedRoot.w_370_0 ctx)
  h376_0 : MxxRuntime.uniformIntervalSample (0) (1) (generatedRoot.w_376_0 ctx)
  h377_0 : MxxRuntime.uniformIntervalSample (0) (1) (generatedRoot.w_377_0 ctx)
  h380_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((generatedRoot.w_9_0 ctx)) (generatedRoot.w_380_0 ctx)
  h382_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) ((generatedRoot.w_9_0 ctx)) (generatedRoot.w_382_0 ctx)
  h388_0 : MxxRuntime.uniformIntervalSample (0) (1) (generatedRoot.w_388_0 ctx)
  h389_0 : MxxRuntime.uniformIntervalSample (0) (1) (generatedRoot.w_389_0 ctx)
  h392_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (2), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((generatedRoot.w_9_0 ctx)) (generatedRoot.w_392_0 ctx)
  h394_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (2), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) ((generatedRoot.w_9_0 ctx)) (generatedRoot.w_394_0 ctx)
  h400_0 : MxxRuntime.uniformIntervalSample (0) (1) (generatedRoot.w_400_0 ctx)
  h401_0 : MxxRuntime.uniformIntervalSample (0) (1) (generatedRoot.w_401_0 ctx)
  h404_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (3), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((generatedRoot.w_9_0 ctx)) (generatedRoot.w_404_0 ctx)

theorem generatedRoot.facts_4 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {ctx : generatedRoot.Ctx backend hashModel params} {outputs : mxxMat3 × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 8 → mxxMat4) × (Fin 8 → mxxMat4) × (Fin 4 → mxxMat4) × (Fin 4 → mxxMat4) × (Fin 2 → mxxMat4) × (Fin 2 → mxxMat4) × (Fin 4 → mxxMat4) × (Fin 4 → mxxMat4) × mxxMat4 × mxxMat4 × mxxMat4 × mxxMat4 × Unit}
    (h : generatedRoot.constraints_4 backend hashModel (generatedRoot.w_9_0 ctx) (generatedRoot.w_242_0 ctx) (generatedRoot.w_312_0 ctx) (generatedRoot.w_313_0 ctx) (generatedRoot.w_315_0 ctx) (generatedRoot.w_319_0 ctx) (generatedRoot.w_324_0 ctx) (generatedRoot.w_327_0 ctx) (generatedRoot.w_328_0 ctx) (generatedRoot.w_330_0 ctx) (generatedRoot.w_331_0 ctx) (generatedRoot.w_333_0 ctx) (generatedRoot.w_337_0 ctx) (generatedRoot.w_342_0 ctx) (generatedRoot.w_345_0 ctx) (generatedRoot.w_346_0 ctx) (generatedRoot.w_348_0 ctx) (generatedRoot.w_349_0 ctx) (generatedRoot.w_351_0 ctx) (generatedRoot.w_355_0 ctx) (generatedRoot.w_360_0 ctx) (generatedRoot.w_363_0 ctx) (generatedRoot.w_364_0 ctx) (generatedRoot.w_366_0 ctx) (generatedRoot.w_367_0 ctx) (generatedRoot.w_368_0 ctx) (generatedRoot.w_370_0 ctx) (generatedRoot.w_376_0 ctx) (generatedRoot.w_377_0 ctx) (generatedRoot.w_380_0 ctx) (generatedRoot.w_382_0 ctx) (generatedRoot.w_388_0 ctx) (generatedRoot.w_389_0 ctx) (generatedRoot.w_392_0 ctx) (generatedRoot.w_394_0 ctx) (generatedRoot.w_400_0 ctx) (generatedRoot.w_401_0 ctx) (generatedRoot.w_404_0 ctx)) :
    generatedRoot.Facts_4 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_5 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) (outputs : mxxMat3 × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 8 → mxxMat4) × (Fin 8 → mxxMat4) × (Fin 4 → mxxMat4) × (Fin 4 → mxxMat4) × (Fin 2 → mxxMat4) × (Fin 2 → mxxMat4) × (Fin 4 → mxxMat4) × (Fin 4 → mxxMat4) × mxxMat4 × mxxMat4 × mxxMat4 × mxxMat4 × Unit) : Prop where
  h406_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (3), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) ((generatedRoot.w_9_0 ctx)) (generatedRoot.w_406_0 ctx)
  h412_0 : MxxRuntime.uniformIntervalSample (0) (1) (generatedRoot.w_412_0 ctx)
  h413_0 : MxxRuntime.uniformIntervalSample (0) (1) (generatedRoot.w_413_0 ctx)
  h416_0 : MxxRuntime.familyPack (4) [(generatedRoot.w_379_0 ctx), (generatedRoot.w_391_0 ctx), (generatedRoot.w_403_0 ctx), (generatedRoot.w_415_0 ctx)] (generatedRoot.w_416_0 ctx)
  h417_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((generatedRoot.w_9_0 ctx)) (generatedRoot.w_417_0 ctx)
  h419_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) ((generatedRoot.w_9_0 ctx)) (generatedRoot.w_419_0 ctx)
  h425_0 : MxxRuntime.gadgetMatrixRuns backend 4096 25 (generatedRoot.w_425_0 ctx)
  h428_0 : MxxRuntime.uniformIntervalSample (0) (1) (generatedRoot.w_428_0 ctx)
  h429_0 : MxxRuntime.uniformIntervalSample (0) (1) (generatedRoot.w_429_0 ctx)
  h432_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((generatedRoot.w_9_0 ctx)) (generatedRoot.w_432_0 ctx)
  h434_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) ((generatedRoot.w_9_0 ctx)) (generatedRoot.w_434_0 ctx)
  h441_0 : MxxRuntime.uniformIntervalSample (0) (1) (generatedRoot.w_441_0 ctx)
  h442_0 : MxxRuntime.uniformIntervalSample (0) (1) (generatedRoot.w_442_0 ctx)
  h445_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 115, 116, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((generatedRoot.w_9_0 ctx)) (generatedRoot.w_445_0 ctx)
  h447_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 115, 116, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) ((generatedRoot.w_9_0 ctx)) (generatedRoot.w_447_0 ctx)
  h456_0 : MxxRuntime.gadgetMatrixRuns backend 4096 25 (generatedRoot.w_456_0 ctx)
  h459_0 : MxxRuntime.uniformIntervalSample (0) (1) (generatedRoot.w_459_0 ctx)
  h460_0 : MxxRuntime.uniformIntervalSample (0) (1) (generatedRoot.w_460_0 ctx)
  h463_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 115, 116, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((generatedRoot.w_9_0 ctx)) (generatedRoot.w_463_0 ctx)
  h465_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 115, 116, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) ((generatedRoot.w_9_0 ctx)) (generatedRoot.w_465_0 ctx)
  h471_0 : MxxRuntime.uniformIntervalSample (0) (1) (generatedRoot.w_471_0 ctx)
  h472_0 : MxxRuntime.uniformIntervalSample (0) (1) (generatedRoot.w_472_0 ctx)
  hout : outputs = generatedRoot.outs ctx

theorem generatedRoot.facts_5 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {ctx : generatedRoot.Ctx backend hashModel params} {outputs : mxxMat3 × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 8 → mxxMat4) × (Fin 8 → mxxMat4) × (Fin 4 → mxxMat4) × (Fin 4 → mxxMat4) × (Fin 2 → mxxMat4) × (Fin 2 → mxxMat4) × (Fin 4 → mxxMat4) × (Fin 4 → mxxMat4) × mxxMat4 × mxxMat4 × mxxMat4 × mxxMat4 × Unit}
    (h : generatedRoot.constraints_5 backend hashModel (generatedRoot.w_9_0 ctx) (generatedRoot.w_8_0 ctx) (generatedRoot.w_24_0 ctx) (generatedRoot.w_37_0 ctx) (generatedRoot.w_52_0 ctx) (generatedRoot.w_65_0 ctx) (generatedRoot.w_80_0 ctx) (generatedRoot.w_93_0 ctx) (generatedRoot.w_108_0 ctx) (generatedRoot.w_121_0 ctx) (generatedRoot.w_136_0 ctx) (generatedRoot.w_149_0 ctx) (generatedRoot.w_164_0 ctx) (generatedRoot.w_177_0 ctx) (generatedRoot.w_192_0 ctx) (generatedRoot.w_205_0 ctx) (generatedRoot.w_220_0 ctx) (generatedRoot.w_233_0 ctx) (generatedRoot.w_234_0 ctx) (generatedRoot.w_235_0 ctx) (generatedRoot.w_269_0 ctx) (generatedRoot.w_294_0 ctx) (generatedRoot.w_367_0 ctx) (generatedRoot.w_379_0 ctx) (generatedRoot.w_391_0 ctx) (generatedRoot.w_403_0 ctx) (generatedRoot.w_406_0 ctx) (generatedRoot.w_412_0 ctx) (generatedRoot.w_413_0 ctx) (generatedRoot.w_415_0 ctx) (generatedRoot.w_416_0 ctx) (generatedRoot.w_417_0 ctx) (generatedRoot.w_419_0 ctx) (generatedRoot.w_425_0 ctx) (generatedRoot.w_428_0 ctx) (generatedRoot.w_429_0 ctx) (generatedRoot.w_431_0 ctx) (generatedRoot.w_432_0 ctx) (generatedRoot.w_434_0 ctx) (generatedRoot.w_441_0 ctx) (generatedRoot.w_442_0 ctx) (generatedRoot.w_444_0 ctx) (generatedRoot.w_445_0 ctx) (generatedRoot.w_447_0 ctx) (generatedRoot.w_456_0 ctx) (generatedRoot.w_459_0 ctx) (generatedRoot.w_460_0 ctx) (generatedRoot.w_462_0 ctx) (generatedRoot.w_463_0 ctx) (generatedRoot.w_465_0 ctx) (generatedRoot.w_471_0 ctx) (generatedRoot.w_472_0 ctx) (generatedRoot.w_474_0 ctx) outputs) :
    generatedRoot.Facts_5 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) (outputs : mxxMat3 × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 8 → mxxMat4) × (Fin 8 → mxxMat4) × (Fin 4 → mxxMat4) × (Fin 4 → mxxMat4) × (Fin 2 → mxxMat4) × (Fin 2 → mxxMat4) × (Fin 4 → mxxMat4) × (Fin 4 → mxxMat4) × mxxMat4 × mxxMat4 × mxxMat4 × mxxMat4 × Unit) : Prop where
  c0 : generatedRoot.Facts_0 ctx
  c1 : generatedRoot.Facts_1 ctx
  c2 : generatedRoot.Facts_2 ctx
  c3 : generatedRoot.Facts_3 ctx
  c4 : generatedRoot.Facts_4 ctx
  c5 : generatedRoot.Facts_5 ctx outputs

theorem generatedRoot.facts {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) {outputs : mxxMat3 × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 1 → mxxMat4) × (Fin 8 → mxxMat4) × (Fin 8 → mxxMat4) × (Fin 4 → mxxMat4) × (Fin 4 → mxxMat4) × (Fin 2 → mxxMat4) × (Fin 2 → mxxMat4) × (Fin 4 → mxxMat4) × (Fin 4 → mxxMat4) × mxxMat4 × mxxMat4 × mxxMat4 × mxxMat4 × Unit}
    (h : generatedRoot.body backend hashModel params ctx.inputs outputs ctx.witness) :
    generatedRoot.Facts ctx outputs :=
  have t0 := h
  have r0 := generatedRoot.facts_0 (ctx := ctx) (outputs := outputs) t0.1
  have t1 := t0.2
  have r1 := generatedRoot.facts_1 (ctx := ctx) (outputs := outputs) t1.1
  have t2 := t1.2
  have r2 := generatedRoot.facts_2 (ctx := ctx) (outputs := outputs) t2.1
  have t3 := t2.2
  have r3 := generatedRoot.facts_3 (ctx := ctx) (outputs := outputs) t3.1
  have t4 := t3.2
  have r4 := generatedRoot.facts_4 (ctx := ctx) (outputs := outputs) t4.1
  have t5 := t4.2
  have r5 := generatedRoot.facts_5 (ctx := ctx) (outputs := outputs) t5
  ⟨r0, r1, r2, r3, r4, r5⟩

end W_Stage_initial

end
