import Stage_import_1

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000
set_option linter.unusedVariables false

-- The wire types repeat the ring modulus `X ^ n + 1`; unfolding it while checking a
-- projection against its field type expands it at the ring dimension.
attribute [local irreducible] Mxx.Primitives.negacyclicModulus

noncomputable section

namespace Stage_import_1

local notation "mxxMat0" => Mxx.Primitives.ExactMatrix modulus_0 65536 1 270
local notation "mxxMat1" => Mxx.Primitives.ExactMatrix modulus_0 65536 1 1

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Ctx (params : Params) where
  inputs : (generatedRoot.InputType)
  witness : generatedRoot.Witness

section
variable {params : Params}
local notation "generatedRootCtx" => (generatedRoot.Ctx params)

def generatedRoot.w_0_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.inputs.1

def generatedRoot.w_1_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.inputs.2.1

def generatedRoot.w_2_0  (ctx : generatedRootCtx) : Fin 1025 → mxxMat0 :=
  ctx.inputs.2.2.1

def generatedRoot.w_3_0  (ctx : generatedRootCtx) : Fin 1025 → mxxMat0 :=
  ctx.inputs.2.2.2.1

def generatedRoot.w_4_0  (ctx : generatedRootCtx) : Fin 1025 → Int :=
  ctx.inputs.2.2.2.2.1

def generatedRoot.w_5_0  (ctx : generatedRootCtx) : Fin 1025 → Int :=
  ctx.inputs.2.2.2.2.2.1

def generatedRoot.w_7_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.1

def generatedRoot.w_8_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.1

def generatedRoot.w_9_0  (ctx : generatedRootCtx) : Fin 4 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.1

def generatedRoot.w_10_0  (ctx : generatedRootCtx) : Fin 4 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_12_0  (ctx : generatedRootCtx) : Fin 128 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_13_0  (ctx : generatedRootCtx) : Fin 128 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_6_0  (ctx : generatedRootCtx) : Fin 1025 → mxxMat0 :=
  ctx.witness.w_6_0

def generatedRoot.w_6_1  (ctx : generatedRootCtx) : Fin 1025 → mxxMat0 :=
  ctx.witness.w_6_1

def generatedRoot.w_6_2  (ctx : generatedRootCtx) : Fin 1025 → mxxMat1 :=
  ctx.witness.w_6_2

def generatedRoot.w_11_0  (ctx : generatedRootCtx) : Fin 4 → mxxMat0 :=
  (generatedRoot.w_9_0 ctx)

def generatedRoot.w_11_1  (ctx : generatedRootCtx) : Fin 4 → mxxMat0 :=
  (generatedRoot.w_10_0 ctx)

def generatedRoot.w_14_0  (ctx : generatedRootCtx) : Fin 128 → mxxMat0 :=
  (generatedRoot.w_12_0 ctx)

def generatedRoot.w_14_1  (ctx : generatedRootCtx) : Fin 128 → mxxMat0 :=
  (generatedRoot.w_13_0 ctx)

def generatedRoot.w_15_0  (ctx : generatedRootCtx) : Fin 4 → mxxMat0 :=
  (generatedRoot.w_9_0 ctx)

def generatedRoot.w_15_1  (ctx : generatedRootCtx) : Fin 4 → mxxMat0 :=
  (generatedRoot.w_10_0 ctx)

def generatedRoot.w_16_0  (ctx : generatedRootCtx) : Fin 128 → mxxMat0 :=
  (generatedRoot.w_12_0 ctx)

def generatedRoot.w_16_1  (ctx : generatedRootCtx) : Fin 128 → mxxMat0 :=
  (generatedRoot.w_13_0 ctx)

def generatedRoot.w_17_0  (ctx : generatedRootCtx) : Fin 4 → mxxMat0 :=
  (generatedRoot.w_9_0 ctx)

def generatedRoot.w_17_1  (ctx : generatedRootCtx) : Fin 4 → mxxMat0 :=
  (generatedRoot.w_10_0 ctx)

def generatedRoot.w_18_0  (ctx : generatedRootCtx) : Fin 128 → mxxMat0 :=
  (generatedRoot.w_12_0 ctx)

def generatedRoot.w_18_1  (ctx : generatedRootCtx) : Fin 128 → mxxMat0 :=
  (generatedRoot.w_13_0 ctx)

def generatedRoot.w_19_0  (ctx : generatedRootCtx) : Fin 4 → mxxMat0 :=
  (generatedRoot.w_9_0 ctx)

def generatedRoot.w_19_1  (ctx : generatedRootCtx) : Fin 4 → mxxMat0 :=
  (generatedRoot.w_10_0 ctx)

def generatedRoot.w_20_0  (ctx : generatedRootCtx) : Fin 128 → mxxMat0 :=
  (generatedRoot.w_12_0 ctx)

def generatedRoot.w_20_1  (ctx : generatedRootCtx) : Fin 128 → mxxMat0 :=
  (generatedRoot.w_13_0 ctx)

def generatedRoot.w_21_0  (ctx : generatedRootCtx) : Fin 4 → mxxMat0 :=
  (generatedRoot.w_9_0 ctx)

def generatedRoot.w_21_1  (ctx : generatedRootCtx) : Fin 4 → mxxMat0 :=
  (generatedRoot.w_10_0 ctx)

def generatedRoot.w_22_0  (ctx : generatedRootCtx) : Fin 128 → mxxMat0 :=
  (generatedRoot.w_12_0 ctx)

def generatedRoot.w_22_1  (ctx : generatedRootCtx) : Fin 128 → mxxMat0 :=
  (generatedRoot.w_13_0 ctx)

def generatedRoot.w_23_0  (ctx : generatedRootCtx) : Fin 4 → mxxMat0 :=
  (generatedRoot.w_9_0 ctx)

def generatedRoot.w_23_1  (ctx : generatedRootCtx) : Fin 4 → mxxMat0 :=
  (generatedRoot.w_10_0 ctx)

def generatedRoot.w_24_0  (ctx : generatedRootCtx) : Fin 128 → mxxMat0 :=
  (generatedRoot.w_12_0 ctx)

def generatedRoot.w_24_1  (ctx : generatedRootCtx) : Fin 128 → mxxMat0 :=
  (generatedRoot.w_13_0 ctx)

def generatedRoot.w_25_0  (ctx : generatedRootCtx) : Fin 4 → mxxMat0 :=
  (generatedRoot.w_9_0 ctx)

def generatedRoot.w_25_1  (ctx : generatedRootCtx) : Fin 4 → mxxMat0 :=
  (generatedRoot.w_10_0 ctx)

def generatedRoot.w_26_0  (ctx : generatedRootCtx) : Fin 128 → mxxMat0 :=
  (generatedRoot.w_12_0 ctx)

def generatedRoot.w_26_1  (ctx : generatedRootCtx) : Fin 128 → mxxMat0 :=
  (generatedRoot.w_13_0 ctx)

def generatedRoot.w_27_0  (ctx : generatedRootCtx) : Fin 4 → mxxMat0 :=
  (generatedRoot.w_9_0 ctx)

def generatedRoot.w_27_1  (ctx : generatedRootCtx) : Fin 4 → mxxMat0 :=
  (generatedRoot.w_10_0 ctx)

def generatedRoot.w_28_0  (ctx : generatedRootCtx) : Fin 128 → mxxMat0 :=
  (generatedRoot.w_12_0 ctx)

def generatedRoot.w_28_1  (ctx : generatedRootCtx) : Fin 128 → mxxMat0 :=
  (generatedRoot.w_13_0 ctx)

def generatedRoot.w_29_0  (ctx : generatedRootCtx) : Fin 4 → mxxMat0 :=
  (generatedRoot.w_9_0 ctx)

def generatedRoot.w_29_1  (ctx : generatedRootCtx) : Fin 4 → mxxMat0 :=
  (generatedRoot.w_10_0 ctx)

def generatedRoot.w_30_0  (ctx : generatedRootCtx) : Fin 128 → mxxMat0 :=
  (generatedRoot.w_12_0 ctx)

def generatedRoot.w_30_1  (ctx : generatedRootCtx) : Fin 128 → mxxMat0 :=
  (generatedRoot.w_13_0 ctx)

def generatedRoot.w_31_0  (ctx : generatedRootCtx) : Fin 4 → mxxMat0 :=
  (generatedRoot.w_9_0 ctx)

def generatedRoot.w_31_1  (ctx : generatedRootCtx) : Fin 4 → mxxMat0 :=
  (generatedRoot.w_10_0 ctx)

def generatedRoot.w_32_0  (ctx : generatedRootCtx) : Fin 128 → mxxMat0 :=
  (generatedRoot.w_12_0 ctx)

def generatedRoot.w_32_1  (ctx : generatedRootCtx) : Fin 128 → mxxMat0 :=
  (generatedRoot.w_13_0 ctx)

def generatedRoot.w_33_0  (ctx : generatedRootCtx) : Fin 4 → mxxMat0 :=
  (generatedRoot.w_9_0 ctx)

def generatedRoot.w_33_1  (ctx : generatedRootCtx) : Fin 4 → mxxMat0 :=
  (generatedRoot.w_10_0 ctx)

def generatedRoot.w_34_0  (ctx : generatedRootCtx) : Fin 128 → mxxMat0 :=
  (generatedRoot.w_12_0 ctx)

def generatedRoot.w_34_1  (ctx : generatedRootCtx) : Fin 128 → mxxMat0 :=
  (generatedRoot.w_13_0 ctx)

def generatedRoot.w_35_0  (ctx : generatedRootCtx) : Fin 4 → mxxMat0 :=
  (generatedRoot.w_9_0 ctx)

def generatedRoot.w_35_1  (ctx : generatedRootCtx) : Fin 4 → mxxMat0 :=
  (generatedRoot.w_10_0 ctx)

def generatedRoot.w_36_0  (ctx : generatedRootCtx) : Fin 128 → mxxMat0 :=
  (generatedRoot.w_12_0 ctx)

def generatedRoot.w_36_1  (ctx : generatedRootCtx) : Fin 128 → mxxMat0 :=
  (generatedRoot.w_13_0 ctx)

def generatedRoot.w_37_0  (ctx : generatedRootCtx) : Fin 4 → mxxMat0 :=
  (generatedRoot.w_9_0 ctx)

def generatedRoot.w_37_1  (ctx : generatedRootCtx) : Fin 4 → mxxMat0 :=
  (generatedRoot.w_10_0 ctx)

def generatedRoot.w_38_0  (ctx : generatedRootCtx) : Fin 128 → mxxMat0 :=
  (generatedRoot.w_12_0 ctx)

def generatedRoot.w_38_1  (ctx : generatedRootCtx) : Fin 128 → mxxMat0 :=
  (generatedRoot.w_13_0 ctx)

def generatedRoot.w_39_0  (ctx : generatedRootCtx) : Fin 4 → mxxMat0 :=
  (generatedRoot.w_9_0 ctx)

def generatedRoot.w_39_1  (ctx : generatedRootCtx) : Fin 4 → mxxMat0 :=
  (generatedRoot.w_10_0 ctx)

def generatedRoot.w_40_0  (ctx : generatedRootCtx) : Fin 128 → mxxMat0 :=
  (generatedRoot.w_12_0 ctx)

def generatedRoot.w_40_1  (ctx : generatedRootCtx) : Fin 128 → mxxMat0 :=
  (generatedRoot.w_13_0 ctx)

def generatedRoot.w_41_0  (ctx : generatedRootCtx) : Fin 4 → mxxMat0 :=
  (generatedRoot.w_9_0 ctx)

def generatedRoot.w_41_1  (ctx : generatedRootCtx) : Fin 4 → mxxMat0 :=
  (generatedRoot.w_10_0 ctx)

def generatedRoot.w_42_0  (ctx : generatedRootCtx) : Fin 128 → mxxMat0 :=
  (generatedRoot.w_12_0 ctx)

def generatedRoot.w_42_1  (ctx : generatedRootCtx) : Fin 128 → mxxMat0 :=
  (generatedRoot.w_13_0 ctx)

def generatedRoot.w_43_0  (ctx : generatedRootCtx) : Fin 4 → mxxMat0 :=
  (generatedRoot.w_9_0 ctx)

def generatedRoot.w_43_1  (ctx : generatedRootCtx) : Fin 4 → mxxMat0 :=
  (generatedRoot.w_10_0 ctx)

def generatedRoot.w_44_0  (ctx : generatedRootCtx) : Fin 128 → mxxMat0 :=
  (generatedRoot.w_12_0 ctx)

def generatedRoot.w_44_1  (ctx : generatedRootCtx) : Fin 128 → mxxMat0 :=
  (generatedRoot.w_13_0 ctx)

def generatedRoot.w_45_0  (ctx : generatedRootCtx) : Fin 4 → mxxMat0 :=
  (generatedRoot.w_9_0 ctx)

def generatedRoot.w_45_1  (ctx : generatedRootCtx) : Fin 4 → mxxMat0 :=
  (generatedRoot.w_10_0 ctx)

def generatedRoot.w_46_0  (ctx : generatedRootCtx) : Fin 128 → mxxMat0 :=
  (generatedRoot.w_12_0 ctx)

def generatedRoot.w_46_1  (ctx : generatedRootCtx) : Fin 128 → mxxMat0 :=
  (generatedRoot.w_13_0 ctx)

def generatedRoot.w_47_0  (ctx : generatedRootCtx) : Fin 4 → mxxMat0 :=
  (generatedRoot.w_9_0 ctx)

def generatedRoot.w_47_1  (ctx : generatedRootCtx) : Fin 4 → mxxMat0 :=
  (generatedRoot.w_10_0 ctx)

def generatedRoot.w_48_0  (ctx : generatedRootCtx) : Fin 128 → mxxMat0 :=
  (generatedRoot.w_12_0 ctx)

def generatedRoot.w_48_1  (ctx : generatedRootCtx) : Fin 128 → mxxMat0 :=
  (generatedRoot.w_13_0 ctx)

def generatedRoot.w_49_0  (ctx : generatedRootCtx) : Fin 4 → mxxMat0 :=
  (generatedRoot.w_9_0 ctx)

def generatedRoot.w_49_1  (ctx : generatedRootCtx) : Fin 4 → mxxMat0 :=
  (generatedRoot.w_10_0 ctx)

def generatedRoot.w_50_0  (ctx : generatedRootCtx) : Fin 128 → mxxMat0 :=
  (generatedRoot.w_12_0 ctx)

def generatedRoot.w_50_1  (ctx : generatedRootCtx) : Fin 128 → mxxMat0 :=
  (generatedRoot.w_13_0 ctx)

def generatedRoot.w_51_0  (ctx : generatedRootCtx) : Fin 4 → mxxMat0 :=
  (generatedRoot.w_9_0 ctx)

def generatedRoot.w_51_1  (ctx : generatedRootCtx) : Fin 4 → mxxMat0 :=
  (generatedRoot.w_10_0 ctx)

def generatedRoot.w_52_0  (ctx : generatedRootCtx) : Fin 128 → mxxMat0 :=
  (generatedRoot.w_12_0 ctx)

def generatedRoot.w_52_1  (ctx : generatedRootCtx) : Fin 128 → mxxMat0 :=
  (generatedRoot.w_13_0 ctx)

def generatedRoot.w_53_0  (ctx : generatedRootCtx) : Fin 4 → mxxMat0 :=
  (generatedRoot.w_9_0 ctx)

def generatedRoot.w_53_1  (ctx : generatedRootCtx) : Fin 4 → mxxMat0 :=
  (generatedRoot.w_10_0 ctx)

def generatedRoot.w_54_0  (ctx : generatedRootCtx) : Fin 128 → mxxMat0 :=
  (generatedRoot.w_12_0 ctx)

def generatedRoot.w_54_1  (ctx : generatedRootCtx) : Fin 128 → mxxMat0 :=
  (generatedRoot.w_13_0 ctx)

def generatedRoot.w_55_0  (ctx : generatedRootCtx) : Fin 4 → mxxMat0 :=
  (generatedRoot.w_9_0 ctx)

def generatedRoot.w_55_1  (ctx : generatedRootCtx) : Fin 4 → mxxMat0 :=
  (generatedRoot.w_10_0 ctx)

def generatedRoot.w_56_0  (ctx : generatedRootCtx) : Fin 128 → mxxMat0 :=
  (generatedRoot.w_12_0 ctx)

def generatedRoot.w_56_1  (ctx : generatedRootCtx) : Fin 128 → mxxMat0 :=
  (generatedRoot.w_13_0 ctx)

def generatedRoot.w_57_0  (ctx : generatedRootCtx) : Fin 4 → mxxMat0 :=
  (generatedRoot.w_9_0 ctx)

def generatedRoot.w_57_1  (ctx : generatedRootCtx) : Fin 4 → mxxMat0 :=
  (generatedRoot.w_10_0 ctx)

def generatedRoot.w_58_0  (ctx : generatedRootCtx) : Fin 128 → mxxMat0 :=
  (generatedRoot.w_12_0 ctx)

def generatedRoot.w_58_1  (ctx : generatedRootCtx) : Fin 128 → mxxMat0 :=
  (generatedRoot.w_13_0 ctx)

def generatedRoot.w_59_0  (ctx : generatedRootCtx) : Fin 4 → mxxMat0 :=
  (generatedRoot.w_9_0 ctx)

def generatedRoot.w_59_1  (ctx : generatedRootCtx) : Fin 4 → mxxMat0 :=
  (generatedRoot.w_10_0 ctx)

def generatedRoot.w_60_0  (ctx : generatedRootCtx) : Fin 128 → mxxMat0 :=
  (generatedRoot.w_12_0 ctx)

def generatedRoot.w_60_1  (ctx : generatedRootCtx) : Fin 128 → mxxMat0 :=
  (generatedRoot.w_13_0 ctx)

def generatedRoot.w_61_0  (ctx : generatedRootCtx) : Fin 4 → mxxMat0 :=
  (generatedRoot.w_9_0 ctx)

def generatedRoot.w_61_1  (ctx : generatedRootCtx) : Fin 4 → mxxMat0 :=
  (generatedRoot.w_10_0 ctx)

def generatedRoot.w_62_0  (ctx : generatedRootCtx) : Fin 128 → mxxMat0 :=
  (generatedRoot.w_12_0 ctx)

def generatedRoot.w_62_1  (ctx : generatedRootCtx) : Fin 128 → mxxMat0 :=
  (generatedRoot.w_13_0 ctx)

def generatedRoot.w_63_0  (ctx : generatedRootCtx) : Fin 4 → mxxMat0 :=
  (generatedRoot.w_9_0 ctx)

def generatedRoot.w_63_1  (ctx : generatedRootCtx) : Fin 4 → mxxMat0 :=
  (generatedRoot.w_10_0 ctx)

def generatedRoot.w_64_0  (ctx : generatedRootCtx) : Fin 128 → mxxMat0 :=
  (generatedRoot.w_12_0 ctx)

def generatedRoot.w_64_1  (ctx : generatedRootCtx) : Fin 128 → mxxMat0 :=
  (generatedRoot.w_13_0 ctx)

def generatedRoot.w_65_0  (ctx : generatedRootCtx) : Fin 4 → mxxMat0 :=
  (generatedRoot.w_9_0 ctx)

def generatedRoot.w_65_1  (ctx : generatedRootCtx) : Fin 4 → mxxMat0 :=
  (generatedRoot.w_10_0 ctx)

def generatedRoot.w_66_0  (ctx : generatedRootCtx) : Fin 128 → mxxMat0 :=
  (generatedRoot.w_12_0 ctx)

def generatedRoot.w_66_1  (ctx : generatedRootCtx) : Fin 128 → mxxMat0 :=
  (generatedRoot.w_13_0 ctx)

def generatedRoot.w_67_0  (ctx : generatedRootCtx) : Fin 4 → mxxMat0 :=
  (generatedRoot.w_9_0 ctx)

def generatedRoot.w_67_1  (ctx : generatedRootCtx) : Fin 4 → mxxMat0 :=
  (generatedRoot.w_10_0 ctx)

def generatedRoot.w_68_0  (ctx : generatedRootCtx) : Fin 128 → mxxMat0 :=
  (generatedRoot.w_12_0 ctx)

def generatedRoot.w_68_1  (ctx : generatedRootCtx) : Fin 128 → mxxMat0 :=
  (generatedRoot.w_13_0 ctx)

def generatedRoot.w_69_0  (ctx : generatedRootCtx) : Fin 4 → mxxMat0 :=
  (generatedRoot.w_9_0 ctx)

def generatedRoot.w_69_1  (ctx : generatedRootCtx) : Fin 4 → mxxMat0 :=
  (generatedRoot.w_10_0 ctx)

def generatedRoot.w_70_0  (ctx : generatedRootCtx) : Fin 128 → mxxMat0 :=
  (generatedRoot.w_12_0 ctx)

def generatedRoot.w_70_1  (ctx : generatedRootCtx) : Fin 128 → mxxMat0 :=
  (generatedRoot.w_13_0 ctx)

def generatedRoot.w_71_0  (ctx : generatedRootCtx) : Fin 4 → mxxMat0 :=
  (generatedRoot.w_9_0 ctx)

def generatedRoot.w_71_1  (ctx : generatedRootCtx) : Fin 4 → mxxMat0 :=
  (generatedRoot.w_10_0 ctx)

def generatedRoot.w_72_0  (ctx : generatedRootCtx) : Fin 128 → mxxMat0 :=
  (generatedRoot.w_12_0 ctx)

def generatedRoot.w_72_1  (ctx : generatedRootCtx) : Fin 128 → mxxMat0 :=
  (generatedRoot.w_13_0 ctx)

def generatedRoot.w_73_0  (ctx : generatedRootCtx) : Fin 4 → mxxMat0 :=
  (generatedRoot.w_9_0 ctx)

def generatedRoot.w_73_1  (ctx : generatedRootCtx) : Fin 4 → mxxMat0 :=
  (generatedRoot.w_10_0 ctx)

def generatedRoot.w_74_0  (ctx : generatedRootCtx) : Fin 128 → mxxMat0 :=
  (generatedRoot.w_12_0 ctx)

def generatedRoot.w_74_1  (ctx : generatedRootCtx) : Fin 128 → mxxMat0 :=
  (generatedRoot.w_13_0 ctx)

def generatedRoot.w_75_0  (ctx : generatedRootCtx) : Fin 4 → mxxMat0 :=
  (generatedRoot.w_9_0 ctx)

def generatedRoot.w_75_1  (ctx : generatedRootCtx) : Fin 4 → mxxMat0 :=
  (generatedRoot.w_10_0 ctx)

def generatedRoot.w_76_0  (ctx : generatedRootCtx) : Fin 128 → mxxMat0 :=
  (generatedRoot.w_12_0 ctx)

def generatedRoot.w_76_1  (ctx : generatedRootCtx) : Fin 128 → mxxMat0 :=
  (generatedRoot.w_13_0 ctx)

def generatedRoot.w_77_0  (ctx : generatedRootCtx) : Fin 4 → mxxMat0 :=
  (generatedRoot.w_9_0 ctx)

def generatedRoot.w_77_1  (ctx : generatedRootCtx) : Fin 4 → mxxMat0 :=
  (generatedRoot.w_10_0 ctx)

def generatedRoot.w_78_0  (ctx : generatedRootCtx) : Fin 128 → mxxMat0 :=
  (generatedRoot.w_12_0 ctx)

def generatedRoot.w_78_1  (ctx : generatedRootCtx) : Fin 128 → mxxMat0 :=
  (generatedRoot.w_13_0 ctx)

def generatedRoot.w_79_0  (ctx : generatedRootCtx) : Fin 4 → mxxMat0 :=
  (generatedRoot.w_9_0 ctx)

def generatedRoot.w_79_1  (ctx : generatedRootCtx) : Fin 4 → mxxMat0 :=
  (generatedRoot.w_10_0 ctx)

def generatedRoot.w_80_0  (ctx : generatedRootCtx) : Fin 128 → mxxMat0 :=
  (generatedRoot.w_12_0 ctx)

def generatedRoot.w_80_1  (ctx : generatedRootCtx) : Fin 128 → mxxMat0 :=
  (generatedRoot.w_13_0 ctx)

def generatedRoot.w_81_0  (ctx : generatedRootCtx) : Fin 4 → mxxMat0 :=
  (generatedRoot.w_9_0 ctx)

def generatedRoot.w_81_1  (ctx : generatedRootCtx) : Fin 4 → mxxMat0 :=
  (generatedRoot.w_10_0 ctx)

def generatedRoot.w_82_0  (ctx : generatedRootCtx) : Fin 128 → mxxMat0 :=
  (generatedRoot.w_12_0 ctx)

def generatedRoot.w_82_1  (ctx : generatedRootCtx) : Fin 128 → mxxMat0 :=
  (generatedRoot.w_13_0 ctx)

def generatedRoot.w_83_0  (ctx : generatedRootCtx) : Fin 4 → mxxMat0 :=
  (generatedRoot.w_9_0 ctx)

def generatedRoot.w_83_1  (ctx : generatedRootCtx) : Fin 4 → mxxMat0 :=
  (generatedRoot.w_10_0 ctx)

def generatedRoot.w_84_0  (ctx : generatedRootCtx) : Fin 128 → mxxMat0 :=
  (generatedRoot.w_12_0 ctx)

def generatedRoot.w_84_1  (ctx : generatedRootCtx) : Fin 128 → mxxMat0 :=
  (generatedRoot.w_13_0 ctx)

def generatedRoot.w_85_0  (ctx : generatedRootCtx) : Fin 4 → mxxMat0 :=
  (generatedRoot.w_9_0 ctx)

def generatedRoot.w_85_1  (ctx : generatedRootCtx) : Fin 4 → mxxMat0 :=
  (generatedRoot.w_10_0 ctx)

def generatedRoot.w_86_0  (ctx : generatedRootCtx) : Fin 128 → mxxMat0 :=
  (generatedRoot.w_12_0 ctx)

def generatedRoot.w_86_1  (ctx : generatedRootCtx) : Fin 128 → mxxMat0 :=
  (generatedRoot.w_13_0 ctx)

def generatedRoot.outs  (ctx : generatedRootCtx) : (generatedRoot.OutputType) :=
  ((generatedRoot.w_0_0 ctx), ((generatedRoot.w_1_0 ctx), ((generatedRoot.w_6_0 ctx), ((generatedRoot.w_6_1 ctx), ((generatedRoot.w_6_2 ctx), ((generatedRoot.w_7_0 ctx), ((generatedRoot.w_8_0 ctx), ((generatedRoot.w_11_0 ctx), ((generatedRoot.w_11_1 ctx), ((generatedRoot.w_14_0 ctx), ((generatedRoot.w_14_1 ctx), ((generatedRoot.w_15_0 ctx), ((generatedRoot.w_15_1 ctx), ((generatedRoot.w_16_0 ctx), ((generatedRoot.w_16_1 ctx), ((generatedRoot.w_17_0 ctx), ((generatedRoot.w_17_1 ctx), ((generatedRoot.w_18_0 ctx), ((generatedRoot.w_18_1 ctx), ((generatedRoot.w_19_0 ctx), ((generatedRoot.w_19_1 ctx), ((generatedRoot.w_20_0 ctx), ((generatedRoot.w_20_1 ctx), ((generatedRoot.w_21_0 ctx), ((generatedRoot.w_21_1 ctx), ((generatedRoot.w_22_0 ctx), ((generatedRoot.w_22_1 ctx), ((generatedRoot.w_23_0 ctx), ((generatedRoot.w_23_1 ctx), ((generatedRoot.w_24_0 ctx), ((generatedRoot.w_24_1 ctx), ((generatedRoot.w_25_0 ctx), ((generatedRoot.w_25_1 ctx), ((generatedRoot.w_26_0 ctx), ((generatedRoot.w_26_1 ctx), ((generatedRoot.w_27_0 ctx), ((generatedRoot.w_27_1 ctx), ((generatedRoot.w_28_0 ctx), ((generatedRoot.w_28_1 ctx), ((generatedRoot.w_29_0 ctx), ((generatedRoot.w_29_1 ctx), ((generatedRoot.w_30_0 ctx), ((generatedRoot.w_30_1 ctx), ((generatedRoot.w_31_0 ctx), ((generatedRoot.w_31_1 ctx), ((generatedRoot.w_32_0 ctx), ((generatedRoot.w_32_1 ctx), ((generatedRoot.w_33_0 ctx), ((generatedRoot.w_33_1 ctx), ((generatedRoot.w_34_0 ctx), ((generatedRoot.w_34_1 ctx), ((generatedRoot.w_35_0 ctx), ((generatedRoot.w_35_1 ctx), ((generatedRoot.w_36_0 ctx), ((generatedRoot.w_36_1 ctx), ((generatedRoot.w_37_0 ctx), ((generatedRoot.w_37_1 ctx), ((generatedRoot.w_38_0 ctx), ((generatedRoot.w_38_1 ctx), ((generatedRoot.w_39_0 ctx), ((generatedRoot.w_39_1 ctx), ((generatedRoot.w_40_0 ctx), ((generatedRoot.w_40_1 ctx), ((generatedRoot.w_41_0 ctx), ((generatedRoot.w_41_1 ctx), ((generatedRoot.w_42_0 ctx), ((generatedRoot.w_42_1 ctx), ((generatedRoot.w_43_0 ctx), ((generatedRoot.w_43_1 ctx), ((generatedRoot.w_44_0 ctx), ((generatedRoot.w_44_1 ctx), ((generatedRoot.w_45_0 ctx), ((generatedRoot.w_45_1 ctx), ((generatedRoot.w_46_0 ctx), ((generatedRoot.w_46_1 ctx), ((generatedRoot.w_47_0 ctx), ((generatedRoot.w_47_1 ctx), ((generatedRoot.w_48_0 ctx), ((generatedRoot.w_48_1 ctx), ((generatedRoot.w_49_0 ctx), ((generatedRoot.w_49_1 ctx), ((generatedRoot.w_50_0 ctx), ((generatedRoot.w_50_1 ctx), ((generatedRoot.w_51_0 ctx), ((generatedRoot.w_51_1 ctx), ((generatedRoot.w_52_0 ctx), ((generatedRoot.w_52_1 ctx), ((generatedRoot.w_53_0 ctx), ((generatedRoot.w_53_1 ctx), ((generatedRoot.w_54_0 ctx), ((generatedRoot.w_54_1 ctx), ((generatedRoot.w_55_0 ctx), ((generatedRoot.w_55_1 ctx), ((generatedRoot.w_56_0 ctx), ((generatedRoot.w_56_1 ctx), ((generatedRoot.w_57_0 ctx), ((generatedRoot.w_57_1 ctx), ((generatedRoot.w_58_0 ctx), ((generatedRoot.w_58_1 ctx), ((generatedRoot.w_59_0 ctx), ((generatedRoot.w_59_1 ctx), ((generatedRoot.w_60_0 ctx), ((generatedRoot.w_60_1 ctx), ((generatedRoot.w_61_0 ctx), ((generatedRoot.w_61_1 ctx), ((generatedRoot.w_62_0 ctx), ((generatedRoot.w_62_1 ctx), ((generatedRoot.w_63_0 ctx), ((generatedRoot.w_63_1 ctx), ((generatedRoot.w_64_0 ctx), ((generatedRoot.w_64_1 ctx), ((generatedRoot.w_65_0 ctx), ((generatedRoot.w_65_1 ctx), ((generatedRoot.w_66_0 ctx), ((generatedRoot.w_66_1 ctx), ((generatedRoot.w_67_0 ctx), ((generatedRoot.w_67_1 ctx), ((generatedRoot.w_68_0 ctx), ((generatedRoot.w_68_1 ctx), ((generatedRoot.w_69_0 ctx), ((generatedRoot.w_69_1 ctx), ((generatedRoot.w_70_0 ctx), ((generatedRoot.w_70_1 ctx), ((generatedRoot.w_71_0 ctx), ((generatedRoot.w_71_1 ctx), ((generatedRoot.w_72_0 ctx), ((generatedRoot.w_72_1 ctx), ((generatedRoot.w_73_0 ctx), ((generatedRoot.w_73_1 ctx), ((generatedRoot.w_74_0 ctx), ((generatedRoot.w_74_1 ctx), ((generatedRoot.w_75_0 ctx), ((generatedRoot.w_75_1 ctx), ((generatedRoot.w_76_0 ctx), ((generatedRoot.w_76_1 ctx), ((generatedRoot.w_77_0 ctx), ((generatedRoot.w_77_1 ctx), ((generatedRoot.w_78_0 ctx), ((generatedRoot.w_78_1 ctx), ((generatedRoot.w_79_0 ctx), ((generatedRoot.w_79_1 ctx), ((generatedRoot.w_80_0 ctx), ((generatedRoot.w_80_1 ctx), ((generatedRoot.w_81_0 ctx), ((generatedRoot.w_81_1 ctx), ((generatedRoot.w_82_0 ctx), ((generatedRoot.w_82_1 ctx), ((generatedRoot.w_83_0 ctx), ((generatedRoot.w_83_1 ctx), ((generatedRoot.w_84_0 ctx), ((generatedRoot.w_84_1 ctx), ((generatedRoot.w_85_0 ctx), ((generatedRoot.w_85_1 ctx), ((generatedRoot.w_86_0 ctx), ((generatedRoot.w_86_1 ctx), ())))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))

structure generatedRoot.Facts_0  (ctx : generatedRootCtx) : Prop where
  t0 : (1025) = 1025
  h6_2 : (∀ i : Fin 1025, parallel_generatedRoot_6 params i (((generatedRoot.w_2_0 ctx) i), (((generatedRoot.w_3_0 ctx) i), (((generatedRoot.w_4_0 ctx) i), (((generatedRoot.w_5_0 ctx) i), ())))) (((generatedRoot.w_6_0 ctx) i), (((generatedRoot.w_6_1 ctx) i), (((generatedRoot.w_6_2 ctx) i), ()))))
  t2 : (4) = 4
  t3 : (128) = 128
  t4 : (4) = 4
  t5 : (128) = 128
  t6 : (4) = 4
  t7 : (128) = 128
  t8 : (4) = 4
  t9 : (128) = 128
  t10 : (4) = 4
  t11 : (128) = 128
  t12 : (4) = 4
  t13 : (128) = 128
  t14 : (4) = 4
  t15 : (128) = 128
  t16 : (4) = 4
  t17 : (128) = 128
  t18 : (4) = 4
  t19 : (128) = 128
  t20 : (4) = 4
  t21 : (128) = 128
  t22 : (4) = 4
  t23 : (128) = 128
  t24 : (4) = 4
  t25 : (128) = 128
  t26 : (4) = 4
  t27 : (128) = 128
  t28 : (4) = 4
  t29 : (128) = 128
  t30 : (4) = 4
  t31 : (128) = 128

theorem generatedRoot.facts_0  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_0 params (generatedRoot.w_2_0 ctx) (generatedRoot.w_3_0 ctx) (generatedRoot.w_4_0 ctx) (generatedRoot.w_5_0 ctx) (generatedRoot.w_6_0 ctx) (generatedRoot.w_6_1 ctx) (generatedRoot.w_6_2 ctx)) :
    generatedRoot.Facts_0 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_1  (ctx : generatedRootCtx) : Prop where
  t32 : (4) = 4
  t33 : (128) = 128
  t34 : (4) = 4
  t35 : (128) = 128
  t36 : (4) = 4
  t37 : (128) = 128
  t38 : (4) = 4
  t39 : (128) = 128
  t40 : (4) = 4
  t41 : (128) = 128
  t42 : (4) = 4
  t43 : (128) = 128
  t44 : (4) = 4
  t45 : (128) = 128
  t46 : (4) = 4
  t47 : (128) = 128
  t48 : (4) = 4
  t49 : (128) = 128
  t50 : (4) = 4
  t51 : (128) = 128
  t52 : (4) = 4
  t53 : (128) = 128
  t54 : (4) = 4
  t55 : (128) = 128
  t56 : (4) = 4
  t57 : (128) = 128
  t58 : (4) = 4
  t59 : (128) = 128
  t60 : (4) = 4
  t61 : (128) = 128
  t62 : (4) = 4
  t63 : (128) = 128

theorem generatedRoot.facts_1  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_1 ) :
    generatedRoot.Facts_1 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_2  (ctx : generatedRootCtx) (outputs : (generatedRoot.OutputType)) : Prop where
  t64 : (4) = 4
  t65 : (128) = 128
  t66 : (4) = 4
  t67 : (128) = 128
  t68 : (4) = 4
  t69 : (128) = 128
  t70 : (4) = 4
  t71 : (128) = 128
  t72 : (4) = 4
  t73 : (128) = 128
  t74 : (4) = 4
  t75 : (128) = 128
  hout : outputs = generatedRoot.outs ctx

theorem generatedRoot.facts_2  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_2 (generatedRoot.w_0_0 ctx) (generatedRoot.w_1_0 ctx) (generatedRoot.w_7_0 ctx) (generatedRoot.w_8_0 ctx) (generatedRoot.w_6_0 ctx) (generatedRoot.w_6_1 ctx) (generatedRoot.w_6_2 ctx) (generatedRoot.w_11_0 ctx) (generatedRoot.w_11_1 ctx) (generatedRoot.w_14_0 ctx) (generatedRoot.w_14_1 ctx) (generatedRoot.w_15_0 ctx) (generatedRoot.w_15_1 ctx) (generatedRoot.w_16_0 ctx) (generatedRoot.w_16_1 ctx) (generatedRoot.w_17_0 ctx) (generatedRoot.w_17_1 ctx) (generatedRoot.w_18_0 ctx) (generatedRoot.w_18_1 ctx) (generatedRoot.w_19_0 ctx) (generatedRoot.w_19_1 ctx) (generatedRoot.w_20_0 ctx) (generatedRoot.w_20_1 ctx) (generatedRoot.w_21_0 ctx) (generatedRoot.w_21_1 ctx) (generatedRoot.w_22_0 ctx) (generatedRoot.w_22_1 ctx) (generatedRoot.w_23_0 ctx) (generatedRoot.w_23_1 ctx) (generatedRoot.w_24_0 ctx) (generatedRoot.w_24_1 ctx) (generatedRoot.w_25_0 ctx) (generatedRoot.w_25_1 ctx) (generatedRoot.w_26_0 ctx) (generatedRoot.w_26_1 ctx) (generatedRoot.w_27_0 ctx) (generatedRoot.w_27_1 ctx) (generatedRoot.w_28_0 ctx) (generatedRoot.w_28_1 ctx) (generatedRoot.w_29_0 ctx) (generatedRoot.w_29_1 ctx) (generatedRoot.w_30_0 ctx) (generatedRoot.w_30_1 ctx) (generatedRoot.w_31_0 ctx) (generatedRoot.w_31_1 ctx) (generatedRoot.w_32_0 ctx) (generatedRoot.w_32_1 ctx) (generatedRoot.w_33_0 ctx) (generatedRoot.w_33_1 ctx) (generatedRoot.w_34_0 ctx) (generatedRoot.w_34_1 ctx) (generatedRoot.w_35_0 ctx) (generatedRoot.w_35_1 ctx) (generatedRoot.w_36_0 ctx) (generatedRoot.w_36_1 ctx) (generatedRoot.w_37_0 ctx) (generatedRoot.w_37_1 ctx) (generatedRoot.w_38_0 ctx) (generatedRoot.w_38_1 ctx) (generatedRoot.w_39_0 ctx) (generatedRoot.w_39_1 ctx) (generatedRoot.w_40_0 ctx) (generatedRoot.w_40_1 ctx) (generatedRoot.w_41_0 ctx) (generatedRoot.w_41_1 ctx) (generatedRoot.w_42_0 ctx) (generatedRoot.w_42_1 ctx) (generatedRoot.w_43_0 ctx) (generatedRoot.w_43_1 ctx) (generatedRoot.w_44_0 ctx) (generatedRoot.w_44_1 ctx) (generatedRoot.w_45_0 ctx) (generatedRoot.w_45_1 ctx) (generatedRoot.w_46_0 ctx) (generatedRoot.w_46_1 ctx) (generatedRoot.w_47_0 ctx) (generatedRoot.w_47_1 ctx) (generatedRoot.w_48_0 ctx) (generatedRoot.w_48_1 ctx) (generatedRoot.w_49_0 ctx) (generatedRoot.w_49_1 ctx) (generatedRoot.w_50_0 ctx) (generatedRoot.w_50_1 ctx) (generatedRoot.w_51_0 ctx) (generatedRoot.w_51_1 ctx) (generatedRoot.w_52_0 ctx) (generatedRoot.w_52_1 ctx) (generatedRoot.w_53_0 ctx) (generatedRoot.w_53_1 ctx) (generatedRoot.w_54_0 ctx) (generatedRoot.w_54_1 ctx) (generatedRoot.w_55_0 ctx) (generatedRoot.w_55_1 ctx) (generatedRoot.w_56_0 ctx) (generatedRoot.w_56_1 ctx) (generatedRoot.w_57_0 ctx) (generatedRoot.w_57_1 ctx) (generatedRoot.w_58_0 ctx) (generatedRoot.w_58_1 ctx) (generatedRoot.w_59_0 ctx) (generatedRoot.w_59_1 ctx) (generatedRoot.w_60_0 ctx) (generatedRoot.w_60_1 ctx) (generatedRoot.w_61_0 ctx) (generatedRoot.w_61_1 ctx) (generatedRoot.w_62_0 ctx) (generatedRoot.w_62_1 ctx) (generatedRoot.w_63_0 ctx) (generatedRoot.w_63_1 ctx) (generatedRoot.w_64_0 ctx) (generatedRoot.w_64_1 ctx) (generatedRoot.w_65_0 ctx) (generatedRoot.w_65_1 ctx) (generatedRoot.w_66_0 ctx) (generatedRoot.w_66_1 ctx) (generatedRoot.w_67_0 ctx) (generatedRoot.w_67_1 ctx) (generatedRoot.w_68_0 ctx) (generatedRoot.w_68_1 ctx) (generatedRoot.w_69_0 ctx) (generatedRoot.w_69_1 ctx) (generatedRoot.w_70_0 ctx) (generatedRoot.w_70_1 ctx) (generatedRoot.w_71_0 ctx) (generatedRoot.w_71_1 ctx) (generatedRoot.w_72_0 ctx) (generatedRoot.w_72_1 ctx) (generatedRoot.w_73_0 ctx) (generatedRoot.w_73_1 ctx) (generatedRoot.w_74_0 ctx) (generatedRoot.w_74_1 ctx) (generatedRoot.w_75_0 ctx) (generatedRoot.w_75_1 ctx) (generatedRoot.w_76_0 ctx) (generatedRoot.w_76_1 ctx) (generatedRoot.w_77_0 ctx) (generatedRoot.w_77_1 ctx) (generatedRoot.w_78_0 ctx) (generatedRoot.w_78_1 ctx) (generatedRoot.w_79_0 ctx) (generatedRoot.w_79_1 ctx) (generatedRoot.w_80_0 ctx) (generatedRoot.w_80_1 ctx) (generatedRoot.w_81_0 ctx) (generatedRoot.w_81_1 ctx) (generatedRoot.w_82_0 ctx) (generatedRoot.w_82_1 ctx) (generatedRoot.w_83_0 ctx) (generatedRoot.w_83_1 ctx) (generatedRoot.w_84_0 ctx) (generatedRoot.w_84_1 ctx) (generatedRoot.w_85_0 ctx) (generatedRoot.w_85_1 ctx) (generatedRoot.w_86_0 ctx) (generatedRoot.w_86_1 ctx) outputs) :
    generatedRoot.Facts_2 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts  (ctx : generatedRootCtx) (outputs : (generatedRoot.OutputType)) : Prop where
  c0 : generatedRoot.Facts_0 ctx
  c1 : generatedRoot.Facts_1 ctx
  c2 : generatedRoot.Facts_2 ctx outputs

theorem generatedRoot.facts  (ctx : generatedRootCtx) {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.body params ctx.inputs outputs ctx.witness) :
    generatedRoot.Facts ctx outputs :=
  have t0 := h
  have r0 := generatedRoot.facts_0 (ctx := ctx) (outputs := outputs) t0.1
  have t1 := t0.2
  have r1 := generatedRoot.facts_1 (ctx := ctx) (outputs := outputs) t1.1
  have t2 := t1.2
  have r2 := generatedRoot.facts_2 (ctx := ctx) (outputs := outputs) t2
  ⟨r0, r1, r2⟩

end

end Stage_import_1

end
