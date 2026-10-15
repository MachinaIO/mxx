import Stage_keys_import_0_0

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000
set_option linter.unusedVariables false

-- The wire types repeat the ring modulus `X ^ n + 1`; unfolding it while checking a
-- projection against its field type expands it at the ring dimension.
attribute [local irreducible] Mxx.Primitives.negacyclicModulus

noncomputable section

namespace Stage_keys_import_0_0

local notation "mxxMat0" => Mxx.Primitives.ExactMatrix modulus_0 8 1 25
local notation "mxxMat1" => Mxx.Primitives.ExactMatrix modulus_0 8 1 1

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Ctx (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) where
  inputs : (generatedRoot.InputType)
  witness : generatedRoot.Witness

section
variable {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params}
local notation "generatedRootCtx" => (generatedRoot.Ctx tape path params)

def generatedRoot.w_0_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.inputs.1

def generatedRoot.w_1_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.inputs.2.1

def generatedRoot.w_2_0  (ctx : generatedRootCtx) : Fin 2 → mxxMat0 :=
  ctx.inputs.2.2.1

def generatedRoot.w_3_0  (ctx : generatedRootCtx) : Fin 2 → mxxMat0 :=
  ctx.inputs.2.2.2.1

def generatedRoot.w_4_0  (ctx : generatedRootCtx) : Fin 2 → Int :=
  ctx.inputs.2.2.2.2.1

def generatedRoot.w_5_0  (ctx : generatedRootCtx) : Fin 2 → Int :=
  ctx.inputs.2.2.2.2.2.1

def generatedRoot.w_7_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.1

def generatedRoot.w_8_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.1

def generatedRoot.w_9_0  (ctx : generatedRootCtx) : Fin 2 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.1

def generatedRoot.w_10_0  (ctx : generatedRootCtx) : Fin 2 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_12_0  (ctx : generatedRootCtx) : Fin 2 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_13_0  (ctx : generatedRootCtx) : Fin 2 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_15_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_16_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_18_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_19_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_21_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_22_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_24_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_25_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_27_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_28_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_30_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_31_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_33_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_34_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_36_0  (ctx : generatedRootCtx) : Fin 8 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_37_0  (ctx : generatedRootCtx) : Fin 8 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_39_0  (ctx : generatedRootCtx) : Fin 4 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_40_0  (ctx : generatedRootCtx) : Fin 4 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_6_0  (ctx : generatedRootCtx) : Fin 2 → mxxMat0 :=
  ctx.witness.w_6_0

def generatedRoot.w_6_1  (ctx : generatedRootCtx) : Fin 2 → mxxMat0 :=
  ctx.witness.w_6_1

def generatedRoot.w_6_2  (ctx : generatedRootCtx) : Fin 2 → mxxMat1 :=
  ctx.witness.w_6_2

def generatedRoot.w_11_0  (ctx : generatedRootCtx) : Fin 2 → mxxMat0 :=
  ctx.witness.w_11_0

def generatedRoot.w_11_1  (ctx : generatedRootCtx) : Fin 2 → mxxMat0 :=
  ctx.witness.w_11_1

def generatedRoot.w_11_2  (ctx : generatedRootCtx) : Fin 2 → mxxMat1 :=
  ctx.witness.w_11_2

def generatedRoot.w_14_0  (ctx : generatedRootCtx) : Fin 2 → mxxMat0 :=
  ctx.witness.w_14_0

def generatedRoot.w_14_1  (ctx : generatedRootCtx) : Fin 2 → mxxMat0 :=
  ctx.witness.w_14_1

def generatedRoot.w_14_2  (ctx : generatedRootCtx) : Fin 2 → mxxMat1 :=
  ctx.witness.w_14_2

def generatedRoot.w_17_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  (generatedRoot.w_15_0 ctx)

def generatedRoot.w_17_1  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  (generatedRoot.w_16_0 ctx)

def generatedRoot.w_20_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  (generatedRoot.w_18_0 ctx)

def generatedRoot.w_20_1  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  (generatedRoot.w_19_0 ctx)

def generatedRoot.w_23_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  (generatedRoot.w_21_0 ctx)

def generatedRoot.w_23_1  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  (generatedRoot.w_22_0 ctx)

def generatedRoot.w_26_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  (generatedRoot.w_24_0 ctx)

def generatedRoot.w_26_1  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  (generatedRoot.w_25_0 ctx)

def generatedRoot.w_29_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  (generatedRoot.w_27_0 ctx)

def generatedRoot.w_29_1  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  (generatedRoot.w_28_0 ctx)

def generatedRoot.w_32_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  (generatedRoot.w_30_0 ctx)

def generatedRoot.w_32_1  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  (generatedRoot.w_31_0 ctx)

def generatedRoot.w_35_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  (generatedRoot.w_33_0 ctx)

def generatedRoot.w_35_1  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  (generatedRoot.w_34_0 ctx)

def generatedRoot.w_38_0  (ctx : generatedRootCtx) : Fin 8 → mxxMat0 :=
  (generatedRoot.w_36_0 ctx)

def generatedRoot.w_38_1  (ctx : generatedRootCtx) : Fin 8 → mxxMat0 :=
  (generatedRoot.w_37_0 ctx)

def generatedRoot.w_41_0  (ctx : generatedRootCtx) : Fin 4 → mxxMat0 :=
  (generatedRoot.w_39_0 ctx)

def generatedRoot.w_41_1  (ctx : generatedRootCtx) : Fin 4 → mxxMat0 :=
  (generatedRoot.w_40_0 ctx)

def generatedRoot.w_42_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  (generatedRoot.w_15_0 ctx)

def generatedRoot.w_42_1  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  (generatedRoot.w_16_0 ctx)

def generatedRoot.w_43_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  (generatedRoot.w_18_0 ctx)

def generatedRoot.w_43_1  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  (generatedRoot.w_19_0 ctx)

def generatedRoot.w_44_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  (generatedRoot.w_21_0 ctx)

def generatedRoot.w_44_1  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  (generatedRoot.w_22_0 ctx)

def generatedRoot.w_45_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  (generatedRoot.w_24_0 ctx)

def generatedRoot.w_45_1  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  (generatedRoot.w_25_0 ctx)

def generatedRoot.w_46_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  (generatedRoot.w_27_0 ctx)

def generatedRoot.w_46_1  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  (generatedRoot.w_28_0 ctx)

def generatedRoot.w_47_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  (generatedRoot.w_30_0 ctx)

def generatedRoot.w_47_1  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  (generatedRoot.w_31_0 ctx)

def generatedRoot.w_48_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  (generatedRoot.w_33_0 ctx)

def generatedRoot.w_48_1  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  (generatedRoot.w_34_0 ctx)

def generatedRoot.w_49_0  (ctx : generatedRootCtx) : Fin 8 → mxxMat0 :=
  (generatedRoot.w_36_0 ctx)

def generatedRoot.w_49_1  (ctx : generatedRootCtx) : Fin 8 → mxxMat0 :=
  (generatedRoot.w_37_0 ctx)

def generatedRoot.w_50_0  (ctx : generatedRootCtx) : Fin 4 → mxxMat0 :=
  (generatedRoot.w_39_0 ctx)

def generatedRoot.w_50_1  (ctx : generatedRootCtx) : Fin 4 → mxxMat0 :=
  (generatedRoot.w_40_0 ctx)

def generatedRoot.w_51_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  (generatedRoot.w_15_0 ctx)

def generatedRoot.w_51_1  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  (generatedRoot.w_16_0 ctx)

def generatedRoot.w_52_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  (generatedRoot.w_18_0 ctx)

def generatedRoot.w_52_1  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  (generatedRoot.w_19_0 ctx)

def generatedRoot.w_53_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  (generatedRoot.w_21_0 ctx)

def generatedRoot.w_53_1  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  (generatedRoot.w_22_0 ctx)

def generatedRoot.w_54_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  (generatedRoot.w_24_0 ctx)

def generatedRoot.w_54_1  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  (generatedRoot.w_25_0 ctx)

def generatedRoot.w_55_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  (generatedRoot.w_27_0 ctx)

def generatedRoot.w_55_1  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  (generatedRoot.w_28_0 ctx)

def generatedRoot.w_56_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  (generatedRoot.w_30_0 ctx)

def generatedRoot.w_56_1  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  (generatedRoot.w_31_0 ctx)

def generatedRoot.w_57_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  (generatedRoot.w_33_0 ctx)

def generatedRoot.w_57_1  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  (generatedRoot.w_34_0 ctx)

def generatedRoot.w_58_0  (ctx : generatedRootCtx) : Fin 8 → mxxMat0 :=
  (generatedRoot.w_36_0 ctx)

def generatedRoot.w_58_1  (ctx : generatedRootCtx) : Fin 8 → mxxMat0 :=
  (generatedRoot.w_37_0 ctx)

def generatedRoot.w_59_0  (ctx : generatedRootCtx) : Fin 4 → mxxMat0 :=
  (generatedRoot.w_39_0 ctx)

def generatedRoot.w_59_1  (ctx : generatedRootCtx) : Fin 4 → mxxMat0 :=
  (generatedRoot.w_40_0 ctx)

def generatedRoot.w_60_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  (generatedRoot.w_15_0 ctx)

def generatedRoot.w_60_1  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  (generatedRoot.w_16_0 ctx)

def generatedRoot.w_61_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  (generatedRoot.w_18_0 ctx)

def generatedRoot.w_61_1  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  (generatedRoot.w_19_0 ctx)

def generatedRoot.w_62_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  (generatedRoot.w_21_0 ctx)

def generatedRoot.w_62_1  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  (generatedRoot.w_22_0 ctx)

def generatedRoot.w_63_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  (generatedRoot.w_24_0 ctx)

def generatedRoot.w_63_1  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  (generatedRoot.w_25_0 ctx)

def generatedRoot.w_64_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  (generatedRoot.w_27_0 ctx)

def generatedRoot.w_64_1  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  (generatedRoot.w_28_0 ctx)

def generatedRoot.w_65_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  (generatedRoot.w_30_0 ctx)

def generatedRoot.w_65_1  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  (generatedRoot.w_31_0 ctx)

def generatedRoot.w_66_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  (generatedRoot.w_33_0 ctx)

def generatedRoot.w_66_1  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  (generatedRoot.w_34_0 ctx)

def generatedRoot.w_67_0  (ctx : generatedRootCtx) : Fin 8 → mxxMat0 :=
  (generatedRoot.w_36_0 ctx)

def generatedRoot.w_67_1  (ctx : generatedRootCtx) : Fin 8 → mxxMat0 :=
  (generatedRoot.w_37_0 ctx)

def generatedRoot.w_68_0  (ctx : generatedRootCtx) : Fin 4 → mxxMat0 :=
  (generatedRoot.w_39_0 ctx)

def generatedRoot.w_68_1  (ctx : generatedRootCtx) : Fin 4 → mxxMat0 :=
  (generatedRoot.w_40_0 ctx)

def generatedRoot.w_69_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  (generatedRoot.w_15_0 ctx)

def generatedRoot.w_69_1  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  (generatedRoot.w_16_0 ctx)

def generatedRoot.w_70_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  (generatedRoot.w_18_0 ctx)

def generatedRoot.w_70_1  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  (generatedRoot.w_19_0 ctx)

def generatedRoot.w_71_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  (generatedRoot.w_21_0 ctx)

def generatedRoot.w_71_1  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  (generatedRoot.w_22_0 ctx)

def generatedRoot.w_72_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  (generatedRoot.w_24_0 ctx)

def generatedRoot.w_72_1  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  (generatedRoot.w_25_0 ctx)

def generatedRoot.w_73_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  (generatedRoot.w_27_0 ctx)

def generatedRoot.w_73_1  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  (generatedRoot.w_28_0 ctx)

def generatedRoot.w_74_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  (generatedRoot.w_30_0 ctx)

def generatedRoot.w_74_1  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  (generatedRoot.w_31_0 ctx)

def generatedRoot.w_75_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  (generatedRoot.w_33_0 ctx)

def generatedRoot.w_75_1  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  (generatedRoot.w_34_0 ctx)

def generatedRoot.w_76_0  (ctx : generatedRootCtx) : Fin 8 → mxxMat0 :=
  (generatedRoot.w_36_0 ctx)

def generatedRoot.w_76_1  (ctx : generatedRootCtx) : Fin 8 → mxxMat0 :=
  (generatedRoot.w_37_0 ctx)

def generatedRoot.w_77_0  (ctx : generatedRootCtx) : Fin 4 → mxxMat0 :=
  (generatedRoot.w_39_0 ctx)

def generatedRoot.w_77_1  (ctx : generatedRootCtx) : Fin 4 → mxxMat0 :=
  (generatedRoot.w_40_0 ctx)

def generatedRoot.outs  (ctx : generatedRootCtx) : (generatedRoot.OutputType) :=
  ((generatedRoot.w_0_0 ctx), ((generatedRoot.w_1_0 ctx), ((generatedRoot.w_6_0 ctx), ((generatedRoot.w_6_1 ctx), ((generatedRoot.w_6_2 ctx), ((generatedRoot.w_7_0 ctx), ((generatedRoot.w_8_0 ctx), ((generatedRoot.w_0_0 ctx), ((generatedRoot.w_1_0 ctx), ((generatedRoot.w_11_0 ctx), ((generatedRoot.w_11_1 ctx), ((generatedRoot.w_11_2 ctx), ((generatedRoot.w_0_0 ctx), ((generatedRoot.w_1_0 ctx), ((generatedRoot.w_14_0 ctx), ((generatedRoot.w_14_1 ctx), ((generatedRoot.w_14_2 ctx), ((generatedRoot.w_17_0 ctx), ((generatedRoot.w_17_1 ctx), ((generatedRoot.w_20_0 ctx), ((generatedRoot.w_20_1 ctx), ((generatedRoot.w_23_0 ctx), ((generatedRoot.w_23_1 ctx), ((generatedRoot.w_26_0 ctx), ((generatedRoot.w_26_1 ctx), ((generatedRoot.w_29_0 ctx), ((generatedRoot.w_29_1 ctx), ((generatedRoot.w_32_0 ctx), ((generatedRoot.w_32_1 ctx), ((generatedRoot.w_35_0 ctx), ((generatedRoot.w_35_1 ctx), ((generatedRoot.w_38_0 ctx), ((generatedRoot.w_38_1 ctx), ((generatedRoot.w_41_0 ctx), ((generatedRoot.w_41_1 ctx), ((generatedRoot.w_42_0 ctx), ((generatedRoot.w_42_1 ctx), ((generatedRoot.w_43_0 ctx), ((generatedRoot.w_43_1 ctx), ((generatedRoot.w_44_0 ctx), ((generatedRoot.w_44_1 ctx), ((generatedRoot.w_45_0 ctx), ((generatedRoot.w_45_1 ctx), ((generatedRoot.w_46_0 ctx), ((generatedRoot.w_46_1 ctx), ((generatedRoot.w_47_0 ctx), ((generatedRoot.w_47_1 ctx), ((generatedRoot.w_48_0 ctx), ((generatedRoot.w_48_1 ctx), ((generatedRoot.w_49_0 ctx), ((generatedRoot.w_49_1 ctx), ((generatedRoot.w_50_0 ctx), ((generatedRoot.w_50_1 ctx), ((generatedRoot.w_51_0 ctx), ((generatedRoot.w_51_1 ctx), ((generatedRoot.w_52_0 ctx), ((generatedRoot.w_52_1 ctx), ((generatedRoot.w_53_0 ctx), ((generatedRoot.w_53_1 ctx), ((generatedRoot.w_54_0 ctx), ((generatedRoot.w_54_1 ctx), ((generatedRoot.w_55_0 ctx), ((generatedRoot.w_55_1 ctx), ((generatedRoot.w_56_0 ctx), ((generatedRoot.w_56_1 ctx), ((generatedRoot.w_57_0 ctx), ((generatedRoot.w_57_1 ctx), ((generatedRoot.w_58_0 ctx), ((generatedRoot.w_58_1 ctx), ((generatedRoot.w_59_0 ctx), ((generatedRoot.w_59_1 ctx), ((generatedRoot.w_60_0 ctx), ((generatedRoot.w_60_1 ctx), ((generatedRoot.w_61_0 ctx), ((generatedRoot.w_61_1 ctx), ((generatedRoot.w_62_0 ctx), ((generatedRoot.w_62_1 ctx), ((generatedRoot.w_63_0 ctx), ((generatedRoot.w_63_1 ctx), ((generatedRoot.w_64_0 ctx), ((generatedRoot.w_64_1 ctx), ((generatedRoot.w_65_0 ctx), ((generatedRoot.w_65_1 ctx), ((generatedRoot.w_66_0 ctx), ((generatedRoot.w_66_1 ctx), ((generatedRoot.w_67_0 ctx), ((generatedRoot.w_67_1 ctx), ((generatedRoot.w_68_0 ctx), ((generatedRoot.w_68_1 ctx), ((generatedRoot.w_69_0 ctx), ((generatedRoot.w_69_1 ctx), ((generatedRoot.w_70_0 ctx), ((generatedRoot.w_70_1 ctx), ((generatedRoot.w_71_0 ctx), ((generatedRoot.w_71_1 ctx), ((generatedRoot.w_72_0 ctx), ((generatedRoot.w_72_1 ctx), ((generatedRoot.w_73_0 ctx), ((generatedRoot.w_73_1 ctx), ((generatedRoot.w_74_0 ctx), ((generatedRoot.w_74_1 ctx), ((generatedRoot.w_75_0 ctx), ((generatedRoot.w_75_1 ctx), ((generatedRoot.w_76_0 ctx), ((generatedRoot.w_76_1 ctx), ((generatedRoot.w_77_0 ctx), ((generatedRoot.w_77_1 ctx), ())))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))

structure generatedRoot.Facts_0  (ctx : generatedRootCtx) : Prop where
  t0 : (2) = 2
  h6_2 : (∀ i : Fin 2, parallel_generatedRoot_6 tape (path ++ [6, i.val]) params i (((generatedRoot.w_2_0 ctx) i), (((generatedRoot.w_3_0 ctx) i), (((generatedRoot.w_4_0 ctx) i), (((generatedRoot.w_5_0 ctx) i), ())))) (((generatedRoot.w_6_0 ctx) i), (((generatedRoot.w_6_1 ctx) i), (((generatedRoot.w_6_2 ctx) i), ()))))
  t2 : (2) = 2
  h11_2 : (∀ i : Fin 2, parallel_generatedRoot_11 tape (path ++ [11, i.val]) params i (((generatedRoot.w_2_0 ctx) i), (((generatedRoot.w_3_0 ctx) i), (((generatedRoot.w_9_0 ctx) i), (((generatedRoot.w_10_0 ctx) i), ())))) (((generatedRoot.w_11_0 ctx) i), (((generatedRoot.w_11_1 ctx) i), (((generatedRoot.w_11_2 ctx) i), ()))))
  t4 : (2) = 2
  h14_2 : (∀ i : Fin 2, parallel_generatedRoot_14 tape (path ++ [14, i.val]) params i (((generatedRoot.w_2_0 ctx) i), (((generatedRoot.w_3_0 ctx) i), (((generatedRoot.w_12_0 ctx) i), (((generatedRoot.w_13_0 ctx) i), ())))) (((generatedRoot.w_14_0 ctx) i), (((generatedRoot.w_14_1 ctx) i), (((generatedRoot.w_14_2 ctx) i), ()))))
  t6 : (1) = 1
  t7 : (1) = 1
  t8 : (1) = 1
  t9 : (1) = 1
  t10 : (1) = 1
  t11 : (1) = 1
  t12 : (1) = 1
  t13 : (8) = 8
  t14 : (4) = 4
  t15 : (1) = 1
  t16 : (1) = 1
  t17 : (1) = 1
  t18 : (1) = 1
  t19 : (1) = 1
  t20 : (1) = 1
  t21 : (1) = 1
  t22 : (8) = 8
  t23 : (4) = 4
  t24 : (1) = 1
  t25 : (1) = 1
  t26 : (1) = 1
  t27 : (1) = 1
  t28 : (1) = 1
  t29 : (1) = 1
  t30 : (1) = 1
  t31 : (8) = 8

theorem generatedRoot.facts_0  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_0 tape path params (generatedRoot.w_2_0 ctx) (generatedRoot.w_3_0 ctx) (generatedRoot.w_4_0 ctx) (generatedRoot.w_5_0 ctx) (generatedRoot.w_9_0 ctx) (generatedRoot.w_10_0 ctx) (generatedRoot.w_12_0 ctx) (generatedRoot.w_13_0 ctx) (generatedRoot.w_6_0 ctx) (generatedRoot.w_6_1 ctx) (generatedRoot.w_6_2 ctx) (generatedRoot.w_11_0 ctx) (generatedRoot.w_11_1 ctx) (generatedRoot.w_11_2 ctx) (generatedRoot.w_14_0 ctx) (generatedRoot.w_14_1 ctx) (generatedRoot.w_14_2 ctx)) :
    generatedRoot.Facts_0 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_1  (ctx : generatedRootCtx) (outputs : (generatedRoot.OutputType)) : Prop where
  t32 : (4) = 4
  t33 : (1) = 1
  t34 : (1) = 1
  t35 : (1) = 1
  t36 : (1) = 1
  t37 : (1) = 1
  t38 : (1) = 1
  t39 : (1) = 1
  t40 : (8) = 8
  t41 : (4) = 4
  t42 : (1) = 1
  t43 : (1) = 1
  t44 : (1) = 1
  t45 : (1) = 1
  t46 : (1) = 1
  t47 : (1) = 1
  t48 : (1) = 1
  t49 : (8) = 8
  t50 : (4) = 4
  hout : outputs = generatedRoot.outs ctx

theorem generatedRoot.facts_1  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_1 (generatedRoot.w_0_0 ctx) (generatedRoot.w_1_0 ctx) (generatedRoot.w_7_0 ctx) (generatedRoot.w_8_0 ctx) (generatedRoot.w_6_0 ctx) (generatedRoot.w_6_1 ctx) (generatedRoot.w_6_2 ctx) (generatedRoot.w_11_0 ctx) (generatedRoot.w_11_1 ctx) (generatedRoot.w_11_2 ctx) (generatedRoot.w_14_0 ctx) (generatedRoot.w_14_1 ctx) (generatedRoot.w_14_2 ctx) (generatedRoot.w_17_0 ctx) (generatedRoot.w_17_1 ctx) (generatedRoot.w_20_0 ctx) (generatedRoot.w_20_1 ctx) (generatedRoot.w_23_0 ctx) (generatedRoot.w_23_1 ctx) (generatedRoot.w_26_0 ctx) (generatedRoot.w_26_1 ctx) (generatedRoot.w_29_0 ctx) (generatedRoot.w_29_1 ctx) (generatedRoot.w_32_0 ctx) (generatedRoot.w_32_1 ctx) (generatedRoot.w_35_0 ctx) (generatedRoot.w_35_1 ctx) (generatedRoot.w_38_0 ctx) (generatedRoot.w_38_1 ctx) (generatedRoot.w_41_0 ctx) (generatedRoot.w_41_1 ctx) (generatedRoot.w_42_0 ctx) (generatedRoot.w_42_1 ctx) (generatedRoot.w_43_0 ctx) (generatedRoot.w_43_1 ctx) (generatedRoot.w_44_0 ctx) (generatedRoot.w_44_1 ctx) (generatedRoot.w_45_0 ctx) (generatedRoot.w_45_1 ctx) (generatedRoot.w_46_0 ctx) (generatedRoot.w_46_1 ctx) (generatedRoot.w_47_0 ctx) (generatedRoot.w_47_1 ctx) (generatedRoot.w_48_0 ctx) (generatedRoot.w_48_1 ctx) (generatedRoot.w_49_0 ctx) (generatedRoot.w_49_1 ctx) (generatedRoot.w_50_0 ctx) (generatedRoot.w_50_1 ctx) (generatedRoot.w_51_0 ctx) (generatedRoot.w_51_1 ctx) (generatedRoot.w_52_0 ctx) (generatedRoot.w_52_1 ctx) (generatedRoot.w_53_0 ctx) (generatedRoot.w_53_1 ctx) (generatedRoot.w_54_0 ctx) (generatedRoot.w_54_1 ctx) (generatedRoot.w_55_0 ctx) (generatedRoot.w_55_1 ctx) (generatedRoot.w_56_0 ctx) (generatedRoot.w_56_1 ctx) (generatedRoot.w_57_0 ctx) (generatedRoot.w_57_1 ctx) (generatedRoot.w_58_0 ctx) (generatedRoot.w_58_1 ctx) (generatedRoot.w_59_0 ctx) (generatedRoot.w_59_1 ctx) (generatedRoot.w_60_0 ctx) (generatedRoot.w_60_1 ctx) (generatedRoot.w_61_0 ctx) (generatedRoot.w_61_1 ctx) (generatedRoot.w_62_0 ctx) (generatedRoot.w_62_1 ctx) (generatedRoot.w_63_0 ctx) (generatedRoot.w_63_1 ctx) (generatedRoot.w_64_0 ctx) (generatedRoot.w_64_1 ctx) (generatedRoot.w_65_0 ctx) (generatedRoot.w_65_1 ctx) (generatedRoot.w_66_0 ctx) (generatedRoot.w_66_1 ctx) (generatedRoot.w_67_0 ctx) (generatedRoot.w_67_1 ctx) (generatedRoot.w_68_0 ctx) (generatedRoot.w_68_1 ctx) (generatedRoot.w_69_0 ctx) (generatedRoot.w_69_1 ctx) (generatedRoot.w_70_0 ctx) (generatedRoot.w_70_1 ctx) (generatedRoot.w_71_0 ctx) (generatedRoot.w_71_1 ctx) (generatedRoot.w_72_0 ctx) (generatedRoot.w_72_1 ctx) (generatedRoot.w_73_0 ctx) (generatedRoot.w_73_1 ctx) (generatedRoot.w_74_0 ctx) (generatedRoot.w_74_1 ctx) (generatedRoot.w_75_0 ctx) (generatedRoot.w_75_1 ctx) (generatedRoot.w_76_0 ctx) (generatedRoot.w_76_1 ctx) (generatedRoot.w_77_0 ctx) (generatedRoot.w_77_1 ctx) outputs) :
    generatedRoot.Facts_1 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts  (ctx : generatedRootCtx) (outputs : (generatedRoot.OutputType)) : Prop where
  c0 : generatedRoot.Facts_0 ctx
  c1 : generatedRoot.Facts_1 ctx outputs

theorem generatedRoot.facts  (ctx : generatedRootCtx) {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.body tape path params ctx.inputs outputs ctx.witness) :
    generatedRoot.Facts ctx outputs :=
  have t0 := h
  have r0 := generatedRoot.facts_0 (ctx := ctx) (outputs := outputs) t0.1
  have t1 := t0.2
  have r1 := generatedRoot.facts_1 (ctx := ctx) (outputs := outputs) t1
  ⟨r0, r1⟩

end

end Stage_keys_import_0_0

end
