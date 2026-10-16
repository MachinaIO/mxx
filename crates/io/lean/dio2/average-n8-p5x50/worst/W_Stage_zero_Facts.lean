import W_Stage_zero

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000
set_option linter.unusedVariables false

-- The wire types repeat the ring modulus `X ^ n + 1`; unfolding it while checking a
-- projection against its field type expands it at the ring dimension.
attribute [local irreducible] Mxx.Primitives.negacyclicModulus

noncomputable section

namespace W_Stage_zero

local notation "mxxMat0" => Mxx.Primitives.ExactMatrix modulus_0 8 1 54
local notation "mxxMat1" => Mxx.Primitives.ExactMatrix modulus_0 8 1 25

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Ctx (params : Params) where
  inputs : Unit
  witness : generatedRoot.Witness

def generatedRoot.w_0_0 {params : Params} (ctx : generatedRoot.Ctx params) : mxxMat0 :=
  (0 : mxxMat0)

def generatedRoot.w_1_0 {params : Params} (ctx : generatedRoot.Ctx params) : mxxMat1 :=
  (0 : mxxMat1)

def generatedRoot.w_2_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 1 → mxxMat1 :=
  ctx.witness.w_2_0

def generatedRoot.w_3_0 {params : Params} (ctx : generatedRoot.Ctx params) : mxxMat1 :=
  (0 : mxxMat1)

def generatedRoot.w_4_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 1 → mxxMat1 :=
  ctx.witness.w_4_0

def generatedRoot.w_5_0 {params : Params} (ctx : generatedRoot.Ctx params) : mxxMat1 :=
  (0 : mxxMat1)

def generatedRoot.w_6_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 1 → mxxMat1 :=
  ctx.witness.w_6_0

def generatedRoot.w_7_0 {params : Params} (ctx : generatedRoot.Ctx params) : mxxMat1 :=
  (0 : mxxMat1)

def generatedRoot.w_8_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 1 → mxxMat1 :=
  ctx.witness.w_8_0

def generatedRoot.w_9_0 {params : Params} (ctx : generatedRoot.Ctx params) : mxxMat1 :=
  (0 : mxxMat1)

def generatedRoot.w_10_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 1 → mxxMat1 :=
  ctx.witness.w_10_0

def generatedRoot.w_11_0 {params : Params} (ctx : generatedRoot.Ctx params) : mxxMat1 :=
  (0 : mxxMat1)

def generatedRoot.w_12_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 1 → mxxMat1 :=
  ctx.witness.w_12_0

def generatedRoot.w_13_0 {params : Params} (ctx : generatedRoot.Ctx params) : mxxMat1 :=
  (0 : mxxMat1)

def generatedRoot.w_14_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 1 → mxxMat1 :=
  ctx.witness.w_14_0

def generatedRoot.w_15_0 {params : Params} (ctx : generatedRoot.Ctx params) : mxxMat1 :=
  (0 : mxxMat1)

def generatedRoot.w_16_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 1 → mxxMat1 :=
  ctx.witness.w_16_0

def generatedRoot.w_17_0 {params : Params} (ctx : generatedRoot.Ctx params) : mxxMat1 :=
  (0 : mxxMat1)

def generatedRoot.w_18_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 1 → mxxMat1 :=
  ctx.witness.w_18_0

def generatedRoot.w_19_0 {params : Params} (ctx : generatedRoot.Ctx params) : mxxMat1 :=
  (0 : mxxMat1)

def generatedRoot.w_20_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 1 → mxxMat1 :=
  ctx.witness.w_20_0

def generatedRoot.w_21_0 {params : Params} (ctx : generatedRoot.Ctx params) : mxxMat1 :=
  (0 : mxxMat1)

def generatedRoot.w_22_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 1 → mxxMat1 :=
  ctx.witness.w_22_0

def generatedRoot.w_23_0 {params : Params} (ctx : generatedRoot.Ctx params) : mxxMat1 :=
  (0 : mxxMat1)

def generatedRoot.w_24_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 1 → mxxMat1 :=
  ctx.witness.w_24_0

def generatedRoot.w_25_0 {params : Params} (ctx : generatedRoot.Ctx params) : mxxMat1 :=
  (0 : mxxMat1)

def generatedRoot.w_26_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 1 → mxxMat1 :=
  ctx.witness.w_26_0

def generatedRoot.w_27_0 {params : Params} (ctx : generatedRoot.Ctx params) : mxxMat1 :=
  (0 : mxxMat1)

def generatedRoot.w_28_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 1 → mxxMat1 :=
  ctx.witness.w_28_0

def generatedRoot.w_29_0 {params : Params} (ctx : generatedRoot.Ctx params) : mxxMat1 :=
  (0 : mxxMat1)

def generatedRoot.w_30_0 {params : Params} (ctx : generatedRoot.Ctx params) : mxxMat1 :=
  (0 : mxxMat1)

def generatedRoot.w_31_0 {params : Params} (ctx : generatedRoot.Ctx params) : mxxMat1 :=
  (0 : mxxMat1)

def generatedRoot.w_32_0 {params : Params} (ctx : generatedRoot.Ctx params) : mxxMat1 :=
  (0 : mxxMat1)

def generatedRoot.w_33_0 {params : Params} (ctx : generatedRoot.Ctx params) : mxxMat1 :=
  (0 : mxxMat1)

def generatedRoot.w_34_0 {params : Params} (ctx : generatedRoot.Ctx params) : mxxMat1 :=
  (0 : mxxMat1)

def generatedRoot.w_35_0 {params : Params} (ctx : generatedRoot.Ctx params) : mxxMat1 :=
  (0 : mxxMat1)

def generatedRoot.w_36_0 {params : Params} (ctx : generatedRoot.Ctx params) : mxxMat1 :=
  (0 : mxxMat1)

def generatedRoot.w_37_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 8 → mxxMat1 :=
  ctx.witness.w_37_0

def generatedRoot.w_38_0 {params : Params} (ctx : generatedRoot.Ctx params) : mxxMat1 :=
  (0 : mxxMat1)

def generatedRoot.w_39_0 {params : Params} (ctx : generatedRoot.Ctx params) : mxxMat1 :=
  (0 : mxxMat1)

def generatedRoot.w_40_0 {params : Params} (ctx : generatedRoot.Ctx params) : mxxMat1 :=
  (0 : mxxMat1)

def generatedRoot.w_41_0 {params : Params} (ctx : generatedRoot.Ctx params) : mxxMat1 :=
  (0 : mxxMat1)

def generatedRoot.w_42_0 {params : Params} (ctx : generatedRoot.Ctx params) : mxxMat1 :=
  (0 : mxxMat1)

def generatedRoot.w_43_0 {params : Params} (ctx : generatedRoot.Ctx params) : mxxMat1 :=
  (0 : mxxMat1)

def generatedRoot.w_44_0 {params : Params} (ctx : generatedRoot.Ctx params) : mxxMat1 :=
  (0 : mxxMat1)

def generatedRoot.w_45_0 {params : Params} (ctx : generatedRoot.Ctx params) : mxxMat1 :=
  (0 : mxxMat1)

def generatedRoot.w_46_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 8 → mxxMat1 :=
  ctx.witness.w_46_0

def generatedRoot.w_47_0 {params : Params} (ctx : generatedRoot.Ctx params) : mxxMat1 :=
  (0 : mxxMat1)

def generatedRoot.w_48_0 {params : Params} (ctx : generatedRoot.Ctx params) : mxxMat1 :=
  (0 : mxxMat1)

def generatedRoot.w_49_0 {params : Params} (ctx : generatedRoot.Ctx params) : mxxMat1 :=
  (0 : mxxMat1)

def generatedRoot.w_50_0 {params : Params} (ctx : generatedRoot.Ctx params) : mxxMat1 :=
  (0 : mxxMat1)

def generatedRoot.w_51_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 4 → mxxMat1 :=
  ctx.witness.w_51_0

def generatedRoot.w_52_0 {params : Params} (ctx : generatedRoot.Ctx params) : mxxMat1 :=
  (0 : mxxMat1)

def generatedRoot.w_53_0 {params : Params} (ctx : generatedRoot.Ctx params) : mxxMat1 :=
  (0 : mxxMat1)

def generatedRoot.w_54_0 {params : Params} (ctx : generatedRoot.Ctx params) : mxxMat1 :=
  (0 : mxxMat1)

def generatedRoot.w_55_0 {params : Params} (ctx : generatedRoot.Ctx params) : mxxMat1 :=
  (0 : mxxMat1)

def generatedRoot.w_56_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 4 → mxxMat1 :=
  ctx.witness.w_56_0

def generatedRoot.w_57_0 {params : Params} (ctx : generatedRoot.Ctx params) : mxxMat1 :=
  (0 : mxxMat1)

def generatedRoot.w_58_0 {params : Params} (ctx : generatedRoot.Ctx params) : mxxMat1 :=
  (0 : mxxMat1)

def generatedRoot.w_59_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 2 → mxxMat1 :=
  ctx.witness.w_59_0

def generatedRoot.w_60_0 {params : Params} (ctx : generatedRoot.Ctx params) : mxxMat1 :=
  (0 : mxxMat1)

def generatedRoot.w_61_0 {params : Params} (ctx : generatedRoot.Ctx params) : mxxMat1 :=
  (0 : mxxMat1)

def generatedRoot.w_62_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 2 → mxxMat1 :=
  ctx.witness.w_62_0

def generatedRoot.w_63_0 {params : Params} (ctx : generatedRoot.Ctx params) : mxxMat1 :=
  (0 : mxxMat1)

def generatedRoot.w_64_0 {params : Params} (ctx : generatedRoot.Ctx params) : mxxMat1 :=
  (0 : mxxMat1)

def generatedRoot.w_65_0 {params : Params} (ctx : generatedRoot.Ctx params) : mxxMat1 :=
  (0 : mxxMat1)

def generatedRoot.w_66_0 {params : Params} (ctx : generatedRoot.Ctx params) : mxxMat1 :=
  (0 : mxxMat1)

def generatedRoot.w_67_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 4 → mxxMat1 :=
  ctx.witness.w_67_0

def generatedRoot.w_68_0 {params : Params} (ctx : generatedRoot.Ctx params) : mxxMat1 :=
  (0 : mxxMat1)

def generatedRoot.w_69_0 {params : Params} (ctx : generatedRoot.Ctx params) : mxxMat1 :=
  (0 : mxxMat1)

def generatedRoot.w_70_0 {params : Params} (ctx : generatedRoot.Ctx params) : mxxMat1 :=
  (0 : mxxMat1)

def generatedRoot.w_71_0 {params : Params} (ctx : generatedRoot.Ctx params) : mxxMat1 :=
  (0 : mxxMat1)

def generatedRoot.w_72_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 4 → mxxMat1 :=
  ctx.witness.w_72_0

def generatedRoot.w_73_0 {params : Params} (ctx : generatedRoot.Ctx params) : mxxMat1 :=
  (0 : mxxMat1)

def generatedRoot.w_74_0 {params : Params} (ctx : generatedRoot.Ctx params) : mxxMat1 :=
  (0 : mxxMat1)

def generatedRoot.w_75_0 {params : Params} (ctx : generatedRoot.Ctx params) : mxxMat1 :=
  (0 : mxxMat1)

def generatedRoot.w_76_0 {params : Params} (ctx : generatedRoot.Ctx params) : mxxMat1 :=
  (0 : mxxMat1)

def generatedRoot.outs {params : Params} (ctx : generatedRoot.Ctx params) : mxxMat0 × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 2 → mxxMat1) × (Fin 2 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit :=
  ((generatedRoot.w_0_0 ctx), ((generatedRoot.w_2_0 ctx), ((generatedRoot.w_4_0 ctx), ((generatedRoot.w_6_0 ctx), ((generatedRoot.w_8_0 ctx), ((generatedRoot.w_10_0 ctx), ((generatedRoot.w_12_0 ctx), ((generatedRoot.w_14_0 ctx), ((generatedRoot.w_16_0 ctx), ((generatedRoot.w_18_0 ctx), ((generatedRoot.w_20_0 ctx), ((generatedRoot.w_22_0 ctx), ((generatedRoot.w_24_0 ctx), ((generatedRoot.w_26_0 ctx), ((generatedRoot.w_28_0 ctx), ((generatedRoot.w_37_0 ctx), ((generatedRoot.w_46_0 ctx), ((generatedRoot.w_51_0 ctx), ((generatedRoot.w_56_0 ctx), ((generatedRoot.w_59_0 ctx), ((generatedRoot.w_62_0 ctx), ((generatedRoot.w_67_0 ctx), ((generatedRoot.w_72_0 ctx), ((generatedRoot.w_73_0 ctx), ((generatedRoot.w_74_0 ctx), ((generatedRoot.w_75_0 ctx), ((generatedRoot.w_76_0 ctx), ())))))))))))))))))))))))))))

structure generatedRoot.Facts_0 {params : Params} (ctx : generatedRoot.Ctx params) (outputs : mxxMat0 × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 2 → mxxMat1) × (Fin 2 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  h2_0 : MxxRuntime.familyPack (1) [(generatedRoot.w_1_0 ctx)] (generatedRoot.w_2_0 ctx)
  h4_0 : MxxRuntime.familyPack (1) [(generatedRoot.w_3_0 ctx)] (generatedRoot.w_4_0 ctx)
  h6_0 : MxxRuntime.familyPack (1) [(generatedRoot.w_5_0 ctx)] (generatedRoot.w_6_0 ctx)
  h8_0 : MxxRuntime.familyPack (1) [(generatedRoot.w_7_0 ctx)] (generatedRoot.w_8_0 ctx)
  h10_0 : MxxRuntime.familyPack (1) [(generatedRoot.w_9_0 ctx)] (generatedRoot.w_10_0 ctx)
  h12_0 : MxxRuntime.familyPack (1) [(generatedRoot.w_11_0 ctx)] (generatedRoot.w_12_0 ctx)
  h14_0 : MxxRuntime.familyPack (1) [(generatedRoot.w_13_0 ctx)] (generatedRoot.w_14_0 ctx)
  h16_0 : MxxRuntime.familyPack (1) [(generatedRoot.w_15_0 ctx)] (generatedRoot.w_16_0 ctx)
  h18_0 : MxxRuntime.familyPack (1) [(generatedRoot.w_17_0 ctx)] (generatedRoot.w_18_0 ctx)
  h20_0 : MxxRuntime.familyPack (1) [(generatedRoot.w_19_0 ctx)] (generatedRoot.w_20_0 ctx)
  h22_0 : MxxRuntime.familyPack (1) [(generatedRoot.w_21_0 ctx)] (generatedRoot.w_22_0 ctx)
  h24_0 : MxxRuntime.familyPack (1) [(generatedRoot.w_23_0 ctx)] (generatedRoot.w_24_0 ctx)
  h26_0 : MxxRuntime.familyPack (1) [(generatedRoot.w_25_0 ctx)] (generatedRoot.w_26_0 ctx)
  h28_0 : MxxRuntime.familyPack (1) [(generatedRoot.w_27_0 ctx)] (generatedRoot.w_28_0 ctx)
  h37_0 : MxxRuntime.familyPack (8) [(generatedRoot.w_29_0 ctx), (generatedRoot.w_30_0 ctx), (generatedRoot.w_31_0 ctx), (generatedRoot.w_32_0 ctx), (generatedRoot.w_33_0 ctx), (generatedRoot.w_34_0 ctx), (generatedRoot.w_35_0 ctx), (generatedRoot.w_36_0 ctx)] (generatedRoot.w_37_0 ctx)
  h46_0 : MxxRuntime.familyPack (8) [(generatedRoot.w_38_0 ctx), (generatedRoot.w_39_0 ctx), (generatedRoot.w_40_0 ctx), (generatedRoot.w_41_0 ctx), (generatedRoot.w_42_0 ctx), (generatedRoot.w_43_0 ctx), (generatedRoot.w_44_0 ctx), (generatedRoot.w_45_0 ctx)] (generatedRoot.w_46_0 ctx)
  h51_0 : MxxRuntime.familyPack (4) [(generatedRoot.w_47_0 ctx), (generatedRoot.w_48_0 ctx), (generatedRoot.w_49_0 ctx), (generatedRoot.w_50_0 ctx)] (generatedRoot.w_51_0 ctx)
  h56_0 : MxxRuntime.familyPack (4) [(generatedRoot.w_52_0 ctx), (generatedRoot.w_53_0 ctx), (generatedRoot.w_54_0 ctx), (generatedRoot.w_55_0 ctx)] (generatedRoot.w_56_0 ctx)
  h59_0 : MxxRuntime.familyPack (2) [(generatedRoot.w_57_0 ctx), (generatedRoot.w_58_0 ctx)] (generatedRoot.w_59_0 ctx)
  h62_0 : MxxRuntime.familyPack (2) [(generatedRoot.w_60_0 ctx), (generatedRoot.w_61_0 ctx)] (generatedRoot.w_62_0 ctx)
  h67_0 : MxxRuntime.familyPack (4) [(generatedRoot.w_63_0 ctx), (generatedRoot.w_64_0 ctx), (generatedRoot.w_65_0 ctx), (generatedRoot.w_66_0 ctx)] (generatedRoot.w_67_0 ctx)
  h72_0 : MxxRuntime.familyPack (4) [(generatedRoot.w_68_0 ctx), (generatedRoot.w_69_0 ctx), (generatedRoot.w_70_0 ctx), (generatedRoot.w_71_0 ctx)] (generatedRoot.w_72_0 ctx)
  hout : outputs = generatedRoot.outs ctx

theorem generatedRoot.facts_0 {params : Params} {ctx : generatedRoot.Ctx params} {outputs : mxxMat0 × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 2 → mxxMat1) × (Fin 2 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : generatedRoot.constraints_0 (generatedRoot.w_0_0 ctx) (generatedRoot.w_1_0 ctx) (generatedRoot.w_2_0 ctx) (generatedRoot.w_3_0 ctx) (generatedRoot.w_4_0 ctx) (generatedRoot.w_5_0 ctx) (generatedRoot.w_6_0 ctx) (generatedRoot.w_7_0 ctx) (generatedRoot.w_8_0 ctx) (generatedRoot.w_9_0 ctx) (generatedRoot.w_10_0 ctx) (generatedRoot.w_11_0 ctx) (generatedRoot.w_12_0 ctx) (generatedRoot.w_13_0 ctx) (generatedRoot.w_14_0 ctx) (generatedRoot.w_15_0 ctx) (generatedRoot.w_16_0 ctx) (generatedRoot.w_17_0 ctx) (generatedRoot.w_18_0 ctx) (generatedRoot.w_19_0 ctx) (generatedRoot.w_20_0 ctx) (generatedRoot.w_21_0 ctx) (generatedRoot.w_22_0 ctx) (generatedRoot.w_23_0 ctx) (generatedRoot.w_24_0 ctx) (generatedRoot.w_25_0 ctx) (generatedRoot.w_26_0 ctx) (generatedRoot.w_27_0 ctx) (generatedRoot.w_28_0 ctx) (generatedRoot.w_29_0 ctx) (generatedRoot.w_30_0 ctx) (generatedRoot.w_31_0 ctx) (generatedRoot.w_32_0 ctx) (generatedRoot.w_33_0 ctx) (generatedRoot.w_34_0 ctx) (generatedRoot.w_35_0 ctx) (generatedRoot.w_36_0 ctx) (generatedRoot.w_37_0 ctx) (generatedRoot.w_38_0 ctx) (generatedRoot.w_39_0 ctx) (generatedRoot.w_40_0 ctx) (generatedRoot.w_41_0 ctx) (generatedRoot.w_42_0 ctx) (generatedRoot.w_43_0 ctx) (generatedRoot.w_44_0 ctx) (generatedRoot.w_45_0 ctx) (generatedRoot.w_46_0 ctx) (generatedRoot.w_47_0 ctx) (generatedRoot.w_48_0 ctx) (generatedRoot.w_49_0 ctx) (generatedRoot.w_50_0 ctx) (generatedRoot.w_51_0 ctx) (generatedRoot.w_52_0 ctx) (generatedRoot.w_53_0 ctx) (generatedRoot.w_54_0 ctx) (generatedRoot.w_55_0 ctx) (generatedRoot.w_56_0 ctx) (generatedRoot.w_57_0 ctx) (generatedRoot.w_58_0 ctx) (generatedRoot.w_59_0 ctx) (generatedRoot.w_60_0 ctx) (generatedRoot.w_61_0 ctx) (generatedRoot.w_62_0 ctx) (generatedRoot.w_63_0 ctx) (generatedRoot.w_64_0 ctx) (generatedRoot.w_65_0 ctx) (generatedRoot.w_66_0 ctx) (generatedRoot.w_67_0 ctx) (generatedRoot.w_68_0 ctx) (generatedRoot.w_69_0 ctx) (generatedRoot.w_70_0 ctx) (generatedRoot.w_71_0 ctx) (generatedRoot.w_72_0 ctx) (generatedRoot.w_73_0 ctx) (generatedRoot.w_74_0 ctx) (generatedRoot.w_75_0 ctx) (generatedRoot.w_76_0 ctx) outputs) :
    generatedRoot.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts {params : Params} (ctx : generatedRoot.Ctx params) (outputs : mxxMat0 × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 2 → mxxMat1) × (Fin 2 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  c0 : generatedRoot.Facts_0 ctx outputs

theorem generatedRoot.facts {params : Params} (ctx : generatedRoot.Ctx params) {outputs : mxxMat0 × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 2 → mxxMat1) × (Fin 2 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : generatedRoot.body params ctx.inputs outputs ctx.witness) :
    generatedRoot.Facts ctx outputs :=
  have t0 := h
  have r0 := generatedRoot.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end W_Stage_zero

end
