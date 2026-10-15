import Stage_import_0

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000
set_option linter.unusedVariables false

-- The wire types repeat the ring modulus `X ^ n + 1`; unfolding it while checking a
-- projection against its field type expands it at the ring dimension.
attribute [local irreducible] Mxx.Primitives.negacyclicModulus

noncomputable section

namespace Stage_import_0

local notation "mxxMat0" => Mxx.Primitives.ExactMatrix modulus_0 8 1 25
local notation "mxxMat1" => Mxx.Primitives.ExactMatrix modulus_0 8 1 1

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Ctx (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) where
  inputs : mxxMat0 × mxxMat0 × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat0) × (Fin 2 → Int) × (Fin 2 → Int) × mxxMat0 × mxxMat0 × (Fin 2 → Int) × (Fin 2 → Int) × (Fin 2 → Int) × (Fin 2 → Int) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → Int) × (Fin 4 → Int) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × Unit
  witness : generatedRoot.Witness

def generatedRoot.w_0_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : mxxMat0 :=
  ctx.inputs.1

def generatedRoot.w_1_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : mxxMat0 :=
  ctx.inputs.2.1

def generatedRoot.w_2_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 2 → mxxMat0 :=
  ctx.inputs.2.2.1

def generatedRoot.w_3_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 2 → mxxMat0 :=
  ctx.inputs.2.2.2.1

def generatedRoot.w_4_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 2 → Int :=
  ctx.inputs.2.2.2.2.1

def generatedRoot.w_5_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 2 → Int :=
  ctx.inputs.2.2.2.2.2.1

def generatedRoot.w_7_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.1

def generatedRoot.w_8_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.1

def generatedRoot.w_9_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 2 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.1

def generatedRoot.w_10_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 2 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_12_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 2 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_13_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 2 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_15_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 4 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_16_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 4 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_17_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 4 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_18_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 4 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_20_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 1 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_21_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 1 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_23_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 1 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_24_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 1 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_26_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 1 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_27_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 1 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_29_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 1 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_30_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 1 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_32_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 1 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_33_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 1 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_35_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 1 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_36_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 1 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_38_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 1 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_39_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 1 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_41_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 8 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_42_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 8 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_44_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 4 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_45_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 4 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_6_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 2 → mxxMat0 :=
  ctx.witness.w_6_0

def generatedRoot.w_6_1 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 2 → mxxMat0 :=
  ctx.witness.w_6_1

def generatedRoot.w_6_2 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 2 → mxxMat1 :=
  ctx.witness.w_6_2

def generatedRoot.w_11_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 2 → mxxMat0 :=
  ctx.witness.w_11_0

def generatedRoot.w_11_1 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 2 → mxxMat0 :=
  ctx.witness.w_11_1

def generatedRoot.w_11_2 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 2 → mxxMat1 :=
  ctx.witness.w_11_2

def generatedRoot.w_14_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 2 → mxxMat0 :=
  ctx.witness.w_14_0

def generatedRoot.w_14_1 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 2 → mxxMat0 :=
  ctx.witness.w_14_1

def generatedRoot.w_14_2 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 2 → mxxMat1 :=
  ctx.witness.w_14_2

def generatedRoot.w_19_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 4 → mxxMat0 :=
  ctx.witness.w_19_0

def generatedRoot.w_19_1 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 4 → mxxMat0 :=
  ctx.witness.w_19_1

def generatedRoot.w_19_2 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 4 → mxxMat1 :=
  ctx.witness.w_19_2

def generatedRoot.w_22_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 1 → mxxMat0 :=
  ctx.witness.w_22_0

def generatedRoot.w_22_1 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 1 → mxxMat0 :=
  ctx.witness.w_22_1

def generatedRoot.w_25_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 1 → mxxMat0 :=
  ctx.witness.w_25_0

def generatedRoot.w_25_1 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 1 → mxxMat0 :=
  ctx.witness.w_25_1

def generatedRoot.w_28_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 1 → mxxMat0 :=
  ctx.witness.w_28_0

def generatedRoot.w_28_1 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 1 → mxxMat0 :=
  ctx.witness.w_28_1

def generatedRoot.w_31_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 1 → mxxMat0 :=
  ctx.witness.w_31_0

def generatedRoot.w_31_1 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 1 → mxxMat0 :=
  ctx.witness.w_31_1

def generatedRoot.w_34_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 1 → mxxMat0 :=
  ctx.witness.w_34_0

def generatedRoot.w_34_1 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 1 → mxxMat0 :=
  ctx.witness.w_34_1

def generatedRoot.w_37_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 1 → mxxMat0 :=
  ctx.witness.w_37_0

def generatedRoot.w_37_1 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 1 → mxxMat0 :=
  ctx.witness.w_37_1

def generatedRoot.w_40_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 1 → mxxMat0 :=
  ctx.witness.w_40_0

def generatedRoot.w_40_1 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 1 → mxxMat0 :=
  ctx.witness.w_40_1

def generatedRoot.w_43_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 8 → mxxMat0 :=
  ctx.witness.w_43_0

def generatedRoot.w_43_1 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 8 → mxxMat0 :=
  ctx.witness.w_43_1

def generatedRoot.w_46_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 4 → mxxMat0 :=
  ctx.witness.w_46_0

def generatedRoot.w_46_1 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 4 → mxxMat0 :=
  ctx.witness.w_46_1

def generatedRoot.w_47_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 1 → mxxMat0 :=
  ctx.witness.w_47_0

def generatedRoot.w_47_1 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 1 → mxxMat0 :=
  ctx.witness.w_47_1

def generatedRoot.w_48_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 1 → mxxMat0 :=
  ctx.witness.w_48_0

def generatedRoot.w_48_1 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 1 → mxxMat0 :=
  ctx.witness.w_48_1

def generatedRoot.w_49_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 1 → mxxMat0 :=
  ctx.witness.w_49_0

def generatedRoot.w_49_1 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 1 → mxxMat0 :=
  ctx.witness.w_49_1

def generatedRoot.w_50_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 1 → mxxMat0 :=
  ctx.witness.w_50_0

def generatedRoot.w_50_1 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 1 → mxxMat0 :=
  ctx.witness.w_50_1

def generatedRoot.w_51_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 1 → mxxMat0 :=
  ctx.witness.w_51_0

def generatedRoot.w_51_1 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 1 → mxxMat0 :=
  ctx.witness.w_51_1

def generatedRoot.w_52_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 1 → mxxMat0 :=
  ctx.witness.w_52_0

def generatedRoot.w_52_1 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 1 → mxxMat0 :=
  ctx.witness.w_52_1

def generatedRoot.w_53_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 1 → mxxMat0 :=
  ctx.witness.w_53_0

def generatedRoot.w_53_1 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 1 → mxxMat0 :=
  ctx.witness.w_53_1

def generatedRoot.w_54_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 8 → mxxMat0 :=
  ctx.witness.w_54_0

def generatedRoot.w_54_1 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 8 → mxxMat0 :=
  ctx.witness.w_54_1

def generatedRoot.w_55_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 4 → mxxMat0 :=
  ctx.witness.w_55_0

def generatedRoot.w_55_1 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 4 → mxxMat0 :=
  ctx.witness.w_55_1

def generatedRoot.w_56_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 1 → mxxMat0 :=
  ctx.witness.w_56_0

def generatedRoot.w_56_1 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 1 → mxxMat0 :=
  ctx.witness.w_56_1

def generatedRoot.w_57_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 1 → mxxMat0 :=
  ctx.witness.w_57_0

def generatedRoot.w_57_1 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 1 → mxxMat0 :=
  ctx.witness.w_57_1

def generatedRoot.w_58_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 1 → mxxMat0 :=
  ctx.witness.w_58_0

def generatedRoot.w_58_1 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 1 → mxxMat0 :=
  ctx.witness.w_58_1

def generatedRoot.w_59_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 1 → mxxMat0 :=
  ctx.witness.w_59_0

def generatedRoot.w_59_1 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 1 → mxxMat0 :=
  ctx.witness.w_59_1

def generatedRoot.w_60_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 1 → mxxMat0 :=
  ctx.witness.w_60_0

def generatedRoot.w_60_1 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 1 → mxxMat0 :=
  ctx.witness.w_60_1

def generatedRoot.w_61_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 1 → mxxMat0 :=
  ctx.witness.w_61_0

def generatedRoot.w_61_1 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 1 → mxxMat0 :=
  ctx.witness.w_61_1

def generatedRoot.w_62_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 1 → mxxMat0 :=
  ctx.witness.w_62_0

def generatedRoot.w_62_1 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 1 → mxxMat0 :=
  ctx.witness.w_62_1

def generatedRoot.w_63_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 8 → mxxMat0 :=
  ctx.witness.w_63_0

def generatedRoot.w_63_1 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 8 → mxxMat0 :=
  ctx.witness.w_63_1

def generatedRoot.w_64_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 4 → mxxMat0 :=
  ctx.witness.w_64_0

def generatedRoot.w_64_1 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 4 → mxxMat0 :=
  ctx.witness.w_64_1

def generatedRoot.w_65_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 1 → mxxMat0 :=
  ctx.witness.w_65_0

def generatedRoot.w_65_1 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 1 → mxxMat0 :=
  ctx.witness.w_65_1

def generatedRoot.w_66_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 1 → mxxMat0 :=
  ctx.witness.w_66_0

def generatedRoot.w_66_1 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 1 → mxxMat0 :=
  ctx.witness.w_66_1

def generatedRoot.w_67_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 1 → mxxMat0 :=
  ctx.witness.w_67_0

def generatedRoot.w_67_1 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 1 → mxxMat0 :=
  ctx.witness.w_67_1

def generatedRoot.w_68_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 1 → mxxMat0 :=
  ctx.witness.w_68_0

def generatedRoot.w_68_1 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 1 → mxxMat0 :=
  ctx.witness.w_68_1

def generatedRoot.w_69_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 1 → mxxMat0 :=
  ctx.witness.w_69_0

def generatedRoot.w_69_1 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 1 → mxxMat0 :=
  ctx.witness.w_69_1

def generatedRoot.w_70_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 1 → mxxMat0 :=
  ctx.witness.w_70_0

def generatedRoot.w_70_1 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 1 → mxxMat0 :=
  ctx.witness.w_70_1

def generatedRoot.w_71_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 1 → mxxMat0 :=
  ctx.witness.w_71_0

def generatedRoot.w_71_1 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 1 → mxxMat0 :=
  ctx.witness.w_71_1

def generatedRoot.w_72_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 8 → mxxMat0 :=
  ctx.witness.w_72_0

def generatedRoot.w_72_1 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 8 → mxxMat0 :=
  ctx.witness.w_72_1

def generatedRoot.w_73_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 4 → mxxMat0 :=
  ctx.witness.w_73_0

def generatedRoot.w_73_1 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 4 → mxxMat0 :=
  ctx.witness.w_73_1

def generatedRoot.w_74_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 1 → mxxMat0 :=
  ctx.witness.w_74_0

def generatedRoot.w_74_1 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 1 → mxxMat0 :=
  ctx.witness.w_74_1

def generatedRoot.w_75_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 1 → mxxMat0 :=
  ctx.witness.w_75_0

def generatedRoot.w_75_1 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 1 → mxxMat0 :=
  ctx.witness.w_75_1

def generatedRoot.w_76_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 1 → mxxMat0 :=
  ctx.witness.w_76_0

def generatedRoot.w_76_1 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 1 → mxxMat0 :=
  ctx.witness.w_76_1

def generatedRoot.w_77_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 1 → mxxMat0 :=
  ctx.witness.w_77_0

def generatedRoot.w_77_1 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 1 → mxxMat0 :=
  ctx.witness.w_77_1

def generatedRoot.w_78_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 1 → mxxMat0 :=
  ctx.witness.w_78_0

def generatedRoot.w_78_1 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 1 → mxxMat0 :=
  ctx.witness.w_78_1

def generatedRoot.w_79_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 1 → mxxMat0 :=
  ctx.witness.w_79_0

def generatedRoot.w_79_1 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 1 → mxxMat0 :=
  ctx.witness.w_79_1

def generatedRoot.w_80_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 1 → mxxMat0 :=
  ctx.witness.w_80_0

def generatedRoot.w_80_1 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 1 → mxxMat0 :=
  ctx.witness.w_80_1

def generatedRoot.w_81_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 8 → mxxMat0 :=
  ctx.witness.w_81_0

def generatedRoot.w_81_1 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 8 → mxxMat0 :=
  ctx.witness.w_81_1

def generatedRoot.w_82_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 4 → mxxMat0 :=
  ctx.witness.w_82_0

def generatedRoot.w_82_1 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Fin 4 → mxxMat0 :=
  ctx.witness.w_82_1

def generatedRoot.outs {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : mxxMat0 × mxxMat0 × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat1) × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat1) × mxxMat0 × mxxMat0 × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat1) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat1) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × Unit :=
  ((generatedRoot.w_0_0 ctx), ((generatedRoot.w_1_0 ctx), ((generatedRoot.w_6_0 ctx), ((generatedRoot.w_6_1 ctx), ((generatedRoot.w_6_2 ctx), ((generatedRoot.w_7_0 ctx), ((generatedRoot.w_8_0 ctx), ((generatedRoot.w_0_0 ctx), ((generatedRoot.w_1_0 ctx), ((generatedRoot.w_11_0 ctx), ((generatedRoot.w_11_1 ctx), ((generatedRoot.w_11_2 ctx), ((generatedRoot.w_0_0 ctx), ((generatedRoot.w_1_0 ctx), ((generatedRoot.w_14_0 ctx), ((generatedRoot.w_14_1 ctx), ((generatedRoot.w_14_2 ctx), ((generatedRoot.w_19_0 ctx), ((generatedRoot.w_19_1 ctx), ((generatedRoot.w_19_2 ctx), ((generatedRoot.w_22_0 ctx), ((generatedRoot.w_22_1 ctx), ((generatedRoot.w_25_0 ctx), ((generatedRoot.w_25_1 ctx), ((generatedRoot.w_28_0 ctx), ((generatedRoot.w_28_1 ctx), ((generatedRoot.w_31_0 ctx), ((generatedRoot.w_31_1 ctx), ((generatedRoot.w_34_0 ctx), ((generatedRoot.w_34_1 ctx), ((generatedRoot.w_37_0 ctx), ((generatedRoot.w_37_1 ctx), ((generatedRoot.w_40_0 ctx), ((generatedRoot.w_40_1 ctx), ((generatedRoot.w_43_0 ctx), ((generatedRoot.w_43_1 ctx), ((generatedRoot.w_46_0 ctx), ((generatedRoot.w_46_1 ctx), ((generatedRoot.w_47_0 ctx), ((generatedRoot.w_47_1 ctx), ((generatedRoot.w_48_0 ctx), ((generatedRoot.w_48_1 ctx), ((generatedRoot.w_49_0 ctx), ((generatedRoot.w_49_1 ctx), ((generatedRoot.w_50_0 ctx), ((generatedRoot.w_50_1 ctx), ((generatedRoot.w_51_0 ctx), ((generatedRoot.w_51_1 ctx), ((generatedRoot.w_52_0 ctx), ((generatedRoot.w_52_1 ctx), ((generatedRoot.w_53_0 ctx), ((generatedRoot.w_53_1 ctx), ((generatedRoot.w_54_0 ctx), ((generatedRoot.w_54_1 ctx), ((generatedRoot.w_55_0 ctx), ((generatedRoot.w_55_1 ctx), ((generatedRoot.w_56_0 ctx), ((generatedRoot.w_56_1 ctx), ((generatedRoot.w_57_0 ctx), ((generatedRoot.w_57_1 ctx), ((generatedRoot.w_58_0 ctx), ((generatedRoot.w_58_1 ctx), ((generatedRoot.w_59_0 ctx), ((generatedRoot.w_59_1 ctx), ((generatedRoot.w_60_0 ctx), ((generatedRoot.w_60_1 ctx), ((generatedRoot.w_61_0 ctx), ((generatedRoot.w_61_1 ctx), ((generatedRoot.w_62_0 ctx), ((generatedRoot.w_62_1 ctx), ((generatedRoot.w_63_0 ctx), ((generatedRoot.w_63_1 ctx), ((generatedRoot.w_64_0 ctx), ((generatedRoot.w_64_1 ctx), ((generatedRoot.w_65_0 ctx), ((generatedRoot.w_65_1 ctx), ((generatedRoot.w_66_0 ctx), ((generatedRoot.w_66_1 ctx), ((generatedRoot.w_67_0 ctx), ((generatedRoot.w_67_1 ctx), ((generatedRoot.w_68_0 ctx), ((generatedRoot.w_68_1 ctx), ((generatedRoot.w_69_0 ctx), ((generatedRoot.w_69_1 ctx), ((generatedRoot.w_70_0 ctx), ((generatedRoot.w_70_1 ctx), ((generatedRoot.w_71_0 ctx), ((generatedRoot.w_71_1 ctx), ((generatedRoot.w_72_0 ctx), ((generatedRoot.w_72_1 ctx), ((generatedRoot.w_73_0 ctx), ((generatedRoot.w_73_1 ctx), ((generatedRoot.w_74_0 ctx), ((generatedRoot.w_74_1 ctx), ((generatedRoot.w_75_0 ctx), ((generatedRoot.w_75_1 ctx), ((generatedRoot.w_76_0 ctx), ((generatedRoot.w_76_1 ctx), ((generatedRoot.w_77_0 ctx), ((generatedRoot.w_77_1 ctx), ((generatedRoot.w_78_0 ctx), ((generatedRoot.w_78_1 ctx), ((generatedRoot.w_79_0 ctx), ((generatedRoot.w_79_1 ctx), ((generatedRoot.w_80_0 ctx), ((generatedRoot.w_80_1 ctx), ((generatedRoot.w_81_0 ctx), ((generatedRoot.w_81_1 ctx), ((generatedRoot.w_82_0 ctx), ((generatedRoot.w_82_1 ctx), ()))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))

structure generatedRoot.Facts_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Prop where
  t0 : (2) = 2
  h6_2 : (∀ i : Fin 2, parallel_generatedRoot_6 tape (path ++ [6, i.val]) params i (((generatedRoot.w_2_0 ctx) i), (((generatedRoot.w_3_0 ctx) i), (((generatedRoot.w_4_0 ctx) i), (((generatedRoot.w_5_0 ctx) i), ())))) (((generatedRoot.w_6_0 ctx) i), (((generatedRoot.w_6_1 ctx) i), (((generatedRoot.w_6_2 ctx) i), ()))))
  t2 : (2) = 2
  h11_2 : (∀ i : Fin 2, parallel_generatedRoot_11 tape (path ++ [11, i.val]) params i (((generatedRoot.w_2_0 ctx) i), (((generatedRoot.w_3_0 ctx) i), (((generatedRoot.w_9_0 ctx) i), (((generatedRoot.w_10_0 ctx) i), ())))) (((generatedRoot.w_11_0 ctx) i), (((generatedRoot.w_11_1 ctx) i), (((generatedRoot.w_11_2 ctx) i), ()))))
  t4 : (2) = 2
  h14_2 : (∀ i : Fin 2, parallel_generatedRoot_14 tape (path ++ [14, i.val]) params i (((generatedRoot.w_2_0 ctx) i), (((generatedRoot.w_3_0 ctx) i), (((generatedRoot.w_12_0 ctx) i), (((generatedRoot.w_13_0 ctx) i), ())))) (((generatedRoot.w_14_0 ctx) i), (((generatedRoot.w_14_1 ctx) i), (((generatedRoot.w_14_2 ctx) i), ()))))
  t6 : (4) = 4
  h19_2 : (∀ i : Fin 4, parallel_generatedRoot_19 tape (path ++ [19, i.val]) params i (((generatedRoot.w_15_0 ctx) i), (((generatedRoot.w_16_0 ctx) i), (((generatedRoot.w_17_0 ctx) i), (((generatedRoot.w_18_0 ctx) i), ())))) (((generatedRoot.w_19_0 ctx) i), (((generatedRoot.w_19_1 ctx) i), (((generatedRoot.w_19_2 ctx) i), ()))))
  t8 : (1) = 1
  h22_1 : (∀ i : Fin 1, parallel_generatedRoot_22 tape (path ++ [22, i.val]) params i (((generatedRoot.w_20_0 ctx) i), (((generatedRoot.w_21_0 ctx) i), ())) (((generatedRoot.w_22_0 ctx) i), (((generatedRoot.w_22_1 ctx) i), ())))
  t10 : (1) = 1
  h25_1 : (∀ i : Fin 1, parallel_generatedRoot_25 tape (path ++ [25, i.val]) params i (((generatedRoot.w_23_0 ctx) i), (((generatedRoot.w_24_0 ctx) i), ())) (((generatedRoot.w_25_0 ctx) i), (((generatedRoot.w_25_1 ctx) i), ())))
  t12 : (1) = 1
  h28_1 : (∀ i : Fin 1, parallel_generatedRoot_28 tape (path ++ [28, i.val]) params i (((generatedRoot.w_26_0 ctx) i), (((generatedRoot.w_27_0 ctx) i), ())) (((generatedRoot.w_28_0 ctx) i), (((generatedRoot.w_28_1 ctx) i), ())))
  t14 : (1) = 1
  h31_1 : (∀ i : Fin 1, parallel_generatedRoot_31 tape (path ++ [31, i.val]) params i (((generatedRoot.w_29_0 ctx) i), (((generatedRoot.w_30_0 ctx) i), ())) (((generatedRoot.w_31_0 ctx) i), (((generatedRoot.w_31_1 ctx) i), ())))
  t16 : (1) = 1
  h34_1 : (∀ i : Fin 1, parallel_generatedRoot_34 tape (path ++ [34, i.val]) params i (((generatedRoot.w_32_0 ctx) i), (((generatedRoot.w_33_0 ctx) i), ())) (((generatedRoot.w_34_0 ctx) i), (((generatedRoot.w_34_1 ctx) i), ())))
  t18 : (1) = 1
  h37_1 : (∀ i : Fin 1, parallel_generatedRoot_37 tape (path ++ [37, i.val]) params i (((generatedRoot.w_35_0 ctx) i), (((generatedRoot.w_36_0 ctx) i), ())) (((generatedRoot.w_37_0 ctx) i), (((generatedRoot.w_37_1 ctx) i), ())))
  t20 : (1) = 1
  h40_1 : (∀ i : Fin 1, parallel_generatedRoot_40 tape (path ++ [40, i.val]) params i (((generatedRoot.w_38_0 ctx) i), (((generatedRoot.w_39_0 ctx) i), ())) (((generatedRoot.w_40_0 ctx) i), (((generatedRoot.w_40_1 ctx) i), ())))
  t22 : (8) = 8
  h43_1 : (∀ i : Fin 8, parallel_generatedRoot_43 tape (path ++ [43, i.val]) params i (((generatedRoot.w_41_0 ctx) i), (((generatedRoot.w_42_0 ctx) i), ())) (((generatedRoot.w_43_0 ctx) i), (((generatedRoot.w_43_1 ctx) i), ())))
  t24 : (4) = 4
  h46_1 : (∀ i : Fin 4, parallel_generatedRoot_46 tape (path ++ [46, i.val]) params i (((generatedRoot.w_44_0 ctx) i), (((generatedRoot.w_45_0 ctx) i), ())) (((generatedRoot.w_46_0 ctx) i), (((generatedRoot.w_46_1 ctx) i), ())))
  t26 : (1) = 1
  h47_1 : (∀ i : Fin 1, parallel_generatedRoot_47 tape (path ++ [47, i.val]) params i (((generatedRoot.w_20_0 ctx) i), (((generatedRoot.w_21_0 ctx) i), ())) (((generatedRoot.w_47_0 ctx) i), (((generatedRoot.w_47_1 ctx) i), ())))
  t28 : (1) = 1
  h48_1 : (∀ i : Fin 1, parallel_generatedRoot_48 tape (path ++ [48, i.val]) params i (((generatedRoot.w_23_0 ctx) i), (((generatedRoot.w_24_0 ctx) i), ())) (((generatedRoot.w_48_0 ctx) i), (((generatedRoot.w_48_1 ctx) i), ())))
  t30 : (1) = 1
  h49_1 : (∀ i : Fin 1, parallel_generatedRoot_49 tape (path ++ [49, i.val]) params i (((generatedRoot.w_26_0 ctx) i), (((generatedRoot.w_27_0 ctx) i), ())) (((generatedRoot.w_49_0 ctx) i), (((generatedRoot.w_49_1 ctx) i), ())))

theorem generatedRoot.facts_0 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {ctx : generatedRoot.Ctx tape path params} {outputs : mxxMat0 × mxxMat0 × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat1) × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat1) × mxxMat0 × mxxMat0 × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat1) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat1) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × Unit}
    (h : generatedRoot.constraints_0 tape path params (generatedRoot.w_2_0 ctx) (generatedRoot.w_3_0 ctx) (generatedRoot.w_4_0 ctx) (generatedRoot.w_5_0 ctx) (generatedRoot.w_9_0 ctx) (generatedRoot.w_10_0 ctx) (generatedRoot.w_12_0 ctx) (generatedRoot.w_13_0 ctx) (generatedRoot.w_15_0 ctx) (generatedRoot.w_16_0 ctx) (generatedRoot.w_17_0 ctx) (generatedRoot.w_18_0 ctx) (generatedRoot.w_20_0 ctx) (generatedRoot.w_21_0 ctx) (generatedRoot.w_23_0 ctx) (generatedRoot.w_24_0 ctx) (generatedRoot.w_26_0 ctx) (generatedRoot.w_27_0 ctx) (generatedRoot.w_29_0 ctx) (generatedRoot.w_30_0 ctx) (generatedRoot.w_32_0 ctx) (generatedRoot.w_33_0 ctx) (generatedRoot.w_35_0 ctx) (generatedRoot.w_36_0 ctx) (generatedRoot.w_38_0 ctx) (generatedRoot.w_39_0 ctx) (generatedRoot.w_41_0 ctx) (generatedRoot.w_42_0 ctx) (generatedRoot.w_44_0 ctx) (generatedRoot.w_45_0 ctx) (generatedRoot.w_6_0 ctx) (generatedRoot.w_6_1 ctx) (generatedRoot.w_6_2 ctx) (generatedRoot.w_11_0 ctx) (generatedRoot.w_11_1 ctx) (generatedRoot.w_11_2 ctx) (generatedRoot.w_14_0 ctx) (generatedRoot.w_14_1 ctx) (generatedRoot.w_14_2 ctx) (generatedRoot.w_19_0 ctx) (generatedRoot.w_19_1 ctx) (generatedRoot.w_19_2 ctx) (generatedRoot.w_22_0 ctx) (generatedRoot.w_22_1 ctx) (generatedRoot.w_25_0 ctx) (generatedRoot.w_25_1 ctx) (generatedRoot.w_28_0 ctx) (generatedRoot.w_28_1 ctx) (generatedRoot.w_31_0 ctx) (generatedRoot.w_31_1 ctx) (generatedRoot.w_34_0 ctx) (generatedRoot.w_34_1 ctx) (generatedRoot.w_37_0 ctx) (generatedRoot.w_37_1 ctx) (generatedRoot.w_40_0 ctx) (generatedRoot.w_40_1 ctx) (generatedRoot.w_43_0 ctx) (generatedRoot.w_43_1 ctx) (generatedRoot.w_46_0 ctx) (generatedRoot.w_46_1 ctx) (generatedRoot.w_47_0 ctx) (generatedRoot.w_47_1 ctx) (generatedRoot.w_48_0 ctx) (generatedRoot.w_48_1 ctx) (generatedRoot.w_49_0 ctx) (generatedRoot.w_49_1 ctx)) :
    generatedRoot.Facts_0 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_1 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Prop where
  t32 : (1) = 1
  h50_1 : (∀ i : Fin 1, parallel_generatedRoot_50 tape (path ++ [50, i.val]) params i (((generatedRoot.w_29_0 ctx) i), (((generatedRoot.w_30_0 ctx) i), ())) (((generatedRoot.w_50_0 ctx) i), (((generatedRoot.w_50_1 ctx) i), ())))
  t34 : (1) = 1
  h51_1 : (∀ i : Fin 1, parallel_generatedRoot_51 tape (path ++ [51, i.val]) params i (((generatedRoot.w_32_0 ctx) i), (((generatedRoot.w_33_0 ctx) i), ())) (((generatedRoot.w_51_0 ctx) i), (((generatedRoot.w_51_1 ctx) i), ())))
  t36 : (1) = 1
  h52_1 : (∀ i : Fin 1, parallel_generatedRoot_52 tape (path ++ [52, i.val]) params i (((generatedRoot.w_35_0 ctx) i), (((generatedRoot.w_36_0 ctx) i), ())) (((generatedRoot.w_52_0 ctx) i), (((generatedRoot.w_52_1 ctx) i), ())))
  t38 : (1) = 1
  h53_1 : (∀ i : Fin 1, parallel_generatedRoot_53 tape (path ++ [53, i.val]) params i (((generatedRoot.w_38_0 ctx) i), (((generatedRoot.w_39_0 ctx) i), ())) (((generatedRoot.w_53_0 ctx) i), (((generatedRoot.w_53_1 ctx) i), ())))
  t40 : (8) = 8
  h54_1 : (∀ i : Fin 8, parallel_generatedRoot_54 tape (path ++ [54, i.val]) params i (((generatedRoot.w_41_0 ctx) i), (((generatedRoot.w_42_0 ctx) i), ())) (((generatedRoot.w_54_0 ctx) i), (((generatedRoot.w_54_1 ctx) i), ())))
  t42 : (4) = 4
  h55_1 : (∀ i : Fin 4, parallel_generatedRoot_55 tape (path ++ [55, i.val]) params i (((generatedRoot.w_44_0 ctx) i), (((generatedRoot.w_45_0 ctx) i), ())) (((generatedRoot.w_55_0 ctx) i), (((generatedRoot.w_55_1 ctx) i), ())))
  t44 : (1) = 1
  h56_1 : (∀ i : Fin 1, parallel_generatedRoot_56 tape (path ++ [56, i.val]) params i (((generatedRoot.w_20_0 ctx) i), (((generatedRoot.w_21_0 ctx) i), ())) (((generatedRoot.w_56_0 ctx) i), (((generatedRoot.w_56_1 ctx) i), ())))
  t46 : (1) = 1
  h57_1 : (∀ i : Fin 1, parallel_generatedRoot_57 tape (path ++ [57, i.val]) params i (((generatedRoot.w_23_0 ctx) i), (((generatedRoot.w_24_0 ctx) i), ())) (((generatedRoot.w_57_0 ctx) i), (((generatedRoot.w_57_1 ctx) i), ())))
  t48 : (1) = 1
  h58_1 : (∀ i : Fin 1, parallel_generatedRoot_58 tape (path ++ [58, i.val]) params i (((generatedRoot.w_26_0 ctx) i), (((generatedRoot.w_27_0 ctx) i), ())) (((generatedRoot.w_58_0 ctx) i), (((generatedRoot.w_58_1 ctx) i), ())))
  t50 : (1) = 1
  h59_1 : (∀ i : Fin 1, parallel_generatedRoot_59 tape (path ++ [59, i.val]) params i (((generatedRoot.w_29_0 ctx) i), (((generatedRoot.w_30_0 ctx) i), ())) (((generatedRoot.w_59_0 ctx) i), (((generatedRoot.w_59_1 ctx) i), ())))
  t52 : (1) = 1
  h60_1 : (∀ i : Fin 1, parallel_generatedRoot_60 tape (path ++ [60, i.val]) params i (((generatedRoot.w_32_0 ctx) i), (((generatedRoot.w_33_0 ctx) i), ())) (((generatedRoot.w_60_0 ctx) i), (((generatedRoot.w_60_1 ctx) i), ())))
  t54 : (1) = 1
  h61_1 : (∀ i : Fin 1, parallel_generatedRoot_61 tape (path ++ [61, i.val]) params i (((generatedRoot.w_35_0 ctx) i), (((generatedRoot.w_36_0 ctx) i), ())) (((generatedRoot.w_61_0 ctx) i), (((generatedRoot.w_61_1 ctx) i), ())))
  t56 : (1) = 1
  h62_1 : (∀ i : Fin 1, parallel_generatedRoot_62 tape (path ++ [62, i.val]) params i (((generatedRoot.w_38_0 ctx) i), (((generatedRoot.w_39_0 ctx) i), ())) (((generatedRoot.w_62_0 ctx) i), (((generatedRoot.w_62_1 ctx) i), ())))
  t58 : (8) = 8
  h63_1 : (∀ i : Fin 8, parallel_generatedRoot_63 tape (path ++ [63, i.val]) params i (((generatedRoot.w_41_0 ctx) i), (((generatedRoot.w_42_0 ctx) i), ())) (((generatedRoot.w_63_0 ctx) i), (((generatedRoot.w_63_1 ctx) i), ())))
  t60 : (4) = 4
  h64_1 : (∀ i : Fin 4, parallel_generatedRoot_64 tape (path ++ [64, i.val]) params i (((generatedRoot.w_44_0 ctx) i), (((generatedRoot.w_45_0 ctx) i), ())) (((generatedRoot.w_64_0 ctx) i), (((generatedRoot.w_64_1 ctx) i), ())))
  t62 : (1) = 1
  h65_1 : (∀ i : Fin 1, parallel_generatedRoot_65 tape (path ++ [65, i.val]) params i (((generatedRoot.w_20_0 ctx) i), (((generatedRoot.w_21_0 ctx) i), ())) (((generatedRoot.w_65_0 ctx) i), (((generatedRoot.w_65_1 ctx) i), ())))

theorem generatedRoot.facts_1 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {ctx : generatedRoot.Ctx tape path params} {outputs : mxxMat0 × mxxMat0 × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat1) × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat1) × mxxMat0 × mxxMat0 × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat1) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat1) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × Unit}
    (h : generatedRoot.constraints_1 tape path params (generatedRoot.w_20_0 ctx) (generatedRoot.w_21_0 ctx) (generatedRoot.w_23_0 ctx) (generatedRoot.w_24_0 ctx) (generatedRoot.w_26_0 ctx) (generatedRoot.w_27_0 ctx) (generatedRoot.w_29_0 ctx) (generatedRoot.w_30_0 ctx) (generatedRoot.w_32_0 ctx) (generatedRoot.w_33_0 ctx) (generatedRoot.w_35_0 ctx) (generatedRoot.w_36_0 ctx) (generatedRoot.w_38_0 ctx) (generatedRoot.w_39_0 ctx) (generatedRoot.w_41_0 ctx) (generatedRoot.w_42_0 ctx) (generatedRoot.w_44_0 ctx) (generatedRoot.w_45_0 ctx) (generatedRoot.w_50_0 ctx) (generatedRoot.w_50_1 ctx) (generatedRoot.w_51_0 ctx) (generatedRoot.w_51_1 ctx) (generatedRoot.w_52_0 ctx) (generatedRoot.w_52_1 ctx) (generatedRoot.w_53_0 ctx) (generatedRoot.w_53_1 ctx) (generatedRoot.w_54_0 ctx) (generatedRoot.w_54_1 ctx) (generatedRoot.w_55_0 ctx) (generatedRoot.w_55_1 ctx) (generatedRoot.w_56_0 ctx) (generatedRoot.w_56_1 ctx) (generatedRoot.w_57_0 ctx) (generatedRoot.w_57_1 ctx) (generatedRoot.w_58_0 ctx) (generatedRoot.w_58_1 ctx) (generatedRoot.w_59_0 ctx) (generatedRoot.w_59_1 ctx) (generatedRoot.w_60_0 ctx) (generatedRoot.w_60_1 ctx) (generatedRoot.w_61_0 ctx) (generatedRoot.w_61_1 ctx) (generatedRoot.w_62_0 ctx) (generatedRoot.w_62_1 ctx) (generatedRoot.w_63_0 ctx) (generatedRoot.w_63_1 ctx) (generatedRoot.w_64_0 ctx) (generatedRoot.w_64_1 ctx) (generatedRoot.w_65_0 ctx) (generatedRoot.w_65_1 ctx)) :
    generatedRoot.Facts_1 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_2 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) : Prop where
  t64 : (1) = 1
  h66_1 : (∀ i : Fin 1, parallel_generatedRoot_66 tape (path ++ [66, i.val]) params i (((generatedRoot.w_23_0 ctx) i), (((generatedRoot.w_24_0 ctx) i), ())) (((generatedRoot.w_66_0 ctx) i), (((generatedRoot.w_66_1 ctx) i), ())))
  t66 : (1) = 1
  h67_1 : (∀ i : Fin 1, parallel_generatedRoot_67 tape (path ++ [67, i.val]) params i (((generatedRoot.w_26_0 ctx) i), (((generatedRoot.w_27_0 ctx) i), ())) (((generatedRoot.w_67_0 ctx) i), (((generatedRoot.w_67_1 ctx) i), ())))
  t68 : (1) = 1
  h68_1 : (∀ i : Fin 1, parallel_generatedRoot_68 tape (path ++ [68, i.val]) params i (((generatedRoot.w_29_0 ctx) i), (((generatedRoot.w_30_0 ctx) i), ())) (((generatedRoot.w_68_0 ctx) i), (((generatedRoot.w_68_1 ctx) i), ())))
  t70 : (1) = 1
  h69_1 : (∀ i : Fin 1, parallel_generatedRoot_69 tape (path ++ [69, i.val]) params i (((generatedRoot.w_32_0 ctx) i), (((generatedRoot.w_33_0 ctx) i), ())) (((generatedRoot.w_69_0 ctx) i), (((generatedRoot.w_69_1 ctx) i), ())))
  t72 : (1) = 1
  h70_1 : (∀ i : Fin 1, parallel_generatedRoot_70 tape (path ++ [70, i.val]) params i (((generatedRoot.w_35_0 ctx) i), (((generatedRoot.w_36_0 ctx) i), ())) (((generatedRoot.w_70_0 ctx) i), (((generatedRoot.w_70_1 ctx) i), ())))
  t74 : (1) = 1
  h71_1 : (∀ i : Fin 1, parallel_generatedRoot_71 tape (path ++ [71, i.val]) params i (((generatedRoot.w_38_0 ctx) i), (((generatedRoot.w_39_0 ctx) i), ())) (((generatedRoot.w_71_0 ctx) i), (((generatedRoot.w_71_1 ctx) i), ())))
  t76 : (8) = 8
  h72_1 : (∀ i : Fin 8, parallel_generatedRoot_72 tape (path ++ [72, i.val]) params i (((generatedRoot.w_41_0 ctx) i), (((generatedRoot.w_42_0 ctx) i), ())) (((generatedRoot.w_72_0 ctx) i), (((generatedRoot.w_72_1 ctx) i), ())))
  t78 : (4) = 4
  h73_1 : (∀ i : Fin 4, parallel_generatedRoot_73 tape (path ++ [73, i.val]) params i (((generatedRoot.w_44_0 ctx) i), (((generatedRoot.w_45_0 ctx) i), ())) (((generatedRoot.w_73_0 ctx) i), (((generatedRoot.w_73_1 ctx) i), ())))
  t80 : (1) = 1
  h74_1 : (∀ i : Fin 1, parallel_generatedRoot_74 tape (path ++ [74, i.val]) params i (((generatedRoot.w_20_0 ctx) i), (((generatedRoot.w_21_0 ctx) i), ())) (((generatedRoot.w_74_0 ctx) i), (((generatedRoot.w_74_1 ctx) i), ())))
  t82 : (1) = 1
  h75_1 : (∀ i : Fin 1, parallel_generatedRoot_75 tape (path ++ [75, i.val]) params i (((generatedRoot.w_23_0 ctx) i), (((generatedRoot.w_24_0 ctx) i), ())) (((generatedRoot.w_75_0 ctx) i), (((generatedRoot.w_75_1 ctx) i), ())))
  t84 : (1) = 1
  h76_1 : (∀ i : Fin 1, parallel_generatedRoot_76 tape (path ++ [76, i.val]) params i (((generatedRoot.w_26_0 ctx) i), (((generatedRoot.w_27_0 ctx) i), ())) (((generatedRoot.w_76_0 ctx) i), (((generatedRoot.w_76_1 ctx) i), ())))
  t86 : (1) = 1
  h77_1 : (∀ i : Fin 1, parallel_generatedRoot_77 tape (path ++ [77, i.val]) params i (((generatedRoot.w_29_0 ctx) i), (((generatedRoot.w_30_0 ctx) i), ())) (((generatedRoot.w_77_0 ctx) i), (((generatedRoot.w_77_1 ctx) i), ())))
  t88 : (1) = 1
  h78_1 : (∀ i : Fin 1, parallel_generatedRoot_78 tape (path ++ [78, i.val]) params i (((generatedRoot.w_32_0 ctx) i), (((generatedRoot.w_33_0 ctx) i), ())) (((generatedRoot.w_78_0 ctx) i), (((generatedRoot.w_78_1 ctx) i), ())))
  t90 : (1) = 1
  h79_1 : (∀ i : Fin 1, parallel_generatedRoot_79 tape (path ++ [79, i.val]) params i (((generatedRoot.w_35_0 ctx) i), (((generatedRoot.w_36_0 ctx) i), ())) (((generatedRoot.w_79_0 ctx) i), (((generatedRoot.w_79_1 ctx) i), ())))
  t92 : (1) = 1
  h80_1 : (∀ i : Fin 1, parallel_generatedRoot_80 tape (path ++ [80, i.val]) params i (((generatedRoot.w_38_0 ctx) i), (((generatedRoot.w_39_0 ctx) i), ())) (((generatedRoot.w_80_0 ctx) i), (((generatedRoot.w_80_1 ctx) i), ())))
  t94 : (8) = 8
  h81_1 : (∀ i : Fin 8, parallel_generatedRoot_81 tape (path ++ [81, i.val]) params i (((generatedRoot.w_41_0 ctx) i), (((generatedRoot.w_42_0 ctx) i), ())) (((generatedRoot.w_81_0 ctx) i), (((generatedRoot.w_81_1 ctx) i), ())))

theorem generatedRoot.facts_2 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {ctx : generatedRoot.Ctx tape path params} {outputs : mxxMat0 × mxxMat0 × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat1) × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat1) × mxxMat0 × mxxMat0 × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat1) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat1) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × Unit}
    (h : generatedRoot.constraints_2 tape path params (generatedRoot.w_20_0 ctx) (generatedRoot.w_21_0 ctx) (generatedRoot.w_23_0 ctx) (generatedRoot.w_24_0 ctx) (generatedRoot.w_26_0 ctx) (generatedRoot.w_27_0 ctx) (generatedRoot.w_29_0 ctx) (generatedRoot.w_30_0 ctx) (generatedRoot.w_32_0 ctx) (generatedRoot.w_33_0 ctx) (generatedRoot.w_35_0 ctx) (generatedRoot.w_36_0 ctx) (generatedRoot.w_38_0 ctx) (generatedRoot.w_39_0 ctx) (generatedRoot.w_41_0 ctx) (generatedRoot.w_42_0 ctx) (generatedRoot.w_44_0 ctx) (generatedRoot.w_45_0 ctx) (generatedRoot.w_66_0 ctx) (generatedRoot.w_66_1 ctx) (generatedRoot.w_67_0 ctx) (generatedRoot.w_67_1 ctx) (generatedRoot.w_68_0 ctx) (generatedRoot.w_68_1 ctx) (generatedRoot.w_69_0 ctx) (generatedRoot.w_69_1 ctx) (generatedRoot.w_70_0 ctx) (generatedRoot.w_70_1 ctx) (generatedRoot.w_71_0 ctx) (generatedRoot.w_71_1 ctx) (generatedRoot.w_72_0 ctx) (generatedRoot.w_72_1 ctx) (generatedRoot.w_73_0 ctx) (generatedRoot.w_73_1 ctx) (generatedRoot.w_74_0 ctx) (generatedRoot.w_74_1 ctx) (generatedRoot.w_75_0 ctx) (generatedRoot.w_75_1 ctx) (generatedRoot.w_76_0 ctx) (generatedRoot.w_76_1 ctx) (generatedRoot.w_77_0 ctx) (generatedRoot.w_77_1 ctx) (generatedRoot.w_78_0 ctx) (generatedRoot.w_78_1 ctx) (generatedRoot.w_79_0 ctx) (generatedRoot.w_79_1 ctx) (generatedRoot.w_80_0 ctx) (generatedRoot.w_80_1 ctx) (generatedRoot.w_81_0 ctx) (generatedRoot.w_81_1 ctx)) :
    generatedRoot.Facts_2 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_3 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) (outputs : mxxMat0 × mxxMat0 × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat1) × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat1) × mxxMat0 × mxxMat0 × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat1) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat1) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × Unit) : Prop where
  t96 : (4) = 4
  h82_1 : (∀ i : Fin 4, parallel_generatedRoot_82 tape (path ++ [82, i.val]) params i (((generatedRoot.w_44_0 ctx) i), (((generatedRoot.w_45_0 ctx) i), ())) (((generatedRoot.w_82_0 ctx) i), (((generatedRoot.w_82_1 ctx) i), ())))
  hout : outputs = generatedRoot.outs ctx

theorem generatedRoot.facts_3 {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {ctx : generatedRoot.Ctx tape path params} {outputs : mxxMat0 × mxxMat0 × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat1) × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat1) × mxxMat0 × mxxMat0 × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat1) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat1) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × Unit}
    (h : generatedRoot.constraints_3 tape path params (generatedRoot.w_0_0 ctx) (generatedRoot.w_1_0 ctx) (generatedRoot.w_7_0 ctx) (generatedRoot.w_8_0 ctx) (generatedRoot.w_44_0 ctx) (generatedRoot.w_45_0 ctx) (generatedRoot.w_6_0 ctx) (generatedRoot.w_6_1 ctx) (generatedRoot.w_6_2 ctx) (generatedRoot.w_11_0 ctx) (generatedRoot.w_11_1 ctx) (generatedRoot.w_11_2 ctx) (generatedRoot.w_14_0 ctx) (generatedRoot.w_14_1 ctx) (generatedRoot.w_14_2 ctx) (generatedRoot.w_19_0 ctx) (generatedRoot.w_19_1 ctx) (generatedRoot.w_19_2 ctx) (generatedRoot.w_22_0 ctx) (generatedRoot.w_22_1 ctx) (generatedRoot.w_25_0 ctx) (generatedRoot.w_25_1 ctx) (generatedRoot.w_28_0 ctx) (generatedRoot.w_28_1 ctx) (generatedRoot.w_31_0 ctx) (generatedRoot.w_31_1 ctx) (generatedRoot.w_34_0 ctx) (generatedRoot.w_34_1 ctx) (generatedRoot.w_37_0 ctx) (generatedRoot.w_37_1 ctx) (generatedRoot.w_40_0 ctx) (generatedRoot.w_40_1 ctx) (generatedRoot.w_43_0 ctx) (generatedRoot.w_43_1 ctx) (generatedRoot.w_46_0 ctx) (generatedRoot.w_46_1 ctx) (generatedRoot.w_47_0 ctx) (generatedRoot.w_47_1 ctx) (generatedRoot.w_48_0 ctx) (generatedRoot.w_48_1 ctx) (generatedRoot.w_49_0 ctx) (generatedRoot.w_49_1 ctx) (generatedRoot.w_50_0 ctx) (generatedRoot.w_50_1 ctx) (generatedRoot.w_51_0 ctx) (generatedRoot.w_51_1 ctx) (generatedRoot.w_52_0 ctx) (generatedRoot.w_52_1 ctx) (generatedRoot.w_53_0 ctx) (generatedRoot.w_53_1 ctx) (generatedRoot.w_54_0 ctx) (generatedRoot.w_54_1 ctx) (generatedRoot.w_55_0 ctx) (generatedRoot.w_55_1 ctx) (generatedRoot.w_56_0 ctx) (generatedRoot.w_56_1 ctx) (generatedRoot.w_57_0 ctx) (generatedRoot.w_57_1 ctx) (generatedRoot.w_58_0 ctx) (generatedRoot.w_58_1 ctx) (generatedRoot.w_59_0 ctx) (generatedRoot.w_59_1 ctx) (generatedRoot.w_60_0 ctx) (generatedRoot.w_60_1 ctx) (generatedRoot.w_61_0 ctx) (generatedRoot.w_61_1 ctx) (generatedRoot.w_62_0 ctx) (generatedRoot.w_62_1 ctx) (generatedRoot.w_63_0 ctx) (generatedRoot.w_63_1 ctx) (generatedRoot.w_64_0 ctx) (generatedRoot.w_64_1 ctx) (generatedRoot.w_65_0 ctx) (generatedRoot.w_65_1 ctx) (generatedRoot.w_66_0 ctx) (generatedRoot.w_66_1 ctx) (generatedRoot.w_67_0 ctx) (generatedRoot.w_67_1 ctx) (generatedRoot.w_68_0 ctx) (generatedRoot.w_68_1 ctx) (generatedRoot.w_69_0 ctx) (generatedRoot.w_69_1 ctx) (generatedRoot.w_70_0 ctx) (generatedRoot.w_70_1 ctx) (generatedRoot.w_71_0 ctx) (generatedRoot.w_71_1 ctx) (generatedRoot.w_72_0 ctx) (generatedRoot.w_72_1 ctx) (generatedRoot.w_73_0 ctx) (generatedRoot.w_73_1 ctx) (generatedRoot.w_74_0 ctx) (generatedRoot.w_74_1 ctx) (generatedRoot.w_75_0 ctx) (generatedRoot.w_75_1 ctx) (generatedRoot.w_76_0 ctx) (generatedRoot.w_76_1 ctx) (generatedRoot.w_77_0 ctx) (generatedRoot.w_77_1 ctx) (generatedRoot.w_78_0 ctx) (generatedRoot.w_78_1 ctx) (generatedRoot.w_79_0 ctx) (generatedRoot.w_79_1 ctx) (generatedRoot.w_80_0 ctx) (generatedRoot.w_80_1 ctx) (generatedRoot.w_81_0 ctx) (generatedRoot.w_81_1 ctx) (generatedRoot.w_82_0 ctx) (generatedRoot.w_82_1 ctx) outputs) :
    generatedRoot.Facts_3 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure generatedRoot.Facts {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) (outputs : mxxMat0 × mxxMat0 × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat1) × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat1) × mxxMat0 × mxxMat0 × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat1) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat1) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × Unit) : Prop where
  c0 : generatedRoot.Facts_0 ctx
  c1 : generatedRoot.Facts_1 ctx
  c2 : generatedRoot.Facts_2 ctx
  c3 : generatedRoot.Facts_3 ctx outputs

theorem generatedRoot.facts {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx tape path params) {outputs : mxxMat0 × mxxMat0 × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat1) × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat1) × mxxMat0 × mxxMat0 × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat1) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat1) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × Unit}
    (h : generatedRoot.body tape path params ctx.inputs outputs ctx.witness) :
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

end Stage_import_0

end
