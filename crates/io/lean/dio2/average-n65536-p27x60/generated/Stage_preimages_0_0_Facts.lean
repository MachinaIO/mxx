import Stage_preimages_0_0

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000
set_option linter.unusedVariables false

-- The wire types repeat the ring modulus `X ^ n + 1`; unfolding it while checking a
-- projection against its field type expands it at the ring dimension.
attribute [local irreducible] Mxx.Primitives.negacyclicModulus

noncomputable section

namespace Stage_preimages_0_0

local notation "mxxMat0" => Mxx.Primitives.ExactMatrix modulus_0 65536 2 544
local notation "mxxMat1" => Mxx.Primitives.ExactMatrix modulus_0 65536 2 270
local notation "mxxMat2" => Mxx.Primitives.ExactMatrix modulus_0 65536 544 270

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Ctx (backend : MxxRuntime.BackendContext) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) where
  inputs : (generatedRoot.InputType)
  witness : generatedRoot.Witness

section
variable {backend : MxxRuntime.BackendContext} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params}
local notation "generatedRootCtx" => (generatedRoot.Ctx backend tape path params)

def generatedRoot.w_0_0  (ctx : generatedRootCtx) : MxxRuntime.TrapdoorValue (mxxMat0) Unit :=
  ctx.inputs.1

def generatedRoot.w_2_0  (ctx : generatedRootCtx) : Fin 16779536 → mxxMat1 :=
  ctx.inputs.2.1

def generatedRoot.w_4_0  (ctx : generatedRootCtx) : Fin 16779536 → mxxMat1 :=
  ctx.inputs.2.2.1

def generatedRoot.w_6_0  (ctx : generatedRootCtx) : Fin 16779536 → mxxMat1 :=
  ctx.inputs.2.2.2.1

def generatedRoot.w_8_0  (ctx : generatedRootCtx) : Fin 16779536 → mxxMat1 :=
  ctx.inputs.2.2.2.2.1

def generatedRoot.w_10_0  (ctx : generatedRootCtx) : Fin 16779536 → mxxMat1 :=
  ctx.inputs.2.2.2.2.2.1

def generatedRoot.w_12_0  (ctx : generatedRootCtx) : Fin 16779536 → mxxMat1 :=
  ctx.inputs.2.2.2.2.2.2.1

def generatedRoot.w_14_0  (ctx : generatedRootCtx) : Fin 16779536 → mxxMat1 :=
  ctx.inputs.2.2.2.2.2.2.2.1

def generatedRoot.w_16_0  (ctx : generatedRootCtx) : Fin 16779536 → mxxMat1 :=
  ctx.inputs.2.2.2.2.2.2.2.2.1

def generatedRoot.w_18_0  (ctx : generatedRootCtx) : Fin 16779536 → mxxMat1 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_20_0  (ctx : generatedRootCtx) : Fin 16779536 → mxxMat1 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_22_0  (ctx : generatedRootCtx) : Fin 16779536 → mxxMat1 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_24_0  (ctx : generatedRootCtx) : Fin 16779536 → mxxMat1 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_26_0  (ctx : generatedRootCtx) : Fin 16779536 → mxxMat1 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_28_0  (ctx : generatedRootCtx) : Fin 16779536 → mxxMat1 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_30_0  (ctx : generatedRootCtx) : Fin 16779536 → mxxMat1 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_32_0  (ctx : generatedRootCtx) : Fin 16779536 → mxxMat1 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_34_0  (ctx : generatedRootCtx) : Fin 16779536 → mxxMat1 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_36_0  (ctx : generatedRootCtx) : Fin 16779536 → mxxMat1 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_38_0  (ctx : generatedRootCtx) : Fin 16779536 → mxxMat1 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_40_0  (ctx : generatedRootCtx) : Fin 16779536 → mxxMat1 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_42_0  (ctx : generatedRootCtx) : Fin 16779536 → mxxMat1 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_44_0  (ctx : generatedRootCtx) : Fin 16779536 → mxxMat1 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_46_0  (ctx : generatedRootCtx) : Fin 16779536 → mxxMat1 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_48_0  (ctx : generatedRootCtx) : Fin 16779536 → mxxMat1 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_50_0  (ctx : generatedRootCtx) : Fin 16779536 → mxxMat1 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_52_0  (ctx : generatedRootCtx) : Fin 16779536 → mxxMat1 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_54_0  (ctx : generatedRootCtx) : Fin 16779536 → mxxMat1 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_1_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.trapdoorPublic (generatedRoot.w_0_0 ctx)

def generatedRoot.w_3_0  (ctx : generatedRootCtx) : Fin 16779536 → mxxMat2 :=
  ctx.witness.w_3_0

def generatedRoot.w_5_0  (ctx : generatedRootCtx) : Fin 16779536 → mxxMat2 :=
  ctx.witness.w_5_0

def generatedRoot.w_7_0  (ctx : generatedRootCtx) : Fin 16779536 → mxxMat2 :=
  ctx.witness.w_7_0

def generatedRoot.w_9_0  (ctx : generatedRootCtx) : Fin 16779536 → mxxMat2 :=
  ctx.witness.w_9_0

def generatedRoot.w_11_0  (ctx : generatedRootCtx) : Fin 16779536 → mxxMat2 :=
  ctx.witness.w_11_0

def generatedRoot.w_13_0  (ctx : generatedRootCtx) : Fin 16779536 → mxxMat2 :=
  ctx.witness.w_13_0

def generatedRoot.w_15_0  (ctx : generatedRootCtx) : Fin 16779536 → mxxMat2 :=
  ctx.witness.w_15_0

def generatedRoot.w_17_0  (ctx : generatedRootCtx) : Fin 16779536 → mxxMat2 :=
  ctx.witness.w_17_0

def generatedRoot.w_19_0  (ctx : generatedRootCtx) : Fin 16779536 → mxxMat2 :=
  ctx.witness.w_19_0

def generatedRoot.w_21_0  (ctx : generatedRootCtx) : Fin 16779536 → mxxMat2 :=
  ctx.witness.w_21_0

def generatedRoot.w_23_0  (ctx : generatedRootCtx) : Fin 16779536 → mxxMat2 :=
  ctx.witness.w_23_0

def generatedRoot.w_25_0  (ctx : generatedRootCtx) : Fin 16779536 → mxxMat2 :=
  ctx.witness.w_25_0

def generatedRoot.w_27_0  (ctx : generatedRootCtx) : Fin 16779536 → mxxMat2 :=
  ctx.witness.w_27_0

def generatedRoot.w_29_0  (ctx : generatedRootCtx) : Fin 16779536 → mxxMat2 :=
  ctx.witness.w_29_0

def generatedRoot.w_31_0  (ctx : generatedRootCtx) : Fin 16779536 → mxxMat2 :=
  ctx.witness.w_31_0

def generatedRoot.w_33_0  (ctx : generatedRootCtx) : Fin 16779536 → mxxMat2 :=
  ctx.witness.w_33_0

def generatedRoot.w_35_0  (ctx : generatedRootCtx) : Fin 16779536 → mxxMat2 :=
  ctx.witness.w_35_0

def generatedRoot.w_37_0  (ctx : generatedRootCtx) : Fin 16779536 → mxxMat2 :=
  ctx.witness.w_37_0

def generatedRoot.w_39_0  (ctx : generatedRootCtx) : Fin 16779536 → mxxMat2 :=
  ctx.witness.w_39_0

def generatedRoot.w_41_0  (ctx : generatedRootCtx) : Fin 16779536 → mxxMat2 :=
  ctx.witness.w_41_0

def generatedRoot.w_43_0  (ctx : generatedRootCtx) : Fin 16779536 → mxxMat2 :=
  ctx.witness.w_43_0

def generatedRoot.w_45_0  (ctx : generatedRootCtx) : Fin 16779536 → mxxMat2 :=
  ctx.witness.w_45_0

def generatedRoot.w_47_0  (ctx : generatedRootCtx) : Fin 16779536 → mxxMat2 :=
  ctx.witness.w_47_0

def generatedRoot.w_49_0  (ctx : generatedRootCtx) : Fin 16779536 → mxxMat2 :=
  ctx.witness.w_49_0

def generatedRoot.w_51_0  (ctx : generatedRootCtx) : Fin 16779536 → mxxMat2 :=
  ctx.witness.w_51_0

def generatedRoot.w_53_0  (ctx : generatedRootCtx) : Fin 16779536 → mxxMat2 :=
  ctx.witness.w_53_0

def generatedRoot.w_55_0  (ctx : generatedRootCtx) : Fin 16779536 → mxxMat2 :=
  ctx.witness.w_55_0

def generatedRoot.outs  (ctx : generatedRootCtx) : (generatedRoot.OutputType) :=
  ((generatedRoot.w_3_0 ctx), ((generatedRoot.w_5_0 ctx), ((generatedRoot.w_7_0 ctx), ((generatedRoot.w_9_0 ctx), ((generatedRoot.w_11_0 ctx), ((generatedRoot.w_13_0 ctx), ((generatedRoot.w_15_0 ctx), ((generatedRoot.w_17_0 ctx), ((generatedRoot.w_19_0 ctx), ((generatedRoot.w_21_0 ctx), ((generatedRoot.w_23_0 ctx), ((generatedRoot.w_25_0 ctx), ((generatedRoot.w_27_0 ctx), ((generatedRoot.w_29_0 ctx), ((generatedRoot.w_31_0 ctx), ((generatedRoot.w_33_0 ctx), ((generatedRoot.w_35_0 ctx), ((generatedRoot.w_37_0 ctx), ((generatedRoot.w_39_0 ctx), ((generatedRoot.w_41_0 ctx), ((generatedRoot.w_43_0 ctx), ((generatedRoot.w_45_0 ctx), ((generatedRoot.w_47_0 ctx), ((generatedRoot.w_49_0 ctx), ((generatedRoot.w_51_0 ctx), ((generatedRoot.w_53_0 ctx), ((generatedRoot.w_55_0 ctx), ())))))))))))))))))))))))))))

structure generatedRoot.Facts_0  (ctx : generatedRootCtx) : Prop where
  t0 : (16779536) = 16779536
  h3_0 : (∀ i : Fin 16779536, parallel_generatedRoot_3 backend tape (path ++ [3, i.val]) params i ((generatedRoot.w_1_0 ctx), ((generatedRoot.w_0_0 ctx), (((generatedRoot.w_2_0 ctx) i), ()))) ((generatedRoot.w_3_0 ctx) i))
  t2 : (16779536) = 16779536
  h5_0 : (∀ i : Fin 16779536, parallel_generatedRoot_5 backend tape (path ++ [5, i.val]) params i ((generatedRoot.w_1_0 ctx), ((generatedRoot.w_0_0 ctx), (((generatedRoot.w_4_0 ctx) i), ()))) ((generatedRoot.w_5_0 ctx) i))
  t4 : (16779536) = 16779536
  h7_0 : (∀ i : Fin 16779536, parallel_generatedRoot_7 backend tape (path ++ [7, i.val]) params i ((generatedRoot.w_1_0 ctx), ((generatedRoot.w_0_0 ctx), (((generatedRoot.w_6_0 ctx) i), ()))) ((generatedRoot.w_7_0 ctx) i))
  t6 : (16779536) = 16779536
  h9_0 : (∀ i : Fin 16779536, parallel_generatedRoot_9 backend tape (path ++ [9, i.val]) params i ((generatedRoot.w_1_0 ctx), ((generatedRoot.w_0_0 ctx), (((generatedRoot.w_8_0 ctx) i), ()))) ((generatedRoot.w_9_0 ctx) i))
  t8 : (16779536) = 16779536
  h11_0 : (∀ i : Fin 16779536, parallel_generatedRoot_11 backend tape (path ++ [11, i.val]) params i ((generatedRoot.w_1_0 ctx), ((generatedRoot.w_0_0 ctx), (((generatedRoot.w_10_0 ctx) i), ()))) ((generatedRoot.w_11_0 ctx) i))
  t10 : (16779536) = 16779536
  h13_0 : (∀ i : Fin 16779536, parallel_generatedRoot_13 backend tape (path ++ [13, i.val]) params i ((generatedRoot.w_1_0 ctx), ((generatedRoot.w_0_0 ctx), (((generatedRoot.w_12_0 ctx) i), ()))) ((generatedRoot.w_13_0 ctx) i))
  t12 : (16779536) = 16779536
  h15_0 : (∀ i : Fin 16779536, parallel_generatedRoot_15 backend tape (path ++ [15, i.val]) params i ((generatedRoot.w_1_0 ctx), ((generatedRoot.w_0_0 ctx), (((generatedRoot.w_14_0 ctx) i), ()))) ((generatedRoot.w_15_0 ctx) i))
  t14 : (16779536) = 16779536
  h17_0 : (∀ i : Fin 16779536, parallel_generatedRoot_17 backend tape (path ++ [17, i.val]) params i ((generatedRoot.w_1_0 ctx), ((generatedRoot.w_0_0 ctx), (((generatedRoot.w_16_0 ctx) i), ()))) ((generatedRoot.w_17_0 ctx) i))
  t16 : (16779536) = 16779536
  h19_0 : (∀ i : Fin 16779536, parallel_generatedRoot_19 backend tape (path ++ [19, i.val]) params i ((generatedRoot.w_1_0 ctx), ((generatedRoot.w_0_0 ctx), (((generatedRoot.w_18_0 ctx) i), ()))) ((generatedRoot.w_19_0 ctx) i))
  t18 : (16779536) = 16779536
  h21_0 : (∀ i : Fin 16779536, parallel_generatedRoot_21 backend tape (path ++ [21, i.val]) params i ((generatedRoot.w_1_0 ctx), ((generatedRoot.w_0_0 ctx), (((generatedRoot.w_20_0 ctx) i), ()))) ((generatedRoot.w_21_0 ctx) i))
  t20 : (16779536) = 16779536
  h23_0 : (∀ i : Fin 16779536, parallel_generatedRoot_23 backend tape (path ++ [23, i.val]) params i ((generatedRoot.w_1_0 ctx), ((generatedRoot.w_0_0 ctx), (((generatedRoot.w_22_0 ctx) i), ()))) ((generatedRoot.w_23_0 ctx) i))
  t22 : (16779536) = 16779536
  h25_0 : (∀ i : Fin 16779536, parallel_generatedRoot_25 backend tape (path ++ [25, i.val]) params i ((generatedRoot.w_1_0 ctx), ((generatedRoot.w_0_0 ctx), (((generatedRoot.w_24_0 ctx) i), ()))) ((generatedRoot.w_25_0 ctx) i))
  t24 : (16779536) = 16779536
  h27_0 : (∀ i : Fin 16779536, parallel_generatedRoot_27 backend tape (path ++ [27, i.val]) params i ((generatedRoot.w_1_0 ctx), ((generatedRoot.w_0_0 ctx), (((generatedRoot.w_26_0 ctx) i), ()))) ((generatedRoot.w_27_0 ctx) i))
  t26 : (16779536) = 16779536
  h29_0 : (∀ i : Fin 16779536, parallel_generatedRoot_29 backend tape (path ++ [29, i.val]) params i ((generatedRoot.w_1_0 ctx), ((generatedRoot.w_0_0 ctx), (((generatedRoot.w_28_0 ctx) i), ()))) ((generatedRoot.w_29_0 ctx) i))
  t28 : (16779536) = 16779536
  h31_0 : (∀ i : Fin 16779536, parallel_generatedRoot_31 backend tape (path ++ [31, i.val]) params i ((generatedRoot.w_1_0 ctx), ((generatedRoot.w_0_0 ctx), (((generatedRoot.w_30_0 ctx) i), ()))) ((generatedRoot.w_31_0 ctx) i))
  t30 : (16779536) = 16779536
  h33_0 : (∀ i : Fin 16779536, parallel_generatedRoot_33 backend tape (path ++ [33, i.val]) params i ((generatedRoot.w_1_0 ctx), ((generatedRoot.w_0_0 ctx), (((generatedRoot.w_32_0 ctx) i), ()))) ((generatedRoot.w_33_0 ctx) i))

theorem generatedRoot.facts_0  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_0 backend tape path params (generatedRoot.w_0_0 ctx) (generatedRoot.w_2_0 ctx) (generatedRoot.w_4_0 ctx) (generatedRoot.w_6_0 ctx) (generatedRoot.w_8_0 ctx) (generatedRoot.w_10_0 ctx) (generatedRoot.w_12_0 ctx) (generatedRoot.w_14_0 ctx) (generatedRoot.w_16_0 ctx) (generatedRoot.w_18_0 ctx) (generatedRoot.w_20_0 ctx) (generatedRoot.w_22_0 ctx) (generatedRoot.w_24_0 ctx) (generatedRoot.w_26_0 ctx) (generatedRoot.w_28_0 ctx) (generatedRoot.w_30_0 ctx) (generatedRoot.w_32_0 ctx) (generatedRoot.w_1_0 ctx) (generatedRoot.w_3_0 ctx) (generatedRoot.w_5_0 ctx) (generatedRoot.w_7_0 ctx) (generatedRoot.w_9_0 ctx) (generatedRoot.w_11_0 ctx) (generatedRoot.w_13_0 ctx) (generatedRoot.w_15_0 ctx) (generatedRoot.w_17_0 ctx) (generatedRoot.w_19_0 ctx) (generatedRoot.w_21_0 ctx) (generatedRoot.w_23_0 ctx) (generatedRoot.w_25_0 ctx) (generatedRoot.w_27_0 ctx) (generatedRoot.w_29_0 ctx) (generatedRoot.w_31_0 ctx) (generatedRoot.w_33_0 ctx)) :
    generatedRoot.Facts_0 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_1  (ctx : generatedRootCtx) (outputs : (generatedRoot.OutputType)) : Prop where
  t32 : (16779536) = 16779536
  h35_0 : (∀ i : Fin 16779536, parallel_generatedRoot_35 backend tape (path ++ [35, i.val]) params i ((generatedRoot.w_1_0 ctx), ((generatedRoot.w_0_0 ctx), (((generatedRoot.w_34_0 ctx) i), ()))) ((generatedRoot.w_35_0 ctx) i))
  t34 : (16779536) = 16779536
  h37_0 : (∀ i : Fin 16779536, parallel_generatedRoot_37 backend tape (path ++ [37, i.val]) params i ((generatedRoot.w_1_0 ctx), ((generatedRoot.w_0_0 ctx), (((generatedRoot.w_36_0 ctx) i), ()))) ((generatedRoot.w_37_0 ctx) i))
  t36 : (16779536) = 16779536
  h39_0 : (∀ i : Fin 16779536, parallel_generatedRoot_39 backend tape (path ++ [39, i.val]) params i ((generatedRoot.w_1_0 ctx), ((generatedRoot.w_0_0 ctx), (((generatedRoot.w_38_0 ctx) i), ()))) ((generatedRoot.w_39_0 ctx) i))
  t38 : (16779536) = 16779536
  h41_0 : (∀ i : Fin 16779536, parallel_generatedRoot_41 backend tape (path ++ [41, i.val]) params i ((generatedRoot.w_1_0 ctx), ((generatedRoot.w_0_0 ctx), (((generatedRoot.w_40_0 ctx) i), ()))) ((generatedRoot.w_41_0 ctx) i))
  t40 : (16779536) = 16779536
  h43_0 : (∀ i : Fin 16779536, parallel_generatedRoot_43 backend tape (path ++ [43, i.val]) params i ((generatedRoot.w_1_0 ctx), ((generatedRoot.w_0_0 ctx), (((generatedRoot.w_42_0 ctx) i), ()))) ((generatedRoot.w_43_0 ctx) i))
  t42 : (16779536) = 16779536
  h45_0 : (∀ i : Fin 16779536, parallel_generatedRoot_45 backend tape (path ++ [45, i.val]) params i ((generatedRoot.w_1_0 ctx), ((generatedRoot.w_0_0 ctx), (((generatedRoot.w_44_0 ctx) i), ()))) ((generatedRoot.w_45_0 ctx) i))
  t44 : (16779536) = 16779536
  h47_0 : (∀ i : Fin 16779536, parallel_generatedRoot_47 backend tape (path ++ [47, i.val]) params i ((generatedRoot.w_1_0 ctx), ((generatedRoot.w_0_0 ctx), (((generatedRoot.w_46_0 ctx) i), ()))) ((generatedRoot.w_47_0 ctx) i))
  t46 : (16779536) = 16779536
  h49_0 : (∀ i : Fin 16779536, parallel_generatedRoot_49 backend tape (path ++ [49, i.val]) params i ((generatedRoot.w_1_0 ctx), ((generatedRoot.w_0_0 ctx), (((generatedRoot.w_48_0 ctx) i), ()))) ((generatedRoot.w_49_0 ctx) i))
  t48 : (16779536) = 16779536
  h51_0 : (∀ i : Fin 16779536, parallel_generatedRoot_51 backend tape (path ++ [51, i.val]) params i ((generatedRoot.w_1_0 ctx), ((generatedRoot.w_0_0 ctx), (((generatedRoot.w_50_0 ctx) i), ()))) ((generatedRoot.w_51_0 ctx) i))
  t50 : (16779536) = 16779536
  h53_0 : (∀ i : Fin 16779536, parallel_generatedRoot_53 backend tape (path ++ [53, i.val]) params i ((generatedRoot.w_1_0 ctx), ((generatedRoot.w_0_0 ctx), (((generatedRoot.w_52_0 ctx) i), ()))) ((generatedRoot.w_53_0 ctx) i))
  t52 : (16779536) = 16779536
  h55_0 : (∀ i : Fin 16779536, parallel_generatedRoot_55 backend tape (path ++ [55, i.val]) params i ((generatedRoot.w_1_0 ctx), ((generatedRoot.w_0_0 ctx), (((generatedRoot.w_54_0 ctx) i), ()))) ((generatedRoot.w_55_0 ctx) i))
  hout : outputs = generatedRoot.outs ctx

theorem generatedRoot.facts_1  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_1 backend tape path params (generatedRoot.w_0_0 ctx) (generatedRoot.w_34_0 ctx) (generatedRoot.w_36_0 ctx) (generatedRoot.w_38_0 ctx) (generatedRoot.w_40_0 ctx) (generatedRoot.w_42_0 ctx) (generatedRoot.w_44_0 ctx) (generatedRoot.w_46_0 ctx) (generatedRoot.w_48_0 ctx) (generatedRoot.w_50_0 ctx) (generatedRoot.w_52_0 ctx) (generatedRoot.w_54_0 ctx) (generatedRoot.w_1_0 ctx) (generatedRoot.w_3_0 ctx) (generatedRoot.w_5_0 ctx) (generatedRoot.w_7_0 ctx) (generatedRoot.w_9_0 ctx) (generatedRoot.w_11_0 ctx) (generatedRoot.w_13_0 ctx) (generatedRoot.w_15_0 ctx) (generatedRoot.w_17_0 ctx) (generatedRoot.w_19_0 ctx) (generatedRoot.w_21_0 ctx) (generatedRoot.w_23_0 ctx) (generatedRoot.w_25_0 ctx) (generatedRoot.w_27_0 ctx) (generatedRoot.w_29_0 ctx) (generatedRoot.w_31_0 ctx) (generatedRoot.w_33_0 ctx) (generatedRoot.w_35_0 ctx) (generatedRoot.w_37_0 ctx) (generatedRoot.w_39_0 ctx) (generatedRoot.w_41_0 ctx) (generatedRoot.w_43_0 ctx) (generatedRoot.w_45_0 ctx) (generatedRoot.w_47_0 ctx) (generatedRoot.w_49_0 ctx) (generatedRoot.w_51_0 ctx) (generatedRoot.w_53_0 ctx) (generatedRoot.w_55_0 ctx) outputs) :
    generatedRoot.Facts_1 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts  (ctx : generatedRootCtx) (outputs : (generatedRoot.OutputType)) : Prop where
  c0 : generatedRoot.Facts_0 ctx
  c1 : generatedRoot.Facts_1 ctx outputs

theorem generatedRoot.facts  (ctx : generatedRootCtx) {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.body backend tape path params ctx.inputs outputs ctx.witness) :
    generatedRoot.Facts ctx outputs :=
  have t0 := h
  have r0 := generatedRoot.facts_0 (ctx := ctx) (outputs := outputs) t0.1
  have t1 := t0.2
  have r1 := generatedRoot.facts_1 (ctx := ctx) (outputs := outputs) t1
  ⟨r0, r1⟩

end

end Stage_preimages_0_0

end
