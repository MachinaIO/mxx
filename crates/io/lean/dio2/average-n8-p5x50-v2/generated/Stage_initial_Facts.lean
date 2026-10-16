import Stage_initial

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000
set_option linter.unusedVariables false

-- The wire types repeat the ring modulus `X ^ n + 1`; unfolding it while checking a
-- projection against its field type expands it at the ring dimension.
attribute [local irreducible] Mxx.Primitives.negacyclicModulus

noncomputable section

namespace Stage_initial

local notation "mxxMat0" => Mxx.Primitives.ExactMatrix modulus_0 8 1 1
local notation "mxxMat1" => Mxx.Primitives.ExactMatrix modulus_0 8 2 54
local notation "mxxMat2" => Mxx.Primitives.ExactMatrix modulus_0 8 1 2
local notation "mxxMat3" => Mxx.Primitives.ExactMatrix modulus_0 8 1 54
local notation "mxxMat4" => Mxx.Primitives.ExactMatrix modulus_0 8 1 25

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) where
  inputs : mxxMat0 × mxxMat1 × ByteArray × (Fin 5 → Int) × mxxMat0 × Int × Unit
  witness : generatedRoot.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params}
local notation "generatedRootCtx" => (generatedRoot.Ctx backend hashModel tape path params)

def generatedRoot.w_0_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.inputs.1

def generatedRoot.w_3_0  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.inputs.2.1

def generatedRoot.w_9_0  (ctx : generatedRootCtx) : ByteArray :=
  ctx.inputs.2.2.1

def generatedRoot.w_28_0  (ctx : generatedRootCtx) : Fin 5 → Int :=
  ctx.inputs.2.2.2.1

def generatedRoot.w_39_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.1

def generatedRoot.w_69_0  (ctx : generatedRootCtx) : Int :=
  ctx.inputs.2.2.2.2.2.1

def generatedRoot.w_1_0  (ctx : generatedRootCtx) : mxxMat0 :=
  (1 : mxxMat0)

def generatedRoot.w_2_0  (ctx : generatedRootCtx) : mxxMat2 :=
  ctx.witness.w_2_0

def generatedRoot.w_4_0  (ctx : generatedRootCtx) : mxxMat3 :=
  MxxRuntime.matrixMul (generatedRoot.w_2_0 ctx) (generatedRoot.w_3_0 ctx)

def generatedRoot.w_5_0  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_5_0

def generatedRoot.w_6_0  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_6_0

def generatedRoot.w_7_0  (ctx : generatedRootCtx) : mxxMat3 :=
  MxxRuntime.matrixSub (generatedRoot.w_5_0 ctx) (generatedRoot.w_6_0 ctx)

def generatedRoot.w_8_0  (ctx : generatedRootCtx) : mxxMat3 :=
  MxxRuntime.matrixAdd (generatedRoot.w_4_0 ctx) (generatedRoot.w_7_0 ctx)

def generatedRoot.w_10_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat4 :=
  ctx.witness.w_10_0

def generatedRoot.w_11_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat4 :=
  ctx.witness.w_11_0

def generatedRoot.w_12_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat4 :=
  ctx.witness.w_12_0

def generatedRoot.w_13_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat4 :=
  ctx.witness.w_13_0

def generatedRoot.w_14_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat4 :=
  ctx.witness.w_14_0

def generatedRoot.w_15_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat4 :=
  ctx.witness.w_15_0

def generatedRoot.w_16_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat4 :=
  ctx.witness.w_16_0

def generatedRoot.w_17_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat4 :=
  ctx.witness.w_17_0

def generatedRoot.w_18_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat4 :=
  ctx.witness.w_18_0

def generatedRoot.w_19_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat4 :=
  ctx.witness.w_19_0

def generatedRoot.w_20_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat4 :=
  ctx.witness.w_20_0

def generatedRoot.w_21_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat4 :=
  ctx.witness.w_21_0

def generatedRoot.w_22_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat4 :=
  ctx.witness.w_22_0

def generatedRoot.w_23_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat4 :=
  ctx.witness.w_23_0

def generatedRoot.w_24_0  (ctx : generatedRootCtx) : Fin 8 → mxxMat4 :=
  ctx.witness.w_24_0

def generatedRoot.w_25_0  (ctx : generatedRootCtx) : Fin 8 → mxxMat4 :=
  ctx.witness.w_25_0

def generatedRoot.w_26_0  (ctx : generatedRootCtx) : Fin 4 → mxxMat4 :=
  ctx.witness.w_26_0

def generatedRoot.w_27_0  (ctx : generatedRootCtx) : Fin 4 → mxxMat4 :=
  ctx.witness.w_27_0

def generatedRoot.w_29_0  (ctx : generatedRootCtx) : Fin 2 → mxxMat4 :=
  ctx.witness.w_29_0

def generatedRoot.w_30_0  (ctx : generatedRootCtx) : Fin 2 → mxxMat4 :=
  ctx.witness.w_30_0

def generatedRoot.w_31_0  (ctx : generatedRootCtx) : Fin 4 → mxxMat4 :=
  ctx.witness.w_31_0

def generatedRoot.w_32_0  (ctx : generatedRootCtx) : Fin 4 → mxxMat4 :=
  ctx.witness.w_32_0

def generatedRoot.w_33_0  (ctx : generatedRootCtx) : mxxMat4 :=
  ctx.witness.w_33_0

def generatedRoot.w_34_0  (ctx : generatedRootCtx) : mxxMat4 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_0_0 ctx) (generatedRoot.w_33_0 ctx)

def generatedRoot.w_35_0  (ctx : generatedRootCtx) : mxxMat4 :=
  ctx.witness.w_35_0

def generatedRoot.w_36_0  (ctx : generatedRootCtx) : mxxMat4 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_1_0 ctx) (generatedRoot.w_35_0 ctx)

def generatedRoot.w_37_0  (ctx : generatedRootCtx) : mxxMat4 :=
  MxxRuntime.matrixAdd (generatedRoot.w_34_0 ctx) (generatedRoot.w_36_0 ctx)

def generatedRoot.w_38_0  (ctx : generatedRootCtx) : mxxMat0 :=
  (1 : mxxMat0)

def generatedRoot.w_40_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_38_0 ctx) (generatedRoot.w_39_0 ctx)

def generatedRoot.w_41_0  (ctx : generatedRootCtx) : mxxMat4 :=
  ctx.witness.w_41_0

def generatedRoot.w_42_0  (ctx : generatedRootCtx) : mxxMat4 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_40_0 ctx) (generatedRoot.w_41_0 ctx)

def generatedRoot.w_43_0  (ctx : generatedRootCtx) : mxxMat4 :=
  MxxRuntime.matrixSub (generatedRoot.w_37_0 ctx) (generatedRoot.w_42_0 ctx)

def generatedRoot.w_44_0  (ctx : generatedRootCtx) : mxxMat4 :=
  ctx.witness.w_44_0

def generatedRoot.w_45_0  (ctx : generatedRootCtx) : mxxMat4 :=
  ctx.witness.w_45_0

def generatedRoot.w_46_0  (ctx : generatedRootCtx) : mxxMat4 :=
  MxxRuntime.matrixSub (generatedRoot.w_44_0 ctx) (generatedRoot.w_45_0 ctx)

def generatedRoot.w_47_0  (ctx : generatedRootCtx) : mxxMat4 :=
  MxxRuntime.matrixAdd (generatedRoot.w_43_0 ctx) (generatedRoot.w_46_0 ctx)

def generatedRoot.w_48_0  (ctx : generatedRootCtx) : mxxMat4 :=
  ctx.witness.w_48_0

def generatedRoot.w_49_0  (ctx : generatedRootCtx) : mxxMat4 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_0_0 ctx) (generatedRoot.w_48_0 ctx)

def generatedRoot.w_50_0  (ctx : generatedRootCtx) : mxxMat4 :=
  ctx.witness.w_50_0

def generatedRoot.w_51_0  (ctx : generatedRootCtx) : mxxMat4 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_1_0 ctx) (generatedRoot.w_50_0 ctx)

def generatedRoot.w_52_0  (ctx : generatedRootCtx) : mxxMat4 :=
  MxxRuntime.matrixAdd (generatedRoot.w_49_0 ctx) (generatedRoot.w_51_0 ctx)

def generatedRoot.w_53_0  (ctx : generatedRootCtx) : mxxMat0 :=
  (1 : mxxMat0)

def generatedRoot.w_54_0  (ctx : generatedRootCtx) : mxxMat0 :=
  (1 : mxxMat0)

def generatedRoot.w_55_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_53_0 ctx) (generatedRoot.w_54_0 ctx)

def generatedRoot.w_56_0  (ctx : generatedRootCtx) : mxxMat4 :=
  ctx.witness.w_56_0

def generatedRoot.w_57_0  (ctx : generatedRootCtx) : mxxMat4 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_55_0 ctx) (generatedRoot.w_56_0 ctx)

def generatedRoot.w_58_0  (ctx : generatedRootCtx) : mxxMat4 :=
  MxxRuntime.matrixSub (generatedRoot.w_52_0 ctx) (generatedRoot.w_57_0 ctx)

def generatedRoot.w_59_0  (ctx : generatedRootCtx) : mxxMat4 :=
  ctx.witness.w_59_0

def generatedRoot.w_60_0  (ctx : generatedRootCtx) : mxxMat4 :=
  ctx.witness.w_60_0

def generatedRoot.w_61_0  (ctx : generatedRootCtx) : mxxMat4 :=
  MxxRuntime.matrixSub (generatedRoot.w_59_0 ctx) (generatedRoot.w_60_0 ctx)

def generatedRoot.w_62_0  (ctx : generatedRootCtx) : mxxMat4 :=
  MxxRuntime.matrixAdd (generatedRoot.w_58_0 ctx) (generatedRoot.w_61_0 ctx)

def generatedRoot.w_63_0  (ctx : generatedRootCtx) : mxxMat4 :=
  ctx.witness.w_63_0

def generatedRoot.w_64_0  (ctx : generatedRootCtx) : mxxMat4 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_0_0 ctx) (generatedRoot.w_63_0 ctx)

def generatedRoot.w_65_0  (ctx : generatedRootCtx) : mxxMat4 :=
  ctx.witness.w_65_0

def generatedRoot.w_66_0  (ctx : generatedRootCtx) : mxxMat4 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_1_0 ctx) (generatedRoot.w_65_0 ctx)

def generatedRoot.w_67_0  (ctx : generatedRootCtx) : mxxMat4 :=
  MxxRuntime.matrixAdd (generatedRoot.w_64_0 ctx) (generatedRoot.w_66_0 ctx)

def generatedRoot.w_68_0  (ctx : generatedRootCtx) : mxxMat0 :=
  (1 : mxxMat0)

def generatedRoot.w_70_0  (ctx : generatedRootCtx) : Int :=
  2

def generatedRoot.w_71_0  (ctx : generatedRootCtx) : Int :=
  ((generatedRoot.w_69_0 ctx) * (generatedRoot.w_70_0 ctx))

def generatedRoot.w_72_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.multiplyMonomial (generatedRoot.w_68_0 ctx) (generatedRoot.w_71_0 ctx)

def generatedRoot.w_73_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_72_0 ctx) (generatedRoot.w_39_0 ctx)

def generatedRoot.w_74_0  (ctx : generatedRootCtx) : mxxMat4 :=
  ctx.witness.w_74_0

def generatedRoot.w_75_0  (ctx : generatedRootCtx) : mxxMat4 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_73_0 ctx) (generatedRoot.w_74_0 ctx)

def generatedRoot.w_76_0  (ctx : generatedRootCtx) : mxxMat4 :=
  MxxRuntime.matrixSub (generatedRoot.w_67_0 ctx) (generatedRoot.w_75_0 ctx)

def generatedRoot.w_77_0  (ctx : generatedRootCtx) : mxxMat4 :=
  ctx.witness.w_77_0

def generatedRoot.w_78_0  (ctx : generatedRootCtx) : mxxMat4 :=
  ctx.witness.w_78_0

def generatedRoot.w_79_0  (ctx : generatedRootCtx) : mxxMat4 :=
  MxxRuntime.matrixSub (generatedRoot.w_77_0 ctx) (generatedRoot.w_78_0 ctx)

def generatedRoot.w_80_0  (ctx : generatedRootCtx) : mxxMat4 :=
  MxxRuntime.matrixAdd (generatedRoot.w_76_0 ctx) (generatedRoot.w_79_0 ctx)

def generatedRoot.w_81_0  (ctx : generatedRootCtx) : mxxMat4 :=
  ctx.witness.w_81_0

def generatedRoot.w_82_0  (ctx : generatedRootCtx) : mxxMat4 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_0_0 ctx) (generatedRoot.w_81_0 ctx)

def generatedRoot.w_83_0  (ctx : generatedRootCtx) : mxxMat4 :=
  ctx.witness.w_83_0

def generatedRoot.w_84_0  (ctx : generatedRootCtx) : mxxMat4 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_1_0 ctx) (generatedRoot.w_83_0 ctx)

def generatedRoot.w_85_0  (ctx : generatedRootCtx) : mxxMat4 :=
  MxxRuntime.matrixAdd (generatedRoot.w_82_0 ctx) (generatedRoot.w_84_0 ctx)

def generatedRoot.w_86_0  (ctx : generatedRootCtx) : mxxMat0 :=
  (1 : mxxMat0)

def generatedRoot.w_87_0  (ctx : generatedRootCtx) : Int :=
  2

def generatedRoot.w_88_0  (ctx : generatedRootCtx) : Int :=
  ((generatedRoot.w_69_0 ctx) * (generatedRoot.w_87_0 ctx))

def generatedRoot.w_89_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.multiplyMonomial (generatedRoot.w_86_0 ctx) (generatedRoot.w_88_0 ctx)

def generatedRoot.w_90_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_89_0 ctx) (generatedRoot.w_54_0 ctx)

def generatedRoot.w_91_0  (ctx : generatedRootCtx) : mxxMat4 :=
  ctx.witness.w_91_0

def generatedRoot.w_92_0  (ctx : generatedRootCtx) : mxxMat4 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_90_0 ctx) (generatedRoot.w_91_0 ctx)

def generatedRoot.w_93_0  (ctx : generatedRootCtx) : mxxMat4 :=
  MxxRuntime.matrixSub (generatedRoot.w_85_0 ctx) (generatedRoot.w_92_0 ctx)

def generatedRoot.w_94_0  (ctx : generatedRootCtx) : mxxMat4 :=
  ctx.witness.w_94_0

def generatedRoot.w_95_0  (ctx : generatedRootCtx) : mxxMat4 :=
  ctx.witness.w_95_0

def generatedRoot.w_96_0  (ctx : generatedRootCtx) : mxxMat4 :=
  MxxRuntime.matrixSub (generatedRoot.w_94_0 ctx) (generatedRoot.w_95_0 ctx)

def generatedRoot.w_97_0  (ctx : generatedRootCtx) : mxxMat4 :=
  MxxRuntime.matrixAdd (generatedRoot.w_93_0 ctx) (generatedRoot.w_96_0 ctx)

def generatedRoot.outs  (ctx : generatedRootCtx) : (generatedRoot.OutputType) :=
  ((generatedRoot.w_8_0 ctx), ((generatedRoot.w_10_0 ctx), ((generatedRoot.w_11_0 ctx), ((generatedRoot.w_12_0 ctx), ((generatedRoot.w_13_0 ctx), ((generatedRoot.w_14_0 ctx), ((generatedRoot.w_15_0 ctx), ((generatedRoot.w_16_0 ctx), ((generatedRoot.w_17_0 ctx), ((generatedRoot.w_18_0 ctx), ((generatedRoot.w_19_0 ctx), ((generatedRoot.w_20_0 ctx), ((generatedRoot.w_21_0 ctx), ((generatedRoot.w_22_0 ctx), ((generatedRoot.w_23_0 ctx), ((generatedRoot.w_24_0 ctx), ((generatedRoot.w_25_0 ctx), ((generatedRoot.w_26_0 ctx), ((generatedRoot.w_27_0 ctx), ((generatedRoot.w_29_0 ctx), ((generatedRoot.w_30_0 ctx), ((generatedRoot.w_31_0 ctx), ((generatedRoot.w_32_0 ctx), ((generatedRoot.w_47_0 ctx), ((generatedRoot.w_62_0 ctx), ((generatedRoot.w_80_0 ctx), ((generatedRoot.w_97_0 ctx), ())))))))))))))))))))))))))))

structure generatedRoot.Facts_0  (ctx : generatedRootCtx) : Prop where
  h2_0 : MxxRuntime.concatColumns (generatedRoot.w_0_0 ctx) (generatedRoot.w_1_0 ctx) (generatedRoot.w_2_0 ctx)
  h5_0 : MxxRuntime.uniformIntervalSampleAt tape (path ++ [5]) (0) (1) (generatedRoot.w_5_0 ctx)
  h6_0 : MxxRuntime.uniformIntervalSampleAt tape (path ++ [6]) (0) (1) (generatedRoot.w_6_0 ctx)
  t3 : (1) = 1
  h10_0 : (∀ i : Fin 1, parallel_generatedRoot_10 backend hashModel tape (path ++ [10, i.val]) params i ((generatedRoot.w_0_0 ctx), ((generatedRoot.w_9_0 ctx), ((generatedRoot.w_1_0 ctx), ()))) ((generatedRoot.w_10_0 ctx) i))
  t5 : (1) = 1
  h11_0 : (∀ i : Fin 1, parallel_generatedRoot_11 backend hashModel tape (path ++ [11, i.val]) params i ((generatedRoot.w_0_0 ctx), ((generatedRoot.w_9_0 ctx), ((generatedRoot.w_1_0 ctx), ()))) ((generatedRoot.w_11_0 ctx) i))
  t7 : (1) = 1
  h12_0 : (∀ i : Fin 1, parallel_generatedRoot_12 backend hashModel tape (path ++ [12, i.val]) params i ((generatedRoot.w_0_0 ctx), ((generatedRoot.w_9_0 ctx), ((generatedRoot.w_1_0 ctx), ()))) ((generatedRoot.w_12_0 ctx) i))
  t9 : (1) = 1
  h13_0 : (∀ i : Fin 1, parallel_generatedRoot_13 backend hashModel tape (path ++ [13, i.val]) params i ((generatedRoot.w_0_0 ctx), ((generatedRoot.w_9_0 ctx), ((generatedRoot.w_1_0 ctx), ()))) ((generatedRoot.w_13_0 ctx) i))
  t11 : (1) = 1
  h14_0 : (∀ i : Fin 1, parallel_generatedRoot_14 backend hashModel tape (path ++ [14, i.val]) params i ((generatedRoot.w_0_0 ctx), ((generatedRoot.w_9_0 ctx), ((generatedRoot.w_1_0 ctx), ()))) ((generatedRoot.w_14_0 ctx) i))
  t13 : (1) = 1
  h15_0 : (∀ i : Fin 1, parallel_generatedRoot_15 backend hashModel tape (path ++ [15, i.val]) params i ((generatedRoot.w_0_0 ctx), ((generatedRoot.w_9_0 ctx), ((generatedRoot.w_1_0 ctx), ()))) ((generatedRoot.w_15_0 ctx) i))
  t15 : (1) = 1
  h16_0 : (∀ i : Fin 1, parallel_generatedRoot_16 backend hashModel tape (path ++ [16, i.val]) params i ((generatedRoot.w_0_0 ctx), ((generatedRoot.w_9_0 ctx), ((generatedRoot.w_1_0 ctx), ()))) ((generatedRoot.w_16_0 ctx) i))
  t17 : (1) = 1
  h17_0 : (∀ i : Fin 1, parallel_generatedRoot_17 backend hashModel tape (path ++ [17, i.val]) params i ((generatedRoot.w_0_0 ctx), ((generatedRoot.w_9_0 ctx), ((generatedRoot.w_1_0 ctx), ()))) ((generatedRoot.w_17_0 ctx) i))
  t19 : (1) = 1
  h18_0 : (∀ i : Fin 1, parallel_generatedRoot_18 backend hashModel tape (path ++ [18, i.val]) params i ((generatedRoot.w_0_0 ctx), ((generatedRoot.w_9_0 ctx), ((generatedRoot.w_1_0 ctx), ()))) ((generatedRoot.w_18_0 ctx) i))
  t21 : (1) = 1
  h19_0 : (∀ i : Fin 1, parallel_generatedRoot_19 backend hashModel tape (path ++ [19, i.val]) params i ((generatedRoot.w_0_0 ctx), ((generatedRoot.w_9_0 ctx), ((generatedRoot.w_1_0 ctx), ()))) ((generatedRoot.w_19_0 ctx) i))
  t23 : (1) = 1
  h20_0 : (∀ i : Fin 1, parallel_generatedRoot_20 backend hashModel tape (path ++ [20, i.val]) params i ((generatedRoot.w_0_0 ctx), ((generatedRoot.w_9_0 ctx), ((generatedRoot.w_1_0 ctx), ()))) ((generatedRoot.w_20_0 ctx) i))
  t25 : (1) = 1
  h21_0 : (∀ i : Fin 1, parallel_generatedRoot_21 backend hashModel tape (path ++ [21, i.val]) params i ((generatedRoot.w_0_0 ctx), ((generatedRoot.w_9_0 ctx), ((generatedRoot.w_1_0 ctx), ()))) ((generatedRoot.w_21_0 ctx) i))
  t27 : (1) = 1
  h22_0 : (∀ i : Fin 1, parallel_generatedRoot_22 backend hashModel tape (path ++ [22, i.val]) params i ((generatedRoot.w_0_0 ctx), ((generatedRoot.w_9_0 ctx), ((generatedRoot.w_1_0 ctx), ()))) ((generatedRoot.w_22_0 ctx) i))
  t29 : (1) = 1
  h23_0 : (∀ i : Fin 1, parallel_generatedRoot_23 backend hashModel tape (path ++ [23, i.val]) params i ((generatedRoot.w_0_0 ctx), ((generatedRoot.w_9_0 ctx), ((generatedRoot.w_1_0 ctx), ()))) ((generatedRoot.w_23_0 ctx) i))
  t31 : (8) = 8

theorem generatedRoot.facts_0  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_0 backend hashModel tape path params (generatedRoot.w_0_0 ctx) (generatedRoot.w_9_0 ctx) (generatedRoot.w_1_0 ctx) (generatedRoot.w_2_0 ctx) (generatedRoot.w_5_0 ctx) (generatedRoot.w_6_0 ctx) (generatedRoot.w_10_0 ctx) (generatedRoot.w_11_0 ctx) (generatedRoot.w_12_0 ctx) (generatedRoot.w_13_0 ctx) (generatedRoot.w_14_0 ctx) (generatedRoot.w_15_0 ctx) (generatedRoot.w_16_0 ctx) (generatedRoot.w_17_0 ctx) (generatedRoot.w_18_0 ctx) (generatedRoot.w_19_0 ctx) (generatedRoot.w_20_0 ctx) (generatedRoot.w_21_0 ctx) (generatedRoot.w_22_0 ctx) (generatedRoot.w_23_0 ctx)) :
    generatedRoot.Facts_0 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_1  (ctx : generatedRootCtx) : Prop where
  h24_0 : (∀ i : Fin 8, parallel_generatedRoot_24 backend hashModel tape (path ++ [24, i.val]) params i ((generatedRoot.w_0_0 ctx), ((generatedRoot.w_9_0 ctx), ((generatedRoot.w_1_0 ctx), ()))) ((generatedRoot.w_24_0 ctx) i))
  t33 : (8) = 8
  h25_0 : (∀ i : Fin 8, parallel_generatedRoot_25 backend hashModel tape (path ++ [25, i.val]) params i ((generatedRoot.w_0_0 ctx), ((generatedRoot.w_9_0 ctx), ((generatedRoot.w_1_0 ctx), ()))) ((generatedRoot.w_25_0 ctx) i))
  t35 : (4) = 4
  h26_0 : (∀ i : Fin 4, parallel_generatedRoot_26 backend hashModel tape (path ++ [26, i.val]) params i ((generatedRoot.w_0_0 ctx), ((generatedRoot.w_9_0 ctx), ((generatedRoot.w_1_0 ctx), ()))) ((generatedRoot.w_26_0 ctx) i))
  t37 : (4) = 4
  h27_0 : (∀ i : Fin 4, parallel_generatedRoot_27 backend hashModel tape (path ++ [27, i.val]) params i ((generatedRoot.w_0_0 ctx), ((generatedRoot.w_9_0 ctx), ((generatedRoot.w_1_0 ctx), ()))) ((generatedRoot.w_27_0 ctx) i))
  t39 : (2) = 2
  h29_0 : (∀ i : Fin 2, parallel_generatedRoot_29 backend hashModel tape (path ++ [29, i.val]) params i ((generatedRoot.w_0_0 ctx), ((generatedRoot.w_9_0 ctx), ((generatedRoot.w_1_0 ctx), ((generatedRoot.w_28_0 ctx), ())))) ((generatedRoot.w_29_0 ctx) i))
  t41 : (2) = 2
  h30_0 : (∀ i : Fin 2, parallel_generatedRoot_30 backend hashModel tape (path ++ [30, i.val]) params i ((generatedRoot.w_0_0 ctx), ((generatedRoot.w_9_0 ctx), ((generatedRoot.w_1_0 ctx), ((generatedRoot.w_28_0 ctx), ())))) ((generatedRoot.w_30_0 ctx) i))
  t43 : (4) = 4
  h31_0 : (∀ i : Fin 4, parallel_generatedRoot_31 backend hashModel tape (path ++ [31, i.val]) params i ((generatedRoot.w_0_0 ctx), ((generatedRoot.w_9_0 ctx), ((generatedRoot.w_1_0 ctx), ((generatedRoot.w_28_0 ctx), ())))) ((generatedRoot.w_31_0 ctx) i))
  t45 : (4) = 4
  h32_0 : (∀ i : Fin 4, parallel_generatedRoot_32 backend hashModel tape (path ++ [32, i.val]) params i ((generatedRoot.w_0_0 ctx), ((generatedRoot.w_9_0 ctx), ((generatedRoot.w_1_0 ctx), ((generatedRoot.w_28_0 ctx), ())))) ((generatedRoot.w_32_0 ctx) i))
  h33_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((generatedRoot.w_9_0 ctx)) (generatedRoot.w_33_0 ctx)
  h35_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) ((generatedRoot.w_9_0 ctx)) (generatedRoot.w_35_0 ctx)
  h41_0 : MxxRuntime.gadgetMatrixRuns backend 4096 25 (generatedRoot.w_41_0 ctx)
  h44_0 : MxxRuntime.uniformIntervalSampleAt tape (path ++ [44]) (0) (1) (generatedRoot.w_44_0 ctx)
  h45_0 : MxxRuntime.uniformIntervalSampleAt tape (path ++ [45]) (0) (1) (generatedRoot.w_45_0 ctx)
  h48_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((generatedRoot.w_9_0 ctx)) (generatedRoot.w_48_0 ctx)
  h50_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) ((generatedRoot.w_9_0 ctx)) (generatedRoot.w_50_0 ctx)
  h56_0 : MxxRuntime.gadgetMatrixRuns backend 4096 25 (generatedRoot.w_56_0 ctx)
  h59_0 : MxxRuntime.uniformIntervalSampleAt tape (path ++ [59]) (0) (1) (generatedRoot.w_59_0 ctx)
  h60_0 : MxxRuntime.uniformIntervalSampleAt tape (path ++ [60]) (0) (1) (generatedRoot.w_60_0 ctx)
  h63_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 115, 116, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((generatedRoot.w_9_0 ctx)) (generatedRoot.w_63_0 ctx)
  h65_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 115, 116, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) ((generatedRoot.w_9_0 ctx)) (generatedRoot.w_65_0 ctx)
  h74_0 : MxxRuntime.gadgetMatrixRuns backend 4096 25 (generatedRoot.w_74_0 ctx)
  h77_0 : MxxRuntime.uniformIntervalSampleAt tape (path ++ [77]) (0) (1) (generatedRoot.w_77_0 ctx)
  h78_0 : MxxRuntime.uniformIntervalSampleAt tape (path ++ [78]) (0) (1) (generatedRoot.w_78_0 ctx)
  h81_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 115, 116, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((generatedRoot.w_9_0 ctx)) (generatedRoot.w_81_0 ctx)
  h83_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 115, 116, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) ((generatedRoot.w_9_0 ctx)) (generatedRoot.w_83_0 ctx)

theorem generatedRoot.facts_1  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_1 backend hashModel tape path params (generatedRoot.w_0_0 ctx) (generatedRoot.w_9_0 ctx) (generatedRoot.w_28_0 ctx) (generatedRoot.w_1_0 ctx) (generatedRoot.w_24_0 ctx) (generatedRoot.w_25_0 ctx) (generatedRoot.w_26_0 ctx) (generatedRoot.w_27_0 ctx) (generatedRoot.w_29_0 ctx) (generatedRoot.w_30_0 ctx) (generatedRoot.w_31_0 ctx) (generatedRoot.w_32_0 ctx) (generatedRoot.w_33_0 ctx) (generatedRoot.w_35_0 ctx) (generatedRoot.w_41_0 ctx) (generatedRoot.w_44_0 ctx) (generatedRoot.w_45_0 ctx) (generatedRoot.w_48_0 ctx) (generatedRoot.w_50_0 ctx) (generatedRoot.w_56_0 ctx) (generatedRoot.w_59_0 ctx) (generatedRoot.w_60_0 ctx) (generatedRoot.w_63_0 ctx) (generatedRoot.w_65_0 ctx) (generatedRoot.w_74_0 ctx) (generatedRoot.w_77_0 ctx) (generatedRoot.w_78_0 ctx) (generatedRoot.w_81_0 ctx) (generatedRoot.w_83_0 ctx)) :
    generatedRoot.Facts_1 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_2  (ctx : generatedRootCtx) (outputs : (generatedRoot.OutputType)) : Prop where
  h91_0 : MxxRuntime.gadgetMatrixRuns backend 4096 25 (generatedRoot.w_91_0 ctx)
  h94_0 : MxxRuntime.uniformIntervalSampleAt tape (path ++ [94]) (0) (1) (generatedRoot.w_94_0 ctx)
  h95_0 : MxxRuntime.uniformIntervalSampleAt tape (path ++ [95]) (0) (1) (generatedRoot.w_95_0 ctx)
  hout : outputs = generatedRoot.outs ctx

theorem generatedRoot.facts_2  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_2 backend tape path (generatedRoot.w_8_0 ctx) (generatedRoot.w_10_0 ctx) (generatedRoot.w_11_0 ctx) (generatedRoot.w_12_0 ctx) (generatedRoot.w_13_0 ctx) (generatedRoot.w_14_0 ctx) (generatedRoot.w_15_0 ctx) (generatedRoot.w_16_0 ctx) (generatedRoot.w_17_0 ctx) (generatedRoot.w_18_0 ctx) (generatedRoot.w_19_0 ctx) (generatedRoot.w_20_0 ctx) (generatedRoot.w_21_0 ctx) (generatedRoot.w_22_0 ctx) (generatedRoot.w_23_0 ctx) (generatedRoot.w_24_0 ctx) (generatedRoot.w_25_0 ctx) (generatedRoot.w_26_0 ctx) (generatedRoot.w_27_0 ctx) (generatedRoot.w_29_0 ctx) (generatedRoot.w_30_0 ctx) (generatedRoot.w_31_0 ctx) (generatedRoot.w_32_0 ctx) (generatedRoot.w_47_0 ctx) (generatedRoot.w_62_0 ctx) (generatedRoot.w_80_0 ctx) (generatedRoot.w_91_0 ctx) (generatedRoot.w_94_0 ctx) (generatedRoot.w_95_0 ctx) (generatedRoot.w_97_0 ctx) outputs) :
    generatedRoot.Facts_2 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2⟩

structure generatedRoot.Facts  (ctx : generatedRootCtx) (outputs : (generatedRoot.OutputType)) : Prop where
  c0 : generatedRoot.Facts_0 ctx
  c1 : generatedRoot.Facts_1 ctx
  c2 : generatedRoot.Facts_2 ctx outputs

theorem generatedRoot.facts  (ctx : generatedRootCtx) {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.body backend hashModel tape path params ctx.inputs outputs ctx.witness) :
    generatedRoot.Facts ctx outputs :=
  have t0 := h
  have r0 := generatedRoot.facts_0 (ctx := ctx) (outputs := outputs) t0.1
  have t1 := t0.2
  have r1 := generatedRoot.facts_1 (ctx := ctx) (outputs := outputs) t1.1
  have t2 := t1.2
  have r2 := generatedRoot.facts_2 (ctx := ctx) (outputs := outputs) t2
  ⟨r0, r1, r2⟩

end

end Stage_initial

end
