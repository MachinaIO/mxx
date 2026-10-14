import Stage_sigma_encrypt

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000
set_option linter.unusedVariables false

-- The wire types repeat the ring modulus `X ^ n + 1`; unfolding it while checking a
-- projection against its field type expands it at the ring dimension.
attribute [local irreducible] Mxx.Primitives.negacyclicModulus

noncomputable section

namespace Stage_sigma_encrypt

local notation "mxxMat0" => Mxx.Primitives.ExactMatrix modulus_30 131072 1 1
local notation "mxxMat1" => Mxx.Primitives.ExactMatrix modulus_0 131072 1 1
local notation "mxxMat2" => Mxx.Primitives.ExactMatrix modulus_0 131072 1 168
local notation "mxxMat3" => Mxx.Primitives.ExactMatrix modulus_1 131072 1 168
local notation "mxxMat4" => Mxx.Primitives.ExactMatrix modulus_0 131072 1 172
local notation "mxxMat5" => Mxx.Primitives.ExactMatrix modulus_1 131072 1 164
local notation "mxxMat6" => Mxx.Primitives.ExactMatrix modulus_3 131072 1 128
local notation "mxxMat7" => Mxx.Primitives.ExactMatrix modulus_4 131072 1 124
local notation "mxxMat8" => Mxx.Primitives.ExactMatrix modulus_5 131072 1 120
local notation "mxxMat9" => Mxx.Primitives.ExactMatrix modulus_6 131072 1 116
local notation "mxxMat10" => Mxx.Primitives.ExactMatrix modulus_7 131072 1 112
local notation "mxxMat11" => Mxx.Primitives.ExactMatrix modulus_8 131072 1 108
local notation "mxxMat12" => Mxx.Primitives.ExactMatrix modulus_9 131072 1 104
local notation "mxxMat13" => Mxx.Primitives.ExactMatrix modulus_10 131072 1 100
local notation "mxxMat14" => Mxx.Primitives.ExactMatrix modulus_11 131072 1 96
local notation "mxxMat15" => Mxx.Primitives.ExactMatrix modulus_12 131072 1 92
local notation "mxxMat16" => Mxx.Primitives.ExactMatrix modulus_2 131072 1 160
local notation "mxxMat17" => Mxx.Primitives.ExactMatrix modulus_13 131072 1 88
local notation "mxxMat18" => Mxx.Primitives.ExactMatrix modulus_15 131072 1 84
local notation "mxxMat19" => Mxx.Primitives.ExactMatrix modulus_16 131072 1 80
local notation "mxxMat20" => Mxx.Primitives.ExactMatrix modulus_17 131072 1 76
local notation "mxxMat21" => Mxx.Primitives.ExactMatrix modulus_18 131072 1 72
local notation "mxxMat22" => Mxx.Primitives.ExactMatrix modulus_19 131072 1 68
local notation "mxxMat23" => Mxx.Primitives.ExactMatrix modulus_20 131072 1 64
local notation "mxxMat24" => Mxx.Primitives.ExactMatrix modulus_21 131072 1 60
local notation "mxxMat25" => Mxx.Primitives.ExactMatrix modulus_22 131072 1 56
local notation "mxxMat26" => Mxx.Primitives.ExactMatrix modulus_23 131072 1 52
local notation "mxxMat27" => Mxx.Primitives.ExactMatrix modulus_14 131072 1 156
local notation "mxxMat28" => Mxx.Primitives.ExactMatrix modulus_24 131072 1 48
local notation "mxxMat29" => Mxx.Primitives.ExactMatrix modulus_26 131072 1 44
local notation "mxxMat30" => Mxx.Primitives.ExactMatrix modulus_27 131072 1 40
local notation "mxxMat31" => Mxx.Primitives.ExactMatrix modulus_28 131072 1 36
local notation "mxxMat32" => Mxx.Primitives.ExactMatrix modulus_29 131072 1 32
local notation "mxxMat33" => Mxx.Primitives.ExactMatrix modulus_30 131072 1 28
local notation "mxxMat34" => Mxx.Primitives.ExactMatrix modulus_25 131072 1 152
local notation "mxxMat35" => Mxx.Primitives.ExactMatrix modulus_31 131072 1 148
local notation "mxxMat36" => Mxx.Primitives.ExactMatrix modulus_32 131072 1 144
local notation "mxxMat37" => Mxx.Primitives.ExactMatrix modulus_33 131072 1 140
local notation "mxxMat38" => Mxx.Primitives.ExactMatrix modulus_34 131072 1 136
local notation "mxxMat39" => Mxx.Primitives.ExactMatrix modulus_35 131072 1 132

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_7.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) where
  inputs : mxxMat0 × ByteArray × mxxMat0 × Int × Int × (Fin 1025 → Int) × (Fin 1025 → Int) × Unit
  witness : parallel_generatedRoot_7.Witness

def parallel_generatedRoot_7.w_0_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : mxxMat0 :=
  ctx.inputs.1

def parallel_generatedRoot_7.w_2_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : ByteArray :=
  ctx.inputs.2.1

def parallel_generatedRoot_7.w_10_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_7.w_16_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Int :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_7.w_18_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Int :=
  ctx.inputs.2.2.2.2.1

def parallel_generatedRoot_7.w_37_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 1025 → Int :=
  ctx.inputs.2.2.2.2.2.1

def parallel_generatedRoot_7.w_38_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 1025 → Int :=
  ctx.inputs.2.2.2.2.2.2.1

def parallel_generatedRoot_7.w_1_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : mxxMat1 :=
  ctx.witness.w_1_0

def parallel_generatedRoot_7.w_3_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : mxxMat2 :=
  ctx.witness.w_3_0

def parallel_generatedRoot_7.w_4_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : mxxMat2 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_7.w_1_0 ctx) (parallel_generatedRoot_7.w_3_0 ctx)

def parallel_generatedRoot_7.w_5_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : mxxMat0 :=
  (1 : mxxMat0)

def parallel_generatedRoot_7.w_6_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : mxxMat1 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_7.w_7_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : mxxMat2 :=
  ctx.witness.w_7_0

def parallel_generatedRoot_7.w_8_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : mxxMat2 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_7.w_6_0 ctx) (parallel_generatedRoot_7.w_7_0 ctx)

def parallel_generatedRoot_7.w_9_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : mxxMat2 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_7.w_4_0 ctx) (parallel_generatedRoot_7.w_8_0 ctx)

def parallel_generatedRoot_7.w_11_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : mxxMat1 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_7.w_12_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : mxxMat3 :=
  ctx.witness.w_12_0

def parallel_generatedRoot_7.w_13_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : mxxMat2 :=
  ctx.witness.w_13_0

def parallel_generatedRoot_7.w_14_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : mxxMat2 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_7.w_11_0 ctx) (parallel_generatedRoot_7.w_13_0 ctx)

def parallel_generatedRoot_7.w_15_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : mxxMat1 :=
  (MxxRuntime.matrixPolynomial [(1099380817921)] : mxxMat1)

def parallel_generatedRoot_7.w_17_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : mxxMat1 :=
  MxxRuntime.liftInteger (parallel_generatedRoot_7.w_16_0 ctx)

def parallel_generatedRoot_7.w_19_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : mxxMat1 :=
  MxxRuntime.multiplyMonomial (parallel_generatedRoot_7.w_17_0 ctx) (parallel_generatedRoot_7.w_18_0 ctx)

def parallel_generatedRoot_7.w_20_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_7.w_15_0 ctx) (parallel_generatedRoot_7.w_19_0 ctx)

def parallel_generatedRoot_7.w_21_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : mxxMat2 :=
  MxxRuntime.matrixMulScalarRight (parallel_generatedRoot_7.w_14_0 ctx) (parallel_generatedRoot_7.w_20_0 ctx)

def parallel_generatedRoot_7.w_22_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : mxxMat2 :=
  MxxRuntime.matrixSub (parallel_generatedRoot_7.w_9_0 ctx) (parallel_generatedRoot_7.w_21_0 ctx)

def parallel_generatedRoot_7.w_23_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : mxxMat2 :=
  ctx.witness.w_23_0

def parallel_generatedRoot_7.w_24_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : mxxMat2 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_7.w_22_0 ctx) (parallel_generatedRoot_7.w_23_0 ctx)

def parallel_generatedRoot_7.w_25_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : mxxMat2 :=
  ctx.witness.w_25_0

def parallel_generatedRoot_7.w_26_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : mxxMat2 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_7.w_1_0 ctx) (parallel_generatedRoot_7.w_25_0 ctx)

def parallel_generatedRoot_7.w_27_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : mxxMat2 :=
  ctx.witness.w_27_0

def parallel_generatedRoot_7.w_28_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : mxxMat2 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_7.w_6_0 ctx) (parallel_generatedRoot_7.w_27_0 ctx)

def parallel_generatedRoot_7.w_29_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : mxxMat2 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_7.w_26_0 ctx) (parallel_generatedRoot_7.w_28_0 ctx)

def parallel_generatedRoot_7.w_30_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : mxxMat0 :=
  (1 : mxxMat0)

def parallel_generatedRoot_7.w_31_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : mxxMat1 :=
  ctx.witness.w_31_0

def parallel_generatedRoot_7.w_32_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : mxxMat2 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_7.w_31_0 ctx) (parallel_generatedRoot_7.w_13_0 ctx)

def parallel_generatedRoot_7.w_33_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : mxxMat2 :=
  MxxRuntime.matrixMulScalarRight (parallel_generatedRoot_7.w_32_0 ctx) (parallel_generatedRoot_7.w_20_0 ctx)

def parallel_generatedRoot_7.w_34_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : mxxMat2 :=
  MxxRuntime.matrixSub (parallel_generatedRoot_7.w_29_0 ctx) (parallel_generatedRoot_7.w_33_0 ctx)

def parallel_generatedRoot_7.w_35_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : mxxMat2 :=
  ctx.witness.w_35_0

def parallel_generatedRoot_7.w_36_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : mxxMat2 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_7.w_34_0 ctx) (parallel_generatedRoot_7.w_35_0 ctx)

def parallel_generatedRoot_7.w_39_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 1025 → mxxMat4 :=
  ctx.witness.w_39_0

def parallel_generatedRoot_7.w_39_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 1025 → mxxMat4 :=
  ctx.witness.w_39_1

def parallel_generatedRoot_7.w_39_2 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 1025 → mxxMat1 :=
  ctx.witness.w_39_2

def parallel_generatedRoot_7.w_40_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 128 → mxxMat2 :=
  ctx.witness.w_40_0

def parallel_generatedRoot_7.w_40_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 128 → mxxMat2 :=
  ctx.witness.w_40_1

def parallel_generatedRoot_7.w_41_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 32 → mxxMat2 :=
  ctx.witness.w_41_0

def parallel_generatedRoot_7.w_41_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 32 → mxxMat2 :=
  ctx.witness.w_41_1

def parallel_generatedRoot_7.w_42_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 128 → mxxMat5 :=
  ctx.witness.w_42_0

def parallel_generatedRoot_7.w_42_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 128 → mxxMat5 :=
  ctx.witness.w_42_1

def parallel_generatedRoot_7.w_43_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 32 → mxxMat5 :=
  ctx.witness.w_43_0

def parallel_generatedRoot_7.w_43_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 32 → mxxMat5 :=
  ctx.witness.w_43_1

def parallel_generatedRoot_7.w_44_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 128 → mxxMat6 :=
  ctx.witness.w_44_0

def parallel_generatedRoot_7.w_44_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 128 → mxxMat6 :=
  ctx.witness.w_44_1

def parallel_generatedRoot_7.w_45_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 32 → mxxMat6 :=
  ctx.witness.w_45_0

def parallel_generatedRoot_7.w_45_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 32 → mxxMat6 :=
  ctx.witness.w_45_1

def parallel_generatedRoot_7.w_46_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 128 → mxxMat7 :=
  ctx.witness.w_46_0

def parallel_generatedRoot_7.w_46_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 128 → mxxMat7 :=
  ctx.witness.w_46_1

def parallel_generatedRoot_7.w_47_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 32 → mxxMat7 :=
  ctx.witness.w_47_0

def parallel_generatedRoot_7.w_47_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 32 → mxxMat7 :=
  ctx.witness.w_47_1

def parallel_generatedRoot_7.w_48_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 128 → mxxMat8 :=
  ctx.witness.w_48_0

def parallel_generatedRoot_7.w_48_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 128 → mxxMat8 :=
  ctx.witness.w_48_1

def parallel_generatedRoot_7.w_49_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 32 → mxxMat8 :=
  ctx.witness.w_49_0

def parallel_generatedRoot_7.w_49_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 32 → mxxMat8 :=
  ctx.witness.w_49_1

def parallel_generatedRoot_7.w_50_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 128 → mxxMat9 :=
  ctx.witness.w_50_0

def parallel_generatedRoot_7.w_50_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 128 → mxxMat9 :=
  ctx.witness.w_50_1

def parallel_generatedRoot_7.w_51_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 32 → mxxMat9 :=
  ctx.witness.w_51_0

def parallel_generatedRoot_7.w_51_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 32 → mxxMat9 :=
  ctx.witness.w_51_1

def parallel_generatedRoot_7.w_52_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 128 → mxxMat10 :=
  ctx.witness.w_52_0

def parallel_generatedRoot_7.w_52_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 128 → mxxMat10 :=
  ctx.witness.w_52_1

def parallel_generatedRoot_7.w_53_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 32 → mxxMat10 :=
  ctx.witness.w_53_0

def parallel_generatedRoot_7.w_53_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 32 → mxxMat10 :=
  ctx.witness.w_53_1

def parallel_generatedRoot_7.w_54_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 128 → mxxMat11 :=
  ctx.witness.w_54_0

def parallel_generatedRoot_7.w_54_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 128 → mxxMat11 :=
  ctx.witness.w_54_1

def parallel_generatedRoot_7.w_55_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 32 → mxxMat11 :=
  ctx.witness.w_55_0

def parallel_generatedRoot_7.w_55_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 32 → mxxMat11 :=
  ctx.witness.w_55_1

def parallel_generatedRoot_7.w_56_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 128 → mxxMat12 :=
  ctx.witness.w_56_0

def parallel_generatedRoot_7.w_56_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 128 → mxxMat12 :=
  ctx.witness.w_56_1

def parallel_generatedRoot_7.w_57_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 32 → mxxMat12 :=
  ctx.witness.w_57_0

def parallel_generatedRoot_7.w_57_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 32 → mxxMat12 :=
  ctx.witness.w_57_1

def parallel_generatedRoot_7.w_58_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 128 → mxxMat13 :=
  ctx.witness.w_58_0

def parallel_generatedRoot_7.w_58_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 128 → mxxMat13 :=
  ctx.witness.w_58_1

def parallel_generatedRoot_7.w_59_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 32 → mxxMat13 :=
  ctx.witness.w_59_0

def parallel_generatedRoot_7.w_59_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 32 → mxxMat13 :=
  ctx.witness.w_59_1

def parallel_generatedRoot_7.w_60_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 128 → mxxMat14 :=
  ctx.witness.w_60_0

def parallel_generatedRoot_7.w_60_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 128 → mxxMat14 :=
  ctx.witness.w_60_1

def parallel_generatedRoot_7.w_61_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 32 → mxxMat14 :=
  ctx.witness.w_61_0

def parallel_generatedRoot_7.w_61_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 32 → mxxMat14 :=
  ctx.witness.w_61_1

def parallel_generatedRoot_7.w_62_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 128 → mxxMat15 :=
  ctx.witness.w_62_0

def parallel_generatedRoot_7.w_62_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 128 → mxxMat15 :=
  ctx.witness.w_62_1

def parallel_generatedRoot_7.w_63_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 32 → mxxMat15 :=
  ctx.witness.w_63_0

def parallel_generatedRoot_7.w_63_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 32 → mxxMat15 :=
  ctx.witness.w_63_1

def parallel_generatedRoot_7.w_64_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 128 → mxxMat16 :=
  ctx.witness.w_64_0

def parallel_generatedRoot_7.w_64_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 128 → mxxMat16 :=
  ctx.witness.w_64_1

def parallel_generatedRoot_7.w_65_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 32 → mxxMat16 :=
  ctx.witness.w_65_0

def parallel_generatedRoot_7.w_65_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 32 → mxxMat16 :=
  ctx.witness.w_65_1

def parallel_generatedRoot_7.w_66_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 128 → mxxMat17 :=
  ctx.witness.w_66_0

def parallel_generatedRoot_7.w_66_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 128 → mxxMat17 :=
  ctx.witness.w_66_1

def parallel_generatedRoot_7.w_67_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 32 → mxxMat17 :=
  ctx.witness.w_67_0

def parallel_generatedRoot_7.w_67_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 32 → mxxMat17 :=
  ctx.witness.w_67_1

def parallel_generatedRoot_7.w_68_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 128 → mxxMat18 :=
  ctx.witness.w_68_0

def parallel_generatedRoot_7.w_68_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 128 → mxxMat18 :=
  ctx.witness.w_68_1

def parallel_generatedRoot_7.w_69_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 32 → mxxMat18 :=
  ctx.witness.w_69_0

def parallel_generatedRoot_7.w_69_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 32 → mxxMat18 :=
  ctx.witness.w_69_1

def parallel_generatedRoot_7.w_70_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 128 → mxxMat19 :=
  ctx.witness.w_70_0

def parallel_generatedRoot_7.w_70_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 128 → mxxMat19 :=
  ctx.witness.w_70_1

def parallel_generatedRoot_7.w_71_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 32 → mxxMat19 :=
  ctx.witness.w_71_0

def parallel_generatedRoot_7.w_71_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 32 → mxxMat19 :=
  ctx.witness.w_71_1

def parallel_generatedRoot_7.w_72_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 128 → mxxMat20 :=
  ctx.witness.w_72_0

def parallel_generatedRoot_7.w_72_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 128 → mxxMat20 :=
  ctx.witness.w_72_1

def parallel_generatedRoot_7.w_73_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 32 → mxxMat20 :=
  ctx.witness.w_73_0

def parallel_generatedRoot_7.w_73_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 32 → mxxMat20 :=
  ctx.witness.w_73_1

def parallel_generatedRoot_7.w_74_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 128 → mxxMat21 :=
  ctx.witness.w_74_0

def parallel_generatedRoot_7.w_74_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 128 → mxxMat21 :=
  ctx.witness.w_74_1

def parallel_generatedRoot_7.w_75_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 32 → mxxMat21 :=
  ctx.witness.w_75_0

def parallel_generatedRoot_7.w_75_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 32 → mxxMat21 :=
  ctx.witness.w_75_1

def parallel_generatedRoot_7.w_76_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 128 → mxxMat22 :=
  ctx.witness.w_76_0

def parallel_generatedRoot_7.w_76_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 128 → mxxMat22 :=
  ctx.witness.w_76_1

def parallel_generatedRoot_7.w_77_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 32 → mxxMat22 :=
  ctx.witness.w_77_0

def parallel_generatedRoot_7.w_77_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 32 → mxxMat22 :=
  ctx.witness.w_77_1

def parallel_generatedRoot_7.w_78_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 128 → mxxMat23 :=
  ctx.witness.w_78_0

def parallel_generatedRoot_7.w_78_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 128 → mxxMat23 :=
  ctx.witness.w_78_1

def parallel_generatedRoot_7.w_79_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 32 → mxxMat23 :=
  ctx.witness.w_79_0

def parallel_generatedRoot_7.w_79_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 32 → mxxMat23 :=
  ctx.witness.w_79_1

def parallel_generatedRoot_7.w_80_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 128 → mxxMat24 :=
  ctx.witness.w_80_0

def parallel_generatedRoot_7.w_80_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 128 → mxxMat24 :=
  ctx.witness.w_80_1

def parallel_generatedRoot_7.w_81_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 32 → mxxMat24 :=
  ctx.witness.w_81_0

def parallel_generatedRoot_7.w_81_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 32 → mxxMat24 :=
  ctx.witness.w_81_1

def parallel_generatedRoot_7.w_82_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 128 → mxxMat25 :=
  ctx.witness.w_82_0

def parallel_generatedRoot_7.w_82_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 128 → mxxMat25 :=
  ctx.witness.w_82_1

def parallel_generatedRoot_7.w_83_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 32 → mxxMat25 :=
  ctx.witness.w_83_0

def parallel_generatedRoot_7.w_83_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 32 → mxxMat25 :=
  ctx.witness.w_83_1

def parallel_generatedRoot_7.w_84_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 128 → mxxMat26 :=
  ctx.witness.w_84_0

def parallel_generatedRoot_7.w_84_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 128 → mxxMat26 :=
  ctx.witness.w_84_1

def parallel_generatedRoot_7.w_85_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 32 → mxxMat26 :=
  ctx.witness.w_85_0

def parallel_generatedRoot_7.w_85_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 32 → mxxMat26 :=
  ctx.witness.w_85_1

def parallel_generatedRoot_7.w_86_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 128 → mxxMat27 :=
  ctx.witness.w_86_0

def parallel_generatedRoot_7.w_86_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 128 → mxxMat27 :=
  ctx.witness.w_86_1

def parallel_generatedRoot_7.w_87_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 32 → mxxMat27 :=
  ctx.witness.w_87_0

def parallel_generatedRoot_7.w_87_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 32 → mxxMat27 :=
  ctx.witness.w_87_1

def parallel_generatedRoot_7.w_88_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 128 → mxxMat28 :=
  ctx.witness.w_88_0

def parallel_generatedRoot_7.w_88_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 128 → mxxMat28 :=
  ctx.witness.w_88_1

def parallel_generatedRoot_7.w_89_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 32 → mxxMat28 :=
  ctx.witness.w_89_0

def parallel_generatedRoot_7.w_89_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 32 → mxxMat28 :=
  ctx.witness.w_89_1

def parallel_generatedRoot_7.w_90_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 128 → mxxMat29 :=
  ctx.witness.w_90_0

def parallel_generatedRoot_7.w_90_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 128 → mxxMat29 :=
  ctx.witness.w_90_1

def parallel_generatedRoot_7.w_91_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 32 → mxxMat29 :=
  ctx.witness.w_91_0

def parallel_generatedRoot_7.w_91_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 32 → mxxMat29 :=
  ctx.witness.w_91_1

def parallel_generatedRoot_7.w_92_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 128 → mxxMat30 :=
  ctx.witness.w_92_0

def parallel_generatedRoot_7.w_92_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 128 → mxxMat30 :=
  ctx.witness.w_92_1

def parallel_generatedRoot_7.w_93_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 32 → mxxMat30 :=
  ctx.witness.w_93_0

def parallel_generatedRoot_7.w_93_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 32 → mxxMat30 :=
  ctx.witness.w_93_1

def parallel_generatedRoot_7.w_94_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 128 → mxxMat31 :=
  ctx.witness.w_94_0

def parallel_generatedRoot_7.w_94_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 128 → mxxMat31 :=
  ctx.witness.w_94_1

def parallel_generatedRoot_7.w_95_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 32 → mxxMat31 :=
  ctx.witness.w_95_0

def parallel_generatedRoot_7.w_95_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 32 → mxxMat31 :=
  ctx.witness.w_95_1

def parallel_generatedRoot_7.w_96_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 128 → mxxMat32 :=
  ctx.witness.w_96_0

def parallel_generatedRoot_7.w_96_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 128 → mxxMat32 :=
  ctx.witness.w_96_1

def parallel_generatedRoot_7.w_97_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 32 → mxxMat32 :=
  ctx.witness.w_97_0

def parallel_generatedRoot_7.w_97_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 32 → mxxMat32 :=
  ctx.witness.w_97_1

def parallel_generatedRoot_7.w_98_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 128 → mxxMat33 :=
  ctx.witness.w_98_0

def parallel_generatedRoot_7.w_98_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 128 → mxxMat33 :=
  ctx.witness.w_98_1

def parallel_generatedRoot_7.w_99_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 32 → mxxMat33 :=
  ctx.witness.w_99_0

def parallel_generatedRoot_7.w_99_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 32 → mxxMat33 :=
  ctx.witness.w_99_1

def parallel_generatedRoot_7.w_100_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 128 → mxxMat34 :=
  ctx.witness.w_100_0

def parallel_generatedRoot_7.w_100_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 128 → mxxMat34 :=
  ctx.witness.w_100_1

def parallel_generatedRoot_7.w_101_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 32 → mxxMat34 :=
  ctx.witness.w_101_0

def parallel_generatedRoot_7.w_101_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 32 → mxxMat34 :=
  ctx.witness.w_101_1

def parallel_generatedRoot_7.w_102_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 128 → mxxMat35 :=
  ctx.witness.w_102_0

def parallel_generatedRoot_7.w_102_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 128 → mxxMat35 :=
  ctx.witness.w_102_1

def parallel_generatedRoot_7.w_103_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 32 → mxxMat35 :=
  ctx.witness.w_103_0

def parallel_generatedRoot_7.w_103_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 32 → mxxMat35 :=
  ctx.witness.w_103_1

def parallel_generatedRoot_7.w_104_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 128 → mxxMat36 :=
  ctx.witness.w_104_0

def parallel_generatedRoot_7.w_104_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 128 → mxxMat36 :=
  ctx.witness.w_104_1

def parallel_generatedRoot_7.w_105_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 32 → mxxMat36 :=
  ctx.witness.w_105_0

def parallel_generatedRoot_7.w_105_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 32 → mxxMat36 :=
  ctx.witness.w_105_1

def parallel_generatedRoot_7.w_106_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 128 → mxxMat37 :=
  ctx.witness.w_106_0

def parallel_generatedRoot_7.w_106_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 128 → mxxMat37 :=
  ctx.witness.w_106_1

def parallel_generatedRoot_7.w_107_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 32 → mxxMat37 :=
  ctx.witness.w_107_0

def parallel_generatedRoot_7.w_107_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 32 → mxxMat37 :=
  ctx.witness.w_107_1

def parallel_generatedRoot_7.w_108_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 128 → mxxMat38 :=
  ctx.witness.w_108_0

def parallel_generatedRoot_7.w_108_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 128 → mxxMat38 :=
  ctx.witness.w_108_1

def parallel_generatedRoot_7.w_109_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 32 → mxxMat38 :=
  ctx.witness.w_109_0

def parallel_generatedRoot_7.w_109_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 32 → mxxMat38 :=
  ctx.witness.w_109_1

def parallel_generatedRoot_7.w_110_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 128 → mxxMat39 :=
  ctx.witness.w_110_0

def parallel_generatedRoot_7.w_110_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 128 → mxxMat39 :=
  ctx.witness.w_110_1

def parallel_generatedRoot_7.w_111_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 32 → mxxMat39 :=
  ctx.witness.w_111_0

def parallel_generatedRoot_7.w_111_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Fin 32 → mxxMat39 :=
  ctx.witness.w_111_1

def parallel_generatedRoot_7.outs {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : mxxMat2 × mxxMat2 × (Fin 1025 → mxxMat4) × (Fin 1025 → mxxMat4) × (Fin 1025 → mxxMat1) × (Fin 128 → mxxMat2) × (Fin 128 → mxxMat2) × (Fin 32 → mxxMat2) × (Fin 32 → mxxMat2) × (Fin 128 → mxxMat5) × (Fin 128 → mxxMat5) × (Fin 32 → mxxMat5) × (Fin 32 → mxxMat5) × (Fin 128 → mxxMat6) × (Fin 128 → mxxMat6) × (Fin 32 → mxxMat6) × (Fin 32 → mxxMat6) × (Fin 128 → mxxMat7) × (Fin 128 → mxxMat7) × (Fin 32 → mxxMat7) × (Fin 32 → mxxMat7) × (Fin 128 → mxxMat8) × (Fin 128 → mxxMat8) × (Fin 32 → mxxMat8) × (Fin 32 → mxxMat8) × (Fin 128 → mxxMat9) × (Fin 128 → mxxMat9) × (Fin 32 → mxxMat9) × (Fin 32 → mxxMat9) × (Fin 128 → mxxMat10) × (Fin 128 → mxxMat10) × (Fin 32 → mxxMat10) × (Fin 32 → mxxMat10) × (Fin 128 → mxxMat11) × (Fin 128 → mxxMat11) × (Fin 32 → mxxMat11) × (Fin 32 → mxxMat11) × (Fin 128 → mxxMat12) × (Fin 128 → mxxMat12) × (Fin 32 → mxxMat12) × (Fin 32 → mxxMat12) × (Fin 128 → mxxMat13) × (Fin 128 → mxxMat13) × (Fin 32 → mxxMat13) × (Fin 32 → mxxMat13) × (Fin 128 → mxxMat14) × (Fin 128 → mxxMat14) × (Fin 32 → mxxMat14) × (Fin 32 → mxxMat14) × (Fin 128 → mxxMat15) × (Fin 128 → mxxMat15) × (Fin 32 → mxxMat15) × (Fin 32 → mxxMat15) × (Fin 128 → mxxMat16) × (Fin 128 → mxxMat16) × (Fin 32 → mxxMat16) × (Fin 32 → mxxMat16) × (Fin 128 → mxxMat17) × (Fin 128 → mxxMat17) × (Fin 32 → mxxMat17) × (Fin 32 → mxxMat17) × (Fin 128 → mxxMat18) × (Fin 128 → mxxMat18) × (Fin 32 → mxxMat18) × (Fin 32 → mxxMat18) × (Fin 128 → mxxMat19) × (Fin 128 → mxxMat19) × (Fin 32 → mxxMat19) × (Fin 32 → mxxMat19) × (Fin 128 → mxxMat20) × (Fin 128 → mxxMat20) × (Fin 32 → mxxMat20) × (Fin 32 → mxxMat20) × (Fin 128 → mxxMat21) × (Fin 128 → mxxMat21) × (Fin 32 → mxxMat21) × (Fin 32 → mxxMat21) × (Fin 128 → mxxMat22) × (Fin 128 → mxxMat22) × (Fin 32 → mxxMat22) × (Fin 32 → mxxMat22) × (Fin 128 → mxxMat23) × (Fin 128 → mxxMat23) × (Fin 32 → mxxMat23) × (Fin 32 → mxxMat23) × (Fin 128 → mxxMat24) × (Fin 128 → mxxMat24) × (Fin 32 → mxxMat24) × (Fin 32 → mxxMat24) × (Fin 128 → mxxMat25) × (Fin 128 → mxxMat25) × (Fin 32 → mxxMat25) × (Fin 32 → mxxMat25) × (Fin 128 → mxxMat26) × (Fin 128 → mxxMat26) × (Fin 32 → mxxMat26) × (Fin 32 → mxxMat26) × (Fin 128 → mxxMat27) × (Fin 128 → mxxMat27) × (Fin 32 → mxxMat27) × (Fin 32 → mxxMat27) × (Fin 128 → mxxMat28) × (Fin 128 → mxxMat28) × (Fin 32 → mxxMat28) × (Fin 32 → mxxMat28) × (Fin 128 → mxxMat29) × (Fin 128 → mxxMat29) × (Fin 32 → mxxMat29) × (Fin 32 → mxxMat29) × (Fin 128 → mxxMat30) × (Fin 128 → mxxMat30) × (Fin 32 → mxxMat30) × (Fin 32 → mxxMat30) × (Fin 128 → mxxMat31) × (Fin 128 → mxxMat31) × (Fin 32 → mxxMat31) × (Fin 32 → mxxMat31) × (Fin 128 → mxxMat32) × (Fin 128 → mxxMat32) × (Fin 32 → mxxMat32) × (Fin 32 → mxxMat32) × (Fin 128 → mxxMat33) × (Fin 128 → mxxMat33) × (Fin 32 → mxxMat33) × (Fin 32 → mxxMat33) × (Fin 128 → mxxMat34) × (Fin 128 → mxxMat34) × (Fin 32 → mxxMat34) × (Fin 32 → mxxMat34) × (Fin 128 → mxxMat35) × (Fin 128 → mxxMat35) × (Fin 32 → mxxMat35) × (Fin 32 → mxxMat35) × (Fin 128 → mxxMat36) × (Fin 128 → mxxMat36) × (Fin 32 → mxxMat36) × (Fin 32 → mxxMat36) × (Fin 128 → mxxMat37) × (Fin 128 → mxxMat37) × (Fin 32 → mxxMat37) × (Fin 32 → mxxMat37) × (Fin 128 → mxxMat38) × (Fin 128 → mxxMat38) × (Fin 32 → mxxMat38) × (Fin 32 → mxxMat38) × (Fin 128 → mxxMat39) × (Fin 128 → mxxMat39) × (Fin 32 → mxxMat39) × (Fin 32 → mxxMat39) × Unit :=
  ((parallel_generatedRoot_7.w_24_0 ctx), ((parallel_generatedRoot_7.w_36_0 ctx), ((parallel_generatedRoot_7.w_39_0 ctx), ((parallel_generatedRoot_7.w_39_1 ctx), ((parallel_generatedRoot_7.w_39_2 ctx), ((parallel_generatedRoot_7.w_40_0 ctx), ((parallel_generatedRoot_7.w_40_1 ctx), ((parallel_generatedRoot_7.w_41_0 ctx), ((parallel_generatedRoot_7.w_41_1 ctx), ((parallel_generatedRoot_7.w_42_0 ctx), ((parallel_generatedRoot_7.w_42_1 ctx), ((parallel_generatedRoot_7.w_43_0 ctx), ((parallel_generatedRoot_7.w_43_1 ctx), ((parallel_generatedRoot_7.w_44_0 ctx), ((parallel_generatedRoot_7.w_44_1 ctx), ((parallel_generatedRoot_7.w_45_0 ctx), ((parallel_generatedRoot_7.w_45_1 ctx), ((parallel_generatedRoot_7.w_46_0 ctx), ((parallel_generatedRoot_7.w_46_1 ctx), ((parallel_generatedRoot_7.w_47_0 ctx), ((parallel_generatedRoot_7.w_47_1 ctx), ((parallel_generatedRoot_7.w_48_0 ctx), ((parallel_generatedRoot_7.w_48_1 ctx), ((parallel_generatedRoot_7.w_49_0 ctx), ((parallel_generatedRoot_7.w_49_1 ctx), ((parallel_generatedRoot_7.w_50_0 ctx), ((parallel_generatedRoot_7.w_50_1 ctx), ((parallel_generatedRoot_7.w_51_0 ctx), ((parallel_generatedRoot_7.w_51_1 ctx), ((parallel_generatedRoot_7.w_52_0 ctx), ((parallel_generatedRoot_7.w_52_1 ctx), ((parallel_generatedRoot_7.w_53_0 ctx), ((parallel_generatedRoot_7.w_53_1 ctx), ((parallel_generatedRoot_7.w_54_0 ctx), ((parallel_generatedRoot_7.w_54_1 ctx), ((parallel_generatedRoot_7.w_55_0 ctx), ((parallel_generatedRoot_7.w_55_1 ctx), ((parallel_generatedRoot_7.w_56_0 ctx), ((parallel_generatedRoot_7.w_56_1 ctx), ((parallel_generatedRoot_7.w_57_0 ctx), ((parallel_generatedRoot_7.w_57_1 ctx), ((parallel_generatedRoot_7.w_58_0 ctx), ((parallel_generatedRoot_7.w_58_1 ctx), ((parallel_generatedRoot_7.w_59_0 ctx), ((parallel_generatedRoot_7.w_59_1 ctx), ((parallel_generatedRoot_7.w_60_0 ctx), ((parallel_generatedRoot_7.w_60_1 ctx), ((parallel_generatedRoot_7.w_61_0 ctx), ((parallel_generatedRoot_7.w_61_1 ctx), ((parallel_generatedRoot_7.w_62_0 ctx), ((parallel_generatedRoot_7.w_62_1 ctx), ((parallel_generatedRoot_7.w_63_0 ctx), ((parallel_generatedRoot_7.w_63_1 ctx), ((parallel_generatedRoot_7.w_64_0 ctx), ((parallel_generatedRoot_7.w_64_1 ctx), ((parallel_generatedRoot_7.w_65_0 ctx), ((parallel_generatedRoot_7.w_65_1 ctx), ((parallel_generatedRoot_7.w_66_0 ctx), ((parallel_generatedRoot_7.w_66_1 ctx), ((parallel_generatedRoot_7.w_67_0 ctx), ((parallel_generatedRoot_7.w_67_1 ctx), ((parallel_generatedRoot_7.w_68_0 ctx), ((parallel_generatedRoot_7.w_68_1 ctx), ((parallel_generatedRoot_7.w_69_0 ctx), ((parallel_generatedRoot_7.w_69_1 ctx), ((parallel_generatedRoot_7.w_70_0 ctx), ((parallel_generatedRoot_7.w_70_1 ctx), ((parallel_generatedRoot_7.w_71_0 ctx), ((parallel_generatedRoot_7.w_71_1 ctx), ((parallel_generatedRoot_7.w_72_0 ctx), ((parallel_generatedRoot_7.w_72_1 ctx), ((parallel_generatedRoot_7.w_73_0 ctx), ((parallel_generatedRoot_7.w_73_1 ctx), ((parallel_generatedRoot_7.w_74_0 ctx), ((parallel_generatedRoot_7.w_74_1 ctx), ((parallel_generatedRoot_7.w_75_0 ctx), ((parallel_generatedRoot_7.w_75_1 ctx), ((parallel_generatedRoot_7.w_76_0 ctx), ((parallel_generatedRoot_7.w_76_1 ctx), ((parallel_generatedRoot_7.w_77_0 ctx), ((parallel_generatedRoot_7.w_77_1 ctx), ((parallel_generatedRoot_7.w_78_0 ctx), ((parallel_generatedRoot_7.w_78_1 ctx), ((parallel_generatedRoot_7.w_79_0 ctx), ((parallel_generatedRoot_7.w_79_1 ctx), ((parallel_generatedRoot_7.w_80_0 ctx), ((parallel_generatedRoot_7.w_80_1 ctx), ((parallel_generatedRoot_7.w_81_0 ctx), ((parallel_generatedRoot_7.w_81_1 ctx), ((parallel_generatedRoot_7.w_82_0 ctx), ((parallel_generatedRoot_7.w_82_1 ctx), ((parallel_generatedRoot_7.w_83_0 ctx), ((parallel_generatedRoot_7.w_83_1 ctx), ((parallel_generatedRoot_7.w_84_0 ctx), ((parallel_generatedRoot_7.w_84_1 ctx), ((parallel_generatedRoot_7.w_85_0 ctx), ((parallel_generatedRoot_7.w_85_1 ctx), ((parallel_generatedRoot_7.w_86_0 ctx), ((parallel_generatedRoot_7.w_86_1 ctx), ((parallel_generatedRoot_7.w_87_0 ctx), ((parallel_generatedRoot_7.w_87_1 ctx), ((parallel_generatedRoot_7.w_88_0 ctx), ((parallel_generatedRoot_7.w_88_1 ctx), ((parallel_generatedRoot_7.w_89_0 ctx), ((parallel_generatedRoot_7.w_89_1 ctx), ((parallel_generatedRoot_7.w_90_0 ctx), ((parallel_generatedRoot_7.w_90_1 ctx), ((parallel_generatedRoot_7.w_91_0 ctx), ((parallel_generatedRoot_7.w_91_1 ctx), ((parallel_generatedRoot_7.w_92_0 ctx), ((parallel_generatedRoot_7.w_92_1 ctx), ((parallel_generatedRoot_7.w_93_0 ctx), ((parallel_generatedRoot_7.w_93_1 ctx), ((parallel_generatedRoot_7.w_94_0 ctx), ((parallel_generatedRoot_7.w_94_1 ctx), ((parallel_generatedRoot_7.w_95_0 ctx), ((parallel_generatedRoot_7.w_95_1 ctx), ((parallel_generatedRoot_7.w_96_0 ctx), ((parallel_generatedRoot_7.w_96_1 ctx), ((parallel_generatedRoot_7.w_97_0 ctx), ((parallel_generatedRoot_7.w_97_1 ctx), ((parallel_generatedRoot_7.w_98_0 ctx), ((parallel_generatedRoot_7.w_98_1 ctx), ((parallel_generatedRoot_7.w_99_0 ctx), ((parallel_generatedRoot_7.w_99_1 ctx), ((parallel_generatedRoot_7.w_100_0 ctx), ((parallel_generatedRoot_7.w_100_1 ctx), ((parallel_generatedRoot_7.w_101_0 ctx), ((parallel_generatedRoot_7.w_101_1 ctx), ((parallel_generatedRoot_7.w_102_0 ctx), ((parallel_generatedRoot_7.w_102_1 ctx), ((parallel_generatedRoot_7.w_103_0 ctx), ((parallel_generatedRoot_7.w_103_1 ctx), ((parallel_generatedRoot_7.w_104_0 ctx), ((parallel_generatedRoot_7.w_104_1 ctx), ((parallel_generatedRoot_7.w_105_0 ctx), ((parallel_generatedRoot_7.w_105_1 ctx), ((parallel_generatedRoot_7.w_106_0 ctx), ((parallel_generatedRoot_7.w_106_1 ctx), ((parallel_generatedRoot_7.w_107_0 ctx), ((parallel_generatedRoot_7.w_107_1 ctx), ((parallel_generatedRoot_7.w_108_0 ctx), ((parallel_generatedRoot_7.w_108_1 ctx), ((parallel_generatedRoot_7.w_109_0 ctx), ((parallel_generatedRoot_7.w_109_1 ctx), ((parallel_generatedRoot_7.w_110_0 ctx), ((parallel_generatedRoot_7.w_110_1 ctx), ((parallel_generatedRoot_7.w_111_0 ctx), ((parallel_generatedRoot_7.w_111_1 ctx), ())))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))

structure parallel_generatedRoot_7.Facts_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Prop where
  h1_0 : MxxRuntime.centeredRebaseRuns (parallel_generatedRoot_7.w_0_0 ctx) (parallel_generatedRoot_7.w_1_0 ctx)
  h3_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 105, 110, 112, 117, 116, 47, 87, 105, 114, 101, 73, 100, 40, 48, 41, 47, 48]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((parallel_generatedRoot_7.w_2_0 ctx)) (parallel_generatedRoot_7.w_3_0 ctx)
  h6_0 : MxxRuntime.centeredRebaseRuns (parallel_generatedRoot_7.w_5_0 ctx) (parallel_generatedRoot_7.w_6_0 ctx)
  h7_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 105, 110, 112, 117, 116, 47, 87, 105, 114, 101, 73, 100, 40, 48, 41, 47, 48]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) ((parallel_generatedRoot_7.w_2_0 ctx)) (parallel_generatedRoot_7.w_7_0 ctx)
  h11_0 : MxxRuntime.centeredRebaseRuns (parallel_generatedRoot_7.w_10_0 ctx) (parallel_generatedRoot_7.w_11_0 ctx)
  h12_0 : MxxRuntime.gadgetMatrixRuns backend 1024 168 (parallel_generatedRoot_7.w_12_0 ctx)
  h13_0 : MxxRuntime.centeredRebaseRuns (parallel_generatedRoot_7.w_12_0 ctx) (parallel_generatedRoot_7.w_13_0 ctx)
  h23_0 : MxxRuntime.gaussianSampleAt tape (path ++ [23]) ((4/1)) (26) (parallel_generatedRoot_7.w_23_0 ctx)
  h25_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 105, 110, 112, 117, 116, 47, 87, 105, 114, 101, 73, 100, 40, 48, 41, 47, 48]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((parallel_generatedRoot_7.w_2_0 ctx)) (parallel_generatedRoot_7.w_25_0 ctx)
  h27_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 105, 110, 112, 117, 116, 47, 87, 105, 114, 101, 73, 100, 40, 48, 41, 47, 48]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) ((parallel_generatedRoot_7.w_2_0 ctx)) (parallel_generatedRoot_7.w_27_0 ctx)
  h31_0 : MxxRuntime.centeredRebaseRuns (parallel_generatedRoot_7.w_30_0 ctx) (parallel_generatedRoot_7.w_31_0 ctx)
  h35_0 : MxxRuntime.gaussianSampleAt tape (path ++ [35]) ((4/1)) (26) (parallel_generatedRoot_7.w_35_0 ctx)
  t12 : (1025) = 1025
  h39_2 : (∀ i : Fin 1025, parallel_parallel_generatedRoot_7_39 backend hashModel tape (path ++ [39, i.val]) params i_0 i ((parallel_generatedRoot_7.w_1_0 ctx), ((parallel_generatedRoot_7.w_2_0 ctx), ((parallel_generatedRoot_7.w_6_0 ctx), (((parallel_generatedRoot_7.w_37_0 ctx) i), (((parallel_generatedRoot_7.w_38_0 ctx) i), ()))))) (((parallel_generatedRoot_7.w_39_0 ctx) i), (((parallel_generatedRoot_7.w_39_1 ctx) i), (((parallel_generatedRoot_7.w_39_2 ctx) i), ()))))
  t14 : (128) = 128
  h40_1 : (∀ i : Fin 128, parallel_parallel_generatedRoot_7_40 backend hashModel tape (path ++ [40, i.val]) params i_0 i ((parallel_generatedRoot_7.w_1_0 ctx), ((parallel_generatedRoot_7.w_2_0 ctx), ((parallel_generatedRoot_7.w_6_0 ctx), ()))) (((parallel_generatedRoot_7.w_40_0 ctx) i), (((parallel_generatedRoot_7.w_40_1 ctx) i), ())))
  t16 : (32) = 32
  h41_1 : (∀ i : Fin 32, parallel_parallel_generatedRoot_7_41 backend hashModel tape (path ++ [41, i.val]) params i_0 i ((parallel_generatedRoot_7.w_1_0 ctx), ((parallel_generatedRoot_7.w_2_0 ctx), ((parallel_generatedRoot_7.w_6_0 ctx), ()))) (((parallel_generatedRoot_7.w_41_0 ctx) i), (((parallel_generatedRoot_7.w_41_1 ctx) i), ())))
  t18 : (128) = 128
  h42_1 : (∀ i : Fin 128, parallel_parallel_generatedRoot_7_42 backend hashModel tape (path ++ [42, i.val]) params i_0 i ((parallel_generatedRoot_7.w_1_0 ctx), ((parallel_generatedRoot_7.w_2_0 ctx), ((parallel_generatedRoot_7.w_6_0 ctx), ()))) (((parallel_generatedRoot_7.w_42_0 ctx) i), (((parallel_generatedRoot_7.w_42_1 ctx) i), ())))
  t20 : (32) = 32
  h43_1 : (∀ i : Fin 32, parallel_parallel_generatedRoot_7_43 backend hashModel tape (path ++ [43, i.val]) params i_0 i ((parallel_generatedRoot_7.w_1_0 ctx), ((parallel_generatedRoot_7.w_2_0 ctx), ((parallel_generatedRoot_7.w_6_0 ctx), ()))) (((parallel_generatedRoot_7.w_43_0 ctx) i), (((parallel_generatedRoot_7.w_43_1 ctx) i), ())))
  t22 : (128) = 128
  h44_1 : (∀ i : Fin 128, parallel_parallel_generatedRoot_7_44 backend hashModel tape (path ++ [44, i.val]) params i_0 i ((parallel_generatedRoot_7.w_1_0 ctx), ((parallel_generatedRoot_7.w_2_0 ctx), ((parallel_generatedRoot_7.w_6_0 ctx), ()))) (((parallel_generatedRoot_7.w_44_0 ctx) i), (((parallel_generatedRoot_7.w_44_1 ctx) i), ())))
  t24 : (32) = 32
  h45_1 : (∀ i : Fin 32, parallel_parallel_generatedRoot_7_45 backend hashModel tape (path ++ [45, i.val]) params i_0 i ((parallel_generatedRoot_7.w_1_0 ctx), ((parallel_generatedRoot_7.w_2_0 ctx), ((parallel_generatedRoot_7.w_6_0 ctx), ()))) (((parallel_generatedRoot_7.w_45_0 ctx) i), (((parallel_generatedRoot_7.w_45_1 ctx) i), ())))
  t26 : (128) = 128
  h46_1 : (∀ i : Fin 128, parallel_parallel_generatedRoot_7_46 backend hashModel tape (path ++ [46, i.val]) params i_0 i ((parallel_generatedRoot_7.w_1_0 ctx), ((parallel_generatedRoot_7.w_2_0 ctx), ((parallel_generatedRoot_7.w_6_0 ctx), ()))) (((parallel_generatedRoot_7.w_46_0 ctx) i), (((parallel_generatedRoot_7.w_46_1 ctx) i), ())))
  t28 : (32) = 32
  h47_1 : (∀ i : Fin 32, parallel_parallel_generatedRoot_7_47 backend hashModel tape (path ++ [47, i.val]) params i_0 i ((parallel_generatedRoot_7.w_1_0 ctx), ((parallel_generatedRoot_7.w_2_0 ctx), ((parallel_generatedRoot_7.w_6_0 ctx), ()))) (((parallel_generatedRoot_7.w_47_0 ctx) i), (((parallel_generatedRoot_7.w_47_1 ctx) i), ())))
  t30 : (128) = 128
  h48_1 : (∀ i : Fin 128, parallel_parallel_generatedRoot_7_48 backend hashModel tape (path ++ [48, i.val]) params i_0 i ((parallel_generatedRoot_7.w_1_0 ctx), ((parallel_generatedRoot_7.w_2_0 ctx), ((parallel_generatedRoot_7.w_6_0 ctx), ()))) (((parallel_generatedRoot_7.w_48_0 ctx) i), (((parallel_generatedRoot_7.w_48_1 ctx) i), ())))

theorem parallel_generatedRoot_7.facts_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} {ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0} {outputs : mxxMat2 × mxxMat2 × (Fin 1025 → mxxMat4) × (Fin 1025 → mxxMat4) × (Fin 1025 → mxxMat1) × (Fin 128 → mxxMat2) × (Fin 128 → mxxMat2) × (Fin 32 → mxxMat2) × (Fin 32 → mxxMat2) × (Fin 128 → mxxMat5) × (Fin 128 → mxxMat5) × (Fin 32 → mxxMat5) × (Fin 32 → mxxMat5) × (Fin 128 → mxxMat6) × (Fin 128 → mxxMat6) × (Fin 32 → mxxMat6) × (Fin 32 → mxxMat6) × (Fin 128 → mxxMat7) × (Fin 128 → mxxMat7) × (Fin 32 → mxxMat7) × (Fin 32 → mxxMat7) × (Fin 128 → mxxMat8) × (Fin 128 → mxxMat8) × (Fin 32 → mxxMat8) × (Fin 32 → mxxMat8) × (Fin 128 → mxxMat9) × (Fin 128 → mxxMat9) × (Fin 32 → mxxMat9) × (Fin 32 → mxxMat9) × (Fin 128 → mxxMat10) × (Fin 128 → mxxMat10) × (Fin 32 → mxxMat10) × (Fin 32 → mxxMat10) × (Fin 128 → mxxMat11) × (Fin 128 → mxxMat11) × (Fin 32 → mxxMat11) × (Fin 32 → mxxMat11) × (Fin 128 → mxxMat12) × (Fin 128 → mxxMat12) × (Fin 32 → mxxMat12) × (Fin 32 → mxxMat12) × (Fin 128 → mxxMat13) × (Fin 128 → mxxMat13) × (Fin 32 → mxxMat13) × (Fin 32 → mxxMat13) × (Fin 128 → mxxMat14) × (Fin 128 → mxxMat14) × (Fin 32 → mxxMat14) × (Fin 32 → mxxMat14) × (Fin 128 → mxxMat15) × (Fin 128 → mxxMat15) × (Fin 32 → mxxMat15) × (Fin 32 → mxxMat15) × (Fin 128 → mxxMat16) × (Fin 128 → mxxMat16) × (Fin 32 → mxxMat16) × (Fin 32 → mxxMat16) × (Fin 128 → mxxMat17) × (Fin 128 → mxxMat17) × (Fin 32 → mxxMat17) × (Fin 32 → mxxMat17) × (Fin 128 → mxxMat18) × (Fin 128 → mxxMat18) × (Fin 32 → mxxMat18) × (Fin 32 → mxxMat18) × (Fin 128 → mxxMat19) × (Fin 128 → mxxMat19) × (Fin 32 → mxxMat19) × (Fin 32 → mxxMat19) × (Fin 128 → mxxMat20) × (Fin 128 → mxxMat20) × (Fin 32 → mxxMat20) × (Fin 32 → mxxMat20) × (Fin 128 → mxxMat21) × (Fin 128 → mxxMat21) × (Fin 32 → mxxMat21) × (Fin 32 → mxxMat21) × (Fin 128 → mxxMat22) × (Fin 128 → mxxMat22) × (Fin 32 → mxxMat22) × (Fin 32 → mxxMat22) × (Fin 128 → mxxMat23) × (Fin 128 → mxxMat23) × (Fin 32 → mxxMat23) × (Fin 32 → mxxMat23) × (Fin 128 → mxxMat24) × (Fin 128 → mxxMat24) × (Fin 32 → mxxMat24) × (Fin 32 → mxxMat24) × (Fin 128 → mxxMat25) × (Fin 128 → mxxMat25) × (Fin 32 → mxxMat25) × (Fin 32 → mxxMat25) × (Fin 128 → mxxMat26) × (Fin 128 → mxxMat26) × (Fin 32 → mxxMat26) × (Fin 32 → mxxMat26) × (Fin 128 → mxxMat27) × (Fin 128 → mxxMat27) × (Fin 32 → mxxMat27) × (Fin 32 → mxxMat27) × (Fin 128 → mxxMat28) × (Fin 128 → mxxMat28) × (Fin 32 → mxxMat28) × (Fin 32 → mxxMat28) × (Fin 128 → mxxMat29) × (Fin 128 → mxxMat29) × (Fin 32 → mxxMat29) × (Fin 32 → mxxMat29) × (Fin 128 → mxxMat30) × (Fin 128 → mxxMat30) × (Fin 32 → mxxMat30) × (Fin 32 → mxxMat30) × (Fin 128 → mxxMat31) × (Fin 128 → mxxMat31) × (Fin 32 → mxxMat31) × (Fin 32 → mxxMat31) × (Fin 128 → mxxMat32) × (Fin 128 → mxxMat32) × (Fin 32 → mxxMat32) × (Fin 32 → mxxMat32) × (Fin 128 → mxxMat33) × (Fin 128 → mxxMat33) × (Fin 32 → mxxMat33) × (Fin 32 → mxxMat33) × (Fin 128 → mxxMat34) × (Fin 128 → mxxMat34) × (Fin 32 → mxxMat34) × (Fin 32 → mxxMat34) × (Fin 128 → mxxMat35) × (Fin 128 → mxxMat35) × (Fin 32 → mxxMat35) × (Fin 32 → mxxMat35) × (Fin 128 → mxxMat36) × (Fin 128 → mxxMat36) × (Fin 32 → mxxMat36) × (Fin 32 → mxxMat36) × (Fin 128 → mxxMat37) × (Fin 128 → mxxMat37) × (Fin 32 → mxxMat37) × (Fin 32 → mxxMat37) × (Fin 128 → mxxMat38) × (Fin 128 → mxxMat38) × (Fin 32 → mxxMat38) × (Fin 32 → mxxMat38) × (Fin 128 → mxxMat39) × (Fin 128 → mxxMat39) × (Fin 32 → mxxMat39) × (Fin 32 → mxxMat39) × Unit}
    (h : parallel_generatedRoot_7.constraints_0 backend hashModel tape path params i_0 (parallel_generatedRoot_7.w_0_0 ctx) (parallel_generatedRoot_7.w_2_0 ctx) (parallel_generatedRoot_7.w_10_0 ctx) (parallel_generatedRoot_7.w_37_0 ctx) (parallel_generatedRoot_7.w_38_0 ctx) (parallel_generatedRoot_7.w_1_0 ctx) (parallel_generatedRoot_7.w_3_0 ctx) (parallel_generatedRoot_7.w_5_0 ctx) (parallel_generatedRoot_7.w_6_0 ctx) (parallel_generatedRoot_7.w_7_0 ctx) (parallel_generatedRoot_7.w_11_0 ctx) (parallel_generatedRoot_7.w_12_0 ctx) (parallel_generatedRoot_7.w_13_0 ctx) (parallel_generatedRoot_7.w_23_0 ctx) (parallel_generatedRoot_7.w_25_0 ctx) (parallel_generatedRoot_7.w_27_0 ctx) (parallel_generatedRoot_7.w_30_0 ctx) (parallel_generatedRoot_7.w_31_0 ctx) (parallel_generatedRoot_7.w_35_0 ctx) (parallel_generatedRoot_7.w_39_0 ctx) (parallel_generatedRoot_7.w_39_1 ctx) (parallel_generatedRoot_7.w_39_2 ctx) (parallel_generatedRoot_7.w_40_0 ctx) (parallel_generatedRoot_7.w_40_1 ctx) (parallel_generatedRoot_7.w_41_0 ctx) (parallel_generatedRoot_7.w_41_1 ctx) (parallel_generatedRoot_7.w_42_0 ctx) (parallel_generatedRoot_7.w_42_1 ctx) (parallel_generatedRoot_7.w_43_0 ctx) (parallel_generatedRoot_7.w_43_1 ctx) (parallel_generatedRoot_7.w_44_0 ctx) (parallel_generatedRoot_7.w_44_1 ctx) (parallel_generatedRoot_7.w_45_0 ctx) (parallel_generatedRoot_7.w_45_1 ctx) (parallel_generatedRoot_7.w_46_0 ctx) (parallel_generatedRoot_7.w_46_1 ctx) (parallel_generatedRoot_7.w_47_0 ctx) (parallel_generatedRoot_7.w_47_1 ctx) (parallel_generatedRoot_7.w_48_0 ctx) (parallel_generatedRoot_7.w_48_1 ctx)) :
    parallel_generatedRoot_7.Facts_0 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_7.Facts_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Prop where
  t32 : (32) = 32
  h49_1 : (∀ i : Fin 32, parallel_parallel_generatedRoot_7_49 backend hashModel tape (path ++ [49, i.val]) params i_0 i ((parallel_generatedRoot_7.w_1_0 ctx), ((parallel_generatedRoot_7.w_2_0 ctx), ((parallel_generatedRoot_7.w_6_0 ctx), ()))) (((parallel_generatedRoot_7.w_49_0 ctx) i), (((parallel_generatedRoot_7.w_49_1 ctx) i), ())))
  t34 : (128) = 128
  h50_1 : (∀ i : Fin 128, parallel_parallel_generatedRoot_7_50 backend hashModel tape (path ++ [50, i.val]) params i_0 i ((parallel_generatedRoot_7.w_1_0 ctx), ((parallel_generatedRoot_7.w_2_0 ctx), ((parallel_generatedRoot_7.w_6_0 ctx), ()))) (((parallel_generatedRoot_7.w_50_0 ctx) i), (((parallel_generatedRoot_7.w_50_1 ctx) i), ())))
  t36 : (32) = 32
  h51_1 : (∀ i : Fin 32, parallel_parallel_generatedRoot_7_51 backend hashModel tape (path ++ [51, i.val]) params i_0 i ((parallel_generatedRoot_7.w_1_0 ctx), ((parallel_generatedRoot_7.w_2_0 ctx), ((parallel_generatedRoot_7.w_6_0 ctx), ()))) (((parallel_generatedRoot_7.w_51_0 ctx) i), (((parallel_generatedRoot_7.w_51_1 ctx) i), ())))
  t38 : (128) = 128
  h52_1 : (∀ i : Fin 128, parallel_parallel_generatedRoot_7_52 backend hashModel tape (path ++ [52, i.val]) params i_0 i ((parallel_generatedRoot_7.w_1_0 ctx), ((parallel_generatedRoot_7.w_2_0 ctx), ((parallel_generatedRoot_7.w_6_0 ctx), ()))) (((parallel_generatedRoot_7.w_52_0 ctx) i), (((parallel_generatedRoot_7.w_52_1 ctx) i), ())))
  t40 : (32) = 32
  h53_1 : (∀ i : Fin 32, parallel_parallel_generatedRoot_7_53 backend hashModel tape (path ++ [53, i.val]) params i_0 i ((parallel_generatedRoot_7.w_1_0 ctx), ((parallel_generatedRoot_7.w_2_0 ctx), ((parallel_generatedRoot_7.w_6_0 ctx), ()))) (((parallel_generatedRoot_7.w_53_0 ctx) i), (((parallel_generatedRoot_7.w_53_1 ctx) i), ())))
  t42 : (128) = 128
  h54_1 : (∀ i : Fin 128, parallel_parallel_generatedRoot_7_54 backend hashModel tape (path ++ [54, i.val]) params i_0 i ((parallel_generatedRoot_7.w_1_0 ctx), ((parallel_generatedRoot_7.w_2_0 ctx), ((parallel_generatedRoot_7.w_6_0 ctx), ()))) (((parallel_generatedRoot_7.w_54_0 ctx) i), (((parallel_generatedRoot_7.w_54_1 ctx) i), ())))
  t44 : (32) = 32
  h55_1 : (∀ i : Fin 32, parallel_parallel_generatedRoot_7_55 backend hashModel tape (path ++ [55, i.val]) params i_0 i ((parallel_generatedRoot_7.w_1_0 ctx), ((parallel_generatedRoot_7.w_2_0 ctx), ((parallel_generatedRoot_7.w_6_0 ctx), ()))) (((parallel_generatedRoot_7.w_55_0 ctx) i), (((parallel_generatedRoot_7.w_55_1 ctx) i), ())))
  t46 : (128) = 128
  h56_1 : (∀ i : Fin 128, parallel_parallel_generatedRoot_7_56 backend hashModel tape (path ++ [56, i.val]) params i_0 i ((parallel_generatedRoot_7.w_1_0 ctx), ((parallel_generatedRoot_7.w_2_0 ctx), ((parallel_generatedRoot_7.w_6_0 ctx), ()))) (((parallel_generatedRoot_7.w_56_0 ctx) i), (((parallel_generatedRoot_7.w_56_1 ctx) i), ())))
  t48 : (32) = 32
  h57_1 : (∀ i : Fin 32, parallel_parallel_generatedRoot_7_57 backend hashModel tape (path ++ [57, i.val]) params i_0 i ((parallel_generatedRoot_7.w_1_0 ctx), ((parallel_generatedRoot_7.w_2_0 ctx), ((parallel_generatedRoot_7.w_6_0 ctx), ()))) (((parallel_generatedRoot_7.w_57_0 ctx) i), (((parallel_generatedRoot_7.w_57_1 ctx) i), ())))
  t50 : (128) = 128
  h58_1 : (∀ i : Fin 128, parallel_parallel_generatedRoot_7_58 backend hashModel tape (path ++ [58, i.val]) params i_0 i ((parallel_generatedRoot_7.w_1_0 ctx), ((parallel_generatedRoot_7.w_2_0 ctx), ((parallel_generatedRoot_7.w_6_0 ctx), ()))) (((parallel_generatedRoot_7.w_58_0 ctx) i), (((parallel_generatedRoot_7.w_58_1 ctx) i), ())))
  t52 : (32) = 32
  h59_1 : (∀ i : Fin 32, parallel_parallel_generatedRoot_7_59 backend hashModel tape (path ++ [59, i.val]) params i_0 i ((parallel_generatedRoot_7.w_1_0 ctx), ((parallel_generatedRoot_7.w_2_0 ctx), ((parallel_generatedRoot_7.w_6_0 ctx), ()))) (((parallel_generatedRoot_7.w_59_0 ctx) i), (((parallel_generatedRoot_7.w_59_1 ctx) i), ())))
  t54 : (128) = 128
  h60_1 : (∀ i : Fin 128, parallel_parallel_generatedRoot_7_60 backend hashModel tape (path ++ [60, i.val]) params i_0 i ((parallel_generatedRoot_7.w_1_0 ctx), ((parallel_generatedRoot_7.w_2_0 ctx), ((parallel_generatedRoot_7.w_6_0 ctx), ()))) (((parallel_generatedRoot_7.w_60_0 ctx) i), (((parallel_generatedRoot_7.w_60_1 ctx) i), ())))
  t56 : (32) = 32
  h61_1 : (∀ i : Fin 32, parallel_parallel_generatedRoot_7_61 backend hashModel tape (path ++ [61, i.val]) params i_0 i ((parallel_generatedRoot_7.w_1_0 ctx), ((parallel_generatedRoot_7.w_2_0 ctx), ((parallel_generatedRoot_7.w_6_0 ctx), ()))) (((parallel_generatedRoot_7.w_61_0 ctx) i), (((parallel_generatedRoot_7.w_61_1 ctx) i), ())))
  t58 : (128) = 128
  h62_1 : (∀ i : Fin 128, parallel_parallel_generatedRoot_7_62 backend hashModel tape (path ++ [62, i.val]) params i_0 i ((parallel_generatedRoot_7.w_1_0 ctx), ((parallel_generatedRoot_7.w_2_0 ctx), ((parallel_generatedRoot_7.w_6_0 ctx), ()))) (((parallel_generatedRoot_7.w_62_0 ctx) i), (((parallel_generatedRoot_7.w_62_1 ctx) i), ())))
  t60 : (32) = 32
  h63_1 : (∀ i : Fin 32, parallel_parallel_generatedRoot_7_63 backend hashModel tape (path ++ [63, i.val]) params i_0 i ((parallel_generatedRoot_7.w_1_0 ctx), ((parallel_generatedRoot_7.w_2_0 ctx), ((parallel_generatedRoot_7.w_6_0 ctx), ()))) (((parallel_generatedRoot_7.w_63_0 ctx) i), (((parallel_generatedRoot_7.w_63_1 ctx) i), ())))
  t62 : (128) = 128
  h64_1 : (∀ i : Fin 128, parallel_parallel_generatedRoot_7_64 backend hashModel tape (path ++ [64, i.val]) params i_0 i ((parallel_generatedRoot_7.w_1_0 ctx), ((parallel_generatedRoot_7.w_2_0 ctx), ((parallel_generatedRoot_7.w_6_0 ctx), ()))) (((parallel_generatedRoot_7.w_64_0 ctx) i), (((parallel_generatedRoot_7.w_64_1 ctx) i), ())))

theorem parallel_generatedRoot_7.facts_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} {ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0} {outputs : mxxMat2 × mxxMat2 × (Fin 1025 → mxxMat4) × (Fin 1025 → mxxMat4) × (Fin 1025 → mxxMat1) × (Fin 128 → mxxMat2) × (Fin 128 → mxxMat2) × (Fin 32 → mxxMat2) × (Fin 32 → mxxMat2) × (Fin 128 → mxxMat5) × (Fin 128 → mxxMat5) × (Fin 32 → mxxMat5) × (Fin 32 → mxxMat5) × (Fin 128 → mxxMat6) × (Fin 128 → mxxMat6) × (Fin 32 → mxxMat6) × (Fin 32 → mxxMat6) × (Fin 128 → mxxMat7) × (Fin 128 → mxxMat7) × (Fin 32 → mxxMat7) × (Fin 32 → mxxMat7) × (Fin 128 → mxxMat8) × (Fin 128 → mxxMat8) × (Fin 32 → mxxMat8) × (Fin 32 → mxxMat8) × (Fin 128 → mxxMat9) × (Fin 128 → mxxMat9) × (Fin 32 → mxxMat9) × (Fin 32 → mxxMat9) × (Fin 128 → mxxMat10) × (Fin 128 → mxxMat10) × (Fin 32 → mxxMat10) × (Fin 32 → mxxMat10) × (Fin 128 → mxxMat11) × (Fin 128 → mxxMat11) × (Fin 32 → mxxMat11) × (Fin 32 → mxxMat11) × (Fin 128 → mxxMat12) × (Fin 128 → mxxMat12) × (Fin 32 → mxxMat12) × (Fin 32 → mxxMat12) × (Fin 128 → mxxMat13) × (Fin 128 → mxxMat13) × (Fin 32 → mxxMat13) × (Fin 32 → mxxMat13) × (Fin 128 → mxxMat14) × (Fin 128 → mxxMat14) × (Fin 32 → mxxMat14) × (Fin 32 → mxxMat14) × (Fin 128 → mxxMat15) × (Fin 128 → mxxMat15) × (Fin 32 → mxxMat15) × (Fin 32 → mxxMat15) × (Fin 128 → mxxMat16) × (Fin 128 → mxxMat16) × (Fin 32 → mxxMat16) × (Fin 32 → mxxMat16) × (Fin 128 → mxxMat17) × (Fin 128 → mxxMat17) × (Fin 32 → mxxMat17) × (Fin 32 → mxxMat17) × (Fin 128 → mxxMat18) × (Fin 128 → mxxMat18) × (Fin 32 → mxxMat18) × (Fin 32 → mxxMat18) × (Fin 128 → mxxMat19) × (Fin 128 → mxxMat19) × (Fin 32 → mxxMat19) × (Fin 32 → mxxMat19) × (Fin 128 → mxxMat20) × (Fin 128 → mxxMat20) × (Fin 32 → mxxMat20) × (Fin 32 → mxxMat20) × (Fin 128 → mxxMat21) × (Fin 128 → mxxMat21) × (Fin 32 → mxxMat21) × (Fin 32 → mxxMat21) × (Fin 128 → mxxMat22) × (Fin 128 → mxxMat22) × (Fin 32 → mxxMat22) × (Fin 32 → mxxMat22) × (Fin 128 → mxxMat23) × (Fin 128 → mxxMat23) × (Fin 32 → mxxMat23) × (Fin 32 → mxxMat23) × (Fin 128 → mxxMat24) × (Fin 128 → mxxMat24) × (Fin 32 → mxxMat24) × (Fin 32 → mxxMat24) × (Fin 128 → mxxMat25) × (Fin 128 → mxxMat25) × (Fin 32 → mxxMat25) × (Fin 32 → mxxMat25) × (Fin 128 → mxxMat26) × (Fin 128 → mxxMat26) × (Fin 32 → mxxMat26) × (Fin 32 → mxxMat26) × (Fin 128 → mxxMat27) × (Fin 128 → mxxMat27) × (Fin 32 → mxxMat27) × (Fin 32 → mxxMat27) × (Fin 128 → mxxMat28) × (Fin 128 → mxxMat28) × (Fin 32 → mxxMat28) × (Fin 32 → mxxMat28) × (Fin 128 → mxxMat29) × (Fin 128 → mxxMat29) × (Fin 32 → mxxMat29) × (Fin 32 → mxxMat29) × (Fin 128 → mxxMat30) × (Fin 128 → mxxMat30) × (Fin 32 → mxxMat30) × (Fin 32 → mxxMat30) × (Fin 128 → mxxMat31) × (Fin 128 → mxxMat31) × (Fin 32 → mxxMat31) × (Fin 32 → mxxMat31) × (Fin 128 → mxxMat32) × (Fin 128 → mxxMat32) × (Fin 32 → mxxMat32) × (Fin 32 → mxxMat32) × (Fin 128 → mxxMat33) × (Fin 128 → mxxMat33) × (Fin 32 → mxxMat33) × (Fin 32 → mxxMat33) × (Fin 128 → mxxMat34) × (Fin 128 → mxxMat34) × (Fin 32 → mxxMat34) × (Fin 32 → mxxMat34) × (Fin 128 → mxxMat35) × (Fin 128 → mxxMat35) × (Fin 32 → mxxMat35) × (Fin 32 → mxxMat35) × (Fin 128 → mxxMat36) × (Fin 128 → mxxMat36) × (Fin 32 → mxxMat36) × (Fin 32 → mxxMat36) × (Fin 128 → mxxMat37) × (Fin 128 → mxxMat37) × (Fin 32 → mxxMat37) × (Fin 32 → mxxMat37) × (Fin 128 → mxxMat38) × (Fin 128 → mxxMat38) × (Fin 32 → mxxMat38) × (Fin 32 → mxxMat38) × (Fin 128 → mxxMat39) × (Fin 128 → mxxMat39) × (Fin 32 → mxxMat39) × (Fin 32 → mxxMat39) × Unit}
    (h : parallel_generatedRoot_7.constraints_1 backend hashModel tape path params i_0 (parallel_generatedRoot_7.w_2_0 ctx) (parallel_generatedRoot_7.w_1_0 ctx) (parallel_generatedRoot_7.w_6_0 ctx) (parallel_generatedRoot_7.w_49_0 ctx) (parallel_generatedRoot_7.w_49_1 ctx) (parallel_generatedRoot_7.w_50_0 ctx) (parallel_generatedRoot_7.w_50_1 ctx) (parallel_generatedRoot_7.w_51_0 ctx) (parallel_generatedRoot_7.w_51_1 ctx) (parallel_generatedRoot_7.w_52_0 ctx) (parallel_generatedRoot_7.w_52_1 ctx) (parallel_generatedRoot_7.w_53_0 ctx) (parallel_generatedRoot_7.w_53_1 ctx) (parallel_generatedRoot_7.w_54_0 ctx) (parallel_generatedRoot_7.w_54_1 ctx) (parallel_generatedRoot_7.w_55_0 ctx) (parallel_generatedRoot_7.w_55_1 ctx) (parallel_generatedRoot_7.w_56_0 ctx) (parallel_generatedRoot_7.w_56_1 ctx) (parallel_generatedRoot_7.w_57_0 ctx) (parallel_generatedRoot_7.w_57_1 ctx) (parallel_generatedRoot_7.w_58_0 ctx) (parallel_generatedRoot_7.w_58_1 ctx) (parallel_generatedRoot_7.w_59_0 ctx) (parallel_generatedRoot_7.w_59_1 ctx) (parallel_generatedRoot_7.w_60_0 ctx) (parallel_generatedRoot_7.w_60_1 ctx) (parallel_generatedRoot_7.w_61_0 ctx) (parallel_generatedRoot_7.w_61_1 ctx) (parallel_generatedRoot_7.w_62_0 ctx) (parallel_generatedRoot_7.w_62_1 ctx) (parallel_generatedRoot_7.w_63_0 ctx) (parallel_generatedRoot_7.w_63_1 ctx) (parallel_generatedRoot_7.w_64_0 ctx) (parallel_generatedRoot_7.w_64_1 ctx)) :
    parallel_generatedRoot_7.Facts_1 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_7.Facts_2 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Prop where
  t64 : (32) = 32
  h65_1 : (∀ i : Fin 32, parallel_parallel_generatedRoot_7_65 backend hashModel tape (path ++ [65, i.val]) params i_0 i ((parallel_generatedRoot_7.w_1_0 ctx), ((parallel_generatedRoot_7.w_2_0 ctx), ((parallel_generatedRoot_7.w_6_0 ctx), ()))) (((parallel_generatedRoot_7.w_65_0 ctx) i), (((parallel_generatedRoot_7.w_65_1 ctx) i), ())))
  t66 : (128) = 128
  h66_1 : (∀ i : Fin 128, parallel_parallel_generatedRoot_7_66 backend hashModel tape (path ++ [66, i.val]) params i_0 i ((parallel_generatedRoot_7.w_1_0 ctx), ((parallel_generatedRoot_7.w_2_0 ctx), ((parallel_generatedRoot_7.w_6_0 ctx), ()))) (((parallel_generatedRoot_7.w_66_0 ctx) i), (((parallel_generatedRoot_7.w_66_1 ctx) i), ())))
  t68 : (32) = 32
  h67_1 : (∀ i : Fin 32, parallel_parallel_generatedRoot_7_67 backend hashModel tape (path ++ [67, i.val]) params i_0 i ((parallel_generatedRoot_7.w_1_0 ctx), ((parallel_generatedRoot_7.w_2_0 ctx), ((parallel_generatedRoot_7.w_6_0 ctx), ()))) (((parallel_generatedRoot_7.w_67_0 ctx) i), (((parallel_generatedRoot_7.w_67_1 ctx) i), ())))
  t70 : (128) = 128
  h68_1 : (∀ i : Fin 128, parallel_parallel_generatedRoot_7_68 backend hashModel tape (path ++ [68, i.val]) params i_0 i ((parallel_generatedRoot_7.w_1_0 ctx), ((parallel_generatedRoot_7.w_2_0 ctx), ((parallel_generatedRoot_7.w_6_0 ctx), ()))) (((parallel_generatedRoot_7.w_68_0 ctx) i), (((parallel_generatedRoot_7.w_68_1 ctx) i), ())))
  t72 : (32) = 32
  h69_1 : (∀ i : Fin 32, parallel_parallel_generatedRoot_7_69 backend hashModel tape (path ++ [69, i.val]) params i_0 i ((parallel_generatedRoot_7.w_1_0 ctx), ((parallel_generatedRoot_7.w_2_0 ctx), ((parallel_generatedRoot_7.w_6_0 ctx), ()))) (((parallel_generatedRoot_7.w_69_0 ctx) i), (((parallel_generatedRoot_7.w_69_1 ctx) i), ())))
  t74 : (128) = 128
  h70_1 : (∀ i : Fin 128, parallel_parallel_generatedRoot_7_70 backend hashModel tape (path ++ [70, i.val]) params i_0 i ((parallel_generatedRoot_7.w_1_0 ctx), ((parallel_generatedRoot_7.w_2_0 ctx), ((parallel_generatedRoot_7.w_6_0 ctx), ()))) (((parallel_generatedRoot_7.w_70_0 ctx) i), (((parallel_generatedRoot_7.w_70_1 ctx) i), ())))
  t76 : (32) = 32
  h71_1 : (∀ i : Fin 32, parallel_parallel_generatedRoot_7_71 backend hashModel tape (path ++ [71, i.val]) params i_0 i ((parallel_generatedRoot_7.w_1_0 ctx), ((parallel_generatedRoot_7.w_2_0 ctx), ((parallel_generatedRoot_7.w_6_0 ctx), ()))) (((parallel_generatedRoot_7.w_71_0 ctx) i), (((parallel_generatedRoot_7.w_71_1 ctx) i), ())))
  t78 : (128) = 128
  h72_1 : (∀ i : Fin 128, parallel_parallel_generatedRoot_7_72 backend hashModel tape (path ++ [72, i.val]) params i_0 i ((parallel_generatedRoot_7.w_1_0 ctx), ((parallel_generatedRoot_7.w_2_0 ctx), ((parallel_generatedRoot_7.w_6_0 ctx), ()))) (((parallel_generatedRoot_7.w_72_0 ctx) i), (((parallel_generatedRoot_7.w_72_1 ctx) i), ())))
  t80 : (32) = 32
  h73_1 : (∀ i : Fin 32, parallel_parallel_generatedRoot_7_73 backend hashModel tape (path ++ [73, i.val]) params i_0 i ((parallel_generatedRoot_7.w_1_0 ctx), ((parallel_generatedRoot_7.w_2_0 ctx), ((parallel_generatedRoot_7.w_6_0 ctx), ()))) (((parallel_generatedRoot_7.w_73_0 ctx) i), (((parallel_generatedRoot_7.w_73_1 ctx) i), ())))
  t82 : (128) = 128
  h74_1 : (∀ i : Fin 128, parallel_parallel_generatedRoot_7_74 backend hashModel tape (path ++ [74, i.val]) params i_0 i ((parallel_generatedRoot_7.w_1_0 ctx), ((parallel_generatedRoot_7.w_2_0 ctx), ((parallel_generatedRoot_7.w_6_0 ctx), ()))) (((parallel_generatedRoot_7.w_74_0 ctx) i), (((parallel_generatedRoot_7.w_74_1 ctx) i), ())))
  t84 : (32) = 32
  h75_1 : (∀ i : Fin 32, parallel_parallel_generatedRoot_7_75 backend hashModel tape (path ++ [75, i.val]) params i_0 i ((parallel_generatedRoot_7.w_1_0 ctx), ((parallel_generatedRoot_7.w_2_0 ctx), ((parallel_generatedRoot_7.w_6_0 ctx), ()))) (((parallel_generatedRoot_7.w_75_0 ctx) i), (((parallel_generatedRoot_7.w_75_1 ctx) i), ())))
  t86 : (128) = 128
  h76_1 : (∀ i : Fin 128, parallel_parallel_generatedRoot_7_76 backend hashModel tape (path ++ [76, i.val]) params i_0 i ((parallel_generatedRoot_7.w_1_0 ctx), ((parallel_generatedRoot_7.w_2_0 ctx), ((parallel_generatedRoot_7.w_6_0 ctx), ()))) (((parallel_generatedRoot_7.w_76_0 ctx) i), (((parallel_generatedRoot_7.w_76_1 ctx) i), ())))
  t88 : (32) = 32
  h77_1 : (∀ i : Fin 32, parallel_parallel_generatedRoot_7_77 backend hashModel tape (path ++ [77, i.val]) params i_0 i ((parallel_generatedRoot_7.w_1_0 ctx), ((parallel_generatedRoot_7.w_2_0 ctx), ((parallel_generatedRoot_7.w_6_0 ctx), ()))) (((parallel_generatedRoot_7.w_77_0 ctx) i), (((parallel_generatedRoot_7.w_77_1 ctx) i), ())))
  t90 : (128) = 128
  h78_1 : (∀ i : Fin 128, parallel_parallel_generatedRoot_7_78 backend hashModel tape (path ++ [78, i.val]) params i_0 i ((parallel_generatedRoot_7.w_1_0 ctx), ((parallel_generatedRoot_7.w_2_0 ctx), ((parallel_generatedRoot_7.w_6_0 ctx), ()))) (((parallel_generatedRoot_7.w_78_0 ctx) i), (((parallel_generatedRoot_7.w_78_1 ctx) i), ())))
  t92 : (32) = 32
  h79_1 : (∀ i : Fin 32, parallel_parallel_generatedRoot_7_79 backend hashModel tape (path ++ [79, i.val]) params i_0 i ((parallel_generatedRoot_7.w_1_0 ctx), ((parallel_generatedRoot_7.w_2_0 ctx), ((parallel_generatedRoot_7.w_6_0 ctx), ()))) (((parallel_generatedRoot_7.w_79_0 ctx) i), (((parallel_generatedRoot_7.w_79_1 ctx) i), ())))
  t94 : (128) = 128
  h80_1 : (∀ i : Fin 128, parallel_parallel_generatedRoot_7_80 backend hashModel tape (path ++ [80, i.val]) params i_0 i ((parallel_generatedRoot_7.w_1_0 ctx), ((parallel_generatedRoot_7.w_2_0 ctx), ((parallel_generatedRoot_7.w_6_0 ctx), ()))) (((parallel_generatedRoot_7.w_80_0 ctx) i), (((parallel_generatedRoot_7.w_80_1 ctx) i), ())))

theorem parallel_generatedRoot_7.facts_2 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} {ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0} {outputs : mxxMat2 × mxxMat2 × (Fin 1025 → mxxMat4) × (Fin 1025 → mxxMat4) × (Fin 1025 → mxxMat1) × (Fin 128 → mxxMat2) × (Fin 128 → mxxMat2) × (Fin 32 → mxxMat2) × (Fin 32 → mxxMat2) × (Fin 128 → mxxMat5) × (Fin 128 → mxxMat5) × (Fin 32 → mxxMat5) × (Fin 32 → mxxMat5) × (Fin 128 → mxxMat6) × (Fin 128 → mxxMat6) × (Fin 32 → mxxMat6) × (Fin 32 → mxxMat6) × (Fin 128 → mxxMat7) × (Fin 128 → mxxMat7) × (Fin 32 → mxxMat7) × (Fin 32 → mxxMat7) × (Fin 128 → mxxMat8) × (Fin 128 → mxxMat8) × (Fin 32 → mxxMat8) × (Fin 32 → mxxMat8) × (Fin 128 → mxxMat9) × (Fin 128 → mxxMat9) × (Fin 32 → mxxMat9) × (Fin 32 → mxxMat9) × (Fin 128 → mxxMat10) × (Fin 128 → mxxMat10) × (Fin 32 → mxxMat10) × (Fin 32 → mxxMat10) × (Fin 128 → mxxMat11) × (Fin 128 → mxxMat11) × (Fin 32 → mxxMat11) × (Fin 32 → mxxMat11) × (Fin 128 → mxxMat12) × (Fin 128 → mxxMat12) × (Fin 32 → mxxMat12) × (Fin 32 → mxxMat12) × (Fin 128 → mxxMat13) × (Fin 128 → mxxMat13) × (Fin 32 → mxxMat13) × (Fin 32 → mxxMat13) × (Fin 128 → mxxMat14) × (Fin 128 → mxxMat14) × (Fin 32 → mxxMat14) × (Fin 32 → mxxMat14) × (Fin 128 → mxxMat15) × (Fin 128 → mxxMat15) × (Fin 32 → mxxMat15) × (Fin 32 → mxxMat15) × (Fin 128 → mxxMat16) × (Fin 128 → mxxMat16) × (Fin 32 → mxxMat16) × (Fin 32 → mxxMat16) × (Fin 128 → mxxMat17) × (Fin 128 → mxxMat17) × (Fin 32 → mxxMat17) × (Fin 32 → mxxMat17) × (Fin 128 → mxxMat18) × (Fin 128 → mxxMat18) × (Fin 32 → mxxMat18) × (Fin 32 → mxxMat18) × (Fin 128 → mxxMat19) × (Fin 128 → mxxMat19) × (Fin 32 → mxxMat19) × (Fin 32 → mxxMat19) × (Fin 128 → mxxMat20) × (Fin 128 → mxxMat20) × (Fin 32 → mxxMat20) × (Fin 32 → mxxMat20) × (Fin 128 → mxxMat21) × (Fin 128 → mxxMat21) × (Fin 32 → mxxMat21) × (Fin 32 → mxxMat21) × (Fin 128 → mxxMat22) × (Fin 128 → mxxMat22) × (Fin 32 → mxxMat22) × (Fin 32 → mxxMat22) × (Fin 128 → mxxMat23) × (Fin 128 → mxxMat23) × (Fin 32 → mxxMat23) × (Fin 32 → mxxMat23) × (Fin 128 → mxxMat24) × (Fin 128 → mxxMat24) × (Fin 32 → mxxMat24) × (Fin 32 → mxxMat24) × (Fin 128 → mxxMat25) × (Fin 128 → mxxMat25) × (Fin 32 → mxxMat25) × (Fin 32 → mxxMat25) × (Fin 128 → mxxMat26) × (Fin 128 → mxxMat26) × (Fin 32 → mxxMat26) × (Fin 32 → mxxMat26) × (Fin 128 → mxxMat27) × (Fin 128 → mxxMat27) × (Fin 32 → mxxMat27) × (Fin 32 → mxxMat27) × (Fin 128 → mxxMat28) × (Fin 128 → mxxMat28) × (Fin 32 → mxxMat28) × (Fin 32 → mxxMat28) × (Fin 128 → mxxMat29) × (Fin 128 → mxxMat29) × (Fin 32 → mxxMat29) × (Fin 32 → mxxMat29) × (Fin 128 → mxxMat30) × (Fin 128 → mxxMat30) × (Fin 32 → mxxMat30) × (Fin 32 → mxxMat30) × (Fin 128 → mxxMat31) × (Fin 128 → mxxMat31) × (Fin 32 → mxxMat31) × (Fin 32 → mxxMat31) × (Fin 128 → mxxMat32) × (Fin 128 → mxxMat32) × (Fin 32 → mxxMat32) × (Fin 32 → mxxMat32) × (Fin 128 → mxxMat33) × (Fin 128 → mxxMat33) × (Fin 32 → mxxMat33) × (Fin 32 → mxxMat33) × (Fin 128 → mxxMat34) × (Fin 128 → mxxMat34) × (Fin 32 → mxxMat34) × (Fin 32 → mxxMat34) × (Fin 128 → mxxMat35) × (Fin 128 → mxxMat35) × (Fin 32 → mxxMat35) × (Fin 32 → mxxMat35) × (Fin 128 → mxxMat36) × (Fin 128 → mxxMat36) × (Fin 32 → mxxMat36) × (Fin 32 → mxxMat36) × (Fin 128 → mxxMat37) × (Fin 128 → mxxMat37) × (Fin 32 → mxxMat37) × (Fin 32 → mxxMat37) × (Fin 128 → mxxMat38) × (Fin 128 → mxxMat38) × (Fin 32 → mxxMat38) × (Fin 32 → mxxMat38) × (Fin 128 → mxxMat39) × (Fin 128 → mxxMat39) × (Fin 32 → mxxMat39) × (Fin 32 → mxxMat39) × Unit}
    (h : parallel_generatedRoot_7.constraints_2 backend hashModel tape path params i_0 (parallel_generatedRoot_7.w_2_0 ctx) (parallel_generatedRoot_7.w_1_0 ctx) (parallel_generatedRoot_7.w_6_0 ctx) (parallel_generatedRoot_7.w_65_0 ctx) (parallel_generatedRoot_7.w_65_1 ctx) (parallel_generatedRoot_7.w_66_0 ctx) (parallel_generatedRoot_7.w_66_1 ctx) (parallel_generatedRoot_7.w_67_0 ctx) (parallel_generatedRoot_7.w_67_1 ctx) (parallel_generatedRoot_7.w_68_0 ctx) (parallel_generatedRoot_7.w_68_1 ctx) (parallel_generatedRoot_7.w_69_0 ctx) (parallel_generatedRoot_7.w_69_1 ctx) (parallel_generatedRoot_7.w_70_0 ctx) (parallel_generatedRoot_7.w_70_1 ctx) (parallel_generatedRoot_7.w_71_0 ctx) (parallel_generatedRoot_7.w_71_1 ctx) (parallel_generatedRoot_7.w_72_0 ctx) (parallel_generatedRoot_7.w_72_1 ctx) (parallel_generatedRoot_7.w_73_0 ctx) (parallel_generatedRoot_7.w_73_1 ctx) (parallel_generatedRoot_7.w_74_0 ctx) (parallel_generatedRoot_7.w_74_1 ctx) (parallel_generatedRoot_7.w_75_0 ctx) (parallel_generatedRoot_7.w_75_1 ctx) (parallel_generatedRoot_7.w_76_0 ctx) (parallel_generatedRoot_7.w_76_1 ctx) (parallel_generatedRoot_7.w_77_0 ctx) (parallel_generatedRoot_7.w_77_1 ctx) (parallel_generatedRoot_7.w_78_0 ctx) (parallel_generatedRoot_7.w_78_1 ctx) (parallel_generatedRoot_7.w_79_0 ctx) (parallel_generatedRoot_7.w_79_1 ctx) (parallel_generatedRoot_7.w_80_0 ctx) (parallel_generatedRoot_7.w_80_1 ctx)) :
    parallel_generatedRoot_7.Facts_2 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_7.Facts_3 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) : Prop where
  t96 : (32) = 32
  h81_1 : (∀ i : Fin 32, parallel_parallel_generatedRoot_7_81 backend hashModel tape (path ++ [81, i.val]) params i_0 i ((parallel_generatedRoot_7.w_1_0 ctx), ((parallel_generatedRoot_7.w_2_0 ctx), ((parallel_generatedRoot_7.w_6_0 ctx), ()))) (((parallel_generatedRoot_7.w_81_0 ctx) i), (((parallel_generatedRoot_7.w_81_1 ctx) i), ())))
  t98 : (128) = 128
  h82_1 : (∀ i : Fin 128, parallel_parallel_generatedRoot_7_82 backend hashModel tape (path ++ [82, i.val]) params i_0 i ((parallel_generatedRoot_7.w_1_0 ctx), ((parallel_generatedRoot_7.w_2_0 ctx), ((parallel_generatedRoot_7.w_6_0 ctx), ()))) (((parallel_generatedRoot_7.w_82_0 ctx) i), (((parallel_generatedRoot_7.w_82_1 ctx) i), ())))
  t100 : (32) = 32
  h83_1 : (∀ i : Fin 32, parallel_parallel_generatedRoot_7_83 backend hashModel tape (path ++ [83, i.val]) params i_0 i ((parallel_generatedRoot_7.w_1_0 ctx), ((parallel_generatedRoot_7.w_2_0 ctx), ((parallel_generatedRoot_7.w_6_0 ctx), ()))) (((parallel_generatedRoot_7.w_83_0 ctx) i), (((parallel_generatedRoot_7.w_83_1 ctx) i), ())))
  t102 : (128) = 128
  h84_1 : (∀ i : Fin 128, parallel_parallel_generatedRoot_7_84 backend hashModel tape (path ++ [84, i.val]) params i_0 i ((parallel_generatedRoot_7.w_1_0 ctx), ((parallel_generatedRoot_7.w_2_0 ctx), ((parallel_generatedRoot_7.w_6_0 ctx), ()))) (((parallel_generatedRoot_7.w_84_0 ctx) i), (((parallel_generatedRoot_7.w_84_1 ctx) i), ())))
  t104 : (32) = 32
  h85_1 : (∀ i : Fin 32, parallel_parallel_generatedRoot_7_85 backend hashModel tape (path ++ [85, i.val]) params i_0 i ((parallel_generatedRoot_7.w_1_0 ctx), ((parallel_generatedRoot_7.w_2_0 ctx), ((parallel_generatedRoot_7.w_6_0 ctx), ()))) (((parallel_generatedRoot_7.w_85_0 ctx) i), (((parallel_generatedRoot_7.w_85_1 ctx) i), ())))
  t106 : (128) = 128
  h86_1 : (∀ i : Fin 128, parallel_parallel_generatedRoot_7_86 backend hashModel tape (path ++ [86, i.val]) params i_0 i ((parallel_generatedRoot_7.w_1_0 ctx), ((parallel_generatedRoot_7.w_2_0 ctx), ((parallel_generatedRoot_7.w_6_0 ctx), ()))) (((parallel_generatedRoot_7.w_86_0 ctx) i), (((parallel_generatedRoot_7.w_86_1 ctx) i), ())))
  t108 : (32) = 32
  h87_1 : (∀ i : Fin 32, parallel_parallel_generatedRoot_7_87 backend hashModel tape (path ++ [87, i.val]) params i_0 i ((parallel_generatedRoot_7.w_1_0 ctx), ((parallel_generatedRoot_7.w_2_0 ctx), ((parallel_generatedRoot_7.w_6_0 ctx), ()))) (((parallel_generatedRoot_7.w_87_0 ctx) i), (((parallel_generatedRoot_7.w_87_1 ctx) i), ())))
  t110 : (128) = 128
  h88_1 : (∀ i : Fin 128, parallel_parallel_generatedRoot_7_88 backend hashModel tape (path ++ [88, i.val]) params i_0 i ((parallel_generatedRoot_7.w_1_0 ctx), ((parallel_generatedRoot_7.w_2_0 ctx), ((parallel_generatedRoot_7.w_6_0 ctx), ()))) (((parallel_generatedRoot_7.w_88_0 ctx) i), (((parallel_generatedRoot_7.w_88_1 ctx) i), ())))
  t112 : (32) = 32
  h89_1 : (∀ i : Fin 32, parallel_parallel_generatedRoot_7_89 backend hashModel tape (path ++ [89, i.val]) params i_0 i ((parallel_generatedRoot_7.w_1_0 ctx), ((parallel_generatedRoot_7.w_2_0 ctx), ((parallel_generatedRoot_7.w_6_0 ctx), ()))) (((parallel_generatedRoot_7.w_89_0 ctx) i), (((parallel_generatedRoot_7.w_89_1 ctx) i), ())))
  t114 : (128) = 128
  h90_1 : (∀ i : Fin 128, parallel_parallel_generatedRoot_7_90 backend hashModel tape (path ++ [90, i.val]) params i_0 i ((parallel_generatedRoot_7.w_1_0 ctx), ((parallel_generatedRoot_7.w_2_0 ctx), ((parallel_generatedRoot_7.w_6_0 ctx), ()))) (((parallel_generatedRoot_7.w_90_0 ctx) i), (((parallel_generatedRoot_7.w_90_1 ctx) i), ())))
  t116 : (32) = 32
  h91_1 : (∀ i : Fin 32, parallel_parallel_generatedRoot_7_91 backend hashModel tape (path ++ [91, i.val]) params i_0 i ((parallel_generatedRoot_7.w_1_0 ctx), ((parallel_generatedRoot_7.w_2_0 ctx), ((parallel_generatedRoot_7.w_6_0 ctx), ()))) (((parallel_generatedRoot_7.w_91_0 ctx) i), (((parallel_generatedRoot_7.w_91_1 ctx) i), ())))
  t118 : (128) = 128
  h92_1 : (∀ i : Fin 128, parallel_parallel_generatedRoot_7_92 backend hashModel tape (path ++ [92, i.val]) params i_0 i ((parallel_generatedRoot_7.w_1_0 ctx), ((parallel_generatedRoot_7.w_2_0 ctx), ((parallel_generatedRoot_7.w_6_0 ctx), ()))) (((parallel_generatedRoot_7.w_92_0 ctx) i), (((parallel_generatedRoot_7.w_92_1 ctx) i), ())))
  t120 : (32) = 32
  h93_1 : (∀ i : Fin 32, parallel_parallel_generatedRoot_7_93 backend hashModel tape (path ++ [93, i.val]) params i_0 i ((parallel_generatedRoot_7.w_1_0 ctx), ((parallel_generatedRoot_7.w_2_0 ctx), ((parallel_generatedRoot_7.w_6_0 ctx), ()))) (((parallel_generatedRoot_7.w_93_0 ctx) i), (((parallel_generatedRoot_7.w_93_1 ctx) i), ())))
  t122 : (128) = 128
  h94_1 : (∀ i : Fin 128, parallel_parallel_generatedRoot_7_94 backend hashModel tape (path ++ [94, i.val]) params i_0 i ((parallel_generatedRoot_7.w_1_0 ctx), ((parallel_generatedRoot_7.w_2_0 ctx), ((parallel_generatedRoot_7.w_6_0 ctx), ()))) (((parallel_generatedRoot_7.w_94_0 ctx) i), (((parallel_generatedRoot_7.w_94_1 ctx) i), ())))
  t124 : (32) = 32
  h95_1 : (∀ i : Fin 32, parallel_parallel_generatedRoot_7_95 backend hashModel tape (path ++ [95, i.val]) params i_0 i ((parallel_generatedRoot_7.w_1_0 ctx), ((parallel_generatedRoot_7.w_2_0 ctx), ((parallel_generatedRoot_7.w_6_0 ctx), ()))) (((parallel_generatedRoot_7.w_95_0 ctx) i), (((parallel_generatedRoot_7.w_95_1 ctx) i), ())))
  t126 : (128) = 128
  h96_1 : (∀ i : Fin 128, parallel_parallel_generatedRoot_7_96 backend hashModel tape (path ++ [96, i.val]) params i_0 i ((parallel_generatedRoot_7.w_1_0 ctx), ((parallel_generatedRoot_7.w_2_0 ctx), ((parallel_generatedRoot_7.w_6_0 ctx), ()))) (((parallel_generatedRoot_7.w_96_0 ctx) i), (((parallel_generatedRoot_7.w_96_1 ctx) i), ())))

theorem parallel_generatedRoot_7.facts_3 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} {ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0} {outputs : mxxMat2 × mxxMat2 × (Fin 1025 → mxxMat4) × (Fin 1025 → mxxMat4) × (Fin 1025 → mxxMat1) × (Fin 128 → mxxMat2) × (Fin 128 → mxxMat2) × (Fin 32 → mxxMat2) × (Fin 32 → mxxMat2) × (Fin 128 → mxxMat5) × (Fin 128 → mxxMat5) × (Fin 32 → mxxMat5) × (Fin 32 → mxxMat5) × (Fin 128 → mxxMat6) × (Fin 128 → mxxMat6) × (Fin 32 → mxxMat6) × (Fin 32 → mxxMat6) × (Fin 128 → mxxMat7) × (Fin 128 → mxxMat7) × (Fin 32 → mxxMat7) × (Fin 32 → mxxMat7) × (Fin 128 → mxxMat8) × (Fin 128 → mxxMat8) × (Fin 32 → mxxMat8) × (Fin 32 → mxxMat8) × (Fin 128 → mxxMat9) × (Fin 128 → mxxMat9) × (Fin 32 → mxxMat9) × (Fin 32 → mxxMat9) × (Fin 128 → mxxMat10) × (Fin 128 → mxxMat10) × (Fin 32 → mxxMat10) × (Fin 32 → mxxMat10) × (Fin 128 → mxxMat11) × (Fin 128 → mxxMat11) × (Fin 32 → mxxMat11) × (Fin 32 → mxxMat11) × (Fin 128 → mxxMat12) × (Fin 128 → mxxMat12) × (Fin 32 → mxxMat12) × (Fin 32 → mxxMat12) × (Fin 128 → mxxMat13) × (Fin 128 → mxxMat13) × (Fin 32 → mxxMat13) × (Fin 32 → mxxMat13) × (Fin 128 → mxxMat14) × (Fin 128 → mxxMat14) × (Fin 32 → mxxMat14) × (Fin 32 → mxxMat14) × (Fin 128 → mxxMat15) × (Fin 128 → mxxMat15) × (Fin 32 → mxxMat15) × (Fin 32 → mxxMat15) × (Fin 128 → mxxMat16) × (Fin 128 → mxxMat16) × (Fin 32 → mxxMat16) × (Fin 32 → mxxMat16) × (Fin 128 → mxxMat17) × (Fin 128 → mxxMat17) × (Fin 32 → mxxMat17) × (Fin 32 → mxxMat17) × (Fin 128 → mxxMat18) × (Fin 128 → mxxMat18) × (Fin 32 → mxxMat18) × (Fin 32 → mxxMat18) × (Fin 128 → mxxMat19) × (Fin 128 → mxxMat19) × (Fin 32 → mxxMat19) × (Fin 32 → mxxMat19) × (Fin 128 → mxxMat20) × (Fin 128 → mxxMat20) × (Fin 32 → mxxMat20) × (Fin 32 → mxxMat20) × (Fin 128 → mxxMat21) × (Fin 128 → mxxMat21) × (Fin 32 → mxxMat21) × (Fin 32 → mxxMat21) × (Fin 128 → mxxMat22) × (Fin 128 → mxxMat22) × (Fin 32 → mxxMat22) × (Fin 32 → mxxMat22) × (Fin 128 → mxxMat23) × (Fin 128 → mxxMat23) × (Fin 32 → mxxMat23) × (Fin 32 → mxxMat23) × (Fin 128 → mxxMat24) × (Fin 128 → mxxMat24) × (Fin 32 → mxxMat24) × (Fin 32 → mxxMat24) × (Fin 128 → mxxMat25) × (Fin 128 → mxxMat25) × (Fin 32 → mxxMat25) × (Fin 32 → mxxMat25) × (Fin 128 → mxxMat26) × (Fin 128 → mxxMat26) × (Fin 32 → mxxMat26) × (Fin 32 → mxxMat26) × (Fin 128 → mxxMat27) × (Fin 128 → mxxMat27) × (Fin 32 → mxxMat27) × (Fin 32 → mxxMat27) × (Fin 128 → mxxMat28) × (Fin 128 → mxxMat28) × (Fin 32 → mxxMat28) × (Fin 32 → mxxMat28) × (Fin 128 → mxxMat29) × (Fin 128 → mxxMat29) × (Fin 32 → mxxMat29) × (Fin 32 → mxxMat29) × (Fin 128 → mxxMat30) × (Fin 128 → mxxMat30) × (Fin 32 → mxxMat30) × (Fin 32 → mxxMat30) × (Fin 128 → mxxMat31) × (Fin 128 → mxxMat31) × (Fin 32 → mxxMat31) × (Fin 32 → mxxMat31) × (Fin 128 → mxxMat32) × (Fin 128 → mxxMat32) × (Fin 32 → mxxMat32) × (Fin 32 → mxxMat32) × (Fin 128 → mxxMat33) × (Fin 128 → mxxMat33) × (Fin 32 → mxxMat33) × (Fin 32 → mxxMat33) × (Fin 128 → mxxMat34) × (Fin 128 → mxxMat34) × (Fin 32 → mxxMat34) × (Fin 32 → mxxMat34) × (Fin 128 → mxxMat35) × (Fin 128 → mxxMat35) × (Fin 32 → mxxMat35) × (Fin 32 → mxxMat35) × (Fin 128 → mxxMat36) × (Fin 128 → mxxMat36) × (Fin 32 → mxxMat36) × (Fin 32 → mxxMat36) × (Fin 128 → mxxMat37) × (Fin 128 → mxxMat37) × (Fin 32 → mxxMat37) × (Fin 32 → mxxMat37) × (Fin 128 → mxxMat38) × (Fin 128 → mxxMat38) × (Fin 32 → mxxMat38) × (Fin 32 → mxxMat38) × (Fin 128 → mxxMat39) × (Fin 128 → mxxMat39) × (Fin 32 → mxxMat39) × (Fin 32 → mxxMat39) × Unit}
    (h : parallel_generatedRoot_7.constraints_3 backend hashModel tape path params i_0 (parallel_generatedRoot_7.w_2_0 ctx) (parallel_generatedRoot_7.w_1_0 ctx) (parallel_generatedRoot_7.w_6_0 ctx) (parallel_generatedRoot_7.w_81_0 ctx) (parallel_generatedRoot_7.w_81_1 ctx) (parallel_generatedRoot_7.w_82_0 ctx) (parallel_generatedRoot_7.w_82_1 ctx) (parallel_generatedRoot_7.w_83_0 ctx) (parallel_generatedRoot_7.w_83_1 ctx) (parallel_generatedRoot_7.w_84_0 ctx) (parallel_generatedRoot_7.w_84_1 ctx) (parallel_generatedRoot_7.w_85_0 ctx) (parallel_generatedRoot_7.w_85_1 ctx) (parallel_generatedRoot_7.w_86_0 ctx) (parallel_generatedRoot_7.w_86_1 ctx) (parallel_generatedRoot_7.w_87_0 ctx) (parallel_generatedRoot_7.w_87_1 ctx) (parallel_generatedRoot_7.w_88_0 ctx) (parallel_generatedRoot_7.w_88_1 ctx) (parallel_generatedRoot_7.w_89_0 ctx) (parallel_generatedRoot_7.w_89_1 ctx) (parallel_generatedRoot_7.w_90_0 ctx) (parallel_generatedRoot_7.w_90_1 ctx) (parallel_generatedRoot_7.w_91_0 ctx) (parallel_generatedRoot_7.w_91_1 ctx) (parallel_generatedRoot_7.w_92_0 ctx) (parallel_generatedRoot_7.w_92_1 ctx) (parallel_generatedRoot_7.w_93_0 ctx) (parallel_generatedRoot_7.w_93_1 ctx) (parallel_generatedRoot_7.w_94_0 ctx) (parallel_generatedRoot_7.w_94_1 ctx) (parallel_generatedRoot_7.w_95_0 ctx) (parallel_generatedRoot_7.w_95_1 ctx) (parallel_generatedRoot_7.w_96_0 ctx) (parallel_generatedRoot_7.w_96_1 ctx)) :
    parallel_generatedRoot_7.Facts_3 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_7.Facts_4 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) (outputs : mxxMat2 × mxxMat2 × (Fin 1025 → mxxMat4) × (Fin 1025 → mxxMat4) × (Fin 1025 → mxxMat1) × (Fin 128 → mxxMat2) × (Fin 128 → mxxMat2) × (Fin 32 → mxxMat2) × (Fin 32 → mxxMat2) × (Fin 128 → mxxMat5) × (Fin 128 → mxxMat5) × (Fin 32 → mxxMat5) × (Fin 32 → mxxMat5) × (Fin 128 → mxxMat6) × (Fin 128 → mxxMat6) × (Fin 32 → mxxMat6) × (Fin 32 → mxxMat6) × (Fin 128 → mxxMat7) × (Fin 128 → mxxMat7) × (Fin 32 → mxxMat7) × (Fin 32 → mxxMat7) × (Fin 128 → mxxMat8) × (Fin 128 → mxxMat8) × (Fin 32 → mxxMat8) × (Fin 32 → mxxMat8) × (Fin 128 → mxxMat9) × (Fin 128 → mxxMat9) × (Fin 32 → mxxMat9) × (Fin 32 → mxxMat9) × (Fin 128 → mxxMat10) × (Fin 128 → mxxMat10) × (Fin 32 → mxxMat10) × (Fin 32 → mxxMat10) × (Fin 128 → mxxMat11) × (Fin 128 → mxxMat11) × (Fin 32 → mxxMat11) × (Fin 32 → mxxMat11) × (Fin 128 → mxxMat12) × (Fin 128 → mxxMat12) × (Fin 32 → mxxMat12) × (Fin 32 → mxxMat12) × (Fin 128 → mxxMat13) × (Fin 128 → mxxMat13) × (Fin 32 → mxxMat13) × (Fin 32 → mxxMat13) × (Fin 128 → mxxMat14) × (Fin 128 → mxxMat14) × (Fin 32 → mxxMat14) × (Fin 32 → mxxMat14) × (Fin 128 → mxxMat15) × (Fin 128 → mxxMat15) × (Fin 32 → mxxMat15) × (Fin 32 → mxxMat15) × (Fin 128 → mxxMat16) × (Fin 128 → mxxMat16) × (Fin 32 → mxxMat16) × (Fin 32 → mxxMat16) × (Fin 128 → mxxMat17) × (Fin 128 → mxxMat17) × (Fin 32 → mxxMat17) × (Fin 32 → mxxMat17) × (Fin 128 → mxxMat18) × (Fin 128 → mxxMat18) × (Fin 32 → mxxMat18) × (Fin 32 → mxxMat18) × (Fin 128 → mxxMat19) × (Fin 128 → mxxMat19) × (Fin 32 → mxxMat19) × (Fin 32 → mxxMat19) × (Fin 128 → mxxMat20) × (Fin 128 → mxxMat20) × (Fin 32 → mxxMat20) × (Fin 32 → mxxMat20) × (Fin 128 → mxxMat21) × (Fin 128 → mxxMat21) × (Fin 32 → mxxMat21) × (Fin 32 → mxxMat21) × (Fin 128 → mxxMat22) × (Fin 128 → mxxMat22) × (Fin 32 → mxxMat22) × (Fin 32 → mxxMat22) × (Fin 128 → mxxMat23) × (Fin 128 → mxxMat23) × (Fin 32 → mxxMat23) × (Fin 32 → mxxMat23) × (Fin 128 → mxxMat24) × (Fin 128 → mxxMat24) × (Fin 32 → mxxMat24) × (Fin 32 → mxxMat24) × (Fin 128 → mxxMat25) × (Fin 128 → mxxMat25) × (Fin 32 → mxxMat25) × (Fin 32 → mxxMat25) × (Fin 128 → mxxMat26) × (Fin 128 → mxxMat26) × (Fin 32 → mxxMat26) × (Fin 32 → mxxMat26) × (Fin 128 → mxxMat27) × (Fin 128 → mxxMat27) × (Fin 32 → mxxMat27) × (Fin 32 → mxxMat27) × (Fin 128 → mxxMat28) × (Fin 128 → mxxMat28) × (Fin 32 → mxxMat28) × (Fin 32 → mxxMat28) × (Fin 128 → mxxMat29) × (Fin 128 → mxxMat29) × (Fin 32 → mxxMat29) × (Fin 32 → mxxMat29) × (Fin 128 → mxxMat30) × (Fin 128 → mxxMat30) × (Fin 32 → mxxMat30) × (Fin 32 → mxxMat30) × (Fin 128 → mxxMat31) × (Fin 128 → mxxMat31) × (Fin 32 → mxxMat31) × (Fin 32 → mxxMat31) × (Fin 128 → mxxMat32) × (Fin 128 → mxxMat32) × (Fin 32 → mxxMat32) × (Fin 32 → mxxMat32) × (Fin 128 → mxxMat33) × (Fin 128 → mxxMat33) × (Fin 32 → mxxMat33) × (Fin 32 → mxxMat33) × (Fin 128 → mxxMat34) × (Fin 128 → mxxMat34) × (Fin 32 → mxxMat34) × (Fin 32 → mxxMat34) × (Fin 128 → mxxMat35) × (Fin 128 → mxxMat35) × (Fin 32 → mxxMat35) × (Fin 32 → mxxMat35) × (Fin 128 → mxxMat36) × (Fin 128 → mxxMat36) × (Fin 32 → mxxMat36) × (Fin 32 → mxxMat36) × (Fin 128 → mxxMat37) × (Fin 128 → mxxMat37) × (Fin 32 → mxxMat37) × (Fin 32 → mxxMat37) × (Fin 128 → mxxMat38) × (Fin 128 → mxxMat38) × (Fin 32 → mxxMat38) × (Fin 32 → mxxMat38) × (Fin 128 → mxxMat39) × (Fin 128 → mxxMat39) × (Fin 32 → mxxMat39) × (Fin 32 → mxxMat39) × Unit) : Prop where
  t128 : (32) = 32
  h97_1 : (∀ i : Fin 32, parallel_parallel_generatedRoot_7_97 backend hashModel tape (path ++ [97, i.val]) params i_0 i ((parallel_generatedRoot_7.w_1_0 ctx), ((parallel_generatedRoot_7.w_2_0 ctx), ((parallel_generatedRoot_7.w_6_0 ctx), ()))) (((parallel_generatedRoot_7.w_97_0 ctx) i), (((parallel_generatedRoot_7.w_97_1 ctx) i), ())))
  t130 : (128) = 128
  h98_1 : (∀ i : Fin 128, parallel_parallel_generatedRoot_7_98 backend hashModel tape (path ++ [98, i.val]) params i_0 i ((parallel_generatedRoot_7.w_1_0 ctx), ((parallel_generatedRoot_7.w_2_0 ctx), ((parallel_generatedRoot_7.w_6_0 ctx), ()))) (((parallel_generatedRoot_7.w_98_0 ctx) i), (((parallel_generatedRoot_7.w_98_1 ctx) i), ())))
  t132 : (32) = 32
  h99_1 : (∀ i : Fin 32, parallel_parallel_generatedRoot_7_99 backend hashModel tape (path ++ [99, i.val]) params i_0 i ((parallel_generatedRoot_7.w_1_0 ctx), ((parallel_generatedRoot_7.w_2_0 ctx), ((parallel_generatedRoot_7.w_6_0 ctx), ()))) (((parallel_generatedRoot_7.w_99_0 ctx) i), (((parallel_generatedRoot_7.w_99_1 ctx) i), ())))
  t134 : (128) = 128
  h100_1 : (∀ i : Fin 128, parallel_parallel_generatedRoot_7_100 backend hashModel tape (path ++ [100, i.val]) params i_0 i ((parallel_generatedRoot_7.w_1_0 ctx), ((parallel_generatedRoot_7.w_2_0 ctx), ((parallel_generatedRoot_7.w_6_0 ctx), ()))) (((parallel_generatedRoot_7.w_100_0 ctx) i), (((parallel_generatedRoot_7.w_100_1 ctx) i), ())))
  t136 : (32) = 32
  h101_1 : (∀ i : Fin 32, parallel_parallel_generatedRoot_7_101 backend hashModel tape (path ++ [101, i.val]) params i_0 i ((parallel_generatedRoot_7.w_1_0 ctx), ((parallel_generatedRoot_7.w_2_0 ctx), ((parallel_generatedRoot_7.w_6_0 ctx), ()))) (((parallel_generatedRoot_7.w_101_0 ctx) i), (((parallel_generatedRoot_7.w_101_1 ctx) i), ())))
  t138 : (128) = 128
  h102_1 : (∀ i : Fin 128, parallel_parallel_generatedRoot_7_102 backend hashModel tape (path ++ [102, i.val]) params i_0 i ((parallel_generatedRoot_7.w_1_0 ctx), ((parallel_generatedRoot_7.w_2_0 ctx), ((parallel_generatedRoot_7.w_6_0 ctx), ()))) (((parallel_generatedRoot_7.w_102_0 ctx) i), (((parallel_generatedRoot_7.w_102_1 ctx) i), ())))
  t140 : (32) = 32
  h103_1 : (∀ i : Fin 32, parallel_parallel_generatedRoot_7_103 backend hashModel tape (path ++ [103, i.val]) params i_0 i ((parallel_generatedRoot_7.w_1_0 ctx), ((parallel_generatedRoot_7.w_2_0 ctx), ((parallel_generatedRoot_7.w_6_0 ctx), ()))) (((parallel_generatedRoot_7.w_103_0 ctx) i), (((parallel_generatedRoot_7.w_103_1 ctx) i), ())))
  t142 : (128) = 128
  h104_1 : (∀ i : Fin 128, parallel_parallel_generatedRoot_7_104 backend hashModel tape (path ++ [104, i.val]) params i_0 i ((parallel_generatedRoot_7.w_1_0 ctx), ((parallel_generatedRoot_7.w_2_0 ctx), ((parallel_generatedRoot_7.w_6_0 ctx), ()))) (((parallel_generatedRoot_7.w_104_0 ctx) i), (((parallel_generatedRoot_7.w_104_1 ctx) i), ())))
  t144 : (32) = 32
  h105_1 : (∀ i : Fin 32, parallel_parallel_generatedRoot_7_105 backend hashModel tape (path ++ [105, i.val]) params i_0 i ((parallel_generatedRoot_7.w_1_0 ctx), ((parallel_generatedRoot_7.w_2_0 ctx), ((parallel_generatedRoot_7.w_6_0 ctx), ()))) (((parallel_generatedRoot_7.w_105_0 ctx) i), (((parallel_generatedRoot_7.w_105_1 ctx) i), ())))
  t146 : (128) = 128
  h106_1 : (∀ i : Fin 128, parallel_parallel_generatedRoot_7_106 backend hashModel tape (path ++ [106, i.val]) params i_0 i ((parallel_generatedRoot_7.w_1_0 ctx), ((parallel_generatedRoot_7.w_2_0 ctx), ((parallel_generatedRoot_7.w_6_0 ctx), ()))) (((parallel_generatedRoot_7.w_106_0 ctx) i), (((parallel_generatedRoot_7.w_106_1 ctx) i), ())))
  t148 : (32) = 32
  h107_1 : (∀ i : Fin 32, parallel_parallel_generatedRoot_7_107 backend hashModel tape (path ++ [107, i.val]) params i_0 i ((parallel_generatedRoot_7.w_1_0 ctx), ((parallel_generatedRoot_7.w_2_0 ctx), ((parallel_generatedRoot_7.w_6_0 ctx), ()))) (((parallel_generatedRoot_7.w_107_0 ctx) i), (((parallel_generatedRoot_7.w_107_1 ctx) i), ())))
  t150 : (128) = 128
  h108_1 : (∀ i : Fin 128, parallel_parallel_generatedRoot_7_108 backend hashModel tape (path ++ [108, i.val]) params i_0 i ((parallel_generatedRoot_7.w_1_0 ctx), ((parallel_generatedRoot_7.w_2_0 ctx), ((parallel_generatedRoot_7.w_6_0 ctx), ()))) (((parallel_generatedRoot_7.w_108_0 ctx) i), (((parallel_generatedRoot_7.w_108_1 ctx) i), ())))
  t152 : (32) = 32
  h109_1 : (∀ i : Fin 32, parallel_parallel_generatedRoot_7_109 backend hashModel tape (path ++ [109, i.val]) params i_0 i ((parallel_generatedRoot_7.w_1_0 ctx), ((parallel_generatedRoot_7.w_2_0 ctx), ((parallel_generatedRoot_7.w_6_0 ctx), ()))) (((parallel_generatedRoot_7.w_109_0 ctx) i), (((parallel_generatedRoot_7.w_109_1 ctx) i), ())))
  t154 : (128) = 128
  h110_1 : (∀ i : Fin 128, parallel_parallel_generatedRoot_7_110 backend hashModel tape (path ++ [110, i.val]) params i_0 i ((parallel_generatedRoot_7.w_1_0 ctx), ((parallel_generatedRoot_7.w_2_0 ctx), ((parallel_generatedRoot_7.w_6_0 ctx), ()))) (((parallel_generatedRoot_7.w_110_0 ctx) i), (((parallel_generatedRoot_7.w_110_1 ctx) i), ())))
  t156 : (32) = 32
  h111_1 : (∀ i : Fin 32, parallel_parallel_generatedRoot_7_111 backend hashModel tape (path ++ [111, i.val]) params i_0 i ((parallel_generatedRoot_7.w_1_0 ctx), ((parallel_generatedRoot_7.w_2_0 ctx), ((parallel_generatedRoot_7.w_6_0 ctx), ()))) (((parallel_generatedRoot_7.w_111_0 ctx) i), (((parallel_generatedRoot_7.w_111_1 ctx) i), ())))
  hout : outputs = parallel_generatedRoot_7.outs ctx

theorem parallel_generatedRoot_7.facts_4 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} {ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0} {outputs : mxxMat2 × mxxMat2 × (Fin 1025 → mxxMat4) × (Fin 1025 → mxxMat4) × (Fin 1025 → mxxMat1) × (Fin 128 → mxxMat2) × (Fin 128 → mxxMat2) × (Fin 32 → mxxMat2) × (Fin 32 → mxxMat2) × (Fin 128 → mxxMat5) × (Fin 128 → mxxMat5) × (Fin 32 → mxxMat5) × (Fin 32 → mxxMat5) × (Fin 128 → mxxMat6) × (Fin 128 → mxxMat6) × (Fin 32 → mxxMat6) × (Fin 32 → mxxMat6) × (Fin 128 → mxxMat7) × (Fin 128 → mxxMat7) × (Fin 32 → mxxMat7) × (Fin 32 → mxxMat7) × (Fin 128 → mxxMat8) × (Fin 128 → mxxMat8) × (Fin 32 → mxxMat8) × (Fin 32 → mxxMat8) × (Fin 128 → mxxMat9) × (Fin 128 → mxxMat9) × (Fin 32 → mxxMat9) × (Fin 32 → mxxMat9) × (Fin 128 → mxxMat10) × (Fin 128 → mxxMat10) × (Fin 32 → mxxMat10) × (Fin 32 → mxxMat10) × (Fin 128 → mxxMat11) × (Fin 128 → mxxMat11) × (Fin 32 → mxxMat11) × (Fin 32 → mxxMat11) × (Fin 128 → mxxMat12) × (Fin 128 → mxxMat12) × (Fin 32 → mxxMat12) × (Fin 32 → mxxMat12) × (Fin 128 → mxxMat13) × (Fin 128 → mxxMat13) × (Fin 32 → mxxMat13) × (Fin 32 → mxxMat13) × (Fin 128 → mxxMat14) × (Fin 128 → mxxMat14) × (Fin 32 → mxxMat14) × (Fin 32 → mxxMat14) × (Fin 128 → mxxMat15) × (Fin 128 → mxxMat15) × (Fin 32 → mxxMat15) × (Fin 32 → mxxMat15) × (Fin 128 → mxxMat16) × (Fin 128 → mxxMat16) × (Fin 32 → mxxMat16) × (Fin 32 → mxxMat16) × (Fin 128 → mxxMat17) × (Fin 128 → mxxMat17) × (Fin 32 → mxxMat17) × (Fin 32 → mxxMat17) × (Fin 128 → mxxMat18) × (Fin 128 → mxxMat18) × (Fin 32 → mxxMat18) × (Fin 32 → mxxMat18) × (Fin 128 → mxxMat19) × (Fin 128 → mxxMat19) × (Fin 32 → mxxMat19) × (Fin 32 → mxxMat19) × (Fin 128 → mxxMat20) × (Fin 128 → mxxMat20) × (Fin 32 → mxxMat20) × (Fin 32 → mxxMat20) × (Fin 128 → mxxMat21) × (Fin 128 → mxxMat21) × (Fin 32 → mxxMat21) × (Fin 32 → mxxMat21) × (Fin 128 → mxxMat22) × (Fin 128 → mxxMat22) × (Fin 32 → mxxMat22) × (Fin 32 → mxxMat22) × (Fin 128 → mxxMat23) × (Fin 128 → mxxMat23) × (Fin 32 → mxxMat23) × (Fin 32 → mxxMat23) × (Fin 128 → mxxMat24) × (Fin 128 → mxxMat24) × (Fin 32 → mxxMat24) × (Fin 32 → mxxMat24) × (Fin 128 → mxxMat25) × (Fin 128 → mxxMat25) × (Fin 32 → mxxMat25) × (Fin 32 → mxxMat25) × (Fin 128 → mxxMat26) × (Fin 128 → mxxMat26) × (Fin 32 → mxxMat26) × (Fin 32 → mxxMat26) × (Fin 128 → mxxMat27) × (Fin 128 → mxxMat27) × (Fin 32 → mxxMat27) × (Fin 32 → mxxMat27) × (Fin 128 → mxxMat28) × (Fin 128 → mxxMat28) × (Fin 32 → mxxMat28) × (Fin 32 → mxxMat28) × (Fin 128 → mxxMat29) × (Fin 128 → mxxMat29) × (Fin 32 → mxxMat29) × (Fin 32 → mxxMat29) × (Fin 128 → mxxMat30) × (Fin 128 → mxxMat30) × (Fin 32 → mxxMat30) × (Fin 32 → mxxMat30) × (Fin 128 → mxxMat31) × (Fin 128 → mxxMat31) × (Fin 32 → mxxMat31) × (Fin 32 → mxxMat31) × (Fin 128 → mxxMat32) × (Fin 128 → mxxMat32) × (Fin 32 → mxxMat32) × (Fin 32 → mxxMat32) × (Fin 128 → mxxMat33) × (Fin 128 → mxxMat33) × (Fin 32 → mxxMat33) × (Fin 32 → mxxMat33) × (Fin 128 → mxxMat34) × (Fin 128 → mxxMat34) × (Fin 32 → mxxMat34) × (Fin 32 → mxxMat34) × (Fin 128 → mxxMat35) × (Fin 128 → mxxMat35) × (Fin 32 → mxxMat35) × (Fin 32 → mxxMat35) × (Fin 128 → mxxMat36) × (Fin 128 → mxxMat36) × (Fin 32 → mxxMat36) × (Fin 32 → mxxMat36) × (Fin 128 → mxxMat37) × (Fin 128 → mxxMat37) × (Fin 32 → mxxMat37) × (Fin 32 → mxxMat37) × (Fin 128 → mxxMat38) × (Fin 128 → mxxMat38) × (Fin 32 → mxxMat38) × (Fin 32 → mxxMat38) × (Fin 128 → mxxMat39) × (Fin 128 → mxxMat39) × (Fin 32 → mxxMat39) × (Fin 32 → mxxMat39) × Unit}
    (h : parallel_generatedRoot_7.constraints_4 backend hashModel tape path params i_0 (parallel_generatedRoot_7.w_2_0 ctx) (parallel_generatedRoot_7.w_1_0 ctx) (parallel_generatedRoot_7.w_6_0 ctx) (parallel_generatedRoot_7.w_24_0 ctx) (parallel_generatedRoot_7.w_36_0 ctx) (parallel_generatedRoot_7.w_39_0 ctx) (parallel_generatedRoot_7.w_39_1 ctx) (parallel_generatedRoot_7.w_39_2 ctx) (parallel_generatedRoot_7.w_40_0 ctx) (parallel_generatedRoot_7.w_40_1 ctx) (parallel_generatedRoot_7.w_41_0 ctx) (parallel_generatedRoot_7.w_41_1 ctx) (parallel_generatedRoot_7.w_42_0 ctx) (parallel_generatedRoot_7.w_42_1 ctx) (parallel_generatedRoot_7.w_43_0 ctx) (parallel_generatedRoot_7.w_43_1 ctx) (parallel_generatedRoot_7.w_44_0 ctx) (parallel_generatedRoot_7.w_44_1 ctx) (parallel_generatedRoot_7.w_45_0 ctx) (parallel_generatedRoot_7.w_45_1 ctx) (parallel_generatedRoot_7.w_46_0 ctx) (parallel_generatedRoot_7.w_46_1 ctx) (parallel_generatedRoot_7.w_47_0 ctx) (parallel_generatedRoot_7.w_47_1 ctx) (parallel_generatedRoot_7.w_48_0 ctx) (parallel_generatedRoot_7.w_48_1 ctx) (parallel_generatedRoot_7.w_49_0 ctx) (parallel_generatedRoot_7.w_49_1 ctx) (parallel_generatedRoot_7.w_50_0 ctx) (parallel_generatedRoot_7.w_50_1 ctx) (parallel_generatedRoot_7.w_51_0 ctx) (parallel_generatedRoot_7.w_51_1 ctx) (parallel_generatedRoot_7.w_52_0 ctx) (parallel_generatedRoot_7.w_52_1 ctx) (parallel_generatedRoot_7.w_53_0 ctx) (parallel_generatedRoot_7.w_53_1 ctx) (parallel_generatedRoot_7.w_54_0 ctx) (parallel_generatedRoot_7.w_54_1 ctx) (parallel_generatedRoot_7.w_55_0 ctx) (parallel_generatedRoot_7.w_55_1 ctx) (parallel_generatedRoot_7.w_56_0 ctx) (parallel_generatedRoot_7.w_56_1 ctx) (parallel_generatedRoot_7.w_57_0 ctx) (parallel_generatedRoot_7.w_57_1 ctx) (parallel_generatedRoot_7.w_58_0 ctx) (parallel_generatedRoot_7.w_58_1 ctx) (parallel_generatedRoot_7.w_59_0 ctx) (parallel_generatedRoot_7.w_59_1 ctx) (parallel_generatedRoot_7.w_60_0 ctx) (parallel_generatedRoot_7.w_60_1 ctx) (parallel_generatedRoot_7.w_61_0 ctx) (parallel_generatedRoot_7.w_61_1 ctx) (parallel_generatedRoot_7.w_62_0 ctx) (parallel_generatedRoot_7.w_62_1 ctx) (parallel_generatedRoot_7.w_63_0 ctx) (parallel_generatedRoot_7.w_63_1 ctx) (parallel_generatedRoot_7.w_64_0 ctx) (parallel_generatedRoot_7.w_64_1 ctx) (parallel_generatedRoot_7.w_65_0 ctx) (parallel_generatedRoot_7.w_65_1 ctx) (parallel_generatedRoot_7.w_66_0 ctx) (parallel_generatedRoot_7.w_66_1 ctx) (parallel_generatedRoot_7.w_67_0 ctx) (parallel_generatedRoot_7.w_67_1 ctx) (parallel_generatedRoot_7.w_68_0 ctx) (parallel_generatedRoot_7.w_68_1 ctx) (parallel_generatedRoot_7.w_69_0 ctx) (parallel_generatedRoot_7.w_69_1 ctx) (parallel_generatedRoot_7.w_70_0 ctx) (parallel_generatedRoot_7.w_70_1 ctx) (parallel_generatedRoot_7.w_71_0 ctx) (parallel_generatedRoot_7.w_71_1 ctx) (parallel_generatedRoot_7.w_72_0 ctx) (parallel_generatedRoot_7.w_72_1 ctx) (parallel_generatedRoot_7.w_73_0 ctx) (parallel_generatedRoot_7.w_73_1 ctx) (parallel_generatedRoot_7.w_74_0 ctx) (parallel_generatedRoot_7.w_74_1 ctx) (parallel_generatedRoot_7.w_75_0 ctx) (parallel_generatedRoot_7.w_75_1 ctx) (parallel_generatedRoot_7.w_76_0 ctx) (parallel_generatedRoot_7.w_76_1 ctx) (parallel_generatedRoot_7.w_77_0 ctx) (parallel_generatedRoot_7.w_77_1 ctx) (parallel_generatedRoot_7.w_78_0 ctx) (parallel_generatedRoot_7.w_78_1 ctx) (parallel_generatedRoot_7.w_79_0 ctx) (parallel_generatedRoot_7.w_79_1 ctx) (parallel_generatedRoot_7.w_80_0 ctx) (parallel_generatedRoot_7.w_80_1 ctx) (parallel_generatedRoot_7.w_81_0 ctx) (parallel_generatedRoot_7.w_81_1 ctx) (parallel_generatedRoot_7.w_82_0 ctx) (parallel_generatedRoot_7.w_82_1 ctx) (parallel_generatedRoot_7.w_83_0 ctx) (parallel_generatedRoot_7.w_83_1 ctx) (parallel_generatedRoot_7.w_84_0 ctx) (parallel_generatedRoot_7.w_84_1 ctx) (parallel_generatedRoot_7.w_85_0 ctx) (parallel_generatedRoot_7.w_85_1 ctx) (parallel_generatedRoot_7.w_86_0 ctx) (parallel_generatedRoot_7.w_86_1 ctx) (parallel_generatedRoot_7.w_87_0 ctx) (parallel_generatedRoot_7.w_87_1 ctx) (parallel_generatedRoot_7.w_88_0 ctx) (parallel_generatedRoot_7.w_88_1 ctx) (parallel_generatedRoot_7.w_89_0 ctx) (parallel_generatedRoot_7.w_89_1 ctx) (parallel_generatedRoot_7.w_90_0 ctx) (parallel_generatedRoot_7.w_90_1 ctx) (parallel_generatedRoot_7.w_91_0 ctx) (parallel_generatedRoot_7.w_91_1 ctx) (parallel_generatedRoot_7.w_92_0 ctx) (parallel_generatedRoot_7.w_92_1 ctx) (parallel_generatedRoot_7.w_93_0 ctx) (parallel_generatedRoot_7.w_93_1 ctx) (parallel_generatedRoot_7.w_94_0 ctx) (parallel_generatedRoot_7.w_94_1 ctx) (parallel_generatedRoot_7.w_95_0 ctx) (parallel_generatedRoot_7.w_95_1 ctx) (parallel_generatedRoot_7.w_96_0 ctx) (parallel_generatedRoot_7.w_96_1 ctx) (parallel_generatedRoot_7.w_97_0 ctx) (parallel_generatedRoot_7.w_97_1 ctx) (parallel_generatedRoot_7.w_98_0 ctx) (parallel_generatedRoot_7.w_98_1 ctx) (parallel_generatedRoot_7.w_99_0 ctx) (parallel_generatedRoot_7.w_99_1 ctx) (parallel_generatedRoot_7.w_100_0 ctx) (parallel_generatedRoot_7.w_100_1 ctx) (parallel_generatedRoot_7.w_101_0 ctx) (parallel_generatedRoot_7.w_101_1 ctx) (parallel_generatedRoot_7.w_102_0 ctx) (parallel_generatedRoot_7.w_102_1 ctx) (parallel_generatedRoot_7.w_103_0 ctx) (parallel_generatedRoot_7.w_103_1 ctx) (parallel_generatedRoot_7.w_104_0 ctx) (parallel_generatedRoot_7.w_104_1 ctx) (parallel_generatedRoot_7.w_105_0 ctx) (parallel_generatedRoot_7.w_105_1 ctx) (parallel_generatedRoot_7.w_106_0 ctx) (parallel_generatedRoot_7.w_106_1 ctx) (parallel_generatedRoot_7.w_107_0 ctx) (parallel_generatedRoot_7.w_107_1 ctx) (parallel_generatedRoot_7.w_108_0 ctx) (parallel_generatedRoot_7.w_108_1 ctx) (parallel_generatedRoot_7.w_109_0 ctx) (parallel_generatedRoot_7.w_109_1 ctx) (parallel_generatedRoot_7.w_110_0 ctx) (parallel_generatedRoot_7.w_110_1 ctx) (parallel_generatedRoot_7.w_111_0 ctx) (parallel_generatedRoot_7.w_111_1 ctx) outputs) :
    parallel_generatedRoot_7.Facts_4 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_7.Facts {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) (outputs : mxxMat2 × mxxMat2 × (Fin 1025 → mxxMat4) × (Fin 1025 → mxxMat4) × (Fin 1025 → mxxMat1) × (Fin 128 → mxxMat2) × (Fin 128 → mxxMat2) × (Fin 32 → mxxMat2) × (Fin 32 → mxxMat2) × (Fin 128 → mxxMat5) × (Fin 128 → mxxMat5) × (Fin 32 → mxxMat5) × (Fin 32 → mxxMat5) × (Fin 128 → mxxMat6) × (Fin 128 → mxxMat6) × (Fin 32 → mxxMat6) × (Fin 32 → mxxMat6) × (Fin 128 → mxxMat7) × (Fin 128 → mxxMat7) × (Fin 32 → mxxMat7) × (Fin 32 → mxxMat7) × (Fin 128 → mxxMat8) × (Fin 128 → mxxMat8) × (Fin 32 → mxxMat8) × (Fin 32 → mxxMat8) × (Fin 128 → mxxMat9) × (Fin 128 → mxxMat9) × (Fin 32 → mxxMat9) × (Fin 32 → mxxMat9) × (Fin 128 → mxxMat10) × (Fin 128 → mxxMat10) × (Fin 32 → mxxMat10) × (Fin 32 → mxxMat10) × (Fin 128 → mxxMat11) × (Fin 128 → mxxMat11) × (Fin 32 → mxxMat11) × (Fin 32 → mxxMat11) × (Fin 128 → mxxMat12) × (Fin 128 → mxxMat12) × (Fin 32 → mxxMat12) × (Fin 32 → mxxMat12) × (Fin 128 → mxxMat13) × (Fin 128 → mxxMat13) × (Fin 32 → mxxMat13) × (Fin 32 → mxxMat13) × (Fin 128 → mxxMat14) × (Fin 128 → mxxMat14) × (Fin 32 → mxxMat14) × (Fin 32 → mxxMat14) × (Fin 128 → mxxMat15) × (Fin 128 → mxxMat15) × (Fin 32 → mxxMat15) × (Fin 32 → mxxMat15) × (Fin 128 → mxxMat16) × (Fin 128 → mxxMat16) × (Fin 32 → mxxMat16) × (Fin 32 → mxxMat16) × (Fin 128 → mxxMat17) × (Fin 128 → mxxMat17) × (Fin 32 → mxxMat17) × (Fin 32 → mxxMat17) × (Fin 128 → mxxMat18) × (Fin 128 → mxxMat18) × (Fin 32 → mxxMat18) × (Fin 32 → mxxMat18) × (Fin 128 → mxxMat19) × (Fin 128 → mxxMat19) × (Fin 32 → mxxMat19) × (Fin 32 → mxxMat19) × (Fin 128 → mxxMat20) × (Fin 128 → mxxMat20) × (Fin 32 → mxxMat20) × (Fin 32 → mxxMat20) × (Fin 128 → mxxMat21) × (Fin 128 → mxxMat21) × (Fin 32 → mxxMat21) × (Fin 32 → mxxMat21) × (Fin 128 → mxxMat22) × (Fin 128 → mxxMat22) × (Fin 32 → mxxMat22) × (Fin 32 → mxxMat22) × (Fin 128 → mxxMat23) × (Fin 128 → mxxMat23) × (Fin 32 → mxxMat23) × (Fin 32 → mxxMat23) × (Fin 128 → mxxMat24) × (Fin 128 → mxxMat24) × (Fin 32 → mxxMat24) × (Fin 32 → mxxMat24) × (Fin 128 → mxxMat25) × (Fin 128 → mxxMat25) × (Fin 32 → mxxMat25) × (Fin 32 → mxxMat25) × (Fin 128 → mxxMat26) × (Fin 128 → mxxMat26) × (Fin 32 → mxxMat26) × (Fin 32 → mxxMat26) × (Fin 128 → mxxMat27) × (Fin 128 → mxxMat27) × (Fin 32 → mxxMat27) × (Fin 32 → mxxMat27) × (Fin 128 → mxxMat28) × (Fin 128 → mxxMat28) × (Fin 32 → mxxMat28) × (Fin 32 → mxxMat28) × (Fin 128 → mxxMat29) × (Fin 128 → mxxMat29) × (Fin 32 → mxxMat29) × (Fin 32 → mxxMat29) × (Fin 128 → mxxMat30) × (Fin 128 → mxxMat30) × (Fin 32 → mxxMat30) × (Fin 32 → mxxMat30) × (Fin 128 → mxxMat31) × (Fin 128 → mxxMat31) × (Fin 32 → mxxMat31) × (Fin 32 → mxxMat31) × (Fin 128 → mxxMat32) × (Fin 128 → mxxMat32) × (Fin 32 → mxxMat32) × (Fin 32 → mxxMat32) × (Fin 128 → mxxMat33) × (Fin 128 → mxxMat33) × (Fin 32 → mxxMat33) × (Fin 32 → mxxMat33) × (Fin 128 → mxxMat34) × (Fin 128 → mxxMat34) × (Fin 32 → mxxMat34) × (Fin 32 → mxxMat34) × (Fin 128 → mxxMat35) × (Fin 128 → mxxMat35) × (Fin 32 → mxxMat35) × (Fin 32 → mxxMat35) × (Fin 128 → mxxMat36) × (Fin 128 → mxxMat36) × (Fin 32 → mxxMat36) × (Fin 32 → mxxMat36) × (Fin 128 → mxxMat37) × (Fin 128 → mxxMat37) × (Fin 32 → mxxMat37) × (Fin 32 → mxxMat37) × (Fin 128 → mxxMat38) × (Fin 128 → mxxMat38) × (Fin 32 → mxxMat38) × (Fin 32 → mxxMat38) × (Fin 128 → mxxMat39) × (Fin 128 → mxxMat39) × (Fin 32 → mxxMat39) × (Fin 32 → mxxMat39) × Unit) : Prop where
  c0 : parallel_generatedRoot_7.Facts_0 ctx
  c1 : parallel_generatedRoot_7.Facts_1 ctx
  c2 : parallel_generatedRoot_7.Facts_2 ctx
  c3 : parallel_generatedRoot_7.Facts_3 ctx
  c4 : parallel_generatedRoot_7.Facts_4 ctx outputs

theorem parallel_generatedRoot_7.facts {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel tape path params i_0) {outputs : mxxMat2 × mxxMat2 × (Fin 1025 → mxxMat4) × (Fin 1025 → mxxMat4) × (Fin 1025 → mxxMat1) × (Fin 128 → mxxMat2) × (Fin 128 → mxxMat2) × (Fin 32 → mxxMat2) × (Fin 32 → mxxMat2) × (Fin 128 → mxxMat5) × (Fin 128 → mxxMat5) × (Fin 32 → mxxMat5) × (Fin 32 → mxxMat5) × (Fin 128 → mxxMat6) × (Fin 128 → mxxMat6) × (Fin 32 → mxxMat6) × (Fin 32 → mxxMat6) × (Fin 128 → mxxMat7) × (Fin 128 → mxxMat7) × (Fin 32 → mxxMat7) × (Fin 32 → mxxMat7) × (Fin 128 → mxxMat8) × (Fin 128 → mxxMat8) × (Fin 32 → mxxMat8) × (Fin 32 → mxxMat8) × (Fin 128 → mxxMat9) × (Fin 128 → mxxMat9) × (Fin 32 → mxxMat9) × (Fin 32 → mxxMat9) × (Fin 128 → mxxMat10) × (Fin 128 → mxxMat10) × (Fin 32 → mxxMat10) × (Fin 32 → mxxMat10) × (Fin 128 → mxxMat11) × (Fin 128 → mxxMat11) × (Fin 32 → mxxMat11) × (Fin 32 → mxxMat11) × (Fin 128 → mxxMat12) × (Fin 128 → mxxMat12) × (Fin 32 → mxxMat12) × (Fin 32 → mxxMat12) × (Fin 128 → mxxMat13) × (Fin 128 → mxxMat13) × (Fin 32 → mxxMat13) × (Fin 32 → mxxMat13) × (Fin 128 → mxxMat14) × (Fin 128 → mxxMat14) × (Fin 32 → mxxMat14) × (Fin 32 → mxxMat14) × (Fin 128 → mxxMat15) × (Fin 128 → mxxMat15) × (Fin 32 → mxxMat15) × (Fin 32 → mxxMat15) × (Fin 128 → mxxMat16) × (Fin 128 → mxxMat16) × (Fin 32 → mxxMat16) × (Fin 32 → mxxMat16) × (Fin 128 → mxxMat17) × (Fin 128 → mxxMat17) × (Fin 32 → mxxMat17) × (Fin 32 → mxxMat17) × (Fin 128 → mxxMat18) × (Fin 128 → mxxMat18) × (Fin 32 → mxxMat18) × (Fin 32 → mxxMat18) × (Fin 128 → mxxMat19) × (Fin 128 → mxxMat19) × (Fin 32 → mxxMat19) × (Fin 32 → mxxMat19) × (Fin 128 → mxxMat20) × (Fin 128 → mxxMat20) × (Fin 32 → mxxMat20) × (Fin 32 → mxxMat20) × (Fin 128 → mxxMat21) × (Fin 128 → mxxMat21) × (Fin 32 → mxxMat21) × (Fin 32 → mxxMat21) × (Fin 128 → mxxMat22) × (Fin 128 → mxxMat22) × (Fin 32 → mxxMat22) × (Fin 32 → mxxMat22) × (Fin 128 → mxxMat23) × (Fin 128 → mxxMat23) × (Fin 32 → mxxMat23) × (Fin 32 → mxxMat23) × (Fin 128 → mxxMat24) × (Fin 128 → mxxMat24) × (Fin 32 → mxxMat24) × (Fin 32 → mxxMat24) × (Fin 128 → mxxMat25) × (Fin 128 → mxxMat25) × (Fin 32 → mxxMat25) × (Fin 32 → mxxMat25) × (Fin 128 → mxxMat26) × (Fin 128 → mxxMat26) × (Fin 32 → mxxMat26) × (Fin 32 → mxxMat26) × (Fin 128 → mxxMat27) × (Fin 128 → mxxMat27) × (Fin 32 → mxxMat27) × (Fin 32 → mxxMat27) × (Fin 128 → mxxMat28) × (Fin 128 → mxxMat28) × (Fin 32 → mxxMat28) × (Fin 32 → mxxMat28) × (Fin 128 → mxxMat29) × (Fin 128 → mxxMat29) × (Fin 32 → mxxMat29) × (Fin 32 → mxxMat29) × (Fin 128 → mxxMat30) × (Fin 128 → mxxMat30) × (Fin 32 → mxxMat30) × (Fin 32 → mxxMat30) × (Fin 128 → mxxMat31) × (Fin 128 → mxxMat31) × (Fin 32 → mxxMat31) × (Fin 32 → mxxMat31) × (Fin 128 → mxxMat32) × (Fin 128 → mxxMat32) × (Fin 32 → mxxMat32) × (Fin 32 → mxxMat32) × (Fin 128 → mxxMat33) × (Fin 128 → mxxMat33) × (Fin 32 → mxxMat33) × (Fin 32 → mxxMat33) × (Fin 128 → mxxMat34) × (Fin 128 → mxxMat34) × (Fin 32 → mxxMat34) × (Fin 32 → mxxMat34) × (Fin 128 → mxxMat35) × (Fin 128 → mxxMat35) × (Fin 32 → mxxMat35) × (Fin 32 → mxxMat35) × (Fin 128 → mxxMat36) × (Fin 128 → mxxMat36) × (Fin 32 → mxxMat36) × (Fin 32 → mxxMat36) × (Fin 128 → mxxMat37) × (Fin 128 → mxxMat37) × (Fin 32 → mxxMat37) × (Fin 32 → mxxMat37) × (Fin 128 → mxxMat38) × (Fin 128 → mxxMat38) × (Fin 32 → mxxMat38) × (Fin 32 → mxxMat38) × (Fin 128 → mxxMat39) × (Fin 128 → mxxMat39) × (Fin 32 → mxxMat39) × (Fin 32 → mxxMat39) × Unit}
    (h : parallel_generatedRoot_7.body backend hashModel tape path params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_7.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_7.facts_0 (ctx := ctx) (outputs := outputs) t0.1
  have t1 := t0.2
  have r1 := parallel_generatedRoot_7.facts_1 (ctx := ctx) (outputs := outputs) t1.1
  have t2 := t1.2
  have r2 := parallel_generatedRoot_7.facts_2 (ctx := ctx) (outputs := outputs) t2.1
  have t3 := t2.2
  have r3 := parallel_generatedRoot_7.facts_3 (ctx := ctx) (outputs := outputs) t3.1
  have t4 := t3.2
  have r4 := parallel_generatedRoot_7.facts_4 (ctx := ctx) (outputs := outputs) t4
  ⟨r0, r1, r2, r3, r4⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) where
  inputs : mxxMat0 × (Fin 8 → ByteArray) × mxxMat0 × (Fin 8 → Int) × (Fin 8 → Int) × (Fin 8 → Fin 1025 → Int) × (Fin 8 → Fin 1025 → Int) × Unit
  witness : generatedRoot.Witness

def generatedRoot.w_0_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.inputs.1

def generatedRoot.w_1_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → ByteArray :=
  ctx.inputs.2.1

def generatedRoot.w_2_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.inputs.2.2.1

def generatedRoot.w_3_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Int :=
  ctx.inputs.2.2.2.1

def generatedRoot.w_4_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Int :=
  ctx.inputs.2.2.2.2.1

def generatedRoot.w_5_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 1025 → Int :=
  ctx.inputs.2.2.2.2.2.1

def generatedRoot.w_6_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 1025 → Int :=
  ctx.inputs.2.2.2.2.2.2.1

def generatedRoot.w_7_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → mxxMat2 :=
  ctx.witness.w_7_0

def generatedRoot.w_7_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → mxxMat2 :=
  ctx.witness.w_7_1

def generatedRoot.w_7_2 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 1025 → mxxMat4 :=
  ctx.witness.w_7_2

def generatedRoot.w_7_3 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 1025 → mxxMat4 :=
  ctx.witness.w_7_3

def generatedRoot.w_7_4 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 1025 → mxxMat1 :=
  ctx.witness.w_7_4

def generatedRoot.w_7_5 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 128 → mxxMat2 :=
  ctx.witness.w_7_5

def generatedRoot.w_7_6 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 128 → mxxMat2 :=
  ctx.witness.w_7_6

def generatedRoot.w_7_7 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 32 → mxxMat2 :=
  ctx.witness.w_7_7

def generatedRoot.w_7_8 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 32 → mxxMat2 :=
  ctx.witness.w_7_8

def generatedRoot.w_7_9 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 128 → mxxMat5 :=
  ctx.witness.w_7_9

def generatedRoot.w_7_10 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 128 → mxxMat5 :=
  ctx.witness.w_7_10

def generatedRoot.w_7_11 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 32 → mxxMat5 :=
  ctx.witness.w_7_11

def generatedRoot.w_7_12 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 32 → mxxMat5 :=
  ctx.witness.w_7_12

def generatedRoot.w_7_13 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 128 → mxxMat6 :=
  ctx.witness.w_7_13

def generatedRoot.w_7_14 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 128 → mxxMat6 :=
  ctx.witness.w_7_14

def generatedRoot.w_7_15 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 32 → mxxMat6 :=
  ctx.witness.w_7_15

def generatedRoot.w_7_16 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 32 → mxxMat6 :=
  ctx.witness.w_7_16

def generatedRoot.w_7_17 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 128 → mxxMat7 :=
  ctx.witness.w_7_17

def generatedRoot.w_7_18 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 128 → mxxMat7 :=
  ctx.witness.w_7_18

def generatedRoot.w_7_19 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 32 → mxxMat7 :=
  ctx.witness.w_7_19

def generatedRoot.w_7_20 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 32 → mxxMat7 :=
  ctx.witness.w_7_20

def generatedRoot.w_7_21 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 128 → mxxMat8 :=
  ctx.witness.w_7_21

def generatedRoot.w_7_22 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 128 → mxxMat8 :=
  ctx.witness.w_7_22

def generatedRoot.w_7_23 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 32 → mxxMat8 :=
  ctx.witness.w_7_23

def generatedRoot.w_7_24 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 32 → mxxMat8 :=
  ctx.witness.w_7_24

def generatedRoot.w_7_25 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 128 → mxxMat9 :=
  ctx.witness.w_7_25

def generatedRoot.w_7_26 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 128 → mxxMat9 :=
  ctx.witness.w_7_26

def generatedRoot.w_7_27 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 32 → mxxMat9 :=
  ctx.witness.w_7_27

def generatedRoot.w_7_28 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 32 → mxxMat9 :=
  ctx.witness.w_7_28

def generatedRoot.w_7_29 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 128 → mxxMat10 :=
  ctx.witness.w_7_29

def generatedRoot.w_7_30 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 128 → mxxMat10 :=
  ctx.witness.w_7_30

def generatedRoot.w_7_31 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 32 → mxxMat10 :=
  ctx.witness.w_7_31

def generatedRoot.w_7_32 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 32 → mxxMat10 :=
  ctx.witness.w_7_32

def generatedRoot.w_7_33 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 128 → mxxMat11 :=
  ctx.witness.w_7_33

def generatedRoot.w_7_34 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 128 → mxxMat11 :=
  ctx.witness.w_7_34

def generatedRoot.w_7_35 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 32 → mxxMat11 :=
  ctx.witness.w_7_35

def generatedRoot.w_7_36 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 32 → mxxMat11 :=
  ctx.witness.w_7_36

def generatedRoot.w_7_37 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 128 → mxxMat12 :=
  ctx.witness.w_7_37

def generatedRoot.w_7_38 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 128 → mxxMat12 :=
  ctx.witness.w_7_38

def generatedRoot.w_7_39 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 32 → mxxMat12 :=
  ctx.witness.w_7_39

def generatedRoot.w_7_40 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 32 → mxxMat12 :=
  ctx.witness.w_7_40

def generatedRoot.w_7_41 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 128 → mxxMat13 :=
  ctx.witness.w_7_41

def generatedRoot.w_7_42 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 128 → mxxMat13 :=
  ctx.witness.w_7_42

def generatedRoot.w_7_43 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 32 → mxxMat13 :=
  ctx.witness.w_7_43

def generatedRoot.w_7_44 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 32 → mxxMat13 :=
  ctx.witness.w_7_44

def generatedRoot.w_7_45 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 128 → mxxMat14 :=
  ctx.witness.w_7_45

def generatedRoot.w_7_46 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 128 → mxxMat14 :=
  ctx.witness.w_7_46

def generatedRoot.w_7_47 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 32 → mxxMat14 :=
  ctx.witness.w_7_47

def generatedRoot.w_7_48 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 32 → mxxMat14 :=
  ctx.witness.w_7_48

def generatedRoot.w_7_49 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 128 → mxxMat15 :=
  ctx.witness.w_7_49

def generatedRoot.w_7_50 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 128 → mxxMat15 :=
  ctx.witness.w_7_50

def generatedRoot.w_7_51 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 32 → mxxMat15 :=
  ctx.witness.w_7_51

def generatedRoot.w_7_52 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 32 → mxxMat15 :=
  ctx.witness.w_7_52

def generatedRoot.w_7_53 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 128 → mxxMat16 :=
  ctx.witness.w_7_53

def generatedRoot.w_7_54 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 128 → mxxMat16 :=
  ctx.witness.w_7_54

def generatedRoot.w_7_55 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 32 → mxxMat16 :=
  ctx.witness.w_7_55

def generatedRoot.w_7_56 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 32 → mxxMat16 :=
  ctx.witness.w_7_56

def generatedRoot.w_7_57 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 128 → mxxMat17 :=
  ctx.witness.w_7_57

def generatedRoot.w_7_58 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 128 → mxxMat17 :=
  ctx.witness.w_7_58

def generatedRoot.w_7_59 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 32 → mxxMat17 :=
  ctx.witness.w_7_59

def generatedRoot.w_7_60 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 32 → mxxMat17 :=
  ctx.witness.w_7_60

def generatedRoot.w_7_61 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 128 → mxxMat18 :=
  ctx.witness.w_7_61

def generatedRoot.w_7_62 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 128 → mxxMat18 :=
  ctx.witness.w_7_62

def generatedRoot.w_7_63 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 32 → mxxMat18 :=
  ctx.witness.w_7_63

def generatedRoot.w_7_64 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 32 → mxxMat18 :=
  ctx.witness.w_7_64

def generatedRoot.w_7_65 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 128 → mxxMat19 :=
  ctx.witness.w_7_65

def generatedRoot.w_7_66 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 128 → mxxMat19 :=
  ctx.witness.w_7_66

def generatedRoot.w_7_67 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 32 → mxxMat19 :=
  ctx.witness.w_7_67

def generatedRoot.w_7_68 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 32 → mxxMat19 :=
  ctx.witness.w_7_68

def generatedRoot.w_7_69 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 128 → mxxMat20 :=
  ctx.witness.w_7_69

def generatedRoot.w_7_70 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 128 → mxxMat20 :=
  ctx.witness.w_7_70

def generatedRoot.w_7_71 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 32 → mxxMat20 :=
  ctx.witness.w_7_71

def generatedRoot.w_7_72 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 32 → mxxMat20 :=
  ctx.witness.w_7_72

def generatedRoot.w_7_73 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 128 → mxxMat21 :=
  ctx.witness.w_7_73

def generatedRoot.w_7_74 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 128 → mxxMat21 :=
  ctx.witness.w_7_74

def generatedRoot.w_7_75 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 32 → mxxMat21 :=
  ctx.witness.w_7_75

def generatedRoot.w_7_76 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 32 → mxxMat21 :=
  ctx.witness.w_7_76

def generatedRoot.w_7_77 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 128 → mxxMat22 :=
  ctx.witness.w_7_77

def generatedRoot.w_7_78 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 128 → mxxMat22 :=
  ctx.witness.w_7_78

def generatedRoot.w_7_79 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 32 → mxxMat22 :=
  ctx.witness.w_7_79

def generatedRoot.w_7_80 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 32 → mxxMat22 :=
  ctx.witness.w_7_80

def generatedRoot.w_7_81 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 128 → mxxMat23 :=
  ctx.witness.w_7_81

def generatedRoot.w_7_82 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 128 → mxxMat23 :=
  ctx.witness.w_7_82

def generatedRoot.w_7_83 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 32 → mxxMat23 :=
  ctx.witness.w_7_83

def generatedRoot.w_7_84 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 32 → mxxMat23 :=
  ctx.witness.w_7_84

def generatedRoot.w_7_85 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 128 → mxxMat24 :=
  ctx.witness.w_7_85

def generatedRoot.w_7_86 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 128 → mxxMat24 :=
  ctx.witness.w_7_86

def generatedRoot.w_7_87 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 32 → mxxMat24 :=
  ctx.witness.w_7_87

def generatedRoot.w_7_88 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 32 → mxxMat24 :=
  ctx.witness.w_7_88

def generatedRoot.w_7_89 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 128 → mxxMat25 :=
  ctx.witness.w_7_89

def generatedRoot.w_7_90 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 128 → mxxMat25 :=
  ctx.witness.w_7_90

def generatedRoot.w_7_91 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 32 → mxxMat25 :=
  ctx.witness.w_7_91

def generatedRoot.w_7_92 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 32 → mxxMat25 :=
  ctx.witness.w_7_92

def generatedRoot.w_7_93 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 128 → mxxMat26 :=
  ctx.witness.w_7_93

def generatedRoot.w_7_94 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 128 → mxxMat26 :=
  ctx.witness.w_7_94

def generatedRoot.w_7_95 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 32 → mxxMat26 :=
  ctx.witness.w_7_95

def generatedRoot.w_7_96 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 32 → mxxMat26 :=
  ctx.witness.w_7_96

def generatedRoot.w_7_97 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 128 → mxxMat27 :=
  ctx.witness.w_7_97

def generatedRoot.w_7_98 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 128 → mxxMat27 :=
  ctx.witness.w_7_98

def generatedRoot.w_7_99 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 32 → mxxMat27 :=
  ctx.witness.w_7_99

def generatedRoot.w_7_100 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 32 → mxxMat27 :=
  ctx.witness.w_7_100

def generatedRoot.w_7_101 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 128 → mxxMat28 :=
  ctx.witness.w_7_101

def generatedRoot.w_7_102 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 128 → mxxMat28 :=
  ctx.witness.w_7_102

def generatedRoot.w_7_103 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 32 → mxxMat28 :=
  ctx.witness.w_7_103

def generatedRoot.w_7_104 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 32 → mxxMat28 :=
  ctx.witness.w_7_104

def generatedRoot.w_7_105 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 128 → mxxMat29 :=
  ctx.witness.w_7_105

def generatedRoot.w_7_106 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 128 → mxxMat29 :=
  ctx.witness.w_7_106

def generatedRoot.w_7_107 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 32 → mxxMat29 :=
  ctx.witness.w_7_107

def generatedRoot.w_7_108 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 32 → mxxMat29 :=
  ctx.witness.w_7_108

def generatedRoot.w_7_109 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 128 → mxxMat30 :=
  ctx.witness.w_7_109

def generatedRoot.w_7_110 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 128 → mxxMat30 :=
  ctx.witness.w_7_110

def generatedRoot.w_7_111 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 32 → mxxMat30 :=
  ctx.witness.w_7_111

def generatedRoot.w_7_112 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 32 → mxxMat30 :=
  ctx.witness.w_7_112

def generatedRoot.w_7_113 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 128 → mxxMat31 :=
  ctx.witness.w_7_113

def generatedRoot.w_7_114 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 128 → mxxMat31 :=
  ctx.witness.w_7_114

def generatedRoot.w_7_115 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 32 → mxxMat31 :=
  ctx.witness.w_7_115

def generatedRoot.w_7_116 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 32 → mxxMat31 :=
  ctx.witness.w_7_116

def generatedRoot.w_7_117 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 128 → mxxMat32 :=
  ctx.witness.w_7_117

def generatedRoot.w_7_118 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 128 → mxxMat32 :=
  ctx.witness.w_7_118

def generatedRoot.w_7_119 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 32 → mxxMat32 :=
  ctx.witness.w_7_119

def generatedRoot.w_7_120 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 32 → mxxMat32 :=
  ctx.witness.w_7_120

def generatedRoot.w_7_121 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 128 → mxxMat33 :=
  ctx.witness.w_7_121

def generatedRoot.w_7_122 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 128 → mxxMat33 :=
  ctx.witness.w_7_122

def generatedRoot.w_7_123 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 32 → mxxMat33 :=
  ctx.witness.w_7_123

def generatedRoot.w_7_124 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 32 → mxxMat33 :=
  ctx.witness.w_7_124

def generatedRoot.w_7_125 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 128 → mxxMat34 :=
  ctx.witness.w_7_125

def generatedRoot.w_7_126 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 128 → mxxMat34 :=
  ctx.witness.w_7_126

def generatedRoot.w_7_127 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 32 → mxxMat34 :=
  ctx.witness.w_7_127

def generatedRoot.w_7_128 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 32 → mxxMat34 :=
  ctx.witness.w_7_128

def generatedRoot.w_7_129 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 128 → mxxMat35 :=
  ctx.witness.w_7_129

def generatedRoot.w_7_130 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 128 → mxxMat35 :=
  ctx.witness.w_7_130

def generatedRoot.w_7_131 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 32 → mxxMat35 :=
  ctx.witness.w_7_131

def generatedRoot.w_7_132 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 32 → mxxMat35 :=
  ctx.witness.w_7_132

def generatedRoot.w_7_133 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 128 → mxxMat36 :=
  ctx.witness.w_7_133

def generatedRoot.w_7_134 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 128 → mxxMat36 :=
  ctx.witness.w_7_134

def generatedRoot.w_7_135 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 32 → mxxMat36 :=
  ctx.witness.w_7_135

def generatedRoot.w_7_136 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 32 → mxxMat36 :=
  ctx.witness.w_7_136

def generatedRoot.w_7_137 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 128 → mxxMat37 :=
  ctx.witness.w_7_137

def generatedRoot.w_7_138 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 128 → mxxMat37 :=
  ctx.witness.w_7_138

def generatedRoot.w_7_139 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 32 → mxxMat37 :=
  ctx.witness.w_7_139

def generatedRoot.w_7_140 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 32 → mxxMat37 :=
  ctx.witness.w_7_140

def generatedRoot.w_7_141 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 128 → mxxMat38 :=
  ctx.witness.w_7_141

def generatedRoot.w_7_142 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 128 → mxxMat38 :=
  ctx.witness.w_7_142

def generatedRoot.w_7_143 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 32 → mxxMat38 :=
  ctx.witness.w_7_143

def generatedRoot.w_7_144 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 32 → mxxMat38 :=
  ctx.witness.w_7_144

def generatedRoot.w_7_145 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 128 → mxxMat39 :=
  ctx.witness.w_7_145

def generatedRoot.w_7_146 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 128 → mxxMat39 :=
  ctx.witness.w_7_146

def generatedRoot.w_7_147 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 32 → mxxMat39 :=
  ctx.witness.w_7_147

def generatedRoot.w_7_148 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 8 → Fin 32 → mxxMat39 :=
  ctx.witness.w_7_148

def generatedRoot.outs {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : (Fin 8 → mxxMat2) × (Fin 8 → mxxMat2) × (Fin 8 → Fin 1025 → mxxMat4) × (Fin 8 → Fin 1025 → mxxMat4) × (Fin 8 → Fin 1025 → mxxMat1) × (Fin 8 → Fin 128 → mxxMat2) × (Fin 8 → Fin 128 → mxxMat2) × (Fin 8 → Fin 32 → mxxMat2) × (Fin 8 → Fin 32 → mxxMat2) × (Fin 8 → Fin 128 → mxxMat5) × (Fin 8 → Fin 128 → mxxMat5) × (Fin 8 → Fin 32 → mxxMat5) × (Fin 8 → Fin 32 → mxxMat5) × (Fin 8 → Fin 128 → mxxMat6) × (Fin 8 → Fin 128 → mxxMat6) × (Fin 8 → Fin 32 → mxxMat6) × (Fin 8 → Fin 32 → mxxMat6) × (Fin 8 → Fin 128 → mxxMat7) × (Fin 8 → Fin 128 → mxxMat7) × (Fin 8 → Fin 32 → mxxMat7) × (Fin 8 → Fin 32 → mxxMat7) × (Fin 8 → Fin 128 → mxxMat8) × (Fin 8 → Fin 128 → mxxMat8) × (Fin 8 → Fin 32 → mxxMat8) × (Fin 8 → Fin 32 → mxxMat8) × (Fin 8 → Fin 128 → mxxMat9) × (Fin 8 → Fin 128 → mxxMat9) × (Fin 8 → Fin 32 → mxxMat9) × (Fin 8 → Fin 32 → mxxMat9) × (Fin 8 → Fin 128 → mxxMat10) × (Fin 8 → Fin 128 → mxxMat10) × (Fin 8 → Fin 32 → mxxMat10) × (Fin 8 → Fin 32 → mxxMat10) × (Fin 8 → Fin 128 → mxxMat11) × (Fin 8 → Fin 128 → mxxMat11) × (Fin 8 → Fin 32 → mxxMat11) × (Fin 8 → Fin 32 → mxxMat11) × (Fin 8 → Fin 128 → mxxMat12) × (Fin 8 → Fin 128 → mxxMat12) × (Fin 8 → Fin 32 → mxxMat12) × (Fin 8 → Fin 32 → mxxMat12) × (Fin 8 → Fin 128 → mxxMat13) × (Fin 8 → Fin 128 → mxxMat13) × (Fin 8 → Fin 32 → mxxMat13) × (Fin 8 → Fin 32 → mxxMat13) × (Fin 8 → Fin 128 → mxxMat14) × (Fin 8 → Fin 128 → mxxMat14) × (Fin 8 → Fin 32 → mxxMat14) × (Fin 8 → Fin 32 → mxxMat14) × (Fin 8 → Fin 128 → mxxMat15) × (Fin 8 → Fin 128 → mxxMat15) × (Fin 8 → Fin 32 → mxxMat15) × (Fin 8 → Fin 32 → mxxMat15) × (Fin 8 → Fin 128 → mxxMat16) × (Fin 8 → Fin 128 → mxxMat16) × (Fin 8 → Fin 32 → mxxMat16) × (Fin 8 → Fin 32 → mxxMat16) × (Fin 8 → Fin 128 → mxxMat17) × (Fin 8 → Fin 128 → mxxMat17) × (Fin 8 → Fin 32 → mxxMat17) × (Fin 8 → Fin 32 → mxxMat17) × (Fin 8 → Fin 128 → mxxMat18) × (Fin 8 → Fin 128 → mxxMat18) × (Fin 8 → Fin 32 → mxxMat18) × (Fin 8 → Fin 32 → mxxMat18) × (Fin 8 → Fin 128 → mxxMat19) × (Fin 8 → Fin 128 → mxxMat19) × (Fin 8 → Fin 32 → mxxMat19) × (Fin 8 → Fin 32 → mxxMat19) × (Fin 8 → Fin 128 → mxxMat20) × (Fin 8 → Fin 128 → mxxMat20) × (Fin 8 → Fin 32 → mxxMat20) × (Fin 8 → Fin 32 → mxxMat20) × (Fin 8 → Fin 128 → mxxMat21) × (Fin 8 → Fin 128 → mxxMat21) × (Fin 8 → Fin 32 → mxxMat21) × (Fin 8 → Fin 32 → mxxMat21) × (Fin 8 → Fin 128 → mxxMat22) × (Fin 8 → Fin 128 → mxxMat22) × (Fin 8 → Fin 32 → mxxMat22) × (Fin 8 → Fin 32 → mxxMat22) × (Fin 8 → Fin 128 → mxxMat23) × (Fin 8 → Fin 128 → mxxMat23) × (Fin 8 → Fin 32 → mxxMat23) × (Fin 8 → Fin 32 → mxxMat23) × (Fin 8 → Fin 128 → mxxMat24) × (Fin 8 → Fin 128 → mxxMat24) × (Fin 8 → Fin 32 → mxxMat24) × (Fin 8 → Fin 32 → mxxMat24) × (Fin 8 → Fin 128 → mxxMat25) × (Fin 8 → Fin 128 → mxxMat25) × (Fin 8 → Fin 32 → mxxMat25) × (Fin 8 → Fin 32 → mxxMat25) × (Fin 8 → Fin 128 → mxxMat26) × (Fin 8 → Fin 128 → mxxMat26) × (Fin 8 → Fin 32 → mxxMat26) × (Fin 8 → Fin 32 → mxxMat26) × (Fin 8 → Fin 128 → mxxMat27) × (Fin 8 → Fin 128 → mxxMat27) × (Fin 8 → Fin 32 → mxxMat27) × (Fin 8 → Fin 32 → mxxMat27) × (Fin 8 → Fin 128 → mxxMat28) × (Fin 8 → Fin 128 → mxxMat28) × (Fin 8 → Fin 32 → mxxMat28) × (Fin 8 → Fin 32 → mxxMat28) × (Fin 8 → Fin 128 → mxxMat29) × (Fin 8 → Fin 128 → mxxMat29) × (Fin 8 → Fin 32 → mxxMat29) × (Fin 8 → Fin 32 → mxxMat29) × (Fin 8 → Fin 128 → mxxMat30) × (Fin 8 → Fin 128 → mxxMat30) × (Fin 8 → Fin 32 → mxxMat30) × (Fin 8 → Fin 32 → mxxMat30) × (Fin 8 → Fin 128 → mxxMat31) × (Fin 8 → Fin 128 → mxxMat31) × (Fin 8 → Fin 32 → mxxMat31) × (Fin 8 → Fin 32 → mxxMat31) × (Fin 8 → Fin 128 → mxxMat32) × (Fin 8 → Fin 128 → mxxMat32) × (Fin 8 → Fin 32 → mxxMat32) × (Fin 8 → Fin 32 → mxxMat32) × (Fin 8 → Fin 128 → mxxMat33) × (Fin 8 → Fin 128 → mxxMat33) × (Fin 8 → Fin 32 → mxxMat33) × (Fin 8 → Fin 32 → mxxMat33) × (Fin 8 → Fin 128 → mxxMat34) × (Fin 8 → Fin 128 → mxxMat34) × (Fin 8 → Fin 32 → mxxMat34) × (Fin 8 → Fin 32 → mxxMat34) × (Fin 8 → Fin 128 → mxxMat35) × (Fin 8 → Fin 128 → mxxMat35) × (Fin 8 → Fin 32 → mxxMat35) × (Fin 8 → Fin 32 → mxxMat35) × (Fin 8 → Fin 128 → mxxMat36) × (Fin 8 → Fin 128 → mxxMat36) × (Fin 8 → Fin 32 → mxxMat36) × (Fin 8 → Fin 32 → mxxMat36) × (Fin 8 → Fin 128 → mxxMat37) × (Fin 8 → Fin 128 → mxxMat37) × (Fin 8 → Fin 32 → mxxMat37) × (Fin 8 → Fin 32 → mxxMat37) × (Fin 8 → Fin 128 → mxxMat38) × (Fin 8 → Fin 128 → mxxMat38) × (Fin 8 → Fin 32 → mxxMat38) × (Fin 8 → Fin 32 → mxxMat38) × (Fin 8 → Fin 128 → mxxMat39) × (Fin 8 → Fin 128 → mxxMat39) × (Fin 8 → Fin 32 → mxxMat39) × (Fin 8 → Fin 32 → mxxMat39) × Unit :=
  ((generatedRoot.w_7_0 ctx), ((generatedRoot.w_7_1 ctx), ((generatedRoot.w_7_2 ctx), ((generatedRoot.w_7_3 ctx), ((generatedRoot.w_7_4 ctx), ((generatedRoot.w_7_5 ctx), ((generatedRoot.w_7_6 ctx), ((generatedRoot.w_7_7 ctx), ((generatedRoot.w_7_8 ctx), ((generatedRoot.w_7_9 ctx), ((generatedRoot.w_7_10 ctx), ((generatedRoot.w_7_11 ctx), ((generatedRoot.w_7_12 ctx), ((generatedRoot.w_7_13 ctx), ((generatedRoot.w_7_14 ctx), ((generatedRoot.w_7_15 ctx), ((generatedRoot.w_7_16 ctx), ((generatedRoot.w_7_17 ctx), ((generatedRoot.w_7_18 ctx), ((generatedRoot.w_7_19 ctx), ((generatedRoot.w_7_20 ctx), ((generatedRoot.w_7_21 ctx), ((generatedRoot.w_7_22 ctx), ((generatedRoot.w_7_23 ctx), ((generatedRoot.w_7_24 ctx), ((generatedRoot.w_7_25 ctx), ((generatedRoot.w_7_26 ctx), ((generatedRoot.w_7_27 ctx), ((generatedRoot.w_7_28 ctx), ((generatedRoot.w_7_29 ctx), ((generatedRoot.w_7_30 ctx), ((generatedRoot.w_7_31 ctx), ((generatedRoot.w_7_32 ctx), ((generatedRoot.w_7_33 ctx), ((generatedRoot.w_7_34 ctx), ((generatedRoot.w_7_35 ctx), ((generatedRoot.w_7_36 ctx), ((generatedRoot.w_7_37 ctx), ((generatedRoot.w_7_38 ctx), ((generatedRoot.w_7_39 ctx), ((generatedRoot.w_7_40 ctx), ((generatedRoot.w_7_41 ctx), ((generatedRoot.w_7_42 ctx), ((generatedRoot.w_7_43 ctx), ((generatedRoot.w_7_44 ctx), ((generatedRoot.w_7_45 ctx), ((generatedRoot.w_7_46 ctx), ((generatedRoot.w_7_47 ctx), ((generatedRoot.w_7_48 ctx), ((generatedRoot.w_7_49 ctx), ((generatedRoot.w_7_50 ctx), ((generatedRoot.w_7_51 ctx), ((generatedRoot.w_7_52 ctx), ((generatedRoot.w_7_53 ctx), ((generatedRoot.w_7_54 ctx), ((generatedRoot.w_7_55 ctx), ((generatedRoot.w_7_56 ctx), ((generatedRoot.w_7_57 ctx), ((generatedRoot.w_7_58 ctx), ((generatedRoot.w_7_59 ctx), ((generatedRoot.w_7_60 ctx), ((generatedRoot.w_7_61 ctx), ((generatedRoot.w_7_62 ctx), ((generatedRoot.w_7_63 ctx), ((generatedRoot.w_7_64 ctx), ((generatedRoot.w_7_65 ctx), ((generatedRoot.w_7_66 ctx), ((generatedRoot.w_7_67 ctx), ((generatedRoot.w_7_68 ctx), ((generatedRoot.w_7_69 ctx), ((generatedRoot.w_7_70 ctx), ((generatedRoot.w_7_71 ctx), ((generatedRoot.w_7_72 ctx), ((generatedRoot.w_7_73 ctx), ((generatedRoot.w_7_74 ctx), ((generatedRoot.w_7_75 ctx), ((generatedRoot.w_7_76 ctx), ((generatedRoot.w_7_77 ctx), ((generatedRoot.w_7_78 ctx), ((generatedRoot.w_7_79 ctx), ((generatedRoot.w_7_80 ctx), ((generatedRoot.w_7_81 ctx), ((generatedRoot.w_7_82 ctx), ((generatedRoot.w_7_83 ctx), ((generatedRoot.w_7_84 ctx), ((generatedRoot.w_7_85 ctx), ((generatedRoot.w_7_86 ctx), ((generatedRoot.w_7_87 ctx), ((generatedRoot.w_7_88 ctx), ((generatedRoot.w_7_89 ctx), ((generatedRoot.w_7_90 ctx), ((generatedRoot.w_7_91 ctx), ((generatedRoot.w_7_92 ctx), ((generatedRoot.w_7_93 ctx), ((generatedRoot.w_7_94 ctx), ((generatedRoot.w_7_95 ctx), ((generatedRoot.w_7_96 ctx), ((generatedRoot.w_7_97 ctx), ((generatedRoot.w_7_98 ctx), ((generatedRoot.w_7_99 ctx), ((generatedRoot.w_7_100 ctx), ((generatedRoot.w_7_101 ctx), ((generatedRoot.w_7_102 ctx), ((generatedRoot.w_7_103 ctx), ((generatedRoot.w_7_104 ctx), ((generatedRoot.w_7_105 ctx), ((generatedRoot.w_7_106 ctx), ((generatedRoot.w_7_107 ctx), ((generatedRoot.w_7_108 ctx), ((generatedRoot.w_7_109 ctx), ((generatedRoot.w_7_110 ctx), ((generatedRoot.w_7_111 ctx), ((generatedRoot.w_7_112 ctx), ((generatedRoot.w_7_113 ctx), ((generatedRoot.w_7_114 ctx), ((generatedRoot.w_7_115 ctx), ((generatedRoot.w_7_116 ctx), ((generatedRoot.w_7_117 ctx), ((generatedRoot.w_7_118 ctx), ((generatedRoot.w_7_119 ctx), ((generatedRoot.w_7_120 ctx), ((generatedRoot.w_7_121 ctx), ((generatedRoot.w_7_122 ctx), ((generatedRoot.w_7_123 ctx), ((generatedRoot.w_7_124 ctx), ((generatedRoot.w_7_125 ctx), ((generatedRoot.w_7_126 ctx), ((generatedRoot.w_7_127 ctx), ((generatedRoot.w_7_128 ctx), ((generatedRoot.w_7_129 ctx), ((generatedRoot.w_7_130 ctx), ((generatedRoot.w_7_131 ctx), ((generatedRoot.w_7_132 ctx), ((generatedRoot.w_7_133 ctx), ((generatedRoot.w_7_134 ctx), ((generatedRoot.w_7_135 ctx), ((generatedRoot.w_7_136 ctx), ((generatedRoot.w_7_137 ctx), ((generatedRoot.w_7_138 ctx), ((generatedRoot.w_7_139 ctx), ((generatedRoot.w_7_140 ctx), ((generatedRoot.w_7_141 ctx), ((generatedRoot.w_7_142 ctx), ((generatedRoot.w_7_143 ctx), ((generatedRoot.w_7_144 ctx), ((generatedRoot.w_7_145 ctx), ((generatedRoot.w_7_146 ctx), ((generatedRoot.w_7_147 ctx), ((generatedRoot.w_7_148 ctx), ())))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))

structure generatedRoot.Facts_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) (outputs : (Fin 8 → mxxMat2) × (Fin 8 → mxxMat2) × (Fin 8 → Fin 1025 → mxxMat4) × (Fin 8 → Fin 1025 → mxxMat4) × (Fin 8 → Fin 1025 → mxxMat1) × (Fin 8 → Fin 128 → mxxMat2) × (Fin 8 → Fin 128 → mxxMat2) × (Fin 8 → Fin 32 → mxxMat2) × (Fin 8 → Fin 32 → mxxMat2) × (Fin 8 → Fin 128 → mxxMat5) × (Fin 8 → Fin 128 → mxxMat5) × (Fin 8 → Fin 32 → mxxMat5) × (Fin 8 → Fin 32 → mxxMat5) × (Fin 8 → Fin 128 → mxxMat6) × (Fin 8 → Fin 128 → mxxMat6) × (Fin 8 → Fin 32 → mxxMat6) × (Fin 8 → Fin 32 → mxxMat6) × (Fin 8 → Fin 128 → mxxMat7) × (Fin 8 → Fin 128 → mxxMat7) × (Fin 8 → Fin 32 → mxxMat7) × (Fin 8 → Fin 32 → mxxMat7) × (Fin 8 → Fin 128 → mxxMat8) × (Fin 8 → Fin 128 → mxxMat8) × (Fin 8 → Fin 32 → mxxMat8) × (Fin 8 → Fin 32 → mxxMat8) × (Fin 8 → Fin 128 → mxxMat9) × (Fin 8 → Fin 128 → mxxMat9) × (Fin 8 → Fin 32 → mxxMat9) × (Fin 8 → Fin 32 → mxxMat9) × (Fin 8 → Fin 128 → mxxMat10) × (Fin 8 → Fin 128 → mxxMat10) × (Fin 8 → Fin 32 → mxxMat10) × (Fin 8 → Fin 32 → mxxMat10) × (Fin 8 → Fin 128 → mxxMat11) × (Fin 8 → Fin 128 → mxxMat11) × (Fin 8 → Fin 32 → mxxMat11) × (Fin 8 → Fin 32 → mxxMat11) × (Fin 8 → Fin 128 → mxxMat12) × (Fin 8 → Fin 128 → mxxMat12) × (Fin 8 → Fin 32 → mxxMat12) × (Fin 8 → Fin 32 → mxxMat12) × (Fin 8 → Fin 128 → mxxMat13) × (Fin 8 → Fin 128 → mxxMat13) × (Fin 8 → Fin 32 → mxxMat13) × (Fin 8 → Fin 32 → mxxMat13) × (Fin 8 → Fin 128 → mxxMat14) × (Fin 8 → Fin 128 → mxxMat14) × (Fin 8 → Fin 32 → mxxMat14) × (Fin 8 → Fin 32 → mxxMat14) × (Fin 8 → Fin 128 → mxxMat15) × (Fin 8 → Fin 128 → mxxMat15) × (Fin 8 → Fin 32 → mxxMat15) × (Fin 8 → Fin 32 → mxxMat15) × (Fin 8 → Fin 128 → mxxMat16) × (Fin 8 → Fin 128 → mxxMat16) × (Fin 8 → Fin 32 → mxxMat16) × (Fin 8 → Fin 32 → mxxMat16) × (Fin 8 → Fin 128 → mxxMat17) × (Fin 8 → Fin 128 → mxxMat17) × (Fin 8 → Fin 32 → mxxMat17) × (Fin 8 → Fin 32 → mxxMat17) × (Fin 8 → Fin 128 → mxxMat18) × (Fin 8 → Fin 128 → mxxMat18) × (Fin 8 → Fin 32 → mxxMat18) × (Fin 8 → Fin 32 → mxxMat18) × (Fin 8 → Fin 128 → mxxMat19) × (Fin 8 → Fin 128 → mxxMat19) × (Fin 8 → Fin 32 → mxxMat19) × (Fin 8 → Fin 32 → mxxMat19) × (Fin 8 → Fin 128 → mxxMat20) × (Fin 8 → Fin 128 → mxxMat20) × (Fin 8 → Fin 32 → mxxMat20) × (Fin 8 → Fin 32 → mxxMat20) × (Fin 8 → Fin 128 → mxxMat21) × (Fin 8 → Fin 128 → mxxMat21) × (Fin 8 → Fin 32 → mxxMat21) × (Fin 8 → Fin 32 → mxxMat21) × (Fin 8 → Fin 128 → mxxMat22) × (Fin 8 → Fin 128 → mxxMat22) × (Fin 8 → Fin 32 → mxxMat22) × (Fin 8 → Fin 32 → mxxMat22) × (Fin 8 → Fin 128 → mxxMat23) × (Fin 8 → Fin 128 → mxxMat23) × (Fin 8 → Fin 32 → mxxMat23) × (Fin 8 → Fin 32 → mxxMat23) × (Fin 8 → Fin 128 → mxxMat24) × (Fin 8 → Fin 128 → mxxMat24) × (Fin 8 → Fin 32 → mxxMat24) × (Fin 8 → Fin 32 → mxxMat24) × (Fin 8 → Fin 128 → mxxMat25) × (Fin 8 → Fin 128 → mxxMat25) × (Fin 8 → Fin 32 → mxxMat25) × (Fin 8 → Fin 32 → mxxMat25) × (Fin 8 → Fin 128 → mxxMat26) × (Fin 8 → Fin 128 → mxxMat26) × (Fin 8 → Fin 32 → mxxMat26) × (Fin 8 → Fin 32 → mxxMat26) × (Fin 8 → Fin 128 → mxxMat27) × (Fin 8 → Fin 128 → mxxMat27) × (Fin 8 → Fin 32 → mxxMat27) × (Fin 8 → Fin 32 → mxxMat27) × (Fin 8 → Fin 128 → mxxMat28) × (Fin 8 → Fin 128 → mxxMat28) × (Fin 8 → Fin 32 → mxxMat28) × (Fin 8 → Fin 32 → mxxMat28) × (Fin 8 → Fin 128 → mxxMat29) × (Fin 8 → Fin 128 → mxxMat29) × (Fin 8 → Fin 32 → mxxMat29) × (Fin 8 → Fin 32 → mxxMat29) × (Fin 8 → Fin 128 → mxxMat30) × (Fin 8 → Fin 128 → mxxMat30) × (Fin 8 → Fin 32 → mxxMat30) × (Fin 8 → Fin 32 → mxxMat30) × (Fin 8 → Fin 128 → mxxMat31) × (Fin 8 → Fin 128 → mxxMat31) × (Fin 8 → Fin 32 → mxxMat31) × (Fin 8 → Fin 32 → mxxMat31) × (Fin 8 → Fin 128 → mxxMat32) × (Fin 8 → Fin 128 → mxxMat32) × (Fin 8 → Fin 32 → mxxMat32) × (Fin 8 → Fin 32 → mxxMat32) × (Fin 8 → Fin 128 → mxxMat33) × (Fin 8 → Fin 128 → mxxMat33) × (Fin 8 → Fin 32 → mxxMat33) × (Fin 8 → Fin 32 → mxxMat33) × (Fin 8 → Fin 128 → mxxMat34) × (Fin 8 → Fin 128 → mxxMat34) × (Fin 8 → Fin 32 → mxxMat34) × (Fin 8 → Fin 32 → mxxMat34) × (Fin 8 → Fin 128 → mxxMat35) × (Fin 8 → Fin 128 → mxxMat35) × (Fin 8 → Fin 32 → mxxMat35) × (Fin 8 → Fin 32 → mxxMat35) × (Fin 8 → Fin 128 → mxxMat36) × (Fin 8 → Fin 128 → mxxMat36) × (Fin 8 → Fin 32 → mxxMat36) × (Fin 8 → Fin 32 → mxxMat36) × (Fin 8 → Fin 128 → mxxMat37) × (Fin 8 → Fin 128 → mxxMat37) × (Fin 8 → Fin 32 → mxxMat37) × (Fin 8 → Fin 32 → mxxMat37) × (Fin 8 → Fin 128 → mxxMat38) × (Fin 8 → Fin 128 → mxxMat38) × (Fin 8 → Fin 32 → mxxMat38) × (Fin 8 → Fin 32 → mxxMat38) × (Fin 8 → Fin 128 → mxxMat39) × (Fin 8 → Fin 128 → mxxMat39) × (Fin 8 → Fin 32 → mxxMat39) × (Fin 8 → Fin 32 → mxxMat39) × Unit) : Prop where
  t0 : (8) = 8
  h7_148 : (∀ i : Fin 8, parallel_generatedRoot_7 backend hashModel tape (path ++ [7, i.val]) params i ((generatedRoot.w_0_0 ctx), (((generatedRoot.w_1_0 ctx) i), ((generatedRoot.w_2_0 ctx), (((generatedRoot.w_3_0 ctx) i), (((generatedRoot.w_4_0 ctx) i), (((generatedRoot.w_5_0 ctx) i), (((generatedRoot.w_6_0 ctx) i), ()))))))) (((generatedRoot.w_7_0 ctx) i), (((generatedRoot.w_7_1 ctx) i), (((generatedRoot.w_7_2 ctx) i), (((generatedRoot.w_7_3 ctx) i), (((generatedRoot.w_7_4 ctx) i), (((generatedRoot.w_7_5 ctx) i), (((generatedRoot.w_7_6 ctx) i), (((generatedRoot.w_7_7 ctx) i), (((generatedRoot.w_7_8 ctx) i), (((generatedRoot.w_7_9 ctx) i), (((generatedRoot.w_7_10 ctx) i), (((generatedRoot.w_7_11 ctx) i), (((generatedRoot.w_7_12 ctx) i), (((generatedRoot.w_7_13 ctx) i), (((generatedRoot.w_7_14 ctx) i), (((generatedRoot.w_7_15 ctx) i), (((generatedRoot.w_7_16 ctx) i), (((generatedRoot.w_7_17 ctx) i), (((generatedRoot.w_7_18 ctx) i), (((generatedRoot.w_7_19 ctx) i), (((generatedRoot.w_7_20 ctx) i), (((generatedRoot.w_7_21 ctx) i), (((generatedRoot.w_7_22 ctx) i), (((generatedRoot.w_7_23 ctx) i), (((generatedRoot.w_7_24 ctx) i), (((generatedRoot.w_7_25 ctx) i), (((generatedRoot.w_7_26 ctx) i), (((generatedRoot.w_7_27 ctx) i), (((generatedRoot.w_7_28 ctx) i), (((generatedRoot.w_7_29 ctx) i), (((generatedRoot.w_7_30 ctx) i), (((generatedRoot.w_7_31 ctx) i), (((generatedRoot.w_7_32 ctx) i), (((generatedRoot.w_7_33 ctx) i), (((generatedRoot.w_7_34 ctx) i), (((generatedRoot.w_7_35 ctx) i), (((generatedRoot.w_7_36 ctx) i), (((generatedRoot.w_7_37 ctx) i), (((generatedRoot.w_7_38 ctx) i), (((generatedRoot.w_7_39 ctx) i), (((generatedRoot.w_7_40 ctx) i), (((generatedRoot.w_7_41 ctx) i), (((generatedRoot.w_7_42 ctx) i), (((generatedRoot.w_7_43 ctx) i), (((generatedRoot.w_7_44 ctx) i), (((generatedRoot.w_7_45 ctx) i), (((generatedRoot.w_7_46 ctx) i), (((generatedRoot.w_7_47 ctx) i), (((generatedRoot.w_7_48 ctx) i), (((generatedRoot.w_7_49 ctx) i), (((generatedRoot.w_7_50 ctx) i), (((generatedRoot.w_7_51 ctx) i), (((generatedRoot.w_7_52 ctx) i), (((generatedRoot.w_7_53 ctx) i), (((generatedRoot.w_7_54 ctx) i), (((generatedRoot.w_7_55 ctx) i), (((generatedRoot.w_7_56 ctx) i), (((generatedRoot.w_7_57 ctx) i), (((generatedRoot.w_7_58 ctx) i), (((generatedRoot.w_7_59 ctx) i), (((generatedRoot.w_7_60 ctx) i), (((generatedRoot.w_7_61 ctx) i), (((generatedRoot.w_7_62 ctx) i), (((generatedRoot.w_7_63 ctx) i), (((generatedRoot.w_7_64 ctx) i), (((generatedRoot.w_7_65 ctx) i), (((generatedRoot.w_7_66 ctx) i), (((generatedRoot.w_7_67 ctx) i), (((generatedRoot.w_7_68 ctx) i), (((generatedRoot.w_7_69 ctx) i), (((generatedRoot.w_7_70 ctx) i), (((generatedRoot.w_7_71 ctx) i), (((generatedRoot.w_7_72 ctx) i), (((generatedRoot.w_7_73 ctx) i), (((generatedRoot.w_7_74 ctx) i), (((generatedRoot.w_7_75 ctx) i), (((generatedRoot.w_7_76 ctx) i), (((generatedRoot.w_7_77 ctx) i), (((generatedRoot.w_7_78 ctx) i), (((generatedRoot.w_7_79 ctx) i), (((generatedRoot.w_7_80 ctx) i), (((generatedRoot.w_7_81 ctx) i), (((generatedRoot.w_7_82 ctx) i), (((generatedRoot.w_7_83 ctx) i), (((generatedRoot.w_7_84 ctx) i), (((generatedRoot.w_7_85 ctx) i), (((generatedRoot.w_7_86 ctx) i), (((generatedRoot.w_7_87 ctx) i), (((generatedRoot.w_7_88 ctx) i), (((generatedRoot.w_7_89 ctx) i), (((generatedRoot.w_7_90 ctx) i), (((generatedRoot.w_7_91 ctx) i), (((generatedRoot.w_7_92 ctx) i), (((generatedRoot.w_7_93 ctx) i), (((generatedRoot.w_7_94 ctx) i), (((generatedRoot.w_7_95 ctx) i), (((generatedRoot.w_7_96 ctx) i), (((generatedRoot.w_7_97 ctx) i), (((generatedRoot.w_7_98 ctx) i), (((generatedRoot.w_7_99 ctx) i), (((generatedRoot.w_7_100 ctx) i), (((generatedRoot.w_7_101 ctx) i), (((generatedRoot.w_7_102 ctx) i), (((generatedRoot.w_7_103 ctx) i), (((generatedRoot.w_7_104 ctx) i), (((generatedRoot.w_7_105 ctx) i), (((generatedRoot.w_7_106 ctx) i), (((generatedRoot.w_7_107 ctx) i), (((generatedRoot.w_7_108 ctx) i), (((generatedRoot.w_7_109 ctx) i), (((generatedRoot.w_7_110 ctx) i), (((generatedRoot.w_7_111 ctx) i), (((generatedRoot.w_7_112 ctx) i), (((generatedRoot.w_7_113 ctx) i), (((generatedRoot.w_7_114 ctx) i), (((generatedRoot.w_7_115 ctx) i), (((generatedRoot.w_7_116 ctx) i), (((generatedRoot.w_7_117 ctx) i), (((generatedRoot.w_7_118 ctx) i), (((generatedRoot.w_7_119 ctx) i), (((generatedRoot.w_7_120 ctx) i), (((generatedRoot.w_7_121 ctx) i), (((generatedRoot.w_7_122 ctx) i), (((generatedRoot.w_7_123 ctx) i), (((generatedRoot.w_7_124 ctx) i), (((generatedRoot.w_7_125 ctx) i), (((generatedRoot.w_7_126 ctx) i), (((generatedRoot.w_7_127 ctx) i), (((generatedRoot.w_7_128 ctx) i), (((generatedRoot.w_7_129 ctx) i), (((generatedRoot.w_7_130 ctx) i), (((generatedRoot.w_7_131 ctx) i), (((generatedRoot.w_7_132 ctx) i), (((generatedRoot.w_7_133 ctx) i), (((generatedRoot.w_7_134 ctx) i), (((generatedRoot.w_7_135 ctx) i), (((generatedRoot.w_7_136 ctx) i), (((generatedRoot.w_7_137 ctx) i), (((generatedRoot.w_7_138 ctx) i), (((generatedRoot.w_7_139 ctx) i), (((generatedRoot.w_7_140 ctx) i), (((generatedRoot.w_7_141 ctx) i), (((generatedRoot.w_7_142 ctx) i), (((generatedRoot.w_7_143 ctx) i), (((generatedRoot.w_7_144 ctx) i), (((generatedRoot.w_7_145 ctx) i), (((generatedRoot.w_7_146 ctx) i), (((generatedRoot.w_7_147 ctx) i), (((generatedRoot.w_7_148 ctx) i), ()))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))
  hout : outputs = generatedRoot.outs ctx

theorem generatedRoot.facts_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {ctx : generatedRoot.Ctx backend hashModel tape path params} {outputs : (Fin 8 → mxxMat2) × (Fin 8 → mxxMat2) × (Fin 8 → Fin 1025 → mxxMat4) × (Fin 8 → Fin 1025 → mxxMat4) × (Fin 8 → Fin 1025 → mxxMat1) × (Fin 8 → Fin 128 → mxxMat2) × (Fin 8 → Fin 128 → mxxMat2) × (Fin 8 → Fin 32 → mxxMat2) × (Fin 8 → Fin 32 → mxxMat2) × (Fin 8 → Fin 128 → mxxMat5) × (Fin 8 → Fin 128 → mxxMat5) × (Fin 8 → Fin 32 → mxxMat5) × (Fin 8 → Fin 32 → mxxMat5) × (Fin 8 → Fin 128 → mxxMat6) × (Fin 8 → Fin 128 → mxxMat6) × (Fin 8 → Fin 32 → mxxMat6) × (Fin 8 → Fin 32 → mxxMat6) × (Fin 8 → Fin 128 → mxxMat7) × (Fin 8 → Fin 128 → mxxMat7) × (Fin 8 → Fin 32 → mxxMat7) × (Fin 8 → Fin 32 → mxxMat7) × (Fin 8 → Fin 128 → mxxMat8) × (Fin 8 → Fin 128 → mxxMat8) × (Fin 8 → Fin 32 → mxxMat8) × (Fin 8 → Fin 32 → mxxMat8) × (Fin 8 → Fin 128 → mxxMat9) × (Fin 8 → Fin 128 → mxxMat9) × (Fin 8 → Fin 32 → mxxMat9) × (Fin 8 → Fin 32 → mxxMat9) × (Fin 8 → Fin 128 → mxxMat10) × (Fin 8 → Fin 128 → mxxMat10) × (Fin 8 → Fin 32 → mxxMat10) × (Fin 8 → Fin 32 → mxxMat10) × (Fin 8 → Fin 128 → mxxMat11) × (Fin 8 → Fin 128 → mxxMat11) × (Fin 8 → Fin 32 → mxxMat11) × (Fin 8 → Fin 32 → mxxMat11) × (Fin 8 → Fin 128 → mxxMat12) × (Fin 8 → Fin 128 → mxxMat12) × (Fin 8 → Fin 32 → mxxMat12) × (Fin 8 → Fin 32 → mxxMat12) × (Fin 8 → Fin 128 → mxxMat13) × (Fin 8 → Fin 128 → mxxMat13) × (Fin 8 → Fin 32 → mxxMat13) × (Fin 8 → Fin 32 → mxxMat13) × (Fin 8 → Fin 128 → mxxMat14) × (Fin 8 → Fin 128 → mxxMat14) × (Fin 8 → Fin 32 → mxxMat14) × (Fin 8 → Fin 32 → mxxMat14) × (Fin 8 → Fin 128 → mxxMat15) × (Fin 8 → Fin 128 → mxxMat15) × (Fin 8 → Fin 32 → mxxMat15) × (Fin 8 → Fin 32 → mxxMat15) × (Fin 8 → Fin 128 → mxxMat16) × (Fin 8 → Fin 128 → mxxMat16) × (Fin 8 → Fin 32 → mxxMat16) × (Fin 8 → Fin 32 → mxxMat16) × (Fin 8 → Fin 128 → mxxMat17) × (Fin 8 → Fin 128 → mxxMat17) × (Fin 8 → Fin 32 → mxxMat17) × (Fin 8 → Fin 32 → mxxMat17) × (Fin 8 → Fin 128 → mxxMat18) × (Fin 8 → Fin 128 → mxxMat18) × (Fin 8 → Fin 32 → mxxMat18) × (Fin 8 → Fin 32 → mxxMat18) × (Fin 8 → Fin 128 → mxxMat19) × (Fin 8 → Fin 128 → mxxMat19) × (Fin 8 → Fin 32 → mxxMat19) × (Fin 8 → Fin 32 → mxxMat19) × (Fin 8 → Fin 128 → mxxMat20) × (Fin 8 → Fin 128 → mxxMat20) × (Fin 8 → Fin 32 → mxxMat20) × (Fin 8 → Fin 32 → mxxMat20) × (Fin 8 → Fin 128 → mxxMat21) × (Fin 8 → Fin 128 → mxxMat21) × (Fin 8 → Fin 32 → mxxMat21) × (Fin 8 → Fin 32 → mxxMat21) × (Fin 8 → Fin 128 → mxxMat22) × (Fin 8 → Fin 128 → mxxMat22) × (Fin 8 → Fin 32 → mxxMat22) × (Fin 8 → Fin 32 → mxxMat22) × (Fin 8 → Fin 128 → mxxMat23) × (Fin 8 → Fin 128 → mxxMat23) × (Fin 8 → Fin 32 → mxxMat23) × (Fin 8 → Fin 32 → mxxMat23) × (Fin 8 → Fin 128 → mxxMat24) × (Fin 8 → Fin 128 → mxxMat24) × (Fin 8 → Fin 32 → mxxMat24) × (Fin 8 → Fin 32 → mxxMat24) × (Fin 8 → Fin 128 → mxxMat25) × (Fin 8 → Fin 128 → mxxMat25) × (Fin 8 → Fin 32 → mxxMat25) × (Fin 8 → Fin 32 → mxxMat25) × (Fin 8 → Fin 128 → mxxMat26) × (Fin 8 → Fin 128 → mxxMat26) × (Fin 8 → Fin 32 → mxxMat26) × (Fin 8 → Fin 32 → mxxMat26) × (Fin 8 → Fin 128 → mxxMat27) × (Fin 8 → Fin 128 → mxxMat27) × (Fin 8 → Fin 32 → mxxMat27) × (Fin 8 → Fin 32 → mxxMat27) × (Fin 8 → Fin 128 → mxxMat28) × (Fin 8 → Fin 128 → mxxMat28) × (Fin 8 → Fin 32 → mxxMat28) × (Fin 8 → Fin 32 → mxxMat28) × (Fin 8 → Fin 128 → mxxMat29) × (Fin 8 → Fin 128 → mxxMat29) × (Fin 8 → Fin 32 → mxxMat29) × (Fin 8 → Fin 32 → mxxMat29) × (Fin 8 → Fin 128 → mxxMat30) × (Fin 8 → Fin 128 → mxxMat30) × (Fin 8 → Fin 32 → mxxMat30) × (Fin 8 → Fin 32 → mxxMat30) × (Fin 8 → Fin 128 → mxxMat31) × (Fin 8 → Fin 128 → mxxMat31) × (Fin 8 → Fin 32 → mxxMat31) × (Fin 8 → Fin 32 → mxxMat31) × (Fin 8 → Fin 128 → mxxMat32) × (Fin 8 → Fin 128 → mxxMat32) × (Fin 8 → Fin 32 → mxxMat32) × (Fin 8 → Fin 32 → mxxMat32) × (Fin 8 → Fin 128 → mxxMat33) × (Fin 8 → Fin 128 → mxxMat33) × (Fin 8 → Fin 32 → mxxMat33) × (Fin 8 → Fin 32 → mxxMat33) × (Fin 8 → Fin 128 → mxxMat34) × (Fin 8 → Fin 128 → mxxMat34) × (Fin 8 → Fin 32 → mxxMat34) × (Fin 8 → Fin 32 → mxxMat34) × (Fin 8 → Fin 128 → mxxMat35) × (Fin 8 → Fin 128 → mxxMat35) × (Fin 8 → Fin 32 → mxxMat35) × (Fin 8 → Fin 32 → mxxMat35) × (Fin 8 → Fin 128 → mxxMat36) × (Fin 8 → Fin 128 → mxxMat36) × (Fin 8 → Fin 32 → mxxMat36) × (Fin 8 → Fin 32 → mxxMat36) × (Fin 8 → Fin 128 → mxxMat37) × (Fin 8 → Fin 128 → mxxMat37) × (Fin 8 → Fin 32 → mxxMat37) × (Fin 8 → Fin 32 → mxxMat37) × (Fin 8 → Fin 128 → mxxMat38) × (Fin 8 → Fin 128 → mxxMat38) × (Fin 8 → Fin 32 → mxxMat38) × (Fin 8 → Fin 32 → mxxMat38) × (Fin 8 → Fin 128 → mxxMat39) × (Fin 8 → Fin 128 → mxxMat39) × (Fin 8 → Fin 32 → mxxMat39) × (Fin 8 → Fin 32 → mxxMat39) × Unit}
    (h : generatedRoot.constraints_0 backend hashModel tape path params (generatedRoot.w_0_0 ctx) (generatedRoot.w_1_0 ctx) (generatedRoot.w_2_0 ctx) (generatedRoot.w_3_0 ctx) (generatedRoot.w_4_0 ctx) (generatedRoot.w_5_0 ctx) (generatedRoot.w_6_0 ctx) (generatedRoot.w_7_0 ctx) (generatedRoot.w_7_1 ctx) (generatedRoot.w_7_2 ctx) (generatedRoot.w_7_3 ctx) (generatedRoot.w_7_4 ctx) (generatedRoot.w_7_5 ctx) (generatedRoot.w_7_6 ctx) (generatedRoot.w_7_7 ctx) (generatedRoot.w_7_8 ctx) (generatedRoot.w_7_9 ctx) (generatedRoot.w_7_10 ctx) (generatedRoot.w_7_11 ctx) (generatedRoot.w_7_12 ctx) (generatedRoot.w_7_13 ctx) (generatedRoot.w_7_14 ctx) (generatedRoot.w_7_15 ctx) (generatedRoot.w_7_16 ctx) (generatedRoot.w_7_17 ctx) (generatedRoot.w_7_18 ctx) (generatedRoot.w_7_19 ctx) (generatedRoot.w_7_20 ctx) (generatedRoot.w_7_21 ctx) (generatedRoot.w_7_22 ctx) (generatedRoot.w_7_23 ctx) (generatedRoot.w_7_24 ctx) (generatedRoot.w_7_25 ctx) (generatedRoot.w_7_26 ctx) (generatedRoot.w_7_27 ctx) (generatedRoot.w_7_28 ctx) (generatedRoot.w_7_29 ctx) (generatedRoot.w_7_30 ctx) (generatedRoot.w_7_31 ctx) (generatedRoot.w_7_32 ctx) (generatedRoot.w_7_33 ctx) (generatedRoot.w_7_34 ctx) (generatedRoot.w_7_35 ctx) (generatedRoot.w_7_36 ctx) (generatedRoot.w_7_37 ctx) (generatedRoot.w_7_38 ctx) (generatedRoot.w_7_39 ctx) (generatedRoot.w_7_40 ctx) (generatedRoot.w_7_41 ctx) (generatedRoot.w_7_42 ctx) (generatedRoot.w_7_43 ctx) (generatedRoot.w_7_44 ctx) (generatedRoot.w_7_45 ctx) (generatedRoot.w_7_46 ctx) (generatedRoot.w_7_47 ctx) (generatedRoot.w_7_48 ctx) (generatedRoot.w_7_49 ctx) (generatedRoot.w_7_50 ctx) (generatedRoot.w_7_51 ctx) (generatedRoot.w_7_52 ctx) (generatedRoot.w_7_53 ctx) (generatedRoot.w_7_54 ctx) (generatedRoot.w_7_55 ctx) (generatedRoot.w_7_56 ctx) (generatedRoot.w_7_57 ctx) (generatedRoot.w_7_58 ctx) (generatedRoot.w_7_59 ctx) (generatedRoot.w_7_60 ctx) (generatedRoot.w_7_61 ctx) (generatedRoot.w_7_62 ctx) (generatedRoot.w_7_63 ctx) (generatedRoot.w_7_64 ctx) (generatedRoot.w_7_65 ctx) (generatedRoot.w_7_66 ctx) (generatedRoot.w_7_67 ctx) (generatedRoot.w_7_68 ctx) (generatedRoot.w_7_69 ctx) (generatedRoot.w_7_70 ctx) (generatedRoot.w_7_71 ctx) (generatedRoot.w_7_72 ctx) (generatedRoot.w_7_73 ctx) (generatedRoot.w_7_74 ctx) (generatedRoot.w_7_75 ctx) (generatedRoot.w_7_76 ctx) (generatedRoot.w_7_77 ctx) (generatedRoot.w_7_78 ctx) (generatedRoot.w_7_79 ctx) (generatedRoot.w_7_80 ctx) (generatedRoot.w_7_81 ctx) (generatedRoot.w_7_82 ctx) (generatedRoot.w_7_83 ctx) (generatedRoot.w_7_84 ctx) (generatedRoot.w_7_85 ctx) (generatedRoot.w_7_86 ctx) (generatedRoot.w_7_87 ctx) (generatedRoot.w_7_88 ctx) (generatedRoot.w_7_89 ctx) (generatedRoot.w_7_90 ctx) (generatedRoot.w_7_91 ctx) (generatedRoot.w_7_92 ctx) (generatedRoot.w_7_93 ctx) (generatedRoot.w_7_94 ctx) (generatedRoot.w_7_95 ctx) (generatedRoot.w_7_96 ctx) (generatedRoot.w_7_97 ctx) (generatedRoot.w_7_98 ctx) (generatedRoot.w_7_99 ctx) (generatedRoot.w_7_100 ctx) (generatedRoot.w_7_101 ctx) (generatedRoot.w_7_102 ctx) (generatedRoot.w_7_103 ctx) (generatedRoot.w_7_104 ctx) (generatedRoot.w_7_105 ctx) (generatedRoot.w_7_106 ctx) (generatedRoot.w_7_107 ctx) (generatedRoot.w_7_108 ctx) (generatedRoot.w_7_109 ctx) (generatedRoot.w_7_110 ctx) (generatedRoot.w_7_111 ctx) (generatedRoot.w_7_112 ctx) (generatedRoot.w_7_113 ctx) (generatedRoot.w_7_114 ctx) (generatedRoot.w_7_115 ctx) (generatedRoot.w_7_116 ctx) (generatedRoot.w_7_117 ctx) (generatedRoot.w_7_118 ctx) (generatedRoot.w_7_119 ctx) (generatedRoot.w_7_120 ctx) (generatedRoot.w_7_121 ctx) (generatedRoot.w_7_122 ctx) (generatedRoot.w_7_123 ctx) (generatedRoot.w_7_124 ctx) (generatedRoot.w_7_125 ctx) (generatedRoot.w_7_126 ctx) (generatedRoot.w_7_127 ctx) (generatedRoot.w_7_128 ctx) (generatedRoot.w_7_129 ctx) (generatedRoot.w_7_130 ctx) (generatedRoot.w_7_131 ctx) (generatedRoot.w_7_132 ctx) (generatedRoot.w_7_133 ctx) (generatedRoot.w_7_134 ctx) (generatedRoot.w_7_135 ctx) (generatedRoot.w_7_136 ctx) (generatedRoot.w_7_137 ctx) (generatedRoot.w_7_138 ctx) (generatedRoot.w_7_139 ctx) (generatedRoot.w_7_140 ctx) (generatedRoot.w_7_141 ctx) (generatedRoot.w_7_142 ctx) (generatedRoot.w_7_143 ctx) (generatedRoot.w_7_144 ctx) (generatedRoot.w_7_145 ctx) (generatedRoot.w_7_146 ctx) (generatedRoot.w_7_147 ctx) (generatedRoot.w_7_148 ctx) outputs) :
    generatedRoot.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure generatedRoot.Facts {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) (outputs : (Fin 8 → mxxMat2) × (Fin 8 → mxxMat2) × (Fin 8 → Fin 1025 → mxxMat4) × (Fin 8 → Fin 1025 → mxxMat4) × (Fin 8 → Fin 1025 → mxxMat1) × (Fin 8 → Fin 128 → mxxMat2) × (Fin 8 → Fin 128 → mxxMat2) × (Fin 8 → Fin 32 → mxxMat2) × (Fin 8 → Fin 32 → mxxMat2) × (Fin 8 → Fin 128 → mxxMat5) × (Fin 8 → Fin 128 → mxxMat5) × (Fin 8 → Fin 32 → mxxMat5) × (Fin 8 → Fin 32 → mxxMat5) × (Fin 8 → Fin 128 → mxxMat6) × (Fin 8 → Fin 128 → mxxMat6) × (Fin 8 → Fin 32 → mxxMat6) × (Fin 8 → Fin 32 → mxxMat6) × (Fin 8 → Fin 128 → mxxMat7) × (Fin 8 → Fin 128 → mxxMat7) × (Fin 8 → Fin 32 → mxxMat7) × (Fin 8 → Fin 32 → mxxMat7) × (Fin 8 → Fin 128 → mxxMat8) × (Fin 8 → Fin 128 → mxxMat8) × (Fin 8 → Fin 32 → mxxMat8) × (Fin 8 → Fin 32 → mxxMat8) × (Fin 8 → Fin 128 → mxxMat9) × (Fin 8 → Fin 128 → mxxMat9) × (Fin 8 → Fin 32 → mxxMat9) × (Fin 8 → Fin 32 → mxxMat9) × (Fin 8 → Fin 128 → mxxMat10) × (Fin 8 → Fin 128 → mxxMat10) × (Fin 8 → Fin 32 → mxxMat10) × (Fin 8 → Fin 32 → mxxMat10) × (Fin 8 → Fin 128 → mxxMat11) × (Fin 8 → Fin 128 → mxxMat11) × (Fin 8 → Fin 32 → mxxMat11) × (Fin 8 → Fin 32 → mxxMat11) × (Fin 8 → Fin 128 → mxxMat12) × (Fin 8 → Fin 128 → mxxMat12) × (Fin 8 → Fin 32 → mxxMat12) × (Fin 8 → Fin 32 → mxxMat12) × (Fin 8 → Fin 128 → mxxMat13) × (Fin 8 → Fin 128 → mxxMat13) × (Fin 8 → Fin 32 → mxxMat13) × (Fin 8 → Fin 32 → mxxMat13) × (Fin 8 → Fin 128 → mxxMat14) × (Fin 8 → Fin 128 → mxxMat14) × (Fin 8 → Fin 32 → mxxMat14) × (Fin 8 → Fin 32 → mxxMat14) × (Fin 8 → Fin 128 → mxxMat15) × (Fin 8 → Fin 128 → mxxMat15) × (Fin 8 → Fin 32 → mxxMat15) × (Fin 8 → Fin 32 → mxxMat15) × (Fin 8 → Fin 128 → mxxMat16) × (Fin 8 → Fin 128 → mxxMat16) × (Fin 8 → Fin 32 → mxxMat16) × (Fin 8 → Fin 32 → mxxMat16) × (Fin 8 → Fin 128 → mxxMat17) × (Fin 8 → Fin 128 → mxxMat17) × (Fin 8 → Fin 32 → mxxMat17) × (Fin 8 → Fin 32 → mxxMat17) × (Fin 8 → Fin 128 → mxxMat18) × (Fin 8 → Fin 128 → mxxMat18) × (Fin 8 → Fin 32 → mxxMat18) × (Fin 8 → Fin 32 → mxxMat18) × (Fin 8 → Fin 128 → mxxMat19) × (Fin 8 → Fin 128 → mxxMat19) × (Fin 8 → Fin 32 → mxxMat19) × (Fin 8 → Fin 32 → mxxMat19) × (Fin 8 → Fin 128 → mxxMat20) × (Fin 8 → Fin 128 → mxxMat20) × (Fin 8 → Fin 32 → mxxMat20) × (Fin 8 → Fin 32 → mxxMat20) × (Fin 8 → Fin 128 → mxxMat21) × (Fin 8 → Fin 128 → mxxMat21) × (Fin 8 → Fin 32 → mxxMat21) × (Fin 8 → Fin 32 → mxxMat21) × (Fin 8 → Fin 128 → mxxMat22) × (Fin 8 → Fin 128 → mxxMat22) × (Fin 8 → Fin 32 → mxxMat22) × (Fin 8 → Fin 32 → mxxMat22) × (Fin 8 → Fin 128 → mxxMat23) × (Fin 8 → Fin 128 → mxxMat23) × (Fin 8 → Fin 32 → mxxMat23) × (Fin 8 → Fin 32 → mxxMat23) × (Fin 8 → Fin 128 → mxxMat24) × (Fin 8 → Fin 128 → mxxMat24) × (Fin 8 → Fin 32 → mxxMat24) × (Fin 8 → Fin 32 → mxxMat24) × (Fin 8 → Fin 128 → mxxMat25) × (Fin 8 → Fin 128 → mxxMat25) × (Fin 8 → Fin 32 → mxxMat25) × (Fin 8 → Fin 32 → mxxMat25) × (Fin 8 → Fin 128 → mxxMat26) × (Fin 8 → Fin 128 → mxxMat26) × (Fin 8 → Fin 32 → mxxMat26) × (Fin 8 → Fin 32 → mxxMat26) × (Fin 8 → Fin 128 → mxxMat27) × (Fin 8 → Fin 128 → mxxMat27) × (Fin 8 → Fin 32 → mxxMat27) × (Fin 8 → Fin 32 → mxxMat27) × (Fin 8 → Fin 128 → mxxMat28) × (Fin 8 → Fin 128 → mxxMat28) × (Fin 8 → Fin 32 → mxxMat28) × (Fin 8 → Fin 32 → mxxMat28) × (Fin 8 → Fin 128 → mxxMat29) × (Fin 8 → Fin 128 → mxxMat29) × (Fin 8 → Fin 32 → mxxMat29) × (Fin 8 → Fin 32 → mxxMat29) × (Fin 8 → Fin 128 → mxxMat30) × (Fin 8 → Fin 128 → mxxMat30) × (Fin 8 → Fin 32 → mxxMat30) × (Fin 8 → Fin 32 → mxxMat30) × (Fin 8 → Fin 128 → mxxMat31) × (Fin 8 → Fin 128 → mxxMat31) × (Fin 8 → Fin 32 → mxxMat31) × (Fin 8 → Fin 32 → mxxMat31) × (Fin 8 → Fin 128 → mxxMat32) × (Fin 8 → Fin 128 → mxxMat32) × (Fin 8 → Fin 32 → mxxMat32) × (Fin 8 → Fin 32 → mxxMat32) × (Fin 8 → Fin 128 → mxxMat33) × (Fin 8 → Fin 128 → mxxMat33) × (Fin 8 → Fin 32 → mxxMat33) × (Fin 8 → Fin 32 → mxxMat33) × (Fin 8 → Fin 128 → mxxMat34) × (Fin 8 → Fin 128 → mxxMat34) × (Fin 8 → Fin 32 → mxxMat34) × (Fin 8 → Fin 32 → mxxMat34) × (Fin 8 → Fin 128 → mxxMat35) × (Fin 8 → Fin 128 → mxxMat35) × (Fin 8 → Fin 32 → mxxMat35) × (Fin 8 → Fin 32 → mxxMat35) × (Fin 8 → Fin 128 → mxxMat36) × (Fin 8 → Fin 128 → mxxMat36) × (Fin 8 → Fin 32 → mxxMat36) × (Fin 8 → Fin 32 → mxxMat36) × (Fin 8 → Fin 128 → mxxMat37) × (Fin 8 → Fin 128 → mxxMat37) × (Fin 8 → Fin 32 → mxxMat37) × (Fin 8 → Fin 32 → mxxMat37) × (Fin 8 → Fin 128 → mxxMat38) × (Fin 8 → Fin 128 → mxxMat38) × (Fin 8 → Fin 32 → mxxMat38) × (Fin 8 → Fin 32 → mxxMat38) × (Fin 8 → Fin 128 → mxxMat39) × (Fin 8 → Fin 128 → mxxMat39) × (Fin 8 → Fin 32 → mxxMat39) × (Fin 8 → Fin 32 → mxxMat39) × Unit) : Prop where
  c0 : generatedRoot.Facts_0 ctx outputs

theorem generatedRoot.facts {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) {outputs : (Fin 8 → mxxMat2) × (Fin 8 → mxxMat2) × (Fin 8 → Fin 1025 → mxxMat4) × (Fin 8 → Fin 1025 → mxxMat4) × (Fin 8 → Fin 1025 → mxxMat1) × (Fin 8 → Fin 128 → mxxMat2) × (Fin 8 → Fin 128 → mxxMat2) × (Fin 8 → Fin 32 → mxxMat2) × (Fin 8 → Fin 32 → mxxMat2) × (Fin 8 → Fin 128 → mxxMat5) × (Fin 8 → Fin 128 → mxxMat5) × (Fin 8 → Fin 32 → mxxMat5) × (Fin 8 → Fin 32 → mxxMat5) × (Fin 8 → Fin 128 → mxxMat6) × (Fin 8 → Fin 128 → mxxMat6) × (Fin 8 → Fin 32 → mxxMat6) × (Fin 8 → Fin 32 → mxxMat6) × (Fin 8 → Fin 128 → mxxMat7) × (Fin 8 → Fin 128 → mxxMat7) × (Fin 8 → Fin 32 → mxxMat7) × (Fin 8 → Fin 32 → mxxMat7) × (Fin 8 → Fin 128 → mxxMat8) × (Fin 8 → Fin 128 → mxxMat8) × (Fin 8 → Fin 32 → mxxMat8) × (Fin 8 → Fin 32 → mxxMat8) × (Fin 8 → Fin 128 → mxxMat9) × (Fin 8 → Fin 128 → mxxMat9) × (Fin 8 → Fin 32 → mxxMat9) × (Fin 8 → Fin 32 → mxxMat9) × (Fin 8 → Fin 128 → mxxMat10) × (Fin 8 → Fin 128 → mxxMat10) × (Fin 8 → Fin 32 → mxxMat10) × (Fin 8 → Fin 32 → mxxMat10) × (Fin 8 → Fin 128 → mxxMat11) × (Fin 8 → Fin 128 → mxxMat11) × (Fin 8 → Fin 32 → mxxMat11) × (Fin 8 → Fin 32 → mxxMat11) × (Fin 8 → Fin 128 → mxxMat12) × (Fin 8 → Fin 128 → mxxMat12) × (Fin 8 → Fin 32 → mxxMat12) × (Fin 8 → Fin 32 → mxxMat12) × (Fin 8 → Fin 128 → mxxMat13) × (Fin 8 → Fin 128 → mxxMat13) × (Fin 8 → Fin 32 → mxxMat13) × (Fin 8 → Fin 32 → mxxMat13) × (Fin 8 → Fin 128 → mxxMat14) × (Fin 8 → Fin 128 → mxxMat14) × (Fin 8 → Fin 32 → mxxMat14) × (Fin 8 → Fin 32 → mxxMat14) × (Fin 8 → Fin 128 → mxxMat15) × (Fin 8 → Fin 128 → mxxMat15) × (Fin 8 → Fin 32 → mxxMat15) × (Fin 8 → Fin 32 → mxxMat15) × (Fin 8 → Fin 128 → mxxMat16) × (Fin 8 → Fin 128 → mxxMat16) × (Fin 8 → Fin 32 → mxxMat16) × (Fin 8 → Fin 32 → mxxMat16) × (Fin 8 → Fin 128 → mxxMat17) × (Fin 8 → Fin 128 → mxxMat17) × (Fin 8 → Fin 32 → mxxMat17) × (Fin 8 → Fin 32 → mxxMat17) × (Fin 8 → Fin 128 → mxxMat18) × (Fin 8 → Fin 128 → mxxMat18) × (Fin 8 → Fin 32 → mxxMat18) × (Fin 8 → Fin 32 → mxxMat18) × (Fin 8 → Fin 128 → mxxMat19) × (Fin 8 → Fin 128 → mxxMat19) × (Fin 8 → Fin 32 → mxxMat19) × (Fin 8 → Fin 32 → mxxMat19) × (Fin 8 → Fin 128 → mxxMat20) × (Fin 8 → Fin 128 → mxxMat20) × (Fin 8 → Fin 32 → mxxMat20) × (Fin 8 → Fin 32 → mxxMat20) × (Fin 8 → Fin 128 → mxxMat21) × (Fin 8 → Fin 128 → mxxMat21) × (Fin 8 → Fin 32 → mxxMat21) × (Fin 8 → Fin 32 → mxxMat21) × (Fin 8 → Fin 128 → mxxMat22) × (Fin 8 → Fin 128 → mxxMat22) × (Fin 8 → Fin 32 → mxxMat22) × (Fin 8 → Fin 32 → mxxMat22) × (Fin 8 → Fin 128 → mxxMat23) × (Fin 8 → Fin 128 → mxxMat23) × (Fin 8 → Fin 32 → mxxMat23) × (Fin 8 → Fin 32 → mxxMat23) × (Fin 8 → Fin 128 → mxxMat24) × (Fin 8 → Fin 128 → mxxMat24) × (Fin 8 → Fin 32 → mxxMat24) × (Fin 8 → Fin 32 → mxxMat24) × (Fin 8 → Fin 128 → mxxMat25) × (Fin 8 → Fin 128 → mxxMat25) × (Fin 8 → Fin 32 → mxxMat25) × (Fin 8 → Fin 32 → mxxMat25) × (Fin 8 → Fin 128 → mxxMat26) × (Fin 8 → Fin 128 → mxxMat26) × (Fin 8 → Fin 32 → mxxMat26) × (Fin 8 → Fin 32 → mxxMat26) × (Fin 8 → Fin 128 → mxxMat27) × (Fin 8 → Fin 128 → mxxMat27) × (Fin 8 → Fin 32 → mxxMat27) × (Fin 8 → Fin 32 → mxxMat27) × (Fin 8 → Fin 128 → mxxMat28) × (Fin 8 → Fin 128 → mxxMat28) × (Fin 8 → Fin 32 → mxxMat28) × (Fin 8 → Fin 32 → mxxMat28) × (Fin 8 → Fin 128 → mxxMat29) × (Fin 8 → Fin 128 → mxxMat29) × (Fin 8 → Fin 32 → mxxMat29) × (Fin 8 → Fin 32 → mxxMat29) × (Fin 8 → Fin 128 → mxxMat30) × (Fin 8 → Fin 128 → mxxMat30) × (Fin 8 → Fin 32 → mxxMat30) × (Fin 8 → Fin 32 → mxxMat30) × (Fin 8 → Fin 128 → mxxMat31) × (Fin 8 → Fin 128 → mxxMat31) × (Fin 8 → Fin 32 → mxxMat31) × (Fin 8 → Fin 32 → mxxMat31) × (Fin 8 → Fin 128 → mxxMat32) × (Fin 8 → Fin 128 → mxxMat32) × (Fin 8 → Fin 32 → mxxMat32) × (Fin 8 → Fin 32 → mxxMat32) × (Fin 8 → Fin 128 → mxxMat33) × (Fin 8 → Fin 128 → mxxMat33) × (Fin 8 → Fin 32 → mxxMat33) × (Fin 8 → Fin 32 → mxxMat33) × (Fin 8 → Fin 128 → mxxMat34) × (Fin 8 → Fin 128 → mxxMat34) × (Fin 8 → Fin 32 → mxxMat34) × (Fin 8 → Fin 32 → mxxMat34) × (Fin 8 → Fin 128 → mxxMat35) × (Fin 8 → Fin 128 → mxxMat35) × (Fin 8 → Fin 32 → mxxMat35) × (Fin 8 → Fin 32 → mxxMat35) × (Fin 8 → Fin 128 → mxxMat36) × (Fin 8 → Fin 128 → mxxMat36) × (Fin 8 → Fin 32 → mxxMat36) × (Fin 8 → Fin 32 → mxxMat36) × (Fin 8 → Fin 128 → mxxMat37) × (Fin 8 → Fin 128 → mxxMat37) × (Fin 8 → Fin 32 → mxxMat37) × (Fin 8 → Fin 32 → mxxMat37) × (Fin 8 → Fin 128 → mxxMat38) × (Fin 8 → Fin 128 → mxxMat38) × (Fin 8 → Fin 32 → mxxMat38) × (Fin 8 → Fin 32 → mxxMat38) × (Fin 8 → Fin 128 → mxxMat39) × (Fin 8 → Fin 128 → mxxMat39) × (Fin 8 → Fin 32 → mxxMat39) × (Fin 8 → Fin 32 → mxxMat39) × Unit}
    (h : generatedRoot.body backend hashModel tape path params ctx.inputs outputs ctx.witness) :
    generatedRoot.Facts ctx outputs :=
  have t0 := h
  have r0 := generatedRoot.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

theorem generatedRoot.h7_148_at {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {ctx : generatedRoot.Ctx backend hashModel tape path params} {outputs : (Fin 8 → mxxMat2) × (Fin 8 → mxxMat2) × (Fin 8 → Fin 1025 → mxxMat4) × (Fin 8 → Fin 1025 → mxxMat4) × (Fin 8 → Fin 1025 → mxxMat1) × (Fin 8 → Fin 128 → mxxMat2) × (Fin 8 → Fin 128 → mxxMat2) × (Fin 8 → Fin 32 → mxxMat2) × (Fin 8 → Fin 32 → mxxMat2) × (Fin 8 → Fin 128 → mxxMat5) × (Fin 8 → Fin 128 → mxxMat5) × (Fin 8 → Fin 32 → mxxMat5) × (Fin 8 → Fin 32 → mxxMat5) × (Fin 8 → Fin 128 → mxxMat6) × (Fin 8 → Fin 128 → mxxMat6) × (Fin 8 → Fin 32 → mxxMat6) × (Fin 8 → Fin 32 → mxxMat6) × (Fin 8 → Fin 128 → mxxMat7) × (Fin 8 → Fin 128 → mxxMat7) × (Fin 8 → Fin 32 → mxxMat7) × (Fin 8 → Fin 32 → mxxMat7) × (Fin 8 → Fin 128 → mxxMat8) × (Fin 8 → Fin 128 → mxxMat8) × (Fin 8 → Fin 32 → mxxMat8) × (Fin 8 → Fin 32 → mxxMat8) × (Fin 8 → Fin 128 → mxxMat9) × (Fin 8 → Fin 128 → mxxMat9) × (Fin 8 → Fin 32 → mxxMat9) × (Fin 8 → Fin 32 → mxxMat9) × (Fin 8 → Fin 128 → mxxMat10) × (Fin 8 → Fin 128 → mxxMat10) × (Fin 8 → Fin 32 → mxxMat10) × (Fin 8 → Fin 32 → mxxMat10) × (Fin 8 → Fin 128 → mxxMat11) × (Fin 8 → Fin 128 → mxxMat11) × (Fin 8 → Fin 32 → mxxMat11) × (Fin 8 → Fin 32 → mxxMat11) × (Fin 8 → Fin 128 → mxxMat12) × (Fin 8 → Fin 128 → mxxMat12) × (Fin 8 → Fin 32 → mxxMat12) × (Fin 8 → Fin 32 → mxxMat12) × (Fin 8 → Fin 128 → mxxMat13) × (Fin 8 → Fin 128 → mxxMat13) × (Fin 8 → Fin 32 → mxxMat13) × (Fin 8 → Fin 32 → mxxMat13) × (Fin 8 → Fin 128 → mxxMat14) × (Fin 8 → Fin 128 → mxxMat14) × (Fin 8 → Fin 32 → mxxMat14) × (Fin 8 → Fin 32 → mxxMat14) × (Fin 8 → Fin 128 → mxxMat15) × (Fin 8 → Fin 128 → mxxMat15) × (Fin 8 → Fin 32 → mxxMat15) × (Fin 8 → Fin 32 → mxxMat15) × (Fin 8 → Fin 128 → mxxMat16) × (Fin 8 → Fin 128 → mxxMat16) × (Fin 8 → Fin 32 → mxxMat16) × (Fin 8 → Fin 32 → mxxMat16) × (Fin 8 → Fin 128 → mxxMat17) × (Fin 8 → Fin 128 → mxxMat17) × (Fin 8 → Fin 32 → mxxMat17) × (Fin 8 → Fin 32 → mxxMat17) × (Fin 8 → Fin 128 → mxxMat18) × (Fin 8 → Fin 128 → mxxMat18) × (Fin 8 → Fin 32 → mxxMat18) × (Fin 8 → Fin 32 → mxxMat18) × (Fin 8 → Fin 128 → mxxMat19) × (Fin 8 → Fin 128 → mxxMat19) × (Fin 8 → Fin 32 → mxxMat19) × (Fin 8 → Fin 32 → mxxMat19) × (Fin 8 → Fin 128 → mxxMat20) × (Fin 8 → Fin 128 → mxxMat20) × (Fin 8 → Fin 32 → mxxMat20) × (Fin 8 → Fin 32 → mxxMat20) × (Fin 8 → Fin 128 → mxxMat21) × (Fin 8 → Fin 128 → mxxMat21) × (Fin 8 → Fin 32 → mxxMat21) × (Fin 8 → Fin 32 → mxxMat21) × (Fin 8 → Fin 128 → mxxMat22) × (Fin 8 → Fin 128 → mxxMat22) × (Fin 8 → Fin 32 → mxxMat22) × (Fin 8 → Fin 32 → mxxMat22) × (Fin 8 → Fin 128 → mxxMat23) × (Fin 8 → Fin 128 → mxxMat23) × (Fin 8 → Fin 32 → mxxMat23) × (Fin 8 → Fin 32 → mxxMat23) × (Fin 8 → Fin 128 → mxxMat24) × (Fin 8 → Fin 128 → mxxMat24) × (Fin 8 → Fin 32 → mxxMat24) × (Fin 8 → Fin 32 → mxxMat24) × (Fin 8 → Fin 128 → mxxMat25) × (Fin 8 → Fin 128 → mxxMat25) × (Fin 8 → Fin 32 → mxxMat25) × (Fin 8 → Fin 32 → mxxMat25) × (Fin 8 → Fin 128 → mxxMat26) × (Fin 8 → Fin 128 → mxxMat26) × (Fin 8 → Fin 32 → mxxMat26) × (Fin 8 → Fin 32 → mxxMat26) × (Fin 8 → Fin 128 → mxxMat27) × (Fin 8 → Fin 128 → mxxMat27) × (Fin 8 → Fin 32 → mxxMat27) × (Fin 8 → Fin 32 → mxxMat27) × (Fin 8 → Fin 128 → mxxMat28) × (Fin 8 → Fin 128 → mxxMat28) × (Fin 8 → Fin 32 → mxxMat28) × (Fin 8 → Fin 32 → mxxMat28) × (Fin 8 → Fin 128 → mxxMat29) × (Fin 8 → Fin 128 → mxxMat29) × (Fin 8 → Fin 32 → mxxMat29) × (Fin 8 → Fin 32 → mxxMat29) × (Fin 8 → Fin 128 → mxxMat30) × (Fin 8 → Fin 128 → mxxMat30) × (Fin 8 → Fin 32 → mxxMat30) × (Fin 8 → Fin 32 → mxxMat30) × (Fin 8 → Fin 128 → mxxMat31) × (Fin 8 → Fin 128 → mxxMat31) × (Fin 8 → Fin 32 → mxxMat31) × (Fin 8 → Fin 32 → mxxMat31) × (Fin 8 → Fin 128 → mxxMat32) × (Fin 8 → Fin 128 → mxxMat32) × (Fin 8 → Fin 32 → mxxMat32) × (Fin 8 → Fin 32 → mxxMat32) × (Fin 8 → Fin 128 → mxxMat33) × (Fin 8 → Fin 128 → mxxMat33) × (Fin 8 → Fin 32 → mxxMat33) × (Fin 8 → Fin 32 → mxxMat33) × (Fin 8 → Fin 128 → mxxMat34) × (Fin 8 → Fin 128 → mxxMat34) × (Fin 8 → Fin 32 → mxxMat34) × (Fin 8 → Fin 32 → mxxMat34) × (Fin 8 → Fin 128 → mxxMat35) × (Fin 8 → Fin 128 → mxxMat35) × (Fin 8 → Fin 32 → mxxMat35) × (Fin 8 → Fin 32 → mxxMat35) × (Fin 8 → Fin 128 → mxxMat36) × (Fin 8 → Fin 128 → mxxMat36) × (Fin 8 → Fin 32 → mxxMat36) × (Fin 8 → Fin 32 → mxxMat36) × (Fin 8 → Fin 128 → mxxMat37) × (Fin 8 → Fin 128 → mxxMat37) × (Fin 8 → Fin 32 → mxxMat37) × (Fin 8 → Fin 32 → mxxMat37) × (Fin 8 → Fin 128 → mxxMat38) × (Fin 8 → Fin 128 → mxxMat38) × (Fin 8 → Fin 32 → mxxMat38) × (Fin 8 → Fin 32 → mxxMat38) × (Fin 8 → Fin 128 → mxxMat39) × (Fin 8 → Fin 128 → mxxMat39) × (Fin 8 → Fin 32 → mxxMat39) × (Fin 8 → Fin 32 → mxxMat39) × Unit}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 8) :
    ∃ w : parallel_generatedRoot_7.Witness, parallel_generatedRoot_7.Facts (⟨((generatedRoot.w_0_0 ctx), (((generatedRoot.w_1_0 ctx) i), ((generatedRoot.w_2_0 ctx), (((generatedRoot.w_3_0 ctx) i), (((generatedRoot.w_4_0 ctx) i), (((generatedRoot.w_5_0 ctx) i), (((generatedRoot.w_6_0 ctx) i), ()))))))), w⟩ : parallel_generatedRoot_7.Ctx backend hashModel tape (path ++ [7, i.val]) params i) (((generatedRoot.w_7_0 ctx) i), (((generatedRoot.w_7_1 ctx) i), (((generatedRoot.w_7_2 ctx) i), (((generatedRoot.w_7_3 ctx) i), (((generatedRoot.w_7_4 ctx) i), (((generatedRoot.w_7_5 ctx) i), (((generatedRoot.w_7_6 ctx) i), (((generatedRoot.w_7_7 ctx) i), (((generatedRoot.w_7_8 ctx) i), (((generatedRoot.w_7_9 ctx) i), (((generatedRoot.w_7_10 ctx) i), (((generatedRoot.w_7_11 ctx) i), (((generatedRoot.w_7_12 ctx) i), (((generatedRoot.w_7_13 ctx) i), (((generatedRoot.w_7_14 ctx) i), (((generatedRoot.w_7_15 ctx) i), (((generatedRoot.w_7_16 ctx) i), (((generatedRoot.w_7_17 ctx) i), (((generatedRoot.w_7_18 ctx) i), (((generatedRoot.w_7_19 ctx) i), (((generatedRoot.w_7_20 ctx) i), (((generatedRoot.w_7_21 ctx) i), (((generatedRoot.w_7_22 ctx) i), (((generatedRoot.w_7_23 ctx) i), (((generatedRoot.w_7_24 ctx) i), (((generatedRoot.w_7_25 ctx) i), (((generatedRoot.w_7_26 ctx) i), (((generatedRoot.w_7_27 ctx) i), (((generatedRoot.w_7_28 ctx) i), (((generatedRoot.w_7_29 ctx) i), (((generatedRoot.w_7_30 ctx) i), (((generatedRoot.w_7_31 ctx) i), (((generatedRoot.w_7_32 ctx) i), (((generatedRoot.w_7_33 ctx) i), (((generatedRoot.w_7_34 ctx) i), (((generatedRoot.w_7_35 ctx) i), (((generatedRoot.w_7_36 ctx) i), (((generatedRoot.w_7_37 ctx) i), (((generatedRoot.w_7_38 ctx) i), (((generatedRoot.w_7_39 ctx) i), (((generatedRoot.w_7_40 ctx) i), (((generatedRoot.w_7_41 ctx) i), (((generatedRoot.w_7_42 ctx) i), (((generatedRoot.w_7_43 ctx) i), (((generatedRoot.w_7_44 ctx) i), (((generatedRoot.w_7_45 ctx) i), (((generatedRoot.w_7_46 ctx) i), (((generatedRoot.w_7_47 ctx) i), (((generatedRoot.w_7_48 ctx) i), (((generatedRoot.w_7_49 ctx) i), (((generatedRoot.w_7_50 ctx) i), (((generatedRoot.w_7_51 ctx) i), (((generatedRoot.w_7_52 ctx) i), (((generatedRoot.w_7_53 ctx) i), (((generatedRoot.w_7_54 ctx) i), (((generatedRoot.w_7_55 ctx) i), (((generatedRoot.w_7_56 ctx) i), (((generatedRoot.w_7_57 ctx) i), (((generatedRoot.w_7_58 ctx) i), (((generatedRoot.w_7_59 ctx) i), (((generatedRoot.w_7_60 ctx) i), (((generatedRoot.w_7_61 ctx) i), (((generatedRoot.w_7_62 ctx) i), (((generatedRoot.w_7_63 ctx) i), (((generatedRoot.w_7_64 ctx) i), (((generatedRoot.w_7_65 ctx) i), (((generatedRoot.w_7_66 ctx) i), (((generatedRoot.w_7_67 ctx) i), (((generatedRoot.w_7_68 ctx) i), (((generatedRoot.w_7_69 ctx) i), (((generatedRoot.w_7_70 ctx) i), (((generatedRoot.w_7_71 ctx) i), (((generatedRoot.w_7_72 ctx) i), (((generatedRoot.w_7_73 ctx) i), (((generatedRoot.w_7_74 ctx) i), (((generatedRoot.w_7_75 ctx) i), (((generatedRoot.w_7_76 ctx) i), (((generatedRoot.w_7_77 ctx) i), (((generatedRoot.w_7_78 ctx) i), (((generatedRoot.w_7_79 ctx) i), (((generatedRoot.w_7_80 ctx) i), (((generatedRoot.w_7_81 ctx) i), (((generatedRoot.w_7_82 ctx) i), (((generatedRoot.w_7_83 ctx) i), (((generatedRoot.w_7_84 ctx) i), (((generatedRoot.w_7_85 ctx) i), (((generatedRoot.w_7_86 ctx) i), (((generatedRoot.w_7_87 ctx) i), (((generatedRoot.w_7_88 ctx) i), (((generatedRoot.w_7_89 ctx) i), (((generatedRoot.w_7_90 ctx) i), (((generatedRoot.w_7_91 ctx) i), (((generatedRoot.w_7_92 ctx) i), (((generatedRoot.w_7_93 ctx) i), (((generatedRoot.w_7_94 ctx) i), (((generatedRoot.w_7_95 ctx) i), (((generatedRoot.w_7_96 ctx) i), (((generatedRoot.w_7_97 ctx) i), (((generatedRoot.w_7_98 ctx) i), (((generatedRoot.w_7_99 ctx) i), (((generatedRoot.w_7_100 ctx) i), (((generatedRoot.w_7_101 ctx) i), (((generatedRoot.w_7_102 ctx) i), (((generatedRoot.w_7_103 ctx) i), (((generatedRoot.w_7_104 ctx) i), (((generatedRoot.w_7_105 ctx) i), (((generatedRoot.w_7_106 ctx) i), (((generatedRoot.w_7_107 ctx) i), (((generatedRoot.w_7_108 ctx) i), (((generatedRoot.w_7_109 ctx) i), (((generatedRoot.w_7_110 ctx) i), (((generatedRoot.w_7_111 ctx) i), (((generatedRoot.w_7_112 ctx) i), (((generatedRoot.w_7_113 ctx) i), (((generatedRoot.w_7_114 ctx) i), (((generatedRoot.w_7_115 ctx) i), (((generatedRoot.w_7_116 ctx) i), (((generatedRoot.w_7_117 ctx) i), (((generatedRoot.w_7_118 ctx) i), (((generatedRoot.w_7_119 ctx) i), (((generatedRoot.w_7_120 ctx) i), (((generatedRoot.w_7_121 ctx) i), (((generatedRoot.w_7_122 ctx) i), (((generatedRoot.w_7_123 ctx) i), (((generatedRoot.w_7_124 ctx) i), (((generatedRoot.w_7_125 ctx) i), (((generatedRoot.w_7_126 ctx) i), (((generatedRoot.w_7_127 ctx) i), (((generatedRoot.w_7_128 ctx) i), (((generatedRoot.w_7_129 ctx) i), (((generatedRoot.w_7_130 ctx) i), (((generatedRoot.w_7_131 ctx) i), (((generatedRoot.w_7_132 ctx) i), (((generatedRoot.w_7_133 ctx) i), (((generatedRoot.w_7_134 ctx) i), (((generatedRoot.w_7_135 ctx) i), (((generatedRoot.w_7_136 ctx) i), (((generatedRoot.w_7_137 ctx) i), (((generatedRoot.w_7_138 ctx) i), (((generatedRoot.w_7_139 ctx) i), (((generatedRoot.w_7_140 ctx) i), (((generatedRoot.w_7_141 ctx) i), (((generatedRoot.w_7_142 ctx) i), (((generatedRoot.w_7_143 ctx) i), (((generatedRoot.w_7_144 ctx) i), (((generatedRoot.w_7_145 ctx) i), (((generatedRoot.w_7_146 ctx) i), (((generatedRoot.w_7_147 ctx) i), (((generatedRoot.w_7_148 ctx) i), ()))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))) :=
  let ⟨w, h⟩ := F.c0.h7_148 i
  ⟨w, parallel_generatedRoot_7.facts _ h⟩

end Stage_sigma_encrypt

end
