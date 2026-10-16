import Stage_decode

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000
set_option linter.unusedVariables false

-- The wire types repeat the ring modulus `X ^ n + 1`; unfolding it while checking a
-- projection against its field type expands it at the ring dimension.
attribute [local irreducible] Mxx.Primitives.negacyclicModulus

noncomputable section

namespace Stage_decode

local notation "mxxMat0" => Mxx.Primitives.ExactMatrix modulus_0 32768 1 1
local notation "mxxMat1" => Mxx.Primitives.ExactMatrix modulus_0 32768 1 30
local notation "mxxMat2" => Mxx.Primitives.ExactMatrix modulus_0 32768 1 60
local notation "mxxMat3" => Mxx.Primitives.ExactMatrix modulus_0 32768 60 45
local notation "mxxMat4" => Mxx.Primitives.ExactMatrix modulus_1 32768 1 295
local notation "mxxMat5" => Mxx.Primitives.ExactMatrix modulus_0 32768 295 1
local notation "mxxMat6" => Mxx.Primitives.ExactMatrix modulus_0 32768 30 1
local notation "mxxMat7" => Mxx.Primitives.ExactMatrix modulus_0 32768 60 1
local notation "mxxMat8" => Mxx.Primitives.ExactMatrix modulus_0 32768 1 295
local notation "mxxMat9" => Mxx.Primitives.ExactMatrix modulus_1 32768 1 1
local notation "mxxMat10" => Mxx.Primitives.ExactMatrix modulus_1 32768 1 30

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_37.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) where
  inputs : mxxMat0 × (Fin 45 → mxxMat1) × (Fin 45 → mxxMat1) × ByteArray × mxxMat2 × (Fin 256 → mxxMat3) × Unit
  witness : sequential_generatedRoot_37.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_37Ctx" => (sequential_generatedRoot_37.Ctx backend hashModel tape path params i_0)

def sequential_generatedRoot_37.w_0_0  (ctx : sequential_generatedRoot_37Ctx) : mxxMat0 :=
  ctx.inputs.1

def sequential_generatedRoot_37.w_1_0  (ctx : sequential_generatedRoot_37Ctx) : Fin 45 → mxxMat1 :=
  ctx.inputs.2.1

def sequential_generatedRoot_37.w_2_0  (ctx : sequential_generatedRoot_37Ctx) : Fin 45 → mxxMat1 :=
  ctx.inputs.2.2.1

def sequential_generatedRoot_37.w_3_0  (ctx : sequential_generatedRoot_37Ctx) : ByteArray :=
  ctx.inputs.2.2.2.1

def sequential_generatedRoot_37.w_4_0  (ctx : sequential_generatedRoot_37Ctx) : mxxMat2 :=
  ctx.inputs.2.2.2.2.1

def sequential_generatedRoot_37.w_5_0  (ctx : sequential_generatedRoot_37Ctx) : Fin 256 → mxxMat3 :=
  ctx.inputs.2.2.2.2.2.1

def sequential_generatedRoot_37.w_6_0  (ctx : sequential_generatedRoot_37Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_37.w_7_0  (ctx : sequential_generatedRoot_37Ctx) : mxxMat3 :=
  ctx.witness.w_7_0

def sequential_generatedRoot_37.w_8_0  (ctx : sequential_generatedRoot_37Ctx) : mxxMat0 :=
  ctx.witness.w_8_0

def sequential_generatedRoot_37.outs  (ctx : sequential_generatedRoot_37Ctx) : mxxMat0 :=
  (sequential_generatedRoot_37.w_8_0 ctx)

structure sequential_generatedRoot_37.Facts_0  (ctx : sequential_generatedRoot_37Ctx) (outputs : mxxMat0) : Prop where
  h6_0 : 0 ≤ (sequential_generatedRoot_37.w_6_0 ctx)
  h6_0_1 : (sequential_generatedRoot_37.w_6_0 ctx) < 256
  h7_0 : MxxRuntime.familyGetDynamic (sequential_generatedRoot_37.w_5_0 ctx) (sequential_generatedRoot_37.w_6_0 ctx) (sequential_generatedRoot_37.w_7_0 ctx)
  t3 : 0 ≤ (45)
  h8_0 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0) => sequential_sequential_generatedRoot_37_8 backend hashModel tape (path ++ [8, i]) params i_0 i (current, ((sequential_generatedRoot_37.w_1_0 ctx), ((sequential_generatedRoot_37.w_2_0 ctx), ((sequential_generatedRoot_37.w_3_0 ctx), ((sequential_generatedRoot_37.w_4_0 ctx), ((sequential_generatedRoot_37.w_7_0 ctx), ())))))) next) (Int.toNat (45)) (sequential_generatedRoot_37.w_0_0 ctx) (sequential_generatedRoot_37.w_8_0 ctx)
  hout : outputs = sequential_generatedRoot_37.outs ctx

theorem sequential_generatedRoot_37.facts_0  {ctx : sequential_generatedRoot_37Ctx} {outputs : mxxMat0}
    (h : sequential_generatedRoot_37.constraints_0 backend hashModel tape path params i_0 (sequential_generatedRoot_37.w_0_0 ctx) (sequential_generatedRoot_37.w_1_0 ctx) (sequential_generatedRoot_37.w_2_0 ctx) (sequential_generatedRoot_37.w_3_0 ctx) (sequential_generatedRoot_37.w_4_0 ctx) (sequential_generatedRoot_37.w_5_0 ctx) (sequential_generatedRoot_37.w_6_0 ctx) (sequential_generatedRoot_37.w_7_0 ctx) (sequential_generatedRoot_37.w_8_0 ctx) outputs) :
    sequential_generatedRoot_37.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2⟩

structure sequential_generatedRoot_37.Facts  (ctx : sequential_generatedRoot_37Ctx) (outputs : mxxMat0) : Prop where
  c0 : sequential_generatedRoot_37.Facts_0 ctx outputs

theorem sequential_generatedRoot_37.facts  (ctx : sequential_generatedRoot_37Ctx) {outputs : mxxMat0}
    (h : sequential_generatedRoot_37.body backend hashModel tape path params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_37.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_37.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) where
  inputs : (generatedRoot.InputType)
  witness : generatedRoot.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params}
local notation "generatedRootCtx" => (generatedRoot.Ctx backend hashModel tape path params)

def generatedRoot.w_0_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.inputs.1

def generatedRoot.w_1_0  (ctx : generatedRootCtx) : ByteArray :=
  ctx.inputs.2.1

def generatedRoot.w_5_0  (ctx : generatedRootCtx) : mxxMat4 :=
  ctx.inputs.2.2.1

def generatedRoot.w_10_0  (ctx : generatedRootCtx) : Fin 256 → mxxMat5 :=
  ctx.inputs.2.2.2.1

def generatedRoot.w_27_0  (ctx : generatedRootCtx) : Fin 256 → mxxMat6 :=
  ctx.inputs.2.2.2.2.1

def generatedRoot.w_31_0  (ctx : generatedRootCtx) : Fin 45 → mxxMat1 :=
  ctx.inputs.2.2.2.2.2.1

def generatedRoot.w_32_0  (ctx : generatedRootCtx) : Fin 45 → mxxMat1 :=
  ctx.inputs.2.2.2.2.2.2.1

def generatedRoot.w_36_0  (ctx : generatedRootCtx) : Fin 256 → mxxMat3 :=
  ctx.inputs.2.2.2.2.2.2.2.1

def generatedRoot.w_39_0  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.inputs.2.2.2.2.2.2.2.2.1

def generatedRoot.w_40_0  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_42_0  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_43_0  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_46_0  (ctx : generatedRootCtx) : mxxMat7 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_2_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_2_0

def generatedRoot.w_3_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixSub (generatedRoot.w_0_0 ctx) (generatedRoot.w_2_0 ctx)

def generatedRoot.w_4_0  (ctx : generatedRootCtx) : mxxMat0 :=
  (0 : mxxMat0)

def generatedRoot.w_6_0  (ctx : generatedRootCtx) : mxxMat8 :=
  ctx.witness.w_6_0

def generatedRoot.w_7_0  (ctx : generatedRootCtx) : mxxMat4 :=
  ctx.witness.w_7_0

def generatedRoot.w_8_0  (ctx : generatedRootCtx) : mxxMat8 :=
  ctx.witness.w_8_0

def generatedRoot.w_9_0  (ctx : generatedRootCtx) : mxxMat8 :=
  MxxRuntime.matrixSub (generatedRoot.w_6_0 ctx) (generatedRoot.w_8_0 ctx)

def generatedRoot.w_11_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_11_0

def generatedRoot.w_12_0  (ctx : generatedRootCtx) : mxxMat0 :=
  (0 : mxxMat0)

def generatedRoot.w_13_0  (ctx : generatedRootCtx) : mxxMat9 :=
  ctx.witness.w_13_0

def generatedRoot.w_14_0  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.witness.w_14_0

def generatedRoot.w_15_0  (ctx : generatedRootCtx) : mxxMat10 :=
  ctx.witness.w_15_0

def generatedRoot.w_16_0  (ctx : generatedRootCtx) : mxxMat10 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_13_0 ctx) (generatedRoot.w_15_0 ctx)

def generatedRoot.w_17_0  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.witness.w_17_0

def generatedRoot.w_18_0  (ctx : generatedRootCtx) : mxxMat1 :=
  (generatedRoot.w_17_0 ctx)

def generatedRoot.w_19_0  (ctx : generatedRootCtx) : mxxMat4 :=
  ctx.witness.w_19_0

def generatedRoot.w_20_0  (ctx : generatedRootCtx) : mxxMat9 :=
  ctx.witness.w_20_0

def generatedRoot.w_21_0  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.witness.w_21_0

def generatedRoot.w_22_0  (ctx : generatedRootCtx) : mxxMat10 :=
  ctx.witness.w_22_0

def generatedRoot.w_23_0  (ctx : generatedRootCtx) : mxxMat10 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_20_0 ctx) (generatedRoot.w_22_0 ctx)

def generatedRoot.w_24_0  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.witness.w_24_0

def generatedRoot.w_25_0  (ctx : generatedRootCtx) : mxxMat1 :=
  (generatedRoot.w_24_0 ctx)

def generatedRoot.w_26_0  (ctx : generatedRootCtx) : mxxMat1 :=
  MxxRuntime.matrixSub (generatedRoot.w_18_0 ctx) (generatedRoot.w_25_0 ctx)

def generatedRoot.w_28_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_28_0

def generatedRoot.w_29_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_11_0 ctx) (generatedRoot.w_28_0 ctx)

def generatedRoot.w_30_0  (ctx : generatedRootCtx) : mxxMat0 :=
  (0 : mxxMat0)

def generatedRoot.w_33_0  (ctx : generatedRootCtx) : mxxMat1 :=
  (0 : mxxMat1)

def generatedRoot.w_34_0  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.witness.w_34_0

def generatedRoot.w_35_0  (ctx : generatedRootCtx) : mxxMat2 :=
  ctx.witness.w_35_0

def generatedRoot.w_37_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_37_0

def generatedRoot.w_38_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_29_0 ctx) (generatedRoot.w_37_0 ctx)

def generatedRoot.w_41_0  (ctx : generatedRootCtx) : mxxMat2 :=
  ctx.witness.w_41_0

def generatedRoot.w_44_0  (ctx : generatedRootCtx) : mxxMat2 :=
  ctx.witness.w_44_0

def generatedRoot.w_45_0  (ctx : generatedRootCtx) : mxxMat2 :=
  MxxRuntime.matrixSub (generatedRoot.w_41_0 ctx) (generatedRoot.w_44_0 ctx)

def generatedRoot.w_47_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_45_0 ctx) (generatedRoot.w_46_0 ctx)

def generatedRoot.w_48_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_38_0 ctx) (generatedRoot.w_47_0 ctx)

def generatedRoot.w_49_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixSub (generatedRoot.w_3_0 ctx) (generatedRoot.w_48_0 ctx)

def generatedRoot.w_50_0  (ctx : generatedRootCtx) : Fin 32768 → Int :=
  ctx.witness.w_50_0

def generatedRoot.outs  (ctx : generatedRootCtx) : Fin 32768 → Int :=
  (generatedRoot.w_50_0 ctx)

structure generatedRoot.Facts_0  (ctx : generatedRootCtx) : Prop where
  h2_0 : MxxRuntime.hashSample (hashModel) ([109, 120, 120, 45, 112, 101]) ([.bytes [51, 52, 101, 99, 56, 49, 55, 48, 55, 98, 100, 100, 54, 99, 102, 98, 102, 99, 99, 52, 50, 51, 99, 55, 99, 98, 54, 97, 53, 51, 51, 100, 52, 57, 98, 102, 98, 101, 99, 56, 98, 57, 100, 97, 57, 101, 98, 98, 56, 48, 97, 52, 54, 57, 52, 51, 53, 53, 52, 48, 99, 101, 50, 48], .bytes [109, 101, 115, 115, 97, 103, 101, 45, 114, 111, 119], .u64Le (1)]) ((generatedRoot.w_1_0 ctx)) (generatedRoot.w_2_0 ctx)
  h6_0 : MxxRuntime.modulusSwitchRuns (generatedRoot.w_5_0 ctx) (generatedRoot.w_6_0 ctx)
  h7_0 : MxxRuntime.hashSample (hashModel) ([109, 120, 120, 45, 112, 101]) ([.bytes [51, 52, 101, 99, 56, 49, 55, 48, 55, 98, 100, 100, 54, 99, 102, 98, 102, 99, 99, 52, 50, 51, 99, 55, 99, 98, 54, 97, 53, 51, 51, 100, 52, 57, 98, 102, 98, 101, 99, 56, 98, 57, 100, 97, 57, 101, 98, 98, 56, 48, 97, 52, 54, 57, 52, 51, 53, 53, 52, 48, 99, 101, 50, 48], .bytes [97, 110, 99, 104, 111, 114, 45, 115, 101, 99, 111, 110, 100, 45, 114, 111, 119]]) ((generatedRoot.w_1_0 ctx)) (generatedRoot.w_7_0 ctx)
  h8_0 : MxxRuntime.modulusSwitchRuns (generatedRoot.w_7_0 ctx) (generatedRoot.w_8_0 ctx)
  t4 : 0 ≤ (256)
  h11_0 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0) => sequential_generatedRoot_11 backend hashModel tape (path ++ [11, i]) params i (current, ((generatedRoot.w_9_0 ctx), ((generatedRoot.w_10_0 ctx), ()))) next) (Int.toNat (256)) (generatedRoot.w_4_0 ctx) (generatedRoot.w_11_0 ctx)
  t6 : 0 ≤ 0
  t7 : 0 < 1
  t8 : 1 ≤ 1
  t9 : 0 ≤ 0
  t10 : 0 < 1
  t11 : 1 ≤ 295
  h13_0 : MxxRuntime.sliceMatrix (generatedRoot.w_5_0 ctx) 0 1 0 1 (generatedRoot.w_13_0 ctx)
  h14_0 : MxxRuntime.gadgetMatrixRuns backend 4096 30 (generatedRoot.w_14_0 ctx)
  h15_0 : MxxRuntime.centeredRebaseRuns (generatedRoot.w_14_0 ctx) (generatedRoot.w_15_0 ctx)
  h17_0 : MxxRuntime.modulusSwitchRuns (generatedRoot.w_16_0 ctx) (generatedRoot.w_17_0 ctx)
  h19_0 : MxxRuntime.hashSample (hashModel) ([109, 120, 120, 45, 112, 101]) ([.bytes [51, 52, 101, 99, 56, 49, 55, 48, 55, 98, 100, 100, 54, 99, 102, 98, 102, 99, 99, 52, 50, 51, 99, 55, 99, 98, 54, 97, 53, 51, 51, 100, 52, 57, 98, 102, 98, 101, 99, 56, 98, 57, 100, 97, 57, 101, 98, 98, 56, 48, 97, 52, 54, 57, 52, 51, 53, 53, 52, 48, 99, 101, 50, 48], .bytes [97, 110, 99, 104, 111, 114, 45, 115, 101, 99, 111, 110, 100, 45, 114, 111, 119]]) ((generatedRoot.w_1_0 ctx)) (generatedRoot.w_19_0 ctx)
  t17 : 0 ≤ 0
  t18 : 0 < 1
  t19 : 1 ≤ 1
  t20 : 0 ≤ 0
  t21 : 0 < 1
  t22 : 1 ≤ 295
  h20_0 : MxxRuntime.sliceMatrix (generatedRoot.w_19_0 ctx) 0 1 0 1 (generatedRoot.w_20_0 ctx)
  h21_0 : MxxRuntime.gadgetMatrixRuns backend 4096 30 (generatedRoot.w_21_0 ctx)
  h22_0 : MxxRuntime.centeredRebaseRuns (generatedRoot.w_21_0 ctx) (generatedRoot.w_22_0 ctx)
  h24_0 : MxxRuntime.modulusSwitchRuns (generatedRoot.w_23_0 ctx) (generatedRoot.w_24_0 ctx)
  t27 : 0 ≤ (256)
  h28_0 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0) => sequential_generatedRoot_28 backend hashModel tape (path ++ [28, i]) params i (current, ((generatedRoot.w_26_0 ctx), ((generatedRoot.w_27_0 ctx), ()))) next) (Int.toNat (256)) (generatedRoot.w_12_0 ctx) (generatedRoot.w_28_0 ctx)
  h34_0 : MxxRuntime.gadgetMatrixRuns backend 4096 30 (generatedRoot.w_34_0 ctx)
  h35_0 : MxxRuntime.concatColumns (generatedRoot.w_33_0 ctx) (generatedRoot.w_34_0 ctx) (generatedRoot.w_35_0 ctx)
  t31 : 0 ≤ (256)

theorem generatedRoot.facts_0  {ctx : generatedRootCtx} {outputs : Fin 32768 → Int}
    (h : generatedRoot.constraints_0 backend hashModel tape path params (generatedRoot.w_1_0 ctx) (generatedRoot.w_5_0 ctx) (generatedRoot.w_10_0 ctx) (generatedRoot.w_27_0 ctx) (generatedRoot.w_2_0 ctx) (generatedRoot.w_4_0 ctx) (generatedRoot.w_6_0 ctx) (generatedRoot.w_7_0 ctx) (generatedRoot.w_8_0 ctx) (generatedRoot.w_9_0 ctx) (generatedRoot.w_11_0 ctx) (generatedRoot.w_12_0 ctx) (generatedRoot.w_13_0 ctx) (generatedRoot.w_14_0 ctx) (generatedRoot.w_15_0 ctx) (generatedRoot.w_16_0 ctx) (generatedRoot.w_17_0 ctx) (generatedRoot.w_19_0 ctx) (generatedRoot.w_20_0 ctx) (generatedRoot.w_21_0 ctx) (generatedRoot.w_22_0 ctx) (generatedRoot.w_23_0 ctx) (generatedRoot.w_24_0 ctx) (generatedRoot.w_26_0 ctx) (generatedRoot.w_28_0 ctx) (generatedRoot.w_33_0 ctx) (generatedRoot.w_34_0 ctx) (generatedRoot.w_35_0 ctx)) :
    generatedRoot.Facts_0 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_1  (ctx : generatedRootCtx) (outputs : Fin 32768 → Int) : Prop where
  h37_0 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0) => sequential_generatedRoot_37 backend hashModel tape (path ++ [37, i]) params i (current, ((generatedRoot.w_31_0 ctx), ((generatedRoot.w_32_0 ctx), ((generatedRoot.w_1_0 ctx), ((generatedRoot.w_35_0 ctx), ((generatedRoot.w_36_0 ctx), ())))))) next) (Int.toNat (256)) (generatedRoot.w_30_0 ctx) (generatedRoot.w_37_0 ctx)
  h41_0 : MxxRuntime.concatColumns (generatedRoot.w_39_0 ctx) (generatedRoot.w_40_0 ctx) (generatedRoot.w_41_0 ctx)
  h44_0 : MxxRuntime.concatColumns (generatedRoot.w_42_0 ctx) (generatedRoot.w_43_0 ctx) (generatedRoot.w_44_0 ctx)
  t35 : (32768) = 32768
  h50_0 : (∀ k : Fin 32768, MxxRuntime.thresholdDecode (2) (32768) (k.val : Int) (generatedRoot.w_49_0 ctx) ((generatedRoot.w_50_0 ctx) k))
  hout : outputs = generatedRoot.outs ctx

theorem generatedRoot.facts_1  {ctx : generatedRootCtx} {outputs : Fin 32768 → Int}
    (h : generatedRoot.constraints_1 backend hashModel tape path params (generatedRoot.w_1_0 ctx) (generatedRoot.w_31_0 ctx) (generatedRoot.w_32_0 ctx) (generatedRoot.w_36_0 ctx) (generatedRoot.w_39_0 ctx) (generatedRoot.w_40_0 ctx) (generatedRoot.w_42_0 ctx) (generatedRoot.w_43_0 ctx) (generatedRoot.w_30_0 ctx) (generatedRoot.w_35_0 ctx) (generatedRoot.w_37_0 ctx) (generatedRoot.w_41_0 ctx) (generatedRoot.w_44_0 ctx) (generatedRoot.w_49_0 ctx) (generatedRoot.w_50_0 ctx) outputs) :
    generatedRoot.Facts_1 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2⟩

structure generatedRoot.Facts  (ctx : generatedRootCtx) (outputs : Fin 32768 → Int) : Prop where
  c0 : generatedRoot.Facts_0 ctx
  c1 : generatedRoot.Facts_1 ctx outputs

theorem generatedRoot.facts  (ctx : generatedRootCtx) {outputs : Fin 32768 → Int}
    (h : generatedRoot.body backend hashModel tape path params ctx.inputs outputs ctx.witness) :
    generatedRoot.Facts ctx outputs :=
  have t0 := h
  have r0 := generatedRoot.facts_0 (ctx := ctx) (outputs := outputs) t0.1
  have t1 := t0.2
  have r1 := generatedRoot.facts_1 (ctx := ctx) (outputs := outputs) t1
  ⟨r0, r1⟩

end

end Stage_decode

end
