import Stage_preprocess_0_0

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000
set_option linter.unusedVariables false

-- The wire types repeat the ring modulus `X ^ n + 1`; unfolding it while checking a
-- projection against its field type expands it at the ring dimension.
attribute [local irreducible] Mxx.Primitives.negacyclicModulus

noncomputable section

namespace Stage_preprocess_0_0

local notation "mxxNat0" => 1356938545745524308187737599098762797340022013559604731140326430414605958745
local notation "mxxNat1" => 1583094970036445026219027198948556596896692349152872186330380835483706951869
local notation "mxxMat0" => Mxx.Primitives.ExactMatrix modulus_0 8 1 1
local notation "mxxMat1" => Mxx.Primitives.ExactMatrix modulus_0 8 1 25
local notation "mxxMat2" => Mxx.Primitives.ExactMatrix modulus_0 8 1 112500
local notation "mxxMat3" => Mxx.Primitives.ExactMatrix modulus_0 8 1 4500
local notation "mxxMat4" => Mxx.Primitives.ExactMatrix modulus_0 8 1 22500
local notation "mxxMat5" => Mxx.Primitives.ExactMatrix modulus_0 8 1 900
local notation "mxxMat6" => Mxx.Primitives.ExactMatrix modulus_0 8 2 25
local notation "mxxMat7" => Mxx.Primitives.ExactMatrix modulus_0 8 25 25

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_3.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) where
  inputs : ByteArray × mxxMat0 × mxxMat0 × Unit
  witness : parallel_generatedRoot_3.Witness

def parallel_generatedRoot_3.w_0_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_3.Ctx backend hashModel params i_0) : ByteArray :=
  ctx.inputs.1

def parallel_generatedRoot_3.w_3_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_3.Ctx backend hashModel params i_0) : mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_3.w_26_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_3.Ctx backend hashModel params i_0) : mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_3.w_1_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_3.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_1_0

def parallel_generatedRoot_3.w_2_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_3.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_2_0

def parallel_generatedRoot_3.w_4_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_3.Ctx backend hashModel params i_0) : mxxMat0 :=
  ctx.witness.w_4_0

def parallel_generatedRoot_3.w_5_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_3.Ctx backend hashModel params i_0) : mxxMat0 :=
  MxxRuntime.matrixNeg (parallel_generatedRoot_3.w_4_0 ctx)

def parallel_generatedRoot_3.w_6_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_3.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_3.w_5_0 ctx) (parallel_generatedRoot_3.w_1_0 ctx)

def parallel_generatedRoot_3.w_7_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_3.Ctx backend hashModel params i_0) : mxxMat0 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat0)

def parallel_generatedRoot_3.w_8_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_3.Ctx backend hashModel params i_0) : mxxMat0 :=
  (MxxRuntime.matrixPolynomial [(mxxNat0)] : mxxMat0)

def parallel_generatedRoot_3.w_9_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_3.Ctx backend hashModel params i_0) : mxxMat0 :=
  (0 : mxxMat0)

def parallel_generatedRoot_3.w_10_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_3.Ctx backend hashModel params i_0) : mxxMat0 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat0)

def parallel_generatedRoot_3.w_11_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_3.Ctx backend hashModel params i_0) : Int :=
  -2

def parallel_generatedRoot_3.w_12_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_3.Ctx backend hashModel params i_0) : Fin 4 → Int :=
  ctx.witness.w_12_0

def parallel_generatedRoot_3.w_13_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_3.Ctx backend hashModel params i_0) : mxxMat0 :=
  ctx.witness.w_13_0

def parallel_generatedRoot_3.w_14_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_3.Ctx backend hashModel params i_0) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_3.w_8_0 ctx) (parallel_generatedRoot_3.w_13_0 ctx)

def parallel_generatedRoot_3.w_15_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_3.Ctx backend hashModel params i_0) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_3.w_7_0 ctx) (parallel_generatedRoot_3.w_14_0 ctx)

def parallel_generatedRoot_3.w_16_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_3.Ctx backend hashModel params i_0) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_3.w_15_0 ctx) (parallel_generatedRoot_3.w_3_0 ctx)

def parallel_generatedRoot_3.w_17_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_3.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_3.w_18_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_3.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_18_0

def parallel_generatedRoot_3.w_19_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_3.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_3.w_16_0 ctx) (parallel_generatedRoot_3.w_18_0 ctx)

def parallel_generatedRoot_3.w_20_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_3.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_3.w_6_0 ctx) (parallel_generatedRoot_3.w_19_0 ctx)

def parallel_generatedRoot_3.w_21_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_3.Ctx backend hashModel params i_0) : mxxMat1 :=
  (0 : mxxMat1)

def parallel_generatedRoot_3.w_22_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_3.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_3.w_20_0 ctx) (parallel_generatedRoot_3.w_21_0 ctx)

def parallel_generatedRoot_3.w_23_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_3.Ctx backend hashModel params i_0) : mxxMat0 :=
  MxxRuntime.matrixNeg (parallel_generatedRoot_3.w_4_0 ctx)

def parallel_generatedRoot_3.w_24_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_3.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_3.w_23_0 ctx) (parallel_generatedRoot_3.w_2_0 ctx)

def parallel_generatedRoot_3.w_25_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_3.Ctx backend hashModel params i_0) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_3.w_7_0 ctx) (parallel_generatedRoot_3.w_14_0 ctx)

def parallel_generatedRoot_3.w_27_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_3.Ctx backend hashModel params i_0) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_3.w_25_0 ctx) (parallel_generatedRoot_3.w_26_0 ctx)

def parallel_generatedRoot_3.w_28_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_3.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_3.w_27_0 ctx) (parallel_generatedRoot_3.w_18_0 ctx)

def parallel_generatedRoot_3.w_29_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_3.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_3.w_24_0 ctx) (parallel_generatedRoot_3.w_28_0 ctx)

def parallel_generatedRoot_3.w_30_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_3.Ctx backend hashModel params i_0) : mxxMat1 :=
  (0 : mxxMat1)

def parallel_generatedRoot_3.w_31_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_3.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_3.w_29_0 ctx) (parallel_generatedRoot_3.w_30_0 ctx)

def parallel_generatedRoot_3.outs {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_3.Ctx backend hashModel params i_0) : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit :=
  ((parallel_generatedRoot_3.w_1_0 ctx), ((parallel_generatedRoot_3.w_2_0 ctx), ((parallel_generatedRoot_3.w_22_0 ctx), ((parallel_generatedRoot_3.w_31_0 ctx), ()))))

structure parallel_generatedRoot_3.Facts_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_3.Ctx backend hashModel params i_0) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  t0 : 0 ≤ ((Int.ofNat i_0))
  t1 : ((Int.ofNat i_0)) < 4
  h1_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 98, 47, 76, 117, 116, 73, 100, 40, 49, 41, 47, 49, 47, 50]) ([.bytes [115, 105, 103, 109, 97], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(1), (3), (5), (15)])), .bytes [98, 114, 97, 110, 99, 104], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((parallel_generatedRoot_3.w_0_0 ctx)) (parallel_generatedRoot_3.w_1_0 ctx)
  t3 : 0 ≤ ((Int.ofNat i_0))
  t4 : ((Int.ofNat i_0)) < 4
  h2_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 98, 47, 76, 117, 116, 73, 100, 40, 49, 41, 47, 49, 47, 50]) ([.bytes [115, 105, 103, 109, 97], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(1), (3), (5), (15)])), .bytes [98, 114, 97, 110, 99, 104], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((parallel_generatedRoot_3.w_0_0 ctx)) (parallel_generatedRoot_3.w_2_0 ctx)
  t6 : 0 ≤ ((Int.ofNat i_0))
  t7 : ((Int.ofNat i_0)) < 4
  h4_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(1), (3), (5), (15)])) (parallel_generatedRoot_3.w_3_0 ctx) (parallel_generatedRoot_3.w_4_0 ctx)
  t9 : (4) = 4
  h12_0 : (∀ i : Fin 4, parallel_parallel_generatedRoot_3_12 backend hashModel params i_0 i () ((parallel_generatedRoot_3.w_12_0 ctx) i))
  t11 : 0 ≤ (4)
  h13_0 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0) => sequential_parallel_generatedRoot_3_13 backend hashModel params i_0 i (current, ((parallel_generatedRoot_3.w_10_0 ctx), ((parallel_generatedRoot_3.w_11_0 ctx), ((parallel_generatedRoot_3.w_12_0 ctx), ())))) next) (Int.toNat (4)) (parallel_generatedRoot_3.w_9_0 ctx) (parallel_generatedRoot_3.w_13_0 ctx)
  h17_0 : MxxRuntime.gadgetMatrixRuns backend 4096 25 (parallel_generatedRoot_3.w_17_0 ctx)
  h18_0 : MxxRuntime.centeredRebaseRuns (parallel_generatedRoot_3.w_17_0 ctx) (parallel_generatedRoot_3.w_18_0 ctx)
  hout : outputs = parallel_generatedRoot_3.outs ctx

theorem parallel_generatedRoot_3.facts_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} {ctx : parallel_generatedRoot_3.Ctx backend hashModel params i_0} {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_3.constraints_0 backend hashModel params i_0 (parallel_generatedRoot_3.w_0_0 ctx) (parallel_generatedRoot_3.w_3_0 ctx) (parallel_generatedRoot_3.w_1_0 ctx) (parallel_generatedRoot_3.w_2_0 ctx) (parallel_generatedRoot_3.w_4_0 ctx) (parallel_generatedRoot_3.w_9_0 ctx) (parallel_generatedRoot_3.w_10_0 ctx) (parallel_generatedRoot_3.w_11_0 ctx) (parallel_generatedRoot_3.w_12_0 ctx) (parallel_generatedRoot_3.w_13_0 ctx) (parallel_generatedRoot_3.w_17_0 ctx) (parallel_generatedRoot_3.w_18_0 ctx) (parallel_generatedRoot_3.w_22_0 ctx) (parallel_generatedRoot_3.w_31_0 ctx) outputs) :
    parallel_generatedRoot_3.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_3.Facts {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_3.Ctx backend hashModel params i_0) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  c0 : parallel_generatedRoot_3.Facts_0 ctx outputs

theorem parallel_generatedRoot_3.facts {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_3.Ctx backend hashModel params i_0) {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_3.body backend hashModel params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_3.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_3.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_4.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) where
  inputs : ByteArray × mxxMat0 × mxxMat0 × Unit
  witness : parallel_generatedRoot_4.Witness

def parallel_generatedRoot_4.w_0_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_4.Ctx backend hashModel params i_0) : ByteArray :=
  ctx.inputs.1

def parallel_generatedRoot_4.w_3_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_4.Ctx backend hashModel params i_0) : mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_4.w_25_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_4.Ctx backend hashModel params i_0) : mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_4.w_1_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_4.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_1_0

def parallel_generatedRoot_4.w_2_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_4.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_2_0

def parallel_generatedRoot_4.w_4_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_4.Ctx backend hashModel params i_0) : mxxMat0 :=
  ctx.witness.w_4_0

def parallel_generatedRoot_4.w_5_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_4.Ctx backend hashModel params i_0) : mxxMat0 :=
  MxxRuntime.matrixNeg (parallel_generatedRoot_4.w_4_0 ctx)

def parallel_generatedRoot_4.w_6_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_4.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_4.w_5_0 ctx) (parallel_generatedRoot_4.w_1_0 ctx)

def parallel_generatedRoot_4.w_7_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_4.Ctx backend hashModel params i_0) : mxxMat0 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat0)

def parallel_generatedRoot_4.w_8_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_4.Ctx backend hashModel params i_0) : mxxMat0 :=
  (MxxRuntime.matrixPolynomial [(mxxNat1)] : mxxMat0)

def parallel_generatedRoot_4.w_9_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_4.Ctx backend hashModel params i_0) : mxxMat0 :=
  (0 : mxxMat0)

def parallel_generatedRoot_4.w_10_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_4.Ctx backend hashModel params i_0) : Fin 8 → Int :=
  ctx.witness.w_10_0

def parallel_generatedRoot_4.w_11_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_4.Ctx backend hashModel params i_0) : mxxMat0 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat0)

def parallel_generatedRoot_4.w_12_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_4.Ctx backend hashModel params i_0) : Int :=
  -1

def parallel_generatedRoot_4.w_13_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_4.Ctx backend hashModel params i_0) : mxxMat0 :=
  ctx.witness.w_13_0

def parallel_generatedRoot_4.w_14_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_4.Ctx backend hashModel params i_0) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_4.w_8_0 ctx) (parallel_generatedRoot_4.w_13_0 ctx)

def parallel_generatedRoot_4.w_15_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_4.Ctx backend hashModel params i_0) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_4.w_7_0 ctx) (parallel_generatedRoot_4.w_14_0 ctx)

def parallel_generatedRoot_4.w_16_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_4.Ctx backend hashModel params i_0) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_4.w_15_0 ctx) (parallel_generatedRoot_4.w_3_0 ctx)

def parallel_generatedRoot_4.w_17_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_4.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_4.w_18_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_4.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_4.w_16_0 ctx) (parallel_generatedRoot_4.w_17_0 ctx)

def parallel_generatedRoot_4.w_19_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_4.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_4.w_6_0 ctx) (parallel_generatedRoot_4.w_18_0 ctx)

def parallel_generatedRoot_4.w_20_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_4.Ctx backend hashModel params i_0) : mxxMat1 :=
  (0 : mxxMat1)

def parallel_generatedRoot_4.w_21_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_4.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_4.w_19_0 ctx) (parallel_generatedRoot_4.w_20_0 ctx)

def parallel_generatedRoot_4.w_22_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_4.Ctx backend hashModel params i_0) : mxxMat0 :=
  MxxRuntime.matrixNeg (parallel_generatedRoot_4.w_4_0 ctx)

def parallel_generatedRoot_4.w_23_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_4.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_4.w_22_0 ctx) (parallel_generatedRoot_4.w_2_0 ctx)

def parallel_generatedRoot_4.w_24_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_4.Ctx backend hashModel params i_0) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_4.w_7_0 ctx) (parallel_generatedRoot_4.w_14_0 ctx)

def parallel_generatedRoot_4.w_26_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_4.Ctx backend hashModel params i_0) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_4.w_24_0 ctx) (parallel_generatedRoot_4.w_25_0 ctx)

def parallel_generatedRoot_4.w_27_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_4.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_4.w_26_0 ctx) (parallel_generatedRoot_4.w_17_0 ctx)

def parallel_generatedRoot_4.w_28_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_4.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_4.w_23_0 ctx) (parallel_generatedRoot_4.w_27_0 ctx)

def parallel_generatedRoot_4.w_29_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_4.Ctx backend hashModel params i_0) : mxxMat1 :=
  (0 : mxxMat1)

def parallel_generatedRoot_4.w_30_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_4.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_4.w_28_0 ctx) (parallel_generatedRoot_4.w_29_0 ctx)

def parallel_generatedRoot_4.outs {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_4.Ctx backend hashModel params i_0) : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit :=
  ((parallel_generatedRoot_4.w_1_0 ctx), ((parallel_generatedRoot_4.w_2_0 ctx), ((parallel_generatedRoot_4.w_21_0 ctx), ((parallel_generatedRoot_4.w_30_0 ctx), ()))))

structure parallel_generatedRoot_4.Facts_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_4.Ctx backend hashModel params i_0) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  t0 : 0 ≤ ((Int.ofNat i_0))
  t1 : ((Int.ofNat i_0)) < 8
  h1_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 98, 47, 76, 117, 116, 73, 100, 40, 48, 41, 47, 50, 47, 51]) ([.bytes [115, 105, 103, 109, 97], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(1), (3), (5), (7), (9), (11), (13), (15)])), .bytes [98, 114, 97, 110, 99, 104], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((parallel_generatedRoot_4.w_0_0 ctx)) (parallel_generatedRoot_4.w_1_0 ctx)
  t3 : 0 ≤ ((Int.ofNat i_0))
  t4 : ((Int.ofNat i_0)) < 8
  h2_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 98, 47, 76, 117, 116, 73, 100, 40, 48, 41, 47, 50, 47, 51]) ([.bytes [115, 105, 103, 109, 97], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(1), (3), (5), (7), (9), (11), (13), (15)])), .bytes [98, 114, 97, 110, 99, 104], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((parallel_generatedRoot_4.w_0_0 ctx)) (parallel_generatedRoot_4.w_2_0 ctx)
  t6 : 0 ≤ ((Int.ofNat i_0))
  t7 : ((Int.ofNat i_0)) < 8
  h4_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(1), (3), (5), (7), (9), (11), (13), (15)])) (parallel_generatedRoot_4.w_3_0 ctx) (parallel_generatedRoot_4.w_4_0 ctx)
  t9 : (8) = 8
  h10_0 : (∀ i : Fin 8, parallel_parallel_generatedRoot_4_10 backend hashModel params i_0 i () ((parallel_generatedRoot_4.w_10_0 ctx) i))
  t11 : 0 ≤ (8)
  h13_0 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0) => sequential_parallel_generatedRoot_4_13 backend hashModel params i_0 i (current, ((parallel_generatedRoot_4.w_10_0 ctx), ((parallel_generatedRoot_4.w_11_0 ctx), ((parallel_generatedRoot_4.w_12_0 ctx), ())))) next) (Int.toNat (8)) (parallel_generatedRoot_4.w_9_0 ctx) (parallel_generatedRoot_4.w_13_0 ctx)
  h17_0 : MxxRuntime.gadgetMatrixRuns backend 4096 25 (parallel_generatedRoot_4.w_17_0 ctx)
  hout : outputs = parallel_generatedRoot_4.outs ctx

theorem parallel_generatedRoot_4.facts_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} {ctx : parallel_generatedRoot_4.Ctx backend hashModel params i_0} {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_4.constraints_0 backend hashModel params i_0 (parallel_generatedRoot_4.w_0_0 ctx) (parallel_generatedRoot_4.w_3_0 ctx) (parallel_generatedRoot_4.w_1_0 ctx) (parallel_generatedRoot_4.w_2_0 ctx) (parallel_generatedRoot_4.w_4_0 ctx) (parallel_generatedRoot_4.w_9_0 ctx) (parallel_generatedRoot_4.w_10_0 ctx) (parallel_generatedRoot_4.w_11_0 ctx) (parallel_generatedRoot_4.w_12_0 ctx) (parallel_generatedRoot_4.w_13_0 ctx) (parallel_generatedRoot_4.w_17_0 ctx) (parallel_generatedRoot_4.w_21_0 ctx) (parallel_generatedRoot_4.w_30_0 ctx) outputs) :
    parallel_generatedRoot_4.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_4.Facts {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_4.Ctx backend hashModel params i_0) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  c0 : parallel_generatedRoot_4.Facts_0 ctx outputs

theorem parallel_generatedRoot_4.facts {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_4.Ctx backend hashModel params i_0) {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_4.body backend hashModel params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_4.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_4.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_5.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) where
  inputs : ByteArray × mxxMat0 × mxxMat0 × Unit
  witness : parallel_generatedRoot_5.Witness

def parallel_generatedRoot_5.w_0_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_5.Ctx backend hashModel params i_0) : ByteArray :=
  ctx.inputs.1

def parallel_generatedRoot_5.w_3_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_5.Ctx backend hashModel params i_0) : mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_5.w_26_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_5.Ctx backend hashModel params i_0) : mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_5.w_1_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_5.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_1_0

def parallel_generatedRoot_5.w_2_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_5.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_2_0

def parallel_generatedRoot_5.w_4_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_5.Ctx backend hashModel params i_0) : mxxMat0 :=
  ctx.witness.w_4_0

def parallel_generatedRoot_5.w_5_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_5.Ctx backend hashModel params i_0) : mxxMat0 :=
  MxxRuntime.matrixNeg (parallel_generatedRoot_5.w_4_0 ctx)

def parallel_generatedRoot_5.w_6_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_5.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_5.w_5_0 ctx) (parallel_generatedRoot_5.w_1_0 ctx)

def parallel_generatedRoot_5.w_7_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_5.Ctx backend hashModel params i_0) : mxxMat0 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat0)

def parallel_generatedRoot_5.w_8_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_5.Ctx backend hashModel params i_0) : mxxMat0 :=
  (MxxRuntime.matrixPolynomial [(mxxNat1)] : mxxMat0)

def parallel_generatedRoot_5.w_9_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_5.Ctx backend hashModel params i_0) : mxxMat0 :=
  (0 : mxxMat0)

def parallel_generatedRoot_5.w_10_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_5.Ctx backend hashModel params i_0) : mxxMat0 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat0)

def parallel_generatedRoot_5.w_11_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_5.Ctx backend hashModel params i_0) : Int :=
  -1

def parallel_generatedRoot_5.w_12_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_5.Ctx backend hashModel params i_0) : Fin 8 → Int :=
  ctx.witness.w_12_0

def parallel_generatedRoot_5.w_13_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_5.Ctx backend hashModel params i_0) : mxxMat0 :=
  ctx.witness.w_13_0

def parallel_generatedRoot_5.w_14_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_5.Ctx backend hashModel params i_0) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_5.w_8_0 ctx) (parallel_generatedRoot_5.w_13_0 ctx)

def parallel_generatedRoot_5.w_15_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_5.Ctx backend hashModel params i_0) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_5.w_7_0 ctx) (parallel_generatedRoot_5.w_14_0 ctx)

def parallel_generatedRoot_5.w_16_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_5.Ctx backend hashModel params i_0) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_5.w_15_0 ctx) (parallel_generatedRoot_5.w_3_0 ctx)

def parallel_generatedRoot_5.w_17_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_5.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_5.w_18_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_5.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_18_0

def parallel_generatedRoot_5.w_19_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_5.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_5.w_16_0 ctx) (parallel_generatedRoot_5.w_18_0 ctx)

def parallel_generatedRoot_5.w_20_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_5.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_5.w_6_0 ctx) (parallel_generatedRoot_5.w_19_0 ctx)

def parallel_generatedRoot_5.w_21_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_5.Ctx backend hashModel params i_0) : mxxMat1 :=
  (0 : mxxMat1)

def parallel_generatedRoot_5.w_22_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_5.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_5.w_20_0 ctx) (parallel_generatedRoot_5.w_21_0 ctx)

def parallel_generatedRoot_5.w_23_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_5.Ctx backend hashModel params i_0) : mxxMat0 :=
  MxxRuntime.matrixNeg (parallel_generatedRoot_5.w_4_0 ctx)

def parallel_generatedRoot_5.w_24_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_5.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_5.w_23_0 ctx) (parallel_generatedRoot_5.w_2_0 ctx)

def parallel_generatedRoot_5.w_25_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_5.Ctx backend hashModel params i_0) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_5.w_7_0 ctx) (parallel_generatedRoot_5.w_14_0 ctx)

def parallel_generatedRoot_5.w_27_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_5.Ctx backend hashModel params i_0) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_5.w_25_0 ctx) (parallel_generatedRoot_5.w_26_0 ctx)

def parallel_generatedRoot_5.w_28_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_5.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_5.w_27_0 ctx) (parallel_generatedRoot_5.w_18_0 ctx)

def parallel_generatedRoot_5.w_29_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_5.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_5.w_24_0 ctx) (parallel_generatedRoot_5.w_28_0 ctx)

def parallel_generatedRoot_5.w_30_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_5.Ctx backend hashModel params i_0) : mxxMat1 :=
  (0 : mxxMat1)

def parallel_generatedRoot_5.w_31_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_5.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_5.w_29_0 ctx) (parallel_generatedRoot_5.w_30_0 ctx)

def parallel_generatedRoot_5.outs {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_5.Ctx backend hashModel params i_0) : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit :=
  ((parallel_generatedRoot_5.w_1_0 ctx), ((parallel_generatedRoot_5.w_2_0 ctx), ((parallel_generatedRoot_5.w_22_0 ctx), ((parallel_generatedRoot_5.w_31_0 ctx), ()))))

structure parallel_generatedRoot_5.Facts_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_5.Ctx backend hashModel params i_0) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  t0 : 0 ≤ ((Int.ofNat i_0))
  t1 : ((Int.ofNat i_0)) < 8
  h1_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 98, 47, 76, 117, 116, 73, 100, 40, 49, 49, 41, 47, 50, 47, 51]) ([.bytes [115, 105, 103, 109, 97], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(1), (3), (5), (7), (9), (11), (13), (15)])), .bytes [98, 114, 97, 110, 99, 104], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((parallel_generatedRoot_5.w_0_0 ctx)) (parallel_generatedRoot_5.w_1_0 ctx)
  t3 : 0 ≤ ((Int.ofNat i_0))
  t4 : ((Int.ofNat i_0)) < 8
  h2_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 98, 47, 76, 117, 116, 73, 100, 40, 49, 49, 41, 47, 50, 47, 51]) ([.bytes [115, 105, 103, 109, 97], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(1), (3), (5), (7), (9), (11), (13), (15)])), .bytes [98, 114, 97, 110, 99, 104], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((parallel_generatedRoot_5.w_0_0 ctx)) (parallel_generatedRoot_5.w_2_0 ctx)
  t6 : 0 ≤ ((Int.ofNat i_0))
  t7 : ((Int.ofNat i_0)) < 8
  h4_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(1), (3), (5), (7), (9), (11), (13), (15)])) (parallel_generatedRoot_5.w_3_0 ctx) (parallel_generatedRoot_5.w_4_0 ctx)
  t9 : (8) = 8
  h12_0 : (∀ i : Fin 8, parallel_parallel_generatedRoot_5_12 backend hashModel params i_0 i () ((parallel_generatedRoot_5.w_12_0 ctx) i))
  t11 : 0 ≤ (8)
  h13_0 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0) => sequential_parallel_generatedRoot_5_13 backend hashModel params i_0 i (current, ((parallel_generatedRoot_5.w_10_0 ctx), ((parallel_generatedRoot_5.w_11_0 ctx), ((parallel_generatedRoot_5.w_12_0 ctx), ())))) next) (Int.toNat (8)) (parallel_generatedRoot_5.w_9_0 ctx) (parallel_generatedRoot_5.w_13_0 ctx)
  h17_0 : MxxRuntime.gadgetMatrixRuns backend 4096 25 (parallel_generatedRoot_5.w_17_0 ctx)
  h18_0 : MxxRuntime.centeredRebaseRuns (parallel_generatedRoot_5.w_17_0 ctx) (parallel_generatedRoot_5.w_18_0 ctx)
  hout : outputs = parallel_generatedRoot_5.outs ctx

theorem parallel_generatedRoot_5.facts_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} {ctx : parallel_generatedRoot_5.Ctx backend hashModel params i_0} {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_5.constraints_0 backend hashModel params i_0 (parallel_generatedRoot_5.w_0_0 ctx) (parallel_generatedRoot_5.w_3_0 ctx) (parallel_generatedRoot_5.w_1_0 ctx) (parallel_generatedRoot_5.w_2_0 ctx) (parallel_generatedRoot_5.w_4_0 ctx) (parallel_generatedRoot_5.w_9_0 ctx) (parallel_generatedRoot_5.w_10_0 ctx) (parallel_generatedRoot_5.w_11_0 ctx) (parallel_generatedRoot_5.w_12_0 ctx) (parallel_generatedRoot_5.w_13_0 ctx) (parallel_generatedRoot_5.w_17_0 ctx) (parallel_generatedRoot_5.w_18_0 ctx) (parallel_generatedRoot_5.w_22_0 ctx) (parallel_generatedRoot_5.w_31_0 ctx) outputs) :
    parallel_generatedRoot_5.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_5.Facts {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_5.Ctx backend hashModel params i_0) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  c0 : parallel_generatedRoot_5.Facts_0 ctx outputs

theorem parallel_generatedRoot_5.facts {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_5.Ctx backend hashModel params i_0) {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_5.body backend hashModel params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_5.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_5.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_6.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) where
  inputs : ByteArray × mxxMat0 × mxxMat0 × Unit
  witness : parallel_generatedRoot_6.Witness

def parallel_generatedRoot_6.w_0_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_6.Ctx backend hashModel params i_0) : ByteArray :=
  ctx.inputs.1

def parallel_generatedRoot_6.w_3_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_6.Ctx backend hashModel params i_0) : mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_6.w_26_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_6.Ctx backend hashModel params i_0) : mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_6.w_1_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_6.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_1_0

def parallel_generatedRoot_6.w_2_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_6.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_2_0

def parallel_generatedRoot_6.w_4_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_6.Ctx backend hashModel params i_0) : mxxMat0 :=
  ctx.witness.w_4_0

def parallel_generatedRoot_6.w_5_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_6.Ctx backend hashModel params i_0) : mxxMat0 :=
  MxxRuntime.matrixNeg (parallel_generatedRoot_6.w_4_0 ctx)

def parallel_generatedRoot_6.w_6_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_6.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_6.w_5_0 ctx) (parallel_generatedRoot_6.w_1_0 ctx)

def parallel_generatedRoot_6.w_7_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_6.Ctx backend hashModel params i_0) : mxxMat0 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat0)

def parallel_generatedRoot_6.w_8_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_6.Ctx backend hashModel params i_0) : mxxMat0 :=
  (MxxRuntime.matrixPolynomial [(mxxNat1)] : mxxMat0)

def parallel_generatedRoot_6.w_9_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_6.Ctx backend hashModel params i_0) : mxxMat0 :=
  (0 : mxxMat0)

def parallel_generatedRoot_6.w_10_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_6.Ctx backend hashModel params i_0) : mxxMat0 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat0)

def parallel_generatedRoot_6.w_11_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_6.Ctx backend hashModel params i_0) : Int :=
  -1

def parallel_generatedRoot_6.w_12_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_6.Ctx backend hashModel params i_0) : Fin 8 → Int :=
  ctx.witness.w_12_0

def parallel_generatedRoot_6.w_13_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_6.Ctx backend hashModel params i_0) : mxxMat0 :=
  ctx.witness.w_13_0

def parallel_generatedRoot_6.w_14_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_6.Ctx backend hashModel params i_0) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_6.w_8_0 ctx) (parallel_generatedRoot_6.w_13_0 ctx)

def parallel_generatedRoot_6.w_15_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_6.Ctx backend hashModel params i_0) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_6.w_7_0 ctx) (parallel_generatedRoot_6.w_14_0 ctx)

def parallel_generatedRoot_6.w_16_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_6.Ctx backend hashModel params i_0) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_6.w_15_0 ctx) (parallel_generatedRoot_6.w_3_0 ctx)

def parallel_generatedRoot_6.w_17_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_6.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_6.w_18_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_6.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_18_0

def parallel_generatedRoot_6.w_19_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_6.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_6.w_16_0 ctx) (parallel_generatedRoot_6.w_18_0 ctx)

def parallel_generatedRoot_6.w_20_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_6.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_6.w_6_0 ctx) (parallel_generatedRoot_6.w_19_0 ctx)

def parallel_generatedRoot_6.w_21_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_6.Ctx backend hashModel params i_0) : mxxMat1 :=
  (0 : mxxMat1)

def parallel_generatedRoot_6.w_22_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_6.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_6.w_20_0 ctx) (parallel_generatedRoot_6.w_21_0 ctx)

def parallel_generatedRoot_6.w_23_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_6.Ctx backend hashModel params i_0) : mxxMat0 :=
  MxxRuntime.matrixNeg (parallel_generatedRoot_6.w_4_0 ctx)

def parallel_generatedRoot_6.w_24_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_6.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_6.w_23_0 ctx) (parallel_generatedRoot_6.w_2_0 ctx)

def parallel_generatedRoot_6.w_25_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_6.Ctx backend hashModel params i_0) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_6.w_7_0 ctx) (parallel_generatedRoot_6.w_14_0 ctx)

def parallel_generatedRoot_6.w_27_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_6.Ctx backend hashModel params i_0) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_6.w_25_0 ctx) (parallel_generatedRoot_6.w_26_0 ctx)

def parallel_generatedRoot_6.w_28_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_6.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_6.w_27_0 ctx) (parallel_generatedRoot_6.w_18_0 ctx)

def parallel_generatedRoot_6.w_29_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_6.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_6.w_24_0 ctx) (parallel_generatedRoot_6.w_28_0 ctx)

def parallel_generatedRoot_6.w_30_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_6.Ctx backend hashModel params i_0) : mxxMat1 :=
  (0 : mxxMat1)

def parallel_generatedRoot_6.w_31_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_6.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_6.w_29_0 ctx) (parallel_generatedRoot_6.w_30_0 ctx)

def parallel_generatedRoot_6.outs {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_6.Ctx backend hashModel params i_0) : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit :=
  ((parallel_generatedRoot_6.w_1_0 ctx), ((parallel_generatedRoot_6.w_2_0 ctx), ((parallel_generatedRoot_6.w_22_0 ctx), ((parallel_generatedRoot_6.w_31_0 ctx), ()))))

structure parallel_generatedRoot_6.Facts_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_6.Ctx backend hashModel params i_0) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  t0 : 0 ≤ ((Int.ofNat i_0))
  t1 : ((Int.ofNat i_0)) < 8
  h1_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 98, 47, 76, 117, 116, 73, 100, 40, 50, 41, 47, 50, 47, 51]) ([.bytes [115, 105, 103, 109, 97], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(1), (3), (5), (7), (9), (11), (13), (15)])), .bytes [98, 114, 97, 110, 99, 104], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((parallel_generatedRoot_6.w_0_0 ctx)) (parallel_generatedRoot_6.w_1_0 ctx)
  t3 : 0 ≤ ((Int.ofNat i_0))
  t4 : ((Int.ofNat i_0)) < 8
  h2_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 98, 47, 76, 117, 116, 73, 100, 40, 50, 41, 47, 50, 47, 51]) ([.bytes [115, 105, 103, 109, 97], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(1), (3), (5), (7), (9), (11), (13), (15)])), .bytes [98, 114, 97, 110, 99, 104], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((parallel_generatedRoot_6.w_0_0 ctx)) (parallel_generatedRoot_6.w_2_0 ctx)
  t6 : 0 ≤ ((Int.ofNat i_0))
  t7 : ((Int.ofNat i_0)) < 8
  h4_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(1), (3), (5), (7), (9), (11), (13), (15)])) (parallel_generatedRoot_6.w_3_0 ctx) (parallel_generatedRoot_6.w_4_0 ctx)
  t9 : (8) = 8
  h12_0 : (∀ i : Fin 8, parallel_parallel_generatedRoot_6_12 backend hashModel params i_0 i () ((parallel_generatedRoot_6.w_12_0 ctx) i))
  t11 : 0 ≤ (8)
  h13_0 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0) => sequential_parallel_generatedRoot_6_13 backend hashModel params i_0 i (current, ((parallel_generatedRoot_6.w_10_0 ctx), ((parallel_generatedRoot_6.w_11_0 ctx), ((parallel_generatedRoot_6.w_12_0 ctx), ())))) next) (Int.toNat (8)) (parallel_generatedRoot_6.w_9_0 ctx) (parallel_generatedRoot_6.w_13_0 ctx)
  h17_0 : MxxRuntime.gadgetMatrixRuns backend 4096 25 (parallel_generatedRoot_6.w_17_0 ctx)
  h18_0 : MxxRuntime.centeredRebaseRuns (parallel_generatedRoot_6.w_17_0 ctx) (parallel_generatedRoot_6.w_18_0 ctx)
  hout : outputs = parallel_generatedRoot_6.outs ctx

theorem parallel_generatedRoot_6.facts_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} {ctx : parallel_generatedRoot_6.Ctx backend hashModel params i_0} {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_6.constraints_0 backend hashModel params i_0 (parallel_generatedRoot_6.w_0_0 ctx) (parallel_generatedRoot_6.w_3_0 ctx) (parallel_generatedRoot_6.w_1_0 ctx) (parallel_generatedRoot_6.w_2_0 ctx) (parallel_generatedRoot_6.w_4_0 ctx) (parallel_generatedRoot_6.w_9_0 ctx) (parallel_generatedRoot_6.w_10_0 ctx) (parallel_generatedRoot_6.w_11_0 ctx) (parallel_generatedRoot_6.w_12_0 ctx) (parallel_generatedRoot_6.w_13_0 ctx) (parallel_generatedRoot_6.w_17_0 ctx) (parallel_generatedRoot_6.w_18_0 ctx) (parallel_generatedRoot_6.w_22_0 ctx) (parallel_generatedRoot_6.w_31_0 ctx) outputs) :
    parallel_generatedRoot_6.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_6.Facts {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_6.Ctx backend hashModel params i_0) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  c0 : parallel_generatedRoot_6.Facts_0 ctx outputs

theorem parallel_generatedRoot_6.facts {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_6.Ctx backend hashModel params i_0) {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_6.body backend hashModel params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_6.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_6.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_7.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) where
  inputs : ByteArray × mxxMat0 × mxxMat0 × Unit
  witness : parallel_generatedRoot_7.Witness

def parallel_generatedRoot_7.w_0_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel params i_0) : ByteArray :=
  ctx.inputs.1

def parallel_generatedRoot_7.w_3_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel params i_0) : mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_7.w_25_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel params i_0) : mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_7.w_1_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_1_0

def parallel_generatedRoot_7.w_2_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_2_0

def parallel_generatedRoot_7.w_4_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel params i_0) : mxxMat0 :=
  ctx.witness.w_4_0

def parallel_generatedRoot_7.w_5_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel params i_0) : mxxMat0 :=
  MxxRuntime.matrixNeg (parallel_generatedRoot_7.w_4_0 ctx)

def parallel_generatedRoot_7.w_6_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_7.w_5_0 ctx) (parallel_generatedRoot_7.w_1_0 ctx)

def parallel_generatedRoot_7.w_7_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel params i_0) : mxxMat0 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat0)

def parallel_generatedRoot_7.w_8_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel params i_0) : mxxMat0 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat0)

def parallel_generatedRoot_7.w_9_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel params i_0) : mxxMat0 :=
  (0 : mxxMat0)

def parallel_generatedRoot_7.w_10_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel params i_0) : Fin 1 → Int :=
  ctx.witness.w_10_0

def parallel_generatedRoot_7.w_11_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel params i_0) : mxxMat0 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat0)

def parallel_generatedRoot_7.w_12_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel params i_0) : Int :=
  -1

def parallel_generatedRoot_7.w_13_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel params i_0) : mxxMat0 :=
  ctx.witness.w_13_0

def parallel_generatedRoot_7.w_14_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel params i_0) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_7.w_8_0 ctx) (parallel_generatedRoot_7.w_13_0 ctx)

def parallel_generatedRoot_7.w_15_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel params i_0) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_7.w_7_0 ctx) (parallel_generatedRoot_7.w_14_0 ctx)

def parallel_generatedRoot_7.w_16_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel params i_0) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_7.w_15_0 ctx) (parallel_generatedRoot_7.w_3_0 ctx)

def parallel_generatedRoot_7.w_17_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_7.w_18_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_7.w_16_0 ctx) (parallel_generatedRoot_7.w_17_0 ctx)

def parallel_generatedRoot_7.w_19_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_7.w_6_0 ctx) (parallel_generatedRoot_7.w_18_0 ctx)

def parallel_generatedRoot_7.w_20_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel params i_0) : mxxMat1 :=
  (0 : mxxMat1)

def parallel_generatedRoot_7.w_21_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_7.w_19_0 ctx) (parallel_generatedRoot_7.w_20_0 ctx)

def parallel_generatedRoot_7.w_22_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel params i_0) : mxxMat0 :=
  MxxRuntime.matrixNeg (parallel_generatedRoot_7.w_4_0 ctx)

def parallel_generatedRoot_7.w_23_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_7.w_22_0 ctx) (parallel_generatedRoot_7.w_2_0 ctx)

def parallel_generatedRoot_7.w_24_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel params i_0) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_7.w_7_0 ctx) (parallel_generatedRoot_7.w_14_0 ctx)

def parallel_generatedRoot_7.w_26_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel params i_0) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_7.w_24_0 ctx) (parallel_generatedRoot_7.w_25_0 ctx)

def parallel_generatedRoot_7.w_27_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_7.w_26_0 ctx) (parallel_generatedRoot_7.w_17_0 ctx)

def parallel_generatedRoot_7.w_28_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_7.w_23_0 ctx) (parallel_generatedRoot_7.w_27_0 ctx)

def parallel_generatedRoot_7.w_29_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel params i_0) : mxxMat1 :=
  (0 : mxxMat1)

def parallel_generatedRoot_7.w_30_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_7.w_28_0 ctx) (parallel_generatedRoot_7.w_29_0 ctx)

def parallel_generatedRoot_7.outs {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel params i_0) : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit :=
  ((parallel_generatedRoot_7.w_1_0 ctx), ((parallel_generatedRoot_7.w_2_0 ctx), ((parallel_generatedRoot_7.w_21_0 ctx), ((parallel_generatedRoot_7.w_30_0 ctx), ()))))

structure parallel_generatedRoot_7.Facts_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel params i_0) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  t0 : 0 ≤ ((Int.ofNat i_0))
  t1 : ((Int.ofNat i_0)) < 1
  h1_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 98, 47, 76, 117, 116, 73, 100, 40, 49, 48, 41, 47, 51, 47, 52]) ([.bytes [115, 105, 103, 109, 97], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(15)])), .bytes [98, 114, 97, 110, 99, 104], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((parallel_generatedRoot_7.w_0_0 ctx)) (parallel_generatedRoot_7.w_1_0 ctx)
  t3 : 0 ≤ ((Int.ofNat i_0))
  t4 : ((Int.ofNat i_0)) < 1
  h2_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 98, 47, 76, 117, 116, 73, 100, 40, 49, 48, 41, 47, 51, 47, 52]) ([.bytes [115, 105, 103, 109, 97], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(15)])), .bytes [98, 114, 97, 110, 99, 104], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((parallel_generatedRoot_7.w_0_0 ctx)) (parallel_generatedRoot_7.w_2_0 ctx)
  t6 : 0 ≤ ((Int.ofNat i_0))
  t7 : ((Int.ofNat i_0)) < 1
  h4_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(15)])) (parallel_generatedRoot_7.w_3_0 ctx) (parallel_generatedRoot_7.w_4_0 ctx)
  t9 : (1) = 1
  h10_0 : (∀ i : Fin 1, parallel_parallel_generatedRoot_7_10 backend hashModel params i_0 i () ((parallel_generatedRoot_7.w_10_0 ctx) i))
  t11 : 0 ≤ (1)
  h13_0 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0) => sequential_parallel_generatedRoot_7_13 backend hashModel params i_0 i (current, ((parallel_generatedRoot_7.w_10_0 ctx), ((parallel_generatedRoot_7.w_11_0 ctx), ((parallel_generatedRoot_7.w_12_0 ctx), ())))) next) (Int.toNat (1)) (parallel_generatedRoot_7.w_9_0 ctx) (parallel_generatedRoot_7.w_13_0 ctx)
  h17_0 : MxxRuntime.gadgetMatrixRuns backend 4096 25 (parallel_generatedRoot_7.w_17_0 ctx)
  hout : outputs = parallel_generatedRoot_7.outs ctx

theorem parallel_generatedRoot_7.facts_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} {ctx : parallel_generatedRoot_7.Ctx backend hashModel params i_0} {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_7.constraints_0 backend hashModel params i_0 (parallel_generatedRoot_7.w_0_0 ctx) (parallel_generatedRoot_7.w_3_0 ctx) (parallel_generatedRoot_7.w_1_0 ctx) (parallel_generatedRoot_7.w_2_0 ctx) (parallel_generatedRoot_7.w_4_0 ctx) (parallel_generatedRoot_7.w_9_0 ctx) (parallel_generatedRoot_7.w_10_0 ctx) (parallel_generatedRoot_7.w_11_0 ctx) (parallel_generatedRoot_7.w_12_0 ctx) (parallel_generatedRoot_7.w_13_0 ctx) (parallel_generatedRoot_7.w_17_0 ctx) (parallel_generatedRoot_7.w_21_0 ctx) (parallel_generatedRoot_7.w_30_0 ctx) outputs) :
    parallel_generatedRoot_7.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_7.Facts {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel params i_0) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  c0 : parallel_generatedRoot_7.Facts_0 ctx outputs

theorem parallel_generatedRoot_7.facts {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_7.Ctx backend hashModel params i_0) {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_7.body backend hashModel params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_7.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_7.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_8.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) where
  inputs : ByteArray × mxxMat0 × mxxMat0 × Unit
  witness : parallel_generatedRoot_8.Witness

def parallel_generatedRoot_8.w_0_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_8.Ctx backend hashModel params i_0) : ByteArray :=
  ctx.inputs.1

def parallel_generatedRoot_8.w_3_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_8.Ctx backend hashModel params i_0) : mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_8.w_25_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_8.Ctx backend hashModel params i_0) : mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_8.w_1_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_8.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_1_0

def parallel_generatedRoot_8.w_2_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_8.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_2_0

def parallel_generatedRoot_8.w_4_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_8.Ctx backend hashModel params i_0) : mxxMat0 :=
  ctx.witness.w_4_0

def parallel_generatedRoot_8.w_5_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_8.Ctx backend hashModel params i_0) : mxxMat0 :=
  MxxRuntime.matrixNeg (parallel_generatedRoot_8.w_4_0 ctx)

def parallel_generatedRoot_8.w_6_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_8.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_8.w_5_0 ctx) (parallel_generatedRoot_8.w_1_0 ctx)

def parallel_generatedRoot_8.w_7_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_8.Ctx backend hashModel params i_0) : mxxMat0 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat0)

def parallel_generatedRoot_8.w_8_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_8.Ctx backend hashModel params i_0) : mxxMat0 :=
  (MxxRuntime.matrixPolynomial [(mxxNat1)] : mxxMat0)

def parallel_generatedRoot_8.w_9_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_8.Ctx backend hashModel params i_0) : mxxMat0 :=
  (0 : mxxMat0)

def parallel_generatedRoot_8.w_10_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_8.Ctx backend hashModel params i_0) : Fin 8 → Int :=
  ctx.witness.w_10_0

def parallel_generatedRoot_8.w_11_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_8.Ctx backend hashModel params i_0) : mxxMat0 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat0)

def parallel_generatedRoot_8.w_12_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_8.Ctx backend hashModel params i_0) : Int :=
  -1

def parallel_generatedRoot_8.w_13_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_8.Ctx backend hashModel params i_0) : mxxMat0 :=
  ctx.witness.w_13_0

def parallel_generatedRoot_8.w_14_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_8.Ctx backend hashModel params i_0) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_8.w_8_0 ctx) (parallel_generatedRoot_8.w_13_0 ctx)

def parallel_generatedRoot_8.w_15_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_8.Ctx backend hashModel params i_0) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_8.w_7_0 ctx) (parallel_generatedRoot_8.w_14_0 ctx)

def parallel_generatedRoot_8.w_16_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_8.Ctx backend hashModel params i_0) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_8.w_15_0 ctx) (parallel_generatedRoot_8.w_3_0 ctx)

def parallel_generatedRoot_8.w_17_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_8.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_8.w_18_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_8.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_8.w_16_0 ctx) (parallel_generatedRoot_8.w_17_0 ctx)

def parallel_generatedRoot_8.w_19_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_8.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_8.w_6_0 ctx) (parallel_generatedRoot_8.w_18_0 ctx)

def parallel_generatedRoot_8.w_20_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_8.Ctx backend hashModel params i_0) : mxxMat1 :=
  (0 : mxxMat1)

def parallel_generatedRoot_8.w_21_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_8.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_8.w_19_0 ctx) (parallel_generatedRoot_8.w_20_0 ctx)

def parallel_generatedRoot_8.w_22_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_8.Ctx backend hashModel params i_0) : mxxMat0 :=
  MxxRuntime.matrixNeg (parallel_generatedRoot_8.w_4_0 ctx)

def parallel_generatedRoot_8.w_23_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_8.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_8.w_22_0 ctx) (parallel_generatedRoot_8.w_2_0 ctx)

def parallel_generatedRoot_8.w_24_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_8.Ctx backend hashModel params i_0) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_8.w_7_0 ctx) (parallel_generatedRoot_8.w_14_0 ctx)

def parallel_generatedRoot_8.w_26_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_8.Ctx backend hashModel params i_0) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_8.w_24_0 ctx) (parallel_generatedRoot_8.w_25_0 ctx)

def parallel_generatedRoot_8.w_27_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_8.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_8.w_26_0 ctx) (parallel_generatedRoot_8.w_17_0 ctx)

def parallel_generatedRoot_8.w_28_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_8.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_8.w_23_0 ctx) (parallel_generatedRoot_8.w_27_0 ctx)

def parallel_generatedRoot_8.w_29_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_8.Ctx backend hashModel params i_0) : mxxMat1 :=
  (0 : mxxMat1)

def parallel_generatedRoot_8.w_30_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_8.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_8.w_28_0 ctx) (parallel_generatedRoot_8.w_29_0 ctx)

def parallel_generatedRoot_8.outs {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_8.Ctx backend hashModel params i_0) : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit :=
  ((parallel_generatedRoot_8.w_1_0 ctx), ((parallel_generatedRoot_8.w_2_0 ctx), ((parallel_generatedRoot_8.w_21_0 ctx), ((parallel_generatedRoot_8.w_30_0 ctx), ()))))

structure parallel_generatedRoot_8.Facts_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_8.Ctx backend hashModel params i_0) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  t0 : 0 ≤ ((Int.ofNat i_0))
  t1 : ((Int.ofNat i_0)) < 8
  h1_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 98, 47, 76, 117, 116, 73, 100, 40, 49, 50, 41, 47, 51, 47, 52]) ([.bytes [115, 105, 103, 109, 97], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(1), (3), (5), (7), (9), (11), (13), (15)])), .bytes [98, 114, 97, 110, 99, 104], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((parallel_generatedRoot_8.w_0_0 ctx)) (parallel_generatedRoot_8.w_1_0 ctx)
  t3 : 0 ≤ ((Int.ofNat i_0))
  t4 : ((Int.ofNat i_0)) < 8
  h2_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 98, 47, 76, 117, 116, 73, 100, 40, 49, 50, 41, 47, 51, 47, 52]) ([.bytes [115, 105, 103, 109, 97], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(1), (3), (5), (7), (9), (11), (13), (15)])), .bytes [98, 114, 97, 110, 99, 104], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((parallel_generatedRoot_8.w_0_0 ctx)) (parallel_generatedRoot_8.w_2_0 ctx)
  t6 : 0 ≤ ((Int.ofNat i_0))
  t7 : ((Int.ofNat i_0)) < 8
  h4_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(1), (3), (5), (7), (9), (11), (13), (15)])) (parallel_generatedRoot_8.w_3_0 ctx) (parallel_generatedRoot_8.w_4_0 ctx)
  t9 : (8) = 8
  h10_0 : (∀ i : Fin 8, parallel_parallel_generatedRoot_8_10 backend hashModel params i_0 i () ((parallel_generatedRoot_8.w_10_0 ctx) i))
  t11 : 0 ≤ (8)
  h13_0 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0) => sequential_parallel_generatedRoot_8_13 backend hashModel params i_0 i (current, ((parallel_generatedRoot_8.w_10_0 ctx), ((parallel_generatedRoot_8.w_11_0 ctx), ((parallel_generatedRoot_8.w_12_0 ctx), ())))) next) (Int.toNat (8)) (parallel_generatedRoot_8.w_9_0 ctx) (parallel_generatedRoot_8.w_13_0 ctx)
  h17_0 : MxxRuntime.gadgetMatrixRuns backend 4096 25 (parallel_generatedRoot_8.w_17_0 ctx)
  hout : outputs = parallel_generatedRoot_8.outs ctx

theorem parallel_generatedRoot_8.facts_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} {ctx : parallel_generatedRoot_8.Ctx backend hashModel params i_0} {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_8.constraints_0 backend hashModel params i_0 (parallel_generatedRoot_8.w_0_0 ctx) (parallel_generatedRoot_8.w_3_0 ctx) (parallel_generatedRoot_8.w_1_0 ctx) (parallel_generatedRoot_8.w_2_0 ctx) (parallel_generatedRoot_8.w_4_0 ctx) (parallel_generatedRoot_8.w_9_0 ctx) (parallel_generatedRoot_8.w_10_0 ctx) (parallel_generatedRoot_8.w_11_0 ctx) (parallel_generatedRoot_8.w_12_0 ctx) (parallel_generatedRoot_8.w_13_0 ctx) (parallel_generatedRoot_8.w_17_0 ctx) (parallel_generatedRoot_8.w_21_0 ctx) (parallel_generatedRoot_8.w_30_0 ctx) outputs) :
    parallel_generatedRoot_8.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_8.Facts {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_8.Ctx backend hashModel params i_0) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  c0 : parallel_generatedRoot_8.Facts_0 ctx outputs

theorem parallel_generatedRoot_8.facts {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_8.Ctx backend hashModel params i_0) {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_8.body backend hashModel params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_8.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_8.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_9.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) where
  inputs : ByteArray × mxxMat0 × mxxMat0 × Unit
  witness : parallel_generatedRoot_9.Witness

def parallel_generatedRoot_9.w_0_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_9.Ctx backend hashModel params i_0) : ByteArray :=
  ctx.inputs.1

def parallel_generatedRoot_9.w_3_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_9.Ctx backend hashModel params i_0) : mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_9.w_25_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_9.Ctx backend hashModel params i_0) : mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_9.w_1_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_9.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_1_0

def parallel_generatedRoot_9.w_2_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_9.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_2_0

def parallel_generatedRoot_9.w_4_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_9.Ctx backend hashModel params i_0) : mxxMat0 :=
  ctx.witness.w_4_0

def parallel_generatedRoot_9.w_5_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_9.Ctx backend hashModel params i_0) : mxxMat0 :=
  MxxRuntime.matrixNeg (parallel_generatedRoot_9.w_4_0 ctx)

def parallel_generatedRoot_9.w_6_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_9.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_9.w_5_0 ctx) (parallel_generatedRoot_9.w_1_0 ctx)

def parallel_generatedRoot_9.w_7_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_9.Ctx backend hashModel params i_0) : mxxMat0 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat0)

def parallel_generatedRoot_9.w_8_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_9.Ctx backend hashModel params i_0) : mxxMat0 :=
  (MxxRuntime.matrixPolynomial [(mxxNat1)] : mxxMat0)

def parallel_generatedRoot_9.w_9_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_9.Ctx backend hashModel params i_0) : mxxMat0 :=
  (0 : mxxMat0)

def parallel_generatedRoot_9.w_10_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_9.Ctx backend hashModel params i_0) : Fin 8 → Int :=
  ctx.witness.w_10_0

def parallel_generatedRoot_9.w_11_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_9.Ctx backend hashModel params i_0) : mxxMat0 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat0)

def parallel_generatedRoot_9.w_12_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_9.Ctx backend hashModel params i_0) : Int :=
  -1

def parallel_generatedRoot_9.w_13_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_9.Ctx backend hashModel params i_0) : mxxMat0 :=
  ctx.witness.w_13_0

def parallel_generatedRoot_9.w_14_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_9.Ctx backend hashModel params i_0) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_9.w_8_0 ctx) (parallel_generatedRoot_9.w_13_0 ctx)

def parallel_generatedRoot_9.w_15_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_9.Ctx backend hashModel params i_0) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_9.w_7_0 ctx) (parallel_generatedRoot_9.w_14_0 ctx)

def parallel_generatedRoot_9.w_16_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_9.Ctx backend hashModel params i_0) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_9.w_15_0 ctx) (parallel_generatedRoot_9.w_3_0 ctx)

def parallel_generatedRoot_9.w_17_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_9.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_9.w_18_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_9.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_9.w_16_0 ctx) (parallel_generatedRoot_9.w_17_0 ctx)

def parallel_generatedRoot_9.w_19_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_9.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_9.w_6_0 ctx) (parallel_generatedRoot_9.w_18_0 ctx)

def parallel_generatedRoot_9.w_20_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_9.Ctx backend hashModel params i_0) : mxxMat1 :=
  (0 : mxxMat1)

def parallel_generatedRoot_9.w_21_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_9.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_9.w_19_0 ctx) (parallel_generatedRoot_9.w_20_0 ctx)

def parallel_generatedRoot_9.w_22_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_9.Ctx backend hashModel params i_0) : mxxMat0 :=
  MxxRuntime.matrixNeg (parallel_generatedRoot_9.w_4_0 ctx)

def parallel_generatedRoot_9.w_23_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_9.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_9.w_22_0 ctx) (parallel_generatedRoot_9.w_2_0 ctx)

def parallel_generatedRoot_9.w_24_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_9.Ctx backend hashModel params i_0) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_9.w_7_0 ctx) (parallel_generatedRoot_9.w_14_0 ctx)

def parallel_generatedRoot_9.w_26_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_9.Ctx backend hashModel params i_0) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_9.w_24_0 ctx) (parallel_generatedRoot_9.w_25_0 ctx)

def parallel_generatedRoot_9.w_27_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_9.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_9.w_26_0 ctx) (parallel_generatedRoot_9.w_17_0 ctx)

def parallel_generatedRoot_9.w_28_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_9.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_9.w_23_0 ctx) (parallel_generatedRoot_9.w_27_0 ctx)

def parallel_generatedRoot_9.w_29_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_9.Ctx backend hashModel params i_0) : mxxMat1 :=
  (0 : mxxMat1)

def parallel_generatedRoot_9.w_30_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_9.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_9.w_28_0 ctx) (parallel_generatedRoot_9.w_29_0 ctx)

def parallel_generatedRoot_9.outs {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_9.Ctx backend hashModel params i_0) : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit :=
  ((parallel_generatedRoot_9.w_1_0 ctx), ((parallel_generatedRoot_9.w_2_0 ctx), ((parallel_generatedRoot_9.w_21_0 ctx), ((parallel_generatedRoot_9.w_30_0 ctx), ()))))

structure parallel_generatedRoot_9.Facts_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_9.Ctx backend hashModel params i_0) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  t0 : 0 ≤ ((Int.ofNat i_0))
  t1 : ((Int.ofNat i_0)) < 8
  h1_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 98, 47, 76, 117, 116, 73, 100, 40, 51, 41, 47, 51, 47, 52]) ([.bytes [115, 105, 103, 109, 97], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(1), (3), (5), (7), (9), (11), (13), (15)])), .bytes [98, 114, 97, 110, 99, 104], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((parallel_generatedRoot_9.w_0_0 ctx)) (parallel_generatedRoot_9.w_1_0 ctx)
  t3 : 0 ≤ ((Int.ofNat i_0))
  t4 : ((Int.ofNat i_0)) < 8
  h2_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 98, 47, 76, 117, 116, 73, 100, 40, 51, 41, 47, 51, 47, 52]) ([.bytes [115, 105, 103, 109, 97], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(1), (3), (5), (7), (9), (11), (13), (15)])), .bytes [98, 114, 97, 110, 99, 104], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((parallel_generatedRoot_9.w_0_0 ctx)) (parallel_generatedRoot_9.w_2_0 ctx)
  t6 : 0 ≤ ((Int.ofNat i_0))
  t7 : ((Int.ofNat i_0)) < 8
  h4_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(1), (3), (5), (7), (9), (11), (13), (15)])) (parallel_generatedRoot_9.w_3_0 ctx) (parallel_generatedRoot_9.w_4_0 ctx)
  t9 : (8) = 8
  h10_0 : (∀ i : Fin 8, parallel_parallel_generatedRoot_9_10 backend hashModel params i_0 i () ((parallel_generatedRoot_9.w_10_0 ctx) i))
  t11 : 0 ≤ (8)
  h13_0 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0) => sequential_parallel_generatedRoot_9_13 backend hashModel params i_0 i (current, ((parallel_generatedRoot_9.w_10_0 ctx), ((parallel_generatedRoot_9.w_11_0 ctx), ((parallel_generatedRoot_9.w_12_0 ctx), ())))) next) (Int.toNat (8)) (parallel_generatedRoot_9.w_9_0 ctx) (parallel_generatedRoot_9.w_13_0 ctx)
  h17_0 : MxxRuntime.gadgetMatrixRuns backend 4096 25 (parallel_generatedRoot_9.w_17_0 ctx)
  hout : outputs = parallel_generatedRoot_9.outs ctx

theorem parallel_generatedRoot_9.facts_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} {ctx : parallel_generatedRoot_9.Ctx backend hashModel params i_0} {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_9.constraints_0 backend hashModel params i_0 (parallel_generatedRoot_9.w_0_0 ctx) (parallel_generatedRoot_9.w_3_0 ctx) (parallel_generatedRoot_9.w_1_0 ctx) (parallel_generatedRoot_9.w_2_0 ctx) (parallel_generatedRoot_9.w_4_0 ctx) (parallel_generatedRoot_9.w_9_0 ctx) (parallel_generatedRoot_9.w_10_0 ctx) (parallel_generatedRoot_9.w_11_0 ctx) (parallel_generatedRoot_9.w_12_0 ctx) (parallel_generatedRoot_9.w_13_0 ctx) (parallel_generatedRoot_9.w_17_0 ctx) (parallel_generatedRoot_9.w_21_0 ctx) (parallel_generatedRoot_9.w_30_0 ctx) outputs) :
    parallel_generatedRoot_9.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_9.Facts {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_9.Ctx backend hashModel params i_0) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  c0 : parallel_generatedRoot_9.Facts_0 ctx outputs

theorem parallel_generatedRoot_9.facts {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_9.Ctx backend hashModel params i_0) {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_9.body backend hashModel params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_9.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_9.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_10.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) where
  inputs : ByteArray × mxxMat0 × mxxMat0 × Unit
  witness : parallel_generatedRoot_10.Witness

def parallel_generatedRoot_10.w_0_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_10.Ctx backend hashModel params i_0) : ByteArray :=
  ctx.inputs.1

def parallel_generatedRoot_10.w_3_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_10.Ctx backend hashModel params i_0) : mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_10.w_25_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_10.Ctx backend hashModel params i_0) : mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_10.w_1_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_10.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_1_0

def parallel_generatedRoot_10.w_2_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_10.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_2_0

def parallel_generatedRoot_10.w_4_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_10.Ctx backend hashModel params i_0) : mxxMat0 :=
  ctx.witness.w_4_0

def parallel_generatedRoot_10.w_5_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_10.Ctx backend hashModel params i_0) : mxxMat0 :=
  MxxRuntime.matrixNeg (parallel_generatedRoot_10.w_4_0 ctx)

def parallel_generatedRoot_10.w_6_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_10.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_10.w_5_0 ctx) (parallel_generatedRoot_10.w_1_0 ctx)

def parallel_generatedRoot_10.w_7_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_10.Ctx backend hashModel params i_0) : mxxMat0 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat0)

def parallel_generatedRoot_10.w_8_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_10.Ctx backend hashModel params i_0) : mxxMat0 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat0)

def parallel_generatedRoot_10.w_9_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_10.Ctx backend hashModel params i_0) : mxxMat0 :=
  (0 : mxxMat0)

def parallel_generatedRoot_10.w_10_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_10.Ctx backend hashModel params i_0) : Fin 1 → Int :=
  ctx.witness.w_10_0

def parallel_generatedRoot_10.w_11_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_10.Ctx backend hashModel params i_0) : mxxMat0 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat0)

def parallel_generatedRoot_10.w_12_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_10.Ctx backend hashModel params i_0) : Int :=
  -1

def parallel_generatedRoot_10.w_13_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_10.Ctx backend hashModel params i_0) : mxxMat0 :=
  ctx.witness.w_13_0

def parallel_generatedRoot_10.w_14_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_10.Ctx backend hashModel params i_0) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_10.w_8_0 ctx) (parallel_generatedRoot_10.w_13_0 ctx)

def parallel_generatedRoot_10.w_15_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_10.Ctx backend hashModel params i_0) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_10.w_7_0 ctx) (parallel_generatedRoot_10.w_14_0 ctx)

def parallel_generatedRoot_10.w_16_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_10.Ctx backend hashModel params i_0) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_10.w_15_0 ctx) (parallel_generatedRoot_10.w_3_0 ctx)

def parallel_generatedRoot_10.w_17_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_10.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_10.w_18_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_10.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_10.w_16_0 ctx) (parallel_generatedRoot_10.w_17_0 ctx)

def parallel_generatedRoot_10.w_19_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_10.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_10.w_6_0 ctx) (parallel_generatedRoot_10.w_18_0 ctx)

def parallel_generatedRoot_10.w_20_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_10.Ctx backend hashModel params i_0) : mxxMat1 :=
  (0 : mxxMat1)

def parallel_generatedRoot_10.w_21_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_10.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_10.w_19_0 ctx) (parallel_generatedRoot_10.w_20_0 ctx)

def parallel_generatedRoot_10.w_22_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_10.Ctx backend hashModel params i_0) : mxxMat0 :=
  MxxRuntime.matrixNeg (parallel_generatedRoot_10.w_4_0 ctx)

def parallel_generatedRoot_10.w_23_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_10.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_10.w_22_0 ctx) (parallel_generatedRoot_10.w_2_0 ctx)

def parallel_generatedRoot_10.w_24_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_10.Ctx backend hashModel params i_0) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_10.w_7_0 ctx) (parallel_generatedRoot_10.w_14_0 ctx)

def parallel_generatedRoot_10.w_26_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_10.Ctx backend hashModel params i_0) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_10.w_24_0 ctx) (parallel_generatedRoot_10.w_25_0 ctx)

def parallel_generatedRoot_10.w_27_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_10.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_10.w_26_0 ctx) (parallel_generatedRoot_10.w_17_0 ctx)

def parallel_generatedRoot_10.w_28_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_10.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_10.w_23_0 ctx) (parallel_generatedRoot_10.w_27_0 ctx)

def parallel_generatedRoot_10.w_29_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_10.Ctx backend hashModel params i_0) : mxxMat1 :=
  (0 : mxxMat1)

def parallel_generatedRoot_10.w_30_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_10.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_10.w_28_0 ctx) (parallel_generatedRoot_10.w_29_0 ctx)

def parallel_generatedRoot_10.outs {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_10.Ctx backend hashModel params i_0) : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit :=
  ((parallel_generatedRoot_10.w_1_0 ctx), ((parallel_generatedRoot_10.w_2_0 ctx), ((parallel_generatedRoot_10.w_21_0 ctx), ((parallel_generatedRoot_10.w_30_0 ctx), ()))))

structure parallel_generatedRoot_10.Facts_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_10.Ctx backend hashModel params i_0) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  t0 : 0 ≤ ((Int.ofNat i_0))
  t1 : ((Int.ofNat i_0)) < 1
  h1_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 98, 47, 76, 117, 116, 73, 100, 40, 52, 41, 47, 51, 47, 52]) ([.bytes [115, 105, 103, 109, 97], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(3)])), .bytes [98, 114, 97, 110, 99, 104], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((parallel_generatedRoot_10.w_0_0 ctx)) (parallel_generatedRoot_10.w_1_0 ctx)
  t3 : 0 ≤ ((Int.ofNat i_0))
  t4 : ((Int.ofNat i_0)) < 1
  h2_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 98, 47, 76, 117, 116, 73, 100, 40, 52, 41, 47, 51, 47, 52]) ([.bytes [115, 105, 103, 109, 97], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(3)])), .bytes [98, 114, 97, 110, 99, 104], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((parallel_generatedRoot_10.w_0_0 ctx)) (parallel_generatedRoot_10.w_2_0 ctx)
  t6 : 0 ≤ ((Int.ofNat i_0))
  t7 : ((Int.ofNat i_0)) < 1
  h4_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(3)])) (parallel_generatedRoot_10.w_3_0 ctx) (parallel_generatedRoot_10.w_4_0 ctx)
  t9 : (1) = 1
  h10_0 : (∀ i : Fin 1, parallel_parallel_generatedRoot_10_10 backend hashModel params i_0 i () ((parallel_generatedRoot_10.w_10_0 ctx) i))
  t11 : 0 ≤ (1)
  h13_0 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0) => sequential_parallel_generatedRoot_10_13 backend hashModel params i_0 i (current, ((parallel_generatedRoot_10.w_10_0 ctx), ((parallel_generatedRoot_10.w_11_0 ctx), ((parallel_generatedRoot_10.w_12_0 ctx), ())))) next) (Int.toNat (1)) (parallel_generatedRoot_10.w_9_0 ctx) (parallel_generatedRoot_10.w_13_0 ctx)
  h17_0 : MxxRuntime.gadgetMatrixRuns backend 4096 25 (parallel_generatedRoot_10.w_17_0 ctx)
  hout : outputs = parallel_generatedRoot_10.outs ctx

theorem parallel_generatedRoot_10.facts_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} {ctx : parallel_generatedRoot_10.Ctx backend hashModel params i_0} {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_10.constraints_0 backend hashModel params i_0 (parallel_generatedRoot_10.w_0_0 ctx) (parallel_generatedRoot_10.w_3_0 ctx) (parallel_generatedRoot_10.w_1_0 ctx) (parallel_generatedRoot_10.w_2_0 ctx) (parallel_generatedRoot_10.w_4_0 ctx) (parallel_generatedRoot_10.w_9_0 ctx) (parallel_generatedRoot_10.w_10_0 ctx) (parallel_generatedRoot_10.w_11_0 ctx) (parallel_generatedRoot_10.w_12_0 ctx) (parallel_generatedRoot_10.w_13_0 ctx) (parallel_generatedRoot_10.w_17_0 ctx) (parallel_generatedRoot_10.w_21_0 ctx) (parallel_generatedRoot_10.w_30_0 ctx) outputs) :
    parallel_generatedRoot_10.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_10.Facts {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_10.Ctx backend hashModel params i_0) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  c0 : parallel_generatedRoot_10.Facts_0 ctx outputs

theorem parallel_generatedRoot_10.facts {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_10.Ctx backend hashModel params i_0) {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_10.body backend hashModel params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_10.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_10.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_11.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) where
  inputs : ByteArray × mxxMat0 × mxxMat0 × Unit
  witness : parallel_generatedRoot_11.Witness

def parallel_generatedRoot_11.w_0_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_11.Ctx backend hashModel params i_0) : ByteArray :=
  ctx.inputs.1

def parallel_generatedRoot_11.w_3_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_11.Ctx backend hashModel params i_0) : mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_11.w_25_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_11.Ctx backend hashModel params i_0) : mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_11.w_1_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_11.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_1_0

def parallel_generatedRoot_11.w_2_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_11.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_2_0

def parallel_generatedRoot_11.w_4_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_11.Ctx backend hashModel params i_0) : mxxMat0 :=
  ctx.witness.w_4_0

def parallel_generatedRoot_11.w_5_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_11.Ctx backend hashModel params i_0) : mxxMat0 :=
  MxxRuntime.matrixNeg (parallel_generatedRoot_11.w_4_0 ctx)

def parallel_generatedRoot_11.w_6_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_11.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_11.w_5_0 ctx) (parallel_generatedRoot_11.w_1_0 ctx)

def parallel_generatedRoot_11.w_7_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_11.Ctx backend hashModel params i_0) : mxxMat0 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat0)

def parallel_generatedRoot_11.w_8_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_11.Ctx backend hashModel params i_0) : mxxMat0 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat0)

def parallel_generatedRoot_11.w_9_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_11.Ctx backend hashModel params i_0) : mxxMat0 :=
  (0 : mxxMat0)

def parallel_generatedRoot_11.w_10_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_11.Ctx backend hashModel params i_0) : Fin 1 → Int :=
  ctx.witness.w_10_0

def parallel_generatedRoot_11.w_11_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_11.Ctx backend hashModel params i_0) : mxxMat0 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat0)

def parallel_generatedRoot_11.w_12_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_11.Ctx backend hashModel params i_0) : Int :=
  -1

def parallel_generatedRoot_11.w_13_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_11.Ctx backend hashModel params i_0) : mxxMat0 :=
  ctx.witness.w_13_0

def parallel_generatedRoot_11.w_14_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_11.Ctx backend hashModel params i_0) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_11.w_8_0 ctx) (parallel_generatedRoot_11.w_13_0 ctx)

def parallel_generatedRoot_11.w_15_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_11.Ctx backend hashModel params i_0) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_11.w_7_0 ctx) (parallel_generatedRoot_11.w_14_0 ctx)

def parallel_generatedRoot_11.w_16_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_11.Ctx backend hashModel params i_0) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_11.w_15_0 ctx) (parallel_generatedRoot_11.w_3_0 ctx)

def parallel_generatedRoot_11.w_17_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_11.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_11.w_18_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_11.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_11.w_16_0 ctx) (parallel_generatedRoot_11.w_17_0 ctx)

def parallel_generatedRoot_11.w_19_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_11.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_11.w_6_0 ctx) (parallel_generatedRoot_11.w_18_0 ctx)

def parallel_generatedRoot_11.w_20_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_11.Ctx backend hashModel params i_0) : mxxMat1 :=
  (0 : mxxMat1)

def parallel_generatedRoot_11.w_21_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_11.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_11.w_19_0 ctx) (parallel_generatedRoot_11.w_20_0 ctx)

def parallel_generatedRoot_11.w_22_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_11.Ctx backend hashModel params i_0) : mxxMat0 :=
  MxxRuntime.matrixNeg (parallel_generatedRoot_11.w_4_0 ctx)

def parallel_generatedRoot_11.w_23_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_11.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_11.w_22_0 ctx) (parallel_generatedRoot_11.w_2_0 ctx)

def parallel_generatedRoot_11.w_24_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_11.Ctx backend hashModel params i_0) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_11.w_7_0 ctx) (parallel_generatedRoot_11.w_14_0 ctx)

def parallel_generatedRoot_11.w_26_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_11.Ctx backend hashModel params i_0) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_11.w_24_0 ctx) (parallel_generatedRoot_11.w_25_0 ctx)

def parallel_generatedRoot_11.w_27_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_11.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_11.w_26_0 ctx) (parallel_generatedRoot_11.w_17_0 ctx)

def parallel_generatedRoot_11.w_28_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_11.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_11.w_23_0 ctx) (parallel_generatedRoot_11.w_27_0 ctx)

def parallel_generatedRoot_11.w_29_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_11.Ctx backend hashModel params i_0) : mxxMat1 :=
  (0 : mxxMat1)

def parallel_generatedRoot_11.w_30_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_11.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_11.w_28_0 ctx) (parallel_generatedRoot_11.w_29_0 ctx)

def parallel_generatedRoot_11.outs {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_11.Ctx backend hashModel params i_0) : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit :=
  ((parallel_generatedRoot_11.w_1_0 ctx), ((parallel_generatedRoot_11.w_2_0 ctx), ((parallel_generatedRoot_11.w_21_0 ctx), ((parallel_generatedRoot_11.w_30_0 ctx), ()))))

structure parallel_generatedRoot_11.Facts_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_11.Ctx backend hashModel params i_0) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  t0 : 0 ≤ ((Int.ofNat i_0))
  t1 : ((Int.ofNat i_0)) < 1
  h1_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 98, 47, 76, 117, 116, 73, 100, 40, 53, 41, 47, 51, 47, 52]) ([.bytes [115, 105, 103, 109, 97], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(5)])), .bytes [98, 114, 97, 110, 99, 104], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((parallel_generatedRoot_11.w_0_0 ctx)) (parallel_generatedRoot_11.w_1_0 ctx)
  t3 : 0 ≤ ((Int.ofNat i_0))
  t4 : ((Int.ofNat i_0)) < 1
  h2_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 98, 47, 76, 117, 116, 73, 100, 40, 53, 41, 47, 51, 47, 52]) ([.bytes [115, 105, 103, 109, 97], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(5)])), .bytes [98, 114, 97, 110, 99, 104], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((parallel_generatedRoot_11.w_0_0 ctx)) (parallel_generatedRoot_11.w_2_0 ctx)
  t6 : 0 ≤ ((Int.ofNat i_0))
  t7 : ((Int.ofNat i_0)) < 1
  h4_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(5)])) (parallel_generatedRoot_11.w_3_0 ctx) (parallel_generatedRoot_11.w_4_0 ctx)
  t9 : (1) = 1
  h10_0 : (∀ i : Fin 1, parallel_parallel_generatedRoot_11_10 backend hashModel params i_0 i () ((parallel_generatedRoot_11.w_10_0 ctx) i))
  t11 : 0 ≤ (1)
  h13_0 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0) => sequential_parallel_generatedRoot_11_13 backend hashModel params i_0 i (current, ((parallel_generatedRoot_11.w_10_0 ctx), ((parallel_generatedRoot_11.w_11_0 ctx), ((parallel_generatedRoot_11.w_12_0 ctx), ())))) next) (Int.toNat (1)) (parallel_generatedRoot_11.w_9_0 ctx) (parallel_generatedRoot_11.w_13_0 ctx)
  h17_0 : MxxRuntime.gadgetMatrixRuns backend 4096 25 (parallel_generatedRoot_11.w_17_0 ctx)
  hout : outputs = parallel_generatedRoot_11.outs ctx

theorem parallel_generatedRoot_11.facts_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} {ctx : parallel_generatedRoot_11.Ctx backend hashModel params i_0} {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_11.constraints_0 backend hashModel params i_0 (parallel_generatedRoot_11.w_0_0 ctx) (parallel_generatedRoot_11.w_3_0 ctx) (parallel_generatedRoot_11.w_1_0 ctx) (parallel_generatedRoot_11.w_2_0 ctx) (parallel_generatedRoot_11.w_4_0 ctx) (parallel_generatedRoot_11.w_9_0 ctx) (parallel_generatedRoot_11.w_10_0 ctx) (parallel_generatedRoot_11.w_11_0 ctx) (parallel_generatedRoot_11.w_12_0 ctx) (parallel_generatedRoot_11.w_13_0 ctx) (parallel_generatedRoot_11.w_17_0 ctx) (parallel_generatedRoot_11.w_21_0 ctx) (parallel_generatedRoot_11.w_30_0 ctx) outputs) :
    parallel_generatedRoot_11.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_11.Facts {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_11.Ctx backend hashModel params i_0) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  c0 : parallel_generatedRoot_11.Facts_0 ctx outputs

theorem parallel_generatedRoot_11.facts {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_11.Ctx backend hashModel params i_0) {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_11.body backend hashModel params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_11.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_11.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_12.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) where
  inputs : ByteArray × mxxMat0 × mxxMat0 × Unit
  witness : parallel_generatedRoot_12.Witness

def parallel_generatedRoot_12.w_0_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_12.Ctx backend hashModel params i_0) : ByteArray :=
  ctx.inputs.1

def parallel_generatedRoot_12.w_3_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_12.Ctx backend hashModel params i_0) : mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_12.w_25_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_12.Ctx backend hashModel params i_0) : mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_12.w_1_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_12.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_1_0

def parallel_generatedRoot_12.w_2_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_12.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_2_0

def parallel_generatedRoot_12.w_4_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_12.Ctx backend hashModel params i_0) : mxxMat0 :=
  ctx.witness.w_4_0

def parallel_generatedRoot_12.w_5_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_12.Ctx backend hashModel params i_0) : mxxMat0 :=
  MxxRuntime.matrixNeg (parallel_generatedRoot_12.w_4_0 ctx)

def parallel_generatedRoot_12.w_6_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_12.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_12.w_5_0 ctx) (parallel_generatedRoot_12.w_1_0 ctx)

def parallel_generatedRoot_12.w_7_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_12.Ctx backend hashModel params i_0) : mxxMat0 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat0)

def parallel_generatedRoot_12.w_8_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_12.Ctx backend hashModel params i_0) : mxxMat0 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat0)

def parallel_generatedRoot_12.w_9_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_12.Ctx backend hashModel params i_0) : mxxMat0 :=
  (0 : mxxMat0)

def parallel_generatedRoot_12.w_10_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_12.Ctx backend hashModel params i_0) : Fin 1 → Int :=
  ctx.witness.w_10_0

def parallel_generatedRoot_12.w_11_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_12.Ctx backend hashModel params i_0) : mxxMat0 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat0)

def parallel_generatedRoot_12.w_12_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_12.Ctx backend hashModel params i_0) : Int :=
  -1

def parallel_generatedRoot_12.w_13_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_12.Ctx backend hashModel params i_0) : mxxMat0 :=
  ctx.witness.w_13_0

def parallel_generatedRoot_12.w_14_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_12.Ctx backend hashModel params i_0) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_12.w_8_0 ctx) (parallel_generatedRoot_12.w_13_0 ctx)

def parallel_generatedRoot_12.w_15_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_12.Ctx backend hashModel params i_0) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_12.w_7_0 ctx) (parallel_generatedRoot_12.w_14_0 ctx)

def parallel_generatedRoot_12.w_16_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_12.Ctx backend hashModel params i_0) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_12.w_15_0 ctx) (parallel_generatedRoot_12.w_3_0 ctx)

def parallel_generatedRoot_12.w_17_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_12.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_12.w_18_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_12.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_12.w_16_0 ctx) (parallel_generatedRoot_12.w_17_0 ctx)

def parallel_generatedRoot_12.w_19_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_12.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_12.w_6_0 ctx) (parallel_generatedRoot_12.w_18_0 ctx)

def parallel_generatedRoot_12.w_20_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_12.Ctx backend hashModel params i_0) : mxxMat1 :=
  (0 : mxxMat1)

def parallel_generatedRoot_12.w_21_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_12.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_12.w_19_0 ctx) (parallel_generatedRoot_12.w_20_0 ctx)

def parallel_generatedRoot_12.w_22_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_12.Ctx backend hashModel params i_0) : mxxMat0 :=
  MxxRuntime.matrixNeg (parallel_generatedRoot_12.w_4_0 ctx)

def parallel_generatedRoot_12.w_23_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_12.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_12.w_22_0 ctx) (parallel_generatedRoot_12.w_2_0 ctx)

def parallel_generatedRoot_12.w_24_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_12.Ctx backend hashModel params i_0) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_12.w_7_0 ctx) (parallel_generatedRoot_12.w_14_0 ctx)

def parallel_generatedRoot_12.w_26_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_12.Ctx backend hashModel params i_0) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_12.w_24_0 ctx) (parallel_generatedRoot_12.w_25_0 ctx)

def parallel_generatedRoot_12.w_27_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_12.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_12.w_26_0 ctx) (parallel_generatedRoot_12.w_17_0 ctx)

def parallel_generatedRoot_12.w_28_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_12.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_12.w_23_0 ctx) (parallel_generatedRoot_12.w_27_0 ctx)

def parallel_generatedRoot_12.w_29_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_12.Ctx backend hashModel params i_0) : mxxMat1 :=
  (0 : mxxMat1)

def parallel_generatedRoot_12.w_30_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_12.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_12.w_28_0 ctx) (parallel_generatedRoot_12.w_29_0 ctx)

def parallel_generatedRoot_12.outs {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_12.Ctx backend hashModel params i_0) : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit :=
  ((parallel_generatedRoot_12.w_1_0 ctx), ((parallel_generatedRoot_12.w_2_0 ctx), ((parallel_generatedRoot_12.w_21_0 ctx), ((parallel_generatedRoot_12.w_30_0 ctx), ()))))

structure parallel_generatedRoot_12.Facts_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_12.Ctx backend hashModel params i_0) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  t0 : 0 ≤ ((Int.ofNat i_0))
  t1 : ((Int.ofNat i_0)) < 1
  h1_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 98, 47, 76, 117, 116, 73, 100, 40, 54, 41, 47, 51, 47, 52]) ([.bytes [115, 105, 103, 109, 97], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(7)])), .bytes [98, 114, 97, 110, 99, 104], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((parallel_generatedRoot_12.w_0_0 ctx)) (parallel_generatedRoot_12.w_1_0 ctx)
  t3 : 0 ≤ ((Int.ofNat i_0))
  t4 : ((Int.ofNat i_0)) < 1
  h2_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 98, 47, 76, 117, 116, 73, 100, 40, 54, 41, 47, 51, 47, 52]) ([.bytes [115, 105, 103, 109, 97], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(7)])), .bytes [98, 114, 97, 110, 99, 104], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((parallel_generatedRoot_12.w_0_0 ctx)) (parallel_generatedRoot_12.w_2_0 ctx)
  t6 : 0 ≤ ((Int.ofNat i_0))
  t7 : ((Int.ofNat i_0)) < 1
  h4_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(7)])) (parallel_generatedRoot_12.w_3_0 ctx) (parallel_generatedRoot_12.w_4_0 ctx)
  t9 : (1) = 1
  h10_0 : (∀ i : Fin 1, parallel_parallel_generatedRoot_12_10 backend hashModel params i_0 i () ((parallel_generatedRoot_12.w_10_0 ctx) i))
  t11 : 0 ≤ (1)
  h13_0 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0) => sequential_parallel_generatedRoot_12_13 backend hashModel params i_0 i (current, ((parallel_generatedRoot_12.w_10_0 ctx), ((parallel_generatedRoot_12.w_11_0 ctx), ((parallel_generatedRoot_12.w_12_0 ctx), ())))) next) (Int.toNat (1)) (parallel_generatedRoot_12.w_9_0 ctx) (parallel_generatedRoot_12.w_13_0 ctx)
  h17_0 : MxxRuntime.gadgetMatrixRuns backend 4096 25 (parallel_generatedRoot_12.w_17_0 ctx)
  hout : outputs = parallel_generatedRoot_12.outs ctx

theorem parallel_generatedRoot_12.facts_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} {ctx : parallel_generatedRoot_12.Ctx backend hashModel params i_0} {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_12.constraints_0 backend hashModel params i_0 (parallel_generatedRoot_12.w_0_0 ctx) (parallel_generatedRoot_12.w_3_0 ctx) (parallel_generatedRoot_12.w_1_0 ctx) (parallel_generatedRoot_12.w_2_0 ctx) (parallel_generatedRoot_12.w_4_0 ctx) (parallel_generatedRoot_12.w_9_0 ctx) (parallel_generatedRoot_12.w_10_0 ctx) (parallel_generatedRoot_12.w_11_0 ctx) (parallel_generatedRoot_12.w_12_0 ctx) (parallel_generatedRoot_12.w_13_0 ctx) (parallel_generatedRoot_12.w_17_0 ctx) (parallel_generatedRoot_12.w_21_0 ctx) (parallel_generatedRoot_12.w_30_0 ctx) outputs) :
    parallel_generatedRoot_12.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_12.Facts {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_12.Ctx backend hashModel params i_0) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  c0 : parallel_generatedRoot_12.Facts_0 ctx outputs

theorem parallel_generatedRoot_12.facts {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_12.Ctx backend hashModel params i_0) {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_12.body backend hashModel params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_12.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_12.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_13.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) where
  inputs : ByteArray × mxxMat0 × mxxMat0 × Unit
  witness : parallel_generatedRoot_13.Witness

def parallel_generatedRoot_13.w_0_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_13.Ctx backend hashModel params i_0) : ByteArray :=
  ctx.inputs.1

def parallel_generatedRoot_13.w_3_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_13.Ctx backend hashModel params i_0) : mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_13.w_25_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_13.Ctx backend hashModel params i_0) : mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_13.w_1_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_13.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_1_0

def parallel_generatedRoot_13.w_2_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_13.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_2_0

def parallel_generatedRoot_13.w_4_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_13.Ctx backend hashModel params i_0) : mxxMat0 :=
  ctx.witness.w_4_0

def parallel_generatedRoot_13.w_5_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_13.Ctx backend hashModel params i_0) : mxxMat0 :=
  MxxRuntime.matrixNeg (parallel_generatedRoot_13.w_4_0 ctx)

def parallel_generatedRoot_13.w_6_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_13.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_13.w_5_0 ctx) (parallel_generatedRoot_13.w_1_0 ctx)

def parallel_generatedRoot_13.w_7_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_13.Ctx backend hashModel params i_0) : mxxMat0 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat0)

def parallel_generatedRoot_13.w_8_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_13.Ctx backend hashModel params i_0) : mxxMat0 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat0)

def parallel_generatedRoot_13.w_9_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_13.Ctx backend hashModel params i_0) : mxxMat0 :=
  (0 : mxxMat0)

def parallel_generatedRoot_13.w_10_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_13.Ctx backend hashModel params i_0) : Fin 1 → Int :=
  ctx.witness.w_10_0

def parallel_generatedRoot_13.w_11_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_13.Ctx backend hashModel params i_0) : mxxMat0 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat0)

def parallel_generatedRoot_13.w_12_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_13.Ctx backend hashModel params i_0) : Int :=
  -1

def parallel_generatedRoot_13.w_13_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_13.Ctx backend hashModel params i_0) : mxxMat0 :=
  ctx.witness.w_13_0

def parallel_generatedRoot_13.w_14_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_13.Ctx backend hashModel params i_0) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_13.w_8_0 ctx) (parallel_generatedRoot_13.w_13_0 ctx)

def parallel_generatedRoot_13.w_15_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_13.Ctx backend hashModel params i_0) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_13.w_7_0 ctx) (parallel_generatedRoot_13.w_14_0 ctx)

def parallel_generatedRoot_13.w_16_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_13.Ctx backend hashModel params i_0) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_13.w_15_0 ctx) (parallel_generatedRoot_13.w_3_0 ctx)

def parallel_generatedRoot_13.w_17_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_13.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_13.w_18_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_13.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_13.w_16_0 ctx) (parallel_generatedRoot_13.w_17_0 ctx)

def parallel_generatedRoot_13.w_19_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_13.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_13.w_6_0 ctx) (parallel_generatedRoot_13.w_18_0 ctx)

def parallel_generatedRoot_13.w_20_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_13.Ctx backend hashModel params i_0) : mxxMat1 :=
  (0 : mxxMat1)

def parallel_generatedRoot_13.w_21_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_13.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_13.w_19_0 ctx) (parallel_generatedRoot_13.w_20_0 ctx)

def parallel_generatedRoot_13.w_22_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_13.Ctx backend hashModel params i_0) : mxxMat0 :=
  MxxRuntime.matrixNeg (parallel_generatedRoot_13.w_4_0 ctx)

def parallel_generatedRoot_13.w_23_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_13.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_13.w_22_0 ctx) (parallel_generatedRoot_13.w_2_0 ctx)

def parallel_generatedRoot_13.w_24_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_13.Ctx backend hashModel params i_0) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_13.w_7_0 ctx) (parallel_generatedRoot_13.w_14_0 ctx)

def parallel_generatedRoot_13.w_26_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_13.Ctx backend hashModel params i_0) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_13.w_24_0 ctx) (parallel_generatedRoot_13.w_25_0 ctx)

def parallel_generatedRoot_13.w_27_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_13.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_13.w_26_0 ctx) (parallel_generatedRoot_13.w_17_0 ctx)

def parallel_generatedRoot_13.w_28_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_13.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_13.w_23_0 ctx) (parallel_generatedRoot_13.w_27_0 ctx)

def parallel_generatedRoot_13.w_29_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_13.Ctx backend hashModel params i_0) : mxxMat1 :=
  (0 : mxxMat1)

def parallel_generatedRoot_13.w_30_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_13.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_13.w_28_0 ctx) (parallel_generatedRoot_13.w_29_0 ctx)

def parallel_generatedRoot_13.outs {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_13.Ctx backend hashModel params i_0) : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit :=
  ((parallel_generatedRoot_13.w_1_0 ctx), ((parallel_generatedRoot_13.w_2_0 ctx), ((parallel_generatedRoot_13.w_21_0 ctx), ((parallel_generatedRoot_13.w_30_0 ctx), ()))))

structure parallel_generatedRoot_13.Facts_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_13.Ctx backend hashModel params i_0) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  t0 : 0 ≤ ((Int.ofNat i_0))
  t1 : ((Int.ofNat i_0)) < 1
  h1_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 98, 47, 76, 117, 116, 73, 100, 40, 55, 41, 47, 51, 47, 52]) ([.bytes [115, 105, 103, 109, 97], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(9)])), .bytes [98, 114, 97, 110, 99, 104], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((parallel_generatedRoot_13.w_0_0 ctx)) (parallel_generatedRoot_13.w_1_0 ctx)
  t3 : 0 ≤ ((Int.ofNat i_0))
  t4 : ((Int.ofNat i_0)) < 1
  h2_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 98, 47, 76, 117, 116, 73, 100, 40, 55, 41, 47, 51, 47, 52]) ([.bytes [115, 105, 103, 109, 97], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(9)])), .bytes [98, 114, 97, 110, 99, 104], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((parallel_generatedRoot_13.w_0_0 ctx)) (parallel_generatedRoot_13.w_2_0 ctx)
  t6 : 0 ≤ ((Int.ofNat i_0))
  t7 : ((Int.ofNat i_0)) < 1
  h4_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(9)])) (parallel_generatedRoot_13.w_3_0 ctx) (parallel_generatedRoot_13.w_4_0 ctx)
  t9 : (1) = 1
  h10_0 : (∀ i : Fin 1, parallel_parallel_generatedRoot_13_10 backend hashModel params i_0 i () ((parallel_generatedRoot_13.w_10_0 ctx) i))
  t11 : 0 ≤ (1)
  h13_0 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0) => sequential_parallel_generatedRoot_13_13 backend hashModel params i_0 i (current, ((parallel_generatedRoot_13.w_10_0 ctx), ((parallel_generatedRoot_13.w_11_0 ctx), ((parallel_generatedRoot_13.w_12_0 ctx), ())))) next) (Int.toNat (1)) (parallel_generatedRoot_13.w_9_0 ctx) (parallel_generatedRoot_13.w_13_0 ctx)
  h17_0 : MxxRuntime.gadgetMatrixRuns backend 4096 25 (parallel_generatedRoot_13.w_17_0 ctx)
  hout : outputs = parallel_generatedRoot_13.outs ctx

theorem parallel_generatedRoot_13.facts_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} {ctx : parallel_generatedRoot_13.Ctx backend hashModel params i_0} {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_13.constraints_0 backend hashModel params i_0 (parallel_generatedRoot_13.w_0_0 ctx) (parallel_generatedRoot_13.w_3_0 ctx) (parallel_generatedRoot_13.w_1_0 ctx) (parallel_generatedRoot_13.w_2_0 ctx) (parallel_generatedRoot_13.w_4_0 ctx) (parallel_generatedRoot_13.w_9_0 ctx) (parallel_generatedRoot_13.w_10_0 ctx) (parallel_generatedRoot_13.w_11_0 ctx) (parallel_generatedRoot_13.w_12_0 ctx) (parallel_generatedRoot_13.w_13_0 ctx) (parallel_generatedRoot_13.w_17_0 ctx) (parallel_generatedRoot_13.w_21_0 ctx) (parallel_generatedRoot_13.w_30_0 ctx) outputs) :
    parallel_generatedRoot_13.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_13.Facts {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_13.Ctx backend hashModel params i_0) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  c0 : parallel_generatedRoot_13.Facts_0 ctx outputs

theorem parallel_generatedRoot_13.facts {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_13.Ctx backend hashModel params i_0) {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_13.body backend hashModel params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_13.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_13.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_14.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) where
  inputs : ByteArray × mxxMat0 × mxxMat0 × Unit
  witness : parallel_generatedRoot_14.Witness

def parallel_generatedRoot_14.w_0_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_14.Ctx backend hashModel params i_0) : ByteArray :=
  ctx.inputs.1

def parallel_generatedRoot_14.w_3_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_14.Ctx backend hashModel params i_0) : mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_14.w_25_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_14.Ctx backend hashModel params i_0) : mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_14.w_1_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_14.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_1_0

def parallel_generatedRoot_14.w_2_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_14.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_2_0

def parallel_generatedRoot_14.w_4_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_14.Ctx backend hashModel params i_0) : mxxMat0 :=
  ctx.witness.w_4_0

def parallel_generatedRoot_14.w_5_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_14.Ctx backend hashModel params i_0) : mxxMat0 :=
  MxxRuntime.matrixNeg (parallel_generatedRoot_14.w_4_0 ctx)

def parallel_generatedRoot_14.w_6_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_14.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_14.w_5_0 ctx) (parallel_generatedRoot_14.w_1_0 ctx)

def parallel_generatedRoot_14.w_7_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_14.Ctx backend hashModel params i_0) : mxxMat0 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat0)

def parallel_generatedRoot_14.w_8_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_14.Ctx backend hashModel params i_0) : mxxMat0 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat0)

def parallel_generatedRoot_14.w_9_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_14.Ctx backend hashModel params i_0) : mxxMat0 :=
  (0 : mxxMat0)

def parallel_generatedRoot_14.w_10_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_14.Ctx backend hashModel params i_0) : Fin 1 → Int :=
  ctx.witness.w_10_0

def parallel_generatedRoot_14.w_11_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_14.Ctx backend hashModel params i_0) : mxxMat0 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat0)

def parallel_generatedRoot_14.w_12_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_14.Ctx backend hashModel params i_0) : Int :=
  -1

def parallel_generatedRoot_14.w_13_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_14.Ctx backend hashModel params i_0) : mxxMat0 :=
  ctx.witness.w_13_0

def parallel_generatedRoot_14.w_14_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_14.Ctx backend hashModel params i_0) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_14.w_8_0 ctx) (parallel_generatedRoot_14.w_13_0 ctx)

def parallel_generatedRoot_14.w_15_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_14.Ctx backend hashModel params i_0) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_14.w_7_0 ctx) (parallel_generatedRoot_14.w_14_0 ctx)

def parallel_generatedRoot_14.w_16_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_14.Ctx backend hashModel params i_0) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_14.w_15_0 ctx) (parallel_generatedRoot_14.w_3_0 ctx)

def parallel_generatedRoot_14.w_17_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_14.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_14.w_18_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_14.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_14.w_16_0 ctx) (parallel_generatedRoot_14.w_17_0 ctx)

def parallel_generatedRoot_14.w_19_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_14.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_14.w_6_0 ctx) (parallel_generatedRoot_14.w_18_0 ctx)

def parallel_generatedRoot_14.w_20_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_14.Ctx backend hashModel params i_0) : mxxMat1 :=
  (0 : mxxMat1)

def parallel_generatedRoot_14.w_21_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_14.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_14.w_19_0 ctx) (parallel_generatedRoot_14.w_20_0 ctx)

def parallel_generatedRoot_14.w_22_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_14.Ctx backend hashModel params i_0) : mxxMat0 :=
  MxxRuntime.matrixNeg (parallel_generatedRoot_14.w_4_0 ctx)

def parallel_generatedRoot_14.w_23_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_14.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_14.w_22_0 ctx) (parallel_generatedRoot_14.w_2_0 ctx)

def parallel_generatedRoot_14.w_24_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_14.Ctx backend hashModel params i_0) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_14.w_7_0 ctx) (parallel_generatedRoot_14.w_14_0 ctx)

def parallel_generatedRoot_14.w_26_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_14.Ctx backend hashModel params i_0) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_14.w_24_0 ctx) (parallel_generatedRoot_14.w_25_0 ctx)

def parallel_generatedRoot_14.w_27_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_14.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_14.w_26_0 ctx) (parallel_generatedRoot_14.w_17_0 ctx)

def parallel_generatedRoot_14.w_28_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_14.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_14.w_23_0 ctx) (parallel_generatedRoot_14.w_27_0 ctx)

def parallel_generatedRoot_14.w_29_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_14.Ctx backend hashModel params i_0) : mxxMat1 :=
  (0 : mxxMat1)

def parallel_generatedRoot_14.w_30_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_14.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_14.w_28_0 ctx) (parallel_generatedRoot_14.w_29_0 ctx)

def parallel_generatedRoot_14.outs {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_14.Ctx backend hashModel params i_0) : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit :=
  ((parallel_generatedRoot_14.w_1_0 ctx), ((parallel_generatedRoot_14.w_2_0 ctx), ((parallel_generatedRoot_14.w_21_0 ctx), ((parallel_generatedRoot_14.w_30_0 ctx), ()))))

structure parallel_generatedRoot_14.Facts_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_14.Ctx backend hashModel params i_0) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  t0 : 0 ≤ ((Int.ofNat i_0))
  t1 : ((Int.ofNat i_0)) < 1
  h1_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 98, 47, 76, 117, 116, 73, 100, 40, 56, 41, 47, 51, 47, 52]) ([.bytes [115, 105, 103, 109, 97], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(11)])), .bytes [98, 114, 97, 110, 99, 104], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((parallel_generatedRoot_14.w_0_0 ctx)) (parallel_generatedRoot_14.w_1_0 ctx)
  t3 : 0 ≤ ((Int.ofNat i_0))
  t4 : ((Int.ofNat i_0)) < 1
  h2_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 98, 47, 76, 117, 116, 73, 100, 40, 56, 41, 47, 51, 47, 52]) ([.bytes [115, 105, 103, 109, 97], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(11)])), .bytes [98, 114, 97, 110, 99, 104], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((parallel_generatedRoot_14.w_0_0 ctx)) (parallel_generatedRoot_14.w_2_0 ctx)
  t6 : 0 ≤ ((Int.ofNat i_0))
  t7 : ((Int.ofNat i_0)) < 1
  h4_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(11)])) (parallel_generatedRoot_14.w_3_0 ctx) (parallel_generatedRoot_14.w_4_0 ctx)
  t9 : (1) = 1
  h10_0 : (∀ i : Fin 1, parallel_parallel_generatedRoot_14_10 backend hashModel params i_0 i () ((parallel_generatedRoot_14.w_10_0 ctx) i))
  t11 : 0 ≤ (1)
  h13_0 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0) => sequential_parallel_generatedRoot_14_13 backend hashModel params i_0 i (current, ((parallel_generatedRoot_14.w_10_0 ctx), ((parallel_generatedRoot_14.w_11_0 ctx), ((parallel_generatedRoot_14.w_12_0 ctx), ())))) next) (Int.toNat (1)) (parallel_generatedRoot_14.w_9_0 ctx) (parallel_generatedRoot_14.w_13_0 ctx)
  h17_0 : MxxRuntime.gadgetMatrixRuns backend 4096 25 (parallel_generatedRoot_14.w_17_0 ctx)
  hout : outputs = parallel_generatedRoot_14.outs ctx

theorem parallel_generatedRoot_14.facts_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} {ctx : parallel_generatedRoot_14.Ctx backend hashModel params i_0} {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_14.constraints_0 backend hashModel params i_0 (parallel_generatedRoot_14.w_0_0 ctx) (parallel_generatedRoot_14.w_3_0 ctx) (parallel_generatedRoot_14.w_1_0 ctx) (parallel_generatedRoot_14.w_2_0 ctx) (parallel_generatedRoot_14.w_4_0 ctx) (parallel_generatedRoot_14.w_9_0 ctx) (parallel_generatedRoot_14.w_10_0 ctx) (parallel_generatedRoot_14.w_11_0 ctx) (parallel_generatedRoot_14.w_12_0 ctx) (parallel_generatedRoot_14.w_13_0 ctx) (parallel_generatedRoot_14.w_17_0 ctx) (parallel_generatedRoot_14.w_21_0 ctx) (parallel_generatedRoot_14.w_30_0 ctx) outputs) :
    parallel_generatedRoot_14.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_14.Facts {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_14.Ctx backend hashModel params i_0) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  c0 : parallel_generatedRoot_14.Facts_0 ctx outputs

theorem parallel_generatedRoot_14.facts {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_14.Ctx backend hashModel params i_0) {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_14.body backend hashModel params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_14.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_14.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_15.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) where
  inputs : ByteArray × mxxMat0 × mxxMat0 × Unit
  witness : parallel_generatedRoot_15.Witness

def parallel_generatedRoot_15.w_0_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_15.Ctx backend hashModel params i_0) : ByteArray :=
  ctx.inputs.1

def parallel_generatedRoot_15.w_3_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_15.Ctx backend hashModel params i_0) : mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_15.w_25_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_15.Ctx backend hashModel params i_0) : mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_15.w_1_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_15.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_1_0

def parallel_generatedRoot_15.w_2_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_15.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_2_0

def parallel_generatedRoot_15.w_4_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_15.Ctx backend hashModel params i_0) : mxxMat0 :=
  ctx.witness.w_4_0

def parallel_generatedRoot_15.w_5_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_15.Ctx backend hashModel params i_0) : mxxMat0 :=
  MxxRuntime.matrixNeg (parallel_generatedRoot_15.w_4_0 ctx)

def parallel_generatedRoot_15.w_6_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_15.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_15.w_5_0 ctx) (parallel_generatedRoot_15.w_1_0 ctx)

def parallel_generatedRoot_15.w_7_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_15.Ctx backend hashModel params i_0) : mxxMat0 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat0)

def parallel_generatedRoot_15.w_8_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_15.Ctx backend hashModel params i_0) : mxxMat0 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat0)

def parallel_generatedRoot_15.w_9_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_15.Ctx backend hashModel params i_0) : mxxMat0 :=
  (0 : mxxMat0)

def parallel_generatedRoot_15.w_10_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_15.Ctx backend hashModel params i_0) : Fin 1 → Int :=
  ctx.witness.w_10_0

def parallel_generatedRoot_15.w_11_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_15.Ctx backend hashModel params i_0) : mxxMat0 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat0)

def parallel_generatedRoot_15.w_12_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_15.Ctx backend hashModel params i_0) : Int :=
  -1

def parallel_generatedRoot_15.w_13_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_15.Ctx backend hashModel params i_0) : mxxMat0 :=
  ctx.witness.w_13_0

def parallel_generatedRoot_15.w_14_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_15.Ctx backend hashModel params i_0) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_15.w_8_0 ctx) (parallel_generatedRoot_15.w_13_0 ctx)

def parallel_generatedRoot_15.w_15_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_15.Ctx backend hashModel params i_0) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_15.w_7_0 ctx) (parallel_generatedRoot_15.w_14_0 ctx)

def parallel_generatedRoot_15.w_16_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_15.Ctx backend hashModel params i_0) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_15.w_15_0 ctx) (parallel_generatedRoot_15.w_3_0 ctx)

def parallel_generatedRoot_15.w_17_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_15.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_15.w_18_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_15.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_15.w_16_0 ctx) (parallel_generatedRoot_15.w_17_0 ctx)

def parallel_generatedRoot_15.w_19_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_15.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_15.w_6_0 ctx) (parallel_generatedRoot_15.w_18_0 ctx)

def parallel_generatedRoot_15.w_20_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_15.Ctx backend hashModel params i_0) : mxxMat1 :=
  (0 : mxxMat1)

def parallel_generatedRoot_15.w_21_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_15.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_15.w_19_0 ctx) (parallel_generatedRoot_15.w_20_0 ctx)

def parallel_generatedRoot_15.w_22_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_15.Ctx backend hashModel params i_0) : mxxMat0 :=
  MxxRuntime.matrixNeg (parallel_generatedRoot_15.w_4_0 ctx)

def parallel_generatedRoot_15.w_23_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_15.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_15.w_22_0 ctx) (parallel_generatedRoot_15.w_2_0 ctx)

def parallel_generatedRoot_15.w_24_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_15.Ctx backend hashModel params i_0) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_15.w_7_0 ctx) (parallel_generatedRoot_15.w_14_0 ctx)

def parallel_generatedRoot_15.w_26_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_15.Ctx backend hashModel params i_0) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_15.w_24_0 ctx) (parallel_generatedRoot_15.w_25_0 ctx)

def parallel_generatedRoot_15.w_27_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_15.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_15.w_26_0 ctx) (parallel_generatedRoot_15.w_17_0 ctx)

def parallel_generatedRoot_15.w_28_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_15.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_15.w_23_0 ctx) (parallel_generatedRoot_15.w_27_0 ctx)

def parallel_generatedRoot_15.w_29_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_15.Ctx backend hashModel params i_0) : mxxMat1 :=
  (0 : mxxMat1)

def parallel_generatedRoot_15.w_30_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_15.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_15.w_28_0 ctx) (parallel_generatedRoot_15.w_29_0 ctx)

def parallel_generatedRoot_15.outs {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_15.Ctx backend hashModel params i_0) : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit :=
  ((parallel_generatedRoot_15.w_1_0 ctx), ((parallel_generatedRoot_15.w_2_0 ctx), ((parallel_generatedRoot_15.w_21_0 ctx), ((parallel_generatedRoot_15.w_30_0 ctx), ()))))

structure parallel_generatedRoot_15.Facts_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_15.Ctx backend hashModel params i_0) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  t0 : 0 ≤ ((Int.ofNat i_0))
  t1 : ((Int.ofNat i_0)) < 1
  h1_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 98, 47, 76, 117, 116, 73, 100, 40, 57, 41, 47, 51, 47, 52]) ([.bytes [115, 105, 103, 109, 97], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(13)])), .bytes [98, 114, 97, 110, 99, 104], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((parallel_generatedRoot_15.w_0_0 ctx)) (parallel_generatedRoot_15.w_1_0 ctx)
  t3 : 0 ≤ ((Int.ofNat i_0))
  t4 : ((Int.ofNat i_0)) < 1
  h2_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 98, 47, 76, 117, 116, 73, 100, 40, 57, 41, 47, 51, 47, 52]) ([.bytes [115, 105, 103, 109, 97], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(13)])), .bytes [98, 114, 97, 110, 99, 104], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((parallel_generatedRoot_15.w_0_0 ctx)) (parallel_generatedRoot_15.w_2_0 ctx)
  t6 : 0 ≤ ((Int.ofNat i_0))
  t7 : ((Int.ofNat i_0)) < 1
  h4_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(13)])) (parallel_generatedRoot_15.w_3_0 ctx) (parallel_generatedRoot_15.w_4_0 ctx)
  t9 : (1) = 1
  h10_0 : (∀ i : Fin 1, parallel_parallel_generatedRoot_15_10 backend hashModel params i_0 i () ((parallel_generatedRoot_15.w_10_0 ctx) i))
  t11 : 0 ≤ (1)
  h13_0 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0) => sequential_parallel_generatedRoot_15_13 backend hashModel params i_0 i (current, ((parallel_generatedRoot_15.w_10_0 ctx), ((parallel_generatedRoot_15.w_11_0 ctx), ((parallel_generatedRoot_15.w_12_0 ctx), ())))) next) (Int.toNat (1)) (parallel_generatedRoot_15.w_9_0 ctx) (parallel_generatedRoot_15.w_13_0 ctx)
  h17_0 : MxxRuntime.gadgetMatrixRuns backend 4096 25 (parallel_generatedRoot_15.w_17_0 ctx)
  hout : outputs = parallel_generatedRoot_15.outs ctx

theorem parallel_generatedRoot_15.facts_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} {ctx : parallel_generatedRoot_15.Ctx backend hashModel params i_0} {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_15.constraints_0 backend hashModel params i_0 (parallel_generatedRoot_15.w_0_0 ctx) (parallel_generatedRoot_15.w_3_0 ctx) (parallel_generatedRoot_15.w_1_0 ctx) (parallel_generatedRoot_15.w_2_0 ctx) (parallel_generatedRoot_15.w_4_0 ctx) (parallel_generatedRoot_15.w_9_0 ctx) (parallel_generatedRoot_15.w_10_0 ctx) (parallel_generatedRoot_15.w_11_0 ctx) (parallel_generatedRoot_15.w_12_0 ctx) (parallel_generatedRoot_15.w_13_0 ctx) (parallel_generatedRoot_15.w_17_0 ctx) (parallel_generatedRoot_15.w_21_0 ctx) (parallel_generatedRoot_15.w_30_0 ctx) outputs) :
    parallel_generatedRoot_15.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_15.Facts {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_15.Ctx backend hashModel params i_0) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  c0 : parallel_generatedRoot_15.Facts_0 ctx outputs

theorem parallel_generatedRoot_15.facts {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_15.Ctx backend hashModel params i_0) {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_15.body backend hashModel params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_15.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_15.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_80.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) where
  inputs : (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × Unit
  witness : parallel_generatedRoot_80.Witness

def parallel_generatedRoot_80.w_0_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_80.Ctx backend hashModel params i_0) : Fin 72000 → mxxMat1 :=
  ctx.inputs.1

def parallel_generatedRoot_80.w_1_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_80.Ctx backend hashModel params i_0) : Fin 72000 → mxxMat1 :=
  ctx.inputs.2.1

def parallel_generatedRoot_80.w_2_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_80.Ctx backend hashModel params i_0) : Fin 72000 → mxxMat1 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_80.w_3_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_80.Ctx backend hashModel params i_0) : Fin 72000 → mxxMat1 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_80.w_4_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_80.Ctx backend hashModel params i_0) : Fin 2 → mxxMat1 :=
  ctx.witness.w_4_0

def parallel_generatedRoot_80.w_4_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_80.Ctx backend hashModel params i_0) : Fin 2 → mxxMat1 :=
  ctx.witness.w_4_1

def parallel_generatedRoot_80.w_4_2 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_80.Ctx backend hashModel params i_0) : Fin 2 → mxxMat1 :=
  ctx.witness.w_4_2

def parallel_generatedRoot_80.w_4_3 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_80.Ctx backend hashModel params i_0) : Fin 2 → mxxMat1 :=
  ctx.witness.w_4_3

def parallel_generatedRoot_80.w_5_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_80.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_5_0

def parallel_generatedRoot_80.w_6_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_80.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_80.w_7_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_80.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_80.w_5_0 ctx) (parallel_generatedRoot_80.w_6_0 ctx)

def parallel_generatedRoot_80.w_8_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_80.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_80.w_9_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_80.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_9_0

def parallel_generatedRoot_80.w_10_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_80.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_80.w_8_0 ctx) (parallel_generatedRoot_80.w_9_0 ctx)

def parallel_generatedRoot_80.w_11_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_80.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_80.w_12_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_80.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_12_0

def parallel_generatedRoot_80.w_13_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_80.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_80.w_11_0 ctx) (parallel_generatedRoot_80.w_12_0 ctx)

def parallel_generatedRoot_80.w_14_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_80.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_80.w_15_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_80.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_15_0

def parallel_generatedRoot_80.w_16_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_80.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_80.w_14_0 ctx) (parallel_generatedRoot_80.w_15_0 ctx)

def parallel_generatedRoot_80.w_17_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_80.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_80.w_17_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_80.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_17_1

def parallel_generatedRoot_80.w_17_2 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_80.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_17_2

def parallel_generatedRoot_80.w_17_3 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_80.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_17_3

def parallel_generatedRoot_80.outs {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_80.Ctx backend hashModel params i_0) : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit :=
  ((parallel_generatedRoot_80.w_17_0 ctx), ((parallel_generatedRoot_80.w_17_1 ctx), ((parallel_generatedRoot_80.w_17_2 ctx), ((parallel_generatedRoot_80.w_17_3 ctx), ()))))

structure parallel_generatedRoot_80.Facts_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_80.Ctx backend hashModel params i_0) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  t0 : (2) = 2
  h4_3 : (∀ i : Fin 2, parallel_parallel_generatedRoot_80_4 backend hashModel params i_0 i ((parallel_generatedRoot_80.w_0_0 ctx), ((parallel_generatedRoot_80.w_1_0 ctx), ((parallel_generatedRoot_80.w_2_0 ctx), ((parallel_generatedRoot_80.w_3_0 ctx), ())))) (((parallel_generatedRoot_80.w_4_0 ctx) i), (((parallel_generatedRoot_80.w_4_1 ctx) i), (((parallel_generatedRoot_80.w_4_2 ctx) i), (((parallel_generatedRoot_80.w_4_3 ctx) i), ())))))
  h5_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_80.w_4_0 ctx) (0) (parallel_generatedRoot_80.w_5_0 ctx)
  h6_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_80.w_4_0 ctx) (1) (parallel_generatedRoot_80.w_6_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_80.w_4_1 ctx) (0) (parallel_generatedRoot_80.w_8_0 ctx)
  h9_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_80.w_4_1 ctx) (1) (parallel_generatedRoot_80.w_9_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_80.w_4_2 ctx) (0) (parallel_generatedRoot_80.w_11_0 ctx)
  h12_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_80.w_4_2 ctx) (1) (parallel_generatedRoot_80.w_12_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_80.w_4_3 ctx) (0) (parallel_generatedRoot_80.w_14_0 ctx)
  h15_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_80.w_4_3 ctx) (1) (parallel_generatedRoot_80.w_15_0 ctx)
  t10 : 0 ≤ (0)
  h17_3 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_parallel_generatedRoot_80_17 backend hashModel params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, ((parallel_generatedRoot_80.w_4_0 ctx), ((parallel_generatedRoot_80.w_4_1 ctx), ((parallel_generatedRoot_80.w_4_2 ctx), ((parallel_generatedRoot_80.w_4_3 ctx), ())))))))) next) (Int.toNat (0)) ((parallel_generatedRoot_80.w_7_0 ctx), ((parallel_generatedRoot_80.w_10_0 ctx), ((parallel_generatedRoot_80.w_13_0 ctx), ((parallel_generatedRoot_80.w_16_0 ctx), ())))) ((parallel_generatedRoot_80.w_17_0 ctx), ((parallel_generatedRoot_80.w_17_1 ctx), ((parallel_generatedRoot_80.w_17_2 ctx), ((parallel_generatedRoot_80.w_17_3 ctx), ()))))
  hout : outputs = parallel_generatedRoot_80.outs ctx

theorem parallel_generatedRoot_80.facts_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} {ctx : parallel_generatedRoot_80.Ctx backend hashModel params i_0} {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_80.constraints_0 backend hashModel params i_0 (parallel_generatedRoot_80.w_0_0 ctx) (parallel_generatedRoot_80.w_1_0 ctx) (parallel_generatedRoot_80.w_2_0 ctx) (parallel_generatedRoot_80.w_3_0 ctx) (parallel_generatedRoot_80.w_4_0 ctx) (parallel_generatedRoot_80.w_4_1 ctx) (parallel_generatedRoot_80.w_4_2 ctx) (parallel_generatedRoot_80.w_4_3 ctx) (parallel_generatedRoot_80.w_5_0 ctx) (parallel_generatedRoot_80.w_6_0 ctx) (parallel_generatedRoot_80.w_7_0 ctx) (parallel_generatedRoot_80.w_8_0 ctx) (parallel_generatedRoot_80.w_9_0 ctx) (parallel_generatedRoot_80.w_10_0 ctx) (parallel_generatedRoot_80.w_11_0 ctx) (parallel_generatedRoot_80.w_12_0 ctx) (parallel_generatedRoot_80.w_13_0 ctx) (parallel_generatedRoot_80.w_14_0 ctx) (parallel_generatedRoot_80.w_15_0 ctx) (parallel_generatedRoot_80.w_16_0 ctx) (parallel_generatedRoot_80.w_17_0 ctx) (parallel_generatedRoot_80.w_17_1 ctx) (parallel_generatedRoot_80.w_17_2 ctx) (parallel_generatedRoot_80.w_17_3 ctx) outputs) :
    parallel_generatedRoot_80.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_80.Facts {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_80.Ctx backend hashModel params i_0) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  c0 : parallel_generatedRoot_80.Facts_0 ctx outputs

theorem parallel_generatedRoot_80.facts {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_80.Ctx backend hashModel params i_0) {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_80.body backend hashModel params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_80.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_80.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_84.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) where
  inputs : (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × Unit
  witness : parallel_generatedRoot_84.Witness

def parallel_generatedRoot_84.w_0_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_84.Ctx backend hashModel params i_0) : Fin 72000 → mxxMat1 :=
  ctx.inputs.1

def parallel_generatedRoot_84.w_1_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_84.Ctx backend hashModel params i_0) : Fin 72000 → mxxMat1 :=
  ctx.inputs.2.1

def parallel_generatedRoot_84.w_2_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_84.Ctx backend hashModel params i_0) : Fin 72000 → mxxMat1 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_84.w_3_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_84.Ctx backend hashModel params i_0) : Fin 72000 → mxxMat1 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_84.w_4_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_84.Ctx backend hashModel params i_0) : Fin 2 → mxxMat1 :=
  ctx.witness.w_4_0

def parallel_generatedRoot_84.w_4_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_84.Ctx backend hashModel params i_0) : Fin 2 → mxxMat1 :=
  ctx.witness.w_4_1

def parallel_generatedRoot_84.w_4_2 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_84.Ctx backend hashModel params i_0) : Fin 2 → mxxMat1 :=
  ctx.witness.w_4_2

def parallel_generatedRoot_84.w_4_3 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_84.Ctx backend hashModel params i_0) : Fin 2 → mxxMat1 :=
  ctx.witness.w_4_3

def parallel_generatedRoot_84.w_5_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_84.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_5_0

def parallel_generatedRoot_84.w_6_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_84.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_84.w_7_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_84.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_84.w_5_0 ctx) (parallel_generatedRoot_84.w_6_0 ctx)

def parallel_generatedRoot_84.w_8_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_84.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_84.w_9_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_84.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_9_0

def parallel_generatedRoot_84.w_10_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_84.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_84.w_8_0 ctx) (parallel_generatedRoot_84.w_9_0 ctx)

def parallel_generatedRoot_84.w_11_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_84.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_84.w_12_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_84.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_12_0

def parallel_generatedRoot_84.w_13_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_84.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_84.w_11_0 ctx) (parallel_generatedRoot_84.w_12_0 ctx)

def parallel_generatedRoot_84.w_14_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_84.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_84.w_15_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_84.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_15_0

def parallel_generatedRoot_84.w_16_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_84.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_84.w_14_0 ctx) (parallel_generatedRoot_84.w_15_0 ctx)

def parallel_generatedRoot_84.w_17_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_84.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_84.w_17_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_84.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_17_1

def parallel_generatedRoot_84.w_17_2 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_84.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_17_2

def parallel_generatedRoot_84.w_17_3 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_84.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_17_3

def parallel_generatedRoot_84.outs {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_84.Ctx backend hashModel params i_0) : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit :=
  ((parallel_generatedRoot_84.w_17_0 ctx), ((parallel_generatedRoot_84.w_17_1 ctx), ((parallel_generatedRoot_84.w_17_2 ctx), ((parallel_generatedRoot_84.w_17_3 ctx), ()))))

structure parallel_generatedRoot_84.Facts_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_84.Ctx backend hashModel params i_0) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  t0 : (2) = 2
  h4_3 : (∀ i : Fin 2, parallel_parallel_generatedRoot_84_4 backend hashModel params i_0 i ((parallel_generatedRoot_84.w_0_0 ctx), ((parallel_generatedRoot_84.w_1_0 ctx), ((parallel_generatedRoot_84.w_2_0 ctx), ((parallel_generatedRoot_84.w_3_0 ctx), ())))) (((parallel_generatedRoot_84.w_4_0 ctx) i), (((parallel_generatedRoot_84.w_4_1 ctx) i), (((parallel_generatedRoot_84.w_4_2 ctx) i), (((parallel_generatedRoot_84.w_4_3 ctx) i), ())))))
  h5_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_84.w_4_0 ctx) (0) (parallel_generatedRoot_84.w_5_0 ctx)
  h6_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_84.w_4_0 ctx) (1) (parallel_generatedRoot_84.w_6_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_84.w_4_1 ctx) (0) (parallel_generatedRoot_84.w_8_0 ctx)
  h9_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_84.w_4_1 ctx) (1) (parallel_generatedRoot_84.w_9_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_84.w_4_2 ctx) (0) (parallel_generatedRoot_84.w_11_0 ctx)
  h12_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_84.w_4_2 ctx) (1) (parallel_generatedRoot_84.w_12_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_84.w_4_3 ctx) (0) (parallel_generatedRoot_84.w_14_0 ctx)
  h15_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_84.w_4_3 ctx) (1) (parallel_generatedRoot_84.w_15_0 ctx)
  t10 : 0 ≤ (0)
  h17_3 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_parallel_generatedRoot_84_17 backend hashModel params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, ((parallel_generatedRoot_84.w_4_0 ctx), ((parallel_generatedRoot_84.w_4_1 ctx), ((parallel_generatedRoot_84.w_4_2 ctx), ((parallel_generatedRoot_84.w_4_3 ctx), ())))))))) next) (Int.toNat (0)) ((parallel_generatedRoot_84.w_7_0 ctx), ((parallel_generatedRoot_84.w_10_0 ctx), ((parallel_generatedRoot_84.w_13_0 ctx), ((parallel_generatedRoot_84.w_16_0 ctx), ())))) ((parallel_generatedRoot_84.w_17_0 ctx), ((parallel_generatedRoot_84.w_17_1 ctx), ((parallel_generatedRoot_84.w_17_2 ctx), ((parallel_generatedRoot_84.w_17_3 ctx), ()))))
  hout : outputs = parallel_generatedRoot_84.outs ctx

theorem parallel_generatedRoot_84.facts_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} {ctx : parallel_generatedRoot_84.Ctx backend hashModel params i_0} {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_84.constraints_0 backend hashModel params i_0 (parallel_generatedRoot_84.w_0_0 ctx) (parallel_generatedRoot_84.w_1_0 ctx) (parallel_generatedRoot_84.w_2_0 ctx) (parallel_generatedRoot_84.w_3_0 ctx) (parallel_generatedRoot_84.w_4_0 ctx) (parallel_generatedRoot_84.w_4_1 ctx) (parallel_generatedRoot_84.w_4_2 ctx) (parallel_generatedRoot_84.w_4_3 ctx) (parallel_generatedRoot_84.w_5_0 ctx) (parallel_generatedRoot_84.w_6_0 ctx) (parallel_generatedRoot_84.w_7_0 ctx) (parallel_generatedRoot_84.w_8_0 ctx) (parallel_generatedRoot_84.w_9_0 ctx) (parallel_generatedRoot_84.w_10_0 ctx) (parallel_generatedRoot_84.w_11_0 ctx) (parallel_generatedRoot_84.w_12_0 ctx) (parallel_generatedRoot_84.w_13_0 ctx) (parallel_generatedRoot_84.w_14_0 ctx) (parallel_generatedRoot_84.w_15_0 ctx) (parallel_generatedRoot_84.w_16_0 ctx) (parallel_generatedRoot_84.w_17_0 ctx) (parallel_generatedRoot_84.w_17_1 ctx) (parallel_generatedRoot_84.w_17_2 ctx) (parallel_generatedRoot_84.w_17_3 ctx) outputs) :
    parallel_generatedRoot_84.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_84.Facts {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_84.Ctx backend hashModel params i_0) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  c0 : parallel_generatedRoot_84.Facts_0 ctx outputs

theorem parallel_generatedRoot_84.facts {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_84.Ctx backend hashModel params i_0) {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_84.body backend hashModel params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_84.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_84.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_85.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) where
  inputs : ByteArray × mxxMat1 × mxxMat1 × (Fin 8 → mxxMat1) × mxxMat1 × mxxMat1 × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × Unit
  witness : parallel_generatedRoot_85.Witness

def parallel_generatedRoot_85.w_1_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_85.Ctx backend hashModel params i_0) : ByteArray :=
  ctx.inputs.1

def parallel_generatedRoot_85.w_2_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_85.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.inputs.2.1

def parallel_generatedRoot_85.w_3_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_85.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_85.w_4_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_85.Ctx backend hashModel params i_0) : Fin 8 → mxxMat1 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_85.w_5_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_85.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.inputs.2.2.2.2.1

def parallel_generatedRoot_85.w_6_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_85.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.inputs.2.2.2.2.2.1

def parallel_generatedRoot_85.w_7_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_85.Ctx backend hashModel params i_0) : Fin 8 → mxxMat1 :=
  ctx.inputs.2.2.2.2.2.2.1

def parallel_generatedRoot_85.w_8_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_85.Ctx backend hashModel params i_0) : Fin 8 → mxxMat1 :=
  ctx.inputs.2.2.2.2.2.2.2.1

def parallel_generatedRoot_85.w_9_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_85.Ctx backend hashModel params i_0) : Fin 8 → mxxMat1 :=
  ctx.inputs.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_85.w_0_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_85.Ctx backend hashModel params i_0) : mxxMat1 :=
  (0 : mxxMat1)

def parallel_generatedRoot_85.w_10_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_85.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_10_0

def parallel_generatedRoot_85.w_10_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_85.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_10_1

def parallel_generatedRoot_85.w_10_2 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_85.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_10_2

def parallel_generatedRoot_85.w_10_3 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_85.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_10_3

def parallel_generatedRoot_85.outs {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_85.Ctx backend hashModel params i_0) : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit :=
  ((parallel_generatedRoot_85.w_10_0 ctx), ((parallel_generatedRoot_85.w_10_1 ctx), ((parallel_generatedRoot_85.w_10_2 ctx), ((parallel_generatedRoot_85.w_10_3 ctx), ()))))

structure parallel_generatedRoot_85.Facts_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_85.Ctx backend hashModel params i_0) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  t0 : 0 ≤ (8)
  h10_3 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_parallel_generatedRoot_85_10 backend hashModel params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, ((parallel_generatedRoot_85.w_1_0 ctx), ((parallel_generatedRoot_85.w_2_0 ctx), ((parallel_generatedRoot_85.w_3_0 ctx), ((parallel_generatedRoot_85.w_4_0 ctx), ((parallel_generatedRoot_85.w_5_0 ctx), ((parallel_generatedRoot_85.w_6_0 ctx), ((parallel_generatedRoot_85.w_7_0 ctx), ((parallel_generatedRoot_85.w_8_0 ctx), ((parallel_generatedRoot_85.w_9_0 ctx), ()))))))))))))) next) (Int.toNat (8)) ((parallel_generatedRoot_85.w_0_0 ctx), ((parallel_generatedRoot_85.w_0_0 ctx), ((parallel_generatedRoot_85.w_0_0 ctx), ((parallel_generatedRoot_85.w_0_0 ctx), ())))) ((parallel_generatedRoot_85.w_10_0 ctx), ((parallel_generatedRoot_85.w_10_1 ctx), ((parallel_generatedRoot_85.w_10_2 ctx), ((parallel_generatedRoot_85.w_10_3 ctx), ()))))
  hout : outputs = parallel_generatedRoot_85.outs ctx

theorem parallel_generatedRoot_85.facts_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} {ctx : parallel_generatedRoot_85.Ctx backend hashModel params i_0} {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_85.constraints_0 backend hashModel params i_0 (parallel_generatedRoot_85.w_1_0 ctx) (parallel_generatedRoot_85.w_2_0 ctx) (parallel_generatedRoot_85.w_3_0 ctx) (parallel_generatedRoot_85.w_4_0 ctx) (parallel_generatedRoot_85.w_5_0 ctx) (parallel_generatedRoot_85.w_6_0 ctx) (parallel_generatedRoot_85.w_7_0 ctx) (parallel_generatedRoot_85.w_8_0 ctx) (parallel_generatedRoot_85.w_9_0 ctx) (parallel_generatedRoot_85.w_0_0 ctx) (parallel_generatedRoot_85.w_10_0 ctx) (parallel_generatedRoot_85.w_10_1 ctx) (parallel_generatedRoot_85.w_10_2 ctx) (parallel_generatedRoot_85.w_10_3 ctx) outputs) :
    parallel_generatedRoot_85.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure parallel_generatedRoot_85.Facts {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_85.Ctx backend hashModel params i_0) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  c0 : parallel_generatedRoot_85.Facts_0 ctx outputs

theorem parallel_generatedRoot_85.facts {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_85.Ctx backend hashModel params i_0) {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_85.body backend hashModel params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_85.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_85.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_107.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) where
  inputs : mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × (Fin 36000 → mxxMat1) × (Fin 36000 → mxxMat1) × (Fin 36000 → mxxMat1) × (Fin 36000 → mxxMat1) × Unit
  witness : sequential_generatedRoot_107.Witness

def sequential_generatedRoot_107.w_0_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_107.Ctx backend hashModel params i_0) : mxxMat2 :=
  ctx.inputs.1

def sequential_generatedRoot_107.w_23_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_107.Ctx backend hashModel params i_0) : mxxMat2 :=
  ctx.inputs.2.1

def sequential_generatedRoot_107.w_27_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_107.Ctx backend hashModel params i_0) : mxxMat2 :=
  ctx.inputs.2.2.1

def sequential_generatedRoot_107.w_31_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_107.Ctx backend hashModel params i_0) : mxxMat2 :=
  ctx.inputs.2.2.2.1

def sequential_generatedRoot_107.w_2_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_107.Ctx backend hashModel params i_0) : Fin 36000 → mxxMat1 :=
  ctx.inputs.2.2.2.2.1

def sequential_generatedRoot_107.w_8_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_107.Ctx backend hashModel params i_0) : Fin 36000 → mxxMat1 :=
  ctx.inputs.2.2.2.2.2.1

def sequential_generatedRoot_107.w_12_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_107.Ctx backend hashModel params i_0) : Fin 36000 → mxxMat1 :=
  ctx.inputs.2.2.2.2.2.2.1

def sequential_generatedRoot_107.w_16_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_107.Ctx backend hashModel params i_0) : Fin 36000 → mxxMat1 :=
  ctx.inputs.2.2.2.2.2.2.2.1

def sequential_generatedRoot_107.w_1_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_107.Ctx backend hashModel params i_0) : mxxMat3 :=
  (MxxRuntime.unitRow ((1 + (Int.ofNat i_0))) : mxxMat3)

def sequential_generatedRoot_107.w_3_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_107.Ctx backend hashModel params i_0) : Int :=
  (8 + (8 * (Int.ofNat i_0)))

def sequential_generatedRoot_107.w_4_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_107.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_4_0

def sequential_generatedRoot_107.w_5_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_107.Ctx backend hashModel params i_0) : Int :=
  (9 + (8 * (Int.ofNat i_0)))

def sequential_generatedRoot_107.w_6_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_107.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_6_0

def sequential_generatedRoot_107.w_7_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_107.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixAdd (sequential_generatedRoot_107.w_4_0 ctx) (sequential_generatedRoot_107.w_6_0 ctx)

def sequential_generatedRoot_107.w_9_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_107.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_9_0

def sequential_generatedRoot_107.w_10_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_107.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_10_0

def sequential_generatedRoot_107.w_11_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_107.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixAdd (sequential_generatedRoot_107.w_9_0 ctx) (sequential_generatedRoot_107.w_10_0 ctx)

def sequential_generatedRoot_107.w_13_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_107.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_13_0

def sequential_generatedRoot_107.w_14_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_107.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_14_0

def sequential_generatedRoot_107.w_15_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_107.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixAdd (sequential_generatedRoot_107.w_13_0 ctx) (sequential_generatedRoot_107.w_14_0 ctx)

def sequential_generatedRoot_107.w_17_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_107.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_17_0

def sequential_generatedRoot_107.w_18_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_107.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_18_0

def sequential_generatedRoot_107.w_19_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_107.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixAdd (sequential_generatedRoot_107.w_17_0 ctx) (sequential_generatedRoot_107.w_18_0 ctx)

def sequential_generatedRoot_107.w_20_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_107.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_20_0

def sequential_generatedRoot_107.w_20_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_107.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_20_1

def sequential_generatedRoot_107.w_20_2 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_107.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_20_2

def sequential_generatedRoot_107.w_20_3 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_107.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_20_3

def sequential_generatedRoot_107.w_21_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_107.Ctx backend hashModel params i_0) : mxxMat2 :=
  ctx.witness.w_21_0

def sequential_generatedRoot_107.w_22_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_107.Ctx backend hashModel params i_0) : mxxMat2 :=
  MxxRuntime.matrixAdd (sequential_generatedRoot_107.w_0_0 ctx) (sequential_generatedRoot_107.w_21_0 ctx)

def sequential_generatedRoot_107.w_24_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_107.Ctx backend hashModel params i_0) : mxxMat3 :=
  (MxxRuntime.unitRow ((1 + (Int.ofNat i_0))) : mxxMat3)

def sequential_generatedRoot_107.w_25_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_107.Ctx backend hashModel params i_0) : mxxMat2 :=
  ctx.witness.w_25_0

def sequential_generatedRoot_107.w_26_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_107.Ctx backend hashModel params i_0) : mxxMat2 :=
  MxxRuntime.matrixAdd (sequential_generatedRoot_107.w_23_0 ctx) (sequential_generatedRoot_107.w_25_0 ctx)

def sequential_generatedRoot_107.w_28_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_107.Ctx backend hashModel params i_0) : mxxMat3 :=
  (MxxRuntime.unitRow ((1 + (Int.ofNat i_0))) : mxxMat3)

def sequential_generatedRoot_107.w_29_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_107.Ctx backend hashModel params i_0) : mxxMat2 :=
  ctx.witness.w_29_0

def sequential_generatedRoot_107.w_30_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_107.Ctx backend hashModel params i_0) : mxxMat2 :=
  MxxRuntime.matrixAdd (sequential_generatedRoot_107.w_27_0 ctx) (sequential_generatedRoot_107.w_29_0 ctx)

def sequential_generatedRoot_107.w_32_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_107.Ctx backend hashModel params i_0) : mxxMat3 :=
  (MxxRuntime.unitRow ((1 + (Int.ofNat i_0))) : mxxMat3)

def sequential_generatedRoot_107.w_33_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_107.Ctx backend hashModel params i_0) : mxxMat2 :=
  ctx.witness.w_33_0

def sequential_generatedRoot_107.w_34_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_107.Ctx backend hashModel params i_0) : mxxMat2 :=
  MxxRuntime.matrixAdd (sequential_generatedRoot_107.w_31_0 ctx) (sequential_generatedRoot_107.w_33_0 ctx)

def sequential_generatedRoot_107.outs {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_107.Ctx backend hashModel params i_0) : mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × Unit :=
  ((sequential_generatedRoot_107.w_22_0 ctx), ((sequential_generatedRoot_107.w_26_0 ctx), ((sequential_generatedRoot_107.w_30_0 ctx), ((sequential_generatedRoot_107.w_34_0 ctx), ()))))

structure sequential_generatedRoot_107.Facts_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_107.Ctx backend hashModel params i_0) : Prop where
  t0 : 0 ≤ ((1 + (Int.ofNat i_0)))
  t1 : ((1 + (Int.ofNat i_0))) < 4500
  h3_0 : 0 ≤ (sequential_generatedRoot_107.w_3_0 ctx)
  h3_0_1 : (sequential_generatedRoot_107.w_3_0 ctx) < 36000
  h4_0 : MxxRuntime.familyGetDynamic (sequential_generatedRoot_107.w_2_0 ctx) (sequential_generatedRoot_107.w_3_0 ctx) (sequential_generatedRoot_107.w_4_0 ctx)
  h5_0 : 0 ≤ (sequential_generatedRoot_107.w_5_0 ctx)
  h5_0_1 : (sequential_generatedRoot_107.w_5_0 ctx) < 36000
  h6_0 : MxxRuntime.familyGetDynamic (sequential_generatedRoot_107.w_2_0 ctx) (sequential_generatedRoot_107.w_5_0 ctx) (sequential_generatedRoot_107.w_6_0 ctx)
  h3_0_2 : 0 ≤ (sequential_generatedRoot_107.w_3_0 ctx)
  h3_0_3 : (sequential_generatedRoot_107.w_3_0 ctx) < 36000
  h9_0 : MxxRuntime.familyGetDynamic (sequential_generatedRoot_107.w_8_0 ctx) (sequential_generatedRoot_107.w_3_0 ctx) (sequential_generatedRoot_107.w_9_0 ctx)
  h5_0_2 : 0 ≤ (sequential_generatedRoot_107.w_5_0 ctx)
  h5_0_3 : (sequential_generatedRoot_107.w_5_0 ctx) < 36000
  h10_0 : MxxRuntime.familyGetDynamic (sequential_generatedRoot_107.w_8_0 ctx) (sequential_generatedRoot_107.w_5_0 ctx) (sequential_generatedRoot_107.w_10_0 ctx)
  h3_0_4 : 0 ≤ (sequential_generatedRoot_107.w_3_0 ctx)
  h3_0_5 : (sequential_generatedRoot_107.w_3_0 ctx) < 36000
  h13_0 : MxxRuntime.familyGetDynamic (sequential_generatedRoot_107.w_12_0 ctx) (sequential_generatedRoot_107.w_3_0 ctx) (sequential_generatedRoot_107.w_13_0 ctx)
  h5_0_4 : 0 ≤ (sequential_generatedRoot_107.w_5_0 ctx)
  h5_0_5 : (sequential_generatedRoot_107.w_5_0 ctx) < 36000
  h14_0 : MxxRuntime.familyGetDynamic (sequential_generatedRoot_107.w_12_0 ctx) (sequential_generatedRoot_107.w_5_0 ctx) (sequential_generatedRoot_107.w_14_0 ctx)
  h3_0_6 : 0 ≤ (sequential_generatedRoot_107.w_3_0 ctx)
  h3_0_7 : (sequential_generatedRoot_107.w_3_0 ctx) < 36000
  h17_0 : MxxRuntime.familyGetDynamic (sequential_generatedRoot_107.w_16_0 ctx) (sequential_generatedRoot_107.w_3_0 ctx) (sequential_generatedRoot_107.w_17_0 ctx)
  h5_0_6 : 0 ≤ (sequential_generatedRoot_107.w_5_0 ctx)
  h5_0_7 : (sequential_generatedRoot_107.w_5_0 ctx) < 36000
  h18_0 : MxxRuntime.familyGetDynamic (sequential_generatedRoot_107.w_16_0 ctx) (sequential_generatedRoot_107.w_5_0 ctx) (sequential_generatedRoot_107.w_18_0 ctx)
  t26 : 0 ≤ (6)
  h20_3 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_sequential_generatedRoot_107_20 backend hashModel params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, ((sequential_generatedRoot_107.w_2_0 ctx), ((sequential_generatedRoot_107.w_8_0 ctx), ((sequential_generatedRoot_107.w_12_0 ctx), ((sequential_generatedRoot_107.w_16_0 ctx), ())))))))) next) (Int.toNat (6)) ((sequential_generatedRoot_107.w_7_0 ctx), ((sequential_generatedRoot_107.w_11_0 ctx), ((sequential_generatedRoot_107.w_15_0 ctx), ((sequential_generatedRoot_107.w_19_0 ctx), ())))) ((sequential_generatedRoot_107.w_20_0 ctx), ((sequential_generatedRoot_107.w_20_1 ctx), ((sequential_generatedRoot_107.w_20_2 ctx), ((sequential_generatedRoot_107.w_20_3 ctx), ()))))
  h21_0 : MxxRuntime.tensorRuns (sequential_generatedRoot_107.w_1_0 ctx) (sequential_generatedRoot_107.w_20_0 ctx) (sequential_generatedRoot_107.w_21_0 ctx)
  t29 : 0 ≤ ((1 + (Int.ofNat i_0)))
  t30 : ((1 + (Int.ofNat i_0))) < 4500
  h25_0 : MxxRuntime.tensorRuns (sequential_generatedRoot_107.w_24_0 ctx) (sequential_generatedRoot_107.w_20_1 ctx) (sequential_generatedRoot_107.w_25_0 ctx)

theorem sequential_generatedRoot_107.facts_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} {ctx : sequential_generatedRoot_107.Ctx backend hashModel params i_0} {outputs : mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × Unit}
    (h : sequential_generatedRoot_107.constraints_0 backend hashModel params i_0 (sequential_generatedRoot_107.w_2_0 ctx) (sequential_generatedRoot_107.w_8_0 ctx) (sequential_generatedRoot_107.w_12_0 ctx) (sequential_generatedRoot_107.w_16_0 ctx) (sequential_generatedRoot_107.w_1_0 ctx) (sequential_generatedRoot_107.w_3_0 ctx) (sequential_generatedRoot_107.w_4_0 ctx) (sequential_generatedRoot_107.w_5_0 ctx) (sequential_generatedRoot_107.w_6_0 ctx) (sequential_generatedRoot_107.w_7_0 ctx) (sequential_generatedRoot_107.w_9_0 ctx) (sequential_generatedRoot_107.w_10_0 ctx) (sequential_generatedRoot_107.w_11_0 ctx) (sequential_generatedRoot_107.w_13_0 ctx) (sequential_generatedRoot_107.w_14_0 ctx) (sequential_generatedRoot_107.w_15_0 ctx) (sequential_generatedRoot_107.w_17_0 ctx) (sequential_generatedRoot_107.w_18_0 ctx) (sequential_generatedRoot_107.w_19_0 ctx) (sequential_generatedRoot_107.w_20_0 ctx) (sequential_generatedRoot_107.w_20_1 ctx) (sequential_generatedRoot_107.w_20_2 ctx) (sequential_generatedRoot_107.w_20_3 ctx) (sequential_generatedRoot_107.w_21_0 ctx) (sequential_generatedRoot_107.w_24_0 ctx) (sequential_generatedRoot_107.w_25_0 ctx)) :
    sequential_generatedRoot_107.Facts_0 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure sequential_generatedRoot_107.Facts_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_107.Ctx backend hashModel params i_0) (outputs : mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × Unit) : Prop where
  t32 : 0 ≤ ((1 + (Int.ofNat i_0)))
  t33 : ((1 + (Int.ofNat i_0))) < 4500
  h29_0 : MxxRuntime.tensorRuns (sequential_generatedRoot_107.w_28_0 ctx) (sequential_generatedRoot_107.w_20_2 ctx) (sequential_generatedRoot_107.w_29_0 ctx)
  t35 : 0 ≤ ((1 + (Int.ofNat i_0)))
  t36 : ((1 + (Int.ofNat i_0))) < 4500
  h33_0 : MxxRuntime.tensorRuns (sequential_generatedRoot_107.w_32_0 ctx) (sequential_generatedRoot_107.w_20_3 ctx) (sequential_generatedRoot_107.w_33_0 ctx)
  hout : outputs = sequential_generatedRoot_107.outs ctx

theorem sequential_generatedRoot_107.facts_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} {ctx : sequential_generatedRoot_107.Ctx backend hashModel params i_0} {outputs : mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × Unit}
    (h : sequential_generatedRoot_107.constraints_1 i_0 (sequential_generatedRoot_107.w_20_2 ctx) (sequential_generatedRoot_107.w_20_3 ctx) (sequential_generatedRoot_107.w_22_0 ctx) (sequential_generatedRoot_107.w_26_0 ctx) (sequential_generatedRoot_107.w_28_0 ctx) (sequential_generatedRoot_107.w_29_0 ctx) (sequential_generatedRoot_107.w_30_0 ctx) (sequential_generatedRoot_107.w_32_0 ctx) (sequential_generatedRoot_107.w_33_0 ctx) (sequential_generatedRoot_107.w_34_0 ctx) outputs) :
    sequential_generatedRoot_107.Facts_1 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2⟩

structure sequential_generatedRoot_107.Facts {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_107.Ctx backend hashModel params i_0) (outputs : mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × Unit) : Prop where
  c0 : sequential_generatedRoot_107.Facts_0 ctx
  c1 : sequential_generatedRoot_107.Facts_1 ctx outputs

theorem sequential_generatedRoot_107.facts {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_107.Ctx backend hashModel params i_0) {outputs : mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × Unit}
    (h : sequential_generatedRoot_107.body backend hashModel params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_107.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_107.facts_0 (ctx := ctx) (outputs := outputs) t0.1
  have t1 := t0.2
  have r1 := sequential_generatedRoot_107.facts_1 (ctx := ctx) (outputs := outputs) t1
  ⟨r0, r1⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_117.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) where
  inputs : (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × Unit
  witness : parallel_generatedRoot_117.Witness

def parallel_generatedRoot_117.w_0_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_117.Ctx backend hashModel params i_0) : Fin 16 → mxxMat1 :=
  ctx.inputs.1

def parallel_generatedRoot_117.w_1_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_117.Ctx backend hashModel params i_0) : Fin 16 → mxxMat1 :=
  ctx.inputs.2.1

def parallel_generatedRoot_117.w_2_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_117.Ctx backend hashModel params i_0) : Fin 16 → mxxMat1 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_117.w_3_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_117.Ctx backend hashModel params i_0) : Fin 16 → mxxMat1 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_117.w_4_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_117.Ctx backend hashModel params i_0) : Fin 2 → mxxMat1 :=
  ctx.witness.w_4_0

def parallel_generatedRoot_117.w_4_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_117.Ctx backend hashModel params i_0) : Fin 2 → mxxMat1 :=
  ctx.witness.w_4_1

def parallel_generatedRoot_117.w_4_2 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_117.Ctx backend hashModel params i_0) : Fin 2 → mxxMat1 :=
  ctx.witness.w_4_2

def parallel_generatedRoot_117.w_4_3 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_117.Ctx backend hashModel params i_0) : Fin 2 → mxxMat1 :=
  ctx.witness.w_4_3

def parallel_generatedRoot_117.w_5_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_117.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_5_0

def parallel_generatedRoot_117.w_6_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_117.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_117.w_7_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_117.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_117.w_5_0 ctx) (parallel_generatedRoot_117.w_6_0 ctx)

def parallel_generatedRoot_117.w_8_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_117.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_117.w_9_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_117.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_9_0

def parallel_generatedRoot_117.w_10_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_117.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_117.w_8_0 ctx) (parallel_generatedRoot_117.w_9_0 ctx)

def parallel_generatedRoot_117.w_11_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_117.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_117.w_12_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_117.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_12_0

def parallel_generatedRoot_117.w_13_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_117.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_117.w_11_0 ctx) (parallel_generatedRoot_117.w_12_0 ctx)

def parallel_generatedRoot_117.w_14_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_117.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_117.w_15_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_117.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_15_0

def parallel_generatedRoot_117.w_16_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_117.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_117.w_14_0 ctx) (parallel_generatedRoot_117.w_15_0 ctx)

def parallel_generatedRoot_117.w_17_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_117.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_117.w_17_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_117.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_17_1

def parallel_generatedRoot_117.w_17_2 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_117.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_17_2

def parallel_generatedRoot_117.w_17_3 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_117.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_17_3

def parallel_generatedRoot_117.outs {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_117.Ctx backend hashModel params i_0) : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit :=
  ((parallel_generatedRoot_117.w_17_0 ctx), ((parallel_generatedRoot_117.w_17_1 ctx), ((parallel_generatedRoot_117.w_17_2 ctx), ((parallel_generatedRoot_117.w_17_3 ctx), ()))))

structure parallel_generatedRoot_117.Facts_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_117.Ctx backend hashModel params i_0) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  t0 : (2) = 2
  h4_3 : (∀ i : Fin 2, parallel_parallel_generatedRoot_117_4 backend hashModel params i_0 i ((parallel_generatedRoot_117.w_0_0 ctx), ((parallel_generatedRoot_117.w_1_0 ctx), ((parallel_generatedRoot_117.w_2_0 ctx), ((parallel_generatedRoot_117.w_3_0 ctx), ())))) (((parallel_generatedRoot_117.w_4_0 ctx) i), (((parallel_generatedRoot_117.w_4_1 ctx) i), (((parallel_generatedRoot_117.w_4_2 ctx) i), (((parallel_generatedRoot_117.w_4_3 ctx) i), ())))))
  h5_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_117.w_4_0 ctx) (0) (parallel_generatedRoot_117.w_5_0 ctx)
  h6_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_117.w_4_0 ctx) (1) (parallel_generatedRoot_117.w_6_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_117.w_4_1 ctx) (0) (parallel_generatedRoot_117.w_8_0 ctx)
  h9_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_117.w_4_1 ctx) (1) (parallel_generatedRoot_117.w_9_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_117.w_4_2 ctx) (0) (parallel_generatedRoot_117.w_11_0 ctx)
  h12_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_117.w_4_2 ctx) (1) (parallel_generatedRoot_117.w_12_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_117.w_4_3 ctx) (0) (parallel_generatedRoot_117.w_14_0 ctx)
  h15_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_117.w_4_3 ctx) (1) (parallel_generatedRoot_117.w_15_0 ctx)
  t10 : 0 ≤ (0)
  h17_3 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_parallel_generatedRoot_117_17 backend hashModel params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, ((parallel_generatedRoot_117.w_4_0 ctx), ((parallel_generatedRoot_117.w_4_1 ctx), ((parallel_generatedRoot_117.w_4_2 ctx), ((parallel_generatedRoot_117.w_4_3 ctx), ())))))))) next) (Int.toNat (0)) ((parallel_generatedRoot_117.w_7_0 ctx), ((parallel_generatedRoot_117.w_10_0 ctx), ((parallel_generatedRoot_117.w_13_0 ctx), ((parallel_generatedRoot_117.w_16_0 ctx), ())))) ((parallel_generatedRoot_117.w_17_0 ctx), ((parallel_generatedRoot_117.w_17_1 ctx), ((parallel_generatedRoot_117.w_17_2 ctx), ((parallel_generatedRoot_117.w_17_3 ctx), ()))))
  hout : outputs = parallel_generatedRoot_117.outs ctx

theorem parallel_generatedRoot_117.facts_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} {ctx : parallel_generatedRoot_117.Ctx backend hashModel params i_0} {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_117.constraints_0 backend hashModel params i_0 (parallel_generatedRoot_117.w_0_0 ctx) (parallel_generatedRoot_117.w_1_0 ctx) (parallel_generatedRoot_117.w_2_0 ctx) (parallel_generatedRoot_117.w_3_0 ctx) (parallel_generatedRoot_117.w_4_0 ctx) (parallel_generatedRoot_117.w_4_1 ctx) (parallel_generatedRoot_117.w_4_2 ctx) (parallel_generatedRoot_117.w_4_3 ctx) (parallel_generatedRoot_117.w_5_0 ctx) (parallel_generatedRoot_117.w_6_0 ctx) (parallel_generatedRoot_117.w_7_0 ctx) (parallel_generatedRoot_117.w_8_0 ctx) (parallel_generatedRoot_117.w_9_0 ctx) (parallel_generatedRoot_117.w_10_0 ctx) (parallel_generatedRoot_117.w_11_0 ctx) (parallel_generatedRoot_117.w_12_0 ctx) (parallel_generatedRoot_117.w_13_0 ctx) (parallel_generatedRoot_117.w_14_0 ctx) (parallel_generatedRoot_117.w_15_0 ctx) (parallel_generatedRoot_117.w_16_0 ctx) (parallel_generatedRoot_117.w_17_0 ctx) (parallel_generatedRoot_117.w_17_1 ctx) (parallel_generatedRoot_117.w_17_2 ctx) (parallel_generatedRoot_117.w_17_3 ctx) outputs) :
    parallel_generatedRoot_117.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_117.Facts {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_117.Ctx backend hashModel params i_0) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  c0 : parallel_generatedRoot_117.Facts_0 ctx outputs

theorem parallel_generatedRoot_117.facts {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_117.Ctx backend hashModel params i_0) {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_117.body backend hashModel params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_117.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_117.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_121.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) where
  inputs : (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × Unit
  witness : parallel_generatedRoot_121.Witness

def parallel_generatedRoot_121.w_0_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_121.Ctx backend hashModel params i_0) : Fin 16 → mxxMat1 :=
  ctx.inputs.1

def parallel_generatedRoot_121.w_1_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_121.Ctx backend hashModel params i_0) : Fin 16 → mxxMat1 :=
  ctx.inputs.2.1

def parallel_generatedRoot_121.w_2_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_121.Ctx backend hashModel params i_0) : Fin 16 → mxxMat1 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_121.w_3_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_121.Ctx backend hashModel params i_0) : Fin 16 → mxxMat1 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_121.w_4_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_121.Ctx backend hashModel params i_0) : Fin 2 → mxxMat1 :=
  ctx.witness.w_4_0

def parallel_generatedRoot_121.w_4_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_121.Ctx backend hashModel params i_0) : Fin 2 → mxxMat1 :=
  ctx.witness.w_4_1

def parallel_generatedRoot_121.w_4_2 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_121.Ctx backend hashModel params i_0) : Fin 2 → mxxMat1 :=
  ctx.witness.w_4_2

def parallel_generatedRoot_121.w_4_3 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_121.Ctx backend hashModel params i_0) : Fin 2 → mxxMat1 :=
  ctx.witness.w_4_3

def parallel_generatedRoot_121.w_5_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_121.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_5_0

def parallel_generatedRoot_121.w_6_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_121.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_121.w_7_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_121.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_121.w_5_0 ctx) (parallel_generatedRoot_121.w_6_0 ctx)

def parallel_generatedRoot_121.w_8_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_121.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_121.w_9_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_121.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_9_0

def parallel_generatedRoot_121.w_10_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_121.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_121.w_8_0 ctx) (parallel_generatedRoot_121.w_9_0 ctx)

def parallel_generatedRoot_121.w_11_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_121.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_121.w_12_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_121.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_12_0

def parallel_generatedRoot_121.w_13_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_121.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_121.w_11_0 ctx) (parallel_generatedRoot_121.w_12_0 ctx)

def parallel_generatedRoot_121.w_14_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_121.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_121.w_15_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_121.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_15_0

def parallel_generatedRoot_121.w_16_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_121.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_121.w_14_0 ctx) (parallel_generatedRoot_121.w_15_0 ctx)

def parallel_generatedRoot_121.w_17_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_121.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_121.w_17_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_121.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_17_1

def parallel_generatedRoot_121.w_17_2 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_121.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_17_2

def parallel_generatedRoot_121.w_17_3 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_121.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_17_3

def parallel_generatedRoot_121.outs {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_121.Ctx backend hashModel params i_0) : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit :=
  ((parallel_generatedRoot_121.w_17_0 ctx), ((parallel_generatedRoot_121.w_17_1 ctx), ((parallel_generatedRoot_121.w_17_2 ctx), ((parallel_generatedRoot_121.w_17_3 ctx), ()))))

structure parallel_generatedRoot_121.Facts_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_121.Ctx backend hashModel params i_0) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  t0 : (2) = 2
  h4_3 : (∀ i : Fin 2, parallel_parallel_generatedRoot_121_4 backend hashModel params i_0 i ((parallel_generatedRoot_121.w_0_0 ctx), ((parallel_generatedRoot_121.w_1_0 ctx), ((parallel_generatedRoot_121.w_2_0 ctx), ((parallel_generatedRoot_121.w_3_0 ctx), ())))) (((parallel_generatedRoot_121.w_4_0 ctx) i), (((parallel_generatedRoot_121.w_4_1 ctx) i), (((parallel_generatedRoot_121.w_4_2 ctx) i), (((parallel_generatedRoot_121.w_4_3 ctx) i), ())))))
  h5_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_121.w_4_0 ctx) (0) (parallel_generatedRoot_121.w_5_0 ctx)
  h6_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_121.w_4_0 ctx) (1) (parallel_generatedRoot_121.w_6_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_121.w_4_1 ctx) (0) (parallel_generatedRoot_121.w_8_0 ctx)
  h9_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_121.w_4_1 ctx) (1) (parallel_generatedRoot_121.w_9_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_121.w_4_2 ctx) (0) (parallel_generatedRoot_121.w_11_0 ctx)
  h12_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_121.w_4_2 ctx) (1) (parallel_generatedRoot_121.w_12_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_121.w_4_3 ctx) (0) (parallel_generatedRoot_121.w_14_0 ctx)
  h15_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_121.w_4_3 ctx) (1) (parallel_generatedRoot_121.w_15_0 ctx)
  t10 : 0 ≤ (0)
  h17_3 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_parallel_generatedRoot_121_17 backend hashModel params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, ((parallel_generatedRoot_121.w_4_0 ctx), ((parallel_generatedRoot_121.w_4_1 ctx), ((parallel_generatedRoot_121.w_4_2 ctx), ((parallel_generatedRoot_121.w_4_3 ctx), ())))))))) next) (Int.toNat (0)) ((parallel_generatedRoot_121.w_7_0 ctx), ((parallel_generatedRoot_121.w_10_0 ctx), ((parallel_generatedRoot_121.w_13_0 ctx), ((parallel_generatedRoot_121.w_16_0 ctx), ())))) ((parallel_generatedRoot_121.w_17_0 ctx), ((parallel_generatedRoot_121.w_17_1 ctx), ((parallel_generatedRoot_121.w_17_2 ctx), ((parallel_generatedRoot_121.w_17_3 ctx), ()))))
  hout : outputs = parallel_generatedRoot_121.outs ctx

theorem parallel_generatedRoot_121.facts_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} {ctx : parallel_generatedRoot_121.Ctx backend hashModel params i_0} {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_121.constraints_0 backend hashModel params i_0 (parallel_generatedRoot_121.w_0_0 ctx) (parallel_generatedRoot_121.w_1_0 ctx) (parallel_generatedRoot_121.w_2_0 ctx) (parallel_generatedRoot_121.w_3_0 ctx) (parallel_generatedRoot_121.w_4_0 ctx) (parallel_generatedRoot_121.w_4_1 ctx) (parallel_generatedRoot_121.w_4_2 ctx) (parallel_generatedRoot_121.w_4_3 ctx) (parallel_generatedRoot_121.w_5_0 ctx) (parallel_generatedRoot_121.w_6_0 ctx) (parallel_generatedRoot_121.w_7_0 ctx) (parallel_generatedRoot_121.w_8_0 ctx) (parallel_generatedRoot_121.w_9_0 ctx) (parallel_generatedRoot_121.w_10_0 ctx) (parallel_generatedRoot_121.w_11_0 ctx) (parallel_generatedRoot_121.w_12_0 ctx) (parallel_generatedRoot_121.w_13_0 ctx) (parallel_generatedRoot_121.w_14_0 ctx) (parallel_generatedRoot_121.w_15_0 ctx) (parallel_generatedRoot_121.w_16_0 ctx) (parallel_generatedRoot_121.w_17_0 ctx) (parallel_generatedRoot_121.w_17_1 ctx) (parallel_generatedRoot_121.w_17_2 ctx) (parallel_generatedRoot_121.w_17_3 ctx) outputs) :
    parallel_generatedRoot_121.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_121.Facts {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_121.Ctx backend hashModel params i_0) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  c0 : parallel_generatedRoot_121.Facts_0 ctx outputs

theorem parallel_generatedRoot_121.facts {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_121.Ctx backend hashModel params i_0) {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_121.body backend hashModel params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_121.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_121.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_122.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) where
  inputs : ByteArray × mxxMat1 × mxxMat1 × (Fin 8 → mxxMat1) × mxxMat1 × mxxMat1 × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × Unit
  witness : parallel_generatedRoot_122.Witness

def parallel_generatedRoot_122.w_1_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_122.Ctx backend hashModel params i_0) : ByteArray :=
  ctx.inputs.1

def parallel_generatedRoot_122.w_2_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_122.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.inputs.2.1

def parallel_generatedRoot_122.w_3_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_122.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_122.w_4_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_122.Ctx backend hashModel params i_0) : Fin 8 → mxxMat1 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_122.w_5_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_122.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.inputs.2.2.2.2.1

def parallel_generatedRoot_122.w_6_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_122.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.inputs.2.2.2.2.2.1

def parallel_generatedRoot_122.w_7_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_122.Ctx backend hashModel params i_0) : Fin 8 → mxxMat1 :=
  ctx.inputs.2.2.2.2.2.2.1

def parallel_generatedRoot_122.w_8_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_122.Ctx backend hashModel params i_0) : Fin 8 → mxxMat1 :=
  ctx.inputs.2.2.2.2.2.2.2.1

def parallel_generatedRoot_122.w_9_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_122.Ctx backend hashModel params i_0) : Fin 8 → mxxMat1 :=
  ctx.inputs.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_122.w_0_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_122.Ctx backend hashModel params i_0) : mxxMat1 :=
  (0 : mxxMat1)

def parallel_generatedRoot_122.w_10_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_122.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_10_0

def parallel_generatedRoot_122.w_10_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_122.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_10_1

def parallel_generatedRoot_122.w_10_2 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_122.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_10_2

def parallel_generatedRoot_122.w_10_3 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_122.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_10_3

def parallel_generatedRoot_122.outs {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_122.Ctx backend hashModel params i_0) : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit :=
  ((parallel_generatedRoot_122.w_10_0 ctx), ((parallel_generatedRoot_122.w_10_1 ctx), ((parallel_generatedRoot_122.w_10_2 ctx), ((parallel_generatedRoot_122.w_10_3 ctx), ()))))

structure parallel_generatedRoot_122.Facts_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_122.Ctx backend hashModel params i_0) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  t0 : 0 ≤ (8)
  h10_3 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_parallel_generatedRoot_122_10 backend hashModel params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, ((parallel_generatedRoot_122.w_1_0 ctx), ((parallel_generatedRoot_122.w_2_0 ctx), ((parallel_generatedRoot_122.w_3_0 ctx), ((parallel_generatedRoot_122.w_4_0 ctx), ((parallel_generatedRoot_122.w_5_0 ctx), ((parallel_generatedRoot_122.w_6_0 ctx), ((parallel_generatedRoot_122.w_7_0 ctx), ((parallel_generatedRoot_122.w_8_0 ctx), ((parallel_generatedRoot_122.w_9_0 ctx), ()))))))))))))) next) (Int.toNat (8)) ((parallel_generatedRoot_122.w_0_0 ctx), ((parallel_generatedRoot_122.w_0_0 ctx), ((parallel_generatedRoot_122.w_0_0 ctx), ((parallel_generatedRoot_122.w_0_0 ctx), ())))) ((parallel_generatedRoot_122.w_10_0 ctx), ((parallel_generatedRoot_122.w_10_1 ctx), ((parallel_generatedRoot_122.w_10_2 ctx), ((parallel_generatedRoot_122.w_10_3 ctx), ()))))
  hout : outputs = parallel_generatedRoot_122.outs ctx

theorem parallel_generatedRoot_122.facts_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} {ctx : parallel_generatedRoot_122.Ctx backend hashModel params i_0} {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_122.constraints_0 backend hashModel params i_0 (parallel_generatedRoot_122.w_1_0 ctx) (parallel_generatedRoot_122.w_2_0 ctx) (parallel_generatedRoot_122.w_3_0 ctx) (parallel_generatedRoot_122.w_4_0 ctx) (parallel_generatedRoot_122.w_5_0 ctx) (parallel_generatedRoot_122.w_6_0 ctx) (parallel_generatedRoot_122.w_7_0 ctx) (parallel_generatedRoot_122.w_8_0 ctx) (parallel_generatedRoot_122.w_9_0 ctx) (parallel_generatedRoot_122.w_0_0 ctx) (parallel_generatedRoot_122.w_10_0 ctx) (parallel_generatedRoot_122.w_10_1 ctx) (parallel_generatedRoot_122.w_10_2 ctx) (parallel_generatedRoot_122.w_10_3 ctx) outputs) :
    parallel_generatedRoot_122.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure parallel_generatedRoot_122.Facts {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_122.Ctx backend hashModel params i_0) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  c0 : parallel_generatedRoot_122.Facts_0 ctx outputs

theorem parallel_generatedRoot_122.facts {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_122.Ctx backend hashModel params i_0) {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_122.body backend hashModel params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_122.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_122.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_123.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) where
  inputs : ByteArray × mxxMat1 × mxxMat1 × (Fin 8 → mxxMat1) × mxxMat1 × mxxMat1 × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × Unit
  witness : parallel_generatedRoot_123.Witness

def parallel_generatedRoot_123.w_1_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_123.Ctx backend hashModel params i_0) : ByteArray :=
  ctx.inputs.1

def parallel_generatedRoot_123.w_2_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_123.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.inputs.2.1

def parallel_generatedRoot_123.w_3_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_123.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_123.w_4_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_123.Ctx backend hashModel params i_0) : Fin 8 → mxxMat1 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_123.w_5_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_123.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.inputs.2.2.2.2.1

def parallel_generatedRoot_123.w_6_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_123.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.inputs.2.2.2.2.2.1

def parallel_generatedRoot_123.w_7_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_123.Ctx backend hashModel params i_0) : Fin 8 → mxxMat1 :=
  ctx.inputs.2.2.2.2.2.2.1

def parallel_generatedRoot_123.w_8_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_123.Ctx backend hashModel params i_0) : Fin 8 → mxxMat1 :=
  ctx.inputs.2.2.2.2.2.2.2.1

def parallel_generatedRoot_123.w_9_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_123.Ctx backend hashModel params i_0) : Fin 8 → mxxMat1 :=
  ctx.inputs.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_123.w_0_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_123.Ctx backend hashModel params i_0) : mxxMat1 :=
  (0 : mxxMat1)

def parallel_generatedRoot_123.w_10_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_123.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_10_0

def parallel_generatedRoot_123.w_10_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_123.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_10_1

def parallel_generatedRoot_123.w_10_2 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_123.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_10_2

def parallel_generatedRoot_123.w_10_3 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_123.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_10_3

def parallel_generatedRoot_123.outs {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_123.Ctx backend hashModel params i_0) : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit :=
  ((parallel_generatedRoot_123.w_10_0 ctx), ((parallel_generatedRoot_123.w_10_1 ctx), ((parallel_generatedRoot_123.w_10_2 ctx), ((parallel_generatedRoot_123.w_10_3 ctx), ()))))

structure parallel_generatedRoot_123.Facts_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_123.Ctx backend hashModel params i_0) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  t0 : 0 ≤ (8)
  h10_3 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_parallel_generatedRoot_123_10 backend hashModel params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, ((parallel_generatedRoot_123.w_1_0 ctx), ((parallel_generatedRoot_123.w_2_0 ctx), ((parallel_generatedRoot_123.w_3_0 ctx), ((parallel_generatedRoot_123.w_4_0 ctx), ((parallel_generatedRoot_123.w_5_0 ctx), ((parallel_generatedRoot_123.w_6_0 ctx), ((parallel_generatedRoot_123.w_7_0 ctx), ((parallel_generatedRoot_123.w_8_0 ctx), ((parallel_generatedRoot_123.w_9_0 ctx), ()))))))))))))) next) (Int.toNat (8)) ((parallel_generatedRoot_123.w_0_0 ctx), ((parallel_generatedRoot_123.w_0_0 ctx), ((parallel_generatedRoot_123.w_0_0 ctx), ((parallel_generatedRoot_123.w_0_0 ctx), ())))) ((parallel_generatedRoot_123.w_10_0 ctx), ((parallel_generatedRoot_123.w_10_1 ctx), ((parallel_generatedRoot_123.w_10_2 ctx), ((parallel_generatedRoot_123.w_10_3 ctx), ()))))
  hout : outputs = parallel_generatedRoot_123.outs ctx

theorem parallel_generatedRoot_123.facts_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} {ctx : parallel_generatedRoot_123.Ctx backend hashModel params i_0} {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_123.constraints_0 backend hashModel params i_0 (parallel_generatedRoot_123.w_1_0 ctx) (parallel_generatedRoot_123.w_2_0 ctx) (parallel_generatedRoot_123.w_3_0 ctx) (parallel_generatedRoot_123.w_4_0 ctx) (parallel_generatedRoot_123.w_5_0 ctx) (parallel_generatedRoot_123.w_6_0 ctx) (parallel_generatedRoot_123.w_7_0 ctx) (parallel_generatedRoot_123.w_8_0 ctx) (parallel_generatedRoot_123.w_9_0 ctx) (parallel_generatedRoot_123.w_0_0 ctx) (parallel_generatedRoot_123.w_10_0 ctx) (parallel_generatedRoot_123.w_10_1 ctx) (parallel_generatedRoot_123.w_10_2 ctx) (parallel_generatedRoot_123.w_10_3 ctx) outputs) :
    parallel_generatedRoot_123.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure parallel_generatedRoot_123.Facts {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_123.Ctx backend hashModel params i_0) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  c0 : parallel_generatedRoot_123.Facts_0 ctx outputs

theorem parallel_generatedRoot_123.facts {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_123.Ctx backend hashModel params i_0) {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_123.body backend hashModel params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_123.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_123.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_162.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) where
  inputs : (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × Unit
  witness : parallel_generatedRoot_162.Witness

def parallel_generatedRoot_162.w_0_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_162.Ctx backend hashModel params i_0) : Fin 14400 → mxxMat1 :=
  ctx.inputs.1

def parallel_generatedRoot_162.w_1_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_162.Ctx backend hashModel params i_0) : Fin 14400 → mxxMat1 :=
  ctx.inputs.2.1

def parallel_generatedRoot_162.w_2_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_162.Ctx backend hashModel params i_0) : Fin 14400 → mxxMat1 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_162.w_3_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_162.Ctx backend hashModel params i_0) : Fin 14400 → mxxMat1 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_162.w_4_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_162.Ctx backend hashModel params i_0) : Fin 2 → mxxMat1 :=
  ctx.witness.w_4_0

def parallel_generatedRoot_162.w_4_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_162.Ctx backend hashModel params i_0) : Fin 2 → mxxMat1 :=
  ctx.witness.w_4_1

def parallel_generatedRoot_162.w_4_2 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_162.Ctx backend hashModel params i_0) : Fin 2 → mxxMat1 :=
  ctx.witness.w_4_2

def parallel_generatedRoot_162.w_4_3 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_162.Ctx backend hashModel params i_0) : Fin 2 → mxxMat1 :=
  ctx.witness.w_4_3

def parallel_generatedRoot_162.w_5_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_162.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_5_0

def parallel_generatedRoot_162.w_6_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_162.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_162.w_7_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_162.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_162.w_5_0 ctx) (parallel_generatedRoot_162.w_6_0 ctx)

def parallel_generatedRoot_162.w_8_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_162.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_162.w_9_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_162.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_9_0

def parallel_generatedRoot_162.w_10_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_162.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_162.w_8_0 ctx) (parallel_generatedRoot_162.w_9_0 ctx)

def parallel_generatedRoot_162.w_11_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_162.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_162.w_12_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_162.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_12_0

def parallel_generatedRoot_162.w_13_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_162.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_162.w_11_0 ctx) (parallel_generatedRoot_162.w_12_0 ctx)

def parallel_generatedRoot_162.w_14_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_162.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_162.w_15_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_162.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_15_0

def parallel_generatedRoot_162.w_16_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_162.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_162.w_14_0 ctx) (parallel_generatedRoot_162.w_15_0 ctx)

def parallel_generatedRoot_162.w_17_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_162.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_162.w_17_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_162.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_17_1

def parallel_generatedRoot_162.w_17_2 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_162.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_17_2

def parallel_generatedRoot_162.w_17_3 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_162.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_17_3

def parallel_generatedRoot_162.outs {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_162.Ctx backend hashModel params i_0) : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit :=
  ((parallel_generatedRoot_162.w_17_0 ctx), ((parallel_generatedRoot_162.w_17_1 ctx), ((parallel_generatedRoot_162.w_17_2 ctx), ((parallel_generatedRoot_162.w_17_3 ctx), ()))))

structure parallel_generatedRoot_162.Facts_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_162.Ctx backend hashModel params i_0) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  t0 : (2) = 2
  h4_3 : (∀ i : Fin 2, parallel_parallel_generatedRoot_162_4 backend hashModel params i_0 i ((parallel_generatedRoot_162.w_0_0 ctx), ((parallel_generatedRoot_162.w_1_0 ctx), ((parallel_generatedRoot_162.w_2_0 ctx), ((parallel_generatedRoot_162.w_3_0 ctx), ())))) (((parallel_generatedRoot_162.w_4_0 ctx) i), (((parallel_generatedRoot_162.w_4_1 ctx) i), (((parallel_generatedRoot_162.w_4_2 ctx) i), (((parallel_generatedRoot_162.w_4_3 ctx) i), ())))))
  h5_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_162.w_4_0 ctx) (0) (parallel_generatedRoot_162.w_5_0 ctx)
  h6_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_162.w_4_0 ctx) (1) (parallel_generatedRoot_162.w_6_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_162.w_4_1 ctx) (0) (parallel_generatedRoot_162.w_8_0 ctx)
  h9_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_162.w_4_1 ctx) (1) (parallel_generatedRoot_162.w_9_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_162.w_4_2 ctx) (0) (parallel_generatedRoot_162.w_11_0 ctx)
  h12_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_162.w_4_2 ctx) (1) (parallel_generatedRoot_162.w_12_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_162.w_4_3 ctx) (0) (parallel_generatedRoot_162.w_14_0 ctx)
  h15_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_162.w_4_3 ctx) (1) (parallel_generatedRoot_162.w_15_0 ctx)
  t10 : 0 ≤ (0)
  h17_3 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_parallel_generatedRoot_162_17 backend hashModel params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, ((parallel_generatedRoot_162.w_4_0 ctx), ((parallel_generatedRoot_162.w_4_1 ctx), ((parallel_generatedRoot_162.w_4_2 ctx), ((parallel_generatedRoot_162.w_4_3 ctx), ())))))))) next) (Int.toNat (0)) ((parallel_generatedRoot_162.w_7_0 ctx), ((parallel_generatedRoot_162.w_10_0 ctx), ((parallel_generatedRoot_162.w_13_0 ctx), ((parallel_generatedRoot_162.w_16_0 ctx), ())))) ((parallel_generatedRoot_162.w_17_0 ctx), ((parallel_generatedRoot_162.w_17_1 ctx), ((parallel_generatedRoot_162.w_17_2 ctx), ((parallel_generatedRoot_162.w_17_3 ctx), ()))))
  hout : outputs = parallel_generatedRoot_162.outs ctx

theorem parallel_generatedRoot_162.facts_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} {ctx : parallel_generatedRoot_162.Ctx backend hashModel params i_0} {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_162.constraints_0 backend hashModel params i_0 (parallel_generatedRoot_162.w_0_0 ctx) (parallel_generatedRoot_162.w_1_0 ctx) (parallel_generatedRoot_162.w_2_0 ctx) (parallel_generatedRoot_162.w_3_0 ctx) (parallel_generatedRoot_162.w_4_0 ctx) (parallel_generatedRoot_162.w_4_1 ctx) (parallel_generatedRoot_162.w_4_2 ctx) (parallel_generatedRoot_162.w_4_3 ctx) (parallel_generatedRoot_162.w_5_0 ctx) (parallel_generatedRoot_162.w_6_0 ctx) (parallel_generatedRoot_162.w_7_0 ctx) (parallel_generatedRoot_162.w_8_0 ctx) (parallel_generatedRoot_162.w_9_0 ctx) (parallel_generatedRoot_162.w_10_0 ctx) (parallel_generatedRoot_162.w_11_0 ctx) (parallel_generatedRoot_162.w_12_0 ctx) (parallel_generatedRoot_162.w_13_0 ctx) (parallel_generatedRoot_162.w_14_0 ctx) (parallel_generatedRoot_162.w_15_0 ctx) (parallel_generatedRoot_162.w_16_0 ctx) (parallel_generatedRoot_162.w_17_0 ctx) (parallel_generatedRoot_162.w_17_1 ctx) (parallel_generatedRoot_162.w_17_2 ctx) (parallel_generatedRoot_162.w_17_3 ctx) outputs) :
    parallel_generatedRoot_162.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_162.Facts {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_162.Ctx backend hashModel params i_0) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  c0 : parallel_generatedRoot_162.Facts_0 ctx outputs

theorem parallel_generatedRoot_162.facts {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_162.Ctx backend hashModel params i_0) {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_162.body backend hashModel params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_162.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_162.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_166.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) where
  inputs : (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × Unit
  witness : parallel_generatedRoot_166.Witness

def parallel_generatedRoot_166.w_0_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_166.Ctx backend hashModel params i_0) : Fin 14400 → mxxMat1 :=
  ctx.inputs.1

def parallel_generatedRoot_166.w_1_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_166.Ctx backend hashModel params i_0) : Fin 14400 → mxxMat1 :=
  ctx.inputs.2.1

def parallel_generatedRoot_166.w_2_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_166.Ctx backend hashModel params i_0) : Fin 14400 → mxxMat1 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_166.w_3_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_166.Ctx backend hashModel params i_0) : Fin 14400 → mxxMat1 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_166.w_4_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_166.Ctx backend hashModel params i_0) : Fin 2 → mxxMat1 :=
  ctx.witness.w_4_0

def parallel_generatedRoot_166.w_4_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_166.Ctx backend hashModel params i_0) : Fin 2 → mxxMat1 :=
  ctx.witness.w_4_1

def parallel_generatedRoot_166.w_4_2 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_166.Ctx backend hashModel params i_0) : Fin 2 → mxxMat1 :=
  ctx.witness.w_4_2

def parallel_generatedRoot_166.w_4_3 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_166.Ctx backend hashModel params i_0) : Fin 2 → mxxMat1 :=
  ctx.witness.w_4_3

def parallel_generatedRoot_166.w_5_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_166.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_5_0

def parallel_generatedRoot_166.w_6_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_166.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_166.w_7_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_166.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_166.w_5_0 ctx) (parallel_generatedRoot_166.w_6_0 ctx)

def parallel_generatedRoot_166.w_8_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_166.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_166.w_9_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_166.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_9_0

def parallel_generatedRoot_166.w_10_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_166.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_166.w_8_0 ctx) (parallel_generatedRoot_166.w_9_0 ctx)

def parallel_generatedRoot_166.w_11_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_166.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_166.w_12_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_166.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_12_0

def parallel_generatedRoot_166.w_13_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_166.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_166.w_11_0 ctx) (parallel_generatedRoot_166.w_12_0 ctx)

def parallel_generatedRoot_166.w_14_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_166.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_166.w_15_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_166.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_15_0

def parallel_generatedRoot_166.w_16_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_166.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_166.w_14_0 ctx) (parallel_generatedRoot_166.w_15_0 ctx)

def parallel_generatedRoot_166.w_17_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_166.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_166.w_17_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_166.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_17_1

def parallel_generatedRoot_166.w_17_2 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_166.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_17_2

def parallel_generatedRoot_166.w_17_3 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_166.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_17_3

def parallel_generatedRoot_166.outs {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_166.Ctx backend hashModel params i_0) : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit :=
  ((parallel_generatedRoot_166.w_17_0 ctx), ((parallel_generatedRoot_166.w_17_1 ctx), ((parallel_generatedRoot_166.w_17_2 ctx), ((parallel_generatedRoot_166.w_17_3 ctx), ()))))

structure parallel_generatedRoot_166.Facts_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_166.Ctx backend hashModel params i_0) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  t0 : (2) = 2
  h4_3 : (∀ i : Fin 2, parallel_parallel_generatedRoot_166_4 backend hashModel params i_0 i ((parallel_generatedRoot_166.w_0_0 ctx), ((parallel_generatedRoot_166.w_1_0 ctx), ((parallel_generatedRoot_166.w_2_0 ctx), ((parallel_generatedRoot_166.w_3_0 ctx), ())))) (((parallel_generatedRoot_166.w_4_0 ctx) i), (((parallel_generatedRoot_166.w_4_1 ctx) i), (((parallel_generatedRoot_166.w_4_2 ctx) i), (((parallel_generatedRoot_166.w_4_3 ctx) i), ())))))
  h5_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_166.w_4_0 ctx) (0) (parallel_generatedRoot_166.w_5_0 ctx)
  h6_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_166.w_4_0 ctx) (1) (parallel_generatedRoot_166.w_6_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_166.w_4_1 ctx) (0) (parallel_generatedRoot_166.w_8_0 ctx)
  h9_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_166.w_4_1 ctx) (1) (parallel_generatedRoot_166.w_9_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_166.w_4_2 ctx) (0) (parallel_generatedRoot_166.w_11_0 ctx)
  h12_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_166.w_4_2 ctx) (1) (parallel_generatedRoot_166.w_12_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_166.w_4_3 ctx) (0) (parallel_generatedRoot_166.w_14_0 ctx)
  h15_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_166.w_4_3 ctx) (1) (parallel_generatedRoot_166.w_15_0 ctx)
  t10 : 0 ≤ (0)
  h17_3 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_parallel_generatedRoot_166_17 backend hashModel params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, ((parallel_generatedRoot_166.w_4_0 ctx), ((parallel_generatedRoot_166.w_4_1 ctx), ((parallel_generatedRoot_166.w_4_2 ctx), ((parallel_generatedRoot_166.w_4_3 ctx), ())))))))) next) (Int.toNat (0)) ((parallel_generatedRoot_166.w_7_0 ctx), ((parallel_generatedRoot_166.w_10_0 ctx), ((parallel_generatedRoot_166.w_13_0 ctx), ((parallel_generatedRoot_166.w_16_0 ctx), ())))) ((parallel_generatedRoot_166.w_17_0 ctx), ((parallel_generatedRoot_166.w_17_1 ctx), ((parallel_generatedRoot_166.w_17_2 ctx), ((parallel_generatedRoot_166.w_17_3 ctx), ()))))
  hout : outputs = parallel_generatedRoot_166.outs ctx

theorem parallel_generatedRoot_166.facts_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} {ctx : parallel_generatedRoot_166.Ctx backend hashModel params i_0} {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_166.constraints_0 backend hashModel params i_0 (parallel_generatedRoot_166.w_0_0 ctx) (parallel_generatedRoot_166.w_1_0 ctx) (parallel_generatedRoot_166.w_2_0 ctx) (parallel_generatedRoot_166.w_3_0 ctx) (parallel_generatedRoot_166.w_4_0 ctx) (parallel_generatedRoot_166.w_4_1 ctx) (parallel_generatedRoot_166.w_4_2 ctx) (parallel_generatedRoot_166.w_4_3 ctx) (parallel_generatedRoot_166.w_5_0 ctx) (parallel_generatedRoot_166.w_6_0 ctx) (parallel_generatedRoot_166.w_7_0 ctx) (parallel_generatedRoot_166.w_8_0 ctx) (parallel_generatedRoot_166.w_9_0 ctx) (parallel_generatedRoot_166.w_10_0 ctx) (parallel_generatedRoot_166.w_11_0 ctx) (parallel_generatedRoot_166.w_12_0 ctx) (parallel_generatedRoot_166.w_13_0 ctx) (parallel_generatedRoot_166.w_14_0 ctx) (parallel_generatedRoot_166.w_15_0 ctx) (parallel_generatedRoot_166.w_16_0 ctx) (parallel_generatedRoot_166.w_17_0 ctx) (parallel_generatedRoot_166.w_17_1 ctx) (parallel_generatedRoot_166.w_17_2 ctx) (parallel_generatedRoot_166.w_17_3 ctx) outputs) :
    parallel_generatedRoot_166.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_166.Facts {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_166.Ctx backend hashModel params i_0) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  c0 : parallel_generatedRoot_166.Facts_0 ctx outputs

theorem parallel_generatedRoot_166.facts {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_166.Ctx backend hashModel params i_0) {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_166.body backend hashModel params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_166.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_166.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_167.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) where
  inputs : ByteArray × mxxMat1 × mxxMat1 × (Fin 8 → mxxMat1) × mxxMat1 × mxxMat1 × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × Unit
  witness : parallel_generatedRoot_167.Witness

def parallel_generatedRoot_167.w_1_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_167.Ctx backend hashModel params i_0) : ByteArray :=
  ctx.inputs.1

def parallel_generatedRoot_167.w_2_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_167.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.inputs.2.1

def parallel_generatedRoot_167.w_3_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_167.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_167.w_4_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_167.Ctx backend hashModel params i_0) : Fin 8 → mxxMat1 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_167.w_5_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_167.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.inputs.2.2.2.2.1

def parallel_generatedRoot_167.w_6_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_167.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.inputs.2.2.2.2.2.1

def parallel_generatedRoot_167.w_7_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_167.Ctx backend hashModel params i_0) : Fin 8 → mxxMat1 :=
  ctx.inputs.2.2.2.2.2.2.1

def parallel_generatedRoot_167.w_8_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_167.Ctx backend hashModel params i_0) : Fin 8 → mxxMat1 :=
  ctx.inputs.2.2.2.2.2.2.2.1

def parallel_generatedRoot_167.w_9_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_167.Ctx backend hashModel params i_0) : Fin 8 → mxxMat1 :=
  ctx.inputs.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_167.w_0_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_167.Ctx backend hashModel params i_0) : mxxMat1 :=
  (0 : mxxMat1)

def parallel_generatedRoot_167.w_10_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_167.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_10_0

def parallel_generatedRoot_167.w_10_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_167.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_10_1

def parallel_generatedRoot_167.w_10_2 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_167.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_10_2

def parallel_generatedRoot_167.w_10_3 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_167.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_10_3

def parallel_generatedRoot_167.outs {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_167.Ctx backend hashModel params i_0) : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit :=
  ((parallel_generatedRoot_167.w_10_0 ctx), ((parallel_generatedRoot_167.w_10_1 ctx), ((parallel_generatedRoot_167.w_10_2 ctx), ((parallel_generatedRoot_167.w_10_3 ctx), ()))))

structure parallel_generatedRoot_167.Facts_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_167.Ctx backend hashModel params i_0) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  t0 : 0 ≤ (8)
  h10_3 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_parallel_generatedRoot_167_10 backend hashModel params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, ((parallel_generatedRoot_167.w_1_0 ctx), ((parallel_generatedRoot_167.w_2_0 ctx), ((parallel_generatedRoot_167.w_3_0 ctx), ((parallel_generatedRoot_167.w_4_0 ctx), ((parallel_generatedRoot_167.w_5_0 ctx), ((parallel_generatedRoot_167.w_6_0 ctx), ((parallel_generatedRoot_167.w_7_0 ctx), ((parallel_generatedRoot_167.w_8_0 ctx), ((parallel_generatedRoot_167.w_9_0 ctx), ()))))))))))))) next) (Int.toNat (8)) ((parallel_generatedRoot_167.w_0_0 ctx), ((parallel_generatedRoot_167.w_0_0 ctx), ((parallel_generatedRoot_167.w_0_0 ctx), ((parallel_generatedRoot_167.w_0_0 ctx), ())))) ((parallel_generatedRoot_167.w_10_0 ctx), ((parallel_generatedRoot_167.w_10_1 ctx), ((parallel_generatedRoot_167.w_10_2 ctx), ((parallel_generatedRoot_167.w_10_3 ctx), ()))))
  hout : outputs = parallel_generatedRoot_167.outs ctx

theorem parallel_generatedRoot_167.facts_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} {ctx : parallel_generatedRoot_167.Ctx backend hashModel params i_0} {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_167.constraints_0 backend hashModel params i_0 (parallel_generatedRoot_167.w_1_0 ctx) (parallel_generatedRoot_167.w_2_0 ctx) (parallel_generatedRoot_167.w_3_0 ctx) (parallel_generatedRoot_167.w_4_0 ctx) (parallel_generatedRoot_167.w_5_0 ctx) (parallel_generatedRoot_167.w_6_0 ctx) (parallel_generatedRoot_167.w_7_0 ctx) (parallel_generatedRoot_167.w_8_0 ctx) (parallel_generatedRoot_167.w_9_0 ctx) (parallel_generatedRoot_167.w_0_0 ctx) (parallel_generatedRoot_167.w_10_0 ctx) (parallel_generatedRoot_167.w_10_1 ctx) (parallel_generatedRoot_167.w_10_2 ctx) (parallel_generatedRoot_167.w_10_3 ctx) outputs) :
    parallel_generatedRoot_167.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure parallel_generatedRoot_167.Facts {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_167.Ctx backend hashModel params i_0) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  c0 : parallel_generatedRoot_167.Facts_0 ctx outputs

theorem parallel_generatedRoot_167.facts {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : parallel_generatedRoot_167.Ctx backend hashModel params i_0) {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_167.body backend hashModel params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_167.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_167.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_189.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) where
  inputs : mxxMat4 × mxxMat4 × mxxMat4 × mxxMat4 × (Fin 7200 → mxxMat1) × (Fin 7200 → mxxMat1) × (Fin 7200 → mxxMat1) × (Fin 7200 → mxxMat1) × Unit
  witness : sequential_generatedRoot_189.Witness

def sequential_generatedRoot_189.w_0_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_189.Ctx backend hashModel params i_0) : mxxMat4 :=
  ctx.inputs.1

def sequential_generatedRoot_189.w_23_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_189.Ctx backend hashModel params i_0) : mxxMat4 :=
  ctx.inputs.2.1

def sequential_generatedRoot_189.w_27_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_189.Ctx backend hashModel params i_0) : mxxMat4 :=
  ctx.inputs.2.2.1

def sequential_generatedRoot_189.w_31_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_189.Ctx backend hashModel params i_0) : mxxMat4 :=
  ctx.inputs.2.2.2.1

def sequential_generatedRoot_189.w_2_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_189.Ctx backend hashModel params i_0) : Fin 7200 → mxxMat1 :=
  ctx.inputs.2.2.2.2.1

def sequential_generatedRoot_189.w_8_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_189.Ctx backend hashModel params i_0) : Fin 7200 → mxxMat1 :=
  ctx.inputs.2.2.2.2.2.1

def sequential_generatedRoot_189.w_12_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_189.Ctx backend hashModel params i_0) : Fin 7200 → mxxMat1 :=
  ctx.inputs.2.2.2.2.2.2.1

def sequential_generatedRoot_189.w_16_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_189.Ctx backend hashModel params i_0) : Fin 7200 → mxxMat1 :=
  ctx.inputs.2.2.2.2.2.2.2.1

def sequential_generatedRoot_189.w_1_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_189.Ctx backend hashModel params i_0) : mxxMat5 :=
  (MxxRuntime.unitRow ((1 + (Int.ofNat i_0))) : mxxMat5)

def sequential_generatedRoot_189.w_3_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_189.Ctx backend hashModel params i_0) : Int :=
  (8 + (8 * (Int.ofNat i_0)))

def sequential_generatedRoot_189.w_4_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_189.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_4_0

def sequential_generatedRoot_189.w_5_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_189.Ctx backend hashModel params i_0) : Int :=
  (9 + (8 * (Int.ofNat i_0)))

def sequential_generatedRoot_189.w_6_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_189.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_6_0

def sequential_generatedRoot_189.w_7_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_189.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixAdd (sequential_generatedRoot_189.w_4_0 ctx) (sequential_generatedRoot_189.w_6_0 ctx)

def sequential_generatedRoot_189.w_9_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_189.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_9_0

def sequential_generatedRoot_189.w_10_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_189.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_10_0

def sequential_generatedRoot_189.w_11_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_189.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixAdd (sequential_generatedRoot_189.w_9_0 ctx) (sequential_generatedRoot_189.w_10_0 ctx)

def sequential_generatedRoot_189.w_13_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_189.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_13_0

def sequential_generatedRoot_189.w_14_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_189.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_14_0

def sequential_generatedRoot_189.w_15_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_189.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixAdd (sequential_generatedRoot_189.w_13_0 ctx) (sequential_generatedRoot_189.w_14_0 ctx)

def sequential_generatedRoot_189.w_17_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_189.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_17_0

def sequential_generatedRoot_189.w_18_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_189.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_18_0

def sequential_generatedRoot_189.w_19_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_189.Ctx backend hashModel params i_0) : mxxMat1 :=
  MxxRuntime.matrixAdd (sequential_generatedRoot_189.w_17_0 ctx) (sequential_generatedRoot_189.w_18_0 ctx)

def sequential_generatedRoot_189.w_20_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_189.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_20_0

def sequential_generatedRoot_189.w_20_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_189.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_20_1

def sequential_generatedRoot_189.w_20_2 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_189.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_20_2

def sequential_generatedRoot_189.w_20_3 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_189.Ctx backend hashModel params i_0) : mxxMat1 :=
  ctx.witness.w_20_3

def sequential_generatedRoot_189.w_21_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_189.Ctx backend hashModel params i_0) : mxxMat4 :=
  ctx.witness.w_21_0

def sequential_generatedRoot_189.w_22_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_189.Ctx backend hashModel params i_0) : mxxMat4 :=
  MxxRuntime.matrixAdd (sequential_generatedRoot_189.w_0_0 ctx) (sequential_generatedRoot_189.w_21_0 ctx)

def sequential_generatedRoot_189.w_24_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_189.Ctx backend hashModel params i_0) : mxxMat5 :=
  (MxxRuntime.unitRow ((1 + (Int.ofNat i_0))) : mxxMat5)

def sequential_generatedRoot_189.w_25_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_189.Ctx backend hashModel params i_0) : mxxMat4 :=
  ctx.witness.w_25_0

def sequential_generatedRoot_189.w_26_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_189.Ctx backend hashModel params i_0) : mxxMat4 :=
  MxxRuntime.matrixAdd (sequential_generatedRoot_189.w_23_0 ctx) (sequential_generatedRoot_189.w_25_0 ctx)

def sequential_generatedRoot_189.w_28_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_189.Ctx backend hashModel params i_0) : mxxMat5 :=
  (MxxRuntime.unitRow ((1 + (Int.ofNat i_0))) : mxxMat5)

def sequential_generatedRoot_189.w_29_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_189.Ctx backend hashModel params i_0) : mxxMat4 :=
  ctx.witness.w_29_0

def sequential_generatedRoot_189.w_30_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_189.Ctx backend hashModel params i_0) : mxxMat4 :=
  MxxRuntime.matrixAdd (sequential_generatedRoot_189.w_27_0 ctx) (sequential_generatedRoot_189.w_29_0 ctx)

def sequential_generatedRoot_189.w_32_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_189.Ctx backend hashModel params i_0) : mxxMat5 :=
  (MxxRuntime.unitRow ((1 + (Int.ofNat i_0))) : mxxMat5)

def sequential_generatedRoot_189.w_33_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_189.Ctx backend hashModel params i_0) : mxxMat4 :=
  ctx.witness.w_33_0

def sequential_generatedRoot_189.w_34_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_189.Ctx backend hashModel params i_0) : mxxMat4 :=
  MxxRuntime.matrixAdd (sequential_generatedRoot_189.w_31_0 ctx) (sequential_generatedRoot_189.w_33_0 ctx)

def sequential_generatedRoot_189.outs {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_189.Ctx backend hashModel params i_0) : mxxMat4 × mxxMat4 × mxxMat4 × mxxMat4 × Unit :=
  ((sequential_generatedRoot_189.w_22_0 ctx), ((sequential_generatedRoot_189.w_26_0 ctx), ((sequential_generatedRoot_189.w_30_0 ctx), ((sequential_generatedRoot_189.w_34_0 ctx), ()))))

structure sequential_generatedRoot_189.Facts_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_189.Ctx backend hashModel params i_0) : Prop where
  t0 : 0 ≤ ((1 + (Int.ofNat i_0)))
  t1 : ((1 + (Int.ofNat i_0))) < 900
  h3_0 : 0 ≤ (sequential_generatedRoot_189.w_3_0 ctx)
  h3_0_1 : (sequential_generatedRoot_189.w_3_0 ctx) < 7200
  h4_0 : MxxRuntime.familyGetDynamic (sequential_generatedRoot_189.w_2_0 ctx) (sequential_generatedRoot_189.w_3_0 ctx) (sequential_generatedRoot_189.w_4_0 ctx)
  h5_0 : 0 ≤ (sequential_generatedRoot_189.w_5_0 ctx)
  h5_0_1 : (sequential_generatedRoot_189.w_5_0 ctx) < 7200
  h6_0 : MxxRuntime.familyGetDynamic (sequential_generatedRoot_189.w_2_0 ctx) (sequential_generatedRoot_189.w_5_0 ctx) (sequential_generatedRoot_189.w_6_0 ctx)
  h3_0_2 : 0 ≤ (sequential_generatedRoot_189.w_3_0 ctx)
  h3_0_3 : (sequential_generatedRoot_189.w_3_0 ctx) < 7200
  h9_0 : MxxRuntime.familyGetDynamic (sequential_generatedRoot_189.w_8_0 ctx) (sequential_generatedRoot_189.w_3_0 ctx) (sequential_generatedRoot_189.w_9_0 ctx)
  h5_0_2 : 0 ≤ (sequential_generatedRoot_189.w_5_0 ctx)
  h5_0_3 : (sequential_generatedRoot_189.w_5_0 ctx) < 7200
  h10_0 : MxxRuntime.familyGetDynamic (sequential_generatedRoot_189.w_8_0 ctx) (sequential_generatedRoot_189.w_5_0 ctx) (sequential_generatedRoot_189.w_10_0 ctx)
  h3_0_4 : 0 ≤ (sequential_generatedRoot_189.w_3_0 ctx)
  h3_0_5 : (sequential_generatedRoot_189.w_3_0 ctx) < 7200
  h13_0 : MxxRuntime.familyGetDynamic (sequential_generatedRoot_189.w_12_0 ctx) (sequential_generatedRoot_189.w_3_0 ctx) (sequential_generatedRoot_189.w_13_0 ctx)
  h5_0_4 : 0 ≤ (sequential_generatedRoot_189.w_5_0 ctx)
  h5_0_5 : (sequential_generatedRoot_189.w_5_0 ctx) < 7200
  h14_0 : MxxRuntime.familyGetDynamic (sequential_generatedRoot_189.w_12_0 ctx) (sequential_generatedRoot_189.w_5_0 ctx) (sequential_generatedRoot_189.w_14_0 ctx)
  h3_0_6 : 0 ≤ (sequential_generatedRoot_189.w_3_0 ctx)
  h3_0_7 : (sequential_generatedRoot_189.w_3_0 ctx) < 7200
  h17_0 : MxxRuntime.familyGetDynamic (sequential_generatedRoot_189.w_16_0 ctx) (sequential_generatedRoot_189.w_3_0 ctx) (sequential_generatedRoot_189.w_17_0 ctx)
  h5_0_6 : 0 ≤ (sequential_generatedRoot_189.w_5_0 ctx)
  h5_0_7 : (sequential_generatedRoot_189.w_5_0 ctx) < 7200
  h18_0 : MxxRuntime.familyGetDynamic (sequential_generatedRoot_189.w_16_0 ctx) (sequential_generatedRoot_189.w_5_0 ctx) (sequential_generatedRoot_189.w_18_0 ctx)
  t26 : 0 ≤ (6)
  h20_3 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_sequential_generatedRoot_189_20 backend hashModel params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, ((sequential_generatedRoot_189.w_2_0 ctx), ((sequential_generatedRoot_189.w_8_0 ctx), ((sequential_generatedRoot_189.w_12_0 ctx), ((sequential_generatedRoot_189.w_16_0 ctx), ())))))))) next) (Int.toNat (6)) ((sequential_generatedRoot_189.w_7_0 ctx), ((sequential_generatedRoot_189.w_11_0 ctx), ((sequential_generatedRoot_189.w_15_0 ctx), ((sequential_generatedRoot_189.w_19_0 ctx), ())))) ((sequential_generatedRoot_189.w_20_0 ctx), ((sequential_generatedRoot_189.w_20_1 ctx), ((sequential_generatedRoot_189.w_20_2 ctx), ((sequential_generatedRoot_189.w_20_3 ctx), ()))))
  h21_0 : MxxRuntime.tensorRuns (sequential_generatedRoot_189.w_1_0 ctx) (sequential_generatedRoot_189.w_20_0 ctx) (sequential_generatedRoot_189.w_21_0 ctx)
  t29 : 0 ≤ ((1 + (Int.ofNat i_0)))
  t30 : ((1 + (Int.ofNat i_0))) < 900
  h25_0 : MxxRuntime.tensorRuns (sequential_generatedRoot_189.w_24_0 ctx) (sequential_generatedRoot_189.w_20_1 ctx) (sequential_generatedRoot_189.w_25_0 ctx)

theorem sequential_generatedRoot_189.facts_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} {ctx : sequential_generatedRoot_189.Ctx backend hashModel params i_0} {outputs : mxxMat4 × mxxMat4 × mxxMat4 × mxxMat4 × Unit}
    (h : sequential_generatedRoot_189.constraints_0 backend hashModel params i_0 (sequential_generatedRoot_189.w_2_0 ctx) (sequential_generatedRoot_189.w_8_0 ctx) (sequential_generatedRoot_189.w_12_0 ctx) (sequential_generatedRoot_189.w_16_0 ctx) (sequential_generatedRoot_189.w_1_0 ctx) (sequential_generatedRoot_189.w_3_0 ctx) (sequential_generatedRoot_189.w_4_0 ctx) (sequential_generatedRoot_189.w_5_0 ctx) (sequential_generatedRoot_189.w_6_0 ctx) (sequential_generatedRoot_189.w_7_0 ctx) (sequential_generatedRoot_189.w_9_0 ctx) (sequential_generatedRoot_189.w_10_0 ctx) (sequential_generatedRoot_189.w_11_0 ctx) (sequential_generatedRoot_189.w_13_0 ctx) (sequential_generatedRoot_189.w_14_0 ctx) (sequential_generatedRoot_189.w_15_0 ctx) (sequential_generatedRoot_189.w_17_0 ctx) (sequential_generatedRoot_189.w_18_0 ctx) (sequential_generatedRoot_189.w_19_0 ctx) (sequential_generatedRoot_189.w_20_0 ctx) (sequential_generatedRoot_189.w_20_1 ctx) (sequential_generatedRoot_189.w_20_2 ctx) (sequential_generatedRoot_189.w_20_3 ctx) (sequential_generatedRoot_189.w_21_0 ctx) (sequential_generatedRoot_189.w_24_0 ctx) (sequential_generatedRoot_189.w_25_0 ctx)) :
    sequential_generatedRoot_189.Facts_0 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure sequential_generatedRoot_189.Facts_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_189.Ctx backend hashModel params i_0) (outputs : mxxMat4 × mxxMat4 × mxxMat4 × mxxMat4 × Unit) : Prop where
  t32 : 0 ≤ ((1 + (Int.ofNat i_0)))
  t33 : ((1 + (Int.ofNat i_0))) < 900
  h29_0 : MxxRuntime.tensorRuns (sequential_generatedRoot_189.w_28_0 ctx) (sequential_generatedRoot_189.w_20_2 ctx) (sequential_generatedRoot_189.w_29_0 ctx)
  t35 : 0 ≤ ((1 + (Int.ofNat i_0)))
  t36 : ((1 + (Int.ofNat i_0))) < 900
  h33_0 : MxxRuntime.tensorRuns (sequential_generatedRoot_189.w_32_0 ctx) (sequential_generatedRoot_189.w_20_3 ctx) (sequential_generatedRoot_189.w_33_0 ctx)
  hout : outputs = sequential_generatedRoot_189.outs ctx

theorem sequential_generatedRoot_189.facts_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} {ctx : sequential_generatedRoot_189.Ctx backend hashModel params i_0} {outputs : mxxMat4 × mxxMat4 × mxxMat4 × mxxMat4 × Unit}
    (h : sequential_generatedRoot_189.constraints_1 i_0 (sequential_generatedRoot_189.w_20_2 ctx) (sequential_generatedRoot_189.w_20_3 ctx) (sequential_generatedRoot_189.w_22_0 ctx) (sequential_generatedRoot_189.w_26_0 ctx) (sequential_generatedRoot_189.w_28_0 ctx) (sequential_generatedRoot_189.w_29_0 ctx) (sequential_generatedRoot_189.w_30_0 ctx) (sequential_generatedRoot_189.w_32_0 ctx) (sequential_generatedRoot_189.w_33_0 ctx) (sequential_generatedRoot_189.w_34_0 ctx) outputs) :
    sequential_generatedRoot_189.Facts_1 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2⟩

structure sequential_generatedRoot_189.Facts {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_189.Ctx backend hashModel params i_0) (outputs : mxxMat4 × mxxMat4 × mxxMat4 × mxxMat4 × Unit) : Prop where
  c0 : sequential_generatedRoot_189.Facts_0 ctx
  c1 : sequential_generatedRoot_189.Facts_1 ctx outputs

theorem sequential_generatedRoot_189.facts {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_189.Ctx backend hashModel params i_0) {outputs : mxxMat4 × mxxMat4 × mxxMat4 × mxxMat4 × Unit}
    (h : sequential_generatedRoot_189.body backend hashModel params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_189.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_189.facts_0 (ctx := ctx) (outputs := outputs) t0.1
  have t1 := t0.2
  have r1 := sequential_generatedRoot_189.facts_1 (ctx := ctx) (outputs := outputs) t1
  ⟨r0, r1⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) where
  inputs : ByteArray × mxxMat0 × ByteArray × (Fin 1 → Int) × (Fin 1 → Int) × (Fin 72000 → Int) × (Fin 72000 → Int) × (Fin 72000 → Int) × (Fin 72000 → Int) × (Fin 72000 → Int) × (Fin 72000 → Int) × (Fin 16 → Int) × (Fin 16 → Int) × (Fin 16 → Int) × (Fin 16 → Int) × (Fin 16 → Int) × (Fin 16 → Int) × (Fin 14400 → Int) × (Fin 14400 → Int) × (Fin 14400 → Int) × (Fin 14400 → Int) × (Fin 14400 → Int) × (Fin 14400 → Int) × Unit
  witness : generatedRoot.Witness

def generatedRoot.w_0_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : ByteArray :=
  ctx.inputs.1

def generatedRoot.w_1_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat0 :=
  ctx.inputs.2.1

def generatedRoot.w_17_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : ByteArray :=
  ctx.inputs.2.2.1

def generatedRoot.w_34_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 1 → Int :=
  ctx.inputs.2.2.2.1

def generatedRoot.w_35_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 1 → Int :=
  ctx.inputs.2.2.2.2.1

def generatedRoot.w_71_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 72000 → Int :=
  ctx.inputs.2.2.2.2.2.1

def generatedRoot.w_72_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 72000 → Int :=
  ctx.inputs.2.2.2.2.2.2.1

def generatedRoot.w_77_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 72000 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.1

def generatedRoot.w_78_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 72000 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.1

def generatedRoot.w_81_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 72000 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_82_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 72000 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_108_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 16 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_109_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 16 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_114_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 16 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_115_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 16 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_118_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 16 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_119_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 16 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_153_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 14400 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_154_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 14400 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_159_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 14400 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_160_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 14400 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_163_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 14400 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_164_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 14400 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_2_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat0 :=
  (1 : mxxMat0)

def generatedRoot.w_3_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 4 → mxxMat1 :=
  ctx.witness.w_3_0

def generatedRoot.w_3_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 4 → mxxMat1 :=
  ctx.witness.w_3_1

def generatedRoot.w_3_2 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 4 → mxxMat1 :=
  ctx.witness.w_3_2

def generatedRoot.w_3_3 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 4 → mxxMat1 :=
  ctx.witness.w_3_3

def generatedRoot.w_4_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 8 → mxxMat1 :=
  ctx.witness.w_4_0

def generatedRoot.w_4_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 8 → mxxMat1 :=
  ctx.witness.w_4_1

def generatedRoot.w_4_2 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 8 → mxxMat1 :=
  ctx.witness.w_4_2

def generatedRoot.w_4_3 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 8 → mxxMat1 :=
  ctx.witness.w_4_3

def generatedRoot.w_5_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 8 → mxxMat1 :=
  ctx.witness.w_5_0

def generatedRoot.w_5_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 8 → mxxMat1 :=
  ctx.witness.w_5_1

def generatedRoot.w_5_2 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 8 → mxxMat1 :=
  ctx.witness.w_5_2

def generatedRoot.w_5_3 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 8 → mxxMat1 :=
  ctx.witness.w_5_3

def generatedRoot.w_6_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 8 → mxxMat1 :=
  ctx.witness.w_6_0

def generatedRoot.w_6_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 8 → mxxMat1 :=
  ctx.witness.w_6_1

def generatedRoot.w_6_2 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 8 → mxxMat1 :=
  ctx.witness.w_6_2

def generatedRoot.w_6_3 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 8 → mxxMat1 :=
  ctx.witness.w_6_3

def generatedRoot.w_7_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 1 → mxxMat1 :=
  ctx.witness.w_7_0

def generatedRoot.w_7_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 1 → mxxMat1 :=
  ctx.witness.w_7_1

def generatedRoot.w_7_2 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 1 → mxxMat1 :=
  ctx.witness.w_7_2

def generatedRoot.w_7_3 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 1 → mxxMat1 :=
  ctx.witness.w_7_3

def generatedRoot.w_8_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 8 → mxxMat1 :=
  ctx.witness.w_8_0

def generatedRoot.w_8_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 8 → mxxMat1 :=
  ctx.witness.w_8_1

def generatedRoot.w_8_2 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 8 → mxxMat1 :=
  ctx.witness.w_8_2

def generatedRoot.w_8_3 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 8 → mxxMat1 :=
  ctx.witness.w_8_3

def generatedRoot.w_9_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 8 → mxxMat1 :=
  ctx.witness.w_9_0

def generatedRoot.w_9_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 8 → mxxMat1 :=
  ctx.witness.w_9_1

def generatedRoot.w_9_2 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 8 → mxxMat1 :=
  ctx.witness.w_9_2

def generatedRoot.w_9_3 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 8 → mxxMat1 :=
  ctx.witness.w_9_3

def generatedRoot.w_10_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 1 → mxxMat1 :=
  ctx.witness.w_10_0

def generatedRoot.w_10_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 1 → mxxMat1 :=
  ctx.witness.w_10_1

def generatedRoot.w_10_2 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 1 → mxxMat1 :=
  ctx.witness.w_10_2

def generatedRoot.w_10_3 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 1 → mxxMat1 :=
  ctx.witness.w_10_3

def generatedRoot.w_11_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 1 → mxxMat1 :=
  ctx.witness.w_11_0

def generatedRoot.w_11_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 1 → mxxMat1 :=
  ctx.witness.w_11_1

def generatedRoot.w_11_2 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 1 → mxxMat1 :=
  ctx.witness.w_11_2

def generatedRoot.w_11_3 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 1 → mxxMat1 :=
  ctx.witness.w_11_3

def generatedRoot.w_12_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 1 → mxxMat1 :=
  ctx.witness.w_12_0

def generatedRoot.w_12_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 1 → mxxMat1 :=
  ctx.witness.w_12_1

def generatedRoot.w_12_2 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 1 → mxxMat1 :=
  ctx.witness.w_12_2

def generatedRoot.w_12_3 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 1 → mxxMat1 :=
  ctx.witness.w_12_3

def generatedRoot.w_13_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 1 → mxxMat1 :=
  ctx.witness.w_13_0

def generatedRoot.w_13_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 1 → mxxMat1 :=
  ctx.witness.w_13_1

def generatedRoot.w_13_2 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 1 → mxxMat1 :=
  ctx.witness.w_13_2

def generatedRoot.w_13_3 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 1 → mxxMat1 :=
  ctx.witness.w_13_3

def generatedRoot.w_14_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 1 → mxxMat1 :=
  ctx.witness.w_14_0

def generatedRoot.w_14_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 1 → mxxMat1 :=
  ctx.witness.w_14_1

def generatedRoot.w_14_2 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 1 → mxxMat1 :=
  ctx.witness.w_14_2

def generatedRoot.w_14_3 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 1 → mxxMat1 :=
  ctx.witness.w_14_3

def generatedRoot.w_15_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 1 → mxxMat1 :=
  ctx.witness.w_15_0

def generatedRoot.w_15_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 1 → mxxMat1 :=
  ctx.witness.w_15_1

def generatedRoot.w_15_2 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 1 → mxxMat1 :=
  ctx.witness.w_15_2

def generatedRoot.w_15_3 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 1 → mxxMat1 :=
  ctx.witness.w_15_3

def generatedRoot.w_16_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  (0 : mxxMat1)

def generatedRoot.w_18_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_18_0

def generatedRoot.w_19_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_19_0

def generatedRoot.w_20_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat6 :=
  ctx.witness.w_20_0

def generatedRoot.w_21_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_21_0

def generatedRoot.w_22_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat7 :=
  ctx.witness.w_22_0

def generatedRoot.w_23_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat6 :=
  MxxRuntime.matrixMul (generatedRoot.w_20_0 ctx) (generatedRoot.w_22_0 ctx)

def generatedRoot.w_24_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_24_0

def generatedRoot.w_25_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_25_0

def generatedRoot.w_26_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat6 :=
  ctx.witness.w_26_0

def generatedRoot.w_27_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_27_0

def generatedRoot.w_28_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat7 :=
  ctx.witness.w_28_0

def generatedRoot.w_29_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat6 :=
  MxxRuntime.matrixMul (generatedRoot.w_26_0 ctx) (generatedRoot.w_28_0 ctx)

def generatedRoot.w_30_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat6 :=
  MxxRuntime.matrixAdd (generatedRoot.w_23_0 ctx) (generatedRoot.w_29_0 ctx)

def generatedRoot.w_31_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_31_0

def generatedRoot.w_32_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_32_0

def generatedRoot.w_33_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat6 :=
  ctx.witness.w_33_0

def generatedRoot.w_36_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 1 → mxxMat1 :=
  ctx.witness.w_36_0

def generatedRoot.w_36_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 1 → mxxMat1 :=
  ctx.witness.w_36_1

def generatedRoot.w_36_2 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 1 → mxxMat1 :=
  ctx.witness.w_36_2

def generatedRoot.w_36_3 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 1 → mxxMat1 :=
  ctx.witness.w_36_3

def generatedRoot.w_37_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_37_0

def generatedRoot.w_38_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat7 :=
  ctx.witness.w_38_0

def generatedRoot.w_39_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat7 :=
  ctx.witness.w_39_0

def generatedRoot.w_40_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat6 :=
  MxxRuntime.matrixMul (generatedRoot.w_33_0 ctx) (generatedRoot.w_39_0 ctx)

def generatedRoot.w_41_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_41_0

def generatedRoot.w_42_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat7 :=
  ctx.witness.w_42_0

def generatedRoot.w_43_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat6 :=
  MxxRuntime.matrixMul (generatedRoot.w_20_0 ctx) (generatedRoot.w_42_0 ctx)

def generatedRoot.w_44_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_44_0

def generatedRoot.w_45_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat7 :=
  ctx.witness.w_45_0

def generatedRoot.w_46_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat6 :=
  MxxRuntime.matrixMul (generatedRoot.w_26_0 ctx) (generatedRoot.w_45_0 ctx)

def generatedRoot.w_47_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat6 :=
  MxxRuntime.matrixAdd (generatedRoot.w_43_0 ctx) (generatedRoot.w_46_0 ctx)

def generatedRoot.w_48_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_48_0

def generatedRoot.w_49_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_49_0

def generatedRoot.w_50_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat6 :=
  ctx.witness.w_50_0

def generatedRoot.w_51_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_51_0

def generatedRoot.w_52_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat7 :=
  ctx.witness.w_52_0

def generatedRoot.w_53_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat7 :=
  ctx.witness.w_53_0

def generatedRoot.w_54_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat6 :=
  MxxRuntime.matrixMul (generatedRoot.w_50_0 ctx) (generatedRoot.w_53_0 ctx)

def generatedRoot.w_55_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat6 :=
  MxxRuntime.matrixAdd (generatedRoot.w_40_0 ctx) (generatedRoot.w_54_0 ctx)

def generatedRoot.w_56_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_56_0

def generatedRoot.w_57_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_57_0

def generatedRoot.w_58_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_58_0

def generatedRoot.w_59_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat7 :=
  ctx.witness.w_59_0

def generatedRoot.w_60_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat7 :=
  ctx.witness.w_60_0

def generatedRoot.w_61_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat6 :=
  MxxRuntime.matrixMul (generatedRoot.w_33_0 ctx) (generatedRoot.w_60_0 ctx)

def generatedRoot.w_62_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_62_0

def generatedRoot.w_63_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat7 :=
  ctx.witness.w_63_0

def generatedRoot.w_64_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat7 :=
  ctx.witness.w_64_0

def generatedRoot.w_65_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat6 :=
  MxxRuntime.matrixMul (generatedRoot.w_50_0 ctx) (generatedRoot.w_64_0 ctx)

def generatedRoot.w_66_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat6 :=
  MxxRuntime.matrixAdd (generatedRoot.w_61_0 ctx) (generatedRoot.w_65_0 ctx)

def generatedRoot.w_67_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_67_0

def generatedRoot.w_68_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_68_0

def generatedRoot.w_69_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_69_0

def generatedRoot.w_69_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_69_1

def generatedRoot.w_69_2 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_69_2

def generatedRoot.w_69_3 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_69_3

def generatedRoot.w_70_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat3 :=
  (MxxRuntime.unitRow (0) : mxxMat3)

def generatedRoot.w_73_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_73_0

def generatedRoot.w_74_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_74_0

def generatedRoot.w_75_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_75_0

def generatedRoot.w_76_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_76_0

def generatedRoot.w_79_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 72000 → mxxMat1 :=
  ctx.witness.w_79_0

def generatedRoot.w_79_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 72000 → mxxMat1 :=
  ctx.witness.w_79_1

def generatedRoot.w_79_2 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 72000 → mxxMat1 :=
  ctx.witness.w_79_2

def generatedRoot.w_79_3 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 72000 → mxxMat1 :=
  ctx.witness.w_79_3

def generatedRoot.w_79_4 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 72000 → mxxMat1 :=
  ctx.witness.w_79_4

def generatedRoot.w_79_5 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 72000 → mxxMat1 :=
  ctx.witness.w_79_5

def generatedRoot.w_79_6 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 72000 → mxxMat1 :=
  ctx.witness.w_79_6

def generatedRoot.w_79_7 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 72000 → mxxMat1 :=
  ctx.witness.w_79_7

def generatedRoot.w_80_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 36000 → mxxMat1 :=
  ctx.witness.w_80_0

def generatedRoot.w_80_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 36000 → mxxMat1 :=
  ctx.witness.w_80_1

def generatedRoot.w_80_2 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 36000 → mxxMat1 :=
  ctx.witness.w_80_2

def generatedRoot.w_80_3 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 36000 → mxxMat1 :=
  ctx.witness.w_80_3

def generatedRoot.w_83_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 72000 → mxxMat1 :=
  ctx.witness.w_83_0

def generatedRoot.w_83_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 72000 → mxxMat1 :=
  ctx.witness.w_83_1

def generatedRoot.w_83_2 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 72000 → mxxMat1 :=
  ctx.witness.w_83_2

def generatedRoot.w_83_3 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 72000 → mxxMat1 :=
  ctx.witness.w_83_3

def generatedRoot.w_83_4 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 72000 → mxxMat1 :=
  ctx.witness.w_83_4

def generatedRoot.w_83_5 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 72000 → mxxMat1 :=
  ctx.witness.w_83_5

def generatedRoot.w_83_6 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 72000 → mxxMat1 :=
  ctx.witness.w_83_6

def generatedRoot.w_83_7 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 72000 → mxxMat1 :=
  ctx.witness.w_83_7

def generatedRoot.w_84_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 36000 → mxxMat1 :=
  ctx.witness.w_84_0

def generatedRoot.w_84_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 36000 → mxxMat1 :=
  ctx.witness.w_84_1

def generatedRoot.w_84_2 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 36000 → mxxMat1 :=
  ctx.witness.w_84_2

def generatedRoot.w_84_3 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 36000 → mxxMat1 :=
  ctx.witness.w_84_3

def generatedRoot.w_85_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 36000 → mxxMat1 :=
  ctx.witness.w_85_0

def generatedRoot.w_85_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 36000 → mxxMat1 :=
  ctx.witness.w_85_1

def generatedRoot.w_85_2 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 36000 → mxxMat1 :=
  ctx.witness.w_85_2

def generatedRoot.w_85_3 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 36000 → mxxMat1 :=
  ctx.witness.w_85_3

def generatedRoot.w_86_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 36000 → mxxMat1 :=
  ctx.witness.w_86_0

def generatedRoot.w_86_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 36000 → mxxMat1 :=
  ctx.witness.w_86_1

def generatedRoot.w_86_2 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 36000 → mxxMat1 :=
  ctx.witness.w_86_2

def generatedRoot.w_86_3 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 36000 → mxxMat1 :=
  ctx.witness.w_86_3

def generatedRoot.w_87_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_87_0

def generatedRoot.w_88_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_88_0

def generatedRoot.w_89_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  MxxRuntime.matrixAdd (generatedRoot.w_87_0 ctx) (generatedRoot.w_88_0 ctx)

def generatedRoot.w_90_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_90_0

def generatedRoot.w_91_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_91_0

def generatedRoot.w_92_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  MxxRuntime.matrixAdd (generatedRoot.w_90_0 ctx) (generatedRoot.w_91_0 ctx)

def generatedRoot.w_93_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_93_0

def generatedRoot.w_94_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_94_0

def generatedRoot.w_95_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  MxxRuntime.matrixAdd (generatedRoot.w_93_0 ctx) (generatedRoot.w_94_0 ctx)

def generatedRoot.w_96_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_96_0

def generatedRoot.w_97_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_97_0

def generatedRoot.w_98_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  MxxRuntime.matrixAdd (generatedRoot.w_96_0 ctx) (generatedRoot.w_97_0 ctx)

def generatedRoot.w_99_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_99_0

def generatedRoot.w_99_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_99_1

def generatedRoot.w_99_2 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_99_2

def generatedRoot.w_99_3 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_99_3

def generatedRoot.w_100_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat2 :=
  ctx.witness.w_100_0

def generatedRoot.w_101_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat3 :=
  (MxxRuntime.unitRow (0) : mxxMat3)

def generatedRoot.w_102_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat2 :=
  ctx.witness.w_102_0

def generatedRoot.w_103_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat3 :=
  (MxxRuntime.unitRow (0) : mxxMat3)

def generatedRoot.w_104_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat2 :=
  ctx.witness.w_104_0

def generatedRoot.w_105_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat3 :=
  (MxxRuntime.unitRow (0) : mxxMat3)

def generatedRoot.w_106_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat2 :=
  ctx.witness.w_106_0

def generatedRoot.w_107_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat2 :=
  ctx.witness.w_107_0

def generatedRoot.w_107_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat2 :=
  ctx.witness.w_107_1

def generatedRoot.w_107_2 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat2 :=
  ctx.witness.w_107_2

def generatedRoot.w_107_3 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat2 :=
  ctx.witness.w_107_3

def generatedRoot.w_110_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_110_0

def generatedRoot.w_111_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_111_0

def generatedRoot.w_112_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_112_0

def generatedRoot.w_113_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_113_0

def generatedRoot.w_116_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 16 → mxxMat1 :=
  ctx.witness.w_116_0

def generatedRoot.w_116_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 16 → mxxMat1 :=
  ctx.witness.w_116_1

def generatedRoot.w_116_2 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 16 → mxxMat1 :=
  ctx.witness.w_116_2

def generatedRoot.w_116_3 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 16 → mxxMat1 :=
  ctx.witness.w_116_3

def generatedRoot.w_116_4 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 16 → mxxMat1 :=
  ctx.witness.w_116_4

def generatedRoot.w_116_5 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 16 → mxxMat1 :=
  ctx.witness.w_116_5

def generatedRoot.w_116_6 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 16 → mxxMat1 :=
  ctx.witness.w_116_6

def generatedRoot.w_116_7 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 16 → mxxMat1 :=
  ctx.witness.w_116_7

def generatedRoot.w_117_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 8 → mxxMat1 :=
  ctx.witness.w_117_0

def generatedRoot.w_117_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 8 → mxxMat1 :=
  ctx.witness.w_117_1

def generatedRoot.w_117_2 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 8 → mxxMat1 :=
  ctx.witness.w_117_2

def generatedRoot.w_117_3 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 8 → mxxMat1 :=
  ctx.witness.w_117_3

def generatedRoot.w_120_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 16 → mxxMat1 :=
  ctx.witness.w_120_0

def generatedRoot.w_120_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 16 → mxxMat1 :=
  ctx.witness.w_120_1

def generatedRoot.w_120_2 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 16 → mxxMat1 :=
  ctx.witness.w_120_2

def generatedRoot.w_120_3 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 16 → mxxMat1 :=
  ctx.witness.w_120_3

def generatedRoot.w_120_4 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 16 → mxxMat1 :=
  ctx.witness.w_120_4

def generatedRoot.w_120_5 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 16 → mxxMat1 :=
  ctx.witness.w_120_5

def generatedRoot.w_120_6 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 16 → mxxMat1 :=
  ctx.witness.w_120_6

def generatedRoot.w_120_7 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 16 → mxxMat1 :=
  ctx.witness.w_120_7

def generatedRoot.w_121_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 8 → mxxMat1 :=
  ctx.witness.w_121_0

def generatedRoot.w_121_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 8 → mxxMat1 :=
  ctx.witness.w_121_1

def generatedRoot.w_121_2 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 8 → mxxMat1 :=
  ctx.witness.w_121_2

def generatedRoot.w_121_3 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 8 → mxxMat1 :=
  ctx.witness.w_121_3

def generatedRoot.w_122_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 8 → mxxMat1 :=
  ctx.witness.w_122_0

def generatedRoot.w_122_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 8 → mxxMat1 :=
  ctx.witness.w_122_1

def generatedRoot.w_122_2 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 8 → mxxMat1 :=
  ctx.witness.w_122_2

def generatedRoot.w_122_3 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 8 → mxxMat1 :=
  ctx.witness.w_122_3

def generatedRoot.w_123_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 8 → mxxMat1 :=
  ctx.witness.w_123_0

def generatedRoot.w_123_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 8 → mxxMat1 :=
  ctx.witness.w_123_1

def generatedRoot.w_123_2 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 8 → mxxMat1 :=
  ctx.witness.w_123_2

def generatedRoot.w_123_3 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 8 → mxxMat1 :=
  ctx.witness.w_123_3

def generatedRoot.w_124_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 8 → mxxMat1 :=
  ctx.witness.w_124_0

def generatedRoot.w_124_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 8 → mxxMat1 :=
  ctx.witness.w_124_1

def generatedRoot.w_124_2 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 8 → mxxMat1 :=
  ctx.witness.w_124_2

def generatedRoot.w_124_3 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 8 → mxxMat1 :=
  ctx.witness.w_124_3

def generatedRoot.w_125_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_125_0

def generatedRoot.w_126_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_126_0

def generatedRoot.w_127_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  MxxRuntime.matrixAdd (generatedRoot.w_125_0 ctx) (generatedRoot.w_126_0 ctx)

def generatedRoot.w_128_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_128_0

def generatedRoot.w_129_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_129_0

def generatedRoot.w_130_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  MxxRuntime.matrixAdd (generatedRoot.w_128_0 ctx) (generatedRoot.w_129_0 ctx)

def generatedRoot.w_131_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_131_0

def generatedRoot.w_132_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_132_0

def generatedRoot.w_133_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  MxxRuntime.matrixAdd (generatedRoot.w_131_0 ctx) (generatedRoot.w_132_0 ctx)

def generatedRoot.w_134_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_134_0

def generatedRoot.w_135_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_135_0

def generatedRoot.w_136_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  MxxRuntime.matrixAdd (generatedRoot.w_134_0 ctx) (generatedRoot.w_135_0 ctx)

def generatedRoot.w_137_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_137_0

def generatedRoot.w_137_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_137_1

def generatedRoot.w_137_2 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_137_2

def generatedRoot.w_137_3 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_137_3

def generatedRoot.w_138_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  (0 : mxxMat1)

def generatedRoot.w_139_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_139_0

def generatedRoot.w_139_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_139_1

def generatedRoot.w_139_2 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_139_2

def generatedRoot.w_139_3 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_139_3

def generatedRoot.w_140_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  (0 : mxxMat1)

def generatedRoot.w_141_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_141_0

def generatedRoot.w_141_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_141_1

def generatedRoot.w_141_2 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_141_2

def generatedRoot.w_141_3 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_141_3

def generatedRoot.w_142_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  (0 : mxxMat1)

def generatedRoot.w_143_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_143_0

def generatedRoot.w_143_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_143_1

def generatedRoot.w_143_2 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_143_2

def generatedRoot.w_143_3 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_143_3

def generatedRoot.w_144_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  (0 : mxxMat1)

def generatedRoot.w_145_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_145_0

def generatedRoot.w_145_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_145_1

def generatedRoot.w_145_2 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_145_2

def generatedRoot.w_145_3 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_145_3

def generatedRoot.w_146_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  (0 : mxxMat1)

def generatedRoot.w_147_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_147_0

def generatedRoot.w_147_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_147_1

def generatedRoot.w_147_2 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_147_2

def generatedRoot.w_147_3 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_147_3

def generatedRoot.w_148_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  (0 : mxxMat1)

def generatedRoot.w_149_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_149_0

def generatedRoot.w_149_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_149_1

def generatedRoot.w_149_2 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_149_2

def generatedRoot.w_149_3 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_149_3

def generatedRoot.w_150_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  (0 : mxxMat1)

def generatedRoot.w_151_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_151_0

def generatedRoot.w_151_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_151_1

def generatedRoot.w_151_2 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_151_2

def generatedRoot.w_151_3 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_151_3

def generatedRoot.w_152_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat5 :=
  (MxxRuntime.unitRow (0) : mxxMat5)

def generatedRoot.w_155_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_155_0

def generatedRoot.w_156_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_156_0

def generatedRoot.w_157_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_157_0

def generatedRoot.w_158_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_158_0

def generatedRoot.w_161_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 14400 → mxxMat1 :=
  ctx.witness.w_161_0

def generatedRoot.w_161_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 14400 → mxxMat1 :=
  ctx.witness.w_161_1

def generatedRoot.w_161_2 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 14400 → mxxMat1 :=
  ctx.witness.w_161_2

def generatedRoot.w_161_3 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 14400 → mxxMat1 :=
  ctx.witness.w_161_3

def generatedRoot.w_161_4 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 14400 → mxxMat1 :=
  ctx.witness.w_161_4

def generatedRoot.w_161_5 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 14400 → mxxMat1 :=
  ctx.witness.w_161_5

def generatedRoot.w_161_6 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 14400 → mxxMat1 :=
  ctx.witness.w_161_6

def generatedRoot.w_161_7 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 14400 → mxxMat1 :=
  ctx.witness.w_161_7

def generatedRoot.w_162_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 7200 → mxxMat1 :=
  ctx.witness.w_162_0

def generatedRoot.w_162_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 7200 → mxxMat1 :=
  ctx.witness.w_162_1

def generatedRoot.w_162_2 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 7200 → mxxMat1 :=
  ctx.witness.w_162_2

def generatedRoot.w_162_3 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 7200 → mxxMat1 :=
  ctx.witness.w_162_3

def generatedRoot.w_165_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 14400 → mxxMat1 :=
  ctx.witness.w_165_0

def generatedRoot.w_165_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 14400 → mxxMat1 :=
  ctx.witness.w_165_1

def generatedRoot.w_165_2 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 14400 → mxxMat1 :=
  ctx.witness.w_165_2

def generatedRoot.w_165_3 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 14400 → mxxMat1 :=
  ctx.witness.w_165_3

def generatedRoot.w_165_4 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 14400 → mxxMat1 :=
  ctx.witness.w_165_4

def generatedRoot.w_165_5 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 14400 → mxxMat1 :=
  ctx.witness.w_165_5

def generatedRoot.w_165_6 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 14400 → mxxMat1 :=
  ctx.witness.w_165_6

def generatedRoot.w_165_7 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 14400 → mxxMat1 :=
  ctx.witness.w_165_7

def generatedRoot.w_166_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 7200 → mxxMat1 :=
  ctx.witness.w_166_0

def generatedRoot.w_166_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 7200 → mxxMat1 :=
  ctx.witness.w_166_1

def generatedRoot.w_166_2 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 7200 → mxxMat1 :=
  ctx.witness.w_166_2

def generatedRoot.w_166_3 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 7200 → mxxMat1 :=
  ctx.witness.w_166_3

def generatedRoot.w_167_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 7200 → mxxMat1 :=
  ctx.witness.w_167_0

def generatedRoot.w_167_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 7200 → mxxMat1 :=
  ctx.witness.w_167_1

def generatedRoot.w_167_2 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 7200 → mxxMat1 :=
  ctx.witness.w_167_2

def generatedRoot.w_167_3 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 7200 → mxxMat1 :=
  ctx.witness.w_167_3

def generatedRoot.w_168_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 7200 → mxxMat1 :=
  ctx.witness.w_168_0

def generatedRoot.w_168_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 7200 → mxxMat1 :=
  ctx.witness.w_168_1

def generatedRoot.w_168_2 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 7200 → mxxMat1 :=
  ctx.witness.w_168_2

def generatedRoot.w_168_3 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Fin 7200 → mxxMat1 :=
  ctx.witness.w_168_3

def generatedRoot.w_169_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_169_0

def generatedRoot.w_170_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_170_0

def generatedRoot.w_171_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  MxxRuntime.matrixAdd (generatedRoot.w_169_0 ctx) (generatedRoot.w_170_0 ctx)

def generatedRoot.w_172_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_172_0

def generatedRoot.w_173_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_173_0

def generatedRoot.w_174_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  MxxRuntime.matrixAdd (generatedRoot.w_172_0 ctx) (generatedRoot.w_173_0 ctx)

def generatedRoot.w_175_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_175_0

def generatedRoot.w_176_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_176_0

def generatedRoot.w_177_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  MxxRuntime.matrixAdd (generatedRoot.w_175_0 ctx) (generatedRoot.w_176_0 ctx)

def generatedRoot.w_178_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_178_0

def generatedRoot.w_179_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_179_0

def generatedRoot.w_180_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  MxxRuntime.matrixAdd (generatedRoot.w_178_0 ctx) (generatedRoot.w_179_0 ctx)

def generatedRoot.w_181_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_181_0

def generatedRoot.w_181_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_181_1

def generatedRoot.w_181_2 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_181_2

def generatedRoot.w_181_3 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_181_3

def generatedRoot.w_182_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_182_0

def generatedRoot.w_183_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat5 :=
  (MxxRuntime.unitRow (0) : mxxMat5)

def generatedRoot.w_184_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_184_0

def generatedRoot.w_185_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat5 :=
  (MxxRuntime.unitRow (0) : mxxMat5)

def generatedRoot.w_186_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_186_0

def generatedRoot.w_187_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat5 :=
  (MxxRuntime.unitRow (0) : mxxMat5)

def generatedRoot.w_188_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_188_0

def generatedRoot.w_189_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_189_0

def generatedRoot.w_189_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_189_1

def generatedRoot.w_189_2 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_189_2

def generatedRoot.w_189_3 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat4 :=
  ctx.witness.w_189_3

def generatedRoot.w_190_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_190_0

def generatedRoot.w_191_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_191_0

def generatedRoot.w_192_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat6 :=
  ctx.witness.w_192_0

def generatedRoot.w_193_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_193_0

def generatedRoot.w_194_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat7 :=
  ctx.witness.w_194_0

def generatedRoot.w_195_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat6 :=
  MxxRuntime.matrixMul (generatedRoot.w_192_0 ctx) (generatedRoot.w_194_0 ctx)

def generatedRoot.w_196_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_196_0

def generatedRoot.w_197_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_197_0

def generatedRoot.w_198_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat6 :=
  ctx.witness.w_198_0

def generatedRoot.w_199_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_199_0

def generatedRoot.w_200_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat7 :=
  ctx.witness.w_200_0

def generatedRoot.w_201_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat6 :=
  MxxRuntime.matrixMul (generatedRoot.w_198_0 ctx) (generatedRoot.w_200_0 ctx)

def generatedRoot.w_202_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat6 :=
  MxxRuntime.matrixAdd (generatedRoot.w_195_0 ctx) (generatedRoot.w_201_0 ctx)

def generatedRoot.w_203_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_203_0

def generatedRoot.w_204_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_204_0

def generatedRoot.w_205_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat7 :=
  ctx.witness.w_205_0

def generatedRoot.w_206_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat6 :=
  MxxRuntime.matrixMul (generatedRoot.w_192_0 ctx) (generatedRoot.w_205_0 ctx)

def generatedRoot.w_207_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_207_0

def generatedRoot.w_208_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat7 :=
  ctx.witness.w_208_0

def generatedRoot.w_209_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat6 :=
  MxxRuntime.matrixMul (generatedRoot.w_198_0 ctx) (generatedRoot.w_208_0 ctx)

def generatedRoot.w_210_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat6 :=
  MxxRuntime.matrixAdd (generatedRoot.w_206_0 ctx) (generatedRoot.w_209_0 ctx)

def generatedRoot.w_211_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_211_0

def generatedRoot.w_212_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_212_0

def generatedRoot.w_213_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_213_0

def generatedRoot.w_214_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_214_0

def generatedRoot.w_215_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_215_0

def generatedRoot.w_216_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat6 :=
  ctx.witness.w_216_0

def generatedRoot.w_217_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_217_0

def generatedRoot.w_218_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat7 :=
  ctx.witness.w_218_0

def generatedRoot.w_219_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat6 :=
  MxxRuntime.matrixMul (generatedRoot.w_216_0 ctx) (generatedRoot.w_218_0 ctx)

def generatedRoot.w_220_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_220_0

def generatedRoot.w_221_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_221_0

def generatedRoot.w_222_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat6 :=
  ctx.witness.w_222_0

def generatedRoot.w_223_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_223_0

def generatedRoot.w_224_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat7 :=
  ctx.witness.w_224_0

def generatedRoot.w_225_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat6 :=
  MxxRuntime.matrixMul (generatedRoot.w_222_0 ctx) (generatedRoot.w_224_0 ctx)

def generatedRoot.w_226_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat6 :=
  MxxRuntime.matrixAdd (generatedRoot.w_219_0 ctx) (generatedRoot.w_225_0 ctx)

def generatedRoot.w_227_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_227_0

def generatedRoot.w_228_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_228_0

def generatedRoot.w_229_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat6 :=
  ctx.witness.w_229_0

def generatedRoot.w_230_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_230_0

def generatedRoot.w_231_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_231_0

def generatedRoot.w_232_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat6 :=
  ctx.witness.w_232_0

def generatedRoot.w_233_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_233_0

def generatedRoot.w_234_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat7 :=
  ctx.witness.w_234_0

def generatedRoot.w_235_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat6 :=
  MxxRuntime.matrixMul (generatedRoot.w_232_0 ctx) (generatedRoot.w_234_0 ctx)

def generatedRoot.w_236_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_236_0

def generatedRoot.w_237_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_237_0

def generatedRoot.w_238_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat6 :=
  ctx.witness.w_238_0

def generatedRoot.w_239_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_239_0

def generatedRoot.w_240_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat7 :=
  ctx.witness.w_240_0

def generatedRoot.w_241_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat6 :=
  MxxRuntime.matrixMul (generatedRoot.w_238_0 ctx) (generatedRoot.w_240_0 ctx)

def generatedRoot.w_242_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat6 :=
  MxxRuntime.matrixAdd (generatedRoot.w_235_0 ctx) (generatedRoot.w_241_0 ctx)

def generatedRoot.w_243_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_243_0

def generatedRoot.w_244_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat7 :=
  ctx.witness.w_244_0

def generatedRoot.w_245_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat6 :=
  MxxRuntime.matrixMul (generatedRoot.w_229_0 ctx) (generatedRoot.w_244_0 ctx)

def generatedRoot.w_246_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_246_0

def generatedRoot.w_247_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat7 :=
  ctx.witness.w_247_0

def generatedRoot.w_248_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat6 :=
  MxxRuntime.matrixMul (generatedRoot.w_216_0 ctx) (generatedRoot.w_247_0 ctx)

def generatedRoot.w_249_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_249_0

def generatedRoot.w_250_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat7 :=
  ctx.witness.w_250_0

def generatedRoot.w_251_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat6 :=
  MxxRuntime.matrixMul (generatedRoot.w_222_0 ctx) (generatedRoot.w_250_0 ctx)

def generatedRoot.w_252_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat6 :=
  MxxRuntime.matrixAdd (generatedRoot.w_248_0 ctx) (generatedRoot.w_251_0 ctx)

def generatedRoot.w_253_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_253_0

def generatedRoot.w_254_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_254_0

def generatedRoot.w_255_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat6 :=
  ctx.witness.w_255_0

def generatedRoot.w_256_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_256_0

def generatedRoot.w_257_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat7 :=
  ctx.witness.w_257_0

def generatedRoot.w_258_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat6 :=
  MxxRuntime.matrixMul (generatedRoot.w_255_0 ctx) (generatedRoot.w_257_0 ctx)

def generatedRoot.w_259_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat6 :=
  MxxRuntime.matrixAdd (generatedRoot.w_245_0 ctx) (generatedRoot.w_258_0 ctx)

def generatedRoot.w_260_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_260_0

def generatedRoot.w_261_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_261_0

def generatedRoot.w_262_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat7 :=
  ctx.witness.w_262_0

def generatedRoot.w_263_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat6 :=
  MxxRuntime.matrixMul (generatedRoot.w_232_0 ctx) (generatedRoot.w_262_0 ctx)

def generatedRoot.w_264_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_264_0

def generatedRoot.w_265_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat7 :=
  ctx.witness.w_265_0

def generatedRoot.w_266_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat6 :=
  MxxRuntime.matrixMul (generatedRoot.w_238_0 ctx) (generatedRoot.w_265_0 ctx)

def generatedRoot.w_267_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat6 :=
  MxxRuntime.matrixAdd (generatedRoot.w_263_0 ctx) (generatedRoot.w_266_0 ctx)

def generatedRoot.w_268_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_268_0

def generatedRoot.w_269_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat7 :=
  ctx.witness.w_269_0

def generatedRoot.w_270_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat6 :=
  MxxRuntime.matrixMul (generatedRoot.w_229_0 ctx) (generatedRoot.w_269_0 ctx)

def generatedRoot.w_271_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_271_0

def generatedRoot.w_272_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat7 :=
  ctx.witness.w_272_0

def generatedRoot.w_273_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat6 :=
  MxxRuntime.matrixMul (generatedRoot.w_255_0 ctx) (generatedRoot.w_272_0 ctx)

def generatedRoot.w_274_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat6 :=
  MxxRuntime.matrixAdd (generatedRoot.w_270_0 ctx) (generatedRoot.w_273_0 ctx)

def generatedRoot.w_275_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_275_0

def generatedRoot.w_276_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_276_0

def generatedRoot.w_277_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_277_0

def generatedRoot.w_278_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_278_0

def generatedRoot.w_279_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_279_0

def generatedRoot.w_280_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat6 :=
  ctx.witness.w_280_0

def generatedRoot.w_281_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_281_0

def generatedRoot.w_282_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat7 :=
  ctx.witness.w_282_0

def generatedRoot.w_283_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat6 :=
  MxxRuntime.matrixMul (generatedRoot.w_280_0 ctx) (generatedRoot.w_282_0 ctx)

def generatedRoot.w_284_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_284_0

def generatedRoot.w_285_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_285_0

def generatedRoot.w_286_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat6 :=
  ctx.witness.w_286_0

def generatedRoot.w_287_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_287_0

def generatedRoot.w_288_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat7 :=
  ctx.witness.w_288_0

def generatedRoot.w_289_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat6 :=
  MxxRuntime.matrixMul (generatedRoot.w_286_0 ctx) (generatedRoot.w_288_0 ctx)

def generatedRoot.w_290_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat6 :=
  MxxRuntime.matrixAdd (generatedRoot.w_283_0 ctx) (generatedRoot.w_289_0 ctx)

def generatedRoot.w_291_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_291_0

def generatedRoot.w_292_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_292_0

def generatedRoot.w_293_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat6 :=
  ctx.witness.w_293_0

def generatedRoot.w_294_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_294_0

def generatedRoot.w_295_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat7 :=
  ctx.witness.w_295_0

def generatedRoot.w_296_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat6 :=
  MxxRuntime.matrixMul (generatedRoot.w_293_0 ctx) (generatedRoot.w_295_0 ctx)

def generatedRoot.w_297_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_297_0

def generatedRoot.w_298_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat7 :=
  ctx.witness.w_298_0

def generatedRoot.w_299_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat6 :=
  MxxRuntime.matrixMul (generatedRoot.w_280_0 ctx) (generatedRoot.w_298_0 ctx)

def generatedRoot.w_300_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_300_0

def generatedRoot.w_301_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat7 :=
  ctx.witness.w_301_0

def generatedRoot.w_302_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat6 :=
  MxxRuntime.matrixMul (generatedRoot.w_286_0 ctx) (generatedRoot.w_301_0 ctx)

def generatedRoot.w_303_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat6 :=
  MxxRuntime.matrixAdd (generatedRoot.w_299_0 ctx) (generatedRoot.w_302_0 ctx)

def generatedRoot.w_304_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_304_0

def generatedRoot.w_305_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_305_0

def generatedRoot.w_306_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat6 :=
  ctx.witness.w_306_0

def generatedRoot.w_307_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_307_0

def generatedRoot.w_308_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat7 :=
  ctx.witness.w_308_0

def generatedRoot.w_309_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat6 :=
  MxxRuntime.matrixMul (generatedRoot.w_306_0 ctx) (generatedRoot.w_308_0 ctx)

def generatedRoot.w_310_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat6 :=
  MxxRuntime.matrixAdd (generatedRoot.w_296_0 ctx) (generatedRoot.w_309_0 ctx)

def generatedRoot.w_311_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_311_0

def generatedRoot.w_312_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_312_0

def generatedRoot.w_313_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat7 :=
  ctx.witness.w_313_0

def generatedRoot.w_314_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat6 :=
  MxxRuntime.matrixMul (generatedRoot.w_293_0 ctx) (generatedRoot.w_313_0 ctx)

def generatedRoot.w_315_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_315_0

def generatedRoot.w_316_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat7 :=
  ctx.witness.w_316_0

def generatedRoot.w_317_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat6 :=
  MxxRuntime.matrixMul (generatedRoot.w_306_0 ctx) (generatedRoot.w_316_0 ctx)

def generatedRoot.w_318_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat6 :=
  MxxRuntime.matrixAdd (generatedRoot.w_314_0 ctx) (generatedRoot.w_317_0 ctx)

def generatedRoot.w_319_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_319_0

def generatedRoot.w_320_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_320_0

def generatedRoot.w_321_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_321_0

def generatedRoot.w_322_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_322_0

def generatedRoot.w_323_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_323_0

def generatedRoot.w_324_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat6 :=
  ctx.witness.w_324_0

def generatedRoot.w_325_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_325_0

def generatedRoot.w_326_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat7 :=
  ctx.witness.w_326_0

def generatedRoot.w_327_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat6 :=
  MxxRuntime.matrixMul (generatedRoot.w_324_0 ctx) (generatedRoot.w_326_0 ctx)

def generatedRoot.w_328_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_328_0

def generatedRoot.w_329_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_329_0

def generatedRoot.w_330_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat6 :=
  ctx.witness.w_330_0

def generatedRoot.w_331_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_331_0

def generatedRoot.w_332_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat7 :=
  ctx.witness.w_332_0

def generatedRoot.w_333_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat6 :=
  MxxRuntime.matrixMul (generatedRoot.w_330_0 ctx) (generatedRoot.w_332_0 ctx)

def generatedRoot.w_334_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat6 :=
  MxxRuntime.matrixAdd (generatedRoot.w_327_0 ctx) (generatedRoot.w_333_0 ctx)

def generatedRoot.w_335_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_335_0

def generatedRoot.w_336_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_336_0

def generatedRoot.w_337_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat7 :=
  ctx.witness.w_337_0

def generatedRoot.w_338_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat6 :=
  MxxRuntime.matrixMul (generatedRoot.w_324_0 ctx) (generatedRoot.w_337_0 ctx)

def generatedRoot.w_339_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_339_0

def generatedRoot.w_340_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat7 :=
  ctx.witness.w_340_0

def generatedRoot.w_341_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat6 :=
  MxxRuntime.matrixMul (generatedRoot.w_330_0 ctx) (generatedRoot.w_340_0 ctx)

def generatedRoot.w_342_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat6 :=
  MxxRuntime.matrixAdd (generatedRoot.w_338_0 ctx) (generatedRoot.w_341_0 ctx)

def generatedRoot.w_343_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_343_0

def generatedRoot.w_344_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_344_0

def generatedRoot.w_345_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_345_0

def generatedRoot.w_346_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_346_0

def generatedRoot.w_347_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_347_0

def generatedRoot.w_348_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat6 :=
  ctx.witness.w_348_0

def generatedRoot.w_349_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_349_0

def generatedRoot.w_350_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat7 :=
  ctx.witness.w_350_0

def generatedRoot.w_351_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat6 :=
  MxxRuntime.matrixMul (generatedRoot.w_348_0 ctx) (generatedRoot.w_350_0 ctx)

def generatedRoot.w_352_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_352_0

def generatedRoot.w_353_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_353_0

def generatedRoot.w_354_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat6 :=
  ctx.witness.w_354_0

def generatedRoot.w_355_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_355_0

def generatedRoot.w_356_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat7 :=
  ctx.witness.w_356_0

def generatedRoot.w_357_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat6 :=
  MxxRuntime.matrixMul (generatedRoot.w_354_0 ctx) (generatedRoot.w_356_0 ctx)

def generatedRoot.w_358_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat6 :=
  MxxRuntime.matrixAdd (generatedRoot.w_351_0 ctx) (generatedRoot.w_357_0 ctx)

def generatedRoot.w_359_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_359_0

def generatedRoot.w_360_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_360_0

def generatedRoot.w_361_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat7 :=
  ctx.witness.w_361_0

def generatedRoot.w_362_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat6 :=
  MxxRuntime.matrixMul (generatedRoot.w_348_0 ctx) (generatedRoot.w_361_0 ctx)

def generatedRoot.w_363_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_363_0

def generatedRoot.w_364_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat7 :=
  ctx.witness.w_364_0

def generatedRoot.w_365_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat6 :=
  MxxRuntime.matrixMul (generatedRoot.w_354_0 ctx) (generatedRoot.w_364_0 ctx)

def generatedRoot.w_366_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat6 :=
  MxxRuntime.matrixAdd (generatedRoot.w_362_0 ctx) (generatedRoot.w_365_0 ctx)

def generatedRoot.w_367_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_367_0

def generatedRoot.w_368_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_368_0

def generatedRoot.w_369_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : mxxMat1 :=
  ctx.witness.w_369_0

def generatedRoot.outs {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat4 × mxxMat4 × mxxMat4 × mxxMat4 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × Unit :=
  ((generatedRoot.w_3_0 ctx), ((generatedRoot.w_3_1 ctx), ((generatedRoot.w_3_2 ctx), ((generatedRoot.w_3_3 ctx), ((generatedRoot.w_4_0 ctx), ((generatedRoot.w_4_1 ctx), ((generatedRoot.w_4_2 ctx), ((generatedRoot.w_4_3 ctx), ((generatedRoot.w_5_0 ctx), ((generatedRoot.w_5_1 ctx), ((generatedRoot.w_5_2 ctx), ((generatedRoot.w_5_3 ctx), ((generatedRoot.w_6_0 ctx), ((generatedRoot.w_6_1 ctx), ((generatedRoot.w_6_2 ctx), ((generatedRoot.w_6_3 ctx), ((generatedRoot.w_7_0 ctx), ((generatedRoot.w_7_1 ctx), ((generatedRoot.w_7_2 ctx), ((generatedRoot.w_7_3 ctx), ((generatedRoot.w_8_0 ctx), ((generatedRoot.w_8_1 ctx), ((generatedRoot.w_8_2 ctx), ((generatedRoot.w_8_3 ctx), ((generatedRoot.w_9_0 ctx), ((generatedRoot.w_9_1 ctx), ((generatedRoot.w_9_2 ctx), ((generatedRoot.w_9_3 ctx), ((generatedRoot.w_10_0 ctx), ((generatedRoot.w_10_1 ctx), ((generatedRoot.w_10_2 ctx), ((generatedRoot.w_10_3 ctx), ((generatedRoot.w_11_0 ctx), ((generatedRoot.w_11_1 ctx), ((generatedRoot.w_11_2 ctx), ((generatedRoot.w_11_3 ctx), ((generatedRoot.w_12_0 ctx), ((generatedRoot.w_12_1 ctx), ((generatedRoot.w_12_2 ctx), ((generatedRoot.w_12_3 ctx), ((generatedRoot.w_13_0 ctx), ((generatedRoot.w_13_1 ctx), ((generatedRoot.w_13_2 ctx), ((generatedRoot.w_13_3 ctx), ((generatedRoot.w_14_0 ctx), ((generatedRoot.w_14_1 ctx), ((generatedRoot.w_14_2 ctx), ((generatedRoot.w_14_3 ctx), ((generatedRoot.w_15_0 ctx), ((generatedRoot.w_15_1 ctx), ((generatedRoot.w_15_2 ctx), ((generatedRoot.w_15_3 ctx), ((generatedRoot.w_69_0 ctx), ((generatedRoot.w_69_1 ctx), ((generatedRoot.w_69_2 ctx), ((generatedRoot.w_69_3 ctx), ((generatedRoot.w_107_0 ctx), ((generatedRoot.w_107_1 ctx), ((generatedRoot.w_107_2 ctx), ((generatedRoot.w_107_3 ctx), ((generatedRoot.w_137_0 ctx), ((generatedRoot.w_137_1 ctx), ((generatedRoot.w_137_2 ctx), ((generatedRoot.w_137_3 ctx), ((generatedRoot.w_139_0 ctx), ((generatedRoot.w_139_1 ctx), ((generatedRoot.w_139_2 ctx), ((generatedRoot.w_139_3 ctx), ((generatedRoot.w_141_0 ctx), ((generatedRoot.w_141_1 ctx), ((generatedRoot.w_141_2 ctx), ((generatedRoot.w_141_3 ctx), ((generatedRoot.w_143_0 ctx), ((generatedRoot.w_143_1 ctx), ((generatedRoot.w_143_2 ctx), ((generatedRoot.w_143_3 ctx), ((generatedRoot.w_145_0 ctx), ((generatedRoot.w_145_1 ctx), ((generatedRoot.w_145_2 ctx), ((generatedRoot.w_145_3 ctx), ((generatedRoot.w_147_0 ctx), ((generatedRoot.w_147_1 ctx), ((generatedRoot.w_147_2 ctx), ((generatedRoot.w_147_3 ctx), ((generatedRoot.w_149_0 ctx), ((generatedRoot.w_149_1 ctx), ((generatedRoot.w_149_2 ctx), ((generatedRoot.w_149_3 ctx), ((generatedRoot.w_151_0 ctx), ((generatedRoot.w_151_1 ctx), ((generatedRoot.w_151_2 ctx), ((generatedRoot.w_151_3 ctx), ((generatedRoot.w_189_0 ctx), ((generatedRoot.w_189_1 ctx), ((generatedRoot.w_189_2 ctx), ((generatedRoot.w_189_3 ctx), ((generatedRoot.w_203_0 ctx), ((generatedRoot.w_211_0 ctx), ((generatedRoot.w_212_0 ctx), ((generatedRoot.w_213_0 ctx), ((generatedRoot.w_260_0 ctx), ((generatedRoot.w_275_0 ctx), ((generatedRoot.w_276_0 ctx), ((generatedRoot.w_277_0 ctx), ((generatedRoot.w_311_0 ctx), ((generatedRoot.w_319_0 ctx), ((generatedRoot.w_320_0 ctx), ((generatedRoot.w_321_0 ctx), ((generatedRoot.w_335_0 ctx), ((generatedRoot.w_343_0 ctx), ((generatedRoot.w_344_0 ctx), ((generatedRoot.w_345_0 ctx), ((generatedRoot.w_359_0 ctx), ((generatedRoot.w_367_0 ctx), ((generatedRoot.w_368_0 ctx), ((generatedRoot.w_369_0 ctx), ((generatedRoot.w_79_6 ctx), ((generatedRoot.w_79_7 ctx), ((generatedRoot.w_83_6 ctx), ((generatedRoot.w_83_7 ctx), ((generatedRoot.w_36_2 ctx), ((generatedRoot.w_36_3 ctx), ((generatedRoot.w_116_6 ctx), ((generatedRoot.w_116_7 ctx), ((generatedRoot.w_120_6 ctx), ((generatedRoot.w_120_7 ctx), ((generatedRoot.w_161_6 ctx), ((generatedRoot.w_161_7 ctx), ((generatedRoot.w_165_6 ctx), ((generatedRoot.w_165_7 ctx), ()))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))

structure generatedRoot.Facts_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Prop where
  t0 : (4) = 4
  h3_3 : (∀ i : Fin 4, parallel_generatedRoot_3 backend hashModel params i ((generatedRoot.w_0_0 ctx), ((generatedRoot.w_1_0 ctx), ((generatedRoot.w_2_0 ctx), ()))) (((generatedRoot.w_3_0 ctx) i), (((generatedRoot.w_3_1 ctx) i), (((generatedRoot.w_3_2 ctx) i), (((generatedRoot.w_3_3 ctx) i), ())))))
  t2 : (8) = 8
  h4_3 : (∀ i : Fin 8, parallel_generatedRoot_4 backend hashModel params i ((generatedRoot.w_0_0 ctx), ((generatedRoot.w_1_0 ctx), ((generatedRoot.w_2_0 ctx), ()))) (((generatedRoot.w_4_0 ctx) i), (((generatedRoot.w_4_1 ctx) i), (((generatedRoot.w_4_2 ctx) i), (((generatedRoot.w_4_3 ctx) i), ())))))
  t4 : (8) = 8
  h5_3 : (∀ i : Fin 8, parallel_generatedRoot_5 backend hashModel params i ((generatedRoot.w_0_0 ctx), ((generatedRoot.w_1_0 ctx), ((generatedRoot.w_2_0 ctx), ()))) (((generatedRoot.w_5_0 ctx) i), (((generatedRoot.w_5_1 ctx) i), (((generatedRoot.w_5_2 ctx) i), (((generatedRoot.w_5_3 ctx) i), ())))))
  t6 : (8) = 8
  h6_3 : (∀ i : Fin 8, parallel_generatedRoot_6 backend hashModel params i ((generatedRoot.w_0_0 ctx), ((generatedRoot.w_1_0 ctx), ((generatedRoot.w_2_0 ctx), ()))) (((generatedRoot.w_6_0 ctx) i), (((generatedRoot.w_6_1 ctx) i), (((generatedRoot.w_6_2 ctx) i), (((generatedRoot.w_6_3 ctx) i), ())))))
  t8 : (1) = 1
  h7_3 : (∀ i : Fin 1, parallel_generatedRoot_7 backend hashModel params i ((generatedRoot.w_0_0 ctx), ((generatedRoot.w_1_0 ctx), ((generatedRoot.w_2_0 ctx), ()))) (((generatedRoot.w_7_0 ctx) i), (((generatedRoot.w_7_1 ctx) i), (((generatedRoot.w_7_2 ctx) i), (((generatedRoot.w_7_3 ctx) i), ())))))
  t10 : (8) = 8
  h8_3 : (∀ i : Fin 8, parallel_generatedRoot_8 backend hashModel params i ((generatedRoot.w_0_0 ctx), ((generatedRoot.w_1_0 ctx), ((generatedRoot.w_2_0 ctx), ()))) (((generatedRoot.w_8_0 ctx) i), (((generatedRoot.w_8_1 ctx) i), (((generatedRoot.w_8_2 ctx) i), (((generatedRoot.w_8_3 ctx) i), ())))))
  t12 : (8) = 8
  h9_3 : (∀ i : Fin 8, parallel_generatedRoot_9 backend hashModel params i ((generatedRoot.w_0_0 ctx), ((generatedRoot.w_1_0 ctx), ((generatedRoot.w_2_0 ctx), ()))) (((generatedRoot.w_9_0 ctx) i), (((generatedRoot.w_9_1 ctx) i), (((generatedRoot.w_9_2 ctx) i), (((generatedRoot.w_9_3 ctx) i), ())))))
  t14 : (1) = 1
  h10_3 : (∀ i : Fin 1, parallel_generatedRoot_10 backend hashModel params i ((generatedRoot.w_0_0 ctx), ((generatedRoot.w_1_0 ctx), ((generatedRoot.w_2_0 ctx), ()))) (((generatedRoot.w_10_0 ctx) i), (((generatedRoot.w_10_1 ctx) i), (((generatedRoot.w_10_2 ctx) i), (((generatedRoot.w_10_3 ctx) i), ())))))
  t16 : (1) = 1
  h11_3 : (∀ i : Fin 1, parallel_generatedRoot_11 backend hashModel params i ((generatedRoot.w_0_0 ctx), ((generatedRoot.w_1_0 ctx), ((generatedRoot.w_2_0 ctx), ()))) (((generatedRoot.w_11_0 ctx) i), (((generatedRoot.w_11_1 ctx) i), (((generatedRoot.w_11_2 ctx) i), (((generatedRoot.w_11_3 ctx) i), ())))))
  t18 : (1) = 1
  h12_3 : (∀ i : Fin 1, parallel_generatedRoot_12 backend hashModel params i ((generatedRoot.w_0_0 ctx), ((generatedRoot.w_1_0 ctx), ((generatedRoot.w_2_0 ctx), ()))) (((generatedRoot.w_12_0 ctx) i), (((generatedRoot.w_12_1 ctx) i), (((generatedRoot.w_12_2 ctx) i), (((generatedRoot.w_12_3 ctx) i), ())))))
  t20 : (1) = 1
  h13_3 : (∀ i : Fin 1, parallel_generatedRoot_13 backend hashModel params i ((generatedRoot.w_0_0 ctx), ((generatedRoot.w_1_0 ctx), ((generatedRoot.w_2_0 ctx), ()))) (((generatedRoot.w_13_0 ctx) i), (((generatedRoot.w_13_1 ctx) i), (((generatedRoot.w_13_2 ctx) i), (((generatedRoot.w_13_3 ctx) i), ())))))
  t22 : (1) = 1
  h14_3 : (∀ i : Fin 1, parallel_generatedRoot_14 backend hashModel params i ((generatedRoot.w_0_0 ctx), ((generatedRoot.w_1_0 ctx), ((generatedRoot.w_2_0 ctx), ()))) (((generatedRoot.w_14_0 ctx) i), (((generatedRoot.w_14_1 ctx) i), (((generatedRoot.w_14_2 ctx) i), (((generatedRoot.w_14_3 ctx) i), ())))))
  t24 : (1) = 1
  h15_3 : (∀ i : Fin 1, parallel_generatedRoot_15 backend hashModel params i ((generatedRoot.w_0_0 ctx), ((generatedRoot.w_1_0 ctx), ((generatedRoot.w_2_0 ctx), ()))) (((generatedRoot.w_15_0 ctx) i), (((generatedRoot.w_15_1 ctx) i), (((generatedRoot.w_15_2 ctx) i), (((generatedRoot.w_15_3 ctx) i), ())))))
  h18_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((generatedRoot.w_17_0 ctx)) (generatedRoot.w_18_0 ctx)
  h19_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) ((generatedRoot.w_17_0 ctx)) (generatedRoot.w_19_0 ctx)
  h20_0 : MxxRuntime.concatRows (generatedRoot.w_18_0 ctx) (generatedRoot.w_19_0 ctx) (generatedRoot.w_20_0 ctx)
  h21_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 115, 116, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((generatedRoot.w_17_0 ctx)) (generatedRoot.w_21_0 ctx)
  h22_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_21_0 ctx) (generatedRoot.w_22_0 ctx)
  h24_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((generatedRoot.w_17_0 ctx)) (generatedRoot.w_24_0 ctx)

theorem generatedRoot.facts_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {ctx : generatedRoot.Ctx backend hashModel params} {outputs : (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat4 × mxxMat4 × mxxMat4 × mxxMat4 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × Unit}
    (h : generatedRoot.constraints_0 backend hashModel params (generatedRoot.w_0_0 ctx) (generatedRoot.w_1_0 ctx) (generatedRoot.w_17_0 ctx) (generatedRoot.w_2_0 ctx) (generatedRoot.w_3_0 ctx) (generatedRoot.w_3_1 ctx) (generatedRoot.w_3_2 ctx) (generatedRoot.w_3_3 ctx) (generatedRoot.w_4_0 ctx) (generatedRoot.w_4_1 ctx) (generatedRoot.w_4_2 ctx) (generatedRoot.w_4_3 ctx) (generatedRoot.w_5_0 ctx) (generatedRoot.w_5_1 ctx) (generatedRoot.w_5_2 ctx) (generatedRoot.w_5_3 ctx) (generatedRoot.w_6_0 ctx) (generatedRoot.w_6_1 ctx) (generatedRoot.w_6_2 ctx) (generatedRoot.w_6_3 ctx) (generatedRoot.w_7_0 ctx) (generatedRoot.w_7_1 ctx) (generatedRoot.w_7_2 ctx) (generatedRoot.w_7_3 ctx) (generatedRoot.w_8_0 ctx) (generatedRoot.w_8_1 ctx) (generatedRoot.w_8_2 ctx) (generatedRoot.w_8_3 ctx) (generatedRoot.w_9_0 ctx) (generatedRoot.w_9_1 ctx) (generatedRoot.w_9_2 ctx) (generatedRoot.w_9_3 ctx) (generatedRoot.w_10_0 ctx) (generatedRoot.w_10_1 ctx) (generatedRoot.w_10_2 ctx) (generatedRoot.w_10_3 ctx) (generatedRoot.w_11_0 ctx) (generatedRoot.w_11_1 ctx) (generatedRoot.w_11_2 ctx) (generatedRoot.w_11_3 ctx) (generatedRoot.w_12_0 ctx) (generatedRoot.w_12_1 ctx) (generatedRoot.w_12_2 ctx) (generatedRoot.w_12_3 ctx) (generatedRoot.w_13_0 ctx) (generatedRoot.w_13_1 ctx) (generatedRoot.w_13_2 ctx) (generatedRoot.w_13_3 ctx) (generatedRoot.w_14_0 ctx) (generatedRoot.w_14_1 ctx) (generatedRoot.w_14_2 ctx) (generatedRoot.w_14_3 ctx) (generatedRoot.w_15_0 ctx) (generatedRoot.w_15_1 ctx) (generatedRoot.w_15_2 ctx) (generatedRoot.w_15_3 ctx) (generatedRoot.w_18_0 ctx) (generatedRoot.w_19_0 ctx) (generatedRoot.w_20_0 ctx) (generatedRoot.w_21_0 ctx) (generatedRoot.w_22_0 ctx) (generatedRoot.w_24_0 ctx)) :
    generatedRoot.Facts_0 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Prop where
  h25_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) ((generatedRoot.w_17_0 ctx)) (generatedRoot.w_25_0 ctx)
  h26_0 : MxxRuntime.concatRows (generatedRoot.w_24_0 ctx) (generatedRoot.w_25_0 ctx) (generatedRoot.w_26_0 ctx)
  h27_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 115, 116, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) ((generatedRoot.w_17_0 ctx)) (generatedRoot.w_27_0 ctx)
  h28_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_27_0 ctx) (generatedRoot.w_28_0 ctx)
  t36 : 0 ≤ 0
  t37 : 0 < 1
  t38 : 1 ≤ 2
  t39 : 0 ≤ 0
  t40 : 0 < 25
  t41 : 25 ≤ 25
  h31_0 : MxxRuntime.sliceMatrix (generatedRoot.w_30_0 ctx) 0 1 0 25 (generatedRoot.w_31_0 ctx)
  t43 : 0 ≤ 1
  t44 : 1 < 2
  t45 : 2 ≤ 2
  t46 : 0 ≤ 0
  t47 : 0 < 25
  t48 : 25 ≤ 25
  h32_0 : MxxRuntime.sliceMatrix (generatedRoot.w_30_0 ctx) 1 2 0 25 (generatedRoot.w_32_0 ctx)
  h33_0 : MxxRuntime.concatRows (generatedRoot.w_31_0 ctx) (generatedRoot.w_32_0 ctx) (generatedRoot.w_33_0 ctx)
  t51 : (1) = 1
  h36_3 : (∀ i : Fin 1, parallel_generatedRoot_36 backend hashModel params i ((generatedRoot.w_0_0 ctx), ((generatedRoot.w_1_0 ctx), (((generatedRoot.w_34_0 ctx) i), (((generatedRoot.w_35_0 ctx) i), ((generatedRoot.w_2_0 ctx), ()))))) (((generatedRoot.w_36_0 ctx) i), (((generatedRoot.w_36_1 ctx) i), (((generatedRoot.w_36_2 ctx) i), (((generatedRoot.w_36_3 ctx) i), ())))))
  h37_0 : MxxRuntime.familyGetStatic (generatedRoot.w_36_0 ctx) (0) (generatedRoot.w_37_0 ctx)
  h38_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_37_0 ctx) (generatedRoot.w_38_0 ctx)
  h39_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_38_0 ctx) (generatedRoot.w_39_0 ctx)
  h41_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 115, 116, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((generatedRoot.w_17_0 ctx)) (generatedRoot.w_41_0 ctx)
  h42_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_41_0 ctx) (generatedRoot.w_42_0 ctx)
  h44_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 115, 116, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) ((generatedRoot.w_17_0 ctx)) (generatedRoot.w_44_0 ctx)
  h45_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_44_0 ctx) (generatedRoot.w_45_0 ctx)
  t60 : 0 ≤ 0
  t61 : 0 < 1
  t62 : 1 ≤ 2
  t63 : 0 ≤ 0

theorem generatedRoot.facts_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {ctx : generatedRoot.Ctx backend hashModel params} {outputs : (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat4 × mxxMat4 × mxxMat4 × mxxMat4 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × Unit}
    (h : generatedRoot.constraints_1 backend hashModel params (generatedRoot.w_0_0 ctx) (generatedRoot.w_1_0 ctx) (generatedRoot.w_17_0 ctx) (generatedRoot.w_34_0 ctx) (generatedRoot.w_35_0 ctx) (generatedRoot.w_2_0 ctx) (generatedRoot.w_24_0 ctx) (generatedRoot.w_25_0 ctx) (generatedRoot.w_26_0 ctx) (generatedRoot.w_27_0 ctx) (generatedRoot.w_28_0 ctx) (generatedRoot.w_30_0 ctx) (generatedRoot.w_31_0 ctx) (generatedRoot.w_32_0 ctx) (generatedRoot.w_33_0 ctx) (generatedRoot.w_36_0 ctx) (generatedRoot.w_36_1 ctx) (generatedRoot.w_36_2 ctx) (generatedRoot.w_36_3 ctx) (generatedRoot.w_37_0 ctx) (generatedRoot.w_38_0 ctx) (generatedRoot.w_39_0 ctx) (generatedRoot.w_41_0 ctx) (generatedRoot.w_42_0 ctx) (generatedRoot.w_44_0 ctx) (generatedRoot.w_45_0 ctx)) :
    generatedRoot.Facts_1 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_2 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Prop where
  t64 : 0 < 25
  t65 : 25 ≤ 25
  h48_0 : MxxRuntime.sliceMatrix (generatedRoot.w_47_0 ctx) 0 1 0 25 (generatedRoot.w_48_0 ctx)
  t67 : 0 ≤ 1
  t68 : 1 < 2
  t69 : 2 ≤ 2
  t70 : 0 ≤ 0
  t71 : 0 < 25
  t72 : 25 ≤ 25
  h49_0 : MxxRuntime.sliceMatrix (generatedRoot.w_47_0 ctx) 1 2 0 25 (generatedRoot.w_49_0 ctx)
  h50_0 : MxxRuntime.concatRows (generatedRoot.w_48_0 ctx) (generatedRoot.w_49_0 ctx) (generatedRoot.w_50_0 ctx)
  h51_0 : MxxRuntime.familyGetStatic (generatedRoot.w_36_2 ctx) (0) (generatedRoot.w_51_0 ctx)
  h52_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_51_0 ctx) (generatedRoot.w_52_0 ctx)
  h53_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_52_0 ctx) (generatedRoot.w_53_0 ctx)
  t78 : 0 ≤ 0
  t79 : 0 < 1
  t80 : 1 ≤ 2
  t81 : 0 ≤ 0
  t82 : 0 < 25
  t83 : 25 ≤ 25
  h56_0 : MxxRuntime.sliceMatrix (generatedRoot.w_55_0 ctx) 0 1 0 25 (generatedRoot.w_56_0 ctx)
  t85 : 0 ≤ 1
  t86 : 1 < 2
  t87 : 2 ≤ 2
  t88 : 0 ≤ 0
  t89 : 0 < 25
  t90 : 25 ≤ 25
  h57_0 : MxxRuntime.sliceMatrix (generatedRoot.w_55_0 ctx) 1 2 0 25 (generatedRoot.w_57_0 ctx)
  h58_0 : MxxRuntime.familyGetStatic (generatedRoot.w_36_1 ctx) (0) (generatedRoot.w_58_0 ctx)
  h59_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_58_0 ctx) (generatedRoot.w_59_0 ctx)
  h60_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_59_0 ctx) (generatedRoot.w_60_0 ctx)
  h62_0 : MxxRuntime.familyGetStatic (generatedRoot.w_36_3 ctx) (0) (generatedRoot.w_62_0 ctx)

theorem generatedRoot.facts_2 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {ctx : generatedRoot.Ctx backend hashModel params} {outputs : (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat4 × mxxMat4 × mxxMat4 × mxxMat4 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × Unit}
    (h : generatedRoot.constraints_2 backend (generatedRoot.w_36_1 ctx) (generatedRoot.w_36_2 ctx) (generatedRoot.w_36_3 ctx) (generatedRoot.w_47_0 ctx) (generatedRoot.w_48_0 ctx) (generatedRoot.w_49_0 ctx) (generatedRoot.w_50_0 ctx) (generatedRoot.w_51_0 ctx) (generatedRoot.w_52_0 ctx) (generatedRoot.w_53_0 ctx) (generatedRoot.w_55_0 ctx) (generatedRoot.w_56_0 ctx) (generatedRoot.w_57_0 ctx) (generatedRoot.w_58_0 ctx) (generatedRoot.w_59_0 ctx) (generatedRoot.w_60_0 ctx) (generatedRoot.w_62_0 ctx)) :
    generatedRoot.Facts_2 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_3 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Prop where
  h63_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_62_0 ctx) (generatedRoot.w_63_0 ctx)
  h64_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_63_0 ctx) (generatedRoot.w_64_0 ctx)
  t98 : 0 ≤ 0
  t99 : 0 < 1
  t100 : 1 ≤ 2
  t101 : 0 ≤ 0
  t102 : 0 < 25
  t103 : 25 ≤ 25
  h67_0 : MxxRuntime.sliceMatrix (generatedRoot.w_66_0 ctx) 0 1 0 25 (generatedRoot.w_67_0 ctx)
  t105 : 0 ≤ 1
  t106 : 1 < 2
  t107 : 2 ≤ 2
  t108 : 0 ≤ 0
  t109 : 0 < 25
  t110 : 25 ≤ 25
  h68_0 : MxxRuntime.sliceMatrix (generatedRoot.w_66_0 ctx) 1 2 0 25 (generatedRoot.w_68_0 ctx)
  t112 : 0 ≤ (4)
  h69_3 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_generatedRoot_69 backend hashModel params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, ((generatedRoot.w_17_0 ctx), ((generatedRoot.w_56_0 ctx), ((generatedRoot.w_57_0 ctx), ((generatedRoot.w_3_0 ctx), ((generatedRoot.w_67_0 ctx), ((generatedRoot.w_68_0 ctx), ((generatedRoot.w_3_2 ctx), ((generatedRoot.w_3_1 ctx), ((generatedRoot.w_3_3 ctx), ()))))))))))))) next) (Int.toNat (4)) ((generatedRoot.w_16_0 ctx), ((generatedRoot.w_16_0 ctx), ((generatedRoot.w_16_0 ctx), ((generatedRoot.w_16_0 ctx), ())))) ((generatedRoot.w_69_0 ctx), ((generatedRoot.w_69_1 ctx), ((generatedRoot.w_69_2 ctx), ((generatedRoot.w_69_3 ctx), ()))))
  t114 : 0 ≤ (0)
  t115 : (0) < 4500
  h73_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((generatedRoot.w_17_0 ctx)) (generatedRoot.w_73_0 ctx)
  h74_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) ((generatedRoot.w_17_0 ctx)) (generatedRoot.w_74_0 ctx)
  h75_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((generatedRoot.w_17_0 ctx)) (generatedRoot.w_75_0 ctx)
  h76_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) ((generatedRoot.w_17_0 ctx)) (generatedRoot.w_76_0 ctx)
  t120 : (72000) = 72000
  h79_7 : (∀ i : Fin 72000, parallel_generatedRoot_79 backend hashModel params i ((generatedRoot.w_17_0 ctx), (((generatedRoot.w_72_0 ctx) i), ((generatedRoot.w_73_0 ctx), ((generatedRoot.w_74_0 ctx), ((generatedRoot.w_0_0 ctx), ((generatedRoot.w_75_0 ctx), ((generatedRoot.w_76_0 ctx), ((generatedRoot.w_1_0 ctx), (((generatedRoot.w_77_0 ctx) i), (((generatedRoot.w_78_0 ctx) i), ((generatedRoot.w_2_0 ctx), ()))))))))))) (((generatedRoot.w_79_0 ctx) i), (((generatedRoot.w_79_1 ctx) i), (((generatedRoot.w_79_2 ctx) i), (((generatedRoot.w_79_3 ctx) i), (((generatedRoot.w_79_4 ctx) i), (((generatedRoot.w_79_5 ctx) i), (((generatedRoot.w_79_6 ctx) i), (((generatedRoot.w_79_7 ctx) i), ())))))))))
  t122 : (36000) = 36000
  h80_3 : (∀ i : Fin 36000, parallel_generatedRoot_80 backend hashModel params i ((generatedRoot.w_79_0 ctx), ((generatedRoot.w_79_1 ctx), ((generatedRoot.w_79_2 ctx), ((generatedRoot.w_79_3 ctx), ())))) (((generatedRoot.w_80_0 ctx) i), (((generatedRoot.w_80_1 ctx) i), (((generatedRoot.w_80_2 ctx) i), (((generatedRoot.w_80_3 ctx) i), ())))))
  t124 : (72000) = 72000
  h83_7 : (∀ i : Fin 72000, parallel_generatedRoot_83 backend hashModel params i ((generatedRoot.w_17_0 ctx), (((generatedRoot.w_71_0 ctx) i), ((generatedRoot.w_80_0 ctx), ((generatedRoot.w_80_2 ctx), ((generatedRoot.w_0_0 ctx), ((generatedRoot.w_80_1 ctx), ((generatedRoot.w_80_3 ctx), ((generatedRoot.w_1_0 ctx), (((generatedRoot.w_81_0 ctx) i), (((generatedRoot.w_82_0 ctx) i), ((generatedRoot.w_2_0 ctx), ()))))))))))) (((generatedRoot.w_83_0 ctx) i), (((generatedRoot.w_83_1 ctx) i), (((generatedRoot.w_83_2 ctx) i), (((generatedRoot.w_83_3 ctx) i), (((generatedRoot.w_83_4 ctx) i), (((generatedRoot.w_83_5 ctx) i), (((generatedRoot.w_83_6 ctx) i), (((generatedRoot.w_83_7 ctx) i), ())))))))))
  t126 : (36000) = 36000
  h84_3 : (∀ i : Fin 36000, parallel_generatedRoot_84 backend hashModel params i ((generatedRoot.w_83_0 ctx), ((generatedRoot.w_83_1 ctx), ((generatedRoot.w_83_2 ctx), ((generatedRoot.w_83_3 ctx), ())))) (((generatedRoot.w_84_0 ctx) i), (((generatedRoot.w_84_1 ctx) i), (((generatedRoot.w_84_2 ctx) i), (((generatedRoot.w_84_3 ctx) i), ())))))

theorem generatedRoot.facts_3 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {ctx : generatedRoot.Ctx backend hashModel params} {outputs : (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat4 × mxxMat4 × mxxMat4 × mxxMat4 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × Unit}
    (h : generatedRoot.constraints_3 backend hashModel params (generatedRoot.w_0_0 ctx) (generatedRoot.w_1_0 ctx) (generatedRoot.w_17_0 ctx) (generatedRoot.w_71_0 ctx) (generatedRoot.w_72_0 ctx) (generatedRoot.w_77_0 ctx) (generatedRoot.w_78_0 ctx) (generatedRoot.w_81_0 ctx) (generatedRoot.w_82_0 ctx) (generatedRoot.w_2_0 ctx) (generatedRoot.w_3_0 ctx) (generatedRoot.w_3_1 ctx) (generatedRoot.w_3_2 ctx) (generatedRoot.w_3_3 ctx) (generatedRoot.w_16_0 ctx) (generatedRoot.w_56_0 ctx) (generatedRoot.w_57_0 ctx) (generatedRoot.w_62_0 ctx) (generatedRoot.w_63_0 ctx) (generatedRoot.w_64_0 ctx) (generatedRoot.w_66_0 ctx) (generatedRoot.w_67_0 ctx) (generatedRoot.w_68_0 ctx) (generatedRoot.w_69_0 ctx) (generatedRoot.w_69_1 ctx) (generatedRoot.w_69_2 ctx) (generatedRoot.w_69_3 ctx) (generatedRoot.w_73_0 ctx) (generatedRoot.w_74_0 ctx) (generatedRoot.w_75_0 ctx) (generatedRoot.w_76_0 ctx) (generatedRoot.w_79_0 ctx) (generatedRoot.w_79_1 ctx) (generatedRoot.w_79_2 ctx) (generatedRoot.w_79_3 ctx) (generatedRoot.w_79_4 ctx) (generatedRoot.w_79_5 ctx) (generatedRoot.w_79_6 ctx) (generatedRoot.w_79_7 ctx) (generatedRoot.w_80_0 ctx) (generatedRoot.w_80_1 ctx) (generatedRoot.w_80_2 ctx) (generatedRoot.w_80_3 ctx) (generatedRoot.w_83_0 ctx) (generatedRoot.w_83_1 ctx) (generatedRoot.w_83_2 ctx) (generatedRoot.w_83_3 ctx) (generatedRoot.w_83_4 ctx) (generatedRoot.w_83_5 ctx) (generatedRoot.w_83_6 ctx) (generatedRoot.w_83_7 ctx) (generatedRoot.w_84_0 ctx) (generatedRoot.w_84_1 ctx) (generatedRoot.w_84_2 ctx) (generatedRoot.w_84_3 ctx)) :
    generatedRoot.Facts_3 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_4 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Prop where
  t128 : (36000) = 36000
  h85_3 : (∀ i : Fin 36000, parallel_generatedRoot_85 backend hashModel params i ((generatedRoot.w_17_0 ctx), (((generatedRoot.w_84_0 ctx) i), (((generatedRoot.w_84_2 ctx) i), ((generatedRoot.w_4_0 ctx), (((generatedRoot.w_84_1 ctx) i), (((generatedRoot.w_84_3 ctx) i), ((generatedRoot.w_4_2 ctx), ((generatedRoot.w_4_1 ctx), ((generatedRoot.w_4_3 ctx), ()))))))))) (((generatedRoot.w_85_0 ctx) i), (((generatedRoot.w_85_1 ctx) i), (((generatedRoot.w_85_2 ctx) i), (((generatedRoot.w_85_3 ctx) i), ())))))
  t130 : (36000) = 36000
  h86_3 : (∀ i : Fin 36000, parallel_generatedRoot_86 backend hashModel params i (((generatedRoot.w_85_0 ctx) i), (((generatedRoot.w_85_1 ctx) i), (((generatedRoot.w_85_2 ctx) i), (((generatedRoot.w_85_3 ctx) i), ())))) (((generatedRoot.w_86_0 ctx) i), (((generatedRoot.w_86_1 ctx) i), (((generatedRoot.w_86_2 ctx) i), (((generatedRoot.w_86_3 ctx) i), ())))))
  h87_0 : MxxRuntime.familyGetStatic (generatedRoot.w_86_0 ctx) (0) (generatedRoot.w_87_0 ctx)
  h88_0 : MxxRuntime.familyGetStatic (generatedRoot.w_86_0 ctx) (1) (generatedRoot.w_88_0 ctx)
  h90_0 : MxxRuntime.familyGetStatic (generatedRoot.w_86_1 ctx) (0) (generatedRoot.w_90_0 ctx)
  h91_0 : MxxRuntime.familyGetStatic (generatedRoot.w_86_1 ctx) (1) (generatedRoot.w_91_0 ctx)
  h93_0 : MxxRuntime.familyGetStatic (generatedRoot.w_86_2 ctx) (0) (generatedRoot.w_93_0 ctx)
  h94_0 : MxxRuntime.familyGetStatic (generatedRoot.w_86_2 ctx) (1) (generatedRoot.w_94_0 ctx)
  h96_0 : MxxRuntime.familyGetStatic (generatedRoot.w_86_3 ctx) (0) (generatedRoot.w_96_0 ctx)
  h97_0 : MxxRuntime.familyGetStatic (generatedRoot.w_86_3 ctx) (1) (generatedRoot.w_97_0 ctx)
  t140 : 0 ≤ (6)
  h99_3 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_generatedRoot_99 backend hashModel params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, ((generatedRoot.w_86_0 ctx), ((generatedRoot.w_86_1 ctx), ((generatedRoot.w_86_2 ctx), ((generatedRoot.w_86_3 ctx), ())))))))) next) (Int.toNat (6)) ((generatedRoot.w_89_0 ctx), ((generatedRoot.w_92_0 ctx), ((generatedRoot.w_95_0 ctx), ((generatedRoot.w_98_0 ctx), ())))) ((generatedRoot.w_99_0 ctx), ((generatedRoot.w_99_1 ctx), ((generatedRoot.w_99_2 ctx), ((generatedRoot.w_99_3 ctx), ()))))
  h100_0 : MxxRuntime.tensorRuns (generatedRoot.w_70_0 ctx) (generatedRoot.w_99_0 ctx) (generatedRoot.w_100_0 ctx)
  t143 : 0 ≤ (0)
  t144 : (0) < 4500
  h102_0 : MxxRuntime.tensorRuns (generatedRoot.w_101_0 ctx) (generatedRoot.w_99_1 ctx) (generatedRoot.w_102_0 ctx)
  t146 : 0 ≤ (0)
  t147 : (0) < 4500
  h104_0 : MxxRuntime.tensorRuns (generatedRoot.w_103_0 ctx) (generatedRoot.w_99_2 ctx) (generatedRoot.w_104_0 ctx)
  t149 : 0 ≤ (0)
  t150 : (0) < 4500
  h106_0 : MxxRuntime.tensorRuns (generatedRoot.w_105_0 ctx) (generatedRoot.w_99_3 ctx) (generatedRoot.w_106_0 ctx)
  t152 : 0 ≤ (4499)
  h107_3 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × Unit) => sequential_generatedRoot_107 backend hashModel params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, ((generatedRoot.w_86_0 ctx), ((generatedRoot.w_86_1 ctx), ((generatedRoot.w_86_2 ctx), ((generatedRoot.w_86_3 ctx), ())))))))) next) (Int.toNat (4499)) ((generatedRoot.w_100_0 ctx), ((generatedRoot.w_102_0 ctx), ((generatedRoot.w_104_0 ctx), ((generatedRoot.w_106_0 ctx), ())))) ((generatedRoot.w_107_0 ctx), ((generatedRoot.w_107_1 ctx), ((generatedRoot.w_107_2 ctx), ((generatedRoot.w_107_3 ctx), ()))))
  h110_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((generatedRoot.w_17_0 ctx)) (generatedRoot.w_110_0 ctx)
  h111_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) ((generatedRoot.w_17_0 ctx)) (generatedRoot.w_111_0 ctx)
  h112_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((generatedRoot.w_17_0 ctx)) (generatedRoot.w_112_0 ctx)
  h113_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) ((generatedRoot.w_17_0 ctx)) (generatedRoot.w_113_0 ctx)
  t158 : (16) = 16
  h116_7 : (∀ i : Fin 16, parallel_generatedRoot_116 backend hashModel params i ((generatedRoot.w_17_0 ctx), (((generatedRoot.w_109_0 ctx) i), ((generatedRoot.w_110_0 ctx), ((generatedRoot.w_111_0 ctx), ((generatedRoot.w_0_0 ctx), ((generatedRoot.w_112_0 ctx), ((generatedRoot.w_113_0 ctx), ((generatedRoot.w_1_0 ctx), (((generatedRoot.w_114_0 ctx) i), (((generatedRoot.w_115_0 ctx) i), ((generatedRoot.w_2_0 ctx), ()))))))))))) (((generatedRoot.w_116_0 ctx) i), (((generatedRoot.w_116_1 ctx) i), (((generatedRoot.w_116_2 ctx) i), (((generatedRoot.w_116_3 ctx) i), (((generatedRoot.w_116_4 ctx) i), (((generatedRoot.w_116_5 ctx) i), (((generatedRoot.w_116_6 ctx) i), (((generatedRoot.w_116_7 ctx) i), ())))))))))

theorem generatedRoot.facts_4 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {ctx : generatedRoot.Ctx backend hashModel params} {outputs : (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat4 × mxxMat4 × mxxMat4 × mxxMat4 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × Unit}
    (h : generatedRoot.constraints_4 backend hashModel params (generatedRoot.w_0_0 ctx) (generatedRoot.w_1_0 ctx) (generatedRoot.w_17_0 ctx) (generatedRoot.w_109_0 ctx) (generatedRoot.w_114_0 ctx) (generatedRoot.w_115_0 ctx) (generatedRoot.w_2_0 ctx) (generatedRoot.w_4_0 ctx) (generatedRoot.w_4_1 ctx) (generatedRoot.w_4_2 ctx) (generatedRoot.w_4_3 ctx) (generatedRoot.w_70_0 ctx) (generatedRoot.w_84_0 ctx) (generatedRoot.w_84_1 ctx) (generatedRoot.w_84_2 ctx) (generatedRoot.w_84_3 ctx) (generatedRoot.w_85_0 ctx) (generatedRoot.w_85_1 ctx) (generatedRoot.w_85_2 ctx) (generatedRoot.w_85_3 ctx) (generatedRoot.w_86_0 ctx) (generatedRoot.w_86_1 ctx) (generatedRoot.w_86_2 ctx) (generatedRoot.w_86_3 ctx) (generatedRoot.w_87_0 ctx) (generatedRoot.w_88_0 ctx) (generatedRoot.w_89_0 ctx) (generatedRoot.w_90_0 ctx) (generatedRoot.w_91_0 ctx) (generatedRoot.w_92_0 ctx) (generatedRoot.w_93_0 ctx) (generatedRoot.w_94_0 ctx) (generatedRoot.w_95_0 ctx) (generatedRoot.w_96_0 ctx) (generatedRoot.w_97_0 ctx) (generatedRoot.w_98_0 ctx) (generatedRoot.w_99_0 ctx) (generatedRoot.w_99_1 ctx) (generatedRoot.w_99_2 ctx) (generatedRoot.w_99_3 ctx) (generatedRoot.w_100_0 ctx) (generatedRoot.w_101_0 ctx) (generatedRoot.w_102_0 ctx) (generatedRoot.w_103_0 ctx) (generatedRoot.w_104_0 ctx) (generatedRoot.w_105_0 ctx) (generatedRoot.w_106_0 ctx) (generatedRoot.w_107_0 ctx) (generatedRoot.w_107_1 ctx) (generatedRoot.w_107_2 ctx) (generatedRoot.w_107_3 ctx) (generatedRoot.w_110_0 ctx) (generatedRoot.w_111_0 ctx) (generatedRoot.w_112_0 ctx) (generatedRoot.w_113_0 ctx) (generatedRoot.w_116_0 ctx) (generatedRoot.w_116_1 ctx) (generatedRoot.w_116_2 ctx) (generatedRoot.w_116_3 ctx) (generatedRoot.w_116_4 ctx) (generatedRoot.w_116_5 ctx) (generatedRoot.w_116_6 ctx) (generatedRoot.w_116_7 ctx)) :
    generatedRoot.Facts_4 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_5 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Prop where
  t160 : (8) = 8
  h117_3 : (∀ i : Fin 8, parallel_generatedRoot_117 backend hashModel params i ((generatedRoot.w_116_0 ctx), ((generatedRoot.w_116_1 ctx), ((generatedRoot.w_116_2 ctx), ((generatedRoot.w_116_3 ctx), ())))) (((generatedRoot.w_117_0 ctx) i), (((generatedRoot.w_117_1 ctx) i), (((generatedRoot.w_117_2 ctx) i), (((generatedRoot.w_117_3 ctx) i), ())))))
  t162 : (16) = 16
  h120_7 : (∀ i : Fin 16, parallel_generatedRoot_120 backend hashModel params i ((generatedRoot.w_17_0 ctx), (((generatedRoot.w_108_0 ctx) i), ((generatedRoot.w_117_0 ctx), ((generatedRoot.w_117_2 ctx), ((generatedRoot.w_0_0 ctx), ((generatedRoot.w_117_1 ctx), ((generatedRoot.w_117_3 ctx), ((generatedRoot.w_1_0 ctx), (((generatedRoot.w_118_0 ctx) i), (((generatedRoot.w_119_0 ctx) i), ((generatedRoot.w_2_0 ctx), ()))))))))))) (((generatedRoot.w_120_0 ctx) i), (((generatedRoot.w_120_1 ctx) i), (((generatedRoot.w_120_2 ctx) i), (((generatedRoot.w_120_3 ctx) i), (((generatedRoot.w_120_4 ctx) i), (((generatedRoot.w_120_5 ctx) i), (((generatedRoot.w_120_6 ctx) i), (((generatedRoot.w_120_7 ctx) i), ())))))))))
  t164 : (8) = 8
  h121_3 : (∀ i : Fin 8, parallel_generatedRoot_121 backend hashModel params i ((generatedRoot.w_120_0 ctx), ((generatedRoot.w_120_1 ctx), ((generatedRoot.w_120_2 ctx), ((generatedRoot.w_120_3 ctx), ())))) (((generatedRoot.w_121_0 ctx) i), (((generatedRoot.w_121_1 ctx) i), (((generatedRoot.w_121_2 ctx) i), (((generatedRoot.w_121_3 ctx) i), ())))))
  t166 : (8) = 8
  h122_3 : (∀ i : Fin 8, parallel_generatedRoot_122 backend hashModel params i ((generatedRoot.w_17_0 ctx), (((generatedRoot.w_121_0 ctx) i), (((generatedRoot.w_121_2 ctx) i), ((generatedRoot.w_6_0 ctx), (((generatedRoot.w_121_1 ctx) i), (((generatedRoot.w_121_3 ctx) i), ((generatedRoot.w_6_2 ctx), ((generatedRoot.w_6_1 ctx), ((generatedRoot.w_6_3 ctx), ()))))))))) (((generatedRoot.w_122_0 ctx) i), (((generatedRoot.w_122_1 ctx) i), (((generatedRoot.w_122_2 ctx) i), (((generatedRoot.w_122_3 ctx) i), ())))))
  t168 : (8) = 8
  h123_3 : (∀ i : Fin 8, parallel_generatedRoot_123 backend hashModel params i ((generatedRoot.w_17_0 ctx), (((generatedRoot.w_122_0 ctx) i), (((generatedRoot.w_122_2 ctx) i), ((generatedRoot.w_9_0 ctx), (((generatedRoot.w_122_1 ctx) i), (((generatedRoot.w_122_3 ctx) i), ((generatedRoot.w_9_2 ctx), ((generatedRoot.w_9_1 ctx), ((generatedRoot.w_9_3 ctx), ()))))))))) (((generatedRoot.w_123_0 ctx) i), (((generatedRoot.w_123_1 ctx) i), (((generatedRoot.w_123_2 ctx) i), (((generatedRoot.w_123_3 ctx) i), ())))))
  t170 : (8) = 8
  h124_3 : (∀ i : Fin 8, parallel_generatedRoot_124 backend hashModel params i (((generatedRoot.w_123_0 ctx) i), (((generatedRoot.w_123_1 ctx) i), (((generatedRoot.w_123_2 ctx) i), (((generatedRoot.w_123_3 ctx) i), ())))) (((generatedRoot.w_124_0 ctx) i), (((generatedRoot.w_124_1 ctx) i), (((generatedRoot.w_124_2 ctx) i), (((generatedRoot.w_124_3 ctx) i), ())))))
  h125_0 : MxxRuntime.familyGetStatic (generatedRoot.w_124_0 ctx) (0) (generatedRoot.w_125_0 ctx)
  h126_0 : MxxRuntime.familyGetStatic (generatedRoot.w_124_0 ctx) (1) (generatedRoot.w_126_0 ctx)
  h128_0 : MxxRuntime.familyGetStatic (generatedRoot.w_124_1 ctx) (0) (generatedRoot.w_128_0 ctx)
  h129_0 : MxxRuntime.familyGetStatic (generatedRoot.w_124_1 ctx) (1) (generatedRoot.w_129_0 ctx)
  h131_0 : MxxRuntime.familyGetStatic (generatedRoot.w_124_2 ctx) (0) (generatedRoot.w_131_0 ctx)
  h132_0 : MxxRuntime.familyGetStatic (generatedRoot.w_124_2 ctx) (1) (generatedRoot.w_132_0 ctx)
  h134_0 : MxxRuntime.familyGetStatic (generatedRoot.w_124_3 ctx) (0) (generatedRoot.w_134_0 ctx)
  h135_0 : MxxRuntime.familyGetStatic (generatedRoot.w_124_3 ctx) (1) (generatedRoot.w_135_0 ctx)
  t180 : 0 ≤ (6)
  h137_3 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_generatedRoot_137 backend hashModel params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, ((generatedRoot.w_124_0 ctx), ((generatedRoot.w_124_1 ctx), ((generatedRoot.w_124_2 ctx), ((generatedRoot.w_124_3 ctx), ())))))))) next) (Int.toNat (6)) ((generatedRoot.w_127_0 ctx), ((generatedRoot.w_130_0 ctx), ((generatedRoot.w_133_0 ctx), ((generatedRoot.w_136_0 ctx), ())))) ((generatedRoot.w_137_0 ctx), ((generatedRoot.w_137_1 ctx), ((generatedRoot.w_137_2 ctx), ((generatedRoot.w_137_3 ctx), ()))))
  t182 : 0 ≤ (1)
  h139_3 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_generatedRoot_139 backend hashModel params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, ((generatedRoot.w_17_0 ctx), ((generatedRoot.w_137_0 ctx), ((generatedRoot.w_137_2 ctx), ((generatedRoot.w_10_0 ctx), ((generatedRoot.w_137_1 ctx), ((generatedRoot.w_137_3 ctx), ((generatedRoot.w_10_2 ctx), ((generatedRoot.w_10_1 ctx), ((generatedRoot.w_10_3 ctx), ()))))))))))))) next) (Int.toNat (1)) ((generatedRoot.w_138_0 ctx), ((generatedRoot.w_138_0 ctx), ((generatedRoot.w_138_0 ctx), ((generatedRoot.w_138_0 ctx), ())))) ((generatedRoot.w_139_0 ctx), ((generatedRoot.w_139_1 ctx), ((generatedRoot.w_139_2 ctx), ((generatedRoot.w_139_3 ctx), ()))))
  t184 : 0 ≤ (1)
  h141_3 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_generatedRoot_141 backend hashModel params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, ((generatedRoot.w_17_0 ctx), ((generatedRoot.w_137_0 ctx), ((generatedRoot.w_137_2 ctx), ((generatedRoot.w_11_0 ctx), ((generatedRoot.w_137_1 ctx), ((generatedRoot.w_137_3 ctx), ((generatedRoot.w_11_2 ctx), ((generatedRoot.w_11_1 ctx), ((generatedRoot.w_11_3 ctx), ()))))))))))))) next) (Int.toNat (1)) ((generatedRoot.w_140_0 ctx), ((generatedRoot.w_140_0 ctx), ((generatedRoot.w_140_0 ctx), ((generatedRoot.w_140_0 ctx), ())))) ((generatedRoot.w_141_0 ctx), ((generatedRoot.w_141_1 ctx), ((generatedRoot.w_141_2 ctx), ((generatedRoot.w_141_3 ctx), ()))))
  t186 : 0 ≤ (1)
  h143_3 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_generatedRoot_143 backend hashModel params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, ((generatedRoot.w_17_0 ctx), ((generatedRoot.w_137_0 ctx), ((generatedRoot.w_137_2 ctx), ((generatedRoot.w_12_0 ctx), ((generatedRoot.w_137_1 ctx), ((generatedRoot.w_137_3 ctx), ((generatedRoot.w_12_2 ctx), ((generatedRoot.w_12_1 ctx), ((generatedRoot.w_12_3 ctx), ()))))))))))))) next) (Int.toNat (1)) ((generatedRoot.w_142_0 ctx), ((generatedRoot.w_142_0 ctx), ((generatedRoot.w_142_0 ctx), ((generatedRoot.w_142_0 ctx), ())))) ((generatedRoot.w_143_0 ctx), ((generatedRoot.w_143_1 ctx), ((generatedRoot.w_143_2 ctx), ((generatedRoot.w_143_3 ctx), ()))))
  t188 : 0 ≤ (1)
  h145_3 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_generatedRoot_145 backend hashModel params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, ((generatedRoot.w_17_0 ctx), ((generatedRoot.w_137_0 ctx), ((generatedRoot.w_137_2 ctx), ((generatedRoot.w_13_0 ctx), ((generatedRoot.w_137_1 ctx), ((generatedRoot.w_137_3 ctx), ((generatedRoot.w_13_2 ctx), ((generatedRoot.w_13_1 ctx), ((generatedRoot.w_13_3 ctx), ()))))))))))))) next) (Int.toNat (1)) ((generatedRoot.w_144_0 ctx), ((generatedRoot.w_144_0 ctx), ((generatedRoot.w_144_0 ctx), ((generatedRoot.w_144_0 ctx), ())))) ((generatedRoot.w_145_0 ctx), ((generatedRoot.w_145_1 ctx), ((generatedRoot.w_145_2 ctx), ((generatedRoot.w_145_3 ctx), ()))))
  t190 : 0 ≤ (1)
  h147_3 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_generatedRoot_147 backend hashModel params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, ((generatedRoot.w_17_0 ctx), ((generatedRoot.w_137_0 ctx), ((generatedRoot.w_137_2 ctx), ((generatedRoot.w_14_0 ctx), ((generatedRoot.w_137_1 ctx), ((generatedRoot.w_137_3 ctx), ((generatedRoot.w_14_2 ctx), ((generatedRoot.w_14_1 ctx), ((generatedRoot.w_14_3 ctx), ()))))))))))))) next) (Int.toNat (1)) ((generatedRoot.w_146_0 ctx), ((generatedRoot.w_146_0 ctx), ((generatedRoot.w_146_0 ctx), ((generatedRoot.w_146_0 ctx), ())))) ((generatedRoot.w_147_0 ctx), ((generatedRoot.w_147_1 ctx), ((generatedRoot.w_147_2 ctx), ((generatedRoot.w_147_3 ctx), ()))))

theorem generatedRoot.facts_5 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {ctx : generatedRoot.Ctx backend hashModel params} {outputs : (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat4 × mxxMat4 × mxxMat4 × mxxMat4 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × Unit}
    (h : generatedRoot.constraints_5 backend hashModel params (generatedRoot.w_0_0 ctx) (generatedRoot.w_1_0 ctx) (generatedRoot.w_17_0 ctx) (generatedRoot.w_108_0 ctx) (generatedRoot.w_118_0 ctx) (generatedRoot.w_119_0 ctx) (generatedRoot.w_2_0 ctx) (generatedRoot.w_6_0 ctx) (generatedRoot.w_6_1 ctx) (generatedRoot.w_6_2 ctx) (generatedRoot.w_6_3 ctx) (generatedRoot.w_9_0 ctx) (generatedRoot.w_9_1 ctx) (generatedRoot.w_9_2 ctx) (generatedRoot.w_9_3 ctx) (generatedRoot.w_10_0 ctx) (generatedRoot.w_10_1 ctx) (generatedRoot.w_10_2 ctx) (generatedRoot.w_10_3 ctx) (generatedRoot.w_11_0 ctx) (generatedRoot.w_11_1 ctx) (generatedRoot.w_11_2 ctx) (generatedRoot.w_11_3 ctx) (generatedRoot.w_12_0 ctx) (generatedRoot.w_12_1 ctx) (generatedRoot.w_12_2 ctx) (generatedRoot.w_12_3 ctx) (generatedRoot.w_13_0 ctx) (generatedRoot.w_13_1 ctx) (generatedRoot.w_13_2 ctx) (generatedRoot.w_13_3 ctx) (generatedRoot.w_14_0 ctx) (generatedRoot.w_14_1 ctx) (generatedRoot.w_14_2 ctx) (generatedRoot.w_14_3 ctx) (generatedRoot.w_116_0 ctx) (generatedRoot.w_116_1 ctx) (generatedRoot.w_116_2 ctx) (generatedRoot.w_116_3 ctx) (generatedRoot.w_117_0 ctx) (generatedRoot.w_117_1 ctx) (generatedRoot.w_117_2 ctx) (generatedRoot.w_117_3 ctx) (generatedRoot.w_120_0 ctx) (generatedRoot.w_120_1 ctx) (generatedRoot.w_120_2 ctx) (generatedRoot.w_120_3 ctx) (generatedRoot.w_120_4 ctx) (generatedRoot.w_120_5 ctx) (generatedRoot.w_120_6 ctx) (generatedRoot.w_120_7 ctx) (generatedRoot.w_121_0 ctx) (generatedRoot.w_121_1 ctx) (generatedRoot.w_121_2 ctx) (generatedRoot.w_121_3 ctx) (generatedRoot.w_122_0 ctx) (generatedRoot.w_122_1 ctx) (generatedRoot.w_122_2 ctx) (generatedRoot.w_122_3 ctx) (generatedRoot.w_123_0 ctx) (generatedRoot.w_123_1 ctx) (generatedRoot.w_123_2 ctx) (generatedRoot.w_123_3 ctx) (generatedRoot.w_124_0 ctx) (generatedRoot.w_124_1 ctx) (generatedRoot.w_124_2 ctx) (generatedRoot.w_124_3 ctx) (generatedRoot.w_125_0 ctx) (generatedRoot.w_126_0 ctx) (generatedRoot.w_127_0 ctx) (generatedRoot.w_128_0 ctx) (generatedRoot.w_129_0 ctx) (generatedRoot.w_130_0 ctx) (generatedRoot.w_131_0 ctx) (generatedRoot.w_132_0 ctx) (generatedRoot.w_133_0 ctx) (generatedRoot.w_134_0 ctx) (generatedRoot.w_135_0 ctx) (generatedRoot.w_136_0 ctx) (generatedRoot.w_137_0 ctx) (generatedRoot.w_137_1 ctx) (generatedRoot.w_137_2 ctx) (generatedRoot.w_137_3 ctx) (generatedRoot.w_138_0 ctx) (generatedRoot.w_139_0 ctx) (generatedRoot.w_139_1 ctx) (generatedRoot.w_139_2 ctx) (generatedRoot.w_139_3 ctx) (generatedRoot.w_140_0 ctx) (generatedRoot.w_141_0 ctx) (generatedRoot.w_141_1 ctx) (generatedRoot.w_141_2 ctx) (generatedRoot.w_141_3 ctx) (generatedRoot.w_142_0 ctx) (generatedRoot.w_143_0 ctx) (generatedRoot.w_143_1 ctx) (generatedRoot.w_143_2 ctx) (generatedRoot.w_143_3 ctx) (generatedRoot.w_144_0 ctx) (generatedRoot.w_145_0 ctx) (generatedRoot.w_145_1 ctx) (generatedRoot.w_145_2 ctx) (generatedRoot.w_145_3 ctx) (generatedRoot.w_146_0 ctx) (generatedRoot.w_147_0 ctx) (generatedRoot.w_147_1 ctx) (generatedRoot.w_147_2 ctx) (generatedRoot.w_147_3 ctx)) :
    generatedRoot.Facts_5 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_6 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Prop where
  t192 : 0 ≤ (1)
  h149_3 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_generatedRoot_149 backend hashModel params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, ((generatedRoot.w_17_0 ctx), ((generatedRoot.w_137_0 ctx), ((generatedRoot.w_137_2 ctx), ((generatedRoot.w_15_0 ctx), ((generatedRoot.w_137_1 ctx), ((generatedRoot.w_137_3 ctx), ((generatedRoot.w_15_2 ctx), ((generatedRoot.w_15_1 ctx), ((generatedRoot.w_15_3 ctx), ()))))))))))))) next) (Int.toNat (1)) ((generatedRoot.w_148_0 ctx), ((generatedRoot.w_148_0 ctx), ((generatedRoot.w_148_0 ctx), ((generatedRoot.w_148_0 ctx), ())))) ((generatedRoot.w_149_0 ctx), ((generatedRoot.w_149_1 ctx), ((generatedRoot.w_149_2 ctx), ((generatedRoot.w_149_3 ctx), ()))))
  t194 : 0 ≤ (1)
  h151_3 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_generatedRoot_151 backend hashModel params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, ((generatedRoot.w_17_0 ctx), ((generatedRoot.w_137_0 ctx), ((generatedRoot.w_137_2 ctx), ((generatedRoot.w_7_0 ctx), ((generatedRoot.w_137_1 ctx), ((generatedRoot.w_137_3 ctx), ((generatedRoot.w_7_2 ctx), ((generatedRoot.w_7_1 ctx), ((generatedRoot.w_7_3 ctx), ()))))))))))))) next) (Int.toNat (1)) ((generatedRoot.w_150_0 ctx), ((generatedRoot.w_150_0 ctx), ((generatedRoot.w_150_0 ctx), ((generatedRoot.w_150_0 ctx), ())))) ((generatedRoot.w_151_0 ctx), ((generatedRoot.w_151_1 ctx), ((generatedRoot.w_151_2 ctx), ((generatedRoot.w_151_3 ctx), ()))))
  t196 : 0 ≤ (0)
  t197 : (0) < 900
  h155_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((generatedRoot.w_17_0 ctx)) (generatedRoot.w_155_0 ctx)
  h156_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) ((generatedRoot.w_17_0 ctx)) (generatedRoot.w_156_0 ctx)
  h157_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((generatedRoot.w_17_0 ctx)) (generatedRoot.w_157_0 ctx)
  h158_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) ((generatedRoot.w_17_0 ctx)) (generatedRoot.w_158_0 ctx)
  t202 : (14400) = 14400
  h161_7 : (∀ i : Fin 14400, parallel_generatedRoot_161 backend hashModel params i ((generatedRoot.w_17_0 ctx), (((generatedRoot.w_154_0 ctx) i), ((generatedRoot.w_155_0 ctx), ((generatedRoot.w_156_0 ctx), ((generatedRoot.w_0_0 ctx), ((generatedRoot.w_157_0 ctx), ((generatedRoot.w_158_0 ctx), ((generatedRoot.w_1_0 ctx), (((generatedRoot.w_159_0 ctx) i), (((generatedRoot.w_160_0 ctx) i), ((generatedRoot.w_2_0 ctx), ()))))))))))) (((generatedRoot.w_161_0 ctx) i), (((generatedRoot.w_161_1 ctx) i), (((generatedRoot.w_161_2 ctx) i), (((generatedRoot.w_161_3 ctx) i), (((generatedRoot.w_161_4 ctx) i), (((generatedRoot.w_161_5 ctx) i), (((generatedRoot.w_161_6 ctx) i), (((generatedRoot.w_161_7 ctx) i), ())))))))))
  t204 : (7200) = 7200
  h162_3 : (∀ i : Fin 7200, parallel_generatedRoot_162 backend hashModel params i ((generatedRoot.w_161_0 ctx), ((generatedRoot.w_161_1 ctx), ((generatedRoot.w_161_2 ctx), ((generatedRoot.w_161_3 ctx), ())))) (((generatedRoot.w_162_0 ctx) i), (((generatedRoot.w_162_1 ctx) i), (((generatedRoot.w_162_2 ctx) i), (((generatedRoot.w_162_3 ctx) i), ())))))
  t206 : (14400) = 14400
  h165_7 : (∀ i : Fin 14400, parallel_generatedRoot_165 backend hashModel params i ((generatedRoot.w_17_0 ctx), (((generatedRoot.w_153_0 ctx) i), ((generatedRoot.w_162_0 ctx), ((generatedRoot.w_162_2 ctx), ((generatedRoot.w_0_0 ctx), ((generatedRoot.w_162_1 ctx), ((generatedRoot.w_162_3 ctx), ((generatedRoot.w_1_0 ctx), (((generatedRoot.w_163_0 ctx) i), (((generatedRoot.w_164_0 ctx) i), ((generatedRoot.w_2_0 ctx), ()))))))))))) (((generatedRoot.w_165_0 ctx) i), (((generatedRoot.w_165_1 ctx) i), (((generatedRoot.w_165_2 ctx) i), (((generatedRoot.w_165_3 ctx) i), (((generatedRoot.w_165_4 ctx) i), (((generatedRoot.w_165_5 ctx) i), (((generatedRoot.w_165_6 ctx) i), (((generatedRoot.w_165_7 ctx) i), ())))))))))
  t208 : (7200) = 7200
  h166_3 : (∀ i : Fin 7200, parallel_generatedRoot_166 backend hashModel params i ((generatedRoot.w_165_0 ctx), ((generatedRoot.w_165_1 ctx), ((generatedRoot.w_165_2 ctx), ((generatedRoot.w_165_3 ctx), ())))) (((generatedRoot.w_166_0 ctx) i), (((generatedRoot.w_166_1 ctx) i), (((generatedRoot.w_166_2 ctx) i), (((generatedRoot.w_166_3 ctx) i), ())))))
  t210 : (7200) = 7200
  h167_3 : (∀ i : Fin 7200, parallel_generatedRoot_167 backend hashModel params i ((generatedRoot.w_17_0 ctx), (((generatedRoot.w_166_0 ctx) i), (((generatedRoot.w_166_2 ctx) i), ((generatedRoot.w_5_0 ctx), (((generatedRoot.w_166_1 ctx) i), (((generatedRoot.w_166_3 ctx) i), ((generatedRoot.w_5_2 ctx), ((generatedRoot.w_5_1 ctx), ((generatedRoot.w_5_3 ctx), ()))))))))) (((generatedRoot.w_167_0 ctx) i), (((generatedRoot.w_167_1 ctx) i), (((generatedRoot.w_167_2 ctx) i), (((generatedRoot.w_167_3 ctx) i), ())))))
  t212 : (7200) = 7200
  h168_3 : (∀ i : Fin 7200, parallel_generatedRoot_168 backend hashModel params i (((generatedRoot.w_167_0 ctx) i), (((generatedRoot.w_167_1 ctx) i), (((generatedRoot.w_167_2 ctx) i), (((generatedRoot.w_167_3 ctx) i), ())))) (((generatedRoot.w_168_0 ctx) i), (((generatedRoot.w_168_1 ctx) i), (((generatedRoot.w_168_2 ctx) i), (((generatedRoot.w_168_3 ctx) i), ())))))
  h169_0 : MxxRuntime.familyGetStatic (generatedRoot.w_168_0 ctx) (0) (generatedRoot.w_169_0 ctx)
  h170_0 : MxxRuntime.familyGetStatic (generatedRoot.w_168_0 ctx) (1) (generatedRoot.w_170_0 ctx)
  h172_0 : MxxRuntime.familyGetStatic (generatedRoot.w_168_1 ctx) (0) (generatedRoot.w_172_0 ctx)
  h173_0 : MxxRuntime.familyGetStatic (generatedRoot.w_168_1 ctx) (1) (generatedRoot.w_173_0 ctx)
  h175_0 : MxxRuntime.familyGetStatic (generatedRoot.w_168_2 ctx) (0) (generatedRoot.w_175_0 ctx)
  h176_0 : MxxRuntime.familyGetStatic (generatedRoot.w_168_2 ctx) (1) (generatedRoot.w_176_0 ctx)
  h178_0 : MxxRuntime.familyGetStatic (generatedRoot.w_168_3 ctx) (0) (generatedRoot.w_178_0 ctx)
  h179_0 : MxxRuntime.familyGetStatic (generatedRoot.w_168_3 ctx) (1) (generatedRoot.w_179_0 ctx)
  t222 : 0 ≤ (6)
  h181_3 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_generatedRoot_181 backend hashModel params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, ((generatedRoot.w_168_0 ctx), ((generatedRoot.w_168_1 ctx), ((generatedRoot.w_168_2 ctx), ((generatedRoot.w_168_3 ctx), ())))))))) next) (Int.toNat (6)) ((generatedRoot.w_171_0 ctx), ((generatedRoot.w_174_0 ctx), ((generatedRoot.w_177_0 ctx), ((generatedRoot.w_180_0 ctx), ())))) ((generatedRoot.w_181_0 ctx), ((generatedRoot.w_181_1 ctx), ((generatedRoot.w_181_2 ctx), ((generatedRoot.w_181_3 ctx), ()))))

theorem generatedRoot.facts_6 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {ctx : generatedRoot.Ctx backend hashModel params} {outputs : (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat4 × mxxMat4 × mxxMat4 × mxxMat4 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × Unit}
    (h : generatedRoot.constraints_6 backend hashModel params (generatedRoot.w_0_0 ctx) (generatedRoot.w_1_0 ctx) (generatedRoot.w_17_0 ctx) (generatedRoot.w_153_0 ctx) (generatedRoot.w_154_0 ctx) (generatedRoot.w_159_0 ctx) (generatedRoot.w_160_0 ctx) (generatedRoot.w_163_0 ctx) (generatedRoot.w_164_0 ctx) (generatedRoot.w_2_0 ctx) (generatedRoot.w_5_0 ctx) (generatedRoot.w_5_1 ctx) (generatedRoot.w_5_2 ctx) (generatedRoot.w_5_3 ctx) (generatedRoot.w_7_0 ctx) (generatedRoot.w_7_1 ctx) (generatedRoot.w_7_2 ctx) (generatedRoot.w_7_3 ctx) (generatedRoot.w_15_0 ctx) (generatedRoot.w_15_1 ctx) (generatedRoot.w_15_2 ctx) (generatedRoot.w_15_3 ctx) (generatedRoot.w_137_0 ctx) (generatedRoot.w_137_1 ctx) (generatedRoot.w_137_2 ctx) (generatedRoot.w_137_3 ctx) (generatedRoot.w_148_0 ctx) (generatedRoot.w_149_0 ctx) (generatedRoot.w_149_1 ctx) (generatedRoot.w_149_2 ctx) (generatedRoot.w_149_3 ctx) (generatedRoot.w_150_0 ctx) (generatedRoot.w_151_0 ctx) (generatedRoot.w_151_1 ctx) (generatedRoot.w_151_2 ctx) (generatedRoot.w_151_3 ctx) (generatedRoot.w_155_0 ctx) (generatedRoot.w_156_0 ctx) (generatedRoot.w_157_0 ctx) (generatedRoot.w_158_0 ctx) (generatedRoot.w_161_0 ctx) (generatedRoot.w_161_1 ctx) (generatedRoot.w_161_2 ctx) (generatedRoot.w_161_3 ctx) (generatedRoot.w_161_4 ctx) (generatedRoot.w_161_5 ctx) (generatedRoot.w_161_6 ctx) (generatedRoot.w_161_7 ctx) (generatedRoot.w_162_0 ctx) (generatedRoot.w_162_1 ctx) (generatedRoot.w_162_2 ctx) (generatedRoot.w_162_3 ctx) (generatedRoot.w_165_0 ctx) (generatedRoot.w_165_1 ctx) (generatedRoot.w_165_2 ctx) (generatedRoot.w_165_3 ctx) (generatedRoot.w_165_4 ctx) (generatedRoot.w_165_5 ctx) (generatedRoot.w_165_6 ctx) (generatedRoot.w_165_7 ctx) (generatedRoot.w_166_0 ctx) (generatedRoot.w_166_1 ctx) (generatedRoot.w_166_2 ctx) (generatedRoot.w_166_3 ctx) (generatedRoot.w_167_0 ctx) (generatedRoot.w_167_1 ctx) (generatedRoot.w_167_2 ctx) (generatedRoot.w_167_3 ctx) (generatedRoot.w_168_0 ctx) (generatedRoot.w_168_1 ctx) (generatedRoot.w_168_2 ctx) (generatedRoot.w_168_3 ctx) (generatedRoot.w_169_0 ctx) (generatedRoot.w_170_0 ctx) (generatedRoot.w_171_0 ctx) (generatedRoot.w_172_0 ctx) (generatedRoot.w_173_0 ctx) (generatedRoot.w_174_0 ctx) (generatedRoot.w_175_0 ctx) (generatedRoot.w_176_0 ctx) (generatedRoot.w_177_0 ctx) (generatedRoot.w_178_0 ctx) (generatedRoot.w_179_0 ctx) (generatedRoot.w_180_0 ctx) (generatedRoot.w_181_0 ctx) (generatedRoot.w_181_1 ctx) (generatedRoot.w_181_2 ctx) (generatedRoot.w_181_3 ctx)) :
    generatedRoot.Facts_6 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_7 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Prop where
  h182_0 : MxxRuntime.tensorRuns (generatedRoot.w_152_0 ctx) (generatedRoot.w_181_0 ctx) (generatedRoot.w_182_0 ctx)
  t225 : 0 ≤ (0)
  t226 : (0) < 900
  h184_0 : MxxRuntime.tensorRuns (generatedRoot.w_183_0 ctx) (generatedRoot.w_181_1 ctx) (generatedRoot.w_184_0 ctx)
  t228 : 0 ≤ (0)
  t229 : (0) < 900
  h186_0 : MxxRuntime.tensorRuns (generatedRoot.w_185_0 ctx) (generatedRoot.w_181_2 ctx) (generatedRoot.w_186_0 ctx)
  t231 : 0 ≤ (0)
  t232 : (0) < 900
  h188_0 : MxxRuntime.tensorRuns (generatedRoot.w_187_0 ctx) (generatedRoot.w_181_3 ctx) (generatedRoot.w_188_0 ctx)
  t234 : 0 ≤ (899)
  h189_3 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat4 × mxxMat4 × mxxMat4 × mxxMat4 × Unit) => sequential_generatedRoot_189 backend hashModel params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, ((generatedRoot.w_168_0 ctx), ((generatedRoot.w_168_1 ctx), ((generatedRoot.w_168_2 ctx), ((generatedRoot.w_168_3 ctx), ())))))))) next) (Int.toNat (899)) ((generatedRoot.w_182_0 ctx), ((generatedRoot.w_184_0 ctx), ((generatedRoot.w_186_0 ctx), ((generatedRoot.w_188_0 ctx), ())))) ((generatedRoot.w_189_0 ctx), ((generatedRoot.w_189_1 ctx), ((generatedRoot.w_189_2 ctx), ((generatedRoot.w_189_3 ctx), ()))))
  h190_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((generatedRoot.w_17_0 ctx)) (generatedRoot.w_190_0 ctx)
  h191_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) ((generatedRoot.w_17_0 ctx)) (generatedRoot.w_191_0 ctx)
  h192_0 : MxxRuntime.concatRows (generatedRoot.w_190_0 ctx) (generatedRoot.w_191_0 ctx) (generatedRoot.w_192_0 ctx)
  h193_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (3), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((generatedRoot.w_17_0 ctx)) (generatedRoot.w_193_0 ctx)
  h194_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_193_0 ctx) (generatedRoot.w_194_0 ctx)
  h196_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((generatedRoot.w_17_0 ctx)) (generatedRoot.w_196_0 ctx)
  h197_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) ((generatedRoot.w_17_0 ctx)) (generatedRoot.w_197_0 ctx)
  h198_0 : MxxRuntime.concatRows (generatedRoot.w_196_0 ctx) (generatedRoot.w_197_0 ctx) (generatedRoot.w_198_0 ctx)
  h199_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (3), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) ((generatedRoot.w_17_0 ctx)) (generatedRoot.w_199_0 ctx)
  h200_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_199_0 ctx) (generatedRoot.w_200_0 ctx)
  t246 : 0 ≤ 0
  t247 : 0 < 1
  t248 : 1 ≤ 2
  t249 : 0 ≤ 0
  t250 : 0 < 25
  t251 : 25 ≤ 25
  h203_0 : MxxRuntime.sliceMatrix (generatedRoot.w_202_0 ctx) 0 1 0 25 (generatedRoot.w_203_0 ctx)
  h204_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (3), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((generatedRoot.w_17_0 ctx)) (generatedRoot.w_204_0 ctx)
  h205_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_204_0 ctx) (generatedRoot.w_205_0 ctx)
  h207_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (3), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) ((generatedRoot.w_17_0 ctx)) (generatedRoot.w_207_0 ctx)

theorem generatedRoot.facts_7 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {ctx : generatedRoot.Ctx backend hashModel params} {outputs : (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat4 × mxxMat4 × mxxMat4 × mxxMat4 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × Unit}
    (h : generatedRoot.constraints_7 backend hashModel params (generatedRoot.w_17_0 ctx) (generatedRoot.w_152_0 ctx) (generatedRoot.w_168_0 ctx) (generatedRoot.w_168_1 ctx) (generatedRoot.w_168_2 ctx) (generatedRoot.w_168_3 ctx) (generatedRoot.w_181_0 ctx) (generatedRoot.w_181_1 ctx) (generatedRoot.w_181_2 ctx) (generatedRoot.w_181_3 ctx) (generatedRoot.w_182_0 ctx) (generatedRoot.w_183_0 ctx) (generatedRoot.w_184_0 ctx) (generatedRoot.w_185_0 ctx) (generatedRoot.w_186_0 ctx) (generatedRoot.w_187_0 ctx) (generatedRoot.w_188_0 ctx) (generatedRoot.w_189_0 ctx) (generatedRoot.w_189_1 ctx) (generatedRoot.w_189_2 ctx) (generatedRoot.w_189_3 ctx) (generatedRoot.w_190_0 ctx) (generatedRoot.w_191_0 ctx) (generatedRoot.w_192_0 ctx) (generatedRoot.w_193_0 ctx) (generatedRoot.w_194_0 ctx) (generatedRoot.w_196_0 ctx) (generatedRoot.w_197_0 ctx) (generatedRoot.w_198_0 ctx) (generatedRoot.w_199_0 ctx) (generatedRoot.w_200_0 ctx) (generatedRoot.w_202_0 ctx) (generatedRoot.w_203_0 ctx) (generatedRoot.w_204_0 ctx) (generatedRoot.w_205_0 ctx) (generatedRoot.w_207_0 ctx)) :
    generatedRoot.Facts_7 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_8 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Prop where
  h208_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_207_0 ctx) (generatedRoot.w_208_0 ctx)
  t257 : 0 ≤ 0
  t258 : 0 < 1
  t259 : 1 ≤ 2
  t260 : 0 ≤ 0
  t261 : 0 < 25
  t262 : 25 ≤ 25
  h211_0 : MxxRuntime.sliceMatrix (generatedRoot.w_210_0 ctx) 0 1 0 25 (generatedRoot.w_211_0 ctx)
  t264 : 0 ≤ 1
  t265 : 1 < 2
  t266 : 2 ≤ 2
  t267 : 0 ≤ 0
  t268 : 0 < 25
  t269 : 25 ≤ 25
  h212_0 : MxxRuntime.sliceMatrix (generatedRoot.w_202_0 ctx) 1 2 0 25 (generatedRoot.w_212_0 ctx)
  t271 : 0 ≤ 1
  t272 : 1 < 2
  t273 : 2 ≤ 2
  t274 : 0 ≤ 0
  t275 : 0 < 25
  t276 : 25 ≤ 25
  h213_0 : MxxRuntime.sliceMatrix (generatedRoot.w_210_0 ctx) 1 2 0 25 (generatedRoot.w_213_0 ctx)
  h214_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (0), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((generatedRoot.w_17_0 ctx)) (generatedRoot.w_214_0 ctx)
  h215_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (0), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) ((generatedRoot.w_17_0 ctx)) (generatedRoot.w_215_0 ctx)
  h216_0 : MxxRuntime.concatRows (generatedRoot.w_214_0 ctx) (generatedRoot.w_215_0 ctx) (generatedRoot.w_216_0 ctx)
  h217_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((generatedRoot.w_17_0 ctx)) (generatedRoot.w_217_0 ctx)
  h218_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_217_0 ctx) (generatedRoot.w_218_0 ctx)
  h220_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (0), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((generatedRoot.w_17_0 ctx)) (generatedRoot.w_220_0 ctx)
  h221_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (0), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) ((generatedRoot.w_17_0 ctx)) (generatedRoot.w_221_0 ctx)
  h222_0 : MxxRuntime.concatRows (generatedRoot.w_220_0 ctx) (generatedRoot.w_221_0 ctx) (generatedRoot.w_222_0 ctx)
  h223_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) ((generatedRoot.w_17_0 ctx)) (generatedRoot.w_223_0 ctx)
  h224_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_223_0 ctx) (generatedRoot.w_224_0 ctx)

theorem generatedRoot.facts_8 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {ctx : generatedRoot.Ctx backend hashModel params} {outputs : (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat4 × mxxMat4 × mxxMat4 × mxxMat4 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × Unit}
    (h : generatedRoot.constraints_8 backend hashModel (generatedRoot.w_17_0 ctx) (generatedRoot.w_202_0 ctx) (generatedRoot.w_207_0 ctx) (generatedRoot.w_208_0 ctx) (generatedRoot.w_210_0 ctx) (generatedRoot.w_211_0 ctx) (generatedRoot.w_212_0 ctx) (generatedRoot.w_213_0 ctx) (generatedRoot.w_214_0 ctx) (generatedRoot.w_215_0 ctx) (generatedRoot.w_216_0 ctx) (generatedRoot.w_217_0 ctx) (generatedRoot.w_218_0 ctx) (generatedRoot.w_220_0 ctx) (generatedRoot.w_221_0 ctx) (generatedRoot.w_222_0 ctx) (generatedRoot.w_223_0 ctx) (generatedRoot.w_224_0 ctx)) :
    generatedRoot.Facts_8 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_9 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Prop where
  t288 : 0 ≤ 0
  t289 : 0 < 1
  t290 : 1 ≤ 2
  t291 : 0 ≤ 0
  t292 : 0 < 25
  t293 : 25 ≤ 25
  h227_0 : MxxRuntime.sliceMatrix (generatedRoot.w_226_0 ctx) 0 1 0 25 (generatedRoot.w_227_0 ctx)
  t295 : 0 ≤ 1
  t296 : 1 < 2
  t297 : 2 ≤ 2
  t298 : 0 ≤ 0
  t299 : 0 < 25
  t300 : 25 ≤ 25
  h228_0 : MxxRuntime.sliceMatrix (generatedRoot.w_226_0 ctx) 1 2 0 25 (generatedRoot.w_228_0 ctx)
  h229_0 : MxxRuntime.concatRows (generatedRoot.w_227_0 ctx) (generatedRoot.w_228_0 ctx) (generatedRoot.w_229_0 ctx)
  h230_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (2), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((generatedRoot.w_17_0 ctx)) (generatedRoot.w_230_0 ctx)
  h231_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (2), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) ((generatedRoot.w_17_0 ctx)) (generatedRoot.w_231_0 ctx)
  h232_0 : MxxRuntime.concatRows (generatedRoot.w_230_0 ctx) (generatedRoot.w_231_0 ctx) (generatedRoot.w_232_0 ctx)
  h233_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (3), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((generatedRoot.w_17_0 ctx)) (generatedRoot.w_233_0 ctx)
  h234_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_233_0 ctx) (generatedRoot.w_234_0 ctx)
  h236_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (2), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((generatedRoot.w_17_0 ctx)) (generatedRoot.w_236_0 ctx)
  h237_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (2), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) ((generatedRoot.w_17_0 ctx)) (generatedRoot.w_237_0 ctx)
  h238_0 : MxxRuntime.concatRows (generatedRoot.w_236_0 ctx) (generatedRoot.w_237_0 ctx) (generatedRoot.w_238_0 ctx)
  h239_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (3), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) ((generatedRoot.w_17_0 ctx)) (generatedRoot.w_239_0 ctx)
  h240_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_239_0 ctx) (generatedRoot.w_240_0 ctx)
  t313 : 0 ≤ 0
  t314 : 0 < 1
  t315 : 1 ≤ 2
  t316 : 0 ≤ 0
  t317 : 0 < 25
  t318 : 25 ≤ 25
  h243_0 : MxxRuntime.sliceMatrix (generatedRoot.w_242_0 ctx) 0 1 0 25 (generatedRoot.w_243_0 ctx)

theorem generatedRoot.facts_9 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {ctx : generatedRoot.Ctx backend hashModel params} {outputs : (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat4 × mxxMat4 × mxxMat4 × mxxMat4 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × Unit}
    (h : generatedRoot.constraints_9 backend hashModel (generatedRoot.w_17_0 ctx) (generatedRoot.w_226_0 ctx) (generatedRoot.w_227_0 ctx) (generatedRoot.w_228_0 ctx) (generatedRoot.w_229_0 ctx) (generatedRoot.w_230_0 ctx) (generatedRoot.w_231_0 ctx) (generatedRoot.w_232_0 ctx) (generatedRoot.w_233_0 ctx) (generatedRoot.w_234_0 ctx) (generatedRoot.w_236_0 ctx) (generatedRoot.w_237_0 ctx) (generatedRoot.w_238_0 ctx) (generatedRoot.w_239_0 ctx) (generatedRoot.w_240_0 ctx) (generatedRoot.w_242_0 ctx) (generatedRoot.w_243_0 ctx)) :
    generatedRoot.Facts_9 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_10 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Prop where
  h244_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_243_0 ctx) (generatedRoot.w_244_0 ctx)
  h246_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((generatedRoot.w_17_0 ctx)) (generatedRoot.w_246_0 ctx)
  h247_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_246_0 ctx) (generatedRoot.w_247_0 ctx)
  h249_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) ((generatedRoot.w_17_0 ctx)) (generatedRoot.w_249_0 ctx)
  h250_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_249_0 ctx) (generatedRoot.w_250_0 ctx)
  t325 : 0 ≤ 0
  t326 : 0 < 1
  t327 : 1 ≤ 2
  t328 : 0 ≤ 0
  t329 : 0 < 25
  t330 : 25 ≤ 25
  h253_0 : MxxRuntime.sliceMatrix (generatedRoot.w_252_0 ctx) 0 1 0 25 (generatedRoot.w_253_0 ctx)
  t332 : 0 ≤ 1
  t333 : 1 < 2
  t334 : 2 ≤ 2
  t335 : 0 ≤ 0
  t336 : 0 < 25
  t337 : 25 ≤ 25
  h254_0 : MxxRuntime.sliceMatrix (generatedRoot.w_252_0 ctx) 1 2 0 25 (generatedRoot.w_254_0 ctx)
  h255_0 : MxxRuntime.concatRows (generatedRoot.w_253_0 ctx) (generatedRoot.w_254_0 ctx) (generatedRoot.w_255_0 ctx)
  t340 : 0 ≤ 1
  t341 : 1 < 2
  t342 : 2 ≤ 2
  t343 : 0 ≤ 0
  t344 : 0 < 25
  t345 : 25 ≤ 25
  h256_0 : MxxRuntime.sliceMatrix (generatedRoot.w_242_0 ctx) 1 2 0 25 (generatedRoot.w_256_0 ctx)
  h257_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_256_0 ctx) (generatedRoot.w_257_0 ctx)
  t348 : 0 ≤ 0
  t349 : 0 < 1
  t350 : 1 ≤ 2
  t351 : 0 ≤ 0

theorem generatedRoot.facts_10 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {ctx : generatedRoot.Ctx backend hashModel params} {outputs : (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat4 × mxxMat4 × mxxMat4 × mxxMat4 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × Unit}
    (h : generatedRoot.constraints_10 backend hashModel (generatedRoot.w_17_0 ctx) (generatedRoot.w_242_0 ctx) (generatedRoot.w_243_0 ctx) (generatedRoot.w_244_0 ctx) (generatedRoot.w_246_0 ctx) (generatedRoot.w_247_0 ctx) (generatedRoot.w_249_0 ctx) (generatedRoot.w_250_0 ctx) (generatedRoot.w_252_0 ctx) (generatedRoot.w_253_0 ctx) (generatedRoot.w_254_0 ctx) (generatedRoot.w_255_0 ctx) (generatedRoot.w_256_0 ctx) (generatedRoot.w_257_0 ctx)) :
    generatedRoot.Facts_10 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_11 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Prop where
  t352 : 0 < 25
  t353 : 25 ≤ 25
  h260_0 : MxxRuntime.sliceMatrix (generatedRoot.w_259_0 ctx) 0 1 0 25 (generatedRoot.w_260_0 ctx)
  h261_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (3), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((generatedRoot.w_17_0 ctx)) (generatedRoot.w_261_0 ctx)
  h262_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_261_0 ctx) (generatedRoot.w_262_0 ctx)
  h264_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (3), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) ((generatedRoot.w_17_0 ctx)) (generatedRoot.w_264_0 ctx)
  h265_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_264_0 ctx) (generatedRoot.w_265_0 ctx)
  t359 : 0 ≤ 0
  t360 : 0 < 1
  t361 : 1 ≤ 2
  t362 : 0 ≤ 0
  t363 : 0 < 25
  t364 : 25 ≤ 25
  h268_0 : MxxRuntime.sliceMatrix (generatedRoot.w_267_0 ctx) 0 1 0 25 (generatedRoot.w_268_0 ctx)
  h269_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_268_0 ctx) (generatedRoot.w_269_0 ctx)
  t367 : 0 ≤ 1
  t368 : 1 < 2
  t369 : 2 ≤ 2
  t370 : 0 ≤ 0
  t371 : 0 < 25
  t372 : 25 ≤ 25
  h271_0 : MxxRuntime.sliceMatrix (generatedRoot.w_267_0 ctx) 1 2 0 25 (generatedRoot.w_271_0 ctx)
  h272_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_271_0 ctx) (generatedRoot.w_272_0 ctx)
  t375 : 0 ≤ 0
  t376 : 0 < 1
  t377 : 1 ≤ 2
  t378 : 0 ≤ 0
  t379 : 0 < 25
  t380 : 25 ≤ 25
  h275_0 : MxxRuntime.sliceMatrix (generatedRoot.w_274_0 ctx) 0 1 0 25 (generatedRoot.w_275_0 ctx)
  t382 : 0 ≤ 1
  t383 : 1 < 2

theorem generatedRoot.facts_11 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {ctx : generatedRoot.Ctx backend hashModel params} {outputs : (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat4 × mxxMat4 × mxxMat4 × mxxMat4 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × Unit}
    (h : generatedRoot.constraints_11 backend hashModel (generatedRoot.w_17_0 ctx) (generatedRoot.w_259_0 ctx) (generatedRoot.w_260_0 ctx) (generatedRoot.w_261_0 ctx) (generatedRoot.w_262_0 ctx) (generatedRoot.w_264_0 ctx) (generatedRoot.w_265_0 ctx) (generatedRoot.w_267_0 ctx) (generatedRoot.w_268_0 ctx) (generatedRoot.w_269_0 ctx) (generatedRoot.w_271_0 ctx) (generatedRoot.w_272_0 ctx) (generatedRoot.w_274_0 ctx) (generatedRoot.w_275_0 ctx)) :
    generatedRoot.Facts_11 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_12 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Prop where
  t384 : 2 ≤ 2
  t385 : 0 ≤ 0
  t386 : 0 < 25
  t387 : 25 ≤ 25
  h276_0 : MxxRuntime.sliceMatrix (generatedRoot.w_259_0 ctx) 1 2 0 25 (generatedRoot.w_276_0 ctx)
  t389 : 0 ≤ 1
  t390 : 1 < 2
  t391 : 2 ≤ 2
  t392 : 0 ≤ 0
  t393 : 0 < 25
  t394 : 25 ≤ 25
  h277_0 : MxxRuntime.sliceMatrix (generatedRoot.w_274_0 ctx) 1 2 0 25 (generatedRoot.w_277_0 ctx)
  h278_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (0), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((generatedRoot.w_17_0 ctx)) (generatedRoot.w_278_0 ctx)
  h279_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (0), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) ((generatedRoot.w_17_0 ctx)) (generatedRoot.w_279_0 ctx)
  h280_0 : MxxRuntime.concatRows (generatedRoot.w_278_0 ctx) (generatedRoot.w_279_0 ctx) (generatedRoot.w_280_0 ctx)
  h281_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (2), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((generatedRoot.w_17_0 ctx)) (generatedRoot.w_281_0 ctx)
  h282_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_281_0 ctx) (generatedRoot.w_282_0 ctx)
  h284_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (0), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((generatedRoot.w_17_0 ctx)) (generatedRoot.w_284_0 ctx)
  h285_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (0), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) ((generatedRoot.w_17_0 ctx)) (generatedRoot.w_285_0 ctx)
  h286_0 : MxxRuntime.concatRows (generatedRoot.w_284_0 ctx) (generatedRoot.w_285_0 ctx) (generatedRoot.w_286_0 ctx)
  h287_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (2), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) ((generatedRoot.w_17_0 ctx)) (generatedRoot.w_287_0 ctx)
  h288_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_287_0 ctx) (generatedRoot.w_288_0 ctx)
  t406 : 0 ≤ 0
  t407 : 0 < 1
  t408 : 1 ≤ 2
  t409 : 0 ≤ 0
  t410 : 0 < 25
  t411 : 25 ≤ 25
  h291_0 : MxxRuntime.sliceMatrix (generatedRoot.w_290_0 ctx) 0 1 0 25 (generatedRoot.w_291_0 ctx)
  t413 : 0 ≤ 1
  t414 : 1 < 2
  t415 : 2 ≤ 2

theorem generatedRoot.facts_12 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {ctx : generatedRoot.Ctx backend hashModel params} {outputs : (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat4 × mxxMat4 × mxxMat4 × mxxMat4 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × Unit}
    (h : generatedRoot.constraints_12 backend hashModel (generatedRoot.w_17_0 ctx) (generatedRoot.w_259_0 ctx) (generatedRoot.w_274_0 ctx) (generatedRoot.w_276_0 ctx) (generatedRoot.w_277_0 ctx) (generatedRoot.w_278_0 ctx) (generatedRoot.w_279_0 ctx) (generatedRoot.w_280_0 ctx) (generatedRoot.w_281_0 ctx) (generatedRoot.w_282_0 ctx) (generatedRoot.w_284_0 ctx) (generatedRoot.w_285_0 ctx) (generatedRoot.w_286_0 ctx) (generatedRoot.w_287_0 ctx) (generatedRoot.w_288_0 ctx) (generatedRoot.w_290_0 ctx) (generatedRoot.w_291_0 ctx)) :
    generatedRoot.Facts_12 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_13 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Prop where
  t416 : 0 ≤ 0
  t417 : 0 < 25
  t418 : 25 ≤ 25
  h292_0 : MxxRuntime.sliceMatrix (generatedRoot.w_290_0 ctx) 1 2 0 25 (generatedRoot.w_292_0 ctx)
  h293_0 : MxxRuntime.concatRows (generatedRoot.w_291_0 ctx) (generatedRoot.w_292_0 ctx) (generatedRoot.w_293_0 ctx)
  h294_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (3), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((generatedRoot.w_17_0 ctx)) (generatedRoot.w_294_0 ctx)
  h295_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_294_0 ctx) (generatedRoot.w_295_0 ctx)
  h297_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (2), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((generatedRoot.w_17_0 ctx)) (generatedRoot.w_297_0 ctx)
  h298_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_297_0 ctx) (generatedRoot.w_298_0 ctx)
  h300_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (2), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) ((generatedRoot.w_17_0 ctx)) (generatedRoot.w_300_0 ctx)
  h301_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_300_0 ctx) (generatedRoot.w_301_0 ctx)
  t427 : 0 ≤ 0
  t428 : 0 < 1
  t429 : 1 ≤ 2
  t430 : 0 ≤ 0
  t431 : 0 < 25
  t432 : 25 ≤ 25
  h304_0 : MxxRuntime.sliceMatrix (generatedRoot.w_303_0 ctx) 0 1 0 25 (generatedRoot.w_304_0 ctx)
  t434 : 0 ≤ 1
  t435 : 1 < 2
  t436 : 2 ≤ 2
  t437 : 0 ≤ 0
  t438 : 0 < 25
  t439 : 25 ≤ 25
  h305_0 : MxxRuntime.sliceMatrix (generatedRoot.w_303_0 ctx) 1 2 0 25 (generatedRoot.w_305_0 ctx)
  h306_0 : MxxRuntime.concatRows (generatedRoot.w_304_0 ctx) (generatedRoot.w_305_0 ctx) (generatedRoot.w_306_0 ctx)
  h307_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (3), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) ((generatedRoot.w_17_0 ctx)) (generatedRoot.w_307_0 ctx)
  h308_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_307_0 ctx) (generatedRoot.w_308_0 ctx)
  t444 : 0 ≤ 0
  t445 : 0 < 1
  t446 : 1 ≤ 2
  t447 : 0 ≤ 0

theorem generatedRoot.facts_13 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {ctx : generatedRoot.Ctx backend hashModel params} {outputs : (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat4 × mxxMat4 × mxxMat4 × mxxMat4 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × Unit}
    (h : generatedRoot.constraints_13 backend hashModel (generatedRoot.w_17_0 ctx) (generatedRoot.w_290_0 ctx) (generatedRoot.w_291_0 ctx) (generatedRoot.w_292_0 ctx) (generatedRoot.w_293_0 ctx) (generatedRoot.w_294_0 ctx) (generatedRoot.w_295_0 ctx) (generatedRoot.w_297_0 ctx) (generatedRoot.w_298_0 ctx) (generatedRoot.w_300_0 ctx) (generatedRoot.w_301_0 ctx) (generatedRoot.w_303_0 ctx) (generatedRoot.w_304_0 ctx) (generatedRoot.w_305_0 ctx) (generatedRoot.w_306_0 ctx) (generatedRoot.w_307_0 ctx) (generatedRoot.w_308_0 ctx)) :
    generatedRoot.Facts_13 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_14 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Prop where
  t448 : 0 < 25
  t449 : 25 ≤ 25
  h311_0 : MxxRuntime.sliceMatrix (generatedRoot.w_310_0 ctx) 0 1 0 25 (generatedRoot.w_311_0 ctx)
  h312_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (3), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((generatedRoot.w_17_0 ctx)) (generatedRoot.w_312_0 ctx)
  h313_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_312_0 ctx) (generatedRoot.w_313_0 ctx)
  h315_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (3), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) ((generatedRoot.w_17_0 ctx)) (generatedRoot.w_315_0 ctx)
  h316_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_315_0 ctx) (generatedRoot.w_316_0 ctx)
  t455 : 0 ≤ 0
  t456 : 0 < 1
  t457 : 1 ≤ 2
  t458 : 0 ≤ 0
  t459 : 0 < 25
  t460 : 25 ≤ 25
  h319_0 : MxxRuntime.sliceMatrix (generatedRoot.w_318_0 ctx) 0 1 0 25 (generatedRoot.w_319_0 ctx)
  t462 : 0 ≤ 1
  t463 : 1 < 2
  t464 : 2 ≤ 2
  t465 : 0 ≤ 0
  t466 : 0 < 25
  t467 : 25 ≤ 25
  h320_0 : MxxRuntime.sliceMatrix (generatedRoot.w_310_0 ctx) 1 2 0 25 (generatedRoot.w_320_0 ctx)
  t469 : 0 ≤ 1
  t470 : 1 < 2
  t471 : 2 ≤ 2
  t472 : 0 ≤ 0
  t473 : 0 < 25
  t474 : 25 ≤ 25
  h321_0 : MxxRuntime.sliceMatrix (generatedRoot.w_318_0 ctx) 1 2 0 25 (generatedRoot.w_321_0 ctx)
  h322_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((generatedRoot.w_17_0 ctx)) (generatedRoot.w_322_0 ctx)
  h323_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) ((generatedRoot.w_17_0 ctx)) (generatedRoot.w_323_0 ctx)
  h324_0 : MxxRuntime.concatRows (generatedRoot.w_322_0 ctx) (generatedRoot.w_323_0 ctx) (generatedRoot.w_324_0 ctx)
  h325_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((generatedRoot.w_17_0 ctx)) (generatedRoot.w_325_0 ctx)

theorem generatedRoot.facts_14 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {ctx : generatedRoot.Ctx backend hashModel params} {outputs : (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat4 × mxxMat4 × mxxMat4 × mxxMat4 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × Unit}
    (h : generatedRoot.constraints_14 backend hashModel (generatedRoot.w_17_0 ctx) (generatedRoot.w_310_0 ctx) (generatedRoot.w_311_0 ctx) (generatedRoot.w_312_0 ctx) (generatedRoot.w_313_0 ctx) (generatedRoot.w_315_0 ctx) (generatedRoot.w_316_0 ctx) (generatedRoot.w_318_0 ctx) (generatedRoot.w_319_0 ctx) (generatedRoot.w_320_0 ctx) (generatedRoot.w_321_0 ctx) (generatedRoot.w_322_0 ctx) (generatedRoot.w_323_0 ctx) (generatedRoot.w_324_0 ctx) (generatedRoot.w_325_0 ctx)) :
    generatedRoot.Facts_14 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_15 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Prop where
  h326_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_325_0 ctx) (generatedRoot.w_326_0 ctx)
  h328_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((generatedRoot.w_17_0 ctx)) (generatedRoot.w_328_0 ctx)
  h329_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) ((generatedRoot.w_17_0 ctx)) (generatedRoot.w_329_0 ctx)
  h330_0 : MxxRuntime.concatRows (generatedRoot.w_328_0 ctx) (generatedRoot.w_329_0 ctx) (generatedRoot.w_330_0 ctx)
  h331_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) ((generatedRoot.w_17_0 ctx)) (generatedRoot.w_331_0 ctx)
  h332_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_331_0 ctx) (generatedRoot.w_332_0 ctx)
  t486 : 0 ≤ 0
  t487 : 0 < 1
  t488 : 1 ≤ 2
  t489 : 0 ≤ 0
  t490 : 0 < 25
  t491 : 25 ≤ 25
  h335_0 : MxxRuntime.sliceMatrix (generatedRoot.w_334_0 ctx) 0 1 0 25 (generatedRoot.w_335_0 ctx)
  h336_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((generatedRoot.w_17_0 ctx)) (generatedRoot.w_336_0 ctx)
  h337_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_336_0 ctx) (generatedRoot.w_337_0 ctx)
  h339_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) ((generatedRoot.w_17_0 ctx)) (generatedRoot.w_339_0 ctx)
  h340_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_339_0 ctx) (generatedRoot.w_340_0 ctx)
  t497 : 0 ≤ 0
  t498 : 0 < 1
  t499 : 1 ≤ 2
  t500 : 0 ≤ 0
  t501 : 0 < 25
  t502 : 25 ≤ 25
  h343_0 : MxxRuntime.sliceMatrix (generatedRoot.w_342_0 ctx) 0 1 0 25 (generatedRoot.w_343_0 ctx)
  t504 : 0 ≤ 1
  t505 : 1 < 2
  t506 : 2 ≤ 2
  t507 : 0 ≤ 0
  t508 : 0 < 25
  t509 : 25 ≤ 25
  h344_0 : MxxRuntime.sliceMatrix (generatedRoot.w_334_0 ctx) 1 2 0 25 (generatedRoot.w_344_0 ctx)
  t511 : 0 ≤ 1

theorem generatedRoot.facts_15 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {ctx : generatedRoot.Ctx backend hashModel params} {outputs : (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat4 × mxxMat4 × mxxMat4 × mxxMat4 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × Unit}
    (h : generatedRoot.constraints_15 backend hashModel (generatedRoot.w_17_0 ctx) (generatedRoot.w_325_0 ctx) (generatedRoot.w_326_0 ctx) (generatedRoot.w_328_0 ctx) (generatedRoot.w_329_0 ctx) (generatedRoot.w_330_0 ctx) (generatedRoot.w_331_0 ctx) (generatedRoot.w_332_0 ctx) (generatedRoot.w_334_0 ctx) (generatedRoot.w_335_0 ctx) (generatedRoot.w_336_0 ctx) (generatedRoot.w_337_0 ctx) (generatedRoot.w_339_0 ctx) (generatedRoot.w_340_0 ctx) (generatedRoot.w_342_0 ctx) (generatedRoot.w_343_0 ctx) (generatedRoot.w_344_0 ctx)) :
    generatedRoot.Facts_15 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_16 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) : Prop where
  t512 : 1 < 2
  t513 : 2 ≤ 2
  t514 : 0 ≤ 0
  t515 : 0 < 25
  t516 : 25 ≤ 25
  h345_0 : MxxRuntime.sliceMatrix (generatedRoot.w_342_0 ctx) 1 2 0 25 (generatedRoot.w_345_0 ctx)
  h346_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((generatedRoot.w_17_0 ctx)) (generatedRoot.w_346_0 ctx)
  h347_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) ((generatedRoot.w_17_0 ctx)) (generatedRoot.w_347_0 ctx)
  h348_0 : MxxRuntime.concatRows (generatedRoot.w_346_0 ctx) (generatedRoot.w_347_0 ctx) (generatedRoot.w_348_0 ctx)
  h349_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (2), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((generatedRoot.w_17_0 ctx)) (generatedRoot.w_349_0 ctx)
  h350_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_349_0 ctx) (generatedRoot.w_350_0 ctx)
  h352_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((generatedRoot.w_17_0 ctx)) (generatedRoot.w_352_0 ctx)
  h353_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) ((generatedRoot.w_17_0 ctx)) (generatedRoot.w_353_0 ctx)
  h354_0 : MxxRuntime.concatRows (generatedRoot.w_352_0 ctx) (generatedRoot.w_353_0 ctx) (generatedRoot.w_354_0 ctx)
  h355_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (2), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) ((generatedRoot.w_17_0 ctx)) (generatedRoot.w_355_0 ctx)
  h356_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_355_0 ctx) (generatedRoot.w_356_0 ctx)
  t528 : 0 ≤ 0
  t529 : 0 < 1
  t530 : 1 ≤ 2
  t531 : 0 ≤ 0
  t532 : 0 < 25
  t533 : 25 ≤ 25
  h359_0 : MxxRuntime.sliceMatrix (generatedRoot.w_358_0 ctx) 0 1 0 25 (generatedRoot.w_359_0 ctx)
  h360_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (2), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((generatedRoot.w_17_0 ctx)) (generatedRoot.w_360_0 ctx)
  h361_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_360_0 ctx) (generatedRoot.w_361_0 ctx)
  h363_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (2), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) ((generatedRoot.w_17_0 ctx)) (generatedRoot.w_363_0 ctx)
  h364_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_363_0 ctx) (generatedRoot.w_364_0 ctx)
  t539 : 0 ≤ 0
  t540 : 0 < 1
  t541 : 1 ≤ 2
  t542 : 0 ≤ 0
  t543 : 0 < 25

theorem generatedRoot.facts_16 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {ctx : generatedRoot.Ctx backend hashModel params} {outputs : (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat4 × mxxMat4 × mxxMat4 × mxxMat4 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × Unit}
    (h : generatedRoot.constraints_16 backend hashModel (generatedRoot.w_17_0 ctx) (generatedRoot.w_342_0 ctx) (generatedRoot.w_345_0 ctx) (generatedRoot.w_346_0 ctx) (generatedRoot.w_347_0 ctx) (generatedRoot.w_348_0 ctx) (generatedRoot.w_349_0 ctx) (generatedRoot.w_350_0 ctx) (generatedRoot.w_352_0 ctx) (generatedRoot.w_353_0 ctx) (generatedRoot.w_354_0 ctx) (generatedRoot.w_355_0 ctx) (generatedRoot.w_356_0 ctx) (generatedRoot.w_358_0 ctx) (generatedRoot.w_359_0 ctx) (generatedRoot.w_360_0 ctx) (generatedRoot.w_361_0 ctx) (generatedRoot.w_363_0 ctx) (generatedRoot.w_364_0 ctx)) :
    generatedRoot.Facts_16 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_17 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) (outputs : (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat4 × mxxMat4 × mxxMat4 × mxxMat4 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × Unit) : Prop where
  t544 : 25 ≤ 25
  h367_0 : MxxRuntime.sliceMatrix (generatedRoot.w_366_0 ctx) 0 1 0 25 (generatedRoot.w_367_0 ctx)
  t546 : 0 ≤ 1
  t547 : 1 < 2
  t548 : 2 ≤ 2
  t549 : 0 ≤ 0
  t550 : 0 < 25
  t551 : 25 ≤ 25
  h368_0 : MxxRuntime.sliceMatrix (generatedRoot.w_358_0 ctx) 1 2 0 25 (generatedRoot.w_368_0 ctx)
  t553 : 0 ≤ 1
  t554 : 1 < 2
  t555 : 2 ≤ 2
  t556 : 0 ≤ 0
  t557 : 0 < 25
  t558 : 25 ≤ 25
  h369_0 : MxxRuntime.sliceMatrix (generatedRoot.w_366_0 ctx) 1 2 0 25 (generatedRoot.w_369_0 ctx)
  hout : outputs = generatedRoot.outs ctx

theorem generatedRoot.facts_17 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {ctx : generatedRoot.Ctx backend hashModel params} {outputs : (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat4 × mxxMat4 × mxxMat4 × mxxMat4 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × Unit}
    (h : generatedRoot.constraints_17 (generatedRoot.w_3_0 ctx) (generatedRoot.w_3_1 ctx) (generatedRoot.w_3_2 ctx) (generatedRoot.w_3_3 ctx) (generatedRoot.w_4_0 ctx) (generatedRoot.w_4_1 ctx) (generatedRoot.w_4_2 ctx) (generatedRoot.w_4_3 ctx) (generatedRoot.w_5_0 ctx) (generatedRoot.w_5_1 ctx) (generatedRoot.w_5_2 ctx) (generatedRoot.w_5_3 ctx) (generatedRoot.w_6_0 ctx) (generatedRoot.w_6_1 ctx) (generatedRoot.w_6_2 ctx) (generatedRoot.w_6_3 ctx) (generatedRoot.w_7_0 ctx) (generatedRoot.w_7_1 ctx) (generatedRoot.w_7_2 ctx) (generatedRoot.w_7_3 ctx) (generatedRoot.w_8_0 ctx) (generatedRoot.w_8_1 ctx) (generatedRoot.w_8_2 ctx) (generatedRoot.w_8_3 ctx) (generatedRoot.w_9_0 ctx) (generatedRoot.w_9_1 ctx) (generatedRoot.w_9_2 ctx) (generatedRoot.w_9_3 ctx) (generatedRoot.w_10_0 ctx) (generatedRoot.w_10_1 ctx) (generatedRoot.w_10_2 ctx) (generatedRoot.w_10_3 ctx) (generatedRoot.w_11_0 ctx) (generatedRoot.w_11_1 ctx) (generatedRoot.w_11_2 ctx) (generatedRoot.w_11_3 ctx) (generatedRoot.w_12_0 ctx) (generatedRoot.w_12_1 ctx) (generatedRoot.w_12_2 ctx) (generatedRoot.w_12_3 ctx) (generatedRoot.w_13_0 ctx) (generatedRoot.w_13_1 ctx) (generatedRoot.w_13_2 ctx) (generatedRoot.w_13_3 ctx) (generatedRoot.w_14_0 ctx) (generatedRoot.w_14_1 ctx) (generatedRoot.w_14_2 ctx) (generatedRoot.w_14_3 ctx) (generatedRoot.w_15_0 ctx) (generatedRoot.w_15_1 ctx) (generatedRoot.w_15_2 ctx) (generatedRoot.w_15_3 ctx) (generatedRoot.w_36_2 ctx) (generatedRoot.w_36_3 ctx) (generatedRoot.w_69_0 ctx) (generatedRoot.w_69_1 ctx) (generatedRoot.w_69_2 ctx) (generatedRoot.w_69_3 ctx) (generatedRoot.w_79_6 ctx) (generatedRoot.w_79_7 ctx) (generatedRoot.w_83_6 ctx) (generatedRoot.w_83_7 ctx) (generatedRoot.w_107_0 ctx) (generatedRoot.w_107_1 ctx) (generatedRoot.w_107_2 ctx) (generatedRoot.w_107_3 ctx) (generatedRoot.w_116_6 ctx) (generatedRoot.w_116_7 ctx) (generatedRoot.w_120_6 ctx) (generatedRoot.w_120_7 ctx) (generatedRoot.w_137_0 ctx) (generatedRoot.w_137_1 ctx) (generatedRoot.w_137_2 ctx) (generatedRoot.w_137_3 ctx) (generatedRoot.w_139_0 ctx) (generatedRoot.w_139_1 ctx) (generatedRoot.w_139_2 ctx) (generatedRoot.w_139_3 ctx) (generatedRoot.w_141_0 ctx) (generatedRoot.w_141_1 ctx) (generatedRoot.w_141_2 ctx) (generatedRoot.w_141_3 ctx) (generatedRoot.w_143_0 ctx) (generatedRoot.w_143_1 ctx) (generatedRoot.w_143_2 ctx) (generatedRoot.w_143_3 ctx) (generatedRoot.w_145_0 ctx) (generatedRoot.w_145_1 ctx) (generatedRoot.w_145_2 ctx) (generatedRoot.w_145_3 ctx) (generatedRoot.w_147_0 ctx) (generatedRoot.w_147_1 ctx) (generatedRoot.w_147_2 ctx) (generatedRoot.w_147_3 ctx) (generatedRoot.w_149_0 ctx) (generatedRoot.w_149_1 ctx) (generatedRoot.w_149_2 ctx) (generatedRoot.w_149_3 ctx) (generatedRoot.w_151_0 ctx) (generatedRoot.w_151_1 ctx) (generatedRoot.w_151_2 ctx) (generatedRoot.w_151_3 ctx) (generatedRoot.w_161_6 ctx) (generatedRoot.w_161_7 ctx) (generatedRoot.w_165_6 ctx) (generatedRoot.w_165_7 ctx) (generatedRoot.w_189_0 ctx) (generatedRoot.w_189_1 ctx) (generatedRoot.w_189_2 ctx) (generatedRoot.w_189_3 ctx) (generatedRoot.w_203_0 ctx) (generatedRoot.w_211_0 ctx) (generatedRoot.w_212_0 ctx) (generatedRoot.w_213_0 ctx) (generatedRoot.w_260_0 ctx) (generatedRoot.w_275_0 ctx) (generatedRoot.w_276_0 ctx) (generatedRoot.w_277_0 ctx) (generatedRoot.w_311_0 ctx) (generatedRoot.w_319_0 ctx) (generatedRoot.w_320_0 ctx) (generatedRoot.w_321_0 ctx) (generatedRoot.w_335_0 ctx) (generatedRoot.w_343_0 ctx) (generatedRoot.w_344_0 ctx) (generatedRoot.w_345_0 ctx) (generatedRoot.w_358_0 ctx) (generatedRoot.w_359_0 ctx) (generatedRoot.w_366_0 ctx) (generatedRoot.w_367_0 ctx) (generatedRoot.w_368_0 ctx) (generatedRoot.w_369_0 ctx) outputs) :
    generatedRoot.Facts_17 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) (outputs : (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat4 × mxxMat4 × mxxMat4 × mxxMat4 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × Unit) : Prop where
  c0 : generatedRoot.Facts_0 ctx
  c1 : generatedRoot.Facts_1 ctx
  c2 : generatedRoot.Facts_2 ctx
  c3 : generatedRoot.Facts_3 ctx
  c4 : generatedRoot.Facts_4 ctx
  c5 : generatedRoot.Facts_5 ctx
  c6 : generatedRoot.Facts_6 ctx
  c7 : generatedRoot.Facts_7 ctx
  c8 : generatedRoot.Facts_8 ctx
  c9 : generatedRoot.Facts_9 ctx
  c10 : generatedRoot.Facts_10 ctx
  c11 : generatedRoot.Facts_11 ctx
  c12 : generatedRoot.Facts_12 ctx
  c13 : generatedRoot.Facts_13 ctx
  c14 : generatedRoot.Facts_14 ctx
  c15 : generatedRoot.Facts_15 ctx
  c16 : generatedRoot.Facts_16 ctx
  c17 : generatedRoot.Facts_17 ctx outputs

theorem generatedRoot.facts {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) {outputs : (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat4 × mxxMat4 × mxxMat4 × mxxMat4 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × Unit}
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
  have r5 := generatedRoot.facts_5 (ctx := ctx) (outputs := outputs) t5.1
  have t6 := t5.2
  have r6 := generatedRoot.facts_6 (ctx := ctx) (outputs := outputs) t6.1
  have t7 := t6.2
  have r7 := generatedRoot.facts_7 (ctx := ctx) (outputs := outputs) t7.1
  have t8 := t7.2
  have r8 := generatedRoot.facts_8 (ctx := ctx) (outputs := outputs) t8.1
  have t9 := t8.2
  have r9 := generatedRoot.facts_9 (ctx := ctx) (outputs := outputs) t9.1
  have t10 := t9.2
  have r10 := generatedRoot.facts_10 (ctx := ctx) (outputs := outputs) t10.1
  have t11 := t10.2
  have r11 := generatedRoot.facts_11 (ctx := ctx) (outputs := outputs) t11.1
  have t12 := t11.2
  have r12 := generatedRoot.facts_12 (ctx := ctx) (outputs := outputs) t12.1
  have t13 := t12.2
  have r13 := generatedRoot.facts_13 (ctx := ctx) (outputs := outputs) t13.1
  have t14 := t13.2
  have r14 := generatedRoot.facts_14 (ctx := ctx) (outputs := outputs) t14.1
  have t15 := t14.2
  have r15 := generatedRoot.facts_15 (ctx := ctx) (outputs := outputs) t15.1
  have t16 := t15.2
  have r16 := generatedRoot.facts_16 (ctx := ctx) (outputs := outputs) t16.1
  have t17 := t16.2
  have r17 := generatedRoot.facts_17 (ctx := ctx) (outputs := outputs) t17
  ⟨r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17⟩

theorem generatedRoot.h3_3_at {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {ctx : generatedRoot.Ctx backend hashModel params} {outputs : (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat4 × mxxMat4 × mxxMat4 × mxxMat4 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × Unit}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 4) :
    ∃ w : parallel_generatedRoot_3.Witness, parallel_generatedRoot_3.Facts (⟨((generatedRoot.w_0_0 ctx), ((generatedRoot.w_1_0 ctx), ((generatedRoot.w_2_0 ctx), ()))), w⟩ : parallel_generatedRoot_3.Ctx backend hashModel params i) (((generatedRoot.w_3_0 ctx) i), (((generatedRoot.w_3_1 ctx) i), (((generatedRoot.w_3_2 ctx) i), (((generatedRoot.w_3_3 ctx) i), ())))) :=
  let ⟨w, h⟩ := F.c0.h3_3 i
  ⟨w, parallel_generatedRoot_3.facts _ h⟩

theorem generatedRoot.h4_3_at {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {ctx : generatedRoot.Ctx backend hashModel params} {outputs : (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat4 × mxxMat4 × mxxMat4 × mxxMat4 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × Unit}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 8) :
    ∃ w : parallel_generatedRoot_4.Witness, parallel_generatedRoot_4.Facts (⟨((generatedRoot.w_0_0 ctx), ((generatedRoot.w_1_0 ctx), ((generatedRoot.w_2_0 ctx), ()))), w⟩ : parallel_generatedRoot_4.Ctx backend hashModel params i) (((generatedRoot.w_4_0 ctx) i), (((generatedRoot.w_4_1 ctx) i), (((generatedRoot.w_4_2 ctx) i), (((generatedRoot.w_4_3 ctx) i), ())))) :=
  let ⟨w, h⟩ := F.c0.h4_3 i
  ⟨w, parallel_generatedRoot_4.facts _ h⟩

theorem generatedRoot.h5_3_at {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {ctx : generatedRoot.Ctx backend hashModel params} {outputs : (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat4 × mxxMat4 × mxxMat4 × mxxMat4 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × Unit}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 8) :
    ∃ w : parallel_generatedRoot_5.Witness, parallel_generatedRoot_5.Facts (⟨((generatedRoot.w_0_0 ctx), ((generatedRoot.w_1_0 ctx), ((generatedRoot.w_2_0 ctx), ()))), w⟩ : parallel_generatedRoot_5.Ctx backend hashModel params i) (((generatedRoot.w_5_0 ctx) i), (((generatedRoot.w_5_1 ctx) i), (((generatedRoot.w_5_2 ctx) i), (((generatedRoot.w_5_3 ctx) i), ())))) :=
  let ⟨w, h⟩ := F.c0.h5_3 i
  ⟨w, parallel_generatedRoot_5.facts _ h⟩

theorem generatedRoot.h6_3_at {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {ctx : generatedRoot.Ctx backend hashModel params} {outputs : (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat4 × mxxMat4 × mxxMat4 × mxxMat4 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × Unit}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 8) :
    ∃ w : parallel_generatedRoot_6.Witness, parallel_generatedRoot_6.Facts (⟨((generatedRoot.w_0_0 ctx), ((generatedRoot.w_1_0 ctx), ((generatedRoot.w_2_0 ctx), ()))), w⟩ : parallel_generatedRoot_6.Ctx backend hashModel params i) (((generatedRoot.w_6_0 ctx) i), (((generatedRoot.w_6_1 ctx) i), (((generatedRoot.w_6_2 ctx) i), (((generatedRoot.w_6_3 ctx) i), ())))) :=
  let ⟨w, h⟩ := F.c0.h6_3 i
  ⟨w, parallel_generatedRoot_6.facts _ h⟩

theorem generatedRoot.h7_3_at {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {ctx : generatedRoot.Ctx backend hashModel params} {outputs : (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat4 × mxxMat4 × mxxMat4 × mxxMat4 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × Unit}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 1) :
    ∃ w : parallel_generatedRoot_7.Witness, parallel_generatedRoot_7.Facts (⟨((generatedRoot.w_0_0 ctx), ((generatedRoot.w_1_0 ctx), ((generatedRoot.w_2_0 ctx), ()))), w⟩ : parallel_generatedRoot_7.Ctx backend hashModel params i) (((generatedRoot.w_7_0 ctx) i), (((generatedRoot.w_7_1 ctx) i), (((generatedRoot.w_7_2 ctx) i), (((generatedRoot.w_7_3 ctx) i), ())))) :=
  let ⟨w, h⟩ := F.c0.h7_3 i
  ⟨w, parallel_generatedRoot_7.facts _ h⟩

theorem generatedRoot.h8_3_at {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {ctx : generatedRoot.Ctx backend hashModel params} {outputs : (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat4 × mxxMat4 × mxxMat4 × mxxMat4 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × Unit}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 8) :
    ∃ w : parallel_generatedRoot_8.Witness, parallel_generatedRoot_8.Facts (⟨((generatedRoot.w_0_0 ctx), ((generatedRoot.w_1_0 ctx), ((generatedRoot.w_2_0 ctx), ()))), w⟩ : parallel_generatedRoot_8.Ctx backend hashModel params i) (((generatedRoot.w_8_0 ctx) i), (((generatedRoot.w_8_1 ctx) i), (((generatedRoot.w_8_2 ctx) i), (((generatedRoot.w_8_3 ctx) i), ())))) :=
  let ⟨w, h⟩ := F.c0.h8_3 i
  ⟨w, parallel_generatedRoot_8.facts _ h⟩

theorem generatedRoot.h9_3_at {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {ctx : generatedRoot.Ctx backend hashModel params} {outputs : (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat4 × mxxMat4 × mxxMat4 × mxxMat4 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × Unit}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 8) :
    ∃ w : parallel_generatedRoot_9.Witness, parallel_generatedRoot_9.Facts (⟨((generatedRoot.w_0_0 ctx), ((generatedRoot.w_1_0 ctx), ((generatedRoot.w_2_0 ctx), ()))), w⟩ : parallel_generatedRoot_9.Ctx backend hashModel params i) (((generatedRoot.w_9_0 ctx) i), (((generatedRoot.w_9_1 ctx) i), (((generatedRoot.w_9_2 ctx) i), (((generatedRoot.w_9_3 ctx) i), ())))) :=
  let ⟨w, h⟩ := F.c0.h9_3 i
  ⟨w, parallel_generatedRoot_9.facts _ h⟩

theorem generatedRoot.h10_3_at {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {ctx : generatedRoot.Ctx backend hashModel params} {outputs : (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat4 × mxxMat4 × mxxMat4 × mxxMat4 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × Unit}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 1) :
    ∃ w : parallel_generatedRoot_10.Witness, parallel_generatedRoot_10.Facts (⟨((generatedRoot.w_0_0 ctx), ((generatedRoot.w_1_0 ctx), ((generatedRoot.w_2_0 ctx), ()))), w⟩ : parallel_generatedRoot_10.Ctx backend hashModel params i) (((generatedRoot.w_10_0 ctx) i), (((generatedRoot.w_10_1 ctx) i), (((generatedRoot.w_10_2 ctx) i), (((generatedRoot.w_10_3 ctx) i), ())))) :=
  let ⟨w, h⟩ := F.c0.h10_3 i
  ⟨w, parallel_generatedRoot_10.facts _ h⟩

theorem generatedRoot.h11_3_at {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {ctx : generatedRoot.Ctx backend hashModel params} {outputs : (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat4 × mxxMat4 × mxxMat4 × mxxMat4 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × Unit}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 1) :
    ∃ w : parallel_generatedRoot_11.Witness, parallel_generatedRoot_11.Facts (⟨((generatedRoot.w_0_0 ctx), ((generatedRoot.w_1_0 ctx), ((generatedRoot.w_2_0 ctx), ()))), w⟩ : parallel_generatedRoot_11.Ctx backend hashModel params i) (((generatedRoot.w_11_0 ctx) i), (((generatedRoot.w_11_1 ctx) i), (((generatedRoot.w_11_2 ctx) i), (((generatedRoot.w_11_3 ctx) i), ())))) :=
  let ⟨w, h⟩ := F.c0.h11_3 i
  ⟨w, parallel_generatedRoot_11.facts _ h⟩

theorem generatedRoot.h12_3_at {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {ctx : generatedRoot.Ctx backend hashModel params} {outputs : (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat4 × mxxMat4 × mxxMat4 × mxxMat4 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × Unit}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 1) :
    ∃ w : parallel_generatedRoot_12.Witness, parallel_generatedRoot_12.Facts (⟨((generatedRoot.w_0_0 ctx), ((generatedRoot.w_1_0 ctx), ((generatedRoot.w_2_0 ctx), ()))), w⟩ : parallel_generatedRoot_12.Ctx backend hashModel params i) (((generatedRoot.w_12_0 ctx) i), (((generatedRoot.w_12_1 ctx) i), (((generatedRoot.w_12_2 ctx) i), (((generatedRoot.w_12_3 ctx) i), ())))) :=
  let ⟨w, h⟩ := F.c0.h12_3 i
  ⟨w, parallel_generatedRoot_12.facts _ h⟩

theorem generatedRoot.h13_3_at {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {ctx : generatedRoot.Ctx backend hashModel params} {outputs : (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat4 × mxxMat4 × mxxMat4 × mxxMat4 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × Unit}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 1) :
    ∃ w : parallel_generatedRoot_13.Witness, parallel_generatedRoot_13.Facts (⟨((generatedRoot.w_0_0 ctx), ((generatedRoot.w_1_0 ctx), ((generatedRoot.w_2_0 ctx), ()))), w⟩ : parallel_generatedRoot_13.Ctx backend hashModel params i) (((generatedRoot.w_13_0 ctx) i), (((generatedRoot.w_13_1 ctx) i), (((generatedRoot.w_13_2 ctx) i), (((generatedRoot.w_13_3 ctx) i), ())))) :=
  let ⟨w, h⟩ := F.c0.h13_3 i
  ⟨w, parallel_generatedRoot_13.facts _ h⟩

theorem generatedRoot.h14_3_at {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {ctx : generatedRoot.Ctx backend hashModel params} {outputs : (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat4 × mxxMat4 × mxxMat4 × mxxMat4 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × Unit}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 1) :
    ∃ w : parallel_generatedRoot_14.Witness, parallel_generatedRoot_14.Facts (⟨((generatedRoot.w_0_0 ctx), ((generatedRoot.w_1_0 ctx), ((generatedRoot.w_2_0 ctx), ()))), w⟩ : parallel_generatedRoot_14.Ctx backend hashModel params i) (((generatedRoot.w_14_0 ctx) i), (((generatedRoot.w_14_1 ctx) i), (((generatedRoot.w_14_2 ctx) i), (((generatedRoot.w_14_3 ctx) i), ())))) :=
  let ⟨w, h⟩ := F.c0.h14_3 i
  ⟨w, parallel_generatedRoot_14.facts _ h⟩

theorem generatedRoot.h15_3_at {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {ctx : generatedRoot.Ctx backend hashModel params} {outputs : (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat4 × mxxMat4 × mxxMat4 × mxxMat4 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × Unit}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 1) :
    ∃ w : parallel_generatedRoot_15.Witness, parallel_generatedRoot_15.Facts (⟨((generatedRoot.w_0_0 ctx), ((generatedRoot.w_1_0 ctx), ((generatedRoot.w_2_0 ctx), ()))), w⟩ : parallel_generatedRoot_15.Ctx backend hashModel params i) (((generatedRoot.w_15_0 ctx) i), (((generatedRoot.w_15_1 ctx) i), (((generatedRoot.w_15_2 ctx) i), (((generatedRoot.w_15_3 ctx) i), ())))) :=
  let ⟨w, h⟩ := F.c0.h15_3 i
  ⟨w, parallel_generatedRoot_15.facts _ h⟩

theorem generatedRoot.h80_3_at {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {ctx : generatedRoot.Ctx backend hashModel params} {outputs : (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat4 × mxxMat4 × mxxMat4 × mxxMat4 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × Unit}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 36000) :
    ∃ w : parallel_generatedRoot_80.Witness, parallel_generatedRoot_80.Facts (⟨((generatedRoot.w_79_0 ctx), ((generatedRoot.w_79_1 ctx), ((generatedRoot.w_79_2 ctx), ((generatedRoot.w_79_3 ctx), ())))), w⟩ : parallel_generatedRoot_80.Ctx backend hashModel params i) (((generatedRoot.w_80_0 ctx) i), (((generatedRoot.w_80_1 ctx) i), (((generatedRoot.w_80_2 ctx) i), (((generatedRoot.w_80_3 ctx) i), ())))) :=
  let ⟨w, h⟩ := F.c3.h80_3 i
  ⟨w, parallel_generatedRoot_80.facts _ h⟩

theorem generatedRoot.h84_3_at {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {ctx : generatedRoot.Ctx backend hashModel params} {outputs : (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat4 × mxxMat4 × mxxMat4 × mxxMat4 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × Unit}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 36000) :
    ∃ w : parallel_generatedRoot_84.Witness, parallel_generatedRoot_84.Facts (⟨((generatedRoot.w_83_0 ctx), ((generatedRoot.w_83_1 ctx), ((generatedRoot.w_83_2 ctx), ((generatedRoot.w_83_3 ctx), ())))), w⟩ : parallel_generatedRoot_84.Ctx backend hashModel params i) (((generatedRoot.w_84_0 ctx) i), (((generatedRoot.w_84_1 ctx) i), (((generatedRoot.w_84_2 ctx) i), (((generatedRoot.w_84_3 ctx) i), ())))) :=
  let ⟨w, h⟩ := F.c3.h84_3 i
  ⟨w, parallel_generatedRoot_84.facts _ h⟩

theorem generatedRoot.h85_3_at {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {ctx : generatedRoot.Ctx backend hashModel params} {outputs : (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat4 × mxxMat4 × mxxMat4 × mxxMat4 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × Unit}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 36000) :
    ∃ w : parallel_generatedRoot_85.Witness, parallel_generatedRoot_85.Facts (⟨((generatedRoot.w_17_0 ctx), (((generatedRoot.w_84_0 ctx) i), (((generatedRoot.w_84_2 ctx) i), ((generatedRoot.w_4_0 ctx), (((generatedRoot.w_84_1 ctx) i), (((generatedRoot.w_84_3 ctx) i), ((generatedRoot.w_4_2 ctx), ((generatedRoot.w_4_1 ctx), ((generatedRoot.w_4_3 ctx), ()))))))))), w⟩ : parallel_generatedRoot_85.Ctx backend hashModel params i) (((generatedRoot.w_85_0 ctx) i), (((generatedRoot.w_85_1 ctx) i), (((generatedRoot.w_85_2 ctx) i), (((generatedRoot.w_85_3 ctx) i), ())))) :=
  let ⟨w, h⟩ := F.c4.h85_3 i
  ⟨w, parallel_generatedRoot_85.facts _ h⟩

theorem generatedRoot.h117_3_at {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {ctx : generatedRoot.Ctx backend hashModel params} {outputs : (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat4 × mxxMat4 × mxxMat4 × mxxMat4 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × Unit}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 8) :
    ∃ w : parallel_generatedRoot_117.Witness, parallel_generatedRoot_117.Facts (⟨((generatedRoot.w_116_0 ctx), ((generatedRoot.w_116_1 ctx), ((generatedRoot.w_116_2 ctx), ((generatedRoot.w_116_3 ctx), ())))), w⟩ : parallel_generatedRoot_117.Ctx backend hashModel params i) (((generatedRoot.w_117_0 ctx) i), (((generatedRoot.w_117_1 ctx) i), (((generatedRoot.w_117_2 ctx) i), (((generatedRoot.w_117_3 ctx) i), ())))) :=
  let ⟨w, h⟩ := F.c5.h117_3 i
  ⟨w, parallel_generatedRoot_117.facts _ h⟩

theorem generatedRoot.h121_3_at {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {ctx : generatedRoot.Ctx backend hashModel params} {outputs : (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat4 × mxxMat4 × mxxMat4 × mxxMat4 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × Unit}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 8) :
    ∃ w : parallel_generatedRoot_121.Witness, parallel_generatedRoot_121.Facts (⟨((generatedRoot.w_120_0 ctx), ((generatedRoot.w_120_1 ctx), ((generatedRoot.w_120_2 ctx), ((generatedRoot.w_120_3 ctx), ())))), w⟩ : parallel_generatedRoot_121.Ctx backend hashModel params i) (((generatedRoot.w_121_0 ctx) i), (((generatedRoot.w_121_1 ctx) i), (((generatedRoot.w_121_2 ctx) i), (((generatedRoot.w_121_3 ctx) i), ())))) :=
  let ⟨w, h⟩ := F.c5.h121_3 i
  ⟨w, parallel_generatedRoot_121.facts _ h⟩

theorem generatedRoot.h122_3_at {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {ctx : generatedRoot.Ctx backend hashModel params} {outputs : (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat4 × mxxMat4 × mxxMat4 × mxxMat4 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × Unit}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 8) :
    ∃ w : parallel_generatedRoot_122.Witness, parallel_generatedRoot_122.Facts (⟨((generatedRoot.w_17_0 ctx), (((generatedRoot.w_121_0 ctx) i), (((generatedRoot.w_121_2 ctx) i), ((generatedRoot.w_6_0 ctx), (((generatedRoot.w_121_1 ctx) i), (((generatedRoot.w_121_3 ctx) i), ((generatedRoot.w_6_2 ctx), ((generatedRoot.w_6_1 ctx), ((generatedRoot.w_6_3 ctx), ()))))))))), w⟩ : parallel_generatedRoot_122.Ctx backend hashModel params i) (((generatedRoot.w_122_0 ctx) i), (((generatedRoot.w_122_1 ctx) i), (((generatedRoot.w_122_2 ctx) i), (((generatedRoot.w_122_3 ctx) i), ())))) :=
  let ⟨w, h⟩ := F.c5.h122_3 i
  ⟨w, parallel_generatedRoot_122.facts _ h⟩

theorem generatedRoot.h123_3_at {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {ctx : generatedRoot.Ctx backend hashModel params} {outputs : (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat4 × mxxMat4 × mxxMat4 × mxxMat4 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × Unit}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 8) :
    ∃ w : parallel_generatedRoot_123.Witness, parallel_generatedRoot_123.Facts (⟨((generatedRoot.w_17_0 ctx), (((generatedRoot.w_122_0 ctx) i), (((generatedRoot.w_122_2 ctx) i), ((generatedRoot.w_9_0 ctx), (((generatedRoot.w_122_1 ctx) i), (((generatedRoot.w_122_3 ctx) i), ((generatedRoot.w_9_2 ctx), ((generatedRoot.w_9_1 ctx), ((generatedRoot.w_9_3 ctx), ()))))))))), w⟩ : parallel_generatedRoot_123.Ctx backend hashModel params i) (((generatedRoot.w_123_0 ctx) i), (((generatedRoot.w_123_1 ctx) i), (((generatedRoot.w_123_2 ctx) i), (((generatedRoot.w_123_3 ctx) i), ())))) :=
  let ⟨w, h⟩ := F.c5.h123_3 i
  ⟨w, parallel_generatedRoot_123.facts _ h⟩

theorem generatedRoot.h162_3_at {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {ctx : generatedRoot.Ctx backend hashModel params} {outputs : (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat4 × mxxMat4 × mxxMat4 × mxxMat4 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × Unit}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 7200) :
    ∃ w : parallel_generatedRoot_162.Witness, parallel_generatedRoot_162.Facts (⟨((generatedRoot.w_161_0 ctx), ((generatedRoot.w_161_1 ctx), ((generatedRoot.w_161_2 ctx), ((generatedRoot.w_161_3 ctx), ())))), w⟩ : parallel_generatedRoot_162.Ctx backend hashModel params i) (((generatedRoot.w_162_0 ctx) i), (((generatedRoot.w_162_1 ctx) i), (((generatedRoot.w_162_2 ctx) i), (((generatedRoot.w_162_3 ctx) i), ())))) :=
  let ⟨w, h⟩ := F.c6.h162_3 i
  ⟨w, parallel_generatedRoot_162.facts _ h⟩

theorem generatedRoot.h166_3_at {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {ctx : generatedRoot.Ctx backend hashModel params} {outputs : (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat4 × mxxMat4 × mxxMat4 × mxxMat4 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × Unit}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 7200) :
    ∃ w : parallel_generatedRoot_166.Witness, parallel_generatedRoot_166.Facts (⟨((generatedRoot.w_165_0 ctx), ((generatedRoot.w_165_1 ctx), ((generatedRoot.w_165_2 ctx), ((generatedRoot.w_165_3 ctx), ())))), w⟩ : parallel_generatedRoot_166.Ctx backend hashModel params i) (((generatedRoot.w_166_0 ctx) i), (((generatedRoot.w_166_1 ctx) i), (((generatedRoot.w_166_2 ctx) i), (((generatedRoot.w_166_3 ctx) i), ())))) :=
  let ⟨w, h⟩ := F.c6.h166_3 i
  ⟨w, parallel_generatedRoot_166.facts _ h⟩

theorem generatedRoot.h167_3_at {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {ctx : generatedRoot.Ctx backend hashModel params} {outputs : (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat4 × mxxMat4 × mxxMat4 × mxxMat4 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × Unit}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 7200) :
    ∃ w : parallel_generatedRoot_167.Witness, parallel_generatedRoot_167.Facts (⟨((generatedRoot.w_17_0 ctx), (((generatedRoot.w_166_0 ctx) i), (((generatedRoot.w_166_2 ctx) i), ((generatedRoot.w_5_0 ctx), (((generatedRoot.w_166_1 ctx) i), (((generatedRoot.w_166_3 ctx) i), ((generatedRoot.w_5_2 ctx), ((generatedRoot.w_5_1 ctx), ((generatedRoot.w_5_3 ctx), ()))))))))), w⟩ : parallel_generatedRoot_167.Ctx backend hashModel params i) (((generatedRoot.w_167_0 ctx) i), (((generatedRoot.w_167_1 ctx) i), (((generatedRoot.w_167_2 ctx) i), (((generatedRoot.w_167_3 ctx) i), ())))) :=
  let ⟨w, h⟩ := F.c6.h167_3 i
  ⟨w, parallel_generatedRoot_167.facts _ h⟩

end Stage_preprocess_0_0

end
