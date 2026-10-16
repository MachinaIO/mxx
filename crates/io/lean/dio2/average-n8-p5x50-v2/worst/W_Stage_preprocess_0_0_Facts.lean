import W_Stage_preprocess_0_0

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000
set_option linter.unusedVariables false

-- The wire types repeat the ring modulus `X ^ n + 1`; unfolding it while checking a
-- projection against its field type expands it at the ring dimension.
attribute [local irreducible] Mxx.Primitives.negacyclicModulus

noncomputable section

namespace W_Stage_preprocess_0_0

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

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_3Ctx" => (parallel_generatedRoot_3.Ctx backend hashModel params i_0)

def parallel_generatedRoot_3.w_0_0  (ctx : parallel_generatedRoot_3Ctx) : ByteArray :=
  ctx.inputs.1

def parallel_generatedRoot_3.w_3_0  (ctx : parallel_generatedRoot_3Ctx) : mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_3.w_26_0  (ctx : parallel_generatedRoot_3Ctx) : mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_3.w_1_0  (ctx : parallel_generatedRoot_3Ctx) : mxxMat1 :=
  ctx.witness.w_1_0

def parallel_generatedRoot_3.w_2_0  (ctx : parallel_generatedRoot_3Ctx) : mxxMat1 :=
  ctx.witness.w_2_0

def parallel_generatedRoot_3.w_4_0  (ctx : parallel_generatedRoot_3Ctx) : mxxMat0 :=
  ctx.witness.w_4_0

def parallel_generatedRoot_3.w_5_0  (ctx : parallel_generatedRoot_3Ctx) : mxxMat0 :=
  MxxRuntime.matrixNeg (parallel_generatedRoot_3.w_4_0 ctx)

def parallel_generatedRoot_3.w_6_0  (ctx : parallel_generatedRoot_3Ctx) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_3.w_5_0 ctx) (parallel_generatedRoot_3.w_1_0 ctx)

def parallel_generatedRoot_3.w_7_0  (ctx : parallel_generatedRoot_3Ctx) : mxxMat0 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat0)

def parallel_generatedRoot_3.w_8_0  (ctx : parallel_generatedRoot_3Ctx) : mxxMat0 :=
  (MxxRuntime.matrixPolynomial [(mxxNat0)] : mxxMat0)

def parallel_generatedRoot_3.w_9_0  (ctx : parallel_generatedRoot_3Ctx) : mxxMat0 :=
  (0 : mxxMat0)

def parallel_generatedRoot_3.w_10_0  (ctx : parallel_generatedRoot_3Ctx) : mxxMat0 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat0)

def parallel_generatedRoot_3.w_11_0  (ctx : parallel_generatedRoot_3Ctx) : Int :=
  -2

def parallel_generatedRoot_3.w_12_0  (ctx : parallel_generatedRoot_3Ctx) : Fin 4 → Int :=
  ctx.witness.w_12_0

def parallel_generatedRoot_3.w_13_0  (ctx : parallel_generatedRoot_3Ctx) : mxxMat0 :=
  ctx.witness.w_13_0

def parallel_generatedRoot_3.w_14_0  (ctx : parallel_generatedRoot_3Ctx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_3.w_8_0 ctx) (parallel_generatedRoot_3.w_13_0 ctx)

def parallel_generatedRoot_3.w_15_0  (ctx : parallel_generatedRoot_3Ctx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_3.w_7_0 ctx) (parallel_generatedRoot_3.w_14_0 ctx)

def parallel_generatedRoot_3.w_16_0  (ctx : parallel_generatedRoot_3Ctx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_3.w_15_0 ctx) (parallel_generatedRoot_3.w_3_0 ctx)

def parallel_generatedRoot_3.w_17_0  (ctx : parallel_generatedRoot_3Ctx) : mxxMat1 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_3.w_18_0  (ctx : parallel_generatedRoot_3Ctx) : mxxMat1 :=
  ctx.witness.w_18_0

def parallel_generatedRoot_3.w_19_0  (ctx : parallel_generatedRoot_3Ctx) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_3.w_16_0 ctx) (parallel_generatedRoot_3.w_18_0 ctx)

def parallel_generatedRoot_3.w_20_0  (ctx : parallel_generatedRoot_3Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_3.w_6_0 ctx) (parallel_generatedRoot_3.w_19_0 ctx)

def parallel_generatedRoot_3.w_21_0  (ctx : parallel_generatedRoot_3Ctx) : mxxMat1 :=
  (0 : mxxMat1)

def parallel_generatedRoot_3.w_22_0  (ctx : parallel_generatedRoot_3Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_3.w_20_0 ctx) (parallel_generatedRoot_3.w_21_0 ctx)

def parallel_generatedRoot_3.w_23_0  (ctx : parallel_generatedRoot_3Ctx) : mxxMat0 :=
  MxxRuntime.matrixNeg (parallel_generatedRoot_3.w_4_0 ctx)

def parallel_generatedRoot_3.w_24_0  (ctx : parallel_generatedRoot_3Ctx) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_3.w_23_0 ctx) (parallel_generatedRoot_3.w_2_0 ctx)

def parallel_generatedRoot_3.w_25_0  (ctx : parallel_generatedRoot_3Ctx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_3.w_7_0 ctx) (parallel_generatedRoot_3.w_14_0 ctx)

def parallel_generatedRoot_3.w_27_0  (ctx : parallel_generatedRoot_3Ctx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_3.w_25_0 ctx) (parallel_generatedRoot_3.w_26_0 ctx)

def parallel_generatedRoot_3.w_28_0  (ctx : parallel_generatedRoot_3Ctx) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_3.w_27_0 ctx) (parallel_generatedRoot_3.w_18_0 ctx)

def parallel_generatedRoot_3.w_29_0  (ctx : parallel_generatedRoot_3Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_3.w_24_0 ctx) (parallel_generatedRoot_3.w_28_0 ctx)

def parallel_generatedRoot_3.w_30_0  (ctx : parallel_generatedRoot_3Ctx) : mxxMat1 :=
  (0 : mxxMat1)

def parallel_generatedRoot_3.w_31_0  (ctx : parallel_generatedRoot_3Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_3.w_29_0 ctx) (parallel_generatedRoot_3.w_30_0 ctx)

def parallel_generatedRoot_3.outs  (ctx : parallel_generatedRoot_3Ctx) : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit :=
  ((parallel_generatedRoot_3.w_1_0 ctx), ((parallel_generatedRoot_3.w_2_0 ctx), ((parallel_generatedRoot_3.w_22_0 ctx), ((parallel_generatedRoot_3.w_31_0 ctx), ()))))

structure parallel_generatedRoot_3.Facts_0  (ctx : parallel_generatedRoot_3Ctx) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
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

theorem parallel_generatedRoot_3.facts_0  {ctx : parallel_generatedRoot_3Ctx} {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_3.constraints_0 backend hashModel params i_0 (parallel_generatedRoot_3.w_0_0 ctx) (parallel_generatedRoot_3.w_3_0 ctx) (parallel_generatedRoot_3.w_1_0 ctx) (parallel_generatedRoot_3.w_2_0 ctx) (parallel_generatedRoot_3.w_4_0 ctx) (parallel_generatedRoot_3.w_9_0 ctx) (parallel_generatedRoot_3.w_10_0 ctx) (parallel_generatedRoot_3.w_11_0 ctx) (parallel_generatedRoot_3.w_12_0 ctx) (parallel_generatedRoot_3.w_13_0 ctx) (parallel_generatedRoot_3.w_17_0 ctx) (parallel_generatedRoot_3.w_18_0 ctx) (parallel_generatedRoot_3.w_22_0 ctx) (parallel_generatedRoot_3.w_31_0 ctx) outputs) :
    parallel_generatedRoot_3.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_3.Facts  (ctx : parallel_generatedRoot_3Ctx) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  c0 : parallel_generatedRoot_3.Facts_0 ctx outputs

theorem parallel_generatedRoot_3.facts  (ctx : parallel_generatedRoot_3Ctx) {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_3.body backend hashModel params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_3.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_3.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_4.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) where
  inputs : ByteArray × mxxMat0 × mxxMat0 × Unit
  witness : parallel_generatedRoot_4.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_4Ctx" => (parallel_generatedRoot_4.Ctx backend hashModel params i_0)

def parallel_generatedRoot_4.w_0_0  (ctx : parallel_generatedRoot_4Ctx) : ByteArray :=
  ctx.inputs.1

def parallel_generatedRoot_4.w_3_0  (ctx : parallel_generatedRoot_4Ctx) : mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_4.w_25_0  (ctx : parallel_generatedRoot_4Ctx) : mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_4.w_1_0  (ctx : parallel_generatedRoot_4Ctx) : mxxMat1 :=
  ctx.witness.w_1_0

def parallel_generatedRoot_4.w_2_0  (ctx : parallel_generatedRoot_4Ctx) : mxxMat1 :=
  ctx.witness.w_2_0

def parallel_generatedRoot_4.w_4_0  (ctx : parallel_generatedRoot_4Ctx) : mxxMat0 :=
  ctx.witness.w_4_0

def parallel_generatedRoot_4.w_5_0  (ctx : parallel_generatedRoot_4Ctx) : mxxMat0 :=
  MxxRuntime.matrixNeg (parallel_generatedRoot_4.w_4_0 ctx)

def parallel_generatedRoot_4.w_6_0  (ctx : parallel_generatedRoot_4Ctx) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_4.w_5_0 ctx) (parallel_generatedRoot_4.w_1_0 ctx)

def parallel_generatedRoot_4.w_7_0  (ctx : parallel_generatedRoot_4Ctx) : mxxMat0 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat0)

def parallel_generatedRoot_4.w_8_0  (ctx : parallel_generatedRoot_4Ctx) : mxxMat0 :=
  (MxxRuntime.matrixPolynomial [(mxxNat1)] : mxxMat0)

def parallel_generatedRoot_4.w_9_0  (ctx : parallel_generatedRoot_4Ctx) : mxxMat0 :=
  (0 : mxxMat0)

def parallel_generatedRoot_4.w_10_0  (ctx : parallel_generatedRoot_4Ctx) : Fin 8 → Int :=
  ctx.witness.w_10_0

def parallel_generatedRoot_4.w_11_0  (ctx : parallel_generatedRoot_4Ctx) : mxxMat0 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat0)

def parallel_generatedRoot_4.w_12_0  (ctx : parallel_generatedRoot_4Ctx) : Int :=
  -1

def parallel_generatedRoot_4.w_13_0  (ctx : parallel_generatedRoot_4Ctx) : mxxMat0 :=
  ctx.witness.w_13_0

def parallel_generatedRoot_4.w_14_0  (ctx : parallel_generatedRoot_4Ctx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_4.w_8_0 ctx) (parallel_generatedRoot_4.w_13_0 ctx)

def parallel_generatedRoot_4.w_15_0  (ctx : parallel_generatedRoot_4Ctx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_4.w_7_0 ctx) (parallel_generatedRoot_4.w_14_0 ctx)

def parallel_generatedRoot_4.w_16_0  (ctx : parallel_generatedRoot_4Ctx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_4.w_15_0 ctx) (parallel_generatedRoot_4.w_3_0 ctx)

def parallel_generatedRoot_4.w_17_0  (ctx : parallel_generatedRoot_4Ctx) : mxxMat1 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_4.w_18_0  (ctx : parallel_generatedRoot_4Ctx) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_4.w_16_0 ctx) (parallel_generatedRoot_4.w_17_0 ctx)

def parallel_generatedRoot_4.w_19_0  (ctx : parallel_generatedRoot_4Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_4.w_6_0 ctx) (parallel_generatedRoot_4.w_18_0 ctx)

def parallel_generatedRoot_4.w_20_0  (ctx : parallel_generatedRoot_4Ctx) : mxxMat1 :=
  (0 : mxxMat1)

def parallel_generatedRoot_4.w_21_0  (ctx : parallel_generatedRoot_4Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_4.w_19_0 ctx) (parallel_generatedRoot_4.w_20_0 ctx)

def parallel_generatedRoot_4.w_22_0  (ctx : parallel_generatedRoot_4Ctx) : mxxMat0 :=
  MxxRuntime.matrixNeg (parallel_generatedRoot_4.w_4_0 ctx)

def parallel_generatedRoot_4.w_23_0  (ctx : parallel_generatedRoot_4Ctx) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_4.w_22_0 ctx) (parallel_generatedRoot_4.w_2_0 ctx)

def parallel_generatedRoot_4.w_24_0  (ctx : parallel_generatedRoot_4Ctx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_4.w_7_0 ctx) (parallel_generatedRoot_4.w_14_0 ctx)

def parallel_generatedRoot_4.w_26_0  (ctx : parallel_generatedRoot_4Ctx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_4.w_24_0 ctx) (parallel_generatedRoot_4.w_25_0 ctx)

def parallel_generatedRoot_4.w_27_0  (ctx : parallel_generatedRoot_4Ctx) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_4.w_26_0 ctx) (parallel_generatedRoot_4.w_17_0 ctx)

def parallel_generatedRoot_4.w_28_0  (ctx : parallel_generatedRoot_4Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_4.w_23_0 ctx) (parallel_generatedRoot_4.w_27_0 ctx)

def parallel_generatedRoot_4.w_29_0  (ctx : parallel_generatedRoot_4Ctx) : mxxMat1 :=
  (0 : mxxMat1)

def parallel_generatedRoot_4.w_30_0  (ctx : parallel_generatedRoot_4Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_4.w_28_0 ctx) (parallel_generatedRoot_4.w_29_0 ctx)

def parallel_generatedRoot_4.outs  (ctx : parallel_generatedRoot_4Ctx) : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit :=
  ((parallel_generatedRoot_4.w_1_0 ctx), ((parallel_generatedRoot_4.w_2_0 ctx), ((parallel_generatedRoot_4.w_21_0 ctx), ((parallel_generatedRoot_4.w_30_0 ctx), ()))))

structure parallel_generatedRoot_4.Facts_0  (ctx : parallel_generatedRoot_4Ctx) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
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

theorem parallel_generatedRoot_4.facts_0  {ctx : parallel_generatedRoot_4Ctx} {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_4.constraints_0 backend hashModel params i_0 (parallel_generatedRoot_4.w_0_0 ctx) (parallel_generatedRoot_4.w_3_0 ctx) (parallel_generatedRoot_4.w_1_0 ctx) (parallel_generatedRoot_4.w_2_0 ctx) (parallel_generatedRoot_4.w_4_0 ctx) (parallel_generatedRoot_4.w_9_0 ctx) (parallel_generatedRoot_4.w_10_0 ctx) (parallel_generatedRoot_4.w_11_0 ctx) (parallel_generatedRoot_4.w_12_0 ctx) (parallel_generatedRoot_4.w_13_0 ctx) (parallel_generatedRoot_4.w_17_0 ctx) (parallel_generatedRoot_4.w_21_0 ctx) (parallel_generatedRoot_4.w_30_0 ctx) outputs) :
    parallel_generatedRoot_4.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_4.Facts  (ctx : parallel_generatedRoot_4Ctx) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  c0 : parallel_generatedRoot_4.Facts_0 ctx outputs

theorem parallel_generatedRoot_4.facts  (ctx : parallel_generatedRoot_4Ctx) {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_4.body backend hashModel params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_4.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_4.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_5.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) where
  inputs : ByteArray × mxxMat0 × mxxMat0 × Unit
  witness : parallel_generatedRoot_5.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_5Ctx" => (parallel_generatedRoot_5.Ctx backend hashModel params i_0)

def parallel_generatedRoot_5.w_0_0  (ctx : parallel_generatedRoot_5Ctx) : ByteArray :=
  ctx.inputs.1

def parallel_generatedRoot_5.w_3_0  (ctx : parallel_generatedRoot_5Ctx) : mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_5.w_26_0  (ctx : parallel_generatedRoot_5Ctx) : mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_5.w_1_0  (ctx : parallel_generatedRoot_5Ctx) : mxxMat1 :=
  ctx.witness.w_1_0

def parallel_generatedRoot_5.w_2_0  (ctx : parallel_generatedRoot_5Ctx) : mxxMat1 :=
  ctx.witness.w_2_0

def parallel_generatedRoot_5.w_4_0  (ctx : parallel_generatedRoot_5Ctx) : mxxMat0 :=
  ctx.witness.w_4_0

def parallel_generatedRoot_5.w_5_0  (ctx : parallel_generatedRoot_5Ctx) : mxxMat0 :=
  MxxRuntime.matrixNeg (parallel_generatedRoot_5.w_4_0 ctx)

def parallel_generatedRoot_5.w_6_0  (ctx : parallel_generatedRoot_5Ctx) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_5.w_5_0 ctx) (parallel_generatedRoot_5.w_1_0 ctx)

def parallel_generatedRoot_5.w_7_0  (ctx : parallel_generatedRoot_5Ctx) : mxxMat0 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat0)

def parallel_generatedRoot_5.w_8_0  (ctx : parallel_generatedRoot_5Ctx) : mxxMat0 :=
  (MxxRuntime.matrixPolynomial [(mxxNat1)] : mxxMat0)

def parallel_generatedRoot_5.w_9_0  (ctx : parallel_generatedRoot_5Ctx) : mxxMat0 :=
  (0 : mxxMat0)

def parallel_generatedRoot_5.w_10_0  (ctx : parallel_generatedRoot_5Ctx) : mxxMat0 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat0)

def parallel_generatedRoot_5.w_11_0  (ctx : parallel_generatedRoot_5Ctx) : Int :=
  -1

def parallel_generatedRoot_5.w_12_0  (ctx : parallel_generatedRoot_5Ctx) : Fin 8 → Int :=
  ctx.witness.w_12_0

def parallel_generatedRoot_5.w_13_0  (ctx : parallel_generatedRoot_5Ctx) : mxxMat0 :=
  ctx.witness.w_13_0

def parallel_generatedRoot_5.w_14_0  (ctx : parallel_generatedRoot_5Ctx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_5.w_8_0 ctx) (parallel_generatedRoot_5.w_13_0 ctx)

def parallel_generatedRoot_5.w_15_0  (ctx : parallel_generatedRoot_5Ctx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_5.w_7_0 ctx) (parallel_generatedRoot_5.w_14_0 ctx)

def parallel_generatedRoot_5.w_16_0  (ctx : parallel_generatedRoot_5Ctx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_5.w_15_0 ctx) (parallel_generatedRoot_5.w_3_0 ctx)

def parallel_generatedRoot_5.w_17_0  (ctx : parallel_generatedRoot_5Ctx) : mxxMat1 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_5.w_18_0  (ctx : parallel_generatedRoot_5Ctx) : mxxMat1 :=
  ctx.witness.w_18_0

def parallel_generatedRoot_5.w_19_0  (ctx : parallel_generatedRoot_5Ctx) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_5.w_16_0 ctx) (parallel_generatedRoot_5.w_18_0 ctx)

def parallel_generatedRoot_5.w_20_0  (ctx : parallel_generatedRoot_5Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_5.w_6_0 ctx) (parallel_generatedRoot_5.w_19_0 ctx)

def parallel_generatedRoot_5.w_21_0  (ctx : parallel_generatedRoot_5Ctx) : mxxMat1 :=
  (0 : mxxMat1)

def parallel_generatedRoot_5.w_22_0  (ctx : parallel_generatedRoot_5Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_5.w_20_0 ctx) (parallel_generatedRoot_5.w_21_0 ctx)

def parallel_generatedRoot_5.w_23_0  (ctx : parallel_generatedRoot_5Ctx) : mxxMat0 :=
  MxxRuntime.matrixNeg (parallel_generatedRoot_5.w_4_0 ctx)

def parallel_generatedRoot_5.w_24_0  (ctx : parallel_generatedRoot_5Ctx) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_5.w_23_0 ctx) (parallel_generatedRoot_5.w_2_0 ctx)

def parallel_generatedRoot_5.w_25_0  (ctx : parallel_generatedRoot_5Ctx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_5.w_7_0 ctx) (parallel_generatedRoot_5.w_14_0 ctx)

def parallel_generatedRoot_5.w_27_0  (ctx : parallel_generatedRoot_5Ctx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_5.w_25_0 ctx) (parallel_generatedRoot_5.w_26_0 ctx)

def parallel_generatedRoot_5.w_28_0  (ctx : parallel_generatedRoot_5Ctx) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_5.w_27_0 ctx) (parallel_generatedRoot_5.w_18_0 ctx)

def parallel_generatedRoot_5.w_29_0  (ctx : parallel_generatedRoot_5Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_5.w_24_0 ctx) (parallel_generatedRoot_5.w_28_0 ctx)

def parallel_generatedRoot_5.w_30_0  (ctx : parallel_generatedRoot_5Ctx) : mxxMat1 :=
  (0 : mxxMat1)

def parallel_generatedRoot_5.w_31_0  (ctx : parallel_generatedRoot_5Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_5.w_29_0 ctx) (parallel_generatedRoot_5.w_30_0 ctx)

def parallel_generatedRoot_5.outs  (ctx : parallel_generatedRoot_5Ctx) : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit :=
  ((parallel_generatedRoot_5.w_1_0 ctx), ((parallel_generatedRoot_5.w_2_0 ctx), ((parallel_generatedRoot_5.w_22_0 ctx), ((parallel_generatedRoot_5.w_31_0 ctx), ()))))

structure parallel_generatedRoot_5.Facts_0  (ctx : parallel_generatedRoot_5Ctx) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
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

theorem parallel_generatedRoot_5.facts_0  {ctx : parallel_generatedRoot_5Ctx} {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_5.constraints_0 backend hashModel params i_0 (parallel_generatedRoot_5.w_0_0 ctx) (parallel_generatedRoot_5.w_3_0 ctx) (parallel_generatedRoot_5.w_1_0 ctx) (parallel_generatedRoot_5.w_2_0 ctx) (parallel_generatedRoot_5.w_4_0 ctx) (parallel_generatedRoot_5.w_9_0 ctx) (parallel_generatedRoot_5.w_10_0 ctx) (parallel_generatedRoot_5.w_11_0 ctx) (parallel_generatedRoot_5.w_12_0 ctx) (parallel_generatedRoot_5.w_13_0 ctx) (parallel_generatedRoot_5.w_17_0 ctx) (parallel_generatedRoot_5.w_18_0 ctx) (parallel_generatedRoot_5.w_22_0 ctx) (parallel_generatedRoot_5.w_31_0 ctx) outputs) :
    parallel_generatedRoot_5.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_5.Facts  (ctx : parallel_generatedRoot_5Ctx) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  c0 : parallel_generatedRoot_5.Facts_0 ctx outputs

theorem parallel_generatedRoot_5.facts  (ctx : parallel_generatedRoot_5Ctx) {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_5.body backend hashModel params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_5.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_5.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_6.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) where
  inputs : ByteArray × mxxMat0 × mxxMat0 × Unit
  witness : parallel_generatedRoot_6.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_6Ctx" => (parallel_generatedRoot_6.Ctx backend hashModel params i_0)

def parallel_generatedRoot_6.w_0_0  (ctx : parallel_generatedRoot_6Ctx) : ByteArray :=
  ctx.inputs.1

def parallel_generatedRoot_6.w_3_0  (ctx : parallel_generatedRoot_6Ctx) : mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_6.w_26_0  (ctx : parallel_generatedRoot_6Ctx) : mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_6.w_1_0  (ctx : parallel_generatedRoot_6Ctx) : mxxMat1 :=
  ctx.witness.w_1_0

def parallel_generatedRoot_6.w_2_0  (ctx : parallel_generatedRoot_6Ctx) : mxxMat1 :=
  ctx.witness.w_2_0

def parallel_generatedRoot_6.w_4_0  (ctx : parallel_generatedRoot_6Ctx) : mxxMat0 :=
  ctx.witness.w_4_0

def parallel_generatedRoot_6.w_5_0  (ctx : parallel_generatedRoot_6Ctx) : mxxMat0 :=
  MxxRuntime.matrixNeg (parallel_generatedRoot_6.w_4_0 ctx)

def parallel_generatedRoot_6.w_6_0  (ctx : parallel_generatedRoot_6Ctx) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_6.w_5_0 ctx) (parallel_generatedRoot_6.w_1_0 ctx)

def parallel_generatedRoot_6.w_7_0  (ctx : parallel_generatedRoot_6Ctx) : mxxMat0 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat0)

def parallel_generatedRoot_6.w_8_0  (ctx : parallel_generatedRoot_6Ctx) : mxxMat0 :=
  (MxxRuntime.matrixPolynomial [(mxxNat1)] : mxxMat0)

def parallel_generatedRoot_6.w_9_0  (ctx : parallel_generatedRoot_6Ctx) : mxxMat0 :=
  (0 : mxxMat0)

def parallel_generatedRoot_6.w_10_0  (ctx : parallel_generatedRoot_6Ctx) : mxxMat0 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat0)

def parallel_generatedRoot_6.w_11_0  (ctx : parallel_generatedRoot_6Ctx) : Int :=
  -1

def parallel_generatedRoot_6.w_12_0  (ctx : parallel_generatedRoot_6Ctx) : Fin 8 → Int :=
  ctx.witness.w_12_0

def parallel_generatedRoot_6.w_13_0  (ctx : parallel_generatedRoot_6Ctx) : mxxMat0 :=
  ctx.witness.w_13_0

def parallel_generatedRoot_6.w_14_0  (ctx : parallel_generatedRoot_6Ctx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_6.w_8_0 ctx) (parallel_generatedRoot_6.w_13_0 ctx)

def parallel_generatedRoot_6.w_15_0  (ctx : parallel_generatedRoot_6Ctx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_6.w_7_0 ctx) (parallel_generatedRoot_6.w_14_0 ctx)

def parallel_generatedRoot_6.w_16_0  (ctx : parallel_generatedRoot_6Ctx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_6.w_15_0 ctx) (parallel_generatedRoot_6.w_3_0 ctx)

def parallel_generatedRoot_6.w_17_0  (ctx : parallel_generatedRoot_6Ctx) : mxxMat1 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_6.w_18_0  (ctx : parallel_generatedRoot_6Ctx) : mxxMat1 :=
  ctx.witness.w_18_0

def parallel_generatedRoot_6.w_19_0  (ctx : parallel_generatedRoot_6Ctx) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_6.w_16_0 ctx) (parallel_generatedRoot_6.w_18_0 ctx)

def parallel_generatedRoot_6.w_20_0  (ctx : parallel_generatedRoot_6Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_6.w_6_0 ctx) (parallel_generatedRoot_6.w_19_0 ctx)

def parallel_generatedRoot_6.w_21_0  (ctx : parallel_generatedRoot_6Ctx) : mxxMat1 :=
  (0 : mxxMat1)

def parallel_generatedRoot_6.w_22_0  (ctx : parallel_generatedRoot_6Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_6.w_20_0 ctx) (parallel_generatedRoot_6.w_21_0 ctx)

def parallel_generatedRoot_6.w_23_0  (ctx : parallel_generatedRoot_6Ctx) : mxxMat0 :=
  MxxRuntime.matrixNeg (parallel_generatedRoot_6.w_4_0 ctx)

def parallel_generatedRoot_6.w_24_0  (ctx : parallel_generatedRoot_6Ctx) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_6.w_23_0 ctx) (parallel_generatedRoot_6.w_2_0 ctx)

def parallel_generatedRoot_6.w_25_0  (ctx : parallel_generatedRoot_6Ctx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_6.w_7_0 ctx) (parallel_generatedRoot_6.w_14_0 ctx)

def parallel_generatedRoot_6.w_27_0  (ctx : parallel_generatedRoot_6Ctx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_6.w_25_0 ctx) (parallel_generatedRoot_6.w_26_0 ctx)

def parallel_generatedRoot_6.w_28_0  (ctx : parallel_generatedRoot_6Ctx) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_6.w_27_0 ctx) (parallel_generatedRoot_6.w_18_0 ctx)

def parallel_generatedRoot_6.w_29_0  (ctx : parallel_generatedRoot_6Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_6.w_24_0 ctx) (parallel_generatedRoot_6.w_28_0 ctx)

def parallel_generatedRoot_6.w_30_0  (ctx : parallel_generatedRoot_6Ctx) : mxxMat1 :=
  (0 : mxxMat1)

def parallel_generatedRoot_6.w_31_0  (ctx : parallel_generatedRoot_6Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_6.w_29_0 ctx) (parallel_generatedRoot_6.w_30_0 ctx)

def parallel_generatedRoot_6.outs  (ctx : parallel_generatedRoot_6Ctx) : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit :=
  ((parallel_generatedRoot_6.w_1_0 ctx), ((parallel_generatedRoot_6.w_2_0 ctx), ((parallel_generatedRoot_6.w_22_0 ctx), ((parallel_generatedRoot_6.w_31_0 ctx), ()))))

structure parallel_generatedRoot_6.Facts_0  (ctx : parallel_generatedRoot_6Ctx) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
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

theorem parallel_generatedRoot_6.facts_0  {ctx : parallel_generatedRoot_6Ctx} {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_6.constraints_0 backend hashModel params i_0 (parallel_generatedRoot_6.w_0_0 ctx) (parallel_generatedRoot_6.w_3_0 ctx) (parallel_generatedRoot_6.w_1_0 ctx) (parallel_generatedRoot_6.w_2_0 ctx) (parallel_generatedRoot_6.w_4_0 ctx) (parallel_generatedRoot_6.w_9_0 ctx) (parallel_generatedRoot_6.w_10_0 ctx) (parallel_generatedRoot_6.w_11_0 ctx) (parallel_generatedRoot_6.w_12_0 ctx) (parallel_generatedRoot_6.w_13_0 ctx) (parallel_generatedRoot_6.w_17_0 ctx) (parallel_generatedRoot_6.w_18_0 ctx) (parallel_generatedRoot_6.w_22_0 ctx) (parallel_generatedRoot_6.w_31_0 ctx) outputs) :
    parallel_generatedRoot_6.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_6.Facts  (ctx : parallel_generatedRoot_6Ctx) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  c0 : parallel_generatedRoot_6.Facts_0 ctx outputs

theorem parallel_generatedRoot_6.facts  (ctx : parallel_generatedRoot_6Ctx) {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_6.body backend hashModel params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_6.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_6.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_7.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) where
  inputs : ByteArray × mxxMat0 × mxxMat0 × Unit
  witness : parallel_generatedRoot_7.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_7Ctx" => (parallel_generatedRoot_7.Ctx backend hashModel params i_0)

def parallel_generatedRoot_7.w_0_0  (ctx : parallel_generatedRoot_7Ctx) : ByteArray :=
  ctx.inputs.1

def parallel_generatedRoot_7.w_3_0  (ctx : parallel_generatedRoot_7Ctx) : mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_7.w_25_0  (ctx : parallel_generatedRoot_7Ctx) : mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_7.w_1_0  (ctx : parallel_generatedRoot_7Ctx) : mxxMat1 :=
  ctx.witness.w_1_0

def parallel_generatedRoot_7.w_2_0  (ctx : parallel_generatedRoot_7Ctx) : mxxMat1 :=
  ctx.witness.w_2_0

def parallel_generatedRoot_7.w_4_0  (ctx : parallel_generatedRoot_7Ctx) : mxxMat0 :=
  ctx.witness.w_4_0

def parallel_generatedRoot_7.w_5_0  (ctx : parallel_generatedRoot_7Ctx) : mxxMat0 :=
  MxxRuntime.matrixNeg (parallel_generatedRoot_7.w_4_0 ctx)

def parallel_generatedRoot_7.w_6_0  (ctx : parallel_generatedRoot_7Ctx) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_7.w_5_0 ctx) (parallel_generatedRoot_7.w_1_0 ctx)

def parallel_generatedRoot_7.w_7_0  (ctx : parallel_generatedRoot_7Ctx) : mxxMat0 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat0)

def parallel_generatedRoot_7.w_8_0  (ctx : parallel_generatedRoot_7Ctx) : mxxMat0 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat0)

def parallel_generatedRoot_7.w_9_0  (ctx : parallel_generatedRoot_7Ctx) : mxxMat0 :=
  (0 : mxxMat0)

def parallel_generatedRoot_7.w_10_0  (ctx : parallel_generatedRoot_7Ctx) : Fin 1 → Int :=
  ctx.witness.w_10_0

def parallel_generatedRoot_7.w_11_0  (ctx : parallel_generatedRoot_7Ctx) : mxxMat0 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat0)

def parallel_generatedRoot_7.w_12_0  (ctx : parallel_generatedRoot_7Ctx) : Int :=
  -1

def parallel_generatedRoot_7.w_13_0  (ctx : parallel_generatedRoot_7Ctx) : mxxMat0 :=
  ctx.witness.w_13_0

def parallel_generatedRoot_7.w_14_0  (ctx : parallel_generatedRoot_7Ctx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_7.w_8_0 ctx) (parallel_generatedRoot_7.w_13_0 ctx)

def parallel_generatedRoot_7.w_15_0  (ctx : parallel_generatedRoot_7Ctx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_7.w_7_0 ctx) (parallel_generatedRoot_7.w_14_0 ctx)

def parallel_generatedRoot_7.w_16_0  (ctx : parallel_generatedRoot_7Ctx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_7.w_15_0 ctx) (parallel_generatedRoot_7.w_3_0 ctx)

def parallel_generatedRoot_7.w_17_0  (ctx : parallel_generatedRoot_7Ctx) : mxxMat1 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_7.w_18_0  (ctx : parallel_generatedRoot_7Ctx) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_7.w_16_0 ctx) (parallel_generatedRoot_7.w_17_0 ctx)

def parallel_generatedRoot_7.w_19_0  (ctx : parallel_generatedRoot_7Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_7.w_6_0 ctx) (parallel_generatedRoot_7.w_18_0 ctx)

def parallel_generatedRoot_7.w_20_0  (ctx : parallel_generatedRoot_7Ctx) : mxxMat1 :=
  (0 : mxxMat1)

def parallel_generatedRoot_7.w_21_0  (ctx : parallel_generatedRoot_7Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_7.w_19_0 ctx) (parallel_generatedRoot_7.w_20_0 ctx)

def parallel_generatedRoot_7.w_22_0  (ctx : parallel_generatedRoot_7Ctx) : mxxMat0 :=
  MxxRuntime.matrixNeg (parallel_generatedRoot_7.w_4_0 ctx)

def parallel_generatedRoot_7.w_23_0  (ctx : parallel_generatedRoot_7Ctx) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_7.w_22_0 ctx) (parallel_generatedRoot_7.w_2_0 ctx)

def parallel_generatedRoot_7.w_24_0  (ctx : parallel_generatedRoot_7Ctx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_7.w_7_0 ctx) (parallel_generatedRoot_7.w_14_0 ctx)

def parallel_generatedRoot_7.w_26_0  (ctx : parallel_generatedRoot_7Ctx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_7.w_24_0 ctx) (parallel_generatedRoot_7.w_25_0 ctx)

def parallel_generatedRoot_7.w_27_0  (ctx : parallel_generatedRoot_7Ctx) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_7.w_26_0 ctx) (parallel_generatedRoot_7.w_17_0 ctx)

def parallel_generatedRoot_7.w_28_0  (ctx : parallel_generatedRoot_7Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_7.w_23_0 ctx) (parallel_generatedRoot_7.w_27_0 ctx)

def parallel_generatedRoot_7.w_29_0  (ctx : parallel_generatedRoot_7Ctx) : mxxMat1 :=
  (0 : mxxMat1)

def parallel_generatedRoot_7.w_30_0  (ctx : parallel_generatedRoot_7Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_7.w_28_0 ctx) (parallel_generatedRoot_7.w_29_0 ctx)

def parallel_generatedRoot_7.outs  (ctx : parallel_generatedRoot_7Ctx) : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit :=
  ((parallel_generatedRoot_7.w_1_0 ctx), ((parallel_generatedRoot_7.w_2_0 ctx), ((parallel_generatedRoot_7.w_21_0 ctx), ((parallel_generatedRoot_7.w_30_0 ctx), ()))))

structure parallel_generatedRoot_7.Facts_0  (ctx : parallel_generatedRoot_7Ctx) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
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

theorem parallel_generatedRoot_7.facts_0  {ctx : parallel_generatedRoot_7Ctx} {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_7.constraints_0 backend hashModel params i_0 (parallel_generatedRoot_7.w_0_0 ctx) (parallel_generatedRoot_7.w_3_0 ctx) (parallel_generatedRoot_7.w_1_0 ctx) (parallel_generatedRoot_7.w_2_0 ctx) (parallel_generatedRoot_7.w_4_0 ctx) (parallel_generatedRoot_7.w_9_0 ctx) (parallel_generatedRoot_7.w_10_0 ctx) (parallel_generatedRoot_7.w_11_0 ctx) (parallel_generatedRoot_7.w_12_0 ctx) (parallel_generatedRoot_7.w_13_0 ctx) (parallel_generatedRoot_7.w_17_0 ctx) (parallel_generatedRoot_7.w_21_0 ctx) (parallel_generatedRoot_7.w_30_0 ctx) outputs) :
    parallel_generatedRoot_7.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_7.Facts  (ctx : parallel_generatedRoot_7Ctx) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  c0 : parallel_generatedRoot_7.Facts_0 ctx outputs

theorem parallel_generatedRoot_7.facts  (ctx : parallel_generatedRoot_7Ctx) {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_7.body backend hashModel params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_7.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_7.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_8.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) where
  inputs : ByteArray × mxxMat0 × mxxMat0 × Unit
  witness : parallel_generatedRoot_8.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_8Ctx" => (parallel_generatedRoot_8.Ctx backend hashModel params i_0)

def parallel_generatedRoot_8.w_0_0  (ctx : parallel_generatedRoot_8Ctx) : ByteArray :=
  ctx.inputs.1

def parallel_generatedRoot_8.w_3_0  (ctx : parallel_generatedRoot_8Ctx) : mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_8.w_25_0  (ctx : parallel_generatedRoot_8Ctx) : mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_8.w_1_0  (ctx : parallel_generatedRoot_8Ctx) : mxxMat1 :=
  ctx.witness.w_1_0

def parallel_generatedRoot_8.w_2_0  (ctx : parallel_generatedRoot_8Ctx) : mxxMat1 :=
  ctx.witness.w_2_0

def parallel_generatedRoot_8.w_4_0  (ctx : parallel_generatedRoot_8Ctx) : mxxMat0 :=
  ctx.witness.w_4_0

def parallel_generatedRoot_8.w_5_0  (ctx : parallel_generatedRoot_8Ctx) : mxxMat0 :=
  MxxRuntime.matrixNeg (parallel_generatedRoot_8.w_4_0 ctx)

def parallel_generatedRoot_8.w_6_0  (ctx : parallel_generatedRoot_8Ctx) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_8.w_5_0 ctx) (parallel_generatedRoot_8.w_1_0 ctx)

def parallel_generatedRoot_8.w_7_0  (ctx : parallel_generatedRoot_8Ctx) : mxxMat0 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat0)

def parallel_generatedRoot_8.w_8_0  (ctx : parallel_generatedRoot_8Ctx) : mxxMat0 :=
  (MxxRuntime.matrixPolynomial [(mxxNat1)] : mxxMat0)

def parallel_generatedRoot_8.w_9_0  (ctx : parallel_generatedRoot_8Ctx) : mxxMat0 :=
  (0 : mxxMat0)

def parallel_generatedRoot_8.w_10_0  (ctx : parallel_generatedRoot_8Ctx) : Fin 8 → Int :=
  ctx.witness.w_10_0

def parallel_generatedRoot_8.w_11_0  (ctx : parallel_generatedRoot_8Ctx) : mxxMat0 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat0)

def parallel_generatedRoot_8.w_12_0  (ctx : parallel_generatedRoot_8Ctx) : Int :=
  -1

def parallel_generatedRoot_8.w_13_0  (ctx : parallel_generatedRoot_8Ctx) : mxxMat0 :=
  ctx.witness.w_13_0

def parallel_generatedRoot_8.w_14_0  (ctx : parallel_generatedRoot_8Ctx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_8.w_8_0 ctx) (parallel_generatedRoot_8.w_13_0 ctx)

def parallel_generatedRoot_8.w_15_0  (ctx : parallel_generatedRoot_8Ctx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_8.w_7_0 ctx) (parallel_generatedRoot_8.w_14_0 ctx)

def parallel_generatedRoot_8.w_16_0  (ctx : parallel_generatedRoot_8Ctx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_8.w_15_0 ctx) (parallel_generatedRoot_8.w_3_0 ctx)

def parallel_generatedRoot_8.w_17_0  (ctx : parallel_generatedRoot_8Ctx) : mxxMat1 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_8.w_18_0  (ctx : parallel_generatedRoot_8Ctx) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_8.w_16_0 ctx) (parallel_generatedRoot_8.w_17_0 ctx)

def parallel_generatedRoot_8.w_19_0  (ctx : parallel_generatedRoot_8Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_8.w_6_0 ctx) (parallel_generatedRoot_8.w_18_0 ctx)

def parallel_generatedRoot_8.w_20_0  (ctx : parallel_generatedRoot_8Ctx) : mxxMat1 :=
  (0 : mxxMat1)

def parallel_generatedRoot_8.w_21_0  (ctx : parallel_generatedRoot_8Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_8.w_19_0 ctx) (parallel_generatedRoot_8.w_20_0 ctx)

def parallel_generatedRoot_8.w_22_0  (ctx : parallel_generatedRoot_8Ctx) : mxxMat0 :=
  MxxRuntime.matrixNeg (parallel_generatedRoot_8.w_4_0 ctx)

def parallel_generatedRoot_8.w_23_0  (ctx : parallel_generatedRoot_8Ctx) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_8.w_22_0 ctx) (parallel_generatedRoot_8.w_2_0 ctx)

def parallel_generatedRoot_8.w_24_0  (ctx : parallel_generatedRoot_8Ctx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_8.w_7_0 ctx) (parallel_generatedRoot_8.w_14_0 ctx)

def parallel_generatedRoot_8.w_26_0  (ctx : parallel_generatedRoot_8Ctx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_8.w_24_0 ctx) (parallel_generatedRoot_8.w_25_0 ctx)

def parallel_generatedRoot_8.w_27_0  (ctx : parallel_generatedRoot_8Ctx) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_8.w_26_0 ctx) (parallel_generatedRoot_8.w_17_0 ctx)

def parallel_generatedRoot_8.w_28_0  (ctx : parallel_generatedRoot_8Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_8.w_23_0 ctx) (parallel_generatedRoot_8.w_27_0 ctx)

def parallel_generatedRoot_8.w_29_0  (ctx : parallel_generatedRoot_8Ctx) : mxxMat1 :=
  (0 : mxxMat1)

def parallel_generatedRoot_8.w_30_0  (ctx : parallel_generatedRoot_8Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_8.w_28_0 ctx) (parallel_generatedRoot_8.w_29_0 ctx)

def parallel_generatedRoot_8.outs  (ctx : parallel_generatedRoot_8Ctx) : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit :=
  ((parallel_generatedRoot_8.w_1_0 ctx), ((parallel_generatedRoot_8.w_2_0 ctx), ((parallel_generatedRoot_8.w_21_0 ctx), ((parallel_generatedRoot_8.w_30_0 ctx), ()))))

structure parallel_generatedRoot_8.Facts_0  (ctx : parallel_generatedRoot_8Ctx) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
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

theorem parallel_generatedRoot_8.facts_0  {ctx : parallel_generatedRoot_8Ctx} {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_8.constraints_0 backend hashModel params i_0 (parallel_generatedRoot_8.w_0_0 ctx) (parallel_generatedRoot_8.w_3_0 ctx) (parallel_generatedRoot_8.w_1_0 ctx) (parallel_generatedRoot_8.w_2_0 ctx) (parallel_generatedRoot_8.w_4_0 ctx) (parallel_generatedRoot_8.w_9_0 ctx) (parallel_generatedRoot_8.w_10_0 ctx) (parallel_generatedRoot_8.w_11_0 ctx) (parallel_generatedRoot_8.w_12_0 ctx) (parallel_generatedRoot_8.w_13_0 ctx) (parallel_generatedRoot_8.w_17_0 ctx) (parallel_generatedRoot_8.w_21_0 ctx) (parallel_generatedRoot_8.w_30_0 ctx) outputs) :
    parallel_generatedRoot_8.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_8.Facts  (ctx : parallel_generatedRoot_8Ctx) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  c0 : parallel_generatedRoot_8.Facts_0 ctx outputs

theorem parallel_generatedRoot_8.facts  (ctx : parallel_generatedRoot_8Ctx) {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_8.body backend hashModel params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_8.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_8.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_9.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) where
  inputs : ByteArray × mxxMat0 × mxxMat0 × Unit
  witness : parallel_generatedRoot_9.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_9Ctx" => (parallel_generatedRoot_9.Ctx backend hashModel params i_0)

def parallel_generatedRoot_9.w_0_0  (ctx : parallel_generatedRoot_9Ctx) : ByteArray :=
  ctx.inputs.1

def parallel_generatedRoot_9.w_3_0  (ctx : parallel_generatedRoot_9Ctx) : mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_9.w_25_0  (ctx : parallel_generatedRoot_9Ctx) : mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_9.w_1_0  (ctx : parallel_generatedRoot_9Ctx) : mxxMat1 :=
  ctx.witness.w_1_0

def parallel_generatedRoot_9.w_2_0  (ctx : parallel_generatedRoot_9Ctx) : mxxMat1 :=
  ctx.witness.w_2_0

def parallel_generatedRoot_9.w_4_0  (ctx : parallel_generatedRoot_9Ctx) : mxxMat0 :=
  ctx.witness.w_4_0

def parallel_generatedRoot_9.w_5_0  (ctx : parallel_generatedRoot_9Ctx) : mxxMat0 :=
  MxxRuntime.matrixNeg (parallel_generatedRoot_9.w_4_0 ctx)

def parallel_generatedRoot_9.w_6_0  (ctx : parallel_generatedRoot_9Ctx) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_9.w_5_0 ctx) (parallel_generatedRoot_9.w_1_0 ctx)

def parallel_generatedRoot_9.w_7_0  (ctx : parallel_generatedRoot_9Ctx) : mxxMat0 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat0)

def parallel_generatedRoot_9.w_8_0  (ctx : parallel_generatedRoot_9Ctx) : mxxMat0 :=
  (MxxRuntime.matrixPolynomial [(mxxNat1)] : mxxMat0)

def parallel_generatedRoot_9.w_9_0  (ctx : parallel_generatedRoot_9Ctx) : mxxMat0 :=
  (0 : mxxMat0)

def parallel_generatedRoot_9.w_10_0  (ctx : parallel_generatedRoot_9Ctx) : Fin 8 → Int :=
  ctx.witness.w_10_0

def parallel_generatedRoot_9.w_11_0  (ctx : parallel_generatedRoot_9Ctx) : mxxMat0 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat0)

def parallel_generatedRoot_9.w_12_0  (ctx : parallel_generatedRoot_9Ctx) : Int :=
  -1

def parallel_generatedRoot_9.w_13_0  (ctx : parallel_generatedRoot_9Ctx) : mxxMat0 :=
  ctx.witness.w_13_0

def parallel_generatedRoot_9.w_14_0  (ctx : parallel_generatedRoot_9Ctx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_9.w_8_0 ctx) (parallel_generatedRoot_9.w_13_0 ctx)

def parallel_generatedRoot_9.w_15_0  (ctx : parallel_generatedRoot_9Ctx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_9.w_7_0 ctx) (parallel_generatedRoot_9.w_14_0 ctx)

def parallel_generatedRoot_9.w_16_0  (ctx : parallel_generatedRoot_9Ctx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_9.w_15_0 ctx) (parallel_generatedRoot_9.w_3_0 ctx)

def parallel_generatedRoot_9.w_17_0  (ctx : parallel_generatedRoot_9Ctx) : mxxMat1 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_9.w_18_0  (ctx : parallel_generatedRoot_9Ctx) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_9.w_16_0 ctx) (parallel_generatedRoot_9.w_17_0 ctx)

def parallel_generatedRoot_9.w_19_0  (ctx : parallel_generatedRoot_9Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_9.w_6_0 ctx) (parallel_generatedRoot_9.w_18_0 ctx)

def parallel_generatedRoot_9.w_20_0  (ctx : parallel_generatedRoot_9Ctx) : mxxMat1 :=
  (0 : mxxMat1)

def parallel_generatedRoot_9.w_21_0  (ctx : parallel_generatedRoot_9Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_9.w_19_0 ctx) (parallel_generatedRoot_9.w_20_0 ctx)

def parallel_generatedRoot_9.w_22_0  (ctx : parallel_generatedRoot_9Ctx) : mxxMat0 :=
  MxxRuntime.matrixNeg (parallel_generatedRoot_9.w_4_0 ctx)

def parallel_generatedRoot_9.w_23_0  (ctx : parallel_generatedRoot_9Ctx) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_9.w_22_0 ctx) (parallel_generatedRoot_9.w_2_0 ctx)

def parallel_generatedRoot_9.w_24_0  (ctx : parallel_generatedRoot_9Ctx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_9.w_7_0 ctx) (parallel_generatedRoot_9.w_14_0 ctx)

def parallel_generatedRoot_9.w_26_0  (ctx : parallel_generatedRoot_9Ctx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_9.w_24_0 ctx) (parallel_generatedRoot_9.w_25_0 ctx)

def parallel_generatedRoot_9.w_27_0  (ctx : parallel_generatedRoot_9Ctx) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_9.w_26_0 ctx) (parallel_generatedRoot_9.w_17_0 ctx)

def parallel_generatedRoot_9.w_28_0  (ctx : parallel_generatedRoot_9Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_9.w_23_0 ctx) (parallel_generatedRoot_9.w_27_0 ctx)

def parallel_generatedRoot_9.w_29_0  (ctx : parallel_generatedRoot_9Ctx) : mxxMat1 :=
  (0 : mxxMat1)

def parallel_generatedRoot_9.w_30_0  (ctx : parallel_generatedRoot_9Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_9.w_28_0 ctx) (parallel_generatedRoot_9.w_29_0 ctx)

def parallel_generatedRoot_9.outs  (ctx : parallel_generatedRoot_9Ctx) : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit :=
  ((parallel_generatedRoot_9.w_1_0 ctx), ((parallel_generatedRoot_9.w_2_0 ctx), ((parallel_generatedRoot_9.w_21_0 ctx), ((parallel_generatedRoot_9.w_30_0 ctx), ()))))

structure parallel_generatedRoot_9.Facts_0  (ctx : parallel_generatedRoot_9Ctx) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
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

theorem parallel_generatedRoot_9.facts_0  {ctx : parallel_generatedRoot_9Ctx} {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_9.constraints_0 backend hashModel params i_0 (parallel_generatedRoot_9.w_0_0 ctx) (parallel_generatedRoot_9.w_3_0 ctx) (parallel_generatedRoot_9.w_1_0 ctx) (parallel_generatedRoot_9.w_2_0 ctx) (parallel_generatedRoot_9.w_4_0 ctx) (parallel_generatedRoot_9.w_9_0 ctx) (parallel_generatedRoot_9.w_10_0 ctx) (parallel_generatedRoot_9.w_11_0 ctx) (parallel_generatedRoot_9.w_12_0 ctx) (parallel_generatedRoot_9.w_13_0 ctx) (parallel_generatedRoot_9.w_17_0 ctx) (parallel_generatedRoot_9.w_21_0 ctx) (parallel_generatedRoot_9.w_30_0 ctx) outputs) :
    parallel_generatedRoot_9.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_9.Facts  (ctx : parallel_generatedRoot_9Ctx) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  c0 : parallel_generatedRoot_9.Facts_0 ctx outputs

theorem parallel_generatedRoot_9.facts  (ctx : parallel_generatedRoot_9Ctx) {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_9.body backend hashModel params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_9.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_9.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_10.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) where
  inputs : ByteArray × mxxMat0 × mxxMat0 × Unit
  witness : parallel_generatedRoot_10.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_10Ctx" => (parallel_generatedRoot_10.Ctx backend hashModel params i_0)

def parallel_generatedRoot_10.w_0_0  (ctx : parallel_generatedRoot_10Ctx) : ByteArray :=
  ctx.inputs.1

def parallel_generatedRoot_10.w_3_0  (ctx : parallel_generatedRoot_10Ctx) : mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_10.w_25_0  (ctx : parallel_generatedRoot_10Ctx) : mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_10.w_1_0  (ctx : parallel_generatedRoot_10Ctx) : mxxMat1 :=
  ctx.witness.w_1_0

def parallel_generatedRoot_10.w_2_0  (ctx : parallel_generatedRoot_10Ctx) : mxxMat1 :=
  ctx.witness.w_2_0

def parallel_generatedRoot_10.w_4_0  (ctx : parallel_generatedRoot_10Ctx) : mxxMat0 :=
  ctx.witness.w_4_0

def parallel_generatedRoot_10.w_5_0  (ctx : parallel_generatedRoot_10Ctx) : mxxMat0 :=
  MxxRuntime.matrixNeg (parallel_generatedRoot_10.w_4_0 ctx)

def parallel_generatedRoot_10.w_6_0  (ctx : parallel_generatedRoot_10Ctx) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_10.w_5_0 ctx) (parallel_generatedRoot_10.w_1_0 ctx)

def parallel_generatedRoot_10.w_7_0  (ctx : parallel_generatedRoot_10Ctx) : mxxMat0 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat0)

def parallel_generatedRoot_10.w_8_0  (ctx : parallel_generatedRoot_10Ctx) : mxxMat0 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat0)

def parallel_generatedRoot_10.w_9_0  (ctx : parallel_generatedRoot_10Ctx) : mxxMat0 :=
  (0 : mxxMat0)

def parallel_generatedRoot_10.w_10_0  (ctx : parallel_generatedRoot_10Ctx) : Fin 1 → Int :=
  ctx.witness.w_10_0

def parallel_generatedRoot_10.w_11_0  (ctx : parallel_generatedRoot_10Ctx) : mxxMat0 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat0)

def parallel_generatedRoot_10.w_12_0  (ctx : parallel_generatedRoot_10Ctx) : Int :=
  -1

def parallel_generatedRoot_10.w_13_0  (ctx : parallel_generatedRoot_10Ctx) : mxxMat0 :=
  ctx.witness.w_13_0

def parallel_generatedRoot_10.w_14_0  (ctx : parallel_generatedRoot_10Ctx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_10.w_8_0 ctx) (parallel_generatedRoot_10.w_13_0 ctx)

def parallel_generatedRoot_10.w_15_0  (ctx : parallel_generatedRoot_10Ctx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_10.w_7_0 ctx) (parallel_generatedRoot_10.w_14_0 ctx)

def parallel_generatedRoot_10.w_16_0  (ctx : parallel_generatedRoot_10Ctx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_10.w_15_0 ctx) (parallel_generatedRoot_10.w_3_0 ctx)

def parallel_generatedRoot_10.w_17_0  (ctx : parallel_generatedRoot_10Ctx) : mxxMat1 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_10.w_18_0  (ctx : parallel_generatedRoot_10Ctx) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_10.w_16_0 ctx) (parallel_generatedRoot_10.w_17_0 ctx)

def parallel_generatedRoot_10.w_19_0  (ctx : parallel_generatedRoot_10Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_10.w_6_0 ctx) (parallel_generatedRoot_10.w_18_0 ctx)

def parallel_generatedRoot_10.w_20_0  (ctx : parallel_generatedRoot_10Ctx) : mxxMat1 :=
  (0 : mxxMat1)

def parallel_generatedRoot_10.w_21_0  (ctx : parallel_generatedRoot_10Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_10.w_19_0 ctx) (parallel_generatedRoot_10.w_20_0 ctx)

def parallel_generatedRoot_10.w_22_0  (ctx : parallel_generatedRoot_10Ctx) : mxxMat0 :=
  MxxRuntime.matrixNeg (parallel_generatedRoot_10.w_4_0 ctx)

def parallel_generatedRoot_10.w_23_0  (ctx : parallel_generatedRoot_10Ctx) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_10.w_22_0 ctx) (parallel_generatedRoot_10.w_2_0 ctx)

def parallel_generatedRoot_10.w_24_0  (ctx : parallel_generatedRoot_10Ctx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_10.w_7_0 ctx) (parallel_generatedRoot_10.w_14_0 ctx)

def parallel_generatedRoot_10.w_26_0  (ctx : parallel_generatedRoot_10Ctx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_10.w_24_0 ctx) (parallel_generatedRoot_10.w_25_0 ctx)

def parallel_generatedRoot_10.w_27_0  (ctx : parallel_generatedRoot_10Ctx) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_10.w_26_0 ctx) (parallel_generatedRoot_10.w_17_0 ctx)

def parallel_generatedRoot_10.w_28_0  (ctx : parallel_generatedRoot_10Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_10.w_23_0 ctx) (parallel_generatedRoot_10.w_27_0 ctx)

def parallel_generatedRoot_10.w_29_0  (ctx : parallel_generatedRoot_10Ctx) : mxxMat1 :=
  (0 : mxxMat1)

def parallel_generatedRoot_10.w_30_0  (ctx : parallel_generatedRoot_10Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_10.w_28_0 ctx) (parallel_generatedRoot_10.w_29_0 ctx)

def parallel_generatedRoot_10.outs  (ctx : parallel_generatedRoot_10Ctx) : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit :=
  ((parallel_generatedRoot_10.w_1_0 ctx), ((parallel_generatedRoot_10.w_2_0 ctx), ((parallel_generatedRoot_10.w_21_0 ctx), ((parallel_generatedRoot_10.w_30_0 ctx), ()))))

structure parallel_generatedRoot_10.Facts_0  (ctx : parallel_generatedRoot_10Ctx) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
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

theorem parallel_generatedRoot_10.facts_0  {ctx : parallel_generatedRoot_10Ctx} {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_10.constraints_0 backend hashModel params i_0 (parallel_generatedRoot_10.w_0_0 ctx) (parallel_generatedRoot_10.w_3_0 ctx) (parallel_generatedRoot_10.w_1_0 ctx) (parallel_generatedRoot_10.w_2_0 ctx) (parallel_generatedRoot_10.w_4_0 ctx) (parallel_generatedRoot_10.w_9_0 ctx) (parallel_generatedRoot_10.w_10_0 ctx) (parallel_generatedRoot_10.w_11_0 ctx) (parallel_generatedRoot_10.w_12_0 ctx) (parallel_generatedRoot_10.w_13_0 ctx) (parallel_generatedRoot_10.w_17_0 ctx) (parallel_generatedRoot_10.w_21_0 ctx) (parallel_generatedRoot_10.w_30_0 ctx) outputs) :
    parallel_generatedRoot_10.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_10.Facts  (ctx : parallel_generatedRoot_10Ctx) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  c0 : parallel_generatedRoot_10.Facts_0 ctx outputs

theorem parallel_generatedRoot_10.facts  (ctx : parallel_generatedRoot_10Ctx) {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_10.body backend hashModel params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_10.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_10.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_11.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) where
  inputs : ByteArray × mxxMat0 × mxxMat0 × Unit
  witness : parallel_generatedRoot_11.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_11Ctx" => (parallel_generatedRoot_11.Ctx backend hashModel params i_0)

def parallel_generatedRoot_11.w_0_0  (ctx : parallel_generatedRoot_11Ctx) : ByteArray :=
  ctx.inputs.1

def parallel_generatedRoot_11.w_3_0  (ctx : parallel_generatedRoot_11Ctx) : mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_11.w_25_0  (ctx : parallel_generatedRoot_11Ctx) : mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_11.w_1_0  (ctx : parallel_generatedRoot_11Ctx) : mxxMat1 :=
  ctx.witness.w_1_0

def parallel_generatedRoot_11.w_2_0  (ctx : parallel_generatedRoot_11Ctx) : mxxMat1 :=
  ctx.witness.w_2_0

def parallel_generatedRoot_11.w_4_0  (ctx : parallel_generatedRoot_11Ctx) : mxxMat0 :=
  ctx.witness.w_4_0

def parallel_generatedRoot_11.w_5_0  (ctx : parallel_generatedRoot_11Ctx) : mxxMat0 :=
  MxxRuntime.matrixNeg (parallel_generatedRoot_11.w_4_0 ctx)

def parallel_generatedRoot_11.w_6_0  (ctx : parallel_generatedRoot_11Ctx) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_11.w_5_0 ctx) (parallel_generatedRoot_11.w_1_0 ctx)

def parallel_generatedRoot_11.w_7_0  (ctx : parallel_generatedRoot_11Ctx) : mxxMat0 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat0)

def parallel_generatedRoot_11.w_8_0  (ctx : parallel_generatedRoot_11Ctx) : mxxMat0 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat0)

def parallel_generatedRoot_11.w_9_0  (ctx : parallel_generatedRoot_11Ctx) : mxxMat0 :=
  (0 : mxxMat0)

def parallel_generatedRoot_11.w_10_0  (ctx : parallel_generatedRoot_11Ctx) : Fin 1 → Int :=
  ctx.witness.w_10_0

def parallel_generatedRoot_11.w_11_0  (ctx : parallel_generatedRoot_11Ctx) : mxxMat0 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat0)

def parallel_generatedRoot_11.w_12_0  (ctx : parallel_generatedRoot_11Ctx) : Int :=
  -1

def parallel_generatedRoot_11.w_13_0  (ctx : parallel_generatedRoot_11Ctx) : mxxMat0 :=
  ctx.witness.w_13_0

def parallel_generatedRoot_11.w_14_0  (ctx : parallel_generatedRoot_11Ctx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_11.w_8_0 ctx) (parallel_generatedRoot_11.w_13_0 ctx)

def parallel_generatedRoot_11.w_15_0  (ctx : parallel_generatedRoot_11Ctx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_11.w_7_0 ctx) (parallel_generatedRoot_11.w_14_0 ctx)

def parallel_generatedRoot_11.w_16_0  (ctx : parallel_generatedRoot_11Ctx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_11.w_15_0 ctx) (parallel_generatedRoot_11.w_3_0 ctx)

def parallel_generatedRoot_11.w_17_0  (ctx : parallel_generatedRoot_11Ctx) : mxxMat1 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_11.w_18_0  (ctx : parallel_generatedRoot_11Ctx) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_11.w_16_0 ctx) (parallel_generatedRoot_11.w_17_0 ctx)

def parallel_generatedRoot_11.w_19_0  (ctx : parallel_generatedRoot_11Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_11.w_6_0 ctx) (parallel_generatedRoot_11.w_18_0 ctx)

def parallel_generatedRoot_11.w_20_0  (ctx : parallel_generatedRoot_11Ctx) : mxxMat1 :=
  (0 : mxxMat1)

def parallel_generatedRoot_11.w_21_0  (ctx : parallel_generatedRoot_11Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_11.w_19_0 ctx) (parallel_generatedRoot_11.w_20_0 ctx)

def parallel_generatedRoot_11.w_22_0  (ctx : parallel_generatedRoot_11Ctx) : mxxMat0 :=
  MxxRuntime.matrixNeg (parallel_generatedRoot_11.w_4_0 ctx)

def parallel_generatedRoot_11.w_23_0  (ctx : parallel_generatedRoot_11Ctx) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_11.w_22_0 ctx) (parallel_generatedRoot_11.w_2_0 ctx)

def parallel_generatedRoot_11.w_24_0  (ctx : parallel_generatedRoot_11Ctx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_11.w_7_0 ctx) (parallel_generatedRoot_11.w_14_0 ctx)

def parallel_generatedRoot_11.w_26_0  (ctx : parallel_generatedRoot_11Ctx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_11.w_24_0 ctx) (parallel_generatedRoot_11.w_25_0 ctx)

def parallel_generatedRoot_11.w_27_0  (ctx : parallel_generatedRoot_11Ctx) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_11.w_26_0 ctx) (parallel_generatedRoot_11.w_17_0 ctx)

def parallel_generatedRoot_11.w_28_0  (ctx : parallel_generatedRoot_11Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_11.w_23_0 ctx) (parallel_generatedRoot_11.w_27_0 ctx)

def parallel_generatedRoot_11.w_29_0  (ctx : parallel_generatedRoot_11Ctx) : mxxMat1 :=
  (0 : mxxMat1)

def parallel_generatedRoot_11.w_30_0  (ctx : parallel_generatedRoot_11Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_11.w_28_0 ctx) (parallel_generatedRoot_11.w_29_0 ctx)

def parallel_generatedRoot_11.outs  (ctx : parallel_generatedRoot_11Ctx) : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit :=
  ((parallel_generatedRoot_11.w_1_0 ctx), ((parallel_generatedRoot_11.w_2_0 ctx), ((parallel_generatedRoot_11.w_21_0 ctx), ((parallel_generatedRoot_11.w_30_0 ctx), ()))))

structure parallel_generatedRoot_11.Facts_0  (ctx : parallel_generatedRoot_11Ctx) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
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

theorem parallel_generatedRoot_11.facts_0  {ctx : parallel_generatedRoot_11Ctx} {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_11.constraints_0 backend hashModel params i_0 (parallel_generatedRoot_11.w_0_0 ctx) (parallel_generatedRoot_11.w_3_0 ctx) (parallel_generatedRoot_11.w_1_0 ctx) (parallel_generatedRoot_11.w_2_0 ctx) (parallel_generatedRoot_11.w_4_0 ctx) (parallel_generatedRoot_11.w_9_0 ctx) (parallel_generatedRoot_11.w_10_0 ctx) (parallel_generatedRoot_11.w_11_0 ctx) (parallel_generatedRoot_11.w_12_0 ctx) (parallel_generatedRoot_11.w_13_0 ctx) (parallel_generatedRoot_11.w_17_0 ctx) (parallel_generatedRoot_11.w_21_0 ctx) (parallel_generatedRoot_11.w_30_0 ctx) outputs) :
    parallel_generatedRoot_11.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_11.Facts  (ctx : parallel_generatedRoot_11Ctx) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  c0 : parallel_generatedRoot_11.Facts_0 ctx outputs

theorem parallel_generatedRoot_11.facts  (ctx : parallel_generatedRoot_11Ctx) {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_11.body backend hashModel params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_11.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_11.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_12.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) where
  inputs : ByteArray × mxxMat0 × mxxMat0 × Unit
  witness : parallel_generatedRoot_12.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_12Ctx" => (parallel_generatedRoot_12.Ctx backend hashModel params i_0)

def parallel_generatedRoot_12.w_0_0  (ctx : parallel_generatedRoot_12Ctx) : ByteArray :=
  ctx.inputs.1

def parallel_generatedRoot_12.w_3_0  (ctx : parallel_generatedRoot_12Ctx) : mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_12.w_25_0  (ctx : parallel_generatedRoot_12Ctx) : mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_12.w_1_0  (ctx : parallel_generatedRoot_12Ctx) : mxxMat1 :=
  ctx.witness.w_1_0

def parallel_generatedRoot_12.w_2_0  (ctx : parallel_generatedRoot_12Ctx) : mxxMat1 :=
  ctx.witness.w_2_0

def parallel_generatedRoot_12.w_4_0  (ctx : parallel_generatedRoot_12Ctx) : mxxMat0 :=
  ctx.witness.w_4_0

def parallel_generatedRoot_12.w_5_0  (ctx : parallel_generatedRoot_12Ctx) : mxxMat0 :=
  MxxRuntime.matrixNeg (parallel_generatedRoot_12.w_4_0 ctx)

def parallel_generatedRoot_12.w_6_0  (ctx : parallel_generatedRoot_12Ctx) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_12.w_5_0 ctx) (parallel_generatedRoot_12.w_1_0 ctx)

def parallel_generatedRoot_12.w_7_0  (ctx : parallel_generatedRoot_12Ctx) : mxxMat0 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat0)

def parallel_generatedRoot_12.w_8_0  (ctx : parallel_generatedRoot_12Ctx) : mxxMat0 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat0)

def parallel_generatedRoot_12.w_9_0  (ctx : parallel_generatedRoot_12Ctx) : mxxMat0 :=
  (0 : mxxMat0)

def parallel_generatedRoot_12.w_10_0  (ctx : parallel_generatedRoot_12Ctx) : Fin 1 → Int :=
  ctx.witness.w_10_0

def parallel_generatedRoot_12.w_11_0  (ctx : parallel_generatedRoot_12Ctx) : mxxMat0 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat0)

def parallel_generatedRoot_12.w_12_0  (ctx : parallel_generatedRoot_12Ctx) : Int :=
  -1

def parallel_generatedRoot_12.w_13_0  (ctx : parallel_generatedRoot_12Ctx) : mxxMat0 :=
  ctx.witness.w_13_0

def parallel_generatedRoot_12.w_14_0  (ctx : parallel_generatedRoot_12Ctx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_12.w_8_0 ctx) (parallel_generatedRoot_12.w_13_0 ctx)

def parallel_generatedRoot_12.w_15_0  (ctx : parallel_generatedRoot_12Ctx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_12.w_7_0 ctx) (parallel_generatedRoot_12.w_14_0 ctx)

def parallel_generatedRoot_12.w_16_0  (ctx : parallel_generatedRoot_12Ctx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_12.w_15_0 ctx) (parallel_generatedRoot_12.w_3_0 ctx)

def parallel_generatedRoot_12.w_17_0  (ctx : parallel_generatedRoot_12Ctx) : mxxMat1 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_12.w_18_0  (ctx : parallel_generatedRoot_12Ctx) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_12.w_16_0 ctx) (parallel_generatedRoot_12.w_17_0 ctx)

def parallel_generatedRoot_12.w_19_0  (ctx : parallel_generatedRoot_12Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_12.w_6_0 ctx) (parallel_generatedRoot_12.w_18_0 ctx)

def parallel_generatedRoot_12.w_20_0  (ctx : parallel_generatedRoot_12Ctx) : mxxMat1 :=
  (0 : mxxMat1)

def parallel_generatedRoot_12.w_21_0  (ctx : parallel_generatedRoot_12Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_12.w_19_0 ctx) (parallel_generatedRoot_12.w_20_0 ctx)

def parallel_generatedRoot_12.w_22_0  (ctx : parallel_generatedRoot_12Ctx) : mxxMat0 :=
  MxxRuntime.matrixNeg (parallel_generatedRoot_12.w_4_0 ctx)

def parallel_generatedRoot_12.w_23_0  (ctx : parallel_generatedRoot_12Ctx) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_12.w_22_0 ctx) (parallel_generatedRoot_12.w_2_0 ctx)

def parallel_generatedRoot_12.w_24_0  (ctx : parallel_generatedRoot_12Ctx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_12.w_7_0 ctx) (parallel_generatedRoot_12.w_14_0 ctx)

def parallel_generatedRoot_12.w_26_0  (ctx : parallel_generatedRoot_12Ctx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_12.w_24_0 ctx) (parallel_generatedRoot_12.w_25_0 ctx)

def parallel_generatedRoot_12.w_27_0  (ctx : parallel_generatedRoot_12Ctx) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_12.w_26_0 ctx) (parallel_generatedRoot_12.w_17_0 ctx)

def parallel_generatedRoot_12.w_28_0  (ctx : parallel_generatedRoot_12Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_12.w_23_0 ctx) (parallel_generatedRoot_12.w_27_0 ctx)

def parallel_generatedRoot_12.w_29_0  (ctx : parallel_generatedRoot_12Ctx) : mxxMat1 :=
  (0 : mxxMat1)

def parallel_generatedRoot_12.w_30_0  (ctx : parallel_generatedRoot_12Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_12.w_28_0 ctx) (parallel_generatedRoot_12.w_29_0 ctx)

def parallel_generatedRoot_12.outs  (ctx : parallel_generatedRoot_12Ctx) : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit :=
  ((parallel_generatedRoot_12.w_1_0 ctx), ((parallel_generatedRoot_12.w_2_0 ctx), ((parallel_generatedRoot_12.w_21_0 ctx), ((parallel_generatedRoot_12.w_30_0 ctx), ()))))

structure parallel_generatedRoot_12.Facts_0  (ctx : parallel_generatedRoot_12Ctx) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
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

theorem parallel_generatedRoot_12.facts_0  {ctx : parallel_generatedRoot_12Ctx} {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_12.constraints_0 backend hashModel params i_0 (parallel_generatedRoot_12.w_0_0 ctx) (parallel_generatedRoot_12.w_3_0 ctx) (parallel_generatedRoot_12.w_1_0 ctx) (parallel_generatedRoot_12.w_2_0 ctx) (parallel_generatedRoot_12.w_4_0 ctx) (parallel_generatedRoot_12.w_9_0 ctx) (parallel_generatedRoot_12.w_10_0 ctx) (parallel_generatedRoot_12.w_11_0 ctx) (parallel_generatedRoot_12.w_12_0 ctx) (parallel_generatedRoot_12.w_13_0 ctx) (parallel_generatedRoot_12.w_17_0 ctx) (parallel_generatedRoot_12.w_21_0 ctx) (parallel_generatedRoot_12.w_30_0 ctx) outputs) :
    parallel_generatedRoot_12.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_12.Facts  (ctx : parallel_generatedRoot_12Ctx) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  c0 : parallel_generatedRoot_12.Facts_0 ctx outputs

theorem parallel_generatedRoot_12.facts  (ctx : parallel_generatedRoot_12Ctx) {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_12.body backend hashModel params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_12.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_12.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_13.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) where
  inputs : ByteArray × mxxMat0 × mxxMat0 × Unit
  witness : parallel_generatedRoot_13.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_13Ctx" => (parallel_generatedRoot_13.Ctx backend hashModel params i_0)

def parallel_generatedRoot_13.w_0_0  (ctx : parallel_generatedRoot_13Ctx) : ByteArray :=
  ctx.inputs.1

def parallel_generatedRoot_13.w_3_0  (ctx : parallel_generatedRoot_13Ctx) : mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_13.w_25_0  (ctx : parallel_generatedRoot_13Ctx) : mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_13.w_1_0  (ctx : parallel_generatedRoot_13Ctx) : mxxMat1 :=
  ctx.witness.w_1_0

def parallel_generatedRoot_13.w_2_0  (ctx : parallel_generatedRoot_13Ctx) : mxxMat1 :=
  ctx.witness.w_2_0

def parallel_generatedRoot_13.w_4_0  (ctx : parallel_generatedRoot_13Ctx) : mxxMat0 :=
  ctx.witness.w_4_0

def parallel_generatedRoot_13.w_5_0  (ctx : parallel_generatedRoot_13Ctx) : mxxMat0 :=
  MxxRuntime.matrixNeg (parallel_generatedRoot_13.w_4_0 ctx)

def parallel_generatedRoot_13.w_6_0  (ctx : parallel_generatedRoot_13Ctx) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_13.w_5_0 ctx) (parallel_generatedRoot_13.w_1_0 ctx)

def parallel_generatedRoot_13.w_7_0  (ctx : parallel_generatedRoot_13Ctx) : mxxMat0 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat0)

def parallel_generatedRoot_13.w_8_0  (ctx : parallel_generatedRoot_13Ctx) : mxxMat0 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat0)

def parallel_generatedRoot_13.w_9_0  (ctx : parallel_generatedRoot_13Ctx) : mxxMat0 :=
  (0 : mxxMat0)

def parallel_generatedRoot_13.w_10_0  (ctx : parallel_generatedRoot_13Ctx) : Fin 1 → Int :=
  ctx.witness.w_10_0

def parallel_generatedRoot_13.w_11_0  (ctx : parallel_generatedRoot_13Ctx) : mxxMat0 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat0)

def parallel_generatedRoot_13.w_12_0  (ctx : parallel_generatedRoot_13Ctx) : Int :=
  -1

def parallel_generatedRoot_13.w_13_0  (ctx : parallel_generatedRoot_13Ctx) : mxxMat0 :=
  ctx.witness.w_13_0

def parallel_generatedRoot_13.w_14_0  (ctx : parallel_generatedRoot_13Ctx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_13.w_8_0 ctx) (parallel_generatedRoot_13.w_13_0 ctx)

def parallel_generatedRoot_13.w_15_0  (ctx : parallel_generatedRoot_13Ctx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_13.w_7_0 ctx) (parallel_generatedRoot_13.w_14_0 ctx)

def parallel_generatedRoot_13.w_16_0  (ctx : parallel_generatedRoot_13Ctx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_13.w_15_0 ctx) (parallel_generatedRoot_13.w_3_0 ctx)

def parallel_generatedRoot_13.w_17_0  (ctx : parallel_generatedRoot_13Ctx) : mxxMat1 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_13.w_18_0  (ctx : parallel_generatedRoot_13Ctx) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_13.w_16_0 ctx) (parallel_generatedRoot_13.w_17_0 ctx)

def parallel_generatedRoot_13.w_19_0  (ctx : parallel_generatedRoot_13Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_13.w_6_0 ctx) (parallel_generatedRoot_13.w_18_0 ctx)

def parallel_generatedRoot_13.w_20_0  (ctx : parallel_generatedRoot_13Ctx) : mxxMat1 :=
  (0 : mxxMat1)

def parallel_generatedRoot_13.w_21_0  (ctx : parallel_generatedRoot_13Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_13.w_19_0 ctx) (parallel_generatedRoot_13.w_20_0 ctx)

def parallel_generatedRoot_13.w_22_0  (ctx : parallel_generatedRoot_13Ctx) : mxxMat0 :=
  MxxRuntime.matrixNeg (parallel_generatedRoot_13.w_4_0 ctx)

def parallel_generatedRoot_13.w_23_0  (ctx : parallel_generatedRoot_13Ctx) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_13.w_22_0 ctx) (parallel_generatedRoot_13.w_2_0 ctx)

def parallel_generatedRoot_13.w_24_0  (ctx : parallel_generatedRoot_13Ctx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_13.w_7_0 ctx) (parallel_generatedRoot_13.w_14_0 ctx)

def parallel_generatedRoot_13.w_26_0  (ctx : parallel_generatedRoot_13Ctx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_13.w_24_0 ctx) (parallel_generatedRoot_13.w_25_0 ctx)

def parallel_generatedRoot_13.w_27_0  (ctx : parallel_generatedRoot_13Ctx) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_13.w_26_0 ctx) (parallel_generatedRoot_13.w_17_0 ctx)

def parallel_generatedRoot_13.w_28_0  (ctx : parallel_generatedRoot_13Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_13.w_23_0 ctx) (parallel_generatedRoot_13.w_27_0 ctx)

def parallel_generatedRoot_13.w_29_0  (ctx : parallel_generatedRoot_13Ctx) : mxxMat1 :=
  (0 : mxxMat1)

def parallel_generatedRoot_13.w_30_0  (ctx : parallel_generatedRoot_13Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_13.w_28_0 ctx) (parallel_generatedRoot_13.w_29_0 ctx)

def parallel_generatedRoot_13.outs  (ctx : parallel_generatedRoot_13Ctx) : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit :=
  ((parallel_generatedRoot_13.w_1_0 ctx), ((parallel_generatedRoot_13.w_2_0 ctx), ((parallel_generatedRoot_13.w_21_0 ctx), ((parallel_generatedRoot_13.w_30_0 ctx), ()))))

structure parallel_generatedRoot_13.Facts_0  (ctx : parallel_generatedRoot_13Ctx) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
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

theorem parallel_generatedRoot_13.facts_0  {ctx : parallel_generatedRoot_13Ctx} {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_13.constraints_0 backend hashModel params i_0 (parallel_generatedRoot_13.w_0_0 ctx) (parallel_generatedRoot_13.w_3_0 ctx) (parallel_generatedRoot_13.w_1_0 ctx) (parallel_generatedRoot_13.w_2_0 ctx) (parallel_generatedRoot_13.w_4_0 ctx) (parallel_generatedRoot_13.w_9_0 ctx) (parallel_generatedRoot_13.w_10_0 ctx) (parallel_generatedRoot_13.w_11_0 ctx) (parallel_generatedRoot_13.w_12_0 ctx) (parallel_generatedRoot_13.w_13_0 ctx) (parallel_generatedRoot_13.w_17_0 ctx) (parallel_generatedRoot_13.w_21_0 ctx) (parallel_generatedRoot_13.w_30_0 ctx) outputs) :
    parallel_generatedRoot_13.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_13.Facts  (ctx : parallel_generatedRoot_13Ctx) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  c0 : parallel_generatedRoot_13.Facts_0 ctx outputs

theorem parallel_generatedRoot_13.facts  (ctx : parallel_generatedRoot_13Ctx) {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_13.body backend hashModel params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_13.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_13.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_14.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) where
  inputs : ByteArray × mxxMat0 × mxxMat0 × Unit
  witness : parallel_generatedRoot_14.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_14Ctx" => (parallel_generatedRoot_14.Ctx backend hashModel params i_0)

def parallel_generatedRoot_14.w_0_0  (ctx : parallel_generatedRoot_14Ctx) : ByteArray :=
  ctx.inputs.1

def parallel_generatedRoot_14.w_3_0  (ctx : parallel_generatedRoot_14Ctx) : mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_14.w_25_0  (ctx : parallel_generatedRoot_14Ctx) : mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_14.w_1_0  (ctx : parallel_generatedRoot_14Ctx) : mxxMat1 :=
  ctx.witness.w_1_0

def parallel_generatedRoot_14.w_2_0  (ctx : parallel_generatedRoot_14Ctx) : mxxMat1 :=
  ctx.witness.w_2_0

def parallel_generatedRoot_14.w_4_0  (ctx : parallel_generatedRoot_14Ctx) : mxxMat0 :=
  ctx.witness.w_4_0

def parallel_generatedRoot_14.w_5_0  (ctx : parallel_generatedRoot_14Ctx) : mxxMat0 :=
  MxxRuntime.matrixNeg (parallel_generatedRoot_14.w_4_0 ctx)

def parallel_generatedRoot_14.w_6_0  (ctx : parallel_generatedRoot_14Ctx) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_14.w_5_0 ctx) (parallel_generatedRoot_14.w_1_0 ctx)

def parallel_generatedRoot_14.w_7_0  (ctx : parallel_generatedRoot_14Ctx) : mxxMat0 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat0)

def parallel_generatedRoot_14.w_8_0  (ctx : parallel_generatedRoot_14Ctx) : mxxMat0 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat0)

def parallel_generatedRoot_14.w_9_0  (ctx : parallel_generatedRoot_14Ctx) : mxxMat0 :=
  (0 : mxxMat0)

def parallel_generatedRoot_14.w_10_0  (ctx : parallel_generatedRoot_14Ctx) : Fin 1 → Int :=
  ctx.witness.w_10_0

def parallel_generatedRoot_14.w_11_0  (ctx : parallel_generatedRoot_14Ctx) : mxxMat0 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat0)

def parallel_generatedRoot_14.w_12_0  (ctx : parallel_generatedRoot_14Ctx) : Int :=
  -1

def parallel_generatedRoot_14.w_13_0  (ctx : parallel_generatedRoot_14Ctx) : mxxMat0 :=
  ctx.witness.w_13_0

def parallel_generatedRoot_14.w_14_0  (ctx : parallel_generatedRoot_14Ctx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_14.w_8_0 ctx) (parallel_generatedRoot_14.w_13_0 ctx)

def parallel_generatedRoot_14.w_15_0  (ctx : parallel_generatedRoot_14Ctx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_14.w_7_0 ctx) (parallel_generatedRoot_14.w_14_0 ctx)

def parallel_generatedRoot_14.w_16_0  (ctx : parallel_generatedRoot_14Ctx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_14.w_15_0 ctx) (parallel_generatedRoot_14.w_3_0 ctx)

def parallel_generatedRoot_14.w_17_0  (ctx : parallel_generatedRoot_14Ctx) : mxxMat1 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_14.w_18_0  (ctx : parallel_generatedRoot_14Ctx) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_14.w_16_0 ctx) (parallel_generatedRoot_14.w_17_0 ctx)

def parallel_generatedRoot_14.w_19_0  (ctx : parallel_generatedRoot_14Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_14.w_6_0 ctx) (parallel_generatedRoot_14.w_18_0 ctx)

def parallel_generatedRoot_14.w_20_0  (ctx : parallel_generatedRoot_14Ctx) : mxxMat1 :=
  (0 : mxxMat1)

def parallel_generatedRoot_14.w_21_0  (ctx : parallel_generatedRoot_14Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_14.w_19_0 ctx) (parallel_generatedRoot_14.w_20_0 ctx)

def parallel_generatedRoot_14.w_22_0  (ctx : parallel_generatedRoot_14Ctx) : mxxMat0 :=
  MxxRuntime.matrixNeg (parallel_generatedRoot_14.w_4_0 ctx)

def parallel_generatedRoot_14.w_23_0  (ctx : parallel_generatedRoot_14Ctx) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_14.w_22_0 ctx) (parallel_generatedRoot_14.w_2_0 ctx)

def parallel_generatedRoot_14.w_24_0  (ctx : parallel_generatedRoot_14Ctx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_14.w_7_0 ctx) (parallel_generatedRoot_14.w_14_0 ctx)

def parallel_generatedRoot_14.w_26_0  (ctx : parallel_generatedRoot_14Ctx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_14.w_24_0 ctx) (parallel_generatedRoot_14.w_25_0 ctx)

def parallel_generatedRoot_14.w_27_0  (ctx : parallel_generatedRoot_14Ctx) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_14.w_26_0 ctx) (parallel_generatedRoot_14.w_17_0 ctx)

def parallel_generatedRoot_14.w_28_0  (ctx : parallel_generatedRoot_14Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_14.w_23_0 ctx) (parallel_generatedRoot_14.w_27_0 ctx)

def parallel_generatedRoot_14.w_29_0  (ctx : parallel_generatedRoot_14Ctx) : mxxMat1 :=
  (0 : mxxMat1)

def parallel_generatedRoot_14.w_30_0  (ctx : parallel_generatedRoot_14Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_14.w_28_0 ctx) (parallel_generatedRoot_14.w_29_0 ctx)

def parallel_generatedRoot_14.outs  (ctx : parallel_generatedRoot_14Ctx) : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit :=
  ((parallel_generatedRoot_14.w_1_0 ctx), ((parallel_generatedRoot_14.w_2_0 ctx), ((parallel_generatedRoot_14.w_21_0 ctx), ((parallel_generatedRoot_14.w_30_0 ctx), ()))))

structure parallel_generatedRoot_14.Facts_0  (ctx : parallel_generatedRoot_14Ctx) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
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

theorem parallel_generatedRoot_14.facts_0  {ctx : parallel_generatedRoot_14Ctx} {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_14.constraints_0 backend hashModel params i_0 (parallel_generatedRoot_14.w_0_0 ctx) (parallel_generatedRoot_14.w_3_0 ctx) (parallel_generatedRoot_14.w_1_0 ctx) (parallel_generatedRoot_14.w_2_0 ctx) (parallel_generatedRoot_14.w_4_0 ctx) (parallel_generatedRoot_14.w_9_0 ctx) (parallel_generatedRoot_14.w_10_0 ctx) (parallel_generatedRoot_14.w_11_0 ctx) (parallel_generatedRoot_14.w_12_0 ctx) (parallel_generatedRoot_14.w_13_0 ctx) (parallel_generatedRoot_14.w_17_0 ctx) (parallel_generatedRoot_14.w_21_0 ctx) (parallel_generatedRoot_14.w_30_0 ctx) outputs) :
    parallel_generatedRoot_14.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_14.Facts  (ctx : parallel_generatedRoot_14Ctx) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  c0 : parallel_generatedRoot_14.Facts_0 ctx outputs

theorem parallel_generatedRoot_14.facts  (ctx : parallel_generatedRoot_14Ctx) {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_14.body backend hashModel params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_14.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_14.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_15.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) where
  inputs : ByteArray × mxxMat0 × mxxMat0 × Unit
  witness : parallel_generatedRoot_15.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_15Ctx" => (parallel_generatedRoot_15.Ctx backend hashModel params i_0)

def parallel_generatedRoot_15.w_0_0  (ctx : parallel_generatedRoot_15Ctx) : ByteArray :=
  ctx.inputs.1

def parallel_generatedRoot_15.w_3_0  (ctx : parallel_generatedRoot_15Ctx) : mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_15.w_25_0  (ctx : parallel_generatedRoot_15Ctx) : mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_15.w_1_0  (ctx : parallel_generatedRoot_15Ctx) : mxxMat1 :=
  ctx.witness.w_1_0

def parallel_generatedRoot_15.w_2_0  (ctx : parallel_generatedRoot_15Ctx) : mxxMat1 :=
  ctx.witness.w_2_0

def parallel_generatedRoot_15.w_4_0  (ctx : parallel_generatedRoot_15Ctx) : mxxMat0 :=
  ctx.witness.w_4_0

def parallel_generatedRoot_15.w_5_0  (ctx : parallel_generatedRoot_15Ctx) : mxxMat0 :=
  MxxRuntime.matrixNeg (parallel_generatedRoot_15.w_4_0 ctx)

def parallel_generatedRoot_15.w_6_0  (ctx : parallel_generatedRoot_15Ctx) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_15.w_5_0 ctx) (parallel_generatedRoot_15.w_1_0 ctx)

def parallel_generatedRoot_15.w_7_0  (ctx : parallel_generatedRoot_15Ctx) : mxxMat0 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat0)

def parallel_generatedRoot_15.w_8_0  (ctx : parallel_generatedRoot_15Ctx) : mxxMat0 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat0)

def parallel_generatedRoot_15.w_9_0  (ctx : parallel_generatedRoot_15Ctx) : mxxMat0 :=
  (0 : mxxMat0)

def parallel_generatedRoot_15.w_10_0  (ctx : parallel_generatedRoot_15Ctx) : Fin 1 → Int :=
  ctx.witness.w_10_0

def parallel_generatedRoot_15.w_11_0  (ctx : parallel_generatedRoot_15Ctx) : mxxMat0 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat0)

def parallel_generatedRoot_15.w_12_0  (ctx : parallel_generatedRoot_15Ctx) : Int :=
  -1

def parallel_generatedRoot_15.w_13_0  (ctx : parallel_generatedRoot_15Ctx) : mxxMat0 :=
  ctx.witness.w_13_0

def parallel_generatedRoot_15.w_14_0  (ctx : parallel_generatedRoot_15Ctx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_15.w_8_0 ctx) (parallel_generatedRoot_15.w_13_0 ctx)

def parallel_generatedRoot_15.w_15_0  (ctx : parallel_generatedRoot_15Ctx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_15.w_7_0 ctx) (parallel_generatedRoot_15.w_14_0 ctx)

def parallel_generatedRoot_15.w_16_0  (ctx : parallel_generatedRoot_15Ctx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_15.w_15_0 ctx) (parallel_generatedRoot_15.w_3_0 ctx)

def parallel_generatedRoot_15.w_17_0  (ctx : parallel_generatedRoot_15Ctx) : mxxMat1 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_15.w_18_0  (ctx : parallel_generatedRoot_15Ctx) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_15.w_16_0 ctx) (parallel_generatedRoot_15.w_17_0 ctx)

def parallel_generatedRoot_15.w_19_0  (ctx : parallel_generatedRoot_15Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_15.w_6_0 ctx) (parallel_generatedRoot_15.w_18_0 ctx)

def parallel_generatedRoot_15.w_20_0  (ctx : parallel_generatedRoot_15Ctx) : mxxMat1 :=
  (0 : mxxMat1)

def parallel_generatedRoot_15.w_21_0  (ctx : parallel_generatedRoot_15Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_15.w_19_0 ctx) (parallel_generatedRoot_15.w_20_0 ctx)

def parallel_generatedRoot_15.w_22_0  (ctx : parallel_generatedRoot_15Ctx) : mxxMat0 :=
  MxxRuntime.matrixNeg (parallel_generatedRoot_15.w_4_0 ctx)

def parallel_generatedRoot_15.w_23_0  (ctx : parallel_generatedRoot_15Ctx) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_15.w_22_0 ctx) (parallel_generatedRoot_15.w_2_0 ctx)

def parallel_generatedRoot_15.w_24_0  (ctx : parallel_generatedRoot_15Ctx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_15.w_7_0 ctx) (parallel_generatedRoot_15.w_14_0 ctx)

def parallel_generatedRoot_15.w_26_0  (ctx : parallel_generatedRoot_15Ctx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_15.w_24_0 ctx) (parallel_generatedRoot_15.w_25_0 ctx)

def parallel_generatedRoot_15.w_27_0  (ctx : parallel_generatedRoot_15Ctx) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_15.w_26_0 ctx) (parallel_generatedRoot_15.w_17_0 ctx)

def parallel_generatedRoot_15.w_28_0  (ctx : parallel_generatedRoot_15Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_15.w_23_0 ctx) (parallel_generatedRoot_15.w_27_0 ctx)

def parallel_generatedRoot_15.w_29_0  (ctx : parallel_generatedRoot_15Ctx) : mxxMat1 :=
  (0 : mxxMat1)

def parallel_generatedRoot_15.w_30_0  (ctx : parallel_generatedRoot_15Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_15.w_28_0 ctx) (parallel_generatedRoot_15.w_29_0 ctx)

def parallel_generatedRoot_15.outs  (ctx : parallel_generatedRoot_15Ctx) : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit :=
  ((parallel_generatedRoot_15.w_1_0 ctx), ((parallel_generatedRoot_15.w_2_0 ctx), ((parallel_generatedRoot_15.w_21_0 ctx), ((parallel_generatedRoot_15.w_30_0 ctx), ()))))

structure parallel_generatedRoot_15.Facts_0  (ctx : parallel_generatedRoot_15Ctx) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
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

theorem parallel_generatedRoot_15.facts_0  {ctx : parallel_generatedRoot_15Ctx} {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_15.constraints_0 backend hashModel params i_0 (parallel_generatedRoot_15.w_0_0 ctx) (parallel_generatedRoot_15.w_3_0 ctx) (parallel_generatedRoot_15.w_1_0 ctx) (parallel_generatedRoot_15.w_2_0 ctx) (parallel_generatedRoot_15.w_4_0 ctx) (parallel_generatedRoot_15.w_9_0 ctx) (parallel_generatedRoot_15.w_10_0 ctx) (parallel_generatedRoot_15.w_11_0 ctx) (parallel_generatedRoot_15.w_12_0 ctx) (parallel_generatedRoot_15.w_13_0 ctx) (parallel_generatedRoot_15.w_17_0 ctx) (parallel_generatedRoot_15.w_21_0 ctx) (parallel_generatedRoot_15.w_30_0 ctx) outputs) :
    parallel_generatedRoot_15.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_15.Facts  (ctx : parallel_generatedRoot_15Ctx) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  c0 : parallel_generatedRoot_15.Facts_0 ctx outputs

theorem parallel_generatedRoot_15.facts  (ctx : parallel_generatedRoot_15Ctx) {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_15.body backend hashModel params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_15.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_15.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_80.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) where
  inputs : (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × Unit
  witness : parallel_generatedRoot_80.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_80Ctx" => (parallel_generatedRoot_80.Ctx backend hashModel params i_0)

def parallel_generatedRoot_80.w_0_0  (ctx : parallel_generatedRoot_80Ctx) : Fin 72000 → mxxMat1 :=
  ctx.inputs.1

def parallel_generatedRoot_80.w_1_0  (ctx : parallel_generatedRoot_80Ctx) : Fin 72000 → mxxMat1 :=
  ctx.inputs.2.1

def parallel_generatedRoot_80.w_2_0  (ctx : parallel_generatedRoot_80Ctx) : Fin 72000 → mxxMat1 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_80.w_3_0  (ctx : parallel_generatedRoot_80Ctx) : Fin 72000 → mxxMat1 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_80.w_4_0  (ctx : parallel_generatedRoot_80Ctx) : Fin 2 → mxxMat1 :=
  ctx.witness.w_4_0

def parallel_generatedRoot_80.w_4_1  (ctx : parallel_generatedRoot_80Ctx) : Fin 2 → mxxMat1 :=
  ctx.witness.w_4_1

def parallel_generatedRoot_80.w_4_2  (ctx : parallel_generatedRoot_80Ctx) : Fin 2 → mxxMat1 :=
  ctx.witness.w_4_2

def parallel_generatedRoot_80.w_4_3  (ctx : parallel_generatedRoot_80Ctx) : Fin 2 → mxxMat1 :=
  ctx.witness.w_4_3

def parallel_generatedRoot_80.w_5_0  (ctx : parallel_generatedRoot_80Ctx) : mxxMat1 :=
  ctx.witness.w_5_0

def parallel_generatedRoot_80.w_6_0  (ctx : parallel_generatedRoot_80Ctx) : mxxMat1 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_80.w_7_0  (ctx : parallel_generatedRoot_80Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_80.w_5_0 ctx) (parallel_generatedRoot_80.w_6_0 ctx)

def parallel_generatedRoot_80.w_8_0  (ctx : parallel_generatedRoot_80Ctx) : mxxMat1 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_80.w_9_0  (ctx : parallel_generatedRoot_80Ctx) : mxxMat1 :=
  ctx.witness.w_9_0

def parallel_generatedRoot_80.w_10_0  (ctx : parallel_generatedRoot_80Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_80.w_8_0 ctx) (parallel_generatedRoot_80.w_9_0 ctx)

def parallel_generatedRoot_80.w_11_0  (ctx : parallel_generatedRoot_80Ctx) : mxxMat1 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_80.w_12_0  (ctx : parallel_generatedRoot_80Ctx) : mxxMat1 :=
  ctx.witness.w_12_0

def parallel_generatedRoot_80.w_13_0  (ctx : parallel_generatedRoot_80Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_80.w_11_0 ctx) (parallel_generatedRoot_80.w_12_0 ctx)

def parallel_generatedRoot_80.w_14_0  (ctx : parallel_generatedRoot_80Ctx) : mxxMat1 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_80.w_15_0  (ctx : parallel_generatedRoot_80Ctx) : mxxMat1 :=
  ctx.witness.w_15_0

def parallel_generatedRoot_80.w_16_0  (ctx : parallel_generatedRoot_80Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_80.w_14_0 ctx) (parallel_generatedRoot_80.w_15_0 ctx)

def parallel_generatedRoot_80.w_17_0  (ctx : parallel_generatedRoot_80Ctx) : mxxMat1 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_80.w_17_1  (ctx : parallel_generatedRoot_80Ctx) : mxxMat1 :=
  ctx.witness.w_17_1

def parallel_generatedRoot_80.w_17_2  (ctx : parallel_generatedRoot_80Ctx) : mxxMat1 :=
  ctx.witness.w_17_2

def parallel_generatedRoot_80.w_17_3  (ctx : parallel_generatedRoot_80Ctx) : mxxMat1 :=
  ctx.witness.w_17_3

def parallel_generatedRoot_80.outs  (ctx : parallel_generatedRoot_80Ctx) : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit :=
  ((parallel_generatedRoot_80.w_17_0 ctx), ((parallel_generatedRoot_80.w_17_1 ctx), ((parallel_generatedRoot_80.w_17_2 ctx), ((parallel_generatedRoot_80.w_17_3 ctx), ()))))

structure parallel_generatedRoot_80.Facts_0  (ctx : parallel_generatedRoot_80Ctx) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
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

theorem parallel_generatedRoot_80.facts_0  {ctx : parallel_generatedRoot_80Ctx} {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_80.constraints_0 backend hashModel params i_0 (parallel_generatedRoot_80.w_0_0 ctx) (parallel_generatedRoot_80.w_1_0 ctx) (parallel_generatedRoot_80.w_2_0 ctx) (parallel_generatedRoot_80.w_3_0 ctx) (parallel_generatedRoot_80.w_4_0 ctx) (parallel_generatedRoot_80.w_4_1 ctx) (parallel_generatedRoot_80.w_4_2 ctx) (parallel_generatedRoot_80.w_4_3 ctx) (parallel_generatedRoot_80.w_5_0 ctx) (parallel_generatedRoot_80.w_6_0 ctx) (parallel_generatedRoot_80.w_7_0 ctx) (parallel_generatedRoot_80.w_8_0 ctx) (parallel_generatedRoot_80.w_9_0 ctx) (parallel_generatedRoot_80.w_10_0 ctx) (parallel_generatedRoot_80.w_11_0 ctx) (parallel_generatedRoot_80.w_12_0 ctx) (parallel_generatedRoot_80.w_13_0 ctx) (parallel_generatedRoot_80.w_14_0 ctx) (parallel_generatedRoot_80.w_15_0 ctx) (parallel_generatedRoot_80.w_16_0 ctx) (parallel_generatedRoot_80.w_17_0 ctx) (parallel_generatedRoot_80.w_17_1 ctx) (parallel_generatedRoot_80.w_17_2 ctx) (parallel_generatedRoot_80.w_17_3 ctx) outputs) :
    parallel_generatedRoot_80.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_80.Facts  (ctx : parallel_generatedRoot_80Ctx) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  c0 : parallel_generatedRoot_80.Facts_0 ctx outputs

theorem parallel_generatedRoot_80.facts  (ctx : parallel_generatedRoot_80Ctx) {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_80.body backend hashModel params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_80.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_80.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_84.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) where
  inputs : (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × Unit
  witness : parallel_generatedRoot_84.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_84Ctx" => (parallel_generatedRoot_84.Ctx backend hashModel params i_0)

def parallel_generatedRoot_84.w_0_0  (ctx : parallel_generatedRoot_84Ctx) : Fin 72000 → mxxMat1 :=
  ctx.inputs.1

def parallel_generatedRoot_84.w_1_0  (ctx : parallel_generatedRoot_84Ctx) : Fin 72000 → mxxMat1 :=
  ctx.inputs.2.1

def parallel_generatedRoot_84.w_2_0  (ctx : parallel_generatedRoot_84Ctx) : Fin 72000 → mxxMat1 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_84.w_3_0  (ctx : parallel_generatedRoot_84Ctx) : Fin 72000 → mxxMat1 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_84.w_4_0  (ctx : parallel_generatedRoot_84Ctx) : Fin 2 → mxxMat1 :=
  ctx.witness.w_4_0

def parallel_generatedRoot_84.w_4_1  (ctx : parallel_generatedRoot_84Ctx) : Fin 2 → mxxMat1 :=
  ctx.witness.w_4_1

def parallel_generatedRoot_84.w_4_2  (ctx : parallel_generatedRoot_84Ctx) : Fin 2 → mxxMat1 :=
  ctx.witness.w_4_2

def parallel_generatedRoot_84.w_4_3  (ctx : parallel_generatedRoot_84Ctx) : Fin 2 → mxxMat1 :=
  ctx.witness.w_4_3

def parallel_generatedRoot_84.w_5_0  (ctx : parallel_generatedRoot_84Ctx) : mxxMat1 :=
  ctx.witness.w_5_0

def parallel_generatedRoot_84.w_6_0  (ctx : parallel_generatedRoot_84Ctx) : mxxMat1 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_84.w_7_0  (ctx : parallel_generatedRoot_84Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_84.w_5_0 ctx) (parallel_generatedRoot_84.w_6_0 ctx)

def parallel_generatedRoot_84.w_8_0  (ctx : parallel_generatedRoot_84Ctx) : mxxMat1 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_84.w_9_0  (ctx : parallel_generatedRoot_84Ctx) : mxxMat1 :=
  ctx.witness.w_9_0

def parallel_generatedRoot_84.w_10_0  (ctx : parallel_generatedRoot_84Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_84.w_8_0 ctx) (parallel_generatedRoot_84.w_9_0 ctx)

def parallel_generatedRoot_84.w_11_0  (ctx : parallel_generatedRoot_84Ctx) : mxxMat1 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_84.w_12_0  (ctx : parallel_generatedRoot_84Ctx) : mxxMat1 :=
  ctx.witness.w_12_0

def parallel_generatedRoot_84.w_13_0  (ctx : parallel_generatedRoot_84Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_84.w_11_0 ctx) (parallel_generatedRoot_84.w_12_0 ctx)

def parallel_generatedRoot_84.w_14_0  (ctx : parallel_generatedRoot_84Ctx) : mxxMat1 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_84.w_15_0  (ctx : parallel_generatedRoot_84Ctx) : mxxMat1 :=
  ctx.witness.w_15_0

def parallel_generatedRoot_84.w_16_0  (ctx : parallel_generatedRoot_84Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_84.w_14_0 ctx) (parallel_generatedRoot_84.w_15_0 ctx)

def parallel_generatedRoot_84.w_17_0  (ctx : parallel_generatedRoot_84Ctx) : mxxMat1 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_84.w_17_1  (ctx : parallel_generatedRoot_84Ctx) : mxxMat1 :=
  ctx.witness.w_17_1

def parallel_generatedRoot_84.w_17_2  (ctx : parallel_generatedRoot_84Ctx) : mxxMat1 :=
  ctx.witness.w_17_2

def parallel_generatedRoot_84.w_17_3  (ctx : parallel_generatedRoot_84Ctx) : mxxMat1 :=
  ctx.witness.w_17_3

def parallel_generatedRoot_84.outs  (ctx : parallel_generatedRoot_84Ctx) : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit :=
  ((parallel_generatedRoot_84.w_17_0 ctx), ((parallel_generatedRoot_84.w_17_1 ctx), ((parallel_generatedRoot_84.w_17_2 ctx), ((parallel_generatedRoot_84.w_17_3 ctx), ()))))

structure parallel_generatedRoot_84.Facts_0  (ctx : parallel_generatedRoot_84Ctx) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
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

theorem parallel_generatedRoot_84.facts_0  {ctx : parallel_generatedRoot_84Ctx} {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_84.constraints_0 backend hashModel params i_0 (parallel_generatedRoot_84.w_0_0 ctx) (parallel_generatedRoot_84.w_1_0 ctx) (parallel_generatedRoot_84.w_2_0 ctx) (parallel_generatedRoot_84.w_3_0 ctx) (parallel_generatedRoot_84.w_4_0 ctx) (parallel_generatedRoot_84.w_4_1 ctx) (parallel_generatedRoot_84.w_4_2 ctx) (parallel_generatedRoot_84.w_4_3 ctx) (parallel_generatedRoot_84.w_5_0 ctx) (parallel_generatedRoot_84.w_6_0 ctx) (parallel_generatedRoot_84.w_7_0 ctx) (parallel_generatedRoot_84.w_8_0 ctx) (parallel_generatedRoot_84.w_9_0 ctx) (parallel_generatedRoot_84.w_10_0 ctx) (parallel_generatedRoot_84.w_11_0 ctx) (parallel_generatedRoot_84.w_12_0 ctx) (parallel_generatedRoot_84.w_13_0 ctx) (parallel_generatedRoot_84.w_14_0 ctx) (parallel_generatedRoot_84.w_15_0 ctx) (parallel_generatedRoot_84.w_16_0 ctx) (parallel_generatedRoot_84.w_17_0 ctx) (parallel_generatedRoot_84.w_17_1 ctx) (parallel_generatedRoot_84.w_17_2 ctx) (parallel_generatedRoot_84.w_17_3 ctx) outputs) :
    parallel_generatedRoot_84.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_84.Facts  (ctx : parallel_generatedRoot_84Ctx) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  c0 : parallel_generatedRoot_84.Facts_0 ctx outputs

theorem parallel_generatedRoot_84.facts  (ctx : parallel_generatedRoot_84Ctx) {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_84.body backend hashModel params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_84.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_84.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_85.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) where
  inputs : ByteArray × mxxMat1 × mxxMat1 × (Fin 8 → mxxMat1) × mxxMat1 × mxxMat1 × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × Unit
  witness : parallel_generatedRoot_85.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_85Ctx" => (parallel_generatedRoot_85.Ctx backend hashModel params i_0)

def parallel_generatedRoot_85.w_1_0  (ctx : parallel_generatedRoot_85Ctx) : ByteArray :=
  ctx.inputs.1

def parallel_generatedRoot_85.w_2_0  (ctx : parallel_generatedRoot_85Ctx) : mxxMat1 :=
  ctx.inputs.2.1

def parallel_generatedRoot_85.w_3_0  (ctx : parallel_generatedRoot_85Ctx) : mxxMat1 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_85.w_4_0  (ctx : parallel_generatedRoot_85Ctx) : Fin 8 → mxxMat1 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_85.w_5_0  (ctx : parallel_generatedRoot_85Ctx) : mxxMat1 :=
  ctx.inputs.2.2.2.2.1

def parallel_generatedRoot_85.w_6_0  (ctx : parallel_generatedRoot_85Ctx) : mxxMat1 :=
  ctx.inputs.2.2.2.2.2.1

def parallel_generatedRoot_85.w_7_0  (ctx : parallel_generatedRoot_85Ctx) : Fin 8 → mxxMat1 :=
  ctx.inputs.2.2.2.2.2.2.1

def parallel_generatedRoot_85.w_8_0  (ctx : parallel_generatedRoot_85Ctx) : Fin 8 → mxxMat1 :=
  ctx.inputs.2.2.2.2.2.2.2.1

def parallel_generatedRoot_85.w_9_0  (ctx : parallel_generatedRoot_85Ctx) : Fin 8 → mxxMat1 :=
  ctx.inputs.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_85.w_0_0  (ctx : parallel_generatedRoot_85Ctx) : mxxMat1 :=
  (0 : mxxMat1)

def parallel_generatedRoot_85.w_10_0  (ctx : parallel_generatedRoot_85Ctx) : mxxMat1 :=
  ctx.witness.w_10_0

def parallel_generatedRoot_85.w_10_1  (ctx : parallel_generatedRoot_85Ctx) : mxxMat1 :=
  ctx.witness.w_10_1

def parallel_generatedRoot_85.w_10_2  (ctx : parallel_generatedRoot_85Ctx) : mxxMat1 :=
  ctx.witness.w_10_2

def parallel_generatedRoot_85.w_10_3  (ctx : parallel_generatedRoot_85Ctx) : mxxMat1 :=
  ctx.witness.w_10_3

def parallel_generatedRoot_85.outs  (ctx : parallel_generatedRoot_85Ctx) : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit :=
  ((parallel_generatedRoot_85.w_10_0 ctx), ((parallel_generatedRoot_85.w_10_1 ctx), ((parallel_generatedRoot_85.w_10_2 ctx), ((parallel_generatedRoot_85.w_10_3 ctx), ()))))

structure parallel_generatedRoot_85.Facts_0  (ctx : parallel_generatedRoot_85Ctx) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  t0 : 0 ≤ (8)
  h10_3 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_parallel_generatedRoot_85_10 backend hashModel params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, ((parallel_generatedRoot_85.w_1_0 ctx), ((parallel_generatedRoot_85.w_2_0 ctx), ((parallel_generatedRoot_85.w_3_0 ctx), ((parallel_generatedRoot_85.w_4_0 ctx), ((parallel_generatedRoot_85.w_5_0 ctx), ((parallel_generatedRoot_85.w_6_0 ctx), ((parallel_generatedRoot_85.w_7_0 ctx), ((parallel_generatedRoot_85.w_8_0 ctx), ((parallel_generatedRoot_85.w_9_0 ctx), ()))))))))))))) next) (Int.toNat (8)) ((parallel_generatedRoot_85.w_0_0 ctx), ((parallel_generatedRoot_85.w_0_0 ctx), ((parallel_generatedRoot_85.w_0_0 ctx), ((parallel_generatedRoot_85.w_0_0 ctx), ())))) ((parallel_generatedRoot_85.w_10_0 ctx), ((parallel_generatedRoot_85.w_10_1 ctx), ((parallel_generatedRoot_85.w_10_2 ctx), ((parallel_generatedRoot_85.w_10_3 ctx), ()))))
  hout : outputs = parallel_generatedRoot_85.outs ctx

theorem parallel_generatedRoot_85.facts_0  {ctx : parallel_generatedRoot_85Ctx} {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_85.constraints_0 backend hashModel params i_0 (parallel_generatedRoot_85.w_1_0 ctx) (parallel_generatedRoot_85.w_2_0 ctx) (parallel_generatedRoot_85.w_3_0 ctx) (parallel_generatedRoot_85.w_4_0 ctx) (parallel_generatedRoot_85.w_5_0 ctx) (parallel_generatedRoot_85.w_6_0 ctx) (parallel_generatedRoot_85.w_7_0 ctx) (parallel_generatedRoot_85.w_8_0 ctx) (parallel_generatedRoot_85.w_9_0 ctx) (parallel_generatedRoot_85.w_0_0 ctx) (parallel_generatedRoot_85.w_10_0 ctx) (parallel_generatedRoot_85.w_10_1 ctx) (parallel_generatedRoot_85.w_10_2 ctx) (parallel_generatedRoot_85.w_10_3 ctx) outputs) :
    parallel_generatedRoot_85.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure parallel_generatedRoot_85.Facts  (ctx : parallel_generatedRoot_85Ctx) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  c0 : parallel_generatedRoot_85.Facts_0 ctx outputs

theorem parallel_generatedRoot_85.facts  (ctx : parallel_generatedRoot_85Ctx) {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_85.body backend hashModel params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_85.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_85.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_107.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) where
  inputs : mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × (Fin 36000 → mxxMat1) × (Fin 36000 → mxxMat1) × (Fin 36000 → mxxMat1) × (Fin 36000 → mxxMat1) × Unit
  witness : sequential_generatedRoot_107.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_107Ctx" => (sequential_generatedRoot_107.Ctx backend hashModel params i_0)

def sequential_generatedRoot_107.w_0_0  (ctx : sequential_generatedRoot_107Ctx) : mxxMat2 :=
  ctx.inputs.1

def sequential_generatedRoot_107.w_23_0  (ctx : sequential_generatedRoot_107Ctx) : mxxMat2 :=
  ctx.inputs.2.1

def sequential_generatedRoot_107.w_27_0  (ctx : sequential_generatedRoot_107Ctx) : mxxMat2 :=
  ctx.inputs.2.2.1

def sequential_generatedRoot_107.w_31_0  (ctx : sequential_generatedRoot_107Ctx) : mxxMat2 :=
  ctx.inputs.2.2.2.1

def sequential_generatedRoot_107.w_2_0  (ctx : sequential_generatedRoot_107Ctx) : Fin 36000 → mxxMat1 :=
  ctx.inputs.2.2.2.2.1

def sequential_generatedRoot_107.w_8_0  (ctx : sequential_generatedRoot_107Ctx) : Fin 36000 → mxxMat1 :=
  ctx.inputs.2.2.2.2.2.1

def sequential_generatedRoot_107.w_12_0  (ctx : sequential_generatedRoot_107Ctx) : Fin 36000 → mxxMat1 :=
  ctx.inputs.2.2.2.2.2.2.1

def sequential_generatedRoot_107.w_16_0  (ctx : sequential_generatedRoot_107Ctx) : Fin 36000 → mxxMat1 :=
  ctx.inputs.2.2.2.2.2.2.2.1

def sequential_generatedRoot_107.w_1_0  (ctx : sequential_generatedRoot_107Ctx) : mxxMat3 :=
  (MxxRuntime.unitRow ((1 + (Int.ofNat i_0))) : mxxMat3)

def sequential_generatedRoot_107.w_3_0  (ctx : sequential_generatedRoot_107Ctx) : Int :=
  (8 + (8 * (Int.ofNat i_0)))

def sequential_generatedRoot_107.w_4_0  (ctx : sequential_generatedRoot_107Ctx) : mxxMat1 :=
  ctx.witness.w_4_0

def sequential_generatedRoot_107.w_5_0  (ctx : sequential_generatedRoot_107Ctx) : Int :=
  (9 + (8 * (Int.ofNat i_0)))

def sequential_generatedRoot_107.w_6_0  (ctx : sequential_generatedRoot_107Ctx) : mxxMat1 :=
  ctx.witness.w_6_0

def sequential_generatedRoot_107.w_7_0  (ctx : sequential_generatedRoot_107Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (sequential_generatedRoot_107.w_4_0 ctx) (sequential_generatedRoot_107.w_6_0 ctx)

def sequential_generatedRoot_107.w_9_0  (ctx : sequential_generatedRoot_107Ctx) : mxxMat1 :=
  ctx.witness.w_9_0

def sequential_generatedRoot_107.w_10_0  (ctx : sequential_generatedRoot_107Ctx) : mxxMat1 :=
  ctx.witness.w_10_0

def sequential_generatedRoot_107.w_11_0  (ctx : sequential_generatedRoot_107Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (sequential_generatedRoot_107.w_9_0 ctx) (sequential_generatedRoot_107.w_10_0 ctx)

def sequential_generatedRoot_107.w_13_0  (ctx : sequential_generatedRoot_107Ctx) : mxxMat1 :=
  ctx.witness.w_13_0

def sequential_generatedRoot_107.w_14_0  (ctx : sequential_generatedRoot_107Ctx) : mxxMat1 :=
  ctx.witness.w_14_0

def sequential_generatedRoot_107.w_15_0  (ctx : sequential_generatedRoot_107Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (sequential_generatedRoot_107.w_13_0 ctx) (sequential_generatedRoot_107.w_14_0 ctx)

def sequential_generatedRoot_107.w_17_0  (ctx : sequential_generatedRoot_107Ctx) : mxxMat1 :=
  ctx.witness.w_17_0

def sequential_generatedRoot_107.w_18_0  (ctx : sequential_generatedRoot_107Ctx) : mxxMat1 :=
  ctx.witness.w_18_0

def sequential_generatedRoot_107.w_19_0  (ctx : sequential_generatedRoot_107Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (sequential_generatedRoot_107.w_17_0 ctx) (sequential_generatedRoot_107.w_18_0 ctx)

def sequential_generatedRoot_107.w_20_0  (ctx : sequential_generatedRoot_107Ctx) : mxxMat1 :=
  ctx.witness.w_20_0

def sequential_generatedRoot_107.w_20_1  (ctx : sequential_generatedRoot_107Ctx) : mxxMat1 :=
  ctx.witness.w_20_1

def sequential_generatedRoot_107.w_20_2  (ctx : sequential_generatedRoot_107Ctx) : mxxMat1 :=
  ctx.witness.w_20_2

def sequential_generatedRoot_107.w_20_3  (ctx : sequential_generatedRoot_107Ctx) : mxxMat1 :=
  ctx.witness.w_20_3

def sequential_generatedRoot_107.w_21_0  (ctx : sequential_generatedRoot_107Ctx) : mxxMat2 :=
  ctx.witness.w_21_0

def sequential_generatedRoot_107.w_22_0  (ctx : sequential_generatedRoot_107Ctx) : mxxMat2 :=
  MxxRuntime.matrixAdd (sequential_generatedRoot_107.w_0_0 ctx) (sequential_generatedRoot_107.w_21_0 ctx)

def sequential_generatedRoot_107.w_24_0  (ctx : sequential_generatedRoot_107Ctx) : mxxMat3 :=
  (MxxRuntime.unitRow ((1 + (Int.ofNat i_0))) : mxxMat3)

def sequential_generatedRoot_107.w_25_0  (ctx : sequential_generatedRoot_107Ctx) : mxxMat2 :=
  ctx.witness.w_25_0

def sequential_generatedRoot_107.w_26_0  (ctx : sequential_generatedRoot_107Ctx) : mxxMat2 :=
  MxxRuntime.matrixAdd (sequential_generatedRoot_107.w_23_0 ctx) (sequential_generatedRoot_107.w_25_0 ctx)

def sequential_generatedRoot_107.w_28_0  (ctx : sequential_generatedRoot_107Ctx) : mxxMat3 :=
  (MxxRuntime.unitRow ((1 + (Int.ofNat i_0))) : mxxMat3)

def sequential_generatedRoot_107.w_29_0  (ctx : sequential_generatedRoot_107Ctx) : mxxMat2 :=
  ctx.witness.w_29_0

def sequential_generatedRoot_107.w_30_0  (ctx : sequential_generatedRoot_107Ctx) : mxxMat2 :=
  MxxRuntime.matrixAdd (sequential_generatedRoot_107.w_27_0 ctx) (sequential_generatedRoot_107.w_29_0 ctx)

def sequential_generatedRoot_107.w_32_0  (ctx : sequential_generatedRoot_107Ctx) : mxxMat3 :=
  (MxxRuntime.unitRow ((1 + (Int.ofNat i_0))) : mxxMat3)

def sequential_generatedRoot_107.w_33_0  (ctx : sequential_generatedRoot_107Ctx) : mxxMat2 :=
  ctx.witness.w_33_0

def sequential_generatedRoot_107.w_34_0  (ctx : sequential_generatedRoot_107Ctx) : mxxMat2 :=
  MxxRuntime.matrixAdd (sequential_generatedRoot_107.w_31_0 ctx) (sequential_generatedRoot_107.w_33_0 ctx)

def sequential_generatedRoot_107.outs  (ctx : sequential_generatedRoot_107Ctx) : mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × Unit :=
  ((sequential_generatedRoot_107.w_22_0 ctx), ((sequential_generatedRoot_107.w_26_0 ctx), ((sequential_generatedRoot_107.w_30_0 ctx), ((sequential_generatedRoot_107.w_34_0 ctx), ()))))

structure sequential_generatedRoot_107.Facts_0  (ctx : sequential_generatedRoot_107Ctx) : Prop where
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
  h20_3 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => scope_89 backend hashModel params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, ((sequential_generatedRoot_107.w_2_0 ctx), ((sequential_generatedRoot_107.w_8_0 ctx), ((sequential_generatedRoot_107.w_12_0 ctx), ((sequential_generatedRoot_107.w_16_0 ctx), ())))))))) next) (Int.toNat (6)) ((sequential_generatedRoot_107.w_7_0 ctx), ((sequential_generatedRoot_107.w_11_0 ctx), ((sequential_generatedRoot_107.w_15_0 ctx), ((sequential_generatedRoot_107.w_19_0 ctx), ())))) ((sequential_generatedRoot_107.w_20_0 ctx), ((sequential_generatedRoot_107.w_20_1 ctx), ((sequential_generatedRoot_107.w_20_2 ctx), ((sequential_generatedRoot_107.w_20_3 ctx), ()))))
  h21_0 : MxxRuntime.tensorRuns (sequential_generatedRoot_107.w_1_0 ctx) (sequential_generatedRoot_107.w_20_0 ctx) (sequential_generatedRoot_107.w_21_0 ctx)
  t29 : 0 ≤ ((1 + (Int.ofNat i_0)))
  t30 : ((1 + (Int.ofNat i_0))) < 4500
  h25_0 : MxxRuntime.tensorRuns (sequential_generatedRoot_107.w_24_0 ctx) (sequential_generatedRoot_107.w_20_1 ctx) (sequential_generatedRoot_107.w_25_0 ctx)

theorem sequential_generatedRoot_107.facts_0  {ctx : sequential_generatedRoot_107Ctx} {outputs : mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × Unit}
    (h : sequential_generatedRoot_107.constraints_0 backend hashModel params i_0 (sequential_generatedRoot_107.w_2_0 ctx) (sequential_generatedRoot_107.w_8_0 ctx) (sequential_generatedRoot_107.w_12_0 ctx) (sequential_generatedRoot_107.w_16_0 ctx) (sequential_generatedRoot_107.w_1_0 ctx) (sequential_generatedRoot_107.w_3_0 ctx) (sequential_generatedRoot_107.w_4_0 ctx) (sequential_generatedRoot_107.w_5_0 ctx) (sequential_generatedRoot_107.w_6_0 ctx) (sequential_generatedRoot_107.w_7_0 ctx) (sequential_generatedRoot_107.w_9_0 ctx) (sequential_generatedRoot_107.w_10_0 ctx) (sequential_generatedRoot_107.w_11_0 ctx) (sequential_generatedRoot_107.w_13_0 ctx) (sequential_generatedRoot_107.w_14_0 ctx) (sequential_generatedRoot_107.w_15_0 ctx) (sequential_generatedRoot_107.w_17_0 ctx) (sequential_generatedRoot_107.w_18_0 ctx) (sequential_generatedRoot_107.w_19_0 ctx) (sequential_generatedRoot_107.w_20_0 ctx) (sequential_generatedRoot_107.w_20_1 ctx) (sequential_generatedRoot_107.w_20_2 ctx) (sequential_generatedRoot_107.w_20_3 ctx) (sequential_generatedRoot_107.w_21_0 ctx) (sequential_generatedRoot_107.w_24_0 ctx) (sequential_generatedRoot_107.w_25_0 ctx)) :
    sequential_generatedRoot_107.Facts_0 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure sequential_generatedRoot_107.Facts_1  (ctx : sequential_generatedRoot_107Ctx) (outputs : mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × Unit) : Prop where
  t32 : 0 ≤ ((1 + (Int.ofNat i_0)))
  t33 : ((1 + (Int.ofNat i_0))) < 4500
  h29_0 : MxxRuntime.tensorRuns (sequential_generatedRoot_107.w_28_0 ctx) (sequential_generatedRoot_107.w_20_2 ctx) (sequential_generatedRoot_107.w_29_0 ctx)
  t35 : 0 ≤ ((1 + (Int.ofNat i_0)))
  t36 : ((1 + (Int.ofNat i_0))) < 4500
  h33_0 : MxxRuntime.tensorRuns (sequential_generatedRoot_107.w_32_0 ctx) (sequential_generatedRoot_107.w_20_3 ctx) (sequential_generatedRoot_107.w_33_0 ctx)
  hout : outputs = sequential_generatedRoot_107.outs ctx

theorem sequential_generatedRoot_107.facts_1  {ctx : sequential_generatedRoot_107Ctx} {outputs : mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × Unit}
    (h : sequential_generatedRoot_107.constraints_1 i_0 (sequential_generatedRoot_107.w_20_2 ctx) (sequential_generatedRoot_107.w_20_3 ctx) (sequential_generatedRoot_107.w_22_0 ctx) (sequential_generatedRoot_107.w_26_0 ctx) (sequential_generatedRoot_107.w_28_0 ctx) (sequential_generatedRoot_107.w_29_0 ctx) (sequential_generatedRoot_107.w_30_0 ctx) (sequential_generatedRoot_107.w_32_0 ctx) (sequential_generatedRoot_107.w_33_0 ctx) (sequential_generatedRoot_107.w_34_0 ctx) outputs) :
    sequential_generatedRoot_107.Facts_1 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2⟩

structure sequential_generatedRoot_107.Facts  (ctx : sequential_generatedRoot_107Ctx) (outputs : mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × Unit) : Prop where
  c0 : sequential_generatedRoot_107.Facts_0 ctx
  c1 : sequential_generatedRoot_107.Facts_1 ctx outputs

theorem sequential_generatedRoot_107.facts  (ctx : sequential_generatedRoot_107Ctx) {outputs : mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × Unit}
    (h : sequential_generatedRoot_107.body backend hashModel params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_107.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_107.facts_0 (ctx := ctx) (outputs := outputs) t0.1
  have t1 := t0.2
  have r1 := sequential_generatedRoot_107.facts_1 (ctx := ctx) (outputs := outputs) t1
  ⟨r0, r1⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_117.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) where
  inputs : (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × Unit
  witness : parallel_generatedRoot_117.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_117Ctx" => (parallel_generatedRoot_117.Ctx backend hashModel params i_0)

def parallel_generatedRoot_117.w_0_0  (ctx : parallel_generatedRoot_117Ctx) : Fin 16 → mxxMat1 :=
  ctx.inputs.1

def parallel_generatedRoot_117.w_1_0  (ctx : parallel_generatedRoot_117Ctx) : Fin 16 → mxxMat1 :=
  ctx.inputs.2.1

def parallel_generatedRoot_117.w_2_0  (ctx : parallel_generatedRoot_117Ctx) : Fin 16 → mxxMat1 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_117.w_3_0  (ctx : parallel_generatedRoot_117Ctx) : Fin 16 → mxxMat1 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_117.w_4_0  (ctx : parallel_generatedRoot_117Ctx) : Fin 2 → mxxMat1 :=
  ctx.witness.w_4_0

def parallel_generatedRoot_117.w_4_1  (ctx : parallel_generatedRoot_117Ctx) : Fin 2 → mxxMat1 :=
  ctx.witness.w_4_1

def parallel_generatedRoot_117.w_4_2  (ctx : parallel_generatedRoot_117Ctx) : Fin 2 → mxxMat1 :=
  ctx.witness.w_4_2

def parallel_generatedRoot_117.w_4_3  (ctx : parallel_generatedRoot_117Ctx) : Fin 2 → mxxMat1 :=
  ctx.witness.w_4_3

def parallel_generatedRoot_117.w_5_0  (ctx : parallel_generatedRoot_117Ctx) : mxxMat1 :=
  ctx.witness.w_5_0

def parallel_generatedRoot_117.w_6_0  (ctx : parallel_generatedRoot_117Ctx) : mxxMat1 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_117.w_7_0  (ctx : parallel_generatedRoot_117Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_117.w_5_0 ctx) (parallel_generatedRoot_117.w_6_0 ctx)

def parallel_generatedRoot_117.w_8_0  (ctx : parallel_generatedRoot_117Ctx) : mxxMat1 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_117.w_9_0  (ctx : parallel_generatedRoot_117Ctx) : mxxMat1 :=
  ctx.witness.w_9_0

def parallel_generatedRoot_117.w_10_0  (ctx : parallel_generatedRoot_117Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_117.w_8_0 ctx) (parallel_generatedRoot_117.w_9_0 ctx)

def parallel_generatedRoot_117.w_11_0  (ctx : parallel_generatedRoot_117Ctx) : mxxMat1 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_117.w_12_0  (ctx : parallel_generatedRoot_117Ctx) : mxxMat1 :=
  ctx.witness.w_12_0

def parallel_generatedRoot_117.w_13_0  (ctx : parallel_generatedRoot_117Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_117.w_11_0 ctx) (parallel_generatedRoot_117.w_12_0 ctx)

def parallel_generatedRoot_117.w_14_0  (ctx : parallel_generatedRoot_117Ctx) : mxxMat1 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_117.w_15_0  (ctx : parallel_generatedRoot_117Ctx) : mxxMat1 :=
  ctx.witness.w_15_0

def parallel_generatedRoot_117.w_16_0  (ctx : parallel_generatedRoot_117Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_117.w_14_0 ctx) (parallel_generatedRoot_117.w_15_0 ctx)

def parallel_generatedRoot_117.w_17_0  (ctx : parallel_generatedRoot_117Ctx) : mxxMat1 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_117.w_17_1  (ctx : parallel_generatedRoot_117Ctx) : mxxMat1 :=
  ctx.witness.w_17_1

def parallel_generatedRoot_117.w_17_2  (ctx : parallel_generatedRoot_117Ctx) : mxxMat1 :=
  ctx.witness.w_17_2

def parallel_generatedRoot_117.w_17_3  (ctx : parallel_generatedRoot_117Ctx) : mxxMat1 :=
  ctx.witness.w_17_3

def parallel_generatedRoot_117.outs  (ctx : parallel_generatedRoot_117Ctx) : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit :=
  ((parallel_generatedRoot_117.w_17_0 ctx), ((parallel_generatedRoot_117.w_17_1 ctx), ((parallel_generatedRoot_117.w_17_2 ctx), ((parallel_generatedRoot_117.w_17_3 ctx), ()))))

structure parallel_generatedRoot_117.Facts_0  (ctx : parallel_generatedRoot_117Ctx) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
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

theorem parallel_generatedRoot_117.facts_0  {ctx : parallel_generatedRoot_117Ctx} {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_117.constraints_0 backend hashModel params i_0 (parallel_generatedRoot_117.w_0_0 ctx) (parallel_generatedRoot_117.w_1_0 ctx) (parallel_generatedRoot_117.w_2_0 ctx) (parallel_generatedRoot_117.w_3_0 ctx) (parallel_generatedRoot_117.w_4_0 ctx) (parallel_generatedRoot_117.w_4_1 ctx) (parallel_generatedRoot_117.w_4_2 ctx) (parallel_generatedRoot_117.w_4_3 ctx) (parallel_generatedRoot_117.w_5_0 ctx) (parallel_generatedRoot_117.w_6_0 ctx) (parallel_generatedRoot_117.w_7_0 ctx) (parallel_generatedRoot_117.w_8_0 ctx) (parallel_generatedRoot_117.w_9_0 ctx) (parallel_generatedRoot_117.w_10_0 ctx) (parallel_generatedRoot_117.w_11_0 ctx) (parallel_generatedRoot_117.w_12_0 ctx) (parallel_generatedRoot_117.w_13_0 ctx) (parallel_generatedRoot_117.w_14_0 ctx) (parallel_generatedRoot_117.w_15_0 ctx) (parallel_generatedRoot_117.w_16_0 ctx) (parallel_generatedRoot_117.w_17_0 ctx) (parallel_generatedRoot_117.w_17_1 ctx) (parallel_generatedRoot_117.w_17_2 ctx) (parallel_generatedRoot_117.w_17_3 ctx) outputs) :
    parallel_generatedRoot_117.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_117.Facts  (ctx : parallel_generatedRoot_117Ctx) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  c0 : parallel_generatedRoot_117.Facts_0 ctx outputs

theorem parallel_generatedRoot_117.facts  (ctx : parallel_generatedRoot_117Ctx) {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_117.body backend hashModel params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_117.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_117.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_121.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) where
  inputs : (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × Unit
  witness : parallel_generatedRoot_121.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_121Ctx" => (parallel_generatedRoot_121.Ctx backend hashModel params i_0)

def parallel_generatedRoot_121.w_0_0  (ctx : parallel_generatedRoot_121Ctx) : Fin 16 → mxxMat1 :=
  ctx.inputs.1

def parallel_generatedRoot_121.w_1_0  (ctx : parallel_generatedRoot_121Ctx) : Fin 16 → mxxMat1 :=
  ctx.inputs.2.1

def parallel_generatedRoot_121.w_2_0  (ctx : parallel_generatedRoot_121Ctx) : Fin 16 → mxxMat1 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_121.w_3_0  (ctx : parallel_generatedRoot_121Ctx) : Fin 16 → mxxMat1 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_121.w_4_0  (ctx : parallel_generatedRoot_121Ctx) : Fin 2 → mxxMat1 :=
  ctx.witness.w_4_0

def parallel_generatedRoot_121.w_4_1  (ctx : parallel_generatedRoot_121Ctx) : Fin 2 → mxxMat1 :=
  ctx.witness.w_4_1

def parallel_generatedRoot_121.w_4_2  (ctx : parallel_generatedRoot_121Ctx) : Fin 2 → mxxMat1 :=
  ctx.witness.w_4_2

def parallel_generatedRoot_121.w_4_3  (ctx : parallel_generatedRoot_121Ctx) : Fin 2 → mxxMat1 :=
  ctx.witness.w_4_3

def parallel_generatedRoot_121.w_5_0  (ctx : parallel_generatedRoot_121Ctx) : mxxMat1 :=
  ctx.witness.w_5_0

def parallel_generatedRoot_121.w_6_0  (ctx : parallel_generatedRoot_121Ctx) : mxxMat1 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_121.w_7_0  (ctx : parallel_generatedRoot_121Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_121.w_5_0 ctx) (parallel_generatedRoot_121.w_6_0 ctx)

def parallel_generatedRoot_121.w_8_0  (ctx : parallel_generatedRoot_121Ctx) : mxxMat1 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_121.w_9_0  (ctx : parallel_generatedRoot_121Ctx) : mxxMat1 :=
  ctx.witness.w_9_0

def parallel_generatedRoot_121.w_10_0  (ctx : parallel_generatedRoot_121Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_121.w_8_0 ctx) (parallel_generatedRoot_121.w_9_0 ctx)

def parallel_generatedRoot_121.w_11_0  (ctx : parallel_generatedRoot_121Ctx) : mxxMat1 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_121.w_12_0  (ctx : parallel_generatedRoot_121Ctx) : mxxMat1 :=
  ctx.witness.w_12_0

def parallel_generatedRoot_121.w_13_0  (ctx : parallel_generatedRoot_121Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_121.w_11_0 ctx) (parallel_generatedRoot_121.w_12_0 ctx)

def parallel_generatedRoot_121.w_14_0  (ctx : parallel_generatedRoot_121Ctx) : mxxMat1 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_121.w_15_0  (ctx : parallel_generatedRoot_121Ctx) : mxxMat1 :=
  ctx.witness.w_15_0

def parallel_generatedRoot_121.w_16_0  (ctx : parallel_generatedRoot_121Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_121.w_14_0 ctx) (parallel_generatedRoot_121.w_15_0 ctx)

def parallel_generatedRoot_121.w_17_0  (ctx : parallel_generatedRoot_121Ctx) : mxxMat1 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_121.w_17_1  (ctx : parallel_generatedRoot_121Ctx) : mxxMat1 :=
  ctx.witness.w_17_1

def parallel_generatedRoot_121.w_17_2  (ctx : parallel_generatedRoot_121Ctx) : mxxMat1 :=
  ctx.witness.w_17_2

def parallel_generatedRoot_121.w_17_3  (ctx : parallel_generatedRoot_121Ctx) : mxxMat1 :=
  ctx.witness.w_17_3

def parallel_generatedRoot_121.outs  (ctx : parallel_generatedRoot_121Ctx) : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit :=
  ((parallel_generatedRoot_121.w_17_0 ctx), ((parallel_generatedRoot_121.w_17_1 ctx), ((parallel_generatedRoot_121.w_17_2 ctx), ((parallel_generatedRoot_121.w_17_3 ctx), ()))))

structure parallel_generatedRoot_121.Facts_0  (ctx : parallel_generatedRoot_121Ctx) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
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

theorem parallel_generatedRoot_121.facts_0  {ctx : parallel_generatedRoot_121Ctx} {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_121.constraints_0 backend hashModel params i_0 (parallel_generatedRoot_121.w_0_0 ctx) (parallel_generatedRoot_121.w_1_0 ctx) (parallel_generatedRoot_121.w_2_0 ctx) (parallel_generatedRoot_121.w_3_0 ctx) (parallel_generatedRoot_121.w_4_0 ctx) (parallel_generatedRoot_121.w_4_1 ctx) (parallel_generatedRoot_121.w_4_2 ctx) (parallel_generatedRoot_121.w_4_3 ctx) (parallel_generatedRoot_121.w_5_0 ctx) (parallel_generatedRoot_121.w_6_0 ctx) (parallel_generatedRoot_121.w_7_0 ctx) (parallel_generatedRoot_121.w_8_0 ctx) (parallel_generatedRoot_121.w_9_0 ctx) (parallel_generatedRoot_121.w_10_0 ctx) (parallel_generatedRoot_121.w_11_0 ctx) (parallel_generatedRoot_121.w_12_0 ctx) (parallel_generatedRoot_121.w_13_0 ctx) (parallel_generatedRoot_121.w_14_0 ctx) (parallel_generatedRoot_121.w_15_0 ctx) (parallel_generatedRoot_121.w_16_0 ctx) (parallel_generatedRoot_121.w_17_0 ctx) (parallel_generatedRoot_121.w_17_1 ctx) (parallel_generatedRoot_121.w_17_2 ctx) (parallel_generatedRoot_121.w_17_3 ctx) outputs) :
    parallel_generatedRoot_121.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_121.Facts  (ctx : parallel_generatedRoot_121Ctx) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  c0 : parallel_generatedRoot_121.Facts_0 ctx outputs

theorem parallel_generatedRoot_121.facts  (ctx : parallel_generatedRoot_121Ctx) {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_121.body backend hashModel params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_121.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_121.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_122.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) where
  inputs : ByteArray × mxxMat1 × mxxMat1 × (Fin 8 → mxxMat1) × mxxMat1 × mxxMat1 × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × Unit
  witness : parallel_generatedRoot_122.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_122Ctx" => (parallel_generatedRoot_122.Ctx backend hashModel params i_0)

def parallel_generatedRoot_122.w_1_0  (ctx : parallel_generatedRoot_122Ctx) : ByteArray :=
  ctx.inputs.1

def parallel_generatedRoot_122.w_2_0  (ctx : parallel_generatedRoot_122Ctx) : mxxMat1 :=
  ctx.inputs.2.1

def parallel_generatedRoot_122.w_3_0  (ctx : parallel_generatedRoot_122Ctx) : mxxMat1 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_122.w_4_0  (ctx : parallel_generatedRoot_122Ctx) : Fin 8 → mxxMat1 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_122.w_5_0  (ctx : parallel_generatedRoot_122Ctx) : mxxMat1 :=
  ctx.inputs.2.2.2.2.1

def parallel_generatedRoot_122.w_6_0  (ctx : parallel_generatedRoot_122Ctx) : mxxMat1 :=
  ctx.inputs.2.2.2.2.2.1

def parallel_generatedRoot_122.w_7_0  (ctx : parallel_generatedRoot_122Ctx) : Fin 8 → mxxMat1 :=
  ctx.inputs.2.2.2.2.2.2.1

def parallel_generatedRoot_122.w_8_0  (ctx : parallel_generatedRoot_122Ctx) : Fin 8 → mxxMat1 :=
  ctx.inputs.2.2.2.2.2.2.2.1

def parallel_generatedRoot_122.w_9_0  (ctx : parallel_generatedRoot_122Ctx) : Fin 8 → mxxMat1 :=
  ctx.inputs.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_122.w_0_0  (ctx : parallel_generatedRoot_122Ctx) : mxxMat1 :=
  (0 : mxxMat1)

def parallel_generatedRoot_122.w_10_0  (ctx : parallel_generatedRoot_122Ctx) : mxxMat1 :=
  ctx.witness.w_10_0

def parallel_generatedRoot_122.w_10_1  (ctx : parallel_generatedRoot_122Ctx) : mxxMat1 :=
  ctx.witness.w_10_1

def parallel_generatedRoot_122.w_10_2  (ctx : parallel_generatedRoot_122Ctx) : mxxMat1 :=
  ctx.witness.w_10_2

def parallel_generatedRoot_122.w_10_3  (ctx : parallel_generatedRoot_122Ctx) : mxxMat1 :=
  ctx.witness.w_10_3

def parallel_generatedRoot_122.outs  (ctx : parallel_generatedRoot_122Ctx) : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit :=
  ((parallel_generatedRoot_122.w_10_0 ctx), ((parallel_generatedRoot_122.w_10_1 ctx), ((parallel_generatedRoot_122.w_10_2 ctx), ((parallel_generatedRoot_122.w_10_3 ctx), ()))))

structure parallel_generatedRoot_122.Facts_0  (ctx : parallel_generatedRoot_122Ctx) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  t0 : 0 ≤ (8)
  h10_3 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_parallel_generatedRoot_122_10 backend hashModel params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, ((parallel_generatedRoot_122.w_1_0 ctx), ((parallel_generatedRoot_122.w_2_0 ctx), ((parallel_generatedRoot_122.w_3_0 ctx), ((parallel_generatedRoot_122.w_4_0 ctx), ((parallel_generatedRoot_122.w_5_0 ctx), ((parallel_generatedRoot_122.w_6_0 ctx), ((parallel_generatedRoot_122.w_7_0 ctx), ((parallel_generatedRoot_122.w_8_0 ctx), ((parallel_generatedRoot_122.w_9_0 ctx), ()))))))))))))) next) (Int.toNat (8)) ((parallel_generatedRoot_122.w_0_0 ctx), ((parallel_generatedRoot_122.w_0_0 ctx), ((parallel_generatedRoot_122.w_0_0 ctx), ((parallel_generatedRoot_122.w_0_0 ctx), ())))) ((parallel_generatedRoot_122.w_10_0 ctx), ((parallel_generatedRoot_122.w_10_1 ctx), ((parallel_generatedRoot_122.w_10_2 ctx), ((parallel_generatedRoot_122.w_10_3 ctx), ()))))
  hout : outputs = parallel_generatedRoot_122.outs ctx

theorem parallel_generatedRoot_122.facts_0  {ctx : parallel_generatedRoot_122Ctx} {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_122.constraints_0 backend hashModel params i_0 (parallel_generatedRoot_122.w_1_0 ctx) (parallel_generatedRoot_122.w_2_0 ctx) (parallel_generatedRoot_122.w_3_0 ctx) (parallel_generatedRoot_122.w_4_0 ctx) (parallel_generatedRoot_122.w_5_0 ctx) (parallel_generatedRoot_122.w_6_0 ctx) (parallel_generatedRoot_122.w_7_0 ctx) (parallel_generatedRoot_122.w_8_0 ctx) (parallel_generatedRoot_122.w_9_0 ctx) (parallel_generatedRoot_122.w_0_0 ctx) (parallel_generatedRoot_122.w_10_0 ctx) (parallel_generatedRoot_122.w_10_1 ctx) (parallel_generatedRoot_122.w_10_2 ctx) (parallel_generatedRoot_122.w_10_3 ctx) outputs) :
    parallel_generatedRoot_122.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure parallel_generatedRoot_122.Facts  (ctx : parallel_generatedRoot_122Ctx) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  c0 : parallel_generatedRoot_122.Facts_0 ctx outputs

theorem parallel_generatedRoot_122.facts  (ctx : parallel_generatedRoot_122Ctx) {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_122.body backend hashModel params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_122.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_122.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_123.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) where
  inputs : ByteArray × mxxMat1 × mxxMat1 × (Fin 8 → mxxMat1) × mxxMat1 × mxxMat1 × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × Unit
  witness : parallel_generatedRoot_123.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_123Ctx" => (parallel_generatedRoot_123.Ctx backend hashModel params i_0)

def parallel_generatedRoot_123.w_1_0  (ctx : parallel_generatedRoot_123Ctx) : ByteArray :=
  ctx.inputs.1

def parallel_generatedRoot_123.w_2_0  (ctx : parallel_generatedRoot_123Ctx) : mxxMat1 :=
  ctx.inputs.2.1

def parallel_generatedRoot_123.w_3_0  (ctx : parallel_generatedRoot_123Ctx) : mxxMat1 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_123.w_4_0  (ctx : parallel_generatedRoot_123Ctx) : Fin 8 → mxxMat1 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_123.w_5_0  (ctx : parallel_generatedRoot_123Ctx) : mxxMat1 :=
  ctx.inputs.2.2.2.2.1

def parallel_generatedRoot_123.w_6_0  (ctx : parallel_generatedRoot_123Ctx) : mxxMat1 :=
  ctx.inputs.2.2.2.2.2.1

def parallel_generatedRoot_123.w_7_0  (ctx : parallel_generatedRoot_123Ctx) : Fin 8 → mxxMat1 :=
  ctx.inputs.2.2.2.2.2.2.1

def parallel_generatedRoot_123.w_8_0  (ctx : parallel_generatedRoot_123Ctx) : Fin 8 → mxxMat1 :=
  ctx.inputs.2.2.2.2.2.2.2.1

def parallel_generatedRoot_123.w_9_0  (ctx : parallel_generatedRoot_123Ctx) : Fin 8 → mxxMat1 :=
  ctx.inputs.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_123.w_0_0  (ctx : parallel_generatedRoot_123Ctx) : mxxMat1 :=
  (0 : mxxMat1)

def parallel_generatedRoot_123.w_10_0  (ctx : parallel_generatedRoot_123Ctx) : mxxMat1 :=
  ctx.witness.w_10_0

def parallel_generatedRoot_123.w_10_1  (ctx : parallel_generatedRoot_123Ctx) : mxxMat1 :=
  ctx.witness.w_10_1

def parallel_generatedRoot_123.w_10_2  (ctx : parallel_generatedRoot_123Ctx) : mxxMat1 :=
  ctx.witness.w_10_2

def parallel_generatedRoot_123.w_10_3  (ctx : parallel_generatedRoot_123Ctx) : mxxMat1 :=
  ctx.witness.w_10_3

def parallel_generatedRoot_123.outs  (ctx : parallel_generatedRoot_123Ctx) : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit :=
  ((parallel_generatedRoot_123.w_10_0 ctx), ((parallel_generatedRoot_123.w_10_1 ctx), ((parallel_generatedRoot_123.w_10_2 ctx), ((parallel_generatedRoot_123.w_10_3 ctx), ()))))

structure parallel_generatedRoot_123.Facts_0  (ctx : parallel_generatedRoot_123Ctx) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  t0 : 0 ≤ (8)
  h10_3 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_parallel_generatedRoot_123_10 backend hashModel params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, ((parallel_generatedRoot_123.w_1_0 ctx), ((parallel_generatedRoot_123.w_2_0 ctx), ((parallel_generatedRoot_123.w_3_0 ctx), ((parallel_generatedRoot_123.w_4_0 ctx), ((parallel_generatedRoot_123.w_5_0 ctx), ((parallel_generatedRoot_123.w_6_0 ctx), ((parallel_generatedRoot_123.w_7_0 ctx), ((parallel_generatedRoot_123.w_8_0 ctx), ((parallel_generatedRoot_123.w_9_0 ctx), ()))))))))))))) next) (Int.toNat (8)) ((parallel_generatedRoot_123.w_0_0 ctx), ((parallel_generatedRoot_123.w_0_0 ctx), ((parallel_generatedRoot_123.w_0_0 ctx), ((parallel_generatedRoot_123.w_0_0 ctx), ())))) ((parallel_generatedRoot_123.w_10_0 ctx), ((parallel_generatedRoot_123.w_10_1 ctx), ((parallel_generatedRoot_123.w_10_2 ctx), ((parallel_generatedRoot_123.w_10_3 ctx), ()))))
  hout : outputs = parallel_generatedRoot_123.outs ctx

theorem parallel_generatedRoot_123.facts_0  {ctx : parallel_generatedRoot_123Ctx} {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_123.constraints_0 backend hashModel params i_0 (parallel_generatedRoot_123.w_1_0 ctx) (parallel_generatedRoot_123.w_2_0 ctx) (parallel_generatedRoot_123.w_3_0 ctx) (parallel_generatedRoot_123.w_4_0 ctx) (parallel_generatedRoot_123.w_5_0 ctx) (parallel_generatedRoot_123.w_6_0 ctx) (parallel_generatedRoot_123.w_7_0 ctx) (parallel_generatedRoot_123.w_8_0 ctx) (parallel_generatedRoot_123.w_9_0 ctx) (parallel_generatedRoot_123.w_0_0 ctx) (parallel_generatedRoot_123.w_10_0 ctx) (parallel_generatedRoot_123.w_10_1 ctx) (parallel_generatedRoot_123.w_10_2 ctx) (parallel_generatedRoot_123.w_10_3 ctx) outputs) :
    parallel_generatedRoot_123.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure parallel_generatedRoot_123.Facts  (ctx : parallel_generatedRoot_123Ctx) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  c0 : parallel_generatedRoot_123.Facts_0 ctx outputs

theorem parallel_generatedRoot_123.facts  (ctx : parallel_generatedRoot_123Ctx) {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_123.body backend hashModel params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_123.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_123.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_162.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) where
  inputs : (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × Unit
  witness : parallel_generatedRoot_162.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_162Ctx" => (parallel_generatedRoot_162.Ctx backend hashModel params i_0)

def parallel_generatedRoot_162.w_0_0  (ctx : parallel_generatedRoot_162Ctx) : Fin 14400 → mxxMat1 :=
  ctx.inputs.1

def parallel_generatedRoot_162.w_1_0  (ctx : parallel_generatedRoot_162Ctx) : Fin 14400 → mxxMat1 :=
  ctx.inputs.2.1

def parallel_generatedRoot_162.w_2_0  (ctx : parallel_generatedRoot_162Ctx) : Fin 14400 → mxxMat1 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_162.w_3_0  (ctx : parallel_generatedRoot_162Ctx) : Fin 14400 → mxxMat1 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_162.w_4_0  (ctx : parallel_generatedRoot_162Ctx) : Fin 2 → mxxMat1 :=
  ctx.witness.w_4_0

def parallel_generatedRoot_162.w_4_1  (ctx : parallel_generatedRoot_162Ctx) : Fin 2 → mxxMat1 :=
  ctx.witness.w_4_1

def parallel_generatedRoot_162.w_4_2  (ctx : parallel_generatedRoot_162Ctx) : Fin 2 → mxxMat1 :=
  ctx.witness.w_4_2

def parallel_generatedRoot_162.w_4_3  (ctx : parallel_generatedRoot_162Ctx) : Fin 2 → mxxMat1 :=
  ctx.witness.w_4_3

def parallel_generatedRoot_162.w_5_0  (ctx : parallel_generatedRoot_162Ctx) : mxxMat1 :=
  ctx.witness.w_5_0

def parallel_generatedRoot_162.w_6_0  (ctx : parallel_generatedRoot_162Ctx) : mxxMat1 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_162.w_7_0  (ctx : parallel_generatedRoot_162Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_162.w_5_0 ctx) (parallel_generatedRoot_162.w_6_0 ctx)

def parallel_generatedRoot_162.w_8_0  (ctx : parallel_generatedRoot_162Ctx) : mxxMat1 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_162.w_9_0  (ctx : parallel_generatedRoot_162Ctx) : mxxMat1 :=
  ctx.witness.w_9_0

def parallel_generatedRoot_162.w_10_0  (ctx : parallel_generatedRoot_162Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_162.w_8_0 ctx) (parallel_generatedRoot_162.w_9_0 ctx)

def parallel_generatedRoot_162.w_11_0  (ctx : parallel_generatedRoot_162Ctx) : mxxMat1 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_162.w_12_0  (ctx : parallel_generatedRoot_162Ctx) : mxxMat1 :=
  ctx.witness.w_12_0

def parallel_generatedRoot_162.w_13_0  (ctx : parallel_generatedRoot_162Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_162.w_11_0 ctx) (parallel_generatedRoot_162.w_12_0 ctx)

def parallel_generatedRoot_162.w_14_0  (ctx : parallel_generatedRoot_162Ctx) : mxxMat1 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_162.w_15_0  (ctx : parallel_generatedRoot_162Ctx) : mxxMat1 :=
  ctx.witness.w_15_0

def parallel_generatedRoot_162.w_16_0  (ctx : parallel_generatedRoot_162Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_162.w_14_0 ctx) (parallel_generatedRoot_162.w_15_0 ctx)

def parallel_generatedRoot_162.w_17_0  (ctx : parallel_generatedRoot_162Ctx) : mxxMat1 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_162.w_17_1  (ctx : parallel_generatedRoot_162Ctx) : mxxMat1 :=
  ctx.witness.w_17_1

def parallel_generatedRoot_162.w_17_2  (ctx : parallel_generatedRoot_162Ctx) : mxxMat1 :=
  ctx.witness.w_17_2

def parallel_generatedRoot_162.w_17_3  (ctx : parallel_generatedRoot_162Ctx) : mxxMat1 :=
  ctx.witness.w_17_3

def parallel_generatedRoot_162.outs  (ctx : parallel_generatedRoot_162Ctx) : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit :=
  ((parallel_generatedRoot_162.w_17_0 ctx), ((parallel_generatedRoot_162.w_17_1 ctx), ((parallel_generatedRoot_162.w_17_2 ctx), ((parallel_generatedRoot_162.w_17_3 ctx), ()))))

structure parallel_generatedRoot_162.Facts_0  (ctx : parallel_generatedRoot_162Ctx) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
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

theorem parallel_generatedRoot_162.facts_0  {ctx : parallel_generatedRoot_162Ctx} {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_162.constraints_0 backend hashModel params i_0 (parallel_generatedRoot_162.w_0_0 ctx) (parallel_generatedRoot_162.w_1_0 ctx) (parallel_generatedRoot_162.w_2_0 ctx) (parallel_generatedRoot_162.w_3_0 ctx) (parallel_generatedRoot_162.w_4_0 ctx) (parallel_generatedRoot_162.w_4_1 ctx) (parallel_generatedRoot_162.w_4_2 ctx) (parallel_generatedRoot_162.w_4_3 ctx) (parallel_generatedRoot_162.w_5_0 ctx) (parallel_generatedRoot_162.w_6_0 ctx) (parallel_generatedRoot_162.w_7_0 ctx) (parallel_generatedRoot_162.w_8_0 ctx) (parallel_generatedRoot_162.w_9_0 ctx) (parallel_generatedRoot_162.w_10_0 ctx) (parallel_generatedRoot_162.w_11_0 ctx) (parallel_generatedRoot_162.w_12_0 ctx) (parallel_generatedRoot_162.w_13_0 ctx) (parallel_generatedRoot_162.w_14_0 ctx) (parallel_generatedRoot_162.w_15_0 ctx) (parallel_generatedRoot_162.w_16_0 ctx) (parallel_generatedRoot_162.w_17_0 ctx) (parallel_generatedRoot_162.w_17_1 ctx) (parallel_generatedRoot_162.w_17_2 ctx) (parallel_generatedRoot_162.w_17_3 ctx) outputs) :
    parallel_generatedRoot_162.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_162.Facts  (ctx : parallel_generatedRoot_162Ctx) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  c0 : parallel_generatedRoot_162.Facts_0 ctx outputs

theorem parallel_generatedRoot_162.facts  (ctx : parallel_generatedRoot_162Ctx) {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_162.body backend hashModel params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_162.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_162.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_166.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) where
  inputs : (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × Unit
  witness : parallel_generatedRoot_166.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_166Ctx" => (parallel_generatedRoot_166.Ctx backend hashModel params i_0)

def parallel_generatedRoot_166.w_0_0  (ctx : parallel_generatedRoot_166Ctx) : Fin 14400 → mxxMat1 :=
  ctx.inputs.1

def parallel_generatedRoot_166.w_1_0  (ctx : parallel_generatedRoot_166Ctx) : Fin 14400 → mxxMat1 :=
  ctx.inputs.2.1

def parallel_generatedRoot_166.w_2_0  (ctx : parallel_generatedRoot_166Ctx) : Fin 14400 → mxxMat1 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_166.w_3_0  (ctx : parallel_generatedRoot_166Ctx) : Fin 14400 → mxxMat1 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_166.w_4_0  (ctx : parallel_generatedRoot_166Ctx) : Fin 2 → mxxMat1 :=
  ctx.witness.w_4_0

def parallel_generatedRoot_166.w_4_1  (ctx : parallel_generatedRoot_166Ctx) : Fin 2 → mxxMat1 :=
  ctx.witness.w_4_1

def parallel_generatedRoot_166.w_4_2  (ctx : parallel_generatedRoot_166Ctx) : Fin 2 → mxxMat1 :=
  ctx.witness.w_4_2

def parallel_generatedRoot_166.w_4_3  (ctx : parallel_generatedRoot_166Ctx) : Fin 2 → mxxMat1 :=
  ctx.witness.w_4_3

def parallel_generatedRoot_166.w_5_0  (ctx : parallel_generatedRoot_166Ctx) : mxxMat1 :=
  ctx.witness.w_5_0

def parallel_generatedRoot_166.w_6_0  (ctx : parallel_generatedRoot_166Ctx) : mxxMat1 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_166.w_7_0  (ctx : parallel_generatedRoot_166Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_166.w_5_0 ctx) (parallel_generatedRoot_166.w_6_0 ctx)

def parallel_generatedRoot_166.w_8_0  (ctx : parallel_generatedRoot_166Ctx) : mxxMat1 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_166.w_9_0  (ctx : parallel_generatedRoot_166Ctx) : mxxMat1 :=
  ctx.witness.w_9_0

def parallel_generatedRoot_166.w_10_0  (ctx : parallel_generatedRoot_166Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_166.w_8_0 ctx) (parallel_generatedRoot_166.w_9_0 ctx)

def parallel_generatedRoot_166.w_11_0  (ctx : parallel_generatedRoot_166Ctx) : mxxMat1 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_166.w_12_0  (ctx : parallel_generatedRoot_166Ctx) : mxxMat1 :=
  ctx.witness.w_12_0

def parallel_generatedRoot_166.w_13_0  (ctx : parallel_generatedRoot_166Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_166.w_11_0 ctx) (parallel_generatedRoot_166.w_12_0 ctx)

def parallel_generatedRoot_166.w_14_0  (ctx : parallel_generatedRoot_166Ctx) : mxxMat1 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_166.w_15_0  (ctx : parallel_generatedRoot_166Ctx) : mxxMat1 :=
  ctx.witness.w_15_0

def parallel_generatedRoot_166.w_16_0  (ctx : parallel_generatedRoot_166Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_166.w_14_0 ctx) (parallel_generatedRoot_166.w_15_0 ctx)

def parallel_generatedRoot_166.w_17_0  (ctx : parallel_generatedRoot_166Ctx) : mxxMat1 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_166.w_17_1  (ctx : parallel_generatedRoot_166Ctx) : mxxMat1 :=
  ctx.witness.w_17_1

def parallel_generatedRoot_166.w_17_2  (ctx : parallel_generatedRoot_166Ctx) : mxxMat1 :=
  ctx.witness.w_17_2

def parallel_generatedRoot_166.w_17_3  (ctx : parallel_generatedRoot_166Ctx) : mxxMat1 :=
  ctx.witness.w_17_3

def parallel_generatedRoot_166.outs  (ctx : parallel_generatedRoot_166Ctx) : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit :=
  ((parallel_generatedRoot_166.w_17_0 ctx), ((parallel_generatedRoot_166.w_17_1 ctx), ((parallel_generatedRoot_166.w_17_2 ctx), ((parallel_generatedRoot_166.w_17_3 ctx), ()))))

structure parallel_generatedRoot_166.Facts_0  (ctx : parallel_generatedRoot_166Ctx) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
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

theorem parallel_generatedRoot_166.facts_0  {ctx : parallel_generatedRoot_166Ctx} {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_166.constraints_0 backend hashModel params i_0 (parallel_generatedRoot_166.w_0_0 ctx) (parallel_generatedRoot_166.w_1_0 ctx) (parallel_generatedRoot_166.w_2_0 ctx) (parallel_generatedRoot_166.w_3_0 ctx) (parallel_generatedRoot_166.w_4_0 ctx) (parallel_generatedRoot_166.w_4_1 ctx) (parallel_generatedRoot_166.w_4_2 ctx) (parallel_generatedRoot_166.w_4_3 ctx) (parallel_generatedRoot_166.w_5_0 ctx) (parallel_generatedRoot_166.w_6_0 ctx) (parallel_generatedRoot_166.w_7_0 ctx) (parallel_generatedRoot_166.w_8_0 ctx) (parallel_generatedRoot_166.w_9_0 ctx) (parallel_generatedRoot_166.w_10_0 ctx) (parallel_generatedRoot_166.w_11_0 ctx) (parallel_generatedRoot_166.w_12_0 ctx) (parallel_generatedRoot_166.w_13_0 ctx) (parallel_generatedRoot_166.w_14_0 ctx) (parallel_generatedRoot_166.w_15_0 ctx) (parallel_generatedRoot_166.w_16_0 ctx) (parallel_generatedRoot_166.w_17_0 ctx) (parallel_generatedRoot_166.w_17_1 ctx) (parallel_generatedRoot_166.w_17_2 ctx) (parallel_generatedRoot_166.w_17_3 ctx) outputs) :
    parallel_generatedRoot_166.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_166.Facts  (ctx : parallel_generatedRoot_166Ctx) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  c0 : parallel_generatedRoot_166.Facts_0 ctx outputs

theorem parallel_generatedRoot_166.facts  (ctx : parallel_generatedRoot_166Ctx) {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_166.body backend hashModel params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_166.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_166.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_167.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) where
  inputs : ByteArray × mxxMat1 × mxxMat1 × (Fin 8 → mxxMat1) × mxxMat1 × mxxMat1 × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × Unit
  witness : parallel_generatedRoot_167.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_167Ctx" => (parallel_generatedRoot_167.Ctx backend hashModel params i_0)

def parallel_generatedRoot_167.w_1_0  (ctx : parallel_generatedRoot_167Ctx) : ByteArray :=
  ctx.inputs.1

def parallel_generatedRoot_167.w_2_0  (ctx : parallel_generatedRoot_167Ctx) : mxxMat1 :=
  ctx.inputs.2.1

def parallel_generatedRoot_167.w_3_0  (ctx : parallel_generatedRoot_167Ctx) : mxxMat1 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_167.w_4_0  (ctx : parallel_generatedRoot_167Ctx) : Fin 8 → mxxMat1 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_167.w_5_0  (ctx : parallel_generatedRoot_167Ctx) : mxxMat1 :=
  ctx.inputs.2.2.2.2.1

def parallel_generatedRoot_167.w_6_0  (ctx : parallel_generatedRoot_167Ctx) : mxxMat1 :=
  ctx.inputs.2.2.2.2.2.1

def parallel_generatedRoot_167.w_7_0  (ctx : parallel_generatedRoot_167Ctx) : Fin 8 → mxxMat1 :=
  ctx.inputs.2.2.2.2.2.2.1

def parallel_generatedRoot_167.w_8_0  (ctx : parallel_generatedRoot_167Ctx) : Fin 8 → mxxMat1 :=
  ctx.inputs.2.2.2.2.2.2.2.1

def parallel_generatedRoot_167.w_9_0  (ctx : parallel_generatedRoot_167Ctx) : Fin 8 → mxxMat1 :=
  ctx.inputs.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_167.w_0_0  (ctx : parallel_generatedRoot_167Ctx) : mxxMat1 :=
  (0 : mxxMat1)

def parallel_generatedRoot_167.w_10_0  (ctx : parallel_generatedRoot_167Ctx) : mxxMat1 :=
  ctx.witness.w_10_0

def parallel_generatedRoot_167.w_10_1  (ctx : parallel_generatedRoot_167Ctx) : mxxMat1 :=
  ctx.witness.w_10_1

def parallel_generatedRoot_167.w_10_2  (ctx : parallel_generatedRoot_167Ctx) : mxxMat1 :=
  ctx.witness.w_10_2

def parallel_generatedRoot_167.w_10_3  (ctx : parallel_generatedRoot_167Ctx) : mxxMat1 :=
  ctx.witness.w_10_3

def parallel_generatedRoot_167.outs  (ctx : parallel_generatedRoot_167Ctx) : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit :=
  ((parallel_generatedRoot_167.w_10_0 ctx), ((parallel_generatedRoot_167.w_10_1 ctx), ((parallel_generatedRoot_167.w_10_2 ctx), ((parallel_generatedRoot_167.w_10_3 ctx), ()))))

structure parallel_generatedRoot_167.Facts_0  (ctx : parallel_generatedRoot_167Ctx) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  t0 : 0 ≤ (8)
  h10_3 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_parallel_generatedRoot_167_10 backend hashModel params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, ((parallel_generatedRoot_167.w_1_0 ctx), ((parallel_generatedRoot_167.w_2_0 ctx), ((parallel_generatedRoot_167.w_3_0 ctx), ((parallel_generatedRoot_167.w_4_0 ctx), ((parallel_generatedRoot_167.w_5_0 ctx), ((parallel_generatedRoot_167.w_6_0 ctx), ((parallel_generatedRoot_167.w_7_0 ctx), ((parallel_generatedRoot_167.w_8_0 ctx), ((parallel_generatedRoot_167.w_9_0 ctx), ()))))))))))))) next) (Int.toNat (8)) ((parallel_generatedRoot_167.w_0_0 ctx), ((parallel_generatedRoot_167.w_0_0 ctx), ((parallel_generatedRoot_167.w_0_0 ctx), ((parallel_generatedRoot_167.w_0_0 ctx), ())))) ((parallel_generatedRoot_167.w_10_0 ctx), ((parallel_generatedRoot_167.w_10_1 ctx), ((parallel_generatedRoot_167.w_10_2 ctx), ((parallel_generatedRoot_167.w_10_3 ctx), ()))))
  hout : outputs = parallel_generatedRoot_167.outs ctx

theorem parallel_generatedRoot_167.facts_0  {ctx : parallel_generatedRoot_167Ctx} {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_167.constraints_0 backend hashModel params i_0 (parallel_generatedRoot_167.w_1_0 ctx) (parallel_generatedRoot_167.w_2_0 ctx) (parallel_generatedRoot_167.w_3_0 ctx) (parallel_generatedRoot_167.w_4_0 ctx) (parallel_generatedRoot_167.w_5_0 ctx) (parallel_generatedRoot_167.w_6_0 ctx) (parallel_generatedRoot_167.w_7_0 ctx) (parallel_generatedRoot_167.w_8_0 ctx) (parallel_generatedRoot_167.w_9_0 ctx) (parallel_generatedRoot_167.w_0_0 ctx) (parallel_generatedRoot_167.w_10_0 ctx) (parallel_generatedRoot_167.w_10_1 ctx) (parallel_generatedRoot_167.w_10_2 ctx) (parallel_generatedRoot_167.w_10_3 ctx) outputs) :
    parallel_generatedRoot_167.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure parallel_generatedRoot_167.Facts  (ctx : parallel_generatedRoot_167Ctx) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop where
  c0 : parallel_generatedRoot_167.Facts_0 ctx outputs

theorem parallel_generatedRoot_167.facts  (ctx : parallel_generatedRoot_167Ctx) {outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit}
    (h : parallel_generatedRoot_167.body backend hashModel params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_167.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_167.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_189.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) where
  inputs : mxxMat4 × mxxMat4 × mxxMat4 × mxxMat4 × (Fin 7200 → mxxMat1) × (Fin 7200 → mxxMat1) × (Fin 7200 → mxxMat1) × (Fin 7200 → mxxMat1) × Unit
  witness : sequential_generatedRoot_189.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_189Ctx" => (sequential_generatedRoot_189.Ctx backend hashModel params i_0)

def sequential_generatedRoot_189.w_0_0  (ctx : sequential_generatedRoot_189Ctx) : mxxMat4 :=
  ctx.inputs.1

def sequential_generatedRoot_189.w_23_0  (ctx : sequential_generatedRoot_189Ctx) : mxxMat4 :=
  ctx.inputs.2.1

def sequential_generatedRoot_189.w_27_0  (ctx : sequential_generatedRoot_189Ctx) : mxxMat4 :=
  ctx.inputs.2.2.1

def sequential_generatedRoot_189.w_31_0  (ctx : sequential_generatedRoot_189Ctx) : mxxMat4 :=
  ctx.inputs.2.2.2.1

def sequential_generatedRoot_189.w_2_0  (ctx : sequential_generatedRoot_189Ctx) : Fin 7200 → mxxMat1 :=
  ctx.inputs.2.2.2.2.1

def sequential_generatedRoot_189.w_8_0  (ctx : sequential_generatedRoot_189Ctx) : Fin 7200 → mxxMat1 :=
  ctx.inputs.2.2.2.2.2.1

def sequential_generatedRoot_189.w_12_0  (ctx : sequential_generatedRoot_189Ctx) : Fin 7200 → mxxMat1 :=
  ctx.inputs.2.2.2.2.2.2.1

def sequential_generatedRoot_189.w_16_0  (ctx : sequential_generatedRoot_189Ctx) : Fin 7200 → mxxMat1 :=
  ctx.inputs.2.2.2.2.2.2.2.1

def sequential_generatedRoot_189.w_1_0  (ctx : sequential_generatedRoot_189Ctx) : mxxMat5 :=
  (MxxRuntime.unitRow ((1 + (Int.ofNat i_0))) : mxxMat5)

def sequential_generatedRoot_189.w_3_0  (ctx : sequential_generatedRoot_189Ctx) : Int :=
  (8 + (8 * (Int.ofNat i_0)))

def sequential_generatedRoot_189.w_4_0  (ctx : sequential_generatedRoot_189Ctx) : mxxMat1 :=
  ctx.witness.w_4_0

def sequential_generatedRoot_189.w_5_0  (ctx : sequential_generatedRoot_189Ctx) : Int :=
  (9 + (8 * (Int.ofNat i_0)))

def sequential_generatedRoot_189.w_6_0  (ctx : sequential_generatedRoot_189Ctx) : mxxMat1 :=
  ctx.witness.w_6_0

def sequential_generatedRoot_189.w_7_0  (ctx : sequential_generatedRoot_189Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (sequential_generatedRoot_189.w_4_0 ctx) (sequential_generatedRoot_189.w_6_0 ctx)

def sequential_generatedRoot_189.w_9_0  (ctx : sequential_generatedRoot_189Ctx) : mxxMat1 :=
  ctx.witness.w_9_0

def sequential_generatedRoot_189.w_10_0  (ctx : sequential_generatedRoot_189Ctx) : mxxMat1 :=
  ctx.witness.w_10_0

def sequential_generatedRoot_189.w_11_0  (ctx : sequential_generatedRoot_189Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (sequential_generatedRoot_189.w_9_0 ctx) (sequential_generatedRoot_189.w_10_0 ctx)

def sequential_generatedRoot_189.w_13_0  (ctx : sequential_generatedRoot_189Ctx) : mxxMat1 :=
  ctx.witness.w_13_0

def sequential_generatedRoot_189.w_14_0  (ctx : sequential_generatedRoot_189Ctx) : mxxMat1 :=
  ctx.witness.w_14_0

def sequential_generatedRoot_189.w_15_0  (ctx : sequential_generatedRoot_189Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (sequential_generatedRoot_189.w_13_0 ctx) (sequential_generatedRoot_189.w_14_0 ctx)

def sequential_generatedRoot_189.w_17_0  (ctx : sequential_generatedRoot_189Ctx) : mxxMat1 :=
  ctx.witness.w_17_0

def sequential_generatedRoot_189.w_18_0  (ctx : sequential_generatedRoot_189Ctx) : mxxMat1 :=
  ctx.witness.w_18_0

def sequential_generatedRoot_189.w_19_0  (ctx : sequential_generatedRoot_189Ctx) : mxxMat1 :=
  MxxRuntime.matrixAdd (sequential_generatedRoot_189.w_17_0 ctx) (sequential_generatedRoot_189.w_18_0 ctx)

def sequential_generatedRoot_189.w_20_0  (ctx : sequential_generatedRoot_189Ctx) : mxxMat1 :=
  ctx.witness.w_20_0

def sequential_generatedRoot_189.w_20_1  (ctx : sequential_generatedRoot_189Ctx) : mxxMat1 :=
  ctx.witness.w_20_1

def sequential_generatedRoot_189.w_20_2  (ctx : sequential_generatedRoot_189Ctx) : mxxMat1 :=
  ctx.witness.w_20_2

def sequential_generatedRoot_189.w_20_3  (ctx : sequential_generatedRoot_189Ctx) : mxxMat1 :=
  ctx.witness.w_20_3

def sequential_generatedRoot_189.w_21_0  (ctx : sequential_generatedRoot_189Ctx) : mxxMat4 :=
  ctx.witness.w_21_0

def sequential_generatedRoot_189.w_22_0  (ctx : sequential_generatedRoot_189Ctx) : mxxMat4 :=
  MxxRuntime.matrixAdd (sequential_generatedRoot_189.w_0_0 ctx) (sequential_generatedRoot_189.w_21_0 ctx)

def sequential_generatedRoot_189.w_24_0  (ctx : sequential_generatedRoot_189Ctx) : mxxMat5 :=
  (MxxRuntime.unitRow ((1 + (Int.ofNat i_0))) : mxxMat5)

def sequential_generatedRoot_189.w_25_0  (ctx : sequential_generatedRoot_189Ctx) : mxxMat4 :=
  ctx.witness.w_25_0

def sequential_generatedRoot_189.w_26_0  (ctx : sequential_generatedRoot_189Ctx) : mxxMat4 :=
  MxxRuntime.matrixAdd (sequential_generatedRoot_189.w_23_0 ctx) (sequential_generatedRoot_189.w_25_0 ctx)

def sequential_generatedRoot_189.w_28_0  (ctx : sequential_generatedRoot_189Ctx) : mxxMat5 :=
  (MxxRuntime.unitRow ((1 + (Int.ofNat i_0))) : mxxMat5)

def sequential_generatedRoot_189.w_29_0  (ctx : sequential_generatedRoot_189Ctx) : mxxMat4 :=
  ctx.witness.w_29_0

def sequential_generatedRoot_189.w_30_0  (ctx : sequential_generatedRoot_189Ctx) : mxxMat4 :=
  MxxRuntime.matrixAdd (sequential_generatedRoot_189.w_27_0 ctx) (sequential_generatedRoot_189.w_29_0 ctx)

def sequential_generatedRoot_189.w_32_0  (ctx : sequential_generatedRoot_189Ctx) : mxxMat5 :=
  (MxxRuntime.unitRow ((1 + (Int.ofNat i_0))) : mxxMat5)

def sequential_generatedRoot_189.w_33_0  (ctx : sequential_generatedRoot_189Ctx) : mxxMat4 :=
  ctx.witness.w_33_0

def sequential_generatedRoot_189.w_34_0  (ctx : sequential_generatedRoot_189Ctx) : mxxMat4 :=
  MxxRuntime.matrixAdd (sequential_generatedRoot_189.w_31_0 ctx) (sequential_generatedRoot_189.w_33_0 ctx)

def sequential_generatedRoot_189.outs  (ctx : sequential_generatedRoot_189Ctx) : mxxMat4 × mxxMat4 × mxxMat4 × mxxMat4 × Unit :=
  ((sequential_generatedRoot_189.w_22_0 ctx), ((sequential_generatedRoot_189.w_26_0 ctx), ((sequential_generatedRoot_189.w_30_0 ctx), ((sequential_generatedRoot_189.w_34_0 ctx), ()))))

structure sequential_generatedRoot_189.Facts_0  (ctx : sequential_generatedRoot_189Ctx) : Prop where
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
  h20_3 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => scope_90 backend hashModel params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, ((sequential_generatedRoot_189.w_2_0 ctx), ((sequential_generatedRoot_189.w_8_0 ctx), ((sequential_generatedRoot_189.w_12_0 ctx), ((sequential_generatedRoot_189.w_16_0 ctx), ())))))))) next) (Int.toNat (6)) ((sequential_generatedRoot_189.w_7_0 ctx), ((sequential_generatedRoot_189.w_11_0 ctx), ((sequential_generatedRoot_189.w_15_0 ctx), ((sequential_generatedRoot_189.w_19_0 ctx), ())))) ((sequential_generatedRoot_189.w_20_0 ctx), ((sequential_generatedRoot_189.w_20_1 ctx), ((sequential_generatedRoot_189.w_20_2 ctx), ((sequential_generatedRoot_189.w_20_3 ctx), ()))))
  h21_0 : MxxRuntime.tensorRuns (sequential_generatedRoot_189.w_1_0 ctx) (sequential_generatedRoot_189.w_20_0 ctx) (sequential_generatedRoot_189.w_21_0 ctx)
  t29 : 0 ≤ ((1 + (Int.ofNat i_0)))
  t30 : ((1 + (Int.ofNat i_0))) < 900
  h25_0 : MxxRuntime.tensorRuns (sequential_generatedRoot_189.w_24_0 ctx) (sequential_generatedRoot_189.w_20_1 ctx) (sequential_generatedRoot_189.w_25_0 ctx)

theorem sequential_generatedRoot_189.facts_0  {ctx : sequential_generatedRoot_189Ctx} {outputs : mxxMat4 × mxxMat4 × mxxMat4 × mxxMat4 × Unit}
    (h : sequential_generatedRoot_189.constraints_0 backend hashModel params i_0 (sequential_generatedRoot_189.w_2_0 ctx) (sequential_generatedRoot_189.w_8_0 ctx) (sequential_generatedRoot_189.w_12_0 ctx) (sequential_generatedRoot_189.w_16_0 ctx) (sequential_generatedRoot_189.w_1_0 ctx) (sequential_generatedRoot_189.w_3_0 ctx) (sequential_generatedRoot_189.w_4_0 ctx) (sequential_generatedRoot_189.w_5_0 ctx) (sequential_generatedRoot_189.w_6_0 ctx) (sequential_generatedRoot_189.w_7_0 ctx) (sequential_generatedRoot_189.w_9_0 ctx) (sequential_generatedRoot_189.w_10_0 ctx) (sequential_generatedRoot_189.w_11_0 ctx) (sequential_generatedRoot_189.w_13_0 ctx) (sequential_generatedRoot_189.w_14_0 ctx) (sequential_generatedRoot_189.w_15_0 ctx) (sequential_generatedRoot_189.w_17_0 ctx) (sequential_generatedRoot_189.w_18_0 ctx) (sequential_generatedRoot_189.w_19_0 ctx) (sequential_generatedRoot_189.w_20_0 ctx) (sequential_generatedRoot_189.w_20_1 ctx) (sequential_generatedRoot_189.w_20_2 ctx) (sequential_generatedRoot_189.w_20_3 ctx) (sequential_generatedRoot_189.w_21_0 ctx) (sequential_generatedRoot_189.w_24_0 ctx) (sequential_generatedRoot_189.w_25_0 ctx)) :
    sequential_generatedRoot_189.Facts_0 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure sequential_generatedRoot_189.Facts_1  (ctx : sequential_generatedRoot_189Ctx) (outputs : mxxMat4 × mxxMat4 × mxxMat4 × mxxMat4 × Unit) : Prop where
  t32 : 0 ≤ ((1 + (Int.ofNat i_0)))
  t33 : ((1 + (Int.ofNat i_0))) < 900
  h29_0 : MxxRuntime.tensorRuns (sequential_generatedRoot_189.w_28_0 ctx) (sequential_generatedRoot_189.w_20_2 ctx) (sequential_generatedRoot_189.w_29_0 ctx)
  t35 : 0 ≤ ((1 + (Int.ofNat i_0)))
  t36 : ((1 + (Int.ofNat i_0))) < 900
  h33_0 : MxxRuntime.tensorRuns (sequential_generatedRoot_189.w_32_0 ctx) (sequential_generatedRoot_189.w_20_3 ctx) (sequential_generatedRoot_189.w_33_0 ctx)
  hout : outputs = sequential_generatedRoot_189.outs ctx

theorem sequential_generatedRoot_189.facts_1  {ctx : sequential_generatedRoot_189Ctx} {outputs : mxxMat4 × mxxMat4 × mxxMat4 × mxxMat4 × Unit}
    (h : sequential_generatedRoot_189.constraints_1 i_0 (sequential_generatedRoot_189.w_20_2 ctx) (sequential_generatedRoot_189.w_20_3 ctx) (sequential_generatedRoot_189.w_22_0 ctx) (sequential_generatedRoot_189.w_26_0 ctx) (sequential_generatedRoot_189.w_28_0 ctx) (sequential_generatedRoot_189.w_29_0 ctx) (sequential_generatedRoot_189.w_30_0 ctx) (sequential_generatedRoot_189.w_32_0 ctx) (sequential_generatedRoot_189.w_33_0 ctx) (sequential_generatedRoot_189.w_34_0 ctx) outputs) :
    sequential_generatedRoot_189.Facts_1 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2⟩

structure sequential_generatedRoot_189.Facts  (ctx : sequential_generatedRoot_189Ctx) (outputs : mxxMat4 × mxxMat4 × mxxMat4 × mxxMat4 × Unit) : Prop where
  c0 : sequential_generatedRoot_189.Facts_0 ctx
  c1 : sequential_generatedRoot_189.Facts_1 ctx outputs

theorem sequential_generatedRoot_189.facts  (ctx : sequential_generatedRoot_189Ctx) {outputs : mxxMat4 × mxxMat4 × mxxMat4 × mxxMat4 × Unit}
    (h : sequential_generatedRoot_189.body backend hashModel params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_189.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_189.facts_0 (ctx := ctx) (outputs := outputs) t0.1
  have t1 := t0.2
  have r1 := sequential_generatedRoot_189.facts_1 (ctx := ctx) (outputs := outputs) t1
  ⟨r0, r1⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) where
  inputs : ByteArray × mxxMat0 × ByteArray × (Fin 1 → Int) × (Fin 1 → Int) × (Fin 72000 → Int) × (Fin 72000 → Int) × (Fin 72000 → Int) × (Fin 72000 → Int) × (Fin 72000 → Int) × (Fin 72000 → Int) × (Fin 16 → Int) × (Fin 16 → Int) × (Fin 16 → Int) × (Fin 16 → Int) × (Fin 16 → Int) × (Fin 16 → Int) × (Fin 14400 → Int) × (Fin 14400 → Int) × (Fin 14400 → Int) × (Fin 14400 → Int) × (Fin 14400 → Int) × (Fin 14400 → Int) × Unit
  witness : generatedRoot.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {params : Params}
local notation "generatedRootCtx" => (generatedRoot.Ctx backend hashModel params)

def generatedRoot.w_0_0  (ctx : generatedRootCtx) : ByteArray :=
  ctx.inputs.1

def generatedRoot.w_1_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.inputs.2.1

def generatedRoot.w_17_0  (ctx : generatedRootCtx) : ByteArray :=
  ctx.inputs.2.2.1

def generatedRoot.w_34_0  (ctx : generatedRootCtx) : Fin 1 → Int :=
  ctx.inputs.2.2.2.1

def generatedRoot.w_35_0  (ctx : generatedRootCtx) : Fin 1 → Int :=
  ctx.inputs.2.2.2.2.1

def generatedRoot.w_71_0  (ctx : generatedRootCtx) : Fin 72000 → Int :=
  ctx.inputs.2.2.2.2.2.1

def generatedRoot.w_72_0  (ctx : generatedRootCtx) : Fin 72000 → Int :=
  ctx.inputs.2.2.2.2.2.2.1

def generatedRoot.w_77_0  (ctx : generatedRootCtx) : Fin 72000 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.1

def generatedRoot.w_78_0  (ctx : generatedRootCtx) : Fin 72000 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.1

def generatedRoot.w_81_0  (ctx : generatedRootCtx) : Fin 72000 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_82_0  (ctx : generatedRootCtx) : Fin 72000 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_108_0  (ctx : generatedRootCtx) : Fin 16 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_109_0  (ctx : generatedRootCtx) : Fin 16 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_114_0  (ctx : generatedRootCtx) : Fin 16 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_115_0  (ctx : generatedRootCtx) : Fin 16 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_118_0  (ctx : generatedRootCtx) : Fin 16 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_119_0  (ctx : generatedRootCtx) : Fin 16 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_153_0  (ctx : generatedRootCtx) : Fin 14400 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_154_0  (ctx : generatedRootCtx) : Fin 14400 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_159_0  (ctx : generatedRootCtx) : Fin 14400 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_160_0  (ctx : generatedRootCtx) : Fin 14400 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_163_0  (ctx : generatedRootCtx) : Fin 14400 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_164_0  (ctx : generatedRootCtx) : Fin 14400 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_2_0  (ctx : generatedRootCtx) : mxxMat0 :=
  (1 : mxxMat0)

def generatedRoot.w_3_0  (ctx : generatedRootCtx) : Fin 4 → mxxMat1 :=
  ctx.witness.w_3_0

def generatedRoot.w_3_1  (ctx : generatedRootCtx) : Fin 4 → mxxMat1 :=
  ctx.witness.w_3_1

def generatedRoot.w_3_2  (ctx : generatedRootCtx) : Fin 4 → mxxMat1 :=
  ctx.witness.w_3_2

def generatedRoot.w_3_3  (ctx : generatedRootCtx) : Fin 4 → mxxMat1 :=
  ctx.witness.w_3_3

def generatedRoot.w_4_0  (ctx : generatedRootCtx) : Fin 8 → mxxMat1 :=
  ctx.witness.w_4_0

def generatedRoot.w_4_1  (ctx : generatedRootCtx) : Fin 8 → mxxMat1 :=
  ctx.witness.w_4_1

def generatedRoot.w_4_2  (ctx : generatedRootCtx) : Fin 8 → mxxMat1 :=
  ctx.witness.w_4_2

def generatedRoot.w_4_3  (ctx : generatedRootCtx) : Fin 8 → mxxMat1 :=
  ctx.witness.w_4_3

def generatedRoot.w_5_0  (ctx : generatedRootCtx) : Fin 8 → mxxMat1 :=
  ctx.witness.w_5_0

def generatedRoot.w_5_1  (ctx : generatedRootCtx) : Fin 8 → mxxMat1 :=
  ctx.witness.w_5_1

def generatedRoot.w_5_2  (ctx : generatedRootCtx) : Fin 8 → mxxMat1 :=
  ctx.witness.w_5_2

def generatedRoot.w_5_3  (ctx : generatedRootCtx) : Fin 8 → mxxMat1 :=
  ctx.witness.w_5_3

def generatedRoot.w_6_0  (ctx : generatedRootCtx) : Fin 8 → mxxMat1 :=
  ctx.witness.w_6_0

def generatedRoot.w_6_1  (ctx : generatedRootCtx) : Fin 8 → mxxMat1 :=
  ctx.witness.w_6_1

def generatedRoot.w_6_2  (ctx : generatedRootCtx) : Fin 8 → mxxMat1 :=
  ctx.witness.w_6_2

def generatedRoot.w_6_3  (ctx : generatedRootCtx) : Fin 8 → mxxMat1 :=
  ctx.witness.w_6_3

def generatedRoot.w_7_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat1 :=
  ctx.witness.w_7_0

def generatedRoot.w_7_1  (ctx : generatedRootCtx) : Fin 1 → mxxMat1 :=
  ctx.witness.w_7_1

def generatedRoot.w_7_2  (ctx : generatedRootCtx) : Fin 1 → mxxMat1 :=
  ctx.witness.w_7_2

def generatedRoot.w_7_3  (ctx : generatedRootCtx) : Fin 1 → mxxMat1 :=
  ctx.witness.w_7_3

def generatedRoot.w_8_0  (ctx : generatedRootCtx) : Fin 8 → mxxMat1 :=
  ctx.witness.w_8_0

def generatedRoot.w_8_1  (ctx : generatedRootCtx) : Fin 8 → mxxMat1 :=
  ctx.witness.w_8_1

def generatedRoot.w_8_2  (ctx : generatedRootCtx) : Fin 8 → mxxMat1 :=
  ctx.witness.w_8_2

def generatedRoot.w_8_3  (ctx : generatedRootCtx) : Fin 8 → mxxMat1 :=
  ctx.witness.w_8_3

def generatedRoot.w_9_0  (ctx : generatedRootCtx) : Fin 8 → mxxMat1 :=
  ctx.witness.w_9_0

def generatedRoot.w_9_1  (ctx : generatedRootCtx) : Fin 8 → mxxMat1 :=
  ctx.witness.w_9_1

def generatedRoot.w_9_2  (ctx : generatedRootCtx) : Fin 8 → mxxMat1 :=
  ctx.witness.w_9_2

def generatedRoot.w_9_3  (ctx : generatedRootCtx) : Fin 8 → mxxMat1 :=
  ctx.witness.w_9_3

def generatedRoot.w_10_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat1 :=
  ctx.witness.w_10_0

def generatedRoot.w_10_1  (ctx : generatedRootCtx) : Fin 1 → mxxMat1 :=
  ctx.witness.w_10_1

def generatedRoot.w_10_2  (ctx : generatedRootCtx) : Fin 1 → mxxMat1 :=
  ctx.witness.w_10_2

def generatedRoot.w_10_3  (ctx : generatedRootCtx) : Fin 1 → mxxMat1 :=
  ctx.witness.w_10_3

def generatedRoot.w_11_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat1 :=
  ctx.witness.w_11_0

def generatedRoot.w_11_1  (ctx : generatedRootCtx) : Fin 1 → mxxMat1 :=
  ctx.witness.w_11_1

def generatedRoot.w_11_2  (ctx : generatedRootCtx) : Fin 1 → mxxMat1 :=
  ctx.witness.w_11_2

def generatedRoot.w_11_3  (ctx : generatedRootCtx) : Fin 1 → mxxMat1 :=
  ctx.witness.w_11_3

def generatedRoot.w_12_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat1 :=
  ctx.witness.w_12_0

def generatedRoot.w_12_1  (ctx : generatedRootCtx) : Fin 1 → mxxMat1 :=
  ctx.witness.w_12_1

def generatedRoot.w_12_2  (ctx : generatedRootCtx) : Fin 1 → mxxMat1 :=
  ctx.witness.w_12_2

def generatedRoot.w_12_3  (ctx : generatedRootCtx) : Fin 1 → mxxMat1 :=
  ctx.witness.w_12_3

def generatedRoot.w_13_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat1 :=
  ctx.witness.w_13_0

def generatedRoot.w_13_1  (ctx : generatedRootCtx) : Fin 1 → mxxMat1 :=
  ctx.witness.w_13_1

def generatedRoot.w_13_2  (ctx : generatedRootCtx) : Fin 1 → mxxMat1 :=
  ctx.witness.w_13_2

def generatedRoot.w_13_3  (ctx : generatedRootCtx) : Fin 1 → mxxMat1 :=
  ctx.witness.w_13_3

def generatedRoot.w_14_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat1 :=
  ctx.witness.w_14_0

def generatedRoot.w_14_1  (ctx : generatedRootCtx) : Fin 1 → mxxMat1 :=
  ctx.witness.w_14_1

def generatedRoot.w_14_2  (ctx : generatedRootCtx) : Fin 1 → mxxMat1 :=
  ctx.witness.w_14_2

def generatedRoot.w_14_3  (ctx : generatedRootCtx) : Fin 1 → mxxMat1 :=
  ctx.witness.w_14_3

def generatedRoot.w_15_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat1 :=
  ctx.witness.w_15_0

def generatedRoot.w_15_1  (ctx : generatedRootCtx) : Fin 1 → mxxMat1 :=
  ctx.witness.w_15_1

def generatedRoot.w_15_2  (ctx : generatedRootCtx) : Fin 1 → mxxMat1 :=
  ctx.witness.w_15_2

def generatedRoot.w_15_3  (ctx : generatedRootCtx) : Fin 1 → mxxMat1 :=
  ctx.witness.w_15_3

def generatedRoot.w_16_0  (ctx : generatedRootCtx) : mxxMat1 :=
  (0 : mxxMat1)

def generatedRoot.w_18_0  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.witness.w_18_0

def generatedRoot.w_19_0  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.witness.w_19_0

def generatedRoot.w_20_0  (ctx : generatedRootCtx) : mxxMat6 :=
  ctx.witness.w_20_0

def generatedRoot.w_21_0  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.witness.w_21_0

def generatedRoot.w_22_0  (ctx : generatedRootCtx) : mxxMat7 :=
  ctx.witness.w_22_0

def generatedRoot.w_23_0  (ctx : generatedRootCtx) : mxxMat6 :=
  MxxRuntime.matrixMul (generatedRoot.w_20_0 ctx) (generatedRoot.w_22_0 ctx)

def generatedRoot.w_24_0  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.witness.w_24_0

def generatedRoot.w_25_0  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.witness.w_25_0

def generatedRoot.w_26_0  (ctx : generatedRootCtx) : mxxMat6 :=
  ctx.witness.w_26_0

def generatedRoot.w_27_0  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.witness.w_27_0

def generatedRoot.w_28_0  (ctx : generatedRootCtx) : mxxMat7 :=
  ctx.witness.w_28_0

def generatedRoot.w_29_0  (ctx : generatedRootCtx) : mxxMat6 :=
  MxxRuntime.matrixMul (generatedRoot.w_26_0 ctx) (generatedRoot.w_28_0 ctx)

def generatedRoot.w_30_0  (ctx : generatedRootCtx) : mxxMat6 :=
  MxxRuntime.matrixAdd (generatedRoot.w_23_0 ctx) (generatedRoot.w_29_0 ctx)

def generatedRoot.w_31_0  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.witness.w_31_0

def generatedRoot.w_32_0  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.witness.w_32_0

def generatedRoot.w_33_0  (ctx : generatedRootCtx) : mxxMat6 :=
  ctx.witness.w_33_0

def generatedRoot.w_36_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat1 :=
  ctx.witness.w_36_0

def generatedRoot.w_36_1  (ctx : generatedRootCtx) : Fin 1 → mxxMat1 :=
  ctx.witness.w_36_1

def generatedRoot.w_36_2  (ctx : generatedRootCtx) : Fin 1 → mxxMat1 :=
  ctx.witness.w_36_2

def generatedRoot.w_36_3  (ctx : generatedRootCtx) : Fin 1 → mxxMat1 :=
  ctx.witness.w_36_3

def generatedRoot.w_37_0  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.witness.w_37_0

def generatedRoot.w_38_0  (ctx : generatedRootCtx) : mxxMat7 :=
  ctx.witness.w_38_0

def generatedRoot.w_39_0  (ctx : generatedRootCtx) : mxxMat7 :=
  ctx.witness.w_39_0

def generatedRoot.w_40_0  (ctx : generatedRootCtx) : mxxMat6 :=
  MxxRuntime.matrixMul (generatedRoot.w_33_0 ctx) (generatedRoot.w_39_0 ctx)

def generatedRoot.w_41_0  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.witness.w_41_0

def generatedRoot.w_42_0  (ctx : generatedRootCtx) : mxxMat7 :=
  ctx.witness.w_42_0

def generatedRoot.w_43_0  (ctx : generatedRootCtx) : mxxMat6 :=
  MxxRuntime.matrixMul (generatedRoot.w_20_0 ctx) (generatedRoot.w_42_0 ctx)

def generatedRoot.w_44_0  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.witness.w_44_0

def generatedRoot.w_45_0  (ctx : generatedRootCtx) : mxxMat7 :=
  ctx.witness.w_45_0

def generatedRoot.w_46_0  (ctx : generatedRootCtx) : mxxMat6 :=
  MxxRuntime.matrixMul (generatedRoot.w_26_0 ctx) (generatedRoot.w_45_0 ctx)

def generatedRoot.w_47_0  (ctx : generatedRootCtx) : mxxMat6 :=
  MxxRuntime.matrixAdd (generatedRoot.w_43_0 ctx) (generatedRoot.w_46_0 ctx)

def generatedRoot.w_48_0  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.witness.w_48_0

def generatedRoot.w_49_0  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.witness.w_49_0

def generatedRoot.w_50_0  (ctx : generatedRootCtx) : mxxMat6 :=
  ctx.witness.w_50_0

def generatedRoot.w_51_0  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.witness.w_51_0

def generatedRoot.w_52_0  (ctx : generatedRootCtx) : mxxMat7 :=
  ctx.witness.w_52_0

def generatedRoot.w_53_0  (ctx : generatedRootCtx) : mxxMat7 :=
  ctx.witness.w_53_0

def generatedRoot.w_54_0  (ctx : generatedRootCtx) : mxxMat6 :=
  MxxRuntime.matrixMul (generatedRoot.w_50_0 ctx) (generatedRoot.w_53_0 ctx)

def generatedRoot.w_55_0  (ctx : generatedRootCtx) : mxxMat6 :=
  MxxRuntime.matrixAdd (generatedRoot.w_40_0 ctx) (generatedRoot.w_54_0 ctx)

def generatedRoot.w_56_0  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.witness.w_56_0

def generatedRoot.w_57_0  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.witness.w_57_0

def generatedRoot.w_58_0  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.witness.w_58_0

def generatedRoot.w_59_0  (ctx : generatedRootCtx) : mxxMat7 :=
  ctx.witness.w_59_0

def generatedRoot.w_60_0  (ctx : generatedRootCtx) : mxxMat7 :=
  ctx.witness.w_60_0

def generatedRoot.w_61_0  (ctx : generatedRootCtx) : mxxMat6 :=
  MxxRuntime.matrixMul (generatedRoot.w_33_0 ctx) (generatedRoot.w_60_0 ctx)

def generatedRoot.w_62_0  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.witness.w_62_0

def generatedRoot.w_63_0  (ctx : generatedRootCtx) : mxxMat7 :=
  ctx.witness.w_63_0

def generatedRoot.w_64_0  (ctx : generatedRootCtx) : mxxMat7 :=
  ctx.witness.w_64_0

def generatedRoot.w_65_0  (ctx : generatedRootCtx) : mxxMat6 :=
  MxxRuntime.matrixMul (generatedRoot.w_50_0 ctx) (generatedRoot.w_64_0 ctx)

def generatedRoot.w_66_0  (ctx : generatedRootCtx) : mxxMat6 :=
  MxxRuntime.matrixAdd (generatedRoot.w_61_0 ctx) (generatedRoot.w_65_0 ctx)

def generatedRoot.w_67_0  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.witness.w_67_0

def generatedRoot.w_68_0  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.witness.w_68_0

def generatedRoot.w_69_0  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.witness.w_69_0

def generatedRoot.w_69_1  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.witness.w_69_1

def generatedRoot.w_69_2  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.witness.w_69_2

def generatedRoot.w_69_3  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.witness.w_69_3

def generatedRoot.w_70_0  (ctx : generatedRootCtx) : mxxMat3 :=
  (MxxRuntime.unitRow (0) : mxxMat3)

def generatedRoot.w_73_0  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.witness.w_73_0

def generatedRoot.w_74_0  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.witness.w_74_0

def generatedRoot.w_75_0  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.witness.w_75_0

def generatedRoot.w_76_0  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.witness.w_76_0

def generatedRoot.w_79_0  (ctx : generatedRootCtx) : Fin 72000 → mxxMat1 :=
  ctx.witness.w_79_0

def generatedRoot.w_79_1  (ctx : generatedRootCtx) : Fin 72000 → mxxMat1 :=
  ctx.witness.w_79_1

def generatedRoot.w_79_2  (ctx : generatedRootCtx) : Fin 72000 → mxxMat1 :=
  ctx.witness.w_79_2

def generatedRoot.w_79_3  (ctx : generatedRootCtx) : Fin 72000 → mxxMat1 :=
  ctx.witness.w_79_3

def generatedRoot.w_79_4  (ctx : generatedRootCtx) : Fin 72000 → mxxMat1 :=
  ctx.witness.w_79_4

def generatedRoot.w_79_5  (ctx : generatedRootCtx) : Fin 72000 → mxxMat1 :=
  ctx.witness.w_79_5

def generatedRoot.w_79_6  (ctx : generatedRootCtx) : Fin 72000 → mxxMat1 :=
  ctx.witness.w_79_6

def generatedRoot.w_79_7  (ctx : generatedRootCtx) : Fin 72000 → mxxMat1 :=
  ctx.witness.w_79_7

def generatedRoot.w_80_0  (ctx : generatedRootCtx) : Fin 36000 → mxxMat1 :=
  ctx.witness.w_80_0

def generatedRoot.w_80_1  (ctx : generatedRootCtx) : Fin 36000 → mxxMat1 :=
  ctx.witness.w_80_1

def generatedRoot.w_80_2  (ctx : generatedRootCtx) : Fin 36000 → mxxMat1 :=
  ctx.witness.w_80_2

def generatedRoot.w_80_3  (ctx : generatedRootCtx) : Fin 36000 → mxxMat1 :=
  ctx.witness.w_80_3

def generatedRoot.w_83_0  (ctx : generatedRootCtx) : Fin 72000 → mxxMat1 :=
  ctx.witness.w_83_0

def generatedRoot.w_83_1  (ctx : generatedRootCtx) : Fin 72000 → mxxMat1 :=
  ctx.witness.w_83_1

def generatedRoot.w_83_2  (ctx : generatedRootCtx) : Fin 72000 → mxxMat1 :=
  ctx.witness.w_83_2

def generatedRoot.w_83_3  (ctx : generatedRootCtx) : Fin 72000 → mxxMat1 :=
  ctx.witness.w_83_3

def generatedRoot.w_83_4  (ctx : generatedRootCtx) : Fin 72000 → mxxMat1 :=
  ctx.witness.w_83_4

def generatedRoot.w_83_5  (ctx : generatedRootCtx) : Fin 72000 → mxxMat1 :=
  ctx.witness.w_83_5

def generatedRoot.w_83_6  (ctx : generatedRootCtx) : Fin 72000 → mxxMat1 :=
  ctx.witness.w_83_6

def generatedRoot.w_83_7  (ctx : generatedRootCtx) : Fin 72000 → mxxMat1 :=
  ctx.witness.w_83_7

def generatedRoot.w_84_0  (ctx : generatedRootCtx) : Fin 36000 → mxxMat1 :=
  ctx.witness.w_84_0

def generatedRoot.w_84_1  (ctx : generatedRootCtx) : Fin 36000 → mxxMat1 :=
  ctx.witness.w_84_1

def generatedRoot.w_84_2  (ctx : generatedRootCtx) : Fin 36000 → mxxMat1 :=
  ctx.witness.w_84_2

def generatedRoot.w_84_3  (ctx : generatedRootCtx) : Fin 36000 → mxxMat1 :=
  ctx.witness.w_84_3

def generatedRoot.w_85_0  (ctx : generatedRootCtx) : Fin 36000 → mxxMat1 :=
  ctx.witness.w_85_0

def generatedRoot.w_85_1  (ctx : generatedRootCtx) : Fin 36000 → mxxMat1 :=
  ctx.witness.w_85_1

def generatedRoot.w_85_2  (ctx : generatedRootCtx) : Fin 36000 → mxxMat1 :=
  ctx.witness.w_85_2

def generatedRoot.w_85_3  (ctx : generatedRootCtx) : Fin 36000 → mxxMat1 :=
  ctx.witness.w_85_3

def generatedRoot.w_86_0  (ctx : generatedRootCtx) : Fin 36000 → mxxMat1 :=
  ctx.witness.w_86_0

def generatedRoot.w_86_1  (ctx : generatedRootCtx) : Fin 36000 → mxxMat1 :=
  ctx.witness.w_86_1

def generatedRoot.w_86_2  (ctx : generatedRootCtx) : Fin 36000 → mxxMat1 :=
  ctx.witness.w_86_2

def generatedRoot.w_86_3  (ctx : generatedRootCtx) : Fin 36000 → mxxMat1 :=
  ctx.witness.w_86_3

def generatedRoot.w_87_0  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.witness.w_87_0

def generatedRoot.w_88_0  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.witness.w_88_0

def generatedRoot.w_89_0  (ctx : generatedRootCtx) : mxxMat1 :=
  MxxRuntime.matrixAdd (generatedRoot.w_87_0 ctx) (generatedRoot.w_88_0 ctx)

def generatedRoot.w_90_0  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.witness.w_90_0

def generatedRoot.w_91_0  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.witness.w_91_0

def generatedRoot.w_92_0  (ctx : generatedRootCtx) : mxxMat1 :=
  MxxRuntime.matrixAdd (generatedRoot.w_90_0 ctx) (generatedRoot.w_91_0 ctx)

def generatedRoot.w_93_0  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.witness.w_93_0

def generatedRoot.w_94_0  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.witness.w_94_0

def generatedRoot.w_95_0  (ctx : generatedRootCtx) : mxxMat1 :=
  MxxRuntime.matrixAdd (generatedRoot.w_93_0 ctx) (generatedRoot.w_94_0 ctx)

def generatedRoot.w_96_0  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.witness.w_96_0

def generatedRoot.w_97_0  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.witness.w_97_0

def generatedRoot.w_98_0  (ctx : generatedRootCtx) : mxxMat1 :=
  MxxRuntime.matrixAdd (generatedRoot.w_96_0 ctx) (generatedRoot.w_97_0 ctx)

def generatedRoot.w_99_0  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.witness.w_99_0

def generatedRoot.w_99_1  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.witness.w_99_1

def generatedRoot.w_99_2  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.witness.w_99_2

def generatedRoot.w_99_3  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.witness.w_99_3

def generatedRoot.w_100_0  (ctx : generatedRootCtx) : mxxMat2 :=
  ctx.witness.w_100_0

def generatedRoot.w_101_0  (ctx : generatedRootCtx) : mxxMat3 :=
  (MxxRuntime.unitRow (0) : mxxMat3)

def generatedRoot.w_102_0  (ctx : generatedRootCtx) : mxxMat2 :=
  ctx.witness.w_102_0

def generatedRoot.w_103_0  (ctx : generatedRootCtx) : mxxMat3 :=
  (MxxRuntime.unitRow (0) : mxxMat3)

def generatedRoot.w_104_0  (ctx : generatedRootCtx) : mxxMat2 :=
  ctx.witness.w_104_0

def generatedRoot.w_105_0  (ctx : generatedRootCtx) : mxxMat3 :=
  (MxxRuntime.unitRow (0) : mxxMat3)

def generatedRoot.w_106_0  (ctx : generatedRootCtx) : mxxMat2 :=
  ctx.witness.w_106_0

def generatedRoot.w_107_0  (ctx : generatedRootCtx) : mxxMat2 :=
  ctx.witness.w_107_0

def generatedRoot.w_107_1  (ctx : generatedRootCtx) : mxxMat2 :=
  ctx.witness.w_107_1

def generatedRoot.w_107_2  (ctx : generatedRootCtx) : mxxMat2 :=
  ctx.witness.w_107_2

def generatedRoot.w_107_3  (ctx : generatedRootCtx) : mxxMat2 :=
  ctx.witness.w_107_3

def generatedRoot.w_110_0  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.witness.w_110_0

def generatedRoot.w_111_0  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.witness.w_111_0

def generatedRoot.w_112_0  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.witness.w_112_0

def generatedRoot.w_113_0  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.witness.w_113_0

def generatedRoot.w_116_0  (ctx : generatedRootCtx) : Fin 16 → mxxMat1 :=
  ctx.witness.w_116_0

def generatedRoot.w_116_1  (ctx : generatedRootCtx) : Fin 16 → mxxMat1 :=
  ctx.witness.w_116_1

def generatedRoot.w_116_2  (ctx : generatedRootCtx) : Fin 16 → mxxMat1 :=
  ctx.witness.w_116_2

def generatedRoot.w_116_3  (ctx : generatedRootCtx) : Fin 16 → mxxMat1 :=
  ctx.witness.w_116_3

def generatedRoot.w_116_4  (ctx : generatedRootCtx) : Fin 16 → mxxMat1 :=
  ctx.witness.w_116_4

def generatedRoot.w_116_5  (ctx : generatedRootCtx) : Fin 16 → mxxMat1 :=
  ctx.witness.w_116_5

def generatedRoot.w_116_6  (ctx : generatedRootCtx) : Fin 16 → mxxMat1 :=
  ctx.witness.w_116_6

def generatedRoot.w_116_7  (ctx : generatedRootCtx) : Fin 16 → mxxMat1 :=
  ctx.witness.w_116_7

def generatedRoot.w_117_0  (ctx : generatedRootCtx) : Fin 8 → mxxMat1 :=
  ctx.witness.w_117_0

def generatedRoot.w_117_1  (ctx : generatedRootCtx) : Fin 8 → mxxMat1 :=
  ctx.witness.w_117_1

def generatedRoot.w_117_2  (ctx : generatedRootCtx) : Fin 8 → mxxMat1 :=
  ctx.witness.w_117_2

def generatedRoot.w_117_3  (ctx : generatedRootCtx) : Fin 8 → mxxMat1 :=
  ctx.witness.w_117_3

def generatedRoot.w_120_0  (ctx : generatedRootCtx) : Fin 16 → mxxMat1 :=
  ctx.witness.w_120_0

def generatedRoot.w_120_1  (ctx : generatedRootCtx) : Fin 16 → mxxMat1 :=
  ctx.witness.w_120_1

def generatedRoot.w_120_2  (ctx : generatedRootCtx) : Fin 16 → mxxMat1 :=
  ctx.witness.w_120_2

def generatedRoot.w_120_3  (ctx : generatedRootCtx) : Fin 16 → mxxMat1 :=
  ctx.witness.w_120_3

def generatedRoot.w_120_4  (ctx : generatedRootCtx) : Fin 16 → mxxMat1 :=
  ctx.witness.w_120_4

def generatedRoot.w_120_5  (ctx : generatedRootCtx) : Fin 16 → mxxMat1 :=
  ctx.witness.w_120_5

def generatedRoot.w_120_6  (ctx : generatedRootCtx) : Fin 16 → mxxMat1 :=
  ctx.witness.w_120_6

def generatedRoot.w_120_7  (ctx : generatedRootCtx) : Fin 16 → mxxMat1 :=
  ctx.witness.w_120_7

def generatedRoot.w_121_0  (ctx : generatedRootCtx) : Fin 8 → mxxMat1 :=
  ctx.witness.w_121_0

def generatedRoot.w_121_1  (ctx : generatedRootCtx) : Fin 8 → mxxMat1 :=
  ctx.witness.w_121_1

def generatedRoot.w_121_2  (ctx : generatedRootCtx) : Fin 8 → mxxMat1 :=
  ctx.witness.w_121_2

def generatedRoot.w_121_3  (ctx : generatedRootCtx) : Fin 8 → mxxMat1 :=
  ctx.witness.w_121_3

def generatedRoot.w_122_0  (ctx : generatedRootCtx) : Fin 8 → mxxMat1 :=
  ctx.witness.w_122_0

def generatedRoot.w_122_1  (ctx : generatedRootCtx) : Fin 8 → mxxMat1 :=
  ctx.witness.w_122_1

def generatedRoot.w_122_2  (ctx : generatedRootCtx) : Fin 8 → mxxMat1 :=
  ctx.witness.w_122_2

def generatedRoot.w_122_3  (ctx : generatedRootCtx) : Fin 8 → mxxMat1 :=
  ctx.witness.w_122_3

def generatedRoot.w_123_0  (ctx : generatedRootCtx) : Fin 8 → mxxMat1 :=
  ctx.witness.w_123_0

def generatedRoot.w_123_1  (ctx : generatedRootCtx) : Fin 8 → mxxMat1 :=
  ctx.witness.w_123_1

def generatedRoot.w_123_2  (ctx : generatedRootCtx) : Fin 8 → mxxMat1 :=
  ctx.witness.w_123_2

def generatedRoot.w_123_3  (ctx : generatedRootCtx) : Fin 8 → mxxMat1 :=
  ctx.witness.w_123_3

def generatedRoot.w_124_0  (ctx : generatedRootCtx) : Fin 8 → mxxMat1 :=
  ctx.witness.w_124_0

def generatedRoot.w_124_1  (ctx : generatedRootCtx) : Fin 8 → mxxMat1 :=
  ctx.witness.w_124_1

def generatedRoot.w_124_2  (ctx : generatedRootCtx) : Fin 8 → mxxMat1 :=
  ctx.witness.w_124_2

def generatedRoot.w_124_3  (ctx : generatedRootCtx) : Fin 8 → mxxMat1 :=
  ctx.witness.w_124_3

def generatedRoot.w_125_0  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.witness.w_125_0

def generatedRoot.w_126_0  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.witness.w_126_0

def generatedRoot.w_127_0  (ctx : generatedRootCtx) : mxxMat1 :=
  MxxRuntime.matrixAdd (generatedRoot.w_125_0 ctx) (generatedRoot.w_126_0 ctx)

def generatedRoot.w_128_0  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.witness.w_128_0

def generatedRoot.w_129_0  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.witness.w_129_0

def generatedRoot.w_130_0  (ctx : generatedRootCtx) : mxxMat1 :=
  MxxRuntime.matrixAdd (generatedRoot.w_128_0 ctx) (generatedRoot.w_129_0 ctx)

def generatedRoot.w_131_0  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.witness.w_131_0

def generatedRoot.w_132_0  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.witness.w_132_0

def generatedRoot.w_133_0  (ctx : generatedRootCtx) : mxxMat1 :=
  MxxRuntime.matrixAdd (generatedRoot.w_131_0 ctx) (generatedRoot.w_132_0 ctx)

def generatedRoot.w_134_0  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.witness.w_134_0

def generatedRoot.w_135_0  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.witness.w_135_0

def generatedRoot.w_136_0  (ctx : generatedRootCtx) : mxxMat1 :=
  MxxRuntime.matrixAdd (generatedRoot.w_134_0 ctx) (generatedRoot.w_135_0 ctx)

def generatedRoot.w_137_0  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.witness.w_137_0

def generatedRoot.w_137_1  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.witness.w_137_1

def generatedRoot.w_137_2  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.witness.w_137_2

def generatedRoot.w_137_3  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.witness.w_137_3

def generatedRoot.w_138_0  (ctx : generatedRootCtx) : mxxMat1 :=
  (0 : mxxMat1)

def generatedRoot.w_139_0  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.witness.w_139_0

def generatedRoot.w_139_1  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.witness.w_139_1

def generatedRoot.w_139_2  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.witness.w_139_2

def generatedRoot.w_139_3  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.witness.w_139_3

def generatedRoot.w_140_0  (ctx : generatedRootCtx) : mxxMat1 :=
  (0 : mxxMat1)

def generatedRoot.w_141_0  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.witness.w_141_0

def generatedRoot.w_141_1  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.witness.w_141_1

def generatedRoot.w_141_2  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.witness.w_141_2

def generatedRoot.w_141_3  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.witness.w_141_3

def generatedRoot.w_142_0  (ctx : generatedRootCtx) : mxxMat1 :=
  (0 : mxxMat1)

def generatedRoot.w_143_0  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.witness.w_143_0

def generatedRoot.w_143_1  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.witness.w_143_1

def generatedRoot.w_143_2  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.witness.w_143_2

def generatedRoot.w_143_3  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.witness.w_143_3

def generatedRoot.w_144_0  (ctx : generatedRootCtx) : mxxMat1 :=
  (0 : mxxMat1)

def generatedRoot.w_145_0  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.witness.w_145_0

def generatedRoot.w_145_1  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.witness.w_145_1

def generatedRoot.w_145_2  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.witness.w_145_2

def generatedRoot.w_145_3  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.witness.w_145_3

def generatedRoot.w_146_0  (ctx : generatedRootCtx) : mxxMat1 :=
  (0 : mxxMat1)

def generatedRoot.w_147_0  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.witness.w_147_0

def generatedRoot.w_147_1  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.witness.w_147_1

def generatedRoot.w_147_2  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.witness.w_147_2

def generatedRoot.w_147_3  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.witness.w_147_3

def generatedRoot.w_148_0  (ctx : generatedRootCtx) : mxxMat1 :=
  (0 : mxxMat1)

def generatedRoot.w_149_0  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.witness.w_149_0

def generatedRoot.w_149_1  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.witness.w_149_1

def generatedRoot.w_149_2  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.witness.w_149_2

def generatedRoot.w_149_3  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.witness.w_149_3

def generatedRoot.w_150_0  (ctx : generatedRootCtx) : mxxMat1 :=
  (0 : mxxMat1)

def generatedRoot.w_151_0  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.witness.w_151_0

def generatedRoot.w_151_1  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.witness.w_151_1

def generatedRoot.w_151_2  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.witness.w_151_2

def generatedRoot.w_151_3  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.witness.w_151_3

def generatedRoot.w_152_0  (ctx : generatedRootCtx) : mxxMat5 :=
  (MxxRuntime.unitRow (0) : mxxMat5)

def generatedRoot.w_155_0  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.witness.w_155_0

def generatedRoot.w_156_0  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.witness.w_156_0

def generatedRoot.w_157_0  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.witness.w_157_0

def generatedRoot.w_158_0  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.witness.w_158_0

def generatedRoot.w_161_0  (ctx : generatedRootCtx) : Fin 14400 → mxxMat1 :=
  ctx.witness.w_161_0

def generatedRoot.w_161_1  (ctx : generatedRootCtx) : Fin 14400 → mxxMat1 :=
  ctx.witness.w_161_1

def generatedRoot.w_161_2  (ctx : generatedRootCtx) : Fin 14400 → mxxMat1 :=
  ctx.witness.w_161_2

def generatedRoot.w_161_3  (ctx : generatedRootCtx) : Fin 14400 → mxxMat1 :=
  ctx.witness.w_161_3

def generatedRoot.w_161_4  (ctx : generatedRootCtx) : Fin 14400 → mxxMat1 :=
  ctx.witness.w_161_4

def generatedRoot.w_161_5  (ctx : generatedRootCtx) : Fin 14400 → mxxMat1 :=
  ctx.witness.w_161_5

def generatedRoot.w_161_6  (ctx : generatedRootCtx) : Fin 14400 → mxxMat1 :=
  ctx.witness.w_161_6

def generatedRoot.w_161_7  (ctx : generatedRootCtx) : Fin 14400 → mxxMat1 :=
  ctx.witness.w_161_7

def generatedRoot.w_162_0  (ctx : generatedRootCtx) : Fin 7200 → mxxMat1 :=
  ctx.witness.w_162_0

def generatedRoot.w_162_1  (ctx : generatedRootCtx) : Fin 7200 → mxxMat1 :=
  ctx.witness.w_162_1

def generatedRoot.w_162_2  (ctx : generatedRootCtx) : Fin 7200 → mxxMat1 :=
  ctx.witness.w_162_2

def generatedRoot.w_162_3  (ctx : generatedRootCtx) : Fin 7200 → mxxMat1 :=
  ctx.witness.w_162_3

def generatedRoot.w_165_0  (ctx : generatedRootCtx) : Fin 14400 → mxxMat1 :=
  ctx.witness.w_165_0

def generatedRoot.w_165_1  (ctx : generatedRootCtx) : Fin 14400 → mxxMat1 :=
  ctx.witness.w_165_1

def generatedRoot.w_165_2  (ctx : generatedRootCtx) : Fin 14400 → mxxMat1 :=
  ctx.witness.w_165_2

def generatedRoot.w_165_3  (ctx : generatedRootCtx) : Fin 14400 → mxxMat1 :=
  ctx.witness.w_165_3

def generatedRoot.w_165_4  (ctx : generatedRootCtx) : Fin 14400 → mxxMat1 :=
  ctx.witness.w_165_4

def generatedRoot.w_165_5  (ctx : generatedRootCtx) : Fin 14400 → mxxMat1 :=
  ctx.witness.w_165_5

def generatedRoot.w_165_6  (ctx : generatedRootCtx) : Fin 14400 → mxxMat1 :=
  ctx.witness.w_165_6

def generatedRoot.w_165_7  (ctx : generatedRootCtx) : Fin 14400 → mxxMat1 :=
  ctx.witness.w_165_7

def generatedRoot.w_166_0  (ctx : generatedRootCtx) : Fin 7200 → mxxMat1 :=
  ctx.witness.w_166_0

def generatedRoot.w_166_1  (ctx : generatedRootCtx) : Fin 7200 → mxxMat1 :=
  ctx.witness.w_166_1

def generatedRoot.w_166_2  (ctx : generatedRootCtx) : Fin 7200 → mxxMat1 :=
  ctx.witness.w_166_2

def generatedRoot.w_166_3  (ctx : generatedRootCtx) : Fin 7200 → mxxMat1 :=
  ctx.witness.w_166_3

def generatedRoot.w_167_0  (ctx : generatedRootCtx) : Fin 7200 → mxxMat1 :=
  ctx.witness.w_167_0

def generatedRoot.w_167_1  (ctx : generatedRootCtx) : Fin 7200 → mxxMat1 :=
  ctx.witness.w_167_1

def generatedRoot.w_167_2  (ctx : generatedRootCtx) : Fin 7200 → mxxMat1 :=
  ctx.witness.w_167_2

def generatedRoot.w_167_3  (ctx : generatedRootCtx) : Fin 7200 → mxxMat1 :=
  ctx.witness.w_167_3

def generatedRoot.w_168_0  (ctx : generatedRootCtx) : Fin 7200 → mxxMat1 :=
  ctx.witness.w_168_0

def generatedRoot.w_168_1  (ctx : generatedRootCtx) : Fin 7200 → mxxMat1 :=
  ctx.witness.w_168_1

def generatedRoot.w_168_2  (ctx : generatedRootCtx) : Fin 7200 → mxxMat1 :=
  ctx.witness.w_168_2

def generatedRoot.w_168_3  (ctx : generatedRootCtx) : Fin 7200 → mxxMat1 :=
  ctx.witness.w_168_3

def generatedRoot.w_169_0  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.witness.w_169_0

def generatedRoot.w_170_0  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.witness.w_170_0

def generatedRoot.w_171_0  (ctx : generatedRootCtx) : mxxMat1 :=
  MxxRuntime.matrixAdd (generatedRoot.w_169_0 ctx) (generatedRoot.w_170_0 ctx)

def generatedRoot.w_172_0  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.witness.w_172_0

def generatedRoot.w_173_0  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.witness.w_173_0

def generatedRoot.w_174_0  (ctx : generatedRootCtx) : mxxMat1 :=
  MxxRuntime.matrixAdd (generatedRoot.w_172_0 ctx) (generatedRoot.w_173_0 ctx)

def generatedRoot.w_175_0  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.witness.w_175_0

def generatedRoot.w_176_0  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.witness.w_176_0

def generatedRoot.w_177_0  (ctx : generatedRootCtx) : mxxMat1 :=
  MxxRuntime.matrixAdd (generatedRoot.w_175_0 ctx) (generatedRoot.w_176_0 ctx)

def generatedRoot.w_178_0  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.witness.w_178_0

def generatedRoot.w_179_0  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.witness.w_179_0

def generatedRoot.w_180_0  (ctx : generatedRootCtx) : mxxMat1 :=
  MxxRuntime.matrixAdd (generatedRoot.w_178_0 ctx) (generatedRoot.w_179_0 ctx)

def generatedRoot.w_181_0  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.witness.w_181_0

def generatedRoot.w_181_1  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.witness.w_181_1

def generatedRoot.w_181_2  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.witness.w_181_2

def generatedRoot.w_181_3  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.witness.w_181_3

def generatedRoot.w_182_0  (ctx : generatedRootCtx) : mxxMat4 :=
  ctx.witness.w_182_0

def generatedRoot.w_183_0  (ctx : generatedRootCtx) : mxxMat5 :=
  (MxxRuntime.unitRow (0) : mxxMat5)

def generatedRoot.w_184_0  (ctx : generatedRootCtx) : mxxMat4 :=
  ctx.witness.w_184_0

def generatedRoot.w_185_0  (ctx : generatedRootCtx) : mxxMat5 :=
  (MxxRuntime.unitRow (0) : mxxMat5)

def generatedRoot.w_186_0  (ctx : generatedRootCtx) : mxxMat4 :=
  ctx.witness.w_186_0

def generatedRoot.w_187_0  (ctx : generatedRootCtx) : mxxMat5 :=
  (MxxRuntime.unitRow (0) : mxxMat5)

def generatedRoot.w_188_0  (ctx : generatedRootCtx) : mxxMat4 :=
  ctx.witness.w_188_0

def generatedRoot.w_189_0  (ctx : generatedRootCtx) : mxxMat4 :=
  ctx.witness.w_189_0

def generatedRoot.w_189_1  (ctx : generatedRootCtx) : mxxMat4 :=
  ctx.witness.w_189_1

def generatedRoot.w_189_2  (ctx : generatedRootCtx) : mxxMat4 :=
  ctx.witness.w_189_2

def generatedRoot.w_189_3  (ctx : generatedRootCtx) : mxxMat4 :=
  ctx.witness.w_189_3

def generatedRoot.outs  (ctx : generatedRootCtx) : (generatedRoot.OutputType) :=
  ((generatedRoot.w_3_0 ctx), ((generatedRoot.w_3_1 ctx), ((generatedRoot.w_3_2 ctx), ((generatedRoot.w_3_3 ctx), ((generatedRoot.w_4_0 ctx), ((generatedRoot.w_4_1 ctx), ((generatedRoot.w_4_2 ctx), ((generatedRoot.w_4_3 ctx), ((generatedRoot.w_5_0 ctx), ((generatedRoot.w_5_1 ctx), ((generatedRoot.w_5_2 ctx), ((generatedRoot.w_5_3 ctx), ((generatedRoot.w_6_0 ctx), ((generatedRoot.w_6_1 ctx), ((generatedRoot.w_6_2 ctx), ((generatedRoot.w_6_3 ctx), ((generatedRoot.w_7_0 ctx), ((generatedRoot.w_7_1 ctx), ((generatedRoot.w_7_2 ctx), ((generatedRoot.w_7_3 ctx), ((generatedRoot.w_8_0 ctx), ((generatedRoot.w_8_1 ctx), ((generatedRoot.w_8_2 ctx), ((generatedRoot.w_8_3 ctx), ((generatedRoot.w_9_0 ctx), ((generatedRoot.w_9_1 ctx), ((generatedRoot.w_9_2 ctx), ((generatedRoot.w_9_3 ctx), ((generatedRoot.w_10_0 ctx), ((generatedRoot.w_10_1 ctx), ((generatedRoot.w_10_2 ctx), ((generatedRoot.w_10_3 ctx), ((generatedRoot.w_11_0 ctx), ((generatedRoot.w_11_1 ctx), ((generatedRoot.w_11_2 ctx), ((generatedRoot.w_11_3 ctx), ((generatedRoot.w_12_0 ctx), ((generatedRoot.w_12_1 ctx), ((generatedRoot.w_12_2 ctx), ((generatedRoot.w_12_3 ctx), ((generatedRoot.w_13_0 ctx), ((generatedRoot.w_13_1 ctx), ((generatedRoot.w_13_2 ctx), ((generatedRoot.w_13_3 ctx), ((generatedRoot.w_14_0 ctx), ((generatedRoot.w_14_1 ctx), ((generatedRoot.w_14_2 ctx), ((generatedRoot.w_14_3 ctx), ((generatedRoot.w_15_0 ctx), ((generatedRoot.w_15_1 ctx), ((generatedRoot.w_15_2 ctx), ((generatedRoot.w_15_3 ctx), ((generatedRoot.w_69_0 ctx), ((generatedRoot.w_69_1 ctx), ((generatedRoot.w_69_2 ctx), ((generatedRoot.w_69_3 ctx), ((generatedRoot.w_107_0 ctx), ((generatedRoot.w_107_1 ctx), ((generatedRoot.w_107_2 ctx), ((generatedRoot.w_107_3 ctx), ((generatedRoot.w_137_0 ctx), ((generatedRoot.w_137_1 ctx), ((generatedRoot.w_137_2 ctx), ((generatedRoot.w_137_3 ctx), ((generatedRoot.w_139_0 ctx), ((generatedRoot.w_139_1 ctx), ((generatedRoot.w_139_2 ctx), ((generatedRoot.w_139_3 ctx), ((generatedRoot.w_141_0 ctx), ((generatedRoot.w_141_1 ctx), ((generatedRoot.w_141_2 ctx), ((generatedRoot.w_141_3 ctx), ((generatedRoot.w_143_0 ctx), ((generatedRoot.w_143_1 ctx), ((generatedRoot.w_143_2 ctx), ((generatedRoot.w_143_3 ctx), ((generatedRoot.w_145_0 ctx), ((generatedRoot.w_145_1 ctx), ((generatedRoot.w_145_2 ctx), ((generatedRoot.w_145_3 ctx), ((generatedRoot.w_147_0 ctx), ((generatedRoot.w_147_1 ctx), ((generatedRoot.w_147_2 ctx), ((generatedRoot.w_147_3 ctx), ((generatedRoot.w_149_0 ctx), ((generatedRoot.w_149_1 ctx), ((generatedRoot.w_149_2 ctx), ((generatedRoot.w_149_3 ctx), ((generatedRoot.w_151_0 ctx), ((generatedRoot.w_151_1 ctx), ((generatedRoot.w_151_2 ctx), ((generatedRoot.w_151_3 ctx), ((generatedRoot.w_189_0 ctx), ((generatedRoot.w_189_1 ctx), ((generatedRoot.w_189_2 ctx), ((generatedRoot.w_189_3 ctx), ((generatedRoot.w_79_6 ctx), ((generatedRoot.w_79_7 ctx), ((generatedRoot.w_83_6 ctx), ((generatedRoot.w_83_7 ctx), ((generatedRoot.w_36_2 ctx), ((generatedRoot.w_36_3 ctx), ((generatedRoot.w_116_6 ctx), ((generatedRoot.w_116_7 ctx), ((generatedRoot.w_120_6 ctx), ((generatedRoot.w_120_7 ctx), ((generatedRoot.w_161_6 ctx), ((generatedRoot.w_161_7 ctx), ((generatedRoot.w_165_6 ctx), ((generatedRoot.w_165_7 ctx), ()))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))

structure generatedRoot.Facts_0  (ctx : generatedRootCtx) : Prop where
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

theorem generatedRoot.facts_0  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_0 backend hashModel params (generatedRoot.w_0_0 ctx) (generatedRoot.w_1_0 ctx) (generatedRoot.w_17_0 ctx) (generatedRoot.w_2_0 ctx) (generatedRoot.w_3_0 ctx) (generatedRoot.w_3_1 ctx) (generatedRoot.w_3_2 ctx) (generatedRoot.w_3_3 ctx) (generatedRoot.w_4_0 ctx) (generatedRoot.w_4_1 ctx) (generatedRoot.w_4_2 ctx) (generatedRoot.w_4_3 ctx) (generatedRoot.w_5_0 ctx) (generatedRoot.w_5_1 ctx) (generatedRoot.w_5_2 ctx) (generatedRoot.w_5_3 ctx) (generatedRoot.w_6_0 ctx) (generatedRoot.w_6_1 ctx) (generatedRoot.w_6_2 ctx) (generatedRoot.w_6_3 ctx) (generatedRoot.w_7_0 ctx) (generatedRoot.w_7_1 ctx) (generatedRoot.w_7_2 ctx) (generatedRoot.w_7_3 ctx) (generatedRoot.w_8_0 ctx) (generatedRoot.w_8_1 ctx) (generatedRoot.w_8_2 ctx) (generatedRoot.w_8_3 ctx) (generatedRoot.w_9_0 ctx) (generatedRoot.w_9_1 ctx) (generatedRoot.w_9_2 ctx) (generatedRoot.w_9_3 ctx) (generatedRoot.w_10_0 ctx) (generatedRoot.w_10_1 ctx) (generatedRoot.w_10_2 ctx) (generatedRoot.w_10_3 ctx) (generatedRoot.w_11_0 ctx) (generatedRoot.w_11_1 ctx) (generatedRoot.w_11_2 ctx) (generatedRoot.w_11_3 ctx) (generatedRoot.w_12_0 ctx) (generatedRoot.w_12_1 ctx) (generatedRoot.w_12_2 ctx) (generatedRoot.w_12_3 ctx) (generatedRoot.w_13_0 ctx) (generatedRoot.w_13_1 ctx) (generatedRoot.w_13_2 ctx) (generatedRoot.w_13_3 ctx) (generatedRoot.w_14_0 ctx) (generatedRoot.w_14_1 ctx) (generatedRoot.w_14_2 ctx) (generatedRoot.w_14_3 ctx) (generatedRoot.w_15_0 ctx) (generatedRoot.w_15_1 ctx) (generatedRoot.w_15_2 ctx) (generatedRoot.w_15_3 ctx) (generatedRoot.w_18_0 ctx) (generatedRoot.w_19_0 ctx) (generatedRoot.w_20_0 ctx) (generatedRoot.w_21_0 ctx) (generatedRoot.w_22_0 ctx) (generatedRoot.w_24_0 ctx)) :
    generatedRoot.Facts_0 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_1  (ctx : generatedRootCtx) : Prop where
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

theorem generatedRoot.facts_1  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_1 backend hashModel params (generatedRoot.w_0_0 ctx) (generatedRoot.w_1_0 ctx) (generatedRoot.w_17_0 ctx) (generatedRoot.w_34_0 ctx) (generatedRoot.w_35_0 ctx) (generatedRoot.w_2_0 ctx) (generatedRoot.w_24_0 ctx) (generatedRoot.w_25_0 ctx) (generatedRoot.w_26_0 ctx) (generatedRoot.w_27_0 ctx) (generatedRoot.w_28_0 ctx) (generatedRoot.w_30_0 ctx) (generatedRoot.w_31_0 ctx) (generatedRoot.w_32_0 ctx) (generatedRoot.w_33_0 ctx) (generatedRoot.w_36_0 ctx) (generatedRoot.w_36_1 ctx) (generatedRoot.w_36_2 ctx) (generatedRoot.w_36_3 ctx) (generatedRoot.w_37_0 ctx) (generatedRoot.w_38_0 ctx) (generatedRoot.w_39_0 ctx) (generatedRoot.w_41_0 ctx) (generatedRoot.w_42_0 ctx) (generatedRoot.w_44_0 ctx) (generatedRoot.w_45_0 ctx)) :
    generatedRoot.Facts_1 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_2  (ctx : generatedRootCtx) : Prop where
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

theorem generatedRoot.facts_2  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_2 backend (generatedRoot.w_36_1 ctx) (generatedRoot.w_36_2 ctx) (generatedRoot.w_36_3 ctx) (generatedRoot.w_47_0 ctx) (generatedRoot.w_48_0 ctx) (generatedRoot.w_49_0 ctx) (generatedRoot.w_50_0 ctx) (generatedRoot.w_51_0 ctx) (generatedRoot.w_52_0 ctx) (generatedRoot.w_53_0 ctx) (generatedRoot.w_55_0 ctx) (generatedRoot.w_56_0 ctx) (generatedRoot.w_57_0 ctx) (generatedRoot.w_58_0 ctx) (generatedRoot.w_59_0 ctx) (generatedRoot.w_60_0 ctx) (generatedRoot.w_62_0 ctx)) :
    generatedRoot.Facts_2 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_3  (ctx : generatedRootCtx) : Prop where
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

theorem generatedRoot.facts_3  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_3 backend hashModel params (generatedRoot.w_0_0 ctx) (generatedRoot.w_1_0 ctx) (generatedRoot.w_17_0 ctx) (generatedRoot.w_71_0 ctx) (generatedRoot.w_72_0 ctx) (generatedRoot.w_77_0 ctx) (generatedRoot.w_78_0 ctx) (generatedRoot.w_81_0 ctx) (generatedRoot.w_82_0 ctx) (generatedRoot.w_2_0 ctx) (generatedRoot.w_3_0 ctx) (generatedRoot.w_3_1 ctx) (generatedRoot.w_3_2 ctx) (generatedRoot.w_3_3 ctx) (generatedRoot.w_16_0 ctx) (generatedRoot.w_56_0 ctx) (generatedRoot.w_57_0 ctx) (generatedRoot.w_62_0 ctx) (generatedRoot.w_63_0 ctx) (generatedRoot.w_64_0 ctx) (generatedRoot.w_66_0 ctx) (generatedRoot.w_67_0 ctx) (generatedRoot.w_68_0 ctx) (generatedRoot.w_69_0 ctx) (generatedRoot.w_69_1 ctx) (generatedRoot.w_69_2 ctx) (generatedRoot.w_69_3 ctx) (generatedRoot.w_73_0 ctx) (generatedRoot.w_74_0 ctx) (generatedRoot.w_75_0 ctx) (generatedRoot.w_76_0 ctx) (generatedRoot.w_79_0 ctx) (generatedRoot.w_79_1 ctx) (generatedRoot.w_79_2 ctx) (generatedRoot.w_79_3 ctx) (generatedRoot.w_79_4 ctx) (generatedRoot.w_79_5 ctx) (generatedRoot.w_79_6 ctx) (generatedRoot.w_79_7 ctx) (generatedRoot.w_80_0 ctx) (generatedRoot.w_80_1 ctx) (generatedRoot.w_80_2 ctx) (generatedRoot.w_80_3 ctx) (generatedRoot.w_83_0 ctx) (generatedRoot.w_83_1 ctx) (generatedRoot.w_83_2 ctx) (generatedRoot.w_83_3 ctx) (generatedRoot.w_83_4 ctx) (generatedRoot.w_83_5 ctx) (generatedRoot.w_83_6 ctx) (generatedRoot.w_83_7 ctx) (generatedRoot.w_84_0 ctx) (generatedRoot.w_84_1 ctx) (generatedRoot.w_84_2 ctx) (generatedRoot.w_84_3 ctx)) :
    generatedRoot.Facts_3 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_4  (ctx : generatedRootCtx) : Prop where
  t128 : (36000) = 36000
  h85_3 : (∀ i : Fin 36000, parallel_generatedRoot_85 backend hashModel params i ((generatedRoot.w_17_0 ctx), (((generatedRoot.w_84_0 ctx) i), (((generatedRoot.w_84_2 ctx) i), ((generatedRoot.w_4_0 ctx), (((generatedRoot.w_84_1 ctx) i), (((generatedRoot.w_84_3 ctx) i), ((generatedRoot.w_4_2 ctx), ((generatedRoot.w_4_1 ctx), ((generatedRoot.w_4_3 ctx), ()))))))))) (((generatedRoot.w_85_0 ctx) i), (((generatedRoot.w_85_1 ctx) i), (((generatedRoot.w_85_2 ctx) i), (((generatedRoot.w_85_3 ctx) i), ())))))
  t130 : (36000) = 36000
  h86_3 : (∀ i : Fin 36000, parallel_generatedRoot_86 backend hashModel params i ((generatedRoot.w_85_0 ctx), ((generatedRoot.w_85_1 ctx), ((generatedRoot.w_85_2 ctx), ((generatedRoot.w_85_3 ctx), ())))) (((generatedRoot.w_86_0 ctx) i), (((generatedRoot.w_86_1 ctx) i), (((generatedRoot.w_86_2 ctx) i), (((generatedRoot.w_86_3 ctx) i), ())))))
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

theorem generatedRoot.facts_4  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_4 backend hashModel params (generatedRoot.w_0_0 ctx) (generatedRoot.w_1_0 ctx) (generatedRoot.w_17_0 ctx) (generatedRoot.w_109_0 ctx) (generatedRoot.w_114_0 ctx) (generatedRoot.w_115_0 ctx) (generatedRoot.w_2_0 ctx) (generatedRoot.w_4_0 ctx) (generatedRoot.w_4_1 ctx) (generatedRoot.w_4_2 ctx) (generatedRoot.w_4_3 ctx) (generatedRoot.w_70_0 ctx) (generatedRoot.w_84_0 ctx) (generatedRoot.w_84_1 ctx) (generatedRoot.w_84_2 ctx) (generatedRoot.w_84_3 ctx) (generatedRoot.w_85_0 ctx) (generatedRoot.w_85_1 ctx) (generatedRoot.w_85_2 ctx) (generatedRoot.w_85_3 ctx) (generatedRoot.w_86_0 ctx) (generatedRoot.w_86_1 ctx) (generatedRoot.w_86_2 ctx) (generatedRoot.w_86_3 ctx) (generatedRoot.w_87_0 ctx) (generatedRoot.w_88_0 ctx) (generatedRoot.w_89_0 ctx) (generatedRoot.w_90_0 ctx) (generatedRoot.w_91_0 ctx) (generatedRoot.w_92_0 ctx) (generatedRoot.w_93_0 ctx) (generatedRoot.w_94_0 ctx) (generatedRoot.w_95_0 ctx) (generatedRoot.w_96_0 ctx) (generatedRoot.w_97_0 ctx) (generatedRoot.w_98_0 ctx) (generatedRoot.w_99_0 ctx) (generatedRoot.w_99_1 ctx) (generatedRoot.w_99_2 ctx) (generatedRoot.w_99_3 ctx) (generatedRoot.w_100_0 ctx) (generatedRoot.w_101_0 ctx) (generatedRoot.w_102_0 ctx) (generatedRoot.w_103_0 ctx) (generatedRoot.w_104_0 ctx) (generatedRoot.w_105_0 ctx) (generatedRoot.w_106_0 ctx) (generatedRoot.w_107_0 ctx) (generatedRoot.w_107_1 ctx) (generatedRoot.w_107_2 ctx) (generatedRoot.w_107_3 ctx) (generatedRoot.w_110_0 ctx) (generatedRoot.w_111_0 ctx) (generatedRoot.w_112_0 ctx) (generatedRoot.w_113_0 ctx) (generatedRoot.w_116_0 ctx) (generatedRoot.w_116_1 ctx) (generatedRoot.w_116_2 ctx) (generatedRoot.w_116_3 ctx) (generatedRoot.w_116_4 ctx) (generatedRoot.w_116_5 ctx) (generatedRoot.w_116_6 ctx) (generatedRoot.w_116_7 ctx)) :
    generatedRoot.Facts_4 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_5  (ctx : generatedRootCtx) : Prop where
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

theorem generatedRoot.facts_5  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_5 backend hashModel params (generatedRoot.w_0_0 ctx) (generatedRoot.w_1_0 ctx) (generatedRoot.w_17_0 ctx) (generatedRoot.w_108_0 ctx) (generatedRoot.w_118_0 ctx) (generatedRoot.w_119_0 ctx) (generatedRoot.w_2_0 ctx) (generatedRoot.w_6_0 ctx) (generatedRoot.w_6_1 ctx) (generatedRoot.w_6_2 ctx) (generatedRoot.w_6_3 ctx) (generatedRoot.w_9_0 ctx) (generatedRoot.w_9_1 ctx) (generatedRoot.w_9_2 ctx) (generatedRoot.w_9_3 ctx) (generatedRoot.w_10_0 ctx) (generatedRoot.w_10_1 ctx) (generatedRoot.w_10_2 ctx) (generatedRoot.w_10_3 ctx) (generatedRoot.w_11_0 ctx) (generatedRoot.w_11_1 ctx) (generatedRoot.w_11_2 ctx) (generatedRoot.w_11_3 ctx) (generatedRoot.w_12_0 ctx) (generatedRoot.w_12_1 ctx) (generatedRoot.w_12_2 ctx) (generatedRoot.w_12_3 ctx) (generatedRoot.w_13_0 ctx) (generatedRoot.w_13_1 ctx) (generatedRoot.w_13_2 ctx) (generatedRoot.w_13_3 ctx) (generatedRoot.w_14_0 ctx) (generatedRoot.w_14_1 ctx) (generatedRoot.w_14_2 ctx) (generatedRoot.w_14_3 ctx) (generatedRoot.w_116_0 ctx) (generatedRoot.w_116_1 ctx) (generatedRoot.w_116_2 ctx) (generatedRoot.w_116_3 ctx) (generatedRoot.w_117_0 ctx) (generatedRoot.w_117_1 ctx) (generatedRoot.w_117_2 ctx) (generatedRoot.w_117_3 ctx) (generatedRoot.w_120_0 ctx) (generatedRoot.w_120_1 ctx) (generatedRoot.w_120_2 ctx) (generatedRoot.w_120_3 ctx) (generatedRoot.w_120_4 ctx) (generatedRoot.w_120_5 ctx) (generatedRoot.w_120_6 ctx) (generatedRoot.w_120_7 ctx) (generatedRoot.w_121_0 ctx) (generatedRoot.w_121_1 ctx) (generatedRoot.w_121_2 ctx) (generatedRoot.w_121_3 ctx) (generatedRoot.w_122_0 ctx) (generatedRoot.w_122_1 ctx) (generatedRoot.w_122_2 ctx) (generatedRoot.w_122_3 ctx) (generatedRoot.w_123_0 ctx) (generatedRoot.w_123_1 ctx) (generatedRoot.w_123_2 ctx) (generatedRoot.w_123_3 ctx) (generatedRoot.w_124_0 ctx) (generatedRoot.w_124_1 ctx) (generatedRoot.w_124_2 ctx) (generatedRoot.w_124_3 ctx) (generatedRoot.w_125_0 ctx) (generatedRoot.w_126_0 ctx) (generatedRoot.w_127_0 ctx) (generatedRoot.w_128_0 ctx) (generatedRoot.w_129_0 ctx) (generatedRoot.w_130_0 ctx) (generatedRoot.w_131_0 ctx) (generatedRoot.w_132_0 ctx) (generatedRoot.w_133_0 ctx) (generatedRoot.w_134_0 ctx) (generatedRoot.w_135_0 ctx) (generatedRoot.w_136_0 ctx) (generatedRoot.w_137_0 ctx) (generatedRoot.w_137_1 ctx) (generatedRoot.w_137_2 ctx) (generatedRoot.w_137_3 ctx) (generatedRoot.w_138_0 ctx) (generatedRoot.w_139_0 ctx) (generatedRoot.w_139_1 ctx) (generatedRoot.w_139_2 ctx) (generatedRoot.w_139_3 ctx) (generatedRoot.w_140_0 ctx) (generatedRoot.w_141_0 ctx) (generatedRoot.w_141_1 ctx) (generatedRoot.w_141_2 ctx) (generatedRoot.w_141_3 ctx) (generatedRoot.w_142_0 ctx) (generatedRoot.w_143_0 ctx) (generatedRoot.w_143_1 ctx) (generatedRoot.w_143_2 ctx) (generatedRoot.w_143_3 ctx) (generatedRoot.w_144_0 ctx) (generatedRoot.w_145_0 ctx) (generatedRoot.w_145_1 ctx) (generatedRoot.w_145_2 ctx) (generatedRoot.w_145_3 ctx) (generatedRoot.w_146_0 ctx) (generatedRoot.w_147_0 ctx) (generatedRoot.w_147_1 ctx) (generatedRoot.w_147_2 ctx) (generatedRoot.w_147_3 ctx)) :
    generatedRoot.Facts_5 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_6  (ctx : generatedRootCtx) : Prop where
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
  h168_3 : (∀ i : Fin 7200, parallel_generatedRoot_168 backend hashModel params i ((generatedRoot.w_167_0 ctx), ((generatedRoot.w_167_1 ctx), ((generatedRoot.w_167_2 ctx), ((generatedRoot.w_167_3 ctx), ())))) (((generatedRoot.w_168_0 ctx) i), (((generatedRoot.w_168_1 ctx) i), (((generatedRoot.w_168_2 ctx) i), (((generatedRoot.w_168_3 ctx) i), ())))))
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

theorem generatedRoot.facts_6  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_6 backend hashModel params (generatedRoot.w_0_0 ctx) (generatedRoot.w_1_0 ctx) (generatedRoot.w_17_0 ctx) (generatedRoot.w_153_0 ctx) (generatedRoot.w_154_0 ctx) (generatedRoot.w_159_0 ctx) (generatedRoot.w_160_0 ctx) (generatedRoot.w_163_0 ctx) (generatedRoot.w_164_0 ctx) (generatedRoot.w_2_0 ctx) (generatedRoot.w_5_0 ctx) (generatedRoot.w_5_1 ctx) (generatedRoot.w_5_2 ctx) (generatedRoot.w_5_3 ctx) (generatedRoot.w_7_0 ctx) (generatedRoot.w_7_1 ctx) (generatedRoot.w_7_2 ctx) (generatedRoot.w_7_3 ctx) (generatedRoot.w_15_0 ctx) (generatedRoot.w_15_1 ctx) (generatedRoot.w_15_2 ctx) (generatedRoot.w_15_3 ctx) (generatedRoot.w_137_0 ctx) (generatedRoot.w_137_1 ctx) (generatedRoot.w_137_2 ctx) (generatedRoot.w_137_3 ctx) (generatedRoot.w_148_0 ctx) (generatedRoot.w_149_0 ctx) (generatedRoot.w_149_1 ctx) (generatedRoot.w_149_2 ctx) (generatedRoot.w_149_3 ctx) (generatedRoot.w_150_0 ctx) (generatedRoot.w_151_0 ctx) (generatedRoot.w_151_1 ctx) (generatedRoot.w_151_2 ctx) (generatedRoot.w_151_3 ctx) (generatedRoot.w_155_0 ctx) (generatedRoot.w_156_0 ctx) (generatedRoot.w_157_0 ctx) (generatedRoot.w_158_0 ctx) (generatedRoot.w_161_0 ctx) (generatedRoot.w_161_1 ctx) (generatedRoot.w_161_2 ctx) (generatedRoot.w_161_3 ctx) (generatedRoot.w_161_4 ctx) (generatedRoot.w_161_5 ctx) (generatedRoot.w_161_6 ctx) (generatedRoot.w_161_7 ctx) (generatedRoot.w_162_0 ctx) (generatedRoot.w_162_1 ctx) (generatedRoot.w_162_2 ctx) (generatedRoot.w_162_3 ctx) (generatedRoot.w_165_0 ctx) (generatedRoot.w_165_1 ctx) (generatedRoot.w_165_2 ctx) (generatedRoot.w_165_3 ctx) (generatedRoot.w_165_4 ctx) (generatedRoot.w_165_5 ctx) (generatedRoot.w_165_6 ctx) (generatedRoot.w_165_7 ctx) (generatedRoot.w_166_0 ctx) (generatedRoot.w_166_1 ctx) (generatedRoot.w_166_2 ctx) (generatedRoot.w_166_3 ctx) (generatedRoot.w_167_0 ctx) (generatedRoot.w_167_1 ctx) (generatedRoot.w_167_2 ctx) (generatedRoot.w_167_3 ctx) (generatedRoot.w_168_0 ctx) (generatedRoot.w_168_1 ctx) (generatedRoot.w_168_2 ctx) (generatedRoot.w_168_3 ctx) (generatedRoot.w_169_0 ctx) (generatedRoot.w_170_0 ctx) (generatedRoot.w_171_0 ctx) (generatedRoot.w_172_0 ctx) (generatedRoot.w_173_0 ctx) (generatedRoot.w_174_0 ctx) (generatedRoot.w_175_0 ctx) (generatedRoot.w_176_0 ctx) (generatedRoot.w_177_0 ctx) (generatedRoot.w_178_0 ctx) (generatedRoot.w_179_0 ctx) (generatedRoot.w_180_0 ctx) (generatedRoot.w_181_0 ctx) (generatedRoot.w_181_1 ctx) (generatedRoot.w_181_2 ctx) (generatedRoot.w_181_3 ctx)) :
    generatedRoot.Facts_6 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_7  (ctx : generatedRootCtx) (outputs : (generatedRoot.OutputType)) : Prop where
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
  hout : outputs = generatedRoot.outs ctx

theorem generatedRoot.facts_7  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_7 backend hashModel params (generatedRoot.w_3_0 ctx) (generatedRoot.w_3_1 ctx) (generatedRoot.w_3_2 ctx) (generatedRoot.w_3_3 ctx) (generatedRoot.w_4_0 ctx) (generatedRoot.w_4_1 ctx) (generatedRoot.w_4_2 ctx) (generatedRoot.w_4_3 ctx) (generatedRoot.w_5_0 ctx) (generatedRoot.w_5_1 ctx) (generatedRoot.w_5_2 ctx) (generatedRoot.w_5_3 ctx) (generatedRoot.w_6_0 ctx) (generatedRoot.w_6_1 ctx) (generatedRoot.w_6_2 ctx) (generatedRoot.w_6_3 ctx) (generatedRoot.w_7_0 ctx) (generatedRoot.w_7_1 ctx) (generatedRoot.w_7_2 ctx) (generatedRoot.w_7_3 ctx) (generatedRoot.w_8_0 ctx) (generatedRoot.w_8_1 ctx) (generatedRoot.w_8_2 ctx) (generatedRoot.w_8_3 ctx) (generatedRoot.w_9_0 ctx) (generatedRoot.w_9_1 ctx) (generatedRoot.w_9_2 ctx) (generatedRoot.w_9_3 ctx) (generatedRoot.w_10_0 ctx) (generatedRoot.w_10_1 ctx) (generatedRoot.w_10_2 ctx) (generatedRoot.w_10_3 ctx) (generatedRoot.w_11_0 ctx) (generatedRoot.w_11_1 ctx) (generatedRoot.w_11_2 ctx) (generatedRoot.w_11_3 ctx) (generatedRoot.w_12_0 ctx) (generatedRoot.w_12_1 ctx) (generatedRoot.w_12_2 ctx) (generatedRoot.w_12_3 ctx) (generatedRoot.w_13_0 ctx) (generatedRoot.w_13_1 ctx) (generatedRoot.w_13_2 ctx) (generatedRoot.w_13_3 ctx) (generatedRoot.w_14_0 ctx) (generatedRoot.w_14_1 ctx) (generatedRoot.w_14_2 ctx) (generatedRoot.w_14_3 ctx) (generatedRoot.w_15_0 ctx) (generatedRoot.w_15_1 ctx) (generatedRoot.w_15_2 ctx) (generatedRoot.w_15_3 ctx) (generatedRoot.w_36_2 ctx) (generatedRoot.w_36_3 ctx) (generatedRoot.w_69_0 ctx) (generatedRoot.w_69_1 ctx) (generatedRoot.w_69_2 ctx) (generatedRoot.w_69_3 ctx) (generatedRoot.w_79_6 ctx) (generatedRoot.w_79_7 ctx) (generatedRoot.w_83_6 ctx) (generatedRoot.w_83_7 ctx) (generatedRoot.w_107_0 ctx) (generatedRoot.w_107_1 ctx) (generatedRoot.w_107_2 ctx) (generatedRoot.w_107_3 ctx) (generatedRoot.w_116_6 ctx) (generatedRoot.w_116_7 ctx) (generatedRoot.w_120_6 ctx) (generatedRoot.w_120_7 ctx) (generatedRoot.w_137_0 ctx) (generatedRoot.w_137_1 ctx) (generatedRoot.w_137_2 ctx) (generatedRoot.w_137_3 ctx) (generatedRoot.w_139_0 ctx) (generatedRoot.w_139_1 ctx) (generatedRoot.w_139_2 ctx) (generatedRoot.w_139_3 ctx) (generatedRoot.w_141_0 ctx) (generatedRoot.w_141_1 ctx) (generatedRoot.w_141_2 ctx) (generatedRoot.w_141_3 ctx) (generatedRoot.w_143_0 ctx) (generatedRoot.w_143_1 ctx) (generatedRoot.w_143_2 ctx) (generatedRoot.w_143_3 ctx) (generatedRoot.w_145_0 ctx) (generatedRoot.w_145_1 ctx) (generatedRoot.w_145_2 ctx) (generatedRoot.w_145_3 ctx) (generatedRoot.w_147_0 ctx) (generatedRoot.w_147_1 ctx) (generatedRoot.w_147_2 ctx) (generatedRoot.w_147_3 ctx) (generatedRoot.w_149_0 ctx) (generatedRoot.w_149_1 ctx) (generatedRoot.w_149_2 ctx) (generatedRoot.w_149_3 ctx) (generatedRoot.w_151_0 ctx) (generatedRoot.w_151_1 ctx) (generatedRoot.w_151_2 ctx) (generatedRoot.w_151_3 ctx) (generatedRoot.w_152_0 ctx) (generatedRoot.w_161_6 ctx) (generatedRoot.w_161_7 ctx) (generatedRoot.w_165_6 ctx) (generatedRoot.w_165_7 ctx) (generatedRoot.w_168_0 ctx) (generatedRoot.w_168_1 ctx) (generatedRoot.w_168_2 ctx) (generatedRoot.w_168_3 ctx) (generatedRoot.w_181_0 ctx) (generatedRoot.w_181_1 ctx) (generatedRoot.w_181_2 ctx) (generatedRoot.w_181_3 ctx) (generatedRoot.w_182_0 ctx) (generatedRoot.w_183_0 ctx) (generatedRoot.w_184_0 ctx) (generatedRoot.w_185_0 ctx) (generatedRoot.w_186_0 ctx) (generatedRoot.w_187_0 ctx) (generatedRoot.w_188_0 ctx) (generatedRoot.w_189_0 ctx) (generatedRoot.w_189_1 ctx) (generatedRoot.w_189_2 ctx) (generatedRoot.w_189_3 ctx) outputs) :
    generatedRoot.Facts_7 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts  (ctx : generatedRootCtx) (outputs : (generatedRoot.OutputType)) : Prop where
  c0 : generatedRoot.Facts_0 ctx
  c1 : generatedRoot.Facts_1 ctx
  c2 : generatedRoot.Facts_2 ctx
  c3 : generatedRoot.Facts_3 ctx
  c4 : generatedRoot.Facts_4 ctx
  c5 : generatedRoot.Facts_5 ctx
  c6 : generatedRoot.Facts_6 ctx
  c7 : generatedRoot.Facts_7 ctx outputs

theorem generatedRoot.facts  (ctx : generatedRootCtx) {outputs : (generatedRoot.OutputType)}
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
  have r7 := generatedRoot.facts_7 (ctx := ctx) (outputs := outputs) t7
  ⟨r0, r1, r2, r3, r4, r5, r6, r7⟩

theorem generatedRoot.h3_3_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 4) :
    ∃ w : parallel_generatedRoot_3.Witness, parallel_generatedRoot_3.Facts (⟨((generatedRoot.w_0_0 ctx), ((generatedRoot.w_1_0 ctx), ((generatedRoot.w_2_0 ctx), ()))), w⟩ : parallel_generatedRoot_3.Ctx backend hashModel params i) (((generatedRoot.w_3_0 ctx) i), (((generatedRoot.w_3_1 ctx) i), (((generatedRoot.w_3_2 ctx) i), (((generatedRoot.w_3_3 ctx) i), ())))) :=
  let ⟨w, h⟩ := F.c0.h3_3 i
  ⟨w, parallel_generatedRoot_3.facts _ h⟩

theorem generatedRoot.h4_3_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 8) :
    ∃ w : parallel_generatedRoot_4.Witness, parallel_generatedRoot_4.Facts (⟨((generatedRoot.w_0_0 ctx), ((generatedRoot.w_1_0 ctx), ((generatedRoot.w_2_0 ctx), ()))), w⟩ : parallel_generatedRoot_4.Ctx backend hashModel params i) (((generatedRoot.w_4_0 ctx) i), (((generatedRoot.w_4_1 ctx) i), (((generatedRoot.w_4_2 ctx) i), (((generatedRoot.w_4_3 ctx) i), ())))) :=
  let ⟨w, h⟩ := F.c0.h4_3 i
  ⟨w, parallel_generatedRoot_4.facts _ h⟩

theorem generatedRoot.h5_3_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 8) :
    ∃ w : parallel_generatedRoot_5.Witness, parallel_generatedRoot_5.Facts (⟨((generatedRoot.w_0_0 ctx), ((generatedRoot.w_1_0 ctx), ((generatedRoot.w_2_0 ctx), ()))), w⟩ : parallel_generatedRoot_5.Ctx backend hashModel params i) (((generatedRoot.w_5_0 ctx) i), (((generatedRoot.w_5_1 ctx) i), (((generatedRoot.w_5_2 ctx) i), (((generatedRoot.w_5_3 ctx) i), ())))) :=
  let ⟨w, h⟩ := F.c0.h5_3 i
  ⟨w, parallel_generatedRoot_5.facts _ h⟩

theorem generatedRoot.h6_3_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 8) :
    ∃ w : parallel_generatedRoot_6.Witness, parallel_generatedRoot_6.Facts (⟨((generatedRoot.w_0_0 ctx), ((generatedRoot.w_1_0 ctx), ((generatedRoot.w_2_0 ctx), ()))), w⟩ : parallel_generatedRoot_6.Ctx backend hashModel params i) (((generatedRoot.w_6_0 ctx) i), (((generatedRoot.w_6_1 ctx) i), (((generatedRoot.w_6_2 ctx) i), (((generatedRoot.w_6_3 ctx) i), ())))) :=
  let ⟨w, h⟩ := F.c0.h6_3 i
  ⟨w, parallel_generatedRoot_6.facts _ h⟩

theorem generatedRoot.h7_3_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 1) :
    ∃ w : parallel_generatedRoot_7.Witness, parallel_generatedRoot_7.Facts (⟨((generatedRoot.w_0_0 ctx), ((generatedRoot.w_1_0 ctx), ((generatedRoot.w_2_0 ctx), ()))), w⟩ : parallel_generatedRoot_7.Ctx backend hashModel params i) (((generatedRoot.w_7_0 ctx) i), (((generatedRoot.w_7_1 ctx) i), (((generatedRoot.w_7_2 ctx) i), (((generatedRoot.w_7_3 ctx) i), ())))) :=
  let ⟨w, h⟩ := F.c0.h7_3 i
  ⟨w, parallel_generatedRoot_7.facts _ h⟩

theorem generatedRoot.h8_3_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 8) :
    ∃ w : parallel_generatedRoot_8.Witness, parallel_generatedRoot_8.Facts (⟨((generatedRoot.w_0_0 ctx), ((generatedRoot.w_1_0 ctx), ((generatedRoot.w_2_0 ctx), ()))), w⟩ : parallel_generatedRoot_8.Ctx backend hashModel params i) (((generatedRoot.w_8_0 ctx) i), (((generatedRoot.w_8_1 ctx) i), (((generatedRoot.w_8_2 ctx) i), (((generatedRoot.w_8_3 ctx) i), ())))) :=
  let ⟨w, h⟩ := F.c0.h8_3 i
  ⟨w, parallel_generatedRoot_8.facts _ h⟩

theorem generatedRoot.h9_3_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 8) :
    ∃ w : parallel_generatedRoot_9.Witness, parallel_generatedRoot_9.Facts (⟨((generatedRoot.w_0_0 ctx), ((generatedRoot.w_1_0 ctx), ((generatedRoot.w_2_0 ctx), ()))), w⟩ : parallel_generatedRoot_9.Ctx backend hashModel params i) (((generatedRoot.w_9_0 ctx) i), (((generatedRoot.w_9_1 ctx) i), (((generatedRoot.w_9_2 ctx) i), (((generatedRoot.w_9_3 ctx) i), ())))) :=
  let ⟨w, h⟩ := F.c0.h9_3 i
  ⟨w, parallel_generatedRoot_9.facts _ h⟩

theorem generatedRoot.h10_3_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 1) :
    ∃ w : parallel_generatedRoot_10.Witness, parallel_generatedRoot_10.Facts (⟨((generatedRoot.w_0_0 ctx), ((generatedRoot.w_1_0 ctx), ((generatedRoot.w_2_0 ctx), ()))), w⟩ : parallel_generatedRoot_10.Ctx backend hashModel params i) (((generatedRoot.w_10_0 ctx) i), (((generatedRoot.w_10_1 ctx) i), (((generatedRoot.w_10_2 ctx) i), (((generatedRoot.w_10_3 ctx) i), ())))) :=
  let ⟨w, h⟩ := F.c0.h10_3 i
  ⟨w, parallel_generatedRoot_10.facts _ h⟩

theorem generatedRoot.h11_3_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 1) :
    ∃ w : parallel_generatedRoot_11.Witness, parallel_generatedRoot_11.Facts (⟨((generatedRoot.w_0_0 ctx), ((generatedRoot.w_1_0 ctx), ((generatedRoot.w_2_0 ctx), ()))), w⟩ : parallel_generatedRoot_11.Ctx backend hashModel params i) (((generatedRoot.w_11_0 ctx) i), (((generatedRoot.w_11_1 ctx) i), (((generatedRoot.w_11_2 ctx) i), (((generatedRoot.w_11_3 ctx) i), ())))) :=
  let ⟨w, h⟩ := F.c0.h11_3 i
  ⟨w, parallel_generatedRoot_11.facts _ h⟩

theorem generatedRoot.h12_3_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 1) :
    ∃ w : parallel_generatedRoot_12.Witness, parallel_generatedRoot_12.Facts (⟨((generatedRoot.w_0_0 ctx), ((generatedRoot.w_1_0 ctx), ((generatedRoot.w_2_0 ctx), ()))), w⟩ : parallel_generatedRoot_12.Ctx backend hashModel params i) (((generatedRoot.w_12_0 ctx) i), (((generatedRoot.w_12_1 ctx) i), (((generatedRoot.w_12_2 ctx) i), (((generatedRoot.w_12_3 ctx) i), ())))) :=
  let ⟨w, h⟩ := F.c0.h12_3 i
  ⟨w, parallel_generatedRoot_12.facts _ h⟩

theorem generatedRoot.h13_3_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 1) :
    ∃ w : parallel_generatedRoot_13.Witness, parallel_generatedRoot_13.Facts (⟨((generatedRoot.w_0_0 ctx), ((generatedRoot.w_1_0 ctx), ((generatedRoot.w_2_0 ctx), ()))), w⟩ : parallel_generatedRoot_13.Ctx backend hashModel params i) (((generatedRoot.w_13_0 ctx) i), (((generatedRoot.w_13_1 ctx) i), (((generatedRoot.w_13_2 ctx) i), (((generatedRoot.w_13_3 ctx) i), ())))) :=
  let ⟨w, h⟩ := F.c0.h13_3 i
  ⟨w, parallel_generatedRoot_13.facts _ h⟩

theorem generatedRoot.h14_3_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 1) :
    ∃ w : parallel_generatedRoot_14.Witness, parallel_generatedRoot_14.Facts (⟨((generatedRoot.w_0_0 ctx), ((generatedRoot.w_1_0 ctx), ((generatedRoot.w_2_0 ctx), ()))), w⟩ : parallel_generatedRoot_14.Ctx backend hashModel params i) (((generatedRoot.w_14_0 ctx) i), (((generatedRoot.w_14_1 ctx) i), (((generatedRoot.w_14_2 ctx) i), (((generatedRoot.w_14_3 ctx) i), ())))) :=
  let ⟨w, h⟩ := F.c0.h14_3 i
  ⟨w, parallel_generatedRoot_14.facts _ h⟩

theorem generatedRoot.h15_3_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 1) :
    ∃ w : parallel_generatedRoot_15.Witness, parallel_generatedRoot_15.Facts (⟨((generatedRoot.w_0_0 ctx), ((generatedRoot.w_1_0 ctx), ((generatedRoot.w_2_0 ctx), ()))), w⟩ : parallel_generatedRoot_15.Ctx backend hashModel params i) (((generatedRoot.w_15_0 ctx) i), (((generatedRoot.w_15_1 ctx) i), (((generatedRoot.w_15_2 ctx) i), (((generatedRoot.w_15_3 ctx) i), ())))) :=
  let ⟨w, h⟩ := F.c0.h15_3 i
  ⟨w, parallel_generatedRoot_15.facts _ h⟩

theorem generatedRoot.h80_3_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 36000) :
    ∃ w : parallel_generatedRoot_80.Witness, parallel_generatedRoot_80.Facts (⟨((generatedRoot.w_79_0 ctx), ((generatedRoot.w_79_1 ctx), ((generatedRoot.w_79_2 ctx), ((generatedRoot.w_79_3 ctx), ())))), w⟩ : parallel_generatedRoot_80.Ctx backend hashModel params i) (((generatedRoot.w_80_0 ctx) i), (((generatedRoot.w_80_1 ctx) i), (((generatedRoot.w_80_2 ctx) i), (((generatedRoot.w_80_3 ctx) i), ())))) :=
  let ⟨w, h⟩ := F.c3.h80_3 i
  ⟨w, parallel_generatedRoot_80.facts _ h⟩

theorem generatedRoot.h84_3_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 36000) :
    ∃ w : parallel_generatedRoot_84.Witness, parallel_generatedRoot_84.Facts (⟨((generatedRoot.w_83_0 ctx), ((generatedRoot.w_83_1 ctx), ((generatedRoot.w_83_2 ctx), ((generatedRoot.w_83_3 ctx), ())))), w⟩ : parallel_generatedRoot_84.Ctx backend hashModel params i) (((generatedRoot.w_84_0 ctx) i), (((generatedRoot.w_84_1 ctx) i), (((generatedRoot.w_84_2 ctx) i), (((generatedRoot.w_84_3 ctx) i), ())))) :=
  let ⟨w, h⟩ := F.c3.h84_3 i
  ⟨w, parallel_generatedRoot_84.facts _ h⟩

theorem generatedRoot.h85_3_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 36000) :
    ∃ w : parallel_generatedRoot_85.Witness, parallel_generatedRoot_85.Facts (⟨((generatedRoot.w_17_0 ctx), (((generatedRoot.w_84_0 ctx) i), (((generatedRoot.w_84_2 ctx) i), ((generatedRoot.w_4_0 ctx), (((generatedRoot.w_84_1 ctx) i), (((generatedRoot.w_84_3 ctx) i), ((generatedRoot.w_4_2 ctx), ((generatedRoot.w_4_1 ctx), ((generatedRoot.w_4_3 ctx), ()))))))))), w⟩ : parallel_generatedRoot_85.Ctx backend hashModel params i) (((generatedRoot.w_85_0 ctx) i), (((generatedRoot.w_85_1 ctx) i), (((generatedRoot.w_85_2 ctx) i), (((generatedRoot.w_85_3 ctx) i), ())))) :=
  let ⟨w, h⟩ := F.c4.h85_3 i
  ⟨w, parallel_generatedRoot_85.facts _ h⟩

theorem generatedRoot.h117_3_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 8) :
    ∃ w : parallel_generatedRoot_117.Witness, parallel_generatedRoot_117.Facts (⟨((generatedRoot.w_116_0 ctx), ((generatedRoot.w_116_1 ctx), ((generatedRoot.w_116_2 ctx), ((generatedRoot.w_116_3 ctx), ())))), w⟩ : parallel_generatedRoot_117.Ctx backend hashModel params i) (((generatedRoot.w_117_0 ctx) i), (((generatedRoot.w_117_1 ctx) i), (((generatedRoot.w_117_2 ctx) i), (((generatedRoot.w_117_3 ctx) i), ())))) :=
  let ⟨w, h⟩ := F.c5.h117_3 i
  ⟨w, parallel_generatedRoot_117.facts _ h⟩

theorem generatedRoot.h121_3_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 8) :
    ∃ w : parallel_generatedRoot_121.Witness, parallel_generatedRoot_121.Facts (⟨((generatedRoot.w_120_0 ctx), ((generatedRoot.w_120_1 ctx), ((generatedRoot.w_120_2 ctx), ((generatedRoot.w_120_3 ctx), ())))), w⟩ : parallel_generatedRoot_121.Ctx backend hashModel params i) (((generatedRoot.w_121_0 ctx) i), (((generatedRoot.w_121_1 ctx) i), (((generatedRoot.w_121_2 ctx) i), (((generatedRoot.w_121_3 ctx) i), ())))) :=
  let ⟨w, h⟩ := F.c5.h121_3 i
  ⟨w, parallel_generatedRoot_121.facts _ h⟩

theorem generatedRoot.h122_3_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 8) :
    ∃ w : parallel_generatedRoot_122.Witness, parallel_generatedRoot_122.Facts (⟨((generatedRoot.w_17_0 ctx), (((generatedRoot.w_121_0 ctx) i), (((generatedRoot.w_121_2 ctx) i), ((generatedRoot.w_6_0 ctx), (((generatedRoot.w_121_1 ctx) i), (((generatedRoot.w_121_3 ctx) i), ((generatedRoot.w_6_2 ctx), ((generatedRoot.w_6_1 ctx), ((generatedRoot.w_6_3 ctx), ()))))))))), w⟩ : parallel_generatedRoot_122.Ctx backend hashModel params i) (((generatedRoot.w_122_0 ctx) i), (((generatedRoot.w_122_1 ctx) i), (((generatedRoot.w_122_2 ctx) i), (((generatedRoot.w_122_3 ctx) i), ())))) :=
  let ⟨w, h⟩ := F.c5.h122_3 i
  ⟨w, parallel_generatedRoot_122.facts _ h⟩

theorem generatedRoot.h123_3_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 8) :
    ∃ w : parallel_generatedRoot_123.Witness, parallel_generatedRoot_123.Facts (⟨((generatedRoot.w_17_0 ctx), (((generatedRoot.w_122_0 ctx) i), (((generatedRoot.w_122_2 ctx) i), ((generatedRoot.w_9_0 ctx), (((generatedRoot.w_122_1 ctx) i), (((generatedRoot.w_122_3 ctx) i), ((generatedRoot.w_9_2 ctx), ((generatedRoot.w_9_1 ctx), ((generatedRoot.w_9_3 ctx), ()))))))))), w⟩ : parallel_generatedRoot_123.Ctx backend hashModel params i) (((generatedRoot.w_123_0 ctx) i), (((generatedRoot.w_123_1 ctx) i), (((generatedRoot.w_123_2 ctx) i), (((generatedRoot.w_123_3 ctx) i), ())))) :=
  let ⟨w, h⟩ := F.c5.h123_3 i
  ⟨w, parallel_generatedRoot_123.facts _ h⟩

theorem generatedRoot.h162_3_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 7200) :
    ∃ w : parallel_generatedRoot_162.Witness, parallel_generatedRoot_162.Facts (⟨((generatedRoot.w_161_0 ctx), ((generatedRoot.w_161_1 ctx), ((generatedRoot.w_161_2 ctx), ((generatedRoot.w_161_3 ctx), ())))), w⟩ : parallel_generatedRoot_162.Ctx backend hashModel params i) (((generatedRoot.w_162_0 ctx) i), (((generatedRoot.w_162_1 ctx) i), (((generatedRoot.w_162_2 ctx) i), (((generatedRoot.w_162_3 ctx) i), ())))) :=
  let ⟨w, h⟩ := F.c6.h162_3 i
  ⟨w, parallel_generatedRoot_162.facts _ h⟩

theorem generatedRoot.h166_3_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 7200) :
    ∃ w : parallel_generatedRoot_166.Witness, parallel_generatedRoot_166.Facts (⟨((generatedRoot.w_165_0 ctx), ((generatedRoot.w_165_1 ctx), ((generatedRoot.w_165_2 ctx), ((generatedRoot.w_165_3 ctx), ())))), w⟩ : parallel_generatedRoot_166.Ctx backend hashModel params i) (((generatedRoot.w_166_0 ctx) i), (((generatedRoot.w_166_1 ctx) i), (((generatedRoot.w_166_2 ctx) i), (((generatedRoot.w_166_3 ctx) i), ())))) :=
  let ⟨w, h⟩ := F.c6.h166_3 i
  ⟨w, parallel_generatedRoot_166.facts _ h⟩

theorem generatedRoot.h167_3_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 7200) :
    ∃ w : parallel_generatedRoot_167.Witness, parallel_generatedRoot_167.Facts (⟨((generatedRoot.w_17_0 ctx), (((generatedRoot.w_166_0 ctx) i), (((generatedRoot.w_166_2 ctx) i), ((generatedRoot.w_5_0 ctx), (((generatedRoot.w_166_1 ctx) i), (((generatedRoot.w_166_3 ctx) i), ((generatedRoot.w_5_2 ctx), ((generatedRoot.w_5_1 ctx), ((generatedRoot.w_5_3 ctx), ()))))))))), w⟩ : parallel_generatedRoot_167.Ctx backend hashModel params i) (((generatedRoot.w_167_0 ctx) i), (((generatedRoot.w_167_1 ctx) i), (((generatedRoot.w_167_2 ctx) i), (((generatedRoot.w_167_3 ctx) i), ())))) :=
  let ⟨w, h⟩ := F.c6.h167_3 i
  ⟨w, parallel_generatedRoot_167.facts _ h⟩

end

end W_Stage_preprocess_0_0

end
