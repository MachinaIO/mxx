import Stage_keygen

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000
set_option linter.unusedVariables false

-- The wire types repeat the ring modulus `X ^ n + 1`; unfolding it while checking a
-- projection against its field type expands it at the ring dimension.
attribute [local irreducible] Mxx.Primitives.negacyclicModulus

noncomputable section

namespace Stage_keygen

local notation "mxxNat0" => 2007780228562889017327695763426836351403150354206288662473911661538034308791619296312261592598031117
local notation "mxxNat1" => 5000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
local notation "mxxMat0" => Mxx.Primitives.ExactMatrix modulus_0 32768 60 45
local notation "mxxMat1" => Mxx.Primitives.ExactMatrix modulus_0 32768 1 60
local notation "mxxMat2" => Mxx.Primitives.ExactMatrix modulus_0 32768 1 256
local notation "mxxMat3" => Mxx.Primitives.ExactMatrix modulus_0 32768 1 45
local notation "mxxMat4" => Mxx.Primitives.ExactMatrix modulus_0 32768 1 295
local notation "mxxMat5" => Mxx.Primitives.ExactMatrix modulus_0 32768 1 30
local notation "mxxMat6" => Mxx.Primitives.ExactMatrix modulus_1 32768 1 295
local notation "mxxMat7" => Mxx.Primitives.ExactMatrix modulus_0 32768 1 1
local notation "mxxMat8" => Mxx.Primitives.ExactMatrix modulus_0 32768 60 1
local notation "mxxMat9" => Mxx.Primitives.ExactMatrix modulus_1 32768 1 1
local notation "mxxMat10" => Mxx.Primitives.ExactMatrix modulus_1 32768 1 30
local notation "mxxMat11" => Mxx.Primitives.ExactMatrix modulus_0 32768 30 1
local notation "mxxMat12" => Mxx.Primitives.ExactMatrix modulus_0 32768 295 1

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_14.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) where
  inputs : ByteArray × mxxMat0 × mxxMat1 × Unit
  witness : parallel_generatedRoot_14.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_14Ctx" => (parallel_generatedRoot_14.Ctx backend hashModel tape path params i_0)

def parallel_generatedRoot_14.w_1_0  (ctx : parallel_generatedRoot_14Ctx) : ByteArray :=
  ctx.inputs.1

def parallel_generatedRoot_14.w_2_0  (ctx : parallel_generatedRoot_14Ctx) : mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_14.w_3_0  (ctx : parallel_generatedRoot_14Ctx) : mxxMat1 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_14.w_0_0  (ctx : parallel_generatedRoot_14Ctx) : mxxMat2 :=
  (0 : mxxMat2)

def parallel_generatedRoot_14.w_4_0  (ctx : parallel_generatedRoot_14Ctx) : mxxMat3 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_14.w_3_0 ctx) (parallel_generatedRoot_14.w_2_0 ctx)

def parallel_generatedRoot_14.w_5_0  (ctx : parallel_generatedRoot_14Ctx) : mxxMat2 :=
  ctx.witness.w_5_0

def parallel_generatedRoot_14.outs  (ctx : parallel_generatedRoot_14Ctx) : mxxMat2 :=
  (parallel_generatedRoot_14.w_5_0 ctx)

structure parallel_generatedRoot_14.Facts_0  (ctx : parallel_generatedRoot_14Ctx) (outputs : mxxMat2) : Prop where
  t0 : 0 ≤ (45)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat2) => sequential_parallel_generatedRoot_14_5 backend hashModel tape (path ++ [5, i]) params i_0 i (current, ((parallel_generatedRoot_14.w_1_0 ctx), ((parallel_generatedRoot_14.w_2_0 ctx), ((parallel_generatedRoot_14.w_4_0 ctx), ())))) next) (Int.toNat (45)) (parallel_generatedRoot_14.w_0_0 ctx) (parallel_generatedRoot_14.w_5_0 ctx)
  hout : outputs = parallel_generatedRoot_14.outs ctx

theorem parallel_generatedRoot_14.facts_0  {ctx : parallel_generatedRoot_14Ctx} {outputs : mxxMat2}
    (h : parallel_generatedRoot_14.constraints_0 backend hashModel tape path params i_0 (parallel_generatedRoot_14.w_1_0 ctx) (parallel_generatedRoot_14.w_2_0 ctx) (parallel_generatedRoot_14.w_0_0 ctx) (parallel_generatedRoot_14.w_4_0 ctx) (parallel_generatedRoot_14.w_5_0 ctx) outputs) :
    parallel_generatedRoot_14.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure parallel_generatedRoot_14.Facts  (ctx : parallel_generatedRoot_14Ctx) (outputs : mxxMat2) : Prop where
  c0 : parallel_generatedRoot_14.Facts_0 ctx outputs

theorem parallel_generatedRoot_14.facts  (ctx : parallel_generatedRoot_14Ctx) {outputs : mxxMat2}
    (h : parallel_generatedRoot_14.body backend hashModel tape path params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_14.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_14.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) where
  inputs : MxxRuntime.TrapdoorValue (mxxMat4) Unit × ByteArray × mxxMat5 × mxxMat5 × (Fin 256 → mxxMat0) × mxxMat6 × Unit
  witness : generatedRoot.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params}
local notation "generatedRootCtx" => (generatedRoot.Ctx backend hashModel tape path params)

def generatedRoot.w_0_0  (ctx : generatedRootCtx) : MxxRuntime.TrapdoorValue (mxxMat4) Unit :=
  ctx.inputs.1

def generatedRoot.w_2_0  (ctx : generatedRootCtx) : ByteArray :=
  ctx.inputs.2.1

def generatedRoot.w_4_0  (ctx : generatedRootCtx) : mxxMat5 :=
  ctx.inputs.2.2.1

def generatedRoot.w_5_0  (ctx : generatedRootCtx) : mxxMat5 :=
  ctx.inputs.2.2.2.1

def generatedRoot.w_10_0  (ctx : generatedRootCtx) : Fin 256 → mxxMat0 :=
  ctx.inputs.2.2.2.2.1

def generatedRoot.w_19_0  (ctx : generatedRootCtx) : mxxMat6 :=
  ctx.inputs.2.2.2.2.2.1

def generatedRoot.w_1_0  (ctx : generatedRootCtx) : mxxMat4 :=
  MxxRuntime.trapdoorPublic (generatedRoot.w_0_0 ctx)

def generatedRoot.w_3_0  (ctx : generatedRootCtx) : mxxMat7 :=
  ctx.witness.w_3_0

def generatedRoot.w_6_0  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.witness.w_6_0

def generatedRoot.w_7_0  (ctx : generatedRootCtx) : mxxMat8 :=
  ctx.witness.w_7_0

def generatedRoot.w_8_0  (ctx : generatedRootCtx) : mxxMat7 :=
  MxxRuntime.matrixMul (generatedRoot.w_6_0 ctx) (generatedRoot.w_7_0 ctx)

def generatedRoot.w_9_0  (ctx : generatedRootCtx) : mxxMat7 :=
  MxxRuntime.matrixSub (generatedRoot.w_3_0 ctx) (generatedRoot.w_8_0 ctx)

def generatedRoot.w_11_0  (ctx : generatedRootCtx) : mxxMat5 :=
  ctx.witness.w_11_0

def generatedRoot.w_12_0  (ctx : generatedRootCtx) : mxxMat5 :=
  (0 : mxxMat5)

def generatedRoot.w_13_0  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.witness.w_13_0

def generatedRoot.w_14_0  (ctx : generatedRootCtx) : Fin 256 → mxxMat2 :=
  ctx.witness.w_14_0

def generatedRoot.w_15_0  (ctx : generatedRootCtx) : mxxMat2 :=
  ctx.witness.w_15_0

def generatedRoot.w_16_0  (ctx : generatedRootCtx) : mxxMat2 :=
  ctx.witness.w_16_0

def generatedRoot.w_17_0  (ctx : generatedRootCtx) : mxxMat2 :=
  MxxRuntime.matrixAdd (generatedRoot.w_15_0 ctx) (generatedRoot.w_16_0 ctx)

def generatedRoot.w_18_0  (ctx : generatedRootCtx) : mxxMat2 :=
  ctx.witness.w_18_0

def generatedRoot.w_20_0  (ctx : generatedRootCtx) : mxxMat9 :=
  ctx.witness.w_20_0

def generatedRoot.w_21_0  (ctx : generatedRootCtx) : mxxMat5 :=
  ctx.witness.w_21_0

def generatedRoot.w_22_0  (ctx : generatedRootCtx) : mxxMat10 :=
  ctx.witness.w_22_0

def generatedRoot.w_23_0  (ctx : generatedRootCtx) : mxxMat10 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_20_0 ctx) (generatedRoot.w_22_0 ctx)

def generatedRoot.w_24_0  (ctx : generatedRootCtx) : mxxMat5 :=
  ctx.witness.w_24_0

def generatedRoot.w_25_0  (ctx : generatedRootCtx) : mxxMat5 :=
  (generatedRoot.w_24_0 ctx)

def generatedRoot.w_26_0  (ctx : generatedRootCtx) : Fin 256 → mxxMat11 :=
  ctx.witness.w_26_0

def generatedRoot.w_26_1  (ctx : generatedRootCtx) : Fin 256 → mxxMat7 :=
  ctx.witness.w_26_1

def generatedRoot.w_26_2  (ctx : generatedRootCtx) : Fin 256 → mxxMat7 :=
  ctx.witness.w_26_2

def generatedRoot.w_27_0  (ctx : generatedRootCtx) : Fin 256 → mxxMat7 :=
  ctx.witness.w_27_0

def generatedRoot.w_28_0  (ctx : generatedRootCtx) : Fin 256 → mxxMat12 :=
  ctx.witness.w_28_0

def generatedRoot.outs  (ctx : generatedRootCtx) : (Fin 256 → mxxMat12) × (Fin 256 → mxxMat11) × (Fin 256 → mxxMat7) × mxxMat8 × Unit :=
  ((generatedRoot.w_28_0 ctx), ((generatedRoot.w_26_0 ctx), ((generatedRoot.w_26_1 ctx), ((generatedRoot.w_7_0 ctx), ()))))

structure generatedRoot.Facts_0  (ctx : generatedRootCtx) (outputs : (Fin 256 → mxxMat12) × (Fin 256 → mxxMat11) × (Fin 256 → mxxMat7) × mxxMat8 × Unit) : Prop where
  h3_0 : MxxRuntime.hashSample (hashModel) ([109, 120, 120, 45, 112, 101]) ([.bytes [51, 52, 101, 99, 56, 49, 55, 48, 55, 98, 100, 100, 54, 99, 102, 98, 102, 99, 99, 52, 50, 51, 99, 55, 99, 98, 54, 97, 53, 51, 51, 100, 52, 57, 98, 102, 98, 101, 99, 56, 98, 57, 100, 97, 57, 101, 98, 98, 56, 48, 97, 52, 54, 57, 52, 51, 53, 53, 52, 48, 99, 101, 50, 48], .bytes [109, 101, 115, 115, 97, 103, 101, 45, 114, 111, 119], .u64Le (0)]) ((generatedRoot.w_2_0 ctx)) (generatedRoot.w_3_0 ctx)
  h6_0 : MxxRuntime.concatColumns (generatedRoot.w_4_0 ctx) (generatedRoot.w_5_0 ctx) (generatedRoot.w_6_0 ctx)
  h7_0 : MxxRuntime.gaussianSampleAt tape (path ++ [7]) ((mxxNat0/mxxNat1)) (2610114297) (generatedRoot.w_7_0 ctx)
  h11_0 : MxxRuntime.gadgetMatrixRuns backend 4096 30 (generatedRoot.w_11_0 ctx)
  h13_0 : MxxRuntime.concatColumns (generatedRoot.w_11_0 ctx) (generatedRoot.w_12_0 ctx) (generatedRoot.w_13_0 ctx)
  t5 : (256) = 256
  h14_0 : (∀ i : Fin 256, parallel_generatedRoot_14 backend hashModel tape (path ++ [14, i.val]) params i ((generatedRoot.w_2_0 ctx), (((generatedRoot.w_10_0 ctx) i), ((generatedRoot.w_13_0 ctx), ()))) ((generatedRoot.w_14_0 ctx) i))
  h15_0 : MxxRuntime.familyGetStatic (generatedRoot.w_14_0 ctx) (0) (generatedRoot.w_15_0 ctx)
  h16_0 : MxxRuntime.familyGetStatic (generatedRoot.w_14_0 ctx) (1) (generatedRoot.w_16_0 ctx)
  t9 : 0 ≤ (254)
  h18_0 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat2) => sequential_generatedRoot_18 backend hashModel tape (path ++ [18, i]) params i (current, ((generatedRoot.w_14_0 ctx), ())) next) (Int.toNat (254)) (generatedRoot.w_17_0 ctx) (generatedRoot.w_18_0 ctx)
  t11 : 0 ≤ 0
  t12 : 0 < 1
  t13 : 1 ≤ 1
  t14 : 0 ≤ 0
  t15 : 0 < 1
  t16 : 1 ≤ 295
  h20_0 : MxxRuntime.sliceMatrix (generatedRoot.w_19_0 ctx) 0 1 0 1 (generatedRoot.w_20_0 ctx)
  h21_0 : MxxRuntime.gadgetMatrixRuns backend 4096 30 (generatedRoot.w_21_0 ctx)
  h22_0 : MxxRuntime.centeredRebaseRuns (generatedRoot.w_21_0 ctx) (generatedRoot.w_22_0 ctx)
  h24_0 : MxxRuntime.modulusSwitchRuns (generatedRoot.w_23_0 ctx) (generatedRoot.w_24_0 ctx)
  t21 : (256) = 256
  h26_2 : (∀ i : Fin 256, parallel_generatedRoot_26 backend hashModel tape (path ++ [26, i.val]) params i (generatedRoot.w_25_0 ctx) (((generatedRoot.w_26_0 ctx) i), (((generatedRoot.w_26_1 ctx) i), (((generatedRoot.w_26_2 ctx) i), ()))))
  t23 : (256) = 256
  h27_0 : (∀ i : Fin 256, parallel_generatedRoot_27 backend hashModel tape (path ++ [27, i.val]) params i ((generatedRoot.w_9_0 ctx), ((generatedRoot.w_18_0 ctx), (((generatedRoot.w_26_2 ctx) i), ()))) ((generatedRoot.w_27_0 ctx) i))
  t25 : (256) = 256
  h28_0 : (∀ i : Fin 256, parallel_generatedRoot_28 backend hashModel tape (path ++ [28, i.val]) params i ((generatedRoot.w_1_0 ctx), ((generatedRoot.w_0_0 ctx), (((generatedRoot.w_27_0 ctx) i), ()))) ((generatedRoot.w_28_0 ctx) i))
  hout : outputs = generatedRoot.outs ctx

theorem generatedRoot.facts_0  {ctx : generatedRootCtx} {outputs : (Fin 256 → mxxMat12) × (Fin 256 → mxxMat11) × (Fin 256 → mxxMat7) × mxxMat8 × Unit}
    (h : generatedRoot.constraints_0 backend hashModel tape path params (generatedRoot.w_0_0 ctx) (generatedRoot.w_2_0 ctx) (generatedRoot.w_4_0 ctx) (generatedRoot.w_5_0 ctx) (generatedRoot.w_10_0 ctx) (generatedRoot.w_19_0 ctx) (generatedRoot.w_1_0 ctx) (generatedRoot.w_3_0 ctx) (generatedRoot.w_6_0 ctx) (generatedRoot.w_7_0 ctx) (generatedRoot.w_9_0 ctx) (generatedRoot.w_11_0 ctx) (generatedRoot.w_12_0 ctx) (generatedRoot.w_13_0 ctx) (generatedRoot.w_14_0 ctx) (generatedRoot.w_15_0 ctx) (generatedRoot.w_16_0 ctx) (generatedRoot.w_17_0 ctx) (generatedRoot.w_18_0 ctx) (generatedRoot.w_20_0 ctx) (generatedRoot.w_21_0 ctx) (generatedRoot.w_22_0 ctx) (generatedRoot.w_23_0 ctx) (generatedRoot.w_24_0 ctx) (generatedRoot.w_25_0 ctx) (generatedRoot.w_26_0 ctx) (generatedRoot.w_26_1 ctx) (generatedRoot.w_26_2 ctx) (generatedRoot.w_27_0 ctx) (generatedRoot.w_28_0 ctx) outputs) :
    generatedRoot.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts  (ctx : generatedRootCtx) (outputs : (Fin 256 → mxxMat12) × (Fin 256 → mxxMat11) × (Fin 256 → mxxMat7) × mxxMat8 × Unit) : Prop where
  c0 : generatedRoot.Facts_0 ctx outputs

theorem generatedRoot.facts  (ctx : generatedRootCtx) {outputs : (Fin 256 → mxxMat12) × (Fin 256 → mxxMat11) × (Fin 256 → mxxMat7) × mxxMat8 × Unit}
    (h : generatedRoot.body backend hashModel tape path params ctx.inputs outputs ctx.witness) :
    generatedRoot.Facts ctx outputs :=
  have t0 := h
  have r0 := generatedRoot.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

theorem generatedRoot.h14_0_at  {ctx : generatedRootCtx} {outputs : (Fin 256 → mxxMat12) × (Fin 256 → mxxMat11) × (Fin 256 → mxxMat7) × mxxMat8 × Unit}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 256) :
    ∃ w : parallel_generatedRoot_14.Witness, parallel_generatedRoot_14.Facts (⟨((generatedRoot.w_2_0 ctx), (((generatedRoot.w_10_0 ctx) i), ((generatedRoot.w_13_0 ctx), ()))), w⟩ : parallel_generatedRoot_14.Ctx backend hashModel tape (path ++ [14, i.val]) params i) ((generatedRoot.w_14_0 ctx) i) :=
  let ⟨w, h⟩ := F.c0.h14_0 i
  ⟨w, parallel_generatedRoot_14.facts _ h⟩

end

end Stage_keygen

end
