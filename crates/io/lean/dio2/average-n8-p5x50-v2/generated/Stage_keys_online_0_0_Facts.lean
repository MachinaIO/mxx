import Stage_keys_online_0_0

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000
set_option linter.unusedVariables false

-- The wire types repeat the ring modulus `X ^ n + 1`; unfolding it while checking a
-- projection against its field type expands it at the ring dimension.
attribute [local irreducible] Mxx.Primitives.negacyclicModulus

noncomputable section

namespace Stage_keys_online_0_0

local notation "mxxNat0" => 1224430409422355437355602951403039070751149016291967033667224370016023747552
local notation "mxxNat1" => 1577331461107503857687361648278377550964861539914994428956938744873164216254
local notation "mxxNat2" => 231620637999516441280060882203998849893024391639068770271694775176939662912
local notation "mxxNat3" => 286408645742909803465705372805876867840931182364166978575391001671831167302
local notation "mxxNat4" => 298711634382445948711902742905408453456759239282082071569621589367657095967
local notation "mxxNat5" => 29270270146341990167195247313302370525046749248022274291861926128744555509284
local notation "mxxNat6" => 102443552169918673375612164323677696578152944978499947864697498749418303762829
local notation "mxxMat0" => Mxx.Primitives.ExactMatrix modulus_0 8 1 25
local notation "mxxMat1" => Mxx.Primitives.ExactMatrix modulus_0 8 3 25
local notation "mxxMat2" => Mxx.Primitives.ExactMatrix modulus_0 8 2 25
local notation "mxxMat3" => Mxx.Primitives.ExactMatrix modulus_0 8 3 50
local notation "mxxMat4" => Mxx.Primitives.ExactMatrix modulus_0 8 50 25
local notation "mxxMat5" => Mxx.Primitives.ExactMatrix modulus_0 8 1 1
local notation "mxxMat6" => Mxx.Primitives.ExactMatrix modulus_0 8 25 25
local notation "mxxMat7" => Mxx.Primitives.ExactMatrix modulus_0 8 1 22500
local notation "mxxMat8" => Mxx.Primitives.ExactMatrix modulus_0 8 1 900
local notation "mxxMat9" => Mxx.Primitives.ExactMatrix modulus_0 8 1 112500
local notation "mxxMat10" => Mxx.Primitives.ExactMatrix modulus_0 8 1 4500
local notation "mxxMat11" => Mxx.Primitives.ExactMatrix modulus_0 8 1 50
local notation "mxxMat12" => Mxx.Primitives.ExactMatrix modulus_1 8 1 25
local notation "mxxMat13" => Mxx.Primitives.ExactMatrix modulus_1 8 1 1
local notation "mxxMat14" => Mxx.Primitives.ExactMatrix modulus_0 8 1 625
local notation "mxxMat15" => Mxx.Primitives.ExactMatrix modulus_0 8 25 625
local notation "mxxMat16" => Mxx.Primitives.ExactMatrix modulus_0 8 625 625
local notation "mxxMat17" => Mxx.Primitives.ExactMatrix modulus_0 8 25 1
local notation "mxxMat18" => Mxx.Primitives.ExactMatrix modulus_2 8 1 25
local notation "mxxMat19" => Mxx.Primitives.ExactMatrix modulus_2 8 1 1
local notation "mxxMat20" => Mxx.Primitives.ExactMatrix modulus_3 8 1 25
local notation "mxxMat21" => Mxx.Primitives.ExactMatrix modulus_3 8 1 1
local notation "mxxMat22" => Mxx.Primitives.ExactMatrix modulus_4 8 1 25
local notation "mxxMat23" => Mxx.Primitives.ExactMatrix modulus_4 8 1 1
local notation "mxxMat24" => Mxx.Primitives.ExactMatrix modulus_5 8 1 25
local notation "mxxMat25" => Mxx.Primitives.ExactMatrix modulus_5 8 1 1

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_28.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) where
  inputs : (Fin 16 → mxxMat0) × (Fin 16 → mxxMat0) × (Fin 16 → mxxMat0) × (Fin 16 → mxxMat0) × (Fin 16 → mxxMat0) × (Fin 16 → mxxMat0) × Unit
  witness : parallel_generatedRoot_28.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_28Ctx" => (parallel_generatedRoot_28.Ctx backend hashModel tape path params i_0)

def parallel_generatedRoot_28.w_0_0  (ctx : parallel_generatedRoot_28Ctx) : Fin 16 → mxxMat0 :=
  ctx.inputs.1

def parallel_generatedRoot_28.w_1_0  (ctx : parallel_generatedRoot_28Ctx) : Fin 16 → mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_28.w_2_0  (ctx : parallel_generatedRoot_28Ctx) : Fin 16 → mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_28.w_3_0  (ctx : parallel_generatedRoot_28Ctx) : Fin 16 → mxxMat0 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_28.w_4_0  (ctx : parallel_generatedRoot_28Ctx) : Fin 16 → mxxMat0 :=
  ctx.inputs.2.2.2.2.1

def parallel_generatedRoot_28.w_5_0  (ctx : parallel_generatedRoot_28Ctx) : Fin 16 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.1

def parallel_generatedRoot_28.w_6_0  (ctx : parallel_generatedRoot_28Ctx) : Fin 2 → mxxMat0 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_28.w_6_1  (ctx : parallel_generatedRoot_28Ctx) : Fin 2 → mxxMat0 :=
  ctx.witness.w_6_1

def parallel_generatedRoot_28.w_6_2  (ctx : parallel_generatedRoot_28Ctx) : Fin 2 → mxxMat0 :=
  ctx.witness.w_6_2

def parallel_generatedRoot_28.w_6_3  (ctx : parallel_generatedRoot_28Ctx) : Fin 2 → mxxMat0 :=
  ctx.witness.w_6_3

def parallel_generatedRoot_28.w_6_4  (ctx : parallel_generatedRoot_28Ctx) : Fin 2 → mxxMat0 :=
  ctx.witness.w_6_4

def parallel_generatedRoot_28.w_6_5  (ctx : parallel_generatedRoot_28Ctx) : Fin 2 → mxxMat0 :=
  ctx.witness.w_6_5

def parallel_generatedRoot_28.w_7_0  (ctx : parallel_generatedRoot_28Ctx) : mxxMat0 :=
  ctx.witness.w_7_0

def parallel_generatedRoot_28.w_8_0  (ctx : parallel_generatedRoot_28Ctx) : mxxMat0 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_28.w_9_0  (ctx : parallel_generatedRoot_28Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_28.w_7_0 ctx) (parallel_generatedRoot_28.w_8_0 ctx)

def parallel_generatedRoot_28.w_10_0  (ctx : parallel_generatedRoot_28Ctx) : mxxMat0 :=
  ctx.witness.w_10_0

def parallel_generatedRoot_28.w_11_0  (ctx : parallel_generatedRoot_28Ctx) : mxxMat0 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_28.w_12_0  (ctx : parallel_generatedRoot_28Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_28.w_10_0 ctx) (parallel_generatedRoot_28.w_11_0 ctx)

def parallel_generatedRoot_28.w_13_0  (ctx : parallel_generatedRoot_28Ctx) : mxxMat0 :=
  ctx.witness.w_13_0

def parallel_generatedRoot_28.w_14_0  (ctx : parallel_generatedRoot_28Ctx) : mxxMat0 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_28.w_15_0  (ctx : parallel_generatedRoot_28Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_28.w_13_0 ctx) (parallel_generatedRoot_28.w_14_0 ctx)

def parallel_generatedRoot_28.w_16_0  (ctx : parallel_generatedRoot_28Ctx) : mxxMat0 :=
  ctx.witness.w_16_0

def parallel_generatedRoot_28.w_17_0  (ctx : parallel_generatedRoot_28Ctx) : mxxMat0 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_28.w_18_0  (ctx : parallel_generatedRoot_28Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_28.w_16_0 ctx) (parallel_generatedRoot_28.w_17_0 ctx)

def parallel_generatedRoot_28.w_19_0  (ctx : parallel_generatedRoot_28Ctx) : mxxMat0 :=
  ctx.witness.w_19_0

def parallel_generatedRoot_28.w_20_0  (ctx : parallel_generatedRoot_28Ctx) : mxxMat0 :=
  ctx.witness.w_20_0

def parallel_generatedRoot_28.w_21_0  (ctx : parallel_generatedRoot_28Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_28.w_19_0 ctx) (parallel_generatedRoot_28.w_20_0 ctx)

def parallel_generatedRoot_28.w_22_0  (ctx : parallel_generatedRoot_28Ctx) : mxxMat0 :=
  ctx.witness.w_22_0

def parallel_generatedRoot_28.w_23_0  (ctx : parallel_generatedRoot_28Ctx) : mxxMat0 :=
  ctx.witness.w_23_0

def parallel_generatedRoot_28.w_24_0  (ctx : parallel_generatedRoot_28Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_28.w_22_0 ctx) (parallel_generatedRoot_28.w_23_0 ctx)

def parallel_generatedRoot_28.w_25_0  (ctx : parallel_generatedRoot_28Ctx) : mxxMat0 :=
  ctx.witness.w_25_0

def parallel_generatedRoot_28.w_25_1  (ctx : parallel_generatedRoot_28Ctx) : mxxMat0 :=
  ctx.witness.w_25_1

def parallel_generatedRoot_28.w_25_2  (ctx : parallel_generatedRoot_28Ctx) : mxxMat0 :=
  ctx.witness.w_25_2

def parallel_generatedRoot_28.w_25_3  (ctx : parallel_generatedRoot_28Ctx) : mxxMat0 :=
  ctx.witness.w_25_3

def parallel_generatedRoot_28.w_25_4  (ctx : parallel_generatedRoot_28Ctx) : mxxMat0 :=
  ctx.witness.w_25_4

def parallel_generatedRoot_28.w_25_5  (ctx : parallel_generatedRoot_28Ctx) : mxxMat0 :=
  ctx.witness.w_25_5

def parallel_generatedRoot_28.outs  (ctx : parallel_generatedRoot_28Ctx) : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit :=
  ((parallel_generatedRoot_28.w_25_0 ctx), ((parallel_generatedRoot_28.w_25_1 ctx), ((parallel_generatedRoot_28.w_25_2 ctx), ((parallel_generatedRoot_28.w_25_3 ctx), ((parallel_generatedRoot_28.w_25_4 ctx), ((parallel_generatedRoot_28.w_25_5 ctx), ()))))))

structure parallel_generatedRoot_28.Facts_0  (ctx : parallel_generatedRoot_28Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  t0 : (2) = 2
  h6_5 : (∀ i : Fin 2, parallel_parallel_generatedRoot_28_6 backend hashModel tape (path ++ [6, i.val]) params i_0 i ((parallel_generatedRoot_28.w_0_0 ctx), ((parallel_generatedRoot_28.w_1_0 ctx), ((parallel_generatedRoot_28.w_2_0 ctx), ((parallel_generatedRoot_28.w_3_0 ctx), ((parallel_generatedRoot_28.w_4_0 ctx), ((parallel_generatedRoot_28.w_5_0 ctx), ())))))) (((parallel_generatedRoot_28.w_6_0 ctx) i), (((parallel_generatedRoot_28.w_6_1 ctx) i), (((parallel_generatedRoot_28.w_6_2 ctx) i), (((parallel_generatedRoot_28.w_6_3 ctx) i), (((parallel_generatedRoot_28.w_6_4 ctx) i), (((parallel_generatedRoot_28.w_6_5 ctx) i), ())))))))
  h7_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_28.w_6_0 ctx) (0) (parallel_generatedRoot_28.w_7_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_28.w_6_0 ctx) (1) (parallel_generatedRoot_28.w_8_0 ctx)
  h10_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_28.w_6_1 ctx) (0) (parallel_generatedRoot_28.w_10_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_28.w_6_1 ctx) (1) (parallel_generatedRoot_28.w_11_0 ctx)
  h13_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_28.w_6_2 ctx) (0) (parallel_generatedRoot_28.w_13_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_28.w_6_2 ctx) (1) (parallel_generatedRoot_28.w_14_0 ctx)
  h16_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_28.w_6_3 ctx) (0) (parallel_generatedRoot_28.w_16_0 ctx)
  h17_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_28.w_6_3 ctx) (1) (parallel_generatedRoot_28.w_17_0 ctx)
  h19_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_28.w_6_4 ctx) (0) (parallel_generatedRoot_28.w_19_0 ctx)
  h20_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_28.w_6_4 ctx) (1) (parallel_generatedRoot_28.w_20_0 ctx)
  h22_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_28.w_6_5 ctx) (0) (parallel_generatedRoot_28.w_22_0 ctx)
  h23_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_28.w_6_5 ctx) (1) (parallel_generatedRoot_28.w_23_0 ctx)
  t14 : 0 ≤ (0)
  h25_5 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_28_25 backend hashModel tape (path ++ [25, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, ((parallel_generatedRoot_28.w_6_0 ctx), ((parallel_generatedRoot_28.w_6_1 ctx), ((parallel_generatedRoot_28.w_6_2 ctx), ((parallel_generatedRoot_28.w_6_3 ctx), ((parallel_generatedRoot_28.w_6_4 ctx), ((parallel_generatedRoot_28.w_6_5 ctx), ())))))))))))) next) (Int.toNat (0)) ((parallel_generatedRoot_28.w_9_0 ctx), ((parallel_generatedRoot_28.w_12_0 ctx), ((parallel_generatedRoot_28.w_15_0 ctx), ((parallel_generatedRoot_28.w_18_0 ctx), ((parallel_generatedRoot_28.w_21_0 ctx), ((parallel_generatedRoot_28.w_24_0 ctx), ())))))) ((parallel_generatedRoot_28.w_25_0 ctx), ((parallel_generatedRoot_28.w_25_1 ctx), ((parallel_generatedRoot_28.w_25_2 ctx), ((parallel_generatedRoot_28.w_25_3 ctx), ((parallel_generatedRoot_28.w_25_4 ctx), ((parallel_generatedRoot_28.w_25_5 ctx), ()))))))
  hout : outputs = parallel_generatedRoot_28.outs ctx

theorem parallel_generatedRoot_28.facts_0  {ctx : parallel_generatedRoot_28Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_28.constraints_0 backend hashModel tape path params i_0 (parallel_generatedRoot_28.w_0_0 ctx) (parallel_generatedRoot_28.w_1_0 ctx) (parallel_generatedRoot_28.w_2_0 ctx) (parallel_generatedRoot_28.w_3_0 ctx) (parallel_generatedRoot_28.w_4_0 ctx) (parallel_generatedRoot_28.w_5_0 ctx) (parallel_generatedRoot_28.w_6_0 ctx) (parallel_generatedRoot_28.w_6_1 ctx) (parallel_generatedRoot_28.w_6_2 ctx) (parallel_generatedRoot_28.w_6_3 ctx) (parallel_generatedRoot_28.w_6_4 ctx) (parallel_generatedRoot_28.w_6_5 ctx) (parallel_generatedRoot_28.w_7_0 ctx) (parallel_generatedRoot_28.w_8_0 ctx) (parallel_generatedRoot_28.w_9_0 ctx) (parallel_generatedRoot_28.w_10_0 ctx) (parallel_generatedRoot_28.w_11_0 ctx) (parallel_generatedRoot_28.w_12_0 ctx) (parallel_generatedRoot_28.w_13_0 ctx) (parallel_generatedRoot_28.w_14_0 ctx) (parallel_generatedRoot_28.w_15_0 ctx) (parallel_generatedRoot_28.w_16_0 ctx) (parallel_generatedRoot_28.w_17_0 ctx) (parallel_generatedRoot_28.w_18_0 ctx) (parallel_generatedRoot_28.w_19_0 ctx) (parallel_generatedRoot_28.w_20_0 ctx) (parallel_generatedRoot_28.w_21_0 ctx) (parallel_generatedRoot_28.w_22_0 ctx) (parallel_generatedRoot_28.w_23_0 ctx) (parallel_generatedRoot_28.w_24_0 ctx) (parallel_generatedRoot_28.w_25_0 ctx) (parallel_generatedRoot_28.w_25_1 ctx) (parallel_generatedRoot_28.w_25_2 ctx) (parallel_generatedRoot_28.w_25_3 ctx) (parallel_generatedRoot_28.w_25_4 ctx) (parallel_generatedRoot_28.w_25_5 ctx) outputs) :
    parallel_generatedRoot_28.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_28.Facts  (ctx : parallel_generatedRoot_28Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  c0 : parallel_generatedRoot_28.Facts_0 ctx outputs

theorem parallel_generatedRoot_28.facts  (ctx : parallel_generatedRoot_28Ctx) {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_28.body backend hashModel tape path params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_28.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_28.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_32.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) where
  inputs : (Fin 16 → mxxMat0) × (Fin 16 → mxxMat0) × (Fin 16 → mxxMat0) × (Fin 16 → mxxMat0) × (Fin 16 → mxxMat0) × (Fin 16 → mxxMat0) × Unit
  witness : parallel_generatedRoot_32.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_32Ctx" => (parallel_generatedRoot_32.Ctx backend hashModel tape path params i_0)

def parallel_generatedRoot_32.w_0_0  (ctx : parallel_generatedRoot_32Ctx) : Fin 16 → mxxMat0 :=
  ctx.inputs.1

def parallel_generatedRoot_32.w_1_0  (ctx : parallel_generatedRoot_32Ctx) : Fin 16 → mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_32.w_2_0  (ctx : parallel_generatedRoot_32Ctx) : Fin 16 → mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_32.w_3_0  (ctx : parallel_generatedRoot_32Ctx) : Fin 16 → mxxMat0 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_32.w_4_0  (ctx : parallel_generatedRoot_32Ctx) : Fin 16 → mxxMat0 :=
  ctx.inputs.2.2.2.2.1

def parallel_generatedRoot_32.w_5_0  (ctx : parallel_generatedRoot_32Ctx) : Fin 16 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.1

def parallel_generatedRoot_32.w_6_0  (ctx : parallel_generatedRoot_32Ctx) : Fin 2 → mxxMat0 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_32.w_6_1  (ctx : parallel_generatedRoot_32Ctx) : Fin 2 → mxxMat0 :=
  ctx.witness.w_6_1

def parallel_generatedRoot_32.w_6_2  (ctx : parallel_generatedRoot_32Ctx) : Fin 2 → mxxMat0 :=
  ctx.witness.w_6_2

def parallel_generatedRoot_32.w_6_3  (ctx : parallel_generatedRoot_32Ctx) : Fin 2 → mxxMat0 :=
  ctx.witness.w_6_3

def parallel_generatedRoot_32.w_6_4  (ctx : parallel_generatedRoot_32Ctx) : Fin 2 → mxxMat0 :=
  ctx.witness.w_6_4

def parallel_generatedRoot_32.w_6_5  (ctx : parallel_generatedRoot_32Ctx) : Fin 2 → mxxMat0 :=
  ctx.witness.w_6_5

def parallel_generatedRoot_32.w_7_0  (ctx : parallel_generatedRoot_32Ctx) : mxxMat0 :=
  ctx.witness.w_7_0

def parallel_generatedRoot_32.w_8_0  (ctx : parallel_generatedRoot_32Ctx) : mxxMat0 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_32.w_9_0  (ctx : parallel_generatedRoot_32Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_32.w_7_0 ctx) (parallel_generatedRoot_32.w_8_0 ctx)

def parallel_generatedRoot_32.w_10_0  (ctx : parallel_generatedRoot_32Ctx) : mxxMat0 :=
  ctx.witness.w_10_0

def parallel_generatedRoot_32.w_11_0  (ctx : parallel_generatedRoot_32Ctx) : mxxMat0 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_32.w_12_0  (ctx : parallel_generatedRoot_32Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_32.w_10_0 ctx) (parallel_generatedRoot_32.w_11_0 ctx)

def parallel_generatedRoot_32.w_13_0  (ctx : parallel_generatedRoot_32Ctx) : mxxMat0 :=
  ctx.witness.w_13_0

def parallel_generatedRoot_32.w_14_0  (ctx : parallel_generatedRoot_32Ctx) : mxxMat0 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_32.w_15_0  (ctx : parallel_generatedRoot_32Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_32.w_13_0 ctx) (parallel_generatedRoot_32.w_14_0 ctx)

def parallel_generatedRoot_32.w_16_0  (ctx : parallel_generatedRoot_32Ctx) : mxxMat0 :=
  ctx.witness.w_16_0

def parallel_generatedRoot_32.w_17_0  (ctx : parallel_generatedRoot_32Ctx) : mxxMat0 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_32.w_18_0  (ctx : parallel_generatedRoot_32Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_32.w_16_0 ctx) (parallel_generatedRoot_32.w_17_0 ctx)

def parallel_generatedRoot_32.w_19_0  (ctx : parallel_generatedRoot_32Ctx) : mxxMat0 :=
  ctx.witness.w_19_0

def parallel_generatedRoot_32.w_20_0  (ctx : parallel_generatedRoot_32Ctx) : mxxMat0 :=
  ctx.witness.w_20_0

def parallel_generatedRoot_32.w_21_0  (ctx : parallel_generatedRoot_32Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_32.w_19_0 ctx) (parallel_generatedRoot_32.w_20_0 ctx)

def parallel_generatedRoot_32.w_22_0  (ctx : parallel_generatedRoot_32Ctx) : mxxMat0 :=
  ctx.witness.w_22_0

def parallel_generatedRoot_32.w_23_0  (ctx : parallel_generatedRoot_32Ctx) : mxxMat0 :=
  ctx.witness.w_23_0

def parallel_generatedRoot_32.w_24_0  (ctx : parallel_generatedRoot_32Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_32.w_22_0 ctx) (parallel_generatedRoot_32.w_23_0 ctx)

def parallel_generatedRoot_32.w_25_0  (ctx : parallel_generatedRoot_32Ctx) : mxxMat0 :=
  ctx.witness.w_25_0

def parallel_generatedRoot_32.w_25_1  (ctx : parallel_generatedRoot_32Ctx) : mxxMat0 :=
  ctx.witness.w_25_1

def parallel_generatedRoot_32.w_25_2  (ctx : parallel_generatedRoot_32Ctx) : mxxMat0 :=
  ctx.witness.w_25_2

def parallel_generatedRoot_32.w_25_3  (ctx : parallel_generatedRoot_32Ctx) : mxxMat0 :=
  ctx.witness.w_25_3

def parallel_generatedRoot_32.w_25_4  (ctx : parallel_generatedRoot_32Ctx) : mxxMat0 :=
  ctx.witness.w_25_4

def parallel_generatedRoot_32.w_25_5  (ctx : parallel_generatedRoot_32Ctx) : mxxMat0 :=
  ctx.witness.w_25_5

def parallel_generatedRoot_32.outs  (ctx : parallel_generatedRoot_32Ctx) : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit :=
  ((parallel_generatedRoot_32.w_25_0 ctx), ((parallel_generatedRoot_32.w_25_1 ctx), ((parallel_generatedRoot_32.w_25_2 ctx), ((parallel_generatedRoot_32.w_25_3 ctx), ((parallel_generatedRoot_32.w_25_4 ctx), ((parallel_generatedRoot_32.w_25_5 ctx), ()))))))

structure parallel_generatedRoot_32.Facts_0  (ctx : parallel_generatedRoot_32Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  t0 : (2) = 2
  h6_5 : (∀ i : Fin 2, parallel_parallel_generatedRoot_32_6 backend hashModel tape (path ++ [6, i.val]) params i_0 i ((parallel_generatedRoot_32.w_0_0 ctx), ((parallel_generatedRoot_32.w_1_0 ctx), ((parallel_generatedRoot_32.w_2_0 ctx), ((parallel_generatedRoot_32.w_3_0 ctx), ((parallel_generatedRoot_32.w_4_0 ctx), ((parallel_generatedRoot_32.w_5_0 ctx), ())))))) (((parallel_generatedRoot_32.w_6_0 ctx) i), (((parallel_generatedRoot_32.w_6_1 ctx) i), (((parallel_generatedRoot_32.w_6_2 ctx) i), (((parallel_generatedRoot_32.w_6_3 ctx) i), (((parallel_generatedRoot_32.w_6_4 ctx) i), (((parallel_generatedRoot_32.w_6_5 ctx) i), ())))))))
  h7_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_32.w_6_0 ctx) (0) (parallel_generatedRoot_32.w_7_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_32.w_6_0 ctx) (1) (parallel_generatedRoot_32.w_8_0 ctx)
  h10_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_32.w_6_1 ctx) (0) (parallel_generatedRoot_32.w_10_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_32.w_6_1 ctx) (1) (parallel_generatedRoot_32.w_11_0 ctx)
  h13_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_32.w_6_2 ctx) (0) (parallel_generatedRoot_32.w_13_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_32.w_6_2 ctx) (1) (parallel_generatedRoot_32.w_14_0 ctx)
  h16_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_32.w_6_3 ctx) (0) (parallel_generatedRoot_32.w_16_0 ctx)
  h17_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_32.w_6_3 ctx) (1) (parallel_generatedRoot_32.w_17_0 ctx)
  h19_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_32.w_6_4 ctx) (0) (parallel_generatedRoot_32.w_19_0 ctx)
  h20_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_32.w_6_4 ctx) (1) (parallel_generatedRoot_32.w_20_0 ctx)
  h22_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_32.w_6_5 ctx) (0) (parallel_generatedRoot_32.w_22_0 ctx)
  h23_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_32.w_6_5 ctx) (1) (parallel_generatedRoot_32.w_23_0 ctx)
  t14 : 0 ≤ (0)
  h25_5 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_32_25 backend hashModel tape (path ++ [25, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, ((parallel_generatedRoot_32.w_6_0 ctx), ((parallel_generatedRoot_32.w_6_1 ctx), ((parallel_generatedRoot_32.w_6_2 ctx), ((parallel_generatedRoot_32.w_6_3 ctx), ((parallel_generatedRoot_32.w_6_4 ctx), ((parallel_generatedRoot_32.w_6_5 ctx), ())))))))))))) next) (Int.toNat (0)) ((parallel_generatedRoot_32.w_9_0 ctx), ((parallel_generatedRoot_32.w_12_0 ctx), ((parallel_generatedRoot_32.w_15_0 ctx), ((parallel_generatedRoot_32.w_18_0 ctx), ((parallel_generatedRoot_32.w_21_0 ctx), ((parallel_generatedRoot_32.w_24_0 ctx), ())))))) ((parallel_generatedRoot_32.w_25_0 ctx), ((parallel_generatedRoot_32.w_25_1 ctx), ((parallel_generatedRoot_32.w_25_2 ctx), ((parallel_generatedRoot_32.w_25_3 ctx), ((parallel_generatedRoot_32.w_25_4 ctx), ((parallel_generatedRoot_32.w_25_5 ctx), ()))))))
  hout : outputs = parallel_generatedRoot_32.outs ctx

theorem parallel_generatedRoot_32.facts_0  {ctx : parallel_generatedRoot_32Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_32.constraints_0 backend hashModel tape path params i_0 (parallel_generatedRoot_32.w_0_0 ctx) (parallel_generatedRoot_32.w_1_0 ctx) (parallel_generatedRoot_32.w_2_0 ctx) (parallel_generatedRoot_32.w_3_0 ctx) (parallel_generatedRoot_32.w_4_0 ctx) (parallel_generatedRoot_32.w_5_0 ctx) (parallel_generatedRoot_32.w_6_0 ctx) (parallel_generatedRoot_32.w_6_1 ctx) (parallel_generatedRoot_32.w_6_2 ctx) (parallel_generatedRoot_32.w_6_3 ctx) (parallel_generatedRoot_32.w_6_4 ctx) (parallel_generatedRoot_32.w_6_5 ctx) (parallel_generatedRoot_32.w_7_0 ctx) (parallel_generatedRoot_32.w_8_0 ctx) (parallel_generatedRoot_32.w_9_0 ctx) (parallel_generatedRoot_32.w_10_0 ctx) (parallel_generatedRoot_32.w_11_0 ctx) (parallel_generatedRoot_32.w_12_0 ctx) (parallel_generatedRoot_32.w_13_0 ctx) (parallel_generatedRoot_32.w_14_0 ctx) (parallel_generatedRoot_32.w_15_0 ctx) (parallel_generatedRoot_32.w_16_0 ctx) (parallel_generatedRoot_32.w_17_0 ctx) (parallel_generatedRoot_32.w_18_0 ctx) (parallel_generatedRoot_32.w_19_0 ctx) (parallel_generatedRoot_32.w_20_0 ctx) (parallel_generatedRoot_32.w_21_0 ctx) (parallel_generatedRoot_32.w_22_0 ctx) (parallel_generatedRoot_32.w_23_0 ctx) (parallel_generatedRoot_32.w_24_0 ctx) (parallel_generatedRoot_32.w_25_0 ctx) (parallel_generatedRoot_32.w_25_1 ctx) (parallel_generatedRoot_32.w_25_2 ctx) (parallel_generatedRoot_32.w_25_3 ctx) (parallel_generatedRoot_32.w_25_4 ctx) (parallel_generatedRoot_32.w_25_5 ctx) outputs) :
    parallel_generatedRoot_32.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_32.Facts  (ctx : parallel_generatedRoot_32Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  c0 : parallel_generatedRoot_32.Facts_0 ctx outputs

theorem parallel_generatedRoot_32.facts  (ctx : parallel_generatedRoot_32Ctx) {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_32.body backend hashModel tape path params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_32.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_32.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_33.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) where
  inputs : (parallel_generatedRoot_33.InputType)
  witness : parallel_generatedRoot_33.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_33Ctx" => (parallel_generatedRoot_33.Ctx backend hashModel tape path params i_0)

def parallel_generatedRoot_33.w_1_0  (ctx : parallel_generatedRoot_33Ctx) : Fin 8 → mxxMat0 :=
  ctx.inputs.1

def parallel_generatedRoot_33.w_3_0  (ctx : parallel_generatedRoot_33Ctx) : Fin 8 → mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_33.w_5_0  (ctx : parallel_generatedRoot_33Ctx) : Fin 8 → mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_33.w_7_0  (ctx : parallel_generatedRoot_33Ctx) : Fin 8 → mxxMat0 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_33.w_9_0  (ctx : parallel_generatedRoot_33Ctx) : Fin 8 → mxxMat0 :=
  ctx.inputs.2.2.2.2.1

def parallel_generatedRoot_33.w_10_0  (ctx : parallel_generatedRoot_33Ctx) : ByteArray :=
  ctx.inputs.2.2.2.2.2.1

def parallel_generatedRoot_33.w_11_0  (ctx : parallel_generatedRoot_33Ctx) : Fin 8 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.1

def parallel_generatedRoot_33.w_12_0  (ctx : parallel_generatedRoot_33Ctx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.1

def parallel_generatedRoot_33.w_13_0  (ctx : parallel_generatedRoot_33Ctx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_33.w_14_0  (ctx : parallel_generatedRoot_33Ctx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_33.w_15_0  (ctx : parallel_generatedRoot_33Ctx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_33.w_16_0  (ctx : parallel_generatedRoot_33Ctx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_33.w_17_0  (ctx : parallel_generatedRoot_33Ctx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_33.w_0_0  (ctx : parallel_generatedRoot_33Ctx) : mxxMat0 :=
  (0 : mxxMat0)

def parallel_generatedRoot_33.w_2_0  (ctx : parallel_generatedRoot_33Ctx) : mxxMat0 :=
  ctx.witness.w_2_0

def parallel_generatedRoot_33.w_4_0  (ctx : parallel_generatedRoot_33Ctx) : mxxMat0 :=
  ctx.witness.w_4_0

def parallel_generatedRoot_33.w_6_0  (ctx : parallel_generatedRoot_33Ctx) : mxxMat0 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_33.w_8_0  (ctx : parallel_generatedRoot_33Ctx) : mxxMat0 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_33.w_18_0  (ctx : parallel_generatedRoot_33Ctx) : mxxMat0 :=
  ctx.witness.w_18_0

def parallel_generatedRoot_33.w_18_1  (ctx : parallel_generatedRoot_33Ctx) : mxxMat0 :=
  ctx.witness.w_18_1

def parallel_generatedRoot_33.w_18_2  (ctx : parallel_generatedRoot_33Ctx) : mxxMat0 :=
  ctx.witness.w_18_2

def parallel_generatedRoot_33.w_18_3  (ctx : parallel_generatedRoot_33Ctx) : mxxMat0 :=
  ctx.witness.w_18_3

def parallel_generatedRoot_33.w_18_4  (ctx : parallel_generatedRoot_33Ctx) : mxxMat0 :=
  ctx.witness.w_18_4

def parallel_generatedRoot_33.w_18_5  (ctx : parallel_generatedRoot_33Ctx) : mxxMat0 :=
  ctx.witness.w_18_5

def parallel_generatedRoot_33.w_18_6  (ctx : parallel_generatedRoot_33Ctx) : mxxMat0 :=
  ctx.witness.w_18_6

def parallel_generatedRoot_33.w_18_7  (ctx : parallel_generatedRoot_33Ctx) : mxxMat0 :=
  ctx.witness.w_18_7

def parallel_generatedRoot_33.w_18_8  (ctx : parallel_generatedRoot_33Ctx) : mxxMat0 :=
  ctx.witness.w_18_8

def parallel_generatedRoot_33.w_18_9  (ctx : parallel_generatedRoot_33Ctx) : mxxMat0 :=
  ctx.witness.w_18_9

def parallel_generatedRoot_33.w_19_0  (ctx : parallel_generatedRoot_33Ctx) : mxxMat0 :=
  ctx.witness.w_19_0

def parallel_generatedRoot_33.w_20_0  (ctx : parallel_generatedRoot_33Ctx) : Int :=
  (MxxIR.selectInt (7) [(1), (3), (5), (7), (9), (11), (13), (15)])

def parallel_generatedRoot_33.w_21_0  (ctx : parallel_generatedRoot_33Ctx) : mxxMat0 :=
  ctx.witness.w_21_0

def parallel_generatedRoot_33.w_22_0  (ctx : parallel_generatedRoot_33Ctx) : mxxMat0 :=
  ctx.witness.w_22_0

def parallel_generatedRoot_33.w_23_0  (ctx : parallel_generatedRoot_33Ctx) : mxxMat1 :=
  ctx.witness.w_23_0

def parallel_generatedRoot_33.w_23_concat_1  (ctx : parallel_generatedRoot_33Ctx) : mxxMat2 :=
  ctx.witness.w_23_concat_1

def parallel_generatedRoot_33.w_24_0  (ctx : parallel_generatedRoot_33Ctx) : mxxMat0 :=
  ctx.witness.w_24_0

def parallel_generatedRoot_33.w_25_0  (ctx : parallel_generatedRoot_33Ctx) : mxxMat0 :=
  ctx.witness.w_25_0

def parallel_generatedRoot_33.w_26_0  (ctx : parallel_generatedRoot_33Ctx) : mxxMat0 :=
  ctx.witness.w_26_0

def parallel_generatedRoot_33.w_27_0  (ctx : parallel_generatedRoot_33Ctx) : mxxMat1 :=
  ctx.witness.w_27_0

def parallel_generatedRoot_33.w_27_concat_1  (ctx : parallel_generatedRoot_33Ctx) : mxxMat2 :=
  ctx.witness.w_27_concat_1

def parallel_generatedRoot_33.w_28_0  (ctx : parallel_generatedRoot_33Ctx) : mxxMat3 :=
  ctx.witness.w_28_0

def parallel_generatedRoot_33.w_29_0  (ctx : parallel_generatedRoot_33Ctx) : mxxMat0 :=
  ctx.witness.w_29_0

def parallel_generatedRoot_33.w_30_0  (ctx : parallel_generatedRoot_33Ctx) : mxxMat0 :=
  ctx.witness.w_30_0

def parallel_generatedRoot_33.w_31_0  (ctx : parallel_generatedRoot_33Ctx) : mxxMat0 :=
  ctx.witness.w_31_0

def parallel_generatedRoot_33.w_32_0  (ctx : parallel_generatedRoot_33Ctx) : mxxMat1 :=
  ctx.witness.w_32_0

def parallel_generatedRoot_33.w_32_concat_1  (ctx : parallel_generatedRoot_33Ctx) : mxxMat2 :=
  ctx.witness.w_32_concat_1

def parallel_generatedRoot_33.w_33_0  (ctx : parallel_generatedRoot_33Ctx) : mxxMat0 :=
  ctx.witness.w_33_0

def parallel_generatedRoot_33.w_34_0  (ctx : parallel_generatedRoot_33Ctx) : mxxMat0 :=
  ctx.witness.w_34_0

def parallel_generatedRoot_33.w_35_0  (ctx : parallel_generatedRoot_33Ctx) : mxxMat0 :=
  ctx.witness.w_35_0

def parallel_generatedRoot_33.w_36_0  (ctx : parallel_generatedRoot_33Ctx) : mxxMat1 :=
  ctx.witness.w_36_0

def parallel_generatedRoot_33.w_36_concat_1  (ctx : parallel_generatedRoot_33Ctx) : mxxMat2 :=
  ctx.witness.w_36_concat_1

def parallel_generatedRoot_33.w_37_0  (ctx : parallel_generatedRoot_33Ctx) : mxxMat3 :=
  ctx.witness.w_37_0

def parallel_generatedRoot_33.w_38_0  (ctx : parallel_generatedRoot_33Ctx) : mxxMat2 :=
  ctx.witness.w_38_0

def parallel_generatedRoot_33.w_39_0  (ctx : parallel_generatedRoot_33Ctx) : mxxMat4 :=
  ctx.witness.w_39_0

def parallel_generatedRoot_33.w_40_0  (ctx : parallel_generatedRoot_33Ctx) : mxxMat4 :=
  ctx.witness.w_40_0

def parallel_generatedRoot_33.w_41_0  (ctx : parallel_generatedRoot_33Ctx) : mxxMat1 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_33.w_37_0 ctx) (parallel_generatedRoot_33.w_40_0 ctx)

def parallel_generatedRoot_33.w_42_0  (ctx : parallel_generatedRoot_33Ctx) : mxxMat0 :=
  ctx.witness.w_42_0

def parallel_generatedRoot_33.w_43_0  (ctx : parallel_generatedRoot_33Ctx) : mxxMat0 :=
  ctx.witness.w_43_0

def parallel_generatedRoot_33.w_44_0  (ctx : parallel_generatedRoot_33Ctx) : mxxMat2 :=
  ctx.witness.w_44_0

def parallel_generatedRoot_33.w_45_0  (ctx : parallel_generatedRoot_33Ctx) : mxxMat4 :=
  ctx.witness.w_45_0

def parallel_generatedRoot_33.w_46_0  (ctx : parallel_generatedRoot_33Ctx) : mxxMat4 :=
  ctx.witness.w_46_0

def parallel_generatedRoot_33.w_47_0  (ctx : parallel_generatedRoot_33Ctx) : mxxMat1 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_33.w_28_0 ctx) (parallel_generatedRoot_33.w_46_0 ctx)

def parallel_generatedRoot_33.w_48_0  (ctx : parallel_generatedRoot_33Ctx) : mxxMat0 :=
  ctx.witness.w_48_0

def parallel_generatedRoot_33.w_49_0  (ctx : parallel_generatedRoot_33Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_33.w_18_2 ctx) (parallel_generatedRoot_33.w_48_0 ctx)

def parallel_generatedRoot_33.w_50_0  (ctx : parallel_generatedRoot_33Ctx) : mxxMat2 :=
  ctx.witness.w_50_0

def parallel_generatedRoot_33.w_51_0  (ctx : parallel_generatedRoot_33Ctx) : mxxMat4 :=
  ctx.witness.w_51_0

def parallel_generatedRoot_33.w_52_0  (ctx : parallel_generatedRoot_33Ctx) : mxxMat4 :=
  ctx.witness.w_52_0

def parallel_generatedRoot_33.w_53_0  (ctx : parallel_generatedRoot_33Ctx) : mxxMat1 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_33.w_37_0 ctx) (parallel_generatedRoot_33.w_52_0 ctx)

def parallel_generatedRoot_33.w_54_0  (ctx : parallel_generatedRoot_33Ctx) : mxxMat0 :=
  ctx.witness.w_54_0

def parallel_generatedRoot_33.w_55_0  (ctx : parallel_generatedRoot_33Ctx) : mxxMat0 :=
  ctx.witness.w_55_0

def parallel_generatedRoot_33.w_56_0  (ctx : parallel_generatedRoot_33Ctx) : mxxMat2 :=
  ctx.witness.w_56_0

def parallel_generatedRoot_33.w_57_0  (ctx : parallel_generatedRoot_33Ctx) : mxxMat4 :=
  ctx.witness.w_57_0

def parallel_generatedRoot_33.w_58_0  (ctx : parallel_generatedRoot_33Ctx) : mxxMat4 :=
  ctx.witness.w_58_0

def parallel_generatedRoot_33.w_59_0  (ctx : parallel_generatedRoot_33Ctx) : mxxMat1 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_33.w_28_0 ctx) (parallel_generatedRoot_33.w_58_0 ctx)

def parallel_generatedRoot_33.w_60_0  (ctx : parallel_generatedRoot_33Ctx) : mxxMat0 :=
  ctx.witness.w_60_0

def parallel_generatedRoot_33.w_61_0  (ctx : parallel_generatedRoot_33Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_33.w_18_3 ctx) (parallel_generatedRoot_33.w_60_0 ctx)

def parallel_generatedRoot_33.w_62_0  (ctx : parallel_generatedRoot_33Ctx) : mxxMat0 :=
  ctx.witness.w_62_0

def parallel_generatedRoot_33.w_63_0  (ctx : parallel_generatedRoot_33Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_33.w_18_4 ctx) (parallel_generatedRoot_33.w_62_0 ctx)

def parallel_generatedRoot_33.w_64_0  (ctx : parallel_generatedRoot_33Ctx) : mxxMat0 :=
  ctx.witness.w_64_0

def parallel_generatedRoot_33.w_65_0  (ctx : parallel_generatedRoot_33Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_33.w_18_5 ctx) (parallel_generatedRoot_33.w_64_0 ctx)

def parallel_generatedRoot_33.w_66_0  (ctx : parallel_generatedRoot_33Ctx) : mxxMat0 :=
  ctx.witness.w_66_0

def parallel_generatedRoot_33.w_67_0  (ctx : parallel_generatedRoot_33Ctx) : mxxMat0 :=
  ctx.witness.w_67_0

def parallel_generatedRoot_33.w_68_0  (ctx : parallel_generatedRoot_33Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_33.w_66_0 ctx) (parallel_generatedRoot_33.w_67_0 ctx)

def parallel_generatedRoot_33.w_69_0  (ctx : parallel_generatedRoot_33Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_33.w_18_0 ctx) (parallel_generatedRoot_33.w_68_0 ctx)

def parallel_generatedRoot_33.w_70_0  (ctx : parallel_generatedRoot_33Ctx) : mxxMat0 :=
  ctx.witness.w_70_0

def parallel_generatedRoot_33.w_71_0  (ctx : parallel_generatedRoot_33Ctx) : mxxMat0 :=
  ctx.witness.w_71_0

def parallel_generatedRoot_33.w_72_0  (ctx : parallel_generatedRoot_33Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_33.w_70_0 ctx) (parallel_generatedRoot_33.w_71_0 ctx)

def parallel_generatedRoot_33.w_73_0  (ctx : parallel_generatedRoot_33Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_33.w_18_1 ctx) (parallel_generatedRoot_33.w_72_0 ctx)

def parallel_generatedRoot_33.outs  (ctx : parallel_generatedRoot_33Ctx) : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit :=
  ((parallel_generatedRoot_33.w_49_0 ctx), ((parallel_generatedRoot_33.w_61_0 ctx), ((parallel_generatedRoot_33.w_63_0 ctx), ((parallel_generatedRoot_33.w_65_0 ctx), ((parallel_generatedRoot_33.w_69_0 ctx), ((parallel_generatedRoot_33.w_73_0 ctx), ()))))))

structure parallel_generatedRoot_33.Facts_0  (ctx : parallel_generatedRoot_33Ctx) : Prop where
  h2_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_33.w_1_0 ctx) (0) (parallel_generatedRoot_33.w_2_0 ctx)
  h4_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_33.w_3_0 ctx) (0) (parallel_generatedRoot_33.w_4_0 ctx)
  h6_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_33.w_5_0 ctx) (0) (parallel_generatedRoot_33.w_6_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_33.w_7_0 ctx) (0) (parallel_generatedRoot_33.w_8_0 ctx)
  t4 : 0 ≤ (7)
  h18_9 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_33_18 backend hashModel tape (path ++ [18, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (current.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.2.1, ((parallel_generatedRoot_33.w_9_0 ctx), ((parallel_generatedRoot_33.w_10_0 ctx), ((parallel_generatedRoot_33.w_11_0 ctx), ((parallel_generatedRoot_33.w_12_0 ctx), ((parallel_generatedRoot_33.w_13_0 ctx), ((parallel_generatedRoot_33.w_14_0 ctx), ((parallel_generatedRoot_33.w_15_0 ctx), ((parallel_generatedRoot_33.w_16_0 ctx), ((parallel_generatedRoot_33.w_17_0 ctx), ((parallel_generatedRoot_33.w_1_0 ctx), ((parallel_generatedRoot_33.w_3_0 ctx), ((parallel_generatedRoot_33.w_5_0 ctx), ((parallel_generatedRoot_33.w_7_0 ctx), ()))))))))))))))))))))))) next) (Int.toNat (7)) ((parallel_generatedRoot_33.w_0_0 ctx), ((parallel_generatedRoot_33.w_0_0 ctx), ((parallel_generatedRoot_33.w_0_0 ctx), ((parallel_generatedRoot_33.w_0_0 ctx), ((parallel_generatedRoot_33.w_0_0 ctx), ((parallel_generatedRoot_33.w_0_0 ctx), ((parallel_generatedRoot_33.w_2_0 ctx), ((parallel_generatedRoot_33.w_4_0 ctx), ((parallel_generatedRoot_33.w_6_0 ctx), ((parallel_generatedRoot_33.w_8_0 ctx), ())))))))))) ((parallel_generatedRoot_33.w_18_0 ctx), ((parallel_generatedRoot_33.w_18_1 ctx), ((parallel_generatedRoot_33.w_18_2 ctx), ((parallel_generatedRoot_33.w_18_3 ctx), ((parallel_generatedRoot_33.w_18_4 ctx), ((parallel_generatedRoot_33.w_18_5 ctx), ((parallel_generatedRoot_33.w_18_6 ctx), ((parallel_generatedRoot_33.w_18_7 ctx), ((parallel_generatedRoot_33.w_18_8 ctx), ((parallel_generatedRoot_33.w_18_9 ctx), ()))))))))))
  h19_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_33.w_9_0 ctx) (7) (parallel_generatedRoot_33.w_19_0 ctx)
  t7 : 0 ≤ (7)
  t8 : (7) < 8
  h21_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (parallel_generatedRoot_33.w_20_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((parallel_generatedRoot_33.w_10_0 ctx)) (parallel_generatedRoot_33.w_21_0 ctx)
  h22_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (parallel_generatedRoot_33.w_20_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) ((parallel_generatedRoot_33.w_10_0 ctx)) (parallel_generatedRoot_33.w_22_0 ctx)
  h21_0_1 : MxxRuntime.concatRows (parallel_generatedRoot_33.w_19_0 ctx) (parallel_generatedRoot_33.w_21_0 ctx) (parallel_generatedRoot_33.w_23_concat_1 ctx)
  h23_0 : MxxRuntime.concatRows (parallel_generatedRoot_33.w_23_concat_1 ctx) (parallel_generatedRoot_33.w_22_0 ctx) (parallel_generatedRoot_33.w_23_0 ctx)
  h24_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_33.w_11_0 ctx) (7) (parallel_generatedRoot_33.w_24_0 ctx)
  h25_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (parallel_generatedRoot_33.w_20_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((parallel_generatedRoot_33.w_10_0 ctx)) (parallel_generatedRoot_33.w_25_0 ctx)
  h26_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (parallel_generatedRoot_33.w_20_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) ((parallel_generatedRoot_33.w_10_0 ctx)) (parallel_generatedRoot_33.w_26_0 ctx)
  h25_0_1 : MxxRuntime.concatRows (parallel_generatedRoot_33.w_24_0 ctx) (parallel_generatedRoot_33.w_25_0 ctx) (parallel_generatedRoot_33.w_27_concat_1 ctx)
  h27_0 : MxxRuntime.concatRows (parallel_generatedRoot_33.w_27_concat_1 ctx) (parallel_generatedRoot_33.w_26_0 ctx) (parallel_generatedRoot_33.w_27_0 ctx)
  h28_0 : MxxRuntime.concatColumns (parallel_generatedRoot_33.w_23_0 ctx) (parallel_generatedRoot_33.w_27_0 ctx) (parallel_generatedRoot_33.w_28_0 ctx)
  t19 : 0 ≤ (7)
  t20 : (7) < 8
  h29_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (7) [(1), (3), (5), (7), (9), (11), (13), (15)])) (parallel_generatedRoot_33.w_12_0 ctx) (parallel_generatedRoot_33.w_29_0 ctx)
  t22 : 0 ≤ (7)
  t23 : (7) < 8
  h30_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (7) [(1), (3), (5), (7), (9), (11), (13), (15)])) (parallel_generatedRoot_33.w_13_0 ctx) (parallel_generatedRoot_33.w_30_0 ctx)
  t25 : 0 ≤ (7)
  t26 : (7) < 8
  h31_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (7) [(1), (3), (5), (7), (9), (11), (13), (15)])) (parallel_generatedRoot_33.w_14_0 ctx) (parallel_generatedRoot_33.w_31_0 ctx)
  h30_0_1 : MxxRuntime.concatRows (parallel_generatedRoot_33.w_29_0 ctx) (parallel_generatedRoot_33.w_30_0 ctx) (parallel_generatedRoot_33.w_32_concat_1 ctx)
  h32_0 : MxxRuntime.concatRows (parallel_generatedRoot_33.w_32_concat_1 ctx) (parallel_generatedRoot_33.w_31_0 ctx) (parallel_generatedRoot_33.w_32_0 ctx)
  t30 : 0 ≤ (7)
  t31 : (7) < 8

theorem parallel_generatedRoot_33.facts_0  {ctx : parallel_generatedRoot_33Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_33.constraints_0 backend hashModel tape path params i_0 (parallel_generatedRoot_33.w_1_0 ctx) (parallel_generatedRoot_33.w_3_0 ctx) (parallel_generatedRoot_33.w_5_0 ctx) (parallel_generatedRoot_33.w_7_0 ctx) (parallel_generatedRoot_33.w_9_0 ctx) (parallel_generatedRoot_33.w_10_0 ctx) (parallel_generatedRoot_33.w_11_0 ctx) (parallel_generatedRoot_33.w_12_0 ctx) (parallel_generatedRoot_33.w_13_0 ctx) (parallel_generatedRoot_33.w_14_0 ctx) (parallel_generatedRoot_33.w_15_0 ctx) (parallel_generatedRoot_33.w_16_0 ctx) (parallel_generatedRoot_33.w_17_0 ctx) (parallel_generatedRoot_33.w_0_0 ctx) (parallel_generatedRoot_33.w_2_0 ctx) (parallel_generatedRoot_33.w_4_0 ctx) (parallel_generatedRoot_33.w_6_0 ctx) (parallel_generatedRoot_33.w_8_0 ctx) (parallel_generatedRoot_33.w_18_0 ctx) (parallel_generatedRoot_33.w_18_1 ctx) (parallel_generatedRoot_33.w_18_2 ctx) (parallel_generatedRoot_33.w_18_3 ctx) (parallel_generatedRoot_33.w_18_4 ctx) (parallel_generatedRoot_33.w_18_5 ctx) (parallel_generatedRoot_33.w_18_6 ctx) (parallel_generatedRoot_33.w_18_7 ctx) (parallel_generatedRoot_33.w_18_8 ctx) (parallel_generatedRoot_33.w_18_9 ctx) (parallel_generatedRoot_33.w_19_0 ctx) (parallel_generatedRoot_33.w_20_0 ctx) (parallel_generatedRoot_33.w_21_0 ctx) (parallel_generatedRoot_33.w_22_0 ctx) (parallel_generatedRoot_33.w_23_0 ctx) (parallel_generatedRoot_33.w_23_concat_1 ctx) (parallel_generatedRoot_33.w_24_0 ctx) (parallel_generatedRoot_33.w_25_0 ctx) (parallel_generatedRoot_33.w_26_0 ctx) (parallel_generatedRoot_33.w_27_0 ctx) (parallel_generatedRoot_33.w_27_concat_1 ctx) (parallel_generatedRoot_33.w_28_0 ctx) (parallel_generatedRoot_33.w_29_0 ctx) (parallel_generatedRoot_33.w_30_0 ctx) (parallel_generatedRoot_33.w_31_0 ctx) (parallel_generatedRoot_33.w_32_0 ctx) (parallel_generatedRoot_33.w_32_concat_1 ctx)) :
    parallel_generatedRoot_33.Facts_0 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_33.Facts_1  (ctx : parallel_generatedRoot_33Ctx) : Prop where
  h33_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (7) [(1), (3), (5), (7), (9), (11), (13), (15)])) (parallel_generatedRoot_33.w_15_0 ctx) (parallel_generatedRoot_33.w_33_0 ctx)
  t33 : 0 ≤ (7)
  t34 : (7) < 8
  h34_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (7) [(1), (3), (5), (7), (9), (11), (13), (15)])) (parallel_generatedRoot_33.w_16_0 ctx) (parallel_generatedRoot_33.w_34_0 ctx)
  t36 : 0 ≤ (7)
  t37 : (7) < 8
  h35_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (7) [(1), (3), (5), (7), (9), (11), (13), (15)])) (parallel_generatedRoot_33.w_17_0 ctx) (parallel_generatedRoot_33.w_35_0 ctx)
  h34_0_1 : MxxRuntime.concatRows (parallel_generatedRoot_33.w_33_0 ctx) (parallel_generatedRoot_33.w_34_0 ctx) (parallel_generatedRoot_33.w_36_concat_1 ctx)
  h36_0 : MxxRuntime.concatRows (parallel_generatedRoot_33.w_36_concat_1 ctx) (parallel_generatedRoot_33.w_35_0 ctx) (parallel_generatedRoot_33.w_36_0 ctx)
  h37_0 : MxxRuntime.concatColumns (parallel_generatedRoot_33.w_32_0 ctx) (parallel_generatedRoot_33.w_36_0 ctx) (parallel_generatedRoot_33.w_37_0 ctx)
  h38_0 : MxxRuntime.concatRows (parallel_generatedRoot_33.w_18_6 ctx) (parallel_generatedRoot_33.w_18_8 ctx) (parallel_generatedRoot_33.w_38_0 ctx)
  h39_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (parallel_generatedRoot_33.w_38_0 ctx) (parallel_generatedRoot_33.w_39_0 ctx)
  h40_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (parallel_generatedRoot_33.w_39_0 ctx) (parallel_generatedRoot_33.w_40_0 ctx)
  t45 : 0 ≤ 1
  t46 : 1 < 2
  t47 : 2 ≤ 3
  t48 : 0 ≤ 0
  t49 : 0 < 25
  t50 : 25 ≤ 25
  h42_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_33.w_41_0 ctx) 1 2 0 25 (parallel_generatedRoot_33.w_42_0 ctx)
  t52 : 0 ≤ 2
  t53 : 2 < 3
  t54 : 3 ≤ 3
  t55 : 0 ≤ 0
  t56 : 0 < 25
  t57 : 25 ≤ 25
  h43_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_33.w_41_0 ctx) 2 3 0 25 (parallel_generatedRoot_33.w_43_0 ctx)
  h44_0 : MxxRuntime.concatRows (parallel_generatedRoot_33.w_42_0 ctx) (parallel_generatedRoot_33.w_43_0 ctx) (parallel_generatedRoot_33.w_44_0 ctx)
  h45_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (parallel_generatedRoot_33.w_44_0 ctx) (parallel_generatedRoot_33.w_45_0 ctx)
  h46_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (parallel_generatedRoot_33.w_45_0 ctx) (parallel_generatedRoot_33.w_46_0 ctx)
  t62 : 0 ≤ 1
  t63 : 1 < 2

theorem parallel_generatedRoot_33.facts_1  {ctx : parallel_generatedRoot_33Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_33.constraints_1 backend (parallel_generatedRoot_33.w_15_0 ctx) (parallel_generatedRoot_33.w_16_0 ctx) (parallel_generatedRoot_33.w_17_0 ctx) (parallel_generatedRoot_33.w_18_6 ctx) (parallel_generatedRoot_33.w_18_8 ctx) (parallel_generatedRoot_33.w_32_0 ctx) (parallel_generatedRoot_33.w_33_0 ctx) (parallel_generatedRoot_33.w_34_0 ctx) (parallel_generatedRoot_33.w_35_0 ctx) (parallel_generatedRoot_33.w_36_0 ctx) (parallel_generatedRoot_33.w_36_concat_1 ctx) (parallel_generatedRoot_33.w_37_0 ctx) (parallel_generatedRoot_33.w_38_0 ctx) (parallel_generatedRoot_33.w_39_0 ctx) (parallel_generatedRoot_33.w_40_0 ctx) (parallel_generatedRoot_33.w_41_0 ctx) (parallel_generatedRoot_33.w_42_0 ctx) (parallel_generatedRoot_33.w_43_0 ctx) (parallel_generatedRoot_33.w_44_0 ctx) (parallel_generatedRoot_33.w_45_0 ctx) (parallel_generatedRoot_33.w_46_0 ctx)) :
    parallel_generatedRoot_33.Facts_1 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_33.Facts_2  (ctx : parallel_generatedRoot_33Ctx) : Prop where
  t64 : 2 ≤ 3
  t65 : 0 ≤ 0
  t66 : 0 < 25
  t67 : 25 ≤ 25
  h48_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_33.w_47_0 ctx) 1 2 0 25 (parallel_generatedRoot_33.w_48_0 ctx)
  h50_0 : MxxRuntime.concatRows (parallel_generatedRoot_33.w_18_7 ctx) (parallel_generatedRoot_33.w_18_9 ctx) (parallel_generatedRoot_33.w_50_0 ctx)
  h51_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (parallel_generatedRoot_33.w_50_0 ctx) (parallel_generatedRoot_33.w_51_0 ctx)
  h52_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (parallel_generatedRoot_33.w_51_0 ctx) (parallel_generatedRoot_33.w_52_0 ctx)
  t72 : 0 ≤ 1
  t73 : 1 < 2
  t74 : 2 ≤ 3
  t75 : 0 ≤ 0
  t76 : 0 < 25
  t77 : 25 ≤ 25
  h54_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_33.w_53_0 ctx) 1 2 0 25 (parallel_generatedRoot_33.w_54_0 ctx)
  t79 : 0 ≤ 2
  t80 : 2 < 3
  t81 : 3 ≤ 3
  t82 : 0 ≤ 0
  t83 : 0 < 25
  t84 : 25 ≤ 25
  h55_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_33.w_53_0 ctx) 2 3 0 25 (parallel_generatedRoot_33.w_55_0 ctx)
  h56_0 : MxxRuntime.concatRows (parallel_generatedRoot_33.w_54_0 ctx) (parallel_generatedRoot_33.w_55_0 ctx) (parallel_generatedRoot_33.w_56_0 ctx)
  h57_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (parallel_generatedRoot_33.w_56_0 ctx) (parallel_generatedRoot_33.w_57_0 ctx)
  h58_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (parallel_generatedRoot_33.w_57_0 ctx) (parallel_generatedRoot_33.w_58_0 ctx)
  t89 : 0 ≤ 1
  t90 : 1 < 2
  t91 : 2 ≤ 3
  t92 : 0 ≤ 0
  t93 : 0 < 25
  t94 : 25 ≤ 25
  h60_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_33.w_59_0 ctx) 1 2 0 25 (parallel_generatedRoot_33.w_60_0 ctx)

theorem parallel_generatedRoot_33.facts_2  {ctx : parallel_generatedRoot_33Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_33.constraints_2 backend (parallel_generatedRoot_33.w_18_7 ctx) (parallel_generatedRoot_33.w_18_9 ctx) (parallel_generatedRoot_33.w_47_0 ctx) (parallel_generatedRoot_33.w_48_0 ctx) (parallel_generatedRoot_33.w_50_0 ctx) (parallel_generatedRoot_33.w_51_0 ctx) (parallel_generatedRoot_33.w_52_0 ctx) (parallel_generatedRoot_33.w_53_0 ctx) (parallel_generatedRoot_33.w_54_0 ctx) (parallel_generatedRoot_33.w_55_0 ctx) (parallel_generatedRoot_33.w_56_0 ctx) (parallel_generatedRoot_33.w_57_0 ctx) (parallel_generatedRoot_33.w_58_0 ctx) (parallel_generatedRoot_33.w_59_0 ctx) (parallel_generatedRoot_33.w_60_0 ctx)) :
    parallel_generatedRoot_33.Facts_2 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_33.Facts_3  (ctx : parallel_generatedRoot_33Ctx) : Prop where
  t96 : 0 ≤ 2
  t97 : 2 < 3
  t98 : 3 ≤ 3
  t99 : 0 ≤ 0
  t100 : 0 < 25
  t101 : 25 ≤ 25
  h62_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_33.w_47_0 ctx) 2 3 0 25 (parallel_generatedRoot_33.w_62_0 ctx)
  t103 : 0 ≤ 2
  t104 : 2 < 3
  t105 : 3 ≤ 3
  t106 : 0 ≤ 0
  t107 : 0 < 25
  t108 : 25 ≤ 25
  h64_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_33.w_59_0 ctx) 2 3 0 25 (parallel_generatedRoot_33.w_64_0 ctx)
  t110 : 0 ≤ 0
  t111 : 0 < 1
  t112 : 1 ≤ 3
  t113 : 0 ≤ 0
  t114 : 0 < 25
  t115 : 25 ≤ 25
  h66_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_33.w_47_0 ctx) 0 1 0 25 (parallel_generatedRoot_33.w_66_0 ctx)
  t117 : 0 ≤ 0
  t118 : 0 < 1
  t119 : 1 ≤ 3
  t120 : 0 ≤ 0
  t121 : 0 < 25
  t122 : 25 ≤ 25
  h67_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_33.w_41_0 ctx) 0 1 0 25 (parallel_generatedRoot_33.w_67_0 ctx)
  t124 : 0 ≤ 0
  t125 : 0 < 1
  t126 : 1 ≤ 3
  t127 : 0 ≤ 0

theorem parallel_generatedRoot_33.facts_3  {ctx : parallel_generatedRoot_33Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_33.constraints_3 (parallel_generatedRoot_33.w_41_0 ctx) (parallel_generatedRoot_33.w_47_0 ctx) (parallel_generatedRoot_33.w_59_0 ctx) (parallel_generatedRoot_33.w_62_0 ctx) (parallel_generatedRoot_33.w_64_0 ctx) (parallel_generatedRoot_33.w_66_0 ctx) (parallel_generatedRoot_33.w_67_0 ctx)) :
    parallel_generatedRoot_33.Facts_3 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_33.Facts_4  (ctx : parallel_generatedRoot_33Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  t128 : 0 < 25
  t129 : 25 ≤ 25
  h70_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_33.w_59_0 ctx) 0 1 0 25 (parallel_generatedRoot_33.w_70_0 ctx)
  t131 : 0 ≤ 0
  t132 : 0 < 1
  t133 : 1 ≤ 3
  t134 : 0 ≤ 0
  t135 : 0 < 25
  t136 : 25 ≤ 25
  h71_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_33.w_53_0 ctx) 0 1 0 25 (parallel_generatedRoot_33.w_71_0 ctx)
  hout : outputs = parallel_generatedRoot_33.outs ctx

theorem parallel_generatedRoot_33.facts_4  {ctx : parallel_generatedRoot_33Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_33.constraints_4 (parallel_generatedRoot_33.w_49_0 ctx) (parallel_generatedRoot_33.w_53_0 ctx) (parallel_generatedRoot_33.w_59_0 ctx) (parallel_generatedRoot_33.w_61_0 ctx) (parallel_generatedRoot_33.w_63_0 ctx) (parallel_generatedRoot_33.w_65_0 ctx) (parallel_generatedRoot_33.w_69_0 ctx) (parallel_generatedRoot_33.w_70_0 ctx) (parallel_generatedRoot_33.w_71_0 ctx) (parallel_generatedRoot_33.w_73_0 ctx) outputs) :
    parallel_generatedRoot_33.Facts_4 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_33.Facts  (ctx : parallel_generatedRoot_33Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  c0 : parallel_generatedRoot_33.Facts_0 ctx
  c1 : parallel_generatedRoot_33.Facts_1 ctx
  c2 : parallel_generatedRoot_33.Facts_2 ctx
  c3 : parallel_generatedRoot_33.Facts_3 ctx
  c4 : parallel_generatedRoot_33.Facts_4 ctx outputs

theorem parallel_generatedRoot_33.facts  (ctx : parallel_generatedRoot_33Ctx) {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_33.body backend hashModel tape path params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_33.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_33.facts_0 (ctx := ctx) (outputs := outputs) t0.1
  have t1 := t0.2
  have r1 := parallel_generatedRoot_33.facts_1 (ctx := ctx) (outputs := outputs) t1.1
  have t2 := t1.2
  have r2 := parallel_generatedRoot_33.facts_2 (ctx := ctx) (outputs := outputs) t2.1
  have t3 := t2.2
  have r3 := parallel_generatedRoot_33.facts_3 (ctx := ctx) (outputs := outputs) t3.1
  have t4 := t3.2
  have r4 := parallel_generatedRoot_33.facts_4 (ctx := ctx) (outputs := outputs) t4
  ⟨r0, r1, r2, r3, r4⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_34.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) where
  inputs : (parallel_generatedRoot_34.InputType)
  witness : parallel_generatedRoot_34.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_34Ctx" => (parallel_generatedRoot_34.Ctx backend hashModel tape path params i_0)

def parallel_generatedRoot_34.w_1_0  (ctx : parallel_generatedRoot_34Ctx) : Fin 8 → mxxMat0 :=
  ctx.inputs.1

def parallel_generatedRoot_34.w_3_0  (ctx : parallel_generatedRoot_34Ctx) : Fin 8 → mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_34.w_5_0  (ctx : parallel_generatedRoot_34Ctx) : Fin 8 → mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_34.w_7_0  (ctx : parallel_generatedRoot_34Ctx) : Fin 8 → mxxMat0 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_34.w_9_0  (ctx : parallel_generatedRoot_34Ctx) : Fin 8 → mxxMat0 :=
  ctx.inputs.2.2.2.2.1

def parallel_generatedRoot_34.w_10_0  (ctx : parallel_generatedRoot_34Ctx) : ByteArray :=
  ctx.inputs.2.2.2.2.2.1

def parallel_generatedRoot_34.w_11_0  (ctx : parallel_generatedRoot_34Ctx) : Fin 8 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.1

def parallel_generatedRoot_34.w_12_0  (ctx : parallel_generatedRoot_34Ctx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.1

def parallel_generatedRoot_34.w_13_0  (ctx : parallel_generatedRoot_34Ctx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_34.w_14_0  (ctx : parallel_generatedRoot_34Ctx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_34.w_15_0  (ctx : parallel_generatedRoot_34Ctx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_34.w_16_0  (ctx : parallel_generatedRoot_34Ctx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_34.w_17_0  (ctx : parallel_generatedRoot_34Ctx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_34.w_0_0  (ctx : parallel_generatedRoot_34Ctx) : mxxMat0 :=
  (0 : mxxMat0)

def parallel_generatedRoot_34.w_2_0  (ctx : parallel_generatedRoot_34Ctx) : mxxMat0 :=
  ctx.witness.w_2_0

def parallel_generatedRoot_34.w_4_0  (ctx : parallel_generatedRoot_34Ctx) : mxxMat0 :=
  ctx.witness.w_4_0

def parallel_generatedRoot_34.w_6_0  (ctx : parallel_generatedRoot_34Ctx) : mxxMat0 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_34.w_8_0  (ctx : parallel_generatedRoot_34Ctx) : mxxMat0 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_34.w_18_0  (ctx : parallel_generatedRoot_34Ctx) : mxxMat0 :=
  ctx.witness.w_18_0

def parallel_generatedRoot_34.w_18_1  (ctx : parallel_generatedRoot_34Ctx) : mxxMat0 :=
  ctx.witness.w_18_1

def parallel_generatedRoot_34.w_18_2  (ctx : parallel_generatedRoot_34Ctx) : mxxMat0 :=
  ctx.witness.w_18_2

def parallel_generatedRoot_34.w_18_3  (ctx : parallel_generatedRoot_34Ctx) : mxxMat0 :=
  ctx.witness.w_18_3

def parallel_generatedRoot_34.w_18_4  (ctx : parallel_generatedRoot_34Ctx) : mxxMat0 :=
  ctx.witness.w_18_4

def parallel_generatedRoot_34.w_18_5  (ctx : parallel_generatedRoot_34Ctx) : mxxMat0 :=
  ctx.witness.w_18_5

def parallel_generatedRoot_34.w_18_6  (ctx : parallel_generatedRoot_34Ctx) : mxxMat0 :=
  ctx.witness.w_18_6

def parallel_generatedRoot_34.w_18_7  (ctx : parallel_generatedRoot_34Ctx) : mxxMat0 :=
  ctx.witness.w_18_7

def parallel_generatedRoot_34.w_18_8  (ctx : parallel_generatedRoot_34Ctx) : mxxMat0 :=
  ctx.witness.w_18_8

def parallel_generatedRoot_34.w_18_9  (ctx : parallel_generatedRoot_34Ctx) : mxxMat0 :=
  ctx.witness.w_18_9

def parallel_generatedRoot_34.w_19_0  (ctx : parallel_generatedRoot_34Ctx) : mxxMat0 :=
  ctx.witness.w_19_0

def parallel_generatedRoot_34.w_20_0  (ctx : parallel_generatedRoot_34Ctx) : Int :=
  (MxxIR.selectInt (7) [(1), (3), (5), (7), (9), (11), (13), (15)])

def parallel_generatedRoot_34.w_21_0  (ctx : parallel_generatedRoot_34Ctx) : mxxMat0 :=
  ctx.witness.w_21_0

def parallel_generatedRoot_34.w_22_0  (ctx : parallel_generatedRoot_34Ctx) : mxxMat0 :=
  ctx.witness.w_22_0

def parallel_generatedRoot_34.w_23_0  (ctx : parallel_generatedRoot_34Ctx) : mxxMat1 :=
  ctx.witness.w_23_0

def parallel_generatedRoot_34.w_23_concat_1  (ctx : parallel_generatedRoot_34Ctx) : mxxMat2 :=
  ctx.witness.w_23_concat_1

def parallel_generatedRoot_34.w_24_0  (ctx : parallel_generatedRoot_34Ctx) : mxxMat0 :=
  ctx.witness.w_24_0

def parallel_generatedRoot_34.w_25_0  (ctx : parallel_generatedRoot_34Ctx) : mxxMat0 :=
  ctx.witness.w_25_0

def parallel_generatedRoot_34.w_26_0  (ctx : parallel_generatedRoot_34Ctx) : mxxMat0 :=
  ctx.witness.w_26_0

def parallel_generatedRoot_34.w_27_0  (ctx : parallel_generatedRoot_34Ctx) : mxxMat1 :=
  ctx.witness.w_27_0

def parallel_generatedRoot_34.w_27_concat_1  (ctx : parallel_generatedRoot_34Ctx) : mxxMat2 :=
  ctx.witness.w_27_concat_1

def parallel_generatedRoot_34.w_28_0  (ctx : parallel_generatedRoot_34Ctx) : mxxMat3 :=
  ctx.witness.w_28_0

def parallel_generatedRoot_34.w_29_0  (ctx : parallel_generatedRoot_34Ctx) : mxxMat0 :=
  ctx.witness.w_29_0

def parallel_generatedRoot_34.w_30_0  (ctx : parallel_generatedRoot_34Ctx) : mxxMat0 :=
  ctx.witness.w_30_0

def parallel_generatedRoot_34.w_31_0  (ctx : parallel_generatedRoot_34Ctx) : mxxMat0 :=
  ctx.witness.w_31_0

def parallel_generatedRoot_34.w_32_0  (ctx : parallel_generatedRoot_34Ctx) : mxxMat1 :=
  ctx.witness.w_32_0

def parallel_generatedRoot_34.w_32_concat_1  (ctx : parallel_generatedRoot_34Ctx) : mxxMat2 :=
  ctx.witness.w_32_concat_1

def parallel_generatedRoot_34.w_33_0  (ctx : parallel_generatedRoot_34Ctx) : mxxMat0 :=
  ctx.witness.w_33_0

def parallel_generatedRoot_34.w_34_0  (ctx : parallel_generatedRoot_34Ctx) : mxxMat0 :=
  ctx.witness.w_34_0

def parallel_generatedRoot_34.w_35_0  (ctx : parallel_generatedRoot_34Ctx) : mxxMat0 :=
  ctx.witness.w_35_0

def parallel_generatedRoot_34.w_36_0  (ctx : parallel_generatedRoot_34Ctx) : mxxMat1 :=
  ctx.witness.w_36_0

def parallel_generatedRoot_34.w_36_concat_1  (ctx : parallel_generatedRoot_34Ctx) : mxxMat2 :=
  ctx.witness.w_36_concat_1

def parallel_generatedRoot_34.w_37_0  (ctx : parallel_generatedRoot_34Ctx) : mxxMat3 :=
  ctx.witness.w_37_0

def parallel_generatedRoot_34.w_38_0  (ctx : parallel_generatedRoot_34Ctx) : mxxMat2 :=
  ctx.witness.w_38_0

def parallel_generatedRoot_34.w_39_0  (ctx : parallel_generatedRoot_34Ctx) : mxxMat4 :=
  ctx.witness.w_39_0

def parallel_generatedRoot_34.w_40_0  (ctx : parallel_generatedRoot_34Ctx) : mxxMat4 :=
  ctx.witness.w_40_0

def parallel_generatedRoot_34.w_41_0  (ctx : parallel_generatedRoot_34Ctx) : mxxMat1 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_34.w_37_0 ctx) (parallel_generatedRoot_34.w_40_0 ctx)

def parallel_generatedRoot_34.w_42_0  (ctx : parallel_generatedRoot_34Ctx) : mxxMat0 :=
  ctx.witness.w_42_0

def parallel_generatedRoot_34.w_43_0  (ctx : parallel_generatedRoot_34Ctx) : mxxMat0 :=
  ctx.witness.w_43_0

def parallel_generatedRoot_34.w_44_0  (ctx : parallel_generatedRoot_34Ctx) : mxxMat2 :=
  ctx.witness.w_44_0

def parallel_generatedRoot_34.w_45_0  (ctx : parallel_generatedRoot_34Ctx) : mxxMat4 :=
  ctx.witness.w_45_0

def parallel_generatedRoot_34.w_46_0  (ctx : parallel_generatedRoot_34Ctx) : mxxMat4 :=
  ctx.witness.w_46_0

def parallel_generatedRoot_34.w_47_0  (ctx : parallel_generatedRoot_34Ctx) : mxxMat1 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_34.w_28_0 ctx) (parallel_generatedRoot_34.w_46_0 ctx)

def parallel_generatedRoot_34.w_48_0  (ctx : parallel_generatedRoot_34Ctx) : mxxMat0 :=
  ctx.witness.w_48_0

def parallel_generatedRoot_34.w_49_0  (ctx : parallel_generatedRoot_34Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_34.w_18_2 ctx) (parallel_generatedRoot_34.w_48_0 ctx)

def parallel_generatedRoot_34.w_50_0  (ctx : parallel_generatedRoot_34Ctx) : mxxMat2 :=
  ctx.witness.w_50_0

def parallel_generatedRoot_34.w_51_0  (ctx : parallel_generatedRoot_34Ctx) : mxxMat4 :=
  ctx.witness.w_51_0

def parallel_generatedRoot_34.w_52_0  (ctx : parallel_generatedRoot_34Ctx) : mxxMat4 :=
  ctx.witness.w_52_0

def parallel_generatedRoot_34.w_53_0  (ctx : parallel_generatedRoot_34Ctx) : mxxMat1 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_34.w_37_0 ctx) (parallel_generatedRoot_34.w_52_0 ctx)

def parallel_generatedRoot_34.w_54_0  (ctx : parallel_generatedRoot_34Ctx) : mxxMat0 :=
  ctx.witness.w_54_0

def parallel_generatedRoot_34.w_55_0  (ctx : parallel_generatedRoot_34Ctx) : mxxMat0 :=
  ctx.witness.w_55_0

def parallel_generatedRoot_34.w_56_0  (ctx : parallel_generatedRoot_34Ctx) : mxxMat2 :=
  ctx.witness.w_56_0

def parallel_generatedRoot_34.w_57_0  (ctx : parallel_generatedRoot_34Ctx) : mxxMat4 :=
  ctx.witness.w_57_0

def parallel_generatedRoot_34.w_58_0  (ctx : parallel_generatedRoot_34Ctx) : mxxMat4 :=
  ctx.witness.w_58_0

def parallel_generatedRoot_34.w_59_0  (ctx : parallel_generatedRoot_34Ctx) : mxxMat1 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_34.w_28_0 ctx) (parallel_generatedRoot_34.w_58_0 ctx)

def parallel_generatedRoot_34.w_60_0  (ctx : parallel_generatedRoot_34Ctx) : mxxMat0 :=
  ctx.witness.w_60_0

def parallel_generatedRoot_34.w_61_0  (ctx : parallel_generatedRoot_34Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_34.w_18_3 ctx) (parallel_generatedRoot_34.w_60_0 ctx)

def parallel_generatedRoot_34.w_62_0  (ctx : parallel_generatedRoot_34Ctx) : mxxMat0 :=
  ctx.witness.w_62_0

def parallel_generatedRoot_34.w_63_0  (ctx : parallel_generatedRoot_34Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_34.w_18_4 ctx) (parallel_generatedRoot_34.w_62_0 ctx)

def parallel_generatedRoot_34.w_64_0  (ctx : parallel_generatedRoot_34Ctx) : mxxMat0 :=
  ctx.witness.w_64_0

def parallel_generatedRoot_34.w_65_0  (ctx : parallel_generatedRoot_34Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_34.w_18_5 ctx) (parallel_generatedRoot_34.w_64_0 ctx)

def parallel_generatedRoot_34.w_66_0  (ctx : parallel_generatedRoot_34Ctx) : mxxMat0 :=
  ctx.witness.w_66_0

def parallel_generatedRoot_34.w_67_0  (ctx : parallel_generatedRoot_34Ctx) : mxxMat0 :=
  ctx.witness.w_67_0

def parallel_generatedRoot_34.w_68_0  (ctx : parallel_generatedRoot_34Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_34.w_66_0 ctx) (parallel_generatedRoot_34.w_67_0 ctx)

def parallel_generatedRoot_34.w_69_0  (ctx : parallel_generatedRoot_34Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_34.w_18_0 ctx) (parallel_generatedRoot_34.w_68_0 ctx)

def parallel_generatedRoot_34.w_70_0  (ctx : parallel_generatedRoot_34Ctx) : mxxMat0 :=
  ctx.witness.w_70_0

def parallel_generatedRoot_34.w_71_0  (ctx : parallel_generatedRoot_34Ctx) : mxxMat0 :=
  ctx.witness.w_71_0

def parallel_generatedRoot_34.w_72_0  (ctx : parallel_generatedRoot_34Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_34.w_70_0 ctx) (parallel_generatedRoot_34.w_71_0 ctx)

def parallel_generatedRoot_34.w_73_0  (ctx : parallel_generatedRoot_34Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_34.w_18_1 ctx) (parallel_generatedRoot_34.w_72_0 ctx)

def parallel_generatedRoot_34.outs  (ctx : parallel_generatedRoot_34Ctx) : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit :=
  ((parallel_generatedRoot_34.w_49_0 ctx), ((parallel_generatedRoot_34.w_61_0 ctx), ((parallel_generatedRoot_34.w_63_0 ctx), ((parallel_generatedRoot_34.w_65_0 ctx), ((parallel_generatedRoot_34.w_69_0 ctx), ((parallel_generatedRoot_34.w_73_0 ctx), ()))))))

structure parallel_generatedRoot_34.Facts_0  (ctx : parallel_generatedRoot_34Ctx) : Prop where
  h2_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_34.w_1_0 ctx) (0) (parallel_generatedRoot_34.w_2_0 ctx)
  h4_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_34.w_3_0 ctx) (0) (parallel_generatedRoot_34.w_4_0 ctx)
  h6_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_34.w_5_0 ctx) (0) (parallel_generatedRoot_34.w_6_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_34.w_7_0 ctx) (0) (parallel_generatedRoot_34.w_8_0 ctx)
  t4 : 0 ≤ (7)
  h18_9 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_34_18 backend hashModel tape (path ++ [18, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (current.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.2.1, ((parallel_generatedRoot_34.w_9_0 ctx), ((parallel_generatedRoot_34.w_10_0 ctx), ((parallel_generatedRoot_34.w_11_0 ctx), ((parallel_generatedRoot_34.w_12_0 ctx), ((parallel_generatedRoot_34.w_13_0 ctx), ((parallel_generatedRoot_34.w_14_0 ctx), ((parallel_generatedRoot_34.w_15_0 ctx), ((parallel_generatedRoot_34.w_16_0 ctx), ((parallel_generatedRoot_34.w_17_0 ctx), ((parallel_generatedRoot_34.w_1_0 ctx), ((parallel_generatedRoot_34.w_3_0 ctx), ((parallel_generatedRoot_34.w_5_0 ctx), ((parallel_generatedRoot_34.w_7_0 ctx), ()))))))))))))))))))))))) next) (Int.toNat (7)) ((parallel_generatedRoot_34.w_0_0 ctx), ((parallel_generatedRoot_34.w_0_0 ctx), ((parallel_generatedRoot_34.w_0_0 ctx), ((parallel_generatedRoot_34.w_0_0 ctx), ((parallel_generatedRoot_34.w_0_0 ctx), ((parallel_generatedRoot_34.w_0_0 ctx), ((parallel_generatedRoot_34.w_2_0 ctx), ((parallel_generatedRoot_34.w_4_0 ctx), ((parallel_generatedRoot_34.w_6_0 ctx), ((parallel_generatedRoot_34.w_8_0 ctx), ())))))))))) ((parallel_generatedRoot_34.w_18_0 ctx), ((parallel_generatedRoot_34.w_18_1 ctx), ((parallel_generatedRoot_34.w_18_2 ctx), ((parallel_generatedRoot_34.w_18_3 ctx), ((parallel_generatedRoot_34.w_18_4 ctx), ((parallel_generatedRoot_34.w_18_5 ctx), ((parallel_generatedRoot_34.w_18_6 ctx), ((parallel_generatedRoot_34.w_18_7 ctx), ((parallel_generatedRoot_34.w_18_8 ctx), ((parallel_generatedRoot_34.w_18_9 ctx), ()))))))))))
  h19_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_34.w_9_0 ctx) (7) (parallel_generatedRoot_34.w_19_0 ctx)
  t7 : 0 ≤ (7)
  t8 : (7) < 8
  h21_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (parallel_generatedRoot_34.w_20_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((parallel_generatedRoot_34.w_10_0 ctx)) (parallel_generatedRoot_34.w_21_0 ctx)
  h22_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (parallel_generatedRoot_34.w_20_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) ((parallel_generatedRoot_34.w_10_0 ctx)) (parallel_generatedRoot_34.w_22_0 ctx)
  h21_0_1 : MxxRuntime.concatRows (parallel_generatedRoot_34.w_19_0 ctx) (parallel_generatedRoot_34.w_21_0 ctx) (parallel_generatedRoot_34.w_23_concat_1 ctx)
  h23_0 : MxxRuntime.concatRows (parallel_generatedRoot_34.w_23_concat_1 ctx) (parallel_generatedRoot_34.w_22_0 ctx) (parallel_generatedRoot_34.w_23_0 ctx)
  h24_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_34.w_11_0 ctx) (7) (parallel_generatedRoot_34.w_24_0 ctx)
  h25_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (parallel_generatedRoot_34.w_20_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((parallel_generatedRoot_34.w_10_0 ctx)) (parallel_generatedRoot_34.w_25_0 ctx)
  h26_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (parallel_generatedRoot_34.w_20_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) ((parallel_generatedRoot_34.w_10_0 ctx)) (parallel_generatedRoot_34.w_26_0 ctx)
  h25_0_1 : MxxRuntime.concatRows (parallel_generatedRoot_34.w_24_0 ctx) (parallel_generatedRoot_34.w_25_0 ctx) (parallel_generatedRoot_34.w_27_concat_1 ctx)
  h27_0 : MxxRuntime.concatRows (parallel_generatedRoot_34.w_27_concat_1 ctx) (parallel_generatedRoot_34.w_26_0 ctx) (parallel_generatedRoot_34.w_27_0 ctx)
  h28_0 : MxxRuntime.concatColumns (parallel_generatedRoot_34.w_23_0 ctx) (parallel_generatedRoot_34.w_27_0 ctx) (parallel_generatedRoot_34.w_28_0 ctx)
  t19 : 0 ≤ (7)
  t20 : (7) < 8
  h29_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (7) [(1), (3), (5), (7), (9), (11), (13), (15)])) (parallel_generatedRoot_34.w_12_0 ctx) (parallel_generatedRoot_34.w_29_0 ctx)
  t22 : 0 ≤ (7)
  t23 : (7) < 8
  h30_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (7) [(1), (3), (5), (7), (9), (11), (13), (15)])) (parallel_generatedRoot_34.w_13_0 ctx) (parallel_generatedRoot_34.w_30_0 ctx)
  t25 : 0 ≤ (7)
  t26 : (7) < 8
  h31_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (7) [(1), (3), (5), (7), (9), (11), (13), (15)])) (parallel_generatedRoot_34.w_14_0 ctx) (parallel_generatedRoot_34.w_31_0 ctx)
  h30_0_1 : MxxRuntime.concatRows (parallel_generatedRoot_34.w_29_0 ctx) (parallel_generatedRoot_34.w_30_0 ctx) (parallel_generatedRoot_34.w_32_concat_1 ctx)
  h32_0 : MxxRuntime.concatRows (parallel_generatedRoot_34.w_32_concat_1 ctx) (parallel_generatedRoot_34.w_31_0 ctx) (parallel_generatedRoot_34.w_32_0 ctx)
  t30 : 0 ≤ (7)
  t31 : (7) < 8

theorem parallel_generatedRoot_34.facts_0  {ctx : parallel_generatedRoot_34Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_34.constraints_0 backend hashModel tape path params i_0 (parallel_generatedRoot_34.w_1_0 ctx) (parallel_generatedRoot_34.w_3_0 ctx) (parallel_generatedRoot_34.w_5_0 ctx) (parallel_generatedRoot_34.w_7_0 ctx) (parallel_generatedRoot_34.w_9_0 ctx) (parallel_generatedRoot_34.w_10_0 ctx) (parallel_generatedRoot_34.w_11_0 ctx) (parallel_generatedRoot_34.w_12_0 ctx) (parallel_generatedRoot_34.w_13_0 ctx) (parallel_generatedRoot_34.w_14_0 ctx) (parallel_generatedRoot_34.w_15_0 ctx) (parallel_generatedRoot_34.w_16_0 ctx) (parallel_generatedRoot_34.w_17_0 ctx) (parallel_generatedRoot_34.w_0_0 ctx) (parallel_generatedRoot_34.w_2_0 ctx) (parallel_generatedRoot_34.w_4_0 ctx) (parallel_generatedRoot_34.w_6_0 ctx) (parallel_generatedRoot_34.w_8_0 ctx) (parallel_generatedRoot_34.w_18_0 ctx) (parallel_generatedRoot_34.w_18_1 ctx) (parallel_generatedRoot_34.w_18_2 ctx) (parallel_generatedRoot_34.w_18_3 ctx) (parallel_generatedRoot_34.w_18_4 ctx) (parallel_generatedRoot_34.w_18_5 ctx) (parallel_generatedRoot_34.w_18_6 ctx) (parallel_generatedRoot_34.w_18_7 ctx) (parallel_generatedRoot_34.w_18_8 ctx) (parallel_generatedRoot_34.w_18_9 ctx) (parallel_generatedRoot_34.w_19_0 ctx) (parallel_generatedRoot_34.w_20_0 ctx) (parallel_generatedRoot_34.w_21_0 ctx) (parallel_generatedRoot_34.w_22_0 ctx) (parallel_generatedRoot_34.w_23_0 ctx) (parallel_generatedRoot_34.w_23_concat_1 ctx) (parallel_generatedRoot_34.w_24_0 ctx) (parallel_generatedRoot_34.w_25_0 ctx) (parallel_generatedRoot_34.w_26_0 ctx) (parallel_generatedRoot_34.w_27_0 ctx) (parallel_generatedRoot_34.w_27_concat_1 ctx) (parallel_generatedRoot_34.w_28_0 ctx) (parallel_generatedRoot_34.w_29_0 ctx) (parallel_generatedRoot_34.w_30_0 ctx) (parallel_generatedRoot_34.w_31_0 ctx) (parallel_generatedRoot_34.w_32_0 ctx) (parallel_generatedRoot_34.w_32_concat_1 ctx)) :
    parallel_generatedRoot_34.Facts_0 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_34.Facts_1  (ctx : parallel_generatedRoot_34Ctx) : Prop where
  h33_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (7) [(1), (3), (5), (7), (9), (11), (13), (15)])) (parallel_generatedRoot_34.w_15_0 ctx) (parallel_generatedRoot_34.w_33_0 ctx)
  t33 : 0 ≤ (7)
  t34 : (7) < 8
  h34_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (7) [(1), (3), (5), (7), (9), (11), (13), (15)])) (parallel_generatedRoot_34.w_16_0 ctx) (parallel_generatedRoot_34.w_34_0 ctx)
  t36 : 0 ≤ (7)
  t37 : (7) < 8
  h35_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (7) [(1), (3), (5), (7), (9), (11), (13), (15)])) (parallel_generatedRoot_34.w_17_0 ctx) (parallel_generatedRoot_34.w_35_0 ctx)
  h34_0_1 : MxxRuntime.concatRows (parallel_generatedRoot_34.w_33_0 ctx) (parallel_generatedRoot_34.w_34_0 ctx) (parallel_generatedRoot_34.w_36_concat_1 ctx)
  h36_0 : MxxRuntime.concatRows (parallel_generatedRoot_34.w_36_concat_1 ctx) (parallel_generatedRoot_34.w_35_0 ctx) (parallel_generatedRoot_34.w_36_0 ctx)
  h37_0 : MxxRuntime.concatColumns (parallel_generatedRoot_34.w_32_0 ctx) (parallel_generatedRoot_34.w_36_0 ctx) (parallel_generatedRoot_34.w_37_0 ctx)
  h38_0 : MxxRuntime.concatRows (parallel_generatedRoot_34.w_18_6 ctx) (parallel_generatedRoot_34.w_18_8 ctx) (parallel_generatedRoot_34.w_38_0 ctx)
  h39_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (parallel_generatedRoot_34.w_38_0 ctx) (parallel_generatedRoot_34.w_39_0 ctx)
  h40_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (parallel_generatedRoot_34.w_39_0 ctx) (parallel_generatedRoot_34.w_40_0 ctx)
  t45 : 0 ≤ 1
  t46 : 1 < 2
  t47 : 2 ≤ 3
  t48 : 0 ≤ 0
  t49 : 0 < 25
  t50 : 25 ≤ 25
  h42_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_34.w_41_0 ctx) 1 2 0 25 (parallel_generatedRoot_34.w_42_0 ctx)
  t52 : 0 ≤ 2
  t53 : 2 < 3
  t54 : 3 ≤ 3
  t55 : 0 ≤ 0
  t56 : 0 < 25
  t57 : 25 ≤ 25
  h43_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_34.w_41_0 ctx) 2 3 0 25 (parallel_generatedRoot_34.w_43_0 ctx)
  h44_0 : MxxRuntime.concatRows (parallel_generatedRoot_34.w_42_0 ctx) (parallel_generatedRoot_34.w_43_0 ctx) (parallel_generatedRoot_34.w_44_0 ctx)
  h45_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (parallel_generatedRoot_34.w_44_0 ctx) (parallel_generatedRoot_34.w_45_0 ctx)
  h46_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (parallel_generatedRoot_34.w_45_0 ctx) (parallel_generatedRoot_34.w_46_0 ctx)
  t62 : 0 ≤ 1
  t63 : 1 < 2

theorem parallel_generatedRoot_34.facts_1  {ctx : parallel_generatedRoot_34Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_34.constraints_1 backend (parallel_generatedRoot_34.w_15_0 ctx) (parallel_generatedRoot_34.w_16_0 ctx) (parallel_generatedRoot_34.w_17_0 ctx) (parallel_generatedRoot_34.w_18_6 ctx) (parallel_generatedRoot_34.w_18_8 ctx) (parallel_generatedRoot_34.w_32_0 ctx) (parallel_generatedRoot_34.w_33_0 ctx) (parallel_generatedRoot_34.w_34_0 ctx) (parallel_generatedRoot_34.w_35_0 ctx) (parallel_generatedRoot_34.w_36_0 ctx) (parallel_generatedRoot_34.w_36_concat_1 ctx) (parallel_generatedRoot_34.w_37_0 ctx) (parallel_generatedRoot_34.w_38_0 ctx) (parallel_generatedRoot_34.w_39_0 ctx) (parallel_generatedRoot_34.w_40_0 ctx) (parallel_generatedRoot_34.w_41_0 ctx) (parallel_generatedRoot_34.w_42_0 ctx) (parallel_generatedRoot_34.w_43_0 ctx) (parallel_generatedRoot_34.w_44_0 ctx) (parallel_generatedRoot_34.w_45_0 ctx) (parallel_generatedRoot_34.w_46_0 ctx)) :
    parallel_generatedRoot_34.Facts_1 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_34.Facts_2  (ctx : parallel_generatedRoot_34Ctx) : Prop where
  t64 : 2 ≤ 3
  t65 : 0 ≤ 0
  t66 : 0 < 25
  t67 : 25 ≤ 25
  h48_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_34.w_47_0 ctx) 1 2 0 25 (parallel_generatedRoot_34.w_48_0 ctx)
  h50_0 : MxxRuntime.concatRows (parallel_generatedRoot_34.w_18_7 ctx) (parallel_generatedRoot_34.w_18_9 ctx) (parallel_generatedRoot_34.w_50_0 ctx)
  h51_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (parallel_generatedRoot_34.w_50_0 ctx) (parallel_generatedRoot_34.w_51_0 ctx)
  h52_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (parallel_generatedRoot_34.w_51_0 ctx) (parallel_generatedRoot_34.w_52_0 ctx)
  t72 : 0 ≤ 1
  t73 : 1 < 2
  t74 : 2 ≤ 3
  t75 : 0 ≤ 0
  t76 : 0 < 25
  t77 : 25 ≤ 25
  h54_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_34.w_53_0 ctx) 1 2 0 25 (parallel_generatedRoot_34.w_54_0 ctx)
  t79 : 0 ≤ 2
  t80 : 2 < 3
  t81 : 3 ≤ 3
  t82 : 0 ≤ 0
  t83 : 0 < 25
  t84 : 25 ≤ 25
  h55_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_34.w_53_0 ctx) 2 3 0 25 (parallel_generatedRoot_34.w_55_0 ctx)
  h56_0 : MxxRuntime.concatRows (parallel_generatedRoot_34.w_54_0 ctx) (parallel_generatedRoot_34.w_55_0 ctx) (parallel_generatedRoot_34.w_56_0 ctx)
  h57_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (parallel_generatedRoot_34.w_56_0 ctx) (parallel_generatedRoot_34.w_57_0 ctx)
  h58_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (parallel_generatedRoot_34.w_57_0 ctx) (parallel_generatedRoot_34.w_58_0 ctx)
  t89 : 0 ≤ 1
  t90 : 1 < 2
  t91 : 2 ≤ 3
  t92 : 0 ≤ 0
  t93 : 0 < 25
  t94 : 25 ≤ 25
  h60_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_34.w_59_0 ctx) 1 2 0 25 (parallel_generatedRoot_34.w_60_0 ctx)

theorem parallel_generatedRoot_34.facts_2  {ctx : parallel_generatedRoot_34Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_34.constraints_2 backend (parallel_generatedRoot_34.w_18_7 ctx) (parallel_generatedRoot_34.w_18_9 ctx) (parallel_generatedRoot_34.w_47_0 ctx) (parallel_generatedRoot_34.w_48_0 ctx) (parallel_generatedRoot_34.w_50_0 ctx) (parallel_generatedRoot_34.w_51_0 ctx) (parallel_generatedRoot_34.w_52_0 ctx) (parallel_generatedRoot_34.w_53_0 ctx) (parallel_generatedRoot_34.w_54_0 ctx) (parallel_generatedRoot_34.w_55_0 ctx) (parallel_generatedRoot_34.w_56_0 ctx) (parallel_generatedRoot_34.w_57_0 ctx) (parallel_generatedRoot_34.w_58_0 ctx) (parallel_generatedRoot_34.w_59_0 ctx) (parallel_generatedRoot_34.w_60_0 ctx)) :
    parallel_generatedRoot_34.Facts_2 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_34.Facts_3  (ctx : parallel_generatedRoot_34Ctx) : Prop where
  t96 : 0 ≤ 2
  t97 : 2 < 3
  t98 : 3 ≤ 3
  t99 : 0 ≤ 0
  t100 : 0 < 25
  t101 : 25 ≤ 25
  h62_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_34.w_47_0 ctx) 2 3 0 25 (parallel_generatedRoot_34.w_62_0 ctx)
  t103 : 0 ≤ 2
  t104 : 2 < 3
  t105 : 3 ≤ 3
  t106 : 0 ≤ 0
  t107 : 0 < 25
  t108 : 25 ≤ 25
  h64_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_34.w_59_0 ctx) 2 3 0 25 (parallel_generatedRoot_34.w_64_0 ctx)
  t110 : 0 ≤ 0
  t111 : 0 < 1
  t112 : 1 ≤ 3
  t113 : 0 ≤ 0
  t114 : 0 < 25
  t115 : 25 ≤ 25
  h66_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_34.w_47_0 ctx) 0 1 0 25 (parallel_generatedRoot_34.w_66_0 ctx)
  t117 : 0 ≤ 0
  t118 : 0 < 1
  t119 : 1 ≤ 3
  t120 : 0 ≤ 0
  t121 : 0 < 25
  t122 : 25 ≤ 25
  h67_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_34.w_41_0 ctx) 0 1 0 25 (parallel_generatedRoot_34.w_67_0 ctx)
  t124 : 0 ≤ 0
  t125 : 0 < 1
  t126 : 1 ≤ 3
  t127 : 0 ≤ 0

theorem parallel_generatedRoot_34.facts_3  {ctx : parallel_generatedRoot_34Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_34.constraints_3 (parallel_generatedRoot_34.w_41_0 ctx) (parallel_generatedRoot_34.w_47_0 ctx) (parallel_generatedRoot_34.w_59_0 ctx) (parallel_generatedRoot_34.w_62_0 ctx) (parallel_generatedRoot_34.w_64_0 ctx) (parallel_generatedRoot_34.w_66_0 ctx) (parallel_generatedRoot_34.w_67_0 ctx)) :
    parallel_generatedRoot_34.Facts_3 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_34.Facts_4  (ctx : parallel_generatedRoot_34Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  t128 : 0 < 25
  t129 : 25 ≤ 25
  h70_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_34.w_59_0 ctx) 0 1 0 25 (parallel_generatedRoot_34.w_70_0 ctx)
  t131 : 0 ≤ 0
  t132 : 0 < 1
  t133 : 1 ≤ 3
  t134 : 0 ≤ 0
  t135 : 0 < 25
  t136 : 25 ≤ 25
  h71_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_34.w_53_0 ctx) 0 1 0 25 (parallel_generatedRoot_34.w_71_0 ctx)
  hout : outputs = parallel_generatedRoot_34.outs ctx

theorem parallel_generatedRoot_34.facts_4  {ctx : parallel_generatedRoot_34Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_34.constraints_4 (parallel_generatedRoot_34.w_49_0 ctx) (parallel_generatedRoot_34.w_53_0 ctx) (parallel_generatedRoot_34.w_59_0 ctx) (parallel_generatedRoot_34.w_61_0 ctx) (parallel_generatedRoot_34.w_63_0 ctx) (parallel_generatedRoot_34.w_65_0 ctx) (parallel_generatedRoot_34.w_69_0 ctx) (parallel_generatedRoot_34.w_70_0 ctx) (parallel_generatedRoot_34.w_71_0 ctx) (parallel_generatedRoot_34.w_73_0 ctx) outputs) :
    parallel_generatedRoot_34.Facts_4 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_34.Facts  (ctx : parallel_generatedRoot_34Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  c0 : parallel_generatedRoot_34.Facts_0 ctx
  c1 : parallel_generatedRoot_34.Facts_1 ctx
  c2 : parallel_generatedRoot_34.Facts_2 ctx
  c3 : parallel_generatedRoot_34.Facts_3 ctx
  c4 : parallel_generatedRoot_34.Facts_4 ctx outputs

theorem parallel_generatedRoot_34.facts  (ctx : parallel_generatedRoot_34Ctx) {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_34.body backend hashModel tape path params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_34.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_34.facts_0 (ctx := ctx) (outputs := outputs) t0.1
  have t1 := t0.2
  have r1 := parallel_generatedRoot_34.facts_1 (ctx := ctx) (outputs := outputs) t1.1
  have t2 := t1.2
  have r2 := parallel_generatedRoot_34.facts_2 (ctx := ctx) (outputs := outputs) t2.1
  have t3 := t2.2
  have r3 := parallel_generatedRoot_34.facts_3 (ctx := ctx) (outputs := outputs) t3.1
  have t4 := t3.2
  have r4 := parallel_generatedRoot_34.facts_4 (ctx := ctx) (outputs := outputs) t4
  ⟨r0, r1, r2, r3, r4⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_658.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) where
  inputs : (parallel_generatedRoot_658.InputType)
  witness : parallel_generatedRoot_658.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_658Ctx" => (parallel_generatedRoot_658.Ctx backend hashModel tape path params i_0)

def parallel_generatedRoot_658.w_0_0  (ctx : parallel_generatedRoot_658Ctx) : Fin 5 → Int :=
  ctx.inputs.1

def parallel_generatedRoot_658.w_16_0  (ctx : parallel_generatedRoot_658Ctx) : mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_658.w_17_0  (ctx : parallel_generatedRoot_658Ctx) : Fin 4 → mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_658.w_20_0  (ctx : parallel_generatedRoot_658Ctx) : Fin 5 → Int :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_658.w_22_0  (ctx : parallel_generatedRoot_658Ctx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.1

def parallel_generatedRoot_658.w_23_0  (ctx : parallel_generatedRoot_658Ctx) : Fin 4 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.1

def parallel_generatedRoot_658.w_24_0  (ctx : parallel_generatedRoot_658Ctx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.1

def parallel_generatedRoot_658.w_25_0  (ctx : parallel_generatedRoot_658Ctx) : ByteArray :=
  ctx.inputs.2.2.2.2.2.2.2.1

def parallel_generatedRoot_658.w_26_0  (ctx : parallel_generatedRoot_658Ctx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_658.w_27_0  (ctx : parallel_generatedRoot_658Ctx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_658.w_28_0  (ctx : parallel_generatedRoot_658Ctx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_658.w_29_0  (ctx : parallel_generatedRoot_658Ctx) : mxxMat5 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_658.w_30_0  (ctx : parallel_generatedRoot_658Ctx) : Fin 5 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_658.w_35_0  (ctx : parallel_generatedRoot_658Ctx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_658.w_42_0  (ctx : parallel_generatedRoot_658Ctx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_658.w_47_0  (ctx : parallel_generatedRoot_658Ctx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_658.w_52_0  (ctx : parallel_generatedRoot_658Ctx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_658.w_55_0  (ctx : parallel_generatedRoot_658Ctx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_658.w_58_0  (ctx : parallel_generatedRoot_658Ctx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_658.w_62_0  (ctx : parallel_generatedRoot_658Ctx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_658.w_1_0  (ctx : parallel_generatedRoot_658Ctx) : Int :=
  (Int.ofNat i_0)

def parallel_generatedRoot_658.w_2_0  (ctx : parallel_generatedRoot_658Ctx) : Int :=
  5

def parallel_generatedRoot_658.w_3_0  (ctx : parallel_generatedRoot_658Ctx) : Int :=
  ((parallel_generatedRoot_658.w_1_0 ctx) % (parallel_generatedRoot_658.w_2_0 ctx))

def parallel_generatedRoot_658.w_4_0  (ctx : parallel_generatedRoot_658Ctx) : Int :=
  ctx.witness.w_4_0

def parallel_generatedRoot_658.w_5_0  (ctx : parallel_generatedRoot_658Ctx) : Int :=
  2

def parallel_generatedRoot_658.w_6_0  (ctx : parallel_generatedRoot_658Ctx) : Bool :=
  decide ((parallel_generatedRoot_658.w_4_0 ctx) < (parallel_generatedRoot_658.w_5_0 ctx))

def parallel_generatedRoot_658.w_7_0  (ctx : parallel_generatedRoot_658Ctx) : Int :=
  if (parallel_generatedRoot_658.w_6_0 ctx) then 1 else 0

def parallel_generatedRoot_658.w_8_0  (ctx : parallel_generatedRoot_658Ctx) : Int :=
  2

def parallel_generatedRoot_658.w_9_0  (ctx : parallel_generatedRoot_658Ctx) : Int :=
  0

def parallel_generatedRoot_658.w_10_0  (ctx : parallel_generatedRoot_658Ctx) : Int :=
  ((parallel_generatedRoot_658.w_8_0 ctx) + (parallel_generatedRoot_658.w_9_0 ctx))

def parallel_generatedRoot_658.w_11_0  (ctx : parallel_generatedRoot_658Ctx) : Int :=
  ctx.witness.w_11_0

def parallel_generatedRoot_658.w_12_0  (ctx : parallel_generatedRoot_658Ctx) : Int :=
  1

def parallel_generatedRoot_658.w_13_0  (ctx : parallel_generatedRoot_658Ctx) : Int :=
  ((parallel_generatedRoot_658.w_11_0 ctx) + (parallel_generatedRoot_658.w_12_0 ctx))

def parallel_generatedRoot_658.w_14_0  (ctx : parallel_generatedRoot_658Ctx) : Int :=
  2

def parallel_generatedRoot_658.w_15_0  (ctx : parallel_generatedRoot_658Ctx) : Int :=
  ((parallel_generatedRoot_658.w_13_0 ctx) / (parallel_generatedRoot_658.w_14_0 ctx))

def parallel_generatedRoot_658.w_18_0  (ctx : parallel_generatedRoot_658Ctx) : Int :=
  5

def parallel_generatedRoot_658.w_19_0  (ctx : parallel_generatedRoot_658Ctx) : Int :=
  ((parallel_generatedRoot_658.w_1_0 ctx) / (parallel_generatedRoot_658.w_18_0 ctx))

def parallel_generatedRoot_658.w_21_0  (ctx : parallel_generatedRoot_658Ctx) : Int :=
  ctx.witness.w_21_0

def parallel_generatedRoot_658.w_31_0  (ctx : parallel_generatedRoot_658Ctx) : Fin 4 → mxxMat0 :=
  ctx.witness.w_31_0

def parallel_generatedRoot_658.w_31_1  (ctx : parallel_generatedRoot_658Ctx) : Fin 4 → mxxMat0 :=
  ctx.witness.w_31_1

def parallel_generatedRoot_658.w_31_2  (ctx : parallel_generatedRoot_658Ctx) : Fin 4 → mxxMat0 :=
  ctx.witness.w_31_2

def parallel_generatedRoot_658.w_31_3  (ctx : parallel_generatedRoot_658Ctx) : Fin 4 → mxxMat0 :=
  ctx.witness.w_31_3

def parallel_generatedRoot_658.w_31_4  (ctx : parallel_generatedRoot_658Ctx) : Fin 4 → mxxMat0 :=
  ctx.witness.w_31_4

def parallel_generatedRoot_658.w_31_5  (ctx : parallel_generatedRoot_658Ctx) : Fin 4 → mxxMat0 :=
  ctx.witness.w_31_5

def parallel_generatedRoot_658.w_31_6  (ctx : parallel_generatedRoot_658Ctx) : Fin 4 → mxxMat5 :=
  ctx.witness.w_31_6

def parallel_generatedRoot_658.w_32_0  (ctx : parallel_generatedRoot_658Ctx) : Fin 2 → mxxMat0 :=
  ctx.witness.w_32_0

def parallel_generatedRoot_658.w_32_1  (ctx : parallel_generatedRoot_658Ctx) : Fin 2 → mxxMat0 :=
  ctx.witness.w_32_1

def parallel_generatedRoot_658.w_32_2  (ctx : parallel_generatedRoot_658Ctx) : Fin 2 → mxxMat0 :=
  ctx.witness.w_32_2

def parallel_generatedRoot_658.w_32_3  (ctx : parallel_generatedRoot_658Ctx) : Fin 2 → mxxMat0 :=
  ctx.witness.w_32_3

def parallel_generatedRoot_658.w_32_4  (ctx : parallel_generatedRoot_658Ctx) : Fin 2 → mxxMat0 :=
  ctx.witness.w_32_4

def parallel_generatedRoot_658.w_32_5  (ctx : parallel_generatedRoot_658Ctx) : Fin 2 → mxxMat0 :=
  ctx.witness.w_32_5

def parallel_generatedRoot_658.w_32_6  (ctx : parallel_generatedRoot_658Ctx) : Fin 2 → mxxMat5 :=
  ctx.witness.w_32_6

def parallel_generatedRoot_658.w_33_0  (ctx : parallel_generatedRoot_658Ctx) : Fin 1 → mxxMat0 :=
  ctx.witness.w_33_0

def parallel_generatedRoot_658.w_33_1  (ctx : parallel_generatedRoot_658Ctx) : Fin 1 → mxxMat0 :=
  ctx.witness.w_33_1

def parallel_generatedRoot_658.w_33_2  (ctx : parallel_generatedRoot_658Ctx) : Fin 1 → mxxMat0 :=
  ctx.witness.w_33_2

def parallel_generatedRoot_658.w_33_3  (ctx : parallel_generatedRoot_658Ctx) : Fin 1 → mxxMat0 :=
  ctx.witness.w_33_3

def parallel_generatedRoot_658.w_33_4  (ctx : parallel_generatedRoot_658Ctx) : Fin 1 → mxxMat0 :=
  ctx.witness.w_33_4

def parallel_generatedRoot_658.w_33_5  (ctx : parallel_generatedRoot_658Ctx) : Fin 1 → mxxMat0 :=
  ctx.witness.w_33_5

def parallel_generatedRoot_658.w_33_6  (ctx : parallel_generatedRoot_658Ctx) : Fin 1 → mxxMat5 :=
  ctx.witness.w_33_6

def parallel_generatedRoot_658.w_34_0  (ctx : parallel_generatedRoot_658Ctx) : mxxMat0 :=
  ctx.witness.w_34_0

def parallel_generatedRoot_658.w_36_0  (ctx : parallel_generatedRoot_658Ctx) : mxxMat0 :=
  ctx.witness.w_36_0

def parallel_generatedRoot_658.w_37_0  (ctx : parallel_generatedRoot_658Ctx) : mxxMat5 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat5)

def parallel_generatedRoot_658.w_38_0  (ctx : parallel_generatedRoot_658Ctx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarRight (parallel_generatedRoot_658.w_36_0 ctx) (parallel_generatedRoot_658.w_37_0 ctx)

def parallel_generatedRoot_658.w_39_0  (ctx : parallel_generatedRoot_658Ctx) : mxxMat6 :=
  ctx.witness.w_39_0

def parallel_generatedRoot_658.w_40_0  (ctx : parallel_generatedRoot_658Ctx) : mxxMat6 :=
  ctx.witness.w_40_0

def parallel_generatedRoot_658.w_41_0  (ctx : parallel_generatedRoot_658Ctx) : mxxMat0 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_658.w_34_0 ctx) (parallel_generatedRoot_658.w_40_0 ctx)

def parallel_generatedRoot_658.w_43_0  (ctx : parallel_generatedRoot_658Ctx) : mxxMat6 :=
  ctx.witness.w_43_0

def parallel_generatedRoot_658.w_44_0  (ctx : parallel_generatedRoot_658Ctx) : mxxMat0 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_658.w_41_0 ctx) (parallel_generatedRoot_658.w_43_0 ctx)

def parallel_generatedRoot_658.w_45_0  (ctx : parallel_generatedRoot_658Ctx) : mxxMat0 :=
  ctx.witness.w_45_0

def parallel_generatedRoot_658.w_46_0  (ctx : parallel_generatedRoot_658Ctx) : mxxMat0 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_658.w_45_0 ctx) (parallel_generatedRoot_658.w_40_0 ctx)

def parallel_generatedRoot_658.w_48_0  (ctx : parallel_generatedRoot_658Ctx) : mxxMat6 :=
  ctx.witness.w_48_0

def parallel_generatedRoot_658.w_49_0  (ctx : parallel_generatedRoot_658Ctx) : mxxMat0 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_658.w_46_0 ctx) (parallel_generatedRoot_658.w_48_0 ctx)

def parallel_generatedRoot_658.w_50_0  (ctx : parallel_generatedRoot_658Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_658.w_44_0 ctx) (parallel_generatedRoot_658.w_49_0 ctx)

def parallel_generatedRoot_658.w_51_0  (ctx : parallel_generatedRoot_658Ctx) : mxxMat5 :=
  ctx.witness.w_51_0

def parallel_generatedRoot_658.w_53_0  (ctx : parallel_generatedRoot_658Ctx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_658.w_51_0 ctx) (parallel_generatedRoot_658.w_52_0 ctx)

def parallel_generatedRoot_658.w_54_0  (ctx : parallel_generatedRoot_658Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_658.w_50_0 ctx) (parallel_generatedRoot_658.w_53_0 ctx)

def parallel_generatedRoot_658.w_56_0  (ctx : parallel_generatedRoot_658Ctx) : mxxMat6 :=
  ctx.witness.w_56_0

def parallel_generatedRoot_658.w_57_0  (ctx : parallel_generatedRoot_658Ctx) : mxxMat0 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_658.w_41_0 ctx) (parallel_generatedRoot_658.w_56_0 ctx)

def parallel_generatedRoot_658.w_59_0  (ctx : parallel_generatedRoot_658Ctx) : mxxMat6 :=
  ctx.witness.w_59_0

def parallel_generatedRoot_658.w_60_0  (ctx : parallel_generatedRoot_658Ctx) : mxxMat0 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_658.w_46_0 ctx) (parallel_generatedRoot_658.w_59_0 ctx)

def parallel_generatedRoot_658.w_61_0  (ctx : parallel_generatedRoot_658Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_658.w_57_0 ctx) (parallel_generatedRoot_658.w_60_0 ctx)

def parallel_generatedRoot_658.w_63_0  (ctx : parallel_generatedRoot_658Ctx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_658.w_51_0 ctx) (parallel_generatedRoot_658.w_62_0 ctx)

def parallel_generatedRoot_658.w_64_0  (ctx : parallel_generatedRoot_658Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_658.w_61_0 ctx) (parallel_generatedRoot_658.w_63_0 ctx)

def parallel_generatedRoot_658.w_65_0  (ctx : parallel_generatedRoot_658Ctx) : mxxMat0 :=
  ctx.witness.w_65_0

def parallel_generatedRoot_658.w_66_0  (ctx : parallel_generatedRoot_658Ctx) : mxxMat0 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_658.w_65_0 ctx) (parallel_generatedRoot_658.w_40_0 ctx)

def parallel_generatedRoot_658.w_67_0  (ctx : parallel_generatedRoot_658Ctx) : mxxMat0 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_658.w_66_0 ctx) (parallel_generatedRoot_658.w_43_0 ctx)

def parallel_generatedRoot_658.w_68_0  (ctx : parallel_generatedRoot_658Ctx) : mxxMat0 :=
  ctx.witness.w_68_0

def parallel_generatedRoot_658.w_69_0  (ctx : parallel_generatedRoot_658Ctx) : mxxMat0 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_658.w_68_0 ctx) (parallel_generatedRoot_658.w_40_0 ctx)

def parallel_generatedRoot_658.w_70_0  (ctx : parallel_generatedRoot_658Ctx) : mxxMat0 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_658.w_69_0 ctx) (parallel_generatedRoot_658.w_48_0 ctx)

def parallel_generatedRoot_658.w_71_0  (ctx : parallel_generatedRoot_658Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_658.w_67_0 ctx) (parallel_generatedRoot_658.w_70_0 ctx)

def parallel_generatedRoot_658.w_72_0  (ctx : parallel_generatedRoot_658Ctx) : mxxMat0 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_658.w_66_0 ctx) (parallel_generatedRoot_658.w_56_0 ctx)

def parallel_generatedRoot_658.w_73_0  (ctx : parallel_generatedRoot_658Ctx) : mxxMat0 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_658.w_69_0 ctx) (parallel_generatedRoot_658.w_59_0 ctx)

def parallel_generatedRoot_658.w_74_0  (ctx : parallel_generatedRoot_658Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_658.w_72_0 ctx) (parallel_generatedRoot_658.w_73_0 ctx)

def parallel_generatedRoot_658.w_75_0  (ctx : parallel_generatedRoot_658Ctx) : mxxMat0 :=
  ctx.witness.w_75_0

def parallel_generatedRoot_658.w_76_0  (ctx : parallel_generatedRoot_658Ctx) : mxxMat0 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_658.w_75_0 ctx) (parallel_generatedRoot_658.w_40_0 ctx)

def parallel_generatedRoot_658.w_77_0  (ctx : parallel_generatedRoot_658Ctx) : mxxMat0 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_658.w_76_0 ctx) (parallel_generatedRoot_658.w_43_0 ctx)

def parallel_generatedRoot_658.w_78_0  (ctx : parallel_generatedRoot_658Ctx) : mxxMat0 :=
  ctx.witness.w_78_0

def parallel_generatedRoot_658.w_79_0  (ctx : parallel_generatedRoot_658Ctx) : mxxMat0 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_658.w_78_0 ctx) (parallel_generatedRoot_658.w_40_0 ctx)

def parallel_generatedRoot_658.w_80_0  (ctx : parallel_generatedRoot_658Ctx) : mxxMat0 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_658.w_79_0 ctx) (parallel_generatedRoot_658.w_48_0 ctx)

def parallel_generatedRoot_658.w_81_0  (ctx : parallel_generatedRoot_658Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_658.w_77_0 ctx) (parallel_generatedRoot_658.w_80_0 ctx)

def parallel_generatedRoot_658.w_82_0  (ctx : parallel_generatedRoot_658Ctx) : mxxMat0 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_658.w_76_0 ctx) (parallel_generatedRoot_658.w_56_0 ctx)

def parallel_generatedRoot_658.w_83_0  (ctx : parallel_generatedRoot_658Ctx) : mxxMat0 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_658.w_79_0 ctx) (parallel_generatedRoot_658.w_59_0 ctx)

def parallel_generatedRoot_658.w_84_0  (ctx : parallel_generatedRoot_658Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_658.w_82_0 ctx) (parallel_generatedRoot_658.w_83_0 ctx)

def parallel_generatedRoot_658.outs  (ctx : parallel_generatedRoot_658Ctx) : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit :=
  ((parallel_generatedRoot_658.w_54_0 ctx), ((parallel_generatedRoot_658.w_64_0 ctx), ((parallel_generatedRoot_658.w_71_0 ctx), ((parallel_generatedRoot_658.w_74_0 ctx), ((parallel_generatedRoot_658.w_81_0 ctx), ((parallel_generatedRoot_658.w_84_0 ctx), ()))))))

structure parallel_generatedRoot_658.Facts_0  (ctx : parallel_generatedRoot_658Ctx) : Prop where
  h2_0 : (parallel_generatedRoot_658.w_2_0 ctx) ≠ 0
  h3_0 : 0 ≤ (parallel_generatedRoot_658.w_3_0 ctx)
  h3_0_1 : (parallel_generatedRoot_658.w_3_0 ctx) < 5
  h4_0 : MxxRuntime.familyGetDynamic (parallel_generatedRoot_658.w_0_0 ctx) (parallel_generatedRoot_658.w_3_0 ctx) (parallel_generatedRoot_658.w_4_0 ctx)
  h7_0 : 0 ≤ (parallel_generatedRoot_658.w_7_0 ctx)
  h7_0_1 : (parallel_generatedRoot_658.w_7_0 ctx) < 2
  t6 : 2 = 2
  h11_0 : MxxRuntime.select (parallel_generatedRoot_658.w_7_0 ctx) [(parallel_generatedRoot_658.w_4_0 ctx), (parallel_generatedRoot_658.w_10_0 ctx)] (parallel_generatedRoot_658.w_11_0 ctx)
  h14_0 : (parallel_generatedRoot_658.w_14_0 ctx) ≠ 0
  h18_0 : (parallel_generatedRoot_658.w_18_0 ctx) ≠ 0
  h3_0_2 : 0 ≤ (parallel_generatedRoot_658.w_3_0 ctx)
  h3_0_3 : (parallel_generatedRoot_658.w_3_0 ctx) < 5
  h21_0 : MxxRuntime.familyGetDynamic (parallel_generatedRoot_658.w_20_0 ctx) (parallel_generatedRoot_658.w_3_0 ctx) (parallel_generatedRoot_658.w_21_0 ctx)
  t13 : (4) = 4
  h31_6 : (∀ i : Fin 4, parallel_parallel_generatedRoot_658_31 backend hashModel tape (path ++ [31, i.val]) params i_0 i ((parallel_generatedRoot_658.w_4_0 ctx), ((parallel_generatedRoot_658.w_16_0 ctx), ((parallel_generatedRoot_658.w_17_0 ctx), ((parallel_generatedRoot_658.w_19_0 ctx), ((parallel_generatedRoot_658.w_21_0 ctx), ((parallel_generatedRoot_658.w_22_0 ctx), ((parallel_generatedRoot_658.w_23_0 ctx), ((parallel_generatedRoot_658.w_24_0 ctx), ((parallel_generatedRoot_658.w_25_0 ctx), ((parallel_generatedRoot_658.w_26_0 ctx), ((parallel_generatedRoot_658.w_27_0 ctx), ((parallel_generatedRoot_658.w_28_0 ctx), ((parallel_generatedRoot_658.w_29_0 ctx), ((parallel_generatedRoot_658.w_30_0 ctx), ())))))))))))))) (((parallel_generatedRoot_658.w_31_0 ctx) i), (((parallel_generatedRoot_658.w_31_1 ctx) i), (((parallel_generatedRoot_658.w_31_2 ctx) i), (((parallel_generatedRoot_658.w_31_3 ctx) i), (((parallel_generatedRoot_658.w_31_4 ctx) i), (((parallel_generatedRoot_658.w_31_5 ctx) i), (((parallel_generatedRoot_658.w_31_6 ctx) i), ()))))))))
  t15 : (2) = 2
  h32_6 : (∀ i : Fin 2, parallel_parallel_generatedRoot_658_32 backend hashModel tape (path ++ [32, i.val]) params i_0 i ((parallel_generatedRoot_658.w_11_0 ctx), ((parallel_generatedRoot_658.w_31_0 ctx), ((parallel_generatedRoot_658.w_31_2 ctx), ((parallel_generatedRoot_658.w_31_1 ctx), ((parallel_generatedRoot_658.w_31_4 ctx), ((parallel_generatedRoot_658.w_31_6 ctx), ((parallel_generatedRoot_658.w_31_3 ctx), ((parallel_generatedRoot_658.w_31_5 ctx), ())))))))) (((parallel_generatedRoot_658.w_32_0 ctx) i), (((parallel_generatedRoot_658.w_32_1 ctx) i), (((parallel_generatedRoot_658.w_32_2 ctx) i), (((parallel_generatedRoot_658.w_32_3 ctx) i), (((parallel_generatedRoot_658.w_32_4 ctx) i), (((parallel_generatedRoot_658.w_32_5 ctx) i), (((parallel_generatedRoot_658.w_32_6 ctx) i), ()))))))))
  t17 : (1) = 1
  h33_6 : (∀ i : Fin 1, parallel_parallel_generatedRoot_658_33 backend hashModel tape (path ++ [33, i.val]) params i_0 i ((parallel_generatedRoot_658.w_15_0 ctx), ((parallel_generatedRoot_658.w_32_0 ctx), ((parallel_generatedRoot_658.w_32_2 ctx), ((parallel_generatedRoot_658.w_32_1 ctx), ((parallel_generatedRoot_658.w_32_4 ctx), ((parallel_generatedRoot_658.w_32_6 ctx), ((parallel_generatedRoot_658.w_32_3 ctx), ((parallel_generatedRoot_658.w_32_5 ctx), ())))))))) (((parallel_generatedRoot_658.w_33_0 ctx) i), (((parallel_generatedRoot_658.w_33_1 ctx) i), (((parallel_generatedRoot_658.w_33_2 ctx) i), (((parallel_generatedRoot_658.w_33_3 ctx) i), (((parallel_generatedRoot_658.w_33_4 ctx) i), (((parallel_generatedRoot_658.w_33_5 ctx) i), (((parallel_generatedRoot_658.w_33_6 ctx) i), ()))))))))
  h34_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_658.w_33_0 ctx) (0) (parallel_generatedRoot_658.w_34_0 ctx)
  h36_0 : MxxRuntime.centeredRebaseRuns (parallel_generatedRoot_658.w_35_0 ctx) (parallel_generatedRoot_658.w_36_0 ctx)
  h39_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (parallel_generatedRoot_658.w_38_0 ctx) (parallel_generatedRoot_658.w_39_0 ctx)
  h40_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (parallel_generatedRoot_658.w_39_0 ctx) (parallel_generatedRoot_658.w_40_0 ctx)
  h43_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (parallel_generatedRoot_658.w_42_0 ctx) (parallel_generatedRoot_658.w_43_0 ctx)
  h45_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_658.w_33_1 ctx) (0) (parallel_generatedRoot_658.w_45_0 ctx)
  h48_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (parallel_generatedRoot_658.w_47_0 ctx) (parallel_generatedRoot_658.w_48_0 ctx)
  h51_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_658.w_33_6 ctx) (0) (parallel_generatedRoot_658.w_51_0 ctx)
  h56_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (parallel_generatedRoot_658.w_55_0 ctx) (parallel_generatedRoot_658.w_56_0 ctx)
  h59_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (parallel_generatedRoot_658.w_58_0 ctx) (parallel_generatedRoot_658.w_59_0 ctx)
  h65_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_658.w_33_2 ctx) (0) (parallel_generatedRoot_658.w_65_0 ctx)
  h68_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_658.w_33_3 ctx) (0) (parallel_generatedRoot_658.w_68_0 ctx)
  h75_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_658.w_33_4 ctx) (0) (parallel_generatedRoot_658.w_75_0 ctx)

theorem parallel_generatedRoot_658.facts_0  {ctx : parallel_generatedRoot_658Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_658.constraints_0 backend hashModel tape path params i_0 (parallel_generatedRoot_658.w_0_0 ctx) (parallel_generatedRoot_658.w_16_0 ctx) (parallel_generatedRoot_658.w_17_0 ctx) (parallel_generatedRoot_658.w_20_0 ctx) (parallel_generatedRoot_658.w_22_0 ctx) (parallel_generatedRoot_658.w_23_0 ctx) (parallel_generatedRoot_658.w_24_0 ctx) (parallel_generatedRoot_658.w_25_0 ctx) (parallel_generatedRoot_658.w_26_0 ctx) (parallel_generatedRoot_658.w_27_0 ctx) (parallel_generatedRoot_658.w_28_0 ctx) (parallel_generatedRoot_658.w_29_0 ctx) (parallel_generatedRoot_658.w_30_0 ctx) (parallel_generatedRoot_658.w_35_0 ctx) (parallel_generatedRoot_658.w_42_0 ctx) (parallel_generatedRoot_658.w_47_0 ctx) (parallel_generatedRoot_658.w_55_0 ctx) (parallel_generatedRoot_658.w_58_0 ctx) (parallel_generatedRoot_658.w_2_0 ctx) (parallel_generatedRoot_658.w_3_0 ctx) (parallel_generatedRoot_658.w_4_0 ctx) (parallel_generatedRoot_658.w_7_0 ctx) (parallel_generatedRoot_658.w_10_0 ctx) (parallel_generatedRoot_658.w_11_0 ctx) (parallel_generatedRoot_658.w_14_0 ctx) (parallel_generatedRoot_658.w_15_0 ctx) (parallel_generatedRoot_658.w_18_0 ctx) (parallel_generatedRoot_658.w_19_0 ctx) (parallel_generatedRoot_658.w_21_0 ctx) (parallel_generatedRoot_658.w_31_0 ctx) (parallel_generatedRoot_658.w_31_1 ctx) (parallel_generatedRoot_658.w_31_2 ctx) (parallel_generatedRoot_658.w_31_3 ctx) (parallel_generatedRoot_658.w_31_4 ctx) (parallel_generatedRoot_658.w_31_5 ctx) (parallel_generatedRoot_658.w_31_6 ctx) (parallel_generatedRoot_658.w_32_0 ctx) (parallel_generatedRoot_658.w_32_1 ctx) (parallel_generatedRoot_658.w_32_2 ctx) (parallel_generatedRoot_658.w_32_3 ctx) (parallel_generatedRoot_658.w_32_4 ctx) (parallel_generatedRoot_658.w_32_5 ctx) (parallel_generatedRoot_658.w_32_6 ctx) (parallel_generatedRoot_658.w_33_0 ctx) (parallel_generatedRoot_658.w_33_1 ctx) (parallel_generatedRoot_658.w_33_2 ctx) (parallel_generatedRoot_658.w_33_3 ctx) (parallel_generatedRoot_658.w_33_4 ctx) (parallel_generatedRoot_658.w_33_5 ctx) (parallel_generatedRoot_658.w_33_6 ctx) (parallel_generatedRoot_658.w_34_0 ctx) (parallel_generatedRoot_658.w_36_0 ctx) (parallel_generatedRoot_658.w_38_0 ctx) (parallel_generatedRoot_658.w_39_0 ctx) (parallel_generatedRoot_658.w_40_0 ctx) (parallel_generatedRoot_658.w_43_0 ctx) (parallel_generatedRoot_658.w_45_0 ctx) (parallel_generatedRoot_658.w_48_0 ctx) (parallel_generatedRoot_658.w_51_0 ctx) (parallel_generatedRoot_658.w_56_0 ctx) (parallel_generatedRoot_658.w_59_0 ctx) (parallel_generatedRoot_658.w_65_0 ctx) (parallel_generatedRoot_658.w_68_0 ctx) (parallel_generatedRoot_658.w_75_0 ctx)) :
    parallel_generatedRoot_658.Facts_0 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_658.Facts_1  (ctx : parallel_generatedRoot_658Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  h78_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_658.w_33_5 ctx) (0) (parallel_generatedRoot_658.w_78_0 ctx)
  hout : outputs = parallel_generatedRoot_658.outs ctx

theorem parallel_generatedRoot_658.facts_1  {ctx : parallel_generatedRoot_658Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_658.constraints_1 (parallel_generatedRoot_658.w_33_5 ctx) (parallel_generatedRoot_658.w_54_0 ctx) (parallel_generatedRoot_658.w_64_0 ctx) (parallel_generatedRoot_658.w_71_0 ctx) (parallel_generatedRoot_658.w_74_0 ctx) (parallel_generatedRoot_658.w_78_0 ctx) (parallel_generatedRoot_658.w_81_0 ctx) (parallel_generatedRoot_658.w_84_0 ctx) outputs) :
    parallel_generatedRoot_658.Facts_1 ctx outputs :=
  ⟨h.1, h.2⟩

structure parallel_generatedRoot_658.Facts  (ctx : parallel_generatedRoot_658Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  c0 : parallel_generatedRoot_658.Facts_0 ctx
  c1 : parallel_generatedRoot_658.Facts_1 ctx outputs

theorem parallel_generatedRoot_658.facts  (ctx : parallel_generatedRoot_658Ctx) {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_658.body backend hashModel tape path params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_658.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_658.facts_0 (ctx := ctx) (outputs := outputs) t0.1
  have t1 := t0.2
  have r1 := parallel_generatedRoot_658.facts_1 (ctx := ctx) (outputs := outputs) t1
  ⟨r0, r1⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_890.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) where
  inputs : (Fin 14400 → mxxMat0) × (Fin 14400 → mxxMat0) × (Fin 14400 → mxxMat0) × (Fin 14400 → mxxMat0) × (Fin 14400 → mxxMat0) × (Fin 14400 → mxxMat0) × Unit
  witness : parallel_generatedRoot_890.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_890Ctx" => (parallel_generatedRoot_890.Ctx backend hashModel tape path params i_0)

def parallel_generatedRoot_890.w_0_0  (ctx : parallel_generatedRoot_890Ctx) : Fin 14400 → mxxMat0 :=
  ctx.inputs.1

def parallel_generatedRoot_890.w_1_0  (ctx : parallel_generatedRoot_890Ctx) : Fin 14400 → mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_890.w_2_0  (ctx : parallel_generatedRoot_890Ctx) : Fin 14400 → mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_890.w_3_0  (ctx : parallel_generatedRoot_890Ctx) : Fin 14400 → mxxMat0 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_890.w_4_0  (ctx : parallel_generatedRoot_890Ctx) : Fin 14400 → mxxMat0 :=
  ctx.inputs.2.2.2.2.1

def parallel_generatedRoot_890.w_5_0  (ctx : parallel_generatedRoot_890Ctx) : Fin 14400 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.1

def parallel_generatedRoot_890.w_6_0  (ctx : parallel_generatedRoot_890Ctx) : Fin 2 → mxxMat0 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_890.w_6_1  (ctx : parallel_generatedRoot_890Ctx) : Fin 2 → mxxMat0 :=
  ctx.witness.w_6_1

def parallel_generatedRoot_890.w_6_2  (ctx : parallel_generatedRoot_890Ctx) : Fin 2 → mxxMat0 :=
  ctx.witness.w_6_2

def parallel_generatedRoot_890.w_6_3  (ctx : parallel_generatedRoot_890Ctx) : Fin 2 → mxxMat0 :=
  ctx.witness.w_6_3

def parallel_generatedRoot_890.w_6_4  (ctx : parallel_generatedRoot_890Ctx) : Fin 2 → mxxMat0 :=
  ctx.witness.w_6_4

def parallel_generatedRoot_890.w_6_5  (ctx : parallel_generatedRoot_890Ctx) : Fin 2 → mxxMat0 :=
  ctx.witness.w_6_5

def parallel_generatedRoot_890.w_7_0  (ctx : parallel_generatedRoot_890Ctx) : mxxMat0 :=
  ctx.witness.w_7_0

def parallel_generatedRoot_890.w_8_0  (ctx : parallel_generatedRoot_890Ctx) : mxxMat0 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_890.w_9_0  (ctx : parallel_generatedRoot_890Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_890.w_7_0 ctx) (parallel_generatedRoot_890.w_8_0 ctx)

def parallel_generatedRoot_890.w_10_0  (ctx : parallel_generatedRoot_890Ctx) : mxxMat0 :=
  ctx.witness.w_10_0

def parallel_generatedRoot_890.w_11_0  (ctx : parallel_generatedRoot_890Ctx) : mxxMat0 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_890.w_12_0  (ctx : parallel_generatedRoot_890Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_890.w_10_0 ctx) (parallel_generatedRoot_890.w_11_0 ctx)

def parallel_generatedRoot_890.w_13_0  (ctx : parallel_generatedRoot_890Ctx) : mxxMat0 :=
  ctx.witness.w_13_0

def parallel_generatedRoot_890.w_14_0  (ctx : parallel_generatedRoot_890Ctx) : mxxMat0 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_890.w_15_0  (ctx : parallel_generatedRoot_890Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_890.w_13_0 ctx) (parallel_generatedRoot_890.w_14_0 ctx)

def parallel_generatedRoot_890.w_16_0  (ctx : parallel_generatedRoot_890Ctx) : mxxMat0 :=
  ctx.witness.w_16_0

def parallel_generatedRoot_890.w_17_0  (ctx : parallel_generatedRoot_890Ctx) : mxxMat0 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_890.w_18_0  (ctx : parallel_generatedRoot_890Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_890.w_16_0 ctx) (parallel_generatedRoot_890.w_17_0 ctx)

def parallel_generatedRoot_890.w_19_0  (ctx : parallel_generatedRoot_890Ctx) : mxxMat0 :=
  ctx.witness.w_19_0

def parallel_generatedRoot_890.w_20_0  (ctx : parallel_generatedRoot_890Ctx) : mxxMat0 :=
  ctx.witness.w_20_0

def parallel_generatedRoot_890.w_21_0  (ctx : parallel_generatedRoot_890Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_890.w_19_0 ctx) (parallel_generatedRoot_890.w_20_0 ctx)

def parallel_generatedRoot_890.w_22_0  (ctx : parallel_generatedRoot_890Ctx) : mxxMat0 :=
  ctx.witness.w_22_0

def parallel_generatedRoot_890.w_23_0  (ctx : parallel_generatedRoot_890Ctx) : mxxMat0 :=
  ctx.witness.w_23_0

def parallel_generatedRoot_890.w_24_0  (ctx : parallel_generatedRoot_890Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_890.w_22_0 ctx) (parallel_generatedRoot_890.w_23_0 ctx)

def parallel_generatedRoot_890.w_25_0  (ctx : parallel_generatedRoot_890Ctx) : mxxMat0 :=
  ctx.witness.w_25_0

def parallel_generatedRoot_890.w_25_1  (ctx : parallel_generatedRoot_890Ctx) : mxxMat0 :=
  ctx.witness.w_25_1

def parallel_generatedRoot_890.w_25_2  (ctx : parallel_generatedRoot_890Ctx) : mxxMat0 :=
  ctx.witness.w_25_2

def parallel_generatedRoot_890.w_25_3  (ctx : parallel_generatedRoot_890Ctx) : mxxMat0 :=
  ctx.witness.w_25_3

def parallel_generatedRoot_890.w_25_4  (ctx : parallel_generatedRoot_890Ctx) : mxxMat0 :=
  ctx.witness.w_25_4

def parallel_generatedRoot_890.w_25_5  (ctx : parallel_generatedRoot_890Ctx) : mxxMat0 :=
  ctx.witness.w_25_5

def parallel_generatedRoot_890.outs  (ctx : parallel_generatedRoot_890Ctx) : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit :=
  ((parallel_generatedRoot_890.w_25_0 ctx), ((parallel_generatedRoot_890.w_25_1 ctx), ((parallel_generatedRoot_890.w_25_2 ctx), ((parallel_generatedRoot_890.w_25_3 ctx), ((parallel_generatedRoot_890.w_25_4 ctx), ((parallel_generatedRoot_890.w_25_5 ctx), ()))))))

structure parallel_generatedRoot_890.Facts_0  (ctx : parallel_generatedRoot_890Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  t0 : (2) = 2
  h6_5 : (∀ i : Fin 2, parallel_parallel_generatedRoot_890_6 backend hashModel tape (path ++ [6, i.val]) params i_0 i ((parallel_generatedRoot_890.w_0_0 ctx), ((parallel_generatedRoot_890.w_1_0 ctx), ((parallel_generatedRoot_890.w_2_0 ctx), ((parallel_generatedRoot_890.w_3_0 ctx), ((parallel_generatedRoot_890.w_4_0 ctx), ((parallel_generatedRoot_890.w_5_0 ctx), ())))))) (((parallel_generatedRoot_890.w_6_0 ctx) i), (((parallel_generatedRoot_890.w_6_1 ctx) i), (((parallel_generatedRoot_890.w_6_2 ctx) i), (((parallel_generatedRoot_890.w_6_3 ctx) i), (((parallel_generatedRoot_890.w_6_4 ctx) i), (((parallel_generatedRoot_890.w_6_5 ctx) i), ())))))))
  h7_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_890.w_6_0 ctx) (0) (parallel_generatedRoot_890.w_7_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_890.w_6_0 ctx) (1) (parallel_generatedRoot_890.w_8_0 ctx)
  h10_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_890.w_6_1 ctx) (0) (parallel_generatedRoot_890.w_10_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_890.w_6_1 ctx) (1) (parallel_generatedRoot_890.w_11_0 ctx)
  h13_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_890.w_6_2 ctx) (0) (parallel_generatedRoot_890.w_13_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_890.w_6_2 ctx) (1) (parallel_generatedRoot_890.w_14_0 ctx)
  h16_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_890.w_6_3 ctx) (0) (parallel_generatedRoot_890.w_16_0 ctx)
  h17_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_890.w_6_3 ctx) (1) (parallel_generatedRoot_890.w_17_0 ctx)
  h19_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_890.w_6_4 ctx) (0) (parallel_generatedRoot_890.w_19_0 ctx)
  h20_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_890.w_6_4 ctx) (1) (parallel_generatedRoot_890.w_20_0 ctx)
  h22_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_890.w_6_5 ctx) (0) (parallel_generatedRoot_890.w_22_0 ctx)
  h23_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_890.w_6_5 ctx) (1) (parallel_generatedRoot_890.w_23_0 ctx)
  t14 : 0 ≤ (0)
  h25_5 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_890_25 backend hashModel tape (path ++ [25, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, ((parallel_generatedRoot_890.w_6_0 ctx), ((parallel_generatedRoot_890.w_6_1 ctx), ((parallel_generatedRoot_890.w_6_2 ctx), ((parallel_generatedRoot_890.w_6_3 ctx), ((parallel_generatedRoot_890.w_6_4 ctx), ((parallel_generatedRoot_890.w_6_5 ctx), ())))))))))))) next) (Int.toNat (0)) ((parallel_generatedRoot_890.w_9_0 ctx), ((parallel_generatedRoot_890.w_12_0 ctx), ((parallel_generatedRoot_890.w_15_0 ctx), ((parallel_generatedRoot_890.w_18_0 ctx), ((parallel_generatedRoot_890.w_21_0 ctx), ((parallel_generatedRoot_890.w_24_0 ctx), ())))))) ((parallel_generatedRoot_890.w_25_0 ctx), ((parallel_generatedRoot_890.w_25_1 ctx), ((parallel_generatedRoot_890.w_25_2 ctx), ((parallel_generatedRoot_890.w_25_3 ctx), ((parallel_generatedRoot_890.w_25_4 ctx), ((parallel_generatedRoot_890.w_25_5 ctx), ()))))))
  hout : outputs = parallel_generatedRoot_890.outs ctx

theorem parallel_generatedRoot_890.facts_0  {ctx : parallel_generatedRoot_890Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_890.constraints_0 backend hashModel tape path params i_0 (parallel_generatedRoot_890.w_0_0 ctx) (parallel_generatedRoot_890.w_1_0 ctx) (parallel_generatedRoot_890.w_2_0 ctx) (parallel_generatedRoot_890.w_3_0 ctx) (parallel_generatedRoot_890.w_4_0 ctx) (parallel_generatedRoot_890.w_5_0 ctx) (parallel_generatedRoot_890.w_6_0 ctx) (parallel_generatedRoot_890.w_6_1 ctx) (parallel_generatedRoot_890.w_6_2 ctx) (parallel_generatedRoot_890.w_6_3 ctx) (parallel_generatedRoot_890.w_6_4 ctx) (parallel_generatedRoot_890.w_6_5 ctx) (parallel_generatedRoot_890.w_7_0 ctx) (parallel_generatedRoot_890.w_8_0 ctx) (parallel_generatedRoot_890.w_9_0 ctx) (parallel_generatedRoot_890.w_10_0 ctx) (parallel_generatedRoot_890.w_11_0 ctx) (parallel_generatedRoot_890.w_12_0 ctx) (parallel_generatedRoot_890.w_13_0 ctx) (parallel_generatedRoot_890.w_14_0 ctx) (parallel_generatedRoot_890.w_15_0 ctx) (parallel_generatedRoot_890.w_16_0 ctx) (parallel_generatedRoot_890.w_17_0 ctx) (parallel_generatedRoot_890.w_18_0 ctx) (parallel_generatedRoot_890.w_19_0 ctx) (parallel_generatedRoot_890.w_20_0 ctx) (parallel_generatedRoot_890.w_21_0 ctx) (parallel_generatedRoot_890.w_22_0 ctx) (parallel_generatedRoot_890.w_23_0 ctx) (parallel_generatedRoot_890.w_24_0 ctx) (parallel_generatedRoot_890.w_25_0 ctx) (parallel_generatedRoot_890.w_25_1 ctx) (parallel_generatedRoot_890.w_25_2 ctx) (parallel_generatedRoot_890.w_25_3 ctx) (parallel_generatedRoot_890.w_25_4 ctx) (parallel_generatedRoot_890.w_25_5 ctx) outputs) :
    parallel_generatedRoot_890.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_890.Facts  (ctx : parallel_generatedRoot_890Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  c0 : parallel_generatedRoot_890.Facts_0 ctx outputs

theorem parallel_generatedRoot_890.facts  (ctx : parallel_generatedRoot_890Ctx) {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_890.body backend hashModel tape path params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_890.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_890.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_894.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) where
  inputs : (Fin 14400 → mxxMat0) × (Fin 14400 → mxxMat0) × (Fin 14400 → mxxMat0) × (Fin 14400 → mxxMat0) × (Fin 14400 → mxxMat0) × (Fin 14400 → mxxMat0) × Unit
  witness : parallel_generatedRoot_894.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_894Ctx" => (parallel_generatedRoot_894.Ctx backend hashModel tape path params i_0)

def parallel_generatedRoot_894.w_0_0  (ctx : parallel_generatedRoot_894Ctx) : Fin 14400 → mxxMat0 :=
  ctx.inputs.1

def parallel_generatedRoot_894.w_1_0  (ctx : parallel_generatedRoot_894Ctx) : Fin 14400 → mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_894.w_2_0  (ctx : parallel_generatedRoot_894Ctx) : Fin 14400 → mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_894.w_3_0  (ctx : parallel_generatedRoot_894Ctx) : Fin 14400 → mxxMat0 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_894.w_4_0  (ctx : parallel_generatedRoot_894Ctx) : Fin 14400 → mxxMat0 :=
  ctx.inputs.2.2.2.2.1

def parallel_generatedRoot_894.w_5_0  (ctx : parallel_generatedRoot_894Ctx) : Fin 14400 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.1

def parallel_generatedRoot_894.w_6_0  (ctx : parallel_generatedRoot_894Ctx) : Fin 2 → mxxMat0 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_894.w_6_1  (ctx : parallel_generatedRoot_894Ctx) : Fin 2 → mxxMat0 :=
  ctx.witness.w_6_1

def parallel_generatedRoot_894.w_6_2  (ctx : parallel_generatedRoot_894Ctx) : Fin 2 → mxxMat0 :=
  ctx.witness.w_6_2

def parallel_generatedRoot_894.w_6_3  (ctx : parallel_generatedRoot_894Ctx) : Fin 2 → mxxMat0 :=
  ctx.witness.w_6_3

def parallel_generatedRoot_894.w_6_4  (ctx : parallel_generatedRoot_894Ctx) : Fin 2 → mxxMat0 :=
  ctx.witness.w_6_4

def parallel_generatedRoot_894.w_6_5  (ctx : parallel_generatedRoot_894Ctx) : Fin 2 → mxxMat0 :=
  ctx.witness.w_6_5

def parallel_generatedRoot_894.w_7_0  (ctx : parallel_generatedRoot_894Ctx) : mxxMat0 :=
  ctx.witness.w_7_0

def parallel_generatedRoot_894.w_8_0  (ctx : parallel_generatedRoot_894Ctx) : mxxMat0 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_894.w_9_0  (ctx : parallel_generatedRoot_894Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_894.w_7_0 ctx) (parallel_generatedRoot_894.w_8_0 ctx)

def parallel_generatedRoot_894.w_10_0  (ctx : parallel_generatedRoot_894Ctx) : mxxMat0 :=
  ctx.witness.w_10_0

def parallel_generatedRoot_894.w_11_0  (ctx : parallel_generatedRoot_894Ctx) : mxxMat0 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_894.w_12_0  (ctx : parallel_generatedRoot_894Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_894.w_10_0 ctx) (parallel_generatedRoot_894.w_11_0 ctx)

def parallel_generatedRoot_894.w_13_0  (ctx : parallel_generatedRoot_894Ctx) : mxxMat0 :=
  ctx.witness.w_13_0

def parallel_generatedRoot_894.w_14_0  (ctx : parallel_generatedRoot_894Ctx) : mxxMat0 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_894.w_15_0  (ctx : parallel_generatedRoot_894Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_894.w_13_0 ctx) (parallel_generatedRoot_894.w_14_0 ctx)

def parallel_generatedRoot_894.w_16_0  (ctx : parallel_generatedRoot_894Ctx) : mxxMat0 :=
  ctx.witness.w_16_0

def parallel_generatedRoot_894.w_17_0  (ctx : parallel_generatedRoot_894Ctx) : mxxMat0 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_894.w_18_0  (ctx : parallel_generatedRoot_894Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_894.w_16_0 ctx) (parallel_generatedRoot_894.w_17_0 ctx)

def parallel_generatedRoot_894.w_19_0  (ctx : parallel_generatedRoot_894Ctx) : mxxMat0 :=
  ctx.witness.w_19_0

def parallel_generatedRoot_894.w_20_0  (ctx : parallel_generatedRoot_894Ctx) : mxxMat0 :=
  ctx.witness.w_20_0

def parallel_generatedRoot_894.w_21_0  (ctx : parallel_generatedRoot_894Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_894.w_19_0 ctx) (parallel_generatedRoot_894.w_20_0 ctx)

def parallel_generatedRoot_894.w_22_0  (ctx : parallel_generatedRoot_894Ctx) : mxxMat0 :=
  ctx.witness.w_22_0

def parallel_generatedRoot_894.w_23_0  (ctx : parallel_generatedRoot_894Ctx) : mxxMat0 :=
  ctx.witness.w_23_0

def parallel_generatedRoot_894.w_24_0  (ctx : parallel_generatedRoot_894Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_894.w_22_0 ctx) (parallel_generatedRoot_894.w_23_0 ctx)

def parallel_generatedRoot_894.w_25_0  (ctx : parallel_generatedRoot_894Ctx) : mxxMat0 :=
  ctx.witness.w_25_0

def parallel_generatedRoot_894.w_25_1  (ctx : parallel_generatedRoot_894Ctx) : mxxMat0 :=
  ctx.witness.w_25_1

def parallel_generatedRoot_894.w_25_2  (ctx : parallel_generatedRoot_894Ctx) : mxxMat0 :=
  ctx.witness.w_25_2

def parallel_generatedRoot_894.w_25_3  (ctx : parallel_generatedRoot_894Ctx) : mxxMat0 :=
  ctx.witness.w_25_3

def parallel_generatedRoot_894.w_25_4  (ctx : parallel_generatedRoot_894Ctx) : mxxMat0 :=
  ctx.witness.w_25_4

def parallel_generatedRoot_894.w_25_5  (ctx : parallel_generatedRoot_894Ctx) : mxxMat0 :=
  ctx.witness.w_25_5

def parallel_generatedRoot_894.outs  (ctx : parallel_generatedRoot_894Ctx) : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit :=
  ((parallel_generatedRoot_894.w_25_0 ctx), ((parallel_generatedRoot_894.w_25_1 ctx), ((parallel_generatedRoot_894.w_25_2 ctx), ((parallel_generatedRoot_894.w_25_3 ctx), ((parallel_generatedRoot_894.w_25_4 ctx), ((parallel_generatedRoot_894.w_25_5 ctx), ()))))))

structure parallel_generatedRoot_894.Facts_0  (ctx : parallel_generatedRoot_894Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  t0 : (2) = 2
  h6_5 : (∀ i : Fin 2, parallel_parallel_generatedRoot_894_6 backend hashModel tape (path ++ [6, i.val]) params i_0 i ((parallel_generatedRoot_894.w_0_0 ctx), ((parallel_generatedRoot_894.w_1_0 ctx), ((parallel_generatedRoot_894.w_2_0 ctx), ((parallel_generatedRoot_894.w_3_0 ctx), ((parallel_generatedRoot_894.w_4_0 ctx), ((parallel_generatedRoot_894.w_5_0 ctx), ())))))) (((parallel_generatedRoot_894.w_6_0 ctx) i), (((parallel_generatedRoot_894.w_6_1 ctx) i), (((parallel_generatedRoot_894.w_6_2 ctx) i), (((parallel_generatedRoot_894.w_6_3 ctx) i), (((parallel_generatedRoot_894.w_6_4 ctx) i), (((parallel_generatedRoot_894.w_6_5 ctx) i), ())))))))
  h7_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_894.w_6_0 ctx) (0) (parallel_generatedRoot_894.w_7_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_894.w_6_0 ctx) (1) (parallel_generatedRoot_894.w_8_0 ctx)
  h10_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_894.w_6_1 ctx) (0) (parallel_generatedRoot_894.w_10_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_894.w_6_1 ctx) (1) (parallel_generatedRoot_894.w_11_0 ctx)
  h13_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_894.w_6_2 ctx) (0) (parallel_generatedRoot_894.w_13_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_894.w_6_2 ctx) (1) (parallel_generatedRoot_894.w_14_0 ctx)
  h16_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_894.w_6_3 ctx) (0) (parallel_generatedRoot_894.w_16_0 ctx)
  h17_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_894.w_6_3 ctx) (1) (parallel_generatedRoot_894.w_17_0 ctx)
  h19_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_894.w_6_4 ctx) (0) (parallel_generatedRoot_894.w_19_0 ctx)
  h20_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_894.w_6_4 ctx) (1) (parallel_generatedRoot_894.w_20_0 ctx)
  h22_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_894.w_6_5 ctx) (0) (parallel_generatedRoot_894.w_22_0 ctx)
  h23_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_894.w_6_5 ctx) (1) (parallel_generatedRoot_894.w_23_0 ctx)
  t14 : 0 ≤ (0)
  h25_5 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_894_25 backend hashModel tape (path ++ [25, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, ((parallel_generatedRoot_894.w_6_0 ctx), ((parallel_generatedRoot_894.w_6_1 ctx), ((parallel_generatedRoot_894.w_6_2 ctx), ((parallel_generatedRoot_894.w_6_3 ctx), ((parallel_generatedRoot_894.w_6_4 ctx), ((parallel_generatedRoot_894.w_6_5 ctx), ())))))))))))) next) (Int.toNat (0)) ((parallel_generatedRoot_894.w_9_0 ctx), ((parallel_generatedRoot_894.w_12_0 ctx), ((parallel_generatedRoot_894.w_15_0 ctx), ((parallel_generatedRoot_894.w_18_0 ctx), ((parallel_generatedRoot_894.w_21_0 ctx), ((parallel_generatedRoot_894.w_24_0 ctx), ())))))) ((parallel_generatedRoot_894.w_25_0 ctx), ((parallel_generatedRoot_894.w_25_1 ctx), ((parallel_generatedRoot_894.w_25_2 ctx), ((parallel_generatedRoot_894.w_25_3 ctx), ((parallel_generatedRoot_894.w_25_4 ctx), ((parallel_generatedRoot_894.w_25_5 ctx), ()))))))
  hout : outputs = parallel_generatedRoot_894.outs ctx

theorem parallel_generatedRoot_894.facts_0  {ctx : parallel_generatedRoot_894Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_894.constraints_0 backend hashModel tape path params i_0 (parallel_generatedRoot_894.w_0_0 ctx) (parallel_generatedRoot_894.w_1_0 ctx) (parallel_generatedRoot_894.w_2_0 ctx) (parallel_generatedRoot_894.w_3_0 ctx) (parallel_generatedRoot_894.w_4_0 ctx) (parallel_generatedRoot_894.w_5_0 ctx) (parallel_generatedRoot_894.w_6_0 ctx) (parallel_generatedRoot_894.w_6_1 ctx) (parallel_generatedRoot_894.w_6_2 ctx) (parallel_generatedRoot_894.w_6_3 ctx) (parallel_generatedRoot_894.w_6_4 ctx) (parallel_generatedRoot_894.w_6_5 ctx) (parallel_generatedRoot_894.w_7_0 ctx) (parallel_generatedRoot_894.w_8_0 ctx) (parallel_generatedRoot_894.w_9_0 ctx) (parallel_generatedRoot_894.w_10_0 ctx) (parallel_generatedRoot_894.w_11_0 ctx) (parallel_generatedRoot_894.w_12_0 ctx) (parallel_generatedRoot_894.w_13_0 ctx) (parallel_generatedRoot_894.w_14_0 ctx) (parallel_generatedRoot_894.w_15_0 ctx) (parallel_generatedRoot_894.w_16_0 ctx) (parallel_generatedRoot_894.w_17_0 ctx) (parallel_generatedRoot_894.w_18_0 ctx) (parallel_generatedRoot_894.w_19_0 ctx) (parallel_generatedRoot_894.w_20_0 ctx) (parallel_generatedRoot_894.w_21_0 ctx) (parallel_generatedRoot_894.w_22_0 ctx) (parallel_generatedRoot_894.w_23_0 ctx) (parallel_generatedRoot_894.w_24_0 ctx) (parallel_generatedRoot_894.w_25_0 ctx) (parallel_generatedRoot_894.w_25_1 ctx) (parallel_generatedRoot_894.w_25_2 ctx) (parallel_generatedRoot_894.w_25_3 ctx) (parallel_generatedRoot_894.w_25_4 ctx) (parallel_generatedRoot_894.w_25_5 ctx) outputs) :
    parallel_generatedRoot_894.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_894.Facts  (ctx : parallel_generatedRoot_894Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  c0 : parallel_generatedRoot_894.Facts_0 ctx outputs

theorem parallel_generatedRoot_894.facts  (ctx : parallel_generatedRoot_894Ctx) {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_894.body backend hashModel tape path params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_894.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_894.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_895.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) where
  inputs : (parallel_generatedRoot_895.InputType)
  witness : parallel_generatedRoot_895.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_895Ctx" => (parallel_generatedRoot_895.Ctx backend hashModel tape path params i_0)

def parallel_generatedRoot_895.w_1_0  (ctx : parallel_generatedRoot_895Ctx) : Fin 8 → mxxMat0 :=
  ctx.inputs.1

def parallel_generatedRoot_895.w_3_0  (ctx : parallel_generatedRoot_895Ctx) : Fin 8 → mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_895.w_5_0  (ctx : parallel_generatedRoot_895Ctx) : Fin 8 → mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_895.w_7_0  (ctx : parallel_generatedRoot_895Ctx) : Fin 8 → mxxMat0 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_895.w_9_0  (ctx : parallel_generatedRoot_895Ctx) : Fin 8 → mxxMat0 :=
  ctx.inputs.2.2.2.2.1

def parallel_generatedRoot_895.w_10_0  (ctx : parallel_generatedRoot_895Ctx) : ByteArray :=
  ctx.inputs.2.2.2.2.2.1

def parallel_generatedRoot_895.w_11_0  (ctx : parallel_generatedRoot_895Ctx) : Fin 8 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.1

def parallel_generatedRoot_895.w_12_0  (ctx : parallel_generatedRoot_895Ctx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.1

def parallel_generatedRoot_895.w_13_0  (ctx : parallel_generatedRoot_895Ctx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_895.w_14_0  (ctx : parallel_generatedRoot_895Ctx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_895.w_15_0  (ctx : parallel_generatedRoot_895Ctx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_895.w_16_0  (ctx : parallel_generatedRoot_895Ctx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_895.w_17_0  (ctx : parallel_generatedRoot_895Ctx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_895.w_0_0  (ctx : parallel_generatedRoot_895Ctx) : mxxMat0 :=
  (0 : mxxMat0)

def parallel_generatedRoot_895.w_2_0  (ctx : parallel_generatedRoot_895Ctx) : mxxMat0 :=
  ctx.witness.w_2_0

def parallel_generatedRoot_895.w_4_0  (ctx : parallel_generatedRoot_895Ctx) : mxxMat0 :=
  ctx.witness.w_4_0

def parallel_generatedRoot_895.w_6_0  (ctx : parallel_generatedRoot_895Ctx) : mxxMat0 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_895.w_8_0  (ctx : parallel_generatedRoot_895Ctx) : mxxMat0 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_895.w_18_0  (ctx : parallel_generatedRoot_895Ctx) : mxxMat0 :=
  ctx.witness.w_18_0

def parallel_generatedRoot_895.w_18_1  (ctx : parallel_generatedRoot_895Ctx) : mxxMat0 :=
  ctx.witness.w_18_1

def parallel_generatedRoot_895.w_18_2  (ctx : parallel_generatedRoot_895Ctx) : mxxMat0 :=
  ctx.witness.w_18_2

def parallel_generatedRoot_895.w_18_3  (ctx : parallel_generatedRoot_895Ctx) : mxxMat0 :=
  ctx.witness.w_18_3

def parallel_generatedRoot_895.w_18_4  (ctx : parallel_generatedRoot_895Ctx) : mxxMat0 :=
  ctx.witness.w_18_4

def parallel_generatedRoot_895.w_18_5  (ctx : parallel_generatedRoot_895Ctx) : mxxMat0 :=
  ctx.witness.w_18_5

def parallel_generatedRoot_895.w_18_6  (ctx : parallel_generatedRoot_895Ctx) : mxxMat0 :=
  ctx.witness.w_18_6

def parallel_generatedRoot_895.w_18_7  (ctx : parallel_generatedRoot_895Ctx) : mxxMat0 :=
  ctx.witness.w_18_7

def parallel_generatedRoot_895.w_18_8  (ctx : parallel_generatedRoot_895Ctx) : mxxMat0 :=
  ctx.witness.w_18_8

def parallel_generatedRoot_895.w_18_9  (ctx : parallel_generatedRoot_895Ctx) : mxxMat0 :=
  ctx.witness.w_18_9

def parallel_generatedRoot_895.w_19_0  (ctx : parallel_generatedRoot_895Ctx) : mxxMat0 :=
  ctx.witness.w_19_0

def parallel_generatedRoot_895.w_20_0  (ctx : parallel_generatedRoot_895Ctx) : Int :=
  (MxxIR.selectInt (7) [(1), (3), (5), (7), (9), (11), (13), (15)])

def parallel_generatedRoot_895.w_21_0  (ctx : parallel_generatedRoot_895Ctx) : mxxMat0 :=
  ctx.witness.w_21_0

def parallel_generatedRoot_895.w_22_0  (ctx : parallel_generatedRoot_895Ctx) : mxxMat0 :=
  ctx.witness.w_22_0

def parallel_generatedRoot_895.w_23_0  (ctx : parallel_generatedRoot_895Ctx) : mxxMat1 :=
  ctx.witness.w_23_0

def parallel_generatedRoot_895.w_23_concat_1  (ctx : parallel_generatedRoot_895Ctx) : mxxMat2 :=
  ctx.witness.w_23_concat_1

def parallel_generatedRoot_895.w_24_0  (ctx : parallel_generatedRoot_895Ctx) : mxxMat0 :=
  ctx.witness.w_24_0

def parallel_generatedRoot_895.w_25_0  (ctx : parallel_generatedRoot_895Ctx) : mxxMat0 :=
  ctx.witness.w_25_0

def parallel_generatedRoot_895.w_26_0  (ctx : parallel_generatedRoot_895Ctx) : mxxMat0 :=
  ctx.witness.w_26_0

def parallel_generatedRoot_895.w_27_0  (ctx : parallel_generatedRoot_895Ctx) : mxxMat1 :=
  ctx.witness.w_27_0

def parallel_generatedRoot_895.w_27_concat_1  (ctx : parallel_generatedRoot_895Ctx) : mxxMat2 :=
  ctx.witness.w_27_concat_1

def parallel_generatedRoot_895.w_28_0  (ctx : parallel_generatedRoot_895Ctx) : mxxMat3 :=
  ctx.witness.w_28_0

def parallel_generatedRoot_895.w_29_0  (ctx : parallel_generatedRoot_895Ctx) : mxxMat0 :=
  ctx.witness.w_29_0

def parallel_generatedRoot_895.w_30_0  (ctx : parallel_generatedRoot_895Ctx) : mxxMat0 :=
  ctx.witness.w_30_0

def parallel_generatedRoot_895.w_31_0  (ctx : parallel_generatedRoot_895Ctx) : mxxMat0 :=
  ctx.witness.w_31_0

def parallel_generatedRoot_895.w_32_0  (ctx : parallel_generatedRoot_895Ctx) : mxxMat1 :=
  ctx.witness.w_32_0

def parallel_generatedRoot_895.w_32_concat_1  (ctx : parallel_generatedRoot_895Ctx) : mxxMat2 :=
  ctx.witness.w_32_concat_1

def parallel_generatedRoot_895.w_33_0  (ctx : parallel_generatedRoot_895Ctx) : mxxMat0 :=
  ctx.witness.w_33_0

def parallel_generatedRoot_895.w_34_0  (ctx : parallel_generatedRoot_895Ctx) : mxxMat0 :=
  ctx.witness.w_34_0

def parallel_generatedRoot_895.w_35_0  (ctx : parallel_generatedRoot_895Ctx) : mxxMat0 :=
  ctx.witness.w_35_0

def parallel_generatedRoot_895.w_36_0  (ctx : parallel_generatedRoot_895Ctx) : mxxMat1 :=
  ctx.witness.w_36_0

def parallel_generatedRoot_895.w_36_concat_1  (ctx : parallel_generatedRoot_895Ctx) : mxxMat2 :=
  ctx.witness.w_36_concat_1

def parallel_generatedRoot_895.w_37_0  (ctx : parallel_generatedRoot_895Ctx) : mxxMat3 :=
  ctx.witness.w_37_0

def parallel_generatedRoot_895.w_38_0  (ctx : parallel_generatedRoot_895Ctx) : mxxMat2 :=
  ctx.witness.w_38_0

def parallel_generatedRoot_895.w_39_0  (ctx : parallel_generatedRoot_895Ctx) : mxxMat4 :=
  ctx.witness.w_39_0

def parallel_generatedRoot_895.w_40_0  (ctx : parallel_generatedRoot_895Ctx) : mxxMat4 :=
  ctx.witness.w_40_0

def parallel_generatedRoot_895.w_41_0  (ctx : parallel_generatedRoot_895Ctx) : mxxMat1 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_895.w_37_0 ctx) (parallel_generatedRoot_895.w_40_0 ctx)

def parallel_generatedRoot_895.w_42_0  (ctx : parallel_generatedRoot_895Ctx) : mxxMat0 :=
  ctx.witness.w_42_0

def parallel_generatedRoot_895.w_43_0  (ctx : parallel_generatedRoot_895Ctx) : mxxMat0 :=
  ctx.witness.w_43_0

def parallel_generatedRoot_895.w_44_0  (ctx : parallel_generatedRoot_895Ctx) : mxxMat2 :=
  ctx.witness.w_44_0

def parallel_generatedRoot_895.w_45_0  (ctx : parallel_generatedRoot_895Ctx) : mxxMat4 :=
  ctx.witness.w_45_0

def parallel_generatedRoot_895.w_46_0  (ctx : parallel_generatedRoot_895Ctx) : mxxMat4 :=
  ctx.witness.w_46_0

def parallel_generatedRoot_895.w_47_0  (ctx : parallel_generatedRoot_895Ctx) : mxxMat1 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_895.w_28_0 ctx) (parallel_generatedRoot_895.w_46_0 ctx)

def parallel_generatedRoot_895.w_48_0  (ctx : parallel_generatedRoot_895Ctx) : mxxMat0 :=
  ctx.witness.w_48_0

def parallel_generatedRoot_895.w_49_0  (ctx : parallel_generatedRoot_895Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_895.w_18_2 ctx) (parallel_generatedRoot_895.w_48_0 ctx)

def parallel_generatedRoot_895.w_50_0  (ctx : parallel_generatedRoot_895Ctx) : mxxMat2 :=
  ctx.witness.w_50_0

def parallel_generatedRoot_895.w_51_0  (ctx : parallel_generatedRoot_895Ctx) : mxxMat4 :=
  ctx.witness.w_51_0

def parallel_generatedRoot_895.w_52_0  (ctx : parallel_generatedRoot_895Ctx) : mxxMat4 :=
  ctx.witness.w_52_0

def parallel_generatedRoot_895.w_53_0  (ctx : parallel_generatedRoot_895Ctx) : mxxMat1 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_895.w_37_0 ctx) (parallel_generatedRoot_895.w_52_0 ctx)

def parallel_generatedRoot_895.w_54_0  (ctx : parallel_generatedRoot_895Ctx) : mxxMat0 :=
  ctx.witness.w_54_0

def parallel_generatedRoot_895.w_55_0  (ctx : parallel_generatedRoot_895Ctx) : mxxMat0 :=
  ctx.witness.w_55_0

def parallel_generatedRoot_895.w_56_0  (ctx : parallel_generatedRoot_895Ctx) : mxxMat2 :=
  ctx.witness.w_56_0

def parallel_generatedRoot_895.w_57_0  (ctx : parallel_generatedRoot_895Ctx) : mxxMat4 :=
  ctx.witness.w_57_0

def parallel_generatedRoot_895.w_58_0  (ctx : parallel_generatedRoot_895Ctx) : mxxMat4 :=
  ctx.witness.w_58_0

def parallel_generatedRoot_895.w_59_0  (ctx : parallel_generatedRoot_895Ctx) : mxxMat1 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_895.w_28_0 ctx) (parallel_generatedRoot_895.w_58_0 ctx)

def parallel_generatedRoot_895.w_60_0  (ctx : parallel_generatedRoot_895Ctx) : mxxMat0 :=
  ctx.witness.w_60_0

def parallel_generatedRoot_895.w_61_0  (ctx : parallel_generatedRoot_895Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_895.w_18_3 ctx) (parallel_generatedRoot_895.w_60_0 ctx)

def parallel_generatedRoot_895.w_62_0  (ctx : parallel_generatedRoot_895Ctx) : mxxMat0 :=
  ctx.witness.w_62_0

def parallel_generatedRoot_895.w_63_0  (ctx : parallel_generatedRoot_895Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_895.w_18_4 ctx) (parallel_generatedRoot_895.w_62_0 ctx)

def parallel_generatedRoot_895.w_64_0  (ctx : parallel_generatedRoot_895Ctx) : mxxMat0 :=
  ctx.witness.w_64_0

def parallel_generatedRoot_895.w_65_0  (ctx : parallel_generatedRoot_895Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_895.w_18_5 ctx) (parallel_generatedRoot_895.w_64_0 ctx)

def parallel_generatedRoot_895.w_66_0  (ctx : parallel_generatedRoot_895Ctx) : mxxMat0 :=
  ctx.witness.w_66_0

def parallel_generatedRoot_895.w_67_0  (ctx : parallel_generatedRoot_895Ctx) : mxxMat0 :=
  ctx.witness.w_67_0

def parallel_generatedRoot_895.w_68_0  (ctx : parallel_generatedRoot_895Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_895.w_66_0 ctx) (parallel_generatedRoot_895.w_67_0 ctx)

def parallel_generatedRoot_895.w_69_0  (ctx : parallel_generatedRoot_895Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_895.w_18_0 ctx) (parallel_generatedRoot_895.w_68_0 ctx)

def parallel_generatedRoot_895.w_70_0  (ctx : parallel_generatedRoot_895Ctx) : mxxMat0 :=
  ctx.witness.w_70_0

def parallel_generatedRoot_895.w_71_0  (ctx : parallel_generatedRoot_895Ctx) : mxxMat0 :=
  ctx.witness.w_71_0

def parallel_generatedRoot_895.w_72_0  (ctx : parallel_generatedRoot_895Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_895.w_70_0 ctx) (parallel_generatedRoot_895.w_71_0 ctx)

def parallel_generatedRoot_895.w_73_0  (ctx : parallel_generatedRoot_895Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_895.w_18_1 ctx) (parallel_generatedRoot_895.w_72_0 ctx)

def parallel_generatedRoot_895.outs  (ctx : parallel_generatedRoot_895Ctx) : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit :=
  ((parallel_generatedRoot_895.w_49_0 ctx), ((parallel_generatedRoot_895.w_61_0 ctx), ((parallel_generatedRoot_895.w_63_0 ctx), ((parallel_generatedRoot_895.w_65_0 ctx), ((parallel_generatedRoot_895.w_69_0 ctx), ((parallel_generatedRoot_895.w_73_0 ctx), ()))))))

structure parallel_generatedRoot_895.Facts_0  (ctx : parallel_generatedRoot_895Ctx) : Prop where
  h2_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_895.w_1_0 ctx) (0) (parallel_generatedRoot_895.w_2_0 ctx)
  h4_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_895.w_3_0 ctx) (0) (parallel_generatedRoot_895.w_4_0 ctx)
  h6_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_895.w_5_0 ctx) (0) (parallel_generatedRoot_895.w_6_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_895.w_7_0 ctx) (0) (parallel_generatedRoot_895.w_8_0 ctx)
  t4 : 0 ≤ (7)
  h18_9 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_895_18 backend hashModel tape (path ++ [18, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (current.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.2.1, ((parallel_generatedRoot_895.w_9_0 ctx), ((parallel_generatedRoot_895.w_10_0 ctx), ((parallel_generatedRoot_895.w_11_0 ctx), ((parallel_generatedRoot_895.w_12_0 ctx), ((parallel_generatedRoot_895.w_13_0 ctx), ((parallel_generatedRoot_895.w_14_0 ctx), ((parallel_generatedRoot_895.w_15_0 ctx), ((parallel_generatedRoot_895.w_16_0 ctx), ((parallel_generatedRoot_895.w_17_0 ctx), ((parallel_generatedRoot_895.w_1_0 ctx), ((parallel_generatedRoot_895.w_3_0 ctx), ((parallel_generatedRoot_895.w_5_0 ctx), ((parallel_generatedRoot_895.w_7_0 ctx), ()))))))))))))))))))))))) next) (Int.toNat (7)) ((parallel_generatedRoot_895.w_0_0 ctx), ((parallel_generatedRoot_895.w_0_0 ctx), ((parallel_generatedRoot_895.w_0_0 ctx), ((parallel_generatedRoot_895.w_0_0 ctx), ((parallel_generatedRoot_895.w_0_0 ctx), ((parallel_generatedRoot_895.w_0_0 ctx), ((parallel_generatedRoot_895.w_2_0 ctx), ((parallel_generatedRoot_895.w_4_0 ctx), ((parallel_generatedRoot_895.w_6_0 ctx), ((parallel_generatedRoot_895.w_8_0 ctx), ())))))))))) ((parallel_generatedRoot_895.w_18_0 ctx), ((parallel_generatedRoot_895.w_18_1 ctx), ((parallel_generatedRoot_895.w_18_2 ctx), ((parallel_generatedRoot_895.w_18_3 ctx), ((parallel_generatedRoot_895.w_18_4 ctx), ((parallel_generatedRoot_895.w_18_5 ctx), ((parallel_generatedRoot_895.w_18_6 ctx), ((parallel_generatedRoot_895.w_18_7 ctx), ((parallel_generatedRoot_895.w_18_8 ctx), ((parallel_generatedRoot_895.w_18_9 ctx), ()))))))))))
  h19_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_895.w_9_0 ctx) (7) (parallel_generatedRoot_895.w_19_0 ctx)
  t7 : 0 ≤ (7)
  t8 : (7) < 8
  h21_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (parallel_generatedRoot_895.w_20_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((parallel_generatedRoot_895.w_10_0 ctx)) (parallel_generatedRoot_895.w_21_0 ctx)
  h22_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (parallel_generatedRoot_895.w_20_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) ((parallel_generatedRoot_895.w_10_0 ctx)) (parallel_generatedRoot_895.w_22_0 ctx)
  h21_0_1 : MxxRuntime.concatRows (parallel_generatedRoot_895.w_19_0 ctx) (parallel_generatedRoot_895.w_21_0 ctx) (parallel_generatedRoot_895.w_23_concat_1 ctx)
  h23_0 : MxxRuntime.concatRows (parallel_generatedRoot_895.w_23_concat_1 ctx) (parallel_generatedRoot_895.w_22_0 ctx) (parallel_generatedRoot_895.w_23_0 ctx)
  h24_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_895.w_11_0 ctx) (7) (parallel_generatedRoot_895.w_24_0 ctx)
  h25_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (parallel_generatedRoot_895.w_20_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((parallel_generatedRoot_895.w_10_0 ctx)) (parallel_generatedRoot_895.w_25_0 ctx)
  h26_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (parallel_generatedRoot_895.w_20_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) ((parallel_generatedRoot_895.w_10_0 ctx)) (parallel_generatedRoot_895.w_26_0 ctx)
  h25_0_1 : MxxRuntime.concatRows (parallel_generatedRoot_895.w_24_0 ctx) (parallel_generatedRoot_895.w_25_0 ctx) (parallel_generatedRoot_895.w_27_concat_1 ctx)
  h27_0 : MxxRuntime.concatRows (parallel_generatedRoot_895.w_27_concat_1 ctx) (parallel_generatedRoot_895.w_26_0 ctx) (parallel_generatedRoot_895.w_27_0 ctx)
  h28_0 : MxxRuntime.concatColumns (parallel_generatedRoot_895.w_23_0 ctx) (parallel_generatedRoot_895.w_27_0 ctx) (parallel_generatedRoot_895.w_28_0 ctx)
  t19 : 0 ≤ (7)
  t20 : (7) < 8
  h29_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (7) [(1), (3), (5), (7), (9), (11), (13), (15)])) (parallel_generatedRoot_895.w_12_0 ctx) (parallel_generatedRoot_895.w_29_0 ctx)
  t22 : 0 ≤ (7)
  t23 : (7) < 8
  h30_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (7) [(1), (3), (5), (7), (9), (11), (13), (15)])) (parallel_generatedRoot_895.w_13_0 ctx) (parallel_generatedRoot_895.w_30_0 ctx)
  t25 : 0 ≤ (7)
  t26 : (7) < 8
  h31_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (7) [(1), (3), (5), (7), (9), (11), (13), (15)])) (parallel_generatedRoot_895.w_14_0 ctx) (parallel_generatedRoot_895.w_31_0 ctx)
  h30_0_1 : MxxRuntime.concatRows (parallel_generatedRoot_895.w_29_0 ctx) (parallel_generatedRoot_895.w_30_0 ctx) (parallel_generatedRoot_895.w_32_concat_1 ctx)
  h32_0 : MxxRuntime.concatRows (parallel_generatedRoot_895.w_32_concat_1 ctx) (parallel_generatedRoot_895.w_31_0 ctx) (parallel_generatedRoot_895.w_32_0 ctx)
  t30 : 0 ≤ (7)
  t31 : (7) < 8

theorem parallel_generatedRoot_895.facts_0  {ctx : parallel_generatedRoot_895Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_895.constraints_0 backend hashModel tape path params i_0 (parallel_generatedRoot_895.w_1_0 ctx) (parallel_generatedRoot_895.w_3_0 ctx) (parallel_generatedRoot_895.w_5_0 ctx) (parallel_generatedRoot_895.w_7_0 ctx) (parallel_generatedRoot_895.w_9_0 ctx) (parallel_generatedRoot_895.w_10_0 ctx) (parallel_generatedRoot_895.w_11_0 ctx) (parallel_generatedRoot_895.w_12_0 ctx) (parallel_generatedRoot_895.w_13_0 ctx) (parallel_generatedRoot_895.w_14_0 ctx) (parallel_generatedRoot_895.w_15_0 ctx) (parallel_generatedRoot_895.w_16_0 ctx) (parallel_generatedRoot_895.w_17_0 ctx) (parallel_generatedRoot_895.w_0_0 ctx) (parallel_generatedRoot_895.w_2_0 ctx) (parallel_generatedRoot_895.w_4_0 ctx) (parallel_generatedRoot_895.w_6_0 ctx) (parallel_generatedRoot_895.w_8_0 ctx) (parallel_generatedRoot_895.w_18_0 ctx) (parallel_generatedRoot_895.w_18_1 ctx) (parallel_generatedRoot_895.w_18_2 ctx) (parallel_generatedRoot_895.w_18_3 ctx) (parallel_generatedRoot_895.w_18_4 ctx) (parallel_generatedRoot_895.w_18_5 ctx) (parallel_generatedRoot_895.w_18_6 ctx) (parallel_generatedRoot_895.w_18_7 ctx) (parallel_generatedRoot_895.w_18_8 ctx) (parallel_generatedRoot_895.w_18_9 ctx) (parallel_generatedRoot_895.w_19_0 ctx) (parallel_generatedRoot_895.w_20_0 ctx) (parallel_generatedRoot_895.w_21_0 ctx) (parallel_generatedRoot_895.w_22_0 ctx) (parallel_generatedRoot_895.w_23_0 ctx) (parallel_generatedRoot_895.w_23_concat_1 ctx) (parallel_generatedRoot_895.w_24_0 ctx) (parallel_generatedRoot_895.w_25_0 ctx) (parallel_generatedRoot_895.w_26_0 ctx) (parallel_generatedRoot_895.w_27_0 ctx) (parallel_generatedRoot_895.w_27_concat_1 ctx) (parallel_generatedRoot_895.w_28_0 ctx) (parallel_generatedRoot_895.w_29_0 ctx) (parallel_generatedRoot_895.w_30_0 ctx) (parallel_generatedRoot_895.w_31_0 ctx) (parallel_generatedRoot_895.w_32_0 ctx) (parallel_generatedRoot_895.w_32_concat_1 ctx)) :
    parallel_generatedRoot_895.Facts_0 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_895.Facts_1  (ctx : parallel_generatedRoot_895Ctx) : Prop where
  h33_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (7) [(1), (3), (5), (7), (9), (11), (13), (15)])) (parallel_generatedRoot_895.w_15_0 ctx) (parallel_generatedRoot_895.w_33_0 ctx)
  t33 : 0 ≤ (7)
  t34 : (7) < 8
  h34_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (7) [(1), (3), (5), (7), (9), (11), (13), (15)])) (parallel_generatedRoot_895.w_16_0 ctx) (parallel_generatedRoot_895.w_34_0 ctx)
  t36 : 0 ≤ (7)
  t37 : (7) < 8
  h35_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (7) [(1), (3), (5), (7), (9), (11), (13), (15)])) (parallel_generatedRoot_895.w_17_0 ctx) (parallel_generatedRoot_895.w_35_0 ctx)
  h34_0_1 : MxxRuntime.concatRows (parallel_generatedRoot_895.w_33_0 ctx) (parallel_generatedRoot_895.w_34_0 ctx) (parallel_generatedRoot_895.w_36_concat_1 ctx)
  h36_0 : MxxRuntime.concatRows (parallel_generatedRoot_895.w_36_concat_1 ctx) (parallel_generatedRoot_895.w_35_0 ctx) (parallel_generatedRoot_895.w_36_0 ctx)
  h37_0 : MxxRuntime.concatColumns (parallel_generatedRoot_895.w_32_0 ctx) (parallel_generatedRoot_895.w_36_0 ctx) (parallel_generatedRoot_895.w_37_0 ctx)
  h38_0 : MxxRuntime.concatRows (parallel_generatedRoot_895.w_18_6 ctx) (parallel_generatedRoot_895.w_18_8 ctx) (parallel_generatedRoot_895.w_38_0 ctx)
  h39_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (parallel_generatedRoot_895.w_38_0 ctx) (parallel_generatedRoot_895.w_39_0 ctx)
  h40_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (parallel_generatedRoot_895.w_39_0 ctx) (parallel_generatedRoot_895.w_40_0 ctx)
  t45 : 0 ≤ 1
  t46 : 1 < 2
  t47 : 2 ≤ 3
  t48 : 0 ≤ 0
  t49 : 0 < 25
  t50 : 25 ≤ 25
  h42_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_895.w_41_0 ctx) 1 2 0 25 (parallel_generatedRoot_895.w_42_0 ctx)
  t52 : 0 ≤ 2
  t53 : 2 < 3
  t54 : 3 ≤ 3
  t55 : 0 ≤ 0
  t56 : 0 < 25
  t57 : 25 ≤ 25
  h43_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_895.w_41_0 ctx) 2 3 0 25 (parallel_generatedRoot_895.w_43_0 ctx)
  h44_0 : MxxRuntime.concatRows (parallel_generatedRoot_895.w_42_0 ctx) (parallel_generatedRoot_895.w_43_0 ctx) (parallel_generatedRoot_895.w_44_0 ctx)
  h45_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (parallel_generatedRoot_895.w_44_0 ctx) (parallel_generatedRoot_895.w_45_0 ctx)
  h46_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (parallel_generatedRoot_895.w_45_0 ctx) (parallel_generatedRoot_895.w_46_0 ctx)
  t62 : 0 ≤ 1
  t63 : 1 < 2

theorem parallel_generatedRoot_895.facts_1  {ctx : parallel_generatedRoot_895Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_895.constraints_1 backend (parallel_generatedRoot_895.w_15_0 ctx) (parallel_generatedRoot_895.w_16_0 ctx) (parallel_generatedRoot_895.w_17_0 ctx) (parallel_generatedRoot_895.w_18_6 ctx) (parallel_generatedRoot_895.w_18_8 ctx) (parallel_generatedRoot_895.w_32_0 ctx) (parallel_generatedRoot_895.w_33_0 ctx) (parallel_generatedRoot_895.w_34_0 ctx) (parallel_generatedRoot_895.w_35_0 ctx) (parallel_generatedRoot_895.w_36_0 ctx) (parallel_generatedRoot_895.w_36_concat_1 ctx) (parallel_generatedRoot_895.w_37_0 ctx) (parallel_generatedRoot_895.w_38_0 ctx) (parallel_generatedRoot_895.w_39_0 ctx) (parallel_generatedRoot_895.w_40_0 ctx) (parallel_generatedRoot_895.w_41_0 ctx) (parallel_generatedRoot_895.w_42_0 ctx) (parallel_generatedRoot_895.w_43_0 ctx) (parallel_generatedRoot_895.w_44_0 ctx) (parallel_generatedRoot_895.w_45_0 ctx) (parallel_generatedRoot_895.w_46_0 ctx)) :
    parallel_generatedRoot_895.Facts_1 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_895.Facts_2  (ctx : parallel_generatedRoot_895Ctx) : Prop where
  t64 : 2 ≤ 3
  t65 : 0 ≤ 0
  t66 : 0 < 25
  t67 : 25 ≤ 25
  h48_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_895.w_47_0 ctx) 1 2 0 25 (parallel_generatedRoot_895.w_48_0 ctx)
  h50_0 : MxxRuntime.concatRows (parallel_generatedRoot_895.w_18_7 ctx) (parallel_generatedRoot_895.w_18_9 ctx) (parallel_generatedRoot_895.w_50_0 ctx)
  h51_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (parallel_generatedRoot_895.w_50_0 ctx) (parallel_generatedRoot_895.w_51_0 ctx)
  h52_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (parallel_generatedRoot_895.w_51_0 ctx) (parallel_generatedRoot_895.w_52_0 ctx)
  t72 : 0 ≤ 1
  t73 : 1 < 2
  t74 : 2 ≤ 3
  t75 : 0 ≤ 0
  t76 : 0 < 25
  t77 : 25 ≤ 25
  h54_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_895.w_53_0 ctx) 1 2 0 25 (parallel_generatedRoot_895.w_54_0 ctx)
  t79 : 0 ≤ 2
  t80 : 2 < 3
  t81 : 3 ≤ 3
  t82 : 0 ≤ 0
  t83 : 0 < 25
  t84 : 25 ≤ 25
  h55_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_895.w_53_0 ctx) 2 3 0 25 (parallel_generatedRoot_895.w_55_0 ctx)
  h56_0 : MxxRuntime.concatRows (parallel_generatedRoot_895.w_54_0 ctx) (parallel_generatedRoot_895.w_55_0 ctx) (parallel_generatedRoot_895.w_56_0 ctx)
  h57_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (parallel_generatedRoot_895.w_56_0 ctx) (parallel_generatedRoot_895.w_57_0 ctx)
  h58_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (parallel_generatedRoot_895.w_57_0 ctx) (parallel_generatedRoot_895.w_58_0 ctx)
  t89 : 0 ≤ 1
  t90 : 1 < 2
  t91 : 2 ≤ 3
  t92 : 0 ≤ 0
  t93 : 0 < 25
  t94 : 25 ≤ 25
  h60_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_895.w_59_0 ctx) 1 2 0 25 (parallel_generatedRoot_895.w_60_0 ctx)

theorem parallel_generatedRoot_895.facts_2  {ctx : parallel_generatedRoot_895Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_895.constraints_2 backend (parallel_generatedRoot_895.w_18_7 ctx) (parallel_generatedRoot_895.w_18_9 ctx) (parallel_generatedRoot_895.w_47_0 ctx) (parallel_generatedRoot_895.w_48_0 ctx) (parallel_generatedRoot_895.w_50_0 ctx) (parallel_generatedRoot_895.w_51_0 ctx) (parallel_generatedRoot_895.w_52_0 ctx) (parallel_generatedRoot_895.w_53_0 ctx) (parallel_generatedRoot_895.w_54_0 ctx) (parallel_generatedRoot_895.w_55_0 ctx) (parallel_generatedRoot_895.w_56_0 ctx) (parallel_generatedRoot_895.w_57_0 ctx) (parallel_generatedRoot_895.w_58_0 ctx) (parallel_generatedRoot_895.w_59_0 ctx) (parallel_generatedRoot_895.w_60_0 ctx)) :
    parallel_generatedRoot_895.Facts_2 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_895.Facts_3  (ctx : parallel_generatedRoot_895Ctx) : Prop where
  t96 : 0 ≤ 2
  t97 : 2 < 3
  t98 : 3 ≤ 3
  t99 : 0 ≤ 0
  t100 : 0 < 25
  t101 : 25 ≤ 25
  h62_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_895.w_47_0 ctx) 2 3 0 25 (parallel_generatedRoot_895.w_62_0 ctx)
  t103 : 0 ≤ 2
  t104 : 2 < 3
  t105 : 3 ≤ 3
  t106 : 0 ≤ 0
  t107 : 0 < 25
  t108 : 25 ≤ 25
  h64_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_895.w_59_0 ctx) 2 3 0 25 (parallel_generatedRoot_895.w_64_0 ctx)
  t110 : 0 ≤ 0
  t111 : 0 < 1
  t112 : 1 ≤ 3
  t113 : 0 ≤ 0
  t114 : 0 < 25
  t115 : 25 ≤ 25
  h66_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_895.w_47_0 ctx) 0 1 0 25 (parallel_generatedRoot_895.w_66_0 ctx)
  t117 : 0 ≤ 0
  t118 : 0 < 1
  t119 : 1 ≤ 3
  t120 : 0 ≤ 0
  t121 : 0 < 25
  t122 : 25 ≤ 25
  h67_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_895.w_41_0 ctx) 0 1 0 25 (parallel_generatedRoot_895.w_67_0 ctx)
  t124 : 0 ≤ 0
  t125 : 0 < 1
  t126 : 1 ≤ 3
  t127 : 0 ≤ 0

theorem parallel_generatedRoot_895.facts_3  {ctx : parallel_generatedRoot_895Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_895.constraints_3 (parallel_generatedRoot_895.w_41_0 ctx) (parallel_generatedRoot_895.w_47_0 ctx) (parallel_generatedRoot_895.w_59_0 ctx) (parallel_generatedRoot_895.w_62_0 ctx) (parallel_generatedRoot_895.w_64_0 ctx) (parallel_generatedRoot_895.w_66_0 ctx) (parallel_generatedRoot_895.w_67_0 ctx)) :
    parallel_generatedRoot_895.Facts_3 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_895.Facts_4  (ctx : parallel_generatedRoot_895Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  t128 : 0 < 25
  t129 : 25 ≤ 25
  h70_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_895.w_59_0 ctx) 0 1 0 25 (parallel_generatedRoot_895.w_70_0 ctx)
  t131 : 0 ≤ 0
  t132 : 0 < 1
  t133 : 1 ≤ 3
  t134 : 0 ≤ 0
  t135 : 0 < 25
  t136 : 25 ≤ 25
  h71_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_895.w_53_0 ctx) 0 1 0 25 (parallel_generatedRoot_895.w_71_0 ctx)
  hout : outputs = parallel_generatedRoot_895.outs ctx

theorem parallel_generatedRoot_895.facts_4  {ctx : parallel_generatedRoot_895Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_895.constraints_4 (parallel_generatedRoot_895.w_49_0 ctx) (parallel_generatedRoot_895.w_53_0 ctx) (parallel_generatedRoot_895.w_59_0 ctx) (parallel_generatedRoot_895.w_61_0 ctx) (parallel_generatedRoot_895.w_63_0 ctx) (parallel_generatedRoot_895.w_65_0 ctx) (parallel_generatedRoot_895.w_69_0 ctx) (parallel_generatedRoot_895.w_70_0 ctx) (parallel_generatedRoot_895.w_71_0 ctx) (parallel_generatedRoot_895.w_73_0 ctx) outputs) :
    parallel_generatedRoot_895.Facts_4 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_895.Facts  (ctx : parallel_generatedRoot_895Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  c0 : parallel_generatedRoot_895.Facts_0 ctx
  c1 : parallel_generatedRoot_895.Facts_1 ctx
  c2 : parallel_generatedRoot_895.Facts_2 ctx
  c3 : parallel_generatedRoot_895.Facts_3 ctx
  c4 : parallel_generatedRoot_895.Facts_4 ctx outputs

theorem parallel_generatedRoot_895.facts  (ctx : parallel_generatedRoot_895Ctx) {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_895.body backend hashModel tape path params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_895.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_895.facts_0 (ctx := ctx) (outputs := outputs) t0.1
  have t1 := t0.2
  have r1 := parallel_generatedRoot_895.facts_1 (ctx := ctx) (outputs := outputs) t1.1
  have t2 := t1.2
  have r2 := parallel_generatedRoot_895.facts_2 (ctx := ctx) (outputs := outputs) t2.1
  have t3 := t2.2
  have r3 := parallel_generatedRoot_895.facts_3 (ctx := ctx) (outputs := outputs) t3.1
  have t4 := t3.2
  have r4 := parallel_generatedRoot_895.facts_4 (ctx := ctx) (outputs := outputs) t4
  ⟨r0, r1, r2, r3, r4⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_927.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) where
  inputs : (sequential_generatedRoot_927.InputType)
  witness : sequential_generatedRoot_927.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_927Ctx" => (sequential_generatedRoot_927.Ctx backend hashModel tape path params i_0)

def sequential_generatedRoot_927.w_0_0  (ctx : sequential_generatedRoot_927Ctx) : mxxMat7 :=
  ctx.inputs.1

def sequential_generatedRoot_927.w_31_0  (ctx : sequential_generatedRoot_927Ctx) : mxxMat7 :=
  ctx.inputs.2.1

def sequential_generatedRoot_927.w_35_0  (ctx : sequential_generatedRoot_927Ctx) : mxxMat7 :=
  ctx.inputs.2.2.1

def sequential_generatedRoot_927.w_39_0  (ctx : sequential_generatedRoot_927Ctx) : mxxMat7 :=
  ctx.inputs.2.2.2.1

def sequential_generatedRoot_927.w_43_0  (ctx : sequential_generatedRoot_927Ctx) : mxxMat7 :=
  ctx.inputs.2.2.2.2.1

def sequential_generatedRoot_927.w_47_0  (ctx : sequential_generatedRoot_927Ctx) : mxxMat7 :=
  ctx.inputs.2.2.2.2.2.1

def sequential_generatedRoot_927.w_2_0  (ctx : sequential_generatedRoot_927Ctx) : Fin 7200 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.1

def sequential_generatedRoot_927.w_8_0  (ctx : sequential_generatedRoot_927Ctx) : Fin 7200 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.1

def sequential_generatedRoot_927.w_12_0  (ctx : sequential_generatedRoot_927Ctx) : Fin 7200 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.1

def sequential_generatedRoot_927.w_16_0  (ctx : sequential_generatedRoot_927Ctx) : Fin 7200 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.1

def sequential_generatedRoot_927.w_20_0  (ctx : sequential_generatedRoot_927Ctx) : Fin 7200 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.1

def sequential_generatedRoot_927.w_24_0  (ctx : sequential_generatedRoot_927Ctx) : Fin 7200 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.1

def sequential_generatedRoot_927.w_1_0  (ctx : sequential_generatedRoot_927Ctx) : mxxMat8 :=
  (MxxRuntime.unitRow ((1 + (Int.ofNat i_0))) : mxxMat8)

def sequential_generatedRoot_927.w_3_0  (ctx : sequential_generatedRoot_927Ctx) : Int :=
  (8 + (8 * (Int.ofNat i_0)))

def sequential_generatedRoot_927.w_4_0  (ctx : sequential_generatedRoot_927Ctx) : mxxMat0 :=
  ctx.witness.w_4_0

def sequential_generatedRoot_927.w_5_0  (ctx : sequential_generatedRoot_927Ctx) : Int :=
  (9 + (8 * (Int.ofNat i_0)))

def sequential_generatedRoot_927.w_6_0  (ctx : sequential_generatedRoot_927Ctx) : mxxMat0 :=
  ctx.witness.w_6_0

def sequential_generatedRoot_927.w_7_0  (ctx : sequential_generatedRoot_927Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (sequential_generatedRoot_927.w_4_0 ctx) (sequential_generatedRoot_927.w_6_0 ctx)

def sequential_generatedRoot_927.w_9_0  (ctx : sequential_generatedRoot_927Ctx) : mxxMat0 :=
  ctx.witness.w_9_0

def sequential_generatedRoot_927.w_10_0  (ctx : sequential_generatedRoot_927Ctx) : mxxMat0 :=
  ctx.witness.w_10_0

def sequential_generatedRoot_927.w_11_0  (ctx : sequential_generatedRoot_927Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (sequential_generatedRoot_927.w_9_0 ctx) (sequential_generatedRoot_927.w_10_0 ctx)

def sequential_generatedRoot_927.w_13_0  (ctx : sequential_generatedRoot_927Ctx) : mxxMat0 :=
  ctx.witness.w_13_0

def sequential_generatedRoot_927.w_14_0  (ctx : sequential_generatedRoot_927Ctx) : mxxMat0 :=
  ctx.witness.w_14_0

def sequential_generatedRoot_927.w_15_0  (ctx : sequential_generatedRoot_927Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (sequential_generatedRoot_927.w_13_0 ctx) (sequential_generatedRoot_927.w_14_0 ctx)

def sequential_generatedRoot_927.w_17_0  (ctx : sequential_generatedRoot_927Ctx) : mxxMat0 :=
  ctx.witness.w_17_0

def sequential_generatedRoot_927.w_18_0  (ctx : sequential_generatedRoot_927Ctx) : mxxMat0 :=
  ctx.witness.w_18_0

def sequential_generatedRoot_927.w_19_0  (ctx : sequential_generatedRoot_927Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (sequential_generatedRoot_927.w_17_0 ctx) (sequential_generatedRoot_927.w_18_0 ctx)

def sequential_generatedRoot_927.w_21_0  (ctx : sequential_generatedRoot_927Ctx) : mxxMat0 :=
  ctx.witness.w_21_0

def sequential_generatedRoot_927.w_22_0  (ctx : sequential_generatedRoot_927Ctx) : mxxMat0 :=
  ctx.witness.w_22_0

def sequential_generatedRoot_927.w_23_0  (ctx : sequential_generatedRoot_927Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (sequential_generatedRoot_927.w_21_0 ctx) (sequential_generatedRoot_927.w_22_0 ctx)

def sequential_generatedRoot_927.w_25_0  (ctx : sequential_generatedRoot_927Ctx) : mxxMat0 :=
  ctx.witness.w_25_0

def sequential_generatedRoot_927.w_26_0  (ctx : sequential_generatedRoot_927Ctx) : mxxMat0 :=
  ctx.witness.w_26_0

def sequential_generatedRoot_927.w_27_0  (ctx : sequential_generatedRoot_927Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (sequential_generatedRoot_927.w_25_0 ctx) (sequential_generatedRoot_927.w_26_0 ctx)

def sequential_generatedRoot_927.w_28_0  (ctx : sequential_generatedRoot_927Ctx) : mxxMat0 :=
  ctx.witness.w_28_0

def sequential_generatedRoot_927.w_28_1  (ctx : sequential_generatedRoot_927Ctx) : mxxMat0 :=
  ctx.witness.w_28_1

def sequential_generatedRoot_927.w_28_2  (ctx : sequential_generatedRoot_927Ctx) : mxxMat0 :=
  ctx.witness.w_28_2

def sequential_generatedRoot_927.w_28_3  (ctx : sequential_generatedRoot_927Ctx) : mxxMat0 :=
  ctx.witness.w_28_3

def sequential_generatedRoot_927.w_28_4  (ctx : sequential_generatedRoot_927Ctx) : mxxMat0 :=
  ctx.witness.w_28_4

def sequential_generatedRoot_927.w_28_5  (ctx : sequential_generatedRoot_927Ctx) : mxxMat0 :=
  ctx.witness.w_28_5

def sequential_generatedRoot_927.w_29_0  (ctx : sequential_generatedRoot_927Ctx) : mxxMat7 :=
  ctx.witness.w_29_0

def sequential_generatedRoot_927.w_30_0  (ctx : sequential_generatedRoot_927Ctx) : mxxMat7 :=
  MxxRuntime.matrixAdd (sequential_generatedRoot_927.w_0_0 ctx) (sequential_generatedRoot_927.w_29_0 ctx)

def sequential_generatedRoot_927.w_32_0  (ctx : sequential_generatedRoot_927Ctx) : mxxMat8 :=
  (MxxRuntime.unitRow ((1 + (Int.ofNat i_0))) : mxxMat8)

def sequential_generatedRoot_927.w_33_0  (ctx : sequential_generatedRoot_927Ctx) : mxxMat7 :=
  ctx.witness.w_33_0

def sequential_generatedRoot_927.w_34_0  (ctx : sequential_generatedRoot_927Ctx) : mxxMat7 :=
  MxxRuntime.matrixAdd (sequential_generatedRoot_927.w_31_0 ctx) (sequential_generatedRoot_927.w_33_0 ctx)

def sequential_generatedRoot_927.w_36_0  (ctx : sequential_generatedRoot_927Ctx) : mxxMat8 :=
  (MxxRuntime.unitRow ((1 + (Int.ofNat i_0))) : mxxMat8)

def sequential_generatedRoot_927.w_37_0  (ctx : sequential_generatedRoot_927Ctx) : mxxMat7 :=
  ctx.witness.w_37_0

def sequential_generatedRoot_927.w_38_0  (ctx : sequential_generatedRoot_927Ctx) : mxxMat7 :=
  MxxRuntime.matrixAdd (sequential_generatedRoot_927.w_35_0 ctx) (sequential_generatedRoot_927.w_37_0 ctx)

def sequential_generatedRoot_927.w_40_0  (ctx : sequential_generatedRoot_927Ctx) : mxxMat8 :=
  (MxxRuntime.unitRow ((1 + (Int.ofNat i_0))) : mxxMat8)

def sequential_generatedRoot_927.w_41_0  (ctx : sequential_generatedRoot_927Ctx) : mxxMat7 :=
  ctx.witness.w_41_0

def sequential_generatedRoot_927.w_42_0  (ctx : sequential_generatedRoot_927Ctx) : mxxMat7 :=
  MxxRuntime.matrixAdd (sequential_generatedRoot_927.w_39_0 ctx) (sequential_generatedRoot_927.w_41_0 ctx)

def sequential_generatedRoot_927.w_44_0  (ctx : sequential_generatedRoot_927Ctx) : mxxMat8 :=
  (MxxRuntime.unitRow ((1 + (Int.ofNat i_0))) : mxxMat8)

def sequential_generatedRoot_927.w_45_0  (ctx : sequential_generatedRoot_927Ctx) : mxxMat7 :=
  ctx.witness.w_45_0

def sequential_generatedRoot_927.w_46_0  (ctx : sequential_generatedRoot_927Ctx) : mxxMat7 :=
  MxxRuntime.matrixAdd (sequential_generatedRoot_927.w_43_0 ctx) (sequential_generatedRoot_927.w_45_0 ctx)

def sequential_generatedRoot_927.w_48_0  (ctx : sequential_generatedRoot_927Ctx) : mxxMat8 :=
  (MxxRuntime.unitRow ((1 + (Int.ofNat i_0))) : mxxMat8)

def sequential_generatedRoot_927.w_49_0  (ctx : sequential_generatedRoot_927Ctx) : mxxMat7 :=
  ctx.witness.w_49_0

def sequential_generatedRoot_927.w_50_0  (ctx : sequential_generatedRoot_927Ctx) : mxxMat7 :=
  MxxRuntime.matrixAdd (sequential_generatedRoot_927.w_47_0 ctx) (sequential_generatedRoot_927.w_49_0 ctx)

def sequential_generatedRoot_927.outs  (ctx : sequential_generatedRoot_927Ctx) : mxxMat7 × mxxMat7 × mxxMat7 × mxxMat7 × mxxMat7 × mxxMat7 × Unit :=
  ((sequential_generatedRoot_927.w_30_0 ctx), ((sequential_generatedRoot_927.w_34_0 ctx), ((sequential_generatedRoot_927.w_38_0 ctx), ((sequential_generatedRoot_927.w_42_0 ctx), ((sequential_generatedRoot_927.w_46_0 ctx), ((sequential_generatedRoot_927.w_50_0 ctx), ()))))))

structure sequential_generatedRoot_927.Facts_0  (ctx : sequential_generatedRoot_927Ctx) : Prop where
  t0 : 0 ≤ ((1 + (Int.ofNat i_0)))
  t1 : ((1 + (Int.ofNat i_0))) < 900
  h3_0 : 0 ≤ (sequential_generatedRoot_927.w_3_0 ctx)
  h3_0_1 : (sequential_generatedRoot_927.w_3_0 ctx) < 7200
  h4_0 : MxxRuntime.familyGetDynamic (sequential_generatedRoot_927.w_2_0 ctx) (sequential_generatedRoot_927.w_3_0 ctx) (sequential_generatedRoot_927.w_4_0 ctx)
  h5_0 : 0 ≤ (sequential_generatedRoot_927.w_5_0 ctx)
  h5_0_1 : (sequential_generatedRoot_927.w_5_0 ctx) < 7200
  h6_0 : MxxRuntime.familyGetDynamic (sequential_generatedRoot_927.w_2_0 ctx) (sequential_generatedRoot_927.w_5_0 ctx) (sequential_generatedRoot_927.w_6_0 ctx)
  h3_0_2 : 0 ≤ (sequential_generatedRoot_927.w_3_0 ctx)
  h3_0_3 : (sequential_generatedRoot_927.w_3_0 ctx) < 7200
  h9_0 : MxxRuntime.familyGetDynamic (sequential_generatedRoot_927.w_8_0 ctx) (sequential_generatedRoot_927.w_3_0 ctx) (sequential_generatedRoot_927.w_9_0 ctx)
  h5_0_2 : 0 ≤ (sequential_generatedRoot_927.w_5_0 ctx)
  h5_0_3 : (sequential_generatedRoot_927.w_5_0 ctx) < 7200
  h10_0 : MxxRuntime.familyGetDynamic (sequential_generatedRoot_927.w_8_0 ctx) (sequential_generatedRoot_927.w_5_0 ctx) (sequential_generatedRoot_927.w_10_0 ctx)
  h3_0_4 : 0 ≤ (sequential_generatedRoot_927.w_3_0 ctx)
  h3_0_5 : (sequential_generatedRoot_927.w_3_0 ctx) < 7200
  h13_0 : MxxRuntime.familyGetDynamic (sequential_generatedRoot_927.w_12_0 ctx) (sequential_generatedRoot_927.w_3_0 ctx) (sequential_generatedRoot_927.w_13_0 ctx)
  h5_0_4 : 0 ≤ (sequential_generatedRoot_927.w_5_0 ctx)
  h5_0_5 : (sequential_generatedRoot_927.w_5_0 ctx) < 7200
  h14_0 : MxxRuntime.familyGetDynamic (sequential_generatedRoot_927.w_12_0 ctx) (sequential_generatedRoot_927.w_5_0 ctx) (sequential_generatedRoot_927.w_14_0 ctx)
  h3_0_6 : 0 ≤ (sequential_generatedRoot_927.w_3_0 ctx)
  h3_0_7 : (sequential_generatedRoot_927.w_3_0 ctx) < 7200
  h17_0 : MxxRuntime.familyGetDynamic (sequential_generatedRoot_927.w_16_0 ctx) (sequential_generatedRoot_927.w_3_0 ctx) (sequential_generatedRoot_927.w_17_0 ctx)
  h5_0_6 : 0 ≤ (sequential_generatedRoot_927.w_5_0 ctx)
  h5_0_7 : (sequential_generatedRoot_927.w_5_0 ctx) < 7200
  h18_0 : MxxRuntime.familyGetDynamic (sequential_generatedRoot_927.w_16_0 ctx) (sequential_generatedRoot_927.w_5_0 ctx) (sequential_generatedRoot_927.w_18_0 ctx)
  h3_0_8 : 0 ≤ (sequential_generatedRoot_927.w_3_0 ctx)
  h3_0_9 : (sequential_generatedRoot_927.w_3_0 ctx) < 7200
  h21_0 : MxxRuntime.familyGetDynamic (sequential_generatedRoot_927.w_20_0 ctx) (sequential_generatedRoot_927.w_3_0 ctx) (sequential_generatedRoot_927.w_21_0 ctx)
  h5_0_8 : 0 ≤ (sequential_generatedRoot_927.w_5_0 ctx)
  h5_0_9 : (sequential_generatedRoot_927.w_5_0 ctx) < 7200
  h22_0 : MxxRuntime.familyGetDynamic (sequential_generatedRoot_927.w_20_0 ctx) (sequential_generatedRoot_927.w_5_0 ctx) (sequential_generatedRoot_927.w_22_0 ctx)

theorem sequential_generatedRoot_927.facts_0  {ctx : sequential_generatedRoot_927Ctx} {outputs : mxxMat7 × mxxMat7 × mxxMat7 × mxxMat7 × mxxMat7 × mxxMat7 × Unit}
    (h : sequential_generatedRoot_927.constraints_0 i_0 (sequential_generatedRoot_927.w_2_0 ctx) (sequential_generatedRoot_927.w_8_0 ctx) (sequential_generatedRoot_927.w_12_0 ctx) (sequential_generatedRoot_927.w_16_0 ctx) (sequential_generatedRoot_927.w_20_0 ctx) (sequential_generatedRoot_927.w_3_0 ctx) (sequential_generatedRoot_927.w_4_0 ctx) (sequential_generatedRoot_927.w_5_0 ctx) (sequential_generatedRoot_927.w_6_0 ctx) (sequential_generatedRoot_927.w_9_0 ctx) (sequential_generatedRoot_927.w_10_0 ctx) (sequential_generatedRoot_927.w_13_0 ctx) (sequential_generatedRoot_927.w_14_0 ctx) (sequential_generatedRoot_927.w_17_0 ctx) (sequential_generatedRoot_927.w_18_0 ctx) (sequential_generatedRoot_927.w_21_0 ctx) (sequential_generatedRoot_927.w_22_0 ctx)) :
    sequential_generatedRoot_927.Facts_0 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure sequential_generatedRoot_927.Facts_1  (ctx : sequential_generatedRoot_927Ctx) (outputs : mxxMat7 × mxxMat7 × mxxMat7 × mxxMat7 × mxxMat7 × mxxMat7 × Unit) : Prop where
  h3_0_10 : 0 ≤ (sequential_generatedRoot_927.w_3_0 ctx)
  h3_0_11 : (sequential_generatedRoot_927.w_3_0 ctx) < 7200
  h25_0 : MxxRuntime.familyGetDynamic (sequential_generatedRoot_927.w_24_0 ctx) (sequential_generatedRoot_927.w_3_0 ctx) (sequential_generatedRoot_927.w_25_0 ctx)
  h5_0_10 : 0 ≤ (sequential_generatedRoot_927.w_5_0 ctx)
  h5_0_11 : (sequential_generatedRoot_927.w_5_0 ctx) < 7200
  h26_0 : MxxRuntime.familyGetDynamic (sequential_generatedRoot_927.w_24_0 ctx) (sequential_generatedRoot_927.w_5_0 ctx) (sequential_generatedRoot_927.w_26_0 ctx)
  t38 : 0 ≤ (6)
  h28_5 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => scope_80 backend hashModel tape (path ++ [28, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, ((sequential_generatedRoot_927.w_2_0 ctx), ((sequential_generatedRoot_927.w_8_0 ctx), ((sequential_generatedRoot_927.w_12_0 ctx), ((sequential_generatedRoot_927.w_16_0 ctx), ((sequential_generatedRoot_927.w_20_0 ctx), ((sequential_generatedRoot_927.w_24_0 ctx), ())))))))))))) next) (Int.toNat (6)) ((sequential_generatedRoot_927.w_7_0 ctx), ((sequential_generatedRoot_927.w_11_0 ctx), ((sequential_generatedRoot_927.w_15_0 ctx), ((sequential_generatedRoot_927.w_19_0 ctx), ((sequential_generatedRoot_927.w_23_0 ctx), ((sequential_generatedRoot_927.w_27_0 ctx), ())))))) ((sequential_generatedRoot_927.w_28_0 ctx), ((sequential_generatedRoot_927.w_28_1 ctx), ((sequential_generatedRoot_927.w_28_2 ctx), ((sequential_generatedRoot_927.w_28_3 ctx), ((sequential_generatedRoot_927.w_28_4 ctx), ((sequential_generatedRoot_927.w_28_5 ctx), ()))))))
  h29_0 : MxxRuntime.tensorRuns (sequential_generatedRoot_927.w_1_0 ctx) (sequential_generatedRoot_927.w_28_0 ctx) (sequential_generatedRoot_927.w_29_0 ctx)
  t41 : 0 ≤ ((1 + (Int.ofNat i_0)))
  t42 : ((1 + (Int.ofNat i_0))) < 900
  h33_0 : MxxRuntime.tensorRuns (sequential_generatedRoot_927.w_32_0 ctx) (sequential_generatedRoot_927.w_28_1 ctx) (sequential_generatedRoot_927.w_33_0 ctx)
  t44 : 0 ≤ ((1 + (Int.ofNat i_0)))
  t45 : ((1 + (Int.ofNat i_0))) < 900
  h37_0 : MxxRuntime.tensorRuns (sequential_generatedRoot_927.w_36_0 ctx) (sequential_generatedRoot_927.w_28_2 ctx) (sequential_generatedRoot_927.w_37_0 ctx)
  t47 : 0 ≤ ((1 + (Int.ofNat i_0)))
  t48 : ((1 + (Int.ofNat i_0))) < 900
  h41_0 : MxxRuntime.tensorRuns (sequential_generatedRoot_927.w_40_0 ctx) (sequential_generatedRoot_927.w_28_3 ctx) (sequential_generatedRoot_927.w_41_0 ctx)
  t50 : 0 ≤ ((1 + (Int.ofNat i_0)))
  t51 : ((1 + (Int.ofNat i_0))) < 900
  h45_0 : MxxRuntime.tensorRuns (sequential_generatedRoot_927.w_44_0 ctx) (sequential_generatedRoot_927.w_28_4 ctx) (sequential_generatedRoot_927.w_45_0 ctx)
  t53 : 0 ≤ ((1 + (Int.ofNat i_0)))
  t54 : ((1 + (Int.ofNat i_0))) < 900
  h49_0 : MxxRuntime.tensorRuns (sequential_generatedRoot_927.w_48_0 ctx) (sequential_generatedRoot_927.w_28_5 ctx) (sequential_generatedRoot_927.w_49_0 ctx)
  hout : outputs = sequential_generatedRoot_927.outs ctx

theorem sequential_generatedRoot_927.facts_1  {ctx : sequential_generatedRoot_927Ctx} {outputs : mxxMat7 × mxxMat7 × mxxMat7 × mxxMat7 × mxxMat7 × mxxMat7 × Unit}
    (h : sequential_generatedRoot_927.constraints_1 backend hashModel tape path params i_0 (sequential_generatedRoot_927.w_2_0 ctx) (sequential_generatedRoot_927.w_8_0 ctx) (sequential_generatedRoot_927.w_12_0 ctx) (sequential_generatedRoot_927.w_16_0 ctx) (sequential_generatedRoot_927.w_20_0 ctx) (sequential_generatedRoot_927.w_24_0 ctx) (sequential_generatedRoot_927.w_1_0 ctx) (sequential_generatedRoot_927.w_3_0 ctx) (sequential_generatedRoot_927.w_5_0 ctx) (sequential_generatedRoot_927.w_7_0 ctx) (sequential_generatedRoot_927.w_11_0 ctx) (sequential_generatedRoot_927.w_15_0 ctx) (sequential_generatedRoot_927.w_19_0 ctx) (sequential_generatedRoot_927.w_23_0 ctx) (sequential_generatedRoot_927.w_25_0 ctx) (sequential_generatedRoot_927.w_26_0 ctx) (sequential_generatedRoot_927.w_27_0 ctx) (sequential_generatedRoot_927.w_28_0 ctx) (sequential_generatedRoot_927.w_28_1 ctx) (sequential_generatedRoot_927.w_28_2 ctx) (sequential_generatedRoot_927.w_28_3 ctx) (sequential_generatedRoot_927.w_28_4 ctx) (sequential_generatedRoot_927.w_28_5 ctx) (sequential_generatedRoot_927.w_29_0 ctx) (sequential_generatedRoot_927.w_30_0 ctx) (sequential_generatedRoot_927.w_32_0 ctx) (sequential_generatedRoot_927.w_33_0 ctx) (sequential_generatedRoot_927.w_34_0 ctx) (sequential_generatedRoot_927.w_36_0 ctx) (sequential_generatedRoot_927.w_37_0 ctx) (sequential_generatedRoot_927.w_38_0 ctx) (sequential_generatedRoot_927.w_40_0 ctx) (sequential_generatedRoot_927.w_41_0 ctx) (sequential_generatedRoot_927.w_42_0 ctx) (sequential_generatedRoot_927.w_44_0 ctx) (sequential_generatedRoot_927.w_45_0 ctx) (sequential_generatedRoot_927.w_46_0 ctx) (sequential_generatedRoot_927.w_48_0 ctx) (sequential_generatedRoot_927.w_49_0 ctx) (sequential_generatedRoot_927.w_50_0 ctx) outputs) :
    sequential_generatedRoot_927.Facts_1 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure sequential_generatedRoot_927.Facts  (ctx : sequential_generatedRoot_927Ctx) (outputs : mxxMat7 × mxxMat7 × mxxMat7 × mxxMat7 × mxxMat7 × mxxMat7 × Unit) : Prop where
  c0 : sequential_generatedRoot_927.Facts_0 ctx
  c1 : sequential_generatedRoot_927.Facts_1 ctx outputs

theorem sequential_generatedRoot_927.facts  (ctx : sequential_generatedRoot_927Ctx) {outputs : mxxMat7 × mxxMat7 × mxxMat7 × mxxMat7 × mxxMat7 × mxxMat7 × Unit}
    (h : sequential_generatedRoot_927.body backend hashModel tape path params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_927.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_927.facts_0 (ctx := ctx) (outputs := outputs) t0.1
  have t1 := t0.2
  have r1 := sequential_generatedRoot_927.facts_1 (ctx := ctx) (outputs := outputs) t1
  ⟨r0, r1⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_944.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) where
  inputs : (Fin 72000 → mxxMat0) × (Fin 72000 → mxxMat0) × (Fin 72000 → mxxMat0) × (Fin 72000 → mxxMat0) × (Fin 72000 → mxxMat0) × (Fin 72000 → mxxMat0) × Unit
  witness : parallel_generatedRoot_944.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_944Ctx" => (parallel_generatedRoot_944.Ctx backend hashModel tape path params i_0)

def parallel_generatedRoot_944.w_0_0  (ctx : parallel_generatedRoot_944Ctx) : Fin 72000 → mxxMat0 :=
  ctx.inputs.1

def parallel_generatedRoot_944.w_1_0  (ctx : parallel_generatedRoot_944Ctx) : Fin 72000 → mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_944.w_2_0  (ctx : parallel_generatedRoot_944Ctx) : Fin 72000 → mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_944.w_3_0  (ctx : parallel_generatedRoot_944Ctx) : Fin 72000 → mxxMat0 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_944.w_4_0  (ctx : parallel_generatedRoot_944Ctx) : Fin 72000 → mxxMat0 :=
  ctx.inputs.2.2.2.2.1

def parallel_generatedRoot_944.w_5_0  (ctx : parallel_generatedRoot_944Ctx) : Fin 72000 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.1

def parallel_generatedRoot_944.w_6_0  (ctx : parallel_generatedRoot_944Ctx) : Fin 2 → mxxMat0 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_944.w_6_1  (ctx : parallel_generatedRoot_944Ctx) : Fin 2 → mxxMat0 :=
  ctx.witness.w_6_1

def parallel_generatedRoot_944.w_6_2  (ctx : parallel_generatedRoot_944Ctx) : Fin 2 → mxxMat0 :=
  ctx.witness.w_6_2

def parallel_generatedRoot_944.w_6_3  (ctx : parallel_generatedRoot_944Ctx) : Fin 2 → mxxMat0 :=
  ctx.witness.w_6_3

def parallel_generatedRoot_944.w_6_4  (ctx : parallel_generatedRoot_944Ctx) : Fin 2 → mxxMat0 :=
  ctx.witness.w_6_4

def parallel_generatedRoot_944.w_6_5  (ctx : parallel_generatedRoot_944Ctx) : Fin 2 → mxxMat0 :=
  ctx.witness.w_6_5

def parallel_generatedRoot_944.w_7_0  (ctx : parallel_generatedRoot_944Ctx) : mxxMat0 :=
  ctx.witness.w_7_0

def parallel_generatedRoot_944.w_8_0  (ctx : parallel_generatedRoot_944Ctx) : mxxMat0 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_944.w_9_0  (ctx : parallel_generatedRoot_944Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_944.w_7_0 ctx) (parallel_generatedRoot_944.w_8_0 ctx)

def parallel_generatedRoot_944.w_10_0  (ctx : parallel_generatedRoot_944Ctx) : mxxMat0 :=
  ctx.witness.w_10_0

def parallel_generatedRoot_944.w_11_0  (ctx : parallel_generatedRoot_944Ctx) : mxxMat0 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_944.w_12_0  (ctx : parallel_generatedRoot_944Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_944.w_10_0 ctx) (parallel_generatedRoot_944.w_11_0 ctx)

def parallel_generatedRoot_944.w_13_0  (ctx : parallel_generatedRoot_944Ctx) : mxxMat0 :=
  ctx.witness.w_13_0

def parallel_generatedRoot_944.w_14_0  (ctx : parallel_generatedRoot_944Ctx) : mxxMat0 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_944.w_15_0  (ctx : parallel_generatedRoot_944Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_944.w_13_0 ctx) (parallel_generatedRoot_944.w_14_0 ctx)

def parallel_generatedRoot_944.w_16_0  (ctx : parallel_generatedRoot_944Ctx) : mxxMat0 :=
  ctx.witness.w_16_0

def parallel_generatedRoot_944.w_17_0  (ctx : parallel_generatedRoot_944Ctx) : mxxMat0 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_944.w_18_0  (ctx : parallel_generatedRoot_944Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_944.w_16_0 ctx) (parallel_generatedRoot_944.w_17_0 ctx)

def parallel_generatedRoot_944.w_19_0  (ctx : parallel_generatedRoot_944Ctx) : mxxMat0 :=
  ctx.witness.w_19_0

def parallel_generatedRoot_944.w_20_0  (ctx : parallel_generatedRoot_944Ctx) : mxxMat0 :=
  ctx.witness.w_20_0

def parallel_generatedRoot_944.w_21_0  (ctx : parallel_generatedRoot_944Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_944.w_19_0 ctx) (parallel_generatedRoot_944.w_20_0 ctx)

def parallel_generatedRoot_944.w_22_0  (ctx : parallel_generatedRoot_944Ctx) : mxxMat0 :=
  ctx.witness.w_22_0

def parallel_generatedRoot_944.w_23_0  (ctx : parallel_generatedRoot_944Ctx) : mxxMat0 :=
  ctx.witness.w_23_0

def parallel_generatedRoot_944.w_24_0  (ctx : parallel_generatedRoot_944Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_944.w_22_0 ctx) (parallel_generatedRoot_944.w_23_0 ctx)

def parallel_generatedRoot_944.w_25_0  (ctx : parallel_generatedRoot_944Ctx) : mxxMat0 :=
  ctx.witness.w_25_0

def parallel_generatedRoot_944.w_25_1  (ctx : parallel_generatedRoot_944Ctx) : mxxMat0 :=
  ctx.witness.w_25_1

def parallel_generatedRoot_944.w_25_2  (ctx : parallel_generatedRoot_944Ctx) : mxxMat0 :=
  ctx.witness.w_25_2

def parallel_generatedRoot_944.w_25_3  (ctx : parallel_generatedRoot_944Ctx) : mxxMat0 :=
  ctx.witness.w_25_3

def parallel_generatedRoot_944.w_25_4  (ctx : parallel_generatedRoot_944Ctx) : mxxMat0 :=
  ctx.witness.w_25_4

def parallel_generatedRoot_944.w_25_5  (ctx : parallel_generatedRoot_944Ctx) : mxxMat0 :=
  ctx.witness.w_25_5

def parallel_generatedRoot_944.outs  (ctx : parallel_generatedRoot_944Ctx) : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit :=
  ((parallel_generatedRoot_944.w_25_0 ctx), ((parallel_generatedRoot_944.w_25_1 ctx), ((parallel_generatedRoot_944.w_25_2 ctx), ((parallel_generatedRoot_944.w_25_3 ctx), ((parallel_generatedRoot_944.w_25_4 ctx), ((parallel_generatedRoot_944.w_25_5 ctx), ()))))))

structure parallel_generatedRoot_944.Facts_0  (ctx : parallel_generatedRoot_944Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  t0 : (2) = 2
  h6_5 : (∀ i : Fin 2, parallel_parallel_generatedRoot_944_6 backend hashModel tape (path ++ [6, i.val]) params i_0 i ((parallel_generatedRoot_944.w_0_0 ctx), ((parallel_generatedRoot_944.w_1_0 ctx), ((parallel_generatedRoot_944.w_2_0 ctx), ((parallel_generatedRoot_944.w_3_0 ctx), ((parallel_generatedRoot_944.w_4_0 ctx), ((parallel_generatedRoot_944.w_5_0 ctx), ())))))) (((parallel_generatedRoot_944.w_6_0 ctx) i), (((parallel_generatedRoot_944.w_6_1 ctx) i), (((parallel_generatedRoot_944.w_6_2 ctx) i), (((parallel_generatedRoot_944.w_6_3 ctx) i), (((parallel_generatedRoot_944.w_6_4 ctx) i), (((parallel_generatedRoot_944.w_6_5 ctx) i), ())))))))
  h7_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_944.w_6_0 ctx) (0) (parallel_generatedRoot_944.w_7_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_944.w_6_0 ctx) (1) (parallel_generatedRoot_944.w_8_0 ctx)
  h10_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_944.w_6_1 ctx) (0) (parallel_generatedRoot_944.w_10_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_944.w_6_1 ctx) (1) (parallel_generatedRoot_944.w_11_0 ctx)
  h13_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_944.w_6_2 ctx) (0) (parallel_generatedRoot_944.w_13_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_944.w_6_2 ctx) (1) (parallel_generatedRoot_944.w_14_0 ctx)
  h16_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_944.w_6_3 ctx) (0) (parallel_generatedRoot_944.w_16_0 ctx)
  h17_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_944.w_6_3 ctx) (1) (parallel_generatedRoot_944.w_17_0 ctx)
  h19_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_944.w_6_4 ctx) (0) (parallel_generatedRoot_944.w_19_0 ctx)
  h20_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_944.w_6_4 ctx) (1) (parallel_generatedRoot_944.w_20_0 ctx)
  h22_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_944.w_6_5 ctx) (0) (parallel_generatedRoot_944.w_22_0 ctx)
  h23_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_944.w_6_5 ctx) (1) (parallel_generatedRoot_944.w_23_0 ctx)
  t14 : 0 ≤ (0)
  h25_5 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_944_25 backend hashModel tape (path ++ [25, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, ((parallel_generatedRoot_944.w_6_0 ctx), ((parallel_generatedRoot_944.w_6_1 ctx), ((parallel_generatedRoot_944.w_6_2 ctx), ((parallel_generatedRoot_944.w_6_3 ctx), ((parallel_generatedRoot_944.w_6_4 ctx), ((parallel_generatedRoot_944.w_6_5 ctx), ())))))))))))) next) (Int.toNat (0)) ((parallel_generatedRoot_944.w_9_0 ctx), ((parallel_generatedRoot_944.w_12_0 ctx), ((parallel_generatedRoot_944.w_15_0 ctx), ((parallel_generatedRoot_944.w_18_0 ctx), ((parallel_generatedRoot_944.w_21_0 ctx), ((parallel_generatedRoot_944.w_24_0 ctx), ())))))) ((parallel_generatedRoot_944.w_25_0 ctx), ((parallel_generatedRoot_944.w_25_1 ctx), ((parallel_generatedRoot_944.w_25_2 ctx), ((parallel_generatedRoot_944.w_25_3 ctx), ((parallel_generatedRoot_944.w_25_4 ctx), ((parallel_generatedRoot_944.w_25_5 ctx), ()))))))
  hout : outputs = parallel_generatedRoot_944.outs ctx

theorem parallel_generatedRoot_944.facts_0  {ctx : parallel_generatedRoot_944Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_944.constraints_0 backend hashModel tape path params i_0 (parallel_generatedRoot_944.w_0_0 ctx) (parallel_generatedRoot_944.w_1_0 ctx) (parallel_generatedRoot_944.w_2_0 ctx) (parallel_generatedRoot_944.w_3_0 ctx) (parallel_generatedRoot_944.w_4_0 ctx) (parallel_generatedRoot_944.w_5_0 ctx) (parallel_generatedRoot_944.w_6_0 ctx) (parallel_generatedRoot_944.w_6_1 ctx) (parallel_generatedRoot_944.w_6_2 ctx) (parallel_generatedRoot_944.w_6_3 ctx) (parallel_generatedRoot_944.w_6_4 ctx) (parallel_generatedRoot_944.w_6_5 ctx) (parallel_generatedRoot_944.w_7_0 ctx) (parallel_generatedRoot_944.w_8_0 ctx) (parallel_generatedRoot_944.w_9_0 ctx) (parallel_generatedRoot_944.w_10_0 ctx) (parallel_generatedRoot_944.w_11_0 ctx) (parallel_generatedRoot_944.w_12_0 ctx) (parallel_generatedRoot_944.w_13_0 ctx) (parallel_generatedRoot_944.w_14_0 ctx) (parallel_generatedRoot_944.w_15_0 ctx) (parallel_generatedRoot_944.w_16_0 ctx) (parallel_generatedRoot_944.w_17_0 ctx) (parallel_generatedRoot_944.w_18_0 ctx) (parallel_generatedRoot_944.w_19_0 ctx) (parallel_generatedRoot_944.w_20_0 ctx) (parallel_generatedRoot_944.w_21_0 ctx) (parallel_generatedRoot_944.w_22_0 ctx) (parallel_generatedRoot_944.w_23_0 ctx) (parallel_generatedRoot_944.w_24_0 ctx) (parallel_generatedRoot_944.w_25_0 ctx) (parallel_generatedRoot_944.w_25_1 ctx) (parallel_generatedRoot_944.w_25_2 ctx) (parallel_generatedRoot_944.w_25_3 ctx) (parallel_generatedRoot_944.w_25_4 ctx) (parallel_generatedRoot_944.w_25_5 ctx) outputs) :
    parallel_generatedRoot_944.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_944.Facts  (ctx : parallel_generatedRoot_944Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  c0 : parallel_generatedRoot_944.Facts_0 ctx outputs

theorem parallel_generatedRoot_944.facts  (ctx : parallel_generatedRoot_944Ctx) {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_944.body backend hashModel tape path params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_944.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_944.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_948.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) where
  inputs : (Fin 72000 → mxxMat0) × (Fin 72000 → mxxMat0) × (Fin 72000 → mxxMat0) × (Fin 72000 → mxxMat0) × (Fin 72000 → mxxMat0) × (Fin 72000 → mxxMat0) × Unit
  witness : parallel_generatedRoot_948.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_948Ctx" => (parallel_generatedRoot_948.Ctx backend hashModel tape path params i_0)

def parallel_generatedRoot_948.w_0_0  (ctx : parallel_generatedRoot_948Ctx) : Fin 72000 → mxxMat0 :=
  ctx.inputs.1

def parallel_generatedRoot_948.w_1_0  (ctx : parallel_generatedRoot_948Ctx) : Fin 72000 → mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_948.w_2_0  (ctx : parallel_generatedRoot_948Ctx) : Fin 72000 → mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_948.w_3_0  (ctx : parallel_generatedRoot_948Ctx) : Fin 72000 → mxxMat0 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_948.w_4_0  (ctx : parallel_generatedRoot_948Ctx) : Fin 72000 → mxxMat0 :=
  ctx.inputs.2.2.2.2.1

def parallel_generatedRoot_948.w_5_0  (ctx : parallel_generatedRoot_948Ctx) : Fin 72000 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.1

def parallel_generatedRoot_948.w_6_0  (ctx : parallel_generatedRoot_948Ctx) : Fin 2 → mxxMat0 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_948.w_6_1  (ctx : parallel_generatedRoot_948Ctx) : Fin 2 → mxxMat0 :=
  ctx.witness.w_6_1

def parallel_generatedRoot_948.w_6_2  (ctx : parallel_generatedRoot_948Ctx) : Fin 2 → mxxMat0 :=
  ctx.witness.w_6_2

def parallel_generatedRoot_948.w_6_3  (ctx : parallel_generatedRoot_948Ctx) : Fin 2 → mxxMat0 :=
  ctx.witness.w_6_3

def parallel_generatedRoot_948.w_6_4  (ctx : parallel_generatedRoot_948Ctx) : Fin 2 → mxxMat0 :=
  ctx.witness.w_6_4

def parallel_generatedRoot_948.w_6_5  (ctx : parallel_generatedRoot_948Ctx) : Fin 2 → mxxMat0 :=
  ctx.witness.w_6_5

def parallel_generatedRoot_948.w_7_0  (ctx : parallel_generatedRoot_948Ctx) : mxxMat0 :=
  ctx.witness.w_7_0

def parallel_generatedRoot_948.w_8_0  (ctx : parallel_generatedRoot_948Ctx) : mxxMat0 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_948.w_9_0  (ctx : parallel_generatedRoot_948Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_948.w_7_0 ctx) (parallel_generatedRoot_948.w_8_0 ctx)

def parallel_generatedRoot_948.w_10_0  (ctx : parallel_generatedRoot_948Ctx) : mxxMat0 :=
  ctx.witness.w_10_0

def parallel_generatedRoot_948.w_11_0  (ctx : parallel_generatedRoot_948Ctx) : mxxMat0 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_948.w_12_0  (ctx : parallel_generatedRoot_948Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_948.w_10_0 ctx) (parallel_generatedRoot_948.w_11_0 ctx)

def parallel_generatedRoot_948.w_13_0  (ctx : parallel_generatedRoot_948Ctx) : mxxMat0 :=
  ctx.witness.w_13_0

def parallel_generatedRoot_948.w_14_0  (ctx : parallel_generatedRoot_948Ctx) : mxxMat0 :=
  ctx.witness.w_14_0

def parallel_generatedRoot_948.w_15_0  (ctx : parallel_generatedRoot_948Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_948.w_13_0 ctx) (parallel_generatedRoot_948.w_14_0 ctx)

def parallel_generatedRoot_948.w_16_0  (ctx : parallel_generatedRoot_948Ctx) : mxxMat0 :=
  ctx.witness.w_16_0

def parallel_generatedRoot_948.w_17_0  (ctx : parallel_generatedRoot_948Ctx) : mxxMat0 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_948.w_18_0  (ctx : parallel_generatedRoot_948Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_948.w_16_0 ctx) (parallel_generatedRoot_948.w_17_0 ctx)

def parallel_generatedRoot_948.w_19_0  (ctx : parallel_generatedRoot_948Ctx) : mxxMat0 :=
  ctx.witness.w_19_0

def parallel_generatedRoot_948.w_20_0  (ctx : parallel_generatedRoot_948Ctx) : mxxMat0 :=
  ctx.witness.w_20_0

def parallel_generatedRoot_948.w_21_0  (ctx : parallel_generatedRoot_948Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_948.w_19_0 ctx) (parallel_generatedRoot_948.w_20_0 ctx)

def parallel_generatedRoot_948.w_22_0  (ctx : parallel_generatedRoot_948Ctx) : mxxMat0 :=
  ctx.witness.w_22_0

def parallel_generatedRoot_948.w_23_0  (ctx : parallel_generatedRoot_948Ctx) : mxxMat0 :=
  ctx.witness.w_23_0

def parallel_generatedRoot_948.w_24_0  (ctx : parallel_generatedRoot_948Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_948.w_22_0 ctx) (parallel_generatedRoot_948.w_23_0 ctx)

def parallel_generatedRoot_948.w_25_0  (ctx : parallel_generatedRoot_948Ctx) : mxxMat0 :=
  ctx.witness.w_25_0

def parallel_generatedRoot_948.w_25_1  (ctx : parallel_generatedRoot_948Ctx) : mxxMat0 :=
  ctx.witness.w_25_1

def parallel_generatedRoot_948.w_25_2  (ctx : parallel_generatedRoot_948Ctx) : mxxMat0 :=
  ctx.witness.w_25_2

def parallel_generatedRoot_948.w_25_3  (ctx : parallel_generatedRoot_948Ctx) : mxxMat0 :=
  ctx.witness.w_25_3

def parallel_generatedRoot_948.w_25_4  (ctx : parallel_generatedRoot_948Ctx) : mxxMat0 :=
  ctx.witness.w_25_4

def parallel_generatedRoot_948.w_25_5  (ctx : parallel_generatedRoot_948Ctx) : mxxMat0 :=
  ctx.witness.w_25_5

def parallel_generatedRoot_948.outs  (ctx : parallel_generatedRoot_948Ctx) : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit :=
  ((parallel_generatedRoot_948.w_25_0 ctx), ((parallel_generatedRoot_948.w_25_1 ctx), ((parallel_generatedRoot_948.w_25_2 ctx), ((parallel_generatedRoot_948.w_25_3 ctx), ((parallel_generatedRoot_948.w_25_4 ctx), ((parallel_generatedRoot_948.w_25_5 ctx), ()))))))

structure parallel_generatedRoot_948.Facts_0  (ctx : parallel_generatedRoot_948Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  t0 : (2) = 2
  h6_5 : (∀ i : Fin 2, parallel_parallel_generatedRoot_948_6 backend hashModel tape (path ++ [6, i.val]) params i_0 i ((parallel_generatedRoot_948.w_0_0 ctx), ((parallel_generatedRoot_948.w_1_0 ctx), ((parallel_generatedRoot_948.w_2_0 ctx), ((parallel_generatedRoot_948.w_3_0 ctx), ((parallel_generatedRoot_948.w_4_0 ctx), ((parallel_generatedRoot_948.w_5_0 ctx), ())))))) (((parallel_generatedRoot_948.w_6_0 ctx) i), (((parallel_generatedRoot_948.w_6_1 ctx) i), (((parallel_generatedRoot_948.w_6_2 ctx) i), (((parallel_generatedRoot_948.w_6_3 ctx) i), (((parallel_generatedRoot_948.w_6_4 ctx) i), (((parallel_generatedRoot_948.w_6_5 ctx) i), ())))))))
  h7_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_948.w_6_0 ctx) (0) (parallel_generatedRoot_948.w_7_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_948.w_6_0 ctx) (1) (parallel_generatedRoot_948.w_8_0 ctx)
  h10_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_948.w_6_1 ctx) (0) (parallel_generatedRoot_948.w_10_0 ctx)
  h11_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_948.w_6_1 ctx) (1) (parallel_generatedRoot_948.w_11_0 ctx)
  h13_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_948.w_6_2 ctx) (0) (parallel_generatedRoot_948.w_13_0 ctx)
  h14_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_948.w_6_2 ctx) (1) (parallel_generatedRoot_948.w_14_0 ctx)
  h16_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_948.w_6_3 ctx) (0) (parallel_generatedRoot_948.w_16_0 ctx)
  h17_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_948.w_6_3 ctx) (1) (parallel_generatedRoot_948.w_17_0 ctx)
  h19_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_948.w_6_4 ctx) (0) (parallel_generatedRoot_948.w_19_0 ctx)
  h20_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_948.w_6_4 ctx) (1) (parallel_generatedRoot_948.w_20_0 ctx)
  h22_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_948.w_6_5 ctx) (0) (parallel_generatedRoot_948.w_22_0 ctx)
  h23_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_948.w_6_5 ctx) (1) (parallel_generatedRoot_948.w_23_0 ctx)
  t14 : 0 ≤ (0)
  h25_5 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_948_25 backend hashModel tape (path ++ [25, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, ((parallel_generatedRoot_948.w_6_0 ctx), ((parallel_generatedRoot_948.w_6_1 ctx), ((parallel_generatedRoot_948.w_6_2 ctx), ((parallel_generatedRoot_948.w_6_3 ctx), ((parallel_generatedRoot_948.w_6_4 ctx), ((parallel_generatedRoot_948.w_6_5 ctx), ())))))))))))) next) (Int.toNat (0)) ((parallel_generatedRoot_948.w_9_0 ctx), ((parallel_generatedRoot_948.w_12_0 ctx), ((parallel_generatedRoot_948.w_15_0 ctx), ((parallel_generatedRoot_948.w_18_0 ctx), ((parallel_generatedRoot_948.w_21_0 ctx), ((parallel_generatedRoot_948.w_24_0 ctx), ())))))) ((parallel_generatedRoot_948.w_25_0 ctx), ((parallel_generatedRoot_948.w_25_1 ctx), ((parallel_generatedRoot_948.w_25_2 ctx), ((parallel_generatedRoot_948.w_25_3 ctx), ((parallel_generatedRoot_948.w_25_4 ctx), ((parallel_generatedRoot_948.w_25_5 ctx), ()))))))
  hout : outputs = parallel_generatedRoot_948.outs ctx

theorem parallel_generatedRoot_948.facts_0  {ctx : parallel_generatedRoot_948Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_948.constraints_0 backend hashModel tape path params i_0 (parallel_generatedRoot_948.w_0_0 ctx) (parallel_generatedRoot_948.w_1_0 ctx) (parallel_generatedRoot_948.w_2_0 ctx) (parallel_generatedRoot_948.w_3_0 ctx) (parallel_generatedRoot_948.w_4_0 ctx) (parallel_generatedRoot_948.w_5_0 ctx) (parallel_generatedRoot_948.w_6_0 ctx) (parallel_generatedRoot_948.w_6_1 ctx) (parallel_generatedRoot_948.w_6_2 ctx) (parallel_generatedRoot_948.w_6_3 ctx) (parallel_generatedRoot_948.w_6_4 ctx) (parallel_generatedRoot_948.w_6_5 ctx) (parallel_generatedRoot_948.w_7_0 ctx) (parallel_generatedRoot_948.w_8_0 ctx) (parallel_generatedRoot_948.w_9_0 ctx) (parallel_generatedRoot_948.w_10_0 ctx) (parallel_generatedRoot_948.w_11_0 ctx) (parallel_generatedRoot_948.w_12_0 ctx) (parallel_generatedRoot_948.w_13_0 ctx) (parallel_generatedRoot_948.w_14_0 ctx) (parallel_generatedRoot_948.w_15_0 ctx) (parallel_generatedRoot_948.w_16_0 ctx) (parallel_generatedRoot_948.w_17_0 ctx) (parallel_generatedRoot_948.w_18_0 ctx) (parallel_generatedRoot_948.w_19_0 ctx) (parallel_generatedRoot_948.w_20_0 ctx) (parallel_generatedRoot_948.w_21_0 ctx) (parallel_generatedRoot_948.w_22_0 ctx) (parallel_generatedRoot_948.w_23_0 ctx) (parallel_generatedRoot_948.w_24_0 ctx) (parallel_generatedRoot_948.w_25_0 ctx) (parallel_generatedRoot_948.w_25_1 ctx) (parallel_generatedRoot_948.w_25_2 ctx) (parallel_generatedRoot_948.w_25_3 ctx) (parallel_generatedRoot_948.w_25_4 ctx) (parallel_generatedRoot_948.w_25_5 ctx) outputs) :
    parallel_generatedRoot_948.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_948.Facts  (ctx : parallel_generatedRoot_948Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  c0 : parallel_generatedRoot_948.Facts_0 ctx outputs

theorem parallel_generatedRoot_948.facts  (ctx : parallel_generatedRoot_948Ctx) {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_948.body backend hashModel tape path params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_948.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_948.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_949.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) where
  inputs : (parallel_generatedRoot_949.InputType)
  witness : parallel_generatedRoot_949.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_949Ctx" => (parallel_generatedRoot_949.Ctx backend hashModel tape path params i_0)

def parallel_generatedRoot_949.w_1_0  (ctx : parallel_generatedRoot_949Ctx) : Fin 8 → mxxMat0 :=
  ctx.inputs.1

def parallel_generatedRoot_949.w_3_0  (ctx : parallel_generatedRoot_949Ctx) : Fin 8 → mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_949.w_5_0  (ctx : parallel_generatedRoot_949Ctx) : Fin 8 → mxxMat0 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_949.w_7_0  (ctx : parallel_generatedRoot_949Ctx) : Fin 8 → mxxMat0 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_949.w_9_0  (ctx : parallel_generatedRoot_949Ctx) : Fin 8 → mxxMat0 :=
  ctx.inputs.2.2.2.2.1

def parallel_generatedRoot_949.w_10_0  (ctx : parallel_generatedRoot_949Ctx) : ByteArray :=
  ctx.inputs.2.2.2.2.2.1

def parallel_generatedRoot_949.w_11_0  (ctx : parallel_generatedRoot_949Ctx) : Fin 8 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.1

def parallel_generatedRoot_949.w_12_0  (ctx : parallel_generatedRoot_949Ctx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.1

def parallel_generatedRoot_949.w_13_0  (ctx : parallel_generatedRoot_949Ctx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_949.w_14_0  (ctx : parallel_generatedRoot_949Ctx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_949.w_15_0  (ctx : parallel_generatedRoot_949Ctx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_949.w_16_0  (ctx : parallel_generatedRoot_949Ctx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_949.w_17_0  (ctx : parallel_generatedRoot_949Ctx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_949.w_0_0  (ctx : parallel_generatedRoot_949Ctx) : mxxMat0 :=
  (0 : mxxMat0)

def parallel_generatedRoot_949.w_2_0  (ctx : parallel_generatedRoot_949Ctx) : mxxMat0 :=
  ctx.witness.w_2_0

def parallel_generatedRoot_949.w_4_0  (ctx : parallel_generatedRoot_949Ctx) : mxxMat0 :=
  ctx.witness.w_4_0

def parallel_generatedRoot_949.w_6_0  (ctx : parallel_generatedRoot_949Ctx) : mxxMat0 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_949.w_8_0  (ctx : parallel_generatedRoot_949Ctx) : mxxMat0 :=
  ctx.witness.w_8_0

def parallel_generatedRoot_949.w_18_0  (ctx : parallel_generatedRoot_949Ctx) : mxxMat0 :=
  ctx.witness.w_18_0

def parallel_generatedRoot_949.w_18_1  (ctx : parallel_generatedRoot_949Ctx) : mxxMat0 :=
  ctx.witness.w_18_1

def parallel_generatedRoot_949.w_18_2  (ctx : parallel_generatedRoot_949Ctx) : mxxMat0 :=
  ctx.witness.w_18_2

def parallel_generatedRoot_949.w_18_3  (ctx : parallel_generatedRoot_949Ctx) : mxxMat0 :=
  ctx.witness.w_18_3

def parallel_generatedRoot_949.w_18_4  (ctx : parallel_generatedRoot_949Ctx) : mxxMat0 :=
  ctx.witness.w_18_4

def parallel_generatedRoot_949.w_18_5  (ctx : parallel_generatedRoot_949Ctx) : mxxMat0 :=
  ctx.witness.w_18_5

def parallel_generatedRoot_949.w_18_6  (ctx : parallel_generatedRoot_949Ctx) : mxxMat0 :=
  ctx.witness.w_18_6

def parallel_generatedRoot_949.w_18_7  (ctx : parallel_generatedRoot_949Ctx) : mxxMat0 :=
  ctx.witness.w_18_7

def parallel_generatedRoot_949.w_18_8  (ctx : parallel_generatedRoot_949Ctx) : mxxMat0 :=
  ctx.witness.w_18_8

def parallel_generatedRoot_949.w_18_9  (ctx : parallel_generatedRoot_949Ctx) : mxxMat0 :=
  ctx.witness.w_18_9

def parallel_generatedRoot_949.w_19_0  (ctx : parallel_generatedRoot_949Ctx) : mxxMat0 :=
  ctx.witness.w_19_0

def parallel_generatedRoot_949.w_20_0  (ctx : parallel_generatedRoot_949Ctx) : Int :=
  (MxxIR.selectInt (7) [(1), (3), (5), (7), (9), (11), (13), (15)])

def parallel_generatedRoot_949.w_21_0  (ctx : parallel_generatedRoot_949Ctx) : mxxMat0 :=
  ctx.witness.w_21_0

def parallel_generatedRoot_949.w_22_0  (ctx : parallel_generatedRoot_949Ctx) : mxxMat0 :=
  ctx.witness.w_22_0

def parallel_generatedRoot_949.w_23_0  (ctx : parallel_generatedRoot_949Ctx) : mxxMat1 :=
  ctx.witness.w_23_0

def parallel_generatedRoot_949.w_23_concat_1  (ctx : parallel_generatedRoot_949Ctx) : mxxMat2 :=
  ctx.witness.w_23_concat_1

def parallel_generatedRoot_949.w_24_0  (ctx : parallel_generatedRoot_949Ctx) : mxxMat0 :=
  ctx.witness.w_24_0

def parallel_generatedRoot_949.w_25_0  (ctx : parallel_generatedRoot_949Ctx) : mxxMat0 :=
  ctx.witness.w_25_0

def parallel_generatedRoot_949.w_26_0  (ctx : parallel_generatedRoot_949Ctx) : mxxMat0 :=
  ctx.witness.w_26_0

def parallel_generatedRoot_949.w_27_0  (ctx : parallel_generatedRoot_949Ctx) : mxxMat1 :=
  ctx.witness.w_27_0

def parallel_generatedRoot_949.w_27_concat_1  (ctx : parallel_generatedRoot_949Ctx) : mxxMat2 :=
  ctx.witness.w_27_concat_1

def parallel_generatedRoot_949.w_28_0  (ctx : parallel_generatedRoot_949Ctx) : mxxMat3 :=
  ctx.witness.w_28_0

def parallel_generatedRoot_949.w_29_0  (ctx : parallel_generatedRoot_949Ctx) : mxxMat0 :=
  ctx.witness.w_29_0

def parallel_generatedRoot_949.w_30_0  (ctx : parallel_generatedRoot_949Ctx) : mxxMat0 :=
  ctx.witness.w_30_0

def parallel_generatedRoot_949.w_31_0  (ctx : parallel_generatedRoot_949Ctx) : mxxMat0 :=
  ctx.witness.w_31_0

def parallel_generatedRoot_949.w_32_0  (ctx : parallel_generatedRoot_949Ctx) : mxxMat1 :=
  ctx.witness.w_32_0

def parallel_generatedRoot_949.w_32_concat_1  (ctx : parallel_generatedRoot_949Ctx) : mxxMat2 :=
  ctx.witness.w_32_concat_1

def parallel_generatedRoot_949.w_33_0  (ctx : parallel_generatedRoot_949Ctx) : mxxMat0 :=
  ctx.witness.w_33_0

def parallel_generatedRoot_949.w_34_0  (ctx : parallel_generatedRoot_949Ctx) : mxxMat0 :=
  ctx.witness.w_34_0

def parallel_generatedRoot_949.w_35_0  (ctx : parallel_generatedRoot_949Ctx) : mxxMat0 :=
  ctx.witness.w_35_0

def parallel_generatedRoot_949.w_36_0  (ctx : parallel_generatedRoot_949Ctx) : mxxMat1 :=
  ctx.witness.w_36_0

def parallel_generatedRoot_949.w_36_concat_1  (ctx : parallel_generatedRoot_949Ctx) : mxxMat2 :=
  ctx.witness.w_36_concat_1

def parallel_generatedRoot_949.w_37_0  (ctx : parallel_generatedRoot_949Ctx) : mxxMat3 :=
  ctx.witness.w_37_0

def parallel_generatedRoot_949.w_38_0  (ctx : parallel_generatedRoot_949Ctx) : mxxMat2 :=
  ctx.witness.w_38_0

def parallel_generatedRoot_949.w_39_0  (ctx : parallel_generatedRoot_949Ctx) : mxxMat4 :=
  ctx.witness.w_39_0

def parallel_generatedRoot_949.w_40_0  (ctx : parallel_generatedRoot_949Ctx) : mxxMat4 :=
  ctx.witness.w_40_0

def parallel_generatedRoot_949.w_41_0  (ctx : parallel_generatedRoot_949Ctx) : mxxMat1 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_949.w_37_0 ctx) (parallel_generatedRoot_949.w_40_0 ctx)

def parallel_generatedRoot_949.w_42_0  (ctx : parallel_generatedRoot_949Ctx) : mxxMat0 :=
  ctx.witness.w_42_0

def parallel_generatedRoot_949.w_43_0  (ctx : parallel_generatedRoot_949Ctx) : mxxMat0 :=
  ctx.witness.w_43_0

def parallel_generatedRoot_949.w_44_0  (ctx : parallel_generatedRoot_949Ctx) : mxxMat2 :=
  ctx.witness.w_44_0

def parallel_generatedRoot_949.w_45_0  (ctx : parallel_generatedRoot_949Ctx) : mxxMat4 :=
  ctx.witness.w_45_0

def parallel_generatedRoot_949.w_46_0  (ctx : parallel_generatedRoot_949Ctx) : mxxMat4 :=
  ctx.witness.w_46_0

def parallel_generatedRoot_949.w_47_0  (ctx : parallel_generatedRoot_949Ctx) : mxxMat1 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_949.w_28_0 ctx) (parallel_generatedRoot_949.w_46_0 ctx)

def parallel_generatedRoot_949.w_48_0  (ctx : parallel_generatedRoot_949Ctx) : mxxMat0 :=
  ctx.witness.w_48_0

def parallel_generatedRoot_949.w_49_0  (ctx : parallel_generatedRoot_949Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_949.w_18_2 ctx) (parallel_generatedRoot_949.w_48_0 ctx)

def parallel_generatedRoot_949.w_50_0  (ctx : parallel_generatedRoot_949Ctx) : mxxMat2 :=
  ctx.witness.w_50_0

def parallel_generatedRoot_949.w_51_0  (ctx : parallel_generatedRoot_949Ctx) : mxxMat4 :=
  ctx.witness.w_51_0

def parallel_generatedRoot_949.w_52_0  (ctx : parallel_generatedRoot_949Ctx) : mxxMat4 :=
  ctx.witness.w_52_0

def parallel_generatedRoot_949.w_53_0  (ctx : parallel_generatedRoot_949Ctx) : mxxMat1 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_949.w_37_0 ctx) (parallel_generatedRoot_949.w_52_0 ctx)

def parallel_generatedRoot_949.w_54_0  (ctx : parallel_generatedRoot_949Ctx) : mxxMat0 :=
  ctx.witness.w_54_0

def parallel_generatedRoot_949.w_55_0  (ctx : parallel_generatedRoot_949Ctx) : mxxMat0 :=
  ctx.witness.w_55_0

def parallel_generatedRoot_949.w_56_0  (ctx : parallel_generatedRoot_949Ctx) : mxxMat2 :=
  ctx.witness.w_56_0

def parallel_generatedRoot_949.w_57_0  (ctx : parallel_generatedRoot_949Ctx) : mxxMat4 :=
  ctx.witness.w_57_0

def parallel_generatedRoot_949.w_58_0  (ctx : parallel_generatedRoot_949Ctx) : mxxMat4 :=
  ctx.witness.w_58_0

def parallel_generatedRoot_949.w_59_0  (ctx : parallel_generatedRoot_949Ctx) : mxxMat1 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_949.w_28_0 ctx) (parallel_generatedRoot_949.w_58_0 ctx)

def parallel_generatedRoot_949.w_60_0  (ctx : parallel_generatedRoot_949Ctx) : mxxMat0 :=
  ctx.witness.w_60_0

def parallel_generatedRoot_949.w_61_0  (ctx : parallel_generatedRoot_949Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_949.w_18_3 ctx) (parallel_generatedRoot_949.w_60_0 ctx)

def parallel_generatedRoot_949.w_62_0  (ctx : parallel_generatedRoot_949Ctx) : mxxMat0 :=
  ctx.witness.w_62_0

def parallel_generatedRoot_949.w_63_0  (ctx : parallel_generatedRoot_949Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_949.w_18_4 ctx) (parallel_generatedRoot_949.w_62_0 ctx)

def parallel_generatedRoot_949.w_64_0  (ctx : parallel_generatedRoot_949Ctx) : mxxMat0 :=
  ctx.witness.w_64_0

def parallel_generatedRoot_949.w_65_0  (ctx : parallel_generatedRoot_949Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_949.w_18_5 ctx) (parallel_generatedRoot_949.w_64_0 ctx)

def parallel_generatedRoot_949.w_66_0  (ctx : parallel_generatedRoot_949Ctx) : mxxMat0 :=
  ctx.witness.w_66_0

def parallel_generatedRoot_949.w_67_0  (ctx : parallel_generatedRoot_949Ctx) : mxxMat0 :=
  ctx.witness.w_67_0

def parallel_generatedRoot_949.w_68_0  (ctx : parallel_generatedRoot_949Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_949.w_66_0 ctx) (parallel_generatedRoot_949.w_67_0 ctx)

def parallel_generatedRoot_949.w_69_0  (ctx : parallel_generatedRoot_949Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_949.w_18_0 ctx) (parallel_generatedRoot_949.w_68_0 ctx)

def parallel_generatedRoot_949.w_70_0  (ctx : parallel_generatedRoot_949Ctx) : mxxMat0 :=
  ctx.witness.w_70_0

def parallel_generatedRoot_949.w_71_0  (ctx : parallel_generatedRoot_949Ctx) : mxxMat0 :=
  ctx.witness.w_71_0

def parallel_generatedRoot_949.w_72_0  (ctx : parallel_generatedRoot_949Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_949.w_70_0 ctx) (parallel_generatedRoot_949.w_71_0 ctx)

def parallel_generatedRoot_949.w_73_0  (ctx : parallel_generatedRoot_949Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_949.w_18_1 ctx) (parallel_generatedRoot_949.w_72_0 ctx)

def parallel_generatedRoot_949.outs  (ctx : parallel_generatedRoot_949Ctx) : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit :=
  ((parallel_generatedRoot_949.w_49_0 ctx), ((parallel_generatedRoot_949.w_61_0 ctx), ((parallel_generatedRoot_949.w_63_0 ctx), ((parallel_generatedRoot_949.w_65_0 ctx), ((parallel_generatedRoot_949.w_69_0 ctx), ((parallel_generatedRoot_949.w_73_0 ctx), ()))))))

structure parallel_generatedRoot_949.Facts_0  (ctx : parallel_generatedRoot_949Ctx) : Prop where
  h2_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_949.w_1_0 ctx) (0) (parallel_generatedRoot_949.w_2_0 ctx)
  h4_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_949.w_3_0 ctx) (0) (parallel_generatedRoot_949.w_4_0 ctx)
  h6_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_949.w_5_0 ctx) (0) (parallel_generatedRoot_949.w_6_0 ctx)
  h8_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_949.w_7_0 ctx) (0) (parallel_generatedRoot_949.w_8_0 ctx)
  t4 : 0 ≤ (7)
  h18_9 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_949_18 backend hashModel tape (path ++ [18, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (current.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.2.1, ((parallel_generatedRoot_949.w_9_0 ctx), ((parallel_generatedRoot_949.w_10_0 ctx), ((parallel_generatedRoot_949.w_11_0 ctx), ((parallel_generatedRoot_949.w_12_0 ctx), ((parallel_generatedRoot_949.w_13_0 ctx), ((parallel_generatedRoot_949.w_14_0 ctx), ((parallel_generatedRoot_949.w_15_0 ctx), ((parallel_generatedRoot_949.w_16_0 ctx), ((parallel_generatedRoot_949.w_17_0 ctx), ((parallel_generatedRoot_949.w_1_0 ctx), ((parallel_generatedRoot_949.w_3_0 ctx), ((parallel_generatedRoot_949.w_5_0 ctx), ((parallel_generatedRoot_949.w_7_0 ctx), ()))))))))))))))))))))))) next) (Int.toNat (7)) ((parallel_generatedRoot_949.w_0_0 ctx), ((parallel_generatedRoot_949.w_0_0 ctx), ((parallel_generatedRoot_949.w_0_0 ctx), ((parallel_generatedRoot_949.w_0_0 ctx), ((parallel_generatedRoot_949.w_0_0 ctx), ((parallel_generatedRoot_949.w_0_0 ctx), ((parallel_generatedRoot_949.w_2_0 ctx), ((parallel_generatedRoot_949.w_4_0 ctx), ((parallel_generatedRoot_949.w_6_0 ctx), ((parallel_generatedRoot_949.w_8_0 ctx), ())))))))))) ((parallel_generatedRoot_949.w_18_0 ctx), ((parallel_generatedRoot_949.w_18_1 ctx), ((parallel_generatedRoot_949.w_18_2 ctx), ((parallel_generatedRoot_949.w_18_3 ctx), ((parallel_generatedRoot_949.w_18_4 ctx), ((parallel_generatedRoot_949.w_18_5 ctx), ((parallel_generatedRoot_949.w_18_6 ctx), ((parallel_generatedRoot_949.w_18_7 ctx), ((parallel_generatedRoot_949.w_18_8 ctx), ((parallel_generatedRoot_949.w_18_9 ctx), ()))))))))))
  h19_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_949.w_9_0 ctx) (7) (parallel_generatedRoot_949.w_19_0 ctx)
  t7 : 0 ≤ (7)
  t8 : (7) < 8
  h21_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (parallel_generatedRoot_949.w_20_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((parallel_generatedRoot_949.w_10_0 ctx)) (parallel_generatedRoot_949.w_21_0 ctx)
  h22_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (parallel_generatedRoot_949.w_20_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) ((parallel_generatedRoot_949.w_10_0 ctx)) (parallel_generatedRoot_949.w_22_0 ctx)
  h21_0_1 : MxxRuntime.concatRows (parallel_generatedRoot_949.w_19_0 ctx) (parallel_generatedRoot_949.w_21_0 ctx) (parallel_generatedRoot_949.w_23_concat_1 ctx)
  h23_0 : MxxRuntime.concatRows (parallel_generatedRoot_949.w_23_concat_1 ctx) (parallel_generatedRoot_949.w_22_0 ctx) (parallel_generatedRoot_949.w_23_0 ctx)
  h24_0 : MxxRuntime.familyGetStatic (parallel_generatedRoot_949.w_11_0 ctx) (7) (parallel_generatedRoot_949.w_24_0 ctx)
  h25_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (parallel_generatedRoot_949.w_20_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((parallel_generatedRoot_949.w_10_0 ctx)) (parallel_generatedRoot_949.w_25_0 ctx)
  h26_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (parallel_generatedRoot_949.w_20_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) ((parallel_generatedRoot_949.w_10_0 ctx)) (parallel_generatedRoot_949.w_26_0 ctx)
  h25_0_1 : MxxRuntime.concatRows (parallel_generatedRoot_949.w_24_0 ctx) (parallel_generatedRoot_949.w_25_0 ctx) (parallel_generatedRoot_949.w_27_concat_1 ctx)
  h27_0 : MxxRuntime.concatRows (parallel_generatedRoot_949.w_27_concat_1 ctx) (parallel_generatedRoot_949.w_26_0 ctx) (parallel_generatedRoot_949.w_27_0 ctx)
  h28_0 : MxxRuntime.concatColumns (parallel_generatedRoot_949.w_23_0 ctx) (parallel_generatedRoot_949.w_27_0 ctx) (parallel_generatedRoot_949.w_28_0 ctx)
  t19 : 0 ≤ (7)
  t20 : (7) < 8
  h29_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (7) [(1), (3), (5), (7), (9), (11), (13), (15)])) (parallel_generatedRoot_949.w_12_0 ctx) (parallel_generatedRoot_949.w_29_0 ctx)
  t22 : 0 ≤ (7)
  t23 : (7) < 8
  h30_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (7) [(1), (3), (5), (7), (9), (11), (13), (15)])) (parallel_generatedRoot_949.w_13_0 ctx) (parallel_generatedRoot_949.w_30_0 ctx)
  t25 : 0 ≤ (7)
  t26 : (7) < 8
  h31_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (7) [(1), (3), (5), (7), (9), (11), (13), (15)])) (parallel_generatedRoot_949.w_14_0 ctx) (parallel_generatedRoot_949.w_31_0 ctx)
  h30_0_1 : MxxRuntime.concatRows (parallel_generatedRoot_949.w_29_0 ctx) (parallel_generatedRoot_949.w_30_0 ctx) (parallel_generatedRoot_949.w_32_concat_1 ctx)
  h32_0 : MxxRuntime.concatRows (parallel_generatedRoot_949.w_32_concat_1 ctx) (parallel_generatedRoot_949.w_31_0 ctx) (parallel_generatedRoot_949.w_32_0 ctx)
  t30 : 0 ≤ (7)
  t31 : (7) < 8

theorem parallel_generatedRoot_949.facts_0  {ctx : parallel_generatedRoot_949Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_949.constraints_0 backend hashModel tape path params i_0 (parallel_generatedRoot_949.w_1_0 ctx) (parallel_generatedRoot_949.w_3_0 ctx) (parallel_generatedRoot_949.w_5_0 ctx) (parallel_generatedRoot_949.w_7_0 ctx) (parallel_generatedRoot_949.w_9_0 ctx) (parallel_generatedRoot_949.w_10_0 ctx) (parallel_generatedRoot_949.w_11_0 ctx) (parallel_generatedRoot_949.w_12_0 ctx) (parallel_generatedRoot_949.w_13_0 ctx) (parallel_generatedRoot_949.w_14_0 ctx) (parallel_generatedRoot_949.w_15_0 ctx) (parallel_generatedRoot_949.w_16_0 ctx) (parallel_generatedRoot_949.w_17_0 ctx) (parallel_generatedRoot_949.w_0_0 ctx) (parallel_generatedRoot_949.w_2_0 ctx) (parallel_generatedRoot_949.w_4_0 ctx) (parallel_generatedRoot_949.w_6_0 ctx) (parallel_generatedRoot_949.w_8_0 ctx) (parallel_generatedRoot_949.w_18_0 ctx) (parallel_generatedRoot_949.w_18_1 ctx) (parallel_generatedRoot_949.w_18_2 ctx) (parallel_generatedRoot_949.w_18_3 ctx) (parallel_generatedRoot_949.w_18_4 ctx) (parallel_generatedRoot_949.w_18_5 ctx) (parallel_generatedRoot_949.w_18_6 ctx) (parallel_generatedRoot_949.w_18_7 ctx) (parallel_generatedRoot_949.w_18_8 ctx) (parallel_generatedRoot_949.w_18_9 ctx) (parallel_generatedRoot_949.w_19_0 ctx) (parallel_generatedRoot_949.w_20_0 ctx) (parallel_generatedRoot_949.w_21_0 ctx) (parallel_generatedRoot_949.w_22_0 ctx) (parallel_generatedRoot_949.w_23_0 ctx) (parallel_generatedRoot_949.w_23_concat_1 ctx) (parallel_generatedRoot_949.w_24_0 ctx) (parallel_generatedRoot_949.w_25_0 ctx) (parallel_generatedRoot_949.w_26_0 ctx) (parallel_generatedRoot_949.w_27_0 ctx) (parallel_generatedRoot_949.w_27_concat_1 ctx) (parallel_generatedRoot_949.w_28_0 ctx) (parallel_generatedRoot_949.w_29_0 ctx) (parallel_generatedRoot_949.w_30_0 ctx) (parallel_generatedRoot_949.w_31_0 ctx) (parallel_generatedRoot_949.w_32_0 ctx) (parallel_generatedRoot_949.w_32_concat_1 ctx)) :
    parallel_generatedRoot_949.Facts_0 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_949.Facts_1  (ctx : parallel_generatedRoot_949Ctx) : Prop where
  h33_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (7) [(1), (3), (5), (7), (9), (11), (13), (15)])) (parallel_generatedRoot_949.w_15_0 ctx) (parallel_generatedRoot_949.w_33_0 ctx)
  t33 : 0 ≤ (7)
  t34 : (7) < 8
  h34_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (7) [(1), (3), (5), (7), (9), (11), (13), (15)])) (parallel_generatedRoot_949.w_16_0 ctx) (parallel_generatedRoot_949.w_34_0 ctx)
  t36 : 0 ≤ (7)
  t37 : (7) < 8
  h35_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (7) [(1), (3), (5), (7), (9), (11), (13), (15)])) (parallel_generatedRoot_949.w_17_0 ctx) (parallel_generatedRoot_949.w_35_0 ctx)
  h34_0_1 : MxxRuntime.concatRows (parallel_generatedRoot_949.w_33_0 ctx) (parallel_generatedRoot_949.w_34_0 ctx) (parallel_generatedRoot_949.w_36_concat_1 ctx)
  h36_0 : MxxRuntime.concatRows (parallel_generatedRoot_949.w_36_concat_1 ctx) (parallel_generatedRoot_949.w_35_0 ctx) (parallel_generatedRoot_949.w_36_0 ctx)
  h37_0 : MxxRuntime.concatColumns (parallel_generatedRoot_949.w_32_0 ctx) (parallel_generatedRoot_949.w_36_0 ctx) (parallel_generatedRoot_949.w_37_0 ctx)
  h38_0 : MxxRuntime.concatRows (parallel_generatedRoot_949.w_18_6 ctx) (parallel_generatedRoot_949.w_18_8 ctx) (parallel_generatedRoot_949.w_38_0 ctx)
  h39_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (parallel_generatedRoot_949.w_38_0 ctx) (parallel_generatedRoot_949.w_39_0 ctx)
  h40_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (parallel_generatedRoot_949.w_39_0 ctx) (parallel_generatedRoot_949.w_40_0 ctx)
  t45 : 0 ≤ 1
  t46 : 1 < 2
  t47 : 2 ≤ 3
  t48 : 0 ≤ 0
  t49 : 0 < 25
  t50 : 25 ≤ 25
  h42_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_949.w_41_0 ctx) 1 2 0 25 (parallel_generatedRoot_949.w_42_0 ctx)
  t52 : 0 ≤ 2
  t53 : 2 < 3
  t54 : 3 ≤ 3
  t55 : 0 ≤ 0
  t56 : 0 < 25
  t57 : 25 ≤ 25
  h43_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_949.w_41_0 ctx) 2 3 0 25 (parallel_generatedRoot_949.w_43_0 ctx)
  h44_0 : MxxRuntime.concatRows (parallel_generatedRoot_949.w_42_0 ctx) (parallel_generatedRoot_949.w_43_0 ctx) (parallel_generatedRoot_949.w_44_0 ctx)
  h45_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (parallel_generatedRoot_949.w_44_0 ctx) (parallel_generatedRoot_949.w_45_0 ctx)
  h46_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (parallel_generatedRoot_949.w_45_0 ctx) (parallel_generatedRoot_949.w_46_0 ctx)
  t62 : 0 ≤ 1
  t63 : 1 < 2

theorem parallel_generatedRoot_949.facts_1  {ctx : parallel_generatedRoot_949Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_949.constraints_1 backend (parallel_generatedRoot_949.w_15_0 ctx) (parallel_generatedRoot_949.w_16_0 ctx) (parallel_generatedRoot_949.w_17_0 ctx) (parallel_generatedRoot_949.w_18_6 ctx) (parallel_generatedRoot_949.w_18_8 ctx) (parallel_generatedRoot_949.w_32_0 ctx) (parallel_generatedRoot_949.w_33_0 ctx) (parallel_generatedRoot_949.w_34_0 ctx) (parallel_generatedRoot_949.w_35_0 ctx) (parallel_generatedRoot_949.w_36_0 ctx) (parallel_generatedRoot_949.w_36_concat_1 ctx) (parallel_generatedRoot_949.w_37_0 ctx) (parallel_generatedRoot_949.w_38_0 ctx) (parallel_generatedRoot_949.w_39_0 ctx) (parallel_generatedRoot_949.w_40_0 ctx) (parallel_generatedRoot_949.w_41_0 ctx) (parallel_generatedRoot_949.w_42_0 ctx) (parallel_generatedRoot_949.w_43_0 ctx) (parallel_generatedRoot_949.w_44_0 ctx) (parallel_generatedRoot_949.w_45_0 ctx) (parallel_generatedRoot_949.w_46_0 ctx)) :
    parallel_generatedRoot_949.Facts_1 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_949.Facts_2  (ctx : parallel_generatedRoot_949Ctx) : Prop where
  t64 : 2 ≤ 3
  t65 : 0 ≤ 0
  t66 : 0 < 25
  t67 : 25 ≤ 25
  h48_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_949.w_47_0 ctx) 1 2 0 25 (parallel_generatedRoot_949.w_48_0 ctx)
  h50_0 : MxxRuntime.concatRows (parallel_generatedRoot_949.w_18_7 ctx) (parallel_generatedRoot_949.w_18_9 ctx) (parallel_generatedRoot_949.w_50_0 ctx)
  h51_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (parallel_generatedRoot_949.w_50_0 ctx) (parallel_generatedRoot_949.w_51_0 ctx)
  h52_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (parallel_generatedRoot_949.w_51_0 ctx) (parallel_generatedRoot_949.w_52_0 ctx)
  t72 : 0 ≤ 1
  t73 : 1 < 2
  t74 : 2 ≤ 3
  t75 : 0 ≤ 0
  t76 : 0 < 25
  t77 : 25 ≤ 25
  h54_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_949.w_53_0 ctx) 1 2 0 25 (parallel_generatedRoot_949.w_54_0 ctx)
  t79 : 0 ≤ 2
  t80 : 2 < 3
  t81 : 3 ≤ 3
  t82 : 0 ≤ 0
  t83 : 0 < 25
  t84 : 25 ≤ 25
  h55_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_949.w_53_0 ctx) 2 3 0 25 (parallel_generatedRoot_949.w_55_0 ctx)
  h56_0 : MxxRuntime.concatRows (parallel_generatedRoot_949.w_54_0 ctx) (parallel_generatedRoot_949.w_55_0 ctx) (parallel_generatedRoot_949.w_56_0 ctx)
  h57_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (parallel_generatedRoot_949.w_56_0 ctx) (parallel_generatedRoot_949.w_57_0 ctx)
  h58_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (parallel_generatedRoot_949.w_57_0 ctx) (parallel_generatedRoot_949.w_58_0 ctx)
  t89 : 0 ≤ 1
  t90 : 1 < 2
  t91 : 2 ≤ 3
  t92 : 0 ≤ 0
  t93 : 0 < 25
  t94 : 25 ≤ 25
  h60_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_949.w_59_0 ctx) 1 2 0 25 (parallel_generatedRoot_949.w_60_0 ctx)

theorem parallel_generatedRoot_949.facts_2  {ctx : parallel_generatedRoot_949Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_949.constraints_2 backend (parallel_generatedRoot_949.w_18_7 ctx) (parallel_generatedRoot_949.w_18_9 ctx) (parallel_generatedRoot_949.w_47_0 ctx) (parallel_generatedRoot_949.w_48_0 ctx) (parallel_generatedRoot_949.w_50_0 ctx) (parallel_generatedRoot_949.w_51_0 ctx) (parallel_generatedRoot_949.w_52_0 ctx) (parallel_generatedRoot_949.w_53_0 ctx) (parallel_generatedRoot_949.w_54_0 ctx) (parallel_generatedRoot_949.w_55_0 ctx) (parallel_generatedRoot_949.w_56_0 ctx) (parallel_generatedRoot_949.w_57_0 ctx) (parallel_generatedRoot_949.w_58_0 ctx) (parallel_generatedRoot_949.w_59_0 ctx) (parallel_generatedRoot_949.w_60_0 ctx)) :
    parallel_generatedRoot_949.Facts_2 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_949.Facts_3  (ctx : parallel_generatedRoot_949Ctx) : Prop where
  t96 : 0 ≤ 2
  t97 : 2 < 3
  t98 : 3 ≤ 3
  t99 : 0 ≤ 0
  t100 : 0 < 25
  t101 : 25 ≤ 25
  h62_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_949.w_47_0 ctx) 2 3 0 25 (parallel_generatedRoot_949.w_62_0 ctx)
  t103 : 0 ≤ 2
  t104 : 2 < 3
  t105 : 3 ≤ 3
  t106 : 0 ≤ 0
  t107 : 0 < 25
  t108 : 25 ≤ 25
  h64_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_949.w_59_0 ctx) 2 3 0 25 (parallel_generatedRoot_949.w_64_0 ctx)
  t110 : 0 ≤ 0
  t111 : 0 < 1
  t112 : 1 ≤ 3
  t113 : 0 ≤ 0
  t114 : 0 < 25
  t115 : 25 ≤ 25
  h66_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_949.w_47_0 ctx) 0 1 0 25 (parallel_generatedRoot_949.w_66_0 ctx)
  t117 : 0 ≤ 0
  t118 : 0 < 1
  t119 : 1 ≤ 3
  t120 : 0 ≤ 0
  t121 : 0 < 25
  t122 : 25 ≤ 25
  h67_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_949.w_41_0 ctx) 0 1 0 25 (parallel_generatedRoot_949.w_67_0 ctx)
  t124 : 0 ≤ 0
  t125 : 0 < 1
  t126 : 1 ≤ 3
  t127 : 0 ≤ 0

theorem parallel_generatedRoot_949.facts_3  {ctx : parallel_generatedRoot_949Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_949.constraints_3 (parallel_generatedRoot_949.w_41_0 ctx) (parallel_generatedRoot_949.w_47_0 ctx) (parallel_generatedRoot_949.w_59_0 ctx) (parallel_generatedRoot_949.w_62_0 ctx) (parallel_generatedRoot_949.w_64_0 ctx) (parallel_generatedRoot_949.w_66_0 ctx) (parallel_generatedRoot_949.w_67_0 ctx)) :
    parallel_generatedRoot_949.Facts_3 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_949.Facts_4  (ctx : parallel_generatedRoot_949Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  t128 : 0 < 25
  t129 : 25 ≤ 25
  h70_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_949.w_59_0 ctx) 0 1 0 25 (parallel_generatedRoot_949.w_70_0 ctx)
  t131 : 0 ≤ 0
  t132 : 0 < 1
  t133 : 1 ≤ 3
  t134 : 0 ≤ 0
  t135 : 0 < 25
  t136 : 25 ≤ 25
  h71_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_949.w_53_0 ctx) 0 1 0 25 (parallel_generatedRoot_949.w_71_0 ctx)
  hout : outputs = parallel_generatedRoot_949.outs ctx

theorem parallel_generatedRoot_949.facts_4  {ctx : parallel_generatedRoot_949Ctx} {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_949.constraints_4 (parallel_generatedRoot_949.w_49_0 ctx) (parallel_generatedRoot_949.w_53_0 ctx) (parallel_generatedRoot_949.w_59_0 ctx) (parallel_generatedRoot_949.w_61_0 ctx) (parallel_generatedRoot_949.w_63_0 ctx) (parallel_generatedRoot_949.w_65_0 ctx) (parallel_generatedRoot_949.w_69_0 ctx) (parallel_generatedRoot_949.w_70_0 ctx) (parallel_generatedRoot_949.w_71_0 ctx) (parallel_generatedRoot_949.w_73_0 ctx) outputs) :
    parallel_generatedRoot_949.Facts_4 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_949.Facts  (ctx : parallel_generatedRoot_949Ctx) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop where
  c0 : parallel_generatedRoot_949.Facts_0 ctx
  c1 : parallel_generatedRoot_949.Facts_1 ctx
  c2 : parallel_generatedRoot_949.Facts_2 ctx
  c3 : parallel_generatedRoot_949.Facts_3 ctx
  c4 : parallel_generatedRoot_949.Facts_4 ctx outputs

theorem parallel_generatedRoot_949.facts  (ctx : parallel_generatedRoot_949Ctx) {outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit}
    (h : parallel_generatedRoot_949.body backend hashModel tape path params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_949.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_949.facts_0 (ctx := ctx) (outputs := outputs) t0.1
  have t1 := t0.2
  have r1 := parallel_generatedRoot_949.facts_1 (ctx := ctx) (outputs := outputs) t1.1
  have t2 := t1.2
  have r2 := parallel_generatedRoot_949.facts_2 (ctx := ctx) (outputs := outputs) t2.1
  have t3 := t2.2
  have r3 := parallel_generatedRoot_949.facts_3 (ctx := ctx) (outputs := outputs) t3.1
  have t4 := t3.2
  have r4 := parallel_generatedRoot_949.facts_4 (ctx := ctx) (outputs := outputs) t4
  ⟨r0, r1, r2, r3, r4⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_981.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) where
  inputs : (sequential_generatedRoot_981.InputType)
  witness : sequential_generatedRoot_981.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_981Ctx" => (sequential_generatedRoot_981.Ctx backend hashModel tape path params i_0)

def sequential_generatedRoot_981.w_0_0  (ctx : sequential_generatedRoot_981Ctx) : mxxMat9 :=
  ctx.inputs.1

def sequential_generatedRoot_981.w_31_0  (ctx : sequential_generatedRoot_981Ctx) : mxxMat9 :=
  ctx.inputs.2.1

def sequential_generatedRoot_981.w_35_0  (ctx : sequential_generatedRoot_981Ctx) : mxxMat9 :=
  ctx.inputs.2.2.1

def sequential_generatedRoot_981.w_39_0  (ctx : sequential_generatedRoot_981Ctx) : mxxMat9 :=
  ctx.inputs.2.2.2.1

def sequential_generatedRoot_981.w_43_0  (ctx : sequential_generatedRoot_981Ctx) : mxxMat9 :=
  ctx.inputs.2.2.2.2.1

def sequential_generatedRoot_981.w_47_0  (ctx : sequential_generatedRoot_981Ctx) : mxxMat9 :=
  ctx.inputs.2.2.2.2.2.1

def sequential_generatedRoot_981.w_2_0  (ctx : sequential_generatedRoot_981Ctx) : Fin 36000 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.1

def sequential_generatedRoot_981.w_8_0  (ctx : sequential_generatedRoot_981Ctx) : Fin 36000 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.1

def sequential_generatedRoot_981.w_12_0  (ctx : sequential_generatedRoot_981Ctx) : Fin 36000 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.1

def sequential_generatedRoot_981.w_16_0  (ctx : sequential_generatedRoot_981Ctx) : Fin 36000 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.1

def sequential_generatedRoot_981.w_20_0  (ctx : sequential_generatedRoot_981Ctx) : Fin 36000 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.1

def sequential_generatedRoot_981.w_24_0  (ctx : sequential_generatedRoot_981Ctx) : Fin 36000 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.1

def sequential_generatedRoot_981.w_1_0  (ctx : sequential_generatedRoot_981Ctx) : mxxMat10 :=
  (MxxRuntime.unitRow ((1 + (Int.ofNat i_0))) : mxxMat10)

def sequential_generatedRoot_981.w_3_0  (ctx : sequential_generatedRoot_981Ctx) : Int :=
  (8 + (8 * (Int.ofNat i_0)))

def sequential_generatedRoot_981.w_4_0  (ctx : sequential_generatedRoot_981Ctx) : mxxMat0 :=
  ctx.witness.w_4_0

def sequential_generatedRoot_981.w_5_0  (ctx : sequential_generatedRoot_981Ctx) : Int :=
  (9 + (8 * (Int.ofNat i_0)))

def sequential_generatedRoot_981.w_6_0  (ctx : sequential_generatedRoot_981Ctx) : mxxMat0 :=
  ctx.witness.w_6_0

def sequential_generatedRoot_981.w_7_0  (ctx : sequential_generatedRoot_981Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (sequential_generatedRoot_981.w_4_0 ctx) (sequential_generatedRoot_981.w_6_0 ctx)

def sequential_generatedRoot_981.w_9_0  (ctx : sequential_generatedRoot_981Ctx) : mxxMat0 :=
  ctx.witness.w_9_0

def sequential_generatedRoot_981.w_10_0  (ctx : sequential_generatedRoot_981Ctx) : mxxMat0 :=
  ctx.witness.w_10_0

def sequential_generatedRoot_981.w_11_0  (ctx : sequential_generatedRoot_981Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (sequential_generatedRoot_981.w_9_0 ctx) (sequential_generatedRoot_981.w_10_0 ctx)

def sequential_generatedRoot_981.w_13_0  (ctx : sequential_generatedRoot_981Ctx) : mxxMat0 :=
  ctx.witness.w_13_0

def sequential_generatedRoot_981.w_14_0  (ctx : sequential_generatedRoot_981Ctx) : mxxMat0 :=
  ctx.witness.w_14_0

def sequential_generatedRoot_981.w_15_0  (ctx : sequential_generatedRoot_981Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (sequential_generatedRoot_981.w_13_0 ctx) (sequential_generatedRoot_981.w_14_0 ctx)

def sequential_generatedRoot_981.w_17_0  (ctx : sequential_generatedRoot_981Ctx) : mxxMat0 :=
  ctx.witness.w_17_0

def sequential_generatedRoot_981.w_18_0  (ctx : sequential_generatedRoot_981Ctx) : mxxMat0 :=
  ctx.witness.w_18_0

def sequential_generatedRoot_981.w_19_0  (ctx : sequential_generatedRoot_981Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (sequential_generatedRoot_981.w_17_0 ctx) (sequential_generatedRoot_981.w_18_0 ctx)

def sequential_generatedRoot_981.w_21_0  (ctx : sequential_generatedRoot_981Ctx) : mxxMat0 :=
  ctx.witness.w_21_0

def sequential_generatedRoot_981.w_22_0  (ctx : sequential_generatedRoot_981Ctx) : mxxMat0 :=
  ctx.witness.w_22_0

def sequential_generatedRoot_981.w_23_0  (ctx : sequential_generatedRoot_981Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (sequential_generatedRoot_981.w_21_0 ctx) (sequential_generatedRoot_981.w_22_0 ctx)

def sequential_generatedRoot_981.w_25_0  (ctx : sequential_generatedRoot_981Ctx) : mxxMat0 :=
  ctx.witness.w_25_0

def sequential_generatedRoot_981.w_26_0  (ctx : sequential_generatedRoot_981Ctx) : mxxMat0 :=
  ctx.witness.w_26_0

def sequential_generatedRoot_981.w_27_0  (ctx : sequential_generatedRoot_981Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (sequential_generatedRoot_981.w_25_0 ctx) (sequential_generatedRoot_981.w_26_0 ctx)

def sequential_generatedRoot_981.w_28_0  (ctx : sequential_generatedRoot_981Ctx) : mxxMat0 :=
  ctx.witness.w_28_0

def sequential_generatedRoot_981.w_28_1  (ctx : sequential_generatedRoot_981Ctx) : mxxMat0 :=
  ctx.witness.w_28_1

def sequential_generatedRoot_981.w_28_2  (ctx : sequential_generatedRoot_981Ctx) : mxxMat0 :=
  ctx.witness.w_28_2

def sequential_generatedRoot_981.w_28_3  (ctx : sequential_generatedRoot_981Ctx) : mxxMat0 :=
  ctx.witness.w_28_3

def sequential_generatedRoot_981.w_28_4  (ctx : sequential_generatedRoot_981Ctx) : mxxMat0 :=
  ctx.witness.w_28_4

def sequential_generatedRoot_981.w_28_5  (ctx : sequential_generatedRoot_981Ctx) : mxxMat0 :=
  ctx.witness.w_28_5

def sequential_generatedRoot_981.w_29_0  (ctx : sequential_generatedRoot_981Ctx) : mxxMat9 :=
  ctx.witness.w_29_0

def sequential_generatedRoot_981.w_30_0  (ctx : sequential_generatedRoot_981Ctx) : mxxMat9 :=
  MxxRuntime.matrixAdd (sequential_generatedRoot_981.w_0_0 ctx) (sequential_generatedRoot_981.w_29_0 ctx)

def sequential_generatedRoot_981.w_32_0  (ctx : sequential_generatedRoot_981Ctx) : mxxMat10 :=
  (MxxRuntime.unitRow ((1 + (Int.ofNat i_0))) : mxxMat10)

def sequential_generatedRoot_981.w_33_0  (ctx : sequential_generatedRoot_981Ctx) : mxxMat9 :=
  ctx.witness.w_33_0

def sequential_generatedRoot_981.w_34_0  (ctx : sequential_generatedRoot_981Ctx) : mxxMat9 :=
  MxxRuntime.matrixAdd (sequential_generatedRoot_981.w_31_0 ctx) (sequential_generatedRoot_981.w_33_0 ctx)

def sequential_generatedRoot_981.w_36_0  (ctx : sequential_generatedRoot_981Ctx) : mxxMat10 :=
  (MxxRuntime.unitRow ((1 + (Int.ofNat i_0))) : mxxMat10)

def sequential_generatedRoot_981.w_37_0  (ctx : sequential_generatedRoot_981Ctx) : mxxMat9 :=
  ctx.witness.w_37_0

def sequential_generatedRoot_981.w_38_0  (ctx : sequential_generatedRoot_981Ctx) : mxxMat9 :=
  MxxRuntime.matrixAdd (sequential_generatedRoot_981.w_35_0 ctx) (sequential_generatedRoot_981.w_37_0 ctx)

def sequential_generatedRoot_981.w_40_0  (ctx : sequential_generatedRoot_981Ctx) : mxxMat10 :=
  (MxxRuntime.unitRow ((1 + (Int.ofNat i_0))) : mxxMat10)

def sequential_generatedRoot_981.w_41_0  (ctx : sequential_generatedRoot_981Ctx) : mxxMat9 :=
  ctx.witness.w_41_0

def sequential_generatedRoot_981.w_42_0  (ctx : sequential_generatedRoot_981Ctx) : mxxMat9 :=
  MxxRuntime.matrixAdd (sequential_generatedRoot_981.w_39_0 ctx) (sequential_generatedRoot_981.w_41_0 ctx)

def sequential_generatedRoot_981.w_44_0  (ctx : sequential_generatedRoot_981Ctx) : mxxMat10 :=
  (MxxRuntime.unitRow ((1 + (Int.ofNat i_0))) : mxxMat10)

def sequential_generatedRoot_981.w_45_0  (ctx : sequential_generatedRoot_981Ctx) : mxxMat9 :=
  ctx.witness.w_45_0

def sequential_generatedRoot_981.w_46_0  (ctx : sequential_generatedRoot_981Ctx) : mxxMat9 :=
  MxxRuntime.matrixAdd (sequential_generatedRoot_981.w_43_0 ctx) (sequential_generatedRoot_981.w_45_0 ctx)

def sequential_generatedRoot_981.w_48_0  (ctx : sequential_generatedRoot_981Ctx) : mxxMat10 :=
  (MxxRuntime.unitRow ((1 + (Int.ofNat i_0))) : mxxMat10)

def sequential_generatedRoot_981.w_49_0  (ctx : sequential_generatedRoot_981Ctx) : mxxMat9 :=
  ctx.witness.w_49_0

def sequential_generatedRoot_981.w_50_0  (ctx : sequential_generatedRoot_981Ctx) : mxxMat9 :=
  MxxRuntime.matrixAdd (sequential_generatedRoot_981.w_47_0 ctx) (sequential_generatedRoot_981.w_49_0 ctx)

def sequential_generatedRoot_981.outs  (ctx : sequential_generatedRoot_981Ctx) : mxxMat9 × mxxMat9 × mxxMat9 × mxxMat9 × mxxMat9 × mxxMat9 × Unit :=
  ((sequential_generatedRoot_981.w_30_0 ctx), ((sequential_generatedRoot_981.w_34_0 ctx), ((sequential_generatedRoot_981.w_38_0 ctx), ((sequential_generatedRoot_981.w_42_0 ctx), ((sequential_generatedRoot_981.w_46_0 ctx), ((sequential_generatedRoot_981.w_50_0 ctx), ()))))))

structure sequential_generatedRoot_981.Facts_0  (ctx : sequential_generatedRoot_981Ctx) : Prop where
  t0 : 0 ≤ ((1 + (Int.ofNat i_0)))
  t1 : ((1 + (Int.ofNat i_0))) < 4500
  h3_0 : 0 ≤ (sequential_generatedRoot_981.w_3_0 ctx)
  h3_0_1 : (sequential_generatedRoot_981.w_3_0 ctx) < 36000
  h4_0 : MxxRuntime.familyGetDynamic (sequential_generatedRoot_981.w_2_0 ctx) (sequential_generatedRoot_981.w_3_0 ctx) (sequential_generatedRoot_981.w_4_0 ctx)
  h5_0 : 0 ≤ (sequential_generatedRoot_981.w_5_0 ctx)
  h5_0_1 : (sequential_generatedRoot_981.w_5_0 ctx) < 36000
  h6_0 : MxxRuntime.familyGetDynamic (sequential_generatedRoot_981.w_2_0 ctx) (sequential_generatedRoot_981.w_5_0 ctx) (sequential_generatedRoot_981.w_6_0 ctx)
  h3_0_2 : 0 ≤ (sequential_generatedRoot_981.w_3_0 ctx)
  h3_0_3 : (sequential_generatedRoot_981.w_3_0 ctx) < 36000
  h9_0 : MxxRuntime.familyGetDynamic (sequential_generatedRoot_981.w_8_0 ctx) (sequential_generatedRoot_981.w_3_0 ctx) (sequential_generatedRoot_981.w_9_0 ctx)
  h5_0_2 : 0 ≤ (sequential_generatedRoot_981.w_5_0 ctx)
  h5_0_3 : (sequential_generatedRoot_981.w_5_0 ctx) < 36000
  h10_0 : MxxRuntime.familyGetDynamic (sequential_generatedRoot_981.w_8_0 ctx) (sequential_generatedRoot_981.w_5_0 ctx) (sequential_generatedRoot_981.w_10_0 ctx)
  h3_0_4 : 0 ≤ (sequential_generatedRoot_981.w_3_0 ctx)
  h3_0_5 : (sequential_generatedRoot_981.w_3_0 ctx) < 36000
  h13_0 : MxxRuntime.familyGetDynamic (sequential_generatedRoot_981.w_12_0 ctx) (sequential_generatedRoot_981.w_3_0 ctx) (sequential_generatedRoot_981.w_13_0 ctx)
  h5_0_4 : 0 ≤ (sequential_generatedRoot_981.w_5_0 ctx)
  h5_0_5 : (sequential_generatedRoot_981.w_5_0 ctx) < 36000
  h14_0 : MxxRuntime.familyGetDynamic (sequential_generatedRoot_981.w_12_0 ctx) (sequential_generatedRoot_981.w_5_0 ctx) (sequential_generatedRoot_981.w_14_0 ctx)
  h3_0_6 : 0 ≤ (sequential_generatedRoot_981.w_3_0 ctx)
  h3_0_7 : (sequential_generatedRoot_981.w_3_0 ctx) < 36000
  h17_0 : MxxRuntime.familyGetDynamic (sequential_generatedRoot_981.w_16_0 ctx) (sequential_generatedRoot_981.w_3_0 ctx) (sequential_generatedRoot_981.w_17_0 ctx)
  h5_0_6 : 0 ≤ (sequential_generatedRoot_981.w_5_0 ctx)
  h5_0_7 : (sequential_generatedRoot_981.w_5_0 ctx) < 36000
  h18_0 : MxxRuntime.familyGetDynamic (sequential_generatedRoot_981.w_16_0 ctx) (sequential_generatedRoot_981.w_5_0 ctx) (sequential_generatedRoot_981.w_18_0 ctx)
  h3_0_8 : 0 ≤ (sequential_generatedRoot_981.w_3_0 ctx)
  h3_0_9 : (sequential_generatedRoot_981.w_3_0 ctx) < 36000
  h21_0 : MxxRuntime.familyGetDynamic (sequential_generatedRoot_981.w_20_0 ctx) (sequential_generatedRoot_981.w_3_0 ctx) (sequential_generatedRoot_981.w_21_0 ctx)
  h5_0_8 : 0 ≤ (sequential_generatedRoot_981.w_5_0 ctx)
  h5_0_9 : (sequential_generatedRoot_981.w_5_0 ctx) < 36000
  h22_0 : MxxRuntime.familyGetDynamic (sequential_generatedRoot_981.w_20_0 ctx) (sequential_generatedRoot_981.w_5_0 ctx) (sequential_generatedRoot_981.w_22_0 ctx)

theorem sequential_generatedRoot_981.facts_0  {ctx : sequential_generatedRoot_981Ctx} {outputs : mxxMat9 × mxxMat9 × mxxMat9 × mxxMat9 × mxxMat9 × mxxMat9 × Unit}
    (h : sequential_generatedRoot_981.constraints_0 i_0 (sequential_generatedRoot_981.w_2_0 ctx) (sequential_generatedRoot_981.w_8_0 ctx) (sequential_generatedRoot_981.w_12_0 ctx) (sequential_generatedRoot_981.w_16_0 ctx) (sequential_generatedRoot_981.w_20_0 ctx) (sequential_generatedRoot_981.w_3_0 ctx) (sequential_generatedRoot_981.w_4_0 ctx) (sequential_generatedRoot_981.w_5_0 ctx) (sequential_generatedRoot_981.w_6_0 ctx) (sequential_generatedRoot_981.w_9_0 ctx) (sequential_generatedRoot_981.w_10_0 ctx) (sequential_generatedRoot_981.w_13_0 ctx) (sequential_generatedRoot_981.w_14_0 ctx) (sequential_generatedRoot_981.w_17_0 ctx) (sequential_generatedRoot_981.w_18_0 ctx) (sequential_generatedRoot_981.w_21_0 ctx) (sequential_generatedRoot_981.w_22_0 ctx)) :
    sequential_generatedRoot_981.Facts_0 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure sequential_generatedRoot_981.Facts_1  (ctx : sequential_generatedRoot_981Ctx) (outputs : mxxMat9 × mxxMat9 × mxxMat9 × mxxMat9 × mxxMat9 × mxxMat9 × Unit) : Prop where
  h3_0_10 : 0 ≤ (sequential_generatedRoot_981.w_3_0 ctx)
  h3_0_11 : (sequential_generatedRoot_981.w_3_0 ctx) < 36000
  h25_0 : MxxRuntime.familyGetDynamic (sequential_generatedRoot_981.w_24_0 ctx) (sequential_generatedRoot_981.w_3_0 ctx) (sequential_generatedRoot_981.w_25_0 ctx)
  h5_0_10 : 0 ≤ (sequential_generatedRoot_981.w_5_0 ctx)
  h5_0_11 : (sequential_generatedRoot_981.w_5_0 ctx) < 36000
  h26_0 : MxxRuntime.familyGetDynamic (sequential_generatedRoot_981.w_24_0 ctx) (sequential_generatedRoot_981.w_5_0 ctx) (sequential_generatedRoot_981.w_26_0 ctx)
  t38 : 0 ≤ (6)
  h28_5 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => scope_81 backend hashModel tape (path ++ [28, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, ((sequential_generatedRoot_981.w_2_0 ctx), ((sequential_generatedRoot_981.w_8_0 ctx), ((sequential_generatedRoot_981.w_12_0 ctx), ((sequential_generatedRoot_981.w_16_0 ctx), ((sequential_generatedRoot_981.w_20_0 ctx), ((sequential_generatedRoot_981.w_24_0 ctx), ())))))))))))) next) (Int.toNat (6)) ((sequential_generatedRoot_981.w_7_0 ctx), ((sequential_generatedRoot_981.w_11_0 ctx), ((sequential_generatedRoot_981.w_15_0 ctx), ((sequential_generatedRoot_981.w_19_0 ctx), ((sequential_generatedRoot_981.w_23_0 ctx), ((sequential_generatedRoot_981.w_27_0 ctx), ())))))) ((sequential_generatedRoot_981.w_28_0 ctx), ((sequential_generatedRoot_981.w_28_1 ctx), ((sequential_generatedRoot_981.w_28_2 ctx), ((sequential_generatedRoot_981.w_28_3 ctx), ((sequential_generatedRoot_981.w_28_4 ctx), ((sequential_generatedRoot_981.w_28_5 ctx), ()))))))
  h29_0 : MxxRuntime.tensorRuns (sequential_generatedRoot_981.w_1_0 ctx) (sequential_generatedRoot_981.w_28_0 ctx) (sequential_generatedRoot_981.w_29_0 ctx)
  t41 : 0 ≤ ((1 + (Int.ofNat i_0)))
  t42 : ((1 + (Int.ofNat i_0))) < 4500
  h33_0 : MxxRuntime.tensorRuns (sequential_generatedRoot_981.w_32_0 ctx) (sequential_generatedRoot_981.w_28_1 ctx) (sequential_generatedRoot_981.w_33_0 ctx)
  t44 : 0 ≤ ((1 + (Int.ofNat i_0)))
  t45 : ((1 + (Int.ofNat i_0))) < 4500
  h37_0 : MxxRuntime.tensorRuns (sequential_generatedRoot_981.w_36_0 ctx) (sequential_generatedRoot_981.w_28_2 ctx) (sequential_generatedRoot_981.w_37_0 ctx)
  t47 : 0 ≤ ((1 + (Int.ofNat i_0)))
  t48 : ((1 + (Int.ofNat i_0))) < 4500
  h41_0 : MxxRuntime.tensorRuns (sequential_generatedRoot_981.w_40_0 ctx) (sequential_generatedRoot_981.w_28_3 ctx) (sequential_generatedRoot_981.w_41_0 ctx)
  t50 : 0 ≤ ((1 + (Int.ofNat i_0)))
  t51 : ((1 + (Int.ofNat i_0))) < 4500
  h45_0 : MxxRuntime.tensorRuns (sequential_generatedRoot_981.w_44_0 ctx) (sequential_generatedRoot_981.w_28_4 ctx) (sequential_generatedRoot_981.w_45_0 ctx)
  t53 : 0 ≤ ((1 + (Int.ofNat i_0)))
  t54 : ((1 + (Int.ofNat i_0))) < 4500
  h49_0 : MxxRuntime.tensorRuns (sequential_generatedRoot_981.w_48_0 ctx) (sequential_generatedRoot_981.w_28_5 ctx) (sequential_generatedRoot_981.w_49_0 ctx)
  hout : outputs = sequential_generatedRoot_981.outs ctx

theorem sequential_generatedRoot_981.facts_1  {ctx : sequential_generatedRoot_981Ctx} {outputs : mxxMat9 × mxxMat9 × mxxMat9 × mxxMat9 × mxxMat9 × mxxMat9 × Unit}
    (h : sequential_generatedRoot_981.constraints_1 backend hashModel tape path params i_0 (sequential_generatedRoot_981.w_2_0 ctx) (sequential_generatedRoot_981.w_8_0 ctx) (sequential_generatedRoot_981.w_12_0 ctx) (sequential_generatedRoot_981.w_16_0 ctx) (sequential_generatedRoot_981.w_20_0 ctx) (sequential_generatedRoot_981.w_24_0 ctx) (sequential_generatedRoot_981.w_1_0 ctx) (sequential_generatedRoot_981.w_3_0 ctx) (sequential_generatedRoot_981.w_5_0 ctx) (sequential_generatedRoot_981.w_7_0 ctx) (sequential_generatedRoot_981.w_11_0 ctx) (sequential_generatedRoot_981.w_15_0 ctx) (sequential_generatedRoot_981.w_19_0 ctx) (sequential_generatedRoot_981.w_23_0 ctx) (sequential_generatedRoot_981.w_25_0 ctx) (sequential_generatedRoot_981.w_26_0 ctx) (sequential_generatedRoot_981.w_27_0 ctx) (sequential_generatedRoot_981.w_28_0 ctx) (sequential_generatedRoot_981.w_28_1 ctx) (sequential_generatedRoot_981.w_28_2 ctx) (sequential_generatedRoot_981.w_28_3 ctx) (sequential_generatedRoot_981.w_28_4 ctx) (sequential_generatedRoot_981.w_28_5 ctx) (sequential_generatedRoot_981.w_29_0 ctx) (sequential_generatedRoot_981.w_30_0 ctx) (sequential_generatedRoot_981.w_32_0 ctx) (sequential_generatedRoot_981.w_33_0 ctx) (sequential_generatedRoot_981.w_34_0 ctx) (sequential_generatedRoot_981.w_36_0 ctx) (sequential_generatedRoot_981.w_37_0 ctx) (sequential_generatedRoot_981.w_38_0 ctx) (sequential_generatedRoot_981.w_40_0 ctx) (sequential_generatedRoot_981.w_41_0 ctx) (sequential_generatedRoot_981.w_42_0 ctx) (sequential_generatedRoot_981.w_44_0 ctx) (sequential_generatedRoot_981.w_45_0 ctx) (sequential_generatedRoot_981.w_46_0 ctx) (sequential_generatedRoot_981.w_48_0 ctx) (sequential_generatedRoot_981.w_49_0 ctx) (sequential_generatedRoot_981.w_50_0 ctx) outputs) :
    sequential_generatedRoot_981.Facts_1 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure sequential_generatedRoot_981.Facts  (ctx : sequential_generatedRoot_981Ctx) (outputs : mxxMat9 × mxxMat9 × mxxMat9 × mxxMat9 × mxxMat9 × mxxMat9 × Unit) : Prop where
  c0 : sequential_generatedRoot_981.Facts_0 ctx
  c1 : sequential_generatedRoot_981.Facts_1 ctx outputs

theorem sequential_generatedRoot_981.facts  (ctx : sequential_generatedRoot_981Ctx) {outputs : mxxMat9 × mxxMat9 × mxxMat9 × mxxMat9 × mxxMat9 × mxxMat9 × Unit}
    (h : sequential_generatedRoot_981.body backend hashModel tape path params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_981.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_981.facts_0 (ctx := ctx) (outputs := outputs) t0.1
  have t1 := t0.2
  have r1 := sequential_generatedRoot_981.facts_1 (ctx := ctx) (outputs := outputs) t1
  ⟨r0, r1⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_984.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) where
  inputs : (parallel_generatedRoot_984.InputType)
  witness : parallel_generatedRoot_984.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_984Ctx" => (parallel_generatedRoot_984.Ctx backend hashModel tape path params i_0)

def parallel_generatedRoot_984.w_0_0  (ctx : parallel_generatedRoot_984Ctx) : mxxMat0 :=
  ctx.inputs.1

def parallel_generatedRoot_984.w_1_0  (ctx : parallel_generatedRoot_984Ctx) : mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_984.w_4_0  (ctx : parallel_generatedRoot_984Ctx) : mxxMat2 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_984.w_9_0  (ctx : parallel_generatedRoot_984Ctx) : mxxMat0 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_984.w_25_0  (ctx : parallel_generatedRoot_984Ctx) : mxxMat7 :=
  ctx.inputs.2.2.2.2.1

def parallel_generatedRoot_984.w_27_0  (ctx : parallel_generatedRoot_984Ctx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.1

def parallel_generatedRoot_984.w_44_0  (ctx : parallel_generatedRoot_984Ctx) : mxxMat9 :=
  ctx.inputs.2.2.2.2.2.2.1

def parallel_generatedRoot_984.w_45_0  (ctx : parallel_generatedRoot_984Ctx) : Fin 1 → mxxMat5 :=
  ctx.inputs.2.2.2.2.2.2.2.1

def parallel_generatedRoot_984.w_46_0  (ctx : parallel_generatedRoot_984Ctx) : mxxMat9 :=
  ctx.inputs.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_984.w_47_0  (ctx : parallel_generatedRoot_984Ctx) : mxxMat9 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_984.w_50_0  (ctx : parallel_generatedRoot_984Ctx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_984.w_51_0  (ctx : parallel_generatedRoot_984Ctx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_984.w_58_0  (ctx : parallel_generatedRoot_984Ctx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_984.w_74_0  (ctx : parallel_generatedRoot_984Ctx) : mxxMat7 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_984.w_2_0  (ctx : parallel_generatedRoot_984Ctx) : mxxMat11 :=
  ctx.witness.w_2_0

def parallel_generatedRoot_984.w_3_0  (ctx : parallel_generatedRoot_984Ctx) : mxxMat5 :=
  (MxxRuntime.matrixPolynomial [(1606938044254428791916078459060479968194539524630727034176641)] : mxxMat5)

def parallel_generatedRoot_984.w_5_0  (ctx : parallel_generatedRoot_984Ctx) : mxxMat2 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_984.w_3_0 ctx) (parallel_generatedRoot_984.w_4_0 ctx)

def parallel_generatedRoot_984.w_6_0  (ctx : parallel_generatedRoot_984Ctx) : mxxMat4 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_984.w_7_0  (ctx : parallel_generatedRoot_984Ctx) : mxxMat4 :=
  ctx.witness.w_7_0

def parallel_generatedRoot_984.w_8_0  (ctx : parallel_generatedRoot_984Ctx) : mxxMat0 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_984.w_2_0 ctx) (parallel_generatedRoot_984.w_7_0 ctx)

def parallel_generatedRoot_984.w_10_0  (ctx : parallel_generatedRoot_984Ctx) : mxxMat0 :=
  ctx.witness.w_10_0

def parallel_generatedRoot_984.w_11_0  (ctx : parallel_generatedRoot_984Ctx) : mxxMat0 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_984.w_12_0  (ctx : parallel_generatedRoot_984Ctx) : mxxMat12 :=
  ctx.witness.w_12_0

def parallel_generatedRoot_984.w_13_0  (ctx : parallel_generatedRoot_984Ctx) : mxxMat13 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat13)

def parallel_generatedRoot_984.w_14_0  (ctx : parallel_generatedRoot_984Ctx) : mxxMat12 :=
  MxxRuntime.matrixMulScalarRight (parallel_generatedRoot_984.w_12_0 ctx) (parallel_generatedRoot_984.w_13_0 ctx)

def parallel_generatedRoot_984.w_15_0  (ctx : parallel_generatedRoot_984Ctx) : mxxMat13 :=
  (MxxRuntime.matrixPolynomial [(11560550400)] : mxxMat13)

def parallel_generatedRoot_984.w_16_0  (ctx : parallel_generatedRoot_984Ctx) : mxxMat12 :=
  MxxRuntime.matrixMulScalarRight (parallel_generatedRoot_984.w_14_0 ctx) (parallel_generatedRoot_984.w_15_0 ctx)

def parallel_generatedRoot_984.w_17_0  (ctx : parallel_generatedRoot_984Ctx) : mxxMat0 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_984.w_18_0  (ctx : parallel_generatedRoot_984Ctx) : mxxMat5 :=
  (MxxRuntime.matrixPolynomial [(mxxNat0)] : mxxMat5)

def parallel_generatedRoot_984.w_19_0  (ctx : parallel_generatedRoot_984Ctx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarRight (parallel_generatedRoot_984.w_17_0 ctx) (parallel_generatedRoot_984.w_18_0 ctx)

def parallel_generatedRoot_984.w_20_0  (ctx : parallel_generatedRoot_984Ctx) : mxxMat6 :=
  ctx.witness.w_20_0

def parallel_generatedRoot_984.w_21_0  (ctx : parallel_generatedRoot_984Ctx) : mxxMat6 :=
  ctx.witness.w_21_0

def parallel_generatedRoot_984.w_22_0  (ctx : parallel_generatedRoot_984Ctx) : mxxMat0 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_984.w_9_0 ctx) (parallel_generatedRoot_984.w_21_0 ctx)

def parallel_generatedRoot_984.w_23_0  (ctx : parallel_generatedRoot_984Ctx) : mxxMat0 :=
  MxxRuntime.matrixSub (parallel_generatedRoot_984.w_8_0 ctx) (parallel_generatedRoot_984.w_22_0 ctx)

def parallel_generatedRoot_984.w_24_0  (ctx : parallel_generatedRoot_984Ctx) : mxxMat0 :=
  (MxxRuntime.unitRow (0) : mxxMat0)

def parallel_generatedRoot_984.w_26_0  (ctx : parallel_generatedRoot_984Ctx) : mxxMat14 :=
  ctx.witness.w_26_0

def parallel_generatedRoot_984.w_28_0  (ctx : parallel_generatedRoot_984Ctx) : mxxMat15 :=
  ctx.witness.w_28_0

def parallel_generatedRoot_984.w_29_0  (ctx : parallel_generatedRoot_984Ctx) : mxxMat15 :=
  ctx.witness.w_29_0

def parallel_generatedRoot_984.w_30_0  (ctx : parallel_generatedRoot_984Ctx) : mxxMat5 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat5)

def parallel_generatedRoot_984.w_31_0  (ctx : parallel_generatedRoot_984Ctx) : mxxMat15 :=
  MxxRuntime.matrixMulScalarRight (parallel_generatedRoot_984.w_29_0 ctx) (parallel_generatedRoot_984.w_30_0 ctx)

def parallel_generatedRoot_984.w_32_0  (ctx : parallel_generatedRoot_984Ctx) : mxxMat16 :=
  ctx.witness.w_32_0

def parallel_generatedRoot_984.w_33_0  (ctx : parallel_generatedRoot_984Ctx) : mxxMat16 :=
  ctx.witness.w_33_0

def parallel_generatedRoot_984.w_34_0  (ctx : parallel_generatedRoot_984Ctx) : mxxMat14 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_984.w_26_0 ctx) (parallel_generatedRoot_984.w_33_0 ctx)

def parallel_generatedRoot_984.w_35_0  (ctx : parallel_generatedRoot_984Ctx) : mxxMat0 :=
  ctx.witness.w_35_0

def parallel_generatedRoot_984.w_36_0  (ctx : parallel_generatedRoot_984Ctx) : mxxMat5 :=
  (MxxRuntime.matrixPolynomial [(1606938044254428791916078459060479968194539524630727034176641)] : mxxMat5)

def parallel_generatedRoot_984.w_37_0  (ctx : parallel_generatedRoot_984Ctx) : mxxMat17 :=
  ctx.witness.w_37_0

def parallel_generatedRoot_984.w_38_0  (ctx : parallel_generatedRoot_984Ctx) : mxxMat17 :=
  ctx.witness.w_38_0

def parallel_generatedRoot_984.w_39_0  (ctx : parallel_generatedRoot_984Ctx) : mxxMat5 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_984.w_35_0 ctx) (parallel_generatedRoot_984.w_38_0 ctx)

def parallel_generatedRoot_984.w_40_0  (ctx : parallel_generatedRoot_984Ctx) : mxxMat0 :=
  ctx.witness.w_40_0

def parallel_generatedRoot_984.w_41_0  (ctx : parallel_generatedRoot_984Ctx) : mxxMat0 :=
  ctx.witness.w_41_0

def parallel_generatedRoot_984.w_42_0  (ctx : parallel_generatedRoot_984Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_984.w_23_0 ctx) (parallel_generatedRoot_984.w_41_0 ctx)

def parallel_generatedRoot_984.w_43_0  (ctx : parallel_generatedRoot_984Ctx) : mxxMat0 :=
  (0 : mxxMat0)

def parallel_generatedRoot_984.w_48_0  (ctx : parallel_generatedRoot_984Ctx) : mxxMat0 :=
  ctx.witness.w_48_0

def parallel_generatedRoot_984.w_48_1  (ctx : parallel_generatedRoot_984Ctx) : mxxMat0 :=
  ctx.witness.w_48_1

def parallel_generatedRoot_984.w_48_2  (ctx : parallel_generatedRoot_984Ctx) : mxxMat0 :=
  ctx.witness.w_48_2

def parallel_generatedRoot_984.w_49_0  (ctx : parallel_generatedRoot_984Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_984.w_42_0 ctx) (parallel_generatedRoot_984.w_48_1 ctx)

def parallel_generatedRoot_984.w_52_0  (ctx : parallel_generatedRoot_984Ctx) : mxxMat11 :=
  ctx.witness.w_52_0

def parallel_generatedRoot_984.w_53_0  (ctx : parallel_generatedRoot_984Ctx) : mxxMat5 :=
  (MxxRuntime.matrixPolynomial [(1606938044254428791916078459060479968194539524630727034176641)] : mxxMat5)

def parallel_generatedRoot_984.w_54_0  (ctx : parallel_generatedRoot_984Ctx) : mxxMat2 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_984.w_53_0 ctx) (parallel_generatedRoot_984.w_4_0 ctx)

def parallel_generatedRoot_984.w_55_0  (ctx : parallel_generatedRoot_984Ctx) : mxxMat4 :=
  ctx.witness.w_55_0

def parallel_generatedRoot_984.w_56_0  (ctx : parallel_generatedRoot_984Ctx) : mxxMat4 :=
  ctx.witness.w_56_0

def parallel_generatedRoot_984.w_57_0  (ctx : parallel_generatedRoot_984Ctx) : mxxMat0 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_984.w_52_0 ctx) (parallel_generatedRoot_984.w_56_0 ctx)

def parallel_generatedRoot_984.w_59_0  (ctx : parallel_generatedRoot_984Ctx) : mxxMat0 :=
  ctx.witness.w_59_0

def parallel_generatedRoot_984.w_60_0  (ctx : parallel_generatedRoot_984Ctx) : mxxMat0 :=
  ctx.witness.w_60_0

def parallel_generatedRoot_984.w_61_0  (ctx : parallel_generatedRoot_984Ctx) : mxxMat12 :=
  ctx.witness.w_61_0

def parallel_generatedRoot_984.w_62_0  (ctx : parallel_generatedRoot_984Ctx) : mxxMat13 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat13)

def parallel_generatedRoot_984.w_63_0  (ctx : parallel_generatedRoot_984Ctx) : mxxMat12 :=
  MxxRuntime.matrixMulScalarRight (parallel_generatedRoot_984.w_61_0 ctx) (parallel_generatedRoot_984.w_62_0 ctx)

def parallel_generatedRoot_984.w_64_0  (ctx : parallel_generatedRoot_984Ctx) : mxxMat13 :=
  (MxxRuntime.matrixPolynomial [(11560550400)] : mxxMat13)

def parallel_generatedRoot_984.w_65_0  (ctx : parallel_generatedRoot_984Ctx) : mxxMat12 :=
  MxxRuntime.matrixMulScalarRight (parallel_generatedRoot_984.w_63_0 ctx) (parallel_generatedRoot_984.w_64_0 ctx)

def parallel_generatedRoot_984.w_66_0  (ctx : parallel_generatedRoot_984Ctx) : mxxMat0 :=
  ctx.witness.w_66_0

def parallel_generatedRoot_984.w_67_0  (ctx : parallel_generatedRoot_984Ctx) : mxxMat5 :=
  (MxxRuntime.matrixPolynomial [(mxxNat0)] : mxxMat5)

def parallel_generatedRoot_984.w_68_0  (ctx : parallel_generatedRoot_984Ctx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarRight (parallel_generatedRoot_984.w_66_0 ctx) (parallel_generatedRoot_984.w_67_0 ctx)

def parallel_generatedRoot_984.w_69_0  (ctx : parallel_generatedRoot_984Ctx) : mxxMat6 :=
  ctx.witness.w_69_0

def parallel_generatedRoot_984.w_70_0  (ctx : parallel_generatedRoot_984Ctx) : mxxMat6 :=
  ctx.witness.w_70_0

def parallel_generatedRoot_984.w_71_0  (ctx : parallel_generatedRoot_984Ctx) : mxxMat0 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_984.w_58_0 ctx) (parallel_generatedRoot_984.w_70_0 ctx)

def parallel_generatedRoot_984.w_72_0  (ctx : parallel_generatedRoot_984Ctx) : mxxMat0 :=
  MxxRuntime.matrixSub (parallel_generatedRoot_984.w_57_0 ctx) (parallel_generatedRoot_984.w_71_0 ctx)

def parallel_generatedRoot_984.w_73_0  (ctx : parallel_generatedRoot_984Ctx) : mxxMat0 :=
  (MxxRuntime.unitRow (0) : mxxMat0)

def parallel_generatedRoot_984.w_75_0  (ctx : parallel_generatedRoot_984Ctx) : mxxMat14 :=
  ctx.witness.w_75_0

def parallel_generatedRoot_984.w_76_0  (ctx : parallel_generatedRoot_984Ctx) : mxxMat15 :=
  ctx.witness.w_76_0

def parallel_generatedRoot_984.w_77_0  (ctx : parallel_generatedRoot_984Ctx) : mxxMat15 :=
  ctx.witness.w_77_0

def parallel_generatedRoot_984.w_78_0  (ctx : parallel_generatedRoot_984Ctx) : mxxMat5 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat5)

def parallel_generatedRoot_984.w_79_0  (ctx : parallel_generatedRoot_984Ctx) : mxxMat15 :=
  MxxRuntime.matrixMulScalarRight (parallel_generatedRoot_984.w_77_0 ctx) (parallel_generatedRoot_984.w_78_0 ctx)

def parallel_generatedRoot_984.w_80_0  (ctx : parallel_generatedRoot_984Ctx) : mxxMat16 :=
  ctx.witness.w_80_0

def parallel_generatedRoot_984.w_81_0  (ctx : parallel_generatedRoot_984Ctx) : mxxMat16 :=
  ctx.witness.w_81_0

def parallel_generatedRoot_984.w_82_0  (ctx : parallel_generatedRoot_984Ctx) : mxxMat14 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_984.w_75_0 ctx) (parallel_generatedRoot_984.w_81_0 ctx)

def parallel_generatedRoot_984.w_83_0  (ctx : parallel_generatedRoot_984Ctx) : mxxMat0 :=
  ctx.witness.w_83_0

def parallel_generatedRoot_984.w_84_0  (ctx : parallel_generatedRoot_984Ctx) : mxxMat5 :=
  (MxxRuntime.matrixPolynomial [(1606938044254428791916078459060479968194539524630727034176641)] : mxxMat5)

def parallel_generatedRoot_984.w_85_0  (ctx : parallel_generatedRoot_984Ctx) : mxxMat17 :=
  ctx.witness.w_85_0

def parallel_generatedRoot_984.w_86_0  (ctx : parallel_generatedRoot_984Ctx) : mxxMat17 :=
  ctx.witness.w_86_0

def parallel_generatedRoot_984.w_87_0  (ctx : parallel_generatedRoot_984Ctx) : mxxMat5 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_984.w_83_0 ctx) (parallel_generatedRoot_984.w_86_0 ctx)

def parallel_generatedRoot_984.w_88_0  (ctx : parallel_generatedRoot_984Ctx) : mxxMat0 :=
  ctx.witness.w_88_0

def parallel_generatedRoot_984.w_89_0  (ctx : parallel_generatedRoot_984Ctx) : mxxMat0 :=
  ctx.witness.w_89_0

def parallel_generatedRoot_984.w_90_0  (ctx : parallel_generatedRoot_984Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_984.w_72_0 ctx) (parallel_generatedRoot_984.w_89_0 ctx)

def parallel_generatedRoot_984.w_91_0  (ctx : parallel_generatedRoot_984Ctx) : mxxMat0 :=
  (0 : mxxMat0)

def parallel_generatedRoot_984.w_92_0  (ctx : parallel_generatedRoot_984Ctx) : mxxMat0 :=
  ctx.witness.w_92_0

def parallel_generatedRoot_984.w_92_1  (ctx : parallel_generatedRoot_984Ctx) : mxxMat0 :=
  ctx.witness.w_92_1

def parallel_generatedRoot_984.w_92_2  (ctx : parallel_generatedRoot_984Ctx) : mxxMat0 :=
  ctx.witness.w_92_2

def parallel_generatedRoot_984.w_93_0  (ctx : parallel_generatedRoot_984Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_984.w_90_0 ctx) (parallel_generatedRoot_984.w_92_2 ctx)

def parallel_generatedRoot_984.w_94_0  (ctx : parallel_generatedRoot_984Ctx) : mxxMat2 :=
  ctx.witness.w_94_0

def parallel_generatedRoot_984.outs  (ctx : parallel_generatedRoot_984Ctx) : mxxMat2 :=
  (parallel_generatedRoot_984.w_94_0 ctx)

structure parallel_generatedRoot_984.Facts_0  (ctx : parallel_generatedRoot_984Ctx) : Prop where
  h2_0 : MxxRuntime.concatColumns (parallel_generatedRoot_984.w_0_0 ctx) (parallel_generatedRoot_984.w_1_0 ctx) (parallel_generatedRoot_984.w_2_0 ctx)
  h6_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (parallel_generatedRoot_984.w_5_0 ctx) (parallel_generatedRoot_984.w_6_0 ctx)
  h7_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (parallel_generatedRoot_984.w_6_0 ctx) (parallel_generatedRoot_984.w_7_0 ctx)
  h10_0 : MxxRuntime.gadgetMatrixRuns backend 4096 25 (parallel_generatedRoot_984.w_10_0 ctx)
  h11_0 : MxxRuntime.centeredRebaseRuns (parallel_generatedRoot_984.w_10_0 ctx) (parallel_generatedRoot_984.w_11_0 ctx)
  h12_0 : MxxRuntime.modulusReduceRuns (parallel_generatedRoot_984.w_11_0 ctx) (parallel_generatedRoot_984.w_12_0 ctx)
  h17_0 : MxxRuntime.centeredRebaseRuns (parallel_generatedRoot_984.w_16_0 ctx) (parallel_generatedRoot_984.w_17_0 ctx)
  h20_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (parallel_generatedRoot_984.w_19_0 ctx) (parallel_generatedRoot_984.w_20_0 ctx)
  h21_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (parallel_generatedRoot_984.w_20_0 ctx) (parallel_generatedRoot_984.w_21_0 ctx)
  t9 : 0 ≤ (0)
  t10 : (0) < 25
  t11 : 0 ≤ 0
  t12 : 0 < 1
  t13 : 1 ≤ 1
  t14 : 0 ≤ (625 * (Int.ofNat i_0))
  t15 : (625 * (Int.ofNat i_0)) < (625 + (625 * (Int.ofNat i_0)))
  t16 : (625 + (625 * (Int.ofNat i_0))) ≤ 22500
  h26_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_984.w_25_0 ctx) 0 1 (625 * (Int.ofNat i_0)) (625 + (625 * (Int.ofNat i_0))) (parallel_generatedRoot_984.w_26_0 ctx)
  h28_0 : MxxRuntime.blockDiagonalRuns 25 (parallel_generatedRoot_984.w_27_0 ctx) (parallel_generatedRoot_984.w_28_0 ctx)
  h29_0 : MxxRuntime.centeredRebaseRuns (parallel_generatedRoot_984.w_28_0 ctx) (parallel_generatedRoot_984.w_29_0 ctx)
  h32_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (parallel_generatedRoot_984.w_31_0 ctx) (parallel_generatedRoot_984.w_32_0 ctx)
  h33_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (parallel_generatedRoot_984.w_32_0 ctx) (parallel_generatedRoot_984.w_33_0 ctx)
  t22 : 0 ≤ 0
  t23 : 0 < 1
  t24 : 1 ≤ 1
  t25 : 0 ≤ 0
  t26 : 0 < 25
  t27 : 25 ≤ 625
  h35_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_984.w_34_0 ctx) 0 1 0 25 (parallel_generatedRoot_984.w_35_0 ctx)
  h37_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (parallel_generatedRoot_984.w_36_0 ctx) (parallel_generatedRoot_984.w_37_0 ctx)
  h38_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (parallel_generatedRoot_984.w_37_0 ctx) (parallel_generatedRoot_984.w_38_0 ctx)
  h40_0 : MxxRuntime.tensorRuns (parallel_generatedRoot_984.w_24_0 ctx) (parallel_generatedRoot_984.w_39_0 ctx) (parallel_generatedRoot_984.w_40_0 ctx)

theorem parallel_generatedRoot_984.facts_0  {ctx : parallel_generatedRoot_984Ctx} {outputs : mxxMat2}
    (h : parallel_generatedRoot_984.constraints_0 backend i_0 (parallel_generatedRoot_984.w_0_0 ctx) (parallel_generatedRoot_984.w_1_0 ctx) (parallel_generatedRoot_984.w_25_0 ctx) (parallel_generatedRoot_984.w_27_0 ctx) (parallel_generatedRoot_984.w_2_0 ctx) (parallel_generatedRoot_984.w_5_0 ctx) (parallel_generatedRoot_984.w_6_0 ctx) (parallel_generatedRoot_984.w_7_0 ctx) (parallel_generatedRoot_984.w_10_0 ctx) (parallel_generatedRoot_984.w_11_0 ctx) (parallel_generatedRoot_984.w_12_0 ctx) (parallel_generatedRoot_984.w_16_0 ctx) (parallel_generatedRoot_984.w_17_0 ctx) (parallel_generatedRoot_984.w_19_0 ctx) (parallel_generatedRoot_984.w_20_0 ctx) (parallel_generatedRoot_984.w_21_0 ctx) (parallel_generatedRoot_984.w_24_0 ctx) (parallel_generatedRoot_984.w_26_0 ctx) (parallel_generatedRoot_984.w_28_0 ctx) (parallel_generatedRoot_984.w_29_0 ctx) (parallel_generatedRoot_984.w_31_0 ctx) (parallel_generatedRoot_984.w_32_0 ctx) (parallel_generatedRoot_984.w_33_0 ctx) (parallel_generatedRoot_984.w_34_0 ctx) (parallel_generatedRoot_984.w_35_0 ctx) (parallel_generatedRoot_984.w_36_0 ctx) (parallel_generatedRoot_984.w_37_0 ctx) (parallel_generatedRoot_984.w_38_0 ctx) (parallel_generatedRoot_984.w_39_0 ctx) (parallel_generatedRoot_984.w_40_0 ctx)) :
    parallel_generatedRoot_984.Facts_0 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_984.Facts_1  (ctx : parallel_generatedRoot_984Ctx) : Prop where
  t32 : 0 ≤ (24)
  h41_0 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0) => sequential_parallel_generatedRoot_984_41 backend hashModel tape (path ++ [41, i]) params i_0 i (current, ((parallel_generatedRoot_984.w_34_0 ctx), ((parallel_generatedRoot_984.w_38_0 ctx), ()))) next) (Int.toNat (24)) (parallel_generatedRoot_984.w_40_0 ctx) (parallel_generatedRoot_984.w_41_0 ctx)
  t34 : 0 ≤ (1)
  h48_2 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_984_48 backend hashModel tape (path ++ [48, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, ((parallel_generatedRoot_984.w_44_0 ctx), ((parallel_generatedRoot_984.w_45_0 ctx), ((parallel_generatedRoot_984.w_46_0 ctx), ((parallel_generatedRoot_984.w_47_0 ctx), ()))))))) next) (Int.toNat (1)) ((parallel_generatedRoot_984.w_43_0 ctx), ((parallel_generatedRoot_984.w_43_0 ctx), ((parallel_generatedRoot_984.w_43_0 ctx), ()))) ((parallel_generatedRoot_984.w_48_0 ctx), ((parallel_generatedRoot_984.w_48_1 ctx), ((parallel_generatedRoot_984.w_48_2 ctx), ())))
  h52_0 : MxxRuntime.concatColumns (parallel_generatedRoot_984.w_50_0 ctx) (parallel_generatedRoot_984.w_51_0 ctx) (parallel_generatedRoot_984.w_52_0 ctx)
  h55_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (parallel_generatedRoot_984.w_54_0 ctx) (parallel_generatedRoot_984.w_55_0 ctx)
  h56_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (parallel_generatedRoot_984.w_55_0 ctx) (parallel_generatedRoot_984.w_56_0 ctx)
  h59_0 : MxxRuntime.gadgetMatrixRuns backend 4096 25 (parallel_generatedRoot_984.w_59_0 ctx)
  h60_0 : MxxRuntime.centeredRebaseRuns (parallel_generatedRoot_984.w_59_0 ctx) (parallel_generatedRoot_984.w_60_0 ctx)
  h61_0 : MxxRuntime.modulusReduceRuns (parallel_generatedRoot_984.w_60_0 ctx) (parallel_generatedRoot_984.w_61_0 ctx)
  h66_0 : MxxRuntime.centeredRebaseRuns (parallel_generatedRoot_984.w_65_0 ctx) (parallel_generatedRoot_984.w_66_0 ctx)
  h69_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (parallel_generatedRoot_984.w_68_0 ctx) (parallel_generatedRoot_984.w_69_0 ctx)
  h70_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (parallel_generatedRoot_984.w_69_0 ctx) (parallel_generatedRoot_984.w_70_0 ctx)
  t45 : 0 ≤ (0)
  t46 : (0) < 25
  t47 : 0 ≤ 0
  t48 : 0 < 1
  t49 : 1 ≤ 1
  t50 : 0 ≤ (625 * (Int.ofNat i_0))
  t51 : (625 * (Int.ofNat i_0)) < (625 + (625 * (Int.ofNat i_0)))
  t52 : (625 + (625 * (Int.ofNat i_0))) ≤ 22500
  h75_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_984.w_74_0 ctx) 0 1 (625 * (Int.ofNat i_0)) (625 + (625 * (Int.ofNat i_0))) (parallel_generatedRoot_984.w_75_0 ctx)
  h76_0 : MxxRuntime.blockDiagonalRuns 25 (parallel_generatedRoot_984.w_27_0 ctx) (parallel_generatedRoot_984.w_76_0 ctx)
  h77_0 : MxxRuntime.centeredRebaseRuns (parallel_generatedRoot_984.w_76_0 ctx) (parallel_generatedRoot_984.w_77_0 ctx)
  h80_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (parallel_generatedRoot_984.w_79_0 ctx) (parallel_generatedRoot_984.w_80_0 ctx)
  h81_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (parallel_generatedRoot_984.w_80_0 ctx) (parallel_generatedRoot_984.w_81_0 ctx)
  t58 : 0 ≤ 0
  t59 : 0 < 1
  t60 : 1 ≤ 1
  t61 : 0 ≤ 0
  t62 : 0 < 25
  t63 : 25 ≤ 625

theorem parallel_generatedRoot_984.facts_1  {ctx : parallel_generatedRoot_984Ctx} {outputs : mxxMat2}
    (h : parallel_generatedRoot_984.constraints_1 backend hashModel tape path params i_0 (parallel_generatedRoot_984.w_27_0 ctx) (parallel_generatedRoot_984.w_44_0 ctx) (parallel_generatedRoot_984.w_45_0 ctx) (parallel_generatedRoot_984.w_46_0 ctx) (parallel_generatedRoot_984.w_47_0 ctx) (parallel_generatedRoot_984.w_50_0 ctx) (parallel_generatedRoot_984.w_51_0 ctx) (parallel_generatedRoot_984.w_74_0 ctx) (parallel_generatedRoot_984.w_34_0 ctx) (parallel_generatedRoot_984.w_38_0 ctx) (parallel_generatedRoot_984.w_40_0 ctx) (parallel_generatedRoot_984.w_41_0 ctx) (parallel_generatedRoot_984.w_43_0 ctx) (parallel_generatedRoot_984.w_48_0 ctx) (parallel_generatedRoot_984.w_48_1 ctx) (parallel_generatedRoot_984.w_48_2 ctx) (parallel_generatedRoot_984.w_52_0 ctx) (parallel_generatedRoot_984.w_54_0 ctx) (parallel_generatedRoot_984.w_55_0 ctx) (parallel_generatedRoot_984.w_56_0 ctx) (parallel_generatedRoot_984.w_59_0 ctx) (parallel_generatedRoot_984.w_60_0 ctx) (parallel_generatedRoot_984.w_61_0 ctx) (parallel_generatedRoot_984.w_65_0 ctx) (parallel_generatedRoot_984.w_66_0 ctx) (parallel_generatedRoot_984.w_68_0 ctx) (parallel_generatedRoot_984.w_69_0 ctx) (parallel_generatedRoot_984.w_70_0 ctx) (parallel_generatedRoot_984.w_75_0 ctx) (parallel_generatedRoot_984.w_76_0 ctx) (parallel_generatedRoot_984.w_77_0 ctx) (parallel_generatedRoot_984.w_79_0 ctx) (parallel_generatedRoot_984.w_80_0 ctx) (parallel_generatedRoot_984.w_81_0 ctx)) :
    parallel_generatedRoot_984.Facts_1 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_984.Facts_2  (ctx : parallel_generatedRoot_984Ctx) (outputs : mxxMat2) : Prop where
  h83_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_984.w_82_0 ctx) 0 1 0 25 (parallel_generatedRoot_984.w_83_0 ctx)
  h85_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (parallel_generatedRoot_984.w_84_0 ctx) (parallel_generatedRoot_984.w_85_0 ctx)
  h86_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (parallel_generatedRoot_984.w_85_0 ctx) (parallel_generatedRoot_984.w_86_0 ctx)
  h88_0 : MxxRuntime.tensorRuns (parallel_generatedRoot_984.w_73_0 ctx) (parallel_generatedRoot_984.w_87_0 ctx) (parallel_generatedRoot_984.w_88_0 ctx)
  t68 : 0 ≤ (24)
  h89_0 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0) => sequential_parallel_generatedRoot_984_89 backend hashModel tape (path ++ [89, i]) params i_0 i (current, ((parallel_generatedRoot_984.w_82_0 ctx), ((parallel_generatedRoot_984.w_86_0 ctx), ()))) next) (Int.toNat (24)) (parallel_generatedRoot_984.w_88_0 ctx) (parallel_generatedRoot_984.w_89_0 ctx)
  t70 : 0 ≤ (1)
  h92_2 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_984_92 backend hashModel tape (path ++ [92, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, ((parallel_generatedRoot_984.w_44_0 ctx), ((parallel_generatedRoot_984.w_45_0 ctx), ((parallel_generatedRoot_984.w_46_0 ctx), ((parallel_generatedRoot_984.w_47_0 ctx), ()))))))) next) (Int.toNat (1)) ((parallel_generatedRoot_984.w_91_0 ctx), ((parallel_generatedRoot_984.w_91_0 ctx), ((parallel_generatedRoot_984.w_91_0 ctx), ()))) ((parallel_generatedRoot_984.w_92_0 ctx), ((parallel_generatedRoot_984.w_92_1 ctx), ((parallel_generatedRoot_984.w_92_2 ctx), ())))
  h94_0 : MxxRuntime.concatRows (parallel_generatedRoot_984.w_49_0 ctx) (parallel_generatedRoot_984.w_93_0 ctx) (parallel_generatedRoot_984.w_94_0 ctx)
  hout : outputs = parallel_generatedRoot_984.outs ctx

theorem parallel_generatedRoot_984.facts_2  {ctx : parallel_generatedRoot_984Ctx} {outputs : mxxMat2}
    (h : parallel_generatedRoot_984.constraints_2 backend hashModel tape path params i_0 (parallel_generatedRoot_984.w_44_0 ctx) (parallel_generatedRoot_984.w_45_0 ctx) (parallel_generatedRoot_984.w_46_0 ctx) (parallel_generatedRoot_984.w_47_0 ctx) (parallel_generatedRoot_984.w_49_0 ctx) (parallel_generatedRoot_984.w_73_0 ctx) (parallel_generatedRoot_984.w_82_0 ctx) (parallel_generatedRoot_984.w_83_0 ctx) (parallel_generatedRoot_984.w_84_0 ctx) (parallel_generatedRoot_984.w_85_0 ctx) (parallel_generatedRoot_984.w_86_0 ctx) (parallel_generatedRoot_984.w_87_0 ctx) (parallel_generatedRoot_984.w_88_0 ctx) (parallel_generatedRoot_984.w_89_0 ctx) (parallel_generatedRoot_984.w_91_0 ctx) (parallel_generatedRoot_984.w_92_0 ctx) (parallel_generatedRoot_984.w_92_1 ctx) (parallel_generatedRoot_984.w_92_2 ctx) (parallel_generatedRoot_984.w_93_0 ctx) (parallel_generatedRoot_984.w_94_0 ctx) outputs) :
    parallel_generatedRoot_984.Facts_2 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_984.Facts  (ctx : parallel_generatedRoot_984Ctx) (outputs : mxxMat2) : Prop where
  c0 : parallel_generatedRoot_984.Facts_0 ctx
  c1 : parallel_generatedRoot_984.Facts_1 ctx
  c2 : parallel_generatedRoot_984.Facts_2 ctx outputs

theorem parallel_generatedRoot_984.facts  (ctx : parallel_generatedRoot_984Ctx) {outputs : mxxMat2}
    (h : parallel_generatedRoot_984.body backend hashModel tape path params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_984.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_984.facts_0 (ctx := ctx) (outputs := outputs) t0.1
  have t1 := t0.2
  have r1 := parallel_generatedRoot_984.facts_1 (ctx := ctx) (outputs := outputs) t1.1
  have t2 := t1.2
  have r2 := parallel_generatedRoot_984.facts_2 (ctx := ctx) (outputs := outputs) t2
  ⟨r0, r1, r2⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_985.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) where
  inputs : (parallel_generatedRoot_985.InputType)
  witness : parallel_generatedRoot_985.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_985Ctx" => (parallel_generatedRoot_985.Ctx backend hashModel tape path params i_0)

def parallel_generatedRoot_985.w_0_0  (ctx : parallel_generatedRoot_985Ctx) : mxxMat0 :=
  ctx.inputs.1

def parallel_generatedRoot_985.w_1_0  (ctx : parallel_generatedRoot_985Ctx) : mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_985.w_4_0  (ctx : parallel_generatedRoot_985Ctx) : mxxMat2 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_985.w_9_0  (ctx : parallel_generatedRoot_985Ctx) : mxxMat0 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_985.w_25_0  (ctx : parallel_generatedRoot_985Ctx) : mxxMat7 :=
  ctx.inputs.2.2.2.2.1

def parallel_generatedRoot_985.w_27_0  (ctx : parallel_generatedRoot_985Ctx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.1

def parallel_generatedRoot_985.w_44_0  (ctx : parallel_generatedRoot_985Ctx) : mxxMat9 :=
  ctx.inputs.2.2.2.2.2.2.1

def parallel_generatedRoot_985.w_45_0  (ctx : parallel_generatedRoot_985Ctx) : Fin 1 → mxxMat5 :=
  ctx.inputs.2.2.2.2.2.2.2.1

def parallel_generatedRoot_985.w_46_0  (ctx : parallel_generatedRoot_985Ctx) : mxxMat9 :=
  ctx.inputs.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_985.w_47_0  (ctx : parallel_generatedRoot_985Ctx) : mxxMat9 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_985.w_50_0  (ctx : parallel_generatedRoot_985Ctx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_985.w_51_0  (ctx : parallel_generatedRoot_985Ctx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_985.w_58_0  (ctx : parallel_generatedRoot_985Ctx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_985.w_74_0  (ctx : parallel_generatedRoot_985Ctx) : mxxMat7 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_985.w_2_0  (ctx : parallel_generatedRoot_985Ctx) : mxxMat11 :=
  ctx.witness.w_2_0

def parallel_generatedRoot_985.w_3_0  (ctx : parallel_generatedRoot_985Ctx) : mxxMat5 :=
  (MxxRuntime.matrixPolynomial [(1606938044254565807694577896670997561277593686540426466090209)] : mxxMat5)

def parallel_generatedRoot_985.w_5_0  (ctx : parallel_generatedRoot_985Ctx) : mxxMat2 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_985.w_3_0 ctx) (parallel_generatedRoot_985.w_4_0 ctx)

def parallel_generatedRoot_985.w_6_0  (ctx : parallel_generatedRoot_985Ctx) : mxxMat4 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_985.w_7_0  (ctx : parallel_generatedRoot_985Ctx) : mxxMat4 :=
  ctx.witness.w_7_0

def parallel_generatedRoot_985.w_8_0  (ctx : parallel_generatedRoot_985Ctx) : mxxMat0 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_985.w_2_0 ctx) (parallel_generatedRoot_985.w_7_0 ctx)

def parallel_generatedRoot_985.w_10_0  (ctx : parallel_generatedRoot_985Ctx) : mxxMat0 :=
  ctx.witness.w_10_0

def parallel_generatedRoot_985.w_11_0  (ctx : parallel_generatedRoot_985Ctx) : mxxMat0 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_985.w_12_0  (ctx : parallel_generatedRoot_985Ctx) : mxxMat18 :=
  ctx.witness.w_12_0

def parallel_generatedRoot_985.w_13_0  (ctx : parallel_generatedRoot_985Ctx) : mxxMat19 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat19)

def parallel_generatedRoot_985.w_14_0  (ctx : parallel_generatedRoot_985Ctx) : mxxMat18 :=
  MxxRuntime.matrixMulScalarRight (parallel_generatedRoot_985.w_12_0 ctx) (parallel_generatedRoot_985.w_13_0 ctx)

def parallel_generatedRoot_985.w_15_0  (ctx : parallel_generatedRoot_985Ctx) : mxxMat19 :=
  (MxxRuntime.matrixPolynomial [(1125894775373377)] : mxxMat19)

def parallel_generatedRoot_985.w_16_0  (ctx : parallel_generatedRoot_985Ctx) : mxxMat18 :=
  MxxRuntime.matrixMulScalarRight (parallel_generatedRoot_985.w_14_0 ctx) (parallel_generatedRoot_985.w_15_0 ctx)

def parallel_generatedRoot_985.w_17_0  (ctx : parallel_generatedRoot_985Ctx) : mxxMat0 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_985.w_18_0  (ctx : parallel_generatedRoot_985Ctx) : mxxMat5 :=
  (MxxRuntime.matrixPolynomial [(mxxNat1)] : mxxMat5)

def parallel_generatedRoot_985.w_19_0  (ctx : parallel_generatedRoot_985Ctx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarRight (parallel_generatedRoot_985.w_17_0 ctx) (parallel_generatedRoot_985.w_18_0 ctx)

def parallel_generatedRoot_985.w_20_0  (ctx : parallel_generatedRoot_985Ctx) : mxxMat6 :=
  ctx.witness.w_20_0

def parallel_generatedRoot_985.w_21_0  (ctx : parallel_generatedRoot_985Ctx) : mxxMat6 :=
  ctx.witness.w_21_0

def parallel_generatedRoot_985.w_22_0  (ctx : parallel_generatedRoot_985Ctx) : mxxMat0 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_985.w_9_0 ctx) (parallel_generatedRoot_985.w_21_0 ctx)

def parallel_generatedRoot_985.w_23_0  (ctx : parallel_generatedRoot_985Ctx) : mxxMat0 :=
  MxxRuntime.matrixSub (parallel_generatedRoot_985.w_8_0 ctx) (parallel_generatedRoot_985.w_22_0 ctx)

def parallel_generatedRoot_985.w_24_0  (ctx : parallel_generatedRoot_985Ctx) : mxxMat0 :=
  (MxxRuntime.unitRow (0) : mxxMat0)

def parallel_generatedRoot_985.w_26_0  (ctx : parallel_generatedRoot_985Ctx) : mxxMat14 :=
  ctx.witness.w_26_0

def parallel_generatedRoot_985.w_28_0  (ctx : parallel_generatedRoot_985Ctx) : mxxMat15 :=
  ctx.witness.w_28_0

def parallel_generatedRoot_985.w_29_0  (ctx : parallel_generatedRoot_985Ctx) : mxxMat15 :=
  ctx.witness.w_29_0

def parallel_generatedRoot_985.w_30_0  (ctx : parallel_generatedRoot_985Ctx) : mxxMat5 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat5)

def parallel_generatedRoot_985.w_31_0  (ctx : parallel_generatedRoot_985Ctx) : mxxMat15 :=
  MxxRuntime.matrixMulScalarRight (parallel_generatedRoot_985.w_29_0 ctx) (parallel_generatedRoot_985.w_30_0 ctx)

def parallel_generatedRoot_985.w_32_0  (ctx : parallel_generatedRoot_985Ctx) : mxxMat16 :=
  ctx.witness.w_32_0

def parallel_generatedRoot_985.w_33_0  (ctx : parallel_generatedRoot_985Ctx) : mxxMat16 :=
  ctx.witness.w_33_0

def parallel_generatedRoot_985.w_34_0  (ctx : parallel_generatedRoot_985Ctx) : mxxMat14 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_985.w_26_0 ctx) (parallel_generatedRoot_985.w_33_0 ctx)

def parallel_generatedRoot_985.w_35_0  (ctx : parallel_generatedRoot_985Ctx) : mxxMat0 :=
  ctx.witness.w_35_0

def parallel_generatedRoot_985.w_36_0  (ctx : parallel_generatedRoot_985Ctx) : mxxMat5 :=
  (MxxRuntime.matrixPolynomial [(1606938044254565807694577896670997561277593686540426466090209)] : mxxMat5)

def parallel_generatedRoot_985.w_37_0  (ctx : parallel_generatedRoot_985Ctx) : mxxMat17 :=
  ctx.witness.w_37_0

def parallel_generatedRoot_985.w_38_0  (ctx : parallel_generatedRoot_985Ctx) : mxxMat17 :=
  ctx.witness.w_38_0

def parallel_generatedRoot_985.w_39_0  (ctx : parallel_generatedRoot_985Ctx) : mxxMat5 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_985.w_35_0 ctx) (parallel_generatedRoot_985.w_38_0 ctx)

def parallel_generatedRoot_985.w_40_0  (ctx : parallel_generatedRoot_985Ctx) : mxxMat0 :=
  ctx.witness.w_40_0

def parallel_generatedRoot_985.w_41_0  (ctx : parallel_generatedRoot_985Ctx) : mxxMat0 :=
  ctx.witness.w_41_0

def parallel_generatedRoot_985.w_42_0  (ctx : parallel_generatedRoot_985Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_985.w_23_0 ctx) (parallel_generatedRoot_985.w_41_0 ctx)

def parallel_generatedRoot_985.w_43_0  (ctx : parallel_generatedRoot_985Ctx) : mxxMat0 :=
  (0 : mxxMat0)

def parallel_generatedRoot_985.w_48_0  (ctx : parallel_generatedRoot_985Ctx) : mxxMat0 :=
  ctx.witness.w_48_0

def parallel_generatedRoot_985.w_48_1  (ctx : parallel_generatedRoot_985Ctx) : mxxMat0 :=
  ctx.witness.w_48_1

def parallel_generatedRoot_985.w_48_2  (ctx : parallel_generatedRoot_985Ctx) : mxxMat0 :=
  ctx.witness.w_48_2

def parallel_generatedRoot_985.w_49_0  (ctx : parallel_generatedRoot_985Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_985.w_42_0 ctx) (parallel_generatedRoot_985.w_48_1 ctx)

def parallel_generatedRoot_985.w_52_0  (ctx : parallel_generatedRoot_985Ctx) : mxxMat11 :=
  ctx.witness.w_52_0

def parallel_generatedRoot_985.w_53_0  (ctx : parallel_generatedRoot_985Ctx) : mxxMat5 :=
  (MxxRuntime.matrixPolynomial [(1606938044254565807694577896670997561277593686540426466090209)] : mxxMat5)

def parallel_generatedRoot_985.w_54_0  (ctx : parallel_generatedRoot_985Ctx) : mxxMat2 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_985.w_53_0 ctx) (parallel_generatedRoot_985.w_4_0 ctx)

def parallel_generatedRoot_985.w_55_0  (ctx : parallel_generatedRoot_985Ctx) : mxxMat4 :=
  ctx.witness.w_55_0

def parallel_generatedRoot_985.w_56_0  (ctx : parallel_generatedRoot_985Ctx) : mxxMat4 :=
  ctx.witness.w_56_0

def parallel_generatedRoot_985.w_57_0  (ctx : parallel_generatedRoot_985Ctx) : mxxMat0 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_985.w_52_0 ctx) (parallel_generatedRoot_985.w_56_0 ctx)

def parallel_generatedRoot_985.w_59_0  (ctx : parallel_generatedRoot_985Ctx) : mxxMat0 :=
  ctx.witness.w_59_0

def parallel_generatedRoot_985.w_60_0  (ctx : parallel_generatedRoot_985Ctx) : mxxMat0 :=
  ctx.witness.w_60_0

def parallel_generatedRoot_985.w_61_0  (ctx : parallel_generatedRoot_985Ctx) : mxxMat18 :=
  ctx.witness.w_61_0

def parallel_generatedRoot_985.w_62_0  (ctx : parallel_generatedRoot_985Ctx) : mxxMat19 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat19)

def parallel_generatedRoot_985.w_63_0  (ctx : parallel_generatedRoot_985Ctx) : mxxMat18 :=
  MxxRuntime.matrixMulScalarRight (parallel_generatedRoot_985.w_61_0 ctx) (parallel_generatedRoot_985.w_62_0 ctx)

def parallel_generatedRoot_985.w_64_0  (ctx : parallel_generatedRoot_985Ctx) : mxxMat19 :=
  (MxxRuntime.matrixPolynomial [(1125894775373377)] : mxxMat19)

def parallel_generatedRoot_985.w_65_0  (ctx : parallel_generatedRoot_985Ctx) : mxxMat18 :=
  MxxRuntime.matrixMulScalarRight (parallel_generatedRoot_985.w_63_0 ctx) (parallel_generatedRoot_985.w_64_0 ctx)

def parallel_generatedRoot_985.w_66_0  (ctx : parallel_generatedRoot_985Ctx) : mxxMat0 :=
  ctx.witness.w_66_0

def parallel_generatedRoot_985.w_67_0  (ctx : parallel_generatedRoot_985Ctx) : mxxMat5 :=
  (MxxRuntime.matrixPolynomial [(mxxNat1)] : mxxMat5)

def parallel_generatedRoot_985.w_68_0  (ctx : parallel_generatedRoot_985Ctx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarRight (parallel_generatedRoot_985.w_66_0 ctx) (parallel_generatedRoot_985.w_67_0 ctx)

def parallel_generatedRoot_985.w_69_0  (ctx : parallel_generatedRoot_985Ctx) : mxxMat6 :=
  ctx.witness.w_69_0

def parallel_generatedRoot_985.w_70_0  (ctx : parallel_generatedRoot_985Ctx) : mxxMat6 :=
  ctx.witness.w_70_0

def parallel_generatedRoot_985.w_71_0  (ctx : parallel_generatedRoot_985Ctx) : mxxMat0 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_985.w_58_0 ctx) (parallel_generatedRoot_985.w_70_0 ctx)

def parallel_generatedRoot_985.w_72_0  (ctx : parallel_generatedRoot_985Ctx) : mxxMat0 :=
  MxxRuntime.matrixSub (parallel_generatedRoot_985.w_57_0 ctx) (parallel_generatedRoot_985.w_71_0 ctx)

def parallel_generatedRoot_985.w_73_0  (ctx : parallel_generatedRoot_985Ctx) : mxxMat0 :=
  (MxxRuntime.unitRow (0) : mxxMat0)

def parallel_generatedRoot_985.w_75_0  (ctx : parallel_generatedRoot_985Ctx) : mxxMat14 :=
  ctx.witness.w_75_0

def parallel_generatedRoot_985.w_76_0  (ctx : parallel_generatedRoot_985Ctx) : mxxMat15 :=
  ctx.witness.w_76_0

def parallel_generatedRoot_985.w_77_0  (ctx : parallel_generatedRoot_985Ctx) : mxxMat15 :=
  ctx.witness.w_77_0

def parallel_generatedRoot_985.w_78_0  (ctx : parallel_generatedRoot_985Ctx) : mxxMat5 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat5)

def parallel_generatedRoot_985.w_79_0  (ctx : parallel_generatedRoot_985Ctx) : mxxMat15 :=
  MxxRuntime.matrixMulScalarRight (parallel_generatedRoot_985.w_77_0 ctx) (parallel_generatedRoot_985.w_78_0 ctx)

def parallel_generatedRoot_985.w_80_0  (ctx : parallel_generatedRoot_985Ctx) : mxxMat16 :=
  ctx.witness.w_80_0

def parallel_generatedRoot_985.w_81_0  (ctx : parallel_generatedRoot_985Ctx) : mxxMat16 :=
  ctx.witness.w_81_0

def parallel_generatedRoot_985.w_82_0  (ctx : parallel_generatedRoot_985Ctx) : mxxMat14 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_985.w_75_0 ctx) (parallel_generatedRoot_985.w_81_0 ctx)

def parallel_generatedRoot_985.w_83_0  (ctx : parallel_generatedRoot_985Ctx) : mxxMat0 :=
  ctx.witness.w_83_0

def parallel_generatedRoot_985.w_84_0  (ctx : parallel_generatedRoot_985Ctx) : mxxMat5 :=
  (MxxRuntime.matrixPolynomial [(1606938044254565807694577896670997561277593686540426466090209)] : mxxMat5)

def parallel_generatedRoot_985.w_85_0  (ctx : parallel_generatedRoot_985Ctx) : mxxMat17 :=
  ctx.witness.w_85_0

def parallel_generatedRoot_985.w_86_0  (ctx : parallel_generatedRoot_985Ctx) : mxxMat17 :=
  ctx.witness.w_86_0

def parallel_generatedRoot_985.w_87_0  (ctx : parallel_generatedRoot_985Ctx) : mxxMat5 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_985.w_83_0 ctx) (parallel_generatedRoot_985.w_86_0 ctx)

def parallel_generatedRoot_985.w_88_0  (ctx : parallel_generatedRoot_985Ctx) : mxxMat0 :=
  ctx.witness.w_88_0

def parallel_generatedRoot_985.w_89_0  (ctx : parallel_generatedRoot_985Ctx) : mxxMat0 :=
  ctx.witness.w_89_0

def parallel_generatedRoot_985.w_90_0  (ctx : parallel_generatedRoot_985Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_985.w_72_0 ctx) (parallel_generatedRoot_985.w_89_0 ctx)

def parallel_generatedRoot_985.w_91_0  (ctx : parallel_generatedRoot_985Ctx) : mxxMat0 :=
  (0 : mxxMat0)

def parallel_generatedRoot_985.w_92_0  (ctx : parallel_generatedRoot_985Ctx) : mxxMat0 :=
  ctx.witness.w_92_0

def parallel_generatedRoot_985.w_92_1  (ctx : parallel_generatedRoot_985Ctx) : mxxMat0 :=
  ctx.witness.w_92_1

def parallel_generatedRoot_985.w_92_2  (ctx : parallel_generatedRoot_985Ctx) : mxxMat0 :=
  ctx.witness.w_92_2

def parallel_generatedRoot_985.w_93_0  (ctx : parallel_generatedRoot_985Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_985.w_90_0 ctx) (parallel_generatedRoot_985.w_92_2 ctx)

def parallel_generatedRoot_985.w_94_0  (ctx : parallel_generatedRoot_985Ctx) : mxxMat2 :=
  ctx.witness.w_94_0

def parallel_generatedRoot_985.outs  (ctx : parallel_generatedRoot_985Ctx) : mxxMat2 :=
  (parallel_generatedRoot_985.w_94_0 ctx)

structure parallel_generatedRoot_985.Facts_0  (ctx : parallel_generatedRoot_985Ctx) : Prop where
  h2_0 : MxxRuntime.concatColumns (parallel_generatedRoot_985.w_0_0 ctx) (parallel_generatedRoot_985.w_1_0 ctx) (parallel_generatedRoot_985.w_2_0 ctx)
  h6_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (parallel_generatedRoot_985.w_5_0 ctx) (parallel_generatedRoot_985.w_6_0 ctx)
  h7_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (parallel_generatedRoot_985.w_6_0 ctx) (parallel_generatedRoot_985.w_7_0 ctx)
  h10_0 : MxxRuntime.gadgetMatrixRuns backend 4096 25 (parallel_generatedRoot_985.w_10_0 ctx)
  h11_0 : MxxRuntime.centeredRebaseRuns (parallel_generatedRoot_985.w_10_0 ctx) (parallel_generatedRoot_985.w_11_0 ctx)
  h12_0 : MxxRuntime.modulusReduceRuns (parallel_generatedRoot_985.w_11_0 ctx) (parallel_generatedRoot_985.w_12_0 ctx)
  h17_0 : MxxRuntime.centeredRebaseRuns (parallel_generatedRoot_985.w_16_0 ctx) (parallel_generatedRoot_985.w_17_0 ctx)
  h20_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (parallel_generatedRoot_985.w_19_0 ctx) (parallel_generatedRoot_985.w_20_0 ctx)
  h21_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (parallel_generatedRoot_985.w_20_0 ctx) (parallel_generatedRoot_985.w_21_0 ctx)
  t9 : 0 ≤ (0)
  t10 : (0) < 25
  t11 : 0 ≤ 0
  t12 : 0 < 1
  t13 : 1 ≤ 1
  t14 : 0 ≤ (625 * (Int.ofNat i_0))
  t15 : (625 * (Int.ofNat i_0)) < (625 + (625 * (Int.ofNat i_0)))
  t16 : (625 + (625 * (Int.ofNat i_0))) ≤ 22500
  h26_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_985.w_25_0 ctx) 0 1 (625 * (Int.ofNat i_0)) (625 + (625 * (Int.ofNat i_0))) (parallel_generatedRoot_985.w_26_0 ctx)
  h28_0 : MxxRuntime.blockDiagonalRuns 25 (parallel_generatedRoot_985.w_27_0 ctx) (parallel_generatedRoot_985.w_28_0 ctx)
  h29_0 : MxxRuntime.centeredRebaseRuns (parallel_generatedRoot_985.w_28_0 ctx) (parallel_generatedRoot_985.w_29_0 ctx)
  h32_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (parallel_generatedRoot_985.w_31_0 ctx) (parallel_generatedRoot_985.w_32_0 ctx)
  h33_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (parallel_generatedRoot_985.w_32_0 ctx) (parallel_generatedRoot_985.w_33_0 ctx)
  t22 : 0 ≤ 0
  t23 : 0 < 1
  t24 : 1 ≤ 1
  t25 : 0 ≤ 0
  t26 : 0 < 25
  t27 : 25 ≤ 625
  h35_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_985.w_34_0 ctx) 0 1 0 25 (parallel_generatedRoot_985.w_35_0 ctx)
  h37_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (parallel_generatedRoot_985.w_36_0 ctx) (parallel_generatedRoot_985.w_37_0 ctx)
  h38_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (parallel_generatedRoot_985.w_37_0 ctx) (parallel_generatedRoot_985.w_38_0 ctx)
  h40_0 : MxxRuntime.tensorRuns (parallel_generatedRoot_985.w_24_0 ctx) (parallel_generatedRoot_985.w_39_0 ctx) (parallel_generatedRoot_985.w_40_0 ctx)

theorem parallel_generatedRoot_985.facts_0  {ctx : parallel_generatedRoot_985Ctx} {outputs : mxxMat2}
    (h : parallel_generatedRoot_985.constraints_0 backend i_0 (parallel_generatedRoot_985.w_0_0 ctx) (parallel_generatedRoot_985.w_1_0 ctx) (parallel_generatedRoot_985.w_25_0 ctx) (parallel_generatedRoot_985.w_27_0 ctx) (parallel_generatedRoot_985.w_2_0 ctx) (parallel_generatedRoot_985.w_5_0 ctx) (parallel_generatedRoot_985.w_6_0 ctx) (parallel_generatedRoot_985.w_7_0 ctx) (parallel_generatedRoot_985.w_10_0 ctx) (parallel_generatedRoot_985.w_11_0 ctx) (parallel_generatedRoot_985.w_12_0 ctx) (parallel_generatedRoot_985.w_16_0 ctx) (parallel_generatedRoot_985.w_17_0 ctx) (parallel_generatedRoot_985.w_19_0 ctx) (parallel_generatedRoot_985.w_20_0 ctx) (parallel_generatedRoot_985.w_21_0 ctx) (parallel_generatedRoot_985.w_24_0 ctx) (parallel_generatedRoot_985.w_26_0 ctx) (parallel_generatedRoot_985.w_28_0 ctx) (parallel_generatedRoot_985.w_29_0 ctx) (parallel_generatedRoot_985.w_31_0 ctx) (parallel_generatedRoot_985.w_32_0 ctx) (parallel_generatedRoot_985.w_33_0 ctx) (parallel_generatedRoot_985.w_34_0 ctx) (parallel_generatedRoot_985.w_35_0 ctx) (parallel_generatedRoot_985.w_36_0 ctx) (parallel_generatedRoot_985.w_37_0 ctx) (parallel_generatedRoot_985.w_38_0 ctx) (parallel_generatedRoot_985.w_39_0 ctx) (parallel_generatedRoot_985.w_40_0 ctx)) :
    parallel_generatedRoot_985.Facts_0 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_985.Facts_1  (ctx : parallel_generatedRoot_985Ctx) : Prop where
  t32 : 0 ≤ (24)
  h41_0 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0) => sequential_parallel_generatedRoot_985_41 backend hashModel tape (path ++ [41, i]) params i_0 i (current, ((parallel_generatedRoot_985.w_34_0 ctx), ((parallel_generatedRoot_985.w_38_0 ctx), ()))) next) (Int.toNat (24)) (parallel_generatedRoot_985.w_40_0 ctx) (parallel_generatedRoot_985.w_41_0 ctx)
  t34 : 0 ≤ (1)
  h48_2 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_985_48 backend hashModel tape (path ++ [48, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, ((parallel_generatedRoot_985.w_44_0 ctx), ((parallel_generatedRoot_985.w_45_0 ctx), ((parallel_generatedRoot_985.w_46_0 ctx), ((parallel_generatedRoot_985.w_47_0 ctx), ()))))))) next) (Int.toNat (1)) ((parallel_generatedRoot_985.w_43_0 ctx), ((parallel_generatedRoot_985.w_43_0 ctx), ((parallel_generatedRoot_985.w_43_0 ctx), ()))) ((parallel_generatedRoot_985.w_48_0 ctx), ((parallel_generatedRoot_985.w_48_1 ctx), ((parallel_generatedRoot_985.w_48_2 ctx), ())))
  h52_0 : MxxRuntime.concatColumns (parallel_generatedRoot_985.w_50_0 ctx) (parallel_generatedRoot_985.w_51_0 ctx) (parallel_generatedRoot_985.w_52_0 ctx)
  h55_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (parallel_generatedRoot_985.w_54_0 ctx) (parallel_generatedRoot_985.w_55_0 ctx)
  h56_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (parallel_generatedRoot_985.w_55_0 ctx) (parallel_generatedRoot_985.w_56_0 ctx)
  h59_0 : MxxRuntime.gadgetMatrixRuns backend 4096 25 (parallel_generatedRoot_985.w_59_0 ctx)
  h60_0 : MxxRuntime.centeredRebaseRuns (parallel_generatedRoot_985.w_59_0 ctx) (parallel_generatedRoot_985.w_60_0 ctx)
  h61_0 : MxxRuntime.modulusReduceRuns (parallel_generatedRoot_985.w_60_0 ctx) (parallel_generatedRoot_985.w_61_0 ctx)
  h66_0 : MxxRuntime.centeredRebaseRuns (parallel_generatedRoot_985.w_65_0 ctx) (parallel_generatedRoot_985.w_66_0 ctx)
  h69_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (parallel_generatedRoot_985.w_68_0 ctx) (parallel_generatedRoot_985.w_69_0 ctx)
  h70_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (parallel_generatedRoot_985.w_69_0 ctx) (parallel_generatedRoot_985.w_70_0 ctx)
  t45 : 0 ≤ (0)
  t46 : (0) < 25
  t47 : 0 ≤ 0
  t48 : 0 < 1
  t49 : 1 ≤ 1
  t50 : 0 ≤ (625 * (Int.ofNat i_0))
  t51 : (625 * (Int.ofNat i_0)) < (625 + (625 * (Int.ofNat i_0)))
  t52 : (625 + (625 * (Int.ofNat i_0))) ≤ 22500
  h75_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_985.w_74_0 ctx) 0 1 (625 * (Int.ofNat i_0)) (625 + (625 * (Int.ofNat i_0))) (parallel_generatedRoot_985.w_75_0 ctx)
  h76_0 : MxxRuntime.blockDiagonalRuns 25 (parallel_generatedRoot_985.w_27_0 ctx) (parallel_generatedRoot_985.w_76_0 ctx)
  h77_0 : MxxRuntime.centeredRebaseRuns (parallel_generatedRoot_985.w_76_0 ctx) (parallel_generatedRoot_985.w_77_0 ctx)
  h80_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (parallel_generatedRoot_985.w_79_0 ctx) (parallel_generatedRoot_985.w_80_0 ctx)
  h81_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (parallel_generatedRoot_985.w_80_0 ctx) (parallel_generatedRoot_985.w_81_0 ctx)
  t58 : 0 ≤ 0
  t59 : 0 < 1
  t60 : 1 ≤ 1
  t61 : 0 ≤ 0
  t62 : 0 < 25
  t63 : 25 ≤ 625

theorem parallel_generatedRoot_985.facts_1  {ctx : parallel_generatedRoot_985Ctx} {outputs : mxxMat2}
    (h : parallel_generatedRoot_985.constraints_1 backend hashModel tape path params i_0 (parallel_generatedRoot_985.w_27_0 ctx) (parallel_generatedRoot_985.w_44_0 ctx) (parallel_generatedRoot_985.w_45_0 ctx) (parallel_generatedRoot_985.w_46_0 ctx) (parallel_generatedRoot_985.w_47_0 ctx) (parallel_generatedRoot_985.w_50_0 ctx) (parallel_generatedRoot_985.w_51_0 ctx) (parallel_generatedRoot_985.w_74_0 ctx) (parallel_generatedRoot_985.w_34_0 ctx) (parallel_generatedRoot_985.w_38_0 ctx) (parallel_generatedRoot_985.w_40_0 ctx) (parallel_generatedRoot_985.w_41_0 ctx) (parallel_generatedRoot_985.w_43_0 ctx) (parallel_generatedRoot_985.w_48_0 ctx) (parallel_generatedRoot_985.w_48_1 ctx) (parallel_generatedRoot_985.w_48_2 ctx) (parallel_generatedRoot_985.w_52_0 ctx) (parallel_generatedRoot_985.w_54_0 ctx) (parallel_generatedRoot_985.w_55_0 ctx) (parallel_generatedRoot_985.w_56_0 ctx) (parallel_generatedRoot_985.w_59_0 ctx) (parallel_generatedRoot_985.w_60_0 ctx) (parallel_generatedRoot_985.w_61_0 ctx) (parallel_generatedRoot_985.w_65_0 ctx) (parallel_generatedRoot_985.w_66_0 ctx) (parallel_generatedRoot_985.w_68_0 ctx) (parallel_generatedRoot_985.w_69_0 ctx) (parallel_generatedRoot_985.w_70_0 ctx) (parallel_generatedRoot_985.w_75_0 ctx) (parallel_generatedRoot_985.w_76_0 ctx) (parallel_generatedRoot_985.w_77_0 ctx) (parallel_generatedRoot_985.w_79_0 ctx) (parallel_generatedRoot_985.w_80_0 ctx) (parallel_generatedRoot_985.w_81_0 ctx)) :
    parallel_generatedRoot_985.Facts_1 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_985.Facts_2  (ctx : parallel_generatedRoot_985Ctx) (outputs : mxxMat2) : Prop where
  h83_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_985.w_82_0 ctx) 0 1 0 25 (parallel_generatedRoot_985.w_83_0 ctx)
  h85_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (parallel_generatedRoot_985.w_84_0 ctx) (parallel_generatedRoot_985.w_85_0 ctx)
  h86_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (parallel_generatedRoot_985.w_85_0 ctx) (parallel_generatedRoot_985.w_86_0 ctx)
  h88_0 : MxxRuntime.tensorRuns (parallel_generatedRoot_985.w_73_0 ctx) (parallel_generatedRoot_985.w_87_0 ctx) (parallel_generatedRoot_985.w_88_0 ctx)
  t68 : 0 ≤ (24)
  h89_0 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0) => sequential_parallel_generatedRoot_985_89 backend hashModel tape (path ++ [89, i]) params i_0 i (current, ((parallel_generatedRoot_985.w_82_0 ctx), ((parallel_generatedRoot_985.w_86_0 ctx), ()))) next) (Int.toNat (24)) (parallel_generatedRoot_985.w_88_0 ctx) (parallel_generatedRoot_985.w_89_0 ctx)
  t70 : 0 ≤ (1)
  h92_2 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_985_92 backend hashModel tape (path ++ [92, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, ((parallel_generatedRoot_985.w_44_0 ctx), ((parallel_generatedRoot_985.w_45_0 ctx), ((parallel_generatedRoot_985.w_46_0 ctx), ((parallel_generatedRoot_985.w_47_0 ctx), ()))))))) next) (Int.toNat (1)) ((parallel_generatedRoot_985.w_91_0 ctx), ((parallel_generatedRoot_985.w_91_0 ctx), ((parallel_generatedRoot_985.w_91_0 ctx), ()))) ((parallel_generatedRoot_985.w_92_0 ctx), ((parallel_generatedRoot_985.w_92_1 ctx), ((parallel_generatedRoot_985.w_92_2 ctx), ())))
  h94_0 : MxxRuntime.concatRows (parallel_generatedRoot_985.w_49_0 ctx) (parallel_generatedRoot_985.w_93_0 ctx) (parallel_generatedRoot_985.w_94_0 ctx)
  hout : outputs = parallel_generatedRoot_985.outs ctx

theorem parallel_generatedRoot_985.facts_2  {ctx : parallel_generatedRoot_985Ctx} {outputs : mxxMat2}
    (h : parallel_generatedRoot_985.constraints_2 backend hashModel tape path params i_0 (parallel_generatedRoot_985.w_44_0 ctx) (parallel_generatedRoot_985.w_45_0 ctx) (parallel_generatedRoot_985.w_46_0 ctx) (parallel_generatedRoot_985.w_47_0 ctx) (parallel_generatedRoot_985.w_49_0 ctx) (parallel_generatedRoot_985.w_73_0 ctx) (parallel_generatedRoot_985.w_82_0 ctx) (parallel_generatedRoot_985.w_83_0 ctx) (parallel_generatedRoot_985.w_84_0 ctx) (parallel_generatedRoot_985.w_85_0 ctx) (parallel_generatedRoot_985.w_86_0 ctx) (parallel_generatedRoot_985.w_87_0 ctx) (parallel_generatedRoot_985.w_88_0 ctx) (parallel_generatedRoot_985.w_89_0 ctx) (parallel_generatedRoot_985.w_91_0 ctx) (parallel_generatedRoot_985.w_92_0 ctx) (parallel_generatedRoot_985.w_92_1 ctx) (parallel_generatedRoot_985.w_92_2 ctx) (parallel_generatedRoot_985.w_93_0 ctx) (parallel_generatedRoot_985.w_94_0 ctx) outputs) :
    parallel_generatedRoot_985.Facts_2 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_985.Facts  (ctx : parallel_generatedRoot_985Ctx) (outputs : mxxMat2) : Prop where
  c0 : parallel_generatedRoot_985.Facts_0 ctx
  c1 : parallel_generatedRoot_985.Facts_1 ctx
  c2 : parallel_generatedRoot_985.Facts_2 ctx outputs

theorem parallel_generatedRoot_985.facts  (ctx : parallel_generatedRoot_985Ctx) {outputs : mxxMat2}
    (h : parallel_generatedRoot_985.body backend hashModel tape path params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_985.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_985.facts_0 (ctx := ctx) (outputs := outputs) t0.1
  have t1 := t0.2
  have r1 := parallel_generatedRoot_985.facts_1 (ctx := ctx) (outputs := outputs) t1.1
  have t2 := t1.2
  have r2 := parallel_generatedRoot_985.facts_2 (ctx := ctx) (outputs := outputs) t2
  ⟨r0, r1, r2⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_986.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) where
  inputs : (parallel_generatedRoot_986.InputType)
  witness : parallel_generatedRoot_986.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_986Ctx" => (parallel_generatedRoot_986.Ctx backend hashModel tape path params i_0)

def parallel_generatedRoot_986.w_0_0  (ctx : parallel_generatedRoot_986Ctx) : mxxMat0 :=
  ctx.inputs.1

def parallel_generatedRoot_986.w_1_0  (ctx : parallel_generatedRoot_986Ctx) : mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_986.w_4_0  (ctx : parallel_generatedRoot_986Ctx) : mxxMat2 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_986.w_9_0  (ctx : parallel_generatedRoot_986Ctx) : mxxMat0 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_986.w_25_0  (ctx : parallel_generatedRoot_986Ctx) : mxxMat7 :=
  ctx.inputs.2.2.2.2.1

def parallel_generatedRoot_986.w_27_0  (ctx : parallel_generatedRoot_986Ctx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.1

def parallel_generatedRoot_986.w_44_0  (ctx : parallel_generatedRoot_986Ctx) : mxxMat9 :=
  ctx.inputs.2.2.2.2.2.2.1

def parallel_generatedRoot_986.w_45_0  (ctx : parallel_generatedRoot_986Ctx) : Fin 1 → mxxMat5 :=
  ctx.inputs.2.2.2.2.2.2.2.1

def parallel_generatedRoot_986.w_46_0  (ctx : parallel_generatedRoot_986Ctx) : mxxMat9 :=
  ctx.inputs.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_986.w_47_0  (ctx : parallel_generatedRoot_986Ctx) : mxxMat9 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_986.w_50_0  (ctx : parallel_generatedRoot_986Ctx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_986.w_51_0  (ctx : parallel_generatedRoot_986Ctx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_986.w_58_0  (ctx : parallel_generatedRoot_986Ctx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_986.w_74_0  (ctx : parallel_generatedRoot_986Ctx) : mxxMat7 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_986.w_2_0  (ctx : parallel_generatedRoot_986Ctx) : mxxMat11 :=
  ctx.witness.w_2_0

def parallel_generatedRoot_986.w_3_0  (ctx : parallel_generatedRoot_986Ctx) : mxxMat5 :=
  (MxxRuntime.matrixPolynomial [(1606938044254771331362327096896778694705812889646799470123121)] : mxxMat5)

def parallel_generatedRoot_986.w_5_0  (ctx : parallel_generatedRoot_986Ctx) : mxxMat2 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_986.w_3_0 ctx) (parallel_generatedRoot_986.w_4_0 ctx)

def parallel_generatedRoot_986.w_6_0  (ctx : parallel_generatedRoot_986Ctx) : mxxMat4 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_986.w_7_0  (ctx : parallel_generatedRoot_986Ctx) : mxxMat4 :=
  ctx.witness.w_7_0

def parallel_generatedRoot_986.w_8_0  (ctx : parallel_generatedRoot_986Ctx) : mxxMat0 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_986.w_2_0 ctx) (parallel_generatedRoot_986.w_7_0 ctx)

def parallel_generatedRoot_986.w_10_0  (ctx : parallel_generatedRoot_986Ctx) : mxxMat0 :=
  ctx.witness.w_10_0

def parallel_generatedRoot_986.w_11_0  (ctx : parallel_generatedRoot_986Ctx) : mxxMat0 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_986.w_12_0  (ctx : parallel_generatedRoot_986Ctx) : mxxMat20 :=
  ctx.witness.w_12_0

def parallel_generatedRoot_986.w_13_0  (ctx : parallel_generatedRoot_986Ctx) : mxxMat21 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat21)

def parallel_generatedRoot_986.w_14_0  (ctx : parallel_generatedRoot_986Ctx) : mxxMat20 :=
  MxxRuntime.matrixMulScalarRight (parallel_generatedRoot_986.w_12_0 ctx) (parallel_generatedRoot_986.w_13_0 ctx)

def parallel_generatedRoot_986.w_15_0  (ctx : parallel_generatedRoot_986Ctx) : mxxMat21 :=
  (MxxRuntime.matrixPolynomial [(7254835200)] : mxxMat21)

def parallel_generatedRoot_986.w_16_0  (ctx : parallel_generatedRoot_986Ctx) : mxxMat20 :=
  MxxRuntime.matrixMulScalarRight (parallel_generatedRoot_986.w_14_0 ctx) (parallel_generatedRoot_986.w_15_0 ctx)

def parallel_generatedRoot_986.w_17_0  (ctx : parallel_generatedRoot_986Ctx) : mxxMat0 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_986.w_18_0  (ctx : parallel_generatedRoot_986Ctx) : mxxMat5 :=
  (MxxRuntime.matrixPolynomial [(mxxNat2)] : mxxMat5)

def parallel_generatedRoot_986.w_19_0  (ctx : parallel_generatedRoot_986Ctx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarRight (parallel_generatedRoot_986.w_17_0 ctx) (parallel_generatedRoot_986.w_18_0 ctx)

def parallel_generatedRoot_986.w_20_0  (ctx : parallel_generatedRoot_986Ctx) : mxxMat6 :=
  ctx.witness.w_20_0

def parallel_generatedRoot_986.w_21_0  (ctx : parallel_generatedRoot_986Ctx) : mxxMat6 :=
  ctx.witness.w_21_0

def parallel_generatedRoot_986.w_22_0  (ctx : parallel_generatedRoot_986Ctx) : mxxMat0 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_986.w_9_0 ctx) (parallel_generatedRoot_986.w_21_0 ctx)

def parallel_generatedRoot_986.w_23_0  (ctx : parallel_generatedRoot_986Ctx) : mxxMat0 :=
  MxxRuntime.matrixSub (parallel_generatedRoot_986.w_8_0 ctx) (parallel_generatedRoot_986.w_22_0 ctx)

def parallel_generatedRoot_986.w_24_0  (ctx : parallel_generatedRoot_986Ctx) : mxxMat0 :=
  (MxxRuntime.unitRow (0) : mxxMat0)

def parallel_generatedRoot_986.w_26_0  (ctx : parallel_generatedRoot_986Ctx) : mxxMat14 :=
  ctx.witness.w_26_0

def parallel_generatedRoot_986.w_28_0  (ctx : parallel_generatedRoot_986Ctx) : mxxMat15 :=
  ctx.witness.w_28_0

def parallel_generatedRoot_986.w_29_0  (ctx : parallel_generatedRoot_986Ctx) : mxxMat15 :=
  ctx.witness.w_29_0

def parallel_generatedRoot_986.w_30_0  (ctx : parallel_generatedRoot_986Ctx) : mxxMat5 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat5)

def parallel_generatedRoot_986.w_31_0  (ctx : parallel_generatedRoot_986Ctx) : mxxMat15 :=
  MxxRuntime.matrixMulScalarRight (parallel_generatedRoot_986.w_29_0 ctx) (parallel_generatedRoot_986.w_30_0 ctx)

def parallel_generatedRoot_986.w_32_0  (ctx : parallel_generatedRoot_986Ctx) : mxxMat16 :=
  ctx.witness.w_32_0

def parallel_generatedRoot_986.w_33_0  (ctx : parallel_generatedRoot_986Ctx) : mxxMat16 :=
  ctx.witness.w_33_0

def parallel_generatedRoot_986.w_34_0  (ctx : parallel_generatedRoot_986Ctx) : mxxMat14 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_986.w_26_0 ctx) (parallel_generatedRoot_986.w_33_0 ctx)

def parallel_generatedRoot_986.w_35_0  (ctx : parallel_generatedRoot_986Ctx) : mxxMat0 :=
  ctx.witness.w_35_0

def parallel_generatedRoot_986.w_36_0  (ctx : parallel_generatedRoot_986Ctx) : mxxMat5 :=
  (MxxRuntime.matrixPolynomial [(1606938044254771331362327096896778694705812889646799470123121)] : mxxMat5)

def parallel_generatedRoot_986.w_37_0  (ctx : parallel_generatedRoot_986Ctx) : mxxMat17 :=
  ctx.witness.w_37_0

def parallel_generatedRoot_986.w_38_0  (ctx : parallel_generatedRoot_986Ctx) : mxxMat17 :=
  ctx.witness.w_38_0

def parallel_generatedRoot_986.w_39_0  (ctx : parallel_generatedRoot_986Ctx) : mxxMat5 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_986.w_35_0 ctx) (parallel_generatedRoot_986.w_38_0 ctx)

def parallel_generatedRoot_986.w_40_0  (ctx : parallel_generatedRoot_986Ctx) : mxxMat0 :=
  ctx.witness.w_40_0

def parallel_generatedRoot_986.w_41_0  (ctx : parallel_generatedRoot_986Ctx) : mxxMat0 :=
  ctx.witness.w_41_0

def parallel_generatedRoot_986.w_42_0  (ctx : parallel_generatedRoot_986Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_986.w_23_0 ctx) (parallel_generatedRoot_986.w_41_0 ctx)

def parallel_generatedRoot_986.w_43_0  (ctx : parallel_generatedRoot_986Ctx) : mxxMat0 :=
  (0 : mxxMat0)

def parallel_generatedRoot_986.w_48_0  (ctx : parallel_generatedRoot_986Ctx) : mxxMat0 :=
  ctx.witness.w_48_0

def parallel_generatedRoot_986.w_48_1  (ctx : parallel_generatedRoot_986Ctx) : mxxMat0 :=
  ctx.witness.w_48_1

def parallel_generatedRoot_986.w_48_2  (ctx : parallel_generatedRoot_986Ctx) : mxxMat0 :=
  ctx.witness.w_48_2

def parallel_generatedRoot_986.w_49_0  (ctx : parallel_generatedRoot_986Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_986.w_42_0 ctx) (parallel_generatedRoot_986.w_48_1 ctx)

def parallel_generatedRoot_986.w_52_0  (ctx : parallel_generatedRoot_986Ctx) : mxxMat11 :=
  ctx.witness.w_52_0

def parallel_generatedRoot_986.w_53_0  (ctx : parallel_generatedRoot_986Ctx) : mxxMat5 :=
  (MxxRuntime.matrixPolynomial [(1606938044254771331362327096896778694705812889646799470123121)] : mxxMat5)

def parallel_generatedRoot_986.w_54_0  (ctx : parallel_generatedRoot_986Ctx) : mxxMat2 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_986.w_53_0 ctx) (parallel_generatedRoot_986.w_4_0 ctx)

def parallel_generatedRoot_986.w_55_0  (ctx : parallel_generatedRoot_986Ctx) : mxxMat4 :=
  ctx.witness.w_55_0

def parallel_generatedRoot_986.w_56_0  (ctx : parallel_generatedRoot_986Ctx) : mxxMat4 :=
  ctx.witness.w_56_0

def parallel_generatedRoot_986.w_57_0  (ctx : parallel_generatedRoot_986Ctx) : mxxMat0 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_986.w_52_0 ctx) (parallel_generatedRoot_986.w_56_0 ctx)

def parallel_generatedRoot_986.w_59_0  (ctx : parallel_generatedRoot_986Ctx) : mxxMat0 :=
  ctx.witness.w_59_0

def parallel_generatedRoot_986.w_60_0  (ctx : parallel_generatedRoot_986Ctx) : mxxMat0 :=
  ctx.witness.w_60_0

def parallel_generatedRoot_986.w_61_0  (ctx : parallel_generatedRoot_986Ctx) : mxxMat20 :=
  ctx.witness.w_61_0

def parallel_generatedRoot_986.w_62_0  (ctx : parallel_generatedRoot_986Ctx) : mxxMat21 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat21)

def parallel_generatedRoot_986.w_63_0  (ctx : parallel_generatedRoot_986Ctx) : mxxMat20 :=
  MxxRuntime.matrixMulScalarRight (parallel_generatedRoot_986.w_61_0 ctx) (parallel_generatedRoot_986.w_62_0 ctx)

def parallel_generatedRoot_986.w_64_0  (ctx : parallel_generatedRoot_986Ctx) : mxxMat21 :=
  (MxxRuntime.matrixPolynomial [(7254835200)] : mxxMat21)

def parallel_generatedRoot_986.w_65_0  (ctx : parallel_generatedRoot_986Ctx) : mxxMat20 :=
  MxxRuntime.matrixMulScalarRight (parallel_generatedRoot_986.w_63_0 ctx) (parallel_generatedRoot_986.w_64_0 ctx)

def parallel_generatedRoot_986.w_66_0  (ctx : parallel_generatedRoot_986Ctx) : mxxMat0 :=
  ctx.witness.w_66_0

def parallel_generatedRoot_986.w_67_0  (ctx : parallel_generatedRoot_986Ctx) : mxxMat5 :=
  (MxxRuntime.matrixPolynomial [(mxxNat2)] : mxxMat5)

def parallel_generatedRoot_986.w_68_0  (ctx : parallel_generatedRoot_986Ctx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarRight (parallel_generatedRoot_986.w_66_0 ctx) (parallel_generatedRoot_986.w_67_0 ctx)

def parallel_generatedRoot_986.w_69_0  (ctx : parallel_generatedRoot_986Ctx) : mxxMat6 :=
  ctx.witness.w_69_0

def parallel_generatedRoot_986.w_70_0  (ctx : parallel_generatedRoot_986Ctx) : mxxMat6 :=
  ctx.witness.w_70_0

def parallel_generatedRoot_986.w_71_0  (ctx : parallel_generatedRoot_986Ctx) : mxxMat0 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_986.w_58_0 ctx) (parallel_generatedRoot_986.w_70_0 ctx)

def parallel_generatedRoot_986.w_72_0  (ctx : parallel_generatedRoot_986Ctx) : mxxMat0 :=
  MxxRuntime.matrixSub (parallel_generatedRoot_986.w_57_0 ctx) (parallel_generatedRoot_986.w_71_0 ctx)

def parallel_generatedRoot_986.w_73_0  (ctx : parallel_generatedRoot_986Ctx) : mxxMat0 :=
  (MxxRuntime.unitRow (0) : mxxMat0)

def parallel_generatedRoot_986.w_75_0  (ctx : parallel_generatedRoot_986Ctx) : mxxMat14 :=
  ctx.witness.w_75_0

def parallel_generatedRoot_986.w_76_0  (ctx : parallel_generatedRoot_986Ctx) : mxxMat15 :=
  ctx.witness.w_76_0

def parallel_generatedRoot_986.w_77_0  (ctx : parallel_generatedRoot_986Ctx) : mxxMat15 :=
  ctx.witness.w_77_0

def parallel_generatedRoot_986.w_78_0  (ctx : parallel_generatedRoot_986Ctx) : mxxMat5 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat5)

def parallel_generatedRoot_986.w_79_0  (ctx : parallel_generatedRoot_986Ctx) : mxxMat15 :=
  MxxRuntime.matrixMulScalarRight (parallel_generatedRoot_986.w_77_0 ctx) (parallel_generatedRoot_986.w_78_0 ctx)

def parallel_generatedRoot_986.w_80_0  (ctx : parallel_generatedRoot_986Ctx) : mxxMat16 :=
  ctx.witness.w_80_0

def parallel_generatedRoot_986.w_81_0  (ctx : parallel_generatedRoot_986Ctx) : mxxMat16 :=
  ctx.witness.w_81_0

def parallel_generatedRoot_986.w_82_0  (ctx : parallel_generatedRoot_986Ctx) : mxxMat14 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_986.w_75_0 ctx) (parallel_generatedRoot_986.w_81_0 ctx)

def parallel_generatedRoot_986.w_83_0  (ctx : parallel_generatedRoot_986Ctx) : mxxMat0 :=
  ctx.witness.w_83_0

def parallel_generatedRoot_986.w_84_0  (ctx : parallel_generatedRoot_986Ctx) : mxxMat5 :=
  (MxxRuntime.matrixPolynomial [(1606938044254771331362327096896778694705812889646799470123121)] : mxxMat5)

def parallel_generatedRoot_986.w_85_0  (ctx : parallel_generatedRoot_986Ctx) : mxxMat17 :=
  ctx.witness.w_85_0

def parallel_generatedRoot_986.w_86_0  (ctx : parallel_generatedRoot_986Ctx) : mxxMat17 :=
  ctx.witness.w_86_0

def parallel_generatedRoot_986.w_87_0  (ctx : parallel_generatedRoot_986Ctx) : mxxMat5 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_986.w_83_0 ctx) (parallel_generatedRoot_986.w_86_0 ctx)

def parallel_generatedRoot_986.w_88_0  (ctx : parallel_generatedRoot_986Ctx) : mxxMat0 :=
  ctx.witness.w_88_0

def parallel_generatedRoot_986.w_89_0  (ctx : parallel_generatedRoot_986Ctx) : mxxMat0 :=
  ctx.witness.w_89_0

def parallel_generatedRoot_986.w_90_0  (ctx : parallel_generatedRoot_986Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_986.w_72_0 ctx) (parallel_generatedRoot_986.w_89_0 ctx)

def parallel_generatedRoot_986.w_91_0  (ctx : parallel_generatedRoot_986Ctx) : mxxMat0 :=
  (0 : mxxMat0)

def parallel_generatedRoot_986.w_92_0  (ctx : parallel_generatedRoot_986Ctx) : mxxMat0 :=
  ctx.witness.w_92_0

def parallel_generatedRoot_986.w_92_1  (ctx : parallel_generatedRoot_986Ctx) : mxxMat0 :=
  ctx.witness.w_92_1

def parallel_generatedRoot_986.w_92_2  (ctx : parallel_generatedRoot_986Ctx) : mxxMat0 :=
  ctx.witness.w_92_2

def parallel_generatedRoot_986.w_93_0  (ctx : parallel_generatedRoot_986Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_986.w_90_0 ctx) (parallel_generatedRoot_986.w_92_2 ctx)

def parallel_generatedRoot_986.w_94_0  (ctx : parallel_generatedRoot_986Ctx) : mxxMat2 :=
  ctx.witness.w_94_0

def parallel_generatedRoot_986.outs  (ctx : parallel_generatedRoot_986Ctx) : mxxMat2 :=
  (parallel_generatedRoot_986.w_94_0 ctx)

structure parallel_generatedRoot_986.Facts_0  (ctx : parallel_generatedRoot_986Ctx) : Prop where
  h2_0 : MxxRuntime.concatColumns (parallel_generatedRoot_986.w_0_0 ctx) (parallel_generatedRoot_986.w_1_0 ctx) (parallel_generatedRoot_986.w_2_0 ctx)
  h6_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (parallel_generatedRoot_986.w_5_0 ctx) (parallel_generatedRoot_986.w_6_0 ctx)
  h7_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (parallel_generatedRoot_986.w_6_0 ctx) (parallel_generatedRoot_986.w_7_0 ctx)
  h10_0 : MxxRuntime.gadgetMatrixRuns backend 4096 25 (parallel_generatedRoot_986.w_10_0 ctx)
  h11_0 : MxxRuntime.centeredRebaseRuns (parallel_generatedRoot_986.w_10_0 ctx) (parallel_generatedRoot_986.w_11_0 ctx)
  h12_0 : MxxRuntime.modulusReduceRuns (parallel_generatedRoot_986.w_11_0 ctx) (parallel_generatedRoot_986.w_12_0 ctx)
  h17_0 : MxxRuntime.centeredRebaseRuns (parallel_generatedRoot_986.w_16_0 ctx) (parallel_generatedRoot_986.w_17_0 ctx)
  h20_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (parallel_generatedRoot_986.w_19_0 ctx) (parallel_generatedRoot_986.w_20_0 ctx)
  h21_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (parallel_generatedRoot_986.w_20_0 ctx) (parallel_generatedRoot_986.w_21_0 ctx)
  t9 : 0 ≤ (0)
  t10 : (0) < 25
  t11 : 0 ≤ 0
  t12 : 0 < 1
  t13 : 1 ≤ 1
  t14 : 0 ≤ (625 * (Int.ofNat i_0))
  t15 : (625 * (Int.ofNat i_0)) < (625 + (625 * (Int.ofNat i_0)))
  t16 : (625 + (625 * (Int.ofNat i_0))) ≤ 22500
  h26_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_986.w_25_0 ctx) 0 1 (625 * (Int.ofNat i_0)) (625 + (625 * (Int.ofNat i_0))) (parallel_generatedRoot_986.w_26_0 ctx)
  h28_0 : MxxRuntime.blockDiagonalRuns 25 (parallel_generatedRoot_986.w_27_0 ctx) (parallel_generatedRoot_986.w_28_0 ctx)
  h29_0 : MxxRuntime.centeredRebaseRuns (parallel_generatedRoot_986.w_28_0 ctx) (parallel_generatedRoot_986.w_29_0 ctx)
  h32_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (parallel_generatedRoot_986.w_31_0 ctx) (parallel_generatedRoot_986.w_32_0 ctx)
  h33_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (parallel_generatedRoot_986.w_32_0 ctx) (parallel_generatedRoot_986.w_33_0 ctx)
  t22 : 0 ≤ 0
  t23 : 0 < 1
  t24 : 1 ≤ 1
  t25 : 0 ≤ 0
  t26 : 0 < 25
  t27 : 25 ≤ 625
  h35_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_986.w_34_0 ctx) 0 1 0 25 (parallel_generatedRoot_986.w_35_0 ctx)
  h37_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (parallel_generatedRoot_986.w_36_0 ctx) (parallel_generatedRoot_986.w_37_0 ctx)
  h38_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (parallel_generatedRoot_986.w_37_0 ctx) (parallel_generatedRoot_986.w_38_0 ctx)
  h40_0 : MxxRuntime.tensorRuns (parallel_generatedRoot_986.w_24_0 ctx) (parallel_generatedRoot_986.w_39_0 ctx) (parallel_generatedRoot_986.w_40_0 ctx)

theorem parallel_generatedRoot_986.facts_0  {ctx : parallel_generatedRoot_986Ctx} {outputs : mxxMat2}
    (h : parallel_generatedRoot_986.constraints_0 backend i_0 (parallel_generatedRoot_986.w_0_0 ctx) (parallel_generatedRoot_986.w_1_0 ctx) (parallel_generatedRoot_986.w_25_0 ctx) (parallel_generatedRoot_986.w_27_0 ctx) (parallel_generatedRoot_986.w_2_0 ctx) (parallel_generatedRoot_986.w_5_0 ctx) (parallel_generatedRoot_986.w_6_0 ctx) (parallel_generatedRoot_986.w_7_0 ctx) (parallel_generatedRoot_986.w_10_0 ctx) (parallel_generatedRoot_986.w_11_0 ctx) (parallel_generatedRoot_986.w_12_0 ctx) (parallel_generatedRoot_986.w_16_0 ctx) (parallel_generatedRoot_986.w_17_0 ctx) (parallel_generatedRoot_986.w_19_0 ctx) (parallel_generatedRoot_986.w_20_0 ctx) (parallel_generatedRoot_986.w_21_0 ctx) (parallel_generatedRoot_986.w_24_0 ctx) (parallel_generatedRoot_986.w_26_0 ctx) (parallel_generatedRoot_986.w_28_0 ctx) (parallel_generatedRoot_986.w_29_0 ctx) (parallel_generatedRoot_986.w_31_0 ctx) (parallel_generatedRoot_986.w_32_0 ctx) (parallel_generatedRoot_986.w_33_0 ctx) (parallel_generatedRoot_986.w_34_0 ctx) (parallel_generatedRoot_986.w_35_0 ctx) (parallel_generatedRoot_986.w_36_0 ctx) (parallel_generatedRoot_986.w_37_0 ctx) (parallel_generatedRoot_986.w_38_0 ctx) (parallel_generatedRoot_986.w_39_0 ctx) (parallel_generatedRoot_986.w_40_0 ctx)) :
    parallel_generatedRoot_986.Facts_0 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_986.Facts_1  (ctx : parallel_generatedRoot_986Ctx) : Prop where
  t32 : 0 ≤ (24)
  h41_0 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0) => sequential_parallel_generatedRoot_986_41 backend hashModel tape (path ++ [41, i]) params i_0 i (current, ((parallel_generatedRoot_986.w_34_0 ctx), ((parallel_generatedRoot_986.w_38_0 ctx), ()))) next) (Int.toNat (24)) (parallel_generatedRoot_986.w_40_0 ctx) (parallel_generatedRoot_986.w_41_0 ctx)
  t34 : 0 ≤ (1)
  h48_2 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_986_48 backend hashModel tape (path ++ [48, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, ((parallel_generatedRoot_986.w_44_0 ctx), ((parallel_generatedRoot_986.w_45_0 ctx), ((parallel_generatedRoot_986.w_46_0 ctx), ((parallel_generatedRoot_986.w_47_0 ctx), ()))))))) next) (Int.toNat (1)) ((parallel_generatedRoot_986.w_43_0 ctx), ((parallel_generatedRoot_986.w_43_0 ctx), ((parallel_generatedRoot_986.w_43_0 ctx), ()))) ((parallel_generatedRoot_986.w_48_0 ctx), ((parallel_generatedRoot_986.w_48_1 ctx), ((parallel_generatedRoot_986.w_48_2 ctx), ())))
  h52_0 : MxxRuntime.concatColumns (parallel_generatedRoot_986.w_50_0 ctx) (parallel_generatedRoot_986.w_51_0 ctx) (parallel_generatedRoot_986.w_52_0 ctx)
  h55_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (parallel_generatedRoot_986.w_54_0 ctx) (parallel_generatedRoot_986.w_55_0 ctx)
  h56_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (parallel_generatedRoot_986.w_55_0 ctx) (parallel_generatedRoot_986.w_56_0 ctx)
  h59_0 : MxxRuntime.gadgetMatrixRuns backend 4096 25 (parallel_generatedRoot_986.w_59_0 ctx)
  h60_0 : MxxRuntime.centeredRebaseRuns (parallel_generatedRoot_986.w_59_0 ctx) (parallel_generatedRoot_986.w_60_0 ctx)
  h61_0 : MxxRuntime.modulusReduceRuns (parallel_generatedRoot_986.w_60_0 ctx) (parallel_generatedRoot_986.w_61_0 ctx)
  h66_0 : MxxRuntime.centeredRebaseRuns (parallel_generatedRoot_986.w_65_0 ctx) (parallel_generatedRoot_986.w_66_0 ctx)
  h69_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (parallel_generatedRoot_986.w_68_0 ctx) (parallel_generatedRoot_986.w_69_0 ctx)
  h70_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (parallel_generatedRoot_986.w_69_0 ctx) (parallel_generatedRoot_986.w_70_0 ctx)
  t45 : 0 ≤ (0)
  t46 : (0) < 25
  t47 : 0 ≤ 0
  t48 : 0 < 1
  t49 : 1 ≤ 1
  t50 : 0 ≤ (625 * (Int.ofNat i_0))
  t51 : (625 * (Int.ofNat i_0)) < (625 + (625 * (Int.ofNat i_0)))
  t52 : (625 + (625 * (Int.ofNat i_0))) ≤ 22500
  h75_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_986.w_74_0 ctx) 0 1 (625 * (Int.ofNat i_0)) (625 + (625 * (Int.ofNat i_0))) (parallel_generatedRoot_986.w_75_0 ctx)
  h76_0 : MxxRuntime.blockDiagonalRuns 25 (parallel_generatedRoot_986.w_27_0 ctx) (parallel_generatedRoot_986.w_76_0 ctx)
  h77_0 : MxxRuntime.centeredRebaseRuns (parallel_generatedRoot_986.w_76_0 ctx) (parallel_generatedRoot_986.w_77_0 ctx)
  h80_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (parallel_generatedRoot_986.w_79_0 ctx) (parallel_generatedRoot_986.w_80_0 ctx)
  h81_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (parallel_generatedRoot_986.w_80_0 ctx) (parallel_generatedRoot_986.w_81_0 ctx)
  t58 : 0 ≤ 0
  t59 : 0 < 1
  t60 : 1 ≤ 1
  t61 : 0 ≤ 0
  t62 : 0 < 25
  t63 : 25 ≤ 625

theorem parallel_generatedRoot_986.facts_1  {ctx : parallel_generatedRoot_986Ctx} {outputs : mxxMat2}
    (h : parallel_generatedRoot_986.constraints_1 backend hashModel tape path params i_0 (parallel_generatedRoot_986.w_27_0 ctx) (parallel_generatedRoot_986.w_44_0 ctx) (parallel_generatedRoot_986.w_45_0 ctx) (parallel_generatedRoot_986.w_46_0 ctx) (parallel_generatedRoot_986.w_47_0 ctx) (parallel_generatedRoot_986.w_50_0 ctx) (parallel_generatedRoot_986.w_51_0 ctx) (parallel_generatedRoot_986.w_74_0 ctx) (parallel_generatedRoot_986.w_34_0 ctx) (parallel_generatedRoot_986.w_38_0 ctx) (parallel_generatedRoot_986.w_40_0 ctx) (parallel_generatedRoot_986.w_41_0 ctx) (parallel_generatedRoot_986.w_43_0 ctx) (parallel_generatedRoot_986.w_48_0 ctx) (parallel_generatedRoot_986.w_48_1 ctx) (parallel_generatedRoot_986.w_48_2 ctx) (parallel_generatedRoot_986.w_52_0 ctx) (parallel_generatedRoot_986.w_54_0 ctx) (parallel_generatedRoot_986.w_55_0 ctx) (parallel_generatedRoot_986.w_56_0 ctx) (parallel_generatedRoot_986.w_59_0 ctx) (parallel_generatedRoot_986.w_60_0 ctx) (parallel_generatedRoot_986.w_61_0 ctx) (parallel_generatedRoot_986.w_65_0 ctx) (parallel_generatedRoot_986.w_66_0 ctx) (parallel_generatedRoot_986.w_68_0 ctx) (parallel_generatedRoot_986.w_69_0 ctx) (parallel_generatedRoot_986.w_70_0 ctx) (parallel_generatedRoot_986.w_75_0 ctx) (parallel_generatedRoot_986.w_76_0 ctx) (parallel_generatedRoot_986.w_77_0 ctx) (parallel_generatedRoot_986.w_79_0 ctx) (parallel_generatedRoot_986.w_80_0 ctx) (parallel_generatedRoot_986.w_81_0 ctx)) :
    parallel_generatedRoot_986.Facts_1 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_986.Facts_2  (ctx : parallel_generatedRoot_986Ctx) (outputs : mxxMat2) : Prop where
  h83_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_986.w_82_0 ctx) 0 1 0 25 (parallel_generatedRoot_986.w_83_0 ctx)
  h85_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (parallel_generatedRoot_986.w_84_0 ctx) (parallel_generatedRoot_986.w_85_0 ctx)
  h86_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (parallel_generatedRoot_986.w_85_0 ctx) (parallel_generatedRoot_986.w_86_0 ctx)
  h88_0 : MxxRuntime.tensorRuns (parallel_generatedRoot_986.w_73_0 ctx) (parallel_generatedRoot_986.w_87_0 ctx) (parallel_generatedRoot_986.w_88_0 ctx)
  t68 : 0 ≤ (24)
  h89_0 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0) => sequential_parallel_generatedRoot_986_89 backend hashModel tape (path ++ [89, i]) params i_0 i (current, ((parallel_generatedRoot_986.w_82_0 ctx), ((parallel_generatedRoot_986.w_86_0 ctx), ()))) next) (Int.toNat (24)) (parallel_generatedRoot_986.w_88_0 ctx) (parallel_generatedRoot_986.w_89_0 ctx)
  t70 : 0 ≤ (1)
  h92_2 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_986_92 backend hashModel tape (path ++ [92, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, ((parallel_generatedRoot_986.w_44_0 ctx), ((parallel_generatedRoot_986.w_45_0 ctx), ((parallel_generatedRoot_986.w_46_0 ctx), ((parallel_generatedRoot_986.w_47_0 ctx), ()))))))) next) (Int.toNat (1)) ((parallel_generatedRoot_986.w_91_0 ctx), ((parallel_generatedRoot_986.w_91_0 ctx), ((parallel_generatedRoot_986.w_91_0 ctx), ()))) ((parallel_generatedRoot_986.w_92_0 ctx), ((parallel_generatedRoot_986.w_92_1 ctx), ((parallel_generatedRoot_986.w_92_2 ctx), ())))
  h94_0 : MxxRuntime.concatRows (parallel_generatedRoot_986.w_49_0 ctx) (parallel_generatedRoot_986.w_93_0 ctx) (parallel_generatedRoot_986.w_94_0 ctx)
  hout : outputs = parallel_generatedRoot_986.outs ctx

theorem parallel_generatedRoot_986.facts_2  {ctx : parallel_generatedRoot_986Ctx} {outputs : mxxMat2}
    (h : parallel_generatedRoot_986.constraints_2 backend hashModel tape path params i_0 (parallel_generatedRoot_986.w_44_0 ctx) (parallel_generatedRoot_986.w_45_0 ctx) (parallel_generatedRoot_986.w_46_0 ctx) (parallel_generatedRoot_986.w_47_0 ctx) (parallel_generatedRoot_986.w_49_0 ctx) (parallel_generatedRoot_986.w_73_0 ctx) (parallel_generatedRoot_986.w_82_0 ctx) (parallel_generatedRoot_986.w_83_0 ctx) (parallel_generatedRoot_986.w_84_0 ctx) (parallel_generatedRoot_986.w_85_0 ctx) (parallel_generatedRoot_986.w_86_0 ctx) (parallel_generatedRoot_986.w_87_0 ctx) (parallel_generatedRoot_986.w_88_0 ctx) (parallel_generatedRoot_986.w_89_0 ctx) (parallel_generatedRoot_986.w_91_0 ctx) (parallel_generatedRoot_986.w_92_0 ctx) (parallel_generatedRoot_986.w_92_1 ctx) (parallel_generatedRoot_986.w_92_2 ctx) (parallel_generatedRoot_986.w_93_0 ctx) (parallel_generatedRoot_986.w_94_0 ctx) outputs) :
    parallel_generatedRoot_986.Facts_2 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_986.Facts  (ctx : parallel_generatedRoot_986Ctx) (outputs : mxxMat2) : Prop where
  c0 : parallel_generatedRoot_986.Facts_0 ctx
  c1 : parallel_generatedRoot_986.Facts_1 ctx
  c2 : parallel_generatedRoot_986.Facts_2 ctx outputs

theorem parallel_generatedRoot_986.facts  (ctx : parallel_generatedRoot_986Ctx) {outputs : mxxMat2}
    (h : parallel_generatedRoot_986.body backend hashModel tape path params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_986.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_986.facts_0 (ctx := ctx) (outputs := outputs) t0.1
  have t1 := t0.2
  have r1 := parallel_generatedRoot_986.facts_1 (ctx := ctx) (outputs := outputs) t1.1
  have t2 := t1.2
  have r2 := parallel_generatedRoot_986.facts_2 (ctx := ctx) (outputs := outputs) t2
  ⟨r0, r1, r2⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_987.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) where
  inputs : (parallel_generatedRoot_987.InputType)
  witness : parallel_generatedRoot_987.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_987Ctx" => (parallel_generatedRoot_987.Ctx backend hashModel tape path params i_0)

def parallel_generatedRoot_987.w_0_0  (ctx : parallel_generatedRoot_987Ctx) : mxxMat0 :=
  ctx.inputs.1

def parallel_generatedRoot_987.w_1_0  (ctx : parallel_generatedRoot_987Ctx) : mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_987.w_4_0  (ctx : parallel_generatedRoot_987Ctx) : mxxMat2 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_987.w_9_0  (ctx : parallel_generatedRoot_987Ctx) : mxxMat0 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_987.w_25_0  (ctx : parallel_generatedRoot_987Ctx) : mxxMat7 :=
  ctx.inputs.2.2.2.2.1

def parallel_generatedRoot_987.w_27_0  (ctx : parallel_generatedRoot_987Ctx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.1

def parallel_generatedRoot_987.w_44_0  (ctx : parallel_generatedRoot_987Ctx) : mxxMat9 :=
  ctx.inputs.2.2.2.2.2.2.1

def parallel_generatedRoot_987.w_45_0  (ctx : parallel_generatedRoot_987Ctx) : Fin 1 → mxxMat5 :=
  ctx.inputs.2.2.2.2.2.2.2.1

def parallel_generatedRoot_987.w_46_0  (ctx : parallel_generatedRoot_987Ctx) : mxxMat9 :=
  ctx.inputs.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_987.w_47_0  (ctx : parallel_generatedRoot_987Ctx) : mxxMat9 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_987.w_50_0  (ctx : parallel_generatedRoot_987Ctx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_987.w_51_0  (ctx : parallel_generatedRoot_987Ctx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_987.w_58_0  (ctx : parallel_generatedRoot_987Ctx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_987.w_74_0  (ctx : parallel_generatedRoot_987Ctx) : mxxMat7 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_987.w_2_0  (ctx : parallel_generatedRoot_987Ctx) : mxxMat11 :=
  ctx.witness.w_2_0

def parallel_generatedRoot_987.w_3_0  (ctx : parallel_generatedRoot_987Ctx) : mxxMat5 :=
  (MxxRuntime.matrixPolynomial [(1606938044255228050623992174508164557277769211170177223088561)] : mxxMat5)

def parallel_generatedRoot_987.w_5_0  (ctx : parallel_generatedRoot_987Ctx) : mxxMat2 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_987.w_3_0 ctx) (parallel_generatedRoot_987.w_4_0 ctx)

def parallel_generatedRoot_987.w_6_0  (ctx : parallel_generatedRoot_987Ctx) : mxxMat4 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_987.w_7_0  (ctx : parallel_generatedRoot_987Ctx) : mxxMat4 :=
  ctx.witness.w_7_0

def parallel_generatedRoot_987.w_8_0  (ctx : parallel_generatedRoot_987Ctx) : mxxMat0 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_987.w_2_0 ctx) (parallel_generatedRoot_987.w_7_0 ctx)

def parallel_generatedRoot_987.w_10_0  (ctx : parallel_generatedRoot_987Ctx) : mxxMat0 :=
  ctx.witness.w_10_0

def parallel_generatedRoot_987.w_11_0  (ctx : parallel_generatedRoot_987Ctx) : mxxMat0 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_987.w_12_0  (ctx : parallel_generatedRoot_987Ctx) : mxxMat22 :=
  ctx.witness.w_12_0

def parallel_generatedRoot_987.w_13_0  (ctx : parallel_generatedRoot_987Ctx) : mxxMat23 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat23)

def parallel_generatedRoot_987.w_14_0  (ctx : parallel_generatedRoot_987Ctx) : mxxMat22 :=
  MxxRuntime.matrixMulScalarRight (parallel_generatedRoot_987.w_12_0 ctx) (parallel_generatedRoot_987.w_13_0 ctx)

def parallel_generatedRoot_987.w_15_0  (ctx : parallel_generatedRoot_987Ctx) : mxxMat23 :=
  (MxxRuntime.matrixPolynomial [(1125871968844913)] : mxxMat23)

def parallel_generatedRoot_987.w_16_0  (ctx : parallel_generatedRoot_987Ctx) : mxxMat22 :=
  MxxRuntime.matrixMulScalarRight (parallel_generatedRoot_987.w_14_0 ctx) (parallel_generatedRoot_987.w_15_0 ctx)

def parallel_generatedRoot_987.w_17_0  (ctx : parallel_generatedRoot_987Ctx) : mxxMat0 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_987.w_18_0  (ctx : parallel_generatedRoot_987Ctx) : mxxMat5 :=
  (MxxRuntime.matrixPolynomial [(mxxNat3)] : mxxMat5)

def parallel_generatedRoot_987.w_19_0  (ctx : parallel_generatedRoot_987Ctx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarRight (parallel_generatedRoot_987.w_17_0 ctx) (parallel_generatedRoot_987.w_18_0 ctx)

def parallel_generatedRoot_987.w_20_0  (ctx : parallel_generatedRoot_987Ctx) : mxxMat6 :=
  ctx.witness.w_20_0

def parallel_generatedRoot_987.w_21_0  (ctx : parallel_generatedRoot_987Ctx) : mxxMat6 :=
  ctx.witness.w_21_0

def parallel_generatedRoot_987.w_22_0  (ctx : parallel_generatedRoot_987Ctx) : mxxMat0 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_987.w_9_0 ctx) (parallel_generatedRoot_987.w_21_0 ctx)

def parallel_generatedRoot_987.w_23_0  (ctx : parallel_generatedRoot_987Ctx) : mxxMat0 :=
  MxxRuntime.matrixSub (parallel_generatedRoot_987.w_8_0 ctx) (parallel_generatedRoot_987.w_22_0 ctx)

def parallel_generatedRoot_987.w_24_0  (ctx : parallel_generatedRoot_987Ctx) : mxxMat0 :=
  (MxxRuntime.unitRow (0) : mxxMat0)

def parallel_generatedRoot_987.w_26_0  (ctx : parallel_generatedRoot_987Ctx) : mxxMat14 :=
  ctx.witness.w_26_0

def parallel_generatedRoot_987.w_28_0  (ctx : parallel_generatedRoot_987Ctx) : mxxMat15 :=
  ctx.witness.w_28_0

def parallel_generatedRoot_987.w_29_0  (ctx : parallel_generatedRoot_987Ctx) : mxxMat15 :=
  ctx.witness.w_29_0

def parallel_generatedRoot_987.w_30_0  (ctx : parallel_generatedRoot_987Ctx) : mxxMat5 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat5)

def parallel_generatedRoot_987.w_31_0  (ctx : parallel_generatedRoot_987Ctx) : mxxMat15 :=
  MxxRuntime.matrixMulScalarRight (parallel_generatedRoot_987.w_29_0 ctx) (parallel_generatedRoot_987.w_30_0 ctx)

def parallel_generatedRoot_987.w_32_0  (ctx : parallel_generatedRoot_987Ctx) : mxxMat16 :=
  ctx.witness.w_32_0

def parallel_generatedRoot_987.w_33_0  (ctx : parallel_generatedRoot_987Ctx) : mxxMat16 :=
  ctx.witness.w_33_0

def parallel_generatedRoot_987.w_34_0  (ctx : parallel_generatedRoot_987Ctx) : mxxMat14 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_987.w_26_0 ctx) (parallel_generatedRoot_987.w_33_0 ctx)

def parallel_generatedRoot_987.w_35_0  (ctx : parallel_generatedRoot_987Ctx) : mxxMat0 :=
  ctx.witness.w_35_0

def parallel_generatedRoot_987.w_36_0  (ctx : parallel_generatedRoot_987Ctx) : mxxMat5 :=
  (MxxRuntime.matrixPolynomial [(1606938044255228050623992174508164557277769211170177223088561)] : mxxMat5)

def parallel_generatedRoot_987.w_37_0  (ctx : parallel_generatedRoot_987Ctx) : mxxMat17 :=
  ctx.witness.w_37_0

def parallel_generatedRoot_987.w_38_0  (ctx : parallel_generatedRoot_987Ctx) : mxxMat17 :=
  ctx.witness.w_38_0

def parallel_generatedRoot_987.w_39_0  (ctx : parallel_generatedRoot_987Ctx) : mxxMat5 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_987.w_35_0 ctx) (parallel_generatedRoot_987.w_38_0 ctx)

def parallel_generatedRoot_987.w_40_0  (ctx : parallel_generatedRoot_987Ctx) : mxxMat0 :=
  ctx.witness.w_40_0

def parallel_generatedRoot_987.w_41_0  (ctx : parallel_generatedRoot_987Ctx) : mxxMat0 :=
  ctx.witness.w_41_0

def parallel_generatedRoot_987.w_42_0  (ctx : parallel_generatedRoot_987Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_987.w_23_0 ctx) (parallel_generatedRoot_987.w_41_0 ctx)

def parallel_generatedRoot_987.w_43_0  (ctx : parallel_generatedRoot_987Ctx) : mxxMat0 :=
  (0 : mxxMat0)

def parallel_generatedRoot_987.w_48_0  (ctx : parallel_generatedRoot_987Ctx) : mxxMat0 :=
  ctx.witness.w_48_0

def parallel_generatedRoot_987.w_48_1  (ctx : parallel_generatedRoot_987Ctx) : mxxMat0 :=
  ctx.witness.w_48_1

def parallel_generatedRoot_987.w_48_2  (ctx : parallel_generatedRoot_987Ctx) : mxxMat0 :=
  ctx.witness.w_48_2

def parallel_generatedRoot_987.w_49_0  (ctx : parallel_generatedRoot_987Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_987.w_42_0 ctx) (parallel_generatedRoot_987.w_48_1 ctx)

def parallel_generatedRoot_987.w_52_0  (ctx : parallel_generatedRoot_987Ctx) : mxxMat11 :=
  ctx.witness.w_52_0

def parallel_generatedRoot_987.w_53_0  (ctx : parallel_generatedRoot_987Ctx) : mxxMat5 :=
  (MxxRuntime.matrixPolynomial [(1606938044255228050623992174508164557277769211170177223088561)] : mxxMat5)

def parallel_generatedRoot_987.w_54_0  (ctx : parallel_generatedRoot_987Ctx) : mxxMat2 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_987.w_53_0 ctx) (parallel_generatedRoot_987.w_4_0 ctx)

def parallel_generatedRoot_987.w_55_0  (ctx : parallel_generatedRoot_987Ctx) : mxxMat4 :=
  ctx.witness.w_55_0

def parallel_generatedRoot_987.w_56_0  (ctx : parallel_generatedRoot_987Ctx) : mxxMat4 :=
  ctx.witness.w_56_0

def parallel_generatedRoot_987.w_57_0  (ctx : parallel_generatedRoot_987Ctx) : mxxMat0 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_987.w_52_0 ctx) (parallel_generatedRoot_987.w_56_0 ctx)

def parallel_generatedRoot_987.w_59_0  (ctx : parallel_generatedRoot_987Ctx) : mxxMat0 :=
  ctx.witness.w_59_0

def parallel_generatedRoot_987.w_60_0  (ctx : parallel_generatedRoot_987Ctx) : mxxMat0 :=
  ctx.witness.w_60_0

def parallel_generatedRoot_987.w_61_0  (ctx : parallel_generatedRoot_987Ctx) : mxxMat22 :=
  ctx.witness.w_61_0

def parallel_generatedRoot_987.w_62_0  (ctx : parallel_generatedRoot_987Ctx) : mxxMat23 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat23)

def parallel_generatedRoot_987.w_63_0  (ctx : parallel_generatedRoot_987Ctx) : mxxMat22 :=
  MxxRuntime.matrixMulScalarRight (parallel_generatedRoot_987.w_61_0 ctx) (parallel_generatedRoot_987.w_62_0 ctx)

def parallel_generatedRoot_987.w_64_0  (ctx : parallel_generatedRoot_987Ctx) : mxxMat23 :=
  (MxxRuntime.matrixPolynomial [(1125871968844913)] : mxxMat23)

def parallel_generatedRoot_987.w_65_0  (ctx : parallel_generatedRoot_987Ctx) : mxxMat22 :=
  MxxRuntime.matrixMulScalarRight (parallel_generatedRoot_987.w_63_0 ctx) (parallel_generatedRoot_987.w_64_0 ctx)

def parallel_generatedRoot_987.w_66_0  (ctx : parallel_generatedRoot_987Ctx) : mxxMat0 :=
  ctx.witness.w_66_0

def parallel_generatedRoot_987.w_67_0  (ctx : parallel_generatedRoot_987Ctx) : mxxMat5 :=
  (MxxRuntime.matrixPolynomial [(mxxNat3)] : mxxMat5)

def parallel_generatedRoot_987.w_68_0  (ctx : parallel_generatedRoot_987Ctx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarRight (parallel_generatedRoot_987.w_66_0 ctx) (parallel_generatedRoot_987.w_67_0 ctx)

def parallel_generatedRoot_987.w_69_0  (ctx : parallel_generatedRoot_987Ctx) : mxxMat6 :=
  ctx.witness.w_69_0

def parallel_generatedRoot_987.w_70_0  (ctx : parallel_generatedRoot_987Ctx) : mxxMat6 :=
  ctx.witness.w_70_0

def parallel_generatedRoot_987.w_71_0  (ctx : parallel_generatedRoot_987Ctx) : mxxMat0 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_987.w_58_0 ctx) (parallel_generatedRoot_987.w_70_0 ctx)

def parallel_generatedRoot_987.w_72_0  (ctx : parallel_generatedRoot_987Ctx) : mxxMat0 :=
  MxxRuntime.matrixSub (parallel_generatedRoot_987.w_57_0 ctx) (parallel_generatedRoot_987.w_71_0 ctx)

def parallel_generatedRoot_987.w_73_0  (ctx : parallel_generatedRoot_987Ctx) : mxxMat0 :=
  (MxxRuntime.unitRow (0) : mxxMat0)

def parallel_generatedRoot_987.w_75_0  (ctx : parallel_generatedRoot_987Ctx) : mxxMat14 :=
  ctx.witness.w_75_0

def parallel_generatedRoot_987.w_76_0  (ctx : parallel_generatedRoot_987Ctx) : mxxMat15 :=
  ctx.witness.w_76_0

def parallel_generatedRoot_987.w_77_0  (ctx : parallel_generatedRoot_987Ctx) : mxxMat15 :=
  ctx.witness.w_77_0

def parallel_generatedRoot_987.w_78_0  (ctx : parallel_generatedRoot_987Ctx) : mxxMat5 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat5)

def parallel_generatedRoot_987.w_79_0  (ctx : parallel_generatedRoot_987Ctx) : mxxMat15 :=
  MxxRuntime.matrixMulScalarRight (parallel_generatedRoot_987.w_77_0 ctx) (parallel_generatedRoot_987.w_78_0 ctx)

def parallel_generatedRoot_987.w_80_0  (ctx : parallel_generatedRoot_987Ctx) : mxxMat16 :=
  ctx.witness.w_80_0

def parallel_generatedRoot_987.w_81_0  (ctx : parallel_generatedRoot_987Ctx) : mxxMat16 :=
  ctx.witness.w_81_0

def parallel_generatedRoot_987.w_82_0  (ctx : parallel_generatedRoot_987Ctx) : mxxMat14 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_987.w_75_0 ctx) (parallel_generatedRoot_987.w_81_0 ctx)

def parallel_generatedRoot_987.w_83_0  (ctx : parallel_generatedRoot_987Ctx) : mxxMat0 :=
  ctx.witness.w_83_0

def parallel_generatedRoot_987.w_84_0  (ctx : parallel_generatedRoot_987Ctx) : mxxMat5 :=
  (MxxRuntime.matrixPolynomial [(1606938044255228050623992174508164557277769211170177223088561)] : mxxMat5)

def parallel_generatedRoot_987.w_85_0  (ctx : parallel_generatedRoot_987Ctx) : mxxMat17 :=
  ctx.witness.w_85_0

def parallel_generatedRoot_987.w_86_0  (ctx : parallel_generatedRoot_987Ctx) : mxxMat17 :=
  ctx.witness.w_86_0

def parallel_generatedRoot_987.w_87_0  (ctx : parallel_generatedRoot_987Ctx) : mxxMat5 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_987.w_83_0 ctx) (parallel_generatedRoot_987.w_86_0 ctx)

def parallel_generatedRoot_987.w_88_0  (ctx : parallel_generatedRoot_987Ctx) : mxxMat0 :=
  ctx.witness.w_88_0

def parallel_generatedRoot_987.w_89_0  (ctx : parallel_generatedRoot_987Ctx) : mxxMat0 :=
  ctx.witness.w_89_0

def parallel_generatedRoot_987.w_90_0  (ctx : parallel_generatedRoot_987Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_987.w_72_0 ctx) (parallel_generatedRoot_987.w_89_0 ctx)

def parallel_generatedRoot_987.w_91_0  (ctx : parallel_generatedRoot_987Ctx) : mxxMat0 :=
  (0 : mxxMat0)

def parallel_generatedRoot_987.w_92_0  (ctx : parallel_generatedRoot_987Ctx) : mxxMat0 :=
  ctx.witness.w_92_0

def parallel_generatedRoot_987.w_92_1  (ctx : parallel_generatedRoot_987Ctx) : mxxMat0 :=
  ctx.witness.w_92_1

def parallel_generatedRoot_987.w_92_2  (ctx : parallel_generatedRoot_987Ctx) : mxxMat0 :=
  ctx.witness.w_92_2

def parallel_generatedRoot_987.w_93_0  (ctx : parallel_generatedRoot_987Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_987.w_90_0 ctx) (parallel_generatedRoot_987.w_92_2 ctx)

def parallel_generatedRoot_987.w_94_0  (ctx : parallel_generatedRoot_987Ctx) : mxxMat2 :=
  ctx.witness.w_94_0

def parallel_generatedRoot_987.outs  (ctx : parallel_generatedRoot_987Ctx) : mxxMat2 :=
  (parallel_generatedRoot_987.w_94_0 ctx)

structure parallel_generatedRoot_987.Facts_0  (ctx : parallel_generatedRoot_987Ctx) : Prop where
  h2_0 : MxxRuntime.concatColumns (parallel_generatedRoot_987.w_0_0 ctx) (parallel_generatedRoot_987.w_1_0 ctx) (parallel_generatedRoot_987.w_2_0 ctx)
  h6_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (parallel_generatedRoot_987.w_5_0 ctx) (parallel_generatedRoot_987.w_6_0 ctx)
  h7_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (parallel_generatedRoot_987.w_6_0 ctx) (parallel_generatedRoot_987.w_7_0 ctx)
  h10_0 : MxxRuntime.gadgetMatrixRuns backend 4096 25 (parallel_generatedRoot_987.w_10_0 ctx)
  h11_0 : MxxRuntime.centeredRebaseRuns (parallel_generatedRoot_987.w_10_0 ctx) (parallel_generatedRoot_987.w_11_0 ctx)
  h12_0 : MxxRuntime.modulusReduceRuns (parallel_generatedRoot_987.w_11_0 ctx) (parallel_generatedRoot_987.w_12_0 ctx)
  h17_0 : MxxRuntime.centeredRebaseRuns (parallel_generatedRoot_987.w_16_0 ctx) (parallel_generatedRoot_987.w_17_0 ctx)
  h20_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (parallel_generatedRoot_987.w_19_0 ctx) (parallel_generatedRoot_987.w_20_0 ctx)
  h21_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (parallel_generatedRoot_987.w_20_0 ctx) (parallel_generatedRoot_987.w_21_0 ctx)
  t9 : 0 ≤ (0)
  t10 : (0) < 25
  t11 : 0 ≤ 0
  t12 : 0 < 1
  t13 : 1 ≤ 1
  t14 : 0 ≤ (625 * (Int.ofNat i_0))
  t15 : (625 * (Int.ofNat i_0)) < (625 + (625 * (Int.ofNat i_0)))
  t16 : (625 + (625 * (Int.ofNat i_0))) ≤ 22500
  h26_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_987.w_25_0 ctx) 0 1 (625 * (Int.ofNat i_0)) (625 + (625 * (Int.ofNat i_0))) (parallel_generatedRoot_987.w_26_0 ctx)
  h28_0 : MxxRuntime.blockDiagonalRuns 25 (parallel_generatedRoot_987.w_27_0 ctx) (parallel_generatedRoot_987.w_28_0 ctx)
  h29_0 : MxxRuntime.centeredRebaseRuns (parallel_generatedRoot_987.w_28_0 ctx) (parallel_generatedRoot_987.w_29_0 ctx)
  h32_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (parallel_generatedRoot_987.w_31_0 ctx) (parallel_generatedRoot_987.w_32_0 ctx)
  h33_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (parallel_generatedRoot_987.w_32_0 ctx) (parallel_generatedRoot_987.w_33_0 ctx)
  t22 : 0 ≤ 0
  t23 : 0 < 1
  t24 : 1 ≤ 1
  t25 : 0 ≤ 0
  t26 : 0 < 25
  t27 : 25 ≤ 625
  h35_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_987.w_34_0 ctx) 0 1 0 25 (parallel_generatedRoot_987.w_35_0 ctx)
  h37_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (parallel_generatedRoot_987.w_36_0 ctx) (parallel_generatedRoot_987.w_37_0 ctx)
  h38_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (parallel_generatedRoot_987.w_37_0 ctx) (parallel_generatedRoot_987.w_38_0 ctx)
  h40_0 : MxxRuntime.tensorRuns (parallel_generatedRoot_987.w_24_0 ctx) (parallel_generatedRoot_987.w_39_0 ctx) (parallel_generatedRoot_987.w_40_0 ctx)

theorem parallel_generatedRoot_987.facts_0  {ctx : parallel_generatedRoot_987Ctx} {outputs : mxxMat2}
    (h : parallel_generatedRoot_987.constraints_0 backend i_0 (parallel_generatedRoot_987.w_0_0 ctx) (parallel_generatedRoot_987.w_1_0 ctx) (parallel_generatedRoot_987.w_25_0 ctx) (parallel_generatedRoot_987.w_27_0 ctx) (parallel_generatedRoot_987.w_2_0 ctx) (parallel_generatedRoot_987.w_5_0 ctx) (parallel_generatedRoot_987.w_6_0 ctx) (parallel_generatedRoot_987.w_7_0 ctx) (parallel_generatedRoot_987.w_10_0 ctx) (parallel_generatedRoot_987.w_11_0 ctx) (parallel_generatedRoot_987.w_12_0 ctx) (parallel_generatedRoot_987.w_16_0 ctx) (parallel_generatedRoot_987.w_17_0 ctx) (parallel_generatedRoot_987.w_19_0 ctx) (parallel_generatedRoot_987.w_20_0 ctx) (parallel_generatedRoot_987.w_21_0 ctx) (parallel_generatedRoot_987.w_24_0 ctx) (parallel_generatedRoot_987.w_26_0 ctx) (parallel_generatedRoot_987.w_28_0 ctx) (parallel_generatedRoot_987.w_29_0 ctx) (parallel_generatedRoot_987.w_31_0 ctx) (parallel_generatedRoot_987.w_32_0 ctx) (parallel_generatedRoot_987.w_33_0 ctx) (parallel_generatedRoot_987.w_34_0 ctx) (parallel_generatedRoot_987.w_35_0 ctx) (parallel_generatedRoot_987.w_36_0 ctx) (parallel_generatedRoot_987.w_37_0 ctx) (parallel_generatedRoot_987.w_38_0 ctx) (parallel_generatedRoot_987.w_39_0 ctx) (parallel_generatedRoot_987.w_40_0 ctx)) :
    parallel_generatedRoot_987.Facts_0 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_987.Facts_1  (ctx : parallel_generatedRoot_987Ctx) : Prop where
  t32 : 0 ≤ (24)
  h41_0 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0) => sequential_parallel_generatedRoot_987_41 backend hashModel tape (path ++ [41, i]) params i_0 i (current, ((parallel_generatedRoot_987.w_34_0 ctx), ((parallel_generatedRoot_987.w_38_0 ctx), ()))) next) (Int.toNat (24)) (parallel_generatedRoot_987.w_40_0 ctx) (parallel_generatedRoot_987.w_41_0 ctx)
  t34 : 0 ≤ (1)
  h48_2 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_987_48 backend hashModel tape (path ++ [48, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, ((parallel_generatedRoot_987.w_44_0 ctx), ((parallel_generatedRoot_987.w_45_0 ctx), ((parallel_generatedRoot_987.w_46_0 ctx), ((parallel_generatedRoot_987.w_47_0 ctx), ()))))))) next) (Int.toNat (1)) ((parallel_generatedRoot_987.w_43_0 ctx), ((parallel_generatedRoot_987.w_43_0 ctx), ((parallel_generatedRoot_987.w_43_0 ctx), ()))) ((parallel_generatedRoot_987.w_48_0 ctx), ((parallel_generatedRoot_987.w_48_1 ctx), ((parallel_generatedRoot_987.w_48_2 ctx), ())))
  h52_0 : MxxRuntime.concatColumns (parallel_generatedRoot_987.w_50_0 ctx) (parallel_generatedRoot_987.w_51_0 ctx) (parallel_generatedRoot_987.w_52_0 ctx)
  h55_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (parallel_generatedRoot_987.w_54_0 ctx) (parallel_generatedRoot_987.w_55_0 ctx)
  h56_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (parallel_generatedRoot_987.w_55_0 ctx) (parallel_generatedRoot_987.w_56_0 ctx)
  h59_0 : MxxRuntime.gadgetMatrixRuns backend 4096 25 (parallel_generatedRoot_987.w_59_0 ctx)
  h60_0 : MxxRuntime.centeredRebaseRuns (parallel_generatedRoot_987.w_59_0 ctx) (parallel_generatedRoot_987.w_60_0 ctx)
  h61_0 : MxxRuntime.modulusReduceRuns (parallel_generatedRoot_987.w_60_0 ctx) (parallel_generatedRoot_987.w_61_0 ctx)
  h66_0 : MxxRuntime.centeredRebaseRuns (parallel_generatedRoot_987.w_65_0 ctx) (parallel_generatedRoot_987.w_66_0 ctx)
  h69_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (parallel_generatedRoot_987.w_68_0 ctx) (parallel_generatedRoot_987.w_69_0 ctx)
  h70_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (parallel_generatedRoot_987.w_69_0 ctx) (parallel_generatedRoot_987.w_70_0 ctx)
  t45 : 0 ≤ (0)
  t46 : (0) < 25
  t47 : 0 ≤ 0
  t48 : 0 < 1
  t49 : 1 ≤ 1
  t50 : 0 ≤ (625 * (Int.ofNat i_0))
  t51 : (625 * (Int.ofNat i_0)) < (625 + (625 * (Int.ofNat i_0)))
  t52 : (625 + (625 * (Int.ofNat i_0))) ≤ 22500
  h75_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_987.w_74_0 ctx) 0 1 (625 * (Int.ofNat i_0)) (625 + (625 * (Int.ofNat i_0))) (parallel_generatedRoot_987.w_75_0 ctx)
  h76_0 : MxxRuntime.blockDiagonalRuns 25 (parallel_generatedRoot_987.w_27_0 ctx) (parallel_generatedRoot_987.w_76_0 ctx)
  h77_0 : MxxRuntime.centeredRebaseRuns (parallel_generatedRoot_987.w_76_0 ctx) (parallel_generatedRoot_987.w_77_0 ctx)
  h80_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (parallel_generatedRoot_987.w_79_0 ctx) (parallel_generatedRoot_987.w_80_0 ctx)
  h81_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (parallel_generatedRoot_987.w_80_0 ctx) (parallel_generatedRoot_987.w_81_0 ctx)
  t58 : 0 ≤ 0
  t59 : 0 < 1
  t60 : 1 ≤ 1
  t61 : 0 ≤ 0
  t62 : 0 < 25
  t63 : 25 ≤ 625

theorem parallel_generatedRoot_987.facts_1  {ctx : parallel_generatedRoot_987Ctx} {outputs : mxxMat2}
    (h : parallel_generatedRoot_987.constraints_1 backend hashModel tape path params i_0 (parallel_generatedRoot_987.w_27_0 ctx) (parallel_generatedRoot_987.w_44_0 ctx) (parallel_generatedRoot_987.w_45_0 ctx) (parallel_generatedRoot_987.w_46_0 ctx) (parallel_generatedRoot_987.w_47_0 ctx) (parallel_generatedRoot_987.w_50_0 ctx) (parallel_generatedRoot_987.w_51_0 ctx) (parallel_generatedRoot_987.w_74_0 ctx) (parallel_generatedRoot_987.w_34_0 ctx) (parallel_generatedRoot_987.w_38_0 ctx) (parallel_generatedRoot_987.w_40_0 ctx) (parallel_generatedRoot_987.w_41_0 ctx) (parallel_generatedRoot_987.w_43_0 ctx) (parallel_generatedRoot_987.w_48_0 ctx) (parallel_generatedRoot_987.w_48_1 ctx) (parallel_generatedRoot_987.w_48_2 ctx) (parallel_generatedRoot_987.w_52_0 ctx) (parallel_generatedRoot_987.w_54_0 ctx) (parallel_generatedRoot_987.w_55_0 ctx) (parallel_generatedRoot_987.w_56_0 ctx) (parallel_generatedRoot_987.w_59_0 ctx) (parallel_generatedRoot_987.w_60_0 ctx) (parallel_generatedRoot_987.w_61_0 ctx) (parallel_generatedRoot_987.w_65_0 ctx) (parallel_generatedRoot_987.w_66_0 ctx) (parallel_generatedRoot_987.w_68_0 ctx) (parallel_generatedRoot_987.w_69_0 ctx) (parallel_generatedRoot_987.w_70_0 ctx) (parallel_generatedRoot_987.w_75_0 ctx) (parallel_generatedRoot_987.w_76_0 ctx) (parallel_generatedRoot_987.w_77_0 ctx) (parallel_generatedRoot_987.w_79_0 ctx) (parallel_generatedRoot_987.w_80_0 ctx) (parallel_generatedRoot_987.w_81_0 ctx)) :
    parallel_generatedRoot_987.Facts_1 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_987.Facts_2  (ctx : parallel_generatedRoot_987Ctx) (outputs : mxxMat2) : Prop where
  h83_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_987.w_82_0 ctx) 0 1 0 25 (parallel_generatedRoot_987.w_83_0 ctx)
  h85_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (parallel_generatedRoot_987.w_84_0 ctx) (parallel_generatedRoot_987.w_85_0 ctx)
  h86_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (parallel_generatedRoot_987.w_85_0 ctx) (parallel_generatedRoot_987.w_86_0 ctx)
  h88_0 : MxxRuntime.tensorRuns (parallel_generatedRoot_987.w_73_0 ctx) (parallel_generatedRoot_987.w_87_0 ctx) (parallel_generatedRoot_987.w_88_0 ctx)
  t68 : 0 ≤ (24)
  h89_0 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0) => sequential_parallel_generatedRoot_987_89 backend hashModel tape (path ++ [89, i]) params i_0 i (current, ((parallel_generatedRoot_987.w_82_0 ctx), ((parallel_generatedRoot_987.w_86_0 ctx), ()))) next) (Int.toNat (24)) (parallel_generatedRoot_987.w_88_0 ctx) (parallel_generatedRoot_987.w_89_0 ctx)
  t70 : 0 ≤ (1)
  h92_2 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_987_92 backend hashModel tape (path ++ [92, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, ((parallel_generatedRoot_987.w_44_0 ctx), ((parallel_generatedRoot_987.w_45_0 ctx), ((parallel_generatedRoot_987.w_46_0 ctx), ((parallel_generatedRoot_987.w_47_0 ctx), ()))))))) next) (Int.toNat (1)) ((parallel_generatedRoot_987.w_91_0 ctx), ((parallel_generatedRoot_987.w_91_0 ctx), ((parallel_generatedRoot_987.w_91_0 ctx), ()))) ((parallel_generatedRoot_987.w_92_0 ctx), ((parallel_generatedRoot_987.w_92_1 ctx), ((parallel_generatedRoot_987.w_92_2 ctx), ())))
  h94_0 : MxxRuntime.concatRows (parallel_generatedRoot_987.w_49_0 ctx) (parallel_generatedRoot_987.w_93_0 ctx) (parallel_generatedRoot_987.w_94_0 ctx)
  hout : outputs = parallel_generatedRoot_987.outs ctx

theorem parallel_generatedRoot_987.facts_2  {ctx : parallel_generatedRoot_987Ctx} {outputs : mxxMat2}
    (h : parallel_generatedRoot_987.constraints_2 backend hashModel tape path params i_0 (parallel_generatedRoot_987.w_44_0 ctx) (parallel_generatedRoot_987.w_45_0 ctx) (parallel_generatedRoot_987.w_46_0 ctx) (parallel_generatedRoot_987.w_47_0 ctx) (parallel_generatedRoot_987.w_49_0 ctx) (parallel_generatedRoot_987.w_73_0 ctx) (parallel_generatedRoot_987.w_82_0 ctx) (parallel_generatedRoot_987.w_83_0 ctx) (parallel_generatedRoot_987.w_84_0 ctx) (parallel_generatedRoot_987.w_85_0 ctx) (parallel_generatedRoot_987.w_86_0 ctx) (parallel_generatedRoot_987.w_87_0 ctx) (parallel_generatedRoot_987.w_88_0 ctx) (parallel_generatedRoot_987.w_89_0 ctx) (parallel_generatedRoot_987.w_91_0 ctx) (parallel_generatedRoot_987.w_92_0 ctx) (parallel_generatedRoot_987.w_92_1 ctx) (parallel_generatedRoot_987.w_92_2 ctx) (parallel_generatedRoot_987.w_93_0 ctx) (parallel_generatedRoot_987.w_94_0 ctx) outputs) :
    parallel_generatedRoot_987.Facts_2 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_987.Facts  (ctx : parallel_generatedRoot_987Ctx) (outputs : mxxMat2) : Prop where
  c0 : parallel_generatedRoot_987.Facts_0 ctx
  c1 : parallel_generatedRoot_987.Facts_1 ctx
  c2 : parallel_generatedRoot_987.Facts_2 ctx outputs

theorem parallel_generatedRoot_987.facts  (ctx : parallel_generatedRoot_987Ctx) {outputs : mxxMat2}
    (h : parallel_generatedRoot_987.body backend hashModel tape path params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_987.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_987.facts_0 (ctx := ctx) (outputs := outputs) t0.1
  have t1 := t0.2
  have r1 := parallel_generatedRoot_987.facts_1 (ctx := ctx) (outputs := outputs) t1.1
  have t2 := t1.2
  have r2 := parallel_generatedRoot_987.facts_2 (ctx := ctx) (outputs := outputs) t2
  ⟨r0, r1, r2⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_988.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) where
  inputs : (parallel_generatedRoot_988.InputType)
  witness : parallel_generatedRoot_988.Witness

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_988Ctx" => (parallel_generatedRoot_988.Ctx backend hashModel tape path params i_0)

def parallel_generatedRoot_988.w_0_0  (ctx : parallel_generatedRoot_988Ctx) : mxxMat0 :=
  ctx.inputs.1

def parallel_generatedRoot_988.w_1_0  (ctx : parallel_generatedRoot_988Ctx) : mxxMat0 :=
  ctx.inputs.2.1

def parallel_generatedRoot_988.w_4_0  (ctx : parallel_generatedRoot_988Ctx) : mxxMat2 :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_988.w_9_0  (ctx : parallel_generatedRoot_988Ctx) : mxxMat0 :=
  ctx.inputs.2.2.2.1

def parallel_generatedRoot_988.w_25_0  (ctx : parallel_generatedRoot_988Ctx) : mxxMat7 :=
  ctx.inputs.2.2.2.2.1

def parallel_generatedRoot_988.w_27_0  (ctx : parallel_generatedRoot_988Ctx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.1

def parallel_generatedRoot_988.w_44_0  (ctx : parallel_generatedRoot_988Ctx) : mxxMat9 :=
  ctx.inputs.2.2.2.2.2.2.1

def parallel_generatedRoot_988.w_45_0  (ctx : parallel_generatedRoot_988Ctx) : Fin 1 → mxxMat5 :=
  ctx.inputs.2.2.2.2.2.2.2.1

def parallel_generatedRoot_988.w_46_0  (ctx : parallel_generatedRoot_988Ctx) : mxxMat9 :=
  ctx.inputs.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_988.w_47_0  (ctx : parallel_generatedRoot_988Ctx) : mxxMat9 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_988.w_50_0  (ctx : parallel_generatedRoot_988Ctx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_988.w_51_0  (ctx : parallel_generatedRoot_988Ctx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_988.w_58_0  (ctx : parallel_generatedRoot_988Ctx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_988.w_74_0  (ctx : parallel_generatedRoot_988Ctx) : mxxMat7 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def parallel_generatedRoot_988.w_2_0  (ctx : parallel_generatedRoot_988Ctx) : mxxMat11 :=
  ctx.witness.w_2_0

def parallel_generatedRoot_988.w_3_0  (ctx : parallel_generatedRoot_988Ctx) : mxxMat5 :=
  (MxxRuntime.matrixPolynomial [(1606938044255707605848740785410594412685698358681526360823809)] : mxxMat5)

def parallel_generatedRoot_988.w_5_0  (ctx : parallel_generatedRoot_988Ctx) : mxxMat2 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_988.w_3_0 ctx) (parallel_generatedRoot_988.w_4_0 ctx)

def parallel_generatedRoot_988.w_6_0  (ctx : parallel_generatedRoot_988Ctx) : mxxMat4 :=
  ctx.witness.w_6_0

def parallel_generatedRoot_988.w_7_0  (ctx : parallel_generatedRoot_988Ctx) : mxxMat4 :=
  ctx.witness.w_7_0

def parallel_generatedRoot_988.w_8_0  (ctx : parallel_generatedRoot_988Ctx) : mxxMat0 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_988.w_2_0 ctx) (parallel_generatedRoot_988.w_7_0 ctx)

def parallel_generatedRoot_988.w_10_0  (ctx : parallel_generatedRoot_988Ctx) : mxxMat0 :=
  ctx.witness.w_10_0

def parallel_generatedRoot_988.w_11_0  (ctx : parallel_generatedRoot_988Ctx) : mxxMat0 :=
  ctx.witness.w_11_0

def parallel_generatedRoot_988.w_12_0  (ctx : parallel_generatedRoot_988Ctx) : mxxMat24 :=
  ctx.witness.w_12_0

def parallel_generatedRoot_988.w_13_0  (ctx : parallel_generatedRoot_988Ctx) : mxxMat25 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat25)

def parallel_generatedRoot_988.w_14_0  (ctx : parallel_generatedRoot_988Ctx) : mxxMat24 :=
  MxxRuntime.matrixMulScalarRight (parallel_generatedRoot_988.w_12_0 ctx) (parallel_generatedRoot_988.w_13_0 ctx)

def parallel_generatedRoot_988.w_15_0  (ctx : parallel_generatedRoot_988Ctx) : mxxMat25 :=
  (MxxRuntime.matrixPolynomial [(157994188800)] : mxxMat25)

def parallel_generatedRoot_988.w_16_0  (ctx : parallel_generatedRoot_988Ctx) : mxxMat24 :=
  MxxRuntime.matrixMulScalarRight (parallel_generatedRoot_988.w_14_0 ctx) (parallel_generatedRoot_988.w_15_0 ctx)

def parallel_generatedRoot_988.w_17_0  (ctx : parallel_generatedRoot_988Ctx) : mxxMat0 :=
  ctx.witness.w_17_0

def parallel_generatedRoot_988.w_18_0  (ctx : parallel_generatedRoot_988Ctx) : mxxMat5 :=
  (MxxRuntime.matrixPolynomial [(mxxNat4)] : mxxMat5)

def parallel_generatedRoot_988.w_19_0  (ctx : parallel_generatedRoot_988Ctx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarRight (parallel_generatedRoot_988.w_17_0 ctx) (parallel_generatedRoot_988.w_18_0 ctx)

def parallel_generatedRoot_988.w_20_0  (ctx : parallel_generatedRoot_988Ctx) : mxxMat6 :=
  ctx.witness.w_20_0

def parallel_generatedRoot_988.w_21_0  (ctx : parallel_generatedRoot_988Ctx) : mxxMat6 :=
  ctx.witness.w_21_0

def parallel_generatedRoot_988.w_22_0  (ctx : parallel_generatedRoot_988Ctx) : mxxMat0 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_988.w_9_0 ctx) (parallel_generatedRoot_988.w_21_0 ctx)

def parallel_generatedRoot_988.w_23_0  (ctx : parallel_generatedRoot_988Ctx) : mxxMat0 :=
  MxxRuntime.matrixSub (parallel_generatedRoot_988.w_8_0 ctx) (parallel_generatedRoot_988.w_22_0 ctx)

def parallel_generatedRoot_988.w_24_0  (ctx : parallel_generatedRoot_988Ctx) : mxxMat0 :=
  (MxxRuntime.unitRow (0) : mxxMat0)

def parallel_generatedRoot_988.w_26_0  (ctx : parallel_generatedRoot_988Ctx) : mxxMat14 :=
  ctx.witness.w_26_0

def parallel_generatedRoot_988.w_28_0  (ctx : parallel_generatedRoot_988Ctx) : mxxMat15 :=
  ctx.witness.w_28_0

def parallel_generatedRoot_988.w_29_0  (ctx : parallel_generatedRoot_988Ctx) : mxxMat15 :=
  ctx.witness.w_29_0

def parallel_generatedRoot_988.w_30_0  (ctx : parallel_generatedRoot_988Ctx) : mxxMat5 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat5)

def parallel_generatedRoot_988.w_31_0  (ctx : parallel_generatedRoot_988Ctx) : mxxMat15 :=
  MxxRuntime.matrixMulScalarRight (parallel_generatedRoot_988.w_29_0 ctx) (parallel_generatedRoot_988.w_30_0 ctx)

def parallel_generatedRoot_988.w_32_0  (ctx : parallel_generatedRoot_988Ctx) : mxxMat16 :=
  ctx.witness.w_32_0

def parallel_generatedRoot_988.w_33_0  (ctx : parallel_generatedRoot_988Ctx) : mxxMat16 :=
  ctx.witness.w_33_0

def parallel_generatedRoot_988.w_34_0  (ctx : parallel_generatedRoot_988Ctx) : mxxMat14 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_988.w_26_0 ctx) (parallel_generatedRoot_988.w_33_0 ctx)

def parallel_generatedRoot_988.w_35_0  (ctx : parallel_generatedRoot_988Ctx) : mxxMat0 :=
  ctx.witness.w_35_0

def parallel_generatedRoot_988.w_36_0  (ctx : parallel_generatedRoot_988Ctx) : mxxMat5 :=
  (MxxRuntime.matrixPolynomial [(1606938044255707605848740785410594412685698358681526360823809)] : mxxMat5)

def parallel_generatedRoot_988.w_37_0  (ctx : parallel_generatedRoot_988Ctx) : mxxMat17 :=
  ctx.witness.w_37_0

def parallel_generatedRoot_988.w_38_0  (ctx : parallel_generatedRoot_988Ctx) : mxxMat17 :=
  ctx.witness.w_38_0

def parallel_generatedRoot_988.w_39_0  (ctx : parallel_generatedRoot_988Ctx) : mxxMat5 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_988.w_35_0 ctx) (parallel_generatedRoot_988.w_38_0 ctx)

def parallel_generatedRoot_988.w_40_0  (ctx : parallel_generatedRoot_988Ctx) : mxxMat0 :=
  ctx.witness.w_40_0

def parallel_generatedRoot_988.w_41_0  (ctx : parallel_generatedRoot_988Ctx) : mxxMat0 :=
  ctx.witness.w_41_0

def parallel_generatedRoot_988.w_42_0  (ctx : parallel_generatedRoot_988Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_988.w_23_0 ctx) (parallel_generatedRoot_988.w_41_0 ctx)

def parallel_generatedRoot_988.w_43_0  (ctx : parallel_generatedRoot_988Ctx) : mxxMat0 :=
  (0 : mxxMat0)

def parallel_generatedRoot_988.w_48_0  (ctx : parallel_generatedRoot_988Ctx) : mxxMat0 :=
  ctx.witness.w_48_0

def parallel_generatedRoot_988.w_48_1  (ctx : parallel_generatedRoot_988Ctx) : mxxMat0 :=
  ctx.witness.w_48_1

def parallel_generatedRoot_988.w_48_2  (ctx : parallel_generatedRoot_988Ctx) : mxxMat0 :=
  ctx.witness.w_48_2

def parallel_generatedRoot_988.w_49_0  (ctx : parallel_generatedRoot_988Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_988.w_42_0 ctx) (parallel_generatedRoot_988.w_48_1 ctx)

def parallel_generatedRoot_988.w_52_0  (ctx : parallel_generatedRoot_988Ctx) : mxxMat11 :=
  ctx.witness.w_52_0

def parallel_generatedRoot_988.w_53_0  (ctx : parallel_generatedRoot_988Ctx) : mxxMat5 :=
  (MxxRuntime.matrixPolynomial [(1606938044255707605848740785410594412685698358681526360823809)] : mxxMat5)

def parallel_generatedRoot_988.w_54_0  (ctx : parallel_generatedRoot_988Ctx) : mxxMat2 :=
  MxxRuntime.matrixMulScalarLeft (parallel_generatedRoot_988.w_53_0 ctx) (parallel_generatedRoot_988.w_4_0 ctx)

def parallel_generatedRoot_988.w_55_0  (ctx : parallel_generatedRoot_988Ctx) : mxxMat4 :=
  ctx.witness.w_55_0

def parallel_generatedRoot_988.w_56_0  (ctx : parallel_generatedRoot_988Ctx) : mxxMat4 :=
  ctx.witness.w_56_0

def parallel_generatedRoot_988.w_57_0  (ctx : parallel_generatedRoot_988Ctx) : mxxMat0 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_988.w_52_0 ctx) (parallel_generatedRoot_988.w_56_0 ctx)

def parallel_generatedRoot_988.w_59_0  (ctx : parallel_generatedRoot_988Ctx) : mxxMat0 :=
  ctx.witness.w_59_0

def parallel_generatedRoot_988.w_60_0  (ctx : parallel_generatedRoot_988Ctx) : mxxMat0 :=
  ctx.witness.w_60_0

def parallel_generatedRoot_988.w_61_0  (ctx : parallel_generatedRoot_988Ctx) : mxxMat24 :=
  ctx.witness.w_61_0

def parallel_generatedRoot_988.w_62_0  (ctx : parallel_generatedRoot_988Ctx) : mxxMat25 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat25)

def parallel_generatedRoot_988.w_63_0  (ctx : parallel_generatedRoot_988Ctx) : mxxMat24 :=
  MxxRuntime.matrixMulScalarRight (parallel_generatedRoot_988.w_61_0 ctx) (parallel_generatedRoot_988.w_62_0 ctx)

def parallel_generatedRoot_988.w_64_0  (ctx : parallel_generatedRoot_988Ctx) : mxxMat25 :=
  (MxxRuntime.matrixPolynomial [(157994188800)] : mxxMat25)

def parallel_generatedRoot_988.w_65_0  (ctx : parallel_generatedRoot_988Ctx) : mxxMat24 :=
  MxxRuntime.matrixMulScalarRight (parallel_generatedRoot_988.w_63_0 ctx) (parallel_generatedRoot_988.w_64_0 ctx)

def parallel_generatedRoot_988.w_66_0  (ctx : parallel_generatedRoot_988Ctx) : mxxMat0 :=
  ctx.witness.w_66_0

def parallel_generatedRoot_988.w_67_0  (ctx : parallel_generatedRoot_988Ctx) : mxxMat5 :=
  (MxxRuntime.matrixPolynomial [(mxxNat4)] : mxxMat5)

def parallel_generatedRoot_988.w_68_0  (ctx : parallel_generatedRoot_988Ctx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarRight (parallel_generatedRoot_988.w_66_0 ctx) (parallel_generatedRoot_988.w_67_0 ctx)

def parallel_generatedRoot_988.w_69_0  (ctx : parallel_generatedRoot_988Ctx) : mxxMat6 :=
  ctx.witness.w_69_0

def parallel_generatedRoot_988.w_70_0  (ctx : parallel_generatedRoot_988Ctx) : mxxMat6 :=
  ctx.witness.w_70_0

def parallel_generatedRoot_988.w_71_0  (ctx : parallel_generatedRoot_988Ctx) : mxxMat0 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_988.w_58_0 ctx) (parallel_generatedRoot_988.w_70_0 ctx)

def parallel_generatedRoot_988.w_72_0  (ctx : parallel_generatedRoot_988Ctx) : mxxMat0 :=
  MxxRuntime.matrixSub (parallel_generatedRoot_988.w_57_0 ctx) (parallel_generatedRoot_988.w_71_0 ctx)

def parallel_generatedRoot_988.w_73_0  (ctx : parallel_generatedRoot_988Ctx) : mxxMat0 :=
  (MxxRuntime.unitRow (0) : mxxMat0)

def parallel_generatedRoot_988.w_75_0  (ctx : parallel_generatedRoot_988Ctx) : mxxMat14 :=
  ctx.witness.w_75_0

def parallel_generatedRoot_988.w_76_0  (ctx : parallel_generatedRoot_988Ctx) : mxxMat15 :=
  ctx.witness.w_76_0

def parallel_generatedRoot_988.w_77_0  (ctx : parallel_generatedRoot_988Ctx) : mxxMat15 :=
  ctx.witness.w_77_0

def parallel_generatedRoot_988.w_78_0  (ctx : parallel_generatedRoot_988Ctx) : mxxMat5 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat5)

def parallel_generatedRoot_988.w_79_0  (ctx : parallel_generatedRoot_988Ctx) : mxxMat15 :=
  MxxRuntime.matrixMulScalarRight (parallel_generatedRoot_988.w_77_0 ctx) (parallel_generatedRoot_988.w_78_0 ctx)

def parallel_generatedRoot_988.w_80_0  (ctx : parallel_generatedRoot_988Ctx) : mxxMat16 :=
  ctx.witness.w_80_0

def parallel_generatedRoot_988.w_81_0  (ctx : parallel_generatedRoot_988Ctx) : mxxMat16 :=
  ctx.witness.w_81_0

def parallel_generatedRoot_988.w_82_0  (ctx : parallel_generatedRoot_988Ctx) : mxxMat14 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_988.w_75_0 ctx) (parallel_generatedRoot_988.w_81_0 ctx)

def parallel_generatedRoot_988.w_83_0  (ctx : parallel_generatedRoot_988Ctx) : mxxMat0 :=
  ctx.witness.w_83_0

def parallel_generatedRoot_988.w_84_0  (ctx : parallel_generatedRoot_988Ctx) : mxxMat5 :=
  (MxxRuntime.matrixPolynomial [(1606938044255707605848740785410594412685698358681526360823809)] : mxxMat5)

def parallel_generatedRoot_988.w_85_0  (ctx : parallel_generatedRoot_988Ctx) : mxxMat17 :=
  ctx.witness.w_85_0

def parallel_generatedRoot_988.w_86_0  (ctx : parallel_generatedRoot_988Ctx) : mxxMat17 :=
  ctx.witness.w_86_0

def parallel_generatedRoot_988.w_87_0  (ctx : parallel_generatedRoot_988Ctx) : mxxMat5 :=
  MxxRuntime.matrixMul (parallel_generatedRoot_988.w_83_0 ctx) (parallel_generatedRoot_988.w_86_0 ctx)

def parallel_generatedRoot_988.w_88_0  (ctx : parallel_generatedRoot_988Ctx) : mxxMat0 :=
  ctx.witness.w_88_0

def parallel_generatedRoot_988.w_89_0  (ctx : parallel_generatedRoot_988Ctx) : mxxMat0 :=
  ctx.witness.w_89_0

def parallel_generatedRoot_988.w_90_0  (ctx : parallel_generatedRoot_988Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_988.w_72_0 ctx) (parallel_generatedRoot_988.w_89_0 ctx)

def parallel_generatedRoot_988.w_91_0  (ctx : parallel_generatedRoot_988Ctx) : mxxMat0 :=
  (0 : mxxMat0)

def parallel_generatedRoot_988.w_92_0  (ctx : parallel_generatedRoot_988Ctx) : mxxMat0 :=
  ctx.witness.w_92_0

def parallel_generatedRoot_988.w_92_1  (ctx : parallel_generatedRoot_988Ctx) : mxxMat0 :=
  ctx.witness.w_92_1

def parallel_generatedRoot_988.w_92_2  (ctx : parallel_generatedRoot_988Ctx) : mxxMat0 :=
  ctx.witness.w_92_2

def parallel_generatedRoot_988.w_93_0  (ctx : parallel_generatedRoot_988Ctx) : mxxMat0 :=
  MxxRuntime.matrixAdd (parallel_generatedRoot_988.w_90_0 ctx) (parallel_generatedRoot_988.w_92_2 ctx)

def parallel_generatedRoot_988.w_94_0  (ctx : parallel_generatedRoot_988Ctx) : mxxMat2 :=
  ctx.witness.w_94_0

def parallel_generatedRoot_988.outs  (ctx : parallel_generatedRoot_988Ctx) : mxxMat2 :=
  (parallel_generatedRoot_988.w_94_0 ctx)

structure parallel_generatedRoot_988.Facts_0  (ctx : parallel_generatedRoot_988Ctx) : Prop where
  h2_0 : MxxRuntime.concatColumns (parallel_generatedRoot_988.w_0_0 ctx) (parallel_generatedRoot_988.w_1_0 ctx) (parallel_generatedRoot_988.w_2_0 ctx)
  h6_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (parallel_generatedRoot_988.w_5_0 ctx) (parallel_generatedRoot_988.w_6_0 ctx)
  h7_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (parallel_generatedRoot_988.w_6_0 ctx) (parallel_generatedRoot_988.w_7_0 ctx)
  h10_0 : MxxRuntime.gadgetMatrixRuns backend 4096 25 (parallel_generatedRoot_988.w_10_0 ctx)
  h11_0 : MxxRuntime.centeredRebaseRuns (parallel_generatedRoot_988.w_10_0 ctx) (parallel_generatedRoot_988.w_11_0 ctx)
  h12_0 : MxxRuntime.modulusReduceRuns (parallel_generatedRoot_988.w_11_0 ctx) (parallel_generatedRoot_988.w_12_0 ctx)
  h17_0 : MxxRuntime.centeredRebaseRuns (parallel_generatedRoot_988.w_16_0 ctx) (parallel_generatedRoot_988.w_17_0 ctx)
  h20_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (parallel_generatedRoot_988.w_19_0 ctx) (parallel_generatedRoot_988.w_20_0 ctx)
  h21_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (parallel_generatedRoot_988.w_20_0 ctx) (parallel_generatedRoot_988.w_21_0 ctx)
  t9 : 0 ≤ (0)
  t10 : (0) < 25
  t11 : 0 ≤ 0
  t12 : 0 < 1
  t13 : 1 ≤ 1
  t14 : 0 ≤ (625 * (Int.ofNat i_0))
  t15 : (625 * (Int.ofNat i_0)) < (625 + (625 * (Int.ofNat i_0)))
  t16 : (625 + (625 * (Int.ofNat i_0))) ≤ 22500
  h26_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_988.w_25_0 ctx) 0 1 (625 * (Int.ofNat i_0)) (625 + (625 * (Int.ofNat i_0))) (parallel_generatedRoot_988.w_26_0 ctx)
  h28_0 : MxxRuntime.blockDiagonalRuns 25 (parallel_generatedRoot_988.w_27_0 ctx) (parallel_generatedRoot_988.w_28_0 ctx)
  h29_0 : MxxRuntime.centeredRebaseRuns (parallel_generatedRoot_988.w_28_0 ctx) (parallel_generatedRoot_988.w_29_0 ctx)
  h32_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (parallel_generatedRoot_988.w_31_0 ctx) (parallel_generatedRoot_988.w_32_0 ctx)
  h33_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (parallel_generatedRoot_988.w_32_0 ctx) (parallel_generatedRoot_988.w_33_0 ctx)
  t22 : 0 ≤ 0
  t23 : 0 < 1
  t24 : 1 ≤ 1
  t25 : 0 ≤ 0
  t26 : 0 < 25
  t27 : 25 ≤ 625
  h35_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_988.w_34_0 ctx) 0 1 0 25 (parallel_generatedRoot_988.w_35_0 ctx)
  h37_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (parallel_generatedRoot_988.w_36_0 ctx) (parallel_generatedRoot_988.w_37_0 ctx)
  h38_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (parallel_generatedRoot_988.w_37_0 ctx) (parallel_generatedRoot_988.w_38_0 ctx)
  h40_0 : MxxRuntime.tensorRuns (parallel_generatedRoot_988.w_24_0 ctx) (parallel_generatedRoot_988.w_39_0 ctx) (parallel_generatedRoot_988.w_40_0 ctx)

theorem parallel_generatedRoot_988.facts_0  {ctx : parallel_generatedRoot_988Ctx} {outputs : mxxMat2}
    (h : parallel_generatedRoot_988.constraints_0 backend i_0 (parallel_generatedRoot_988.w_0_0 ctx) (parallel_generatedRoot_988.w_1_0 ctx) (parallel_generatedRoot_988.w_25_0 ctx) (parallel_generatedRoot_988.w_27_0 ctx) (parallel_generatedRoot_988.w_2_0 ctx) (parallel_generatedRoot_988.w_5_0 ctx) (parallel_generatedRoot_988.w_6_0 ctx) (parallel_generatedRoot_988.w_7_0 ctx) (parallel_generatedRoot_988.w_10_0 ctx) (parallel_generatedRoot_988.w_11_0 ctx) (parallel_generatedRoot_988.w_12_0 ctx) (parallel_generatedRoot_988.w_16_0 ctx) (parallel_generatedRoot_988.w_17_0 ctx) (parallel_generatedRoot_988.w_19_0 ctx) (parallel_generatedRoot_988.w_20_0 ctx) (parallel_generatedRoot_988.w_21_0 ctx) (parallel_generatedRoot_988.w_24_0 ctx) (parallel_generatedRoot_988.w_26_0 ctx) (parallel_generatedRoot_988.w_28_0 ctx) (parallel_generatedRoot_988.w_29_0 ctx) (parallel_generatedRoot_988.w_31_0 ctx) (parallel_generatedRoot_988.w_32_0 ctx) (parallel_generatedRoot_988.w_33_0 ctx) (parallel_generatedRoot_988.w_34_0 ctx) (parallel_generatedRoot_988.w_35_0 ctx) (parallel_generatedRoot_988.w_36_0 ctx) (parallel_generatedRoot_988.w_37_0 ctx) (parallel_generatedRoot_988.w_38_0 ctx) (parallel_generatedRoot_988.w_39_0 ctx) (parallel_generatedRoot_988.w_40_0 ctx)) :
    parallel_generatedRoot_988.Facts_0 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_988.Facts_1  (ctx : parallel_generatedRoot_988Ctx) : Prop where
  t32 : 0 ≤ (24)
  h41_0 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0) => sequential_parallel_generatedRoot_988_41 backend hashModel tape (path ++ [41, i]) params i_0 i (current, ((parallel_generatedRoot_988.w_34_0 ctx), ((parallel_generatedRoot_988.w_38_0 ctx), ()))) next) (Int.toNat (24)) (parallel_generatedRoot_988.w_40_0 ctx) (parallel_generatedRoot_988.w_41_0 ctx)
  t34 : 0 ≤ (1)
  h48_2 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_988_48 backend hashModel tape (path ++ [48, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, ((parallel_generatedRoot_988.w_44_0 ctx), ((parallel_generatedRoot_988.w_45_0 ctx), ((parallel_generatedRoot_988.w_46_0 ctx), ((parallel_generatedRoot_988.w_47_0 ctx), ()))))))) next) (Int.toNat (1)) ((parallel_generatedRoot_988.w_43_0 ctx), ((parallel_generatedRoot_988.w_43_0 ctx), ((parallel_generatedRoot_988.w_43_0 ctx), ()))) ((parallel_generatedRoot_988.w_48_0 ctx), ((parallel_generatedRoot_988.w_48_1 ctx), ((parallel_generatedRoot_988.w_48_2 ctx), ())))
  h52_0 : MxxRuntime.concatColumns (parallel_generatedRoot_988.w_50_0 ctx) (parallel_generatedRoot_988.w_51_0 ctx) (parallel_generatedRoot_988.w_52_0 ctx)
  h55_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (parallel_generatedRoot_988.w_54_0 ctx) (parallel_generatedRoot_988.w_55_0 ctx)
  h56_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (parallel_generatedRoot_988.w_55_0 ctx) (parallel_generatedRoot_988.w_56_0 ctx)
  h59_0 : MxxRuntime.gadgetMatrixRuns backend 4096 25 (parallel_generatedRoot_988.w_59_0 ctx)
  h60_0 : MxxRuntime.centeredRebaseRuns (parallel_generatedRoot_988.w_59_0 ctx) (parallel_generatedRoot_988.w_60_0 ctx)
  h61_0 : MxxRuntime.modulusReduceRuns (parallel_generatedRoot_988.w_60_0 ctx) (parallel_generatedRoot_988.w_61_0 ctx)
  h66_0 : MxxRuntime.centeredRebaseRuns (parallel_generatedRoot_988.w_65_0 ctx) (parallel_generatedRoot_988.w_66_0 ctx)
  h69_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (parallel_generatedRoot_988.w_68_0 ctx) (parallel_generatedRoot_988.w_69_0 ctx)
  h70_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (parallel_generatedRoot_988.w_69_0 ctx) (parallel_generatedRoot_988.w_70_0 ctx)
  t45 : 0 ≤ (0)
  t46 : (0) < 25
  t47 : 0 ≤ 0
  t48 : 0 < 1
  t49 : 1 ≤ 1
  t50 : 0 ≤ (625 * (Int.ofNat i_0))
  t51 : (625 * (Int.ofNat i_0)) < (625 + (625 * (Int.ofNat i_0)))
  t52 : (625 + (625 * (Int.ofNat i_0))) ≤ 22500
  h75_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_988.w_74_0 ctx) 0 1 (625 * (Int.ofNat i_0)) (625 + (625 * (Int.ofNat i_0))) (parallel_generatedRoot_988.w_75_0 ctx)
  h76_0 : MxxRuntime.blockDiagonalRuns 25 (parallel_generatedRoot_988.w_27_0 ctx) (parallel_generatedRoot_988.w_76_0 ctx)
  h77_0 : MxxRuntime.centeredRebaseRuns (parallel_generatedRoot_988.w_76_0 ctx) (parallel_generatedRoot_988.w_77_0 ctx)
  h80_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (parallel_generatedRoot_988.w_79_0 ctx) (parallel_generatedRoot_988.w_80_0 ctx)
  h81_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (parallel_generatedRoot_988.w_80_0 ctx) (parallel_generatedRoot_988.w_81_0 ctx)
  t58 : 0 ≤ 0
  t59 : 0 < 1
  t60 : 1 ≤ 1
  t61 : 0 ≤ 0
  t62 : 0 < 25
  t63 : 25 ≤ 625

theorem parallel_generatedRoot_988.facts_1  {ctx : parallel_generatedRoot_988Ctx} {outputs : mxxMat2}
    (h : parallel_generatedRoot_988.constraints_1 backend hashModel tape path params i_0 (parallel_generatedRoot_988.w_27_0 ctx) (parallel_generatedRoot_988.w_44_0 ctx) (parallel_generatedRoot_988.w_45_0 ctx) (parallel_generatedRoot_988.w_46_0 ctx) (parallel_generatedRoot_988.w_47_0 ctx) (parallel_generatedRoot_988.w_50_0 ctx) (parallel_generatedRoot_988.w_51_0 ctx) (parallel_generatedRoot_988.w_74_0 ctx) (parallel_generatedRoot_988.w_34_0 ctx) (parallel_generatedRoot_988.w_38_0 ctx) (parallel_generatedRoot_988.w_40_0 ctx) (parallel_generatedRoot_988.w_41_0 ctx) (parallel_generatedRoot_988.w_43_0 ctx) (parallel_generatedRoot_988.w_48_0 ctx) (parallel_generatedRoot_988.w_48_1 ctx) (parallel_generatedRoot_988.w_48_2 ctx) (parallel_generatedRoot_988.w_52_0 ctx) (parallel_generatedRoot_988.w_54_0 ctx) (parallel_generatedRoot_988.w_55_0 ctx) (parallel_generatedRoot_988.w_56_0 ctx) (parallel_generatedRoot_988.w_59_0 ctx) (parallel_generatedRoot_988.w_60_0 ctx) (parallel_generatedRoot_988.w_61_0 ctx) (parallel_generatedRoot_988.w_65_0 ctx) (parallel_generatedRoot_988.w_66_0 ctx) (parallel_generatedRoot_988.w_68_0 ctx) (parallel_generatedRoot_988.w_69_0 ctx) (parallel_generatedRoot_988.w_70_0 ctx) (parallel_generatedRoot_988.w_75_0 ctx) (parallel_generatedRoot_988.w_76_0 ctx) (parallel_generatedRoot_988.w_77_0 ctx) (parallel_generatedRoot_988.w_79_0 ctx) (parallel_generatedRoot_988.w_80_0 ctx) (parallel_generatedRoot_988.w_81_0 ctx)) :
    parallel_generatedRoot_988.Facts_1 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_988.Facts_2  (ctx : parallel_generatedRoot_988Ctx) (outputs : mxxMat2) : Prop where
  h83_0 : MxxRuntime.sliceMatrix (parallel_generatedRoot_988.w_82_0 ctx) 0 1 0 25 (parallel_generatedRoot_988.w_83_0 ctx)
  h85_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (parallel_generatedRoot_988.w_84_0 ctx) (parallel_generatedRoot_988.w_85_0 ctx)
  h86_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (parallel_generatedRoot_988.w_85_0 ctx) (parallel_generatedRoot_988.w_86_0 ctx)
  h88_0 : MxxRuntime.tensorRuns (parallel_generatedRoot_988.w_73_0 ctx) (parallel_generatedRoot_988.w_87_0 ctx) (parallel_generatedRoot_988.w_88_0 ctx)
  t68 : 0 ≤ (24)
  h89_0 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0) => sequential_parallel_generatedRoot_988_89 backend hashModel tape (path ++ [89, i]) params i_0 i (current, ((parallel_generatedRoot_988.w_82_0 ctx), ((parallel_generatedRoot_988.w_86_0 ctx), ()))) next) (Int.toNat (24)) (parallel_generatedRoot_988.w_88_0 ctx) (parallel_generatedRoot_988.w_89_0 ctx)
  t70 : 0 ≤ (1)
  h92_2 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_988_92 backend hashModel tape (path ++ [92, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, ((parallel_generatedRoot_988.w_44_0 ctx), ((parallel_generatedRoot_988.w_45_0 ctx), ((parallel_generatedRoot_988.w_46_0 ctx), ((parallel_generatedRoot_988.w_47_0 ctx), ()))))))) next) (Int.toNat (1)) ((parallel_generatedRoot_988.w_91_0 ctx), ((parallel_generatedRoot_988.w_91_0 ctx), ((parallel_generatedRoot_988.w_91_0 ctx), ()))) ((parallel_generatedRoot_988.w_92_0 ctx), ((parallel_generatedRoot_988.w_92_1 ctx), ((parallel_generatedRoot_988.w_92_2 ctx), ())))
  h94_0 : MxxRuntime.concatRows (parallel_generatedRoot_988.w_49_0 ctx) (parallel_generatedRoot_988.w_93_0 ctx) (parallel_generatedRoot_988.w_94_0 ctx)
  hout : outputs = parallel_generatedRoot_988.outs ctx

theorem parallel_generatedRoot_988.facts_2  {ctx : parallel_generatedRoot_988Ctx} {outputs : mxxMat2}
    (h : parallel_generatedRoot_988.constraints_2 backend hashModel tape path params i_0 (parallel_generatedRoot_988.w_44_0 ctx) (parallel_generatedRoot_988.w_45_0 ctx) (parallel_generatedRoot_988.w_46_0 ctx) (parallel_generatedRoot_988.w_47_0 ctx) (parallel_generatedRoot_988.w_49_0 ctx) (parallel_generatedRoot_988.w_73_0 ctx) (parallel_generatedRoot_988.w_82_0 ctx) (parallel_generatedRoot_988.w_83_0 ctx) (parallel_generatedRoot_988.w_84_0 ctx) (parallel_generatedRoot_988.w_85_0 ctx) (parallel_generatedRoot_988.w_86_0 ctx) (parallel_generatedRoot_988.w_87_0 ctx) (parallel_generatedRoot_988.w_88_0 ctx) (parallel_generatedRoot_988.w_89_0 ctx) (parallel_generatedRoot_988.w_91_0 ctx) (parallel_generatedRoot_988.w_92_0 ctx) (parallel_generatedRoot_988.w_92_1 ctx) (parallel_generatedRoot_988.w_92_2 ctx) (parallel_generatedRoot_988.w_93_0 ctx) (parallel_generatedRoot_988.w_94_0 ctx) outputs) :
    parallel_generatedRoot_988.Facts_2 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_988.Facts  (ctx : parallel_generatedRoot_988Ctx) (outputs : mxxMat2) : Prop where
  c0 : parallel_generatedRoot_988.Facts_0 ctx
  c1 : parallel_generatedRoot_988.Facts_1 ctx
  c2 : parallel_generatedRoot_988.Facts_2 ctx outputs

theorem parallel_generatedRoot_988.facts  (ctx : parallel_generatedRoot_988Ctx) {outputs : mxxMat2}
    (h : parallel_generatedRoot_988.body backend hashModel tape path params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_988.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_988.facts_0 (ctx := ctx) (outputs := outputs) t0.1
  have t1 := t0.2
  have r1 := parallel_generatedRoot_988.facts_1 (ctx := ctx) (outputs := outputs) t1.1
  have t2 := t1.2
  have r2 := parallel_generatedRoot_988.facts_2 (ctx := ctx) (outputs := outputs) t2
  ⟨r0, r1, r2⟩

end

section
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params}
local notation "generatedRootCtx" => (generatedRoot.Ctx backend hashModel tape path params)

structure generatedRoot.Facts_0  (ctx : generatedRootCtx) : Prop where
  h17_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((generatedRoot.w_5_0 ctx)) (generatedRoot.w_17_0 ctx)
  h18_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) ((generatedRoot.w_5_0 ctx)) (generatedRoot.w_18_0 ctx)
  h21_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((generatedRoot.w_5_0 ctx)) (generatedRoot.w_21_0 ctx)
  h22_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) ((generatedRoot.w_5_0 ctx)) (generatedRoot.w_22_0 ctx)
  t4 : (16) = 16
  h27_5 : (∀ i : Fin 16, ∃ w_14_0_gathered : mxxMat0, ∃ w_15_0_gathered : mxxMat0, ∃ w_20_0_gathered : mxxMat5, 0 ≤ ((generatedRoot.w_16_0 ctx) i) ∧ ((generatedRoot.w_16_0 ctx) i) < 2 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_14_0 ctx) ((generatedRoot.w_16_0 ctx) i) w_14_0_gathered ∧ 0 ≤ ((generatedRoot.w_16_0 ctx) i) ∧ ((generatedRoot.w_16_0 ctx) i) < 2 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_15_0 ctx) ((generatedRoot.w_16_0 ctx) i) w_15_0_gathered ∧ 0 ≤ ((generatedRoot.w_16_0 ctx) i) ∧ ((generatedRoot.w_16_0 ctx) i) < 2 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_20_0 ctx) ((generatedRoot.w_16_0 ctx) i) w_20_0_gathered ∧ parallel_generatedRoot_27 backend hashModel tape (path ++ [27, i.val]) params i (((generatedRoot.w_16_0 ctx) i), (w_14_0_gathered, ((generatedRoot.w_5_0 ctx), (w_15_0_gathered, ((generatedRoot.w_17_0 ctx), ((generatedRoot.w_18_0 ctx), ((generatedRoot.w_19_0 ctx), (w_20_0_gathered, ((generatedRoot.w_21_0 ctx), ((generatedRoot.w_22_0 ctx), ((generatedRoot.w_23_0 ctx), ((generatedRoot.w_24_0 ctx), (((generatedRoot.w_25_0 ctx) i), (((generatedRoot.w_26_0 ctx) i), ())))))))))))))) (((generatedRoot.w_27_0 ctx) i), (((generatedRoot.w_27_1 ctx) i), (((generatedRoot.w_27_2 ctx) i), (((generatedRoot.w_27_3 ctx) i), (((generatedRoot.w_27_4 ctx) i), (((generatedRoot.w_27_5 ctx) i), ())))))))
  t6 : (8) = 8
  h28_5 : (∀ i : Fin 8, parallel_generatedRoot_28 backend hashModel tape (path ++ [28, i.val]) params i ((generatedRoot.w_27_0 ctx), ((generatedRoot.w_27_1 ctx), ((generatedRoot.w_27_2 ctx), ((generatedRoot.w_27_3 ctx), ((generatedRoot.w_27_4 ctx), ((generatedRoot.w_27_5 ctx), ())))))) (((generatedRoot.w_28_0 ctx) i), (((generatedRoot.w_28_1 ctx) i), (((generatedRoot.w_28_2 ctx) i), (((generatedRoot.w_28_3 ctx) i), (((generatedRoot.w_28_4 ctx) i), (((generatedRoot.w_28_5 ctx) i), ())))))))
  t8 : (16) = 16
  h31_5 : (∀ i : Fin 16, ∃ w_14_0_gathered : mxxMat0, ∃ w_15_0_gathered : mxxMat0, ∃ w_20_0_gathered : mxxMat5, 0 ≤ ((generatedRoot.w_13_0 ctx) i) ∧ ((generatedRoot.w_13_0 ctx) i) < 2 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_14_0 ctx) ((generatedRoot.w_13_0 ctx) i) w_14_0_gathered ∧ 0 ≤ ((generatedRoot.w_13_0 ctx) i) ∧ ((generatedRoot.w_13_0 ctx) i) < 2 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_15_0 ctx) ((generatedRoot.w_13_0 ctx) i) w_15_0_gathered ∧ 0 ≤ ((generatedRoot.w_13_0 ctx) i) ∧ ((generatedRoot.w_13_0 ctx) i) < 2 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_20_0 ctx) ((generatedRoot.w_13_0 ctx) i) w_20_0_gathered ∧ parallel_generatedRoot_31 backend hashModel tape (path ++ [31, i.val]) params i (((generatedRoot.w_13_0 ctx) i), (w_14_0_gathered, ((generatedRoot.w_5_0 ctx), (w_15_0_gathered, ((generatedRoot.w_28_0 ctx), ((generatedRoot.w_28_2 ctx), ((generatedRoot.w_28_4 ctx), (w_20_0_gathered, ((generatedRoot.w_28_1 ctx), ((generatedRoot.w_28_3 ctx), ((generatedRoot.w_28_5 ctx), ((generatedRoot.w_24_0 ctx), (((generatedRoot.w_29_0 ctx) i), (((generatedRoot.w_30_0 ctx) i), ())))))))))))))) (((generatedRoot.w_31_0 ctx) i), (((generatedRoot.w_31_1 ctx) i), (((generatedRoot.w_31_2 ctx) i), (((generatedRoot.w_31_3 ctx) i), (((generatedRoot.w_31_4 ctx) i), (((generatedRoot.w_31_5 ctx) i), ())))))))
  t10 : (8) = 8
  h32_5 : (∀ i : Fin 8, parallel_generatedRoot_32 backend hashModel tape (path ++ [32, i.val]) params i ((generatedRoot.w_31_0 ctx), ((generatedRoot.w_31_1 ctx), ((generatedRoot.w_31_2 ctx), ((generatedRoot.w_31_3 ctx), ((generatedRoot.w_31_4 ctx), ((generatedRoot.w_31_5 ctx), ())))))) (((generatedRoot.w_32_0 ctx) i), (((generatedRoot.w_32_1 ctx) i), (((generatedRoot.w_32_2 ctx) i), (((generatedRoot.w_32_3 ctx) i), (((generatedRoot.w_32_4 ctx) i), (((generatedRoot.w_32_5 ctx) i), ())))))))
  t12 : (8) = 8
  h33_5 : (∀ i : Fin 8, parallel_generatedRoot_33 backend hashModel tape (path ++ [33, i.val]) params i ((generatedRoot.w_7_0 ctx), ((generatedRoot.w_8_0 ctx), ((generatedRoot.w_9_0 ctx), ((generatedRoot.w_10_0 ctx), ((generatedRoot.w_11_0 ctx), ((generatedRoot.w_5_0 ctx), ((generatedRoot.w_12_0 ctx), (((generatedRoot.w_32_4 ctx) i), (((generatedRoot.w_32_0 ctx) i), (((generatedRoot.w_32_2 ctx) i), (((generatedRoot.w_32_5 ctx) i), (((generatedRoot.w_32_1 ctx) i), (((generatedRoot.w_32_3 ctx) i), ()))))))))))))) (((generatedRoot.w_33_0 ctx) i), (((generatedRoot.w_33_1 ctx) i), (((generatedRoot.w_33_2 ctx) i), (((generatedRoot.w_33_3 ctx) i), (((generatedRoot.w_33_4 ctx) i), (((generatedRoot.w_33_5 ctx) i), ())))))))
  t14 : (8) = 8
  h34_5 : (∀ i : Fin 8, parallel_generatedRoot_34 backend hashModel tape (path ++ [34, i.val]) params i ((generatedRoot.w_0_0 ctx), ((generatedRoot.w_1_0 ctx), ((generatedRoot.w_2_0 ctx), ((generatedRoot.w_3_0 ctx), ((generatedRoot.w_4_0 ctx), ((generatedRoot.w_5_0 ctx), ((generatedRoot.w_6_0 ctx), (((generatedRoot.w_33_4 ctx) i), (((generatedRoot.w_33_0 ctx) i), (((generatedRoot.w_33_2 ctx) i), (((generatedRoot.w_33_5 ctx) i), (((generatedRoot.w_33_1 ctx) i), (((generatedRoot.w_33_3 ctx) i), ()))))))))))))) (((generatedRoot.w_34_0 ctx) i), (((generatedRoot.w_34_1 ctx) i), (((generatedRoot.w_34_2 ctx) i), (((generatedRoot.w_34_3 ctx) i), (((generatedRoot.w_34_4 ctx) i), (((generatedRoot.w_34_5 ctx) i), ())))))))
  t16 : (8) = 8
  h35_5 : (∀ i : Fin 8, parallel_generatedRoot_35 backend hashModel tape (path ++ [35, i.val]) params i (((generatedRoot.w_34_0 ctx) i), (((generatedRoot.w_34_1 ctx) i), (((generatedRoot.w_34_2 ctx) i), (((generatedRoot.w_34_3 ctx) i), (((generatedRoot.w_34_4 ctx) i), (((generatedRoot.w_34_5 ctx) i), ())))))) (((generatedRoot.w_35_0 ctx) i), (((generatedRoot.w_35_1 ctx) i), (((generatedRoot.w_35_2 ctx) i), (((generatedRoot.w_35_3 ctx) i), (((generatedRoot.w_35_4 ctx) i), (((generatedRoot.w_35_5 ctx) i), ())))))))
  h36_0 : MxxRuntime.familyGetStatic (generatedRoot.w_35_0 ctx) (0) (generatedRoot.w_36_0 ctx)
  h37_0 : MxxRuntime.familyGetStatic (generatedRoot.w_35_0 ctx) (1) (generatedRoot.w_37_0 ctx)
  h39_0 : MxxRuntime.familyGetStatic (generatedRoot.w_35_1 ctx) (0) (generatedRoot.w_39_0 ctx)
  h40_0 : MxxRuntime.familyGetStatic (generatedRoot.w_35_1 ctx) (1) (generatedRoot.w_40_0 ctx)
  h42_0 : MxxRuntime.familyGetStatic (generatedRoot.w_35_2 ctx) (0) (generatedRoot.w_42_0 ctx)
  h43_0 : MxxRuntime.familyGetStatic (generatedRoot.w_35_2 ctx) (1) (generatedRoot.w_43_0 ctx)
  h45_0 : MxxRuntime.familyGetStatic (generatedRoot.w_35_3 ctx) (0) (generatedRoot.w_45_0 ctx)
  h46_0 : MxxRuntime.familyGetStatic (generatedRoot.w_35_3 ctx) (1) (generatedRoot.w_46_0 ctx)
  h48_0 : MxxRuntime.familyGetStatic (generatedRoot.w_35_4 ctx) (0) (generatedRoot.w_48_0 ctx)
  h49_0 : MxxRuntime.familyGetStatic (generatedRoot.w_35_4 ctx) (1) (generatedRoot.w_49_0 ctx)
  h51_0 : MxxRuntime.familyGetStatic (generatedRoot.w_35_5 ctx) (0) (generatedRoot.w_51_0 ctx)
  h52_0 : MxxRuntime.familyGetStatic (generatedRoot.w_35_5 ctx) (1) (generatedRoot.w_52_0 ctx)
  t30 : 0 ≤ (6)
  h54_5 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_generatedRoot_54 backend hashModel tape (path ++ [54, i]) params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, ((generatedRoot.w_35_0 ctx), ((generatedRoot.w_35_1 ctx), ((generatedRoot.w_35_2 ctx), ((generatedRoot.w_35_3 ctx), ((generatedRoot.w_35_4 ctx), ((generatedRoot.w_35_5 ctx), ())))))))))))) next) (Int.toNat (6)) ((generatedRoot.w_38_0 ctx), ((generatedRoot.w_41_0 ctx), ((generatedRoot.w_44_0 ctx), ((generatedRoot.w_47_0 ctx), ((generatedRoot.w_50_0 ctx), ((generatedRoot.w_53_0 ctx), ())))))) ((generatedRoot.w_54_0 ctx), ((generatedRoot.w_54_1 ctx), ((generatedRoot.w_54_2 ctx), ((generatedRoot.w_54_3 ctx), ((generatedRoot.w_54_4 ctx), ((generatedRoot.w_54_5 ctx), ()))))))

theorem generatedRoot.facts_0  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_0 backend hashModel tape path params (generatedRoot.w_0_0 ctx) (generatedRoot.w_1_0 ctx) (generatedRoot.w_2_0 ctx) (generatedRoot.w_3_0 ctx) (generatedRoot.w_4_0 ctx) (generatedRoot.w_5_0 ctx) (generatedRoot.w_6_0 ctx) (generatedRoot.w_7_0 ctx) (generatedRoot.w_8_0 ctx) (generatedRoot.w_9_0 ctx) (generatedRoot.w_10_0 ctx) (generatedRoot.w_11_0 ctx) (generatedRoot.w_12_0 ctx) (generatedRoot.w_13_0 ctx) (generatedRoot.w_14_0 ctx) (generatedRoot.w_15_0 ctx) (generatedRoot.w_16_0 ctx) (generatedRoot.w_19_0 ctx) (generatedRoot.w_20_0 ctx) (generatedRoot.w_23_0 ctx) (generatedRoot.w_24_0 ctx) (generatedRoot.w_25_0 ctx) (generatedRoot.w_26_0 ctx) (generatedRoot.w_29_0 ctx) (generatedRoot.w_30_0 ctx) (generatedRoot.w_17_0 ctx) (generatedRoot.w_18_0 ctx) (generatedRoot.w_21_0 ctx) (generatedRoot.w_22_0 ctx) (generatedRoot.w_27_0 ctx) (generatedRoot.w_27_1 ctx) (generatedRoot.w_27_2 ctx) (generatedRoot.w_27_3 ctx) (generatedRoot.w_27_4 ctx) (generatedRoot.w_27_5 ctx) (generatedRoot.w_28_0 ctx) (generatedRoot.w_28_1 ctx) (generatedRoot.w_28_2 ctx) (generatedRoot.w_28_3 ctx) (generatedRoot.w_28_4 ctx) (generatedRoot.w_28_5 ctx) (generatedRoot.w_31_0 ctx) (generatedRoot.w_31_1 ctx) (generatedRoot.w_31_2 ctx) (generatedRoot.w_31_3 ctx) (generatedRoot.w_31_4 ctx) (generatedRoot.w_31_5 ctx) (generatedRoot.w_32_0 ctx) (generatedRoot.w_32_1 ctx) (generatedRoot.w_32_2 ctx) (generatedRoot.w_32_3 ctx) (generatedRoot.w_32_4 ctx) (generatedRoot.w_32_5 ctx) (generatedRoot.w_33_0 ctx) (generatedRoot.w_33_1 ctx) (generatedRoot.w_33_2 ctx) (generatedRoot.w_33_3 ctx) (generatedRoot.w_33_4 ctx) (generatedRoot.w_33_5 ctx) (generatedRoot.w_34_0 ctx) (generatedRoot.w_34_1 ctx) (generatedRoot.w_34_2 ctx) (generatedRoot.w_34_3 ctx) (generatedRoot.w_34_4 ctx) (generatedRoot.w_34_5 ctx) (generatedRoot.w_35_0 ctx) (generatedRoot.w_35_1 ctx) (generatedRoot.w_35_2 ctx) (generatedRoot.w_35_3 ctx) (generatedRoot.w_35_4 ctx) (generatedRoot.w_35_5 ctx) (generatedRoot.w_36_0 ctx) (generatedRoot.w_37_0 ctx) (generatedRoot.w_38_0 ctx) (generatedRoot.w_39_0 ctx) (generatedRoot.w_40_0 ctx) (generatedRoot.w_41_0 ctx) (generatedRoot.w_42_0 ctx) (generatedRoot.w_43_0 ctx) (generatedRoot.w_44_0 ctx) (generatedRoot.w_45_0 ctx) (generatedRoot.w_46_0 ctx) (generatedRoot.w_47_0 ctx) (generatedRoot.w_48_0 ctx) (generatedRoot.w_49_0 ctx) (generatedRoot.w_50_0 ctx) (generatedRoot.w_51_0 ctx) (generatedRoot.w_52_0 ctx) (generatedRoot.w_53_0 ctx) (generatedRoot.w_54_0 ctx) (generatedRoot.w_54_1 ctx) (generatedRoot.w_54_2 ctx) (generatedRoot.w_54_3 ctx) (generatedRoot.w_54_4 ctx) (generatedRoot.w_54_5 ctx)) :
    generatedRoot.Facts_0 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_1  (ctx : generatedRootCtx) : Prop where
  h55_0 : MxxRuntime.gadgetMatrixRuns backend 4096 25 (generatedRoot.w_55_0 ctx)
  h56_0 : MxxRuntime.centeredRebaseRuns (generatedRoot.w_55_0 ctx) (generatedRoot.w_56_0 ctx)
  h59_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_58_0 ctx) (generatedRoot.w_59_0 ctx)
  h60_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_59_0 ctx) (generatedRoot.w_60_0 ctx)
  h62_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((generatedRoot.w_5_0 ctx)) (generatedRoot.w_62_0 ctx)
  h63_0 : MxxRuntime.centeredRebaseRuns (generatedRoot.w_55_0 ctx) (generatedRoot.w_63_0 ctx)
  h66_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_65_0 ctx) (generatedRoot.w_66_0 ctx)
  h67_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_66_0 ctx) (generatedRoot.w_67_0 ctx)
  h70_0 : MxxRuntime.gadgetMatrixRuns backend 4096 25 (generatedRoot.w_70_0 ctx)
  h71_0 : MxxRuntime.centeredRebaseRuns (generatedRoot.w_70_0 ctx) (generatedRoot.w_71_0 ctx)
  h73_0 : MxxRuntime.centeredRebaseRuns (generatedRoot.w_72_0 ctx) (generatedRoot.w_73_0 ctx)
  h76_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_75_0 ctx) (generatedRoot.w_76_0 ctx)
  h77_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_76_0 ctx) (generatedRoot.w_77_0 ctx)
  h81_0 : MxxRuntime.centeredRebaseRuns (generatedRoot.w_72_0 ctx) (generatedRoot.w_81_0 ctx)
  h84_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_83_0 ctx) (generatedRoot.w_84_0 ctx)
  h85_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_84_0 ctx) (generatedRoot.w_85_0 ctx)
  h87_0 : MxxRuntime.familyGetStatic (generatedRoot.w_15_0 ctx) (1) (generatedRoot.w_87_0 ctx)
  h92_0 : MxxRuntime.familyGetStatic (generatedRoot.w_91_0 ctx) (0) (generatedRoot.w_92_0 ctx)
  h94_0 : MxxRuntime.familyGetStatic (generatedRoot.w_93_0 ctx) (0) (generatedRoot.w_94_0 ctx)
  h96_0 : MxxRuntime.familyGetStatic (generatedRoot.w_95_0 ctx) (0) (generatedRoot.w_96_0 ctx)
  h98_0 : MxxRuntime.familyGetStatic (generatedRoot.w_97_0 ctx) (0) (generatedRoot.w_98_0 ctx)
  h102_0 : MxxRuntime.familyGetStatic (generatedRoot.w_101_0 ctx) (1) (generatedRoot.w_102_0 ctx)
  h103_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((generatedRoot.w_5_0 ctx)) (generatedRoot.w_103_0 ctx)
  h104_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) ((generatedRoot.w_5_0 ctx)) (generatedRoot.w_104_0 ctx)
  h103_0_1 : MxxRuntime.concatRows (generatedRoot.w_102_0 ctx) (generatedRoot.w_103_0 ctx) (generatedRoot.w_105_concat_1 ctx)
  h105_0 : MxxRuntime.concatRows (generatedRoot.w_105_concat_1 ctx) (generatedRoot.w_104_0 ctx) (generatedRoot.w_105_0 ctx)
  h107_0 : MxxRuntime.familyGetStatic (generatedRoot.w_106_0 ctx) (1) (generatedRoot.w_107_0 ctx)
  h108_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((generatedRoot.w_5_0 ctx)) (generatedRoot.w_108_0 ctx)
  h109_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) ((generatedRoot.w_5_0 ctx)) (generatedRoot.w_109_0 ctx)
  h108_0_1 : MxxRuntime.concatRows (generatedRoot.w_107_0 ctx) (generatedRoot.w_108_0 ctx) (generatedRoot.w_110_concat_1 ctx)
  h110_0 : MxxRuntime.concatRows (generatedRoot.w_110_concat_1 ctx) (generatedRoot.w_109_0 ctx) (generatedRoot.w_110_0 ctx)
  h111_0 : MxxRuntime.concatColumns (generatedRoot.w_105_0 ctx) (generatedRoot.w_110_0 ctx) (generatedRoot.w_111_0 ctx)

theorem generatedRoot.facts_1  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_1 backend hashModel (generatedRoot.w_5_0 ctx) (generatedRoot.w_15_0 ctx) (generatedRoot.w_91_0 ctx) (generatedRoot.w_93_0 ctx) (generatedRoot.w_95_0 ctx) (generatedRoot.w_97_0 ctx) (generatedRoot.w_101_0 ctx) (generatedRoot.w_106_0 ctx) (generatedRoot.w_55_0 ctx) (generatedRoot.w_56_0 ctx) (generatedRoot.w_58_0 ctx) (generatedRoot.w_59_0 ctx) (generatedRoot.w_60_0 ctx) (generatedRoot.w_62_0 ctx) (generatedRoot.w_63_0 ctx) (generatedRoot.w_65_0 ctx) (generatedRoot.w_66_0 ctx) (generatedRoot.w_67_0 ctx) (generatedRoot.w_70_0 ctx) (generatedRoot.w_71_0 ctx) (generatedRoot.w_72_0 ctx) (generatedRoot.w_73_0 ctx) (generatedRoot.w_75_0 ctx) (generatedRoot.w_76_0 ctx) (generatedRoot.w_77_0 ctx) (generatedRoot.w_81_0 ctx) (generatedRoot.w_83_0 ctx) (generatedRoot.w_84_0 ctx) (generatedRoot.w_85_0 ctx) (generatedRoot.w_87_0 ctx) (generatedRoot.w_92_0 ctx) (generatedRoot.w_94_0 ctx) (generatedRoot.w_96_0 ctx) (generatedRoot.w_98_0 ctx) (generatedRoot.w_102_0 ctx) (generatedRoot.w_103_0 ctx) (generatedRoot.w_104_0 ctx) (generatedRoot.w_105_0 ctx) (generatedRoot.w_105_concat_1 ctx) (generatedRoot.w_107_0 ctx) (generatedRoot.w_108_0 ctx) (generatedRoot.w_109_0 ctx) (generatedRoot.w_110_0 ctx) (generatedRoot.w_110_concat_1 ctx) (generatedRoot.w_111_0 ctx)) :
    generatedRoot.Facts_1 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_2  (ctx : generatedRootCtx) : Prop where
  h112_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 115, 116, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((generatedRoot.w_5_0 ctx)) (generatedRoot.w_112_0 ctx)
  h113_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 115, 116, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) ((generatedRoot.w_5_0 ctx)) (generatedRoot.w_113_0 ctx)
  h114_0 : MxxRuntime.concatRows (generatedRoot.w_112_0 ctx) (generatedRoot.w_113_0 ctx) (generatedRoot.w_114_0 ctx)
  h115_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_114_0 ctx) (generatedRoot.w_115_0 ctx)
  h116_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_115_0 ctx) (generatedRoot.w_116_0 ctx)
  t69 : 0 ≤ 0
  t70 : 0 < 1
  t71 : 1 ≤ 3
  t72 : 0 ≤ 0
  t73 : 0 < 25
  t74 : 25 ≤ 25
  h118_0 : MxxRuntime.sliceMatrix (generatedRoot.w_117_0 ctx) 0 1 0 25 (generatedRoot.w_118_0 ctx)
  h121_0 : MxxRuntime.familyGetStatic (generatedRoot.w_120_0 ctx) (1) (generatedRoot.w_121_0 ctx)
  t77 : 0 ≤ 1
  t78 : 1 < 2
  t79 : 2 ≤ 3
  t80 : 0 ≤ 0
  t81 : 0 < 25
  t82 : 25 ≤ 25
  h124_0 : MxxRuntime.sliceMatrix (generatedRoot.w_117_0 ctx) 1 2 0 25 (generatedRoot.w_124_0 ctx)
  t84 : 0 ≤ 2
  t85 : 2 < 3
  t86 : 3 ≤ 3
  t87 : 0 ≤ 0
  t88 : 0 < 25
  t89 : 25 ≤ 25
  h125_0 : MxxRuntime.sliceMatrix (generatedRoot.w_117_0 ctx) 2 3 0 25 (generatedRoot.w_125_0 ctx)
  h124_0_1 : MxxRuntime.concatRows (generatedRoot.w_123_0 ctx) (generatedRoot.w_124_0 ctx) (generatedRoot.w_126_concat_1 ctx)
  h126_0 : MxxRuntime.concatRows (generatedRoot.w_126_concat_1 ctx) (generatedRoot.w_125_0 ctx) (generatedRoot.w_126_0 ctx)
  h127_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 115, 116, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((generatedRoot.w_5_0 ctx)) (generatedRoot.w_127_0 ctx)
  h128_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 115, 116, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) ((generatedRoot.w_5_0 ctx)) (generatedRoot.w_128_0 ctx)
  h129_0 : MxxRuntime.concatRows (generatedRoot.w_127_0 ctx) (generatedRoot.w_128_0 ctx) (generatedRoot.w_129_0 ctx)

theorem generatedRoot.facts_2  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_2 backend hashModel (generatedRoot.w_5_0 ctx) (generatedRoot.w_120_0 ctx) (generatedRoot.w_112_0 ctx) (generatedRoot.w_113_0 ctx) (generatedRoot.w_114_0 ctx) (generatedRoot.w_115_0 ctx) (generatedRoot.w_116_0 ctx) (generatedRoot.w_117_0 ctx) (generatedRoot.w_118_0 ctx) (generatedRoot.w_121_0 ctx) (generatedRoot.w_123_0 ctx) (generatedRoot.w_124_0 ctx) (generatedRoot.w_125_0 ctx) (generatedRoot.w_126_0 ctx) (generatedRoot.w_126_concat_1 ctx) (generatedRoot.w_127_0 ctx) (generatedRoot.w_128_0 ctx) (generatedRoot.w_129_0 ctx)) :
    generatedRoot.Facts_2 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_3  (ctx : generatedRootCtx) : Prop where
  h130_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_129_0 ctx) (generatedRoot.w_130_0 ctx)
  h131_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_130_0 ctx) (generatedRoot.w_131_0 ctx)
  t98 : 0 ≤ 0
  t99 : 0 < 1
  t100 : 1 ≤ 3
  t101 : 0 ≤ 0
  t102 : 0 < 25
  t103 : 25 ≤ 25
  h133_0 : MxxRuntime.sliceMatrix (generatedRoot.w_132_0 ctx) 0 1 0 25 (generatedRoot.w_133_0 ctx)
  t105 : 0 ≤ 1
  t106 : 1 < 2
  t107 : 2 ≤ 3
  t108 : 0 ≤ 0
  t109 : 0 < 25
  t110 : 25 ≤ 25
  h137_0 : MxxRuntime.sliceMatrix (generatedRoot.w_132_0 ctx) 1 2 0 25 (generatedRoot.w_137_0 ctx)
  t112 : 0 ≤ 2
  t113 : 2 < 3
  t114 : 3 ≤ 3
  t115 : 0 ≤ 0
  t116 : 0 < 25
  t117 : 25 ≤ 25
  h138_0 : MxxRuntime.sliceMatrix (generatedRoot.w_132_0 ctx) 2 3 0 25 (generatedRoot.w_138_0 ctx)
  h137_0_1 : MxxRuntime.concatRows (generatedRoot.w_136_0 ctx) (generatedRoot.w_137_0 ctx) (generatedRoot.w_139_concat_1 ctx)
  h139_0 : MxxRuntime.concatRows (generatedRoot.w_139_concat_1 ctx) (generatedRoot.w_138_0 ctx) (generatedRoot.w_139_0 ctx)
  h140_0 : MxxRuntime.concatColumns (generatedRoot.w_126_0 ctx) (generatedRoot.w_139_0 ctx) (generatedRoot.w_140_0 ctx)
  h141_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 104, 115, 47, 82, 104, 115, 73, 100, 40, 50, 41, 47, 48, 47, 49]) ([.bytes [112, 97, 114, 97, 109, 101, 116, 101, 114, 115], .decimal (mxxNat5), .bytes [112, 114, 111, 103, 114, 97, 109], .decimal (mxxNat6), .bytes [109, 101, 109, 98, 101, 114], .u64Le (0), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((generatedRoot.w_24_0 ctx)) (generatedRoot.w_141_0 ctx)
  h143_0 : MxxRuntime.familyGetStatic (generatedRoot.w_142_0 ctx) (0) (generatedRoot.w_143_0 ctx)
  h144_0 : MxxRuntime.concatRows (generatedRoot.w_141_0 ctx) (generatedRoot.w_143_0 ctx) (generatedRoot.w_144_0 ctx)
  h145_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_144_0 ctx) (generatedRoot.w_145_0 ctx)
  h146_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_145_0 ctx) (generatedRoot.w_146_0 ctx)
  t127 : 0 ≤ 0

theorem generatedRoot.facts_3  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_3 backend hashModel (generatedRoot.w_24_0 ctx) (generatedRoot.w_142_0 ctx) (generatedRoot.w_126_0 ctx) (generatedRoot.w_129_0 ctx) (generatedRoot.w_130_0 ctx) (generatedRoot.w_131_0 ctx) (generatedRoot.w_132_0 ctx) (generatedRoot.w_133_0 ctx) (generatedRoot.w_136_0 ctx) (generatedRoot.w_137_0 ctx) (generatedRoot.w_138_0 ctx) (generatedRoot.w_139_0 ctx) (generatedRoot.w_139_concat_1 ctx) (generatedRoot.w_140_0 ctx) (generatedRoot.w_141_0 ctx) (generatedRoot.w_143_0 ctx) (generatedRoot.w_144_0 ctx) (generatedRoot.w_145_0 ctx) (generatedRoot.w_146_0 ctx)) :
    generatedRoot.Facts_3 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_4  (ctx : generatedRootCtx) : Prop where
  t128 : 0 < 1
  t129 : 1 ≤ 3
  t130 : 0 ≤ 0
  t131 : 0 < 25
  t132 : 25 ≤ 25
  h148_0 : MxxRuntime.sliceMatrix (generatedRoot.w_147_0 ctx) 0 1 0 25 (generatedRoot.w_148_0 ctx)
  t134 : 0 ≤ 1
  t135 : 1 < 2
  t136 : 2 ≤ 3
  t137 : 0 ≤ 0
  t138 : 0 < 25
  t139 : 25 ≤ 25
  h149_0 : MxxRuntime.sliceMatrix (generatedRoot.w_147_0 ctx) 1 2 0 25 (generatedRoot.w_149_0 ctx)
  t141 : 0 ≤ 2
  t142 : 2 < 3
  t143 : 3 ≤ 3
  t144 : 0 ≤ 0
  t145 : 0 < 25
  t146 : 25 ≤ 25
  h150_0 : MxxRuntime.sliceMatrix (generatedRoot.w_147_0 ctx) 2 3 0 25 (generatedRoot.w_150_0 ctx)
  h151_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 104, 115, 47, 82, 104, 115, 73, 100, 40, 50, 41, 47, 48, 47, 49]) ([.bytes [112, 97, 114, 97, 109, 101, 116, 101, 114, 115], .decimal (mxxNat5), .bytes [112, 114, 111, 103, 114, 97, 109], .decimal (mxxNat6), .bytes [109, 101, 109, 98, 101, 114], .u64Le (0), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((generatedRoot.w_24_0 ctx)) (generatedRoot.w_151_0 ctx)
  h153_0 : MxxRuntime.familyGetStatic (generatedRoot.w_152_0 ctx) (0) (generatedRoot.w_153_0 ctx)
  h154_0 : MxxRuntime.concatRows (generatedRoot.w_151_0 ctx) (generatedRoot.w_153_0 ctx) (generatedRoot.w_154_0 ctx)
  h155_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_154_0 ctx) (generatedRoot.w_155_0 ctx)
  h156_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_155_0 ctx) (generatedRoot.w_156_0 ctx)
  t153 : 0 ≤ 0
  t154 : 0 < 1
  t155 : 1 ≤ 3
  t156 : 0 ≤ 0
  t157 : 0 < 25
  t158 : 25 ≤ 25
  h158_0 : MxxRuntime.sliceMatrix (generatedRoot.w_157_0 ctx) 0 1 0 25 (generatedRoot.w_158_0 ctx)

theorem generatedRoot.facts_4  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_4 backend hashModel (generatedRoot.w_24_0 ctx) (generatedRoot.w_152_0 ctx) (generatedRoot.w_147_0 ctx) (generatedRoot.w_148_0 ctx) (generatedRoot.w_149_0 ctx) (generatedRoot.w_150_0 ctx) (generatedRoot.w_151_0 ctx) (generatedRoot.w_153_0 ctx) (generatedRoot.w_154_0 ctx) (generatedRoot.w_155_0 ctx) (generatedRoot.w_156_0 ctx) (generatedRoot.w_157_0 ctx) (generatedRoot.w_158_0 ctx)) :
    generatedRoot.Facts_4 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_5  (ctx : generatedRootCtx) : Prop where
  t160 : 0 ≤ 1
  t161 : 1 < 2
  t162 : 2 ≤ 3
  t163 : 0 ≤ 0
  t164 : 0 < 25
  t165 : 25 ≤ 25
  h159_0 : MxxRuntime.sliceMatrix (generatedRoot.w_157_0 ctx) 1 2 0 25 (generatedRoot.w_159_0 ctx)
  t167 : 0 ≤ 2
  t168 : 2 < 3
  t169 : 3 ≤ 3
  t170 : 0 ≤ 0
  t171 : 0 < 25
  t172 : 25 ≤ 25
  h160_0 : MxxRuntime.sliceMatrix (generatedRoot.w_157_0 ctx) 2 3 0 25 (generatedRoot.w_160_0 ctx)
  t174 : 0 ≤ (3)
  h161_9 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_generatedRoot_161 backend hashModel tape (path ++ [161, i]) params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (current.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.2.1, ((generatedRoot.w_99_0 ctx), ((generatedRoot.w_5_0 ctx), ((generatedRoot.w_100_0 ctx), ((generatedRoot.w_148_0 ctx), ((generatedRoot.w_149_0 ctx), ((generatedRoot.w_150_0 ctx), ((generatedRoot.w_158_0 ctx), ((generatedRoot.w_159_0 ctx), ((generatedRoot.w_160_0 ctx), ((generatedRoot.w_91_0 ctx), ((generatedRoot.w_93_0 ctx), ((generatedRoot.w_95_0 ctx), ((generatedRoot.w_97_0 ctx), ()))))))))))))))))))))))) next) (Int.toNat (3)) ((generatedRoot.w_90_0 ctx), ((generatedRoot.w_90_0 ctx), ((generatedRoot.w_90_0 ctx), ((generatedRoot.w_90_0 ctx), ((generatedRoot.w_90_0 ctx), ((generatedRoot.w_90_0 ctx), ((generatedRoot.w_92_0 ctx), ((generatedRoot.w_94_0 ctx), ((generatedRoot.w_96_0 ctx), ((generatedRoot.w_98_0 ctx), ())))))))))) ((generatedRoot.w_161_0 ctx), ((generatedRoot.w_161_1 ctx), ((generatedRoot.w_161_2 ctx), ((generatedRoot.w_161_3 ctx), ((generatedRoot.w_161_4 ctx), ((generatedRoot.w_161_5 ctx), ((generatedRoot.w_161_6 ctx), ((generatedRoot.w_161_7 ctx), ((generatedRoot.w_161_8 ctx), ((generatedRoot.w_161_9 ctx), ()))))))))))
  h162_0 : MxxRuntime.familyGetStatic (generatedRoot.w_99_0 ctx) (3) (generatedRoot.w_162_0 ctx)
  t177 : 0 ≤ (3)
  t178 : (3) < 4
  h164_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_163_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((generatedRoot.w_5_0 ctx)) (generatedRoot.w_164_0 ctx)
  h165_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_163_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) ((generatedRoot.w_5_0 ctx)) (generatedRoot.w_165_0 ctx)
  h164_0_1 : MxxRuntime.concatRows (generatedRoot.w_162_0 ctx) (generatedRoot.w_164_0 ctx) (generatedRoot.w_166_concat_1 ctx)
  h166_0 : MxxRuntime.concatRows (generatedRoot.w_166_concat_1 ctx) (generatedRoot.w_165_0 ctx) (generatedRoot.w_166_0 ctx)
  h167_0 : MxxRuntime.familyGetStatic (generatedRoot.w_100_0 ctx) (3) (generatedRoot.w_167_0 ctx)
  h168_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_163_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((generatedRoot.w_5_0 ctx)) (generatedRoot.w_168_0 ctx)
  h169_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_163_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) ((generatedRoot.w_5_0 ctx)) (generatedRoot.w_169_0 ctx)
  h168_0_1 : MxxRuntime.concatRows (generatedRoot.w_167_0 ctx) (generatedRoot.w_168_0 ctx) (generatedRoot.w_170_concat_1 ctx)
  h170_0 : MxxRuntime.concatRows (generatedRoot.w_170_concat_1 ctx) (generatedRoot.w_169_0 ctx) (generatedRoot.w_170_0 ctx)
  h171_0 : MxxRuntime.concatColumns (generatedRoot.w_166_0 ctx) (generatedRoot.w_170_0 ctx) (generatedRoot.w_171_0 ctx)
  t189 : 0 ≤ (3)
  t190 : (3) < 4
  h172_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (3) [(1), (3), (5), (15)])) (generatedRoot.w_148_0 ctx) (generatedRoot.w_172_0 ctx)

theorem generatedRoot.facts_5  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_5 backend hashModel tape path params (generatedRoot.w_5_0 ctx) (generatedRoot.w_91_0 ctx) (generatedRoot.w_93_0 ctx) (generatedRoot.w_95_0 ctx) (generatedRoot.w_97_0 ctx) (generatedRoot.w_99_0 ctx) (generatedRoot.w_100_0 ctx) (generatedRoot.w_90_0 ctx) (generatedRoot.w_92_0 ctx) (generatedRoot.w_94_0 ctx) (generatedRoot.w_96_0 ctx) (generatedRoot.w_98_0 ctx) (generatedRoot.w_148_0 ctx) (generatedRoot.w_149_0 ctx) (generatedRoot.w_150_0 ctx) (generatedRoot.w_157_0 ctx) (generatedRoot.w_158_0 ctx) (generatedRoot.w_159_0 ctx) (generatedRoot.w_160_0 ctx) (generatedRoot.w_161_0 ctx) (generatedRoot.w_161_1 ctx) (generatedRoot.w_161_2 ctx) (generatedRoot.w_161_3 ctx) (generatedRoot.w_161_4 ctx) (generatedRoot.w_161_5 ctx) (generatedRoot.w_161_6 ctx) (generatedRoot.w_161_7 ctx) (generatedRoot.w_161_8 ctx) (generatedRoot.w_161_9 ctx) (generatedRoot.w_162_0 ctx) (generatedRoot.w_163_0 ctx) (generatedRoot.w_164_0 ctx) (generatedRoot.w_165_0 ctx) (generatedRoot.w_166_0 ctx) (generatedRoot.w_166_concat_1 ctx) (generatedRoot.w_167_0 ctx) (generatedRoot.w_168_0 ctx) (generatedRoot.w_169_0 ctx) (generatedRoot.w_170_0 ctx) (generatedRoot.w_170_concat_1 ctx) (generatedRoot.w_171_0 ctx) (generatedRoot.w_172_0 ctx)) :
    generatedRoot.Facts_5 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_6  (ctx : generatedRootCtx) : Prop where
  t192 : 0 ≤ (3)
  t193 : (3) < 4
  h173_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (3) [(1), (3), (5), (15)])) (generatedRoot.w_149_0 ctx) (generatedRoot.w_173_0 ctx)
  t195 : 0 ≤ (3)
  t196 : (3) < 4
  h174_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (3) [(1), (3), (5), (15)])) (generatedRoot.w_150_0 ctx) (generatedRoot.w_174_0 ctx)
  h173_0_1 : MxxRuntime.concatRows (generatedRoot.w_172_0 ctx) (generatedRoot.w_173_0 ctx) (generatedRoot.w_175_concat_1 ctx)
  h175_0 : MxxRuntime.concatRows (generatedRoot.w_175_concat_1 ctx) (generatedRoot.w_174_0 ctx) (generatedRoot.w_175_0 ctx)
  t200 : 0 ≤ (3)
  t201 : (3) < 4
  h176_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (3) [(1), (3), (5), (15)])) (generatedRoot.w_158_0 ctx) (generatedRoot.w_176_0 ctx)
  t203 : 0 ≤ (3)
  t204 : (3) < 4
  h177_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (3) [(1), (3), (5), (15)])) (generatedRoot.w_159_0 ctx) (generatedRoot.w_177_0 ctx)
  t206 : 0 ≤ (3)
  t207 : (3) < 4
  h178_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (3) [(1), (3), (5), (15)])) (generatedRoot.w_160_0 ctx) (generatedRoot.w_178_0 ctx)
  h177_0_1 : MxxRuntime.concatRows (generatedRoot.w_176_0 ctx) (generatedRoot.w_177_0 ctx) (generatedRoot.w_179_concat_1 ctx)
  h179_0 : MxxRuntime.concatRows (generatedRoot.w_179_concat_1 ctx) (generatedRoot.w_178_0 ctx) (generatedRoot.w_179_0 ctx)
  h180_0 : MxxRuntime.concatColumns (generatedRoot.w_175_0 ctx) (generatedRoot.w_179_0 ctx) (generatedRoot.w_180_0 ctx)
  h181_0 : MxxRuntime.concatRows (generatedRoot.w_161_6 ctx) (generatedRoot.w_161_8 ctx) (generatedRoot.w_181_0 ctx)
  h182_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_181_0 ctx) (generatedRoot.w_182_0 ctx)
  h183_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_182_0 ctx) (generatedRoot.w_183_0 ctx)
  t215 : 0 ≤ 1
  t216 : 1 < 2
  t217 : 2 ≤ 3
  t218 : 0 ≤ 0
  t219 : 0 < 25
  t220 : 25 ≤ 25
  h185_0 : MxxRuntime.sliceMatrix (generatedRoot.w_184_0 ctx) 1 2 0 25 (generatedRoot.w_185_0 ctx)
  t222 : 0 ≤ 2
  t223 : 2 < 3

theorem generatedRoot.facts_6  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_6 backend (generatedRoot.w_149_0 ctx) (generatedRoot.w_150_0 ctx) (generatedRoot.w_158_0 ctx) (generatedRoot.w_159_0 ctx) (generatedRoot.w_160_0 ctx) (generatedRoot.w_161_6 ctx) (generatedRoot.w_161_8 ctx) (generatedRoot.w_172_0 ctx) (generatedRoot.w_173_0 ctx) (generatedRoot.w_174_0 ctx) (generatedRoot.w_175_0 ctx) (generatedRoot.w_175_concat_1 ctx) (generatedRoot.w_176_0 ctx) (generatedRoot.w_177_0 ctx) (generatedRoot.w_178_0 ctx) (generatedRoot.w_179_0 ctx) (generatedRoot.w_179_concat_1 ctx) (generatedRoot.w_180_0 ctx) (generatedRoot.w_181_0 ctx) (generatedRoot.w_182_0 ctx) (generatedRoot.w_183_0 ctx) (generatedRoot.w_184_0 ctx) (generatedRoot.w_185_0 ctx)) :
    generatedRoot.Facts_6 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_7  (ctx : generatedRootCtx) : Prop where
  t224 : 3 ≤ 3
  t225 : 0 ≤ 0
  t226 : 0 < 25
  t227 : 25 ≤ 25
  h186_0 : MxxRuntime.sliceMatrix (generatedRoot.w_184_0 ctx) 2 3 0 25 (generatedRoot.w_186_0 ctx)
  h187_0 : MxxRuntime.concatRows (generatedRoot.w_185_0 ctx) (generatedRoot.w_186_0 ctx) (generatedRoot.w_187_0 ctx)
  h188_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_187_0 ctx) (generatedRoot.w_188_0 ctx)
  h189_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_188_0 ctx) (generatedRoot.w_189_0 ctx)
  t232 : 0 ≤ 0
  t233 : 0 < 1
  t234 : 1 ≤ 3
  t235 : 0 ≤ 0
  t236 : 0 < 25
  t237 : 25 ≤ 25
  h191_0 : MxxRuntime.sliceMatrix (generatedRoot.w_190_0 ctx) 0 1 0 25 (generatedRoot.w_191_0 ctx)
  t239 : 0 ≤ 0
  t240 : 0 < 1
  t241 : 1 ≤ 3
  t242 : 0 ≤ 0
  t243 : 0 < 25
  t244 : 25 ≤ 25
  h192_0 : MxxRuntime.sliceMatrix (generatedRoot.w_184_0 ctx) 0 1 0 25 (generatedRoot.w_192_0 ctx)
  h195_0 : MxxRuntime.centeredRebaseRuns (generatedRoot.w_72_0 ctx) (generatedRoot.w_195_0 ctx)
  h198_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_197_0 ctx) (generatedRoot.w_198_0 ctx)
  h199_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_198_0 ctx) (generatedRoot.w_199_0 ctx)
  h201_0 : MxxRuntime.concatRows (generatedRoot.w_161_7 ctx) (generatedRoot.w_161_9 ctx) (generatedRoot.w_201_0 ctx)
  h202_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_201_0 ctx) (generatedRoot.w_202_0 ctx)
  h203_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_202_0 ctx) (generatedRoot.w_203_0 ctx)
  t252 : 0 ≤ 1
  t253 : 1 < 2
  t254 : 2 ≤ 3
  t255 : 0 ≤ 0

theorem generatedRoot.facts_7  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_7 backend (generatedRoot.w_72_0 ctx) (generatedRoot.w_161_7 ctx) (generatedRoot.w_161_9 ctx) (generatedRoot.w_184_0 ctx) (generatedRoot.w_185_0 ctx) (generatedRoot.w_186_0 ctx) (generatedRoot.w_187_0 ctx) (generatedRoot.w_188_0 ctx) (generatedRoot.w_189_0 ctx) (generatedRoot.w_190_0 ctx) (generatedRoot.w_191_0 ctx) (generatedRoot.w_192_0 ctx) (generatedRoot.w_195_0 ctx) (generatedRoot.w_197_0 ctx) (generatedRoot.w_198_0 ctx) (generatedRoot.w_199_0 ctx) (generatedRoot.w_201_0 ctx) (generatedRoot.w_202_0 ctx) (generatedRoot.w_203_0 ctx)) :
    generatedRoot.Facts_7 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_8  (ctx : generatedRootCtx) : Prop where
  t256 : 0 < 25
  t257 : 25 ≤ 25
  h205_0 : MxxRuntime.sliceMatrix (generatedRoot.w_204_0 ctx) 1 2 0 25 (generatedRoot.w_205_0 ctx)
  t259 : 0 ≤ 2
  t260 : 2 < 3
  t261 : 3 ≤ 3
  t262 : 0 ≤ 0
  t263 : 0 < 25
  t264 : 25 ≤ 25
  h206_0 : MxxRuntime.sliceMatrix (generatedRoot.w_204_0 ctx) 2 3 0 25 (generatedRoot.w_206_0 ctx)
  h207_0 : MxxRuntime.concatRows (generatedRoot.w_205_0 ctx) (generatedRoot.w_206_0 ctx) (generatedRoot.w_207_0 ctx)
  h208_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_207_0 ctx) (generatedRoot.w_208_0 ctx)
  h209_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_208_0 ctx) (generatedRoot.w_209_0 ctx)
  t269 : 0 ≤ 0
  t270 : 0 < 1
  t271 : 1 ≤ 3
  t272 : 0 ≤ 0
  t273 : 0 < 25
  t274 : 25 ≤ 25
  h211_0 : MxxRuntime.sliceMatrix (generatedRoot.w_210_0 ctx) 0 1 0 25 (generatedRoot.w_211_0 ctx)
  t276 : 0 ≤ 0
  t277 : 0 < 1
  t278 : 1 ≤ 3
  t279 : 0 ≤ 0
  t280 : 0 < 25
  t281 : 25 ≤ 25
  h212_0 : MxxRuntime.sliceMatrix (generatedRoot.w_204_0 ctx) 0 1 0 25 (generatedRoot.w_212_0 ctx)
  h216_0 : MxxRuntime.centeredRebaseRuns (generatedRoot.w_55_0 ctx) (generatedRoot.w_216_0 ctx)
  h219_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_218_0 ctx) (generatedRoot.w_219_0 ctx)
  h220_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_219_0 ctx) (generatedRoot.w_220_0 ctx)
  h222_0 : MxxRuntime.centeredRebaseRuns (generatedRoot.w_72_0 ctx) (generatedRoot.w_222_0 ctx)
  h225_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_224_0 ctx) (generatedRoot.w_225_0 ctx)

theorem generatedRoot.facts_8  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_8 backend (generatedRoot.w_55_0 ctx) (generatedRoot.w_72_0 ctx) (generatedRoot.w_204_0 ctx) (generatedRoot.w_205_0 ctx) (generatedRoot.w_206_0 ctx) (generatedRoot.w_207_0 ctx) (generatedRoot.w_208_0 ctx) (generatedRoot.w_209_0 ctx) (generatedRoot.w_210_0 ctx) (generatedRoot.w_211_0 ctx) (generatedRoot.w_212_0 ctx) (generatedRoot.w_216_0 ctx) (generatedRoot.w_218_0 ctx) (generatedRoot.w_219_0 ctx) (generatedRoot.w_220_0 ctx) (generatedRoot.w_222_0 ctx) (generatedRoot.w_224_0 ctx) (generatedRoot.w_225_0 ctx)) :
    generatedRoot.Facts_8 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_9  (ctx : generatedRootCtx) : Prop where
  h226_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_225_0 ctx) (generatedRoot.w_226_0 ctx)
  h231_0 : MxxRuntime.familyGetStatic (generatedRoot.w_230_0 ctx) (0) (generatedRoot.w_231_0 ctx)
  h233_0 : MxxRuntime.familyGetStatic (generatedRoot.w_232_0 ctx) (0) (generatedRoot.w_233_0 ctx)
  h235_0 : MxxRuntime.familyGetStatic (generatedRoot.w_234_0 ctx) (0) (generatedRoot.w_235_0 ctx)
  h237_0 : MxxRuntime.familyGetStatic (generatedRoot.w_236_0 ctx) (0) (generatedRoot.w_237_0 ctx)
  t293 : 0 ≤ (0)
  h240_9 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_generatedRoot_240 backend hashModel tape (path ++ [240, i]) params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (current.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.2.1, ((generatedRoot.w_238_0 ctx), ((generatedRoot.w_5_0 ctx), ((generatedRoot.w_239_0 ctx), ((generatedRoot.w_54_4 ctx), ((generatedRoot.w_54_0 ctx), ((generatedRoot.w_54_2 ctx), ((generatedRoot.w_54_5 ctx), ((generatedRoot.w_54_1 ctx), ((generatedRoot.w_54_3 ctx), ((generatedRoot.w_230_0 ctx), ((generatedRoot.w_232_0 ctx), ((generatedRoot.w_234_0 ctx), ((generatedRoot.w_236_0 ctx), ()))))))))))))))))))))))) next) (Int.toNat (0)) ((generatedRoot.w_229_0 ctx), ((generatedRoot.w_229_0 ctx), ((generatedRoot.w_229_0 ctx), ((generatedRoot.w_229_0 ctx), ((generatedRoot.w_229_0 ctx), ((generatedRoot.w_229_0 ctx), ((generatedRoot.w_231_0 ctx), ((generatedRoot.w_233_0 ctx), ((generatedRoot.w_235_0 ctx), ((generatedRoot.w_237_0 ctx), ())))))))))) ((generatedRoot.w_240_0 ctx), ((generatedRoot.w_240_1 ctx), ((generatedRoot.w_240_2 ctx), ((generatedRoot.w_240_3 ctx), ((generatedRoot.w_240_4 ctx), ((generatedRoot.w_240_5 ctx), ((generatedRoot.w_240_6 ctx), ((generatedRoot.w_240_7 ctx), ((generatedRoot.w_240_8 ctx), ((generatedRoot.w_240_9 ctx), ()))))))))))
  h241_0 : MxxRuntime.familyGetStatic (generatedRoot.w_238_0 ctx) (0) (generatedRoot.w_241_0 ctx)
  t296 : 0 ≤ (0)
  t297 : (0) < 1
  h243_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_242_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((generatedRoot.w_5_0 ctx)) (generatedRoot.w_243_0 ctx)
  h244_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_242_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) ((generatedRoot.w_5_0 ctx)) (generatedRoot.w_244_0 ctx)
  h243_0_1 : MxxRuntime.concatRows (generatedRoot.w_241_0 ctx) (generatedRoot.w_243_0 ctx) (generatedRoot.w_245_concat_1 ctx)
  h245_0 : MxxRuntime.concatRows (generatedRoot.w_245_concat_1 ctx) (generatedRoot.w_244_0 ctx) (generatedRoot.w_245_0 ctx)
  h246_0 : MxxRuntime.familyGetStatic (generatedRoot.w_239_0 ctx) (0) (generatedRoot.w_246_0 ctx)
  h247_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_242_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((generatedRoot.w_5_0 ctx)) (generatedRoot.w_247_0 ctx)
  h248_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_242_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) ((generatedRoot.w_5_0 ctx)) (generatedRoot.w_248_0 ctx)
  h247_0_1 : MxxRuntime.concatRows (generatedRoot.w_246_0 ctx) (generatedRoot.w_247_0 ctx) (generatedRoot.w_249_concat_1 ctx)
  h249_0 : MxxRuntime.concatRows (generatedRoot.w_249_concat_1 ctx) (generatedRoot.w_248_0 ctx) (generatedRoot.w_249_0 ctx)
  h250_0 : MxxRuntime.concatColumns (generatedRoot.w_245_0 ctx) (generatedRoot.w_249_0 ctx) (generatedRoot.w_250_0 ctx)
  t308 : 0 ≤ (0)
  t309 : (0) < 1
  h251_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(3)])) (generatedRoot.w_54_4 ctx) (generatedRoot.w_251_0 ctx)
  t311 : 0 ≤ (0)
  t312 : (0) < 1
  h252_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(3)])) (generatedRoot.w_54_0 ctx) (generatedRoot.w_252_0 ctx)
  t314 : 0 ≤ (0)
  t315 : (0) < 1
  h253_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(3)])) (generatedRoot.w_54_2 ctx) (generatedRoot.w_253_0 ctx)
  h252_0_1 : MxxRuntime.concatRows (generatedRoot.w_251_0 ctx) (generatedRoot.w_252_0 ctx) (generatedRoot.w_254_concat_1 ctx)
  h254_0 : MxxRuntime.concatRows (generatedRoot.w_254_concat_1 ctx) (generatedRoot.w_253_0 ctx) (generatedRoot.w_254_0 ctx)
  t319 : 0 ≤ (0)

theorem generatedRoot.facts_9  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_9 backend hashModel tape path params (generatedRoot.w_5_0 ctx) (generatedRoot.w_230_0 ctx) (generatedRoot.w_232_0 ctx) (generatedRoot.w_234_0 ctx) (generatedRoot.w_236_0 ctx) (generatedRoot.w_238_0 ctx) (generatedRoot.w_239_0 ctx) (generatedRoot.w_54_0 ctx) (generatedRoot.w_54_1 ctx) (generatedRoot.w_54_2 ctx) (generatedRoot.w_54_3 ctx) (generatedRoot.w_54_4 ctx) (generatedRoot.w_54_5 ctx) (generatedRoot.w_225_0 ctx) (generatedRoot.w_226_0 ctx) (generatedRoot.w_229_0 ctx) (generatedRoot.w_231_0 ctx) (generatedRoot.w_233_0 ctx) (generatedRoot.w_235_0 ctx) (generatedRoot.w_237_0 ctx) (generatedRoot.w_240_0 ctx) (generatedRoot.w_240_1 ctx) (generatedRoot.w_240_2 ctx) (generatedRoot.w_240_3 ctx) (generatedRoot.w_240_4 ctx) (generatedRoot.w_240_5 ctx) (generatedRoot.w_240_6 ctx) (generatedRoot.w_240_7 ctx) (generatedRoot.w_240_8 ctx) (generatedRoot.w_240_9 ctx) (generatedRoot.w_241_0 ctx) (generatedRoot.w_242_0 ctx) (generatedRoot.w_243_0 ctx) (generatedRoot.w_244_0 ctx) (generatedRoot.w_245_0 ctx) (generatedRoot.w_245_concat_1 ctx) (generatedRoot.w_246_0 ctx) (generatedRoot.w_247_0 ctx) (generatedRoot.w_248_0 ctx) (generatedRoot.w_249_0 ctx) (generatedRoot.w_249_concat_1 ctx) (generatedRoot.w_250_0 ctx) (generatedRoot.w_251_0 ctx) (generatedRoot.w_252_0 ctx) (generatedRoot.w_253_0 ctx) (generatedRoot.w_254_0 ctx) (generatedRoot.w_254_concat_1 ctx)) :
    generatedRoot.Facts_9 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_10  (ctx : generatedRootCtx) : Prop where
  t320 : (0) < 1
  h255_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(3)])) (generatedRoot.w_54_5 ctx) (generatedRoot.w_255_0 ctx)
  t322 : 0 ≤ (0)
  t323 : (0) < 1
  h256_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(3)])) (generatedRoot.w_54_1 ctx) (generatedRoot.w_256_0 ctx)
  t325 : 0 ≤ (0)
  t326 : (0) < 1
  h257_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(3)])) (generatedRoot.w_54_3 ctx) (generatedRoot.w_257_0 ctx)
  h256_0_1 : MxxRuntime.concatRows (generatedRoot.w_255_0 ctx) (generatedRoot.w_256_0 ctx) (generatedRoot.w_258_concat_1 ctx)
  h258_0 : MxxRuntime.concatRows (generatedRoot.w_258_concat_1 ctx) (generatedRoot.w_257_0 ctx) (generatedRoot.w_258_0 ctx)
  h259_0 : MxxRuntime.concatColumns (generatedRoot.w_254_0 ctx) (generatedRoot.w_258_0 ctx) (generatedRoot.w_259_0 ctx)
  h260_0 : MxxRuntime.concatRows (generatedRoot.w_240_7 ctx) (generatedRoot.w_240_9 ctx) (generatedRoot.w_260_0 ctx)
  h261_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_260_0 ctx) (generatedRoot.w_261_0 ctx)
  h262_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_261_0 ctx) (generatedRoot.w_262_0 ctx)
  t334 : 0 ≤ 1
  t335 : 1 < 2
  t336 : 2 ≤ 3
  t337 : 0 ≤ 0
  t338 : 0 < 25
  t339 : 25 ≤ 25
  h264_0 : MxxRuntime.sliceMatrix (generatedRoot.w_263_0 ctx) 1 2 0 25 (generatedRoot.w_264_0 ctx)
  t341 : 0 ≤ 2
  t342 : 2 < 3
  t343 : 3 ≤ 3
  t344 : 0 ≤ 0
  t345 : 0 < 25
  t346 : 25 ≤ 25
  h265_0 : MxxRuntime.sliceMatrix (generatedRoot.w_263_0 ctx) 2 3 0 25 (generatedRoot.w_265_0 ctx)
  h266_0 : MxxRuntime.concatRows (generatedRoot.w_264_0 ctx) (generatedRoot.w_265_0 ctx) (generatedRoot.w_266_0 ctx)
  h267_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_266_0 ctx) (generatedRoot.w_267_0 ctx)
  h268_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_267_0 ctx) (generatedRoot.w_268_0 ctx)
  t351 : 0 ≤ 0

theorem generatedRoot.facts_10  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_10 backend (generatedRoot.w_54_1 ctx) (generatedRoot.w_54_3 ctx) (generatedRoot.w_54_5 ctx) (generatedRoot.w_240_7 ctx) (generatedRoot.w_240_9 ctx) (generatedRoot.w_254_0 ctx) (generatedRoot.w_255_0 ctx) (generatedRoot.w_256_0 ctx) (generatedRoot.w_257_0 ctx) (generatedRoot.w_258_0 ctx) (generatedRoot.w_258_concat_1 ctx) (generatedRoot.w_259_0 ctx) (generatedRoot.w_260_0 ctx) (generatedRoot.w_261_0 ctx) (generatedRoot.w_262_0 ctx) (generatedRoot.w_263_0 ctx) (generatedRoot.w_264_0 ctx) (generatedRoot.w_265_0 ctx) (generatedRoot.w_266_0 ctx) (generatedRoot.w_267_0 ctx) (generatedRoot.w_268_0 ctx)) :
    generatedRoot.Facts_10 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_11  (ctx : generatedRootCtx) : Prop where
  t352 : 0 < 1
  t353 : 1 ≤ 3
  t354 : 0 ≤ 0
  t355 : 0 < 25
  t356 : 25 ≤ 25
  h270_0 : MxxRuntime.sliceMatrix (generatedRoot.w_269_0 ctx) 0 1 0 25 (generatedRoot.w_270_0 ctx)
  t358 : 0 ≤ 0
  t359 : 0 < 1
  t360 : 1 ≤ 3
  t361 : 0 ≤ 0
  t362 : 0 < 25
  t363 : 25 ≤ 25
  h271_0 : MxxRuntime.sliceMatrix (generatedRoot.w_263_0 ctx) 0 1 0 25 (generatedRoot.w_271_0 ctx)
  h274_0 : MxxRuntime.centeredRebaseRuns (generatedRoot.w_55_0 ctx) (generatedRoot.w_274_0 ctx)
  h277_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_276_0 ctx) (generatedRoot.w_277_0 ctx)
  h278_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_277_0 ctx) (generatedRoot.w_278_0 ctx)
  h280_0 : MxxRuntime.centeredRebaseRuns (generatedRoot.w_72_0 ctx) (generatedRoot.w_280_0 ctx)
  h283_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_282_0 ctx) (generatedRoot.w_283_0 ctx)
  h284_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_283_0 ctx) (generatedRoot.w_284_0 ctx)
  h289_0 : MxxRuntime.familyGetStatic (generatedRoot.w_288_0 ctx) (0) (generatedRoot.w_289_0 ctx)
  h291_0 : MxxRuntime.familyGetStatic (generatedRoot.w_290_0 ctx) (0) (generatedRoot.w_291_0 ctx)
  h293_0 : MxxRuntime.familyGetStatic (generatedRoot.w_292_0 ctx) (0) (generatedRoot.w_293_0 ctx)
  h295_0 : MxxRuntime.familyGetStatic (generatedRoot.w_294_0 ctx) (0) (generatedRoot.w_295_0 ctx)
  t375 : 0 ≤ (0)
  h298_9 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_generatedRoot_298 backend hashModel tape (path ++ [298, i]) params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (current.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.2.1, ((generatedRoot.w_296_0 ctx), ((generatedRoot.w_5_0 ctx), ((generatedRoot.w_297_0 ctx), ((generatedRoot.w_54_4 ctx), ((generatedRoot.w_54_0 ctx), ((generatedRoot.w_54_2 ctx), ((generatedRoot.w_54_5 ctx), ((generatedRoot.w_54_1 ctx), ((generatedRoot.w_54_3 ctx), ((generatedRoot.w_288_0 ctx), ((generatedRoot.w_290_0 ctx), ((generatedRoot.w_292_0 ctx), ((generatedRoot.w_294_0 ctx), ()))))))))))))))))))))))) next) (Int.toNat (0)) ((generatedRoot.w_287_0 ctx), ((generatedRoot.w_287_0 ctx), ((generatedRoot.w_287_0 ctx), ((generatedRoot.w_287_0 ctx), ((generatedRoot.w_287_0 ctx), ((generatedRoot.w_287_0 ctx), ((generatedRoot.w_289_0 ctx), ((generatedRoot.w_291_0 ctx), ((generatedRoot.w_293_0 ctx), ((generatedRoot.w_295_0 ctx), ())))))))))) ((generatedRoot.w_298_0 ctx), ((generatedRoot.w_298_1 ctx), ((generatedRoot.w_298_2 ctx), ((generatedRoot.w_298_3 ctx), ((generatedRoot.w_298_4 ctx), ((generatedRoot.w_298_5 ctx), ((generatedRoot.w_298_6 ctx), ((generatedRoot.w_298_7 ctx), ((generatedRoot.w_298_8 ctx), ((generatedRoot.w_298_9 ctx), ()))))))))))
  h299_0 : MxxRuntime.familyGetStatic (generatedRoot.w_296_0 ctx) (0) (generatedRoot.w_299_0 ctx)
  t378 : 0 ≤ (0)
  t379 : (0) < 1
  h301_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_300_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((generatedRoot.w_5_0 ctx)) (generatedRoot.w_301_0 ctx)
  h302_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_300_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) ((generatedRoot.w_5_0 ctx)) (generatedRoot.w_302_0 ctx)
  h301_0_1 : MxxRuntime.concatRows (generatedRoot.w_299_0 ctx) (generatedRoot.w_301_0 ctx) (generatedRoot.w_303_concat_1 ctx)
  h303_0 : MxxRuntime.concatRows (generatedRoot.w_303_concat_1 ctx) (generatedRoot.w_302_0 ctx) (generatedRoot.w_303_0 ctx)

theorem generatedRoot.facts_11  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_11 backend hashModel tape path params (generatedRoot.w_5_0 ctx) (generatedRoot.w_288_0 ctx) (generatedRoot.w_290_0 ctx) (generatedRoot.w_292_0 ctx) (generatedRoot.w_294_0 ctx) (generatedRoot.w_296_0 ctx) (generatedRoot.w_297_0 ctx) (generatedRoot.w_54_0 ctx) (generatedRoot.w_54_1 ctx) (generatedRoot.w_54_2 ctx) (generatedRoot.w_54_3 ctx) (generatedRoot.w_54_4 ctx) (generatedRoot.w_54_5 ctx) (generatedRoot.w_55_0 ctx) (generatedRoot.w_72_0 ctx) (generatedRoot.w_263_0 ctx) (generatedRoot.w_269_0 ctx) (generatedRoot.w_270_0 ctx) (generatedRoot.w_271_0 ctx) (generatedRoot.w_274_0 ctx) (generatedRoot.w_276_0 ctx) (generatedRoot.w_277_0 ctx) (generatedRoot.w_278_0 ctx) (generatedRoot.w_280_0 ctx) (generatedRoot.w_282_0 ctx) (generatedRoot.w_283_0 ctx) (generatedRoot.w_284_0 ctx) (generatedRoot.w_287_0 ctx) (generatedRoot.w_289_0 ctx) (generatedRoot.w_291_0 ctx) (generatedRoot.w_293_0 ctx) (generatedRoot.w_295_0 ctx) (generatedRoot.w_298_0 ctx) (generatedRoot.w_298_1 ctx) (generatedRoot.w_298_2 ctx) (generatedRoot.w_298_3 ctx) (generatedRoot.w_298_4 ctx) (generatedRoot.w_298_5 ctx) (generatedRoot.w_298_6 ctx) (generatedRoot.w_298_7 ctx) (generatedRoot.w_298_8 ctx) (generatedRoot.w_298_9 ctx) (generatedRoot.w_299_0 ctx) (generatedRoot.w_300_0 ctx) (generatedRoot.w_301_0 ctx) (generatedRoot.w_302_0 ctx) (generatedRoot.w_303_0 ctx) (generatedRoot.w_303_concat_1 ctx)) :
    generatedRoot.Facts_11 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_12  (ctx : generatedRootCtx) : Prop where
  h304_0 : MxxRuntime.familyGetStatic (generatedRoot.w_297_0 ctx) (0) (generatedRoot.w_304_0 ctx)
  h305_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_300_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((generatedRoot.w_5_0 ctx)) (generatedRoot.w_305_0 ctx)
  h306_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_300_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) ((generatedRoot.w_5_0 ctx)) (generatedRoot.w_306_0 ctx)
  h305_0_1 : MxxRuntime.concatRows (generatedRoot.w_304_0 ctx) (generatedRoot.w_305_0 ctx) (generatedRoot.w_307_concat_1 ctx)
  h307_0 : MxxRuntime.concatRows (generatedRoot.w_307_concat_1 ctx) (generatedRoot.w_306_0 ctx) (generatedRoot.w_307_0 ctx)
  h308_0 : MxxRuntime.concatColumns (generatedRoot.w_303_0 ctx) (generatedRoot.w_307_0 ctx) (generatedRoot.w_308_0 ctx)
  t390 : 0 ≤ (0)
  t391 : (0) < 1
  h309_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(5)])) (generatedRoot.w_54_4 ctx) (generatedRoot.w_309_0 ctx)
  t393 : 0 ≤ (0)
  t394 : (0) < 1
  h310_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(5)])) (generatedRoot.w_54_0 ctx) (generatedRoot.w_310_0 ctx)
  t396 : 0 ≤ (0)
  t397 : (0) < 1
  h311_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(5)])) (generatedRoot.w_54_2 ctx) (generatedRoot.w_311_0 ctx)
  h310_0_1 : MxxRuntime.concatRows (generatedRoot.w_309_0 ctx) (generatedRoot.w_310_0 ctx) (generatedRoot.w_312_concat_1 ctx)
  h312_0 : MxxRuntime.concatRows (generatedRoot.w_312_concat_1 ctx) (generatedRoot.w_311_0 ctx) (generatedRoot.w_312_0 ctx)
  t401 : 0 ≤ (0)
  t402 : (0) < 1
  h313_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(5)])) (generatedRoot.w_54_5 ctx) (generatedRoot.w_313_0 ctx)
  t404 : 0 ≤ (0)
  t405 : (0) < 1
  h314_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(5)])) (generatedRoot.w_54_1 ctx) (generatedRoot.w_314_0 ctx)
  t407 : 0 ≤ (0)
  t408 : (0) < 1
  h315_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(5)])) (generatedRoot.w_54_3 ctx) (generatedRoot.w_315_0 ctx)
  h314_0_1 : MxxRuntime.concatRows (generatedRoot.w_313_0 ctx) (generatedRoot.w_314_0 ctx) (generatedRoot.w_316_concat_1 ctx)
  h316_0 : MxxRuntime.concatRows (generatedRoot.w_316_concat_1 ctx) (generatedRoot.w_315_0 ctx) (generatedRoot.w_316_0 ctx)
  h317_0 : MxxRuntime.concatColumns (generatedRoot.w_312_0 ctx) (generatedRoot.w_316_0 ctx) (generatedRoot.w_317_0 ctx)
  h318_0 : MxxRuntime.concatRows (generatedRoot.w_298_7 ctx) (generatedRoot.w_298_9 ctx) (generatedRoot.w_318_0 ctx)
  h319_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_318_0 ctx) (generatedRoot.w_319_0 ctx)
  h320_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_319_0 ctx) (generatedRoot.w_320_0 ctx)

theorem generatedRoot.facts_12  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_12 backend hashModel (generatedRoot.w_5_0 ctx) (generatedRoot.w_297_0 ctx) (generatedRoot.w_54_0 ctx) (generatedRoot.w_54_1 ctx) (generatedRoot.w_54_2 ctx) (generatedRoot.w_54_3 ctx) (generatedRoot.w_54_4 ctx) (generatedRoot.w_54_5 ctx) (generatedRoot.w_298_7 ctx) (generatedRoot.w_298_9 ctx) (generatedRoot.w_300_0 ctx) (generatedRoot.w_303_0 ctx) (generatedRoot.w_304_0 ctx) (generatedRoot.w_305_0 ctx) (generatedRoot.w_306_0 ctx) (generatedRoot.w_307_0 ctx) (generatedRoot.w_307_concat_1 ctx) (generatedRoot.w_308_0 ctx) (generatedRoot.w_309_0 ctx) (generatedRoot.w_310_0 ctx) (generatedRoot.w_311_0 ctx) (generatedRoot.w_312_0 ctx) (generatedRoot.w_312_concat_1 ctx) (generatedRoot.w_313_0 ctx) (generatedRoot.w_314_0 ctx) (generatedRoot.w_315_0 ctx) (generatedRoot.w_316_0 ctx) (generatedRoot.w_316_concat_1 ctx) (generatedRoot.w_317_0 ctx) (generatedRoot.w_318_0 ctx) (generatedRoot.w_319_0 ctx) (generatedRoot.w_320_0 ctx)) :
    generatedRoot.Facts_12 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_13  (ctx : generatedRootCtx) : Prop where
  t416 : 0 ≤ 1
  t417 : 1 < 2
  t418 : 2 ≤ 3
  t419 : 0 ≤ 0
  t420 : 0 < 25
  t421 : 25 ≤ 25
  h322_0 : MxxRuntime.sliceMatrix (generatedRoot.w_321_0 ctx) 1 2 0 25 (generatedRoot.w_322_0 ctx)
  t423 : 0 ≤ 2
  t424 : 2 < 3
  t425 : 3 ≤ 3
  t426 : 0 ≤ 0
  t427 : 0 < 25
  t428 : 25 ≤ 25
  h323_0 : MxxRuntime.sliceMatrix (generatedRoot.w_321_0 ctx) 2 3 0 25 (generatedRoot.w_323_0 ctx)
  h324_0 : MxxRuntime.concatRows (generatedRoot.w_322_0 ctx) (generatedRoot.w_323_0 ctx) (generatedRoot.w_324_0 ctx)
  h325_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_324_0 ctx) (generatedRoot.w_325_0 ctx)
  h326_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_325_0 ctx) (generatedRoot.w_326_0 ctx)
  t433 : 0 ≤ 0
  t434 : 0 < 1
  t435 : 1 ≤ 3
  t436 : 0 ≤ 0
  t437 : 0 < 25
  t438 : 25 ≤ 25
  h328_0 : MxxRuntime.sliceMatrix (generatedRoot.w_327_0 ctx) 0 1 0 25 (generatedRoot.w_328_0 ctx)
  t440 : 0 ≤ 0
  t441 : 0 < 1
  t442 : 1 ≤ 3
  t443 : 0 ≤ 0
  t444 : 0 < 25
  t445 : 25 ≤ 25
  h329_0 : MxxRuntime.sliceMatrix (generatedRoot.w_321_0 ctx) 0 1 0 25 (generatedRoot.w_329_0 ctx)
  h332_0 : MxxRuntime.centeredRebaseRuns (generatedRoot.w_55_0 ctx) (generatedRoot.w_332_0 ctx)

theorem generatedRoot.facts_13  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_13 backend (generatedRoot.w_55_0 ctx) (generatedRoot.w_321_0 ctx) (generatedRoot.w_322_0 ctx) (generatedRoot.w_323_0 ctx) (generatedRoot.w_324_0 ctx) (generatedRoot.w_325_0 ctx) (generatedRoot.w_326_0 ctx) (generatedRoot.w_327_0 ctx) (generatedRoot.w_328_0 ctx) (generatedRoot.w_329_0 ctx) (generatedRoot.w_332_0 ctx)) :
    generatedRoot.Facts_13 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_14  (ctx : generatedRootCtx) : Prop where
  h335_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_334_0 ctx) (generatedRoot.w_335_0 ctx)
  h336_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_335_0 ctx) (generatedRoot.w_336_0 ctx)
  h338_0 : MxxRuntime.centeredRebaseRuns (generatedRoot.w_72_0 ctx) (generatedRoot.w_338_0 ctx)
  h341_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_340_0 ctx) (generatedRoot.w_341_0 ctx)
  h342_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_341_0 ctx) (generatedRoot.w_342_0 ctx)
  h347_0 : MxxRuntime.familyGetStatic (generatedRoot.w_346_0 ctx) (0) (generatedRoot.w_347_0 ctx)
  h349_0 : MxxRuntime.familyGetStatic (generatedRoot.w_348_0 ctx) (0) (generatedRoot.w_349_0 ctx)
  h351_0 : MxxRuntime.familyGetStatic (generatedRoot.w_350_0 ctx) (0) (generatedRoot.w_351_0 ctx)
  h353_0 : MxxRuntime.familyGetStatic (generatedRoot.w_352_0 ctx) (0) (generatedRoot.w_353_0 ctx)
  t457 : 0 ≤ (0)
  h356_9 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_generatedRoot_356 backend hashModel tape (path ++ [356, i]) params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (current.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.2.1, ((generatedRoot.w_354_0 ctx), ((generatedRoot.w_5_0 ctx), ((generatedRoot.w_355_0 ctx), ((generatedRoot.w_54_4 ctx), ((generatedRoot.w_54_0 ctx), ((generatedRoot.w_54_2 ctx), ((generatedRoot.w_54_5 ctx), ((generatedRoot.w_54_1 ctx), ((generatedRoot.w_54_3 ctx), ((generatedRoot.w_346_0 ctx), ((generatedRoot.w_348_0 ctx), ((generatedRoot.w_350_0 ctx), ((generatedRoot.w_352_0 ctx), ()))))))))))))))))))))))) next) (Int.toNat (0)) ((generatedRoot.w_345_0 ctx), ((generatedRoot.w_345_0 ctx), ((generatedRoot.w_345_0 ctx), ((generatedRoot.w_345_0 ctx), ((generatedRoot.w_345_0 ctx), ((generatedRoot.w_345_0 ctx), ((generatedRoot.w_347_0 ctx), ((generatedRoot.w_349_0 ctx), ((generatedRoot.w_351_0 ctx), ((generatedRoot.w_353_0 ctx), ())))))))))) ((generatedRoot.w_356_0 ctx), ((generatedRoot.w_356_1 ctx), ((generatedRoot.w_356_2 ctx), ((generatedRoot.w_356_3 ctx), ((generatedRoot.w_356_4 ctx), ((generatedRoot.w_356_5 ctx), ((generatedRoot.w_356_6 ctx), ((generatedRoot.w_356_7 ctx), ((generatedRoot.w_356_8 ctx), ((generatedRoot.w_356_9 ctx), ()))))))))))
  h357_0 : MxxRuntime.familyGetStatic (generatedRoot.w_354_0 ctx) (0) (generatedRoot.w_357_0 ctx)
  t460 : 0 ≤ (0)
  t461 : (0) < 1
  h359_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_358_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((generatedRoot.w_5_0 ctx)) (generatedRoot.w_359_0 ctx)
  h360_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_358_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) ((generatedRoot.w_5_0 ctx)) (generatedRoot.w_360_0 ctx)
  h359_0_1 : MxxRuntime.concatRows (generatedRoot.w_357_0 ctx) (generatedRoot.w_359_0 ctx) (generatedRoot.w_361_concat_1 ctx)
  h361_0 : MxxRuntime.concatRows (generatedRoot.w_361_concat_1 ctx) (generatedRoot.w_360_0 ctx) (generatedRoot.w_361_0 ctx)
  h362_0 : MxxRuntime.familyGetStatic (generatedRoot.w_355_0 ctx) (0) (generatedRoot.w_362_0 ctx)
  h363_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_358_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((generatedRoot.w_5_0 ctx)) (generatedRoot.w_363_0 ctx)
  h364_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_358_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) ((generatedRoot.w_5_0 ctx)) (generatedRoot.w_364_0 ctx)
  h363_0_1 : MxxRuntime.concatRows (generatedRoot.w_362_0 ctx) (generatedRoot.w_363_0 ctx) (generatedRoot.w_365_concat_1 ctx)
  h365_0 : MxxRuntime.concatRows (generatedRoot.w_365_concat_1 ctx) (generatedRoot.w_364_0 ctx) (generatedRoot.w_365_0 ctx)
  h366_0 : MxxRuntime.concatColumns (generatedRoot.w_361_0 ctx) (generatedRoot.w_365_0 ctx) (generatedRoot.w_366_0 ctx)
  t472 : 0 ≤ (0)
  t473 : (0) < 1
  h367_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(7)])) (generatedRoot.w_54_4 ctx) (generatedRoot.w_367_0 ctx)
  t475 : 0 ≤ (0)
  t476 : (0) < 1
  h368_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(7)])) (generatedRoot.w_54_0 ctx) (generatedRoot.w_368_0 ctx)
  t478 : 0 ≤ (0)
  t479 : (0) < 1

theorem generatedRoot.facts_14  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_14 backend hashModel tape path params (generatedRoot.w_5_0 ctx) (generatedRoot.w_346_0 ctx) (generatedRoot.w_348_0 ctx) (generatedRoot.w_350_0 ctx) (generatedRoot.w_352_0 ctx) (generatedRoot.w_354_0 ctx) (generatedRoot.w_355_0 ctx) (generatedRoot.w_54_0 ctx) (generatedRoot.w_54_1 ctx) (generatedRoot.w_54_2 ctx) (generatedRoot.w_54_3 ctx) (generatedRoot.w_54_4 ctx) (generatedRoot.w_54_5 ctx) (generatedRoot.w_72_0 ctx) (generatedRoot.w_334_0 ctx) (generatedRoot.w_335_0 ctx) (generatedRoot.w_336_0 ctx) (generatedRoot.w_338_0 ctx) (generatedRoot.w_340_0 ctx) (generatedRoot.w_341_0 ctx) (generatedRoot.w_342_0 ctx) (generatedRoot.w_345_0 ctx) (generatedRoot.w_347_0 ctx) (generatedRoot.w_349_0 ctx) (generatedRoot.w_351_0 ctx) (generatedRoot.w_353_0 ctx) (generatedRoot.w_356_0 ctx) (generatedRoot.w_356_1 ctx) (generatedRoot.w_356_2 ctx) (generatedRoot.w_356_3 ctx) (generatedRoot.w_356_4 ctx) (generatedRoot.w_356_5 ctx) (generatedRoot.w_356_6 ctx) (generatedRoot.w_356_7 ctx) (generatedRoot.w_356_8 ctx) (generatedRoot.w_356_9 ctx) (generatedRoot.w_357_0 ctx) (generatedRoot.w_358_0 ctx) (generatedRoot.w_359_0 ctx) (generatedRoot.w_360_0 ctx) (generatedRoot.w_361_0 ctx) (generatedRoot.w_361_concat_1 ctx) (generatedRoot.w_362_0 ctx) (generatedRoot.w_363_0 ctx) (generatedRoot.w_364_0 ctx) (generatedRoot.w_365_0 ctx) (generatedRoot.w_365_concat_1 ctx) (generatedRoot.w_366_0 ctx) (generatedRoot.w_367_0 ctx) (generatedRoot.w_368_0 ctx)) :
    generatedRoot.Facts_14 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_15  (ctx : generatedRootCtx) : Prop where
  h369_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(7)])) (generatedRoot.w_54_2 ctx) (generatedRoot.w_369_0 ctx)
  h368_0_1 : MxxRuntime.concatRows (generatedRoot.w_367_0 ctx) (generatedRoot.w_368_0 ctx) (generatedRoot.w_370_concat_1 ctx)
  h370_0 : MxxRuntime.concatRows (generatedRoot.w_370_concat_1 ctx) (generatedRoot.w_369_0 ctx) (generatedRoot.w_370_0 ctx)
  t483 : 0 ≤ (0)
  t484 : (0) < 1
  h371_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(7)])) (generatedRoot.w_54_5 ctx) (generatedRoot.w_371_0 ctx)
  t486 : 0 ≤ (0)
  t487 : (0) < 1
  h372_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(7)])) (generatedRoot.w_54_1 ctx) (generatedRoot.w_372_0 ctx)
  t489 : 0 ≤ (0)
  t490 : (0) < 1
  h373_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(7)])) (generatedRoot.w_54_3 ctx) (generatedRoot.w_373_0 ctx)
  h372_0_1 : MxxRuntime.concatRows (generatedRoot.w_371_0 ctx) (generatedRoot.w_372_0 ctx) (generatedRoot.w_374_concat_1 ctx)
  h374_0 : MxxRuntime.concatRows (generatedRoot.w_374_concat_1 ctx) (generatedRoot.w_373_0 ctx) (generatedRoot.w_374_0 ctx)
  h375_0 : MxxRuntime.concatColumns (generatedRoot.w_370_0 ctx) (generatedRoot.w_374_0 ctx) (generatedRoot.w_375_0 ctx)
  h376_0 : MxxRuntime.concatRows (generatedRoot.w_356_7 ctx) (generatedRoot.w_356_9 ctx) (generatedRoot.w_376_0 ctx)
  h377_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_376_0 ctx) (generatedRoot.w_377_0 ctx)
  h378_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_377_0 ctx) (generatedRoot.w_378_0 ctx)
  t498 : 0 ≤ 1
  t499 : 1 < 2
  t500 : 2 ≤ 3
  t501 : 0 ≤ 0
  t502 : 0 < 25
  t503 : 25 ≤ 25
  h380_0 : MxxRuntime.sliceMatrix (generatedRoot.w_379_0 ctx) 1 2 0 25 (generatedRoot.w_380_0 ctx)
  t505 : 0 ≤ 2
  t506 : 2 < 3
  t507 : 3 ≤ 3
  t508 : 0 ≤ 0
  t509 : 0 < 25
  t510 : 25 ≤ 25
  h381_0 : MxxRuntime.sliceMatrix (generatedRoot.w_379_0 ctx) 2 3 0 25 (generatedRoot.w_381_0 ctx)

theorem generatedRoot.facts_15  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_15 backend (generatedRoot.w_54_1 ctx) (generatedRoot.w_54_2 ctx) (generatedRoot.w_54_3 ctx) (generatedRoot.w_54_5 ctx) (generatedRoot.w_356_7 ctx) (generatedRoot.w_356_9 ctx) (generatedRoot.w_367_0 ctx) (generatedRoot.w_368_0 ctx) (generatedRoot.w_369_0 ctx) (generatedRoot.w_370_0 ctx) (generatedRoot.w_370_concat_1 ctx) (generatedRoot.w_371_0 ctx) (generatedRoot.w_372_0 ctx) (generatedRoot.w_373_0 ctx) (generatedRoot.w_374_0 ctx) (generatedRoot.w_374_concat_1 ctx) (generatedRoot.w_375_0 ctx) (generatedRoot.w_376_0 ctx) (generatedRoot.w_377_0 ctx) (generatedRoot.w_378_0 ctx) (generatedRoot.w_379_0 ctx) (generatedRoot.w_380_0 ctx) (generatedRoot.w_381_0 ctx)) :
    generatedRoot.Facts_15 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_16  (ctx : generatedRootCtx) : Prop where
  h382_0 : MxxRuntime.concatRows (generatedRoot.w_380_0 ctx) (generatedRoot.w_381_0 ctx) (generatedRoot.w_382_0 ctx)
  h383_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_382_0 ctx) (generatedRoot.w_383_0 ctx)
  h384_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_383_0 ctx) (generatedRoot.w_384_0 ctx)
  t515 : 0 ≤ 0
  t516 : 0 < 1
  t517 : 1 ≤ 3
  t518 : 0 ≤ 0
  t519 : 0 < 25
  t520 : 25 ≤ 25
  h386_0 : MxxRuntime.sliceMatrix (generatedRoot.w_385_0 ctx) 0 1 0 25 (generatedRoot.w_386_0 ctx)
  t522 : 0 ≤ 0
  t523 : 0 < 1
  t524 : 1 ≤ 3
  t525 : 0 ≤ 0
  t526 : 0 < 25
  t527 : 25 ≤ 25
  h387_0 : MxxRuntime.sliceMatrix (generatedRoot.w_379_0 ctx) 0 1 0 25 (generatedRoot.w_387_0 ctx)
  h390_0 : MxxRuntime.centeredRebaseRuns (generatedRoot.w_55_0 ctx) (generatedRoot.w_390_0 ctx)
  h393_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_392_0 ctx) (generatedRoot.w_393_0 ctx)
  h394_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_393_0 ctx) (generatedRoot.w_394_0 ctx)
  h396_0 : MxxRuntime.centeredRebaseRuns (generatedRoot.w_72_0 ctx) (generatedRoot.w_396_0 ctx)
  h399_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_398_0 ctx) (generatedRoot.w_399_0 ctx)
  h400_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_399_0 ctx) (generatedRoot.w_400_0 ctx)
  h405_0 : MxxRuntime.familyGetStatic (generatedRoot.w_404_0 ctx) (0) (generatedRoot.w_405_0 ctx)
  h407_0 : MxxRuntime.familyGetStatic (generatedRoot.w_406_0 ctx) (0) (generatedRoot.w_407_0 ctx)
  h409_0 : MxxRuntime.familyGetStatic (generatedRoot.w_408_0 ctx) (0) (generatedRoot.w_409_0 ctx)
  h411_0 : MxxRuntime.familyGetStatic (generatedRoot.w_410_0 ctx) (0) (generatedRoot.w_411_0 ctx)
  t539 : 0 ≤ (0)
  h414_9 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_generatedRoot_414 backend hashModel tape (path ++ [414, i]) params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (current.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.2.1, ((generatedRoot.w_412_0 ctx), ((generatedRoot.w_5_0 ctx), ((generatedRoot.w_413_0 ctx), ((generatedRoot.w_54_4 ctx), ((generatedRoot.w_54_0 ctx), ((generatedRoot.w_54_2 ctx), ((generatedRoot.w_54_5 ctx), ((generatedRoot.w_54_1 ctx), ((generatedRoot.w_54_3 ctx), ((generatedRoot.w_404_0 ctx), ((generatedRoot.w_406_0 ctx), ((generatedRoot.w_408_0 ctx), ((generatedRoot.w_410_0 ctx), ()))))))))))))))))))))))) next) (Int.toNat (0)) ((generatedRoot.w_403_0 ctx), ((generatedRoot.w_403_0 ctx), ((generatedRoot.w_403_0 ctx), ((generatedRoot.w_403_0 ctx), ((generatedRoot.w_403_0 ctx), ((generatedRoot.w_403_0 ctx), ((generatedRoot.w_405_0 ctx), ((generatedRoot.w_407_0 ctx), ((generatedRoot.w_409_0 ctx), ((generatedRoot.w_411_0 ctx), ())))))))))) ((generatedRoot.w_414_0 ctx), ((generatedRoot.w_414_1 ctx), ((generatedRoot.w_414_2 ctx), ((generatedRoot.w_414_3 ctx), ((generatedRoot.w_414_4 ctx), ((generatedRoot.w_414_5 ctx), ((generatedRoot.w_414_6 ctx), ((generatedRoot.w_414_7 ctx), ((generatedRoot.w_414_8 ctx), ((generatedRoot.w_414_9 ctx), ()))))))))))
  h415_0 : MxxRuntime.familyGetStatic (generatedRoot.w_412_0 ctx) (0) (generatedRoot.w_415_0 ctx)
  t542 : 0 ≤ (0)
  t543 : (0) < 1

theorem generatedRoot.facts_16  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_16 backend hashModel tape path params (generatedRoot.w_5_0 ctx) (generatedRoot.w_404_0 ctx) (generatedRoot.w_406_0 ctx) (generatedRoot.w_408_0 ctx) (generatedRoot.w_410_0 ctx) (generatedRoot.w_412_0 ctx) (generatedRoot.w_413_0 ctx) (generatedRoot.w_54_0 ctx) (generatedRoot.w_54_1 ctx) (generatedRoot.w_54_2 ctx) (generatedRoot.w_54_3 ctx) (generatedRoot.w_54_4 ctx) (generatedRoot.w_54_5 ctx) (generatedRoot.w_55_0 ctx) (generatedRoot.w_72_0 ctx) (generatedRoot.w_379_0 ctx) (generatedRoot.w_380_0 ctx) (generatedRoot.w_381_0 ctx) (generatedRoot.w_382_0 ctx) (generatedRoot.w_383_0 ctx) (generatedRoot.w_384_0 ctx) (generatedRoot.w_385_0 ctx) (generatedRoot.w_386_0 ctx) (generatedRoot.w_387_0 ctx) (generatedRoot.w_390_0 ctx) (generatedRoot.w_392_0 ctx) (generatedRoot.w_393_0 ctx) (generatedRoot.w_394_0 ctx) (generatedRoot.w_396_0 ctx) (generatedRoot.w_398_0 ctx) (generatedRoot.w_399_0 ctx) (generatedRoot.w_400_0 ctx) (generatedRoot.w_403_0 ctx) (generatedRoot.w_405_0 ctx) (generatedRoot.w_407_0 ctx) (generatedRoot.w_409_0 ctx) (generatedRoot.w_411_0 ctx) (generatedRoot.w_414_0 ctx) (generatedRoot.w_414_1 ctx) (generatedRoot.w_414_2 ctx) (generatedRoot.w_414_3 ctx) (generatedRoot.w_414_4 ctx) (generatedRoot.w_414_5 ctx) (generatedRoot.w_414_6 ctx) (generatedRoot.w_414_7 ctx) (generatedRoot.w_414_8 ctx) (generatedRoot.w_414_9 ctx) (generatedRoot.w_415_0 ctx)) :
    generatedRoot.Facts_16 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_17  (ctx : generatedRootCtx) : Prop where
  h417_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_416_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((generatedRoot.w_5_0 ctx)) (generatedRoot.w_417_0 ctx)
  h418_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_416_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) ((generatedRoot.w_5_0 ctx)) (generatedRoot.w_418_0 ctx)
  h417_0_1 : MxxRuntime.concatRows (generatedRoot.w_415_0 ctx) (generatedRoot.w_417_0 ctx) (generatedRoot.w_419_concat_1 ctx)
  h419_0 : MxxRuntime.concatRows (generatedRoot.w_419_concat_1 ctx) (generatedRoot.w_418_0 ctx) (generatedRoot.w_419_0 ctx)
  h420_0 : MxxRuntime.familyGetStatic (generatedRoot.w_413_0 ctx) (0) (generatedRoot.w_420_0 ctx)
  h421_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_416_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((generatedRoot.w_5_0 ctx)) (generatedRoot.w_421_0 ctx)
  h422_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_416_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) ((generatedRoot.w_5_0 ctx)) (generatedRoot.w_422_0 ctx)
  h421_0_1 : MxxRuntime.concatRows (generatedRoot.w_420_0 ctx) (generatedRoot.w_421_0 ctx) (generatedRoot.w_423_concat_1 ctx)
  h423_0 : MxxRuntime.concatRows (generatedRoot.w_423_concat_1 ctx) (generatedRoot.w_422_0 ctx) (generatedRoot.w_423_0 ctx)
  h424_0 : MxxRuntime.concatColumns (generatedRoot.w_419_0 ctx) (generatedRoot.w_423_0 ctx) (generatedRoot.w_424_0 ctx)
  t554 : 0 ≤ (0)
  t555 : (0) < 1
  h425_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(9)])) (generatedRoot.w_54_4 ctx) (generatedRoot.w_425_0 ctx)
  t557 : 0 ≤ (0)
  t558 : (0) < 1
  h426_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(9)])) (generatedRoot.w_54_0 ctx) (generatedRoot.w_426_0 ctx)
  t560 : 0 ≤ (0)
  t561 : (0) < 1
  h427_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(9)])) (generatedRoot.w_54_2 ctx) (generatedRoot.w_427_0 ctx)
  h426_0_1 : MxxRuntime.concatRows (generatedRoot.w_425_0 ctx) (generatedRoot.w_426_0 ctx) (generatedRoot.w_428_concat_1 ctx)
  h428_0 : MxxRuntime.concatRows (generatedRoot.w_428_concat_1 ctx) (generatedRoot.w_427_0 ctx) (generatedRoot.w_428_0 ctx)
  t565 : 0 ≤ (0)
  t566 : (0) < 1
  h429_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(9)])) (generatedRoot.w_54_5 ctx) (generatedRoot.w_429_0 ctx)
  t568 : 0 ≤ (0)
  t569 : (0) < 1
  h430_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(9)])) (generatedRoot.w_54_1 ctx) (generatedRoot.w_430_0 ctx)
  t571 : 0 ≤ (0)
  t572 : (0) < 1
  h431_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(9)])) (generatedRoot.w_54_3 ctx) (generatedRoot.w_431_0 ctx)
  h430_0_1 : MxxRuntime.concatRows (generatedRoot.w_429_0 ctx) (generatedRoot.w_430_0 ctx) (generatedRoot.w_432_concat_1 ctx)
  h432_0 : MxxRuntime.concatRows (generatedRoot.w_432_concat_1 ctx) (generatedRoot.w_431_0 ctx) (generatedRoot.w_432_0 ctx)

theorem generatedRoot.facts_17  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_17 hashModel (generatedRoot.w_5_0 ctx) (generatedRoot.w_413_0 ctx) (generatedRoot.w_54_0 ctx) (generatedRoot.w_54_1 ctx) (generatedRoot.w_54_2 ctx) (generatedRoot.w_54_3 ctx) (generatedRoot.w_54_4 ctx) (generatedRoot.w_54_5 ctx) (generatedRoot.w_415_0 ctx) (generatedRoot.w_416_0 ctx) (generatedRoot.w_417_0 ctx) (generatedRoot.w_418_0 ctx) (generatedRoot.w_419_0 ctx) (generatedRoot.w_419_concat_1 ctx) (generatedRoot.w_420_0 ctx) (generatedRoot.w_421_0 ctx) (generatedRoot.w_422_0 ctx) (generatedRoot.w_423_0 ctx) (generatedRoot.w_423_concat_1 ctx) (generatedRoot.w_424_0 ctx) (generatedRoot.w_425_0 ctx) (generatedRoot.w_426_0 ctx) (generatedRoot.w_427_0 ctx) (generatedRoot.w_428_0 ctx) (generatedRoot.w_428_concat_1 ctx) (generatedRoot.w_429_0 ctx) (generatedRoot.w_430_0 ctx) (generatedRoot.w_431_0 ctx) (generatedRoot.w_432_0 ctx) (generatedRoot.w_432_concat_1 ctx)) :
    generatedRoot.Facts_17 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_18  (ctx : generatedRootCtx) : Prop where
  h433_0 : MxxRuntime.concatColumns (generatedRoot.w_428_0 ctx) (generatedRoot.w_432_0 ctx) (generatedRoot.w_433_0 ctx)
  h434_0 : MxxRuntime.concatRows (generatedRoot.w_414_7 ctx) (generatedRoot.w_414_9 ctx) (generatedRoot.w_434_0 ctx)
  h435_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_434_0 ctx) (generatedRoot.w_435_0 ctx)
  h436_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_435_0 ctx) (generatedRoot.w_436_0 ctx)
  t580 : 0 ≤ 1
  t581 : 1 < 2
  t582 : 2 ≤ 3
  t583 : 0 ≤ 0
  t584 : 0 < 25
  t585 : 25 ≤ 25
  h438_0 : MxxRuntime.sliceMatrix (generatedRoot.w_437_0 ctx) 1 2 0 25 (generatedRoot.w_438_0 ctx)
  t587 : 0 ≤ 2
  t588 : 2 < 3
  t589 : 3 ≤ 3
  t590 : 0 ≤ 0
  t591 : 0 < 25
  t592 : 25 ≤ 25
  h439_0 : MxxRuntime.sliceMatrix (generatedRoot.w_437_0 ctx) 2 3 0 25 (generatedRoot.w_439_0 ctx)
  h440_0 : MxxRuntime.concatRows (generatedRoot.w_438_0 ctx) (generatedRoot.w_439_0 ctx) (generatedRoot.w_440_0 ctx)
  h441_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_440_0 ctx) (generatedRoot.w_441_0 ctx)
  h442_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_441_0 ctx) (generatedRoot.w_442_0 ctx)
  t597 : 0 ≤ 0
  t598 : 0 < 1
  t599 : 1 ≤ 3
  t600 : 0 ≤ 0
  t601 : 0 < 25
  t602 : 25 ≤ 25
  h444_0 : MxxRuntime.sliceMatrix (generatedRoot.w_443_0 ctx) 0 1 0 25 (generatedRoot.w_444_0 ctx)
  t604 : 0 ≤ 0
  t605 : 0 < 1
  t606 : 1 ≤ 3
  t607 : 0 ≤ 0

theorem generatedRoot.facts_18  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_18 backend (generatedRoot.w_414_7 ctx) (generatedRoot.w_414_9 ctx) (generatedRoot.w_428_0 ctx) (generatedRoot.w_432_0 ctx) (generatedRoot.w_433_0 ctx) (generatedRoot.w_434_0 ctx) (generatedRoot.w_435_0 ctx) (generatedRoot.w_436_0 ctx) (generatedRoot.w_437_0 ctx) (generatedRoot.w_438_0 ctx) (generatedRoot.w_439_0 ctx) (generatedRoot.w_440_0 ctx) (generatedRoot.w_441_0 ctx) (generatedRoot.w_442_0 ctx) (generatedRoot.w_443_0 ctx) (generatedRoot.w_444_0 ctx)) :
    generatedRoot.Facts_18 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_19  (ctx : generatedRootCtx) : Prop where
  t608 : 0 < 25
  t609 : 25 ≤ 25
  h445_0 : MxxRuntime.sliceMatrix (generatedRoot.w_437_0 ctx) 0 1 0 25 (generatedRoot.w_445_0 ctx)
  h448_0 : MxxRuntime.centeredRebaseRuns (generatedRoot.w_55_0 ctx) (generatedRoot.w_448_0 ctx)
  h451_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_450_0 ctx) (generatedRoot.w_451_0 ctx)
  h452_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_451_0 ctx) (generatedRoot.w_452_0 ctx)
  h454_0 : MxxRuntime.centeredRebaseRuns (generatedRoot.w_72_0 ctx) (generatedRoot.w_454_0 ctx)
  h457_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_456_0 ctx) (generatedRoot.w_457_0 ctx)
  h458_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_457_0 ctx) (generatedRoot.w_458_0 ctx)
  h463_0 : MxxRuntime.familyGetStatic (generatedRoot.w_462_0 ctx) (0) (generatedRoot.w_463_0 ctx)
  h465_0 : MxxRuntime.familyGetStatic (generatedRoot.w_464_0 ctx) (0) (generatedRoot.w_465_0 ctx)
  h467_0 : MxxRuntime.familyGetStatic (generatedRoot.w_466_0 ctx) (0) (generatedRoot.w_467_0 ctx)
  h469_0 : MxxRuntime.familyGetStatic (generatedRoot.w_468_0 ctx) (0) (generatedRoot.w_469_0 ctx)
  t621 : 0 ≤ (0)
  h472_9 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_generatedRoot_472 backend hashModel tape (path ++ [472, i]) params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (current.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.2.1, ((generatedRoot.w_470_0 ctx), ((generatedRoot.w_5_0 ctx), ((generatedRoot.w_471_0 ctx), ((generatedRoot.w_54_4 ctx), ((generatedRoot.w_54_0 ctx), ((generatedRoot.w_54_2 ctx), ((generatedRoot.w_54_5 ctx), ((generatedRoot.w_54_1 ctx), ((generatedRoot.w_54_3 ctx), ((generatedRoot.w_462_0 ctx), ((generatedRoot.w_464_0 ctx), ((generatedRoot.w_466_0 ctx), ((generatedRoot.w_468_0 ctx), ()))))))))))))))))))))))) next) (Int.toNat (0)) ((generatedRoot.w_461_0 ctx), ((generatedRoot.w_461_0 ctx), ((generatedRoot.w_461_0 ctx), ((generatedRoot.w_461_0 ctx), ((generatedRoot.w_461_0 ctx), ((generatedRoot.w_461_0 ctx), ((generatedRoot.w_463_0 ctx), ((generatedRoot.w_465_0 ctx), ((generatedRoot.w_467_0 ctx), ((generatedRoot.w_469_0 ctx), ())))))))))) ((generatedRoot.w_472_0 ctx), ((generatedRoot.w_472_1 ctx), ((generatedRoot.w_472_2 ctx), ((generatedRoot.w_472_3 ctx), ((generatedRoot.w_472_4 ctx), ((generatedRoot.w_472_5 ctx), ((generatedRoot.w_472_6 ctx), ((generatedRoot.w_472_7 ctx), ((generatedRoot.w_472_8 ctx), ((generatedRoot.w_472_9 ctx), ()))))))))))
  h473_0 : MxxRuntime.familyGetStatic (generatedRoot.w_470_0 ctx) (0) (generatedRoot.w_473_0 ctx)
  t624 : 0 ≤ (0)
  t625 : (0) < 1
  h475_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_474_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((generatedRoot.w_5_0 ctx)) (generatedRoot.w_475_0 ctx)
  h476_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_474_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) ((generatedRoot.w_5_0 ctx)) (generatedRoot.w_476_0 ctx)
  h475_0_1 : MxxRuntime.concatRows (generatedRoot.w_473_0 ctx) (generatedRoot.w_475_0 ctx) (generatedRoot.w_477_concat_1 ctx)
  h477_0 : MxxRuntime.concatRows (generatedRoot.w_477_concat_1 ctx) (generatedRoot.w_476_0 ctx) (generatedRoot.w_477_0 ctx)
  h478_0 : MxxRuntime.familyGetStatic (generatedRoot.w_471_0 ctx) (0) (generatedRoot.w_478_0 ctx)
  h479_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_474_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((generatedRoot.w_5_0 ctx)) (generatedRoot.w_479_0 ctx)
  h480_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_474_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) ((generatedRoot.w_5_0 ctx)) (generatedRoot.w_480_0 ctx)
  h479_0_1 : MxxRuntime.concatRows (generatedRoot.w_478_0 ctx) (generatedRoot.w_479_0 ctx) (generatedRoot.w_481_concat_1 ctx)
  h481_0 : MxxRuntime.concatRows (generatedRoot.w_481_concat_1 ctx) (generatedRoot.w_480_0 ctx) (generatedRoot.w_481_0 ctx)
  h482_0 : MxxRuntime.concatColumns (generatedRoot.w_477_0 ctx) (generatedRoot.w_481_0 ctx) (generatedRoot.w_482_0 ctx)
  t636 : 0 ≤ (0)
  t637 : (0) < 1
  h483_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(11)])) (generatedRoot.w_54_4 ctx) (generatedRoot.w_483_0 ctx)
  t639 : 0 ≤ (0)

theorem generatedRoot.facts_19  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_19 backend hashModel tape path params (generatedRoot.w_5_0 ctx) (generatedRoot.w_462_0 ctx) (generatedRoot.w_464_0 ctx) (generatedRoot.w_466_0 ctx) (generatedRoot.w_468_0 ctx) (generatedRoot.w_470_0 ctx) (generatedRoot.w_471_0 ctx) (generatedRoot.w_54_0 ctx) (generatedRoot.w_54_1 ctx) (generatedRoot.w_54_2 ctx) (generatedRoot.w_54_3 ctx) (generatedRoot.w_54_4 ctx) (generatedRoot.w_54_5 ctx) (generatedRoot.w_55_0 ctx) (generatedRoot.w_72_0 ctx) (generatedRoot.w_437_0 ctx) (generatedRoot.w_445_0 ctx) (generatedRoot.w_448_0 ctx) (generatedRoot.w_450_0 ctx) (generatedRoot.w_451_0 ctx) (generatedRoot.w_452_0 ctx) (generatedRoot.w_454_0 ctx) (generatedRoot.w_456_0 ctx) (generatedRoot.w_457_0 ctx) (generatedRoot.w_458_0 ctx) (generatedRoot.w_461_0 ctx) (generatedRoot.w_463_0 ctx) (generatedRoot.w_465_0 ctx) (generatedRoot.w_467_0 ctx) (generatedRoot.w_469_0 ctx) (generatedRoot.w_472_0 ctx) (generatedRoot.w_472_1 ctx) (generatedRoot.w_472_2 ctx) (generatedRoot.w_472_3 ctx) (generatedRoot.w_472_4 ctx) (generatedRoot.w_472_5 ctx) (generatedRoot.w_472_6 ctx) (generatedRoot.w_472_7 ctx) (generatedRoot.w_472_8 ctx) (generatedRoot.w_472_9 ctx) (generatedRoot.w_473_0 ctx) (generatedRoot.w_474_0 ctx) (generatedRoot.w_475_0 ctx) (generatedRoot.w_476_0 ctx) (generatedRoot.w_477_0 ctx) (generatedRoot.w_477_concat_1 ctx) (generatedRoot.w_478_0 ctx) (generatedRoot.w_479_0 ctx) (generatedRoot.w_480_0 ctx) (generatedRoot.w_481_0 ctx) (generatedRoot.w_481_concat_1 ctx) (generatedRoot.w_482_0 ctx) (generatedRoot.w_483_0 ctx)) :
    generatedRoot.Facts_19 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_20  (ctx : generatedRootCtx) : Prop where
  t640 : (0) < 1
  h484_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(11)])) (generatedRoot.w_54_0 ctx) (generatedRoot.w_484_0 ctx)
  t642 : 0 ≤ (0)
  t643 : (0) < 1
  h485_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(11)])) (generatedRoot.w_54_2 ctx) (generatedRoot.w_485_0 ctx)
  h484_0_1 : MxxRuntime.concatRows (generatedRoot.w_483_0 ctx) (generatedRoot.w_484_0 ctx) (generatedRoot.w_486_concat_1 ctx)
  h486_0 : MxxRuntime.concatRows (generatedRoot.w_486_concat_1 ctx) (generatedRoot.w_485_0 ctx) (generatedRoot.w_486_0 ctx)
  t647 : 0 ≤ (0)
  t648 : (0) < 1
  h487_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(11)])) (generatedRoot.w_54_5 ctx) (generatedRoot.w_487_0 ctx)
  t650 : 0 ≤ (0)
  t651 : (0) < 1
  h488_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(11)])) (generatedRoot.w_54_1 ctx) (generatedRoot.w_488_0 ctx)
  t653 : 0 ≤ (0)
  t654 : (0) < 1
  h489_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(11)])) (generatedRoot.w_54_3 ctx) (generatedRoot.w_489_0 ctx)
  h488_0_1 : MxxRuntime.concatRows (generatedRoot.w_487_0 ctx) (generatedRoot.w_488_0 ctx) (generatedRoot.w_490_concat_1 ctx)
  h490_0 : MxxRuntime.concatRows (generatedRoot.w_490_concat_1 ctx) (generatedRoot.w_489_0 ctx) (generatedRoot.w_490_0 ctx)
  h491_0 : MxxRuntime.concatColumns (generatedRoot.w_486_0 ctx) (generatedRoot.w_490_0 ctx) (generatedRoot.w_491_0 ctx)
  h492_0 : MxxRuntime.concatRows (generatedRoot.w_472_7 ctx) (generatedRoot.w_472_9 ctx) (generatedRoot.w_492_0 ctx)
  h493_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_492_0 ctx) (generatedRoot.w_493_0 ctx)
  h494_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_493_0 ctx) (generatedRoot.w_494_0 ctx)
  t662 : 0 ≤ 1
  t663 : 1 < 2
  t664 : 2 ≤ 3
  t665 : 0 ≤ 0
  t666 : 0 < 25
  t667 : 25 ≤ 25
  h496_0 : MxxRuntime.sliceMatrix (generatedRoot.w_495_0 ctx) 1 2 0 25 (generatedRoot.w_496_0 ctx)
  t669 : 0 ≤ 2
  t670 : 2 < 3
  t671 : 3 ≤ 3

theorem generatedRoot.facts_20  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_20 backend (generatedRoot.w_54_0 ctx) (generatedRoot.w_54_1 ctx) (generatedRoot.w_54_2 ctx) (generatedRoot.w_54_3 ctx) (generatedRoot.w_54_5 ctx) (generatedRoot.w_472_7 ctx) (generatedRoot.w_472_9 ctx) (generatedRoot.w_483_0 ctx) (generatedRoot.w_484_0 ctx) (generatedRoot.w_485_0 ctx) (generatedRoot.w_486_0 ctx) (generatedRoot.w_486_concat_1 ctx) (generatedRoot.w_487_0 ctx) (generatedRoot.w_488_0 ctx) (generatedRoot.w_489_0 ctx) (generatedRoot.w_490_0 ctx) (generatedRoot.w_490_concat_1 ctx) (generatedRoot.w_491_0 ctx) (generatedRoot.w_492_0 ctx) (generatedRoot.w_493_0 ctx) (generatedRoot.w_494_0 ctx) (generatedRoot.w_495_0 ctx) (generatedRoot.w_496_0 ctx)) :
    generatedRoot.Facts_20 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_21  (ctx : generatedRootCtx) : Prop where
  t672 : 0 ≤ 0
  t673 : 0 < 25
  t674 : 25 ≤ 25
  h497_0 : MxxRuntime.sliceMatrix (generatedRoot.w_495_0 ctx) 2 3 0 25 (generatedRoot.w_497_0 ctx)
  h498_0 : MxxRuntime.concatRows (generatedRoot.w_496_0 ctx) (generatedRoot.w_497_0 ctx) (generatedRoot.w_498_0 ctx)
  h499_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_498_0 ctx) (generatedRoot.w_499_0 ctx)
  h500_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_499_0 ctx) (generatedRoot.w_500_0 ctx)
  t679 : 0 ≤ 0
  t680 : 0 < 1
  t681 : 1 ≤ 3
  t682 : 0 ≤ 0
  t683 : 0 < 25
  t684 : 25 ≤ 25
  h502_0 : MxxRuntime.sliceMatrix (generatedRoot.w_501_0 ctx) 0 1 0 25 (generatedRoot.w_502_0 ctx)
  t686 : 0 ≤ 0
  t687 : 0 < 1
  t688 : 1 ≤ 3
  t689 : 0 ≤ 0
  t690 : 0 < 25
  t691 : 25 ≤ 25
  h503_0 : MxxRuntime.sliceMatrix (generatedRoot.w_495_0 ctx) 0 1 0 25 (generatedRoot.w_503_0 ctx)
  h506_0 : MxxRuntime.centeredRebaseRuns (generatedRoot.w_55_0 ctx) (generatedRoot.w_506_0 ctx)
  h509_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_508_0 ctx) (generatedRoot.w_509_0 ctx)
  h510_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_509_0 ctx) (generatedRoot.w_510_0 ctx)
  h512_0 : MxxRuntime.centeredRebaseRuns (generatedRoot.w_72_0 ctx) (generatedRoot.w_512_0 ctx)
  h515_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_514_0 ctx) (generatedRoot.w_515_0 ctx)
  h516_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_515_0 ctx) (generatedRoot.w_516_0 ctx)
  h521_0 : MxxRuntime.familyGetStatic (generatedRoot.w_520_0 ctx) (0) (generatedRoot.w_521_0 ctx)
  h523_0 : MxxRuntime.familyGetStatic (generatedRoot.w_522_0 ctx) (0) (generatedRoot.w_523_0 ctx)
  h525_0 : MxxRuntime.familyGetStatic (generatedRoot.w_524_0 ctx) (0) (generatedRoot.w_525_0 ctx)
  h527_0 : MxxRuntime.familyGetStatic (generatedRoot.w_526_0 ctx) (0) (generatedRoot.w_527_0 ctx)
  t703 : 0 ≤ (0)

theorem generatedRoot.facts_21  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_21 backend (generatedRoot.w_520_0 ctx) (generatedRoot.w_522_0 ctx) (generatedRoot.w_524_0 ctx) (generatedRoot.w_526_0 ctx) (generatedRoot.w_55_0 ctx) (generatedRoot.w_72_0 ctx) (generatedRoot.w_495_0 ctx) (generatedRoot.w_496_0 ctx) (generatedRoot.w_497_0 ctx) (generatedRoot.w_498_0 ctx) (generatedRoot.w_499_0 ctx) (generatedRoot.w_500_0 ctx) (generatedRoot.w_501_0 ctx) (generatedRoot.w_502_0 ctx) (generatedRoot.w_503_0 ctx) (generatedRoot.w_506_0 ctx) (generatedRoot.w_508_0 ctx) (generatedRoot.w_509_0 ctx) (generatedRoot.w_510_0 ctx) (generatedRoot.w_512_0 ctx) (generatedRoot.w_514_0 ctx) (generatedRoot.w_515_0 ctx) (generatedRoot.w_516_0 ctx) (generatedRoot.w_521_0 ctx) (generatedRoot.w_523_0 ctx) (generatedRoot.w_525_0 ctx) (generatedRoot.w_527_0 ctx)) :
    generatedRoot.Facts_21 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_22  (ctx : generatedRootCtx) : Prop where
  h530_9 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_generatedRoot_530 backend hashModel tape (path ++ [530, i]) params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (current.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.2.1, ((generatedRoot.w_528_0 ctx), ((generatedRoot.w_5_0 ctx), ((generatedRoot.w_529_0 ctx), ((generatedRoot.w_54_4 ctx), ((generatedRoot.w_54_0 ctx), ((generatedRoot.w_54_2 ctx), ((generatedRoot.w_54_5 ctx), ((generatedRoot.w_54_1 ctx), ((generatedRoot.w_54_3 ctx), ((generatedRoot.w_520_0 ctx), ((generatedRoot.w_522_0 ctx), ((generatedRoot.w_524_0 ctx), ((generatedRoot.w_526_0 ctx), ()))))))))))))))))))))))) next) (Int.toNat (0)) ((generatedRoot.w_519_0 ctx), ((generatedRoot.w_519_0 ctx), ((generatedRoot.w_519_0 ctx), ((generatedRoot.w_519_0 ctx), ((generatedRoot.w_519_0 ctx), ((generatedRoot.w_519_0 ctx), ((generatedRoot.w_521_0 ctx), ((generatedRoot.w_523_0 ctx), ((generatedRoot.w_525_0 ctx), ((generatedRoot.w_527_0 ctx), ())))))))))) ((generatedRoot.w_530_0 ctx), ((generatedRoot.w_530_1 ctx), ((generatedRoot.w_530_2 ctx), ((generatedRoot.w_530_3 ctx), ((generatedRoot.w_530_4 ctx), ((generatedRoot.w_530_5 ctx), ((generatedRoot.w_530_6 ctx), ((generatedRoot.w_530_7 ctx), ((generatedRoot.w_530_8 ctx), ((generatedRoot.w_530_9 ctx), ()))))))))))
  h531_0 : MxxRuntime.familyGetStatic (generatedRoot.w_528_0 ctx) (0) (generatedRoot.w_531_0 ctx)
  t706 : 0 ≤ (0)
  t707 : (0) < 1
  h533_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_532_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((generatedRoot.w_5_0 ctx)) (generatedRoot.w_533_0 ctx)
  h534_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_532_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) ((generatedRoot.w_5_0 ctx)) (generatedRoot.w_534_0 ctx)
  h533_0_1 : MxxRuntime.concatRows (generatedRoot.w_531_0 ctx) (generatedRoot.w_533_0 ctx) (generatedRoot.w_535_concat_1 ctx)
  h535_0 : MxxRuntime.concatRows (generatedRoot.w_535_concat_1 ctx) (generatedRoot.w_534_0 ctx) (generatedRoot.w_535_0 ctx)
  h536_0 : MxxRuntime.familyGetStatic (generatedRoot.w_529_0 ctx) (0) (generatedRoot.w_536_0 ctx)
  h537_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_532_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((generatedRoot.w_5_0 ctx)) (generatedRoot.w_537_0 ctx)
  h538_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_532_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) ((generatedRoot.w_5_0 ctx)) (generatedRoot.w_538_0 ctx)
  h537_0_1 : MxxRuntime.concatRows (generatedRoot.w_536_0 ctx) (generatedRoot.w_537_0 ctx) (generatedRoot.w_539_concat_1 ctx)
  h539_0 : MxxRuntime.concatRows (generatedRoot.w_539_concat_1 ctx) (generatedRoot.w_538_0 ctx) (generatedRoot.w_539_0 ctx)
  h540_0 : MxxRuntime.concatColumns (generatedRoot.w_535_0 ctx) (generatedRoot.w_539_0 ctx) (generatedRoot.w_540_0 ctx)
  t718 : 0 ≤ (0)
  t719 : (0) < 1
  h541_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(13)])) (generatedRoot.w_54_4 ctx) (generatedRoot.w_541_0 ctx)
  t721 : 0 ≤ (0)
  t722 : (0) < 1
  h542_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(13)])) (generatedRoot.w_54_0 ctx) (generatedRoot.w_542_0 ctx)
  t724 : 0 ≤ (0)
  t725 : (0) < 1
  h543_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(13)])) (generatedRoot.w_54_2 ctx) (generatedRoot.w_543_0 ctx)
  h542_0_1 : MxxRuntime.concatRows (generatedRoot.w_541_0 ctx) (generatedRoot.w_542_0 ctx) (generatedRoot.w_544_concat_1 ctx)
  h544_0 : MxxRuntime.concatRows (generatedRoot.w_544_concat_1 ctx) (generatedRoot.w_543_0 ctx) (generatedRoot.w_544_0 ctx)
  t729 : 0 ≤ (0)
  t730 : (0) < 1
  h545_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(13)])) (generatedRoot.w_54_5 ctx) (generatedRoot.w_545_0 ctx)
  t732 : 0 ≤ (0)
  t733 : (0) < 1
  h546_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(13)])) (generatedRoot.w_54_1 ctx) (generatedRoot.w_546_0 ctx)
  t735 : 0 ≤ (0)

theorem generatedRoot.facts_22  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_22 backend hashModel tape path params (generatedRoot.w_5_0 ctx) (generatedRoot.w_520_0 ctx) (generatedRoot.w_522_0 ctx) (generatedRoot.w_524_0 ctx) (generatedRoot.w_526_0 ctx) (generatedRoot.w_528_0 ctx) (generatedRoot.w_529_0 ctx) (generatedRoot.w_54_0 ctx) (generatedRoot.w_54_1 ctx) (generatedRoot.w_54_2 ctx) (generatedRoot.w_54_3 ctx) (generatedRoot.w_54_4 ctx) (generatedRoot.w_54_5 ctx) (generatedRoot.w_519_0 ctx) (generatedRoot.w_521_0 ctx) (generatedRoot.w_523_0 ctx) (generatedRoot.w_525_0 ctx) (generatedRoot.w_527_0 ctx) (generatedRoot.w_530_0 ctx) (generatedRoot.w_530_1 ctx) (generatedRoot.w_530_2 ctx) (generatedRoot.w_530_3 ctx) (generatedRoot.w_530_4 ctx) (generatedRoot.w_530_5 ctx) (generatedRoot.w_530_6 ctx) (generatedRoot.w_530_7 ctx) (generatedRoot.w_530_8 ctx) (generatedRoot.w_530_9 ctx) (generatedRoot.w_531_0 ctx) (generatedRoot.w_532_0 ctx) (generatedRoot.w_533_0 ctx) (generatedRoot.w_534_0 ctx) (generatedRoot.w_535_0 ctx) (generatedRoot.w_535_concat_1 ctx) (generatedRoot.w_536_0 ctx) (generatedRoot.w_537_0 ctx) (generatedRoot.w_538_0 ctx) (generatedRoot.w_539_0 ctx) (generatedRoot.w_539_concat_1 ctx) (generatedRoot.w_540_0 ctx) (generatedRoot.w_541_0 ctx) (generatedRoot.w_542_0 ctx) (generatedRoot.w_543_0 ctx) (generatedRoot.w_544_0 ctx) (generatedRoot.w_544_concat_1 ctx) (generatedRoot.w_545_0 ctx) (generatedRoot.w_546_0 ctx)) :
    generatedRoot.Facts_22 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_23  (ctx : generatedRootCtx) : Prop where
  t736 : (0) < 1
  h547_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(13)])) (generatedRoot.w_54_3 ctx) (generatedRoot.w_547_0 ctx)
  h546_0_1 : MxxRuntime.concatRows (generatedRoot.w_545_0 ctx) (generatedRoot.w_546_0 ctx) (generatedRoot.w_548_concat_1 ctx)
  h548_0 : MxxRuntime.concatRows (generatedRoot.w_548_concat_1 ctx) (generatedRoot.w_547_0 ctx) (generatedRoot.w_548_0 ctx)
  h549_0 : MxxRuntime.concatColumns (generatedRoot.w_544_0 ctx) (generatedRoot.w_548_0 ctx) (generatedRoot.w_549_0 ctx)
  h550_0 : MxxRuntime.concatRows (generatedRoot.w_530_7 ctx) (generatedRoot.w_530_9 ctx) (generatedRoot.w_550_0 ctx)
  h551_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_550_0 ctx) (generatedRoot.w_551_0 ctx)
  h552_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_551_0 ctx) (generatedRoot.w_552_0 ctx)
  t744 : 0 ≤ 1
  t745 : 1 < 2
  t746 : 2 ≤ 3
  t747 : 0 ≤ 0
  t748 : 0 < 25
  t749 : 25 ≤ 25
  h554_0 : MxxRuntime.sliceMatrix (generatedRoot.w_553_0 ctx) 1 2 0 25 (generatedRoot.w_554_0 ctx)
  t751 : 0 ≤ 2
  t752 : 2 < 3
  t753 : 3 ≤ 3
  t754 : 0 ≤ 0
  t755 : 0 < 25
  t756 : 25 ≤ 25
  h555_0 : MxxRuntime.sliceMatrix (generatedRoot.w_553_0 ctx) 2 3 0 25 (generatedRoot.w_555_0 ctx)
  h556_0 : MxxRuntime.concatRows (generatedRoot.w_554_0 ctx) (generatedRoot.w_555_0 ctx) (generatedRoot.w_556_0 ctx)
  h557_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_556_0 ctx) (generatedRoot.w_557_0 ctx)
  h558_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_557_0 ctx) (generatedRoot.w_558_0 ctx)
  t761 : 0 ≤ 0
  t762 : 0 < 1
  t763 : 1 ≤ 3
  t764 : 0 ≤ 0
  t765 : 0 < 25
  t766 : 25 ≤ 25
  h560_0 : MxxRuntime.sliceMatrix (generatedRoot.w_559_0 ctx) 0 1 0 25 (generatedRoot.w_560_0 ctx)

theorem generatedRoot.facts_23  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_23 backend (generatedRoot.w_54_3 ctx) (generatedRoot.w_530_7 ctx) (generatedRoot.w_530_9 ctx) (generatedRoot.w_544_0 ctx) (generatedRoot.w_545_0 ctx) (generatedRoot.w_546_0 ctx) (generatedRoot.w_547_0 ctx) (generatedRoot.w_548_0 ctx) (generatedRoot.w_548_concat_1 ctx) (generatedRoot.w_549_0 ctx) (generatedRoot.w_550_0 ctx) (generatedRoot.w_551_0 ctx) (generatedRoot.w_552_0 ctx) (generatedRoot.w_553_0 ctx) (generatedRoot.w_554_0 ctx) (generatedRoot.w_555_0 ctx) (generatedRoot.w_556_0 ctx) (generatedRoot.w_557_0 ctx) (generatedRoot.w_558_0 ctx) (generatedRoot.w_559_0 ctx) (generatedRoot.w_560_0 ctx)) :
    generatedRoot.Facts_23 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_24  (ctx : generatedRootCtx) : Prop where
  t768 : 0 ≤ 0
  t769 : 0 < 1
  t770 : 1 ≤ 3
  t771 : 0 ≤ 0
  t772 : 0 < 25
  t773 : 25 ≤ 25
  h561_0 : MxxRuntime.sliceMatrix (generatedRoot.w_553_0 ctx) 0 1 0 25 (generatedRoot.w_561_0 ctx)
  h564_0 : MxxRuntime.centeredRebaseRuns (generatedRoot.w_55_0 ctx) (generatedRoot.w_564_0 ctx)
  h567_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_566_0 ctx) (generatedRoot.w_567_0 ctx)
  h568_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_567_0 ctx) (generatedRoot.w_568_0 ctx)
  h570_0 : MxxRuntime.centeredRebaseRuns (generatedRoot.w_72_0 ctx) (generatedRoot.w_570_0 ctx)
  h573_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_572_0 ctx) (generatedRoot.w_573_0 ctx)
  h574_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_573_0 ctx) (generatedRoot.w_574_0 ctx)
  h579_0 : MxxRuntime.familyGetStatic (generatedRoot.w_578_0 ctx) (0) (generatedRoot.w_579_0 ctx)
  h581_0 : MxxRuntime.familyGetStatic (generatedRoot.w_580_0 ctx) (0) (generatedRoot.w_581_0 ctx)
  h583_0 : MxxRuntime.familyGetStatic (generatedRoot.w_582_0 ctx) (0) (generatedRoot.w_583_0 ctx)
  h585_0 : MxxRuntime.familyGetStatic (generatedRoot.w_584_0 ctx) (0) (generatedRoot.w_585_0 ctx)
  t785 : 0 ≤ (0)
  h588_9 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_generatedRoot_588 backend hashModel tape (path ++ [588, i]) params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (current.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.2.1, ((generatedRoot.w_586_0 ctx), ((generatedRoot.w_5_0 ctx), ((generatedRoot.w_587_0 ctx), ((generatedRoot.w_54_4 ctx), ((generatedRoot.w_54_0 ctx), ((generatedRoot.w_54_2 ctx), ((generatedRoot.w_54_5 ctx), ((generatedRoot.w_54_1 ctx), ((generatedRoot.w_54_3 ctx), ((generatedRoot.w_578_0 ctx), ((generatedRoot.w_580_0 ctx), ((generatedRoot.w_582_0 ctx), ((generatedRoot.w_584_0 ctx), ()))))))))))))))))))))))) next) (Int.toNat (0)) ((generatedRoot.w_577_0 ctx), ((generatedRoot.w_577_0 ctx), ((generatedRoot.w_577_0 ctx), ((generatedRoot.w_577_0 ctx), ((generatedRoot.w_577_0 ctx), ((generatedRoot.w_577_0 ctx), ((generatedRoot.w_579_0 ctx), ((generatedRoot.w_581_0 ctx), ((generatedRoot.w_583_0 ctx), ((generatedRoot.w_585_0 ctx), ())))))))))) ((generatedRoot.w_588_0 ctx), ((generatedRoot.w_588_1 ctx), ((generatedRoot.w_588_2 ctx), ((generatedRoot.w_588_3 ctx), ((generatedRoot.w_588_4 ctx), ((generatedRoot.w_588_5 ctx), ((generatedRoot.w_588_6 ctx), ((generatedRoot.w_588_7 ctx), ((generatedRoot.w_588_8 ctx), ((generatedRoot.w_588_9 ctx), ()))))))))))
  h589_0 : MxxRuntime.familyGetStatic (generatedRoot.w_586_0 ctx) (0) (generatedRoot.w_589_0 ctx)
  t788 : 0 ≤ (0)
  t789 : (0) < 1
  h591_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_590_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((generatedRoot.w_5_0 ctx)) (generatedRoot.w_591_0 ctx)
  h592_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_590_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) ((generatedRoot.w_5_0 ctx)) (generatedRoot.w_592_0 ctx)
  h591_0_1 : MxxRuntime.concatRows (generatedRoot.w_589_0 ctx) (generatedRoot.w_591_0 ctx) (generatedRoot.w_593_concat_1 ctx)
  h593_0 : MxxRuntime.concatRows (generatedRoot.w_593_concat_1 ctx) (generatedRoot.w_592_0 ctx) (generatedRoot.w_593_0 ctx)
  h594_0 : MxxRuntime.familyGetStatic (generatedRoot.w_587_0 ctx) (0) (generatedRoot.w_594_0 ctx)
  h595_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_590_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((generatedRoot.w_5_0 ctx)) (generatedRoot.w_595_0 ctx)
  h596_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_590_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) ((generatedRoot.w_5_0 ctx)) (generatedRoot.w_596_0 ctx)
  h595_0_1 : MxxRuntime.concatRows (generatedRoot.w_594_0 ctx) (generatedRoot.w_595_0 ctx) (generatedRoot.w_597_concat_1 ctx)
  h597_0 : MxxRuntime.concatRows (generatedRoot.w_597_concat_1 ctx) (generatedRoot.w_596_0 ctx) (generatedRoot.w_597_0 ctx)
  h598_0 : MxxRuntime.concatColumns (generatedRoot.w_593_0 ctx) (generatedRoot.w_597_0 ctx) (generatedRoot.w_598_0 ctx)

theorem generatedRoot.facts_24  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_24 backend hashModel tape path params (generatedRoot.w_5_0 ctx) (generatedRoot.w_578_0 ctx) (generatedRoot.w_580_0 ctx) (generatedRoot.w_582_0 ctx) (generatedRoot.w_584_0 ctx) (generatedRoot.w_586_0 ctx) (generatedRoot.w_587_0 ctx) (generatedRoot.w_54_0 ctx) (generatedRoot.w_54_1 ctx) (generatedRoot.w_54_2 ctx) (generatedRoot.w_54_3 ctx) (generatedRoot.w_54_4 ctx) (generatedRoot.w_54_5 ctx) (generatedRoot.w_55_0 ctx) (generatedRoot.w_72_0 ctx) (generatedRoot.w_553_0 ctx) (generatedRoot.w_561_0 ctx) (generatedRoot.w_564_0 ctx) (generatedRoot.w_566_0 ctx) (generatedRoot.w_567_0 ctx) (generatedRoot.w_568_0 ctx) (generatedRoot.w_570_0 ctx) (generatedRoot.w_572_0 ctx) (generatedRoot.w_573_0 ctx) (generatedRoot.w_574_0 ctx) (generatedRoot.w_577_0 ctx) (generatedRoot.w_579_0 ctx) (generatedRoot.w_581_0 ctx) (generatedRoot.w_583_0 ctx) (generatedRoot.w_585_0 ctx) (generatedRoot.w_588_0 ctx) (generatedRoot.w_588_1 ctx) (generatedRoot.w_588_2 ctx) (generatedRoot.w_588_3 ctx) (generatedRoot.w_588_4 ctx) (generatedRoot.w_588_5 ctx) (generatedRoot.w_588_6 ctx) (generatedRoot.w_588_7 ctx) (generatedRoot.w_588_8 ctx) (generatedRoot.w_588_9 ctx) (generatedRoot.w_589_0 ctx) (generatedRoot.w_590_0 ctx) (generatedRoot.w_591_0 ctx) (generatedRoot.w_592_0 ctx) (generatedRoot.w_593_0 ctx) (generatedRoot.w_593_concat_1 ctx) (generatedRoot.w_594_0 ctx) (generatedRoot.w_595_0 ctx) (generatedRoot.w_596_0 ctx) (generatedRoot.w_597_0 ctx) (generatedRoot.w_597_concat_1 ctx) (generatedRoot.w_598_0 ctx)) :
    generatedRoot.Facts_24 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_25  (ctx : generatedRootCtx) : Prop where
  t800 : 0 ≤ (0)
  t801 : (0) < 1
  h599_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(15)])) (generatedRoot.w_54_4 ctx) (generatedRoot.w_599_0 ctx)
  t803 : 0 ≤ (0)
  t804 : (0) < 1
  h600_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(15)])) (generatedRoot.w_54_0 ctx) (generatedRoot.w_600_0 ctx)
  t806 : 0 ≤ (0)
  t807 : (0) < 1
  h601_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(15)])) (generatedRoot.w_54_2 ctx) (generatedRoot.w_601_0 ctx)
  h600_0_1 : MxxRuntime.concatRows (generatedRoot.w_599_0 ctx) (generatedRoot.w_600_0 ctx) (generatedRoot.w_602_concat_1 ctx)
  h602_0 : MxxRuntime.concatRows (generatedRoot.w_602_concat_1 ctx) (generatedRoot.w_601_0 ctx) (generatedRoot.w_602_0 ctx)
  t811 : 0 ≤ (0)
  t812 : (0) < 1
  h603_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(15)])) (generatedRoot.w_54_5 ctx) (generatedRoot.w_603_0 ctx)
  t814 : 0 ≤ (0)
  t815 : (0) < 1
  h604_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(15)])) (generatedRoot.w_54_1 ctx) (generatedRoot.w_604_0 ctx)
  t817 : 0 ≤ (0)
  t818 : (0) < 1
  h605_0 : MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(15)])) (generatedRoot.w_54_3 ctx) (generatedRoot.w_605_0 ctx)
  h604_0_1 : MxxRuntime.concatRows (generatedRoot.w_603_0 ctx) (generatedRoot.w_604_0 ctx) (generatedRoot.w_606_concat_1 ctx)
  h606_0 : MxxRuntime.concatRows (generatedRoot.w_606_concat_1 ctx) (generatedRoot.w_605_0 ctx) (generatedRoot.w_606_0 ctx)
  h607_0 : MxxRuntime.concatColumns (generatedRoot.w_602_0 ctx) (generatedRoot.w_606_0 ctx) (generatedRoot.w_607_0 ctx)
  h608_0 : MxxRuntime.concatRows (generatedRoot.w_588_7 ctx) (generatedRoot.w_588_9 ctx) (generatedRoot.w_608_0 ctx)
  h609_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_608_0 ctx) (generatedRoot.w_609_0 ctx)
  h610_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_609_0 ctx) (generatedRoot.w_610_0 ctx)
  t826 : 0 ≤ 1
  t827 : 1 < 2
  t828 : 2 ≤ 3
  t829 : 0 ≤ 0
  t830 : 0 < 25
  t831 : 25 ≤ 25

theorem generatedRoot.facts_25  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_25 backend (generatedRoot.w_54_0 ctx) (generatedRoot.w_54_1 ctx) (generatedRoot.w_54_2 ctx) (generatedRoot.w_54_3 ctx) (generatedRoot.w_54_4 ctx) (generatedRoot.w_54_5 ctx) (generatedRoot.w_588_7 ctx) (generatedRoot.w_588_9 ctx) (generatedRoot.w_599_0 ctx) (generatedRoot.w_600_0 ctx) (generatedRoot.w_601_0 ctx) (generatedRoot.w_602_0 ctx) (generatedRoot.w_602_concat_1 ctx) (generatedRoot.w_603_0 ctx) (generatedRoot.w_604_0 ctx) (generatedRoot.w_605_0 ctx) (generatedRoot.w_606_0 ctx) (generatedRoot.w_606_concat_1 ctx) (generatedRoot.w_607_0 ctx) (generatedRoot.w_608_0 ctx) (generatedRoot.w_609_0 ctx) (generatedRoot.w_610_0 ctx)) :
    generatedRoot.Facts_25 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_26  (ctx : generatedRootCtx) : Prop where
  h612_0 : MxxRuntime.sliceMatrix (generatedRoot.w_611_0 ctx) 1 2 0 25 (generatedRoot.w_612_0 ctx)
  t833 : 0 ≤ 2
  t834 : 2 < 3
  t835 : 3 ≤ 3
  t836 : 0 ≤ 0
  t837 : 0 < 25
  t838 : 25 ≤ 25
  h613_0 : MxxRuntime.sliceMatrix (generatedRoot.w_611_0 ctx) 2 3 0 25 (generatedRoot.w_613_0 ctx)
  h614_0 : MxxRuntime.concatRows (generatedRoot.w_612_0 ctx) (generatedRoot.w_613_0 ctx) (generatedRoot.w_614_0 ctx)
  h615_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_614_0 ctx) (generatedRoot.w_615_0 ctx)
  h616_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_615_0 ctx) (generatedRoot.w_616_0 ctx)
  t843 : 0 ≤ 0
  t844 : 0 < 1
  t845 : 1 ≤ 3
  t846 : 0 ≤ 0
  t847 : 0 < 25
  t848 : 25 ≤ 25
  h618_0 : MxxRuntime.sliceMatrix (generatedRoot.w_617_0 ctx) 0 1 0 25 (generatedRoot.w_618_0 ctx)
  t850 : 0 ≤ 0
  t851 : 0 < 1
  t852 : 1 ≤ 3
  t853 : 0 ≤ 0
  t854 : 0 < 25
  t855 : 25 ≤ 25
  h619_0 : MxxRuntime.sliceMatrix (generatedRoot.w_611_0 ctx) 0 1 0 25 (generatedRoot.w_619_0 ctx)
  h622_0 : MxxRuntime.centeredRebaseRuns (generatedRoot.w_55_0 ctx) (generatedRoot.w_622_0 ctx)
  h625_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_624_0 ctx) (generatedRoot.w_625_0 ctx)
  h626_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_625_0 ctx) (generatedRoot.w_626_0 ctx)
  h628_0 : MxxRuntime.centeredRebaseRuns (generatedRoot.w_72_0 ctx) (generatedRoot.w_628_0 ctx)
  h631_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_630_0 ctx) (generatedRoot.w_631_0 ctx)
  h632_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_631_0 ctx) (generatedRoot.w_632_0 ctx)
  h636_0 : MxxRuntime.familyPack (26) [(generatedRoot.w_78_0 ctx), (generatedRoot.w_79_0 ctx), (generatedRoot.w_86_0 ctx), (generatedRoot.w_89_0 ctx), (generatedRoot.w_200_0 ctx), (generatedRoot.w_215_0 ctx), (generatedRoot.w_227_0 ctx), (generatedRoot.w_228_0 ctx), (generatedRoot.w_285_0 ctx), (generatedRoot.w_286_0 ctx), (generatedRoot.w_343_0 ctx), (generatedRoot.w_344_0 ctx), (generatedRoot.w_401_0 ctx), (generatedRoot.w_402_0 ctx), (generatedRoot.w_459_0 ctx), (generatedRoot.w_460_0 ctx), (generatedRoot.w_517_0 ctx), (generatedRoot.w_518_0 ctx), (generatedRoot.w_575_0 ctx), (generatedRoot.w_576_0 ctx), (generatedRoot.w_633_0 ctx), (generatedRoot.w_634_0 ctx), (generatedRoot.w_635_0 ctx), (generatedRoot.w_635_0 ctx), (generatedRoot.w_635_0 ctx), (generatedRoot.w_635_0 ctx)] (generatedRoot.w_636_0 ctx)

theorem generatedRoot.facts_26  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_26 backend (generatedRoot.w_55_0 ctx) (generatedRoot.w_72_0 ctx) (generatedRoot.w_78_0 ctx) (generatedRoot.w_79_0 ctx) (generatedRoot.w_86_0 ctx) (generatedRoot.w_89_0 ctx) (generatedRoot.w_200_0 ctx) (generatedRoot.w_215_0 ctx) (generatedRoot.w_227_0 ctx) (generatedRoot.w_228_0 ctx) (generatedRoot.w_285_0 ctx) (generatedRoot.w_286_0 ctx) (generatedRoot.w_343_0 ctx) (generatedRoot.w_344_0 ctx) (generatedRoot.w_401_0 ctx) (generatedRoot.w_402_0 ctx) (generatedRoot.w_459_0 ctx) (generatedRoot.w_460_0 ctx) (generatedRoot.w_517_0 ctx) (generatedRoot.w_518_0 ctx) (generatedRoot.w_575_0 ctx) (generatedRoot.w_576_0 ctx) (generatedRoot.w_611_0 ctx) (generatedRoot.w_612_0 ctx) (generatedRoot.w_613_0 ctx) (generatedRoot.w_614_0 ctx) (generatedRoot.w_615_0 ctx) (generatedRoot.w_616_0 ctx) (generatedRoot.w_617_0 ctx) (generatedRoot.w_618_0 ctx) (generatedRoot.w_619_0 ctx) (generatedRoot.w_622_0 ctx) (generatedRoot.w_624_0 ctx) (generatedRoot.w_625_0 ctx) (generatedRoot.w_626_0 ctx) (generatedRoot.w_628_0 ctx) (generatedRoot.w_630_0 ctx) (generatedRoot.w_631_0 ctx) (generatedRoot.w_632_0 ctx) (generatedRoot.w_633_0 ctx) (generatedRoot.w_634_0 ctx) (generatedRoot.w_635_0 ctx) (generatedRoot.w_636_0 ctx)) :
    generatedRoot.Facts_26 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_27  (ctx : generatedRootCtx) : Prop where
  h637_1 : MxxRuntime.sha256BitColumns [17, 229, 242, 240, 20, 113, 12, 254, 126, 237, 25, 125, 17, 124, 200, 38, 220, 135, 58, 4, 215, 86, 146, 138, 135, 103, 188, 97, 153, 28, 159, 14] 4 5 2 (generatedRoot.w_637_0 ctx) (generatedRoot.w_637_1 ctx)
  h639_0 : MxxRuntime.familyGetStatic (generatedRoot.w_638_0 ctx) (1) (generatedRoot.w_639_0 ctx)
  h642_0 : MxxRuntime.familyGetStatic (generatedRoot.w_641_0 ctx) (1) (generatedRoot.w_642_0 ctx)
  h644_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((generatedRoot.w_5_0 ctx)) (generatedRoot.w_644_0 ctx)
  h645_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((generatedRoot.w_5_0 ctx)) (generatedRoot.w_645_0 ctx)
  h646_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) ((generatedRoot.w_5_0 ctx)) (generatedRoot.w_646_0 ctx)
  h647_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) ((generatedRoot.w_5_0 ctx)) (generatedRoot.w_647_0 ctx)
  h650_0 : MxxRuntime.gadgetMatrixRuns backend 4096 25 (generatedRoot.w_650_0 ctx)
  h655_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) ((generatedRoot.w_5_0 ctx)) (generatedRoot.w_655_0 ctx)
  t873 : (5) = 5
  h658_5 : (∀ i : Fin 5, parallel_generatedRoot_658 backend hashModel tape (path ++ [658, i.val]) params i ((generatedRoot.w_637_1 ctx), ((generatedRoot.w_639_0 ctx), ((generatedRoot.w_640_0 ctx), ((generatedRoot.w_637_0 ctx), ((generatedRoot.w_642_0 ctx), ((generatedRoot.w_643_0 ctx), ((generatedRoot.w_644_0 ctx), ((generatedRoot.w_5_0 ctx), ((generatedRoot.w_645_0 ctx), ((generatedRoot.w_646_0 ctx), ((generatedRoot.w_647_0 ctx), ((generatedRoot.w_648_0 ctx), ((generatedRoot.w_649_0 ctx), ((generatedRoot.w_650_0 ctx), ((generatedRoot.w_651_0 ctx), ((generatedRoot.w_653_0 ctx), ((generatedRoot.w_86_0 ctx), ((generatedRoot.w_654_0 ctx), ((generatedRoot.w_657_0 ctx), ((generatedRoot.w_89_0 ctx), ())))))))))))))))))))) (((generatedRoot.w_658_0 ctx) i), (((generatedRoot.w_658_1 ctx) i), (((generatedRoot.w_658_2 ctx) i), (((generatedRoot.w_658_3 ctx) i), (((generatedRoot.w_658_4 ctx) i), (((generatedRoot.w_658_5 ctx) i), ())))))))
  t875 : 0 ≤ 1
  t876 : 1 < 2
  t877 : 2 ≤ 3
  t878 : 0 ≤ 0
  t879 : 0 < 25
  t880 : 25 ≤ 25
  h661_0 : MxxRuntime.sliceMatrix (generatedRoot.w_190_0 ctx) 1 2 0 25 (generatedRoot.w_661_0 ctx)
  t882 : 0 ≤ 1
  t883 : 1 < 2
  t884 : 2 ≤ 3
  t885 : 0 ≤ 0
  t886 : 0 < 25
  t887 : 25 ≤ 25
  h664_0 : MxxRuntime.sliceMatrix (generatedRoot.w_210_0 ctx) 1 2 0 25 (generatedRoot.w_664_0 ctx)
  t889 : 0 ≤ 1
  t890 : 1 < 2
  t891 : 2 ≤ 3
  t892 : 0 ≤ 0
  t893 : 0 < 25
  t894 : 25 ≤ 25
  h670_0 : MxxRuntime.sliceMatrix (generatedRoot.w_269_0 ctx) 1 2 0 25 (generatedRoot.w_670_0 ctx)

theorem generatedRoot.facts_27  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_27 backend hashModel tape path params (generatedRoot.w_5_0 ctx) (generatedRoot.w_638_0 ctx) (generatedRoot.w_640_0 ctx) (generatedRoot.w_641_0 ctx) (generatedRoot.w_643_0 ctx) (generatedRoot.w_649_0 ctx) (generatedRoot.w_86_0 ctx) (generatedRoot.w_89_0 ctx) (generatedRoot.w_190_0 ctx) (generatedRoot.w_210_0 ctx) (generatedRoot.w_269_0 ctx) (generatedRoot.w_637_0 ctx) (generatedRoot.w_637_1 ctx) (generatedRoot.w_639_0 ctx) (generatedRoot.w_642_0 ctx) (generatedRoot.w_644_0 ctx) (generatedRoot.w_645_0 ctx) (generatedRoot.w_646_0 ctx) (generatedRoot.w_647_0 ctx) (generatedRoot.w_648_0 ctx) (generatedRoot.w_650_0 ctx) (generatedRoot.w_651_0 ctx) (generatedRoot.w_653_0 ctx) (generatedRoot.w_654_0 ctx) (generatedRoot.w_655_0 ctx) (generatedRoot.w_657_0 ctx) (generatedRoot.w_658_0 ctx) (generatedRoot.w_658_1 ctx) (generatedRoot.w_658_2 ctx) (generatedRoot.w_658_3 ctx) (generatedRoot.w_658_4 ctx) (generatedRoot.w_658_5 ctx) (generatedRoot.w_661_0 ctx) (generatedRoot.w_664_0 ctx) (generatedRoot.w_670_0 ctx)) :
    generatedRoot.Facts_27 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_28  (ctx : generatedRootCtx) : Prop where
  t896 : 0 ≤ 1
  t897 : 1 < 2
  t898 : 2 ≤ 3
  t899 : 0 ≤ 0
  t900 : 0 < 25
  t901 : 25 ≤ 25
  h675_0 : MxxRuntime.sliceMatrix (generatedRoot.w_327_0 ctx) 1 2 0 25 (generatedRoot.w_675_0 ctx)
  t903 : 0 ≤ 1
  t904 : 1 < 2
  t905 : 2 ≤ 3
  t906 : 0 ≤ 0
  t907 : 0 < 25
  t908 : 25 ≤ 25
  h680_0 : MxxRuntime.sliceMatrix (generatedRoot.w_385_0 ctx) 1 2 0 25 (generatedRoot.w_680_0 ctx)
  t910 : 0 ≤ 1
  t911 : 1 < 2
  t912 : 2 ≤ 3
  t913 : 0 ≤ 0
  t914 : 0 < 25
  t915 : 25 ≤ 25
  h685_0 : MxxRuntime.sliceMatrix (generatedRoot.w_443_0 ctx) 1 2 0 25 (generatedRoot.w_685_0 ctx)
  t917 : 0 ≤ 1
  t918 : 1 < 2
  t919 : 2 ≤ 3
  t920 : 0 ≤ 0
  t921 : 0 < 25
  t922 : 25 ≤ 25
  h690_0 : MxxRuntime.sliceMatrix (generatedRoot.w_501_0 ctx) 1 2 0 25 (generatedRoot.w_690_0 ctx)
  t924 : 0 ≤ 1
  t925 : 1 < 2
  t926 : 2 ≤ 3
  t927 : 0 ≤ 0

theorem generatedRoot.facts_28  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_28 (generatedRoot.w_327_0 ctx) (generatedRoot.w_385_0 ctx) (generatedRoot.w_443_0 ctx) (generatedRoot.w_501_0 ctx) (generatedRoot.w_675_0 ctx) (generatedRoot.w_680_0 ctx) (generatedRoot.w_685_0 ctx) (generatedRoot.w_690_0 ctx)) :
    generatedRoot.Facts_28 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_29  (ctx : generatedRootCtx) : Prop where
  t928 : 0 < 25
  t929 : 25 ≤ 25
  h695_0 : MxxRuntime.sliceMatrix (generatedRoot.w_559_0 ctx) 1 2 0 25 (generatedRoot.w_695_0 ctx)
  t931 : 0 ≤ 1
  t932 : 1 < 2
  t933 : 2 ≤ 3
  t934 : 0 ≤ 0
  t935 : 0 < 25
  t936 : 25 ≤ 25
  h700_0 : MxxRuntime.sliceMatrix (generatedRoot.w_617_0 ctx) 1 2 0 25 (generatedRoot.w_700_0 ctx)
  h705_0 : MxxRuntime.familyPack (26) [(generatedRoot.w_659_0 ctx), (generatedRoot.w_660_0 ctx), (generatedRoot.w_651_0 ctx), (generatedRoot.w_654_0 ctx), (generatedRoot.w_663_0 ctx), (generatedRoot.w_666_0 ctx), (generatedRoot.w_668_0 ctx), (generatedRoot.w_669_0 ctx), (generatedRoot.w_673_0 ctx), (generatedRoot.w_674_0 ctx), (generatedRoot.w_678_0 ctx), (generatedRoot.w_679_0 ctx), (generatedRoot.w_683_0 ctx), (generatedRoot.w_684_0 ctx), (generatedRoot.w_688_0 ctx), (generatedRoot.w_689_0 ctx), (generatedRoot.w_693_0 ctx), (generatedRoot.w_694_0 ctx), (generatedRoot.w_698_0 ctx), (generatedRoot.w_699_0 ctx), (generatedRoot.w_703_0 ctx), (generatedRoot.w_704_0 ctx), (generatedRoot.w_635_0 ctx), (generatedRoot.w_635_0 ctx), (generatedRoot.w_635_0 ctx), (generatedRoot.w_635_0 ctx)] (generatedRoot.w_705_0 ctx)
  t939 : 0 ≤ 2
  t940 : 2 < 3
  t941 : 3 ≤ 3
  t942 : 0 ≤ 0
  t943 : 0 < 25
  t944 : 25 ≤ 25
  h708_0 : MxxRuntime.sliceMatrix (generatedRoot.w_190_0 ctx) 2 3 0 25 (generatedRoot.w_708_0 ctx)
  t946 : 0 ≤ 2
  t947 : 2 < 3
  t948 : 3 ≤ 3
  t949 : 0 ≤ 0
  t950 : 0 < 25
  t951 : 25 ≤ 25
  h711_0 : MxxRuntime.sliceMatrix (generatedRoot.w_210_0 ctx) 2 3 0 25 (generatedRoot.w_711_0 ctx)
  t953 : 0 ≤ 2
  t954 : 2 < 3
  t955 : 3 ≤ 3
  t956 : 0 ≤ 0
  t957 : 0 < 25
  t958 : 25 ≤ 25
  h717_0 : MxxRuntime.sliceMatrix (generatedRoot.w_269_0 ctx) 2 3 0 25 (generatedRoot.w_717_0 ctx)

theorem generatedRoot.facts_29  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_29 (generatedRoot.w_190_0 ctx) (generatedRoot.w_210_0 ctx) (generatedRoot.w_269_0 ctx) (generatedRoot.w_559_0 ctx) (generatedRoot.w_617_0 ctx) (generatedRoot.w_635_0 ctx) (generatedRoot.w_651_0 ctx) (generatedRoot.w_654_0 ctx) (generatedRoot.w_659_0 ctx) (generatedRoot.w_660_0 ctx) (generatedRoot.w_663_0 ctx) (generatedRoot.w_666_0 ctx) (generatedRoot.w_668_0 ctx) (generatedRoot.w_669_0 ctx) (generatedRoot.w_673_0 ctx) (generatedRoot.w_674_0 ctx) (generatedRoot.w_678_0 ctx) (generatedRoot.w_679_0 ctx) (generatedRoot.w_683_0 ctx) (generatedRoot.w_684_0 ctx) (generatedRoot.w_688_0 ctx) (generatedRoot.w_689_0 ctx) (generatedRoot.w_693_0 ctx) (generatedRoot.w_694_0 ctx) (generatedRoot.w_695_0 ctx) (generatedRoot.w_698_0 ctx) (generatedRoot.w_699_0 ctx) (generatedRoot.w_700_0 ctx) (generatedRoot.w_703_0 ctx) (generatedRoot.w_704_0 ctx) (generatedRoot.w_705_0 ctx) (generatedRoot.w_708_0 ctx) (generatedRoot.w_711_0 ctx) (generatedRoot.w_717_0 ctx)) :
    generatedRoot.Facts_29 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_30  (ctx : generatedRootCtx) : Prop where
  t960 : 0 ≤ 2
  t961 : 2 < 3
  t962 : 3 ≤ 3
  t963 : 0 ≤ 0
  t964 : 0 < 25
  t965 : 25 ≤ 25
  h722_0 : MxxRuntime.sliceMatrix (generatedRoot.w_327_0 ctx) 2 3 0 25 (generatedRoot.w_722_0 ctx)
  t967 : 0 ≤ 2
  t968 : 2 < 3
  t969 : 3 ≤ 3
  t970 : 0 ≤ 0
  t971 : 0 < 25
  t972 : 25 ≤ 25
  h727_0 : MxxRuntime.sliceMatrix (generatedRoot.w_385_0 ctx) 2 3 0 25 (generatedRoot.w_727_0 ctx)
  t974 : 0 ≤ 2
  t975 : 2 < 3
  t976 : 3 ≤ 3
  t977 : 0 ≤ 0
  t978 : 0 < 25
  t979 : 25 ≤ 25
  h732_0 : MxxRuntime.sliceMatrix (generatedRoot.w_443_0 ctx) 2 3 0 25 (generatedRoot.w_732_0 ctx)
  t981 : 0 ≤ 2
  t982 : 2 < 3
  t983 : 3 ≤ 3
  t984 : 0 ≤ 0
  t985 : 0 < 25
  t986 : 25 ≤ 25
  h737_0 : MxxRuntime.sliceMatrix (generatedRoot.w_501_0 ctx) 2 3 0 25 (generatedRoot.w_737_0 ctx)
  t988 : 0 ≤ 2
  t989 : 2 < 3
  t990 : 3 ≤ 3
  t991 : 0 ≤ 0

theorem generatedRoot.facts_30  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_30 (generatedRoot.w_327_0 ctx) (generatedRoot.w_385_0 ctx) (generatedRoot.w_443_0 ctx) (generatedRoot.w_501_0 ctx) (generatedRoot.w_722_0 ctx) (generatedRoot.w_727_0 ctx) (generatedRoot.w_732_0 ctx) (generatedRoot.w_737_0 ctx)) :
    generatedRoot.Facts_30 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_31  (ctx : generatedRootCtx) : Prop where
  t992 : 0 < 25
  t993 : 25 ≤ 25
  h742_0 : MxxRuntime.sliceMatrix (generatedRoot.w_559_0 ctx) 2 3 0 25 (generatedRoot.w_742_0 ctx)
  t995 : 0 ≤ 2
  t996 : 2 < 3
  t997 : 3 ≤ 3
  t998 : 0 ≤ 0
  t999 : 0 < 25
  t1000 : 25 ≤ 25
  h747_0 : MxxRuntime.sliceMatrix (generatedRoot.w_617_0 ctx) 2 3 0 25 (generatedRoot.w_747_0 ctx)
  h752_0 : MxxRuntime.familyPack (26) [(generatedRoot.w_706_0 ctx), (generatedRoot.w_707_0 ctx), (generatedRoot.w_653_0 ctx), (generatedRoot.w_657_0 ctx), (generatedRoot.w_710_0 ctx), (generatedRoot.w_713_0 ctx), (generatedRoot.w_715_0 ctx), (generatedRoot.w_716_0 ctx), (generatedRoot.w_720_0 ctx), (generatedRoot.w_721_0 ctx), (generatedRoot.w_725_0 ctx), (generatedRoot.w_726_0 ctx), (generatedRoot.w_730_0 ctx), (generatedRoot.w_731_0 ctx), (generatedRoot.w_735_0 ctx), (generatedRoot.w_736_0 ctx), (generatedRoot.w_740_0 ctx), (generatedRoot.w_741_0 ctx), (generatedRoot.w_745_0 ctx), (generatedRoot.w_746_0 ctx), (generatedRoot.w_750_0 ctx), (generatedRoot.w_751_0 ctx), (generatedRoot.w_635_0 ctx), (generatedRoot.w_635_0 ctx), (generatedRoot.w_635_0 ctx), (generatedRoot.w_635_0 ctx)] (generatedRoot.w_752_0 ctx)
  h754_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((generatedRoot.w_753_0 ctx)) (generatedRoot.w_754_0 ctx)
  h755_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) ((generatedRoot.w_753_0 ctx)) (generatedRoot.w_755_0 ctx)
  h756_0 : MxxRuntime.concatRows (generatedRoot.w_754_0 ctx) (generatedRoot.w_755_0 ctx) (generatedRoot.w_756_0 ctx)
  h757_0 : MxxRuntime.modulusReduceRuns (generatedRoot.w_756_0 ctx) (generatedRoot.w_757_0 ctx)
  h758_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((generatedRoot.w_753_0 ctx)) (generatedRoot.w_758_0 ctx)
  h759_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) ((generatedRoot.w_753_0 ctx)) (generatedRoot.w_759_0 ctx)
  h760_0 : MxxRuntime.concatRows (generatedRoot.w_758_0 ctx) (generatedRoot.w_759_0 ctx) (generatedRoot.w_760_0 ctx)
  h761_0 : MxxRuntime.modulusReduceRuns (generatedRoot.w_760_0 ctx) (generatedRoot.w_761_0 ctx)
  h762_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((generatedRoot.w_753_0 ctx)) (generatedRoot.w_762_0 ctx)
  h763_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) ((generatedRoot.w_753_0 ctx)) (generatedRoot.w_763_0 ctx)
  h764_0 : MxxRuntime.concatRows (generatedRoot.w_762_0 ctx) (generatedRoot.w_763_0 ctx) (generatedRoot.w_764_0 ctx)
  h765_0 : MxxRuntime.modulusReduceRuns (generatedRoot.w_764_0 ctx) (generatedRoot.w_765_0 ctx)
  h766_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((generatedRoot.w_753_0 ctx)) (generatedRoot.w_766_0 ctx)
  h767_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) ((generatedRoot.w_753_0 ctx)) (generatedRoot.w_767_0 ctx)
  h768_0 : MxxRuntime.concatRows (generatedRoot.w_766_0 ctx) (generatedRoot.w_767_0 ctx) (generatedRoot.w_768_0 ctx)
  h769_0 : MxxRuntime.modulusReduceRuns (generatedRoot.w_768_0 ctx) (generatedRoot.w_769_0 ctx)
  h770_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 115, 116, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((generatedRoot.w_753_0 ctx)) (generatedRoot.w_770_0 ctx)
  h771_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 115, 116, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) ((generatedRoot.w_753_0 ctx)) (generatedRoot.w_771_0 ctx)
  h772_0 : MxxRuntime.concatRows (generatedRoot.w_770_0 ctx) (generatedRoot.w_771_0 ctx) (generatedRoot.w_772_0 ctx)
  h773_0 : MxxRuntime.modulusReduceRuns (generatedRoot.w_772_0 ctx) (generatedRoot.w_773_0 ctx)
  h774_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 115, 116, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((generatedRoot.w_753_0 ctx)) (generatedRoot.w_774_0 ctx)

theorem generatedRoot.facts_31  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_31 hashModel (generatedRoot.w_753_0 ctx) (generatedRoot.w_559_0 ctx) (generatedRoot.w_617_0 ctx) (generatedRoot.w_635_0 ctx) (generatedRoot.w_653_0 ctx) (generatedRoot.w_657_0 ctx) (generatedRoot.w_706_0 ctx) (generatedRoot.w_707_0 ctx) (generatedRoot.w_710_0 ctx) (generatedRoot.w_713_0 ctx) (generatedRoot.w_715_0 ctx) (generatedRoot.w_716_0 ctx) (generatedRoot.w_720_0 ctx) (generatedRoot.w_721_0 ctx) (generatedRoot.w_725_0 ctx) (generatedRoot.w_726_0 ctx) (generatedRoot.w_730_0 ctx) (generatedRoot.w_731_0 ctx) (generatedRoot.w_735_0 ctx) (generatedRoot.w_736_0 ctx) (generatedRoot.w_740_0 ctx) (generatedRoot.w_741_0 ctx) (generatedRoot.w_742_0 ctx) (generatedRoot.w_745_0 ctx) (generatedRoot.w_746_0 ctx) (generatedRoot.w_747_0 ctx) (generatedRoot.w_750_0 ctx) (generatedRoot.w_751_0 ctx) (generatedRoot.w_752_0 ctx) (generatedRoot.w_754_0 ctx) (generatedRoot.w_755_0 ctx) (generatedRoot.w_756_0 ctx) (generatedRoot.w_757_0 ctx) (generatedRoot.w_758_0 ctx) (generatedRoot.w_759_0 ctx) (generatedRoot.w_760_0 ctx) (generatedRoot.w_761_0 ctx) (generatedRoot.w_762_0 ctx) (generatedRoot.w_763_0 ctx) (generatedRoot.w_764_0 ctx) (generatedRoot.w_765_0 ctx) (generatedRoot.w_766_0 ctx) (generatedRoot.w_767_0 ctx) (generatedRoot.w_768_0 ctx) (generatedRoot.w_769_0 ctx) (generatedRoot.w_770_0 ctx) (generatedRoot.w_771_0 ctx) (generatedRoot.w_772_0 ctx) (generatedRoot.w_773_0 ctx) (generatedRoot.w_774_0 ctx)) :
    generatedRoot.Facts_31 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_32  (ctx : generatedRootCtx) : Prop where
  h775_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 115, 116, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) ((generatedRoot.w_753_0 ctx)) (generatedRoot.w_775_0 ctx)
  h776_0 : MxxRuntime.concatRows (generatedRoot.w_774_0 ctx) (generatedRoot.w_775_0 ctx) (generatedRoot.w_776_0 ctx)
  h777_0 : MxxRuntime.modulusReduceRuns (generatedRoot.w_776_0 ctx) (generatedRoot.w_777_0 ctx)
  h779_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_778_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((generatedRoot.w_753_0 ctx)) (generatedRoot.w_779_0 ctx)
  h780_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_778_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) ((generatedRoot.w_753_0 ctx)) (generatedRoot.w_780_0 ctx)
  h781_0 : MxxRuntime.concatRows (generatedRoot.w_779_0 ctx) (generatedRoot.w_780_0 ctx) (generatedRoot.w_781_0 ctx)
  h782_0 : MxxRuntime.modulusReduceRuns (generatedRoot.w_781_0 ctx) (generatedRoot.w_782_0 ctx)
  h784_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_783_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((generatedRoot.w_753_0 ctx)) (generatedRoot.w_784_0 ctx)
  h785_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_783_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) ((generatedRoot.w_753_0 ctx)) (generatedRoot.w_785_0 ctx)
  h786_0 : MxxRuntime.concatRows (generatedRoot.w_784_0 ctx) (generatedRoot.w_785_0 ctx) (generatedRoot.w_786_0 ctx)
  h787_0 : MxxRuntime.modulusReduceRuns (generatedRoot.w_786_0 ctx) (generatedRoot.w_787_0 ctx)
  h789_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_788_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((generatedRoot.w_753_0 ctx)) (generatedRoot.w_789_0 ctx)
  h790_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_788_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) ((generatedRoot.w_753_0 ctx)) (generatedRoot.w_790_0 ctx)
  h791_0 : MxxRuntime.concatRows (generatedRoot.w_789_0 ctx) (generatedRoot.w_790_0 ctx) (generatedRoot.w_791_0 ctx)
  h792_0 : MxxRuntime.modulusReduceRuns (generatedRoot.w_791_0 ctx) (generatedRoot.w_792_0 ctx)
  h794_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_793_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((generatedRoot.w_753_0 ctx)) (generatedRoot.w_794_0 ctx)
  h795_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_793_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) ((generatedRoot.w_753_0 ctx)) (generatedRoot.w_795_0 ctx)
  h796_0 : MxxRuntime.concatRows (generatedRoot.w_794_0 ctx) (generatedRoot.w_795_0 ctx) (generatedRoot.w_796_0 ctx)
  h797_0 : MxxRuntime.modulusReduceRuns (generatedRoot.w_796_0 ctx) (generatedRoot.w_797_0 ctx)
  h799_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_798_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((generatedRoot.w_753_0 ctx)) (generatedRoot.w_799_0 ctx)
  h800_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_798_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) ((generatedRoot.w_753_0 ctx)) (generatedRoot.w_800_0 ctx)
  h801_0 : MxxRuntime.concatRows (generatedRoot.w_799_0 ctx) (generatedRoot.w_800_0 ctx) (generatedRoot.w_801_0 ctx)
  h802_0 : MxxRuntime.modulusReduceRuns (generatedRoot.w_801_0 ctx) (generatedRoot.w_802_0 ctx)
  h804_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_803_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((generatedRoot.w_753_0 ctx)) (generatedRoot.w_804_0 ctx)
  h805_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_803_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) ((generatedRoot.w_753_0 ctx)) (generatedRoot.w_805_0 ctx)
  h806_0 : MxxRuntime.concatRows (generatedRoot.w_804_0 ctx) (generatedRoot.w_805_0 ctx) (generatedRoot.w_806_0 ctx)
  h807_0 : MxxRuntime.modulusReduceRuns (generatedRoot.w_806_0 ctx) (generatedRoot.w_807_0 ctx)
  h809_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_808_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((generatedRoot.w_753_0 ctx)) (generatedRoot.w_809_0 ctx)
  h810_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_808_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) ((generatedRoot.w_753_0 ctx)) (generatedRoot.w_810_0 ctx)
  h811_0 : MxxRuntime.concatRows (generatedRoot.w_809_0 ctx) (generatedRoot.w_810_0 ctx) (generatedRoot.w_811_0 ctx)
  h812_0 : MxxRuntime.modulusReduceRuns (generatedRoot.w_811_0 ctx) (generatedRoot.w_812_0 ctx)
  h814_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_813_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((generatedRoot.w_753_0 ctx)) (generatedRoot.w_814_0 ctx)

theorem generatedRoot.facts_32  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_32 hashModel (generatedRoot.w_753_0 ctx) (generatedRoot.w_774_0 ctx) (generatedRoot.w_775_0 ctx) (generatedRoot.w_776_0 ctx) (generatedRoot.w_777_0 ctx) (generatedRoot.w_778_0 ctx) (generatedRoot.w_779_0 ctx) (generatedRoot.w_780_0 ctx) (generatedRoot.w_781_0 ctx) (generatedRoot.w_782_0 ctx) (generatedRoot.w_783_0 ctx) (generatedRoot.w_784_0 ctx) (generatedRoot.w_785_0 ctx) (generatedRoot.w_786_0 ctx) (generatedRoot.w_787_0 ctx) (generatedRoot.w_788_0 ctx) (generatedRoot.w_789_0 ctx) (generatedRoot.w_790_0 ctx) (generatedRoot.w_791_0 ctx) (generatedRoot.w_792_0 ctx) (generatedRoot.w_793_0 ctx) (generatedRoot.w_794_0 ctx) (generatedRoot.w_795_0 ctx) (generatedRoot.w_796_0 ctx) (generatedRoot.w_797_0 ctx) (generatedRoot.w_798_0 ctx) (generatedRoot.w_799_0 ctx) (generatedRoot.w_800_0 ctx) (generatedRoot.w_801_0 ctx) (generatedRoot.w_802_0 ctx) (generatedRoot.w_803_0 ctx) (generatedRoot.w_804_0 ctx) (generatedRoot.w_805_0 ctx) (generatedRoot.w_806_0 ctx) (generatedRoot.w_807_0 ctx) (generatedRoot.w_808_0 ctx) (generatedRoot.w_809_0 ctx) (generatedRoot.w_810_0 ctx) (generatedRoot.w_811_0 ctx) (generatedRoot.w_812_0 ctx) (generatedRoot.w_813_0 ctx) (generatedRoot.w_814_0 ctx)) :
    generatedRoot.Facts_32 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_33  (ctx : generatedRootCtx) : Prop where
  h815_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_813_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) ((generatedRoot.w_753_0 ctx)) (generatedRoot.w_815_0 ctx)
  h816_0 : MxxRuntime.concatRows (generatedRoot.w_814_0 ctx) (generatedRoot.w_815_0 ctx) (generatedRoot.w_816_0 ctx)
  h817_0 : MxxRuntime.modulusReduceRuns (generatedRoot.w_816_0 ctx) (generatedRoot.w_817_0 ctx)
  h819_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_818_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((generatedRoot.w_753_0 ctx)) (generatedRoot.w_819_0 ctx)
  h820_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_818_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) ((generatedRoot.w_753_0 ctx)) (generatedRoot.w_820_0 ctx)
  h821_0 : MxxRuntime.concatRows (generatedRoot.w_819_0 ctx) (generatedRoot.w_820_0 ctx) (generatedRoot.w_821_0 ctx)
  h822_0 : MxxRuntime.modulusReduceRuns (generatedRoot.w_821_0 ctx) (generatedRoot.w_822_0 ctx)
  h824_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_823_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((generatedRoot.w_753_0 ctx)) (generatedRoot.w_824_0 ctx)
  h825_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_823_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) ((generatedRoot.w_753_0 ctx)) (generatedRoot.w_825_0 ctx)
  h826_0 : MxxRuntime.concatRows (generatedRoot.w_824_0 ctx) (generatedRoot.w_825_0 ctx) (generatedRoot.w_826_0 ctx)
  h827_0 : MxxRuntime.modulusReduceRuns (generatedRoot.w_826_0 ctx) (generatedRoot.w_827_0 ctx)
  h829_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_828_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((generatedRoot.w_753_0 ctx)) (generatedRoot.w_829_0 ctx)
  h830_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_828_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) ((generatedRoot.w_753_0 ctx)) (generatedRoot.w_830_0 ctx)
  h831_0 : MxxRuntime.concatRows (generatedRoot.w_829_0 ctx) (generatedRoot.w_830_0 ctx) (generatedRoot.w_831_0 ctx)
  h832_0 : MxxRuntime.modulusReduceRuns (generatedRoot.w_831_0 ctx) (generatedRoot.w_832_0 ctx)
  h834_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_833_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((generatedRoot.w_753_0 ctx)) (generatedRoot.w_834_0 ctx)
  h835_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_833_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) ((generatedRoot.w_753_0 ctx)) (generatedRoot.w_835_0 ctx)
  h836_0 : MxxRuntime.concatRows (generatedRoot.w_834_0 ctx) (generatedRoot.w_835_0 ctx) (generatedRoot.w_836_0 ctx)
  h837_0 : MxxRuntime.modulusReduceRuns (generatedRoot.w_836_0 ctx) (generatedRoot.w_837_0 ctx)
  h839_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_838_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((generatedRoot.w_753_0 ctx)) (generatedRoot.w_839_0 ctx)
  h840_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_838_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) ((generatedRoot.w_753_0 ctx)) (generatedRoot.w_840_0 ctx)
  h841_0 : MxxRuntime.concatRows (generatedRoot.w_839_0 ctx) (generatedRoot.w_840_0 ctx) (generatedRoot.w_841_0 ctx)
  h842_0 : MxxRuntime.modulusReduceRuns (generatedRoot.w_841_0 ctx) (generatedRoot.w_842_0 ctx)
  h844_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_843_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((generatedRoot.w_753_0 ctx)) (generatedRoot.w_844_0 ctx)
  h845_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_843_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) ((generatedRoot.w_753_0 ctx)) (generatedRoot.w_845_0 ctx)
  h846_0 : MxxRuntime.concatRows (generatedRoot.w_844_0 ctx) (generatedRoot.w_845_0 ctx) (generatedRoot.w_846_0 ctx)
  h847_0 : MxxRuntime.modulusReduceRuns (generatedRoot.w_846_0 ctx) (generatedRoot.w_847_0 ctx)
  h849_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_848_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((generatedRoot.w_753_0 ctx)) (generatedRoot.w_849_0 ctx)
  h850_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_848_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) ((generatedRoot.w_753_0 ctx)) (generatedRoot.w_850_0 ctx)
  h851_0 : MxxRuntime.concatRows (generatedRoot.w_849_0 ctx) (generatedRoot.w_850_0 ctx) (generatedRoot.w_851_0 ctx)
  h852_0 : MxxRuntime.modulusReduceRuns (generatedRoot.w_851_0 ctx) (generatedRoot.w_852_0 ctx)
  h854_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_853_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((generatedRoot.w_753_0 ctx)) (generatedRoot.w_854_0 ctx)

theorem generatedRoot.facts_33  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_33 hashModel (generatedRoot.w_753_0 ctx) (generatedRoot.w_813_0 ctx) (generatedRoot.w_814_0 ctx) (generatedRoot.w_815_0 ctx) (generatedRoot.w_816_0 ctx) (generatedRoot.w_817_0 ctx) (generatedRoot.w_818_0 ctx) (generatedRoot.w_819_0 ctx) (generatedRoot.w_820_0 ctx) (generatedRoot.w_821_0 ctx) (generatedRoot.w_822_0 ctx) (generatedRoot.w_823_0 ctx) (generatedRoot.w_824_0 ctx) (generatedRoot.w_825_0 ctx) (generatedRoot.w_826_0 ctx) (generatedRoot.w_827_0 ctx) (generatedRoot.w_828_0 ctx) (generatedRoot.w_829_0 ctx) (generatedRoot.w_830_0 ctx) (generatedRoot.w_831_0 ctx) (generatedRoot.w_832_0 ctx) (generatedRoot.w_833_0 ctx) (generatedRoot.w_834_0 ctx) (generatedRoot.w_835_0 ctx) (generatedRoot.w_836_0 ctx) (generatedRoot.w_837_0 ctx) (generatedRoot.w_838_0 ctx) (generatedRoot.w_839_0 ctx) (generatedRoot.w_840_0 ctx) (generatedRoot.w_841_0 ctx) (generatedRoot.w_842_0 ctx) (generatedRoot.w_843_0 ctx) (generatedRoot.w_844_0 ctx) (generatedRoot.w_845_0 ctx) (generatedRoot.w_846_0 ctx) (generatedRoot.w_847_0 ctx) (generatedRoot.w_848_0 ctx) (generatedRoot.w_849_0 ctx) (generatedRoot.w_850_0 ctx) (generatedRoot.w_851_0 ctx) (generatedRoot.w_852_0 ctx) (generatedRoot.w_853_0 ctx) (generatedRoot.w_854_0 ctx)) :
    generatedRoot.Facts_33 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_34  (ctx : generatedRootCtx) : Prop where
  h855_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer (generatedRoot.w_853_0 ctx), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) ((generatedRoot.w_753_0 ctx)) (generatedRoot.w_855_0 ctx)
  h856_0 : MxxRuntime.concatRows (generatedRoot.w_854_0 ctx) (generatedRoot.w_855_0 ctx) (generatedRoot.w_856_0 ctx)
  h857_0 : MxxRuntime.modulusReduceRuns (generatedRoot.w_856_0 ctx) (generatedRoot.w_857_0 ctx)
  h859_0 : MxxRuntime.hashSample (hashModel) ([100, 105, 111, 50, 47, 97, 110, 99, 104, 111, 114, 45, 112, 97, 100, 100, 105, 110, 103]) ([]) ((generatedRoot.w_753_0 ctx)) (generatedRoot.w_859_0 ctx)
  h860_0 : MxxRuntime.concatColumns (generatedRoot.w_858_0 ctx) (generatedRoot.w_859_0 ctx) (generatedRoot.w_860_0 ctx)
  t1093 : 0 ≤ 0
  t1094 : 0 < 2
  t1095 : 2 ≤ 2
  t1096 : 0 ≤ 0
  t1097 : 0 < 25
  t1098 : 25 ≤ 100
  h861_0 : MxxRuntime.sliceMatrix (generatedRoot.w_860_0 ctx) 0 2 0 25 (generatedRoot.w_861_0 ctx)
  h862_0 : MxxRuntime.modulusReduceRuns (generatedRoot.w_861_0 ctx) (generatedRoot.w_862_0 ctx)
  t1101 : 0 ≤ 0
  t1102 : 0 < 2
  t1103 : 2 ≤ 2
  t1104 : 0 ≤ 25
  t1105 : 25 < 50
  t1106 : 50 ≤ 100
  h863_0 : MxxRuntime.sliceMatrix (generatedRoot.w_860_0 ctx) 0 2 25 50 (generatedRoot.w_863_0 ctx)
  h864_0 : MxxRuntime.modulusReduceRuns (generatedRoot.w_863_0 ctx) (generatedRoot.w_864_0 ctx)
  t1109 : 0 ≤ 0
  t1110 : 0 < 2
  t1111 : 2 ≤ 2
  t1112 : 0 ≤ 50
  t1113 : 50 < 75
  t1114 : 75 ≤ 100
  h865_0 : MxxRuntime.sliceMatrix (generatedRoot.w_860_0 ctx) 0 2 50 75 (generatedRoot.w_865_0 ctx)
  h866_0 : MxxRuntime.modulusReduceRuns (generatedRoot.w_865_0 ctx) (generatedRoot.w_866_0 ctx)
  t1117 : 0 ≤ 0
  t1118 : 0 < 2
  t1119 : 2 ≤ 2

theorem generatedRoot.facts_34  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_34 hashModel (generatedRoot.w_753_0 ctx) (generatedRoot.w_858_0 ctx) (generatedRoot.w_853_0 ctx) (generatedRoot.w_854_0 ctx) (generatedRoot.w_855_0 ctx) (generatedRoot.w_856_0 ctx) (generatedRoot.w_857_0 ctx) (generatedRoot.w_859_0 ctx) (generatedRoot.w_860_0 ctx) (generatedRoot.w_861_0 ctx) (generatedRoot.w_862_0 ctx) (generatedRoot.w_863_0 ctx) (generatedRoot.w_864_0 ctx) (generatedRoot.w_865_0 ctx) (generatedRoot.w_866_0 ctx)) :
    generatedRoot.Facts_34 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_35  (ctx : generatedRootCtx) : Prop where
  t1120 : 0 ≤ 75
  t1121 : 75 < 100
  t1122 : 100 ≤ 100
  h867_0 : MxxRuntime.sliceMatrix (generatedRoot.w_860_0 ctx) 0 2 75 100 (generatedRoot.w_867_0 ctx)
  h868_0 : MxxRuntime.modulusReduceRuns (generatedRoot.w_867_0 ctx) (generatedRoot.w_868_0 ctx)
  h869_0 : MxxRuntime.familyPack (26) [(generatedRoot.w_757_0 ctx), (generatedRoot.w_761_0 ctx), (generatedRoot.w_765_0 ctx), (generatedRoot.w_769_0 ctx), (generatedRoot.w_773_0 ctx), (generatedRoot.w_777_0 ctx), (generatedRoot.w_782_0 ctx), (generatedRoot.w_787_0 ctx), (generatedRoot.w_792_0 ctx), (generatedRoot.w_797_0 ctx), (generatedRoot.w_802_0 ctx), (generatedRoot.w_807_0 ctx), (generatedRoot.w_812_0 ctx), (generatedRoot.w_817_0 ctx), (generatedRoot.w_822_0 ctx), (generatedRoot.w_827_0 ctx), (generatedRoot.w_832_0 ctx), (generatedRoot.w_837_0 ctx), (generatedRoot.w_842_0 ctx), (generatedRoot.w_847_0 ctx), (generatedRoot.w_852_0 ctx), (generatedRoot.w_857_0 ctx), (generatedRoot.w_862_0 ctx), (generatedRoot.w_864_0 ctx), (generatedRoot.w_866_0 ctx), (generatedRoot.w_868_0 ctx)] (generatedRoot.w_869_0 ctx)
  t1126 : (36) = 36
  h870_3 : (∀ i : Fin 36, parallel_generatedRoot_870 backend hashModel tape (path ++ [870, i.val]) params i ((generatedRoot.w_636_0 ctx), ((generatedRoot.w_658_0 ctx), ((generatedRoot.w_658_1 ctx), ((generatedRoot.w_705_0 ctx), ((generatedRoot.w_658_2 ctx), ((generatedRoot.w_658_3 ctx), ((generatedRoot.w_752_0 ctx), ((generatedRoot.w_658_4 ctx), ((generatedRoot.w_658_5 ctx), ((generatedRoot.w_869_0 ctx), ((generatedRoot.w_753_0 ctx), ()))))))))))) (((generatedRoot.w_870_0 ctx) i), (((generatedRoot.w_870_1 ctx) i), (((generatedRoot.w_870_2 ctx) i), (((generatedRoot.w_870_3 ctx) i), ())))))
  t1128 : 0 ≤ (0)
  t1129 : (0) < 900
  h880_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((generatedRoot.w_5_0 ctx)) (generatedRoot.w_880_0 ctx)
  h881_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) ((generatedRoot.w_5_0 ctx)) (generatedRoot.w_881_0 ctx)
  h884_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((generatedRoot.w_5_0 ctx)) (generatedRoot.w_884_0 ctx)
  h885_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) ((generatedRoot.w_5_0 ctx)) (generatedRoot.w_885_0 ctx)
  t1134 : (14400) = 14400
  h889_5 : (∀ i : Fin 14400, ∃ w_877_0_gathered : mxxMat0, ∃ w_878_0_gathered : mxxMat0, ∃ w_883_0_gathered : mxxMat5, 0 ≤ ((generatedRoot.w_879_0 ctx) i) ∧ ((generatedRoot.w_879_0 ctx) i) < 2 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_877_0 ctx) ((generatedRoot.w_879_0 ctx) i) w_877_0_gathered ∧ 0 ≤ ((generatedRoot.w_879_0 ctx) i) ∧ ((generatedRoot.w_879_0 ctx) i) < 2 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_878_0 ctx) ((generatedRoot.w_879_0 ctx) i) w_878_0_gathered ∧ 0 ≤ ((generatedRoot.w_879_0 ctx) i) ∧ ((generatedRoot.w_879_0 ctx) i) < 2 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_883_0 ctx) ((generatedRoot.w_879_0 ctx) i) w_883_0_gathered ∧ parallel_generatedRoot_889 backend hashModel tape (path ++ [889, i.val]) params i (((generatedRoot.w_879_0 ctx) i), (w_877_0_gathered, ((generatedRoot.w_5_0 ctx), (w_878_0_gathered, ((generatedRoot.w_880_0 ctx), ((generatedRoot.w_881_0 ctx), ((generatedRoot.w_882_0 ctx), (w_883_0_gathered, ((generatedRoot.w_884_0 ctx), ((generatedRoot.w_885_0 ctx), ((generatedRoot.w_886_0 ctx), ((generatedRoot.w_24_0 ctx), (((generatedRoot.w_887_0 ctx) i), (((generatedRoot.w_888_0 ctx) i), ())))))))))))))) (((generatedRoot.w_889_0 ctx) i), (((generatedRoot.w_889_1 ctx) i), (((generatedRoot.w_889_2 ctx) i), (((generatedRoot.w_889_3 ctx) i), (((generatedRoot.w_889_4 ctx) i), (((generatedRoot.w_889_5 ctx) i), ())))))))
  t1136 : (7200) = 7200
  h890_5 : (∀ i : Fin 7200, parallel_generatedRoot_890 backend hashModel tape (path ++ [890, i.val]) params i ((generatedRoot.w_889_0 ctx), ((generatedRoot.w_889_1 ctx), ((generatedRoot.w_889_2 ctx), ((generatedRoot.w_889_3 ctx), ((generatedRoot.w_889_4 ctx), ((generatedRoot.w_889_5 ctx), ())))))) (((generatedRoot.w_890_0 ctx) i), (((generatedRoot.w_890_1 ctx) i), (((generatedRoot.w_890_2 ctx) i), (((generatedRoot.w_890_3 ctx) i), (((generatedRoot.w_890_4 ctx) i), (((generatedRoot.w_890_5 ctx) i), ())))))))
  t1138 : (14400) = 14400
  h893_5 : (∀ i : Fin 14400, ∃ w_877_0_gathered : mxxMat0, ∃ w_878_0_gathered : mxxMat0, ∃ w_883_0_gathered : mxxMat5, 0 ≤ ((generatedRoot.w_876_0 ctx) i) ∧ ((generatedRoot.w_876_0 ctx) i) < 2 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_877_0 ctx) ((generatedRoot.w_876_0 ctx) i) w_877_0_gathered ∧ 0 ≤ ((generatedRoot.w_876_0 ctx) i) ∧ ((generatedRoot.w_876_0 ctx) i) < 2 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_878_0 ctx) ((generatedRoot.w_876_0 ctx) i) w_878_0_gathered ∧ 0 ≤ ((generatedRoot.w_876_0 ctx) i) ∧ ((generatedRoot.w_876_0 ctx) i) < 2 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_883_0 ctx) ((generatedRoot.w_876_0 ctx) i) w_883_0_gathered ∧ parallel_generatedRoot_893 backend hashModel tape (path ++ [893, i.val]) params i (((generatedRoot.w_876_0 ctx) i), (w_877_0_gathered, ((generatedRoot.w_5_0 ctx), (w_878_0_gathered, ((generatedRoot.w_890_0 ctx), ((generatedRoot.w_890_2 ctx), ((generatedRoot.w_890_4 ctx), (w_883_0_gathered, ((generatedRoot.w_890_1 ctx), ((generatedRoot.w_890_3 ctx), ((generatedRoot.w_890_5 ctx), ((generatedRoot.w_24_0 ctx), (((generatedRoot.w_891_0 ctx) i), (((generatedRoot.w_892_0 ctx) i), ())))))))))))))) (((generatedRoot.w_893_0 ctx) i), (((generatedRoot.w_893_1 ctx) i), (((generatedRoot.w_893_2 ctx) i), (((generatedRoot.w_893_3 ctx) i), (((generatedRoot.w_893_4 ctx) i), (((generatedRoot.w_893_5 ctx) i), ())))))))
  t1140 : (7200) = 7200
  h894_5 : (∀ i : Fin 7200, parallel_generatedRoot_894 backend hashModel tape (path ++ [894, i.val]) params i ((generatedRoot.w_893_0 ctx), ((generatedRoot.w_893_1 ctx), ((generatedRoot.w_893_2 ctx), ((generatedRoot.w_893_3 ctx), ((generatedRoot.w_893_4 ctx), ((generatedRoot.w_893_5 ctx), ())))))) (((generatedRoot.w_894_0 ctx) i), (((generatedRoot.w_894_1 ctx) i), (((generatedRoot.w_894_2 ctx) i), (((generatedRoot.w_894_3 ctx) i), (((generatedRoot.w_894_4 ctx) i), (((generatedRoot.w_894_5 ctx) i), ())))))))
  t1142 : (7200) = 7200
  h895_5 : (∀ i : Fin 7200, parallel_generatedRoot_895 backend hashModel tape (path ++ [895, i.val]) params i ((generatedRoot.w_872_0 ctx), ((generatedRoot.w_873_0 ctx), ((generatedRoot.w_874_0 ctx), ((generatedRoot.w_875_0 ctx), ((generatedRoot.w_11_0 ctx), ((generatedRoot.w_5_0 ctx), ((generatedRoot.w_12_0 ctx), (((generatedRoot.w_894_4 ctx) i), (((generatedRoot.w_894_0 ctx) i), (((generatedRoot.w_894_2 ctx) i), (((generatedRoot.w_894_5 ctx) i), (((generatedRoot.w_894_1 ctx) i), (((generatedRoot.w_894_3 ctx) i), ()))))))))))))) (((generatedRoot.w_895_0 ctx) i), (((generatedRoot.w_895_1 ctx) i), (((generatedRoot.w_895_2 ctx) i), (((generatedRoot.w_895_3 ctx) i), (((generatedRoot.w_895_4 ctx) i), (((generatedRoot.w_895_5 ctx) i), ())))))))
  t1144 : (7200) = 7200
  h896_5 : (∀ i : Fin 7200, parallel_generatedRoot_896 backend hashModel tape (path ++ [896, i.val]) params i ((generatedRoot.w_895_0 ctx), ((generatedRoot.w_895_1 ctx), ((generatedRoot.w_895_2 ctx), ((generatedRoot.w_895_3 ctx), ((generatedRoot.w_895_4 ctx), ((generatedRoot.w_895_5 ctx), ())))))) (((generatedRoot.w_896_0 ctx) i), (((generatedRoot.w_896_1 ctx) i), (((generatedRoot.w_896_2 ctx) i), (((generatedRoot.w_896_3 ctx) i), (((generatedRoot.w_896_4 ctx) i), (((generatedRoot.w_896_5 ctx) i), ())))))))
  h897_0 : MxxRuntime.familyGetStatic (generatedRoot.w_896_0 ctx) (0) (generatedRoot.w_897_0 ctx)
  h898_0 : MxxRuntime.familyGetStatic (generatedRoot.w_896_0 ctx) (1) (generatedRoot.w_898_0 ctx)
  h900_0 : MxxRuntime.familyGetStatic (generatedRoot.w_896_1 ctx) (0) (generatedRoot.w_900_0 ctx)
  h901_0 : MxxRuntime.familyGetStatic (generatedRoot.w_896_1 ctx) (1) (generatedRoot.w_901_0 ctx)
  h903_0 : MxxRuntime.familyGetStatic (generatedRoot.w_896_2 ctx) (0) (generatedRoot.w_903_0 ctx)
  h904_0 : MxxRuntime.familyGetStatic (generatedRoot.w_896_2 ctx) (1) (generatedRoot.w_904_0 ctx)

theorem generatedRoot.facts_35  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_35 backend hashModel tape path params (generatedRoot.w_5_0 ctx) (generatedRoot.w_11_0 ctx) (generatedRoot.w_12_0 ctx) (generatedRoot.w_24_0 ctx) (generatedRoot.w_753_0 ctx) (generatedRoot.w_872_0 ctx) (generatedRoot.w_873_0 ctx) (generatedRoot.w_874_0 ctx) (generatedRoot.w_875_0 ctx) (generatedRoot.w_876_0 ctx) (generatedRoot.w_877_0 ctx) (generatedRoot.w_878_0 ctx) (generatedRoot.w_879_0 ctx) (generatedRoot.w_882_0 ctx) (generatedRoot.w_883_0 ctx) (generatedRoot.w_886_0 ctx) (generatedRoot.w_887_0 ctx) (generatedRoot.w_888_0 ctx) (generatedRoot.w_891_0 ctx) (generatedRoot.w_892_0 ctx) (generatedRoot.w_636_0 ctx) (generatedRoot.w_658_0 ctx) (generatedRoot.w_658_1 ctx) (generatedRoot.w_658_2 ctx) (generatedRoot.w_658_3 ctx) (generatedRoot.w_658_4 ctx) (generatedRoot.w_658_5 ctx) (generatedRoot.w_705_0 ctx) (generatedRoot.w_752_0 ctx) (generatedRoot.w_757_0 ctx) (generatedRoot.w_761_0 ctx) (generatedRoot.w_765_0 ctx) (generatedRoot.w_769_0 ctx) (generatedRoot.w_773_0 ctx) (generatedRoot.w_777_0 ctx) (generatedRoot.w_782_0 ctx) (generatedRoot.w_787_0 ctx) (generatedRoot.w_792_0 ctx) (generatedRoot.w_797_0 ctx) (generatedRoot.w_802_0 ctx) (generatedRoot.w_807_0 ctx) (generatedRoot.w_812_0 ctx) (generatedRoot.w_817_0 ctx) (generatedRoot.w_822_0 ctx) (generatedRoot.w_827_0 ctx) (generatedRoot.w_832_0 ctx) (generatedRoot.w_837_0 ctx) (generatedRoot.w_842_0 ctx) (generatedRoot.w_847_0 ctx) (generatedRoot.w_852_0 ctx) (generatedRoot.w_857_0 ctx) (generatedRoot.w_860_0 ctx) (generatedRoot.w_862_0 ctx) (generatedRoot.w_864_0 ctx) (generatedRoot.w_866_0 ctx) (generatedRoot.w_867_0 ctx) (generatedRoot.w_868_0 ctx) (generatedRoot.w_869_0 ctx) (generatedRoot.w_870_0 ctx) (generatedRoot.w_870_1 ctx) (generatedRoot.w_870_2 ctx) (generatedRoot.w_870_3 ctx) (generatedRoot.w_880_0 ctx) (generatedRoot.w_881_0 ctx) (generatedRoot.w_884_0 ctx) (generatedRoot.w_885_0 ctx) (generatedRoot.w_889_0 ctx) (generatedRoot.w_889_1 ctx) (generatedRoot.w_889_2 ctx) (generatedRoot.w_889_3 ctx) (generatedRoot.w_889_4 ctx) (generatedRoot.w_889_5 ctx) (generatedRoot.w_890_0 ctx) (generatedRoot.w_890_1 ctx) (generatedRoot.w_890_2 ctx) (generatedRoot.w_890_3 ctx) (generatedRoot.w_890_4 ctx) (generatedRoot.w_890_5 ctx) (generatedRoot.w_893_0 ctx) (generatedRoot.w_893_1 ctx) (generatedRoot.w_893_2 ctx) (generatedRoot.w_893_3 ctx) (generatedRoot.w_893_4 ctx) (generatedRoot.w_893_5 ctx) (generatedRoot.w_894_0 ctx) (generatedRoot.w_894_1 ctx) (generatedRoot.w_894_2 ctx) (generatedRoot.w_894_3 ctx) (generatedRoot.w_894_4 ctx) (generatedRoot.w_894_5 ctx) (generatedRoot.w_895_0 ctx) (generatedRoot.w_895_1 ctx) (generatedRoot.w_895_2 ctx) (generatedRoot.w_895_3 ctx) (generatedRoot.w_895_4 ctx) (generatedRoot.w_895_5 ctx) (generatedRoot.w_896_0 ctx) (generatedRoot.w_896_1 ctx) (generatedRoot.w_896_2 ctx) (generatedRoot.w_896_3 ctx) (generatedRoot.w_896_4 ctx) (generatedRoot.w_896_5 ctx) (generatedRoot.w_897_0 ctx) (generatedRoot.w_898_0 ctx) (generatedRoot.w_900_0 ctx) (generatedRoot.w_901_0 ctx) (generatedRoot.w_903_0 ctx) (generatedRoot.w_904_0 ctx)) :
    generatedRoot.Facts_35 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_36  (ctx : generatedRootCtx) : Prop where
  h906_0 : MxxRuntime.familyGetStatic (generatedRoot.w_896_3 ctx) (0) (generatedRoot.w_906_0 ctx)
  h907_0 : MxxRuntime.familyGetStatic (generatedRoot.w_896_3 ctx) (1) (generatedRoot.w_907_0 ctx)
  h909_0 : MxxRuntime.familyGetStatic (generatedRoot.w_896_4 ctx) (0) (generatedRoot.w_909_0 ctx)
  h910_0 : MxxRuntime.familyGetStatic (generatedRoot.w_896_4 ctx) (1) (generatedRoot.w_910_0 ctx)
  h912_0 : MxxRuntime.familyGetStatic (generatedRoot.w_896_5 ctx) (0) (generatedRoot.w_912_0 ctx)
  h913_0 : MxxRuntime.familyGetStatic (generatedRoot.w_896_5 ctx) (1) (generatedRoot.w_913_0 ctx)
  t1158 : 0 ≤ (6)
  h915_5 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_generatedRoot_915 backend hashModel tape (path ++ [915, i]) params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, ((generatedRoot.w_896_0 ctx), ((generatedRoot.w_896_1 ctx), ((generatedRoot.w_896_2 ctx), ((generatedRoot.w_896_3 ctx), ((generatedRoot.w_896_4 ctx), ((generatedRoot.w_896_5 ctx), ())))))))))))) next) (Int.toNat (6)) ((generatedRoot.w_899_0 ctx), ((generatedRoot.w_902_0 ctx), ((generatedRoot.w_905_0 ctx), ((generatedRoot.w_908_0 ctx), ((generatedRoot.w_911_0 ctx), ((generatedRoot.w_914_0 ctx), ())))))) ((generatedRoot.w_915_0 ctx), ((generatedRoot.w_915_1 ctx), ((generatedRoot.w_915_2 ctx), ((generatedRoot.w_915_3 ctx), ((generatedRoot.w_915_4 ctx), ((generatedRoot.w_915_5 ctx), ()))))))
  h916_0 : MxxRuntime.tensorRuns (generatedRoot.w_871_0 ctx) (generatedRoot.w_915_0 ctx) (generatedRoot.w_916_0 ctx)
  t1161 : 0 ≤ (0)
  t1162 : (0) < 900
  h918_0 : MxxRuntime.tensorRuns (generatedRoot.w_917_0 ctx) (generatedRoot.w_915_1 ctx) (generatedRoot.w_918_0 ctx)
  t1164 : 0 ≤ (0)
  t1165 : (0) < 900
  h920_0 : MxxRuntime.tensorRuns (generatedRoot.w_919_0 ctx) (generatedRoot.w_915_2 ctx) (generatedRoot.w_920_0 ctx)
  t1167 : 0 ≤ (0)
  t1168 : (0) < 900
  h922_0 : MxxRuntime.tensorRuns (generatedRoot.w_921_0 ctx) (generatedRoot.w_915_3 ctx) (generatedRoot.w_922_0 ctx)
  t1170 : 0 ≤ (0)
  t1171 : (0) < 900
  h924_0 : MxxRuntime.tensorRuns (generatedRoot.w_923_0 ctx) (generatedRoot.w_915_4 ctx) (generatedRoot.w_924_0 ctx)
  t1173 : 0 ≤ (0)
  t1174 : (0) < 900
  h926_0 : MxxRuntime.tensorRuns (generatedRoot.w_925_0 ctx) (generatedRoot.w_915_5 ctx) (generatedRoot.w_926_0 ctx)
  t1176 : 0 ≤ (899)
  h927_5 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat7 × mxxMat7 × mxxMat7 × mxxMat7 × mxxMat7 × mxxMat7 × Unit) => sequential_generatedRoot_927 backend hashModel tape (path ++ [927, i]) params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, ((generatedRoot.w_896_0 ctx), ((generatedRoot.w_896_1 ctx), ((generatedRoot.w_896_2 ctx), ((generatedRoot.w_896_3 ctx), ((generatedRoot.w_896_4 ctx), ((generatedRoot.w_896_5 ctx), ())))))))))))) next) (Int.toNat (899)) ((generatedRoot.w_916_0 ctx), ((generatedRoot.w_918_0 ctx), ((generatedRoot.w_920_0 ctx), ((generatedRoot.w_922_0 ctx), ((generatedRoot.w_924_0 ctx), ((generatedRoot.w_926_0 ctx), ())))))) ((generatedRoot.w_927_0 ctx), ((generatedRoot.w_927_1 ctx), ((generatedRoot.w_927_2 ctx), ((generatedRoot.w_927_3 ctx), ((generatedRoot.w_927_4 ctx), ((generatedRoot.w_927_5 ctx), ()))))))
  t1178 : 0 ≤ (0)
  t1179 : (0) < 4500
  h935_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((generatedRoot.w_5_0 ctx)) (generatedRoot.w_935_0 ctx)
  h936_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) ((generatedRoot.w_5_0 ctx)) (generatedRoot.w_936_0 ctx)
  h938_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) ((generatedRoot.w_5_0 ctx)) (generatedRoot.w_938_0 ctx)
  h939_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) ((generatedRoot.w_5_0 ctx)) (generatedRoot.w_939_0 ctx)

theorem generatedRoot.facts_36  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_36 backend hashModel tape path params (generatedRoot.w_5_0 ctx) (generatedRoot.w_871_0 ctx) (generatedRoot.w_896_0 ctx) (generatedRoot.w_896_1 ctx) (generatedRoot.w_896_2 ctx) (generatedRoot.w_896_3 ctx) (generatedRoot.w_896_4 ctx) (generatedRoot.w_896_5 ctx) (generatedRoot.w_899_0 ctx) (generatedRoot.w_902_0 ctx) (generatedRoot.w_905_0 ctx) (generatedRoot.w_906_0 ctx) (generatedRoot.w_907_0 ctx) (generatedRoot.w_908_0 ctx) (generatedRoot.w_909_0 ctx) (generatedRoot.w_910_0 ctx) (generatedRoot.w_911_0 ctx) (generatedRoot.w_912_0 ctx) (generatedRoot.w_913_0 ctx) (generatedRoot.w_914_0 ctx) (generatedRoot.w_915_0 ctx) (generatedRoot.w_915_1 ctx) (generatedRoot.w_915_2 ctx) (generatedRoot.w_915_3 ctx) (generatedRoot.w_915_4 ctx) (generatedRoot.w_915_5 ctx) (generatedRoot.w_916_0 ctx) (generatedRoot.w_917_0 ctx) (generatedRoot.w_918_0 ctx) (generatedRoot.w_919_0 ctx) (generatedRoot.w_920_0 ctx) (generatedRoot.w_921_0 ctx) (generatedRoot.w_922_0 ctx) (generatedRoot.w_923_0 ctx) (generatedRoot.w_924_0 ctx) (generatedRoot.w_925_0 ctx) (generatedRoot.w_926_0 ctx) (generatedRoot.w_927_0 ctx) (generatedRoot.w_927_1 ctx) (generatedRoot.w_927_2 ctx) (generatedRoot.w_927_3 ctx) (generatedRoot.w_927_4 ctx) (generatedRoot.w_927_5 ctx) (generatedRoot.w_935_0 ctx) (generatedRoot.w_936_0 ctx) (generatedRoot.w_938_0 ctx) (generatedRoot.w_939_0 ctx)) :
    generatedRoot.Facts_36 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_37  (ctx : generatedRootCtx) : Prop where
  t1184 : (72000) = 72000
  h943_5 : (∀ i : Fin 72000, ∃ w_101_0_gathered : mxxMat0, ∃ w_106_0_gathered : mxxMat0, ∃ w_120_0_gathered : mxxMat5, 0 ≤ ((generatedRoot.w_934_0 ctx) i) ∧ ((generatedRoot.w_934_0 ctx) i) < 2 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_101_0 ctx) ((generatedRoot.w_934_0 ctx) i) w_101_0_gathered ∧ 0 ≤ ((generatedRoot.w_934_0 ctx) i) ∧ ((generatedRoot.w_934_0 ctx) i) < 2 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_106_0 ctx) ((generatedRoot.w_934_0 ctx) i) w_106_0_gathered ∧ 0 ≤ ((generatedRoot.w_934_0 ctx) i) ∧ ((generatedRoot.w_934_0 ctx) i) < 2 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_120_0 ctx) ((generatedRoot.w_934_0 ctx) i) w_120_0_gathered ∧ parallel_generatedRoot_943 backend hashModel tape (path ++ [943, i.val]) params i (((generatedRoot.w_934_0 ctx) i), (w_101_0_gathered, ((generatedRoot.w_5_0 ctx), (w_106_0_gathered, ((generatedRoot.w_935_0 ctx), ((generatedRoot.w_936_0 ctx), ((generatedRoot.w_937_0 ctx), (w_120_0_gathered, ((generatedRoot.w_938_0 ctx), ((generatedRoot.w_939_0 ctx), ((generatedRoot.w_940_0 ctx), ((generatedRoot.w_24_0 ctx), (((generatedRoot.w_941_0 ctx) i), (((generatedRoot.w_942_0 ctx) i), ())))))))))))))) (((generatedRoot.w_943_0 ctx) i), (((generatedRoot.w_943_1 ctx) i), (((generatedRoot.w_943_2 ctx) i), (((generatedRoot.w_943_3 ctx) i), (((generatedRoot.w_943_4 ctx) i), (((generatedRoot.w_943_5 ctx) i), ())))))))
  t1186 : (36000) = 36000
  h944_5 : (∀ i : Fin 36000, parallel_generatedRoot_944 backend hashModel tape (path ++ [944, i.val]) params i ((generatedRoot.w_943_0 ctx), ((generatedRoot.w_943_1 ctx), ((generatedRoot.w_943_2 ctx), ((generatedRoot.w_943_3 ctx), ((generatedRoot.w_943_4 ctx), ((generatedRoot.w_943_5 ctx), ())))))) (((generatedRoot.w_944_0 ctx) i), (((generatedRoot.w_944_1 ctx) i), (((generatedRoot.w_944_2 ctx) i), (((generatedRoot.w_944_3 ctx) i), (((generatedRoot.w_944_4 ctx) i), (((generatedRoot.w_944_5 ctx) i), ())))))))
  t1188 : (72000) = 72000
  h947_5 : (∀ i : Fin 72000, ∃ w_101_0_gathered : mxxMat0, ∃ w_106_0_gathered : mxxMat0, ∃ w_120_0_gathered : mxxMat5, 0 ≤ ((generatedRoot.w_933_0 ctx) i) ∧ ((generatedRoot.w_933_0 ctx) i) < 2 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_101_0 ctx) ((generatedRoot.w_933_0 ctx) i) w_101_0_gathered ∧ 0 ≤ ((generatedRoot.w_933_0 ctx) i) ∧ ((generatedRoot.w_933_0 ctx) i) < 2 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_106_0 ctx) ((generatedRoot.w_933_0 ctx) i) w_106_0_gathered ∧ 0 ≤ ((generatedRoot.w_933_0 ctx) i) ∧ ((generatedRoot.w_933_0 ctx) i) < 2 ∧ MxxRuntime.familyGetDynamic (generatedRoot.w_120_0 ctx) ((generatedRoot.w_933_0 ctx) i) w_120_0_gathered ∧ parallel_generatedRoot_947 backend hashModel tape (path ++ [947, i.val]) params i (((generatedRoot.w_933_0 ctx) i), (w_101_0_gathered, ((generatedRoot.w_5_0 ctx), (w_106_0_gathered, ((generatedRoot.w_944_0 ctx), ((generatedRoot.w_944_2 ctx), ((generatedRoot.w_944_4 ctx), (w_120_0_gathered, ((generatedRoot.w_944_1 ctx), ((generatedRoot.w_944_3 ctx), ((generatedRoot.w_944_5 ctx), ((generatedRoot.w_24_0 ctx), (((generatedRoot.w_945_0 ctx) i), (((generatedRoot.w_946_0 ctx) i), ())))))))))))))) (((generatedRoot.w_947_0 ctx) i), (((generatedRoot.w_947_1 ctx) i), (((generatedRoot.w_947_2 ctx) i), (((generatedRoot.w_947_3 ctx) i), (((generatedRoot.w_947_4 ctx) i), (((generatedRoot.w_947_5 ctx) i), ())))))))
  t1190 : (36000) = 36000
  h948_5 : (∀ i : Fin 36000, parallel_generatedRoot_948 backend hashModel tape (path ++ [948, i.val]) params i ((generatedRoot.w_947_0 ctx), ((generatedRoot.w_947_1 ctx), ((generatedRoot.w_947_2 ctx), ((generatedRoot.w_947_3 ctx), ((generatedRoot.w_947_4 ctx), ((generatedRoot.w_947_5 ctx), ())))))) (((generatedRoot.w_948_0 ctx) i), (((generatedRoot.w_948_1 ctx) i), (((generatedRoot.w_948_2 ctx) i), (((generatedRoot.w_948_3 ctx) i), (((generatedRoot.w_948_4 ctx) i), (((generatedRoot.w_948_5 ctx) i), ())))))))
  t1192 : (36000) = 36000
  h949_5 : (∀ i : Fin 36000, parallel_generatedRoot_949 backend hashModel tape (path ++ [949, i.val]) params i ((generatedRoot.w_929_0 ctx), ((generatedRoot.w_930_0 ctx), ((generatedRoot.w_931_0 ctx), ((generatedRoot.w_932_0 ctx), ((generatedRoot.w_11_0 ctx), ((generatedRoot.w_5_0 ctx), ((generatedRoot.w_12_0 ctx), (((generatedRoot.w_948_4 ctx) i), (((generatedRoot.w_948_0 ctx) i), (((generatedRoot.w_948_2 ctx) i), (((generatedRoot.w_948_5 ctx) i), (((generatedRoot.w_948_1 ctx) i), (((generatedRoot.w_948_3 ctx) i), ()))))))))))))) (((generatedRoot.w_949_0 ctx) i), (((generatedRoot.w_949_1 ctx) i), (((generatedRoot.w_949_2 ctx) i), (((generatedRoot.w_949_3 ctx) i), (((generatedRoot.w_949_4 ctx) i), (((generatedRoot.w_949_5 ctx) i), ())))))))
  t1194 : (36000) = 36000
  h950_5 : (∀ i : Fin 36000, parallel_generatedRoot_950 backend hashModel tape (path ++ [950, i.val]) params i ((generatedRoot.w_949_0 ctx), ((generatedRoot.w_949_1 ctx), ((generatedRoot.w_949_2 ctx), ((generatedRoot.w_949_3 ctx), ((generatedRoot.w_949_4 ctx), ((generatedRoot.w_949_5 ctx), ())))))) (((generatedRoot.w_950_0 ctx) i), (((generatedRoot.w_950_1 ctx) i), (((generatedRoot.w_950_2 ctx) i), (((generatedRoot.w_950_3 ctx) i), (((generatedRoot.w_950_4 ctx) i), (((generatedRoot.w_950_5 ctx) i), ())))))))
  h951_0 : MxxRuntime.familyGetStatic (generatedRoot.w_950_0 ctx) (0) (generatedRoot.w_951_0 ctx)
  h952_0 : MxxRuntime.familyGetStatic (generatedRoot.w_950_0 ctx) (1) (generatedRoot.w_952_0 ctx)
  h954_0 : MxxRuntime.familyGetStatic (generatedRoot.w_950_1 ctx) (0) (generatedRoot.w_954_0 ctx)
  h955_0 : MxxRuntime.familyGetStatic (generatedRoot.w_950_1 ctx) (1) (generatedRoot.w_955_0 ctx)
  h957_0 : MxxRuntime.familyGetStatic (generatedRoot.w_950_2 ctx) (0) (generatedRoot.w_957_0 ctx)
  h958_0 : MxxRuntime.familyGetStatic (generatedRoot.w_950_2 ctx) (1) (generatedRoot.w_958_0 ctx)
  h960_0 : MxxRuntime.familyGetStatic (generatedRoot.w_950_3 ctx) (0) (generatedRoot.w_960_0 ctx)
  h961_0 : MxxRuntime.familyGetStatic (generatedRoot.w_950_3 ctx) (1) (generatedRoot.w_961_0 ctx)
  h963_0 : MxxRuntime.familyGetStatic (generatedRoot.w_950_4 ctx) (0) (generatedRoot.w_963_0 ctx)
  h964_0 : MxxRuntime.familyGetStatic (generatedRoot.w_950_4 ctx) (1) (generatedRoot.w_964_0 ctx)
  h966_0 : MxxRuntime.familyGetStatic (generatedRoot.w_950_5 ctx) (0) (generatedRoot.w_966_0 ctx)
  h967_0 : MxxRuntime.familyGetStatic (generatedRoot.w_950_5 ctx) (1) (generatedRoot.w_967_0 ctx)
  t1208 : 0 ≤ (6)
  h969_5 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_generatedRoot_969 backend hashModel tape (path ++ [969, i]) params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, ((generatedRoot.w_950_0 ctx), ((generatedRoot.w_950_1 ctx), ((generatedRoot.w_950_2 ctx), ((generatedRoot.w_950_3 ctx), ((generatedRoot.w_950_4 ctx), ((generatedRoot.w_950_5 ctx), ())))))))))))) next) (Int.toNat (6)) ((generatedRoot.w_953_0 ctx), ((generatedRoot.w_956_0 ctx), ((generatedRoot.w_959_0 ctx), ((generatedRoot.w_962_0 ctx), ((generatedRoot.w_965_0 ctx), ((generatedRoot.w_968_0 ctx), ())))))) ((generatedRoot.w_969_0 ctx), ((generatedRoot.w_969_1 ctx), ((generatedRoot.w_969_2 ctx), ((generatedRoot.w_969_3 ctx), ((generatedRoot.w_969_4 ctx), ((generatedRoot.w_969_5 ctx), ()))))))
  h970_0 : MxxRuntime.tensorRuns (generatedRoot.w_928_0 ctx) (generatedRoot.w_969_0 ctx) (generatedRoot.w_970_0 ctx)
  t1211 : 0 ≤ (0)
  t1212 : (0) < 4500
  h972_0 : MxxRuntime.tensorRuns (generatedRoot.w_971_0 ctx) (generatedRoot.w_969_1 ctx) (generatedRoot.w_972_0 ctx)
  t1214 : 0 ≤ (0)
  t1215 : (0) < 4500

theorem generatedRoot.facts_37  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_37 backend hashModel tape path params (generatedRoot.w_5_0 ctx) (generatedRoot.w_11_0 ctx) (generatedRoot.w_12_0 ctx) (generatedRoot.w_24_0 ctx) (generatedRoot.w_101_0 ctx) (generatedRoot.w_106_0 ctx) (generatedRoot.w_120_0 ctx) (generatedRoot.w_929_0 ctx) (generatedRoot.w_930_0 ctx) (generatedRoot.w_931_0 ctx) (generatedRoot.w_932_0 ctx) (generatedRoot.w_933_0 ctx) (generatedRoot.w_934_0 ctx) (generatedRoot.w_937_0 ctx) (generatedRoot.w_940_0 ctx) (generatedRoot.w_941_0 ctx) (generatedRoot.w_942_0 ctx) (generatedRoot.w_945_0 ctx) (generatedRoot.w_946_0 ctx) (generatedRoot.w_928_0 ctx) (generatedRoot.w_935_0 ctx) (generatedRoot.w_936_0 ctx) (generatedRoot.w_938_0 ctx) (generatedRoot.w_939_0 ctx) (generatedRoot.w_943_0 ctx) (generatedRoot.w_943_1 ctx) (generatedRoot.w_943_2 ctx) (generatedRoot.w_943_3 ctx) (generatedRoot.w_943_4 ctx) (generatedRoot.w_943_5 ctx) (generatedRoot.w_944_0 ctx) (generatedRoot.w_944_1 ctx) (generatedRoot.w_944_2 ctx) (generatedRoot.w_944_3 ctx) (generatedRoot.w_944_4 ctx) (generatedRoot.w_944_5 ctx) (generatedRoot.w_947_0 ctx) (generatedRoot.w_947_1 ctx) (generatedRoot.w_947_2 ctx) (generatedRoot.w_947_3 ctx) (generatedRoot.w_947_4 ctx) (generatedRoot.w_947_5 ctx) (generatedRoot.w_948_0 ctx) (generatedRoot.w_948_1 ctx) (generatedRoot.w_948_2 ctx) (generatedRoot.w_948_3 ctx) (generatedRoot.w_948_4 ctx) (generatedRoot.w_948_5 ctx) (generatedRoot.w_949_0 ctx) (generatedRoot.w_949_1 ctx) (generatedRoot.w_949_2 ctx) (generatedRoot.w_949_3 ctx) (generatedRoot.w_949_4 ctx) (generatedRoot.w_949_5 ctx) (generatedRoot.w_950_0 ctx) (generatedRoot.w_950_1 ctx) (generatedRoot.w_950_2 ctx) (generatedRoot.w_950_3 ctx) (generatedRoot.w_950_4 ctx) (generatedRoot.w_950_5 ctx) (generatedRoot.w_951_0 ctx) (generatedRoot.w_952_0 ctx) (generatedRoot.w_953_0 ctx) (generatedRoot.w_954_0 ctx) (generatedRoot.w_955_0 ctx) (generatedRoot.w_956_0 ctx) (generatedRoot.w_957_0 ctx) (generatedRoot.w_958_0 ctx) (generatedRoot.w_959_0 ctx) (generatedRoot.w_960_0 ctx) (generatedRoot.w_961_0 ctx) (generatedRoot.w_962_0 ctx) (generatedRoot.w_963_0 ctx) (generatedRoot.w_964_0 ctx) (generatedRoot.w_965_0 ctx) (generatedRoot.w_966_0 ctx) (generatedRoot.w_967_0 ctx) (generatedRoot.w_968_0 ctx) (generatedRoot.w_969_0 ctx) (generatedRoot.w_969_1 ctx) (generatedRoot.w_969_2 ctx) (generatedRoot.w_969_3 ctx) (generatedRoot.w_969_4 ctx) (generatedRoot.w_969_5 ctx) (generatedRoot.w_970_0 ctx) (generatedRoot.w_971_0 ctx) (generatedRoot.w_972_0 ctx)) :
    generatedRoot.Facts_37 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_38  (ctx : generatedRootCtx) : Prop where
  h974_0 : MxxRuntime.tensorRuns (generatedRoot.w_973_0 ctx) (generatedRoot.w_969_2 ctx) (generatedRoot.w_974_0 ctx)
  t1217 : 0 ≤ (0)
  t1218 : (0) < 4500
  h976_0 : MxxRuntime.tensorRuns (generatedRoot.w_975_0 ctx) (generatedRoot.w_969_3 ctx) (generatedRoot.w_976_0 ctx)
  t1220 : 0 ≤ (0)
  t1221 : (0) < 4500
  h978_0 : MxxRuntime.tensorRuns (generatedRoot.w_977_0 ctx) (generatedRoot.w_969_4 ctx) (generatedRoot.w_978_0 ctx)
  t1223 : 0 ≤ (0)
  t1224 : (0) < 4500
  h980_0 : MxxRuntime.tensorRuns (generatedRoot.w_979_0 ctx) (generatedRoot.w_969_5 ctx) (generatedRoot.w_980_0 ctx)
  t1226 : 0 ≤ (4499)
  h981_5 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat9 × mxxMat9 × mxxMat9 × mxxMat9 × mxxMat9 × mxxMat9 × Unit) => sequential_generatedRoot_981 backend hashModel tape (path ++ [981, i]) params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, ((generatedRoot.w_950_0 ctx), ((generatedRoot.w_950_1 ctx), ((generatedRoot.w_950_2 ctx), ((generatedRoot.w_950_3 ctx), ((generatedRoot.w_950_4 ctx), ((generatedRoot.w_950_5 ctx), ())))))))))))) next) (Int.toNat (4499)) ((generatedRoot.w_970_0 ctx), ((generatedRoot.w_972_0 ctx), ((generatedRoot.w_974_0 ctx), ((generatedRoot.w_976_0 ctx), ((generatedRoot.w_978_0 ctx), ((generatedRoot.w_980_0 ctx), ())))))) ((generatedRoot.w_981_0 ctx), ((generatedRoot.w_981_1 ctx), ((generatedRoot.w_981_2 ctx), ((generatedRoot.w_981_3 ctx), ((generatedRoot.w_981_4 ctx), ((generatedRoot.w_981_5 ctx), ()))))))
  t1228 : (1) = 1
  h983_0 : (∀ i : Fin 1, parallel_generatedRoot_983 backend hashModel tape (path ++ [983, i.val]) params i (generatedRoot.w_982_0 ctx) ((generatedRoot.w_983_0 ctx) i))
  t1230 : (36) = 36
  h984_0 : (∀ i : Fin 36, parallel_generatedRoot_984 backend hashModel tape (path ++ [984, i.val]) params i ((generatedRoot.w_61_0 ctx), ((generatedRoot.w_68_0 ctx), (((generatedRoot.w_870_3 ctx) i), (((generatedRoot.w_870_1 ctx) i), ((generatedRoot.w_927_1 ctx), ((generatedRoot.w_55_0 ctx), ((generatedRoot.w_981_5 ctx), ((generatedRoot.w_983_0 ctx), ((generatedRoot.w_981_1 ctx), ((generatedRoot.w_981_3 ctx), ((generatedRoot.w_652_0 ctx), ((generatedRoot.w_656_0 ctx), (((generatedRoot.w_870_2 ctx) i), ((generatedRoot.w_927_3 ctx), ())))))))))))))) ((generatedRoot.w_984_0 ctx) i))
  t1232 : (36) = 36
  h985_0 : (∀ i : Fin 36, parallel_generatedRoot_985 backend hashModel tape (path ++ [985, i.val]) params i ((generatedRoot.w_61_0 ctx), ((generatedRoot.w_68_0 ctx), (((generatedRoot.w_870_3 ctx) i), (((generatedRoot.w_870_1 ctx) i), ((generatedRoot.w_927_1 ctx), ((generatedRoot.w_55_0 ctx), ((generatedRoot.w_981_5 ctx), ((generatedRoot.w_983_0 ctx), ((generatedRoot.w_981_1 ctx), ((generatedRoot.w_981_3 ctx), ((generatedRoot.w_652_0 ctx), ((generatedRoot.w_656_0 ctx), (((generatedRoot.w_870_2 ctx) i), ((generatedRoot.w_927_3 ctx), ())))))))))))))) ((generatedRoot.w_985_0 ctx) i))
  t1234 : (36) = 36
  h986_0 : (∀ i : Fin 36, parallel_generatedRoot_986 backend hashModel tape (path ++ [986, i.val]) params i ((generatedRoot.w_61_0 ctx), ((generatedRoot.w_68_0 ctx), (((generatedRoot.w_870_3 ctx) i), (((generatedRoot.w_870_1 ctx) i), ((generatedRoot.w_927_1 ctx), ((generatedRoot.w_55_0 ctx), ((generatedRoot.w_981_5 ctx), ((generatedRoot.w_983_0 ctx), ((generatedRoot.w_981_1 ctx), ((generatedRoot.w_981_3 ctx), ((generatedRoot.w_652_0 ctx), ((generatedRoot.w_656_0 ctx), (((generatedRoot.w_870_2 ctx) i), ((generatedRoot.w_927_3 ctx), ())))))))))))))) ((generatedRoot.w_986_0 ctx) i))
  t1236 : (36) = 36
  h987_0 : (∀ i : Fin 36, parallel_generatedRoot_987 backend hashModel tape (path ++ [987, i.val]) params i ((generatedRoot.w_61_0 ctx), ((generatedRoot.w_68_0 ctx), (((generatedRoot.w_870_3 ctx) i), (((generatedRoot.w_870_1 ctx) i), ((generatedRoot.w_927_1 ctx), ((generatedRoot.w_55_0 ctx), ((generatedRoot.w_981_5 ctx), ((generatedRoot.w_983_0 ctx), ((generatedRoot.w_981_1 ctx), ((generatedRoot.w_981_3 ctx), ((generatedRoot.w_652_0 ctx), ((generatedRoot.w_656_0 ctx), (((generatedRoot.w_870_2 ctx) i), ((generatedRoot.w_927_3 ctx), ())))))))))))))) ((generatedRoot.w_987_0 ctx) i))
  t1238 : (36) = 36
  h988_0 : (∀ i : Fin 36, parallel_generatedRoot_988 backend hashModel tape (path ++ [988, i.val]) params i ((generatedRoot.w_61_0 ctx), ((generatedRoot.w_68_0 ctx), (((generatedRoot.w_870_3 ctx) i), (((generatedRoot.w_870_1 ctx) i), ((generatedRoot.w_927_1 ctx), ((generatedRoot.w_55_0 ctx), ((generatedRoot.w_981_5 ctx), ((generatedRoot.w_983_0 ctx), ((generatedRoot.w_981_1 ctx), ((generatedRoot.w_981_3 ctx), ((generatedRoot.w_652_0 ctx), ((generatedRoot.w_656_0 ctx), (((generatedRoot.w_870_2 ctx) i), ((generatedRoot.w_927_3 ctx), ())))))))))))))) ((generatedRoot.w_988_0 ctx) i))
  h989_0 : MxxRuntime.concatRows (generatedRoot.w_530_6 ctx) (generatedRoot.w_530_8 ctx) (generatedRoot.w_989_0 ctx)
  h990_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_989_0 ctx) (generatedRoot.w_990_0 ctx)
  h991_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_990_0 ctx) (generatedRoot.w_991_0 ctx)
  t1243 : 0 ≤ 1
  t1244 : 1 < 2
  t1245 : 2 ≤ 3
  t1246 : 0 ≤ 0
  t1247 : 0 < 25

theorem generatedRoot.facts_38  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_38 backend hashModel tape path params (generatedRoot.w_55_0 ctx) (generatedRoot.w_61_0 ctx) (generatedRoot.w_68_0 ctx) (generatedRoot.w_530_6 ctx) (generatedRoot.w_530_8 ctx) (generatedRoot.w_652_0 ctx) (generatedRoot.w_656_0 ctx) (generatedRoot.w_870_1 ctx) (generatedRoot.w_870_2 ctx) (generatedRoot.w_870_3 ctx) (generatedRoot.w_927_1 ctx) (generatedRoot.w_927_3 ctx) (generatedRoot.w_950_0 ctx) (generatedRoot.w_950_1 ctx) (generatedRoot.w_950_2 ctx) (generatedRoot.w_950_3 ctx) (generatedRoot.w_950_4 ctx) (generatedRoot.w_950_5 ctx) (generatedRoot.w_969_2 ctx) (generatedRoot.w_969_3 ctx) (generatedRoot.w_969_4 ctx) (generatedRoot.w_969_5 ctx) (generatedRoot.w_970_0 ctx) (generatedRoot.w_972_0 ctx) (generatedRoot.w_973_0 ctx) (generatedRoot.w_974_0 ctx) (generatedRoot.w_975_0 ctx) (generatedRoot.w_976_0 ctx) (generatedRoot.w_977_0 ctx) (generatedRoot.w_978_0 ctx) (generatedRoot.w_979_0 ctx) (generatedRoot.w_980_0 ctx) (generatedRoot.w_981_0 ctx) (generatedRoot.w_981_1 ctx) (generatedRoot.w_981_2 ctx) (generatedRoot.w_981_3 ctx) (generatedRoot.w_981_4 ctx) (generatedRoot.w_981_5 ctx) (generatedRoot.w_982_0 ctx) (generatedRoot.w_983_0 ctx) (generatedRoot.w_984_0 ctx) (generatedRoot.w_985_0 ctx) (generatedRoot.w_986_0 ctx) (generatedRoot.w_987_0 ctx) (generatedRoot.w_988_0 ctx) (generatedRoot.w_989_0 ctx) (generatedRoot.w_990_0 ctx) (generatedRoot.w_991_0 ctx)) :
    generatedRoot.Facts_38 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_39  (ctx : generatedRootCtx) : Prop where
  t1248 : 25 ≤ 25
  h993_0 : MxxRuntime.sliceMatrix (generatedRoot.w_992_0 ctx) 1 2 0 25 (generatedRoot.w_993_0 ctx)
  t1250 : 0 ≤ 2
  t1251 : 2 < 3
  t1252 : 3 ≤ 3
  t1253 : 0 ≤ 0
  t1254 : 0 < 25
  t1255 : 25 ≤ 25
  h994_0 : MxxRuntime.sliceMatrix (generatedRoot.w_992_0 ctx) 2 3 0 25 (generatedRoot.w_994_0 ctx)
  h995_0 : MxxRuntime.concatRows (generatedRoot.w_993_0 ctx) (generatedRoot.w_994_0 ctx) (generatedRoot.w_995_0 ctx)
  h996_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_995_0 ctx) (generatedRoot.w_996_0 ctx)
  h997_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_996_0 ctx) (generatedRoot.w_997_0 ctx)
  t1260 : 0 ≤ 1
  t1261 : 1 < 2
  t1262 : 2 ≤ 3
  t1263 : 0 ≤ 0
  t1264 : 0 < 25
  t1265 : 25 ≤ 25
  h999_0 : MxxRuntime.sliceMatrix (generatedRoot.w_998_0 ctx) 1 2 0 25 (generatedRoot.w_999_0 ctx)
  t1267 : 0 ≤ 2
  t1268 : 2 < 3
  t1269 : 3 ≤ 3
  t1270 : 0 ≤ 0
  t1271 : 0 < 25
  t1272 : 25 ≤ 25
  h1001_0 : MxxRuntime.sliceMatrix (generatedRoot.w_998_0 ctx) 2 3 0 25 (generatedRoot.w_1001_0 ctx)
  t1274 : 0 ≤ 0
  t1275 : 0 < 1
  t1276 : 1 ≤ 3
  t1277 : 0 ≤ 0
  t1278 : 0 < 25
  t1279 : 25 ≤ 25

theorem generatedRoot.facts_39  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_39 backend (generatedRoot.w_992_0 ctx) (generatedRoot.w_993_0 ctx) (generatedRoot.w_994_0 ctx) (generatedRoot.w_995_0 ctx) (generatedRoot.w_996_0 ctx) (generatedRoot.w_997_0 ctx) (generatedRoot.w_998_0 ctx) (generatedRoot.w_999_0 ctx) (generatedRoot.w_1001_0 ctx)) :
    generatedRoot.Facts_39 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_40  (ctx : generatedRootCtx) : Prop where
  h1003_0 : MxxRuntime.sliceMatrix (generatedRoot.w_998_0 ctx) 0 1 0 25 (generatedRoot.w_1003_0 ctx)
  t1281 : 0 ≤ 0
  t1282 : 0 < 1
  t1283 : 1 ≤ 3
  t1284 : 0 ≤ 0
  t1285 : 0 < 25
  t1286 : 25 ≤ 25
  h1004_0 : MxxRuntime.sliceMatrix (generatedRoot.w_992_0 ctx) 0 1 0 25 (generatedRoot.w_1004_0 ctx)
  h1007_0 : MxxRuntime.concatRows (generatedRoot.w_588_6 ctx) (generatedRoot.w_588_8 ctx) (generatedRoot.w_1007_0 ctx)
  h1008_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_1007_0 ctx) (generatedRoot.w_1008_0 ctx)
  h1009_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_1008_0 ctx) (generatedRoot.w_1009_0 ctx)
  t1291 : 0 ≤ 1
  t1292 : 1 < 2
  t1293 : 2 ≤ 3
  t1294 : 0 ≤ 0
  t1295 : 0 < 25
  t1296 : 25 ≤ 25
  h1011_0 : MxxRuntime.sliceMatrix (generatedRoot.w_1010_0 ctx) 1 2 0 25 (generatedRoot.w_1011_0 ctx)
  t1298 : 0 ≤ 2
  t1299 : 2 < 3
  t1300 : 3 ≤ 3
  t1301 : 0 ≤ 0
  t1302 : 0 < 25
  t1303 : 25 ≤ 25
  h1012_0 : MxxRuntime.sliceMatrix (generatedRoot.w_1010_0 ctx) 2 3 0 25 (generatedRoot.w_1012_0 ctx)
  h1013_0 : MxxRuntime.concatRows (generatedRoot.w_1011_0 ctx) (generatedRoot.w_1012_0 ctx) (generatedRoot.w_1013_0 ctx)
  h1014_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_1013_0 ctx) (generatedRoot.w_1014_0 ctx)
  h1015_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_1014_0 ctx) (generatedRoot.w_1015_0 ctx)
  t1308 : 0 ≤ 1
  t1309 : 1 < 2
  t1310 : 2 ≤ 3
  t1311 : 0 ≤ 0

theorem generatedRoot.facts_40  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_40 backend (generatedRoot.w_588_6 ctx) (generatedRoot.w_588_8 ctx) (generatedRoot.w_992_0 ctx) (generatedRoot.w_998_0 ctx) (generatedRoot.w_1003_0 ctx) (generatedRoot.w_1004_0 ctx) (generatedRoot.w_1007_0 ctx) (generatedRoot.w_1008_0 ctx) (generatedRoot.w_1009_0 ctx) (generatedRoot.w_1010_0 ctx) (generatedRoot.w_1011_0 ctx) (generatedRoot.w_1012_0 ctx) (generatedRoot.w_1013_0 ctx) (generatedRoot.w_1014_0 ctx) (generatedRoot.w_1015_0 ctx)) :
    generatedRoot.Facts_40 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_41  (ctx : generatedRootCtx) : Prop where
  t1312 : 0 < 25
  t1313 : 25 ≤ 25
  h1017_0 : MxxRuntime.sliceMatrix (generatedRoot.w_1016_0 ctx) 1 2 0 25 (generatedRoot.w_1017_0 ctx)
  t1315 : 0 ≤ 2
  t1316 : 2 < 3
  t1317 : 3 ≤ 3
  t1318 : 0 ≤ 0
  t1319 : 0 < 25
  t1320 : 25 ≤ 25
  h1019_0 : MxxRuntime.sliceMatrix (generatedRoot.w_1016_0 ctx) 2 3 0 25 (generatedRoot.w_1019_0 ctx)
  t1322 : 0 ≤ 0
  t1323 : 0 < 1
  t1324 : 1 ≤ 3
  t1325 : 0 ≤ 0
  t1326 : 0 < 25
  t1327 : 25 ≤ 25
  h1021_0 : MxxRuntime.sliceMatrix (generatedRoot.w_1016_0 ctx) 0 1 0 25 (generatedRoot.w_1021_0 ctx)
  t1329 : 0 ≤ 0
  t1330 : 0 < 1
  t1331 : 1 ≤ 3
  t1332 : 0 ≤ 0
  t1333 : 0 < 25
  t1334 : 25 ≤ 25
  h1022_0 : MxxRuntime.sliceMatrix (generatedRoot.w_1010_0 ctx) 0 1 0 25 (generatedRoot.w_1022_0 ctx)
  h1025_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) ((generatedRoot.w_5_0 ctx)) (generatedRoot.w_1025_0 ctx)
  h1026_0 : MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) ((generatedRoot.w_5_0 ctx)) (generatedRoot.w_1026_0 ctx)
  h1027_0 : MxxRuntime.familyGetStatic (generatedRoot.w_14_0 ctx) (1) (generatedRoot.w_1027_0 ctx)
  h1028_0 : MxxRuntime.familyGetStatic (generatedRoot.w_20_0 ctx) (1) (generatedRoot.w_1028_0 ctx)
  h1029_0 : MxxRuntime.concatRows (generatedRoot.w_240_6 ctx) (generatedRoot.w_240_8 ctx) (generatedRoot.w_1029_0 ctx)
  h1030_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_1029_0 ctx) (generatedRoot.w_1030_0 ctx)
  h1031_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_1030_0 ctx) (generatedRoot.w_1031_0 ctx)
  t1343 : 0 ≤ 1

theorem generatedRoot.facts_41  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_41 backend hashModel (generatedRoot.w_5_0 ctx) (generatedRoot.w_14_0 ctx) (generatedRoot.w_20_0 ctx) (generatedRoot.w_240_6 ctx) (generatedRoot.w_240_8 ctx) (generatedRoot.w_1010_0 ctx) (generatedRoot.w_1016_0 ctx) (generatedRoot.w_1017_0 ctx) (generatedRoot.w_1019_0 ctx) (generatedRoot.w_1021_0 ctx) (generatedRoot.w_1022_0 ctx) (generatedRoot.w_1025_0 ctx) (generatedRoot.w_1026_0 ctx) (generatedRoot.w_1027_0 ctx) (generatedRoot.w_1028_0 ctx) (generatedRoot.w_1029_0 ctx) (generatedRoot.w_1030_0 ctx) (generatedRoot.w_1031_0 ctx)) :
    generatedRoot.Facts_41 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_42  (ctx : generatedRootCtx) : Prop where
  t1344 : 1 < 2
  t1345 : 2 ≤ 3
  t1346 : 0 ≤ 0
  t1347 : 0 < 25
  t1348 : 25 ≤ 25
  h1033_0 : MxxRuntime.sliceMatrix (generatedRoot.w_1032_0 ctx) 1 2 0 25 (generatedRoot.w_1033_0 ctx)
  t1350 : 0 ≤ 2
  t1351 : 2 < 3
  t1352 : 3 ≤ 3
  t1353 : 0 ≤ 0
  t1354 : 0 < 25
  t1355 : 25 ≤ 25
  h1034_0 : MxxRuntime.sliceMatrix (generatedRoot.w_1032_0 ctx) 2 3 0 25 (generatedRoot.w_1034_0 ctx)
  h1035_0 : MxxRuntime.concatRows (generatedRoot.w_1033_0 ctx) (generatedRoot.w_1034_0 ctx) (generatedRoot.w_1035_0 ctx)
  h1036_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_1035_0 ctx) (generatedRoot.w_1036_0 ctx)
  h1037_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_1036_0 ctx) (generatedRoot.w_1037_0 ctx)
  t1360 : 0 ≤ 1
  t1361 : 1 < 2
  t1362 : 2 ≤ 3
  t1363 : 0 ≤ 0
  t1364 : 0 < 25
  t1365 : 25 ≤ 25
  h1039_0 : MxxRuntime.sliceMatrix (generatedRoot.w_1038_0 ctx) 1 2 0 25 (generatedRoot.w_1039_0 ctx)
  t1367 : 0 ≤ 2
  t1368 : 2 < 3
  t1369 : 3 ≤ 3
  t1370 : 0 ≤ 0
  t1371 : 0 < 25
  t1372 : 25 ≤ 25
  h1041_0 : MxxRuntime.sliceMatrix (generatedRoot.w_1038_0 ctx) 2 3 0 25 (generatedRoot.w_1041_0 ctx)
  t1374 : 0 ≤ 0
  t1375 : 0 < 1

theorem generatedRoot.facts_42  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_42 backend (generatedRoot.w_1032_0 ctx) (generatedRoot.w_1033_0 ctx) (generatedRoot.w_1034_0 ctx) (generatedRoot.w_1035_0 ctx) (generatedRoot.w_1036_0 ctx) (generatedRoot.w_1037_0 ctx) (generatedRoot.w_1038_0 ctx) (generatedRoot.w_1039_0 ctx) (generatedRoot.w_1041_0 ctx)) :
    generatedRoot.Facts_42 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_43  (ctx : generatedRootCtx) : Prop where
  t1376 : 1 ≤ 3
  t1377 : 0 ≤ 0
  t1378 : 0 < 25
  t1379 : 25 ≤ 25
  h1043_0 : MxxRuntime.sliceMatrix (generatedRoot.w_1038_0 ctx) 0 1 0 25 (generatedRoot.w_1043_0 ctx)
  t1381 : 0 ≤ 0
  t1382 : 0 < 1
  t1383 : 1 ≤ 3
  t1384 : 0 ≤ 0
  t1385 : 0 < 25
  t1386 : 25 ≤ 25
  h1044_0 : MxxRuntime.sliceMatrix (generatedRoot.w_1032_0 ctx) 0 1 0 25 (generatedRoot.w_1044_0 ctx)
  h1047_0 : MxxRuntime.concatRows (generatedRoot.w_298_6 ctx) (generatedRoot.w_298_8 ctx) (generatedRoot.w_1047_0 ctx)
  h1048_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_1047_0 ctx) (generatedRoot.w_1048_0 ctx)
  h1049_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_1048_0 ctx) (generatedRoot.w_1049_0 ctx)
  t1391 : 0 ≤ 1
  t1392 : 1 < 2
  t1393 : 2 ≤ 3
  t1394 : 0 ≤ 0
  t1395 : 0 < 25
  t1396 : 25 ≤ 25
  h1051_0 : MxxRuntime.sliceMatrix (generatedRoot.w_1050_0 ctx) 1 2 0 25 (generatedRoot.w_1051_0 ctx)
  t1398 : 0 ≤ 2
  t1399 : 2 < 3
  t1400 : 3 ≤ 3
  t1401 : 0 ≤ 0
  t1402 : 0 < 25
  t1403 : 25 ≤ 25
  h1052_0 : MxxRuntime.sliceMatrix (generatedRoot.w_1050_0 ctx) 2 3 0 25 (generatedRoot.w_1052_0 ctx)
  h1053_0 : MxxRuntime.concatRows (generatedRoot.w_1051_0 ctx) (generatedRoot.w_1052_0 ctx) (generatedRoot.w_1053_0 ctx)
  h1054_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_1053_0 ctx) (generatedRoot.w_1054_0 ctx)
  h1055_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_1054_0 ctx) (generatedRoot.w_1055_0 ctx)

theorem generatedRoot.facts_43  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_43 backend (generatedRoot.w_298_6 ctx) (generatedRoot.w_298_8 ctx) (generatedRoot.w_1032_0 ctx) (generatedRoot.w_1038_0 ctx) (generatedRoot.w_1043_0 ctx) (generatedRoot.w_1044_0 ctx) (generatedRoot.w_1047_0 ctx) (generatedRoot.w_1048_0 ctx) (generatedRoot.w_1049_0 ctx) (generatedRoot.w_1050_0 ctx) (generatedRoot.w_1051_0 ctx) (generatedRoot.w_1052_0 ctx) (generatedRoot.w_1053_0 ctx) (generatedRoot.w_1054_0 ctx) (generatedRoot.w_1055_0 ctx)) :
    generatedRoot.Facts_43 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_44  (ctx : generatedRootCtx) : Prop where
  t1408 : 0 ≤ 1
  t1409 : 1 < 2
  t1410 : 2 ≤ 3
  t1411 : 0 ≤ 0
  t1412 : 0 < 25
  t1413 : 25 ≤ 25
  h1057_0 : MxxRuntime.sliceMatrix (generatedRoot.w_1056_0 ctx) 1 2 0 25 (generatedRoot.w_1057_0 ctx)
  t1415 : 0 ≤ 2
  t1416 : 2 < 3
  t1417 : 3 ≤ 3
  t1418 : 0 ≤ 0
  t1419 : 0 < 25
  t1420 : 25 ≤ 25
  h1059_0 : MxxRuntime.sliceMatrix (generatedRoot.w_1056_0 ctx) 2 3 0 25 (generatedRoot.w_1059_0 ctx)
  t1422 : 0 ≤ 0
  t1423 : 0 < 1
  t1424 : 1 ≤ 3
  t1425 : 0 ≤ 0
  t1426 : 0 < 25
  t1427 : 25 ≤ 25
  h1061_0 : MxxRuntime.sliceMatrix (generatedRoot.w_1056_0 ctx) 0 1 0 25 (generatedRoot.w_1061_0 ctx)
  t1429 : 0 ≤ 0
  t1430 : 0 < 1
  t1431 : 1 ≤ 3
  t1432 : 0 ≤ 0
  t1433 : 0 < 25
  t1434 : 25 ≤ 25
  h1062_0 : MxxRuntime.sliceMatrix (generatedRoot.w_1050_0 ctx) 0 1 0 25 (generatedRoot.w_1062_0 ctx)
  h1065_0 : MxxRuntime.concatRows (generatedRoot.w_356_6 ctx) (generatedRoot.w_356_8 ctx) (generatedRoot.w_1065_0 ctx)
  h1066_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_1065_0 ctx) (generatedRoot.w_1066_0 ctx)
  h1067_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_1066_0 ctx) (generatedRoot.w_1067_0 ctx)
  t1439 : 0 ≤ 1

theorem generatedRoot.facts_44  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_44 backend (generatedRoot.w_356_6 ctx) (generatedRoot.w_356_8 ctx) (generatedRoot.w_1050_0 ctx) (generatedRoot.w_1056_0 ctx) (generatedRoot.w_1057_0 ctx) (generatedRoot.w_1059_0 ctx) (generatedRoot.w_1061_0 ctx) (generatedRoot.w_1062_0 ctx) (generatedRoot.w_1065_0 ctx) (generatedRoot.w_1066_0 ctx) (generatedRoot.w_1067_0 ctx)) :
    generatedRoot.Facts_44 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_45  (ctx : generatedRootCtx) : Prop where
  t1440 : 1 < 2
  t1441 : 2 ≤ 3
  t1442 : 0 ≤ 0
  t1443 : 0 < 25
  t1444 : 25 ≤ 25
  h1069_0 : MxxRuntime.sliceMatrix (generatedRoot.w_1068_0 ctx) 1 2 0 25 (generatedRoot.w_1069_0 ctx)
  t1446 : 0 ≤ 2
  t1447 : 2 < 3
  t1448 : 3 ≤ 3
  t1449 : 0 ≤ 0
  t1450 : 0 < 25
  t1451 : 25 ≤ 25
  h1070_0 : MxxRuntime.sliceMatrix (generatedRoot.w_1068_0 ctx) 2 3 0 25 (generatedRoot.w_1070_0 ctx)
  h1071_0 : MxxRuntime.concatRows (generatedRoot.w_1069_0 ctx) (generatedRoot.w_1070_0 ctx) (generatedRoot.w_1071_0 ctx)
  h1072_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_1071_0 ctx) (generatedRoot.w_1072_0 ctx)
  h1073_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_1072_0 ctx) (generatedRoot.w_1073_0 ctx)
  t1456 : 0 ≤ 1
  t1457 : 1 < 2
  t1458 : 2 ≤ 3
  t1459 : 0 ≤ 0
  t1460 : 0 < 25
  t1461 : 25 ≤ 25
  h1075_0 : MxxRuntime.sliceMatrix (generatedRoot.w_1074_0 ctx) 1 2 0 25 (generatedRoot.w_1075_0 ctx)
  t1463 : 0 ≤ 2
  t1464 : 2 < 3
  t1465 : 3 ≤ 3
  t1466 : 0 ≤ 0
  t1467 : 0 < 25
  t1468 : 25 ≤ 25
  h1077_0 : MxxRuntime.sliceMatrix (generatedRoot.w_1074_0 ctx) 2 3 0 25 (generatedRoot.w_1077_0 ctx)
  t1470 : 0 ≤ 0
  t1471 : 0 < 1

theorem generatedRoot.facts_45  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_45 backend (generatedRoot.w_1068_0 ctx) (generatedRoot.w_1069_0 ctx) (generatedRoot.w_1070_0 ctx) (generatedRoot.w_1071_0 ctx) (generatedRoot.w_1072_0 ctx) (generatedRoot.w_1073_0 ctx) (generatedRoot.w_1074_0 ctx) (generatedRoot.w_1075_0 ctx) (generatedRoot.w_1077_0 ctx)) :
    generatedRoot.Facts_45 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_46  (ctx : generatedRootCtx) : Prop where
  t1472 : 1 ≤ 3
  t1473 : 0 ≤ 0
  t1474 : 0 < 25
  t1475 : 25 ≤ 25
  h1079_0 : MxxRuntime.sliceMatrix (generatedRoot.w_1074_0 ctx) 0 1 0 25 (generatedRoot.w_1079_0 ctx)
  t1477 : 0 ≤ 0
  t1478 : 0 < 1
  t1479 : 1 ≤ 3
  t1480 : 0 ≤ 0
  t1481 : 0 < 25
  t1482 : 25 ≤ 25
  h1080_0 : MxxRuntime.sliceMatrix (generatedRoot.w_1068_0 ctx) 0 1 0 25 (generatedRoot.w_1080_0 ctx)
  h1083_0 : MxxRuntime.concatRows (generatedRoot.w_414_6 ctx) (generatedRoot.w_414_8 ctx) (generatedRoot.w_1083_0 ctx)
  h1084_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_1083_0 ctx) (generatedRoot.w_1084_0 ctx)
  h1085_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_1084_0 ctx) (generatedRoot.w_1085_0 ctx)
  t1487 : 0 ≤ 1
  t1488 : 1 < 2
  t1489 : 2 ≤ 3
  t1490 : 0 ≤ 0
  t1491 : 0 < 25
  t1492 : 25 ≤ 25
  h1087_0 : MxxRuntime.sliceMatrix (generatedRoot.w_1086_0 ctx) 1 2 0 25 (generatedRoot.w_1087_0 ctx)
  t1494 : 0 ≤ 2
  t1495 : 2 < 3
  t1496 : 3 ≤ 3
  t1497 : 0 ≤ 0
  t1498 : 0 < 25
  t1499 : 25 ≤ 25
  h1088_0 : MxxRuntime.sliceMatrix (generatedRoot.w_1086_0 ctx) 2 3 0 25 (generatedRoot.w_1088_0 ctx)
  h1089_0 : MxxRuntime.concatRows (generatedRoot.w_1087_0 ctx) (generatedRoot.w_1088_0 ctx) (generatedRoot.w_1089_0 ctx)
  h1090_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_1089_0 ctx) (generatedRoot.w_1090_0 ctx)
  h1091_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_1090_0 ctx) (generatedRoot.w_1091_0 ctx)

theorem generatedRoot.facts_46  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_46 backend (generatedRoot.w_414_6 ctx) (generatedRoot.w_414_8 ctx) (generatedRoot.w_1068_0 ctx) (generatedRoot.w_1074_0 ctx) (generatedRoot.w_1079_0 ctx) (generatedRoot.w_1080_0 ctx) (generatedRoot.w_1083_0 ctx) (generatedRoot.w_1084_0 ctx) (generatedRoot.w_1085_0 ctx) (generatedRoot.w_1086_0 ctx) (generatedRoot.w_1087_0 ctx) (generatedRoot.w_1088_0 ctx) (generatedRoot.w_1089_0 ctx) (generatedRoot.w_1090_0 ctx) (generatedRoot.w_1091_0 ctx)) :
    generatedRoot.Facts_46 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_47  (ctx : generatedRootCtx) : Prop where
  t1504 : 0 ≤ 1
  t1505 : 1 < 2
  t1506 : 2 ≤ 3
  t1507 : 0 ≤ 0
  t1508 : 0 < 25
  t1509 : 25 ≤ 25
  h1093_0 : MxxRuntime.sliceMatrix (generatedRoot.w_1092_0 ctx) 1 2 0 25 (generatedRoot.w_1093_0 ctx)
  t1511 : 0 ≤ 2
  t1512 : 2 < 3
  t1513 : 3 ≤ 3
  t1514 : 0 ≤ 0
  t1515 : 0 < 25
  t1516 : 25 ≤ 25
  h1095_0 : MxxRuntime.sliceMatrix (generatedRoot.w_1092_0 ctx) 2 3 0 25 (generatedRoot.w_1095_0 ctx)
  t1518 : 0 ≤ 0
  t1519 : 0 < 1
  t1520 : 1 ≤ 3
  t1521 : 0 ≤ 0
  t1522 : 0 < 25
  t1523 : 25 ≤ 25
  h1097_0 : MxxRuntime.sliceMatrix (generatedRoot.w_1092_0 ctx) 0 1 0 25 (generatedRoot.w_1097_0 ctx)
  t1525 : 0 ≤ 0
  t1526 : 0 < 1
  t1527 : 1 ≤ 3
  t1528 : 0 ≤ 0
  t1529 : 0 < 25
  t1530 : 25 ≤ 25
  h1098_0 : MxxRuntime.sliceMatrix (generatedRoot.w_1086_0 ctx) 0 1 0 25 (generatedRoot.w_1098_0 ctx)
  h1101_0 : MxxRuntime.concatRows (generatedRoot.w_472_6 ctx) (generatedRoot.w_472_8 ctx) (generatedRoot.w_1101_0 ctx)
  h1102_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_1101_0 ctx) (generatedRoot.w_1102_0 ctx)
  h1103_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_1102_0 ctx) (generatedRoot.w_1103_0 ctx)
  t1535 : 0 ≤ 1

theorem generatedRoot.facts_47  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_47 backend (generatedRoot.w_472_6 ctx) (generatedRoot.w_472_8 ctx) (generatedRoot.w_1086_0 ctx) (generatedRoot.w_1092_0 ctx) (generatedRoot.w_1093_0 ctx) (generatedRoot.w_1095_0 ctx) (generatedRoot.w_1097_0 ctx) (generatedRoot.w_1098_0 ctx) (generatedRoot.w_1101_0 ctx) (generatedRoot.w_1102_0 ctx) (generatedRoot.w_1103_0 ctx)) :
    generatedRoot.Facts_47 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_48  (ctx : generatedRootCtx) : Prop where
  t1536 : 1 < 2
  t1537 : 2 ≤ 3
  t1538 : 0 ≤ 0
  t1539 : 0 < 25
  t1540 : 25 ≤ 25
  h1105_0 : MxxRuntime.sliceMatrix (generatedRoot.w_1104_0 ctx) 1 2 0 25 (generatedRoot.w_1105_0 ctx)
  t1542 : 0 ≤ 2
  t1543 : 2 < 3
  t1544 : 3 ≤ 3
  t1545 : 0 ≤ 0
  t1546 : 0 < 25
  t1547 : 25 ≤ 25
  h1106_0 : MxxRuntime.sliceMatrix (generatedRoot.w_1104_0 ctx) 2 3 0 25 (generatedRoot.w_1106_0 ctx)
  h1107_0 : MxxRuntime.concatRows (generatedRoot.w_1105_0 ctx) (generatedRoot.w_1106_0 ctx) (generatedRoot.w_1107_0 ctx)
  h1108_0 : MxxRuntime.gadgetDecomposeRuns backend 4096 25 (generatedRoot.w_1107_0 ctx) (generatedRoot.w_1108_0 ctx)
  h1109_0 : MxxRuntime.compactCenteredRebaseRuns 2048 (generatedRoot.w_1108_0 ctx) (generatedRoot.w_1109_0 ctx)
  t1552 : 0 ≤ 1
  t1553 : 1 < 2
  t1554 : 2 ≤ 3
  t1555 : 0 ≤ 0
  t1556 : 0 < 25
  t1557 : 25 ≤ 25
  h1111_0 : MxxRuntime.sliceMatrix (generatedRoot.w_1110_0 ctx) 1 2 0 25 (generatedRoot.w_1111_0 ctx)
  t1559 : 0 ≤ 2
  t1560 : 2 < 3
  t1561 : 3 ≤ 3
  t1562 : 0 ≤ 0
  t1563 : 0 < 25
  t1564 : 25 ≤ 25
  h1113_0 : MxxRuntime.sliceMatrix (generatedRoot.w_1110_0 ctx) 2 3 0 25 (generatedRoot.w_1113_0 ctx)
  t1566 : 0 ≤ 0
  t1567 : 0 < 1

theorem generatedRoot.facts_48  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_48 backend (generatedRoot.w_1104_0 ctx) (generatedRoot.w_1105_0 ctx) (generatedRoot.w_1106_0 ctx) (generatedRoot.w_1107_0 ctx) (generatedRoot.w_1108_0 ctx) (generatedRoot.w_1109_0 ctx) (generatedRoot.w_1110_0 ctx) (generatedRoot.w_1111_0 ctx) (generatedRoot.w_1113_0 ctx)) :
    generatedRoot.Facts_48 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_49  (ctx : generatedRootCtx) (outputs : (generatedRoot.OutputType)) : Prop where
  t1568 : 1 ≤ 3
  t1569 : 0 ≤ 0
  t1570 : 0 < 25
  t1571 : 25 ≤ 25
  h1115_0 : MxxRuntime.sliceMatrix (generatedRoot.w_1110_0 ctx) 0 1 0 25 (generatedRoot.w_1115_0 ctx)
  t1573 : 0 ≤ 0
  t1574 : 0 < 1
  t1575 : 1 ≤ 3
  t1576 : 0 ≤ 0
  t1577 : 0 < 25
  t1578 : 25 ≤ 25
  h1116_0 : MxxRuntime.sliceMatrix (generatedRoot.w_1104_0 ctx) 0 1 0 25 (generatedRoot.w_1116_0 ctx)
  hout : outputs = generatedRoot.outs ctx

theorem generatedRoot.facts_49  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.constraints_49 (generatedRoot.w_19_0 ctx) (generatedRoot.w_23_0 ctx) (generatedRoot.w_17_0 ctx) (generatedRoot.w_18_0 ctx) (generatedRoot.w_21_0 ctx) (generatedRoot.w_22_0 ctx) (generatedRoot.w_54_0 ctx) (generatedRoot.w_54_1 ctx) (generatedRoot.w_54_2 ctx) (generatedRoot.w_54_3 ctx) (generatedRoot.w_54_4 ctx) (generatedRoot.w_54_5 ctx) (generatedRoot.w_62_0 ctx) (generatedRoot.w_87_0 ctx) (generatedRoot.w_194_0 ctx) (generatedRoot.w_214_0 ctx) (generatedRoot.w_273_0 ctx) (generatedRoot.w_331_0 ctx) (generatedRoot.w_389_0 ctx) (generatedRoot.w_447_0 ctx) (generatedRoot.w_505_0 ctx) (generatedRoot.w_563_0 ctx) (generatedRoot.w_621_0 ctx) (generatedRoot.w_655_0 ctx) (generatedRoot.w_662_0 ctx) (generatedRoot.w_665_0 ctx) (generatedRoot.w_671_0 ctx) (generatedRoot.w_676_0 ctx) (generatedRoot.w_681_0 ctx) (generatedRoot.w_686_0 ctx) (generatedRoot.w_691_0 ctx) (generatedRoot.w_696_0 ctx) (generatedRoot.w_701_0 ctx) (generatedRoot.w_709_0 ctx) (generatedRoot.w_712_0 ctx) (generatedRoot.w_718_0 ctx) (generatedRoot.w_723_0 ctx) (generatedRoot.w_728_0 ctx) (generatedRoot.w_733_0 ctx) (generatedRoot.w_738_0 ctx) (generatedRoot.w_743_0 ctx) (generatedRoot.w_748_0 ctx) (generatedRoot.w_927_0 ctx) (generatedRoot.w_927_1 ctx) (generatedRoot.w_927_2 ctx) (generatedRoot.w_927_3 ctx) (generatedRoot.w_927_4 ctx) (generatedRoot.w_927_5 ctx) (generatedRoot.w_981_0 ctx) (generatedRoot.w_981_1 ctx) (generatedRoot.w_981_2 ctx) (generatedRoot.w_981_3 ctx) (generatedRoot.w_981_4 ctx) (generatedRoot.w_981_5 ctx) (generatedRoot.w_984_0 ctx) (generatedRoot.w_985_0 ctx) (generatedRoot.w_986_0 ctx) (generatedRoot.w_987_0 ctx) (generatedRoot.w_988_0 ctx) (generatedRoot.w_1000_0 ctx) (generatedRoot.w_1002_0 ctx) (generatedRoot.w_1006_0 ctx) (generatedRoot.w_1018_0 ctx) (generatedRoot.w_1020_0 ctx) (generatedRoot.w_1024_0 ctx) (generatedRoot.w_1025_0 ctx) (generatedRoot.w_1026_0 ctx) (generatedRoot.w_1027_0 ctx) (generatedRoot.w_1028_0 ctx) (generatedRoot.w_1040_0 ctx) (generatedRoot.w_1042_0 ctx) (generatedRoot.w_1046_0 ctx) (generatedRoot.w_1058_0 ctx) (generatedRoot.w_1060_0 ctx) (generatedRoot.w_1064_0 ctx) (generatedRoot.w_1076_0 ctx) (generatedRoot.w_1078_0 ctx) (generatedRoot.w_1082_0 ctx) (generatedRoot.w_1094_0 ctx) (generatedRoot.w_1096_0 ctx) (generatedRoot.w_1100_0 ctx) (generatedRoot.w_1104_0 ctx) (generatedRoot.w_1110_0 ctx) (generatedRoot.w_1112_0 ctx) (generatedRoot.w_1114_0 ctx) (generatedRoot.w_1115_0 ctx) (generatedRoot.w_1116_0 ctx) (generatedRoot.w_1118_0 ctx) outputs) :
    generatedRoot.Facts_49 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts  (ctx : generatedRootCtx) (outputs : (generatedRoot.OutputType)) : Prop where
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
  c17 : generatedRoot.Facts_17 ctx
  c18 : generatedRoot.Facts_18 ctx
  c19 : generatedRoot.Facts_19 ctx
  c20 : generatedRoot.Facts_20 ctx
  c21 : generatedRoot.Facts_21 ctx
  c22 : generatedRoot.Facts_22 ctx
  c23 : generatedRoot.Facts_23 ctx
  c24 : generatedRoot.Facts_24 ctx
  c25 : generatedRoot.Facts_25 ctx
  c26 : generatedRoot.Facts_26 ctx
  c27 : generatedRoot.Facts_27 ctx
  c28 : generatedRoot.Facts_28 ctx
  c29 : generatedRoot.Facts_29 ctx
  c30 : generatedRoot.Facts_30 ctx
  c31 : generatedRoot.Facts_31 ctx
  c32 : generatedRoot.Facts_32 ctx
  c33 : generatedRoot.Facts_33 ctx
  c34 : generatedRoot.Facts_34 ctx
  c35 : generatedRoot.Facts_35 ctx
  c36 : generatedRoot.Facts_36 ctx
  c37 : generatedRoot.Facts_37 ctx
  c38 : generatedRoot.Facts_38 ctx
  c39 : generatedRoot.Facts_39 ctx
  c40 : generatedRoot.Facts_40 ctx
  c41 : generatedRoot.Facts_41 ctx
  c42 : generatedRoot.Facts_42 ctx
  c43 : generatedRoot.Facts_43 ctx
  c44 : generatedRoot.Facts_44 ctx
  c45 : generatedRoot.Facts_45 ctx
  c46 : generatedRoot.Facts_46 ctx
  c47 : generatedRoot.Facts_47 ctx
  c48 : generatedRoot.Facts_48 ctx
  c49 : generatedRoot.Facts_49 ctx outputs

theorem generatedRoot.facts  (ctx : generatedRootCtx) {outputs : (generatedRoot.OutputType)}
    (h : generatedRoot.body backend hashModel tape path params ctx.inputs outputs ctx.witness) :
    generatedRoot.Facts ctx outputs :=
  ⟨(generatedRoot.facts_0 (ctx := ctx) (outputs := outputs) h.1.1.1.1.1), (generatedRoot.facts_1 (ctx := ctx) (outputs := outputs) h.1.1.1.1.2.1), (generatedRoot.facts_2 (ctx := ctx) (outputs := outputs) h.1.1.1.1.2.2), (generatedRoot.facts_3 (ctx := ctx) (outputs := outputs) h.1.1.1.2.1), (generatedRoot.facts_4 (ctx := ctx) (outputs := outputs) h.1.1.1.2.2.1), (generatedRoot.facts_5 (ctx := ctx) (outputs := outputs) h.1.1.1.2.2.2), (generatedRoot.facts_6 (ctx := ctx) (outputs := outputs) h.1.1.2.1.1), (generatedRoot.facts_7 (ctx := ctx) (outputs := outputs) h.1.1.2.1.2.1), (generatedRoot.facts_8 (ctx := ctx) (outputs := outputs) h.1.1.2.1.2.2), (generatedRoot.facts_9 (ctx := ctx) (outputs := outputs) h.1.1.2.2.1), (generatedRoot.facts_10 (ctx := ctx) (outputs := outputs) h.1.1.2.2.2.1), (generatedRoot.facts_11 (ctx := ctx) (outputs := outputs) h.1.1.2.2.2.2), (generatedRoot.facts_12 (ctx := ctx) (outputs := outputs) h.1.2.1.1.1), (generatedRoot.facts_13 (ctx := ctx) (outputs := outputs) h.1.2.1.1.2.1), (generatedRoot.facts_14 (ctx := ctx) (outputs := outputs) h.1.2.1.1.2.2), (generatedRoot.facts_15 (ctx := ctx) (outputs := outputs) h.1.2.1.2.1), (generatedRoot.facts_16 (ctx := ctx) (outputs := outputs) h.1.2.1.2.2.1), (generatedRoot.facts_17 (ctx := ctx) (outputs := outputs) h.1.2.1.2.2.2), (generatedRoot.facts_18 (ctx := ctx) (outputs := outputs) h.1.2.2.1.1), (generatedRoot.facts_19 (ctx := ctx) (outputs := outputs) h.1.2.2.1.2.1), (generatedRoot.facts_20 (ctx := ctx) (outputs := outputs) h.1.2.2.1.2.2), (generatedRoot.facts_21 (ctx := ctx) (outputs := outputs) h.1.2.2.2.1.1), (generatedRoot.facts_22 (ctx := ctx) (outputs := outputs) h.1.2.2.2.1.2), (generatedRoot.facts_23 (ctx := ctx) (outputs := outputs) h.1.2.2.2.2.1), (generatedRoot.facts_24 (ctx := ctx) (outputs := outputs) h.1.2.2.2.2.2), (generatedRoot.facts_25 (ctx := ctx) (outputs := outputs) h.2.1.1.1.1), (generatedRoot.facts_26 (ctx := ctx) (outputs := outputs) h.2.1.1.1.2.1), (generatedRoot.facts_27 (ctx := ctx) (outputs := outputs) h.2.1.1.1.2.2), (generatedRoot.facts_28 (ctx := ctx) (outputs := outputs) h.2.1.1.2.1), (generatedRoot.facts_29 (ctx := ctx) (outputs := outputs) h.2.1.1.2.2.1), (generatedRoot.facts_30 (ctx := ctx) (outputs := outputs) h.2.1.1.2.2.2), (generatedRoot.facts_31 (ctx := ctx) (outputs := outputs) h.2.1.2.1.1), (generatedRoot.facts_32 (ctx := ctx) (outputs := outputs) h.2.1.2.1.2.1), (generatedRoot.facts_33 (ctx := ctx) (outputs := outputs) h.2.1.2.1.2.2), (generatedRoot.facts_34 (ctx := ctx) (outputs := outputs) h.2.1.2.2.1), (generatedRoot.facts_35 (ctx := ctx) (outputs := outputs) h.2.1.2.2.2.1), (generatedRoot.facts_36 (ctx := ctx) (outputs := outputs) h.2.1.2.2.2.2), (generatedRoot.facts_37 (ctx := ctx) (outputs := outputs) h.2.2.1.1.1), (generatedRoot.facts_38 (ctx := ctx) (outputs := outputs) h.2.2.1.1.2.1), (generatedRoot.facts_39 (ctx := ctx) (outputs := outputs) h.2.2.1.1.2.2), (generatedRoot.facts_40 (ctx := ctx) (outputs := outputs) h.2.2.1.2.1), (generatedRoot.facts_41 (ctx := ctx) (outputs := outputs) h.2.2.1.2.2.1), (generatedRoot.facts_42 (ctx := ctx) (outputs := outputs) h.2.2.1.2.2.2), (generatedRoot.facts_43 (ctx := ctx) (outputs := outputs) h.2.2.2.1.1), (generatedRoot.facts_44 (ctx := ctx) (outputs := outputs) h.2.2.2.1.2.1), (generatedRoot.facts_45 (ctx := ctx) (outputs := outputs) h.2.2.2.1.2.2), (generatedRoot.facts_46 (ctx := ctx) (outputs := outputs) h.2.2.2.2.1.1), (generatedRoot.facts_47 (ctx := ctx) (outputs := outputs) h.2.2.2.2.1.2), (generatedRoot.facts_48 (ctx := ctx) (outputs := outputs) h.2.2.2.2.2.1), (generatedRoot.facts_49 (ctx := ctx) (outputs := outputs) h.2.2.2.2.2.2)⟩

theorem generatedRoot.h28_5_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 8) :
    ∃ w : parallel_generatedRoot_28.Witness, parallel_generatedRoot_28.Facts (⟨((generatedRoot.w_27_0 ctx), ((generatedRoot.w_27_1 ctx), ((generatedRoot.w_27_2 ctx), ((generatedRoot.w_27_3 ctx), ((generatedRoot.w_27_4 ctx), ((generatedRoot.w_27_5 ctx), ())))))), w⟩ : parallel_generatedRoot_28.Ctx backend hashModel tape (path ++ [28, i.val]) params i) (((generatedRoot.w_28_0 ctx) i), (((generatedRoot.w_28_1 ctx) i), (((generatedRoot.w_28_2 ctx) i), (((generatedRoot.w_28_3 ctx) i), (((generatedRoot.w_28_4 ctx) i), (((generatedRoot.w_28_5 ctx) i), ())))))) :=
  let ⟨w, h⟩ := F.c0.h28_5 i
  ⟨w, parallel_generatedRoot_28.facts _ h⟩

theorem generatedRoot.h32_5_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 8) :
    ∃ w : parallel_generatedRoot_32.Witness, parallel_generatedRoot_32.Facts (⟨((generatedRoot.w_31_0 ctx), ((generatedRoot.w_31_1 ctx), ((generatedRoot.w_31_2 ctx), ((generatedRoot.w_31_3 ctx), ((generatedRoot.w_31_4 ctx), ((generatedRoot.w_31_5 ctx), ())))))), w⟩ : parallel_generatedRoot_32.Ctx backend hashModel tape (path ++ [32, i.val]) params i) (((generatedRoot.w_32_0 ctx) i), (((generatedRoot.w_32_1 ctx) i), (((generatedRoot.w_32_2 ctx) i), (((generatedRoot.w_32_3 ctx) i), (((generatedRoot.w_32_4 ctx) i), (((generatedRoot.w_32_5 ctx) i), ())))))) :=
  let ⟨w, h⟩ := F.c0.h32_5 i
  ⟨w, parallel_generatedRoot_32.facts _ h⟩

theorem generatedRoot.h33_5_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 8) :
    ∃ w : parallel_generatedRoot_33.Witness, parallel_generatedRoot_33.Facts (⟨((generatedRoot.w_7_0 ctx), ((generatedRoot.w_8_0 ctx), ((generatedRoot.w_9_0 ctx), ((generatedRoot.w_10_0 ctx), ((generatedRoot.w_11_0 ctx), ((generatedRoot.w_5_0 ctx), ((generatedRoot.w_12_0 ctx), (((generatedRoot.w_32_4 ctx) i), (((generatedRoot.w_32_0 ctx) i), (((generatedRoot.w_32_2 ctx) i), (((generatedRoot.w_32_5 ctx) i), (((generatedRoot.w_32_1 ctx) i), (((generatedRoot.w_32_3 ctx) i), ()))))))))))))), w⟩ : parallel_generatedRoot_33.Ctx backend hashModel tape (path ++ [33, i.val]) params i) (((generatedRoot.w_33_0 ctx) i), (((generatedRoot.w_33_1 ctx) i), (((generatedRoot.w_33_2 ctx) i), (((generatedRoot.w_33_3 ctx) i), (((generatedRoot.w_33_4 ctx) i), (((generatedRoot.w_33_5 ctx) i), ())))))) :=
  let ⟨w, h⟩ := F.c0.h33_5 i
  ⟨w, parallel_generatedRoot_33.facts _ h⟩

theorem generatedRoot.h34_5_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 8) :
    ∃ w : parallel_generatedRoot_34.Witness, parallel_generatedRoot_34.Facts (⟨((generatedRoot.w_0_0 ctx), ((generatedRoot.w_1_0 ctx), ((generatedRoot.w_2_0 ctx), ((generatedRoot.w_3_0 ctx), ((generatedRoot.w_4_0 ctx), ((generatedRoot.w_5_0 ctx), ((generatedRoot.w_6_0 ctx), (((generatedRoot.w_33_4 ctx) i), (((generatedRoot.w_33_0 ctx) i), (((generatedRoot.w_33_2 ctx) i), (((generatedRoot.w_33_5 ctx) i), (((generatedRoot.w_33_1 ctx) i), (((generatedRoot.w_33_3 ctx) i), ()))))))))))))), w⟩ : parallel_generatedRoot_34.Ctx backend hashModel tape (path ++ [34, i.val]) params i) (((generatedRoot.w_34_0 ctx) i), (((generatedRoot.w_34_1 ctx) i), (((generatedRoot.w_34_2 ctx) i), (((generatedRoot.w_34_3 ctx) i), (((generatedRoot.w_34_4 ctx) i), (((generatedRoot.w_34_5 ctx) i), ())))))) :=
  let ⟨w, h⟩ := F.c0.h34_5 i
  ⟨w, parallel_generatedRoot_34.facts _ h⟩

theorem generatedRoot.h658_5_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 5) :
    ∃ w : parallel_generatedRoot_658.Witness, parallel_generatedRoot_658.Facts (⟨((generatedRoot.w_637_1 ctx), ((generatedRoot.w_639_0 ctx), ((generatedRoot.w_640_0 ctx), ((generatedRoot.w_637_0 ctx), ((generatedRoot.w_642_0 ctx), ((generatedRoot.w_643_0 ctx), ((generatedRoot.w_644_0 ctx), ((generatedRoot.w_5_0 ctx), ((generatedRoot.w_645_0 ctx), ((generatedRoot.w_646_0 ctx), ((generatedRoot.w_647_0 ctx), ((generatedRoot.w_648_0 ctx), ((generatedRoot.w_649_0 ctx), ((generatedRoot.w_650_0 ctx), ((generatedRoot.w_651_0 ctx), ((generatedRoot.w_653_0 ctx), ((generatedRoot.w_86_0 ctx), ((generatedRoot.w_654_0 ctx), ((generatedRoot.w_657_0 ctx), ((generatedRoot.w_89_0 ctx), ())))))))))))))))))))), w⟩ : parallel_generatedRoot_658.Ctx backend hashModel tape (path ++ [658, i.val]) params i) (((generatedRoot.w_658_0 ctx) i), (((generatedRoot.w_658_1 ctx) i), (((generatedRoot.w_658_2 ctx) i), (((generatedRoot.w_658_3 ctx) i), (((generatedRoot.w_658_4 ctx) i), (((generatedRoot.w_658_5 ctx) i), ())))))) :=
  let ⟨w, h⟩ := F.c27.h658_5 i
  ⟨w, parallel_generatedRoot_658.facts _ h⟩

theorem generatedRoot.h890_5_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 7200) :
    ∃ w : parallel_generatedRoot_890.Witness, parallel_generatedRoot_890.Facts (⟨((generatedRoot.w_889_0 ctx), ((generatedRoot.w_889_1 ctx), ((generatedRoot.w_889_2 ctx), ((generatedRoot.w_889_3 ctx), ((generatedRoot.w_889_4 ctx), ((generatedRoot.w_889_5 ctx), ())))))), w⟩ : parallel_generatedRoot_890.Ctx backend hashModel tape (path ++ [890, i.val]) params i) (((generatedRoot.w_890_0 ctx) i), (((generatedRoot.w_890_1 ctx) i), (((generatedRoot.w_890_2 ctx) i), (((generatedRoot.w_890_3 ctx) i), (((generatedRoot.w_890_4 ctx) i), (((generatedRoot.w_890_5 ctx) i), ())))))) :=
  let ⟨w, h⟩ := F.c35.h890_5 i
  ⟨w, parallel_generatedRoot_890.facts _ h⟩

theorem generatedRoot.h894_5_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 7200) :
    ∃ w : parallel_generatedRoot_894.Witness, parallel_generatedRoot_894.Facts (⟨((generatedRoot.w_893_0 ctx), ((generatedRoot.w_893_1 ctx), ((generatedRoot.w_893_2 ctx), ((generatedRoot.w_893_3 ctx), ((generatedRoot.w_893_4 ctx), ((generatedRoot.w_893_5 ctx), ())))))), w⟩ : parallel_generatedRoot_894.Ctx backend hashModel tape (path ++ [894, i.val]) params i) (((generatedRoot.w_894_0 ctx) i), (((generatedRoot.w_894_1 ctx) i), (((generatedRoot.w_894_2 ctx) i), (((generatedRoot.w_894_3 ctx) i), (((generatedRoot.w_894_4 ctx) i), (((generatedRoot.w_894_5 ctx) i), ())))))) :=
  let ⟨w, h⟩ := F.c35.h894_5 i
  ⟨w, parallel_generatedRoot_894.facts _ h⟩

theorem generatedRoot.h895_5_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 7200) :
    ∃ w : parallel_generatedRoot_895.Witness, parallel_generatedRoot_895.Facts (⟨((generatedRoot.w_872_0 ctx), ((generatedRoot.w_873_0 ctx), ((generatedRoot.w_874_0 ctx), ((generatedRoot.w_875_0 ctx), ((generatedRoot.w_11_0 ctx), ((generatedRoot.w_5_0 ctx), ((generatedRoot.w_12_0 ctx), (((generatedRoot.w_894_4 ctx) i), (((generatedRoot.w_894_0 ctx) i), (((generatedRoot.w_894_2 ctx) i), (((generatedRoot.w_894_5 ctx) i), (((generatedRoot.w_894_1 ctx) i), (((generatedRoot.w_894_3 ctx) i), ()))))))))))))), w⟩ : parallel_generatedRoot_895.Ctx backend hashModel tape (path ++ [895, i.val]) params i) (((generatedRoot.w_895_0 ctx) i), (((generatedRoot.w_895_1 ctx) i), (((generatedRoot.w_895_2 ctx) i), (((generatedRoot.w_895_3 ctx) i), (((generatedRoot.w_895_4 ctx) i), (((generatedRoot.w_895_5 ctx) i), ())))))) :=
  let ⟨w, h⟩ := F.c35.h895_5 i
  ⟨w, parallel_generatedRoot_895.facts _ h⟩

theorem generatedRoot.h944_5_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 36000) :
    ∃ w : parallel_generatedRoot_944.Witness, parallel_generatedRoot_944.Facts (⟨((generatedRoot.w_943_0 ctx), ((generatedRoot.w_943_1 ctx), ((generatedRoot.w_943_2 ctx), ((generatedRoot.w_943_3 ctx), ((generatedRoot.w_943_4 ctx), ((generatedRoot.w_943_5 ctx), ())))))), w⟩ : parallel_generatedRoot_944.Ctx backend hashModel tape (path ++ [944, i.val]) params i) (((generatedRoot.w_944_0 ctx) i), (((generatedRoot.w_944_1 ctx) i), (((generatedRoot.w_944_2 ctx) i), (((generatedRoot.w_944_3 ctx) i), (((generatedRoot.w_944_4 ctx) i), (((generatedRoot.w_944_5 ctx) i), ())))))) :=
  let ⟨w, h⟩ := F.c37.h944_5 i
  ⟨w, parallel_generatedRoot_944.facts _ h⟩

theorem generatedRoot.h948_5_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 36000) :
    ∃ w : parallel_generatedRoot_948.Witness, parallel_generatedRoot_948.Facts (⟨((generatedRoot.w_947_0 ctx), ((generatedRoot.w_947_1 ctx), ((generatedRoot.w_947_2 ctx), ((generatedRoot.w_947_3 ctx), ((generatedRoot.w_947_4 ctx), ((generatedRoot.w_947_5 ctx), ())))))), w⟩ : parallel_generatedRoot_948.Ctx backend hashModel tape (path ++ [948, i.val]) params i) (((generatedRoot.w_948_0 ctx) i), (((generatedRoot.w_948_1 ctx) i), (((generatedRoot.w_948_2 ctx) i), (((generatedRoot.w_948_3 ctx) i), (((generatedRoot.w_948_4 ctx) i), (((generatedRoot.w_948_5 ctx) i), ())))))) :=
  let ⟨w, h⟩ := F.c37.h948_5 i
  ⟨w, parallel_generatedRoot_948.facts _ h⟩

theorem generatedRoot.h949_5_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 36000) :
    ∃ w : parallel_generatedRoot_949.Witness, parallel_generatedRoot_949.Facts (⟨((generatedRoot.w_929_0 ctx), ((generatedRoot.w_930_0 ctx), ((generatedRoot.w_931_0 ctx), ((generatedRoot.w_932_0 ctx), ((generatedRoot.w_11_0 ctx), ((generatedRoot.w_5_0 ctx), ((generatedRoot.w_12_0 ctx), (((generatedRoot.w_948_4 ctx) i), (((generatedRoot.w_948_0 ctx) i), (((generatedRoot.w_948_2 ctx) i), (((generatedRoot.w_948_5 ctx) i), (((generatedRoot.w_948_1 ctx) i), (((generatedRoot.w_948_3 ctx) i), ()))))))))))))), w⟩ : parallel_generatedRoot_949.Ctx backend hashModel tape (path ++ [949, i.val]) params i) (((generatedRoot.w_949_0 ctx) i), (((generatedRoot.w_949_1 ctx) i), (((generatedRoot.w_949_2 ctx) i), (((generatedRoot.w_949_3 ctx) i), (((generatedRoot.w_949_4 ctx) i), (((generatedRoot.w_949_5 ctx) i), ())))))) :=
  let ⟨w, h⟩ := F.c37.h949_5 i
  ⟨w, parallel_generatedRoot_949.facts _ h⟩

theorem generatedRoot.h984_0_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 36) :
    ∃ w : parallel_generatedRoot_984.Witness, parallel_generatedRoot_984.Facts (⟨((generatedRoot.w_61_0 ctx), ((generatedRoot.w_68_0 ctx), (((generatedRoot.w_870_3 ctx) i), (((generatedRoot.w_870_1 ctx) i), ((generatedRoot.w_927_1 ctx), ((generatedRoot.w_55_0 ctx), ((generatedRoot.w_981_5 ctx), ((generatedRoot.w_983_0 ctx), ((generatedRoot.w_981_1 ctx), ((generatedRoot.w_981_3 ctx), ((generatedRoot.w_652_0 ctx), ((generatedRoot.w_656_0 ctx), (((generatedRoot.w_870_2 ctx) i), ((generatedRoot.w_927_3 ctx), ())))))))))))))), w⟩ : parallel_generatedRoot_984.Ctx backend hashModel tape (path ++ [984, i.val]) params i) ((generatedRoot.w_984_0 ctx) i) :=
  let ⟨w, h⟩ := F.c38.h984_0 i
  ⟨w, parallel_generatedRoot_984.facts _ h⟩

theorem generatedRoot.h985_0_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 36) :
    ∃ w : parallel_generatedRoot_985.Witness, parallel_generatedRoot_985.Facts (⟨((generatedRoot.w_61_0 ctx), ((generatedRoot.w_68_0 ctx), (((generatedRoot.w_870_3 ctx) i), (((generatedRoot.w_870_1 ctx) i), ((generatedRoot.w_927_1 ctx), ((generatedRoot.w_55_0 ctx), ((generatedRoot.w_981_5 ctx), ((generatedRoot.w_983_0 ctx), ((generatedRoot.w_981_1 ctx), ((generatedRoot.w_981_3 ctx), ((generatedRoot.w_652_0 ctx), ((generatedRoot.w_656_0 ctx), (((generatedRoot.w_870_2 ctx) i), ((generatedRoot.w_927_3 ctx), ())))))))))))))), w⟩ : parallel_generatedRoot_985.Ctx backend hashModel tape (path ++ [985, i.val]) params i) ((generatedRoot.w_985_0 ctx) i) :=
  let ⟨w, h⟩ := F.c38.h985_0 i
  ⟨w, parallel_generatedRoot_985.facts _ h⟩

theorem generatedRoot.h986_0_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 36) :
    ∃ w : parallel_generatedRoot_986.Witness, parallel_generatedRoot_986.Facts (⟨((generatedRoot.w_61_0 ctx), ((generatedRoot.w_68_0 ctx), (((generatedRoot.w_870_3 ctx) i), (((generatedRoot.w_870_1 ctx) i), ((generatedRoot.w_927_1 ctx), ((generatedRoot.w_55_0 ctx), ((generatedRoot.w_981_5 ctx), ((generatedRoot.w_983_0 ctx), ((generatedRoot.w_981_1 ctx), ((generatedRoot.w_981_3 ctx), ((generatedRoot.w_652_0 ctx), ((generatedRoot.w_656_0 ctx), (((generatedRoot.w_870_2 ctx) i), ((generatedRoot.w_927_3 ctx), ())))))))))))))), w⟩ : parallel_generatedRoot_986.Ctx backend hashModel tape (path ++ [986, i.val]) params i) ((generatedRoot.w_986_0 ctx) i) :=
  let ⟨w, h⟩ := F.c38.h986_0 i
  ⟨w, parallel_generatedRoot_986.facts _ h⟩

theorem generatedRoot.h987_0_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 36) :
    ∃ w : parallel_generatedRoot_987.Witness, parallel_generatedRoot_987.Facts (⟨((generatedRoot.w_61_0 ctx), ((generatedRoot.w_68_0 ctx), (((generatedRoot.w_870_3 ctx) i), (((generatedRoot.w_870_1 ctx) i), ((generatedRoot.w_927_1 ctx), ((generatedRoot.w_55_0 ctx), ((generatedRoot.w_981_5 ctx), ((generatedRoot.w_983_0 ctx), ((generatedRoot.w_981_1 ctx), ((generatedRoot.w_981_3 ctx), ((generatedRoot.w_652_0 ctx), ((generatedRoot.w_656_0 ctx), (((generatedRoot.w_870_2 ctx) i), ((generatedRoot.w_927_3 ctx), ())))))))))))))), w⟩ : parallel_generatedRoot_987.Ctx backend hashModel tape (path ++ [987, i.val]) params i) ((generatedRoot.w_987_0 ctx) i) :=
  let ⟨w, h⟩ := F.c38.h987_0 i
  ⟨w, parallel_generatedRoot_987.facts _ h⟩

theorem generatedRoot.h988_0_at  {ctx : generatedRootCtx} {outputs : (generatedRoot.OutputType)}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 36) :
    ∃ w : parallel_generatedRoot_988.Witness, parallel_generatedRoot_988.Facts (⟨((generatedRoot.w_61_0 ctx), ((generatedRoot.w_68_0 ctx), (((generatedRoot.w_870_3 ctx) i), (((generatedRoot.w_870_1 ctx) i), ((generatedRoot.w_927_1 ctx), ((generatedRoot.w_55_0 ctx), ((generatedRoot.w_981_5 ctx), ((generatedRoot.w_983_0 ctx), ((generatedRoot.w_981_1 ctx), ((generatedRoot.w_981_3 ctx), ((generatedRoot.w_652_0 ctx), ((generatedRoot.w_656_0 ctx), (((generatedRoot.w_870_2 ctx) i), ((generatedRoot.w_927_3 ctx), ())))))))))))))), w⟩ : parallel_generatedRoot_988.Ctx backend hashModel tape (path ++ [988, i.val]) params i) ((generatedRoot.w_988_0 ctx) i) :=
  let ⟨w, h⟩ := F.c38.h988_0 i
  ⟨w, parallel_generatedRoot_988.facts _ h⟩

end

end Stage_keys_online_0_0

end
